"""Render representative trajectories for baseline and scaling winners."""

import gc
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

from examples.to.open_loop_benchmark import (
    STATE_FIELDS,
    compile_controller,
    execution_steps,
    make_controller,
    make_task,
    run_episode,
)

CAMERAS = {
    "double_cart_pole": "fixed",
}


def _render_video(
    path: Path,
    task_name: str,
    task: Any,
    initial_state: Any,
    states: dict[str, Any],
    total_horizon: float,
    render_cfg: DictConfig,
) -> None:
    values = {
        name: jax.device_get(
            jnp.concatenate((getattr(initial_state, name)[None], states[name]))
        )
        for name in STATE_FIELDS
    }
    frame_count = round(total_horizon * float(render_cfg.fps))
    if frame_count < 2:
        raise ValueError("Rendering requires at least two frames")
    indices = np.rint(
        np.linspace(0, len(values["time"]) - 1, frame_count)
    ).astype(int)
    width, height = int(render_cfg.width), int(render_cfg.height)
    task.mj_model.vis.global_.offwidth = width
    task.mj_model.vis.global_.offheight = height
    data, camera = mujoco.MjData(task.mj_model), mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = task.mj_model.camera(CAMERAS[task_name]).id
    renderer = mujoco.Renderer(task.mj_model, height=height, width=width)
    path.parent.mkdir(parents=True, exist_ok=True)
    video_tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    poster = path.with_suffix(".jpg")
    poster_tmp = poster.with_name(f"{poster.stem}.tmp{poster.suffix}")
    encode = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(render_cfg.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "22",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        str(video_tmp),
    ]
    process = subprocess.Popen(encode, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg input pipe was not created")
    last_frame: bytes | None = None
    try:
        for index in indices:
            data.qpos[:] = values["qpos"][index]
            data.qvel[:] = values["qvel"][index]
            data.mocap_pos[:] = values["mocap_pos"][index]
            data.mocap_quat[:] = values["mocap_quat"][index]
            data.time = values["time"][index]
            mujoco.mj_forward(task.mj_model, data)
            renderer.update_scene(data, camera=camera)
            last_frame = renderer.render().tobytes()
            process.stdin.write(last_frame)
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, encode)
    if last_frame is None:
        raise RuntimeError("No frame was rendered")
    poster_command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-loglevel",
        "error",
        str(poster_tmp),
    ]
    subprocess.run(poster_command, input=last_frame, check=True)
    video_tmp.replace(path)
    poster_tmp.replace(poster)


def _jobs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for row in summary["scaling"]:
        jobs.append(
            row
            | {
                "kind": "scaling",
                "path": row["video"],
            }
        )
    for row in summary["baseline"]:
        jobs.append(
            row
            | {
                "kind": "baseline",
                "path": row["video"],
            }
        )
    return jobs


def _job_key(job: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(job["algorithm"]),
        int(job["iterations"]),
        int(job["samples"]),
        int(job["video_seed"]),
    )


def _run(cfg: DictConfig) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("Media trajectories must be generated on a GPU")
    output_dir = Path(str(cfg.output_dir)).resolve()
    summary = json.loads((output_dir / "summary.json").read_text())
    selections = OmegaConf.load(Path(str(cfg.selected_parameters)).resolve())
    task, state = make_task(str(cfg.task), str(cfg.backend))
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for job in _jobs(summary):
        grouped[_job_key(job)].append(job)
    manifest = []

    for index, (key, jobs) in enumerate(grouped.items(), start=1):
        algorithm, iterations, samples, seed = key
        first_path = output_dir / str(jobs[0]["path"])
        existing = all(
            (output_dir / str(job["path"])).exists()
            and (output_dir / str(job["path"])).with_suffix(".jpg").exists()
            for job in jobs
        )
        if bool(cfg.resume) and existing:
            print(
                f"[{index:02d}/{len(grouped)}] reuse {first_path.name}",
                flush=True,
            )
            continue
        selection = selections[cfg.task][algorithm]
        controller = make_controller(
            algorithm,
            task,
            cfg.task_config,
            selection.parameters,
            int(cfg.domain_seed),
            samples,
            iterations,
        )
        executed_steps = execution_steps(
            controller, float(cfg.execution_fraction)
        )
        compiled = compile_controller(
            controller, state, int(cfg.warmup_seed), executed_steps
        )
        result = run_episode(
            controller,
            compiled,
            state,
            seed,
            float(cfg.task_config.total_horizon),
            executed_steps,
            True,
        )
        expected_cost = float(jobs[0]["video_seed_cost"])
        if not math.isclose(
            result.episode_cost, expected_cost, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError(
                "Replay cost mismatch: "
                f"{result.episode_cost} != {expected_cost}"
            )
        _render_video(
            first_path,
            str(cfg.task),
            task,
            state,
            result.trajectory,
            float(cfg.task_config.total_horizon),
            cfg.render,
        )
        for job in jobs[1:]:
            destination = output_dir / str(job["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first_path, destination)
            shutil.copy2(
                first_path.with_suffix(".jpg"), destination.with_suffix(".jpg")
            )
        for job in jobs:
            manifest.append(
                {
                    "kind": job["kind"],
                    "algorithm": algorithm,
                    "nominal_operations": int(job["nominal_operations"]),
                    "iterations": iterations,
                    "samples": samples,
                    "seed": seed,
                    "episode_cost": result.episode_cost,
                    "video": str(job["path"]),
                    "poster": str(Path(str(job["path"])).with_suffix(".jpg")),
                }
            )
        print(
            f"[{index:02d}/{len(grouped)}] {algorithm:9s} "
            f"I={iterations:2d} N={samples:4d} seed={seed} "
            f"cost={result.episode_cost:.6g}",
            flush=True,
        )
        del compiled, controller, result
        gc.collect()
        jax.clear_caches()

    (output_dir / "media_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


@hydra.main(version_base=None, config_path=".", config_name="grid")
def main(cfg: DictConfig) -> None:
    """Render the selected baseline and scaling trajectories."""
    _run(cfg)


if __name__ == "__main__":
    main()
