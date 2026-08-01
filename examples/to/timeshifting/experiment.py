"""Run the complete CBO/CEM time-shifting mode experiment."""

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from examples.to.benchmarking.render_media import _render_video
from examples.to.open_loop_benchmark import (
    compile_controller,
    make_controller,
    make_task,
    run_episode,
)

FIELDS = (
    "algorithm mode shift_name shift_steps shift_seconds seed episode_cost "
    "planning_time_seconds"
).split()


def validate(cfg: DictConfig) -> None:
    """Validate the fixed, GPU-only paired experimental design."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("The complete experiment must run on a GPU")
    if list(cfg.algorithms) != ["cbo", "cem"]:
        raise ValueError("The experiment requires CBO and CEM")
    if list(cfg.modes) != ["legacy", "reset", "shift"]:
        raise ValueError("modes must be legacy, reset, shift")
    if len(cfg.seeds) != 5 or len(set(int(seed) for seed in cfg.seeds)) != 5:
        raise ValueError("The experiment requires exactly five distinct seeds")
    steps = [int(value) for value in cfg.shift_steps.values()]
    if len(set(steps)) != len(steps) or any(step < 1 for step in steps):
        raise ValueError("shift_steps must contain distinct positive integers")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write per-seed episode metrics."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(rows: list[dict[str, Any]], cfg: DictConfig) -> list[dict[str, Any]]:
    """Summarize every paired mode and step-count condition."""
    results = []
    for algorithm in cfg.algorithms:
        for mode in cfg.modes:
            for shift_name, shift_steps_value in cfg.shift_steps.items():
                shift_steps = int(shift_steps_value)
                subset = [
                    row
                    for row in rows
                    if row["algorithm"] == algorithm
                    and row["mode"] == mode
                    and row["shift_name"] == shift_name
                ]
                if len(subset) != len(cfg.seeds):
                    raise ValueError(f"Incomplete condition: {algorithm} {mode} {shift_name}")
                costs = np.asarray([float(row["episode_cost"]) for row in subset])
                times = np.asarray([float(row["planning_time_seconds"]) for row in subset])
                media = f"media/{algorithm}_{mode}_{shift_name}.mp4"
                results.append({
                    "algorithm": algorithm,
                    "mode": mode,
                    "shift_name": shift_name,
                    "shift_steps": shift_steps,
                    "shift_seconds": float(subset[0]["shift_seconds"]),
                    "mean_cost": float(costs.mean()),
                    "std_cost": float(costs.std(ddof=1)),
                    "mean_planning_time_seconds": float(times.mean()),
                    "std_planning_time_seconds": float(times.std(ddof=1)),
                    "seed_count": len(subset),
                    "video": media,
                    "poster": media.replace(".mp4", ".jpg"),
                })
    return results


def render_dashboard(path: Path, summary: dict[str, Any]) -> None:
    """Create a self-contained local comparison dashboard."""
    template = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(summary, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    path.write_text(template.replace("__TIME_SHIFT_DATA__", payload), encoding="utf-8")


def run(cfg: DictConfig) -> None:
    """Run every full episode, then render fixed-seed videos and HTML."""
    validate(cfg)
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    task, state = make_task(str(cfg.task), str(cfg.backend))
    selections = OmegaConf.load(Path(str(cfg.selected_parameters)).resolve())
    rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, str, str], Any] = {}

    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        candidate = selections[cfg.task][algorithm].parameters
        for mode_value in cfg.modes:
            mode = str(mode_value)
            controller = make_controller(
                algorithm,
                task,
                cfg.task_config,
                candidate,
                int(cfg.domain_seed),
                int(cfg.samples),
                int(cfg.iterations),
                mode,
            )
            for shift_name, shift_steps_value in cfg.shift_steps.items():
                shift_steps = int(shift_steps_value)
                if shift_steps > controller.ctrl_steps:
                    raise ValueError(
                        f"{shift_name}={shift_steps} exceeds {controller.ctrl_steps} control steps"
                    )
                compiled = compile_controller(
                    controller, state, int(cfg.warmup_seed), shift_steps
                )
                for seed_value in cfg.seeds:
                    seed = int(seed_value)
                    result = run_episode(
                        controller,
                        compiled,
                        state,
                        seed,
                        float(cfg.task_config.total_horizon),
                        shift_steps,
                        seed == int(cfg.video_seed),
                    )
                    rows.append({
                        "algorithm": algorithm,
                        "mode": mode,
                        "shift_name": str(shift_name),
                        "shift_steps": shift_steps,
                        "shift_seconds": shift_steps * controller.dt,
                        "seed": seed,
                        "episode_cost": result.episode_cost,
                        "planning_time_seconds": result.planning_time_seconds,
                    })
                    print(
                        f"{algorithm} mode={mode} shift={shift_steps} steps "
                        f"({shift_steps * controller.dt:.3f}s) seed={seed} "
                        f"cost={result.episode_cost:.6g} "
                        f"time={result.planning_time_seconds:.3f}s",
                        flush=True,
                    )
                    if seed == int(cfg.video_seed):
                        trajectories[(algorithm, mode, str(shift_name))] = result.trajectory

    write_rows(output_dir / "episode_results.csv", rows)
    for (algorithm, mode, shift_name), trajectory in trajectories.items():
        _render_video(
            output_dir / f"media/{algorithm}_{mode}_{shift_name}.mp4",
            str(cfg.task),
            task,
            state,
            trajectory,
            float(cfg.task_config.total_horizon),
            cfg.render,
        )
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "aggregates": aggregate(rows, cfg),
        "episodes": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_dashboard(output_dir / "index.html", summary)


@hydra.main(version_base=None, config_path=".", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the complete experiment."""
    run(cfg)


if __name__ == "__main__":
    main()
