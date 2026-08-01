"""Run the complete CBO/CEM time-shifting experiment and publish evidence."""

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import jax
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

from examples.to.benchmarking.render_media import _render_video
from examples.to.open_loop_benchmark import (
    compile_controller,
    execution_steps,
    make_controller,
    make_task,
    run_episode,
)

FIELDS = "algorithm solution shift_length seed episode_cost planning_time_seconds execution_fraction execution_horizon".split()


def validate(cfg: DictConfig) -> None:
    """Validate the fixed, GPU-only paired experimental design."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("The complete time-shifting experiment must run on a GPU")
    if list(cfg.algorithms) != ["cbo", "cem"] or list(cfg.solutions) != [False, True]:
        raise ValueError("The experiment requires CBO/CEM and legacy/corrected modes")
    if len(cfg.seeds) != 5 or len(set(int(seed) for seed in cfg.seeds)) != 5:
        raise ValueError("The experiment requires exactly five distinct seeds")
    if float(cfg.shift_lengths.short.execution_fraction) >= float(cfg.shift_lengths.long.execution_fraction):
        raise ValueError("short shifting must be smaller than long shifting")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write per-seed episode metrics."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(rows: list[dict[str, Any]], cfg: DictConfig) -> list[dict[str, Any]]:
    """Summarize every paired correction/shift condition."""
    results = []
    for algorithm in cfg.algorithms:
        for enabled in cfg.solutions:
            for shift_length in cfg.shift_lengths:
                subset = [row for row in rows if row["algorithm"] == algorithm and row["solution"] == str(bool(enabled)).lower() and row["shift_length"] == shift_length]
                if len(subset) != len(cfg.seeds):
                    raise ValueError(f"Incomplete condition: {algorithm} {enabled} {shift_length}")
                costs = np.asarray([float(row["episode_cost"]) for row in subset])
                times = np.asarray([float(row["planning_time_seconds"]) for row in subset])
                mode = "corrected" if enabled else "legacy"
                media = f"media/{algorithm}_{mode}_{shift_length}.mp4"
                results.append({
                    "algorithm": algorithm, "solution": mode, "shift_length": shift_length,
                    "mean_cost": float(costs.mean()), "std_cost": float(costs.std(ddof=1)),
                    "mean_planning_time_seconds": float(times.mean()), "std_planning_time_seconds": float(times.std(ddof=1)),
                    "seed_count": len(subset), "video": media, "poster": media.replace(".mp4", ".jpg"),
                })
    return results


def render_dashboard(path: Path, summary: dict[str, Any]) -> None:
    """Create a self-contained local HTML comparison dashboard."""
    template = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(summary, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    path.write_text(template.replace("__TIME_SHIFT_DATA__", payload), encoding="utf-8")


def run(cfg: DictConfig) -> None:
    """Run every full episode, then render fixed-seed videos and HTML."""
    validate(cfg)
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    task, state = make_task(str(cfg.task), str(cfg.backend))
    selections = OmegaConf.load(Path(str(cfg.selected_parameters)).resolve())
    rows, trajectories = [], {}
    for algorithm in cfg.algorithms:
        for enabled in cfg.solutions:
            for shift_length, shift_cfg in cfg.shift_lengths.items():
                controller = make_controller(algorithm, task, cfg.task_config, selections[cfg.task][algorithm].parameters, int(cfg.domain_seed), int(cfg.samples), int(cfg.iterations), bool(enabled))
                steps = execution_steps(controller, float(shift_cfg.execution_fraction))
                compiled = compile_controller(controller, state, int(cfg.warmup_seed), steps)
                for seed in cfg.seeds:
                    record = int(seed) == int(cfg.video_seed)
                    result = run_episode(controller, compiled, state, int(seed), float(cfg.task_config.total_horizon), steps, record)
                    rows.append({"algorithm": algorithm, "solution": str(bool(enabled)).lower(), "shift_length": shift_length, "seed": int(seed), "episode_cost": result.episode_cost, "planning_time_seconds": result.planning_time_seconds, "execution_fraction": steps / controller.ctrl_steps, "execution_horizon": steps * controller.dt})
                    print(f"{algorithm} solution={enabled} shift={shift_length} seed={seed} cost={result.episode_cost:.6g} time={result.planning_time_seconds:.3f}s", flush=True)
                    if record:
                        trajectories[(algorithm, bool(enabled), shift_length)] = result.trajectory
    write_rows(output_dir / "episode_results.csv", rows)
    for (algorithm, enabled, shift_length), trajectory in trajectories.items():
        mode = "corrected" if enabled else "legacy"
        _render_video(output_dir / f"media/{algorithm}_{mode}_{shift_length}.mp4", str(cfg.task), task, state, trajectory, float(cfg.task_config.total_horizon), cfg.render)
    summary = {"created_at": datetime.now().astimezone().isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "backend": jax.default_backend(), "device": str(jax.devices()[0]), "config": OmegaConf.to_container(cfg, resolve=True), "aggregates": aggregate(rows, cfg), "episodes": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_dashboard(output_dir / "index.html", summary)


@hydra.main(version_base=None, config_path=".", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the complete experiment."""
    run(cfg)


if __name__ == "__main__":
    main()
