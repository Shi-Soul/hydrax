"""Run the complete CBO/CEM time-shifting mode experiment."""

import csv
import json
import math
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
    "raw_episode_cost planning_calls planning_time_seconds"
).split()


def validate(cfg: DictConfig) -> None:
    """Validate the fixed, GPU-only paired experimental design."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("The complete experiment must run on a GPU")
    if list(cfg.algorithms) != ["cbo", "cem"]:
        raise ValueError("The experiment requires CBO and CEM")
    if list(cfg.modes) != ["legacy", "reset", "shift"]:
        raise ValueError("modes must be legacy, reset, shift")
    if len(cfg.seeds) < 3 or len(set(int(seed) for seed in cfg.seeds)) != len(cfg.seeds):
        raise ValueError("The experiment requires at least three distinct seeds")
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


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load previously completed episodes for resumable execution."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected result schema in {path}")
        return list(reader)


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
    results_path = output_dir / "episode_results.csv"
    rows: dict[tuple[str, str, str, int], dict[str, Any]] = {
        (
            str(row["algorithm"]),
            str(row["mode"]),
            str(row["shift_name"]),
            int(row["seed"]),
        ): row
        for row in load_rows(results_path)
    }
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
                    key = (algorithm, mode, str(shift_name), seed)
                    record = seed == int(cfg.video_seed)
                    if not record and key in rows:
                        continue
                    result = run_episode(
                        controller,
                        compiled,
                        state,
                        seed,
                        float(cfg.task_config.total_horizon),
                        shift_steps,
                        record,
                    )
                    total_steps = int(
                        round(float(cfg.task_config.total_horizon) / controller.dt)
                    )
                    planning_calls = math.ceil(total_steps / shift_steps)
                    raw_cost = result.episode_cost
                    row = {
                        "algorithm": algorithm,
                        "mode": mode,
                        "shift_name": str(shift_name),
                        "shift_steps": shift_steps,
                        "shift_seconds": shift_steps * controller.dt,
                        "seed": seed,
                        "episode_cost": raw_cost / planning_calls,
                        "raw_episode_cost": raw_cost,
                        "planning_calls": planning_calls,
                        "planning_time_seconds": result.planning_time_seconds,
                    }
                    rows[key] = row
                    write_rows(
                        results_path,
                        [
                            rows[existing_key]
                            for existing_key in sorted(rows)
                        ],
                    )
                    print(
                        f"{algorithm} mode={mode} shift={shift_steps} steps "
                        f"({shift_steps * controller.dt:.3f}s) seed={seed} "
                        f"cost={result.episode_cost:.6g} "
                        f"time={result.planning_time_seconds:.3f}s",
                        flush=True,
                    )
                    if record:
                        trajectories[(algorithm, mode, str(shift_name))] = result.trajectory

    final_rows = [rows[key] for key in sorted(rows)]
    write_rows(results_path, final_rows)
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
        "aggregates": aggregate(final_rows, cfg),
        "episodes": final_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_figures(output_dir, summary["aggregates"])
    render_dashboard(output_dir / "index.html", summary)


def render_figures(output_dir: Path, aggregates: list[dict[str, Any]]) -> None:
    """Render PNG cost curves with mean and one-standard-deviation bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    steps = sorted({int(row["shift_steps"]) for row in aggregates})
    colors = {
        "legacy": "#b46a2d",
        "reset": "#6c4fa1",
        "shift": "#0f8a72",
    }
    linestyles = {"cbo": "-", "cem": "--"}
    markers = {"cbo": "o", "cem": "s"}
    scopes = [
        ("cost_all.png", "All algorithm / mode combinations", ("cbo", "cem")),
        ("cost_cbo.png", "CBO", ("cbo",)),
        ("cost_cem.png", "CEM", ("cem",)),
    ]
    all_costs = []
    for row in aggregates:
        all_costs.extend(
            [row["mean_cost"] - row["std_cost"], row["mean_cost"] + row["std_cost"]]
        )
    y_low = max(min(all_costs) * 0.7, 1e-3)
    y_high = max(all_costs) * 1.3

    for filename, title, algorithms in scopes:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for algorithm in algorithms:
            for mode in ("legacy", "reset", "shift"):
                rows = sorted(
                    (
                        row
                        for row in aggregates
                        if row["algorithm"] == algorithm and row["mode"] == mode
                    ),
                    key=lambda row: row["shift_steps"],
                )
                xs = [int(row["shift_steps"]) for row in rows]
                ys = [float(row["mean_cost"]) for row in rows]
                errors = [float(row["std_cost"]) for row in rows]
                label = (
                    f"{algorithm.upper()} {mode.capitalize()}"
                    if len(algorithms) > 1
                    else mode.capitalize()
                )
                ax.errorbar(
                    xs,
                    ys,
                    yerr=errors,
                    fmt=markers[algorithm] + linestyles[algorithm],
                    color=colors[mode],
                    capsize=4,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_ylim(y_low, y_high)
        ax.set_xticks(steps)
        ax.set_xlabel("Shift steps")
        ax.set_ylabel("Mean episode cost / planning call (log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)


@hydra.main(version_base=None, config_path=".", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the complete experiment."""
    run(cfg)


if __name__ == "__main__":
    main()
