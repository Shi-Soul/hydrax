"""Aggregate the operation-budget grid and create publication figures."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METHOD_NAMES = {
    "cbo": "CBO",
    "mppi_cma": "MPPI-CMA",
    "cmaes": "CMA-ES",
    "cem": "CEM",
    "ps": "Predictive Sampling",
    "dial": "DIAL",
}
METHOD_COLORS = {
    "cbo": "#E69F00",
    "mppi_cma": "#56B4E9",
    "cmaes": "#009E73",
    "cem": "#0072B2",
    "ps": "#D55E00",
    "dial": "#CC79A7",
}
MARKERS = {
    "cbo": "o",
    "mppi_cma": "s",
    "cmaes": "^",
    "cem": "D",
    "ps": "v",
    "dial": "P",
}
AGGREGATE_FIELDS = (
    "algorithm nominal_operations actual_operations iterations samples "
    "mean_cost std_cost mean_planning_time_seconds "
    "std_planning_time_seconds finite_seeds seed_count video_seed "
    "video_seed_cost"
).split()


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
            "lines.markersize": 5,
        }
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = []
    for row in raw_rows:
        rows.append(
            row
            | {
                "nominal_operations": int(row["nominal_operations"]),
                "actual_operations": int(row["actual_operations"]),
                "iterations": int(row["iterations"]),
                "samples": int(row["samples"]),
                "seed": int(row["seed"]),
                "episode_cost": float(row["episode_cost"]),
                "planning_time_seconds": float(row["planning_time_seconds"]),
            }
        )
    return rows


def _finite_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        return None, None
    return float(np.mean(array)), float(np.std(array, ddof=1))


def _aggregate(
    rows: list[dict[str, Any]], cfg: DictConfig
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row["algorithm"]),
            int(row["nominal_operations"]),
            int(row["iterations"]),
        )
        grouped[key].append(row)
    expected_groups = (
        len(cfg.algorithms) * len(cfg.operation_budgets) * len(cfg.iterations)
    )
    if len(grouped) != expected_groups:
        raise ValueError(f"Expected {expected_groups} configurations")

    aggregates = []
    expected_seeds = sorted(int(seed) for seed in cfg.seeds)
    for (algorithm, operations, iterations), selected in grouped.items():
        selected.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in selected] != expected_seeds:
            raise ValueError(
                "Incomplete seeds for "
                f"{algorithm}, O={operations}, I={iterations}"
            )
        samples = {int(row["samples"]) for row in selected}
        actual_operations = {int(row["actual_operations"]) for row in selected}
        if len(samples) != 1 or len(actual_operations) != 1:
            raise ValueError("Configuration values differ across seeds")
        costs = [float(row["episode_cost"]) for row in selected]
        times = [float(row["planning_time_seconds"]) for row in selected]
        mean_cost, std_cost = _finite_mean_std(costs)
        mean_time, std_time = _finite_mean_std(times)
        finite_rows = [
            row for row in selected if math.isfinite(float(row["episode_cost"]))
        ]
        video_row = min(
            finite_rows or selected, key=lambda row: float(row["episode_cost"])
        )
        aggregates.append(
            {
                "algorithm": algorithm,
                "nominal_operations": operations,
                "actual_operations": actual_operations.pop(),
                "iterations": iterations,
                "samples": samples.pop(),
                "mean_cost": mean_cost,
                "std_cost": std_cost,
                "mean_planning_time_seconds": mean_time,
                "std_planning_time_seconds": std_time,
                "finite_seeds": len(finite_rows),
                "seed_count": len(selected),
                "video_seed": int(video_row["seed"]),
                "video_seed_cost": float(video_row["episode_cost"]),
                "seed_results": [
                    {
                        "seed": int(row["seed"]),
                        "cost": (
                            float(row["episode_cost"])
                            if math.isfinite(float(row["episode_cost"]))
                            else None
                        ),
                        "planning_time_seconds": float(
                            row["planning_time_seconds"]
                        ),
                    }
                    for row in selected
                ],
            }
        )
    order = {str(name): index for index, name in enumerate(cfg.algorithms)}
    return sorted(
        aggregates,
        key=lambda row: (
            order[row["algorithm"]],
            row["nominal_operations"],
            row["iterations"],
        ),
    )


def _best_configurations(
    aggregates: list[dict[str, Any]], cfg: DictConfig
) -> list[dict[str, Any]]:
    scaling = []
    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        for operations_value in cfg.operation_budgets:
            operations = int(operations_value)
            candidates = [
                row
                for row in aggregates
                if row["algorithm"] == algorithm
                and row["nominal_operations"] == operations
            ]
            best = min(
                candidates,
                key=lambda row: (
                    row["mean_cost"]
                    if row["mean_cost"] is not None
                    else math.inf,
                    row["iterations"],
                ),
            ).copy()
            best["video"] = f"media/{algorithm}__O{operations}.mp4"
            best["poster"] = f"media/{algorithm}__O{operations}.jpg"
            scaling.append(best)
    return scaling


def _baseline(
    aggregates: list[dict[str, Any]], cfg: DictConfig
) -> list[dict[str, Any]]:
    baseline = []
    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        selected = [
            row.copy()
            for row in aggregates
            if row["algorithm"] == algorithm
            and row["nominal_operations"] == int(cfg.baseline.operation_budget)
            and row["iterations"] == int(cfg.baseline.iterations)
            and row["samples"] == int(cfg.baseline.samples)
        ]
        if len(selected) != 1:
            raise ValueError(f"Missing baseline for {algorithm}")
        selected[0]["video"] = f"media/{algorithm}__baseline.mp4"
        selected[0]["poster"] = f"media/{algorithm}__baseline.jpg"
        baseline.extend(selected)
    return baseline


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, AGGREGATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in AGGREGATE_FIELDS})
    temporary.replace(path)


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.pdf")
    fig.savefig(figures_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _method_rows(
    rows: list[dict[str, Any]], algorithm: str
) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["algorithm"] == algorithm],
        key=lambda row: row["nominal_operations"],
    )


def _plot_scaling(
    scaling: list[dict[str, Any]], cfg: DictConfig, figures_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(6.75, 3.4))
    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        rows = _method_rows(scaling, algorithm)
        x = [row["nominal_operations"] for row in rows]
        y = [row["mean_cost"] for row in rows]
        error = [row["std_cost"] for row in rows]
        axis.errorbar(
            x,
            y,
            yerr=error,
            label=METHOD_NAMES[algorithm],
            color=METHOD_COLORS[algorithm],
            marker=MARKERS[algorithm],
            capsize=2.5,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(int(value) for value in cfg.operation_budgets))
    axis.set_xticklabels(sorted(int(value) for value in cfg.operation_budgets))
    axis.set_xlabel("Operation budget O")
    axis.set_ylabel("Episode cost (lower is better)")
    axis.set_title("Best-over-I scaling on Double Cart Pole")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _save_figure(fig, figures_dir, "scaling_curve")


def _plot_optimal_hyperparameters(
    scaling: list[dict[str, Any]], cfg: DictConfig, figures_dir: Path
) -> None:
    for field, label, stem in (
        ("iterations", "Selected iterations I*", "optimal_iterations"),
        ("samples", "Selected samples N*", "optimal_samples"),
    ):
        fig, axis = plt.subplots(figsize=(6.75, 3.3))
        for algorithm_value in cfg.algorithms:
            algorithm = str(algorithm_value)
            rows = _method_rows(scaling, algorithm)
            axis.plot(
                [row["nominal_operations"] for row in rows],
                [row[field] for row in rows],
                label=METHOD_NAMES[algorithm],
                color=METHOD_COLORS[algorithm],
                marker=MARKERS[algorithm],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(sorted(int(value) for value in cfg.operation_budgets))
        axis.set_xticklabels(
            sorted(int(value) for value in cfg.operation_budgets)
        )
        ticks = sorted({int(row[field]) for row in scaling})
        axis.set_yticks(ticks)
        axis.set_yticklabels(ticks)
        axis.set_xlabel("Operation budget O")
        axis.set_ylabel(label)
        axis.set_title(f"{label} by operation budget")
        axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        _save_figure(fig, figures_dir, stem)


def _plot_performance_sensitivity(
    aggregates: list[dict[str, Any]],
    cfg: DictConfig,
    figures_dir: Path,
    x_field: str,
    x_label: str,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True)
    budgets = sorted(int(value) for value in cfg.operation_budgets)
    budget_colors = plt.colormaps["viridis"](
        np.linspace(0.12, 0.88, len(budgets))
    )
    for axis, algorithm_value in zip(axes.flat, cfg.algorithms, strict=True):
        algorithm = str(algorithm_value)
        for operations, color in zip(budgets, budget_colors, strict=True):
            rows = sorted(
                [
                    row
                    for row in aggregates
                    if row["algorithm"] == algorithm
                    and row["nominal_operations"] == operations
                ],
                key=lambda row: row[x_field],
            )
            axis.errorbar(
                [row[x_field] for row in rows],
                [row["mean_cost"] for row in rows],
                yerr=[row["std_cost"] for row in rows],
                color=color,
                marker="o",
                markersize=3.5,
                capsize=2,
                label=f"O={operations}",
            )
        axis.set_xscale("log", base=2)
        axis.set_title(METHOD_NAMES[algorithm])
        axis.set_xlabel(x_label)
        axis.set_ylabel("Episode cost")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.01)
    )
    fig.suptitle(
        f"Performance across all I,N allocations by {x_label}",
        y=1.06,
        weight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir, stem)


def _plot_optimal_trajectory(
    scaling: list[dict[str, Any]], cfg: DictConfig, figures_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    budgets = sorted(int(value) for value in cfg.operation_budgets)
    budget_colors = plt.colormaps["viridis"](
        np.linspace(0.12, 0.88, len(budgets))
    )
    budget_map = dict(zip(budgets, budget_colors, strict=True))
    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        rows = _method_rows(scaling, algorithm)
        axis.plot(
            [row["samples"] for row in rows],
            [row["iterations"] for row in rows],
            color=METHOD_COLORS[algorithm],
            marker=None,
            alpha=0.8,
            label=METHOD_NAMES[algorithm],
        )
        for row in rows:
            axis.scatter(
                row["samples"],
                row["iterations"],
                color=budget_map[row["nominal_operations"]],
                edgecolor=METHOD_COLORS[algorithm],
                linewidth=1.2,
                marker=MARKERS[algorithm],
                s=38,
                zorder=3,
            )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Selected samples N*")
    axis.set_ylabel("Selected iterations I*")
    axis.set_title("Optimal configuration trajectories")
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[algorithm],
            marker=MARKERS[algorithm],
            label=METHOD_NAMES[algorithm],
        )
        for algorithm in METHOD_NAMES
    ]
    budget_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor=budget_map[operations],
            markeredgecolor="#333333",
            label=f"O={operations}",
        )
        for operations in budgets
    ]
    fig.subplots_adjust(right=0.72)
    fig.legend(
        handles=method_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.73, 0.9),
    )
    fig.legend(
        handles=budget_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.73, 0.46),
    )
    _save_figure(fig, figures_dir, "optimal_configuration_trajectory")


def _format_stat(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "$\\infty \\pm \\mathrm{N/A}$"
    return f"${mean:.6g} \\pm {std:.3g}$"


def _write_report(
    path: Path,
    cfg: DictConfig,
    baseline: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
) -> None:
    lines = [
        "# Double Cart Pole Operation-Budget Benchmark",
        "",
        "The operation proxy is $O=I N$, where $I$ is the number of optimizer "
        "iterations and $N$ is the number of samples per iteration. The best "
        "configuration minimizes the five-seed mean episode cost.",
        "",
        f"- Seeds: `{list(cfg.seeds)}`",
        f"- Total horizon: ${float(cfg.task_config.total_horizon):g}$ s",
        f"- Planning horizon: ${float(cfg.task_config.segment_horizon):g}$ s",
        f"- Knots: ${int(cfg.task_config.knots)}$",
        "- All algorithm-specific parameters are fixed across the grid.",
        "",
        "## Fixed Baseline",
        "",
        "| Method | $I$ | $N$ | $O$ | Episode cost | Planning time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        planning_stat = _format_stat(
            row["mean_planning_time_seconds"],
            row["std_planning_time_seconds"],
        )
        lines.append(
            f"| {METHOD_NAMES[row['algorithm']]} | {row['iterations']} | "
            f"{row['samples']} | {row['nominal_operations']} | "
            f"{_format_stat(row['mean_cost'], row['std_cost'])} | "
            f"{planning_stat} |"
        )
    lines += [
        "",
        "## Best Allocation by Operation Budget",
        "",
        "| Method | $O$ | $I^*$ | $N^*$ | Episode cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scaling:
        lines.append(
            f"| {METHOD_NAMES[row['algorithm']]} | "
            f"{row['nominal_operations']} | "
            f"{row['iterations']} | {row['samples']} | "
            f"{_format_stat(row['mean_cost'], row['std_cost'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir)).resolve()
    rows = _read_rows(output_dir / "grid_results.csv")
    expected = (
        len(cfg.algorithms)
        * len(cfg.operation_budgets)
        * len(cfg.iterations)
        * len(cfg.seeds)
    )
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} result rows, found {len(rows)}")
    aggregates = _aggregate(rows, cfg)
    scaling = _best_configurations(aggregates, cfg)
    baseline = _baseline(aggregates, cfg)
    _write_csv(output_dir / "aggregate_results.csv", aggregates)
    _write_csv(output_dir / "scaling_results.csv", scaling)
    _write_csv(output_dir / "baseline_results.csv", baseline)
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    summary = {
        "metadata": metadata
        | {
            "task": str(cfg.task),
            "seeds": [int(seed) for seed in cfg.seeds],
            "algorithms": [str(value) for value in cfg.algorithms],
            "operation_budgets": sorted(
                int(value) for value in cfg.operation_budgets
            ),
            "iterations": sorted(int(value) for value in cfg.iterations),
            "task_config": dict(cfg.task_config),
            "baseline": dict(cfg.baseline),
            "cost_direction": "minimize",
        },
        "method_names": METHOD_NAMES,
        "method_colors": METHOD_COLORS,
        "aggregates": aggregates,
        "scaling": scaling,
        "baseline": baseline,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _configure_plot_style()
    _plot_scaling(scaling, cfg, figures_dir)
    _plot_optimal_hyperparameters(scaling, cfg, figures_dir)
    _plot_performance_sensitivity(
        aggregates,
        cfg,
        figures_dir,
        "samples",
        "Samples N",
        "performance_vs_samples",
    )
    _plot_performance_sensitivity(
        aggregates,
        cfg,
        figures_dir,
        "iterations",
        "Iterations I",
        "performance_vs_iterations",
    )
    _plot_optimal_trajectory(scaling, cfg, figures_dir)
    _write_report(output_dir / "report.md", cfg, baseline, scaling)


@hydra.main(version_base=None, config_path=".", config_name="grid")
def main(cfg: DictConfig) -> None:
    """Aggregate the complete grid and write all requested static figures."""
    _run(cfg)


if __name__ == "__main__":
    main()
