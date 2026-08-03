"""Render publication figures for aggregated benchmark results."""

from pathlib import Path
from typing import Any

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
        axis.errorbar(
            [row["nominal_operations"] for row in rows],
            [row["mean_cost"] for row in rows],
            yerr=[row["std_cost"] for row in rows],
            label=METHOD_NAMES[algorithm],
            color=METHOD_COLORS[algorithm],
            marker=MARKERS[algorithm],
            capsize=2.5,
        )
    budgets = sorted(int(value) for value in cfg.operation_budgets)
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets)
    axis.set_xticklabels(budgets)
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
        budgets = sorted(int(value) for value in cfg.operation_budgets)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(budgets)
        axis.set_xticklabels(budgets)
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


def _plot_planning_time_heatmap(
    aggregates: list[dict[str, Any]], cfg: DictConfig, figures_dir: Path
) -> None:
    budgets = sorted(int(value) for value in cfg.operation_budgets)
    iterations = sorted(int(value) for value in cfg.iterations)
    times = np.asarray(
        [row["mean_planning_time_seconds"] for row in aggregates],
        dtype=float,
    )
    if not np.all(np.isfinite(times)):
        raise ValueError("Planning-time heatmap requires finite aggregates")
    color_map = plt.colormaps["viridis"]
    normalization = matplotlib.colors.Normalize(
        vmin=float(np.min(times)), vmax=float(np.max(times))
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1), sharex=True, sharey=True)
    image = None
    for axis, algorithm_value in zip(axes.flat, cfg.algorithms, strict=True):
        algorithm = str(algorithm_value)
        indexed = {
            (row["iterations"], row["nominal_operations"]): row
            for row in aggregates
            if row["algorithm"] == algorithm
        }
        expected_cells = len(iterations) * len(budgets)
        if len(indexed) != expected_cells:
            raise ValueError(
                f"Expected {expected_cells} planning-time cells for {algorithm}"
            )
        matrix = np.asarray(
            [
                [
                    indexed[iteration, budget]["mean_planning_time_seconds"]
                    for budget in budgets
                ]
                for iteration in iterations
            ],
            dtype=float,
        )
        image = axis.imshow(
            matrix,
            cmap=color_map,
            norm=normalization,
            origin="lower",
            aspect="auto",
        )
        axis.grid(False)
        axis.set_title(METHOD_NAMES[algorithm])
        axis.set_xticks(range(len(budgets)), budgets, rotation=30)
        axis.set_yticks(range(len(iterations)), iterations)
        axis.set_xlabel("Operation budget O")
        axis.set_ylabel("Iterations I")
        for row_index in range(len(iterations)):
            for column_index in range(len(budgets)):
                value = matrix[row_index, column_index]
                red, green, blue, _ = color_map(normalization(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if luminance < 0.48 else "#171b19",
                    fontsize=6.5,
                )
    if image is None:
        raise ValueError("Planning-time heatmap has no methods")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Mean planning time per episode (s)",
        fraction=0.025,
        pad=0.025,
    )
    fig.suptitle("Actual compute time across I,O allocations", weight="bold")
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.34, wspace=0.24)
    _save_figure(fig, figures_dir, "planning_time_heatmap")


def _plot_optimal_planning_time(
    scaling: list[dict[str, Any]], cfg: DictConfig, figures_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(6.75, 3.4))
    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        rows = _method_rows(scaling, algorithm)
        axis.errorbar(
            [row["nominal_operations"] for row in rows],
            [row["mean_planning_time_seconds"] for row in rows],
            yerr=[row["std_planning_time_seconds"] for row in rows],
            label=METHOD_NAMES[algorithm],
            color=METHOD_COLORS[algorithm],
            marker=MARKERS[algorithm],
            capsize=2.5,
        )
    budgets = sorted(int(value) for value in cfg.operation_budgets)
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets)
    axis.set_xticklabels(budgets)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Operation budget O")
    axis.set_ylabel("Planning time per episode (s)")
    axis.set_title("Compute time of cost-optimal allocations")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _save_figure(fig, figures_dir, "optimal_configuration_planning_time")


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


def write_benchmark_figures(
    aggregates: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
    cfg: DictConfig,
    figures_dir: Path,
) -> None:
    """Write every publication figure from validated aggregate rows."""
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
    _plot_planning_time_heatmap(aggregates, cfg, figures_dir)
    _plot_optimal_planning_time(scaling, cfg, figures_dir)
    _plot_optimal_trajectory(scaling, cfg, figures_dir)
