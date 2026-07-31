"""Aggregate the operation-budget grid and write validated result artifacts."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from examples.to.benchmarking.benchmark_figures import (
    METHOD_COLORS,
    METHOD_NAMES,
    write_benchmark_figures,
)

AGGREGATE_FIELDS = (
    "algorithm nominal_operations actual_operations iterations samples "
    "mean_cost std_cost mean_planning_time_seconds "
    "std_planning_time_seconds finite_seeds seed_count video_seed "
    "video_seed_cost"
).split()


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
    write_benchmark_figures(aggregates, scaling, cfg, figures_dir)
    _write_report(output_dir / "report.md", cfg, baseline, scaling)


@hydra.main(version_base=None, config_path=".", config_name="grid")
def main(cfg: DictConfig) -> None:
    """Aggregate the complete grid and write all requested static figures."""
    _run(cfg)


if __name__ == "__main__":
    main()
