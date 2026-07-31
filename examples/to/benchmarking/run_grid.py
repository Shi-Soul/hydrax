"""Run the complete Double Cart Pole operation-budget grid."""

import csv
import gc
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import jax
import mujoco
from omegaconf import DictConfig, OmegaConf

from examples.to.open_loop_benchmark import (
    ALGORITHMS,
    compile_controller,
    execution_steps,
    make_controller,
    make_task,
    run_episode,
)

RESULT_FIELDS = (
    "task algorithm nominal_operations actual_operations "
    "iterations samples seed "
    "episode_cost planning_time_seconds segment_horizon total_horizon "
    "execution_fraction execution_horizon planning_calls randomizations knots "
    "spline frequency max_episode_steps base_noise candidate parameters"
).split()


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["algorithm"]),
        int(row["nominal_operations"]),
        int(row["iterations"]),
        int(row["seed"]),
    )


def _load_rows(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not path.exists() or not resume:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(f"Unexpected result schema in {path}")
        return list(reader)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_run_metadata(output_dir: Path, cfg: DictConfig) -> None:
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (output_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(),
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _validate(cfg: DictConfig) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("The complete benchmark must run on a GPU")
    algorithms = [str(value) for value in cfg.algorithms]
    if set(algorithms) != set(ALGORITHMS) or len(algorithms) != len(ALGORITHMS):
        raise ValueError("The grid must contain every algorithm exactly once")
    if len(cfg.seeds) != 5 or len(set(int(seed) for seed in cfg.seeds)) != 5:
        raise ValueError("The grid requires exactly five distinct seeds")
    for operations in cfg.operation_budgets:
        for iterations in cfg.iterations:
            if int(operations) % int(iterations) != 0:
                raise ValueError(
                    "Every operation budget must divide by iterations"
                )
    if int(cfg.task_config.knots) != 6:
        raise ValueError("Double Cart Pole grid requires six knots")
    if float(cfg.task_config.total_horizon) != 10.0:
        raise ValueError("Double Cart Pole grid requires a ten-second episode")


def _result_row(
    cfg: DictConfig,
    algorithm: str,
    operations: int,
    iterations: int,
    samples: int,
    seed: int,
    controller: Any,
    executed_steps: int,
    selection: DictConfig,
    episode_cost: float,
    planning_time: float,
) -> dict[str, Any]:
    total_horizon = float(cfg.task_config.total_horizon)
    total_steps = int(round(total_horizon / controller.dt))
    planning_calls = -(-total_steps // executed_steps)
    parameters = json.dumps(
        OmegaConf.to_container(selection.parameters, resolve=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "task": str(cfg.task),
        "algorithm": algorithm,
        "nominal_operations": operations,
        "actual_operations": iterations * samples,
        "iterations": iterations,
        "samples": samples,
        "seed": seed,
        "episode_cost": episode_cost,
        "planning_time_seconds": planning_time,
        "segment_horizon": controller.plan_horizon,
        "total_horizon": total_horizon,
        "execution_fraction": executed_steps / controller.ctrl_steps,
        "execution_horizon": executed_steps * controller.dt,
        "planning_calls": planning_calls,
        "randomizations": controller.num_randomizations,
        "knots": controller.num_knots,
        "spline": controller.spline_type,
        "frequency": 1.0 / controller.dt,
        "max_episode_steps": total_steps,
        "base_noise": float(cfg.task_config.base_noise),
        "candidate": int(selection.candidate),
        "parameters": parameters,
    }


def _run(cfg: DictConfig) -> None:
    _validate(cfg)
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "grid_results.csv"
    rows = _load_rows(results_path, bool(cfg.resume))
    completed = {_row_key(row) for row in rows}
    expected = (
        len(cfg.algorithms)
        * len(cfg.operation_budgets)
        * len(cfg.iterations)
        * len(cfg.seeds)
    )
    if len(completed) != len(rows):
        raise ValueError("Duplicate rows found in existing grid results")
    _write_run_metadata(output_dir, cfg)
    selections = OmegaConf.load(Path(str(cfg.selected_parameters)).resolve())
    task, state = make_task(str(cfg.task), str(cfg.backend))

    for algorithm_value in cfg.algorithms:
        algorithm = str(algorithm_value)
        selection = selections[cfg.task][algorithm]
        for operations_value in cfg.operation_budgets:
            operations = int(operations_value)
            for iterations_value in cfg.iterations:
                iterations = int(iterations_value)
                samples = operations // iterations
                pending_seeds = [
                    int(seed)
                    for seed in cfg.seeds
                    if (algorithm, operations, iterations, int(seed))
                    not in completed
                ]
                if not pending_seeds:
                    continue
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
                for seed in pending_seeds:
                    result = run_episode(
                        controller,
                        compiled,
                        state,
                        seed,
                        float(cfg.task_config.total_horizon),
                        executed_steps,
                        False,
                    )
                    row = _result_row(
                        cfg,
                        algorithm,
                        operations,
                        iterations,
                        samples,
                        seed,
                        controller,
                        executed_steps,
                        selection,
                        result.episode_cost,
                        result.planning_time_seconds,
                    )
                    rows.append(row)
                    completed.add(_row_key(row))
                    _write_rows(results_path, rows)
                    print(
                        f"[{len(rows):03d}/{expected}] {algorithm:9s} "
                        f"O={operations:4d} I={iterations:2d} N={samples:4d} "
                        f"seed={seed} cost={result.episode_cost:.6g} "
                        f"time={result.planning_time_seconds:.3f}s",
                        flush=True,
                    )
                del compiled, controller
                gc.collect()
                jax.clear_caches()

    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")


@hydra.main(version_base=None, config_path=".", config_name="grid")
def main(cfg: DictConfig) -> None:
    """Run or resume the complete operation-budget grid."""
    _run(cfg)


if __name__ == "__main__":
    main()
