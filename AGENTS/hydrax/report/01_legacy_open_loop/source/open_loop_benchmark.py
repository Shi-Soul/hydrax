import csv
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import hydra
import jax
import jax.numpy as jnp
from evosax.algorithms.distribution_based import CMA_ES
from omegaconf import DictConfig, OmegaConf

from hydrax.alg_base import SamplingBasedController
from hydrax.algs import CBO, CEM, DIAL, Evosax, MppiCma, PredictiveSampling
from hydrax.risk import AverageCost
from hydrax.task_base import Task
from hydrax.tasks.bugtrap import BugTrap
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.tasks.pusht import PushT

TASK_TYPES: dict[str, type[Task]] = {
    "cart_pole": CartPole,
    "double_cart_pole": DoubleCartPole,
    "humanoid_standup": HumanoidStandup,
    "pusht": PushT,
    "bugtrap": BugTrap,
}
ALGORITHM_NAMES = ("cbo", "mppi_cma", "cmaes", "cem", "ps", "dial")
RESULT_FIELDS = (
    "task",
    "algorithm",
    "candidate",
    "seed",
    "best_cost",
    "time_seconds",
    "iterations",
    "horizon",
    "samples",
    "randomizations",
    "knots",
    "spline",
    "frequency",
    "max_episode_steps",
    "base_noise",
    "parameters",
)


def make_task(task_name: str, backend: str) -> tuple[Task, Any]:
    """Create a task and the fixed initial state used by its example."""
    if task_name not in TASK_TYPES:
        raise ValueError(f"Unknown task: {task_name}")

    task = TASK_TYPES[task_name](impl=backend)
    state = task.make_data()

    if task_name == "humanoid_standup":
        qpos = jnp.asarray(task.mj_model.keyframe("stand").qpos)
        qpos = qpos.at[3:7].set(jnp.array([0.7, 0.0, -0.7, 0.0]))
        state = state.replace(qpos=qpos)
    elif task_name == "pusht":
        state = state.replace(qpos=jnp.array([0.1, 0.1, 1.3, 0.0, 0.0]))
    elif task_name == "bugtrap":
        qpos = state.qpos.at[:2].set(jnp.array([-0.15, 0.0]))
        target = jnp.array([0.25, 0.0, 0.01])
        mocap_pos = state.mocap_pos.at[0].set(target)
        state = state.replace(qpos=qpos, mocap_pos=mocap_pos)

    return task, state


def common_arguments(
    task: Task, task_cfg: DictConfig, domain_seed: int
) -> dict[str, Any]:
    """Return the optimization budget shared by every algorithm."""
    return {
        "task": task,
        "num_samples": int(task_cfg.samples),
        "num_randomizations": int(task_cfg.randomizations),
        "risk_strategy": AverageCost(),
        "seed": domain_seed,
        "plan_horizon": float(task_cfg.horizon),
        "spline_type": str(task_cfg.spline),
        "num_knots": int(task_cfg.knots),
        "iterations": int(task_cfg.iterations),
    }


def make_cmaes(
    task: Task,
    task_cfg: DictConfig,
    candidate: DictConfig,
    domain_seed: int,
) -> Evosax:
    """Create CMA-ES with the task-level initial standard deviation."""
    num_samples = int(task_cfg.samples)
    num_dims = task.model.nu * int(task_cfg.knots)
    strategy = CMA_ES(
        population_size=num_samples,
        solution=jnp.zeros(num_dims),
    )
    base_noise = float(task_cfg.base_noise)
    es_params = strategy.default_params.replace(
        std_init=base_noise,
        std_min=base_noise * float(candidate.std_min_ratio),
        std_max=base_noise * float(candidate.std_max_ratio),
        c_mean=float(candidate.c_mean),
    )
    arguments = common_arguments(task, task_cfg, domain_seed)
    return Evosax(optimizer=CMA_ES, es_params=es_params, **arguments)


def make_controller(
    algorithm: str,
    task: Task,
    task_cfg: DictConfig,
    candidate: DictConfig,
    domain_seed: int,
) -> SamplingBasedController:
    """Create one optimizer while keeping the shared budget explicit."""
    arguments = common_arguments(task, task_cfg, domain_seed)
    base_noise = float(task_cfg.base_noise)
    if algorithm == "cbo":
        return CBO(
            initial_noise_level=base_noise,
            temperature=(
                float(task_cfg.temperature) * float(candidate.temperature_scale)
            ),
            consensus_weight=float(candidate.consensus_weight),
            noise_weight=float(candidate.noise_weight),
            step_size=float(candidate.step_size),
            **arguments,
        )
    if algorithm == "mppi_cma":
        return MppiCma(
            initial_noise_level=base_noise,
            temperature=(
                float(task_cfg.temperature) * float(candidate.temperature_scale)
            ),
            minimum_noise_level=(
                base_noise * float(candidate.minimum_noise_ratio)
            ),
            covariance_adaptation_rate=float(candidate.adaptation_rate),
            **arguments,
        )
    if algorithm == "cmaes":
        return make_cmaes(task, task_cfg, candidate, domain_seed)
    if algorithm == "cem":
        num_elites = round(
            int(task_cfg.samples) * float(candidate.elite_fraction)
        )
        num_elites = max(2, num_elites)
        return CEM(
            num_elites=num_elites,
            sigma_start=base_noise,
            sigma_min=base_noise * float(candidate.sigma_min_ratio),
            explore_fraction=float(candidate.explore_fraction),
            **arguments,
        )
    if algorithm == "ps":
        return PredictiveSampling(noise_level=base_noise, **arguments)
    if algorithm == "dial":
        return DIAL(
            noise_level=base_noise,
            beta_opt_iter=float(candidate.beta_opt_iter),
            beta_horizon=float(candidate.beta_horizon),
            temperature=(
                float(task_cfg.temperature) * float(candidate.temperature_scale)
            ),
            **arguments,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def compile_optimizer(
    controller: SamplingBasedController,
    state: Any,
    warmup_seed: int,
) -> Callable[[Any, Any], Any]:
    """Compile with disposable parameters so warmup cannot improve a run."""
    optimizer = jax.jit(controller.optimize)
    warmup_params = controller.init_params(initial_knots=None, seed=warmup_seed)
    _, warmup_rollouts = optimizer(state, warmup_params)
    jax.block_until_ready(warmup_rollouts.costs)
    return optimizer


def run_seed(
    controller: SamplingBasedController,
    optimizer: Callable[[Any, Any], Any],
    state: Any,
    seed: int,
) -> tuple[float, float]:
    """Measure one complete open-loop optimization after compilation."""
    params = controller.init_params(initial_knots=None, seed=seed)
    jax.block_until_ready(params)
    start = time.perf_counter()
    _, rollouts = optimizer(state, params)
    jax.block_until_ready(rollouts.costs)
    elapsed = time.perf_counter() - start
    costs = jnp.sum(rollouts.costs, axis=1)
    best_cost = float(jnp.min(jnp.where(jnp.isfinite(costs), costs, jnp.inf)))
    return best_cost, elapsed


def parameter_json(candidate: DictConfig) -> str:
    """Serialize a resolved candidate for raw result files."""
    parameters = OmegaConf.to_container(candidate, resolve=True)
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def result_row(
    phase: str,
    task_name: str,
    algorithm: str,
    candidate_index: int,
    seed: int,
    best_cost: float,
    elapsed: float,
    controller: SamplingBasedController,
    task_cfg: DictConfig,
    candidate: DictConfig,
) -> dict[str, Any]:
    """Create one self-contained raw benchmark record."""
    return {
        "task": task_name,
        "algorithm": algorithm,
        "candidate": candidate_index,
        "seed": seed,
        "best_cost": best_cost,
        "time_seconds": elapsed,
        "iterations": controller.iterations,
        "horizon": controller.plan_horizon,
        "samples": int(task_cfg.samples),
        "randomizations": controller.num_randomizations,
        "knots": controller.num_knots,
        "spline": controller.spline_type,
        "frequency": 1.0 / controller.dt,
        "max_episode_steps": controller.ctrl_steps,
        "base_noise": float(task_cfg.base_noise),
        "parameters": parameter_json(candidate),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a benchmark CSV with all completed rows."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def run_evaluation(
    phase: str,
    task_name: str,
    algorithm: str,
    candidate_index: int,
    candidate: DictConfig,
    controller: SamplingBasedController,
    optimizer: Callable[[Any, Any], Any],
    state: Any,
    seeds: list[int],
    task_cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Run every seed for one task, algorithm, and parameter candidate."""
    rows = []
    for seed in seeds:
        best_cost, elapsed = run_seed(controller, optimizer, state, seed)
        row = result_row(
            phase,
            task_name,
            algorithm,
            candidate_index,
            seed,
            best_cost,
            elapsed,
            controller,
            task_cfg,
            candidate,
        )
        rows.append(row)
        print(
            f"{phase:5s} {task_name:20s} {algorithm:9s} "
            f"candidate={candidate_index} seed={seed} "
            f"cost={best_cost:.6g} time={elapsed:.4f}s"
        )
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]], task_name: str, algorithm: str
) -> tuple[float, float, float, float]:
    """Aggregate cost and synchronized optimization time across seeds."""
    selected = [
        row
        for row in rows
        if row["task"] == task_name and row["algorithm"] == algorithm
    ]
    if len(selected) < 2:
        raise ValueError("At least two evaluation seeds are required")
    costs = [float(row["best_cost"]) for row in selected]
    times_ms = [1000.0 * float(row["time_seconds"]) for row in selected]
    return (
        statistics.mean(costs),
        statistics.stdev(costs),
        statistics.mean(times_ms),
        statistics.stdev(times_ms),
    )


def write_report(
    path: Path,
    cfg: DictConfig,
    rows: list[dict[str, Any]],
    selections: dict[str, Any],
) -> None:
    """Write the requested per-task Markdown summary table."""
    lines = [
        "# Hydrax Open-Loop Planning Benchmark",
        "",
        "Each run optimizes once from the fixed example initial state. JIT "
        "compilation, setup, metric reduction, and output are excluded from "
        "planning time. The reported best cost is the minimum "
        "total rollout cost in the final internal optimization iteration.",
        "",
        f"- Backend: `{cfg.backend}`",
        f"- Device: `{jax.devices()[0]}`",
        f"- JAX: `{version('jax')}`",
        f"- MuJoCo: `{version('mujoco')}`",
        f"- Evaluation seeds: `{list(cfg.evaluation_seeds)}`",
        "- Control frequency and maximum episode steps are open-loop "
        "discretization metadata, with $f = 1 / \\Delta t$ and "
        "$N = \\operatorname{round}(T / \\Delta t)$.",
        "",
        "| Task | Algorithm | Final best cost | Planning time (ms) | "
        "Iterations | Horizon | Samples | Randomizations | Knots | Spline | "
        "Frequency | Steps | Base noise | Selected parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]

    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        for algorithm in cfg.algorithms_to_run:
            cost_mean, cost_std, time_mean, time_std = aggregate_rows(
                rows, task_name, algorithm
            )
            task_rows = [
                row
                for row in rows
                if row["task"] == task_name and row["algorithm"] == algorithm
            ]
            row = task_rows[0]
            params = json.dumps(
                selections[task_name][algorithm]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.append(
                f"| {task_name} | {algorithm} | "
                f"${cost_mean:.6g} \\pm {cost_std:.3g}$ | "
                f"${time_mean:.3f} \\pm {time_std:.3f}$ | "
                f"{row['iterations']} | {float(row['horizon']):.3g} | "
                f"{row['samples']} | {row['randomizations']} | "
                f"{row['knots']} | {row['spline']} | "
                f"{float(row['frequency']):.3g} | "
                f"{row['max_episode_steps']} | "
                f"{float(task_cfg.base_noise):.3g} | `{params}` |"
            )

    lines.extend(
        [
            "",
            "## Tuning Protocol",
            "",
            "Candidates were compared by mean final best cost on the tuning "
            "seeds. Evaluation uses disjoint seeds. Predictive sampling has no "
            "algorithm-specific parameter once shared base noise is fixed. "
            "The complete candidate results are in `tuning_results.csv`, and "
            "the exact selected values are in `selected_parameters.yaml`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_config(cfg: DictConfig) -> None:
    """Reject incomplete or unfair benchmark configurations early."""
    if cfg.mode not in {"all", "tune", "benchmark"}:
        raise ValueError(f"Invalid mode: {cfg.mode}")
    if len(cfg.tuning_seeds) < 2 or len(cfg.evaluation_seeds) < 2:
        raise ValueError(
            "Tuning and evaluation each require at least two seeds"
        )
    if set(cfg.tuning_seeds) & set(cfg.evaluation_seeds):
        raise ValueError("Tuning and evaluation seeds must be disjoint")
    if not cfg.tasks_to_run or not cfg.algorithms_to_run:
        raise ValueError("At least one task and algorithm are required")

    for task_name in cfg.tasks_to_run:
        if task_name not in TASK_TYPES or task_name not in cfg.tasks:
            raise ValueError(f"Invalid task: {task_name}")
        task_cfg = cfg.tasks[task_name]
        positive_values = (
            task_cfg.iterations,
            task_cfg.horizon,
            task_cfg.samples,
            task_cfg.randomizations,
            task_cfg.knots,
            task_cfg.base_noise,
            task_cfg.temperature,
        )
        if any(float(value) <= 0 for value in positive_values):
            raise ValueError(f"Task values must be positive: {task_name}")

    for algorithm in cfg.algorithms_to_run:
        if algorithm not in ALGORITHM_NAMES or algorithm not in cfg.candidates:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        if not cfg.candidates[algorithm]:
            raise ValueError(f"No candidates for algorithm: {algorithm}")


def run_tuning_and_benchmark(
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Tune every pair and immediately evaluate the selected controller."""
    tuning_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    tuning_path = output_dir / "tuning_results.csv"
    benchmark_path = output_dir / "benchmark_results.csv"
    selection_path = output_dir / "selected_parameters.yaml"

    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        task, state = make_task(task_name, str(cfg.backend))
        selections[task_name] = {}

        for algorithm in cfg.algorithms_to_run:
            best_mean = float("inf")
            best_bundle: tuple[
                int,
                DictConfig,
                SamplingBasedController,
                Callable[[Any, Any], Any],
            ]

            for candidate_index, candidate in enumerate(
                cfg.candidates[algorithm]
            ):
                controller = make_controller(
                    algorithm,
                    task,
                    task_cfg,
                    candidate,
                    int(cfg.domain_seed),
                )
                optimizer = compile_optimizer(
                    controller, state, int(cfg.warmup_seed)
                )
                candidate_rows = run_evaluation(
                    "tune",
                    task_name,
                    algorithm,
                    candidate_index,
                    candidate,
                    controller,
                    optimizer,
                    state,
                    [int(seed) for seed in cfg.tuning_seeds],
                    task_cfg,
                )
                tuning_rows.extend(candidate_rows)
                write_csv(tuning_path, tuning_rows)
                candidate_mean = statistics.mean(
                    float(row["best_cost"]) for row in candidate_rows
                )
                if candidate_index == 0 or candidate_mean < best_mean:
                    best_mean = candidate_mean
                    best_bundle = (
                        candidate_index,
                        candidate,
                        controller,
                        optimizer,
                    )

            candidate_index, candidate, controller, optimizer = best_bundle
            parameters = OmegaConf.to_container(candidate, resolve=True)
            selections[task_name][algorithm] = {
                "candidate": candidate_index,
                "mean_tuning_cost": best_mean,
                "parameters": parameters,
            }
            OmegaConf.save(
                config=OmegaConf.create(selections), f=selection_path
            )

            if cfg.mode == "all":
                evaluation_rows = run_evaluation(
                    "eval",
                    task_name,
                    algorithm,
                    candidate_index,
                    candidate,
                    controller,
                    optimizer,
                    state,
                    [int(seed) for seed in cfg.evaluation_seeds],
                    task_cfg,
                )
                benchmark_rows.extend(evaluation_rows)
                write_csv(benchmark_path, benchmark_rows)

    return tuning_rows, benchmark_rows, selections


def run_selected_benchmark(
    cfg: DictConfig,
    output_dir: Path,
    selections: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate a previously selected parameter set without retuning."""
    rows: list[dict[str, Any]] = []
    result_path = output_dir / "benchmark_results.csv"

    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        task, state = make_task(task_name, str(cfg.backend))
        for algorithm in cfg.algorithms_to_run:
            selection = selections[task_name][algorithm]
            candidate = OmegaConf.create(selection["parameters"])
            controller = make_controller(
                algorithm,
                task,
                task_cfg,
                candidate,
                int(cfg.domain_seed),
            )
            optimizer = compile_optimizer(
                controller, state, int(cfg.warmup_seed)
            )
            evaluation_rows = run_evaluation(
                "eval",
                task_name,
                algorithm,
                int(selection["candidate"]),
                candidate,
                controller,
                optimizer,
                state,
                [int(seed) for seed in cfg.evaluation_seeds],
                task_cfg,
            )
            rows.extend(evaluation_rows)
            write_csv(result_path, rows)
    return rows


@hydra.main(version_base=None, config_path=".", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Run tuning, evaluation, or both for the complete benchmark."""
    validate_config(cfg)
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_parameters.yaml"

    if cfg.mode in {"all", "tune"}:
        _, benchmark_rows, selections = run_tuning_and_benchmark(
            cfg, output_dir
        )
    else:
        if not selection_path.is_file():
            raise FileNotFoundError(f"Missing selections: {selection_path}")
        selections = OmegaConf.to_container(
            OmegaConf.load(selection_path), resolve=True
        )
        if not isinstance(selections, dict):
            raise TypeError("Selected parameters must be a mapping")
        benchmark_rows = run_selected_benchmark(cfg, output_dir, selections)

    if cfg.mode in {"all", "benchmark"}:
        write_report(
            output_dir / "report.md",
            cfg,
            benchmark_rows,
            selections,
        )


if __name__ == "__main__":
    main()
