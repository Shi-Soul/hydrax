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
ALGORITHMS = ("cbo", "mppi_cma", "cmaes", "cem", "ps", "dial")
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


def _make_task(task_name: str, backend: str) -> tuple[Task, Any]:
    task = TASK_TYPES[task_name](impl=backend)
    state = task.make_data()
    if task_name == "humanoid_standup":
        qpos = jnp.asarray(task.mj_model.keyframe("stand").qpos)
        state = state.replace(
            qpos=qpos.at[3:7].set(jnp.array([0.7, 0.0, -0.7, 0.0]))
        )
    elif task_name == "pusht":
        state = state.replace(qpos=jnp.array([0.1, 0.1, 1.3, 0.0, 0.0]))
    elif task_name == "bugtrap":
        state = state.replace(
            qpos=state.qpos.at[:2].set(jnp.array([-0.15, 0.0])),
            mocap_pos=state.mocap_pos.at[0].set(jnp.array([0.25, 0.0, 0.01])),
        )
    return task, state


def _make_controller(
    algorithm: str,
    task: Task,
    task_cfg: DictConfig,
    candidate: DictConfig,
    domain_seed: int,
) -> SamplingBasedController:
    base_noise = float(task_cfg.base_noise)
    shared = {
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

    if algorithm == "cbo":
        return CBO(
            initial_noise_level=base_noise,
            temperature=(
                float(task_cfg.temperature) * float(candidate.temperature_scale)
            ),
            consensus_weight=float(candidate.consensus_weight),
            noise_weight=float(candidate.noise_weight),
            step_size=float(candidate.step_size),
            **shared,
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
            **shared,
        )
    if algorithm == "cmaes":
        num_dims = task.model.nu * int(task_cfg.knots)
        strategy = CMA_ES(
            population_size=int(task_cfg.samples),
            solution=jnp.zeros(num_dims),
        )
        es_params = strategy.default_params.replace(
            std_init=base_noise,
            std_min=base_noise * float(candidate.std_min_ratio),
            std_max=base_noise * float(candidate.std_max_ratio),
            c_mean=float(candidate.c_mean),
        )
        return Evosax(optimizer=CMA_ES, es_params=es_params, **shared)
    if algorithm == "cem":
        num_elites = max(
            2,
            round(int(task_cfg.samples) * float(candidate.elite_fraction)),
        )
        return CEM(
            num_elites=num_elites,
            sigma_start=base_noise,
            sigma_min=base_noise * float(candidate.sigma_min_ratio),
            explore_fraction=float(candidate.explore_fraction),
            **shared,
        )
    if algorithm == "ps":
        return PredictiveSampling(noise_level=base_noise, **shared)
    if algorithm == "dial":
        return DIAL(
            noise_level=base_noise,
            beta_opt_iter=float(candidate.beta_opt_iter),
            beta_horizon=float(candidate.beta_horizon),
            temperature=(
                float(task_cfg.temperature) * float(candidate.temperature_scale)
            ),
            **shared,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _compile(
    controller: SamplingBasedController, state: Any, seed: int
) -> Callable[[Any, Any], Any]:
    optimizer = jax.jit(controller.optimize)
    params = controller.init_params(initial_knots=None, seed=seed)
    _, rollouts = optimizer(state, params)
    jax.block_until_ready(rollouts.costs)
    return optimizer


def _evaluate(
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
    parameters = json.dumps(
        OmegaConf.to_container(candidate, resolve=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    rows = []
    for seed in seeds:
        params = controller.init_params(initial_knots=None, seed=seed)
        jax.block_until_ready(params)
        start = time.perf_counter()
        _, rollouts = optimizer(state, params)
        jax.block_until_ready(rollouts.costs)
        elapsed = time.perf_counter() - start
        costs = jnp.sum(rollouts.costs, axis=1)
        best_cost = float(
            jnp.min(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
        )
        rows.append(
            {
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
                "parameters": parameters,
            }
        )
        print(
            f"{phase:5s} {task_name:20s} {algorithm:9s} "
            f"candidate={candidate_index} seed={seed} "
            f"cost={best_cost:.6g} time={elapsed:.4f}s",
            flush=True,
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_report(
    path: Path,
    cfg: DictConfig,
    rows: list[dict[str, Any]],
    selections: dict[str, Any],
) -> None:
    lines = [
        "# Hydrax Open-Loop Planning Benchmark",
        "",
        "Each run optimizes once from the fixed example initial state. JIT "
        "compilation, setup, metric reduction, and output are excluded from "
        "planning time. The reported best cost is the minimum total rollout "
        "cost in the final internal optimization iteration.",
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
        for algorithm in cfg.algorithms_to_run:
            selected = [
                row
                for row in rows
                if row["task"] == task_name and row["algorithm"] == algorithm
            ]
            costs = [float(row["best_cost"]) for row in selected]
            times = [1000.0 * float(row["time_seconds"]) for row in selected]
            row = selected[0]
            params = json.dumps(
                selections[task_name][algorithm]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.append(
                f"| {task_name} | {algorithm} | "
                f"${statistics.mean(costs):.6g} "
                f"\\pm {statistics.stdev(costs):.3g}$ | "
                f"${statistics.mean(times):.3f} "
                f"\\pm {statistics.stdev(times):.3f}$ | "
                f"{row['iterations']} | {float(row['horizon']):.3g} | "
                f"{row['samples']} | {row['randomizations']} | "
                f"{row['knots']} | {row['spline']} | "
                f"{float(row['frequency']):.3g} | "
                f"{row['max_episode_steps']} | "
                f"{float(row['base_noise']):.3g} | `{params}` |"
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


def _validate(cfg: DictConfig) -> None:
    if len(cfg.tuning_seeds) < 2 or len(cfg.evaluation_seeds) < 2:
        raise ValueError("Tuning and evaluation each require two seeds")
    if set(cfg.tuning_seeds) & set(cfg.evaluation_seeds):
        raise ValueError("Tuning and evaluation seeds must be disjoint")
    if set(cfg.tasks_to_run) - set(TASK_TYPES):
        raise ValueError("Unknown task in tasks_to_run")
    if set(cfg.algorithms_to_run) - set(ALGORITHMS):
        raise ValueError("Unknown algorithm in algorithms_to_run")
    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        values = (
            task_cfg.iterations,
            task_cfg.horizon,
            task_cfg.samples,
            task_cfg.randomizations,
            task_cfg.knots,
            task_cfg.base_noise,
            task_cfg.temperature,
        )
        if any(float(value) <= 0 for value in values):
            raise ValueError(f"Non-positive task value: {task_name}")
    if any(not cfg.candidates[name] for name in cfg.algorithms_to_run):
        raise ValueError("Every algorithm requires a candidate")


@hydra.main(version_base=None, config_path=".", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Tune and evaluate all configured open-loop planning algorithms."""
    _validate(cfg)
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}

    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        task, state = _make_task(task_name, str(cfg.backend))
        selections[task_name] = {}
        for algorithm in cfg.algorithms_to_run:
            best_score = float("inf")
            best: tuple[
                int,
                DictConfig,
                SamplingBasedController,
                Callable[[Any, Any], Any],
            ]
            for index, candidate in enumerate(cfg.candidates[algorithm]):
                controller = _make_controller(
                    algorithm,
                    task,
                    task_cfg,
                    candidate,
                    int(cfg.domain_seed),
                )
                optimizer = _compile(controller, state, int(cfg.warmup_seed))
                rows = _evaluate(
                    "tune",
                    task_name,
                    algorithm,
                    index,
                    candidate,
                    controller,
                    optimizer,
                    state,
                    [int(seed) for seed in cfg.tuning_seeds],
                    task_cfg,
                )
                tuning_rows.extend(rows)
                _write_csv(output_dir / "tuning_results.csv", tuning_rows)
                score = statistics.mean(float(row["best_cost"]) for row in rows)
                if index == 0 or score < best_score:
                    best_score = score
                    best = index, candidate, controller, optimizer

            index, candidate, controller, optimizer = best
            parameters = OmegaConf.to_container(candidate, resolve=True)
            selections[task_name][algorithm] = {
                "candidate": index,
                "mean_tuning_cost": best_score,
                "parameters": parameters,
            }
            OmegaConf.save(
                config=OmegaConf.create(selections),
                f=output_dir / "selected_parameters.yaml",
            )
            rows = _evaluate(
                "eval",
                task_name,
                algorithm,
                index,
                candidate,
                controller,
                optimizer,
                state,
                [int(seed) for seed in cfg.evaluation_seeds],
                task_cfg,
            )
            benchmark_rows.extend(rows)
            _write_csv(output_dir / "benchmark_results.csv", benchmark_rows)

    _write_report(output_dir / "report.md", cfg, benchmark_rows, selections)


if __name__ == "__main__":
    main()
