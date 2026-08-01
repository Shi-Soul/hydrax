"""Run one fixed Hydrax receding-horizon benchmark episode.

Batch sweeps, aggregation, rendering, and dashboard generation live in
``examples/to/benchmarking``.  This module owns only the task/controller
construction and the single-episode execution path shared by those tools.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

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
STATE_FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat", "time")
DEVICE = jax.devices()[0]


@dataclass(frozen=True)
class EpisodeResult:
    """Measured output from one controller/seed episode."""

    episode_cost: float
    planning_time_seconds: float
    trajectory: dict[str, Any]


def make_task(task_name: str, backend: str) -> tuple[Task, Any]:
    """Construct a task and its canonical initial state."""
    if task_name not in TASK_TYPES:
        raise ValueError(f"Unknown task: {task_name}")
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


def make_controller(
    algorithm: str,
    task: Task,
    task_cfg: Mapping[str, Any],
    candidate: Mapping[str, Any],
    domain_seed: int,
    samples: int,
    iterations: int,
    time_shift_mode: str = "legacy",
) -> SamplingBasedController:
    """Construct one controller with fixed algorithm-specific parameters."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    base_noise = float(task_cfg["base_noise"])
    shared = {
        "task": task,
        "num_samples": samples,
        "num_randomizations": int(task_cfg["randomizations"]),
        "risk_strategy": AverageCost(),
        "seed": domain_seed,
        "plan_horizon": float(task_cfg["segment_horizon"]),
        "spline_type": str(task_cfg["spline"]),
        "num_knots": int(task_cfg["knots"]),
        "iterations": iterations,
    }

    if algorithm == "cbo":
        return CBO(
            initial_noise_level=base_noise,
            temperature=float(task_cfg["temperature"])
            * float(candidate["temperature_scale"]),
            consensus_weight=float(candidate["consensus_weight"]),
            noise_weight=float(candidate["noise_weight"]),
            step_size=float(candidate["step_size"]),
            time_shift_mode=time_shift_mode,
            **shared,
        )
    if algorithm == "mppi_cma":
        return MppiCma(
            initial_noise_level=base_noise,
            temperature=float(task_cfg["temperature"])
            * float(candidate["temperature_scale"]),
            minimum_noise_level=base_noise
            * float(candidate["minimum_noise_ratio"]),
            covariance_adaptation_rate=float(candidate["adaptation_rate"]),
            **shared,
        )
    if algorithm == "cmaes":
        num_dims = task.model.nu * int(task_cfg["knots"])
        strategy = CMA_ES(
            population_size=samples,
            solution=jnp.zeros(num_dims),
        )
        es_params = strategy.default_params.replace(
            std_init=base_noise,
            std_min=base_noise * float(candidate["std_min_ratio"]),
            std_max=base_noise * float(candidate["std_max_ratio"]),
            c_mean=float(candidate["c_mean"]),
        )
        return Evosax(optimizer=CMA_ES, es_params=es_params, **shared)
    if algorithm == "cem":
        num_elites = max(2, round(samples * float(candidate["elite_fraction"])))
        return CEM(
            num_elites=num_elites,
            sigma_start=base_noise,
            sigma_min=base_noise * float(candidate["sigma_min_ratio"]),
            explore_fraction=float(candidate["explore_fraction"]),
            time_shift_mode=time_shift_mode,
            **shared,
        )
    if algorithm == "ps":
        return PredictiveSampling(noise_level=base_noise, **shared)
    return DIAL(
        noise_level=base_noise,
        beta_opt_iter=float(candidate["beta_opt_iter"]),
        beta_horizon=float(candidate["beta_horizon"]),
        temperature=float(task_cfg["temperature"])
        * float(candidate["temperature_scale"]),
        **shared,
    )


def execution_steps(
    controller: SamplingBasedController, execution_fraction: float
) -> int:
    """Convert the execution fraction to an integral number of control steps."""
    steps = int(round(controller.ctrl_steps * execution_fraction))
    if not 0 < steps <= controller.ctrl_steps:
        raise ValueError("Execution fraction produces invalid control steps")
    return steps


def compile_controller(
    controller: SamplingBasedController,
    state: Any,
    warmup_seed: int,
    executed_steps: int,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """JIT compile optimization and rollout execution for one shape."""
    optimizer = jax.jit(controller.optimize)
    initial_knots = jnp.zeros((controller.num_knots, state.ctrl.shape[0]))
    params = jax.device_put(
        controller.init_params(initial_knots, warmup_seed), DEVICE
    )
    params, rollouts = optimizer(state, params)
    jax.block_until_ready((params, rollouts))
    executor = jax.jit(controller.eval_rollouts)
    states, _ = executor(
        controller.task.model,
        state,
        rollouts.controls[:1, :executed_steps],
        rollouts.knots[:1],
    )
    jax.block_until_ready(states.qpos)
    params, rollouts = optimizer(state, params)
    jax.block_until_ready((params, rollouts))
    return optimizer, executor


def run_episode(
    controller: SamplingBasedController,
    compiled: tuple[Callable[..., Any], Callable[..., Any]],
    initial_state: Any,
    seed: int,
    total_horizon: float,
    executed_steps: int,
    record_trajectory: bool,
) -> EpisodeResult:
    """Run one complete receding-horizon episode."""
    optimizer, executor = compiled
    task = controller.task
    total_steps = int(round(total_horizon / controller.dt))
    segments = -(-total_steps // executed_steps)
    initial_knots = jnp.zeros((controller.num_knots, task.model.nu))
    params = jax.device_put(controller.init_params(initial_knots, seed), DEVICE)
    state, completed_steps = initial_state, 0
    episode_cost, planning_time = jnp.array(0.0), 0.0
    trajectory_parts: dict[str, list[Any]] = {name: [] for name in STATE_FIELDS}

    for _ in range(segments):
        jax.block_until_ready((state, params))
        start = time.perf_counter()
        params, rollouts = optimizer(state, params)
        jax.block_until_ready((params, rollouts))
        planning_time += time.perf_counter() - start
        rollout_costs = jnp.sum(rollouts.costs, axis=1)
        finite_costs = jnp.where(
            jnp.isfinite(rollout_costs), rollout_costs, jnp.inf
        )
        best_index = int(jnp.argmin(finite_costs))
        step_count = min(executed_steps, total_steps - completed_steps)
        states, executed = executor(
            task.model,
            state,
            rollouts.controls[best_index : best_index + 1, :step_count],
            rollouts.knots[best_index : best_index + 1],
        )
        jax.block_until_ready((states.qpos, executed.costs))
        episode_cost += jnp.sum(executed.costs[0])
        if record_trajectory:
            for name in STATE_FIELDS:
                trajectory_parts[name].append(getattr(states, name)[0])
        state = jax.tree.map(lambda value: value[0, -1], states)
        completed_steps += step_count

    episode_cost = jnp.where(jnp.isfinite(episode_cost), episode_cost, jnp.inf)
    jax.block_until_ready(episode_cost)
    trajectory = (
        {
            name: jnp.concatenate(parts)
            for name, parts in trajectory_parts.items()
        }
        if record_trajectory
        else {}
    )
    return EpisodeResult(float(episode_cost), planning_time, trajectory)


def _write_result(path: Path, result: EpisodeResult) -> None:
    """Write one CLI result atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "episode_cost": result.episode_cost,
                "planning_time_seconds": result.planning_time_seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_single(cfg: DictConfig) -> None:
    """Validate the minimal single-run Hydra configuration."""
    if cfg.task not in TASK_TYPES:
        raise ValueError(f"Unknown task: {cfg.task}")
    if cfg.algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
    if int(cfg.samples) < 1 or int(cfg.iterations) < 1:
        raise ValueError("samples and iterations must be positive")
    if not 0.0 < float(cfg.execution_fraction) <= 1.0:
        raise ValueError("execution_fraction must be in (0, 1]")


@hydra.main(version_base=None, config_path=".", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Run one configured episode and write its scalar result."""
    _validate_single(cfg)
    task, state = make_task(str(cfg.task), str(cfg.backend))
    selections = OmegaConf.load(Path(str(cfg.selected_parameters)).resolve())
    selection = selections[cfg.task][cfg.algorithm]
    task_cfg = cfg.task_config
    controller = make_controller(
        str(cfg.algorithm),
        task,
        task_cfg,
        selection.parameters,
        int(cfg.domain_seed),
        int(cfg.samples),
        int(cfg.iterations),
    )
    executed_steps = execution_steps(controller, float(cfg.execution_fraction))
    compiled = compile_controller(
        controller, state, int(cfg.warmup_seed), executed_steps
    )
    result = run_episode(
        controller,
        compiled,
        state,
        int(cfg.seed),
        float(task_cfg["total_horizon"]),
        executed_steps,
        False,
    )
    _write_result(Path(str(cfg.output_path)), result)
    print(
        f"{cfg.task} {cfg.algorithm} seed={cfg.seed} "
        f"cost={result.episode_cost:.6g} "
        f"planning_time={result.planning_time_seconds:.4f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
