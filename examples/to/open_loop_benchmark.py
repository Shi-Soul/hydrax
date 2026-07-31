import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import hydra
import jax
import jax.numpy as jnp
import mujoco
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
CAMERAS = {
    "cart_pole": "fixed",
    "double_cart_pole": "fixed",
    "humanoid_standup": ((0.0, 0.0, 0.65), 2.8, 140.0, -18.0),
    "pusht": ((0.0, 0.0, 0.0), 0.75, 90.0, -85.0),
    "bugtrap": "top_view",
}
STATE_FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat", "time")
BUDGET_FIELDS = """iterations segment_horizon total_horizon samples
randomizations knots spline""".split()
POSITIVE_FIELDS = (*BUDGET_FIELDS[:-1], "base_noise", "temperature")
EXECUTORS: dict[int, Callable[..., Any]] = {}
REPLAY_COSTS: dict[tuple[str, str], float] = {}
DEVICE = jax.devices()[0]


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
        "plan_horizon": float(task_cfg.segment_horizon),
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
) -> tuple[Callable[[Any, Any], Any], Callable[..., Any]]:
    optimizer = jax.jit(controller.optimize)
    initial_knots = jnp.zeros((controller.num_knots, state.ctrl.shape[0]))
    params = jax.device_put(controller.init_params(initial_knots, seed), DEVICE)
    params, rollouts = optimizer(state, params)
    jax.block_until_ready((params, rollouts))
    executor = EXECUTORS.setdefault(
        id(controller.task), jax.jit(controller.eval_rollouts)
    )
    states, _ = executor(
        controller.task.model, state, rollouts.controls[:1], rollouts.knots[:1]
    )
    jax.block_until_ready(states.qpos)
    state = jax.tree.map(lambda value: value[0, -1], states)
    params = jax.device_put(controller.init_params(initial_knots, seed), DEVICE)
    params, rollouts = optimizer(state, params)
    jax.block_until_ready((params, rollouts))
    return optimizer, executor


def _run_episode(
    controller: SamplingBasedController,
    compiled: tuple[Callable[[Any, Any], Any], Callable[..., Any]],
    initial_state: Any,
    seed: int,
    total_horizon: float,
    record_trajectory: bool,
) -> tuple[float, float, dict[str, Any]]:
    task, (optimizer, executor) = controller.task, compiled
    total_steps = int(round(total_horizon / controller.dt))
    segments = -(-total_steps // controller.ctrl_steps)
    initial_knots = jnp.zeros((controller.num_knots, task.model.nu))
    state, executed_steps = initial_state, 0
    episode_cost, planning_time = jnp.array(0.0), 0.0
    trajectory_parts: dict[str, list[Any]] = {name: [] for name in STATE_FIELDS}
    for segment_seed in range(seed * segments, (seed + 1) * segments):
        params = controller.init_params(initial_knots, segment_seed)
        params = jax.device_put(params, DEVICE)
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
        step_count = min(controller.ctrl_steps, total_steps - executed_steps)
        states, executed = executor(
            task.model,
            state,
            rollouts.controls[best_index : best_index + 1, :step_count],
            rollouts.knots[best_index : best_index + 1],
        )
        jax.block_until_ready((states.qpos, executed.costs))
        episode_cost += jnp.sum(executed.costs[0])
        selected_knots = rollouts.knots[best_index]
        initial_knots = jnp.repeat(
            selected_knots[-1:], controller.num_knots, axis=0
        )
        if record_trajectory:
            for name in STATE_FIELDS:
                trajectory_parts[name].append(getattr(states, name)[0])
        state = jax.tree.map(lambda value: value[0, -1], states)
        executed_steps += step_count
    episode_cost = jnp.where(jnp.isfinite(episode_cost), episode_cost, jnp.inf)
    jax.block_until_ready(episode_cost)
    trajectory = {
        name: jnp.concatenate(parts) for name, parts in trajectory_parts.items()
    } if record_trajectory else {}
    return float(episode_cost), planning_time, trajectory


def _evaluate(
    phase: str,
    task_name: str,
    algorithm: str,
    candidate_index: int,
    candidate: DictConfig,
    controller: SamplingBasedController,
    compiled: tuple[Callable[[Any, Any], Any], Callable[..., Any]],
    state: Any,
    seeds: list[int],
    task_cfg: DictConfig,
) -> list[dict[str, Any]]:
    parameters = json.dumps(
        OmegaConf.to_container(candidate, resolve=True), sort_keys=True,
        separators=(",", ":")
    )
    rows = []
    for seed in seeds:
        episode_cost, elapsed, _ = _run_episode(
            controller=controller,
            compiled=compiled,
            initial_state=state,
            seed=seed,
            total_horizon=float(task_cfg.total_horizon),
            record_trajectory=False,
        )
        total_steps = int(round(float(task_cfg.total_horizon) / controller.dt))
        segments = -(-total_steps // controller.ctrl_steps)
        rows.append(
            {
                "task": task_name,
                "algorithm": algorithm,
                "candidate": candidate_index,
                "seed": seed,
                "episode_cost": episode_cost,
                "planning_time_seconds": elapsed,
                "iterations": controller.iterations,
                "segment_horizon": controller.plan_horizon,
                "total_horizon": float(task_cfg.total_horizon),
                "segments": segments,
                "samples": int(task_cfg.samples),
                "randomizations": controller.num_randomizations,
                "knots": controller.num_knots,
                "spline": controller.spline_type,
                "frequency": 1.0 / controller.dt,
                "max_episode_steps": total_steps,
                "base_noise": float(task_cfg.base_noise),
                "parameters": parameters,
            }
        )
        print(
            f"{phase:5s} {task_name:20s} {algorithm:9s} "
            f"candidate={candidate_index} seed={seed} "
            f"cost={episode_cost:.6g} time={elapsed:.4f}s",
            flush=True,
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _select_rows(
    rows: list[dict[str, Any]], task_name: str, algorithm: str
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["task"] == task_name and row["algorithm"] == algorithm
    ]


def _render_video(
    path: Path,
    task_name: str,
    task: Task,
    initial_state: Any,
    states: dict[str, Any],
    total_horizon: float,
    render_cfg: DictConfig,
) -> None:
    values = {
        name: jax.device_get(
            jnp.concatenate((getattr(initial_state, name)[None], states[name]))
        )
        for name in STATE_FIELDS
    }
    frame_count = round(total_horizon * float(render_cfg.fps))
    indices = [
        round(index * (len(values["time"]) - 1) / frame_count)
        for index in range(frame_count)
    ]
    width, height = int(render_cfg.width), int(render_cfg.height)
    task.mj_model.vis.global_.offwidth = width
    task.mj_model.vis.global_.offheight = height
    data, camera = mujoco.MjData(task.mj_model), mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera_cfg = CAMERAS[task_name]
    if isinstance(camera_cfg, str):
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = task.mj_model.camera(camera_cfg).id
    else:
        camera.lookat[:] = camera_cfg[0]
        camera.distance, camera.azimuth, camera.elevation = camera_cfg[1:]
    renderer = mujoco.Renderer(task.mj_model, height=height, width=width)
    frames = []
    for index in indices:
        data.qpos[:] = values["qpos"][index]
        data.qvel[:] = values["qvel"][index]
        data.mocap_pos[:] = values["mocap_pos"][index]
        data.mocap_quat[:] = values["mocap_quat"][index]
        data.time = values["time"][index]
        mujoco.mj_forward(task.mj_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().tobytes())
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    video_tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    poster = path.with_suffix(".jpg")
    poster_tmp = poster.with_name(f"{poster.stem}.tmp{poster.suffix}")
    encode = (
        f"ffmpeg -y -f rawvideo -s {width}x{height} -pix_fmt rgb24 "
        f"-r {render_cfg.fps} -i - -an -c:v libx264 -crf 22 -preset medium "
        "-movflags +faststart -pix_fmt yuv420p -loglevel error"
    ).split()
    subprocess.run(
        [*encode, str(video_tmp)], input=b"".join(frames), check=True
    )
    poster_command = (
        f"ffmpeg -y -f rawvideo -s {width}x{height} -pix_fmt rgb24 "
        "-i - -frames:v 1 -q:v 2 -loglevel error"
    ).split()
    subprocess.run(
        [*poster_command, str(poster_tmp)], input=frames[-1], check=True
    )
    video_tmp.replace(path)
    poster_tmp.replace(poster)


def _write_browser_data(
    path: Path, cfg: DictConfig, rows: list[dict[str, str]]
) -> None:
    tasks = {}
    for task_name in cfg.tasks_to_run:
        algorithms = {}
        for algorithm in cfg.algorithms_to_run:
            selected = _select_rows(rows, task_name, algorithm)
            best = min(selected, key=lambda row: float(row["episode_cost"]))
            costs = [float(row["episode_cost"]) for row in selected]
            times = [
                1000.0 * float(row["planning_time_seconds"]) for row in selected
            ]
            algorithms[algorithm] = {
                "mean_cost": float(jnp.mean(jnp.asarray(costs))),
                "std_cost": float(jnp.std(jnp.asarray(costs), ddof=1)),
                "mean_time_ms": statistics.mean(times),
                "std_time_ms": statistics.stdev(times),
                "best_seed": int(best["seed"]),
                "best_seed_cost": REPLAY_COSTS[task_name, algorithm],
                "candidate": int(best["candidate"]),
                "parameters": json.loads(best["parameters"]),
                "video": f"results/media/{task_name}__{algorithm}.mp4",
                "poster": f"results/media/{task_name}__{algorithm}.jpg",
            }
        task_cfg = cfg.tasks[task_name]
        tasks[task_name] = {
            "budget": {name: task_cfg[name] for name in BUDGET_FIELDS}
            | {"segments": int(selected[0]["segments"])},
            "algorithms": algorithms,
        }
    payload = {
        "backend": str(cfg.backend),
        "evaluation_seeds": list(cfg.evaluation_seeds),
        "tasks": tasks,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "window.HYDRAX_RESULTS = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render_results(cfg: DictConfig, output_dir: Path) -> None:
    results_path = output_dir / "benchmark_results.csv"
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row["task"] in cfg.tasks_to_run]
    expected = len(cfg.tasks_to_run) * len(cfg.algorithms_to_run)
    if len(rows) != expected * len(cfg.evaluation_seeds):
        raise ValueError("Benchmark results are incomplete")
    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        task, state = _make_task(task_name, str(cfg.backend))
        for algorithm in cfg.algorithms_to_run:
            selected = _select_rows(rows, task_name, algorithm)
            best = min(selected, key=lambda row: float(row["episode_cost"]))
            candidate = OmegaConf.create(json.loads(best["parameters"]))
            controller = _make_controller(
                algorithm, task, task_cfg, candidate, int(cfg.domain_seed)
            )
            compiled = _compile(controller, state, int(cfg.warmup_seed))
            cost, _, trajectory = _run_episode(
                controller, compiled, state, int(best["seed"]),
                float(task_cfg.total_horizon), True
            )
            REPLAY_COSTS[task_name, algorithm] = cost
            path = output_dir / "media" / f"{task_name}__{algorithm}.mp4"
            _render_video(
                path, task_name, task, state, trajectory,
                float(task_cfg.total_horizon), cfg.render
            )
            print(
                f"render {task_name:20s} {algorithm:9s} "
                f"seed={best['seed']} cost={cost:.6g}",
                flush=True,
            )
    _write_browser_data(output_dir / "browser_data.js", cfg, rows)


def _write_report(
    path: Path,
    cfg: DictConfig,
    rows: list[dict[str, Any]],
    selections: dict[str, Any],
) -> None:
    lines = dedent(f"""\
        # Hydrax Segmented Offline Planning Benchmark

        Each run executes the best segment, then replans from its final state.

        - Backend: `{cfg.backend}`; device: `{jax.devices()[0]}`
        - Evaluation seeds: `{list(cfg.evaluation_seeds)}`
        - Planning time sums optimizer calls and excludes JIT and execution.
        - Tuning and evaluation use disjoint seeds and shared task base noise.
        - Episode cost sums every segment cost, including each terminal cost:
          $J = \\sum_{{k=1}}^{{K}}[\\sum_{{t=1}}^{{H_k}}\\Delta t\\,
          \\ell(x_{{k,t}},u_{{k,t}})+\\phi(x_{{k,H_k}})]$.
        """).splitlines()
    columns = (
        "Task Algorithm Episode-cost Planning-time-(ms) Iterations/segment "
        "Segment-horizon Total-horizon Segments Samples/iter Randomizations "
        "Knots Spline Frequency Steps Base-noise Selected-parameters"
    ).split()
    align = (
        "--- --- ---: ---: ---: ---: ---: ---: ---: ---: ---: --- "
        "---: ---: ---: ---"
    ).split()
    lines += [
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(align) + "|"]
    for task_name in cfg.tasks_to_run:
        for algorithm in cfg.algorithms_to_run:
            selected = _select_rows(rows, task_name, algorithm)
            costs = [float(row["episode_cost"]) for row in selected]
            times = [
                1000.0 * float(row["planning_time_seconds"]) for row in selected
            ]
            row = selected[0]
            params = json.dumps(
                selections[task_name][algorithm]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.append(
                f"| {task_name} | {algorithm} | "
                f"${jnp.mean(jnp.asarray(costs)):.6g} "
                f"\\pm {jnp.std(jnp.asarray(costs), ddof=1):.3g}$ | "
                f"${statistics.mean(times):.3f} "
                f"\\pm {statistics.stdev(times):.3f}$ | "
                f"{row['iterations']} | {float(row['segment_horizon']):.3g} | "
                f"{float(row['total_horizon']):.3g} | {row['segments']} | "
                f"{row['samples']} | {row['randomizations']} | "
                f"{row['knots']} | {row['spline']} | "
                f"{float(row['frequency']):.3g} | "
                f"{row['max_episode_steps']} | "
                f"{float(row['base_noise']):.3g} | `{params}` |"
            )
    bad, good = "$inf \\pm nan$", "$\\infty \\pm \\mathrm{N/A}$"
    report = "\n".join(lines).replace(bad, good)
    path.write_text(report, encoding="utf-8")


def _validate(cfg: DictConfig) -> None:
    if cfg.action not in ("benchmark", "render"):
        raise ValueError("action must be benchmark or render")
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
        values = (task_cfg[name] for name in POSITIVE_FIELDS)
        if any(float(value) <= 0 for value in values):
            raise ValueError(f"Non-positive task value: {task_name}")
    if any(not cfg.candidates[name] for name in cfg.algorithms_to_run):
        raise ValueError("Every algorithm requires a candidate")
    if min(cfg.render.width, cfg.render.height, cfg.render.fps) <= 0:
        raise ValueError("Render dimensions and fps must be positive")


@hydra.main(version_base=None, config_path=".", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Tune and evaluate all configured segmented planning algorithms."""
    _validate(cfg)
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.action == "render":
        _render_results(cfg, output_dir)
        return
    tuning_rows, benchmark_rows, selections = [], [], {}
    for task_name in cfg.tasks_to_run:
        task_cfg = cfg.tasks[task_name]
        task, state = _make_task(task_name, str(cfg.backend))
        selections[task_name] = {}
        for algorithm in cfg.algorithms_to_run:
            best_score = float("inf")
            best: tuple[int, DictConfig, SamplingBasedController, Any]
            for index, candidate in enumerate(cfg.candidates[algorithm]):
                controller = _make_controller(
                    algorithm, task, task_cfg, candidate, int(cfg.domain_seed)
                )
                compiled = _compile(controller, state, int(cfg.warmup_seed))
                rows = _evaluate(
                    "tune", task_name, algorithm, index, candidate, controller,
                    compiled, state,
                    [int(seed) for seed in cfg.tuning_seeds],
                    task_cfg
                )
                tuning_rows.extend(rows)
                _write_csv(output_dir / "tuning_results.csv", tuning_rows)
                score = statistics.mean(
                    float(row["episode_cost"]) for row in rows
                )
                if index == 0 or score < best_score:
                    best_score = score
                    best = index, candidate, controller, compiled
            index, candidate, controller, compiled = best
            selections[task_name][algorithm] = {
                "candidate": index,
                "mean_tuning_cost": best_score,
                "parameters": OmegaConf.to_container(candidate, resolve=True),
            }
            OmegaConf.save(
                config=OmegaConf.create(selections),
                f=output_dir / "selected_parameters.yaml",
            )
            rows = _evaluate(
                "eval", task_name, algorithm, index, candidate, controller,
                compiled, state,
                [int(seed) for seed in cfg.evaluation_seeds],
                task_cfg
            )
            benchmark_rows.extend(rows)
            _write_csv(output_dir / "benchmark_results.csv", benchmark_rows)
    _write_report(output_dir / "report.md", cfg, benchmark_rows, selections)


if __name__ == "__main__":
    main()
