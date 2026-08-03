# Legacy Open-Loop Benchmark

## Provenance

- Experiment type: one-shot open-loop trajectory optimization.
- Source snapshot: Hydrax commit `04a236e` (`2026-07-30`).
- Backend: JAX on `cuda:0`.
- Tasks: Cart Pole, Double Cart Pole, Humanoid Standup, Push-T, and Bugtrap.
- Methods: CBO, MPPI-CMA, CMA-ES, CEM, Predictive Sampling, and DIAL.
- Tuning seeds: `[100, 101]`.
- Evaluation seeds: `[0, 1, 2, 3, 4]`.

The final best cost is the minimum total rollout cost in the last internal
optimization iteration. Lower cost is better. JIT compilation, setup, metric
reduction, and output are excluded from planning time.

## Contents

| Path | Description |
|---|---|
| `source/` | Exact tracked source snapshot from commit `04a236e`. |
| `results/tuning_results.csv` | 160 candidate-by-seed tuning rows. |
| `results/benchmark_results.csv` | 150 evaluation rows covering five tasks, six methods, and five seeds. |
| `results/selected_parameters.yaml` | Parameters selected by the tuning run. |
| `results/report.md` | Full aggregate report with mean and sample standard deviation. |

## Best Evaluation Mean

| Task | Method | Final best cost |
|---|---|---:|
| Cart Pole | DIAL | $8.36805 \pm 0.0173$ |
| Double Cart Pole | CBO | $119.731 \pm 1.33$ |
| Humanoid Standup | DIAL | $22.1286 \pm 0.409$ |
| Push-T | DIAL | $0.497282 \pm 0.0336$ |
| Bugtrap | DIAL | $0.815402 \pm 0.0173$ |

## Reproduction Note

The source files preserve provenance rather than forming a relocated runnable
package. Their Hydra paths target `examples/to/`; restore the snapshot at the
recorded commit before rerunning it. The archived CSV and report are the
authoritative outputs for this entry.
