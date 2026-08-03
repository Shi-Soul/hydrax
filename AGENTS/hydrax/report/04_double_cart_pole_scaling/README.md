# Double Cart Pole Operation-Budget Scaling

## Provenance

- Experiment type: Double Cart Pole receding-horizon operation-budget sweep.
- Pipeline snapshot: Hydrax commit `76da3a4` (`2026-07-31`).
- Backend: JAX on an NVIDIA RTX 4090.
- Methods: CBO, MPPI-CMA, CMA-ES, CEM, Predictive Sampling, and DIAL.
- Seeds: `[0, 1, 2, 3, 4]`.
- Total horizon: $10$ s.
- Planning horizon: $1$ s.
- Knots: $6$.

The operation proxy is $O=I N$, where $I$ is the number of optimizer
iterations and $N$ is the number of samples per iteration. The sweep uses
$I\in\{1,2,4,8,16\}$ and
$O\in\{512,1024,2048,4096,8192\}$. For each method and operation budget, the
selected configuration minimizes mean episode cost over the five seeds.

## Contents

| Path | Description |
|---|---|
| `index.html` | Self-contained result dashboard payload with local figure and media links. |
| `source/` | Exact tracked pipeline and dashboard snapshot from commit `76da3a4`. |
| `grid_results.csv` | 750 episode rows covering six methods, five budgets, five iteration counts, and five seeds. |
| `aggregate_results.csv` | 150 five-seed configuration aggregates. |
| `scaling_results.csv` | 30 best-over-$I$ selections. |
| `baseline_results.csv` | Fixed $I=8$, $N=1024$ baseline for all six methods. |
| `summary.json` | Structured dashboard payload. |
| `report.md` | Baseline and best-allocation tables. |
| `figures/` | 8 publication figures in PDF and PNG formats. |
| `media/` | 30 budget-optimal and 6 fixed-baseline MP4/JPEG pairs. |
| `validation/` | Desktop and mobile dashboard validation screenshots. |

## Figures

- `figures/scaling_curve.pdf`: best-over-$I$ cost versus $O$.
- `figures/optimal_iterations.pdf`: selected $I^*$ versus $O$.
- `figures/optimal_samples.pdf`: selected $N^*$ versus $O$.
- `figures/performance_vs_samples.pdf`: cost sensitivity versus $N$.
- `figures/performance_vs_iterations.pdf`: cost sensitivity versus $I$.
- `figures/optimal_configuration_trajectory.pdf`: selected $(N^*,I^*)$ paths.
- `figures/planning_time_heatmap.pdf`: actual planning time over the $(I,O)$ grid.
- `figures/optimal_configuration_planning_time.pdf`: actual time of each cost-optimal allocation.

## Fixed Baseline Summary

At $I=8$ and $N=1024$, Predictive Sampling has the lowest mean episode cost at
$1074.2 \pm 82.1$, followed by MPPI-CMA at $1087.57 \pm 124$. See `report.md`
for every method and every selected operation-budget allocation.

Planning time is the cumulative GPU optimizer time over a full episode; JIT
compilation and simulation execution are excluded. The archived heatmap makes
the difference between the proxy $O$ and measured planning time explicit.
