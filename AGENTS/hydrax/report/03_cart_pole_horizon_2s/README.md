# Cart Pole Two-Second Receding-Horizon Rerun

## Provenance

- Experiment type: Cart Pole receding-horizon planning.
- Source snapshot: Hydrax commit `96ff79a` (`2026-07-31`).
- Planning horizon: $2$ s.
- Execution horizon: $1$ s.
- Total horizon: $2$ s.
- Methods: CBO, MPPI-CMA, CMA-ES, CEM, Predictive Sampling, and DIAL.
- Evaluation seeds: `[0, 1, 2, 3, 4]`.

Each method uses 10 optimizer iterations, 128 samples per iteration, 4 cubic
spline knots, and one randomization. Lower episode cost is better.

## Contents

| Path | Description |
|---|---|
| `index.html` | Single-task dashboard reconstructed with the historical dashboard source. |
| `source/` | Exact tracked source snapshot from commit `96ff79a`. |
| `results/benchmark_results.csv` | 30 rows covering six methods and five seeds. |
| `results/browser_data.js` | Aggregate statistics consumed by the dashboard. |
| `results/report.md` | Full method table with cost and planning-time statistics. |
| `results/media/` | 6 MP4 replays and 6 JPEG posters. |

## Best Five-Seed Mean

CBO has the lowest mean episode cost at $37.9309 \pm 2.55$. The full method
comparison and selected parameters are recorded in `results/report.md`.

This rerun is also represented in the later five-task dashboard, but it is
kept as a separate entry because it is the only receding-horizon subset in the
workspace with its complete per-seed CSV.
