# Legacy Five-Task Receding-Horizon Dashboard

## Provenance

- Experiment type: receding-horizon planning with half of each planned segment
  executed before replanning.
- Source snapshot: Hydrax commit `96ff79a` (`2026-07-31`).
- Backend: JAX.
- Tasks: Cart Pole, Double Cart Pole, Humanoid Standup, Push-T, and Bugtrap.
- Methods: CBO, MPPI-CMA, CMA-ES, CEM, Predictive Sampling, and DIAL.
- Evaluation seeds represented by the dashboard: `[0, 1, 2, 3, 4]`.

## Contents

| Path | Description |
|---|---|
| `index.html` | Historical five-task dashboard with task tabs and videos. |
| `source/` | Exact tracked source, configuration, and dashboard snapshot from commit `96ff79a`. |
| `results/browser_data.js` | Aggregated method statistics and selected configurations used by the dashboard. |
| `results/media/` | 30 MP4 replays and 30 JPEG posters. |

Serve this archive directory over HTTP and open `index.html`; the preserved
relative paths resolve `results/browser_data.js` and all media locally.

## Best Five-Seed Mean

| Task | Method | Episode cost |
|---|---|---:|
| Cart Pole | CBO | $37.9309 \pm 2.55$ |
| Double Cart Pole | DIAL | $912.822 \pm 155$ |
| Humanoid Standup | CMA-ES | $538.015 \pm 13.1$ |
| Push-T | Predictive Sampling | $1.43553 \pm 0.329$ |
| Bugtrap | DIAL | $1.88213 \pm 0.192$ |

## Data Availability

The current workspace contains the aggregate browser payload and rendered
media, but not the five-task receding-horizon per-seed CSV that originally fed
this dashboard. The similarly named CSV in the legacy open-loop entry has a
different schema and metric and must not be treated as its raw source. This
archive records that limitation explicitly instead of inferring missing rows.
