# CBO / CEM Time-Shifting Study

## Experiment

- Task: Double Cart Pole, 10 s episodes, 50 Hz control
- Optimizers: CBO and CEM, 1024 samples, 4 iterations
- Seeds: 0, 1, 2 (3 repeats per cell)
- Shift lengths: 1 step (0.01 s), 5 steps (0.05 s), 25 steps (0.25 s)
- Modes: `legacy` (shift `params.mean` only), `reset` (reset proposal state),
  `shift` (correctly shift proposal state)
- Video seed: 0, fixed for every cell

## Results

All costs below are episode cost divided by the total number of planning
calls for that shift length (1 step: 1000, 5 steps: 200, 25 steps: 40).

| Algorithm | Shift | Legacy cost/plan | Reset cost/plan | Shift cost/plan |
|---|---:|---:|---:|---:|
| CBO | 1 step | 80.25 | 30.99 | 28.11 |
| CBO | 5 steps | 134.33 | 31.68 | 38.62 |
| CBO | 25 steps | 91.95 | 55.27 | 56.00 |
| CEM | 1 step | 42.37 | 57.14 | 39.08 |
| CEM | 5 steps | 56.23 | 52.85 | 59.33 |
| CEM | 25 steps | 82.58 | 71.56 | 65.99 |

CBO benefits strongly from both reset and correct shifting. CEM shows a
mixed but useful signal: correct shifting is best at 1 step and 25 steps,
while reset is best at 5 steps.

## Artifacts

- `examples/to/results/double_cart_pole_timeshifting/episode_results.csv`
- `examples/to/results/double_cart_pole_timeshifting/summary.json`
- `examples/to/results/double_cart_pole_timeshifting/index.html`
- `examples/to/results/double_cart_pole_timeshifting/media/` (18 videos)
