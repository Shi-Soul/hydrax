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

| Algorithm | Shift | Legacy cost | Reset cost | Shift cost |
|---|---:|---:|---:|---:|
| CBO | 1 step | 80249.69 | 30987.53 | 28110.82 |
| CBO | 5 steps | 26865.02 | 6335.38 | 7723.78 |
| CBO | 25 steps | 3678.09 | 2210.79 | 2239.82 |
| CEM | 1 step | 42371.85 | 57136.86 | 39078.52 |
| CEM | 5 steps | 11245.60 | 10570.65 | 11866.87 |
| CEM | 25 steps | 3303.28 | 2862.30 | 2639.51 |

CBO benefits strongly from both reset and correct shifting. CEM shows a
mixed but useful signal: correct shifting is best at 1 step and 25 steps,
while reset is best at 5 steps.

## Artifacts

- `examples/to/results/double_cart_pole_timeshifting/episode_results.csv`
- `examples/to/results/double_cart_pole_timeshifting/summary.json`
- `examples/to/results/double_cart_pole_timeshifting/index.html`
- `examples/to/results/double_cart_pole_timeshifting/media/` (18 videos)
