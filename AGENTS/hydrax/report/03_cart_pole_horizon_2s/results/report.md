# Hydrax Receding-Horizon Offline Planning Benchmark

Each planning call optimizes $H_k$, executes
$E_k = 0.50H_k$, then replans.

- Backend: `jax`; device: `cuda:0`
- Evaluation seeds: `[0, 1, 2, 3, 4]`
- Planning time sums optimizer calls and excludes JIT and execution.
- Parameters come from `examples/to/selected_parameters.yaml`; tuning is not rerun.
- Cost sums every executed segment and its terminal cost:
  $J = \sum_{k=1}^{K}[\sum_{t=1}^{E_k}\Delta t\,
  \ell(x_{k,t},u_{k,t})+\phi(x_{k,E_k})]$.

| Task | Algorithm | Episode-cost | Planning-time-(ms) | Iterations/segment | Plan-horizon | Execution-horizon | Execution-fraction | Total-horizon | Planning-calls | Samples/iter | Randomizations | Knots | Spline | Frequency | Steps | Base-noise | Selected-parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| cart_pole | cbo | $37.9309 \pm 2.55$ | $546.121 \pm 1.739$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| cart_pole | mppi_cma | $41.4913 \pm 5.06$ | $552.020 \pm 3.210$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.25,"temperature_scale":0.1}` |
| cart_pole | cmaes | $40.2913 \pm 4.91$ | $552.685 \pm 0.399$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"c_mean":0.5,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| cart_pole | cem | $38.6158 \pm 2.72$ | $549.742 \pm 1.251$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"elite_fraction":0.05,"explore_fraction":0.0,"sigma_min_ratio":0.1}` |
| cart_pole | ps | $40.9172 \pm 1.62$ | $546.942 \pm 1.488$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{}` |
| cart_pole | dial | $38.6687 \pm 3.71$ | $547.687 \pm 1.725$ | 10 | 2 | 1 | 0.50 | 2 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"beta_horizon":2.0,"beta_opt_iter":2.0,"temperature_scale":10.0}` |