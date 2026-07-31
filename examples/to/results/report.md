# Hydrax Open-Loop Planning Benchmark

Each run optimizes once from the fixed example initial state. JIT compilation, setup, metric reduction, and output are excluded from planning time. The reported best cost is the minimum total rollout cost in the final internal optimization iteration.

- Backend: `jax`
- Device: `cuda:0`
- JAX: `0.8.3`
- MuJoCo: `3.8.0`
- Evaluation seeds: `[0, 1, 2, 3, 4]`
- Control frequency and maximum episode steps are open-loop discretization metadata, with $f = 1 / \Delta t$ and $N = \operatorname{round}(T / \Delta t)$.

| Task | Algorithm | Final best cost | Planning time (ms) | Iterations | Horizon | Samples | Randomizations | Knots | Spline | Frequency | Steps | Base noise | Selected parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| cart_pole | cbo | $8.61873 \pm 0.109$ | $269.654 \pm 0.442$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| cart_pole | mppi_cma | $8.62314 \pm 0.0796$ | $269.886 \pm 0.275$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"adaptation_rate":0.2,"minimum_noise_ratio":1.0,"temperature_scale":10.0}` |
| cart_pole | cmaes | $9.73962 \pm 0.946$ | $273.719 \pm 0.598$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| cart_pole | cem | $8.51372 \pm 0.0748$ | $270.400 \pm 0.764$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| cart_pole | ps | $8.4785 \pm 0.0738$ | $270.187 \pm 1.239$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{}` |
| cart_pole | dial | $8.36805 \pm 0.0173$ | $269.973 \pm 0.544$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"beta_horizon":1.0,"beta_opt_iter":1.0,"temperature_scale":1.0}` |
| double_cart_pole | cbo | $119.731 \pm 1.33$ | $169.636 \pm 0.518$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| double_cart_pole | mppi_cma | $120.821 \pm 0.114$ | $170.253 \pm 0.805$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.5,"temperature_scale":1.0}` |
| double_cart_pole | cmaes | $124.164 \pm 1.39$ | $173.323 \pm 0.361$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| double_cart_pole | cem | $126.087 \pm 3.77$ | $169.141 \pm 1.017$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| double_cart_pole | ps | $120.75 \pm 0.0277$ | $170.228 \pm 1.290$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{}` |
| double_cart_pole | dial | $120.76 \pm 0.0949$ | $170.678 \pm 1.684$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"beta_horizon":2.0,"beta_opt_iter":2.0,"temperature_scale":10.0}` |
| humanoid_standup | cbo | $22.3789 \pm 0.734$ | $114.442 \pm 0.922$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| humanoid_standup | mppi_cma | $22.9652 \pm 1.14$ | $114.900 \pm 1.063$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.5,"temperature_scale":1.0}` |
| humanoid_standup | cmaes | $24.4978 \pm 0.14$ | $118.639 \pm 1.264$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| humanoid_standup | cem | $22.9636 \pm 0.864$ | $114.173 \pm 0.965$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"elite_fraction":0.05,"explore_fraction":0.0,"sigma_min_ratio":0.1}` |
| humanoid_standup | ps | $22.8673 \pm 0.932$ | $113.571 \pm 0.618$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{}` |
| humanoid_standup | dial | $22.1286 \pm 0.409$ | $114.001 \pm 0.793$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"beta_horizon":1.0,"beta_opt_iter":1.0,"temperature_scale":1.0}` |
| pusht | cbo | $0.518803 \pm 0.0488$ | $125.063 \pm 0.392$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"consensus_weight":1.0,"noise_weight":3.0,"step_size":0.1,"temperature_scale":0.1}` |
| pusht | mppi_cma | $0.578287 \pm 0.0542$ | $125.299 \pm 0.264$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"adaptation_rate":0.2,"minimum_noise_ratio":1.0,"temperature_scale":10.0}` |
| pusht | cmaes | $0.588886 \pm 0.0659$ | $126.483 \pm 0.887$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| pusht | cem | $0.504763 \pm 0.0156$ | $124.397 \pm 0.500$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| pusht | ps | $0.567217 \pm 0.111$ | $124.226 \pm 0.277$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{}` |
| pusht | dial | $0.497282 \pm 0.0336$ | $124.265 \pm 0.579$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"beta_horizon":1.0,"beta_opt_iter":1.0,"temperature_scale":1.0}` |
| bugtrap | cbo | $0.85945 \pm 0.0474$ | $636.795 \pm 12.884$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"consensus_weight":2.0,"noise_weight":3.0,"step_size":0.05,"temperature_scale":10.0}` |
| bugtrap | mppi_cma | $1.06048 \pm 0.0575$ | $689.110 \pm 10.140$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.5,"temperature_scale":1.0}` |
| bugtrap | cmaes | $0.866897 \pm 0.0281$ | $660.011 \pm 4.833$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.1}` |
| bugtrap | cem | $0.823719 \pm 0.0193$ | $579.925 \pm 2.634$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"elite_fraction":0.05,"explore_fraction":0.0,"sigma_min_ratio":0.1}` |
| bugtrap | ps | $0.95431 \pm 0.0217$ | $701.029 \pm 4.594$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{}` |
| bugtrap | dial | $0.815402 \pm 0.0173$ | $579.996 \pm 1.368$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"beta_horizon":0.5,"beta_opt_iter":0.5,"temperature_scale":0.1}` |

## Tuning Protocol

Candidates were compared by mean final best cost on the tuning seeds. Evaluation uses disjoint seeds. Predictive sampling has no algorithm-specific parameter once shared base noise is fixed. The complete candidate results are in `tuning_results.csv`, and the exact selected values are in `selected_parameters.yaml`.
