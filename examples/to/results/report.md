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
| cart_pole | cbo | $8.61873 \pm 0.109$ | $269.516 \pm 0.879$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| cart_pole | mppi_cma | $8.62314 \pm 0.0796$ | $269.968 \pm 0.811$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"adaptation_rate":0.2,"minimum_noise_ratio":1.0,"temperature_scale":10.0}` |
| cart_pole | cmaes | $9.73962 \pm 0.946$ | $270.408 \pm 0.653$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| cart_pole | cem | $8.51372 \pm 0.0748$ | $268.606 \pm 0.519$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| cart_pole | ps | $8.4785 \pm 0.0738$ | $269.041 \pm 0.892$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{}` |
| cart_pole | dial | $8.36805 \pm 0.0173$ | $270.626 \pm 3.902$ | 10 | 2 | 128 | 1 | 4 | cubic | 100 | 200 | 0.3 | `{"beta_horizon":1.0,"beta_opt_iter":1.0,"temperature_scale":1.0}` |
| double_cart_pole | cbo | $119.731 \pm 1.33$ | $168.398 \pm 0.451$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| double_cart_pole | mppi_cma | $120.821 \pm 0.114$ | $168.270 \pm 0.528$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.5,"temperature_scale":1.0}` |
| double_cart_pole | cmaes | $124.164 \pm 1.39$ | $169.873 \pm 0.545$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| double_cart_pole | cem | $126.087 \pm 3.77$ | $169.375 \pm 0.600$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| double_cart_pole | ps | $120.75 \pm 0.0277$ | $168.907 \pm 0.481$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{}` |
| double_cart_pole | dial | $120.76 \pm 0.0949$ | $167.932 \pm 0.351$ | 10 | 1 | 1024 | 1 | 4 | cubic | 100 | 100 | 0.3 | `{"beta_horizon":2.0,"beta_opt_iter":2.0,"temperature_scale":10.0}` |
| humanoid_standup | cbo | $22.5299 \pm 0.905$ | $113.432 \pm 1.268$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"consensus_weight":1.0,"noise_weight":7.0,"step_size":0.1,"temperature_scale":1.0}` |
| humanoid_standup | mppi_cma | $22.6012 \pm 0.535$ | $113.932 \pm 1.194$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.25,"temperature_scale":0.1}` |
| humanoid_standup | cmaes | $23.9586 \pm 0.371$ | $118.420 \pm 1.248$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| humanoid_standup | cem | $23.2211 \pm 0.898$ | $113.275 \pm 1.224$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"elite_fraction":0.05,"explore_fraction":0.0,"sigma_min_ratio":0.1}` |
| humanoid_standup | ps | $22.8354 \pm 0.953$ | $114.304 \pm 2.584$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{}` |
| humanoid_standup | dial | $23.4392 \pm 0.6$ | $113.474 \pm 1.385$ | 5 | 0.6 | 128 | 4 | 4 | zero | 50 | 30 | 0.3 | `{"beta_horizon":0.5,"beta_opt_iter":0.5,"temperature_scale":0.1}` |
| pusht | cbo | $0.518803 \pm 0.0488$ | $125.142 \pm 0.606$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"consensus_weight":1.0,"noise_weight":3.0,"step_size":0.1,"temperature_scale":0.1}` |
| pusht | mppi_cma | $0.578287 \pm 0.0542$ | $125.062 \pm 0.393$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"adaptation_rate":0.2,"minimum_noise_ratio":1.0,"temperature_scale":10.0}` |
| pusht | cmaes | $0.588886 \pm 0.0659$ | $127.776 \pm 0.514$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.0}` |
| pusht | cem | $0.504763 \pm 0.0156$ | $125.095 \pm 0.668$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"elite_fraction":0.1,"explore_fraction":0.25,"sigma_min_ratio":0.25}` |
| pusht | ps | $0.567217 \pm 0.111$ | $124.911 \pm 0.373$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{}` |
| pusht | dial | $0.497282 \pm 0.0336$ | $124.901 \pm 0.542$ | 10 | 0.5 | 128 | 4 | 6 | zero | 100 | 50 | 0.4 | `{"beta_horizon":1.0,"beta_opt_iter":1.0,"temperature_scale":1.0}` |
| bugtrap | cbo | $0.85945 \pm 0.0474$ | $633.297 \pm 12.205$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"consensus_weight":2.0,"noise_weight":3.0,"step_size":0.05,"temperature_scale":10.0}` |
| bugtrap | mppi_cma | $1.06048 \pm 0.0575$ | $689.116 \pm 10.245$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"adaptation_rate":0.1,"minimum_noise_ratio":0.5,"temperature_scale":1.0}` |
| bugtrap | cmaes | $0.866897 \pm 0.0281$ | $658.427 \pm 5.514$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"c_mean":1.0,"std_max_ratio":100.0,"std_min_ratio":0.1}` |
| bugtrap | cem | $0.823719 \pm 0.0193$ | $577.990 \pm 1.779$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"elite_fraction":0.05,"explore_fraction":0.0,"sigma_min_ratio":0.1}` |
| bugtrap | ps | $0.95431 \pm 0.0217$ | $700.171 \pm 4.913$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{}` |
| bugtrap | dial | $0.815402 \pm 0.0173$ | $577.946 \pm 1.692$ | 20 | 1 | 32 | 1 | 11 | zero | 100 | 100 | 1 | `{"beta_horizon":0.5,"beta_opt_iter":0.5,"temperature_scale":0.1}` |

## Tuning Protocol

Candidates were compared by mean final best cost on the tuning seeds. Evaluation uses disjoint seeds. Predictive sampling has no algorithm-specific parameter once shared base noise is fixed. The complete candidate results are in `tuning_results.csv`, and the exact selected values are in `selected_parameters.yaml`.
