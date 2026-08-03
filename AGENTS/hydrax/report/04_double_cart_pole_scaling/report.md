# Double Cart Pole Operation-Budget Benchmark

The operation proxy is $O=I N$, where $I$ is the number of optimizer iterations and $N$ is the number of samples per iteration. The best configuration minimizes the five-seed mean episode cost.

- Seeds: `[0, 1, 2, 3, 4]`
- Total horizon: $10$ s
- Planning horizon: $1$ s
- Knots: $6$
- All algorithm-specific parameters are fixed across the grid.

## Fixed Baseline

| Method | $I$ | $N$ | $O$ | Episode cost | Planning time (s) |
|---|---:|---:|---:|---:|---:|
| CBO | 8 | 1024 | 8192 | $1607.91 \pm 422$ | $2.66259 \pm 0.00061$ |
| MPPI-CMA | 8 | 1024 | 8192 | $1087.57 \pm 124$ | $2.73126 \pm 0.0352$ |
| CMA-ES | 8 | 1024 | 8192 | $6058.59 \pm 2.79e+03$ | $2.68718 \pm 0.00131$ |
| CEM | 8 | 1024 | 8192 | $1346.86 \pm 344$ | $2.68456 \pm 0.0137$ |
| Predictive Sampling | 8 | 1024 | 8192 | $1074.2 \pm 82.1$ | $2.70351 \pm 0.00181$ |
| DIAL | 8 | 1024 | 8192 | $1131.51 \pm 231$ | $2.66766 \pm 0.00108$ |

## Best Allocation by Operation Budget

| Method | $O$ | $I^*$ | $N^*$ | Episode cost |
|---|---:|---:|---:|---:|
| CBO | 8192 | 2 | 4096 | $1422.35 \pm 377$ |
| CBO | 4096 | 1 | 4096 | $1468.3 \pm 315$ |
| CBO | 2048 | 4 | 512 | $1545.04 \pm 297$ |
| CBO | 1024 | 4 | 256 | $1485.26 \pm 235$ |
| CBO | 512 | 2 | 256 | $1736.17 \pm 367$ |
| MPPI-CMA | 8192 | 4 | 2048 | $1060.57 \pm 124$ |
| MPPI-CMA | 4096 | 2 | 2048 | $1115.2 \pm 194$ |
| MPPI-CMA | 2048 | 8 | 256 | $1157.09 \pm 169$ |
| MPPI-CMA | 1024 | 4 | 256 | $1234.3 \pm 172$ |
| MPPI-CMA | 512 | 8 | 64 | $1416.83 \pm 198$ |
| CMA-ES | 8192 | 1 | 8192 | $1684.74 \pm 346$ |
| CMA-ES | 4096 | 1 | 4096 | $1873.21 \pm 678$ |
| CMA-ES | 2048 | 1 | 2048 | $1745.25 \pm 689$ |
| CMA-ES | 1024 | 1 | 1024 | $2050.53 \pm 297$ |
| CMA-ES | 512 | 1 | 512 | $2153.57 \pm 461$ |
| CEM | 8192 | 16 | 512 | $1320.24 \pm 229$ |
| CEM | 4096 | 8 | 512 | $1358.48 \pm 71$ |
| CEM | 2048 | 16 | 128 | $1223.99 \pm 105$ |
| CEM | 1024 | 8 | 128 | $1324.93 \pm 161$ |
| CEM | 512 | 4 | 128 | $1252.53 \pm 74.9$ |
| Predictive Sampling | 8192 | 4 | 2048 | $1063.34 \pm 125$ |
| Predictive Sampling | 4096 | 4 | 1024 | $1079.86 \pm 146$ |
| Predictive Sampling | 2048 | 8 | 256 | $1158.52 \pm 238$ |
| Predictive Sampling | 1024 | 8 | 128 | $1218.34 \pm 162$ |
| Predictive Sampling | 512 | 4 | 128 | $1371.08 \pm 222$ |
| DIAL | 8192 | 16 | 512 | $1090.36 \pm 117$ |
| DIAL | 4096 | 16 | 256 | $1145.26 \pm 161$ |
| DIAL | 2048 | 16 | 128 | $1102.55 \pm 229$ |
| DIAL | 1024 | 16 | 64 | $1222.06 \pm 198$ |
| DIAL | 512 | 16 | 32 | $1229.63 \pm 172$ |
