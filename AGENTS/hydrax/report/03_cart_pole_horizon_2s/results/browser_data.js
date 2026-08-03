window.HYDRAX_RESULTS = {
  "backend": "jax",
  "evaluation_seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "tasks": {
    "cart_pole": {
      "budget": {
        "iterations": 10,
        "segment_horizon": 2.0,
        "total_horizon": 2.0,
        "samples": 128,
        "randomizations": 1,
        "knots": 4,
        "spline": "cubic",
        "execution_horizon": 1.0,
        "execution_fraction": 0.5,
        "segments": 2
      },
      "algorithms": {
        "cbo": {
          "mean_cost": 37.930877685546875,
          "std_cost": 2.5547189712524414,
          "mean_time_ms": 546.1210421286523,
          "std_time_ms": 1.7386757617550161,
          "best_seed": 4,
          "best_seed_cost": 35.29623031616211,
          "candidate": 1,
          "parameters": {
            "consensus_weight": 1.0,
            "noise_weight": 7.0,
            "step_size": 0.1,
            "temperature_scale": 1.0
          },
          "video": "results/media/cart_pole__cbo.mp4",
          "poster": "results/media/cart_pole__cbo.jpg"
        },
        "mppi_cma": {
          "mean_cost": 41.491329193115234,
          "std_cost": 5.064252853393555,
          "mean_time_ms": 552.0199337974191,
          "std_time_ms": 3.209827845133292,
          "best_seed": 4,
          "best_seed_cost": 35.78472900390625,
          "candidate": 0,
          "parameters": {
            "adaptation_rate": 0.1,
            "minimum_noise_ratio": 0.25,
            "temperature_scale": 0.1
          },
          "video": "results/media/cart_pole__mppi_cma.mp4",
          "poster": "results/media/cart_pole__mppi_cma.jpg"
        },
        "cmaes": {
          "mean_cost": 40.29134750366211,
          "std_cost": 4.914037227630615,
          "mean_time_ms": 552.6853869669139,
          "std_time_ms": 0.3994918190915362,
          "best_seed": 3,
          "best_seed_cost": 33.360877990722656,
          "candidate": 0,
          "parameters": {
            "c_mean": 0.5,
            "std_max_ratio": 100.0,
            "std_min_ratio": 0.0
          },
          "video": "results/media/cart_pole__cmaes.mp4",
          "poster": "results/media/cart_pole__cmaes.jpg"
        },
        "cem": {
          "mean_cost": 38.61580276489258,
          "std_cost": 2.718235492706299,
          "mean_time_ms": 549.7418480925262,
          "std_time_ms": 1.2506336967293468,
          "best_seed": 2,
          "best_seed_cost": 35.954200744628906,
          "candidate": 0,
          "parameters": {
            "elite_fraction": 0.05,
            "explore_fraction": 0.0,
            "sigma_min_ratio": 0.1
          },
          "video": "results/media/cart_pole__cem.mp4",
          "poster": "results/media/cart_pole__cem.jpg"
        },
        "ps": {
          "mean_cost": 40.91721725463867,
          "std_cost": 1.6178885698318481,
          "mean_time_ms": 546.9417729415,
          "std_time_ms": 1.4882846093768822,
          "best_seed": 0,
          "best_seed_cost": 39.263328552246094,
          "candidate": 0,
          "parameters": {},
          "video": "results/media/cart_pole__ps.mp4",
          "poster": "results/media/cart_pole__ps.jpg"
        },
        "dial": {
          "mean_cost": 38.66864776611328,
          "std_cost": 3.714895486831665,
          "mean_time_ms": 547.6873537525535,
          "std_time_ms": 1.7253729035928593,
          "best_seed": 2,
          "best_seed_cost": 35.290767669677734,
          "candidate": 2,
          "parameters": {
            "beta_horizon": 2.0,
            "beta_opt_iter": 2.0,
            "temperature_scale": 10.0
          },
          "video": "results/media/cart_pole__dial.mp4",
          "poster": "results/media/cart_pole__dial.jpg"
        }
      }
    }
  }
};
