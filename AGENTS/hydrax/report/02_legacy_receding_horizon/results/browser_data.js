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
    },
    "double_cart_pole": {
      "budget": {
        "iterations": 10,
        "segment_horizon": 1.0,
        "total_horizon": 8.0,
        "samples": 1024,
        "randomizations": 1,
        "knots": 4,
        "spline": "cubic",
        "execution_horizon": 0.5,
        "execution_fraction": 0.5,
        "segments": 16
      },
      "algorithms": {
        "cbo": {
          "mean_cost": 1228.7489013671875,
          "std_cost": 520.45166015625,
          "mean_time_ms": 2680.6463284417987,
          "std_time_ms": 15.923234805869114,
          "best_seed": 0,
          "best_seed_cost": 878.7744750976562,
          "candidate": 0,
          "parameters": {
            "consensus_weight": 1.0,
            "noise_weight": 3.0,
            "step_size": 0.1,
            "temperature_scale": 0.1
          },
          "video": "results/media/double_cart_pole__cbo.mp4",
          "poster": "results/media/double_cart_pole__cbo.jpg"
        },
        "mppi_cma": {
          "mean_cost": 1020.9459228515625,
          "std_cost": 89.08123779296875,
          "mean_time_ms": 2706.8897344172,
          "std_time_ms": 9.212300977086645,
          "best_seed": 0,
          "best_seed_cost": 915.2825317382812,
          "candidate": 2,
          "parameters": {
            "adaptation_rate": 0.2,
            "minimum_noise_ratio": 1.0,
            "temperature_scale": 10.0
          },
          "video": "results/media/double_cart_pole__mppi_cma.mp4",
          "poster": "results/media/double_cart_pole__mppi_cma.jpg"
        },
        "cmaes": {
          "mean_cost": 3687.16845703125,
          "std_cost": 317.8025817871094,
          "mean_time_ms": 2738.677464146167,
          "std_time_ms": 13.35460407314443,
          "best_seed": 1,
          "best_seed_cost": 3322.67041015625,
          "candidate": 1,
          "parameters": {
            "c_mean": 1.0,
            "std_max_ratio": 100.0,
            "std_min_ratio": 0.0
          },
          "video": "results/media/double_cart_pole__cmaes.mp4",
          "poster": "results/media/double_cart_pole__cmaes.jpg"
        },
        "cem": {
          "mean_cost": 1186.9471435546875,
          "std_cost": 389.8112487792969,
          "mean_time_ms": 2700.5801171064377,
          "std_time_ms": 12.00002526537302,
          "best_seed": 2,
          "best_seed_cost": 911.0359497070312,
          "candidate": 1,
          "parameters": {
            "elite_fraction": 0.1,
            "explore_fraction": 0.25,
            "sigma_min_ratio": 0.25
          },
          "video": "results/media/double_cart_pole__cem.mp4",
          "poster": "results/media/double_cart_pole__cem.jpg"
        },
        "ps": {
          "mean_cost": 957.3269653320312,
          "std_cost": 164.7321319580078,
          "mean_time_ms": 2690.3742318041623,
          "std_time_ms": 13.122624121343792,
          "best_seed": 2,
          "best_seed_cost": 755.1474609375,
          "candidate": 0,
          "parameters": {},
          "video": "results/media/double_cart_pole__ps.mp4",
          "poster": "results/media/double_cart_pole__ps.jpg"
        },
        "dial": {
          "mean_cost": 912.8215942382812,
          "std_cost": 154.52928161621094,
          "mean_time_ms": 2682.312559802085,
          "std_time_ms": 1.7530747088729264,
          "best_seed": 3,
          "best_seed_cost": 718.5472412109375,
          "candidate": 2,
          "parameters": {
            "beta_horizon": 2.0,
            "beta_opt_iter": 2.0,
            "temperature_scale": 10.0
          },
          "video": "results/media/double_cart_pole__dial.mp4",
          "poster": "results/media/double_cart_pole__dial.jpg"
        }
      }
    },
    "humanoid_standup": {
      "budget": {
        "iterations": 5,
        "segment_horizon": 0.6,
        "total_horizon": 10.0,
        "samples": 128,
        "randomizations": 4,
        "knots": 4,
        "spline": "zero",
        "execution_horizon": 0.3,
        "execution_fraction": 0.5,
        "segments": 34
      },
      "algorithms": {
        "cbo": {
          "mean_cost": 728.6600952148438,
          "std_cost": 30.283557891845703,
          "mean_time_ms": 3847.7545784786344,
          "std_time_ms": 47.33574414857372,
          "best_seed": 2,
          "best_seed_cost": 723.89306640625,
          "candidate": 1,
          "parameters": {
            "consensus_weight": 1.0,
            "noise_weight": 7.0,
            "step_size": 0.1,
            "temperature_scale": 1.0
          },
          "video": "results/media/humanoid_standup__cbo.mp4",
          "poster": "results/media/humanoid_standup__cbo.jpg"
        },
        "mppi_cma": {
          "mean_cost": 554.1056518554688,
          "std_cost": 35.36598205566406,
          "mean_time_ms": 3806.9237688556314,
          "std_time_ms": 1.6499232151964784,
          "best_seed": 0,
          "best_seed_cost": 570.7935791015625,
          "candidate": 1,
          "parameters": {
            "adaptation_rate": 0.1,
            "minimum_noise_ratio": 0.5,
            "temperature_scale": 1.0
          },
          "video": "results/media/humanoid_standup__mppi_cma.mp4",
          "poster": "results/media/humanoid_standup__mppi_cma.jpg"
        },
        "cmaes": {
          "mean_cost": 538.0148315429688,
          "std_cost": 13.103898048400879,
          "mean_time_ms": 4024.5312788523734,
          "std_time_ms": 30.246735181196996,
          "best_seed": 4,
          "best_seed_cost": 538.0368041992188,
          "candidate": 1,
          "parameters": {
            "c_mean": 1.0,
            "std_max_ratio": 100.0,
            "std_min_ratio": 0.0
          },
          "video": "results/media/humanoid_standup__cmaes.mp4",
          "poster": "results/media/humanoid_standup__cmaes.jpg"
        },
        "cem": {
          "mean_cost": 554.3825073242188,
          "std_cost": 17.041467666625977,
          "mean_time_ms": 3854.421401116997,
          "std_time_ms": 29.055554256950128,
          "best_seed": 2,
          "best_seed_cost": 551.5764770507812,
          "candidate": 2,
          "parameters": {
            "elite_fraction": 0.2,
            "explore_fraction": 0.5,
            "sigma_min_ratio": 0.5
          },
          "video": "results/media/humanoid_standup__cem.mp4",
          "poster": "results/media/humanoid_standup__cem.jpg"
        },
        "ps": {
          "mean_cost": 576.837158203125,
          "std_cost": 15.118361473083496,
          "mean_time_ms": 3851.639903523028,
          "std_time_ms": 10.242414448128084,
          "best_seed": 1,
          "best_seed_cost": 594.8012084960938,
          "candidate": 0,
          "parameters": {},
          "video": "results/media/humanoid_standup__ps.mp4",
          "poster": "results/media/humanoid_standup__ps.jpg"
        },
        "dial": {
          "mean_cost": 597.4671630859375,
          "std_cost": 33.91374206542969,
          "mean_time_ms": 3872.385114058852,
          "std_time_ms": 20.561428020039656,
          "best_seed": 1,
          "best_seed_cost": 617.3988647460938,
          "candidate": 1,
          "parameters": {
            "beta_horizon": 1.0,
            "beta_opt_iter": 1.0,
            "temperature_scale": 1.0
          },
          "video": "results/media/humanoid_standup__dial.mp4",
          "poster": "results/media/humanoid_standup__dial.jpg"
        }
      }
    },
    "pusht": {
      "budget": {
        "iterations": 10,
        "segment_horizon": 0.5,
        "total_horizon": 5.0,
        "samples": 128,
        "randomizations": 4,
        "knots": 6,
        "spline": "zero",
        "execution_horizon": 0.25,
        "execution_fraction": 0.5,
        "segments": 20
      },
      "algorithms": {
        "cbo": {
          "mean_cost": 2.3420586585998535,
          "std_cost": 1.2797772884368896,
          "mean_time_ms": 2470.537205412984,
          "std_time_ms": 11.92873232687953,
          "best_seed": 0,
          "best_seed_cost": 1.1718848943710327,
          "candidate": 0,
          "parameters": {
            "consensus_weight": 1.0,
            "noise_weight": 3.0,
            "step_size": 0.1,
            "temperature_scale": 0.1
          },
          "video": "results/media/pusht__cbo.mp4",
          "poster": "results/media/pusht__cbo.jpg"
        },
        "mppi_cma": {
          "mean_cost": 1.723858118057251,
          "std_cost": 0.4135834574699402,
          "mean_time_ms": 2472.3876268602908,
          "std_time_ms": 1.5657772468141926,
          "best_seed": 0,
          "best_seed_cost": 1.492551565170288,
          "candidate": 0,
          "parameters": {
            "adaptation_rate": 0.1,
            "minimum_noise_ratio": 0.25,
            "temperature_scale": 0.1
          },
          "video": "results/media/pusht__mppi_cma.mp4",
          "poster": "results/media/pusht__mppi_cma.jpg"
        },
        "cmaes": {
          "mean_cost": 1.5587483644485474,
          "std_cost": 0.14058786630630493,
          "mean_time_ms": 2510.884773544967,
          "std_time_ms": 0.4929233782111928,
          "best_seed": 0,
          "best_seed_cost": 1.361858606338501,
          "candidate": 1,
          "parameters": {
            "c_mean": 1.0,
            "std_max_ratio": 100.0,
            "std_min_ratio": 0.0
          },
          "video": "results/media/pusht__cmaes.mp4",
          "poster": "results/media/pusht__cmaes.jpg"
        },
        "cem": {
          "mean_cost": 1.4426463842391968,
          "std_cost": 0.1227082908153534,
          "mean_time_ms": 2467.844565678388,
          "std_time_ms": 2.218391667895567,
          "best_seed": 3,
          "best_seed_cost": 1.3370915651321411,
          "candidate": 1,
          "parameters": {
            "elite_fraction": 0.1,
            "explore_fraction": 0.25,
            "sigma_min_ratio": 0.25
          },
          "video": "results/media/pusht__cem.mp4",
          "poster": "results/media/pusht__cem.jpg"
        },
        "ps": {
          "mean_cost": 1.4355342388153076,
          "std_cost": 0.32898029685020447,
          "mean_time_ms": 2489.314612559974,
          "std_time_ms": 10.642730290821405,
          "best_seed": 4,
          "best_seed_cost": 1.1476489305496216,
          "candidate": 0,
          "parameters": {},
          "video": "results/media/pusht__ps.mp4",
          "poster": "results/media/pusht__ps.jpg"
        },
        "dial": {
          "mean_cost": 2.4600231647491455,
          "std_cost": 0.7937784790992737,
          "mean_time_ms": 2531.3365761190653,
          "std_time_ms": 27.427699095727387,
          "best_seed": 0,
          "best_seed_cost": 1.7992680072784424,
          "candidate": 1,
          "parameters": {
            "beta_horizon": 1.0,
            "beta_opt_iter": 1.0,
            "temperature_scale": 1.0
          },
          "video": "results/media/pusht__dial.mp4",
          "poster": "results/media/pusht__dial.jpg"
        }
      }
    },
    "bugtrap": {
      "budget": {
        "iterations": 5,
        "segment_horizon": 2.0,
        "total_horizon": 10.0,
        "samples": 128,
        "randomizations": 1,
        "knots": 11,
        "spline": "zero",
        "execution_horizon": 1.0,
        "execution_fraction": 0.5,
        "segments": 10
      },
      "algorithms": {
        "cbo": {
          "mean_cost": 1.9886528253555298,
          "std_cost": 0.11974909156560898,
          "mean_time_ms": 3870.879099704325,
          "std_time_ms": 469.81969890355765,
          "best_seed": 2,
          "best_seed_cost": 1.7997926473617554,
          "candidate": 2,
          "parameters": {
            "consensus_weight": 2.0,
            "noise_weight": 3.0,
            "step_size": 0.05,
            "temperature_scale": 10.0
          },
          "video": "results/media/bugtrap__cbo.mp4",
          "poster": "results/media/bugtrap__cbo.jpg"
        },
        "mppi_cma": {
          "mean_cost": 2.1574866771698,
          "std_cost": 0.11684636771678925,
          "mean_time_ms": 3572.7904710918665,
          "std_time_ms": 24.00805591518576,
          "best_seed": 4,
          "best_seed_cost": 2.023266315460205,
          "candidate": 1,
          "parameters": {
            "adaptation_rate": 0.1,
            "minimum_noise_ratio": 0.5,
            "temperature_scale": 1.0
          },
          "video": "results/media/bugtrap__mppi_cma.mp4",
          "poster": "results/media/bugtrap__mppi_cma.jpg"
        },
        "cmaes": {
          "mean_cost": 1.9349721670150757,
          "std_cost": 0.28096437454223633,
          "mean_time_ms": 3534.812272526324,
          "std_time_ms": 39.379053122817126,
          "best_seed": 0,
          "best_seed_cost": 1.5990885496139526,
          "candidate": 2,
          "parameters": {
            "c_mean": 1.0,
            "std_max_ratio": 100.0,
            "std_min_ratio": 0.1
          },
          "video": "results/media/bugtrap__cmaes.mp4",
          "poster": "results/media/bugtrap__cmaes.jpg"
        },
        "cem": {
          "mean_cost": 3.702008008956909,
          "std_cost": 2.126099109649658,
          "mean_time_ms": 3820.749870594591,
          "std_time_ms": 496.0073328833856,
          "best_seed": 4,
          "best_seed_cost": 1.589127540588379,
          "candidate": 0,
          "parameters": {
            "elite_fraction": 0.05,
            "explore_fraction": 0.0,
            "sigma_min_ratio": 0.1
          },
          "video": "results/media/bugtrap__cem.mp4",
          "poster": "results/media/bugtrap__cem.jpg"
        },
        "ps": {
          "mean_cost": 2.7398791313171387,
          "std_cost": 0.3847762644290924,
          "mean_time_ms": 3917.936793528497,
          "std_time_ms": 422.3659017733307,
          "best_seed": 1,
          "best_seed_cost": 2.140744209289551,
          "candidate": 0,
          "parameters": {},
          "video": "results/media/bugtrap__ps.mp4",
          "poster": "results/media/bugtrap__ps.jpg"
        },
        "dial": {
          "mean_cost": 1.882132887840271,
          "std_cost": 0.19177238643169403,
          "mean_time_ms": 3654.857795778662,
          "std_time_ms": 370.1256984573347,
          "best_seed": 2,
          "best_seed_cost": 1.6408066749572754,
          "candidate": 1,
          "parameters": {
            "beta_horizon": 1.0,
            "beta_opt_iter": 1.0,
            "temperature_scale": 1.0
          },
          "video": "results/media/bugtrap__dial.mp4",
          "poster": "results/media/bugtrap__dial.jpg"
        }
      }
    }
  }
};
