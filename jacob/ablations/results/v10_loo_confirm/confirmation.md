# Confirmation: validation, every seed

Step-20 mean geodesic error over all validation windows (degrees). Test was not evaluated.

| Candidate | Params | Seed 42 | Seed 43 | Seed 44 | Mean | SD | vs reference | Selected seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k8_combined (reference) | 7,555 | 4.2471 | 4.2547 | 4.2485 | 4.2501 | 0.0041 | reference | 42 |
| no_prior | 7,555 | 4.2619 | 4.2405 | 4.2466 | 4.2497 | 0.0110 | -0.0004 | 43 |
| no_gravity | 7,459 | 4.2942 | 4.3067 | 4.3009 | 4.3006 | 0.0063 | +0.0505 | 42 |
| no_mirror | 15,113 | 4.2740 | 4.2817 | 4.2692 | 4.2750 | 0.0063 | +0.0249 | 44 |
| quaternion_loss | 7,555 | 4.3058 | 4.2855 | 4.2980 | 4.2964 | 0.0103 | +0.0463 | 43 |
| user_sampling | 7,555 | 4.2790 | 4.2763 | 4.2692 | 4.2748 | 0.0051 | +0.0247 | 44 |
| rf_243 | 11,427 | 4.2625 | 4.2765 | 4.2694 | 4.2695 | 0.0070 | +0.0194 | 42 |
| exact_maps | 7,555 | 4.2419 | 4.2455 | 4.2493 | 4.2456 | 0.0037 | -0.0045 | 42 |
| no_tilt | 7,555 | 4.2469 | 4.2503 | 4.2511 | 4.2494 | 0.0022 | -0.0007 | 42 |

A lower 3-seed mean is the plan's bar, not a decision: the paired per-user comparison and the submission choice follow, with the team.
