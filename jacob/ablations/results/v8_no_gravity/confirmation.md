# v7 confirmation: validation, every seed

Step-20 mean geodesic error over all validation windows (degrees). Test was not evaluated. Plan tolerance for the 3-seed mean: 4.382.

| Candidate | Params | Seed 42 | Seed 43 | Seed 44 | Mean | SD | Within tolerance | Selected seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| v6 control (reference) | 26,012 | 4.3511 | 4.3472 | 4.3285 | 4.3423 | 0.0121 | reference | 44 |
| separable_base3_addons_angular | 6,651 | 4.3386 | 4.3370 | 4.3134 | 4.3297 | 0.0142 | yes | 44 |
| separable_base3_addons_angular_no_gravity | 6,579 | 4.3630 | 4.3686 | 4.3799 | 4.3705 | 0.0086 | yes | 42 |

Plan rule (smallest candidate within tolerance): **separable_base3_addons_angular_no_gravity**.
This is the plan's rule, not a decision: the submission is chosen by the team after the hardware latencies.
