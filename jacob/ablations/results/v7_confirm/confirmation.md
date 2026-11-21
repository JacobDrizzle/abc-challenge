# v7 confirmation: validation, every seed

Step-20 mean geodesic error over all validation windows (degrees). Test was not evaluated. Plan tolerance for the 3-seed mean: 4.382.

| Candidate | Params | Seed 42 | Seed 43 | Seed 44 | Mean | SD | Within tolerance | Selected seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| v6 control (reference) | 26,012 | 4.3511 | 4.3472 | 4.3285 | 4.3423 | 0.0121 | reference | 44 |
| separable_base3_addons | 6,651 | 4.3855 | 4.3699 | 4.3933 | 4.3829 | 0.0119 | no | 43 |
| base3_5_addons | 17,811 | 4.3802 | 4.3561 | 4.3655 | 4.3673 | 0.0121 | yes | 43 |

Plan rule (smallest candidate within tolerance): **base3_5_addons**.
This is the plan's rule, not a decision: the submission is chosen by the team after the hardware latencies.
