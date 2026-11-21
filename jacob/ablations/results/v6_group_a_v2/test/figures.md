# Model performance figures

Split: **test** · 20 users · 931,051 windows. Lower error is better. Figures are drawn from `results.json` and the saved per-window errors.

| Predictor | Endpoint mean (°) | Median (°) | p95 (°) | 100 ms (°) | Equal-user mean (°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TCNN | 4.3584 | 3.6451 | 10.0531 | 2.2305 | 4.2979 |
| Damped velocity | 5.1137 | 4.2985 | 11.7818 | 2.5903 | 5.0729 |
| Hold last | 6.0373 | 5.0155 | 14.1979 | 3.4733 | 5.9381 |
| Constant velocity (1 step) | 6.9438 | 5.6624 | 16.6184 | 3.0542 | 6.9462 |
| Constant velocity (5 steps) | 7.0478 | 5.8086 | 16.5881 | 3.2988 | 7.0410 |

- TCNN has lower mean endpoint error than Damped velocity for **20 of 20** users.
- TCNN minus Damped velocity: **-0.7554°** endpoint error, user-bootstrap 95% CI -0.8905° to -0.6424° (20 users). TCNN is better in 65.5% of windows.

## Horizon relative

Left: percentage error reduction against hold-last at every forecast step. Right: TCNN minus Damped velocity in degrees at each step; bars above zero are steps where Damped velocity is better.

![horizon_relative.png](horizon_relative.png)

## Endpoint cdf

Cumulative distribution of endpoint error per predictor. Dots mark the median and 95th percentile of TCNN and Damped velocity.

![endpoint_cdf.png](endpoint_cdf.png)

## Per user endpoint

Mean endpoint error for each user, sorted by the primary model. Users differ far more than predictors do, which is why user-level uncertainty matters.

![per_user_endpoint.png](per_user_endpoint.png)

## Paired difference

Left: per-window endpoint difference (the 0.5% largest absolute differences are outside the plotted range). Right: per-user mean difference; the line and band are the window-weighted mean and a 95% interval from resampling users, since windows from one user are correlated.

![paired_difference.png](paired_difference.png)

## Error vs speed

Endpoint error grouped by how fast the head was turning just before the forecast. The lower panel shows how many windows fall in each speed band.

![error_vs_speed.png](error_vs_speed.png)

## Example forecasts

Angle from the last observed orientation for real windows at the median, 90th and 99th percentile of TCNN error. Grey is the observed past; black is what actually happened.

![example_forecasts.png](example_forecasts.png)

## Training curves

Training loss, validation endpoint error and learning rate per epoch for each seed, with the damped-velocity and hold-last references measured on the same validation windows.

![training_curves.png](training_curves.png)

## Variant comparison

Validation error of each screened configuration (seed 42), from comparison.json.

![variant_comparison.png](variant_comparison.png)
