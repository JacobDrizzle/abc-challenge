# Int8 post-training quantisation: outputs\v9_confirm\k8_combined\selected_model.pt

Validation users, every full window streamed from each segment's first sample (657,189 windows). Calibration: 1000 streamed inputs from train users. Quantised: block Convs and head; features, output maths and states stay float. Latency: ORT CPU, 1 thread (relative only).

| Graph | Step-20 mean (°) | vs float (°) | p99 per-window change (°) | Median (ms) | KiB | Off-list |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| float32 | 4.2471 | +0.0000 | 0.0000 | 0.0528 | 43.2 | none |
| int8_minmax | 4.3488 | +0.1017 | 1.6635 | 0.0866 | 38.2 | DequantizeLinear, QuantizeLinear |
| int8_percentile | 4.2720 | +0.0249 | 0.7482 | 0.0861 | 38.2 | DequantizeLinear, QuantizeLinear |
