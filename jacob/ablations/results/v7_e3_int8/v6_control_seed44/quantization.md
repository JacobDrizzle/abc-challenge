# Int8 post-training quantisation: outputs\v6_group_a_v2\selected_model.pt

Validation users, every full window streamed from each segment's first sample (657,189 windows). Calibration: 1000 streamed inputs from train users. Quantised: block Convs and head; features, output maths and states stay float. Latency: ORT CPU, 1 thread (relative only).

| Graph | Step-20 mean (°) | vs float (°) | p99 per-window change (°) | Median (ms) | KiB | Off-list |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| float32 | 4.3285 | +0.0000 | 0.0000 | 0.1507 | 129.7 | Sqrt |
| int8_minmax | 4.3498 | +0.0213 | 0.7299 | 0.1378 | 80.7 | DequantizeLinear, QuantizeLinear, Sqrt |
| int8_percentile | 4.3410 | +0.0124 | 0.4581 | 0.1745 | 80.7 | DequantizeLinear, QuantizeLinear, Sqrt |
| w8a16_minmax | 4.3349 | +0.0064 | 0.1345 | 0.1948 | 80.7 | DequantizeLinear, QuantizeLinear, Sqrt |
