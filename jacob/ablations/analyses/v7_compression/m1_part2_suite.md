| Graph | Params | MACs/step | State floats | Nodes | Off-list | Median (ms) | vs v6 graph |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| control (v6 export) | 26,012 | 52,144 | 12,192 | 174 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | 0.1277 | +0.0% |
| control | 26,012 | 52,194 | 12,192 | 145 | Sqrt | 0.1106 | -13.4% |
| speed_l1 | 26,012 | 52,215 | 12,192 | 145 | none | 0.1111 | -13.0% |
| basis_head | 24,771 | 49,986 | 12,192 | 145 | Sqrt | 0.1113 | -12.8% |
| base3_5 | 19,052 | 38,370 | 11,616 | 125 | Sqrt | 0.0936 | -26.7% |
| width_20 | 18,320 | 36,874 | 10,160 | 145 | Sqrt | 0.1074 | -15.9% |
| narrow_full | 11,972 | 24,242 | 8,128 | 145 | Sqrt | 0.1043 | -18.3% |
| separable_24 | 10,388 | 20,946 | 12,192 | 152 | Sqrt | 0.1138 | -10.9% |
| narrow_base3 | 8,868 | 18,098 | 7,744 | 125 | Sqrt | 0.0883 | -30.9% |
