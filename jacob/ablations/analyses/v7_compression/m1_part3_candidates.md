| Graph | Params | MACs/step | State floats | Nodes | Off-list | Median (ms) | p90 (ms) | Round medians spread (ms) | vs v6 graph |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| control (v6 export) | 26,012 | 52,144 | 12,192 | 174 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | 0.1231 | 0.1391 | 0.0008 | +0.0% |
| control | 26,012 | 52,194 | 12,192 | 145 | Sqrt | 0.1068 | 0.1151 | 0.0003 | -13.2% |
| base3_5 | 19,052 | 38,370 | 11,616 | 125 | Sqrt | 0.0900 | 0.0972 | 0.0002 | -26.9% |
| separable_24 | 10,388 | 20,946 | 12,192 | 152 | Sqrt | 0.1075 | 0.1136 | 0.0006 | -12.7% |
| narrow_base3 | 8,868 | 18,098 | 7,744 | 125 | Sqrt | 0.0857 | 0.0940 | 0.0005 | -30.4% |
| separable_base3 | 7,892 | 16,050 | 11,616 | 130 | Sqrt | 0.0892 | 0.0981 | 0.0007 | -27.5% |
| base3_5_addons | 17,811 | 36,183 | 11,616 | 125 | none | 0.0908 | 0.0989 | 0.0014 | -26.2% |
| separable_24_addons | 9,147 | 18,759 | 12,192 | 152 | none | 0.1080 | 0.1155 | 0.0023 | -12.3% |
| narrow_full_addons | 11,139 | 22,871 | 8,128 | 145 | none | 0.1011 | 0.1095 | 0.0004 | -17.9% |
| separable_base3_addons | 6,651 | 13,863 | 11,616 | 130 | none | 0.0889 | 0.0957 | 0.0006 | -27.8% |

Stream parity, separable_base3: max 5.40e-06 deg over 251 windows
Stream parity, separable_base3_addons: max 4.52e-06 deg over 251 windows
