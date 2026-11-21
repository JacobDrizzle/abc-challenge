# v10_narrow: a block that reads fewer channels than it writes

Written by Nic with AI help on 9 Nov 2026, for review. Built on `v10_speed`; nothing in `best/` is edited.

## Question

The last block of k8_combined keeps its last 329 inputs for all 64 channels: 21,056 floats, 84 % of the model's
streaming state. Every call copies it twice, about 27 % of the fast graph's time on the Mac (9 Nov profile). No
exact rewrite removes that. Does long-range context need all 32 channels a stream?

## Change

One config key, `block_inputs`: `null`, or one integer per block. A block with fewer inputs than channels filters
only the first `c` channels of each stream and still updates all of them. Mirror equivariance is kept (tested).

## Latency and size (Mac M5 Pro, ONNX Runtime CPU, 1 thread, one session, fast graph, random weights)

| Layout | `block_inputs` | Parameters | State floats | Median |
| --- | --- | ---: | ---: | ---: |
| k8 (1/8/47) | all 32 | 7,555 | 25,088 | 23.3 us |
| k8 | 32, 32, 16 | 6,403 | 14,560 | 20.5 us |
| k8 | 32, 32, 8 | 5,827 | 9,296 | 19.3 us |
| k8 | 32, 32, 4 | 5,539 | 6,664 | 18.7 us |
| 2 blocks (kernel 16, 1/25) | all 32 | 5,731 | 24,960 | 20.7 us |
| 2 blocks | 32, 16 | 4,451 | 12,960 | 17.0 us |
| 2 blocks | 32, 8 | 3,811 | 6,960 | 14.9 us |
| 2 blocks | 32, 4 | 3,491 | 3,960 | 14.0 us |

For scale: today's shipped graph (k8 reference, lean) ran at 31.3 us in the same session.

## Accuracy

In progress (9 Nov): 1,600 users, 126k steps, seed 42, full validation on 240 users. See the results table Nic
keeps with the run summaries.

## Run

```
python -m unittest experimental.v10_narrow.test_v10 -v
```
