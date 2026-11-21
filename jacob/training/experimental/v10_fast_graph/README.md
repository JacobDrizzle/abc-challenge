# v10_fast_graph: the fastest known streaming graph, built from a checkpoint

Written by Nic with AI help on 9 Nov 2026, for review. Nothing in `best/` is edited and no model is retrained.

## Question

On 6 Nov a chain of exact rewrites of the k8_combined ONNX file cut local CPU latency by about 30 % at identical
accuracy. It existed only as hand edits to one file in a sandbox, and the README listed "needs an export path" as
open. Can that graph be built directly from any checkpoint?

## Answer

`export.py` has `StreamingFast` and `export_fast_onnx(model, path, fold=True)`. It builds the same runtime graph
(42 nodes a call for k8_combined, 37 after ONNX Runtime's fusions) for any native-maps `LeanTCNN` layout, and for
`v10_narrow` models.

| Graph (Mac M5 Pro, ONNX Runtime CPU, 1 thread, one session, 15,000 calls each) | Nodes run | Median |
| --- | ---: | ---: |
| k8 reference, `best/export.py` lean | 55 | 31.0 us |
| k8 reference, A4 (`fused_head`, `padded_rows`) | 46 | 29.8 us |
| k8 hand-edited on 6 Nov (`wsp-v8`) | 37 | 22.5 us |
| **k8 from `export_fast_onnx`** | 37 | **22.9 us** |
| 2 blocks (kernel 16, dilations 1/25) from `export_fast_onnx` | 30 | 20.5 us |

Random weights except the hand-edited file; latency does not depend on the weights' values. Only numbers from one
session compare.

## What the graph does differently

Each step is exact algebra; `export.py`'s docstring has the detail.

1. **Front end in one MatMul.** Every input feature is a product of two terms that are linear in the 32 products of
   `[q[t-1], q[t]]` with `q[t]`. ReLU of the 13 products gives rate+, rate-, gravity+, gravity- and 1.
2. **2-D residual stream.** `[1, 64]` in (channel, stream) order; each block's mirror pointwise layer is one dense
   Gemm. A Gemm costs much less fixed time than a Conv on ONNX Runtime.
3. **No ReLU after the residual Add.** Both terms are already ReLU outputs.
4. **One Gemm for the tail.** Head, polynomial basis, damped prior and x biases are all linear in `[stream, r]`.
5. **No normalise.** The output is `now * (1, r/2)`, norm 1.00 to about 1.03. The organisers renormalise.

## Checked

`test_v10.py`, for five layouts and both file forms:

- Stream forecasts equal the PyTorch model's to under 0.001 degrees (measured: 7e-5, the same as the lean graph).
- Every node a call runs uses a safe-list operator. No Clip, Sub, Div or Sqrt.
- Interface: `context [1, 400, 4]`, `state_in_i`, `prediction [1, 20, 4]`, `state_out_i`, static shapes.

## Two file forms

| | `fold=True` | `fold=False` |
| --- | --- | --- |
| Numbers stored (k8) | 20,304 | 7,555 (the trainable parameters, each once) |
| Constant-only builder nodes | none | 122 |
| Latency on ONNX Runtime | same | same (folded when the session is created) |

Which to ship depends on how the organisers count parameters and whether their toolchain folds constants. That is
an open question for them.

## Open

- **State layout differs from the reference graph**: `[1, c, (k-1)d, 2]` instead of `[1, c, 2, (k-1)d]`. The harness
  only pipes states through, so this should not matter, but the model card must give the shapes.
- **Not timed on an NPU.** The Qualcomm AI Hub run so far is of the reference graph.
- **Real-weights check on the PC** (full validation through the graph) is still to do for the model that ships.
- A harness that refuses non-unit outputs needs the reference graph.
