"""benchmark_latency.py (same flags) for a best-model checkpoint, or any group-a-v2 one since v7, with the streaming graph.

python -m best.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--padded-rows]
    [--fused-head] [--split-mirror] [--st-layout]
An exact-map checkpoint is benchmarked as deployed(model), the same weights with lean maps: the parity check compares
the graph with that model. --fused-head exports A4 (one Conv head), --padded-rows (3, 1) pointwise kernels without the
row Slice and Concat (both store each weight once and rely on ORT folding constants at load), --split-mirror the
sum and difference pointwise Convs at half the pointwise MACs, and --st-layout the output LayerNormalization on
[1, 1, 20, 8], the shape ST Edge AI Core 4.1 compiles correctly (export.py).
"""
import functools
import sys

from best import export
from best.hooks import _swapped, installed


def main():
    import benchmark_latency
    if "--variants" not in sys.argv:
        sys.argv += ["--variants", "stream"]
    options = {}
    for flag, option in (("--fused-head", "fused_head"), ("--padded-rows", "padded_rows"),
                         ("--split-mirror", "split_mirror"), ("--st-layout", "st_layout")):
        options[option] = flag in sys.argv
        if options[option]:
            sys.argv.remove(flag)  # benchmark_latency's own parser does not know it
    # installed() binds export.export_streaming_onnx; the inner swap adds the options and is undone first.
    with installed(), _swapped(benchmark_latency, export_streaming_onnx=functools.partial(
            export.export_streaming_onnx, **options)):
        benchmark_latency.main()


if __name__ == "__main__":
    main()
