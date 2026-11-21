"""benchmark_latency.py (same flags) for a v9, v7 or v6 checkpoint, exported with the v9 graph (Track A).

python -m experimental.v9_lean.benchmark --checkpoint outputs/v9_screen/<variant>/screen_model.pt
An exact-map checkpoint is benchmarked as deployed(model), the same weights with lean maps: the parity check compares
the graph with that model. Add --fused-head to export A4 (one Conv head) and --padded-rows for (3, 1) pointwise
kernels without the row Slice and Concat; both store each weight once and rely on ORT folding constants at load.
"""
import functools
import sys

from experimental.v6_gravity_mirror.hooks import _swapped
from experimental.v9_lean import export
from experimental.v9_lean.hooks import installed


def main():
    import benchmark_latency
    if "--variants" not in sys.argv:
        sys.argv += ["--variants", "stream"]
    options = {}
    for flag, option in (("--fused-head", "fused_head"), ("--padded-rows", "padded_rows")):
        options[option] = flag in sys.argv
        if options[option]:
            sys.argv.remove(flag)  # benchmark_latency's own parser does not know it
    # installed() binds export.export_streaming_onnx; the inner swap adds the options and is undone first.
    with installed(), _swapped(benchmark_latency, export_streaming_onnx=functools.partial(
            export.export_streaming_onnx, **options)):
        benchmark_latency.main()


if __name__ == "__main__":
    main()
