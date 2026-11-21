"""benchmark_latency.py (same flags) for a v7 or v6 checkpoint, exported with the v7 streaming graph.

python -m experimental.v7_compact.benchmark --checkpoint outputs/v7_group_a_v2/<variant>/screen_model.pt
Add --mirror-split to export mirror_split's graph instead: the same outputs at half the block conv MACs.
"""
from contextlib import nullcontext
import sys

from experimental.v6_gravity_mirror.hooks import _swapped
from experimental.v7_compact import mirror_split
from experimental.v7_compact.hooks import installed


def main():
    import benchmark_latency
    if "--variants" not in sys.argv:
        sys.argv += ["--variants", "stream"]
    split = "--mirror-split" in sys.argv
    if split:
        sys.argv.remove("--mirror-split")  # benchmark_latency's own parser does not know it
    # installed() binds export.export_streaming_onnx; the inner swap replaces that and is undone first.
    with installed(), (_swapped(benchmark_latency, export_streaming_onnx=mirror_split.export_streaming_onnx)
                       if split else nullcontext()):
        benchmark_latency.main()


if __name__ == "__main__":
    main()
