"""benchmark_latency.py (same flags) for a v6 checkpoint; only the streaming graph exists, so it is the default.

python -m experimental.v6_gravity_mirror.benchmark --checkpoint outputs/tcnn_v6/selected_model.pt
"""
import sys

from experimental.v6_gravity_mirror.hooks import installed


def main():
    import benchmark_latency
    if "--variants" not in sys.argv:
        sys.argv += ["--variants", "stream"]
    with installed():
        benchmark_latency.main()


if __name__ == "__main__":
    main()
