"""evaluate_tcnn.py (same flags) for a v7 checkpoint: python -m experimental.v7_compact.evaluate ..."""
from experimental.v7_compact.hooks import installed


def main():
    import evaluate_tcnn
    with installed():
        evaluate_tcnn.main()


if __name__ == "__main__":
    main()
