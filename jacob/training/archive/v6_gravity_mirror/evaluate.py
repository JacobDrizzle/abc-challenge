"""evaluate_tcnn.py (same flags) for a v6 checkpoint: python -m experimental.v6_gravity_mirror.evaluate ..."""
from experimental.v6_gravity_mirror.hooks import installed


def main():
    import evaluate_tcnn
    with installed():
        evaluate_tcnn.main()


if __name__ == "__main__":
    main()
