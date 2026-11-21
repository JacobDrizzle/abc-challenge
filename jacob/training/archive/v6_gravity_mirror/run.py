"""Screen and confirm the v6 suite with desktop_next.py (same stages and flags), using the v6 model.

python -m experimental.v6_gravity_mirror.run --stage plan|screen|confirm|all [--variants ...]
"""
import sys

from experimental.v6_gravity_mirror.hooks import installed

HERE = "experimental/v6_gravity_mirror"
DEFAULTS = {"--output": "outputs/v6_group_a_v2",
            "--suite": f"{HERE}/configs/v6_run.json", "--base-config": f"{HERE}/configs/tcnn_v6.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    sys.argv = [sys.argv[0], *with_defaults(sys.argv[1:])]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
