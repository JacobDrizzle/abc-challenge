"""Screen a v9 suite with desktop_next.py, using the v9 model and recipe options.

python -m experimental.v9_lean.run --stage plan|screen [--suite ...] [--output ...] [--variants ...]

Only plan and screen are allowed: the frozen runner's confirm stage would confirm the lowest-error variant, while v9
confirms a combined candidate (confirm.py, plan rule of 28 Oct).
"""
import sys

from experimental.v9_lean.hooks import installed

HERE = "experimental/v9_lean"
DEFAULTS = {"--output": "outputs/v9_screen",
            "--suite": f"{HERE}/configs/v9_screen.json", "--base-config": f"{HERE}/configs/tcnn_v9.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    argv = sys.argv[1:]
    stages = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--stage"]
    stages += [a.split("=", 1)[1] for a in argv if a.startswith("--stage=")]
    if any(stage not in ("plan", "screen") for stage in stages):
        raise SystemExit("v9 runs only --stage plan or screen; confirm.py confirms the combined candidate")
    sys.argv = [sys.argv[0], *with_defaults(argv)]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
