"""Screen a suite with the fast forward: best/run.py with this package's hooks and suite.

python -m experimental.v10_fast.run --stage plan|screen --output outputs/<new folder> [--data ...] [--suite ...]
    [--base-config ...] [--variants ...]

The suite defaults to experimental/v10_fast/configs/screen_800.json (written for the 800-user file: pass it with
--data), on best/configs/base.json (k8_combined). A control trained here is not bit-identical to one trained by
best/run.py, so compare a variant with the control of the same suite. --output is required, and only plan and screen
are allowed, as in best/run.py; confirm.py in this folder confirms the candidates the team picks.
"""
import sys

from experimental.v10_fast.hooks import installed

HERE = "experimental/v10_fast"
DEFAULTS = {"--suite": f"{HERE}/configs/screen_800.json", "--base-config": "best/configs/base.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    argv = sys.argv[1:]
    stages = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--stage"]
    stages += [a.split("=", 1)[1] for a in argv if a.startswith("--stage=")]
    if any(stage not in ("plan", "screen") for stage in stages):
        raise SystemExit("v10_fast.run runs only --stage plan or screen; v10_fast/confirm.py confirms the candidates")
    if not any(a.split("=", 1)[0] == "--output" for a in argv):
        raise SystemExit("--output is required: name a new folder under outputs/, for example outputs/v10_fast_800")
    sys.argv = [sys.argv[0], *with_defaults(argv)]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
