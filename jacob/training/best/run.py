"""Screen a suite on the best model with desktop_next.py.

python -m best.run --stage plan|screen --output outputs/<new folder> [--suite ...] [--base-config ...] [--variants ...]

The suite defaults to best/configs/screen.json (the control only) on best/configs/base.json (k8_combined), so a
screen's control reproduces k8_combined's seed 42. --output is required: the frozen runner's own default is the v5
folder, and a new suite in an old folder would be refused or mixed with it. Only plan and screen are allowed: the
frozen runner's confirm stage would confirm the lowest-error variant, while confirm.py confirms the candidates the
team picks, against k8_combined.
"""
import sys

from best.hooks import installed

HERE = "best"
DEFAULTS = {"--suite": f"{HERE}/configs/screen.json", "--base-config": f"{HERE}/configs/base.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    argv = sys.argv[1:]
    stages = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--stage"]
    stages += [a.split("=", 1)[1] for a in argv if a.startswith("--stage=")]
    if any(stage not in ("plan", "screen") for stage in stages):
        raise SystemExit("best.run runs only --stage plan or screen; confirm.py confirms the candidates")
    if not any(a.split("=", 1)[0] == "--output" for a in argv):
        raise SystemExit("--output is required: name a new folder under outputs/, for example outputs/v10_<name>")
    sys.argv = [sys.argv[0], *with_defaults(argv)]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
