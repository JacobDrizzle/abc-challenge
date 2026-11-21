"""Screen the v7 suite with desktop_next.py, using the v7 model.

python -m experimental.v7_compact.run --stage plan|screen [--variants ...]

Only plan and screen are allowed: the frozen runner's confirm stage would confirm the lowest-error variant,
while v7 confirms the smallest variant within the accuracy tolerance, with the add-ons folded in (plan rule).
"""
import sys

from experimental.v7_compact.hooks import installed

HERE = "experimental/v7_compact"
DEFAULTS = {"--output": "outputs/v7_group_a_v2",
            "--suite": f"{HERE}/configs/v7_run.json", "--base-config": f"{HERE}/configs/tcnn_v7.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    argv = sys.argv[1:]
    stages = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--stage"]
    stages += [a.split("=", 1)[1] for a in argv if a.startswith("--stage=")]
    if any(stage not in ("plan", "screen") for stage in stages):
        raise SystemExit("v7 runs only --stage plan or screen; confirmation follows the plan's tolerance rule")
    sys.argv = [sys.argv[0], *with_defaults(argv)]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
