"""Screen a v10_users suite with desktop_next.py: best/run.py with this package's hooks and suite.

python -m experimental.v10_users.run --stage plan|screen --output outputs/<new folder> [--suite ...] [--variants ...]

The suite defaults to experimental/v10_users/configs/screen.json, on best/configs/base.json (k8_combined). Its
control is k8_combined, which outputs/v9_confirm/k8_combined/seed_42 already holds, so a screen passes --variants to
train only the new variants. --output is required, and only plan and screen are allowed, as in best/run.py.
"""
import sys

from experimental.v10_users.hooks import installed

HERE = "experimental/v10_users"
DEFAULTS = {"--suite": f"{HERE}/configs/screen.json", "--base-config": "best/configs/base.json"}


def with_defaults(argv):
    given = {a.split("=", 1)[0] for a in argv}
    return argv + [x for flag, value in DEFAULTS.items() if flag not in given for x in (flag, value)]


def main():
    import desktop_next
    argv = sys.argv[1:]
    stages = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--stage"]
    stages += [a.split("=", 1)[1] for a in argv if a.startswith("--stage=")]
    if any(stage not in ("plan", "screen") for stage in stages):
        raise SystemExit("v10_users.run runs only --stage plan or screen; best/confirm.py confirms the candidates")
    if not any(a.split("=", 1)[0] == "--output" for a in argv):
        raise SystemExit("--output is required: name a new folder under outputs/, for example outputs/v10_users")
    sys.argv = [sys.argv[0], *with_defaults(argv)]
    with installed():
        desktop_next.main()


if __name__ == "__main__":
    main()
