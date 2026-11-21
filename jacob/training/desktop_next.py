"""Explain desktop experiment progress without changing frozen training code."""
import argparse
from contextlib import contextmanager, redirect_stdout
import json
from pathlib import Path
import re
import sys

import run_next

ROOT = Path(__file__).resolve().parent


class ProgressOutput:
    def __init__(self, stream, label):
        self.stream, self.label, self.pending, self.seed = stream, label, "", None

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = re.search(r"Training seed (\d+) on", line)
            if match:
                self.seed = match.group(1)
            suffix = f" | seed {self.seed}" if self.seed else ""
            self.stream.write(f"[{self.label}{suffix}] {line}\n")
        return len(text)

    def flush(self):
        self.stream.flush()

    def finish(self):
        if self.pending:
            self.write("\n")
        self.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def display_progress(configs):
    original = run_next.run_experiment
    names = list(configs)

    def described_run(data, split, output, config, device="cuda", **kwargs):
        output = Path(output)
        name = output.name
        selection_name = kwargs.get("selection_name", "selection.json")
        screening = selection_name == "screen_selection.json"
        label = (f"Stage 1/2 | variant {names.index(name)+1}/{len(names)} | {name}" if screening
                 else f"Stage 2/2 | confirm {name}")
        print(f"\n{label}", flush=True)
        if (output/selection_name).exists():
            print("Verifying and reusing the saved validation selection; no retraining.", flush=True)
        elif screening:
            history = output/f"seed_{config['seeds'][0]}"/"history.json"
            if history.exists():
                rows = json.loads(history.read_text())
                print(f"Saved through epoch {rows[-1]['epoch']}; resuming from the saved checkpoint.", flush=True)
            else:
                print(f"New configuration, using screening seed {config['seeds'][0]}.", flush=True)
        else:
            print(f"Confirming seeds {config['seeds']}; completed training is reused.", flush=True)
        stream = ProgressOutput(sys.stdout, label)
        try:
            with redirect_stdout(stream):
                return original(data, split, output, config, device, **kwargs)
        finally:
            stream.finish()

    run_next.run_experiment = described_run
    try:
        yield
    finally:
        run_next.run_experiment = original


def main():
    # The underlying runner remains the source of CLI validation and computation.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", type=Path, default=ROOT/"configs/v5_run.json")
    parser.add_argument("--base-config", type=Path, default=ROOT/"configs/tcnn_v5.json")
    parser.add_argument("--recipe-from", type=Path)
    parser.add_argument("--stage", default="plan")
    args, _ = parser.parse_known_args()
    if args.stage == "plan" or any(a in sys.argv for a in ("-h", "--help")):
        run_next.main()
        return
    suite = json.loads(args.suite.read_text())
    configs = run_next.experiment_configs(suite, json.loads(args.base_config.read_text()),
                json.loads(args.recipe_from.read_text()) if args.recipe_from else None)
    print(f"Working folder: {ROOT}", flush=True)
    print(f"Stage 1 compares {len(configs)} configurations with seed {suite['seeds'][0]}.", flush=True)
    print(f"Stage 2 confirms the winning configuration with seeds {suite['seeds']}.", flush=True)
    print("Repeating seed 42 across different configurations is intentional. Saved work is reused.", flush=True)
    with display_progress(configs):
        run_next.main()


if __name__ == "__main__":
    main()
