"""Collect every training run in jacob/training/outputs/ into this folder, so the ablation record survives without it.

python jacob/ablations/harvest.py   (the training venv: it loads checkpoints to count parameters)

outputs/ is gitignored and lives on one PC. This script writes:
- runs.csv: one row per suite, variant and seed. The validation error is the full stride-1 step-20 mean (degrees)
  that seed selection used. `overrides` are the suite's or confirm plan's changes to that suite's own base config,
  not to k8_combined. `code` is the experiment package the run hashed: runs made with experimental/v6-v9 code are
  at git tag v9-ablations.
- results/<suite>/: the suite's JSON, Markdown and scripts, and each seed's history.json, config.json and
  complete.json. Checkpoints, ONNX graphs, NumPy arrays, figures, per-window validation files and test/results.json
  are left out.
- results/logs/: the run logs.
Re-run it after each new suite. It only adds or updates files, and a run with nothing new changes nothing. Rows of
suites missing from this machine's outputs/ are kept, and it refuses to run without outputs/.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent/"training"
OUTPUTS = TRAINING/"outputs"
sys.path.insert(0, str(TRAINING))

COLUMNS = ["suite", "variant", "stage", "seed", "selected", "val_step20_deg", "params", "best_epoch", "epochs_run",
           "overrides", "code", "finished", "checkpoint_sha256"]
SEED_FILES = {"history.json", "config.json", "complete.json"}
TEST_FILES = {"report.md", "figures.md", "test_receipt.json"}
SKIP_FOLDERS = {"__pycache__", "full_validation", "graphs_random", "graphs_trained"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def confirm_overrides():
    """Each confirmed folder's overrides, from every confirmation.json plan: {"<suite>/<variant>": overrides}."""
    found = {}
    for path in sorted(OUTPUTS.glob("*/confirmation.json")):
        for candidate in read(path)["plan"]["candidates"].values():
            found[candidate["output"].removeprefix("outputs/")] = candidate["overrides"]
    return found


def code_package(experiment):
    """The experiment package whose files the run hashed (its hooks add them to the hashed list), or v5."""
    packages = {"/".join(Path(name).parts[:-1]) for name in experiment["code_sha256"]
                if Path(name).parts[0] in ("experimental", "best")}
    return ", ".join(sorted(packages)) or "v5 (train_tcnn)"


def parameters(checkpoint):
    from best.model import load_checkpoint
    model, _ = load_checkpoint(checkpoint)
    return sum(p.numel() for p in model.parameters())


def run_rows():
    overrides, rows = confirm_overrides(), []
    for experiment_path in sorted(OUTPUTS.glob("*/*/experiment.json")):
        folder = experiment_path.parent
        suite, variant = folder.parent.name, folder.name
        selections = [read(folder/name) for name in ("selection.json", "screen_selection.json") if (folder/name).exists()]
        if not selections:
            continue  # no seed finished
        suite_file = folder.parent/"suite.json"
        changes = (read(suite_file)["suite"]["variants"].get(variant) if suite_file.exists() else None)
        if changes is None:
            changes = overrides.get(f"{suite}/{variant}", {})
        # A confirmed folder's selection.json covers every seed; a screen's screen_selection.json only seed 42.
        selected, stage = selections[0]["selected_seed"], "confirm" if (folder/"selection.json").exists() else "screen"
        scores = {c["seed"]: c for s in reversed(selections) for c in s["validation_candidates"]}
        code = code_package(read(experiment_path))
        for seed, candidate in sorted(scores.items()):
            run = folder/f"seed_{seed}"
            history = read(run/"history.json")
            rows.append({"suite": suite, "variant": variant, "stage": stage, "seed": seed,
                         "selected": int(seed == selected), "val_step20_deg": f"{candidate['validation_endpoint_deg']:.6f}",
                         "params": parameters(run/"best.pt"), "best_epoch": history[-1]["best_epoch"],
                         "epochs_run": len(history), "overrides": json.dumps(changes, sort_keys=True), "code": code,
                         "finished": datetime.fromtimestamp((run/"complete.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                         "checkpoint_sha256": candidate["checkpoint_sha256"]})
    return rows


def wanted(path):
    """Whether a file under outputs/ belongs in results/."""
    parts = path.relative_to(OUTPUTS).parts
    if SKIP_FOLDERS & set(parts[:-1]):
        return False
    if any(part.startswith("seed_") for part in parts[:-1]):
        return path.name in SEED_FILES
    if "test" in parts[:-1]:
        return path.name in TEST_FILES
    return path.suffix in (".json", ".md", ".py")


def copy(source, target):
    """Copy if new or changed; returns 1 if it wrote."""
    if target.exists() and target.read_bytes() == source.read_bytes():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return 1


def main():
    if not OUTPUTS.is_dir():
        raise SystemExit(f"{OUTPUTS} does not exist: run this on the machine that holds the training outputs")
    rows = run_rows()
    # Keep the rows of suites this machine doesn't hold, so a partial outputs/ never drops runs from the record.
    found = {row["suite"] for row in rows}
    if (HERE/"runs.csv").exists():
        with (HERE/"runs.csv").open(encoding="utf-8", newline="") as handle:
            rows += [row for row in csv.DictReader(handle) if row["suite"] not in found]
    rows.sort(key=lambda row: (row["suite"], row["variant"], int(row["seed"])))
    with (HERE/"runs.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    written = copy(HERE/"runs.csv.tmp", HERE/"runs.csv")
    (HERE/"runs.csv.tmp").unlink()
    for path in sorted(OUTPUTS.rglob("*")):
        if path.is_file() and path.parent != OUTPUTS and wanted(path):
            written += copy(path, HERE/"results"/path.relative_to(OUTPUTS))
    for path in sorted(OUTPUTS.glob("*.log")):
        written += copy(path, HERE/"results/logs"/path.name)
    digest = hashlib.sha256((HERE/"runs.csv").read_bytes()).hexdigest()[:12]
    print(f"{len(rows)} seed runs in {len({(r['suite'], r['variant']) for r in rows})} variants; "
          f"{written} files written; runs.csv {digest}")


if __name__ == "__main__":
    main()
