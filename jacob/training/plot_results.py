"""Draw performance figures for an existing evaluation folder; never rescores or edits results."""
import argparse
import json
import os
from pathlib import Path

from baseline_pipeline import EvaluationConfig
from tcnn.plots import experiment_files, render_evaluation, plot_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, help="Folder containing results.json from an evaluation")
    p.add_argument("--data", type=Path, help="Evaluated HDF5; enables speed and example figures")
    p.add_argument("--history", type=Path, nargs="*", default=[], help="history.json files to plot")
    p.add_argument("--selection", type=Path, help="selection.json; finds its seed histories and comparison")
    p.add_argument("--comparison", type=Path)
    p.add_argument("--checkpoint", type=Path, help="TCNN checkpoint for example forecasts")
    p.add_argument("--damped-selection", type=Path, help="Damped baseline selection for example forecasts")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, help="Figure folder (defaults to --eval-dir)")
    args = p.parse_args()
    histories, comparison = list(args.history), args.comparison
    if args.selection:
        found, found_comparison = experiment_files(args.selection)
        histories, comparison = histories or found, comparison or found_comparison
    if not args.eval_dir:
        if not histories:
            p.error("Give --eval-dir, --history or --selection")
        out = args.output or histories[0].parent
        print(plot_training(histories, out / "training_curves.png"))
        return
    examples = {}
    if args.checkpoint:
        from tcnn_model_adapter import TCNNPredictor
        examples["tiny_tcnn"] = TCNNPredictor(args.checkpoint, EvaluationConfig(), args.device)
    if args.damped_selection:
        from tcnn.damped import DampedVelocity
        examples["damped_velocity"] = DampedVelocity(EvaluationConfig(), **json.loads(args.damped_selection.read_text())["selected"])
    for path in render_evaluation(args.eval_dir, args.data, histories=histories, comparison=comparison,
                                  examples=examples or None, out_dir=args.output):
        print(path)


if __name__ == "__main__":
    main()
