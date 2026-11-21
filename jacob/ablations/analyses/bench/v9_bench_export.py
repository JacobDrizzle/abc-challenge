"""Export the v9 models for the team bench: jacob/training/outputs/v9_bench/<name>/model_fp32_stream.onnx.

Each graph is the v9 streaming export of the deployed model (lean maps), checked for stream parity against that
PyTorch model on 1000 windows of the longest validation sessions, as final_latency.py did. Writes manifest.json with
the declared parameter count, SHA-256 and parity of each file. CPU only; reads checkpoints, never trains.
Usage (from jacob/training): python <this script>
"""
import hashlib
import json
import sys
from pathlib import Path

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from benchmark_latency import Streamer, ort_session, stream_parity, validation_segments  # noqa: E402
from experimental.v9_lean.export import export_streaming_onnx  # noqa: E402
from experimental.v9_lean.model import deployed, load_checkpoint  # noqa: E402

OUT = TRAINING / "outputs" / "v9_bench"
LEAN, PADDED, A4 = {}, dict(padded_rows=True), dict(padded_rows=True, fused_head=True)
# name -> (checkpoint, export options, note)
MODELS = {
    "v9_sep_angular_lean_ga2": ("outputs/v8_recipe_sep_base3_addons/angular_loss/selected_model.pt", LEAN,
                                "v7_sep_base3_addons_angular_ga2 weights (seed 44) on the v9 lean graph"),
    "v9_k8_combined_ga2": ("outputs/v9_confirm/k8_combined/selected_model.pt", LEAN, "k8_combined seed 42, lean graph"),
    "v9_k8_combined_padded_ga2": ("outputs/v9_confirm/k8_combined/selected_model.pt", PADDED,
                                  "k8_combined seed 42, padded rows"),
    "v9_k8_combined_padded_a4_ga2": ("outputs/v9_confirm/k8_combined/selected_model.pt", A4,
                                     "k8_combined seed 42, padded rows + fused head"),
    "v9_k5_combined_ga2": ("outputs/v9_confirm/k5_combined/selected_model.pt", LEAN, "k5_combined seed 42, lean graph"),
    "v9_k5_combined_padded_ga2": ("outputs/v9_confirm/k5_combined/selected_model.pt", PADDED,
                                  "k5_combined seed 42, padded rows"),
    "v9_k5_combined_padded_a4_ga2": ("outputs/v9_confirm/k5_combined/selected_model.pt", A4,
                                     "k5_combined seed 42, padded rows + fused head"),
    "v9_k8_jitter3_s42_ga2": ("outputs/v9_jitter/jitter_3ms/screen_model.pt", LEAN,
                              "k8 screen recipe + jitter augmentation 3 ms, seed 42 screen run (hedge, not confirmed)"),
}


def main():
    segments = validation_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), 420)
    manifest = {}
    for name, (checkpoint, options, note) in MODELS.items():
        model = deployed(load_checkpoint(TRAINING / checkpoint)[0])
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        path, shapes = export_streaming_onnx(model, folder / "model_fp32_stream.onnx", **options)
        check = stream_parity(model, Streamer(ort_session(path, 1), shapes), segments, 400, 20, 1000)
        manifest[name] = {"onnx": str(Path(path).relative_to(TRAINING.parents[1])), "checkpoint": checkpoint,
                          "options": options, "note": note,
                          "declared_params": sum(p.numel() for p in model.parameters()),
                          "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          "parity_max_deg": check["max_difference_deg"]}
        print(name, {k: manifest[name][k] for k in ("declared_params", "parity_max_deg")}, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
