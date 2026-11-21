"""Frozen TCNN adapter for the unchanged shared baseline evaluator."""
from pathlib import Path
import os

import numpy as np
import torch

from baseline_pipeline import sha256_file, unit_quaternions
from tcnn.model import TinyTCNN
from tcnn.runtime import configure_runtime


class TCNNPredictor:
    name = "tiny_tcnn"
    stateful = False

    def __init__(self, checkpoint, config, device="cpu"):
        runtime = configure_runtime()
        self.config, self.device = config, torch.device(device)
        checkpoint = Path(checkpoint).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format_version") not in (1, 2):
            raise ValueError("Unknown TCNN checkpoint format")
        if state["model_spec"]["context"] != config.context or state["model_spec"]["horizon"] != config.horizon:
            raise ValueError("Checkpoint context/horizon do not match evaluation")
        self.model = TinyTCNN(**state["model_spec"])
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        self.metadata = {"checkpoint_sha256": sha256_file(checkpoint),
                         "trainable_parameters": sum(p.numel() for p in self.model.parameters()),
                         "training_provenance": state["provenance"], "training_seed": state["seed"],
                         "validation_endpoint_deg": state["best_validation_deg"],
                         "device": str(self.device), "model_spec": state["model_spec"], "runtime": runtime}

    def predict_batch(self, histories_wxyz):
        histories_wxyz = unit_quaternions(histories_wxyz, source="TCNN history").astype(np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(histories_wxyz).to(self.device)
            return self.model(tensor).cpu().numpy()

    def predict(self, history_wxyz):
        return self.predict_batch(np.asarray(history_wxyz)[None])[0]

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def build_model(config):
    checkpoint = os.environ.get("ABC_TCNN_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set ABC_TCNN_CHECKPOINT or use evaluate_tcnn.py --checkpoint")
    return TCNNPredictor(checkpoint, config, os.environ.get("ABC_TCNN_DEVICE", "cpu"))
