"""Working adapter example. This repeats the last pose; it is NOT a trained model.

Copy this file for a learned model, replacing predict() with inference only.
"""
import numpy as np


class ExampleModel:
    name = "example_model"
    stateful = False
    metadata = {"trainable_parameters": 0, "description": "Adapter demonstration only: identical to hold-last"}

    def __init__(self, config):
        self.config = config

    def predict(self, history_wxyz):
        # history_wxyz is a float32 [context, 4] array. Future labels are absent.
        # Replace with your network's inference, returning [horizon, 4] in wxyz.
        return np.repeat(history_wxyz[-1:], self.config.horizon, axis=0)

    def predict_batch(self, histories_wxyz):
        # Optional for a STATELESS model: [batch, context, 4] -> [batch, horizon, 4].
        return np.repeat(histories_wxyz[:, -1:, :], self.config.horizon, axis=1)


def build_model(config):
    # Load your frozen checkpoint here once, not inside predict().
    # Include its SHA-256, training split, and parameter count in metadata.
    return ExampleModel(config)
