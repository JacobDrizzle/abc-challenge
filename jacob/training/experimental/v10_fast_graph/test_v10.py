"""v10_fast_graph: the fast streaming graph against the PyTorch model, its operators, what the file stores, and
the layouts it accepts.

python -m unittest experimental.v10_fast_graph.test_v10 -v   (from jacob/training)
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

try:
    import onnx
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from tcnn.onnx_export import SAFE_OPS, geodesic_degrees
from tests.test_onnx import random_walk

from best import hooks as best_hooks
from best.model import LeanTCNN

from experimental.v10_fast_graph.export import export_fast_onnx
from experimental.v10_narrow.model import NarrowTCNN

ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT/"best/configs/base.json").read_text())
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *best_hooks.MODEL_KEYS)
K8 = {key: BASE[key] for key in MODEL_KEYS}
LAYOUTS = {"k8": {}, "two_blocks": dict(num_blocks=2, kernel_sizes=[16, 16], dilations=[1, 25]),
           "k5": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], channels=48),
           "k8_last_8": dict(block_inputs=[32, 32, 8]),
           "two_blocks_last_8": dict(num_blocks=2, kernel_sizes=[16, 16], dilations=[1, 25], block_inputs=[32, 8])}
BUILDERS = {"Unsqueeze", "Transpose"}  # extra operators of the tied file's constant-only nodes


def build(**overrides):
    torch.manual_seed(0)
    model = NarrowTCNN(**{**K8, **overrides})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)
        model.head.weight.mul_(.2)  # forecast rotations of a few degrees, as a trained model's
    return model.eval()


def aligned_stream(length, seed):
    """A random walk with continuous signs, as the organisers' streams are."""
    q = random_walk(1, length, seed)[0]
    flips = np.cumprod(np.sign((q[1:] * q[:-1]).sum(-1)))
    return (q * np.concatenate(([1.], flips))[:, None]).astype(np.float32)


def stream_predictions(path, shapes, stream):
    """[steps, 20, 4] predictions of a streaming graph fed one sample a call, from the first full window on."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    states, predictions = [np.zeros(s, np.float32) for s in shapes], []
    for t in range(1, len(stream)):
        window = stream[max(0, t - 399):t + 1]
        window = np.concatenate((np.repeat(window[:1], 400 - len(window), 0), window))[None]
        outputs = session.run(None, {"context": window, **dict(zip(names[1:], states))})
        states = outputs[1:]
        if t >= 399:
            predictions.append(outputs[0])
    return np.concatenate(predictions)


def runtime_nodes(path):
    """The nodes that depend on a graph input: the ones a runtime that folds constants runs on every call."""
    graph = onnx.load(str(path)).graph
    live, nodes = {i.name for i in graph.input}, []
    for node in graph.node:
        if any(name in live for name in node.input):
            live.update(node.output)
            nodes.append(node)
    return nodes


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class FastGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_graph_is_the_model(self):
        stream = aligned_stream(900, 3)
        windows = torch.from_numpy(np.stack([stream[t - 399:t + 1] for t in range(399, 900)]))
        for name, layout in LAYOUTS.items():
            model = build(**layout)
            with torch.no_grad():
                expected = model(windows).numpy()
            for fold in (True, False):
                with self.subTest(name, fold=fold):
                    path, shapes = export_fast_onnx(model, self.directory/f"{name}_{fold}.onnx", fold=fold)
                    predictions = stream_predictions(path, shapes, stream)
                    self.assertEqual(predictions.shape, (501, 20, 4))
                    self.assertLess(geodesic_degrees(predictions, expected).max(), 1e-3)
                    # (1, r / 2): never shorter than a unit quaternion, and close to one for real rotations.
                    norms = np.linalg.norm(predictions, axis=-1)
                    self.assertGreater(norms.min(), 1 - 1e-5)
                    self.assertLess(norms.max(), 1.5)

    def test_interface_and_operators(self):
        for name, layout in LAYOUTS.items():
            model = build(**layout)
            for fold in (True, False):
                with self.subTest(name, fold=fold):
                    path, shapes = export_fast_onnx(model, self.directory/f"{name}_{fold}.onnx", fold=fold)
                    graph = onnx.load(str(path)).graph
                    dims = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
                    self.assertEqual([(i.name, dims(i)) for i in graph.input],
                                     [("context", [1, 400, 4])] + [(f"state_in_{i}", list(s)) for i, s in enumerate(shapes)])
                    self.assertEqual([(o.name, dims(o)) for o in graph.output],
                                     [("prediction", [1, 20, 4])] + [(f"state_out_{i}", list(s)) for i, s in enumerate(shapes)])
                    inputs = getattr(model, "block_inputs", None) or [model.stream_channels] * len(model.blocks)
                    self.assertEqual(shapes, [(1, c, block.left_padding, 2) for c, block in zip(inputs, model.blocks)])
                    # Every node a call runs uses a safe-list operator; Clip is on that list but is not used.
                    self.assertLessEqual({n.op_type for n in runtime_nodes(path)}, SAFE_OPS - {"Clip"})
                    self.assertLessEqual({n.op_type for n in graph.node} - {"Constant"}, SAFE_OPS | (set() if fold else BUILDERS))
                    stored = sum(int(np.prod(t.dims)) for t in graph.initializer)
                    if fold:  # no builder nodes: everything left depends on the inputs
                        self.assertEqual(len(runtime_nodes(path)), sum(n.op_type != "Constant" for n in graph.node))
                    else:  # the learned weights, each once
                        self.assertEqual(stored, sum(p.numel() for p in model.parameters()))

    def test_k8_runs_42_nodes_a_call(self):
        path, _ = export_fast_onnx(build(), self.directory/"k8.onnx")
        self.assertEqual(len(runtime_nodes(path)), 42)
        tied, _ = export_fast_onnx(build(), self.directory/"k8_tied.onnx", fold=False)
        self.assertEqual(len(runtime_nodes(tied)), 42)

    def test_other_models_are_refused(self):
        for bad in ({"maps": "lean"}, {"gravity_features": False}, {"speed_feature": "norm"}, {"head_basis": None},
                    {"damped_prior_tau_ms": None}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                export_fast_onnx(LeanTCNN(**{**K8, **bad}), self.directory/"bad.onnx")


if __name__ == "__main__":
    unittest.main()
