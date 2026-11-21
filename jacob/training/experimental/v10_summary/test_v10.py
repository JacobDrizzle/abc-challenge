"""v10_summary: the config key, the moving-average weights, SummaryBlock against its streaming recursion,
SummaryTCNN against LeanTCNN and against its own streamed forecasts, the mirror, gradients, a tiny training run, and
the streaming ONNX graph against SummaryTCNN.streamed.

python -m unittest experimental.v10_summary.test_v10 -v   (from jacob/training)
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import run_desktop
from smoke_tcnn import make_fixture
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories
import train_tcnn

from best import hooks as best_hooks
from best import model as best_model
from best.configuration import normalize_config as normalize_best
from best.model import LeanTCNN, mirror

from experimental.v10_summary import hooks
from experimental.v10_summary.export import export_streaming_onnx
from experimental.v10_summary.configuration import normalize_config
from experimental.v10_summary.model import SummaryBlock, SummaryTCNN, ema_weights, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT/"best/configs/base.json").read_text())
SCREEN = json.loads((ROOT/"experimental/v10_summary/configs/screen.json").read_text())
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *best_hooks.MODEL_KEYS)
K8 = {key: BASE[key] for key in MODEL_KEYS}
TAUS = SCREEN["variants"]["summary_tau4_64"]["summary_taus"]
TOLERANCE = 1e-5


def build(**overrides):
    torch.manual_seed(0)
    model = SummaryTCNN(**{**K8, "dropout": 0, **overrides})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)
    return model.eval()


def histories():
    return torch.from_numpy(np.concatenate((smooth_histories(6), random_walk(6, 400, 5))))


def degrees(a, b):
    """Rotation angle between unit quaternions, 2 acos|a.b|, in float64 (atan2 form, accurate near 0; acos of a float32
    dot product is not). atan2(|a - b|, |a + b|) is half of acos|a.b|, hence the 4."""
    a, b = a.double(), b.double()
    sign = torch.sign((a * b).sum(-1, keepdim=True))
    return torch.rad2deg(4 * torch.atan2((a - sign * b).norm(dim=-1), (a + sign * b).norm(dim=-1)))


class ConfigTests(unittest.TestCase):
    def test_default_is_best(self):
        self.assertEqual(normalize_config(BASE), {**normalize_best(BASE), "summary_taus": None})
        self.assertEqual(normalize_config({**BASE, "summary_taus": TAUS})["summary_taus"], TAUS)

    def test_key_is_checked(self):
        for bad in ({"summary_taus": []}, {"summary_taus": 16}, {"summary_taus": [16, 0]}, {"summary_taus": [-4]},
                    {"summary_taus": [16, True]}, {"summary_taus": [float("inf")]}, {"summary_taus": ["16"]},
                    {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**BASE, **bad})
        with self.assertRaises(ValueError):
            normalize_config({**BASE, "separable": False, "kernel_sizes": None, "dilations": None,
                              "summary_taus": TAUS})

    def test_screen_variants_change_only_their_key(self):
        for name, overrides in SCREEN["variants"].items():
            with self.subTest(name):
                self.assertLessEqual(set(overrides), {"summary_taus"})


class WeightTests(unittest.TestCase):
    def test_weights_are_renormalised_averages_from_start(self):
        weights = ema_weights([4., 64.], 400, 64)
        self.assertEqual(tuple(weights.shape), (400, 2))
        self.assertTrue(torch.equal(weights[:64], torch.zeros(64, 2)))
        self.assertTrue(torch.allclose(weights.sum(0), torch.ones(2)))
        a = np.exp(-1 / 4.)
        self.assertAlmostEqual((weights[-2, 0] / weights[-1, 0]).item(), a, places=5)  # one step back: times a
        with self.assertRaises(ValueError):
            ema_weights([4.], 64, 64)


class BlockTests(unittest.TestCase):
    def block(self, taus, start=64):
        torch.manual_seed(1)
        block = SummaryBlock(32, 2, 0., taus, start)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(0, .1)
        return block.eval()

    def test_the_window_forward_is_the_streaming_recursion(self):
        """Fed one long input, the recursion from zero and the window forward over its last 400 positions agree to
        float rounding when the averages forget faster than the 336 valid positions; with tau 64 they differ by the
        0.5 % of weight a stream gives to inputs older than the window."""
        x = torch.rand(3, 64, 1500)
        for taus, bound in (([1., 2., 4.], 1e-5), ([4., 16., 64.], 1e-2)):
            block = self.block(taus)
            state = torch.zeros(3, 64, len(taus))
            with torch.no_grad():
                for t in range(1500):
                    streamed, state = block.step(x[:, :, t], state)
                windowed = block(x[:, :, -400:])[..., 0]
            with self.subTest(taus=taus):
                scale = streamed.abs().max().item()
                self.assertLess((windowed - streamed).abs().max().item(), bound * scale)

    def test_inputs_before_start_are_not_read(self):
        block, x = self.block([4., 64.]), torch.rand(2, 64, 400)
        changed = x.clone()
        changed[:, :, :64] += 5.
        self.assertTrue(torch.equal(block(changed), block(x)))
        changed[:, :, 64] += 5.
        self.assertFalse(torch.equal(block(changed), block(x)))

    def test_output_is_the_last_position(self):
        self.assertEqual(tuple(self.block(TAUS)(torch.rand(2, 64, 400)).shape), (2, 64, 1))


class ModelTests(unittest.TestCase):
    def test_without_summary_it_is_lean(self):
        torch.manual_seed(0)
        lean = LeanTCNN(**{**K8, "dropout": 0}).eval()
        torch.manual_seed(0)
        summary = SummaryTCNN(**{**K8, "dropout": 0}).eval()
        self.assertEqual([n for n, _ in lean.named_parameters()], [n for n, _ in summary.named_parameters()])
        x = histories()
        with torch.no_grad():
            self.assertTrue(torch.equal(lean(x), summary(x)))

    def test_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in build(summary_taus=TAUS).parameters()), 7555)  # k8: 8 taps
        self.assertEqual(sum(p.numel() for p in build(summary_taus=[8., 64.]).parameters()), 7555 - 5 * 32)
        self.assertIsInstance(build(summary_taus=TAUS).blocks[2], SummaryBlock)
        self.assertEqual(build(summary_taus=TAUS).blocks[2].start, 64)  # 1 + 7 * 1 + 7 * 8

    def test_mirror_is_kept(self):
        model, x = build(summary_taus=TAUS), histories()
        with torch.no_grad():
            self.assertLess((model(mirror(x)) - mirror(model(x))).abs().max().item(), TOLERANCE)

    def test_windowed_forecasts_are_the_streamed_ones(self):
        """The training forward (one 400-sample window) against the streaming computation along a whole session:
        equal to float rounding for fast averages, and within 0.002 degrees at step 20 for taus up to 64."""
        stream = torch.from_numpy(random_walk(1, 1500, 7)[0])
        windows = stream.unfold(0, 400, 1).transpose(1, 2)  # windows ending at samples 399 ... 1499
        for taus, components, step20 in (([2., 4., 8.], 1e-6, 1e-4), (TAUS, 1e-4, 2e-3)):
            model = build(summary_taus=taus)
            with torch.no_grad():
                windowed, streamed = model(windows), model.streamed(stream)[399:]
            with self.subTest(taus=taus):
                self.assertLess((windowed - streamed).abs().max().item(), components)
                self.assertLess(degrees(windowed[:, -1], streamed[:, -1]).max().item(), step20)

    def test_gradients_reach_every_parameter(self):
        model = build(summary_taus=TAUS).train()
        target = torch.nn.functional.normalize(torch.randn(12, 20, 4), dim=-1)
        train_tcnn.rotation_loss(model(histories()), target, "angular", .5).backward()
        for name, parameter in model.named_parameters():
            self.assertGreater(parameter.grad.abs().max().item(), 0, name)


class TrainingTests(unittest.TestCase):
    def test_a_summary_config_trains_and_its_checkpoint_loads(self):
        names, hook_names = dict(vars(train_tcnn)), (best_hooks.LeanTCNN, best_hooks.MODEL_KEYS, best_model.LeanTCNN)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, split = make_fixture(directory/"fixture", samples=900)
            config = {**BASE, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20,
                      "summary_taus": TAUS}
            with hooks.installed():
                self.assertIs(best_model.LeanTCNN, SummaryTCNN)
                run_desktop.train(data, split, directory/"run", config, 42, "cpu", synthetic=True)
            model, state = load_checkpoint(directory/"run"/"last.pt")
            experiment = json.loads((directory/"run"/"experiment.json").read_text()) \
                if (directory/"run"/"experiment.json").exists() else None
        self.assertEqual(state["model_spec"]["summary_taus"], TAUS)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7555)
        self.assertIsInstance(model.blocks[2], SummaryBlock)
        self.assertEqual({k: v for k, v in vars(train_tcnn).items() if k in names}, names)
        self.assertEqual((best_hooks.LeanTCNN, best_hooks.MODEL_KEYS, best_model.LeanTCNN), hook_names)
        if experiment is not None:
            self.assertIn("experimental/v10_summary/model.py", json.dumps(experiment))


class ExportTests(unittest.TestCase):
    def test_onnx_stream_is_the_streamed_model(self):
        """Called like the harness (zero states, the latest 400 samples, state_out fed back), the graph gives
        SummaryTCNN.streamed's forecasts once the window is full; its last state holds 7 averages per channel."""
        from benchmark_latency import Streamer, ort_session
        from best.quantize import windows
        model = build(summary_taus=TAUS)
        q = random_walk(1, 700, 11)[0].astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path, shapes = export_streaming_onnx(model, Path(directory)/"summary.onnx", st_layout=True)
            streamer = Streamer(ort_session(path, 1), shapes)
            predictions = {t: streamer(window)[0] for t, window in windows(q, len(q))}
        self.assertEqual(shapes[-1], (1, 32, 2, len(TAUS)))
        with torch.no_grad():
            streamed = model.streamed(torch.from_numpy(q)).numpy()
        graph = np.stack([predictions[t] for t in range(399, len(q))])
        self.assertLess(np.abs(graph - streamed[399:]).max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
