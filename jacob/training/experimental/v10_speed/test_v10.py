"""v10_speed: the two config keys, the last-position forward, dense validation, the host-memory starts, the hooks
and tiny training runs.

python -m unittest experimental.v10_speed.test_v10 -v   (from jacob/training)
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import run_desktop
import run_next
from smoke_tcnn import make_fixture
from tcnn.data import check_corpus
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories
import train_tcnn

from best import hooks as best_hooks
from best.model import LeanTCNN, load_checkpoint, receptive_field
from best.training import TrainingWindows, windows_fingerprint

from experimental.v10_speed import hooks
from experimental.v10_speed.configuration import V10_DEFAULTS, normalize_config
from experimental.v10_speed.model import LastPositionTCNN, lattice_lags
from experimental.v10_speed.training import HostWindows
from experimental.v10_speed.validation import (WindowBatches, anchors_for, check_against_windows, dense_forecasts,
                                               dense_validate)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = json.loads((ROOT/"best/configs/base.json").read_text())
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *best_hooks.MODEL_KEYS)
K8 = {key: BASE[key] for key in MODEL_KEYS}
# Other layouts the forward must follow: kernel-3 blocks (v7), k5_combined, overlapping taps, two blocks, one
# stream, and the other maps.
LAYOUTS = {"k8": {},
           "kernel3": dict(num_blocks=5, kernel_sizes=None, dilations=None),
           "k5": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], channels=48),
           "k16_overlapping": dict(kernel_sizes=[16] * 3, dilations=[1, 4, 21]),
           "two_blocks": dict(num_blocks=2, kernel_sizes=[16, 16], dilations=[1, 25]),
           "one_stream": dict(mirror_equivariant=False),
           "lean_maps": dict(maps="lean"), "exact_maps": dict(maps="exact")}
TOLERANCE = 1e-5  # quaternion components; float32 rounding gives about 1e-7


def tiny(config, **overrides):
    """A config small enough to train on the synthetic fixture in seconds."""
    return {**config, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20, **overrides}


def pair(**overrides):
    """LeanTCNN and LastPositionTCNN with the same random weights (non-zero head, so the network is tested)."""
    torch.manual_seed(0)
    reference = LeanTCNN(**{**K8, "dropout": 0, **overrides})
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, .05)
    ours = LastPositionTCNN(**{**K8, "dropout": 0, **overrides})
    ours.load_state_dict(reference.state_dict())
    return reference.eval(), ours.eval()


def histories():
    return torch.from_numpy(np.concatenate((smooth_histories(6), random_walk(6, 400, 5))))


class ConfigTests(unittest.TestCase):
    def test_defaults_add_the_two_keys_to_best(self):
        self.assertEqual(normalize_config(BASE), {**BASE, **V10_DEFAULTS})
        self.assertFalse(normalize_config({**BASE, "dense_validation": False})["dense_validation"])

    def test_keys_are_checked(self):
        for bad in ({"last_position_forward": 1}, {"dense_validation": "yes"}, {"dense_validation": None},
                    {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**BASE, **bad})

    def test_screen_variants_change_only_their_keys(self):
        with hooks.installed():
            configs = run_next.experiment_configs(json.loads((HERE/"configs/screen.json").read_text()), BASE)
        control = {**BASE, **V10_DEFAULTS}
        self.assertEqual(configs, {"control": control,
                                   "reference": {**control, "last_position_forward": False, "dense_validation": False},
                                   "dense_validation_only": {**control, "last_position_forward": False}})

    def test_hooks_hash_the_new_files_and_restore_every_name(self):
        import tcnn_model_adapter
        modules = (best_hooks, run_desktop, run_next, tcnn_model_adapter, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.FILES):], hooks.FILES)
            self.assertIs(train_tcnn.normalize_config, normalize_config)
            self.assertIs(tcnn_model_adapter.TinyTCNN, LastPositionTCNN)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.FILES:
            self.assertTrue((ROOT/name).exists(), name)


class ModelTests(unittest.TestCase):
    def test_k8_lattice_covers_the_receptive_field(self):
        lags = lattice_lags([8, 8, 8], [1, 8, 47])
        self.assertEqual(len(lags), 512)
        self.assertEqual(lags[:8].tolist(), list(range(392, 384, -1)))  # the oldest output of block 0, oldest tap first
        self.assertEqual(lags[-1].item(), 0)
        self.assertEqual(sorted(set(lags.tolist())), list(range(393)))
        self.assertEqual(receptive_field(3, kernel_sizes=[8] * 3, dilations=[1, 8, 47]), 393)

    def test_forward_is_lean_forward(self):
        x = histories()
        for name, layout in LAYOUTS.items():
            reference, ours = pair(**layout)
            with self.subTest(name), torch.no_grad():
                self.assertIsNotNone(ours.lattice_kernels)
                self.assertLess((ours(x) - reference(x)).abs().max().item(), TOLERANCE)

    def test_gradients_are_lean_gradients(self):
        x = histories()
        target = torch.nn.functional.normalize(torch.randn(len(x), 20, 4, generator=torch.Generator().manual_seed(1)),
                                               dim=-1)
        reference, ours = pair()
        gradients = []
        for model in (reference, ours):
            model.zero_grad()
            train_tcnn.rotation_loss(model(x), target, "angular", .5).backward()
            gradients.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]))
        self.assertGreater(gradients[0].abs().max().item(), 1e-3)
        torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-3, atol=1e-6)

    def test_switch_off_and_unsupported_blocks_use_lean_forward(self):
        x = histories()
        off = LastPositionTCNN(**{**K8, "last_position_forward": False}).eval()
        plain = LastPositionTCNN(**{**K8, "separable": False, "kernel_sizes": None, "dilations": None,
                                    "num_blocks": 5}).eval()
        self.assertIsNone(plain.lattice_kernels)
        for model in (off, plain):
            reference = LeanTCNN(**model.specification()).eval()
            reference.load_state_dict(model.state_dict())
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), reference(x)))

    def test_checkpoints_are_lean_checkpoints(self):
        reference, ours = pair()
        self.assertEqual(ours.specification(), reference.specification())
        self.assertEqual(list(ours.state_dict()), list(reference.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save({"model_spec": ours.specification(), "model_state": ours.state_dict()}, path)
            loaded, _ = load_checkpoint(path)
        self.assertIs(type(loaded), LeanTCNN)
        with torch.no_grad():
            self.assertLess((loaded(histories()) - ours(histories())).abs().max().item(), TOLERANCE)

    def test_dense_is_the_windowed_forecast(self):
        stream = torch.from_numpy(random_walk(1, 1500, 7)[0])
        other = torch.from_numpy(random_walk(1, 700, 8)[0])
        anchors = torch.arange(399, 1500)
        windows = stream.unfold(0, 400, 1).transpose(1, 2)  # [1101, 400, 4]: the window ending at each anchor
        for name, layout in LAYOUTS.items():
            reference, ours = pair(**layout)
            with self.subTest(name), torch.no_grad():
                windowed = reference(windows)
                self.assertLess((ours.dense(stream, anchors) - windowed).abs().max().item(), TOLERANCE)
                # Another recording in front, as in the buffer: it cannot reach an anchor 399 samples in.
                joined = ours.dense(torch.cat((other, stream)), anchors + len(other))
                self.assertLess((joined - windowed).abs().max().item(), TOLERANCE)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data, self.split = make_fixture(self.directory/"fixture", samples=900)
        rows = check_corpus(self.data, self.split, minimum_users=3, minimum_recordings=3, synthetic=True)["segments"]
        self.rows = rows  # three users, one 900-sample segment each: used together as one validation buffer

    def tearDown(self):
        self.temp.cleanup()

    def model(self):
        return pair()[1]

    def test_host_starts_draw_the_same_batches(self):
        ours, best = HostWindows(self.data, self.rows, "cpu"), TrainingWindows(self.data, self.rows, "cpu")
        self.assertEqual(windows_fingerprint(ours), windows_fingerprint(best))
        self.assertEqual(len(ours.starts), len(best.starts))
        for sampling in ("user", "window"):
            a = HostWindows(self.data, self.rows, "cpu", sampling=sampling, tilt_deg=15).batches(42, 3, 4, 16)
            b = TrainingWindows(self.data, self.rows, "cpu", sampling=sampling, tilt_deg=15).batches(42, 3, 4, 16)
            for (x, y), (p, q) in zip(a, b):
                self.assertTrue(torch.equal(x, p) and torch.equal(y, q))

    def test_anchors_are_the_trainers_windows(self):
        windows = HostWindows(self.data, self.rows, "cpu")
        for stride in (1, 5, 20):
            anchors = anchors_for(windows.bounds, 400, 20, stride)
            batches = list(windows.windows(stride, 64))
            last = torch.cat([history[:, -1] for history, _ in batches])
            self.assertEqual(len(anchors), len(last))
            self.assertTrue(torch.equal(windows.buffer[torch.from_numpy(anchors)], last))

    def test_dense_validation_is_the_trainers_validation(self):
        windows, model = HostWindows(self.data, self.rows, "cpu"), self.model()
        for stride in (1, 5):
            dense = dense_validate(model, windows.windows(stride, 64), 20)
            windowed = train_tcnn.validate(model, TrainingWindows(self.data, self.rows, "cpu").windows(stride, 64), 20)
            self.assertEqual(dense["windows"], windowed["windows"])
            np.testing.assert_allclose(dense["mean_deg_by_horizon"], windowed["mean_deg_by_horizon"], atol=1e-6, rtol=0)
            self.assertAlmostEqual(dense["mean_endpoint_deg"], windowed["mean_endpoint_deg"], delta=1e-6)
        self.assertGreater(dense["mean_endpoint_deg"], .1)  # a real error, not zeros

    def test_chunks_do_not_change_the_forecasts(self):
        windows, model = HostWindows(self.data, self.rows, "cpu"), self.model()
        anchors = torch.from_numpy(anchors_for(windows.bounds, 400, 20, 1))
        with torch.no_grad():
            whole = torch.cat([f for f, _ in dense_forecasts(model, windows.buffer, anchors, chunk=10**9)])
            for chunk in (64, 500, 901):
                parts = list(dense_forecasts(model, windows.buffer, anchors, chunk=chunk))
                self.assertTrue(torch.equal(torch.cat([a for _, a in parts]), anchors))
                self.assertLess((torch.cat([f for f, _ in parts]) - whole).abs().max().item(), TOLERANCE)

    def test_check_stops_a_wrong_dense_forward(self):
        windows, model = HostWindows(self.data, self.rows, "cpu"), self.model()
        batches = windows.windows(5, 64)
        anchors = torch.from_numpy(anchors_for(windows.bounds, 400, 20, 5))
        with torch.no_grad():
            self.assertLess(check_against_windows(model, batches, anchors), TOLERANCE)
            model.dense = lambda q, a: LastPositionTCNN.dense(model, q, a - 1)  # one sample out of step
            with self.assertRaises(ValueError):
                check_against_windows(model, batches, anchors)

    def test_without_dense_the_trainers_validate_runs(self):
        windows, model = HostWindows(self.data, self.rows, "cpu", dense_validation=False), self.model()
        batches = windows.windows(5, 64)
        self.assertIsInstance(batches, WindowBatches)
        expected = train_tcnn.validate(model, TrainingWindows(self.data, self.rows, "cpu").windows(5, 64), 20)
        self.assertEqual(dense_validate(model, batches, 20), expected)
        self.assertEqual(dense_validate(model, TrainingWindows(self.data, self.rows, "cpu").windows(5, 64), 20), expected)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data, self.split = make_fixture(self.directory/"fixture", samples=900)

    def tearDown(self):
        self.temp.cleanup()

    def history(self, name, installed, **overrides):
        with installed():
            run_desktop.train(self.data, self.split, self.directory/name, tiny(BASE, **overrides), 42, "cpu",
                              synthetic=True)
        return json.loads((self.directory/name/"history.json").read_text())

    def test_with_both_switches_off_training_is_bests(self):
        best = self.history("best", best_hooks.installed)
        ours = self.history("reference", hooks.installed, last_position_forward=False, dense_validation=False)
        for a, b in zip(ours, best):
            self.assertEqual(a["training_loss"], b["training_loss"])
            self.assertEqual(a["validation"], b["validation"])

    def test_dense_validation_scores_the_same_training(self):
        best = self.history("best", best_hooks.installed)
        ours = self.history("dense", hooks.installed, last_position_forward=False)
        for a, b in zip(ours, best):
            self.assertEqual(a["training_loss"], b["training_loss"])
            self.assertEqual(a["validation"]["windows"], b["validation"]["windows"])
            self.assertAlmostEqual(a["validation_endpoint_deg"], b["validation_endpoint_deg"], delta=1e-6)

    def test_control_trains_and_saves_a_lean_checkpoint(self):
        names = dict(vars(train_tcnn))
        ours = self.history("control", hooks.installed)
        best = self.history("best", best_hooks.installed)
        self.assertEqual(len(ours), 2)
        # Dropout draws differ, so the losses are close, not equal.
        self.assertAlmostEqual(ours[0]["training_loss"], best[0]["training_loss"], delta=.05)
        model, state = load_checkpoint(self.directory/"control"/"last.pt")
        self.assertIs(type(model), LeanTCNN)
        self.assertNotIn("last_position_forward", state["model_spec"])
        self.assertEqual({k: v for k, v in vars(train_tcnn).items() if k in names}, names)


if __name__ == "__main__":
    unittest.main()
