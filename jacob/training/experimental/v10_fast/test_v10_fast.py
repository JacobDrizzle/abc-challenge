"""v10_fast: the short forward against LeanTCNN (outputs, a training step, a tiny training run), the hooks and
the 800-user suite.

python -m unittest experimental.v10_fast.test_v10_fast -v   (from jacob/training)
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
from tcnn.model import rotation_loss
from tcnn.onnx_export import geodesic_degrees  # stable for tiny angles
import tcnn_model_adapter
from tests.test_v5 import smooth_histories
import train_tcnn

from best import hooks as best_hooks
from best.model import MAPS, LeanTCNN, load_checkpoint, receptive_field

from experimental.v10_fast import hooks
from experimental.v10_fast.model import FastLeanTCNN, tap_positions

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = json.loads((ROOT/"best/configs/base.json").read_text())
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *best_hooks.MODEL_KEYS)
K8 = {key: BASE[key] for key in MODEL_KEYS}  # k8_combined's model options
LAYOUTS = {"k8": {},
           "k5": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], channels=48),
           "rf_324": dict(num_blocks=5, kernel_sizes=[3, 3, 3, 3, 4], dilations=[1, 3, 9, 27, 81]),
           "kernel3": dict(num_blocks=5, kernel_sizes=None, dilations=None, effective_context=256)}  # v7's blocks
# Checkpoints on Jacob's PC (outputs/ is gitignored).
CHECKPOINTS = ("v9_confirm/k8_combined/selected_model.pt", "v9_confirm/k5_combined/selected_model.pt",
               "v10_k8_800users_full_pass/full_pass/selected_model.pt")


def pair(scale=.05, **options):
    """LeanTCNN and FastLeanTCNN with the same random weights, large enough that the blocks shape the forecast
    (by more than 10 degrees on smooth_histories) and small enough that float32 rounding stays near 1e-5 degrees."""
    models = []
    for kind in (LeanTCNN, FastLeanTCNN):
        torch.manual_seed(0)
        model = kind(**{**K8, **options})
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, scale)
        models.append(model.eval())
    return models


def tiny(config, **overrides):
    """A config small enough to train on the synthetic fixture in seconds."""
    return {**config, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20, **overrides}


class PositionTests(unittest.TestCase):
    def test_k8_reads_its_receptive_field_and_73_block_steps(self):
        positions = tap_positions(FastLeanTCNN(**K8).blocks, 400)
        self.assertEqual(tuple(positions.shape), (8, 8, 8))  # block 3, block 2, block 1
        self.assertEqual((positions.min().item(), positions.max().item()), (400 - 393, 399))
        self.assertEqual(positions.unique().numel(), 393)  # every step of the receptive field, and no other
        self.assertEqual(positions[-1, -1].tolist(), list(range(392, 400)))  # the last step's own block-1 taps
        self.assertEqual(positions[:, -1, -1].tolist(), [399 - 47 * j for j in range(7, -1, -1)])  # block 3's taps
        # Steps each block outputs: block 1 at 64, block 2 at 8, block 3 at 1.
        self.assertEqual([positions[(..., *[-1] * depth)].numel() for depth in (1, 2, 3)], [64, 8, 1])

    def test_every_layout_reads_exactly_its_receptive_field(self):
        for name, layout in LAYOUTS.items():
            with self.subTest(name):
                model = FastLeanTCNN(**{**K8, **layout})
                field = receptive_field(model.num_blocks, model.dilation_base, model.kernel_sizes, model.dilations)
                self.assertEqual(model.positions.unique().numel(), field)
                self.assertEqual(model.positions.min().item(), model.effective_context - field)


class ForwardTests(unittest.TestCase):
    def test_eval_forecasts_equal_lean_tcnn(self):
        history = torch.from_numpy(smooth_histories(6))
        for name, layout in LAYOUTS.items():
            for maps in MAPS:
                for gravity, mirror in ((True, True), (False, True), (True, False)):
                    with self.subTest(name, maps=maps, gravity=gravity, mirror=mirror):
                        lean, fast = pair(maps=maps, gravity_features=gravity, mirror_equivariant=mirror, **layout)
                        self.assertIsNotNone(fast.positions)
                        with torch.no_grad():
                            expected, got = lean(history), fast(history)
                            # The blocks matter here: without them (a zero head) the forecast is far away.
                            lean.head.weight.zero_()
                            self.assertGreater(geodesic_degrees(lean(history), expected).max(), 1.)
                        self.assertLess(geodesic_degrees(got, expected).max(), 1e-4)  # measured: under 1e-5
                        self.assertLess((got - expected).abs().max().item(), 1e-6)

    def test_a_wrong_tap_order_would_be_caught(self):
        # The same comparison on a model whose taps are reversed in one block: even on these smooth histories the
        # forecast moves by over 100 times the 1e-4 degrees allowed above.
        history = torch.from_numpy(smooth_histories(6))
        for index in range(3):
            with self.subTest(block=index + 1):
                lean, fast = pair()
                with torch.no_grad():
                    fast.blocks[index].depthwise.weight.copy_(fast.blocks[index].depthwise.weight.flip(-1))
                    self.assertGreater(geodesic_degrees(fast(history), lean(history)).max(), 1e-2)

    def test_a_training_step_equals_lean_tcnn(self):
        history = torch.from_numpy(smooth_histories(8, 420))
        history, target = history[:, :400], history[:, 400:]
        lean, fast = pair(scale=.1)
        losses, gradients = {}, {}
        for name, model in (("lean", lean), ("fast", fast)):
            model.train()
            torch.manual_seed(5)  # the dropout masks' generator state
            losses[name] = rotation_loss(model(history), target, "angular", .5)
            losses[name].backward()
            gradients[name] = [parameter.grad for parameter in model.parameters()]
        torch.testing.assert_close(losses["fast"], losses["lean"], rtol=1e-6, atol=0)
        for got, expected in zip(gradients["fast"], gradients["lean"]):
            # Every entry within 1e-5 of that gradient's largest entry (measured: 1.4e-7).
            torch.testing.assert_close(got, expected, rtol=0, atol=1e-5 * expected.abs().max().item())
        # Dropout was on, and it is the masks that matched: another generator state gives another loss.
        torch.manual_seed(6)
        other = rotation_loss(fast(history), target, "angular", .5)
        self.assertGreater(abs(other.item() - losses["fast"].item()), 1e-4 * losses["fast"].item())
        with torch.no_grad():
            self.assertNotEqual(rotation_loss(fast.eval()(history), target, "angular", .5).item(), other.item())

    def test_parameters_specification_and_checkpoints_are_lean_tcnn(self):
        lean, fast = pair()
        self.assertEqual(list(fast.state_dict()), list(lean.state_dict()))
        self.assertEqual(fast.specification(), lean.specification())
        self.assertEqual(sum(p.numel() for p in fast.parameters()), 7555)
        LeanTCNN(**fast.specification()).load_state_dict(fast.state_dict())  # strict: no missing or extra key
        FastLeanTCNN(**lean.specification()).load_state_dict(lean.state_dict())

    def test_other_models_run_lean_tcnn_forward(self):
        history = torch.from_numpy(smooth_histories(3))
        cases = {"full blocks": dict(separable=False, kernel_sizes=None, dilations=None, dilation_base=2, num_blocks=7,
                                     head_basis=None, speed_feature="norm", maps="exact"),
                 "window shorter than the field": dict(effective_context=200)}
        for name, options in cases.items():
            with self.subTest(name):
                lean, fast = pair(**options)
                self.assertIsNone(fast.positions)
                with torch.no_grad():
                    self.assertTrue(torch.equal(fast(history), lean(history)))

    def test_confirmed_checkpoints_forecast_the_same(self):
        history = torch.from_numpy(smooth_histories(16))
        for name in CHECKPOINTS:
            path = ROOT/"outputs"/name
            if not path.exists():
                continue
            with self.subTest(name):
                lean, state = load_checkpoint(path)
                fast = FastLeanTCNN(**state["model_spec"]).eval()
                fast.load_state_dict(state["model_state"])
                with torch.no_grad():
                    self.assertLess(geodesic_degrees(fast(history), lean(history)).max(), 1e-4)


class HookTests(unittest.TestCase):
    def test_hooks_swap_the_model_hash_the_files_and_restore_every_name(self):
        modules = (best_hooks, run_desktop, run_next, tcnn_model_adapter, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertIs(best_hooks.LeanTCNN, FastLeanTCNN)
            self.assertIs(tcnn_model_adapter.TinyTCNN, FastLeanTCNN)
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.FILES):], hooks.FILES)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.FILES:
            self.assertTrue((ROOT/name).exists(), name)

    def test_screen_800_variants_change_only_their_keys(self):
        suite = json.loads((HERE/"configs/screen_800.json").read_text())
        with hooks.installed():
            configs = run_next.experiment_configs(suite, BASE)
        control = {**BASE, "steps_per_epoch": 3162}  # the full pass confirmed on 2 Nov
        self.assertEqual(configs, {"control": control, "dropout_0": {**control, "dropout": 0.},
                                   "passes_2": {**control, "steps_per_epoch": 6324},
                                   "jitter_1_5": {**control, "jitter_augmentation_ms": 1.5},
                                   "width_96": {**control, "channels": 96}})
        self.assertEqual((suite["baseline_lags"], suite["baseline_tau_ms"][-1]), ([1, 3, 5, 10], None))
        wide = FastLeanTCNN(**{key: configs["width_96"][key] for key in MODEL_KEYS})
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 15939)


class TrainingTests(unittest.TestCase):
    def test_a_tiny_run_follows_the_lean_tcnn_run(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data, split = make_fixture(folder/"fixture", samples=900)
            histories = {}
            for name, installed in (("lean", best_hooks.installed), ("fast", hooks.installed)):
                with installed():
                    run_desktop.train(data, split, folder/name, tiny(BASE), 42, "cpu", synthetic=True)
                histories[name] = json.loads((folder/name/"history.json").read_text())
            # Same windows, same dropout masks, same optimiser: six steps apart only by rounding.
            for lean, fast in zip(histories["lean"], histories["fast"]):
                self.assertAlmostEqual(fast["training_loss"] / lean["training_loss"], 1, delta=1e-4)
                self.assertAlmostEqual(fast["validation_endpoint_deg"], lean["validation_endpoint_deg"], delta=1e-3)
            # The checkpoint is a LeanTCNN checkpoint: best's loader, exporter and benchmark take it as it is.
            model, state = load_checkpoint(folder/"fast/last.pt")
            self.assertEqual((type(model), state["epoch"]), (LeanTCNN, 2))
            fast = FastLeanTCNN(**state["model_spec"]).eval()
            fast.load_state_dict(state["model_state"])
            history = torch.from_numpy(smooth_histories(4))
            with torch.no_grad():
                self.assertLess(geodesic_degrees(fast(history), model(history)).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
