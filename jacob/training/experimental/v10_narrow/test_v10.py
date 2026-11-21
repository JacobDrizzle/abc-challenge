"""v10_narrow: the config key, NarrowBlock, the three forwards of NarrowTCNN, the hooks and a tiny training run.

python -m unittest experimental.v10_narrow.test_v10 -v   (from jacob/training)
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
from best.model import KernelBlock, mirror

from experimental.v10_narrow import hooks
from experimental.v10_narrow.configuration import normalize_config
from experimental.v10_narrow.model import NarrowBlock, NarrowTCNN, load_checkpoint
from experimental.v10_speed.configuration import normalize_config as normalize_speed
from experimental.v10_speed.model import LastPositionTCNN

ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT/"best/configs/base.json").read_text())
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *best_hooks.MODEL_KEYS)
K8 = {key: BASE[key] for key in MODEL_KEYS}
NARROW = {"k8_last_8": dict(block_inputs=[32, 32, 8]), "k8_all_narrow": dict(block_inputs=[16, 8, 4]),
          "two_blocks_last_8": dict(num_blocks=2, kernel_sizes=[16, 16], dilations=[1, 25], block_inputs=[32, 8])}
TOLERANCE = 1e-5


def build(**overrides):
    torch.manual_seed(0)
    model = NarrowTCNN(**{**K8, "dropout": 0, **overrides})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)
    return model.eval()


def histories():
    return torch.from_numpy(np.concatenate((smooth_histories(6), random_walk(6, 400, 5))))


class ConfigTests(unittest.TestCase):
    def test_default_is_v10_speed(self):
        self.assertEqual(normalize_config(BASE), {**normalize_speed(BASE), "block_inputs": None})
        self.assertEqual(normalize_config({**BASE, "block_inputs": [32, 32, 8]})["block_inputs"], [32, 32, 8])

    def test_key_is_checked(self):
        for bad in ({"block_inputs": [32, 32]}, {"block_inputs": [32, 32, 33]}, {"block_inputs": [32, 32, 0]},
                    {"block_inputs": 8}, {"block_inputs": [32, 32, 8.]}, {"block_inputs": [32, 32, True]},
                    {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**BASE, **bad})


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        count = lambda **o: sum(p.numel() for p in NarrowTCNN(**{**K8, **o}).parameters())
        self.assertEqual(count(), 7555)
        self.assertEqual(count(block_inputs=[32, 32, 32]), 7555)
        self.assertEqual(count(block_inputs=[32, 32, 8]), 5827)  # last block: 8 x 8 + 32 x 16 + 32 instead of 2,336

    def test_without_narrow_blocks_it_is_last_position_tcnn(self):
        for inputs in (None, [32, 32, 32]):
            torch.manual_seed(0)
            ours = NarrowTCNN(**{**K8, "block_inputs": inputs})
            torch.manual_seed(0)
            reference = LastPositionTCNN(**K8)
            self.assertFalse(any(isinstance(block, NarrowBlock) for block in ours.blocks))
            self.assertEqual(list(ours.state_dict()), list(reference.state_dict()))
            for a, b in zip(ours.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_a_full_width_narrow_block_is_a_kernel_block(self):
        torch.manual_seed(1)
        reference, ours = KernelBlock(8, 2, 5, 0, 4).eval(), NarrowBlock(8, 2, 5, 0, 4, 8).eval()
        ours.load_state_dict(reference.state_dict())
        x = torch.randn(3, 16, 40)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), reference(x))

    def test_a_narrow_block_reads_only_its_inputs(self):
        torch.manual_seed(2)
        block = NarrowBlock(8, 2, 5, 0, 4, 3).eval()
        x = torch.randn(1, 16, 40).relu()
        changed = x.clone()
        changed[:, 3:8, :-1] += 1.  # channels the block does not read, at every position but the last
        changed[:, 11:, :-1] += 1.
        with torch.no_grad():
            self.assertTrue(torch.equal(block(changed)[:, :, -1], block(x)[:, :, -1]))
            changed[:, 2, 20] += 1.  # a channel it reads, at lag 19: no tap (0, 5, 10, 15)
            self.assertTrue(torch.equal(block(changed)[:, :, -1], block(x)[:, :, -1]))
            changed[:, 2, 24] += 1.  # lag 15: a tap
            self.assertFalse(torch.equal(block(changed)[:, :, -1], block(x)[:, :, -1]))

    def test_the_three_forwards_agree_and_keep_the_mirror(self):
        x = histories()
        stream = torch.from_numpy(random_walk(1, 900, 7)[0])
        anchors = torch.arange(399, 900)
        for name, layout in NARROW.items():
            model = build(**layout)
            with self.subTest(name), torch.no_grad():
                fast = model(x)
                model.last_position_forward = False
                self.assertLess((model(x) - fast).abs().max().item(), TOLERANCE)
                windows = stream.unfold(0, 400, 1).transpose(1, 2)
                self.assertLess((model.dense(stream, anchors) - model(windows)).abs().max().item(), TOLERANCE)
                self.assertLess((model(mirror(x)) - mirror(model(x))).abs().max().item(), TOLERANCE)

    def test_gradients_reach_every_parameter(self):
        model = build(block_inputs=[32, 32, 8]).train()
        target = torch.nn.functional.normalize(torch.randn(12, 20, 4), dim=-1)
        train_tcnn.rotation_loss(model(histories()), target, "angular", .5).backward()
        for name, parameter in model.named_parameters():
            self.assertGreater(parameter.grad.abs().max().item(), 0, name)


class TrainingTests(unittest.TestCase):
    def test_a_narrow_config_trains_and_its_checkpoint_loads(self):
        names = dict(vars(train_tcnn))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, split = make_fixture(directory/"fixture", samples=900)
            config = {**BASE, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20,
                      "block_inputs": [32, 32, 8]}
            with hooks.installed():
                run_desktop.train(data, split, directory/"run", config, 42, "cpu", synthetic=True)
            model, state = load_checkpoint(directory/"run"/"last.pt")
        self.assertEqual(state["model_spec"]["block_inputs"], [32, 32, 8])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5827)
        self.assertIsInstance(model.blocks[2], NarrowBlock)
        self.assertEqual({k: v for k, v in vars(train_tcnn).items() if k in names}, names)


if __name__ == "__main__":
    unittest.main()
