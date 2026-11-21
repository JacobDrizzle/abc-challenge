"""Check that cached residual edits are the same exported model operation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import torch

from best.model import LeanTCNN, load_checkpoint, mirror
from tcnn.model import qmul, rotvec_to_quaternion
from .edits import IDENTITY_GAINS, apply_gains, candidates, decompose, reconstruct


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/v10_k8_800users_full_pass/full_pass/selected_model.pt"


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def histories(batch=3):
    t = torch.arange(400, dtype=torch.float32) * .01
    phase = torch.arange(batch, dtype=torch.float32)[:, None] * .7
    angles = torch.stack((.2 * torch.sin(1.1 * t + phase),
                          .8 * torch.sin(.6 * t + phase),
                          .15 * torch.cos(1.3 * t + phase)), -1)
    return rotvec_to_quaternion(angles)


def small_model():
    with torch.random.fork_rng():
        torch.manual_seed(171)
        model = LeanTCNN(context=400, horizon=20, channels=8, num_blocks=2,
                         gravity_features=True, mirror_equivariant=True, separable=True,
                         head_basis=3, speed_feature="l1", kernel_sizes=[3, 3],
                         dilations=[1, 3], maps="native", damped_prior_tau_ms=100)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, .04)
    return model.eval()


class WeightEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.model = small_model()
        self.history = histories()

    def assert_same_rotations(self, actual, expected):
        # Quaternion sign is arbitrary. Align it before comparing all horizons.
        signs = torch.where((actual * expected).sum(-1, keepdim=True) < 0, -1., 1.)
        torch.testing.assert_close(actual * signs, expected, rtol=2e-6, atol=2e-6)

    def test_candidate_list_is_fixed_unique_and_identity_is_separate(self):
        options = candidates()
        self.assertEqual(len(options), 16)
        self.assertEqual(options, candidates())
        self.assertEqual([item["id"] for item in options], sorted(item["id"] for item in options))
        self.assertEqual(len({item["id"] for item in options}), 16)
        self.assertEqual(len({item["gains"] for item in options}), 16)
        self.assertNotIn(IDENTITY_GAINS, [item["gains"] for item in options])
        self.assertEqual(options[0], {"id": "global_090", "gains": (.9, .9, .9)})
        self.assertEqual(options[-1], {"id": "z_110", "gains": (1., 1., 1.1)})

    def test_identity_reconstructs_original_at_every_horizon(self):
        parts = decompose(self.model, self.history)
        with torch.no_grad():
            expected = self.model(self.history)
        self.assertEqual(parts[0].shape, (3, 20, 3))
        self.assertEqual(parts[1].shape, (3, 20, 3))
        self.assertEqual(parts[2].shape, (3, 1, 4))
        self.assertTrue(all(not part.requires_grad for part in parts))
        torch.testing.assert_close(reconstruct(*parts), expected, rtol=0, atol=0)
        self.assertEqual(state_hash(apply_gains(self.model, IDENTITY_GAINS)), state_hash(self.model))

    def test_each_candidate_matches_actual_edited_model_at_all_horizons(self):
        parts = decompose(self.model, self.history)
        for option in candidates():
            with self.subTest(candidate=option["id"]), torch.no_grad():
                edited = apply_gains(self.model, option["gains"])
                actual = edited(self.history)
                self.assert_same_rotations(reconstruct(*parts, option["gains"]), actual)
                torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3, 20), rtol=0, atol=2e-7)

    def test_only_head_changes_and_original_is_unchanged(self):
        original_hash = state_hash(self.model)
        original_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        gains = (.9, 1.05, 1.1)
        edited = apply_gains(self.model, gains)
        for name, value in self.model.state_dict().items():
            other = edited.state_dict()[name]
            self.assertNotEqual(value.data_ptr(), other.data_ptr())
            if name not in ("head.weight", "head_bias_x"):
                self.assertTrue(torch.equal(value, other), name)
        expected_head = self.model.head.weight.detach().reshape(3, 3, -1) * torch.tensor(gains)[None, :, None]
        torch.testing.assert_close(edited.head.weight.reshape(3, 3, -1), expected_head, rtol=0, atol=0)
        torch.testing.assert_close(edited.head_bias_x, self.model.head_bias_x * gains[0], rtol=0, atol=0)
        self.assertEqual(sum(parameter.numel() for parameter in edited.parameters()), original_parameters)
        self.assertEqual(edited.specification(), self.model.specification())
        self.assertEqual(state_hash(self.model), original_hash)

    def test_edits_preserve_sign_mirror_and_world_yaw(self):
        signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.)[None, :, None]
        yaw = rotvec_to_quaternion(torch.tensor([0., .7, 0.])).reshape(1, 1, 4)
        for gains in (IDENTITY_GAINS, (.9, 1.05, 1.1)):
            edited = apply_gains(self.model, gains)
            with torch.no_grad():
                expected = edited(self.history)
                self.assert_same_rotations(edited(self.history * signs), expected)
                self.assert_same_rotations(edited(mirror(self.history)), mirror(expected))
                self.assert_same_rotations(edited(qmul(yaw, self.history)), qmul(yaw, expected))

    def test_rejects_invalid_gains_and_training_cache(self):
        for gains in ((1., 1.), (1., float("nan"), 1.), (1., 0., 1.), (1., float("inf"), 1.)):
            with self.subTest(gains=gains), self.assertRaises(ValueError):
                apply_gains(self.model, gains)
        with self.assertRaises(ValueError):
            decompose(self.model.train(), self.history)

    @unittest.skipUnless(CHECKPOINT.exists(), "The local 800-user checkpoint is unavailable")
    def test_selected_checkpoint_identity_and_every_candidate(self):
        model, _ = load_checkpoint(CHECKPOINT)
        original_hash = state_hash(model)
        history = self.history[:2]
        parts = decompose(model, history)
        with torch.no_grad():
            torch.testing.assert_close(reconstruct(*parts), model(history), rtol=0, atol=0)
        for option in candidates():
            with self.subTest(candidate=option["id"]), torch.no_grad():
                edited = apply_gains(model, option["gains"])
                self.assert_same_rotations(reconstruct(*parts, option["gains"]), edited(history))
                self.assertEqual(sum(parameter.numel() for parameter in edited.parameters()), 7555)
        self.assertEqual(state_hash(model), original_hash)


if __name__ == "__main__":
    unittest.main()
