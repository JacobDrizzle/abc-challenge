"""Independent algebra, model invariants, and split/copy checks for the refit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from baseline_pipeline import sha256_file
from best.model import LeanTCNN, load_checkpoint, mirror
from tcnn.data import DeviceWindows
from tcnn.model import qmul, rotvec_to_quaternion
from experimental.v10_weight_edit.test_edits import histories, state_hash
from experimental.v10_weight_edit.test_run import make_segments
from experimental.v10_weight_edit.metrics import ErrorAccumulator
from .fit import NormalEquations, apply_delta, candidates, endpoint_change, extract, predict, residual_and_prior, solve
from .run import compare_summaries, prepare, restricted_batches, save_candidate

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parents[1] / "outputs/v10_k8_800users_full_pass/full_pass/selected_model.pt"


def model_fixture():
    with torch.random.fork_rng():
        torch.manual_seed(351)
        model = LeanTCNN(context=400, horizon=20, channels=64, num_blocks=1, gravity_features=True,
                         mirror_equivariant=True, separable=True, head_basis=3, speed_feature="l1",
                         kernel_sizes=[3], dilations=[1], maps="native", damped_prior_tau_ms=100)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, .025)
    return model.eval()


def algebra_fixture(count=320):
    rng = np.random.default_rng(315)
    z = rng.normal(size=(count, 65))
    z[:, 32] = 1
    r = rng.normal(0, .15, (count, 3))
    residual = rng.normal(0, .05, (count, 3))
    delta = rng.normal(0, .001, 97)
    a = np.c_[np.full(count, 2.), r + endpoint_change(torch.tensor(z), delta).numpy()]
    q = a / np.linalg.norm(a, axis=1, keepdims=True)
    return z, r, residual, q, delta


def explicit_normal_equations(z, r, q):
    """A deliberately direct per-example 4x97 formulation, independent of batching."""
    gram, rhs, covariance, weight_sum = np.zeros((97, 97)), np.zeros(97), np.zeros((97, 97)), 0.
    for features, rotation, target in zip(z, r, q):
        j = np.zeros((4, 97))
        j[1, :33], j[2, 33:65], j[3, 65:] = features[:33], features[33:], features[33:]
        a = np.r_[2., rotation]
        p = np.eye(4) - np.outer(target, target)
        angle = 2 * np.arccos(np.clip(abs(a @ target) / np.linalg.norm(a), 0, 1))
        u = 1 / max(angle, np.pi / 180)
        gram += u * j.T @ p @ j / (a @ a)
        rhs -= u * j.T @ p @ a / (a @ a)
        covariance += j.T @ j
        weight_sum += u
    return gram / weight_sum, rhs / weight_sum, covariance / len(z)


class RefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def statistics(self, z, r, residual, q):
        stats = NormalEquations()
        for start in range(0, len(z), 17):
            stats.update(*(torch.tensor(a[start:start + 17]) for a in (z, r, residual, q)))
        return stats.arrays()

    def test_normal_equations_match_independent_projector_reference(self):
        z, r, residual, q, _ = algebra_fixture(47)
        arrays = self.statistics(z, r, residual, q)
        gram, rhs, covariance = explicit_normal_equations(z, r, q)
        for actual, expected in ((arrays['gram'], gram), (arrays['rhs'], rhs), (arrays['covariance'], covariance)):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(arrays['scales'], np.sqrt(np.diag(covariance)), rtol=1e-12)

    def test_known_small_correction_and_regularization_limits(self):
        z, r, residual, q, expected = algebra_fixture()
        arrays = self.statistics(z, r, residual, q)
        np.testing.assert_allclose(solve(arrays, 1e-9), expected, atol=1e-8, rtol=1e-5)
        self.assertLess(np.linalg.norm(solve(arrays, 1e8)), 1e-9)
        for penalty in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                solve(arrays, penalty)

    def test_sign_stationary_near_pi_and_inactive_features(self):
        z, r, residual, q, _ = algebra_fixture(50)
        z[:, 4] = z[:, 40] = 0
        r[:3] = 0
        q[:3] = [[1, 0, 0, 0], [0, 1, 0, 0], [1e-12, 0, 1, 0]]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        arrays = self.statistics(z, r, residual, q)
        flipped = self.statistics(z, r, residual, -q)
        for name in ('gram', 'rhs', 'scales'):
            np.testing.assert_array_equal(arrays[name], flipped[name])
        delta = solve(arrays, .1)
        self.assertTrue(np.isfinite(delta).all())
        self.assertEqual(arrays['scales'][4], 1e-6)
        np.testing.assert_allclose(delta[[4, 40, 72]], 0, atol=1e-10)
        with self.assertRaises(ValueError):
            self.statistics(z, r, residual, np.zeros_like(q))

    def test_21_candidates_respect_cap_and_strength(self):
        z, r, residual, q, _ = algebra_fixture()
        arrays = self.statistics(z, r, residual * .001, q)
        options = candidates(arrays)
        self.assertEqual(len(options), 21)
        self.assertEqual(len({c['id'] for c in options}), 21)
        for c in options:
            self.assertLessEqual(c['train_rms_correction'], c['train_rms_limit'] + 1e-12)
        for offset in range(0, 21, 3):
            np.testing.assert_allclose(np.array(options[offset]['delta']) * 4, options[offset + 2]['delta'])
        arrays['residual_rms'] = np.array(0.)
        self.assertTrue(all(np.all(np.array(c['delta']) == 0) for c in candidates(arrays)))

    def test_cached_predictions_and_cubic_shape(self):
        model = model_fixture()
        history = histories()
        delta = np.random.default_rng(7).normal(0, .0007, 97)
        with torch.inference_mode():
            parts = extract(model, history)
            torch.testing.assert_close(predict(parts), model(history), rtol=0, atol=0)
            edited = apply_delta(model, delta)
            torch.testing.assert_close(predict(parts, delta), edited(history), rtol=2e-6, atol=2e-6)
            base_residual, _ = residual_and_prior(parts)
            changed_residual, _ = residual_and_prior(extract(edited, history))
            difference = changed_residual - base_residual
            torch.testing.assert_close(difference[:, 9], difference[:, 19] / 8, rtol=1e-4, atol=1e-7)
            torch.testing.assert_close(difference[:, 19], endpoint_change(parts['z'], delta), rtol=1e-4, atol=1e-7)

    def test_only_cubic_97_coefficients_can_change_and_original_preserved(self):
        model = model_fixture()
        digest = state_hash(model)
        edited = apply_delta(model, np.full(97, .001))
        changed = 0
        for name, value in model.state_dict().items():
            other = edited.state_dict()[name]
            if name == 'head.weight':
                torch.testing.assert_close(value[:6], other[:6], rtol=0, atol=0)
            elif name == 'head_bias_x':
                torch.testing.assert_close(value[:2], other[:2], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            changed += int((value != other).sum())
        self.assertEqual(changed, 97)
        self.assertEqual(state_hash(model), digest)
        self.assertEqual(model.specification(), edited.specification())

    def test_mirror_sign_world_yaw_and_unit_norm(self):
        model = apply_delta(model_fixture(), np.random.default_rng(29).normal(0, .001, 97))
        history = histories()
        yaw = rotvec_to_quaternion(torch.tensor([0., .7, 0.])).reshape(1, 1, 4)
        signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.)[None, :, None]
        with torch.inference_mode():
            expected = model(history)
            for actual, desired in ((model(history * signs), expected), (model(mirror(history)), mirror(expected)),
                                    (model(qmul(yaw, history)), qmul(yaw, expected))):
                sign = torch.where((actual * desired).sum(-1, keepdim=True) < 0, -1., 1.)
                torch.testing.assert_close(actual * sign, desired, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(expected.norm(dim=-1), torch.ones(3, 20), rtol=0, atol=2e-7)

    @unittest.skipUnless(CHECKPOINT.exists(), 'Local 800-user checkpoint unavailable')
    def test_actual_checkpoint_cached_parity(self):
        model, _ = load_checkpoint(CHECKPOINT)
        with torch.inference_mode():
            h = histories(5)
            parts = extract(model, h)
            torch.testing.assert_close(predict(parts), model(h), rtol=0, atol=0)
            delta = np.random.default_rng(51).normal(0, .0001, 97)
            edited = apply_delta(model, delta)
            torch.testing.assert_close(predict(parts, delta), edited(h), rtol=2e-6, atol=2e-6)
            self.assertEqual(sum(p.numel() for p in edited.parameters()), 7555)

    def test_split_guards_before_file_access(self):
        with patch('experimental.v10_endpoint_refit.run.h5py.File') as opened:
            for split, path in (('train', 'validation/a'), ('validation', 'train/a'), ('test', 'test/a'), ('train', 'test/a')):
                with self.assertRaises(ValueError):
                    list(restricted_batches('unused', [{'path': path}], split, 'cpu', 20, 2))
            opened.assert_not_called()

    def test_window_anchors_match_existing_pipeline(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            folder = Path(temporary).resolve()
            self.assertTrue(folder.is_relative_to(HERE))
            path = folder / 'fixture.h5'
            rows = make_segments(path)
            with h5py.File(path, 'r+') as handle:
                handle.copy('validation', 'train')
            for split in ('train', 'validation'):
                scoped = [{**r, 'path': r['path'].replace('validation/', split + '/')} for r in rows]
                expected = DeviceWindows(path, scoped, 'cpu')
                for stride in (1, 5, 20):
                    actual = list(restricted_batches(path, scoped, split, 'cpu', stride, 3))
                    reference = list(expected.windows(stride, 3))
                    for index in (0, 1):
                        torch.testing.assert_close(torch.cat([a[index + 1] for a in actual]),
                                                   torch.cat([a[index] for a in reference]), rtol=0, atol=0)

    def test_saved_copy_reloads_and_existing_run_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            output = Path(temporary).resolve()
            self.assertTrue(output.is_relative_to(HERE))
            (output / 'originals').mkdir()
            (output / 'candidates').mkdir()
            model = model_fixture()
            source = output / 'originals/seed_44.pt'
            torch.save({'model_spec': model.specification(), 'model_state': model.state_dict()}, source)
            digest = sha256_file(source)
            item = {'id': 'fixture', 'delta': np.full(97, .001).tolist()}
            path = save_candidate(output, 44, item)
            reloaded, payload = load_checkpoint(path)
            self.assertEqual(state_hash(reloaded), state_hash(apply_delta(model, item['delta'])))
            self.assertEqual(sha256_file(source), digest)
            self.assertTrue(payload['endpoint_refit']['training_performed'])
            with self.assertRaises(FileExistsError):
                save_candidate(output, 44, item)
            with patch('experimental.v10_endpoint_refit.run.h5py.File') as opened, self.assertRaises(FileExistsError):
                prepare(output, 512)
            opened.assert_not_called()

    def test_gate_requires_beating_previous_gain_as_well_as_original(self):
        summaries = {}
        for name, value in (('identity', 5.), ('global_105', 4.), ('candidate', 4.5)):
            acc = ErrorAccumulator()
            for user in range(40):
                acc.update(str(user), np.full((2, 20), value))
            summaries[name] = acc.summary()
        comparisons, gate = compare_summaries(summaries, 'confirmation')
        self.assertLess(comparisons['identity']['mean_diff_deg'], 0)
        self.assertFalse(gate['checks']['beats_previous_gain_endpoint'])
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()
