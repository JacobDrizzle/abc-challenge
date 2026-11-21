"""Final-model recovery, causality, chunk boundaries, masks and cohort guards."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from experimental.v10_endpoint_refit.fit import apply_delta, predict
from .common import MODEL_SHA, ORIGINAL, ROOT, assessment_gate, sha256, validation_partition
from .engine import CHUNK, batches, dense_parts
from .recover import recover, roundtrip, save_inference, write_patch
from .verify import angles


class FinalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = recover().eval()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_recovery_roundtrip_is_byte_identical(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            result = roundtrip(self.model, Path(temporary) / "roundtrip.onnx")
            self.assertEqual(result["roundtrip_sha256"], MODEL_SHA)
            self.assertEqual(result["parameters"], 7555)

    def test_patch_mask_preservation_and_early_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            folder = Path(temporary)
            edited = apply_delta(self.model, np.linspace(-.00001, .00001, 97))
            result = write_patch(edited, folder / "edit.onnx")
            self.assertEqual(result["allowed_scalars"], 97)
            self.assertLessEqual(result["changed_scalars"], 97)
            self.assertTrue(result["all_other_graph_content_exact"])
            bad = deepcopy(edited)
            with torch.no_grad():
                bad.projection.bias[0] += 1
            with self.assertRaises(ValueError):
                write_patch(bad, folder / "bad.onnx")
            self.assertFalse((folder / "bad.onnx").exists())
            with self.assertRaises(FileExistsError):
                write_patch(edited, folder / "edit.onnx")
            self.assertEqual(sha256(ORIGINAL), MODEL_SHA)

    def test_split_rejection_precedes_file_access(self):
        row = {"path": "test/heldout/recording/segment", "user": "heldout"}
        with patch('experimental.final_v10_s42_weightspace.engine.h5py.File') as opened:
            for split in ("train", "validation", "test"):
                with self.assertRaises(ValueError):
                    list(batches(self.model, "missing", [row], split, ["heldout"], 20))
            opened.assert_not_called()

    def test_partition_is_deterministic_disjoint_and_preserves_old_users(self):
        old = [f"old-{i}" for i in range(120)]
        users = old + [f"new-{i}" for i in range(666)]
        a, b = validation_partition(users, old)
        self.assertEqual((len(a), len(b)), (524, 262))
        self.assertFalse(set(a) & set(b))
        self.assertTrue(set(old) <= set(a))
        self.assertEqual(set(a) | set(b), set(users))
        self.assertEqual((a, b), validation_partition(list(reversed(users)), list(reversed(old))))

    @torch.inference_mode()
    def test_dense_chunk_boundaries_targets_and_no_future_leak(self):
        n = CHUNK + 600
        rng = np.random.default_rng(981)
        q = np.cumsum(rng.normal(0, .001, (n, 4)), axis=0) + [2., 0., 0., 0.]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        row = {"path": "train/user/recording/segment", "user": "user", "samples": n}
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "fixture.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group(row["path"])
                group.create_dataset("time_s", data=np.arange(n) * .01)
                group.create_dataset("quaternion_wxyz", data=q)
            for stride in (1, 5, 20):
                seen = []
                for _, selected, parts, target in batches(self.model, path, [row], "train", ["user"], stride):
                    seen.extend(selected)
                    np.testing.assert_allclose(target.numpy(), q[selected[:, None] + np.arange(1, 21)], atol=1e-15)
                    ids = [0, len(selected) - 1]
                    histories = torch.tensor(np.stack([q[t - 399:t + 1] for t in selected[ids]]), dtype=torch.float32)
                    error = angles(predict(parts)[ids].numpy(), self.model(histories).numpy()).max()
                    self.assertLess(error, .001)
                np.testing.assert_array_equal(seen, np.arange(399, n - 20, stride))
        tensor = torch.tensor(q[:650], dtype=torch.float32)
        a = torch.tensor([399, 450])
        first = predict(dense_parts(self.model, tensor, a))
        tensor[451:] = torch.tensor([0., 1., 0., 0.])
        second = predict(dense_parts(self.model, tensor, a))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_small_positive_evidence_cannot_pass_minimum_gain(self):
        summaries = {"identity": {"p95_endpoint_deg": 10., "mean_all_horizons_deg": 3.},
                     "candidate": {"p95_endpoint_deg": 9., "mean_all_horizons_deg": 2.}}
        comparisons = {"identity": {"users": 262, "mean_diff_deg": -.004, "ci95": [-.006, -.002],
                                     "equal_user_mean_diff_deg": -.004, "users_better": 200},
                       "global_105": {"mean_diff_deg": -.003}}
        self.assertFalse(assessment_gate(summaries, comparisons)["passed"])
        comparisons["identity"]["mean_diff_deg"] = -.005
        self.assertTrue(assessment_gate(summaries, comparisons)["passed"])
        comparisons["identity"]["ci95"][1] = 0
        self.assertFalse(assessment_gate(summaries, comparisons)["passed"])

    def test_decimal_candidate_names_remain_distinct_and_inference_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = [Path(temporary) / f"ridge_0.0001_strength_{s:g}.pt" for s in (.25, .5, 1.)]
            for path in paths:
                save_inference(self.model, path, {"correction_fitted": False})
                checkpoint = torch.load(path, weights_only=True)
                self.assertFalse(checkpoint['provenance']['original_training_checkpoint'])
                self.assertNotIn('optimizer_state', checkpoint)
            self.assertEqual(len(list(Path(temporary).glob('*.pt'))), 3)


if __name__ == "__main__":
    unittest.main()
