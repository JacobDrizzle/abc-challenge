"""Bounded tests for validation isolation and saved-candidate integrity."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch

from baseline_pipeline import sha256_file
from best.model import load_checkpoint
from tcnn.data import DeviceWindows
from tcnn.model import rotvec_to_quaternion
from .edits import apply_gains
from .run import batches, split_users, write_candidate
from .test_edits import histories, small_model


def make_segments(path):
    """Two distinct short streams whose strided starts must reset at each segment."""
    rows = []
    with h5py.File(path, "w") as h5:
        for index, length in enumerate((432, 427)):
            t = torch.arange(length, dtype=torch.float32) * .01
            rotation = torch.stack((.1 * torch.sin(t + index),
                                    .2 * torch.cos(t * .7 + index),
                                    .05 * torch.sin(t * 1.3)), -1)
            # Small norm deviations are permitted and must be normalized identically.
            q = rotvec_to_quaternion(rotation).numpy() * 1.0001
            key = f"user/recording/segment_{index}"
            group_path = f"validation/{key}"
            group = h5.create_group(group_path)
            group.create_dataset("time_s", data=np.arange(length, dtype=np.float64) * .01)
            group.create_dataset("quaternion_wxyz", data=q)
            rows.append({"path": group_path, "key": key, "user": "user", "samples": length})
    return rows


class WeightEditRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_user_split_is_exact_deterministic_disjoint_and_order_independent(self):
        users = [f"user_{i:03d}" for i in range(120)]
        discovery, confirmation = split_users(users)
        self.assertEqual((len(discovery), len(confirmation)), (80, 40))
        self.assertFalse(set(discovery) & set(confirmation))
        self.assertEqual(set(discovery) | set(confirmation), set(users))
        self.assertEqual((discovery, confirmation), split_users(list(reversed(users))))
        shuffled = np.random.default_rng(21).permutation(users).tolist()
        self.assertEqual((discovery, confirmation), split_users(shuffled))
        for invalid in (users[:-1], users + ["extra"], users[:-1] + [users[0]]):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                split_users(invalid)

    def test_segment_windows_match_existing_evaluator_at_both_strides(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "tiny.h5"
            rows = make_segments(path)
            device_windows = DeviceWindows(path, rows, "cpu")
            for stride in (1, 5):
                with self.subTest(stride=stride):
                    actual = list(batches(path, rows, "cpu", stride, 3))
                    expected = list(device_windows.windows(stride, 3))
                    actual_history = torch.cat([history for _, history, _ in actual])
                    actual_target = torch.cat([target for _, _, target in actual])
                    expected_history = torch.cat([history for history, _ in expected])
                    expected_target = torch.cat([target for _, target in expected])
                    expected_starts = torch.cat([torch.arange(0, row["samples"] - 419, stride) for row in rows])
                    torch.testing.assert_close(torch.cat([start for start, _, _ in actual]), expected_starts)
                    torch.testing.assert_close(actual_history, expected_history, rtol=0, atol=0)
                    torch.testing.assert_close(actual_target, expected_target, rtol=0, atol=0)
                    self.assertEqual(actual_history.shape[1:], (400, 4))
                    self.assertEqual(actual_target.shape[1:], (20, 4))

    def test_non_validation_paths_are_rejected_before_any_dataset_access(self):
        for path in ("test/user/recording/segment", "train/user/recording/segment", "/test/user/recording/segment"):
            handle = MagicMock()
            handle.__getitem__.side_effect = AssertionError("A forbidden HDF5 group was accessed")
            file_context = MagicMock()
            file_context.__enter__.return_value = handle
            with self.subTest(path=path), patch("experimental.v10_weight_edit.run.h5py.File", return_value=file_context):
                with self.assertRaisesRegex(ValueError, "validation"):
                    list(batches("unused.h5", [{"path": path, "key": "forbidden"}], "cpu", 1, 4))
                handle.__getitem__.assert_not_called()

    def test_saved_candidate_reloads_as_intended_edit_without_changing_original(self):
        model = small_model()
        winner = {"id": "y_095", "gains": (1., .95, 1.)}
        lock = {"candidate": winner, "further_tuning_allowed": False}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)
            (output / "originals").mkdir()
            original_path = output / "originals/seed_44.pt"
            torch.save({"model_spec": model.specification(), "model_state": model.state_dict(),
                        "optimizer_state": {"stale": True}, "train_model_state": model.state_dict(),
                        "best_validation_deg": 123.}, original_path)
            original_hash = sha256_file(original_path)
            shutil.copy2(original_path, output / "candidate_model.pt")
            candidate_hash = write_candidate(output, winner, lock)
            reloaded, payload = load_checkpoint(output / "candidate_model.pt")
            expected = apply_gains(model, winner["gains"])
            self.assertEqual(sha256_file(original_path), original_hash)
            self.assertEqual(candidate_hash, sha256_file(output / "candidate_model.pt"))
            self.assertNotEqual(candidate_hash, original_hash)
            self.assertFalse((output / "candidate_model.tmp").exists())
            self.assertEqual(set(payload), {"model_spec", "model_state", "weight_edit"})
            self.assertFalse(payload["weight_edit"]["training_performed"])
            self.assertEqual(payload["weight_edit"]["source_sha256"], original_hash)
            self.assertEqual(payload["weight_edit"]["selection"], lock)
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(histories()), expected(histories()), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
