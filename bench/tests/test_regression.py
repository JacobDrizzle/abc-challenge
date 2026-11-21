"""Regression: reproduce jacob/training's own test evaluation of v5/v6 (group-a-v2) on group-a-v2-test20.
Skipped unless the data and the ONNX files are available:

  BENCH_H5       data/h5/qualified_head_100hz_20261026_131953.h5 (SHA-256 a56c3a60...)
  BENCH_V5_ONNX  optional, default jacob/models/tcnn_v5_ga2_stream.onnx
  BENCH_V6_ONNX  optional, default jacob/models/tcnn_v6_ga2_stream.onnx
  BENCH_REF_DIR  optional: jacob/training/outputs (per-window comparison with v{5,6}_group_a_v2/test/endpoint_*.npy)

About 5 minutes: 931,051 batch-1 calls per ONNX model. The expected values are the full-precision means in
jacob/training/outputs/v{5,6}_group_a_v2/test/results.json, written by the trainer's evaluator
(baseline_pipeline.py), and its cohort fingerprint is the bench's legacy fingerprint. It scored v5/v6 from
their PyTorch checkpoints on stateless windows; here their stream graphs are fed one sample at a time from
segment start, so agreement is to float32 rounding. Nic's group-a-v2 models join PATHS and EXPECTED once
they are registered, with his evaluator's means as the expected values.
"""
import os
import unittest
from pathlib import Path

import numpy as np

from bench import contamination, data, efficiency
from bench.baselines import make_baseline
from bench.drivers import OnnxModel
from bench.scoring import BaselinePredictor, score

REPO = Path(__file__).resolve().parents[2]
LEGACY_FINGERPRINT = "015851ffd032833b294db919929d2ad4b141df1794c0f3e4ceaf4b9ac73a51d5"
# name: (expected mean at step 20, reference endpoint file under BENCH_REF_DIR, MACs per call, ONNX initializer elements)
EXPECTED = {
    "hold_last": (6.037257, "v5_group_a_v2/test/endpoint_hold_last.npy", None, None),
    "constant_velocity": (6.943825, "v5_group_a_v2/test/endpoint_constant_velocity_1step.npy", None, None),
    "damped_velocity": (5.113740, "v5_group_a_v2/test/endpoint_damped_velocity.npy", None, None),
    "v6_tcnn_ga2": (4.358376, "v6_group_a_v2/test/endpoint_tiny_tcnn.npy", 52_144, 25_992),
    "v5_tcnn_ga2": (4.453404, "v5_group_a_v2/test/endpoint_tiny_tcnn.npy", 51_920, 51_900),
}
JACOB = REPO / "jacob/models"

PATHS = {"v5_tcnn_ga2": os.environ.get("BENCH_V5_ONNX") or JACOB / "tcnn_v5_ga2_stream.onnx",
         "v6_tcnn_ga2": os.environ.get("BENCH_V6_ONNX") or JACOB / "tcnn_v6_ga2_stream.onnx"}
H5 = os.environ.get("BENCH_H5")
AVAILABLE = H5 and all(p and Path(p).exists() for p in (H5, *PATHS.values()))


@unittest.skipUnless(AVAILABLE, "set BENCH_H5 (and the ONNX files must exist) to run the regression")
class GroupAV2Test20(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = data.Benchmark("group-a-v2-test20", H5)
        cls.reference = os.environ.get("BENCH_REF_DIR")

    @classmethod
    def tearDownClass(cls):
        cls.bench.close()

    def test_cohort(self):
        self.assertEqual(self.bench.windows, 931_051)
        self.assertEqual(len(self.bench.segments), 72)
        self.assertEqual(self.bench.legacy_fingerprint, LEGACY_FINGERPRINT)

    def check(self, name, predictor, protocol):
        expected, reference, _, _ = EXPECTED[name]
        accuracy, endpoint, _ = score(predictor, self.bench, protocol, progress=None)
        print(f"\n  {name}: {accuracy['mean_deg_step20']:.6f} (expected {expected:.6f})", flush=True)
        self.assertAlmostEqual(accuracy["mean_deg_step20"], expected, delta=1e-4)
        if self.reference:
            old = np.load(Path(self.reference) / reference)
            difference = np.abs(endpoint - old)
            self.assertLess(difference.max(), 1e-3, name)
            self.assertLess(difference.mean(), 1e-5, name)

    def test_baselines(self):
        for name in ("hold_last", "constant_velocity", "damped_velocity"):
            with self.subTest(name):
                self.check(name, BaselinePredictor(make_baseline(name)), "stateless")

    def test_onnx_models(self):
        for name, path in PATHS.items():
            with self.subTest(name):
                model = OnnxModel(path, name)
                self.check(name, model, "from-start" if model.stateful else "stateless")

    def test_efficiency_and_contamination(self):
        registry = __import__("json").loads((REPO / "bench/models.json").read_text(encoding="utf-8"))
        for name, path in PATHS.items():
            with self.subTest(name):
                report = efficiency.report(path)
                self.assertEqual(report["macs_per_call"]["total"], EXPECTED[name][2])
                self.assertEqual(report["initializer_elements"], EXPECTED[name][3])
                self.assertEqual(report["stateful"], name.startswith("v"))
                self.assertTrue(report["output_checks"]["finite"])
                self.assertEqual(data.sha256_file(path), registry[name]["onnx_sha256"])
                check = contamination.check(self.bench.players, registry[name]["trained_on"])
                self.assertEqual(check["status"], "clean")


if __name__ == "__main__":
    unittest.main()
