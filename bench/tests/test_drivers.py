"""ONNX drivers: any tensor names, re-normalisation, stateful feedback by name, resets, submissions."""
import tempfile
import textwrap
import unittest
from pathlib import Path

import h5py
import numpy as np
from onnx import helper

from bench import data
from bench.baselines import HoldLast
from bench.drivers import OnnxModel, Submission
from bench.scoring import BaselinePredictor, score
from bench.tests import helpers


class Drivers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.h5 = helpers.make_h5(self.dir / "d.h5", {"u1": [("r", 1000, True)], "u2": [("r", 700, True)]})
        self.definition = helpers.definition(["u1", "u2"])

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, key):
        with h5py.File(self.h5, "r") as h5:
            return data.read_segment(h5, key).astype(np.float32)

    def test_stateless_any_names_matches_baseline(self):
        model = OnnxModel(helpers.hold_last_graph(self.dir / "h.onnx", "foo", "bar", scale=2.0))
        self.assertFalse(model.stateful)
        self.assertEqual((model.context_name, model.prediction_name), ("foo", "bar"))
        with data.Benchmark(self.definition, self.h5) as b:
            onnx_errors = score(model, b, progress=None)[1]
            reference = score(BaselinePredictor(HoldLast()), b, progress=None)[1]
        np.testing.assert_allclose(onnx_errors, reference, atol=1e-4)  # scale 2 removed by re-normalisation

    def test_stateful_feedback_by_name_and_reset(self):
        model = OnnxModel(helpers.counter_graph(self.dir / "c.onnx"))
        self.assertTrue(model.stateful)
        q32 = self.segment("u1/r/segment_000")
        for protocol, calls, scored in (("from-start", 980, 581), ("cold-start", 581, 581)):
            blocks = list(model.predict_segment(q32, 400, 20, protocol))
            self.assertEqual(sum(len(x) for x in blocks), scored)
            np.testing.assert_array_equal(model.state["state_in_0"], np.full((1, 3), calls))
            np.testing.assert_array_equal(model.state["state_in_1"], np.full((2, 2), 2 * calls))
        model.reset()
        self.assertFalse(model.state["state_in_0"].any())

    def test_stateful_protocols_score_the_same_windows(self):
        model = OnnxModel(helpers.counter_graph(self.dir / "c.onnx"))
        with data.Benchmark(self.definition, self.h5) as b:
            a = score(model, b, "from-start", progress=None)[0]
            c = score(model, b, "cold-start", progress=None)[0]
            with self.assertRaises(ValueError):
                score(model, b, "stateless", progress=None)
        self.assertEqual(a["windows"], c["windows"])
        self.assertAlmostEqual(a["mean_deg_step20"], c["mean_deg_step20"], places=9)  # its output ignores state

    def test_unpaired_state_is_rejected(self):
        nodes, inits = helpers._hold_last_nodes("context", "prediction", 1.0)
        graph = helper.make_graph(nodes + [helper.make_node("Identity", ["state_in_0"], ["unused"])], "bad",
                                  [helper.make_tensor_value_info("context", helpers.FLOAT, [1, 400, 4]),
                                   helper.make_tensor_value_info("state_in_0", helpers.FLOAT, [1, 3])],
                                  [helper.make_tensor_value_info("prediction", helpers.FLOAT, [1, 20, 4]),
                                   helper.make_tensor_value_info("unused", helpers.FLOAT, [1, 3])], inits)
        with self.assertRaises(ValueError):
            OnnxModel(helpers.save(graph, self.dir / "bad.onnx"))

    def test_submission_folder_matches_bare_onnx(self):
        folder = self.dir / "TEAM_headpose"
        folder.mkdir()
        helpers.hold_last_graph(folder / "model_fp32.onnx", "ctx", "pred", scale=1.0)
        (folder / "run_inference.py").write_text(textwrap.dedent('''
            import numpy as np, onnxruntime as ort
            from pathlib import Path
            def load_model(model_dir):
                return ort.InferenceSession(str(Path(model_dir) / "model_fp32.onnx"), providers=["CPUExecutionProvider"])
            def predict(model, context):
                return model.run(None, {"ctx": np.asarray(context, np.float32)[None]})[0][0]
        '''), encoding="utf-8")
        q32 = self.segment("u2/r/segment_000")
        bare = OnnxModel(folder / "model_fp32.onnx").stream(q32)
        wrapped = np.concatenate(list(Submission(folder).predict_segment(q32, 400, 20)))
        np.testing.assert_array_equal(bare, wrapped)


if __name__ == "__main__":
    unittest.main()
