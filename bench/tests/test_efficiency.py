"""MAC counter, parameter counts and operator audit on tiny graphs with hand-computed answers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from bench import efficiency
from bench.tests import helpers


def single_node(op, inputs, output_shape, initializers=(), outputs=("y",), **attributes):
    """Graph around one node; `inputs` maps name -> shape for graph inputs (fed with ones)."""
    node = helper.make_node(op, list(inputs) + [t.name for t in initializers], list(outputs), **attributes)
    graph = helper.make_graph([node], op, [helper.make_tensor_value_info(n, helpers.FLOAT, s) for n, s in inputs.items()],
                              [helper.make_tensor_value_info(outputs[0], helpers.FLOAT, output_shape)], list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    feeds = {n: np.ones(s, np.float32) for n, s in inputs.items()}
    return model, feeds


def macs(model, feeds):
    return efficiency.count_macs(model, efficiency.runtime_shapes(model, feeds))


def weights(name, *shape):
    return helpers.const(name, np.full(shape, 0.01, np.float32))


class MACs(unittest.TestCase):
    def test_matmul(self):
        m = macs(*single_node("MatMul", {"a": [1, 5, 3]}, [1, 5, 7], [weights("b", 3, 7)]))
        self.assertEqual(m["by_operator"], {"MatMul": 5 * 7 * 3})

    def test_gemm_plain_and_transposed(self):
        self.assertEqual(macs(*single_node("Gemm", {"a": [2, 3]}, [2, 4], [weights("b", 3, 4)]))["total"], 2 * 4 * 3)
        self.assertEqual(macs(*single_node("Gemm", {"a": [3, 2]}, [2, 4], [weights("b", 3, 4)], transA=1))["total"], 2 * 4 * 3)

    def test_conv1d_dilated_and_depthwise(self):
        m = macs(*single_node("Conv", {"x": [1, 2, 10]}, [1, 4, 6], [weights("w", 4, 2, 3)], dilations=[2]))
        self.assertEqual(m["total"], 4 * 6 * 2 * 3)  # out [1, 4, 6] x (C_in 2 x kernel 3)
        m = macs(*single_node("Conv", {"x": [1, 4, 10]}, [1, 4, 8], [weights("w", 4, 1, 3)], group=4))
        self.assertEqual(m["total"], 4 * 8 * 1 * 3)

    def test_gru_lstm_rnn(self):
        # seq 6, batch 1, input 3, hidden 5: per step gates*hidden*(input + hidden)
        for op, gates in (("GRU", 3), ("LSTM", 4), ("RNN", 1)):
            model, feeds = single_node(op, {"x": [6, 1, 3]}, [6, 1, 1, 5],
                                       [weights("W", 1, gates * 5, 3), weights("R", 1, gates * 5, 5)], hidden_size=5)
            m = macs(model, feeds)
            self.assertEqual(m["by_operator"], {op: 6 * gates * 5 * (3 + 5)}, op)
            self.assertEqual(m["recurrences"][0]["steps"], 6)

    def test_einsum(self):
        model, feeds = single_node("Einsum", {"a": [1, 2, 3], "b": [1, 3, 4]}, [1, 2, 4], equation="bij,bjk->bik")
        self.assertEqual(macs(model, feeds)["total"], 1 * 2 * 3 * 4)


class Audit(unittest.TestCase):
    def test_flags_avoid_float64_custom_and_long_recurrence(self):
        nodes = [helper.make_node("TopK", ["x", "k"], ["values", "indices"]),
                 helper.make_node("Cast", ["values"], ["double"], to=TensorProto.DOUBLE),
                 helper.make_node("Cast", ["double"], ["y"], to=TensorProto.FLOAT),
                 helper.make_node("Magic", ["y"], ["z"], domain="com.example")]
        graph = helper.make_graph(nodes, "bad", [helper.make_tensor_value_info("x", helpers.FLOAT, [1, 8])],
                                  [helper.make_tensor_value_info("z", helpers.FLOAT, None)],
                                  [helpers.const("k", np.array([2], np.int64))])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.example", 1)])
        audit = efficiency.audit_operators(model, [{"op": "GRU", "name": "g", "steps": 400}])
        self.assertEqual(audit["avoid"], ["TopK"])
        self.assertEqual(audit["custom_domain"], ["com.example:Magic"])
        self.assertTrue(audit["float64_tensors"])
        self.assertIn("GRU over 400 steps (> 64)", audit["issues"])
        self.assertIn("float64", audit["issues"])

    def test_clean_graph_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = helpers.hold_last_graph(Path(tmp) / "h.onnx")
            report = efficiency.report(path)
        self.assertEqual(report["operators"]["issues"], [])
        self.assertEqual(report["operators"]["safe"], ["Mul", "Slice"])
        self.assertEqual(report["operators"]["other"], ["Tile"])
        self.assertEqual(report["initializer_elements"], 1 + 1 + 1 + 3 + 1)
        self.assertEqual(report["macs_per_call"]["total"], 0)
        self.assertTrue(report["output_checks"]["deterministic"])
        self.assertEqual(report["output_checks"]["calls"], 64)

    def test_stream_parity_of_identical_maths(self):
        with tempfile.TemporaryDirectory() as tmp:
            parity = efficiency.stream_parity(helpers.counter_graph(Path(tmp) / "c.onnx"),
                                              helpers.hold_last_graph(Path(tmp) / "h.onnx"), windows=50)
        self.assertEqual(parity["windows"], 50)
        self.assertLess(parity["max_difference_deg"], 1e-5)


if __name__ == "__main__":
    unittest.main()
