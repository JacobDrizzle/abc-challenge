import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from tcnn.model import TinyTCNN
from tcnn.onnx_export import ExportableTCNN, audit_operators, count_macs, export_onnx, geodesic_degrees


def random_walk(batch, length, seed):
    """Unit quaternions with random sign flips, so sign alignment is exercised."""
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, .05, (batch, length, 4)), 1) + np.array([1., 0, 0, 0])
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(rng.random((batch, length, 1)) < .1, -1, 1)
    return q.astype(np.float32)


def trained_like(**kwargs):
    torch.manual_seed(0)
    model = TinyTCNN(**kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)  # Non-zero head, so the network path is tested too.
    return model.eval()


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class ONNXExportTests(unittest.TestCase):
    # effective_context=150 puts the strided grid on different offsets than 400 does.
    CASES = [dict(channels=8, velocity_features=True, damped_prior_tau_ms=100),
             dict(channels=8, velocity_features=False),
             dict(channels=8, velocity_features=True, effective_context=150)]

    def test_wrapper_and_onnx_match_pytorch(self):
        histories = random_walk(16, 400, 1)
        for case in self.CASES:
            for strided in (False, True):
                with self.subTest(strided=strided, **case):
                    self.check_case(trained_like(**case), histories, strided)

    def check_case(self, model, histories, strided):
        with tempfile.TemporaryDirectory() as folder:
            wrapper = ExportableTCNN(model, strided).eval()
            with torch.inference_mode():
                reference = model(torch.from_numpy(histories)).numpy()
                wrapped = np.concatenate([wrapper(torch.from_numpy(h[None])).numpy() for h in histories])
            path = export_onnx(model, Path(folder) / "model_fp32.onnx", strided=strided)
            self.assertFalse(model.training, "export must leave the model in eval mode")
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            exported = np.concatenate([session.run(None, {"context": h[None]})[0] for h in histories])
            self.assertEqual(session.get_inputs()[0].name, "context")
            self.assertEqual(session.get_outputs()[0].name, "prediction")
            self.assertEqual(session.run(None, {"context": histories[:1]})[0].shape, (1, 20, 4))
            self.assertLess(geodesic_degrees(wrapped, reference).max(), 1e-3)
            self.assertLess(geodesic_degrees(exported, reference).max(), 1e-3)

    def test_operator_audit_and_macs(self):
        n, c = 400, 8
        shared = 7*c*n + c*60 + (n-1)*(n-1)  # projection, head, sign prefix
        # Outputs per block: 400 each when full; 200, 100, 50, 25, 13, 7, 4, 2 when strided.
        expected = {False: shared + 8*c*c*3*n, True: shared + c*c*3*(200+100+50+25+13+7+4+2)}
        for strided in (False, True):
            with self.subTest(strided=strided), tempfile.TemporaryDirectory() as folder:
                model = trained_like(channels=c, velocity_features=True, damped_prior_tau_ms=100)
                path = export_onnx(model, Path(folder) / "model.onnx", strided=strided)
                audit = audit_operators(path)
                self.assertEqual(audit["avoid"], [])
                self.assertEqual(audit["float64_tensors"], [])
                self.assertIn("Conv", audit["safe"])
                self.assertEqual(count_macs(path)["total"], expected[strided])


if __name__ == "__main__":
    unittest.main()
