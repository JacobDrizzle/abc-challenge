"""v7: compression options, defaults equal to v6, the operator-reduced streaming export, and isolation from the
frozen v5/v6 code.

python -m unittest experimental.v7_compact.test_v7 -v   (from jacob/training)
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from baseline_pipeline import EvaluationConfig, geodesic_degrees, sha256_file
import run_next
from tcnn.damped import DampedVelocity
from tcnn.onnx_export import audit_operators, count_macs
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories

from experimental.v6_gravity_mirror.configuration import normalize_config as normalize_v6
from experimental.v6_gravity_mirror.model import GravityMirrorTCNN, load_checkpoint as load_v6, mirror
from experimental.v6_gravity_mirror.test_v6 import predict
from experimental.v7_compact import hooks
from experimental.v7_compact.configuration import V7_DEFAULTS, normalize_config
from experimental.v7_compact import mirror_split
from experimental.v7_compact.export import export_streaming_onnx
from experimental.v7_compact.model import CompactTCNN, load_checkpoint, receptive_field

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V6_CHECKPOINT = ROOT/"outputs/v6_group_a_v2/selected_model.pt"
V6 = dict(input_features="body_velocity", num_blocks=7, effective_context=256, damped_prior_tau_ms=100,
          gravity_features=True, mirror_equivariant=True)
# Each option alone, then all together; base 3 with 5 blocks sees 243 samples, inside the 256-sample window.
OPTIONS = {"v6": {}, "l1": dict(speed_feature="l1"), "basis": dict(head_basis=3),
           "base3": dict(dilation_base=3, num_blocks=5), "separable": dict(separable=True),
           "no_gravity": dict(gravity_features=False, speed_feature="l1"),
           "all": dict(speed_feature="l1", head_basis=2, dilation_base=3, num_blocks=5, separable=True)}
EXPECTED_PARAMS = {"control": 26012, "speed_l1": 26012, "basis_head": 24771, "base3_5": 19052,
                   "width_20": 18320, "narrow_full": 11972, "separable_24": 10388, "narrow_base3": 8868}
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *hooks.MODEL_KEYS)


def trained_like(**kwargs):
    torch.manual_seed(0)
    model = CompactTCNN(**{"dropout": 0, **V6, **kwargs})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)  # Non-zero head, so the network path is tested too.
    return model.eval()


def suite_configs():
    with hooks.installed():
        return run_next.experiment_configs(json.loads((HERE/"configs/v7_run.json").read_text()),
                                           json.loads((HERE/"configs/tcnn_v7.json").read_text()))


def build(config):
    return CompactTCNN(**{key: config[key] for key in MODEL_KEYS}).eval()


def expected_macs(model):
    """MACs of the v7 streaming graph's Conv, MatMul and Gemm nodes, term by term (needs the damped prior)."""
    sk, features = model.streams * model.stream_channels, 7 if model.gravity_features else 4
    block = 3 * sk + sk * sk if model.separable else 3 * sk * sk
    p = model.head_basis
    head = sk * 60 if p is None else sk * 3 * p + 20 * 3 * (p + 1)  # coefficients, then [basis, elapsed] MatMul
    speed = 24 if model.speed_feature == "l1" else 3
    gravity = 16 + 48 if model.gravity_features else 0
    # Fixed maths: step quaternion 64 + 16, u 3, |v|^2 60, output 64 + 320.
    return sk * features + model.num_blocks * block + head + speed + gravity + 83 + 60 + 384


class ConfigTests(unittest.TestCase):
    def test_suite_control_is_v6_control_with_the_full_schedule(self):
        configs = suite_configs()
        self.assertEqual(list(configs), list(EXPECTED_PARAMS))
        v6 = normalize_v6(json.loads((ROOT/"experimental/v6_gravity_mirror/configs/tcnn_v6.json").read_text()))
        self.assertEqual(configs["control"], {**v6, "patience": 40, **V7_DEFAULTS})
        for name, config in configs.items():
            self.assertGreaterEqual(config["patience"], config["epochs"], name)  # never stops early

    def test_v7_keys_are_checked(self):
        base = json.loads((HERE/"configs/tcnn_v7.json").read_text())
        self.assertEqual(normalize_config(normalize_config(base)), normalize_config(base))
        self.assertEqual(normalize_config({**base, "channels": 40})["channels"], 40)
        for bad in ({"channels": 41}, {"channels": "48"}, {"channels": 0}, {"dilation_base": 5},
                    {"separable": 1}, {"head_basis": 0}, {"head_basis": 21}, {"head_basis": True},
                    {"speed_feature": "l2"}, {"dilation_base": 3, "num_blocks": 6}, {"num_blocks": 8},
                    {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**base, **bad})

    def test_receptive_field(self):
        self.assertEqual([receptive_field(7), receptive_field(5, 3), receptive_field(4, 4)], [255, 243, 171])

    def test_hooks_restore_every_name(self):
        import benchmark_latency, run_desktop, tcnn_model_adapter, train_tcnn
        modules = (benchmark_latency, run_desktop, run_next, tcnn_model_adapter, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertIs(tcnn_model_adapter.TinyTCNN, CompactTCNN)
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.V6_FILES + hooks.V7_FILES):],
                             hooks.V6_FILES + hooks.V7_FILES)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.V6_FILES + hooks.V7_FILES:
            self.assertTrue((ROOT/name).exists(), name)

    def test_frozen_v5_and_v6_code_still_match_their_suite_manifests(self):
        for suite in ("v5_group_a_v2", "v6_group_a_v2"):
            manifest = ROOT/"outputs"/suite/"suite.json"
            if not manifest.exists():
                continue
            for name, digest in json.loads(manifest.read_text())["code_sha256"].items():
                self.assertEqual(sha256_file(ROOT/name), digest, f"{suite}: {name}")


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        for name, config in suite_configs().items():
            self.assertEqual(sum(p.numel() for p in build(config).parameters()), EXPECTED_PARAMS[name], name)

    def test_defaults_are_the_v6_model(self):
        history = torch.from_numpy(random_walk(4, 400, 5))
        torch.manual_seed(7)
        v6 = GravityMirrorTCNN(channels=48, **V6).eval()
        torch.manual_seed(7)
        v7 = CompactTCNN(channels=48, **V6).eval()
        self.assertEqual(list(v6.state_dict()), list(v7.state_dict()))
        for key, value in v6.state_dict().items():
            self.assertTrue(torch.equal(value, v7.state_dict()[key]), key)
        with torch.no_grad():
            for p in (*v6.parameters(), *v7.parameters()):
                p.copy_(torch.sin(torch.arange(p.numel()).reshape(p.shape) * .7) * .05)
            self.assertTrue(torch.equal(v6(history), v7(history)))
        spec = v7.specification()
        self.assertEqual({k: spec[k] for k in v6.specification()}, v6.specification())
        self.assertEqual({k: spec[k] for k in V7_DEFAULTS}, V7_DEFAULTS)

    def test_v6_checkpoint_loads_with_identical_predictions(self):
        if not V6_CHECKPOINT.exists():
            self.skipTest("No v6 checkpoint on this machine")
        history = smooth_histories(8)
        self.assertTrue(np.array_equal(predict(load_v6(V6_CHECKPOINT)[0], history),
                                       predict(load_checkpoint(V6_CHECKPOINT)[0], history)))

    def test_untrained_models_equal_the_damped_baseline(self):
        history = smooth_histories(8)
        expected = DampedVelocity(EvaluationConfig(), 1, 100).predict_batch(history)
        for name, config in suite_configs().items():
            with self.subTest(name):
                self.assertLess(geodesic_degrees(predict(build(config), history), expected).max(), 1e-3)

    def test_mirror_equivariance_is_exact_for_every_option(self):
        history = smooth_histories(6)
        for name, options in OPTIONS.items():
            with self.subTest(name):
                model = trained_like(channels=16, **options)
                mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
                expected = mirror(torch.from_numpy(predict(model, history))).numpy()
                self.assertLess(geodesic_degrees(mirrored, expected).max(), 1e-3)

    def test_l1_speed_replaces_only_the_speed_channel(self):
        window = torch.from_numpy(random_walk(2, 256, 3))
        features, _, _ = trained_like(channels=16, speed_feature="l1").features(window)
        reference, _, _ = trained_like(channels=16).features(window)
        self.assertTrue(torch.allclose(features[..., 3], features[..., :3].abs().sum(-1)))
        self.assertTrue(torch.equal(features[..., :3], reference[..., :3]))
        self.assertTrue(torch.equal(features[..., 4:], reference[..., 4:]))

    def test_head_basis_outputs_are_polynomials_in_time(self):
        model = trained_like(channels=16, head_basis=3)
        time = torch.arange(1, 21.) / 20
        self.assertTrue(torch.allclose(model.basis, torch.stack((time, time**2, time**3), 1)))
        with torch.no_grad():
            residuals = model.readout(torch.randn(5, 16)).reshape(5, 20, 3)
        self.assertEqual(torch.linalg.matrix_rank(residuals.permute(0, 2, 1).reshape(-1, 20)).item(), 3)

    def test_receptive_field_is_what_the_model_sees(self):
        # The output uses samples t - 243 ... t: the oldest one through the velocity of the step after it.
        # Weights of std .5: with .05, the separable path from the oldest sample (one depthwise weight per
        # block) shrinks below float32 resolution and the output does not change at all.
        history = smooth_histories(1)
        for separable in (False, True):
            model = trained_like(channels=8, dilation_base=3, num_blocks=5, separable=separable)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(10)
            base = predict(model, history)
            for index, changes in ((399 - 243, True), (399 - 244, False)):
                with self.subTest(separable=separable, index=index):
                    changed = history.copy()
                    changed[0, index] = (.5, .5, .5, .5)
                    self.assertEqual(not np.array_equal(predict(model, changed), base), changes)


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class StreamingTests(unittest.TestCase):
    def test_streaming_graph_equals_the_model_once_the_window_is_full(self):
        stream = random_walk(1, 650, 4)[0]
        for name, options in OPTIONS.items():
            for mirrored in (True, False):
                model = trained_like(channels=8, **{"mirror_equivariant": mirrored, **options})
                with self.subTest(name, mirror=mirrored), tempfile.TemporaryDirectory() as folder:
                    self.check_stream(model, stream, Path(folder)/"s.onnx")

    def test_v6_checkpoint_streams_through_the_reduced_graph(self):
        if not V6_CHECKPOINT.exists():
            self.skipTest("No v6 checkpoint on this machine")
        with tempfile.TemporaryDirectory() as folder:
            self.check_stream(load_checkpoint(V6_CHECKPOINT)[0], smooth_histories(1, 650)[0], Path(folder)/"s.onnx")

    def check_stream(self, model, stream, path):
        path, shapes = export_streaming_onnx(model, path)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        self.assertEqual(names, ["context", *(f"state_in_{i}" for i in range(len(shapes)))])
        states, predictions = [np.zeros(s, np.float32) for s in shapes], {}
        for t in range(1, len(stream)):
            window = stream[max(0, t - 399):t + 1]
            window = np.concatenate((np.repeat(window[:1], 400 - len(window), 0), window))[None]
            outputs = session.run(None, {"context": window, **dict(zip(names[1:], states))})
            predictions[t], states = outputs[0], outputs[1:]
        steps = range(399, len(stream))
        reference = predict(model, np.stack([stream[t - 399:t + 1] for t in steps]))
        got = np.concatenate([predictions[t] for t in steps])
        self.assertEqual(got.shape[1:], (20, 4))
        self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
        self.assertLess(np.abs(np.linalg.norm(got, axis=-1) - 1).max(), 1e-5)  # left unnormalised
        s, k, n, base = model.streams, model.stream_channels, model.num_blocks, model.dilation_base
        self.assertEqual([tuple(x) for x in shapes], [(s, k, 2 * base**i) for i in range(n)])
        self.assertEqual(count_macs(path)["total"], expected_macs(model))
        audit = audit_operators(path)
        self.assertEqual(audit["other"], [] if model.speed_feature == "l1" else ["Sqrt"])
        self.assertEqual((audit["avoid"], audit["slow_on_npu"], audit["float64_tensors"]), ([], [], []))
        if model.mirror_equivariant:
            # The file stores each block's shared weight once: [K, 2K, 3], or the separable [K, 2K, 1].
            import onnx
            from onnx import numpy_helper
            graph = onnx.load(str(path)).graph
            weights = [numpy_helper.to_array(t).shape for t in graph.initializer]
            weights += [numpy_helper.to_array(node.attribute[0].t).shape for node in graph.node
                        if node.op_type == "Constant" and node.attribute[0].name == "value"]
            shared = (k, 2 * k, 1 if model.separable else 3)
            self.assertEqual(sum(w == shared for w in weights), n)
            self.assertFalse(any(w == (2 * k, 2 * k, shared[2]) for w in weights))


def stream_predictions(path, shapes, stream):
    """[steps, 20, 4] predictions of a streaming graph fed one sample per call, from the first full window on."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    states, predictions = [np.zeros(s, np.float32) for s in shapes], []
    for t in range(1, len(stream)):
        window = stream[max(0, t - 399):t + 1]
        window = np.concatenate((np.repeat(window[:1], 400 - len(window), 0), window))[None]
        outputs = session.run(None, {"context": window, **dict(zip(names[1:], states))})
        states = outputs[1:]
        if t >= 399:
            predictions.append(outputs[0])
    return np.concatenate(predictions)


def initializer_shapes(path):
    """Shapes of a graph's initializers: the trained weights (fixed tables are Constant nodes)."""
    import onnx
    from onnx import numpy_helper
    return [numpy_helper.to_array(t).shape for t in onnx.load(str(path)).graph.initializer]


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class MirrorSplitTests(unittest.TestCase):
    def test_split_graph_is_the_reduced_graph_at_half_the_block_macs(self):
        stream = random_walk(1, 650, 4)[0]
        steps = range(399, len(stream))
        for name, options in OPTIONS.items():
            for mirrored in (True, False):
                # K = 6 matches no input-feature count (4 or 7), so the weight shapes below are unambiguous.
                model = trained_like(channels=12, **{"mirror_equivariant": mirrored, **options})
                with self.subTest(name, mirror=mirrored), tempfile.TemporaryDirectory() as folder:
                    reduced, shapes = export_streaming_onnx(model, Path(folder)/"reduced.onnx")
                    split, split_shapes = mirror_split.export_streaming_onnx(model, Path(folder)/"split.onnx")
                    self.assertEqual(split_shapes, shapes)
                    got = stream_predictions(split, shapes, stream)
                    reference = predict(model, np.stack([stream[t - 399:t + 1] for t in steps]))
                    self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
                    self.assertLess(geodesic_degrees(got, stream_predictions(reduced, shapes, stream)).max(), 1e-4)
                    self.check_graph(model, reduced, split)

    def test_v6_checkpoint_streams_through_the_split_graph(self):
        if not V6_CHECKPOINT.exists():
            self.skipTest("No v6 checkpoint on this machine")
        model, stream = load_checkpoint(V6_CHECKPOINT)[0], smooth_histories(1, 650)[0]
        with tempfile.TemporaryDirectory() as folder:
            reduced, shapes = export_streaming_onnx(model, Path(folder)/"reduced.onnx")
            split, _ = mirror_split.export_streaming_onnx(model, Path(folder)/"split.onnx")
            got = stream_predictions(split, shapes, stream)
            reference = predict(model, np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))]))
            self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
            self.check_graph(model, reduced, split)

    def check_graph(self, model, reduced, split):
        k, n, taps = model.stream_channels, model.num_blocks, 1 if model.separable else 3
        sk = model.streams * k
        # Each block's [[A, B], [B, A]] conv, sk x sk per tap, becomes two K x K convs: half the MACs.
        halved = n * taps * sk * sk // 2 if model.mirror_equivariant else 0
        self.assertEqual(count_macs(split)["total"], expected_macs(model) - halved)
        before, after = audit_operators(reduced), audit_operators(split)
        for key in ("other", "avoid", "slow_on_npu", "float64_tensors"):
            self.assertEqual(after[key], before[key], key)
        weights = initializer_shapes(split)
        if model.mirror_equivariant:
            # The stored [A, B] per block is replaced by (A + B)/2 and (A - B)/2: the same 2K^2 numbers per tap.
            self.assertEqual(sum(w == (k, k, taps) for w in weights), 2 * n)
            self.assertFalse(any(w == (k, 2 * k, taps) for w in weights))
        self.assertEqual(sum(int(np.prod(w)) for w in weights),
                         sum(int(np.prod(w)) for w in initializer_shapes(reduced)))


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class QuantizeTests(unittest.TestCase):
    def test_only_the_network_is_quantised_and_the_interface_stays_float(self):
        from experimental.v7_compact.quantize import calibration_inputs, network_nodes, quantize_stream, stream_errors
        segments = [random_walk(1, 600, seed)[0] for seed in (1, 2)]
        for name in ("v6", "separable", "all"):
            for mirrored in (True, False):
                model = trained_like(channels=8, **{"mirror_equivariant": mirrored, **OPTIONS[name]})
                with self.subTest(name, mirror=mirrored), tempfile.TemporaryDirectory() as folder:
                    float_path, shapes = export_streaming_onnx(model, Path(folder)/"f.onnx")
                    nodes = network_nodes(float_path, model)
                    self.assertEqual(len(nodes), model.num_blocks * (1 + model.separable) + 1)
                    self.assertNotIn("/projection/Conv", nodes)
                    inputs = calibration_inputs(float_path, shapes, segments, 8, every=20)
                    self.assertEqual(len(inputs), 8)
                    path = quantize_stream(float_path, Path(folder)/"q.onnx", model, inputs, "int8_minmax")
                    graph = onnx_graph(path)
                    # One DequantizeLinear per int8 weight (biases are int32; activations get Q/DQ pairs).
                    int8 = {t.name for t in graph.initializer if t.data_type == 3}
                    weights = sum(n.op_type == "DequantizeLinear" and n.input[0] in int8 for n in graph.node)
                    self.assertEqual(weights, len(nodes))
                    for value in (*graph.input, *graph.output):
                        self.assertEqual(value.type.tensor_type.elem_type, 1, value.name)  # float32
                    errors = stream_errors({"float": (float_path, shapes), "int8": (path, shapes)}, segments)
                    self.assertEqual(len(errors["int8"]), sum(len(q) - 419 for q in segments))
                    self.assertTrue(np.isfinite(errors["int8"]).all())


def onnx_graph(path):
    import onnx
    return onnx.load(str(path)).graph


class ConfirmTests(unittest.TestCase):
    def test_candidates_are_the_screened_models_with_both_add_ons(self):
        from experimental.v7_compact.confirm import candidate_configs
        plan = json.loads((HERE/"configs/v7_confirm.json").read_text())
        with hooks.installed():
            configs = candidate_configs(plan, json.loads((HERE/"configs/tcnn_v7.json").read_text()))
            # The screened candidate's config must match its screen exactly, or run_experiment refuses its folder.
            screen = run_next.experiment_configs(json.loads((HERE/"configs/v7_extra.json").read_text()),
                                                 json.loads((HERE/"configs/tcnn_v7.json").read_text()))
        self.assertEqual(configs["separable_base3_addons"], screen["separable_base3_addons"])
        params = {name: sum(p.numel() for p in CompactTCNN(**{k: c[k] for k in MODEL_KEYS}).parameters())
                  for name, c in configs.items()}
        self.assertEqual(params, {"separable_base3_addons": 6651, "base3_5_addons": 17811})
        for config in configs.values():
            self.assertEqual((config["seeds"], config["epochs"], config["patience"]), ([42, 43, 44], 40, 40))
            self.assertEqual((config["speed_feature"], config["head_basis"]), ("l1", 3))

    def test_plan_rule_picks_the_smallest_candidate_within_tolerance(self):
        from experimental.v7_compact.confirm import summarise
        small = {"params": 6651, "scores": {"42": 4.386, "43": 4.381, "44": 4.371}}  # mean 4.3793
        large = {"params": 17811, "scores": {"42": 4.341, "43": 4.338, "44": 4.330}}
        summary = summarise({"small": small, "large": large}, 4.382)
        self.assertEqual(summary["plan_rule_finalist"], "small")
        self.assertAlmostEqual(summary["rows"][0]["mean_deg"], 4.3793333, places=6)
        self.assertEqual(summarise({"small": {**small, "scores": {"42": 4.39, "43": 4.38, "44": 4.39}},
                                    "large": large}, 4.382)["plan_rule_finalist"], "large")
        self.assertIsNone(summarise({"small": {**small, "scores": {"42": 4.4, "43": 4.4, "44": 4.4}}},
                                    4.382)["plan_rule_finalist"])


if __name__ == "__main__":
    unittest.main()
