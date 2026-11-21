"""The best model: base config (k8_combined), model invariants and options, the streaming graph and its options,
int8 quantisation, the recipe options, confirmation and the timing stress test.

Ported from archive/v6_gravity_mirror/test_v6.py, v7_compact/test_v7.py and v9_lean/test_v9.py. Their tests that
compared one version with the one before (v9 defaults train as v7, and so on) proved the layering while both existed;
they are not repeated here.

python -m unittest best.test_best -v   (from jacob/training)
"""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from baseline_pipeline import EvaluationConfig, geodesic_degrees
import run_desktop
import run_next
from smoke_tcnn import make_fixture
from tcnn.damped import DampedVelocity
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import TinyTCNN, body_velocity_features, conjugate, qmul, rotation_loss
from tcnn.onnx_export import audit_operators, count_macs
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories
import train_tcnn

from best import hooks
from best.configuration import normalize_config
from best.confirm import candidate_configs, report, summarise
from best.distill import ensemble, forecast_cache, save_cache
from best.export import export_streaming_onnx
from best.model import (MAPS, LeanTCNN, deployed, gravity_in_head_frame, load_checkpoint, map_quaternion, mirror,
                        receptive_field)
from best.timing_stress import sample_positions, slerp_at, stressed, with_noise
from best.training import TeacherChannel, TrainingWindows, distillation_loss, jitter_offsets, resampled

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = json.loads((HERE/"configs/base.json").read_text())
# k5_combined, the smaller confirmed candidate, as overrides of base.json (k8_combined).
K5 = dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], channels=48)
# Confirmed folders on Jacob's PC (outputs/ is gitignored): their selected checkpoints and parameter counts.
CONFIRMED = {"v9_confirm/k8_combined": 7555, "v9_confirm/k5_combined": 5595,
             "v8_recipe_sep_base3_addons/angular_loss": 6651, "v6_group_a_v2": 26012, "v5_group_a_v2": 51900}
# The v5 body-velocity model's options; with every later option off, LeanTCNN is that model.
BODY = dict(input_features="body_velocity", num_blocks=7, effective_context=256, damped_prior_tau_ms=100)
# v7's sep_base3_addons (the best model before v9) as model options: kernel-3 separable blocks to vary below.
SEPARABLE = dict(input_features="body_velocity", num_blocks=5, effective_context=256, damped_prior_tau_ms=100,
                 gravity_features=True, mirror_equivariant=True, dilation_base=3, separable=True, head_basis=3,
                 speed_feature="l1")
LAYOUTS = {"kernel3": {}, "rf_324": dict(kernel_sizes=[3, 3, 3, 3, 4], dilations=[1, 3, 9, 27, 81], effective_context=400),
           "k5": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], effective_context=400),
           "k8": dict(num_blocks=3, kernel_sizes=[8] * 3, dilations=[1, 8, 47], effective_context=400)}
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *hooks.MODEL_KEYS)


def trained_like(**kwargs):
    torch.manual_seed(0)
    model = LeanTCNN(**{"dropout": 0, **SEPARABLE, **kwargs})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)  # Non-zero head, so the network path is tested too.
    return model.eval()


def build(config):
    return LeanTCNN(**{key: config[key] for key in MODEL_KEYS}).eval()


def predict(model, history):
    with torch.inference_mode():
        return model(torch.from_numpy(np.ascontiguousarray(history, dtype=np.float32))).numpy()


def rotate_world(rotation, q):
    """rotation * q for scalar-first quaternion arrays [..., 4]."""
    turned = rotation * Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]])
    return turned.as_quat()[:, [3, 0, 1, 2]].reshape(q.shape).astype(np.float32)


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


def expected_macs(model, fused=False, padded=False, split=False):
    """MACs of the streaming graph's Conv and MatMul nodes, term by term."""
    k, p, lean = model.stream_channels, model.head_basis, model.maps != "native"
    features, gravity = (7, 64) if model.gravity_features else (4, 0)
    # Pointwise: two output rows of a (2, 1) or (3, 1) kernel, or the split's two K x K Convs on one row each.
    pointwise = 2 * k * k if split else 6 * k * k if padded else 4 * k * k
    blocks = sum(2 * k * block.depthwise.kernel_size[0] + pointwise for block in model.blocks)
    head = 6 * p * k + (0 if fused else 6 * p)  # [W h0, W h1]; the ones MatMul
    basis = 60 * (p + 1 if fused else 2 * p + 1)
    # Fixed maths: d 64 + 16, lean bracket 4, speed 24, Cayley scalar 60, output [M, -M] 128 + 640.
    return 80 + 4 * lean + 24 + gravity + 2 * k * features + blocks + head + basis + 60 * lean + 768


def executed_nodes(path):
    """Non-Constant nodes left after ORT's basic optimisations (constant folding): the graph a session runs."""
    import onnx
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(Path(path).with_suffix(".optimised.onnx"))
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return sum(node.op_type != "Constant" for node in onnx.load(options.optimized_model_filepath).graph.node)


def tiny(config, **overrides):
    """A config small enough to train on the synthetic fixture in seconds."""
    return {**config, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20, **overrides}


class ConfigTests(unittest.TestCase):
    def test_base_config_is_k8_combined(self):
        self.assertEqual(normalize_config(BASE), BASE)  # every key is written out, so it is already normalised
        ran = ROOT/"outputs/v9_confirm/k8_combined/experiment.json"
        if ran.exists():
            self.assertEqual(json.loads(ran.read_text())["config"], BASE)
        self.assertEqual(receptive_field(BASE["num_blocks"], BASE["dilation_base"], BASE["kernel_sizes"],
                                         BASE["dilations"]), 393)
        self.assertGreaterEqual(BASE["patience"], BASE["epochs"])  # never stops early

    def test_screen_template_control_is_the_base(self):
        suite = json.loads((HERE/"configs/screen.json").read_text())
        with hooks.installed():
            configs = run_next.experiment_configs(suite, BASE)
        self.assertEqual(configs, {"control": BASE})
        # run_next.run_suite tunes the damped baseline over these before it trains: without them it stops.
        self.assertEqual(suite["baseline_lags"], [1, 3, 5, 10])
        self.assertEqual(suite["baseline_tau_ms"], [25, 50, 100, 200, 500, None])

    def test_keys_are_checked(self):
        self.assertEqual(normalize_config({**BASE, "jitter_augmentation_ms": 1.5})["jitter_augmentation_ms"], 1.5)
        self.assertEqual(normalize_config({**BASE, **K5})["kernel_sizes"], [5, 5, 5, 5])
        for bad in ({"gravity_features": 1}, {"mirror_equivariant": "yes"}, {"input_features": "anchor"},
                    {"channels": 63}, {"channels": "64"}, {"channels": 0}, {"dilation_base": 5}, {"separable": 1},
                    {"head_basis": 0}, {"head_basis": 21}, {"head_basis": True}, {"speed_feature": "l2"},
                    {"kernel_sizes": [8, 8]}, {"kernel_sizes": [8, 8, 1]}, {"kernel_sizes": [8, 8, 8.]},
                    {"dilations": [1, 8, 0]}, {"effective_context": 393}, {"separable": False},
                    {"maps": "cayley"}, {"sampling": "users"}, {"tilt_augmentation_deg": -1},
                    {"tilt_augmentation_deg": 46}, {"tilt_augmentation_deg": True}, {"distill_beta": .5},
                    {"distill_beta": 1.5, "teacher": "t"}, {"teacher": "t"}, {"distill_beta": .5, "teacher": 3},
                    {"jitter_augmentation_ms": -1}, {"jitter_augmentation_ms": 5}, {"jitter_augmentation_ms": True},
                    {"jitter_augmentation_ms": 2, "distill_beta": .5, "teacher": "t"}, {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**BASE, **bad})

    def test_receptive_field_is_what_the_model_sees(self):
        # The output uses samples t - rf ... t: the oldest one through the rate of the step after it. Weights of
        # std .5: smaller ones shrink the oldest sample's path below float32 resolution.
        history = smooth_histories(1)
        for name, layout in LAYOUTS.items():
            model = trained_like(channels=8, **{"effective_context": 400, **layout})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(10)
            rf = receptive_field(model.num_blocks, model.dilation_base, model.kernel_sizes, model.dilations)
            base = predict(model, history)
            for index, changes in ((399 - rf, True), (398 - rf, False)):
                with self.subTest(name, index=index):
                    changed = history.copy()
                    changed[0, index] = (.5, .5, .5, .5)
                    self.assertEqual(not np.array_equal(predict(model, changed), base), changes)

    def test_hooks_restore_every_name(self):
        import benchmark_latency, tcnn_model_adapter
        modules = (benchmark_latency, run_desktop, run_next, tcnn_model_adapter, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertIs(tcnn_model_adapter.TinyTCNN, LeanTCNN)
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.BEST_FILES):], hooks.BEST_FILES)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.BEST_FILES:
            self.assertTrue((ROOT/name).exists(), name)


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in build(BASE).parameters()), 7555)
        self.assertEqual(sum(p.numel() for p in build(normalize_config({**BASE, **K5})).parameters()), 5595)
        # The earlier models, which load_checkpoint builds from their specifications.
        for options, expected in ((SEPARABLE, 6651), (dict(BODY, gravity_features=True, mirror_equivariant=True), 26012),
                                  (BODY, 51900)):
            self.assertEqual(sum(p.numel() for p in LeanTCNN(channels=48, **options).parameters()), expected)

    def test_with_every_option_off_it_is_the_v5_model(self):
        history = torch.from_numpy(random_walk(4, 400, 5))
        torch.manual_seed(7)
        v5 = TinyTCNN(channels=48, **BODY).eval()
        torch.manual_seed(7)
        best = LeanTCNN(channels=48, **BODY).eval()
        self.assertEqual(list(v5.state_dict()), list(best.state_dict()))
        for key, value in v5.state_dict().items():
            self.assertTrue(torch.equal(value, best.state_dict()[key]), key)
        with torch.no_grad():
            for p in (*v5.parameters(), *best.parameters()):
                p.copy_(torch.sin(torch.arange(p.numel()).reshape(p.shape) * .7) * .05)
            self.assertTrue(torch.equal(v5(history), best(history)))

    def test_confirmed_checkpoints_load(self):
        history = smooth_histories(2)
        for folder, params in CONFIRMED.items():
            path = ROOT/"outputs"/folder/"selected_model.pt"
            if not path.exists():
                continue
            with self.subTest(folder):
                model, state = load_checkpoint(path)
                self.assertEqual(sum(p.numel() for p in model.parameters()), params)
                self.assertTrue(np.isfinite(predict(model, history)).all())

    def test_untrained_models_equal_the_damped_baseline(self):
        history = smooth_histories(8)
        expected = DampedVelocity(EvaluationConfig(), 1, 100).predict_batch(history)
        for options in (BODY, dict(BODY, gravity_features=True, mirror_equivariant=True), SEPARABLE):
            with self.subTest(**options):
                prediction = predict(LeanTCNN(channels=48, dropout=0, **options).eval(), history)
                self.assertLess(geodesic_degrees(prediction, expected).max(), 1e-3)

    def test_gravity_is_world_up_in_the_head_frame_and_sign_invariant(self):
        rotations = Rotation.random(500, random_state=2)
        q = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]])
        expected = rotations.inv().apply([0, 1, 0])
        self.assertLess(np.abs(gravity_in_head_frame(q).numpy() - expected).max(), 1e-12)
        self.assertTrue(torch.equal(gravity_in_head_frame(q), gravity_in_head_frame(-q)))

    def test_mirror_equivariance_is_exact_for_every_option(self):
        history = smooth_histories(6)
        cases = {**LAYOUTS, "kernel3_blocks": dict(BODY, gravity_features=True, mirror_equivariant=True,
                                                   dilation_base=2, separable=False, head_basis=None,
                                                   speed_feature="norm")}
        for name, layout in cases.items():
            for maps in MAPS:
                with self.subTest(name, maps=maps):
                    model = trained_like(channels=16, maps=maps, **layout)
                    mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
                    expected = mirror(torch.from_numpy(predict(model, history))).numpy()
                    self.assertLess(geodesic_degrees(mirrored, expected).max(), 1e-3)
        # Without the constraint, a trained-like model is far from equivariant.
        model = trained_like(channels=16, mirror_equivariant=False)
        mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
        self.assertGreater(geodesic_degrees(mirrored, mirror(torch.from_numpy(predict(model, history))).numpy()).max(), .1)

    def test_yaw_and_sign_invariant_but_tilt_sensitive(self):
        history = smooth_histories(4)
        flips = np.where(np.random.default_rng(1).random(history.shape[:2] + (1,)) < .3, -1, 1).astype(np.float32)
        yaw, tilt = Rotation.from_rotvec([0, 2.1, 0]), Rotation.from_rotvec([.5, 0, .3])
        cases = [dict(BODY, gravity_features=g, mirror_equivariant=m, separable=False, dilation_base=2,
                      head_basis=None, speed_feature="norm") for g in (True, False) for m in (True, False)]
        cases.append(dict(LAYOUTS["k8"], maps="native"))  # k8_combined's layout and maps, with gravity
        for options in cases:
            model = trained_like(channels=16, **options)
            base = predict(model, history)
            for name, rotation in (("yaw", yaw), ("tilt", tilt)):
                with self.subTest(turn=name, **options):
                    got = predict(model, rotate_world(rotation, history) * flips)
                    error = geodesic_degrees(got, rotate_world(rotation, base)).max()
                    if name == "tilt" and model.gravity_features:
                        self.assertGreater(error, .1)  # absolute tilt is an input
                    else:
                        self.assertLess(error, 1e-3)

    def test_l1_speed_replaces_only_the_speed_channel(self):
        window = torch.from_numpy(random_walk(2, 256, 3))
        features, _, _ = trained_like(channels=16, speed_feature="l1").features(window)
        reference, _, _ = trained_like(channels=16, speed_feature="norm").features(window)
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

    def test_rates_under_each_map(self):
        # Constant rotation of theta a step about one axis, with sign flips: exact theta, lean ~theta, native sin.
        axis = np.array([.48, .6, .64])
        for degrees, lean_error in ((6, 1e-5), (20, 6e-4)):
            theta = math.radians(degrees)
            t = np.arange(10)[:, None]
            q = np.concatenate((np.cos(t * theta / 2), np.sin(t * theta / 2) * axis), 1)
            q *= np.where(np.arange(10) % 3 == 1, -1., 1.)[:, None]
            window = torch.tensor(q[None], dtype=torch.float32)
            step = {maps: LeanTCNN(**SEPARABLE, maps=maps).features(window)[2][0].double().numpy() for maps in MAPS}
            np.testing.assert_allclose(step["exact"], theta * axis, rtol=1e-5)
            self.assertLess(np.abs(step["lean"] / (theta * axis) - 1).max(), lean_error)
            np.testing.assert_allclose(step["native"], math.sin(theta) * axis, rtol=1e-5)

    def test_output_quaternion_under_each_map(self):
        axis = torch.tensor([.48, .6, .64], dtype=torch.float64)
        identity = np.array([1., 0, 0, 0])
        for degrees, lean_error in ((20, 1e-3), (30, 4e-3), (60, .1)):
            r = math.radians(degrees) * axis
            exact, lean, native = (map_quaternion(r, maps).numpy() for maps in MAPS)
            self.assertAlmostEqual(geodesic_degrees(exact, identity), degrees, places=9)
            self.assertLess(geodesic_degrees(lean, exact), lean_error)
            self.assertAlmostEqual(geodesic_degrees(native, identity),
                                   math.degrees(2 * math.atan(math.radians(degrees) / 2)), places=9)


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class StreamingTests(unittest.TestCase):
    def test_graph_equals_the_deployed_model_once_the_window_is_full(self):
        stream = random_walk(1, 650, 4)[0]
        cases = {**LAYOUTS, "native": dict(maps="native"), "lean": dict(maps="lean"),
                 "no_gravity": dict(gravity_features=False), "k8_native": dict(maps="native", **LAYOUTS["k8"])}
        options = [(f, p, s, False) for f, p, s in itertools.product((False, True), repeat=3) if not (p and s)]
        options += [(False, False, False, True), (True, True, False, True)]  # st_layout on the lean and A4 graphs
        for name, layout in cases.items():
            for fused, padded, split, st in options:
                model = trained_like(channels=8, **layout)
                with (self.subTest(name, fused=fused, padded=padded, split=split, st=st),
                      tempfile.TemporaryDirectory() as folder):
                    self.check_stream(model, stream, Path(folder)/"s.onnx", fused, padded, split, st)

    def test_st_layout_changes_no_output(self):
        stream = random_walk(1, 650, 4)[0]
        model = trained_like(channels=8, maps="native", **LAYOUTS["k8"])
        with tempfile.TemporaryDirectory() as folder:
            plain, shapes = export_streaming_onnx(model, Path(folder)/"plain.onnx")
            st, _ = export_streaming_onnx(model, Path(folder)/"st.onnx", st_layout=True)
            self.assertTrue(np.array_equal(stream_predictions(st, shapes, stream),
                                           stream_predictions(plain, shapes, stream)))

    def test_split_mirror_is_the_lean_graph_at_half_the_pointwise_macs(self):
        import onnx
        from onnx import numpy_helper
        stream = random_walk(1, 650, 4)[0]
        # K = 6 matches no input-feature count (4 or 7), so the weight shapes below are unambiguous.
        model = trained_like(channels=12, maps="native", **LAYOUTS["k8"])
        with tempfile.TemporaryDirectory() as folder:
            lean, shapes = export_streaming_onnx(model, Path(folder)/"lean.onnx")
            split, split_shapes = export_streaming_onnx(model, Path(folder)/"split.onnx", split_mirror=True)
            self.assertEqual(split_shapes, shapes)
            self.assertLess(geodesic_degrees(stream_predictions(split, shapes, stream),
                                             stream_predictions(lean, shapes, stream)).max(), 1e-4)
            k, n = model.stream_channels, model.num_blocks
            self.assertEqual(count_macs(lean)["total"] - count_macs(split)["total"], 2 * k * k * n)
            # (A + B)/2 and (A - B)/2 replace each [A | B]: two [K, K, 1, 1] weights a block, none [K, K, 2, 1].
            weights = {f: [numpy_helper.to_array(t).shape for t in onnx.load(str(f)).graph.initializer]
                       for f in (lean, split)}
            self.assertEqual(sum(w == (k, k, 1, 1) for w in weights[split]), 2 * n)
            self.assertFalse(any(w == (k, k, 2, 1) for w in weights[split]))
            self.assertEqual(sum(int(np.prod(w)) for w in weights[split]),
                             sum(int(np.prod(w)) for w in weights[lean]))
        with self.assertRaises(ValueError):
            export_streaming_onnx(model, Path(folder)/"both.onnx", padded_rows=True, split_mirror=True)

    def test_k8_combined_streams_through_the_graph(self):
        path = ROOT/"outputs/v9_confirm/k8_combined/selected_model.pt"
        if not path.exists():
            self.skipTest("No k8_combined checkpoint on this machine")
        with tempfile.TemporaryDirectory() as folder:
            self.check_stream(load_checkpoint(path)[0], smooth_histories(1, 650)[0], Path(folder)/"s.onnx")

    def check_stream(self, model, stream, path, fused=False, padded=False, split=False, st=False):
        import onnx
        from onnx import numpy_helper
        path, shapes = export_streaming_onnx(model, path, fused_head=fused, padded_rows=padded, split_mirror=split,
                                             st_layout=st)
        got = stream_predictions(path, shapes, stream)
        reference = predict(deployed(model), np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))]))
        self.assertEqual(got.shape[1:], (20, 4))
        self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
        self.assertLess(np.abs(np.linalg.norm(got, axis=-1) - 1).max(), 1e-5)  # normalised by LayerNormalization
        k, p, n = model.stream_channels, model.head_basis, model.num_blocks
        self.assertEqual([tuple(s) for s in shapes], [(1, k, 2, block.left_padding) for block in model.blocks])
        graph = onnx.load(str(path)).graph
        # st_layout: the LayerNormalization sees [1, 1, 20, 8], the shape ST Edge AI Core 4.1 compiles correctly,
        # for two more nodes (the Reshapes around it).
        shapes_of = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim]
                     for value in onnx.shape_inference.infer_shapes(onnx.load(str(path))).graph.value_info}
        normalisation = [node for node in graph.node if node.op_type == "LayerNormalization"]
        self.assertEqual([shapes_of[node.input[0]] for node in normalisation], [[1, 1, 20, 8] if st else [1, 20, 8]])
        fixed = 39 + 2 * st - 4 * (not model.gravity_features) - 6 * (model.maps == "native")
        # In the file a padded block has one weight-building Concat, and A4 four weight-building nodes; ORT folds
        # them when the session starts, leaving 7 nodes a padded block and 3 fewer in the A4 head. A split block
        # has 15: its Slice and Concat become two Slices, s, d (Mul and Add), the second Conv and P + Q * signs.
        self.assertEqual(sum(node.op_type != "Constant" for node in graph.node),
                         (9 - padded + 6 * split) * n + fixed + fused)
        self.assertEqual(executed_nodes(path), (9 - 2 * padded + 6 * split) * n + fixed - 3 * fused)
        audit = audit_operators(path)
        self.assertEqual((audit["other"], audit["avoid"], audit["slow_on_npu"], audit["float64_tensors"]),
                         ([], [], [], []))
        # Every weight is stored once; without A4 the P head x-biases sit in a Constant.
        stored = sum(numpy_helper.to_array(t).size for t in graph.initializer)
        params = sum(q.numel() for q in model.parameters())
        self.assertEqual(stored, params - (0 if fused else p))
        self.assertEqual(count_macs(path)["total"], expected_macs(model, fused, padded, split))
        return got


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class QuantizeTests(unittest.TestCase):
    def test_only_the_network_is_quantised_and_the_interface_stays_float(self):
        import onnx
        from best.quantize import calibration_inputs, network_nodes, quantize_stream, stream_errors
        segments = [random_walk(1, 600, seed)[0] for seed in (1, 2)]
        for name, layout in (("kernel3", {}), ("k8_native", dict(maps="native", **LAYOUTS["k8"]))):
            model = trained_like(channels=8, **layout)
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                float_path, shapes = export_streaming_onnx(model, Path(folder)/"f.onnx")
                nodes = network_nodes(float_path, model)
                self.assertEqual(len(nodes), 2 * model.num_blocks + 1)
                graph = onnx.load(str(float_path)).graph
                projection = [node.name for node in graph.node if node.op_type == "Conv" and node.input[1] == "projection"]
                self.assertEqual(len(projection), 1)
                self.assertNotIn(projection[0], nodes)
                inputs = calibration_inputs(float_path, shapes, segments, 8, every=20)
                self.assertEqual(len(inputs), 8)
                path = quantize_stream(float_path, Path(folder)/"q.onnx", model, inputs, "int8_minmax")
                graph = onnx.load(str(path)).graph
                # One DequantizeLinear per int8 weight (biases are int32; activations get Q/DQ pairs).
                int8 = {t.name for t in graph.initializer if t.data_type == 3}
                weights = sum(n.op_type == "DequantizeLinear" and n.input[0] in int8 for n in graph.node)
                self.assertEqual(weights, len(nodes))
                for value in (*graph.input, *graph.output):
                    self.assertEqual(value.type.tensor_type.elem_type, 1, value.name)  # float32
                errors = stream_errors({"float": (float_path, shapes), "int8": (path, shapes)}, segments)
                self.assertEqual(len(errors["int8"]), sum(len(q) - 419 for q in segments))
                self.assertTrue(np.isfinite(errors["int8"]).all())


class ConfirmTests(unittest.TestCase):
    def test_k5_candidate_is_the_confirmed_k5_config(self):
        plan = {"version": 1, "seeds": [42, 43, 44], "reference": "outputs/v9_confirm/k8_combined",
                "candidates": {"k5_combined": {"output": "outputs/unused", "overrides": K5}}}
        with hooks.installed():
            configs = candidate_configs(plan, BASE)
        ran = ROOT/"outputs/v9_confirm/k5_combined/experiment.json"
        if ran.exists():  # k5 was confirmed on the v9 base config; on this base it needs only K5's overrides
            self.assertEqual(configs["k5_combined"], json.loads(ran.read_text())["config"])
        self.assertEqual(sum(p.numel() for p in build(configs["k5_combined"]).parameters()), 5595)
        self.assertEqual(receptive_field(4, 3, K5["kernel_sizes"], K5["dilations"]), 397)

    def test_report_without_a_reference(self):
        # A reference folder with no selection.json gives rows whose delta is None (the 800-user confirm, 2 Nov).
        result = {"params": 7555, "selected_seed": 44, "scores": {"42": 4.4111, "43": 4.4056, "44": 4.4036},
                  "cohort_fingerprints": ["f"]}
        row = summarise("full_pass", result)
        self.assertIsNone(row["delta_vs_reference_deg"])
        self.assertIn("| full_pass | 7,555 | 4.4111 | 4.4056 | 4.4036 | 4.4068 | 0.0039 | - | 44 |",
                      report([row], None, [42, 43, 44]))
        reference = summarise("k8 (reference)", result)
        text = report([summarise("candidate", result, reference["mean_deg"])], reference, [42, 43, 44])
        self.assertIn("| reference | 44 |", text)
        self.assertIn("| +0.0000 | 44 |", text)

    def test_confirm_template_compares_with_k8_combined(self):
        plan = json.loads((HERE/"configs/confirm.json").read_text())
        self.assertEqual((plan["version"], plan["seeds"], plan["candidates"]), (1, [42, 43, 44], {}))
        self.assertEqual(plan["reference"], "outputs/v9_confirm/k8_combined")


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data, self.split = make_fixture(self.directory/"fixture", samples=900)
        self.rows = check_corpus(self.data, self.split, minimum_users=3, minimum_recordings=3, synthetic=True)["segments"]
        self.train_rows = [r for r in self.rows if r["split"] == "train"]

    def tearDown(self):
        self.temp.cleanup()

    def unequal_users(self):
        """Three users of 500, 2,000 and 8,000 samples, each held at a constant orientation that names it."""
        path, rows = self.directory/"unequal.h5", []
        with h5py.File(path, "w") as h5:
            for name, n, identity in [("a", 500, [1, 0, 0, 0]), ("b", 2000, [0, 1, 0, 0]), ("c", 8000, [0, 0, 1, 0])]:
                key = f"{name}/recording/segment_000"
                h5.create_dataset(key+"/quaternion_wxyz", data=np.tile(np.float32(identity), (n, 1)))
                rows.append({"user": name, "key": key, "path": key, "samples": n})
        return path, rows

    def teacher(self):
        """A teacher cache for the fixture's training windows, from the damped-velocity forecast."""
        windows = DeviceWindows(self.data, self.train_rows, "cpu")
        damped = train_tcnn.reference_predictors(20)["damped_velocity"]
        return save_cache(self.directory/"teacher", forecast_cache(windows, damped, 50), windows, {}), damped

    def test_default_batches_are_device_windows_batches(self):
        for data, rows in ((self.data, self.train_rows), self.unequal_users()):
            ours = TrainingWindows(data, rows, "cpu").batches(42, 3, 4, 16)
            for (x, y), (a, b) in zip(ours, DeviceWindows(data, rows, "cpu").batches(42, 3, 4, 16)):
                self.assertTrue(torch.equal(x, a) and torch.equal(y, b))

    def test_users_are_drawn_in_proportion_to_their_weights(self):
        path, rows = self.unequal_users()
        windows = np.array([500, 2000, 8000]) - 419
        for sampling, power in (("user", 0), ("sqrt", .5), ("window", 1)):
            x = torch.cat([b[0] for b in TrainingWindows(path, rows, "cpu", sampling=sampling).batches(42, 0, 40, 500)])
            share = [float((x[:, 0, i] == 1).double().mean()) for i in range(3)]
            np.testing.assert_allclose(share, windows ** power / (windows ** power).sum(), atol=.015, err_msg=sampling)

    def test_tilt_moves_only_gravity_and_keeps_the_loss(self):
        plain = TrainingWindows(self.data, self.train_rows, "cpu").batches(42, 0, 2, 64)
        tilted = TrainingWindows(self.data, self.train_rows, "cpu", tilt_deg=15).batches(42, 0, 2, 64)
        angles = []
        for (x, y), (a, b) in zip(plain, tilted):
            turn = qmul(a, conjugate(x))  # each window's world rotation, the same for every sample of it
            turn = turn * torch.where(turn[..., :1] < 0, -1., 1.)
            torch.testing.assert_close(turn, turn[:, :1].expand_as(turn), atol=1e-6, rtol=0)
            future = qmul(b, conjugate(y))
            torch.testing.assert_close(future * torch.where(future[..., :1] < 0, -1., 1.), turn[:, :20], atol=1e-6, rtol=0)
            self.assertLess(turn[:, 0, 2].abs().max().item(), 1e-6)  # about a horizontal axis: y is up
            angle = torch.rad2deg(2 * torch.atan2(turn[:, 0, 1:].norm(dim=-1), turn[:, 0, 0])).numpy()
            angles.append(angle)
            torch.testing.assert_close(body_velocity_features(a)[0], body_velocity_features(x)[0], atol=2e-3, rtol=1e-4)
            moved = (gravity_in_head_frame(a[:, -1]) - gravity_in_head_frame(x[:, -1])).norm(dim=-1).numpy()
            self.assertTrue((moved[angle == 0] == 0).all() and (moved[angle > .1] > 1e-4).all())
            guess = torch.nn.functional.normalize(y + .05 * torch.randn_like(y), dim=-1)
            torch.testing.assert_close(rotation_loss(qmul(turn[:, :20], guess), b, "angular", .5),
                                       rotation_loss(guess, y, "angular", .5), atol=1e-5, rtol=0)
        angles = np.concatenate(angles)
        self.assertLessEqual(angles.max(), 15 + 1e-3)
        self.assertTrue(.3 < (angles == 0).mean() < .7)  # about half of the windows stay untilted

    def test_jitter_moves_each_sample_along_the_motion_and_keeps_the_windows(self):
        offsets = jitter_offsets(4000, 420, 3, torch.Generator().manual_seed(0))
        width = offsets.abs().amax(1)
        self.assertLessEqual(offsets.abs().max().item(), .3)  # 3 ms is .3 of a 10 ms sample
        self.assertTrue(.02 < (width < .1).float().mean().item() < .5)  # per-window widths spread over 0 to .3
        plain = TrainingWindows(self.data, self.train_rows, "cpu").batches(42, 0, 2, 64)
        jittered = TrainingWindows(self.data, self.train_rows, "cpu", jitter_ms=4).batches(42, 0, 2, 64)
        for (x, y), (a, b) in zip(plain, jittered):
            grid, moved = torch.cat((x, y), 1), torch.cat((a, b), 1)
            same = resampled(grid, torch.zeros(grid.shape[:2]))
            self.assertLess(geodesic_degrees(same.numpy(), grid.numpy()).max(), 1e-3)
            self.assertLess(np.abs(np.linalg.norm(moved.numpy(), axis=-1) - 1).max(), 1e-5)
            # Each sample moves at most .4 of a step (4 ms) towards a neighbour: the same windows, resampled.
            steps = geodesic_degrees(grid[:, 1:].numpy(), grid[:, :-1].numpy())
            reach = .4 * np.maximum(np.pad(steps, ((0, 0), (1, 0))), np.pad(steps, ((0, 0), (0, 1)))) + 1e-3
            shift = geodesic_degrees(moved.numpy(), grid.numpy())
            self.assertTrue((shift <= reach).all())
            self.assertGreater(shift.mean(), 0)

    def test_teacher_forecast_follows_each_window_and_its_tilt(self):
        folder, damped = self.teacher()
        channel = TeacherChannel()
        windows = TrainingWindows(self.data, self.train_rows, "cpu", tilt_deg=15, teacher=str(folder), channel=channel)
        for history, _ in windows.batches(42, 0, 3, 32):
            # damped velocity commutes with world rotations, so the cached forecast, tilted, is its forecast.
            self.assertLess(geodesic_degrees(channel.forecast.numpy(), damped(history).numpy()).max(), 1e-3)
        others = [r for r in self.rows if r["split"] == "validation"]
        with self.assertRaisesRegex(ValueError, "other training windows"):
            next(TrainingWindows(self.data, others, "cpu", teacher=str(folder), channel=channel).batches(42, 0, 1, 4))
        np.save(folder/"teacher.npy", np.load(folder/"teacher.npy") * 2)
        with self.assertRaisesRegex(ValueError, "changed"):
            next(TrainingWindows(self.data, self.train_rows, "cpu", teacher=str(folder), channel=channel)
                 .batches(42, 0, 1, 4))

    def test_distillation_loss_weights_truth_and_teacher(self):
        torch.manual_seed(0)
        prediction, target, forecast = (torch.nn.functional.normalize(torch.randn(4, 20, 4), dim=-1) for _ in range(3))
        channel = TeacherChannel()
        channel.forecast = forecast
        truth, teacher = rotation_loss(prediction, target, "angular", .5), rotation_loss(prediction, forecast, "angular", .5)
        for beta in (0, .5, 1):
            torch.testing.assert_close(distillation_loss(beta, channel)(prediction, target, "angular", .5),
                                       (1 - beta) * truth + beta * teacher)

    def test_ensemble_ignores_member_signs(self):
        q = torch.nn.functional.normalize(torch.randn(3, 20, 4), dim=-1)
        torch.testing.assert_close(ensemble([q, -q, q]), q)

    def test_base_and_recipe_options_train(self):
        folder, _ = self.teacher()
        names = dict(vars(train_tcnn))
        for name, overrides in {"base": {}, "k5": K5,
                                "recipe": dict(sampling="sqrt", tilt_augmentation_deg=0, endpoint_weight=1.,
                                               jitter_augmentation_ms=3, maps="lean"),
                                "distill": dict(distill_beta=.5, teacher=str(folder))}.items():
            with self.subTest(name), hooks.installed():
                run_desktop.train(self.data, self.split, self.directory/name, tiny(BASE, **overrides), 42, "cpu",
                                  synthetic=True)
                state = torch.load(self.directory/name/"last.pt", weights_only=True)
                self.assertEqual(state["epoch"], 2)
                self.assertTrue(all(k in state["model_spec"] for k in ("kernel_sizes", "dilations", "maps")))
        self.assertEqual({k: v for k, v in vars(train_tcnn).items() if k in names}, names)


class TimingStressTests(unittest.TestCase):
    @staticmethod
    def spin(length, degrees):
        """[length, 4] float64: a constant rotation of `degrees` a grid step about one axis, with sign flips."""
        half = np.radians(degrees) / 2 * np.arange(length)[:, None]
        q = np.concatenate((np.cos(half), np.sin(half) * np.array([.48, .6, .64])), 1)
        return torch.from_numpy(q * np.where(np.arange(length) % 3 == 1, -1., 1.)[:, None])

    def test_the_100_hz_stream_is_the_grid(self):
        tau = sample_positions(1000, np.random.default_rng(0))
        np.testing.assert_array_equal(tau, np.arange(1000))
        q = self.spin(1000, 2)
        # 1e-5 degrees: arccos of a float64 dot within an ulp of 1 is already ~2e-6 degrees.
        self.assertLess(geodesic_degrees(slerp_at(q, torch.from_numpy(tau)).numpy(), q.numpy()).max(), 1e-5)

    def test_slerp_at_is_the_trainers_resampling(self):
        q = self.spin(420, 3)
        offsets = torch.from_numpy(np.random.default_rng(1).uniform(-.3, .3, 420))
        got = slerp_at(q, (torch.arange(420, dtype=torch.float64) + offsets).clamp(0, 419))
        inner = slice(1, -1)  # resampled extrapolates at the ends, where slerp_at's positions are clipped
        self.assertLess(geodesic_degrees(got[inner].numpy(), resampled(q[None], offsets[None])[0, inner].numpy()).max(),
                        1e-5)

    def test_a_rate_scales_each_step(self):
        q = self.spin(1000, 2)
        for rate in (90, 95, 105, 110):
            with self.subTest(rate=rate):
                tau = sample_positions(1000, np.random.default_rng(0), rate_hz=rate)
                self.assertLessEqual(tau[-1], 999)
                self.assertGreater(tau[-1] + 100 / rate, 999)  # the last tick inside the segment
                out = slerp_at(q, torch.from_numpy(tau)).numpy()
                np.testing.assert_allclose(geodesic_degrees(out[1:], out[:-1]), 2 * 100 / rate, rtol=1e-7)

    def test_drops_skip_ticks_and_repeats_hold_a_sample(self):
        tau = sample_positions(5000, np.random.default_rng(0), drop=.05)
        gaps = np.diff(tau)
        self.assertTrue(set(np.unique(gaps)) <= {1., 2., 3.} and tau[0] == 0)
        self.assertTrue(.03 < (gaps > 1).mean() < .07)
        tau = sample_positions(5000, np.random.default_rng(0), repeat=.05)
        late = np.arange(5000) - tau  # a repeated sample shows the pose one tick before its own
        self.assertTrue(set(np.unique(late)) == {0., 1.} and late[0] == 0)
        self.assertTrue(.03 < late.mean() < .07)

    def test_jitter_stays_within_its_width_and_the_segment(self):
        for scenario in ({"jitter_ms": 3}, {"jitter_ms": 2, "repeat": .01}, {"rate_hz": 98, "jitter_ms": 1}):
            with self.subTest(**scenario):
                tau = sample_positions(3000, np.random.default_rng(0), **scenario)
                nominal = np.arange(len(tau)) * 100 / scenario.get("rate_hz", 100)
                self.assertLessEqual(np.abs(tau - nominal).max(), scenario["jitter_ms"] / 10 + 1 + 1e-12)
                self.assertTrue(tau.min() >= 0 and tau.max() <= 2999)

    def test_stressed_streams_cut_short_segments(self):
        buffer = torch.cat((self.spin(430, 1), self.spin(2000, 2))).float()
        out, bounds = stressed(buffer, [(0, 430), (430, 2000)], 420, {"rate_hz": 95}, 0)
        self.assertEqual(bounds, [(0, len(out))])  # 430 grid samples at 95 Hz give 409 < 420
        self.assertEqual(len(out), math.floor(1999 * .95 + 1e-9) + 1)
        same, kept = stressed(buffer, [(0, 430), (430, 2000)], 420, {}, 0)
        self.assertEqual(kept, [(0, 430), (430, 2000)])
        self.assertLess(geodesic_degrees(same.numpy(), buffer.numpy()).max(), 1e-3)

    def test_noise_turns_each_sample_by_its_rms_angle(self):
        q = self.spin(20000, 2)
        for degrees in (.005, .05):
            with self.subTest(degrees=degrees):
                noisy = with_noise(q, degrees, np.random.default_rng(0))
                self.assertLess(np.abs(np.linalg.norm(noisy.numpy(), axis=-1) - 1).max(), 1e-12)
                turned = geodesic_degrees(noisy.numpy(), q.numpy())
                self.assertAlmostEqual(np.sqrt((turned ** 2).mean()) / degrees, 1, delta=.03)
                # Independent from one sample to the next: neighbouring errors are uncorrelated.
                error = qmul(conjugate(q), noisy)[:, 1:].numpy()
                self.assertLess(abs(np.corrcoef(error[1:, 0], error[:-1, 0])[0, 1]), .03)

    def test_noise_comes_after_the_timing_and_leaves_other_scenarios_alone(self):
        buffer, bounds = torch.cat((self.spin(600, 1), self.spin(2000, 2))).float(), [(0, 600), (600, 2000)]
        jittered, kept = stressed(buffer, bounds, 420, {"jitter_ms": 2}, 0)
        both, kept_both = stressed(buffer, bounds, 420, {"jitter_ms": 2, "noise_deg": .02}, 0)
        self.assertEqual(kept_both, kept)
        # The first segment's timing draws come before any noise draw, so its positions are the jitter-only ones.
        turned = geodesic_degrees(both[:600].double().numpy(), jittered[:600].double().numpy())
        self.assertAlmostEqual(np.sqrt((turned ** 2).mean()) / .02, 1, delta=.15)
        zero, _ = stressed(buffer, bounds, 420, {"jitter_ms": 2, "noise_deg": 0}, 0)
        self.assertTrue(torch.equal(zero, jittered))


if __name__ == "__main__":
    unittest.main()
