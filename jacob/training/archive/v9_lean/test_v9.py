"""v9: block layouts and maps (defaults equal v7), the recipe options, the lean streaming graph, and isolation from
the frozen v5, v6 and v7 code.

python -m unittest experimental.v9_lean.test_v9 -v   (from jacob/training)
"""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from baseline_pipeline import geodesic_degrees, sha256_file
import run_desktop
import run_next
from smoke_tcnn import make_fixture
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import body_velocity_features, conjugate, qmul, rotation_loss
from tcnn.onnx_export import audit_operators, count_macs
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories
import train_tcnn

from experimental.v6_gravity_mirror.model import gravity_in_head_frame, mirror
from experimental.v6_gravity_mirror.test_v6 import predict
from experimental.v7_compact import hooks as v7_hooks
from experimental.v7_compact.configuration import normalize_config as normalize_v7
from experimental.v7_compact.model import CompactTCNN, load_checkpoint as load_v7
from experimental.v7_compact.test_v7 import stream_predictions
from experimental.v9_lean import hooks
from experimental.v9_lean.configuration import V9_DEFAULTS, normalize_config
from experimental.v9_lean.distill import ensemble, forecast_cache, save_cache
from experimental.v9_lean.export import export_streaming_onnx
from experimental.v9_lean.model import MAPS, LeanTCNN, deployed, load_checkpoint, map_quaternion, receptive_field
from experimental.v9_lean.timing_stress import sample_positions, slerp_at, stressed
from experimental.v9_lean.training import TeacherChannel, TrainingWindows, distillation_loss, jitter_offsets, resampled

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BEST_CHECKPOINT = ROOT/"outputs/v8_recipe_sep_base3_addons/angular_loss/seed_44/best.pt"
# The current best, sep_base3_addons, as model options.
BEST = dict(input_features="body_velocity", num_blocks=5, effective_context=256, damped_prior_tau_ms=100,
            gravity_features=True, mirror_equivariant=True, dilation_base=3, separable=True, head_basis=3,
            speed_feature="l1")
LAYOUTS = {"v7": {}, "rf_324": dict(kernel_sizes=[3, 3, 3, 3, 4], dilations=[1, 3, 9, 27, 81], effective_context=400),
           "k5_4blocks": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], effective_context=400),
           "k8_3blocks": dict(num_blocks=3, kernel_sizes=[8] * 3, dilations=[1, 8, 47], effective_context=400)}
EXPECTED_PARAMS = {"control": 6651, "rf_397": 6699, "rf_324": 6675, "sampling_sqrt": 6651, "sampling_window": 6651,
                   "tilt_15": 6651, "endpoint_1": 6651, "native_maps": 6651, "k5_4blocks": 5595,
                   "k8_3blocks_w32": 7555}
MODEL_KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
              "input_features", "num_blocks", *hooks.MODEL_KEYS)


def trained_like(**kwargs):
    torch.manual_seed(0)
    model = LeanTCNN(**{"dropout": 0, **BEST, **kwargs})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)  # Non-zero head, so the network path is tested too.
    return model.eval()


def suite_configs():
    with hooks.installed():
        return run_next.experiment_configs(json.loads((HERE/"configs/v9_screen.json").read_text()),
                                           json.loads((HERE/"configs/tcnn_v9.json").read_text()))


def build(config):
    return LeanTCNN(**{key: config[key] for key in MODEL_KEYS}).eval()


def expected_macs(model, fused=False, padded=False):
    """MACs of the v9 streaming graph's Conv and MatMul nodes, term by term."""
    k, p, lean = model.stream_channels, model.head_basis, model.maps != "native"
    features, gravity = (7, 64) if model.gravity_features else (4, 0)
    pointwise = 6 * k * k if padded else 4 * k * k  # two output rows of a (3, 1) or a (2, 1) kernel
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
    def test_screen_control_is_the_current_best_with_v9_defaults(self):
        configs = suite_configs()
        self.assertEqual(list(configs), list(EXPECTED_PARAMS))
        best = normalize_v7({**json.loads((ROOT/"experimental/v7_compact/configs/tcnn_v7_sep_base3_addons.json")
                                          .read_text()), "loss_kind": "angular"})
        ran = ROOT/"outputs/v8_recipe_sep_base3_addons/angular_loss/experiment.json"
        if ran.exists():
            self.assertEqual(json.loads(ran.read_text())["config"], best)
        self.assertEqual(configs["control"], {**best, **V9_DEFAULTS})
        fields = {"rf_397": 397, "rf_324": 324, "k5_4blocks": 397, "k8_3blocks_w32": 393}
        for name, config in configs.items():
            self.assertGreaterEqual(config["patience"], config["epochs"], name)  # never stops early
            self.assertEqual(receptive_field(config["num_blocks"], config["dilation_base"], config["kernel_sizes"],
                                             config["dilations"]), fields.get(name, 243), name)

    def test_v9_keys_are_checked(self):
        base = json.loads((HERE/"configs/tcnn_v9.json").read_text())
        self.assertEqual(normalize_config(normalize_config(base)), normalize_config(base))
        long = {"kernel_sizes": [3, 3, 3, 3, 5], "dilations": [1, 3, 9, 27, 79]}
        self.assertEqual(normalize_config({**base, **long, "effective_context": 400})["kernel_sizes"], [3, 3, 3, 3, 5])
        self.assertEqual(normalize_config({**base, "jitter_augmentation_ms": 3})["jitter_augmentation_ms"], 3)
        for bad in ({"kernel_sizes": [3, 3, 3, 3]}, {"kernel_sizes": [3, 3, 3, 3, 1]}, {"kernel_sizes": [3, 3, 3, 3, 3.]},
                    {"dilations": [1, 3, 9, 27, 0]}, long, {"kernel_sizes": [3] * 5, "separable": False},
                    {"maps": "cayley"}, {"sampling": "users"}, {"tilt_augmentation_deg": -1},
                    {"tilt_augmentation_deg": 46}, {"tilt_augmentation_deg": True}, {"distill_beta": .5},
                    {"distill_beta": 1.5, "teacher": "t"}, {"teacher": "t"}, {"distill_beta": .5, "teacher": 3},
                    {"jitter_augmentation_ms": -1}, {"jitter_augmentation_ms": 5}, {"jitter_augmentation_ms": True},
                    {"jitter_augmentation_ms": 2, "distill_beta": .5, "teacher": "t"}, {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**base, **bad})

    def test_receptive_field_is_what_the_model_sees(self):
        # The output uses samples t - rf ... t: the oldest one through the rate of the step after it. Weights of
        # std .5, as in v7's test: smaller ones shrink the oldest sample's path below float32 resolution.
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
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.V9_FILES):], hooks.V9_FILES)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.V9_FILES:
            self.assertTrue((ROOT/name).exists(), name)

    def test_frozen_v5_v6_and_v7_code_still_match_their_suite_manifests(self):
        for manifest in sorted((ROOT/"outputs").glob("v[5-8]_*/suite.json")):
            for name, digest in json.loads(manifest.read_text())["code_sha256"].items():
                self.assertEqual(sha256_file(ROOT/name), digest, f"{manifest.parent.name}: {name}")


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        for name, config in suite_configs().items():
            self.assertEqual(sum(p.numel() for p in build(config).parameters()), EXPECTED_PARAMS[name], name)

    def test_defaults_are_the_v7_model(self):
        history = torch.from_numpy(random_walk(4, 400, 5))
        torch.manual_seed(7)
        v7 = CompactTCNN(channels=48, **BEST).eval()
        torch.manual_seed(7)
        v9 = LeanTCNN(channels=48, **BEST).eval()
        self.assertEqual(list(v7.state_dict()), list(v9.state_dict()))
        for key, value in v7.state_dict().items():
            self.assertTrue(torch.equal(value, v9.state_dict()[key]), key)
        with torch.no_grad():
            for parameter in v7.parameters():
                parameter.normal_(0, .05)
            v9.load_state_dict(v7.state_dict())
            self.assertTrue(torch.equal(v7(history), v9(history)))

    def test_best_checkpoint_loads_with_identical_predictions(self):
        if not BEST_CHECKPOINT.exists():
            self.skipTest("No sep_base3_addons checkpoint on this machine")
        history = smooth_histories(8)
        np.testing.assert_array_equal(predict(load_checkpoint(BEST_CHECKPOINT)[0], history),
                                      predict(load_v7(BEST_CHECKPOINT)[0], history))

    def test_mirror_equivariance_is_exact_for_every_option(self):
        history = smooth_histories(6)
        for name, layout in LAYOUTS.items():
            for maps in MAPS:
                with self.subTest(name, maps=maps):
                    model = trained_like(channels=16, maps=maps, **layout)
                    mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
                    expected = mirror(torch.from_numpy(predict(model, history))).numpy()
                    self.assertLess(geodesic_degrees(mirrored, expected).max(), 1e-3)

    def test_rates_under_each_map(self):
        # Constant rotation of theta a step about one axis, with sign flips: exact theta, lean ~theta, native sin.
        axis = np.array([.48, .6, .64])
        for degrees, lean_error in ((6, 1e-5), (20, 6e-4)):
            theta = math.radians(degrees)
            t = np.arange(10)[:, None]
            q = np.concatenate((np.cos(t * theta / 2), np.sin(t * theta / 2) * axis), 1)
            q *= np.where(np.arange(10) % 3 == 1, -1., 1.)[:, None]
            window = torch.tensor(q[None], dtype=torch.float32)
            step = {maps: LeanTCNN(**BEST, maps=maps).features(window)[2][0].double().numpy() for maps in MAPS}
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
    def test_lean_graph_equals_the_deployed_model_once_the_window_is_full(self):
        stream = random_walk(1, 650, 4)[0]
        cases = {**LAYOUTS, "native": dict(maps="native"), "lean": dict(maps="lean"),
                 "no_gravity": dict(gravity_features=False), "native_k5": dict(maps="native", **LAYOUTS["k5_4blocks"])}
        for name, options in cases.items():
            for fused, padded in itertools.product((False, True), repeat=2):
                model = trained_like(channels=8, **options)
                with self.subTest(name, fused=fused, padded=padded), tempfile.TemporaryDirectory() as folder:
                    self.check_stream(model, stream, Path(folder)/"s.onnx", fused, padded)

    def test_best_checkpoint_streams_through_the_lean_graph(self):
        if not BEST_CHECKPOINT.exists():
            self.skipTest("No sep_base3_addons checkpoint on this machine")
        stream = smooth_histories(1, 650)[0]
        windows = np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))])
        exact = predict(load_checkpoint(BEST_CHECKPOINT)[0], windows)
        model = deployed(load_checkpoint(BEST_CHECKPOINT)[0])
        with tempfile.TemporaryDirectory() as folder:
            got = self.check_stream(model, stream, Path(folder)/"s.onnx")
        # Track A's output map is 0.094 degrees off at 60; this fast synthetic stream forecasts up to 57 degrees.
        self.assertLess(geodesic_degrees(got, exact).max(), .1)

    def check_stream(self, model, stream, path, fused=False, padded=False):
        import onnx
        from onnx import numpy_helper
        path, shapes = export_streaming_onnx(model, path, fused_head=fused, padded_rows=padded)
        got = stream_predictions(path, shapes, stream)
        reference = predict(deployed(model), np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))]))
        self.assertEqual(got.shape[1:], (20, 4))
        self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
        self.assertLess(np.abs(np.linalg.norm(got, axis=-1) - 1).max(), 1e-5)  # normalised by LayerNormalization
        k, p, n = model.stream_channels, model.head_basis, model.num_blocks
        self.assertEqual([tuple(s) for s in shapes], [(1, k, 2, block.left_padding) for block in model.blocks])
        graph = onnx.load(str(path)).graph
        fixed = 39 - 4 * (not model.gravity_features) - 6 * (model.maps == "native")
        # In the file a padded block has one weight-building Concat, and A4 four weight-building nodes; ORT folds
        # them when the session starts, leaving 7 nodes a padded block and 3 fewer in the A4 head.
        self.assertEqual(sum(node.op_type != "Constant" for node in graph.node), (9 - padded) * n + fixed + fused)
        self.assertEqual(executed_nodes(path), (9 - 2 * padded) * n + fixed - 3 * fused)
        audit = audit_operators(path)
        self.assertEqual((audit["other"], audit["avoid"], audit["slow_on_npu"], audit["float64_tensors"]),
                         ([], [], [], []))
        # Every weight is stored once; without A4 the P head x-biases sit in a Constant.
        stored = sum(numpy_helper.to_array(t).size for t in graph.initializer)
        params = sum(q.numel() for q in model.parameters())
        self.assertEqual(stored, params - (0 if fused else p))
        self.assertEqual(count_macs(path)["total"], expected_macs(model, fused, padded))
        return got


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

    def test_training_with_v9_defaults_is_training_with_v7(self):
        config = tiny(json.loads((HERE/"configs/tcnn_v9.json").read_text()))
        with v7_hooks.installed():
            run_desktop.train(self.data, self.split, self.directory/"v7", config, 42, "cpu", synthetic=True)
        with hooks.installed():
            run_desktop.train(self.data, self.split, self.directory/"v9", config, 42, "cpu", synthetic=True)
        v7, v9 = (torch.load(self.directory/name/"last.pt", weights_only=True) for name in ("v7", "v9"))
        self.assertEqual(v7["history"][-1]["validation_endpoint_deg"], v9["history"][-1]["validation_endpoint_deg"])
        for key, value in v7["model_state"].items():
            self.assertTrue(torch.equal(value, v9["model_state"][key]), key)

    def test_recipe_options_train(self):
        folder, _ = self.teacher()
        base = json.loads((HERE/"configs/tcnn_v9.json").read_text())
        names = dict(vars(train_tcnn))
        for name, overrides in {"layout": dict(LAYOUTS["k5_4blocks"], maps="native"),
                                "recipe": dict(sampling="sqrt", tilt_augmentation_deg=15, endpoint_weight=1.,
                                               jitter_augmentation_ms=3),
                                "distill": dict(distill_beta=.5, teacher=str(folder))}.items():
            with self.subTest(name), hooks.installed():
                run_desktop.train(self.data, self.split, self.directory/name, tiny(base, **overrides), 42, "cpu",
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


if __name__ == "__main__":
    unittest.main()
