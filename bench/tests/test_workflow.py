"""Contamination specs, bootstrap, latency session, result files and an end-to-end CLI run on synthetic data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench import cli, contamination, data, results, stats
from bench.latency import session
from bench.tests import helpers

REPO = Path(__file__).resolve().parents[2]


class Contamination(unittest.TestCase):
    def test_nic_rule_reproduces_his_split_file(self):
        rows = (REPO / "nic/splits/splits_fri.csv").read_text(encoding="utf-8").splitlines()[1:]
        pairs = [r.split(",") for r in rows]
        self.assertEqual(len(pairs), 504)
        self.assertTrue(all(contamination.nic_split(p) == s for p, s in pairs))

    def test_specs_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "p.txt"
            listing.write_text("# trained on\nu1\nu2\n", encoding="utf-8")
            table = Path(tmp) / "s.csv"
            table.write_text("player_id,split\nu3,train\nu4,test\nu5,val\n", encoding="utf-8")
            split = Path(tmp) / "split.json"
            split.write_text(json.dumps({"train_users": ["u6"], "validation_users": ["u7"], "test_users": ["u8"]}))
            bench = ["u1", "u4", "u5", "u7", "u8"]
            self.assertEqual(contamination.check(bench, [str(listing)])["overlap"], ["u1"])
            self.assertEqual(contamination.check(bench, [f"csv:{table}"])["overlap"], ["u5"])
            self.assertEqual(contamination.check(bench, [f"csv:{table}#test"])["overlap"], ["u4"])
            self.assertEqual(contamination.check(bench, [f"split-json:{split}"])["overlap"], ["u7"])
            both = contamination.check(bench, [str(listing), f"split-json:{split}"])
            self.assertEqual((both["status"], both["overlap"]), ("contaminated", ["u1", "u7"]))
            self.assertEqual(contamination.check(bench, ["none"])["status"], "clean")
            self.assertEqual(contamination.check(bench, [])["status"], "undeclared")

    def test_rule_spec(self):
        players = ["02914184-90cb-498d-b9e6-74f1c377af33", "00a85435-a5d4-4ce3-b452-3d99b6f8b399"]  # test, train
        self.assertEqual(contamination.check(players, ["rule:sha1:0:0.9"])["overlap"], [players[1]])


class Bootstrap(unittest.TestCase):
    @staticmethod
    def accuracy(per_player_means, windows):
        return {"per_player": {f"p{i}": {"windows": n, "sum_deg_by_horizon": [m * n]}
                               for i, (m, n) in enumerate(zip(per_player_means, windows))}}

    def test_constant_difference_has_a_point_interval(self):
        w = [100, 300, 50]
        p = stats.paired(self.accuracy([4.0, 5.0, 6.0], w), self.accuracy([4.5, 5.5, 6.5], w))
        self.assertAlmostEqual(p["mean_diff_deg"], -0.5)
        np.testing.assert_allclose(p["ci95"], [-0.5, -0.5])
        self.assertEqual(p["players_better"], 3)

    def test_window_weighting_and_interval_covers_mean(self):
        w = [1000, 10, 10]
        p = stats.paired(self.accuracy([5.0, 5.0, 5.0], w), self.accuracy([5.1, 4.0, 4.0], w))
        self.assertAlmostEqual(p["mean_diff_deg"], (-0.1 * 1000 + 1 * 20) / 1020)
        self.assertAlmostEqual(p["equal_player_mean_diff_deg"], (-0.1 + 2) / 3)
        self.assertLessEqual(p["ci95"][0], p["mean_diff_deg"])
        self.assertGreaterEqual(p["ci95"][1], p["mean_diff_deg"])

    def test_different_windows_are_refused(self):
        with self.assertRaises(ValueError):
            stats.paired(self.accuracy([1.0], [10]), self.accuracy([1.0], [11]))


class Files(unittest.TestCase):
    def test_replace_retries_through_transient_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            real, calls = results.os.replace, []
            def flaky(a, b):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError(5, "Access is denied")
                real(a, b)
            with mock.patch.object(results.os, "replace", flaky), mock.patch.object(results.time, "sleep"):
                results.write_json(Path(tmp) / "x.json", {"a": "°"})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads((Path(tmp) / "x.json").read_text(encoding="utf-8")), {"a": "°"})

    def test_latency_session_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = {"hold": helpers.hold_last_graph(Path(tmp) / "h.onnx"),
                      "counter": helpers.counter_graph(Path(tmp) / "c.onnx")}
            s = session(models, "test-box", threads=(1, 2), warmup=5, runs=20, rounds=2, progress=None)
        self.assertEqual(set(s["models"]), {"hold", "counter"})
        self.assertTrue(s["models"]["counter"]["stateful"])
        self.assertEqual(s["models"]["hold"]["threads"]["2"]["calls"], 20)
        self.assertTrue(s["machine"]["cpu"] and s["machine"]["onnxruntime"])


class EndToEnd(unittest.TestCase):
    """define -> register -> score (refuse / allow) -> compare -> latency -> leaderboard, in a temp 'repo'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.h5 = helpers.make_h5(root / "d.h5", {"u1": [("r", 900, True)], "u2": [("r", 700, True)],
                                                  "u3": [("r", 800, True)]})
        self.patches = [mock.patch.object(results, k, v) for k, v in
                        {"REPO": root, "RESULTS": root / "results", "RUNS": root / "runs",
                         "REGISTRY": root / "models.json", "LEADERBOARD": root / "LEADERBOARD.md"}.items()]
        for p in self.patches:
            p.start()
        players = root / "players.txt"
        players.write_text("u1\nu2\n", encoding="utf-8")
        self.definition = data.define("synth-v0", self.h5, ["u1", "u2"], "synthetic", out_dir=root / "defs")
        self.root, self.hold = root, helpers.hold_last_graph(root / "hold.onnx")
        self.counter = helpers.counter_graph(root / "counter.onnx")
        (root / "trained.txt").write_text("u2\nu3\n", encoding="utf-8")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([str(a) for a in args])
        return code, out.getvalue()

    def test_workflow(self):
        bench = ["--bench", self.definition, "--h5", self.h5, "--machine", "box"]
        self.run_cli("register", "hold", self.hold, "--trained-on", "none", "--declared-params", "0")
        with self.assertRaises(SystemExit) as refused:
            self.run_cli("score", self.counter, "--name", "counter", "--trained-on", self.root / "trained.txt", *bench)
        self.assertIn("Refusing", str(refused.exception))
        self.run_cli("score", self.counter, "--name", "counter", "--trained-on", self.root / "trained.txt",
                     "--allow-contaminated", *bench)
        self.run_cli("score", self.hold, "--name", "hold", *bench)
        self.run_cli("score", "baseline:damped_velocity", *bench)
        with self.assertRaises(SystemExit):  # registered name, different file
            self.run_cli("score", self.counter, "--name", "hold", *bench)

        hold = json.loads((self.root / "results/synth-v0/hold@box.json").read_text(encoding="utf-8"))
        counter = json.loads((self.root / "results/synth-v0/counter@box.json").read_text(encoding="utf-8"))
        self.assertEqual(hold["contamination"]["status"], "clean")
        self.assertEqual((counter["contamination"]["status"], counter["contamination"]["overlap"]), ("contaminated", ["u2"]))
        self.assertIsNone(hold["accuracy_cold_start"])
        self.assertEqual(counter["accuracy"]["protocol"], "from-start")
        self.assertEqual(counter["accuracy_cold_start"]["protocol"], "cold-start")
        self.assertAlmostEqual(hold["accuracy"]["mean_deg_step20"], counter["accuracy"]["mean_deg_step20"], places=6)
        self.assertEqual(hold["bench"]["windows"], 481 + 281)
        self.assertTrue((self.root / hold["accuracy"]["per_window_file"]["path"]).exists())

        code, out = self.run_cli("compare", "damped_velocity", "hold", "--bench", self.definition)
        self.assertIn("95% CI", out)
        self.run_cli("latency", f"hold={self.hold}", f"counter={self.counter}", "--machine", "box",
                     "--warmup", "2", "--runs", "6", "--rounds", "2")
        self.run_cli("leaderboard")
        board = (self.root / "LEADERBOARD.md").read_text(encoding="utf-8")
        self.assertIn("## Benchmark `synth-v0`", board)
        self.assertIn("| ⚠ | counter (stateful)", board)
        self.assertIn("counter is contaminated (1 benchmark players", board)
        self.assertEqual(board.count("### box ·"), 1)
        self.assertLess(board.index("| 1 | damped_velocity"), board.index("| 2 | hold"))
        self.assertNotIn("(no CI)", board)

    def strip_per_player(self):
        """What the public export does to every result file: drop the per-player sums."""
        for f in (self.root / "results").rglob("*@*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            for key in ("accuracy", "accuracy_cold_start"):
                if r.get(key):
                    r[key].pop("per_player", None)
            f.write_text(json.dumps(r), encoding="utf-8")

    def test_published_results_without_per_player(self):
        bench = ["--bench", self.definition, "--h5", self.h5, "--machine", "box"]
        self.run_cli("register", "hold", self.hold, "--trained-on", "none", "--declared-params", "0")
        self.run_cli("score", self.hold, "--name", "hold", *bench)
        self.run_cli("score", "baseline:damped_velocity", *bench)
        self.strip_per_player()
        hold = results.find_result("synth-v0", "hold")["accuracy"]["mean_deg_step20"]
        damped = results.find_result("synth-v0", "damped_velocity")["accuracy"]["mean_deg_step20"]

        code, out = self.run_cli("compare", "damped_velocity", "hold", "--bench", self.definition)
        self.assertEqual(code, 0)
        self.assertIn(f"damped_velocity - hold = {damped - hold:+.4f} deg. No 95% CI", out)
        self.run_cli("leaderboard")
        board = (self.root / "LEADERBOARD.md").read_text(encoding="utf-8")
        self.assertIn(f"| 2 | hold | {hold:.4f} | {hold - damped:+.4f} (no CI) |", board)
        self.assertIn("(no CI) = the difference of the two means only.", board)


if __name__ == "__main__":
    unittest.main()
