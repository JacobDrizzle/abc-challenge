"""v10_users: the two config keys, the user subset and window fraction, the hooks, and a tiny training run.

python -m unittest experimental.v10_users.test_v10 -v   (from jacob/training)
"""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

import run_desktop
import run_next
from smoke_tcnn import make_fixture
from tcnn.data import check_corpus
import train_tcnn

from best import hooks as best_hooks
from best.training import TrainingWindows

from experimental.v10_users import hooks
from experimental.v10_users.configuration import V10_DEFAULTS, normalize_config
from experimental.v10_users.training import SubsetWindows, subset_order

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = json.loads((ROOT/"best/configs/base.json").read_text())


def tiny(config, **overrides):
    """A config small enough to train on the synthetic fixture in seconds."""
    return {**config, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20, **overrides}


class ConfigTests(unittest.TestCase):
    def test_defaults_add_the_two_keys_to_best(self):
        self.assertEqual(normalize_config(BASE), {**BASE, **V10_DEFAULTS})
        self.assertEqual(normalize_config({**BASE, "train_users": 25})["train_users"], 25)
        self.assertEqual(normalize_config({**BASE, "user_window_fraction": .5})["user_window_fraction"], .5)

    def test_keys_are_checked(self):
        for bad in ({"train_users": 0}, {"train_users": -1}, {"train_users": 2.5}, {"train_users": True},
                    {"train_users": "25"}, {"user_window_fraction": 0}, {"user_window_fraction": 1.5},
                    {"user_window_fraction": True}, {"train_users": 25, "sampling": "user"},
                    {"user_window_fraction": .5, "sampling": "user"}, {"colour": True}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                normalize_config({**BASE, **bad})

    def test_screen_variants_change_only_their_key(self):
        with hooks.installed():
            configs = run_next.experiment_configs(json.loads((HERE/"configs/screen.json").read_text()), BASE)
        control = {**BASE, **V10_DEFAULTS}
        self.assertEqual(configs, {"control": control, "users_25": {**control, "train_users": 25},
                                   "users_50": {**control, "train_users": 50},
                                   "users_100_half": {**control, "user_window_fraction": .5}})

    def test_hooks_hash_the_new_files_and_restore_every_name(self):
        modules = (best_hooks, run_desktop, run_next, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.FILES):], hooks.FILES)
            self.assertIs(train_tcnn.normalize_config, normalize_config)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.FILES:
            self.assertTrue((ROOT/name).exists(), name)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def users(self, counts=(500, 2000, 8000, 1200)):
        """One user per count, each held at a constant orientation that names it (as best's unequal_users)."""
        path, rows = self.directory/"users.h5", []
        with h5py.File(path, "w") as h5:
            for i, n in enumerate(counts):
                key = f"u{i}/recording/segment_000"
                identity = np.eye(4, dtype=np.float32)[i]
                h5.create_dataset(key+"/quaternion_wxyz", data=np.tile(identity, (n, 1)))
                rows.append({"user": f"u{i}", "key": key, "path": key, "samples": n})
        return path, rows

    def drawn_users(self, windows, steps=20, batch_size=500):
        x = torch.cat([b[0] for b in windows.batches(42, 0, steps, batch_size)])
        return {f"u{i}" for i in range(4) if bool((x[:, 0, i] == 1).any())}

    def test_unrestricted_and_everyone_draw_best_batches(self):
        path, rows = self.users()
        for sampling in ("sqrt", "window"):
            for keys in ({}, {"train_users": 4}, {"train_users": 4, "user_window_fraction": 1.}):
                with self.subTest(sampling, **keys):
                    ours = SubsetWindows(path, rows, "cpu", sampling=sampling, **keys).batches(42, 3, 4, 64)
                    for (x, y), (a, b) in zip(ours, TrainingWindows(path, rows, "cpu", sampling=sampling)
                                              .batches(42, 3, 4, 64)):
                        self.assertTrue(torch.equal(x, a) and torch.equal(y, b))

    def test_subsets_are_nested_fixed_and_the_only_users_drawn(self):
        path, rows = self.users()
        order = subset_order([r["user"] for r in rows])
        self.assertEqual(order, subset_order(sorted(order, reverse=True)))  # the input order doesn't matter
        self.assertEqual(subset_order(order[:3]), [u for u in order if u in order[:3]])  # nor who else is present
        for n in (1, 2, 3):
            with self.subTest(n=n):
                self.assertEqual(self.drawn_users(SubsetWindows(path, rows, "cpu", sampling="window", train_users=n)),
                                 set(order[:n]))

    def test_fraction_keeps_each_users_leading_windows(self):
        path, rows = self.users()
        windows = SubsetWindows(path, rows, "cpu", sampling="window", user_window_fraction=.5)
        full = windows.user_count.clone()
        offsets, starts = windows.user_offset, []
        gather = windows._gather
        windows._gather = lambda s: (starts.append(s), gather(s))[1]
        next(windows.batches(42, 0, 1, 4000))
        np.testing.assert_array_equal(windows.user_count, (full.double() * .5).floor().long())
        index = torch.searchsorted(windows.starts, torch.cat(starts))  # each drawn start's row in starts
        user = torch.searchsorted(offsets, index, right=True) - 1
        self.assertTrue(bool((index - offsets[user] < windows.user_count[user]).all()))
        self.assertGreater(len(torch.unique(user)), 3)  # every user is still drawn
        tiny_user = SubsetWindows(path, rows, "cpu", sampling="window", user_window_fraction=1e-6)
        tiny_user.restrict()
        self.assertTrue(bool((tiny_user.user_count == 1).all()))  # at least one window each

    def test_validation_windows_and_errors(self):
        path, rows = self.users()
        windows = SubsetWindows(path, rows, "cpu", sampling="window", train_users=99, user_window_fraction=.5)
        plain = TrainingWindows(path, rows, "cpu", sampling="window")
        for (x, y), (a, b) in zip(windows.windows(20, 256), plain.windows(20, 256)):
            self.assertTrue(torch.equal(x, a) and torch.equal(y, b))  # no restriction, no error
        with self.assertRaises(ValueError):
            next(windows.batches(42, 0, 1, 4))  # 99 of 4 users
        with self.assertRaises(ValueError):
            SubsetWindows(path, rows, "cpu", sampling="user", train_users=2)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data, self.split = make_fixture(self.directory/"fixture", samples=900)
        check_corpus(self.data, self.split, minimum_users=3, minimum_recordings=3, synthetic=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_tiny_runs_train_and_every_user_matches_best(self):
        names = dict(vars(train_tcnn))
        with best_hooks.installed():
            run_desktop.train(self.data, self.split, self.directory/"best", tiny(BASE), 42, "cpu", synthetic=True)
        for name, keys in {"everyone": {"train_users": 1}, "half": {"user_window_fraction": .5}}.items():
            with self.subTest(name), hooks.installed():
                run_desktop.train(self.data, self.split, self.directory/name, tiny(BASE, **keys), 42, "cpu",
                                  synthetic=True)
        states = {name: torch.load(self.directory/name/"last.pt", weights_only=True)
                  for name in ("best", "everyone", "half")}
        self.assertEqual(states["half"]["epoch"], 2)
        for key, value in states["best"]["model_state"].items():  # the fixture has one training user
            self.assertTrue(torch.equal(states["everyone"]["model_state"][key], value), key)
        self.assertFalse(all(torch.equal(states["half"]["model_state"][k], v)
                             for k, v in states["best"]["model_state"].items()))
        self.assertEqual({k: v for k, v in vars(train_tcnn).items() if k in names}, names)


if __name__ == "__main__":
    unittest.main()
