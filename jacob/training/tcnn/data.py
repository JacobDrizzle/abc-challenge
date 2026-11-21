"""Validate frozen data and draw windows without materialising overlapping arrays."""
from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from baseline_pipeline import SPLITS, EvaluationConfig, inventory, load_split, unit_quaternions, sha256_file


def check_corpus(data_path, split_path, *, verify_hash=True, minimum_users=30,
                 minimum_recordings=100, synthetic=False):
    """Read every retained sample once. Refuse unqualified or mislabelled input.

    split_path is the ray/ export's manifest_<run_id>.json; its user lists must equal the HDF5's /<split> groups."""
    data_path, split_path = Path(data_path), Path(split_path)
    config = EvaluationConfig()
    groups, rows = {}, []
    with h5py.File(data_path, "r") as h5:
        if bool(h5.attrs.get("synthetic", False)) and not synthetic:
            raise ValueError("Synthetic fixtures cannot be used for a real training run")
        for name in SPLITS:
            _, users = load_split(split_path, name)
            if set(users) != (set(h5[name]) if name in h5 else set()):
                raise ValueError(f"{name}: the manifest's users differ from the HDF5's /{name} group")
            segments, skipped = inventory(h5, users, config, False)
            if skipped:
                raise ValueError(f"{name}: skipped segments are not allowed in a frozen training corpus")
            if set(users) != {s["user"] for s in segments}:
                raise ValueError(f"{name}: a selected user has no eligible windows")
            groups[name] = users
            for row in segments:
                g = h5[row["path"]]
                previous_t, previous_q = None, None
                for start in range(0, len(g["time_s"]), 65536):
                    t = g["time_s"][start:start+65536]
                    q = unit_quaternions(g["quaternion_wxyz"][start:start+65536])
                    if not np.isfinite(t).all() or not np.allclose(np.diff(t), .01, atol=1e-6, rtol=0):
                        raise ValueError(f"{row['key']}: invalid time grid")
                    if previous_t is not None and not np.isclose(t[0]-previous_t, .01, atol=1e-6, rtol=0):
                        raise ValueError("Gap between validation blocks")
                    if np.any(np.sum(q[1:]*q[:-1], axis=1) < -1e-8) or (
                            previous_q is not None and np.dot(q[0], previous_q) < -1e-8):
                        raise ValueError(f"{row['key']}: discontinuous quaternion signs")
                    previous_t, previous_q = t[-1], q[-1]
                rows.append(dict(row, split=name))
    users = sum(len(g) for g in groups.values())
    recordings = len({(r["user"], r["recording"]) for r in rows})
    if users < minimum_users or recordings < minimum_recordings:
        raise ValueError(f"Cohort below training minimum: {users} users / {recordings} recordings; need {minimum_users}/{minimum_recordings}")
    data_sha, split_sha = sha256_file(data_path), sha256_file(split_path)
    if verify_hash:
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if manifest.get("data_sha256") != data_sha:
            raise ValueError("HDF5 checksum differs from manifest")
    return {"data_sha256": data_sha, "split_sha256": split_sha, "users": groups,
            "recordings": recordings, "segments": rows,
            "windows": {name: sum(r["windows"] for r in rows if r["split"] == name)
                        for name in groups}}


class WindowDataset(Dataset):
    """User-balanced random anchors, reproducible independent of worker scheduling."""
    def __init__(self, path, segments, samples, seed, context=400, horizon=20, cache_mb=128):
        self.path = str(Path(path).resolve())
        self.segments, self.samples, self.seed = segments, samples, seed
        self.context, self.horizon = context, horizon
        self.epoch = 0
        self.handle = None
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.cache_limit = cache_mb * 1024 * 1024
        self.by_user = {}
        for row in segments:
            self.by_user.setdefault(row["user"], []).append(row)
        self.users = sorted(self.by_user)
        if not self.users:
            raise ValueError("Empty training set")
        self.ends = {u: np.cumsum([r["samples"]-context-horizon+1 for r in self.by_user[u]])
                     for u in self.users}

    def __len__(self):
        return self.samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(handle=None, cache=OrderedDict(), cache_bytes=0)
        return state

    def _read(self, row, start):
        key = row["key"]
        if key in self.cache:
            data = self.cache.pop(key)
            self.cache[key] = data
            return data[start:start+self.context+self.horizon].copy()
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        size = row["samples"] * 4 * 4
        dataset = self.handle[row["path"]]["quaternion_wxyz"]
        if size > self.cache_limit:
            return np.asarray(dataset[start:start+self.context+self.horizon], dtype=np.float32)
        while self.cache and self.cache_bytes + size > self.cache_limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.nbytes
        data = np.asarray(dataset[:], dtype=np.float32)
        self.cache[key] = data
        self.cache_bytes += data.nbytes
        return data[start:start+self.context+self.horizon].copy()

    def __getitem__(self, index):
        # Separate streams; additive seeds made consecutive runs share shifted samples.
        key = f"window-v2:{self.seed}:{self.epoch}:{int(index)}".encode("ascii")
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "little"))
        user = self.users[rng.randrange(len(self.users))]
        ends = self.ends[user]
        offset = rng.randrange(int(ends[-1]))
        segment = int(np.searchsorted(ends, offset, side="right"))
        start = offset - (int(ends[segment-1]) if segment else 0)
        q = self._read(self.by_user[user][segment], start)
        return torch.from_numpy(q[:self.context]), torch.from_numpy(q[self.context:])

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.cache.clear()
        self.cache_bytes = 0


class DeviceWindows:
    """Hold selected segments on one device; draw user-balanced batches and strided windows there.

    Avoids per-window HDF5 reads and DataLoader workers; the cohort is tens of megabytes.
    """
    def __init__(self, path, segments, device, context=400, horizon=20):
        self.context, self.horizon, self.device = context, horizon, torch.device(device)
        span = context + horizon
        arrays, self.bounds, by_user, offset = [], [], {}, 0
        with h5py.File(path, "r") as h5:
            for row in segments:
                q = unit_quaternions(h5[row["path"]]["quaternion_wxyz"][:], source=row["key"]).astype(np.float32)
                if len(q) < span:
                    raise ValueError(f"{row['key']}: shorter than one window")
                arrays.append(q)
                self.bounds.append((offset, len(q)))
                by_user.setdefault(row["user"], []).append(np.arange(offset, offset + len(q) - span + 1))
                offset += len(q)
        if not arrays:
            raise ValueError("No segments to load")
        self.buffer = torch.from_numpy(np.concatenate(arrays)).to(self.device)
        self.users = sorted(by_user)
        starts = [np.concatenate(by_user[u]) for u in self.users]
        self.user_count = torch.tensor([len(s) for s in starts], dtype=torch.int64)
        self.user_offset = torch.cumsum(self.user_count, 0) - self.user_count
        self.starts = torch.from_numpy(np.concatenate(starts)).to(self.device)
        self.span = torch.arange(span, device=self.device)

    def _gather(self, starts):
        window = self.buffer[starts[:, None] + self.span]
        return window[:, :self.context], window[:, self.context:]

    def batches(self, seed, epoch, steps, batch_size):
        """Same distribution as WindowDataset: a uniform user, then a uniform window of that user."""
        key = hashlib.sha256(f"device-window-v1:{seed}:{epoch}".encode("ascii")).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little") & (2**63 - 1))
        for _ in range(steps):
            user = torch.randint(len(self.users), (batch_size,), generator=generator)
            count = self.user_count[user]
            within = (torch.rand(batch_size, generator=generator, dtype=torch.float64) * count).long()
            index = self.user_offset[user] + torch.minimum(within, count - 1)
            yield self._gather(self.starts[index.to(self.device)])

    def windows(self, stride, batch_size):
        """Anchors match baseline_pipeline.window_batches: first anchor C-1, then every stride samples."""
        starts = []
        for offset, length in self.bounds:
            count = (length - self.context - self.horizon) // stride + 1
            starts.append(offset + np.arange(count, dtype=np.int64) * stride)
        starts = torch.from_numpy(np.concatenate(starts)).to(self.device)
        for begin in range(0, len(starts), batch_size):
            yield self._gather(starts[begin:begin + batch_size])
