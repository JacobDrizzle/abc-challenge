"""Benchmark definitions, HDF5 inventory, window enumeration and cohort fingerprints.

HDF5 layout (group_a_data_pipeline): /<user>/<recording>/<segment>/{time_s, quaternion_wxyz}, segment
attribute challenge_filter_pass, root attributes quaternion_order="wxyz", rate_hz=100.
The ray/ export (group-a-v2) nests the same groups one level down, /<split>/<user>/..., with the split
groups listed in the root attribute splits. Both read the same: segment keys are always
<user>/<recording>/<segment>, so the fingerprints don't depend on the layout either.

Windows: every stride-1 window of every filter-qualified segment of the benchmark players. For a
segment of N samples the anchors (last context sample) are C-1 ... N-H-1; history is q[a-C+1 : a+1],
targets are q[a+1 : a+H+1]. Inventory, checks and the legacy fingerprint are ported from
ABC_local_test/baseline_pipeline.py (inventory, window_batches, evaluate).
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from bench.geometry import unit_quaternions

PACKAGE = Path(__file__).resolve().parent
DEFINITIONS = PACKAGE / "definitions"
FINGERPRINT_SCHEMA = "bench-cohort-v1"
LEGACY_PROTOCOL = "head-orientation-geodesic-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_definition(name_or_path):
    """A benchmark is a JSON file in bench/definitions/ (or any path): players, protocol, data checksum."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DEFINITIONS / f"{name_or_path}.json"
    definition = json.loads(path.read_text(encoding="utf-8"))
    definition["definition_sha256"] = sha256_file(path)
    players = definition["players"]
    if len(set(players)) != len(players) or players != sorted(players):
        raise ValueError(f"{path}: players must be sorted and unique")
    return definition


def protocol(definition):
    p = definition["protocol"]
    return int(p["context"]), int(p["horizon"]), float(p["rate_hz"])


def player_paths(h5):
    """{player: path of the player's group}. Players sit at the root, unless the root holds exactly the
    split groups named by the attribute splits (the ray/ export); then each player is inside one of them."""
    splits = [str(s) for s in h5.attrs.get("splits", [])]
    if not splits or set(h5.keys()) != set(splits):
        return {player: player for player in h5.keys()}
    paths = {}
    for split in splits:
        for player in h5[split]:
            if player in paths:
                raise ValueError(f"{player} is in two splits: {paths[player].split('/')[0]} and {split}")
            paths[player] = f"{split}/{player}"
    return paths


def inventory(h5, players, context=400, horizon=20, rate_hz=100.0):
    """Filter-qualified segments long enough for one window, in sorted user/recording/segment order.
    Each segment has a key (<user>/<recording>/<segment>, layout-independent) and the path to read it from."""
    if h5.attrs.get("quaternion_order") != "wxyz":
        raise ValueError("HDF5 must declare quaternion_order='wxyz'")
    if not np.isclose(float(h5.attrs.get("rate_hz", np.nan)), rate_hz, rtol=0, atol=1e-9):
        raise ValueError("HDF5 rate_hz does not match the benchmark clock")
    paths = player_paths(h5)
    missing = sorted(set(players) - set(paths))
    if missing:
        raise ValueError(f"{len(missing)} benchmark player(s) absent from the HDF5, e.g. {missing[0]}")
    segments, skipped = [], []
    for user in sorted(players):
        player = h5[paths[user]]
        for recording in sorted(player):
            for segment in sorted(player[recording]):
                key = f"{user}/{recording}/{segment}"
                path = f"{paths[user]}/{recording}/{segment}"
                group = h5[path]
                if not isinstance(group, h5py.Group) or not all(k in group for k in ("time_s", "quaternion_wxyz")):
                    raise ValueError(f"{key}: expected time_s and quaternion_wxyz datasets")
                q, t = group["quaternion_wxyz"], group["time_s"]
                if q.ndim != 2 or q.shape[1] != 4 or t.shape != (len(q),):
                    raise ValueError(f"{key}: invalid dataset shapes")
                flag = group.attrs.get("challenge_filter_pass")
                if flag is not None and not (isinstance(flag, (bool, np.bool_, int, np.integer)) and flag in (0, 1)):
                    raise ValueError(f"{key}: challenge_filter_pass must be a boolean or 0/1")
                if flag is None or not bool(flag):
                    skipped.append({"key": key, "reason": "motion_filter_failed" if flag is not None else "qualification_missing"})
                elif len(q) < context + horizon:
                    skipped.append({"key": key, "reason": "too_short"})
                else:
                    segments.append({"key": key, "path": path, "user": user, "recording": recording,
                                     "samples": len(q), "windows": len(q) - context - horizon + 1})
    if not segments:
        raise ValueError(f"No eligible windows for {len(players)} player(s)")
    return segments, skipped


def read_segment(h5, key, rate_hz=100.0):
    """Whole segment as unit float64 quaternions, after the same checks the old evaluator made per block."""
    group = h5[key]
    times = np.asarray(group["time_s"][:], dtype=np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / rate_hz, rtol=0, atol=1e-6):
        raise ValueError(f"{key}: timestamps are not a finite uniform {rate_hz:g} Hz grid; split gaps upstream")
    q = unit_quaternions(group["quaternion_wxyz"][:], source=key)
    if np.any(np.sum(q[:-1] * q[1:], axis=-1) < -1e-8):
        raise ValueError(f"{key}: quaternion signs are discontinuous; align them in the data preparation step")
    return q


def content_fingerprint(h5, segments, players, context, horizon, rate_hz):
    """SHA-256 of the protocol, the players and the exact samples of every scored segment.

    Unlike the legacy fingerprint it does not depend on the file's checksum, so the full HDF5 and a
    benchmark-only extract of it give the same value: equal fingerprints mean the same windows and values.
    """
    digest = hashlib.sha256()
    header = {"schema": FINGERPRINT_SCHEMA, "context": context, "horizon": horizon, "rate_hz": rate_hz,
              "stride": 1, "players": sorted(players), "segments": [[s["key"], s["samples"]] for s in segments]}
    digest.update(json.dumps(header, sort_keys=True).encode())
    for s in segments:
        group = h5[s["path"]]
        digest.update(s["key"].encode() + b"\0")
        digest.update(np.ascontiguousarray(group["quaternion_wxyz"][:], dtype="<f4").tobytes())
        digest.update(np.ascontiguousarray(group["time_s"][:], dtype="<f8").tobytes())
    return digest.hexdigest()


def legacy_fingerprint(data_sha256, players, segments, context, horizon, rate_hz):
    """baseline_pipeline.py's cohort_fingerprint, reproduced exactly (it includes the file's SHA-256)."""
    cohort = {"data_sha256": data_sha256, "users": sorted(players), "segments": [s["key"] for s in segments],
              "context": context, "horizon": horizon, "rate_hz": float(rate_hz), "stride": 1,
              "protocol": LEGACY_PROTOCOL}
    return hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()


class Benchmark:
    """An opened benchmark: definition + HDF5, checked against each other before anything is scored."""

    def __init__(self, definition, h5_path, verify=True):
        self.definition = definition if isinstance(definition, dict) else load_definition(definition)
        self.name = self.definition["name"]
        self.players = self.definition["players"]
        self.context, self.horizon, self.rate_hz = protocol(self.definition)
        self.h5_path = Path(h5_path).resolve()
        self.h5 = h5py.File(self.h5_path, "r")
        self.segments, self.skipped = inventory(self.h5, self.players, self.context, self.horizon, self.rate_hz)
        self.windows = sum(s["windows"] for s in self.segments)
        self.data_sha256 = sha256_file(self.h5_path)
        self.fingerprint = content_fingerprint(self.h5, self.segments, self.players, self.context, self.horizon,
                                               self.rate_hz)
        self.legacy_fingerprint = legacy_fingerprint(self.data_sha256, self.players, self.segments, self.context,
                                                     self.horizon, self.rate_hz)
        expected = self.definition.get("cohort_fingerprint")
        if verify and expected and expected != self.fingerprint:
            raise ValueError(f"{self.h5_path.name} does not hold benchmark '{self.name}': cohort fingerprint "
                             f"{self.fingerprint[:12]} != {expected[:12]}. Use the shared benchmark HDF5.")

    def segment(self, s):
        return read_segment(self.h5, s["path"], self.rate_hz)

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def define(name, h5_path, players, description, out_dir=DEFINITIONS, legacy=None):
    """Write definitions/<name>.json for these players on this HDF5 (the definition is then frozen)."""
    draft = {"name": name, "players": sorted(set(players)),
             "protocol": {"context": 400, "horizon": 20, "rate_hz": 100.0, "stride": 1}}
    with Benchmark(draft, h5_path, verify=False) as b:
        definition = {
            "name": name, "description": description, "schema": FINGERPRINT_SCHEMA,
            "protocol": {**draft["protocol"], "windows": "every stride-1 window of every challenge_filter_pass "
                         "segment with at least context + horizon samples",
                         "metric": "mean sign-invariant geodesic angle (degrees, float64) at step 20, over all windows",
                         "output": "re-normalised before scoring, as the harness does"},
            "data": {"file": b.h5_path.name, "sha256": b.data_sha256, "bytes": b.h5_path.stat().st_size},
            "cohort_fingerprint": b.fingerprint, "windows": b.windows,
            "players": b.players,
            "players_without_windows": sorted(set(b.players) - {s["user"] for s in b.segments}),
            "segments": [[s["key"], s["windows"]] for s in b.segments]}
        if legacy:
            definition["legacy"] = {**legacy, "cohort_fingerprint": b.legacy_fingerprint}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.json"
    if path.exists():
        raise ValueError(f"{path} exists; benchmark definitions are frozen. Choose a new name/version.")
    path.write_text(json.dumps(definition, indent=1) + "\n", encoding="utf-8")
    return path


def extract(source, players, out_path):
    """Copy the given players' groups verbatim into a new, benchmark-only HDF5 (for sharing in the team).
    The players go at the root, whichever layout the source has."""
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} exists")
    with h5py.File(source, "r") as src, h5py.File(out_path, "w") as dst:
        paths = player_paths(src)
        missing = sorted(set(players) - set(paths))
        if missing:
            raise ValueError(f"{len(missing)} player(s) not in {source}, e.g. {missing[0]}")
        for key, value in src.attrs.items():
            if key != "splits":  # the extract has no split groups
                dst.attrs[key] = value
        dst.attrs["bench_extracted_from_sha256"] = sha256_file(source)
        for user in sorted(players):
            src.copy(src[paths[user]], dst, name=user)
    return out_path
