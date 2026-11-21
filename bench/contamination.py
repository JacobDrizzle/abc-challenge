"""Which players a model was trained or validated on, and whether any of them are benchmark players.

A model is declared with one or more specs (all are combined):
  none                          trained on no BOXRR-23 player (baselines, or other data only)
  rule:sha1:<salt>:<threshold>  every player with sha1("<salt>:" + id) / 16**40 < threshold, i.e. Nic's
                                rule; rule:sha1:0:0.9 = his train + val buckets. Conservative: it also
                                covers players he never downloaded.
  csv:<path>[#<split>,...]      player_id,split rows (nic/splits/splits_fri.csv); default splits train,val
  split-json:<path>             train_users + validation_users of a frozen_user_split.json
  <path>                        a text file, one player id per line (# comments allowed)
Player id = the BOXRR-23 tar name without .tar (the user id), as in both pipelines.
"""
import csv
import hashlib
import json
from pathlib import Path

from bench.data import PACKAGE, sha256_file


def _path(text):
    """Paths in bench/models.json are relative to the repo root; on the command line, to the current folder."""
    path = Path(text)
    return path if path.is_absolute() or path.exists() else PACKAGE.parent / path


def sha1_fraction(player, salt="0"):
    """Nic's split hash (nic/splits/README.md): sha1("0:" + id) read as a number in [0, 1)."""
    return int(hashlib.sha1(f"{salt}:{player}".encode()).hexdigest(), 16) / 16 ** 40


def nic_split(player):
    f = sha1_fraction(player)
    return "train" if f < 0.8 else "val" if f < 0.9 else "test"


def parse(spec):
    """-> dict(spec, kind, sha256, players: set or None, rule: (salt, threshold) or None)."""
    if spec == "none":
        return {"spec": spec, "kind": "none", "sha256": None, "players": set(), "rule": None}
    if spec.startswith("rule:"):
        parts = spec.split(":")
        if len(parts) != 4 or parts[1] != "sha1":
            raise ValueError(f"Rule spec must be rule:sha1:<salt>:<threshold>, got {spec!r}")
        return {"spec": spec, "kind": "rule", "sha256": None, "players": None, "rule": (parts[2], float(parts[3]))}
    kind, _, rest = spec.partition(":") if spec.startswith(("csv:", "split-json:")) else ("list", "", spec)
    if kind == "csv":
        path, _, splits = rest.partition("#")
        path = _path(path)
        wanted = set(splits.split(",")) if splits else {"train", "val", "validation"}
        with open(path, newline="", encoding="utf-8") as handle:
            players = {r["player_id"].strip() for r in csv.DictReader(handle) if r["split"].strip() in wanted}
    elif kind == "split-json":
        path = _path(rest)
        split = json.loads(Path(path).read_text(encoding="utf-8"))
        players = set(split["train_users"]) | set(split["validation_users"])
    else:
        path = _path(rest)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        players = {x.strip() for x in lines if x.strip() and not x.lstrip().startswith("#")}
    if not players:
        raise ValueError(f"{spec}: no players found")
    return {"spec": spec, "kind": kind, "sha256": sha256_file(path), "players": players, "rule": None}


def check(benchmark_players, specs):
    """Overlap between the benchmark and everything the model was fitted on. status: clean/contaminated/undeclared."""
    if not specs:
        return {"status": "undeclared", "trained_on": [], "overlap": [],
                "note": "No --trained-on given: contamination cannot be checked"}
    parsed = [parse(s) for s in specs]
    overlap = set()
    for p in parsed:
        if p["rule"]:
            salt, threshold = p["rule"]
            overlap |= {u for u in benchmark_players if sha1_fraction(u, salt) < threshold}
        else:
            overlap |= set(benchmark_players) & p["players"]
    return {"status": "contaminated" if overlap else "clean",
            "trained_on": [{"spec": p["spec"], "kind": p["kind"], "sha256": p["sha256"],
                            "players": None if p["players"] is None else len(p["players"])} for p in parsed],
            "overlap": sorted(overlap)}
