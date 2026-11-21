"""Hugging Face access: the list of tars, and reading the replays inside a downloaded tar."""

import gzip
import json
import os
import tarfile

from huggingface_hub import list_repo_files

from .config import CACHED_DIR, REPO, TAR_LIST_PATH


def get_tar_names() -> list[str]:
    """Tar names in the repo, cached so the repo is only listed once."""
    if os.path.isfile(TAR_LIST_PATH):
        with open(TAR_LIST_PATH, encoding="utf-8") as f:
            return json.load(f)

    print("Tar list not cached, asking Hugging Face")
    files = list_repo_files(REPO, repo_type="dataset")
    tars = sorted(f for f in files if f.endswith(".tar"))  # sorted = stable indexes

    os.makedirs(CACHED_DIR, exist_ok=True)
    with open(TAR_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(tars, f, indent=2)
    return tars


def iter_tar_replays(tar_path: str):
    """Yield (name, raw bytes) for every replay in a tar, one at a time."""
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            name = os.path.basename(m.name)
            if not name.endswith(".xror"):
                name += ".xror"
            yield name, raw
