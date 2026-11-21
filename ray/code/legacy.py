"""DEPRECATED code, kept only so nothing that still references it breaks.

* cache/manifest.json is DEPRECATED. Nothing reads or writes it any more. It used to list the
  files of every tar extracted under users/; storage.scan_extracted_tars() now finds them by
  looking at the folders. You can delete the file.
* The old "extract the whole tar to users/" flow (save_replays and friends) is DEPRECATED.
  pipeline.process_tar replaced it: it checks every recording and only stores the passes.

Every function here warns with DeprecationWarning when called."""

import gzip
import json
import os
import tarfile
import warnings

from huggingface_hub import hf_hub_download
from xror import XROR

from .config import CACHED_DIR, REPO, TAR_INDEX
from .hf import get_tar_names
from .storage import tar_folder


MANIFEST_PATH = os.path.join(CACHED_DIR, "manifest.json")  # DEPRECATED: {tar_name: [files]}


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"{old}() is deprecated: {new}", DeprecationWarning, stacklevel=3)


def load_manifest() -> dict:
    """DEPRECATED. cache/manifest.json is no longer read or written by anything in this project.
    It used to list the files of every tar extracted under users/. That information now comes
    from scanning the folders (storage.scan_extracted_tars), which cannot go out of sync."""
    _deprecated("load_manifest", "manifest.json is ignored, use storage.scan_extracted_tars()")
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as mp:
            return json.load(mp)
    return {}


def save_manifest(manifest: dict) -> None:
    """DEPRECATED. Nothing needs the manifest any more, see load_manifest()."""
    _deprecated("save_manifest", "manifest.json is ignored, use storage.scan_extracted_tars()")
    os.makedirs(CACHED_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def fetch_replays_from_huggingface(tar_name: str) -> list[tuple[str, bytes]]:
    """DEPRECATED: loads every replay of a tar into memory. process_tar streams them instead."""
    _deprecated("fetch_replays_from_huggingface", "use pipeline.process_tar")
    tar_path = hf_hub_download(REPO, tar_name, repo_type="dataset")

    replays = []
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        print(f"{tar_name}: {len(members)} replays in tar")
        for m in members:
            f = tar.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            replays.append((os.path.basename(m.name), raw))
    return replays


def save_replays(tar_name: str, PRINT: bool = True) -> list[str]:
    """DEPRECATED: extract every replay of a tar into users/<shard>/<tarname>/.
    process_tar replaced this: it checks each replay and only stores the ones that pass.
    (It used to record the files in manifest.json, which is deprecated and no longer written.)"""
    _deprecated("save_replays", "use pipeline.process_tar")
    folder = tar_folder(tar_name)

    if os.path.isdir(folder):
        disk_files = sorted(f for f in os.listdir(folder) if f.endswith(".xror"))
        if disk_files:
            if PRINT:
                print(f"{tar_name} already saved ({len(disk_files)} files)")
            return disk_files

    if PRINT:
        print(f"{tar_name} not saved yet, fetching")

    os.makedirs(folder, exist_ok=True)

    saved = []
    for name, raw in fetch_replays_from_huggingface(tar_name):
        if not name.endswith(".xror"):
            name += ".xror"
        with open(os.path.join(folder, name), "wb") as f:
            f.write(raw)
        saved.append(name)
    return saved


def num_xror(tar_name: str) -> int:
    """DEPRECATED. Number of replays in the tar, downloading it first if needed."""
    _deprecated("num_xror", "use pipeline.process_tar")
    return len(save_replays(tar_name, PRINT=True))


def len_xror(tar_index: int = TAR_INDEX) -> int:
    """DEPRECATED."""
    _deprecated("len_xror", "use pipeline.process_tar")
    return num_xror(get_tar_names()[tar_index])


def xror_path(index: int = 0, tar_index: int = TAR_INDEX) -> str:
    """DEPRECATED."""
    _deprecated("xror_path", "use storage.list_saved")
    tar_name = get_tar_names()[tar_index]
    files = save_replays(tar_name, False)
    return os.path.join(tar_folder(tar_name), files[index])


def load_xror(index: int = 0, tar_index: int = TAR_INDEX) -> XROR:
    """DEPRECATED."""
    _deprecated("load_xror", "use storage.list_saved and XROR.unpack")
    with open(xror_path(index, tar_index), "rb") as f:
        return XROR.unpack(f.read())
