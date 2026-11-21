"""Where files live on disk: users/<shard>/<tar>/ (old extraction) and saved/ (passing recordings)."""

import os
import shutil

from .config import SAVED_ROOT


def tar_folder(tar_name: str) -> str:
    """users/<shard>/<tarname>.tar -> users/<shard>/<tarname>"""
    return os.path.splitext(tar_name)[0]


def save_passed(src_path: str) -> str:
    """users/<shard>/<tarname>/<file>.xror -> saved/users/<shard>/<tarname>/<file>.xror"""
    dest = os.path.join(SAVED_ROOT, src_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src_path, dest)
    return dest


def list_saved() -> list[str]:
    """All .xror files under saved/, sorted."""
    found = []
    for root, _dirs, files in os.walk(SAVED_ROOT):
        for name in files:
            if name.endswith(".xror"):
                found.append(os.path.join(root, name))
    return sorted(found)


def saved_path_to_key(path: str, by_base: dict | None = None) -> tuple[str, str]:
    """saved/users/<shard>/<tarname>/<file>.xror -> ('users/<shard>/<tarname>.tar', '<file>.xror').
    This is exactly the (tar, file) key process_tar logs, so re-checks REPLACE the
    existing row instead of adding a second one."""
    parts = os.path.relpath(path, SAVED_ROOT).replace(os.sep, "/").split("/")
    tar_part, fname = "/".join(parts[:-1]), parts[-1]
    if "/" not in tar_part and by_base:  # old flat layout saved/<tarname>/<file>
        return by_base.get(tar_part + ".tar", tar_part + ".tar"), fname
    return tar_part + ".tar", fname


def scan_extracted_tars(root: str = "users") -> dict[str, list[str]]:
    """{tar name: sorted .xror files} for every tar still extracted under users/<shard>/<tarname>/.

    This REPLACES the deprecated cache/manifest.json: the folders on disk are the single
    source of truth, so nothing can get out of sync with them."""
    found: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        xrors = sorted(f for f in files if f.endswith(".xror"))
        if xrors:
            found[os.path.relpath(dirpath, ".").replace(os.sep, "/") + ".tar"] = xrors
    return found
