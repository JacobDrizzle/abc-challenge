"""Durable JSON writes. Moved unchanged from ABC_local_test/prepare_tcnn_data.py (ray/ replaces the rest of it)."""
import json
import os
from pathlib import Path


def durable_replace(source, target):
    """Flush file contents before publishing its name or recording it as complete."""
    source, target = Path(source), Path(target)
    with source.open("rb+") as handle:
        os.fsync(handle.fileno())
    source.replace(target)
    if os.name == "posix":
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    durable_replace(temp, path)
