"""cache/processed.json: which tars are finished. Thread safe (lock + atomic write)."""

import json
import os
import threading

from .config import CACHED_DIR, PROCESSED_PATH


PROCESSED_LOCK = threading.Lock()   # guards read-modify-write of processed.json


def load_processed() -> dict:
    if os.path.isfile(PROCESSED_PATH):
        with open(PROCESSED_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed(processed: dict) -> None:
    os.makedirs(CACHED_DIR, exist_ok=True)
    tmp = PROCESSED_PATH + f".{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    os.replace(tmp, PROCESSED_PATH)  # atomic: readers never see a half-written file


def record_processed(tar_name: str, record: dict) -> None:
    """Add one tar to processed.json without losing entries written by other threads."""
    with PROCESSED_LOCK:
        processed = load_processed()
        processed[tar_name] = record
        save_processed(processed)
