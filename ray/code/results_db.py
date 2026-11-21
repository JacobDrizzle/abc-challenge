"""cache/results.sqlite: one row per recording (pass or fail) with game, reason, S, V ...
and evaluate_replay, which checks one recording and builds its row."""

import os
import sqlite3

from xror import XROR

from .config import CACHED_DIR, RESULTS_DB
from .head import game_name, peek_info, unusable_reason
from .processing import EVAL_LOCK, LAST_FAIL, LAST_METRICS, _fail, process_func


RESULT_COLUMNS = [
    ("tar", "TEXT"), ("file", "TEXT"), ("game", "TEXT"), ("game_id", "TEXT"), ("version", "TEXT"),
    ("passed", "INTEGER"), ("reason", "TEXT"), ("detail", "TEXT"), ("time_unit", "TEXT"),
    ("hz", "REAL"), ("duration_s", "REAL"), ("n_frames", "INTEGER"), ("n_windows", "INTEGER"),
    ("pitch_S", "REAL"), ("pitch_V", "REAL"), ("yaw_S", "REAL"), ("yaw_V", "REAL"),
    ("roll_S", "REAL"), ("roll_V", "REAL"), ("max_S", "REAL"), ("max_V", "REAL"),
    ("size_bytes", "INTEGER"),
]


METRIC_KEYS = ["time_unit", "hz", "duration_s", "n_frames", "n_windows", "pitch_S", "pitch_V",
               "yaw_S", "yaw_V", "roll_S", "roll_V", "max_S", "max_V"]


def open_results_db() -> sqlite3.Connection:
    os.makedirs(CACHED_DIR, exist_ok=True)
    con = sqlite3.connect(RESULTS_DB, timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL;")  # Enables safe concurrent access
    cols = ", ".join(f"{c} {t}" for c, t in RESULT_COLUMNS)
    con.execute(f"CREATE TABLE IF NOT EXISTS results ({cols}, PRIMARY KEY (tar, file))")
    return con


def log_results(rows: list[dict]) -> None:
    """Insert (or replace) result rows. Safe to call again for the same recording."""
    if not rows:
        return
    names = [c for c, _ in RESULT_COLUMNS]
    sql = (f"INSERT OR REPLACE INTO results ({', '.join(names)}) "
           f"VALUES ({', '.join('?' for _ in names)})")
    con = open_results_db()
    with con:
        con.executemany(sql, [tuple(r.get(c) for c in names) for r in rows])
    con.close()


def make_result_row(tar_name: str, name: str, data: dict | None, size: int,
                    result, error: str = "") -> dict:
    row = {"tar": tar_name, "file": name, "size_bytes": size,
           "passed": 1 if result is not None else 0}
    if data is None:
        row.update(game="unreadable", reason="unreadable file", detail=error[:200])
        return row

    app = (((data.get("info") or {}).get("software") or {}).get("app")) or {}
    row.update(game=game_name(data), game_id=str(app.get("id") or ""),
               version=str(app.get("version") or ""))
    if result is None:
        row["reason"], row["detail"] = LAST_FAIL
    else:
        row["reason"], row["detail"] = "pass", ""
    for key in METRIC_KEYS:
        if key in LAST_METRICS:
            row[key] = LAST_METRICS[key]
    return row


def make_skip_row(tar_name: str, name: str, info: dict, size: int, category: str, detail: str) -> dict:
    """Log row for a recording that was rejected from its metadata alone (no frames decoded)."""
    app = ((info.get("software") or {}).get("app")) or {}
    return {"tar": tar_name, "file": name, "size_bytes": size, "passed": 0,
            "game": app.get("name") or "unknown", "game_id": str(app.get("id") or ""),
            "version": str(app.get("version") or ""), "reason": category, "detail": detail[:200]}


def evaluate_replay(tar_name: str, name: str, raw: bytes) -> tuple[dict, bool]:
    """Unpack one replay, run process_func, and return (result row, passed).
    process_func reports its numbers through module-level globals, so the call and the
    row building happen under EVAL_LOCK. Without it, threads overwrite each other's
    metrics (that is how rows ended up with a NULL hz or the wrong failure reason)."""
    info = peek_info(raw)  # metadata only: no fpzip decode yet
    if info is not None:
        why = unusable_reason(info)
        if why:
            return make_skip_row(tar_name, name, info, len(raw), *why), False

    try:
        data = XROR.unpack(raw).data  # slow part, runs in parallel
    except Exception as e:
        return make_result_row(tar_name, name, None, len(raw), None, str(e)), False

    with EVAL_LOCK:
        try:
            result = process_func(data, quiet=True)
        except Exception as e:
            result = _fail("processing error", str(e)[:100])
        row = make_result_row(tar_name, name, data, len(raw), result)
    return row, result is not None
