"""Look at a recording's metadata (no frame decoding) to find the head device and the game.
Recordings without a head, and excluded games such as Tilt Brush, are not usable."""

import bson

from .config import EXCLUDED_GAMES


UNUSABLE_REASONS = ("excluded game", "no head device", "head has no rotation")


def peek_info(raw: bytes) -> dict | None:
    """Read only the metadata block of a replay. This skips the slow fpzip decode of the
    frames and events, so useless recordings can be rejected cheaply. None = unreadable."""
    try:
        return bson.decode(raw).get("info") or {}
    except Exception:
        return None


def find_head(info: dict) -> tuple[int, list[str]] | None:
    """Locate the head device. Returns (first frame column, axis names) or None.
    Frame columns are: time, then every device's axes in the order the devices are listed,
    so the head's columns start after all the axes of the devices before it."""
    devices = ((info.get("hardware") or {}).get("devices")) or []
    offset = 1
    for dev in devices:
        axes = list(dev.get("axes") or [])
        kind = str(dev.get("type") or "").upper()
        joint = str(dev.get("joint") or "").upper()
        if kind == "HMD" or joint == "HEAD":
            return offset, axes
        offset += len(axes)
    return None


def head_quat_columns(info: dict) -> list[int] | None:
    """Frame columns of the head's rotation quaternion (axes i, j, k, 1), or None."""
    head = find_head(info)
    if head is None:
        return None
    offset, axes = head
    try:
        return [offset + axes.index(a) for a in ("i", "j", "k", "1")]
    except ValueError:
        return None


def unusable_reason(info: dict) -> tuple[str, str] | None:
    """(category, detail) if the recording can't be used for head movement, else None.
    Excluded games (Tilt Brush) and recordings without a head device are skipped."""
    app = ((info.get("software") or {}).get("app")) or {}
    game = str(app.get("name") or "")
    if game.lower() in EXCLUDED_GAMES:
        return ("excluded game", game)
    if find_head(info) is None:
        devices = ((info.get("hardware") or {}).get("devices")) or []
        listed = ", ".join(f"{d.get('name', '?')} ({d.get('type', '?')})" for d in devices) or "no devices"
        return ("no head device", listed[:150])
    if head_quat_columns(info) is None:
        return ("head has no rotation", "head device has no i/j/k/1 axes")
    return None


def describe_game(data: dict) -> str:
    """Name (id, version) of the app/game the recording came from."""
    software = (data.get("info") or {}).get("software") or {}
    app = software.get("app") or {}
    extras = []
    if app.get("id") is not None:
        extras.append(f"id {app['id']}")
    if app.get("version"):
        extras.append(f"v{app['version']}")
    name = app.get("name") or "unknown"
    return name + (f" ({', '.join(extras)})" if extras else "")


def game_name(data: dict) -> str:
    """Just the app/game name (no id or version), for grouping."""
    app = (((data.get("info") or {}).get("software") or {}).get("app")) or {}
    return app.get("name") or "unknown"
