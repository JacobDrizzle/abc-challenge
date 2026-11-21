"""Cut clean segments into (context, target) training windows and split users into train/val/test."""
import hashlib

import numpy as np

CONTEXT = 400   # 4 s at 100 Hz
HORIZON = 20    # next 20 samples


def make_index(segments: list[np.ndarray], context: int = CONTEXT, horizon: int = HORIZON, stride: int = 10) -> np.ndarray:
    """Array [N, 2] of (segment, start). Windows never cross a segment boundary."""
    rows = [(s, a) for s, seg in enumerate(segments)
            for a in range(0, len(seg) - context - horizon + 1, stride)]
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def get_window(segments: list[np.ndarray], seg: int, start: int, context: int = CONTEXT, horizon: int = HORIZON):
    """(context [context,4], target [horizon,4]), both float32, w,x,y,z order."""
    q = segments[seg]
    return q[start:start + context], q[start + context:start + context + horizon]


def user_split(user_id: str, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> str:
    """Stable train/val/test assignment from the user id: the same user always lands in the same split,
    whatever else has been downloaded."""
    h = int(hashlib.sha1(f"{seed}:{user_id}".encode()).hexdigest(), 16) / 16**40
    return "train" if h < fractions[0] else "val" if h < fractions[0] + fractions[1] else "test"
