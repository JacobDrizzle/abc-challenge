"""Recipe options for train_tcnn (added in v9), installed by hooks.installed(); with all of them off, training is the
v5-v8 trainer's.

sampling: "user" (v5-v8) draws a uniform user, then a uniform window of that user, with DeviceWindows.batches'
  generator and draws. "sqrt" draws users in proportion to the square root of their window count and "window" in
  proportion to the count (uniform over windows): A1 (27 Oct) found that user balance re-draws, and the model
  memorises, the small users. k8_combined uses "window".
tilt_augmentation_deg: half of the windows are turned, history and future alike, by a uniform 0..max angle about a
  random horizontal world axis (+y is up). Body rates are unchanged; only gravity in the head frame moves, as it
  would for a tilted world frame or an unusual posture (the worst validation user sits 26 degrees back). The draws
  come from their own generator, so the windows are the ones the run would draw without tilt. k8_combined uses 15.
jitter_augmentation_ms: the organisers' test stream is "roughly ~100 Hz", with "some jitter" (28 Oct), and on
  jittered validation streams every model trained on the exact grid loses 0.16 degrees at +-2 ms. Each window, history
  and future alike, is taken at sample times k + e_k instead of k: per window a half-width uniform in 0..max ms, per
  sample e_k uniform within it, by SLERP between neighbouring samples (extrapolated along the end pair at either end).
  The target is the jittered window's 20th next sample, as the organisers score it. Own generator, as tilt.
distill_beta, teacher: the loss becomes (1 - beta) loss(truth) + beta loss(teacher forecast). distill.py caches the
  teacher forecasts as rotation vectors in the body frame of each window's last sample, so the forecast of a
  tilted window is the tilted forecast. The trainer's target stays the truth, so its logged errors are unchanged.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import sha256_file
from tcnn.data import DeviceWindows
from tcnn.model import conjugate, qmul, quaternion_to_rotvec, rotation_loss, rotvec_to_quaternion

ROOT = Path(__file__).resolve().parents[1]
SAMPLING_POWER = {"sqrt": .5, "window": 1.}  # user weight = window count ** power


def _generator(label, seed, epoch):
    """A CPU generator keyed by SHA-256 of label:seed:epoch, as DeviceWindows.batches keys its own."""
    key = hashlib.sha256(f"{label}:{seed}:{epoch}".encode("ascii")).digest()
    return torch.Generator().manual_seed(int.from_bytes(key[:8], "little") & (2**63 - 1))


def windows_fingerprint(windows):
    """SHA-256 of the samples and the window start order: row i of a teacher cache belongs to windows.starts[i]."""
    digest = hashlib.sha256(windows.buffer.cpu().numpy().tobytes())
    digest.update(windows.starts.cpu().numpy().tobytes())
    return digest.hexdigest()


def tilts(count, max_degrees, generator):
    """[count, 4] world rotations: half identity, half a uniform 0..max angle about a random horizontal axis."""
    angle = torch.rand(count, generator=generator, dtype=torch.float64) * math.radians(max_degrees)
    angle = angle * (torch.rand(count, generator=generator, dtype=torch.float64) < .5)
    heading = torch.rand(count, generator=generator, dtype=torch.float64) * 2 * math.pi
    axis = torch.stack((heading.cos(), torch.zeros_like(heading), heading.sin()), -1)  # no y: horizontal
    return torch.cat(((angle / 2).cos()[:, None], (angle / 2).sin()[:, None] * axis), -1).float()


def jitter_offsets(count, length, max_ms, generator):
    """[count, length] sample-time offsets in samples: per window a half-width uniform in 0..max_ms, per sample
    uniform within it. 10 ms is one sample at 100 Hz."""
    width = torch.rand(count, 1, generator=generator) * max_ms / 10
    return (torch.rand(count, length, generator=generator) * 2 - 1) * width


def resampled(window, offsets):
    """Windows [B, T, 4] taken at positions k + offsets[:, k] by SLERP between the neighbouring samples."""
    length = window.shape[1]
    position = torch.arange(length, device=window.device, dtype=offsets.dtype) + offsets
    low = position.floor().long().clamp(0, length - 2)  # the end pairs extrapolate
    a = torch.gather(window, 1, low[..., None].expand(-1, -1, 4))
    b = torch.gather(window, 1, (low + 1)[..., None].expand(-1, -1, 4))
    step = quaternion_to_rotvec(qmul(conjugate(a), b))  # shortest rotation from a to b
    return F.normalize(qmul(a, rotvec_to_quaternion((position - low)[..., None] * step)), dim=-1)


class TeacherChannel:
    """Hands the teacher forecast [B, horizon, 4] of the batch being trained from the batches to the loss."""
    forecast = None


class TrainingWindows(DeviceWindows):
    """DeviceWindows with the sampling, tilt, jitter and teacher options. Validation uses windows(), unchanged."""

    def __init__(self, path, segments, device, context=400, horizon=20, *, sampling="user", tilt_deg=0,
                 teacher=None, channel=None, jitter_ms=0):
        super().__init__(path, segments, device, context, horizon)
        self.sampling, self.tilt_deg, self.teacher, self.channel = sampling, tilt_deg, teacher, channel
        self.jitter_ms = jitter_ms
        if sampling != "user":
            self.user_weights = self.user_count.double() ** SAMPLING_POWER[sampling]
        self.cache = None

    def teacher_cache(self):
        """[windows, horizon, 3] teacher rotation vectors on the device, refused unless made for these windows."""
        if self.cache is None:
            folder = ROOT / self.teacher  # a relative name is under jacob/training
            info = json.loads((folder / "teacher.json").read_text(encoding="utf-8"))
            if info["windows_fingerprint"] != windows_fingerprint(self):
                raise ValueError(f"{folder}: the teacher cache was made for other training windows")
            if sha256_file(folder / "teacher.npy") != info["sha256"]:
                raise ValueError(f"{folder}: teacher.npy changed after it was cached")
            self.cache = torch.from_numpy(np.load(folder / "teacher.npy")).to(self.device)
        return self.cache

    def batches(self, seed, epoch, steps, batch_size):
        """DeviceWindows.batches (for "user" the same draws, so the same windows), then tilt, jitter and the teacher."""
        # The tilt and jitter generators keep their v9 labels: a new label would draw other angles and offsets.
        generator, turns = _generator("device-window-v1", seed, epoch), _generator("v9-tilt-v1", seed, epoch)
        times = _generator("v9-jitter-v1", seed, epoch)
        cache = self.teacher_cache() if self.teacher else None
        for _ in range(steps):
            if self.sampling == "user":
                user = torch.randint(len(self.users), (batch_size,), generator=generator)
            else:
                user = torch.multinomial(self.user_weights, batch_size, replacement=True, generator=generator)
            count = self.user_count[user]
            within = (torch.rand(batch_size, generator=generator, dtype=torch.float64) * count).long()
            index = (self.user_offset[user] + torch.minimum(within, count - 1)).to(self.device)
            history, future = self._gather(self.starts[index])
            if self.tilt_deg:
                turn = tilts(batch_size, self.tilt_deg, turns).to(self.device)[:, None]  # [B, 1, 4]
                history, future = qmul(turn, history), qmul(turn, future)
            if self.jitter_ms:
                offsets = jitter_offsets(batch_size, self.context + self.horizon, self.jitter_ms, times)
                window = resampled(torch.cat((history, future), 1), offsets.to(self.device))
                history, future = window[:, :self.context], window[:, self.context:]
            if cache is not None:
                last = F.normalize(history[:, -1:], dim=-1)
                self.channel.forecast = qmul(last, rotvec_to_quaternion(cache[index]))
            yield history, future


def distillation_loss(beta, channel):
    """rotation_loss with the same kind and endpoint weight, against truth and teacher: (1 - beta) and beta."""
    def loss(prediction, target, kind, endpoint_weight):
        truth = rotation_loss(prediction, target, kind, endpoint_weight)
        return (1 - beta) * truth + beta * rotation_loss(prediction, channel.forecast, kind, endpoint_weight)
    return loss
