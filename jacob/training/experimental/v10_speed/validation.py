"""Dense validation: the trainer's validation score from one pass along each recording.

train_tcnn.validate cuts a 400-sample window for every scored position and runs the whole network on it. The
network looks back 393 samples (its receptive field), fewer than a window, so the forecast for the window ending at
sample a is the same when the network runs once along the whole recording and the forecast is read at position a
(model.LastPositionTCNN.dense). That costs one position a sample instead of 400 a window.

dense_validate scores the same anchors as DeviceWindows.windows(stride): for each segment, sample 399 and then every
`stride` samples while 20 future samples remain. The buffer holds the segments end to end; it is processed in chunks
with 400 samples of overlap on the left. An anchor is at least 399 samples into its own segment, so the 393 samples
it reads never come from the segment before it or from a chunk's zero padding.

The score is the same quantity as train_tcnn.score_windows: the window-weighted mean geodesic error per horizon
step, summed in float64. The two differ by float32 rounding in the network (about 1e-9 degrees in the mean). The
first call of a run checks that on its first windows and stops the run if they disagree.
"""
import itertools

import numpy as np
import torch

import train_tcnn

CHUNK = 16384  # positions a chunk: about 0.1 GB of activations in the stock block forward at 64 channels
OVERLAP = 400  # more than the receptive field, so every position kept from a chunk has its full history
CHECK_BATCHES = 4  # windowed batches (of 2,048) the first call compares with the dense forecasts
CHECK_PIECE = 256  # windows a forward in that check: a training batch, so the check needs no more memory than a step
CHECK_TOLERANCE = 1e-4  # largest allowed quaternion component difference; float32 rounding gives about 1e-7

windowed_validate = train_tcnn.validate  # the trainer's own, kept for runs without dense validation


class WindowBatches:
    """What DeviceWindows.windows() yields (the trainer's reference baselines iterate it), plus what dense_validate
    needs: the data it was cut from and the stride."""

    def __init__(self, source, stride, batch_size, batches, dense):
        self.source, self.stride, self.batch_size, self.batches, self.dense = source, stride, batch_size, batches, dense

    def __iter__(self):
        return iter(self.batches(self.stride, self.batch_size))


def anchors_for(bounds, context, horizon, stride):
    """The last history sample of every validation window, in buffer order: as DeviceWindows.windows, whose window
    starts are these minus (context - 1)."""
    anchors = []
    for offset, length in bounds:
        count = (length - context - horizon) // stride + 1
        anchors.append(offset + context - 1 + np.arange(count, dtype=np.int64) * stride)
    return np.concatenate(anchors)


def dense_forecasts(model, buffer, anchors, chunk=CHUNK):
    """Yields (forecasts [n, horizon, 4], their anchors [n]) chunk by chunk for sorted anchors on the buffer's
    device."""
    for begin in range(0, len(buffer), chunk):
        end = min(begin + chunk, len(buffer))
        low, high = torch.searchsorted(anchors, torch.tensor([begin, end], device=anchors.device)).tolist()
        if low == high:
            continue
        start = max(0, begin - OVERLAP)
        yield model.dense(buffer[start:end], anchors[low:high] - start), anchors[low:high]


def check_against_windows(model, windows, anchors):
    """Stops the run unless the dense forecasts of the first windows equal the windowed ones."""
    histories = torch.cat([history for history, _ in itertools.islice(windows, CHECK_BATCHES)])
    windowed = torch.cat([model(piece) for piece in histories.split(CHECK_PIECE)])
    first = anchors[:len(windowed)]
    dense = torch.cat([forecast for forecast, _ in dense_forecasts(model, windows.source.buffer, first)])
    difference = (dense - windowed).abs().max().item()
    if not difference <= CHECK_TOLERANCE:
        raise ValueError(f"Dense validation differs from windowed validation by {difference:.3g} on the first "
                         f"{len(windowed):,} windows")
    return difference


def dense_validate(model, windows, horizon):
    """train_tcnn.validate for WindowBatches with dense set; anything else goes to the trainer's own validate."""
    if not getattr(windows, "dense", False):
        return windowed_validate(model, windows, horizon)
    model.eval()
    source = windows.source
    with torch.inference_mode():
        anchors = torch.from_numpy(anchors_for(source.bounds, source.context, horizon, windows.stride)).to(source.device)
        if not getattr(source, "dense_checked", False):
            difference = check_against_windows(model, windows, anchors)
            print(f"Dense validation check: largest difference from windowed forecasts {difference:.2e}.", flush=True)
            source.dense_checked = True
        future = torch.arange(1, horizon + 1, device=source.device)
        sums = torch.zeros(horizon, dtype=torch.float64, device=source.device)
        for forecast, chunk_anchors in dense_forecasts(model, source.buffer, anchors):
            target = source.buffer[chunk_anchors[:, None] + future]
            sums += train_tcnn.geodesic_degrees_torch(forecast, target).sum(0)
    means = (sums / len(anchors)).cpu().numpy()
    return {"windows": len(anchors), "mean_endpoint_deg": float(means[-1]), "mean_deg_by_horizon": means.tolist()}
