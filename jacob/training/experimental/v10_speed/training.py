"""HostWindows: TrainingWindows with the window starts in host memory, and validation windows that can be scored
densely.

DeviceWindows keeps one int64 start per stride-1 window on the GPU: 0.5 GB for 1,600 users and 1.3 GB for 4,000,
half as much again as the samples themselves. A training step looks up 256 of them, so HostStarts keeps the list
in host memory and sends only those 256. They are the same integers, so batches() draws the same windows.

windows() returns validation.WindowBatches: the same batches, plus the data and stride dense_validate reads.
"""
import torch

from best.training import TrainingWindows

from experimental.v10_speed.validation import WindowBatches


class HostStarts:
    """The window starts [windows] in host memory; indexing with a device index returns the starts on the device."""

    def __init__(self, starts, device):
        self.values, self.device = starts.cpu(), device

    def __getitem__(self, index):
        return self.values[index.cpu()].to(self.device)

    def __len__(self):
        return len(self.values)

    def cpu(self):  # best.training.windows_fingerprint reads the starts this way
        return self.values


class HostWindows(TrainingWindows):
    def __init__(self, path, segments, device, *args, dense_validation=True, **kwargs):
        # Built in host memory, so the starts never visit the device; then the samples move there.
        super().__init__(path, segments, "cpu", *args, **kwargs)
        self.device = torch.device(device)
        self.buffer, self.span = self.buffer.to(self.device), self.span.to(self.device)
        self.starts = HostStarts(self.starts, self.device)
        self.dense_validation = dense_validation

    def windows(self, stride, batch_size):
        return WindowBatches(self, stride, batch_size, super().windows, self.dense_validation)
