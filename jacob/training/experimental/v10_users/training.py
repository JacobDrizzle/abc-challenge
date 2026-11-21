"""SubsetWindows: TrainingWindows restricted to a nested subset of the users, or to a leading fraction of each user's
windows (configuration.py).

The trainer builds its validation windows with the same class but reads them only through windows(), so the
restriction is applied on the first batches() call: validation keeps every user. Unrestricted, the class is
TrainingWindows; restricted to every user and every window, it draws the same batches.
"""
import hashlib

import torch

from best.training import SAMPLING_POWER, TrainingWindows


def subset_order(users):
    """The user names in a fixed order keyed by SHA-256 of each name, so it doesn't depend on the training seed or on
    which other users are present. The first n are the n-user subset."""
    return sorted(users, key=lambda user: hashlib.sha256(f"v10-users-v1:{user}".encode("utf-8")).hexdigest())


class SubsetWindows(TrainingWindows):
    def __init__(self, *args, train_users=None, user_window_fraction=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_users, self.user_window_fraction = train_users, user_window_fraction
        self.pending = train_users is not None or user_window_fraction < 1
        if self.pending and self.sampling == "user":
            raise ValueError("train_users and user_window_fraction need sampling sqrt or window")

    def restrict(self):
        """Shrink each user's window count to its leading fraction, then weight only the subset's users."""
        users = len(self.users) if self.train_users is None else self.train_users
        if users > len(self.users):
            raise ValueError(f"train_users is {users}, but there are only {len(self.users)} training users")
        total = int(self.user_count.sum())
        self.user_count = (self.user_count.double() * self.user_window_fraction).floor().long().clamp(min=1)
        keep = torch.zeros(len(self.users), dtype=torch.bool)
        keep[[self.users.index(user) for user in subset_order(self.users)[:users]]] = True
        self.user_weights = self.user_count.double() ** SAMPLING_POWER[self.sampling] * keep
        print(f"v10_users: training on {users} of {len(self.users)} users and "
              f"{int(self.user_count[keep].sum()):,} of {total:,} windows", flush=True)
        self.pending = False

    def batches(self, seed, epoch, steps, batch_size):
        if self.pending:
            self.restrict()
        return super().batches(seed, epoch, steps, batch_size)
