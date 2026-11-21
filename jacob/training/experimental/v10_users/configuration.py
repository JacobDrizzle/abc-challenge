"""Config schema for v10_users: best's schema plus two training-data keys.

train_users: null (every user) or a positive integer: train on that many users, the first n of one fixed order of
  the user names (training.subset_order), so the 25-user subset is inside the 50-user one. Validation keeps all its
  users.
user_window_fraction: a number in (0, 1]: train on the leading fraction of each user's windows (at least one). A
  random fraction would barely cut the information, since neighbouring windows share all but one sample.
Both need sampling "sqrt" or "window", whose user weights the subset sets; "user" draws uniformly and ignores them.
With both at their defaults, training is best's.
"""
from best.configuration import normalize_config as normalize_best

V10_DEFAULTS = {"train_users": None, "user_window_fraction": 1.0}


def normalize_config(config):
    best = normalize_best({k: v for k, v in config.items() if k not in V10_DEFAULTS})
    result = {**best, **V10_DEFAULTS, **{k: config[k] for k in V10_DEFAULTS if k in config}}
    users, fraction = result["train_users"], result["user_window_fraction"]
    if users is not None and (type(users) is not int or users < 1):
        raise ValueError("train_users must be null or a positive integer")
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError("user_window_fraction must be a number in (0, 1]")
    if (users is not None or fraction < 1) and result["sampling"] == "user":
        raise ValueError("train_users and user_window_fraction need sampling sqrt or window")
    return result
