"""Install v10_narrow on top of v10_speed: NarrowTCNN, its config key and its hashed files."""
from contextlib import contextmanager

from experimental.v10_speed import hooks as speed_hooks

from experimental.v10_narrow.configuration import normalize_config
from experimental.v10_narrow.model import NarrowTCNN

FILES = tuple(f"experimental/v10_narrow/{name}.py" for name in ("__init__", "configuration", "hooks", "model"))


@contextmanager
def installed():
    with speed_hooks.installed(model=NarrowTCNN, normalize=normalize_config, model_keys=("block_inputs",), files=FILES):
        yield
