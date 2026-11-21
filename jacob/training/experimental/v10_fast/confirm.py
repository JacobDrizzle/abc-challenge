"""Confirm candidates on every seed with the fast forward: best/confirm.py with this package's hooks.

python -m experimental.v10_fast.confirm --plan <plan>.json --output outputs/<report folder> [--data ...]
    [--candidates ...] [--dry-run]

The plan format, the report and the rule are best/confirm.py's. The reference should be a folder trained with this
package too (the control of a v10_fast suite, confirmed here as a candidate first), so that both sides of the
comparison ran the same code.
"""
import best.confirm
from best.hooks import _swapped

from experimental.v10_fast.hooks import installed


def main():
    # best.confirm.main enters the name `installed` of its own module; here that is this package's.
    with _swapped(best.confirm, installed=installed):
        best.confirm.main()


if __name__ == "__main__":
    main()
