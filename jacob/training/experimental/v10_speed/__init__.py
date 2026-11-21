"""v10_speed: the same k8_combined numbers, trained and validated faster, so the larger user sets are affordable.

Three changes, none to the weights a run learns from a given batch or to what a checkpoint holds:
- model.py: a forward that computes, in each block, only the positions the last position's forecast reads.
- validation.py: validation as one pass along each recording instead of one 400-sample window per forecast.
- training.py: the list of window starts stays in host memory, so 4,000 users fit an 8 GB card.
Built on best/ through hooks.installed(); README.md has the measurements.
"""
