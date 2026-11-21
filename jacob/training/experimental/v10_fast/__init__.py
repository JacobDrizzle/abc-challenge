"""v10_fast: the same k8_combined model, trained and validated with a forward that computes only what the last time
step reads. LeanTCNN runs every block at all 400 steps and then reads one; FastLeanTCNN runs each block at that
step's taps only, about 4 times faster to train on the 3080 and 15 times faster to validate. Parameters, checkpoints
and outputs are LeanTCNN's (float32 rounding apart). Built on best/ through hooks.installed(); README.md has the
measurements, the checks and the 800-user screen that uses the time saved.
"""
