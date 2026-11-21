"""The current best model and everything needed to train, confirm, export and stress-test it.

Promoted on 30 Oct 2026: v9 k8_combined (validation 4.2501 +- 0.0041 deg over seeds 42-44, test 4.2666 deg, 7,555
parameters). The code is the v6, v7 and v9 experiment code (archive/v6_gravity_mirror, v7_compact, v9_lean) moved
into one package with its behaviour unchanged, so best/configs/base.json trains exactly as k8_combined did. The record
of every experiment behind it is jacob/ablations/README.md.

Runs through the unchanged v5 training, validation and benchmark scripts via hooks.installed(). Next-round
experiments go in experimental/ and hook onto this package (experimental/README.md).
"""
