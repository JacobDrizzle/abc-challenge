"""v6 experiment: gravity channels (1a) and exact left-right mirror equivariance (1b) on the v5 streaming TCNN.

Runs through the unchanged v5 training, validation, test and benchmark scripts via hooks.installed(), which
swaps their module-level names in this process only. No v5 file is edited, so v5 manifests still verify.
"""
