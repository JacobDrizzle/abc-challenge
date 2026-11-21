"""v9 experiment: a cheaper streaming graph for the same weights (Track A), longer receptive fields and recipe
options (Track B), and distillation from the 9-checkpoint ensemble (Track C). Plan: 28 Oct, "v9: more accuracy,
less latency, safe ops only".

Runs through the unchanged v5 training, validation and benchmark scripts via hooks.installed(), as v6 and v7 do.
No v5, v6 or v7 file is edited, so their manifests still verify.
"""
