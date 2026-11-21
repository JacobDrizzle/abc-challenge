"""v7 experiment: compression options on the v6 model (dilation base, separable blocks, polynomial head basis,
L1 speed channel) and a streaming graph with fewer off-list operators (E1). Plan: 26 Oct, "v7 compact".

Runs through the unchanged v5 training, validation, test and benchmark scripts via hooks.installed(), as v6
does. No v5 or v6 file is edited, so their manifests still verify.
"""
