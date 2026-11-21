# Final seed-42 endpoint experiment

This package implements the endpoint-only protocol the team approved on 11 November 2026, after the final model and data were verified. It operates on separate copies of the final `moredata_final` seed-42 ONNX. It does not use the older seed-44 correction, run full-network training, or evaluate test users.

## What each file does

- `recover.py` reverses the documented ONNX tensor layouts into an inference-only PyTorch model. All 7,555 learned values are accounted for, including three biases in a Constant node. Re-exporting the recovered model produced the exact original ONNX bytes. Candidate export patches only the approved entries in a copy of that original graph.
- `engine.py` runs the frozen temporal network once along each segment. The 393-feature receptive field fits inside each 400-sample history. A 400-position overlap makes chunk boundaries harmless. Stride selects prediction anchors; it never skips temporal state updates or crosses segment boundaries. Targets are read separately and never enter the network.
- `common.py` fixes model/data identities, user partitioning, limits and the assessment gate.
- `verify.py` compares streaming ONNX with the recovered streaming wrapper and dense CPU/CUDA predictions, checks recurrent states and quaternion invariances, and times original/candidate ONNX in alternating rounds.
- `run.py` establishes the validation baseline, fits one 97-variable correction, screens three strengths and two controls, checks one finalist on full discovery, freezes it, and performs one assessment. It reuses the existing sign-invariant float64 least-squares equations and user-bootstrap metrics without importing the historical runner.
- `test_experiment.py` verifies reconstruction, permitted edits, preservation, causality, chunk boundaries, target indexing, split guards and the new assessment threshold. The preflight also runs the existing fitting-algebra and metric tests.

The dense evaluation uses the actual saved edited head on the unchanged hidden features. This is exact reuse of frozen network computation. On the first batch of each user, it checks the result against both full-window edited-model inference and the cached additive formula.

## Fixed protocol

Fit once on all 5,243 training users at stride 20: 10,237,229 examples. Use penalty 0.0001 and strengths 0.25, 0.5 and 1.0, with correction RMS capped at 10% of the original model's training residual RMS. Only cubic-head entries may change: 96 weights and one x bias. The global 1.05-gain comparator is labelled separately and changes the full 291-parameter residual head.

Discovery contains 524 validation users, including all 120 older development users. Assessment contains 262 other validation users selected by the frozen salted-hash rule. Discovery has 4,048,710 stride-5 screening windows and 20,240,522 stride-1 windows. Assessment has 10,166,278 stride-1 windows.

The screen ranks by endpoint mean, smaller training RMS correction, then identifier. If a control wins, the experiment stops. If the fitted winner fails to beat both controls at full discovery, it stops without trying another. One finalist is frozen before assessment.

Assessment requires at least 0.005-degree endpoint improvement over unchanged, a paired user-bootstrap 95% interval below zero, improved equal-user mean and majority of users, no increase in endpoint p95 or all-horizon mean, and lower endpoint error than the gain comparator. These are development-data results; neither the new partition nor reconstruction establishes independent unseen-user evidence.

## Artifacts and execution

The dedicated environment and all artifacts are under:

`jacob/training/outputs/final_v10_s42_weightspace_20261111/`

`preflight/` holds reconstruction, streaming parity, tests and training-only timing. `execution/` holds the frozen protocol, exact cohorts, source snapshots, baseline, normal equations, partial progress, every candidate, rankings, full discovery, and assessment if reached. `execution_run.log` records progress. The main-run budget is six hours; timeout is an incomplete run, not an accuracy result.

Run from `jacob/training` using the experiment environment:

```powershell
& 'outputs/final_v10_s42_weightspace_20261111/.venv/Scripts/python.exe' -B -m experimental.final_v10_s42_weightspace.run --pilot
& 'outputs/final_v10_s42_weightspace_20261111/.venv/Scripts/python.exe' -B -u -m experimental.final_v10_s42_weightspace.run
```

The main command deliberately refuses an existing `execution/` directory. It also requires passing preflight records with unchanged source hashes. **Do not rerun it over completed evidence.** A repetition requires a new isolated output root and its own recorded protocol; do not delete or overwrite this run to make the command proceed.

The recovered checkpoint contains inference weights/specification and provenance only. It is not Nic's original training checkpoint and contains no optimizer state or recovered training history. The submission, registry and leaderboard are not changed by this package.
