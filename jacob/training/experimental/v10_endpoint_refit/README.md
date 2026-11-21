# v10_endpoint_refit: a small, training-only correction at 200 ms

This separate experiment starts from copies of the original 800-user k8 models.
It fits 97 output coefficients without backpropagation, keeping the temporal
feature extractor, architecture, 7,555 parameters and deployment interface fixed.
The earlier weight-edit experiment is read-only. No test samples are evaluated.

## What the fit means

The completed run met the exploratory criteria on all three seeds. On the 40
reused validation users, seed 44's endpoint error fell from 4.47235° to
4.46649°, with 28 users improving. Its fitted streaming graph verified at
7,555 parameters and 16,222 multiply-accumulates per call. The selected
checkpoint and original are separate files. [Full results and model links](../../../ablations/results/v10_endpoint_refit_20261105/README.md)

The original head has three polynomial terms for each rotation axis. At the
200 ms endpoint all three basis values equal one. Their sum is a linear map
with 96 weights plus the single permitted x-axis bias.

For hidden streams h0 and h1, the x features are h0+h1 plus an intercept; the
y and z features are h0-h1 without intercepts. This preserves mirror symmetry.
The fitted change is added only to the cubic term's weights. In residual space,
the change at time t is `(t/200 ms)^3` times its endpoint value; the quaternion
conversion is nonlinear, so angular changes need not follow that exact ratio.

`fit.py` extracts the frozen features and builds float64 normal equations. With
target unit quaternion q, original unnormalized prediction a=(2,r), and the
linear correction Jacobian J, the objective is:

`sum_i u_i * ||(I - q_i q_i^T)(a_i + J_i delta)||^2 / ||a_i||^2 + lambda * ||D delta||^2`

The weights u are inverse original angular error, floored at one degree, then
normalized to sum to one. D contains training feature RMS values, floored at
1e-6, with the intercept scale fixed at one. The projection does not depend on
quaternion sign and does not divide by its scalar part. This is a quadratic
surrogate, not the exact mean geodesic loss; actual geodesic errors decide results.

Seven penalties (1e-4 through 1e2) each produce one CPU linear solve. Cap each
solution's training RMS residual change at 10% of the original learned residual
RMS, then try 0.25, 0.5 and 1.0 times that capped solution. There are 21 fitted
candidates, alongside the unchanged and previous 1.05-gain controls.

## Frozen protocol

- Fit on all 800 training users at stride 20: exactly 1,620,028 windows.
- Reproduce the original seed-44 validation score, 4.403628579547263 degrees,
  over all 4,755,761 windows before screening; tolerance 0.001 degrees.
- Reuse the prior 80/40 validation-user partition. Both subsets are now reused
  development data, not independent holdouts.
- Screen on the 80 users at stride 5. Rank by endpoint mean, training RMS
  correction, then identifier. Stop if either control wins.
- Check only the winning fitted model on all stride-1 discovery windows using
  actual edited-model inference. Stop unless it beats both controls.
- Freeze its penalty, strength and fitting rule before the remaining 40 users.
  Refit each other seed on its own training features with those same settings;
  do not retune or average independently initialized weights.
- A promising exploratory result requires seed44's endpoint mean to beat both
  controls; its equal-user mean and majority of users must improve against the
  original, the paired 95% interval must lie below zero, and p95 and all-horizon
  mean must not increase. Seeds42/43 must improve endpoint means over originals.
- Pair bootstrap users, not overlapping windows: 10,000 draws, seed0. Report
  every horizon and median/p95/p99. No tuning follows the final assessment.
- Export and compare ONNX parity/topology/operators/latency only for a promising
  candidate. There is no automatic promotion or change to the leaderboard.

## Run

From `jacob/training`, use the existing Torch 2.7.1+cu118 environment:

```powershell
python -m unittest experimental.v10_endpoint_refit.test_refit -v
python -m experimental.v10_endpoint_refit.run --output outputs/NEW_ENDPOINT_REFIT_RUN
```

The output directory must be new. The default inference batch is512 on CUDA.
The CPU solves only97 variables. Training statistics are streamed; only stride-5
discovery features are cached. Run one GPU job at a time and do not edit hashed
source files while the run is active.

## Evidence

The output holds the source/data/code hashes, explicit train and validation
membership, all21 seed44 checkpoint copies, normal equations, complete candidate
ranking, per-user results, frozen recipe and source-preservation checks. A
`candidate_model.pt` is created only after a fitted finalist passes the full
discovery check. It is inference-only and correctly marked as fitted, with no
stale optimizer state or inherited accuracy claim.

`control_won_screen` and `no_full_discovery_improvement` stop the experiment early.
`not_promising` records a completed but unsuccessful exploratory assessment.
`promising_exploratory` requires all agreed gates and successful export checks.
`failed` means a verification or execution problem; it is not an accuracy result.

The final preservation audit also rechecks the first experiment's original
output files. Existing source checkpoints, earlier results, working changes and
Git history are left intact.
