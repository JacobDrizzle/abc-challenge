# v10_weight_edit: direct residual-head weight edits

This pilot tests whether a small, fixed change to the latest 800-user k8 model's
learned motion correction improves validation accuracy. It is inspired by
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/):
edit existing weights, evaluate, and retain evidence. It is a bounded calibration
experiment, not a reproduction of the article's circuit-discovery results.

The original checkpoints are never overwritten. There is no gradient computation,
training, architecture change, automatic promotion, or test-sample evaluation.

## Completed pilot

The 4 November run selected a 1.05 gain on all three correction axes. Its
seed-44 confirmation error changed from 4.472353 to 4.470089 degrees, but the
paired 95% interval for the change was [-0.006367, +0.001776] degrees. The
predeclared confirmation gate therefore failed; no candidate was promoted.
All originals remain unchanged. See the
[full report and evidence](../../../ablations/results/v10_weight_edit_20261104/README.md).

## Edit and baseline

The baseline is seed 44 from
`outputs/v10_k8_800users_full_pass/full_pass/selected_model.pt`: 7,555 parameters,
with recorded mean step-20 error **4.40362858 degrees** on all 120 validation users.
The experiment checks that number to within 0.001 degrees before screening.

The head predicts three polynomial coefficients for each of the x, y and z motion
axes. A candidate scales all three coefficients of an axis together, including
its bias where present. The feature extractor, damped-velocity prior and mirror
weight sharing stay unchanged. The gain is folded into existing weights, so it
does not add an inference operation.

There are **16 fixed edits plus identity**: gains 0.90, 0.95, 1.05 and 1.10 applied
either to every axis or to x, y or z alone. Every attempt is recorded. Cached
residuals accelerate screening; the finalist is evaluated through its actual
edited model.

## Selection and interpretation

- Read the merged data file `qualified_head_100hz_merged_20261101_140915.h5`, whose
  split contains 800 training, 120 validation and 160 test users. Verify its hash
  and the manifest hash against the source checkpoint's selection record.
- Sort validation IDs by SHA-256 of `v10-weight-edit-v1:<user_id>`. The first 80 are
  discovery users and the other 40 are confirmation users. Keep all recordings
  from a user together.
- Rank the fixed candidates on stride-5 discovery windows by mean step-20 error,
  then smallest gain distance from identity, then candidate ID. If identity wins,
  stop. Check only the winning edit on stride-1 discovery windows; stop if its
  mean does not improve. There is no second-candidate fallback.
- Freeze the edit and inference checkpoint before candidate confirmation. Apply
  the same gains to seeds 44, 42 and 43 without further tuning.
- Seed 44 passes when window-weighted and equal-user endpoint means both fall,
  more than 20 of 40 users improve, the paired 95% interval lies below zero, and
  neither endpoint p95 nor the mean across all 20 horizons increases. Endpoint
  means must also improve on seeds 42 and 43. Individual horizon changes are
  reported; they are not individual rejection thresholds.
- A passing candidate is exported alongside its baseline for graph, streaming
  parity, parameter/MAC and local CPU timing checks. Choosing a submission remains
  the team's decision.

Errors are sign-invariant quaternion geodesic angles, computed in float64, on
400-sample histories and 20-sample targets. Step 20 is 200 ms ahead. Intervals use
10,000 paired user-bootstrap draws with seed 0, preserving window weighting;
overlapping windows are not treated as independent examples.

All 120 validation users already influenced base-model early stopping and seed
selection. The 40 confirmation users are **withheld from edit selection**, not a
pristine holdout. The baseline itself is reproduced on all 120 before screening.
The 160 test users remain unused here; 20 belong to the previously reported team
benchmark and 140 are additional users. A passing pilot is validation evidence,
not proof of an improvement on the challenge's hidden users.

## Run and inspect

From `jacob/training`, in the existing training environment:

```powershell
python -m unittest experimental.v10_weight_edit.test_edits experimental.v10_weight_edit.test_metrics -v
python -m experimental.v10_weight_edit.run --output outputs/v10_weight_edit_20261104 --device cuda --batch-size 256
```

Use a new output directory for each run; an existing directory is refused. Run
one GPU job at a time. Do not edit the experiment or imported model/evaluation
code while it runs. Source hashes are recorded in the run manifest.

| Output | Meaning |
| --- | --- |
| `manifest.json` | Source/data hashes, frozen user lists, candidate list and selection rules |
| `originals/seed_*.pt` | Verified copies of all three source checkpoints |
| `baseline_reproduction.json` | Full validation baseline and cached-identity check |
| `cache/` | Discovery-only residual, prior, orientation and target arrays |
| `screen.json` | Every candidate's stride-5 metrics, ranking and diagnostic sums |
| `discovery_seed_44.json` | Actual-weight stride-1 check for the sole finalist |
| `frozen_edit.json` | Winner locked before candidate confirmation |
| `candidate_model.pt` | Separate inference artifact; remains an original copy if screening stops early |
| `confirmation_seed_*.json` | Original/edited metrics, paired intervals and diagnostic gate checks |
| `status.json` | Final verdict, elapsed time and any error |
| `preservation.json` | End-of-run hashes proving originals and their copies remain unchanged |
| `onnx/`, `export_checks.json` | Export verification and local timing, generated only after confirmation passes |

`no_screen_improvement` and `no_full_discovery_improvement` are valid negative
results. `not_confirmed` means the frozen edit did not pass the complete rule;
retain it only as experiment evidence. `promising` means it passed validation
and export checks, with the holdout limitation above. `failed` means an execution
or verification check failed and the run must not support an improvement claim.

The saved candidate is inference-only: it does not carry a resumable optimizer
state or claim that its source model's old validation score belongs to the edit.
