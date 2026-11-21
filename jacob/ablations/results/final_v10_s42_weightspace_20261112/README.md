# Final seed-42 endpoint refit: completed, below the promised gain threshold

Completed on 12 November 2026. **Outcome: `not_promising` under the predeclared gate. Keep the unchanged final model as the reference.**

On 262 assessment users, the frozen correction reduced window-weighted 200 ms error from **4.596399752 degrees to 4.595389022 degrees**. The improvement was **0.001010730 degrees (0.0220%)**, below the required **0.005 degrees**. The paired 95% interval for candidate minus original was **[-0.001523503, -0.000433426] degrees**, and **199/262 users improved**.

The interval supports a small improvement on this development cohort. It does not meet the agreed practical threshold or establish performance on unseen challenge users. No retuning followed assessment. The temporal-timing arm was not run.

## Assessment results

These are 10,166,278 identical eligible windows, evaluated at stride 1 using actual saved edited weights through the verified dense PyTorch path. Lower is better.

| Model | 200 ms mean | 100 ms mean | Equal-user endpoint mean | Endpoint p95 | Endpoint p99 | All-horizon mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged final model | 4.596399752 | 2.281554754 | 4.442106979 | 11.192977050 | 18.290709355 | 2.360265246 |
| 1.05 residual-gain control | 4.599217599 | 2.280691448 | 4.446323745 | 11.178458830 | 18.241109979 | 2.361572656 |
| Frozen endpoint correction | 4.595389022 | 2.281546066 | 4.440780211 | 11.192802610 | 18.287604499 | 2.360156114 |

All values are degrees. The candidate also beat the gain control by 0.003828577 degrees at 200 ms.

| Assessment gate | Result |
| --- | --- |
| All 262 fixed assessment users | Pass |
| At least 0.005-degree endpoint improvement | **Fail** |
| Paired 95% interval below zero | Pass |
| Improved equal-user mean | Pass |
| Majority of users improved | Pass |
| Endpoint p95 no worse | Pass |
| Average across all 20 horizons no worse | Pass |
| Endpoint mean beats gain control | Pass |

The 110-150 ms horizons became slightly worse, even though the average across all horizons improved. These small regressions are retained below.

## What was run

- Starting point: the exact final ONNX identified by the registry as seed 42. Original training checkpoint/logs remain unavailable locally.
- Recovered every learned value from ONNX, including the three bias entries stored in a Constant node. The inference-only checkpoint re-exported to a **byte-identical ONNX**, SHA-256 `28221e1e0b559e8d4d900f8b448acd22a0fbe0da38f904cb12d4e268aca53cad`.
- Fit one sign-invariant regularized least-squares correction on **10,237,229 windows from all 5,243 training users**, stride 20. No backpropagation or full-network training.
- Fixed penalty **0.0001** and strengths **0.25, 0.5, 1.0**. Only 96 cubic-head weights and one cubic x bias change. The temporal network, prior and all other graph values remain exact.
- Three fitted candidates plus unchanged and 1.05-gain controls; no additional search.
- Fixed 524-user discovery / 262-user assessment partition. The older 120 validation users all belong to discovery; remaining membership uses the salted SHA-256 rule in the protocol.
- Screened 4,048,710 discovery windows at stride 5. Checked the sole fitted winner on all 20,240,522 discovery windows at stride 1, then froze it before assessment.
- Selected **ridge_0.0001_strength_1**. Its endpoint correction RMS is **1.941298%** of this model's original training residual RMS; the 10% cap did not bind.
- Main run took **349.3 seconds (5.82 minutes)**, including baseline, fitting, screening and assessment. The training-only pilot's conservative estimate was about 10.3 minutes. Environment setup and preflight checks are outside that duration.
- Paired intervals resample users, not overlapping windows: 10,000 draws, seed 0, preserving window-weighted aggregation.
- No test samples were evaluated, including the standard 20-player cohort and all 1,049 final test users.

Both validation subsets remain development data. The final model may already have used these validation users for original checkpoint selection, and its complete training/selection history was not reconstructed. Matching final-data seeds 43/44 were unavailable; older models were not substituted.

## Complete screen ledger

| Candidate | Discovery stride-5 endpoint mean | Difference from unchanged |
| --- | ---: | ---: |
| `identity` | 4.515848146 | +0.000000000 |
| `global_105` | 4.519777126 | +0.003928981 |
| `ridge_0.0001_strength_0.25` | 4.515394926 | -0.000453220 |
| `ridge_0.0001_strength_0.5` | 4.515025465 | -0.000822681 |
| `ridge_0.0001_strength_1` | 4.514537881 | -0.001310265 |

Full-discovery endpoint errors were 4.515850683 degrees unchanged, 4.519750736 degrees for the gain control and 4.514544235 degrees for the frozen fitted candidate.

The unchanged baseline on all 786 validation users was **4.542781640 degrees** over 30,406,800 windows. It exactly reproduced all per-user baseline sums in the later discovery and assessment passes. This is a different cohort from the saved standard-20 benchmark; the two scores should not be compared as a model change.

## Assessment changes at every horizon

Candidate minus unchanged, degrees. Negative means improvement.

| Horizon | Change |
| --- | ---: |
| 10 ms | -0.000000573 |
| 20 ms | -0.000003407 |
| 30 ms | -0.000009077 |
| 40 ms | -0.000017600 |
| 50 ms | -0.000028060 |
| 60 ms | -0.000037997 |
| 70 ms | -0.000043956 |
| 80 ms | -0.000042103 |
| 90 ms | -0.000030384 |
| 100 ms | -0.000008688 |
| 110 ms | +0.000019025 |
| 120 ms | +0.000046285 |
| 130 ms | +0.000065586 |
| 140 ms | +0.000064749 |
| 150 ms | +0.000032558 |
| 160 ms | -0.000039434 |
| 170 ms | -0.000161928 |
| 180 ms | -0.000351587 |
| 190 ms | -0.000625323 |
| 200 ms | -0.001010730 |

## Verification and preservation

- **26 tests passed**, covering independent normal-equation algebra, quaternion sign/mirror invariance, near-180-degree targets, inactive features, caps, user bootstrap, split guards, target indices, dense chunk boundaries, no future-input leakage, graph recovery, edit masks and original preservation.
- Source-model streaming parity passed on stationary, constant-rotation, accelerating/braking, sign-flipped, extreme/near-180-degree, long-sequence and actual training inputs. CPU/CUDA dense predictions remained below the fixed 0.001-degree discrepancy gate; recurrent states met `atol=1e-5, rtol=1e-5`.
- Full discovery and assessment use the saved candidate head on the unchanged frozen features. Actual head results were checked against cached corrections and complete edited-model window inference for every user. Largest checked differences were 0.000207057 degrees in discovery and 0.000310257 degrees in assessment.
- An independent integrity pass rechecked all per-user sums, exact baseline reproduction, split membership/counts and the paired bootstrap interval.
- All three fitted ONNX copies change exactly **97** learned scalars. The gain control changes **291**. Every other graph value/structure and all frozen PyTorch parameters were checked exact.
- All copies retain **7,555 parameters, 124 ONNX nodes, 44,196 bytes and 16,222 MACs/call**. Operator audits passed.
- Original ONNX and its three untouched starting copies retained their original SHA-256. Recorded pre-existing files and experiment source hashes remained unchanged. Historical experiment files were not edited.
- The independent CUDA environment uses Python 3.12.10, Torch 2.7.1+cu118, NumPy 2.2.6, h5py 3.13.0, ONNX 1.19.1 and ONNX Runtime 1.23.2. Complete installed versions are retained in the environment lock and execution protocol.
- Since the minimum gain failed, the promised further deployment qualification was not run: **no finalist ONNX-vs-PyTorch runtime parity result and no new latency measurement are claimed**. The ONNX copies have structural/weight audits, not final release qualification.
- No model was promoted. The submission, model registry, leaderboard and Git index/history were left unchanged.

## Artifacts

- [Frozen experimental ONNX — below the gain threshold](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/candidates/ridge_0.0001_strength_1.onnx)
- [Matching inference-only fitted checkpoint](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/candidates/ridge_0.0001_strength_1.pt)
- [Preserved unchanged original](../../../training/outputs/final_v10_s42_weightspace_20261111/originals/final_s42.onnx)
- [Frozen protocol and membership](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/protocol.json)
- [Complete candidate ledger](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/candidates.json)
- [Assessment metrics, per user and per horizon](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/assessment.json)
- [Assessment decision and intervals](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/assessment_decision.json)
- [Independent integrity audit](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/independent_integrity_audit.json)
- [Preservation audit](../../../training/outputs/final_v10_s42_weightspace_20261111/execution/preservation.json)
- [Environment lock](../../../training/outputs/final_v10_s42_weightspace_20261111/environment.lock.txt)
- [Implementation explanation](../../../training/experimental/final_v10_s42_weightspace/README.md)

The negative gate result is retained alongside every attempted candidate. Any larger edit, additional fit, timing experiment, promotion or test evaluation would be a separate team decision.
