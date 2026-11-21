# Analyses outside the trainer

These analyses were run in Claude Code sessions between 26 and 30 Oct 2026, in temporary session folders that Windows
can clear. Their results (JSON, Markdown, logs) and the scripts that made them were copied here on 30 Oct.

- The scripts are kept as a record. Their paths point at the old session folders, `jacob/training/outputs/` and the
  archived `experimental.*` packages. To rerun one, fix its paths; the packages are in `jacob/training/archive/`
  (tag `v9-ablations` in the team's private history).
- In the public copy, machine-specific paths in scripts, logs and records are replaced by placeholders: `<repo>` (the repository root), `<scratch>` (a temporary session folder), `<local_test>` (the earlier local workspace) and `<venv>` (the training environment). Per-user scores are removed.
- Left out: ONNX graphs (rebuildable), the 120–140 MB ORT profiles and the torch kernel caches.
- The ledger (`../README.md`) cites each folder by experiment number.

| Folder | Experiment | Contents |
| --- | --- | --- |
| `v7_compression/` | 3, 4 | M1 latency sweep (nodes, not MACs), its part 2 suite and part 3 candidates; M2 head-basis rank; E1 v6 control through the reduced graph; the candidate size table (`confirm_candidates.py`); `v7_metrics.py`, which wrote `results/v7_metrics/`; an NPU latency guide (HTML) |
| `a1_generalisation_gap/` | 8 | Per-user model/damped ratio, train against validation, for v6 control seed 44 |
| `paired_tests/` | 9, 10, 17, 18, 20 | The paired per-user bootstrap tests. `v8_recipe_screen_paired.*` (v8 recipe screen); `angular_confirm_paired.*` (angular confirm, 6 seeds); `v9_jitter_screen_paired.json` (jitter screen; the script is `v9_screen_paired.py`, whose v9 screen output it overwrote); `d_confirm_paired*` and `d_paired.log` (D confirm); `v9_jitter_combined_paired.*` |
| `frame_robustness/` | 11, 18 | World-frame and axis conventions, on the v5–v8 models (`frame_robustness.*`) and the v9 models (`v9_frame_robustness.*`); posture spread and stream gravity estimates |
| `jitter_robustness/` | 17, 18 | Validation resampled at jittered times, for the screen, jitter and D models; the jitter screen plan |
| `timing_stress/` | 20 | The `v9_jitter_combined.json` check and the rate-ratio table. The main reports are in `../results/v9_timing_stress/`. |
| `track_a_graph/` | 14, 19 | Lean-graph accuracy and latency, the export variants (padded rows, A4, folding), ORT optimisation levels and profiles, and the final latency table (`final_latency_report.json`) |
| `distillation/` | 16 | The C0 ensemble teacher check and the C2 plan |
| `bench/` | 13, 19 | Bench export and scoring scripts, the 29 Oct scoring logs, the bench result JSONs from before the 29 Oct re-score (`backup_20261029/`, which the re-score matched), the v8 bench table, the Nic comparison and the mirror-split latency A/B |
| `reports/` | 18 | HTML pages made for the team: the k8_combined architecture and the v9 results |
| `promotion/` | 21 | `best_equivalence.py` (`.json`, `.log`): `best/` against v9, all exact. `v10_pattern_check.py`: the hook pattern in `jacob/training/experimental/README.md`. |
| `generalisation/` | – (5 Nov) | How well the two k8 models (100 and 800 training users) generalise: unseen-user accuracy on test20, test160 and the 100-user model's 965-user unseen pool; the seen-against-unseen gap (model/damped ratio by role); a breakdown by headset family, source rate and motion; validation against test. `score_users.py` (GPU), `report.py`, `test_generalisation.py`, `scores/`, `generalisation_report.{md,json}`. Per-window arrays are in the gitignored `jacob/training/outputs/generalisation/`. |
