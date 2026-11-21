# Nic's two main models, retrained on the team data (26 Oct)

Trained **only** on `qualified_head_100hz_20261026_131953.h5` (group-a-v2): its 100 `/train` users train, its 15 `/validation`
users choose when to stop, its 20 `/test` users are the `group-a-v2-test20` bench. Same model settings as the earlier
main models (`../MODELS.md`); only the data changed. Five seeds each, one folder per seed.

| File in `sNN/` | Model | Registered as | Bench group |
| --- | --- | --- | --- |
| `gru_stateful_window.onnx` | Stateful GRU, velocity-only input, 32 units (`gru_stateful_step.onnx` is the same model with a 1-sample `context`) | `nic_ga2_gru_sNN` | `nic_ga2_gru` |
| `gadst.onnx` | GADS-T, rotation + velocity input | `nic_ga2_gadst_sNN` | `nic_ga2_gadst` |

Interface, chip-safety and how to stream the GRU: as in `../MODELS.md`. `export_check.json` is the ONNX-vs-PyTorch check
(max 0.034 deg). Scores: `LEADERBOARD.md`, section `group-a-v2-test20`.

Best of each, chosen on validation: GRU `s45`, GADS-T `s42`. Code, settings and how to reproduce: `../../code/README.md`
(`h5_to_npz.py` h5 -> per-player .npz, `ga2.sh` training on the PC, `ga2_export.sh` export, register, score).
