# Our chip-safe models vs our best GRU (same 47 test players, every window)

Test set: our 47 fixed test players (none trained or validated on), all 1,752,486 windows (stride 1). Error = degrees at the 20th predicted sample only (0.2 s ahead). Models trained on big data (1,229 players); one trained model each (seed 42); stateful GRU scored as the harness runs it (memory from the start of each recording, one sample at a time). Gap and 95% CI: paired over players (2,000 resamples of the 47 players, window-weighted). Players beating = players whose own average miss is lower than the reference's.

| Model | Error 0.2 s ahead (deg, step 20) | Extra error vs GRU 32 (deg), 95% range over players | Test players where it beats GRU 32 | Weights (params) | Work per prediction (MACs) | Time per prediction, Mac M5 Pro, 1 core | Time per prediction, PC Ryzen 7 5800X3D, 1 core | Operators outside the chip's safe list |
|---|---|---|---|---|---|---|---|---|
| GRU 32 (stateless; rotation + velocity + sizes) | 5.085 | reference | reference | 6,768 | 1,620,000 | 0.342 ms | 0.299 ms | GRU (400-step loop), ReduceL2 |
| Stateful GRU (velocity-only; 1 new sample per call) | 5.172 | +0.087 (+0.069 to +0.105) | 5 of 47 | 6,192 | 6,384 | 0.012 ms | 0.026 ms | none |
| GADS-T (stateless; rotation + velocity) | 5.200 | +0.115 (+0.084 to +0.142) | 7 of 47 | 10,352 | 606,320 | 0.188 ms | 0.234 ms | none |
| Hold last pose (no model) | 6.963 | +1.878 | 0 of 47 | 0 | 0 | - | - | - |
| Damped velocity (no model) | 5.982 | +0.897 | 0 of 47 | 0 | - | - | - | - |

Reference on the same windows: hold-last 6.963, damped velocity 5.982.
Mean of 5 training runs (stride-10 windows): GRU 32 5.084, stateful GRU 5.159, GADS-T 5.159 (seed 42 is GADS-T's second-worst run).

CPUs: Mac = Apple M5 Pro, 18 cores (6 performance + 12 efficiency); PC = AMD Ryzen 7 5800X3D, 8 cores / 16 threads. Both ONNX Runtime 1.31 (CPU), median of 1,000 runs after 100 warm-up runs, batch 1.
Core count does not help these models: with all cores (18 on the Mac, 16 on the PC) every model is the same or slower (GADS-T 0.50 / 0.40 ms, GRU 32 0.33 / 0.39 ms, stateful GRU 0.012 / 0.038 ms), because one prediction is too small to split across cores. So the single-thread number is the one that matters.
