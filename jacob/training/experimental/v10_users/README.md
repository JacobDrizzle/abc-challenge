# v10_users: does k8_combined gain from more training users?

**Question.** k8_combined trains on 100 BOXRR users. Its training and validation error fall together, and it doesn't memorise. Before anyone downloads more users, this screen reads the data-scaling curve: does validation error still fall from 50 to 100 users? If it does, is it the number of users (diversity) or the number of windows (volume) that helps?

## The two keys (`configuration.py`, `training.py`)

| Key | Default | Meaning |
| --- | --- | --- |
| `train_users` | `null` (every user) | Train on the first n users of a fixed order: SHA-256 of `v10-users-v1:<name>`. That order doesn't depend on the seed or on which other users are present, so the subsets are nested (25 ⊂ 50 ⊂ 100). |
| `user_window_fraction` | `1.0` | Train on the leading fraction of each user's windows (at least one). A random fraction would barely cut the information, since neighbouring windows share all but one sample. |

- Both keys need `sampling` `sqrt` or `window`, because the subset works by setting those modes' user weights.
- The trainer builds its validation windows with the same class, so the restriction is applied on the first `batches()` call. That way validation keeps all 15 of its users.
- With both keys at their defaults, or set to every user and every window, training is bit-identical to best's.

## The screen (`configs/screen.json`, seed 42)

| Variant | Override | Users | Training windows | Share |
| --- | --- | ---: | ---: | ---: |
| control (k8_combined, reused from `outputs/v9_confirm/k8_combined/seed_42`) | — | 100 | 3,844,021 | 100% |
| users_25 | `train_users: 25` | 25 | 799,563 | 20.8% |
| users_50 | `train_users: 50` | 50 | 1,703,493 | 44.3% |
| users_100_half | `user_window_fraction: 0.5` | 100 | 1,921,990 | 50.0% |

- Every variant runs the same 40 epochs × 1,000 steps × 256 windows, so a smaller subset is seen more times.
- users_50 and users_100_half have nearly the same volume. users_100_half has 13% more windows, which slightly favours it.

```
python -m experimental.v10_users.run --stage plan --output outputs/v10_users
python -m experimental.v10_users.run --stage screen --output outputs/v10_users --variants users_25 users_50 users_100_half
```

- **Parity (31 Oct):** a 1-epoch run through this package's hooks with `train_users: 100` reproduced k8_combined's seed-42 epoch 1 exactly: training loss 0.0714481890797615, validation 4.498707439826126.
- Tests: `python -m unittest experimental.v10_users.test_v10 -v` (9 tests). `best.test_best` still passes (36 tests).

## How to read it

The results are paired per user against the control, on the 15 validation users.
- **users_50 → 100 gains clearly** (≳ 0.03°, several times the 0.004° seed SD): more users should pay off. The next step would be downloading more BOXRR users (a team decision).
- **users_100_half ≈ control while users_50 is clearly worse:** user diversity matters more than window volume.
- **Both flat:** the curve has levelled off at 100 users, so a download isn't worth the time.

## Result (31 Oct 2026, seed 42, full validation, paired against the control's 4.2471)

| Variant | Error | Δ | Users better | CI |
| --- | ---: | ---: | :---: | --- |
| users_25 | 4.3272 | +0.080 | 0/15 | [+0.039, +0.137] |
| users_50 | 4.2763 | +0.029 | 2/15 | [+0.009, +0.051] |
| users_100_half | 4.2683 | +0.021 | 4/15 | [+0.006, +0.040] |

- **Still data-limited, with diminishing returns.** Doubling the users gained 0.051 (25 → 50), then 0.029 (50 → 100), a ratio of 0.57 per doubling. That projects about −0.017 at 200 users (one seed, so rough).
- **Diversity or volume: not resolved.** users_100_half against users_50 is −0.008, with 10/15 users better and CI [−0.023, +0.009]. It also has 13% more windows. Halving the users and halving the windows cost about the same.
- Downloading more BOXRR users is the team's call (`jacob/ablations/README.md`, experiment 22).
- Analysis: `jacob/ablations/analyses/paired_tests/v10_users_paired.*` and `v10_users_scaling.*`.
