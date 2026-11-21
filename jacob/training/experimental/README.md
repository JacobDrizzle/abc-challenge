# experimental: the next round

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../../Submission/Motion_Pending_headpose/model_card.md).

Each new experiment is a package here, built on the current best model in `best/`. Up to 29 Oct, the v6, v7 and v9
packages lived here. They are now in `archive/`, and their code is `best/`.

## Pattern

```
experimental/v10_<name>/
  __init__.py
  hooks.py        installed(): your swaps, then best.hooks.installed()
  model.py ...    only what changes
  run.py          a copy of best/run.py that imports your installed()
  configs/        suites whose variants override best/configs/base.json
  test_v10.py
  README.md
```

1. **Hooks.** `best.hooks.installed()` reads `LeanTCNN`, `normalize_config`, `MODEL_KEYS` and `BEST_FILES` from its
   own module when it is entered. Swap them there first, and the frozen v5 scripts train your model:

   ```python
   from contextlib import contextmanager

   from best import hooks as best_hooks
   from best.hooks import _swapped

   FILES = tuple(f"experimental/v10_<name>/{n}.py" for n in ("__init__", "hooks", "model"))

   @contextmanager
   def installed():
       with (_swapped(best_hooks, LeanTCNN=NewTCNN, normalize_config=normalize_config,
                      MODEL_KEYS=best_hooks.MODEL_KEYS + ("new_key",), BEST_FILES=best_hooks.BEST_FILES + FILES),
             best_hooks.installed()):
           yield
   ```

   - This was checked on 30 Oct with a `LeanTCNN` subclass and a new config key on a tiny run. The subclass trained,
     the key reached it, the extra files joined the hashed list, and every name was restored afterwards.
   - `normalize_config` must handle your new keys and pass the rest to `best.configuration.normalize_config`, which
     refuses keys it doesn't know.
   - For a checkpoint to load with a new key, the class's `specification()` must include it, and
     `best.model.load_checkpoint` must build your class. Swap `best.model.LeanTCNN` as well.
   - Add every file that can change a training or validation number to `FILES`, so the run's `experiment.json`
     catches later edits. Don't edit those files while a run is using them.
2. **Configs.** The control is `{}`, so it reproduces k8_combined seed 42. That is 4.2471 on validation, and the
   history should match `outputs/v9_confirm/k8_combined/seed_42/history.json` epoch for epoch.
3. **Run.** Write to `outputs/v10_<name>/`, run one GPU job at a time, and screen on seed 42 first:
   `python -m experimental.v10_<name>.run --stage screen --suite experimental/v10_<name>/configs/<suite>.json --output outputs/v10_<name>`.
4. **Record.**
   - Run `python jacob/ablations/harvest.py`.
   - Add an entry to `jacob/ablations/README.md`: the question, the change, the seeds, the result, Δ against k8_combined, users better, and the verdict.
5. **Promote.** If the team adopts a confirmed winner:
   - Move its code into `best/` and its config into `best/configs/base.json`.
   - Move the package to `archive/` and tag the commit the runs used.
   - Log the promotion in the ledger.

## Baselines to compare with

| Model | Validation, 3 seeds | Selected seed | Params |
| --- | ---: | ---: | ---: |
| k8_combined (`best/configs/base.json`) | 4.2501 ± 0.0041 | 42: 4.2471 | 7,555 |
| k5_combined (overrides in `best/README.md`) | 4.2717 ± 0.0067 | 42: 4.2652 | 5,595 |
