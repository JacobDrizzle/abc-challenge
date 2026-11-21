# Player splits behind our results

- `test_players.txt`: the 47 test players. The same 47 are the test set for **every** result (both days, all models). Never train on them.
- `splits_fri.csv`: `player_id,split` for the 504 players of the Friday snapshot (409 train, 48 val, 47 test). All numbers in `../results/friday_grid_results.pdf` and the GADS-T tests.

`player_id` = the BOXRR-23 tar file name without `.tar` (the user id), the same unit Ray splits on. Players not in your own download cannot be used.

How the split is made (reproducible): `sha1("0:" + player_id)` read as a number in [0,1): below 0.8 train, below 0.9 val, else test (`windows.user_split`). The 3x-larger Saturday run (1,229 train / 141 val players) used the same 47 test players; its list is not included because that run changed nothing (see `../FINDINGS.md`, F03).
