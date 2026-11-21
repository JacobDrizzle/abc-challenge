#!/bin/sh
# Inside the PC container (cwd /work). Retrain the two main models on ONLY the team's group-a-v2 data
# (qualified_head_100hz_20261026_131953.h5, converted by h5_to_npz.py): its 100 /train players train, its 15 /validation
# players decide early stopping, its 20 /test players (the group-a-v2-test20 bench) are scored once. Round-4 settings, seeds 42-46.
D=/data/abc/group_a_v2
OUT=/runs/ga2_round
COMMON="--data $D --out $OUT --test-players-file $D/test_players.txt --seeds 42 43 44 45 46 --no-plots --no-windows"
python -u grid.py $COMMON --inputs vel+noq --tag gruvp30_ --extra "--val-players-file $D/val_players.txt --patience 30"
python -u grid.py $COMMON --inputs vel --tag gads_ --extra "--val-players-file $D/val_players.txt --model gads"
echo GA2_DONE > $OUT/DONE
