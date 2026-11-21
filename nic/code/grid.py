"""Train every input set with every seed on the SAME test players, one run after another.

  python grid.py --data /data/abc/snapshot_fri --out /runs/fri_grid --test-players-file fixed_test_players.txt

Each run goes to <out>/<inputs>_s<seed>/ (inputs 'vel+acc' -> folder 'vel+acc_s42'). grid_runs.json is rewritten after every
run, and a run whose metrics.json already exists is reused, so a stopped grid can simply be started again.
Everything except the input columns and the seed is a train.py default. The inputs and seeds are the team's choice;
these are the ones Nic asked for: base, each extra alone, then combinations, seeds 42-46.
"""
import argparse
import json
import os
import shlex
import time
import traceback

import train

INPUTS = ["base", "vel", "acc", "speed", "mag",
          "vel+acc", "vel+speed", "vel+mag", "acc+speed", "acc+mag",
          "vel+acc+speed", "vel+acc+mag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-players-file", default=None)
    ap.add_argument("--inputs", nargs="*", default=INPUTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-windows", action="store_true")
    ap.add_argument("--tag", default="", help="prefix for run folder names, e.g. gads_")
    ap.add_argument("--extra", default="", help='more train.py flags in one quoted string, e.g. --extra "--model gads --groups 10"')
    g = ap.parse_args()
    os.makedirs(g.out, exist_ok=True)
    users = train.load_users(g.data, g.max_users)
    rows, total, t_all = [], len(g.inputs) * len(g.seeds), time.time()
    for feats in g.inputs:
        for seed in g.seeds:
            name = f"{g.tag}{feats}_s{seed}"
            folder = os.path.join(g.out, name)
            row = {"name": name, "features": feats, "seed": seed, "tag": g.tag}
            t0 = time.time()
            print(f"\n===== [{len(rows) + 1}/{total}] {name}", flush=True)
            try:
                if not os.path.exists(os.path.join(folder, "metrics.json")):
                    argv = ["--out", folder, "--steps", str(g.steps), "--features", feats, "--seed", str(seed), "--eval-every", str(g.eval_every), *shlex.split(g.extra)]
                    if g.test_players_file:
                        argv += ["--test-players-file", g.test_players_file]
                    if g.no_plots:
                        argv.append("--no-plots")
                    if g.no_windows:
                        argv.append("--no-windows")
                    a = train.parse(argv)
                    a.data = g.data
                    train.run(a, users)
                s = json.load(open(os.path.join(folder, "metrics.json")))["summary"]
                m = s["methods"]
                row.update(model=m["model"]["mean_step20"], hold=m["hold"]["mean_step20"], damped_cv=m["damped_cv"]["mean_step20"],
                           const_vel=m["const_vel"]["mean_step20"], gap=s["gap_test_minus_train"], steps_run=s["steps_run"],
                           params=s["parameters"], players=s["players"])
            except Exception:
                traceback.print_exc()
                row["error"] = True
            row["minutes"] = round((time.time() - t0) / 60, 1)
            rows.append(row)
            json.dump(rows, open(os.path.join(g.out, "grid_runs.json"), "w"), indent=1)
            done = [r for r in rows if "error" not in r]
            if done:
                print(f"done {len(rows)}/{total} | {(time.time() - t_all) / 60:.0f} min so far", flush=True)
    print("\nall done", flush=True)


if __name__ == "__main__":
    main()
