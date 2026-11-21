"""Convert the team's HDF5 (ray/ group-a-v2 export) into our per-player .npz format, so train.py reads it unchanged.

  uv run --with h5py --with numpy python h5_to_npz.py ../../data/qualified_head_100hz_20261026_131953.h5 ../data/group_a_v2

Writes OUT/<player>.npz (q float32 [N,4] w,x,y,z, offsets [S+1]: segment s is q[offsets[s]:offsets[s+1]]) for every player,
plus OUT/{train,val,test}_players.txt from the file's own /train, /validation and /test groups.
Only segments with challenge_filter_pass are kept, the same ones the team benchmark scores.
"""
import os
import sys

import h5py
import numpy as np

SPLITS = {"train": "train", "validation": "val", "test": "test"}     # HDF5 group -> our list name


def main(src, out):
    os.makedirs(out, exist_ok=True)
    f = h5py.File(src, "r")
    assert f.attrs["quaternion_order"] == "wxyz" and int(f.attrs["rate_hz"]) == 100
    for group, name in SPLITS.items():
        players = sorted(f[group])
        for p in players:
            segs = [f[group][p][rec][seg] for rec in sorted(f[group][p]) for seg in sorted(f[group][p][rec])]
            qs = [s["quaternion_wxyz"][:].astype(np.float32) for s in segs if s.attrs["challenge_filter_pass"]]
            offsets = np.cumsum([0] + [len(q) for q in qs])
            np.savez(os.path.join(out, f"{p}.npz"), q=np.concatenate(qs), offsets=offsets)
        open(os.path.join(out, f"{name}_players.txt"), "w").write("\n".join(players) + "\n")
        print(f"{name}: {len(players)} players")


if __name__ == "__main__":
    main(*sys.argv[1:3])
