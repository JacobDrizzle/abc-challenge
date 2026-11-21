import json, subprocess, glob, os, statistics as st
B = "bench/results/group-a-v2-test20/"
rows = []
def load(name, remote):
    if remote:
        txt = subprocess.run(["git","show",f"origin/main:{B}{name}"],capture_output=True,text=True,check=True).stdout
        return json.loads(txt)
    return json.load(open(B+name))
nic = [f"nic_ga2_{m}_s{s}@nic-mac.json" for m in ("gru","gadst") for s in range(42,47)]
mine = [os.path.basename(p) for p in glob.glob(B+"v7_*angular*@jacob-amd.json")+glob.glob(B+"v6_control_v7graph*@jacob-amd.json")]
defs, data = set(), set()
for n in nic+mine:
    d = load(n, n in nic)
    defs.add(d["bench"]["definition_sha256"]); data.add(d["inputs"]["data_sha256"])
    a, e = d["accuracy"], d["efficiency"]
    rows.append((d["model"]["name"], a["mean_deg_step20"], a["mean_deg_step10"], a["p99_deg_step20"],
                 d["model"].get("declared_trainable_parameters"), e.get("macs_per_call", e.get("macs")), d["contamination"]["status"]))
print("bench defs:", len(defs), " data files:", len(data))
for r in sorted(rows, key=lambda r: r[1]):
    print(f"{r[0]:42s} s20 {r[1]:.4f}  s10 {r[2]:.4f}  p99 {r[3]:.3f}  params {r[4]}  macs {r[5]}  {r[6]}")
for g in ("gru","gadst"):
    v=[r[1] for r in rows if r[0].startswith(f"nic_ga2_{g}_")]
    print(f"nic_ga2_{g}: mean {st.mean(v):.4f}  sd {st.stdev(v):.4f}  best {min(v):.4f}")
