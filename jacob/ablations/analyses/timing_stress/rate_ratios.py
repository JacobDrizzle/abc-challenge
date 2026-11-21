"""Each model's step-20 error as a ratio of the damped baseline's on the same stream (rate rows change the task)."""
import json
import sys

results = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
models = [m for m in next(iter(results.values())) if not m.startswith("damped")]
print(f"{'stream':<28}" + "".join(f"{m:>24}" for m in models))
for stream, row in results.items():
    damped = row["damped velocity (baseline)"]
    print(f"{stream:<28}" + "".join(f"{row[m] / damped:>24.4f}" for m in models))
