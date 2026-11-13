"""Clean timing of one engine run per time step: 4 scenarios per step, executed one at a time
on an otherwise idle machine (the replay records were taken with two runs in parallel)."""
import json, os, tempfile, time, numpy as np, engines
WL = json.load(open("workload.json"))
FIDS = {"ochre": [60, 15], "sumo": [1.0, 0.5], "epanet": [60, 15, 5]}
out = {}
for e, fids in FIDS.items():
    out[e] = {}
    for f in fids:
        ts = []
        for x in WL[e]["granularity"][:4]:
            p = tempfile.mktemp(suffix=".npy"); t0 = time.perf_counter()
            engines.run(e, x, p, fidelity=f); ts.append(time.perf_counter() - t0); os.remove(p)
        out[e][str(float(f))] = float(np.median(ts))
        print(e, f, out[e], flush=True)
json.dump(out, open("results/e5_timing.json", "w"), indent=1)
