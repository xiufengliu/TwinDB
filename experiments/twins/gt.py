"""Workload generation and ground-truth simulation store.

Every query point the evaluation touches is simulated once with the real engine;
the trace and the measured wall-clock cost are kept in a content-addressed store
(gt/<engine>/<signature>.npy + gt/<engine>/index.jsonl). Methods are then replayed
against this store: a method that 'calls the simulator' is charged the measured
cost of that exact call, and its estimate is scored against the true trace.

usage: python gt.py plan            # write workload.json (all query points)
       python gt.py run ENGINE [-j 2]  # simulate every point not yet in the store
       python gt.py pack ENGINE     # consolidate into results/gt_<engine>.npz
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import engines

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt")
WORKLOAD = os.path.join(HERE, "workload.json")
D = 5

N_SESSIONS = {"ochre": 6, "sumo": 8, "epanet": 8}
SESSION_LEN = 50
N_LHS = 60
N_UPDATE = 30            # session queries after a model update
N_GRAN = 20              # queries per alternative fidelity
GRAN_FID = {"ochre": [15], "sumo": [0.5], "epanet": [5, 60]}
UPDATE_VARIANT = {"ochre": "retrofit2", "sumo": "newodpeak", "epanet": "demand_shift"}
N_SESSIONS2 = 4


def session(rng, n=SESSION_LEN):
    """Interactive exploration: sparse local moves, occasional jumps and revisits."""
    x = rng.uniform(0.25, 0.75, D)
    pts = [x.copy()]
    while len(pts) < n:
        u = rng.random()
        if u < 0.04 and len(pts) > 3:                         # exact revisit
            x = pts[rng.integers(len(pts) - 1)].copy()
        elif u < 0.12:                                        # jump to a new region
            x = rng.uniform(0.15, 0.85, D)
        else:                                                 # local sparse move
            k = rng.choice([1, 2, D], p=[0.5, 0.3, 0.2])
            dims = rng.choice(D, size=k, replace=False)
            step = np.exp(rng.uniform(np.log(0.005), np.log(0.12), k)) * rng.choice([-1, 1], k)
            x = np.clip(x.copy(), 0, 1)
            x[dims] = np.clip(x[dims] + step, 0.0, 1.0)
        pts.append(x.copy())
    return np.array(pts)


def lhs(rng, n):
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, D))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(D):
        pts[:, j] = pts[rng.permutation(n), j]
    return 0.1 + 0.8 * pts


def plan():
    wl = {}
    for e in engines.ENGINES:
        rng = np.random.default_rng(20261112 + hash(e) % 1000)
        rng = np.random.default_rng({"ochre": 1, "sumo": 2, "epanet": 3}[e] * 7919)
        sess = [session(rng).tolist() for _ in range(N_SESSIONS[e])]
        lhs_pts = lhs(rng, N_LHS).tolist()
        upd = session(rng, N_UPDATE).tolist()
        upd_lhs = lhs(rng, N_LHS // 2).tolist()
        sess2 = [session(rng).tolist() for _ in range(N_SESSIONS2)]   # second, independent workload draw
        gran = sess[0][:N_GRAN]
        wl[e] = {"sessions": sess, "lhs": lhs_pts, "update_session": upd, "update_lhs": upd_lhs,
                 "update_variant": UPDATE_VARIANT[e], "granularity": gran, "sessions2": sess2, "gran_fid": GRAN_FID[e],
                 "default_fidelity": engines.ENGINES[e]["default_fidelity"]}
    json.dump(wl, open(WORKLOAD, "w"))
    for e, w in wl.items():
        print(e, "sessions", len(w["sessions"]), "pts", sum(len(s) for s in w["sessions"]))


def jobs_for(e):
    w = json.load(open(WORKLOAD))[e]
    fid0 = w["default_fidelity"]
    jobs = []
    for s in w["sessions"]:
        jobs += [(x, fid0, "v0") for x in s]
    jobs += [(x, fid0, "v0") for x in w["lhs"]]
    # model update: the same pre-update session points are needed under the new model too
    jobs += [(x, fid0, w["update_variant"]) for x in w["update_session"]]
    jobs += [(x, fid0, "v0") for x in w["update_session"]]
    jobs += [(x, fid0, w["update_variant"]) for x in w["update_lhs"]]
    for s in w.get("sessions2", []):
        jobs += [(x, fid0, "v0") for x in s]
    for s in w.get("sessions2", []):                 # long post-update horizon (model-update study)
        jobs += [(x, fid0, w["update_variant"]) for x in s]
    for f in w["gran_fid"]:
        jobs += [(x, f, "v0") for x in w["granularity"]]
    seen, out = set(), []
    for x, f, v in jobs:
        sig = engines.signature(e, x, f, v)
        if sig not in seen:
            seen.add(sig)
            out.append((e, x, f, v, sig))
    return out


def _one(job):
    e, x, f, v, sig = job
    d = os.path.join(GT, e)
    out = os.path.join(d, sig + ".npy")
    if os.path.exists(out):
        return None
    try:
        r = engines.run(e, x, out, fidelity=f, variant=v)
        rec = {"sig": sig, "x": x, "fidelity": f, "variant": v, **r, "status": "ok",
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    except Exception as ex:  # failures are kept as evidence
        rec = {"sig": sig, "x": x, "fidelity": f, "variant": v, "status": "failed", "error": str(ex)[-500:]}
    with open(os.path.join(d, "index.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def run(e, j=2):
    os.makedirs(os.path.join(GT, e), exist_ok=True)
    jobs = [jb for jb in jobs_for(e) if not os.path.exists(os.path.join(GT, e, jb[4] + ".npy"))]
    print(e, "pending", len(jobs), flush=True)
    with Pool(j) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            if r and i % 10 == 0:
                print(e, i, r.get("status"), round(r.get("elapsed", 0), 2), flush=True)


def pack(e):
    d = os.path.join(GT, e)
    idx = {}
    for line in open(os.path.join(d, "index.jsonl")):
        r = json.loads(line)
        if r["status"] == "ok":
            idx[r["sig"]] = r
    sigs = sorted(idx)
    traces = {s: np.load(os.path.join(d, s + ".npy")) for s in sigs}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "results", "gt_%s.npz" % e), **traces)
    json.dump(idx, open(os.path.join(HERE, "results", "gt_%s_index.json" % e), "w"))
    print(e, "packed", len(sigs))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    elif cmd == "pack":
        pack(sys.argv[2])
