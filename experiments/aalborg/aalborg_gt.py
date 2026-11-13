"""Workload and ground truth for the Aalborg portfolio study.

plan: per calibrated house one energy-advisor session of 10 scenario queries that starts at the
      house's current state (no retrofit, 21 C) and moves like the analyst model of the main study,
      restricted to the 5 scenario parameters; 6 district-planning queries that apply one scenario to
      every house (SUM over the portfolio); a 60-point Latin-hypercube design over the 7-d space for
      the offline surrogate.
run:  simulate every planned point once with the calibrated OCHRE twin, recording trace and time.
"""
import json, os, sys, time, numpy as np
from concurrent.futures import ProcessPoolExecutor
import aalborg_engine as A

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt_aal")
SC0 = A.norm([1, 1, 0.0, 0.0, 1.0, 1.0, 21.0])[2:]     # current state of every house


def plan(n_q=10, seed=7):
    cal = json.load(open(os.path.join(HERE, "calibration.json")))["houses"]
    rng = np.random.default_rng(seed)
    houses = [h for h in cal if not h["at_grid_edge"] and abs(h["nmbe_monthly"]) <= 0.05 and h["cvrmse_monthly"] <= 0.15]   # ASHRAE Guideline 14 (monthly)
    order = rng.permutation(len(houses))
    sessions = []
    for i in order:
        h = houses[i]
        e = A.norm([h["k_env"], h["k_dhw"], 0, 0, 1, 1, 21])[:2]
        s = SC0.copy(); pts = [np.r_[e, s]]
        while len(pts) < n_q:
            u = rng.random()
            if u < 0.04 and len(pts) > 2:            # revisit
                pts.append(pts[rng.integers(len(pts))].copy()); continue
            s = s.copy()
            if u < 0.12:                              # jump to another retrofit package
                s = rng.random(5)
            else:
                k = rng.choice([1, 2, 5], p=[0.5, 0.3, 0.2])
                idx = rng.choice(5, k, replace=False)
                step = np.exp(rng.uniform(np.log(0.005), np.log(0.12), k)) * rng.choice([-1, 1], k)
                s[idx] = np.clip(s[idx] + step, 0, 1)
            pts.append(np.r_[e, s])
        sessions.append({"meter_id": h["meter_id"], "points": [p.tolist() for p in pts]})
    # district planner: a session of 6 portfolio queries; each applies one scenario to every house,
    # starting from a moderate retrofit package and moving locally like the analysts above
    district = []
    s = rng.random(5)
    for qi in range(6):
        if qi > 0:
            s = s.copy()
            k = rng.choice([1, 2], p=[0.6, 0.4])
            idx = rng.choice(5, k, replace=False)
            s[idx] = np.clip(s[idx] + np.exp(rng.uniform(np.log(0.02), np.log(0.15), k)) * rng.choice([-1, 1], k), 0, 1)
        district.append({"scenario": s.tolist(),
                         "points": [np.r_[A.norm([h["k_env"], h["k_dhw"], 0, 0, 1, 1, 21])[:2], s].tolist() for h in houses]})
    from scipy.stats import qmc
    lhs = qmc.LatinHypercube(d=7, seed=seed).random(60).tolist()
    wl = {"houses": [h["meter_id"] for h in houses], "sessions": sessions, "district": district, "lhs": lhs}
    json.dump(wl, open(os.path.join(HERE, "workload_aal.json"), "w"))
    n = sum(len(s["points"]) for s in sessions) + sum(len(d["points"]) for d in district) + len(lhs)
    print("houses", len(houses), "points", n)


def _one(x):
    sig = A.signature(x)
    f = os.path.join(GT, sig + ".npy")
    if os.path.exists(f):
        return None
    r = A.run(A.denorm(x), f)
    return {"sig": sig, "x": list(x), "elapsed": r["elapsed"], "agg": r["agg"], "status": "ok",
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}


def run(j=2):
    os.makedirs(GT, exist_ok=True)
    wl = json.load(open(os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "workload_aal.json")))
    pts = [p for s in wl["sessions"] for p in s["points"]] + [p for d in wl["district"] for p in d["points"]] + wl["lhs"]
    seen, todo = set(), []
    for p in pts:
        s = A.signature(p)
        if s not in seen:
            seen.add(s); todo.append(p)
    idx = open(os.path.join(GT, "index.jsonl"), "a")
    with ProcessPoolExecutor(j) as ex:
        for r in ex.map(_one, todo):
            if r is not None:
                idx.write(json.dumps(r) + "\n"); idx.flush()
    print("unique points", len(todo))


def extend(seed=11):
    """Extend the workload without changing what was simulated: advisor sessions for the houses
    left out by the calibration filter (all 67 houses are then covered), inserted at random positions
    into the advisor order, and two further district-planner sessions of 6 portfolio queries; every
    district query now covers all houses."""
    cal = json.load(open(os.path.join(HERE, "calibration.json")))["houses"]
    wl = json.load(open(os.path.join(HERE, "workload_aal.json")))
    rng = np.random.default_rng(seed)
    have = set(wl["houses"])
    extra = [h for h in cal if h["meter_id"] not in have]
    desc = {h["meter_id"]: A.norm([h["k_env"], h["k_dhw"], 0, 0, 1, 1, 21])[:2] for h in cal}
    sessions = list(wl["sessions"])
    for h in extra:
        e = desc[h["meter_id"]]
        s = SC0.copy(); pts = [np.r_[e, s]]
        while len(pts) < 10:
            u = rng.random()
            if u < 0.04 and len(pts) > 2:
                pts.append(pts[rng.integers(len(pts))].copy()); continue
            s = s.copy()
            if u < 0.12:
                s = rng.random(5)
            else:
                k = rng.choice([1, 2, 5], p=[0.5, 0.3, 0.2])
                idx = rng.choice(5, k, replace=False)
                step = np.exp(rng.uniform(np.log(0.005), np.log(0.12), k)) * rng.choice([-1, 1], k)
                s[idx] = np.clip(s[idx] + step, 0, 1)
            pts.append(np.r_[e, s])
        sessions.insert(int(rng.integers(0, len(sessions) + 1)), {"meter_id": h["meter_id"], "points": [p.tolist() for p in pts]})
    houses = [s_["meter_id"] for s_ in sessions]
    order = sorted(desc)                                  # district queries cover every house
    planners = [[d["scenario"] for d in wl["district"]]]
    for _ in range(2):
        s = rng.random(5); sc = []
        for qi in range(6):
            if qi > 0:
                s = s.copy()
                k = rng.choice([1, 2], p=[0.6, 0.4]); idx = rng.choice(5, k, replace=False)
                s[idx] = np.clip(s[idx] + np.exp(rng.uniform(np.log(0.02), np.log(0.15), k)) * rng.choice([-1, 1], k), 0, 1)
            sc.append(s.tolist())
        planners.append(sc)
    district = [{"planner": p, "scenario": sc, "houses": order, "points": [np.r_[desc[m], sc].tolist() for m in order]}
                for p, scs in enumerate(planners) for sc in scs]
    wl2 = {"houses": houses, "all_houses": order, "g14": [h["meter_id"] for h in cal if not h["at_grid_edge"] and abs(h["nmbe_monthly"]) <= 0.05 and h["cvrmse_monthly"] <= 0.15],
           "sessions": sessions, "district": district, "lhs": wl["lhs"]}
    json.dump(wl2, open(os.path.join(HERE, "workload_aal_v3.json"), "w"))
    print("houses", len(houses), "district queries", len(district))


if __name__ == "__main__":
    {"plan": plan, "run": run, "extend": extend}[sys.argv[1]]()
