"""E11 (v3): Aalborg portfolio study on all 67 calibrated houses.

Advisor sessions (one per house, 10 queries) followed by three district planners (6 portfolio queries
each, SUM of annual heat over all 67 houses).  Methods for advisor sessions:
  Full-Sim, exact cache, NN library, offline GP (60 LHS runs), online GP (z in {2,3,5}),
  TwinDB with per-house views (5-d view over the scenario only), TwinDB with the portfolio view
  (kernel and first-order operators), audited GP posterior mean on the portfolio view.
Portfolio queries are answered on the same view either term by term (each term within the tolerance)
or with one error budget for the SUM (Section 4: budgeted SUM maintenance).
A second replay interleaves four advisors working concurrently on the shared view.
"""
_TWINDB_ROOT = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), *[".."]*1))
import json, os, sys, numpy as np
sys.path.insert(0, __import__("os").path.join(_TWINDB_ROOT, "twins"))
from twindb2 import TwinDB
import experiments as E
import aalborg_engine as A

HERE = os.path.dirname(os.path.abspath(__file__))
GTD = os.path.join(HERE, "gt_aal")
WL = json.load(open(os.path.join(HERE, "workload_aal_v3.json")))


class GTA:
    def __init__(self):
        self.idx = {}
        for line in open(os.path.join(GTD, "index.jsonl")):
            r = json.loads(line); self.idx[r["sig"]] = r
        self.cache = {}

    def get(self, x, fid=None, variant="v0"):
        x = [float(v) for v in (x if len(x) == 7 else x)]
        s = A.signature(x)
        if s not in self.cache:
            f = os.path.join(GTD, s + ".npy")   # full series if present; else the recorded aggregate (identical)
            agg = float(np.load(f).sum()) if os.path.exists(f) else float(self.idx[s]["agg"])
            self.cache[s] = (agg, float(self.idx[s]["elapsed"]))
        return self.cache[s]

    def engine(self, fid=None, variant="v0"):
        return lambda x: self.get(x)


def row(q, truth):
    return (q.plan, q.answer, truth, q.sim_seconds, q.audit_seconds, q.overhead_s, q.audited)


def sessions_rows(gt, db, sessions, order=None):
    rows = []
    seq = [(i, j) for i, s in enumerate(sessions) for j in range(len(s["points"]))] if order is None else order
    for i, j in seq:
        x = sessions[i]["points"][j]
        rows.append(row(db.query(np.array(x)), gt.get(x)[0]))
    return rows


def planner_rows(gt, db, district, mode):
    out = []
    for d in district:
        pts = [np.array(p) for p in d["points"]]
        if mode == "budget":
            qs = db.query_sum(pts)
        else:
            qs = [db.query(p) for p in pts]
        out.append([row(q, gt.get(p)[0]) for q, p in zip(qs, d["points"])])
    return out


def planner_summary(blocks, eps, full_s):
    flat = [r for b in blocks for r in b]
    errs = []
    for b in blocks:
        a = sum(r[1] for r in b); t = sum(r[2] for r in b)
        errs.append(abs(a - t) / abs(t))
    sm = E.summarize(flat, eps, full_s)
    sm.update({"sum_err_mean": float(np.mean(errs)), "sum_err_max": float(np.max(errs)),
               "sum_viol": float(np.mean([e > eps for e in errs])), "terms_maintained_frac": sm["ism"] / sm["n"]})
    return sm


def avg(reps):
    out = {k: float(np.mean([q[k] for q in reps])) for k in reps[0]}
    out["viol_std"] = float(np.std([q["viol_rate_accepted"] for q in reps]))
    out["speedup_std"] = float(np.std([q["speedup_latency"] for q in reps]))
    return out


def main():
    gt = GTA()
    S = WL["sessions"]; D = WL["district"]
    sess = [p for s in S for p in s["points"]]
    fs_s = sum(gt.get(x)[1] for x in sess)
    fs_d = sum(gt.get(p)[1] for d in D for p in d["points"])
    out = {"n_houses": len(S), "n_session_queries": len(sess), "n_district_queries": len(D),
           "n_terms": sum(len(d["points"]) for d in D), "C_sim_mean_s": fs_s / len(sess), "eps": {}}
    for eps in (0.05, 0.02):
        M = {}
        M["Full-Sim"] = {"sessions": E.summarize([("sim", gt.get(x)[0], gt.get(x)[0], gt.get(x)[1], 0, 0, False) for x in sess], eps, fs_s)}
        db = TwinDB(7, gt.engine(), eps=eps, accept_rule="never", predictor="nn")
        M["Exact cache"] = {"sessions": E.summarize(sessions_rows(gt, db, S), eps, fs_s)}
        M["NN library (r=0.05)"] = {"sessions": E.summarize(E.run_nn_library(gt, [np.array(x) for x in sess], eps, radius=0.05), eps, fs_s)}
        lhs = WL["lhs"]
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        k = ConstantKernel(1.0) * RBF(np.full(7, 0.5), (1e-2, 1e2)) + WhiteKernel(1e-6, (1e-10, 1e-1))
        g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(np.array(lhs), [gt.get(x)[0] for x in lhs])
        tr = sum(gt.get(x)[1] for x in lhs)
        pr = g.predict(np.array(sess))
        blocks = [[("ism", float(v), gt.get(p)[0], 0, 0, 0, False) for v, p in zip(g.predict(np.array(d["points"])), d["points"])] for d in D]
        M["Offline GP (n=60)"] = {"sessions": E.summarize([("ism", float(p), gt.get(x)[0], 0, 0, 0, False) for p, x in zip(pr, sess)], eps, fs_s, extra_seconds=tr),
                                  "planner": planner_summary(blocks, eps, fs_d)}
        M["Offline GP (n=60)"]["planner"]["speedup_latency_with_training"] = fs_d / tr
        for z in (2.0, 3.0, 5.0):
            M["Online GP (z=%g)" % z] = {"sessions": E.summarize(E.run_online_gp(gt, [np.array(x) for x in sess], eps, z=z), eps, fs_s)}
        # per-house views over the 5 scenario parameters
        reps = []
        for sd in range(5):
            rows = []
            for s in S:
                db = TwinDB(5, lambda x5, e=s["points"][0][:2]: gt.get(list(e) + list(x5)), eps=eps, seed=sd)
                for x in s["points"]:
                    rows.append(row(db.query(np.array(x[2:])), gt.get(x)[0]))
            reps.append(E.summarize(rows, eps, fs_s))
        M["TwinDB, per-house views"] = {"sessions": avg(reps)}
        for name, pred in (("TwinDB, portfolio view, kernel", "gp"), ("TwinDB, portfolio view, first-order", "local"),
                           ("Audited GP, portfolio view", "gpmean")):
            rs, pt, cert = [], [], []
            for sd in range(5 if eps == 0.05 else 3):
                db = TwinDB(7, gt.engine(), eps=eps, seed=sd, predictor=pred)
                rs.append(E.summarize(sessions_rows(gt, db, S), eps, fs_s))
                pt.append(planner_summary(planner_rows(gt, db, D, "term"), eps, fs_d))
                cert.append(db.acc.certificate())
            M[name] = {"sessions": avg(rs), "planner_term": avg(pt), "certificate_end": float(np.mean(cert))}
        out["eps"]["%g" % eps] = M
        print(eps, {k: (round(v["sessions"]["speedup_latency"], 2), round(100 * v["sessions"]["viol_rate_accepted"], 1)) for k, v in M.items()}, flush=True)
        for k, v in M.items():
            if "planner_term" in v:
                print("   ", k, "planner", round(v["planner_term"]["speedup_latency"], 2), round(100 * v["planner_term"]["sum_err_mean"], 2),
                      round(100 * v["planner_term"]["sum_viol"], 1), flush=True)
    # four advisors working concurrently: sessions interleaved round-robin in groups of four
    eps = 0.05
    order = []
    for g0 in range(0, len(S), 4):
        grp = list(range(g0, min(g0 + 4, len(S))))
        for j in range(10):
            order += [(i, j) for i in grp]
    reps = []
    for sd in range(5):
        db = TwinDB(7, gt.engine(), eps=eps, seed=sd, predictor="gp")
        rows = sessions_rows(gt, db, S, order)
        reps.append(E.summarize(rows, eps, fs_s))
    out["concurrent4"] = avg(reps)
    # speedup by view age (sessions grouped by position)
    db = TwinDB(7, gt.engine(), eps=eps, seed=0, predictor="gp")
    rows = sessions_rows(gt, db, S)
    age = []
    for i in range(len(S)):
        blk = rows[10 * i:10 * (i + 1)]
        age.append({"frac_answered": sum(1 for r in blk if r[0] in ("ism", "hit")) / 10,
                    "full_s": sum(gt.get(x)[1] for x in S[i]["points"]), "lat_s": sum(r[3] + r[5] for r in blk)})
    out["age_portfolio"] = age
    rows = []
    for s in S:
        db = TwinDB(5, lambda x5, e=s["points"][0][:2]: gt.get(list(e) + list(x5)), eps=eps, seed=0)
        blk = [row(db.query(np.array(x[2:])), gt.get(x)[0]) for x in s["points"]]
        rows.append(sum(1 for r in blk if r[0] in ("ism", "hit")) / 10)
    out["age_perhouse"] = rows
    out["concurrent_note"] = "groups of four sessions interleaved round-robin"
    json.dump(out, open(os.path.join(HERE, "results_aalborg_v3.json"), "w"), indent=1)
    print("concurrent4", round(out["concurrent4"]["speedup_latency"], 2), round(100 * out["concurrent4"]["viol_rate_accepted"], 1))


if __name__ == "__main__":
    main()
