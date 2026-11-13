"""E12: planning portfolio queries on the shared view (Aalborg, eps = 5%).

After the advisor sessions, three planners ask six portfolio queries each (1,206 terms).  Plans:
  term:  every term planned as a single query (each term within the tolerance);
  sum:   one error budget for the SUM, calibrated term bounds added (worst case);
  rss:   one error budget for the SUM, calibrated term bounds combined in quadrature, as approximate
         query processing combines independent per-item errors.
For each plan: latency speedup over simulating every term, mean and max error of the sums, share of
sums outside the tolerance, and violation rate of the terms against their own bounds (from audits)."""
_TWINDB_ROOT = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), *[".."]*1))
import copy, json, os, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, __import__("os").path.join(_TWINDB_ROOT, "twins"))
from twindb2 import TwinDB
import experiments as E
import aal_exp3 as X

HERE = os.path.dirname(os.path.abspath(__file__))


def one(sd, eps=0.05):
    gt = X.GTA(); S = X.WL["sessions"]; D = X.WL["district"]
    fs_d = sum(gt.get(p)[1] for d in D for p in d["points"])
    db = TwinDB(7, gt.engine(), eps=eps, seed=sd, predictor="gp")
    X.sessions_rows(gt, db, S)
    out = {}
    for mode in ("term", "sum", "rss", "cov"):
        d2 = copy.deepcopy(db)
        blocks = []
        for d in D:
            pts = [np.array(p) for p in d["points"]]
            qs = [d2.query(p) for p in pts] if mode == "term" else d2.query_sum(pts, combine=mode)
            blocks.append([X.row(q, gt.get(p)[0]) for q, p in zip(qs, d["points"])])
        sm = X.planner_summary(blocks, eps, fs_d)
        sm["certificate"] = d2.acc.certificate(); sm["accepted_total"] = d2.acc.N
        out[mode] = sm
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(one, range(4)))
    agg = {m: {k: float(np.mean([r[m][k] for r in res])) for k in res[0][m]} for m in res[0]}
    for m in agg:
        agg[m]["sum_viol_any_seed"] = float(max(r[m]["sum_viol"] for r in res))
        agg[m]["sum_err_max_any_seed"] = float(max(r[m]["sum_err_max"] for r in res))
    json.dump({"per_seed": res, "mean": agg, "seeds": 4, "eps": 0.05}, open(os.path.join(HERE, "results_aalborg_planner.json"), "w"), indent=1)
    for m, v in agg.items():
        print(m, round(v["speedup_latency"], 2), "sum err", round(100 * v["sum_err_mean"], 2), "max", round(100 * v["sum_err_max_any_seed"], 2),
              "sums off", round(100 * v["sum_viol_any_seed"], 1), "terms maint", round(100 * v["terms_maintained_frac"], 0), "cert", round(v["certificate"], 2))
