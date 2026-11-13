"""Experiments for TwinDB.  Methods are replayed against the ground-truth
store (gt.py): every simulator call returns the true aggregate and is charged the
wall-clock time measured when that exact run was executed by the real engine.
System overheads (planning, maintenance, persistence) are measured live.

usage: python experiments.py <exp> [args]   (see EXPERIMENTS at the bottom)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np

import engines
from twindb2 import TwinDB, AuditedAcceptor, ScenarioIndex, Sensitivity, PgStore

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WL = json.load(open(os.path.join(HERE, "workload.json")))
ENG = ["ochre", "sumo", "epanet"]


# ------------------------------------------------------------------ ground truth
class GT:
    def __init__(self, e, include_unbalanced=False):
        self.e = e
        d = os.path.join(HERE, "gt", e)
        self.npz = None
        if not os.path.exists(os.path.join(d, "index.jsonl")):
            # packaged ground truth (repository): gt_replay/<e>_index*.jsonl + gt_replay/<e>_<k>.npz shards
            import glob
            d = os.path.join(HERE, "gt_replay")
            self.npz = {}
            for f in sorted(glob.glob(os.path.join(d, e + "_[0-9]*.npz"))):
                z = np.load(f)
                self.npz.update({k: z for k in z.files})
            self._pfx = e + "_"
        else:
            self._pfx = ""
        self.idx = {}
        for line in open(os.path.join(d, self._pfx + "index.jsonl")):
            r = json.loads(line)
            if r["status"] == "ok" and (include_unbalanced or not r.get("unbalanced", False)):
                self.idx[r["sig"]] = r
        # EPANET scenarios whose hydraulics did not converge (first attempt stopped with an unbalanced
        # system; re-runs only completed with the CONTINUE option) are not valid solutions: excluded
        bad = os.path.join(d, self._pfx + "index.failed_attempt1.jsonl")
        self.unbalanced = set()
        if os.path.exists(bad) and not include_unbalanced:
            for line in open(bad):
                self.unbalanced.add(json.loads(line)["sig"])
            for s in self.unbalanced:
                self.idx.pop(s, None)
        self.dir = d
        self.cache = {}

    def get(self, x, fid=None, variant="v0"):
        sig = engines.signature(self.e, x, fid, variant)
        if sig not in self.cache:
            r = self.idx.get(sig)
            if r is None:
                return None
            tr = self.npz[sig][sig] if self.npz is not None else np.load(os.path.join(self.dir, sig + ".npy"))
            self.cache[sig] = (float(tr.sum()), float(r["elapsed"]), tr)
        return self.cache[sig]

    def engine(self, fid=None, variant="v0", counter=None):
        def call(x):
            g = self.get(x, fid, variant)
            if g is None:
                raise KeyError("no ground truth")
            if counter is not None:
                counter.append(g[1])
            return g[0], g[1]
        return call


def sessions(e, gt, key="sessions", fid=None, variant="v0"):
    """Sessions restricted to points with ground truth (EPANET non-converged points are dropped)."""
    out = []
    for s in WL[e][key] if key != "update_session" else [WL[e][key]]:
        pts = [np.array(x) for x in s if gt.get(x, fid, variant) is not None]
        out.append(pts)
    return out


# ------------------------------------------------------------------ policies
def run_twindb(gt, pts, eps, seed=0, fid=None, variant="v0", **kw):
    db = TwinDB(5, gt.engine(fid, variant), eps=eps, seed=seed, **kw)
    rows = []
    for x in pts:
        r = db.query(x)
        truth = gt.get(x, fid, variant)[0]
        rows.append((r.plan, r.answer, truth, r.sim_seconds, r.audit_seconds, r.overhead_s, r.audited))
    return rows, db


def run_fullsim(gt, pts, fid=None, variant="v0"):
    rows = []
    for x in pts:
        y, el = gt.engine(fid, variant)(x)
        rows.append(("sim", y, y, el, 0.0, 0.0, False))
    return rows


def run_exactcache(gt, pts, eps, fid=None, variant="v0"):
    rows, _ = run_twindb(gt, pts, eps, fid=fid, variant=variant, accept_rule="never", predictor="nn")
    return rows


def run_nn_library(gt, pts, eps, radius=0.05, fid=None, variant="v0"):
    """Scenario library: return the nearest stored scenario if within `radius`, else simulate."""
    idx = ScenarioIndex(len(pts[0]))
    eng = gt.engine(fid, variant)
    rows = []
    for x in pts:
        t0 = time.perf_counter()
        h = idx.exact(x)
        if h is not None:
            rows.append(("hit", h, h, 0.0, 0.0, time.perf_counter() - t0, False)); continue
        i = idx.nearest(x)
        if i is not None and np.linalg.norm(idx.X[i] - x) <= radius:
            yhat = float(idx.y[i]); o = time.perf_counter() - t0
            rows.append(("ism", yhat, gt.get(x, fid, variant)[0], 0.0, 0.0, o, False)); continue
        o = time.perf_counter() - t0
        y, el = eng(x)
        idx.insert(x, y)
        rows.append(("sim", y, y, el, 0.0, o, False))
    return rows


def _gp(d=5):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    k = ConstantKernel(1.0) * RBF(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2)) + \
        WhiteKernel(1e-6, (1e-10, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0)


def run_offline_gp(gt, pts, train_pts, fid=None, variant="v0"):
    """DT practice: pre-train a surrogate on a space-filling design, then answer every query from it."""
    eng = gt.engine(fid, variant)
    X = np.array(train_pts); y = []; train_s = 0.0
    for x in X:
        v, el = eng(x); y.append(v); train_s += el
    y = np.array(y)
    t0 = time.perf_counter(); gp = _gp().fit(X, y); fit_s = time.perf_counter() - t0
    rows = []
    for x in pts:
        t0 = time.perf_counter(); yhat = float(gp.predict(x[None])[0]); o = time.perf_counter() - t0
        rows.append(("ism", yhat, gt.get(x, fid, variant)[0], 0.0, 0.0, o, False))
    return rows, train_s, fit_s


def run_online_gp(gt, pts, eps, z=2.0, cold=6, fid=None, variant="v0", refit_every=1):
    """Uncertainty-aware surrogate (active-learning style): refit a GP on all simulated
    scenarios, accept when z * predictive std <= eps |mu| / (1+eps), else simulate."""
    eng = gt.engine(fid, variant)
    X, Y, rows, gp, k = [], [], [], None, 0
    for x in pts:
        t0 = time.perf_counter()
        if len(X) >= cold:
            if gp is None or k >= refit_every:
                gp = _gp(len(X[0])).fit(np.array(X), np.array(Y)); k = 0
            mu, sd = gp.predict(x[None], return_std=True)
            mu, sd = float(mu[0]), float(sd[0])
            if z * sd <= eps * abs(mu) / (1 + eps):
                rows.append(("ism", mu, gt.get(x, fid, variant)[0], 0.0, 0.0, time.perf_counter() - t0, False))
                continue
        o = time.perf_counter() - t0
        y, el = eng(x); X.append(x); Y.append(y); k += 1
        rows.append(("sim", y, y, el, 0.0, o, False))
    return rows


def run_fd_jacobian(gt, pts, eps, h=0.02, fid=None, variant="v0"):
    """Gradient baseline: finite-difference Jacobian at the first query (d+1 runs), then
    first-order extrapolation from that base for every later query (no acceptance test)."""
    eng = gt_engine_any(gt, fid, variant)
    x0 = pts[0]; y0, el0 = eng(x0); J = np.zeros(5); cost = el0
    for j in range(5):
        xp = x0.copy(); xp[j] = min(1.0, xp[j] + h)
        yj, el = eng(xp); J[j] = (yj - y0) / (xp[j] - x0[j]); cost += el
    rows = [("sim", y0, y0, cost, 0.0, 0.0, False)]
    for x in pts[1:]:
        yhat = y0 + J @ (x - x0)
        rows.append(("ism", float(yhat), gt.get(x, fid, variant)[0], 0.0, 0.0, 0.0, False))
    return rows


def gt_engine_any(gt, fid, variant):
    """FD probes are not in the ground-truth store; they are executed live by the real engine and cached."""
    base = gt.engine(fid, variant)
    store = os.path.join(HERE, "gt", gt.e)

    def call(x):
        g = gt.get(x, fid, variant)
        if g is not None:
            return g[0], g[1]
        sig = engines.signature(gt.e, x, fid, variant)
        out = os.path.join(store, sig + ".npy")
        r = engines.run(gt.e, x, out, fidelity=fid, variant=variant)
        rec = {"sig": sig, "x": [float(v) for v in x], "fidelity": fid if fid is not None else engines.ENGINES[gt.e]["default_fidelity"],
               "variant": variant, **r, "status": "ok", "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": "fd-probe"}
        with open(os.path.join(store, "index.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        gt.idx[sig] = rec
        return gt.get(x, fid, variant)[0], r["elapsed"]
    return call


# ------------------------------------------------------------------ metrics
def summarize(rows, eps, full_seconds, extra_seconds=0.0):
    plans = [r[0] for r in rows]
    approx = [(r[1], r[2]) for r in rows if r[0] == "ism"]
    rel = [abs(a - t) / max(abs(t), 1e-12) for a, t in approx]
    viol = [abs(a - t) > eps * abs(t) for a, t in approx]
    crit = sum(r[3] for r in rows) + sum(r[5] for r in rows) + extra_seconds
    audit = sum(r[4] for r in rows)
    nsim = sum(1 for r in rows if r[0] == "sim") + sum(1 for r in rows if r[6])
    return {
        "n": len(rows), "hit": plans.count("hit"), "ism": plans.count("ism"), "sim": plans.count("sim"),
        "audits": int(sum(1 for r in rows if r[6])), "sim_calls": nsim,
        "latency_s": crit, "audit_s": audit, "total_compute_s": crit + audit,
        "speedup_latency": full_seconds / max(crit, 1e-9),
        "speedup_compute": full_seconds / max(crit + audit, 1e-9),
        "mean_rel_err": float(np.mean(rel)) if rel else 0.0,
        "p95_rel_err": float(np.percentile(rel, 95)) if rel else 0.0,
        "max_rel_err": float(np.max(rel)) if rel else 0.0,
        "viol_rate_accepted": float(np.mean(viol)) if viol else 0.0,
        "overhead_ms_per_query": 1000 * sum(r[5] for r in rows) / len(rows),
    }


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, name + ".json")
    json.dump(obj, open(p, "w"), indent=1, default=float)
    print("wrote", p)


# ------------------------------------------------------------------ E1 main comparison
def e1_main(eps=0.02, rho=0.1, engines_=None):
    """Multi-session workload per engine: sessions arrive in order and the scenario view
    (anchors, cache, learned state) persists across sessions, as it does in the database."""
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e)
        ss = sessions(e, gt)
        stream = [x for s in ss for x in s]
        bounds = np.cumsum([0] + [len(s) for s in ss])
        lhs = [np.array(x) for x in WL[e]["lhs"] if gt.get(x) is not None]
        full = run_fullsim(gt, stream)
        fs = sum(r[3] for r in full)
        m = {"Full-Sim": summarize(full, eps, fs),
             "Exact cache": summarize(run_exactcache(gt, stream, eps), eps, fs)}
        for rad in (0.02, 0.05, 0.1):
            m["NN library (r=%g)" % rad] = summarize(run_nn_library(gt, stream, eps, rad), eps, fs)
        for n in (30, 60):
            rows, tr, fit = run_offline_gp(gt, stream, lhs[:n])
            m["Offline GP (n=%d)" % n] = summarize(rows, eps, fs, extra_seconds=tr + fit)
        m["Online GP (2 sigma)"] = summarize(run_online_gp(gt, stream, eps), eps, fs)
        fd = []
        for s in ss:
            fd += run_fd_jacobian(gt, s, eps)
        m["FD Jacobian (per session)"] = summarize(fd, eps, fs)
        curves = {}
        for name, pred in (("TwinDB-RLS (global first-order)", "rls"), ("TwinDB-ISM (first-order)", "local"), ("TwinDB (kernel)", "gp")):
            reps, per = [], []
            for sd in range(3 if pred == "rls" else 10):
                rows, db = run_twindb(gt, stream, eps, seed=sd, rho=rho, predictor=pred)
                sm = summarize(rows, eps, fs); sm["certificate"] = db.acc.certificate(); reps.append(sm)
                per.append([summarize(rows[bounds[i]:bounds[i + 1]], eps,
                                      sum(r[3] for r in full[bounds[i]:bounds[i + 1]])) for i in range(len(ss))])
            m[name] = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
            m[name]["viol_std"] = float(np.std([r["viol_rate_accepted"] for r in reps]))
            m[name]["speedup_std"] = float(np.std([r["speedup_latency"] for r in reps]))
            m[name]["n_seeds"] = len(reps)
            curves[name] = [{k: float(np.mean([p[i][k] for p in per])) for k in per[0][i]} for i in range(len(ss))]
        # ablation of the view layer: same operator and audited threshold, but a fresh view per session
        # (no anchors shared across analysts' sessions, nothing persisted between them)
        reps = []
        for sd in range(10):
            rows = []
            for s_ in ss:
                r_, _ = run_twindb(gt, s_, eps, seed=sd, rho=rho, predictor="gp")
                rows += r_
            reps.append(summarize(rows, eps, fs))
        m["TwinDB (kernel), view per session"] = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        m["TwinDB (kernel), view per session"]["viol_std"] = float(np.std([r["viol_rate_accepted"] for r in reps]))
        if e == "epanet":   # sensitivity: keep the scenarios whose hydraulics did not converge
            gu = GT(e, include_unbalanced=True)
            su = [np.array(x) for s in WL[e]["sessions"] for x in s if gu.get(x) is not None]
            fu = sum(gu.get(x)[1] for x in su)
            rows, db = run_twindb(gu, su, eps, seed=0, rho=rho, predictor="gp")
            m["TwinDB (kernel), unbalanced kept"] = summarize(rows, eps, fu)
            lu = [np.array(x) for x in WL[e]["lhs"] if gu.get(x) is not None]
            rows, tr, fit = run_offline_gp(gu, su, lu[:30])
            m["Offline GP (n=30), unbalanced kept"] = summarize(rows, eps, fu, extra_seconds=tr + fit)
        out[e] = {"methods": m, "per_session": curves, "n_queries": len(stream), "n_sessions": len(ss),
                  "C_sim_mean_s": fs / len(stream)}
        print(e, {k: (round(v["speedup_latency"], 2), round(v["viol_rate_accepted"], 3)) for k, v in m.items()}, flush=True)
    save("e1_main_" + "_".join(engines_ or ENG), out)


# ------------------------------------------------------------------ E2 guarantee validity + ablations
def taylor_gauss_policy(gt, pts, eps, z=1.96):
    """Predicted-bound acceptance: accept if z*sqrt(dx' P dx)*scale <= eps (Gaussian RLS term)."""
    eng = gt.engine()
    idx, sens, rows = ScenarioIndex(5), Sensitivity(5), []
    for x in pts:
        h = idx.exact(x)
        if h is not None:
            rows.append(("hit", h, h, 0, 0, 0, False)); continue
        if len(idx.y) >= 6:
            i = idx.nearest(x); b, yb = idx.X[i], idx.y[i]; dx = x - b
            yhat = yb + sens.beta @ dx
            # RLS covariance scaled by residual variance (as in the earlier prototype)
            sd = math.sqrt(max(float(dx @ sens.P[1:, 1:] @ dx), 0.0))
            resid = getattr(sens, "_s2", 0.0)
            if z * sd * math.sqrt(max(resid, 1e-12)) <= eps * abs(yhat) / (1 + eps):
                rows.append(("ism", float(yhat), gt.get(x)[0], 0, 0, 0, False)); continue
        y, el = eng(x)
        if len(idx.y) >= 1:
            i = idx.nearest(x); pred = idx.y[i] + sens.beta @ (x - idx.X[i])
            sens._s2 = 0.9 * getattr(sens, "_s2", (y - pred) ** 2) + 0.1 * (y - pred) ** 2
        idx.insert(x, y); sens.update(x, y)
        rows.append(("sim", y, y, el, 0, 0, False))
    return rows


def kappa_policy(gt, pts, eps, warm=10):
    """Predicted bound: kappa = 95th pct of |err|/||dx|| over warmup probes, fixed afterwards."""
    eng = gt.engine()
    idx, sens, rows, ratios, kappa = ScenarioIndex(5), Sensitivity(5), [], [], None
    for x in pts:
        h = idx.exact(x)
        if h is not None:
            rows.append(("hit", h, h, 0, 0, 0, False)); continue
        if kappa is not None:
            i = idx.nearest(x); dx = x - idx.X[i]; yhat = idx.y[i] + sens.beta @ dx
            if kappa * np.linalg.norm(dx) <= eps * abs(yhat) / (1 + eps):
                rows.append(("ism", float(yhat), gt.get(x)[0], 0, 0, 0, False)); continue
        y, el = eng(x)
        if len(idx.y) >= 6:
            i = idx.nearest(x); dx = x - idx.X[i]
            ratios.append(abs(y - (idx.y[i] + sens.beta @ dx)) / max(np.linalg.norm(dx), 1e-3))
            if len(ratios) == warm:
                kappa = float(np.percentile(ratios, 95))
        idx.insert(x, y); sens.update(x, y)
        rows.append(("sim", y, y, el, 0, 0, False))
    return rows


def e2_guarantee(eps=0.02, rho=0.1, reps=20, engines_=None):
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e); ss = sessions(e, gt)
        # one long stream per engine: sessions concatenated (exploration of several regions)
        stream = [x for s in ss for x in s]
        fs = sum(gt.get(x)[1] for x in stream)
        r = {"twindb": [], "twindb_ism": [], "no_audit_all_observed": [], "taylor_gauss": None, "kappa_fixed": None}
        for s in range(min(reps, 10)):
            rows, db = run_twindb(gt, stream, eps, seed=s, rho=rho, predictor="local")
            sm = summarize(rows, eps, fs); sm["certificate"] = db.acc.certificate(); sm["N_acc"] = db.acc.N
            sm["cert_ok"] = bool(sm["viol_rate_accepted"] <= sm["certificate"]); r["twindb_ism"].append(sm)
        r["online_gp_2sigma"] = summarize(run_online_gp(gt, stream, eps), eps, fs)
        r["online_gp_zgrid"] = {str(z): summarize(run_online_gp(gt, stream, eps, z=z), eps, fs) for z in (0.5, 1, 1.5, 2.5, 3, 4, 5)}
        for s in range(reps):
            rows, db = run_twindb(gt, stream, eps, seed=s, rho=rho)
            sm = summarize(rows, eps, fs)
            sm["certificate"] = db.acc.certificate(); sm["N_acc"] = db.acc.N
            sm["cert_ok"] = bool(sm["viol_rate_accepted"] <= sm["certificate"])
            r["twindb"].append(sm)
        # ablation: rho=0 and threshold driven by rejected (always observed) queries
        rows, db = run_twindb_allobs(gt, stream, eps)
        r["no_audit_all_observed"] = summarize(rows, eps, fs)
        r["taylor_gauss"] = summarize(taylor_gauss_policy(gt, stream, eps), eps, fs)
        r["kappa_fixed"] = summarize(kappa_policy(gt, stream, eps), eps, fs)
        out[e] = r
        tw = r["twindb"]
        print(e, "viol", np.mean([t["viol_rate_accepted"] for t in tw]).round(4), "cert", np.mean([t["certificate"] for t in tw]).round(3),
              "ok", np.mean([t["cert_ok"] for t in tw]), "allobs", round(r["no_audit_all_observed"]["viol_rate_accepted"], 3),
              "taylor", round(r["taylor_gauss"]["viol_rate_accepted"], 3), "kappa", round(r["kappa_fixed"]["viol_rate_accepted"], 3), flush=True)
    save("e2_guarantee_" + "_".join(engines_ or ENG), out)


def run_twindb_allobs(gt, pts, eps):
    """Ablation: no audits (rho=0); the threshold is then calibrated only on simulated (rejected) queries."""
    return run_twindb(gt, pts, eps, rho=0.0)


# ------------------------------------------------------------------ E3 tolerance and audit-rate sweep
def e3_sweep(reps=2, engines_=None):
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e); ss = sessions(e, gt); stream = [x for s in ss for x in s]
        fs = sum(gt.get(x)[1] for x in stream)
        tab = {}
        for eps in (0.01, 0.02, 0.05):
            for rho in (0.0, 0.05, 0.1, 0.2):
                reps_s = [summarize(run_twindb(gt, stream, eps, seed=s, rho=rho)[0], eps, fs) for s in range(reps)]
                tab["eps=%g,rho=%g" % (eps, rho)] = {k: float(np.mean([r[k] for r in reps_s])) for k in reps_s[0]}
        out[e] = tab
        print(e, "done", flush=True)
    save("e3_sweep_" + "_".join(engines_ or ENG), out)


# ------------------------------------------------------------------ E4 model update (maintenance effort)
def e4_update(eps=0.02, rho=0.1, engines_=None):
    """Structural model update after the view holds all sessions.  The post-update workload is the
    30-query update session followed by the four sessions of the second workload draw (230 queries),
    all under the new model; we report the first 30 queries and the whole post-update workload."""
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e)
        v = WL[e]["update_variant"]
        post = [np.array(x) for x in WL[e]["update_session"] + [y for s in WL[e]["sessions2"] for y in s]
                if gt.get(x, None, v) is not None]
        lhs_new = [np.array(x) for x in WL[e]["update_lhs"] if gt.get(x, None, v) is not None]
        stream = [y for ss_ in sessions(e, gt) for y in ss_]
        res = {"n_post": len(post)}
        stale = [abs(gt.get(x)[0] - gt.get(x, None, v)[0]) / abs(gt.get(x, None, v)[0]) for x in post if gt.get(x) is not None]
        res["stale_rel_err_mean"] = float(np.mean(stale)); res["stale_rel_err_max"] = float(np.max(stale))
        res["stale_viol_frac"] = float(np.mean([s > eps for s in stale]))
        for horizon, tag in ((30, "first30"), (len(post), "all")):
            P = post[:horizon]
            fs = sum(gt.get(x, None, v)[1] for x in P)
            r = {}
            for name, mode in (("TwinDB (invalidate)", "invalidate"), ("TwinDB (delta from stale view)", "delta")):
                reps = []
                for s in range(3):
                    db = TwinDB(5, gt.engine(), eps=eps, seed=s, rho=rho)
                    for x in stream:
                        db.query(x)
                    db.model_update(mode=mode)
                    db.engine = gt.engine(None, v)
                    rows = []
                    for x in P:
                        q = db.query(x)
                        rows.append((q.plan, q.answer, gt.get(x, None, v)[0], q.sim_seconds, q.audit_seconds, q.overhead_s, q.audited))
                    sm = summarize(rows, eps, fs); sm["certificate"] = db.acc.certificate(); reps.append(sm)
                r[name] = {k: float(np.mean([q[k] for q in reps])) for k in reps[0]}
            for n in (15, 30):
                rows, tr, fit = run_offline_gp(gt, P, lhs_new[:n], None, v)
                sm = summarize(rows, eps, fs, extra_seconds=tr + fit); sm["retrain_sims"] = n
                r["Offline GP retrain (n=%d)" % n] = sm
            r["Online GP (2 sigma), cold"] = summarize(run_online_gp(gt, P, eps, fid=None, variant=v), eps, fs)
            r["Full-Sim"] = summarize(run_fullsim(gt, P, None, v), eps, fs)
            res[tag] = r
        out[e] = res
        print(e, {t: {k: (round(x["speedup_latency"], 2), round(x["viol_rate_accepted"], 3)) for k, x in res[t].items()} for t in ("first30", "all")}, flush=True)
    save("e4_update_" + "_".join(engines_ or ENG), out)


# ------------------------------------------------------------------ E5 granularity
def e5_granularity(eps=0.02, rho=0.1, engines_=None):
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e)
        pts0 = [np.array(x) for x in WL[e]["granularity"]]
        fids = [WL[e]["default_fidelity"]] + WL[e]["gran_fid"]
        res = {}
        for f in fids:
            fq = None if f == WL[e]["default_fidelity"] else f
            pts = [x for x in pts0 if gt.get(x, fq) is not None]
            cs = [gt.get(x, fq)[1] for x in pts]
            fs = sum(cs)
            reps = [summarize(run_twindb(gt, pts, eps, seed=s, fid=fq, rho=rho)[0], eps, fs) for s in range(5)]
            sm = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
            sm["C_sim_mean_s"] = float(np.mean(cs))
            ref = [gt.get(x)[0] for x in pts if gt.get(x) is not None]
            mine = [gt.get(x, fq)[0] for x in pts if gt.get(x) is not None]
            sm["agg_shift_vs_default_rel"] = float(np.mean(np.abs(np.array(mine) - np.array(ref)) / np.abs(ref)))
            res["fid=%g" % f] = sm
        out[e] = res
        print(e, {k: (round(v["C_sim_mean_s"], 2), round(v["speedup_latency"], 2), round(v["mean_rel_err"], 4)) for k, v in res.items()}, flush=True)
    save("e5_granularity_" + "_".join(engines_ or ENG), out)


# ------------------------------------------------------------------ E6 index + maintenance micro-benchmarks
def e6_micro():
    from scipy.spatial import cKDTree
    from sklearn.neighbors import BallTree
    rng = np.random.default_rng(0)
    out = {"index": {}, "rls": {}, "pg": {}}
    for d in (5, 10, 20):
        for n in (100, 1000, 10000, 100000):
            X = rng.random((n, d)); Q = rng.random((200, d))
            t = time.perf_counter()
            for q in Q:
                np.argmin(np.einsum("ij,ij->i", X - q, X - q))
            lin = (time.perf_counter() - t) / 200
            t = time.perf_counter(); kd = cKDTree(X); kd_build = time.perf_counter() - t
            t = time.perf_counter()
            for q in Q:
                kd.query(q)
            kdq = (time.perf_counter() - t) / 200
            t = time.perf_counter(); bt = BallTree(X); bt_build = time.perf_counter() - t
            t = time.perf_counter()
            for q in Q:
                bt.query(q[None], k=1)
            btq = (time.perf_counter() - t) / 200
            out["index"]["d=%d,n=%d" % (d, n)] = {"linear_us": lin * 1e6, "kdtree_us": kdq * 1e6, "kdtree_build_ms": kd_build * 1e3,
                                                  "balltree_us": btq * 1e6, "balltree_build_ms": bt_build * 1e3}
            print(d, n, out["index"]["d=%d,n=%d" % (d, n)], flush=True)
    for d in (5, 10, 20, 50):
        s = Sensitivity(d); X = rng.random((2000, d)); y = X @ rng.random(d)
        t = time.perf_counter()
        for x, v in zip(X, y):
            s.update(x, v)
        out["rls"]["d=%d" % d] = {"update_us": (time.perf_counter() - t) / 2000 * 1e6, "state_bytes": s.state_bytes()}
    # persistence cost of ISM state in PostgreSQL (one anchor + sensitivity upsert per simulation)
    try:
        st = PgStore("host=/tmp dbname=twindb user=twindb", entity="bench"); st.reset_tables()
        s = Sensitivity(5); ts = []
        for i in range(500):
            x = rng.random(5); s.update(x, float(x.sum()))
            t = time.perf_counter(); st.persist(x, float(x.sum()), s); ts.append(time.perf_counter() - t)
        out["pg"] = {"persist_ms_median": float(np.median(ts) * 1e3), "persist_ms_p95": float(np.percentile(ts, 95) * 1e3)}
    except Exception as ex:
        out["pg"] = {"error": str(ex)}
    print(out["rls"], out["pg"])
    save("e6_micro", out)


def e8_seed2(eps=0.02, rho=0.1, engines_=None):
    """Second, independent workload draw (4 sessions per engine): main methods only."""
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e)
        ss = sessions(e, gt, key="sessions2")
        stream = [x for s in ss for x in s]
        full = run_fullsim(gt, stream); fs = sum(r[3] for r in full)
        lhs = [np.array(x) for x in WL[e]["lhs"] if gt.get(x) is not None]
        m = {"Exact cache": summarize(run_exactcache(gt, stream, eps), eps, fs)}
        rows, tr, fit = run_offline_gp(gt, stream, lhs[:30]); m["Offline GP (n=30)"] = summarize(rows, eps, fs, extra_seconds=tr + fit)
        m["Online GP (2 sigma)"] = summarize(run_online_gp(gt, stream, eps), eps, fs)
        for name, kw in (("TwinDB-ISM (first-order)", dict(predictor="local")), ("TwinDB (kernel)", dict(predictor="gp")),
                         ("TwinDB (kernel), no audits", dict(predictor="gp", rho=0.0))):
            reps = []
            for sd in range(5):
                k2 = dict(rho=rho); k2.update(kw)
                rows, db = run_twindb(gt, stream, eps, seed=sd, **k2)
                sm = summarize(rows, eps, fs); sm["certificate"] = db.acc.certificate(); reps.append(sm)
            m[name] = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
            m[name]["viol_max"] = float(max(r["viol_rate_accepted"] for r in reps))
        out[e] = {"methods": m, "n_queries": len(stream)}
        print(e, {k: (round(v["speedup_latency"], 2), round(v["viol_rate_accepted"], 3)) for k, v in m.items()}, flush=True)
    save("e8_seed2_" + "_".join(engines_ or ENG), out)



# ------------------------------------------------------------------ E9 audited surrogate without the view's anchor offset
def e9_audited_gp(eps=0.02, rho=0.1, engines_=None):
    """Closest prior design: an online GP that keeps every simulated run and returns its posterior mean
    under the same audited threshold (predictor='gpmean'), against TwinDB's kernel maintenance from the
    nearest anchor, on both workload draws, 10 audit seeds."""
    out = {}
    for e in (engines_ or ENG):
        gt = GT(e)
        res = {}
        for key in ("sessions", "sessions2"):
            stream = [y for s_ in sessions(e, gt, key=key) for y in s_]
            fs = sum(gt.get(x)[1] for x in stream)
            for name, pred in (("Audited GP (posterior mean)", "gpmean"), ("TwinDB (kernel)", "gp")):
                reps = []
                for sd in range(10):
                    rows, db = run_twindb(gt, stream, eps, seed=sd, rho=rho, predictor=pred)
                    sm = summarize(rows, eps, fs); sm["certificate"] = db.acc.certificate(); reps.append(sm)
                r = {k: float(np.mean([q[k] for q in reps])) for k in reps[0]}
                r["viol_std"] = float(np.std([q["viol_rate_accepted"] for q in reps]))
                res["%s|%s" % (key, name)] = r
        out[e] = res
        print(e, {k: (round(v["speedup_latency"], 2), round(v["viol_rate_accepted"], 3)) for k, v in res.items()}, flush=True)
    save("e9_auditedgp_" + "_".join(engines_ or ENG), out)


# ------------------------------------------------------------------ E10 prefetching, session bootstrap, injected miscalibration
def e10_extra(eps=0.02, rho=0.1, engines_=None):
    """(a) Oracle think-time prefetch: while the analyst thinks for tau seconds, the next scenario is
    simulated in advance (an upper bound on any prefetcher); latency per simulated query becomes
    max(0, C - tau).  Applied to Full-Sim and to TwinDB's simulate plans.
    (b) Session bootstrap: 95% intervals of TwinDB's latency speedup and violation rate obtained by
    resampling sessions (10 audit seeds x 1000 resamples).
    (c) Injected miscalibration: after the first half of the workload the operator's answers are
    biased by +5% (beyond the 2% tolerance); we measure how many maintained answers are returned
    before the audited threshold stops accepting (no maintained answer for 20 consecutive
    acceptance tests), and the violation rate after injection, for several audit rates."""
    out = {}
    rng = np.random.default_rng(0)
    for e in (engines_ or ENG):
        gt = GT(e)
        ss = sessions(e, gt)
        stream = [y for s in ss for y in s]
        bounds = np.cumsum([0] + [len(s) for s in ss])
        C = np.array([gt.get(x)[1] for x in stream])
        fs = C.sum()
        res = {}
        # (a) + (b)
        per_seed = []
        for sd in range(10):
            rows, db = run_twindb(gt, stream, eps, seed=sd, rho=rho, predictor="gp")
            per_seed.append(rows)
        for tau in (10.0, 30.0, 60.0):
            pf_full = sum(max(0.0, c - tau) for c in C)
            sp = []
            for rows in per_seed:
                lat = sum((max(0.0, r[3] - tau) if r[0] == "sim" else 0.0) + r[5] for r in rows)
                sp.append(fs / max(lat, 1e-9))
            res["prefetch_tau%d" % tau] = {"fullsim_prefetch_speedup": float(fs / max(pf_full, 1e-9)),
                                           "twindb_prefetch_speedup": float(np.mean(sp))}
        boot_sp, boot_v = [], []
        S = len(ss)
        for rows in per_seed:
            lat_s = np.array([sum(r[3] + r[5] for r in rows[bounds[i]:bounds[i + 1]]) for i in range(S)])
            full_s = np.array([C[bounds[i]:bounds[i + 1]].sum() for i in range(S)])
            acc_s = np.array([sum(1 for r in rows[bounds[i]:bounds[i + 1]] if r[0] == "ism") for i in range(S)])
            vio_s = np.array([sum(1 for r in rows[bounds[i]:bounds[i + 1]] if r[0] == "ism" and abs(r[1] - r[2]) > eps * abs(r[2])) for i in range(S)])
            for _ in range(100):
                k = rng.integers(0, S, S)
                boot_sp.append(full_s[k].sum() / max(lat_s[k].sum(), 1e-9))
                boot_v.append(vio_s[k].sum() / max(acc_s[k].sum(), 1))
        res["bootstrap"] = {"speedup_ci": [float(np.percentile(boot_sp, 2.5)), float(np.percentile(boot_sp, 97.5))],
                            "viol_ci": [float(np.percentile(boot_v, 2.5)), float(np.percentile(boot_v, 97.5))]}
        # (c)
        half = len(stream) // 2
        inj = {}
        for r_ in (0.0, 0.05, 0.1, 0.2):
            reps = []
            for sd in range(5):
                db = TwinDB(5, gt.engine(), eps=eps, seed=sd, rho=r_)
                orig = db._predict
                state = {"on": False}
                def biased(x, orig=orig, state=state):
                    yhat, s = orig(x)
                    return (yhat * 1.05, s) if state["on"] else (yhat, s)
                db._predict = biased
                acc_after, vio_after, last_acc, t_stop = 0, 0, None, None
                since = 0
                for t, x in enumerate(stream):
                    if t == half:
                        state["on"] = True
                    q = db.query(x)
                    if t >= half:
                        if q.plan == "ism":
                            acc_after += 1; since = 0
                            vio_after += abs(q.answer - gt.get(x)[0]) > eps * abs(gt.get(x)[0])
                        elif q.plan == "sim":
                            since += 1
                            if since >= 20 and t_stop is None:
                                t_stop = acc_after
                reps.append({"accepted_after": acc_after, "violations_after": int(vio_after),
                             "viol_rate_after": vio_after / max(acc_after, 1),
                             "accepted_before_stop": t_stop if t_stop is not None else acc_after,
                             "stopped": t_stop is not None, "certificate_end": db.acc.certificate()})
            inj["rho=%g" % r_] = {k: float(np.mean([q[k] for q in reps])) for k in reps[0]}
        res["injected"] = inj
        res["n_queries"] = len(stream); res["n_after"] = len(stream) - half
        out[e] = res
        print(e, json.dumps(res)[:600], flush=True)
    save("e10_extra_" + "_".join(engines_ or ENG), out)

EXPERIMENTS = {"e1": e1_main, "e2": e2_guarantee, "e3": e3_sweep, "e4": e4_update, "e5": e5_granularity, "e6": e6_micro, "e8": e8_seed2, "e9": e9_audited_gp, "e10": e10_extra}

if __name__ == "__main__":
    fn = EXPERIMENTS[sys.argv[1]]
    if len(sys.argv) > 2:
        fn(engines_=sys.argv[2].split(","))
    else:
        fn()
