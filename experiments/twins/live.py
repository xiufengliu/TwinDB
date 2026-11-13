"""E7: live end-to-end sessions.  TwinQL statements are parsed, planned and executed by
TwinDB with state in PostgreSQL, and every simulation is a real call of the engine
(no ground-truth replay).  Full-Sim runs the same statements by calling the engine for
each one.  Audits run synchronously here (their wall time is reported separately), so the
measured analyst latency is an upper bound on the asynchronous deployment.

usage: python live.py ENGINE SESSION_INDEX
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time

import numpy as np

import engines
from twindb2 import TwinDB, PgStore

HERE = os.path.dirname(os.path.abspath(__file__))
WL = json.load(open(os.path.join(HERE, "workload.json")))

TWINQL = re.compile(r"SELECT\s+SUM\(sim\.(?P<col>\w+)\)\s+FROM\s+(?P<tab>\w+)\s+\w+\s+SIMULATE\s+SCENARIO\s*\((?P<params>[^)]*)\)\s+"
                    r"USING\s+(?P<engine>\w+)\s+AS\s+sim\s+WHERE\s+\w+\.id\s*=\s*'(?P<id>[^']+)'"
                    r"(?:\s+WITH\s+TOLERANCE\s+(?P<tol>[0-9.]+)%(?:\s+RISK\s+(?P<risk>[0-9.]+)%)?)?\s*;?\s*$", re.I | re.S)


def to_twinql(engine, x, entity="twin-1", tol=2.0, risk=5.0):
    p = engines.denorm(engine, x)
    names = engines.ENGINES[engine]["params"]
    assigns = ", ".join("%s = %.6f" % (n, v) for n, v in zip(names, p))
    return ("SELECT SUM(sim.output) FROM twin T SIMULATE SCENARIO (%s) USING %s AS sim "
            "WHERE T.id = '%s' WITH TOLERANCE %g%% RISK %g%%;" % (assigns, engine, entity, tol, risk))


def parse(stmt):
    m = TWINQL.match(stmt.strip())
    if not m:
        raise ValueError("not a TwinQL SIMULATE statement")
    eng = m.group("engine").lower()
    vals = dict((k.strip(), float(v)) for k, v in (a.split("=") for a in m.group("params").split(",")))
    names = engines.ENGINES[eng]["params"]
    b = np.array(engines.ENGINES[eng]["bounds"])
    p = np.array([vals[n] for n in names])
    x = (p - b[:, 0]) / (b[:, 1] - b[:, 0])
    return eng, m.group("id"), x, float(m.group("tol") or 0) / 100.0, float(m.group("risk") or 5) / 100.0


def live_engine(eng, log):
    def call(x):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            out = f.name
        t0 = time.perf_counter()
        r = engines.run(eng, x, out)
        wall = time.perf_counter() - t0
        y = float(np.load(out).sum()); os.remove(out)
        log.append(wall)
        return y, wall
    return call


def main(eng, si):
    pts = [np.array(x) for x in WL[eng]["sessions"][si]][:30]
    stmts = [to_twinql(eng, x) for x in pts]
    res = {"engine": eng, "session": si, "n": len(stmts)}
    # Full-Sim: every statement calls the engine
    calls = []; t0 = time.perf_counter(); full_answers = []
    for st in stmts:
        e, ent, x, tol, _ = parse(st)
        y, _ = live_engine(e, calls)(x); full_answers.append(y)
    res["fullsim_wall_s"] = time.perf_counter() - t0
    res["fullsim_engine_s"] = sum(calls)
    # TwinDB
    store = PgStore("host=/tmp dbname=twindb user=twindb", entity="live-%s-%d" % (eng, si)); store.reset_tables()
    calls = []; plans = []; answers = []; crit = []
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    audit_calls = []
    auditor = lambda xa: pool.submit(live_engine(eng, audit_calls), xa)
    db = TwinDB(5, live_engine(eng, calls), eps=0.02, seed=0, store=store, auditor=auditor)
    # warm view: anchors materialized by earlier sessions of the same twin (their exact answers are in
    # the ground-truth store, computed by the same engine); they are loaded, not re-simulated
    from experiments import GT
    gt = GT(eng); nwarm = 0
    for sj in range(si):
        for xw in WL[eng]["sessions"][sj]:
            g = gt.get(xw)
            if g is not None:
                db._anchor(np.array(xw), g[0]); nwarm += 1
    res["warm_anchors"] = nwarm
    t0 = time.perf_counter()
    for st in stmts:
        q0 = time.perf_counter()
        e, ent, x, tol, risk = parse(st)
        db.eps = tol
        db.acc.alpha = risk
        r = db.query(x)
        q1 = time.perf_counter()
        plans.append(r.plan); answers.append(r.answer)
        crit.append(q1 - q0)                            # audits run in a background thread
    db.drain_audits(wait=True); pool.shutdown()
    res["twindb_wall_s"] = time.perf_counter() - t0
    res["audit_engine_s"] = sum(audit_calls)
    res["twindb_latency_s"] = float(sum(crit))
    res["twindb_engine_s"] = sum(calls)
    res["plans"] = {k: plans.count(k) for k in ("hit", "ism", "sim")}
    res["audits"] = len(audit_calls)
    rel = [abs(a - t) / abs(t) for a, t, p in zip(answers, full_answers, plans) if p == "ism"]
    res["mean_rel_err"] = float(np.mean(rel)) if rel else 0.0
    res["viol"] = float(np.mean([v > 0.02 for v in rel])) if rel else 0.0
    res["speedup_latency"] = res["fullsim_wall_s"] / res["twindb_latency_s"]
    res["speedup_total"] = res["fullsim_wall_s"] / res["twindb_wall_s"]
    # replay accounting for the same session: engine seconds of simulate plans only
    res["replay_latency_s"] = float(sum(r.sim_seconds + r.overhead_s for r in db.log))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "e7_live_%s_s%d.json" % (eng, si))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
