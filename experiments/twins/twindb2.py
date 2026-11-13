"""TwinDB: scenario views over a continuous parameter space with
audited first-order maintenance.

Components (one class each, all state also persisted to PostgreSQL when a
connection is given):
  ScenarioIndex     exact-match signature lookup + nearest-anchor search
  Sensitivity       recursive least squares (RLS) fit of the local linear model
  AuditedAcceptor   acceptance threshold, audit coins, observable certificate (Theorem A')
  Optimizer         per-query plan choice: CacheHit | Maintain (ISM) | Simulate
The simulator is reached only through `engine(x) -> (answer, elapsed_seconds)`.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- index
class ScenarioIndex:
    """Anchors = scenarios whose exact answer is materialized.  Exact lookup by
    signature (hash map); nearest-anchor search by linear scan (default) or a
    KD-tree that is rebuilt lazily after `rebuild_every` inserts."""

    def __init__(self, d, kind="auto", rebuild_every=64, switch_at=1000):
        self.d, self.kind, self.rebuild_every, self.switch_at = d, kind, rebuild_every, switch_at
        self.X = np.zeros((0, d))
        self.y = np.zeros(0)
        self.sig = {}
        self._tree, self._tree_n = None, 0

    @staticmethod
    def key(x):
        return tuple(np.round(np.asarray(x, float), 10))

    def insert(self, x, y):
        k = self.key(x)
        if k in self.sig:
            self.y[self.sig[k]] = y
            return
        self.sig[k] = len(self.y)
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y)

    def exact(self, x):
        i = self.sig.get(self.key(x))
        return None if i is None else float(self.y[i])

    def nearest(self, x):
        n = len(self.y)
        if n == 0:
            return None
        if self.kind == "kdtree" or (self.kind == "auto" and n >= self.switch_at):
            from scipy.spatial import cKDTree
            if self._tree is None or n - self._tree_n >= self.rebuild_every:
                self._tree, self._tree_n = cKDTree(self.X[:n]), n
            dist, i = self._tree.query(x)
            # anchors inserted since the last rebuild are scanned linearly
            if self._tree_n < n:
                tail = self.X[self._tree_n:n]
                dt = np.linalg.norm(tail - x, axis=1)
                j = int(np.argmin(dt))
                if dt[j] < dist:
                    i = self._tree_n + j
            return int(i)
        return int(np.argmin(np.einsum("ij,ij->i", self.X - x, self.X - x)))

    def clear(self):
        self.__init__(self.d, self.kind, self.rebuild_every, self.switch_at)


# ----------------------------------------------------------------- sensitivity
class Sensitivity:
    """RLS for y ~ a + beta^T x on anchors, forgetting factor lam, ridge prior delta.
    One rank-1 update per new anchor: O(d^2) time, O(d^2) state."""

    def __init__(self, d, lam=0.98, delta=100.0):
        self.d, self.lam, self.delta = d, lam, delta
        self.reset()

    def reset(self):
        self.w = np.zeros(self.d + 1)
        self.P = self.delta * np.eye(self.d + 1)
        self.n = 0

    def update(self, x, y):
        z = np.concatenate(([1.0], x))
        Pz = self.P @ z
        k = Pz / (self.lam + z @ Pz)
        self.w = self.w + k * (y - z @ self.w)
        self.P = (self.P - np.outer(k, Pz)) / self.lam
        self.n += 1

    @property
    def beta(self):
        return self.w[1:]

    def state_bytes(self):
        return (self.w.nbytes + self.P.nbytes + 8)


# ----------------------------------------------------------------- acceptor
@dataclass
class AuditedAcceptor:
    """Accept iff s * exp(theta) <= eps_t.  theta moves only on audited accepted
    answers: theta += gamma (v - alpha).  The certificate U_T (Theorem A') is
    computed from logged quantities only."""
    alpha: float = 0.05
    rho: float = 0.1
    gamma: float = 0.05
    delta: float = 0.05
    lam_grid: tuple = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
    theta: float = math.log(2.0)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    # epoch counters
    N: int = 0          # accepted (non-hit) answers
    S: int = 0          # violations among accepted answers whose truth is known (audited) -- for logging
    Saud: float = 0.0   # sum over audited accepts of (v - alpha)
    epoch: int = 1
    pending: int = 0

    eta: float = 0.0          # audit-driven correction (log scale)
    base: float = math.log(2.0)  # log quantile of recently observed scores

    def accept(self, s, eps_t):
        # the audit correction eta moves the threshold in both directions: the observed-score
        # calibration is biased (simulated scenarios are the rejected, harder ones), and the audits
        # of returned answers correct it toward the target rate alpha
        return s * math.exp(self.base + self.eta) <= eps_t

    def set_base(self, scores, window=20):
        sc = list(scores)[-window:]
        if len(sc) >= 5:
            self.base = math.log(max(float(np.quantile(sc, 1 - self.alpha)), 1e-12))

    def coin(self):
        return bool(self.rng.random() < self.rho)

    def record_accept(self):
        self.N += 1

    def record_audit(self, violated):
        v = 1.0 if violated else 0.0
        self.Saud += v - self.alpha
        self.S += int(violated)
        self.eta += self.gamma * (v - self.alpha)

    def warm_start(self, scores):
        if len(scores) >= 3:
            q = np.quantile(scores, min(1.0, (1 - self.alpha) * (1 + 1.0 / len(scores))))
            self.theta = math.log(max(q, 1e-12))

    def new_epoch(self, theta0=math.log(2.0)):
        self.epoch += 1
        self.N, self.S, self.Saud = 0, 0, 0.0
        self.eta, self.base = 0.0, theta0

    def certificate(self):
        """U_T for the current epoch, minimised over the lambda grid (union bound)."""
        if self.N == 0 or self.rho <= 0:
            return 1.0
        a, r = self.alpha, self.rho
        dk = self.delta / (self.epoch * (self.epoch + 1))
        L = math.log(len(self.lam_grid) / dk)
        b = max(1 - a, a * (1 - r) / r)
        best = 1.0
        for lam in self.lam_grid:
            if lam >= 3 / b:
                continue
            c = lam * (1 - r) / (2 * r * (1 - lam * b / 3))
            if c * (1 - 2 * a) >= 1:
                continue
            u = (a * (1 + c * a) + ((self.Saud + self.pending * (1 - a)) / r + L / lam) / self.N) / (1 - c * (1 - 2 * a))
            best = min(best, u)
        return best


# ----------------------------------------------------------------- optimizer
@dataclass
class QueryResult:
    plan: str            # hit | ism | sim
    answer: float
    sim_seconds: float   # simulator seconds on the analyst's critical path
    audit_seconds: float # simulator seconds spent on audits (off the critical path)
    overhead_s: float    # measured system time (planning + maintenance + persistence)
    audited: bool = False


class TwinDB:
    """Per-entity, per-aggregate scenario view maintained under a continuous parameter space."""

    def __init__(self, d, engine, eps=0.01, alpha=0.05, rho=0.1, gamma=0.5, seed=0,
                 index="auto", lam=0.98, s_min=1e-6, cold=None, predictor="gp",
                 accept_rule="audited", store=None, reset_after=50, k_local=None, ridge=1e-6, window=20, auditor=None, audit_in_base=False):
        self.d, self.engine, self.eps = d, engine, eps
        self.index = ScenarioIndex(d, index)
        self.sens = Sensitivity(d, lam)
        self.acc = AuditedAcceptor(alpha=alpha, rho=rho, gamma=gamma,
                                   rng=np.random.default_rng(seed))
        self.s_min, self.cold = s_min, (d + 1 if cold is None else cold)
        self.predictor, self.accept_rule = predictor, accept_rule
        self.store, self.reset_after = store, reset_after
        self.k_local, self.ridge = (2 * d + 2 if k_local is None else k_local), ridge
        self.window = window
        self.audit_in_base = audit_in_base
        self.auditor, self.pending = auditor, []     # background audits: auditor(x) -> Future of (y, seconds)
        self.c_sim, self.c_op = [], []      # measured engine and operator times (running medians)
        self._gp, self._gp_n, self._gp_hyper_n, self.hyper_every = None, -1, 0, 10
        self.prior = None
        self.loo = []             # scores of anchors predicted before they were simulated
        self.since_accept = 0
        self.log = []

    # -- prediction ------------------------------------------------------
    def _predict(self, x):
        """First-order maintenance from the nearest materialized scenario b:
        yhat = q(b) + beta^T (x - b).  Returns (yhat, scale) where the scale is the
        standard error of the first-order prediction under the local fit; the scale only
        ranks how risky a reuse is, validity comes from audits (Theorem A')."""
        i = self.index.nearest(x)
        b, yb = self.index.X[i], self.index.y[i]
        dx = x - b
        if self.predictor == "local":
            beta, se = self._local_fit(x, i, dx)
            yhat = yb + float(beta @ dx)
            s = se
        elif self.predictor == "rls":
            yhat = yb + float(self.sens.beta @ dx)
            s = float(np.linalg.norm(dx))
        elif self.predictor in ("gp", "gpmean"):
            n = len(self.index.y)
            if self._gp_n != n:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
                if self._gp is None or n - self._gp_hyper_n >= self.hyper_every:
                    k = ConstantKernel(1.0) * RBF(np.full(self.d, 0.5), (1e-2, 1e2)) + WhiteKernel(1e-6, (1e-10, 1e-1))
                    self._gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0)
                    self._gp.fit(self.index.X, self._targets())
                    self._gp_hyper_n = n
                else:   # incremental refit with the current hyperparameters
                    self._gp = GaussianProcessRegressor(self._gp.kernel_, normalize_y=True, optimizer=None)
                    self._gp.fit(self.index.X, self._targets())
                self._gp_n = n
            # kernel maintenance from the nearest anchor b: qhat(x) = q(b) + (m(x) - m(b)), m the GP posterior
            # mean of the (prior-corrected) view; exact at anchors even when the GP smooths
            mu, sd = self._gp.predict(np.vstack([x, b]), return_std=True)
            pr = self._prior_at(np.vstack([x, b]))
            if self.predictor == "gp":
                yhat = yb + float((mu[0] + pr[0]) - (mu[1] + pr[1]))
            else:     # baseline: the GP posterior mean itself, no offset from the nearest anchor
                yhat = float(mu[0] + pr[0])
            s = float(sd[0])
        elif self.predictor == "nn":            # scenario-library nearest neighbour
            yhat = yb
            s = float(np.linalg.norm(dx))
        else:
            raise ValueError(self.predictor)
        s = max(s, self.s_min * max(abs(yhat), 1e-12))
        return yhat, s

    def _prior_at(self, X):
        """Answer of the invalidated view (delta maintenance after a structural update)."""
        if self.prior is None:
            return np.zeros(len(X))
        return self.prior.predict(X)

    def _targets(self):
        Y = self.index.y
        return Y - self._prior_at(self.index.X) if self.prior is not None else Y

    def _local_fit(self, x, i, dx):
        """Sensitivity at base b = anchor i: weighted least-squares fit of the first-order model
        on the k nearest anchors (stateless; derived from the materialized view).  Returns
        (beta, se) with se = sigma_hat * sqrt(dx^T (A^T W A)^{-1} dx)."""
        X, Y = self.index.X, self.index.y
        n = len(Y)
        k = min(self.k_local, n)
        dist = np.linalg.norm(X - x, axis=1)
        nb = np.argpartition(dist, k - 1)[:k] if k < n else np.arange(n)
        A = X[nb] - X[i]
        w = 1.0 / (dist[nb] + 1e-3)
        r = Y[nb] - Y[i]
        AtW = A.T * w
        M = AtW @ A + self.ridge * np.eye(self.d)
        C = np.linalg.inv(M)
        beta = C @ (AtW @ r)
        res = r - A @ beta
        dof = max(k - self.d - 1, 1)
        s2 = float((w * res ** 2).sum() / max(w.sum(), 1e-12)) * k / dof
        se = math.sqrt(max(s2 * float(dx @ C @ dx), 0.0))
        return beta, se

    def _anchor(self, x, y, yhat=None, s=None):
        if yhat is not None and s is not None:
            self.loo.append(abs(y - yhat) / s)
            if self.accept_rule == "audited":
                self.acc.set_base(self.loo, self.window)
        self.index.insert(x, y)
        self.sens.update(x, y)
        if self.store is not None:
            self.store.persist(x, y, self.sens)

    # -- query -----------------------------------------------------------
    def drain_audits(self, wait=False):
        """Apply finished background audits (all of them if wait=True)."""
        still = []
        for (xa, yhat, eps_t, s, fut) in self.pending:
            if wait or fut.done():
                y, _ = fut.result()
                self.acc.record_audit(abs(y - yhat) > eps_t)
                # audited answers were accepted, i.e. selected for low scores: by default they
                # move the threshold only through eta, not through the window of observed scores
                self._anchor(xa, y, yhat if self.audit_in_base else None, s if self.audit_in_base else None)
                if self.store is not None:
                    self.store.log_audit(xa, yhat, y, eps_t)
            else:
                still.append((xa, yhat, eps_t, s, fut))
        self.pending = still
        self.acc.pending = len(still)

    def query(self, x):
        x = np.asarray(x, float)
        t0 = time.perf_counter()
        if self.pending:
            self.drain_audits()
        hit = self.index.exact(x)
        if hit is not None:
            return self._done(QueryResult("hit", hit, 0.0, 0.0, time.perf_counter() - t0))
        if len(self.index.y) < self.cold:
            o = time.perf_counter() - t0
            y, el = self.engine(x)
            t1 = time.perf_counter()
            yhat = s = None
            if len(self.index.y) > 0:
                yhat, s = self._predict(x)
            self._anchor(x, y, yhat, s)
            return self._done(QueryResult("sim", y, el, 0.0, o + time.perf_counter() - t1))
        if self.c_sim and self.c_op and np.median(self.c_op) >= np.median(self.c_sim):
            ok_cost = False
        else:
            ok_cost = True
        tp = time.perf_counter()
        yhat, s = self._predict(x)
        self.c_op.append(time.perf_counter() - tp)
        eps_t = self.eps * abs(yhat) / (1 + self.eps)
        if not ok_cost:
            ok = False
        elif self.accept_rule == "audited":
            ok = self.acc.accept(s, eps_t)
        elif self.accept_rule == "never":
            ok = False
        elif self.accept_rule == "always":
            ok = True
        else:
            raise ValueError(self.accept_rule)
        if ok:
            self.acc.record_accept()
            self.since_accept = 0
            audited = self.acc.coin()
            o = time.perf_counter() - t0
            aud_s = 0.0
            if audited and self.auditor is not None:
                self.pending.append((x, yhat, eps_t, s, self.auditor(x)))
                self.acc.pending = len(self.pending)
                audited_async = True
            else:
                audited_async = False
            if audited and not audited_async:
                y, aud_s = self.engine(x)              # replay: charged as background time, off the critical path
                t1 = time.perf_counter()
                self.acc.record_audit(abs(y - yhat) > eps_t)
                if self.store is not None:
                    self.store.log_audit(x, yhat, y, eps_t)
                self._anchor(x, y, yhat if self.audit_in_base else None, s if self.audit_in_base else None)
                o += time.perf_counter() - t1
            return self._done(QueryResult("ism", yhat, 0.0, aud_s, o, audited))
        o = time.perf_counter() - t0
        y, el = self.engine(x)
        self.c_sim.append(el)
        t1 = time.perf_counter()
        self._anchor(x, y, yhat, s)
        self.since_accept += 1
        return self._done(QueryResult("sim", y, el, 0.0, o + time.perf_counter() - t1))

    def query_sum(self, xs, eps=None, combine="sum"):
        """Portfolio query SUM_i q(x_i) with one tolerance on the sum.  Each term gets a calibrated
        error bound b_i = s_i exp(theta); terms are accepted in increasing order of b_i while the
        accepted bounds fit the budget eps |sum| / (1 + eps); the remaining terms are simulated.
        Accepted terms are audited with probability rho; an audit counts a violation when the term's
        error exceeds its own bound b_i, which is what the threshold calibrates."""
        eps = self.eps if eps is None else eps
        t0 = time.perf_counter()
        if self.pending:
            self.drain_audits()
        res = [None] * len(xs)
        cand = []
        for i, x in enumerate(xs):
            x = np.asarray(x, float)
            hit = self.index.exact(x)
            if hit is not None:
                res[i] = QueryResult("hit", hit, 0.0, 0.0, 0.0)
            elif len(self.index.y) >= self.cold:
                yhat, sc = self._predict(x)
                cand.append((sc * math.exp(self.acc.base + self.acc.eta), i, x, yhat, sc))
            else:
                cand.append((math.inf, i, x, None, None))
        est = sum(r.answer for r in res if r is not None) + sum(c[3] for c in cand if c[3] is not None)
        budget = eps * abs(est) / (1 + eps) if self.accept_rule == "audited" else 0.0
        cand.sort(key=lambda c: c[0])
        used, acc_ids = 0.0, set()
        if combine == "cov" and self.predictor in ("gp", "gpmean") and self._gp is not None:
            # correlated combination: the GP's joint posterior covariance of the accepted terms,
            # scaled by the calibrated multiplier exp(theta)
            ok = [c for c in cand if c[3] is not None]
            if ok:
                X_ = np.vstack([c[2] for c in ok])
                _, cov = self._gp.predict(X_, return_cov=True)
                mult = math.exp(self.acc.base + self.acc.eta)
                sel, tot = [], 0.0
                for j in range(len(ok)):
                    new_tot = tot + cov[j, j] + 2.0 * sum(cov[j, k] for k in sel)
                    if mult * math.sqrt(max(new_tot, 0.0)) > budget:
                        break
                    sel.append(j); tot = new_tot
                acc_ids = set(ok[j][1] for j in sel)
            cand_iter = []
        else:
            cand_iter = cand
        for b, i, x, yhat, sc in cand_iter:
            if yhat is None:
                break
            nxt = used + b if combine == "sum" else math.sqrt(used ** 2 + b ** 2)
            if nxt > budget:
                break
            used = nxt; acc_ids.add(i)
        over = (time.perf_counter() - t0) / max(len(xs), 1)
        for b, i, x, yhat, sc in cand:
            if i in acc_ids:
                self.acc.record_accept()
                aud, aud_s = self.acc.coin(), 0.0
                if aud:
                    y, aud_s = self.engine(x)
                    self.acc.record_audit(abs(y - yhat) > b)
                    self._anchor(x, y)
                res[i] = QueryResult("ism", yhat, 0.0, aud_s, over, aud)
            else:
                y, el = self.engine(x)
                self.c_sim.append(el)
                self._anchor(x, y, yhat, sc)
                res[i] = QueryResult("sim", y, el, 0.0, over)
        for r in res:
            self.log.append(r)
        return res

    def model_update(self, mode="invalidate", carry_threshold=True):
        """Structural update of the twin model: the view's anchors become stale.
        invalidate: drop anchors and operator state, restart the threshold.
        transfer:   drop anchors, keep kernel hyperparameters and the calibrated threshold.
        delta:      as transfer, and keep the stale view (frozen operator) as a prior; the new
                    view maintains the difference between the new and the stale answers."""
        self.drain_audits(wait=True)     # audits of the closing epoch are credited to it
        if mode == "delta" and self.predictor == "gp" and self._gp is not None:
            import copy
            old = copy.deepcopy(self._gp)
            old_prior = self.prior

            class _Frozen:
                def predict(self_, X):
                    base = old_prior.predict(X) if old_prior is not None else 0.0
                    return old.predict(X) + base
            self.prior = _Frozen()
        else:
            self.prior = None
        self.index.clear()
        self._gp_n, self._gp_hyper_n = -1, 0
        self.sens.reset()
        if mode == "invalidate":
            self._gp = None
            self.acc.new_epoch()
        elif mode == "delta" and not carry_threshold:
            self.acc.new_epoch()          # the threshold restarts; the stale view only serves as prior
        else:
            base, eta = self.acc.base, self.acc.eta
            self.acc.new_epoch()
            self.acc.base, self.acc.eta = base, eta
        self.loo = []
        if mode == "delta":
            self.cold_saved, self.cold = self.cold, 2
        if self.store is not None:
            self.store.invalidate()

    def _done(self, r):
        self.log.append(r)
        return r


# ----------------------------------------------------------------- persistence
class PgStore:
    """Relational state of a scenario view: sim_cache (anchors), op_state (operator state),
    audit_log (audit outcomes)."""

    def __init__(self, dsn, entity="e1"):
        import psycopg2
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False
        self.entity = entity
        with self.conn.cursor() as c:
            c.execute("""create table if not exists sim_cache(entity text, sig text, x double precision[],
                         answer double precision, created timestamptz default now(), primary key(entity, sig));
                         create table if not exists op_state(entity text primary key, w double precision[],
                         p double precision[], n integer, updated timestamptz default now());
                         create table if not exists audit_log(entity text, sig text, predicted double precision,
                         exact double precision, tolerance double precision, violated boolean,
                         at timestamptz default now());""")
        self.conn.commit()

    def persist(self, x, y, sens):
        sig = json.dumps([round(float(v), 10) for v in x])
        with self.conn.cursor() as c:
            c.execute("insert into sim_cache(entity,sig,x,answer) values (%s,%s,%s,%s) "
                      "on conflict (entity,sig) do update set answer=excluded.answer",
                      (self.entity, sig, [float(v) for v in x], float(y)))
            c.execute("""insert into op_state(entity,w,p,n) values (%s,%s,%s,%s)
                         on conflict (entity) do update set w=excluded.w, p=excluded.p, n=excluded.n, updated=now()""",
                      (self.entity, sens.w.tolist(), sens.P.ravel().tolist(), sens.n))
        self.conn.commit()

    def log_audit(self, x, yhat, y, eps_t):
        sig = json.dumps([round(float(v), 10) for v in x])
        with self.conn.cursor() as c:
            c.execute("insert into audit_log(entity,sig,predicted,exact,tolerance,violated) values (%s,%s,%s,%s,%s,%s)",
                      (self.entity, sig, float(yhat), float(y), float(eps_t), bool(abs(y - yhat) > eps_t)))
        self.conn.commit()

    def invalidate(self):
        with self.conn.cursor() as c:
            c.execute("delete from sim_cache where entity=%s; delete from op_state where entity=%s",
                      (self.entity, self.entity))
        self.conn.commit()

    def reset_tables(self):
        with self.conn.cursor() as c:
            c.execute("truncate sim_cache; truncate op_state; truncate audit_log;")
        self.conn.commit()
