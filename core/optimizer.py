"""
TwinDB Query Optimizer - Scenario-aware optimization for TwinQL
Implements cost-based optimization with simulation-specific rewrites

Cost Model: C_total = C_db + C_sim - C_cache_benefit
"""
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from core.operators import (
    OpType, LogicalOp, LogicalOperator,
    Hist, Sim, SimBatch, ScanCache, ISM, Agg, Join, Project, Select, Union,
    print_plan, collect_operators
)
from core.db import execute_query


# ============================================================================
# Cost Model
# ============================================================================

@dataclass
class CostModel:
    """
    Cost model for hybrid DB + simulation queries.
    
    Total cost: C = C_db + C_sim - C_reuse
    
    Calibrated from empirical measurements on TimescaleDB + Python simulation.
    """
    # DB costs (milliseconds per 1000 rows) - calibrated
    scan_cost_per_k: float = 2.5       # ~20ms for 8760 rows
    agg_cost_per_group: float = 0.1
    join_cost_per_k: float = 1.0
    
    # Simulation costs (milliseconds) - calibrated
    sim_setup_cost: float = 100.0      # Model initialization + DB writes
    sim_per_timestep: float = 0.19     # ~1700ms for 8760 steps
    sim_batch_overhead: float = 50.0
    
    # Cache costs - calibrated
    cache_lookup_cost: float = 2.0
    cache_scan_per_k: float = 2.5
    
    # ISM costs - calibrated
    ism_cost_per_param: float = 0.5    # O(d) update
    
    def estimate(self, op: LogicalOp) -> float:
        """Estimate total execution cost in milliseconds"""
        if isinstance(op, Hist):
            return self._cost_hist(op)
        elif isinstance(op, Sim):
            return self._cost_sim(op)
        elif isinstance(op, SimBatch):
            return self._cost_sim_batch(op)
        elif isinstance(op, ScanCache):
            return self._cost_cache_scan(op)
        elif isinstance(op, ISM):
            return self._cost_ism(op)
        elif isinstance(op, Agg):
            return self._cost_agg(op)
        elif isinstance(op, Join):
            return self._cost_join(op)
        elif isinstance(op, (Project, Select)):
            return self.estimate(op.child)  # Pass-through cost
        elif isinstance(op, Union):
            return sum(self.estimate(c) for c in op.children_ops)
        return 1.0
    
    def _cost_hist(self, op: Hist) -> float:
        card = op.cardinality_estimate()
        return self.scan_cost_per_k * (card / 1000)
    
    def _cost_sim(self, op: Sim) -> float:
        card = op.cardinality_estimate()
        return self.sim_setup_cost + self.sim_per_timestep * card
    
    def _cost_sim_batch(self, op: SimBatch) -> float:
        n = len(op.tids)
        card_per_twin = 8760 * len(op.metrics)
        # Batch amortizes setup cost
        return self.sim_setup_cost + self.sim_batch_overhead * n + self.sim_per_timestep * card_per_twin * n
    
    def _cost_cache_scan(self, op: ScanCache) -> float:
        card = op.cardinality_estimate()
        return self.cache_lookup_cost + self.cache_scan_per_k * (card / 1000)
    
    def _cost_ism(self, op: ISM) -> float:
        """Cost of ISM update: Base retrieval + O(d) delta calculation"""
        base_cost = self.estimate(op.base)
        return base_cost + self.ism_cost_per_param * len(op.delta_p)
    
    def _cost_agg(self, op: Agg) -> float:
        child_cost = self.estimate(op.child)
        groups = op.cardinality_estimate()
        return child_cost + self.agg_cost_per_group * groups
    
    def _cost_join(self, op: Join) -> float:
        left_cost = self.estimate(op.left)
        right_cost = self.estimate(op.right)
        join_card = op.cardinality_estimate()
        return left_cost + right_cost + self.join_cost_per_k * (join_card / 1000)


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """Manages scenario simulation cache with signature-based lookup"""
    
    @staticmethod
    def compute_signature(tid: str, cid: str, mid: str, 
                          window: Tuple[str, str], metrics: List[str]) -> str:
        """Compute deterministic cache signature"""
        key = f"{tid}|{cid}|{mid}|{window[0]}|{window[1]}|{','.join(sorted(metrics))}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    @staticmethod
    def check_cache(sig: str) -> bool:
        """Check if simulation result exists in cache"""
        try:
            result = execute_query(
                "SELECT 1 FROM sim_cache WHERE sig = %s", (sig,), as_df=False
            )
            return bool(result)
        except Exception:
            return False
    
    @staticmethod
    def batch_check_cache(sigs: List[str]) -> Set[str]:
        """Batch check which signatures are cached"""
        if not sigs:
            return set()
        try:
            placeholders = ','.join(['%s'] * len(sigs))
            result = execute_query(
                f"SELECT sig FROM sim_cache WHERE sig IN ({placeholders})",
                tuple(sigs), as_df=False
            )
            return {r['sig'] for r in result} if result else set()
        except Exception:
            return set()


# ============================================================================
# Optimizer
# ============================================================================

@dataclass
class OptimizationStats:
    """Statistics collected during optimization"""
    cache_hits: int = 0
    cache_misses: int = 0
    ism_hits: int = 0
    batches_formed: int = 0
    predicates_pushed: int = 0
    original_cost: float = 0.0
    optimized_cost: float = 0.0


class Optimizer:
    """
    TwinQL Query Optimizer
    
    Applies rewrite rules in order:
    1. Cache Rewrite: Sim → ScanCache if result is cached
    2. ISM Rewrite: Sim → ISM if a similar result is cached
    3. Batch Grouping: Multiple Sims with same (scenario, model, window) → SimBatch
    3. Predicate Push-down: Push temporal predicates into Hist/Sim
    4. Aggregation Push-down: Push aggregation to SQL when possible
    """
    
    def __init__(self):
        self.cost_model = CostModel()
        self.cache_manager = CacheManager()
        self.stats = OptimizationStats()
    
    def optimize(self, plan: LogicalOp) -> LogicalOp:
        """Main optimization entry point"""
        self.stats = OptimizationStats()
        self.stats.original_cost = self.cost_model.estimate(plan)
        
        # Apply rewrites
        plan = self._rewrite_cache(plan)
        plan = self._rewrite_ism(plan)
        plan = self._rewrite_batch(plan)
        plan = self._push_predicates(plan)
        
        self.stats.optimized_cost = self.cost_model.estimate(plan)
        return plan
    
    def _rewrite_cache(self, op: LogicalOp) -> LogicalOp:
        """Rule 1: Replace Sim with ScanCache if cached"""
        if isinstance(op, Sim):
            sig = self.cache_manager.compute_signature(
                op.tid, op.scenario_id, op.model_id, op.window, op.metrics
            )
            if self.cache_manager.check_cache(sig):
                self.stats.cache_hits += 1
                return ScanCache(
                    cache_sig=sig, tid=op.tid, scenario_id=op.scenario_id,
                    window=op.window, metrics=op.metrics
                )
            self.stats.cache_misses += 1
            return op
        
        # Recurse
        if isinstance(op, Agg):
            op.child = self._rewrite_cache(op.child)
        elif isinstance(op, Join):
            op.left = self._rewrite_cache(op.left)
            op.right = self._rewrite_cache(op.right)
        elif isinstance(op, (Project, Select)):
            op.child = self._rewrite_cache(op.child)
        elif isinstance(op, Union):
            op.children_ops = [self._rewrite_cache(c) for c in op.children_ops]
        
        return op
    
    def _rewrite_batch(self, op: LogicalOp) -> LogicalOp:
        """Rule 2: Group Sim operators into SimBatch"""
        if isinstance(op, Union):
            # Collect Sim operators that can be batched
            sims = [c for c in op.children_ops if isinstance(c, Sim)]
            others = [c for c in op.children_ops if not isinstance(c, Sim)]
            
            if len(sims) > 1:
                # Group by (scenario, model, window)
                groups: Dict[str, List[Sim]] = {}
                for sim in sims:
                    key = f"{sim.scenario_id}|{sim.model_id}|{sim.window}"
                    groups.setdefault(key, []).append(sim)
                
                new_children = list(others)
                for key, group in groups.items():
                    if len(group) > 1:
                        self.stats.batches_formed += 1
                        first = group[0]
                        batch = SimBatch(
                            tids=[s.tid for s in group],
                            scenario_id=first.scenario_id,
                            model_id=first.model_id,
                            window=first.window,
                            metrics=first.metrics
                        )
                        new_children.append(batch)
                    else:
                        new_children.extend(group)
                
                op.children_ops = new_children
        
        # Recurse
        if isinstance(op, (Agg, Project, Select)):
            op.child = self._rewrite_batch(op.child)
        elif isinstance(op, Join):
            op.left = self._rewrite_batch(op.left)
            op.right = self._rewrite_batch(op.right)
        
        return op
    
    def _push_predicates(self, op: LogicalOp) -> LogicalOp:
        """Rule 3: Push temporal predicates into leaf operators"""
        if isinstance(op, Select) and isinstance(op.child, Hist):
            # Could narrow window based on predicate - simplified for now
            self.stats.predicates_pushed += 1
        
        # Recurse
        if isinstance(op, (Agg, Project, Select)):
            op.child = self._push_predicates(op.child)
        elif isinstance(op, Join):
            op.left = self._push_predicates(op.left)
            op.right = self._push_predicates(op.right)
        elif isinstance(op, Union):
            op.children_ops = [self._push_predicates(c) for c in op.children_ops]
        
        return op
    
    def _rewrite_ism(self, op: LogicalOp) -> LogicalOp:
        """Rule 2: Replace Sim with ISM if a 'nearby' scenario result exists"""
        if isinstance(op, Sim):
            # Search for best ISM candidate in cache
            candidate = self._find_ism_candidate(op)
            if candidate:
                self.stats.ism_hits += 1
                base, scenario_id, delta_p, sensitivity = candidate
                return ISM(
                    base=base, scenario_id=scenario_id, delta_p=delta_p,
                    sensitivity=sensitivity, tid=op.tid, window=op.window, metrics=op.metrics
                )
            return op
        
        # Recurse
        if isinstance(op, Agg):
            op.child = self._rewrite_ism(op.child)
        elif isinstance(op, Join):
            op.left = self._rewrite_ism(op.left)
            op.right = self._rewrite_ism(op.right)
        elif isinstance(op, (Project, Select)):
            op.child = self._rewrite_ism(op.child)
        elif isinstance(op, Union):
            op.children_ops = [self._rewrite_ism(c) for c in op.children_ops]
            
        return op
    
    def _find_ism_candidate(self, op: Sim) -> Optional[Tuple]:
        """Find best cached result to use as ISM base"""
        try:
            sql = "SELECT cid, cfg FROM scenario WHERE cid != %s LIMIT 5"
            results = execute_query(sql, (op.scenario_id,), as_df=False)
            
            target_cfg_query = execute_query("SELECT cfg FROM scenario WHERE cid = %s", (op.scenario_id,), as_df=False)
            if not target_cfg_query: return None
            t_cfg = target_cfg_query[0]['cfg'].get('retrofit_package', {})
            
            best_cand = None
            min_delta = 1.0
            
            for res in results:
                cid = res['cid']
                cfg = res['cfg'].get('retrofit_package', {})
                
                # Check cache existence
                sig = self.cache_manager.compute_signature(op.tid, cid, op.model_id, op.window, op.metrics)
                if not self.cache_manager.check_cache(sig): continue
                
                # Compute delta
                delta_p = {}
                max_d = 0
                for k, v in t_cfg.items():
                    if k in cfg and cfg[k] != 0:
                        d = abs(v - cfg[k]) / cfg[k]
                        delta_p[k] = v - cfg[k]
                        max_d = max(max_d, d)
                
                # Threshold for ISM validity (heuristic 30%)
                if max_d < 0.3 and max_d < min_delta:
                    min_delta = max_d
                    best_cand = (ScanCache(sig, op.tid, cid, op.window, op.metrics), cid, delta_p, {'wall_u': 150000})
            
            return best_cand
        except Exception:
            return None
    
    def explain(self, plan: LogicalOp) -> str:
        """Generate EXPLAIN output"""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                    TwinDB Query Plan                         ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Original Cost:  {self.stats.original_cost:>8.2f} ms                            ║",
            f"║ Optimized Cost: {self.stats.optimized_cost:>8.2f} ms                            ║",
            f"║ Speedup:        {self.stats.original_cost / max(0.01, self.stats.optimized_cost):>8.2f}x                             ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Cache Hits:     {self.stats.cache_hits:>4}                                      ║",
            f"║ Cache Misses:   {self.stats.cache_misses:>4}                                      ║",
            f"║ ISM Hits:       {self.stats.ism_hits:>4}                                      ║",
            f"║ Batches Formed: {self.stats.batches_formed:>4}                                      ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ Plan Tree:                                                   ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            print_plan(plan)
        ]
        return "\n".join(lines)


# ============================================================================
# Multi-Twin Query Optimization
# ============================================================================

def optimize_multi_twin_query(tids: List[str], scenario_id: str, model_id: str,
                               window: Tuple[str, str], metrics: List[str]) -> Tuple[List[str], List[str]]:
    """
    Partition twins into cached vs uncached for batch optimization.
    Returns: (cached_tids, uncached_tids)
    """
    cache_mgr = CacheManager()
    
    # Compute signatures
    sig_to_tid = {}
    for tid in tids:
        sig = cache_mgr.compute_signature(tid, scenario_id, model_id, window, metrics)
        sig_to_tid[sig] = tid
    
    # Batch check
    cached_sigs = cache_mgr.batch_check_cache(list(sig_to_tid.keys()))
    
    cached_tids = [sig_to_tid[s] for s in cached_sigs if s in sig_to_tid]
    uncached_tids = [t for t in tids if t not in cached_tids]
    
    return cached_tids, uncached_tids
