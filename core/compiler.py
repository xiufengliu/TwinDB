"""
TwinDB Query Compiler - Full TwinQL compilation and execution pipeline

Pipeline: TwinQL Text → Parse → AST → Logical Plan → Optimize → Execute
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

from core.operators import (
    LogicalOp, Hist, Sim, SimBatch, ScanCache, Agg, Join, Project, Union, print_plan
)
from core.optimizer import Optimizer, CacheManager, optimize_multi_twin_query
from core.parser import parse_twinql, TwinSelectAST, HistoricalAST, SimulateAST, ScenarioDefAST
from core.db import execute_query, execute_sql
from sim.manager import SimulationManager


# ============================================================================
# AST to Logical Plan Translation
# ============================================================================

class PlanBuilder:
    """Translates TwinQL AST nodes to logical operators"""
    
    @staticmethod
    def from_historical(ast: HistoricalAST) -> LogicalOp:
        """Build plan from HISTORICAL clause"""
        op = Hist(
            tid=ast.twin_id,
            window=(ast.window_start, ast.window_end),
            metrics=[ast.metric]
        )
        if ast.agg_by:
            op = Agg(child=op, group_by=ast.agg_by, agg_funcs={'value': ast.agg_func})
        return op
    
    @staticmethod
    def from_simulate(ast: SimulateAST) -> LogicalOp:
        """Build plan from SIMULATE clause"""
        op = Sim(
            tid=ast.twin_id,
            scenario_id=ast.scenario,
            model_id=ast.model,
            window=(ast.window_start, ast.window_end),
            metrics=[ast.metric]
        )
        if ast.agg_by:
            op = Agg(child=op, group_by=ast.agg_by, agg_funcs={'value': ast.agg_func})
        return op
    
    @staticmethod
    def from_twin_select(ast: TwinSelectAST) -> LogicalOp:
        """Build plan from TWIN SELECT statement"""
        plans = []
        for source in ast.sources:
            if isinstance(source, HistoricalAST):
                plans.append(PlanBuilder.from_historical(source))
            elif isinstance(source, SimulateAST):
                plans.append(PlanBuilder.from_simulate(source))
        
        if len(plans) == 1:
            return plans[0]
        elif len(plans) == 2:
            # Assume hist/sim comparison pattern
            join_key = 'period' if any(isinstance(p, Agg) for p in plans) else 'ts'
            return Join(left=plans[0], right=plans[1], join_key=join_key)
        else:
            return Union(children_ops=plans)


# ============================================================================
# TwinQL Compiler
# ============================================================================

class TwinQLCompiler:
    """
    Full TwinQL compilation pipeline:
    1. Parse text → AST
    2. AST → Logical Plan  
    3. Logical Plan → Optimized Plan
    4. Optimized Plan → Execution
    """
    
    def __init__(self):
        self.optimizer = Optimizer()
        self.sim_manager = SimulationManager()
        self._register_default_models()
    
    def _register_default_models(self):
        """Register built-in simulation models"""
        try:
            from sim.models.building_heat import building_heat_model
            self.sim_manager.register_model('BuildingHeat_v3', building_heat_model)
        except ImportError:
            pass
    
    def compile(self, query: str) -> LogicalOp:
        """Compile TwinQL text to optimized logical plan"""
        ast = parse_twinql(query)
        
        if isinstance(ast, ScenarioDefAST):
            self._handle_scenario_def(ast)
            return None
        elif isinstance(ast, TwinSelectAST):
            plan = PlanBuilder.from_twin_select(ast)
            return self.optimizer.optimize(plan)
        else:
            raise ValueError(f"Unknown AST type: {type(ast)}")
    
    def _handle_scenario_def(self, ast: ScenarioDefAST):
        """Register scenario definition in database"""
        from dsl.twinql import define_scenario
        define_scenario(ast.name, **ast.config)
    
    def compile_twin_select(self, 
                            hist_spec: Dict,
                            sim_spec: Dict,
                            agg_by: str = None,
                            agg_func: str = 'sum') -> LogicalOp:
        """Compile from dict specs (programmatic API)"""
        hist_op = Hist(
            tid=hist_spec['tid'],
            window=(hist_spec['window_start'], hist_spec['window_end']),
            metrics=hist_spec.get('metrics', ['heat_load'])
        )
        
        sim_op = Sim(
            tid=sim_spec['tid'],
            scenario_id=sim_spec['scenario'],
            model_id=sim_spec['model'],
            window=(sim_spec['window_start'], sim_spec['window_end']),
            metrics=sim_spec.get('metrics', ['heat_load'])
        )
        
        if agg_by:
            hist_op = Agg(child=hist_op, group_by=agg_by, agg_funcs={'value': agg_func})
            sim_op = Agg(child=sim_op, group_by=agg_by, agg_funcs={'value': agg_func})
        
        join_key = 'period' if agg_by else 'ts'
        return Join(left=hist_op, right=sim_op, join_key=join_key)
    
    def compile_multi_twin_select(self,
                                   tids: List[str],
                                   scenario_id: str,
                                   model_id: str,
                                   window: Tuple[str, str],
                                   metrics: List[str],
                                   agg_by: str = 'year') -> List[LogicalOp]:
        """Compile multi-twin query with cache-aware optimization"""
        cached_tids, uncached_tids = optimize_multi_twin_query(
            tids, scenario_id, model_id, window, metrics
        )
        
        plans = []
        for tid in tids:
            hist_op = Hist(tid=tid, window=window, metrics=metrics)
            
            if tid in cached_tids:
                sig = CacheManager.compute_signature(tid, scenario_id, model_id, window, metrics)
                sim_op = ScanCache(cache_sig=sig, tid=tid, scenario_id=scenario_id,
                                   window=window, metrics=metrics)
            else:
                sim_op = Sim(tid=tid, scenario_id=scenario_id, model_id=model_id,
                            window=window, metrics=metrics)
            
            if agg_by:
                hist_op = Agg(child=hist_op, group_by=agg_by, agg_funcs={'value': 'sum'})
                sim_op = Agg(child=sim_op, group_by=agg_by, agg_funcs={'value': 'sum'})
            
            plans.append(Join(left=hist_op, right=sim_op, join_key='period' if agg_by else 'ts'))
        
        return plans
    
    def optimize(self, plan: LogicalOp) -> LogicalOp:
        """Apply optimizer to plan"""
        return self.optimizer.optimize(plan)
    
    def execute(self, plan: LogicalOp) -> pd.DataFrame:
        """Execute an optimized plan"""
        return self._execute_op(plan)
    
    def _execute_op(self, op: LogicalOp) -> pd.DataFrame:
        """Recursive plan execution"""
        if isinstance(op, Hist):
            return self._exec_hist(op)
        elif isinstance(op, Sim):
            return self._exec_sim(op)
        elif isinstance(op, ScanCache):
            return self._exec_cache_scan(op)
        elif isinstance(op, SimBatch):
            return self._exec_sim_batch(op)
        elif isinstance(op, Agg):
            return self._exec_agg(self._execute_op(op.child), op)
        elif isinstance(op, Join):
            return self._exec_join(self._execute_op(op.left), self._execute_op(op.right), op)
        elif isinstance(op, Project):
            return self._exec_project(self._execute_op(op.child), op)
        elif isinstance(op, Union):
            dfs = [self._execute_op(c) for c in op.children_ops]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            raise ValueError(f"Unknown operator: {type(op)}")
    
    def _exec_hist(self, op: Hist) -> pd.DataFrame:
        """Execute Hist - push to SQL"""
        metrics_str = ','.join([f"'{m}'" for m in op.metrics])
        sql = f"""
            SELECT tid, ts, metric, value
            FROM timeseries
            WHERE tid = %s AND cid = 'REALITY'
            AND metric IN ({metrics_str})
            AND ts >= %s AND ts < %s
            ORDER BY ts
        """
        return execute_query(sql, (op.tid, op.window[0], op.window[1]))
    
    def _exec_sim(self, op: Sim) -> pd.DataFrame:
        """Execute Sim - invoke simulation"""
        self.sim_manager.simulate(
            [op.tid], op.scenario_id, op.model_id, op.window, op.metrics
        )
        metrics_str = ','.join([f"'{m}'" for m in op.metrics])
        sql = f"""
            SELECT tid, ts, metric, value
            FROM timeseries
            WHERE tid = %s AND cid = %s
            AND metric IN ({metrics_str})
            AND ts >= %s AND ts < %s
            ORDER BY ts
        """
        return execute_query(sql, (op.tid, op.scenario_id, op.window[0], op.window[1]))
    
    def _exec_cache_scan(self, op: ScanCache) -> pd.DataFrame:
        """Execute ScanCache - read cached results"""
        metrics_str = ','.join([f"'{m}'" for m in op.metrics])
        sql = f"""
            SELECT tid, ts, metric, value
            FROM timeseries
            WHERE tid = %s AND cid = %s
            AND metric IN ({metrics_str})
            AND ts >= %s AND ts < %s
            ORDER BY ts
        """
        return execute_query(sql, (op.tid, op.scenario_id, op.window[0], op.window[1]))
    
    def _exec_sim_batch(self, op: SimBatch) -> pd.DataFrame:
        """Execute SimBatch - batch simulation"""
        self.sim_manager.simulate(
            op.tids, op.scenario_id, op.model_id, op.window, op.metrics
        )
        metrics_str = ','.join([f"'{m}'" for m in op.metrics])
        tids_str = ','.join([f"'{t}'" for t in op.tids])
        sql = f"""
            SELECT tid, ts, metric, value
            FROM timeseries
            WHERE tid IN ({tids_str}) AND cid = %s
            AND metric IN ({metrics_str})
            AND ts >= %s AND ts < %s
            ORDER BY tid, ts
        """
        return execute_query(sql, (op.scenario_id, op.window[0], op.window[1]))
    
    def _exec_agg(self, df: pd.DataFrame, op: Agg) -> pd.DataFrame:
        """Execute Agg"""
        if df.empty:
            return df
        
        period_map = {'year': 'Y', 'month': 'M', 'day': 'D', 'hour': 'H'}
        period_code = period_map.get(op.group_by, 'M')
        df['period'] = df['ts'].dt.to_period(period_code).dt.to_timestamp()
        
        agg_dict = {col: func for col, func in op.agg_funcs.items()}
        return df.groupby(['tid', 'period', 'metric']).agg(agg_dict).reset_index()
    
    def _exec_join(self, left: pd.DataFrame, right: pd.DataFrame, op: Join) -> pd.DataFrame:
        """Execute Join"""
        if left.empty or right.empty:
            return pd.DataFrame()
        return left.merge(right, on=op.join_key, suffixes=('_baseline', '_retrofit'))
    
    def _exec_project(self, df: pd.DataFrame, op: Project) -> pd.DataFrame:
        """Execute Project"""
        result = df[op.columns].copy()
        for name, expr in op.computed.items():
            result[name] = eval(expr, {"df": df})
        return result
    
    def explain(self, plan: LogicalOp) -> str:
        """Generate EXPLAIN output"""
        return self.optimizer.explain(plan)


# ============================================================================
# High-Level Engine API
# ============================================================================

class TwinQLEngine:
    """
    High-level TwinQL execution engine.
    Provides simple API for common query patterns (W1, W2 workloads).
    """
    
    def __init__(self):
        self.compiler = TwinQLCompiler()
    
    def execute_query(self, query: str) -> Tuple[pd.DataFrame, str]:
        """Execute a TwinQL query string"""
        plan = self.compiler.compile(query)
        if plan is None:
            return pd.DataFrame(), "Scenario defined"
        
        opt_plan = self.compiler.optimize(plan)
        explain = self.compiler.explain(opt_plan)
        result = self.compiler.execute(opt_plan)
        return result, explain
    
    def compare_scenario(self, tid: str, scenario_id: str, model_id: str,
                         window: Tuple[str, str], metric: str = 'heat_load',
                         agg_by: str = 'month') -> Tuple[pd.DataFrame, str]:
        """
        W1 Pattern: Single-building scenario comparison
        Compares historical vs simulated energy under a retrofit scenario
        """
        plan = self.compiler.compile_twin_select(
            hist_spec={'tid': tid, 'window_start': window[0], 'window_end': window[1], 'metrics': [metric]},
            sim_spec={'tid': tid, 'scenario': scenario_id, 'model': model_id,
                     'window_start': window[0], 'window_end': window[1], 'metrics': [metric]},
            agg_by=agg_by,
            agg_func='sum'
        )
        
        opt_plan = self.compiler.optimize(plan)
        explain = self.compiler.explain(opt_plan)
        result = self.compiler.execute(opt_plan)
        
        if not result.empty and 'value_baseline' in result.columns:
            result['saving'] = result['value_baseline'] - result['value_retrofit']
            result['saving_pct'] = (result['saving'] / result['value_baseline'] * 100).round(2)
        
        return result, explain
    
    def compare_multi_twin(self, tids: List[str], scenario_id: str, model_id: str,
                           window: Tuple[str, str], metric: str = 'heat_load',
                           agg_by: str = 'year') -> Tuple[List[pd.DataFrame], str]:
        """
        W2 Pattern: Multi-building network analysis
        Compares multiple buildings, identifies worst performers
        """
        plans = self.compiler.compile_multi_twin_select(
            tids=tids, scenario_id=scenario_id, model_id=model_id,
            window=window, metrics=[metric], agg_by=agg_by
        )
        
        results = []
        for i, plan in enumerate(plans):
            opt_plan = self.compiler.optimize(plan)
            result = self.compiler.execute(opt_plan)
            if not result.empty and 'value_baseline' in result.columns:
                result['saving'] = result['value_baseline'] - result['value_retrofit']
                result['tid'] = tids[i]
            results.append(result)
        
        explain = f"Multi-twin query: {len(tids)} twins, stats: {self.compiler.optimizer.stats}"
        return results, explain
    
    def find_worst_performers(self, tids: List[str], scenario_id: str, model_id: str,
                              window: Tuple[str, str], metric: str = 'heat_load',
                              percentile: float = 0.2) -> pd.DataFrame:
        """
        W2 Extension: Find worst-performing buildings (highest energy use)
        Returns buildings in the top percentile of energy consumption
        """
        results, _ = self.compare_multi_twin(tids, scenario_id, model_id, window, metric, 'year')
        
        # Aggregate results
        summary = []
        for r in results:
            if not r.empty and 'value_baseline' in r.columns:
                summary.append({
                    'tid': r['tid'].iloc[0],
                    'baseline_total': r['value_baseline'].sum(),
                    'retrofit_total': r['value_retrofit'].sum(),
                    'saving_total': r['saving'].sum()
                })
        
        if not summary:
            return pd.DataFrame()
        
        df = pd.DataFrame(summary)
        threshold = df['baseline_total'].quantile(1 - percentile)
        return df[df['baseline_total'] >= threshold].sort_values('baseline_total', ascending=False)
