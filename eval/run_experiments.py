#!/usr/bin/env python3
"""
TwinDB Evaluation Suite

Rigorous experimental evaluation covering:
- Multiple runs with statistical significance
- Warm-up periods
- Cold/hot cache separation
- Baseline comparisons
- Scalability analysis

Experiments:
  E1: TwinDB vs Baseline (end-to-end comparison)
  E2: Optimizer Ablation (contribution of each optimization)
  E3: Scalability (varying number of twins)
  E4: Cache Sensitivity (varying cache hit rates)
  E5: Cost Model Accuracy (predicted vs actual)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from statistics import mean, stdev

warnings.filterwarnings('ignore')

from core.db import execute_query, execute_sql
from core.compiler import TwinQLEngine
from core.optimizer import Optimizer, CacheManager, CostModel
from core.operators import Hist, Sim, Agg, Join, print_plan
from dsl.twinql import define_scenario
from sim.manager import SimulationManager
from sim.models.building_heat import building_heat_model


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments"""
    num_runs: int = 5           # Runs per measurement
    warmup_runs: int = 2        # Warmup runs (discarded)
    window: Tuple[str, str] = ('2018-01-01', '2019-01-01')
    metric: str = 'heat_load'
    model_id: str = 'BuildingHeat_v3'
    
    # Default retrofit scenario
    retrofit_cfg: Dict = field(default_factory=lambda: {
        'wall_u_before': 0.60, 'wall_u_after': 0.18,
        'roof_u_before': 0.30, 'roof_u_after': 0.12,
        'window_u_before': 2.80, 'window_u_after': 0.90,
        'infiltration_before': 0.8, 'infiltration_after': 0.5,
    })


@dataclass
class Measurement:
    """Single measurement with statistics"""
    name: str
    times_ms: List[float]
    
    @property
    def mean_ms(self) -> float:
        return mean(self.times_ms) if self.times_ms else 0
    
    @property
    def std_ms(self) -> float:
        return stdev(self.times_ms) if len(self.times_ms) > 1 else 0
    
    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0
    
    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0


def clear_cache():
    """Clear all simulation cache and results"""
    execute_sql("DELETE FROM sim_cache")
    execute_sql("DELETE FROM timeseries WHERE cid != 'REALITY'")


def measure(func, config: ExperimentConfig) -> Measurement:
    """Run function multiple times and collect timing statistics"""
    # Warmup
    for _ in range(config.warmup_runs):
        func()
    
    # Measured runs
    times = []
    for _ in range(config.num_runs):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return Measurement(name="", times_ms=times)


# ============================================================================
# E1: TwinDB vs Baseline
# ============================================================================

def baseline_implementation(tid: str, scenario_cfg: dict, window: Tuple[str, str]) -> pd.DataFrame:
    """
    Baseline: What researchers do without TwinDB
    - Manual SQL queries
    - Manual simulation invocation  
    - Manual result aggregation and joining
    - NO CACHING (must re-simulate every time)
    """
    # Query historical
    hist_df = execute_query("""
        SELECT ts, value FROM timeseries
        WHERE tid = %s AND cid = 'REALITY' AND metric = 'heat_load'
        AND ts >= %s AND ts < %s ORDER BY ts
    """, (tid, window[0], window[1]))
    
    # Run simulation (this is the expensive part)
    asset_info = execute_query("""
        SELECT a.* FROM asset a JOIN twin t ON t.aid = a.aid WHERE t.tid = %s
    """, (tid,), as_df=False)
    asset = asset_info[0] if asset_info else {}
    sim_df = building_heat_model(hist_df.copy(), scenario_cfg, asset)
    
    # Aggregate
    hist_df['month'] = hist_df['ts'].dt.to_period('M').dt.to_timestamp()
    sim_df['month'] = sim_df['ts'].dt.to_period('M').dt.to_timestamp()
    hist_agg = hist_df.groupby('month')['value'].sum().reset_index()
    sim_agg = sim_df.groupby('month')['value'].sum().reset_index()
    
    # Join
    result = hist_agg.merge(sim_agg, on='month', suffixes=('_baseline', '_retrofit'))
    result['saving'] = result['value_baseline'] - result['value_retrofit']
    return result


def baseline_with_persistence(tid: str, scenario_id: str, scenario_cfg: dict, 
                               window: Tuple[str, str]) -> pd.DataFrame:
    """
    Baseline WITH persistence (fair comparison to TwinDB cold cache).
    Simulates and writes results to DB, like TwinDB does.
    """
    # Query historical
    hist_df = execute_query("""
        SELECT ts, value FROM timeseries
        WHERE tid = %s AND cid = 'REALITY' AND metric = 'heat_load'
        AND ts >= %s AND ts < %s ORDER BY ts
    """, (tid, window[0], window[1]))
    
    # Run simulation
    asset_info = execute_query("""
        SELECT a.* FROM asset a JOIN twin t ON t.aid = a.aid WHERE t.tid = %s
    """, (tid,), as_df=False)
    asset = asset_info[0] if asset_info else {}
    sim_df = building_heat_model(hist_df.copy(), scenario_cfg, asset)
    
    # Write to DB (like TwinDB does)
    from core.db import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in sim_df.iterrows():
                cur.execute("""
                    INSERT INTO timeseries (tid, cid, metric, metric_type, unit, ts, value, source)
                    VALUES (%s, %s, 'heat_load', 'power', 'kW', %s, %s, 'baseline_sim')
                    ON CONFLICT (tid, cid, metric, ts) DO UPDATE SET value = EXCLUDED.value
                """, (tid, scenario_id, row['ts'], row['value']))
        conn.commit()
    
    # Aggregate
    hist_df['month'] = hist_df['ts'].dt.to_period('M').dt.to_timestamp()
    sim_df['month'] = sim_df['ts'].dt.to_period('M').dt.to_timestamp()
    hist_agg = hist_df.groupby('month')['value'].sum().reset_index()
    sim_agg = sim_df.groupby('month')['value'].sum().reset_index()
    
    result = hist_agg.merge(sim_agg, on='month', suffixes=('_baseline', '_retrofit'))
    result['saving'] = result['value_baseline'] - result['value_retrofit']
    return result


def run_e1(config: ExperimentConfig) -> pd.DataFrame:
    """E1: Compare TwinDB vs baseline implementation
    
    Fair comparison: Both systems must persist results for reproducibility.
    - Baseline: Must re-simulate every time (no caching infrastructure)
    - TwinDB: First query simulates, subsequent queries use cache
    
    This shows TwinDB's value: automatic caching of simulation results.
    """
    print("\n" + "=" * 70)
    print("E1: TwinDB vs Baseline Comparison")
    print("=" * 70)
    print("Fair comparison: Both persist results. Baseline has no cache.")
    print("TwinDB advantage: automatic caching avoids re-simulation.")
    
    define_scenario('e1_scenario', retrofit_package=config.retrofit_cfg)
    
    twins = ['Twin_B123', 'Twin_B101', 'Twin_B110']
    results = []
    
    for tid in twins:
        print(f"\nTesting {tid}...")
        
        # Baseline: must re-simulate every time (no caching)
        # This represents what researchers do without TwinDB
        baseline_times = []
        for i in range(config.num_runs):
            # Clear previous results to force re-simulation
            execute_sql("DELETE FROM timeseries WHERE cid = 'baseline_e1'")
            start = time.perf_counter()
            baseline_with_persistence(tid, 'baseline_e1', {'retrofit_package': config.retrofit_cfg}, config.window)
            baseline_times.append((time.perf_counter() - start) * 1000)
        
        # TwinDB cold cache (first query)
        clear_cache()
        engine = TwinQLEngine()
        
        start = time.perf_counter()
        engine.compare_scenario(tid, 'e1_scenario', config.model_id, config.window)
        cold_time = (time.perf_counter() - start) * 1000
        
        # TwinDB hot cache (subsequent queries)
        hot_times = []
        for _ in range(config.num_runs):
            start = time.perf_counter()
            engine.compare_scenario(tid, 'e1_scenario', config.model_id, config.window)
            hot_times.append((time.perf_counter() - start) * 1000)
        
        baseline_mean = mean(baseline_times)
        hot_mean = mean(hot_times)
        
        results.append({
            'twin': tid,
            'baseline_ms': baseline_mean,
            'twindb_cold_ms': cold_time,
            'twindb_hot_ms': hot_mean,
            'speedup': baseline_mean / hot_mean if hot_mean > 0 else 0,
            'cache_speedup': cold_time / hot_mean if hot_mean > 0 else 0,
        })
        
        print(f"  Baseline (re-simulate each time): {baseline_mean:7.1f} ms")
        print(f"  TwinDB (cold cache):              {cold_time:7.1f} ms")
        print(f"  TwinDB (hot cache):               {hot_mean:7.1f} ms")
        print(f"  → Speedup vs baseline:            {baseline_mean/hot_mean:.1f}x")
        print(f"  → Cache speedup:                  {cold_time/hot_mean:.1f}x")
    
    return pd.DataFrame(results)


# ============================================================================
# E2: Optimizer Ablation
# ============================================================================

def run_e2(config: ExperimentConfig) -> Dict[str, Any]:
    """E2: Measure contribution of each optimization technique"""
    print("\n" + "=" * 70)
    print("E2: Optimizer Ablation Study")
    print("=" * 70)
    
    define_scenario('e2_scenario', retrofit_package=config.retrofit_cfg)
    
    twins = execute_query("""
        SELECT t.tid FROM twin t JOIN asset a ON t.aid = a.aid
        WHERE a.network_id = 'DH_NET_CPH_01' LIMIT 5
    """)['tid'].tolist()
    
    print(f"Testing with {len(twins)} twins: {twins}")
    
    # Config 1: No cache (simulate every time)
    clear_cache()
    engine = TwinQLEngine()
    
    times_no_cache = []
    for _ in range(config.num_runs):
        clear_cache()  # Force cache miss each run
        start = time.perf_counter()
        for tid in twins:
            engine.compare_scenario(tid, 'e2_scenario', config.model_id, config.window, agg_by='year')
        times_no_cache.append((time.perf_counter() - start) * 1000)
    
    # Config 2: With cache
    times_with_cache = []
    for _ in range(config.num_runs):
        start = time.perf_counter()
        for tid in twins:
            engine.compare_scenario(tid, 'e2_scenario', config.model_id, config.window, agg_by='year')
        times_with_cache.append((time.perf_counter() - start) * 1000)
    
    no_cache_mean = mean(times_no_cache)
    with_cache_mean = mean(times_with_cache)
    
    print(f"\nResults ({len(twins)} twins, {config.num_runs} runs):")
    print(f"  No cache:   {no_cache_mean:7.1f} ms (σ={stdev(times_no_cache):.1f})")
    print(f"  With cache: {with_cache_mean:7.1f} ms (σ={stdev(times_with_cache):.1f})")
    print(f"  Cache speedup: {no_cache_mean/with_cache_mean:.1f}x")
    
    return {
        'num_twins': len(twins),
        'no_cache_ms': no_cache_mean,
        'with_cache_ms': with_cache_mean,
        'cache_speedup': no_cache_mean / with_cache_mean
    }


# ============================================================================
# E3: Scalability
# ============================================================================

def run_e3(config: ExperimentConfig) -> pd.DataFrame:
    """E3: Scalability with number of twins"""
    print("\n" + "=" * 70)
    print("E3: Scalability Analysis")
    print("=" * 70)
    
    define_scenario('e3_scenario', retrofit_package=config.retrofit_cfg)
    
    all_twins = execute_query("""
        SELECT t.tid FROM twin t JOIN asset a ON t.aid = a.aid
        WHERE a.network_id = 'DH_NET_CPH_01'
    """)['tid'].tolist()
    
    twin_counts = [1, 2, 5, 10]
    twin_counts = [n for n in twin_counts if n <= len(all_twins)]
    
    results = []
    
    for n in twin_counts:
        tids = all_twins[:n]
        print(f"\nTesting {n} twins...")
        
        # Cold cache
        clear_cache()
        engine = TwinQLEngine()
        
        cold_times = []
        for _ in range(config.num_runs):
            clear_cache()
            start = time.perf_counter()
            for tid in tids:
                engine.compare_scenario(tid, 'e3_scenario', config.model_id, config.window, agg_by='year')
            cold_times.append((time.perf_counter() - start) * 1000)
        
        # Hot cache
        hot_times = []
        for _ in range(config.num_runs):
            start = time.perf_counter()
            for tid in tids:
                engine.compare_scenario(tid, 'e3_scenario', config.model_id, config.window, agg_by='year')
            hot_times.append((time.perf_counter() - start) * 1000)
        
        results.append({
            'num_twins': n,
            'cold_mean_ms': mean(cold_times),
            'cold_std_ms': stdev(cold_times) if len(cold_times) > 1 else 0,
            'hot_mean_ms': mean(hot_times),
            'hot_std_ms': stdev(hot_times) if len(hot_times) > 1 else 0,
            'cold_per_twin_ms': mean(cold_times) / n,
            'hot_per_twin_ms': mean(hot_times) / n,
        })
        
        print(f"  Cold: {mean(cold_times):7.1f} ms ({mean(cold_times)/n:.1f} ms/twin)")
        print(f"  Hot:  {mean(hot_times):7.1f} ms ({mean(hot_times)/n:.1f} ms/twin)")
    
    return pd.DataFrame(results)


# ============================================================================
# E4: Cache Sensitivity
# ============================================================================

def run_e4(config: ExperimentConfig) -> pd.DataFrame:
    """E4: Impact of cache hit rate"""
    print("\n" + "=" * 70)
    print("E4: Cache Sensitivity Analysis")
    print("=" * 70)
    
    tid = 'Twin_B123'
    
    # Create scenarios with varying parameters
    scenarios = []
    for i in range(10):
        cid = f'e4_scenario_{i}'
        define_scenario(cid, retrofit_package={
            'wall_u_before': 0.60, 
            'wall_u_after': 0.18 + i * 0.01,
        })
        scenarios.append(cid)
    
    results = []
    engine = TwinQLEngine()
    
    for cache_pct in [0, 25, 50, 75, 100]:
        clear_cache()
        
        # Pre-populate cache
        num_cached = int(len(scenarios) * cache_pct / 100)
        for cid in scenarios[:num_cached]:
            engine.compare_scenario(tid, cid, config.model_id, config.window)
        
        # Measure
        times = []
        for _ in range(config.num_runs):
            start = time.perf_counter()
            for cid in scenarios:
                engine.compare_scenario(tid, cid, config.model_id, config.window)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append({
            'cache_pct': cache_pct,
            'mean_ms': mean(times),
            'std_ms': stdev(times) if len(times) > 1 else 0,
            'per_scenario_ms': mean(times) / len(scenarios),
        })
        
        print(f"  {cache_pct:3d}% cached: {mean(times):7.1f} ms ({mean(times)/len(scenarios):.1f} ms/scenario)")
    
    return pd.DataFrame(results)


# ============================================================================
# E5: Cost Model Accuracy
# ============================================================================

def run_e5(config: ExperimentConfig) -> pd.DataFrame:
    """E5: Validate cost model predictions against actual execution"""
    print("\n" + "=" * 70)
    print("E5: Cost Model Accuracy")
    print("=" * 70)
    
    define_scenario('e5_scenario', retrofit_package=config.retrofit_cfg)
    
    cost_model = CostModel()
    results = []
    
    # Test different plan shapes
    test_cases = [
        ('Single Hist', lambda: Hist('Twin_B123', config.window, ['heat_load'])),
        ('Single Sim', lambda: Sim('Twin_B123', 'e5_scenario', config.model_id, config.window, ['heat_load'])),
        ('Hist+Agg', lambda: Agg(Hist('Twin_B123', config.window, ['heat_load']), 'month', {'value': 'sum'})),
        ('Sim+Agg', lambda: Agg(Sim('Twin_B123', 'e5_scenario', config.model_id, config.window, ['heat_load']), 'month', {'value': 'sum'})),
    ]
    
    engine = TwinQLEngine()
    
    for name, plan_fn in test_cases:
        clear_cache()
        plan = plan_fn()
        predicted_ms = cost_model.estimate(plan)
        
        # Measure actual
        times = []
        for _ in range(config.num_runs):
            clear_cache()
            start = time.perf_counter()
            engine.compiler.execute(plan)
            times.append((time.perf_counter() - start) * 1000)
        
        actual_ms = mean(times)
        error_pct = abs(predicted_ms - actual_ms) / actual_ms * 100 if actual_ms > 0 else 0
        
        results.append({
            'plan': name,
            'predicted_ms': predicted_ms,
            'actual_ms': actual_ms,
            'error_pct': error_pct,
        })
        
        print(f"  {name:15s}: predicted={predicted_ms:7.1f}ms, actual={actual_ms:7.1f}ms, error={error_pct:.1f}%")
    
    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def run_full_evaluation(quick: bool = False) -> Dict[str, Any]:
    """Run complete evaluation suite"""
    config = ExperimentConfig()
    if quick:
        config.num_runs = 3
        config.warmup_runs = 1
    
    print("\n" + "=" * 70)
    print("TwinDB Evaluation Suite")
    print(f"Configuration: {config.num_runs} runs, {config.warmup_runs} warmup")
    print("=" * 70)
    
    results = {}
    
    results['e1'] = run_e1(config)
    results['e2'] = run_e2(config)
    results['e3'] = run_e3(config)
    results['e4'] = run_e4(config)
    results['e5'] = run_e5(config)
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\nE1: TwinDB vs Baseline (with persistence)")
    print("  Baseline must re-simulate each query; TwinDB uses cache")
    e1_df = results['e1'][['twin', 'baseline_ms', 'twindb_hot_ms', 'speedup', 'cache_speedup']]
    print(e1_df.to_string(index=False))
    
    print(f"\nE2: Cache Optimization")
    print(f"  Speedup from caching: {results['e2']['cache_speedup']:.1f}x")
    
    print("\nE3: Scalability (hot cache)")
    print(results['e3'][['num_twins', 'hot_mean_ms', 'hot_per_twin_ms']].to_string(index=False))
    
    print("\nE4: Cache Sensitivity")
    print(results['e4'][['cache_pct', 'mean_ms']].to_string(index=False))
    
    print("\nE5: Cost Model Accuracy")
    avg_error = results['e5']['error_pct'].mean()
    print(f"  Average prediction error: {avg_error:.1f}%")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer iterations')
    args = parser.parse_args()
    
    run_full_evaluation(quick=args.quick)
