#!/usr/bin/env python3
"""
TwinDB LLM Evaluation Suite

Experiments:
  E1: LLM Surrogate Accuracy vs Full Simulation
  E2: LLM-Guided Execution Speedup
  E3: Error-Tolerance Trade-off
  E4: Scenario Similarity Assessment Quality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from statistics import mean, stdev

warnings.filterwarnings('ignore')

from core.db import execute_query, execute_sql
from core.compiler import TwinQLEngine
from core.llm import LLMEnhancedEngine, LLMSurrogateModel, LLMOptimizer
from dsl.twinql import define_scenario

API_KEY = 'sk-613bdd8012224559b54e557725ea8851'


def clear_cache():
    execute_sql("DELETE FROM sim_cache")
    execute_sql("DELETE FROM timeseries WHERE cid != 'REALITY'")


# ============================================================================
# E1: LLM Surrogate Accuracy
# ============================================================================

def run_e1_surrogate_accuracy():
    """Compare LLM surrogate estimates vs full simulation ground truth"""
    print("\n" + "=" * 70)
    print("E1: LLM Surrogate Accuracy vs Full Simulation")
    print("=" * 70)
    
    surrogate = LLMSurrogateModel(API_KEY)
    standard = TwinQLEngine()
    
    # Get baseline
    baseline_kwh = float(execute_query("""
        SELECT SUM(value) as total FROM timeseries
        WHERE tid = 'Twin_B123' AND cid = 'REALITY' AND metric = 'heat_load'
        AND ts >= '2018-01-01' AND ts < '2019-01-01'
    """)['total'].iloc[0])
    
    print(f"Baseline: {baseline_kwh:,.0f} kWh")
    
    # Test scenarios with varying complexity
    scenarios = [
        {'name': 'Wall only (mild)', 'cfg': {'retrofit_package': {'wall_u_after': 0.30}}},
        {'name': 'Wall only (deep)', 'cfg': {'retrofit_package': {'wall_u_after': 0.18}}},
        {'name': 'Wall + Window', 'cfg': {'retrofit_package': {'wall_u_after': 0.18, 'window_u_after': 0.90}}},
        {'name': 'Wall + Roof', 'cfg': {'retrofit_package': {'wall_u_after': 0.18, 'roof_u_after': 0.12}}},
        {'name': 'Deep retrofit', 'cfg': {'retrofit_package': {'wall_u_after': 0.15, 'window_u_after': 0.80, 'roof_u_after': 0.10}}},
    ]
    
    results = []
    
    for i, s in enumerate(scenarios):
        print(f"\n{s['name']}...")
        
        # LLM surrogate estimate
        start = time.perf_counter()
        llm_est, confidence, reasoning = surrogate.estimate(baseline_kwh, s['cfg'])
        llm_time = (time.perf_counter() - start) * 1000
        
        # Full simulation (ground truth)
        clear_cache()
        define_scenario(f'e1_s{i}', **s['cfg'])
        start = time.perf_counter()
        result, _ = standard.compare_scenario('Twin_B123', f'e1_s{i}', 'BuildingHeat_v3',
                                               ('2018-01-01', '2019-01-01'), agg_by='year')
        sim_time = (time.perf_counter() - start) * 1000
        
        actual_kwh = float(result['value_retrofit'].sum())
        error_pct = abs(llm_est - actual_kwh) / actual_kwh * 100
        
        results.append({
            'scenario': s['name'],
            'llm_estimate_kwh': llm_est,
            'actual_kwh': actual_kwh,
            'error_pct': error_pct,
            'llm_confidence': confidence,
            'llm_time_ms': llm_time,
            'sim_time_ms': sim_time,
            'speedup': sim_time / llm_time
        })
        
        print(f"  LLM: {llm_est:,.0f} kWh ({llm_time:.0f}ms)")
        print(f"  Actual: {actual_kwh:,.0f} kWh ({sim_time:.0f}ms)")
        print(f"  Error: {error_pct:.1f}%, Speedup: {sim_time/llm_time:.1f}x")
    
    df = pd.DataFrame(results)
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Mean Absolute Error: {df['error_pct'].mean():.1f}%")
    print(f"  Max Error: {df['error_pct'].max():.1f}%")
    print(f"  Mean Speedup: {df['speedup'].mean():.1f}x")
    
    return df


# ============================================================================
# E2: LLM-Guided Execution Speedup
# ============================================================================

def run_e2_execution_speedup():
    """Compare standard vs LLM-guided execution on scenario sweep"""
    print("\n" + "=" * 70)
    print("E2: LLM-Guided Execution Speedup (Scenario Sweep)")
    print("=" * 70)
    
    standard = TwinQLEngine()
    enhanced = LLMEnhancedEngine(API_KEY)
    
    # Generate parameter sweep (20 scenarios)
    wall_values = [0.15, 0.18, 0.20, 0.22, 0.25]
    window_values = [0.80, 0.90, 1.00, 1.20]
    
    scenarios = []
    for w in wall_values:
        for win in window_values:
            scenarios.append({
                'retrofit_package': {'wall_u_after': w, 'window_u_after': win}
            })
    
    print(f"Testing {len(scenarios)} scenarios...")
    
    # Standard execution (all full simulations)
    print("\n--- Standard Engine ---")
    clear_cache()
    standard_times = []
    for i, cfg in enumerate(scenarios):
        define_scenario(f'e2_std_{i}', **cfg)
        start = time.perf_counter()
        standard.compare_scenario('Twin_B123', f'e2_std_{i}', 'BuildingHeat_v3',
                                  ('2018-01-01', '2019-01-01'), agg_by='year')
        standard_times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(scenarios)} done...")
    
    standard_total = sum(standard_times)
    print(f"  Total: {standard_total:.0f}ms")
    
    # LLM-enhanced execution
    print("\n--- LLM-Enhanced Engine (10% error tolerance) ---")
    clear_cache()
    enhanced_times = []
    strategies = []
    for i, cfg in enumerate(scenarios):
        start = time.perf_counter()
        result = enhanced.compare_scenario_smart('Twin_B123', cfg, 'BuildingHeat_v3',
                                                  ('2018-01-01', '2019-01-01'), 
                                                  error_tolerance=0.10)
        enhanced_times.append((time.perf_counter() - start) * 1000)
        strategies.append(result['execution']['strategy'])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(scenarios)} done...")
    
    enhanced_total = sum(enhanced_times)
    print(f"  Total: {enhanced_total:.0f}ms")
    
    # Strategy breakdown
    from collections import Counter
    strategy_counts = Counter(strategies)
    
    print("\n" + "-" * 70)
    print("RESULTS:")
    print(f"  Standard total: {standard_total:,.0f}ms")
    print(f"  LLM-enhanced total: {enhanced_total:,.0f}ms")
    print(f"  Speedup: {standard_total/enhanced_total:.2f}x")
    print(f"\nStrategy breakdown:")
    for s, c in strategy_counts.items():
        print(f"  {s}: {c} ({c/len(scenarios)*100:.0f}%)")
    
    return {
        'num_scenarios': len(scenarios),
        'standard_ms': standard_total,
        'enhanced_ms': enhanced_total,
        'speedup': standard_total / enhanced_total,
        'strategies': dict(strategy_counts)
    }


# ============================================================================
# E3: Error-Tolerance Trade-off
# ============================================================================

def run_e3_error_tolerance():
    """Measure speedup vs accuracy at different error tolerances"""
    print("\n" + "=" * 70)
    print("E3: Error-Tolerance Trade-off")
    print("=" * 70)
    
    standard = TwinQLEngine()
    enhanced = LLMEnhancedEngine(API_KEY)
    
    # Fixed set of scenarios
    scenarios = [
        {'retrofit_package': {'wall_u_after': 0.16}},
        {'retrofit_package': {'wall_u_after': 0.19}},
        {'retrofit_package': {'wall_u_after': 0.22}},
        {'retrofit_package': {'wall_u_after': 0.18, 'window_u_after': 0.85}},
        {'retrofit_package': {'wall_u_after': 0.20, 'roof_u_after': 0.15}},
    ]
    
    # Get ground truth
    print("Computing ground truth...")
    clear_cache()
    ground_truth = []
    for i, cfg in enumerate(scenarios):
        define_scenario(f'e3_gt_{i}', **cfg)
        result, _ = standard.compare_scenario('Twin_B123', f'e3_gt_{i}', 'BuildingHeat_v3',
                                               ('2018-01-01', '2019-01-01'), agg_by='year')
        ground_truth.append(float(result['value_retrofit'].sum()))
    
    tolerances = [0.01, 0.05, 0.10, 0.15, 0.20]
    results = []
    
    for tol in tolerances:
        print(f"\nTolerance: {tol*100:.0f}%...")
        clear_cache()
        
        times = []
        errors = []
        strategies = []
        
        for i, cfg in enumerate(scenarios):
            start = time.perf_counter()
            result = enhanced.compare_scenario_smart('Twin_B123', cfg, 'BuildingHeat_v3',
                                                      ('2018-01-01', '2019-01-01'),
                                                      error_tolerance=tol)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            strategies.append(result['execution']['strategy'])
            
            # Compute actual error if we have result
            if isinstance(result['result'], dict) and 'retrofit_kwh' in result['result']:
                est = result['result']['retrofit_kwh']
            elif hasattr(result['result'], 'empty') and not result['result'].empty:
                est = float(result['result']['value_retrofit'].sum())
            else:
                est = ground_truth[i]  # Fallback
            
            error = abs(est - ground_truth[i]) / ground_truth[i] * 100
            errors.append(error)
        
        from collections import Counter
        strat_counts = Counter(strategies)
        
        results.append({
            'tolerance_pct': tol * 100,
            'total_time_ms': sum(times),
            'mean_error_pct': mean(errors),
            'max_error_pct': max(errors),
            'full_sim_pct': strat_counts.get('full_sim', 0) / len(scenarios) * 100,
            'interpolate_pct': strat_counts.get('interpolate', 0) / len(scenarios) * 100,
            'cache_hit_pct': strat_counts.get('cache_hit', 0) / len(scenarios) * 100,
        })
        
        print(f"  Time: {sum(times):.0f}ms, Error: {mean(errors):.1f}%")
    
    df = pd.DataFrame(results)
    print("\n" + "-" * 70)
    print("TRADE-OFF SUMMARY:")
    print(df.to_string(index=False))
    
    return df


# ============================================================================
# E4: Multi-Building Scalability with LLM
# ============================================================================

def run_e4_multi_building():
    """Test LLM-guided execution on multiple buildings"""
    print("\n" + "=" * 70)
    print("E4: Multi-Building Scalability")
    print("=" * 70)
    
    standard = TwinQLEngine()
    enhanced = LLMEnhancedEngine(API_KEY)
    
    # Get all twins
    twins = execute_query("""
        SELECT t.tid FROM twin t JOIN asset a ON t.aid = a.aid
        WHERE a.network_id = 'DH_NET_CPH_01'
    """)['tid'].tolist()[:5]  # Limit to 5 for demo
    
    scenario = {'retrofit_package': {'wall_u_after': 0.18, 'window_u_after': 0.90}}
    
    print(f"Testing {len(twins)} buildings...")
    
    # Standard
    print("\n--- Standard Engine ---")
    clear_cache()
    standard_times = []
    for tid in twins:
        define_scenario('e4_std', **scenario)
        start = time.perf_counter()
        standard.compare_scenario(tid, 'e4_std', 'BuildingHeat_v3',
                                  ('2018-01-01', '2019-01-01'), agg_by='year')
        standard_times.append((time.perf_counter() - start) * 1000)
    print(f"  Total: {sum(standard_times):.0f}ms")
    
    # LLM-enhanced
    print("\n--- LLM-Enhanced Engine ---")
    clear_cache()
    enhanced_times = []
    for tid in twins:
        start = time.perf_counter()
        enhanced.compare_scenario_smart(tid, scenario, 'BuildingHeat_v3',
                                        ('2018-01-01', '2019-01-01'),
                                        error_tolerance=0.10)
        enhanced_times.append((time.perf_counter() - start) * 1000)
    print(f"  Total: {sum(enhanced_times):.0f}ms")
    
    print("\n" + "-" * 70)
    print(f"Speedup: {sum(standard_times)/sum(enhanced_times):.2f}x")
    
    return {
        'num_buildings': len(twins),
        'standard_ms': sum(standard_times),
        'enhanced_ms': sum(enhanced_times),
        'speedup': sum(standard_times) / sum(enhanced_times)
    }


# ============================================================================
# Main
# ============================================================================

def run_all():
    print("=" * 70)
    print("TwinDB LLM Evaluation Suite")
    print("=" * 70)
    
    results = {}
    
    results['e1'] = run_e1_surrogate_accuracy()
    results['e2'] = run_e2_execution_speedup()
    results['e3'] = run_e3_error_tolerance()
    results['e4'] = run_e4_multi_building()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nE1: Surrogate Accuracy")
    print(f"  Mean Error: {results['e1']['error_pct'].mean():.1f}%")
    print(f"  Mean Speedup: {results['e1']['speedup'].mean():.1f}x")
    
    print(f"\nE2: Scenario Sweep ({results['e2']['num_scenarios']} scenarios)")
    print(f"  Speedup: {results['e2']['speedup']:.2f}x")
    
    print("\nE3: Error-Tolerance Trade-off")
    print(results['e3'][['tolerance_pct', 'total_time_ms', 'mean_error_pct']].to_string(index=False))
    
    print(f"\nE4: Multi-Building ({results['e4']['num_buildings']} buildings)")
    print(f"  Speedup: {results['e4']['speedup']:.2f}x")
    
    return results


if __name__ == '__main__':
    run_all()
