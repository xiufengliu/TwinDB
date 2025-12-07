#!/usr/bin/env python3
"""
TwinDB Benchmark Suite
======================
Evaluation scripts for measuring:
- W1 latency (cold vs hot cache)
- W2 scaling behavior
- Cache hit rates
"""
import sys
sys.path.insert(0, '/Users/xiufengliu/Projects/TwinDB')

import time
import pandas as pd
from core.db import execute_query, execute_sql
from dsl.twinql import TwinQL, HIST, SIM, define_scenario
from sim.models.building_heat import building_heat_model

def clear_cache():
    """Clear simulation cache for fresh benchmarks"""
    execute_sql("DELETE FROM sim_cache")
    execute_sql("DELETE FROM timeseries WHERE cid != 'REALITY'")

def benchmark_w1(iterations: int = 3):
    """Benchmark W1 workload: single building retrofit"""
    print("\n" + "=" * 60)
    print("Benchmark: W1 Single Building Retrofit")
    print("=" * 60)
    
    window = ('2018-01-01', '2019-01-01')
    
    # Setup
    define_scenario(
        cid='s_bench_w1',
        name='Benchmark_W1',
        retrofit_package={
            'wall_u_before': 0.60, 'wall_u_after': 0.18,
            'roof_u_before': 0.30, 'roof_u_after': 0.12,
            'window_u_before': 2.80, 'window_u_after': 0.90,
            'infiltration_before': 0.8, 'infiltration_after': 0.5,
        }
    )
    
    twinql = TwinQL()
    twinql.sim_manager.register_model('BuildingHeat_v3', building_heat_model)
    
    results = {'cold': [], 'hot': []}
    
    for i in range(iterations):
        # Cold cache run
        clear_cache()
        hist = HIST(twin_id='Twin_B123', window=window, metric='heat_load',
                    agg_by='month', agg_func='sum')
        sim = SIM(twin_id='Twin_B123', scenario='s_bench_w1', model='BuildingHeat_v3',
                  window=window, metric='heat_load', agg_by='month', agg_func='sum')
        
        start = time.time()
        twinql.compare(hist, sim)
        results['cold'].append(time.time() - start)
        
        # Hot cache run
        start = time.time()
        twinql.compare(hist, sim)
        results['hot'].append(time.time() - start)
    
    print(f"\nResults over {iterations} iterations:")
    print(f"  Cold cache: {sum(results['cold'])/len(results['cold']):.3f}s avg")
    print(f"  Hot cache:  {sum(results['hot'])/len(results['hot']):.3f}s avg")
    print(f"  Speedup:    {sum(results['cold'])/sum(results['hot']):.1f}x")
    
    return results

def benchmark_w2_scaling(building_counts: list = [2, 5, 10]):
    """Benchmark W2 workload scaling with number of buildings"""
    print("\n" + "=" * 60)
    print("Benchmark: W2 Scaling with Building Count")
    print("=" * 60)
    
    window = ('2018-01-01', '2019-01-01')
    
    define_scenario(
        cid='s_bench_w2',
        name='Benchmark_W2',
        retrofit_package={
            'wall_u_before': 0.60, 'wall_u_after': 0.18,
            'roof_u_before': 0.30, 'roof_u_after': 0.12,
            'window_u_before': 2.80, 'window_u_after': 0.90,
            'infiltration_before': 0.8, 'infiltration_after': 0.5,
        }
    )
    
    twinql = TwinQL()
    twinql.sim_manager.register_model('BuildingHeat_v3', building_heat_model)
    
    # Get all available twins
    twins = execute_query("""
        SELECT t.tid FROM twin t JOIN asset a ON t.aid = a.aid
        WHERE a.network_id = 'DH_NET_CPH_01'
    """)['tid'].tolist()
    
    results = []
    
    for n in building_counts:
        if n > len(twins):
            continue
            
        selected = twins[:n]
        clear_cache()
        
        # Cold cache
        start = time.time()
        for tid in selected:
            hist = HIST(twin_id=tid, window=window, metric='heat_load',
                        agg_by='year', agg_func='sum')
            sim = SIM(twin_id=tid, scenario='s_bench_w2', model='BuildingHeat_v3',
                      window=window, metric='heat_load', agg_by='year', agg_func='sum')
            twinql.compare(hist, sim)
        cold_time = time.time() - start
        
        # Hot cache
        start = time.time()
        for tid in selected:
            hist = HIST(twin_id=tid, window=window, metric='heat_load',
                        agg_by='year', agg_func='sum')
            sim = SIM(twin_id=tid, scenario='s_bench_w2', model='BuildingHeat_v3',
                      window=window, metric='heat_load', agg_by='year', agg_func='sum')
            twinql.compare(hist, sim)
        hot_time = time.time() - start
        
        results.append({
            'buildings': n,
            'cold_time': cold_time,
            'hot_time': hot_time,
            'speedup': cold_time / hot_time
        })
        
        print(f"  {n} buildings: cold={cold_time:.2f}s, hot={hot_time:.2f}s, speedup={cold_time/hot_time:.1f}x")
    
    return pd.DataFrame(results)

def run_all_benchmarks():
    """Run complete benchmark suite"""
    print("\n" + "=" * 70)
    print("TwinDB Benchmark Suite")
    print("=" * 70)
    
    w1_results = benchmark_w1(iterations=3)
    w2_results = benchmark_w2_scaling([2, 5, 10])
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nW2 Scaling Results:")
    print(w2_results.to_string(index=False))
    
    return {'w1': w1_results, 'w2': w2_results}

if __name__ == '__main__':
    run_all_benchmarks()
