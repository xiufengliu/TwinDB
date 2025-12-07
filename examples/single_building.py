#!/usr/bin/env python3
"""
W1 Workload: Single Building Retrofit Analysis
===============================================
Demonstrates TwinDB's capability to compare historical vs simulated
energy consumption for a single building retrofit scenario.

Question: Under 2018 Copenhagen weather, how much energy does B123 save
monthly after a medium-strength envelope retrofit?
"""
import sys
sys.path.insert(0, '/Users/xiufengliu/Projects/TwinDB')

import time
import pandas as pd
from core.db import execute_query
from dsl.twinql import TwinQL, HIST, SIM, define_scenario
from sim.manager import SimulationManager
from sim.models.building_heat import building_heat_model

def run_w1_analysis():
    print("=" * 60)
    print("TwinDB W1: Single Building Retrofit Analysis")
    print("=" * 60)
    
    # 1. Define the retrofit scenario
    print("\n[1] Defining retrofit scenario...")
    scenario_id = define_scenario(
        cid='s_retrofit',
        name='B123_medium_retrofit',
        weather_profile='DK_CPH_2018',
        price_profile='HEAT_2024_STD',
        control_policy={'type': 'OTC', 'params': {'slope': 1.2, 'offset': 20.0}},
        retrofit_package={
            'id': 'RETROFIT_MEDIUM',
            'wall_u_before': 0.60, 'wall_u_after': 0.18,
            'roof_u_before': 0.30, 'roof_u_after': 0.12,
            'window_u_before': 2.80, 'window_u_after': 0.90,
            'infiltration_before': 0.8, 'infiltration_after': 0.5,
            'indoor_setpoint': 21.0
        }
    )
    print(f"   Created scenario: {scenario_id}")
    
    # 2. Setup TwinQL and register model
    twinql = TwinQL()
    twinql.sim_manager.register_model('BuildingHeat_v3', building_heat_model)
    
    # 3. Define query components
    window = ('2018-01-01', '2019-01-01')
    
    hist = HIST(
        twin_id='Twin_B123',
        window=window,
        metric='heat_load',
        agg_by='month',
        agg_func='sum'
    )
    
    sim = SIM(
        twin_id='Twin_B123',
        scenario='s_retrofit',
        model='BuildingHeat_v3',
        window=window,
        metric='heat_load',
        agg_by='month',
        agg_func='sum'
    )
    
    # 4. Execute comparison (cold cache)
    print("\n[2] Running comparison query (cold cache)...")
    start_time = time.time()
    result = twinql.compare(hist, sim, compute_saving=True)
    cold_latency = time.time() - start_time
    print(f"   Cold cache latency: {cold_latency:.2f}s")
    
    # 5. Execute again (hot cache)
    print("\n[3] Running comparison query (hot cache)...")
    start_time = time.time()
    result = twinql.compare(hist, sim, compute_saving=True)
    hot_latency = time.time() - start_time
    print(f"   Hot cache latency: {hot_latency:.2f}s")
    
    # 6. Display results
    print("\n[4] Monthly Energy Comparison (kWh):")
    print("-" * 70)
    result['month'] = result['period'].dt.strftime('%Y-%m')
    result['baseline_mwh'] = result['value_baseline'] / 1000
    result['retrofit_mwh'] = result['value_retrofit'] / 1000
    result['saving_mwh'] = result['saving'] / 1000
    
    display_cols = ['month', 'baseline_mwh', 'retrofit_mwh', 'saving_mwh', 'saving_pct']
    print(result[display_cols].to_string(index=False))
    
    # 7. Annual summary
    print("\n[5] Annual Summary:")
    print("-" * 70)
    total_baseline = result['baseline_mwh'].sum()
    total_retrofit = result['retrofit_mwh'].sum()
    total_saving = result['saving_mwh'].sum()
    saving_pct = (total_saving / total_baseline) * 100
    
    print(f"   Baseline annual consumption:  {total_baseline:.1f} MWh")
    print(f"   Retrofit annual consumption:  {total_retrofit:.1f} MWh")
    print(f"   Annual energy savings:        {total_saving:.1f} MWh ({saving_pct:.1f}%)")
    
    # 8. Performance summary
    print("\n[6] Performance Summary:")
    print("-" * 70)
    print(f"   Cold cache query time: {cold_latency:.2f}s")
    print(f"   Hot cache query time:  {hot_latency:.2f}s")
    print(f"   Speedup factor:        {cold_latency/hot_latency:.1f}x")
    
    return result

if __name__ == '__main__':
    run_w1_analysis()
