#!/usr/bin/env python3
"""
W2 Workload: District-Level Multi-Building Retrofit Analysis
=============================================================
Demonstrates TwinDB's capability to:
1. Select worst-performing 20% buildings by energy intensity
2. Apply retrofit scenario to selected buildings
3. Aggregate savings at network level

Question: On DH_NET_CPH_01, if we retrofit the worst 20% buildings,
what are the total annual energy savings and peak load reduction?
"""
import sys
sys.path.insert(0, '/Users/xiufengliu/Projects/TwinDB')

import time
import pandas as pd
from core.db import execute_query
from dsl.twinql import TwinQL, HIST, SIM, define_scenario
from sim.manager import SimulationManager
from sim.models.building_heat import building_heat_model

def get_building_intensities(network_id: str, year: int = 2018) -> pd.DataFrame:
    """Calculate annual energy intensity for all buildings in a network"""
    sql = """
        SELECT a.aid, a.area_m2, a.usage_type, a.energy_label, t.tid,
               SUM(ts.value) as annual_kwh,
               SUM(ts.value) / a.area_m2 as intensity_kwh_m2
        FROM asset a
        JOIN twin t ON t.aid = a.aid
        JOIN timeseries ts ON ts.tid = t.tid
        WHERE a.network_id = %s
          AND ts.cid = 'REALITY'
          AND ts.metric = 'heat_load'
          AND ts.ts >= %s AND ts.ts < %s
        GROUP BY a.aid, a.area_m2, a.usage_type, a.energy_label, t.tid
        ORDER BY intensity_kwh_m2 DESC
    """
    return execute_query(sql, (network_id, f'{year}-01-01', f'{year+1}-01-01'))

def select_worst_buildings(intensities: pd.DataFrame, pct: float = 0.2) -> pd.DataFrame:
    """Select top X% worst-performing buildings by intensity"""
    n = max(1, int(len(intensities) * pct))
    return intensities.head(n)

def run_w2_analysis():
    print("=" * 70)
    print("TwinDB W2: District-Level Multi-Building Retrofit Analysis")
    print("=" * 70)
    
    network_id = 'DH_NET_CPH_01'
    window = ('2018-01-01', '2019-01-01')
    
    # 1. Calculate building intensities
    print(f"\n[1] Analyzing buildings on {network_id}...")
    intensities = get_building_intensities(network_id, 2018)
    print(f"   Found {len(intensities)} buildings")
    print("\n   Building Energy Intensities (kWh/m²/year):")
    print("-" * 70)
    display = intensities[['aid', 'area_m2', 'usage_type', 'energy_label', 'intensity_kwh_m2']].copy()
    display['intensity_kwh_m2'] = display['intensity_kwh_m2'].round(1)
    print(display.to_string(index=False))
    
    # 2. Select worst 20%
    print("\n[2] Selecting worst 20% buildings for retrofit...")
    worst = select_worst_buildings(intensities, 0.2)
    print(f"   Selected {len(worst)} buildings: {', '.join(worst['aid'].tolist())}")
    
    # 3. Define network retrofit scenario
    print("\n[3] Defining network retrofit scenario...")
    scenario_id = define_scenario(
        cid='s_retrofit_net',
        name='Network_medium_retrofit',
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
    
    # 4. Setup TwinQL
    twinql = TwinQL()
    twinql.sim_manager.register_model('BuildingHeat_v3', building_heat_model)
    
    # 5. Run simulations for worst buildings (cold cache)
    print("\n[4] Running retrofit simulations (cold cache)...")
    start_time = time.time()
    
    results = []
    for _, row in worst.iterrows():
        tid = row['tid']
        
        hist = HIST(twin_id=tid, window=window, metric='heat_load', 
                    agg_by='year', agg_func='sum')
        sim = SIM(twin_id=tid, scenario='s_retrofit_net', model='BuildingHeat_v3',
                  window=window, metric='heat_load', agg_by='year', agg_func='sum')
        
        comparison = twinql.compare(hist, sim)
        comparison['aid'] = row['aid']
        comparison['area_m2'] = row['area_m2']
        comparison['usage_type'] = row['usage_type']
        results.append(comparison)
    
    cold_latency = time.time() - start_time
    print(f"   Cold cache latency: {cold_latency:.2f}s")
    
    # 6. Run again (hot cache)
    print("\n[5] Running retrofit simulations (hot cache)...")
    start_time = time.time()
    
    results = []
    for _, row in worst.iterrows():
        tid = row['tid']
        hist = HIST(twin_id=tid, window=window, metric='heat_load',
                    agg_by='year', agg_func='sum')
        sim = SIM(twin_id=tid, scenario='s_retrofit_net', model='BuildingHeat_v3',
                  window=window, metric='heat_load', agg_by='year', agg_func='sum')
        comparison = twinql.compare(hist, sim)
        comparison['aid'] = row['aid']
        comparison['area_m2'] = row['area_m2']
        comparison['usage_type'] = row['usage_type']
        results.append(comparison)
    
    hot_latency = time.time() - start_time
    print(f"   Hot cache latency: {hot_latency:.2f}s")
    
    # 7. Aggregate results
    all_results = pd.concat(results, ignore_index=True)
    
    print("\n[6] Per-Building Retrofit Results:")
    print("-" * 70)
    summary = all_results[['aid', 'area_m2', 'usage_type', 'value_baseline', 
                           'value_retrofit', 'saving', 'saving_pct']].copy()
    summary['baseline_mwh'] = summary['value_baseline'] / 1000
    summary['retrofit_mwh'] = summary['value_retrofit'] / 1000
    summary['saving_mwh'] = summary['saving'] / 1000
    print(summary[['aid', 'usage_type', 'baseline_mwh', 'retrofit_mwh', 
                   'saving_mwh', 'saving_pct']].round(1).to_string(index=False))
    
    # 8. Network-level summary
    print("\n[7] Network-Level Summary (DH_NET_CPH_01):")
    print("-" * 70)
    
    # Get total network baseline
    total_network = execute_query("""
        SELECT SUM(ts.value) / 1000 as total_mwh
        FROM timeseries ts
        JOIN twin t ON t.tid = ts.tid
        JOIN asset a ON a.aid = t.aid
        WHERE a.network_id = %s AND ts.cid = 'REALITY' AND ts.metric = 'heat_load'
        AND ts.ts >= '2018-01-01' AND ts.ts < '2019-01-01'
    """, (network_id,))
    
    network_baseline = total_network['total_mwh'].iloc[0]
    retrofit_savings = summary['saving_mwh'].sum()
    
    print(f"   Total network baseline:     {network_baseline:.1f} MWh")
    print(f"   Buildings retrofitted:      {len(worst)} of {len(intensities)}")
    print(f"   Savings from retrofit:      {retrofit_savings:.1f} MWh")
    print(f"   Network-level savings:      {(retrofit_savings/network_baseline)*100:.1f}%")
    
    # 9. By usage type
    print("\n[8] Savings by Building Type:")
    print("-" * 70)
    by_type = summary.groupby('usage_type').agg({
        'saving_mwh': 'sum',
        'aid': 'count'
    }).rename(columns={'aid': 'count'})
    print(by_type.round(1).to_string())
    
    # 10. Performance summary
    print("\n[9] Performance Summary:")
    print("-" * 70)
    print(f"   Cold cache query time: {cold_latency:.2f}s ({len(worst)} buildings)")
    print(f"   Hot cache query time:  {hot_latency:.2f}s")
    print(f"   Speedup factor:        {cold_latency/hot_latency:.1f}x")
    
    return all_results

if __name__ == '__main__':
    run_w2_analysis()
