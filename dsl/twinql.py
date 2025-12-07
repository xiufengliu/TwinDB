"""TwinQL DSL - Python implementation of TwinDB query language"""
import json
from typing import List, Tuple, Optional, Union
import pandas as pd
from core.db import execute_query, execute_sql
from sim.manager import SimulationManager

def define_scenario(cid: str, name: str = None, weather_profile: str = None,
                    price_profile: str = None, control_policy: dict = None,
                    retrofit_package: dict = None, base_cid: str = None) -> str:
    """Define a scenario in TwinDB"""
    cfg = {}
    if weather_profile: cfg['weather_profile'] = weather_profile
    if price_profile: cfg['price_profile'] = price_profile
    if control_policy: cfg['control_policy'] = control_policy
    if retrofit_package: cfg['retrofit_package'] = retrofit_package
    
    execute_sql("""
        INSERT INTO scenario (cid, name, cfg, base_cid)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (cid) DO UPDATE SET cfg = EXCLUDED.cfg, name = EXCLUDED.name
    """, (cid, name or cid, json.dumps(cfg), base_cid))
    return cid

class HIST:
    """Historical data source in TwinQL"""
    def __init__(self, twin_id: str, window: Tuple[str, str], metric: str,
                 agg_by: str = None, agg_func: str = 'avg'):
        self.twin_id = twin_id
        self.window = window
        self.metric = metric
        self.agg_by = agg_by
        self.agg_func = agg_func
        self._alias = None
    
    def alias(self, name: str) -> 'HIST':
        self._alias = name
        return self
    
    def execute(self) -> pd.DataFrame:
        if self.agg_by:
            sql = f"""
                SELECT date_trunc('{self.agg_by}', ts) as period,
                       {self.agg_func}(value) as value
                FROM timeseries
                WHERE tid = %s AND cid = 'REALITY' AND metric = %s
                AND ts >= %s AND ts < %s
                GROUP BY period ORDER BY period
            """
        else:
            sql = """
                SELECT ts, value FROM timeseries
                WHERE tid = %s AND cid = 'REALITY' AND metric = %s
                AND ts >= %s AND ts < %s ORDER BY ts
            """
        return execute_query(sql, (self.twin_id, self.metric, self.window[0], self.window[1]))

class SIM:
    """Simulation data source in TwinQL"""
    def __init__(self, twin_id: str, scenario: str, model: str,
                 window: Tuple[str, str], metric: str,
                 agg_by: str = None, agg_func: str = 'avg'):
        self.twin_id = twin_id
        self.scenario = scenario
        self.model = model
        self.window = window
        self.metric = metric
        self.agg_by = agg_by
        self.agg_func = agg_func
        self._alias = None
        self._sim_manager = None
    
    def alias(self, name: str) -> 'SIM':
        self._alias = name
        return self
    
    def set_sim_manager(self, manager: SimulationManager):
        self._sim_manager = manager
        return self
    
    def execute(self) -> pd.DataFrame:
        if self._sim_manager:
            self._sim_manager.simulate(
                [self.twin_id], self.scenario, self.model,
                self.window, [self.metric]
            )
        
        if self.agg_by:
            sql = f"""
                SELECT date_trunc('{self.agg_by}', ts) as period,
                       {self.agg_func}(value) as value
                FROM timeseries
                WHERE tid = %s AND cid = %s AND metric = %s
                AND ts >= %s AND ts < %s
                GROUP BY period ORDER BY period
            """
        else:
            sql = """
                SELECT ts, value FROM timeseries
                WHERE tid = %s AND cid = %s AND metric = %s
                AND ts >= %s AND ts < %s ORDER BY ts
            """
        return execute_query(sql, (self.twin_id, self.scenario, self.metric, 
                                   self.window[0], self.window[1]))

class TwinQL:
    """TwinQL query builder"""
    def __init__(self):
        self.sim_manager = SimulationManager()
    
    def select(self, hist: HIST = None, sim: SIM = None) -> dict:
        """Execute a twin select query comparing historical and simulated data"""
        results = {}
        if hist:
            results['hist'] = hist.execute()
        if sim:
            sim.set_sim_manager(self.sim_manager)
            results['sim'] = sim.execute()
        return results
    
    def compare(self, hist: HIST, sim: SIM, 
                compute_saving: bool = True) -> pd.DataFrame:
        """Compare historical vs simulated data"""
        sim.set_sim_manager(self.sim_manager)
        h_df = hist.execute()
        s_df = sim.execute()
        
        # Merge on period/ts
        merge_col = 'period' if hist.agg_by else 'ts'
        result = h_df.merge(s_df, on=merge_col, suffixes=('_baseline', '_retrofit'))
        
        if compute_saving:
            result['saving'] = result['value_baseline'] - result['value_retrofit']
            result['saving_pct'] = (result['saving'] / result['value_baseline'] * 100).round(2)
        
        return result
