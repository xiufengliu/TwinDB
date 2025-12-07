"""Simulation Manager with caching for TwinDB"""
import hashlib
import json
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd
from core.db import get_connection, execute_query, execute_sql

class SimulationManager:
    def __init__(self):
        self.models = {}
    
    def register_model(self, mid: str, model_func):
        """Register a simulation model function"""
        self.models[mid] = model_func
    
    def _compute_signature(self, tid: str, cid: str, mid: str, 
                           window: Tuple[str, str], metrics: List[str]) -> str:
        """Compute cache signature"""
        key = f"{tid}|{cid}|{mid}|{window[0]}|{window[1]}|{','.join(sorted(metrics))}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def _check_cache(self, sig: str) -> bool:
        """Check if simulation result exists in cache"""
        result = execute_query(
            "SELECT 1 FROM sim_cache WHERE sig = %s", (sig,), as_df=False
        )
        return bool(result)
    
    def _write_cache(self, sig: str, tid: str, cid: str, mid: str,
                     window: Tuple[str, str], metrics: List[str]):
        """Write cache entry"""
        execute_sql("""
            INSERT INTO sim_cache (sig, tid, cid, mid, window_start, window_end, metrics)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sig) DO NOTHING
        """, (sig, tid, cid, mid, window[0], window[1], metrics))
    
    def _write_timeseries(self, tid: str, cid: str, df: pd.DataFrame, 
                          metric: str, mid: str, version: str = '1.0'):
        """Write simulation results to timeseries table"""
        source = f"sim:{mid}@{version}"
        with get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO timeseries (tid, cid, metric, metric_type, unit, ts, value, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tid, cid, metric, ts) DO UPDATE SET value = EXCLUDED.value
                    """, (tid, cid, metric, 'power', 'kW', row['ts'], row['value'], source))
            conn.commit()
    
    def simulate(self, tids: List[str], scenario_id: str, model_id: str,
                 window: Tuple[str, str], metrics: List[str]) -> dict:
        """
        Run simulation for given twins under a scenario.
        Returns dict mapping tid -> DataFrame of results.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        results = {}
        to_simulate = []
        
        # Check cache for each twin
        for tid in tids:
            for metric in metrics:
                sig = self._compute_signature(tid, scenario_id, model_id, window, [metric])
                if self._check_cache(sig):
                    # Load from existing timeseries
                    df = execute_query("""
                        SELECT ts, value FROM timeseries
                        WHERE tid = %s AND cid = %s AND metric = %s
                        AND ts >= %s AND ts < %s ORDER BY ts
                    """, (tid, scenario_id, metric, window[0], window[1]))
                    results[(tid, metric)] = df
                else:
                    to_simulate.append((tid, metric, sig))
        
        # Run simulations for cache misses
        if to_simulate:
            # Load scenario config
            scenario_cfg = execute_query(
                "SELECT cfg FROM scenario WHERE cid = %s", (scenario_id,), as_df=False
            )
            cfg = scenario_cfg[0]['cfg'] if scenario_cfg else {}
            
            model_func = self.models[model_id]
            for tid, metric, sig in to_simulate:
                # Load historical data as baseline
                hist_df = execute_query("""
                    SELECT ts, value FROM timeseries
                    WHERE tid = %s AND cid = 'REALITY' AND metric = %s
                    AND ts >= %s AND ts < %s ORDER BY ts
                """, (tid, metric, window[0], window[1]))
                
                # Load asset info
                asset_info = execute_query("""
                    SELECT a.* FROM asset a JOIN twin t ON t.aid = a.aid WHERE t.tid = %s
                """, (tid,), as_df=False)
                asset = asset_info[0] if asset_info else {}
                
                # Run model
                sim_df = model_func(hist_df, cfg, asset)
                
                # Write results
                self._write_timeseries(tid, scenario_id, sim_df, metric, model_id)
                self._write_cache(sig, tid, scenario_id, model_id, window, [metric])
                results[(tid, metric)] = sim_df
        
        return results
