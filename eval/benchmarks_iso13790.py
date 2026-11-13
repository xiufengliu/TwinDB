#!/usr/bin/env python3
"""Micro-benchmarks using REAL ISO 13790 simulation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from core.simulation import BuildingParams, simulate_building, generate_weather, compute_sensitivity

np.random.seed(42)
N_RUNS = 5

# Pre-generate weather (shared across all simulations)
WEATHER = generate_weather()

class RealISMEngine:
    """ISM with real simulation backend."""
    def __init__(self):
        self.cache = {}  # u_value -> (energy, params)
        self.beta = None
        self.timings = {'lookup': [], 'rls': [], 'ism': [], 'sim': []}
        
    def simulate(self, u_wall):
        """Run real simulation."""
        t0 = time.perf_counter()
        params = BuildingParams(wall_u=u_wall)
        energy, _ = simulate_building(params, WEATHER)
        self.timings['sim'].append((time.perf_counter() - t0) * 1000)
        return energy
    
    def lookup(self, u):
        """Find nearest cached result."""
        t0 = time.perf_counter()
        if not self.cache:
            result = None, None
        else:
            u_base = min(self.cache.keys(), key=lambda x: abs(x - u))
            result = u_base, self.cache[u_base]
        self.timings['lookup'].append((time.perf_counter() - t0) * 1000)
        return result
    
    def ism_update(self, e_base, u_base, u_new):
        """ISM incremental update."""
        t0 = time.perf_counter()
        if self.beta is None:
            return None
        e_new = e_base + self.beta * (u_new - u_base)
        self.timings['ism'].append((time.perf_counter() - t0) * 1000)
        return e_new
    
    def rls_update(self, u, e):
        """Update sensitivity via least squares."""
        t0 = time.perf_counter()
        self.cache[u] = e
        if len(self.cache) >= 2:
            us = np.array(list(self.cache.keys()))
            es = np.array(list(self.cache.values()))
            # Linear regression
            u_mean, e_mean = us.mean(), es.mean()
            self.beta = np.sum((us - u_mean) * (es - e_mean)) / (np.sum((us - u_mean)**2) + 1e-10)
        self.timings['rls'].append((time.perf_counter() - t0) * 1000)


def run_accuracy_experiment():
    """Measure ISM accuracy against real simulation."""
    print("\n1. ACCURACY (ISM vs Full Simulation)")
    print("-" * 50)
    
    # Compute true sensitivity
    base_params = BuildingParams(wall_u=0.35)
    beta = compute_sensitivity(base_params, WEATHER, 'wall_u', delta=0.01)
    E_base, _ = simulate_building(base_params, WEATHER)
    
    print(f"  Base energy: {E_base:.1f} kWh")
    print(f"  Sensitivity: {beta:.1f} kWh per W/m²K")
    print()
    print(f"  {'Δu':>8} | {'Actual':>12} | {'ISM':>12} | {'Error (%)':>10}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12} | {'-'*10}")
    
    errors = []
    for delta_u in [0.01, 0.05, 0.10, 0.15, 0.20]:
        u_new = 0.35 + delta_u
        # Actual
        p = BuildingParams(wall_u=u_new)
        E_actual, _ = simulate_building(p, WEATHER)
        # ISM
        E_ism = E_base + beta * delta_u
        error = abs(E_ism - E_actual) / E_actual * 100
        errors.append(error)
        print(f"  {delta_u:>8.2f} | {E_actual:>12.1f} | {E_ism:>12.1f} | {error:>10.3f}")
    
    return np.mean(errors), np.max(errors)


def run_speedup_experiment(n_queries=50):
    """Measure end-to-end speedup."""
    print(f"\n2. SPEEDUP ({n_queries} queries)")
    print("-" * 50)
    
    u_values = np.linspace(0.20, 0.50, n_queries)
    
    # Full simulation baseline
    t0 = time.perf_counter()
    for u in u_values:
        p = BuildingParams(wall_u=u)
        simulate_building(p, WEATHER)
    full_time = time.perf_counter() - t0
    
    # ISM approach
    ism = RealISMEngine()
    t0 = time.perf_counter()
    for i, u in enumerate(u_values):
        u_base, e_base = ism.lookup(u)
        if e_base is not None and ism.beta is not None and abs(u - u_base) < 0.3:
            e = ism.ism_update(e_base, u_base, u)
        else:
            e = ism.simulate(u)
        ism.rls_update(u, e)
    ism_time = time.perf_counter() - t0
    
    speedup = full_time / ism_time
    
    print(f"  Full-Sim time: {full_time*1000:.1f} ms")
    print(f"  ISM time:      {ism_time*1000:.1f} ms")
    print(f"  Speedup:       {speedup:.1f}x")
    print(f"  Simulations:   {len(ism.timings['sim'])} (cold-start)")
    print(f"  ISM updates:   {len(ism.timings['ism'])}")
    
    return speedup, ism.timings


def run_component_timing():
    """Micro-benchmark individual components."""
    print("\n3. COMPONENT TIMING (micro-benchmark)")
    print("-" * 50)
    
    ism = RealISMEngine()
    
    # Warm up cache
    for u in np.linspace(0.20, 0.50, 10):
        e = ism.simulate(u)
        ism.rls_update(u, e)
    
    # Measure components
    for _ in range(100):
        u = np.random.uniform(0.20, 0.50)
        ism.lookup(u)
        if ism.beta:
            ism.ism_update(19000, 0.35, u)
        ism.rls_update(u, 19000)
    
    print(f"  Simulation:    {np.mean(ism.timings['sim']):.2f} ± {np.std(ism.timings['sim']):.2f} ms")
    print(f"  Cache lookup:  {np.mean(ism.timings['lookup'])*1000:.2f} ± {np.std(ism.timings['lookup'])*1000:.2f} μs")
    print(f"  ISM update:    {np.mean(ism.timings['ism'])*1000:.2f} ± {np.std(ism.timings['ism'])*1000:.2f} μs")
    print(f"  RLS update:    {np.mean(ism.timings['rls'])*1000:.2f} ± {np.std(ism.timings['rls'])*1000:.2f} μs")
    
    return ism.timings


if __name__ == '__main__':
    print("=" * 60)
    print("TwinDB Experiments with REAL ISO 13790 Simulation")
    print("=" * 60)
    
    # 1. Accuracy
    mean_err, max_err = run_accuracy_experiment()
    
    # 2. Speedup
    speedup, timings = run_speedup_experiment(50)
    
    # 3. Component timing
    run_component_timing()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"  ISM Mean Error:  {mean_err:.3f}%")
    print(f"  ISM Max Error:   {max_err:.3f}%")
    print(f"  Speedup (50q):   {speedup:.1f}x")
    print(f"  Sim time:        {np.mean(timings['sim']):.1f} ms")
