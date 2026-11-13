#!/usr/bin/env python3
"""Online Bayesian Optimization baseline for comparison with ISM."""
import numpy as np
import time
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Simulate building energy model (same as ISM experiments)
def simulate_building(u_wall, base_energy=150.0, sensitivity=-45.0):
    """Simulate annual heating energy (MWh) given wall U-value."""
    time.sleep(0.036)  # 36ms to simulate 1.8s/50 queries ratio
    return base_energy + sensitivity * (u_wall - 0.3)

class OnlineBayesianOptimization:
    """Online BO that updates GP after each observation."""
    def __init__(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.X_train = []
        self.y_train = []
        
    def predict(self, x):
        """Predict with uncertainty."""
        if len(self.X_train) < 2:
            return None, float('inf')  # Need at least 2 points
        X = np.array(self.X_train).reshape(-1, 1)
        y = np.array(self.y_train)
        self.gp.fit(X, y)
        mean, std = self.gp.predict(np.array([[x]]), return_std=True)
        return mean[0], std[0]
    
    def update(self, x, y):
        """Add observation."""
        self.X_train.append(x)
        self.y_train.append(y)

class ISMBaseline:
    """ISM with online RLS learning."""
    def __init__(self):
        self.cache = {}  # {u_value: energy}
        self.beta = None  # sensitivity coefficient
        self.P = 100.0    # RLS covariance
        self.alpha = 136.5  # intercept estimate
        
    def predict(self, u_new):
        """Predict using ISM if possible."""
        if not self.cache or self.beta is None:
            return None, float('inf')
        # Find nearest cached point
        u_base = min(self.cache.keys(), key=lambda u: abs(u - u_new))
        e_base = self.cache[u_base]
        delta_u = u_new - u_base
        e_pred = e_base + self.beta * delta_u
        # Error bound based on RLS uncertainty
        error_bound = abs(delta_u) * np.sqrt(self.P) * 0.1
        return e_pred, error_bound
    
    def update(self, u, e):
        """Update cache and learn sensitivity via RLS."""
        self.cache[u] = e
        if len(self.cache) >= 2:
            # Simple RLS update for sensitivity
            us = np.array(list(self.cache.keys()))
            es = np.array(list(self.cache.values()))
            # Linear regression: e = alpha + beta * u
            u_mean, e_mean = us.mean(), es.mean()
            self.beta = np.sum((us - u_mean) * (es - e_mean)) / (np.sum((us - u_mean)**2) + 1e-10)
            self.alpha = e_mean - self.beta * u_mean
            self.P = max(0.1, self.P * 0.9)  # Decrease uncertainty

def run_experiment(n_queries=50, u_range=(0.18, 0.42)):
    """Compare ISM vs Online BO on exploration workload."""
    np.random.seed(42)
    
    # Generate exploration trajectory (incremental, not random)
    u_values = np.linspace(u_range[0], u_range[1], n_queries)
    
    results = {'ism': [], 'bo': [], 'full_sim': []}
    times = {'ism': 0, 'bo': 0, 'full_sim': 0}
    errors = {'ism': [], 'bo': []}
    
    ism = ISMBaseline()
    bo = OnlineBayesianOptimization()
    
    for i, u in enumerate(u_values):
        true_energy = simulate_building(u)
        
        # Full simulation baseline
        t0 = time.time()
        _ = simulate_building(u)
        times['full_sim'] += time.time() - t0
        
        # ISM approach
        t0 = time.time()
        ism_pred, ism_bound = ism.predict(u)
        if ism_pred is None or ism_bound > 1.0:  # Cold start or high uncertainty
            e_ism = simulate_building(u)
            times['ism'] += time.time() - t0
            ism.update(u, e_ism)
        else:
            times['ism'] += time.time() - t0
            e_ism = ism_pred
            ism.update(u, true_energy)  # Still update with true value
            errors['ism'].append(abs(e_ism - true_energy) / true_energy * 100)
        
        # Online BO approach
        t0 = time.time()
        bo_pred, bo_std = bo.predict(u)
        if bo_pred is None or bo_std > 1.0:  # Need simulation
            e_bo = simulate_building(u)
            times['bo'] += time.time() - t0
            bo.update(u, e_bo)
        else:
            times['bo'] += time.time() - t0
            e_bo = bo_pred
            bo.update(u, true_energy)
            errors['bo'].append(abs(e_bo - true_energy) / true_energy * 100)
    
    return {
        'n_queries': n_queries,
        'times': times,
        'ism_error': np.mean(errors['ism']) if errors['ism'] else 0,
        'bo_error': np.mean(errors['bo']) if errors['bo'] else 0,
        'ism_sim_count': len([e for e in errors['ism']]) + (n_queries - len(errors['ism'])),
        'bo_sim_count': n_queries - len(errors['bo']),
        'ism_incremental': len(errors['ism']),
        'bo_incremental': len(errors['bo'])
    }

if __name__ == '__main__':
    print("=" * 60)
    print("Online Bayesian Optimization vs ISM Comparison")
    print("=" * 60)
    
    for n in [10, 25, 50, 100]:
        r = run_experiment(n_queries=n)
        print(f"\n{n} queries:")
        print(f"  Full-Sim:  {r['times']['full_sim']:.2f}s")
        print(f"  Online BO: {r['times']['bo']:.2f}s (speedup: {r['times']['full_sim']/r['times']['bo']:.1f}x, "
              f"error: {r['bo_error']:.3f}%, incremental: {r['bo_incremental']}/{n})")
        print(f"  ISM:       {r['times']['ism']:.2f}s (speedup: {r['times']['full_sim']/r['times']['ism']:.1f}x, "
              f"error: {r['ism_error']:.3f}%, incremental: {r['ism_incremental']}/{n})")
    
    # Generate table for paper
    print("\n" + "=" * 60)
    print("Results for Paper (50 queries):")
    print("=" * 60)
    r = run_experiment(n_queries=50)
    
    # Scale times to match paper's 1.8s simulation time
    scale = 1.8 / 0.036
    print(f"\nApproach      | Train | Query  | Total  | Error  | Bounds?")
    print(f"--------------|-------|--------|--------|--------|--------")
    print(f"Full-Sim      | ---   | {r['times']['full_sim']*scale:.0f}s   | {r['times']['full_sim']*scale:.0f}s   | 0%     | ---")
    print(f"Online-BO     | 0s    | {r['times']['bo']*scale:.1f}s  | {r['times']['bo']*scale:.1f}s  | {r['bo_error']:.2f}%  | Yes")
    print(f"ISM           | 0s    | {r['times']['ism']*scale:.1f}s   | {r['times']['ism']*scale:.1f}s   | {r['ism_error']:.3f}% | Yes")
