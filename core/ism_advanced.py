"""
Advanced ISM: Auto-Calibration and Multi-Building Support

Provides:
1. Online learning of sensitivity coefficients via RLS
2. Batch ISM for multi-building queries
3. Formal error bound computation
4. Adaptive threshold tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
import hashlib
import json

from core.db import execute_query, execute_sql
from core.ism import ScenarioDelta, DeltaType, IncrementalResult


# ============================================================================
# Online Sensitivity Learning
# ============================================================================

@dataclass
class CalibrationPoint:
    """A single observation for calibration"""
    param_value: float
    energy: float
    timestamp: float


class OnlineSensitivityLearner:
    """
    Learns sensitivity coefficients online from simulation results.
    
    Algorithm: Recursive Least Squares (RLS)
    - Updates coefficients incrementally as new data arrives
    - No need to store all historical data
    - Converges to optimal linear fit
    
    Model: E = α + β × U
    Sensitivity: ∂E/∂U = β
    """
    
    def __init__(self, forgetting_factor: float = 0.99):
        self.forgetting_factor = forgetting_factor  # λ in RLS
        self.coefficients: Dict[str, Dict[str, float]] = defaultdict(dict)
        # For each (tid, param): store {alpha, beta, P (covariance)}
        self.rls_state: Dict[str, Dict] = defaultdict(lambda: {
            'theta': np.array([0.0, 0.0]),  # [alpha, beta]
            'P': np.eye(2) * 1000,  # Initial covariance (high uncertainty)
            'n_samples': 0
        })
    
    def update(self, tid: str, param: str, param_value: float, energy: float):
        """
        Update sensitivity estimate with new observation.
        Uses Recursive Least Squares for online learning.
        """
        key = f"{tid}:{param}"
        state = self.rls_state[key]
        
        # Feature vector: [1, param_value]
        x = np.array([1.0, param_value])
        y = energy
        
        # RLS update
        P = state['P']
        theta = state['theta']
        λ = self.forgetting_factor
        
        # Kalman gain
        Px = P @ x
        k = Px / (λ + x @ Px)
        
        # Update estimate
        error = y - x @ theta
        theta = theta + k * error
        
        # Update covariance
        P = (P - np.outer(k, x @ P)) / λ
        
        # Store updated state
        state['theta'] = theta
        state['P'] = P
        state['n_samples'] += 1
        
        # Extract sensitivity (beta coefficient)
        self.coefficients[tid][param] = theta[1]
    
    def get_sensitivity(self, tid: str, param: str) -> Tuple[float, float]:
        """
        Get learned sensitivity and uncertainty.
        Returns: (sensitivity, std_error)
        """
        key = f"{tid}:{param}"
        if key not in self.rls_state or self.rls_state[key]['n_samples'] < 2:
            return None, float('inf')
        
        state = self.rls_state[key]
        sensitivity = state['theta'][1]
        # Standard error from covariance matrix
        std_error = np.sqrt(state['P'][1, 1])
        
        return sensitivity, std_error
    
    def get_confidence(self, tid: str, param: str) -> float:
        """Get confidence in sensitivity estimate (0-1)"""
        key = f"{tid}:{param}"
        if key not in self.rls_state:
            return 0.0
        
        n = self.rls_state[key]['n_samples']
        # Confidence increases with samples, saturates around 10
        return min(1.0, n / 10)


# ============================================================================
# Formal Error Bounds
# ============================================================================

@dataclass
class ErrorBound:
    """Formal error bound with confidence interval"""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    
    @property
    def relative_error(self) -> float:
        """Maximum relative error"""
        if self.point_estimate == 0:
            return float('inf')
        max_abs_error = max(
            abs(self.upper_bound - self.point_estimate),
            abs(self.point_estimate - self.lower_bound)
        )
        return max_abs_error / abs(self.point_estimate)


class ErrorBoundComputer:
    """
    Computes formal error bounds for ISM updates.
    
    Theory:
    For linear model E = α + β×U with estimated β̂ and std error σ_β:
    
    ΔE = β̂ × ΔU
    Var(ΔE) = (ΔU)² × σ_β²
    
    95% CI: ΔE ± 1.96 × |ΔU| × σ_β
    """
    
    def __init__(self, learner: OnlineSensitivityLearner):
        self.learner = learner
    
    def compute_bound(self, tid: str, param: str, 
                      old_val: float, new_val: float,
                      base_energy: float,
                      confidence: float = 0.95) -> ErrorBound:
        """Compute error bound for incremental update"""
        
        sensitivity, std_error = self.learner.get_sensitivity(tid, param)
        
        if sensitivity is None:
            # No learned sensitivity - use default with high uncertainty
            sensitivity = 150000  # Default for wall_u
            std_error = sensitivity * 0.20  # 20% uncertainty
        
        delta_u = new_val - old_val
        delta_e = sensitivity * delta_u
        
        # Z-score for confidence level
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        
        # Error margin
        margin = z * abs(delta_u) * std_error
        
        estimated_energy = base_energy + delta_e
        
        return ErrorBound(
            point_estimate=estimated_energy,
            lower_bound=estimated_energy - margin,
            upper_bound=estimated_energy + margin,
            confidence_level=confidence
        )


# ============================================================================
# Multi-Building Batch ISM
# ============================================================================

@dataclass
class BatchISMResult:
    """Result of batch ISM across multiple buildings"""
    results: Dict[str, IncrementalResult]  # tid -> result
    total_time_ms: float
    ism_count: int
    full_sim_count: int
    avg_error_bound_pct: float


class BatchISMEngine:
    """
    Batch ISM for multi-building queries.
    
    Optimization: When querying N buildings with same scenario change,
    we can share sensitivity computations and batch database operations.
    
    Algorithm:
    1. Group buildings by similar characteristics
    2. Compute shared sensitivity estimates
    3. Apply ISM updates in batch
    4. Identify outliers needing full simulation
    """
    
    def __init__(self):
        self.learner = OnlineSensitivityLearner()
        self.error_computer = ErrorBoundComputer(self.learner)
        self.cache: Dict[str, Tuple[Dict, pd.DataFrame]] = {}  # (tid, sig) -> (cfg, result)
    
    def batch_update(self, tids: List[str], 
                     base_cfg: Dict, new_cfg: Dict,
                     base_results: Dict[str, pd.DataFrame],
                     error_tolerance: float = 0.10) -> BatchISMResult:
        """
        Apply ISM to multiple buildings in batch.
        
        Args:
            tids: List of twin IDs
            base_cfg: Base scenario configuration
            new_cfg: New scenario configuration  
            base_results: Cached results for each twin
            error_tolerance: Maximum acceptable error
        
        Returns:
            BatchISMResult with per-building results
        """
        start = time.perf_counter()
        
        # Compute scenario delta once
        delta = ScenarioDelta.compute(base_cfg, new_cfg)
        
        results = {}
        ism_count = 0
        full_sim_count = 0
        error_bounds = []
        
        for tid in tids:
            if tid not in base_results:
                # No cached result - need full sim
                results[tid] = IncrementalResult(
                    success=False, estimated_energy=0, error_bound=float('inf'),
                    error_bound_pct=100, method='recompute', base_scenario='',
                    computation_time_ms=0
                )
                full_sim_count += 1
                continue
            
            base_result = base_results[tid]
            base_energy = float(base_result['value_retrofit'].sum()) if 'value_retrofit' in base_result.columns else float(base_result['value'].sum())
            
            # Check if ISM is applicable
            if delta.delta_type == DeltaType.STRUCTURAL or delta.magnitude > 0.30:
                results[tid] = IncrementalResult(
                    success=False, estimated_energy=0, error_bound=float('inf'),
                    error_bound_pct=100, method='recompute', base_scenario='',
                    computation_time_ms=0
                )
                full_sim_count += 1
                continue
            
            # Compute ISM update with error bounds
            total_delta = 0.0
            total_variance = 0.0
            
            for param, (old_val, new_val) in delta.param_deltas.items():
                sensitivity, std_error = self.learner.get_sensitivity(tid, param)
                if sensitivity is None:
                    # Use default
                    sensitivity = {'wall_u_after': 150000, 'window_u_after': 50000, 
                                   'roof_u_after': 80000}.get(param, 100000)
                    std_error = sensitivity * 0.10
                
                delta_param = new_val - old_val
                total_delta += sensitivity * delta_param
                total_variance += (delta_param * std_error) ** 2
            
            estimated_energy = base_energy + total_delta
            error_bound = 1.96 * np.sqrt(total_variance)  # 95% CI
            error_bound_pct = error_bound / max(estimated_energy, 1) * 100
            
            if error_bound_pct <= error_tolerance * 100:
                results[tid] = IncrementalResult(
                    success=True,
                    estimated_energy=estimated_energy,
                    error_bound=error_bound,
                    error_bound_pct=error_bound_pct,
                    method='batch_ism',
                    base_scenario='',
                    computation_time_ms=0
                )
                ism_count += 1
                error_bounds.append(error_bound_pct)
            else:
                results[tid] = IncrementalResult(
                    success=False, estimated_energy=0, error_bound=float('inf'),
                    error_bound_pct=100, method='recompute', base_scenario='',
                    computation_time_ms=0
                )
                full_sim_count += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return BatchISMResult(
            results=results,
            total_time_ms=elapsed,
            ism_count=ism_count,
            full_sim_count=full_sim_count,
            avg_error_bound_pct=np.mean(error_bounds) if error_bounds else 0
        )
    
    def calibrate_from_simulation(self, tid: str, param: str, 
                                   param_value: float, actual_energy: float):
        """Update sensitivity model with simulation result"""
        self.learner.update(tid, param, param_value, actual_energy)


# ============================================================================
# Adaptive ISM Optimizer
# ============================================================================

class AdaptiveISMOptimizer:
    """
    Adaptive optimizer that learns when ISM is safe.
    
    Key insight: Different buildings have different sensitivity characteristics.
    We learn per-building thresholds for when ISM is accurate.
    
    Algorithm:
    1. Start with conservative thresholds
    2. Track prediction errors
    3. Tighten/loosen thresholds based on observed accuracy
    """
    
    def __init__(self):
        self.batch_engine = BatchISMEngine()
        self.error_history: Dict[str, List[float]] = defaultdict(list)
        self.adaptive_thresholds: Dict[str, float] = defaultdict(lambda: 0.20)
    
    def update_threshold(self, tid: str, predicted: float, actual: float):
        """Update adaptive threshold based on prediction error"""
        error = abs(predicted - actual) / max(actual, 1)
        self.error_history[tid].append(error)
        
        # Keep last 20 errors
        if len(self.error_history[tid]) > 20:
            self.error_history[tid] = self.error_history[tid][-20:]
        
        # Adjust threshold based on recent accuracy
        recent_errors = self.error_history[tid]
        if len(recent_errors) >= 5:
            p95_error = np.percentile(recent_errors, 95)
            # Set threshold to 95th percentile error + margin
            self.adaptive_thresholds[tid] = min(0.30, p95_error * 1.5)
    
    def get_threshold(self, tid: str) -> float:
        """Get adaptive threshold for building"""
        return self.adaptive_thresholds[tid]
    
    def should_use_ism(self, tid: str, delta: ScenarioDelta) -> Tuple[bool, str]:
        """Decide if ISM should be used based on learned thresholds"""
        threshold = self.get_threshold(tid)
        
        if delta.delta_type == DeltaType.STRUCTURAL:
            return False, "Structural change"
        
        if delta.magnitude > threshold:
            return False, f"Delta {delta.magnitude:.1%} > threshold {threshold:.1%}"
        
        return True, f"Delta {delta.magnitude:.1%} within threshold"


# ============================================================================
# ISM Statistics Collector
# ============================================================================

@dataclass
class ISMStatistics:
    """Comprehensive ISM statistics for paper"""
    total_queries: int = 0
    ism_queries: int = 0
    full_sim_queries: int = 0
    
    total_ism_time_ms: float = 0
    total_full_sim_time_ms: float = 0
    
    errors: List[float] = field(default_factory=list)
    speedups: List[float] = field(default_factory=list)
    
    @property
    def ism_rate(self) -> float:
        return self.ism_queries / max(self.total_queries, 1)
    
    @property
    def mean_error(self) -> float:
        return np.mean(self.errors) if self.errors else 0
    
    @property
    def p95_error(self) -> float:
        return np.percentile(self.errors, 95) if self.errors else 0
    
    @property
    def mean_speedup(self) -> float:
        return np.mean(self.speedups) if self.speedups else 1
    
    @property
    def total_speedup(self) -> float:
        baseline = self.total_ism_time_ms + self.total_full_sim_time_ms
        # If all were full sim
        estimated_full = self.total_queries * (self.total_full_sim_time_ms / max(self.full_sim_queries, 1))
        return estimated_full / max(baseline, 1)
    
    def summary(self) -> str:
        return f"""
ISM Statistics Summary
======================
Total queries:     {self.total_queries}
ISM queries:       {self.ism_queries} ({self.ism_rate:.1%})
Full sim queries:  {self.full_sim_queries}

Mean error:        {self.mean_error:.2%}
95th %ile error:   {self.p95_error:.2%}
Mean speedup:      {self.mean_speedup:.1f}x
Total speedup:     {self.total_speedup:.1f}x
"""
