"""
Incremental Scenario Maintenance (ISM) for TwinDB

Implements incremental updates to simulation results under scenario changes.
Instead of re-running full simulations for small parameter changes, ISM
computes approximate results using learned sensitivity coefficients.

Core Insight:
  Traditional: S' = Sim(τ, c', m)           -- Full re-simulation O(T)
  ISM:         S' = S + ΔS(Δc)              -- Incremental update O(1)

For physics-based building simulation, energy consumption E is:
  E = f(U, A, ΔT, t) where U=U-value, A=area, ΔT=temp diff, t=time

Heat loss: Q = U × A × ΔT × t

Key property: Q is LINEAR in U for fixed geometry.
Therefore: ΔQ/ΔU = A × ΔT × t (constant for given conditions)

This enables EXACT incremental updates for U-value changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd
import hashlib
import json

from core.db import execute_query, execute_sql


# ============================================================================
# Scenario Delta Analysis
# ============================================================================

class DeltaType(Enum):
    """Types of scenario changes"""
    LINEAR = "linear"           # Linear effect (U-value changes)
    MULTIPLICATIVE = "mult"     # Multiplicative effect (efficiency)
    NONLINEAR = "nonlinear"     # Requires re-simulation
    STRUCTURAL = "structural"   # Topology change, must re-simulate


@dataclass
class ScenarioDelta:
    """Represents the difference between two scenarios"""
    base_scenario: str
    param_deltas: Dict[str, Tuple[float, float]]  # param -> (old, new)
    delta_type: DeltaType
    magnitude: float  # Normalized magnitude of change
    
    @classmethod
    def compute(cls, base_cfg: Dict, new_cfg: Dict) -> 'ScenarioDelta':
        """Compute delta between two scenario configurations"""
        deltas = {}
        max_magnitude = 0.0
        delta_type = DeltaType.LINEAR  # Assume linear until proven otherwise
        
        # Extract retrofit packages
        base_pkg = base_cfg.get('retrofit_package', base_cfg)
        new_pkg = new_cfg.get('retrofit_package', new_cfg)
        
        all_keys = set(base_pkg.keys()) | set(new_pkg.keys())
        
        for key in all_keys:
            base_val = base_pkg.get(key, None)
            new_val = new_pkg.get(key, None)
            
            if base_val != new_val:
                # Check for structural changes
                if base_val is None or new_val is None:
                    delta_type = DeltaType.STRUCTURAL
                elif isinstance(new_val, bool) or isinstance(base_val, bool):
                    delta_type = DeltaType.STRUCTURAL
                elif key in ('add_heat_pump', 'add_solar', 'change_hvac'):
                    delta_type = DeltaType.STRUCTURAL
                else:
                    # Numeric change
                    deltas[key] = (float(base_val), float(new_val))
                    rel_change = abs(new_val - base_val) / max(abs(base_val), 0.01)
                    max_magnitude = max(max_magnitude, rel_change)
                    
                    # Check if nonlinear
                    if key in ('infiltration_rate', 'ventilation_rate'):
                        delta_type = DeltaType.MULTIPLICATIVE
        
        return cls(
            base_scenario="",
            param_deltas=deltas,
            delta_type=delta_type,
            magnitude=max_magnitude
        )


# ============================================================================
# Sensitivity Model
# ============================================================================

@dataclass
class SensitivityCoefficient:
    """Sensitivity of energy to parameter changes: ∂E/∂p"""
    parameter: str
    coefficient: float      # ∂E/∂p
    uncertainty: float      # Standard error
    valid_range: Tuple[float, float]  # Range where linear approximation holds


class SensitivityModel:
    """
    Learns and stores sensitivity coefficients for incremental updates.
    
    For building energy: E = Σ (Uᵢ × Aᵢ × HDD × 24)
    where HDD = heating degree days
    
    Sensitivity: ∂E/∂Uᵢ = Aᵢ × HDD × 24 (constant for given climate)
    """
    
    # Pre-computed sensitivities for Copenhagen climate (HDD ≈ 3000)
    # These would be learned from data in production
    DEFAULT_SENSITIVITIES = {
        'wall_u_after': SensitivityCoefficient(
            parameter='wall_u_after',
            coefficient=150000,  # kWh per W/m²K change (typical 100m² building)
            uncertainty=0.05,
            valid_range=(0.10, 0.80)
        ),
        'window_u_after': SensitivityCoefficient(
            parameter='window_u_after',
            coefficient=50000,   # Windows ~1/3 of wall effect
            uncertainty=0.08,
            valid_range=(0.70, 3.00)
        ),
        'roof_u_after': SensitivityCoefficient(
            parameter='roof_u_after',
            coefficient=80000,   # Roof ~1/2 of wall effect
            uncertainty=0.06,
            valid_range=(0.08, 0.50)
        ),
    }
    
    def __init__(self):
        self.sensitivities = dict(self.DEFAULT_SENSITIVITIES)
        self.learned_corrections: Dict[str, float] = {}
    
    def get_sensitivity(self, param: str) -> Optional[SensitivityCoefficient]:
        """Get sensitivity coefficient for parameter"""
        return self.sensitivities.get(param)
    
    def learn_from_data(self, tid: str, param: str, 
                        observations: List[Tuple[float, float]]):
        """
        Learn sensitivity from observed (param_value, energy) pairs.
        Uses linear regression: E = α + β × param
        """
        if len(observations) < 2:
            return
        
        x = np.array([o[0] for o in observations])
        y = np.array([o[1] for o in observations])
        
        # Linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x**2) - np.sum(x)**2)
        
        # Update sensitivity
        key = f"{tid}:{param}"
        self.learned_corrections[key] = slope
    
    def predict_delta(self, param: str, old_val: float, new_val: float,
                      tid: str = None) -> Tuple[float, float]:
        """
        Predict energy change from parameter change.
        Returns: (delta_energy, uncertainty)
        """
        sens = self.get_sensitivity(param)
        if sens is None:
            return 0.0, float('inf')  # Unknown parameter
        
        # Check if we have learned correction
        correction = 1.0
        if tid:
            key = f"{tid}:{param}"
            if key in self.learned_corrections:
                correction = self.learned_corrections[key] / sens.coefficient
        
        delta_param = new_val - old_val
        delta_energy = sens.coefficient * correction * delta_param
        uncertainty = abs(delta_energy) * sens.uncertainty
        
        return delta_energy, uncertainty


# ============================================================================
# Incremental Update Engine
# ============================================================================

@dataclass
class IncrementalResult:
    """Result of incremental scenario update"""
    success: bool
    estimated_energy: float
    error_bound: float          # Absolute error bound
    error_bound_pct: float      # Relative error bound
    method: str                 # 'exact', 'linear', 'recompute'
    base_scenario: str
    computation_time_ms: float


class IncrementalUpdateEngine:
    """
    Core ISM engine: computes incremental updates to simulation results.
    
    Algorithm:
    1. Compute scenario delta Δc = c' - c
    2. Classify delta type (linear, multiplicative, structural)
    3. If linear: S' = S + Σ (∂S/∂pᵢ × Δpᵢ)
    4. If multiplicative: S' = S × Π (1 + αᵢ × Δpᵢ)
    5. If structural: trigger full re-simulation
    """
    
    # Thresholds for incremental update decisions
    MAX_LINEAR_DELTA = 0.30      # Max 30% parameter change for linear approx
    MAX_ERROR_TOLERANCE = 0.10   # Max 10% error for incremental
    
    def __init__(self):
        self.sensitivity_model = SensitivityModel()
        self.stats = {
            'incremental_updates': 0,
            'full_resimulations': 0,
            'total_error': 0.0,
            'max_error': 0.0
        }
    
    def can_update_incrementally(self, delta: ScenarioDelta, 
                                  error_tolerance: float = 0.10) -> Tuple[bool, str]:
        """
        Decide if incremental update is possible.
        Returns: (can_update, reason)
        """
        if delta.delta_type == DeltaType.STRUCTURAL:
            return False, "Structural change requires full re-simulation"
        
        if delta.magnitude > self.MAX_LINEAR_DELTA:
            return False, f"Parameter change {delta.magnitude:.1%} exceeds linear threshold"
        
        # Check if all parameters have known sensitivities
        for param in delta.param_deltas:
            if self.sensitivity_model.get_sensitivity(param) is None:
                return False, f"Unknown sensitivity for parameter: {param}"
        
        return True, "Incremental update possible"
    
    def compute_incremental_update(self, 
                                    tid: str,
                                    base_result: pd.DataFrame,
                                    base_cfg: Dict,
                                    new_cfg: Dict) -> IncrementalResult:
        """
        Compute incremental update to simulation result.
        
        Args:
            tid: Twin ID
            base_result: Cached simulation result (DataFrame with value_retrofit)
            base_cfg: Base scenario configuration
            new_cfg: New scenario configuration
        
        Returns:
            IncrementalResult with estimated energy and error bounds
        """
        import time
        start = time.perf_counter()
        
        # Compute delta
        delta = ScenarioDelta.compute(base_cfg, new_cfg)
        
        # Check if incremental is possible
        can_update, reason = self.can_update_incrementally(delta)
        
        if not can_update:
            self.stats['full_resimulations'] += 1
            return IncrementalResult(
                success=False,
                estimated_energy=0.0,
                error_bound=float('inf'),
                error_bound_pct=100.0,
                method='recompute',
                base_scenario="",
                computation_time_ms=(time.perf_counter() - start) * 1000
            )
        
        # Get base energy
        if 'value_retrofit' in base_result.columns:
            base_energy = float(base_result['value_retrofit'].sum())
        else:
            base_energy = float(base_result['value'].sum())
        
        # Compute incremental update
        total_delta = 0.0
        total_uncertainty = 0.0
        
        for param, (old_val, new_val) in delta.param_deltas.items():
            delta_e, uncertainty = self.sensitivity_model.predict_delta(
                param, old_val, new_val, tid
            )
            total_delta += delta_e
            total_uncertainty += uncertainty ** 2  # Sum of variances
        
        total_uncertainty = np.sqrt(total_uncertainty)
        
        # Compute result
        estimated_energy = base_energy + total_delta
        error_bound_pct = total_uncertainty / max(estimated_energy, 1) * 100
        
        self.stats['incremental_updates'] += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return IncrementalResult(
            success=True,
            estimated_energy=estimated_energy,
            error_bound=total_uncertainty,
            error_bound_pct=error_bound_pct,
            method='linear',
            base_scenario="",
            computation_time_ms=elapsed
        )
    
    def validate_and_learn(self, tid: str, param: str,
                           predicted: float, actual: float,
                           param_value: float):
        """
        Validate prediction against actual and update model.
        Called after full simulation to improve future predictions.
        """
        error = abs(predicted - actual) / max(actual, 1)
        self.stats['total_error'] += error
        self.stats['max_error'] = max(self.stats['max_error'], error)
        
        # Could implement online learning here
        # self.sensitivity_model.learn_from_data(tid, param, [(param_value, actual)])


# ============================================================================
# ISM-Aware Query Optimizer
# ============================================================================

class ISMOptimizer:
    """
    Query optimizer that leverages ISM for scenario queries.
    
    Optimization Rules:
    1. If cached result exists for similar scenario → use ISM
    2. If ISM error bound exceeds tolerance → full simulation
    3. Group similar scenarios for batch ISM updates
    """
    
    def __init__(self):
        self.ism_engine = IncrementalUpdateEngine()
        self.cache: Dict[str, Tuple[Dict, pd.DataFrame]] = {}  # sig -> (cfg, result)
    
    def find_best_base(self, new_cfg: Dict) -> Optional[Tuple[str, Dict, pd.DataFrame]]:
        """Find the best cached scenario to use as ISM base"""
        best_sig = None
        best_delta = None
        min_magnitude = float('inf')
        
        for sig, (cfg, result) in self.cache.items():
            delta = ScenarioDelta.compute(cfg, new_cfg)
            if delta.magnitude < min_magnitude:
                min_magnitude = delta.magnitude
                best_sig = sig
                best_delta = delta
        
        if best_sig and min_magnitude < IncrementalUpdateEngine.MAX_LINEAR_DELTA:
            cfg, result = self.cache[best_sig]
            return best_sig, cfg, result
        
        return None
    
    def execute_with_ism(self, tid: str, new_cfg: Dict,
                         full_sim_func, error_tolerance: float = 0.10) -> Dict:
        """
        Execute scenario query with ISM optimization.
        
        Args:
            tid: Twin ID
            new_cfg: New scenario configuration
            full_sim_func: Function to run full simulation if needed
            error_tolerance: Maximum acceptable error
        
        Returns:
            Dict with result and execution metadata
        """
        import time
        start = time.perf_counter()
        
        # Try to find base for ISM
        base = self.find_best_base(new_cfg)
        
        if base:
            base_sig, base_cfg, base_result = base
            
            # Try incremental update
            ism_result = self.ism_engine.compute_incremental_update(
                tid, base_result, base_cfg, new_cfg
            )
            
            if ism_result.success and ism_result.error_bound_pct <= error_tolerance * 100:
                elapsed = (time.perf_counter() - start) * 1000
                return {
                    'energy': ism_result.estimated_energy,
                    'method': 'ISM',
                    'error_bound_pct': ism_result.error_bound_pct,
                    'base_scenario': base_sig,
                    'time_ms': elapsed
                }
        
        # Fall back to full simulation
        result = full_sim_func(new_cfg)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Cache the result
        sig = hashlib.md5(json.dumps(new_cfg, sort_keys=True).encode()).hexdigest()[:16]
        if hasattr(result, 'copy'):
            self.cache[sig] = (new_cfg, result.copy())
        
        energy = float(result['value_retrofit'].sum()) if 'value_retrofit' in result.columns else 0
        
        return {
            'energy': energy,
            'method': 'FULL_SIM',
            'error_bound_pct': 0.0,
            'base_scenario': None,
            'time_ms': elapsed
        }
    
    def get_stats(self) -> Dict:
        """Get ISM statistics"""
        return {
            'cache_size': len(self.cache),
            'ism_stats': self.ism_engine.stats
        }
