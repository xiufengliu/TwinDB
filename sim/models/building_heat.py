"""Building Heat Load Model for TwinDB - Mock implementation for v0.1"""
import pandas as pd
import numpy as np

def building_heat_model(hist_df: pd.DataFrame, scenario_cfg: dict, asset: dict) -> pd.DataFrame:
    """
    Mock building heat model that simulates retrofit effects.
    
    For v0.1, this applies a simple scaling factor based on retrofit parameters.
    Real implementation would use physics-based or ML models.
    
    Args:
        hist_df: Historical timeseries with columns ['ts', 'value']
        scenario_cfg: Scenario configuration with retrofit_package
        asset: Asset information dict
    
    Returns:
        DataFrame with simulated ['ts', 'value']
    """
    if hist_df.empty:
        return hist_df.copy()
    
    result = hist_df.copy()
    retrofit = scenario_cfg.get('retrofit_package', {})
    
    # Calculate reduction factor based on U-value improvements
    # Simplified: assume heat loss proportional to U-values
    wall_factor = retrofit.get('wall_u_after', 0.6) / retrofit.get('wall_u_before', 0.6)
    roof_factor = retrofit.get('roof_u_after', 0.3) / retrofit.get('roof_u_before', 0.3)
    window_factor = retrofit.get('window_u_after', 2.8) / retrofit.get('window_u_before', 2.8)
    infiltration_factor = retrofit.get('infiltration_after', 0.8) / retrofit.get('infiltration_before', 0.8)
    
    # Weighted average (rough approximation of building envelope contribution)
    # Walls ~40%, Roof ~15%, Windows ~25%, Infiltration ~20%
    overall_factor = (0.40 * wall_factor + 0.15 * roof_factor + 
                      0.25 * window_factor + 0.20 * infiltration_factor)
    
    # Apply reduction with small random noise for realism
    np.random.seed(42)  # Reproducibility
    noise = np.random.normal(1.0, 0.02, len(result))
    result['value'] = result['value'] * overall_factor * noise
    result['value'] = result['value'].clip(lower=0)  # No negative heat load
    
    return result
