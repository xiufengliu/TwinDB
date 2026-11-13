"""
ISM Failure Case Experiment
===========================
Tests ISM on a simulation with discontinuity (thermostat comfort threshold)
"""

import numpy as np

def simulate_with_comfort_threshold(setpoint: float, base_setpoint: float = 20.0) -> float:
    """
    Simulate building energy with comfort threshold discontinuity.
    
    Below 18°C: auxiliary heating kicks in (non-linear jump)
    Above 18°C: linear relationship
    """
    base_energy = 50.0  # MWh at base setpoint
    linear_sensitivity = 2.0  # MWh per degree
    
    # Linear component
    energy = base_energy + linear_sensitivity * (setpoint - base_setpoint)
    
    # Non-linear: comfort threshold at 18°C triggers auxiliary heating
    if setpoint < 18.0:
        # Auxiliary heating adds significant energy
        comfort_penalty = 10.0 * (18.0 - setpoint)  # Steep increase below threshold
        energy += comfort_penalty
    
    return energy


def experiment_failure_case():
    """Test ISM on discontinuous simulation"""
    print("=" * 60)
    print("ISM FAILURE CASE: Thermostat with Comfort Threshold")
    print("=" * 60)
    
    # Learn sensitivity from points above threshold (linear region)
    train_setpoints = [20.0, 21.0, 22.0, 19.0]
    train_energies = [simulate_with_comfort_threshold(s) for s in train_setpoints]
    
    # Fit linear model
    beta = np.polyfit(train_setpoints, train_energies, 1)[0]
    alpha = np.mean(train_energies) - beta * np.mean(train_setpoints)
    
    print(f"\nLearned from linear region (setpoint >= 18°C):")
    print(f"  β (sensitivity) = {beta:.2f} MWh/°C")
    print(f"  α (intercept) = {alpha:.2f} MWh")
    
    # Test predictions
    print(f"\n{'Setpoint':>10} {'ISM Pred':>12} {'Actual':>12} {'Error':>10} {'Error%':>10}")
    print("-" * 60)
    
    test_setpoints = [22.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0]
    base_setpoint = 20.0
    base_energy = simulate_with_comfort_threshold(base_setpoint)
    
    results = []
    for sp in test_setpoints:
        actual = simulate_with_comfort_threshold(sp)
        ism_pred = base_energy + beta * (sp - base_setpoint)
        error = abs(ism_pred - actual)
        error_pct = 100 * error / actual
        
        results.append({
            'setpoint': sp,
            'predicted': ism_pred,
            'actual': actual,
            'error': error,
            'error_pct': error_pct
        })
        
        print(f"{sp:>10.1f} {ism_pred:>12.2f} {actual:>12.2f} {error:>10.2f} {error_pct:>10.1f}%")
    
    # Summary
    print("\n--- Summary ---")
    linear_region = [r for r in results if r['setpoint'] >= 18.0]
    nonlinear_region = [r for r in results if r['setpoint'] < 18.0]
    
    print(f"Linear region (≥18°C): Max error = {max(r['error_pct'] for r in linear_region):.2f}%")
    print(f"Non-linear region (<18°C): Max error = {max(r['error_pct'] for r in nonlinear_region):.2f}%")
    
    # Detect failure via residual
    print("\n--- Failure Detection ---")
    sigma_beta = 0.1  # Estimated uncertainty
    for r in results:
        delta_p = abs(r['setpoint'] - base_setpoint)
        expected_bound = 3 * sigma_beta * delta_p  # 3-sigma bound
        residual = r['error']
        detected = residual > expected_bound and expected_bound > 0
        print(f"Setpoint {r['setpoint']}: residual={residual:.2f}, bound={expected_bound:.2f}, "
              f"FAILURE_DETECTED={detected}")
    
    return results


def experiment_prior_sensitivity():
    """Test RLS convergence with good vs bad prior"""
    print("\n" + "=" * 60)
    print("PRIOR SENSITIVITY EXPERIMENT")
    print("=" * 60)
    
    # True model: E = 50 + 40*U
    true_alpha = 50.0
    true_beta = 40.0
    noise_std = 0.5
    
    np.random.seed(42)
    
    def simulate(u):
        return true_alpha + true_beta * u + np.random.normal(0, noise_std)
    
    # RLS with good prior
    def run_rls(beta_0, n_samples=15):
        theta = np.array([50.0, beta_0])  # [alpha, beta]
        P = 1000.0 * np.eye(2)
        lambda_forget = 0.99
        
        errors = []
        u_values = np.linspace(0.3, 0.7, n_samples)
        
        for u in u_values:
            y = simulate(u)
            x = np.array([1.0, u])
            
            # Prediction error before update
            pred = theta[0] + theta[1] * u
            errors.append(abs(theta[1] - true_beta))
            
            # RLS update
            K = P @ x / (lambda_forget + x @ P @ x)
            theta = theta + K * (y - x @ theta)
            P = (P - np.outer(K, x @ P)) / lambda_forget
        
        return errors
    
    # Good prior (physics-informed)
    good_prior = 40.0  # Close to true value
    errors_good = run_rls(good_prior)
    
    # Bad prior
    bad_prior = 0.0  # Far from true value
    errors_bad = run_rls(bad_prior)
    
    print(f"\n{'Samples':>8} {'Good Prior':>15} {'Bad Prior':>15}")
    print("-" * 40)
    for i in range(0, 15, 2):
        print(f"{i+1:>8} {errors_good[i]:>15.4f} {errors_bad[i]:>15.4f}")
    
    # Find convergence point (error < 1.0)
    good_converge = next((i for i, e in enumerate(errors_good) if e < 1.0), 15)
    bad_converge = next((i for i, e in enumerate(errors_bad) if e < 1.0), 15)
    
    print(f"\nConvergence (|β - β_true| < 1.0):")
    print(f"  Good prior (β₀={good_prior}): {good_converge + 1} samples")
    print(f"  Bad prior (β₀={bad_prior}): {bad_converge + 1} samples")
    
    return {
        'good_prior_convergence': good_converge + 1,
        'bad_prior_convergence': bad_converge + 1
    }


if __name__ == "__main__":
    results_failure = experiment_failure_case()
    results_prior = experiment_prior_sensitivity()
    
    print("\n" + "=" * 60)
    print("REAL NUMBERS FOR PAPER")
    print("=" * 60)
    
    # Get the actual failure case numbers
    failure_17 = next(r for r in results_failure if r['setpoint'] == 17.0)
    print(f"\nISM Failure Case (setpoint=17°C):")
    print(f"  ISM predicted: {failure_17['predicted']:.1f} MWh")
    print(f"  Actual: {failure_17['actual']:.1f} MWh")
    print(f"  Error: {failure_17['error_pct']:.1f}%")
    
    print(f"\nPrior Sensitivity:")
    print(f"  Good prior convergence: {results_prior['good_prior_convergence']} samples")
    print(f"  Bad prior convergence: {results_prior['bad_prior_convergence']} samples")
