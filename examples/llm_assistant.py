#!/usr/bin/env python3
"""
LLM-Enhanced TwinDB Demo
========================

Demonstrates the novel LLM integration:
1. LLM-guided execution planning (when to use cache vs simulate)
2. LLM as surrogate model (fast approximate answers)
3. Result explanation generation

This demonstrates the key contribution: intelligent hybrid execution
that goes beyond simple caching.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
warnings.filterwarnings('ignore')


def demo_llm_optimizer():
    """Demo 1: LLM decides execution strategy"""
    print("\n" + "=" * 70)
    print("Demo 1: LLM-Guided Execution Planning")
    print("=" * 70)
    
    from core.llm import LLMOptimizer
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set. Set it to run this demo.")
        print("   export DEEPSEEK_API_KEY='your-key-here'")
        return
    
    optimizer = LLMOptimizer(api_key)
    
    # Scenario 1: Exact match should use cache
    print("\n--- Test 1: Query with potential cache hit ---")
    plan = optimizer.plan_execution(
        tid='Twin_B123',
        target_scenario={'retrofit_package': {'wall_u_after': 0.18}},
        error_tolerance=0.05
    )
    print(f"Strategy: {plan.strategy.value}")
    print(f"Confidence: {plan.confidence:.2f}")
    print(f"Est. Error: {plan.estimated_error:.1f}%")
    print(f"Est. Time: {plan.estimated_time_ms:.0f}ms")
    print(f"Reasoning: {plan.reasoning}")
    
    # Scenario 2: Similar scenario - might interpolate
    print("\n--- Test 2: Similar scenario (might interpolate) ---")
    plan = optimizer.plan_execution(
        tid='Twin_B123',
        target_scenario={'retrofit_package': {'wall_u_after': 0.20}},  # Slightly different
        error_tolerance=0.10  # Higher tolerance
    )
    print(f"Strategy: {plan.strategy.value}")
    print(f"Confidence: {plan.confidence:.2f}")
    print(f"Est. Error: {plan.estimated_error:.1f}%")
    print(f"Reasoning: {plan.reasoning}")
    
    # Scenario 3: Very different - needs full simulation
    print("\n--- Test 3: Novel scenario (needs full sim) ---")
    plan = optimizer.plan_execution(
        tid='Twin_B123',
        target_scenario={'retrofit_package': {
            'wall_u_after': 0.10,
            'roof_u_after': 0.08,
            'window_u_after': 0.70,
            'add_heat_pump': True
        }},
        error_tolerance=0.01  # Low tolerance
    )
    print(f"Strategy: {plan.strategy.value}")
    print(f"Confidence: {plan.confidence:.2f}")
    print(f"Reasoning: {plan.reasoning}")
    
    print(f"\nOptimizer stats: {optimizer.stats}")


def demo_llm_surrogate():
    """Demo 2: LLM as surrogate model"""
    print("\n" + "=" * 70)
    print("Demo 2: LLM as Surrogate Model")
    print("=" * 70)
    
    from core.llm import LLMSurrogateModel
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set.")
        return
    
    surrogate = LLMSurrogateModel(api_key)
    
    # Test surrogate estimation
    baseline_kwh = 278775  # From actual data
    
    scenarios = [
        {'retrofit_package': {'wall_u_after': 0.18}},
        {'retrofit_package': {'wall_u_after': 0.18, 'window_u_after': 0.90}},
        {'retrofit_package': {'wall_u_after': 0.12, 'window_u_after': 0.70, 'roof_u_after': 0.10}},
    ]
    
    print(f"\nBaseline annual heat load: {baseline_kwh:,.0f} kWh")
    print("\nLLM Surrogate Estimates:")
    print("-" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        start = time.perf_counter()
        estimated, confidence, reasoning = surrogate.estimate(baseline_kwh, scenario)
        elapsed = (time.perf_counter() - start) * 1000
        
        saving_pct = (baseline_kwh - estimated) / baseline_kwh * 100
        
        print(f"\nScenario {i}: {scenario}")
        print(f"  Estimated: {estimated:,.0f} kWh (saving: {saving_pct:.1f}%)")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Time: {elapsed:.0f}ms")
        print(f"  Reasoning: {reasoning[:100]}...")


def demo_smart_comparison():
    """Demo 3: Full smart comparison"""
    print("\n" + "=" * 70)
    print("Demo 3: Smart Scenario Comparison")
    print("=" * 70)
    
    from core.llm import LLMEnhancedEngine
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set.")
        return
    
    engine = LLMEnhancedEngine(api_key)
    
    # Compare with different error tolerances
    scenario = {'retrofit_package': {'wall_u_after': 0.18, 'window_u_after': 0.90}}
    
    print("\n--- High accuracy requirement (1% error tolerance) ---")
    result = engine.compare_scenario_smart(
        tid='Twin_B123',
        scenario_cfg=scenario,
        model_id='BuildingHeat_v3',
        window=('2018-01-01', '2019-01-01'),
        error_tolerance=0.01
    )
    print(f"Strategy used: {result['execution']['strategy']}")
    print(f"Time: {result['execution']['actual_time_ms']:.0f}ms")
    print(f"Reasoning: {result['execution']['reasoning']}")
    
    print("\n--- Low accuracy requirement (10% error tolerance) ---")
    result = engine.compare_scenario_smart(
        tid='Twin_B123',
        scenario_cfg=scenario,
        model_id='BuildingHeat_v3',
        window=('2018-01-01', '2019-01-01'),
        error_tolerance=0.10
    )
    print(f"Strategy used: {result['execution']['strategy']}")
    print(f"Time: {result['execution']['actual_time_ms']:.0f}ms")
    print(f"Reasoning: {result['execution']['reasoning']}")


def demo_comparison_with_baseline():
    """Demo 4: Compare LLM-enhanced vs standard execution"""
    print("\n" + "=" * 70)
    print("Demo 4: LLM-Enhanced vs Standard Execution")
    print("=" * 70)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set.")
        return
    
    from core.compiler import TwinQLEngine
    from core.llm import LLMEnhancedEngine
    from dsl.twinql import define_scenario
    from core.db import execute_sql
    
    # Clear cache for fair comparison
    execute_sql("DELETE FROM sim_cache WHERE tid = 'Twin_B123'")
    execute_sql("DELETE FROM timeseries WHERE tid = 'Twin_B123' AND cid != 'REALITY'")
    
    standard = TwinQLEngine()
    enhanced = LLMEnhancedEngine(api_key)
    
    scenarios = [
        {'retrofit_package': {'wall_u_after': 0.18}},
        {'retrofit_package': {'wall_u_after': 0.19}},  # Very similar
        {'retrofit_package': {'wall_u_after': 0.20}},  # Similar
        {'retrofit_package': {'wall_u_after': 0.15}},  # Different
    ]
    
    print("\nRunning 4 scenario queries...")
    print("-" * 70)
    
    # Standard execution (always full sim or cache)
    print("\nStandard Engine:")
    standard_total = 0
    for i, cfg in enumerate(scenarios):
        define_scenario(f'std_{i}', **cfg)
        start = time.perf_counter()
        standard.compare_scenario('Twin_B123', f'std_{i}', 'BuildingHeat_v3', 
                                  ('2018-01-01', '2019-01-01'))
        elapsed = (time.perf_counter() - start) * 1000
        standard_total += elapsed
        print(f"  Scenario {i+1}: {elapsed:.0f}ms")
    print(f"  Total: {standard_total:.0f}ms")
    
    # Clear cache again
    execute_sql("DELETE FROM sim_cache WHERE tid = 'Twin_B123'")
    execute_sql("DELETE FROM timeseries WHERE tid = 'Twin_B123' AND cid != 'REALITY'")
    
    # LLM-enhanced execution
    print("\nLLM-Enhanced Engine (10% error tolerance):")
    enhanced_total = 0
    for i, cfg in enumerate(scenarios):
        start = time.perf_counter()
        result = enhanced.compare_scenario_smart(
            'Twin_B123', cfg, 'BuildingHeat_v3',
            ('2018-01-01', '2019-01-01'),
            error_tolerance=0.10
        )
        elapsed = result['execution']['actual_time_ms']
        enhanced_total += elapsed
        print(f"  Scenario {i+1}: {elapsed:.0f}ms ({result['execution']['strategy']})")
    print(f"  Total: {enhanced_total:.0f}ms")
    
    print(f"\nSpeedup: {standard_total/enhanced_total:.1f}x")
    print(f"LLM optimizer stats: {enhanced.optimizer.stats}")


if __name__ == '__main__':
    print("=" * 70)
    print("TwinDB LLM Integration Demo")
    print("=" * 70)
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n⚠️  To run this demo, set your DeepSeek API key:")
        print("   export DEEPSEEK_API_KEY='your-key-here'")
        print("\nThe demo will show:")
        print("1. LLM-guided execution planning")
        print("2. LLM as surrogate model for fast estimates")
        print("3. Smart scenario comparison with error tolerance")
        print("4. Performance comparison vs standard execution")
    else:
        demo_llm_optimizer()
        demo_llm_surrogate()
        demo_smart_comparison()
        demo_comparison_with_baseline()
