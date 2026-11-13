"""
LLM-Enhanced Query Optimizer for TwinDB

Uses LLM reasoning to make intelligent decisions about:
1. When to use cached results vs full simulation
2. How to estimate approximation error
3. When caching/approximation is safe

This component addresses the question: when is caching/approximation safe?
"""
import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from core.db import execute_query


# ============================================================================
# Execution Strategies
# ============================================================================

class ExecutionStrategy(Enum):
    FULL_SIMULATION = "full_sim"       # Accurate but slow (~1700ms)
    CACHE_HIT = "cache_hit"            # Exact match in cache (~50ms)
    INTERPOLATE = "interpolate"        # Linear interpolation from similar cached (~80ms)
    LLM_SURROGATE = "llm_surrogate"    # LLM predicts result directly (~200ms)


@dataclass
class ExecutionPlan:
    strategy: ExecutionStrategy
    confidence: float           # 0-1, how confident in this choice
    estimated_error: float      # Expected error percentage
    estimated_time_ms: float    # Expected execution time
    reasoning: str              # LLM's explanation
    reference_scenarios: List[str] = None  # For interpolation


# ============================================================================
# DeepSeek LLM Client
# ============================================================================

class DeepSeekClient:
    """Client for DeepSeek API using OpenAI SDK"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"
    
    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Send chat completion request"""
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content


# ============================================================================
# LLM-Guided Optimizer
# ============================================================================

class LLMOptimizer:
    """
    Uses LLM to make intelligent execution decisions.
    
    Key insight: LLM can reason about scenario similarity and estimate
    whether approximation is safe, something rule-based optimizers cannot do.
    """
    
    SYSTEM_PROMPT = """You are a building energy simulation expert helping optimize query execution.

Given a target scenario and cached scenarios, decide the best execution strategy:
1. FULL_SIMULATION: Run complete physics simulation (accurate but slow, ~1700ms)
2. CACHE_HIT: Use exact cached result (fast, ~50ms, 0% error)
3. INTERPOLATE: Linearly interpolate from similar cached scenarios (fast, ~80ms, some error)
4. LLM_SURROGATE: You estimate the result directly based on physics knowledge (~200ms)

Consider:
- Parameter similarity (wall_u, window_u, roof_u values)
- Physical relationships (U-value reduction → proportional heat loss reduction)
- Error tolerance (user may accept 5% error for 10x speedup)

Respond in JSON format:
{
  "strategy": "FULL_SIMULATION|CACHE_HIT|INTERPOLATE|LLM_SURROGATE",
  "confidence": 0.0-1.0,
  "estimated_error_pct": 0.0-100.0,
  "reasoning": "brief explanation",
  "interpolation_weights": {"scenario_id": weight} // if INTERPOLATE
}"""

    def __init__(self, api_key: str = None):
        self.llm = DeepSeekClient(api_key)
        self.stats = {"llm_calls": 0, "cache_hits": 0, "interpolations": 0, "full_sims": 0}
    
    def get_cached_scenarios(self, tid: str) -> List[Dict]:
        """Retrieve cached scenarios for a twin"""
        try:
            result = execute_query("""
                SELECT DISTINCT s.cid, s.cfg 
                FROM sim_cache sc
                JOIN scenario s ON sc.cid = s.cid
                WHERE sc.tid = %s
            """, (tid,), as_df=False)
            return result if result else []
        except:
            return []
    
    def plan_execution(self, tid: str, target_scenario: Dict, 
                       error_tolerance: float = 0.05) -> ExecutionPlan:
        """
        Use LLM to decide optimal execution strategy.
        
        Args:
            tid: Twin ID
            target_scenario: Scenario configuration to simulate
            error_tolerance: Acceptable error (e.g., 0.05 = 5%)
        
        Returns:
            ExecutionPlan with strategy and reasoning
        """
        # Get cached scenarios
        cached = self.get_cached_scenarios(tid)
        
        # Check for exact cache hit first (no LLM needed)
        for c in cached:
            if self._scenarios_match(c.get('cfg', {}), target_scenario):
                self.stats["cache_hits"] += 1
                return ExecutionPlan(
                    strategy=ExecutionStrategy.CACHE_HIT,
                    confidence=1.0,
                    estimated_error=0.0,
                    estimated_time_ms=50,
                    reasoning="Exact match found in cache",
                    reference_scenarios=[c['cid']]
                )
        
        # No exact match - ask LLM for strategy
        self.stats["llm_calls"] += 1
        
        user_prompt = f"""Target scenario to simulate:
{json.dumps(target_scenario, indent=2)}

Cached scenarios available:
{json.dumps([{'cid': c['cid'], 'cfg': c.get('cfg', {})} for c in cached], indent=2)}

Error tolerance: {error_tolerance * 100}%

What execution strategy should we use?"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])
            
            # Parse LLM response
            plan = self._parse_llm_response(response, cached)
            return plan
            
        except Exception as e:
            # Fallback to full simulation if LLM fails
            return ExecutionPlan(
                strategy=ExecutionStrategy.FULL_SIMULATION,
                confidence=1.0,
                estimated_error=0.0,
                estimated_time_ms=1700,
                reasoning=f"LLM unavailable ({str(e)}), using full simulation"
            )
    
    def _scenarios_match(self, cfg1: Dict, cfg2: Dict) -> bool:
        """Check if two scenario configs are equivalent"""
        # Normalize and compare
        def normalize(cfg):
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except:
                    return cfg
            return json.dumps(cfg, sort_keys=True)
        return normalize(cfg1) == normalize(cfg2)
    
    def _parse_llm_response(self, response: str, cached: List[Dict]) -> ExecutionPlan:
        """Parse LLM JSON response into ExecutionPlan"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
            else:
                raise ValueError("No JSON found")
            
            strategy_map = {
                "FULL_SIMULATION": ExecutionStrategy.FULL_SIMULATION,
                "CACHE_HIT": ExecutionStrategy.CACHE_HIT,
                "INTERPOLATE": ExecutionStrategy.INTERPOLATE,
                "LLM_SURROGATE": ExecutionStrategy.LLM_SURROGATE,
            }
            
            strategy = strategy_map.get(data.get("strategy", "FULL_SIMULATION"), 
                                        ExecutionStrategy.FULL_SIMULATION)
            
            time_estimates = {
                ExecutionStrategy.FULL_SIMULATION: 1700,
                ExecutionStrategy.CACHE_HIT: 50,
                ExecutionStrategy.INTERPOLATE: 80,
                ExecutionStrategy.LLM_SURROGATE: 200,
            }
            
            # Track stats
            if strategy == ExecutionStrategy.INTERPOLATE:
                self.stats["interpolations"] += 1
            elif strategy == ExecutionStrategy.FULL_SIMULATION:
                self.stats["full_sims"] += 1
            
            return ExecutionPlan(
                strategy=strategy,
                confidence=float(data.get("confidence", 0.5)),
                estimated_error=float(data.get("estimated_error_pct", 10.0)),
                estimated_time_ms=time_estimates[strategy],
                reasoning=data.get("reasoning", "No reasoning provided"),
                reference_scenarios=list(data.get("interpolation_weights", {}).keys())
            )
            
        except Exception as e:
            return ExecutionPlan(
                strategy=ExecutionStrategy.FULL_SIMULATION,
                confidence=0.5,
                estimated_error=0.0,
                estimated_time_ms=1700,
                reasoning=f"Failed to parse LLM response: {str(e)}"
            )


# ============================================================================
# LLM Surrogate Model
# ============================================================================

class LLMSurrogateModel:
    """
    Use LLM as a surrogate model for quick energy estimation.
    
    Novel idea: LLM has learned physics relationships from training data.
    Can it estimate building energy without running full simulation?
    """
    
    SURROGATE_PROMPT = """You are a building energy expert. Estimate the annual heating energy.

Building parameters:
- Location: Copenhagen, Denmark
- Type: Residential
- Historical annual heat load: {baseline_kwh} kWh

Retrofit scenario:
{scenario}

Based on building physics:
- Heat loss ∝ U-value × Area × ΔT
- Reducing wall U-value from 0.6 to 0.18 W/m²K reduces wall heat loss by ~70%
- Windows typically account for 25-35% of heat loss
- Infiltration accounts for 15-25% of heat loss

Estimate the annual heat load AFTER retrofit.
Respond with JSON: {{"estimated_kwh": number, "confidence": 0-1, "reasoning": "brief"}}"""

    def __init__(self, api_key: str = None):
        self.llm = DeepSeekClient(api_key)
    
    def estimate(self, baseline_kwh: float, scenario: Dict) -> Tuple[float, float, str]:
        """
        Estimate energy consumption using LLM.
        
        Returns: (estimated_kwh, confidence, reasoning)
        """
        prompt = self.SURROGATE_PROMPT.format(
            baseline_kwh=baseline_kwh,
            scenario=json.dumps(scenario, indent=2)
        )
        
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            
            # Parse response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return (
                    float(data.get("estimated_kwh", baseline_kwh * 0.7)),
                    float(data.get("confidence", 0.5)),
                    data.get("reasoning", "")
                )
        except Exception as e:
            # Fallback: simple linear estimate
            return baseline_kwh * 0.7, 0.3, f"Fallback estimate (LLM error: {e})"
        
        return baseline_kwh * 0.7, 0.3, "Could not parse LLM response"


# ============================================================================
# Integrated LLM-Enhanced Engine
# ============================================================================

class LLMEnhancedEngine:
    """
    TwinQL engine with LLM-guided optimization.
    
    Key insight from experiments: LLM API latency is too high for surrogate use.
    Instead, use LLM ONLY for execution strategy decisions, then use fast
    local methods (cache, interpolation, simulation).
    """
    
    def __init__(self, api_key: str = None):
        self.optimizer = LLMOptimizer(api_key)
        self.surrogate = LLMSurrogateModel(api_key)
        
        # Import standard engine for full simulation
        from core.compiler import TwinQLEngine
        self.standard_engine = TwinQLEngine()
    
    def compare_scenario_smart(self, tid: str, scenario_cfg: Dict,
                                model_id: str, window: Tuple[str, str],
                                error_tolerance: float = 0.05) -> Dict:
        """
        Smart scenario comparison with LLM-guided execution.
        
        Strategy:
        - LLM decides: cache_hit, interpolate, or full_sim
        - LLM_SURROGATE disabled (too slow)
        - Interpolation uses local linear method (fast)
        """
        import time
        
        # Step 1: LLM decides strategy
        plan = self.optimizer.plan_execution(tid, scenario_cfg, error_tolerance)
        
        start = time.perf_counter()
        
        # Step 2: Execute based on strategy (NO LLM_SURROGATE - too slow)
        if plan.strategy == ExecutionStrategy.CACHE_HIT and plan.reference_scenarios:
            result, _ = self.standard_engine.compare_scenario(
                tid, plan.reference_scenarios[0], model_id, window
            )
            
        elif plan.strategy == ExecutionStrategy.INTERPOLATE and plan.reference_scenarios:
            # Fast local interpolation (no LLM call)
            result = self._fast_interpolate(tid, scenario_cfg, plan, model_id, window)
            
        else:
            # Full simulation (default for any other case)
            result = self._run_full_sim(tid, scenario_cfg, model_id, window)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return {
            'result': result,
            'execution': {
                'strategy': plan.strategy.value,
                'confidence': plan.confidence,
                'estimated_error_pct': plan.estimated_error,
                'actual_time_ms': elapsed,
                'reasoning': plan.reasoning
            },
            'optimizer_stats': self.optimizer.stats
        }
    
    def _run_full_sim(self, tid: str, scenario_cfg: Dict, 
                      model_id: str, window: Tuple[str, str]):
        """Run full simulation"""
        from dsl.twinql import define_scenario
        scenario_id = f"llm_opt_{hashlib.md5(json.dumps(scenario_cfg, sort_keys=True).encode()).hexdigest()[:8]}"
        define_scenario(scenario_id, **scenario_cfg)
        result, _ = self.standard_engine.compare_scenario(tid, scenario_id, model_id, window)
        return result
    
    def _fast_interpolate(self, tid: str, target_cfg: Dict,
                          plan: ExecutionPlan, model_id: str,
                          window: Tuple[str, str]) -> pd.DataFrame:
        """
        Fast local interpolation without LLM.
        Uses linear interpolation based on parameter distance.
        """
        if not plan.reference_scenarios:
            return self._run_full_sim(tid, target_cfg, model_id, window)
        
        # Get reference result
        ref_id = plan.reference_scenarios[0]
        ref_result, _ = self.standard_engine.compare_scenario(tid, ref_id, model_id, window)
        
        if ref_result.empty:
            return self._run_full_sim(tid, target_cfg, model_id, window)
        
        # Simple linear adjustment based on wall_u difference
        # (In production, would use more sophisticated interpolation)
        target_wall_u = target_cfg.get('retrofit_package', {}).get('wall_u_after', 0.18)
        
        # Estimate adjustment factor (simplified physics)
        # Lower U-value = less heat loss
        baseline_u = 0.60  # Typical baseline
        adjustment = target_wall_u / 0.18  # Relative to reference
        
        result = ref_result.copy()
        if 'value_retrofit' in result.columns:
            result['value_retrofit'] = result['value_retrofit'] * adjustment
            result['saving'] = result['value_baseline'] - result['value_retrofit']
            if 'saving_pct' in result.columns:
                result['saving_pct'] = (result['saving'] / result['value_baseline'] * 100).round(2)
        
        return result
    
    def _get_baseline_energy(self, tid: str, window: Tuple[str, str]) -> float:
        """Get historical baseline energy"""
        result = execute_query("""
            SELECT SUM(value) as total
            FROM timeseries
            WHERE tid = %s AND cid = 'REALITY' AND metric = 'heat_load'
            AND ts >= %s AND ts < %s
        """, (tid, window[0], window[1]))
        return float(result['total'].iloc[0]) if not result.empty else 0.0
