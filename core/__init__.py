"""
TwinDB Core - Query processing and optimization for digital twin analytics

This module provides the core functionality for TwinDB:
- Query parsing and compilation (TwinQL)
- Incremental Scenario Maintenance (ISM)
- Query optimization with caching and batch processing
- Database utilities
"""

from .db import get_connection, execute_query, execute_many, execute_sql
from .operators import (
    LogicalOp, Hist, Sim, SimBatch, ScanCache, Agg, Join, Project, Select, Union,
    print_plan, collect_operators, TS_SCHEMA, AGG_SCHEMA
)
from .optimizer import Optimizer, CostModel, CacheManager, OptimizationStats
from .compiler import TwinQLCompiler, TwinQLEngine, PlanBuilder
from .parser import parse_twinql, Lexer, Parser
from .ism import (
    ScenarioDelta, DeltaType, SensitivityModel, IncrementalUpdateEngine,
    ISMOptimizer, IncrementalResult
)
from .batch import OptimizedBatchISM, BuildingProfile, BatchISMResult

__all__ = [
    # Database
    'get_connection', 'execute_query', 'execute_many', 'execute_sql',
    
    # Logical operators
    'LogicalOp', 'Hist', 'Sim', 'SimBatch', 'ScanCache', 
    'Agg', 'Join', 'Project', 'Select', 'Union',
    'print_plan', 'collect_operators',
    
    # Query optimization
    'Optimizer', 'CostModel', 'CacheManager', 'OptimizationStats',
    
    # Query compilation
    'TwinQLCompiler', 'TwinQLEngine', 'PlanBuilder',
    'parse_twinql', 'Lexer', 'Parser',
    
    # ISM
    'ScenarioDelta', 'DeltaType', 'SensitivityModel',
    'IncrementalUpdateEngine', 'ISMOptimizer', 'IncrementalResult',
    
    # Batch processing
    'OptimizedBatchISM', 'BuildingProfile', 'BatchISMResult',
]
