"""
TwinQL Logical Algebra - Formalization with denotational semantics
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from typing import Union as UnionType
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================================
# Formal Semantics (Denotational)
# ============================================================================
"""
TwinQL Algebra Semantics:

Let T = (tid, ts, metric, value) be the time-series tuple schema.
Let R denote a relation (set of tuples).

Operators:
  Hist(τ, [t₁,t₂], M) → R
    Semantics: {t ∈ TimeSeries | t.tid = τ ∧ t.cid = 'REALITY' ∧ t.ts ∈ [t₁,t₂] ∧ t.metric ∈ M}

  Sim(τ, c, m, [t₁,t₂], M) → R  
    Semantics: Invoke model m with scenario c parameters on twin τ, producing synthetic tuples.
    Formally: {t | t = m(τ, c, ts) for ts ∈ [t₁,t₂], t.metric ∈ M}

  ScanCache(σ) → R
    Semantics: {t ∈ SimCache | hash(t.tid, t.cid, t.mid, t.window, t.metrics) = σ}

  Agg_{G}^{F}(R) → R'
    Semantics: {(g, F(v)) | g ∈ π_G(R), v = {t.value | t ∈ R ∧ t.G = g}}

  Join_{k}(R, S) → R ⋈_k S
    Semantics: {(r, s) | r ∈ R ∧ s ∈ S ∧ r.k = s.k}

Equivalence Rules:
  E1: Sim(τ,c,m,w,M) ≡ ScanCache(σ)  if σ = hash(τ,c,m,w,M) ∈ Cache
  E2: Sim(τ₁,c,m,w) ∪ Sim(τ₂,c,m,w) ≡ SimBatch({τ₁,τ₂},c,m,w)
  E3: σ_p(Hist(τ,w,M)) ≡ Hist(τ, w∩p, M)  if p is temporal predicate
  E4: γ_G^F(Hist(τ,w,M)) can be pushed to SQL
"""

class OpType(Enum):
    HIST = "Hist"
    SIM = "Sim"
    AGG = "Agg"
    JOIN = "Join"
    PROJECT = "Project"
    SELECT = "Select"
    SCAN_CACHE = "ScanCache"
    SIM_BATCH = "SimBatch"
    UNION = "Union"

@dataclass
class Schema:
    """Relation schema: ordered list of (name, type) pairs"""
    columns: List[Tuple[str, str]]
    
    def __contains__(self, col: str) -> bool:
        return any(c[0] == col for c in self.columns)
    
    def project(self, cols: List[str]) -> 'Schema':
        return Schema([(n, t) for n, t in self.columns if n in cols])

# Standard schemas
TS_SCHEMA = Schema([("tid", "str"), ("ts", "timestamp"), ("metric", "str"), ("value", "float")])
AGG_SCHEMA = Schema([("tid", "str"), ("period", "timestamp"), ("metric", "str"), ("value", "float")])
JOINED_SCHEMA = Schema([("tid", "str"), ("period", "timestamp"), ("metric", "str"), 
                        ("value_baseline", "float"), ("value_sim", "float")])


# ============================================================================
# Base Operator Interface
# ============================================================================

class LogicalOperator(ABC):
    """Abstract base for all logical operators"""
    @abstractmethod
    def children(self) -> List['LogicalOperator']: pass
    
    @abstractmethod
    def signature(self) -> str: pass
    
    @abstractmethod
    def output_schema(self) -> Schema: pass
    
    def cardinality_estimate(self) -> int:
        """Estimate output cardinality (default: 8760 = 1 year hourly)"""
        return 8760


# ============================================================================
# Leaf Operators (Data Sources)
# ============================================================================

@dataclass
class Hist(LogicalOperator):
    """
    Historical data scan: Hist(τ, [t₁,t₂], M)
    Retrieves observed time series for twin τ over window [t₁,t₂] for metrics M
    """
    tid: str
    window: Tuple[str, str]
    metrics: List[str]
    op_type: OpType = field(default=OpType.HIST, repr=False)
    _cardinality: int = field(default=8760, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return []
    
    def signature(self) -> str:
        return f"Hist({self.tid},{self.window[0]},{self.window[1]},{','.join(sorted(self.metrics))})"
    
    def output_schema(self) -> Schema:
        return TS_SCHEMA
    
    def cardinality_estimate(self) -> int:
        return self._cardinality * len(self.metrics)


@dataclass
class Sim(LogicalOperator):
    """
    Simulation invocation: Sim(τ, c, m, [t₁,t₂], M)
    Invokes model m under scenario c for twin τ
    """
    tid: str
    scenario_id: str
    model_id: str
    window: Tuple[str, str]
    metrics: List[str]
    op_type: OpType = field(default=OpType.SIM, repr=False)
    _cardinality: int = field(default=8760, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return []
    
    def signature(self) -> str:
        return f"Sim({self.tid},{self.scenario_id},{self.model_id},{self.window[0]},{self.window[1]})"
    
    def output_schema(self) -> Schema:
        return TS_SCHEMA
    
    def cardinality_estimate(self) -> int:
        return self._cardinality * len(self.metrics)


@dataclass
class SimBatch(LogicalOperator):
    """
    Batched simulation: SimBatch({τ₁,...,τₙ}, c, m, [t₁,t₂], M)
    Amortizes simulation overhead across multiple twins
    """
    tids: List[str]
    scenario_id: str
    model_id: str
    window: Tuple[str, str]
    metrics: List[str]
    op_type: OpType = field(default=OpType.SIM_BATCH, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return []
    
    def signature(self) -> str:
        return f"SimBatch({len(self.tids)},{self.scenario_id},{self.model_id})"
    
    def output_schema(self) -> Schema:
        return TS_SCHEMA
    
    def cardinality_estimate(self) -> int:
        return 8760 * len(self.tids) * len(self.metrics)


@dataclass
class ScanCache(LogicalOperator):
    """
    Cache scan: ScanCache(σ)
    Reads pre-computed simulation results by signature
    """
    cache_sig: str
    tid: str
    scenario_id: str
    window: Tuple[str, str]
    metrics: List[str]
    op_type: OpType = field(default=OpType.SCAN_CACHE, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return []
    
    def signature(self) -> str:
        return f"ScanCache({self.cache_sig[:16]})"
    
    def output_schema(self) -> Schema:
        return TS_SCHEMA
    
    def cardinality_estimate(self) -> int:
        return 8760 * len(self.metrics)


# ============================================================================
# Relational Operators
# ============================================================================

@dataclass
class Agg(LogicalOperator):
    """
    Aggregation: γ_{G}^{F}(R)
    Groups by G and applies aggregate functions F
    """
    child: LogicalOperator
    group_by: str  # 'month', 'day', 'year', 'hour'
    agg_funcs: Dict[str, str]  # column -> func (sum, avg, min, max)
    op_type: OpType = field(default=OpType.AGG, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return [self.child]
    
    def signature(self) -> str:
        return f"Agg({self.group_by},{list(self.agg_funcs.keys())})"
    
    def output_schema(self) -> Schema:
        return AGG_SCHEMA
    
    def cardinality_estimate(self) -> int:
        # Reduce cardinality based on grouping
        child_card = self.child.cardinality_estimate()
        reduction = {'hour': 1, 'day': 24, 'month': 730, 'year': 8760}
        return max(1, child_card // reduction.get(self.group_by, 1))


@dataclass
class Join(LogicalOperator):
    """
    Join: R ⋈_{k} S
    Equi-join on key k (typically 'ts' or 'period')
    """
    left: LogicalOperator
    right: LogicalOperator
    join_key: str
    op_type: OpType = field(default=OpType.JOIN, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return [self.left, self.right]
    
    def signature(self) -> str:
        return f"Join({self.join_key})"
    
    def output_schema(self) -> Schema:
        return JOINED_SCHEMA
    
    def cardinality_estimate(self) -> int:
        # Assume 1:1 join on temporal key
        return min(self.left.cardinality_estimate(), self.right.cardinality_estimate())


@dataclass
class Project(LogicalOperator):
    """
    Projection: π_{cols}(R)
    Optionally with computed columns
    """
    child: LogicalOperator
    columns: List[str]
    computed: Dict[str, str] = field(default_factory=dict)  # name -> expression
    op_type: OpType = field(default=OpType.PROJECT, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return [self.child]
    
    def signature(self) -> str:
        return f"Project({self.columns})"
    
    def output_schema(self) -> Schema:
        base = self.child.output_schema().project(self.columns)
        for name in self.computed:
            base.columns.append((name, "float"))
        return base
    
    def cardinality_estimate(self) -> int:
        return self.child.cardinality_estimate()


@dataclass
class Select(LogicalOperator):
    """
    Selection: σ_{p}(R)
    Filters tuples by predicate p
    """
    child: LogicalOperator
    predicate: str
    selectivity: float = 0.5  # Estimated selectivity
    op_type: OpType = field(default=OpType.SELECT, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return [self.child]
    
    def signature(self) -> str:
        return f"Select({self.predicate})"
    
    def output_schema(self) -> Schema:
        return self.child.output_schema()
    
    def cardinality_estimate(self) -> int:
        return int(self.child.cardinality_estimate() * self.selectivity)


@dataclass
class Union(LogicalOperator):
    """
    Union: R ∪ S
    Combines results from multiple operators
    """
    children_ops: List[LogicalOperator]
    op_type: OpType = field(default=OpType.UNION, repr=False)
    
    def children(self) -> List[LogicalOperator]:
        return self.children_ops
    
    def signature(self) -> str:
        return f"Union({len(self.children_ops)})"
    
    def output_schema(self) -> Schema:
        return self.children_ops[0].output_schema() if self.children_ops else TS_SCHEMA
    
    def cardinality_estimate(self) -> int:
        return sum(c.cardinality_estimate() for c in self.children_ops)


# ============================================================================
# Type Alias & Utilities
# ============================================================================

LogicalOp = UnionType[Hist, Sim, SimBatch, ScanCache, Agg, Join, Project, Select, Union]


def print_plan(op: LogicalOp, indent: int = 0) -> str:
    """Pretty print a logical plan tree with cardinality estimates"""
    prefix = "  " * indent
    card = op.cardinality_estimate()
    
    if isinstance(op, Hist):
        line = f"{prefix}Hist(tid={op.tid}, window={op.window}, metrics={op.metrics}) [card={card}]"
    elif isinstance(op, Sim):
        line = f"{prefix}Sim(tid={op.tid}, scenario={op.scenario_id}, model={op.model_id}) [card={card}]"
    elif isinstance(op, SimBatch):
        line = f"{prefix}SimBatch(n={len(op.tids)}, scenario={op.scenario_id}) [card={card}]"
    elif isinstance(op, ScanCache):
        line = f"{prefix}ScanCache(sig={op.cache_sig[:12]}...) [card={card}]"
    elif isinstance(op, Agg):
        line = f"{prefix}Agg(by={op.group_by}, funcs={op.agg_funcs}) [card={card}]"
    elif isinstance(op, Join):
        line = f"{prefix}Join(on={op.join_key}) [card={card}]"
    elif isinstance(op, Project):
        line = f"{prefix}Project(cols={op.columns}) [card={card}]"
    elif isinstance(op, Select):
        line = f"{prefix}Select(pred={op.predicate}, sel={op.selectivity}) [card={card}]"
    elif isinstance(op, Union):
        line = f"{prefix}Union(n={len(op.children_ops)}) [card={card}]"
    else:
        line = f"{prefix}{type(op).__name__} [card={card}]"
    
    lines = [line]
    for child in op.children():
        lines.append(print_plan(child, indent + 1))
    
    return "\n".join(lines)


def collect_operators(op: LogicalOp, op_type: type) -> List[LogicalOp]:
    """Collect all operators of a given type in a plan tree"""
    result = []
    if isinstance(op, op_type):
        result.append(op)
    for child in op.children():
        result.extend(collect_operators(child, op_type))
    return result
