# TwinDB System Architecture

## Overview

TwinDB is a **scenario-aware data management system** for digital twin analytics. It extends traditional time-series databases with first-class support for **what-if scenarios** and **simulation-based queries**.

## Key Contributions

1. **Twin Data Model (TDM)**: A unified data model that treats scenarios as first-class citizens alongside historical observations.

2. **TwinQL**: A declarative query language for expressing scenario-based analytics that seamlessly combines historical data retrieval with simulation invocation.

3. **Scenario-Aware Optimization**: Novel optimization techniques including simulation caching, batch execution, and hybrid cost modeling.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TwinQL Query                                │
│   TWIN SELECT h.value AS baseline, s.value AS retrofit              │
│   FROM HISTORICAL ... AS h, SIMULATE ... AS s                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Parser                                      │
│   • Lexer: Tokenizes TwinQL text                                    │
│   • Parser: Builds AST from tokens                                  │
│   • Grammar: BNF specification for TwinQL                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Plan Builder                                    │
│   • AST → Logical Plan translation                                  │
│   • Operator construction: Hist, Sim, Agg, Join                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Optimizer                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Rewrite Rules:                                               │   │
│   │   R1: Sim → ScanCache (if cached)                           │   │
│   │   R2: Union(Sim,...) → SimBatch (batching)                  │   │
│   │   R3: σ(Hist) → Hist (predicate push-down)                  │   │
│   │   R4: γ(Hist) → SQL (aggregation push-down)                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Cost Model:                                                  │   │
│   │   C_total = C_db + C_sim - C_cache                          │   │
│   │   Cardinality estimation for plan comparison                │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Executor                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│   │   DB Engine  │  │  Sim Manager │  │     Cache Manager        │  │
│   │  (TimescaleDB)│  │  (Python)    │  │  (signature-based)       │  │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ TimescaleDB (PostgreSQL + Hypertables)                      │   │
│   │   • timeseries: (tid, cid, metric, ts, value, source)       │   │
│   │   • sim_cache: (sig, tid, cid, mid, window, metrics)        │   │
│   │   • scenario: (cid, name, cfg)                              │   │
│   │   • twin, asset, model: metadata tables                     │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Twin Data Model (TDM)

### Core Entities

| Entity | Description | Key Attributes |
|--------|-------------|----------------|
| **Asset** | Physical entity (building, equipment) | aid, name, location, properties |
| **Twin** | Digital representation of asset | tid, aid, created_at |
| **Scenario** | What-if configuration | cid, name, cfg (JSON) |
| **Model** | Simulation model | mid, name, version, parameters |
| **TimeSeries** | Observed or simulated data | tid, cid, metric, ts, value |
| **SimCache** | Cached simulation results | sig, tid, cid, mid, window |

### Key Design Decisions

1. **Scenario as First-Class Citizen**: Unlike traditional TSDB where all data is "reality", TwinDB distinguishes `cid='REALITY'` (observed) from scenario IDs (simulated).

2. **Source Provenance**: The `source` column tracks data origin: `'sensor'` for observations, `'sim:ModelName@version'` for simulations.

3. **Signature-Based Caching**: Cache keys are deterministic hashes of `(tid, cid, mid, window, metrics)`, enabling exact-match lookups.

## TwinQL Language

### Grammar (BNF)

```bnf
<twinql_stmt>    ::= <define_scenario> | <twin_select>

<define_scenario> ::= "DEFINE" "SCENARIO" <id> "AS" <config>

<twin_select>    ::= "TWIN" "SELECT" <select_list>
                     "FROM" <source> ("," <source>)*
                     ["WHERE" <predicate>]
                     ["GROUP" "BY" <time_unit>]

<source>         ::= <historical> | <simulate>

<historical>     ::= "HISTORICAL" "twin_id" "=" <string>
                     "WINDOW" <timestamp> "TO" <timestamp>
                     "METRIC" <string>
                     ["AGGREGATE" "BY" <time_unit>] "AS" <alias>

<simulate>       ::= "SIMULATE" "twin_id" "=" <string>
                     "SCENARIO" <id>
                     "MODEL" <string>
                     "WINDOW" <timestamp> "TO" <timestamp>
                     "METRIC" <string>
                     ["AGGREGATE" "BY" <time_unit>] "AS" <alias>
```

### Example Query (W1 Workload)

```sql
DEFINE SCENARIO deep_retrofit AS
  retrofit_package = {"wall_u_after": 0.18, "window_u_after": 0.90}

TWIN SELECT 
  h.value AS baseline,
  s.value AS retrofit,
  (h.value - s.value) AS saving
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS h,
  SIMULATE twin_id='Twin_B123' 
    SCENARIO deep_retrofit 
    MODEL 'BuildingHeat_v3'
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS s
```

## Logical Algebra

### Operators

| Operator | Notation | Semantics |
|----------|----------|-----------|
| **Hist** | Hist(τ, w, M) | Retrieve historical data for twin τ, window w, metrics M |
| **Sim** | Sim(τ, c, m, w, M) | Invoke model m under scenario c for twin τ |
| **ScanCache** | ScanCache(σ) | Read cached simulation result by signature σ |
| **SimBatch** | SimBatch(T, c, m, w, M) | Batch simulation for twin set T |
| **Agg** | γ_G^F(R) | Aggregate R by G using functions F |
| **Join** | R ⋈_k S | Equi-join on key k |

### Equivalence Rules

```
R1: Sim(τ,c,m,w,M) ≡ ScanCache(σ)           if σ = hash(τ,c,m,w,M) ∈ Cache
R2: ⋃ᵢ Sim(τᵢ,c,m,w,M) ≡ SimBatch({τᵢ},c,m,w,M)
R3: σ_p(Hist(τ,w,M)) ≡ Hist(τ, w∩p, M)      if p is temporal
R4: γ_G^F(Hist(τ,w,M)) → push to SQL
```

## Cost Model

```
C_total = C_db + C_sim - C_cache

Where:
  C_db = scan_cost × cardinality + agg_cost × groups + join_cost × pairs
  C_sim = setup_cost + per_step_cost × timesteps
  C_cache = lookup_cost + scan_cost × cardinality (if cached)
```

### Calibrated Parameters (TimescaleDB + Python)

| Parameter | Value | Description |
|-----------|-------|-------------|
| scan_cost_per_k | 2.5 ms | DB scan per 1000 rows |
| sim_setup_cost | 100 ms | Model initialization |
| sim_per_timestep | 0.19 ms | Per hourly simulation step |
| cache_lookup_cost | 2.0 ms | Cache signature lookup |

## Experimental Results

### E1: TwinDB vs Baseline
- **Cold cache**: Similar to baseline (~1.8s for 1-year simulation)
- **Hot cache**: 30-35x speedup over baseline

### E2: Cache Optimization
- Cache provides 33x speedup for repeated queries

### E3: Scalability
- Linear scaling with number of twins
- ~53 ms/twin with hot cache

### E4: Cache Sensitivity
- 100% cache hit: 54 ms total
- 0% cache hit: 6.4s total
- Linear interpolation between extremes

## Limitations and Future Work

1. **Single-node**: Current implementation is single-node; distributed execution is future work.

2. **Model Integration**: Models are Python functions; future work includes model versioning and containerization.

3. **Incremental Maintenance**: Cache invalidation on scenario changes is manual; automatic invalidation is planned.

4. **Query Optimization**: Current optimizer uses heuristics; cost-based optimization with statistics is future work.
