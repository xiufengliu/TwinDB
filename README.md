# TwinDB: Scenario-Aware Data Management for Digital Twin Analytics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 15+](https://img.shields.io/badge/postgresql-15+-blue.svg)](https://www.postgresql.org/)
[![TimescaleDB](https://img.shields.io/badge/timescaledb-latest-orange.svg)](https://www.timescale.com/)

## Overview

TwinDB is a **scenario-aware data management system** that extends time-series databases with first-class support for **what-if analysis** on digital twins. It provides:

- **Twin Data Model (TDM)**: Unified representation of assets, twins, scenarios, and time series
- **TwinQL**: Declarative query language for scenario-based analytics
- **Incremental Scenario Maintenance (ISM)**: Efficient updates for parameter changes
- **Hybrid Execution**: Seamless integration of database queries and simulation invocations

## Key Features

| Feature | Description |
|---------|-------------|
| ISM Speedup | Up to **50x** faster scenario exploration |
| Batch Processing | Efficient multi-building analysis |
| Online Learning | Automatic sensitivity calibration |
| Error Bounds | Formal guarantees on approximation quality |

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for TimescaleDB)

### Installation

```bash
# Clone repository
git clone https://github.com/TwinDB-System/twindb.git
cd twindb

# Install dependencies
pip install -r requirements.txt

# Start TimescaleDB
docker-compose up -d

# Load sample data
python data/loader.py
```

### Example: Building Retrofit Analysis

```python
from core.compiler import TwinQLEngine
from dsl.twinql import define_scenario

# Define a retrofit scenario
define_scenario('deep_retrofit', retrofit_package={
    'wall_u_before': 0.60, 'wall_u_after': 0.18,
    'window_u_before': 2.80, 'window_u_after': 0.90,
})

# Run scenario comparison
engine = TwinQLEngine()
result, explain = engine.compare_scenario(
    tid='Twin_B123',
    scenario_id='deep_retrofit',
    model_id='BuildingHeat_v3',
    window=('2018-01-01', '2019-01-01'),
    metric='heat_load',
    agg_by='month'
)

print(result[['period', 'value_baseline', 'value_retrofit', 'saving_pct']])
```

Output:
```
       period  value_baseline  value_retrofit  saving_pct
0  2018-01-01        30422.99        16734.61       45.0%
1  2018-02-01        27365.27        15088.14       44.9%
...
Total annual saving: 125,318 kWh (45.0%)
```

### Example: Multi-Building Analysis

```python
# Find worst-performing buildings in a district
worst = engine.find_worst_performers(
    tids=['Twin_B101', 'Twin_B102', ..., 'Twin_B110'],
    scenario_id='deep_retrofit',
    model_id='BuildingHeat_v3',
    window=('2018-01-01', '2019-01-01'),
    percentile=0.2  # Top 20% energy consumers
)
print(worst)
```

## TwinQL Language

TwinQL is a declarative query language for scenario-based analytics:

```sql
-- Define a scenario
DEFINE SCENARIO deep_retrofit AS
  retrofit_package = {"wall_u_after": 0.18, "window_u_after": 0.90}

-- Compare historical vs simulated
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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TwinQL Query                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Parser → Plan Builder → Optimizer → Executor               │
│                                                             │
│  Optimization Strategies:                                   │
│    - Cache reuse for exact scenario matches                │
│    - ISM updates for similar scenarios                     │
│    - Batch processing for multi-building queries           │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │TimescaleDB│  │Simulation│  │  Cache   │
        │  (TSDB)   │  │ Manager  │  │ Manager  │
        └──────────┘  └──────────┘  └──────────┘
```

## Project Structure

```
twindb/
├── core/
│   ├── __init__.py       # Public API exports
│   ├── operators.py      # Logical operators (Hist, Sim, Agg, Join)
│   ├── ism.py            # Incremental Scenario Maintenance
│   ├── ism_advanced.py   # Online learning (RLS)
│   ├── batch.py          # Optimized batch processing
│   ├── optimizer.py      # Cost model and rewrite rules
│   ├── compiler.py       # TwinQL compilation pipeline
│   ├── parser.py         # TwinQL lexer and parser
│   ├── rules.py          # Optimization rewrite rules
│   ├── simulation.py     # Simulation interface
│   ├── llm.py            # LLM integration (optional)
│   ├── db.py             # Database utilities
│   └── schema.sql        # Database schema
├── sim/
│   ├── manager.py        # Simulation orchestration
│   └── models/           # Physics-based models
├── dsl/
│   └── twinql.py         # Python DSL for TwinQL
├── eval/
│   ├── run_experiments.py    # Main evaluation suite
│   ├── benchmarks.py         # Micro-benchmarks
│   ├── baselines.py          # Comparison baselines
│   ├── reproduce_results.py  # Paper result reproduction
│   └── results/              # Output directory
├── docs/
│   └── technical_overview.md
└── examples/
    ├── single_building.py    # Single building analysis
    ├── multi_building.py     # Multi-building portfolio
    └── llm_assistant.py      # LLM integration demo
```

## Evaluation

Run the full evaluation suite:

```bash
python eval/run_experiments.py
```

Or quick evaluation:

```bash
python eval/run_experiments.py --quick
```

### Experiments

- **E1**: Single-building scenario exploration
- **E2**: Multi-building portfolio analysis
- **E3**: Scalability with building count
- **E4**: ISM accuracy vs parameter change magnitude
- **E5**: Online learning convergence

## Citation

If you use TwinDB in your research, please cite:

```bibtex
@inproceedings{twindb2025,
  title={TwinDB: Incremental Scenario Maintenance for Digital Twin Queries},
  author={Anonymous},
  booktitle={Proceedings of the International Conference on Management of Data},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
