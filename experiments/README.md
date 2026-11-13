# TwinDB experiments

Code, workloads, recorded ground truth and results for the paper
*Maintaining Simulation Results as Views over Continuous Parameter Spaces for Interactive Digital Twins*.

TwinDB treats every simulated scenario as a point of a materialized view over the continuous
parameter space of a digital twin. A new scenario (or one term of a portfolio query) is answered by
lookup, by maintaining the view from nearby materialized scenarios (first-order ISM or a kernel/GP
operator), or by simulation. Maintained answers are accepted under an audited threshold: a random
fraction of answers is re-simulated in the background, and the observed error ratio corrects the
acceptance rule.

## Layout

| Path | Contents |
|---|---|
| `twins/twindb2.py` | Views, operators (ISM, kernel, GP), audited acceptor, query plans, portfolio queries, model updates |
| `twins/engines.py` | Engine adapters: NREL OCHRE 0.9 (building), SUMO (traffic), EPANET via WNTR (water) |
| `twins/gt.py` | Plans and runs the ground-truth simulations for a workload |
| `twins/experiments.py` | Experiments E1–E10, replayed against recorded ground truth |
| `twins/live.py` | E7: live runs with the simulator in the loop |
| `twins/workload.json` | Analyst-session workloads for the three single twins |
| `twins/gt_replay/` | Recorded ground truth: per-engine index and traces |
| `twins/results/` | Result files of E1–E10 |
| `aalborg/` | Aalborg district study: engine, calibration, workload, ground-truth index (2,066 runs), advisor sessions (`aal_exp3.py`), portfolio planners (`aal_planner.py`), results |
| `inputs/aalborg/` | Weather inputs, selected meter IDs, SHA-256 manifest of the meter files |
| `tools/` | Meter-data download (`aalborg_acquire.py`, `remotezip.py`), tkinter stub for headless OCHRE |

## Setup

```bash
pip install numpy scipy pandas wntr eclipse-sumo        # Python >= 3.11
# OCHRE needs its own interpreter (Python 3.11, numpy < 2):
python3.11 -m venv venv311 && venv311/bin/pip install ochre-nrel==0.9.0
export OCHRE_PY=$PWD/venv311/bin/python
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Replaying the experiments needs only numpy and scipy, because the ground truth is recorded. OCHRE,
SUMO and WNTR are needed only to re-simulate ground truth or for the live runs (E7).

## Reproducing the results

```bash
cd twins
python experiments.py e1 epanet,sumo,ochre     # end-to-end comparison
python experiments.py e2 epanet,sumo,ochre     # violation rate and certificate
python experiments.py e3 epanet,sumo,ochre     # tolerance and audit-rate sweep
python experiments.py e4 epanet,sumo,ochre     # model updates (invalidate vs delta)
python experiments.py e6                        # micro-benchmarks
python experiments.py e9 epanet,sumo,ochre     # audited GP baseline
python experiments.py e10 epanet,sumo,ochre    # oracle prefetch, session bootstrap, injected bias
cd ../aalborg
python aal_exp3.py                              # advisors: portfolio view vs per-house views
python aal_planner.py                           # portfolio planners
```

Speedups are computed from the recorded simulation times, so a replay reproduces the reported
values up to wall-clock bookkeeping overhead.

To re-simulate ground truth instead of replaying it: `python gt.py plan; python gt.py run ENGINE`
in `twins/`, and `python aalborg_gt.py run workload_aal_v3.json` in `aalborg/`.

## Aalborg data

Smart-heat-meter data of Schaffer et al., *Scientific Data* 9:420 (2022), Zenodo
[10.5281/zenodo.6563114](https://doi.org/10.5281/zenodo.6563114), CC-BY-4.0. The meter files are not
redistributed. Download the 67 houses used and verify them with

```bash
python tools/aalborg_acquire.py inputs/aalborg
(cd inputs/aalborg && sha256sum -c MANIFEST_sha256.txt)
```

`aalborg/calibrate.py` fits two descriptors per house (envelope multiplier `k_env`, hot-water
multiplier `k_dhw`); `aalborg/calibration.json` holds the result. Weather: ERA5 2019 (Open-Meteo,
CC-BY-4.0) written into the Aalborg TMYx EPW (climate.onebuilding.org) by `aalborg/make_epw.py`.
