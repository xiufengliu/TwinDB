"""Black-box simulation engines used in the evaluation.

Each engine maps a normalized parameter vector x in [0,1]^d to an output trace
y in R^T by invoking an unmodified, externally developed simulator:

  ochre   NREL OCHRE 0.9.0 residential building model (RC envelope network,
          air-source heat pump with compressor lockout and backup heat,
          water heater, occupancy schedules); annual run, hourly output.
  sumo    Eclipse SUMO microscopic traffic simulation of a 6x6 signalized grid,
          2-hour peak with 9,600 trips; delay aggregated in 5-minute bins.
  epanet  EPA EPANET 2.2 (via WNTR) extended-period hydraulics of the Net6
          network (3,356 nodes, 61 pumps, rule-based controls); 7 days, 15 min.

The engines are deterministic given x (fixed seeds, fixed inputs). The
`fidelity` argument selects the engine's own time discretization, used by the
granularity experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import shutil
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_TWINDB_ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("TWINDB_STUBS", os.path.join(_TWINDB_ROOT, "tools", "stubs"))
OCHRE_DEFAULTS = os.environ.get("OCHRE_DEFAULTS", "")   # path to ochre/defaults of the OCHRE 0.9 install
OCHRE_PY = os.environ.get("OCHRE_PY", sys.executable)   # Python 3.11 interpreter with ochre-nrel 0.9
SUMO_HOME = os.environ.get("SUMO_HOME", "")   # e.g. site-packages/sumo of eclipse-sumo

ENGINES = {
    "ochre": {
        "params": ["wall_R", "ceiling_R", "window_U", "ach50", "heat_setpoint_F"],
        "bounds": [(8.0, 30.0), (20.0, 70.0), (0.25, 1.00), (3.0, 15.0), (66.0, 74.0)],
        "agg": "annual electricity (kWh)",
        "default_fidelity": 60,     # minutes per step
    },
    "sumo": {
        "params": ["demand_scale", "speed_factor", "tau_s", "accel", "min_gap_m"],
        "bounds": [(0.70, 1.30), (0.80, 1.20), (0.60, 1.60), (1.50, 3.50), (1.50, 3.50)],
        "agg": "total delay (veh-h)",
        "default_fidelity": 1.0,    # seconds per step
    },
    "epanet": {
        "params": ["demand_mult", "roughness_scale", "pump_speed", "tank_init_frac", "reservoir_head_m"],
        "bounds": [(0.80, 1.20), (0.85, 1.15), (0.90, 1.10), (0.30, 0.80), (-5.0, 5.0)],
        "agg": "pump energy (kWh)",
        "default_fidelity": 15,     # minutes per hydraulic step
    },
}


def denorm(engine: str, x) -> np.ndarray:
    b = np.array(ENGINES[engine]["bounds"], dtype=float)
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return b[:, 0] + x * (b[:, 1] - b[:, 0])


def signature(engine: str, x, fidelity=None, variant: str = "v0") -> str:
    fid = ENGINES[engine]["default_fidelity"] if fidelity is None else fidelity
    key = json.dumps([engine, [round(float(v), 10) for v in x], fid, variant])
    return hashlib.sha256(key.encode()).hexdigest()[:20]


# --------------------------------------------------------------------------- OCHRE
OCHRE_WORKER = r'''
import sys, os, json, re, time, datetime as dt, tempfile
sys.path.insert(0, os.environ.get("TWINDB_STUBS", ""))   # tkinter stub (OCHRE GUI import only)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ochre import Dwelling
job = json.loads(sys.argv[1])
import ochre as _o; D = job["defaults"] or os.path.join(os.path.dirname(_o.__file__), "defaults")
xml = open(os.path.join(D, "Input Files", "bldg0112631-up11.xml")).read()
wall_R, ceil_R, win_U, ach, setp = job["p"]
def sub_section(x, sec, fn):
    i = x.find("<%s>" % sec); j = x.find("</%s>" % sec)
    return x[:i] + fn(x[i:j]) + x[j:]
# conditioned-space walls (16.8) -> wall_R ; attic floor (61.6) -> ceil_R
xml = sub_section(xml, "Walls", lambda s: s.replace("<AssemblyEffectiveRValue>16.8<", "<AssemblyEffectiveRValue>%.4f<" % wall_R))
xml = sub_section(xml, "Floors", lambda s: s.replace("<AssemblyEffectiveRValue>61.6<", "<AssemblyEffectiveRValue>%.4f<" % ceil_R))
xml = sub_section(xml, "Windows", lambda s: re.sub(r"<UFactor>[0-9.]+<", "<UFactor>%.4f<" % win_U, s))
xml = re.sub(r"(<UnitofMeasure>ACH</UnitofMeasure>\s*<AirLeakage>)[0-9.]+(<)", lambda m: m.group(1) + "%.4f" % ach + m.group(2), xml)
def setp_list(m):
    vals = ", ".join(["%.3f" % setp] * 24)
    return m.group(1) + vals + m.group(2)
xml = re.sub(r"(<WeekdaySetpointTempsHeatingSeason>)[^<]*(<)", setp_list, xml)
xml = re.sub(r"(<WeekendSetpointTempsHeatingSeason>)[^<]*(<)", setp_list, xml)
if job.get("variant") in ("hp_upgrade", "retrofit2"):
    xml = xml.replace("<Value>16.0</Value>", "<Value>20.0</Value>")
    xml = xml.replace("<BackupHeatingLockoutTemperature dataSource='software'>40.0<", "<BackupHeatingLockoutTemperature dataSource='software'>30.0<")
if job.get("variant") == "retrofit2":   # heat-pump water heater replaced by an electric resistance tank
    xml = xml.replace("<WaterHeaterType>heat pump water heater</WaterHeaterType>", "<WaterHeaterType>storage water heater</WaterHeaterType>")
    xml = xml.replace("<UniformEnergyFactor>3.45</UniformEnergyFactor>", "<UniformEnergyFactor>0.92</UniformEnergyFactor><FirstHourRating>60.0</FirstHourRating>")
    xml = re.sub(r"\s*<HPWHOperatingMode[^>]*>[^<]*</HPWHOperatingMode>", "", xml)
tmp = tempfile.mkdtemp(prefix="ochre_")
xf = os.path.join(tmp, "house.xml"); open(xf, "w").write(xml)
weather = job.get("weather") or os.path.join(D, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")
args = dict(name="q", start_time=dt.datetime(2018, 1, 1), time_res=dt.timedelta(minutes=job["fid"]),
            duration=dt.timedelta(days=job.get("days", 365)), initialization_time=dt.timedelta(days=1),
            hpxml_file=xf, hpxml_schedule_file=os.path.join(D, "Input Files", "bldg0112631_schedule.csv"),
            weather_file=weather, output_path=tmp, verbosity=1, save_results=False)
t0 = time.time()
d = Dwelling(**args)
df, metrics, hourly = d.simulate()
el = time.time() - t0
kw = df["Total Electric Power (kW)"].to_numpy(dtype=float)
step_h = job["fid"] / 60.0
energy = kw * step_h                           # kWh per step
# hourly trace regardless of fidelity
per_h = int(round(1 / step_h)) if step_h < 1 else 1
trace = energy.reshape(-1, per_h).sum(axis=1) if per_h > 1 else energy
np.save(job["out"], trace)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps({"elapsed": el, "T": len(trace), "agg": float(trace.sum())}))
'''


def _run_ochre(p, fid, out, weather=None, days=365, variant="v0"):
    job = {"p": [float(v) for v in p], "fid": fid, "out": out, "defaults": OCHRE_DEFAULTS,
           "weather": weather, "days": days, "variant": variant}
    r = subprocess.run([OCHRE_PY, "-c", OCHRE_WORKER, json.dumps(job)], capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- SUMO
def _run_sumo(p, fid, out, variant="v0"):
    demand, sf, tau, accel, mingap = p
    assets = os.path.join(HERE, "assets", "sumo")
    trips = os.path.join(assets, "trips2h.xml" if variant == "v0" else "trips2h_%s.xml" % variant)
    tmp = tempfile.mkdtemp(prefix="sumo_")
    vt = os.path.join(tmp, "vtype.add.xml")
    with open(vt, "w") as f:
        f.write('<additional><vType id="twin" accel="%.4f" decel="4.5" sigma="0.5" tau="%.4f" '
                'minGap="%.4f" speedFactor="%.4f" speedDev="0.1"/></additional>' % (accel, tau, mingap, sf))
    ti = os.path.join(tmp, "ti.xml")
    cmd = [os.path.join(SUMO_HOME, "bin", "sumo"), "-n", os.path.join(assets, "grid.net.xml"),
           "-r", trips, "-a", vt, "--seed", "42", "--scale", "%.4f" % demand,
           "--step-length", "%.3f" % fid, "--no-step-log", "--duration-log.disable", "--no-warnings",
           "--tripinfo-output", ti, "--end", "10800"]
    env = dict(os.environ, SUMO_HOME=SUMO_HOME)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    el = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    import xml.etree.ElementTree as ET
    bins = np.zeros(24)                       # 5-minute departure bins over 2 h
    for t in ET.parse(ti).getroot().iter("tripinfo"):
        b = int(float(t.get("depart")) // 300)
        if 0 <= b < 24:
            bins[b] += float(t.get("timeLoss")) / 3600.0
    shutil.rmtree(tmp, ignore_errors=True)
    np.save(out, bins)
    return {"elapsed": el, "T": 24, "agg": float(bins.sum())}


# --------------------------------------------------------------------------- EPANET
def _run_epanet(p, fid, out, variant="v0"):
    import wntr
    dm, rough, speed, tankf, rhead = p
    net = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net6.inp")
    t0 = time.time()
    wn = wntr.network.WaterNetworkModel(net)
    wn.options.time.duration = 7 * 24 * 3600
    wn.options.time.hydraulic_timestep = int(fid * 60)
    wn.options.time.report_timestep = int(fid * 60)
    wn.options.time.pattern_timestep = 3600
    wn.options.hydraulic.unbalanced = "CONTINUE"     # standard EPANET option: proceed after
    wn.options.hydraulic.unbalanced_value = 10       # 10 extra trials when a step does not converge
    wn.options.hydraulic.demand_multiplier = float(dm) * (1.1 if variant == "demand_shift" else 1.0)
    for _, pipe in wn.pipes():
        pipe.roughness = pipe.roughness * rough
    for _, pump in wn.pumps():
        pump.base_speed = float(speed)
    for _, tank in wn.tanks():
        tank.init_level = tank.min_level + tankf * (tank.max_level - tank.min_level)
    for _, res in wn.reservoirs():
        res.base_head = res.base_head + rhead
    tmp = tempfile.mkdtemp(prefix="epa_")
    # EPANET's default is to stop when a hydraulic step does not converge; such scenarios are flagged
    # 'unbalanced' (excluded from the evaluation) and their output is produced with the CONTINUE option
    unbalanced = False
    try:
        wn.options.hydraulic.unbalanced = "STOP"
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "q"))
    except Exception:
        unbalanced = True
        wn.options.hydraulic.unbalanced = "CONTINUE"
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "q2"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pw = wntr.metrics.pump_power(res.link["flowrate"], res.node["head"], wn)   # W per pump
    power_kw = pw.sum(axis=1).to_numpy(dtype=float) / 1000.0
    step_h = fid / 60.0
    energy = power_kw * step_h
    n = len(energy) - 1                                      # drop final instant
    per_h = int(round(1 / step_h))
    trace = energy[:n - n % per_h].reshape(-1, per_h).sum(axis=1)
    el = time.time() - t0
    np.save(out, trace)
    return {"elapsed": el, "T": int(len(trace)), "agg": float(trace.sum()), "unbalanced": bool(unbalanced)}


def run(engine: str, x, out: str, fidelity=None, variant: str = "v0") -> dict:
    fid = ENGINES[engine]["default_fidelity"] if fidelity is None else fidelity
    p = denorm(engine, x)
    if engine == "ochre":
        return _run_ochre(p, fid, out, variant=variant)
    if engine == "sumo":
        return _run_sumo(p, fid, out, variant=variant)
    if engine == "epanet":
        return _run_epanet(p, fid, out, variant=variant)
    raise ValueError(engine)
