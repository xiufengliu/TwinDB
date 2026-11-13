"""Aalborg district-heating portfolio twin.

Each entity is a real single-family house in Aalborg (Schaffer et al., Sci. Data 9:420, 2022;
Zenodo 10.5281/zenodo.6563114) whose 2019 smart-heat-meter series calibrates an NREL OCHRE 0.9 dwelling
model run with Aalborg 2019 weather (ERA5 via open-meteo on the Aalborg TMYx template).  The building
is heated by district heating, modelled as an ideal heat source: the OCHRE furnace with AFUE 1.0 and a
hot-water tank with recovery efficiency 0.98 and standby losses (EF 0.90), so the simulated gas energy equals delivered heat.

Parameter vector (7 dims, normalized to [0,1] in the view):
  entity descriptors (fixed per house, from calibration)
    0 k_env     conductance multiplier of the envelope (walls, attic floor, windows, infiltration)
    1 k_dhw     hot-water fixture usage multiplier
  scenario parameters (what an energy advisor varies)
    2 wall_add  added wall insulation R (ft2.F.h/BTU)
    3 roof_add  added attic-floor insulation R
    4 win_f     window U-factor multiplier (new glazing < 1)
    5 air_f     infiltration multiplier (sealing < 1)
    6 setp_C    heating setpoint (deg C)
"""
_TWINDB_ROOT = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), *[".."]*1))
import hashlib, json, os, subprocess, tempfile, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = __import__("os").path.join(_TWINDB_ROOT, "inputs/aalborg")
OCHRE_DEFAULTS = __import__("os").environ.get("OCHRE_DEFAULTS", "")
OCHRE_PY = __import__("os").environ.get("OCHRE_PY", __import__("sys").executable)
__import__("os").environ.setdefault("TWINDB_STUBS", __import__("os").path.join(_TWINDB_ROOT, "tools", "stubs"))
EPW = os.path.join(DATA, "Aalborg_2019.epw")

PARAMS = ["k_env", "k_dhw", "wall_add", "roof_add", "win_f", "air_f", "setp_C"]
BOUNDS = [(0.2, 3.0), (0.2, 6.0), (0.0, 20.0), (0.0, 40.0), (0.35, 1.0), (0.4, 1.0), (18.0, 23.0)]
N_ENTITY = 2


def denorm(x):
    b = np.array(BOUNDS, float)
    return b[:, 0] + np.clip(np.asarray(x, float), 0, 1) * (b[:, 1] - b[:, 0])


def norm(p):
    b = np.array(BOUNDS, float)
    return (np.asarray(p, float) - b[:, 0]) / (b[:, 1] - b[:, 0])


def signature(x, fid=60, variant="v0"):
    key = json.dumps(["aalborg", [round(float(v), 10) for v in x], fid, variant])
    return hashlib.sha256(key.encode()).hexdigest()[:20]


WORKER = r'''
import sys, os, json, re, time, datetime as dt, tempfile, shutil
sys.path.insert(0, os.environ.get("TWINDB_STUBS", ""))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ochre import Dwelling
job = json.loads(sys.argv[1])
import ochre as _o; D = job["defaults"] or os.path.join(os.path.dirname(_o.__file__), "defaults")
k_env, k_dhw, wall_add, roof_add, win_f, air_f, setp_C = job["p"]
xml = open(os.path.join(D, "Input Files", "bldg0112631-up00.xml")).read()
def sub_section(x, sec, fn):
    i = x.find("<%s>" % sec); j = x.find("</%s>" % sec)
    return x[:i] + fn(x[i:j]) + x[j:]
xml = sub_section(xml, "Walls", lambda s: s.replace("<AssemblyEffectiveRValue>16.8<", "<AssemblyEffectiveRValue>%.4f<" % (16.8 / k_env + wall_add)))
xml = sub_section(xml, "Floors", lambda s: s.replace("<AssemblyEffectiveRValue>39.6<", "<AssemblyEffectiveRValue>%.4f<" % (39.6 / k_env + roof_add))
                                         .replace("<AssemblyEffectiveRValue>22.6<", "<AssemblyEffectiveRValue>%.4f<" % (22.6 / k_env)))
xml = sub_section(xml, "Windows", lambda s: re.sub(r"<UFactor>[0-9.]+<", "<UFactor>%.4f<" % min(0.76 * k_env * win_f, 1.5), s))
xml = re.sub(r"(<UnitofMeasure>ACH</UnitofMeasure>\s*<AirLeakage>)[0-9.]+(<)", lambda m: m.group(1) + "%.4f" % (15.0 * k_env * air_f) + m.group(2), xml)
sp = setp_C * 9 / 5 + 32
def setp_list(v):
    return lambda m: m.group(1) + ", ".join(["%.3f" % v] * 24) + m.group(2)
xml = re.sub(r"(<WeekdaySetpointTempsHeatingSeason>)[^<]*(<)", setp_list(sp), xml)
xml = re.sub(r"(<WeekendSetpointTempsHeatingSeason>)[^<]*(<)", setp_list(sp), xml)
xml = re.sub(r"(<WeekdaySetpointTempsCoolingSeason>)[^<]*(<)", setp_list(95.0), xml)
xml = re.sub(r"(<WeekendSetpointTempsCoolingSeason>)[^<]*(<)", setp_list(95.0), xml)
xml = xml.replace("<HeatingSetpoint dataSource='software'>70.0<", "<HeatingSetpoint dataSource='software'>%.2f<" % sp)
xml = xml.replace("<CoolingSetpoint dataSource='software'>75.0<", "<CoolingSetpoint dataSource='software'>95.0<")
# district heating as an ideal heat source
xml = xml.replace("<Units>AFUE</Units>\n                <Value>0.925</Value>", "<Units>AFUE</Units>\n                <Value>1.0</Value>")
xml = xml.replace("<EnergyFactor>0.59</EnergyFactor>", "<EnergyFactor>0.90</EnergyFactor>")
xml = xml.replace("<RecoveryEfficiency>0.76</RecoveryEfficiency>", "<RecoveryEfficiency>0.98</RecoveryEfficiency>")
tmp = tempfile.mkdtemp(prefix="aal_")
xml = xml.replace("<WaterFixturesUsageMultiplier>1.0</WaterFixturesUsageMultiplier>", "<WaterFixturesUsageMultiplier>%.4f</WaterFixturesUsageMultiplier>" % k_dhw)
sf = os.path.join(D, "Input Files", "bldg0112631_schedule.csv")
xf = os.path.join(tmp, "house.xml"); open(xf, "w").write(xml)
args = dict(name="q", start_time=dt.datetime(2019, 1, 1), time_res=dt.timedelta(minutes=job["fid"]),
            duration=dt.timedelta(days=365), initialization_time=dt.timedelta(days=7),
            hpxml_file=xf, hpxml_schedule_file=sf, weather_file=job["epw"], output_path=tmp,
            verbosity=1, save_results=False)
t0 = time.time()
d = Dwelling(**args)
df, metrics, hourly = d.simulate()
el = time.time() - t0
gas = df["Total Gas Power (therms/hour)"].to_numpy(dtype=float) * 29.3071   # kW of delivered heat
step_h = job["fid"] / 60.0
energy = gas * step_h
per_h = int(round(1 / step_h)) if step_h < 1 else 1
trace = energy.reshape(-1, per_h).sum(axis=1) if per_h > 1 else energy
np.save(job["out"], trace)
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps({"elapsed": el, "T": len(trace), "agg": float(trace.sum())}))
'''


def run(p, out, fid=60):
    job = {"p": [float(v) for v in p], "fid": fid, "out": out, "defaults": OCHRE_DEFAULTS, "epw": EPW}
    r = subprocess.run([OCHRE_PY, "-c", WORKER, json.dumps(job)], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    import sys
    p = [float(v) for v in sys.argv[1:8]] if len(sys.argv) > 7 else [1, 1, 0, 0, 1, 1, 21]
    print(run(p, "/tmp/aal_test.npy"))
