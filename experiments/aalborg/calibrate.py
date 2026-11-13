"""Calibrate one OCHRE twin per Aalborg house to its 2019 smart-heat-meter data.

Step 1 (grid): simulate the baseline scenario on a grid of entity descriptors (k_env x k_dhw).
Step 2 (fit):  per house, choose (k_env, k_dhw) by interpolating the grid so that the twin matches the
               metered annual heat and the metered summer base load (days with mean T >= 16 C).
Step 3 (check): simulate each calibrated twin once and report ASHRAE Guideline 14 statistics
               (monthly NMBE and CV(RMSE)) against the meter.
Outputs: calibration.json (descriptors and fit statistics per house), grid traces in gt_cal/.
"""
import json, os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
import aalborg_engine as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gt_cal"); os.makedirs(OUT, exist_ok=True)
KE = [0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.0]
KD = [0.2, 0.6, 1.0, 1.8, 2.8, 4.0, 6.0]
BASE = [0.0, 0.0, 1.0, 1.0, 21.0]          # baseline scenario: no retrofit, 21 C


def sim(p):
    sig = A.signature(A.norm(p))
    f = os.path.join(OUT, sig + ".npy")
    if not os.path.exists(f):
        r = A.run(p, f)
        json.dump({"p": list(p), "elapsed": r["elapsed"]}, open(f[:-4] + ".json", "w"))
    return np.load(f)


def stats(trace):
    t = pd.Series(trace, index=pd.date_range("2019-01-01", periods=8760, freq="h", tz="UTC"))
    return t


def main():
    T = json.load(open(os.path.join(A.DATA, "openmeteo_2019.json")))["hourly"]["temperature_2m"]
    Td = pd.Series(T, index=pd.date_range("2019-01-01", periods=8760, freq="h", tz="UTC")).resample("D").mean()
    grid = [[ke, kd] + BASE for ke in KE for kd in KD]
    with ProcessPoolExecutor(2) as ex:
        tr = list(ex.map(sim, grid))
    ann = np.array([t.sum() for t in tr]).reshape(len(KE), len(KD))
    summ = []
    for t in tr:
        d = stats(t).resample("D").sum()
        summ.append(d[Td >= 16].mean())
    summ = np.array(summ).reshape(len(KE), len(KD))
    fa = RegularGridInterpolator((np.log(KE), np.log(KD)), np.log(ann), bounds_error=False, fill_value=None)
    fs = RegularGridInterpolator((np.log(KE), np.log(KD)), np.log(summ), bounds_error=False, fill_value=None)
    sig = pd.read_csv(os.path.join(HERE, "signatures_2019.csv"))
    res = []
    for _, r in sig.iterrows():
        def loss(z):
            return (fa([z])[0] - np.log(r.annual_kwh)) ** 2 + (fs([z])[0] - np.log(r.base_kwh_day)) ** 2
        z = minimize(loss, [0.0, 0.0], bounds=[(np.log(KE[0]), np.log(KE[-1])), (np.log(KD[0]), np.log(KD[-1]))]).x
        res.append(dict(meter_id=int(r.meter_id), k_env=float(np.exp(z[0])), k_dhw=float(np.exp(z[1])),
                        construction_year=int(r.construction_year), energy_label=r.energy_label,
                        meter_annual_kwh=float(r.annual_kwh), meter_base_kwh_day=float(r.base_kwh_day)))
    # check: simulate each calibrated twin and compare monthly with the meter
    ps = [[q["k_env"], q["k_dhw"]] + BASE for q in res]
    with ProcessPoolExecutor(2) as ex:
        trs = list(ex.map(sim, ps))
    for q, t in zip(res, trs):
        m = pd.read_csv(os.path.join(A.DATA, "01_Data/02_processed_final_data/%d.csv" % q["meter_id"]), parse_dates=["time_rounded"])
        s = m.set_index("time_rounded")["energy_heat_kwh"].sort_index()["2018-12-31":"2020-01-02"].diff().clip(lower=0)["2019-01-01":"2019-12-31"]
        mm = s.resample("MS").sum().to_numpy()[:12]
        sm = stats(t).resample("MS").sum().to_numpy()[:12]
        dm = s.resample("D").sum(); ds = stats(t).resample("D").sum()
        n = min(len(dm), len(ds))
        q["twin_annual_kwh"] = float(t.sum())
        q["nmbe_monthly"] = float((mm - sm).sum() / (len(mm) - 1) / mm.mean())
        q["cvrmse_monthly"] = float(np.sqrt(((mm - sm) ** 2).sum() / (len(mm) - 1)) / mm.mean())
        q["cvrmse_daily"] = float(np.sqrt(((dm.to_numpy()[:n] - ds.to_numpy()[:n]) ** 2).sum() / (n - 1)) / dm.mean())
        q["at_grid_edge"] = bool(q["k_env"] <= KE[0] * 1.01 or q["k_env"] >= KE[-1] * 0.99 or q["k_dhw"] <= KD[0] * 1.01 or q["k_dhw"] >= KD[-1] * 0.99)
    json.dump({"grid": {"k_env": KE, "k_dhw": KD, "annual": ann.tolist(), "summer_base": summ.tolist()},
               "houses": res}, open(os.path.join(HERE, "calibration.json"), "w"), indent=1)
    df = pd.DataFrame(res)
    print(df[["k_env", "k_dhw", "nmbe_monthly", "cvrmse_monthly", "cvrmse_daily"]].describe())
    print("within ASHRAE G14 monthly (|NMBE|<=5%, CV<=15%):",
          int(((df.nmbe_monthly.abs() <= 0.05) & (df.cvrmse_monthly <= 0.15)).sum()), "of", len(df),
          "; at grid edge:", int(df.at_grid_edge.sum()))


if __name__ == "__main__":
    main()
