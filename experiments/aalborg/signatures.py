"""Energy signature of each selected Aalborg house in 2019 from its smart heat meter."""
import json, sys, os, numpy as np, pandas as pd
D = sys.argv[1]
om = json.load(open(os.path.join(D, "openmeteo_2019.json")))["hourly"]
T = pd.Series(om["temperature_2m"], index=pd.to_datetime(om["time"]).tz_localize("UTC")).resample("D").mean()
ctx = pd.read_csv(os.path.join(D, "01_Data/03_context_data/contextual_data.csv")).set_index("meter_id")
rows = []
for m in [int(x) for x in open(os.path.join(D, "selected_meters.txt")).read().split()]:
    d = pd.read_csv(os.path.join(D, "01_Data/02_processed_final_data/%d.csv" % m), parse_dates=["time_rounded"])
    s = d.set_index("time_rounded")["energy_heat_kwh"].sort_index()
    s = s["2018-12-31":"2020-01-02"]
    hourly = s.diff().clip(lower=0)["2019-01-01":"2019-12-31"]
    if len(hourly) < 8700:
        continue
    day = hourly.resample("D").sum()
    j = pd.concat([day.rename("q"), T.rename("t")], axis=1).dropna()
    summer = j[j.t >= 16].q.mean()                  # base load (domestic hot water + losses)
    w = j[j.t < 14]
    slope = np.polyfit(w.t, w.q, 1)[0]              # kWh/day per K (negative)
    vol = d.set_index("time_rounded")["volume_flow_m3"]["2019-01-01":"2019-12-31"]
    rows.append(dict(meter_id=m, construction_year=int(ctx.loc[m, "construction_year"]), energy_label=ctx.loc[m, "energy_label"],
                     annual_kwh=float(hourly.sum()), base_kwh_day=float(summer), slope_kwh_day_K=float(-slope),
                     n_hours=int(len(hourly))))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows).describe().T)
