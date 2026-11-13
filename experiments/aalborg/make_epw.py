"""Build an Aalborg 2019 EPW: TMYx 2007-2021 (Aalborg AP, WMO 060300) as template, hourly fields replaced
by open-meteo ERA5 reanalysis for 2019 (57.05N, 9.92E)."""
import json, sys
tmpl, om, out = sys.argv[1:4]
L = open(tmpl).read().splitlines()
hdr, rows = L[:8], L[8:]
hdr[7] = hdr[7].replace("Sunday", "Tuesday")    # 1 January 2019 was a Tuesday
h = json.load(open(om))["hourly"]
assert len(rows) == 8760 == len(h["time"])
new = []
for i, r in enumerate(rows):
    c = r.split(",")
    c[0] = "2019"
    c[6] = "%.1f" % h["temperature_2m"][i]
    c[7] = "%.1f" % h["dew_point_2m"][i]
    c[8] = "%d" % round(h["relative_humidity_2m"][i])
    c[9] = "%d" % round(100 * h["surface_pressure"][i])
    c[13] = "%d" % round(h["shortwave_radiation"][i])
    c[14] = "%d" % round(h["direct_normal_irradiance"][i])
    c[15] = "%d" % round(h["diffuse_radiation"][i])
    c[20] = "%d" % round(h["wind_direction_10m"][i])
    c[21] = "%.1f" % h["wind_speed_10m"][i]
    cc = round(h["cloud_cover"][i] / 10)
    c[22] = "%d" % cc; c[23] = "%d" % cc
    new.append(",".join(c))
open(out, "w").write("\n".join(hdr + new) + "\n")
print("wrote", out)
