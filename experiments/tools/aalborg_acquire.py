"""Acquire the Aalborg smart-heat-meter files used in the study from Zenodo record 6563114 (CC-BY-4.0):
the context file, the description PDF, and the processed series of 67 single-family houses drawn
(seed 0, at most 9 per construction period) among those with construction year and energy label."""
import os, sys, subprocess
import pandas as pd
D = sys.argv[1]
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remotezip.py")
subprocess.run([sys.executable, T, "get", D, "contextual_data.csv"], check=True)
listing = subprocess.run([sys.executable, T, "ls"], check=True, capture_output=True, text=True).stdout.splitlines()
have = set(int(l.split()[0].split('/')[-1][:-4]) for l in listing if '02_processed_final_data/' in l and l.split()[0].endswith('.csv'))
d = pd.read_csv(os.path.join(D, "01_Data/03_context_data/contextual_data.csv"))
d = d[(d.unit_type == 'single_family_house') & d.energy_label.notna() & d.construction_year.notna() & d.meter_id.isin(have)]
d['band'] = pd.cut(d.construction_year, [0, 1930, 1950, 1960, 1973, 1979, 1998, 2006, 2025], labels=False)
sel = []
for b, g in d.groupby('band'):
    sel += list(g.sample(min(9, len(g)), random_state=int(b)).meter_id)
open(os.path.join(D, "selected_meters.txt"), "w").write("\n".join(map(str, sel)))
subprocess.run([sys.executable, T, "get", D] + ["02_processed_final_data/%d.csv" % m for m in sel], check=True)
print("acquired", len(sel))
