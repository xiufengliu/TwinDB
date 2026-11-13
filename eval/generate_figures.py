#!/usr/bin/env python3
"""Generate publication-quality figures for BDG2 validation results."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

df = pd.read_csv('eval/results/bdg2_hvac.csv')

fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

# (a) R² Distribution by Meter Type
ax = axes[0]
meter_types = ['cooling', 'heating_steam', 'heating_hw']
labels = ['Cooling', 'Steam', 'Hot Water']
colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']
data = [df[df['meter_type'] == m]['r2'].values for m in meter_types]
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='ISM threshold')
ax.set_ylabel('R²')
ax.set_title('(a) R² by Meter Type')
ax.set_ylim(-0.05, 1.0)
ax.legend(fontsize=8, loc='upper right')

# (b) Sensitivity Error Distribution  
ax = axes[1]
good = df[df['r2'] > 0.3]
ax.hist(good['sensitivity_error'], bins=20, color='#45B7D1', edgecolor='black', alpha=0.7)
ax.axvline(x=good['sensitivity_error'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {good["sensitivity_error"].mean():.1f}%')
ax.set_xlabel('Sensitivity Error (%)')
ax.set_ylabel('Count')
ax.set_title('(b) ISM Learning Error')
ax.legend(fontsize=8)
ax.set_xlim(0, 30)

# (c) Temperature Correlation vs R²
ax = axes[2]
sample = df.sample(n=min(300, len(df)), random_state=42)
colors_scatter = ['#4ECDC4' if m == 'cooling' else '#FF6B6B' if m == 'heating_steam' else '#FFE66D'
                  for m in sample['meter_type']]
ax.scatter(sample['correlation'], sample['r2'], c=colors_scatter, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Temperature Correlation')
ax.set_ylabel('R²')
ax.set_title('(c) Correlation vs R²')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4ECDC4', label='Cooling'),
                   Patch(facecolor='#FF6B6B', label='Steam'),
                   Patch(facecolor='#FFE66D', label='Hot Water')]
ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('paper/figures/bdg2_validation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/bdg2_validation.png', bbox_inches='tight', dpi=150)
print("Saved: paper/figures/bdg2_validation.pdf")

print("\n=== Summary Statistics ===")
for meter, label in zip(meter_types, labels):
    subset = df[df['meter_type'] == meter]
    good_subset = subset[subset['r2'] > 0.3]
    if len(good_subset) > 0:
        print(f"{label}: n={len(subset)}, R²={subset['r2'].mean():.2f}, "
              f">{0.3}: {len(good_subset)/len(subset)*100:.0f}%, "
              f"sens_err={good_subset['sensitivity_error'].mean():.1f}%")
    else:
        print(f"{label}: n={len(subset)}, R²={subset['r2'].mean():.2f}, no good candidates")
