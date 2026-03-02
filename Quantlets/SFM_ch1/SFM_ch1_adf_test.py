"""
SFM_ch1_adf_test
=================
Augmented Dickey-Fuller Test: Prices vs Returns

Description:
- Download S&P 500 (^GSPC) data via yfinance
- Run ADF test on log-prices and log-returns
- Produce a table-style chart showing test statistic, p-value,
  critical values, and conclusion (reject/fail to reject)

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings('ignore')

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

# Color palette
MAIN_BLUE = '#1A3A6E'
CRIMSON = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'charts'))
os.makedirs(CHART_DIR, exist_ok=True)

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(os.path.join(CHART_DIR, f'{name}.pdf'),
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(CHART_DIR, f'{name}.png'),
                bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf/.png")

print("=" * 70)
print("SFM CHAPTER 1: AUGMENTED DICKEY-FULLER TEST")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2005-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_price = np.log(close).dropna()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(close)}")

# =============================================================================
# 2. Run ADF Tests
# =============================================================================
print("\n2. RUNNING ADF TESTS")
print("-" * 40)

# ADF on log-prices
adf_price = adfuller(log_price, autolag='AIC')
# ADF on log-returns
adf_ret = adfuller(log_ret, autolag='AIC')

print(f"   Log-prices: ADF stat = {adf_price[0]:.4f}, p-value = {adf_price[1]:.4f}")
print(f"   Log-returns: ADF stat = {adf_ret[0]:.4f}, p-value = {adf_ret[1]:.6f}")

# =============================================================================
# 3. Create Table-Style Chart
# =============================================================================
print("\n3. CREATING ADF TEST TABLE CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.axis('off')

# Build table data
col_labels = ['Series', 'ADF Statistic', 'p-value', '1%', '5%', '10%', 'Conclusion']

price_conclusion = 'Fail to reject $H_0$\n(unit root)' if adf_price[1] > 0.05 else 'Reject $H_0$\n(stationary)'
ret_conclusion = 'Fail to reject $H_0$\n(unit root)' if adf_ret[1] > 0.05 else 'Reject $H_0$\n(stationary)'

table_data = [
    ['$\\ln P_t$\n(log-prices)',
     f'{adf_price[0]:.3f}',
     f'{adf_price[1]:.4f}',
     f'{adf_price[4]["1%"]:.3f}',
     f'{adf_price[4]["5%"]:.3f}',
     f'{adf_price[4]["10%"]:.3f}',
     price_conclusion],
    ['$r_t = \\Delta \\ln P_t$\n(log-returns)',
     f'{adf_ret[0]:.3f}',
     f'{adf_ret[1]:.6f}',
     f'{adf_ret[4]["1%"]:.3f}',
     f'{adf_ret[4]["5%"]:.3f}',
     f'{adf_ret[4]["10%"]:.3f}',
     ret_conclusion],
]

# Title
ax.text(0.5, 0.95, 'Augmented Dickey-Fuller Test: S&P 500',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        ha='center', va='top', color=MAIN_BLUE)

ax.text(0.5, 0.85, '$H_0$: series has a unit root (non-stationary)  vs  '
        '$H_1$: series is stationary',
        transform=ax.transAxes, fontsize=8, ha='center', va='top',
        color='#555555', style='italic')

# Create table
table = ax.table(cellText=table_data,
                 colLabels=col_labels,
                 loc='center',
                 cellLoc='center',
                 bbox=[0.0, 0.0, 1.0, 0.75])

table.auto_set_font_size(False)
table.set_fontsize(8)

# Style header row
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor(MAIN_BLUE)
    cell.set_text_props(color='white', fontweight='bold', fontsize=7.5)
    cell.set_edgecolor('white')
    cell.set_height(0.18)

# Style data rows
for i in range(1, 3):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_edgecolor('#cccccc')
        cell.set_height(0.28)
        if i == 1:
            cell.set_facecolor('#fff5f5')  # light red for non-stationary
        else:
            cell.set_facecolor('#f0fff0')  # light green for stationary

    # Color the conclusion cell
    conclusion_cell = table[i, 6]
    if i == 1:
        conclusion_cell.set_text_props(color=CRIMSON, fontweight='bold', fontsize=7.5)
    else:
        conclusion_cell.set_text_props(color=FOREST, fontweight='bold', fontsize=7.5)

save_fig('sfm_ch1_adf_test')

print("\n" + "=" * 70)
print("ADF TEST CHART COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_adf_test.pdf/.png")
