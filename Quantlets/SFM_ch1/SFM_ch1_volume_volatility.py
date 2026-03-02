"""
SFM_ch1_volume_volatility
=========================
Volume-Volatility Correlation Chart

Description:
- Download S&P 500 (^GSPC) data via yfinance
- Create scatter plot of daily volume vs absolute returns
- Add regression line and correlation coefficient
- Illustrates the volume-volatility stylized fact

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
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
print("SFM CHAPTER 1: VOLUME-VOLATILITY CORRELATION")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
volume = data['Volume'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna() * 100  # percentage
abs_ret = log_ret.abs()

# Align volume with returns
volume = volume.loc[abs_ret.index]
vol_millions = volume / 1e6

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(abs_ret)}")

# =============================================================================
# 2. Volume-Volatility Scatter
# =============================================================================
print("\n2. CREATING VOLUME-VOLATILITY SCATTER")
print("-" * 40)

# Compute correlation
corr, pval = stats.pearsonr(vol_millions.values, abs_ret.values)
print(f"   Correlation: {corr:.4f} (p-value: {pval:.2e})")

# Regression line
slope, intercept, _, _, _ = stats.linregress(vol_millions.values, abs_ret.values)

fig, ax = plt.subplots(figsize=(7, 3))

ax.scatter(vol_millions.values, abs_ret.values, s=2, alpha=0.15,
           color=MAIN_BLUE, edgecolors='none', rasterized=True)

# Regression line
x_fit = np.linspace(vol_millions.min(), vol_millions.max(), 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color=CRIMSON, linewidth=1.5, linestyle='-',
        label=f'OLS fit ($\\rho$ = {corr:.3f})')

# Correlation annotation
ax.text(0.97, 0.95, f'$\\rho$ = {corr:.3f}\np < 0.001',
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=CRIMSON, alpha=0.8, linewidth=0.5))

ax.set_xlabel('Daily Volume (millions of shares)')
ax.set_ylabel('$|r_t|$ Absolute Log-Return (\\%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1,
          frameon=False, fontsize=7)

plt.tight_layout()
save_fig('sfm_ch1_volume_vol')

print("\n" + "=" * 70)
print("VOLUME-VOLATILITY CHART COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_volume_vol.pdf/.png")
