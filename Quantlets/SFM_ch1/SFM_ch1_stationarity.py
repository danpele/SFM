"""
SFM_ch1_stationarity
====================
Stationarity and Returns Comparison Charts

Description:
- Download S&P 500 (^GSPC) data via yfinance
- Plot price level vs returns to illustrate stationarity
- Plot simple vs log returns divergence for different return magnitudes

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
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
print("SFM CHAPTER 1: STATIONARITY AND RETURNS COMPARISON")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2005-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna() * 100  # percentage

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(close)}")

# =============================================================================
# 2. Stationarity: Price vs Returns
# =============================================================================
print("\n2. CREATING STATIONARITY CHART")
print("-" * 40)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 2.8), height_ratios=[1, 1])

# Price level (non-stationary)
ax1.plot(close.index, close.values, color=MAIN_BLUE, linewidth=0.6)
ax1.set_ylabel('Price (\\$)')
ax1.set_title('S&P 500 Price Level (non-stationary)', fontsize=9, fontweight='bold')
ax1.text(0.02, 0.88, 'Trending mean\nGrowing variance',
         transform=ax1.transAxes, fontsize=7, color=CRIMSON,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=CRIMSON, alpha=0.8, linewidth=0.5))

# Returns (stationary)
ax2.plot(log_ret.index, log_ret.values, color=FOREST, linewidth=0.3, alpha=0.7)
ax2.axhline(y=0, color='gray', linewidth=0.4, linestyle=':')
ax2.set_ylabel('Log-return (\\%)')
ax2.set_title('S&P 500 Daily Log-Returns (stationary)', fontsize=9, fontweight='bold')
ax2.text(0.02, 0.88, 'Constant mean $\\approx 0$\nFinite variance',
         transform=ax2.transAxes, fontsize=7, color=FOREST,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=FOREST, alpha=0.8, linewidth=0.5))

plt.tight_layout(h_pad=1.0)
save_fig('sfm_ch1_stationarity')

# =============================================================================
# 3. Simple vs Log Returns Divergence
# =============================================================================
print("\n3. CREATING RETURNS DIVERGENCE CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

# Range of simple returns
R = np.linspace(-0.60, 1.0, 500)
r = np.log(1 + R)

ax.plot(R * 100, r * 100, color=CRIMSON, linewidth=1.2,
        label='Log-return $r_t = \\ln(1 + R_t)$')
ax.plot(R * 100, R * 100, color=MAIN_BLUE, linewidth=1.0, linestyle='--',
        label='45° line ($r_t = R_t$)')

# Shade the divergence region
ax.fill_between(R * 100, r * 100, R * 100, alpha=0.08, color=CRIMSON)

# Mark typical daily range
ax.axvspan(-3, 3, alpha=0.06, color=FOREST, zorder=0)
ax.text(0, -45, 'Typical\ndaily range', fontsize=6, color=FOREST,
        ha='center', va='center')

# Annotate divergence at extremes
ax.annotate('Large positive:\n$r_t < R_t$',
            xy=(60, np.log(1.6) * 100), xytext=(70, 25),
            fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))
ax.annotate('Large negative:\n$|r_t| > |R_t|$',
            xy=(-50, np.log(0.5) * 100), xytext=(-55, -25),
            fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))

ax.set_xlabel('Simple return $R_t$ (\\%)')
ax.set_ylabel('Log-return $r_t$ (\\%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
          frameon=False, fontsize=7)

save_fig('sfm_ch1_returns_divergence')

print("\n" + "=" * 70)
print("STATIONARITY CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_stationarity.pdf/.png")
print("  - sfm_ch1_returns_divergence.pdf/.png")
