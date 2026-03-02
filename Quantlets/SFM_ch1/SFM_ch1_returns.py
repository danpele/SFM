"""
SFM_ch1_returns
===============
Simple and Log Returns: Computation and Visualization

Description:
- Download MSFT stock data via yfinance
- Compute simple and log returns
- Plot returns time series
- Scatter plot: simple vs log returns
- Histogram comparison with Normal overlay
- Compute skewness and kurtosis

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
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

import os
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
print("SFM CHAPTER 1: RETURNS ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

ticker = 'MSFT'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()

print(f"   Ticker: {ticker}")
print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(close)}")

# =============================================================================
# 2. Compute Returns
# =============================================================================
print("\n2. COMPUTING RETURNS")
print("-" * 40)

simple_return = close.pct_change().dropna()
log_return = np.log(close / close.shift(1)).dropna()

print(f"   Simple returns - Mean: {simple_return.mean():.6f}, Std: {simple_return.std():.6f}")
print(f"   Log returns    - Mean: {log_return.mean():.6f}, Std: {log_return.std():.6f}")

# =============================================================================
# 3. Descriptive Statistics
# =============================================================================
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)

skew_simple = stats.skew(simple_return)
kurt_simple = stats.kurtosis(simple_return)
skew_log = stats.skew(log_return)
kurt_log = stats.kurtosis(log_return)

print(f"   Simple returns:")
print(f"     Skewness:        {skew_simple:.4f}")
print(f"     Excess Kurtosis: {kurt_simple:.4f}")
print(f"   Log returns:")
print(f"     Skewness:        {skew_log:.4f}")
print(f"     Excess Kurtosis: {kurt_log:.4f}")

# Jarque-Bera test
jb_stat, jb_pval = stats.jarque_bera(log_return)
print(f"\n   Jarque-Bera test (log returns):")
print(f"     Statistic: {jb_stat:.2f}")
print(f"     p-value:   {jb_pval:.6f}")

# =============================================================================
# 4. FIGURE: Returns Analysis (4-panel)
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
fig.suptitle('Ch.1: Returns Analysis', fontweight='bold', fontsize=12)

# Panel A: Close price
axes[0, 0].plot(close.index, close, color='#1A3A6E', linewidth=0.8)
axes[0, 0].set_title(f'{ticker} Close Price', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')

# Panel B: Simple and Log returns time series
axes[0, 1].plot(simple_return.index, simple_return, color='#1A3A6E',
                linewidth=0.5, alpha=0.8, label='Simple Return')
axes[0, 1].plot(log_return.index, log_return, color='#DC3545',
                linewidth=0.5, alpha=0.6, label='Log Return')
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[0, 1].set_title('Simple vs Log Returns', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)

# Panel C: Scatter plot simple vs log
axes[1, 0].scatter(simple_return, log_return, s=2, alpha=0.3, color='#1A3A6E')
lim = max(abs(simple_return.min()), abs(simple_return.max())) * 1.1
axes[1, 0].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='45-degree line')
axes[1, 0].set_title('Simple vs Log Returns (Scatter)', fontweight='bold')
axes[1, 0].set_xlabel('Simple Return')
axes[1, 0].set_ylabel('Log Return')
corr = np.corrcoef(simple_return, log_return)[0, 1]
axes[1, 0].text(0.05, 0.95, f'Corr: {corr:.6f}',
               transform=axes[1, 0].transAxes, ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel D: Distribution comparison
axes[1, 1].hist(log_return, bins=80, density=True, alpha=0.5, color='#1A3A6E',
                edgecolor='white', label='Log Returns')
x = np.linspace(log_return.min(), log_return.max(), 200)
axes[1, 1].plot(x, stats.norm.pdf(x, log_return.mean(), log_return.std()),
               color='#DC3545', linewidth=2, label='Normal PDF')
axes[1, 1].set_title('Log Return Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Return')
axes[1, 1].text(0.95, 0.95,
               f'Skewness: {skew_log:.2f}\nKurtosis: {kurt_log:.2f}',
               transform=axes[1, 1].transAxes, ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig('ch1_returns')

print("\n" + "=" * 70)
print("RETURNS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch1_returns.pdf: 4-panel returns analysis")
