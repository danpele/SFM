"""
SFM_ch1_returns_analysis
========================
Cumulative Returns, Distribution Shapes, and Drawdowns

Description:
- Download S&P 500 (^GSPC) data 2015-2024 via yfinance
- Cumulative log-return path with shaded area
- Density comparison: Normal vs fat-tailed (Student-t) vs left-skewed
- Drawdown chart: price path with MDD highlighted

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
ORANGE = '#E67E22'

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
print("SFM CHAPTER 1: RETURNS ANALYSIS CHARTS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(close)}")

# =============================================================================
# 2. Cumulative Log-Return Path
# =============================================================================
print("\n2. CREATING CUMULATIVE LOG-RETURN CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

cum_log_ret = log_ret.cumsum()

ax.plot(cum_log_ret.index, cum_log_ret.values, color=MAIN_BLUE, linewidth=0.8)
ax.fill_between(cum_log_ret.index, 0, cum_log_ret.values,
                alpha=0.12, color=MAIN_BLUE)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Log-Return')
ax.text(0.95, 0.05, 'CLR path', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=8, color=MAIN_BLUE)

plt.tight_layout()
save_fig('sfm_ch1_cumulative')

# =============================================================================
# 3. Density Comparison: Normal vs Fat-tailed vs Skewed
# =============================================================================
print("\n3. CREATING DENSITY COMPARISON CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

x = np.linspace(-4, 4, 300)

# Normal density
normal_pdf = stats.norm.pdf(x, 0, 1)
ax.plot(x, normal_pdf, color=MAIN_BLUE, linewidth=1.2, label='Normal ($S=0, K=3$)')

# Fat-tailed (Student-t with df=5, standardized to variance=1)
df_t = 5
scale_t = np.sqrt((df_t - 2) / df_t)  # scale so that variance = 1
t_pdf = stats.t.pdf(x / scale_t, df=df_t) / scale_t
ax.plot(x, t_pdf, color=CRIMSON, linewidth=1.2, label='Fat-tailed ($t_5$, $K>3$)')

# Left-skewed (skew-normal)
from scipy.stats import skewnorm
skew_pdf = skewnorm.pdf(x, a=-4, loc=0.5, scale=1.0)
ax.plot(x, skew_pdf, color=FOREST, linewidth=1.2, label='Left-skewed ($S<0$)')

ax.set_xlabel('$r$')
ax.set_ylabel('Density')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)
ax.set_xlim(-4, 4)
ax.set_ylim(0, None)

plt.tight_layout()
save_fig('sfm_ch1_densities')

# =============================================================================
# 4. Drawdown Chart
# =============================================================================
print("\n4. CREATING DRAWDOWN CHART")
print("-" * 40)

# Use a subset with a clear drawdown (2020 COVID crash)
price_sub = close.loc['2019-06-01':'2021-06-01'].copy()

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(price_sub.index, price_sub.values, color=MAIN_BLUE, linewidth=0.8)

# Compute drawdown
running_max = price_sub.cummax()
drawdown = (price_sub - running_max) / running_max

# Find maximum drawdown
mdd_end_idx = drawdown.idxmin()
mdd_value = drawdown.min()
# Find the peak before the trough
peak_idx = price_sub.loc[:mdd_end_idx].idxmax()
peak_price = price_sub.loc[peak_idx]
trough_price = price_sub.loc[mdd_end_idx]

# Highlight MDD region
ax.axhline(y=peak_price, xmin=0, xmax=1, color=AMBER, linestyle='--',
           linewidth=0.6, alpha=0.5)
ax.fill_between(price_sub.loc[peak_idx:mdd_end_idx].index,
                peak_price,
                price_sub.loc[peak_idx:mdd_end_idx].values.flatten(),
                alpha=0.15, color=CRIMSON)

# Annotate peak and trough
ax.annotate('Peak', xy=(peak_idx, peak_price), fontsize=7, color=AMBER,
            xytext=(10, 8), textcoords='offset points')
ax.annotate('Trough', xy=(mdd_end_idx, trough_price), fontsize=7, color=CRIMSON,
            xytext=(10, -12), textcoords='offset points')

# MDD arrow
mid_date = peak_idx + (mdd_end_idx - peak_idx) / 2
ax.annotate('', xy=(mdd_end_idx, trough_price),
            xytext=(mdd_end_idx, peak_price),
            arrowprops=dict(arrowstyle='<->', color=CRIMSON, lw=1.0))
ax.text(mdd_end_idx, (peak_price + trough_price) / 2,
        f'  MDD\n  {mdd_value:.1%}', fontsize=7, color=CRIMSON, va='center')

ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')

plt.tight_layout()
save_fig('sfm_ch1_drawdown')

print("\n" + "=" * 70)
print("RETURNS ANALYSIS CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_cumulative.pdf/.png")
print("  - sfm_ch1_densities.pdf/.png")
print("  - sfm_ch1_drawdown.pdf/.png")
