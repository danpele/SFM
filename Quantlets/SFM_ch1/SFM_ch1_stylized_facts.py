"""
SFM_ch1_stylized_facts
======================
Stylized Facts of Financial Returns: QQ-Plot, ACF, Aggregational Gaussianity

Description:
- Download S&P 500 (^GSPC) data via yfinance
- QQ-plot of returns vs Normal (scipy.stats.probplot)
- ACF of returns and ACF of |returns| (side-by-side, statsmodels)
- Aggregational Gaussianity: daily/weekly/monthly histograms + Normal overlay

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import acf
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
print("SFM CHAPTER 1: STYLIZED FACTS CHARTS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2005-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(log_ret)}")

# =============================================================================
# 2. QQ-Plot
# =============================================================================
print("\n2. CREATING QQ-PLOT")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

# Standardize returns
z = (log_ret - log_ret.mean()) / log_ret.std()

# QQ-plot
theoretical_q, ordered_z = stats.probplot(z, dist="norm")[:2]
theor_quantiles = theoretical_q[0]
sample_quantiles = theoretical_q[1]

ax.scatter(theor_quantiles, sample_quantiles, s=3, alpha=0.4, color=CRIMSON,
           edgecolors='none', label='S&P 500 log-returns')

# 45-degree reference line
q_min, q_max = theor_quantiles.min(), theor_quantiles.max()
ax.plot([q_min, q_max], [q_min, q_max], '--', color='gray', linewidth=0.8,
        label='Normal reference')

# Annotate tails
ax.annotate('Left tail\n(more extreme)', xy=(theor_quantiles[5], sample_quantiles[5]),
            xytext=(-2.5, -1.5), fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))
ax.annotate('Right tail\n(more extreme)', xy=(theor_quantiles[-6], sample_quantiles[-6]),
            xytext=(1.5, 4.5), fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))

ax.set_xlabel('Theoretical Quantiles (Normal)')
ax.set_ylabel('Sample Quantiles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('sfm_ch1_qqplot')

# =============================================================================
# 3. ACF of Returns and ACF of |Returns| (side-by-side)
# =============================================================================
print("\n3. CREATING ACF CHART (2 panels)")
print("-" * 40)

n_lags = 30
acf_ret = acf(log_ret, nlags=n_lags, fft=True)
acf_abs = acf(np.abs(log_ret), nlags=n_lags, fft=True)

# Confidence band (approximate 95%)
conf = 1.96 / np.sqrt(len(log_ret))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Panel 1: ACF of r_t
lags = np.arange(1, n_lags + 1)
ax1.bar(lags, acf_ret[1:], width=0.6, color=MAIN_BLUE, alpha=0.8)
ax1.axhline(y=conf, color=CRIMSON, linestyle='--', linewidth=0.6)
ax1.axhline(y=-conf, color=CRIMSON, linestyle='--', linewidth=0.6)
ax1.axhline(y=0, color='gray', linewidth=0.4)
ax1.set_xlabel('Lag $k$')
ax1.set_ylabel('ACF')
ax1.set_title('Ch.1: ACF of $r_t$ (near zero)', fontweight='bold', fontsize=9)
ax1.set_ylim(-0.08, 0.08)

# Panel 2: ACF of |r_t|
ax2.bar(lags, acf_abs[1:], width=0.6, color=FOREST, alpha=0.8)
ax2.axhline(y=conf, color=CRIMSON, linestyle='--', linewidth=0.6)
ax2.axhline(y=-conf, color=CRIMSON, linestyle='--', linewidth=0.6)
ax2.axhline(y=0, color='gray', linewidth=0.4)
ax2.set_xlabel('Lag $k$')
ax2.set_ylabel('ACF')
ax2.set_title('Ch.1: ACF of $|r_t|$ (slowly decaying)', fontweight='bold', fontsize=9)
ax2.set_ylim(-0.03, max(acf_abs[1:]) * 1.15)

# Remove right/top spines on second panel too
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
save_fig('sfm_ch1_acf')

# =============================================================================
# 4. Aggregational Gaussianity
# =============================================================================
print("\n4. CREATING AGGREGATIONAL GAUSSIANITY CHART")
print("-" * 40)

# Compute weekly and monthly returns
weekly_ret = log_ret.resample('W').sum().dropna()
monthly_ret = log_ret.resample('ME').sum().dropna()

fig, ax = plt.subplots(figsize=(7, 3))

x = np.linspace(-5, 5, 300)

# Standardize each
for ret, label, color, lw in [
    (log_ret, 'Daily', CRIMSON, 1.2),
    (weekly_ret, 'Weekly', AMBER, 1.2),
    (monthly_ret, 'Monthly', FOREST, 1.2),
]:
    z_ret = (ret - ret.mean()) / ret.std()
    # KDE
    kde = stats.gaussian_kde(z_ret)
    ax.plot(x, kde(x), color=color, linewidth=lw, label=label)

# Normal reference
ax.plot(x, stats.norm.pdf(x), color=MAIN_BLUE, linewidth=0.8, linestyle='--',
        label='Normal')

ax.set_xlabel('Standardized return')
ax.set_ylabel('Density')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=7)
ax.set_xlim(-5, 5)
ax.set_ylim(0, None)

plt.tight_layout()
save_fig('sfm_ch1_aggregation')

print("\n" + "=" * 70)
print("STYLIZED FACTS CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_qqplot.pdf/.png")
print("  - sfm_ch1_acf.pdf/.png")
print("  - sfm_ch1_aggregation.pdf/.png")
