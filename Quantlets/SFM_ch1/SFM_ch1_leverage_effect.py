"""
SFM_ch1_leverage_effect
=======================
News Impact Curve: Asymmetric Volatility Response

Description:
- Download S&P 500 (^GSPC) data via yfinance
- Fit GJR-GARCH(1,1) model using arch package
- Plot the news impact curve showing asymmetric volatility response
- Negative shocks (bad news) increase volatility more than positive shocks

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
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
print("SFM CHAPTER 1: LEVERAGE EFFECT / NEWS IMPACT CURVE")
print("=" * 70)

# =============================================================================
# 1. Download Data and Fit GJR-GARCH
# =============================================================================
print("\n1. DOWNLOADING S&P 500 DATA")
print("-" * 40)

data = yf.download('^GSPC', start='2005-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()
ret_pct = log_ret * 100  # in percentage for arch package

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(ret_pct)}")

print("\n2. FITTING GJR-GARCH(1,1)")
print("-" * 40)

gjr = arch_model(ret_pct, vol='GARCH', p=1, o=1, q=1, dist='normal')
res = gjr.fit(disp='off')

omega = res.params['omega']
alpha = res.params['alpha[1]']
gamma = res.params['gamma[1]']
beta = res.params['beta[1]']

print(f"   omega = {omega:.6f}")
print(f"   alpha = {alpha:.4f}")
print(f"   gamma = {gamma:.4f}")
print(f"   beta  = {beta:.4f}")
print(f"   alpha + gamma = {alpha + gamma:.4f} (negative shock weight)")

# =============================================================================
# 2. News Impact Curve
# =============================================================================
print("\n3. CREATING NEWS IMPACT CURVE")
print("-" * 40)

# Unconditional variance
uncond_var = omega / (1 - alpha - gamma / 2 - beta)
sigma2_bar = uncond_var

# Range of shocks
eps = np.linspace(-4, 4, 500)

# GJR-GARCH news impact: sigma_t^2 = omega + (alpha + gamma * I(eps<0)) * eps^2 + beta * sigma2_bar
nic_gjr = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * sigma2_bar

# Symmetric GARCH news impact for comparison
nic_sym = omega + alpha * eps**2 + beta * sigma2_bar

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(eps, nic_gjr, color=CRIMSON, linewidth=1.2, label='GJR-GARCH(1,1)')
ax.plot(eps, nic_sym, color=MAIN_BLUE, linewidth=1.0, linestyle='--',
        label='Symmetric GARCH(1,1)')
ax.axvline(x=0, color='gray', linewidth=0.4, linestyle=':')

# Annotate asymmetry
neg_idx = np.argmin(np.abs(eps - (-2.5)))
pos_idx = np.argmin(np.abs(eps - 2.5))
ax.annotate('Bad news\n($\\alpha + \\gamma$)',
            xy=(eps[neg_idx], nic_gjr[neg_idx]),
            xytext=(-3.5, nic_gjr[neg_idx] * 0.85),
            fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))
ax.annotate('Good news\n($\\alpha$ only)',
            xy=(eps[pos_idx], nic_gjr[pos_idx]),
            xytext=(2.5, nic_gjr[pos_idx] * 1.3),
            fontsize=7, color=FOREST,
            arrowprops=dict(arrowstyle='->', color=FOREST, lw=0.7))

ax.set_xlabel('Shock $\\epsilon_{t-1}$ (\\%)')
ax.set_ylabel('$\\sigma_t^2$ (conditional variance)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
          frameon=False, fontsize=7)

plt.tight_layout()
save_fig('sfm_ch1_leverage')

print("\n" + "=" * 70)
print("LEVERAGE EFFECT CHART COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_leverage.pdf/.png")
