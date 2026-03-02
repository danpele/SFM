"""
SFM_ch4_random_walk
===================
Random Walk Models and Unit Root Tests

Description:
- Simulate RW1 (iid increments), RW2 (independent increments),
  RW3 (uncorrelated increments)
- Plot sample paths for each type
- Compare simulated random walks with real stock data
- Perform ADF and KPSS unit root tests
- Demonstrate variance grows linearly with time

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
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

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("SFM CHAPTER 4: RANDOM WALK MODELS")
print("=" * 70)

# =============================================================================
# 1. Simulate Random Walk Type 1 (RW1)
# =============================================================================
print("\n1. SIMULATING RANDOM WALK TYPE 1 (RW1)")
print("-" * 40)
print("   RW1: P_t = P_{t-1} + epsilon_t")
print("   epsilon_t ~ iid N(0, sigma^2)")

np.random.seed(42)
T = 1000

# RW1: iid Normal increments (no drift, constant variance)
rw1_eps = np.random.normal(0, 1, T)
rw1 = np.cumsum(rw1_eps) + 100

print(f"   Simulated {T} observations")
print(f"   Start={rw1[0]:.1f}, End={rw1[-1]:.1f}")

# =============================================================================
# 2. Simulate Random Walk Type 2 (RW2)
# =============================================================================
print("\n2. SIMULATING RANDOM WALK TYPE 2 (RW2)")
print("-" * 40)
print("   RW2: P_t = mu + P_{t-1} + epsilon_t")
print("   epsilon_t ~ independent but NOT identically distributed")

mu = 0.05
rw2_eps = np.random.normal(0, 1, T)
# Time-varying variance (regime switching)
sigma_t = np.ones(T) * 0.8
sigma_t[300:500] = 2.0   # high-volatility regime
sigma_t[700:800] = 1.5   # medium-volatility regime
rw2 = 100 + np.cumsum(mu + rw2_eps * sigma_t)

print(f"   Drift mu = {mu}")
print(f"   Start={rw2[0]:.1f}, End={rw2[-1]:.1f}")

# =============================================================================
# 3. Simulate Random Walk Type 3 (RW3)
# =============================================================================
print("\n3. SIMULATING RANDOM WALK TYPE 3 (RW3)")
print("-" * 40)
print("   RW3: P_t = P_{t-1} + epsilon_t")
print("   epsilon_t ~ uncorrelated (GARCH effects)")

omega, alpha, beta = 0.01, 0.1, 0.85
eps = np.zeros(T)
h = np.zeros(T)
h[0] = omega / (1 - alpha - beta)
for t in range(1, T):
    h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    eps[t] = np.sqrt(h[t]) * np.random.standard_normal()
rw3 = np.cumsum(eps) + 100

print(f"   GARCH(1,1): omega={omega}, alpha={alpha}, beta={beta}")
print(f"   Start={rw3[0]:.1f}, End={rw3[-1]:.1f}")

# =============================================================================
# 4. Download Real Stock Data
# =============================================================================
print("\n4. DOWNLOADING REAL STOCK DATA")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2020-01-01', end='2024-12-31',
                    progress=False)
close = data['Close'].squeeze()
log_returns = np.log(close / close.shift(1)).dropna()

print(f"   Ticker: {ticker}")
print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to "
      f"{close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(close)}")

# =============================================================================
# 5. Unit Root Tests
# =============================================================================
print("\n5. UNIT ROOT TESTS")
print("-" * 40)

# ADF test on price levels
print("   Augmented Dickey-Fuller Test:")
print("   H0: Unit root (non-stationary)")
print("   H1: Stationary\n")

adf_price = adfuller(close.dropna(), maxlag=20, autolag='AIC')
print(f"   Price Level ({ticker}):")
print(f"     ADF statistic: {adf_price[0]:.4f}")
print(f"     p-value:       {adf_price[1]:.6f}")
print(f"     Lags used:     {adf_price[2]}")
print(f"     Critical values: 1%={adf_price[4]['1%']:.3f}, "
      f"5%={adf_price[4]['5%']:.3f}, 10%={adf_price[4]['10%']:.3f}")
print(f"     Conclusion:    "
      f"{'Reject H0 (stationary)' if adf_price[1] < 0.05 else 'Cannot reject H0 (unit root)'}")

# ADF test on returns
adf_ret = adfuller(log_returns.dropna(), maxlag=20, autolag='AIC')
print(f"\n   Log Returns ({ticker}):")
print(f"     ADF statistic: {adf_ret[0]:.4f}")
print(f"     p-value:       {adf_ret[1]:.6f}")
print(f"     Conclusion:    "
      f"{'Reject H0 (stationary)' if adf_ret[1] < 0.05 else 'Cannot reject H0 (unit root)'}")

# KPSS test
print(f"\n   KPSS Test (H0: stationarity):")
kpss_price = kpss(close.dropna(), regression='ct', nlags='auto')
print(f"   Price: stat={kpss_price[0]:.4f}, p={kpss_price[1]:.4f}")
kpss_ret = kpss(log_returns.dropna(), regression='c', nlags='auto')
print(f"   Returns: stat={kpss_ret[0]:.4f}, p={kpss_ret[1]:.4f}")

# ADF test on simulated RW1
adf_rw1 = adfuller(rw1, maxlag=20, autolag='AIC')
print(f"\n   Simulated RW1:")
print(f"     ADF statistic: {adf_rw1[0]:.4f}")
print(f"     p-value:       {adf_rw1[1]:.6f}")
print(f"     Conclusion:    "
      f"{'Reject H0 (stationary)' if adf_rw1[1] < 0.05 else 'Cannot reject H0 (unit root)'}")

# =============================================================================
# 6. FIGURE: Random Walk Models (4-panel)
# =============================================================================
print("\n6. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: RW1 - iid increments
axes[0, 0].plot(rw1, color='#1A3A6E', linewidth=0.8)
axes[0, 0].axhline(y=100, color='gray', linestyle='--', linewidth=0.8)
axes[0, 0].set_title('RW1: iid Increments', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Price Level')
axes[0, 0].text(0.05, 0.95, r'$\varepsilon_t \sim$ iid $N(0,1)$',
               transform=axes[0, 0].transAxes, ha='left', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: RW2 - independent, non-identical increments
axes[0, 1].plot(rw2, color='#DC3545', linewidth=0.8)
axes[0, 1].axhline(y=100, color='gray', linestyle='--', linewidth=0.8)
axes[0, 1].axvspan(300, 500, alpha=0.1, color='#DC3545',
                    label='High vol regime')
axes[0, 1].axvspan(700, 800, alpha=0.1, color='#FF8C00',
                    label='Med vol regime')
axes[0, 1].set_title('RW2: Independent, Non-identical Increments',
                       fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Price Level')
axes[0, 1].legend(loc='upper left', fontsize=7)

# Panel C: RW3 - uncorrelated (GARCH) increments
axes[1, 0].plot(rw3, color='#2E7D32', linewidth=0.8)
axes[1, 0].axhline(y=100, color='gray', linestyle='--', linewidth=0.8)
axes[1, 0].set_title('RW3: Uncorrelated Increments (GARCH)',
                       fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Price Level')
axes[1, 0].text(0.05, 0.95, 'Uncorrelated but\nNOT independent',
               transform=axes[1, 0].transAxes, ha='left', va='top',
               fontsize=9, color='#DC3545',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel D: Real stock data vs simulated RW
spy_norm = (close.values / close.values[0]) * 100
np.random.seed(123)
rw_sim = np.cumsum(np.random.normal(log_returns.mean(),
                                     log_returns.std(),
                                     len(close) - 1))
rw_price = 100 * np.exp(np.concatenate([[0], rw_sim]))

axes[1, 1].plot(close.index, spy_norm, color='#1A3A6E', linewidth=0.8,
                label=f'{ticker} (actual)')
axes[1, 1].plot(close.index, rw_price[:len(close)], color='#DC3545',
                linewidth=0.8, alpha=0.7,
                label='Random Walk (simulated)')
axes[1, 1].set_title(f'{ticker} vs Random Walk '
                       f'(ADF p={adf_price[1]:.4f})', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Normalized Price')
axes[1, 1].legend(loc='upper left', fontsize=7)

plt.tight_layout()
save_fig('ch4_random_walk')

# =============================================================================
# 7. Summary
# =============================================================================
print("\n7. RANDOM WALK HIERARCHY")
print("-" * 40)

print("   RW1 is a subset of RW2 is a subset of RW3")
print("   RW1: iid increments (strongest form)")
print("   RW2: independent but heteroskedastic increments")
print("   RW3: uncorrelated increments (weakest form)")
print("\n   Implications for EMH:")
print("   RW1 => Strong-form EMH")
print("   RW2 => Semi-strong-form EMH")
print("   RW3 => Weak-form EMH")

print("\n" + "=" * 70)
print("RANDOM WALK ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch4_random_walk.pdf: 4-panel random walk comparison")
