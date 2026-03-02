"""
SFM_ch1_volatility_estimators
=============================
Volatility Estimators: Historical, Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang

Description:
- Implement five volatility estimators as functions
- Download AAPL OHLC data via yfinance
- Compute 20-day rolling volatility for each estimator
- Plot comparison of all estimators

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
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
print("SFM CHAPTER 1: VOLATILITY ESTIMATORS")
print("=" * 70)

# =============================================================================
# 1. Define Volatility Estimator Functions
# =============================================================================
print("\n1. DEFINING VOLATILITY ESTIMATORS")
print("-" * 40)

def historical_volatility(close, window=20):
    """
    Close-to-close (historical) volatility estimator.
    sigma = std(log returns) * sqrt(252)
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window).std() * np.sqrt(252)

def parkinson_volatility(high, low, window=20):
    """
    Parkinson (1980) volatility estimator using high-low range.
    More efficient than close-to-close by factor of ~5.2.
    """
    hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    return np.sqrt(factor * hl.rolling(window=window).mean() * 252)

def garman_klass_volatility(open_, high, low, close, window=20):
    """
    Garman-Klass (1980) volatility estimator using OHLC data.
    More efficient than Parkinson; uses open and close prices.
    """
    hl = 0.5 * np.log(high / low) ** 2
    co = -(2.0 * np.log(2) - 1.0) * np.log(close / open_) ** 2
    gk = hl + co
    return np.sqrt(gk.rolling(window=window).mean() * 252)

def rogers_satchell_volatility(open_, high, low, close, window=20):
    """
    Rogers-Satchell (1991) volatility estimator.
    Handles non-zero drift (trending markets).
    """
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_))
    return np.sqrt(rs.rolling(window=window).mean() * 252)

def yang_zhang_volatility(open_, high, low, close, window=20):
    """
    Yang-Zhang (2000) volatility estimator.
    Combines overnight and Rogers-Satchell; handles drift and opening jumps.
    """
    log_oc = np.log(close / open_)
    log_co = np.log(open_ / close.shift(1))

    # Overnight volatility
    sigma2_o = log_co.rolling(window=window).var()

    # Close-to-open volatility
    sigma2_c = log_oc.rolling(window=window).var()

    # Rogers-Satchell variance
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_))
    sigma2_rs = rs.rolling(window=window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    sigma2_yz = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs

    return np.sqrt(sigma2_yz.clip(lower=0) * 252)

print("   Defined: Historical (Close-to-Close)")
print("   Defined: Parkinson (1980)")
print("   Defined: Garman-Klass (1980)")
print("   Defined: Rogers-Satchell (1991)")
print("   Defined: Yang-Zhang (2000)")

# =============================================================================
# 2. Download Data
# =============================================================================
print("\n2. DOWNLOADING DATA")
print("-" * 40)

ticker = 'AAPL'
data = yf.download(ticker, start='2018-01-01', end='2024-12-31', progress=False)
data.columns = data.columns.get_level_values(0)

print(f"   Ticker: {ticker}")
print(f"   Period: {data.index[0].strftime('%Y-%m-%d')} to "
      f"{data.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(data)}")

# =============================================================================
# 3. Compute Rolling Volatilities
# =============================================================================
print("\n3. COMPUTING 20-DAY ROLLING VOLATILITY")
print("-" * 40)

window = 20

vol_hist = historical_volatility(data['Close'], window)
vol_park = parkinson_volatility(data['High'], data['Low'], window)
vol_gk = garman_klass_volatility(data['Open'], data['High'], data['Low'],
                                  data['Close'], window)
vol_rs = rogers_satchell_volatility(data['Open'], data['High'], data['Low'],
                                     data['Close'], window)
vol_yz = yang_zhang_volatility(data['Open'], data['High'], data['Low'],
                                data['Close'], window)

# Summary statistics
print(f"   {'Estimator':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("   " + "-" * 52)
for name, vol in [('Historical', vol_hist), ('Parkinson', vol_park),
                   ('Garman-Klass', vol_gk), ('Rogers-Satchell', vol_rs),
                   ('Yang-Zhang', vol_yz)]:
    v = vol.dropna()
    print(f"   {name:<20} {v.mean():>8.4f} {v.std():>8.4f} "
          f"{v.min():>8.4f} {v.max():>8.4f}")

# =============================================================================
# 4. FIGURE: Volatility Estimator Comparison
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

colors = ['#1A3A6E', '#DC3545', '#2E7D32', '#FF8C00', '#6A0DAD']

fig, axes = plt.subplots(3, 1, figsize=(14, 7))
fig.suptitle('Ch.1: Volatility Estimators', fontweight='bold', fontsize=12)

# Panel A: Price
axes[0].plot(data.index, data['Close'], color='#1A3A6E', linewidth=0.8)
axes[0].set_title(f'{ticker} Close Price', fontweight='bold')
axes[0].set_ylabel('Price ($)')

# Panel B: All volatility estimators overlaid
for vol, name, color in zip(
    [vol_hist, vol_park, vol_gk, vol_rs, vol_yz],
    ['Historical', 'Parkinson', 'Garman-Klass', 'Rogers-Satchell',
     'Yang-Zhang'],
    colors):
    axes[1].plot(vol.index, vol * 100, color=color, linewidth=0.8,
                 alpha=0.85, label=name)

axes[1].set_title(f'Rolling {window}-day Annualized Volatility (%)',
                   fontweight='bold')
axes[1].set_ylabel('Volatility (%)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=5, frameon=False)

# Panel C: Spread between estimators
spread = (vol_yz - vol_hist).dropna() * 100
axes[2].fill_between(spread.index, 0, spread, where=spread >= 0,
                     alpha=0.4, color='#2E7D32', label='YZ > Hist')
axes[2].fill_between(spread.index, 0, spread, where=spread < 0,
                     alpha=0.4, color='#DC3545', label='YZ < Hist')
axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_title('Yang-Zhang minus Historical Volatility (pp)',
                   fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Difference (pp)')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig('ch1_volatility_estimators')

# Single-panel Beamer-friendly chart: all 5 estimators overlaid
fig, ax = plt.subplots(figsize=(7, 3))

for vol, name, color, lw in zip(
    [vol_hist, vol_park, vol_gk, vol_rs, vol_yz],
    ['Close-to-Close', 'Parkinson', 'Garman-Klass', 'Rogers-Satchell',
     'Yang-Zhang'],
    colors,
    [1.2, 0.7, 0.7, 0.7, 1.0]):
    ax.plot(vol.index, vol * 100, color=color, linewidth=lw,
            alpha=0.85, label=name)

ax.set_ylabel('Annualized Volatility (%)')
ax.set_xlabel('Date')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=5, frameon=False, fontsize=6.5)

save_fig('sfm_ch1_vol_estimators_ts')

# =============================================================================
# 5. Efficiency Comparison
# =============================================================================
print("\n5. EFFICIENCY COMPARISON")
print("-" * 40)

print("   Theoretical efficiency ratios (vs Historical):")
print("   Parkinson:       ~5.2x")
print("   Garman-Klass:    ~7.4x")
print("   Rogers-Satchell: ~8.0x (drift-independent)")
print("   Yang-Zhang:      ~14.0x (drift + jump independent)")

# Empirical correlation matrix
vol_df = pd.DataFrame({
    'Historical': vol_hist,
    'Parkinson': vol_park,
    'Garman-Klass': vol_gk,
    'Rogers-Satchell': vol_rs,
    'Yang-Zhang': vol_yz
}).dropna()

print(f"\n   Correlation matrix:")
corr = vol_df.corr()
for i, row in enumerate(corr.index):
    vals = ' '.join([f'{corr.iloc[i, j]:.3f}'
                     for j in range(len(corr.columns))])
    print(f"   {row:<20} {vals}")

print("\n" + "=" * 70)
print("VOLATILITY ESTIMATORS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch1_volatility_estimators.pdf: 3-panel volatility comparison")
