"""
SFM_sem5_volatility
===================
Seminar 5: Volatility Estimators --- Charts for Beamer presentation

Description:
- Download AAPL OHLC data via yfinance
- Compute 5 volatility estimators (rolling 30-day)
- Generate charts: rolling comparison, bar chart, sqrt(T) rule,
  efficiency heatmap, overnight gap impact
- All charts Beamer-ready (7x3 inches, transparent background)

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# Color palette (matches Beamer template)
MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E67E22'
PURPLE    = '#8E44AD'

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


# =============================================================================
# Volatility Estimator Functions
# =============================================================================

def historical_volatility(close, window=30):
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window).std() * np.sqrt(252)

def parkinson_volatility(high, low, window=30):
    hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    return np.sqrt(factor * hl.rolling(window=window).mean() * 252)

def garman_klass_volatility(open_, high, low, close, window=30):
    hl = 0.5 * np.log(high / low) ** 2
    co = -(2.0 * np.log(2) - 1.0) * np.log(close / open_) ** 2
    return np.sqrt((hl + co).rolling(window=window).mean() * 252)

def rogers_satchell_volatility(open_, high, low, close, window=30):
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_))
    return np.sqrt(rs.rolling(window=window).mean() * 252)

def yang_zhang_volatility(open_, high, low, close, window=30):
    log_oc = np.log(close / open_)
    log_co = np.log(open_ / close.shift(1))
    sigma2_o = log_co.rolling(window=window).var()
    sigma2_c = log_oc.rolling(window=window).var()
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_))
    sigma2_rs = rs.rolling(window=window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    sigma2_yz = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    return np.sqrt(sigma2_yz.clip(lower=0) * 252)


print("=" * 70)
print("SEMINAR 5: VOLATILITY ESTIMATORS --- CHART GENERATION")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = ['AAPL', 'SPY', 'BTC-USD']
data_all = {}
for t in tickers:
    d = yf.download(t, start='2020-01-01', end='2025-12-31', progress=False)
    d.columns = d.columns.get_level_values(0)
    data_all[t] = d
    print(f"   {t}: {len(d)} obs ({d.index[0].strftime('%Y-%m-%d')} to "
          f"{d.index[-1].strftime('%Y-%m-%d')})")

data = data_all['AAPL']
W = 30  # rolling window

# =============================================================================
# 2. Compute estimators for AAPL
# =============================================================================
print("\n2. COMPUTING VOLATILITY ESTIMATORS (AAPL, 30-day)")
print("-" * 40)

vol_cc   = historical_volatility(data['Close'], W)
vol_park = parkinson_volatility(data['High'], data['Low'], W)
vol_gk   = garman_klass_volatility(data['Open'], data['High'],
                                     data['Low'], data['Close'], W)
vol_rs   = rogers_satchell_volatility(data['Open'], data['High'],
                                        data['Low'], data['Close'], W)
vol_yz   = yang_zhang_volatility(data['Open'], data['High'],
                                   data['Low'], data['Close'], W)

for name, vol in [('Close-to-Close', vol_cc), ('Parkinson', vol_park),
                   ('Garman-Klass', vol_gk), ('Rogers-Satchell', vol_rs),
                   ('Yang-Zhang', vol_yz)]:
    v = vol.dropna()
    print(f"   {name:<20} mean={v.mean():.4f}  std={v.std():.4f}")

# =============================================================================
# 3. CHART 1: Rolling volatility comparison (2 panels: price + vol)
# =============================================================================
print("\n3. CHART 1: Rolling volatility time series")
print("-" * 40)

fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 2]})

# Panel A: Price
axes[0].plot(data.index, data['Close'], color=MAIN_BLUE, linewidth=0.8)
axes[0].set_ylabel('Price ($)')
axes[0].set_title('AAPL', fontweight='bold', fontsize=9)

# Panel B: All estimators
colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER, PURPLE]
names = ['Close-to-Close', 'Parkinson', 'Garman-Klass',
         'Rogers-Satchell', 'Yang-Zhang']
vols = [vol_cc, vol_park, vol_gk, vol_rs, vol_yz]
lws = [1.2, 0.7, 0.7, 0.7, 1.0]

for vol, name, color, lw in zip(vols, names, colors, lws):
    axes[1].plot(vol.index, vol * 100, color=color, linewidth=lw,
                 alpha=0.85, label=name)

axes[1].set_ylabel('Annualized Volatility (%)')
axes[1].set_xlabel('')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=5, frameon=False, fontsize=6)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
save_fig('sfm_sem5_vol_rolling')

# =============================================================================
# 4. CHART 2: Bar chart --- full-sample estimators
# =============================================================================
print("\n4. CHART 2: Estimator comparison bar chart")
print("-" * 40)

log_ret = np.log(data['Close'] / data['Close'].shift(1)).dropna()

cc_full = np.sqrt(log_ret.var() * 252)
park_var = (np.log(data['High'] / data['Low']) ** 2) / (4 * np.log(2))
park_full = np.sqrt(park_var.mean() * 252)
u = np.log(data['High'] / data['Open'])
d = np.log(data['Low'] / data['Open'])
c = np.log(data['Close'] / data['Open'])
gk_var = 0.5 * np.log(data['High'] / data['Low'])**2 - (2*np.log(2)-1) * c**2
gk_full = np.sqrt(gk_var.mean() * 252)
rs_var = np.log(data['High']/data['Close'])*u + np.log(data['Low']/data['Close'])*d
rs_full = np.sqrt(rs_var.mean() * 252)

oc_ret = np.log(data['Open'] / data['Close'].shift(1)).dropna()
co_ret = np.log(data['Close'] / data['Open'])
n_f = len(log_ret)
alpha = 1.34
k_f = (alpha - 1) / (alpha + (n_f + 1) / (n_f - 1))
yz_full = np.sqrt((oc_ret.var() + k_f * co_ret.var() +
                    (1 - k_f) * rs_var.mean()) * 252)

fig, ax = plt.subplots(figsize=(7, 3))
bar_names = ['Close-to-\nClose', 'Parkinson', 'Garman-\nKlass',
             'Rogers-\nSatchell', 'Yang-\nZhang']
bar_vals = [cc_full*100, park_full*100, gk_full*100, rs_full*100, yz_full*100]

bars = ax.bar(bar_names, bar_vals, color=colors, width=0.6,
              edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=7,
            fontweight='bold')
ax.set_ylabel('Annualized Volatility (%)')
ax.set_ylim(0, max(bar_vals) * 1.18)
ax.set_title('AAPL Full-Sample Volatility Estimates', fontweight='bold',
             fontsize=9)
plt.tight_layout()
save_fig('sfm_sem5_vol_barplot')

# =============================================================================
# 5. CHART 3: Square root of time rule
# =============================================================================
print("\n5. CHART 3: Square root of time rule")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

daily_vol = 1.5  # typical daily vol in %
horizons = np.arange(1, 253)
sqrt_vol = daily_vol * np.sqrt(horizons)
linear_vol = daily_vol * np.sqrt(252) / 252 * horizons

ax.plot(horizons, sqrt_vol, color=MAIN_BLUE, linewidth=1.2,
        label=r'$\sigma_{\mathrm{daily}} \times \sqrt{T}$ (correct)')
ax.plot(horizons, linear_vol, color=CRIMSON, linewidth=0.8, linestyle='--',
        label='Linear scaling (incorrect)', alpha=0.7)

key_h = [(1, 'Daily'), (5, 'Weekly'), (22, 'Monthly'),
         (63, 'Quarterly'), (252, 'Annual')]
for h, label in key_h:
    vol_h = daily_vol * np.sqrt(h)
    ax.plot(h, vol_h, 'o', color=FOREST, markersize=4, zorder=5)
    dx = 8 if h < 200 else -40
    ax.annotate(f'{label}\n{vol_h:.1f}%',
                xy=(h, vol_h), xytext=(h + dx, vol_h + 1.5),
                fontsize=6.5, color=FOREST,
                arrowprops=dict(arrowstyle='->', color=FOREST, lw=0.5))

ax.set_xlabel('Time Horizon $T$ (trading days)')
ax.set_ylabel('Volatility (%)')
ax.set_xlim(0, 260)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
          frameon=False, fontsize=7)
plt.tight_layout()
save_fig('sfm_sem5_sqrt_time')

# =============================================================================
# 6. CHART 4: Multi-asset volatility comparison (AAPL, SPY, BTC)
# =============================================================================
print("\n6. CHART 4: Multi-asset volatility comparison")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True)
asset_colors = [MAIN_BLUE, FOREST, ORANGE]

for ax, (ticker, d), clr in zip(axes, data_all.items(), asset_colors):
    v_cc = historical_volatility(d['Close'], 30)
    v_pk = parkinson_volatility(d['High'], d['Low'], 30)
    v_yz = yang_zhang_volatility(d['Open'], d['High'], d['Low'],
                                  d['Close'], 30)
    ax.plot(v_cc.index, v_cc*100, color=clr, linewidth=0.6, alpha=0.6,
            label='CC')
    ax.plot(v_yz.index, v_yz*100, color=CRIMSON, linewidth=0.6, alpha=0.8,
            label='YZ')
    ax.set_title(ticker, fontweight='bold', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(labelsize=6)
    if ax == axes[0]:
        ax.set_ylabel('Annualized Volatility (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fontsize=5, ncol=2, frameon=False)

plt.suptitle('Close-to-Close vs Yang-Zhang', fontweight='bold', fontsize=9,
             y=1.02)
plt.tight_layout()
save_fig('sfm_sem5_vol_multiasset')

# =============================================================================
# 7. CHART 5: Overnight gap impact
# =============================================================================
print("\n7. CHART 5: Overnight gaps visualization")
print("-" * 40)

gap = np.log(data['Open'] / data['Close'].shift(1)).dropna() * 100
intraday = np.log(data['Close'] / data['Open']) * 100

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

# Histogram of overnight gaps
axes[0].hist(gap, bins=80, color=MAIN_BLUE, alpha=0.7, edgecolor='white',
             linewidth=0.3, density=True)
axes[0].axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
axes[0].set_xlabel('Overnight Return (%)')
axes[0].set_ylabel('Density')
axes[0].set_title('Overnight Gaps (AAPL)', fontweight='bold', fontsize=8)
axes[0].annotate(f'Std = {gap.std():.3f}%\n'
                 f'Max = {gap.max():.2f}%\n'
                 f'Min = {gap.min():.2f}%',
                 xy=(0.98, 0.95), xycoords='axes fraction',
                 fontsize=6, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='gray', alpha=0.8))

# Scatter: overnight vs intraday
axes[1].scatter(gap, intraday.reindex(gap.index), s=2, alpha=0.3,
                color=MAIN_BLUE)
axes[1].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
axes[1].axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
axes[1].set_xlabel('Overnight Return (%)')
axes[1].set_ylabel('Intraday Return (%)')
axes[1].set_title('Overnight vs Intraday', fontweight='bold', fontsize=8)

# Correlation
corr = gap.corr(intraday.reindex(gap.index))
axes[1].annotate(f'Corr = {corr:.3f}',
                 xy=(0.98, 0.95), xycoords='axes fraction',
                 fontsize=6, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='gray', alpha=0.8))

plt.tight_layout()
save_fig('sfm_sem5_overnight_gaps')

# =============================================================================
# 8. CHART 6: Window sensitivity
# =============================================================================
print("\n8. CHART 6: Window sensitivity")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

for w, ls, alpha in [(10, '--', 0.6), (30, '-', 0.9), (60, '-.', 0.6)]:
    v = historical_volatility(data['Close'], w)
    ax.plot(v.index, v*100, color=MAIN_BLUE, linewidth=0.8,
            linestyle=ls, alpha=alpha, label=f'{w}-day')

ax.set_ylabel('Annualized Volatility (%)')
ax.set_title('Historical Volatility: Window Sensitivity (AAPL)',
             fontweight='bold', fontsize=9)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False, fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
save_fig('sfm_sem5_window_sensitivity')

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SEMINAR 5 CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_sem5_vol_rolling.pdf/.png")
print("  - sfm_sem5_vol_barplot.pdf/.png")
print("  - sfm_sem5_sqrt_time.pdf/.png")
print("  - sfm_sem5_vol_multiasset.pdf/.png")
print("  - sfm_sem5_overnight_gaps.pdf/.png")
print("  - sfm_sem5_window_sensitivity.pdf/.png")
