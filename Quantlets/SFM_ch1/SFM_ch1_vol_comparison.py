"""
SFM_ch1_vol_comparison
======================
Volatility Estimators Comparison, Conditional vs Unconditional,
Volatility Signature Plot, Square Root of Time Rule

Description:
- Download AAPL OHLC data via yfinance
- Compute CC, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang estimators
- Bar chart comparing all estimators on same data
- GARCH conditional vs rolling historical volatility
- Realized volatility signature plot
- Square root of time scaling visualization

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
print("SFM CHAPTER 1: VOLATILITY COMPARISON CHARTS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING AAPL DATA")
print("-" * 40)

data = yf.download('AAPL', start='2020-01-01', end='2024-12-31', progress=False)
open_p = data['Open'].squeeze()
high_p = data['High'].squeeze()
low_p = data['Low'].squeeze()
close_p = data['Close'].squeeze()

log_ret = np.log(close_p / close_p.shift(1)).dropna()

print(f"   Period: {close_p.index[0].strftime('%Y-%m-%d')} to {close_p.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(log_ret)}")

# =============================================================================
# 2. Compute Volatility Estimators (rolling 30-day)
# =============================================================================
print("\n2. COMPUTING VOLATILITY ESTIMATORS")
print("-" * 40)

W = 30  # rolling window

# Close-to-Close (CC)
cc_var = log_ret.rolling(W).var()
cc_vol = np.sqrt(cc_var * 252)

# Parkinson
hl = np.log(high_p / low_p)
park_var = hl**2 / (4 * np.log(2))
park_vol = np.sqrt(park_var.rolling(W).mean() * 252)

# Garman-Klass
u = np.log(high_p / open_p)
d = np.log(low_p / open_p)
c = np.log(close_p / open_p)
gk_var = 0.5 * (u - d)**2 - (2 * np.log(2) - 1) * c**2
gk_vol = np.sqrt(gk_var.rolling(W).mean() * 252)

# Rogers-Satchell: σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
rs_var = np.log(high_p / close_p) * u + np.log(low_p / close_p) * d
rs_vol = np.sqrt(rs_var.rolling(W).mean() * 252)

# Yang-Zhang
n = W
oc_ret = np.log(open_p / close_p.shift(1)).dropna()
co_ret = np.log(close_p / open_p)
alpha = 1.34

# Rolling Yang-Zhang
def yang_zhang_rolling(oc, co, rs_v, n, alpha):
    """Compute rolling Yang-Zhang volatility."""
    yz_vol = pd.Series(index=rs_v.index, dtype=float)
    for i in range(n - 1, len(rs_v)):
        idx = rs_v.index[i-n+1:i+1]
        oc_w = oc.reindex(idx).dropna()
        co_w = co.reindex(idx).dropna()
        rs_w = rs_v.reindex(idx).dropna()
        if len(oc_w) < n // 2:
            continue
        k = (alpha - 1) / (alpha + (n + 1) / (n - 1))
        sigma_o = oc_w.var()
        sigma_c = co_w.var()
        sigma_rs = rs_w.mean()
        yz_var = sigma_o + k * sigma_c + (1 - k) * sigma_rs
        yz_vol.iloc[i] = np.sqrt(max(yz_var, 0) * 252)
    return yz_vol

yz_vol = yang_zhang_rolling(oc_ret, co_ret, rs_var, W, alpha)

# Full-sample estimates for bar chart
cc_full = np.sqrt(log_ret.var() * 252)
park_full = np.sqrt(park_var.mean() * 252)
gk_full = np.sqrt(gk_var.mean() * 252)
rs_full = np.sqrt(rs_var.mean() * 252)

oc_full_var = oc_ret.var()
co_full_var = co_ret.var()
rs_full_mean = rs_var.mean()
n_full = len(log_ret)
k_full = (alpha - 1) / (alpha + (n_full + 1) / (n_full - 1))
yz_full = np.sqrt((oc_full_var + k_full * co_full_var +
                    (1 - k_full) * rs_full_mean) * 252)

print(f"   CC:  {cc_full:.4f}")
print(f"   P:   {park_full:.4f}")
print(f"   GK:  {gk_full:.4f}")
print(f"   RS:  {rs_full:.4f}")
print(f"   YZ:  {yz_full:.4f}")

# =============================================================================
# 3. Bar Chart: All Estimators Comparison
# =============================================================================
print("\n3. CREATING ESTIMATOR COMPARISON BAR CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

names = ['Close-to-\nClose', 'Parkinson', 'Garman-\nKlass',
         'Rogers-\nSatchell', 'Yang-\nZhang']
values = [cc_full * 100, park_full * 100, gk_full * 100,
          rs_full * 100, yz_full * 100]
colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER, ORANGE]

bars = ax.bar(names, values, color=colors, width=0.6, edgecolor='white',
              linewidth=0.5)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}\\%', ha='center', va='bottom', fontsize=7,
            fontweight='bold')

ax.set_ylabel('Annualized Volatility (\\%)')
ax.set_ylim(0, max(values) * 1.15)

plt.tight_layout()
save_fig('sfm_ch1_vol_comparison')

# =============================================================================
# 4. Conditional vs Unconditional Volatility
# =============================================================================
print("\n4. CREATING CONDITIONAL VS UNCONDITIONAL CHART")
print("-" * 40)

# Fit GARCH(1,1) for conditional volatility
ret_pct = log_ret * 100
garch = arch_model(ret_pct, vol='GARCH', p=1, q=1, dist='normal')
garch_res = garch.fit(disp='off')
cond_vol = garch_res.conditional_volatility / 100 * np.sqrt(252)  # annualized

# Rolling 30-day historical vol (unconditional proxy)
hist_vol = log_ret.rolling(30).std() * np.sqrt(252)

# Unconditional (constant)
uncond = np.sqrt(log_ret.var() * 252)

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(cond_vol.index, cond_vol.values, color=CRIMSON, linewidth=0.6,
        alpha=0.8, label='GARCH(1,1) conditional $\\sigma_t$')
ax.plot(hist_vol.index, hist_vol.values, color=MAIN_BLUE, linewidth=0.6,
        alpha=0.6, label=f'30-day rolling historical')
ax.axhline(y=uncond, color=FOREST, linewidth=1.0, linestyle='--',
           label=f'Unconditional $\\sigma$ = {uncond:.1%}')

ax.set_ylabel('Annualized Volatility')
ax.set_xlabel('')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=False, fontsize=6.5)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
save_fig('sfm_ch1_cond_uncond')

# =============================================================================
# 5. Volatility Signature Plot
# =============================================================================
print("\n5. CREATING VOLATILITY SIGNATURE PLOT")
print("-" * 40)

# Use 1-minute data for SPY if available, otherwise simulate with different
# sampling frequencies from daily data
# Since we don't have intraday data, we'll simulate the concept using
# different aggregation levels of daily data

# Download higher-frequency data: 1h for SPY
spy_1h = yf.download('SPY', start='2024-01-01', end='2024-12-31',
                       interval='1h', progress=False)
if len(spy_1h) > 100:
    spy_close_1h = spy_1h['Close'].squeeze()
    # Compute RV at different sampling frequencies (in bars)
    freqs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 65]
    freq_labels = ['1h', '2h', '3h', '5h', '7h', '10h', '15h', '20h',
                   '30h', '50h', '1w']
    rv_list = []
    for f in freqs:
        sampled = spy_close_1h.iloc[::f]
        lr = np.log(sampled / sampled.shift(1)).dropna()
        # Annualize: ~6.5 trading hours/day, 252 days
        bars_per_year = (252 * 6.5) / f
        rv = np.sqrt(lr.var() * bars_per_year)
        rv_list.append(rv * 100)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, rv_list, 'o-', color=MAIN_BLUE, linewidth=1.0,
            markersize=4)
    ax.axhline(y=rv_list[-1], color=FOREST, linewidth=0.8, linestyle='--',
               alpha=0.6, label='Low-frequency estimate')

    ax.set_xlabel('Sampling Frequency (hours)')
    ax.set_ylabel('Annualized Volatility (\\%)')

    # Annotate microstructure noise region
    ax.annotate('Microstructure\nnoise bias',
                xy=(1, rv_list[0]), xytext=(5, rv_list[0] * 1.05),
                fontsize=7, color=CRIMSON,
                arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1,
              frameon=False, fontsize=7)

    plt.tight_layout()
    save_fig('sfm_ch1_vol_signature')
else:
    # Fallback: simulate the signature plot shape
    print("   Using simulated signature plot (no intraday data available)")
    freqs_min = np.array([1, 2, 5, 10, 15, 30, 60, 120, 300, 390])
    # Typical pattern: high at 1min due to microstructure, decays, then flat
    true_vol = 18.0  # 18% annualized
    noise = 0.015  # microstructure noise variance per trade
    n_per_day = 390 / freqs_min  # observations per day
    # RV_m = sigma^2 + 2*noise/delta (simplified Bandi-Russell)
    delta = freqs_min / 390
    rv_est = np.sqrt(true_vol**2 / 10000 + 2 * noise * n_per_day / 252) * 100 * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs_min, rv_est, 'o-', color=MAIN_BLUE, linewidth=1.0,
            markersize=4)
    ax.axhline(y=true_vol, color=FOREST, linewidth=0.8, linestyle='--',
               alpha=0.6, label=f'True $\\sigma$ = {true_vol}\\%')

    ax.set_xlabel('Sampling Frequency (minutes)')
    ax.set_ylabel('Realized Volatility (\\%)')

    ax.annotate('Microstructure\nnoise bias',
                xy=(1, rv_est[0]), xytext=(20, rv_est[0] * 0.98),
                fontsize=7, color=CRIMSON,
                arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1,
              frameon=False, fontsize=7)

    plt.tight_layout()
    save_fig('sfm_ch1_vol_signature')

# =============================================================================
# 6. Square Root of Time Rule
# =============================================================================
print("\n6. CREATING SQUARE ROOT OF TIME CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

daily_vol = 1.2  # typical daily vol in %
horizons = np.arange(1, 253)

# Actual scaling under sqrt rule
scaled_vol = daily_vol * np.sqrt(horizons)

# Linear (incorrect) scaling: σ_annual / 252 * T  (same endpoints, wrong shape)
linear_vol = daily_vol * np.sqrt(252) / 252 * horizons

ax.plot(horizons, scaled_vol, color=MAIN_BLUE, linewidth=1.2,
        label='$\\sigma_{\\mathrm{daily}} \\times \\sqrt{T}$ (correct)')
ax.plot(horizons, linear_vol, color=CRIMSON, linewidth=0.8, linestyle='--',
        label='Linear scaling (incorrect)', alpha=0.7)

# Mark key horizons
key_horizons = [(1, 'Daily'), (5, 'Weekly'), (21, 'Monthly'),
                (63, 'Quarterly'), (252, 'Annual')]
for h, label in key_horizons:
    vol_h = daily_vol * np.sqrt(h)
    ax.plot(h, vol_h, 'o', color=FOREST, markersize=4, zorder=5)
    offset = (5, 3) if h < 100 else (-5, 3)
    ax.annotate(f'{label}\n{vol_h:.1f}\\%',
                xy=(h, vol_h), xytext=(h + offset[0], vol_h + offset[1]),
                fontsize=6.5, color=FOREST,
                arrowprops=dict(arrowstyle='->', color=FOREST, lw=0.5))

ax.set_xlabel('Time Horizon $T$ (trading days)')
ax.set_ylabel('Volatility (\\%)')
ax.set_xlim(0, 260)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
          frameon=False, fontsize=7)

plt.tight_layout()
save_fig('sfm_ch1_sqrt_time')

print("\n" + "=" * 70)
print("VOLATILITY COMPARISON CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_vol_comparison.pdf/.png")
print("  - sfm_ch1_cond_uncond.pdf/.png")
print("  - sfm_ch1_vol_signature.pdf/.png")
print("  - sfm_ch1_sqrt_time.pdf/.png")
