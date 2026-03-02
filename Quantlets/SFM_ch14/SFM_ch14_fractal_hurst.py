"""
SFM_ch14_fractal_hurst
======================
Fractal Market Hypothesis: Hurst Exponent and Long Memory

Description:
- R/S analysis and Detrended Fluctuation Analysis (DFA)
- Hurst exponent estimation for stocks, crypto, and forex
- Rolling Hurst exponent over time
- Multiscale volatility analysis
- Self-similarity of return distributions

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Standard chart style (Nature journal quality) ---
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

# Colors
MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Orange   = '#E67E22'
Purple   = '#8E44AD'


# =============================================================================
# R/S Analysis (Rescaled Range)
# =============================================================================
def rs_analysis(x, min_n=10, max_n=None):
    """
    Rescaled Range (R/S) analysis to estimate the Hurst exponent.

    For each block size n, the series is divided into non-overlapping blocks.
    Within each block:
      1. Compute mean m
      2. Cumulative deviations Y_k = sum_{i=1}^{k} (x_i - m)
      3. Range R = max(Y) - min(Y)
      4. Standard deviation S = std(block, ddof=0)
      5. R/S = R / S

    The Hurst exponent H is the slope of log(R/S) vs log(n).

    Parameters
    ----------
    x : array-like
        Time series data (e.g., log returns).
    min_n : int
        Minimum block size.
    max_n : int or None
        Maximum block size (default: T // 4).

    Returns
    -------
    H : float
        Estimated Hurst exponent.
    ns : ndarray
        Block sizes used.
    rs_vals : ndarray
        Average R/S value for each block size.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if max_n is None:
        max_n = T // 4

    # Logarithmically spaced block sizes
    ns = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), 40).astype(int))
    rs_vals = []

    for n in ns:
        n_blocks = T // n
        if n_blocks < 2:
            rs_vals.append(np.nan)
            continue
        rs_block = []
        for b in range(n_blocks):
            block = x[b * n : (b + 1) * n]
            m = block.mean()
            cumdev = np.cumsum(block - m)
            R = cumdev.max() - cumdev.min()
            S = block.std(ddof=0)
            if S > 1e-15:
                rs_block.append(R / S)
        if len(rs_block) >= 1:
            rs_vals.append(np.mean(rs_block))
        else:
            rs_vals.append(np.nan)

    ns = ns[:len(rs_vals)]
    rs_vals = np.array(rs_vals)
    valid = ~np.isnan(rs_vals) & (rs_vals > 0)
    ns = ns[valid]
    rs_vals = rs_vals[valid]

    # Log-log OLS regression: log(R/S) = H * log(n) + c
    log_n = np.log(ns)
    log_rs = np.log(rs_vals)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)

    return slope, ns, rs_vals


# =============================================================================
# Detrended Fluctuation Analysis (DFA)
# =============================================================================
def dfa(x, min_n=10, max_n=None, order=1):
    """
    Detrended Fluctuation Analysis (DFA) to estimate the scaling exponent.

    Steps:
      1. Compute the profile (cumulative sum of demeaned series):
         Y(k) = sum_{i=1}^{k} (x_i - mean(x))
      2. Divide Y into non-overlapping windows of size n.
      3. In each window, fit a polynomial of given order (linear for DFA-1).
      4. Compute the variance of residuals in each window.
      5. The fluctuation function:
         F(n) = sqrt( (1/N_windows) * sum of variances )
      6. Repeat for different n, then regress log(F(n)) on log(n).

    For stationary data, the DFA exponent alpha relates to H as alpha ~ H.
    alpha = 0.5 => uncorrelated (random walk of cumulated series)
    alpha > 0.5 => persistent long-range correlations
    alpha < 0.5 => anti-persistent

    Parameters
    ----------
    x : array-like
        Time series data.
    min_n : int
        Minimum window size.
    max_n : int or None
        Maximum window size (default: T // 4).
    order : int
        Polynomial detrending order (1 = linear DFA-1).

    Returns
    -------
    alpha : float
        DFA scaling exponent.
    ns : ndarray
        Window sizes used.
    fluct : ndarray
        Fluctuation function F(n) values.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if max_n is None:
        max_n = T // 4

    # Step 1: Compute the profile (cumulative sum of demeaned series)
    profile = np.cumsum(x - x.mean())

    # Logarithmically spaced window sizes
    ns = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), 40).astype(int))
    fluct = []

    for n in ns:
        n_blocks = T // n
        if n_blocks < 2:
            fluct.append(np.nan)
            continue

        var_list = []

        # Forward pass: non-overlapping windows from the start
        for b in range(n_blocks):
            segment = profile[b * n : (b + 1) * n]
            t = np.arange(n)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            var_list.append(np.mean((segment - trend) ** 2))

        # Backward pass: non-overlapping windows from the end
        # (captures data that may be missed by integer division)
        for b in range(n_blocks):
            segment = profile[T - (b + 1) * n : T - b * n]
            t = np.arange(n)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            var_list.append(np.mean((segment - trend) ** 2))

        # F(n) = sqrt(mean of all window variances)
        fluct.append(np.sqrt(np.mean(var_list)))

    ns = ns[:len(fluct)]
    fluct = np.array(fluct)
    valid = ~np.isnan(fluct) & (fluct > 0)
    ns = ns[valid]
    fluct = fluct[valid]

    # Log-log OLS regression: log(F(n)) = alpha * log(n) + c
    log_n = np.log(ns)
    log_f = np.log(fluct)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_f)

    return slope, ns, fluct


# #############################################################################
# MAIN SCRIPT
# #############################################################################

print("=" * 70)
print("SFM CHAPTER 14: FRACTAL MARKET HYPOTHESIS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = {'SPY': 'SPY', 'BTC-USD': 'BTC-USD', 'EUR=X': 'EUR=X'}
labels  = {'SPY': 'SPY',  'BTC-USD': 'BTC',     'EUR=X': 'EUR/USD'}
asset_colors = {'SPY': MainBlue, 'BTC-USD': Orange, 'EUR=X': Forest}

prices_raw = yf.download(list(tickers.keys()), start='2015-01-01',
                         end='2025-12-31', progress=False)['Close']

returns = {}
for t in tickers:
    col = prices_raw[t].dropna()
    r = np.log(col / col.shift(1)).dropna()
    # Remove any inf/nan
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    returns[t] = r
    print(f"   {labels[t]:>8}: {len(r)} observations "
          f"({r.index[0].strftime('%Y-%m-%d')} to "
          f"{r.index[-1].strftime('%Y-%m-%d')})")

# =============================================================================
# 2. R/S Analysis
# =============================================================================
print("\n2. R/S ANALYSIS (Hurst Exponent)")
print("-" * 40)

print(f"   {'Asset':>8} {'H (R/S)':>10} {'Interpretation':>20}")
print("   " + "-" * 42)

rs_results = {}
for t in tickers:
    H, ns, rs_vals = rs_analysis(returns[t].values)
    rs_results[t] = (H, ns, rs_vals)
    if H > 0.55:
        interp = "Persistent"
    elif H < 0.45:
        interp = "Anti-persistent"
    else:
        interp = "Near random walk"
    print(f"   {labels[t]:>8} {H:>10.4f} {interp:>20}")

# =============================================================================
# 3. DFA Analysis
# =============================================================================
print("\n3. DETRENDED FLUCTUATION ANALYSIS")
print("-" * 40)

print(f"   {'Asset':>8} {'alpha(DFA)':>10} {'Interpretation':>20}")
print("   " + "-" * 42)

dfa_results = {}
for t in tickers:
    alpha, ns_d, fluct = dfa(returns[t].values)
    dfa_results[t] = (alpha, ns_d, fluct)
    if alpha > 0.55:
        interp = "Persistent"
    elif alpha < 0.45:
        interp = "Anti-persistent"
    else:
        interp = "Near random walk"
    print(f"   {labels[t]:>8} {alpha:>10.4f} {interp:>20}")

# =============================================================================
# 4. Rolling Hurst Exponent for SPY
# =============================================================================
print("\n4. COMPUTING ROLLING HURST EXPONENT (SPY)")
print("-" * 40)

window_h = 252  # 1-year rolling window
spy_ret = returns['SPY'].values
spy_dates = returns['SPY'].index
rolling_h = []

print(f"   Window: {window_h} trading days")
print(f"   Computing... ", end='', flush=True)

for i in range(window_h, len(spy_ret)):
    h_i, _, _ = rs_analysis(spy_ret[i - window_h : i],
                            min_n=10, max_n=window_h // 4)
    rolling_h.append(h_i)

rolling_dates = spy_dates[window_h:]
rolling_h = np.array(rolling_h)
print(f"done ({len(rolling_h)} estimates)")

# =============================================================================
# 5. Multiscale Volatility for SPY
# =============================================================================
print("\n5. MULTISCALE VOLATILITY ANALYSIS (SPY)")
print("-" * 40)

agg_horizons = [1, 5, 10, 20, 60, 120]
spy_daily = returns['SPY']
ann_vol_empirical = []

for h in agg_horizons:
    agg_ret = spy_daily.rolling(h).sum().dropna()
    # Annualized volatility: std(aggregated returns) * sqrt(252 / h)
    vol_h = agg_ret.std() * np.sqrt(252.0 / h)
    ann_vol_empirical.append(vol_h)

ann_vol_empirical = np.array(ann_vol_empirical)
# Under random walk: vol scales as sqrt(h), so annualized vol is constant
# Theoretical: vol_theoretical(h) = vol_daily * sqrt(h) * sqrt(252/h) = vol_daily * sqrt(252)
vol_daily = spy_daily.std()
ann_vol_rw = np.full(len(agg_horizons), vol_daily * np.sqrt(252.0))

print(f"   {'Horizon':>8} {'Ann.Vol(emp)':>14} {'Ann.Vol(RW)':>14} {'Ratio':>8}")
print("   " + "-" * 48)
for h, v_e, v_rw in zip(agg_horizons, ann_vol_empirical, ann_vol_rw):
    print(f"   {h:>8}d {v_e:>14.4f} {v_rw:>14.4f} {v_e/v_rw:>8.4f}")

# =============================================================================
# 6. Self-Similarity: Return Distributions at Different Frequencies
# =============================================================================
print("\n6. SELF-SIMILARITY OF RETURN DISTRIBUTIONS")
print("-" * 40)

freq_horizons = [1, 5, 20]
freq_names = ['Daily', 'Weekly', 'Monthly']

print(f"   {'Frequency':<12} {'Skewness':>10} {'Kurtosis':>10}")
print("   " + "-" * 34)

freq_returns = {}
for h, name in zip(freq_horizons, freq_names):
    ret_h = spy_daily.rolling(h).sum().dropna()
    freq_returns[h] = ret_h
    sk = stats.skew(ret_h)
    ku = stats.kurtosis(ret_h)
    print(f"   {name:<12} {sk:>10.4f} {ku:>10.4f}")

# =============================================================================
# 7. FIGURE: Fractal Market Hypothesis (6-panel, 3x2)
# =============================================================================
print("\n7. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# ---- Panel A: R/S analysis log-log plot for SPY ----
ax = axes[0, 0]
t = 'SPY'
H_rs, ns_rs, rs_vals = rs_results[t]
log_n = np.log(ns_rs)
log_rs = np.log(rs_vals)
slope_rs, intercept_rs, _, _, _ = stats.linregress(log_n, log_rs)

ax.scatter(ns_rs, rs_vals, s=20, color=MainBlue, zorder=3, edgecolors='none',
           alpha=0.8, label='Empirical R/S')
x_fit = np.linspace(ns_rs.min(), ns_rs.max(), 200)
ax.plot(x_fit, np.exp(intercept_rs) * x_fit ** slope_rs, color=Crimson,
        linewidth=1.5, label=f'OLS fit: H = {H_rs:.3f}')
# Reference line H=0.5
ref_intercept = intercept_rs + (0.5 - slope_rs) * np.mean(log_n)
ax.plot(x_fit, np.exp(ref_intercept) * x_fit ** 0.5, color='gray',
        linewidth=0.8, linestyle='--', alpha=0.7, label='H = 0.5 (random walk)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Panel A: R/S Analysis (SPY)', fontweight='bold')
ax.set_xlabel('Block size n')
ax.set_ylabel('R/S')
ax.text(0.05, 0.95, f'H = {H_rs:.3f}', transform=ax.transAxes,
        ha='left', va='top', fontsize=10, fontweight='bold', color=MainBlue,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(frameon=False, loc='lower right', fontsize=7)

# ---- Panel B: DFA log-log plot for SPY ----
ax = axes[0, 1]
alpha_dfa, ns_dfa, fluct_dfa = dfa_results[t]
log_n_d = np.log(ns_dfa)
log_f = np.log(fluct_dfa)
slope_d, intercept_d, _, _, _ = stats.linregress(log_n_d, log_f)

ax.scatter(ns_dfa, fluct_dfa, s=20, color=MainBlue, zorder=3,
           edgecolors='none', alpha=0.8, label='Empirical F(n)')
x_fit_d = np.linspace(ns_dfa.min(), ns_dfa.max(), 200)
ax.plot(x_fit_d, np.exp(intercept_d) * x_fit_d ** slope_d, color=Crimson,
        linewidth=1.5, label=f'OLS fit: $\\alpha$ = {alpha_dfa:.3f}')
# Reference line alpha=0.5
ref_intercept_d = intercept_d + (0.5 - slope_d) * np.mean(log_n_d)
ax.plot(x_fit_d, np.exp(ref_intercept_d) * x_fit_d ** 0.5, color='gray',
        linewidth=0.8, linestyle='--', alpha=0.7, label='$\\alpha$ = 0.5 (white noise)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Panel B: Detrended Fluctuation Analysis (SPY)', fontweight='bold')
ax.set_xlabel('Window size n')
ax.set_ylabel('F(n)')
ax.text(0.05, 0.95, f'$\\alpha$ = {alpha_dfa:.3f}', transform=ax.transAxes,
        ha='left', va='top', fontsize=10, fontweight='bold', color=MainBlue,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(frameon=False, loc='lower right', fontsize=7)

# ---- Panel C: Hurst exponent comparison bar chart (SPY, BTC, EUR) ----
ax = axes[1, 0]
asset_list = ['SPY', 'BTC-USD', 'EUR=X']
bar_labels = [labels[t] for t in asset_list]
h_rs_vals = [rs_results[t][0] for t in asset_list]
h_dfa_vals = [dfa_results[t][0] for t in asset_list]

x_pos = np.arange(len(asset_list))
width = 0.30
bars1 = ax.bar(x_pos - width / 2, h_rs_vals, width, color=MainBlue, alpha=0.85,
               label='R/S', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width / 2, h_dfa_vals, width, color=Crimson, alpha=0.85,
               label='DFA', edgecolor='white', linewidth=0.5)

# Annotate bar values
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.0,
           label='H = 0.5 (random walk)')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_title('Panel C: Hurst Exponent Comparison', fontweight='bold')
ax.set_ylabel('Hurst Exponent (H)')
ax.set_ylim(0.25, 0.80)
ax.legend(frameon=False, loc='upper right', fontsize=7)
ax.text(0.02, 0.02, 'H > 0.5: persistent\nH < 0.5: anti-persistent\nH = 0.5: random walk',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=7,
        color='gray', style='italic')

# ---- Panel D: Rolling Hurst exponent (252-day window, R/S) for SPY ----
ax = axes[1, 1]
ax.plot(rolling_dates, rolling_h, color=MainBlue, linewidth=0.8, alpha=0.9,
        label=f'Rolling H (R/S, {window_h}d)')
ax.axhline(y=0.5, color=Crimson, linestyle='--', linewidth=1.0,
           label='H = 0.5 (random walk)')
ax.fill_between(rolling_dates, 0.5, rolling_h,
                where=(rolling_h > 0.5), alpha=0.12, color=MainBlue,
                interpolate=True)
ax.fill_between(rolling_dates, 0.5, rolling_h,
                where=(rolling_h < 0.5), alpha=0.12, color=Crimson,
                interpolate=True)

ax.set_title(f'Panel D: Rolling Hurst Exponent (SPY, {window_h}d Window)',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Hurst Exponent')
ax.set_ylim(0.30, 0.80)
ax.legend(frameon=False, loc='upper right', fontsize=7)
ax.text(0.02, 0.02,
        f'Mean H = {np.mean(rolling_h):.3f}, Std = {np.std(rolling_h):.3f}',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=7,
        color=MainBlue,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---- Panel E: Multiscale volatility ----
ax = axes[2, 0]
ax.plot(agg_horizons, ann_vol_empirical, 'o-', color=MainBlue, linewidth=1.5,
        markersize=6, label='Empirical annualized vol', zorder=3)
ax.plot(agg_horizons, ann_vol_rw, 's--', color=Crimson, linewidth=1.0,
        markersize=5, alpha=0.7, label='Random walk ($\\sqrt{n}$ scaling)')

# Shade the gap
ax.fill_between(agg_horizons, ann_vol_empirical, ann_vol_rw, alpha=0.10,
                color=Purple)

ax.set_title('Panel E: Multiscale Volatility (SPY)', fontweight='bold')
ax.set_xlabel('Aggregation horizon (days)')
ax.set_ylabel('Annualized Volatility')
ax.legend(frameon=False, loc='upper left', fontsize=7)

# Compute empirical scaling exponent from variance
var_empirical_raw = []
for h in agg_horizons:
    agg_ret = spy_daily.rolling(h).sum().dropna()
    var_empirical_raw.append(agg_ret.var())
log_h_arr = np.log(np.array(agg_horizons, dtype=float))
log_v_arr = np.log(np.array(var_empirical_raw))
slope_var, _, _, _, _ = stats.linregress(log_h_arr, log_v_arr)
H_var = slope_var / 2.0

ax.text(0.95, 0.05,
        f'Var scaling: Var ~ $n^{{2H}}$\nEstimated H = {H_var:.3f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
        color=Purple, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---- Panel F: Self-similarity illustration ----
ax = axes[2, 1]
freq_colors = [Crimson, Forest, MainBlue]

for h, name, c in zip(freq_horizons, freq_names, freq_colors):
    ret_h = freq_returns[h]
    # Standardize to zero mean, unit variance
    ret_std = (ret_h - ret_h.mean()) / ret_h.std()
    ax.hist(ret_std, bins=120, density=True, alpha=0.30, color=c,
            label=f'{name} (n={len(ret_h)})')
    # Also overlay a KDE
    x_kde = np.linspace(-6, 6, 500)
    kde = stats.gaussian_kde(ret_std.dropna())
    ax.plot(x_kde, kde(x_kde), color=c, linewidth=1.2)

# Normal reference
x_norm = np.linspace(-6, 6, 500)
ax.plot(x_norm, stats.norm.pdf(x_norm), 'k--', linewidth=1.2,
        label='N(0,1)')

ax.set_title('Panel F: Self-Similarity of Return Distributions (SPY)',
             fontweight='bold')
ax.set_xlabel('Standardized Return')
ax.set_ylabel('Density')
ax.set_xlim(-5, 5)
ax.legend(frameon=False, fontsize=7)
ax.text(0.95, 0.95,
        'Self-similar shape\nacross frequencies',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        color=MainBlue, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig('ch14_fractal_hurst')

# =============================================================================
# 8. Summary Statistics
# =============================================================================
print("\n8. SUMMARY OF HURST EXPONENT ESTIMATES")
print("-" * 60)

print(f"\n   {'Asset':>8} {'H (R/S)':>10} {'H (DFA)':>10} {'Mean H':>10} "
      f"{'Interpretation':>18}")
print("   " + "-" * 60)

for t in asset_list:
    h_rs = rs_results[t][0]
    h_dfa = dfa_results[t][0]
    h_mean = (h_rs + h_dfa) / 2.0
    if h_mean > 0.55:
        interp = "Persistent"
    elif h_mean < 0.45:
        interp = "Anti-persistent"
    else:
        interp = "Near random walk"
    print(f"   {labels[t]:>8} {h_rs:>10.4f} {h_dfa:>10.4f} {h_mean:>10.4f} "
          f"{interp:>18}")

print(f"\n   Variance scaling exponent (SPY): H = {H_var:.4f}")
print(f"   Rolling Hurst mean (SPY): {np.mean(rolling_h):.4f} "
      f"+/- {np.std(rolling_h):.4f}")

print("\n   INTERPRETATION:")
print("   - H > 0.5: Persistent (trending) behavior; past trends")
print("     tend to continue. Consistent with Fractal Market Hypothesis.")
print("   - H = 0.5: Pure random walk; no long memory.")
print("   - H < 0.5: Anti-persistent (mean-reverting) behavior;")
print("     past trends tend to reverse.")
print("   - Rolling H captures time-varying market efficiency.")
print("   - Deviation from sqrt(n) volatility scaling reveals")
print("     fractal structure in the return process.")

print("\n" + "=" * 70)
print("FRACTAL MARKET HYPOTHESIS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch14_fractal_hurst.pdf: 6-panel fractal market analysis")
print("  - ch14_fractal_hurst.png: same (300 dpi)")
