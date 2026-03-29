"""
SFM_ch3_hurst_exponent
======================
Hurst Exponent via R/S Analysis

Description:
- Download S&P 500, BTC-USD, Gold, BET daily data via yfinance
- Implement Hurst exponent via Rescaled Range (R/S) analysis
- Panel A: R/S log-log plot for S&P 500 with regression line
- Panel B: Rolling Hurst exponent for S&P 500 and BTC-USD
- Panel C: Bar chart comparing H across all 4 assets

Statistics of Financial Markets course — Chapter 3 (EMH)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── Chart style settings — Nature journal quality ───────────────────────────
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

# ─── Colors ──────────────────────────────────────────────────────────────────
MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E67E22'

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'charts'))
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
# R/S Analysis helpers
# =============================================================================

def rs_analysis(returns, min_window=10, max_divisions=20):
    """Compute Rescaled Range (R/S) statistic for various sub-period lengths.

    Returns
    -------
    log_n : array of log(sub-period length)
    log_rs : array of log(mean R/S) for each sub-period length
    H : Hurst exponent (OLS slope)
    """
    T = len(returns)
    log_n_list = []
    log_rs_list = []

    # Generate sub-period sizes: powers of 2 and intermediate values
    n_values = []
    n = min_window
    while n <= T // 2:
        n_values.append(n)
        n = int(n * 1.3)  # geometric spacing
    n_values = sorted(set(n_values))

    for n in n_values:
        num_blocks = T // n
        if num_blocks < 1:
            continue

        rs_vals = []
        for b in range(num_blocks):
            block = returns[b * n:(b + 1) * n]
            mean_block = np.mean(block)
            deviate = np.cumsum(block - mean_block)
            R = np.max(deviate) - np.min(deviate)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_vals.append(R / S)

        if len(rs_vals) > 0:
            log_n_list.append(np.log(n))
            log_rs_list.append(np.log(np.mean(rs_vals)))

    log_n = np.array(log_n_list)
    log_rs = np.array(log_rs_list)

    # OLS regression: log(R/S) = H * log(n) + c
    if len(log_n) >= 3:
        H, intercept = np.polyfit(log_n, log_rs, 1)
    else:
        H, intercept = np.nan, np.nan

    return log_n, log_rs, H, intercept


def rolling_hurst(returns, window=500, min_window=10):
    """Compute rolling Hurst exponent."""
    n = len(returns)
    h_vals = []
    indices = []
    for i in range(window, n):
        chunk = returns[i - window:i]
        _, _, H, _ = rs_analysis(chunk, min_window=min_window)
        h_vals.append(H)
        indices.append(i)
    return indices, h_vals


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SFM CHAPTER 3: HURST EXPONENT (R/S ANALYSIS)")
    print("=" * 70)

    # =========================================================================
    # 1. Download Data
    # =========================================================================
    print("\n1. DOWNLOADING DATA")
    print("-" * 40)

    tickers = {'^GSPC': 'S&P 500', 'BTC-USD': 'Bitcoin',
               'GC=F': 'Gold', 'EEM': 'Emerging Mkts'}
    ticker_list = list(tickers.keys())

    raw = yf.download(ticker_list, start='2010-01-01', end='2024-12-31',
                      progress=False)['Close']
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    returns_dict = {}
    for tk, name in tickers.items():
        if tk in raw.columns:
            r = raw[tk].dropna().pct_change().dropna()
            returns_dict[name] = r
            print(f"   {name:12s} ({tk}): {len(r)} observations")
        else:
            print(f"   {name:12s} ({tk}): NOT AVAILABLE")

    # =========================================================================
    # 2. Hurst Exponent Estimation
    # =========================================================================
    print("\n2. HURST EXPONENT ESTIMATES")
    print("-" * 40)

    hurst_results = {}
    print(f"   {'Asset':<12} {'H':>8} {'Interpretation':<30}")
    print("   " + "-" * 52)

    for name, ret in returns_dict.items():
        r_arr = ret.values
        log_n, log_rs, H, intercept = rs_analysis(r_arr)
        hurst_results[name] = (log_n, log_rs, H, intercept)

        if H < 0.45:
            interp = "Mean-reverting (anti-persistent)"
        elif H > 0.55:
            interp = "Trending (persistent)"
        else:
            interp = "Random walk (efficient)"
        print(f"   {name:<12} {H:>8.4f} {interp}")

    # =========================================================================
    # 3. FIGURE: Hurst Exponent Analysis
    # =========================================================================
    print("\n3. CREATING FIGURE")
    print("-" * 40)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel A: R/S log-log plot for S&P 500 ---
    if 'S&P 500' in hurst_results:
        log_n, log_rs, H, intercept = hurst_results['S&P 500']
        ax1.plot(log_n, log_rs, 'o', color=MAIN_BLUE, markersize=4, alpha=0.7)
        # Regression line
        x_fit = np.linspace(log_n.min(), log_n.max(), 100)
        y_fit = H * x_fit + intercept
        ax1.plot(x_fit, y_fit, '--', color=CRIMSON, linewidth=1.0,
                 label=fr'$H = {H:.3f}$')
        # Reference lines
        y_05 = 0.5 * x_fit + (log_rs.mean() - 0.5 * log_n.mean())
        ax1.plot(x_fit, y_05, ':', color='gray', linewidth=0.6,
                 label='$H = 0.5$ (random walk)')
        ax1.set_xlabel(r'$\log(n)$')
        ax1.set_ylabel(r'$\log(R/S)$')
        ax1.set_title('Panel A: R/S Analysis (S&P 500)', fontweight='bold')
        ax1.legend(loc='upper left', frameon=False, fontsize=7)

    # --- Panel B: Rolling Hurst for S&P 500 and BTC ---
    roll_assets = {'S&P 500': MAIN_BLUE, 'Bitcoin': CRIMSON}
    window = 500

    for asset_name, color in roll_assets.items():
        if asset_name not in returns_dict:
            continue
        ret = returns_dict[asset_name]
        r_arr = ret.values
        idx = ret.index

        indices, h_vals = rolling_hurst(r_arr, window=window)
        dates = [idx[i] for i in indices]

        ax2.plot(dates, h_vals, color=color, linewidth=0.6,
                 label=asset_name, alpha=0.85)

    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8)
    # Shade efficient zone [0.45, 0.55]
    ax2.axhspan(0.45, 0.55, alpha=0.08, color=FOREST, label='Efficient zone')
    ax2.set_ylabel('Hurst exponent $H$')
    ax2.set_xlabel('Date')
    ax2.set_title('Panel B: Rolling Hurst (500-day window)', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=False, fontsize=7)
    ax2.set_ylim(0.3, 0.8)

    # --- Panel C: Bar chart comparing H across assets ---
    asset_names = list(hurst_results.keys())
    h_values = [hurst_results[a][2] for a in asset_names]
    colors_bar = [MAIN_BLUE, CRIMSON, FOREST, AMBER]

    bars = ax3.bar(range(len(asset_names)), h_values,
                   color=colors_bar[:len(asset_names)],
                   alpha=0.85, edgecolor='none')
    ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8,
                label='$H = 0.5$ (random walk)')
    ax3.axhspan(0.45, 0.55, alpha=0.08, color=FOREST)
    ax3.set_xticks(range(len(asset_names)))
    ax3.set_xticklabels(asset_names, rotation=15, ha='right')
    ax3.set_ylabel('Hurst exponent $H$')
    ax3.set_title('Panel C: Hurst Exponent Comparison', fontweight='bold')

    # Add value labels on bars
    for i, (bar, h) in enumerate(zip(bars, h_values)):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7,
                 color=colors_bar[i])

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig('ch3_hurst_exponent')

    # =========================================================================
    # 4. Summary
    # =========================================================================
    print("\n4. SUMMARY")
    print("-" * 40)
    print("   H = 0.5  => random walk (EMH consistent)")
    print("   H > 0.5  => long-range dependence (trending)")
    print("   H < 0.5  => anti-persistent (mean-reverting)")

    for name in asset_names:
        H = hurst_results[name][2]
        print(f"   {name:12s}: H = {H:.4f}")

    print("\n" + "=" * 70)
    print("HURST EXPONENT ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput: ch3_hurst_exponent.pdf/.png")
