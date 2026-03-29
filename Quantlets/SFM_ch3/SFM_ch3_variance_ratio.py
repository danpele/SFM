"""
SFM_ch3_variance_ratio
======================
Variance Ratio Test: Lo-MacKinlay (1988)

Description:
- Download S&P 500, FTSE 100, BTC-USD, BET daily data via yfinance
- Compute variance ratio VR(q) for q = 2, 5, 10, 20
- Implement Lo-MacKinlay heteroscedasticity-robust z-statistic
- Panel A: Bar chart of VR(q) grouped by asset
- Panel B: Rolling VR(5) for S&P 500 and BTC with crisis shading

Statistics of Financial Markets course — Chapter 3 (EMH)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
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
# Variance Ratio helpers
# =============================================================================

def variance_ratio(returns, q):
    """Compute variance ratio VR(q) = Var(r_t(q)) / (q * Var(r_t))."""
    T = len(returns)
    mu = returns.mean()
    # Overlapping q-period returns
    ret_q = pd.Series(returns).rolling(q).sum().dropna().values
    sigma2_1 = np.sum((returns - mu) ** 2) / (T - 1)
    sigma2_q = np.sum((ret_q - q * mu) ** 2) / (T - q + 1)
    # Bias-corrected
    nq = T - q + 1
    vr = sigma2_q / (q * sigma2_1)
    return vr


def lo_mackinlay_z(returns, q):
    """Lo-MacKinlay (1988) heteroscedasticity-robust z-statistic for VR(q)."""
    T = len(returns)
    mu = returns.mean()
    r = returns - mu

    # Variance ratio
    vr = variance_ratio(returns, q)

    # Heteroscedasticity-robust variance of VR
    delta_sum = 0.0
    for j in range(1, q):
        weight = (2 * (q - j) / q) ** 2
        numer = np.sum(r[j:] ** 2 * r[:-j] ** 2)
        denom = (np.sum(r ** 2)) ** 2
        delta_j = T * numer / denom
        delta_sum += weight * delta_j

    z_star = (vr - 1) / np.sqrt(delta_sum)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_star)))
    return vr, z_star, p_value


def rolling_vr(returns, q, window):
    """Compute rolling variance ratio VR(q) over a given window."""
    n = len(returns)
    dates = []
    vr_vals = []
    for i in range(window, n):
        chunk = returns[i - window:i]
        vr_val = variance_ratio(chunk, q)
        dates.append(i)
        vr_vals.append(vr_val)
    return dates, vr_vals


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SFM CHAPTER 3: VARIANCE RATIO TEST (Lo-MacKinlay 1988)")
    print("=" * 70)

    # =========================================================================
    # 1. Download Data
    # =========================================================================
    print("\n1. DOWNLOADING DATA")
    print("-" * 40)

    tickers = {'^GSPC': 'S&P 500', '^FTSE': 'FTSE 100',
               'BTC-USD': 'Bitcoin', 'EEM': 'Emerging Mkts'}
    ticker_list = list(tickers.keys())

    raw = yf.download(ticker_list, start='2010-01-01', end='2024-12-31',
                      progress=False)['Close']
    # Ensure DataFrame
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    returns_dict = {}
    for tk, name in tickers.items():
        col = tk if tk in raw.columns else None
        if col is None:
            # Try without special characters
            for c in raw.columns:
                if tk in str(c):
                    col = c
                    break
        if col is not None:
            r = raw[col].dropna().pct_change().dropna()
            returns_dict[name] = r
            print(f"   {name:12s} ({tk}): {len(r)} observations")
        else:
            print(f"   {name:12s} ({tk}): NOT AVAILABLE")

    # =========================================================================
    # 2. Variance Ratio Tests
    # =========================================================================
    print("\n2. VARIANCE RATIO TESTS")
    print("-" * 40)

    q_values = [2, 5, 10, 20]
    results = {}  # {asset: {q: (vr, z, p)}}

    header = f"   {'Asset':<12}"
    for q in q_values:
        header += f" {'VR('+str(q)+')':>8} {'z*':>7} {'p':>7}"
    print(header)
    print("   " + "-" * 90)

    for name, ret in returns_dict.items():
        r_arr = ret.values
        results[name] = {}
        line = f"   {name:<12}"
        for q in q_values:
            vr, z, p = lo_mackinlay_z(r_arr, q)
            results[name][q] = (vr, z, p)
            sig = "*" if p < 0.05 else " "
            line += f" {vr:>8.4f} {z:>7.2f} {p:>6.4f}{sig}"
        print(line)

    print("\n   * = significant at 5% level (reject H0: VR=1)")

    # =========================================================================
    # 3. FIGURE: Variance Ratio Analysis
    # =========================================================================
    print("\n3. CREATING FIGURE")
    print("-" * 40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Grouped bar chart of VR(q) ---
    assets = list(results.keys())
    n_assets = len(assets)
    n_q = len(q_values)
    bar_width = 0.18
    colors_bar = [MAIN_BLUE, CRIMSON, FOREST, AMBER]

    x_pos = np.arange(n_q)
    for i, asset in enumerate(assets):
        vr_vals = [results[asset][q][0] for q in q_values]
        offset = (i - n_assets / 2 + 0.5) * bar_width
        bars = ax1.bar(x_pos + offset, vr_vals, bar_width,
                       color=colors_bar[i % len(colors_bar)],
                       alpha=0.85, label=asset, edgecolor='none')

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8,
                label='$VR = 1$ (RW)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'$q = {q}$' for q in q_values])
    ax1.set_ylabel('Variance Ratio $VR(q)$')
    ax1.set_title('Panel A: Variance Ratio by Holding Period', fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=5, frameon=False, fontsize=7)

    # --- Panel B: Rolling VR(5) for S&P 500 and BTC ---
    q_roll = 5
    window = 500

    roll_assets = {'S&P 500': MAIN_BLUE, 'Bitcoin': CRIMSON}

    for asset_name, color in roll_assets.items():
        if asset_name not in returns_dict:
            continue
        ret = returns_dict[asset_name]
        r_arr = ret.values
        idx = ret.index

        n = len(r_arr)
        vr_dates = []
        vr_vals = []
        for i in range(window, n):
            chunk = r_arr[i - window:i]
            vr_val = variance_ratio(chunk, q_roll)
            vr_dates.append(idx[i])
            vr_vals.append(vr_val)

        ax2.plot(vr_dates, vr_vals, color=color, linewidth=0.6,
                 label=asset_name, alpha=0.85)

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)

    # Shade crisis periods
    crisis_periods = [
        ('2020-02-15', '2020-06-01', 'COVID-19'),
        ('2022-05-01', '2022-12-31', 'Crypto crash'),
    ]
    for start, end, label in crisis_periods:
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.10, color=AMBER)
        ax2.text(pd.Timestamp(start), ax2.get_ylim()[1] * 0.98, label,
                 fontsize=6, color=AMBER, va='top', ha='left')

    ax2.set_ylabel('$VR(5)$')
    ax2.set_xlabel('Date')
    ax2.set_title('Panel B: Rolling VR(5), 500-day window', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=False, fontsize=7)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig('ch3_variance_ratio')

    # =========================================================================
    # 4. Summary
    # =========================================================================
    print("\n4. SUMMARY")
    print("-" * 40)
    print("   VR > 1 => positive autocorrelation (momentum)")
    print("   VR < 1 => negative autocorrelation (mean reversion)")
    print("   VR = 1 => random walk (EMH consistent)")

    for name in assets:
        vr5, z5, p5 = results[name][5]
        status = "reject RW" if p5 < 0.05 else "consistent with RW"
        print(f"   {name:12s}: VR(5) = {vr5:.4f}, z* = {z5:.2f} => {status}")

    print("\n" + "=" * 70)
    print("VARIANCE RATIO TEST COMPLETE")
    print("=" * 70)
    print("\nOutput: ch3_variance_ratio.pdf/.png")
