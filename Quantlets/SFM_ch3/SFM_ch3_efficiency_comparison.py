"""
SFM_ch3_efficiency_comparison
=============================
Market Efficiency Comparison Across International Markets

Description:
- Download S&P 500, BTC-USD, BET, FTSE 100, Nikkei 225 daily data via yfinance
- For each market: autocorrelation rho(1), Ljung-Box Q(10), Runs test,
  Variance Ratio VR(5), Hurst exponent H
- Panel A: Heatmap of efficiency metrics (green=efficient, red=inefficient)
- Panel B: Scatter plot of H vs VR(5) with market labels

Statistics of Financial Markets course — Chapter 3 (EMH)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yfinance as yf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
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
# Helper functions
# =============================================================================

def runs_test(x):
    """Wald-Wolfowitz runs test for randomness."""
    median = np.median(x)
    signs = (x >= median).astype(int)
    n1 = signs.sum()
    n2 = len(signs) - n1
    runs = 1 + np.sum(np.diff(signs) != 0)
    mu = (2 * n1 * n2) / (n1 + n2) + 1
    sigma2 = ((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
              ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    z = (runs - mu) / np.sqrt(sigma2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def variance_ratio(returns, q):
    """Compute variance ratio VR(q)."""
    T = len(returns)
    mu = returns.mean()
    ret_q = pd.Series(returns).rolling(q).sum().dropna().values
    sigma2_1 = np.sum((returns - mu) ** 2) / (T - 1)
    sigma2_q = np.sum((ret_q - q * mu) ** 2) / (T - q + 1)
    vr = sigma2_q / (q * sigma2_1)
    return vr


def hurst_rs(returns, min_window=10):
    """Compute Hurst exponent via R/S analysis."""
    T = len(returns)
    n_values = []
    n = min_window
    while n <= T // 2:
        n_values.append(n)
        n = int(n * 1.3)
    n_values = sorted(set(n_values))

    log_n_list = []
    log_rs_list = []

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

    if len(log_n) >= 3:
        H, _ = np.polyfit(log_n, log_rs, 1)
    else:
        H = np.nan
    return H


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SFM CHAPTER 3: MARKET EFFICIENCY COMPARISON")
    print("=" * 70)

    # =========================================================================
    # 1. Download Data
    # =========================================================================
    print("\n1. DOWNLOADING DATA")
    print("-" * 40)

    tickers = {'^GSPC': 'S&P 500', 'BTC-USD': 'Bitcoin',
               'EEM': 'Emerging Mkts', '^FTSE': 'FTSE 100', '^N225': 'Nikkei 225'}
    ticker_list = list(tickers.keys())

    raw = yf.download(ticker_list, start='2015-01-01', end='2024-12-31',
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
    # 2. Compute Efficiency Metrics
    # =========================================================================
    print("\n2. EFFICIENCY METRICS")
    print("-" * 40)

    metrics = {}  # {market: {metric: value}}
    p_values = {}  # {market: {metric: p_value}}

    for name, ret in returns_dict.items():
        r_arr = ret.values
        m = {}
        p = {}

        # (a) Autocorrelation rho(1)
        rho1 = ret.autocorr(lag=1)
        m['rho(1)'] = rho1
        # Approximate p-value for rho(1) under H0
        n = len(r_arr)
        z_rho = rho1 * np.sqrt(n)
        p['rho(1)'] = 2 * (1 - stats.norm.cdf(abs(z_rho)))

        # (b) Ljung-Box Q(10)
        lb = acorr_ljungbox(ret, lags=[10], return_df=True)
        m['Q(10)'] = lb.iloc[0]['lb_stat']
        p['Q(10)'] = lb.iloc[0]['lb_pvalue']

        # (c) Runs test
        z_runs, p_runs = runs_test(r_arr)
        m['Runs Z'] = z_runs
        p['Runs Z'] = p_runs

        # (d) Variance Ratio VR(5)
        vr5 = variance_ratio(r_arr, 5)
        m['VR(5)'] = vr5
        # Approximate p-value: |VR-1| significance
        # Use simplified z under heteroscedasticity
        z_vr = (vr5 - 1) * np.sqrt(n / (2 * (5 - 1)))
        p['VR(5)'] = 2 * (1 - stats.norm.cdf(abs(z_vr)))

        # (e) Hurst exponent
        H = hurst_rs(r_arr)
        m['Hurst H'] = H
        # Approximate: deviation from 0.5
        z_h = (H - 0.5) / (0.5 / np.sqrt(n))
        p['Hurst H'] = 2 * (1 - stats.norm.cdf(abs(z_h)))

        metrics[name] = m
        p_values[name] = p

    # Print results table
    metric_names = ['rho(1)', 'Q(10)', 'Runs Z', 'VR(5)', 'Hurst H']
    header = f"   {'Market':<12}"
    for mn in metric_names:
        header += f" {mn:>10}"
    print(header)
    print("   " + "-" * 65)

    for name in returns_dict.keys():
        line = f"   {name:<12}"
        for mn in metric_names:
            val = metrics[name][mn]
            pv = p_values[name][mn]
            sig = "*" if pv < 0.05 else " "
            line += f" {val:>9.4f}{sig}"
        print(line)

    print("\n   * = significant at 5% level (evidence against EMH)")

    # Detailed p-values
    print(f"\n   {'Market':<12}", end="")
    for mn in metric_names:
        print(f" {'p(' + mn + ')':>10}", end="")
    print()
    print("   " + "-" * 65)
    for name in returns_dict.keys():
        line = f"   {name:<12}"
        for mn in metric_names:
            pv = p_values[name][mn]
            line += f" {pv:>10.4f}"
        print(line)

    # =========================================================================
    # 3. FIGURE: Efficiency Comparison
    # =========================================================================
    print("\n3. CREATING FIGURE")
    print("-" * 40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Heatmap of efficiency metrics ---
    market_names = list(returns_dict.keys())
    n_markets = len(market_names)
    n_metrics = len(metric_names)

    # Build p-value matrix for coloring (low p = red/inefficient)
    pval_matrix = np.zeros((n_markets, n_metrics))
    val_matrix = np.zeros((n_markets, n_metrics))
    for i, name in enumerate(market_names):
        for j, mn in enumerate(metric_names):
            pval_matrix[i, j] = p_values[name][mn]
            val_matrix[i, j] = metrics[name][mn]

    # Custom colormap: red (p<0.05) to green (p>0.1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'eff', [CRIMSON, '#FFFFFF', FOREST], N=256)

    # Clip p-values for display
    pval_display = np.clip(pval_matrix, 0.001, 1.0)

    im = ax1.imshow(pval_display, cmap=cmap, aspect='auto',
                    vmin=0.0, vmax=0.2)

    ax1.set_xticks(range(n_metrics))
    ax1.set_xticklabels(metric_names, rotation=30, ha='right')
    ax1.set_yticks(range(n_markets))
    ax1.set_yticklabels(market_names)
    ax1.set_title('Panel A: Efficiency Metrics (p-values)', fontweight='bold')

    # Re-enable spines for heatmap
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Add text annotations
    for i in range(n_markets):
        for j in range(n_metrics):
            val = val_matrix[i, j]
            pv = pval_matrix[i, j]
            # Choose text color for readability
            text_color = 'white' if pv < 0.03 else 'black'
            sig_marker = '*' if pv < 0.05 else ''
            if metric_names[j] == 'Q(10)':
                txt = f'{val:.1f}{sig_marker}'
            else:
                txt = f'{val:.3f}{sig_marker}'
            ax1.text(j, i, txt, ha='center', va='center',
                     fontsize=7, color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('p-value', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --- Panel B: Scatter H vs VR(5) ---
    colors_scatter = [MAIN_BLUE, CRIMSON, FOREST, AMBER, ORANGE]
    markers = ['o', 's', 'D', '^', 'v']

    for i, name in enumerate(market_names):
        h_val = metrics[name]['Hurst H']
        vr_val = metrics[name]['VR(5)']
        ax2.scatter(h_val, vr_val, color=colors_scatter[i % len(colors_scatter)],
                    marker=markers[i % len(markers)], s=80, zorder=5,
                    edgecolors='white', linewidths=0.5)
        ax2.annotate(name, (h_val, vr_val),
                     xytext=(8, 5), textcoords='offset points',
                     fontsize=7, color=colors_scatter[i % len(colors_scatter)])

    # Reference lines
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.6,
                label='$VR = 1$')
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.6,
                label='$H = 0.5$')

    # Shade efficient region
    ax2.axhspan(0.95, 1.05, alpha=0.06, color=FOREST)
    ax2.axvspan(0.45, 0.55, alpha=0.06, color=FOREST)

    ax2.set_xlabel('Hurst Exponent $H$')
    ax2.set_ylabel('Variance Ratio $VR(5)$')
    ax2.set_title('Panel B: Efficiency Space ($H$ vs $VR$)', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False, fontsize=7)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig('ch3_efficiency_comparison')

    # =========================================================================
    # 4. Summary
    # =========================================================================
    print("\n4. EFFICIENCY RANKING")
    print("-" * 40)

    # Score: count how many tests are NOT significant (p > 0.05)
    print(f"   {'Market':<12} {'Efficient tests':>16} {'Score':>8}")
    print("   " + "-" * 40)

    for i, name in enumerate(market_names):
        n_eff = sum(1 for mn in metric_names if p_values[name][mn] > 0.05)
        score = n_eff / n_metrics
        bar = '#' * n_eff + '.' * (n_metrics - n_eff)
        print(f"   {name:<12} {n_eff}/{n_metrics} [{bar}]  {score:>6.1%}")

    print("\n   Interpretation:")
    print("   - Green (high p-value) = consistent with efficiency")
    print("   - Red (low p-value) = evidence against efficiency")
    print("   - More mature markets tend to be more efficient")

    print("\n" + "=" * 70)
    print("EFFICIENCY COMPARISON COMPLETE")
    print("=" * 70)
    print("\nOutput: ch3_efficiency_comparison.pdf/.png")
