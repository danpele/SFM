"""
SFM_ch5_variance_ratio
======================
Variance Ratio Tests: Lo-MacKinlay and Chow-Denning

Description:
- Implement Lo-MacKinlay variance ratio test
- Implement Chow-Denning multiple variance ratio test
- Download S&P 500 data, compute VR for periods 2, 4, 8, 16
- Plot VR vs holding period
- Rolling VR analysis and simulation under H0

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
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

# =============================================================================
# 1. Define Variance Ratio Test Functions
# =============================================================================

def variance_ratio(returns, q):
    """
    Lo-MacKinlay (1988) variance ratio test.

    VR(q) = Var(r_t(q)) / (q * Var(r_t))

    Under H0 (random walk): VR(q) = 1
    VR > 1 implies positive autocorrelation (momentum)
    VR < 1 implies negative autocorrelation (mean reversion)

    Returns: VR, z_stat (homoskedastic), p_value
    """
    T = len(returns)
    mu = returns.mean()

    # Variance of 1-period returns
    sigma2_1 = np.sum((returns - mu) ** 2) / (T - 1)

    # Variance of q-period returns
    ret_q = pd.Series(returns).rolling(q).sum().dropna().values
    nq = len(ret_q)
    sigma2_q = np.sum((ret_q - q * mu) ** 2) / (nq - 1) / q

    VR = sigma2_q / sigma2_1

    # Homoskedastic test statistic (Lo-MacKinlay)
    phi = 2 * (2 * q - 1) * (q - 1) / (3 * q * T)
    z_stat = (VR - 1) / np.sqrt(phi)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return VR, z_stat, p_value


def variance_ratio_heteroskedastic(returns, q):
    """
    Lo-MacKinlay (1988) heteroskedasticity-robust VR test.
    Uses heteroskedasticity-consistent variance estimate.
    """
    T = len(returns)
    mu = returns.mean()

    sigma2_1 = np.sum((returns - mu) ** 2) / (T - 1)
    ret_q = pd.Series(returns).rolling(q).sum().dropna().values
    nq = len(ret_q)
    sigma2_q = np.sum((ret_q - q * mu) ** 2) / (nq - 1) / q

    VR = sigma2_q / sigma2_1

    # Heteroskedasticity-robust variance
    delta = np.zeros(q - 1)
    for j in range(1, q):
        num = np.sum((returns[j:] - mu) ** 2 *
                     (returns[:-j] - mu) ** 2)
        den = (np.sum((returns - mu) ** 2)) ** 2
        delta[j - 1] = T * num / den

    theta = np.sum([(2 * (q - j) / q) ** 2 * delta[j - 1]
                     for j in range(1, q)])

    z_star = (VR - 1) / np.sqrt(theta)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_star)))

    return VR, z_star, p_value


def chow_denning_test(returns, q_values):
    """
    Chow-Denning (1993) multiple variance ratio test.
    Uses max |z| statistic across multiple holding periods.
    """
    z_stats = []
    for q in q_values:
        _, z, _ = variance_ratio(returns, q)
        z_stats.append(abs(z))
    max_z = max(z_stats)
    # Bonferroni-corrected p-value
    k = len(q_values)
    p_value = min(1.0, k * 2 * (1 - stats.norm.cdf(max_z)))
    return max_z, p_value


print("=" * 70)
print("SFM CHAPTER 5: VARIANCE RATIO TESTS")
print("=" * 70)

# =============================================================================
# 2. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = ['SPY', 'AAPL', 'MSFT', 'BTC-USD']
ticker_names = {'SPY': 'S&P 500 ETF', 'AAPL': 'Apple',
                'MSFT': 'Microsoft', 'BTC-USD': 'Bitcoin'}

prices = yf.download(tickers, start='2015-01-01', end='2024-12-31',
                      progress=False)['Close']
returns = {}
for t in tickers:
    r = np.log(prices[t] / prices[t].shift(1)).dropna()
    returns[t] = r
    print(f"   {ticker_names[t]:15s} ({t}): {len(r)} observations")

# =============================================================================
# 3. Lo-MacKinlay Variance Ratio Test
# =============================================================================
print("\n2. LO-MACKINLAY VARIANCE RATIO TEST")
print("-" * 40)

q_values = [2, 4, 8, 16]

print(f"   {'Asset':<10}", end="")
for q in q_values:
    print(f" {'VR(' + str(q) + ')':>10}", end="")
print()
print("   " + "-" * 50)

vr_results = {}
for t in tickers:
    vr_results[t] = {}
    print(f"   {t:<10}", end="")
    for q in q_values:
        vr, z, p = variance_ratio(returns[t].values, q)
        vr_results[t][q] = (vr, z, p)
        star = "*" if p < 0.05 else ""
        print(f" {vr:>9.4f}{star}", end="")
    print()
print("   (* significant at 5% level)")

# Detailed results for S&P 500
print(f"\n   Detailed results for SPY:")
print(f"   {'q':>5} {'VR(q)':>10} {'Z-stat':>10} {'p-value':>10} "
      f"{'Reject RW':>12}")
print("   " + "-" * 50)
for q in q_values:
    vr, z, p = vr_results['SPY'][q]
    reject = "Yes" if p < 0.05 else "No"
    print(f"   {q:>5} {vr:>10.4f} {z:>10.4f} {p:>10.4f} {reject:>12}")

# =============================================================================
# 4. Heteroskedasticity-Robust Test
# =============================================================================
print("\n3. HETEROSKEDASTICITY-ROBUST VR TEST (SPY)")
print("-" * 40)

print(f"   {'q':>5} {'VR(q)':>10} {'Z*-stat':>10} {'p-value':>10} "
      f"{'Reject RW':>12}")
print("   " + "-" * 50)
for q in q_values:
    vr, z_star, p = variance_ratio_heteroskedastic(
        returns['SPY'].values, q)
    reject = "Yes" if p < 0.05 else "No"
    print(f"   {q:>5} {vr:>10.4f} {z_star:>10.4f} {p:>10.4f} "
          f"{reject:>12}")

# =============================================================================
# 5. Chow-Denning Multiple VR Test
# =============================================================================
print("\n4. CHOW-DENNING MULTIPLE VR TEST")
print("-" * 40)

print(f"   {'Asset':<10} {'Max|Z|':>10} {'p-value':>10} "
      f"{'Reject RW':>12}")
print("   " + "-" * 46)
for t in tickers:
    max_z, p = chow_denning_test(returns[t].values, q_values)
    reject = "Yes" if p < 0.05 else "No"
    print(f"   {t:<10} {max_z:>10.4f} {p:>10.4f} {reject:>12}")

# =============================================================================
# 6. FIGURE: Variance Ratio Tests (4-panel)
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#1A3A6E', '#DC3545', '#2E7D32', '#FF8C00']

# Panel A: VR(q) across holding periods for all assets
q_range = [2, 3, 4, 5, 8, 10, 16, 20, 30, 40, 50]
for t, c in zip(tickers, colors):
    vrs = [variance_ratio(returns[t].values, q)[0] for q in q_range]
    axes[0, 0].plot(q_range, vrs, color=c, marker='o', markersize=4,
                    linewidth=1, label=ticker_names[t])
axes[0, 0].axhline(y=1, color='gray', linestyle='--', linewidth=0.8,
                    label='RW (VR=1)')
axes[0, 0].set_title('Variance Ratio by Holding Period',
                       fontweight='bold')
axes[0, 0].set_xlabel('Holding Period (q)')
axes[0, 0].set_ylabel('VR(q)')
axes[0, 0].legend(loc='upper left', fontsize=7)

# Panel B: Z-statistics for SPY across holding periods
z_stats_spy = [variance_ratio(returns['SPY'].values, q)[1]
               for q in q_range]
bar_colors = ['#DC3545' if abs(z) > 1.96 else '#1A3A6E'
              for z in z_stats_spy]
axes[0, 1].bar(range(len(q_range)), z_stats_spy, color=bar_colors,
               alpha=0.7, edgecolor='white')
axes[0, 1].axhline(y=1.96, color='gray', linestyle='--', linewidth=0.5,
                    label='95% critical value')
axes[0, 1].axhline(y=-1.96, color='gray', linestyle='--', linewidth=0.5)
axes[0, 1].set_xticks(range(len(q_range)))
axes[0, 1].set_xticklabels([str(q) for q in q_range], fontsize=7)
axes[0, 1].set_title('SPY: Z-Statistics by Holding Period',
                       fontweight='bold')
axes[0, 1].set_xlabel('Holding Period (q)')
axes[0, 1].set_ylabel('Z-statistic')
axes[0, 1].legend(loc='upper right', fontsize=7)

# Panel C: Rolling VR(5) for SPY (1-year window)
window_vr = 252
spy_ret = returns['SPY'].values
rolling_vr = []
rolling_idx = returns['SPY'].index[window_vr:]
for i in range(window_vr, len(spy_ret)):
    vr, _, _ = variance_ratio(spy_ret[i - window_vr:i], 5)
    rolling_vr.append(vr)

axes[1, 0].plot(rolling_idx, rolling_vr, color='#1A3A6E', linewidth=0.8)
axes[1, 0].axhline(y=1, color='#DC3545', linestyle='--', linewidth=0.8,
                    label='RW (VR=1)')
axes[1, 0].fill_between(rolling_idx, 1, rolling_vr, alpha=0.2,
                         color='#1A3A6E')
axes[1, 0].set_title('SPY: Rolling VR(5), 1-Year Window',
                       fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('VR(5)')
axes[1, 0].legend(loc='upper right', fontsize=7)

# Panel D: Simulated VR distribution under H0
n_sim = 5000
T_sim = 500
np.random.seed(42)
vr_sim = []
for _ in range(n_sim):
    rw = np.random.normal(0, 1, T_sim)
    vr, _, _ = variance_ratio(rw, 5)
    vr_sim.append(vr)

# Empirical VR for SPY
vr_spy_5 = variance_ratio(returns['SPY'].values, 5)[0]

axes[1, 1].hist(vr_sim, bins=50, density=True, alpha=0.5,
                color='#1A3A6E', edgecolor='white',
                label='Simulated VR(5) under H0')
axes[1, 1].axvline(x=1, color='gray', linewidth=1,
                    linestyle='--', label='Theoretical VR=1')
axes[1, 1].axvline(x=vr_spy_5, color='#DC3545', linewidth=1.5,
                    label=f'SPY VR(5)={vr_spy_5:.3f}')
axes[1, 1].set_title('Distribution of VR(5) under H0',
                       fontweight='bold')
axes[1, 1].set_xlabel('VR(5)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(loc='upper right', fontsize=7)

plt.tight_layout()
save_fig('ch5_variance_ratio')

# =============================================================================
# 7. Interpretation
# =============================================================================
print("\n6. INTERPRETATION")
print("-" * 40)

print("   VR(q) > 1: Positive autocorrelation (momentum)")
print("   VR(q) < 1: Negative autocorrelation (mean-reversion)")
print("   VR(q) = 1: Random walk (no predictability)")
print(f"\n   SPY VR results suggest:")
for q in q_values:
    vr = vr_results['SPY'][q][0]
    if vr > 1:
        interp = "slight positive autocorrelation"
    else:
        interp = "slight negative autocorrelation"
    print(f"     VR({q:>2}) = {vr:.4f} => {interp}")

print("\n" + "=" * 70)
print("VARIANCE RATIO TESTS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch5_variance_ratio.pdf: 4-panel variance ratio analysis")
