"""
SFM_ch6_stylized_facts
======================
Stylized Facts of Financial Returns

Description:
- Download data and compute log returns
- Heavy tails: QQ plot and excess kurtosis
- Volatility clustering: ACF of |returns|
- Leverage effect: asymmetric volatility response
- Aggregational Gaussianity: distribution at different horizons

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf
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
print("SFM CHAPTER 6: STYLIZED FACTS OF FINANCIAL RETURNS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2000-01-01', end='2024-12-31',
                    progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Ticker: {ticker}")
print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to "
      f"{close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(log_ret)}")

# =============================================================================
# 2. Stylized Facts Statistics
# =============================================================================
print("\n2. STYLIZED FACTS STATISTICS")
print("-" * 40)

# Heavy tails
skew = stats.skew(log_ret)
kurt = stats.kurtosis(log_ret)
jb_stat, jb_pval = stats.jarque_bera(log_ret)
print(f"   Skewness:          {skew:.4f}")
print(f"   Excess Kurtosis:   {kurt:.4f}")
print(f"   Jarque-Bera:       {jb_stat:.2f} (p={jb_pval:.6f})")

# Autocorrelation structure
ac1_ret = log_ret.autocorr(1)
ac1_abs = np.abs(log_ret).autocorr(1)
ac1_sq = (log_ret ** 2).autocorr(1)
print(f"\n   Autocorrelation structure:")
print(f"   AC(1) returns:    {ac1_ret:.4f}  (near zero)")
print(f"   AC(1) |returns|:  {ac1_abs:.4f}  (significant)")
print(f"   AC(1) returns^2:  {ac1_sq:.4f}  (significant)")

# Tail probabilities vs Normal
print(f"\n   Tail probabilities vs Normal:")
for k in [3, 4, 5]:
    empirical = np.mean(np.abs(log_ret) > k * log_ret.std())
    normal = 2 * (1 - stats.norm.cdf(k))
    ratio = empirical / normal if normal > 0 else float('inf')
    print(f"   P(|r| > {k}sigma): empirical={empirical:.6f}, "
          f"normal={normal:.6f}, ratio={ratio:.1f}x")

# Leverage effect
print(f"\n   Leverage effect:")
corrs_leverage = []
for lag in range(1, 21):
    c = np.corrcoef(log_ret.iloc[:-lag].values,
                    np.abs(log_ret.iloc[lag:].values))[0, 1]
    corrs_leverage.append(c)
print(f"   Corr(r_t, |r_{{t+1}}|): {corrs_leverage[0]:.4f}")
print(f"   Corr(r_t, |r_{{t+5}}|): {corrs_leverage[4]:.4f}")

# =============================================================================
# 3. Aggregational Gaussianity
# =============================================================================
print("\n3. AGGREGATIONAL GAUSSIANITY")
print("-" * 40)

horizons = [1, 5, 20, 60]
horizon_names = ['Daily', 'Weekly', 'Monthly', 'Quarterly']

print(f"   {'Horizon':<12} {'Skewness':>10} {'Kurtosis':>10} "
      f"{'JB p-val':>10}")
print("   " + "-" * 46)

agg_returns = {}
for h, name in zip(horizons, horizon_names):
    ret_h = log_ret.rolling(h).sum().dropna()
    agg_returns[h] = ret_h
    sk = stats.skew(ret_h)
    ku = stats.kurtosis(ret_h)
    _, jb_p = stats.jarque_bera(ret_h)
    print(f"   {name:<12} {sk:>10.4f} {ku:>10.4f} {jb_p:>10.6f}")

print("\n   As horizon increases, kurtosis decreases toward 0 (Gaussian)")

# =============================================================================
# 4. FIGURE: Stylized Facts (6-panel)
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Panel A: Volatility clustering (returns time series)
axes[0, 0].plot(log_ret.index, log_ret, color='#1A3A6E',
                linewidth=0.3, alpha=0.8)
axes[0, 0].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
axes[0, 0].set_title('Fact 1: Volatility Clustering', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Log Return')

# Panel B: Heavy tails (QQ plot)
(osm, osr), (slope, intercept, r) = probplot(log_ret, dist='norm')
axes[0, 1].scatter(osm, osr, s=2, alpha=0.3, color='#1A3A6E',
                   edgecolors='none')
axes[0, 1].plot(osm, slope * osm + intercept, color='#DC3545',
                linewidth=1.5, label='Normal reference')
axes[0, 1].set_title('Fact 2: Heavy Tails (QQ-Plot)', fontweight='bold')
axes[0, 1].set_xlabel('Theoretical Quantiles (Normal)')
axes[0, 1].set_ylabel('Sample Quantiles')
axes[0, 1].text(0.05, 0.95,
               f'Excess Kurt: {kurt:.2f}\nSkewness: {skew:.2f}',
               transform=axes[0, 1].transAxes, ha='left', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 1].legend(frameon=False, loc='lower right')

# Panel C: ACF of |returns| (volatility clustering persistence)
plot_acf(np.abs(log_ret), lags=60, ax=axes[1, 0],
         color='#DC3545',
         vlines_kwargs={'colors': '#DC3545', 'linewidths': 0.8},
         title='')
axes[1, 0].set_title('Fact 3: ACF of |Returns| (Long Memory)',
                       fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')
axes[1, 0].text(0.95, 0.95, 'Slow decay\n(Long memory)',
               transform=axes[1, 0].transAxes, ha='right', va='top',
               fontsize=8, color='#DC3545',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel D: Leverage effect
axes[1, 1].bar(range(1, 21), corrs_leverage, color='#1A3A6E',
               edgecolor='white', width=0.7)
axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 1].set_title('Fact 4: Leverage Effect', fontweight='bold')
axes[1, 1].set_xlabel('Lag k')
axes[1, 1].set_ylabel(r'Corr($r_t$, $|r_{t+k}|$)')
axes[1, 1].text(0.95, 0.95,
               'Negative returns\nincrease volatility',
               transform=axes[1, 1].transAxes, ha='right', va='top',
               fontsize=8, color='#1A3A6E',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel E: Aggregational Gaussianity (overlaid histograms)
colors_agg = ['#DC3545', '#FF8C00', '#2E7D32', '#1A3A6E']
for h, name, c in zip(horizons, horizon_names, colors_agg):
    ret_h = agg_returns[h]
    ret_std = (ret_h - ret_h.mean()) / ret_h.std()
    axes[2, 0].hist(ret_std, bins=80, density=True, alpha=0.3,
                    color=c, label=name)

x_norm = np.linspace(-5, 5, 200)
axes[2, 0].plot(x_norm, stats.norm.pdf(x_norm), 'k--', linewidth=1.5,
                label='N(0,1)')
axes[2, 0].set_title('Fact 5: Aggregational Gaussianity',
                       fontweight='bold')
axes[2, 0].set_xlabel('Standardized Return')
axes[2, 0].set_ylabel('Density')
axes[2, 0].set_xlim(-5, 5)
axes[2, 0].legend(frameon=False, fontsize=7)

# Panel F: Kurtosis by horizon
horizons_full = [1, 2, 3, 5, 10, 20, 40, 60, 120]
kurtosis_by_h = []
for h in horizons_full:
    ret_h = log_ret.rolling(h).sum().dropna()
    kurtosis_by_h.append(stats.kurtosis(ret_h))

axes[2, 1].plot(horizons_full, kurtosis_by_h, color='#1A3A6E',
                marker='o', markersize=5, linewidth=1.5)
axes[2, 1].axhline(y=0, color='#DC3545', linestyle='--', linewidth=0.8,
                    label='Gaussian (kurtosis=0)')
axes[2, 1].set_title('Excess Kurtosis vs Aggregation Horizon',
                       fontweight='bold')
axes[2, 1].set_xlabel('Horizon (trading days)')
axes[2, 1].set_ylabel('Excess Kurtosis')
axes[2, 1].legend(frameon=False, loc='upper right')
axes[2, 1].text(0.95, 0.85,
               'Kurtosis decays\ntoward Gaussian',
               transform=axes[2, 1].transAxes, ha='right', va='top',
               fontsize=8, color='#2E7D32',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig('ch6_stylized_facts')

# =============================================================================
# 5. Summary of Stylized Facts
# =============================================================================
print("\n5. SUMMARY OF STYLIZED FACTS")
print("-" * 40)

print("   1. Volatility clustering:")
print("      Large returns tend to be followed by large returns")
print(f"   2. Heavy tails (leptokurtosis):")
print(f"      Excess kurtosis = {kurt:.2f} >> 0")
print("   3. Slow decay of ACF of |returns|:")
print("      Long memory in volatility")
print("   4. Leverage effect:")
print("      Negative returns increase future volatility")
print("   5. Aggregational Gaussianity:")
print("      Returns become more Gaussian at lower frequencies")

print("\n" + "=" * 70)
print("STYLIZED FACTS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch6_stylized_facts.pdf: 6-panel stylized facts analysis")
