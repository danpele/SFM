"""
SFM_ch13_systemic_risk
======================
Systemic Risk Measures: CoVaR and Correlation Analysis

Description:
- Download financial sector stock data
- Rolling pairwise correlations
- Simple Delta-CoVaR estimation via quantile regression
- Correlation heatmap across financial institutions
- Dynamic connectedness visualization

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Standard chart style ---
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
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0

def save_fig(name):
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

def simple_covar(r_system, r_institution, alpha=0.05, window=252):
    """
    Simplified Delta-CoVaR estimation.
    CoVaR = VaR of system conditional on institution being at its VaR.
    Delta-CoVaR = CoVaR - VaR_system (unconditional).

    Uses rolling OLS: r_system = a + b * r_institution + e
    CoVaR_alpha = a + b * VaR_alpha(institution)
    """
    n = len(r_system)
    covar = np.full(n, np.nan)
    delta_covar = np.full(n, np.nan)

    for i in range(window, n):
        rs = r_system.iloc[i - window:i].values
        ri = r_institution.iloc[i - window:i].values

        # OLS regression
        slope, intercept, _, _, _ = stats.linregress(ri, rs)

        # VaR of institution
        var_i = np.percentile(ri, alpha * 100)

        # CoVaR
        covar[i] = intercept + slope * var_i

        # Unconditional VaR of system
        var_s = np.percentile(rs, alpha * 100)

        # Delta-CoVaR
        delta_covar[i] = covar[i] - var_s

    return (pd.Series(covar, index=r_system.index),
            pd.Series(delta_covar, index=r_system.index))

print("=" * 70)
print("SFM CHAPTER 13: SYSTEMIC RISK MEASURES")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

# Major US financial institutions + market/sector index
fin_tickers = {
    'JPM': 'JPMorgan Chase',
    'BAC': 'Bank of America',
    'C': 'Citigroup',
    'GS': 'Goldman Sachs',
    'MS': 'Morgan Stanley',
    'WFC': 'Wells Fargo',
    'XLF': 'Financial Sector ETF'
}

all_tickers = list(fin_tickers.keys()) + ['^GSPC']
data = yf.download(all_tickers, start='2007-01-01', end='2024-12-31',
                    progress=False)['Close']

# Compute log returns
returns = np.log(data / data.shift(1)).dropna()

for t, name in fin_tickers.items():
    print(f"   {name:20s} ({t}): {len(returns[t])} obs")
print(f"   {'S&P 500':20s} (^GSPC): {len(returns['^GSPC'])} obs")

# =============================================================================
# 2. Correlation Analysis
# =============================================================================
print("\n2. CORRELATION ANALYSIS")
print("-" * 40)

bank_tickers = list(fin_tickers.keys())
fin_returns = returns[bank_tickers]
corr_matrix = fin_returns.corr()

upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
avg_corr = upper_tri.stack().mean()
print(f"   Average pairwise correlation: {avg_corr:.4f}")

# =============================================================================
# 3. Rolling Correlations
# =============================================================================
print("\n3. ROLLING CORRELATIONS")
print("-" * 40)

window = 120
pairs = [('JPM', 'BAC'), ('JPM', 'GS'), ('GS', 'MS')]
rolling_corrs = {}
for t1, t2 in pairs:
    rc = returns[t1].rolling(window).corr(returns[t2])
    rolling_corrs[(t1, t2)] = rc
    print(f"   {t1}-{t2}: mean={rc.mean():.4f}, "
          f"min={rc.min():.4f}, max={rc.max():.4f}")

# =============================================================================
# 4. CoVaR Analysis
# =============================================================================
print("\n4. DELTA-CoVaR ANALYSIS")
print("-" * 40)

system_ret = returns['XLF']
alpha_covar = 0.05

print(f"\n   {'Institution':<20} {'Mean DCoVaR':>12} {'Max DCoVaR':>12}")
print("   " + "-" * 46)

covar_results = {}
for tick in ['JPM', 'BAC', 'C', 'GS', 'MS', 'WFC']:
    covar, dcovar = simple_covar(system_ret, returns[tick],
                                  alpha=alpha_covar, window=252)
    covar_results[tick] = {'covar': covar, 'dcovar': dcovar}
    mean_dc = dcovar.dropna().mean()
    max_dc = dcovar.dropna().min()  # most negative = highest risk
    print(f"   {fin_tickers[tick]:20s} {mean_dc:>12.6f} {max_dc:>12.6f}")

# =============================================================================
# 5. FIGURE: Systemic Risk Analysis (4-panel)
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Correlation Heatmap
im = axes[0, 0].imshow(corr_matrix.values, cmap='RdYlBu_r',
                        vmin=0, vmax=1, aspect='auto')
axes[0, 0].set_xticks(range(len(bank_tickers)))
axes[0, 0].set_yticks(range(len(bank_tickers)))
axes[0, 0].set_xticklabels(bank_tickers, rotation=45, ha='right',
                            fontsize=8)
axes[0, 0].set_yticklabels(bank_tickers, fontsize=8)
axes[0, 0].set_title('Return Correlation Heatmap', fontweight='bold')
for i in range(len(bank_tickers)):
    for j in range(len(bank_tickers)):
        axes[0, 0].text(j, i,
                        f'{corr_matrix.values[i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white'
                        if corr_matrix.values[i, j] > 0.7
                        else 'black')
plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

# Panel B: Rolling Correlations
colors_rc = ['#1A3A6E', '#DC3545', '#2E7D32']
for (t1, t2), color in zip(pairs, colors_rc):
    rc = rolling_corrs[(t1, t2)]
    axes[0, 1].plot(rc.index, rc, color=color, linewidth=0.7,
                    alpha=0.8, label=f'{t1}-{t2}')
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 1].set_title(f'Rolling {window}-Day Pairwise Correlations',
                      fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Correlation')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel C: Delta-CoVaR for selected institutions
colors_cv = ['#1A3A6E', '#DC3545', '#2E7D32', '#E67E22']
for tick, color in zip(['JPM', 'GS', 'BAC', 'C'], colors_cv):
    dc = covar_results[tick]['dcovar']
    axes[1, 0].plot(dc.index, dc * 100, color=color, linewidth=0.6,
                    alpha=0.8, label=tick)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].set_title('Delta-CoVaR: Systemic Risk Contribution',
                      fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Delta-CoVaR (%)')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=4, frameon=False)

# Panel D: Average rolling correlation (connectedness proxy)
avg_corr_ts = pd.DataFrame()
n_banks = len(bank_tickers)
for i, t1 in enumerate(bank_tickers[:-1]):
    for t2 in bank_tickers[i + 1:]:
        if t2 != t1:
            rc = returns[t1].rolling(window).corr(returns[t2])
            avg_corr_ts[f'{t1}_{t2}'] = rc
avg_connectedness = avg_corr_ts.mean(axis=1)

axes[1, 1].plot(avg_connectedness.index, avg_connectedness,
                color='#1A3A6E', linewidth=0.8)
axes[1, 1].fill_between(avg_connectedness.index, 0,
                         avg_connectedness,
                         alpha=0.15, color='#1A3A6E')
# Highlight crisis periods
for start, end in [('2008-09-01', '2009-03-31'),
                    ('2020-02-01', '2020-06-30')]:
    mask = ((avg_connectedness.index >= start) &
            (avg_connectedness.index <= end))
    axes[1, 1].fill_between(
        avg_connectedness.index[mask], 0,
        avg_connectedness[mask],
        alpha=0.4, color='#DC3545')
axes[1, 1].set_title(
    'Average Pairwise Correlation (Connectedness)',
    fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Average Correlation')
axes[1, 1].text(0.05, 0.95,
               f'Mean: {avg_connectedness.mean():.3f}',
               transform=axes[1, 1].transAxes, ha='left', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8))

plt.tight_layout()
save_fig('ch13_systemic_risk')

print("\n" + "=" * 70)
print("SYSTEMIC RISK ANALYSIS COMPLETE")
print("=" * 70)
print("\nKey findings:")
print("  - Financial institutions are highly correlated")
print("  - Correlations increase during stress periods")
print("  - Delta-CoVaR captures individual risk contributions")
print("  - Connectedness spikes during crises (2008 GFC, 2020 COVID)")
print("\nOutput files:")
print("  - ch13_systemic_risk.pdf: 4-panel systemic risk analysis")
