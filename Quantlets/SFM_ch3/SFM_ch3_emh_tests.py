"""
SFM_ch3_emh_tests
=================
Efficient Market Hypothesis: Weak-Form Tests

Description:
- Download stock data via yfinance
- Test weak-form EMH with Ljung-Box on returns
- Test weak-form EMH with Ljung-Box on squared returns
- Plot ACF of returns and ACF of squared returns
- Demonstrate: returns are nearly uncorrelated but
  squared returns exhibit significant autocorrelation

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    return runs, z, p_value

print("=" * 70)
print("SFM CHAPTER 3: EFFICIENT MARKET HYPOTHESIS TESTS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = ['SPY', 'AAPL', 'MSFT', 'GLD']
ticker_names = {'SPY': 'S&P 500 ETF', 'AAPL': 'Apple',
                'MSFT': 'Microsoft', 'GLD': 'Gold ETF'}

prices = yf.download(tickers, start='2010-01-01', end='2024-12-31',
                      progress=False)['Close']
returns = prices.pct_change().dropna()

for t in tickers:
    print(f"   {ticker_names[t]:15s} ({t}): {len(returns[t])} observations")

# =============================================================================
# 2. Ljung-Box Tests on Returns (Linear Dependence)
# =============================================================================
print("\n2. LJUNG-BOX TEST ON RETURNS (Linear Dependence)")
print("-" * 40)
print("   H0: No autocorrelation up to lag k")
print("   Rejection implies return predictability (EMH violation)\n")

lags_test = [5, 10, 20]

print(f"   {'Ticker':<8}", end="")
for lag in lags_test:
    print(f" {'Q(' + str(lag) + ')':>10} {'p-val':>8}", end="")
print()
print("   " + "-" * 62)

for t in tickers:
    ret = returns[t]
    lb = acorr_ljungbox(ret, lags=lags_test, return_df=True)
    print(f"   {t:<8}", end="")
    for i, lag in enumerate(lags_test):
        row = lb.iloc[i]
        print(f" {row['lb_stat']:>10.2f} {row['lb_pvalue']:>8.4f}",
              end="")
    print()

# =============================================================================
# 3. Ljung-Box Tests on Squared Returns (Non-linear Dependence)
# =============================================================================
print("\n3. LJUNG-BOX TEST ON SQUARED RETURNS (Non-linear Dependence)")
print("-" * 40)
print("   H0: No autocorrelation in squared returns")
print("   Rejection implies volatility clustering (ARCH effects)\n")

print(f"   {'Ticker':<8}", end="")
for lag in lags_test:
    print(f" {'Q(' + str(lag) + ')':>10} {'p-val':>8}", end="")
print()
print("   " + "-" * 62)

for t in tickers:
    ret = returns[t]
    lb = acorr_ljungbox(ret ** 2, lags=lags_test, return_df=True)
    print(f"   {t:<8}", end="")
    for i, lag in enumerate(lags_test):
        row = lb.iloc[i]
        print(f" {row['lb_stat']:>10.2f} {row['lb_pvalue']:>8.4f}",
              end="")
    print()

# =============================================================================
# 4. Runs Test
# =============================================================================
print("\n4. RUNS TEST FOR RANDOMNESS")
print("-" * 40)

print(f"   {'Asset':<8} {'Runs':>8} {'Z-stat':>10} {'p-value':>10} "
      f"{'Random':>10}")
print("   " + "-" * 50)
for t in tickers:
    r, z, p = runs_test(returns[t].values)
    random_str = "Yes" if p > 0.05 else "No"
    print(f"   {t:<8} {r:>8.0f} {z:>10.2f} {p:>10.4f} "
          f"{random_str:>10}")

# =============================================================================
# 5. FIGURE: ACF of Returns and Squared Returns
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

n_lags = 30
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Returns time series (S&P 500)
sp_ret = returns['SPY']
axes[0, 0].plot(sp_ret.index, sp_ret * 100, color='#1A3A6E',
                linewidth=0.5)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[0, 0].set_title('S&P 500 ETF Daily Returns (%)', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Return (%)')

# Panel B: ACF of returns
plot_acf(sp_ret.dropna(), lags=n_lags, ax=axes[0, 1], color='#1A3A6E',
         vlines_kwargs={'colors': '#1A3A6E', 'linewidths': 0.8},
         title='')
axes[0, 1].set_title('ACF of Returns (SPY)', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].text(0.95, 0.95, 'Nearly zero\n(EMH consistent)',
               transform=axes[0, 1].transAxes, ha='right', va='top',
               fontsize=8, color='#2E7D32',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel C: Squared returns time series
axes[1, 0].plot(sp_ret.index, (sp_ret * 100) ** 2, color='#DC3545',
                linewidth=0.5)
axes[1, 0].set_title('S&P 500 ETF Squared Returns', fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel(r'$r_t^2$ (%$^2$)')

# Panel D: ACF of squared returns
plot_acf(sp_ret.dropna() ** 2, lags=n_lags, ax=axes[1, 1],
         color='#DC3545',
         vlines_kwargs={'colors': '#DC3545', 'linewidths': 0.8},
         title='')
axes[1, 1].set_title('ACF of Squared Returns (SPY)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelation')
axes[1, 1].text(0.95, 0.95, 'Significant\n(Volatility clustering)',
               transform=axes[1, 1].transAxes, ha='right', va='top',
               fontsize=8, color='#DC3545',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig('ch3_emh_tests')

# =============================================================================
# 6. Summary and Interpretation
# =============================================================================
print("\n6. SUMMARY AND INTERPRETATION")
print("-" * 40)

print("\n   Key findings:")
print("   - Returns show little to no autocorrelation")
print("     => Consistent with weak-form EMH")
print("   - Squared returns show strong autocorrelation")
print("     => Volatility clustering (ARCH effects)")
print("   - EMH does not imply no volatility predictability")
print("   - Returns are uncorrelated but NOT independent")

# First-order autocorrelation of returns
print(f"\n   First-order autocorrelation:")
for t in tickers:
    rho1 = returns[t].autocorr(lag=1)
    print(f"   {ticker_names[t]:15s}: rho(1) = {rho1:.4f}")

print("\n" + "=" * 70)
print("EMH TESTS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch3_emh_tests.pdf: 4-panel ACF analysis (returns vs squared)")
