"""
SFM_ch9_var_es
==============
Value-at-Risk and Expected Shortfall

Description:
- Compute VaR and ES using Historical Simulation
- Compute VaR and ES using Parametric (Normal) method
- Compute VaR and ES using Monte Carlo simulation
- Backtest with rolling window (Kupiec test)
- Compare methods visually

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

def var_historical(returns, alpha=0.05):
    """Historical simulation VaR at confidence level (1-alpha)."""
    return np.percentile(returns, alpha * 100)

def es_historical(returns, alpha=0.05):
    """Historical simulation Expected Shortfall."""
    var = var_historical(returns, alpha)
    return returns[returns <= var].mean()

def var_parametric(returns, alpha=0.05):
    """Parametric (Gaussian) VaR."""
    mu = returns.mean()
    sigma = returns.std()
    return mu + sigma * stats.norm.ppf(alpha)

def es_parametric(returns, alpha=0.05):
    """Parametric (Gaussian) Expected Shortfall."""
    mu = returns.mean()
    sigma = returns.std()
    return mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

def var_montecarlo(returns, alpha=0.05, n_sims=10000):
    """Monte Carlo VaR assuming normal distribution."""
    mu = returns.mean()
    sigma = returns.std()
    simulated = np.random.normal(mu, sigma, n_sims)
    return np.percentile(simulated, alpha * 100)

def es_montecarlo(returns, alpha=0.05, n_sims=10000):
    """Monte Carlo Expected Shortfall."""
    mu = returns.mean()
    sigma = returns.std()
    simulated = np.random.normal(mu, sigma, n_sims)
    var = np.percentile(simulated, alpha * 100)
    return simulated[simulated <= var].mean()

def kupiec_test(violations, n_obs, alpha=0.05):
    """
    Kupiec (1995) POF test for VaR backtesting.
    H0: violation rate = alpha.
    """
    n_viol = np.sum(violations)
    p_hat = n_viol / n_obs
    if p_hat == 0 or p_hat == 1:
        return np.nan, np.nan
    lr = -2 * (n_viol * np.log(alpha / p_hat) +
               (n_obs - n_viol) * np.log((1 - alpha) / (1 - p_hat)))
    p_value = 1 - stats.chi2.cdf(lr, 1)
    return lr, p_value

print("=" * 70)
print("SFM CHAPTER 9: VALUE-AT-RISK AND EXPECTED SHORTFALL")
print("=" * 70)

np.random.seed(42)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31',
                    progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Ticker: {ticker}")
print(f"   Observations: {len(log_ret)}")

# =============================================================================
# 2. Point Estimates (full sample)
# =============================================================================
print("\n2. POINT ESTIMATES (full sample)")
print("-" * 40)

alpha = 0.05
ret_vals = log_ret.values

print(f"   Confidence level: {(1 - alpha) * 100:.0f}%")
print(f"\n   {'Method':<16} {'VaR':>10} {'ES':>10}")
print("   " + "-" * 38)

var_h = var_historical(ret_vals, alpha)
es_h = es_historical(ret_vals, alpha)
print(f"   {'Historical':<16} {var_h:>10.6f} {es_h:>10.6f}")

var_p = var_parametric(ret_vals, alpha)
es_p = es_parametric(ret_vals, alpha)
print(f"   {'Parametric':<16} {var_p:>10.6f} {es_p:>10.6f}")

var_m = var_montecarlo(ret_vals, alpha)
es_m = es_montecarlo(ret_vals, alpha)
print(f"   {'Monte Carlo':<16} {var_m:>10.6f} {es_m:>10.6f}")

# =============================================================================
# 3. Rolling Window Backtest
# =============================================================================
print("\n3. ROLLING WINDOW BACKTEST")
print("-" * 40)

window = 252  # 1-year rolling window
n_test = len(log_ret) - window

var_hist_roll = np.zeros(n_test)
var_para_roll = np.zeros(n_test)
var_mc_roll = np.zeros(n_test)
actual = np.zeros(n_test)

for i in range(n_test):
    hist_window = ret_vals[i:i + window]
    var_hist_roll[i] = var_historical(hist_window, alpha)
    var_para_roll[i] = var_parametric(hist_window, alpha)
    var_mc_roll[i] = var_montecarlo(hist_window, alpha)
    actual[i] = ret_vals[i + window]

# Violation analysis
viol_hist = actual < var_hist_roll
viol_para = actual < var_para_roll
viol_mc = actual < var_mc_roll

methods = ['Historical', 'Parametric', 'Monte Carlo']
violations = [viol_hist, viol_para, viol_mc]

print(f"   {'Method':<16} {'Violations':>12} {'Rate':>8} "
      f"{'Expected':>10} {'Kupiec LR':>12} {'p-value':>10}")
print("   " + "-" * 72)
for name, viol in zip(methods, violations):
    n_v = np.sum(viol)
    rate = n_v / n_test
    lr, pval = kupiec_test(viol, n_test, alpha)
    reject = " *" if pval < 0.05 else ""
    print(f"   {name:<16} {n_v:>12d} {rate:>8.4f} "
          f"{alpha:>10.4f} {lr:>12.4f} {pval:>10.4f}{reject}")

# =============================================================================
# 4. FIGURE: VaR and ES Analysis (4-panel)
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
dates = log_ret.index[window:]

# Panel A: Returns with Historical VaR
axes[0, 0].plot(dates, actual, color='#1A3A6E', linewidth=0.3,
                alpha=0.7, label='Returns')
axes[0, 0].plot(dates, var_hist_roll, color='#DC3545', linewidth=0.8,
                label=f'VaR ({(1 - alpha) * 100:.0f}%)')
breach_idx = np.where(viol_hist)[0]
axes[0, 0].scatter(dates[breach_idx], actual[breach_idx],
                    color='#DC3545', s=8, zorder=5, alpha=0.6,
                    label=f'Breaches ({np.sum(viol_hist)})')
axes[0, 0].set_title('Historical Simulation VaR', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Log Return')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel B: All three VaR methods compared
axes[0, 1].plot(dates, var_hist_roll, color='#1A3A6E', linewidth=0.8,
                label='Historical')
axes[0, 1].plot(dates, var_para_roll, color='#DC3545', linewidth=0.8,
                label='Parametric')
axes[0, 1].plot(dates, var_mc_roll, color='#2E7D32', linewidth=0.8,
                alpha=0.8, label='Monte Carlo')
axes[0, 1].set_title('VaR Method Comparison', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel(f'VaR ({(1 - alpha) * 100:.0f}%)')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel C: Distribution with VaR and ES marked
axes[1, 0].hist(ret_vals, bins=100, density=True, alpha=0.5,
                color='#1A3A6E', edgecolor='white')
axes[1, 0].axvline(x=var_h, color='#DC3545', linewidth=2,
                    linestyle='--', label=f'VaR = {var_h:.4f}')
axes[1, 0].axvline(x=es_h, color='#E67E22', linewidth=2,
                    linestyle='--', label=f'ES = {es_h:.4f}')
# Shade the tail
x_tail = np.linspace(ret_vals.min(), var_h, 100)
mu_r, sigma_r = ret_vals.mean(), ret_vals.std()
axes[1, 0].fill_between(x_tail,
                         stats.norm.pdf(x_tail, mu_r, sigma_r),
                         alpha=0.3, color='#DC3545')
axes[1, 0].set_title('Return Distribution with VaR and ES',
                      fontweight='bold')
axes[1, 0].set_xlabel('Log Return')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)

# Panel D: Cumulative violations over time
cum_viol_hist = np.cumsum(viol_hist)
cum_viol_para = np.cumsum(viol_para)
expected_line = alpha * np.arange(1, n_test + 1)
axes[1, 1].plot(dates, cum_viol_hist, color='#1A3A6E', linewidth=1,
                label='Historical')
axes[1, 1].plot(dates, cum_viol_para, color='#DC3545', linewidth=1,
                label='Parametric')
axes[1, 1].plot(dates, expected_line, color='gray', linestyle='--',
                linewidth=1, label=f'Expected ({alpha * 100:.0f}%)')
axes[1, 1].set_title('Cumulative VaR Violations', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cumulative Violations')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch9_var_es')

print("\n" + "=" * 70)
print("VaR AND ES ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch9_var_es.pdf: 4-panel VaR/ES analysis")
