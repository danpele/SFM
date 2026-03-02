"""
SFM_ch7_garch
=============
ARCH/GARCH Models: Estimation and Volatility Forecasting

Description:
- Test for ARCH effects (Engle's LM test)
- Estimate ARCH(1), GARCH(1,1) models
- Compare conditional vs unconditional volatility
- Volatility forecasting
- Standardized residual diagnostics

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
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

print("=" * 70)
print("SFM CHAPTER 7: ARCH/GARCH MODELS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
returns = 100 * np.log(close / close.shift(1)).dropna()

print(f"   Ticker: {ticker}")
print(f"   Observations: {len(returns)}")
print(f"   Mean return: {returns.mean():.4f}%")
print(f"   Std return:  {returns.std():.4f}%")

# =============================================================================
# 2. Test for ARCH Effects
# =============================================================================
print("\n2. ARCH EFFECTS TEST (Engle's LM)")
print("-" * 40)

lm_stat, lm_pval, _, _ = het_arch(returns, nlags=10)
print(f"   LM statistic: {lm_stat:.2f}")
print(f"   p-value:       {lm_pval:.6f}")
print(f"   ARCH effects:  {'Yes' if lm_pval < 0.05 else 'No'}")

# =============================================================================
# 3. Estimate GARCH(1,1)
# =============================================================================
print("\n3. GARCH(1,1) ESTIMATION")
print("-" * 40)

garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
garch_res = garch.fit(disp='off')

print(garch_res.summary().tables[1])
print(f"\n   omega:  {garch_res.params['omega']:.6f}")
print(f"   alpha:  {garch_res.params['alpha[1]']:.6f}")
print(f"   beta:   {garch_res.params['beta[1]']:.6f}")
print(f"   alpha+beta: {garch_res.params['alpha[1]'] + garch_res.params['beta[1]']:.6f}")
print(f"   Log-likelihood: {garch_res.loglikelihood:.2f}")
print(f"   AIC: {garch_res.aic:.2f}")
print(f"   BIC: {garch_res.bic:.2f}")

# Unconditional volatility
omega = garch_res.params['omega']
alpha1 = garch_res.params['alpha[1]']
beta1 = garch_res.params['beta[1]']
uncond_var = omega / (1 - alpha1 - beta1)
uncond_vol = np.sqrt(uncond_var) * np.sqrt(252) / 100
print(f"   Unconditional vol (ann.): {uncond_vol:.4f}")

# =============================================================================
# 4. Estimate ARCH(5) for comparison
# =============================================================================
print("\n4. ARCH(5) ESTIMATION")
print("-" * 40)

arch5 = arch_model(returns, vol='ARCH', p=5, dist='normal')
arch5_res = arch5.fit(disp='off')
print(f"   Log-likelihood: {arch5_res.loglikelihood:.2f}")
print(f"   AIC: {arch5_res.aic:.2f}")
print(f"   BIC: {arch5_res.bic:.2f}")

# =============================================================================
# 5. FIGURE: GARCH Analysis (4-panel)
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

cond_vol = garch_res.conditional_volatility
std_resid = garch_res.std_resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Returns with conditional volatility
axes[0, 0].plot(returns.index, returns, color='#1A3A6E', linewidth=0.3, alpha=0.6)
axes[0, 0].plot(returns.index, cond_vol, color='#DC3545', linewidth=0.8, label='Cond. Vol (σ)')
axes[0, 0].plot(returns.index, -cond_vol, color='#DC3545', linewidth=0.8)
axes[0, 0].set_title(f'{ticker}: Returns and GARCH(1,1) Conditional Volatility', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Return (%)')
axes[0, 0].legend(loc='upper right', fontsize=8)

# Panel B: Conditional vs unconditional volatility
axes[0, 1].plot(returns.index, cond_vol * np.sqrt(252) / 100, color='#1A3A6E',
               linewidth=0.8, label='Conditional (ann.)')
axes[0, 1].axhline(y=uncond_vol, color='#DC3545', linestyle='--', linewidth=1,
                   label=f'Unconditional ({uncond_vol:.2%})')
axes[0, 1].set_title('Conditional vs Unconditional Volatility', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Annualized Volatility')
axes[0, 1].legend(loc='upper right', fontsize=8)

# Panel C: Standardized residuals distribution
axes[1, 0].hist(std_resid, bins=80, density=True, alpha=0.5, color='#1A3A6E',
                edgecolor='white', label='Std. Residuals')
x = np.linspace(-5, 5, 200)
axes[1, 0].plot(x, stats.norm.pdf(x), color='#DC3545', linewidth=1.5, label='N(0,1)')
axes[1, 0].set_title('Standardized Residuals Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Standardized Residual')
axes[1, 0].set_ylabel('Density')
kurt_std = stats.kurtosis(std_resid.dropna())
axes[1, 0].text(0.95, 0.95, f'Kurtosis={kurt_std:.2f}',
               transform=axes[1, 0].transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].legend(loc='upper left', fontsize=8)

# Panel D: News Impact Curve
sigma2_uncond = uncond_var
eps_range = np.linspace(-5, 5, 200)
sigma2_next = omega + alpha1 * eps_range**2 + beta1 * sigma2_uncond
axes[1, 1].plot(eps_range, np.sqrt(sigma2_next), color='#1A3A6E', linewidth=1.5)
axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 1].set_title('News Impact Curve: GARCH(1,1)', fontweight='bold')
axes[1, 1].set_xlabel('Shock (ε)')
axes[1, 1].set_ylabel('Next-Period Volatility (σ)')

plt.tight_layout()
save_fig('ch7_garch')

print("\n" + "=" * 70)
print("ARCH/GARCH ANALYSIS COMPLETE")
print("=" * 70)
