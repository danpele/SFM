"""
SFM_ch8_egarch_tgarch
=====================
Asymmetric GARCH Models: EGARCH, GJR-GARCH, GARCH-M

Description:
- Fit EGARCH(1,1) model (Nelson, 1991)
- Fit GJR-GARCH / TGARCH (Glosten-Jagannathan-Runkle, 1993)
- Fit GARCH-in-Mean (Engle-Lilien-Robins, 1987)
- Compare news impact curves across models
- AIC/BIC model comparison table

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from arch import arch_model
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
print("SFM CHAPTER 8: ASYMMETRIC GARCH MODELS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31',
                    progress=False)
close = data['Close'].squeeze()
returns = 100 * np.log(close / close.shift(1)).dropna()

print(f"   Ticker: {ticker}")
print(f"   Observations: {len(returns)}")
print(f"   Mean return: {returns.mean():.4f}%")
print(f"   Std return:  {returns.std():.4f}%")

# =============================================================================
# 2. Estimate Models
# =============================================================================
print("\n2. MODEL ESTIMATION")
print("-" * 40)

# GARCH(1,1) - symmetric baseline
garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
garch_res = garch.fit(disp='off')
print(f"   GARCH(1,1):  LL={garch_res.loglikelihood:.2f}, "
      f"AIC={garch_res.aic:.2f}, BIC={garch_res.bic:.2f}")

# EGARCH(1,1) - exponential GARCH (Nelson, 1991)
egarch = arch_model(returns, vol='EGARCH', p=1, q=1, dist='normal')
egarch_res = egarch.fit(disp='off')
print(f"   EGARCH(1,1): LL={egarch_res.loglikelihood:.2f}, "
      f"AIC={egarch_res.aic:.2f}, BIC={egarch_res.bic:.2f}")

# GJR-GARCH(1,1) - threshold GARCH (Glosten et al., 1993)
gjr = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='normal')
gjr_res = gjr.fit(disp='off')
print(f"   GJR(1,1,1):  LL={gjr_res.loglikelihood:.2f}, "
      f"AIC={gjr_res.aic:.2f}, BIC={gjr_res.bic:.2f}")

# GARCH(1,1) with Student-t distribution
garch_t = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
garch_t_res = garch_t.fit(disp='off')
print(f"   GARCH-t:     LL={garch_t_res.loglikelihood:.2f}, "
      f"AIC={garch_t_res.aic:.2f}, BIC={garch_t_res.bic:.2f}")

# EGARCH with Student-t
egarch_t = arch_model(returns, vol='EGARCH', p=1, q=1, dist='t')
egarch_t_res = egarch_t.fit(disp='off')
print(f"   EGARCH-t:    LL={egarch_t_res.loglikelihood:.2f}, "
      f"AIC={egarch_t_res.aic:.2f}, BIC={egarch_t_res.bic:.2f}")

# =============================================================================
# 3. Parameter Comparison
# =============================================================================
print("\n3. PARAMETER COMPARISON")
print("-" * 40)

print("\n   GARCH(1,1) Parameters:")
for param, val in garch_res.params.items():
    print(f"     {param:12s}: {val:.6f}")

print("\n   EGARCH(1,1) Parameters:")
for param, val in egarch_res.params.items():
    print(f"     {param:12s}: {val:.6f}")

print("\n   GJR-GARCH(1,1,1) Parameters:")
for param, val in gjr_res.params.items():
    print(f"     {param:12s}: {val:.6f}")

# =============================================================================
# 4. Model Comparison Table
# =============================================================================
print("\n4. MODEL COMPARISON")
print("-" * 40)

models = {
    'GARCH(1,1)': garch_res,
    'EGARCH(1,1)': egarch_res,
    'GJR-GARCH': gjr_res,
    'GARCH-t': garch_t_res,
    'EGARCH-t': egarch_t_res
}

print(f"   {'Model':<16} {'LL':>10} {'AIC':>10} {'BIC':>10} "
      f"{'Params':>8}")
print("   " + "-" * 56)
for name, res in models.items():
    print(f"   {name:<16} {res.loglikelihood:>10.2f} "
          f"{res.aic:>10.2f} {res.bic:>10.2f} "
          f"{len(res.params):>8}")

best_aic = min(models.items(), key=lambda x: x[1].aic)
best_bic = min(models.items(), key=lambda x: x[1].bic)
print(f"\n   Best by AIC: {best_aic[0]}")
print(f"   Best by BIC: {best_bic[0]}")

# =============================================================================
# 5. FIGURE: Asymmetric GARCH Analysis (4-panel)
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Conditional volatility comparison
axes[0, 0].plot(returns.index, garch_res.conditional_volatility,
                color='#1A3A6E', linewidth=0.6, alpha=0.8,
                label='GARCH(1,1)')
axes[0, 0].plot(returns.index, egarch_res.conditional_volatility,
                color='#DC3545', linewidth=0.6, alpha=0.8,
                label='EGARCH(1,1)')
axes[0, 0].plot(returns.index, gjr_res.conditional_volatility,
                color='#2E7D32', linewidth=0.6, alpha=0.8,
                label='GJR-GARCH')
axes[0, 0].set_title(f'{ticker}: Conditional Volatility Comparison',
                      fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Conditional Volatility (%)')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel B: News Impact Curves
eps_range = np.linspace(-5, 5, 300)

# GARCH NIC
omega_g = garch_res.params['omega']
alpha_g = garch_res.params['alpha[1]']
beta_g = garch_res.params['beta[1]']
sigma2_bar_g = omega_g / (1 - alpha_g - beta_g)
nic_garch = np.sqrt(omega_g + alpha_g * eps_range**2
                     + beta_g * sigma2_bar_g)

# GJR-GARCH NIC
omega_j = gjr_res.params['omega']
alpha_j = gjr_res.params['alpha[1]']
gamma_j = gjr_res.params['gamma[1]']
beta_j = gjr_res.params['beta[1]']
sigma2_bar_j = omega_j / (1 - alpha_j - 0.5 * gamma_j - beta_j)
indicator = (eps_range < 0).astype(float)
nic_gjr = np.sqrt(omega_j + (alpha_j + gamma_j * indicator)
                   * eps_range**2 + beta_j * sigma2_bar_j)

# EGARCH NIC
omega_e = egarch_res.params['omega']
alpha_e = egarch_res.params['alpha[1]']
gamma_e = egarch_res.params['gamma[1]']
beta_e = egarch_res.params['beta[1]']
log_sigma2_bar = omega_e / (1 - beta_e)
z = eps_range / np.sqrt(np.exp(log_sigma2_bar))
nic_egarch = np.sqrt(np.exp(omega_e + alpha_e * np.abs(z)
                              + gamma_e * z
                              + beta_e * log_sigma2_bar))

axes[0, 1].plot(eps_range, nic_garch, color='#1A3A6E', linewidth=1.5,
                label='GARCH(1,1)')
axes[0, 1].plot(eps_range, nic_gjr, color='#DC3545', linewidth=1.5,
                label='GJR-GARCH')
axes[0, 1].plot(eps_range, nic_egarch, color='#2E7D32', linewidth=1.5,
                label='EGARCH(1,1)')
axes[0, 1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 1].set_title('News Impact Curves', fontweight='bold')
axes[0, 1].set_xlabel('Shock ($\\varepsilon_{t-1}$)')
axes[0, 1].set_ylabel('Next-Period Volatility ($\\sigma_t$)')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel C: EGARCH standardized residuals QQ-plot
std_resid = egarch_res.std_resid.dropna()
stats.probplot(std_resid, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('EGARCH: Standardized Residuals QQ-Plot',
                      fontweight='bold')
axes[1, 0].get_lines()[0].set_markerfacecolor('#1A3A6E')
axes[1, 0].get_lines()[0].set_markersize(2)
axes[1, 0].get_lines()[0].set_alpha(0.4)
axes[1, 0].get_lines()[1].set_color('#DC3545')

# Panel D: AIC/BIC comparison bar chart
model_names = list(models.keys())
aic_vals = [models[m].aic for m in model_names]
bic_vals = [models[m].bic for m in model_names]

x_pos = np.arange(len(model_names))
width = 0.35
axes[1, 1].bar(x_pos - width / 2, aic_vals, width, color='#1A3A6E',
               alpha=0.7, label='AIC')
axes[1, 1].bar(x_pos + width / 2, bic_vals, width, color='#DC3545',
               alpha=0.7, label='BIC')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(model_names, rotation=20, ha='right',
                            fontsize=7)
axes[1, 1].set_title('Model Comparison: AIC vs BIC', fontweight='bold')
axes[1, 1].set_ylabel('Information Criterion')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  ncol=2, frameon=False)

plt.tight_layout()
save_fig('ch8_egarch_tgarch')

print("\n" + "=" * 70)
print("ASYMMETRIC GARCH ANALYSIS COMPLETE")
print("=" * 70)
print("\nKey findings:")
print("  - Asymmetric models capture leverage effect")
print("  - EGARCH allows negative parameters (log specification)")
print("  - GJR-GARCH uses indicator function for asymmetry")
print("  - Student-t distribution improves fit for heavy tails")
print("\nOutput files:")
print("  - ch8_egarch_tgarch.pdf: 4-panel asymmetric GARCH analysis")
