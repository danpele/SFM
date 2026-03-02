"""
SFM_ch2_stable_fit
==================
Stable Distribution Fitting and Risk Analysis

Description:
- Plot PDFs for different alpha values (linear + log scale)
- Fit stable distribution to S&P 500 returns
- Compare VaR: stable vs normal vs empirical
- Survival function (log-log) for tail comparison
- Cross-asset alpha comparison bar chart

Statistics of Financial Markets course
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levy_stable
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

print("=" * 70)
print("SFM CHAPTER 2: STABLE FIT & RISK ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Stable PDFs — linear + log scale (2 panels)
# =============================================================================
print("\n1. STABLE DISTRIBUTION PDFs")
print("-" * 40)

alphas = [0.5, 1.0, 1.5, 2.0]
colors_alpha = [CRIMSON, ORANGE, FOREST, MAIN_BLUE]
labels_alpha = [r'$\alpha=0.5$ (L\'evy)',
                r'$\alpha=1.0$ (Cauchy)',
                r'$\alpha=1.5$',
                r'$\alpha=2.0$ (Gaussian)']

x = np.linspace(-8, 8, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for a, c, lab in zip(alphas, colors_alpha, labels_alpha):
    rv = levy_stable(alpha=a, beta=0)
    pdf_vals = rv.pdf(x)
    ax1.plot(x, pdf_vals, color=c, linewidth=1.2, label=lab)
    ax2.semilogy(x, pdf_vals, color=c, linewidth=1.2, label=lab)

ax1.set_title('A. Stable PDFs (linear scale)', fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_ylim(0, 0.45)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=False)

ax2.set_title('B. Stable PDFs (log scale)', fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('log f(x)')
ax2.set_ylim(1e-6, 1)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
save_fig('ch2_stable_pdfs')
print("   Panel A: linear PDF, Panel B: log-scale PDF")

# =============================================================================
# 2. S&P 500 stable fit (histogram + stable PDF + normal PDF, log-scale y)
# =============================================================================
print("\n2. S&P 500 STABLE FIT")
print("-" * 40)

import yfinance as yf

print("   Downloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                     progress=False)
sp500_close = sp500['Close'].squeeze()
log_returns = np.log(sp500_close / sp500_close.shift(1)).dropna().values

print(f"   Observations: {len(log_returns)}")
print(f"   Mean: {np.mean(log_returns):.6f}")
print(f"   Std:  {np.std(log_returns):.6f}")
print(f"   Kurt: {stats.kurtosis(log_returns):.2f}")

# Fit stable distribution
print("   Fitting stable distribution (this may take a moment)...")
alpha_fit, beta_fit, loc_fit, scale_fit = levy_stable._fitstart(log_returns)
try:
    params = levy_stable.fit(log_returns, floc=np.mean(log_returns))
    alpha_fit, beta_fit, loc_fit, scale_fit = params
except Exception:
    pass
print(f"   Stable fit: alpha={alpha_fit:.4f}, beta={beta_fit:.4f}, "
      f"gamma={scale_fit:.6f}, delta={loc_fit:.6f}")

# Fit normal
mu_norm, sigma_norm = stats.norm.fit(log_returns)
print(f"   Normal fit: mu={mu_norm:.6f}, sigma={sigma_norm:.6f}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

x_range = np.linspace(np.min(log_returns) * 1.2, np.max(log_returns) * 1.2, 1000)

ax.hist(log_returns, bins=150, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')

pdf_stable = levy_stable.pdf(x_range, alpha_fit, beta_fit,
                              loc=loc_fit, scale=scale_fit)
pdf_normal = stats.norm.pdf(x_range, loc=mu_norm, scale=sigma_norm)

ax.semilogy(x_range, pdf_stable, color=CRIMSON, linewidth=1.5,
            label=f'Stable ($\\alpha$={alpha_fit:.2f})')
ax.semilogy(x_range, pdf_normal, color=FOREST, linewidth=1.5,
            linestyle='--', label='Normal')

# Re-draw histogram on log scale
ax.cla()
ax.hist(log_returns, bins=150, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')
ax.plot(x_range, pdf_stable, color=CRIMSON, linewidth=1.5,
        label=f'Stable ($\\alpha$={alpha_fit:.2f})')
ax.plot(x_range, pdf_normal, color=FOREST, linewidth=1.5,
        linestyle='--', label='Normal')
ax.set_yscale('log')
ax.set_ylim(1e-3, None)

ax.set_title('S\\&P 500 Log-Returns: Stable vs Normal Fit', fontweight='bold')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density (log scale)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False)

ax.text(0.02, 0.95,
        f'$\\alpha$={alpha_fit:.3f}\n$\\beta$={beta_fit:.3f}\n'
        f'n={len(log_returns):,}',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_sp500_stable_fit')

# =============================================================================
# 3. VaR Comparison — histogram + vertical VaR lines
# =============================================================================
print("\n3. VaR COMPARISON")
print("-" * 40)

confidence_levels = [0.01, 0.05, 0.001]

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(log_returns, bins=150, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')

# VaR lines at 1% level
var_stable = levy_stable.ppf(0.01, alpha_fit, beta_fit,
                              loc=loc_fit, scale=scale_fit)
var_normal = stats.norm.ppf(0.01, loc=mu_norm, scale=sigma_norm)
var_empirical = np.percentile(log_returns, 1)

ax.axvline(var_stable, color=CRIMSON, linewidth=1.5, linestyle='-',
           label=f'VaR 1% Stable ({var_stable:.4f})')
ax.axvline(var_normal, color=FOREST, linewidth=1.5, linestyle='--',
           label=f'VaR 1% Normal ({var_normal:.4f})')
ax.axvline(var_empirical, color=ORANGE, linewidth=1.5, linestyle='-.',
           label=f'VaR 1% Empirical ({var_empirical:.4f})')

ax.set_title('VaR Comparison: Stable vs Normal vs Empirical', fontweight='bold')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density')
ax.set_xlim(np.percentile(log_returns, 0.1), np.percentile(log_returns, 99.9))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, frameon=False, fontsize=7)

# Table annotation
var_text = "VaR Comparison (portfolio = 1M EUR)\n"
var_text += f"{'Level':>6} {'Stable':>10} {'Normal':>10} {'Empir.':>10}\n"
for cl in confidence_levels:
    vs = levy_stable.ppf(cl, alpha_fit, beta_fit, loc=loc_fit, scale=scale_fit)
    vn = stats.norm.ppf(cl, loc=mu_norm, scale=sigma_norm)
    ve = np.percentile(log_returns, cl * 100)
    var_text += f"{cl*100:>5.1f}% {vs*1e6:>10,.0f} {vn*1e6:>10,.0f} {ve*1e6:>10,.0f}\n"
    print(f"   VaR {cl*100:.1f}%: Stable={vs:.5f}, Normal={vn:.5f}, Empirical={ve:.5f}")

ax.text(0.97, 0.95, var_text.strip(), transform=ax.transAxes,
        fontsize=6, va='top', ha='right', family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_var_comparison')

# =============================================================================
# 4. Survival function log-log: Stable(1.7) vs Normal
# =============================================================================
print("\n4. TAIL SURVIVAL FUNCTION")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 5))

# Right tail: P(X > x)
x_tail = np.linspace(0.01, 8, 500)
stable_17 = levy_stable(alpha=1.7, beta=0)

surv_stable = stable_17.sf(x_tail)  # 1 - CDF
surv_normal = stats.norm.sf(x_tail)

ax.loglog(x_tail, surv_stable, color=CRIMSON, linewidth=1.5,
          label=r'Stable ($\alpha=1.7$)')
ax.loglog(x_tail, surv_normal, color=MAIN_BLUE, linewidth=1.5,
          linestyle='--', label='Normal')

# Theoretical power law reference
surv_power = 0.3 * x_tail ** (-1.7)
ax.loglog(x_tail[50:], surv_power[50:], color=FOREST, linewidth=0.8,
          linestyle=':', alpha=0.7, label=r'$\sim x^{-1.7}$ (reference)')

# Mark key thresholds
for thresh, lbl in [(3, r'$3\sigma$'), (5, r'$5\sigma$')]:
    s_val = stable_17.sf(thresh)
    n_val = stats.norm.sf(thresh)
    ratio = s_val / n_val if n_val > 0 else np.inf
    ax.axvline(thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.text(thresh * 1.05, 0.3, lbl, fontsize=7, color='gray')
    print(f"   At x={thresh}: Stable P={s_val:.6f}, Normal P={n_val:.8f}, "
          f"Ratio={ratio:.1f}x")

ax.set_title('Survival Function: Stable(1.7) vs Normal', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('P(X > x)')
ax.set_xlim(0.1, 10)
ax.set_ylim(1e-8, 1)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_tail_survival')

# =============================================================================
# 5. Cross-asset alpha comparison (bar chart)
# =============================================================================
print("\n5. CROSS-ASSET ALPHA COMPARISON")
print("-" * 40)

print("   Downloading asset data...")
tickers = {
    'S&P 500': '^GSPC',
    'Bitcoin': 'BTC-USD',
    'EUR/USD': 'EURUSD=X',
    'Gold': 'GC=F',
    'Crude Oil': 'CL=F',
    'US 10Y Bond': '^TNX'
}

asset_alphas = {}
for name, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2015-01-01', end='2025-12-31',
                           progress=False)
        close = data['Close'].squeeze().dropna()
        rets = np.log(close / close.shift(1)).dropna().values
        rets = rets[np.isfinite(rets)]
        if len(rets) > 100:
            a_est, _, _, _ = levy_stable._fitstart(rets)
            asset_alphas[name] = a_est
            print(f"   {name:>15}: alpha = {a_est:.4f} (n={len(rets)})")
    except Exception as e:
        print(f"   {name:>15}: FAILED ({e})")

fig, ax = plt.subplots(figsize=(7, 4))

names = list(asset_alphas.keys())
alpha_vals = list(asset_alphas.values())
bar_colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER, ORANGE,
              '#8E44AD'][:len(names)]

bars = ax.bar(range(len(names)), alpha_vals, color=bar_colors,
              width=0.6, edgecolor='white', linewidth=0.5)

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, alpha_vals)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7,
            fontweight='bold')

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=7)
ax.set_ylabel(r'Stability index $\alpha$')
ax.set_title('Stability Index by Asset Class', fontweight='bold')
ax.set_ylim(0, 2.2)

# Reference lines
ax.axhline(2.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
ax.text(len(names) - 0.5, 2.02, 'Gaussian', fontsize=7,
        color='gray', ha='right')

plt.tight_layout()
save_fig('ch2_cross_asset_alpha')

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("STABLE FIT & RISK ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_stable_pdfs.pdf:         Stable PDFs (linear + log)")
print("  - ch2_sp500_stable_fit.pdf:    S&P 500 stable vs normal fit")
print("  - ch2_var_comparison.pdf:      VaR comparison chart")
print("  - ch2_tail_survival.pdf:       Survival function (log-log)")
print("  - ch2_cross_asset_alpha.pdf:   Cross-asset alpha bar chart")
