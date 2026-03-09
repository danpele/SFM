"""
SFM_ch2_student_t_fit
=====================
Student-t Distribution Fitting and Tail Analysis

Description:
- Plot Student-t PDFs for various df vs Normal (linear + log scale)
- Fit Student-t distribution to S&P 500 returns via MLE
- Compare Student-t vs Normal fit with AIC/BIC
- Estimate tail probabilities under Student-t vs Normal

Statistics of Financial Markets course — Section 2.4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as student_t
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
print("SFM CHAPTER 2: STUDENT-t FIT & TAIL ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Student-t PDFs — linear + log scale (2 panels)
# =============================================================================
print("\n1. STUDENT-t DISTRIBUTION PDFs")
print("-" * 40)

degrees = [3, 5, 10, 30]
colors_df = [CRIMSON, ORANGE, FOREST, AMBER]
labels_df = [r'$\nu=3$', r'$\nu=5$', r'$\nu=10$', r'$\nu=30$']

x = np.linspace(-6, 6, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Normal reference on both panels
pdf_normal = stats.norm.pdf(x)
ax1.plot(x, pdf_normal, color=MAIN_BLUE, linewidth=1.5, linestyle='--',
         label='Normal')
ax2.semilogy(x, pdf_normal, color=MAIN_BLUE, linewidth=1.5, linestyle='--',
             label='Normal')

for df, c, lab in zip(degrees, colors_df, labels_df):
    pdf_vals = student_t.pdf(x, df)
    ax1.plot(x, pdf_vals, color=c, linewidth=1.2, label=lab)
    ax2.semilogy(x, pdf_vals, color=c, linewidth=1.2, label=lab)
    print(f"   df={df:>2}: peak={pdf_vals.max():.4f}, "
          f"P(|X|>3)={2*student_t.sf(3, df):.6f}")

ax1.set_title('A. Student-t PDFs (linear scale)', fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_ylim(0, 0.45)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=3, frameon=False)

ax2.set_title('B. Student-t PDFs (log scale)', fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('log f(x)')
ax2.set_ylim(1e-6, 1)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
save_fig('ch2_student_t_pdfs')
print("   Panel A: linear PDF, Panel B: log-scale PDF")

# =============================================================================
# 2. S&P 500 Student-t fit (histogram + t PDF + normal PDF, log-scale y)
# =============================================================================
print("\n2. S&P 500 STUDENT-t FIT")
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
print(f"   Skew: {stats.skew(log_returns):.4f}")
print(f"   Kurt: {stats.kurtosis(log_returns):.2f}")

# Fit Student-t via MLE
print("   Fitting Student-t distribution (MLE)...")
df_fit, loc_fit, scale_fit = student_t.fit(log_returns)
print(f"   Student-t fit: df={df_fit:.4f}, loc={loc_fit:.6f}, "
      f"scale={scale_fit:.6f}")

# Fit normal
mu_norm, sigma_norm = stats.norm.fit(log_returns)
print(f"   Normal fit:    mu={mu_norm:.6f}, sigma={sigma_norm:.6f}")

# AIC / BIC comparison
n = len(log_returns)
# Student-t: 3 parameters (df, loc, scale)
ll_t = np.sum(student_t.logpdf(log_returns, df_fit, loc=loc_fit,
                                scale=scale_fit))
aic_t = -2 * ll_t + 2 * 3
bic_t = -2 * ll_t + np.log(n) * 3

# Normal: 2 parameters (mu, sigma)
ll_n = np.sum(stats.norm.logpdf(log_returns, loc=mu_norm, scale=sigma_norm))
aic_n = -2 * ll_n + 2 * 2
bic_n = -2 * ll_n + np.log(n) * 2

print(f"\n   Model comparison (n={n:,}):")
print(f"   {'':>12} {'Log-Lik':>12} {'AIC':>12} {'BIC':>12}")
print(f"   {'Student-t':>12} {ll_t:>12,.1f} {aic_t:>12,.1f} {bic_t:>12,.1f}")
print(f"   {'Normal':>12} {ll_n:>12,.1f} {aic_n:>12,.1f} {bic_n:>12,.1f}")
print(f"   {'Difference':>12} {ll_t - ll_n:>12,.1f} {aic_t - aic_n:>12,.1f} "
      f"{bic_t - bic_n:>12,.1f}")

if aic_t < aic_n:
    print("   => Student-t preferred by both AIC and BIC")
else:
    print("   => Normal preferred by AIC")

# Tail probabilities
print(f"\n   Tail probabilities:")
print(f"   {'Event':>20} {'Student-t':>14} {'Normal':>14} {'Ratio':>10}")
for k in [3, 4, 5, 6]:
    p_t = 2 * student_t.sf(k, df_fit, loc=0, scale=1)
    p_n = 2 * stats.norm.sf(k)
    ratio = p_t / p_n if p_n > 0 else np.inf
    print(f"   {'|X| > ' + str(k) + ' sigma':>20} {p_t:>14.8f} "
          f"{p_n:>14.8f} {ratio:>9.1f}x")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

x_range = np.linspace(np.min(log_returns) * 1.2,
                       np.max(log_returns) * 1.2, 1000)

ax.hist(log_returns, bins=150, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')

pdf_t = student_t.pdf(x_range, df_fit, loc=loc_fit, scale=scale_fit)
pdf_normal = stats.norm.pdf(x_range, loc=mu_norm, scale=sigma_norm)

ax.plot(x_range, pdf_t, color=CRIMSON, linewidth=1.5,
        label=f'Student-t ($\\nu$={df_fit:.2f})')
ax.plot(x_range, pdf_normal, color=FOREST, linewidth=1.5,
        linestyle='--', label='Normal')
ax.set_yscale('log')
ax.set_ylim(1e-3, None)

ax.set_title('S\\&P 500 Log-Returns: Student-t vs Normal Fit',
             fontweight='bold')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density (log scale)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False)

ax.text(0.02, 0.95,
        f'$\\nu$={df_fit:.2f}\n'
        f'AIC$_t$={aic_t:,.0f}\n'
        f'AIC$_n$={aic_n:,.0f}\n'
        f'n={n:,}',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_student_t_fit')

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("STUDENT-t FIT & TAIL ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_student_t_pdfs.pdf:   Student-t PDFs (linear + log)")
print("  - ch2_student_t_fit.pdf:    S&P 500 Student-t vs Normal fit")
