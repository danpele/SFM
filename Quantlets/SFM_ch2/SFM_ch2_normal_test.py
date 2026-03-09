"""
SFM_ch2_normal_test
===================
Normal Distribution and Normality Testing

Description:
- Fit Normal distribution to S&P 500 log-returns (histogram + PDF overlay)
- Normality tests: Jarque-Bera, Shapiro-Wilk, Anderson-Darling with QQ-plot
- Central Limit Theorem illustration: t(3) sums converging to Normal

Statistics of Financial Markets course
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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
print("SFM CHAPTER 2: NORMAL DISTRIBUTION & NORMALITY TESTS")
print("=" * 70)

# =============================================================================
# 1. S&P 500 histogram + Normal PDF overlay
# =============================================================================
print("\n1. S&P 500 NORMAL FIT")
print("-" * 40)

import yfinance as yf

print("   Downloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                     progress=False)
sp500_close = sp500['Close'].squeeze()
log_returns = np.log(sp500_close / sp500_close.shift(1)).dropna().values

n_obs = len(log_returns)
mu_hat = np.mean(log_returns)
sigma_hat = np.std(log_returns, ddof=1)
skew_hat = stats.skew(log_returns)
kurt_hat = stats.kurtosis(log_returns)

print(f"   Observations: {n_obs:,}")
print(f"   Mean:     {mu_hat:.6f}")
print(f"   Std:      {sigma_hat:.6f}")
print(f"   Skewness: {skew_hat:.4f}")
print(f"   Kurtosis: {kurt_hat:.4f} (excess)")

# Fit normal distribution via MLE
mu_fit, sigma_fit = stats.norm.fit(log_returns)
print(f"   Normal MLE: mu={mu_fit:.6f}, sigma={sigma_fit:.6f}")

fig, ax = plt.subplots(figsize=(7, 5))

x_range = np.linspace(log_returns.min() * 1.3, log_returns.max() * 1.3, 1000)
pdf_normal = stats.norm.pdf(x_range, loc=mu_fit, scale=sigma_fit)

ax.hist(log_returns, bins=150, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')
ax.plot(x_range, pdf_normal, color=CRIMSON, linewidth=1.5,
        label=f'Normal($\\mu$={mu_fit:.5f}, $\\sigma$={sigma_fit:.4f})')

# Mark +/- 1,2,3 sigma regions
for k, ls in [(1, '-'), (2, '--'), (3, ':')]:
    for sign in [-1, 1]:
        xv = mu_fit + sign * k * sigma_fit
        ax.axvline(xv, color=FOREST, linewidth=0.6, linestyle=ls, alpha=0.5)
    if k == 1:
        ax.axvline(mu_fit + k * sigma_fit, color=FOREST, linewidth=0.6,
                   linestyle=ls, alpha=0.5,
                   label=f'$\\pm 1,2,3\\sigma$')

ax.set_title('S\\&P 500 Log-Returns: Normal Distribution Fit', fontweight='bold')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, frameon=False)

# Summary annotation
ann_text = (f'n = {n_obs:,}\n'
            f'Skewness = {skew_hat:.3f}\n'
            f'Excess kurtosis = {kurt_hat:.3f}')
ax.text(0.02, 0.95, ann_text, transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_normal_fit')
print("   Histogram with Normal PDF overlay")

# =============================================================================
# 2. Normality tests: QQ-plot + test results table
# =============================================================================
print("\n2. NORMALITY TESTS")
print("-" * 40)

# --- Run tests ---
# Jarque-Bera
jb_stat, jb_pval = stats.jarque_bera(log_returns)
print(f"   Jarque-Bera:      stat={jb_stat:.2f}, p-value={jb_pval:.4e}")

# Shapiro-Wilk (max 5000 observations)
sw_sample = log_returns[:5000] if n_obs > 5000 else log_returns
sw_stat, sw_pval = stats.shapiro(sw_sample)
sw_note = f" (first {len(sw_sample):,} obs)" if n_obs > 5000 else ""
print(f"   Shapiro-Wilk:     stat={sw_stat:.6f}, p-value={sw_pval:.4e}{sw_note}")

# Anderson-Darling
ad_result = stats.anderson(log_returns, dist='norm')
ad_stat = ad_result.statistic
# Find the 5% critical value
ad_cv_5 = ad_result.critical_values[2]  # index 2 = 5% significance level
ad_sig_5 = ad_result.significance_level[2]
ad_reject = "Yes" if ad_stat > ad_cv_5 else "No"
print(f"   Anderson-Darling: stat={ad_stat:.4f}, "
      f"crit(5%)={ad_cv_5:.4f}, reject={ad_reject}")

# --- Build figure: QQ-plot (left) + table (right) ---
fig, (ax_qq, ax_tab) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={'width_ratios': [1.2, 1]})

# Panel A: QQ-plot
osm, osr = stats.probplot(log_returns, dist='norm', fit=False)
# Standardise observed quantiles
z_theoretical = osm
z_empirical = (osr - mu_fit) / sigma_fit

ax_qq.scatter(z_theoretical, z_empirical, s=1.5, alpha=0.4,
              color=MAIN_BLUE, rasterized=True)
# 45-degree line
qq_lim = max(abs(z_theoretical.min()), abs(z_theoretical.max()),
             abs(z_empirical.min()), abs(z_empirical.max())) * 1.05
ax_qq.plot([-qq_lim, qq_lim], [-qq_lim, qq_lim], color=CRIMSON,
           linewidth=1.2, linestyle='--', label='45-degree line')

ax_qq.set_xlim(-qq_lim, qq_lim)
ax_qq.set_ylim(-qq_lim, qq_lim)
ax_qq.set_xlabel('Theoretical quantiles (Normal)')
ax_qq.set_ylabel('Sample quantiles (standardised)')
ax_qq.set_title('A. QQ-Plot: S\\&P 500 Returns vs Normal', fontweight='bold')
ax_qq.legend(loc='upper left', frameon=False)
ax_qq.set_aspect('equal', adjustable='box')

# Annotate tail deviation
ax_qq.annotate('Heavy left tail', xy=(-3.5, z_empirical[z_theoretical < -3.4].mean()
               if np.any(z_theoretical < -3.4) else -5),
               fontsize=7, color=CRIMSON, ha='center',
               xytext=(-2.5, -6),
               arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.8))
ax_qq.annotate('Heavy right tail', xy=(3.5, z_empirical[z_theoretical > 3.4].mean()
               if np.any(z_theoretical > 3.4) else 5),
               fontsize=7, color=CRIMSON, ha='center',
               xytext=(2.5, 6),
               arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.8))

# Panel B: Test results table
ax_tab.axis('off')
ax_tab.set_title('B. Normality Test Results', fontweight='bold')

# Build table data
col_labels = ['Test', 'Statistic', 'p-value / Critical', 'Reject H\u2080 (5%)']
table_data = [
    ['Jarque-Bera',
     f'{jb_stat:.2f}',
     f'p = {jb_pval:.2e}',
     'Yes' if jb_pval < 0.05 else 'No'],
    ['Shapiro-Wilk',
     f'{sw_stat:.6f}',
     f'p = {sw_pval:.2e}',
     'Yes' if sw_pval < 0.05 else 'No'],
    ['Anderson-Darling',
     f'{ad_stat:.4f}',
     f'cv(5%) = {ad_cv_5:.4f}',
     ad_reject],
]

table = ax_tab.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 2.0)

# Style the table
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    if row == 0:
        cell.set_facecolor(MAIN_BLUE)
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('white')
        # Color the reject column
        if col == 3:
            if cell.get_text().get_text() == 'Yes':
                cell.set_text_props(color=CRIMSON, fontweight='bold')
            else:
                cell.set_text_props(color=FOREST, fontweight='bold')

# Additional statistics below table
summary_text = (
    f"Sample size: n = {n_obs:,}\n"
    f"Skewness: {skew_hat:.4f}  |  Excess kurtosis: {kurt_hat:.4f}\n"
    f"H\u2080: Data follows a Normal distribution"
)
ax_tab.text(0.5, 0.15, summary_text, transform=ax_tab.transAxes,
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8,
                      edgecolor='#CCCCCC'))

plt.tight_layout(rect=[0, 0.02, 1, 1])
save_fig('ch2_normality_tests')
print("   Panel A: QQ-plot, Panel B: test results table")

# =============================================================================
# 3. Central Limit Theorem illustration — t(3) sums → Normal
# =============================================================================
print("\n3. CENTRAL LIMIT THEOREM ILLUSTRATION")
print("-" * 40)

np.random.seed(42)
df_t = 3  # degrees of freedom for t-distribution
n_sims = 100_000
sample_sizes = [1, 5, 30, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flatten()

panel_labels = ['A', 'B', 'C', 'D']
hist_colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER]

for idx, (n_sum, ax) in enumerate(zip(sample_sizes, axes_flat)):
    # Generate sums of n_sum i.i.d. t(3) random variables, then standardise
    raw_sums = np.sum(stats.t.rvs(df=df_t, size=(n_sims, n_sum)), axis=1)

    # Standardise: (S_n - n*mu) / (sigma * sqrt(n))
    # For t(df), mu=0, var = df/(df-2) = 3 when df=3
    t_var = df_t / (df_t - 2)
    standardised = raw_sums / np.sqrt(n_sum * t_var)

    # Histogram
    ax.hist(standardised, bins=120, density=True, alpha=0.45,
            color=hist_colors[idx], edgecolor='white', linewidth=0.2,
            label=f'Standardised sum (n={n_sum})')

    # Overlay standard normal PDF
    x_grid = np.linspace(-6, 6, 500)
    ax.plot(x_grid, stats.norm.pdf(x_grid), color='black',
            linewidth=1.2, linestyle='--', label='N(0, 1)')

    # If n=1 also show the t(3) PDF for reference
    if n_sum == 1:
        ax.plot(x_grid, stats.t.pdf(x_grid * np.sqrt(t_var), df=df_t) * np.sqrt(t_var),
                color=ORANGE, linewidth=1.0, linestyle=':',
                label=f't({df_t}) (rescaled)')

    ax.set_title(f'{panel_labels[idx]}. n = {n_sum}', fontweight='bold')
    ax.set_xlabel('Standardised value')
    ax.set_ylabel('Density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.55)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

    # KS test against normal
    ks_stat, ks_pval = stats.kstest(standardised, 'norm')
    # Sample kurtosis
    kurt_sample = stats.kurtosis(standardised)

    ax.text(0.03, 0.95,
            f'KS stat = {ks_stat:.4f}\nExcess kurt = {kurt_sample:.3f}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    print(f"   n={n_sum:>3}: KS stat={ks_stat:.4f}, p={ks_pval:.4e}, "
          f"kurtosis={kurt_sample:.3f}")

fig.suptitle('Central Limit Theorem: Sum of t(3) Random Variables',
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig('ch2_clt_illustration')
print("   4-panel CLT convergence diagram")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("NORMAL DISTRIBUTION & NORMALITY TESTS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_normal_fit.pdf:          S&P 500 histogram + Normal PDF")
print("  - ch2_normality_tests.pdf:     QQ-plot + normality test table")
print("  - ch2_clt_illustration.pdf:    CLT convergence (t(3) -> Normal)")
