"""
SFM_ch2_dist_comparison
========================
Distribution Comparison: Fit All, AIC/BIC, Overlay PDFs, QQ Panel

Description:
- Fit Normal, Student-t, GED, and Stable distributions to S&P 500 returns
- Compute AIC/BIC for each fitted distribution
- Overlay PDFs on histogram (log-scale y)
- 2x2 QQ panel for all four distributions
- Tail CDF comparison (survival function on log-log scale)

Statistics of Financial Markets course
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levy_stable, gennorm, t as student_t, norm
from scipy.special import gammaln
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
print("SFM CHAPTER 2: DISTRIBUTION COMPARISON (Section 2.8)")
print("=" * 70)

# =============================================================================
# 0. Download S&P 500 data and compute log-returns
# =============================================================================
print("\n0. DATA ACQUISITION")
print("-" * 40)

import yfinance as yf

print("   Downloading S&P 500 data (2000-2025)...")
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                     progress=False)
sp500_close = sp500['Close'].squeeze()
log_returns = np.log(sp500_close / sp500_close.shift(1)).dropna().values

print(f"   Observations: {len(log_returns)}")
print(f"   Mean:     {np.mean(log_returns):.6f}")
print(f"   Std:      {np.std(log_returns):.6f}")
print(f"   Skewness: {stats.skew(log_returns):.4f}")
print(f"   Kurtosis: {stats.kurtosis(log_returns):.2f}")

# =============================================================================
# 1. MLE Fits for all four distributions
# =============================================================================
print("\n1. MLE FITTING")
print("-" * 40)

# --- Normal ---
print("   Fitting Normal distribution...")
norm_loc, norm_scale = norm.fit(log_returns)
print(f"   Normal:    mu={norm_loc:.6f}, sigma={norm_scale:.6f}")

# --- Student-t ---
print("   Fitting Student-t distribution...")
t_df, t_loc, t_scale = student_t.fit(log_returns)
print(f"   Student-t: df={t_df:.4f}, loc={t_loc:.6f}, scale={t_scale:.6f}")

# --- GED (Generalized Error Distribution = gennorm in scipy) ---
print("   Fitting GED (gennorm) distribution...")
ged_beta, ged_loc, ged_scale = gennorm.fit(log_returns)
print(f"   GED:       beta={ged_beta:.4f}, loc={ged_loc:.6f}, scale={ged_scale:.6f}")

# --- Stable ---
print("   Fitting Stable distribution (this may take a moment)...")
alpha_init, beta_init, loc_init, scale_init = levy_stable._fitstart(log_returns)
try:
    stable_params = levy_stable.fit(log_returns, floc=np.mean(log_returns))
    stable_alpha, stable_beta, stable_loc, stable_scale = stable_params
except Exception:
    stable_alpha, stable_beta = alpha_init, beta_init
    stable_loc, stable_scale = loc_init, scale_init
print(f"   Stable:    alpha={stable_alpha:.4f}, beta={stable_beta:.4f}, "
      f"loc={stable_loc:.6f}, scale={stable_scale:.6f}")

# =============================================================================
# 2. AIC / BIC computation
# =============================================================================
print("\n2. AIC / BIC COMPARISON")
print("-" * 40)

n = len(log_returns)

def compute_aic_bic(log_likelihood, k, n):
    """Compute AIC and BIC given log-likelihood, number of params k, sample size n."""
    aic = -2.0 * log_likelihood + 2.0 * k
    bic = -2.0 * log_likelihood + k * np.log(n)
    return aic, bic

# Normal: 2 parameters (mu, sigma)
ll_norm = np.sum(norm.logpdf(log_returns, loc=norm_loc, scale=norm_scale))
aic_norm, bic_norm = compute_aic_bic(ll_norm, 2, n)

# Student-t: 3 parameters (df, loc, scale)
ll_t = np.sum(student_t.logpdf(log_returns, t_df, loc=t_loc, scale=t_scale))
aic_t, bic_t = compute_aic_bic(ll_t, 3, n)

# GED: 3 parameters (beta, loc, scale)
ll_ged = np.sum(gennorm.logpdf(log_returns, ged_beta, loc=ged_loc, scale=ged_scale))
aic_ged, bic_ged = compute_aic_bic(ll_ged, 3, n)

# Stable: 4 parameters (alpha, beta, loc, scale)
ll_stable = np.sum(levy_stable.logpdf(log_returns, stable_alpha, stable_beta,
                                       loc=stable_loc, scale=stable_scale))
aic_stable, bic_stable = compute_aic_bic(ll_stable, 4, n)

# Print comparison table
print(f"\n   {'Distribution':<15} {'Params':>6} {'LogLik':>12} {'AIC':>12} {'BIC':>12}")
print(f"   {'-'*15} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
print(f"   {'Normal':<15} {2:>6} {ll_norm:>12.2f} {aic_norm:>12.2f} {bic_norm:>12.2f}")
print(f"   {'Student-t':<15} {3:>6} {ll_t:>12.2f} {aic_t:>12.2f} {bic_t:>12.2f}")
print(f"   {'GED':<15} {3:>6} {ll_ged:>12.2f} {aic_ged:>12.2f} {bic_ged:>12.2f}")
print(f"   {'Stable':<15} {4:>6} {ll_stable:>12.2f} {aic_stable:>12.2f} {bic_stable:>12.2f}")

# Determine best model
aic_vals = {'Normal': aic_norm, 'Student-t': aic_t, 'GED': aic_ged, 'Stable': aic_stable}
bic_vals = {'Normal': bic_norm, 'Student-t': bic_t, 'GED': bic_ged, 'Stable': bic_stable}
best_aic = min(aic_vals, key=aic_vals.get)
best_bic = min(bic_vals, key=bic_vals.get)
print(f"\n   Best by AIC: {best_aic}")
print(f"   Best by BIC: {best_bic}")

# =============================================================================
# 3. Chart 1: Distribution overlay (histogram + PDFs, log-scale y)
# =============================================================================
print("\n3. PDF OVERLAY CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 5))

x_range = np.linspace(np.min(log_returns) * 1.3, np.max(log_returns) * 1.3, 1000)

# Histogram
ax.hist(log_returns, bins=150, density=True, alpha=0.35,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label='S\\&P 500 log-returns')

# Overlay PDFs
pdf_norm = norm.pdf(x_range, loc=norm_loc, scale=norm_scale)
pdf_t = student_t.pdf(x_range, t_df, loc=t_loc, scale=t_scale)
pdf_ged = gennorm.pdf(x_range, ged_beta, loc=ged_loc, scale=ged_scale)
pdf_stable = levy_stable.pdf(x_range, stable_alpha, stable_beta,
                              loc=stable_loc, scale=stable_scale)

ax.plot(x_range, pdf_norm, color=FOREST, linewidth=1.3, linestyle='--',
        label=f'Normal (AIC={aic_norm:.0f})')
ax.plot(x_range, pdf_t, color=CRIMSON, linewidth=1.3,
        label=f'Student-t (df={t_df:.2f}, AIC={aic_t:.0f})')
ax.plot(x_range, pdf_ged, color=AMBER, linewidth=1.3, linestyle='-.',
        label=f'GED ($\\beta$={ged_beta:.2f}, AIC={aic_ged:.0f})')
ax.plot(x_range, pdf_stable, color=ORANGE, linewidth=1.3, linestyle=':',
        label=f'Stable ($\\alpha$={stable_alpha:.2f}, AIC={aic_stable:.0f})')

ax.set_yscale('log')
ax.set_ylim(1e-3, None)

ax.set_title('S\\&P 500 Log-Returns: Distribution Comparison', fontweight='bold')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density (log scale)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, frameon=False, fontsize=7)

ax.text(0.02, 0.95,
        f'n = {n:,}\nBest AIC: {best_aic}\nBest BIC: {best_bic}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 1])
save_fig('ch2_dist_comparison_overlay')

# =============================================================================
# 4. Chart 2: 2x2 QQ panel
# =============================================================================
print("\n4. QQ PANEL (2x2)")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

sorted_returns = np.sort(log_returns)
n_obs = len(sorted_returns)
theoretical_quantiles_p = (np.arange(1, n_obs + 1) - 0.5) / n_obs

# --- Panel A: Normal QQ ---
ax = axes[0, 0]
theo_q_norm = norm.ppf(theoretical_quantiles_p, loc=norm_loc, scale=norm_scale)
ax.scatter(theo_q_norm, sorted_returns, s=0.5, alpha=0.3, color=FOREST,
           edgecolors='none', rasterized=True)
lims_norm = [min(theo_q_norm.min(), sorted_returns.min()),
             max(theo_q_norm.max(), sorted_returns.max())]
ax.plot(lims_norm, lims_norm, color=CRIMSON, linewidth=1.0, linestyle='--',
        label='45-degree line')
ax.set_title('A. Normal QQ', fontweight='bold')
ax.set_xlabel('Theoretical quantiles (Normal)')
ax.set_ylabel('Sample quantiles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=7)

# --- Panel B: Student-t QQ ---
ax = axes[0, 1]
theo_q_t = student_t.ppf(theoretical_quantiles_p, t_df, loc=t_loc, scale=t_scale)
ax.scatter(theo_q_t, sorted_returns, s=0.5, alpha=0.3, color=CRIMSON,
           edgecolors='none', rasterized=True)
lims_t = [min(theo_q_t.min(), sorted_returns.min()),
          max(theo_q_t.max(), sorted_returns.max())]
ax.plot(lims_t, lims_t, color=MAIN_BLUE, linewidth=1.0, linestyle='--',
        label='45-degree line')
ax.set_title(f'B. Student-t QQ (df={t_df:.2f})', fontweight='bold')
ax.set_xlabel('Theoretical quantiles (Student-t)')
ax.set_ylabel('Sample quantiles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=7)

# --- Panel C: GED QQ ---
ax = axes[1, 0]
theo_q_ged = gennorm.ppf(theoretical_quantiles_p, ged_beta,
                          loc=ged_loc, scale=ged_scale)
ax.scatter(theo_q_ged, sorted_returns, s=0.5, alpha=0.3, color=AMBER,
           edgecolors='none', rasterized=True)
lims_ged = [min(theo_q_ged.min(), sorted_returns.min()),
            max(theo_q_ged.max(), sorted_returns.max())]
ax.plot(lims_ged, lims_ged, color=CRIMSON, linewidth=1.0, linestyle='--',
        label='45-degree line')
ax.set_title(f'C. GED QQ ($\\beta$={ged_beta:.2f})', fontweight='bold')
ax.set_xlabel('Theoretical quantiles (GED)')
ax.set_ylabel('Sample quantiles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=7)

# --- Panel D: Stable QQ ---
ax = axes[1, 1]
theo_q_stable = levy_stable.ppf(theoretical_quantiles_p, stable_alpha, stable_beta,
                                 loc=stable_loc, scale=stable_scale)
ax.scatter(theo_q_stable, sorted_returns, s=0.5, alpha=0.3, color=ORANGE,
           edgecolors='none', rasterized=True)
lims_stable = [min(theo_q_stable.min(), sorted_returns.min()),
               max(theo_q_stable.max(), sorted_returns.max())]
ax.plot(lims_stable, lims_stable, color=MAIN_BLUE, linewidth=1.0, linestyle='--',
        label='45-degree line')
ax.set_title(f'D. Stable QQ ($\\alpha$={stable_alpha:.2f})', fontweight='bold')
ax.set_xlabel('Theoretical quantiles (Stable)')
ax.set_ylabel('Sample quantiles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('ch2_dist_comparison_qq')
print("   Panel A: Normal, Panel B: Student-t, Panel C: GED, Panel D: Stable")

# =============================================================================
# 5. Chart 3: Tail CDF comparison (survival function, log-log)
# =============================================================================
print("\n5. TAIL CDF COMPARISON (SURVIVAL FUNCTION)")
print("-" * 40)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

# Empirical survival function (right tail)
abs_returns = np.abs(log_returns)
sorted_abs = np.sort(abs_returns)
empirical_sf = 1.0 - np.arange(1, n_obs + 1) / n_obs

# ─── Right tail: P(X > x) ───────────────────────────────────────────────────
x_tail = np.linspace(0.005, sorted_abs.max(), 500)

sf_norm = norm.sf(x_tail, loc=norm_loc, scale=norm_scale)
sf_t = student_t.sf(x_tail, t_df, loc=t_loc, scale=t_scale)
sf_ged = gennorm.sf(x_tail, ged_beta, loc=ged_loc, scale=ged_scale)
sf_stable = levy_stable.sf(x_tail, stable_alpha, stable_beta,
                            loc=stable_loc, scale=stable_scale)

# Right tail empirical: P(X > x)
sorted_right = np.sort(log_returns)
empirical_sf_right = 1.0 - np.arange(1, n_obs + 1) / n_obs

ax_left.loglog(sorted_right[sorted_right > 0],
               empirical_sf_right[:np.sum(sorted_right > 0)][::-1],
               '.', markersize=1, alpha=0.3, color='gray', label='Empirical',
               rasterized=True)
ax_left.loglog(x_tail[x_tail > 0], sf_norm[x_tail > 0], color=FOREST,
               linewidth=1.3, linestyle='--', label='Normal')
ax_left.loglog(x_tail[x_tail > 0], sf_t[x_tail > 0], color=CRIMSON,
               linewidth=1.3, label='Student-t')
ax_left.loglog(x_tail[x_tail > 0], sf_ged[x_tail > 0], color=AMBER,
               linewidth=1.3, linestyle='-.', label='GED')
ax_left.loglog(x_tail[x_tail > 0], sf_stable[x_tail > 0], color=ORANGE,
               linewidth=1.3, linestyle=':', label='Stable')

ax_left.set_title('A. Right Tail: P(X > x)', fontweight='bold')
ax_left.set_xlabel('x (log scale)')
ax_left.set_ylabel('P(X > x) (log scale)')
ax_left.set_xlim(1e-3, 0.15)
ax_left.set_ylim(1e-5, 1)
ax_left.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, frameon=False, fontsize=7)

# ─── Left tail: P(X < -x) = CDF(-x) ────────────────────────────────────────
x_tail_neg = -x_tail[::-1]  # negative values for left tail

cdf_norm = norm.cdf(x_tail_neg, loc=norm_loc, scale=norm_scale)
cdf_t = student_t.cdf(x_tail_neg, t_df, loc=t_loc, scale=t_scale)
cdf_ged = gennorm.cdf(x_tail_neg, ged_beta, loc=ged_loc, scale=ged_scale)
cdf_stable = levy_stable.cdf(x_tail_neg, stable_alpha, stable_beta,
                               loc=stable_loc, scale=stable_scale)

# Left tail empirical: P(X < x) for x < 0
sorted_left = np.sort(log_returns)
empirical_cdf_left = np.arange(1, n_obs + 1) / n_obs
mask_neg = sorted_left < 0

ax_right.loglog(np.abs(sorted_left[mask_neg]),
                empirical_cdf_left[mask_neg][::-1],
                '.', markersize=1, alpha=0.3, color='gray', label='Empirical',
                rasterized=True)
# For left tail, we plot |x| vs P(X < -|x|)
x_pos = x_tail[x_tail > 0]
cdf_norm_left = norm.cdf(-x_pos, loc=norm_loc, scale=norm_scale)
cdf_t_left = student_t.cdf(-x_pos, t_df, loc=t_loc, scale=t_scale)
cdf_ged_left = gennorm.cdf(-x_pos, ged_beta, loc=ged_loc, scale=ged_scale)
cdf_stable_left = levy_stable.cdf(-x_pos, stable_alpha, stable_beta,
                                    loc=stable_loc, scale=stable_scale)

ax_right.loglog(x_pos, cdf_norm_left, color=FOREST,
                linewidth=1.3, linestyle='--', label='Normal')
ax_right.loglog(x_pos, cdf_t_left, color=CRIMSON,
                linewidth=1.3, label='Student-t')
ax_right.loglog(x_pos, cdf_ged_left, color=AMBER,
                linewidth=1.3, linestyle='-.', label='GED')
ax_right.loglog(x_pos, cdf_stable_left, color=ORANGE,
                linewidth=1.3, linestyle=':', label='Stable')

ax_right.set_title('B. Left Tail: P(X < -x)', fontweight='bold')
ax_right.set_xlabel('|x| (log scale)')
ax_right.set_ylabel('P(X < -x) (log scale)')
ax_right.set_xlim(1e-3, 0.15)
ax_right.set_ylim(1e-5, 1)
ax_right.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                ncol=3, frameon=False, fontsize=7)

# Mark key sigma thresholds on both panels
for ax_panel in [ax_left, ax_right]:
    for sigma_mult, lbl in [(3, r'$3\sigma$'), (5, r'$5\sigma$')]:
        thresh = sigma_mult * norm_scale
        ax_panel.axvline(thresh, color='gray', linewidth=0.5, linestyle=':',
                         alpha=0.5)
        ax_panel.text(thresh * 1.1, 0.3, lbl, fontsize=6, color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
save_fig('ch2_dist_comparison_tails')

# Print tail probability comparison
print("\n   Tail probability comparison at key thresholds:")
print(f"   {'Threshold':<12} {'Normal':>12} {'Student-t':>12} {'GED':>12} {'Stable':>12}")
print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for sigma_mult in [3, 4, 5]:
    thresh = sigma_mult * norm_scale
    p_norm = norm.sf(thresh, loc=0, scale=norm_scale)
    p_t = student_t.sf(thresh, t_df, loc=0, scale=t_scale)
    p_ged = gennorm.sf(thresh, ged_beta, loc=0, scale=ged_scale)
    p_stable = levy_stable.sf(thresh, stable_alpha, stable_beta,
                               loc=0, scale=stable_scale)
    print(f"   {sigma_mult}*sigma      {p_norm:>12.2e} {p_t:>12.2e} {p_ged:>12.2e} {p_stable:>12.2e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("DISTRIBUTION COMPARISON COMPLETE (Section 2.8)")
print("=" * 70)
print("\nFitted parameters:")
print(f"  Normal:    mu={norm_loc:.6f}, sigma={norm_scale:.6f}")
print(f"  Student-t: df={t_df:.4f}, loc={t_loc:.6f}, scale={t_scale:.6f}")
print(f"  GED:       beta={ged_beta:.4f}, loc={ged_loc:.6f}, scale={ged_scale:.6f}")
print(f"  Stable:    alpha={stable_alpha:.4f}, beta={stable_beta:.4f}, "
      f"loc={stable_loc:.6f}, scale={stable_scale:.6f}")
print(f"\nBest model by AIC: {best_aic}")
print(f"Best model by BIC: {best_bic}")
print("\nOutput files:")
print("  - ch2_dist_comparison_overlay.pdf:  Histogram + 4 distribution PDFs")
print("  - ch2_dist_comparison_qq.pdf:       2x2 QQ panel")
print("  - ch2_dist_comparison_tails.pdf:    Tail CDF comparison (log-log)")
