"""
SFM_ch2_evt_analysis
====================
Extreme Value Theory: Block Maxima (GEV), Peaks Over Threshold (GPD),
and EVT-Based Risk Measures

Description:
- Fit GEV distribution to monthly block maxima of negative S&P 500 returns
- Fit GPD to exceedances above the 95th percentile threshold (POT method)
- Mean excess function plot for threshold selection diagnostics
- Compare VaR and ES from EVT (GPD) vs Normal vs Empirical

Statistics of Financial Markets course — Section 2.7
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genextreme, genpareto, norm
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
print("SFM CHAPTER 2: EXTREME VALUE THEORY ANALYSIS")
print("=" * 70)

# =============================================================================
# Data: Download S&P 500 and compute negative log-returns (losses)
# =============================================================================
print("\n   Downloading S&P 500 data...")
import yfinance as yf

sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                     progress=False)
sp500_close = sp500['Close'].squeeze()
log_returns = np.log(sp500_close / sp500_close.shift(1)).dropna()

# Losses = negative returns (so large positive values are bad days)
losses = -log_returns
losses_arr = losses.values

print(f"   Observations: {len(losses_arr)}")
print(f"   Mean loss:    {np.mean(losses_arr):.6f}")
print(f"   Max loss:     {np.max(losses_arr):.4f} ({np.max(losses_arr)*100:.2f}%)")
print(f"   Kurtosis:     {stats.kurtosis(losses_arr):.2f}")

# =============================================================================
# 1. GEV Fit to Block Maxima (monthly)
# =============================================================================
print("\n1. BLOCK MAXIMA — GEV FIT")
print("-" * 40)

# Compute monthly block maxima of losses
losses_series = pd.Series(losses_arr, index=log_returns.index)
block_maxima = losses_series.resample('ME').max().dropna().values

n_blocks = len(block_maxima)
print(f"   Number of monthly blocks: {n_blocks}")
print(f"   Mean block maximum:  {np.mean(block_maxima):.6f}")
print(f"   Max block maximum:   {np.max(block_maxima):.4f}")

# Fit GEV distribution
# scipy genextreme uses sign convention c = -xi (shape parameter)
# so xi > 0 (Frechet) corresponds to c < 0
c_gev, loc_gev, scale_gev = genextreme.fit(block_maxima)
xi_gev = -c_gev  # Convert to standard EVT notation

print(f"   GEV fit (EVT notation):")
print(f"     xi (shape)    = {xi_gev:.4f}  {'(Frechet: heavy tail)' if xi_gev > 0 else '(Weibull: bounded tail)' if xi_gev < 0 else '(Gumbel)'}")
print(f"     mu (location) = {loc_gev:.6f}")
print(f"     sigma (scale) = {scale_gev:.6f}")

# Plot: histogram of block maxima + fitted GEV PDF
fig, ax = plt.subplots(figsize=(7, 5))

x_gev = np.linspace(
    max(0, np.min(block_maxima) - 0.5 * np.std(block_maxima)),
    np.max(block_maxima) + np.std(block_maxima),
    500
)

ax.hist(block_maxima, bins=40, density=True, alpha=0.4,
        color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
        label=f'Monthly block maxima (n={n_blocks})')

pdf_gev = genextreme.pdf(x_gev, c_gev, loc=loc_gev, scale=scale_gev)
ax.plot(x_gev, pdf_gev, color=CRIMSON, linewidth=1.5,
        label=f'GEV fit ($\\xi$={xi_gev:.3f})')

# Also overlay a normal fit to the block maxima for comparison
mu_bm, sig_bm = norm.fit(block_maxima)
pdf_norm_bm = norm.pdf(x_gev, loc=mu_bm, scale=sig_bm)
ax.plot(x_gev, pdf_norm_bm, color=FOREST, linewidth=1.5, linestyle='--',
        label='Normal fit')

ax.set_title('Block Maxima of Negative S\\&P 500 Returns: GEV Fit',
             fontweight='bold')
ax.set_xlabel('Monthly maximum loss')
ax.set_ylabel('Density')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False)

# Annotation box with parameters
param_text = (f'$\\xi$ = {xi_gev:.4f}\n'
              f'$\\mu$ = {loc_gev:.4f}\n'
              f'$\\sigma$ = {scale_gev:.4f}\n'
              f'Blocks = {n_blocks}')
ax.text(0.97, 0.95, param_text, transform=ax.transAxes, fontsize=8,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_evt_gev_fit')
print("   Heavy right tail captured by positive xi (Frechet domain)")

# =============================================================================
# 2. GPD Fit to POT Exceedances + Mean Excess Plot
# =============================================================================
print("\n2. PEAKS OVER THRESHOLD — GPD FIT + MEAN EXCESS PLOT")
print("-" * 40)

# Threshold: 95th percentile of losses
threshold = np.percentile(losses_arr, 95)
exceedances = losses_arr[losses_arr > threshold] - threshold
n_exceed = len(exceedances)
n_total = len(losses_arr)

print(f"   Threshold (95th pct): {threshold:.6f} ({threshold*100:.3f}%)")
print(f"   Exceedances: {n_exceed} out of {n_total} ({100*n_exceed/n_total:.1f}%)")

# Fit GPD to exceedances
c_gpd, loc_gpd, scale_gpd = genpareto.fit(exceedances, floc=0)
xi_gpd = c_gpd  # scipy genpareto shape = xi directly

print(f"   GPD fit:")
print(f"     xi (shape) = {xi_gpd:.4f}  {'(heavy tail)' if xi_gpd > 0 else '(thin tail)'}")
print(f"     sigma (scale) = {scale_gpd:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel A: Exceedances histogram + GPD PDF ---
x_gpd = np.linspace(0, np.max(exceedances) * 1.1, 500)
pdf_gpd = genpareto.pdf(x_gpd, c_gpd, loc=0, scale=scale_gpd)

ax1.hist(exceedances, bins=40, density=True, alpha=0.4,
         color=MAIN_BLUE, edgecolor='white', linewidth=0.3,
         label=f'Exceedances (n={n_exceed})')
ax1.plot(x_gpd, pdf_gpd, color=CRIMSON, linewidth=1.5,
         label=f'GPD fit ($\\xi$={xi_gpd:.3f}, $\\sigma$={scale_gpd:.4f})')

# Exponential reference (xi=0 case)
pdf_exp = stats.expon.pdf(x_gpd, scale=np.mean(exceedances))
ax1.plot(x_gpd, pdf_exp, color=FOREST, linewidth=1.2, linestyle='--',
         label='Exponential reference')

ax1.set_title('A. POT Exceedances: GPD Fit', fontweight='bold')
ax1.set_xlabel(f'Excess above threshold ({threshold:.4f})')
ax1.set_ylabel('Density')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=False, fontsize=7)

param_text_gpd = (f'Threshold = {threshold:.4f}\n'
                  f'$\\xi$ = {xi_gpd:.4f}\n'
                  f'$\\sigma$ = {scale_gpd:.4f}\n'
                  f'n$_u$ = {n_exceed}')
ax1.text(0.97, 0.95, param_text_gpd, transform=ax1.transAxes, fontsize=8,
         va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Panel B: Mean Excess Function ---
# e(u) = E[X - u | X > u], should be linear in u for GPD tails
sorted_losses = np.sort(losses_arr)
n_points = 200
thresholds = np.linspace(
    np.percentile(losses_arr, 80),
    np.percentile(losses_arr, 99.5),
    n_points
)

mean_excess = np.zeros(n_points)
ci_upper = np.zeros(n_points)
ci_lower = np.zeros(n_points)

for i, u in enumerate(thresholds):
    excess = losses_arr[losses_arr > u] - u
    if len(excess) > 5:
        mean_excess[i] = np.mean(excess)
        se = np.std(excess) / np.sqrt(len(excess))
        ci_upper[i] = mean_excess[i] + 1.96 * se
        ci_lower[i] = mean_excess[i] - 1.96 * se
    else:
        mean_excess[i] = np.nan
        ci_upper[i] = np.nan
        ci_lower[i] = np.nan

# Filter out NaN values for plotting
valid = ~np.isnan(mean_excess)
thresholds_v = thresholds[valid]
mean_excess_v = mean_excess[valid]
ci_upper_v = ci_upper[valid]
ci_lower_v = ci_lower[valid]

ax2.plot(thresholds_v, mean_excess_v, color=MAIN_BLUE, linewidth=1.2,
         label='Mean excess function $e(u)$')
ax2.fill_between(thresholds_v, ci_lower_v, ci_upper_v,
                 color=MAIN_BLUE, alpha=0.15, label='95% CI')

# Mark the chosen threshold
ax2.axvline(threshold, color=CRIMSON, linewidth=1.2, linestyle='--',
            label=f'Chosen threshold ({threshold:.4f})')

# Theoretical GPD mean excess: e(u) = (sigma + xi*u) / (1 - xi) for xi < 1
# After fitting at the chosen threshold, the theoretical mean excess of
# the original data is: e(u) = (scale_gpd + xi_gpd * (u - threshold)) / (1 - xi_gpd)
if xi_gpd < 1:
    me_theoretical = (scale_gpd + xi_gpd * (thresholds_v - threshold)) / (1 - xi_gpd)
    # Only plot where it makes sense (above the chosen threshold)
    mask_above = thresholds_v >= threshold
    ax2.plot(thresholds_v[mask_above], me_theoretical[mask_above],
             color=ORANGE, linewidth=1.2, linestyle=':',
             label=f'GPD theoretical ($\\xi$={xi_gpd:.3f})')

ax2.set_title('B. Mean Excess Plot', fontweight='bold')
ax2.set_xlabel('Threshold $u$')
ax2.set_ylabel('Mean excess $e(u)$')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=False, fontsize=7)

ax2.text(0.03, 0.95,
         'Linear $\\Rightarrow$ GPD tail\nUpward slope $\\Rightarrow$ heavy tail',
         transform=ax2.transAxes, fontsize=7, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 1])
save_fig('ch2_evt_gpd_pot')
print("   Panel A: GPD fit to exceedances above 95th percentile")
print("   Panel B: Mean excess function with 95% CI band")

# =============================================================================
# 3. VaR and ES Comparison: EVT (GPD) vs Normal vs Empirical
# =============================================================================
print("\n3. VaR AND ES COMPARISON — EVT vs NORMAL vs EMPIRICAL")
print("-" * 40)

confidence_levels = [0.95, 0.99, 0.999]
n_u = n_exceed   # number of exceedances
N = n_total      # total observations

# Pre-compute normal fit to full loss series
mu_loss, sigma_loss = norm.fit(losses_arr)
print(f"   Normal fit to losses: mu={mu_loss:.6f}, sigma={sigma_loss:.6f}")

results = []

for alpha_cl in confidence_levels:
    # --- EVT (GPD-based) VaR and ES ---
    # VaR_alpha = u + (sigma/xi) * [((N/n_u)*(1-alpha))^(-xi) - 1]
    p = 1 - alpha_cl  # tail probability
    if xi_gpd != 0:
        var_evt = threshold + (scale_gpd / xi_gpd) * (
            ((n_u / N) / p) ** xi_gpd - 1
        )
    else:
        var_evt = threshold + scale_gpd * np.log((n_u / N) / p)

    # ES_alpha = VaR_alpha / (1 - xi) + (sigma - xi * u) / (1 - xi)
    if xi_gpd < 1:
        es_evt = var_evt / (1 - xi_gpd) + (scale_gpd - xi_gpd * threshold) / (1 - xi_gpd)
    else:
        es_evt = np.nan

    # --- Normal VaR and ES ---
    z_alpha = norm.ppf(alpha_cl)
    var_normal = mu_loss + sigma_loss * z_alpha
    # Normal ES: mu + sigma * phi(z_alpha) / (1-alpha)
    es_normal = mu_loss + sigma_loss * norm.pdf(z_alpha) / (1 - alpha_cl)

    # --- Empirical VaR and ES ---
    var_empirical = np.percentile(losses_arr, alpha_cl * 100)
    es_empirical = np.mean(losses_arr[losses_arr >= var_empirical])

    results.append({
        'level': alpha_cl,
        'var_evt': var_evt,
        'var_normal': var_normal,
        'var_empirical': var_empirical,
        'es_evt': es_evt,
        'es_normal': es_normal,
        'es_empirical': es_empirical,
    })

    print(f"   --- {alpha_cl*100:.1f}% confidence level ---")
    print(f"   VaR:  EVT={var_evt:.5f}  Normal={var_normal:.5f}  Empirical={var_empirical:.5f}")
    print(f"   ES:   EVT={es_evt:.5f}  Normal={es_normal:.5f}  Empirical={es_empirical:.5f}")

# --- Plot: grouped bar chart ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

level_labels = [f'{r["level"]*100:.1f}%' for r in results]
x_pos = np.arange(len(results))
bar_width = 0.25

# Panel A: VaR comparison
var_evt_vals    = [r['var_evt'] for r in results]
var_normal_vals = [r['var_normal'] for r in results]
var_emp_vals    = [r['var_empirical'] for r in results]

bars1 = ax1.bar(x_pos - bar_width, var_evt_vals, bar_width,
                color=CRIMSON, edgecolor='white', linewidth=0.5, label='EVT (GPD)')
bars2 = ax1.bar(x_pos, var_normal_vals, bar_width,
                color=MAIN_BLUE, edgecolor='white', linewidth=0.5, label='Normal')
bars3 = ax1.bar(x_pos + bar_width, var_emp_vals, bar_width,
                color=FOREST, edgecolor='white', linewidth=0.5, label='Empirical')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=6,
                 fontweight='bold', rotation=45)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(level_labels)
ax1.set_xlabel('Confidence level')
ax1.set_ylabel('VaR (loss)')
ax1.set_title('A. Value-at-Risk Comparison', fontweight='bold')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
           ncol=3, frameon=False)

# Panel B: ES comparison
es_evt_vals    = [r['es_evt'] for r in results]
es_normal_vals = [r['es_normal'] for r in results]
es_emp_vals    = [r['es_empirical'] for r in results]

bars1 = ax2.bar(x_pos - bar_width, es_evt_vals, bar_width,
                color=CRIMSON, edgecolor='white', linewidth=0.5, label='EVT (GPD)')
bars2 = ax2.bar(x_pos, es_normal_vals, bar_width,
                color=MAIN_BLUE, edgecolor='white', linewidth=0.5, label='Normal')
bars3 = ax2.bar(x_pos + bar_width, es_emp_vals, bar_width,
                color=FOREST, edgecolor='white', linewidth=0.5, label='Empirical')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=6,
                 fontweight='bold', rotation=45)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(level_labels)
ax2.set_xlabel('Confidence level')
ax2.set_ylabel('ES (loss)')
ax2.set_title('B. Expected Shortfall Comparison', fontweight='bold')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
           ncol=3, frameon=False)

# Summary table annotation in Panel B
table_text = "Portfolio = 1M EUR\n"
table_text += f"{'Level':>6} {'EVT':>10} {'Normal':>10} {'Empir.':>10}\n"
table_text += "-" * 40 + "\n"
for r in results:
    cl_str = f"{r['level']*100:.1f}%"
    table_text += f"{cl_str:>6} {r['es_evt']*1e6:>10,.0f} {r['es_normal']*1e6:>10,.0f} {r['es_empirical']*1e6:>10,.0f}\n"

ax2.text(0.03, 0.95, table_text.strip(), transform=ax2.transAxes,
         fontsize=6, va='top', ha='left', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_evt_var_es')
print("   Panel A: VaR comparison across methods")
print("   Panel B: ES comparison across methods")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("EXTREME VALUE THEORY ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_evt_gev_fit.pdf:   Block maxima GEV fit (monthly)")
print("  - ch2_evt_gpd_pot.pdf:   GPD fit + mean excess plot")
print("  - ch2_evt_var_es.pdf:    VaR and ES comparison (EVT/Normal/Empirical)")
