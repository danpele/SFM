"""
SFM_ch2_risk_measures
=====================
VaR, ES, Rolling Backtest, and Basel Traffic Light Test

Description:
- Rolling 99% VaR backtest on S&P 500 (Normal vs Student-t)
- Basel traffic light test (green / yellow / red zones)
- Risk comparison table: VaR and ES at multiple confidence levels
  under Normal, Student-t, and Empirical distributions

Statistics of Financial Markets course
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as student_t
import yfinance as yf
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
print("SFM CHAPTER 2: VaR / ES / BACKTEST / BASEL TEST")
print("=" * 70)

# ─── Download S&P 500 data ───────────────────────────────────────────────────
print("\n   Downloading S&P 500 data (2000-2025)...")
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                     progress=False)
sp500_close = sp500['Close'].squeeze()
log_returns = np.log(sp500_close / sp500_close.shift(1)).dropna()
print(f"   Observations: {len(log_returns)}")
print(f"   Sample period: {log_returns.index[0].strftime('%Y-%m-%d')} to "
      f"{log_returns.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 1. Rolling VaR Backtest with Basel Traffic Light Test
# =============================================================================
print("\n1. ROLLING VaR BACKTEST (99%, 250-day window)")
print("-" * 40)

window = 250
alpha_var = 0.01  # 99% VaR

returns_arr = log_returns.values
dates_arr = log_returns.index

# Pre-allocate arrays for rolling VaR
n = len(returns_arr)
var_normal = np.full(n, np.nan)
var_t = np.full(n, np.nan)

print("   Computing rolling VaR estimates...")
for i in range(window, n):
    roll = returns_arr[i - window:i]

    # ── Normal VaR ──
    mu_n = np.mean(roll)
    sigma_n = np.std(roll, ddof=1)
    var_normal[i] = mu_n + sigma_n * norm.ppf(alpha_var)

    # ── Student-t VaR ──
    df_t, loc_t, scale_t = student_t.fit(roll)
    var_t[i] = student_t.ppf(alpha_var, df_t, loc=loc_t, scale=scale_t)

# Identify exceedances (return < -VaR means VaR breach; VaR is negative)
valid = ~np.isnan(var_normal)
exc_normal = (returns_arr < var_normal) & valid
exc_t = (returns_arr < var_t) & valid

n_exc_normal = np.sum(exc_normal)
n_exc_t = np.sum(exc_t)
n_valid = np.sum(valid)
expected_exc = n_valid * alpha_var

print(f"   Backtest period: {n_valid} trading days")
print(f"   Expected exceedances (1%): {expected_exc:.1f}")
print(f"   Normal VaR exceedances:    {n_exc_normal}")
print(f"   Student-t VaR exceedances: {n_exc_t}")

# ── Basel Traffic Light Test (last 250 observations) ──
last_250_ret = returns_arr[-250:]
last_250_var_n = var_normal[-250:]
last_250_var_t = var_t[-250:]

basel_exc_normal = np.sum(last_250_ret < last_250_var_n)
basel_exc_t = np.sum(last_250_ret < last_250_var_t)

def basel_zone(exc):
    if exc < 5:
        return 'GREEN'
    elif exc <= 9:
        return 'YELLOW'
    else:
        return 'RED'

def basel_color(exc):
    if exc < 5:
        return FOREST
    elif exc <= 9:
        return AMBER
    else:
        return CRIMSON

print(f"\n   Basel Traffic Light Test (last 250 days):")
print(f"   {'Model':<12} {'Exceedances':>12} {'Zone':>8}")
print(f"   {'-'*34}")
print(f"   {'Normal':<12} {basel_exc_normal:>12d} {basel_zone(basel_exc_normal):>8}")
print(f"   {'Student-t':<12} {basel_exc_t:>12d} {basel_zone(basel_exc_t):>8}")
print(f"   Thresholds: Green <5, Yellow 5-9, Red 10+")

# ── Plot ──
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Panel A: Returns and VaR lines
ax1.plot(dates_arr, returns_arr, color=MAIN_BLUE, linewidth=0.3,
         alpha=0.6, label='S\\&P 500 log-returns')
ax1.plot(dates_arr, var_normal, color=CRIMSON, linewidth=0.8,
         label=f'99% VaR Normal (exc={n_exc_normal})')
ax1.plot(dates_arr, var_t, color=FOREST, linewidth=0.8,
         linestyle='--', label=f'99% VaR Student-t (exc={n_exc_t})')

# Mark exceedances
exc_dates_n = dates_arr[exc_normal]
exc_vals_n = returns_arr[exc_normal]
exc_dates_t = dates_arr[exc_t & ~exc_normal]
exc_vals_t = returns_arr[exc_t & ~exc_normal]

ax1.scatter(exc_dates_n, exc_vals_n, color=CRIMSON, s=12, zorder=5,
            marker='v', alpha=0.8, label=f'Normal breach')
ax1.scatter(exc_dates_t, exc_vals_t, color=ORANGE, s=12, zorder=5,
            marker='v', alpha=0.8, label=f'Student-t only breach')

ax1.set_title('A. Rolling 99% VaR Backtest (250-day window)', fontweight='bold')
ax1.set_ylabel('Log-return')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, frameon=False, fontsize=7)
plt.setp(ax1.get_xticklabels(), visible=False)

# Basel zone annotation box
basel_text = (f"Basel Test (last 250 days)\n"
              f"Normal: {basel_exc_normal} exc. \u2192 {basel_zone(basel_exc_normal)}\n"
              f"Student-t: {basel_exc_t} exc. \u2192 {basel_zone(basel_exc_t)}")
ax1.text(0.98, 0.95, basel_text, transform=ax1.transAxes,
         fontsize=7, va='top', ha='right', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white',
                   edgecolor=basel_color(basel_exc_normal), alpha=0.9,
                   linewidth=1.5))

# Panel B: Cumulative exceedance count
cum_exc_n = np.cumsum(exc_normal)
cum_exc_t = np.cumsum(exc_t)

ax2.plot(dates_arr, cum_exc_n, color=CRIMSON, linewidth=1.0,
         label='Normal cumulative exceedances')
ax2.plot(dates_arr, cum_exc_t, color=FOREST, linewidth=1.0,
         linestyle='--', label='Student-t cumulative exceedances')

# Expected exceedance line
cum_expected = np.cumsum(valid) * alpha_var
ax2.plot(dates_arr, cum_expected, color='gray', linewidth=0.8,
         linestyle=':', label='Expected (1%)')

ax2.set_title('B. Cumulative Exceedances', fontweight='bold', fontsize=9)
ax2.set_xlabel('Date')
ax2.set_ylabel('Count')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
           ncol=3, frameon=False, fontsize=7)

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig('ch2_var_backtest')

# =============================================================================
# 2. Risk Comparison Table: VaR and ES under multiple distributions
# =============================================================================
print("\n2. RISK COMPARISON TABLE (VaR & ES, $1M portfolio)")
print("-" * 40)

# Use full sample for distribution fitting
ret = log_returns.values
portfolio_value = 1_000_000

# Fit distributions
mu, sigma = norm.fit(ret)
df_fit, loc_fit, scale_fit = student_t.fit(ret)

print(f"   Normal fit:    mu={mu:.6f}, sigma={sigma:.6f}")
print(f"   Student-t fit: df={df_fit:.2f}, loc={loc_fit:.6f}, scale={scale_fit:.6f}")

confidence_levels = [0.05, 0.01, 0.001]  # 95%, 99%, 99.9%
level_labels = ['95%', '99%', '99.9%']

# Storage for table
results = []

for cl, cl_label in zip(confidence_levels, level_labels):
    # ── Normal ──
    var_n = -norm.ppf(cl, loc=mu, scale=sigma)
    z_alpha = norm.ppf(cl)
    es_n = -(norm.pdf(z_alpha) / cl * sigma - mu)  # ES = -phi(z)/alpha * sigma + mu
    # Correct sign: ES_normal = mu - sigma * phi(z_alpha) / alpha
    # For loss: VaR = -(mu + sigma*z_alpha), ES = -(mu - sigma*phi(z_alpha)/alpha)
    es_n = -(mu - sigma * norm.pdf(z_alpha) / cl)

    # ── Student-t ──
    t_q = student_t.ppf(cl, df_fit)
    var_st = -(loc_fit + scale_fit * t_q)
    # ES under Student-t: ES = -(df + t_q^2)/(df-1) * f(t_q)/alpha * scale + loc
    # where f is the standard Student-t pdf
    t_pdf_val = student_t.pdf(t_q, df_fit)  # standard t pdf
    es_st = -( loc_fit - (df_fit + t_q**2) / (df_fit - 1) * t_pdf_val / cl * scale_fit )

    # ── Empirical ──
    var_emp = -np.percentile(ret, cl * 100)
    tail_returns = ret[ret <= np.percentile(ret, cl * 100)]
    if len(tail_returns) > 0:
        es_emp = -np.mean(tail_returns)
    else:
        es_emp = var_emp

    results.append({
        'level': cl_label,
        'var_n': var_n, 'es_n': es_n,
        'var_t': var_st, 'es_t': es_st,
        'var_e': var_emp, 'es_e': es_emp
    })

    var_n_dollar = var_n * portfolio_value
    es_n_dollar = es_n * portfolio_value
    var_t_dollar = var_st * portfolio_value
    es_t_dollar = es_st * portfolio_value
    var_e_dollar = var_emp * portfolio_value
    es_e_dollar = es_emp * portfolio_value

    print(f"\n   Confidence level: {cl_label}")
    print(f"   {'':>10} {'Normal':>14} {'Student-t':>14} {'Empirical':>14}")
    print(f"   {'VaR':>10} ${var_n_dollar:>12,.0f} ${var_t_dollar:>12,.0f} ${var_e_dollar:>12,.0f}")
    print(f"   {'ES':>10} ${es_n_dollar:>12,.0f} ${es_t_dollar:>12,.0f} ${es_e_dollar:>12,.0f}")

# ── Create table visualization ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

# Build cell data
col_labels = ['Level', 'Measure',
              'Normal\nVaR ($)', 'Normal\nES ($)',
              'Student-t\nVaR ($)', 'Student-t\nES ($)',
              'Empirical\nVaR ($)', 'Empirical\nES ($)']

cell_data = []
cell_colors = []

row_bg_white = ['#FFFFFF'] * 8
row_bg_light = ['#F5F7FA'] * 8

for idx, r in enumerate(results):
    row = [
        r['level'], '',
        f"${r['var_n'] * portfolio_value:,.0f}",
        f"${r['es_n'] * portfolio_value:,.0f}",
        f"${r['var_t'] * portfolio_value:,.0f}",
        f"${r['es_t'] * portfolio_value:,.0f}",
        f"${r['var_e'] * portfolio_value:,.0f}",
        f"${r['es_e'] * portfolio_value:,.0f}",
    ]
    bg = row_bg_light if idx % 2 == 0 else row_bg_white
    cell_data.append(row)
    cell_colors.append(bg)

# Restructure into a cleaner format: one row per level showing VaR and ES
cell_data_clean = []
cell_colors_clean = []

header_bg = [MAIN_BLUE] * 7
normal_bg = '#F5F7FA'
alt_bg = '#FFFFFF'

col_labels_clean = ['Level',
                    'Normal\nVaR', 'Normal\nES',
                    'Student-t\nVaR', 'Student-t\nES',
                    'Empirical\nVaR', 'Empirical\nES']

for idx, r in enumerate(results):
    row = [
        r['level'],
        f"${r['var_n'] * portfolio_value:>,.0f}",
        f"${r['es_n'] * portfolio_value:>,.0f}",
        f"${r['var_t'] * portfolio_value:>,.0f}",
        f"${r['es_t'] * portfolio_value:>,.0f}",
        f"${r['var_e'] * portfolio_value:>,.0f}",
        f"${r['es_e'] * portfolio_value:>,.0f}",
    ]
    bg_color = normal_bg if idx % 2 == 0 else alt_bg
    cell_data_clean.append(row)
    cell_colors_clean.append([bg_color] * 7)

table = ax.table(cellText=cell_data_clean,
                 colLabels=col_labels_clean,
                 cellColours=cell_colors_clean,
                 colColours=[MAIN_BLUE] * 7,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.0)

# Style header row
for j in range(7):
    cell = table[0, j]
    cell.set_text_props(color='white', fontweight='bold', fontsize=8)
    cell.set_edgecolor('white')
    cell.set_linewidth(0.5)

# Style data rows
for i in range(1, len(cell_data_clean) + 1):
    for j in range(7):
        cell = table[i, j]
        cell.set_edgecolor('#E0E0E0')
        cell.set_linewidth(0.5)
        if j == 0:
            cell.set_text_props(fontweight='bold')
        # Highlight ES columns (they should be larger than VaR)
        if j in [2, 4, 6]:
            cell.set_text_props(color=CRIMSON)

ax.set_title('Risk Measures Comparison ($1M Portfolio)\nVaR and Expected Shortfall under Different Distributions',
             fontweight='bold', fontsize=11, pad=20)

# Footnotes
footnote = (f"Data: S&P 500 log-returns, {log_returns.index[0].strftime('%Y')}"
            f"\u2013{log_returns.index[-1].strftime('%Y')} "
            f"(n={len(ret):,})\n"
            f"Student-t df={df_fit:.2f} | "
            f"ES values shown in red (ES $\\geq$ VaR by construction)")
ax.text(0.5, -0.02, footnote, transform=ax.transAxes,
        fontsize=7, ha='center', va='top', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
save_fig('ch2_risk_comparison_table')

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("VaR / ES / BACKTEST / BASEL TEST COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_var_backtest.pdf:           Rolling VaR backtest + Basel test")
print("  - ch2_risk_comparison_table.pdf:  VaR/ES comparison table")
