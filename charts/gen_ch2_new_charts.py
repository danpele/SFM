"""
gen_ch2_new_charts.py
=====================
Generate 3 charts from real S&P 500 data to replace TikZ diagrams in Ch2 RO:
  1. ch2_normal_vs_empirical.pdf  — Normal vs empirical density with tail shading
  2. ch2_volatility_clustering.pdf — Real S&P 500 returns showing clustering
  3. ch2_skew_t_densities.pdf     — Fitted Skew-t densities comparison

All charts: legend outside bottom, QL logo bottom-right.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ── Style (match existing charts) ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
})

# ── Colors (match Beamer palette) ─────────────────────────────────────────
MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Amber    = '#B5853F'

# ── QL logo ───────────────────────────────────────────────────────────────
ql_logo = Image.open('../logos/ql_logo.png')

def add_ql_logo(fig, x=0.97, y=0.02, zoom=0.04):
    """Add QL logo to bottom-right of figure."""
    imagebox = OffsetImage(np.array(ql_logo), zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False,
                        xycoords='figure fraction',
                        box_alignment=(1.0, 0.0))
    fig.add_artist(ab)

def save(name):
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"  Saved: {name}.pdf + .png")

# ── Download S&P 500 data ─────────────────────────────────────────────────
print("Downloading S&P 500 data...")
import yfinance as yf
sp = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                 auto_adjust=True, progress=False)
close = sp['Close'].squeeze()
ret = np.log(close / close.shift(1)).dropna().values
mu, sigma = ret.mean(), ret.std()
print(f"  {len(ret)} daily log-returns loaded\n")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: Normal vs Empirical density (Section 1)
# ═══════════════════════════════════════════════════════════════════════════
print("1. Normal vs Empirical density...")
fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Histogram
ax.hist(ret, bins=200, density=True, color='#CCCCCC', edgecolor='none',
        alpha=0.7, label='S&P 500 histogram')

# Fitted Normal
x = np.linspace(-0.08, 0.08, 500)
ax.plot(x, stats.norm.pdf(x, mu, sigma), color=MainBlue, lw=1.5,
        label=f'Normal ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})')

# Kernel density (empirical)
kde = gaussian_kde(ret, bw_method=0.15)
ax.plot(x, kde(x), color=Crimson, lw=1.5, label='KDE empiric')

# Shade left tail discrepancy
tail_x = np.linspace(-0.08, mu - 2.5*sigma, 200)
ax.fill_between(tail_x, kde(tail_x), stats.norm.pdf(tail_x, mu, sigma),
                where=kde(tail_x) > stats.norm.pdf(tail_x, mu, sigma),
                color=Crimson, alpha=0.25, label='Hidden tail risk')

# Shade right tail too
tail_r = np.linspace(mu + 2.5*sigma, 0.08, 200)
ax.fill_between(tail_r, kde(tail_r), stats.norm.pdf(tail_r, mu, sigma),
                where=kde(tail_r) > stats.norm.pdf(tail_r, mu, sigma),
                color=Crimson, alpha=0.25)

ax.set_xlabel('Daily log-return')
ax.set_ylabel('Density')
ax.set_xlim(-0.07, 0.07)
ax.set_title('Normal vs. reality — S&P 500 (2000–2025)', fontsize=9)

# Legend outside bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)

add_ql_logo(fig)
save('ch2_normal_vs_empirical')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Volatility clustering (Section 4)
# ═══════════════════════════════════════════════════════════════════════════
print("2. Volatility clustering...")
fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.6), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08})

dates = close.index[1:]
ret_series = pd.Series(ret, index=dates)

# Top panel: returns
ax1 = axes[0]
colors = np.where(np.abs(ret) > 2*sigma, Crimson, MainBlue)
ax1.bar(dates, ret, width=1, color=colors, linewidth=0, alpha=0.8)
ax1.axhline(0, color='gray', lw=0.3)
ax1.axhline(2*sigma, color=Crimson, lw=0.4, ls='--', alpha=0.5)
ax1.axhline(-2*sigma, color=Crimson, lw=0.4, ls='--', alpha=0.5)
ax1.set_ylabel('$r_t$')
ax1.set_title('Volatility clustering — S&P 500 (2000–2025)', fontsize=9)

# Bottom panel: |r_t|
ax2 = axes[1]
abs_ret = np.abs(ret)
ax2.bar(dates, abs_ret, width=1, color='#AAAAAA', linewidth=0, alpha=0.6)
rolling_vol = ret_series.abs().rolling(20).mean()
ax2.plot(dates, rolling_vol, color=Crimson, lw=0.8, label='$|r_t|$ 20-day MA')
ax2.set_ylabel('$|r_t|$')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend outside bottom (below x-axis ticks)
handles1 = [plt.Line2D([0],[0], color=MainBlue, lw=3, alpha=0.8, label='$|r_t| \\leq 2\\sigma$'),
            plt.Line2D([0],[0], color=Crimson, lw=3, alpha=0.8, label='$|r_t| > 2\\sigma$'),
            plt.Line2D([0],[0], color=Crimson, lw=0.8, label='$|r_t|$ 20-day MA')]
fig.legend(handles=handles1, loc='lower center', bbox_to_anchor=(0.5, -0.06),
           ncol=3, fontsize=7)

add_ql_logo(fig, y=-0.06)
save('ch2_volatility_clustering')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: Skew-t densities comparison — dual panel (Section 6)
# ═══════════════════════════════════════════════════════════════════════════
print("3. Skew-t densities comparison (dual panel)...")

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(8.5, 3.2),
                                      gridspec_kw={'wspace': 0.3})

x = np.linspace(-0.06, 0.06, 500)

# Fit models once
df_t, loc_t, scale_t = stats.t.fit(ret)
a_skew, loc_sn, scale_sn = stats.skewnorm.fit(ret)

for ax, is_log in [(ax_lin, False), (ax_log, True)]:
    # Histogram
    ax.hist(ret, bins=200, density=True, color='#CCCCCC', edgecolor='none',
            alpha=0.7, label='S&P 500')

    # Fitted Normal
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=MainBlue, lw=1.2,
            label='Normal')

    # Fitted Student-t (symmetric)
    ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), color=Forest, lw=1.2,
            label=f'Student-$t$ ($\\nu$={df_t:.1f})')

    # Skew-t approximation using skewnorm
    ax.plot(x, stats.skewnorm.pdf(x, a_skew, loc_sn, scale_sn), color=Crimson,
            lw=1.2, label=f'Skewed ($a$={a_skew:.2f})')

    ax.set_xlabel('Daily log-return')
    ax.set_xlim(-0.06, 0.06)

    if is_log:
        ax.set_yscale('log')
        ax.set_ylim(0.05, 200)
        ax.set_ylabel('Density (log)')
        ax.set_title('Logarithmic scale', fontsize=9)
        # Annotate left tail on log panel
        ax.annotate('heavier\nleft tail', xy=(-0.04, 0.8), fontsize=7,
                    color=Crimson, ha='center',
                    arrowprops=dict(arrowstyle='->', color=Crimson, lw=0.8),
                    xytext=(-0.035, 5))
    else:
        ax.set_ylabel('Density')
        ax.set_title('Linear scale', fontsize=9)

# Single legend below both panels
handles, labels = ax_lin.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=4, fontsize=7)

add_ql_logo(fig)
save('ch2_skew_t_densities')

print("\nDone! All 3 charts generated.")
