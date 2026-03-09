"""
gen_ch2_sp500_history.py
========================
Generate a dual-panel chart showing S&P 500 history:
  Top: Price level (log scale) with major crashes annotated
  Bottom: Daily log-returns with extreme events highlighted

Output: ch2_sp500_history.pdf + .png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
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
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
})

MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Amber    = '#B5853F'

# ── QL logo ───────────────────────────────────────────────────────────────
ql_logo = Image.open('../logos/ql_logo.png')

def add_ql_logo(fig, x=0.97, y=0.02, zoom=0.04):
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
print("Downloading S&P 500 data (1985-2025)...")
import yfinance as yf
sp = yf.download('^GSPC', start='1985-01-01', end='2025-12-31',
                 auto_adjust=True, progress=False)
close = sp['Close'].squeeze()
ret = np.log(close / close.shift(1)).dropna()
print(f"  {len(ret)} daily returns loaded\n")

# ── Major crash events ────────────────────────────────────────────────────
crashes = [
    ('1987-10-19', 'Black Monday\n−20.5%',       -0.08, 0.04),
    ('1997-10-27', 'Asian crisis\n−6.9%',        -0.06, 0.03),
    ('2001-09-17', '9/11\n−4.9%',                -0.05, 0.03),
    ('2008-10-15', 'Financial crisis\n−9.0%',    -0.07, 0.04),
    ('2010-05-06', 'Flash Crash\n−3.2%',         -0.04, 0.02),
    ('2020-03-16', 'COVID-19\n−12.0%',           -0.06, 0.04),
]

# ═══════════════════════════════════════════════════════════════════════════
print("Generating S&P 500 history chart...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True,
                                gridspec_kw={'height_ratios': [1.3, 1], 'hspace': 0.08})

# ── Top panel: Price (log scale) ──────────────────────────────────────────
ax1.plot(close.index, close.values, color=MainBlue, lw=0.6, alpha=0.9)
ax1.set_yscale('log')
ax1.set_ylabel('S&P 500 (log scale)')
ax1.set_title('S&P 500: Prices and daily returns (1985–2025)', fontsize=9)

# Shade crash periods
crisis_periods = [
    ('1987-10-01', '1987-12-31', 'Black Monday'),
    ('2007-10-01', '2009-03-31', 'Financial crisis'),
    ('2020-02-15', '2020-04-30', 'COVID-19'),
]
for start, end, label in crisis_periods:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                color=Crimson, alpha=0.08)

# ── Bottom panel: Daily returns ───────────────────────────────────────────
dates = ret.index
sigma = ret.std()
colors = np.where(np.abs(ret.values) > 3*sigma, Crimson, '#AAAAAA')
ax2.bar(dates, ret.values, width=1, color=colors, linewidth=0, alpha=0.7)
ax2.axhline(0, color='gray', lw=0.3)
ax2.axhline(3*sigma, color=Crimson, lw=0.3, ls='--', alpha=0.4)
ax2.axhline(-3*sigma, color=Crimson, lw=0.3, ls='--', alpha=0.4)
ax2.set_ylabel('$r_t = \\ln(P_t/P_{t-1})$')

# Annotate major crashes
for date_str, label, y_offset, arr_len in crashes:
    date = pd.Timestamp(date_str)
    if date in ret.index:
        ret_val = ret.loc[date]
        ax2.annotate(label, xy=(date, ret_val),
                     xytext=(date, ret_val + y_offset),
                     fontsize=5.5, color=Crimson, ha='center',
                     arrowprops=dict(arrowstyle='->', color=Crimson, lw=0.6))

# ── Legend ────────────────────────────────────────────────────────────────
handles = [
    plt.Line2D([0], [0], color='#AAAAAA', lw=4, alpha=0.7, label='$|r_t| \\leq 3\\sigma$'),
    plt.Line2D([0], [0], color=Crimson, lw=4, alpha=0.7, label='$|r_t| > 3\\sigma$ (extreme)'),
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.04),
           ncol=2, fontsize=7)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

add_ql_logo(fig, y=-0.04)
save('ch2_sp500_history')
print("\nDone!")
