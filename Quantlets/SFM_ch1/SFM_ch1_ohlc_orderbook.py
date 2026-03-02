"""
SFM_ch1_ohlc_orderbook
======================
OHLC Candlestick Chart and Order Book Visualization

Description:
- Download recent AAPL OHLC data via yfinance (5 trading days)
- Create candlestick chart using mplfinance
- Simulate order book depth chart (bid/ask levels)

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

# Chart style settings - Nature journal quality
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

# Color palette
MAIN_BLUE = '#1A3A6E'
CRIMSON = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'
ORANGE = '#E67E22'

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'charts'))
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
print("SFM CHAPTER 1: OHLC & ORDER BOOK CHARTS")
print("=" * 70)

# =============================================================================
# 1. Download OHLC Data
# =============================================================================
print("\n1. DOWNLOADING AAPL OHLC DATA")
print("-" * 40)

data = yf.download('AAPL', period='1mo', interval='1d', progress=False)
# Take last 5 trading days
data = data.tail(5).copy()

print(f"   Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
print(f"   Days: {len(data)}")

# =============================================================================
# 2. Candlestick Chart
# =============================================================================
print("\n2. CREATING CANDLESTICK CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

opens = data['Open'].values.flatten()
highs = data['High'].values.flatten()
lows = data['Low'].values.flatten()
closes = data['Close'].values.flatten()
dates = data.index

bar_width = 0.5

for i in range(len(data)):
    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
    color = FOREST if c >= o else CRIMSON

    # Wick (high-low line)
    ax.plot([i, i], [l, h], color=color, linewidth=0.8)

    # Body (open-close rectangle)
    body_bottom = min(o, c)
    body_height = abs(c - o)
    rect = mpatches.FancyBboxPatch(
        (i - bar_width / 2, body_bottom), bar_width, body_height,
        boxstyle="square,pad=0", facecolor=color, edgecolor=color,
        linewidth=0.5, alpha=0.85
    )
    ax.add_patch(rect)

# Format x-axis with day names
day_labels = [d.strftime('%a\n%b %d') for d in dates]
ax.set_xticks(range(len(data)))
ax.set_xticklabels(day_labels, fontsize=7)

ax.set_ylabel('Price ($)')
ax.set_xlim(-0.6, len(data) - 0.4)

# Add legend
legend_elements = [
    Line2D([0], [0], color=FOREST, linewidth=3, label='Up (Close > Open)'),
    Line2D([0], [0], color=CRIMSON, linewidth=3, label='Down (Close < Open)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('sfm_ch1_candlestick')

# =============================================================================
# 3. Order Book Depth Chart
# =============================================================================
print("\n3. CREATING ORDER BOOK CHART")
print("-" * 40)

# Simulated order book around a mid price
mid_price = 100.06

# Ask levels (above mid)
ask_prices = [100.10, 100.12, 100.15, 100.18, 100.22]
ask_sizes = [350, 200, 480, 150, 300]

# Bid levels (below mid)
bid_prices = [100.02, 100.00, 99.97, 99.94, 99.90]
bid_sizes = [600, 800, 420, 550, 280]

fig, ax = plt.subplots(figsize=(7, 3))

# Cumulative sizes for depth visualization
ask_cum = np.cumsum(ask_sizes)
bid_cum = np.cumsum(bid_sizes)

# Plot asks (red, stepping up)
ax.barh(ask_prices, ask_sizes, height=0.018, color=CRIMSON, alpha=0.35,
        edgecolor=CRIMSON, linewidth=0.5, label='Ask (sell)')

# Plot bids (green, stepping down)
ax.barh(bid_prices, bid_sizes, height=0.018, color=FOREST, alpha=0.35,
        edgecolor=FOREST, linewidth=0.5, label='Bid (buy)')

# Mid-price line
ax.axhline(y=mid_price, color=MAIN_BLUE, linestyle='--', linewidth=0.8,
           label=f'Mid ${mid_price:.2f}')

# Annotate spread
ax.annotate('', xy=(850, 100.10), xytext=(850, 100.02),
            arrowprops=dict(arrowstyle='<->', color=AMBER, lw=1.2))
ax.text(870, 100.06, f'Spread\n$0.08', fontsize=7, color=AMBER,
        ha='left', va='center')

# Add size labels
for p, s in zip(ask_prices[:2], ask_sizes[:2]):
    ax.text(s + 15, p, str(s), fontsize=6, va='center', color=CRIMSON)
for p, s in zip(bid_prices[:2], bid_sizes[:2]):
    ax.text(s + 15, p, str(s), fontsize=6, va='center', color=FOREST)

ax.set_xlabel('Order Size (shares)')
ax.set_ylabel('Price ($)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)
ax.set_xlim(0, 1050)

plt.tight_layout()
save_fig('sfm_ch1_orderbook')

print("\n" + "=" * 70)
print("OHLC & ORDER BOOK CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_candlestick.pdf/.png")
print("  - sfm_ch1_orderbook.pdf/.png")
