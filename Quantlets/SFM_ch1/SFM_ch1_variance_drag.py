"""
SFM_ch1_variance_drag
=====================
Variance Drag Visualization

Description:
- Plot the variance drag curve: E[R_geometric] ≈ E[R_arithmetic] - σ²/2
- Show drag as a function of annualized volatility
- Mark typical asset classes (bonds, equities, crypto)

Statistics of Financial Markets course
"""

import numpy as np
import matplotlib.pyplot as plt
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
print("SFM CHAPTER 1: VARIANCE DRAG")
print("=" * 70)

# =============================================================================
# Variance Drag Curve
# =============================================================================
print("\n1. CREATING VARIANCE DRAG CHART")
print("-" * 40)

fig, ax = plt.subplots(figsize=(7, 3))

# Volatility range 0-100%
sigma = np.linspace(0, 1.0, 500)
drag = sigma**2 / 2 * 100  # in percentage points

ax.plot(sigma * 100, drag, color=MAIN_BLUE, linewidth=1.5)
ax.fill_between(sigma * 100, 0, drag, alpha=0.08, color=MAIN_BLUE)

# Mark typical asset classes
assets = [
    ('T-Bills', 1, FOREST),
    ('Bonds', 5, FOREST),
    ('S&P 500', 16, AMBER),
    ('AAPL', 30, ORANGE),
    ('Bitcoin', 75, CRIMSON),
]

for name, vol, color in assets:
    d = (vol / 100)**2 / 2 * 100
    ax.plot(vol, d, 'o', color=color, markersize=5, zorder=5)
    # Position labels to avoid overlap
    if name == 'T-Bills':
        ax.annotate(f'{name}\n$\\sigma$={vol}%, drag={d:.2f}%',
                    xy=(vol, d), xytext=(vol + 8, d + 0.5),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    elif name == 'Bonds':
        ax.annotate(f'{name}\n$\\sigma$={vol}%, drag={d:.2f}%',
                    xy=(vol, d), xytext=(vol + 8, d + 0.3),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    elif name == 'S&P 500':
        ax.annotate(f'{name}\n$\\sigma$={vol}%, drag={d:.2f}%',
                    xy=(vol, d), xytext=(vol - 2, d + 4),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    elif name == 'AAPL':
        ax.annotate(f'{name}\n$\\sigma$={vol}%, drag={d:.1f}%',
                    xy=(vol, d), xytext=(vol + 5, d + 3),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    else:  # Bitcoin
        ax.annotate(f'{name}\n$\\sigma$={vol}%, drag={d:.1f}%',
                    xy=(vol, d), xytext=(vol - 15, d + 3),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))

ax.set_xlabel('Annualized Volatility $\\sigma$ (\\%)')
ax.set_ylabel('Variance Drag $\\sigma^2/2$ (\\%)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 55)

fig.subplots_adjust(left=0.18, right=0.92, top=0.95, bottom=0.15)
plt.savefig(os.path.join(CHART_DIR, 'sfm_ch1_variance_drag.pdf'),
            transparent=True)
plt.savefig(os.path.join(CHART_DIR, 'sfm_ch1_variance_drag.png'),
            transparent=True, dpi=300)
plt.close()
print("   Saved: sfm_ch1_variance_drag.pdf/.png")

print("\n" + "=" * 70)
print("VARIANCE DRAG CHART COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_variance_drag.pdf/.png")
