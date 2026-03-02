"""
SFM_ch1_clt_illustration
=========================
Central Limit Theorem illustration for financial returns

Description:
- Shows how the distribution of sum of returns converges to Normal
  as the number of observations increases
- Uses simulated returns from a heavy-tailed distribution (Student-t)

Statistics of Financial Markets course
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Chart style settings
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
plt.rcParams['legend.fontsize'] = 7
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
print("SFM CHAPTER 1: CLT ILLUSTRATION")
print("=" * 70)

np.random.seed(42)

# Simulate returns from Student-t(df=5) - heavy-tailed
df_t = 5
n_sim = 100000

# Different aggregation periods
periods = [1, 5, 22, 66]
labels = ['$k=1$ (zilnic)', '$k=5$ (săptămânal)', '$k=22$ (lunar)', '$k=66$ (trimestrial)']
colors = [CRIMSON, ORANGE, FOREST, MAIN_BLUE]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)

for i, (k, label, color) in enumerate(zip(periods, labels, colors)):
    ax = axes[i]

    # Generate n_sim sums of k i.i.d. t-distributed returns
    returns = stats.t.rvs(df=df_t, size=(n_sim, k))
    sums = returns.sum(axis=1)

    # Standardize: (sum - mean) / std
    standardized = (sums - sums.mean()) / sums.std()

    # Plot histogram
    ax.hist(standardized, bins=80, density=True, alpha=0.6, color=color,
            edgecolor='none', label='Empiric')

    # Overlay normal density
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x), color='black', linewidth=1.0,
            linestyle='--', label='$\\mathcal{N}(0,1)$')

    ax.set_title(label, fontsize=8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.55)

    if i == 0:
        ax.set_ylabel('Densitate')

    ax.legend(loc='upper right', fontsize=6)

fig.suptitle('Convergența TLC: suma de variabile $t_5$ (cozi groase)',
             fontsize=9, y=1.02)
plt.tight_layout()
save_fig('sfm_ch1_clt')

print("\n" + "=" * 70)
print("CLT ILLUSTRATION COMPLETE")
print("=" * 70)
