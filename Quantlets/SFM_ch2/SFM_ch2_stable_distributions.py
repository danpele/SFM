"""
SFM_ch2_stable_distributions
=============================
Stable (Levy) Distributions in Finance

Description:
- Plot PDFs for different stability indices alpha = 0.5, 1.0, 1.5, 2.0
- Compare heavy-tailed vs Gaussian distributions
- Generate histogram + KDE + Normal for Gaussian case
- QQ-plot comparison with Normal distribution

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levy_stable, probplot
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

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("SFM CHAPTER 2: STABLE DISTRIBUTIONS")
print("=" * 70)

# =============================================================================
# 1. Stable Distribution Parameters
# =============================================================================
print("\n1. STABLE DISTRIBUTION OVERVIEW")
print("-" * 40)

print("   S(alpha, beta, gamma, delta):")
print("   alpha: stability index (0 < alpha <= 2)")
print("   beta:  skewness parameter (-1 <= beta <= 1)")
print("   gamma: scale parameter (gamma > 0)")
print("   delta: location parameter")
print("\n   Special cases:")
print("   alpha = 2.0: Gaussian distribution")
print("   alpha = 1.0: Cauchy distribution (beta=0)")
print("   alpha = 0.5: Levy distribution (beta=1)")

# =============================================================================
# 2. Simulate Stable Distributions for Different Alpha
# =============================================================================
print("\n2. SIMULATING STABLE DISTRIBUTIONS")
print("-" * 40)

np.random.seed(42)
n = 50000

alphas = [0.5, 1.0, 1.5, 2.0]
beta = 0
samples = {}

for a in alphas:
    rv = levy_stable(alpha=a, beta=beta)
    samples[a] = rv.rvs(size=n)
    # Clip extreme values for plotting
    samples[a] = np.clip(samples[a], -20, 20)
    print(f"   alpha={a:.1f}: mean={np.mean(samples[a]):.4f}, "
          f"std={np.std(samples[a]):.4f}")

# =============================================================================
# 3. FIGURE: Stable Distributions (4-panel)
# =============================================================================
print("\n3. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#DC3545', '#FF8C00', '#2E7D32', '#1A3A6E']
labels = [r'$\alpha=0.5$ (Levy)',
          r'$\alpha=1.0$ (Cauchy)',
          r'$\alpha=1.5$',
          r'$\alpha=2.0$ (Gaussian)']

# Panel A: PDF comparison for different alpha (linear scale)
x = np.linspace(-8, 8, 500)
for a, c, label in zip(alphas, colors, labels):
    rv = levy_stable(alpha=a, beta=0)
    pdf_vals = rv.pdf(x)
    axes[0, 0].plot(x, pdf_vals, color=c, linewidth=1.5, label=label)
axes[0, 0].set_title('Stable Distribution PDFs (linear scale)',
                       fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].set_ylim(0, 0.45)
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

# Panel B: PDF comparison on log scale (tail behavior)
for a, c, label in zip(alphas, colors, labels):
    rv = levy_stable(alpha=a, beta=0)
    pdf_vals = rv.pdf(x)
    axes[0, 1].semilogy(x, pdf_vals, color=c, linewidth=1.5, label=label)
axes[0, 1].set_title('Log-Scale PDF (Tail Behavior)', fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('log f(x)')
axes[0, 1].set_ylim(1e-6, 1)
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

# Panel C: Histogram + KDE + Normal for alpha=1.5
alpha_demo = 1.5
sample_15 = samples[alpha_demo]

axes[1, 0].hist(sample_15, bins=100, density=True, alpha=0.4,
                color='#1A3A6E', edgecolor='white', label='Stable samples')

x_hist = np.linspace(-8, 8, 500)
pdf_stable = levy_stable.pdf(x_hist, alpha_demo, beta)
pdf_normal = stats.norm.pdf(x_hist, loc=0,
                            scale=np.std(sample_15))

axes[1, 0].plot(x_hist, pdf_stable, color='#DC3545', linewidth=2,
                label=f'Stable PDF ($\\alpha$={alpha_demo})')
axes[1, 0].plot(x_hist, pdf_normal, color='#2E7D32', linewidth=2,
                linestyle='--', label='Normal PDF')
axes[1, 0].set_title(f'Stable($\\alpha$={alpha_demo}) vs Normal',
                       fontweight='bold')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_xlim(-8, 8)
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)

kurt_15 = stats.kurtosis(sample_15)
axes[1, 0].text(0.95, 0.95,
               f'Kurtosis: {kurt_15:.2f}',
               transform=axes[1, 0].transAxes, ha='right', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

print(f"   Stable(alpha={alpha_demo}) sample kurtosis: {kurt_15:.4f}")

# Panel D: Gaussian case (alpha=2) - histogram + KDE + Normal
samples_gauss = samples[2.0]

axes[1, 1].hist(samples_gauss, bins=80, density=True, alpha=0.4,
                color='#1A3A6E', edgecolor='white',
                label='Stable($\\alpha$=2) samples')

x_gauss = np.linspace(-5, 5, 500)
pdf_stable_g = levy_stable.pdf(x_gauss, 2.0, beta)
pdf_normal_g = stats.norm.pdf(x_gauss, loc=np.mean(samples_gauss),
                               scale=np.std(samples_gauss))

axes[1, 1].plot(x_gauss, pdf_stable_g, color='#DC3545', linewidth=2,
                label='Stable PDF ($\\alpha$=2)')
axes[1, 1].plot(x_gauss, pdf_normal_g, color='#2E7D32', linewidth=2,
                linestyle='--', label='Normal PDF (fitted)')
axes[1, 1].set_title('Stable($\\alpha$=2) = Gaussian', fontweight='bold')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)

# Verify alpha=2 equals Normal
ks_stat, ks_pval = stats.kstest(
    samples_gauss / np.std(samples_gauss), 'norm')
print(f"   Gaussian verification (alpha=2):")
print(f"     KS test: stat={ks_stat:.6f}, p-value={ks_pval:.6f}")

plt.tight_layout()
save_fig('ch2_stable_distributions')

# =============================================================================
# 4. Tail Behavior Analysis
# =============================================================================
print("\n4. TAIL BEHAVIOR ANALYSIS")
print("-" * 40)

print(f"   {'Alpha':<8} {'P(|X|>3)':>12} {'P(|X|>5)':>12} "
      f"{'Kurtosis':>12}")
print("   " + "-" * 48)
for a in alphas:
    s = samples[a]
    p3 = np.mean(np.abs(s) > 3)
    p5 = np.mean(np.abs(s) > 5)
    kurt = stats.kurtosis(s)
    print(f"   {a:<8.1f} {p3:>12.6f} {p5:>12.6f} {kurt:>12.2f}")

print("\n   Normal:   P(|X|>3) = 0.0027, P(|X|>5) = 5.7e-7")

print("\n" + "=" * 70)
print("STABLE DISTRIBUTIONS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_stable_distributions.pdf: 4-panel stable distribution analysis")
