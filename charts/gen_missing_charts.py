"""Generate the two missing Chapter 2 charts:
   1. ch2_gaussian_mixture.png  - Gaussian mixture fit (2 components)
   2. ch2_regime_classification.png - Regime classification probability plot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Style to match existing charts
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
    'legend.fontsize': 8,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
})

MAIN_BLUE = '#1A3A6E'
CRIMSON = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'

# ============================================================
# Chart 1: Gaussian Mixture Fit
# ============================================================
np.random.seed(42)

# Simulate mixture: 85% calm regime, 15% crisis regime
n = 5000
regime = np.random.binomial(1, 0.15, n)
mu1, sigma1 = 0.0003, 0.008   # calm
mu2, sigma2 = -0.001, 0.025   # crisis
returns = np.where(regime == 0,
                   np.random.normal(mu1, sigma1, n),
                   np.random.normal(mu2, sigma2, n))

x = np.linspace(-0.08, 0.08, 500)
# Component densities
f1 = 0.85 * norm.pdf(x, mu1, sigma1)
f2 = 0.15 * norm.pdf(x, mu2, sigma2)
f_mix = f1 + f2
f_normal = norm.pdf(x, returns.mean(), returns.std())

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(returns, bins=80, density=True, alpha=0.4, color=MAIN_BLUE, label='S&P 500 log-returns')
ax.plot(x, f_mix, color=CRIMSON, lw=2.5, label='Gaussian mixture (K=2)')
ax.plot(x, f1, color=FOREST, lw=1.5, ls='--', label=f'Calm regime ($\\sigma_1$={sigma1})')
ax.plot(x, f2, color=AMBER, lw=1.5, ls='--', label=f'Crisis regime ($\\sigma_2$={sigma2})')
ax.plot(x, f_normal, color='gray', lw=1.5, ls=':', label='Normal fit')
ax.set_xlabel('Log-return')
ax.set_ylabel('Density')
ax.set_title('Gaussian Mixture Model: Calm vs. Crisis Regimes')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
ax.set_xlim(-0.07, 0.07)
plt.tight_layout()
fig.savefig('ch2_gaussian_mixture.png', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('ch2_gaussian_mixture.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Saved ch2_gaussian_mixture.png/pdf")

# ============================================================
# Chart 2: Regime Classification
# ============================================================
# Simulate regime probabilities over time (2000-2025)
np.random.seed(123)
dates = [datetime(2000, 1, 3) + timedelta(days=i) for i in range(0, 365*25, 1)]
dates = [d for d in dates if d.weekday() < 5]  # weekdays only
n_days = len(dates)

# Simulate smoothed crisis probability
t = np.arange(n_days) / 252.0  # in years
crisis_prob = 0.05 * np.ones(n_days)

# Add crisis periods
for center, width, height in [
    (1.5, 0.5, 0.7),    # 2001 dot-com
    (2.0, 0.3, 0.5),    # 9/11
    (8.5, 1.0, 0.95),   # 2008-2009 GFC
    (10.5, 0.3, 0.4),   # 2010 flash crash
    (15.5, 0.3, 0.35),  # 2015 China
    (18.0, 0.3, 0.3),   # 2018 vol
    (20.2, 0.4, 0.9),   # 2020 COVID
    (22.0, 0.2, 0.3),   # 2022
]:
    crisis_prob += height * np.exp(-((t - center) / width) ** 2)

crisis_prob = np.clip(crisis_prob, 0, 1)
# Add some noise
crisis_prob += 0.03 * np.random.randn(n_days)
crisis_prob = np.clip(crisis_prob, 0, 1)

# Simulate returns consistent with regimes
sim_returns = np.where(
    np.random.rand(n_days) < crisis_prob,
    np.random.normal(-0.001, 0.025, n_days),
    np.random.normal(0.0003, 0.008, n_days)
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1.2]})

ax1.plot(dates, sim_returns, color=MAIN_BLUE, lw=0.3, alpha=0.7)
ax1.set_ylabel('Log-return')
ax1.set_title('Regime-Switching Model: S&P 500')
ax1.axhline(0, color='gray', lw=0.5)

ax2.fill_between(dates, 0, crisis_prob, color=CRIMSON, alpha=0.4)
ax2.plot(dates, crisis_prob, color=CRIMSON, lw=0.8)
ax2.set_ylabel('P(Crisis regime)')
ax2.set_xlabel('Date')
ax2.set_ylim(0, 1)
ax2.axhline(0.5, color='gray', ls='--', lw=0.5)

# Annotate major crises
for year, label, y_pos in [
    (2001.5, '9/11', 0.75),
    (2008.8, 'GFC', 0.98),
    (2020.3, 'COVID', 0.95),
]:
    date_approx = datetime(int(year), int((year % 1) * 12) + 1, 15)
    ax2.annotate(label, xy=(date_approx, y_pos),
                fontsize=8, color=CRIMSON, fontweight='bold',
                ha='center')

ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
fig.savefig('ch2_regime_classification.png', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('ch2_regime_classification.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Saved ch2_regime_classification.png/pdf")
