"""
Generate Geometric Brownian Motion sample paths for Ch1 illustration.
Shows multiple GBM paths + log-normal terminal distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

np.random.seed(42)

# Parameters
P0 = 100
mu = 0.08       # drift 8%
sigma = 0.20    # vol 20%
T = 1.0         # 1 year
n_steps = 252   # daily
n_paths = 50
dt = T / n_steps

# Simulate GBM paths
t = np.linspace(0, T, n_steps + 1)
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = P0

for i in range(n_paths):
    Z = np.random.standard_normal(n_steps)
    for j in range(n_steps):
        paths[i, j+1] = paths[i, j] * np.exp((mu - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z[j])

# Terminal values
P_T = paths[:, -1]

# Also simulate 5000 terminal values for histogram
n_sim = 5000
Z_all = np.random.standard_normal(n_sim)
P_T_all = P0 * np.exp((mu - sigma**2/2)*T + sigma*np.sqrt(T)*Z_all)

# Theoretical log-normal PDF
from scipy.stats import lognorm
log_mu = np.log(P0) + (mu - sigma**2/2)*T
log_sigma = sigma * np.sqrt(T)
x_range = np.linspace(60, 180, 200)
pdf_vals = lognorm.pdf(x_range, s=log_sigma, scale=np.exp(log_mu))

# ---- Plot ----
fig = plt.figure(figsize=(10, 4.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

# Left: paths
ax1 = fig.add_subplot(gs[0])
for i in range(n_paths):
    alpha = 0.4 if i > 4 else 0.8
    lw = 0.5 if i > 4 else 1.0
    color = '#1a3a6e' if i > 4 else ['#cd0000', '#2e7d32', '#b5853f', '#e67e22', '#8e44ad'][i]
    ax1.plot(t * 252, paths[i], color=color, alpha=alpha, lw=lw)

# Mean path
mean_path = P0 * np.exp(mu * t)
median_path = P0 * np.exp((mu - sigma**2/2) * t)
ax1.plot(t * 252, mean_path, 'k--', lw=1.5, label=r'$E[P_t] = P_0 e^{\mu t}$ (mean)')
ax1.plot(t * 252, median_path, 'k:', lw=1.5, label=r'$\mathrm{Med}[P_t] = P_0 e^{(\mu-\sigma^2/2)t}$ (median)')

ax1.set_xlabel('Trading days', fontsize=9)
ax1.set_ylabel('Price ($P_t$)', fontsize=9)
ax1.set_title('Simulated GBM paths', fontsize=10, fontweight='bold')
ax1.set_xlim(0, 252)
ax1.tick_params(labelsize=8)

# Right: terminal distribution (horizontal histogram)
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax2.hist(P_T_all, bins=50, orientation='horizontal', density=True,
         color='#1a3a6e', alpha=0.5, edgecolor='white', lw=0.3)
ax2.plot(pdf_vals, x_range, color='#cd0000', lw=2, label='Log-normal\nteoretic')
ax2.axhline(y=np.median(P_T_all), color='k', ls=':', lw=1, alpha=0.7)
ax2.axhline(y=np.mean(P_T_all), color='k', ls='--', lw=1, alpha=0.7)
ax2.set_xlabel('Density', fontsize=9)
ax2.set_title(r'$P_T$ at $T=1$ year', fontsize=10, fontweight='bold')
ax2.tick_params(labelleft=False, labelsize=8)
ax2.set_ylim(ax1.get_ylim())

# Collect all legends outside bottom
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2,
           loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=3, frameon=False, fontsize=7)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('ch1_gbm_paths.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('ch1_gbm_paths.png', bbox_inches='tight', dpi=300, transparent=True)
print("Saved ch1_gbm_paths.pdf and .png")
plt.close()
