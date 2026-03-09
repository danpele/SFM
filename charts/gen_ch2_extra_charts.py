"""
gen_ch2_extra_charts.py
=======================
Generate additional pedagogical charts for Ch2 RO:
  1. ch2_gbm_vs_levy.pdf       — GBM paths vs Lévy jump-diffusion paths
  2. ch2_sample_var_diverge.pdf — Sample variance divergence (stable vs Normal)
  3. ch2_hill_plot.pdf          — Hill plot for S&P 500 tail index
  4. ch2_mean_excess_plot.pdf   — Mean Excess plot for threshold selection
  5. ch2_var_subadditivity.pdf  — VaR non-subadditivity counterexample
  6. ch2_spectral_weights.pdf   — Spectral risk measure weight functions

All charts: QL logo bottom-right, transparent background.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────
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

MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Amber    = '#B5853F'
Purple   = '#8E44AD'

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

np.random.seed(42)

# ── Download S&P 500 data ─────────────────────────────────────────────────
print("Downloading S&P 500 data...")
import yfinance as yf
sp = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                 auto_adjust=True, progress=False)
close = sp['Close'].squeeze()
ret = np.log(close / close.shift(1)).dropna().values
print(f"  {len(ret)} daily log-returns loaded\n")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: GBM paths vs Lévy jump-diffusion
# ═══════════════════════════════════════════════════════════════════════════
print("1. GBM vs Lévy jump-diffusion paths...")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True,
                         gridspec_kw={'wspace': 0.12})

T, n_steps, S0 = 1.0, 252, 100
dt = T / n_steps
mu, sigma = 0.08, 0.20
n_paths = 5

# Left panel: GBM
ax = axes[0]
for i in range(n_paths):
    Z = np.random.randn(n_steps)
    log_ret = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_ret))
    S = np.insert(S, 0, S0)
    color = MainBlue if i > 0 else Crimson
    alpha = 0.5 if i > 0 else 0.9
    ax.plot(np.linspace(0, T, n_steps+1), S, color=color, lw=0.7, alpha=alpha)
ax.set_title('Geometric Brownian motion', fontsize=9)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Price $S_t$')
ax.axhline(S0, color='gray', lw=0.3, ls='--')

# Right panel: Jump-diffusion (Merton)
ax = axes[1]
lam, mu_J, sigma_J = 5, -0.02, 0.04  # ~5 jumps/year
for i in range(n_paths):
    Z = np.random.randn(n_steps)
    N_jumps = np.random.poisson(lam * dt, n_steps)
    J = np.zeros(n_steps)
    for k in range(n_steps):
        if N_jumps[k] > 0:
            J[k] = np.sum(np.random.normal(mu_J, sigma_J, N_jumps[k]))
    log_ret = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z + J
    S = S0 * np.exp(np.cumsum(log_ret))
    S = np.insert(S, 0, S0)
    color = MainBlue if i > 0 else Crimson
    alpha = 0.5 if i > 0 else 0.9
    ax.plot(np.linspace(0, T, n_steps+1), S, color=color, lw=0.7, alpha=alpha)
ax.set_title('Jump-diffusion (Merton)', fontsize=9)
ax.set_xlabel('Time (years)')
ax.axhline(S0, color='gray', lw=0.3, ls='--')

fig.legend(['Simulated paths'], loc='upper center',
           bbox_to_anchor=(0.5, -0.02), fontsize=7)
add_ql_logo(fig)
save('ch2_gbm_vs_levy')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Sample variance divergence (stable vs Normal/Student-t)
# ═══════════════════════════════════════════════════════════════════════════
print("2. Sample variance divergence...")
fig, ax = plt.subplots(figsize=(5, 3.2))

n_max = 5000
# Normal: variance converges
normal_data = np.random.randn(n_max) * 0.015
cum_var_normal = np.array([np.var(normal_data[:k]) for k in range(50, n_max+1, 10)])
ns = np.arange(50, n_max+1, 10)

# Student-t (nu=4): variance exists but converges slowly
t4_data = stats.t.rvs(4, scale=0.015/np.sqrt(2), size=n_max)
cum_var_t4 = np.array([np.var(t4_data[:k]) for k in range(50, n_max+1, 10)])

# Stable (alpha=1.7): variance diverges
# Simulate stable-like using Chambers-Mallows-Stuck method
alpha_s = 1.7
V = np.random.uniform(-np.pi/2, np.pi/2, n_max)
W = np.random.exponential(1, n_max)
stable_data = (np.sin(alpha_s * V) / np.cos(V)**(1/alpha_s)) * \
              (np.cos(V - alpha_s*V) / W)**((1-alpha_s)/alpha_s)
stable_data *= 0.005  # scale to financial-like
cum_var_stable = np.array([np.var(stable_data[:k]) for k in range(50, n_max+1, 10)])

ax.plot(ns, cum_var_normal * 1e4, color=MainBlue, lw=1.2, label='Normal (finite $\\sigma^2$)')
ax.plot(ns, cum_var_t4 * 1e4, color=Forest, lw=1.2, label='Student-$t$ ($\\nu=4$, finite $\\sigma^2$)')
ax.plot(ns, cum_var_stable * 1e4, color=Crimson, lw=1.2, label='Stable ($\\alpha=1.7$, $\\sigma^2 = \\infty$)')

ax.set_xlabel('Sample size $n$')
ax.set_ylabel('Sample variance ($\\times 10^{-4}$)')
ax.set_title('Sample variance convergence', fontsize=9)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)

add_ql_logo(fig)
save('ch2_sample_var_diverge')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: Hill plot for S&P 500
# ═══════════════════════════════════════════════════════════════════════════
print("3. Hill plot...")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2),
                         gridspec_kw={'wspace': 0.3})

abs_ret = np.abs(ret)
sorted_ret = np.sort(abs_ret)[::-1]  # descending

# Hill estimator for different k
k_range = np.arange(20, 600)
hill_est = np.zeros(len(k_range))
for idx, k in enumerate(k_range):
    top_k = sorted_ret[:k]
    hill_est[idx] = 1.0 / (np.mean(np.log(top_k)) - np.log(sorted_ret[k]))

# Left panel: Hill plot (1/alpha = tail index)
ax = axes[0]
ax.plot(k_range, hill_est, color=MainBlue, lw=0.8)
ax.axhline(3.5, color=Crimson, lw=0.6, ls='--', label='$\\hat{\\alpha} \\approx 3{,}5$')
ax.fill_between(k_range, 3.0, 4.0, color=Crimson, alpha=0.1)
ax.set_xlabel('Number of order statistics $k$')
ax.set_ylabel('$\\hat{\\alpha}_{\\mathrm{Hill}}$')
ax.set_title('Hill plot — S&P 500', fontsize=9)
ax.set_ylim(1.5, 6)
ax.set_xlim(20, 600)
ax.legend(fontsize=7)

# Right panel: log-log tail plot
ax = axes[1]
n_obs = len(abs_ret)
probs = np.arange(1, n_obs+1) / (n_obs + 1)
surv = 1 - probs
sorted_asc = np.sort(abs_ret)

# Empirical
ax.plot(np.log(sorted_asc), np.log(surv), color='#AAAAAA', lw=0.5, label='Empiric')

# Fitted power law
alpha_hat = 3.5
x_fit = np.linspace(np.log(0.01), np.log(0.12), 100)
ax.plot(x_fit, -alpha_hat * x_fit + np.log(0.5), color=Crimson, lw=1.2,
        ls='--', label=f'Power law ($\\alpha={alpha_hat}$)')

# Normal tail
x_norm = np.linspace(0.005, 0.12, 200)
surv_norm = 1 - stats.norm.cdf(x_norm, 0, ret.std())
ax.plot(np.log(x_norm), np.log(surv_norm), color=MainBlue, lw=1, ls=':',
        label='Normal')

ax.set_xlabel('$\\ln|r|$')
ax.set_ylabel('$\\ln P(|R| > |r|)$')
ax.set_title('Log-log tail plot', fontsize=9)
ax.set_xlim(-5, -1.5)
ax.set_ylim(-10, 0)
ax.legend(fontsize=7)

fig.legend([], [], loc='upper center', bbox_to_anchor=(0.5, -0.02))
add_ql_logo(fig)
save('ch2_hill_plot')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: Mean Excess plot
# ═══════════════════════════════════════════════════════════════════════════
print("4. Mean Excess plot...")
fig, ax = plt.subplots(figsize=(5, 3.2))

losses = -ret  # positive losses
sorted_losses = np.sort(losses)

thresholds = np.linspace(np.percentile(losses, 80), np.percentile(losses, 99), 200)
mean_excess = np.zeros(len(thresholds))
n_exceed = np.zeros(len(thresholds))

for idx, u in enumerate(thresholds):
    exceedances = losses[losses > u] - u
    if len(exceedances) > 5:
        mean_excess[idx] = np.mean(exceedances)
        n_exceed[idx] = len(exceedances)
    else:
        mean_excess[idx] = np.nan
        n_exceed[idx] = np.nan

ax.plot(thresholds * 100, mean_excess * 100, color=MainBlue, lw=1.2)
# Mark suggested threshold
u_star = np.percentile(losses, 95)
ax.axvline(u_star * 100, color=Crimson, lw=0.8, ls='--',
           label=f'Suggested threshold $u = {u_star*100:.1f}\\%$')

ax.set_xlabel('Threshold $u$ (%)')
ax.set_ylabel('Mean excess $e(u)$ (%)')
ax.set_title('Mean Excess plot — S&P 500 daily losses', fontsize=9)
ax.legend(loc='upper left', fontsize=7)

# Annotate linear region
ax.annotate('Linear region\n(GPD suitable)', xy=(u_star*100+0.2, mean_excess[100]*100),
            fontsize=7, color=Forest, ha='left',
            arrowprops=dict(arrowstyle='->', color=Forest, lw=0.8),
            xytext=(u_star*100+0.5, mean_excess[50]*100+0.15))

add_ql_logo(fig)
save('ch2_mean_excess_plot')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: VaR non-subadditivity counterexample
# ═══════════════════════════════════════════════════════════════════════════
print("5. VaR non-subadditivity...")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2),
                         gridspec_kw={'wspace': 0.35})

n_sim = 100000
alpha = 0.05

# Two assets with Pareto-like tails (Student-t nu=3)
nu = 3
X1 = stats.t.rvs(nu, scale=0.02, size=n_sim)
X2 = stats.t.rvs(nu, scale=0.02, size=n_sim)
# Make them slightly negatively correlated (realistic)
rho = -0.1
X2 = rho * X1 + np.sqrt(1 - rho**2) * X2

portfolio = 0.5 * X1 + 0.5 * X2

VaR_X1 = -np.percentile(X1, alpha * 100)
VaR_X2 = -np.percentile(X2, alpha * 100)
VaR_port = -np.percentile(portfolio, alpha * 100)
VaR_sum = 0.5 * VaR_X1 + 0.5 * VaR_X2

# Left panel: bar chart comparison
ax = axes[0]
labels = ['$\\frac{1}{2}$VaR($X_1$)+\n$\\frac{1}{2}$VaR($X_2$)', 'VaR($\\frac{X_1+X_2}{2}$)']
values = [VaR_sum * 100, VaR_port * 100]
colors = [Forest, Crimson]
bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.8, edgecolor='none')
ax.set_ylabel('VaR 5% (%)')
ax.set_title('VaR: diversification may increase risk', fontsize=9)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

if VaR_port > VaR_sum:
    ax.annotate('VaR is not\nsubadditive!', xy=(1, VaR_port*100),
                fontsize=8, color=Crimson, ha='center',
                xytext=(1.3, VaR_port*100 + 0.3),
                arrowprops=dict(arrowstyle='->', color=Crimson, lw=0.8))

# Right panel: ES comparison (always subadditive)
ES_X1 = -np.mean(X1[X1 <= np.percentile(X1, alpha*100)])
ES_X2 = -np.mean(X2[X2 <= np.percentile(X2, alpha*100)])
ES_port = -np.mean(portfolio[portfolio <= np.percentile(portfolio, alpha*100)])
ES_sum = 0.5 * ES_X1 + 0.5 * ES_X2

ax = axes[1]
labels_es = ['$\\frac{1}{2}$ES($X_1$)+\n$\\frac{1}{2}$ES($X_2$)', 'ES($\\frac{X_1+X_2}{2}$)']
values_es = [ES_sum * 100, ES_port * 100]
colors_es = [Forest, MainBlue]
bars = ax.bar(labels_es, values_es, color=colors_es, width=0.5, alpha=0.8, edgecolor='none')
ax.set_ylabel('ES 5% (%)')
ax.set_title('ES: diversification always reduces risk', fontsize=9)

for bar, val in zip(bars, values_es):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

ax.annotate('ES is\nsubadditive ✓', xy=(1, ES_port*100),
            fontsize=8, color=Forest, ha='center',
            xytext=(1.3, ES_port*100 + 0.5),
            arrowprops=dict(arrowstyle='->', color=Forest, lw=0.8))

add_ql_logo(fig)
save('ch2_var_subadditivity')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: Spectral risk measure weight functions
# ═══════════════════════════════════════════════════════════════════════════
print("6. Spectral risk measure weights...")
fig, ax = plt.subplots(figsize=(5, 3.2))

p = np.linspace(0.001, 0.15, 1000)
alpha_level = 0.05

# VaR weight: delta function at alpha (approximate with narrow peak)
var_weight = np.zeros_like(p)
idx_var = np.argmin(np.abs(p - alpha_level))
var_weight[max(0,idx_var-2):idx_var+3] = 1.0 / (p[1]-p[0]) / 5

# ES weight: uniform on [0, alpha]
es_weight = np.where(p <= alpha_level, 1.0/alpha_level, 0)

# Exponential spectral: phi(p) = c * exp(-lambda * p), normalized
lam_spec = 30
exp_weight = lam_spec * np.exp(-lam_spec * p)
exp_weight /= np.sum(exp_weight) * (p[1] - p[0])  # normalize

ax.fill_between(p, 0, es_weight, color=MainBlue, alpha=0.2)
ax.plot(p, es_weight, color=MainBlue, lw=1.5, label=r'ES ($\alpha = 5\%$): uniform on $[0, \alpha]$')
ax.plot(p, exp_weight, color=Crimson, lw=1.2, ls='--',
        label=r'Exponential: penalizing extreme losses')

# VaR as arrow
ax.annotate('', xy=(alpha_level, max(es_weight)*1.3), xytext=(alpha_level, max(es_weight)*1.6),
            arrowprops=dict(arrowstyle='->', color=Forest, lw=2))
ax.text(alpha_level + 0.005, max(es_weight)*1.5, r'VaR: $\delta(p - 5\%)$',
        fontsize=7, color=Forest)

ax.set_xlabel('Probability $p$')
ax.set_ylabel('Weight $\\phi(p)$')
ax.set_title('Spectral risk functions', fontsize=9)
ax.set_xlim(0, 0.15)
ax.set_ylim(0, max(es_weight)*1.8)
ax.legend(loc='upper right', fontsize=7)

add_ql_logo(fig)
save('ch2_spectral_weights')

print("\nDone! All 6 extra charts generated.")
