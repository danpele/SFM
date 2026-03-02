"""
SFM_ch1_volatility_charts
=========================
Volatility Estimation Charts: Rolling, EWMA, Efficiency, Efficient Frontier

Description:
- Download AAPL OHLC data via yfinance
- 30-day rolling close-to-close volatility
- EWMA (lambda=0.94) vs Historical volatility comparison
- Horizontal bar chart of theoretical RE ratios
- Efficient frontier: simulated 5-asset portfolio, CML, tangency

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import optimize
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
PURPLE = '#7B2D8E'
DARK_GRAY = '#4A4A4A'

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
print("SFM CHAPTER 1: VOLATILITY CHARTS")
print("=" * 70)

# =============================================================================
# 1. Download AAPL Data
# =============================================================================
print("\n1. DOWNLOADING AAPL DATA")
print("-" * 40)

data = yf.download('AAPL', start='2015-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

print(f"   Period: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(log_ret)}")

# =============================================================================
# 2. Rolling Close-to-Close Volatility
# =============================================================================
print("\n2. CREATING ROLLING VOLATILITY CHART")
print("-" * 40)

window = 30
rolling_vol = log_ret.rolling(window=window).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(7.5, 3.2))

ax.plot(rolling_vol.index, rolling_vol.values, color=MAIN_BLUE, linewidth=0.7,
        label=f'{window}-day rolling $\\hat{{\\sigma}}$')

# Highlight a volatility cluster (find the highest vol period)
vol_peak_idx = rolling_vol.idxmax()
cluster_start = vol_peak_idx - pd.Timedelta(days=60)
cluster_end = vol_peak_idx + pd.Timedelta(days=60)

ax.axvspan(cluster_start, cluster_end, alpha=0.08, color=CRIMSON)
ax.annotate('Cluster', xy=(vol_peak_idx, rolling_vol.loc[vol_peak_idx]),
            xytext=(vol_peak_idx + pd.Timedelta(days=90),
                    rolling_vol.loc[vol_peak_idx] * 0.85),
            fontsize=7, color=CRIMSON,
            arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.7))

ax.set_xlabel('Date')
ax.set_ylabel('Annualized Volatility')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=7)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
save_fig('sfm_ch1_rolling_vol')

# =============================================================================
# 3. EWMA vs Historical Volatility
# =============================================================================
print("\n3. CREATING EWMA vs HISTORICAL VOLATILITY CHART")
print("-" * 40)

lam = 0.94

# Historical (rolling 30-day)
hist_vol = log_ret.rolling(window=30).std() * np.sqrt(252)

# EWMA variance
ewma_var = pd.Series(index=log_ret.index, dtype=float)
ewma_var.iloc[0] = log_ret.iloc[0] ** 2
for i in range(1, len(log_ret)):
    ewma_var.iloc[i] = lam * ewma_var.iloc[i - 1] + (1 - lam) * log_ret.iloc[i] ** 2
ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)

# Use a 2-year subset for clarity
sub = slice('2020-01-01', '2021-12-31')

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(hist_vol.loc[sub].index, hist_vol.loc[sub].values,
        color=MAIN_BLUE, linewidth=0.8, label='Historical (30-day)')
ax.plot(ewma_vol.loc[sub].index, ewma_vol.loc[sub].values,
        color=CRIMSON, linewidth=0.8, linestyle='--', label=f'EWMA ($\\lambda={lam}$)')

ax.set_xlabel('Date')
ax.set_ylabel('Annualized Volatility')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
save_fig('sfm_ch1_ewma')

# =============================================================================
# 4. Volatility Estimator Efficiency Comparison
# =============================================================================
print("\n4. CREATING EFFICIENCY COMPARISON CHART")
print("-" * 40)

estimators = ['Yang-Zhang', 'Rogers-Satchell', 'Garman-Klass', 'Parkinson', 'Close-to-Close']
re_values = [14.0, 8.0, 7.4, 5.2, 1.0]
colors = [CRIMSON, PURPLE, FOREST, AMBER, MAIN_BLUE]

fig, ax = plt.subplots(figsize=(7, 3))

bars = ax.barh(estimators, re_values, height=0.55, color=colors, alpha=0.75,
               edgecolor=[c for c in colors], linewidth=0.5)

# Add value labels
for bar, val in zip(bars, re_values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}$\\times$', fontsize=7, va='center', ha='left')

ax.set_xlabel('Relative Efficiency (vs CC)')
ax.set_xlim(0, 17)
ax.invert_yaxis()

plt.tight_layout()
save_fig('sfm_ch1_efficiency')

# =============================================================================
# 5. Efficient Frontier
# =============================================================================
print("\n5. CREATING EFFICIENT FRONTIER CHART")
print("-" * 40)

# Download 5 assets
tickers = ['AAPL', 'MSFT', 'AMZN', 'JNJ', 'XOM']
prices = yf.download(tickers, start='2018-01-01', end='2024-12-31', progress=False)['Close']
rets = np.log(prices / prices.shift(1)).dropna()

mu = rets.mean().values * 252  # annualized
cov = rets.cov().values * 252
n_assets = len(tickers)

def portfolio_stats(w, mu, cov):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return ret, vol

# Generate random portfolios
np.random.seed(42)
n_ports = 8000
all_ret, all_vol = [], []
for _ in range(n_ports):
    w = np.random.dirichlet(np.ones(n_assets))
    r, v = portfolio_stats(w, mu, cov)
    all_ret.append(r)
    all_vol.append(v)
all_ret = np.array(all_ret)
all_vol = np.array(all_vol)

# Minimum variance portfolio
def min_vol(mu_target):
    n = len(mu)
    def objective(w):
        return w @ cov @ w
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w @ mu - mu_target}
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = optimize.minimize(objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
    return res.x

# Trace efficient frontier
target_rets = np.linspace(mu.min(), mu.max(), 50)
frontier_vol = []
frontier_ret = []
for tr in target_rets:
    try:
        w_opt = min_vol(tr)
        _, v = portfolio_stats(w_opt, mu, cov)
        frontier_vol.append(v)
        frontier_ret.append(tr)
    except Exception:
        pass

frontier_vol = np.array(frontier_vol)
frontier_ret = np.array(frontier_ret)

# MVP
mvp_idx = np.argmin(frontier_vol)
mvp_vol = frontier_vol[mvp_idx]
mvp_ret = frontier_ret[mvp_idx]

# Risk-free rate
r_f = 0.04  # 4%

# Tangency portfolio (max Sharpe)
def neg_sharpe(w):
    r, v = portfolio_stats(w, mu, cov)
    return -(r - r_f) / v

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = [(0, 1)] * n_assets
w0 = np.ones(n_assets) / n_assets
res = optimize.minimize(neg_sharpe, w0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
w_tan = res.x
tan_ret, tan_vol = portfolio_stats(w_tan, mu, cov)

fig, ax = plt.subplots(figsize=(7, 3))

# Random portfolios as background cloud
ax.scatter(all_vol, all_ret, s=1, alpha=0.15, color='lightgray', zorder=1)

# Efficient frontier (upper portion only)
eff_mask = frontier_ret >= mvp_ret
ineff_mask = frontier_ret < mvp_ret
ax.plot(frontier_vol[eff_mask], frontier_ret[eff_mask], color=MAIN_BLUE,
        linewidth=1.5, zorder=3, label='Efficient frontier')
ax.plot(frontier_vol[ineff_mask], frontier_ret[ineff_mask], color=MAIN_BLUE,
        linewidth=0.8, linestyle='--', alpha=0.5, zorder=3)

# MVP
ax.scatter([mvp_vol], [mvp_ret], s=40, color=AMBER, zorder=4, marker='o')
ax.annotate('MVP', xy=(mvp_vol, mvp_ret), fontsize=7, color=AMBER,
            xytext=(-20, -12), textcoords='offset points')

# Tangency portfolio
ax.scatter([tan_vol], [tan_ret], s=40, color=FOREST, zorder=4, marker='o')
ax.annotate('Tangency', xy=(tan_vol, tan_ret), fontsize=7, color=FOREST,
            xytext=(5, 5), textcoords='offset points')

# CML
cml_x = np.linspace(0, frontier_vol.max() * 1.1, 100)
cml_slope = (tan_ret - r_f) / tan_vol
cml_y = r_f + cml_slope * cml_x
ax.plot(cml_x, cml_y, color=CRIMSON, linewidth=0.9, label='CML', zorder=2)

# Risk-free
ax.scatter([0], [r_f], s=30, color=CRIMSON, zorder=4, marker='o')
ax.annotate('$R_f$', xy=(0, r_f), fontsize=7, color=CRIMSON,
            xytext=(5, -8), textcoords='offset points')

# Individual assets
for i, tick in enumerate(tickers):
    asset_vol = np.sqrt(cov[i, i])
    asset_ret = mu[i]
    ax.scatter([asset_vol], [asset_ret], s=15, color=DARK_GRAY, zorder=4,
               marker='s')
    ax.annotate(tick, xy=(asset_vol, asset_ret), fontsize=6, color=DARK_GRAY,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('$\\sigma_P$ (Volatility)')
ax.set_ylabel('$E[R_P]$ (Expected Return)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_xlim(0, None)

plt.tight_layout()
save_fig('sfm_ch1_frontier')

print("\n" + "=" * 70)
print("VOLATILITY CHARTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {CHART_DIR}")
print("Output files:")
print("  - sfm_ch1_rolling_vol.pdf/.png")
print("  - sfm_ch1_ewma.pdf/.png")
print("  - sfm_ch1_efficiency.pdf/.png")
print("  - sfm_ch1_frontier.pdf/.png")
