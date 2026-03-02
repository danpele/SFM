"""
SFM_ch1_sharpe_ratio
====================
Portfolio Optimization and Efficient Frontier

Description:
- Download SPY, QQQ, GLD, TLT, VNQ ETF data
- Compute individual asset statistics
- Generate 10,000 random portfolios (Monte Carlo)
- Find maximum Sharpe ratio portfolio
- Plot efficient frontier with optimal portfolio

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import optimize
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

import os
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
print("SFM CHAPTER 1: EFFICIENT FRONTIER & SHARPE RATIO")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'VNQ']
data = yf.download(tickers, start='2010-01-01', end='2024-12-31',
                    progress=False)
prices = data['Close']
prices.columns = (prices.columns.get_level_values(0)
                  if hasattr(prices.columns, 'get_level_values')
                  else prices.columns)

print(f"   Assets: {', '.join(tickers)}")
print(f"   Period: {prices.index[0].strftime('%Y-%m-%d')} to "
      f"{prices.index[-1].strftime('%Y-%m-%d')}")
print(f"   Observations: {len(prices)}")

# =============================================================================
# 2. Compute Returns and Statistics
# =============================================================================
print("\n2. INDIVIDUAL ASSET STATISTICS")
print("-" * 40)

log_returns = np.log(prices / prices.shift(1)).dropna()
rf = 0.02  # annual risk-free rate

# Annualized statistics
ann_ret = log_returns.mean() * 252
ann_vol = log_returns.std() * np.sqrt(252)
sharpe = (ann_ret - rf) / ann_vol

print(f"   {'Asset':<6} {'Ann.Return':>10} {'Ann.Vol':>10} {'Sharpe':>10}")
print("   " + "-" * 40)
for t in tickers:
    print(f"   {t:<6} {ann_ret[t]:>10.4f} {ann_vol[t]:>10.4f} "
          f"{sharpe[t]:>10.4f}")

# Correlation matrix
print(f"\n   Correlation Matrix:")
corr = log_returns.corr()
header = "   " + " " * 6 + "".join([f'{t:>8}' for t in tickers])
print(header)
for t in tickers:
    vals = "".join([f'{corr.loc[t, t2]:>8.3f}' for t2 in tickers])
    print(f"   {t:<6}{vals}")

# =============================================================================
# 3. Monte Carlo Simulation (10,000 Random Portfolios)
# =============================================================================
print("\n3. MONTE CARLO SIMULATION")
print("-" * 40)

np.random.seed(42)
n_portfolios = 10000
n_assets = len(tickers)

# Mean returns and covariance matrix (annualized)
mu = log_returns.mean().values * 252
cov = log_returns.cov().values * 252

# Store results
port_returns = np.zeros(n_portfolios)
port_volatility = np.zeros(n_portfolios)
port_sharpe = np.zeros(n_portfolios)
port_weights = np.zeros((n_portfolios, n_assets))

for i in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n_assets))
    port_weights[i] = w
    port_returns[i] = np.dot(w, mu)
    port_volatility[i] = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    port_sharpe[i] = (port_returns[i] - rf) / port_volatility[i]

print(f"   Generated {n_portfolios} random portfolios")

# Maximum Sharpe ratio portfolio (from simulation)
idx_max_sharpe = np.argmax(port_sharpe)
max_sharpe_ret = port_returns[idx_max_sharpe]
max_sharpe_vol = port_volatility[idx_max_sharpe]
max_sharpe_w = port_weights[idx_max_sharpe]

print(f"\n   Max Sharpe Portfolio (Monte Carlo):")
print(f"     Return:     {max_sharpe_ret:.4f}")
print(f"     Volatility: {max_sharpe_vol:.4f}")
print(f"     Sharpe:     {port_sharpe[idx_max_sharpe]:.4f}")
print(f"     Weights:")
for t, w in zip(tickers, max_sharpe_w):
    print(f"       {t}: {w:.4f}")

# Minimum variance portfolio (from simulation)
idx_min_vol = np.argmin(port_volatility)
min_vol_ret = port_returns[idx_min_vol]
min_vol_vol = port_volatility[idx_min_vol]

# =============================================================================
# 4. Analytical Optimization
# =============================================================================
print("\n4. ANALYTICAL OPTIMIZATION")
print("-" * 40)

def neg_sharpe(weights):
    """Negative Sharpe ratio for minimization."""
    p_ret = np.dot(weights, mu)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -(p_ret - rf) / p_vol

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(n_assets))
w0 = np.ones(n_assets) / n_assets

opt_result = optimize.minimize(neg_sharpe, w0, method='SLSQP',
                                bounds=bounds, constraints=constraints)
opt_w = opt_result.x
opt_ret = np.dot(opt_w, mu)
opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(cov, opt_w)))
opt_sharpe = (opt_ret - rf) / opt_vol

print(f"   Optimal Portfolio (Analytical):")
print(f"     Return:     {opt_ret:.4f}")
print(f"     Volatility: {opt_vol:.4f}")
print(f"     Sharpe:     {opt_sharpe:.4f}")
print(f"     Weights:")
for t, w in zip(tickers, opt_w):
    print(f"       {t}: {w:.4f}")

# =============================================================================
# 5. Efficient Frontier (analytical)
# =============================================================================
print("\n5. COMPUTING EFFICIENT FRONTIER")
print("-" * 40)

target_returns = np.linspace(port_returns.min(), port_returns.max(), 100)
ef_volatility = []

for target in target_returns:
    constraints_ef = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
    ]

    def min_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    result = optimize.minimize(min_vol, w0, method='SLSQP',
                                bounds=bounds, constraints=constraints_ef)
    if result.success:
        ef_volatility.append(result.fun)
    else:
        ef_volatility.append(np.nan)

ef_volatility = np.array(ef_volatility)
print(f"   Computed efficient frontier "
      f"({np.sum(~np.isnan(ef_volatility))} valid points)")

# =============================================================================
# 6. FIGURE: Efficient Frontier
# =============================================================================
print("\n6. CREATING FIGURE")
print("-" * 40)

fig, ax = plt.subplots(figsize=(12, 5))

# Random portfolios colored by Sharpe
scatter = ax.scatter(port_volatility * 100, port_returns * 100,
                     c=port_sharpe, cmap='viridis', s=3, alpha=0.4)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Sharpe Ratio', fontsize=9)

# Efficient frontier
valid = ~np.isnan(ef_volatility)
ax.plot(ef_volatility[valid] * 100, target_returns[valid] * 100,
        color='#DC3545', linewidth=2.5, label='Efficient Frontier')

# Optimal portfolio
ax.scatter(opt_vol * 100, opt_ret * 100, marker='*', s=300,
           color='#DC3545', edgecolors='black', linewidth=0.8, zorder=5,
           label=f'Max Sharpe (SR={opt_sharpe:.2f})')

# Minimum variance portfolio
ax.scatter(min_vol_vol * 100, min_vol_ret * 100, marker='D', s=100,
           color='#2E7D32', edgecolors='black', linewidth=0.8, zorder=5,
           label='Min Variance')

# Individual assets
asset_colors = ['#1A3A6E', '#DC3545', '#2E7D32', '#FF8C00', '#6A0DAD']
for i, t in enumerate(tickers):
    ax.scatter(ann_vol[t] * 100, ann_ret[t] * 100, marker='o', s=80,
               color=asset_colors[i], edgecolors='black', linewidth=0.8,
               zorder=5)
    ax.annotate(t, (ann_vol[t] * 100, ann_ret[t] * 100),
                textcoords="offset points", xytext=(8, 5), fontsize=9,
                fontweight='bold')

# Capital Market Line
cml_x = np.linspace(0, max(port_volatility) * 100 * 1.1, 100)
cml_y = rf * 100 + opt_sharpe * cml_x
ax.plot(cml_x, cml_y, color='gray', linestyle='--', linewidth=1,
        alpha=0.7, label='Capital Market Line')

ax.set_xlabel('Annualized Volatility (%)')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Ch.1: Efficient Frontier and Optimal Portfolio', fontweight='bold')
ax.set_xlim(0, max(port_volatility) * 100 * 1.15)
ax.set_ylim(min(port_returns) * 100 * 0.9,
            max(port_returns) * 100 * 1.15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch1_efficient_frontier')

print("\n" + "=" * 70)
print("EFFICIENT FRONTIER ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch1_efficient_frontier.pdf: Efficient frontier with optimal portfolio")
