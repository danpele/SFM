"""
SFM_ch12_crypto
===============
Cryptocurrency Markets: Bitcoin Analysis

Description:
- Download BTC-USD and S&P 500 data
- Compare return distributions and volatility
- Stablecoin peg stability analysis (USDT, USDC)
- Rolling volatility comparison
- Correlation dynamics between BTC and traditional assets

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Standard chart style ---
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
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0

def save_fig(name):
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("SFM CHAPTER 12: CRYPTOCURRENCY MARKETS")
print("=" * 70)

# =============================================================================
# 1. Download Data
# =============================================================================
print("\n1. DOWNLOADING DATA")
print("-" * 40)

tickers = {'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
           '^GSPC': 'S&P 500', 'GLD': 'Gold'}

prices = {}
returns = {}
for tick, name in tickers.items():
    d = yf.download(tick, start='2018-01-01', end='2024-12-31',
                     progress=False)
    p = d['Close'].squeeze()
    prices[tick] = p
    r = np.log(p / p.shift(1)).dropna()
    returns[tick] = r
    print(f"   {name:12s} ({tick}): {len(r)} observations")

# Download stablecoin
stable_tickers = ['USDT-USD', 'USDC-USD']
stable_prices = {}
for st in stable_tickers:
    d = yf.download(st, start='2018-01-01', end='2024-12-31',
                     progress=False)
    stable_prices[st] = d['Close'].squeeze()
    print(f"   {st:12s}: {len(stable_prices[st])} observations")

# =============================================================================
# 2. Descriptive Statistics
# =============================================================================
print("\n2. DESCRIPTIVE STATISTICS (annualized)")
print("-" * 40)

print(f"   {'Asset':<12} {'Return':>10} {'Vol':>10} {'Skew':>8} "
      f"{'Kurt':>8} {'Sharpe':>8}")
print("   " + "-" * 58)

for tick, name in tickers.items():
    r = returns[tick]
    trading_days = 365 if 'USD' in tick else 252
    ann_ret = r.mean() * trading_days
    ann_vol = r.std() * np.sqrt(trading_days)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r)
    sharpe = (ann_ret - 0.02) / ann_vol
    print(f"   {name:<12} {ann_ret:>10.4f} {ann_vol:>10.4f} "
          f"{skew:>8.2f} {kurt:>8.2f} {sharpe:>8.4f}")

# =============================================================================
# 3. Stablecoin Analysis
# =============================================================================
print("\n3. STABLECOIN PEG STABILITY")
print("-" * 40)

for st in stable_tickers:
    if st in stable_prices:
        p = stable_prices[st]
        dev = p - 1.0
        print(f"   {st}:")
        print(f"     Mean deviation from $1: {dev.mean():.6f}")
        print(f"     Std deviation:          {dev.std():.6f}")
        print(f"     Max deviation:          {dev.max():.6f}")
        print(f"     Min deviation:          {dev.min():.6f}")
        print(f"     % within +/-0.01:       "
              f"{(np.abs(dev) < 0.01).mean() * 100:.1f}%")

# =============================================================================
# 4. FIGURE: Crypto Analysis (6-panel)
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Common date range for BTC vs S&P comparison
common_idx = returns['BTC-USD'].index.intersection(
    returns['^GSPC'].index)
btc_r = returns['BTC-USD'].reindex(common_idx).dropna()
sp_r = returns['^GSPC'].reindex(common_idx).dropna()
common = btc_r.index.intersection(sp_r.index)
btc_r = btc_r.loc[common]
sp_r = sp_r.loc[common]

# Panel A: Cumulative returns
colors_p = ['#E67E22', '#8E44AD', '#1A3A6E', '#2E7D32']
for (tick, name), color in zip(tickers.items(), colors_p):
    p = prices[tick]
    cum = p / p.iloc[0]
    axes[0, 0].plot(cum.index, cum, color=color, linewidth=0.8,
                    label=name)
axes[0, 0].set_title('Cumulative Returns (Normalized)',
                      fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Growth of $1')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=4, frameon=False)

# Panel B: Rolling 30-day volatility
window = 30
for (tick, name), color in zip(tickers.items(), colors_p):
    r = returns[tick]
    trading_days = 365 if 'USD' in tick else 252
    vol = r.rolling(window).std() * np.sqrt(trading_days) * 100
    axes[0, 1].plot(vol.index, vol, color=color, linewidth=0.6,
                    alpha=0.8, label=name)
axes[0, 1].set_title(f'Rolling {window}-Day Volatility (%)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Annualized Volatility (%)')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=4, frameon=False)

# Panel C: Return distribution comparison (BTC vs S&P 500)
axes[1, 0].hist(btc_r * 100, bins=100, density=True, alpha=0.5,
                color='#E67E22', edgecolor='white', label='Bitcoin')
axes[1, 0].hist(sp_r * 100, bins=100, density=True, alpha=0.5,
                color='#1A3A6E', edgecolor='white', label='S&P 500')
axes[1, 0].set_title('Return Distribution: BTC vs S&P 500',
                      fontweight='bold')
axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_xlim(-15, 15)
axes[1, 0].text(0.95, 0.95,
               f'BTC Kurt: {stats.kurtosis(btc_r):.1f}\n'
               f'SPX Kurt: {stats.kurtosis(sp_r):.1f}',
               transform=axes[1, 0].transAxes, ha='right', va='top',
               fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8))
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)

# Panel D: Rolling correlation BTC vs S&P 500
roll_corr = btc_r.rolling(60).corr(sp_r)
axes[1, 1].plot(roll_corr.index, roll_corr, color='#1A3A6E',
                linewidth=0.8)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 1].fill_between(roll_corr.index, 0, roll_corr, alpha=0.15,
                         color='#1A3A6E')
axes[1, 1].set_title('Rolling 60-Day Correlation: BTC vs S&P 500',
                      fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Correlation')
avg_corr = roll_corr.mean()
axes[1, 1].text(0.05, 0.95, f'Mean: {avg_corr:.3f}',
               transform=axes[1, 1].transAxes, ha='left', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8))

# Panel E: Stablecoin peg
stable_colors = ['#26A17B', '#2775CA']
for st, c in zip(stable_tickers, stable_colors):
    if st in stable_prices:
        axes[2, 0].plot(stable_prices[st].index, stable_prices[st],
                        color=c, linewidth=0.5,
                        label=st.replace('-USD', ''))
axes[2, 0].axhline(y=1.0, color='#DC3545', linestyle='--',
                    linewidth=0.8, label='$1.00 peg')
axes[2, 0].fill_between(stable_prices[stable_tickers[0]].index,
                          0.99, 1.01, alpha=0.1, color='gray',
                          label='+/- 1% band')
axes[2, 0].set_title('Stablecoin Peg Stability', fontweight='bold')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Price (USD)')
axes[2, 0].set_ylim(0.95, 1.05)
axes[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

# Panel F: QQ-plot of Bitcoin returns
stats.probplot(btc_r, dist="norm", plot=axes[2, 1])
axes[2, 1].set_title('Bitcoin Returns: QQ-Plot vs Normal',
                      fontweight='bold')
axes[2, 1].get_lines()[0].set_markerfacecolor('#E67E22')
axes[2, 1].get_lines()[0].set_markersize(2)
axes[2, 1].get_lines()[0].set_alpha(0.4)
axes[2, 1].get_lines()[1].set_color('#DC3545')

plt.tight_layout()
save_fig('ch12_crypto')

print("\n" + "=" * 70)
print("CRYPTOCURRENCY ANALYSIS COMPLETE")
print("=" * 70)
print("\nKey findings:")
print("  - Crypto returns exhibit much higher volatility than equities")
print("  - Heavy tails are more pronounced in crypto markets")
print("  - BTC-SPX correlation varies significantly over time")
print("  - Stablecoins generally maintain their peg")
print("\nOutput files:")
print("  - ch12_crypto.pdf: 6-panel crypto analysis")
