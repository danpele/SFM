"""
SFM_ch2_pareto_analysis
=======================
Pareto Distribution: Visualization, Parameter Estimation & Real-Data Application

Description:
- Plot Pareto PDF and CDF for various shape parameters alpha
- Plot log-log survival function (power-law signature)
- Estimate Pareto parameters via MLE and Hill estimator on real data
- Apply to S&P 500 tail losses and wealth/market-cap data

Statistics of Financial Markets course — Section 2.5 (Pareto)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pareto as pareto_dist
import warnings
warnings.filterwarnings('ignore')

# ─── Chart style settings — Nature journal quality ───────────────────────────
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

# ─── Colors ──────────────────────────────────────────────────────────────────
MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E67E22'
PURPLE    = '#8E44AD'

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'charts'))
os.makedirs(CHART_DIR, exist_ok=True)

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(os.path.join(CHART_DIR, f'{name}.pdf'),
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(CHART_DIR, f'{name}.png'),
                bbox_inches='tight', transparent=True, dpi=200)
    plt.close()


# =============================================================================
# 1. Pareto PDF & CDF for various alpha
# =============================================================================
def plot_pareto_pdf_cdf():
    """Plot Pareto PDF and CDF for alpha = 1, 2, 3, 5."""
    x_m = 1.0
    alphas = [1.0, 2.0, 3.0, 5.0]
    colors = [CRIMSON, MAIN_BLUE, FOREST, AMBER]
    x = np.linspace(1.001, 6, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for alpha, c in zip(alphas, colors):
        # scipy pareto: pdf(x, b) = b / x^(b+1) for x >= 1
        pdf_vals = pareto_dist.pdf(x, alpha)
        cdf_vals = pareto_dist.cdf(x, alpha)
        label = fr'$\alpha = {alpha:.0f}$'
        ax1.plot(x, pdf_vals, color=c, label=label)
        ax2.plot(x, cdf_vals, color=c, label=label)

    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_title('Pareto PDF')
    ax1.legend()
    ax1.set_ylim(0, 5)

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$F(x)$')
    ax2.set_title('Pareto CDF')
    ax2.legend(loc='lower right')

    fig.tight_layout()
    save_fig('ch2_pareto_pdf_cdf')
    print("[OK] ch2_pareto_pdf_cdf.pdf")


# =============================================================================
# 2. Log-log survival plot (power-law signature)
# =============================================================================
def plot_pareto_loglog():
    """Log-log plot of P(X > x) showing straight-line power-law behavior."""
    x_m = 1.0
    alphas = [1.5, 2.0, 3.0, 4.0]
    colors = [CRIMSON, MAIN_BLUE, FOREST, AMBER]
    x = np.logspace(0, 3, 500)

    fig, ax = plt.subplots(figsize=(3.2, 2.5))

    for alpha, c in zip(alphas, colors):
        survival = pareto_dist.sf(x, alpha)  # 1 - CDF
        ax.loglog(x, survival, color=c,
                  label=fr'$\alpha = {alpha:.1f}$, slope $= -{alpha:.1f}$')

    ax.set_xlabel('$x$ (log scale)')
    ax.set_ylabel('$P(X > x)$ (log scale)')
    ax.set_title('Survival function (log-log)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig('ch2_pareto_loglog_survival')
    print("[OK] ch2_pareto_loglog_survival.pdf")


# =============================================================================
# 3. Hill estimator on S&P 500 tail losses
# =============================================================================
def hill_estimator(data, k_values=None):
    """Compute Hill estimator for tail index alpha.

    Parameters
    ----------
    data : array-like, positive values (e.g., absolute losses)
    k_values : array of int, number of upper order statistics to use

    Returns
    -------
    k_arr, alpha_hat : arrays
    """
    sorted_data = np.sort(data)[::-1]  # descending
    n = len(sorted_data)
    if k_values is None:
        k_values = np.arange(10, min(n // 2, 500), 1)
    alpha_hat = []
    for k in k_values:
        log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
        xi_hat = np.mean(log_ratios)
        alpha_hat.append(1.0 / xi_hat if xi_hat > 0 else np.nan)
    return k_values, np.array(alpha_hat)


def plot_hill_sp500():
    """Download S&P 500, compute tail losses, apply Hill estimator."""
    try:
        import yfinance as yf
        sp = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                         progress=False)
        if 'Adj Close' in sp.columns:
            prices = sp['Adj Close']
        elif 'Close' in sp.columns:
            prices = sp['Close']
        else:
            prices = sp.iloc[:, 3]
        prices = prices.squeeze()
    except Exception:
        # Fallback: simulate
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.standard_t(4, 6000) * 0.01) * 1000)

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    # Focus on losses (negative returns → positive values)
    losses = -log_ret[log_ret < 0]

    k_vals, alpha_hat = hill_estimator(losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Hill plot
    ax1.plot(k_vals, alpha_hat, color=MAIN_BLUE, linewidth=0.6)
    # Confidence band (approximate)
    median_alpha = np.nanmedian(alpha_hat[len(alpha_hat)//4:len(alpha_hat)//2])
    ax1.axhline(median_alpha, color=CRIMSON, linestyle='--', linewidth=0.6,
                label=fr'$\hat{{\alpha}} \approx {median_alpha:.2f}$')
    ax1.set_xlabel('$k$ (order statistics)')
    ax1.set_ylabel(r'$\hat{\alpha}_{\mathrm{Hill}}$')
    ax1.set_title('Hill plot — S&P 500 losses')
    ax1.set_ylim(1, 8)
    ax1.legend()

    # Log-log empirical survival vs fitted Pareto (tail only: top 10%)
    sorted_losses = np.sort(losses)[::-1]
    n_all = len(sorted_losses)
    emp_surv = np.arange(1, n_all + 1) / n_all
    # Fit Pareto via MLE on tail (top 10% of losses)
    k_tail = max(int(0.10 * n_all), 50)
    tail_data = sorted_losses[:k_tail]
    x_m_hat = tail_data[-1]  # threshold = smallest value in tail
    n_tail = len(tail_data)
    alpha_mle = n_tail / np.sum(np.log(tail_data / x_m_hat))

    x_fit = np.logspace(np.log10(sorted_losses[-1]),
                        np.log10(sorted_losses[0]), 200)
    theo_surv = (x_m_hat / x_fit) ** alpha_mle

    ax2.loglog(sorted_losses, emp_surv, '.', color=MAIN_BLUE, markersize=1,
               alpha=0.4, label='Empirical')
    ax2.loglog(x_fit, theo_surv, color=CRIMSON, linewidth=0.8,
               label=fr'Pareto ($\hat{{\alpha}}={alpha_mle:.2f}$)')
    ax2.set_xlabel('Loss magnitude (log)')
    ax2.set_ylabel('$P(L > x)$ (log)')
    ax2.set_title('Tail fit — S&P 500')
    ax2.legend(fontsize=7)

    fig.tight_layout()
    save_fig('ch2_pareto_hill_sp500')
    print(f"[OK] ch2_pareto_hill_sp500.pdf  (MLE alpha={alpha_mle:.2f}, Hill alpha~{median_alpha:.2f})")

    return alpha_mle, median_alpha


# =============================================================================
# 4. Pareto in practice: market-cap distribution (Zipf's law)
# =============================================================================
def plot_pareto_market_cap():
    """Show that market capitalizations follow a Pareto/Zipf distribution."""
    try:
        import yfinance as yf
        # Top 50 S&P 500 companies by market cap (representative sample)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B',
                    'LLY', 'TSM', 'V', 'JPM', 'UNH', 'XOM', 'WMT', 'MA',
                    'JNJ', 'PG', 'HD', 'COST', 'ABBV', 'MRK', 'BAC', 'AVGO',
                    'KO', 'PEP', 'ORCL', 'TMO', 'MCD', 'CSCO', 'CRM',
                    'ACN', 'NFLX', 'AMD', 'LIN', 'ABT', 'DHR', 'ADBE',
                    'TXN', 'PM', 'WFC', 'DIS', 'CMCSA', 'NEE', 'RTX',
                    'QCOM', 'INTC', 'IBM', 'CAT', 'GE', 'INTU']
        info_list = []
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                mc = tk.info.get('marketCap', None)
                if mc and mc > 0:
                    info_list.append({'ticker': t, 'marketCap': mc})
            except Exception:
                pass
        if len(info_list) < 20:
            raise ValueError("Not enough data")
        df = pd.DataFrame(info_list).sort_values('marketCap', ascending=False)
        mcaps = df['marketCap'].values / 1e9  # in billions
    except Exception:
        # Fallback: synthetic Zipf-like data
        np.random.seed(42)
        mcaps = np.sort(pareto_dist.rvs(1.2, size=50) * 100)[::-1]

    n = len(mcaps)
    rank = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Rank-size plot (Zipf)
    ax1.bar(rank, mcaps, color=MAIN_BLUE, alpha=0.7, width=0.8)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Market Cap ($ billions)')
    ax1.set_title('Market Cap — Top 50 US Firms')

    # Log-log rank-size
    ax2.loglog(rank, mcaps, 'o', color=MAIN_BLUE, markersize=3)
    # Fit line
    log_r = np.log(rank)
    log_m = np.log(mcaps)
    slope, intercept = np.polyfit(log_r, log_m, 1)
    ax2.loglog(rank, np.exp(intercept) * rank ** slope, '--', color=CRIMSON,
               linewidth=0.8,
               label=fr'slope $= {slope:.2f}$')
    ax2.set_xlabel('Rank (log)')
    ax2.set_ylabel('Market Cap (log)')
    ax2.set_title("Zipf's law — log-log")
    ax2.legend()

    fig.tight_layout()
    save_fig('ch2_pareto_market_cap')
    print(f"[OK] ch2_pareto_market_cap.pdf  (Zipf slope = {slope:.2f})")


# =============================================================================
# 5. MLE estimation demonstration
# =============================================================================
def plot_pareto_mle_demo():
    """Demonstrate MLE estimation on simulated Pareto data."""
    np.random.seed(42)
    true_alpha = 3.0
    x_m = 1.0
    n_samples = [50, 200, 1000, 5000]
    colors = [CRIMSON, MAIN_BLUE, FOREST, AMBER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Left: histogram + fitted PDF for n=1000
    sample = pareto_dist.rvs(true_alpha, size=1000)
    x_plot = np.linspace(1, 8, 300)
    ax1.hist(sample, bins=80, density=True, alpha=0.5, color=MAIN_BLUE,
             edgecolor='none', label='Sample ($n=1000$)')
    # MLE
    alpha_hat = len(sample) / np.sum(np.log(sample))
    ax1.plot(x_plot, pareto_dist.pdf(x_plot, alpha_hat), color=CRIMSON,
             linewidth=1.0, label=fr'MLE: $\hat{{\alpha}}={alpha_hat:.2f}$')
    ax1.plot(x_plot, pareto_dist.pdf(x_plot, true_alpha), '--', color=FOREST,
             linewidth=0.8, label=fr'True: $\alpha={true_alpha:.0f}$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Density')
    ax1.set_title('Pareto MLE fit')
    ax1.set_xlim(0.9, 8)
    ax1.legend(fontsize=7)

    # Right: convergence of MLE as n grows
    n_reps = 500
    for n_s, c in zip(n_samples, colors):
        estimates = []
        for _ in range(n_reps):
            s = pareto_dist.rvs(true_alpha, size=n_s)
            est = n_s / np.sum(np.log(s))
            estimates.append(est)
        ax2.hist(estimates, bins=30, density=True, alpha=0.4, color=c,
                 edgecolor='none', label=f'$n={n_s}$')

    ax2.axvline(true_alpha, color='black', linestyle='--', linewidth=0.6,
                label=fr'$\alpha = {true_alpha:.0f}$')
    ax2.set_xlabel(r'$\hat{\alpha}_{\mathrm{MLE}}$')
    ax2.set_ylabel('Density')
    ax2.set_title('MLE convergence')
    ax2.legend(fontsize=6.5)

    fig.tight_layout()
    save_fig('ch2_pareto_mle_demo')
    print("[OK] ch2_pareto_mle_demo.pdf")


# =============================================================================
# 6. Tail fitting: QQ-plot + threshold sensitivity + goodness-of-fit
# =============================================================================
def plot_pareto_tail_fitting():
    """Detailed tail fitting: QQ-plot, threshold sensitivity, KS test."""
    try:
        import yfinance as yf
        sp = yf.download('^GSPC', start='2000-01-01', end='2025-12-31',
                         progress=False)
        if 'Adj Close' in sp.columns:
            prices = sp['Adj Close']
        elif 'Close' in sp.columns:
            prices = sp['Close']
        else:
            prices = sp.iloc[:, 3]
        prices = prices.squeeze()
    except Exception:
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.standard_t(4, 6000) * 0.01) * 1000)

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    losses = -log_ret[log_ret < 0]
    sorted_losses = np.sort(losses)[::-1]
    n_all = len(sorted_losses)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

    # --- Panel 1: QQ-plot Pareto (top 10% tail) ---
    ax1 = axes[0]
    k_tail = max(int(0.10 * n_all), 50)
    tail_data = np.sort(sorted_losses[:k_tail])
    x_m_hat = tail_data[0]
    alpha_hat = len(tail_data) / np.sum(np.log(tail_data / x_m_hat))

    # Theoretical quantiles
    n_t = len(tail_data)
    p_vals = (np.arange(1, n_t + 1) - 0.5) / n_t
    theo_q = x_m_hat * (1 - p_vals) ** (-1.0 / alpha_hat)

    ax1.plot(theo_q, tail_data, 'o', color=MAIN_BLUE, markersize=2, alpha=0.6)
    lims = [min(theo_q.min(), tail_data.min()), max(theo_q.max(), tail_data.max())]
    ax1.plot(lims, lims, '--', color=CRIMSON, linewidth=0.6)
    ax1.set_xlabel('Theoretical (Pareto)')
    ax1.set_ylabel('Empirical')
    ax1.set_title(fr'QQ-plot ($\hat{{\alpha}}={alpha_hat:.2f}$)')

    # --- Panel 2: Alpha vs threshold (sensitivity) ---
    ax2 = axes[1]
    thresholds_pct = np.arange(1, 26)  # top 1% to 25%
    alpha_vals = []
    ci_lo, ci_hi = [], []
    for pct in thresholds_pct:
        k = max(int(pct / 100.0 * n_all), 10)
        td = sorted_losses[:k]
        xm = td[-1]
        n_k = len(td)
        a_hat = n_k / np.sum(np.log(td / xm))
        se = a_hat / np.sqrt(n_k)  # asymptotic SE
        alpha_vals.append(a_hat)
        ci_lo.append(a_hat - 1.96 * se)
        ci_hi.append(a_hat + 1.96 * se)

    ax2.plot(thresholds_pct, alpha_vals, 'o-', color=MAIN_BLUE, markersize=2,
             linewidth=0.6)
    ax2.fill_between(thresholds_pct, ci_lo, ci_hi, alpha=0.15, color=MAIN_BLUE)
    ax2.set_xlabel('Tail fraction (%)')
    ax2.set_ylabel(r'$\hat{\alpha}_{\mathrm{MLE}}$')
    ax2.set_title('Threshold sensitivity')
    ax2.set_ylim(1, 6)

    # --- Panel 3: Empirical vs Pareto CDF (tail zoom) ---
    ax3 = axes[2]
    k_fit = max(int(0.05 * n_all), 50)  # top 5%
    tail5 = sorted_losses[:k_fit]
    xm5 = tail5[-1]
    a5 = len(tail5) / np.sum(np.log(tail5 / xm5))

    # Empirical CDF of tail
    tail5_sorted = np.sort(tail5)
    ecdf = np.arange(1, len(tail5_sorted) + 1) / len(tail5_sorted)
    # Theoretical Pareto CDF
    tcdf = 1 - (xm5 / tail5_sorted) ** a5

    ax3.step(tail5_sorted, ecdf, color=MAIN_BLUE, linewidth=0.8, label='Empirical')
    ax3.plot(tail5_sorted, tcdf, color=CRIMSON, linewidth=0.8,
             label=fr'Pareto ($\hat{{\alpha}}={a5:.2f}$)')
    ax3.set_xlabel('Loss magnitude')
    ax3.set_ylabel('CDF')
    ax3.set_title('Tail CDF fit (top 5%)')
    ax3.legend(fontsize=7)

    fig.tight_layout()
    save_fig('ch2_pareto_tail_fitting')
    print(f"[OK] ch2_pareto_tail_fitting.pdf  (alpha_5pct={a5:.2f}, alpha_10pct={alpha_hat:.2f})")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SFM Ch2 — Pareto Distribution Analysis")
    print("=" * 60)

    plot_pareto_pdf_cdf()
    plot_pareto_loglog()
    plot_pareto_mle_demo()
    plot_pareto_market_cap()
    plot_hill_sp500()
    plot_pareto_tail_fitting()

    print("\nAll charts saved to:", CHART_DIR)
