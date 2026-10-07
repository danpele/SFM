"""
generate_all_charts.py -- charts and numbers of Chapter 1 (SFM): Data, returns and indicators
=============================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * data quality   -- EUR/RON from EODHD vs the BNR reference rate; TLV close vs adjusted close; BET vs BET-TR;
  * prices/returns -- S&P 500 price and daily log returns, ADF test on log prices and on returns;
  * simple vs log  -- r = ln(1 + R), the largest daily moves of Bitcoin and the S&P 500;
  * multi-period   -- growth of 100, cumulative simple and log returns, a two-stock portfolio (TLV, SNP);
  * descriptive    -- mean, standard deviation, skewness, excess kurtosis, extremes, histograms vs the Normal;
  * performance    -- CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, volatility drag;
  * rolling 1-year volatility, drawdown (underwater) curves, Sharpe ratios with 95% intervals.
Output: charts/sfm_ch1_*.pdf/.png, Quantlets/Ch_01/ch1_numbers.json, ch1_descriptive.csv, ch1_performance.csv
Based on the SFE Quantlets SFEReturns, SFEcomplogreturns, SFEportlogreturns, SFEDAXlogreturns, SFEfdescStat and
SFEtimeret (Franke, Haerdle and Hafner, Statistics of Financial Markets, 5th ed., 2019), ported to Python.
Run:  python3 Quantlets/Ch_01/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import LABELS, load_close, log_returns, periods_per_year, read_market   # noqa: E402
import sfm_style as st                                                                 # noqa: E402

TABLE_DIR = HERE
START = '2015-01-01'                                   # common comparison period of the chapter
ASSETS = ['sp500', 'dax', 'bet', 'bettr', 'btc', 'tlv', 'snp', 'brd', 'snn', 'tgn']
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'bettr': '#E67E22', 'btc': '#B5853F',
          'tlv': '#2E7D32', 'snp': '#8E44AD', 'brd': '#DC3545', 'snn': '#1A3A6E', 'tgn': '#17A2B8',
          'eurron': '#2E7D32', 'eurron_eodhd': '#CD0000'}
# one distinct colour and marker per asset in the cross-asset scatter charts (indices o, crypto *, BVB stocks s)
SCATTER = {'S&P 500': ('#1A3A6E', 'o'), 'DAX': ('#17A2B8', 'o'), 'BET': ('#CD0000', 'o'), 'BET-TR': ('#E67E22', 'o'),
           'Bitcoin': ('#B5853F', '*'), 'Banca Transilvania': ('#2E7D32', 's'), 'OMV Petrom': ('#8E44AD', 's'),
           'BRD': ('#DC3545', 's'), 'Nuclearelectrica': ('#6D4C41', 's'), 'Transgaz': ('#C2185B', 's')}


# =============================================================================
# DEFINITIONS (one function per indicator; the same code runs in the notebooks)
# =============================================================================
def simple_ret(p):
    """Simple (net) returns R_t = P_t / P_{t-1} - 1, as decimals."""
    return p.pct_change().dropna()


def log_ret(p):
    """Log returns r_t = ln(P_t / P_{t-1}), as decimals."""
    return np.log(p).diff().dropna()


def cagr(p):
    """Compound annual growth rate from the first and last price, over calendar years."""
    years = (p.index[-1] - p.index[0]).days / 365.25
    return (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1


def drawdown(p):
    """Drawdown DD_t = P_t / max_{s<=t} P_s - 1 (zero at a new maximum, negative below it)."""
    return p / p.cummax() - 1


def max_drawdown(p):
    """Maximum drawdown: depth, date of the previous peak, date of the trough, date of recovery (or None)."""
    dd = drawdown(p)
    trough = dd.idxmin()
    peak = p.loc[:trough].idxmax()
    after = p.loc[trough:]
    rec = after[after >= p.loc[peak]]
    return {'mdd': float(dd.min()), 'peak': peak.date().isoformat(), 'trough': trough.date().isoformat(),
            'recovery': rec.index[0].date().isoformat() if len(rec) else None}


def downside_deviation(R, target=0.0):
    """Downside deviation below a target: sqrt(mean(min(R - target, 0)^2)), over all observations."""
    return float(np.sqrt(np.mean(np.minimum(R - target, 0.0) ** 2)))


def sharpe_se(sr, n):
    """Standard error of a Sharpe ratio estimated from n i.i.d. returns (Lo, 2002): sqrt((1 + SR^2/2) / n)."""
    return float(np.sqrt((1 + 0.5 * sr ** 2) / n))


def performance(p, rf=0.0):
    """Performance indicators of one price series on its own calendar (annualised with its actual frequency)."""
    R, r = simple_ret(p), log_ret(p)
    q = periods_per_year(R)
    mu_a = R.mean() * q                                 # arithmetic annual mean of simple returns
    vol = R.std() * np.sqrt(q)                          # annualised volatility
    sr_d = (R.mean() - rf / q) / R.std()                # per-period Sharpe ratio
    sr = sr_d * np.sqrt(q)
    se = sharpe_se(sr_d, len(R)) * np.sqrt(q)
    sortino = (mu_a - rf) / (downside_deviation(R) * np.sqrt(q))
    m = max_drawdown(p)
    g = cagr(p)
    return {'n': len(R), 'obs_per_year': q, 'years': (p.index[-1] - p.index[0]).days / 365.25,
            'mean_arith': mu_a, 'mean_log': r.mean() * q, 'cagr': g, 'vol': vol,
            'drag': mu_a - r.mean() * q, 'half_var': 0.5 * vol ** 2,
            'sharpe': sr, 'sharpe_se': se, 'sharpe_lo': sr - 1.96 * se, 'sharpe_hi': sr + 1.96 * se,
            'sortino': sortino, 'mdd': m['mdd'], 'mdd_peak': m['peak'], 'mdd_trough': m['trough'],
            'mdd_recovery': m['recovery'], 'calmar': g / abs(m['mdd'])}


def describe(r):
    """Descriptive statistics of daily log returns (in %): moments, extremes and the precision of the mean."""
    q = periods_per_year(r)
    return {'n': len(r), 'obs_per_year': q, 'mean': r.mean(), 'sd': r.std(), 'skew': stats.skew(r),
            'exkurt': stats.kurtosis(r), 'min': r.min(), 'min_date': r.idxmin().date().isoformat(),
            'max': r.max(), 'max_date': r.idxmax().date().isoformat(), 'q01': r.quantile(0.01),
            'q99': r.quantile(0.99), 'ann_mean': r.mean() * q, 'ann_vol': r.std() * np.sqrt(q),
            'se_ann_mean': r.std() * np.sqrt(q) / np.sqrt(len(r) / q)}


# =============================================================================
# TABLES
# =============================================================================
def descriptive_table(names=ASSETS, start=START):
    rows = {LABELS[k]: describe(log_returns(k, start)) for k in names}
    return pd.DataFrame(rows).T


def performance_table(names=ASSETS, start=START):
    rows = {LABELS[k]: performance(load_close(k, start)) for k in names}
    return pd.DataFrame(rows).T


# =============================================================================
# CHARTS
# =============================================================================
def fig_eurron(save=True):
    """EUR/RON: daily log returns of the EODHD series and of the BNR reference rate, on common days."""
    x = pd.concat([load_close('eurron'), load_close('eurron_eodhd')], axis=1).dropna()
    r = 100 * np.log(x).diff().dropna()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(r.index, r['eurron_eodhd'], color=COLORS['eurron_eodhd'], lw=0.7, label='EUR/RON series from EODHD')
    ax.plot(r.index, r['eurron'], color=COLORS['eurron'], lw=0.7, label='EUR/RON, BNR reference rate')
    ax.set_ylabel('Daily log return (%)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_eurron_check')
    d = (r['eurron_eodhd'] - r['eurron']).abs()
    q = periods_per_year(r)
    return {'n': len(r), 'days_diff_1pct': int((d > 1).sum()), 'vol_bnr': r['eurron'].std() * np.sqrt(q),
            'vol_eodhd': r['eurron_eodhd'].std() * np.sqrt(q), 'max_eodhd': r['eurron_eodhd'].abs().max(),
            'max_eodhd_date': r['eurron_eodhd'].abs().idxmax().date().isoformat(),
            'max_bnr': r['eurron'].abs().max(), 'max_bnr_date': r['eurron'].abs().idxmax().date().isoformat()}


def fig_tlv_adjusted(start=START, save=True):
    """Banca Transilvania: close vs adjusted close (log scale); the close jumps at corporate actions."""
    t = read_market('TLV.RO').loc[start:]
    t = t[t.index.dayofweek < 5]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t.index, t['close'], color=COLORS['bet'], lw=1.1, label='Close (as traded)')
    ax.plot(t.index, t['adjusted_close'], color=COLORS['tlv'], lw=1.1, label='Adjusted close')
    ax.set_yscale('log')
    ax.set_ylabel('Price, RON (log scale)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_tlv_adjusted')
    rc, ra = np.log(t['close']).diff(), np.log(t['adjusted_close']).diff()
    d = (rc - ra).dropna()
    big = d.abs().idxmax()
    return {'jump_date': big.date().isoformat(), 'jump_close': 100 * rc.loc[big], 'jump_adj': 100 * ra.loc[big],
            'close_before': float(t['close'].shift(1).loc[big]), 'close_after': float(t['close'].loc[big]),
            'n_events_2pct': int((d.abs() > 0.02).sum())}


def fig_bet_bettr(save=True):
    """BET (price index) vs BET-TR (total return index, dividends reinvested), growth of 100."""
    x = pd.concat([load_close('bet', '2014-09-23'), load_close('bettr', '2014-09-23')], axis=1).dropna()
    idx = 100 * x / x.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(idx.index, idx['bet'], color=COLORS['bet'], lw=1.3, label='BET (price index)')
    ax.plot(idx.index, idx['bettr'], color=COLORS['bettr'], lw=1.3, label='BET-TR (dividends reinvested)')
    ax.set_ylabel('Value of 100 invested')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_bet_bettr')
    years = (x.index[-1] - x.index[0]).days / 365.25
    g = (x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1
    return {'start': x.index[0].date().isoformat(), 'end_bet': float(idx['bet'].iloc[-1]),
            'end_bettr': float(idx['bettr'].iloc[-1]), 'cagr_bet': float(g['bet']), 'cagr_bettr': float(g['bettr'])}


def fig_price_returns(save=True):
    """S&P 500 since 2000: price level (top) and daily log returns (bottom); ADF tests on both."""
    from statsmodels.tsa.stattools import adfuller
    p = load_close('sp500')
    r = 100 * log_ret(p)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax[0].plot(p.index, p, color=COLORS['sp500'], lw=1.0, label='S&P 500 price')
    ax[0].set_ylabel('Index points')
    ax[1].plot(r.index, r, color=COLORS['bet'], lw=0.5, label='S&P 500 daily log return (%)')
    ax[1].set_ylabel('Log return (%)')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_price_returns')
    a_p = adfuller(np.log(p), regression='ct', autolag='AIC')
    a_r = adfuller(r, regression='c', autolag='AIC')
    return {'adf_price': a_p[0], 'p_price': a_p[1], 'adf_ret': a_r[0], 'p_ret': a_r[1],
            'crit5_ct': a_p[4]['5%'], 'crit5_c': a_r[4]['5%'], 'n': len(r),
            'first': p.index[0].date().isoformat()}


def fig_simple_log(save=True):
    """r = ln(1 + R) against R, with the largest daily moves of Bitcoin and the S&P 500 marked."""
    R = np.linspace(-0.6, 0.6, 400)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(100 * R, 100 * R, color='black', lw=0.9, ls='--', label='r = R (45-degree line)')
    ax.plot(100 * R, 100 * np.log1p(R), color=COLORS['sp500'], lw=1.8, label='r = ln(1 + R)')
    out = {}
    for k, mk in [('btc', 'o'), ('sp500', 's')]:
        Rk = simple_ret(load_close(k))
        for lab, d in [('min', Rk.idxmin()), ('max', Rk.idxmax())]:
            x = Rk.loc[d]
            ax.scatter(100 * x, 100 * np.log1p(x), color=COLORS[k], marker=mk, s=45, zorder=3,
                       label=f'{LABELS[k]}, largest daily moves' if lab == 'min' else '_')
            out[f'{k}_{lab}'] = {'date': d.date().isoformat(), 'R': 100 * x, 'r': 100 * np.log1p(x)}
    ax.set_xlabel('Simple return R (%)')
    ax.set_ylabel('Log return r (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_simple_log')
    return out


def fig_growth(names=('sp500', 'dax', 'bet', 'btc'), start=START, save=True):
    """Growth of 100 invested at the first common date (log scale): cumulative simple return of each series."""
    px = pd.concat([load_close(k, start) for k in names], axis=1).ffill().dropna()
    idx = 100 * px / px.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k in names:
        ax.plot(idx.index, idx[k], color=COLORS[k], lw=1.3, label=LABELS[k])
    ax.set_yscale('log')
    ax.set_ylabel('Value of 100 invested (log scale)')
    st.legend_outside_bottom(ax, ncol=len(names), y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_growth')
    return {k: {'end': float(idx[k].iloc[-1]), 'cum_simple': float(idx[k].iloc[-1] / 100 - 1),
                'cum_log': float(np.log(idx[k].iloc[-1] / 100))} for k in names}


def portfolio_example(a='tlv', b='snp', w=0.5, start=START):
    """Two-stock portfolio (daily rebalanced to weights w, 1-w): exact simple return vs weighted log returns."""
    px = pd.concat([load_close(a, start), load_close(b, start)], axis=1).dropna()
    R = px.pct_change().dropna()
    r = np.log(px).diff().dropna()
    Rp = w * R[a] + (1 - w) * R[b]                     # exact for simple returns
    rp_exact = np.log1p(Rp)                            # exact portfolio log return
    rp_approx = w * r[a] + (1 - w) * r[b]              # weighted sum of log returns (approximation)
    err = 1e4 * (rp_exact - rp_approx)                 # in basis points
    years = (px.index[-1] - px.index[0]).days / 365.25
    wealth = (1 + Rp).prod()
    return {'n': len(Rp), 'mean_abs_err_bp': float(err.abs().mean()), 'max_err_bp': float(err.max()),
            'max_err_date': err.idxmax().date().isoformat(), 'cagr_port': float(wealth ** (1 / years) - 1),
            'cagr_a': cagr(px[a]), 'cagr_b': cagr(px[b]), 'wealth': float(wealth),
            'sum_exact': float(rp_exact.sum()), 'sum_approx': float(rp_approx.sum()), 'start': px.index[0].date().isoformat()}


def fig_hist(names=('sp500', 'btc'), start=START, save=True):
    """Histograms of daily log returns vs the Normal density with the same mean and standard deviation."""
    fig, axes = plt.subplots(1, len(names), figsize=(10, 3.9))
    out = {}
    for ax, k in zip(axes, names):
        r = log_returns(k, start)
        lo, hi = r.quantile(0.001), r.quantile(0.999)
        x = np.linspace(lo, hi, 300)
        ax.hist(r.clip(lo, hi), bins=90, density=True, color=COLORS[k], alpha=0.55, label=f'{LABELS[k]} (histogram)')
        ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color='black', lw=1.3,
                label='Normal density, same mean and s.d.' if k == names[0] else '_')
        ax.set_xlabel('Daily log return (%)')
        ax.set_yticks([])
        within = float(np.mean(np.abs(r - r.mean()) <= r.std()))
        beyond3 = float(np.mean(np.abs(r - r.mean()) > 3 * r.std()))
        out[k] = {'within1sd': within, 'beyond3sd': beyond3, 'normal_beyond3sd': float(2 * stats.norm.sf(3)),
                  'normal_within1sd': float(1 - 2 * stats.norm.sf(1))}
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_hist_normal')
    return out


def fig_rolling_vol(names=('sp500', 'bet', 'btc'), start=START, window=250, save=True):
    """Rolling one-year annualised volatility of daily log returns (window of `window` observations)."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    for k in names:
        r = log_returns(k, start)
        q = periods_per_year(r)
        w = int(round(q)) if k == 'btc' else window
        v = r.rolling(w).std() * np.sqrt(q)
        ax.plot(v.index, v, color=COLORS[k], lw=1.2, label=LABELS[k])
        out[k] = {'min': float(v.min()), 'max': float(v.max()), 'max_date': v.idxmax().date().isoformat(),
                  'last': float(v.dropna().iloc[-1])}
    ax.set_ylabel('Annualised volatility (%)')
    st.legend_outside_bottom(ax, ncol=len(names), y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_rolling_vol_1y')
    return out


def fig_drawdowns(save=True):
    """Drawdown (underwater) curves over each series' full history: S&P 500, BET, Bitcoin."""
    series = {'sp500': load_close('sp500'), 'bet': load_close('bet', '1997-09-19'), 'btc': load_close('btc')}
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    for k, p in series.items():
        dd = 100 * drawdown(p)
        ax.plot(dd.index, dd, color=COLORS[k], lw=1.0, label=LABELS[k])
        m = max_drawdown(p)
        m['first'] = p.index[0].date().isoformat()
        m['under_share'] = float((drawdown(p) < -0.20).mean())
        out[k] = m
    ax.set_ylabel('Drawdown from the previous peak (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_drawdowns')
    return out


def fig_vol_drag(perf, save=True):
    """Volatility drag: arithmetic minus log annual mean, against half the annual variance (one point per asset)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = 100 * perf['half_var'].astype(float)
    y = 100 * perf['drag'].astype(float)
    lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', label=r'drag $= \sigma^2/2$')
    for lab, xv, yv in zip(perf.index, x, y):
        c, mk = SCATTER[lab]
        ax.scatter(xv, yv, s=70 if mk == '*' else 45, color=c, marker=mk, zorder=3, label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Half the annual variance, $\sigma^2/2$ (% per year, log scale)')
    ax.set_ylabel('Arithmetic minus log mean (% per year)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_vol_drag')


# direct labels of the risk-return scatter (offsets in points, so that nearby assets do not overlap)
RR_SHORT = {'Banca Transilvania': 'TLV', 'OMV Petrom': 'SNP', 'Nuclearelectrica': 'SNN', 'Transgaz': 'TGN'}
RR_LABELS = {'S&P 500': (8, -2, 'left'), 'DAX': (8, -3, 'left'), 'BET': (-8, 0, 'right'), 'BET-TR': (-8, 0, 'right'),
             'Bitcoin': (-9, 0, 'right'), 'Banca Transilvania': (8, 0, 'left'), 'OMV Petrom': (-8, -3, 'right'),
             'BRD': (8, -4, 'left'), 'Nuclearelectrica': (8, 0, 'left'), 'Transgaz': (-8, 3, 'right')}


def fig_risk_return(perf, save=True):
    """Annualised volatility vs CAGR, one point per asset, 2015-2026."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for lab in perf.index:
        c, mk = SCATTER[lab]
        xv, yv = 100 * perf.loc[lab, 'vol'], 100 * perf.loc[lab, 'cagr']
        ax.scatter(xv, yv, s=80 if mk == '*' else 50, color=c, marker=mk, zorder=3, label=lab)
        dx, dy, ha = RR_LABELS.get(lab, (8, 0, 'left'))
        ax.annotate(RR_SHORT.get(lab, lab), (xv, yv), xytext=(dx, dy), textcoords='offset points', fontsize=10.5,
                    color=st.DarkText, ha=ha, va='center')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_xlim(left=8)
    ax.set_xlabel('Annualised volatility (%)')
    ax.set_ylabel('CAGR (%)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_risk_return')


def fig_sharpe_ci(perf, save=True):
    """Sharpe ratios (risk-free rate 0) with 95% intervals from the Lo (2002) standard error."""
    p = perf.sort_values('sharpe')
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(p))
    sr = p['sharpe'].astype(float)
    ax.errorbar(sr, y, xerr=1.96 * p['sharpe_se'].astype(float), fmt='o', color=COLORS['sp500'],
                ecolor=COLORS['bet'], capsize=3, label='Sharpe ratio with 95% interval')
    ax.axvline(0, color='black', lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(p.index)
    ax.set_xlabel('Annualised Sharpe ratio (risk-free rate 0)')
    st.legend_outside_bottom(ax, ncol=1, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch1_sharpe_ci')


def ohlc_last(symbol='TLV.RO', n=5):
    """The last n trading days of OHLCV data of one symbol (as traded, plus the adjusted close)."""
    t = read_market(symbol)
    t = t[(t.index.dayofweek < 5) & (t['volume'] > 0)].tail(n)
    return {d.date().isoformat(): {c: float(t.loc[d, c]) for c in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']}
            for d in t.index}


def to_json(x):
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return x.date().isoformat()
    return x


if __name__ == '__main__':
    st.apply()
    N = {}
    desc = descriptive_table()
    desc.to_csv(os.path.join(TABLE_DIR, 'ch1_descriptive.csv'))
    perf = performance_table()
    perf.to_csv(os.path.join(TABLE_DIR, 'ch1_performance.csv'))
    print(desc.round(3))
    print(perf.round(3))
    N['desc'] = {k: desc.loc[LABELS[k]].to_dict() for k in ASSETS}
    N['perf'] = {k: perf.loc[LABELS[k]].to_dict() for k in ASSETS}
    N['eurron'] = fig_eurron()
    N['tlv'] = fig_tlv_adjusted()
    N['bettr'] = fig_bet_bettr()
    N['adf'] = fig_price_returns()
    N['simplelog'] = fig_simple_log()
    N['growth'] = fig_growth()
    N['port'] = portfolio_example()
    N['hist'] = fig_hist()
    N['rollvol'] = fig_rolling_vol()
    N['dd'] = fig_drawdowns()
    fig_vol_drag(perf)
    fig_risk_return(perf)
    fig_sharpe_ci(perf)
    N['ohlc'] = ohlc_last()
    N['start'] = START
    N['end'] = load_close('sp500').index[-1].date().isoformat()
    with open(os.path.join(TABLE_DIR, 'ch1_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1, default=str)
    print(json.dumps(to_json({k: N[k] for k in ['eurron', 'tlv', 'bettr', 'adf', 'simplelog', 'port', 'hist', 'rollvol', 'dd']}), indent=1, default=str))
