"""
generate_all_charts.py -- charts and numbers of Chapter 5 (SFM): heavy tails and extreme value theory
=====================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * motivation  -- the largest daily losses of the S&P 500, DAX, BET and Bitcoin and their probability under the
                   Normal distribution; light vs heavy tails (Normal, Exponential, Student-t, Pareto);
  * tail index  -- regular variation, the log-log tail plot with the least-squares slope, the Hill estimator,
                   the Hill plot, standard errors (i.i.d. and moving-block bootstrap), the Hill (Weissman) quantile;
  * mean excess -- the empirical mean excess function of daily losses;
  * block maxima-- the three extreme value laws (Gumbel, Frechet, Weibull), maxima of simulated samples,
                   GEV fits to monthly maxima of daily losses, PP and QQ plots, return levels;
  * POT         -- GPD densities, threshold choice (parameter stability), GPD fits to the losses above the 90%
                   quantile (McNeil and Frey, 2000), PP and QQ plots, the fitted tail;
  * risk        -- VaR 1%, ES 2.5% and VaR 0.1% by EVT (POT), historical simulation and the Normal distribution;
                   an out-of-sample check (estimation until 2019, exceedances 2020-2026);
  * BVB stocks  -- Hill and POT for OMV Petrom, BRD, Banca Transilvania and Transgaz since 2010;
  * AI section  -- the Hill index of BET losses in 1997-2009 and in 2010-2026.
Output: charts/sfm_ch5_*.pdf/.png, Quantlets/Ch_05/ch5_numbers.json, ch5_tail_table.csv
Based on the Quantlets SFElshill, SFEhillquantile, SFEMeanExcessFun, SFEevt1, SFEevt2, SFEtailGEV_pp, SFEtailGEV_qq,
SFEtailGPareto_pp, SFEtailGPareto_qq, block_max and var_pot (github.com/QuantLet/SFE), and on Franke, Haerdle and
Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 18.
Run:  python3 Quantlets/Ch_05/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import LABELS, log_returns   # noqa: E402
import sfm_style as st                     # noqa: E402

TABLE_DIR = HERE
START = {'sp500': '1990-01-01', 'dax': '1990-01-01', 'bet': '1997-01-01', 'btc': None,
         'snp': '2010-01-01', 'brd': '2010-01-01', 'tlv': '2010-01-01', 'tgn': '2010-01-01'}
ASSETS = ['sp500', 'dax', 'bet', 'btc']
STOCKS = ['snp', 'brd', 'tlv', 'tgn']
# Banca Transilvania: the adjustment for the bonus shares of May 2016 is applied one day late in the adjusted close
# (log returns of -15% and +20% on two consecutive days, see Chapter 2); the two days are removed.
BAD_DAYS = {'tlv': ['2016-05-30', '2016-05-31']}
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F',
          'snp': '#2E7D32', 'brd': '#8E44AD', 'tlv': '#E67E22', 'tgn': '#DC3545'}
MODEL_COL = {'Data': '#B5853F', 'Normal': '#CD0000', 'EVT': '#1A3A6E', 'Historical': '#2E7D32', 'Hill': '#8E44AD'}
SEED = 2026
HILL_FRAC = 0.025          # k = 2.5% of n for the Hill estimator (Hill plot flat between 1% and 5%)
POT_Q = 0.90               # threshold u = 90% quantile of the losses: the 10% largest losses (McNeil and Frey, 2000)
SPLIT = '2019-12-31'       # out-of-sample check: estimation until this day, test afterwards
BLOCK = 20                 # moving-block bootstrap: blocks of 20 trading days


# =============================================================================
# DATA
# =============================================================================
def returns(k):
    """Daily log returns in % on the series' own calendar (S&P 500 and DAX since 1990, BET since its launch in
    1997, Bitcoin since 2014, BVB stocks since 2010); known data errors removed."""
    r = log_returns(k, START.get(k))
    if k in BAD_DAYS:
        r = r.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return r


def losses(k):
    """Daily losses L_t = -r_t in % (a positive number is a loss)."""
    return -returns(k)


# =============================================================================
# TAIL INDEX: HILL, LEAST SQUARES, WEISSMAN QUANTILE
# =============================================================================
def hill(x, k):
    """Hill (1975): alpha_hat(k) = [ (1/k) sum_{i=1..k} ln(X_(i) / X_(k+1)) ]^(-1), X_(1) >= X_(2) >= ..."""
    x = np.sort(np.asarray(x, float))[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def hill_quantile(x, k, p):
    """Hill (Weissman, 1978) quantile: x_p = X_(k+1) (k / (n p))^(1/alpha_hat), the loss exceeded with probability p."""
    x = np.sort(np.asarray(x, float))[::-1]
    return x[k] * (k / (len(x) * p)) ** (1 / hill(x, k))


def ls_tail_index(x, k):
    """Least-squares tail index (as in SFElshill): slope of ln(i/n) on ln X_(i), i = 1..k, is -alpha."""
    n = len(x)
    x = np.sort(np.asarray(x, float))[::-1][:k]
    return -np.polyfit(np.log(x), np.log(np.arange(1, k + 1) / n), 1)[0]


def block_indices(n, block, rng):
    """Indices of one moving-block bootstrap sample (blocks of `block` consecutive days)."""
    nb = int(np.ceil(n / block))
    s = rng.integers(0, n - block + 1, nb)
    return (s[:, None] + np.arange(block)[None, :]).ravel()[:n]


def hill_inference(x, frac=HILL_FRAC, B=500, block=BLOCK, seed=SEED):
    """Hill alpha_hat at k = frac n with the i.i.d. standard error alpha/sqrt(k), the 95% interval
    alpha (1 +/- 1.96/sqrt(k)) and the moving-block bootstrap standard error (blocks of 20 days)."""
    x = np.asarray(x, float)
    n = len(x)
    k = int(frac * n)
    a = hill(x, k)
    rng = np.random.default_rng(seed)
    bs = np.array([hill(x[block_indices(n, block, rng)], k) for _ in range(B)])
    return {'n': n, 'k': k, 'u': float(np.sort(x)[::-1][k]), 'alpha': a, 'se_iid': a / np.sqrt(k),
            'lo': a * (1 - 1.96 / np.sqrt(k)), 'hi': a * (1 + 1.96 / np.sqrt(k)), 'se_block': float(np.std(bs, ddof=1)),
            'alpha_1': hill(x, int(0.01 * n)), 'alpha_5': hill(x, int(0.05 * n)), 'ls': ls_tail_index(x, k)}


# =============================================================================
# GEV AND GPD: DENSITIES, MAXIMUM LIKELIHOOD, STANDARD ERRORS
# =============================================================================
def gev_cdf(x, xi, mu, sigma):
    """GEV distribution function H(x) = exp{-[1 + xi (x - mu)/sigma]^(-1/xi)} (xi = 0: Gumbel)."""
    return stats.genextreme.cdf(x, -xi, mu, sigma)          # scipy's shape c = -xi


def gev_ppf(p, xi, mu, sigma):
    return stats.genextreme.ppf(p, -xi, mu, sigma)


def gev_pdf(x, xi, mu, sigma):
    return stats.genextreme.pdf(x, -xi, mu, sigma)


def gpd_sf(y, xi, beta):
    """GPD survival function P(Y > y) = (1 + xi y / beta)^(-1/xi) for the excess Y = X - u (xi = 0: exponential)."""
    return stats.genpareto.sf(y, xi, 0, beta)


def num_hessian(f, p, h=1e-4):
    """Numerical Hessian of f at p (central differences)."""
    p = np.asarray(p, float)
    k = len(p)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j) + f(p - e_i - e_j)) / (4 * h * h)
    return H


def fit_gev(m):
    """Maximum likelihood GEV fit (xi, mu, sigma) of block maxima m, standard errors from the numerical Hessian."""
    m = np.asarray(m, float)
    c, mu, sig = stats.genextreme.fit(m, -0.1, loc=np.mean(m), scale=np.std(m))
    nll = lambda p: -np.sum(stats.genextreme.logpdf(m, -p[0], p[1], p[2])) if p[2] > 0 else np.inf
    res = optimize.minimize(nll, [-c, mu, sig], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
    xi, mu, sig = res.x
    cov = np.linalg.inv(num_hessian(nll, res.x))
    se = np.sqrt(np.diag(cov))
    return {'xi': xi, 'mu': mu, 'sigma': sig, 'se_xi': se[0], 'se_mu': se[1], 'se_sigma': se[2], 'nll': res.fun, 'n': len(m)}


def fit_gpd(y):
    """Maximum likelihood GPD fit (xi, beta) of excesses y > 0, standard errors from the numerical Hessian."""
    y = np.asarray(y, float)
    c, _, b = stats.genpareto.fit(y, floc=0)
    nll = lambda p: -np.sum(stats.genpareto.logpdf(y, p[0], 0, p[1])) if p[1] > 0 else np.inf
    res = optimize.minimize(nll, [c, b], method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 5000})
    xi, beta = res.x
    se = np.sqrt(np.diag(np.linalg.inv(num_hessian(nll, res.x))))
    return {'xi': xi, 'beta': beta, 'se_xi': se[0], 'se_beta': se[1], 'nll': res.fun, 'n_u': len(y)}


def return_level(g, months):
    """Level exceeded by the monthly maximum on average once every `months` months: z = H^(-1)(1 - 1/months)."""
    return float(gev_ppf(1 - 1 / months, g['xi'], g['mu'], g['sigma']))


# =============================================================================
# POT: VaR AND ES (McNeil and Frey, 2000; McNeil, Frey and Embrechts, 2015, Sec. 5.2)
# =============================================================================
def pot(x, q=POT_Q):
    """GPD fit to the losses above the threshold u = q-quantile of the losses x."""
    x = np.asarray(x, float)
    u = float(np.quantile(x, q))
    f = fit_gpd(x[x > u] - u)
    f.update({'u': u, 'n': len(x), 'q': q})
    return f


def pot_var(f, p):
    """VaR_p = u + (beta/xi) [ (n p / N_u)^(-xi) - 1 ], the loss exceeded with probability p."""
    return f['u'] + f['beta'] / f['xi'] * ((f['n'] * p / f['n_u']) ** (-f['xi']) - 1)


def pot_es(f, p):
    """ES_p = VaR_p / (1 - xi) + (beta - xi u) / (1 - xi), the mean loss beyond VaR_p (xi < 1)."""
    return pot_var(f, p) / (1 - f['xi']) + (f['beta'] - f['xi'] * f['u']) / (1 - f['xi'])


def pot_tail_prob(f, x):
    """P(L > x) = (N_u / n) (1 + xi (x - u) / beta)^(-1/xi) for x > u."""
    return f['n_u'] / f['n'] * (1 + f['xi'] * (np.asarray(x, float) - f['u']) / f['beta']) ** (-1 / f['xi'])


def normal_var(x, p):
    """Normal VaR_p of losses x: mu_L + sigma_L z_(1-p)."""
    return float(np.mean(x) + np.std(x, ddof=1) * stats.norm.isf(p))


def normal_es(x, p):
    """Normal ES_p of losses x: mu_L + sigma_L phi(z_p) / p."""
    return float(np.mean(x) + np.std(x, ddof=1) * stats.norm.pdf(stats.norm.isf(p)) / p)


def hist_var(x, p):
    """Historical VaR_p: the empirical (1 - p) quantile of the losses."""
    return float(np.quantile(x, 1 - p))


def hist_es(x, p):
    """Historical ES_p: the mean of the losses at or above the historical VaR_p."""
    x = np.asarray(x, float)
    return float(x[x >= hist_var(x, p)].mean())


def risk_table(names=ASSETS):
    """VaR 1%, ES 2.5% and VaR 0.1% of daily losses: EVT (POT), historical simulation, Normal; Hill quantile."""
    out = {}
    for k in names:
        x = losses(k).values
        f = pot(x)
        kk = int(HILL_FRAC * len(x))
        out[k] = {'pot': f, 'n': len(x)}
        for lab, p in [('var1', 0.01), ('var01', 0.001)]:
            out[k][lab] = {'EVT': pot_var(f, p), 'Historical': hist_var(x, p), 'Normal': normal_var(x, p),
                           'Hill': hill_quantile(x, kk, p)}
        out[k]['es25'] = {'EVT': pot_es(f, 0.025), 'Historical': hist_es(x, 0.025), 'Normal': normal_es(x, 0.025)}
        out[k]['max'] = float(x.max())
    return out


# =============================================================================
# 1. MOTIVATION: CRASH DAYS AND LIGHT VS HEAVY TAILS
# =============================================================================
def crash_table(names=ASSETS):
    """The largest daily loss of each series, its z-score under the Normal distribution fitted to the whole sample,
    the Normal probability of such a loss and the implied waiting time in years (252 trading days, 365 for Bitcoin)."""
    out = {}
    for k in names:
        r = returns(k)
        mu, sd = r.mean(), r.std()
        d = r.idxmin()
        z = (r.min() - mu) / sd
        p = stats.norm.cdf(z)
        days = 365 if k == 'btc' else 252
        out[k] = {'date': d.date().isoformat(), 'ret': float(r.min()), 'mu': float(mu), 'sd': float(sd), 'z': float(z),
                  'logp': float(stats.norm.logcdf(z) / np.log(10)), 'n': len(r), 'first': r.index[0].date().isoformat(),
                  'years': float(1 / (p * days)) if p > 0 else float('inf'),
                  'log_years': float(-stats.norm.logcdf(z) / np.log(10) - np.log10(days)),
                  'n_below_4sd': int((r < mu - 4 * sd).sum()), 'exp_below_4sd': float(len(r) * stats.norm.cdf(-4))}
    return out


def worst_days(names=ASSETS, m=3):
    """The m largest daily losses of each series (at most one per calendar month), with their dates."""
    out = {}
    for k in names:
        r = returns(k).nsmallest(40)
        r = r[~pd.Series(r.index.to_period('M'), index=r.index).duplicated()].iloc[:m]
        out[k] = [{'date': d.date().isoformat(), 'ret': float(v)} for d, v in r.items()]
    return out


def crash_1987(k='sp500', drop=0.20):
    """19 October 1987: the S&P 500 fell about 20% (Carlson, 2007). Its log return and Normal z-score with the
    mean and standard deviation of the S&P 500 daily log returns since 1990."""
    r = returns(k)
    x = 100 * np.log(1 - drop)
    z = (x - r.mean()) / r.std()
    return {'logret': float(x), 'z': float(z), 'log10p': float(stats.norm.logcdf(z) / np.log(10))}


def fig_crash_days(save=True):
    """Daily log returns of the S&P 500 and the BET with the level of a 1-in-100-years loss under the Normal law."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=False)
    out = {}
    for ax, k in zip(axes, ['sp500', 'bet']):
        r = returns(k)
        mu, sd = r.mean(), r.std()
        lvl = mu + sd * stats.norm.ppf(1 / (100 * 252))
        ax.plot(r.index, r.values, color=COLORS[k], lw=0.5, label=f'{LABELS[k]} daily log return (%)')
        ax.axhline(lvl, color='black', lw=0.9, ls='--', label='Normal model: loss seen once in 100 years')
        worst = r.nsmallest(20)
        worst = worst[~pd.Series(worst.index.to_period('M'), index=worst.index).duplicated()].iloc[:3]   # one per month
        last, dy = None, -2
        for d, v in worst.sort_index().items():          # labels of nearby dates are stacked, not overprinted
            dy = dy + 11 if last is not None and (d - last).days < 2200 else -2
            ax.annotate(d.strftime('%d %b %Y'), (d, v), xytext=(8, dy), textcoords='offset points', fontsize=9.5, color='black')
            last = d
        ax.set_ylabel('%')
        ax.set_title(LABELS[k], loc='left', fontsize=12)
        out[k] = {'level100': float(lvl), 'n_below': int((r < lvl).sum())}
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_crash_days')
    return out


def fig_light_heavy(save=True):
    """P(X > x) on log-log axes: Normal and Exponential (light tails) against Student-t(3) and Pareto(3) (heavy)."""
    x = np.logspace(0, 2, 300)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sd_t = np.sqrt(3)                                     # standard deviation of the Student-t with 3 d.f.
    ax.loglog(x, stats.norm.sf(x), color=MODEL_COL['Normal'], label='Normal N(0, 1)')
    ax.loglog(x, stats.expon.sf(x), color='#2E7D32', label='Exponential (1)')
    ax.loglog(x, stats.t.sf(x, 3), color='#1A3A6E', label='Student-t, 3 degrees of freedom')
    ax.loglog(x, x ** -3.0, color='#8E44AD', ls='--', label='Pareto: P(X > x) = x^(-3)')
    ax.set_ylim(1e-12, 1.2)
    ax.set_xlabel('x (log scale)')
    ax.set_ylabel('P(X > x) (log scale)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.18)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_light_heavy')
    return {'t10': float(stats.t.sf(10, 3)), 'n10': float(stats.norm.sf(10)), 'e10': float(stats.expon.sf(10)),
            'p10': 10 ** -3.0, 't_ratio': float(stats.t.sf(20, 3) / stats.t.sf(10, 3)), 'sd_t': sd_t}


# =============================================================================
# 2. REGULAR VARIATION: LOG-LOG TAIL PLOT
# =============================================================================
def fig_loglog_real(save=True):
    """Empirical P(L > x) of daily losses on log-log axes with the least-squares line through the 2.5% largest losses
    (slope = -alpha, as in SFElshill) and the Normal tail."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    out = {}
    for ax, k in zip(axes.flat, ASSETS):
        x = losses(k).values
        n = len(x)
        s = np.sort(x[x > 0])[::-1]
        emp = np.arange(1, len(s) + 1) / n
        ax.loglog(s, emp, '.', ms=2.5, color=COLORS[k], label='Data: share of days with a loss above x')
        kk = int(HILL_FRAC * n)
        a_ls = ls_tail_index(x, kk)
        c = np.exp(np.mean(np.log(emp[:kk]) + a_ls * np.log(s[:kk])))
        xx = np.logspace(np.log10(s[kk]), np.log10(1.5 * s[0]), 50)
        ax.loglog(xx, c * xx ** -a_ls, color='black', lw=1.2, label='Least-squares line, 2.5% largest losses')
        xs = np.logspace(np.log10(0.3), np.log10(1.5 * s[0]), 200)
        pn = stats.norm.sf(xs, np.mean(x), np.std(x, ddof=1))
        ax.loglog(xs[pn > 1e-7], pn[pn > 1e-7], color=MODEL_COL['Normal'], ls='--', label='Normal distribution')
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(0.3, 1.5 * s[0])
        ax.set_title(f'{LABELS[k]}: slope -{a_ls:.2f}', loc='left', fontsize=12)
        ax.set_xlabel('Loss x (%, log scale)')
        ax.set_ylabel('P(L > x)')
        out[k] = a_ls
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_loglog_real')
    return out


# =============================================================================
# 3. HILL ESTIMATOR AND HILL PLOT
# =============================================================================
def fig_hill_plot(names=ASSETS, fname='sfm_ch5_hill_plot', band='sp500', save=True):
    """Hill plot: alpha_hat(k) of daily losses against k/n (0.5% to 10%), with the i.i.d. 95% band for one series."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fr = np.round(np.arange(0.005, 0.1001, 0.0025), 4)
    out = {}
    for k in names:
        x = losses(k).values
        n = len(x)
        a = np.array([hill(x, max(int(f * n), 5)) for f in fr])
        ax.plot(100 * fr, a, color=COLORS[k], lw=1.5, label=LABELS[k])
        if k == band:
            kk = np.maximum((fr * n).astype(int), 5)
            ax.fill_between(100 * fr, a * (1 - 1.96 / np.sqrt(kk)), a * (1 + 1.96 / np.sqrt(kk)), color=COLORS[k], alpha=0.15,
                            lw=0, label=f'{LABELS[k]}: 95% interval')
        out[k] = {f'{f:.4f}': float(v) for f, v in zip(fr, a)}
    ax.axvline(100 * HILL_FRAC, color='black', lw=0.8, ls=':', label='k = 2.5% of n')
    ax.axhline(2, color='black', lw=0.8, ls='--', label='alpha = 2: variance finite above')
    ax.axhline(4, color='black', lw=0.8, ls='-.', label='alpha = 4: kurtosis finite above')
    ax.set_xlabel('Share of the largest losses used, k/n (%)')
    ax.set_ylabel('Hill estimate of alpha')
    ax.set_ylim(1, 6)
    st.legend_outside_bottom(ax, ncol=4, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig(fname)
    return out


def hill_table(names=ASSETS + STOCKS):
    """Hill inference for losses and gains of each series; least-squares index; Hill quantiles."""
    out = {}
    for k in names:
        r = returns(k)
        L = -r.values
        h = hill_inference(L)
        g = hill(r.values, h['k'])
        out[k] = dict(h, alpha_gain=g, first=r.index[0].date().isoformat(), last=r.index[-1].date().isoformat())
    return out


# =============================================================================
# 4. MEAN EXCESS FUNCTION
# =============================================================================
def mean_excess(x, us):
    """Empirical mean excess e(u) = mean of (X - u) over the observations X > u."""
    x = np.asarray(x, float)
    return np.array([np.mean(x[x > u] - u) if np.sum(x > u) >= 5 else np.nan for u in us])


def fig_mean_excess(names=ASSETS, save=True):
    """Empirical mean excess function of daily losses (as in SFEMeanExcessFun), with the threshold u (90% quantile)
    and the straight line implied by the GPD fitted above u: e(v) = (beta + xi (v - u)) / (1 - xi)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    out = {}
    for ax, k in zip(axes.flat, names):
        x = losses(k).values
        us = np.quantile(x, np.linspace(0.5, 0.995, 120))
        e = mean_excess(x, us)
        f = pot(x)
        ax.plot(us, e, 'o', ms=3, color=COLORS[k], label='Empirical mean excess e(u)')
        v = np.linspace(f['u'], us[-1], 50)
        ax.plot(v, (f['beta'] + f['xi'] * (v - f['u'])) / (1 - f['xi']), color='black', lw=1.3, label='GPD fitted above u')
        ax.axvline(f['u'], color='black', lw=0.8, ls=':', label='u = 90% quantile of the losses')
        ax.set_title(LABELS[k], loc='left', fontsize=12)
        ax.set_xlabel('Threshold u (daily loss, %)')
        ax.set_ylabel('e(u) (%)')
        out[k] = {'u': f['u'], 'e_u': float(mean_excess(x, [f['u']])[0])}
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_mean_excess')
    return out


# =============================================================================
# 5. BLOCK MAXIMA AND THE GEV
# =============================================================================
def fig_gev_types(save=True):
    """Densities and distribution functions of the three extreme value laws (as in SFEevt1): Gumbel (xi = 0),
    Frechet (xi = 0.5) and Weibull (xi = -0.5), standard location and scale."""
    x = np.linspace(-4, 7, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for xi, lab, c in [(0.0, 'Gumbel (xi = 0)', '#1A3A6E'), (0.5, 'Frechet (xi = 0.5)', '#CD0000'),
                       (-0.5, 'Weibull (xi = -0.5)', '#2E7D32')]:
        axes[0].plot(x, gev_pdf(x, xi, 0, 1), color=c, lw=1.6, label=lab)
        axes[1].plot(x, gev_cdf(x, xi, 0, 1), color=c, lw=1.6, label='_')
    axes[0].set_title('Density h(x)', loc='left', fontsize=12)
    axes[1].set_title('Distribution function H(x)', loc='left', fontsize=12)
    for ax in axes:
        ax.set_xlabel('x')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_gev_types')


def fig_maxima_sim(n=250, reps=5000, save=True):
    """Fisher-Tippett-Gnedenko in a simulation: maxima of n = 250 draws, normalised, against the limit law.
    Exponential(1): M - ln n -> Gumbel; Pareto(3): M / n^(1/3) -> Frechet; Uniform(0, 1): n (M - 1) -> Weibull."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    cases = [('Exponential(1): M - ln n', lambda: rng.exponential(1, (reps, n)).max(1) - np.log(n), 0.0, '#1A3A6E', (-3, 7)),
             ('Pareto(3): M / n^(1/3)', lambda: (rng.pareto(3, (reps, n)) + 1).max(1) / n ** (1 / 3), 1 / 3, '#CD0000', (0, 4)),
             ('Uniform(0, 1): n (M - 1)', lambda: n * (rng.uniform(0, 1, (reps, n)).max(1) - 1), -1.0, '#2E7D32', (-5, 0.2))]
    out = {}
    for ax, (lab, draw, xi, c, lim) in zip(axes, cases):
        m = draw()
        x = np.linspace(*lim, 400)
        bins = np.linspace(*lim, 50)
        ax.hist(m, bins=bins, density=True, color=c, alpha=0.35, label='_')
        if xi == 0:
            f = gev_pdf(x, 0, 0, 1)
        elif xi > 0:
            f = gev_pdf(x, xi, 1, xi)                     # Frechet: P(M/n^(1/3) <= x) -> exp(-x^-3) = GEV(1/3, 1, 1/3)
        else:
            f = gev_pdf(x, -1, -1, 1)                    # Weibull: P(n(M-1) <= x) -> exp(x) for x <= 0 = GEV(-1, -1, 1)
        ax.plot(x, f, color='black', lw=1.4, label='Limit law (GEV)')
        ax.set_title(lab, loc='left', fontsize=11)
        out[lab] = float(np.mean(m))
    axes[0].bar([0], [0], color='#1A3A6E', alpha=0.35, label='Simulated maxima (5,000 samples)')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_maxima_sim')
    return out


def monthly_maxima(k):
    """Largest daily loss in each calendar month (blocks of about 21 trading days; 30 days for Bitcoin)."""
    L = losses(k)
    g = L.groupby([L.index.year, L.index.month])
    m = g.max()[g.count() >= 15]
    return m


def gev_table(names=ASSETS, B=200, seed=SEED):
    """GEV fit to the monthly maxima of daily losses; 10-year and 50-year return levels with bootstrap intervals."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in names:
        m = monthly_maxima(k).values
        g = fit_gev(m)
        rl10, rl50 = return_level(g, 120), return_level(g, 600)
        bs = []
        for _ in range(B):
            s = rng.choice(m, len(m))
            try:
                c, mu, sig = stats.genextreme.fit(s, -g['xi'], loc=g['mu'], scale=g['sigma'])
                bs.append([-c, stats.genextreme.ppf(1 - 1 / 120, c, mu, sig)])
            except Exception:
                pass
        bs = np.array(bs)
        out[k] = dict(g, rl10=rl10, rl50=rl50, rl1=return_level(g, 12), rl10_lo=float(np.percentile(bs[:, 1], 2.5)),
                      rl10_hi=float(np.percentile(bs[:, 1], 97.5)), xi_lo=g['xi'] - 1.96 * g['se_xi'],
                      xi_hi=g['xi'] + 1.96 * g['se_xi'], max=float(m.max()), months=len(m),
                      years=len(m) / 12, n_above_rl10=int((m > rl10).sum()))
    return out


def fig_gev_ppqq(k='sp500', save=True):
    """PP and QQ plots of the GEV fitted to the monthly maxima of daily losses (as in SFEtailGEV_pp and _qq)."""
    m = np.sort(monthly_maxima(k).values)
    g = fit_gev(m)
    p = (np.arange(1, len(m) + 1) - 0.5) / len(m)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(p, gev_cdf(m, g['xi'], g['mu'], g['sigma']), 'o', ms=3, color=COLORS[k], label='Monthly maxima')
    axes[0].plot([0, 1], [0, 1], color='black', lw=0.8, ls='--', label='45-degree line')
    axes[0].set_xlabel('Empirical probability')
    axes[0].set_ylabel('GEV probability')
    axes[0].set_title('PP plot', loc='left', fontsize=12)
    q = gev_ppf(p, g['xi'], g['mu'], g['sigma'])
    axes[1].plot(q, m, 'o', ms=3, color=COLORS[k], label='_')
    lim = [min(q.min(), m.min()), max(q.max(), m.max())]
    axes[1].plot(lim, lim, color='black', lw=0.8, ls='--', label='_')
    axes[1].set_xlabel('GEV quantile (%)')
    axes[1].set_ylabel('Monthly maximum of daily losses (%)')
    axes[1].set_title('QQ plot', loc='left', fontsize=12)
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig(f'sfm_ch5_gev_ppqq_{k}' if k != 'sp500' else 'sfm_ch5_gev_ppqq')
    return g


def fig_return_levels(gt, names=ASSETS, save=True):
    """Return level plot: the monthly maximum of daily losses exceeded on average once in T years (GEV fit)
    against the empirical return levels of the observed monthly maxima."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    T = np.logspace(np.log10(1 / 12 * 1.05), np.log10(100), 200)
    for k in names:
        g = gt[k]
        z = gev_ppf(1 - 1 / (12 * T), g['xi'], g['mu'], g['sigma'])
        ax.semilogx(T, z, color=COLORS[k], lw=1.6, label=f'{LABELS[k]} (GEV)')
        m = np.sort(monthly_maxima(k).values)
        pe = (np.arange(1, len(m) + 1)) / (len(m) + 1)
        ax.semilogx(1 / (12 * (1 - pe)), m, 'o', ms=2.5, color=COLORS[k], alpha=0.6, label='_')
    ax.axvline(10, color='black', lw=0.8, ls=':', label='10-year return period')
    ax.set_xlabel('Return period T (years, log scale)')
    ax.set_ylabel('Return level: daily loss (%)')
    st.legend_outside_bottom(ax, ncol=5, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_return_levels')


# =============================================================================
# 6. PEAKS OVER THRESHOLD AND THE GPD
# =============================================================================
def fig_gpd_densities(save=True):
    """GPD densities for xi = -0.3, 0, 0.3, 0.6 (beta = 1): the sign of xi decides the type of tail."""
    y = np.linspace(0, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for xi, c in [(-0.3, '#2E7D32'), (0.0, '#1A3A6E'), (0.3, '#E67E22'), (0.6, '#CD0000')]:
        axes[0].plot(y, stats.genpareto.pdf(y, xi, 0, 1), color=c, lw=1.6, label=f'xi = {xi:g}')
        yy = np.logspace(-1, 2, 300)
        sf = stats.genpareto.sf(yy, xi, 0, 1)
        ok = sf > 1e-9
        axes[1].loglog(yy[ok], sf[ok], color=c, lw=1.6, label='_')
    axes[0].set_title('Density g(y)', loc='left', fontsize=12)
    axes[1].set_title('P(Y > y), log-log axes', loc='left', fontsize=12)
    axes[0].set_xlabel('Excess y')
    axes[1].set_xlabel('Excess y (log scale)')
    axes[1].set_ylim(1e-6, 1.2)
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_gpd_densities')


def threshold_stability(k, qs=np.round(np.arange(0.80, 0.991, 0.01), 2)):
    """xi_hat of the GPD with its 95% interval for thresholds u at the 80%, ..., 99% quantiles of the losses."""
    x = losses(k).values
    rows = []
    for q in qs:
        f = pot(x, q)
        rows.append({'q': float(q), 'u': f['u'], 'n_u': f['n_u'], 'xi': f['xi'], 'se': f['se_xi'],
                     'var1': pot_var(f, 0.01), 'var01': pot_var(f, 0.001)})
    return rows


def fig_threshold_stability(names=('sp500', 'bet'), save=True):
    """Parameter stability plot: xi_hat (95% interval) and VaR 1% against the threshold quantile."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    out = {}
    for k in names:
        rows = threshold_stability(k)
        q = np.array([r['q'] for r in rows]) * 100
        xi = np.array([r['xi'] for r in rows])
        se = np.array([r['se'] for r in rows])
        axes[0].plot(q, xi, 'o-', ms=3, color=COLORS[k], label=LABELS[k])
        axes[0].fill_between(q, xi - 1.96 * se, xi + 1.96 * se, color=COLORS[k], alpha=0.15, lw=0)
        axes[1].plot(q, [r['var1'] for r in rows], 'o-', ms=3, color=COLORS[k], label='_')
        out[k] = rows
    for ax in axes:
        ax.axvline(100 * POT_Q, color='black', lw=0.8, ls=':')
        ax.set_xlabel('Threshold u: quantile of the losses (%)')
    axes[0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0].set_ylabel('xi-hat (95% interval)')
    axes[1].set_ylabel('EVT VaR 1% (%)')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_threshold_stability')
    return out


def fig_gpd_ppqq(k='sp500', save=True, fname=None):
    """PP and QQ plots of the GPD fitted to the excesses over u (as in SFEtailGPareto_pp and _qq)."""
    x = losses(k).values
    f = pot(x)
    y = np.sort(x[x > f['u']] - f['u'])
    p = (np.arange(1, len(y) + 1) - 0.5) / len(y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(p, stats.genpareto.cdf(y, f['xi'], 0, f['beta']), 'o', ms=2.5, color=COLORS[k], label='Excesses over u')
    axes[0].plot([0, 1], [0, 1], color='black', lw=0.8, ls='--', label='45-degree line')
    axes[0].set_xlabel('Empirical probability')
    axes[0].set_ylabel('GPD probability')
    axes[0].set_title('PP plot', loc='left', fontsize=12)
    q = stats.genpareto.ppf(p, f['xi'], 0, f['beta'])
    axes[1].plot(q, y, 'o', ms=2.5, color=COLORS[k], label='_')
    lim = [0, max(q.max(), y.max())]
    axes[1].plot(lim, lim, color='black', lw=0.8, ls='--', label='_')
    axes[1].set_xlabel('GPD quantile of the excess (%)')
    axes[1].set_ylabel('Observed excess L - u (%)')
    axes[1].set_title('QQ plot', loc='left', fontsize=12)
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig(fname or 'sfm_ch5_gpd_ppqq')
    return {'q_top': float(q[-1]), 'y_top': float(y[-1])}


def fig_tail_fit(rt, names=ASSETS, save=True):
    """The fitted tail: empirical P(L > x) above u (log-log) against the POT tail and the Normal tail, with the
    EVT VaR 1% and VaR 0.1%."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, k in zip(axes.flat, names):
        x = losses(k).values
        n = len(x)
        f = rt[k]['pot']
        s = np.sort(x)[::-1]
        emp = np.arange(1, n + 1) / n
        keep = s > f['u']
        ax.loglog(s[keep], emp[keep], 'o', ms=3.2, color=MODEL_COL['Data'], zorder=3, label='Data: share of days with a loss above x')
        xx = np.logspace(np.log10(f['u']), np.log10(2 * s[0]), 200)
        ax.loglog(xx, pot_tail_prob(f, xx), color=MODEL_COL['EVT'], lw=1.1, zorder=2, label='EVT: GPD above u')
        pn = stats.norm.sf(xx, np.mean(x), np.std(x, ddof=1))
        ok = pn > 1e-7
        ax.loglog(xx[ok], pn[ok], color=MODEL_COL['Normal'], ls='--', label='Normal distribution')
        for p, ls in [(0.01, ':'), (0.001, '-.')]:
            ax.axhline(p, color='black', lw=0.6, ls=ls, label='p = 1%' if p == 0.01 else 'p = 0.1%')
        ax.set_ylim(1e-5, 0.15)
        ax.set_title(f"{LABELS[k]}: xi = {f['xi']:.2f}", loc='left', fontsize=12)
        ax.set_xlabel('Loss x (%, log scale)')
        ax.set_ylabel('P(L > x)')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_tail_fit')


def crash_waiting(rt, ct, c87):
    """Waiting time (years) of the largest observed loss, and of the 1987 crash, under the Normal and the EVT tail."""
    out = {}
    for k in ct:
        f = rt[k]['pot']
        days = 365 if k == 'btc' else 252
        p = float(pot_tail_prob(f, -ct[k]['ret']))
        out[k] = {'p_evt': p, 'years_evt': 1 / (p * days)}
    f = rt['sp500']['pot']
    p = float(pot_tail_prob(f, -c87['logret']))
    out['1987'] = {'p_evt': p, 'years_evt': 1 / (p * 252)}
    return out


# =============================================================================
# 7. OUT-OF-SAMPLE CHECK
# =============================================================================
def out_of_sample(names=ASSETS, split=SPLIT):
    """Estimate VaR 1% and VaR 0.1% on the losses until `split` (EVT, historical, Normal); count the days after
    `split` with a loss above each VaR (expected rates: 1% and 0.1%)."""
    out = {}
    for k in names:
        L = losses(k)
        a, b = L.loc[:split].values, L.loc[split:].iloc[1:].values
        f = pot(a)
        row = {'n_est': len(a), 'n_test': len(b), 'first_test': L.loc[split:].index[1].date().isoformat()}
        for lab, p in [('var1', 0.01), ('var01', 0.001)]:
            v = {'EVT': pot_var(f, p), 'Historical': hist_var(a, p), 'Normal': normal_var(a, p)}
            row[lab] = {m: {'var': x, 'exc': int((b > x).sum())} for m, x in v.items()}
            row[lab]['expected'] = p * len(b)
        out[k] = row
    return out


def fig_oos(k='sp500', split=SPLIT, save=True):
    """Daily losses after the estimation period with the VaR 1% of the three methods (estimated until 2019)."""
    L = losses(k)
    a, b = L.loc[:split].values, L.loc[split:].iloc[1:]
    f = pot(a)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(b.index, b.values, color=COLORS[k], lw=0.6, label=f'{LABELS[k]} daily loss (%)')
    for m, v, ls in [('Normal', normal_var(a, 0.01), '--'), ('Historical', hist_var(a, 0.01), ':'), ('EVT', pot_var(f, 0.01), '-')]:
        ax.axhline(v, color=MODEL_COL[m], lw=1.3, ls=ls, label=f'VaR 1%, {m}: {v:.2f}%')
    v01 = pot_var(f, 0.001)
    ax.axhline(v01, color=MODEL_COL['EVT'], lw=1.0, ls='-.', label=f'VaR 0.1%, EVT: {v01:.2f}%')
    ax.set_ylabel('Daily loss (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch5_oos')


# =============================================================================
# 8. AI SECTION: HAS THE TAIL OF THE BET CHANGED?
# =============================================================================
def bet_periods(k='bet', edge='2010-01-01', B=500, seed=SEED):
    """Hill index of daily BET losses (k = 2.5% of n) before and after 1 January 2010, with moving-block bootstrap
    standard errors and the z statistic of the difference."""
    L = losses(k)
    res = {}
    for lab, x in [('before', L.loc[:edge].iloc[:-1]), ('after', L.loc[edge:])]:
        h = hill_inference(x.values, B=B, seed=seed)
        res[lab] = dict(h, first=x.index[0].date().isoformat(), last=x.index[-1].date().isoformat())
    res['z'] = (res['after']['alpha'] - res['before']['alpha']) / np.sqrt(res['before']['se_block'] ** 2 + res['after']['se_block'] ** 2)
    return res


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


if __name__ == '__main__':
    st.apply()
    N = {}
    N['crash'] = crash_table()
    N['c87'] = crash_1987()
    N['worst'] = worst_days()
    N['crash_days'] = fig_crash_days()
    N['light_heavy'] = fig_light_heavy()
    N['ls'] = fig_loglog_real()
    N['hill_plot'] = fig_hill_plot()
    N['hill'] = hill_table()
    fig_hill_plot(STOCKS, 'sfm_ch5_hill_bvb', band='tlv')
    N['me'] = fig_mean_excess()
    fig_gev_types()
    N['maxima_sim'] = fig_maxima_sim()
    gt = gev_table()
    N['gev'] = gt
    fig_gev_ppqq('sp500')
    fig_return_levels(gt)
    fig_gpd_densities()
    N['stab'] = fig_threshold_stability()
    N['gpd_qq'] = fig_gpd_ppqq('sp500')
    rt = risk_table(ASSETS + STOCKS)
    N['risk'] = rt
    fig_tail_fit(rt)
    N['wait'] = crash_waiting(rt, N['crash'], N['c87'])
    N['oos'] = out_of_sample()
    fig_oos()
    N['bet_periods'] = bet_periods()
    N['end'] = returns('sp500').index[-1].date().isoformat()
    tab = pd.DataFrame({LABELS[k]: {'n': v['n'], 'xi': v['pot']['xi'], 'se_xi': v['pot']['se_xi'], 'u': v['pot']['u'],
                                    'VaR1_EVT': v['var1']['EVT'], 'VaR1_hist': v['var1']['Historical'],
                                    'VaR1_Normal': v['var1']['Normal'], 'ES2.5_EVT': v['es25']['EVT'],
                                    'ES2.5_hist': v['es25']['Historical'], 'ES2.5_Normal': v['es25']['Normal'],
                                    'VaR0.1_EVT': v['var01']['EVT'], 'VaR0.1_hist': v['var01']['Historical'],
                                    'VaR0.1_Normal': v['var01']['Normal']} for k, v in rt.items()}).T
    tab.to_csv(os.path.join(TABLE_DIR, 'ch5_tail_table.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch5_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1, default=str)
    print(tab.round(2))
    print(json.dumps(to_json({k: N[k] for k in ['crash', 'c87', 'wait', 'oos', 'bet_periods', 'gev']}), indent=1)[:6000])
    print(pd.DataFrame(N['hill']).T[['n', 'k', 'alpha', 'lo', 'hi', 'se_block', 'alpha_gain', 'alpha_1', 'alpha_5', 'ls']].round(2))
