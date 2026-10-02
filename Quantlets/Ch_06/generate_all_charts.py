"""
generate_all_charts.py -- charts and numbers of Chapter 6 (SFM): model selection and risk management
=====================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * candidates   -- Normal, Student-t, skewed-t (Hansen, 1994), GED (generalised error distribution), two-component
                    Normal mixture, NIG (normal inverse Gaussian) and alpha-stable laws; densities with variance 1;
  * summary      -- the summary statistics of the Quantlet SFESumm (DAX monthly log returns, July 2004 - May 2014,
                    and the same statistics up to 2026), daily summary statistics of all series;
  * likelihood   -- maximum likelihood fits of the seven models to the daily log returns of the BET, S&P 500, DAX,
                    Bitcoin and three Bucharest stocks; the profile log-likelihood of the Student-t degrees of freedom;
  * comparison   -- likelihood-ratio tests for nested models, the Vuong test for non-nested models, AIC, BIC,
                    Akaike weights;
  * fit tests    -- Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling statistics with parametric bootstrap
                    p-values; a power study with estimated parameters (Lilliefors setting); PP and QQ plots;
  * tails        -- VaR 1% and ES 2.5% under each model, in-sample exceedances and the Kupiec test;
  * validation   -- out-of-sample log score and VaR 1% exceedances (estimation until 2019, test 2020-2026);
                    overfitting with Normal mixtures of 1-6 components; rolling VaR backtest;
                    the VaR reliability QQ plot of the Quantlet SFEVaRqqplot;
  * uncertainty  -- moving-block bootstrap of the AIC winner and of VaR 1%; VaR averaged with Akaike weights.
Output: charts/sfm_ch6_*.pdf/.png, Quantlets/Ch_06/ch6_numbers.json, ch6_fits.csv
Run:  python3 Quantlets/Ch_06/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import LABELS, load_close, log_returns   # noqa: E402
import sfm_style as st                                 # noqa: E402


def _load_ch3():
    """The alpha-stable law of Chapter 3 (Quantlets/Ch_03/generate_all_charts.py), loaded under its own name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('sfm_ch3_stable', os.path.join(HERE, '..', 'Ch_03', 'generate_all_charts.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sfm_ch3_stable'] = mod
    spec.loader.exec_module(mod)
    return mod


_ch3 = _load_ch3()
stable_cf, s0_to_s1, StableGrid = _ch3.stable_cf, _ch3.s0_to_s1, _ch3.StableGrid
mcculloch, mcculloch_quantiles, stable_nll, stable_mle = _ch3.mcculloch, _ch3.mcculloch_quantiles, _ch3.stable_nll, _ch3.stable_mle
STABLE = [stable_cf, s0_to_s1, StableGrid, mcculloch, mcculloch_quantiles, stable_nll, stable_mle]

TABLE_DIR = HERE
START = '2000-01-01'                  # indices since 2000; Bitcoin over its full history; stocks since 2010
STOCK_START = '2010-01-01'
ASSETS = ['bet', 'sp500', 'dax', 'btc']
STOCKS = ['tlv', 'snp', 'brd']
# Banca Transilvania: the adjustment for the bonus shares of May 2016 is applied one day late in the adjusted close
# (log returns of about -15% and +20% on two consecutive days, see Chapter 2); the two days are removed.
BAD_DAYS = {'tlv': ['2016-05-30', '2016-05-31']}
MODELS = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG', 'Stable']
K = {'Normal': 2, 'Student-t': 3, 'Skewed-t': 4, 'GED': 3, 'Mixture': 5, 'NIG': 4, 'Stable': 4}
MODEL_COL = {'Normal': '#CD0000', 'Student-t': '#2E7D32', 'Skewed-t': '#8E44AD', 'GED': '#E67E22',
             'Mixture': '#17A2B8', 'NIG': '#B5853F', 'Stable': '#1A3A6E', 'Data': 'black', 'Historical': '#DC3545'}
SHORT = {'bet': 'BET', 'sp500': 'S&P 500', 'dax': 'DAX', 'btc': 'Bitcoin', 'tlv': 'TLV', 'snp': 'SNP', 'brd': 'BRD'}
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F',
          'tlv': '#E67E22', 'snp': '#2E7D32', 'brd': '#8E44AD'}
SEED = 2026
SPLIT = '2019-12-31'                  # out-of-sample check: estimation until this day, test afterwards
ALPHA_VAR = 0.01                      # VaR 1%
ALPHA_ES = 0.025                      # ES 2.5%
BLOCK = 20                            # moving-block bootstrap: blocks of 20 trading days


# =============================================================================
# DATA
# =============================================================================
def returns(k, start=None, end=None):
    """Daily log returns in %: indices since 2000, Bitcoin over its full history, Bucharest stocks since 2010
    (the two erroneous Banca Transilvania days of May 2016 removed)."""
    s0 = start or (STOCK_START if k in STOCKS else START if k != 'btc' else None)
    r = log_returns(k, s0)
    if end:
        r = r.loc[:end]
    if k in BAD_DAYS:
        r = r.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return r


# =============================================================================
# THE CANDIDATE MODELS: log-density, distribution function, quantile, maximum likelihood
# =============================================================================
def skewt_consts(nu, lam):
    """Constants a, b, c of the skewed Student-t of Hansen (1994), standardised to mean 0 and variance 1."""
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def skewt_logpdf(x, mu, sigma, nu, lam):
    """Log-density of Hansen's skewed-t with mean mu, standard deviation sigma, degrees of freedom nu > 2 and
    skewness lam in (-1, 1); lam < 0: longer left tail; lam = 0: the Student-t with variance sigma^2."""
    a, b, c = skewt_consts(nu, lam)
    z = (np.asarray(x, float) - mu) / sigma
    s = np.where(z < -a / b, 1 - lam, 1 + lam)
    return np.log(b * c / sigma) - (nu + 1) / 2 * np.log1p(((b * z + a) / s) ** 2 / (nu - 2))


def skewt_cdf(x, mu, sigma, nu, lam):
    a, b, c = skewt_consts(nu, lam)
    z = (np.asarray(x, float) - mu) / sigma
    k = np.sqrt(nu / (nu - 2))
    lo = (1 - lam) * stats.t.cdf(k * (b * z + a) / (1 - lam), nu)
    hi = (1 - lam) / 2 + (1 + lam) * (stats.t.cdf(k * (b * z + a) / (1 + lam), nu) - 0.5)
    return np.where(z < -a / b, lo, hi)


def skewt_ppf(p, mu, sigma, nu, lam):
    a, b, c = skewt_consts(nu, lam)
    p = np.asarray(p, float)
    k = np.sqrt((nu - 2) / nu)
    lo = ((1 - lam) * k * stats.t.ppf(np.clip(p / (1 - lam), 1e-300, 1), nu) - a) / b
    hi = ((1 + lam) * k * stats.t.ppf(np.clip(0.5 + (p - (1 - lam) / 2) / (1 + lam), 0, 1 - 1e-16), nu) - a) / b
    return mu + sigma * np.where(p < (1 - lam) / 2, lo, hi)


def skewt_rvs(n, mu, sigma, nu, lam, rng):
    return skewt_ppf(rng.uniform(size=n), mu, sigma, nu, lam)


def mixture_em(x, k=2, starts=4, iters=600, tol=1e-9, seed=SEED):
    """Maximum likelihood fit of a k-component Normal mixture by the EM algorithm (several starts, the best kept)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        if s == 0:      # start: equal weights, means at quantiles, spread scales
            w = np.full(k, 1 / k)
            mu = np.quantile(x, (np.arange(k) + 0.5) / k) * 0.2
            sd = x.std() * np.linspace(0.6, 2.0, k)
        else:
            w = rng.dirichlet(np.ones(k) * 3)
            mu = rng.choice(x, k) * 0.3
            sd = x.std() * rng.uniform(0.4, 2.5, k)
        ll_old = -np.inf
        for _ in range(iters):
            lp = np.log(w) + stats.norm.logpdf(x[:, None], mu, sd)
            m = lp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
            ll = lse.sum()
            g = np.exp(lp - lse[:, None])
            nk = g.sum(axis=0) + 1e-12
            w = nk / len(x)
            mu = (g * x[:, None]).sum(axis=0) / nk
            sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
            sd = np.maximum(sd, 1e-3 * x.std())
            if abs(ll - ll_old) < tol * abs(ll):
                break
            ll_old = ll
        if best is None or ll > best['loglik']:
            o = np.argsort(sd)
            best = {'w': w[o].tolist(), 'mu': mu[o].tolist(), 'sd': sd[o].tolist(), 'loglik': float(ll)}
    return best


def mixture_logpdf(x, w, mu, sd):
    lp = np.log(np.asarray(w)) + stats.norm.logpdf(np.asarray(x, float)[:, None], mu, sd)
    m = lp.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))


def mixture_cdf(x, w, mu, sd):
    return (np.asarray(w) * stats.norm.cdf(np.asarray(x, float)[:, None], mu, sd)).sum(axis=1)


def mixture_ppf(p, w, mu, sd):
    p = np.atleast_1d(np.asarray(p, float))
    lo, hi = min(mu) - 60 * max(sd), max(mu) + 60 * max(sd)
    return np.array([optimize.brentq(lambda z: mixture_cdf([z], w, mu, sd)[0] - q, lo, hi) for q in p])


def nig_fit(x):
    """Maximum likelihood fit of the NIG law (scipy.stats.norminvgauss: tail a > 0, skewness |b| < a, loc, scale)."""
    x = np.asarray(x, float)

    def nll(p):
        a = np.exp(p[0])
        b = a * np.tanh(p[1])
        v = -np.sum(stats.norminvgauss.logpdf(x, a, b, p[2], np.exp(p[3])))
        return v if np.isfinite(v) else 1e12
    sd = x.std()
    p0 = [np.log(1.0), 0.0, np.median(x), np.log(sd)]
    res = optimize.minimize(nll, p0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 6000, 'maxfev': 12000})
    a = float(np.exp(res.x[0]))
    return {'a': a, 'b': float(a * np.tanh(res.x[1])), 'loc': float(res.x[2]), 'scale': float(np.exp(res.x[3])), 'loglik': float(-res.fun)}


def skewt_fit(x, start=None):
    """Maximum likelihood fit of Hansen's skewed-t (mu, sigma, nu, lam)."""
    x = np.asarray(x, float)

    def nll(p):
        nu, lam = 2 + np.exp(p[2]), np.tanh(p[3])
        if nu - 2 < 1e-6 or abs(lam) > 0.999:
            return 1e12
        v = -np.sum(skewt_logpdf(x, p[0], np.exp(p[1]), nu, lam))
        return v if np.isfinite(v) else 1e12
    if start is None:
        nu0 = stats.t.fit(x)[0] if len(x) < 3000 else 4.0
        start = [x.mean(), np.log(x.std()), np.log(max(nu0 - 2, 0.5)), 0.0]
    res = optimize.minimize(nll, start, method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 8000, 'maxfev': 16000})
    return {'mu': float(res.x[0]), 'sigma': float(np.exp(res.x[1])), 'nu': float(2 + np.exp(res.x[2])),
            'lam': float(np.tanh(res.x[3])), 'loglik': float(-res.fun)}


def ged_fit(x):
    """Maximum likelihood fit of the GED (scipy.stats.gennorm: shape beta, loc, scale); beta = 2 is the Normal law,
    beta = 1 the Laplace law, beta < 2 heavier tails than the Normal law."""
    x = np.asarray(x, float)

    def nll(p):
        v = -np.sum(stats.gennorm.logpdf(x, np.exp(p[0]), p[1], np.exp(p[2])))
        return v if np.isfinite(v) else 1e12
    res = optimize.minimize(nll, [np.log(1.2), np.median(x), np.log(x.std())], method='Nelder-Mead',
                            options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 4000})
    return {'beta': float(np.exp(res.x[0])), 'loc': float(res.x[1]), 'scale': float(np.exp(res.x[2])), 'loglik': float(-res.fun)}


def t_fit(x):
    """Maximum likelihood fit of the location-scale Student-t (nu, loc, scale)."""
    x = np.asarray(x, float)
    nu, loc, sc = stats.t.fit(x)
    return {'nu': float(nu), 'loc': float(loc), 'scale': float(sc), 'loglik': float(np.sum(stats.t.logpdf(x, nu, loc, sc)))}


def fit_model(name, x):
    """Maximum likelihood fit of one candidate model; returns its parameters and the maximised log-likelihood."""
    x = np.asarray(x, float)
    if name == 'Normal':
        mu, sd = x.mean(), x.std(ddof=0)
        return {'mu': float(mu), 'sigma': float(sd), 'loglik': float(np.sum(stats.norm.logpdf(x, mu, sd)))}
    if name == 'Student-t':
        return t_fit(x)
    if name == 'Skewed-t':
        return skewt_fit(x)
    if name == 'GED':
        return ged_fit(x)
    if name == 'Mixture':
        return mixture_em(x, 2)
    if name == 'NIG':
        return nig_fit(x)
    if name == 'Stable':
        s = stable_mle(x, se=False)
        return dict(s, loglik=float(s['loglik']))
    raise ValueError(name)


def logpdf(name, x, p):
    x = np.asarray(x, float)
    if name == 'Normal':
        return stats.norm.logpdf(x, p['mu'], p['sigma'])
    if name == 'Student-t':
        return stats.t.logpdf(x, p['nu'], p['loc'], p['scale'])
    if name == 'Skewed-t':
        return skewt_logpdf(x, p['mu'], p['sigma'], p['nu'], p['lam'])
    if name == 'GED':
        return stats.gennorm.logpdf(x, p['beta'], p['loc'], p['scale'])
    if name == 'Mixture':
        return mixture_logpdf(x, p['w'], p['mu'], p['sd'])
    if name == 'NIG':
        return stats.norminvgauss.logpdf(x, p['a'], p['b'], p['loc'], p['scale'])
    if name == 'Stable':
        return np.log(StableGrid(p['alpha'], p['beta']).pdf((x - p['delta0']) / p['gamma']) / p['gamma'])
    raise ValueError(name)


def cdf(name, x, p):
    x = np.asarray(x, float)
    if name == 'Normal':
        return stats.norm.cdf(x, p['mu'], p['sigma'])
    if name == 'Student-t':
        return stats.t.cdf(x, p['nu'], p['loc'], p['scale'])
    if name == 'Skewed-t':
        return skewt_cdf(x, p['mu'], p['sigma'], p['nu'], p['lam'])
    if name == 'GED':
        return stats.gennorm.cdf(x, p['beta'], p['loc'], p['scale'])
    if name == 'Mixture':
        return mixture_cdf(x, p['w'], p['mu'], p['sd'])
    if name == 'NIG':
        return stats.norminvgauss.cdf(x, p['a'], p['b'], p['loc'], p['scale'])
    if name == 'Stable':
        return StableGrid(p['alpha'], p['beta']).cdf((x - p['delta0']) / p['gamma'])
    raise ValueError(name)


def ppf(name, q, p):
    q = np.asarray(q, float)
    if name == 'Normal':
        return stats.norm.ppf(q, p['mu'], p['sigma'])
    if name == 'Student-t':
        return stats.t.ppf(q, p['nu'], p['loc'], p['scale'])
    if name == 'Skewed-t':
        return skewt_ppf(q, p['mu'], p['sigma'], p['nu'], p['lam'])
    if name == 'GED':
        return stats.gennorm.ppf(q, p['beta'], p['loc'], p['scale'])
    if name == 'Mixture':
        return mixture_ppf(q, p['w'], p['mu'], p['sd'])
    if name == 'NIG':
        return stats.norminvgauss.ppf(q, p['a'], p['b'], p['loc'], p['scale'])
    if name == 'Stable':
        return p['delta0'] + p['gamma'] * StableGrid(p['alpha'], p['beta']).ppf(q)
    raise ValueError(name)


def var_es(name, p, a_var=ALPHA_VAR, a_es=ALPHA_ES, m=4000):
    """VaR_a = -q_a and ES_a = -(1/a) int_0^a q_u du (midpoint rule on a fine grid of u, refined near 0)."""
    var = -float(np.ravel(ppf(name, a_var, p))[0])
    u = a_es * np.concatenate([np.logspace(-6, -2, m // 2, endpoint=False), np.linspace(0.01, 1, m // 2 + 1)])
    mid, du = 0.5 * (u[1:] + u[:-1]), np.diff(u)
    qs = ppf(name, mid, p)
    es = -float(np.sum(qs * du) + np.ravel(ppf(name, u[0] / 2, p))[0] * u[0]) / a_es
    return var, es


# =============================================================================
# CHARTS: THE CANDIDATES
# =============================================================================
def standard_candidates():
    """The candidate laws with mean 0 and variance 1 (the stable law: alpha = 1.7 with the same interquartile range
    as the Normal law, since its variance is infinite)."""
    t5 = {'nu': 4.0, 'loc': 0.0, 'scale': np.sqrt(2 / 4)}
    sk = {'mu': 0.0, 'sigma': 1.0, 'nu': 4.0, 'lam': -0.25}
    from scipy.special import gamma as G
    b = 1.2
    ged = {'beta': b, 'loc': 0.0, 'scale': np.sqrt(G(1 / b) / G(3 / b))}
    mix = {'w': [0.85, 0.15], 'mu': [0.0, 0.0], 'sd': [np.sqrt(0.55), np.sqrt((1 - 0.85 * 0.55) / 0.15)]}
    a_, b_ = 1.0, -0.2
    g = np.sqrt(a_ ** 2 - b_ ** 2)
    sc = 1 / np.sqrt(a_ ** 2 / g ** 3)                              # NIG variance = scale^2 a^2 / gamma^3
    nig = {'a': a_, 'b': b_, 'loc': -sc * b_ / g, 'scale': sc}
    G17 = StableGrid(1.7, 0.0)
    gam = stats.norm.ppf(0.75) / G17.ppf(0.75)
    sta = {'alpha': 1.7, 'beta': 0.0, 'gamma': float(gam), 'delta0': 0.0}
    return {'Normal': {'mu': 0.0, 'sigma': 1.0}, 'Student-t': t5, 'Skewed-t': sk, 'GED': ged, 'Mixture': mix,
            'NIG': nig, 'Stable': sta}


def fig_candidates(save=True):
    """Densities of the seven candidates (variance 1; stable with the Normal interquartile range), linear and log scale."""
    P = standard_candidates()
    x = np.linspace(-7, 7, 1401)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    out = {}
    lab = {'Normal': 'Normal', 'Student-t': 'Student-t (nu = 4)', 'Skewed-t': 'Skewed-t (nu = 4, lambda = -0.25)',
           'GED': 'GED (beta = 1.2)', 'Mixture': 'Normal mixture (85% / 15%)', 'NIG': 'NIG', 'Stable': 'Stable (alpha = 1.7)'}
    for m in MODELS:
        f = np.exp(logpdf(m, x, P[m]))
        axes[0].plot(x, f, color=MODEL_COL[m], lw=1.4, label=lab[m])
        axes[1].semilogy(x, f, color=MODEL_COL[m], lw=1.4, label='_')
        out[m] = {'p_below_m4': float(cdf(m, np.array([-4.0]), P[m])[0])}
    axes[0].set_xlim(-4, 4)
    axes[0].set_title('Density')
    axes[1].set_ylim(1e-7, 1)
    axes[1].set_title('Density, log scale')
    for ax in axes:
        ax.set_xlabel('x (standardised return)')
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_candidates')
    return out


# =============================================================================
# SUMMARY STATISTICS (port of the Quantlet SFESumm)
# =============================================================================
def monthly_first_day(name='dax', start='2004-05-07', end=None):
    """Prices on the first trading day of each month and the monthly log returns between them (SFESumm)."""
    p = load_close(name, start, end) if end else load_close(name, start)
    first = p.groupby(p.index.to_period('M')).head(1)
    first = first.iloc[1:]                       # the first month of the file is incomplete (as in SFESumm)
    return np.log(first).diff().dropna()


def summary_stats(r, per_year):
    """Minimum, maximum, mean, median, standard deviation, annual volatility, skewness and kurtosis (SFESumm)."""
    r = np.asarray(r, float)
    return {'n': len(r), 'min': float(r.min()), 'max': float(r.max()), 'mean': float(r.mean()), 'median': float(np.median(r)),
            'sd': float(r.std(ddof=1)), 'ann_vol': float(r.std(ddof=1) * np.sqrt(per_year)),
            'skew': float(stats.skew(r)), 'kurt': float(stats.kurtosis(r, fisher=False)),
            'jb': float(stats.jarque_bera(r).statistic), 'jb_p': float(stats.jarque_bera(r).pvalue)}


def fig_dax_monthly(save=True):
    """DAX monthly log returns (first trading day of each month): the SFESumm period July 2004 - May 2014 and the
    extension to 2026."""
    r = monthly_first_day('dax', '2004-05-07')
    book = r.loc['2004-07':'2014-05']
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(r.index, 100 * r.values, color='#17A2B8', lw=1.2, label='DAX monthly log return, 2004-2026')
    ax.plot(book.index, 100 * book.values, color='#1A3A6E', lw=1.5, label='SFESumm period, July 2004 - May 2014')
    ax.axhline(0, color='#2E7D32', lw=1.0, ls='--', label='zero')
    ax.set_ylabel('Log return (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_dax_monthly')
    return {'book': summary_stats(book.values, 12), 'full': summary_stats(r.values, 12),
            'book_first': book.index[0].date().isoformat(), 'book_last': book.index[-1].date().isoformat(),
            'full_last': r.index[-1].date().isoformat()}


def daily_summary(names=ASSETS + STOCKS):
    out = {}
    for k in names:
        r = returns(k)
        ppy = len(r) / ((r.index[-1] - r.index[0]).days / 365.25)
        out[k] = dict(summary_stats(r.values, ppy), first=r.index[0].date().isoformat(), last=r.index[-1].date().isoformat())
    return out


def tlv_check():
    """Banca Transilvania: the two erroneous days and their effect on the fitted Student-t and on the ranking."""
    raw = log_returns('tlv', STOCK_START)
    clean = returns('tlv')
    out = {'r_bad': [float(raw.loc[d]) for d in BAD_DAYS['tlv']]}
    for lab, r in (('raw', raw), ('clean', clean)):
        x = r.values
        res = {m: fit_model(m, x) for m in ['Normal', 'Student-t', 'Skewed-t', 'GED']}
        aic = {m: 2 * K[m] - 2 * res[m]['loglik'] for m in res}
        out[lab] = {'n': len(x), 'kurt': float(stats.kurtosis(x, fisher=False)), 'nu': res['Student-t']['nu'],
                    'lam': res['Skewed-t']['lam'], 'aic': aic, 'best': min(aic, key=aic.get),
                    'var1_emp': float(-np.quantile(x, 0.01)), 'var1_t': -float(np.ravel(ppf('Student-t', 0.01, res['Student-t']))[0])}
    return out


# =============================================================================
# FITS, INFORMATION CRITERIA, TESTS
# =============================================================================
def fit_all(names=ASSETS + STOCKS, models=MODELS):
    """All candidate fits for each series: parameters, log-likelihood, AIC, BIC, Akaike weights."""
    out = {}
    for k in names:
        r = returns(k)
        x = r.values
        res = {}
        for m in models:
            p = fit_model(m, x)
            ll = p['loglik']
            res[m] = {'params': p, 'loglik': ll, 'k': K[m], 'aic': 2 * K[m] - 2 * ll, 'bic': K[m] * np.log(len(x)) - 2 * ll}
        amin = min(v['aic'] for v in res.values())
        bmin = min(v['bic'] for v in res.values())
        wsum = sum(np.exp(-0.5 * (v['aic'] - amin)) for v in res.values())
        for v in res.values():
            v['daic'] = v['aic'] - amin
            v['dbic'] = v['bic'] - bmin
            v['weight'] = float(np.exp(-0.5 * v['daic']) / wsum)
        out[k] = {'n': len(x), 'first': r.index[0].date().isoformat(), 'last': r.index[-1].date().isoformat(),
                  'models': res, 'best_aic': min(res, key=lambda m: res[m]['aic']), 'best_bic': min(res, key=lambda m: res[m]['bic'])}
        print('  fitted', k, out[k]['best_aic'], out[k]['best_bic'])
    return out


def fits_table(fits):
    rows = []
    for k, f in fits.items():
        for m, v in f['models'].items():
            rows.append({'series': k, 'model': m, 'n': f['n'], 'k': v['k'], 'loglik': v['loglik'], 'aic': v['aic'],
                         'bic': v['bic'], 'daic': v['daic'], 'dbic': v['dbic'], 'weight': v['weight'],
                         **{f'p_{a}': (b if np.isscalar(b) else str(np.round(b, 4).tolist())) for a, b in v['params'].items() if a != 'loglik'}})
    return pd.DataFrame(rows)


def lr_test(ll0, ll1, df, boundary=False):
    """Likelihood-ratio test of a restricted model (ll0) inside a larger one (ll1): LR = 2 (ll1 - ll0) ~ chi2(df);
    on the boundary of the parameter space (Normal inside Student-t, 1/nu = 0) the null law is the 50:50 mixture of
    chi2(0) and chi2(1) (Self and Liang, 1987)."""
    lr = max(2 * (ll1 - ll0), 0.0)
    p = stats.chi2.sf(lr, df)
    return {'lr': float(lr), 'p': float(0.5 * p if boundary else p)}


def vuong(name1, p1, name2, p2, x):
    """Vuong (1989) test of two non-nested models: V = sqrt(n) mean(m) / sd(m), m_i = log f1(x_i) - log f2(x_i);
    V > 1.96: model 1 closer to the truth (Kullback-Leibler); V < -1.96: model 2; otherwise no decision."""
    m = logpdf(name1, x, p1) - logpdf(name2, x, p2)
    v = np.sqrt(len(m)) * m.mean() / m.std(ddof=1)
    return {'v': float(v), 'p': float(2 * stats.norm.sf(abs(v)))}


def tests_all(fits):
    out = {}
    for k, f in fits.items():
        M = f['models']
        x = returns(k).values
        out[k] = {'t_vs_skt': lr_test(M['Student-t']['loglik'], M['Skewed-t']['loglik'], 1),
                  'n_vs_ged': lr_test(M['Normal']['loglik'], M['GED']['loglik'], 1),
                  'n_vs_t': lr_test(M['Normal']['loglik'], M['Student-t']['loglik'], 1, boundary=True),
                  'vu_t_ged': vuong('Student-t', M['Student-t']['params'], 'GED', M['GED']['params'], x),
                  'vu_skt_nig': vuong('Skewed-t', M['Skewed-t']['params'], 'NIG', M['NIG']['params'], x)}
        if 'Stable' in M:
            out[k]['vu_t_stable'] = vuong('Student-t', M['Student-t']['params'], 'Stable', M['Stable']['params'], x)
    return out


def profile_nu(k='sp500', grid=None, save=True):
    """Profile log-likelihood of the Student-t degrees of freedom: for each nu, maximise over location and scale;
    95% interval: all nu with 2 (l_max - l(nu)) <= chi2(1) 95% quantile = 3.84."""
    x = returns(k).values
    grid = grid if grid is not None else np.round(np.arange(2.2, 6.01, 0.1), 2)
    ll = []
    for nu in grid:
        _, loc, sc = stats.t.fit(x, fdf=nu)
        ll.append(np.sum(stats.t.logpdf(x, nu, loc, sc)))
    ll = np.array(ll)
    f = t_fit(x)
    lmax = f['loglik']
    cut = lmax - stats.chi2.ppf(0.95, 1) / 2
    inside = grid[ll >= cut]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(grid, ll - lmax, color='#1A3A6E', lw=1.6, label='profile log-likelihood l(nu) - l(nu_hat)')
    ax.axhline(cut - lmax, color='#CD0000', ls='--', lw=1.0, label='cut-off -3.84 / 2 (95% interval)')
    ax.axvline(f['nu'], color='#2E7D32', ls=':', lw=1.2, label=f'ML estimate nu_hat = {f["nu"]:.2f}')
    ax.set_ylim(max((ll - lmax).min(), -40), 2)
    ax.set_xlabel('Degrees of freedom nu')
    ax.set_ylabel('Log-likelihood difference')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_profile_nu')
    return {'nu': f['nu'], 'lo': float(inside.min()), 'hi': float(inside.max()), 'loglik': lmax,
            'll_3': float(np.interp(3.0, grid, ll) - lmax), 'll_5': float(np.interp(5.0, grid, ll) - lmax)}


def fig_delta_aic(fits, save=True):
    """Delta AIC of each model relative to the best model, per series (heat map, log colour scale)."""
    from matplotlib.colors import LogNorm
    names = list(fits)
    D = np.array([[fits[k]['models'][m]['daic'] for m in MODELS] for k in names])
    fig, ax = plt.subplots(figsize=(10, 4.0))
    im = ax.imshow(D + 1, cmap='Blues', norm=LogNorm(vmin=1, vmax=max(D.max(), 10) + 1), aspect='auto')
    for i in range(len(names)):
        for j in range(len(MODELS)):
            v = D[i, j]
            txt = '0' if v < 0.5 else f'{v:,.0f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10.5,
                    color='white' if v > 300 else 'black', fontweight='bold' if v < 0.5 else 'normal')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(['Normal mixture' if m == 'Mixture' else m for m in MODELS])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[k] for k in names])
    ax.set_title('Delta AIC = AIC - min AIC (0 = best model of the series)')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Delta AIC + 1 (log scale)')
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_delta_aic')


# =============================================================================
# GOODNESS-OF-FIT TESTS
# =============================================================================
def edf_stats(u):
    """Kolmogorov-Smirnov D, Cramer-von Mises W2 and Anderson-Darling A2 from u_i = F(x_i) (F fully specified)."""
    u = np.clip(np.sort(np.asarray(u, float)), 1e-12, 1 - 1e-12)
    n = len(u)
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    w2 = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
    a2 = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
    return {'ks': float(d), 'cvm': float(w2), 'ad': float(a2)}


def gof_bootstrap(name, x, B=199, seed=SEED):
    """EDF statistics of a fitted model and parametric bootstrap p-values: simulate from the fitted model, refit,
    recompute (Stute, Gonzalez Manteiga and Presedo Quindimil, 1993). The naive KS p-value treats the fitted
    parameters as known."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    p = fit_model(name, x)
    obs = edf_stats(cdf(name, x, p))
    naive = float(stats.kstwo.sf(obs['ks'], len(x)))
    sims = {s: [] for s in obs}
    for _ in range(B):
        y = ppf(name, rng.uniform(size=len(x)), p) if name != 'Normal' else rng.normal(p['mu'], p['sigma'], len(x))
        q = fit_model(name, y)
        for s, v in edf_stats(cdf(name, y, q)).items():
            sims[s].append(v)
    pv = {s: float((1 + np.sum(np.array(sims[s]) >= obs[s])) / (B + 1)) for s in obs}
    return {'stat': obs, 'p_boot': pv, 'p_naive_ks': naive, 'B': B,
            'crit_boot': {s: float(np.quantile(sims[s], 0.95)) for s in obs}}


def gof_all(fits, names=ASSETS + STOCKS):
    """EDF statistics (no p-values) for every model and series."""
    out = {}
    for k in names:
        x = returns(k).values
        out[k] = {m: edf_stats(cdf(m, x, fits[k]['models'][m]['params'])) for m in fits[k]['models']}
    return out


def lilliefors_crit(n, reps=4000, seed=SEED):
    """95% critical values of KS, CvM and AD for the Normal law with estimated mean and variance (simulation)."""
    rng = np.random.default_rng(seed)
    S = {'ks': [], 'cvm': [], 'ad': []}
    for _ in range(reps):
        y = rng.standard_normal(n)
        for s, v in edf_stats(stats.norm.cdf(y, y.mean(), y.std())).items():
            S[s].append(v)
    return {s: float(np.quantile(v, 0.95)) for s, v in S.items()}


def gof_power(ns=(50, 100, 250, 500, 1000), reps=1000, seed=SEED, save=True):
    """Power of KS, CvM and AD tests of Normality with estimated parameters (5% level, simulated critical values)
    against (left) Student-t data with nu = 5 and (right) a Normal law with 2% of days from a Normal law with
    four times the standard deviation (differences only in the tails)."""
    rng = np.random.default_rng(seed)
    alts = {'t5': lambda n: rng.standard_t(5, n),
            'contam': lambda n: np.where(rng.uniform(size=n) < 0.02, 4.0, 1.0) * rng.standard_normal(n)}
    out = {}
    for a, gen in alts.items():
        out[a] = {}
        for n in ns:
            crit = lilliefors_crit(n, reps=2000, seed=seed + n)
            rej = {'ks': 0, 'cvm': 0, 'ad': 0}
            for _ in range(reps):
                y = gen(n)
                for s, v in edf_stats(stats.norm.cdf(y, y.mean(), y.std())).items():
                    rej[s] += v > crit[s]
            out[a][n] = {s: rej[s] / reps for s in rej}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    lab = {'ks': 'Kolmogorov-Smirnov', 'cvm': 'Cramer-von Mises', 'ad': 'Anderson-Darling'}
    col = {'ks': '#CD0000', 'cvm': '#E67E22', 'ad': '#1A3A6E'}
    for ax, a, title in zip(axes, ['t5', 'contam'], ['Student-t data, nu = 5', 'Normal data, 2% of days with 4x volatility']):
        for s in ['ks', 'cvm', 'ad']:
            ax.plot(ns, [out[a][n][s] for n in ns], marker='o', color=col[s], label=lab[s] if a == 't5' else '_')
        ax.axhline(0.05, color='#2E7D32', ls='--', lw=1.0, label='5% level' if a == 't5' else '_')
        ax.set_xscale('log')
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel('Sample size n (log scale)')
        ax.set_title(title)
    axes[0].set_ylabel('Rejection rate of Normality')
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_gof_power')
    return {a: {str(n): v for n, v in d.items()} for a, d in out.items()}


def fig_ks_tail(k='sp500', save=True):
    """Where do the KS and AD statistics look? F_n(x) - F(x) for the Normal fit, and the same difference divided by
    sqrt(F(1 - F) / n), the weighting behind the Anderson-Darling statistic."""
    x = np.sort(returns(k).values)
    n = len(x)
    mu, sd = x.mean(), x.std()
    F = stats.norm.cdf(x, mu, sd)
    Fn = np.arange(1, n + 1) / n
    d = Fn - F
    Fc = np.clip(F, 1e-8, 1 - 1e-8)                 # model probabilities below 1e-8 are treated as 1e-8
    z = np.sqrt(n) * d / np.sqrt(Fc * (1 - Fc))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(x, d, color='#1A3A6E', lw=1.2, label='F_n(x) - F(x)')
    i = np.argmax(np.abs(d))
    axes[0].plot(x[i], d[i], 'o', ms=8, color='#CD0000', label=f'KS maximum at x = {x[i]:.2f}%')
    axes[1].plot(x, z, color='#8E44AD', lw=1.0, label='sqrt(n) (F_n - F) / sqrt(F (1 - F))')
    j = np.argmax(np.abs(z))
    axes[1].plot(x[j], z[j], 'o', ms=8, color='#CD0000', label=f'largest weighted gap at x = {x[j]:.2f}%')
    axes[0].set_xlim(-6, 6)
    axes[1].set_xlim(x[0] - 0.5, x[-1] + 0.5)
    axes[1].set_yscale('symlog', linthresh=10)
    axes[1].set_ylabel('Weighted gap (symmetric log scale)')
    axes[0].set_ylabel('Gap')
    for ax, t in zip(axes, ['Unweighted gap (Kolmogorov-Smirnov)', 'Tail-weighted gap (Anderson-Darling weight)']):
        ax.axhline(0, color='#2E7D32', lw=0.9, ls='--', label='_')
        ax.set_xlabel('Daily log return x (%)')
        ax.set_title(t)
        st.legend_outside_bottom(ax, ncol=1, y=-0.18)
    fig.tight_layout()
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_ks_tail')
    return {'ks_x': float(x[i]), 'ks_d': float(d[i]), 'ks_F': float(F[i]), 'w_x': float(x[j]), 'w_z': float(z[j]),
            'n': n, 'mu': float(mu), 'sd': float(sd)}


# =============================================================================
# PP AND QQ PLOTS
# =============================================================================
PANEL = ['Normal', 'Student-t', 'Skewed-t', 'Stable']


def fig_pp_qq(fits, k='sp500', save=True):
    """PP plots (model probability vs empirical probability) and QQ plots (empirical vs model quantiles) of the
    Normal, Student-t, skewed-t and stable fits."""
    x = np.sort(returns(k).values)
    n = len(x)
    pe = (np.arange(1, n + 1) - 0.5) / n
    sel = np.unique(np.concatenate([np.linspace(0, n - 1, 400).astype(int), np.arange(30), np.arange(n - 30, n)]))
    out = {}
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharex=True, sharey=True)
    for ax, m in zip(axes, PANEL):
        p = fits[k]['models'][m]['params']
        u = cdf(m, x, p)
        ax.plot(pe[sel], u[sel], '.', ms=3, color=MODEL_COL[m], label=m)
        ax.plot([0, 1], [0, 1], color='black', lw=0.8, ls='--', label='45-degree line' if m == PANEL[-1] else '_')
        ax.set_title(m)
        ax.set_xlabel('Empirical probability')
        out[m] = {'pp_maxdev': float(np.max(np.abs(u - pe)))}
    axes[0].set_ylabel('Model probability F(x)')
    fig.legend(*zip(*[(h, l) for ax in axes for h, l in zip(*ax.get_legend_handles_labels())]), loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, markerscale=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_pp')
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, m in zip(axes, PANEL):
        p = fits[k]['models'][m]['params']
        q = ppf(m, pe[sel], p)
        ax.plot(q, x[sel], '.', ms=3.5, color=MODEL_COL[m], label=m)
        lim = [x[0], x[-1]]
        ax.plot(lim, lim, color='black', lw=0.8, ls='--', label='45-degree line' if m == PANEL[-1] else '_')
        ax.set_xlim(min(q.min(), x[0]) * 1.05, max(q.max(), x[-1]) * 1.05)
        ax.set_title(m)
        ax.set_xlabel('Model quantile (%)')
        out[m]['q0001'] = float(ppf(m, 0.5 / n, p))
    axes[0].set_ylabel('Empirical quantile (%)')
    fig.legend(*zip(*[(h, l) for ax in axes for h, l in zip(*ax.get_legend_handles_labels())]), loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, markerscale=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_qq')
    out['x_min'] = float(x[0])
    out['x_max'] = float(x[-1])
    out['n'] = n
    return out


def fig_qq_assets(fits, names=('bet', 'btc', 'tlv'), models=('Normal', 'Student-t', 'Skewed-t'), save=True):
    """QQ plots of three more series against the Normal, Student-t and skewed-t fits (one panel per series)."""
    fig, axes = plt.subplots(1, len(names), figsize=(11, 3.6))
    for ax, k in zip(axes, names):
        x = np.sort(returns(k).values)
        n = len(x)
        pe = (np.arange(1, n + 1) - 0.5) / n
        sel = np.unique(np.concatenate([np.linspace(0, n - 1, 300).astype(int), np.arange(25), np.arange(n - 25, n)]))
        for m in models:
            q = ppf(m, pe[sel], fits[k]['models'][m]['params'])
            ax.plot(q, x[sel], '.', ms=3.2, color=MODEL_COL[m], label=m if k == names[0] else '_')
        ax.plot([x[0], x[-1]], [x[0], x[-1]], color='black', lw=0.8, ls='--', label='45-degree line' if k == names[0] else '_')
        ax.set_title(LABELS[k])
        ax.set_xlabel('Model quantile (%)')
    axes[0].set_ylabel('Empirical quantile (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, markerscale=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_qq_assets')


# =============================================================================
# TAILS: VaR 1%, ES 2.5%, EXCEEDANCES
# =============================================================================
def kupiec(x, n, p=ALPHA_VAR):
    """Kupiec (1995) unconditional coverage test: LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)] ~ chi2(1)."""
    phat = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (l0 - l1)
    return {'x': int(x), 'n': int(n), 'rate': float(phat), 'lr': float(lr), 'p': float(stats.chi2.sf(lr, 1))}


def tail_table(fits, names=ASSETS + STOCKS):
    """VaR 1% and ES 2.5% of the data and of each model; in-sample exceedances of each model's VaR 1%."""
    out = {}
    for k in names:
        x = returns(k).values
        q = np.quantile(x, ALPHA_VAR)
        qe = np.quantile(x, ALPHA_ES)
        row = {'emp': {'var1': float(-q), 'es25': float(-x[x <= qe].mean())}}
        for m, v in fits[k]['models'].items():
            var, es = var_es(m, v['params'])
            row[m] = {'var1': var, 'es25': es, 'kupiec': kupiec(int(np.sum(x < -var)), len(x))}
        out[k] = row
    return out


def fig_left_tail(fits, k='sp500', models=('Normal', 'Student-t', 'Skewed-t', 'NIG', 'Stable'), save=True):
    """The left tail on log-log axes: share of days with a loss above x, against the same probability under each model."""
    x = returns(k).values
    L = np.sort(-x)[::-1]
    n = len(x)
    emp = np.arange(1, n + 1) / n
    sel = L > 0.5
    grid = np.logspace(np.log10(0.5), np.log10(L.max() * 1.6), 80)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.loglog(L[sel], emp[sel], '.', ms=4, color=MODEL_COL['Data'], label='data')
    for m in models:
        p = fits[k]['models'][m]['params']
        ax.loglog(grid, np.clip(cdf(m, -grid, p), 1e-9, None), color=MODEL_COL[m], lw=1.5, label=m)
    ax.axhline(ALPHA_VAR, color='#2E7D32', lw=0.9, ls=':', label='_')
    ax.text(grid[1], ALPHA_VAR * 1.25, '1% (VaR 1%)', color='#2E7D32', fontsize=10.5)
    ax.set_ylim(0.5 / n, 0.6)
    ax.set_xlabel('Daily loss x (%, log scale)')
    ax.set_ylabel('P(loss > x) (log scale)')
    st.legend_outside_bottom(ax, ncol=6, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_left_tail')


def fig_var_models(tails, names=ASSETS + STOCKS, models=('Normal', 'Student-t', 'Skewed-t', 'GED', 'NIG', 'Stable'), save=True):
    """VaR 1% of each model divided by the empirical VaR 1% (1 = exact in sample)."""
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    w = 0.8 / len(models)
    xs = np.arange(len(names))
    for j, m in enumerate(models):
        ax.bar(xs + (j - (len(models) - 1) / 2) * w, [tails[k][m]['var1'] / tails[k]['emp']['var1'] for k in names],
               width=w, color=MODEL_COL[m], label=m)
    ax.axhline(1, color='black', lw=0.9, ls='--', label='empirical VaR 1%')
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[k] for k in names])
    ax.set_ylabel('Model VaR 1% / empirical VaR 1%')
    ax.set_ylim(0.6, None)
    st.legend_outside_bottom(ax, ncol=7, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_var_models')


# =============================================================================
# OUT-OF-SAMPLE VALIDATION, OVERFITTING, ROLLING VaR
# =============================================================================
def pinball(r, q, a=ALPHA_VAR):
    """Quantile (pinball) loss of a VaR forecast q (a quantile of the return): mean of (a - 1{r < q})(r - q)."""
    r = np.asarray(r, float)
    return float(np.mean((a - (r < q)) * (r - q)))


def out_of_sample(names=ASSETS, models=MODELS, split=SPLIT):
    """Fit each model until `split`; on the later days: mean log score, VaR 1% exceedances (Kupiec) and pinball loss."""
    out = {}
    for k in names:
        r = returns(k)
        est, test = r.loc[:split].values, r.loc[split:].iloc[1:].values
        row = {'n_est': len(est), 'n_test': len(test), 'emp_var1_est': float(-np.quantile(est, ALPHA_VAR))}
        for m in models:
            p = fit_model(m, est)
            q = float(np.ravel(ppf(m, ALPHA_VAR, p))[0])
            ls = logpdf(m, test, p)
            row[m] = {'logscore': float(np.mean(ls)), 'loglik_in': p['loglik'] / len(est), 'aic': 2 * K[m] - 2 * p['loglik'],
                      'var1': -q, 'kupiec': kupiec(int(np.sum(test < q)), len(test)), 'pinball': pinball(test, q)}
        qh = np.quantile(est, ALPHA_VAR)
        row['Historical'] = {'var1': float(-qh), 'kupiec': kupiec(int(np.sum(test < qh)), len(test)), 'pinball': pinball(test, qh)}
        aics = {m: row[m]['aic'] for m in models}
        row['best_aic_est'] = min(aics, key=aics.get)
        row['best_logscore'] = max(models, key=lambda m: row[m]['logscore'])
        row['best_pinball'] = min(list(models) + ['Historical'], key=lambda m: row[m]['pinball'])
        out[k] = row
        print('  oos', k, row['best_aic_est'], row['best_logscore'], row['best_pinball'])
    return out


def fig_oos(oos, names=ASSETS, models=('Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG', 'Stable'), save=True):
    """Left: out-of-sample mean log score minus that of the Normal model. Right: VaR 1% exceedance rate on the test days."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    w = 0.8 / len(models)
    xs = np.arange(len(names))
    for j, m in enumerate(models):
        off = (j - (len(models) - 1) / 2) * w
        axes[0].bar(xs + off, [oos[k][m]['logscore'] - oos[k]['Normal']['logscore'] for k in names], width=w, color=MODEL_COL[m], label=m)
        axes[1].bar(xs + off, [100 * oos[k][m]['kupiec']['rate'] for k in names], width=w, color=MODEL_COL[m], label='_')
    axes[1].axhline(1, color='black', lw=0.9, ls='--', label='target 1%')
    axes[0].set_title('Out-of-sample log score minus Normal')
    axes[1].set_title('VaR 1% exceedances on the test days (%)')
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[k] for k in names])
    st.fig_legend_bottom(fig, ncol=8, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_oos')


def overfit_mixtures(k='sp500', est=('2017-01-01', '2018-12-31'), test_end=None, kmax=6, save=True):
    """Overfitting: Normal mixtures with 1-6 components fitted on two years of returns; in-sample log-likelihood,
    AIC, BIC and the out-of-sample mean log score on the following years."""
    r = returns(k)
    x = r.loc[est[0]:est[1]].values
    y = r.loc[est[1]:].iloc[1:].values if test_end is None else r.loc[est[1]:test_end].iloc[1:].values
    out = {}
    for c in range(1, kmax + 1):
        p = mixture_em(x, c, starts=12, seed=SEED + c)
        kk = 3 * c - 1
        out[c] = {'k': kk, 'loglik': p['loglik'], 'aic': 2 * kk - 2 * p['loglik'], 'bic': kk * np.log(len(x)) - 2 * p['loglik'],
                  'in': p['loglik'] / len(x), 'out': float(np.mean(mixture_logpdf(y, p['w'], p['mu'], p['sd'])))}
    cs = list(out)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(cs, [out[c]['in'] for c in cs], 'o-', color='#1A3A6E', label='in sample (estimation years)')
    axes[0].plot(cs, [out[c]['out'] for c in cs], 's-', color='#CD0000', label='out of sample (later years)')
    axes[0].set_title('Mean log-likelihood per day')
    axes[1].plot(cs, [out[c]['aic'] - min(out[d]['aic'] for d in cs) for c in cs], 'o-', color='#2E7D32', label='Delta AIC')
    axes[1].plot(cs, [out[c]['bic'] - min(out[d]['bic'] for d in cs) for c in cs], 's-', color='#8E44AD', label='Delta BIC')
    axes[1].set_title('Information criteria (0 = selected)')
    for ax in axes:
        ax.set_xlabel('Number of Normal components')
        ax.set_xticks(cs)
        st.legend_outside_bottom(ax, ncol=2, y=-0.18)
    fig.tight_layout()
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_overfit')
    return {'n_in': len(x), 'n_out': len(y), 'est': list(est),
            'res': {str(c): v for c, v in out.items()},
            'best_in': max(cs, key=lambda c: out[c]['in']), 'best_out': max(cs, key=lambda c: out[c]['out']),
            'best_aic': min(cs, key=lambda c: out[c]['aic']), 'best_bic': min(cs, key=lambda c: out[c]['bic'])}


def rolling_var(k='sp500', window=1000, step=20, models=('Normal', 'Student-t', 'Skewed-t')):
    """Rolling VaR 1%: each model refitted every `step` days on the previous `window` days; historical simulation
    (empirical 1% quantile of the window) as a benchmark."""
    r = returns(k)
    x = r.values
    idx = r.index
    V = {m: np.full(len(x), np.nan) for m in list(models) + ['Historical']}
    for s in range(window, len(x), step):
        w = x[s - window:s]
        for m in models:
            V[m][s:s + step] = np.ravel(ppf(m, ALPHA_VAR, fit_model(m, w)))[0]
        V['Historical'][s:s + step] = np.quantile(w, ALPHA_VAR)
    df = pd.DataFrame(V, index=idx)
    df['r'] = x
    df = df.iloc[window:]
    res = {m: kupiec(int(np.sum(df['r'] < df[m])), len(df)) for m in V}
    for m in V:
        e = (df['r'] < df[m]).astype(int)
        res[m]['max_year'] = int(e.groupby(df.index.year).sum().max())
        res[m]['max_year_y'] = int(e.groupby(df.index.year).sum().idxmax())
    res['first'] = df.index[0].date().isoformat()
    return df, res


def fig_rolling_var(df, k='sp500', since='2018-01-01', save=True):
    d = df.loc[since:]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.plot(d.index, d['r'], color='#17A2B8', lw=0.6, label=f'{LABELS[k]} daily log return (%)')
    for m, c in [('Normal', MODEL_COL['Normal']), ('Student-t', MODEL_COL['Student-t']), ('Historical', '#8E44AD')]:
        ax.plot(d.index, d[m], color=c, lw=1.3, label=f'VaR 1% line, {m}' if m != 'Historical' else 'VaR 1% line, historical simulation')
    e = d['r'] < d['Student-t']
    ax.plot(d.index[e], d['r'][e], 'v', ms=5, color='black', label='exceedance of the Student-t VaR 1%')
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_rolling_var')


def var_rma_ema(y, h=250, lam=0.96, a=ALPHA_VAR):
    """VaR of the Quantlet SFEVaRqqplot: Normal VaR with the volatility from a rectangular moving average (RMA) of the
    previous h squared returns, or an exponential moving average (EMA, weight lam^j on the return j + 1 days back,
    normalised by 1 - lam)."""
    y = np.asarray(y, float)
    n = len(y)
    c = np.concatenate([[0.0], np.cumsum(y ** 2)])
    rma = np.sqrt((c[h:n] - c[:n - h]) / h)                       # days h+1..n: mean of the previous h squares
    wts = lam ** np.arange(h)[::-1]
    ema = np.sqrt((1 - lam) * np.array([np.sum(wts * y[j - h:j] ** 2) for j in range(h, n)]))
    z = stats.norm.ppf(a)
    return z * rma, z * ema


def fig_var_qqplot(k='dax', h=250, lam=0.96, save=True):
    """VaR reliability plot (SFEVaRqqplot): QQ plot of the return divided by the VaR forecast, L/VaR, against the
    Normal quantiles, for the RMA and EMA volatility models; a straight line means a well-specified VaR model."""
    r = returns(k)
    y = r.values / 100
    v_rma, v_ema = var_rma_ema(y, h, lam)
    p = y[h:]
    out = {'n': len(p), 'first': r.index[h].date().isoformat()}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, v, name, c in [(axes[0], v_rma, 'RMA, h = 250', '#1A3A6E'), (axes[1], v_ema, 'EMA, lambda = 0.96', '#E67E22')]:
        ratio = np.sort(p / v)
        qn = stats.norm.ppf((np.arange(1, len(ratio) + 1) - 0.5) / len(ratio))
        ax.plot(qn, ratio, '.', ms=2.5, color=c, label=f'L/VaR, {name}')
        b = np.quantile(ratio, [0.25, 0.75])
        qb = stats.norm.ppf([0.25, 0.75])
        sl = (b[1] - b[0]) / (qb[1] - qb[0])
        ax.plot(qn, b[0] + sl * (qn - qb[0]), color='black', lw=0.9, ls='--', label='line through the quartiles')
        ax.set_title(name)
        ax.set_xlabel('Normal quantiles')
        st.legend_outside_bottom(ax, ncol=1, y=-0.18)
        key = 'rma' if 'RMA' in name else 'ema'
        ex = np.sum(p < v)
        out[key] = dict(kupiec(int(ex), len(p)), kurt_std=float(stats.kurtosis(p / v * stats.norm.ppf(ALPHA_VAR), fisher=False)),
                        min_ratio=float(ratio[0]), max_ratio=float(ratio[-1]))
    axes[0].set_ylabel('L/VaR quantiles')
    fig.tight_layout()
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_var_qqplot')
    return out


# =============================================================================
# MODEL UNCERTAINTY
# =============================================================================
def block_bootstrap(x, block=BLOCK, rng=None):
    """Moving-block bootstrap resample of the same length (blocks of consecutive days keep volatility clusters)."""
    n = len(x)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([x[s:s + block] for s in starts])[:n]


def model_uncertainty(k='sp500', B=200, models=('Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG'), seed=SEED, save=True):
    """Moving-block bootstrap: share of resamples in which each model has the lowest AIC, and the bootstrap spread
    of VaR 1% under each model."""
    x = returns(k).values
    rng = np.random.default_rng(seed)
    wins = {m: 0 for m in models}
    var = {m: [] for m in models}
    for b in range(B):
        y = block_bootstrap(x, rng=rng)
        aic = {}
        for m in models:
            p = fit_model(m, y)
            aic[m] = 2 * K[m] - 2 * p['loglik']
            var[m].append(-float(np.ravel(ppf(m, ALPHA_VAR, p))[0]))
        wins[min(aic, key=aic.get)] += 1
        if (b + 1) % 50 == 0:
            print('   bootstrap', b + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    axes[0].bar(range(len(models)), [100 * wins[m] / B for m in models], color=[MODEL_COL[m] for m in models])
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(['Mixture' if m == 'Mixture' else m for m in models], rotation=20)
    axes[0].set_ylabel('% of resamples with the lowest AIC')
    axes[0].set_title('Which model wins?')
    bp = axes[1].boxplot([var[m] for m in models], patch_artist=True, widths=0.55)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COL[m])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
    for el in ('whiskers', 'caps', 'medians'):
        for line in bp[el]:
            line.set_color('black')
    for fl in bp['fliers']:
        fl.set_markeredgecolor('#CD0000')
    emp = -np.quantile(x, ALPHA_VAR)
    axes[1].axhline(emp, color='#CD0000', ls='--', lw=1.0, label=f'empirical VaR 1% = {emp:.2f}%')
    axes[1].set_xticks(range(1, len(models) + 1))
    axes[1].set_xticklabels(models, rotation=20)
    axes[1].set_ylabel('VaR 1% (%)')
    axes[1].set_title('Bootstrap spread of VaR 1%')
    st.legend_outside_bottom(axes[1], ncol=1, y=-0.25)
    fig.tight_layout()
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch6_model_uncertainty')
    return {'B': B, 'wins': {m: wins[m] / B for m in models},
            'var': {m: {'median': float(np.median(v)), 'q05': float(np.quantile(v, 0.05)), 'q95': float(np.quantile(v, 0.95))} for m, v in var.items()},
            'emp_var1': float(emp)}


def averaged_var(fits, tails, names=ASSETS + STOCKS):
    """VaR 1% averaged over the models with the Akaike weights, and the range of VaR 1% across the models."""
    out = {}
    for k in names:
        M = fits[k]['models']
        avg = sum(M[m]['weight'] * tails[k][m]['var1'] for m in M)
        vs = [tails[k][m]['var1'] for m in M if m != 'Normal']
        out[k] = {'avg': float(avg), 'min_heavy': float(min(vs)), 'max_heavy': float(max(vs)), 'normal': tails[k]['Normal']['var1'],
                  'emp': tails[k]['emp']['var1']}
    return out


# =============================================================================
# MAIN
# =============================================================================
def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


if __name__ == '__main__':
    st.apply()
    N = {}
    N['end'] = returns('sp500').index[-1].date().isoformat()
    print('candidates'); N['candidates'] = fig_candidates()
    print('summary'); N['dax_monthly'] = fig_dax_monthly(); N['daily'] = daily_summary()
    print('tlv'); N['tlv'] = tlv_check()
    print('fits'); fits = fit_all()
    fits_table(fits).to_csv(os.path.join(TABLE_DIR, 'ch6_fits.csv'), index=False)
    N['fits'] = {k: {'n': f['n'], 'first': f['first'], 'last': f['last'], 'best_aic': f['best_aic'], 'best_bic': f['best_bic'],
                     'models': f['models']} for k, f in fits.items()}
    N['tests'] = tests_all(fits)
    N['profile'] = profile_nu('sp500')
    fig_delta_aic(fits)
    print('gof'); N['gof'] = gof_all(fits)
    N['gof_boot'] = {m: gof_bootstrap(m, returns('sp500').values, B=199) for m in ['Normal', 'Student-t', 'Skewed-t']}
    N['lillie'] = {str(n): lilliefors_crit(n) for n in (5, 6, 100, 1000)}
    N['power'] = gof_power()
    N['ks_tail'] = fig_ks_tail('sp500')
    print('pp qq'); N['ppqq'] = fig_pp_qq(fits, 'sp500'); fig_qq_assets(fits)
    print('tails'); N['tails'] = tail_table(fits); fig_left_tail(fits, 'sp500'); fig_var_models(N['tails'])
    N['avg_var'] = averaged_var(fits, N['tails'])
    print('oos'); N['oos'] = out_of_sample(); fig_oos(N['oos'])
    N['overfit'] = overfit_mixtures('sp500')
    print('rolling'); df, N['rolling'] = rolling_var('sp500'); fig_rolling_var(df, 'sp500')
    _, N['rolling_bet'] = rolling_var('bet')
    N['var_qq'] = fig_var_qqplot('dax')
    print('uncertainty'); N['uncertainty'] = model_uncertainty('sp500')
    with open(os.path.join(TABLE_DIR, 'ch6_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1, default=str)
    print('done')
