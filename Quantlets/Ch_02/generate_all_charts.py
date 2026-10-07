"""
generate_all_charts.py -- charts and numbers of Chapter 2 (SFM): Classical distributions and stylised facts
=========================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * the Normal distribution: density, tail probabilities, how often k-sigma days should occur;
  * the lognormal distribution: densities, mean, median and mode;
  * the central limit theorem and the Normal approximation of the binomial distribution;
  * moments (skewness, excess kurtosis), the Jarque-Bera test, kernel density of DAX returns, QQ plots;
  * the Student-t distribution: densities, maximum likelihood fit, QQ plots against the fitted t;
  * the stylised facts of Cont (2001): heavy tails (k-sigma days), aggregational Gaussianity, absence of linear
    autocorrelation, volatility clustering, leverage effect, gain/loss asymmetry; summary table;
  * data check: two misplaced adjustment days of Banca Transilvania (May 2016) and their effect on the kurtosis;
  * tails of Bitcoin and the S&P 500 by calendar year (open question of the chapter).
Output: charts/sfm_ch2_*.pdf/.png, Quantlets/Ch_02/ch2_numbers.json, ch2_moments.csv
Based on the SFE Quantlets SFEDaxReturnDistribution, SFElognormal, SFEclt and SFENormalApprox1-4 (Franke, Haerdle
and Hafner, Statistics of Financial Markets, 5th ed., 2019), ported to Python.
Run:  python3 Quantlets/Ch_02/generate_all_charts.py
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
from sfm_data import LABELS, SERIES, load_close, log_returns, periods_per_year, read_market   # noqa: E402
import sfm_style as st                                                                         # noqa: E402

TABLE_DIR = HERE
START = '2010-01-01'                      # chapter sample: 2010-2026 (later series from their first day)
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'snp', 'brd', 'snn', 'sng']
INDICES = ['sp500', 'dax', 'bet', 'btc']
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F', 'snp': '#8E44AD',
          'brd': '#DC3545', 'snn': '#2E7D32', 'sng': '#E67E22', 'tlv': '#2E7D32'}
SHORT = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'snp': 'SNP', 'brd': 'BRD', 'snn': 'SNN',
         'sng': 'SNG', 'tlv': 'TLV'}
HORIZONS = [1, 2, 5, 10, 21, 63]
PAL = {'blue': '#1A3A6E', 'red': '#CD0000', 'green': '#2E7D32', 'amber': '#B5853F', 'purple': '#8E44AD',
       'teal': '#17A2B8', 'orange': '#E67E22'}          # trading days (Bitcoin: calendar days)


# =============================================================================
# DEFINITIONS (one function per statistic; the same code runs in the notebooks)
# =============================================================================
def returns(name, start=START, end=None):
    """Daily log returns in %, from `start` or from the first day of the series if that is later."""
    s0 = max(start, SERIES[name][4])
    return log_returns(name, s0) if end is None else log_returns(name, s0, end)


def moments(r):
    """Sample moments of returns: mean, standard deviation, skewness, excess kurtosis (population formulas, as in
    the Jarque-Bera test), their standard errors under the Normal distribution, extremes and the 1%/99% quantiles."""
    r = pd.Series(r).dropna()
    n = len(r)
    s, k = stats.skew(r), stats.kurtosis(r)                  # kurtosis(): excess kurtosis (Normal = 0)
    jb = n / 6 * (s ** 2 + k ** 2 / 4)
    out = {'n': n, 'mean': r.mean(), 'sd': r.std(), 'skew': s, 'exkurt': k, 'kurt': k + 3,
           'se_skew': np.sqrt(6 / n), 'se_kurt': np.sqrt(24 / n), 'jb': jb, 'jb_p': stats.chi2.sf(jb, 2),
           'min': r.min(), 'max': r.max(), 'q01': r.quantile(0.01), 'q99': r.quantile(0.99)}
    if isinstance(r.index, pd.DatetimeIndex):
        out.update(min_date=r.idxmin().date().isoformat(), max_date=r.idxmax().date().isoformat(),
                   first=r.index[0].date().isoformat(), last=r.index[-1].date().isoformat())
    return out


def t_fit(r):
    """Maximum likelihood fit of a Student-t with location and scale; log-likelihood and AIC vs the Normal."""
    r = np.asarray(pd.Series(r).dropna())
    nu, loc, scale = stats.t.fit(r)
    ll_t = stats.t.logpdf(r, nu, loc, scale).sum()
    ll_n = stats.norm.logpdf(r, r.mean(), r.std(ddof=0)).sum()
    return {'nu': nu, 'loc': loc, 'scale': scale, 'll_t': ll_t, 'll_n': ll_n,
            'aic_t': 2 * 3 - 2 * ll_t, 'aic_n': 2 * 2 - 2 * ll_n,
            'sd_t': scale * np.sqrt(nu / (nu - 2)) if nu > 2 else np.inf}


def sigma_days(r, ks=(3, 4, 5, 6)):
    """k-sigma days: observed number of days with |r - mean| > k sd, the number expected under the Normal
    distribution, and the Poisson probability of seeing at least as many under the Normal distribution."""
    r = pd.Series(r).dropna()
    z = (r - r.mean()) / r.std()
    out = {}
    for k in ks:
        p = 2 * stats.norm.sf(k)
        obs = int((z.abs() > k).sum())
        exp = len(r) * p
        out[k] = {'obs': obs, 'exp': exp, 'ratio': obs / exp, 'p_normal': p, 'every_normal': 1 / p,
                  'every_obs': len(r) / obs if obs else np.inf, 'poisson_p': stats.poisson.sf(obs - 1, exp),
                  'down': int((z < -k).sum()), 'up': int((z > k).sum())}
    return out


def normal_tail_table(ks=(1, 2, 3, 4, 5, 6), days_per_year=252):
    """Two-sided tail probability P(|Z| > k) of the standard Normal distribution and the waiting time it implies."""
    return {k: {'p': 2 * stats.norm.sf(k), 'every_days': 1 / (2 * stats.norm.sf(k)),
                'every_years': 1 / (2 * stats.norm.sf(k)) / days_per_year} for k in ks}


def acf(x, nlags=50):
    """Sample autocorrelation function rho(1), ..., rho(nlags)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    d = np.sum(x * x)
    return np.array([np.sum(x[h:] * x[:-h]) / d for h in range(1, nlags + 1)])


def ljung_box(x, m=10):
    """Ljung-Box statistic Q(m) = n(n+2) sum_{h<=m} rho(h)^2 / (n-h) and its chi-square(m) p-value."""
    n = len(x)
    rho = acf(x, m)
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, m + 1)))
    return {'Q': q, 'p': stats.chi2.sf(q, m), 'm': m}


def leverage(r, lags=20):
    """Leverage correlation L(k) = corr(r_t, |r_{t+k}|), k = 1..lags: a negative value means that falls are
    followed by higher volatility than rises."""
    r = pd.Series(r).dropna()
    a = r.abs()
    return np.array([np.corrcoef(r.iloc[:-k], a.iloc[k:])[0, 1] for k in range(1, lags + 1)])


def aggregate(r, h):
    """Non-overlapping h-period log returns (sums of h consecutive daily log returns)."""
    r = pd.Series(r).dropna()
    m = len(r) // h
    return pd.Series(r.values[:m * h].reshape(m, h).sum(axis=1))


# =============================================================================
# TABLES
# =============================================================================
def moments_table(names=ASSETS, start=START):
    rows = {}
    for k in names:
        r = returns(k, start)
        rows[LABELS[k]] = {**moments(r), **t_fit(r), 'ppy': periods_per_year(r)}
    return pd.DataFrame(rows).T


def sigma_table(names=ASSETS, start=START, ks=(3, 4, 5, 6)):
    rows = {}
    for k in names:
        sd = sigma_days(returns(k, start), ks)
        rows[LABELS[k]] = {f'{c}{j}': sd[j][c] for j in ks for c in ('obs', 'exp', 'poisson_p', 'down', 'up')}
    return pd.DataFrame(rows).T


def facts_table(names=ASSETS, start=START):
    """Cont (2001) stylised facts, one row per asset: the statistic behind each fact and whether it holds."""
    rows = {}
    for k in names:
        r = returns(k, start)
        n = len(r)
        m = moments(r)
        band = 1.96 / np.sqrt(n)
        rho_r = acf(r, 10)
        rho_a = acf(r.abs(), 10)
        lev = leverage(r, 5)
        month = aggregate(r, 21)
        rows[LABELS[k]] = {
            'exkurt': m['exkurt'], 'jb_p': m['jb_p'], 'heavy': bool(m['exkurt'] > 1 and m['jb_p'] < 0.05),
            'rho1': rho_r[0], 'n_sig_r': int((np.abs(rho_r) > band).sum()), 'band': band,
            'noacf': bool(np.mean(np.abs(rho_r)) < 0.05),
            'rho1_abs': rho_a[0], 'rho10_abs': rho_a[9], 'clust': bool(rho_a[9] > 3 * band),
            'lev1': lev[0], 'lev5': lev.mean(), 'lever': bool(lev.mean() < -band),
            'skew': m['skew'], 'q01': m['q01'], 'q99': m['q99'], 'gl_ratio': -m['q01'] / m['q99'],
            'gainloss': bool(m['skew'] < 0 and -m['q01'] > m['q99']),
            'exkurt_m': stats.kurtosis(month), 'agg': bool(stats.kurtosis(month) < m['exkurt'] / 2)}
    return pd.DataFrame(rows).T


# =============================================================================
# CHARTS: THE NORMAL AND LOGNORMAL DISTRIBUTIONS, CLT
# =============================================================================
def fig_normal_pdf(save=True):
    """Standard Normal density with the areas within 1, 2 and 3 standard deviations."""
    x = np.linspace(-4.2, 4.2, 600)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.plot(x, stats.norm.pdf(x), color=PAL['blue'], lw=2, label='Standard Normal density')
    for k, c, a in [(3, PAL['amber'], 0.35), (2, PAL['green'], 0.35), (1, PAL['teal'], 0.40)]:
        p = 1 - 2 * stats.norm.sf(k)
        ax.fill_between(x, stats.norm.pdf(x), where=(np.abs(x) <= k) & (np.abs(x) >= k - 1), interpolate=True,
                        color=c, alpha=a, lw=0)
        ax.fill_between([], [], color=c, alpha=a, lw=0, label=f'within {k} sd: {100 * p:.2f}%')
    ax.set_xlabel(r'$z = (x - \mu)/\sigma$')
    ax.set_ylabel('Density')
    st.legend_outside_bottom(ax, ncol=4, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_normal_pdf')
    return normal_tail_table()


def fig_lognormal(sigmas=(0.25, 0.5, 1.0), save=True):
    """Lognormal densities LN(0, sigma^2): mode exp(-sigma^2) < median 1 < mean exp(sigma^2/2)."""
    x = np.linspace(0.001, 4, 800)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    out = {}
    for s, c in zip(sigmas, [PAL['blue'], PAL['red'], PAL['green']]):
        ax.plot(x, stats.lognorm.pdf(x, s), color=c, lw=1.8, label=rf'$s = {s}$')
        mean = np.exp(s ** 2 / 2)
        ax.axvline(mean, color=c, lw=0.9, ls='--')
        out[s] = {'mode': np.exp(-s ** 2), 'median': 1.0, 'mean': mean, 'sd': np.sqrt((np.exp(s ** 2) - 1) * np.exp(s ** 2)),
                  'skew': (np.exp(s ** 2) + 2) * np.sqrt(np.exp(s ** 2) - 1)}
    ax.axvline(1, color=st.DarkText, lw=1.0, ls=':', label='median = 1 (all three)')
    ax.plot([], [], color=st.DarkText, lw=0.9, ls='--', label=r'mean $e^{s^2/2}$ (colour of its curve)')
    ax.set_xlabel(r'$x = P_T/P_0$')
    ax.set_ylabel('Density')
    st.legend_outside_bottom(ax, ncol=5, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_lognormal')
    return out


def fig_clt(p=0.2, ns=(5, 30, 300), reps=20000, seed=7, save=True):
    """Central limit theorem: standardised means of n Bernoulli(p) draws against the standard Normal density."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(ns), figsize=(10, 3.4), sharey=True)
    x = np.linspace(-4, 4, 400)
    out = {}
    for ax, n in zip(axes, ns):
        s = rng.binomial(n, p, size=reps)
        z = (s - n * p) / np.sqrt(n * p * (1 - p))
        vals, cnt = np.unique(z, return_counts=True)
        width = 1 / np.sqrt(n * p * (1 - p))
        ax.bar(vals, cnt / reps / width, width=width * 0.9, color=PAL['amber'], alpha=0.7,
               label='standardised mean of n draws' if n == ns[0] else '_')
        ax.plot(x, stats.norm.pdf(x), color=PAL['blue'], lw=1.8, label='N(0, 1) density' if n == ns[0] else '_')
        ax.set_title(f'n = {n}')
        ax.set_xlim(-4, 5)
        ax.set_xlabel('z')
        out[n] = {'skew_mean': (1 - 2 * p) / np.sqrt(n * p * (1 - p)), 'p_tail': float(np.mean(z > 1.96)),
                  'normal_tail': float(stats.norm.sf(1.96))}
    axes[0].set_ylabel('Density')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_clt')
    return out


def fig_normal_approx(cases=((5, 0.5), (20, 0.5), (20, 0.1), (100, 0.1)), save=True):
    """Normal approximation of the binomial distribution B(n, p) by N(np, np(1-p)), for four (n, p)."""
    fig, axes = plt.subplots(1, len(cases), figsize=(10.5, 3.3))
    out = {}
    for ax, (n, p) in zip(axes, cases):
        k = np.arange(0, n + 1)
        pmf = stats.binom.pmf(k, n, p)
        mu, sd = n * p, np.sqrt(n * p * (1 - p))
        lo, hi = max(0, int(mu - 4.5 * sd)), min(n, int(np.ceil(mu + 4.5 * sd)))
        ax.bar(k[lo:hi + 1], pmf[lo:hi + 1], color=PAL['amber'], alpha=0.7, width=0.85,
               label='binomial probabilities' if n == cases[0][0] and p == cases[0][1] else '_')
        x = np.linspace(lo - 0.5, hi + 0.5, 300)
        ax.plot(x, stats.norm.pdf(x, mu, sd), color=PAL['blue'], lw=1.8,
                label='Normal density N(np, np(1-p))' if n == cases[0][0] and p == cases[0][1] else '_')
        ax.set_title(f'n = {n}, p = {p}')
        ax.set_xlabel('k')
        # largest error of the cumulative probabilities (with continuity correction)
        err = np.max(np.abs(stats.binom.cdf(k, n, p) - stats.norm.cdf(k + 0.5, mu, sd)))
        out[f'{n}_{p}'] = {'n': n, 'p': p, 'max_cdf_err': float(err), 'np1p': n * p * (1 - p)}
    axes[0].set_ylabel('Probability')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_normal_approx')
    return out


# =============================================================================
# CHARTS: SHAPE, NORMALITY TESTS, QQ PLOTS
# =============================================================================
def fig_shapes(save=True):
    """Same mean and variance, different shape: Normal vs Student-t(5) scaled to unit variance (kurtosis), and
    skew-Normal densities with negative and positive skewness."""
    x = np.linspace(-5, 5, 800)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    nu = 5
    c = np.sqrt((nu - 2) / nu)
    axes[0].plot(x, stats.norm.pdf(x), color=PAL['blue'], lw=1.8, label='Normal (excess kurtosis 0)')
    axes[0].plot(x, stats.t.pdf(x / c, nu) / c, color=PAL['red'], lw=1.8, label='Student-t(5), unit variance (excess kurtosis 6)')
    axes[0].set_title('Kurtosis: taller centre, heavier tails')
    out = {}
    for a, col, lab in [(-5, PAL['green'], 'negative skewness'), (5, PAL['purple'], 'positive skewness')]:
        m, v, s = stats.skewnorm.stats(a, moments='mvs')
        sd = np.sqrt(v)
        axes[1].plot(x, stats.skewnorm.pdf(x * sd + m, a) * sd, color=col, lw=1.8, label=f'{lab} ({float(s):+.2f})')
        out[lab] = float(s)
    axes[1].plot(x, stats.norm.pdf(x), color=PAL['blue'], lw=1.2, ls='--', label='_')
    axes[1].set_title('Skewness: a longer left or right tail')
    for ax in axes:
        ax.set_xlabel('Standardised value')
    axes[0].set_ylabel('Density')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_shapes')
    return out


def fig_dax_density(start=START, save=True):
    """DAX daily log returns: kernel density estimate against the Normal density with the same mean and standard
    deviation, on a linear and on a log scale (port of SFEDaxReturnDistribution)."""
    r = returns('dax', start)
    kde = stats.gaussian_kde(r)
    x = np.linspace(-9, 9, 700)
    fn = stats.norm.pdf(x, r.mean(), r.std())
    fk = kde(x)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, log in zip(axes, [False, True]):
        ax.plot(x, fk, color=COLORS['dax'], lw=1.8, label='DAX, kernel density estimate')
        ax.plot(x, fn, color=PAL['red'], lw=1.5, ls='--', label='Normal, same mean and sd')
        ax.set_xlabel('Daily log return (%)')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-6, 1)
            ax.set_title('Log scale: the tails')
        else:
            ax.set_xlim(-5, 5)
            ax.set_title('Linear scale: the centre')
    axes[0].set_ylabel('Density')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_dax_density')
    z0 = 0.0
    return {'peak_kde': float(kde(r.mean())[0]), 'peak_normal': float(stats.norm.pdf(z0, 0, r.std())),
            'ratio_at6': float(kde(-6.0)[0] / stats.norm.pdf(-6.0, r.mean(), r.std())),
            'within1': float(np.mean(np.abs(r - r.mean()) <= r.std())), 'normal_within1': float(1 - 2 * stats.norm.sf(1))}


def qq_points(r, dist='norm', params=None):
    """Theoretical and empirical quantiles for a QQ plot (plotting positions (i - 0.5)/n); standardised data
    against N(0,1), or raw data against a fitted t (params = (nu, loc, scale))."""
    r = np.sort(np.asarray(pd.Series(r).dropna()))
    n = len(r)
    u = (np.arange(1, n + 1) - 0.5) / n
    if dist == 'norm':
        return stats.norm.ppf(u), (r - r.mean()) / r.std()
    nu, loc, scale = params
    return stats.t.ppf(u, nu, loc, scale), r


def fig_qq(names=('sp500', 'bet', 'btc'), dist='norm', start=START, save=True):
    """QQ plots of daily log returns against the Normal distribution (standardised) or the fitted Student-t."""
    fig, axes = plt.subplots(1, len(names), figsize=(10.5, 3.6))
    out = {}
    for ax, k in zip(axes, names):
        r = returns(k, start)
        params = None
        if dist == 't':
            f = t_fit(r)
            params = (f['nu'], f['loc'], f['scale'])
        tq, eq = qq_points(r, dist, params)
        ax.scatter(tq, eq, s=6, color=COLORS[k], label=f'{SHORT[k]} daily log returns')
        lim = [min(tq.min(), eq.min()), max(tq.max(), eq.max())]
        ax.plot(lim, lim, color='black', lw=0.9, ls='--', label='45-degree line' if k == names[0] else '_')
        ax.set_title(SHORT[k] + (f' vs t({params[0]:.1f})' if dist == 't' else ' vs Normal'))
        ax.set_xlabel('Theoretical quantile' + (' (N(0,1))' if dist == 'norm' else ' (fitted t, %)'))
        out[k] = {'min_emp': float(eq.min()), 'min_theo': float(tq.min()), 'max_emp': float(eq.max()), 'max_theo': float(tq.max())}
    axes[0].set_ylabel('Empirical quantile' + (' (standardised)' if dist == 'norm' else ' (%)'))
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_qq_normal' if dist == 'norm' else 'sfm_ch2_qq_t')
    return out


# =============================================================================
# CHARTS: STUDENT-t
# =============================================================================
def fig_t_densities(nus=(3, 5, 10), save=True):
    """Student-t densities scaled to unit variance against the standard Normal; linear and log scale."""
    x = np.linspace(-7, 7, 800)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    cols = [PAL['red'], PAL['amber'], PAL['green']]
    out = {}
    for ax, log in zip(axes, [False, True]):
        ax.plot(x, stats.norm.pdf(x), color=PAL['blue'], lw=2, label='Normal')
        for nu, c in zip(nus, cols):
            s = np.sqrt((nu - 2) / nu)
            ax.plot(x, stats.t.pdf(x / s, nu) / s, color=c, lw=1.5, label=f't({nu}), unit variance')
        ax.set_xlabel('Standardised value')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-7, 1)
            ax.set_title('Log scale')
        else:
            ax.set_xlim(-4, 4)
            ax.set_title('Linear scale')
    axes[0].set_ylabel('Density')
    for nu in nus:
        s = np.sqrt((nu - 2) / nu)
        out[nu] = {'p4': float(2 * stats.t.sf(4 / s, nu)), 'exkurt': 6 / (nu - 4) if nu > 4 else np.inf}
    out['normal_p4'] = float(2 * stats.norm.sf(4))
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_t_densities')
    return out


def fig_t_fit(name='sp500', start=START, save=True):
    """Histogram of daily log returns with the fitted Normal and Student-t densities (linear and log scale)."""
    r = returns(name, start)
    f = t_fit(r)
    x = np.linspace(r.min() - 0.5, r.max() + 0.5, 800)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    dens, edges = np.histogram(r, bins=120, density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    for ax, log in zip(axes, [False, True]):
        if log:
            ax.scatter(mid[dens > 0], dens[dens > 0], s=12, color=COLORS[name], label='_')
        else:
            ax.hist(r, bins=160, density=True, color=COLORS[name], alpha=0.8, label=f'{SHORT[name]} (histogram)')
        ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=PAL['red'], lw=1.5, ls='--', label='Normal fit' if not log else '_')
        ax.plot(x, stats.t.pdf(x, f['nu'], f['loc'], f['scale']), color=PAL['green'], lw=1.8,
                label=f"Student-t fit, nu = {f['nu']:.2f}" if not log else '_')
        ax.set_xlabel('Daily log return (%)')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 2)
            ax.set_title('Log scale (points: histogram heights)')
        else:
            ax.set_xlim(-5, 5)
            ax.set_title('Linear scale')
    axes[0].set_ylabel('Density')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig(f'sfm_ch2_t_fit_{name}')
    return f


# =============================================================================
# CHARTS: STYLISED FACTS
# =============================================================================
def fig_sigma_days(names=ASSETS, start=START, k=4, save=True):
    """Number of days beyond k standard deviations: observed vs expected under the Normal distribution."""
    fig, ax = plt.subplots(figsize=(10, 3.9))
    out = {}
    x = np.arange(len(names))
    obs, exp = [], []
    for k_ in names:
        sd = sigma_days(returns(k_, start), (3, 4, 5, 6))
        out[k_] = sd
        obs.append(sd[k]['obs'])
        exp.append(sd[k]['exp'])
    cols_ = [COLORS[k_] for k_ in names]
    ax.bar(x - 0.2, obs, width=0.4, color=cols_, label='_')
    ax.bar(x + 0.2, exp, width=0.4, facecolor='none', hatch='////', edgecolor=cols_, lw=1.0, label='_')
    ax.bar([0], [0], color=st.DarkText, label=f'observed days beyond {k} sd (solid)')
    ax.bar([0], [0], facecolor='none', hatch='////', edgecolor=st.DarkText, label='expected under the Normal distribution (hatched)')
    for xi, o, e in zip(x, obs, exp):
        ax.text(xi - 0.2, o * 1.08, str(o), ha='center', va='bottom', fontsize=10, color=st.DarkText)
        ax.text(xi + 0.2, e * 1.08, f'{e:.2f}', ha='center', va='bottom', fontsize=9, color=st.DarkText)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 80)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k_] for k_ in names])
    ax.set_ylabel('Number of days (log scale)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_sigma_days')
    return out


def fig_aggregation(names=INDICES, start=START, horizons=HORIZONS, save=True):
    """Aggregational Gaussianity: excess kurtosis of non-overlapping h-day log returns, h = 1, ..., 63."""
    fig, ax = plt.subplots(figsize=(9, 3.9))
    out = {}
    for k in names:
        r = returns(k, start)
        ku = [stats.kurtosis(aggregate(r, h)) for h in horizons]
        jbp = [stats.jarque_bera(aggregate(r, h)).pvalue for h in horizons]
        ax.plot(horizons, ku, marker='o', color=COLORS[k], lw=1.6, label=SHORT[k])
        out[k] = {h: {'exkurt': float(a), 'jb_p': float(b), 'n': len(aggregate(r, h))} for h, a, b in zip(horizons, ku, jbp)}
    ax.axhline(0, color='black', lw=0.8, ls='--', label='Normal distribution (0)')
    ax.set_xscale('log')
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel('Horizon h (days, log scale)')
    ax.set_ylabel('Excess kurtosis of h-day returns')
    st.legend_outside_bottom(ax, ncol=5, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_aggregation')
    return out


def fig_qq_horizons(name='sp500', start=START, save=True):
    """QQ plots of daily and monthly (21-day) log returns of one series against the Normal distribution."""
    r = returns(name, start)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    out = {}
    for ax, h, lab in zip(axes, [1, 21], ['daily', 'monthly (21 days)']):
        x = aggregate(r, h)
        tq, eq = qq_points(x)
        ax.scatter(tq, eq, s=7, color=COLORS[name] if h == 1 else PAL['purple'], label=f'{SHORT[name]}, {lab}')
        lim = [min(tq.min(), eq.min()), max(tq.max(), eq.max())]
        ax.plot(lim, lim, color='black', lw=0.9, ls='--', label='45-degree line' if h == 1 else '_')
        ax.set_title(f'{lab}, n = {len(x)}')
        ax.set_xlabel('N(0,1) quantile')
        out[h] = {'n': len(x), 'exkurt': float(stats.kurtosis(x)), 'skew': float(stats.skew(x)), 'jb_p': float(stats.jarque_bera(x).pvalue)}
    axes[0].set_ylabel('Standardised empirical quantile')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_qq_horizons')
    return out


def fig_clustering(names=('sp500', 'bet', 'btc'), start=START, save=True):
    """Daily log returns over time: calm and turbulent periods alternate (volatility clustering)."""
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 5.2), sharex=True)
    for ax, k in zip(axes, names):
        r = returns(k, start)
        ax.plot(r.index, r, color=COLORS[k], lw=0.5, label=f'{SHORT[k]} daily log return (%)')
        ax.set_ylabel('%')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_clustering')


def fig_acf(names=('sp500', 'bet', 'btc'), start=START, nlags=50, save=True):
    """Sample ACF of daily log returns (left) and of absolute returns (right), with the 95% band +-1.96/sqrt(n)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    lags = np.arange(1, nlags + 1)
    out = {}
    nmin = 1e9
    for k in names:
        r = returns(k, start)
        nmin = min(nmin, len(r))
        a_r, a_a = acf(r, nlags), acf(r.abs(), nlags)
        axes[0].plot(lags, a_r, marker='o', ms=3, lw=0.8, color=COLORS[k], label=SHORT[k])
        axes[1].plot(lags, a_a, marker='o', ms=3, lw=0.8, color=COLORS[k], label='_')
        out[k] = {'n': len(r), 'rho_r': a_r[:10].tolist(), 'rho_abs': a_a.tolist(), 'band': 1.96 / np.sqrt(len(r)),
                  'lb_r': ljung_box(r.values, 10), 'lb_abs': ljung_box(r.abs().values, 10),
                  'n_out_r': int((np.abs(a_r) > 1.96 / np.sqrt(len(r))).sum()), 'max_abs_r': float(np.abs(a_r).max()),
                  'rho1_ex2020': float(acf(r.drop(r.loc['2020-02-20':'2020-04-30'].index), 1)[0])}
    b = 1.96 / np.sqrt(nmin)
    for ax in axes:
        ax.axhspan(-b, b, color=PAL['green'], alpha=0.18, lw=0, label='_')
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xlabel('Lag (days)')
    axes[0].plot([], [], color=PAL['green'], alpha=0.35, lw=6, label='95% band under independence')
    axes[0].set_title(r'Returns $r_t$')
    axes[1].set_title(r'Absolute returns $|r_t|$')
    axes[0].set_ylabel('Autocorrelation')
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_acf')
    return out


def fig_leverage(names=INDICES, start=START, lags=20, save=True):
    """Leverage effect: L(k) = corr(r_t, |r_{t+k}|), k = 1..20 days."""
    fig, ax = plt.subplots(figsize=(9, 3.9))
    out = {}
    nmin = 1e9
    for k in names:
        r = returns(k, start)
        nmin = min(nmin, len(r))
        L = leverage(r, lags)
        ax.plot(np.arange(1, lags + 1), L, marker='o', ms=3.5, color=COLORS[k], lw=1.2, label=SHORT[k])
        out[k] = {'L': L.tolist(), 'mean5': float(L[:5].mean()), 'band': 1.96 / np.sqrt(len(r))}
    b = 1.96 / np.sqrt(nmin)
    ax.axhspan(-b, b, color=PAL['green'], alpha=0.18, lw=0)
    ax.plot([], [], color=PAL['green'], alpha=0.35, lw=6, label='95% band under independence')
    ax.set_xticks(range(1, lags + 1))
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('Lag k (days)')
    ax.set_ylabel(r'$L(k) = \mathrm{corr}(r_t, |r_{t+k}|)$')
    st.legend_outside_bottom(ax, ncol=5, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_leverage')
    return out


def fig_gain_loss(names=ASSETS, start=START, save=True):
    """Gain/loss asymmetry: size of the 1% left tail quantile vs the 99% right tail quantile, and the number of
    days below -4 sd vs above +4 sd."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    x = np.arange(len(names))
    out = {}
    for i, k in enumerate(names):
        r = returns(k, start)
        m = moments(r)
        sd = sigma_days(r, (4,))[4]
        out[k] = {'q01': m['q01'], 'q99': m['q99'], 'skew': m['skew'], 'down4': sd['down'], 'up4': sd['up']}
    axes[0].bar(x - 0.2, [-out[k]['q01'] for k in names], width=0.4, color=PAL['red'], label='loss: -(1% quantile)')
    axes[0].bar(x + 0.2, [out[k]['q99'] for k in names], width=0.4, color=PAL['green'], label='gain: 99% quantile')
    axes[0].set_ylabel('% per day')
    axes[0].set_title('Size of a 1-in-100 day')
    axes[1].bar(x - 0.2, [out[k]['down4'] for k in names], width=0.4, color=PAL['red'], label='_')
    axes[1].bar(x + 0.2, [out[k]['up4'] for k in names], width=0.4, color=PAL['green'], label='_')
    axes[1].set_ylabel('Number of days')
    axes[1].set_title('Days below -4 sd and above +4 sd')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[k] for k in names], rotation=45, ha='right')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_gain_loss')
    return out


def tlv_check(start=START):
    """Banca Transilvania: the two largest reversals of the adjusted close (30-31 May 2016) and the excess kurtosis
    with and without them."""
    r = returns('tlv', start)
    t = read_market('TLV.RO').loc['2016-05-26':'2016-06-01']
    bad = ['2016-05-30', '2016-05-31']
    clean = r.drop(pd.to_datetime(bad))
    return {'r_bad': [float(r.loc[d]) for d in bad], 'close': {d.date().isoformat(): float(v) for d, v in t['close'].items()},
            'adj': {d.date().isoformat(): float(v) for d, v in t['adjusted_close'].items()},
            'exkurt': float(stats.kurtosis(r)), 'exkurt_clean': float(stats.kurtosis(clean)),
            'sd': float(r.std()), 'sd_clean': float(clean.std()), 'n': len(r),
            'nu': t_fit(r)['nu'], 'nu_clean': t_fit(clean)['nu']}


def fig_tails_by_year(names=('btc', 'sp500'), save=True):
    """Excess kurtosis and the fitted Student-t degrees of freedom of daily log returns, by calendar year."""
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.4), sharex=True)
    out = {}
    for k in names:
        r = returns(k, '2015-01-01')
        g = r.groupby(r.index.year)
        yrs = [y for y, x in g if len(x) > 150]
        ku = [stats.kurtosis(g.get_group(y)) for y in yrs]
        nu = [t_fit(g.get_group(y))['nu'] for y in yrs]
        axes[0].plot(yrs, ku, marker='o', color=COLORS[k], lw=1.4, label=SHORT[k])
        axes[1].plot(yrs, np.minimum(nu, 30), marker='o', color=COLORS[k], lw=1.4, label='_')
        out[k] = {int(y): {'exkurt': float(a), 'nu': float(b)} for y, a, b in zip(yrs, ku, nu)}
    axes[0].set_title('Excess kurtosis by year')
    axes[1].set_title(r'Student-t $\hat\nu$ by year (capped at 30)')
    axes[1].set_xlabel('Year')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch2_tails_by_year')
    return out


def loss_days(names=('sp500', 'bet', 'btc'), thresholds=(3, 5, 10), start=START):
    """Days with a log return below -thr %: observed count vs the count expected under the Normal distribution
    with the sample mean and standard deviation."""
    out = {}
    for k in names:
        r = returns(k, start)
        out[k] = {thr: {'obs': int((r < -thr).sum()), 'exp': len(r) * stats.norm.cdf((-thr - r.mean()) / r.std()),
                        'z': (-thr - r.mean()) / r.std()} for thr in thresholds}
    return out


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.date().isoformat()
    return x


if __name__ == '__main__':
    st.apply()
    N = {}
    mt = moments_table()
    mt.to_csv(os.path.join(TABLE_DIR, 'ch2_moments.csv'))
    print(mt[['n', 'mean', 'sd', 'skew', 'exkurt', 'jb', 'nu', 'aic_n', 'aic_t']].round(3))
    N['mom'] = {k: mt.loc[LABELS[k]].to_dict() for k in ASSETS}
    N['sigma_table'] = sigma_table().to_dict(orient='index')
    ft = facts_table()
    print(ft.round(3).T)
    N['facts'] = {k: ft.loc[LABELS[k]].to_dict() for k in ASSETS}
    N['normal_tails'] = fig_normal_pdf()
    N['lognormal'] = fig_lognormal()
    N['clt'] = fig_clt()
    N['approx'] = fig_normal_approx()
    N['shapes'] = fig_shapes()
    N['dax'] = fig_dax_density()
    N['qq_n'] = fig_qq(dist='norm')
    N['qq_t'] = fig_qq(dist='t')
    N['tdens'] = fig_t_densities()
    N['tfit'] = fig_t_fit()
    N['sigma'] = fig_sigma_days()
    N['agg'] = fig_aggregation()
    N['qqh'] = fig_qq_horizons()
    fig_clustering()
    N['acf'] = fig_acf()
    N['lev'] = fig_leverage()
    N['gl'] = fig_gain_loss()
    N['tlv'] = tlv_check()
    N['loss'] = loss_days()
    N['years'] = fig_tails_by_year()
    N['start'] = START
    N['end'] = load_close('sp500').index[-1].date().isoformat()
    with open(os.path.join(TABLE_DIR, 'ch2_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1, default=str)
    print(json.dumps(to_json({k: N[k] for k in ['sigma', 'tlv', 'dax', 'tdens', 'clt', 'approx']}), indent=1, default=str)[:5000])
