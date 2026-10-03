"""
generate_fmh_charts.py -- charts and numbers of Chapter 11 (SFM): fractal markets hypothesis and long memory
==========================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * self-similarity -- the standard deviation of h-day returns against h (S&P 500, BET, Bitcoin);
  * processes       -- fractional Brownian motion paths for H = 0.3, 0.5, 0.7 (as in SFEfbmplot); the ACF of fractional
                       Gaussian noise for H = 0.2 and 0.8 (as in SFEfgnacf), hyperbolic against exponential decay;
                       ARFIMA(0,d,0) paths and ACF for d = 0.4 and d = -0.4 (as in SFEarfima); exact simulation by
                       circulant embedding (Davies-Harte);
  * estimators      -- classical R/S (Hurst 1951; Mandelbrot and Wallis 1969), Lo's (1991) modified R/S with the
                       Andrews bandwidth, DFA (Peng et al. 1994), the GPH log-periodogram regression (Geweke and
                       Porter-Hudak 1983); a worked R/S example on eight returns;
  * Monte Carlo     -- small-sample distribution of the estimators under i.i.d. Normal returns (Weron 2002); size and
                       power of the classical and modified R/S tests under AR(1) and fractional Gaussian noise;
                       spurious long memory from a mean shift and from GARCH volatility clustering;
  * real data       -- the Nile flow 1871-1970 and its 1898 break (Cobb 1978); R/S, DFA, GPH and Lo's test for daily
                       returns and absolute returns of the S&P 500, DAX, BET, Bitcoin, Banca Transilvania and OMV
                       Petrom; the ACF of r, |r| and r^2; rolling Hurst exponents (DFA, windows of 1000 days moved by
                       21 days) with Monte Carlo bands.
Output: charts/sfm_ch11_*.pdf/.png, Quantlets/Ch_11/ch11_numbers.json and ch11_memory_table.csv
Based on the Quantlets SFEfbmplot, SFEfgnacf and SFEarfima (github.com/QuantLet/SFE) and on Franke, Haerdle and
Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 14.
(The file is named generate_fmh_charts.py because Quantlets/Ch_11 also holds the older VaR Quantlets of the course.)
Run:  python3 Quantlets/Ch_11/generate_fmh_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import log_returns   # noqa: E402
import sfm_style as st             # noqa: E402

TABLE_DIR = HERE
NAME = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania',
        'snp': 'OMV Petrom'}
START = {'sp500': '2000-01-01', 'dax': '2000-01-01', 'bet': '2000-01-01', 'btc': None,
         'tlv': '2010-01-01', 'snp': '2010-01-01'}
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']
# Banca Transilvania: the adjustment for the bonus shares of May 2016 is applied one day late in the adjusted close
# (Chapter 2); the two days are removed.
BAD_DAYS = {'tlv': ['2016-05-30', '2016-05-31']}
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F', 'tlv': '#E67E22', 'snp': '#2E7D32'}
SEED = 2026
NMIN = 10                    # smallest block size of R/S and DFA
WINDOW = 1000                # rolling window (trading days; calendar days for Bitcoin)
STEP = 21                    # the window moves by 21 observations (about one month)
MC_REPS = 500                # Monte Carlo replications under i.i.d. Normal returns
LO_CRIT = (0.809, 1.862)     # 2.5% and 97.5% quantiles of Lo's V statistic under short memory (Lo 1991, Table II)
GPH_POWER = 0.5              # GPH bandwidth m = [T^0.5] (Geweke and Porter-Hudak 1983)


# =============================================================================
# DATA
# =============================================================================
def returns(k, start=None, end=None):
    """Daily log returns in % on the series' own calendar; known data errors removed."""
    r = log_returns(k, start or START.get(k)) if end is None else log_returns(k, start or START.get(k), end)
    if k in BAD_DAYS:
        r = r.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return r


def nile():
    """Annual flow of the Nile at Aswan, 1871-1970, in 10^8 cubic metres (Cobb 1978; statsmodels dataset)."""
    import statsmodels.api as sm
    d = sm.datasets.nile.load_pandas().data
    return pd.Series(d['volume'].values, index=d['year'].astype(int).values, name='Nile flow')


# =============================================================================
# ESTIMATORS: R/S, LO'S MODIFIED R/S, DFA, GPH
# =============================================================================
def block_sizes(N, nmin=NMIN, nmax=None, num=20):
    """Block sizes for R/S and DFA: about `num` values spaced evenly on a log scale between nmin and N/4."""
    nmax = nmax or N // 4
    return np.unique(np.floor(np.logspace(np.log10(nmin), np.log10(nmax), num)).astype(int))


def rs_block(x):
    """R/S of one block: range of the cumulative deviations from the mean, divided by the standard deviation."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    return (y.max() - y.min()) / x.std()


def rs_curve(x, sizes=None):
    """Average R/S over the non-overlapping blocks of each size n (Hurst 1951; Mandelbrot and Wallis 1969)."""
    x = np.asarray(x, float)
    sizes = block_sizes(len(x)) if sizes is None else np.asarray(sizes)
    out = []
    for n in sizes:
        m = len(x) // n
        X = x[:m * n].reshape(m, n)
        Y = np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)
        R = Y.max(axis=1) - Y.min(axis=1)
        S = X.std(axis=1)
        ok = S > 0
        out.append(np.mean(R[ok] / S[ok]))
    return sizes, np.array(out)


def hurst_rs(x, sizes=None):
    """Hurst exponent: slope of log(R/S)_n on log n."""
    n, rs = rs_curve(x, sizes)
    return float(np.polyfit(np.log10(n), np.log10(rs), 1)[0])


def dfa_curve(x, sizes=None):
    """DFA-1 (Peng et al. 1994): profile Y = cumsum(x - mean); in each block of size n a straight line is fitted
    to Y; F(n) is the root mean square of the deviations from the lines."""
    x = np.asarray(x, float)
    sizes = block_sizes(len(x)) if sizes is None else np.asarray(sizes)
    Y = np.cumsum(x - x.mean())
    F = []
    for n in sizes:
        m = len(Y) // n
        B = Y[:m * n].reshape(m, n)
        t = np.arange(n) - (n - 1) / 2
        b = (B - B.mean(axis=1, keepdims=True)) @ t / (t @ t)
        resid = B - B.mean(axis=1, keepdims=True) - np.outer(b, t)
        F.append(np.sqrt(np.mean(resid ** 2)))
    return sizes, np.array(F)


def hurst_dfa(x, sizes=None):
    """DFA exponent: slope of log F(n) on log n (equal to H for fractional Gaussian noise)."""
    n, F = dfa_curve(x, sizes)
    return float(np.polyfit(np.log10(n), np.log10(F), 1)[0])


def periodogram(x):
    """I(lambda_j) = |sum_t x_t exp(-i lambda_j t)|^2 / (2 pi T) at the Fourier frequencies lambda_j = 2 pi j / T."""
    x = np.asarray(x, float)
    T = len(x)
    I = np.abs(np.fft.fft(x - x.mean())) ** 2 / (2 * np.pi * T)
    j = np.arange(1, T // 2 + 1)
    return 2 * np.pi * j / T, I[1:T // 2 + 1]


def gph(x, power=GPH_POWER):
    """GPH estimator of d (Geweke and Porter-Hudak 1983; FHH 14.5.2): OLS slope of log I(lambda_j) on
    -log(4 sin^2(lambda_j / 2)), j = 1, ..., m = [T^power]; asymptotic standard error pi / sqrt(24 m)."""
    lam, I = periodogram(x)
    m = int(np.floor(len(x) ** power))
    X = -np.log(4 * np.sin(lam[:m] / 2) ** 2)
    y = np.log(I[:m])
    d = float(np.polyfit(X, y, 1)[0])
    return {'d': d, 'se': float(np.pi / np.sqrt(24 * m)), 'm': m, 'H': d + 0.5}


def lo_test(x, q=None):
    """Lo's (1991) modified R/S: V(q) = R / (sqrt(N) sigma(q)), sigma^2(q) the Newey-West variance with Bartlett
    weights 1 - j/(q+1); q = 0 gives the classical statistic. Default q: Andrews (1991) rule used by Lo,
    q = [(3N/2)^(1/3) (2 rho / (1 - rho^2))^(2/3)], rho the first-order autocorrelation.
    Short memory is rejected at 5% if V is outside [0.809, 1.862]."""
    x = np.asarray(x, float)
    N = len(x)
    xc = x - x.mean()
    Y = np.cumsum(xc)
    R = Y.max() - Y.min()
    g0 = xc @ xc / N
    rho = (xc[1:] @ xc[:-1] / N) / g0
    if q is None:
        q = int(np.floor((1.5 * N) ** (1 / 3) * (2 * abs(rho) / (1 - rho ** 2)) ** (2 / 3)))
    s2 = g0 + 2 * sum((1 - j / (q + 1)) * (xc[j:] @ xc[:-j] / N) for j in range(1, q + 1))
    V = R / np.sqrt(N * s2)
    V0 = R / np.sqrt(N * g0)
    return {'V': float(V), 'V0': float(V0), 'q': int(q), 'rho1': float(rho),
            'reject': bool(not LO_CRIT[0] <= V <= LO_CRIT[1]), 'reject0': bool(not LO_CRIT[0] <= V0 <= LO_CRIT[1])}


def acf(x, nlags):
    """Sample autocorrelations at lags 1..nlags."""
    x = np.asarray(x, float) - np.mean(x)
    g0 = x @ x
    return np.array([x[k:] @ x[:-k] / g0 for k in range(1, nlags + 1)])


def all_estimates(x):
    """R/S, DFA, GPH (d and H = d + 1/2) and Lo's test of one series."""
    g = gph(x)
    lo = lo_test(x)
    return {'n': int(len(x)), 'rs': hurst_rs(x), 'dfa': hurst_dfa(x), 'gph_d': g['d'], 'gph_se': g['se'],
            'gph_m': g['m'], 'gph_H': g['H'], 'lo_V': lo['V'], 'lo_V0': lo['V0'], 'lo_q': lo['q'],
            'lo_reject': lo['reject'], 'lo_reject0': lo['reject0']}


# =============================================================================
# SIMULATION: FRACTIONAL GAUSSIAN NOISE, fBm, ARFIMA(0,d,0) (SFEfbmplot, SFEfgnacf, SFEarfima)
# =============================================================================
def fgn_acvf(n, H):
    """Autocovariances of standard fractional Gaussian noise: 0.5 (|k+1|^2H - 2|k|^2H + |k-1|^2H), k = 0..n-1."""
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def arfima_acf(n, d):
    """Autocorrelations of ARFIMA(0,d,0): rho_k = Gamma(1-d) Gamma(k+d) / (Gamma(d) Gamma(k+1-d)), k = 0..n-1,
    computed with the recursion rho_k = rho_{k-1} (k - 1 + d) / (k - d)."""
    r = np.ones(n)
    for k in range(1, n):
        r[k] = r[k - 1] * (k - 1 + d) / (k - d)
    return r


def arfima_var(d):
    """Variance of ARFIMA(0,d,0) with unit innovation variance: Gamma(1 - 2d) / Gamma(1 - d)^2."""
    return float(np.exp(special.gammaln(1 - 2 * d) - 2 * special.gammaln(1 - d)))


def circulant_gaussian(acvf, rng, size=1):
    """Exact simulation of a stationary Gaussian series with autocovariances acvf[0..n-1] by circulant embedding
    (Davies and Harte); returns an array (size, n)."""
    n = len(acvf)
    c = np.concatenate([acvf, acvf[-2:0:-1]])
    M = len(c)
    lam = np.clip(np.fft.fft(c).real, 0, None)
    Z = rng.standard_normal((size, M)) + 1j * rng.standard_normal((size, M))
    W = np.fft.fft(np.sqrt(lam / M) * Z, axis=1)
    return W.real[:, :n]


def sim_fgn(n, H, rng, size=1):
    return circulant_gaussian(fgn_acvf(n, H), rng, size)


def sim_arfima(n, d, rng, size=1):
    return circulant_gaussian(arfima_var(d) * arfima_acf(n, d), rng, size)


def sim_garch(n, omega, alpha, beta, rng, size=1, burn=500):
    """GARCH(1,1) with Normal innovations, `size` independent paths of length n (unconditional variance 1 if
    omega = 1 - alpha - beta)."""
    z = rng.standard_normal((size, n + burn))
    s2 = np.full(size, omega / (1 - alpha - beta))
    e = np.empty((size, n + burn))
    for t in range(n + burn):
        e[:, t] = np.sqrt(s2) * z[:, t]
        s2 = omega + alpha * e[:, t] ** 2 + beta * s2
    return e[:, burn:]


def fig_fbm_paths(n=1000, Hs=(0.3, 0.5, 0.7), save=True):
    """Fractional Brownian motion paths for three Hurst exponents (as in SFEfbmplot); the same random numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    out = {}
    for H, c in zip(Hs, [st.IDAred, st.MainBlue, st.Forest]):
        rng = np.random.default_rng(SEED)
        x = sim_fgn(n, H, rng)[0]
        b = np.concatenate([[0], np.cumsum(x)])
        lab = {0.5: 'H = 0.5: Brownian motion'}.get(H, f'H = {H}: ' + ('anti-persistent' if H < 0.5 else 'persistent'))
        axes[0].plot(np.arange(n + 1), b, color=c, lw=1.0, label=lab)
        axes[1].plot(np.arange(150), x[:150], color=c, lw=0.9, label='_')
        out[str(H)] = {'acf1': float(acf(x, 1)[0]), 'acf1_th': float(2 ** (2 * H - 1) - 1), 'range': float(b.max() - b.min())}
    axes[0].set_title('fractional Brownian motion B_H(t)')
    axes[1].set_title('its increments (fractional Gaussian noise), first 150')
    for ax in axes:
        ax.set_xlabel('time t')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_fbm_paths')
    return out


def fig_fgn_acf(n=5000, nlags=50, save=True):
    """ACF of fractional Gaussian noise for H = 0.2 and 0.8 (as in SFEfgnacf): sample and theoretical; right panel:
    hyperbolic decay of fGn (H = 0.8) against the exponential decay of an AR(1) with the same lag-1 autocorrelation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    k = np.arange(1, nlags + 1)
    out = {}
    for H, c in [(0.8, st.MainBlue), (0.2, st.IDAred)]:
        rng = np.random.default_rng(SEED + int(10 * H))
        x = sim_fgn(n, H, rng)[0]
        th = fgn_acvf(nlags + 1, H)[1:]
        axes[0].plot(k, acf(x, nlags), 'o', ms=3.5, color=c, label=f'H = {H}: sample ACF (n = {n})')
        axes[0].plot(k, th, '-', lw=1.4, color=c, label=f'H = {H}: theoretical ACF')
        out[str(H)] = {'rho1': float(th[0]), 'rho10': float(th[9]), 'rho50': float(th[49])}
    axes[0].axhline(0, color='black', lw=0.6)
    axes[0].set_xlabel('lag k')
    axes[0].set_ylabel('autocorrelation')
    H = 0.8
    kk = np.arange(1, 1001)
    rf = fgn_acvf(1001, H)[1:]
    phi = rf[0]
    axes[1].loglog(kk, rf, color=st.MainBlue, lw=1.6, label=f'fGn, H = {H}: hyperbolic decay, k^(2H-2)')
    axes[1].loglog(kk, phi ** kk, color=st.Orange, lw=1.6, label=f'AR(1), phi = {phi:.3f}: exponential decay')
    axes[1].set_ylim(1e-4, 1)
    axes[1].set_xlabel('lag k (log scale)')
    axes[1].set_ylabel('autocorrelation (log scale)')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_fgn_acf')
    out['ar1_phi'] = float(phi)
    out['ar1_rho10'] = float(phi ** 10)
    out['ar1_rho50'] = float(phi ** 50)
    out['fgn_rho100'] = float(rf[99])
    out['ar1_rho100'] = float(phi ** 100)
    return out


def fig_arfima(n=1000, ds=(0.4, -0.4), nlags=50, save=True):
    """ARFIMA(0,d,0) paths and ACF for d = 0.4 and d = -0.4 (as in SFEarfima), Gaussian white noise N(0, 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), gridspec_kw={'width_ratios': [2.2, 1]})
    k = np.arange(1, nlags + 1)
    out = {}
    for row, (d, c) in enumerate(zip(ds, [st.MainBlue, st.IDAred])):
        rng = np.random.default_rng(SEED + 7 + row)
        x = sim_arfima(n, d, rng)[0]
        axes[row, 0].plot(np.arange(n), x, color=c, lw=0.7, label=f'ARFIMA(0,d,0), d = {d}: path')
        axes[row, 0].set_ylabel('x(t)')
        th = arfima_acf(nlags + 1, d)[1:]
        xl = sim_arfima(20000, d, np.random.default_rng(SEED + 17 + row))[0]
        axes[row, 1].bar(k, acf(xl, nlags), color=c, alpha=0.55, width=0.8, label=f'd = {d}: sample ACF (n = 20000)')
        axes[row, 1].plot(k, th, color='black', lw=1.2, label='theoretical ACF' if row == 0 else '_')
        axes[row, 1].axhline(0, color='black', lw=0.6)
        out[str(d)] = {'rho1': float(th[0]), 'rho2': float(th[1]), 'rho10': float(th[9]), 'rho50': float(th[49]),
                       'sd': float(x.std()), 'var_th': arfima_var(d)}
    axes[1, 0].set_xlabel('time t')
    axes[1, 1].set_xlabel('lag k')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_arfima')
    return out


# =============================================================================
# WORKED R/S EXAMPLE AND THE NILE
# =============================================================================
RS_EXAMPLE = [0.5, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4, -0.7]


def rs_steps(x):
    """All steps of the R/S statistic of one short block: mean, deviations, cumulative deviations, R, S, R/S."""
    x = np.asarray(x, float)
    dev = x - x.mean()
    Y = np.cumsum(dev)
    R = Y.max() - Y.min()
    S = x.std()
    return {'x': x.tolist(), 'mean': float(x.mean()), 'dev': dev.tolist(), 'Y': Y.tolist(), 'max': float(Y.max()),
            'min': float(Y.min()), 'kmax': int(np.argmax(Y)) + 1, 'kmin': int(np.argmin(Y)) + 1, 'R': float(R),
            'S': float(S), 'RS': float(R / S), 'sumsq': float(dev @ dev)}


def fig_rs_example(x=RS_EXAMPLE, save=True):
    """The cumulative deviations Y_k of eight returns, with the range R marked."""
    s = rs_steps(x)
    k = np.arange(1, len(x) + 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(k, s['dev'], color=st.Teal, alpha=0.6, width=0.6, label='deviation from the mean, x_k - mean')
    ax.plot(k, s['Y'], 'o-', color=st.MainBlue, lw=1.6, label='cumulative deviation Y_k')
    ax.axhline(s['max'], color=st.Forest, ls='--', lw=1.0, label=f"max Y_k = {s['max']:.3f} (k = {s['kmax']})")
    ax.axhline(s['min'], color=st.IDAred, ls='--', lw=1.0, label=f"min Y_k = {s['min']:.3f} (k = {s['kmin']})")
    ax.annotate('', xy=(len(x) + 0.45, s['max']), xytext=(len(x) + 0.45, s['min']),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax.text(len(x) + 0.55, (s['max'] + s['min']) / 2, f"R = {s['R']:.3f}", color='black', va='center')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlim(0.4, len(x) + 1.4)
    ax.set_xticks(k)
    ax.set_xlabel('k')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_rs_example')
    return s


NILE_BREAK = 1898            # last year before the change point (Cobb 1978)


def fig_nile(save=True, reps=MC_REPS):
    """The Nile flow with the two regime means, and the R/S plot of the raw and of the break-adjusted series."""
    y = nile()
    N = len(y)
    sizes = block_sizes(N, nmin=6, nmax=N // 2, num=10)
    pre, post = y.loc[:NILE_BREAK], y.loc[NILE_BREAK + 1:]
    adj = pd.concat([pre - pre.mean(), post - post.mean()])
    rng = np.random.default_rng(SEED + 3)
    null = np.array([hurst_rs(rng.standard_normal(N), sizes) for _ in range(reps)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={'width_ratios': [1.6, 1]})
    axes[0].plot(y.index, y.values, color=st.MainBlue, lw=1.2, label='annual flow at Aswan (10^8 m^3)')
    axes[0].hlines(pre.mean(), pre.index[0], pre.index[-1], color=st.IDAred, lw=2, label=f'mean 1871-{NILE_BREAK}: {pre.mean():.0f}')
    axes[0].hlines(post.mean(), post.index[0], post.index[-1], color=st.Forest, lw=2, label=f'mean {NILE_BREAK + 1}-1970: {post.mean():.0f}')
    axes[0].set_xlabel('year')
    n1, rs1 = rs_curve(y.values, sizes)
    n2, rs2 = rs_curve(adj.values, sizes)
    H1 = float(np.polyfit(np.log10(n1), np.log10(rs1), 1)[0])
    H2 = float(np.polyfit(np.log10(n2), np.log10(rs2), 1)[0])
    axes[1].loglog(n1, rs1, 'o-', color=st.MainBlue, lw=1.2, label=f'raw series: H = {H1:.2f}')
    axes[1].loglog(n2, rs2, 's-', color=st.Orange, lw=1.2, label=f'regime means removed: H = {H2:.2f}')
    axes[1].loglog(n1, n1 ** 0.5 * rs1[0] / n1[0] ** 0.5, color='black', ls='--', lw=0.9, label='slope 0.5')
    axes[1].set_xlabel('block size n')
    axes[1].set_ylabel('(R/S)_n')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_nile')
    return {'N': N, 'H_raw': H1, 'H_adj': H2, 'dfa_raw': hurst_dfa(y.values, sizes), 'dfa_adj': hurst_dfa(adj.values, sizes),
            'mean_pre': float(pre.mean()), 'mean_post': float(post.mean()), 'q025': float(np.quantile(null, 0.025)),
            'q975': float(np.quantile(null, 0.975)), 'null_mean': float(null.mean()), 'first': int(y.index[0]), 'last': int(y.index[-1])}


# =============================================================================
# SCALING OF h-DAY RETURNS (SELF-SIMILARITY)
# =============================================================================
H_LIST = [1, 2, 5, 10, 20, 50, 100, 250]


def scaling(r, hs=H_LIST):
    """Standard deviation of non-overlapping h-day sums of returns; slope of log sd(h) on log h (aggregated
    standard deviation estimator of H)."""
    x = np.asarray(r, float)
    sd = []
    for h in hs:
        m = len(x) // h
        sd.append(x[:m * h].reshape(m, h).sum(axis=1).std())
    sd = np.array(sd)
    H = float(np.polyfit(np.log10(hs), np.log10(sd), 1)[0])
    return np.array(hs), sd, H


def fig_scaling(names=('sp500', 'bet', 'btc'), save=True):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    out = {}
    for k in names:
        hs, sd, H = scaling(returns(k))
        ax.loglog(hs, sd / sd[0], 'o-', color=COLORS[k], lw=1.4, label=f'{NAME[k]}: slope H = {H:.2f}')
        out[k] = {'H': H, 'sd1': float(sd[0]), 'sd10': float(sd[3]), 'sd250': float(sd[-1]),
                  'ratio10': float(sd[3] / sd[0]), 'ratio250': float(sd[-1] / sd[0])}
    hs = np.array(H_LIST)
    ax.loglog(hs, hs ** 0.5, color='black', ls='--', lw=1.0, label='square-root-of-time rule, slope 0.5')
    ax.set_xlabel('horizon h in days (log scale)')
    ax.set_ylabel('sd of h-day returns / sd of 1-day returns')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_scaling')
    return out


# =============================================================================
# MONTE CARLO: SMALL-SAMPLE BIAS, LO'S TEST, SPURIOUS LONG MEMORY
# =============================================================================
def mc_null(N, reps=MC_REPS, seed=SEED):
    """Distribution of the R/S, DFA and GPH estimates for i.i.d. Normal series of length N."""
    rng = np.random.default_rng(seed + N)
    rs, dfa, gp = [], [], []
    sizes = block_sizes(N)
    for _ in range(reps):
        x = rng.standard_normal(N)
        rs.append(hurst_rs(x, sizes))
        dfa.append(hurst_dfa(x, sizes))
        gp.append(gph(x)['H'])
    out = {}
    for k, v in [('rs', rs), ('dfa', dfa), ('gph', gp)]:
        v = np.array(v)
        out[k] = {'mean': float(v.mean()), 'sd': float(v.std()), 'q025': float(np.quantile(v, 0.025)),
                  'q975': float(np.quantile(v, 0.975))}
    return out


MC_N = [250, 500, 1000, 2500, 5000]


def fig_mc_bias(Ns=MC_N, reps=MC_REPS, save=True):
    """Mean and 95% Monte Carlo band of the three estimators under i.i.d. Normal returns, against N (Weron 2002)."""
    res = {N: mc_null(N, reps) for N in Ns}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    off = {'rs': -0.06, 'dfa': 0.0, 'gph': 0.06}
    for k, c, lab in [('rs', st.IDAred, 'R/S'), ('dfa', st.MainBlue, 'DFA'), ('gph', st.Forest, 'GPH, H = d + 0.5')]:
        x = np.log10(Ns) + off[k]
        m = np.array([res[N][k]['mean'] for N in Ns])
        lo = np.array([res[N][k]['q025'] for N in Ns])
        hi = np.array([res[N][k]['q975'] for N in Ns])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='o', color=c, capsize=4, lw=1.4, label=f'{lab}: mean and 95% band')
    ax.axhline(0.5, color='black', ls='--', lw=1.0, label='true value H = 0.5')
    ax.set_xticks(np.log10(Ns))
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel('sample size N (log scale)')
    ax.set_ylabel('estimated H')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_mc_bias')
    return {str(N): v for N, v in res.items()}


PHIS = [0.0, 0.1, 0.2, 0.3, 0.5]
HS_POWER = [0.5, 0.6, 0.7, 0.8]


def lo_rejections(N=1000, reps=MC_REPS, save=True):
    """Rejection rates (5%) of the classical (q = 0) and modified (Andrews q) R/S tests under AR(1) short memory and
    under fractional Gaussian noise (long memory)."""
    rng = np.random.default_rng(SEED + 11)
    ar = {}
    for phi in PHIS:
        e = rng.standard_normal((reps, N + 200))
        x = np.empty_like(e)
        x[:, 0] = e[:, 0]
        for t in range(1, N + 200):
            x[:, t] = phi * x[:, t - 1] + e[:, t]
        x = x[:, 200:]
        t = [lo_test(v) for v in x]
        ar[str(phi)] = {'classical': float(np.mean([a['reject0'] for a in t])), 'modified': float(np.mean([a['reject'] for a in t])),
                        'q_mean': float(np.mean([a['q'] for a in t]))}
    fg = {}
    for H in HS_POWER:
        X = sim_fgn(N, H, rng, size=reps)
        t = [lo_test(v) for v in X]
        fg[str(H)] = {'classical': float(np.mean([a['reject0'] for a in t])), 'modified': float(np.mean([a['reject'] for a in t])),
                      'q_mean': float(np.mean([a['q'] for a in t]))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    w = 0.35
    for ax, d, xs, xl in [(axes[0], ar, PHIS, 'AR(1) coefficient phi (short memory: H0 true)'),
                          (axes[1], fg, HS_POWER, 'Hurst exponent H of fGn (long memory if H > 0.5)')]:
        i = np.arange(len(xs))
        ax.bar(i - w / 2, [d[str(v)]['classical'] for v in xs], w, color=st.IDAred, label='classical R/S test (q = 0)')
        ax.bar(i + w / 2, [d[str(v)]['modified'] for v in xs], w, color=st.MainBlue, label="Lo's modified R/S test (Andrews q)")
        ax.axhline(0.05, color='black', ls='--', lw=1.0, label='nominal level 5%')
        ax.set_xticks(i)
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel(xl)
    axes[0].set_ylabel('share of rejections of short memory')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_lo_test')
    return {'N': N, 'ar': ar, 'fgn': fg}


DELTAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
PERS = [0.80, 0.90, 0.95, 0.98, 0.99]


def spurious(N=2000, reps=300, save=True):
    """Spurious long memory: (a) i.i.d. N(0,1) with a mean shift of delta at N/2; (b) absolute returns of GARCH(1,1)
    (alpha = 0.10, unconditional variance 1), a short-memory model, for several persistences alpha + beta."""
    rng = np.random.default_rng(SEED + 21)
    sizes = block_sizes(N)
    br = {}
    for dl in DELTAS:
        X = rng.standard_normal((reps, N))
        X[:, N // 2:] += dl
        br[str(dl)] = {'rs': float(np.mean([hurst_rs(v, sizes) for v in X])), 'dfa': float(np.mean([hurst_dfa(v, sizes) for v in X])),
                       'gph': float(np.mean([gph(v)['H'] for v in X]))}
    ga = {}
    for p in PERS:
        E = sim_garch(N, 1 - p, 0.10, p - 0.10, rng, size=reps)
        A = np.abs(E)
        ga[str(p)] = {'abs_dfa': float(np.mean([hurst_dfa(v, sizes) for v in A])), 'abs_gph': float(np.mean([gph(v)['H'] for v in A])),
                      'ret_dfa': float(np.mean([hurst_dfa(v, sizes) for v in E])),
                      'acf50': float(np.mean([acf(v, 50)[-1] for v in A])), 'acf50_th_factor': float(p ** 49)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, c, lab in [('rs', st.IDAred, 'R/S'), ('dfa', st.MainBlue, 'DFA'), ('gph', st.Forest, 'GPH, d + 0.5')]:
        axes[0].plot(DELTAS, [br[str(d)][k] for d in DELTAS], 'o-', color=c, lw=1.4, label=f'{lab}: mean estimate')
    axes[0].set_xlabel('mean shift at N/2 (in standard deviations)')
    axes[0].set_ylabel('estimated H')
    axes[1].plot(PERS, [ga[str(p)]['abs_dfa'] for p in PERS], 's-', color=st.Purple, lw=1.4, label='DFA of |r| (GARCH)')
    axes[1].plot(PERS, [ga[str(p)]['abs_gph'] for p in PERS], 'd-', color=st.Orange, lw=1.4, label='GPH of |r| (GARCH), d + 0.5')
    axes[1].plot(PERS, [ga[str(p)]['ret_dfa'] for p in PERS], '^-', color=st.Teal, lw=1.4, label='DFA of r (GARCH)')
    axes[1].set_xlabel('GARCH persistence alpha + beta (short memory)')
    for ax in axes:
        ax.axhline(0.5, color='black', ls='--', lw=1.0, label='H = 0.5 (no long memory)' if ax is axes[0] else '_')
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_spurious')
    return {'N': N, 'reps': reps, 'break': br, 'garch': ga}


# =============================================================================
# REAL DATA: LOG-LOG PLOTS, GPH, VOLATILITY MEMORY, SIX MARKETS, ROLLING H
# =============================================================================
def fig_rs_dfa(k='sp500', save=True, reps=MC_REPS):
    """R/S and DFA log-log plots of the returns and absolute returns, with the i.i.d. Monte Carlo mean."""
    r = returns(k)
    x, a = r.values, np.abs(r.values)
    N = len(x)
    sizes = block_sizes(N)
    rng = np.random.default_rng(SEED + 5)
    sims = rng.standard_normal((min(reps, 200), N))
    rs0 = np.mean([rs_curve(v, sizes)[1] for v in sims], axis=0)
    F0 = np.mean([dfa_curve(v, sizes)[1] for v in sims], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    out = {}
    for ax, curve, base, lab in [(axes[0], rs_curve, rs0, '(R/S)_n'), (axes[1], dfa_curve, F0, 'F(n) / F(10)')]:
        for y, c, m, nm in [(x, COLORS[k], 'o', 'returns r'), (a, st.Purple, 's', 'absolute returns |r|')]:
            n, v = curve(y, sizes)
            if lab.startswith('F'):
                v = v / v[0]
            sl = float(np.polyfit(np.log10(n), np.log10(v), 1)[0])
            meth = 'R/S' if lab.startswith('(') else 'DFA'
            ax.loglog(n, v, m, ms=4, color=c, label=f'{meth}, {nm}: slope {sl:.2f}')
            out[f"{'rs' if lab.startswith('(') else 'dfa'}_{'r' if nm.startswith('ret') else 'abs'}"] = sl
        b = base / base[0] if lab.startswith('F') else base
        ax.loglog(sizes, b, color='black', ls='--', lw=1.0, label='i.i.d. Normal (Monte Carlo mean)')
        ax.set_xlabel('block size n (log scale)')
        ax.set_ylabel(lab + ' (log scale)')
    axes[0].set_title('R/S analysis')
    axes[1].set_title('DFA')
    out['rs_iid'] = float(np.polyfit(np.log10(sizes), np.log10(rs0), 1)[0])
    out['dfa_iid'] = float(np.polyfit(np.log10(sizes), np.log10(F0), 1)[0])
    out['rs_n10'] = float(rs0[0])
    out['nmin'] = int(sizes[0])
    out['nmax'] = int(sizes[-1])
    out['nsizes'] = int(len(sizes))
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_rs_dfa')
    return out


def fig_gph(k='sp500', save=True):
    """The GPH log-periodogram regression for the returns and absolute returns."""
    r = returns(k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    out = {}
    for ax, y, c, nm in [(axes[0], r.values, COLORS[k], 'returns r'), (axes[1], np.abs(r.values), st.Purple, 'absolute returns |r|')]:
        lam, I = periodogram(y)
        g = gph(y)
        m = g['m']
        X = -np.log(4 * np.sin(lam[:m] / 2) ** 2)
        ax.plot(X, np.log(I[:m]), 'o', ms=3.5, color=c, label=f'{nm}: log periodogram, m = {m} frequencies')
        b = np.polyfit(X, np.log(I[:m]), 1)
        xx = np.linspace(X.min(), X.max(), 50)
        ax.plot(xx, np.polyval(b, xx), color='black', lw=1.2, label='_')
        ax.set_title(f"{nm}: slope d = {g['d']:.2f} (SE {g['se']:.2f})")
        ax.set_xlabel('-log(4 sin^2(lambda_j / 2))')
        ax.set_ylabel('log I(lambda_j)')
        out['r' if nm.startswith('ret') else 'abs'] = g
    axes[0].plot([], [], color='black', lw=1.2, label='OLS line: slope = d')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_gph')
    return out


def fig_vol_acf(k='sp500', nlags=250, save=True):
    """ACF of r, |r| and r^2 up to lag 250 (Ding, Granger and Engle 1993)."""
    r = returns(k)
    x = r.values
    fig, ax = plt.subplots(figsize=(10, 4.0))
    lags = np.arange(1, nlags + 1)
    out = {}
    for y, c, nm in [(x, COLORS[k], 'returns r'), (np.abs(x), st.Purple, 'absolute returns |r|'), (x ** 2, st.Orange, 'squared returns r^2')]:
        a = acf(y, nlags)
        ax.plot(lags, a, color=c, lw=1.2, label=nm)
        out[nm.split()[0]] = {'1': float(a[0]), '10': float(a[9]), '50': float(a[49]), '100': float(a[99]), '250': float(a[249]),
                              'npos': int(np.sum(a > 1.96 / np.sqrt(len(x))))}
    band = 1.96 / np.sqrt(len(x))
    ax.axhspan(-band, band, color=st.MainBlue, alpha=0.12, lw=0, label='95% band of an i.i.d. series')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('lag k (days)')
    ax.set_ylabel('autocorrelation')
    st.legend_outside_bottom(ax, ncol=4, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_vol_acf')
    out['band'] = float(band)
    out['n'] = int(len(x))
    rng = np.random.default_rng(SEED + 9)
    out['abs_shuffled_dfa'] = hurst_dfa(rng.permutation(np.abs(x)))
    return out


def markets(names=ASSETS, reps=MC_REPS):
    """All estimators for r and |r| of each series, the i.i.d. Monte Carlo band for its length, DFA of shuffled |r|."""
    out = {}
    for k in names:
        r = returns(k)
        x = r.values
        rng = np.random.default_rng(SEED + 31)
        out[k] = {'first': r.index[0].date().isoformat(), 'last': r.index[-1].date().isoformat(),
                  'r': all_estimates(x), 'abs': all_estimates(np.abs(x)), 'mc': mc_null(len(x), reps),
                  'abs_shuffled_dfa': hurst_dfa(rng.permutation(np.abs(x)))}
    return out


def fig_markets(M, save=True):
    """DFA exponents of r and |r| for six series, with the 95% band of i.i.d. series of the same length."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = list(M)
    i = np.arange(len(names))
    for j, k in enumerate(names):
        mc = M[k]['mc']['dfa']
        ax.add_patch(plt.Rectangle((j - 0.3, mc['q025']), 0.6, mc['q975'] - mc['q025'], color=st.MainBlue, alpha=0.15, lw=0,
                                   label='95% band of i.i.d. series of the same length' if j == 0 else '_'))
    ax.plot(i, [M[k]['r']['dfa'] for k in names], 'o', ms=8, color=st.IDAred, label='DFA exponent of returns r')
    ax.plot(i, [M[k]['abs']['dfa'] for k in names], 's', ms=8, color=st.Purple, label='DFA exponent of |r|')
    ax.plot(i, [M[k]['abs_shuffled_dfa'] for k in names], 'x', ms=8, color=st.Forest, label='DFA exponent of |r| after a random shuffle')
    ax.axhline(0.5, color='black', ls='--', lw=1.0, label='H = 0.5')
    ax.set_xticks(i)
    ax.set_xticklabels([NAME[k] for k in names])
    ax.set_ylabel('estimated H')
    st.legend_outside_bottom(ax, ncol=2, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_markets')


def rolling_hurst(x, window=WINDOW, step=STEP, est=hurst_dfa):
    """Hurst exponent on rolling windows; the value is dated at the last day of the window."""
    x = pd.Series(x)
    sizes = block_sizes(window)
    idx, val = [], []
    for e in range(window, len(x) + 1, step):
        idx.append(x.index[e - 1])
        val.append(est(x.values[e - window:e], sizes))
    return pd.Series(val, index=idx)


def fig_rolling(names=('sp500', 'bet', 'btc'), save=True, reps=MC_REPS):
    """Rolling DFA exponents of r and |r| (windows of 1000 observations moved by 21), with the i.i.d. 95% band."""
    band = mc_null(WINDOW, reps)['dfa']
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 5.0), sharex=True)
    out = {'band': band}
    for ax, k in zip(axes, names):
        r = returns(k)
        h = rolling_hurst(r)
        ha = rolling_hurst(np.abs(r))
        ax.axhspan(band['q025'], band['q975'], color=st.MainBlue, alpha=0.13, lw=0, label='95% band of i.i.d. series (N = 1000)' if k == names[0] else '_')
        ax.plot(h.index, h.values, color=COLORS[k], lw=1.3, label=f'{NAME[k]}: DFA exponent of r')
        ax.plot(ha.index, ha.values, color=st.Purple, lw=1.0, ls='-', label='DFA exponent of |r|' if k == names[0] else '_')
        ax.axhline(0.5, color='black', ls='--', lw=0.8)
        ax.set_ylabel('H')
        out[k] = {'n_windows': int(len(h)), 'first_end': h.index[0].date().isoformat(), 'last_end': h.index[-1].date().isoformat(),
                  'min': float(h.min()), 'max': float(h.max()), 'date_max': h.idxmax().date().isoformat(),
                  'date_min': h.idxmin().date().isoformat(), 'last': float(h.iloc[-1]),
                  'above': float(np.mean(h > band['q975'])), 'below': float(np.mean(h < band['q025'])),
                  'abs_min': float(ha.min()), 'abs_max': float(ha.max()), 'abs_last': float(ha.iloc[-1]),
                  'first_third': float(h.iloc[:len(h) // 3].mean()), 'last_third': float(h.iloc[-len(h) // 3:].mean())}
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch11_rolling')
    return out


if __name__ == '__main__':
    st.apply()
    N = {}
    N['rs_example'] = fig_rs_example()
    N['fbm'] = fig_fbm_paths()
    N['fgn'] = fig_fgn_acf()
    N['arfima'] = fig_arfima()
    N['nile'] = fig_nile()
    N['scaling'] = fig_scaling()
    N['rs_dfa'] = fig_rs_dfa()
    N['gph'] = fig_gph()
    N['vol_acf'] = fig_vol_acf()
    print('Monte Carlo ...')
    N['mc'] = fig_mc_bias()
    N['lo'] = lo_rejections()
    N['spurious'] = spurious()
    print('markets ...')
    M = markets()
    N['markets'] = M
    fig_markets(M)
    N['rolling'] = fig_rolling()
    N['end'] = returns('sp500').index[-1].date().isoformat()
    tab = pd.DataFrame({NAME[k]: {'N': v['r']['n'], 'R/S r': v['r']['rs'], 'DFA r': v['r']['dfa'], 'GPH d r': v['r']['gph_d'],
                                  'Lo V r': v['r']['lo_V'], 'Lo q r': v['r']['lo_q'], 'R/S |r|': v['abs']['rs'], 'DFA |r|': v['abs']['dfa'],
                                  'GPH d |r|': v['abs']['gph_d'], 'Lo V |r|': v['abs']['lo_V'],
                                  'DFA band 2.5%': v['mc']['dfa']['q025'], 'DFA band 97.5%': v['mc']['dfa']['q975']}
                        for k, v in M.items()}).T
    tab.to_csv(os.path.join(TABLE_DIR, 'ch11_memory_table.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch11_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
