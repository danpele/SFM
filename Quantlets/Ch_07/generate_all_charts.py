"""
generate_all_charts.py -- charts and numbers of Chapter 7 (SFM): efficient markets, random walk, unit roots and
variance-ratio tests
================================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
  * motivation   -- today's return against yesterday's (S&P 500, BET); simulated random walks against the S&P 500;
  * white noise  -- Gaussian white noise and its sample ACF (as in SFEtimewn); theoretical and sample ACF of AR(1),
                    AR(2), MA(1) and MA(2) processes (as in SFEacfar1, SFEacfar2, SFEacfma1, SFEacfma2);
  * ACF tests    -- sample ACF of daily returns with the i.i.d. and the heteroskedasticity-robust 95% bands, ACF of
                    absolute returns, Box-Pierce and Ljung-Box statistics (classic and robust), the runs test;
  * unit roots   -- ADF, Phillips-Perron and KPSS on log prices and on returns; the spurious regression of two
                    independent random walks (Granger and Newbold, 1974);
  * variance ratio -- Lo and MacKinlay (1988) VR(q) with the homoskedastic Z(q) and the robust Z*(q); the size of both
                    under a GARCH martingale; the Chow-Denning (1993) multiple test; the automatic VR test of Choi (1999)
                    with the wild bootstrap of Kim (2009); VR profiles; developed, emerging and crypto markets;
  * adaptive markets -- rolling VR(5) on 500-day windows (Lo, 2004), sub-periods of the BET, Bitcoin and the S&P 500;
  * anomalies    -- day-of-week and January effects on the S&P 500 and the BET;
  * AI section   -- the BET before and after its upgrade to Secondary Emerging market (FTSE Russell, September 2020).
Output: charts/sfm_ch7_*.pdf/.png, Quantlets/Ch_07/ch7_numbers.json, ch7_tests_table.csv, ch7_vr_table.csv
Based on the Quantlets SFEtimewn, SFEacfar1, SFEacfar2, SFEacfma1 and SFEacfma2 (github.com/QuantLet/SFE), on
Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 11, and on Campbell, Lo and
MacKinlay (1997), The Econometrics of Financial Markets, Ch. 2.
Run:  python3 Quantlets/Ch_07/generate_all_charts.py
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
from sfm_data import SERIES, log_returns, load_close   # noqa: E402
import sfm_style as st                                 # noqa: E402
import statsmodels.api as sm                           # noqa: E402
from statsmodels.tsa.stattools import adfuller, kpss   # noqa: E402
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp   # noqa: E402

TABLE_DIR = HERE
# series that are not in the common list of sfm_data.py (EODHD symbols in data/market)
EXTRA_SERIES = {'bvsp': ('BVSP.INDX', 'Bovespa', 'Equity', 'close', '2000-01-01'),
                'nifty': ('NSEI.INDX', 'Nifty 50', 'Equity', 'close', '2000-01-01'),
                'bist': ('XU100.INDX', 'BIST 100', 'Equity', 'close', '2000-01-01'),
                'ssec': ('SSEC.INDX', 'Shanghai Composite', 'Equity', 'close', '2000-01-01'),
                'kospi': ('KS11.INDX', 'KOSPI', 'Equity', 'close', '2000-01-01'),
                'ipc': ('MXX.INDX', 'IPC Mexico', 'Equity', 'close', '2000-01-01')}
NAME = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'eth': 'Ethereum', 'stoxx50': 'Euro Stoxx 50',
        'nikkei': 'Nikkei 225', 'tlv': 'Banca Transilvania', 'snp': 'OMV Petrom', 'brd': 'BRD', 'tgn': 'Transgaz',
        'wig20': 'WIG20', 'bux': 'BUX', 'px': 'PX', 'bvsp': 'Bovespa', 'nifty': 'Nifty 50', 'bist': 'BIST 100',
        'ssec': 'Shanghai Composite', 'kospi': 'KOSPI', 'ipc': 'IPC Mexico'}
START = {'sp500': '1990-01-01', 'dax': '1990-01-01', 'bet': '1997-01-01', 'btc': None,
         'tlv': '2010-01-01', 'snp': '2010-01-01', 'brd': '2010-01-01', 'tgn': '2010-01-01'}
ASSETS = ['sp500', 'dax', 'bet', 'btc']
STOCKS = ['tlv', 'snp', 'brd', 'tgn']
DEVELOPED = ['sp500', 'dax', 'stoxx50', 'nikkei']
EMERGING = ['bet', 'wig20', 'bux', 'px', 'bist', 'bvsp', 'ipc', 'nifty', 'kospi', 'ssec']
CRYPTO = ['btc', 'eth']
MARKETS_START = '2016-09-19'     # cross-market comparison: the last ten years of data
# Banca Transilvania: the adjustment for the bonus shares of May 2016 is applied one day late in the adjusted close
# (log returns of -15% and +20% on two consecutive days, see Chapter 2); the two days are removed.
BAD_DAYS = {'tlv': ['2016-05-30', '2016-05-31']}
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F',
          'snp': '#2E7D32', 'brd': '#8E44AD', 'tlv': '#E67E22', 'tgn': '#DC3545'}
QS = (2, 5, 10, 20)        # horizons of the variance-ratio tests (days)
LB_LAGS = 10               # Box-Pierce / Ljung-Box statistics with m = 10 lags
WINDOW = 500               # rolling windows of 500 trading days (about two years)
STEP = 21                  # a new window every 21 trading days (about one month)
SEED = 2026
B_BOOT = 500               # wild-bootstrap replications of the automatic VR test
FTSE_UPGRADE = '2020-09-21'   # Romania: Secondary Emerging market in the FTSE indices from September 2020


# =============================================================================
# DATA
# =============================================================================
def returns(k, start=None):
    """Daily log returns in % on the series' own calendar (from `start`, else the default start of the series);
    known data errors removed."""
    if k in EXTRA_SERIES:
        SERIES.setdefault(k, EXTRA_SERIES[k])
    r = log_returns(k, start or START.get(k))
    if k in BAD_DAYS:
        r = r.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return r


def log_price(k, start=None):
    """Natural logarithm of the closing price (adjusted close for stocks)."""
    if k in EXTRA_SERIES:
        SERIES.setdefault(k, EXTRA_SERIES[k])
    p = np.log(load_close(k, start or START.get(k)))
    if k in BAD_DAYS:            # the same two days as in returns(): the late adjustment cancels out
        p = p.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return p


# =============================================================================
# AUTOCORRELATION AND PORTMANTEAU TESTS
# =============================================================================
def acf(x, nlags):
    """Sample autocorrelations rho_hat(1..nlags) = sum (x_t - m)(x_{t-k} - m) / sum (x_t - m)^2."""
    e = np.asarray(x, float) - np.mean(x)
    s = np.sum(e ** 2)
    return np.array([np.sum(e[k:] * e[:-k]) / s for k in range(1, nlags + 1)])


def acf_fft(x, nlags=None):
    """All sample autocorrelations rho_hat(1..nlags) by the fast Fourier transform (used by the automatic VR test)."""
    e = np.asarray(x, float) - np.mean(x)
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n]
    return (c / c[0])[1:(nlags or n - 1) + 1]


def robust_var_acf(x, nlags):
    """delta_hat(k) / T: the variance of rho_hat(k) that allows for conditional heteroskedasticity, with
    delta_hat(k) = T sum e_t^2 e_{t-k}^2 / (sum e_t^2)^2 (Lo and MacKinlay, 1988); under i.i.d. returns it is 1/T."""
    e = np.asarray(x, float) - np.mean(x)
    s2 = np.sum(e ** 2) ** 2
    return np.array([np.sum(e[k:] ** 2 * e[:-k] ** 2) / s2 for k in range(1, nlags + 1)])


def portmanteau(x, m=LB_LAGS):
    """Box-Pierce Q = T sum rho^2, Ljung-Box Q = T(T+2) sum rho^2/(T-k) and the robust Q~ = sum rho^2/var_rob(k),
    each against chi-square(m)."""
    x = np.asarray(x, float)
    T = len(x)
    rho = acf(x, m)
    vr = robust_var_acf(x, m)
    k = np.arange(1, m + 1)
    bp = T * np.sum(rho ** 2)
    lb = T * (T + 2) * np.sum(rho ** 2 / (T - k))
    rob = np.sum(rho ** 2 / vr)
    p = lambda q: float(stats.chi2.sf(q, m))
    return {'T': T, 'm': m, 'rho1': float(rho[0]), 'bp': float(bp), 'p_bp': p(bp), 'lb': float(lb), 'p_lb': p(lb),
            'rob': float(rob), 'p_rob': p(rob), 'crit': float(stats.chi2.ppf(0.95, m))}


def runs_test(x):
    """Runs test (Wald and Wolfowitz, 1940) on the signs of the returns (zero returns dropped): number of runs R,
    E[R] = 2 n1 n2 / n + 1, Var[R] = 2 n1 n2 (2 n1 n2 - n) / (n^2 (n - 1)), z = (R - E[R]) / sd(R)."""
    s = np.sign(np.asarray(x, float))
    s = s[s != 0]
    n1, n2 = int(np.sum(s > 0)), int(np.sum(s < 0))
    n = n1 + n2
    runs = int(1 + np.sum(s[1:] != s[:-1]))
    mean = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mean) / np.sqrt(var)
    return {'n1': n1, 'n2': n2, 'runs': runs, 'mean': mean, 'sd': float(np.sqrt(var)), 'z': float(z),
            'p': float(2 * stats.norm.sf(abs(z)))}


# =============================================================================
# UNIT-ROOT TESTS
# =============================================================================
def adf_test(x, regression='c'):
    """Augmented Dickey-Fuller test (Said and Dickey, 1984): lag order by AIC, at most 12 (T/100)^(1/4);
    p-values and critical values of MacKinnon (1996). H0: unit root."""
    stat, p, lags, nobs, crit, _ = adfuller(np.asarray(x, float), regression=regression, autolag='AIC')
    return {'stat': float(stat), 'p': float(p), 'lags': int(lags), 'nobs': int(nobs), 'crit5': float(crit['5%'])}


def pp_test(x, regression='c'):
    """Phillips-Perron (1988) Z_t test: Dickey-Fuller regression without lags, t-statistic corrected with a
    Newey-West long-run variance (Bartlett weights, l = 12 (T/100)^(1/4) rounded up). H0: unit root."""
    y = np.asarray(x, float)
    dy, ylag = y[1:], y[:-1]
    T = len(dy)
    X = [np.ones(T), ylag] if regression == 'c' else [np.ones(T), np.arange(1, T + 1), ylag]
    X = np.column_stack(X)
    b, *_ = np.linalg.lstsq(X, dy, rcond=None)
    u = dy - X @ b
    k = X.shape[1]
    s2 = np.sum(u ** 2) / (T - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1])
    t_rho = (b[-1] - 1) / se
    lags = int(np.ceil(12 * (T / 100) ** 0.25))
    g0 = np.sum(u ** 2) / T
    lam2 = g0 + 2 * sum((1 - j / (lags + 1)) * np.sum(u[j:] * u[:-j]) / T for j in range(1, lags + 1))
    z = np.sqrt(g0 / lam2) * t_rho - (lam2 - g0) / (2 * np.sqrt(lam2)) * T * se / np.sqrt(s2)
    return {'stat': float(z), 'p': float(mackinnonp(z, regression=regression, N=1)), 'lags': lags,
            'crit5': float(mackinnoncrit(N=1, regression=regression, nobs=T)[1])}


def kpss_test(x, regression='c'):
    """KPSS test (Kwiatkowski, Phillips, Schmidt and Shin, 1992), Bartlett lag l = 12 (T/100)^(1/4) rounded up.
    H0: stationarity (around a constant, or around a trend for regression='ct'). p-values are tabulated
    between 0.01 and 0.10 only."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, p, lags, crit = kpss(np.asarray(x, float), regression=regression, nlags='legacy')
    return {'stat': float(stat), 'p': float(p), 'lags': int(lags), 'crit5': float(crit['5%'])}


def unit_root_table(names=ASSETS + STOCKS):
    """ADF, PP and KPSS on log prices (constant and trend) and on returns (constant)."""
    out = {}
    for k in names:
        p, r = log_price(k).values, returns(k).values
        out[k] = {'price': {'adf': adf_test(p, 'ct'), 'pp': pp_test(p, 'ct'), 'kpss': kpss_test(p, 'ct')},
                  'ret': {'adf': adf_test(r, 'c'), 'pp': pp_test(r, 'c'), 'kpss': kpss_test(r, 'c')},
                  'n': len(r), 'first': returns(k).index[0].date().isoformat()}
    return out


# =============================================================================
# VARIANCE-RATIO TESTS
# =============================================================================
def variance_ratio(x, q):
    """Lo and MacKinlay (1988): overlapping VR(q) with their bias corrections, the homoskedastic statistic
    Z(q) = (VR - 1) / sqrt(2(2q-1)(q-1)/(3qT)) and the heteroskedasticity-robust Z*(q) = (VR - 1) / sqrt(theta/T),
    theta = sum_{j=1}^{q-1} [2(q-j)/q]^2 delta_hat(j)."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    var_a = np.sum(e ** 2) / (T - 1)
    agg = np.convolve(x, np.ones(q), mode='valid') - q * mu          # overlapping q-day returns minus q mu
    m = q * (T - q + 1) * (1 - q / T)
    var_c = np.sum(agg ** 2) / m
    vr = var_c / var_a
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    den = np.sum(e ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * T * np.sum(e[j:] ** 2 * e[:-j] ** 2) / den for j in range(1, q))
    zs = (vr - 1) / np.sqrt(theta / T)
    return {'q': q, 'vr': float(vr), 'z': float(z), 'zs': float(zs), 'p_z': float(2 * stats.norm.sf(abs(z))),
            'p_zs': float(2 * stats.norm.sf(abs(zs))), 'se_rob': float(np.sqrt(theta / T)),
            'se_iid': float(np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T)))}


def chow_denning(x, qs=QS, alpha=0.05):
    """Chow and Denning (1993): CD = max_i |Z*(q_i)|; critical value and p-value from the studentized maximum
    modulus distribution with m = len(qs) and infinite degrees of freedom: P(CD <= c) = (2 Phi(c) - 1)^m."""
    zs = [variance_ratio(x, q)['zs'] for q in qs]
    cd = float(np.max(np.abs(zs)))
    m = len(qs)
    crit = float(stats.norm.ppf(1 - (1 - (1 - alpha) ** (1 / m)) / 2))
    return {'cd': cd, 'p': float(1 - (2 * stats.norm.cdf(cd) - 1) ** m), 'crit': crit, 'm': m}


def qs_kernel(z):
    """Quadratic spectral kernel of Andrews (1991)."""
    z = np.asarray(z, float)
    a = 6 * np.pi * z / 5
    return 25 / (12 * np.pi ** 2 * z ** 2) * (np.sin(a) / a - np.cos(a))


def auto_vr_stat(x):
    """Automatic VR test of Choi (1999): VR(k) = 1 + 2 sum_i m(i/k) rho_hat(i) with the quadratic spectral kernel m,
    the horizon k chosen from the data (Andrews, 1991, AR(1) plug-in: k = 1.3221 (a2 T)^(1/5),
    a2 = 4 rho^2 / (1 - rho)^4), and AVR = sqrt(T/k) (VR(k) - 1) / sqrt(2), N(0, 1) under i.i.d. returns."""
    e = np.asarray(x, float) - np.mean(x)
    T = len(e)
    rho = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1] ** 2)
    k = 1.3221 * (4 * rho ** 2 / (1 - rho) ** 4 * T) ** 0.2
    r = acf_fft(e)
    vr = 1 + 2 * np.sum(qs_kernel(np.arange(1, T) / k) * r)
    return vr, k, np.sqrt(T / k) * (vr - 1) / np.sqrt(2)


def auto_vr(x, B=B_BOOT, seed=SEED):
    """Choi (1999) automatic VR test with the wild-bootstrap p-value of Kim (2009): x*_t = eta_t (x_t - mean),
    eta_t standard Normal, the statistic (with its horizon k) recomputed on each bootstrap sample."""
    x = np.asarray(x, float)
    vr, k, stat = auto_vr_stat(x)
    rng = np.random.default_rng(seed)
    e = x - x.mean()
    boot = np.array([auto_vr_stat(rng.standard_normal(len(e)) * e)[2] for _ in range(B)])
    return {'vr': float(vr), 'k': float(k), 'stat': float(stat), 'p_asy': float(2 * stats.norm.sf(abs(stat))),
            'p_boot': float(np.mean(np.abs(boot) >= abs(stat)))}


def vr_table(names=ASSETS + STOCKS, start=None):
    """VR(q), Z(q), Z*(q) for q in QS; the Chow-Denning test; the automatic VR test; rho_hat(1)."""
    out = {}
    for k in names:
        r = returns(k, start)
        x = r.values
        out[k] = {'n': len(x), 'first': r.index[0].date().isoformat(), 'rho1': float(acf(x, 1)[0]),
                  'vr': {q: variance_ratio(x, q) for q in QS}, 'cd': chow_denning(x), 'avr': auto_vr(x)}
    return out


def simulate_garch(T, omega=0.04, a=0.12, b=0.86, burn=500, rng=None):
    """A GARCH(1,1) martingale difference: e_t = sigma_t z_t, sigma_t^2 = omega + a e_{t-1}^2 + b sigma_{t-1}^2,
    z_t standard Normal (unconditional variance omega / (1 - a - b) = 2)."""
    rng = rng or np.random.default_rng(SEED)
    z = rng.standard_normal(T + burn)
    e = np.empty(T + burn)
    s2 = omega / (1 - a - b)
    for t in range(T + burn):
        e[t] = np.sqrt(s2) * z[t]
        s2 = omega + a * e[t] ** 2 + b * s2
    return e[burn:]


def vr_size(T=2000, reps=1000, qs=(2, 5, 10), seed=SEED):
    """Rejection rates at 5% of Z(q) and Z*(q) when the null of no predictability is TRUE: i.i.d. Normal returns
    and GARCH(1,1) returns (uncorrelated, but with volatility clustering)."""
    rng = np.random.default_rng(seed)
    out = {'iid': {q: [0, 0] for q in qs}, 'garch': {q: [0, 0] for q in qs}}
    for _ in range(reps):
        for lab, x in [('iid', rng.standard_normal(T)), ('garch', simulate_garch(T, rng=rng))]:
            for q in qs:
                v = variance_ratio(x, q)
                out[lab][q][0] += abs(v['z']) > 1.96
                out[lab][q][1] += abs(v['zs']) > 1.96
    return {lab: {q: {'z': c[0] / reps, 'zs': c[1] / reps} for q, c in d.items()} for lab, d in out.items()}


# =============================================================================
# CHARTS: MOTIVATION
# =============================================================================
def fig_scatter_lag(names=('sp500', 'bet'), save=True):
    """Today's return against yesterday's return, with the least-squares line and the correlation rho_hat(1)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    out = {}
    for ax, k in zip(axes, names):
        x = returns(k).values
        ax.scatter(x[:-1], x[1:], s=4, alpha=0.35, color=COLORS[k], label=f'{NAME[k]}: daily returns (r(t-1), r(t))')
        b = np.polyfit(x[:-1], x[1:], 1)
        g = np.linspace(-10, 10, 50)
        ax.plot(g, b[1] + b[0] * g, color='black', lw=1.4, label=f'{NAME[k]}: least-squares line')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('Return yesterday, r(t-1) (%)')
        ax.set_ylabel('Return today, r(t) (%)')
        rho = acf(x, 1)[0]
        ax.set_title(f'{NAME[k]}: correlation {rho:.3f}', loc='left', fontsize=12)
        out[k] = {'rho1': float(rho), 'slope': float(b[0]), 'n': len(x)}
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_scatter_lag')
    return out


def fig_rw_paths(k='sp500', n_sim=4, seed=SEED, save=True):
    """The S&P 500 log price (minus its first value) against random walks with i.i.d. Normal steps of the same
    mean and standard deviation (RW1)."""
    lp = log_price(k)
    x = np.diff(lp.values)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [st.IDAred, st.Forest, st.Purple, st.Orange, st.Teal]
    sims = []
    for i in range(n_sim):
        s = np.r_[0, np.cumsum(rng.normal(x.mean(), x.std(ddof=1), len(x)))]
        sims.append(s)
        ax.plot(lp.index, s, color=cols[i], lw=1.0, label=f'Simulated random walk {i + 1}')
    ax.plot(lp.index, lp.values - lp.values[0], color=COLORS[k], lw=1.6, label=f'{NAME[k]} (log price minus its first value)')
    ax.set_ylabel('Log price (start = 0)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_rw_paths')
    return {'mu': float(x.mean()), 'sd': float(x.std(ddof=1)), 'n': len(x), 'first': lp.index[0].date().isoformat(),
            'final': float(lp.values[-1] - lp.values[0]), 'sim_final': [float(s[-1]) for s in sims]}


# =============================================================================
# CHARTS: WHITE NOISE AND THE ACF (SFEtimewn, SFEacfar1/2, SFEacfma1/2)
# =============================================================================
def fig_white_noise(n=1000, lags=30, seed=10, save=True):
    """Gaussian white noise, n = 1000 (as in SFEtimewn), and its sample ACF with the 95% band +/- 1.96/sqrt(n)."""
    rng = np.random.default_rng(seed)
    wn = rng.standard_normal(n)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.6, 1]})
    axes[0].plot(np.arange(1, n + 1), wn, color=st.MainBlue, lw=0.6, label='Gaussian white noise, N(0, 1)')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    r = acf(wn, lags)
    axes[1].bar(np.arange(1, lags + 1), r, color=st.IDAred, width=0.5, label='Sample ACF')
    b = 1.96 / np.sqrt(n)
    axes[1].axhline(b, color='black', ls='--', lw=0.8, label='95% band: +/- 1.96/sqrt(n)')
    axes[1].axhline(-b, color='black', ls='--', lw=0.8)
    axes[1].axhline(0, color='black', lw=0.6)
    axes[1].set_xlabel('Lag k')
    axes[1].set_ylabel('rho_hat(k)')
    axes[1].set_ylim(-0.2, 0.2)
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_white_noise')
    return {'n': n, 'band': b, 'outside': int(np.sum(np.abs(r) > b)), 'lags': lags}


def arma_acf(ar=(), ma=(), lags=30):
    """Theoretical ACF of a stationary ARMA process x_t = sum a_i x_{t-i} + e_t + sum b_j e_{t-j} (via its MA(inf)
    weights, 2000 terms)."""
    psi = np.zeros(2000)
    psi[0] = 1.0
    for i in range(1, len(psi)):
        psi[i] = (ma[i - 1] if i - 1 < len(ma) else 0) + sum(ar[j] * psi[i - 1 - j] for j in range(len(ar)) if i - 1 - j >= 0)
    g = np.array([np.sum(psi[:len(psi) - k] * psi[k:]) for k in range(lags + 1)])
    return g[1:] / g[0]


def simulate_arma(ar=(), ma=(), n=1000, burn=200, rng=None):
    """x_t = sum a_i x_{t-i} + e_t + sum b_j e_{t-j}, e_t standard Normal."""
    rng = rng or np.random.default_rng(SEED)
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(n + burn):
        x[t] = e[t] + sum(ar[i] * x[t - 1 - i] for i in range(len(ar)) if t - 1 - i >= 0) \
            + sum(ma[j] * e[t - 1 - j] for j in range(len(ma)) if t - 1 - j >= 0)
    return x[burn:]


ACF_CASES = [('AR(1), alpha = 0.9', (0.9,), ()), ('AR(1), alpha = -0.9', (-0.9,), ()),
             ('AR(2), alpha1 = 0.5, alpha2 = 0.4', (0.5, 0.4), ()), ('MA(1), beta = 0.5', (), (0.5,)),
             ('MA(1), beta = -0.5', (), (-0.5,)), ('MA(2), beta1 = 0.5, beta2 = 0.4', (), (0.5, 0.4))]


def fig_acf_processes(lags=20, n=1000, seed=123, save=True):
    """Theoretical ACF (bars) and sample ACF of a simulated path of n = 1000 (points) for the AR(1), AR(2), MA(1)
    and MA(2) processes of the SFE Quantlets SFEacfar1, SFEacfar2, SFEacfma1 and SFEacfma2."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    out = {}
    for ax, (lab, ar, ma) in zip(axes.flat, ACF_CASES):
        th = arma_acf(ar, ma, lags)
        sm_ = acf(simulate_arma(ar, ma, n, rng=rng), lags)
        k = np.arange(1, lags + 1)
        ax.bar(k, th, color=st.MainBlue, width=0.55, label='Theoretical ACF')
        ax.plot(k, sm_, 'o', ms=3.5, color=st.IDAred, label='Sample ACF, n = 1000')
        ax.axhline(0, color='black', lw=0.6)
        ax.set_title(lab, loc='left', fontsize=11)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Lag k')
        out[lab] = [float(v) for v in th[:5]]
    axes[0, 0].set_ylabel('rho(k)')
    axes[1, 0].set_ylabel('rho(k)')
    st.fig_legend_bottom(fig, handles=axes[0, 0].get_legend_handles_labels()[0],
                         labels=axes[0, 0].get_legend_handles_labels()[1], ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_acf_processes')
    return out


def fig_acf_returns(names=ASSETS, lags=20, save=True):
    """Sample ACF of daily returns (lags 1-20) with the i.i.d. band +/- 1.96/sqrt(T) and the heteroskedasticity-robust
    band +/- 1.96 sqrt(delta_hat(k)/T)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    out = {}
    k = np.arange(1, lags + 1)
    for ax, n in zip(axes.flat, names):
        x = returns(n).values
        r = acf(x, lags)
        rb = 1.96 * np.sqrt(robust_var_acf(x, lags))
        b = 1.96 / np.sqrt(len(x))
        ax.bar(k, r, color=COLORS[n], width=0.55, label='Sample ACF of the returns')
        ax.fill_between(k, -rb, rb, color=st.Teal, alpha=0.18, lw=0, label='Robust 95% band')
        ax.axhline(b, color='black', ls='--', lw=0.8, label='i.i.d. 95% band')
        ax.axhline(-b, color='black', ls='--', lw=0.8)
        ax.axhline(0, color='black', lw=0.6)
        ax.set_title(NAME[n], loc='left', fontsize=12)
        ax.set_ylim(-0.2, 0.2)
        out[n] = {'rho': [float(v) for v in r[:5]], 'band_iid': float(b), 'band_rob1': float(rb[0]),
                  'out_iid': int(np.sum(np.abs(r) > b)), 'out_rob': int(np.sum(np.abs(r) > rb))}
    for ax in axes[1]:
        ax.set_xlabel('Lag k (days)')
    st.fig_legend_bottom(fig, handles=axes[0, 0].get_legend_handles_labels()[0],
                         labels=axes[0, 0].get_legend_handles_labels()[1], ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_acf_returns')
    return out


def fig_acf_abs(names=('sp500', 'bet', 'btc'), lags=100, save=True):
    """ACF of the returns and of the absolute returns, lags 1-100: returns almost uncorrelated, |returns| strongly
    and persistently correlated (volatility clustering)."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    k = np.arange(1, lags + 1)
    for n in names:
        x = returns(n).values
        ra = acf(np.abs(x), lags)
        r = acf(x, lags)
        ax.plot(k, ra, color=COLORS[n], lw=1.6, label=f'{NAME[n]}: |returns|')
        ax.plot(k, r, color=COLORS[n], lw=0.9, ls=':', label=f'{NAME[n]}: returns')
        out[n] = {'abs1': float(ra[0]), 'abs20': float(ra[19]), 'abs100': float(ra[99]), 'ret1': float(r[0])}
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('Lag k (days)')
    ax.set_ylabel('Sample autocorrelation')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_acf_abs')
    return out


def tests_table(names=ASSETS + STOCKS):
    """Portmanteau tests (m = 10) on returns and on squared returns, and the runs test, for each series."""
    out = {}
    for k in names:
        x = returns(k).values
        out[k] = {'ret': portmanteau(x), 'sq': portmanteau(x ** 2), 'runs': runs_test(x), 'n': len(x)}
    return out


# =============================================================================
# CHARTS: UNIT ROOTS
# =============================================================================
def fig_unit_root(k='bet', save=True):
    """Log price and daily returns of one series, with the ADF and KPSS statistics of each."""
    lp, r = log_price(k), returns(k)
    a_p, k_p = adf_test(lp.values, 'ct'), kpss_test(lp.values, 'ct')
    a_r, k_r = adf_test(r.values, 'c'), kpss_test(r.values, 'c')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(lp.index, lp.values, color=COLORS.get(k, st.MainBlue), lw=1.1, label=f'{NAME[k]}: log price')
    axes[0].set_title(f'Log price: ADF {a_p["stat"]:.2f}, KPSS {k_p["stat"]:.2f}', loc='left', fontsize=11)
    axes[1].plot(r.index, r.values, color=st.MainBlue, lw=0.4, label=f'{NAME[k]}: daily log return (%)')
    axes[1].set_title(f'Returns: ADF {a_r["stat"]:.1f}, KPSS {k_r["stat"]:.2f}', loc='left', fontsize=11)
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig(f'sfm_ch7_unit_root_{k}')
    return {'price': {'adf': a_p, 'kpss': k_p}, 'ret': {'adf': a_r, 'kpss': k_r}}


def spurious(T=500, reps=2000, seed=SEED):
    """Regress one random walk on another, independent one: share of |t| > 1.96 and median R^2
    (Granger and Newbold, 1974); the same for their increments (white noise)."""
    rng = np.random.default_rng(seed)
    res = {'level': [], 'diff': []}
    keep = None
    for i in range(reps):
        e1, e2 = rng.standard_normal(T), rng.standard_normal(T)
        y, x = np.cumsum(e1), np.cumsum(e2)
        for lab, (yy, xx) in [('level', (y, x)), ('diff', (e1, e2))]:
            X = np.column_stack([np.ones(T), xx])
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            u = yy - X @ b
            se = np.sqrt(np.sum(u ** 2) / (T - 2) * np.linalg.inv(X.T @ X)[1, 1])
            r2 = 1 - np.sum(u ** 2) / np.sum((yy - yy.mean()) ** 2)
            res[lab].append((b[1] / se, r2))
        if i == 0:
            keep = (y, x)
    out = {lab: {'reject': float(np.mean([abs(t) > 1.96 for t, _ in v])), 'r2_med': float(np.median([r for _, r in v])),
                 'beyond40': float(np.mean([abs(t) > 40 for t, _ in v])),
                 't': [float(t) for t, _ in v]} for lab, v in res.items()}
    out['example'] = keep
    return out


def fig_spurious(sp, save=True):
    """Two independent random walks, and the distribution of the t-statistic of the slope when one is regressed on
    the other (levels) and on the increments."""
    y, x = sp['example']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(y, color=st.MainBlue, lw=1.1, label='Random walk y(t)')
    axes[0].plot(x, color=st.IDAred, lw=1.1, label='Independent random walk x(t)')
    axes[0].set_xlabel('t')
    bins = np.linspace(-40, 40, 81)                     # |t| > 40 (a few regressions in levels) not drawn
    axes[1].hist(sp['level']['t'], bins=bins, density=True, color=st.Amber, alpha=0.8,
                 label='t-statistic, regression in levels')
    axes[1].hist(sp['diff']['t'], bins=bins, density=True, color=st.Forest, alpha=0.7, label='t-statistic, regression in increments')
    g = np.linspace(-40, 40, 400)
    axes[1].plot(g, stats.norm.pdf(g), color='black', lw=1, label='N(0, 1)')
    axes[1].set_xlabel('t-statistic of the slope')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_spurious')


# =============================================================================
# CHARTS: VARIANCE RATIO
# =============================================================================
def vr_profile(x, qmax=40):
    """VR(q) and its robust standard error for q = 2..qmax."""
    qs = np.arange(2, qmax + 1)
    v = [variance_ratio(x, q) for q in qs]
    return qs, np.array([a['vr'] for a in v]), np.array([a['se_rob'] for a in v]), np.array([a['se_iid'] for a in v])


def fig_vr_profile(names=ASSETS, qmax=40, save=True):
    """VR(q), q = 2..40, with the robust and the i.i.d. 95% bands around 1."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), sharex=True)
    out = {}
    for ax, k in zip(axes.flat, names):
        qs, vr, se, se0 = vr_profile(returns(k).values, qmax)
        ax.plot(qs, vr, color=COLORS[k], lw=1.8, marker='o', ms=2.5, label='VR(q)')
        ax.fill_between(qs, 1 - 1.96 * se, 1 + 1.96 * se, color=st.Teal, alpha=0.18, lw=0, label='Robust 95% band (Z*)')
        ax.plot(qs, 1 + 1.96 * se0, color='black', ls='--', lw=0.8, label='i.i.d. 95% band (Z)')
        ax.plot(qs, 1 - 1.96 * se0, color='black', ls='--', lw=0.8)
        ax.axhline(1, color='black', lw=0.6)
        ax.set_title(NAME[k], loc='left', fontsize=12)
        out[k] = {'vr40': float(vr[-1]), 'vrmin': float(vr.min()), 'vrmax': float(vr.max())}
    for ax in axes[1]:
        ax.set_xlabel('Horizon q (days)')
    for ax in axes[:, 0]:
        ax.set_ylabel('VR(q)')
    st.fig_legend_bottom(fig, handles=axes[0, 0].get_legend_handles_labels()[0],
                         labels=axes[0, 0].get_legend_handles_labels()[1], ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_vr_profile')
    return out


def fig_vr_size(sz, save=True):
    """Rejection rates at 5% of Z(q) and Z*(q) under i.i.d. and GARCH(1,1) returns (the null hypothesis is true)."""
    qs = list(sz['iid'])
    fig, ax = plt.subplots(figsize=(10, 3.9))
    w = 0.2
    xs = np.arange(len(qs))
    for i, (lab, key, col) in enumerate([('i.i.d. returns, Z', ('iid', 'z'), st.MainBlue), ('i.i.d. returns, Z*', ('iid', 'zs'), st.Teal),
                                         ('GARCH returns, Z', ('garch', 'z'), st.IDAred), ('GARCH returns, Z*', ('garch', 'zs'), st.Amber)]):
        ax.bar(xs + (i - 1.5) * w, [100 * sz[key[0]][q][key[1]] for q in qs], width=w * 0.95, color=col, label=lab)
    ax.axhline(5, color='black', ls='--', lw=0.9, label='Nominal level: 5%')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'q = {q}' for q in qs])
    ax.set_ylabel('Rejections of a true null (%)')
    st.legend_outside_bottom(ax, ncol=5, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_vr_size')


def markets_table(names=DEVELOPED + EMERGING + CRYPTO, start=MARKETS_START, q=5):
    """VR(5) with the robust Z*, rho_hat(1) and the Chow-Denning p-value over the last ten years of data."""
    out = {}
    for k in names:
        x = returns(k, start).values
        v = variance_ratio(x, q)
        out[k] = {'vr': v['vr'], 'zs': v['zs'], 'se': v['se_rob'], 'rho1': float(acf(x, 1)[0]), 'cd_p': chow_denning(x)['p'],
                  'n': len(x), 'group': 'Developed' if k in DEVELOPED else 'Crypto' if k in CRYPTO else 'Emerging'}
    return out


def fig_vr_markets(mt, save=True):
    """VR(5) with its robust 95% interval, by market, over the last ten years."""
    keys = sorted(mt, key=lambda k: mt[k]['vr'])
    gcol = {'Developed': st.MainBlue, 'Emerging': st.IDAred, 'Crypto': st.Amber}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, k in enumerate(keys):
        m = mt[k]
        ax.errorbar(m['vr'], i, xerr=1.96 * m['se'], fmt='o', color=gcol[m['group']], ms=5, capsize=3, lw=1.2)
    for g, c in gcol.items():
        ax.plot([], [], 'o', color=c, label=f'{g} markets')
    ax.axvline(1, color='black', lw=0.8, ls='--', label='VR = 1 (random walk)')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NAME[k] for k in keys])
    ax.set_xlabel('VR(5) with robust 95% interval')
    st.legend_outside_bottom(ax, ncol=4, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_vr_markets')


# =============================================================================
# ADAPTIVE MARKETS: ROLLING AND SUB-PERIOD TESTS
# =============================================================================
def rolling_vr(k, q=5, window=WINDOW, step=STEP):
    """VR(q) and Z*(q) on rolling windows of `window` days, one every `step` days (dated at the window end)."""
    r = returns(k)
    x = r.values
    rows = []
    for end in range(window, len(x) + 1, step):
        v = variance_ratio(x[end - window:end], q)
        rows.append((r.index[end - 1], v['vr'], v['zs']))
    return pd.DataFrame(rows, columns=['date', 'vr', 'zs']).set_index('date')


def fig_rolling_vr(names=('sp500', 'bet', 'btc'), q=5, save=True):
    """Rolling VR(5) and Z*(5) on 500-day windows (adaptive markets: efficiency changes over time)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    out = {}
    for k in names:
        d = rolling_vr(k, q)
        axes[0].plot(d.index, d['vr'], color=COLORS[k], lw=1.3, label=NAME[k])
        axes[1].plot(d.index, d['zs'], color=COLORS[k], lw=1.1)
        out[k] = {'share_rej': float(np.mean(np.abs(d['zs']) > 1.96)), 'share_pos': float(np.mean(d['zs'] > 1.96)),
                  'share_neg': float(np.mean(d['zs'] < -1.96)), 'n_win': len(d), 'vr_min': float(d['vr'].min()),
                  'vr_max': float(d['vr'].max()), 'date_max': d['vr'].idxmax().date().isoformat(),
                  'date_min': d['vr'].idxmin().date().isoformat(), 'first': d.index[0].date().isoformat()}
    axes[0].axhline(1, color='black', lw=0.7, ls='--')
    axes[0].set_ylabel(f'VR({q})')
    axes[1].axhline(1.96, color='black', lw=0.7, ls='--')
    axes[1].axhline(-1.96, color='black', lw=0.7, ls='--')
    axes[1].set_ylabel(f'Robust Z*({q})')
    st.fig_legend_bottom(fig, handles=axes[0].get_legend_handles_labels()[0], labels=axes[0].get_legend_handles_labels()[1], ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_rolling_vr')
    return out


SUBPERIODS = {'bet': [('1997-01-01', '2006-12-31'), ('2007-01-01', '2016-12-31'), ('2017-01-01', None)],
              'sp500': [('1990-01-01', '2001-12-31'), ('2002-01-01', '2013-12-31'), ('2014-01-01', None)],
              'btc': [('2014-01-01', '2017-12-31'), ('2018-01-01', '2021-12-31'), ('2022-01-01', None)]}


def subperiods(table=SUBPERIODS, q=5):
    """rho_hat(1), VR(5), Z*(5) and the Chow-Denning p-value in three sub-periods."""
    out = {}
    for k, per in table.items():
        r = returns(k)
        out[k] = []
        for a, b in per:
            x = r.loc[a:b].values
            v = variance_ratio(x, q)
            out[k].append({'from': r.loc[a:b].index[0].date().isoformat(), 'to': r.loc[a:b].index[-1].date().isoformat(),
                           'n': len(x), 'rho1': float(acf(x, 1)[0]), 'vr': v['vr'], 'zs': v['zs'], 'cd_p': chow_denning(x)['p']})
    return out


def bet_upgrade(edge=FTSE_UPGRADE, years=5, q=5):
    """The BET in the five years before and after its upgrade to Secondary Emerging market (September 2020)."""
    r = returns('bet')
    e = pd.Timestamp(edge)
    out = {}
    for lab, a, b in [('before', e - pd.DateOffset(years=years), e - pd.Timedelta(days=1)), ('after', e, e + pd.DateOffset(years=years))]:
        x = r.loc[a:b].values
        v = variance_ratio(x, q)
        out[lab] = {'from': r.loc[a:b].index[0].date().isoformat(), 'to': r.loc[a:b].index[-1].date().isoformat(), 'n': len(x),
                    'rho1': float(acf(x, 1)[0]), 'vr': v['vr'], 'zs': v['zs'], 'se': v['se_rob'], 'cd_p': chow_denning(x)['p']}
    out['z_diff'] = (out['after']['vr'] - out['before']['vr']) / np.sqrt(out['after']['se'] ** 2 + out['before']['se'] ** 2)
    return out


# =============================================================================
# CALENDAR ANOMALIES
# =============================================================================
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def calendar(k):
    """Mean daily return by weekday and January against the other months; Wald tests with Newey-West (HAC)
    standard errors (5 lags) in regressions on dummies."""
    r = returns(k)
    d = pd.get_dummies(r.index.dayofweek).astype(float).values
    res = sm.OLS(r.values, d).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    R = np.column_stack([np.ones(4), -np.eye(4)])               # Monday = Tuesday = ... = Friday
    w = res.wald_test(R, scalar=True)
    jan = (r.index.month == 1).astype(float)
    rj = sm.OLS(r.values, sm.add_constant(jan)).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    means = [float(r[r.index.dayofweek == i].mean()) for i in range(5)]
    ses = [float(v) for v in res.bse]
    return {'means': means, 'ses': ses, 'wald_p': float(w.pvalue), 'jan_mean': float(r[jan == 1].mean()),
            'other_mean': float(r[jan == 0].mean()), 'jan_t': float(rj.tvalues[1]), 'jan_p': float(rj.pvalues[1]), 'n': len(r)}


def fig_calendar(names=('sp500', 'bet'), save=True):
    """Mean daily return by weekday with HAC 95% intervals."""
    fig, ax = plt.subplots(figsize=(10, 3.9))
    out = {}
    w = 0.35
    for i, k in enumerate(names):
        c = calendar(k)
        xs = np.arange(5) + (i - 0.5) * w
        ax.bar(xs, c['means'], width=w * 0.92, color=COLORS[k], alpha=0.85, label=f'{NAME[k]}: mean return (Wald p = {c["wald_p"]:.2f})')
        ax.errorbar(xs, c['means'], yerr=1.96 * np.array(c['ses']), fmt='none', ecolor='black', capsize=3, lw=1)
        out[k] = c
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(DAYS)
    ax.set_ylabel('Mean daily log return (%)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch7_calendar')
    return out


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
    N['scatter'] = fig_scatter_lag()
    N['rw'] = fig_rw_paths()
    N['wn'] = fig_white_noise()
    N['arma'] = fig_acf_processes()
    N['acf'] = fig_acf_returns()
    N['acf_abs'] = fig_acf_abs()
    N['tests'] = tests_table()
    N['ur'] = unit_root_table()
    N['ur_fig'] = fig_unit_root('bet')
    sp = spurious()
    fig_spurious(sp)
    N['spurious'] = {lab: {c: sp[lab][c] for c in ['reject', 'r2_med', 'beyond40']} for lab in ['level', 'diff']}
    N['vr'] = vr_table()
    N['vr_profile'] = fig_vr_profile()
    sz = vr_size()
    fig_vr_size(sz)
    N['vr_size'] = sz
    mt = markets_table()
    fig_vr_markets(mt)
    N['markets'] = mt
    N['rolling'] = fig_rolling_vr()
    N['sub'] = subperiods()
    N['calendar'] = fig_calendar()
    N['upgrade'] = bet_upgrade()
    N['end'] = returns('sp500').index[-1].date().isoformat()
    N['cd_crit'] = chow_denning(np.random.default_rng(1).standard_normal(500))['crit']
    N['df_crit'] = {'c': float(mackinnoncrit(N=1, regression='c', nobs=10 ** 6)[1]),
                    'ct': float(mackinnoncrit(N=1, regression='ct', nobs=10 ** 6)[1])}
    tt = pd.DataFrame({NAME[k]: {'n': v['n'], 'rho1': v['ret']['rho1'], 'LB(10)': v['ret']['lb'], 'p LB': v['ret']['p_lb'],
                                 'robust Q(10)': v['ret']['rob'], 'p robust': v['ret']['p_rob'],
                                 'LB(10) squared': v['sq']['lb'], 'runs z': v['runs']['z'], 'p runs': v['runs']['p']}
                       for k, v in N['tests'].items()}).T
    tt.to_csv(os.path.join(TABLE_DIR, 'ch7_tests_table.csv'), float_format='%.4f')
    vt = pd.DataFrame({NAME[k]: {**{f'VR({q})': v['vr'][q]['vr'] for q in QS}, **{f'Z*({q})': v['vr'][q]['zs'] for q in QS},
                                 'CD': v['cd']['cd'], 'p CD': v['cd']['p'], 'AVR': v['avr']['stat'], 'k AVR': v['avr']['k'],
                                 'p AVR (wild bootstrap)': v['avr']['p_boot']} for k, v in N['vr'].items()}).T
    vt.to_csv(os.path.join(TABLE_DIR, 'ch7_vr_table.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch7_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1, default=str)
    print(tt.round(3))
    print(vt.round(3))
    print(json.dumps(to_json({k: N[k] for k in ['vr_size', 'sub', 'upgrade', 'spurious', 'rolling', 'cd_crit', 'df_crit']}), indent=1)[:5000])
