"""
generate_all_charts.py -- charts and numbers of Chapter 16 (SFM): review, the course in one notebook
=====================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py), the arch package for GARCH(1,1)-t.
One common window for the three series of the course (S&P 500, BET, Bitcoin), 2 January 2015 to 18 September 2026,
and the key computations of Chapters 1-11, each with the definition used in its chapter:
  * performance   -- annualised mean and volatility with the actual frequency, CAGR, Sharpe ratio, maximum drawdown
                     (Chapter 1);
  * distribution  -- skewness, excess kurtosis, Jarque-Bera (Chapter 2); Hill tail index of the losses with
                     k = 2.5% of n (Chapter 5);
  * dependence    -- ACF of returns and of absolute returns, Ljung-Box with m = 10 lags on r and r^2 (Chapters 7-8);
                     the Lo-MacKinlay variance ratio VR(5) with the robust statistic Z*(5) (Chapter 7);
  * volatility    -- EWMA (lambda = 0.94) and GARCH(1,1) with Student-t innovations: persistence and half-life
                     (Chapters 8-9);
  * risk          -- VaR 1% (VaR_alpha = -q_alpha) and ES 2.5% by historical simulation and with the Normal
                     distribution; one-day VaR 1% forecasts by historical simulation (500 days) and EWMA-Normal since
                     2017, with the Kupiec test (Chapter 10);
  * memory        -- the Hurst exponent by R/S of r and |r| (Chapter 11).
Output: charts/sfm_ch16_*.pdf/.png, Quantlets/Ch_16/ch16_numbers.json and ch16_summary_table.csv
Based on Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., and on the chapters of the course.
Run:  python3 Quantlets/Ch_16/generate_all_charts.py
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
from sfm_data import load_close, log_returns, periods_per_year   # noqa: E402
import sfm_style as st                                           # noqa: E402
from arch import arch_model                                      # noqa: E402

TABLE_DIR = HERE
NAME = {'sp500': 'S&P 500', 'bet': 'BET', 'btc': 'Bitcoin'}
ASSETS = ['sp500', 'bet', 'btc']
COLORS = {'sp500': '#1A3A6E', 'bet': '#CD0000', 'btc': '#B5853F'}
START = '2015-01-01'          # common window of the three series (Bitcoin data start in September 2014)
END = '2026-09-18'
LB_LAGS = 10                  # Ljung-Box with m = 10 lags (Chapters 7-8)
VR_Q = 5                      # variance ratio over one week (Chapter 7)
HILL_FRAC = 0.025             # Hill estimator with k = 2.5% of n (Chapter 5)
LAMBDA = 0.94                 # EWMA (RiskMetrics) decay (Chapter 8)
ALPHA_VAR = 0.01              # VaR 1%
ALPHA_ES = 0.025              # ES 2.5%
WINDOW = 500                  # window of the rolling historical simulation (Chapter 10)
OOS_START = '2017-01-01'      # first day of the out-of-sample VaR forecasts
NMIN = 10                     # smallest block of the R/S statistic (Chapter 11)


# =============================================================================
# DATA AND PERFORMANCE (CHAPTER 1)
# =============================================================================
def returns(k, start=START, end=END):
    """Daily log returns in % on the series' own calendar (Bitcoin: 7 days a week)."""
    return log_returns(k, start, end)


def performance(k, start=START, end=END, rf=0.0):
    """Annualised mean log return and volatility with the actual frequency q, CAGR, Sharpe ratio (arithmetic mean,
    risk-free rate rf in % a year), maximum drawdown MDD = min_t (P_t / max_{s<=t} P_s - 1)."""
    p = load_close(k, start, end)
    r = 100 * np.log(p).diff().dropna()
    R = 100 * p.pct_change().dropna()
    q = periods_per_year(r)
    years = (p.index[-1] - p.index[0]).days / 365.25
    dd = p / p.cummax() - 1
    mean_ar = q * R.mean()
    vol = np.sqrt(q) * r.std()
    return {'n': int(len(r)), 'q': float(q), 'first': r.index[0].date().isoformat(), 'last': r.index[-1].date().isoformat(),
            'mean_log': float(q * r.mean()), 'mean_arith': float(mean_ar), 'vol': float(vol),
            'cagr': float(100 * ((p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1)),
            'sharpe': float((mean_ar - rf) / vol), 'sharpe_se': float(np.sqrt((1 + ((mean_ar - rf) / vol) ** 2 / 2) / years)),
            'mdd': float(100 * dd.min()), 'mdd_date': dd.idxmin().date().isoformat(),
            'peak_date': p.loc[:dd.idxmin()].idxmax().date().isoformat(), 'growth': float(p.iloc[-1] / p.iloc[0])}


# =============================================================================
# DISTRIBUTION AND TAILS (CHAPTERS 2 AND 5)
# =============================================================================
def moments(r):
    """Skewness S = m3 / m2^(3/2), excess kurtosis K = m4 / m2^2 - 3, Jarque-Bera JB = n/6 (S^2 + K^2/4) ~ chi2(2)."""
    x = np.asarray(r, float)
    e = x - x.mean()
    m2, m3, m4 = (np.mean(e ** j) for j in (2, 3, 4))
    S, K = m3 / m2 ** 1.5, m4 / m2 ** 2 - 3
    jb = len(x) / 6 * (S ** 2 + K ** 2 / 4)
    return {'skew': float(S), 'exkurt': float(K), 'jb': float(jb), 'p_jb': float(stats.chi2.sf(jb, 2))}


def hill(x, k):
    """Hill (1975): alpha_hat = [ (1/k) sum_{i=1..k} ln(X_(i) / X_(k+1)) ]^(-1), X_(1) >= X_(2) >= ..."""
    x = np.sort(np.asarray(x, float))[::-1]
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def tail_index(r, frac=HILL_FRAC):
    """Hill tail index of the losses L = -r with k = frac * n, and its i.i.d. standard error alpha / sqrt(k)."""
    L = -np.asarray(r, float)
    k = int(round(frac * len(L)))
    a = hill(L[L > 0], k)
    return {'k': k, 'alpha': a, 'se': a / np.sqrt(k)}


# =============================================================================
# DEPENDENCE AND EFFICIENCY (CHAPTERS 7 AND 8)
# =============================================================================
def acf(x, nlags):
    """Sample autocorrelations rho_hat(1..nlags) = sum (x_t - m)(x_{t-k} - m) / sum (x_t - m)^2."""
    e = np.asarray(x, float) - np.mean(x)
    s = np.sum(e ** 2)
    return np.array([np.sum(e[k:] * e[:-k]) / s for k in range(1, nlags + 1)])


def ljung_box(x, m=LB_LAGS):
    """Ljung-Box Q(m) = T(T+2) sum_{k=1..m} rho_k^2 / (T-k) ~ chi2(m) under no autocorrelation."""
    x = np.asarray(x, float)
    T = len(x)
    rho = acf(x, m)
    q = T * (T + 2) * np.sum(rho ** 2 / (T - np.arange(1, m + 1)))
    return {'q': float(q), 'p': float(stats.chi2.sf(q, m)), 'crit': float(stats.chi2.ppf(0.95, m)), 'rho1': float(rho[0])}


def variance_ratio(x, q=VR_Q):
    """Lo and MacKinlay (1988): overlapping VR(q) with bias corrections, the i.i.d. statistic Z(q) and the
    heteroskedasticity-robust Z*(q) = (VR - 1) / sqrt(theta / T), theta = sum_{j<q} [2(q-j)/q]^2 delta_hat(j)."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    var_a = np.sum(e ** 2) / (T - 1)
    agg = np.convolve(x, np.ones(q), mode='valid') - q * mu
    var_c = np.sum(agg ** 2) / (q * (T - q + 1) * (1 - q / T))
    vr = var_c / var_a
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    den = np.sum(e ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * T * np.sum(e[j:] ** 2 * e[:-j] ** 2) / den for j in range(1, q))
    zs = (vr - 1) / np.sqrt(theta / T)
    return {'q': q, 'vr': float(vr), 'z': float(z), 'zs': float(zs), 'p_zs': float(2 * stats.norm.sf(abs(zs)))}


# =============================================================================
# VOLATILITY (CHAPTERS 8 AND 9)
# =============================================================================
def ewma_var(r, lam=LAMBDA):
    """EWMA (RiskMetrics) variance sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) r_{t-1}^2, started at the variance of the
    first 30 returns; the value at t uses returns up to t-1 only."""
    x = np.asarray(r, float)
    s2 = np.empty(len(x))
    s2[0] = np.var(x[:30])
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(s2, index=r.index)


def garch_t(r):
    """GARCH(1,1) with constant mean and standardised Student-t innovations (maximum likelihood, arch package):
    persistence alpha + beta and half-life ln 0.5 / ln(alpha + beta) in days (None when alpha + beta is 1 to four
    decimals: an integrated GARCH, IGARCH)."""
    res = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off', options={'maxiter': 2000})
    p = res.params
    pers = float(p['alpha[1]'] + p['beta[1]'])
    return {'mu': float(p['mu']), 'omega': float(p['omega']), 'alpha': float(p['alpha[1]']), 'beta': float(p['beta[1]']),
            'nu': float(p['nu']), 'pers': pers, 'half_life': float(np.log(0.5) / np.log(pers)) if pers < 0.9999 else None}


# =============================================================================
# RISK MEASURES AND BACKTESTING (CHAPTER 10)
# =============================================================================
def hs_var(x, a=ALPHA_VAR):
    """Historical simulation: VaR_a = -r_(k), k = ceil(n a), the k-th smallest return."""
    x = np.sort(np.asarray(x, float))
    return float(-x[int(np.ceil(len(x) * a - 1e-9)) - 1])


def hs_es(x, a=ALPHA_ES):
    """Historical simulation: ES_a = minus the mean of the k = ceil(n a) smallest returns."""
    x = np.sort(np.asarray(x, float))
    return float(-x[:int(np.ceil(len(x) * a - 1e-9))].mean())


def normal_var(x, a=ALPHA_VAR):
    """Normal VaR_a = -(mu + sigma z_a)."""
    return float(-(np.mean(x) + np.std(x, ddof=1) * stats.norm.ppf(a)))


def normal_es(x, a=ALPHA_ES):
    """Normal ES_a = -mu + sigma phi(z_a) / a."""
    return float(-np.mean(x) + np.std(x, ddof=1) * stats.norm.pdf(stats.norm.ppf(a)) / a)


def kupiec(x, n, a=ALPHA_VAR):
    """Kupiec (1995): LR_uc = -2 ln[(1-a)^(n-x) a^x / ((1-x/n)^(n-x) (x/n)^x)] ~ chi2(1)."""
    ph = x / n
    l0 = (n - x) * np.log(1 - a) + x * np.log(a)
    l1 = ((n - x) * np.log(1 - ph) if ph < 1 else 0.0) + (x * np.log(ph) if x > 0 else 0.0)
    lr = float(-2 * (l0 - l1))
    return {'x': int(x), 'n': int(n), 'rate': float(100 * ph), 'lr': lr, 'p': float(stats.chi2.sf(lr, 1))}


def var_backtest(k, start=OOS_START, window=WINDOW, a=ALPHA_VAR):
    """One-day VaR 1% forecasts from `start`: historical simulation on the last `window` returns and EWMA-Normal
    (-sigma_t z_a, lambda = 0.94); exceptions r_t < -VaR_t, Kupiec test, most exceptions in 250 days."""
    r = log_returns(k, None, END)
    s2 = ewma_var(r)
    hs = (-r.rolling(window).quantile(a, interpolation='lower')).shift(1)
    ew = -np.sqrt(s2) * stats.norm.ppf(a)
    df = pd.DataFrame({'r': r, 'HS': hs, 'EWMA': ew}).loc[start:].dropna()
    out = {'n': int(len(df)), 'first': df.index[0].date().isoformat()}
    for m in ('HS', 'EWMA'):
        hit = (df['r'] < -df[m]).astype(int)
        out[m] = kupiec(int(hit.sum()), len(df), a)
        out[m]['max250'] = int(hit.rolling(250).sum().max())
        out[m]['mean_var'] = float(df[m].mean())
    return out, df


# =============================================================================
# LONG MEMORY (CHAPTER 11)
# =============================================================================
def block_sizes(N, nmin=NMIN, num=20):
    """About `num` block sizes spaced evenly on a log scale between nmin and N/4."""
    return np.unique(np.floor(np.logspace(np.log10(nmin), np.log10(N // 4), num)).astype(int))


def hurst_rs(x):
    """Hurst exponent: slope of log10 of the average R/S of the non-overlapping blocks of size n on log10 n."""
    x = np.asarray(x, float)
    sizes, out = block_sizes(len(x)), []
    for n in sizes:
        m = len(x) // n
        X = x[:m * n].reshape(m, n)
        Y = np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)
        R, S = Y.max(axis=1) - Y.min(axis=1), X.std(axis=1)
        out.append(np.mean(R[S > 0] / S[S > 0]))
    return float(np.polyfit(np.log10(sizes), np.log10(out), 1)[0])


# =============================================================================
# THE COURSE IN ONE TABLE
# =============================================================================
def summary(k):
    """All key numbers of one series on the common window."""
    r = returns(k)
    out = {'perf': performance(k), 'mom': moments(r), 'hill': tail_index(r), 'lb_r': ljung_box(r),
           'lb_r2': ljung_box(r ** 2), 'acf1_abs': float(acf(np.abs(r), 1)[0]), 'vr': variance_ratio(r),
           'garch': garch_t(r), 'var_hs': hs_var(r), 'var_n': normal_var(r), 'es_hs': hs_es(r), 'es_n': normal_es(r),
           'H_r': hurst_rs(r), 'H_abs': hurst_rs(np.abs(r))}
    out['ewma_half'] = float(np.log(0.5) / np.log(LAMBDA))
    out['gap_n'] = float(100 * (1 - out['var_n'] / out['var_hs']))
    return out


def summary_table(S):
    """The course in one table (rows: quantities, columns: series)."""
    rows = {}
    for k in ASSETS:
        s = S[k]
        rows[NAME[k]] = {
            'observations': s['perf']['n'], 'obs per year': s['perf']['q'], 'mean log return % p.a.': s['perf']['mean_log'],
            'volatility % p.a.': s['perf']['vol'], 'Sharpe (rf = 0)': s['perf']['sharpe'], 'MDD %': s['perf']['mdd'],
            'skewness': s['mom']['skew'], 'excess kurtosis': s['mom']['exkurt'], 'Jarque-Bera': s['mom']['jb'],
            'Hill alpha (k = 2.5%)': s['hill']['alpha'], 'Ljung-Box r (10)': s['lb_r']['q'],
            'Ljung-Box r^2 (10)': s['lb_r2']['q'], 'VR(5)': s['vr']['vr'], 'Z*(5)': s['vr']['zs'],
            'GARCH alpha + beta': s['garch']['pers'], 'VaR 1% HS': s['var_hs'], 'VaR 1% Normal': s['var_n'],
            'ES 2.5% HS': s['es_hs'], 'Hurst R/S r': s['H_r'], 'Hurst R/S |r|': s['H_abs']}
    return pd.DataFrame(rows)


# =============================================================================
# CHARTS
# =============================================================================
def fig_three_series(save=True):
    """Growth of 100 invested on 2 January 2015 (log scale) and the drawdown of each series."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    for k in ASSETS:
        p = load_close(k, START, END)
        a1.plot(p.index, 100 * p / p.iloc[0], color=COLORS[k], lw=1.2, label=NAME[k])
        a2.plot(p.index, 100 * (p / p.cummax() - 1), color=COLORS[k], lw=1.0)
    a1.set_yscale('log')
    a1.set_ylabel('Value of 100 (log scale)')
    a2.set_ylabel('Drawdown (%)')
    a2.axhline(0, color='black', lw=0.5)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch16_three_series')
    else:
        plt.show()


def fig_stylised(lags=50, save=True):
    """ACF of daily returns and of absolute returns, lags 1-50, with the 95% band +-1.96/sqrt(T)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, k in zip(axes, ASSETS):
        r = returns(k)
        band = 1.96 / np.sqrt(len(r))
        lg = np.arange(1, lags + 1)
        ax.bar(lg - 0.2, acf(r, lags), width=0.4, color='#1A3A6E', label='returns $r_t$')
        ax.bar(lg + 0.2, acf(np.abs(r), lags), width=0.4, color='#CD0000', label='absolute returns $|r_t|$')
        ax.axhspan(-band, band, color='#B5853F', alpha=0.18, lw=0)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(NAME[k])
        ax.set_xlabel('lag (days)')
    axes[0].set_ylabel('autocorrelation')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch16_stylised')
    else:
        plt.show()


def fig_backtest(k='bet', save=True):
    """Daily returns since 2017 with minus the one-day VaR 1% of historical simulation (500 days) and EWMA-Normal;
    exceptions marked."""
    bt, df = var_backtest(k)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(df.index, df['r'], color='#1A3A6E', lw=0.5, label=f'{NAME[k]} daily return (%)')
    ax.plot(df.index, -df['HS'], color='#B5853F', lw=1.2, label='$-$VaR 1%, historical simulation (500 days)')
    ax.plot(df.index, -df['EWMA'], color='#CD0000', lw=1.0, label='$-$VaR 1%, EWMA-Normal ($\\lambda = 0.94$)')
    for m, c, mk in (('HS', '#B5853F', 'o'), ('EWMA', '#CD0000', 'x')):
        h = df[df['r'] < -df[m]]
        ax.scatter(h.index, h['r'], color=c, marker=mk, s=18, zorder=3,
                   label=f"exceptions {'HS' if m == 'HS' else 'EWMA'}: {bt[m]['x']} of {bt['n']}")
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch16_backtest')
    else:
        plt.show()
    return bt


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


if __name__ == '__main__':
    st.apply()
    N = {'start': START, 'end': END}
    N['summary'] = {k: summary(k) for k in ASSETS}
    summary_table(N['summary']).round(3).to_csv(os.path.join(TABLE_DIR, 'ch16_summary_table.csv'))
    fig_three_series()
    fig_stylised()
    N['backtest'] = {'bet': fig_backtest('bet')}
    for k in ('sp500', 'btc'):
        N['backtest'][k] = var_backtest(k)[0]
    with open(os.path.join(HERE, 'ch16_numbers.json'), 'w') as f:
        json.dump(to_json(N), f, indent=1)
    print(summary_table(N['summary']).round(3))
    print(json.dumps(to_json(N['backtest']), indent=1))
