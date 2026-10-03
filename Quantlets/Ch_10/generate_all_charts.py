"""
generate_all_charts.py -- charts and numbers of Chapter 10 (SFM): VaR, ES and backtesting
==========================================================================================
Course data (sfm_data.py), chart style (sfm_style.py), the arch package for the GARCH filters.
Every number on the slides comes from here. Convention: VaR at level alpha is VaR_alpha = -q_alpha(r), the loss
exceeded with probability alpha ("VaR 1%"); ES_alpha is the average loss beyond the alpha-quantile ("ES 2.5%").
  * definitions  -- VaR 1% and ES 2.5% on the daily S&P 500 returns since 2000;
  * methods      -- historical simulation, Normal, Student-t, Cornish-Fisher, EVT (peaks over threshold) on six
                    series (as in the SFE Quantlets VaRest and SFEVaRqqplot); VaR across tail probabilities;
  * conditional  -- GARCH(1,1)-t VaR and filtered historical simulation (FHS) for the next day (Chapter 9);
  * rolling      -- one-day VaR 1%, VaR 2.5% and ES 2.5% forecasts since 2005 (Bitcoin since 2017) from historical
                    simulation (500 days), Normal (500 days), GARCH(1,1)-t and FHS, re-estimated every 250 days (as in
                    SFEVaRtimeplot and SFEVaRbank); the 2008 and 2020 stress periods;
  * backtesting  -- exceedance counts, Kupiec POF, Christoffersen independence and conditional coverage, the Basel
                    traffic light (as in SFEvar_pot_backtesting), the Acerbi-Szekely Z2 test of ES;
  * portfolio    -- BET and S&P 500 (50/50): variance-covariance VaR, correlation against tail dependence, Gaussian
                    and t copulas, portfolio VaR and ES by copula simulation.
Output: charts/sfm_ch10_*.pdf/.png, Quantlets/Ch_10/ch10_numbers.json and ch10_*_table.csv
Based on the Quantlets VaRest, SFEVaRbank, SFEVaRtimeplot, SFEVaRqqplot and SFEvar_pot_backtesting
(github.com/QuantLet/SFE), on Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 16-18,
and on McNeil, Frey and Embrechts (2015), Quantitative Risk Management, Ch. 2 and 7.
Run:  python3 Quantlets/Ch_10/generate_all_charts.py
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
from sfm_data import log_returns, simple_returns, load_close, periods_per_year   # noqa: E402
import sfm_style as st                               # noqa: E402
from arch import arch_model                          # noqa: E402

TABLE_DIR = HERE
NAME = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania',
        'snp': 'OMV Petrom', 'brd': 'BRD'}
START = {'sp500': '2000-01-01', 'dax': '2000-01-01', 'bet': '2000-01-01', 'btc': None,
         'tlv': '2010-01-01', 'snp': '2010-01-01', 'brd': '2010-01-01'}
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']
BT_ASSETS = ['sp500', 'dax', 'bet', 'btc']
# Banca Transilvania: the adjustment for the bonus shares of May 2016 is applied one day late in the adjusted close
# (Chapter 2); the two days are removed.
BAD_DAYS = {'tlv': ['2016-05-30', '2016-05-31']}
COLORS = {'sp500': '#1A3A6E', 'dax': '#17A2B8', 'bet': '#CD0000', 'btc': '#B5853F', 'tlv': '#E67E22', 'snp': '#2E7D32'}
EPISODES = {'2008': ('2008-09-01', '2009-03-31'), '2020': ('2020-02-15', '2020-05-31')}
ALPHA_VAR = 0.01             # VaR 1%
ALPHA_ES = 0.025             # ES 2.5% (Basel FRTB)
WINDOW = 500                 # rolling window of historical simulation and of the Normal method (about two years)
REFIT = 250                  # the GARCH filters are re-estimated every 250 observations (expanding window)
OOS_START = {'sp500': '2005-01-01', 'dax': '2005-01-01', 'bet': '2005-01-01', 'btc': '2017-01-01'}
POT_Q = 0.90                 # EVT threshold: the 90% quantile of the losses (McNeil and Frey, 2000), as in Chapter 5
METHODS = ['HS', 'Normal', 'GARCH-t', 'FHS']
MCOL = {'HS': '#B5853F', 'Normal': '#2E7D32', 'GARCH-t': '#CD0000', 'FHS': '#1A3A6E'}
PORT = ('bet', 'sp500')      # the portfolio of Section 6: BET and S&P 500, 50% each
WEIGHTS = (0.5, 0.5)
N_SIM = 200000               # copula simulations
N_Z2 = 1000                  # simulations of the Acerbi-Szekely Z2 statistic under the null
SEED = 2026


# =============================================================================
# DATA
# =============================================================================
def returns(k, start=None, end=None):
    """Daily log returns in % on the series' own calendar; known data errors removed."""
    r = log_returns(k, start or START.get(k)) if end is None else log_returns(k, start or START.get(k), end)
    if k in BAD_DAYS:
        r = r.drop(pd.to_datetime(BAD_DAYS[k]), errors='ignore')
    return r


# =============================================================================
# VaR AND ES: THE METHODS
# =============================================================================
def hs_var(x, a=ALPHA_VAR):
    """Historical simulation: VaR_a = -r_(k), the k-th smallest return, k = ceil(n a)."""
    xs = np.sort(np.asarray(x, float))
    k = int(np.ceil(len(xs) * a - 1e-9))
    return float(-xs[k - 1])


def hs_es(x, a=ALPHA_ES):
    """Historical simulation: ES_a = minus the average of the k smallest returns, k = ceil(n a)."""
    xs = np.sort(np.asarray(x, float))
    k = int(np.ceil(len(xs) * a - 1e-9))
    return float(-xs[:k].mean())


def normal_var(mu, sd, a=ALPHA_VAR):
    """VaR_a = -(mu + sd z_a), z_a the a-quantile of N(0, 1) (negative)."""
    return float(-(mu + sd * stats.norm.ppf(a)))


def normal_es(mu, sd, a=ALPHA_ES):
    """ES_a = -mu + sd phi(z_a) / a."""
    return float(-mu + sd * stats.norm.pdf(stats.norm.ppf(a)) / a)


def t_es_factor(nu, a=ALPHA_ES):
    """ES of a Student-t variable with nu degrees of freedom (scale 1): f(t_a) (nu + t_a^2) / ((nu - 1) a)."""
    q = stats.t.ppf(a, nu)
    return float(stats.t.pdf(q, nu) * (nu + q ** 2) / ((nu - 1) * a))


def std_t_q(nu, a=ALPHA_VAR):
    """a-quantile of the standardised Student-t (variance 1): t_nu^{-1}(a) sqrt((nu - 2)/nu)."""
    return float(stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu))


def std_t_es(nu, a=ALPHA_ES):
    """ES_a of the standardised Student-t (variance 1)."""
    return float(np.sqrt((nu - 2) / nu) * t_es_factor(nu, a))


def t_fit(x):
    """Student-t with location m, scale s and nu degrees of freedom, fitted by maximum likelihood (scipy)."""
    nu, m, s = stats.t.fit(np.asarray(x, float))
    return float(nu), float(m), float(s)


def t_var_es(x, a_var=ALPHA_VAR, a_es=ALPHA_ES):
    nu, m, s = t_fit(x)
    return {'var': float(-(m + s * stats.t.ppf(a_var, nu))), 'es': float(-m + s * t_es_factor(nu, a_es)),
            'nu': nu, 'm': m, 's': s}


def cf_z(a, S, K):
    """Cornish-Fisher quantile of a standardised variable with skewness S and excess kurtosis K."""
    z = stats.norm.ppf(a)
    return z + (z ** 2 - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S ** 2 / 36


def cf_var_es(x, a_var=ALPHA_VAR, a_es=ALPHA_ES):
    """Cornish-Fisher VaR = -(mu + sd z_cf); ES = minus the average of the CF quantiles on (0, a_es)."""
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std(ddof=1)
    S, K = float(stats.skew(x)), float(stats.kurtosis(x))
    u = (np.arange(2000) + 0.5) / 2000 * a_es
    return {'var': float(-(mu + sd * cf_z(a_var, S, K))), 'es': float(-(mu + sd * cf_z(u, S, K).mean())),
            'z': float(cf_z(a_var, S, K)), 'S': S, 'K': K, 'mu': float(mu), 'sd': float(sd)}


def pot_fit(x, q=POT_Q):
    """GPD fit to the losses L = -r above the threshold u = q-quantile of the losses (peaks over threshold)."""
    L = -np.asarray(x, float)
    u = float(np.quantile(L, q))
    exc = L[L > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return {'u': u, 'xi': float(xi), 'beta': float(beta), 'Nu': int(len(exc)), 'n': int(len(L))}


def pot_var(f, a):
    """VaR_a = u + beta/xi [(n a / N_u)^(-xi) - 1], valid for a < N_u / n."""
    return float(f['u'] + f['beta'] / f['xi'] * ((f['n'] * a / f['Nu']) ** (-f['xi']) - 1))


def pot_es(f, a):
    """ES_a = VaR_a / (1 - xi) + (beta - xi u) / (1 - xi), valid for xi < 1."""
    return float(pot_var(f, a) / (1 - f['xi']) + (f['beta'] - f['xi'] * f['u']) / (1 - f['xi']))


def garch_fit(r, last_obs=None):
    """GARCH(1,1) with constant mean and standardised Student-t innovations (arch package)."""
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    return am.fit(disp='off', last_obs=last_obs, options={'maxiter': 2000})


def conditional_next_day(k, a_var=ALPHA_VAR, a_es=ALPHA_ES):
    """VaR and ES for the day after the last observation: GARCH(1,1)-t and filtered historical simulation."""
    r = returns(k)
    res = garch_fit(r)
    p = res.params
    mu, nu = float(p['mu']), float(p['nu'])
    s_next = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
    z = res.std_resid.dropna().values
    return {'mu': mu, 'nu': nu, 'omega': float(p['omega']), 'alpha': float(p['alpha[1]']), 'beta': float(p['beta[1]']),
            'sigma': s_next, 'q_t': std_t_q(nu, a_var), 'es_t': std_t_es(nu, a_es),
            'var_g': float(-(mu + s_next * std_t_q(nu, a_var))), 'es_g': float(-mu + s_next * std_t_es(nu, a_es)),
            'q_z': float(-hs_var(z, a_var)), 'es_z': hs_es(z, a_es),
            'var_f': float(-(mu + s_next * -hs_var(z, a_var))), 'es_f': float(-mu + s_next * hs_es(z, a_es)),
            'n': int(len(r)), 'last': r.index[-1].date().isoformat(), 'r_last': float(r.iloc[-1])}


def methods_table(assets=ASSETS):
    """VaR 1% and ES 2.5% (in %) of each series on its whole sample by five unconditional methods, plus the
    conditional GARCH-t and FHS values for the next day."""
    out = {}
    for k in assets:
        x = returns(k)
        mu, sd = float(x.mean()), float(x.std())
        tt = t_var_es(x)
        cf = cf_var_es(x)
        f = pot_fit(x)
        c = conditional_next_day(k)
        out[k] = {'n': int(len(x)), 'first': x.index[0].date().isoformat(), 'mu': mu, 'sd': sd,
                  'skew': cf['S'], 'exkurt': cf['K'], 'nu': tt['nu'], 'xi': f['xi'],
                  'HS': {'var': hs_var(x), 'es': hs_es(x)},
                  'Normal': {'var': normal_var(mu, sd), 'es': normal_es(mu, sd)},
                  'Student-t': {'var': tt['var'], 'es': tt['es']},
                  'Cornish-Fisher': {'var': cf['var'], 'es': cf['es']},
                  'EVT': {'var': pot_var(f, ALPHA_VAR), 'es': pot_es(f, ALPHA_ES)},
                  'GARCH-t': {'var': c['var_g'], 'es': c['es_g']}, 'FHS': {'var': c['var_f'], 'es': c['es_f']},
                  'pot': f, 'cf_z': cf['z'], 't': tt, 'cond': c}
    return out


# =============================================================================
# CHARTS: DEFINITIONS AND METHODS
# =============================================================================
def fig_var_es_def(k='sp500', save=True):
    """Histogram of daily returns with minus VaR 1% and minus ES 2.5% (historical simulation)."""
    x = returns(k)
    v1, q25, e25 = hs_var(x, ALPHA_VAR), -hs_var(x, ALPHA_ES), hs_es(x, ALPHA_ES)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    bins = np.arange(-13, 12.01, 0.25)
    ax.hist(x, bins=bins, density=True, color=st.MainBlue, alpha=0.75, label=f'daily log returns of the {NAME[k]} (%)')
    ax.axvspan(-13, q25, color=st.IDAred, alpha=0.08, label='worst 2.5% of the days')
    ax.axvline(-v1, color=st.IDAred, lw=2, label=f'minus VaR 1% = {-v1:.2f}%')
    ax.axvline(-e25, color=st.Purple, lw=2, ls='--', label=f'minus ES 2.5% = {-e25:.2f}%')
    ax.set_xlim(-13, 12)
    ax.set_xlabel('daily return (%)')
    ax.set_ylabel('density')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_var_es_def')
    return {'var1': v1, 'q25': q25, 'es25': e25, 'n': int(len(x)), 'n_tail1': int((x < -v1).sum()),
            'min': float(x.min()), 'date_min': x.idxmin().date().isoformat()}


def var_curve(x, alphas):
    """VaR_alpha (positive, in %) for each tail probability by five unconditional methods."""
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    tt = t_var_es(x)
    cf = cf_var_es(x)
    f = pot_fit(x)
    out = {'historical': [hs_var(x, a) for a in alphas], 'Normal': [normal_var(mu, sd, a) for a in alphas],
           'Student-t': [float(-(tt['m'] + tt['s'] * stats.t.ppf(a, tt['nu']))) for a in alphas],
           'Cornish-Fisher': [float(-(mu + sd * cf_z(a, cf['S'], cf['K']))) for a in alphas],
           'EVT (POT)': [pot_var(f, a) if a < f['Nu'] / f['n'] else np.nan for a in alphas]}
    return out


def fig_var_curves(keys=('sp500', 'bet'), save=True):
    """VaR as a function of the tail probability alpha (log axis) for the S&P 500 and the BET."""
    alphas = np.array([0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    sty = {'historical': (st.DarkText, 'o', '-'), 'Normal': (st.Forest, None, '--'), 'Student-t': (st.MainBlue, None, '-'),
           'Cornish-Fisher': (st.Purple, None, '-.'), 'EVT (POT)': (st.IDAred, None, '-')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    out = {}
    for ax, k in zip(axes, keys):
        x = returns(k).values
        cur = var_curve(x, alphas)
        for lab, v in cur.items():
            c, m, ls = sty[lab]
            ax.plot(100 * alphas, v, color=c, marker=m, ms=4, ls=ls if m is None else 'none', lw=1.6, label=lab)
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.25, 0.5, 1, 2.5, 5])
        ax.set_xticklabels(['0.1', '0.25', '0.5', '1', '2.5', '5'])
        ax.invert_xaxis()
        ax.set_xlabel('tail probability alpha (%)')
        ax.set_ylabel('VaR (%)')
        ax.set_title(NAME[k])
        out[k] = {lab: dict(zip([f'{a:g}' for a in alphas], map(float, v))) for lab, v in cur.items()}
    st.fig_legend_bottom(fig, ncol=5, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_var_curves')
    return out


def precision(k='sp500', n_last=500, B=2000, seed=SEED):
    """Bootstrap (Chapter 2) 95% intervals of the historical VaR 1% and ES 2.5%: whole sample and last n_last days."""
    rng = np.random.default_rng(seed)
    x = returns(k).values
    out = {}
    for lab, y in [('all', x), ('last', x[-n_last:])]:
        idx = rng.integers(0, len(y), (B, len(y)))
        bv = np.array([hs_var(y[i]) for i in idx])
        be = np.array([hs_es(y[i]) for i in idx])
        out[lab] = {'n': int(len(y)), 'var': hs_var(y), 'es': hs_es(y), 'var_lo': float(np.quantile(bv, 0.025)),
                    'var_hi': float(np.quantile(bv, 0.975)), 'es_lo': float(np.quantile(be, 0.025)),
                    'es_hi': float(np.quantile(be, 0.975)), 'var_se': float(bv.std()), 'es_se': float(be.std())}
    return out


def horizon_check(k='sp500', h=10):
    """Square-root-of-time rule: sqrt(h) x one-day historical VaR 1% against the historical VaR 1% of overlapping
    h-day log returns."""
    r = returns(k)
    rh = r.rolling(h).sum().dropna()
    return {'var1': hs_var(r), 'var_sqrt': float(np.sqrt(h) * hs_var(r)), 'var_h': hs_var(rh.values), 'h': h,
            'var_n1': normal_var(r.mean(), r.std()), 'var_nh': normal_var(h * r.mean(), np.sqrt(h) * r.std())}


# =============================================================================
# ROLLING FORECASTS (out of sample)
# =============================================================================
def rolling_forecasts(k, oos_start=None, window=WINDOW, refit=REFIT):
    """One-day-ahead VaR 1%, VaR 2.5% and ES 2.5% for every day from oos_start; the forecast for day t uses only data
    up to day t-1. HS and Normal: the last `window` returns. GARCH-t and FHS: GARCH(1,1)-t re-estimated every `refit`
    days on all data up to that day; FHS uses the empirical quantiles of the standardised residuals."""
    r = returns(k)
    x = r.values
    i0 = max(r.index.searchsorted(pd.Timestamp(oos_start or OOS_START[k])), window)
    W = np.lib.stride_tricks.sliding_window_view(x, window)[i0 - window:len(x) - window]    # windows ending at t-1
    Ws = np.sort(W, axis=1)
    k1 = int(np.ceil(window * ALPHA_VAR - 1e-9))
    k25 = int(np.ceil(window * ALPHA_ES - 1e-9))
    df = pd.DataFrame(index=r.index[i0:])
    df['r'] = x[i0:]
    df['var1_HS'] = -Ws[:, k1 - 1]
    df['var25_HS'] = -Ws[:, k25 - 1]
    df['es25_HS'] = -Ws[:, :k25].mean(axis=1)
    m, s = W.mean(axis=1), W.std(axis=1, ddof=1)
    df['mu_N'], df['sd_N'] = m, s
    df['var1_Normal'] = -(m + s * stats.norm.ppf(ALPHA_VAR))
    df['var25_Normal'] = -(m + s * stats.norm.ppf(ALPHA_ES))
    df['es25_Normal'] = -m + s * stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / ALPHA_ES
    cols = {c: [] for c in ['mu_G', 'sig_G', 'nu_G', 'block']}
    zs = []
    for b, s0 in enumerate(range(i0, len(r), refit)):
        e = min(s0 + refit, len(r))
        am = arch_model(r.iloc[:e], mean='Constant', vol='GARCH', p=1, q=1, dist='t')
        res = am.fit(disp='off', last_obs=r.index[s0], options={'maxiter': 2000})
        sig = am.fix(res.params).conditional_volatility.iloc[s0:e].values
        p = res.params
        cols['mu_G'] += [float(p['mu'])] * (e - s0)
        cols['sig_G'] += list(sig)
        cols['nu_G'] += [float(p['nu'])] * (e - s0)
        cols['block'] += [b] * (e - s0)
        zs.append(np.sort(res.std_resid.dropna().values))
    for c, v in cols.items():
        df[c] = v
    mu, sig, nu = df['mu_G'].values, df['sig_G'].values, df['nu_G'].values
    df['var1_GARCH-t'] = -(mu + sig * stats.t.ppf(ALPHA_VAR, nu) * np.sqrt((nu - 2) / nu))
    df['var25_GARCH-t'] = -(mu + sig * stats.t.ppf(ALPHA_ES, nu) * np.sqrt((nu - 2) / nu))
    qa = stats.t.ppf(ALPHA_ES, nu)
    df['es25_GARCH-t'] = -mu + sig * np.sqrt((nu - 2) / nu) * stats.t.pdf(qa, nu) * (nu + qa ** 2) / ((nu - 1) * ALPHA_ES)
    zq1 = np.array([-hs_var(z, ALPHA_VAR) for z in zs])[df['block'].values]
    zq25 = np.array([-hs_var(z, ALPHA_ES) for z in zs])[df['block'].values]
    zes = np.array([hs_es(z, ALPHA_ES) for z in zs])[df['block'].values]
    df['var1_FHS'] = -(mu + sig * zq1)
    df['var25_FHS'] = -(mu + sig * zq25)
    df['es25_FHS'] = -mu + sig * zes
    df.attrs['W'] = W
    df.attrs['zs'] = zs
    return df


# =============================================================================
# BACKTESTS
# =============================================================================
def kupiec(x, n, a=ALPHA_VAR):
    """Kupiec (1995) proportion-of-failures test: LR_uc = -2 ln[(1-a)^(n-x) a^x / ((1-x/n)^(n-x) (x/n)^x)] ~ chi2(1)."""
    ph = x / n
    l0 = (n - x) * np.log(1 - a) + x * np.log(a)
    l1 = (n - x) * np.log(1 - ph) if ph < 1 else 0.0
    l1 += x * np.log(ph) if x > 0 else 0.0
    lr = float(-2 * (l0 - l1))
    return {'x': int(x), 'n': int(n), 'rate': float(ph), 'lr': lr, 'p': float(stats.chi2.sf(lr, 1))}


def christoffersen(hits, a=ALPHA_VAR):
    """Christoffersen (1998): independence LR_ind ~ chi2(1) from the transition counts n_ij (i = yesterday,
    j = today), and conditional coverage LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    h = np.asarray(hits, int)
    h0, h1 = h[:-1], h[1:]
    n00 = int(np.sum((h0 == 0) & (h1 == 0)))
    n01 = int(np.sum((h0 == 0) & (h1 == 1)))
    n10 = int(np.sum((h0 == 1) & (h1 == 0)))
    n11 = int(np.sum((h0 == 1) & (h1 == 1)))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(q, a_, b_):
        return (a_ * np.log(1 - q) if a_ and q < 1 else 0.0) + (b_ * np.log(q) if b_ and q > 0 else 0.0)
    lr_ind = float(-2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11)))
    uc = kupiec(int(h.sum()), len(h), a)
    lr_cc = uc['lr'] + lr_ind
    return {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11, 'p01': float(p01), 'p11': float(p11),
            'lr_ind': lr_ind, 'p_ind': float(stats.chi2.sf(lr_ind, 1)), 'lr_cc': float(lr_cc),
            'p_cc': float(stats.chi2.sf(lr_cc, 2)), 'uc': uc}


PLUS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_zone(x):
    """Basel traffic light for 250 days at VaR 1%: green 0-4, yellow 5-9, red 10 or more exceptions."""
    return 'green' if x <= 4 else ('yellow' if x <= 9 else 'red')


def traffic_table(n=250, a=ALPHA_VAR, xmax=10):
    """Binomial(n, a) probability and cumulative probability of x exceptions, zone and plus factor (BCBS, 1996)."""
    return [{'x': x, 'p': float(stats.binom.pmf(x, n, a)), 'cum': float(stats.binom.cdf(x, n, a)), 'zone': basel_zone(x),
             'plus': PLUS.get(x, 1.0)} for x in range(xmax + 1)]


def z2_stat(r, var25, es25, a=ALPHA_ES):
    """Acerbi-Szekely (2014) Z2 = sum_t r_t I_t / (T a ES_t) + 1, I_t = 1{r_t < -VaR_t}; E[Z2] = 0 under the null,
    negative values: ES too small."""
    r, var25, es25 = (np.asarray(v, float) for v in (r, var25, es25))
    I = r < -var25
    return float(np.sum(r * I / es25) / (len(r) * a) + 1)


def z2_pvalue(df, m, nsim=N_Z2, seed=SEED):
    """p-value of Z2 by simulation: returns are drawn from each day's forecast distribution of method m and Z2 is
    recomputed with the same VaR and ES forecasts; p = share of simulated Z2 below the observed one."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if m == 'Normal':
        X = df['mu_N'].values[:, None] + df['sd_N'].values[:, None] * rng.standard_normal((n, nsim))
    elif m == 'HS':
        W = df.attrs['W']
        X = W[np.arange(n)[:, None], rng.integers(0, W.shape[1], (n, nsim))]
    elif m == 'GARCH-t':
        nu = df['nu_G'].values[:, None]
        z = rng.standard_t(nu, (n, nsim)) * np.sqrt((nu - 2) / nu)
        X = df['mu_G'].values[:, None] + df['sig_G'].values[:, None] * z
    else:
        zs = df.attrs['zs']
        Z = np.empty((n, nsim))
        blk = df['block'].values
        for b, z in enumerate(zs):
            rows = np.where(blk == b)[0]
            Z[rows] = z[rng.integers(0, len(z), (len(rows), nsim))]
        X = df['mu_G'].values[:, None] + df['sig_G'].values[:, None] * Z
    v = df[f'var25_{m}'].values[:, None]
    e = df[f'es25_{m}'].values[:, None]
    zsim = np.sum(X * (X < -v) / e, axis=0) / (n * ALPHA_ES) + 1
    z0 = z2_stat(df['r'], df[f'var25_{m}'], df[f'es25_{m}'])
    return z0, float(np.mean(zsim <= z0))


def quantile_loss(r, var, a=ALPHA_VAR):
    """Average quantile (pinball) loss of VaR forecasts: S(v, x) = (a - 1{x < -v})(x + v); smaller is better."""
    r, var = np.asarray(r, float), np.asarray(var, float)
    return float(np.mean((a - (r < -var)) * (r + var)))


def backtest(df, m):
    """All backtests of method m on a frame of rolling forecasts."""
    hits = (df['r'] < -df[f'var1_{m}']).astype(int).values
    c = christoffersen(hits)
    last = hits[-250:].sum()
    z2, pz2 = z2_pvalue(df, m)
    return {'n': int(len(hits)), 'x': int(hits.sum()), 'rate': float(hits.mean()), 'exp': float(ALPHA_VAR * len(hits)),
            'lr_uc': c['uc']['lr'], 'p_uc': c['uc']['p'], 'lr_ind': c['lr_ind'], 'p_ind': c['p_ind'], 'lr_cc': c['lr_cc'],
            'p_cc': c['p_cc'], 'n11': c['n11'], 'n01': c['n01'], 'n10': c['n10'], 'n00': c['n00'], 'p01': c['p01'],
            'p11': c['p11'], 'last250': int(last), 'zone_last': basel_zone(last), 'z2': z2, 'p_z2': pz2,
            'zones': {z: int(np.sum([basel_zone(v) == z for v in pd.Series(hits).rolling(250).sum().dropna().values]))
                      for z in ('green', 'yellow', 'red')},
            'n_windows': int(len(hits) - 249), 'qloss': quantile_loss(df['r'], df[f'var1_{m}']),
            'max250': int(pd.Series(hits).rolling(250).sum().max())}


def stress_counts(df, m):
    """Exceedances of VaR 1% in the 2008 and 2020 stress periods."""
    out = {}
    for e, (a, b) in EPISODES.items():
        d = df.loc[a:b]
        out[e] = {'n': int(len(d)), 'x': int((d['r'] < -d[f'var1_{m}']).sum()), 'exp': float(ALPHA_VAR * len(d))}
    return out


def backtest_all(assets=BT_ASSETS):
    frames, res = {}, {}
    for k in assets:
        df = rolling_forecasts(k)
        frames[k] = df
        res[k] = {'start': df.index[0].date().isoformat(), 'end': df.index[-1].date().isoformat()}
        for m in METHODS:
            res[k][m] = backtest(df, m)
            res[k][m]['stress'] = stress_counts(df, m)
            res[k][m]['mean_var'] = float(df[f'var1_{m}'].mean())
    return res, frames


# =============================================================================
# CHARTS: BACKTESTING
# =============================================================================
def fig_rolling_var(df, k='sp500', save=True):
    """Daily returns and minus VaR 1% of the four methods in the 2008 and 2020 stress periods."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (a, b) in zip(axes, [('2007-06-01', '2009-12-31'), ('2019-10-01', '2021-03-31')]):
        d = df.loc[a:b]
        ax.plot(d.index, d['r'], color=st.Teal, lw=0.6, alpha=0.7, label='daily return (%)')
        for m in METHODS:
            ax.plot(d.index, -d[f'var1_{m}'], color=MCOL[m], lw=1.5, ls='--' if m in ('HS', 'Normal') else '-',
                    label=f'minus VaR 1%, {m}')
        hit = d['r'] < -d['var1_HS']
        ax.scatter(d.index[hit], d['r'][hit], color=MCOL['HS'], s=22, zorder=5, label='exceedance of the HS VaR')
        ax.set_ylabel('%')
        ax.tick_params(axis='x', labelrotation=30)
    axes[0].set_title(f'{NAME[k]}, 2007-2009')
    axes[1].set_title(f'{NAME[k]}, 2019-2021')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_rolling_var')


def fig_hits(df, k='sp500', save=True):
    """Exceedance dates of the VaR 1% of each method: clusters show a lack of independence."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for e, (a, b) in EPISODES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=st.Amber, alpha=0.15, label='_')
    for i, m in enumerate(METHODS):
        hit = df.index[df['r'] < -df[f'var1_{m}']]
        ax.scatter(hit, np.full(len(hit), i), color=MCOL[m], marker='|', s=180, lw=1.6,
                   label=f'{m}: {len(hit)} exceedances')
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xlabel('date (shaded: September 2008 - March 2009 and February - May 2020)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.32)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_hits')


def fig_binomial(n=250, a=ALPHA_VAR, save=True):
    """Binomial(250, 1%) distribution of the number of exceptions and the Basel zones."""
    tt = traffic_table(n, a, 14)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    col = {'green': st.Forest, 'yellow': st.Amber, 'red': st.IDAred}
    lab = {'green': 'green zone (0-4)', 'yellow': 'yellow zone (5-9)', 'red': 'red zone (10 or more)'}
    seen = set()
    for row in tt:
        z = row['zone']
        ax.bar(row['x'], 100 * row['p'], color=col[z], label=lab[z] if z not in seen else '_')
        seen.add(z)
        if row['x'] <= 9:
            ax.text(row['x'], 100 * row['p'] + 0.6, f"{100 * row['cum']:.2f}", ha='center', fontsize=9, color=st.DarkText)
    ax.set_xlabel('number of exceptions in 250 days (labels: cumulative probability, %)')
    ax.set_ylabel('probability (%)')
    ax.set_xticks(range(0, 15))
    st.legend_outside_bottom(ax, ncol=3, y=-0.25)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_binomial')
    return tt


def fig_traffic_time(df, k='sp500', save=True):
    """Number of exceptions in the last 250 days for HS and GARCH-t, with the zone limits."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for m in ('HS', 'Normal', 'GARCH-t'):
        c = (df['r'] < -df[f'var1_{m}']).astype(int).rolling(250).sum()
        ax.plot(c.index, c, color=MCOL[m], lw=1.5, label=f'{m}')
    ax.axhline(4.5, color=st.Forest, ls='--', lw=1.2, label='upper limit of the green zone (4)')
    ax.axhline(9.5, color=st.IDAred, ls='--', lw=1.2, label='upper limit of the yellow zone (9)')
    ax.set_ylabel('exceptions in 250 days')
    st.legend_outside_bottom(ax, ncol=3, y=-0.18)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_traffic_time')


# =============================================================================
# PORTFOLIO: VARIANCE-COVARIANCE, TAIL DEPENDENCE, COPULAS
# =============================================================================
def portfolio_data(keys=PORT):
    """Prices joined on common days first, then simple returns in % (the portfolio return is linear in them)."""
    P = pd.concat([load_close(k, START.get(k)) for k in keys], axis=1).dropna()
    R = 100 * P.pct_change().dropna()
    R.columns = list(keys)
    return R


def pseudo_obs(R):
    """Ranks divided by n + 1: the empirical copula sample (uniform margins)."""
    return R.rank().values / (len(R) + 1)


def t_copula_loglik(U, rho, nu):
    """Log-likelihood of the bivariate t copula with correlation rho and nu degrees of freedom."""
    x = stats.t.ppf(U, nu)
    q = (x[:, 0] ** 2 - 2 * rho * x[:, 0] * x[:, 1] + x[:, 1] ** 2) / (1 - rho ** 2)
    from scipy.special import gammaln
    lj = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * np.log(1 - rho ** 2)
          - (nu + 2) / 2 * np.log1p(q / nu))
    return float(np.sum(lj - stats.t.logpdf(x[:, 0], nu) - stats.t.logpdf(x[:, 1], nu)))


def gauss_copula_loglik(U, rho):
    x = stats.norm.ppf(U)
    q = (rho ** 2 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 2 * rho * x[:, 0] * x[:, 1]) / (1 - rho ** 2)
    return float(np.sum(-0.5 * np.log(1 - rho ** 2) - 0.5 * q))


def t_tail_dependence(rho, nu):
    """lambda = 2 t_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho))); the Gaussian copula has lambda = 0."""
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


def copula_fit(R):
    """rho from Kendall's tau (rho = sin(pi tau / 2)); nu of the t copula by maximum likelihood over a grid."""
    U = pseudo_obs(R)
    tau = float(stats.kendalltau(R.iloc[:, 0], R.iloc[:, 1])[0])
    rho = float(np.sin(np.pi * tau / 2))
    grid = np.arange(2.0, 40.01, 0.25)
    ll = np.array([t_copula_loglik(U, rho, v) for v in grid])
    nu = float(grid[np.argmax(ll)])
    return {'tau': tau, 'rho': rho, 'nu': nu, 'll_t': float(ll.max()), 'll_g': gauss_copula_loglik(U, rho),
            'lambda_t': t_tail_dependence(rho, nu), 'pearson': float(R.corr().iloc[0, 1]),
            'spearman': float(stats.spearmanr(R.iloc[:, 0], R.iloc[:, 1])[0])}


def empirical_tail_dep(R, qs=(0.05, 0.01)):
    """P(U1 <= q | U2 <= q): the share of the worst q of days of asset 2 that are also among the worst q of asset 1;
    under independence it equals q."""
    U = pseudo_obs(R)
    return {f'{q:g}': float(np.mean((U[:, 0] <= q) & (U[:, 1] <= q)) / q) for q in qs}


def simulate_copula(rho, nu=None, n=N_SIM, seed=SEED):
    """Uniform pairs from the Gaussian copula (nu None) or the t copula with correlation rho."""
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    if nu is None:
        return stats.norm.cdf(Z)
    w = np.sqrt(nu / rng.chisquare(nu, n))
    return stats.t.cdf(Z * w[:, None], nu)


def portfolio_analysis(R=None, w=WEIGHTS):
    """Portfolio VaR 1% and ES 2.5%: variance-covariance (Normal), historical simulation, and copula simulation with
    empirical margins (Gaussian and t copula); tail dependence."""
    R = portfolio_data() if R is None else R
    w = np.asarray(w)
    mu, S = R.mean().values, R.cov().values
    sd = np.sqrt(np.diag(S))
    rp = R.values @ w
    mp, sp = float(w @ mu), float(np.sqrt(w @ S @ w))
    cf = copula_fit(R)
    out = {'n': int(len(R)), 'first': R.index[0].date().isoformat(), 'mu': mu.tolist(), 'sd': sd.tolist(),
           'corr': float(R.corr().iloc[0, 1]), 'mp': mp, 'sp': sp,
           'var_n': normal_var(mp, sp), 'es_n': normal_es(mp, sp),
           'var_n_single': [normal_var(mu[i], sd[i]) for i in range(2)],
           'var_hs': hs_var(rp), 'es_hs': hs_es(rp), 'var_hs_single': [hs_var(R.values[:, i]) for i in range(2)],
           'copula': cf, 'tail_emp': empirical_tail_dep(R)}
    out['var_n_sum'] = float(w @ np.array(out['var_n_single']))
    out['var_hs_sum'] = float(w @ np.array(out['var_hs_single']))
    out['contrib'] = (w * (S @ w) / sp).tolist()               # Euler contributions to sigma_p
    for lab, nu in [('gauss', None), ('t', cf['nu'])]:
        U = simulate_copula(cf['rho'], nu)
        X = np.column_stack([np.quantile(R.values[:, i], U[:, i]) for i in range(2)])
        xp = X @ w
        q1 = [np.quantile(R.values[:, i], 0.01) for i in range(2)]
        out[f'var_{lab}'] = hs_var(xp)
        out[f'es_{lab}'] = hs_es(xp)
        out[f'joint1_{lab}'] = float(np.mean((X[:, 0] <= q1[0]) & (X[:, 1] <= q1[1])))
    q1 = [np.quantile(R.values[:, i], 0.01) for i in range(2)]
    out['joint1_emp'] = float(np.mean((R.values[:, 0] <= q1[0]) & (R.values[:, 1] <= q1[1])))
    out['joint1_ind'] = 0.0001
    for e, (a, b) in EPISODES.items():
        out[f'corr_{e}'] = float(R.loc[a:b].corr().iloc[0, 1])
    out['corr_calm'] = float(R.loc['2017-01-01':'2017-12-31'].corr().iloc[0, 1])
    return out


def fig_rolling_corr(R=None, save=True):
    """250-day rolling correlation of BET and S&P 500 daily returns (common days)."""
    R = portfolio_data() if R is None else R
    c = R.iloc[:, 0].rolling(250).corr(R.iloc[:, 1]).dropna()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for e, (a, b) in EPISODES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=st.Amber, alpha=0.18, label='_')
    ax.plot(c.index, c, color=st.IDAred, lw=1.4, label='250-day correlation, BET and S&P 500')
    ax.axhline(R.corr().iloc[0, 1], color=st.MainBlue, ls='--', lw=1.2, label=f'whole sample: {R.corr().iloc[0, 1]:.2f}')
    ax.set_ylabel('correlation')
    st.legend_outside_bottom(ax, ncol=2, y=-0.15)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_rolling_corr')
    return {'max': float(c.max()), 'date_max': c.idxmax().date().isoformat(), 'min': float(c.min()),
            'last': float(c.iloc[-1])}


def fig_copula(R=None, fit=None, save=True):
    """Lower-left corner [0, 0.25]^2 of the pseudo-observations of BET and S&P 500 against samples of the same size from
    the fitted Gaussian and t copulas; the square marks the days on which both are among their worst 5%."""
    R = portfolio_data() if R is None else R
    cf = copula_fit(R) if fit is None else fit
    U = pseudo_obs(R)
    n = len(U)
    G = simulate_copula(cf['rho'], None, n, SEED + 1)
    Tt = simulate_copula(cf['rho'], cf['nu'], n, SEED + 2)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    q = 0.05
    for ax, (lab, X, c) in zip(axes, [('data (pseudo-observations)', U, st.MainBlue),
                                       (f'Gaussian copula, rho = {cf["rho"]:.2f}', G, st.Forest),
                                       (f't copula, rho = {cf["rho"]:.2f}, nu = {cf["nu"]:.1f}', Tt, st.IDAred)]):
        ax.scatter(X[:, 1], X[:, 0], s=7, color=c, alpha=0.6, label=lab)
        k = int(np.sum((X[:, 0] <= q) & (X[:, 1] <= q)))
        ax.plot([0, q, q], [q, q, 0], color=st.Orange, lw=2.0)
        ax.set_title(f'{k} pairs below 0.05 on both scales', fontsize=11)
        ax.set_xlabel('S&P 500 (uniform scale)')
        ax.set_xlim(0, 0.25)
        ax.set_ylim(0, 0.25)
    axes[0].set_ylabel('BET (uniform scale)')
    leg = st.fig_legend_bottom(fig, ncol=3, y=0.0)
    for h in leg.legend_handles:
        h.set_sizes([30])
        h.set_alpha(1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch10_copula')
    return {'corner_data': int(np.sum((U[:, 0] <= q) & (U[:, 1] <= q))),
            'corner_gauss': int(np.sum((G[:, 0] <= q) & (G[:, 1] <= q))),
            'corner_t': int(np.sum((Tt[:, 0] <= q) & (Tt[:, 1] <= q))), 'n': int(n), 'expected_ind': float(n * q * q)}


if __name__ == '__main__':
    st.apply()
    N = {}
    N['def'] = fig_var_es_def()
    MT = methods_table()
    N['methods'] = MT
    N['curves'] = fig_var_curves()
    N['horizon'] = horizon_check()
    N['precision'] = precision()
    N['traffic'] = fig_binomial()
    bt, frames = backtest_all()
    N['bt'] = bt
    fig_rolling_var(frames['sp500'])
    fig_hits(frames['sp500'])
    fig_traffic_time(frames['sp500'])
    R = portfolio_data()
    P = portfolio_analysis(R)
    N['port'] = P
    N['corr'] = fig_rolling_corr(R)
    N['copula_fig'] = fig_copula(R, P['copula'])
    N['end'] = returns('sp500').index[-1].date().isoformat()
    mt = pd.DataFrame({(NAME[k], m): {'VaR 1% (%)': MT[k][m]['var'], 'ES 2.5% (%)': MT[k][m]['es']}
                       for k in ASSETS for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT', 'GARCH-t', 'FHS']}).T
    mt.to_csv(os.path.join(TABLE_DIR, 'ch10_methods_table.csv'), float_format='%.4f')
    btt = pd.DataFrame({(NAME[k], m): {c: bt[k][m][c] for c in ['n', 'x', 'rate', 'lr_uc', 'p_uc', 'lr_ind', 'p_ind', 'lr_cc',
                                                                'p_cc', 'last250', 'z2', 'p_z2']}
                        for k in BT_ASSETS for m in METHODS}).T
    btt.to_csv(os.path.join(TABLE_DIR, 'ch10_backtest_table.csv'), float_format='%.4f')
    for k, df in frames.items():
        df.attrs = {}
    with open(os.path.join(TABLE_DIR, 'ch10_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
