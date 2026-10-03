"""
generate_all_charts.py -- charts and numbers of Chapter 9 (SFM): ARCH and GARCH models and their extensions
===========================================================================================================
Course data (sfm_data.py), chart style (sfm_style.py), the arch package for maximum-likelihood estimation.
Every number on the slides comes from here.
  * motivation   -- S&P 500 daily returns since 2000 (the 2008 and 2020 episodes);
  * simulation   -- i.i.d. Normal, ARCH(1) and GARCH(1,1) paths with the same unconditional variance (as in
                    SFEtimegarch); the log-likelihood of a simulated ARCH(1) (as in SFElikarch1) and of a simulated
                    GARCH(1,1) over a grid of (alpha, beta) (as in SFElikgarch);
  * estimation   -- ARCH(q) and GARCH(1,1) on the S&P 500; Gaussian GARCH(1,1) estimated step by step (scipy) and
                    with the arch package; classic and robust (Bollerslev-Wooldridge) standard errors;
  * innovations  -- Normal, Student-t and skewed-t innovations; QQ plots of the standardised residuals;
  * markets      -- GARCH(1,1)-t for the S&P 500, DAX, BET, Bitcoin, Banca Transilvania and OMV Petrom; conditional
                    volatility (as in SFEvolgarchest); persistence and half-life; the 2008 and 2020 episodes;
  * asymmetry    -- GJR-GARCH and EGARCH; news impact curves, with a kernel estimate as in SFENewsImpactCurve;
  * diagnostics  -- Ljung-Box tests on standardised residuals and their squares, ARCH-LM, the sign-bias test;
  * selection    -- nine models compared by AIC and BIC;
  * forecasts    -- multi-step forecasts and the volatility term structure; out-of-sample one-day forecasts of
                    GARCH-t, GJR-t and EWMA (lambda = 0.94) compared by QLIKE and the Diebold-Mariano test;
  * VaR          -- one-day VaR 1% from GARCH-t and from EWMA-Normal, exceedances (preview of Chapter 10).
Output: charts/sfm_ch9_*.pdf/.png, Quantlets/Ch_09/ch9_numbers.json and ch9_*_table.csv
Based on the Quantlets SFEtimegarch, SFElikarch1, SFElikgarch, SFEgarchest, SFEvolgarchest and SFENewsImpactCurve
(github.com/QuantLet/SFE), on Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 13,
and on Tsay (2010), Analysis of Financial Time Series, 3rd ed., Ch. 3.
Run:  python3 Quantlets/Ch_09/generate_all_charts.py
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
from sfm_data import log_returns, periods_per_year   # noqa: E402
import sfm_style as st                               # noqa: E402
import statsmodels.api as sm                         # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch   # noqa: E402
from arch import arch_model                          # noqa: E402

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
EPISODES = {'2008': ('2008-09-01', '2009-03-31'), '2020': ('2020-02-15', '2020-05-31')}
OOS_START = '2015-01-01'     # out-of-sample period of the forecast comparison
REFIT = 250                  # the models are re-estimated every 250 observations (expanding window)
LAMBDA = 0.94                # EWMA (RiskMetrics) decay factor, Chapter 8
ALPHA_VAR = 0.01             # VaR level: VaR 1%
SEED = 2026
TERM_DATES = ['2017-11-03', '2020-03-16']   # a calm day and the COVID-19 peak (plus the last day of the data)


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
# SIMULATION AND LIKELIHOOD (SFEtimegarch, SFElikarch1, SFElikgarch)
# =============================================================================
def simulate_garch(n, omega, alpha, beta=0.0, burn=500, seed=SEED, dist='normal', nu=5.0):
    """eps_t = sigma_t z_t, sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2 (beta = 0: ARCH(1));
    z_t standard Normal or standardised Student-t; returns (eps, sigma)."""
    rng = np.random.default_rng(seed)
    m = n + burn
    z = rng.standard_normal(m) if dist == 'normal' else rng.standard_t(nu, m) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(m)
    e = np.empty(m)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, m):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    return e[burn:], np.sqrt(s2[burn:])


def fig_simulated(n=1000, save=True):
    """Three paths with unconditional variance 1: i.i.d. Normal, ARCH(1) (omega = 0.5, alpha = 0.5) and
    GARCH(1,1) (omega = 0.02, alpha = 0.10, beta = 0.88, values typical of daily equity returns)."""
    rng = np.random.default_rng(SEED)
    paths = {'i.i.d. Normal(0, 1)': rng.standard_normal(n),
             'ARCH(1): omega = 0.5, alpha = 0.5': simulate_garch(n, 0.5, 0.5, 0.0, seed=SEED + 1)[0],
             'GARCH(1,1): omega = 0.02, alpha = 0.10, beta = 0.88': simulate_garch(n, 0.02, 0.10, 0.88, seed=SEED + 2)[0]}
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.6), sharex=True, sharey=True)
    out = {}
    for ax, (lab, x), c in zip(axes, paths.items(), [st.MainBlue, st.IDAred, st.Forest]):
        ax.plot(np.arange(n), x, color=c, lw=0.7, label=lab)
        ax.set_ylabel('value')
        k = float(stats.kurtosis(x, fisher=False))
        out[lab.split(':')[0]] = {'sd': float(np.std(x)), 'kurt': k, 'max_abs': float(np.max(np.abs(x)))}
    axes[-1].set_xlabel('time t')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_simulated')
    return out


def arch1_loglik(alpha, x):
    """Conditional Gaussian log-likelihood of ARCH(1) with omega = 1 - alpha (unconditional variance 1), as in
    SFElikarch1: l(alpha) = sum_{t>=2} [-0.5 ln s2_t - 0.5 x_t^2 / s2_t], s2_t = omega + alpha x_{t-1}^2."""
    s2 = (1 - alpha) + alpha * x[:-1] ** 2
    return float(np.sum(-0.5 * np.log(s2) - 0.5 * x[1:] ** 2 / s2))


def fig_lik_arch1(ns=(100, 1000), alpha0=0.5, save=True):
    """Log-likelihood of a simulated ARCH(1) (omega = 0.5, alpha = 0.5) as a function of alpha, for two sample sizes;
    each curve minus its maximum."""
    grid = np.linspace(0.05, 0.95, 181)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    for n, c in zip(ns, [st.MainBlue, st.IDAred]):
        x = simulate_garch(n, 1 - alpha0, alpha0, 0.0, seed=SEED + n)[0]
        ll = np.array([arch1_loglik(a, x) for a in grid])
        res = optimize.minimize_scalar(lambda a: -arch1_loglik(a, x), bounds=(0.001, 0.999), method='bounded')
        ax.plot(grid, ll - ll.max(), color=c, lw=1.6, label=f'n = {n}: log-likelihood minus its maximum')
        ax.axvline(res.x, color=c, ls='--', lw=1.0, label=f'n = {n}: maximum at alpha = {res.x:.3f}')
        # curvature at the maximum: standard error from the observed information
        h = 1e-4
        d2 = (arch1_loglik(res.x + h, x) - 2 * arch1_loglik(res.x, x) + arch1_loglik(res.x - h, x)) / h ** 2
        out[str(n)] = {'alpha_hat': float(res.x), 'se': float(1 / np.sqrt(-d2))}
    ax.axvline(alpha0, color='black', ls=':', lw=1.2, label=f'true alpha = {alpha0}')
    ax.set_ylim(-40, 2)
    ax.set_xlabel('alpha')
    ax.set_ylabel('log-likelihood minus maximum')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_lik_arch1')
    return out


def garch_loglik_vt(a, b, x, s2bar):
    """Gaussian GARCH(1,1) log-likelihood with variance targeting omega = (1 - a - b) s2bar, as in SFElikgarch."""
    om = (1 - a - b) * s2bar
    s2 = np.empty(len(x))
    s2[0] = s2bar
    for t in range(1, len(x)):
        s2[t] = om + a * x[t - 1] ** 2 + b * s2[t - 1]
    return float(np.sum(-0.5 * np.log(s2[1:]) - 0.5 * x[1:] ** 2 / s2[1:]))


def fig_lik_garch(ns=(500, 2000), save=True):
    """Contour plots of the log-likelihood of a simulated GARCH(1,1) (omega = 0.1, alpha = 0.1, beta = 0.8) over a
    grid of (alpha, beta), with omega set by variance targeting, for n = 500 (as in SFElikgarch) and n = 2000."""
    A = np.linspace(0.01, 0.30, 59)
    B = np.linspace(0.50, 0.97, 48)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    out = {}
    cols = [st.Purple, st.MainBlue, st.Teal, st.Forest, st.Amber, st.Orange, st.IDAred]
    for ax, n in zip(axes, ns):
        x = simulate_garch(n, 0.1, 0.1, 0.8, seed=SEED + 7)[0]
        s2bar = float(np.var(x))
        L = np.full((len(B), len(A)), np.nan)
        for i, b in enumerate(B):
            for j, a in enumerate(A):
                if a + b < 0.995:
                    L[i, j] = garch_loglik_vt(a, b, x, s2bar)
        i, j = np.unravel_index(np.nanargmax(L), L.shape)
        lv = np.nanmax(L) - np.array([40, 20, 10, 5, 3, 2, 1])
        cs = ax.contour(A, B, L, levels=lv, colors=cols, linewidths=1.1)
        ax.clabel(cs, fmt=lambda v, m=np.nanmax(L): f'{v - m:.0f}', fontsize=8)
        ax.plot(A[j], B[i], 'o', color=st.IDAred, ms=8, label='maximum on the grid')
        ax.plot(0.1, 0.8, 'x', color='black', ms=9, mew=2, label='true values: alpha = 0.1, beta = 0.8')
        ax.plot(A, 1 - A, color=st.Purple, ls='--', lw=1.2, label='alpha + beta = 1 (IGARCH boundary)')
        ax.set_xlim(A[0], A[-1])
        ax.set_ylim(B[0], B[-1])
        ax.set_title(f'n = {n}: maximum at alpha = {A[j]:.2f}, beta = {B[i]:.2f}', loc='left', fontsize=11)
        ax.set_xlabel('alpha')
        out[str(n)] = {'a_hat': float(A[j]), 'b_hat': float(B[i])}
    axes[0].set_ylabel('beta')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_lik_garch')
    return out


# =============================================================================
# ESTIMATION (arch package)
# =============================================================================
def fit(r, vol='GARCH', dist='t', o=0, p=1, q=1, last_obs=None, cov_type='robust'):
    """ML estimation with the arch package: constant mean; GARCH (o = 0), GJR-GARCH (o = 1), EGARCH, ARCH(p)."""
    if vol == 'ARCH':
        am = arch_model(r, mean='Constant', vol='ARCH', p=p, dist=dist)
    elif vol == 'EGARCH':
        am = arch_model(r, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist=dist)
    else:
        am = arch_model(r, mean='Constant', vol='GARCH', p=p, o=o, q=q, dist=dist)
    return am.fit(disp='off', last_obs=last_obs, cov_type=cov_type, options={'maxiter': 2000})


def persistence(res):
    """alpha + beta (GARCH), alpha + beta + gamma/2 (GJR with symmetric innovations), beta (EGARCH)."""
    p = res.params
    if 'gamma[1]' in p and 'alpha[1]' in p and res.model.volatility.__class__.__name__ == 'EGARCH':
        return float(p['beta[1]'])
    return float(p['alpha[1]'] + p['beta[1]'] + 0.5 * p.get('gamma[1]', 0.0))


def half_life(pers):
    """Days after which half of a variance shock has disappeared: ln 0.5 / ln(persistence)."""
    return float(np.log(0.5) / np.log(pers)) if 0 < pers < 1 else float('inf')


def summary(res, r):
    """Parameters, robust standard errors, persistence, half-life and long-run annualised volatility."""
    p, se = res.params, res.std_err
    pers = persistence(res)
    ppy = periods_per_year(r)
    out = {'n': int(res.nobs), 'first': r.index[0].date().isoformat(), 'ppy': float(ppy),
           'params': {k: float(v) for k, v in p.items()}, 'se': {k: float(v) for k, v in se.items()},
           'loglik': float(res.loglikelihood), 'aic': float(res.aic), 'bic': float(res.bic),
           'k': int(len(p)), 'pers': pers, 'hl': half_life(pers),
           'vol_sample': float(r.std() * np.sqrt(ppy))}
    if 'omega' in p and res.model.volatility.__class__.__name__ != 'EGARCH':
        uv = p['omega'] / (1 - pers)
        out['uv'] = float(uv)
        out['vol_lr'] = float(np.sqrt(uv * ppy))
    return out


def garch11_negloglik(theta, x):
    """Minus the Gaussian log-likelihood of r_t = mu + eps_t, GARCH(1,1); sigma_1^2 = sample variance."""
    mu, om, a, b = theta
    if om <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e10
    e = x - mu
    s2 = np.empty(len(x))
    s2[0] = np.var(x)
    for t in range(1, len(x)):
        s2[t] = om + a * e[t - 1] ** 2 + b * s2[t - 1]
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def fit_step_by_step(r):
    """Gaussian GARCH(1,1) estimated by numerical optimisation of the log-likelihood (scipy, L-BFGS-B), with
    classic standard errors from the inverse of the numerical Hessian."""
    x = np.asarray(r, float)
    v = np.var(x)
    th0 = np.array([x.mean(), 0.05 * v, 0.08, 0.90])
    bnds = [(None, None), (1e-6, None), (1e-6, 0.999), (1e-6, 0.999)]
    cons = [{'type': 'ineq', 'fun': lambda t: 0.9999 - t[2] - t[3]}]       # alpha + beta < 1
    res = optimize.minimize(garch11_negloglik, th0, args=(x,), method='SLSQP', bounds=bnds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-10})
    th = res.x
    from statsmodels.tools.numdiff import approx_hess
    H = approx_hess(th, garch11_negloglik, args=(x,))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {'params': dict(zip(['mu', 'omega', 'alpha[1]', 'beta[1]'], map(float, th))),
            'se': dict(zip(['mu', 'omega', 'alpha[1]', 'beta[1]'], map(float, se))), 'loglik': float(-res.fun)}


def arch_q_table(k='sp500'):
    """ARCH(1), ARCH(5), ARCH(10) and GARCH(1,1), all Gaussian: log-likelihood, AIC, BIC, sum of the ARCH terms."""
    r = returns(k)
    out = {}
    for lab, kw in [('ARCH(1)', dict(vol='ARCH', p=1)), ('ARCH(5)', dict(vol='ARCH', p=5)),
                    ('ARCH(10)', dict(vol='ARCH', p=10)), ('GARCH(1,1)', dict(vol='GARCH'))]:
        res = fit(r, dist='normal', **kw)
        p = res.params
        out[lab] = {'k': int(len(p)), 'loglik': float(res.loglikelihood), 'aic': float(res.aic), 'bic': float(res.bic),
                    'sum_alpha': float(sum(v for n, v in p.items() if n.startswith('alpha'))),
                    'pers': float(sum(v for n, v in p.items() if n.startswith(('alpha', 'beta'))))}
    return out


def estimation_sp500(k='sp500'):
    """S&P 500 GARCH(1,1): Normal innovations step by step and with arch (classic and robust standard errors);
    Student-t and skewed-t innovations with arch."""
    r = returns(k)
    out = {'step': fit_step_by_step(r)}
    rn = fit(r, dist='normal')
    rc = fit(r, dist='normal', cov_type='classic')
    out['normal'] = summary(rn, r)
    out['normal']['se_classic'] = {n: float(v) for n, v in rc.std_err.items()}
    for d in ['t', 'skewt']:
        out[d] = summary(fit(r, dist=d), r)
    z = rn.std_resid.dropna()
    out['kurt_r'] = float(stats.kurtosis(r, fisher=False))
    out['kurt_z'] = float(stats.kurtosis(z, fisher=False))
    out['skew_z'] = float(stats.skew(z))
    return out


def markets_table(names=ASSETS):
    """GARCH(1,1)-t for each series: parameters, robust standard errors, persistence, half-life, volatility."""
    return {k: summary(fit(returns(k), dist='t'), returns(k)) for k in names}


def fig_vol(names=('sp500', 'dax'), fname='sfm_ch9_vol_sp500_dax', save=True):
    """Annualised GARCH(1,1)-t conditional volatility, with the 2008 and 2020 episodes shaded (as in SFEvolgarchest)."""
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 5.4), sharex=False)
    out = {}
    for ax, k in zip(np.atleast_1d(axes), names):
        r = returns(k)
        res = fit(r, dist='t')
        ann = np.sqrt(periods_per_year(r))
        v = res.conditional_volatility * ann
        ax.plot(v.index, v.values, color=COLORS[k], lw=0.9, label=f'{NAME[k]}: GARCH(1,1)-t volatility, annualised (%)')
        ax.axhline(r.std() * ann, color='black', ls='--', lw=0.9, label=f'{NAME[k]}: sample volatility, annualised (%)' if k == names[0] else '_')
        for (a, b), c in zip(EPISODES.values(), [st.Purple, st.Orange]):
            if pd.Timestamp(a) < r.index[0]:
                continue
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.15, lw=0)
        ax.set_ylabel('% per year')
        out[k] = {'last': float(v.iloc[-1]), 'max': float(v.max()), 'date_max': v.idxmax().date().isoformat(),
                  'min': float(v.min()), 'median': float(v.median())}
        for e, (a, b) in EPISODES.items():
            w = v.loc[a:b]
            if len(w):
                out[k][f'peak{e}'] = float(w.max())
                out[k][f'date{e}'] = w.idxmax().date().isoformat()
    h, l = [], []
    for ax in np.atleast_1d(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if not ll.startswith('_'):
                h.append(hh)
                l.append(ll)
    h += [plt.Rectangle((0, 0), 1, 1, color=st.Purple, alpha=0.3), plt.Rectangle((0, 0), 1, 1, color=st.Orange, alpha=0.3)]
    l += ['Sep 2008 - Mar 2009 (global financial crisis)', 'Feb - May 2020 (COVID-19)']
    st.fig_legend_bottom(fig, h, l, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig(fname)
    return out


def fig_returns(k='sp500', save=True):
    """Daily S&P 500 log returns since 2000 with the two largest volatility episodes marked."""
    r = returns(k)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(r.index, r.values, color=COLORS[k], lw=0.5, label=f'{NAME[k]}: daily log return (%)')
    for (a, b), c, lab in zip(EPISODES.values(), [st.Purple, st.Orange],
                              ['Sep 2008 - Mar 2009 (global financial crisis)', 'Feb - May 2020 (COVID-19)']):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.18, lw=0, label=lab)
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_returns')
    out = {'n': len(r), 'first': r.index[0].date().isoformat(), 'last': r.index[-1].date().isoformat(),
           'min': float(r.min()), 'date_min': r.idxmin().date().isoformat(), 'max': float(r.max()),
           'date_max': r.idxmax().date().isoformat()}
    for e, (a, b) in EPISODES.items():
        out[f'sd{e}'] = float(r.loc[a:b].std())
    out['sd_all'] = float(r.std())
    calm = r.loc['2017-01-01':'2017-12-31']
    out['sd2017'] = float(calm.std())
    return out


def fig_persistence(T=None, save=True, horizon=250):
    """Share of a variance shock left after h days, persistence^h, with the half-lives marked."""
    T = T or markets_table()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    h = np.arange(horizon + 1)
    for k, d in T.items():
        lab = (f"{NAME[k]}: alpha + beta = {d['pers']:.3f}, half-life {d['hl']:.0f} days" if d['pers'] < 0.9995
               else f"{NAME[k]}: alpha + beta = 1 (IGARCH), no half-life")
        ax.plot(h, d['pers'] ** h, color=COLORS[k], lw=1.5, label=lab)
    ax.axhline(0.5, color='black', ls=':', lw=1.0, label='half of the shock')
    ax.set_xlabel('days after the shock, h')
    ax.set_ylabel('share of the shock left')
    ax.set_ylim(0, 1.02)
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_persistence')


def fig_qq(k='sp500', save=True):
    """QQ plots of the standardised residuals: GARCH-N against the Normal, GARCH-t against the standardised t."""
    r = returns(k)
    rn, rt = fit(r, dist='normal'), fit(r, dist='t')
    nu = rt.params['nu']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    out = {}
    for ax, res, lab, ppf, c in [(axes[0], rn, 'GARCH(1,1)-N residuals against the Normal', stats.norm.ppf, st.MainBlue),
                                 (axes[1], rt, f'GARCH(1,1)-t residuals against the standardised t({nu:.1f})',
                                  lambda u: stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu), st.IDAred)]:
        z = np.sort(res.std_resid.dropna().values)
        u = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        qth = ppf(u)
        ax.scatter(qth, z, s=5, color=c, alpha=0.6, label=lab)
        lim = [min(qth.min(), z.min()), max(qth.max(), z.max())]
        ax.plot(lim, lim, color='black', lw=1.0, ls='--', label='45-degree line' if c == st.MainBlue else '_')
        ax.set_xlabel('theoretical quantile')
        ax.set_ylabel('sample quantile')
        out[lab.split()[0]] = {'q001': float(np.quantile(z, 0.001)), 'th001': float(ppf(0.001))}
    st.fig_legend_bottom(fig, ncol=1, y=0.0)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_qq')
    return {'nu': float(nu), 'q': out}


# =============================================================================
# ASYMMETRY
# =============================================================================
def news_impact(res, eps, s2bar):
    """sigma_t^2 as a function of the shock eps_{t-1}, with sigma_{t-1}^2 fixed at s2bar (Engle and Ng, 1993)."""
    p = res.params
    if res.model.volatility.__class__.__name__ == 'EGARCH':
        z = eps / np.sqrt(s2bar)
        return np.exp(p['omega'] + p['alpha[1]'] * (np.abs(z) - np.sqrt(2 / np.pi)) + p['gamma[1]'] * z
                      + p['beta[1]'] * np.log(s2bar))
    return p['omega'] + (p['alpha[1]'] + p.get('gamma[1]', 0.0) * (eps < 0)) * eps ** 2 + p['beta[1]'] * s2bar


def kernel_nic(r, grid, h=None):
    """Nadaraya-Watson estimate of E[r_t^2 | r_{t-1} = x] with a Gaussian kernel (the idea of SFENewsImpactCurve);
    bandwidth half a standard deviation of the returns."""
    x, y = r.values[:-1], r.values[1:] ** 2
    h = h or 0.5 * np.std(x)
    w = stats.norm.pdf((grid[:, None] - x[None, :]) / h)
    return (w * y).sum(1) / w.sum(1), h


def fig_nic(names=('sp500', 'btc'), save=True):
    """News impact curves of GARCH-t, GJR-t and EGARCH-t and a kernel estimate, for two series."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    out = {}
    for ax, k in zip(axes, names):
        r = returns(k)
        s2bar = float(r.var())
        lo, hi = np.quantile(r, [0.01, 0.99])
        g = np.linspace(lo, hi, 201)
        for vol, o, lab, c in [('GARCH', 0, 'GARCH(1,1)-t', st.MainBlue), ('GARCH', 1, 'GJR-GARCH(1,1)-t', st.IDAred),
                               ('EGARCH', 0, 'EGARCH(1,1)-t', st.Forest)]:
            res = fit(r, vol=vol, o=o, dist='t')
            ax.plot(g, news_impact(res, g, s2bar), color=c, lw=1.6, label=lab)
        kn, h = kernel_nic(r, g)
        ax.plot(g, kn, color=st.Amber, lw=1.4, ls='--', label='kernel estimate of E[r(t)^2 | r(t-1)]')
        ax.set_title(NAME[k], loc='left', fontsize=12)
        ax.set_xlabel('shock yesterday, eps(t-1) (%)')
        ax.set_ylabel('variance today (%^2)')
        rg = fit(r, o=1, dist='t')
        out[k] = {'ratio_gjr': float(news_impact(rg, np.array([-2.0]), s2bar)[0] / news_impact(rg, np.array([2.0]), s2bar)[0]),
                  'h': float(h)}
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_nic')
    return out


def sign_bias_test(z, eps):
    """Engle-Ng (1993) sign-bias tests: z_t^2 on a constant, S-_{t-1}, S-_{t-1} eps_{t-1}, S+_{t-1} eps_{t-1};
    t statistics and the joint test T R^2 ~ chi2(3)."""
    d = pd.concat([pd.Series(np.asarray(z)), pd.Series(np.asarray(eps))], axis=1, keys=['z', 'e']).dropna()
    zz, e = d['z'].values, d['e'].values
    y = zz[1:] ** 2
    el = e[:-1]
    sneg = (el < 0).astype(float)
    X = np.column_stack([np.ones_like(el), sneg, sneg * el, (1 - sneg) * el])
    ols = sm.OLS(y, X).fit()
    joint = len(y) * ols.rsquared
    return {'sign_t': float(ols.tvalues[1]), 'neg_t': float(ols.tvalues[2]), 'pos_t': float(ols.tvalues[3]),
            'joint': float(joint), 'joint_p': float(stats.chi2.sf(joint, 3))}


def asym_table(names=ASSETS):
    """GJR-GARCH(1,1)-t and EGARCH(1,1)-t: the asymmetry parameter gamma, its robust t statistic, the likelihood
    ratio test of GJR against GARCH, and the sign-bias test on the GARCH-t residuals."""
    out = {}
    for k in names:
        r = returns(k)
        rg, rj, re_ = fit(r, dist='t'), fit(r, o=1, dist='t'), fit(r, vol='EGARCH', dist='t')
        lr = 2 * (rj.loglikelihood - rg.loglikelihood)
        sb = sign_bias_test(rg.std_resid, rg.resid)
        out[k] = {'gjr_gamma': float(rj.params['gamma[1]']), 'gjr_t': float(rj.tvalues['gamma[1]']),
                  'gjr_alpha': float(rj.params['alpha[1]']), 'gjr_beta': float(rj.params['beta[1]']),
                  'gjr_pers': persistence(rj), 'lr': float(lr), 'lr_p': float(stats.chi2.sf(lr, 1)),
                  'eg_gamma': float(re_.params['gamma[1]']), 'eg_t': float(re_.tvalues['gamma[1]']),
                  'eg_beta': float(re_.params['beta[1]']), 'sb': sb,
                  'bic_g': float(rg.bic), 'bic_j': float(rj.bic), 'bic_e': float(re_.bic)}
    return out


# =============================================================================
# DIAGNOSTICS AND MODEL SELECTION
# =============================================================================
def ljung_box(x, lags=10):
    """Ljung-Box Q(lags) and its p-value."""
    lb = acorr_ljungbox(pd.Series(np.asarray(x)).dropna(), lags=[lags])
    return float(lb['lb_stat'].iloc[0]), float(lb['lb_pvalue'].iloc[0])


def diagnostics(k='sp500'):
    """Ljung-Box Q(10) of r, r^2, z, z^2 and ARCH-LM(5) of z, for GARCH-N and GARCH-t."""
    r = returns(k)
    out = {'r': ljung_box(r), 'r2': ljung_box(r ** 2), 'archlm_r': list(map(float, het_arch(r.values, nlags=5)[:2]))}
    for d in ['normal', 't']:
        z = fit(r, dist=d).std_resid.dropna()
        out[d] = {'z': ljung_box(z), 'z2': ljung_box(z ** 2), 'archlm': list(map(float, het_arch(z.values, nlags=5)[:2]))}
    return out


def diag_markets(names=ASSETS):
    """Ljung-Box Q(10) of r^2 and of z^2 (GARCH-t) for each series."""
    out = {}
    for k in names:
        r = returns(k)
        z = fit(r, dist='t').std_resid.dropna()
        out[k] = {'r2': ljung_box(r ** 2), 'z2': ljung_box(z ** 2), 'z': ljung_box(z)}
    return out


def fig_acf_diag(k='sp500', nlags=50, save=True):
    """ACF of squared returns and of squared standardised residuals (GARCH-t), with the 95% band."""
    r = returns(k)
    z = fit(r, dist='t').std_resid.dropna()
    from statsmodels.tsa.stattools import acf as sacf
    a1 = sacf(r ** 2, nlags=nlags, fft=True)[1:]
    a2 = sacf(z ** 2, nlags=nlags, fft=True)[1:]
    lags = np.arange(1, nlags + 1)
    band = 1.96 / np.sqrt(len(r))
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.bar(lags - 0.2, a1, width=0.4, color=COLORS[k], label=f'{NAME[k]}: squared returns')
    ax.bar(lags + 0.2, a2, width=0.4, color=st.IDAred, label=f'{NAME[k]}: squared standardised residuals, GARCH(1,1)-t')
    ax.axhline(band, color='black', ls='--', lw=0.9, label='95% band, +/- 1.96/sqrt(T)')
    ax.axhline(-band, color='black', ls='--', lw=0.9)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('lag k (days)')
    ax.set_ylabel('autocorrelation')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_acf_diag')
    return {'a1_1': float(a1[0]), 'a1_50': float(a1[-1]), 'a2_1': float(a2[0]), 'max_a2': float(np.max(np.abs(a2))),
            'band': float(band)}


def model_selection(k='sp500'):
    """Nine models (GARCH, GJR, EGARCH x Normal, t, skewed t): log-likelihood, number of parameters, AIC, BIC."""
    r = returns(k)
    rows = []
    for vol, o, vlab in [('GARCH', 0, 'GARCH'), ('GARCH', 1, 'GJR'), ('EGARCH', 0, 'EGARCH')]:
        for d, dlab in [('normal', 'Normal'), ('t', 't'), ('skewt', 'skewed t')]:
            res = fit(r, vol=vol, o=o, dist=d)
            rows.append({'model': vlab, 'dist': dlab, 'k': int(len(res.params)), 'loglik': float(res.loglikelihood),
                         'aic': float(res.aic), 'bic': float(res.bic)})
    df = pd.DataFrame(rows)
    df['d_aic'] = df['aic'] - df['aic'].min()
    df['d_bic'] = df['bic'] - df['bic'].min()
    return df


# =============================================================================
# FORECASTS
# =============================================================================
def garch_path_forecast(res, r, date, horizon=250):
    """Multi-step GARCH(1,1) variance forecasts made at the end of `date`:
    sigma^2_{t+h} = s2bar + (alpha + beta)^(h-1) (sigma^2_{t+1} - s2bar), s2bar = omega / (1 - alpha - beta)."""
    p = res.params
    pers = p['alpha[1]'] + p['beta[1]']
    s2bar = p['omega'] / (1 - pers)
    i = r.index.get_loc(pd.Timestamp(date))
    s2t = res.conditional_volatility.iloc[i] ** 2
    e = r.iloc[i] - p['mu']
    s2next = p['omega'] + p['alpha[1]'] * e ** 2 + p['beta[1]'] * s2t
    h = np.arange(1, horizon + 1)
    return s2bar + pers ** (h - 1) * (s2next - s2bar), s2bar, s2next


def fig_term_structure(k='sp500', save=True, horizon=250):
    """Forecasts of the daily volatility h days ahead (annualised) and the average volatility over the next h days,
    made on a calm day, at the COVID-19 peak and on the last day of the data."""
    r = returns(k)
    res = fit(r, dist='t')
    ann = periods_per_year(r)
    dates = TERM_DATES + [r.index[-1].date().isoformat()]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    h = np.arange(1, horizon + 1)
    for d, c in zip(dates, [st.Forest, st.IDAred, st.MainBlue]):
        f, s2bar, s2n = garch_path_forecast(res, r, d, horizon)
        avg = np.sqrt(np.cumsum(f) / h * ann)
        ax.plot(h, np.sqrt(f * ann), color=c, lw=1.6, label=f'forecast made on {d}: daily volatility on day t+h')
        ax.plot(h, avg, color=c, lw=1.2, ls='--', label=f'forecast made on {d}: average volatility over the next h days')
        out[d] = {'h1': float(np.sqrt(f[0] * ann)), 'h10': float(np.sqrt(f[9] * ann)), 'h22': float(np.sqrt(f[21] * ann)),
                  'h250': float(np.sqrt(f[-1] * ann)), 'avg10': float(avg[9]), 'avg22': float(avg[21]), 'avg250': float(avg[-1]),
                  's2next': float(s2n), 'sum10': float(np.sum(f[:10]))}
    ax.axhline(np.sqrt(s2bar * ann), color='black', ls=':', lw=1.0, label='long-run volatility')
    ax.set_xlabel('horizon h (trading days)')
    ax.set_ylabel('% per year')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_term_structure')
    out['lr'] = float(np.sqrt(s2bar * ann))
    out['s2bar'] = float(s2bar)
    out['params'] = {n: float(v) for n, v in res.params.items()}
    return out


def ewma_variance(r, lam=LAMBDA, init=250):
    """EWMA variance forecast for day t made at t-1 (Chapter 8): h_t = lam h_{t-1} + (1 - lam) r_{t-1}^2."""
    x = np.asarray(r, float)
    h = np.empty(len(x))
    h[0] = np.mean(x[:init] ** 2)
    for t in range(1, len(x)):
        h[t] = lam * h[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(h, index=r.index)


def oos_forecasts(k, oos_start=OOS_START, refit=REFIT):
    """One-day-ahead variance forecasts on the out-of-sample period: GARCH-t and GJR-t re-estimated every `refit`
    observations on an expanding window (forecast for day t uses data up to t-1), and EWMA; with the GARCH-t and
    EWMA-Normal VaR 1%."""
    r = returns(k)
    i0 = r.index.searchsorted(pd.Timestamp(oos_start))
    cols = {'garch': [], 'gjr': [], 'var_garch': []}
    idx = []
    for s in range(i0, len(r), refit):
        e = min(s + refit, len(r))
        fits = {}
        for lab, o in [('garch', 0), ('gjr', 1)]:
            am = arch_model(r.iloc[:e], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='t')
            res = am.fit(disp='off', last_obs=r.index[s], options={'maxiter': 2000})
            fixed = am.fix(res.params)
            cols[lab].append(fixed.conditional_volatility.iloc[s:e] ** 2)
            fits[lab] = res.params
        p = fits['garch']
        q = stats.t.ppf(ALPHA_VAR, p['nu']) * np.sqrt((p['nu'] - 2) / p['nu'])
        cols['var_garch'].append(-(p['mu'] + np.sqrt(cols['garch'][-1]) * q))
        idx.append(r.index[s:e])
    df = pd.DataFrame({c: pd.concat(v) for c, v in cols.items()})
    df['ewma'] = ewma_variance(r).loc[df.index]
    df['var_ewma'] = -np.sqrt(df['ewma']) * stats.norm.ppf(ALPHA_VAR)
    df['r'] = r.loc[df.index]
    return df


def qlike(proxy, h):
    """QLIKE loss (Patton, 2011): proxy / h - ln(proxy / h) - 1 is written here as proxy / h + ln h, which differs
    only by a term that does not depend on the forecast and allows a zero proxy."""
    return proxy / h + np.log(h)


def dm_test(la, lb, lags=5):
    """Diebold-Mariano: d = L_a - L_b; HAC (Newey-West) t statistic of the mean of d (negative: model a better)."""
    d = np.asarray(la) - np.asarray(lb)
    ols = sm.OLS(d, np.ones_like(d)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    t = float(ols.tvalues[0])
    return {'mean': float(d.mean()), 't': t, 'p': float(2 * stats.norm.sf(abs(t)))}


def forecast_eval(names=('sp500', 'dax', 'bet', 'btc')):
    """Mean QLIKE and MSE of GARCH-t, GJR-t and EWMA; DM tests against EWMA; VaR 1% exceedance rates."""
    out, frames = {}, {}
    for k in names:
        df = oos_forecasts(k)
        frames[k] = df
        px = df['r'] ** 2
        L = {m: qlike(px, df[m]) for m in ['garch', 'gjr', 'ewma']}
        out[k] = {'n': len(df), 'first': df.index[0].date().isoformat(),
                  'qlike': {m: float(v.mean()) for m, v in L.items()},
                  'mse': {m: float(((px - df[m]) ** 2).mean()) for m in ['garch', 'gjr', 'ewma']},
                  'dm_garch_ewma': dm_test(L['garch'], L['ewma']), 'dm_gjr_garch': dm_test(L['gjr'], L['garch']),
                  'exc_garch': float((df['r'] < -df['var_garch']).mean()), 'exc_ewma': float((df['r'] < -df['var_ewma']).mean()),
                  'nexc_garch': int((df['r'] < -df['var_garch']).sum()), 'nexc_ewma': int((df['r'] < -df['var_ewma']).sum())}
    return out, frames


def fig_forecast_eval(frames, save=True):
    """Cumulative QLIKE difference EWMA minus GARCH-t on the out-of-sample period (rising: GARCH-t better)."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for k, df in frames.items():
        px = df['r'] ** 2
        d = (qlike(px, df['ewma']) - qlike(px, df['garch'])).cumsum()
        ax.plot(d.index, d.values, color=COLORS[k], lw=1.4, label=f'{NAME[k]}')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('cumulative loss difference')
    ax.set_title('Cumulative QLIKE(EWMA) - QLIKE(GARCH-t): rising = GARCH-t better', loc='left', fontsize=12)
    st.legend_outside_bottom(ax, ncol=4, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_forecast_eval')


def fig_var(frames, k='sp500', a='2019-07-01', b='2021-06-30', save=True):
    """Daily returns with the one-day VaR 1% of GARCH-t and of EWMA-Normal, and the exceedances."""
    df = frames[k].loc[a:b]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(df.index, df['r'], color=COLORS[k], lw=0.7, label=f'{NAME[k]}: daily log return (%)')
    ax.plot(df.index, -df['var_garch'], color=st.IDAred, lw=1.3, label='minus VaR 1%, GARCH(1,1)-t')
    ax.plot(df.index, -df['var_ewma'], color=st.Forest, lw=1.1, ls='--', label='minus VaR 1%, EWMA-Normal')
    e1 = df['r'] < -df['var_garch']
    e2 = df['r'] < -df['var_ewma']
    ax.scatter(df.index[e1], df['r'][e1], color=st.IDAred, s=40, marker='o', zorder=5, label='exceedance of the GARCH-t VaR')
    ax.scatter(df.index[e2], df['r'][e2], color=st.Forest, s=60, marker='x', zorder=6, label='exceedance of the EWMA-Normal VaR')
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch9_var')
    return {'n': int(len(df)), 'exc_garch': int(e1.sum()), 'exc_ewma': int(e2.sum())}


if __name__ == '__main__':
    st.apply()
    N = {}
    N['returns'] = fig_returns()
    N['sim'] = fig_simulated()
    N['lik_arch1'] = fig_lik_arch1()
    N['lik_garch'] = fig_lik_garch()
    N['archq'] = arch_q_table()
    N['est'] = estimation_sp500()
    T = markets_table()
    N['markets'] = T
    fig_persistence(T)
    N['vol1'] = fig_vol(('sp500', 'dax'), 'sfm_ch9_vol_sp500_dax')
    N['vol2'] = fig_vol(('bet', 'btc'), 'sfm_ch9_vol_bet_btc')
    N['qq'] = fig_qq()
    N['nic'] = fig_nic()
    N['asym'] = asym_table()
    N['diag'] = diagnostics()
    N['diag_m'] = diag_markets()
    N['acf_diag'] = fig_acf_diag()
    ms = model_selection()
    ms.to_csv(os.path.join(TABLE_DIR, 'ch9_model_selection.csv'), index=False, float_format='%.3f')
    N['ms'] = ms.to_dict(orient='records')
    N['term'] = fig_term_structure()
    fe, frames = forecast_eval()
    N['fe'] = fe
    fig_forecast_eval(frames)
    N['var_fig'] = fig_var(frames)
    N['end'] = returns('sp500').index[-1].date().isoformat()
    gt = pd.DataFrame({NAME[k]: {'n': v['n'], 'mu': v['params']['mu'], 'omega': v['params']['omega'],
                                 'alpha': v['params']['alpha[1]'], 'beta': v['params']['beta[1]'], 'nu': v['params']['nu'],
                                 'alpha+beta': v['pers'], 'half-life (days)': v['hl'], 'long-run vol (% p.a.)': v['vol_lr'],
                                 'sample vol (% p.a.)': v['vol_sample']} for k, v in T.items()}).T
    gt.to_csv(os.path.join(TABLE_DIR, 'ch9_garch_table.csv'), float_format='%.4f')
    at = pd.DataFrame({NAME[k]: {'GJR gamma': v['gjr_gamma'], 't': v['gjr_t'], 'LR': v['lr'], 'p LR': v['lr_p'],
                                 'EGARCH gamma': v['eg_gamma'], 't EGARCH': v['eg_t'], 'sign-bias joint': v['sb']['joint'],
                                 'p sign-bias': v['sb']['joint_p']} for k, v in N['asym'].items()}).T
    at.to_csv(os.path.join(TABLE_DIR, 'ch9_asym_table.csv'), float_format='%.4f')
    ft = pd.DataFrame({NAME[k]: {'n': v['n'], 'QLIKE GARCH-t': v['qlike']['garch'], 'QLIKE GJR-t': v['qlike']['gjr'],
                                 'QLIKE EWMA': v['qlike']['ewma'], 'DM t (GARCH-t vs EWMA)': v['dm_garch_ewma']['t'],
                                 'VaR 1% exceedances GARCH-t (%)': 100 * v['exc_garch'],
                                 'VaR 1% exceedances EWMA-Normal (%)': 100 * v['exc_ewma']} for k, v in fe.items()}).T
    ft.to_csv(os.path.join(TABLE_DIR, 'ch9_forecast_eval.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch9_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
