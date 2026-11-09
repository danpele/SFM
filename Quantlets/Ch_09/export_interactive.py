"""Export the inputs of the Chapter 9 interactive page (SFM/interactive/ch9_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same seeds, the arch package), so the page starts
from the numbers of the slides. Daily log returns (in %) are exported with 6 significant digits; every fitted model is exported
as its parameters, its robust standard errors, its log-likelihood and its first conditional variance, from which the browser
rebuilds the whole conditional variance path with the model's recursion (GARCH, GJR-GARCH, EGARCH), and from it the standardised
residuals, the QQ plots, the news impact curves, the persistence curves, the diagnostics (Ljung-Box, ARCH-LM, sign bias), the
information criteria, the forecasts and the VaR. The out-of-sample forecasts of GARCH-t and GJR-t (re-estimated every 250 days)
come from the chapter's oos_forecasts; the EWMA forecasts, the losses, the Diebold-Mariano tests and the exceedances are
recomputed in the browser. The simulations export the standard Normal draws of the chapter's generator (numpy, the same seeds);
the browser runs the ARCH and GARCH recursions on them.
Extra runs of the same functions: GARCH, GJR and EGARCH with Normal and skewed-t innovations for every market, GARCH fits on
later sample starts, ARCH(q) for q = 1, ..., 10 on every market, the step-by-step estimation on every market, other seeds of the
simulations. The script stops if a recomputed number no longer matches ch9_numbers.json.
Run:  python3 Quantlets/Ch_09/export_interactive.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy import stats

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch9_data.js')
MK = g.ASSETS                                  # sp500, dax, bet, btc, tlv, snp
VOLMK = ['sp500', 'dax', 'bet', 'btc']
STARTS = {'sp500': [2005, 2010, 2015], 'dax': [2005, 2010, 2015], 'bet': [2005, 2010, 2015], 'btc': [2016, 2018, 2020]}
DISTS = {'n': 'normal', 't': 't', 'st': 'skewt'}
SIM_SEEDS = [2026, 2036, 2046, 2056, 2066, 2076]          # 2026: the lecture's (SEED, SEED + 1, SEED + 2)
LIKG_SEEDS = [2033, 2043, 2053, 2063, 2073, 2083]         # 2033 = SEED + 7: the lecture's
BURN = 500


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch9_numbers.json ({b})')


def r6(a, d=6):
    return [None if not np.isfinite(x) else float(f'{x:.{d}g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def days(idx):
    d = idx.values.astype('datetime64[D]').astype(np.int64)
    return {'d0': int(d[0]), 'dd': np.diff(d).tolist()}


def model(res, r, full):
    """Parameters, robust SE, log-likelihood, number of parameters and observations, first conditional variance, and the
    index of the first observation in the full series."""
    p = res.params
    out = {'p': {k: float(v) for k, v in p.items()}, 'se': {k: float(v) for k, v in res.std_err.items()},
           'll': float(res.loglikelihood), 'k': int(len(p)), 'n': int(res.nobs),
           's0': float(res.conditional_volatility.iloc[0] ** 2), 'i0': int(full.index.get_loc(r.index[0]))}
    return out


def fits():
    F = {}
    for k in MK:
        full = g.returns(k)
        for vol, o, m in [('GARCH', 0, 'garch'), ('GARCH', 1, 'gjr'), ('EGARCH', 0, 'egarch')]:
            for dk, d in DISTS.items():
                F[f'{k}:{m}:{dk}'] = model(g.fit(full, vol=vol, o=o, dist=d), full, full)
        F[f'{k}:garch:n']['sec'] = {n: float(v) for n, v in g.fit(full, dist='normal', cov_type='classic').std_err.items()}
        for y in STARTS.get(k, []):
            r = g.returns(k, start=f'{y}-01-01')
            for dk, d in DISTS.items():
                F[f'{k}:garch:{dk}:{y}'] = model(g.fit(r, dist=d), r, full)
        print('  fits', k)
    return F


def archq():
    """ARCH(q), q = 1, ..., 10, Normal innovations, on every market (arch_q_table of the chapter runs q = 1, 5, 10)."""
    A = {}
    for k in MK:
        r = g.returns(k)
        A[k] = []
        for q in range(1, 11):
            res = g.fit(r, dist='normal', vol='ARCH', p=q)
            A[k].append({'ll': float(res.loglikelihood), 'k': int(len(res.params)), 'n': int(res.nobs),
                         'sa': float(sum(v for n, v in res.params.items() if n.startswith('alpha')))})
    return A


def steps():
    """fit_step_by_step of the chapter (scipy) on every market."""
    return {k: g.fit_step_by_step(g.returns(k)) for k in MK}


def sims(N):
    """Standard Normal draws of the chapter's simulations (numpy default_rng)."""
    out = {'paths': {}, 'lik1': {}, 'likg': {}}
    for s in SIM_SEEDS:
        out['paths'][str(s)] = {'iid': r6(np.random.default_rng(s).standard_normal(1000), 7),
                                'arch': r6(np.random.default_rng(s + 1).standard_normal(1000 + BURN), 7),
                                'garch': r6(np.random.default_rng(s + 2).standard_normal(1000 + BURN), 7)}
    sim = g.fig_simulated(save=False)
    matplotlib.pyplot.close('all')
    for lab in sim:
        close(sim[lab]['kurt'], N['sim'][lab]['kurt'], 1e-10, f'simulated kurtosis {lab}')
    for n in (100, 1000):
        out['lik1'][str(n)] = r6(np.random.default_rng(g.SEED + n).standard_normal(n + BURN), 7)
    l1 = g.fig_lik_arch1(save=False)
    matplotlib.pyplot.close('all')
    for n in ('100', '1000'):
        close(l1[n]['alpha_hat'], N['lik_arch1'][n]['alpha_hat'], 1e-8, f'ARCH(1) alpha hat, n = {n}')
    for s in LIKG_SEEDS:
        out['likg'][str(s)] = r6(np.random.default_rng(s).standard_normal(2000 + BURN), 7)
    lg = g.fig_lik_garch(save=False)
    matplotlib.pyplot.close('all')
    for n in ('500', '2000'):
        close(lg[n]['a_hat'], N['lik_garch'][n]['a_hat'], 1e-12, f'GARCH grid alpha, n = {n}')
        close(lg[n]['b_hat'], N['lik_garch'][n]['b_hat'], 1e-12, f'GARCH grid beta, n = {n}')
    return out


def oos(N):
    """oos_forecasts of the chapter (GARCH-t and GJR-t, re-estimated every 250 days) and, for the VaR at other levels, the mean
    and the degrees of freedom of each GARCH-t re-estimation (the same fits)."""
    out = {}
    for k in VOLMK:
        full = g.returns(k)
        df = g.oos_forecasts(k)
        i0 = full.index.get_loc(df.index[0])
        blocks = []
        for s in range(i0, len(full), g.REFIT):
            am = g.arch_model(full.iloc[:min(s + g.REFIT, len(full))], mean='Constant', vol='GARCH', p=1, o=0, q=1, dist='t')
            p = am.fit(disp='off', last_obs=full.index[s], options={'maxiter': 2000}).params
            blocks.append([s - i0, float(p['mu']), float(p['nu'])])
        # the VaR of the chapter, rebuilt from the exported mean, degrees of freedom and variance forecast
        mu = np.empty(len(df))
        nu = np.empty(len(df))
        for j, (a, m, v) in enumerate(blocks):
            b = blocks[j + 1][0] if j + 1 < len(blocks) else len(df)
            mu[a:b], nu[a:b] = m, v
        var = -(mu + np.sqrt(df['garch'].values) * stats.t.ppf(g.ALPHA_VAR, nu) * np.sqrt((nu - 2) / nu))
        assert np.max(np.abs(var - df['var_garch'].values)) < 1e-9, k
        px = df['r'] ** 2
        close(float(g.qlike(px, df['garch']).mean()), N['fe'][k]['qlike']['garch'], 1e-10, f'{k} QLIKE GARCH-t')
        close(float(g.qlike(px, df['gjr']).mean()), N['fe'][k]['qlike']['gjr'], 1e-10, f'{k} QLIKE GJR-t')
        close(float((df['r'] < -df['var_garch']).sum()), N['fe'][k]['nexc_garch'], 0, f'{k} exceedances GARCH-t')
        out[k] = {'i0': int(i0), 'garch': r6(df['garch'].values, 7), 'gjr': r6(df['gjr'].values, 7), 'blocks': blocks}
        print('  oos', k)
    return out


def check(N, F, A, S):
    for k in MK:
        for p, v in N['markets'][k]['params'].items():
            close(F[f'{k}:garch:t']['p'][p], v, 1e-10, f'{k} GARCH-t {p}')
        for p, v in N['markets'][k]['se'].items():
            close(F[f'{k}:garch:t']['se'][p], v, 1e-8, f'{k} GARCH-t robust SE {p}')
        close(F[f'{k}:gjr:t']['p']['gamma[1]'], N['asym'][k]['gjr_gamma'], 1e-10, f'{k} GJR gamma')
        close(F[f'{k}:egarch:t']['p']['gamma[1]'], N['asym'][k]['eg_gamma'], 1e-10, f'{k} EGARCH gamma')
        close(2 * (F[f'{k}:gjr:t']['ll'] - F[f'{k}:garch:t']['ll']), N['asym'][k]['lr'], 1e-8, f'{k} LR GJR')
    E = N['est']
    for p in ('mu', 'omega', 'alpha[1]', 'beta[1]'):
        close(F['sp500:garch:n']['p'][p], E['normal']['params'][p], 1e-10, f'GARCH-N {p}')
        close(F['sp500:garch:n']['se'][p], E['normal']['se'][p], 1e-8, f'GARCH-N robust SE {p}')
        close(F['sp500:garch:n']['sec'][p], E['normal']['se_classic'][p], 1e-8, f'GARCH-N classic SE {p}')
        close(S['sp500']['params'][p], E['step']['params'][p], 1e-6, f'step-by-step {p}')
        close(S['sp500']['se'][p], E['step']['se'][p], 1e-4, f'step-by-step SE {p}')
    close(F['sp500:garch:st']['p']['lambda'], E['skewt']['params']['lambda'], 1e-10, 'skewed t lambda')
    for q, lab in [(1, 'ARCH(1)'), (5, 'ARCH(5)'), (10, 'ARCH(10)')]:
        close(A['sp500'][q - 1]['ll'], N['archq'][lab]['loglik'], 1e-10, f'{lab} log-likelihood')
    for row in N['ms']:
        m = {'GARCH': 'garch', 'GJR': 'gjr', 'EGARCH': 'egarch'}[row['model']]
        d = {'Normal': 'n', 't': 't', 'skewed t': 'st'}[row['dist']]
        close(F[f'sp500:{m}:{d}']['ll'], row['loglik'], 1e-10, f'model selection {m} {d}')


def main():
    N = json.load(open(os.path.join(HERE, 'ch9_numbers.json')))
    print('fits')
    F = fits()
    A = archq()
    print('step by step')
    S = steps()
    check(N, F, A, S)
    print('simulations')
    sim = sims(N)
    print('out-of-sample forecasts')
    O = oos(N)
    d = dict(end=N['end'], names=MK, name=g.NAME, col=g.COLORS, seed=g.SEED, sim_seeds=SIM_SEEDS, likg_seeds=LIKG_SEEDS, starts=STARTS,
             episodes=g.EPISODES, term_dates=g.TERM_DATES, lam=g.LAMBDA, refit=g.REFIT)
    d['ret'] = {k: {**days(g.returns(k).index), 'x': r6(g.returns(k).values)} for k in MK}
    d['fit'] = F
    d['archq'] = A
    d['step'] = S
    d['sim'] = sim
    d['oos'] = O
    d['N'] = {k: N[k] for k in ('returns', 'sim', 'lik_arch1', 'lik_garch', 'archq', 'est', 'markets', 'vol1', 'vol2', 'qq', 'nic', 'asym', 'diag',
                                'diag_m', 'acf_diag', 'term', 'fe', 'var_fig')}
    js = ('// Inputs of the Chapter 9 interactive page, written by Quantlets/Ch_09/export_interactive.py.\n'
          'window.CH9_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
