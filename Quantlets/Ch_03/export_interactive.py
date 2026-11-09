"""Export the inputs of the Chapter 3 interactive page (SFM/interactive/ch3_data.js).

Uses the functions of generate_all_charts.py (same returns, same seed, same simulations, same fits), so the page starts
from the same numbers as the slides. The daily log returns are exported in % with 6 decimals; the browser computes
histograms, empirical quantiles, tail counts and running variances from them, and the stable densities by the same FFT
inversion of the characteristic function. Simulations that the browser cannot repeat with numpy's generator are run here
with the chapter's seed (2026), for the slide's parameters and for the other parameter values the page offers. The ML
fits and the aggregation estimates are read from ch3_numbers.json. The Monte Carlo study of the estimators is re-run with
its seed (the slide reports only its means and standard deviations). The script stops if a recomputed number no longer
matches ch3_numbers.json.
Run:  python3 Quantlets/Ch_03/export_interactive.py
"""
import json
import math
import os
from multiprocessing import Pool

import numpy as np

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch3_data.js')
SHORT = {'bet': 'BET', 'sp500': 'S&P 500', 'dax': 'DAX', 'btc': 'Bitcoin'}


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch3_numbers.json ({b})')


def r6(a):
    return [float(f'{x:.6g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


# ------------------------------------------------------------------ stability (slide: n = 10, m = 50 000, alpha = 1.7)
def stability(N, ns=(2, 5, 10, 20, 50), alpha=1.7, m=50000):
    p = np.linspace(0.005, 0.995, 199)
    out = {'p': r6(p), 'alpha': alpha, 'sums': {}}
    for n in ns:
        rng = np.random.default_rng(g.SEED)               # as fig_stability_qq: one draw first, then the sums
        one = g.rstable(m, alpha, rng=rng)
        sums = g.rstable(m * n, alpha, rng=rng).reshape(m, n).sum(axis=1)
        out['one'], out['q99_one'] = r6(np.quantile(one, p)), float(np.quantile(one, 0.99))
        out['sums'][n] = {'q': r6(np.quantile(sums, p)), 'q99': float(np.quantile(sums, 0.99))}
        if n == 10:
            S = N['stability']
            close(out['q99_one'], S['q99_one'], 1e-9, 'stability q99_one')
            close(out['sums'][n]['q99'] / n ** (1 / alpha), S['q99_alpha'], 1e-9, 'stability q99_alpha')
            close(out['sums'][n]['q99'] / np.sqrt(n), S['q99_sqrt'], 1e-9, 'stability q99_sqrt')
    return out


# ------------------------------------------------------------------ GCLT (slide: Student-t nu = 1.5, n = 1, 10, 100)
def gclt(N, nu=1.5, ns=(1, 10, 100), m=40000):
    rng = np.random.default_rng(g.SEED)
    bins = np.linspace(-15, 15, 121)
    ks = np.round(np.arange(1, 15.01, 0.5), 2)
    out = {'nu': nu, 'ns': list(ns), 'edges': r6(bins), 'ks': ks.tolist(), 'hist': {}, 'tail': {},
           'gamma': N['gclt']['gamma_limit']}
    for n in ns:
        s = rng.standard_t(nu, size=(m, n)).sum(axis=1) / n ** (1 / nu)
        h, _ = np.histogram(s, bins=bins, density=True)
        out['hist'][n] = r6(h)
        out['tail'][n] = [float(np.mean(np.abs(s) > k)) for k in ks]
        close(float(np.mean(np.abs(s) > 10)), N['gclt'][f'n{n}_tail'], 1e-12, f'gclt n{n}_tail')
    return out


# ------------------------------------------------------------------ running variance of simulated samples
def runvar_sim(N, alphas=(1.1, 1.3, 1.5, 1.7, 1.8, 1.9, 1.95, 2.0), n=20000):
    k = np.arange(1, n + 1)
    idx = np.unique(np.concatenate([np.arange(10, 101), np.round(np.logspace(np.log10(100), np.log10(n), 700)).astype(int)])) - 1
    out = {'k': (idx + 1).tolist(), 'alphas': list(alphas), 'paths': {}}
    for a in alphas:
        rng = np.random.default_rng(g.SEED)              # as fig_running_variance: the Normal path first, then 3 paths
        z = g.rstable(n, 2.0, 0.0, 1.0, 0.0, rng=rng)
        v = np.cumsum(z ** 2) / k - (np.cumsum(z) / k) ** 2
        out['normal'] = r6(v[idx])
        out['normal_end'] = float(v[-1])
        P = []
        for i in range(3):
            x = g.rstable(n, a, 0.0, 1.0, 0.0, rng=rng)
            v = np.cumsum(x ** 2) / k - (np.cumsum(x) / k) ** 2
            P.append({'v': r6(v[idx]), 'v1000': float(v[999]), 'vend': float(v[-1]), 'max': float(np.max(np.abs(x)))})
            if a == 1.7:
                close(v[999], N['runvar_sim'][f'path{i + 1}']['v1000'], 1e-9, 'runvar_sim v1000')
                close(v[-1], N['runvar_sim'][f'path{i + 1}']['vend'], 1e-9, 'runvar_sim vend')
        out['paths'][f'{a:g}'] = P
    return out


# ------------------------------------------------------------------ CMS draws (slide: S1(1.5, 0.5, 1, 0), n = 100 000)
def cms(N, alphas=(1.2, 1.5, 1.8), betas=(-1.0, -0.5, 0.0, 0.5, 1.0), n=100000):
    bins = np.linspace(-8, 8, 161)
    out = {'edges': r6(bins), 'n': n, 'alphas': list(alphas), 'betas': list(betas), 'h': {}}
    for a in alphas:
        for b in betas:
            rng = np.random.default_rng(g.SEED)
            x = g.rstable(n, a, b, rng=rng)
            c, _ = np.histogram(x[(x > -8) & (x < 8)], bins=bins)
            out['h'][f'{a:g}_{b:g}'] = {'c': c.tolist(), 'q01': float(np.quantile(x, 0.01)), 'q99': float(np.quantile(x, 0.99)),
                                        'med': float(np.median(x)), 'out': int(np.sum((x <= -8) | (x >= 8)))}
            if a == 1.5 and b == 0.5:
                close(out['h']['1.5_0.5']['q01'], N['cms']['q01'], 1e-9, 'cms q01')
                close(out['h']['1.5_0.5']['q99'], N['cms']['q99'], 1e-9, 'cms q99')
    out['ks_stat'], out['ks_p'] = N['cms']['ks_stat'], N['cms']['ks_p']
    return out


# ------------------------------------------------------------------ Monte Carlo study of the two estimators
def _fit(x):
    m = g.mcculloch(x)
    return m['alpha'], g.stable_mle(x, start=m, se=False)['alpha']


def estimator_mc(N, alpha=1.7, beta=0.0, ns=(500, 2500), reps=200):
    rng = np.random.default_rng(g.SEED)                  # the samples in the order of estimator_mc
    samples = [(n, g.rstable(n, alpha, beta, rng=rng)) for n in ns for _ in range(reps)]
    with Pool(min(12, os.cpu_count() or 1)) as pool:
        est = pool.map(_fit, [x for _, x in samples])
    out = {}
    for n in ns:
        e = np.array([est[i] for i, (nn, _) in enumerate(samples) if nn == n])
        out[n] = {'mcculloch': [round(v, 5) for v in e[:, 0]], 'ml': [round(v, 5) for v in e[:, 1]]}
        for j, k in enumerate(('mcculloch', 'ml')):
            close(float(e[:, j].mean()), N['mc'][str(n)][k]['mean'], 1e-6, f'mc {n} {k} mean')
            close(float(e[:, j].std(ddof=1)), N['mc'][str(n)][k]['sd'], 1e-6, f'mc {n} {k} sd')
    return out


# ------------------------------------------------------------------ McCulloch on real data, by start year
def mcculloch_years(ret, years=range(2000, 2021)):
    out = {}
    for k, r in ret.items():
        out[k] = {}
        for y in years:
            x = r[r.index >= f'{y}-01-01'].values
            m = g.mcculloch(x)
            out[k][y] = {'n': len(x), 'nu_a': m['nu_alpha'], 'nu_b': m['nu_beta'], 'alpha': m['alpha'], 'beta': m['beta'],
                         'gamma': m['gamma'], 'delta0': m['delta0'], 'delta1': m['delta1'], 'q': [m[c] for c in ('q05', 'q25', 'q50', 'q75', 'q95')]}
    # McCulloch's Table III read through the chapter's interpolation: synthetic quantiles with IQR 1 and median 0
    nu = np.round(np.arange(2.44, 6.001, 0.02), 3)
    line = {}
    for nb in (0.0, 0.1, 0.2, 0.3):
        line[str(nb)] = [g.mcculloch_quantiles((nb * v - v) / 2, -0.5, 0.0, 0.5, (v + nb * v) / 2)['alpha'] for v in nu]
    # the nodes of Table III, column nu_beta = 0 (the worked example interpolates between two of them)
    nodes = [2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0]
    a0 = [g.mcculloch_quantiles(-v / 2, -0.5, 0.0, 0.5, v / 2)['alpha'] for v in nodes]
    return out, {'nu': nu.tolist(), 'alpha': line, 'nodes': nodes, 'nodes_alpha': [round(v, 3) for v in a0]}


# ------------------------------------------------------------------ running variance: data vs paths from the fitted law
def runvar_real(N, ret, fits, step=3):
    out = {}
    for k, r in ret.items():
        s = fits[k]['stable']
        n = len(r)
        i = np.arange(1, n + 1)
        rng = np.random.default_rng(g.SEED)
        idx = np.unique(np.concatenate([np.arange(19, n, step), [n - 1]]))
        P = []
        for j in range(3):
            x = g.rstable(n, s['alpha'], s['beta'], s['gamma'], s['delta0'], param=0, rng=rng)
            v = np.cumsum((x - x.mean()) ** 2) / i
            P.append(r6(v[idx]))
            if k == 'sp500':
                close(float(v[-1]), N['runvar_real'][f'sim{j + 1}'], 1e-9, 'runvar_real sim')
        out[k] = {'idx': idx.tolist(), 'paths': P}
        if k == 'sp500':
            v = np.cumsum(r.values ** 2) / i - (np.cumsum(r.values) / i) ** 2
            close(float(v[-1]), N['runvar_real']['data'], 1e-9, 'runvar_real data')
    return out


def main():
    N = json.load(open(os.path.join(HERE, 'ch3_numbers.json')))
    ret = {k: g.returns(k) for k in g.ASSETS}
    fits = N['fits']
    for k, r in ret.items():                             # the slides' samples: same length, dates, minimum, quantiles
        f = fits[k]
        assert len(r) == f['n'] and r.index[0].date().isoformat() == f['first'] and r.index[-1].date().isoformat() == f['last'], k
        close(float(r.min()), f['min'], 1e-9, f'{k} min')
        close(float(r.std()), f['sd'], 1e-9, f'{k} sd')
        m = g.mcculloch(r.values)
        close(m['nu_alpha'], f['mcculloch']['nu_alpha'], 1e-9, f'{k} nu_alpha')
        close(m['alpha'], f['mcculloch']['alpha'], 1e-9, f'{k} McCulloch alpha')
        for x in (5, 10, 15, 20):
            assert int(np.sum(r.values < -x)) == N['tail_counts'][k][f'obs_{x}'], (k, x)
    d = dict(start=N['start'], end=N['end'], assets=g.ASSETS, colors=g.COLORS, model_col=g.MODEL_COL, short=SHORT)
    d['ret'] = {}
    for k, r in ret.items():
        days = (r.index.values.astype('datetime64[D]').astype(np.int64)).tolist()
        d['ret'][k] = dict(d0=days[0], dd=np.diff(days).tolist(), r=np.round(r.values, 6).tolist())
    d['fits'] = {k: {'n': f['n'], 'normal': f['normal'], 't': f['t'], 'stable': f['stable'],
                     'mcculloch': {c: f['mcculloch'][c] for c in ('alpha', 'beta', 'gamma', 'delta0', 'delta1', 'nu_alpha', 'nu_beta',
                                                                  'q05', 'q25', 'q50', 'q75', 'q95')}}
                 for k, f in fits.items()}
    d['density_alpha'] = N['density_alpha']
    # alpha = 0.7: the slide's curve is scipy's levy_stable density (fig_density_alpha), exported as drawn
    x = np.linspace(-8, 8, 801)
    f07 = g.stats.levy_stable.pdf(x, 0.7, 0.0)
    close(float(np.interp(0, x, f07)), N['density_alpha']['0.7']['f0'], 1e-9, 'density alpha 0.7 f0')
    close(float(np.interp(5, x, f07)), N['density_alpha']['0.7']['f5'], 1e-9, 'density alpha 0.7 f5')
    d['da07'] = r6(f07)
    d['s0s1'] = N['s0s1']
    d['tails'] = N['tails']
    d['stab'] = stability(N)
    d['gclt'] = gclt(N)
    d['rvs'] = runvar_sim(N)
    d['cms'] = cms(N)
    d['mcc'], d['mcc_line'] = mcculloch_years(ret)
    for k in g.ASSETS:                                   # the start year 2000 is the slides' sample
        close(d['mcc'][k][2000]['alpha'], fits[k]['mcculloch']['alpha'], 1e-12, f'{k} McCulloch alpha 2000')
        close(d['mcc'][k][2000]['gamma'], fits[k]['mcculloch']['gamma'], 1e-12, f'{k} McCulloch gamma 2000')
    d['tail_table'] = N['tail_table']
    d['agg'] = N['aggregation']
    d['rvr'] = runvar_real(N, ret, fits)
    d['mc'] = estimator_mc(N)
    js = ('// Inputs of the Chapter 3 interactive page, written by Quantlets/Ch_03/export_interactive.py.\n'
          'window.CH3_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
