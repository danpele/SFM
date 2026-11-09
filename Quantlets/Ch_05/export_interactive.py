"""Export the inputs of the Chapter 5 interactive page (SFM/interactive/ch5_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same thresholds, same seeds), so the page starts
from the numbers of the slides. Daily losses L = -r (in %) are exported with 6 significant digits; the browser computes the
survival functions, Hill estimates, mean excess, GPD fits, quantiles and exceedance counts from them. What the browser cannot
repeat is taken from the chapter's code: the GEV fits and their bootstrap intervals, the threshold stability rows, the
out-of-sample VaR, the block bootstrap standard errors (ch5_numbers.json) and the simulated maxima (numpy generator, seed 2026).
The script stops if a recomputed number no longer matches ch5_numbers.json (a revised data file, a changed rule or seed).
Run:  python3 Quantlets/Ch_05/export_interactive.py
"""
import json
import math
import os

import numpy as np
from scipy import stats

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch5_data.js')


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch5_numbers.json ({b})')


def r6(a):
    return [float(f'{x:.6g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def days(idx):
    d = idx.values.astype('datetime64[D]').astype(np.int64)
    return {'d0': int(d[0]), 'dd': np.diff(d).tolist()}


def check_samples(N):
    """The slides' samples: same length, same worst day, same tail index, same POT fit."""
    for k in g.ASSETS:
        r = g.returns(k)
        c = N['crash'][k]
        assert len(r) == c['n'] and r.index[0].date().isoformat() == c['first'], k
        close(float(r.min()), c['ret'], 1e-12, f'{k} worst day')
        close(float(r.std()), c['sd'], 1e-12, f'{k} sd')
        x = -r.values
        close(g.ls_tail_index(x, int(g.HILL_FRAC * len(x))), N['ls'][k], 1e-10, f'{k} least-squares slope')
    for k in g.ASSETS + g.STOCKS:
        x = g.losses(k).values
        h = N['hill'][k]
        assert len(x) == h['n'], k
        close(g.hill(x, h['k']), h['alpha'], 1e-12, f'{k} Hill alpha')
        f = g.pot(x)
        close(f['xi'], N['risk'][k]['pot']['xi'], 1e-6, f'{k} POT xi')
        close(f['beta'], N['risk'][k]['pot']['beta'], 1e-6, f'{k} POT beta')
        close(g.pot_var(f, 0.01), N['risk'][k]['var1']['EVT'], 1e-6, f'{k} EVT VaR 1%')
    for k in g.ASSETS:
        m = g.monthly_maxima(k).values
        assert len(m) == N['gev'][k]['months'], k
        G = g.fit_gev(m)
        close(G['xi'], N['gev'][k]['xi'], 1e-5, f'{k} GEV xi')
        close(g.return_level(G, 120), N['gev'][k]['rl10'], 1e-5, f'{k} 10-year return level')


def maxima_sim(N, ns=(2, 5, 10, 21, 50, 100, 250), reps=5000):
    """fig_maxima_sim of the chapter for several sample sizes n, each with the chapter's seed (n = 250 is the slide)."""
    res = {}
    lims = [(-3, 7), (0, 4), (-5, 0.2)]
    for n in ns:
        rng = np.random.default_rng(g.SEED)
        draws = [rng.exponential(1, (reps, n)).max(1) - np.log(n),
                 (rng.pareto(3, (reps, n)) + 1).max(1) / n ** (1 / 3),
                 n * (rng.uniform(0, 1, (reps, n)).max(1) - 1)]
        row = []
        for m, lim in zip(draws, lims):
            bins = np.linspace(*lim, 50)
            c, _ = np.histogram(m, bins=bins, density=True)
            row.append({'dens': r6(c), 'mean': float(np.mean(m)), 'out': int(((m < lim[0]) | (m > lim[1])).sum())})
        res[str(n)] = row
    for v, (lab, ref) in zip(res['250'], N['maxima_sim'].items()):
        close(v['mean'], ref, 1e-12, f'maxima_sim {lab}')
    return {'ns': list(ns), 'lims': lims, 'reps': reps, 'runs': res}


HILL_FRACS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)


def hill_block_se(N):
    """hill_inference of the chapter (500 moving-block bootstrap samples, seed 2026) at several k/n; k/n = 2.5% is the slide."""
    res = {}
    for k in g.ASSETS + g.STOCKS:
        x = g.losses(k).values
        res[k] = {}
        for fr in HILL_FRACS:
            h = g.hill_inference(x, frac=fr)
            res[k][f'{fr:.4f}'] = h['se_block']
            if abs(fr - g.HILL_FRAC) < 1e-12:
                close(h['se_block'], N['hill'][k]['se_block'], 1e-12, f'{k} block SE')
    return {'fracs': list(HILL_FRACS), 'se': res}


BLOCKS = {'week': ('W', 3, 52), 'quarter': ('Q', 45, 4), 'year': ('Y', 180, 1)}


def gev_blocks(N):
    """fit_gev of the chapter on weekly, quarterly and yearly maxima (extra runs; the slide uses monthly maxima).
    Blocks with fewer days than the minimum are dropped, as months with fewer than 15 days are in the chapter."""
    res = {}
    for k in g.ASSETS:
        L = g.losses(k)
        res[k] = {'month': {key: N['gev'][k][key] for key in ('xi', 'mu', 'sigma', 'se_xi', 'n')} | {'per_year': 12, 'min_days': 15}}
        for b, (freq, mn, py) in BLOCKS.items():
            grp = L.groupby(L.index.to_period(freq))
            m = grp.max()[grp.count() >= mn].values
            G = g.fit_gev(m)
            res[k][b] = {key: G[key] for key in ('xi', 'mu', 'sigma', 'se_xi', 'n')} | {'per_year': py, 'min_days': mn}
    return res


def gev_boot(N, B=200):
    """The bootstrap of gev_table (same generator, same order of the series): the GEV parameters of every resample,
    so that the page can give the 95% interval of the return level for any T; T = 10 years reproduces the slide."""
    rng = np.random.default_rng(g.SEED)
    res = {}
    for k in g.ASSETS:
        m = g.monthly_maxima(k).values
        G = N['gev'][k]
        bs = []
        for _ in range(B):
            s = rng.choice(m, len(m))
            try:
                c, mu, sig = stats.genextreme.fit(s, -G['xi'], loc=G['mu'], scale=G['sigma'])
                bs.append([float(-c), float(mu), float(sig)])
            except Exception:
                pass
        rl = [float(stats.genextreme.ppf(1 - 1 / 120, -x, mu, sig)) for x, mu, sig in bs]
        close(float(np.percentile(rl, 2.5)), G['rl10_lo'], 1e-9, f'{k} bootstrap 10-year level, lower')
        close(float(np.percentile(rl, 97.5)), G['rl10_hi'], 1e-9, f'{k} bootstrap 10-year level, upper')
        res[k] = bs
    return res


def main():
    N = json.load(open(os.path.join(HERE, 'ch5_numbers.json')))
    check_samples(N)
    d = dict(end=N['end'], assets=g.ASSETS, stocks=g.STOCKS, colors=g.COLORS, model_col=g.MODEL_COL, labels={k: g.LABELS[k] for k in g.ASSETS + g.STOCKS},
             hill_frac=g.HILL_FRAC, pot_q=g.POT_Q, split=g.SPLIT)
    d['loss'] = {}
    for k in g.ASSETS:
        r = g.returns(k)
        d['loss'][k] = {**days(r.index), 'x': r6(-r.values)}
    for k in g.STOCKS:
        d['loss'][k] = {'x': r6(g.losses(k).values)}
    d['mmax'] = {}
    for k in g.ASSETS:
        m = g.monthly_maxima(k)
        loc, sc = stats.gumbel_r.fit(m.values)            # extra run: the Gumbel (xi = 0) fit, for comparison with the GEV
        d['mmax'][k] = {'m': r6(m.values), 'ym': [f'{y}-{mo:02d}' for y, mo in m.index], 'gumbel': {'mu': loc, 'sigma': sc}}
    d['N'] = {k: N[k] for k in ('crash', 'crash_days', 'light_heavy', 'ls', 'hill', 'me', 'maxima_sim', 'gev', 'stab', 'gpd_qq', 'risk', 'oos', 'wait')}
    d['msim'] = maxima_sim(N)
    d['N']['c87'] = N['c87']
    d['hillse'] = hill_block_se(N)
    d['gevb'] = gev_blocks(N)
    d['gevboot'] = gev_boot(N)
    js = ('// Inputs of the Chapter 5 interactive page, written by Quantlets/Ch_05/export_interactive.py.\n'
          'window.CH5_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
