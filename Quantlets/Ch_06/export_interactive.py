"""Export the inputs of the Chapter 6 interactive page (SFM/interactive/ch6_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same fitted parameters, same seeds), so the page
starts from the numbers of the slides. Daily log returns (in %) are exported with 6 significant digits; the browser computes
from them the descriptive statistics, the Normal fits, the empirical VaR and ES, the exceedance counts, the Kupiec and
Christoffersen statistics, the KS gaps, the pinball losses and the RMA/EMA VaR of SFEVaRqqplot. What the browser cannot repeat
comes from the chapter's code: the fitted parameters and log-likelihoods (ch6_numbers.json), the model quantiles and
distribution functions (cdf, ppf, var_es with those parameters), the profile log-likelihood of nu, the daily log-density
differences of the Vuong test, the rolling VaR refits and the moving-block bootstrap of the AIC winner (numpy generator,
seed 2026). Levels other than VaR 1% / ES 2.5%, the profile for the other series and the finer profile interval are extra
runs of the same functions.
The script stops if a recomputed number no longer matches ch6_numbers.json (a revised data file, a changed rule or seed).
Run:  python3 Quantlets/Ch_06/export_interactive.py
"""
import json
import math
import os

import numpy as np
from scipy import stats

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch6_data.js')
NAMES = g.ASSETS + g.STOCKS
LEVELS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch6_numbers.json ({b})')


def r6(a, d=6):
    return [float(f'{x:.{d}g}') for x in np.asarray(a, float)]


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


def P(N, k, m):
    return N['fits'][k]['models'][m]['params']


def check_samples(N):
    """The slides' samples: same length, same moments, same Normal fit, same dates."""
    for k in NAMES:
        r = g.returns(k)
        s = N['daily'][k]
        assert len(r) == s['n'] == N['fits'][k]['n'], k
        assert r.index[0].date().isoformat() == s['first'] and r.index[-1].date().isoformat() == s['last'], k
        close(float(r.std(ddof=1)), s['sd'], 1e-12, f'{k} sd')
        close(float(stats.kurtosis(r.values, fisher=False)), s['kurt'], 1e-10, f'{k} kurtosis')
        close(g.fit_model('Normal', r.values)['loglik'], N['fits'][k]['models']['Normal']['loglik'], 1e-10, f'{k} Normal loglik')
        close(g.fit_model('Student-t', r.values)['loglik'], N['fits'][k]['models']['Student-t']['loglik'], 1e-8, f'{k} Student-t loglik')


def profile(k, N):
    """Profile log-likelihood of nu (profile_nu of the chapter: same grid, same fits) and the interval from a finer search."""
    x = g.returns(k).values
    grid = np.round(np.arange(2.2, 6.01, 0.1), 2)
    ll = np.array([np.sum(stats.t.logpdf(x, nu, *stats.t.fit(x, fdf=nu)[1:])) for nu in grid])
    f = g.t_fit(x)
    lmax = f['loglik']
    prof = lambda nu: np.sum(stats.t.logpdf(x, nu, *stats.t.fit(x, fdf=nu)[1:])) - lmax
    res = {'grid': grid.tolist(), 'll': r6(ll - lmax), 'nu': f['nu'], 'lmax': lmax, 'ci': {}, 'grid_ci': {}}
    fine = np.round(np.arange(max(1.6, f['nu'] - 1.2), f['nu'] + 2.0, 0.01), 2)      # extra run: the same profile on a grid of step 0.01
    pf = np.array([prof(v) for v in fine])
    res['fine'] = {'nu': fine.tolist(), 'll': r6(pf)}
    for lev in (0.90, 0.95, 0.99):
        c = -stats.chi2.ppf(lev, 1) / 2
        inside = grid[ll - lmax >= c]
        res['grid_ci'][str(lev)] = [float(inside.min()), float(inside.max())]
        j = np.where(pf >= c)[0]
        lo, hi = j.min(), j.max()
        cross = lambda i0, i1: float(fine[i0] + (c - pf[i0]) * (fine[i1] - fine[i0]) / (pf[i1] - pf[i0]))
        res['ci'][str(lev)] = [cross(lo - 1, lo) if lo > 0 else None, cross(hi, hi + 1) if hi + 1 < len(fine) else None]
    if k == 'sp500':
        R = N['profile']
        close(f['nu'], R['nu'], 1e-10, 'profile nu')
        assert res['grid_ci']['0.95'] == [R['lo'], R['hi']], res['grid_ci']
        close(float(np.interp(5.0, grid, ll) - lmax), R['ll_5'], 1e-8, 'profile l(5)')
        close(float(np.interp(3.0, grid, ll) - lmax), R['ll_3'], 1e-8, 'profile l(3)')
    return res


PAIRS = {'vu_t_ged': ('Student-t', 'GED'), 'vu_t_stable': ('Student-t', 'Stable'), 'vu_skt_nig': ('Skewed-t', 'NIG')}


def vuong_parts(k, N):
    """Daily log-density differences m_t of the three Vuong comparisons: mean, sd and the cumulative sum over time."""
    r = g.returns(k)
    x = r.values
    out = {}
    step = int(np.ceil(len(x) / 600))
    for key, (a, b) in PAIRS.items():
        m = g.logpdf(a, x, P(N, k, a)) - g.logpdf(b, x, P(N, k, b))
        v = np.sqrt(len(m)) * m.mean() / m.std(ddof=1)
        close(v, N['tests'][k][key]['v'], 1e-6, f'{k} Vuong {key}')
        cs = np.cumsum(m)
        idx = np.unique(np.r_[np.arange(0, len(x), step), len(x) - 1])
        out[key] = {'mbar': float(m.mean()), 'sm': float(m.std(ddof=1)), 'n': len(m), 'idx': idx.tolist(), 'cum': r6(cs[idx], 4),
                    'top': [[int(i), float(m[i])] for i in np.argsort(-np.abs(m))[:3]]}
    return out


def safe_ppf(m, ps, p):
    """g.ppf point by point; where scipy's generic inversion fails (NIG far in the tails: its cdf breaks down for large x),
    the root of the tail probability, computed by integrating the model density (lower tail: g.cdf)."""
    from scipy import integrate, optimize
    out = []
    for q in np.atleast_1d(ps):
        try:
            v = float(np.ravel(g.ppf(m, q, p))[0])
            if not np.isfinite(v):
                raise ValueError
        except ValueError:
            mid = float(np.ravel(g.ppf(m, 0.5, p))[0])
            if q < 0.5:
                v = optimize.brentq(lambda z: float(np.ravel(g.cdf(m, np.array([z]), p))[0]) - q, -1000, mid, xtol=1e-10)
            else:
                up = lambda z: integrate.quad(lambda t: np.exp(g.logpdf(m, np.array([t]), p)[0]), z, np.inf, epsabs=1e-14)[0]
                v = optimize.brentq(lambda z: up(z) - (1 - q), mid, 1000, xtol=1e-10)
        out.append(v)
    return np.array(out)


def ppqq(k, N):
    """PP and QQ coordinates of every model at the points the chapter plots (union of the two selections of fig_pp_qq and
    fig_qq_assets), the largest PP deviation over all days and the model quantile at p = 0.5/n."""
    x = np.sort(g.returns(k).values)
    n = len(x)
    pe = (np.arange(1, n + 1) - 0.5) / n
    s1 = np.unique(np.concatenate([np.linspace(0, n - 1, 400).astype(int), np.arange(30), np.arange(n - 30, n)]))
    s2 = np.unique(np.concatenate([np.linspace(0, n - 1, 300).astype(int), np.arange(25), np.arange(n - 25, n)]))
    sel = np.union1d(s1, s2)
    out = {'sel': sel.tolist(), 's1': np.searchsorted(sel, s1).tolist(), 's2': np.searchsorted(sel, s2).tolist(), 'm': {}}
    for m in g.MODELS:
        p = P(N, k, m)
        u = g.cdf(m, x, p)
        q = safe_ppf(m, pe[sel], p) if m in ('NIG', 'GED') else np.ravel(g.ppf(m, pe[sel], p))
        out['m'][m] = {'u': r6(u[sel], 4), 'q': r6(q, 4), 'maxdev': float(np.max(np.abs(u - pe))), 'qmin': float(safe_ppf(m, 0.5 / n, p)[0])}
    if k == 'sp500':
        for m in g.PANEL:
            close(out['m'][m]['maxdev'], N['ppqq'][m]['pp_maxdev'], 1e-9, f'pp {m}')
            close(out['m'][m]['qmin'], N['ppqq'][m]['q0001'], 1e-9, f'qq {m}')
    return out


def left_tail(k, N):
    x = g.returns(k).values
    L = np.sort(-x)[::-1]
    grid = np.logspace(np.log10(0.5), np.log10(L.max() * 1.6), 80)
    return {'grid': r6(grid), 'm': {m: r6(np.clip(g.cdf(m, -grid, P(N, k, m)), 1e-9, None), 5) for m in g.MODELS}}


def tails(k, N):
    """VaR and ES of every model at LEVELS (var_es of the chapter; 1% and 2.5% are the slide's)."""
    out = {}
    for m in g.MODELS:
        v, e = [], []
        for a in LEVELS:
            var, es = g.var_es(m, P(N, k, m), a_var=a, a_es=a)
            v.append(var)
            e.append(es)
        close(v[LEVELS.index(0.01)], N['tails'][k][m]['var1'], 1e-9, f'{k} {m} VaR 1%')
        close(e[LEVELS.index(0.025)], N['tails'][k][m]['es25'], 1e-9, f'{k} {m} ES 2.5%')
        out[m] = {'var': v, 'es': e}
    return out


def rolling(k, N, key):
    df, res = g.rolling_var(k)
    for m in ('Normal', 'Student-t', 'Skewed-t', 'Historical'):
        assert res[m]['x'] == N[key][m]['x'] and res[m]['n'] == N[key][m]['n'], (k, m)
    r = g.returns(k)
    i0 = r.index.get_loc(df.index[0])
    steps = list(range(0, len(df), 20))
    return {'i0': int(i0), 'step': 20, 'm': {m: r6(df[m].values[steps]) for m in ('Normal', 'Student-t', 'Skewed-t', 'Historical')}}


def uncertainty(N, k='sp500', B=200, models=('Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG')):
    """The loop of model_uncertainty (same generator, same order of draws), keeping every bootstrap VaR 1%."""
    x = g.returns(k).values
    rng = np.random.default_rng(g.SEED)
    wins = {m: 0 for m in models}
    var = {m: [] for m in models}
    winner = []
    for _ in range(B):
        y = g.block_bootstrap(x, rng=rng)
        aic = {}
        for m in models:
            p = g.fit_model(m, y)
            aic[m] = 2 * g.K[m] - 2 * p['loglik']
            var[m].append(-float(np.ravel(g.ppf(m, g.ALPHA_VAR, p))[0]))
        wins[min(aic, key=aic.get)] += 1
        winner.append(models.index(min(aic, key=aic.get)))
    U = N['uncertainty']
    for m in models:
        close(wins[m] / B, U['wins'][m], 1e-12, f'bootstrap wins {m}')
        close(float(np.median(var[m])), U['var'][m]['median'], 1e-9, f'bootstrap median VaR {m}')
        close(float(np.quantile(var[m], 0.05)), U['var'][m]['q05'], 1e-9, f'bootstrap q05 VaR {m}')
    return {'wins': {m: wins[m] / B for m in models}, 'models': list(models), 'winner': winner, 'var': {m: r6(v, 5) for m, v in var.items()}}


def main():
    N = json.load(open(os.path.join(HERE, 'ch6_numbers.json')))
    check_samples(N)
    d = dict(end=N['end'], names=NAMES, assets=g.ASSETS, stocks=g.STOCKS, models=g.MODELS, K=g.K, mcol=g.MODEL_COL, scol=g.COLORS,
             short=g.SHORT, labels={k: g.LABELS[k] for k in NAMES}, split=g.SPLIT, levels=LEVELS)
    print('returns')
    d['ret'] = {}
    for k in NAMES:
        r = g.returns(k)
        d['ret'][k] = {**days(r.index), 'x': r6(r.values)}
    raw = g.log_returns('tlv', g.STOCK_START)
    clean_idx = g.returns('tlv').index
    d['tlv_bad'] = [{'date': dd, 'x': float(raw.loc[dd]), 'pos': int(np.searchsorted(clean_idx.values, np.datetime64(dd)))} for dd in g.BAD_DAYS['tlv']]
    for b, v in zip(d['tlv_bad'], N['tlv']['r_bad']):
        close(b['x'], v, 1e-12, 'TLV bad day')
    m = g.monthly_first_day('dax', '2004-05-07')
    book = m.loc['2004-07':'2014-05']
    close(g.summary_stats(book.values, 12)['kurt'], N['dax_monthly']['book']['kurt'], 1e-10, 'DAX monthly kurtosis')
    d['dax_m'] = {'ym': [t.date().isoformat() for t in m.index], 'x': r6(100 * m.values),
                  'book': [int(m.index.get_loc(book.index[0])), int(m.index.get_loc(book.index[-1]))]}
    print('candidates')
    S = g.standard_candidates()
    xg = np.round(np.linspace(-7, 7, 701), 4)
    thr = np.arange(2, 6.01, 0.5)
    d['cand'] = {'x': xg.tolist(), 'f': {mm: r6(np.exp(g.logpdf(mm, xg, S[mm])), 5) for mm in g.MODELS}, 'thr': thr.tolist(),
                 'below': {mm: [float(np.ravel(g.cdf(mm, np.array([-t]), S[mm]))[0]) for t in thr] for mm in g.MODELS}}
    for mm in g.MODELS:
        close(d['cand']['below'][mm][4], N['candidates'][mm]['p_below_m4'], 1e-9, f'candidate {mm}')
    d['fits'] = {k: {'n': N['fits'][k]['n'], 'models': {mm: {'params': v['params'], 'll': v['loglik'], 'k': v['k']} for mm, v in N['fits'][k]['models'].items()}} for k in NAMES}
    print('profile')
    d['prof'] = {k: profile(k, N) for k in NAMES}
    print('vuong')
    d['vu'] = {k: vuong_parts(k, N) for k in NAMES}
    print('pp qq')
    d['pq'] = {k: ppqq(k, N) for k in NAMES}
    d['lt'] = {k: left_tail(k, N) for k in NAMES}
    print('tails')
    d['tails'] = {k: tails(k, N) for k in NAMES}
    print('rolling')
    d['roll'] = {'sp500': rolling('sp500', N, 'rolling'), 'bet': rolling('bet', N, 'rolling_bet')}
    print('bootstrap')
    d['unc'] = uncertainty(N)
    d['N'] = {k: N[k] for k in ('daily', 'dax_monthly', 'tlv', 'tests', 'profile', 'gof', 'gof_boot', 'lillie', 'power', 'ks_tail', 'ppqq',
                                'tails', 'avg_var', 'oos', 'overfit', 'rolling', 'rolling_bet', 'var_qq', 'uncertainty', 'candidates')}
    js = ('// Inputs of the Chapter 6 interactive page, written by Quantlets/Ch_06/export_interactive.py.\n'
          'window.CH6_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
