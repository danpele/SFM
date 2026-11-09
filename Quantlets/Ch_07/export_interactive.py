"""Export the inputs of the Chapter 7 interactive page (SFM/interactive/ch7_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same seeds), so the page starts from the numbers of
the slides. Daily log returns (in %) are exported with 6 significant digits and the log prices to 1e-7; the browser computes from
them the autocorrelations and their bands, the Box-Pierce, Ljung-Box and robust portmanteau statistics, the runs test, the
Dickey-Fuller and ADF regressions (at the lag order chosen by AIC in the chapter's code), Phillips-Perron and KPSS, the variance
ratios with Z and Z*, Chow-Denning, the automatic VR statistic, the rolling and sub-period tests and the calendar regressions.
What the browser does not repeat comes from the chapter's code: the simulated random walks, the Gaussian white noise and the
ARMA paths (numpy generators with the chapter's seeds), the spurious-regression Monte Carlo, the size simulation of Z and Z*,
the ADF lag orders chosen by AIC, the MacKinnon p-values and the wild-bootstrap p-values of the automatic VR test.
Extra runs of the same functions: the spurious regression for T other than 500 and the size simulation for GARCH reactions
alpha other than 0.12.
The script stops if a recomputed number no longer matches ch7_numbers.json (a revised data file, a changed rule or seed).
Run:  python3 Quantlets/Ch_07/export_interactive.py
"""
import json
import math
import os

import numpy as np

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch7_data.js')
NAMES = g.ASSETS + g.STOCKS
MARKETS = g.DEVELOPED + g.EMERGING + g.CRYPTO
SPUR_T = [100, 250, 500, 1000, 2000]
GARCH_A = [0.0, 0.03, 0.06, 0.09, 0.12]


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch7_numbers.json ({b})')


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


def check(N):
    """The slides' samples and tests: same lengths, same first autocorrelation, same portmanteau and VR statistics."""
    for k in NAMES:
        r = g.returns(k)
        x = r.values
        assert len(x) == N['tests'][k]['n'] == N['vr'][k]['n'], k
        close(float(g.acf(x, 1)[0]), N['tests'][k]['ret']['rho1'], 1e-12, f'{k} rho1')
        close(g.portmanteau(x)['lb'], N['tests'][k]['ret']['lb'], 1e-10, f'{k} LB')
        close(g.runs_test(x)['z'], N['tests'][k]['runs']['z'], 1e-10, f'{k} runs')
        for q in g.QS:
            close(g.variance_ratio(x, q)['zs'], N['vr'][k]['vr'][str(q)]['zs'], 1e-10, f'{k} Z*({q})')
        close(g.adf_test(g.log_price(k).values, 'ct')['stat'], N['ur'][k]['price']['adf']['stat'], 1e-8, f'{k} ADF price')
    for k in MARKETS:
        close(g.variance_ratio(g.returns(k, g.MARKETS_START).values, 5)['vr'], N['markets'][k]['vr'], 1e-10, f'{k} VR(5) markets')


def rw_paths(N, step=5):
    """fig_rw_paths: the four random walks (same generator, same order of draws), every `step`-th day and the last day."""
    lp = g.log_price('sp500')
    x = np.diff(lp.values)
    rng = np.random.default_rng(g.SEED)
    sims = [np.r_[0, np.cumsum(rng.normal(x.mean(), x.std(ddof=1), len(x)))] for _ in range(4)]
    for s, v in zip(sims, N['rw']['sim_final']):
        close(float(s[-1]), v, 1e-12, 'random walk final value')
    close(float(lp.values[-1] - lp.values[0]), N['rw']['final'], 1e-12, 'S&P 500 final log price')
    idx = np.unique(np.r_[np.arange(0, len(lp), step), len(lp) - 1])
    return {'idx': idx.tolist(), 'sims': [r6(s[idx], 5) for s in sims], 'mu': float(x.mean()), 'sd': float(x.std(ddof=1))}


def white_noise(N):
    wn = np.random.default_rng(10).standard_normal(1000)
    b = 1.96 / np.sqrt(1000)
    assert int(np.sum(np.abs(g.acf(wn, 30)) > b)) == N['wn']['outside']
    return r6(wn)


def arma_paths(N):
    """fig_acf_processes: one simulated path of n = 1000 per process, seed 123, in the order of ACF_CASES."""
    rng = np.random.default_rng(123)
    out = []
    for lab, ar, ma in g.ACF_CASES:
        th = g.arma_acf(ar, ma, 20)
        for a, b in zip(th[:5], N['arma'][lab]):
            close(float(a), b, 1e-12, f'ARMA {lab}')
        out.append({'lab': lab, 'ar': list(ar), 'ma': list(ma), 'x': r6(g.simulate_arma(ar, ma, 1000, rng=rng), 6)})
    return out


def spurious_runs(N):
    """spurious() of the chapter for several T (T = 500 is the slide's)."""
    res = {}
    for T in SPUR_T:
        sp = g.spurious(T=T)
        if T == 500:
            for lab in ('level', 'diff'):
                for c in ('reject', 'r2_med', 'beyond40'):
                    close(sp[lab][c], N['spurious'][lab][c], 1e-12, f'spurious {lab} {c}')
        y, x = sp['example']
        res[str(T)] = {'y': r6(y, 5), 'x': r6(x, 5), **{lab: {'t': r6(sp[lab]['t'], 4), 'reject': sp[lab]['reject'], 'r2_med': sp[lab]['r2_med'],
                                                               'beyond40': sp[lab]['beyond40']} for lab in ('level', 'diff')}}
        print('  spurious T =', T)
    return res


def vr_size_a(a, T=2000, reps=1000, qs=(2, 5, 10), seed=g.SEED):
    """vr_size of the chapter with the GARCH reaction a (b = 0.86 and omega = 0.04 as in simulate_garch)."""
    rng = np.random.default_rng(seed)
    out = {'iid': {q: [0, 0] for q in qs}, 'garch': {q: [0, 0] for q in qs}}
    for _ in range(reps):
        for lab, x in [('iid', rng.standard_normal(T)), ('garch', g.simulate_garch(T, a=a, rng=rng))]:
            for q in qs:
                v = g.variance_ratio(x, q)
                out[lab][q][0] += abs(v['z']) > 1.96
                out[lab][q][1] += abs(v['zs']) > 1.96
    return {lab: {str(q): {'z': c[0] / reps, 'zs': c[1] / reps} for q, c in d.items()} for lab, d in out.items()}


def lp_int(k):
    """Log price rounded to 1e-7, as the first value and the integer differences."""
    lp = g.log_price(k)
    v = np.round(lp.values * 1e7).astype(np.int64)
    return {**days(lp.index), 'p0': int(v[0]), 'dp': np.diff(v).tolist()}


def main():
    N = json.load(open(os.path.join(HERE, 'ch7_numbers.json')))
    print('check')
    check(N)
    d = dict(end=N['end'], names=NAMES, assets=g.ASSETS, stocks=g.STOCKS, markets=MARKETS, name=g.NAME, col=g.COLORS, qs=list(g.QS),
             developed=g.DEVELOPED, emerging=g.EMERGING, crypto=g.CRYPTO, mstart=g.MARKETS_START, window=g.WINDOW, step=g.STEP,
             subperiods=g.SUBPERIODS, bad=g.BAD_DAYS)
    print('returns, log prices')
    d['ret'] = {k: {**days(g.returns(k).index), 'x': r6(g.returns(k).values)} for k in NAMES}
    d['lp'] = {k: lp_int(k) for k in NAMES}
    d['mk'] = {k: {**days(g.returns(k, g.MARKETS_START).index), 'x': r6(g.returns(k, g.MARKETS_START).values)} for k in MARKETS}
    print('simulations')
    d['rw'] = rw_paths(N)
    d['wn'] = white_noise(N)
    d['arma'] = arma_paths(N)
    d['spur'] = spurious_runs(N)
    d['size'] = {}
    for a in GARCH_A:
        s = vr_size_a(a)
        if a == 0.12:
            for lab in ('iid', 'garch'):
                for q in ('2', '5', '10'):
                    for c in ('z', 'zs'):
                        close(s[lab][q][c], N['vr_size'][lab][q][c], 1e-12, f'size {lab} {q} {c}')
        d['size'][f'{a:.2f}'] = s
        print('  size a =', a)
    d['adf_lags'] = {k: {'price': N['ur'][k]['price']['adf']['lags'], 'ret': N['ur'][k]['ret']['adf']['lags']} for k in NAMES}
    d['N'] = {k: N[k] for k in ('scatter', 'rw', 'wn', 'arma', 'acf', 'acf_abs', 'tests', 'ur', 'spurious', 'vr', 'vr_profile', 'vr_size', 'markets',
                                'rolling', 'sub', 'calendar', 'cd_crit', 'df_crit')}
    js = ('// Inputs of the Chapter 7 interactive page, written by Quantlets/Ch_07/export_interactive.py.\n'
          'window.CH7_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
