"""Export the inputs of the Chapter 4 interactive page (SFM/interactive/ch4_data.js).

Uses the functions of generate_all_charts.py (same returns, same seeds, same simulations), so the page starts from the
same numbers as the slides. Daily log returns are exported in % with 6 decimals; the browser computes the CDF, quantiles,
correlations, conditional moments and running means from them. Simulations that the browser cannot repeat with numpy's
generator are run here with the chapter's seeds and exported as drawn (or as the summaries the page needs). The script
stops if a recomputed number no longer matches ch4_numbers.json (a revised data file, a changed seed).
Run:  python3 Quantlets/Ch_04/export_interactive.py
"""
import json
import math
import os

import numpy as np
from scipy import stats

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch4_data.js')


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch4_numbers.json ({b})')


def r4(a, d=4):
    return [round(float(x), d) for x in np.asarray(a, float)]


def r6(a):
    return [float(f'{x:.6g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def days(idx):
    d = idx.values.astype('datetime64[D]').astype(np.int64)
    return {'d0': int(d[0]), 'dd': np.diff(d).tolist()}


# ------------------------------------------------------------------ 1. up days per week (S&P 500, DAX, BET)
def up_days(N):
    out = {}
    for k in ('sp500', 'dax', 'bet'):
        U = g.up_days_per_week(k)
        out[k] = {'p': U['p'], 'emp': r6(U['emp']), 'n_weeks': U['n_weeks'], 'mean': U['mean_emp'], 'var': U['var_emp'],
                  'cnt': [int(v) for v in U['K'].value_counts().reindex(range(6), fill_value=0).values]}
        if k == 'sp500':
            close(U['p'], N['rv']['p'], 1e-12, 'rv p')
            close(U['var_emp'], N['rv']['var_emp'], 1e-12, 'rv var_emp')
            assert U['n_weeks'] == N['rv']['n_weeks']
    return out


# ------------------------------------------------------------------ 3. joint returns on common days
def joint(N):
    out = {}
    for a, b in (('sp500', 'dax'), ('sp500', 'bet')):
        R = g.joint_returns(a, b)
        close(float(np.corrcoef(R[a], R[b])[0, 1]), N['joint'][f'{a}_{b}']['corr'], 1e-12, f'joint {a}_{b} corr')
        out[f'{a}_{b}'] = {**days(R.index), 'x': r4(R[a], 4), 'y': r4(R[b], 4)}
    return out


# ------------------------------------------------------------------ 3b. pair statistics on common days (all six pairs)
def pairs(N):
    import itertools
    out = {}
    for a, b in itertools.combinations(g.ASSETS, 2):
        P = g.pair_stats(a, b)
        if f'{a}_{b}' in N['joint']:
            for key in ('s1', 's2', 'cov', 'corr', 'sd_p'):
                close(P[key], N['joint'][f'{a}_{b}'][key], 1e-12, f'pair {a}_{b} {key}')
        out[f'{a}_{b}'] = {k: P[k] for k in ('n', 'm1', 'm2', 's1', 's2', 'cov', 'corr')}
    return out


# ------------------------------------------------------------------ 4. Y = X^2 (seed 4)
def xsq(N):
    X = np.random.default_rng(4).standard_normal(3000)
    close(float(np.corrcoef(X, X ** 2)[0, 1]), N['dep']['xy_corr'], 1e-12, 'xy_corr')
    return r4(X, 5)


# ------------------------------------------------------------------ 6. LLN: the five simulated paths (seed 11)
def lln(N, r):
    m, s, n = r.mean(), r.std(), len(r)
    rng = np.random.default_rng(11)
    grid = np.unique(np.round(np.logspace(0, np.log10(n), 700)).astype(int))
    k = np.arange(1, n + 1)
    P = []
    for _ in range(5):
        z = rng.normal(m, s, n)
        e = (z - m) / s                                 # numpy: normal(m, s) = m + s * standard normal
        P.append(r6((np.cumsum(e) / k)[grid - 1]))
    close(m / (s / np.sqrt(n)), N['lln']['t'], 1e-9, 'lln t')
    return {'grid': grid.tolist(), 'e': P}


# ------------------------------------------------------------------ 7. RANDU and PCG64 for odd seeds
def randu(N, seeds=range(1, 100, 2), n=30000):
    pcg = {}
    for sd in seeds:
        v = np.random.default_rng(sd).random(n)
        c = 9 * v[:-2] - 6 * v[1:-1] + v[2:]
        h, e = np.histogram(c, bins=150)                # matplotlib hist(bins=150): range = (min, max)
        pcg[sd] = {'lo': float(e[0]), 'hi': float(e[-1]), 'c': h.tolist(), 'nd': int(len(np.unique(c)))}
    x, u = g.randu(n, 1)
    comb = (9 * x[:-2] - 6 * x[1:-1] + x[2:]) // g.RANDU_M
    assert len(np.unique(comb)) == N['randu']['n_values']
    close(float(np.corrcoef(u[:-1], u[1:])[0, 1]), N['randu']['corr_pairs'], 1e-12, 'randu corr')
    return pcg


# ------------------------------------------------------------------ 8. inverse transform (seed 3)
def inverse(N, r):
    n = 200000
    U = np.random.default_rng(3).random(n)
    V = -np.log(1 - U)                                  # E = V / lambda
    fine = np.histogram(V, bins=2400, range=(0, 12))[0]
    m, s = r.mean(), r.std()
    Nn = m + s * stats.norm.ppf(U)
    H = np.quantile(r.values, U)
    bins = np.linspace(-12, 12, 121)
    xs = np.round(np.arange(1, 12.01, 0.5), 2)
    out = {'n': n, 'fine': fine.tolist(), 'meanV': float(V.mean()), 'edges': r6(bins),
           'hN': r6(np.histogram(Nn, bins=bins, density=True)[0]), 'hH': r6(np.histogram(H, bins=bins, density=True)[0]),
           'xs': xs.tolist(), 'cN': [int((Nn < -x).sum()) for x in xs], 'cH': [int((H < -x).sum()) for x in xs],
           'minN': float(Nn.min()), 'minH': float(H.min())}
    close(V.mean() / 2, N['inv']['mean_E'], 1e-12, 'inv mean_E')
    close(float((Nn < -5).mean()), N['inv']['p5_norm'], 1e-12, 'inv p5_norm')
    close(float((H < -5).mean()), N['inv']['p5_hist'], 1e-12, 'inv p5_hist')
    return out


# ------------------------------------------------------------------ 9. Monte Carlo, several loss thresholds (seed 5)
def monte_carlo(N, r, h=21, Nmax=10 ** 6):
    m, s = r.mean(), r.std()
    sims = np.random.default_rng(5).normal(h * m, s * np.sqrt(h), Nmax)
    grid = np.unique(np.logspace(1, 6, 400).astype(int)) - 1
    Ns = np.arange(1, Nmax + 1)
    out = {'h': h, 'Ns': (grid + 1).tolist(), 'losses': list(range(5, 21)), 'est': {}, 'at': {}}
    for L in out['losses']:
        hits = np.cumsum(sims < -L)
        out['est'][L] = r6((hits / Ns)[grid])
        out['at'][L] = {str(k): int(hits[k - 1]) for k in (100, 1000, 10000, 100000, 1000000)}
        if L == 10:
            for k in (100, 1000, 10000, 100000, 1000000):
                close(hits[k - 1] / k, N['mc'][f'est_{k}'], 1e-12, f'mc est_{k}')
    return out


# ------------------------------------------------------------------ 10. binomial paths for several n (seed 8)
def binomial(N, r, ns=(2, 5, 10, 21, 63, 126, 252), S0=100.0, n_paths=12):
    gp = g.gbm_params(r)
    out = {'mu': gp['mu'], 'sigma': gp['sigma'], 'paths': {}}
    for n in ns:
        B = g.binomial_crr(S0, gp['mu'], gp['sigma'], 1.0, n)
        rng = np.random.default_rng(8)
        steps = np.where(rng.random((n_paths, n)) < B['p'], np.log(B['u']), np.log(B['d']))
        paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)]))
        out['paths'][n] = [r4(p, 3) for p in paths]
        if n == 252:
            close(B['p'], N['binom']['p'], 1e-12, 'binom p')
            close(float(B['prob'][B['S'] < S0].sum()), N['binom']['ploss_bin'], 1e-12, 'binom ploss')
    return out


# ------------------------------------------------------------------ 11. shocks of the processes (seed 21)
def processes(N):
    return r4(np.random.default_rng(21).standard_normal(500), 5)


# ------------------------------------------------------------------ 12. which one is real (seed 31)
def spot(N):
    P = g.load_close('sp500', g.START)
    r = 100 * np.log(P).diff().dropna()
    m, s = r.mean(), r.std()
    rng = np.random.default_rng(31)
    sims = [rng.normal(m, s, len(r)) for _ in range(3)]
    close(float(stats.kurtosis(r)), N['spot']['kurt'][2], 1e-12, 'spot kurt')
    close(float(stats.kurtosis(sims[0])), N['spot']['kurt'][0], 1e-12, 'spot kurt A')
    return {'lp0': float(np.log(P.iloc[0])), 'sims': [r4(x, 3) for x in sims]}


# ------------------------------------------------------------------ 13. Wiener (seed 13)
def wiener(N, n_paths=20):
    rng = np.random.default_rng(13)
    rw = {}
    for n in (10, 100, 10000):
        t, w = g.scaled_random_walk(n, rng)
        rw[n] = np.round(w * np.sqrt(n)).astype(int).tolist()     # integer partial sums of +/-1
    steps = 1000
    dW = rng.standard_normal((n_paths, steps)) * np.sqrt(1 / steps)
    W = np.hstack([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)])
    big = np.cumsum(np.random.default_rng(14).standard_normal((10000, 1000)) * np.sqrt(1 / 1000), axis=1)
    ks = np.round(np.arange(0.5, 3.001, 0.1), 2)
    share = [float(np.mean(np.abs(big[:, -1]) < k)) for k in ks]
    close(share[5], N['wiener']['share_in_1'], 1e-12, 'wiener share_in_1')
    close(float(big[:, -1].var()), N['wiener']['var_W1'], 1e-12, 'wiener var_W1')
    return {'rw': rw, 'W': [r4(w[::2], 3) for w in W], 'ks': ks.tolist(), 'share': share, 'endW': r4(W[:, -1], 5)}


# ------------------------------------------------------------------ 14. GBM paths (seed 17): W(t) on every 5th step
def gbm_paths(N, r, T=10.0, n_paths=40):
    gp = g.gbm_params(r)
    steps = int(T * 252)
    z = np.random.default_rng(17).standard_normal((n_paths, steps))
    W = np.hstack([np.zeros((n_paths, 1)), np.cumsum(np.sqrt(T / steps) * z, axis=1)])
    S = g.simulate_gbm(100.0, gp['mu'], gp['sigma'], T, steps, n_paths, 17)
    t = np.linspace(0, T, steps + 1)
    S2 = 100 * np.exp((gp['mu'] - gp['sigma'] ** 2 / 2) * t + gp['sigma'] * W)
    assert np.max(np.abs(np.log(S2 / S))) < 1e-9
    idx = np.arange(0, steps + 1, 10)
    close(gp['sigma'], N['gbm']['sigma'], 1e-12, 'gbm sigma')
    return {'T': T, 'steps': steps, 'idx': idx.tolist(), 'W': [r4(w[idx], 4) for w in W], 'mu': gp['mu'], 'sigma': gp['sigma'],
            'share_below': N['gbm']['share_below_mean']}


# ------------------------------------------------------------------ 15. GBM fans (seed 19): empirical quantiles and ranks
BANDS = (0.5, 0.8, 0.9, 0.95, 0.99)
LEV = [0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.995]


def gbm_fan(N, ret, n_paths=4000, every=8):
    out = {}
    for k in ('sp500', 'bet', 'btc'):
        P = g.load_close(k, max(g.START, g.SERIES[k][4]))
        r = 100 * np.log(P).diff().dropna()
        n = len(r)
        rng = np.random.default_rng(19)
        z = rng.normal(r.mean() / 100, r.std() / 100, (n_paths, n))
        lp = np.log(P.iloc[0]) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(z, axis=1)])
        del z
        idx = np.unique(np.concatenate([np.arange(0, n + 1, every), [n]]))
        q = np.quantile(lp[:, idx], LEV, axis=0)
        cnt = (lp < np.log(P.values)[None, :]).sum(axis=0)
        rank = cnt / n_paths
        o90 = float(np.mean((rank < 0.05) | (rank > 0.95)))
        close(o90, N['fan'][k]['outside90'], 1e-12, f'fan {k} outside90')
        assert len(ret[k]) == n and (ret[k].index == r.index).all(), k
        outside = {str(b): float(np.mean((rank < (1 - b) / 2) | (rank > (1 + b) / 2))) for b in BANDS}
        close(outside['0.9'], o90, 1e-12, f'fan {k} band 90')
        out[k] = {'lp0': float(np.log(P.iloc[0])), 'every': every, 'q': [r4(v, 3) for v in q], 'outside': outside,
                  'sigma': N['fan'][k]['sigma']}
        del lp
    out['lev'] = LEV
    return out


# ------------------------------------------------------------------ 16. what GBM misses (seed 23): add the k-sigma days
def gbm_check(N, ks=np.round(np.arange(3, 8.01, 0.5), 2), n_sims=300):
    out = {'ks': ks.tolist()}
    for k in ('sp500', 'bet', 'btc'):
        P = g.load_close(k, max(g.START, g.SERIES[k][4]))
        r = 100 * np.log(P).diff().dropna()
        m, s, n = r.mean(), r.std(), len(r)
        rng = np.random.default_rng(23)
        K, A, DD, F = [], [], [], []
        for _ in range(n_sims):
            x = rng.normal(m, s, n)
            K.append(stats.kurtosis(x))
            A.append(g.acf(x ** 2, 1)[0])
            DD.append(g.max_drawdown(np.exp(np.cumsum(x) / 100)))
            d = np.abs(x - x.mean()) / x.std()
            F.append([int((d > kk).sum()) for kk in ks])
        F = np.array(F)
        C = N['check'][k]
        for key, v in (('kurt', K), ('acf2', A), ('dd', DD)):
            close(float(np.quantile(v, 0.05)), C[key]['q05'], 1e-12, f'check {k} {key} q05')
        dr = np.abs(r.values - m) / s
        out[k] = {'real': C['real'], 'kurt': C['kurt'], 'acf2': C['acf2'], 'dd': C['dd'], 'n': n,
                  'fr': [int((dr > kk).sum()) for kk in ks],
                  'f05': np.quantile(F, 0.05, axis=0).tolist(), 'f50': np.median(F, axis=0).tolist(),
                  'f95': np.quantile(F, 0.95, axis=0).tolist(), 'fmax': F.max(axis=0).tolist()}
        assert out[k]['fr'][2] == C['real']['four'], k
    return out


def main():
    N = json.load(open(os.path.join(HERE, 'ch4_numbers.json')))
    ret = {k: g.returns(k) for k in g.ASSETS}
    for k, r in ret.items():                             # the slides' samples: same length and moments
        assert len(r) == N['dep'][k]['n'], k
        close(g.lag_dependence(r)['rho_r2'], N['dep'][k]['rho_r2'], 1e-12, f'{k} rho_r2')
    sp = ret['sp500']
    close(float(sp.mean()), N['cdf']['mean'], 1e-12, 'sp500 mean')
    close(float(sp.std()), N['cdf']['sd'], 1e-12, 'sp500 sd')
    close(float(np.quantile(sp, 0.01)), N['cdf']['q01_emp'], 1e-12, 'sp500 q01')
    for k in ('sp500', 'btc'):
        c = g.conditional_moments(ret[k])
        close(c['ratio'], N['cond'][k]['ratio'], 1e-12, f'cond {k} ratio')
    d = dict(start=N['start'], end=N['end'], assets=g.ASSETS, colors=g.COLORS, short=g.SHORT, pal=g.PAL, palette=g.st.PALETTE)
    d['ret'] = {k: {**days(r.index), 'r': np.round(r.values, 5).tolist()} for k, r in ret.items()}
    d['N'] = {k: N[k] for k in ('down', 'rv', 'cdf', 'dep', 'cond', 'lln', 'randu', 'inv', 'mc', 'binom', 'proc', 'spot', 'wiener', 'gbm', 'fan')}
    d['up'] = up_days(N)
    d['joint'] = joint(N)
    d['pairs'] = pairs(N)
    d['xsq'] = xsq(N)
    d['lln'] = lln(N, sp)
    d['pcg'] = randu(N)
    d['inv'] = inverse(N, sp)
    d['mc'] = monte_carlo(N, sp)
    d['bin'] = binomial(N, sp)
    d['eps'] = processes(N)
    d['spot'] = spot(N)
    d['wien'] = wiener(N)
    d['gbm'] = gbm_paths(N, sp)
    d['fan'] = gbm_fan(N, ret)
    d['chk'] = gbm_check(N)
    js = ('// Inputs of the Chapter 4 interactive page, written by Quantlets/Ch_04/export_interactive.py.\n'
          'window.CH4_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
