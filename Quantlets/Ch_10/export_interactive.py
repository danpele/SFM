"""Export the inputs of the Chapter 10 interactive page (SFM/interactive/ch10_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same seeds, the arch package), so the page starts
from the numbers of the slides. Daily log returns (in %) of the six series and the simple returns of the BET and S&P 500
portfolio are exported with 9 significant digits; the browser recomputes from them the historical, Normal, Cornish-Fisher and
EVT VaR and ES, the rolling historical and Normal forecasts (500 days), the correlations, the pseudo-observations and the
tail dependence. The fitted Student-t, GPD and GARCH(1,1)-t parameters come from the chapter's functions; the rolling GARCH-t
forecasts are exported as the conditional volatility of each day and, for each 250-day re-estimation, the mean, the degrees of
freedom and the quantiles and ES of the standardised residuals (FHS). From them the browser rebuilds every one-day VaR 1%, VaR 2.5%
and ES 2.5% forecast and recomputes the exceedances, the Kupiec, Christoffersen and Basel results, the quantile loss and Z2.
The p-values of Z2 (simulation, 1000 draws) and the copula simulations (200,000 draws) come from the chapter's functions.
Extra runs of the same functions: other EVT thresholds, other bootstrap windows, Z2 p-values for later evaluation starts, the
portfolio with other weights, the copula samples with other seeds. The script recomputes the slide numbers from the exported
(rounded) values with the formulas of the page and stops if one no longer matches ch10_numbers.json. It writes only
interactive/ch10_data.js.
Run:  python3 Quantlets/Ch_10/export_interactive.py
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
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch10_data.js')
MK = g.ASSETS
BT = g.BT_ASSETS
METHODS = g.METHODS
POT_QS = [0.85, 0.90, 0.95, 0.975]
BOOT_N = [250, 500, 1000, 2500]
BT_STARTS = {'sp500': ['2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01'], 'dax': ['2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01'],
             'bet': ['2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01'], 'btc': ['2017-01-01', '2019-01-01', '2021-01-01', '2023-01-01']}
WEIGHTS = [0.1, 0.25, 0.5, 0.75, 0.9]
COP_SEEDS = [g.SEED + 1, g.SEED + 11, g.SEED + 21]          # Gaussian sample; the t sample uses seed + 1 (the lecture: 2027, 2028)


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch10_numbers.json ({b})')


def same(a, b, what):
    if a != b:
        raise SystemExit(f'{what}: {a} does not match ch10_numbers.json ({b})')


def r9(a, d=9):
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
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def days(idx):
    d = idx.values.astype('datetime64[D]').astype(np.int64)
    return {'d0': int(d[0]), 'dd': np.diff(d).tolist()}


# ---------------------------------------------------------------- returns, unconditional methods
def returns_export(N):
    out = {}
    for k in MK:
        r = g.returns(k)
        x = np.array(r9(r.values))
        m = N['methods'][k]
        same(len(x), m['n'], f'{k} n')
        close(g.hs_var(x), m['HS']['var'], 1e-7, f'{k} HS VaR 1% (rounded returns)')
        close(g.hs_es(x), m['HS']['es'], 1e-7, f'{k} HS ES 2.5% (rounded returns)')
        close(g.normal_var(x.mean(), x.std(ddof=1)), m['Normal']['var'], 1e-7, f'{k} Normal VaR')
        cf = g.cf_var_es(x)
        close(cf['var'], m['Cornish-Fisher']['var'], 1e-6, f'{k} Cornish-Fisher VaR')
        close(cf['es'], m['Cornish-Fisher']['es'], 1e-6, f'{k} Cornish-Fisher ES')
        out[k] = {**days(r.index), 'x': x.tolist()}
    d = N['def']
    x = np.array(out['sp500']['x'])
    same(int((x < -d['var1']).sum()), d['n_tail1'], 'days beyond VaR 1%')
    return out


def pot_export(N):
    out = {}
    for k in MK:
        x = g.returns(k).values
        out[k] = {f'{q:g}': g.pot_fit(x, q) for q in POT_QS}
        f = out[k]['0.9']
        for p in ('u', 'xi', 'beta', 'Nu', 'n'):
            close(f[p], N['methods'][k]['pot'][p], 1e-10, f'{k} POT {p}')
        close(g.pot_var(f, g.ALPHA_VAR), N['methods'][k]['EVT']['var'], 1e-10, f'{k} EVT VaR')
    return out


def boot_export(N, B=2000, nb=40):
    """precision() of the chapter for other windows: the same generator, the whole sample first, then the last n days."""
    def summ(y, idx):
        S = np.sort(y[idx], axis=1)
        k1, k25 = int(np.ceil(len(y) * g.ALPHA_VAR - 1e-9)), int(np.ceil(len(y) * g.ALPHA_ES - 1e-9))
        bv, be = -S[:, k1 - 1], -S[:, :k25].mean(axis=1)
        lo, hi = min(bv.min(), be.min()), max(bv.max(), be.max())
        edges = np.linspace(lo, hi, nb + 1)
        return {'n': int(len(y)), 'var': g.hs_var(y), 'es': g.hs_es(y), 'var_lo': float(np.quantile(bv, 0.025)), 'var_hi': float(np.quantile(bv, 0.975)),
                'es_lo': float(np.quantile(be, 0.025)), 'es_hi': float(np.quantile(be, 0.975)), 'var_se': float(bv.std()), 'es_se': float(be.std()),
                'h0': float(lo), 'hw': float((hi - lo) / nb), 'hv': np.histogram(bv, edges)[0].tolist(), 'he': np.histogram(be, edges)[0].tolist()}
    out = {}
    for k in MK:
        x = g.returns(k).values
        out[k] = {'last': {}}
        for j, nl in enumerate(BOOT_N):
            rng = np.random.default_rng(g.SEED)
            ia = rng.integers(0, len(x), (B, len(x)))
            if j == 0:
                out[k]['all'] = summ(x, ia)
            y = x[-nl:]
            out[k]['last'][str(nl)] = summ(y, rng.integers(0, len(y), (B, len(y))))
        print('  bootstrap', k)
    P = N['precision']
    for lab, o in (('all', out['sp500']['all']), ('last', out['sp500']['last']['500'])):
        for p in ('var', 'es', 'var_lo', 'var_hi', 'es_lo', 'es_hi', 'var_se', 'es_se'):
            close(o[p], P[lab][p], 1e-10, f'bootstrap {lab} {p}')
    return out


def cond_export(N):
    """Standardised residuals of the GARCH(1,1)-t fitted to all data: the smallest 6% (FHS quantiles and ES up to 5%)."""
    out = {}
    for k in MK:
        res = g.garch_fit(g.returns(k))
        z = np.sort(res.std_resid.dropna().values)
        c = N['methods'][k]['cond']
        close(-g.hs_var(z, g.ALPHA_VAR), c['q_z'], 1e-10, f'{k} FHS quantile')
        close(g.hs_es(z, g.ALPHA_ES), c['es_z'], 1e-10, f'{k} FHS ES')
        out[k] = {'n': int(len(z)), 'z': r9(z[:int(np.ceil(0.06 * len(z)))])}
    return out


# ---------------------------------------------------------------- rolling forecasts and backtests
def page_frame(k, R, F):
    """The rolling forecasts rebuilt from the exported values with the formulas of the page (HS and Normal on the last
    500 rounded returns; GARCH-t and FHS from the exported volatility and the parameters of each re-estimation)."""
    x = np.array(R[k]['x'])
    i0, n = F['i0'], len(F['sig'])
    W = np.lib.stride_tricks.sliding_window_view(x, g.WINDOW)[i0 - g.WINDOW:i0 - g.WINDOW + n]
    Ws = np.sort(W, axis=1)
    m, s = W.mean(axis=1), W.std(axis=1, ddof=1)
    out = {'r': x[i0:i0 + n], 'var1_HS': -Ws[:, 4], 'var25_HS': -Ws[:, 12], 'es25_HS': -Ws[:, :13].mean(axis=1),
           'var1_Normal': -(m + s * stats.norm.ppf(0.01)), 'var25_Normal': -(m + s * stats.norm.ppf(0.025)),
           'es25_Normal': -m + s * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025}
    mu, nu, q1, q25, es = (np.empty(n) for _ in range(5))
    for j, b in enumerate(F['blocks']):
        e = F['blocks'][j + 1][0] if j + 1 < len(F['blocks']) else n
        mu[b[0]:e], nu[b[0]:e], q1[b[0]:e], q25[b[0]:e], es[b[0]:e] = b[1], b[2], b[3], b[4], b[5]
    sig = np.array(F['sig'])
    sc = np.sqrt((nu - 2) / nu)
    qa = stats.t.ppf(0.025, nu)
    out['var1_GARCH-t'] = -(mu + sig * stats.t.ppf(0.01, nu) * sc)
    out['var25_GARCH-t'] = -(mu + sig * qa * sc)
    out['es25_GARCH-t'] = -mu + sig * sc * stats.t.pdf(qa, nu) * (nu + qa ** 2) / ((nu - 1) * 0.025)
    out['var1_FHS'], out['var25_FHS'], out['es25_FHS'] = -(mu + sig * q1), -(mu + sig * q25), -mu + sig * es
    return out


def bt_export(N, R):
    out, frames = {}, {}
    for k in BT:
        df = g.rolling_forecasts(k)
        frames[k] = df
        r = g.returns(k)
        i0 = int(r.index.get_loc(df.index[0]))
        blocks = []
        for b, z in enumerate(df.attrs['zs']):
            j = int(np.argmax(df['block'].values == b))
            blocks.append([j, float(df['mu_G'].iloc[j]), float(df['nu_G'].iloc[j]), -g.hs_var(z, 0.01), -g.hs_var(z, 0.025), g.hs_es(z, 0.025)])
        F = {'i0': i0, 'sig': r9(df['sig_G'].values), 'blocks': blocks}
        P = page_frame(k, R, F)
        for c in ('var1', 'var25', 'es25'):
            for m in METHODS:
                d = np.max(np.abs(P[f'{c}_{m}'] - df[f'{c}_{m}'].values))
                if d > 1e-6:
                    raise SystemExit(f'{k} {c} {m}: rebuilt forecasts differ by {d}')
        for m in METHODS:
            hits = (P['r'] < -P[f'var1_{m}']).astype(int)
            B = N['bt'][k][m]
            same(int(hits.sum()), B['x'], f'{k} {m} exceedances')
            c = g.christoffersen(hits)
            for p in ('n00', 'n01', 'n10', 'n11'):
                same(c[p], B[p], f'{k} {m} {p}')
            same(int(pd.Series(hits).rolling(250).sum().max()), B['max250'], f'{k} {m} max250')
            same(int(hits[-250:].sum()), B['last250'], f'{k} {m} last250')
            close(g.quantile_loss(P['r'], P[f'var1_{m}']), B['qloss'], 1e-6, f'{k} {m} quantile loss')
            close(g.z2_stat(P['r'], P[f'var25_{m}'], P[f'es25_{m}']), B['z2'], 1e-6, f'{k} {m} Z2')
            for e, (a, b) in g.EPISODES.items():
                sel = (df.index >= a) & (df.index <= b)
                same(int(hits[sel].sum()), B['stress'][e]['x'], f'{k} {m} stress {e}')
        # p-values of Z2 (the chapter's z2_pvalue) for later evaluation starts; the first start is the lecture's
        pz = {}
        for s in BT_STARTS[k]:
            j0 = int(df.index.searchsorted(pd.Timestamp(s)))
            sub = df.iloc[j0:].copy()
            sub.attrs = {'W': df.attrs['W'][j0:], 'zs': df.attrs['zs']}
            pz[s[:4]] = {m: g.z2_pvalue(sub, m)[1] for m in METHODS}
        for m in METHODS:
            close(pz[BT_STARTS[k][0][:4]][m], N['bt'][k][m]['p_z2'], 1e-12, f'{k} {m} p of Z2')
        F['pz2'] = pz
        out[k] = F
        print('  backtest', k)
    return out


# ---------------------------------------------------------------- portfolio
def port_export(N):
    R = g.portfolio_data()
    P = N['port']
    same(len(R), P['n'], 'portfolio days')
    out = {**days(R.index), 'bet': r9(R['bet'].values), 'sp500': r9(R['sp500'].values), 'w': {}}
    for w in WEIGHTS:
        a = g.portfolio_analysis(R, (w, 1 - w))
        out['w'][f'{w:g}'] = {p: a[p] for p in ('var_n', 'es_n', 'var_gauss', 'es_gauss', 'joint1_gauss', 'var_t', 'es_t', 'joint1_t', 'var_hs', 'es_hs', 'joint1_emp',
                                                 'var_n_sum', 'var_hs_sum')}
        if w == 0.5:
            for p in out['w']['0.5']:
                close(a[p], P[p], 1e-10, f'portfolio {p}')
        print('  portfolio w =', w)
    cf = P['copula']
    U = g.pseudo_obs(R)
    grid = np.arange(2.0, 40.01, 0.25)
    out['ll_grid'] = {'nu': grid.tolist(), 'll': [g.t_copula_loglik(U, cf['rho'], v) for v in grid]}
    close(max(out['ll_grid']['ll']), cf['ll_t'], 1e-10, 't copula log-likelihood')
    corner = lambda X: [r9(X[(X[:, 0] <= 0.25) & (X[:, 1] <= 0.25), i], 5) for i in (0, 1)]
    out['corner'] = {'data': corner(U)}
    n = len(U)
    for s in COP_SEEDS:
        G = g.simulate_copula(cf['rho'], None, n, s)
        T = g.simulate_copula(cf['rho'], cf['nu'], n, s + 1)
        out['corner'][str(s)] = {'gauss': corner(G), 't': corner(T)}
        if s == g.SEED + 1:
            q = 0.05
            same(int(np.sum((G[:, 0] <= q) & (G[:, 1] <= q))), N['copula_fig']['corner_gauss'], 'Gaussian copula corner')
            same(int(np.sum((T[:, 0] <= q) & (T[:, 1] <= q))), N['copula_fig']['corner_t'], 't copula corner')
    return out


def main():
    N = json.load(open(os.path.join(HERE, 'ch10_numbers.json')))
    same(g.returns('sp500').index[-1].date().isoformat(), N['end'], 'last day')
    print('returns')
    R = returns_export(N)
    pot = pot_export(N)
    print('bootstrap')
    boot = boot_export(N)
    cond = cond_export(N)
    print('rolling forecasts')
    bt = bt_export(N, R)
    print('portfolio')
    port = port_export(N)
    d = dict(end=N['end'], names=MK, bt_names=BT, name=g.NAME, col=g.COLORS, mcol=g.MCOL, methods=METHODS, episodes=g.EPISODES, window=g.WINDOW,
             refit=g.REFIT, plus=g.PLUS, pot_qs=POT_QS, boot_n=BOOT_N, bt_starts=BT_STARTS, weights=WEIGHTS, cop_seeds=COP_SEEDS)
    d['ret'] = R
    d['pot'] = pot
    d['boot'] = boot
    d['cond'] = cond
    d['bt'] = bt
    d['port'] = port
    d['N'] = {k: N[k] for k in ('def', 'methods', 'curves', 'horizon', 'precision', 'traffic', 'bt', 'port', 'corr', 'copula_fig')}
    js = ('// Inputs of the Chapter 10 interactive page, written by Quantlets/Ch_10/export_interactive.py.\n'
          'window.CH10_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
