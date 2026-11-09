"""Export the inputs of the Chapter 11 interactive page (SFM/interactive/ch11_data.js).

Uses the functions of generate_fmh_charts.py (same data files, same samples, same seeds), so the page starts from the numbers
of the slides. Daily log returns (in %) of the six series are exported with 9 significant digits; from them the browser
recomputes the scaling of h-day returns, R/S, DFA, the GPH regression, Lo's modified R/S, the ACF of r, |r| and r^2, the
six-market table and the rolling DFA exponents. The Nile flow is exported as it is (statsmodels data set). The simulated
series of the chapter cannot be redrawn in the browser (numpy generator), so they are exported: the fractional Gaussian noise
of the fBm paths (seed 2026), the sample ACF of the fGn and ARFIMA simulations, the ARFIMA paths, every Monte Carlo estimate
(R/S, DFA, GPH under i.i.d. Normal series; Lo's V and V(q) under AR(1) and fGn; R/S, DFA and GPH under a mean shift and under
GARCH), the i.i.d. Normal mean curves of the log-log plots and the Nile null distribution. The browser recomputes means,
rejection rates and bands from them.
Extra runs of the same functions: fBm for H = 0.1, ..., 0.9; fGn ACF for H = 0.6, 0.7, 0.9 (and 1 - H); ARFIMA for other d;
Lo's test with N = 250, 500, 2000; the spurious-memory simulations with N = 500, 1000, 4000; the Monte Carlo band for N = 2000;
the i.i.d. mean curves for the other five series; R/S and GPH of the shuffled |r|.
The script checks every slide number against ch11_numbers.json (chapter functions on the saved data: to 1e-12; the formulas
of the page on the exported, rounded values: to 1e-6) and stops if one no longer matches. It writes only
interactive/ch11_data.js.
Run:  python3 Quantlets/Ch_11/export_interactive.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

import generate_fmh_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch11_data.js')
MK = g.ASSETS
FBM_H = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FGN_H = [0.6, 0.7, 0.8, 0.9]
ARF_D = [0.1, 0.2, 0.3, 0.4, 0.45]
LO_N = [250, 500, 1000, 2000]
SP_N = [500, 1000, 2000, 4000]
ROLL_W = [250, 500, 1000, 2000]


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch11_numbers.json ({b})')


def same(a, b, what):
    if a != b:
        raise SystemExit(f'{what}: {a} does not match ch11_numbers.json ({b})')


def rd(a, d=9):
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


def summ(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'sd': float(v.std()), 'q025': float(np.quantile(v, 0.025)), 'q975': float(np.quantile(v, 0.975))}


def check_summ(a, b, tol, what):
    for p in ('mean', 'sd', 'q025', 'q975'):
        close(a[p], b[p], tol, f'{what} {p}')


# ---------------------------------------------------------------- real data
def returns_export(N):
    out, sizes = {}, {}
    for k in MK:
        r = g.returns(k)
        x = r.values
        M = N['markets'][k]
        same(r.index[0].date().isoformat(), M['first'], f'{k} first day')
        same(r.index[-1].date().isoformat(), M['last'], f'{k} last day')
        for lab, y in (('r', x), ('abs', np.abs(x))):
            e = g.all_estimates(y)
            for p in ('n', 'gph_m', 'lo_q', 'lo_reject', 'lo_reject0'):
                same(e[p], M[lab][p], f'{k} {lab} {p}')
            for p in ('rs', 'dfa', 'gph_d', 'gph_se', 'lo_V', 'lo_V0'):
                close(e[p], M[lab][p], 1e-12, f'{k} {lab} {p}')
        xr = np.array(rd(x))
        for lab, y in (('r', xr), ('abs', np.abs(xr))):          # the page's inputs: rounded returns
            e = g.all_estimates(y)
            for p in ('rs', 'dfa', 'gph_d', 'lo_V', 'lo_V0'):
                close(e[p], M[lab][p], 1e-6, f'{k} {lab} {p} (rounded returns)')
            same(e['lo_q'], M[lab]['lo_q'], f'{k} {lab} lo_q (rounded returns)')
        sizes[str(len(x))] = g.block_sizes(len(x)).tolist()
        out[k] = {**days(r.index), 'x': xr.tolist()}
    S = N['scaling']
    for k in ('sp500', 'bet', 'btc'):
        hs, sd, H = g.scaling(np.array(out[k]['x']))
        close(H, S[k]['H'], 1e-6, f'{k} scaling H')
        close(sd[3] / sd[0], S[k]['ratio10'], 1e-6, f'{k} ratio10')
        close(sd[-1] / sd[0], S[k]['ratio250'], 1e-6, f'{k} ratio250')
    x = np.array(out['sp500']['x'])
    G = N['gph']
    for lab, y in (('r', x), ('abs', np.abs(x))):
        close(g.gph(y)['d'], G[lab]['d'], 1e-6, f'GPH {lab}')
    V = N['vol_acf']
    for lab, y in (('returns', x), ('absolute', np.abs(x)), ('squared', x ** 2)):
        a = g.acf(y, 250)
        for L in (1, 10, 50, 100, 250):
            close(a[L - 1], V[lab][str(L)], 1e-6, f'ACF {lab} {L}')
        band = 1.96 / np.sqrt(len(x))
        same(int(np.sum(a > band)), V[lab]['npos'], f'ACF {lab} npos')
        same(int(np.sum(np.abs(a[1:]) > band)), V[lab]['nout2'], f'ACF {lab} nout2')
        same([int(v) for v in sorted(np.argsort(-np.abs(a[1:]))[:3] + 2)], V[lab]['top_lags'], f'ACF {lab} top lags')
    return out, sizes


def iid_curves(N, R, sizes, reps=g.MC_REPS):
    """fig_rs_dfa: the i.i.d. Normal mean curves (seed + 5, 200 series of the same length); the lecture shows the S&P 500."""
    out = {}
    for k in MK:
        n = len(R[k]['x'])
        sz = np.array(sizes[str(n)])
        rng = np.random.default_rng(g.SEED + 5)
        sims = rng.standard_normal((min(reps, 200), n))
        rs0 = np.mean([g.rs_curve(v, sz)[1] for v in sims], axis=0)
        F0 = np.mean([g.dfa_curve(v, sz)[1] for v in sims], axis=0)
        out[k] = {'rs0': rs0.tolist(), 'F0': F0.tolist()}
        if k == 'sp500':
            D = N['rs_dfa']
            close(float(np.polyfit(np.log10(sz), np.log10(rs0), 1)[0]), D['rs_iid'], 1e-12, 'R/S i.i.d. slope')
            close(float(np.polyfit(np.log10(sz), np.log10(F0), 1)[0]), D['dfa_iid'], 1e-12, 'DFA i.i.d. slope')
            close(float(rs0[0]), D['rs_n10'], 1e-12, 'R/S i.i.d. at n = 10')
            same([int(sz[0]), int(sz[-1]), len(sz)], [D['nmin'], D['nmax'], D['nsizes']], 'block sizes')
            x = np.array(R[k]['x'])
            close(g.hurst_rs(x, sz), D['rs_r'], 1e-6, 'R/S slope r')
            close(g.hurst_rs(np.abs(x), sz), D['rs_abs'], 1e-6, 'R/S slope |r|')
            n_, F = g.dfa_curve(np.abs(x), sz)
            close(float(np.polyfit(np.log10(n_), np.log10(F / F[0]), 1)[0]), D['dfa_abs'], 1e-6, 'DFA slope |r|')
        print('  i.i.d. curves', k)
    return out


def shuffled(N):
    """markets(): DFA of |r| after a random permutation (seed + 31, first draw); R/S and GPH of the same permutation."""
    out = {}
    for k in MK:
        a = np.abs(g.returns(k).values)
        p = np.random.default_rng(g.SEED + 31).permutation(a)
        d = g.hurst_dfa(p)
        close(d, N['markets'][k]['abs_shuffled_dfa'], 1e-12, f'{k} shuffled DFA')
        out[k] = {'dfa': d, 'rs': g.hurst_rs(p), 'gph': g.gph(p)['H']}
    return out


def rolling_check(N, R, sizes):
    """fig_rolling on the rounded returns with the formulas of the page (block sizes of the window, OLS on log10)."""
    band = N['rolling']['band']
    for k in ('sp500', 'bet', 'btc'):
        r = g.returns(k)
        x = np.array(R[k]['x'])
        for lab, y in (('r', x), ('abs', np.abs(x))):
            idx, val = [], []
            sz = np.array(sizes['1000'])
            for e in range(g.WINDOW, len(y) + 1, g.STEP):
                idx.append(r.index[e - 1])
                val.append(g.hurst_dfa(y[e - g.WINDOW:e], sz))
            h = np.array(val)
            P = N['rolling'][k]
            if lab == 'r':
                same(len(h), P['n_windows'], f'{k} windows')
                same(idx[0].date().isoformat(), P['first_end'], f'{k} first window')
                for p, v in (('min', h.min()), ('max', h.max()), ('last', h[-1]), ('first_third', h[:len(h) // 3].mean()), ('last_third', h[-len(h) // 3:].mean())):
                    close(v, P[p], 1e-6, f'{k} rolling {p}')
                same(idx[int(np.argmax(h))].date().isoformat(), P['date_max'], f'{k} date of max')
                same(idx[int(np.argmin(h))].date().isoformat(), P['date_min'], f'{k} date of min')
                close(np.mean(h > band['q975']), P['above'], 1e-12, f'{k} above')
                close(np.mean(h < band['q025']), P['below'], 1e-12, f'{k} below')
            else:
                for p, v in (('abs_min', h.min()), ('abs_max', h.max()), ('abs_last', h[-1])):
                    close(v, P[p], 1e-6, f'{k} rolling {p}')
        print('  rolling', k)


# ---------------------------------------------------------------- simulations of the chapter
def fbm_export(N):
    out = {}
    for H in FBM_H:
        x = g.sim_fgn(1000, H, np.random.default_rng(g.SEED))[0]
        out[f'{H:g}'] = rd(x, 8)
        if f'{H:g}' in N['fbm']:
            y = np.array(out[f'{H:g}'])
            b = np.concatenate([[0], np.cumsum(y)])
            close(float(g.acf(y, 1)[0]), N['fbm'][f'{H:g}']['acf1'], 1e-6, f'fBm H = {H} acf1')
            close(float(b.max() - b.min()), N['fbm'][f'{H:g}']['range'], 1e-6, f'fBm H = {H} range')
    return out


def fgn_export(N):
    """fig_fgn_acf: sample ACF (50 lags) of n = 5000 fGn values, seed SEED + int(10 H), for H and 1 - H."""
    out = {}
    for H in FGN_H:
        for h in (H, round(1 - H, 10)):
            x = g.sim_fgn(5000, h, np.random.default_rng(g.SEED + int(10 * h)))[0]
            out[f'{h:g}'] = rd(g.acf(x, 50), 8)
    for h in ('0.8', '0.2'):
        close(g.fgn_acvf(2, float(h))[1], N['fgn'][h]['rho1'], 1e-12, f'fGn rho1 {h}')
    return out


def arfima_export(N):
    out = {}
    for d in ARF_D:
        for row, dd in enumerate((d, -d)):
            x = g.sim_arfima(1000, dd, np.random.default_rng(g.SEED + 7 + row))[0]
            xl = g.sim_arfima(20000, dd, np.random.default_rng(g.SEED + 17 + row))[0]
            out[f'{dd:g}'] = {'path': rd(x, 7), 'acf': rd(g.acf(xl, 50), 8)}
            if f'{dd:g}' in N['arfima']:
                close(float(np.std(np.array(out[f'{dd:g}']['path']))), N['arfima'][f'{dd:g}']['sd'], 1e-6, f'ARFIMA d = {dd} sd')
                th = g.arfima_acf(51, dd)
                for L in (1, 2, 10, 50):
                    close(th[L], N['arfima'][f'{dd:g}'][f'rho{L}'], 1e-12, f'ARFIMA d = {dd} rho{L}')
    return out


def nile_export(N):
    y = g.nile()
    n = len(y)
    sz = g.block_sizes(n, nmin=6, nmax=n // 2, num=10)
    rng = np.random.default_rng(g.SEED + 3)
    null = np.array([g.hurst_rs(rng.standard_normal(n), sz) for _ in range(g.MC_REPS)])
    Ni = N['nile']
    close(float(np.quantile(null, 0.025)), Ni['q025'], 1e-12, 'Nile null q025')
    close(float(np.quantile(null, 0.975)), Ni['q975'], 1e-12, 'Nile null q975')
    nr = np.array(rd(null, 8))
    close(float(np.quantile(nr, 0.025)), Ni['q025'], 1e-6, 'Nile null q025 (rounded)')
    close(float(nr.mean()), Ni['null_mean'], 1e-6, 'Nile null mean (rounded)')
    v = y.values.astype(float)
    close(g.hurst_rs(v, sz), Ni['H_raw'], 1e-12, 'Nile H raw')
    close(g.hurst_dfa(v, sz), Ni['dfa_raw'], 1e-12, 'Nile DFA raw')
    pre, post = y.loc[:g.NILE_BREAK], y.loc[g.NILE_BREAK + 1:]
    adj = np.concatenate([pre - pre.mean(), post - post.mean()])
    close(g.hurst_rs(adj, sz), Ni['H_adj'], 1e-12, 'Nile H adjusted')
    close(g.hurst_dfa(adj, sz), Ni['dfa_adj'], 1e-12, 'Nile DFA adjusted')
    return {'year': y.index.tolist(), 'flow': v.tolist(), 'sizes': sz.tolist(), 'null': nr.tolist(), 'brk': g.NILE_BREAK}


def mc_export(N):
    """mc_null of the chapter, every replication kept (R/S, DFA, GPH H); N = 2000 is an extra run (rolling window of 2000)."""
    out = {}
    for n in g.MC_N + [2000]:
        rng = np.random.default_rng(g.SEED + n)
        sz = g.block_sizes(n)
        rs, dfa, gp = [], [], []
        for _ in range(g.MC_REPS):
            x = rng.standard_normal(n)
            rs.append(g.hurst_rs(x, sz))
            dfa.append(g.hurst_dfa(x, sz))
            gp.append(g.gph(x)['H'])
        o = {'rs': rd(rs, 8), 'dfa': rd(dfa, 8), 'gph': rd(gp, 8)}
        if str(n) in N['mc']:
            for e, v in (('rs', rs), ('dfa', dfa), ('gph', gp)):
                check_summ(summ(v), N['mc'][str(n)][e], 1e-12, f'MC N = {n} {e}')
                check_summ(summ(o[e]), N['mc'][str(n)][e], 1e-6, f'MC N = {n} {e} (rounded)')
        out[str(n)] = o
        print('  Monte Carlo N =', n)
    check_summ(summ(out['1000']['dfa']), N['rolling']['band'], 1e-6, 'rolling band')
    return out


def lo_export(N):
    """lo_rejections of the chapter for several N, with V(q), V(0) and q of every replication."""
    out = {}
    for n in LO_N:
        rng = np.random.default_rng(g.SEED + 11)
        reps = g.MC_REPS
        o = {'ar': {}, 'fgn': {}}
        for phi in g.PHIS:
            e = rng.standard_normal((reps, n + 200))
            x = np.empty_like(e)
            x[:, 0] = e[:, 0]
            for t in range(1, n + 200):
                x[:, t] = phi * x[:, t - 1] + e[:, t]
            x = x[:, 200:]
            t_ = [g.lo_test(v) for v in x]
            o['ar'][f'{phi:g}'] = {'V': rd([a['V'] for a in t_], 6), 'V0': rd([a['V0'] for a in t_], 6), 'q': [a['q'] for a in t_]}
        for H in g.HS_POWER:
            X = g.sim_fgn(n, H, rng, size=reps)
            t_ = [g.lo_test(v) for v in X]
            o['fgn'][f'{H:g}'] = {'V': rd([a['V'] for a in t_], 6), 'V0': rd([a['V0'] for a in t_], 6), 'q': [a['q'] for a in t_]}
        if n == N['lo']['N']:
            lo, hi = g.LO_CRIT
            rej = lambda v: float(np.mean([not lo <= a <= hi for a in v]))
            for grp, keys in (('ar', g.PHIS), ('fgn', g.HS_POWER)):
                for kk in keys:
                    s = o[grp][f'{kk:g}']
                    ref = N['lo'][grp][str(float(kk))]
                    close(rej(s['V0']), ref['classical'], 1e-12, f'Lo {grp} {kk} classical')
                    close(rej(s['V']), ref['modified'], 1e-12, f'Lo {grp} {kk} modified')
                    close(float(np.mean(s['q'])), ref['q_mean'], 1e-12, f'Lo {grp} {kk} q')
        out[str(n)] = o
        print('  Lo N =', n)
    return out


def spurious_export(N, reps=300):
    """spurious() of the chapter for several N, every replication kept."""
    out = {}
    for n in SP_N:
        rng = np.random.default_rng(g.SEED + 21)
        sz = g.block_sizes(n)
        br, ga = {}, {}
        for dl in g.DELTAS:
            X = rng.standard_normal((reps, n))
            X[:, n // 2:] += dl
            br[f'{dl:g}'] = {'rs': rd([g.hurst_rs(v, sz) for v in X], 6), 'dfa': rd([g.hurst_dfa(v, sz) for v in X], 6), 'gph': rd([g.gph(v)['H'] for v in X], 6)}
        for p in g.PERS:
            E = g.sim_garch(n, 1 - p, 0.10, p - 0.10, rng, size=reps)
            A = np.abs(E)
            ga[f'{p:g}'] = {'abs_dfa': rd([g.hurst_dfa(v, sz) for v in A], 6), 'abs_gph': rd([g.gph(v)['H'] for v in A], 6), 'ret_dfa': rd([g.hurst_dfa(v, sz) for v in E], 6)}
        if n == N['spurious']['N']:
            for dl in g.DELTAS:
                for e in ('rs', 'dfa', 'gph'):
                    close(float(np.mean(br[f'{dl:g}'][e])), N['spurious']['break'][str(float(dl))][e], 1e-6, f'spurious break {dl} {e}')
            for p in g.PERS:
                for e in ('abs_dfa', 'abs_gph', 'ret_dfa'):
                    close(float(np.mean(ga[f'{p:g}'][e])), N['spurious']['garch'][str(p)][e], 1e-6, f'spurious GARCH {p} {e}')
        out[str(n)] = {'break': br, 'garch': ga}
        print('  spurious N =', n)
    return out


def rs_example_check(N):
    s = g.rs_steps(g.RS_EXAMPLE)
    for p in ('mean', 'R', 'S', 'RS', 'sumsq'):
        close(s[p], N['rs_example'][p], 1e-12, f'R/S example {p}')


def main():
    N = json.load(open(os.path.join(HERE, 'ch11_numbers.json')))
    same(g.returns('sp500').index[-1].date().isoformat(), N['end'], 'last day')
    rs_example_check(N)
    print('returns')
    R, sizes = returns_export(N)
    for n in ROLL_W:
        sizes[str(n)] = g.block_sizes(n).tolist()
    for n in SP_N:
        sizes[str(n)] = g.block_sizes(n).tolist()
    print('i.i.d. curves')
    iid = iid_curves(N, R, sizes)
    shuf = shuffled(N)
    print('rolling')
    rolling_check(N, R, sizes)
    print('simulations')
    fbm = fbm_export(N)
    fgn = fgn_export(N)
    arf = arfima_export(N)
    nile = nile_export(N)
    mc = mc_export(N)
    lo = lo_export(N)
    sp = spurious_export(N)
    band2000 = summ(mc['2000']['dfa'])
    d = dict(end=N['end'], names=MK, name=g.NAME, col=g.COLORS, seed=g.SEED, window=g.WINDOW, step=g.STEP, lo_crit=g.LO_CRIT, gph_power=g.GPH_POWER,
             h_list=g.H_LIST, mc_n=g.MC_N, phis=g.PHIS, hs_power=g.HS_POWER, deltas=g.DELTAS, pers=g.PERS, rs_example=g.RS_EXAMPLE, lo_n=LO_N, sp_n=SP_N,
             roll_w=ROLL_W, fbm_h=FBM_H, fgn_h=FGN_H, arf_d=ARF_D)
    d.update(ret=R, sizes=sizes, iid=iid, shuffled=shuf, fbm=fbm, fgn=fgn, arfima=arf, nile=nile, mc=mc, lo=lo, spurious=sp, band2000=band2000, N=N)
    js = ('// Inputs of the Chapter 11 interactive page, written by Quantlets/Ch_11/export_interactive.py.\n'
          'window.CH11_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
