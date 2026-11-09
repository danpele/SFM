"""Export the inputs of the Chapter 8 interactive page (SFM/interactive/ch8_data.js).

Uses the functions of generate_all_charts.py (same data files, same samples, same seeds), so the page starts from the numbers of
the slides. Daily log returns (in %) are exported with 6 significant digits, the open, high, low and close prices with 7, the VIX
closes on the S&P 500 calendar with 4; the browser computes from them the annualised volatilities, the rolling and EWMA
volatilities, the range-based estimators, the autocorrelations, the Ljung-Box and ARCH-LM tests, the annual and monthly
volatilities, the VIX comparison, the leverage correlations, the local linear volatility curves and the forecast losses.
What the browser does not repeat comes from the chapter's code: the simulated trading day (seed 3), the efficiency simulation
(1,000 windows, seed 2026), the simulated one-minute prices of the realised-variance slides and the shuffle of the S&P 500 returns.
Extra runs of the same functions: the simulated day for seeds other than 3, the efficiency simulation for windows of 5, 10 and 63
days, the signature plot for other noise levels and the shuffle of blocks of 5 to 252 days (the same generator and seed).
The script stops if a recomputed number no longer matches ch8_numbers.json (a revised data file, a changed rule or seed).
Run:  python3 Quantlets/Ch_08/export_interactive.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch8_data.js')
RET = g.ASSETS + g.STOCKS                  # sp500, dax, bet, btc, tlv, snp
OHLC = g.OHLC_ASSETS                       # sp500, dax, btc, tlv, snp
EFF_N = [5, 10, 21, 63]
SEEDS = list(range(1, 11))                 # simulated trading day; 3 is the lecture's
NOISE = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1]
BLOCKS = [1, 5, 21, 63, 252]               # shuffle of blocks of b days; b = 1 is the lecture's shuffle


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b))):
        raise SystemExit(f'{what}: {a} does not match ch8_numbers.json ({b})')


def r6(a, d=6):
    return [None if not np.isfinite(x) else float(f'{x:.{d}g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (pd.Timestamp,)):
        return x.date().isoformat()
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


def check(N):
    """Key numbers of the slides, recomputed with the chapter's functions."""
    for k in g.ASSETS:
        r = g.returns(k)
        assert len(r) == N['bursts'][k]['n'], k
        close(float(r.std() * np.sqrt(g.ann(r))), N['bursts'][k]['vol'], 1e-10, f'{k} annual volatility')
    est = g.estimators_table()
    for k in OHLC:
        for e in ('cc', 'park', 'gk', 'rs', 'yz', 'night_share', 'corr_oc'):
            close(est[k][e], N['est'][k][e], 1e-10, f'{k} {e}')
    cl = g.clustering_table()
    for k in RET:
        close(cl[k]['lb_r2']['q'], N['clust'][k]['lb_r2']['q'], 1e-10, f'{k} Q(10) r^2')
        close(cl[k]['lb_r']['q'], N['clust'][k]['lb_r']['q'], 1e-10, f'{k} Q(10) r')
        close(cl[k]['arch']['lm'], N['clust'][k]['arch']['lm'], 1e-10, f'{k} ARCH-LM')
    m, ar = g.monthly_rv('sp500')
    close(ar['phi'], N['long']['ar']['phi'], 1e-10, 'AR(1) of log monthly RV')
    fc = g.forecast_table()
    for mth in fc['r2']:
        close(fc['r2'][mth]['qlike'], N['fc']['r2'][mth]['qlike'], 1e-10, f'QLIKE {mth}')
    close(fc['dm_ewma_hist21']['t'], N['fc']['dm_ewma_hist21']['t'], 1e-8, 'DM EWMA - hist21')
    for k in ('sp500', 'bet', 'btc'):
        q = g.lambda_qlike(k)
        close(float(q.idxmin()), N['lambda'][k]['best'], 1e-12, f'{k} best lambda')
    for k in ('sp500', 'dax'):
        s = g.nonparametric_vol(k)
        close(float(np.interp(-2, s.index, s.values)), N['npvol'][k]['m2'], 1e-10, f'{k} sigma(-2)')
    L = g.leverage('sp500')
    close(L['cc'][1], N['lev']['sp500']['cc']['1'], 1e-10, 'leverage corr j = 1')


def ohlc_paths(N, steps=390):
    """fig_ohlc_path of the chapter for several seeds: Brownian log price, 390 one-minute steps, overnight gap 0.35%."""
    out = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x = np.concatenate([[0], np.cumsum(rng.standard_normal(steps) / np.sqrt(steps))])
        path = 0.35 + x
        if seed == 3:
            ref = g.fig_ohlc_path(save=False)
            matplotlib.pyplot.close('all')
            close(float(path.max() - 0.35), ref['u'], 1e-12, 'OHLC path u')
            close(float(path[-1] - 0.35), ref['c'], 1e-12, 'OHLC path c')
            close(float(path.min() - 0.35), N['ohlc_path']['d'], 1e-12, 'OHLC path d')
        out[str(seed)] = r6(path, 6)
    return out


def efficiency(N):
    """efficiency_sim of the chapter: n = 21 is the lecture's, other n are extra runs (same seed)."""
    res = {}
    for n in EFF_N:
        summ, draws = g.efficiency_sim(n=n)
        if n == 21:
            for sc in summ:
                for e in summ[sc]:
                    for c in ('bias', 'eff', 'mse_ratio'):
                        close(summ[sc][e][c], N['eff'][sc][e][c], 1e-9, f'efficiency {sc} {e} {c}')
        res[str(n)] = {sc: {e: r6(v, 6) for e, v in d.items()} for sc, d in draws.items()}
        print('  efficiency n =', n)
    return res


def intraday(N):
    """fig_realised (60 simulated days) and fig_signature (500 days, several noise levels)."""
    sig, obs = g.simulate_intraday(60)
    rv5 = g.realised_var(obs, 5)
    close(float(np.mean(np.abs(np.sqrt(rv5) - sig) / sig)), N['rv']['mae_rv5'], 1e-10, 'RV 5-minute error')
    every = (1, 2, 3, 5, 10, 15, 30, 65, 130, 390)
    sig5, clean_ = g.simulate_intraday(500, noise=0.0)
    sigd = {}
    for nz in NOISE:
        _, noisy = g.simulate_intraday(500, noise=nz)
        sigd[str(nz)] = [float(np.mean(g.realised_var(noisy, m) / sig5 ** 2)) for m in every]
    for i, m in enumerate(every):
        close(sigd['0.025'][i], N['signature']['noisy'][str(m)], 1e-10, f'signature {m}')
        close(sigd['0.0'][i], N['signature']['clean'][str(m)], 1e-10, f'signature clean {m}')
    return {'sig': r6(sig, 8), 'obs': [r6(row, 7) for row in obs], 'every': list(every), 'sig500': r6(sig5, 8), 'curves': sigd}


def shuffles(N):
    """fig_shuffle: rng(2026).permutation of the S&P 500 returns; blocks of b days: the same generator on the block order."""
    r = g.returns('sp500').values
    out = {}
    for b in BLOCKS:
        rng = np.random.default_rng(g.SEED)
        if b == 1:
            perm = rng.permutation(len(r))
            s = rng_check = np.random.default_rng(g.SEED).permutation(r)
            assert np.array_equal(r[perm], rng_check), 'shuffle order'
            close(float(g.acf(np.abs(s), 1)[0]), N['shuffle']['acf1_abs_shuffled'], 1e-12, 'shuffled ACF(1)')
            out['1'] = perm.tolist()
        else:
            out[str(b)] = rng.permutation(int(np.ceil(len(r) / b))).tolist()
    return out


def main():
    N = json.load(open(os.path.join(HERE, 'ch8_numbers.json')))
    print('check')
    check(N)
    d = dict(end=N['end'], ret_names=RET, ohlc_names=OHLC, name=g.NAME, col=g.COLORS, est_col=g.EST_COLORS, est_label=g.EST_LABEL,
             start=g.START, ohlc_start=g.OHLC_START, seed=g.SEED, eff_n=EFF_N, seeds=SEEDS, noise=NOISE, blocks=BLOCKS)
    print('returns, OHLC, VIX')
    d['ret'] = {k: {**days(g.returns(k).index), 'x': r6(g.returns(k).values)} for k in RET}
    d['ohlc'] = {}
    for k in OHLC:
        t = g.ohlc(k)
        d['ohlc'][k] = {**days(t.index), **{c[0]: r6(t[c].values, 7) for c in ('open', 'high', 'low', 'close')}}
    r = g.returns('sp500')
    v = g.load_close('vix', g.START['sp500']).reindex(r.index).ffill()
    assert v.notna().all()
    d['vix'] = r6(v.values, 4)
    print('simulations')
    d['path'] = ohlc_paths(N)
    d['eff'] = efficiency(N)
    d['intra'] = intraday(N)
    d['perm'] = shuffles(N)
    d['N'] = {k: N[k] for k in ('bursts', 'windows', 'se', 'ewma', 'ghost', 'ohlc_path', 'eff', 'est', 'range_rolling', 'rv', 'signature', 'clust',
                                'acf', 'shuffle', 'kurt', 'long', 'vix', 'lev', 'npvol', 'fc', 'lambda')}
    js = ('// Inputs of the Chapter 8 interactive page, written by Quantlets/Ch_08/export_interactive.py.\n'
          'window.CH8_DATA = ' + json.dumps(clean(d), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
