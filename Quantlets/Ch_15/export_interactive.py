"""Export the inputs of the Chapter 15 interactive page (SFM/interactive/ch15_data.js).

Calls the functions of generate_all_charts.py (same data files in data/market, same start date 2005-01-01 and end date
2026-09-18), so the page starts from the numbers of the slides. Exported: the daily adjusted closes of the 13 banks (after the
chapter's filters: weekdays, no unchanged closes, BVB days with trades) and the closes of the S&P 500, Euro Stoxx 50 and BET
(the browser joins them on common days and recomputes the returns, the bank portfolios, the episodes, the rolling
correlations, the Forbes-Rigobon test, the empirical tail dependence, the quantiles, CoVaR and Delta-CoVaR from the exported
quantile-regression coefficients, MES, LRMES, SRISK, the Granger networks and the Diebold-Yilmaz table); the quantile-
regression coefficients of every bank at the levels 1%, 2.5%, 5% and 10% and at the median; the block-bootstrap intervals of
Delta-CoVaR 1% and MES 5%; the t-copula fits; the time-varying Delta-CoVaR of JPMorgan Chase, Deutsche Bank and Banca
Transilvania at 1%, 2.5% and 5%; the rolling total connectedness for windows of 100, 150, 200, 250 and 300 days.
Extra runs of the chapter's code (labelled on the page): the levels 2.5% and 10% (1% and 5% for the time-varying Delta-CoVaR
are extended by 2.5%), the rolling windows other than 200 days.
The script checks every number of ch15_numbers.json against the chapter functions (to 1e-12 relative, exact for counts,
dates and labels) and stops if one no longer matches. It writes only interactive/ch15_data.js (no chart, no table of the
chapter: every chart function is called with save=False).
Run:  python3 Quantlets/Ch_15/export_interactive.py      (a few minutes: the bootstrap intervals are recomputed)
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import generate_all_charts as g
import sfm_style as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch15_data.js')
LEVELS = (0.01, 0.025, 0.05, 0.1)
DYN_LEVELS = (0.01, 0.025, 0.05)
DY_WINDOWS = (100, 150, 200, 250, 300)


def tree(a, b, what, tol=1e-12):
    """a (recomputed) against b (ch15_numbers.json): floats to tol (relative), the rest exactly; stops on a mismatch."""
    if isinstance(b, dict):
        A = {str(k): v for k, v in a.items()}
        if set(A) != set(b):
            raise SystemExit(f'{what}: keys {sorted(A)} do not match ch15_numbers.json {sorted(b)}')
        for k in b:
            tree(A[k], b[k], f'{what}/{k}', tol)
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{what}: length {len(a)} does not match ch15_numbers.json ({len(b)})')
        for i, (x, y) in enumerate(zip(a, b)):
            tree(x, y, f'{what}[{i}]', tol)
    elif b is None or isinstance(b, (bool, str)) or (isinstance(b, int) and not isinstance(a, (float, np.floating))):
        if a != b and not (isinstance(a, (int, np.integer)) and int(a) == b):
            raise SystemExit(f'{what}: {a} does not match ch15_numbers.json ({b})')
    else:
        a, b = float(a), float(b)
        if math.isnan(b) and math.isnan(a):
            return
        if not abs(a - b) <= tol * max(1.0, abs(b)):
            raise SystemExit(f'{what}: {a} does not match ch15_numbers.json ({b})')


def rd(a, d=10):
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


def ser(s, d=None):
    """A dated series as {t0, t (day offsets), v}."""
    s = s.dropna()
    t0 = s.index[0]
    return dict(t0=t0.date().isoformat(), t=[int(v) for v in (s.index - t0).days],
                v=[float(x) for x in s.values] if d is None else rd(s.values, d))


def sec_numbers(N):
    """Every number of ch15_numbers.json, recomputed by the chapter functions (save=False: no file is written)."""
    tree(g.fig_bank_indices(save=False), N['indices'], 'indices')
    tree(g.fig_episodes(save=False), N['episodes'], 'episodes')
    tree(g.fig_rolling_corr(save=False), N['corr'], 'corr')
    tree(g.contagion_test(), N['fr'], 'fr')
    T = g.tail_table()
    tree(T, N['tail'], 'tail')
    tree(g.fig_tail_scatter(save=False), N['tail_fig'], 'tail_fig')
    tree(g.fig_covar_qr(save=False), N['qr'], 'qr')
    print('  covar table (bootstrap)')
    CT = g.covar_table()
    tree(CT, N['covar'], 'covar')
    g.fig_dcovar(CT, save=False)
    tree(g.fig_var_vs_dcovar(CT, save=False), N['spearman_var_dcovar'], 'spearman_var_dcovar')
    tree(float(stats.spearmanr([CT[k]['dcovar'] for k in CT], [CT[k]['dcovar5'] for k in CT])[0]), N['spearman_1_5'], 'spearman_1_5')
    tree(g.fig_dcovar_dynamic(save=False), N['dyn'], 'dyn')
    print('  MES table (bootstrap)')
    MT = g.mes_table()
    tree(MT, N['mes'], 'mes')
    g.fig_mes(MT, save=False)
    tree(g.fig_srisk(MT, save=False), N['srisk_pick'], 'srisk_pick')
    tree(float(stats.spearmanr([MT[k]['mes5'] for k in g.ALL], [CT[k]['dcovar'] for k in g.ALL])[0]), N['spearman_mes_dcovar'], 'spearman_mes_dcovar')
    res = g.mes_vs_crisis()
    tree({'banks': res[0], **res[1]}, N['mes_crisis'], 'mes_crisis')
    g.fig_mes_crisis(res, save=False)
    print('  networks')
    tree(g.fig_dgc(save=False), N['dgc'], 'dgc')
    tree(g.fig_granger_net(save=False), N['gnet'], 'gnet')
    c, p, n = g.dy_static()
    tree({'p': p, 'n': n, 'total': c['total'], 'to': c['to'].to_dict(), 'from': c['from'].to_dict(), 'net': c['net'].to_dict()}, N['dy'], 'dy')
    g.fig_dy_table(c, save=False)
    tree(g.fig_dy_rolling(save=False), N['dy_roll'], 'dy_roll')
    tree(g.END, N['end'], 'end')
    plt.close('all')
    return CT, MT, T, c


def sec_prices():
    """Closes of the banks (bank_price) and of the indices (load_close); the browser joins them as prices() does."""
    P = {k: ser(g.bank_price(k)) for k in g.ALL}
    for k in ('sp500', 'stoxx50', 'bet'):
        P[k] = ser(g.load_close(k, g.START, g.END))
    for reg in ('US', 'EU', 'RO'):                       # the joined returns of these closes are the chapter's returns
        keys = g.REGIONS[reg] + [g.INDEX[reg]]
        cols = []
        for k in keys:
            s = P[k]
            idx = pd.Timestamp(s['t0']) + pd.to_timedelta(s['t'], unit='D')
            cols.append(pd.Series(s['v'], index=idx, name=k))
        J = pd.concat(cols, axis=1, join='inner').dropna().loc[g.REGION_START[reg]:]
        R = 100 * np.log(J).diff().dropna()
        for k, days in g.BAD_DAYS.items():
            if k in R.columns:
                R = R.drop(pd.to_datetime(days), errors='ignore')
        if not R.equals(g.region_returns(reg)[keys]):
            raise SystemExit(f'prices {reg}: the exported closes do not give the returns of the chapter')
    return P


def sec_fits(CT):
    """Quantile regressions of the regional index on each bank (covar) at the levels of LEVELS and at the median."""
    out = {}
    for reg in ('US', 'EU', 'RO'):
        R = g.region_returns(reg)
        for k in g.REGIONS[reg]:
            f = {f'{a:g}': g.covar(R[g.INDEX[reg]], R[k], a) for a in LEVELS}
            for key, a in (('0.01', 'dcovar'), ('0.05', 'dcovar5')):
                tree(f[key]['dcovar'], CT[k][a], f'fits/{k}/{key}')
            a50, b50 = g.qreg(R[g.INDEX[reg]].values, R[k].values, 0.5)
            out[k] = dict(ab={q: [v['a'], v['b']] for q, v in f.items()}, chk={q: [v['covar'], v['dcovar'], v['var_i'], v['var_sys']] for q, v in f.items()},
                          med=[float(a50), float(b50)], ci=CT[k]['ci'])
    return out


def sec_dyn(N):
    """Time-varying Delta-CoVaR (dynamic_dcovar) of JPM, DBK and TLV at the levels of DYN_LEVELS."""
    out = {}
    for k in ('JPM', 'DBK', 'TLV'):
        out[k] = {}
        for a in DYN_LEVELS:
            s, b = g.dynamic_dcovar(k, a)
            if a == 0.01:
                tree({'mean': float(s.mean()), 'max': float(s.max()), 'date_max': s.idxmax().date().isoformat(), 'b': b,
                      'y2017': float(s.loc['2017'].mean()), 'last': float(s.iloc[-1])}, N['dyn'][k], f'dyn/{k}')
            out[k][f'{a:g}'] = dict(ser(s, 8), b=b)
    return out


def sec_dyroll(N):
    """Rolling total connectedness (rolling_total) on windows of DY_WINDOWS days, moved by 5 days."""
    out = {}
    for w in DY_WINDOWS:
        tot = g.rolling_total(window=w)
        if w == g.DY_WINDOW:
            tree({'max': float(tot.max()), 'date_max': tot.idxmax().date().isoformat(), 'y2016': float(tot.loc['2016'].mean()),
                  'max2016': float(tot.loc['2016'].max())}, {k: N['dy_roll'][k] for k in ('max', 'date_max', 'y2016', 'max2016')}, 'dy_roll/200')
        out[str(w)] = dict(d=[d.date().isoformat() for d in tot.index], v=rd(tot.values, 8))
    return out


def main():
    N = json.load(open(os.path.join(HERE, 'ch15_numbers.json')))
    print('numbers of ch15_numbers.json')
    CT, MT, T, c = sec_numbers(N)
    out = dict(end=g.END, banks={k: dict(name=v[1], region=v[2]) for k, v in g.BANKS.items()}, regions=g.REGIONS, index=g.INDEX,
               index_name=g.INDEX_NAME, reg_col=g.REG_COL, reg_lab=g.REG_LAB, region_start=g.REGION_START, bad_days=g.BAD_DAYS,
               events=g.EVENTS, episodes=g.EPISODES, colors=dict(MainBlue=st.MainBlue, IDAred=st.IDAred, Forest=st.Forest, Amber=st.Amber,
                                                                 Orange=st.Orange, Purple=st.Purple, Teal=st.Teal, Crimson=st.Crimson),
               k_cap=g.K_CAP, dy=dict(p=N['dy']['p'], h=g.DY_H, window=g.DY_WINDOW, step=g.DY_STEP, table=c['table'].values.tolist()), N=N)
    print('prices')
    out['px'] = sec_prices()
    print('quantile regressions')
    out['fits'] = sec_fits(CT)
    out['copula'] = {k: {q: T[k][q] for q in ('tau', 'rho', 'nu', 'lambda')} for k in T}
    out['mes_ci'] = {k: [MT[k]['lo'], MT[k]['hi']] for k in MT}
    print('time-varying Delta-CoVaR')
    out['dyn'] = sec_dyn(N)
    print('rolling connectedness')
    out['dyroll'] = sec_dyroll(N)
    js = ('// Inputs of the Chapter 15 interactive page, written by Quantlets/Ch_15/export_interactive.py.\n'
          'window.CH15_DATA = ' + json.dumps(clean(out), separators=(',', ':'), allow_nan=False) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
