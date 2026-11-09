"""Export the inputs of the Chapter 16 interactive page (SFM/interactive/ch16_data.js).

Calls the functions of generate_all_charts.py (same data files in data/market, same window 2015-01-01 to 2026-09-18), so the
page starts from the numbers of the slides. Exported: the daily closes of the S&P 500, the BET and Bitcoin over their full
history in the course data (load_close; the browser cuts the window, recomputes the returns, the performance measures, the
moments, the Hill index, the Ljung-Box and variance-ratio tests, the ACF, VaR and ES, the Hurst exponent, the one-day VaR 1%
forecasts of historical simulation and EWMA-Normal with their exceptions and the Kupiec test); the GARCH(1,1)-t fits of
garch_t on the windows of WINDOWS (the browser does not refit the GARCH model) and the block sizes of the R/S statistic
(block_sizes) on these windows; the ML estimates of the stable alpha for daily,
weekly and monthly returns from Quantlets/Ch_03/ch3_numbers.json (Chapter 3, quoted on the slide „Capitolul 3 (2/2)”).
Extra runs of the chapter's code (labelled on the page): the windows that start in 2017, 2019 and 2021.
The script checks every number of ch16_numbers.json against the chapter functions (to 1e-12 relative, exact for counts, dates
and labels), the table ch16_summary_table.csv against summary_table, the running variance of Chapter 3 against
ch3_numbers.json, and stops if one no longer matches. It writes only interactive/ch16_data.js (no chart, no table: every
chart function is called with save=False).
Run:  python3 Quantlets/Ch_16/export_interactive.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_all_charts as g

plt.show = lambda *a, **k: None              # the chart functions are called with save=False: draw, show nothing

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch16_data.js')
WINDOWS = ('2015-01-01', '2017-01-01', '2019-01-01', '2021-01-01')     # 2015: the lecture; the others: extra runs
CH3_START = '2000-01-01'                                                # Chapter 3: indices since 2000, Bitcoin since 2014


def tree(a, b, what, tol=1e-12):
    """a (recomputed) against b (reference): floats to tol (relative), the rest exactly; stops on a mismatch."""
    if isinstance(b, dict):
        A = {str(k): v for k, v in a.items()}
        if set(A) != set(b):
            raise SystemExit(f'{what}: keys {sorted(A)} do not match {sorted(b)}')
        for k in b:
            tree(A[k], b[k], f'{what}/{k}', tol)
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{what}: length {len(a)} does not match ({len(b)})')
        for i, (x, y) in enumerate(zip(a, b)):
            tree(x, y, f'{what}[{i}]', tol)
    elif b is None or isinstance(b, (bool, str)) or (isinstance(b, int) and not isinstance(a, (float, np.floating))):
        if a != b and not (isinstance(a, (int, np.integer)) and int(a) == b):
            raise SystemExit(f'{what}: {a} does not match ({b})')
    else:
        a, b = float(a), float(b)
        if math.isnan(b) and math.isnan(a):
            return
        if not abs(a - b) <= tol * max(1.0, abs(b)):
            raise SystemExit(f'{what}: {a} does not match ({b})')


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


def sec_numbers(N):
    """Every number of ch16_numbers.json, recomputed by the chapter functions (save=False: no file is written)."""
    tree(g.START, N['start'], 'start')
    tree(g.END, N['end'], 'end')
    S = {}
    for k in g.ASSETS:
        print('  summary', k)
        S[k] = g.summary(k)
        tree(S[k], N['summary'][k], f'summary/{k}')
    T = g.summary_table(S).round(3)
    csv = pd.read_csv(os.path.join(HERE, 'ch16_summary_table.csv'), index_col=0)
    if not (list(csv.index) == list(T.index) and list(csv.columns) == list(T.columns) and np.allclose(csv.values, T.values, rtol=0, atol=5e-4)):
        raise SystemExit('ch16_summary_table.csv does not match summary_table')
    print('  backtests')
    bt = g.fig_backtest('bet', save=False)
    tree(bt, N['backtest']['bet'], 'backtest/bet')
    for k in ('sp500', 'btc'):
        tree(g.var_backtest(k)[0], N['backtest'][k], f'backtest/{k}')
    g.fig_three_series(save=False)
    g.fig_stylised(save=False)
    plt.close('all')
    return S


def sec_prices():
    """Closes of the three series over their full history; each window of WINDOWS and the full history must be slices of it."""
    P = {}
    for k in g.ASSETS:
        s = g.load_close(k, None, g.END)
        for w in WINDOWS:
            if not g.load_close(k, w, g.END).equals(s.loc[w:]):
                raise SystemExit(f'prices {k}: the window from {w} is not a slice of the full series')
        if not g.log_returns(k, None, g.END).equals(100 * np.log(s).diff().dropna().rename(k)):
            raise SystemExit(f'prices {k}: the returns of the full series differ from log_returns')
        t0 = s.index[0]
        P[k] = dict(t0=t0.date().isoformat(), t=[int(v) for v in (s.index - t0).days], v=[float(x) for x in s.values])
    return P


def sec_garch(S):
    """GARCH(1,1)-t of every series on every window (the lecture window is checked against ch16_numbers.json)."""
    out = {}
    for k in g.ASSETS:
        out[k] = {}
        for w in WINDOWS:
            f = S[k]['garch'] if w == g.START else g.garch_t(g.returns(k, w))
            out[k][w] = f
    return out


def sec_sizes():
    """Block sizes of the R/S statistic (block_sizes) for the returns of every series on every window."""
    return {k: {w: [int(n) for n in g.block_sizes(len(g.returns(k, w)))] for w in WINDOWS} for k in g.ASSETS}


def sec_ch3():
    """Chapter 3: ML alpha by horizon (ch3_numbers.json) and the check of the running sample variance of the S&P 500."""
    N3 = json.load(open(os.path.join(HERE, '..', 'Ch_03', 'ch3_numbers.json')))
    for k in g.ASSETS:
        r = g.log_returns(k, CH3_START if k != 'btc' else None, g.END)
        tree(len(r), N3['aggregation'][k]['daily']['n'], f'ch3/{k}/n')
        if not r.equals(g.log_returns(k, None, g.END)):
            raise SystemExit(f'ch3 {k}: the returns of Chapter 3 are not the full series')
    r = g.log_returns('sp500', CH3_START, g.END)
    i = np.arange(1, len(r) + 1)
    v = np.cumsum(r.values ** 2) / i - (np.cumsum(r.values) / i) ** 2
    tree({'data': float(v[-1]), 'data_2010': float(v[np.searchsorted(r.index, pd.Timestamp('2010-01-01'))])},
         {k: N3['runvar_real'][k] for k in ('data', 'data_2010')}, 'ch3/runvar_real')
    return {k: N3['aggregation'][k] for k in g.ASSETS}, {k: N3['runvar_real'][k] for k in ('data', 'data_2010')}


def main():
    N = json.load(open(os.path.join(HERE, 'ch16_numbers.json')))
    print('numbers of ch16_numbers.json')
    S = sec_numbers(N)
    out = dict(start=g.START, end=g.END, assets=g.ASSETS, name=g.NAME, colors=g.COLORS, windows=list(WINDOWS),
               consts=dict(lb=g.LB_LAGS, vr=g.VR_Q, hill=g.HILL_FRAC, lam=g.LAMBDA, a_var=g.ALPHA_VAR, a_es=g.ALPHA_ES,
                           window=g.WINDOW, oos=g.OOS_START, nmin=g.NMIN), N=N)
    print('prices')
    out['px'] = sec_prices()
    print('GARCH(1,1)-t fits on the windows')
    out['garch'] = sec_garch(S)
    out['sizes'] = sec_sizes()
    print('Chapter 3')
    out['agg'], out['runvar'] = sec_ch3()
    js = ('// Inputs of the Chapter 16 interactive page, written by Quantlets/Ch_16/export_interactive.py.\n'
          'window.CH16_DATA = ' + json.dumps(clean(out), separators=(',', ':'), allow_nan=False) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
