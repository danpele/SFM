"""Export the inputs of the Chapter 0 interactive page (SFM/interactive/ch0_data.js).

Loads every series with the loaders and constants of generate_all_charts.py (same start dates, same cleaning, same
columns), so the page starts from the numbers of the slides. Exported: the daily closes of the five markets of the growth
chart and of the eleven markets of the risk-return chart from January 2015, of the S&P 500 and the BET from January 2000
(drawdowns, daily returns, histogram) and of the VIX from January 2000, each on its own calendar, with 9 significant
digits. The browser recomputes the growth of 100, the annualised means and volatilities, the drawdowns, the VIX
statistics, the rolling volatility, the histogram and the tail counts from these prices.
The script checks that compute_values() still reproduces ch0_values.json (the numbers quoted in the lecture) and that
the S&P 500 from 2015 is the 2000 series cut at 2015, and stops otherwise. It writes only interactive/ch0_data.js.
Run:  python3 Quantlets/Ch_00/export_interactive.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

import generate_all_charts as g
from sfm_data import END                                # noqa: E402  (path set by generate_all_charts)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch0_data.js')


def pack(s):
    """Dates as day offsets from 1970-01-01, values with 9 significant digits."""
    days = s.index.values.astype('datetime64[D]').astype(np.int64).tolist()
    return dict(d0=days[0], dd=np.diff(days).tolist(), v=[float(f'{x:.9g}') for x in s.values])


def main():
    V = json.load(open(os.path.join(HERE, 'ch0_values.json')))
    W = g.compute_values()
    for k, b in V.items():
        a = W[k]
        ok = a == b if isinstance(b, str) else math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        if not ok:
            raise SystemExit(f'ch0_values.json: {k} = {b}, compute_values gives {a}')
    short = sorted(set(g.MARKETS_CH0) | set(g.RISK_RETURN))
    px = {k: g.load_close(k, g.START_CH0) for k in short}
    long = {k: g.load_close(k, g.START_LONG) for k in ('sp500', 'bet', 'vix')}
    for k in ('sp500', 'bet'):
        a, b = long[k].loc[g.START_CH0:], g.load_close(k, g.START_CH0)
        assert a.equals(b), k
    px.pop('sp500')                                     # the page cuts the 2000 series at 2015
    s = {k: pack(p) for k, p in px.items()}
    s.update({k + '_all': pack(p) for k, p in long.items()})
    d = dict(start=g.START_CH0, start_long=g.START_LONG, end=END,
             markets=g.MARKETS_CH0, rr=g.RISK_RETURN, labels={k: g.LABELS[k] for k in short + ['bet', 'vix']},
             colors={k: g.st.COL[k] for k in g.MARKETS_CH0 + ['bet', 'vix']}, colors_rr=g.COLOURS_RR,
             markers_rr={k: {'s': 'square', 'D': 'diamond'}[m] for k, m in g.MARKERS_RR.items()},
             offsets_rr={k: list(v) for k, v in g.OFFSETS_RR.items()},
             palette=dict(blue=g.st.MainBlue, red=g.st.IDAred, dark=g.st.DarkText), s=s, V=V)
    js = ('// Inputs of the Chapter 0 interactive page, written by Quantlets/Ch_00/export_interactive.py.\n'
          'window.CH0_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
