"""Export the inputs of the Chapter 2 interactive page (SFM/interactive/ch2_data.js).

Uses the functions of generate_all_charts.py (same sample, same returns, same Student-t fits), so the page starts
from the same numbers as the slides. The daily log returns are exported in % with 6 decimals; the browser computes
moments, quantiles, kernel densities, autocorrelations and leverage correlations from them. The Student-t fits and
the simulated CLT tail shares are taken from ch2_numbers.json (written by generate_all_charts.py).
The Student-t fits on the samples that start in 2011, ..., 2024 use the chapter's t_fit().
Run:  python3 Quantlets/Ch_02/export_interactive.py
"""
import json
import os

import numpy as np

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch2_data.js')


def main():
    N = json.load(open(os.path.join(HERE, 'ch2_numbers.json')))
    d = dict(start=N['start'], end=N['end'], assets=g.ASSETS, indices=g.INDICES, colors={k: g.COLORS[k] for k in g.ASSETS},
             short={k: g.SHORT[k] for k in g.ASSETS}, pal=g.PAL, horizons=g.HORIZONS)
    ret = {}
    for k in g.ASSETS:
        r = g.returns(k)
        days = (r.index.values.astype('datetime64[D]').astype(np.int64)).tolist()   # days since 1970-01-01
        ret[k] = dict(d0=days[0], dd=np.diff(days).tolist(), r=np.round(r.values, 6).tolist())
    d['ret'] = ret
    d['tfit'] = {k: {c: N['mom'][k][c] for c in ('nu', 'loc', 'scale', 'll_t', 'll_n', 'aic_t', 'aic_n')} for k in g.ASSETS}
    # Student-t fits of the chapter (t_fit: scipy.stats.t.fit) on the samples that start on 1 January of 2011, ..., 2024
    d['tfit_y'] = {}
    for k in g.ASSETS:
        d['tfit_y'][k] = {}
        for y in range(2011, 2025):
            f = g.t_fit(g.returns(k, f'{y}-01-01'))
            d['tfit_y'][k][y] = [round(float(f[c]), 6) for c in ('nu', 'loc', 'scale', 'll_t', 'll_n')]
    d['clt_sim'] = {n: v['p_tail'] for n, v in N['clt'].items()}          # 20,000 simulations, seed 7 (slide)
    d['crash'] = ['2020-02-20', '2020-04-30']                              # window dropped for rho1_ex2020 in the chapter
    js = ('// Inputs of the Chapter 2 interactive page, written by Quantlets/Ch_02/export_interactive.py.\n'
          'window.CH2_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
