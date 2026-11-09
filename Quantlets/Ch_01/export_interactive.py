"""Export the inputs of the Chapter 1 interactive page (SFM/interactive/ch1_data.js).

Loads every series with the loaders of generate_all_charts.py (same start dates, same cleaning, same columns), so the
page starts from the same numbers as the slides. Prices are exported with 9 significant digits; the browser computes
returns, drawdowns, rolling volatilities and performance indicators from them. The ADF statistics are taken from
ch1_numbers.json (written by generate_all_charts.py). The EUR/RON reference rate is read from the BNR archive by
sfm_data.read_reference_rate, as in the chapter; the script stops if it no longer reproduces ch1_numbers.json.
Run:  python3 Quantlets/Ch_01/export_interactive.py
"""
import json
import os

import numpy as np
import pandas as pd

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch1_data.js')
SHORT = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'bettr': 'BET-TR', 'btc': 'Bitcoin', 'tlv': 'TLV', 'snp': 'SNP',
         'brd': 'BRD', 'snn': 'SNN', 'tgn': 'TGN'}


def pack(df):
    """Dates as day offsets, values with 9 significant digits; df: Series or DataFrame on one calendar."""
    df = df.to_frame() if isinstance(df, pd.Series) else df
    days = df.index.values.astype('datetime64[D]').astype(np.int64).tolist()
    return dict(d0=days[0], dd=np.diff(days).tolist(),
                v={c: [float(f'{x:.9g}') for x in df[c].values] for c in df.columns})


def main():
    N = json.load(open(os.path.join(HERE, 'ch1_numbers.json')))
    S = g.START
    s = {}
    # performance, growth, histograms, rolling volatility: each asset on its own calendar from START; for S&P 500, BET
    # and Bitcoin, load_close(k, START) equals the full history below from START on (checked), so the page slices it
    full = {'sp500': g.load_close('sp500'), 'bet': g.load_close('bet', '1997-09-19'), 'btc': g.load_close('btc')}
    for k in g.ASSETS:
        if k in full:
            assert g.load_close(k, S).equals(full[k].loc[S:]), k
        else:
            s[k] = pack(g.load_close(k, S))
    # full histories: S&P 500 from 2000 (price/returns, drawdowns), BET from 1997, Bitcoin from 2014 (drawdowns)
    for k, p in full.items():
        s[k + '_all'] = pack(p)
    # BET vs BET-TR from the first common day
    s['betpair'] = pack(pd.concat([g.load_close('bet', '2014-09-23'), g.load_close('bettr', '2014-09-23')], axis=1).dropna())
    # Banca Transilvania: close as traded and adjusted close, weekdays from START
    t = g.read_market('TLV.RO').loc[S:]
    t = t[t.index.dayofweek < 5]
    s['tlvpair'] = pack(t[['close', 'adjusted_close']])
    # EUR/RON: BNR reference rate and the EODHD series on common days
    x = pd.concat([g.load_close('eurron'), g.load_close('eurron_eodhd')], axis=1).dropna()
    s['eurpair'] = pack(x)
    r = 100 * np.log(x).diff().dropna()
    n1 = int(((r['eurron_eodhd'] - r['eurron']).abs() > 1).sum())
    assert len(r) == N['eurron']['n'] and n1 == N['eurron']['days_diff_1pct'], ('EUR/RON differs from ch1_numbers.json', len(r), n1)
    d = dict(start=S, end=N['end'], assets=g.ASSETS, colors={k: g.COLORS[k] for k in list(g.ASSETS) + ['eurron', 'eurron_eodhd']},
             labels={k: g.LABELS[k] for k in g.ASSETS}, short=SHORT, s=s,
             scatter={k: g.SCATTER[g.LABELS[k]] for k in g.ASSETS},
             adf={k: N['adf'][k] for k in ('adf_price', 'p_price', 'adf_ret', 'crit5_ct', 'crit5_c', 'first')})
    js = ('// Inputs of the Chapter 1 interactive page, written by Quantlets/Ch_01/export_interactive.py.\n'
          'window.CH1_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
