"""
generate_all_charts.py -- charts and numbers of Chapter 0 (SFM): Introduction
=============================================================================
Pipeline proof for the new SFM build: the course data (sfm_data.py) and the chart style (sfm_style.py).
  * fig_markets   -- growth of 100 invested in the S&P 500, DAX, BET-TR, Bitcoin and gold (log scale);
  * market_table  -- annualised return and volatility of each series, on its own calendar.
Output: charts/sfm_ch0_markets.pdf/.png, Quantlets/Ch_00/ch0_markets.csv
Run:  python3 Quantlets/Ch_00/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import LABELS, load_close, log_returns, periods_per_year   # noqa: E402
import sfm_style as st                                                    # noqa: E402

TABLE_DIR = HERE
MARKETS_CH0 = ['sp500', 'dax', 'bettr', 'btc', 'gold']
START_CH0 = '2015-01-01'


def market_table(names=MARKETS_CH0, start=START_CH0):
    """Annualised mean log return and volatility (%), with the actual observation frequency of each series."""
    rows = {}
    for k in names:
        r = log_returns(k, start)
        ppy = periods_per_year(r)
        rows[LABELS[k]] = {'first': r.index[0].date(), 'last': r.index[-1].date(), 'n': len(r),
                           'obs_per_year': ppy, 'ann_return': r.mean() * ppy, 'ann_vol': r.std() * np.sqrt(ppy)}
    return pd.DataFrame(rows).T


def fig_markets(names=MARKETS_CH0, start=START_CH0, save=True):
    """Growth of 100 invested at the first common date, log scale, legend below the plot."""
    px = pd.concat([load_close(k, start) for k in names], axis=1).ffill().dropna()
    idx = 100 * px / px.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k in names:
        ax.plot(idx.index, idx[k], color=st.COL[k], lw=1.3, label=LABELS[k])
    ax.set_yscale('log')
    ax.set_ylabel('Value of 100 invested (log scale)')
    st.legend_outside_bottom(ax, ncol=len(names), y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_markets')
    return idx


if __name__ == '__main__':
    st.apply()
    tab = market_table()
    tab.to_csv(os.path.join(TABLE_DIR, 'ch0_markets.csv'))
    print(tab.round(2))
    fig_markets()
