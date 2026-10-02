"""
build_quantlets.py -- Quantlet folders of Chapter 0 (SFM): Introduction
=======================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Run:  python3 Quantlets/Ch_00/generate_all_charts.py && python3 Quantlets/Ch_00/build_quantlets.py
      python3 notebooks/add_colab_banner.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Friday, 2 October 2026'
DATA = 'Daily market data from EODHD (S&P 500, DAX, BET-TR, Bitcoin, gold XAU/USD), data/market of the SFM repository'

QUANTLETS = [
    dict(name='SFM_ch0_markets',
         desc='Growth of 100 invested in the S&P 500, DAX, BET-TR, Bitcoin and gold since January 2015 (log scale), and '
              'the annualised mean log return and volatility of each series, annualised with its actual observation '
              'frequency (about 252 days a year for exchanges, 365 for Bitcoin).',
         keywords='financial markets, stock index, Bitcoin, gold, BET-TR, log returns, volatility, annualisation',
         consts=['MARKETS_CH0 = ' + repr(g.MARKETS_CH0), f'START_CH0 = {g.START_CH0!r}'],
         funcs=[g.market_table, g.fig_markets],
         run="tab = market_table()\nprint(tab.round(2))\nfig_markets()",
         charts=['sfm_ch0_markets'], extra=['ch0_markets.csv']),
]

if __name__ == '__main__':
    build_all(QUANTLETS, 0, 'Introduction', HERE, data=DATA, submitted=SUBMITTED)
