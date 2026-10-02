"""
build_quantlets.py -- Quantlet folders of Chapter 0 (SFM): Introduction
=======================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Lecture Quantlets take their code from generate_all_charts.py; the seminar Quantlet (SFM_ch0_seminar) from
seminar0.py (instructor file). After building, the seminar folder holds the full version: run
    python3 notebooks/split_seminar_notebooks.py 0
to replace it with the student version (the full copy goes to ../instructor/Quantlets/Ch_00).
Run:  python3 Quantlets/Ch_00/generate_all_charts.py && python3 Quantlets/Ch_00/seminar0.py
      python3 Quantlets/Ch_00/build_quantlets.py && python3 notebooks/add_colab_banner.py
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
DATA = 'Daily market data from EODHD (EOD Historical Data), data/market of the SFM repository'
# the palette names as attributes of the style namespace `st` (the chapter scripts use st.MainBlue, st.Teal, ...)
ST_PALETTE = ("for _c in ('MainBlue', 'IDAred', 'Forest', 'Amber', 'Purple', 'Orange', 'Teal', 'Crimson', 'DarkText'):\n"
              "    setattr(st, _c, globals()[_c])")
LECTURE_CONSTS = [ST_PALETTE, 'MARKETS_CH0 = ' + repr(g.MARKETS_CH0), f'START_CH0 = {g.START_CH0!r}', f'START_LONG = {g.START_LONG!r}']

QUANTLETS = [
    dict(name='SFM_ch0_markets',
         desc='Growth of 100 invested in the S&P 500, DAX, BET-TR, Bitcoin and gold since January 2015 (log scale), and '
              'the annualised mean log return and volatility of each series, annualised with its actual observation '
              'frequency (about 252 days a year for exchanges, 365 for Bitcoin).',
         keywords='financial markets, stock index, Bitcoin, gold, BET-TR, log returns, volatility, annualisation',
         consts=LECTURE_CONSTS, funcs=[g.market_table, g.fig_markets],
         run="tab = market_table()\nprint(tab.round(2))\nfig_markets()",
         charts=['sfm_ch0_markets'], extra=['ch0_markets.csv'],
         data='Daily market data from EODHD (S&P 500, DAX, BET-TR, Bitcoin, gold XAU/USD), data/market of the SFM repository'),
    dict(name='SFM_ch0_risk_return',
         desc='Annualised volatility against annualised mean log return of eleven markets (stock indices, gold, long US '
              'Treasuries, Bitcoin and Ether), January 2015 to September 2026, each series on its own calendar.',
         keywords='risk, return, volatility, annualisation, stock index, crypto assets, scatter plot',
         consts=LECTURE_CONSTS + ['RISK_RETURN = ' + repr(g.RISK_RETURN), 'COLOURS_RR = ' + repr(g.COLOURS_RR),
                                  'MARKERS_RR = ' + repr(g.MARKERS_RR), 'OFFSETS_RR = ' + repr(g.OFFSETS_RR)],
         funcs=[g.market_table, g.fig_risk_return],
         run="print(fig_risk_return().round(2)[['ann_return', 'ann_vol']])",
         charts=['sfm_ch0_risk_return']),
    dict(name='SFM_ch0_drawdowns',
         desc='Drawdown (loss from the running maximum of the closing price) of the S&P 500 and of the BET index, '
              'January 2000 to September 2026; maximum drawdowns of 2008-2009, 2020 and 2022.',
         keywords='drawdown, maximum drawdown, financial crisis, S&P 500, BET, Bucharest Stock Exchange',
         consts=LECTURE_CONSTS, funcs=[g.drawdown, g.fig_drawdowns],
         run="dd = fig_drawdowns()\nfor k, d in dd.items():\n    print(k, round(d.min(), 1), d.idxmin().date())",
         charts=['sfm_ch0_drawdowns']),
    dict(name='SFM_ch0_vix',
         desc='The VIX index (volatility of the S&P 500 expected by option traders over the next 30 days), January 2000 '
              'to September 2026, with its mean and its two highest closes (November 2008 and March 2020).',
         keywords='VIX, implied volatility, fear index, volatility clustering, financial crisis',
         consts=LECTURE_CONSTS, funcs=[g.fig_vix],
         run="v, top = fig_vix()\nprint(round(v.mean(), 1), [(d.date(), v[d]) for d in top])",
         charts=['sfm_ch0_vix']),
    dict(name='SFM_ch0_bet_history',
         desc='The BET index of the Bucharest Stock Exchange since its launch on 19 September 1997 (1000 points), '
              'log scale, to September 2026.',
         keywords='BET, Bucharest Stock Exchange, Romania, stock index, log scale',
         consts=[ST_PALETTE], funcs=[g.fig_bet_history],
         run="p = fig_bet_history()\nprint(p.iloc[0], p.iloc[-1], round(p.iloc[-1] / p.iloc[0], 1))",
         charts=['sfm_ch0_bet_history']),
    dict(name='SFM_ch0_returns',
         desc='Daily log returns of the S&P 500, January 2000 to September 2026: the time series (volatility clustering) '
              'and the histogram against the Normal density with the same mean and variance (heavy tails, kurtosis).',
         keywords='log returns, volatility clustering, heavy tails, kurtosis, Normal distribution, histogram',
         consts=LECTURE_CONSTS, funcs=[g.fig_returns, g.fig_histogram],
         run="r = fig_returns()\nfig_histogram()\nprint(len(r), round(r.std(), 3), round(stats.kurtosis(r, fisher=False), 2))",
         charts=['sfm_ch0_returns', 'sfm_ch0_histogram']),
    dict(name='SFM_ch0_btc_vol',
         desc='Annualised volatility of daily log returns in each calendar year, 2015 to September 2026: Bitcoin '
              '(365 days a year) against the S&P 500 (its own frequency, about 252). Mini-case of the section AI for '
              'scientific discovery: has Bitcoin become less volatile since the spot ETFs of January 2024?',
         keywords='Bitcoin, volatility, annualisation, spot Bitcoin ETF, crypto assets, AI for scientific discovery',
         consts=LECTURE_CONSTS, funcs=[g.annual_vol, g.fig_btc_vol],
         run="print(fig_btc_vol().round(1))",
         charts=['sfm_ch0_btc_vol']),
]

try:                                            # instructor file (git-ignored): seminar computations
    import seminar0 as s0
    QUANTLETS.append(dict(
        name='SFM_ch0_seminar',
        desc='Seminar 0 of Statistics of Financial Markets: prices to simple and log returns, annualised mean and '
             'volatility, cumulative return, drawdown and a data check of the EUR/RON series from EODHD against the '
             'BNR reference rate (S&P 500, BET-TR and Bitcoin, January 2015 to September 2026).',
        keywords='simple return, log return, annualisation, volatility, drawdown, data check, EUR/RON, seminar',
        consts=[ST_PALETTE, f'START = {s0.START!r}'],
        funcs=[s0.paper_returns, s0.paper_drawdown, s0.annualise, s0.describe, s0.fig_growth, s0.fig_drawdown,
               s0.eurron_check, s0.fig_eurron],
        run="d = describe('sp500')\nprint({k: v for k, v in d.items() if not hasattr(v, 'index')})\n"
            "fig_growth(d, 'ch0_sem_b1_growth', 'S&P 500', st.MainBlue)\n"
            "fig_drawdown(d, 'ch0_sem_b1_drawdown', 'S&P 500', st.MainBlue)\n"
            "c = eurron_check()\nprint(c['date'], c['quote'], c['bnr'])\nfig_eurron(c)",
        charts=['ch0_sem_b1_growth', 'ch0_sem_b1_drawdown', 'ch0_sem_b2_eurron'],
        data='Daily market data from EODHD (S&P 500, BET-TR, Bitcoin, EUR/RON), data/market of the SFM repository; '
             'EUR/RON reference rate of the BNR (National Bank of Romania)'))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 0, 'Introduction', HERE, data=DATA, submitted=SUBMITTED)
