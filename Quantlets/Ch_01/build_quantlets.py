"""
build_quantlets.py -- Quantlet folders of Chapter 1 (SFM): Data, returns and indicators
======================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The older Chapter 1 folders (SFM_ch1_returns, SFM_ch1_sharpe_ratio, ...) are kept unchanged.
Run:  python3 Quantlets/Ch_01/generate_all_charts.py && python3 Quantlets/Ch_01/seminar1.py
      python3 Quantlets/Ch_01/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 1
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
DATA = ('Daily market data from EODHD (S&P 500, DAX, BET, BET-TR, Bitcoin, BVB stocks), data/market of the SFM repository; '
        'EUR/RON reference rate of the BNR')
CONSTS = [f'START = {g.START!r}', 'ASSETS = ' + repr(g.ASSETS), 'COLORS = ' + repr(g.COLORS), 'SCATTER = ' + repr(g.SCATTER)]
CORE = [g.simple_ret, g.log_ret, g.cagr, g.drawdown, g.max_drawdown, g.downside_deviation, g.sharpe_se, g.performance,
        g.describe]

QUANTLETS = [
    dict(name='SFM_ch1_data_quality',
         desc='Data checks on real series: the EUR/RON series from EODHD against the BNR reference rate (days with gaps '
              'above one percentage point, annualised volatilities), the close and adjusted close of Banca Transilvania '
              '(TLV) with the 10-to-1 share consolidation of August 2022, the BET price index against the BET-TR total '
              'return index, and the last five OHLCV bars of TLV. Based on the SFE Quantlet SFEtimeret.',
         keywords='data quality, adjusted close, corporate actions, dividends, share consolidation, total return index, '
                  'exchange rate, BNR reference rate, OHLCV',
         consts=CONSTS, funcs=[g.fig_eurron, g.fig_tlv_adjusted, g.fig_bet_bettr, g.ohlc_last],
         run="print(fig_eurron())\nprint(fig_tlv_adjusted())\nprint(fig_bet_bettr())\nprint(pd.DataFrame(ohlc_last()).T)",
         charts=['sfm_ch1_eurron_check', 'sfm_ch1_tlv_adjusted', 'sfm_ch1_bet_bettr']),
    dict(name='SFM_ch1_prices_returns',
         desc='S&P 500 since 2000: the price level and the daily log returns, with the augmented Dickey-Fuller test on '
              'the log price (with trend) and on the log returns. Based on the SFE Quantlets SFEDAXlogreturns and '
              'SFEtimeret.',
         keywords='prices, log returns, stationarity, unit root, augmented Dickey-Fuller test, S&P 500',
         consts=CONSTS, funcs=[g.log_ret, g.fig_price_returns],
         run="print(fig_price_returns())",
         charts=['sfm_ch1_price_returns']),
    dict(name='SFM_ch1_simple_log_returns',
         desc='Simple returns R and log returns r = ln(1 + R): the curve r(R) against the 45-degree line, with the largest '
              'daily moves of Bitcoin and the S&P 500 marked. Based on the SFE Quantlets SFEReturns and SFEcomplogreturns.',
         keywords='simple returns, log returns, continuously compounded returns, Taylor expansion, Bitcoin, S&P 500',
         consts=CONSTS, funcs=[g.simple_ret, g.fig_simple_log],
         run="print(fig_simple_log())",
         charts=['sfm_ch1_simple_log']),
    dict(name='SFM_ch1_portfolio_returns',
         desc='Multi-period and portfolio returns: growth of 100 invested in the S&P 500, DAX, BET and Bitcoin since 2015 '
              '(log scale), and a 50/50 daily rebalanced portfolio of Banca Transilvania and OMV Petrom, comparing the '
              'exact portfolio log return with the weighted sum of log returns. Based on the SFE Quantlet SFEportlogreturns.',
         keywords='cumulative return, wealth, portfolio return, rebalancing, log returns, aggregation over time',
         consts=CONSTS, funcs=[g.cagr, g.fig_growth, g.portfolio_example],
         run="print(fig_growth())\nprint(portfolio_example())",
         charts=['sfm_ch1_growth']),
    dict(name='SFM_ch1_descriptive_stats',
         desc='Descriptive statistics of daily log returns, 2015-2026 (S&P 500, DAX, BET, BET-TR, Bitcoin and five BVB '
              'stocks): mean, standard deviation, skewness, excess kurtosis, extremes with dates, the annualised mean '
              'with its standard error; histograms against the Normal distribution and rolling one-year volatility. '
              'Based on the SFE Quantlet SFEfdescStat.',
         keywords='descriptive statistics, skewness, kurtosis, heavy tails, standard error, annualisation, rolling volatility',
         consts=CONSTS, funcs=[g.describe, g.descriptive_table, g.fig_hist, g.fig_rolling_vol],
         run="tab = descriptive_table()\nprint(tab.round(3))\nprint(fig_hist())\nprint(fig_rolling_vol())",
         charts=['sfm_ch1_hist_normal', 'sfm_ch1_rolling_vol_1y'], extra=['ch1_descriptive.csv']),
    dict(name='SFM_ch1_performance',
         desc='Performance indicators, 2015-2026, for indices, Bitcoin and BVB stocks: CAGR, annualised volatility, '
              'Sharpe ratio with the Lo (2002) standard error, Sortino ratio, maximum drawdown with dates, Calmar ratio; '
              'drawdown curves over the full history of the S&P 500, BET and Bitcoin; the risk-return chart and the '
              'Sharpe ratios with 95% intervals.',
         keywords='performance indicators, CAGR, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, standard error',
         consts=CONSTS, funcs=CORE + [g.performance_table, g.fig_drawdowns, g.fig_risk_return, g.fig_sharpe_ci],
         run="perf = performance_table()\nprint(perf.round(3))\nprint(fig_drawdowns())\nfig_risk_return(perf)\nfig_sharpe_ci(perf)",
         charts=['sfm_ch1_drawdowns', 'sfm_ch1_risk_return', 'sfm_ch1_sharpe_ci'], extra=['ch1_performance.csv']),
    dict(name='SFM_ch1_volatility_drag',
         desc='Volatility drag on real data: for each asset, the arithmetic annual mean of simple returns minus the annual '
              'mean of log returns, against half the annual variance (2015-2026).',
         keywords='volatility drag, variance drag, Jensen inequality, geometric mean, arithmetic mean, CAGR',
         consts=CONSTS, funcs=CORE + [g.performance_table, g.fig_vol_drag],
         run="perf = performance_table()\nprint(perf[['mean_arith', 'mean_log', 'drag', 'half_var']].round(4))\nfig_vol_drag(perf)",
         charts=['sfm_ch1_vol_drag']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar1 as s
    QUANTLETS.append(dict(
        name='SFM_ch1_seminar',
        desc='Seminar 1 of Statistics of Financial Markets: returns and adjusted prices on paper, the EUR/RON data check, '
             'the mean return of the BET with its confidence interval, the Sharpe ratio of the S&P 500 with the Lo (2002) '
             'standard error and a bootstrap interval, the maximum drawdown of the BET since 1997.',
        keywords='seminar, returns, adjusted prices, confidence interval, Sharpe ratio, bootstrap, maximum drawdown',
        consts=CONSTS + [f'SEM_START = {s.SEM_START!r}'],
        funcs=CORE + [s.returns_table, s.adjusted_prices, s.path_indicators, s.b1_eurron, s.b2_mean_ci, s.b2_chart,
                      s.bootstrap_sharpe, s.b4_sharpe, s.b6_drawdown],
        run="print(returns_table([50, 55, 48]))\nprint(b1_eurron())\nb2_chart()\nprint(b2_mean_ci('bet'))\n"
            "print(b4_sharpe())\nprint(b6_drawdown())",
        charts=['ch1_sem_b1', 'ch1_sem_b2', 'ch1_sem_b4', 'ch1_sem_b6']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 1, 'Data, returns and indicators', HERE, data=DATA, submitted=SUBMITTED)
