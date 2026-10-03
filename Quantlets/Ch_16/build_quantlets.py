"""
build_quantlets.py -- Quantlet folders of Chapter 16 (SFM): review, the course in one notebook
==============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The notebooks install the arch package first (pip install arch), as Google Colab does not include it.
Run:  python3 Quantlets/Ch_16/generate_all_charts.py && python3 Quantlets/Ch_16/seminar16.py
      python3 Quantlets/Ch_16/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 16
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'notebooks'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Saturday, 3 October 2026'
INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
DATA = ('Daily market data from EODHD (S&P 500 and BET since 2000, Bitcoin since 2014), data/market of the SFM repository')
CONSTS = ['from arch import arch_model',
          f'NAME = {g.NAME!r}', f'ASSETS = {g.ASSETS!r}', f'COLORS = {g.COLORS!r}', f'START = {g.START!r}',
          f'LB_LAGS = {g.LB_LAGS!r}', f'VR_Q = {g.VR_Q!r}', f'HILL_FRAC = {g.HILL_FRAC!r}', f'LAMBDA = {g.LAMBDA!r}',
          f'ALPHA_VAR = {g.ALPHA_VAR!r}', f'ALPHA_ES = {g.ALPHA_ES!r}', f'WINDOW = {g.WINDOW!r}',
          f'OOS_START = {g.OOS_START!r}', f'NMIN = {g.NMIN!r}']
BASE = [g.returns, g.performance, g.moments, g.hill, g.tail_index, g.acf, g.ljung_box, g.variance_ratio, g.ewma_var,
        g.garch_t, g.hs_var, g.hs_es, g.normal_var, g.normal_es, g.kupiec, g.var_backtest, g.block_sizes, g.hurst_rs]

QUANTLETS = [
    dict(name='SFM_ch16_course_summary',
         desc='The course in one notebook: daily log returns of the S&P 500, the BET and Bitcoin on one common window '
              '(2 January 2015 to 18 September 2026). Annualised mean and volatility with the actual frequency, CAGR, '
              'Sharpe ratio with its standard error and maximum drawdown (Chapter 1); skewness, excess kurtosis, '
              'Jarque-Bera and the Hill tail index of the losses with k = 2.5% of n (Chapters 2 and 5); autocorrelation '
              'of returns and absolute returns, Ljung-Box Q(10) on r and r^2, the Lo-MacKinlay robust variance ratio '
              'Z*(5) (Chapters 7-8); GARCH(1,1) with Student-t innovations (Chapter 9); VaR 1% and ES 2.5% by historical '
              'simulation and with the Normal distribution, one-day VaR 1% forecasts since 2017 by historical simulation '
              '(500 days) and EWMA-Normal with the Kupiec test (Chapter 10); the Hurst exponent by R/S of r and |r| '
              '(Chapter 11).',
         keywords='review, log returns, Sharpe ratio, drawdown, excess kurtosis, Jarque-Bera, Hill estimator, '
                  'Ljung-Box, variance ratio, GARCH, value at risk, expected shortfall, backtesting, Kupiec test, '
                  'Hurst exponent, S&P 500, BET, Bitcoin',
         consts=CONSTS, funcs=BASE + [g.summary, g.summary_table, g.fig_three_series, g.fig_stylised, g.fig_backtest],
         run="S = {k: summary(k) for k in ASSETS}\nprint(summary_table(S).round(3))\n"
             "fig_three_series()\nfig_stylised()\nprint(fig_backtest('bet'))",
         charts=['sfm_ch16_three_series', 'sfm_ch16_stylised', 'sfm_ch16_backtest'], extra=['ch16_summary_table.csv']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar16 as s
    QUANTLETS.append(dict(
        name='SFM_ch16_seminar',
        desc='Seminar 16 of Statistics of Financial Markets, review and exam practice: returns, drawdown and the Sharpe '
             'ratio; Jarque-Bera, Student-t, Hill and stable scaling; Ljung-Box and the variance ratio; GARCH, EWMA and '
             'Parkinson volatility; VaR 1%, ES 2.5%, the Kupiec test and the Basel traffic light; logit PD, expected '
             'loss, AUC and Delta-CoVaR, on paper; a full check-up of the BET since 2015; VaR 1% backtests of the '
             'S&P 500 and Bitcoin; efficiency and memory of the BET and the S&P 500 in two windows; the numbers needed '
             'to audit an AI answer.',
        keywords='seminar, review, exam practice, returns, Jarque-Bera, Hill estimator, variance ratio, GARCH, value at '
                 'risk, Kupiec test, CoVaR, Hurst exponent, BET',
        consts=CONSTS + ['Z01, Z025 = stats.norm.ppf(0.01), stats.norm.ppf(0.025)'],
        funcs=BASE + [s.a1_returns, s.a2_tails, s.a3_vr, s.a4_vol, s.a5_var, s.a6_credit, s.b1_checkup,
                      s.b2_backtests, s.b3_memory, s.c2_check],
        run="print(a1_returns())\nprint(a3_vr())\nprint(b1_checkup('bet'))\nprint(b2_backtests())\nprint(b3_memory())",
        charts=['ch16_sem_b1']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 16, 'Review', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
