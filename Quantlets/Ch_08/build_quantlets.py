"""
build_quantlets.py -- Quantlet folders of Chapter 8 (SFM): volatility estimators and volatility clustering
=========================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Run:  python3 Quantlets/Ch_08/generate_all_charts.py && python3 Quantlets/Ch_08/seminar8.py
      python3 Quantlets/Ch_08/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 8
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Saturday, 3 October 2026'
DATA = ('Daily market data from EODHD (S&P 500 and DAX since 1990, BET since 1997, Bitcoin since 2014, Banca '
        'Transilvania, OMV Petrom, BRD and Transgaz since 2010, open, high, low and close prices of the S&P 500 since '
        '2008 and of the DAX since 2006, VIX since 1990), data/market of the SFM repository')
CONSTS = [f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'OHLC_START = {g.OHLC_START!r}', f'BAD_DAYS = {g.BAD_DAYS!r}',
          f'ASSETS = {g.ASSETS!r}', f'STOCKS = {g.STOCKS!r}', f'BVB = {g.BVB!r}', f'OHLC_ASSETS = {g.OHLC_ASSETS!r}',
          f'COLORS = {g.COLORS!r}', f'EST_COLORS = {g.EST_COLORS!r}', f'EST_LABEL = {g.EST_LABEL!r}',
          f'WINDOWS = {g.WINDOWS!r}', f'LAMBDAS = {g.LAMBDAS!r}', f'LB_LAGS = {g.LB_LAGS!r}', f'ARCH_LAGS = {g.ARCH_LAGS!r}',
          f'OOS_START = {g.OOS_START!r}', f'SEED = {g.SEED!r}', f'GHOST = {g.GHOST!r}']
DATAF = [g.returns, g.ohlc, g.ann]
EWMA = [g.ewma_var, g.ewma_series, g.ewma_facts]
RANGE = [g.range_parts, g.daily_estimators, g.yz_k, g.window_estimators, g.rolling_estimators]
TESTS = [g.acf, g.ljung_box, g.arch_lm]
FC = [g.ols, g.forecasts, g.losses, g.mz_table, g.dm_test, g.lambda_qlike]
NODATA = 'Simulated data only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch8_returns_volatility',
         desc='Volatility as the standard deviation of daily log returns, annualised with the actual number of observations '
              'per year: daily returns of the S&P 500 and DAX (since 1990), BET (since 1997) and Bitcoin (since 2014) with '
              'their annual volatility, largest daily move and excess kurtosis; rolling 21-, 63- and 252-day volatility of the '
              'S&P 500; the relative standard error of a sample volatility, 1/sqrt(2n) for Normal returns and '
              'sqrt((kappa - 1)/(4n)) with kurtosis kappa.',
         keywords='volatility, historical volatility, rolling window, annualisation, standard error, kurtosis, S&P 500, DAX, BET, Bitcoin',
         consts=CONSTS, funcs=DATAF + [g.fig_returns_bursts, g.rolling_vol, g.fig_rolling_windows, g.vol_standard_error],
         run="print(fig_returns_bursts())\nprint(fig_rolling_windows())\nprint(vol_standard_error())",
         charts=['sfm_ch8_returns_bursts', 'sfm_ch8_rolling_windows']),
    dict(name='SFM_ch8_ewma',
         desc='The exponentially weighted moving average (EWMA) of RiskMetrics (J.P. Morgan/Reuters, 1996): weights of past '
              'squared returns for lambda = 0.94 and 0.97 against an equally weighted 63-day window, half-life and the number '
              'of days with 99% of the weight; the ghost effect of the 63-day rolling volatility of the S&P 500 after the '
              'crash of March 2020, against EWMA.',
         keywords='EWMA, RiskMetrics, decay factor, half-life, rolling window, ghost effect, volatility, S&P 500, COVID-19',
         consts=CONSTS, funcs=DATAF + EWMA + [g.rolling_vol, g.fig_ewma_weights, g.fig_ghost],
         run="print(fig_ewma_weights())\nprint(fig_ghost())",
         charts=['sfm_ch8_ewma_weights', 'sfm_ch8_ghost']),
    dict(name='SFM_ch8_range_estimators',
         desc='Range-based volatility estimators from open, high, low and close prices: Parkinson (1980), Garman-Klass '
              '(1980), Rogers-Satchell (1991) and Yang-Zhang (2000) against the close-to-close estimator; a simulated '
              'trading day with its open, high, low and close; whole-sample annualised volatility, the overnight share of '
              'the variance and the correlation of overnight and open-to-close returns for the S&P 500 (since 2008), DAX '
              '(since 2006), Bitcoin (since 2014), Banca Transilvania and OMV Petrom (since 2010, adjusted prices); rolling '
              '21-day estimates of the S&P 500 since 2019.',
         keywords='range-based volatility, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, OHLC, overnight return, '
                  'S&P 500, DAX, Bitcoin, BVB',
         consts=CONSTS, funcs=DATAF + RANGE + [g.estimators_table, g.fig_ohlc_path, g.fig_range_rolling],
         run="print(fig_ohlc_path())\nprint(pd.DataFrame(estimators_table()).T.round(3))\nprint(fig_range_rolling())",
         charts=['sfm_ch8_ohlc_path', 'sfm_ch8_range_rolling'], extra=['ch8_estimators_table.csv']),
    dict(name='SFM_ch8_range_efficiency',
         desc='Monte Carlo comparison of the close-to-close, Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang '
              'variance estimators on 1000 windows of 21 days (true daily volatility 1%, intraday Brownian motion on 4680 '
              'steps): bias, efficiency Var(close-to-close)/Var(estimator) and MSE ratio with continuous trading, with an '
              'overnight jump carrying 20% of the daily variance, and with discrete trading (26 observed prices a day).',
         keywords='range-based volatility, efficiency, bias, Monte Carlo, simulation, overnight jump, discrete trading',
         consts=CONSTS, funcs=RANGE + [g.simulate_ohlc, g.efficiency_sim, g.fig_efficiency],
         run="eff, draws = efficiency_sim()\nfig_efficiency(draws)\nprint(pd.DataFrame({(s, e): v for s, d in eff.items() for e, v in d.items()}).T.round(3))",
         charts=['sfm_ch8_efficiency'], data=NODATA),
    dict(name='SFM_ch8_realised_volatility',
         desc='Realised variance from simulated one-minute prices (Andersen, Bollerslev, Diebold and Labys, 2003; '
              'Barndorff-Nielsen and Shephard, 2002): 60 days with a log-AR(1) daily volatility, 5-minute realised volatility '
              'against the absolute daily return and the Parkinson estimator; the signature plot of realised variance '
              'against the sampling interval, with and without microstructure noise.',
         keywords='realised volatility, realised variance, integrated variance, high-frequency data, microstructure noise, '
                  'signature plot, simulation',
         consts=CONSTS, funcs=[g.simulate_intraday, g.realised_var, g.fig_realised, g.fig_signature],
         run="print(fig_realised())\nprint(fig_signature())",
         charts=['sfm_ch8_realised', 'sfm_ch8_signature'], data=NODATA),
    dict(name='SFM_ch8_clustering',
         desc='Volatility clustering in daily returns of the S&P 500, DAX, BET, Bitcoin, Banca Transilvania and OMV Petrom: '
              'the sample ACF of returns, squared returns and absolute returns (lags 1-100) with the i.i.d. 95% band; the '
              'Ljung-Box Q(10) statistic of returns, squared returns (McLeod-Li test) and absolute returns; Engle\'s (1982) '
              'ARCH-LM test with 5 lags; the same S&P 500 returns in a random order, which keep their distribution but lose '
              'the clustering.',
         keywords='volatility clustering, ARCH effects, ACF of squared returns, Ljung-Box, McLeod-Li, ARCH-LM test, Taylor '
                  'effect, permutation, S&P 500, BET, Bitcoin, BVB',
         consts=CONSTS, funcs=[g.returns] + TESTS + [g.clustering_table, g.fig_acf_clustering, g.fig_shuffle],
         run="ct = clustering_table()\n"
             "print(pd.DataFrame({NAME[k]: {'rho1 r^2': v['rho_r2'], 'LB(10) r': v['lb_r']['q'], 'LB(10) r^2': v['lb_r2']['q'], "
             "'LB(10) |r|': v['lb_abs']['q'], 'ARCH-LM(5)': v['arch']['lm'], 'p ARCH-LM': v['arch']['p']} for k, v in ct.items()}).T.round(3))\n"
             "print(fig_acf_clustering())\nprint(fig_shuffle())",
         charts=['sfm_ch8_acf_clustering', 'sfm_ch8_shuffle'], extra=['ch8_clustering_table.csv']),
    dict(name='SFM_ch8_kurgarch',
         desc='Clustering creates heavy tails: the kurtosis of a Normal variance mixture of calm and turbulent days as a '
              'function of the share of turbulent days, and the kurtosis 3 + 6 alpha^2 / (1 - beta^2 - 2 alpha beta - '
              '3 alpha^2) of a GARCH(1,1) process with Normal innovations as a function of alpha for beta = 0.80, 0.85 and '
              '0.90 (as in SFEkurgarch).',
         keywords='kurtosis, heavy tails, variance mixture, GARCH(1,1), fourth moment, volatility clustering',
         consts=CONSTS, funcs=[g.mixture_kurtosis, g.garch_kurtosis, g.fig_kurtosis],
         run="print(fig_kurtosis())",
         charts=['sfm_ch8_kurtosis'], data=NODATA),
    dict(name='SFM_ch8_long_term_vix',
         desc='Long-run behaviour of volatility: the volatility of each calendar year for the S&P 500, DAX, BET and Bitcoin; '
              'the AR(1) of the log monthly realised volatility of the S&P 500, its half-life and skewness; the VIX (implied '
              'volatility of the S&P 500 for the next 30 days) against the realised volatility of the next 21 trading days, '
              'the share of days with the VIX above it and Mincer-Zarnowitz regressions.',
         keywords='long-run volatility, mean reversion, persistence, half-life, VIX, implied volatility, realised volatility, '
                  'variance risk premium, S&P 500',
         consts=CONSTS, funcs=[g.returns, g.ann, g.annual_vol, g.monthly_rv, g.fig_long_term, g.vix_data, g.ols, g.fig_vix],
         run="print(fig_long_term())\nprint(fig_vix())",
         charts=['sfm_ch8_long_term', 'sfm_ch8_vix']),
    dict(name='SFM_ch8_leverage',
         desc='The leverage effect: correlations of today\'s return with future absolute returns, corr(r_t, |r_t+j|) for '
              'j = 1..10, for the S&P 500, DAX, BET and Bitcoin, and the mean squared return after down and up days; the '
              'nonparametric conditional volatility of today\'s return given yesterday\'s return (local linear regression, '
              'quartic kernel, bandwidth 0.04 on decimal returns, as in SFEvolnonparest) for the S&P 500 and the DAX.',
         keywords='leverage effect, asymmetric volatility, volatility feedback, nonparametric regression, local linear, '
                  'quartic kernel, news impact, S&P 500, DAX, Bitcoin',
         consts=CONSTS, funcs=[g.returns, g.leverage, g.fig_leverage, g.quadk, g.lpregest, g.nonparametric_vol, g.fig_news_impact],
         run="lv = fig_leverage()\nprint({k: (round(v['cc'][1], 3), round(v['ratio'], 2)) for k, v in lv.items()})\nprint(fig_news_impact())",
         charts=['sfm_ch8_leverage', 'sfm_ch8_news_impact']),
    dict(name='SFM_ch8_forecast_evaluation',
         desc='Evaluation of next-day variance forecasts of the S&P 500 from 2010: rolling means of squared returns (21, 63, '
              '252 days), EWMA with lambda = 0.94 and 0.97, and the squared VIX; MSE and QLIKE losses (Patton, 2011) with '
              'the proxy r^2, Mincer-Zarnowitz R^2 with r^2 and with a range-based proxy, Diebold-Mariano tests; the QLIKE '
              'of EWMA forecasts as a function of lambda for the S&P 500, BET and Bitcoin; forecasts during 2020.',
         keywords='volatility forecasting, forecast evaluation, QLIKE, MSE, Mincer-Zarnowitz, Diebold-Mariano, EWMA, VIX, '
                  'decay factor, S&P 500, BET, Bitcoin',
         consts=CONSTS, funcs=DATAF + EWMA + RANGE + FC + [g.forecast_table, g.fig_lambda, g.fig_forecast_paths],
         run="ft = forecast_table()\nprint(pd.DataFrame(ft['r2']).T.round(4))\nprint(ft['dm_vix_ewma'], ft['dm_ewma_hist21'])\n"
             "print(fig_lambda())\nfig_forecast_paths()",
         charts=['sfm_ch8_lambda', 'sfm_ch8_forecast_paths'], extra=['ch8_forecast_table.csv']),
    dict(name='SFM_ch8_bvb_range',
         desc='Range-based estimators on the Bucharest Stock Exchange: for Banca Transilvania, OMV Petrom, BRD and Transgaz '
              '(adjusted prices, since 2010) and the S&P 500, the share of trading days with an unchanged close, the ratios '
              'of Parkinson and Yang-Zhang to close-to-close volatility, and the QLIKE of next-day forecasts by the 21-day '
              'close-to-close, Parkinson and Yang-Zhang variances (from 2010).',
         keywords='range-based volatility, thin trading, Bucharest Stock Exchange, BVB, Yang-Zhang, Parkinson, QLIKE',
         consts=CONSTS, funcs=DATAF + RANGE + [g.bvb_range],
         run="print(pd.DataFrame({k: {'flat': v['flat'], 'P/CC': v['park_cc'], 'YZ/CC': v['yz_cc'], **{f'QLIKE {e}': q for e, q in v['qlike'].items()}} "
             "for k, v in bvb_range().items()}).T.round(3))"),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar8 as s
    QUANTLETS.append(dict(
        name='SFM_ch8_seminar',
        desc='Seminar 8 of Statistics of Financial Markets: historical volatility, EWMA, range-based estimators, '
             'Ljung-Box and ARCH-LM statistics and forecast losses on paper; rolling and EWMA volatility of the S&P 500; '
             'range-based estimators of the S&P 500; volatility clustering tests on the S&P 500; next-day variance '
             'forecasts of the S&P 500 with the VIX, compared by MSE, QLIKE and Diebold-Mariano tests.',
        keywords='seminar, volatility, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, volatility clustering, '
                 'ARCH-LM, QLIKE, VIX',
        consts=CONSTS, funcs=DATAF + EWMA + RANGE + TESTS + FC + [g.rolling_vol, s.hist_vol, s.ewma_steps, s.ohlc_one_day,
                                                               s.ohlc_days, s.clustering_from_acf, s.losses_by_hand,
                                                               s.b1_hist, s.b3_range, s.b5_clustering, s.b7_forecast],
        run="print(hist_vol([1.2, -0.8, 0.5, -1.5, 0.6]))\nprint(ewma_steps([-2.0, 0.5, 3.0], 0.94, 1.0))\n"
            "print(ohlc_one_day(100.0, 101.0, 103.5, 99.8, 102.2))\n"
            "print(clustering_from_acf([0.20, 0.15, 0.12, 0.10, 0.08], 1000, r2=0.08, n=995))\n"
            "print(losses_by_hand([1.0, 4.0, 0.25, 2.25], {'A': [1.5] * 4, 'B': [1.0, 2.5, 0.8, 1.8]}))\n"
            "print(b1_hist())\nprint(b3_range())\nprint(b5_clustering())\nprint(b7_forecast())",
        charts=['ch8_sem_b1', 'ch8_sem_b3', 'ch8_sem_b5', 'ch8_sem_b7']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 8, 'volatility estimators and volatility clustering', HERE, data=DATA, submitted=SUBMITTED)
