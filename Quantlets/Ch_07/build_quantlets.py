"""
build_quantlets.py -- Quantlet folders of Chapter 7 (SFM): efficient markets, random walk, unit roots and
variance-ratio tests
=======================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Run:  python3 Quantlets/Ch_07/generate_all_charts.py && python3 Quantlets/Ch_07/seminar7.py
      python3 Quantlets/Ch_07/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 7
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
DATA = ('Daily market data from EODHD (S&P 500 and DAX since 1990, BET since 1997, Bitcoin since 2014, Banca '
        'Transilvania, OMV Petrom, BRD and Transgaz since 2010; Euro Stoxx 50, Nikkei 225, WIG20, BUX, PX, BIST 100, '
        'Bovespa, IPC Mexico, Nifty 50, KOSPI, Shanghai Composite and Ethereum), data/market of the SFM repository')
CONSTS = ['import statsmodels.api as sm', 'from statsmodels.tsa.stattools import adfuller, kpss',
          'from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp',
          f'EXTRA_SERIES = {g.EXTRA_SERIES!r}', f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}',
          f'STOCKS = {g.STOCKS!r}', f'DEVELOPED = {g.DEVELOPED!r}', f'EMERGING = {g.EMERGING!r}', f'CRYPTO = {g.CRYPTO!r}',
          f'MARKETS_START = {g.MARKETS_START!r}', f'BAD_DAYS = {g.BAD_DAYS!r}', f'COLORS = {g.COLORS!r}', f'QS = {g.QS!r}',
          f'LB_LAGS = {g.LB_LAGS!r}', f'WINDOW = {g.WINDOW!r}', f'STEP = {g.STEP!r}', f'SEED = {g.SEED!r}',
          f'B_BOOT = {g.B_BOOT!r}', f'FTSE_UPGRADE = {g.FTSE_UPGRADE!r}', f'SUBPERIODS = {g.SUBPERIODS!r}',
          f'DAYS = {g.DAYS!r}', f'ACF_CASES = {g.ACF_CASES!r}']
DATAF = [g.returns, g.log_price]
ACF = [g.acf, g.acf_fft, g.robust_var_acf, g.portmanteau, g.runs_test]
UR = [g.adf_test, g.pp_test, g.kpss_test]
VR = [g.variance_ratio, g.chow_denning, g.qs_kernel, g.auto_vr_stat, g.auto_vr]
NODATA = 'Simulated data only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch7_random_walk',
         desc='Is today\'s return predictable from yesterday\'s? Scatter plots of r(t) against r(t-1) for the S&P 500 '
              '(since 1990) and the BET (since 1997) with the least-squares line and the first-order autocorrelation; '
              'the S&P 500 log price against four random walks with i.i.d. Normal steps of the same mean and standard '
              'deviation (the RW1 hypothesis of Campbell, Lo and MacKinlay, 1997).',
         keywords='random walk, RW1, efficient market hypothesis, autocorrelation, predictability, S&P 500, BET, simulation',
         consts=CONSTS, funcs=DATAF + [g.acf, g.fig_scatter_lag, g.fig_rw_paths],
         run="print(fig_scatter_lag())\nprint(fig_rw_paths())",
         charts=['sfm_ch7_scatter_lag', 'sfm_ch7_rw_paths']),
    dict(name='SFM_ch7_white_noise_acf',
         desc='Gaussian white noise (n = 1000, as in SFEtimewn) and its sample autocorrelation function with the 95% band '
              '+/- 1.96/sqrt(n); the theoretical ACF and the sample ACF of a simulated path (n = 1000) of the AR(1) '
              'processes with alpha = 0.9 and -0.9 (SFEacfar1), the AR(2) process with alpha1 = 0.5 and alpha2 = 0.4 '
              '(SFEacfar2), the MA(1) processes with beta = 0.5 and -0.5 (SFEacfma1) and the MA(2) process with '
              'beta1 = 0.5 and beta2 = 0.4 (SFEacfma2).',
         keywords='white noise, autocorrelation function, ACF, AR(1), AR(2), MA(1), MA(2), stationary process, simulation',
         consts=CONSTS, funcs=[g.acf, g.fig_white_noise, g.arma_acf, g.simulate_arma, g.fig_acf_processes],
         run="print(fig_white_noise())\nprint(fig_acf_processes())",
         charts=['sfm_ch7_white_noise', 'sfm_ch7_acf_processes'], data=NODATA),
    dict(name='SFM_ch7_autocorrelation_tests',
         desc='Autocorrelation tests of the random walk hypothesis on daily log returns of the S&P 500, DAX, BET, Bitcoin '
              'and four BVB stocks: the sample ACF with the i.i.d. and the heteroskedasticity-robust 95% bands, the ACF '
              'of absolute returns (volatility clustering), the Box-Pierce (1970) and Ljung-Box (1978) Q(10) statistics, '
              'the robust portmanteau statistic with the variance of Lo and MacKinlay (1988), and the runs test of Wald '
              'and Wolfowitz (1940).',
         keywords='autocorrelation, Ljung-Box, Box-Pierce, portmanteau test, robust test, runs test, random walk, '
                  'volatility clustering, S&P 500, DAX, BET, Bitcoin, BVB',
         consts=CONSTS, funcs=DATAF + ACF + [g.fig_acf_returns, g.fig_acf_abs, g.tests_table],
         run="fig_acf_returns()\nprint(fig_acf_abs())\ntt = tests_table()\n"
             "print(pd.DataFrame({NAME[k]: {'rho1': v['ret']['rho1'], 'LB(10)': v['ret']['lb'], 'p LB': v['ret']['p_lb'], "
             "'robust Q(10)': v['ret']['rob'], 'p robust': v['ret']['p_rob'], 'runs z': v['runs']['z'], 'p runs': v['runs']['p']} "
             "for k, v in tt.items()}).T.round(3))",
         charts=['sfm_ch7_acf_returns', 'sfm_ch7_acf_abs'], extra=['ch7_tests_table.csv']),
    dict(name='SFM_ch7_unit_roots',
         desc='Unit-root tests on log prices (constant and trend) and on daily returns (constant) of the S&P 500, DAX, BET, '
              'Bitcoin and four BVB stocks: the augmented Dickey-Fuller test (lag order by AIC), the Phillips-Perron (1988) '
              'Z_t test with a Newey-West long-run variance, and the KPSS (1992) stationarity test; the BET log price and '
              'returns; the spurious regression of two independent random walks (Granger and Newbold, 1974).',
         keywords='unit root, augmented Dickey-Fuller, Phillips-Perron, KPSS, stationarity, integrated process, '
                  'spurious regression, random walk, BET',
         consts=CONSTS, funcs=DATAF + UR + [g.unit_root_table, g.fig_unit_root, g.spurious, g.fig_spurious],
         run="ur = unit_root_table()\n"
             "print(pd.DataFrame({NAME[k]: {f'{s} {t}': v[s][t]['stat'] for s in ['price', 'ret'] for t in ['adf', 'pp', 'kpss']} for k, v in ur.items()}).T.round(2))\n"
             "print(fig_unit_root('bet'))\nsp = spurious()\nprint({lab: (sp[lab]['reject'], sp[lab]['r2_med']) for lab in ['level', 'diff']})\nfig_spurious(sp)",
         charts=['sfm_ch7_unit_root_bet', 'sfm_ch7_spurious']),
    dict(name='SFM_ch7_variance_ratio',
         desc='Variance-ratio tests of the random walk hypothesis: the Lo and MacKinlay (1988) VR(q) with the homoskedastic '
              'Z(q) and the heteroskedasticity-robust Z*(q) for q = 2, 5, 10, 20; the Chow and Denning (1993) multiple test; '
              'the automatic VR test of Choi (1999) with the wild-bootstrap p-value of Kim (2009); VR profiles for q = 2 to '
              '40 (S&P 500, DAX, BET, Bitcoin); a Monte Carlo study of the size of Z and Z* under i.i.d. and GARCH(1,1) returns.',
         keywords='variance ratio, Lo-MacKinlay, heteroskedasticity-robust test, Chow-Denning, automatic variance ratio, '
                  'wild bootstrap, random walk, Monte Carlo, size of a test',
         consts=CONSTS, funcs=DATAF + [g.acf] + ACF[1:2] + VR + [g.vr_table, g.vr_profile, g.fig_vr_profile, g.simulate_garch,
                                                                 g.vr_size, g.fig_vr_size],
         run="vt = vr_table()\n"
             "print(pd.DataFrame({NAME[k]: {**{f'VR({q})': v['vr'][q]['vr'] for q in QS}, **{f'Z*({q})': v['vr'][q]['zs'] for q in QS}, "
             "'CD': v['cd']['cd'], 'p CD': v['cd']['p'], 'AVR': v['avr']['stat'], 'p AVR': v['avr']['p_boot']} for k, v in vt.items()}).T.round(3))\n"
             "print(fig_vr_profile())\nsz = vr_size()\nprint(sz)\nfig_vr_size(sz)",
         charts=['sfm_ch7_vr_profile', 'sfm_ch7_vr_size'], extra=['ch7_vr_table.csv']),
    dict(name='SFM_ch7_markets_efficiency',
         desc='VR(5) with the robust 95% interval, the first-order autocorrelation and the Chow-Denning p-value over the '
              'last ten years of data (from 19 September 2016) for developed markets (S&P 500, DAX, Euro Stoxx 50, Nikkei '
              '225), emerging markets (BET, WIG20, BUX, PX, BIST 100, Bovespa, IPC Mexico, Nifty 50, KOSPI, Shanghai '
              'Composite) and crypto assets (Bitcoin, Ethereum).',
         keywords='market efficiency, emerging markets, variance ratio, cross-market comparison, BET, Bitcoin',
         consts=CONSTS, funcs=DATAF + [g.acf, g.variance_ratio, g.chow_denning, g.markets_table, g.fig_vr_markets],
         run="mt = markets_table()\nprint(pd.DataFrame(mt).T.round(3))\nfig_vr_markets(mt)",
         charts=['sfm_ch7_vr_markets']),
    dict(name='SFM_ch7_adaptive_markets',
         desc='Efficiency over time (the adaptive markets hypothesis of Lo, 2004): VR(5) and the robust Z*(5) on rolling '
              'windows of 500 trading days for the S&P 500, the BET and Bitcoin; three sub-periods for each; the BET in the '
              'five years before and after its upgrade to Secondary Emerging market by FTSE Russell (September 2020).',
         keywords='adaptive markets hypothesis, rolling window, time-varying efficiency, variance ratio, BET, Bitcoin, '
                  'S&P 500, emerging market upgrade',
         consts=CONSTS, funcs=DATAF + [g.acf, g.variance_ratio, g.chow_denning, g.rolling_vr, g.fig_rolling_vr, g.subperiods,
                                       g.bet_upgrade],
         run="print(fig_rolling_vr())\nprint(subperiods())\nprint(bet_upgrade())",
         charts=['sfm_ch7_rolling_vr']),
    dict(name='SFM_ch7_calendar_anomalies',
         desc='Calendar anomalies on the S&P 500 (since 1990) and the BET (since 1997): the mean daily return by weekday '
              'with 95% intervals and a Wald test of equal means, and January against the other months, both with '
              'Newey-West (HAC) standard errors.',
         keywords='calendar anomalies, day-of-week effect, weekend effect, January effect, HAC standard errors, S&P 500, BET',
         consts=CONSTS, funcs=DATAF + [g.calendar, g.fig_calendar],
         run="print(fig_calendar())",
         charts=['sfm_ch7_calendar']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar7 as s
    QUANTLETS.append(dict(
        name='SFM_ch7_seminar',
        desc='Seminar 7 of Statistics of Financial Markets: variance ratios from autocorrelations, Box-Pierce and Ljung-Box '
             'statistics, the runs test and Dickey-Fuller decisions on paper; the ACF of S&P 500 returns and squared '
             'returns with Ljung-Box and robust portmanteau tests; ADF, Phillips-Perron and KPSS tests on the DAX log '
             'price and returns; the Lo-MacKinlay variance-ratio test and the Chow-Denning test on the S&P 500.',
        keywords='seminar, random walk, autocorrelation, Ljung-Box, runs test, unit root, ADF, KPSS, variance ratio, '
                 'Chow-Denning',
        consts=CONSTS, funcs=DATAF + ACF + UR + VR + [g.vr_profile, s.vr_from_acf, s.rw_horizon, s.portmanteau_from_acf,
                                                     s.runs_from_signs, s.adf_decision, s.b1_acf, s.b3_unit_root, s.b5_vr],
        run="print(vr_from_acf([0.10, 0.05], 5, 2500))\nprint(portmanteau_from_acf([0.12, -0.05, 0.04, 0.03, -0.06], 500))\n"
            "print(runs_from_signs('++-+++--+---++-+--++'))\nprint(adf_decision(-0.0028, 0.0019, 'c', 2.10))\n"
            "print(b1_acf('sp500'))\nprint(b3_unit_root('dax'))\nprint(b5_vr('sp500'))",
        charts=['ch7_sem_b1', 'ch7_sem_b3', 'ch7_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 7, 'efficient markets, random walk and variance-ratio tests', HERE, data=DATA, submitted=SUBMITTED)
