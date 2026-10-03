"""
build_fmh_quantlets.py -- Quantlet folders of Chapter 11 (SFM): fractal markets hypothesis and long memory
=========================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Only the SFM_ch11_* folders listed below are written; the older VaR Quantlets in Quantlets/Ch_11 are not touched.
Run:  python3 Quantlets/Ch_11/generate_fmh_charts.py && python3 Quantlets/Ch_11/seminar11.py
      python3 Quantlets/Ch_11/build_fmh_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 11
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_fmh_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Saturday, 3 October 2026'
DATA = ('Daily market data from EODHD (S&P 500, DAX and BET since 2000, Bitcoin since 2014, Banca Transilvania and '
        'OMV Petrom since 2010), data/market of the SFM repository')
CONSTS = ['from scipy import special',
          f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}', f'BAD_DAYS = {g.BAD_DAYS!r}',
          f'COLORS = {g.COLORS!r}', f'SEED = {g.SEED!r}', f'NMIN = {g.NMIN!r}', f'WINDOW = {g.WINDOW!r}', f'STEP = {g.STEP!r}',
          f'MC_REPS = {g.MC_REPS!r}', f'LO_CRIT = {g.LO_CRIT!r}', f'GPH_POWER = {g.GPH_POWER!r}', f'RS_EXAMPLE = {g.RS_EXAMPLE!r}',
          f'NILE_BREAK = {g.NILE_BREAK!r}', f'H_LIST = {g.H_LIST!r}', f'MC_N = {g.MC_N!r}', f'PHIS = {g.PHIS!r}',
          f'HS_POWER = {g.HS_POWER!r}', f'DELTAS = {g.DELTAS!r}', f'PERS = {g.PERS!r}']
EST = [g.returns, g.block_sizes, g.rs_block, g.rs_curve, g.hurst_rs, g.dfa_curve, g.hurst_dfa, g.periodogram, g.gph,
       g.lo_test, g.acf, g.all_estimates]
SIM = [g.fgn_acvf, g.arfima_acf, g.arfima_var, g.circulant_gaussian, g.sim_fgn, g.sim_arfima]
NODATA = 'Simulated data only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch11_self_similarity',
         desc='Self-similarity of returns: the standard deviation of non-overlapping h-day log returns (h = 1 to 250 days) '
              'against h on log scales for the S&P 500 and the BET (since 2000) and Bitcoin (since 2014); the slope estimates '
              'the scaling (Hurst) exponent H; the square-root-of-time rule corresponds to H = 0.5.',
         keywords='self-similarity, scaling, Hurst exponent, square-root-of-time rule, aggregation, S&P 500, BET, Bitcoin',
         consts=CONSTS, funcs=[g.returns, g.scaling, g.fig_scaling], run="print(fig_scaling())",
         charts=['sfm_ch11_scaling']),
    dict(name='SFM_ch11_rs_estimators',
         desc='Estimators of long memory (Franke, Haerdle and Hafner, 2019, Ch. 14): the R/S statistic of eight returns step '
              'by step; classical R/S (Hurst 1951; Mandelbrot and Wallis 1969), DFA (Peng et al. 1994), the GPH log-periodogram '
              'regression (Geweke and Porter-Hudak 1983) and the modified R/S test of Lo (1991) for daily S&P 500 returns and '
              'absolute returns since 2000, with the mean curve of i.i.d. Normal series.',
         keywords='long memory, Hurst exponent, rescaled range, R/S analysis, modified R/S, Lo test, detrended fluctuation '
                  'analysis, DFA, GPH estimator, periodogram, S&P 500',
         consts=CONSTS, funcs=EST + [g.rs_steps, g.fig_rs_example, g.fig_rs_dfa, g.fig_gph],
         run="print(fig_rs_example())\nprint(fig_rs_dfa())\nprint(fig_gph())\nr = returns('sp500')\n"
             "print(pd.DataFrame({'returns': all_estimates(r.values), '|returns|': all_estimates(np.abs(r.values))}).T)",
         charts=['sfm_ch11_rs_example', 'sfm_ch11_rs_dfa', 'sfm_ch11_gph']),
    dict(name='SFM_ch11_fbm_fgn',
         desc='Fractional Brownian motion and fractional Gaussian noise, simulated exactly by circulant embedding '
              '(Davies-Harte): fBm paths for H = 0.3, 0.5, 0.7 with the same random numbers (as in SFEfbmplot); sample and '
              'theoretical autocorrelations of fGn for H = 0.2 and 0.8 (as in SFEfgnacf); hyperbolic decay of fGn against '
              'the exponential decay of an AR(1) with the same lag-1 autocorrelation.',
         keywords='fractional Brownian motion, fractional Gaussian noise, Hurst exponent, long memory, autocorrelation, '
                  'self-similarity, circulant embedding, simulation',
         consts=CONSTS, funcs=[g.acf] + SIM + [g.fig_fbm_paths, g.fig_fgn_acf],
         run="print(fig_fbm_paths())\nprint(fig_fgn_acf())", charts=['sfm_ch11_fbm_paths', 'sfm_ch11_fgn_acf'], data=NODATA),
    dict(name='SFM_ch11_arfima',
         desc='ARFIMA(0,d,0) processes (Granger and Joyeux 1980; Hosking 1981) with Gaussian white noise N(0, 1): simulated '
              'paths of 1000 values for d = 0.4 and d = -0.4 (as in SFEarfima), exact simulation by circulant embedding; sample '
              'autocorrelations of 20000 values against the theoretical ones.',
         keywords='ARFIMA, fractional differencing, fractional integration, long memory, anti-persistence, autocorrelation, simulation',
         consts=CONSTS, funcs=[g.acf] + SIM + [g.fig_arfima], run="print(fig_arfima())", charts=['sfm_ch11_arfima'], data=NODATA),
    dict(name='SFM_ch11_monte_carlo',
         desc='Monte Carlo study of the long-memory estimators: mean and 95% band of R/S, DFA and GPH estimates for i.i.d. '
              'Normal series of length 250 to 5000 (as in Weron 2002); rejection rates of the classical and of the modified '
              'R/S test (Lo 1991) under AR(1) short memory and under fractional Gaussian noise.',
         keywords='Monte Carlo, small-sample bias, confidence band, Hurst exponent, R/S, DFA, GPH, Lo test, size, power',
         consts=CONSTS, funcs=EST + SIM + [g.mc_null, g.fig_mc_bias, g.lo_rejections],
         run="print(fig_mc_bias())\nprint(lo_rejections())", charts=['sfm_ch11_mc_bias', 'sfm_ch11_lo_test'], data=NODATA),
    dict(name='SFM_ch11_spurious',
         desc='Spurious long memory: the annual flow of the Nile at Aswan, 1871-1970 (Cobb 1978; statsmodels dataset), with '
              'its change point after 1898, R/S of the raw and of the break-adjusted series with an i.i.d. Monte Carlo band; '
              'simulated i.i.d. series with a mean shift and absolute returns of GARCH(1,1) models (short memory) for several '
              'persistences, with their R/S, DFA and GPH estimates.',
         keywords='spurious long memory, structural break, change point, Nile, regime switching, volatility clustering, '
                  'GARCH, Hurst exponent',
         consts=CONSTS, funcs=EST + [g.sim_garch, g.nile, g.fig_nile, g.spurious],
         run="print(fig_nile())\nprint(spurious())", charts=['sfm_ch11_nile', 'sfm_ch11_spurious'],
         data='Nile flow at Aswan 1871-1970 (statsmodels dataset, Cobb 1978); simulated data'),
    dict(name='SFM_ch11_volatility_memory',
         desc='Long memory in returns and in volatility for the S&P 500, DAX, BET, Bitcoin, Banca Transilvania and OMV Petrom: '
              'the ACF of r, |r| and r^2 up to lag 250 (Ding, Granger and Engle 1993); R/S, DFA, GPH and Lo\'s test for r and '
              '|r| with i.i.d. Monte Carlo bands of the same length; the DFA exponent of |r| after a random shuffle.',
         keywords='long memory, volatility, absolute returns, squared returns, shuffle test, Hurst exponent, GPH, DFA, '
                  'S&P 500, DAX, BET, Bitcoin, BVB',
         consts=CONSTS, funcs=EST + [g.mc_null, g.fig_vol_acf, g.markets, g.fig_markets],
         run="print(fig_vol_acf())\nM = markets()\nfig_markets(M)\n"
             "print(pd.DataFrame({NAME[k]: {'DFA r': v['r']['dfa'], 'GPH d r': v['r']['gph_d'], 'DFA |r|': v['abs']['dfa'], "
             "'GPH d |r|': v['abs']['gph_d'], 'Lo V |r|': v['abs']['lo_V']} for k, v in M.items()}).T.round(3))",
         charts=['sfm_ch11_vol_acf', 'sfm_ch11_markets'], extra=['ch11_memory_table.csv']),
    dict(name='SFM_ch11_rolling_hurst',
         desc='Rolling Hurst exponents: DFA exponents of daily returns and absolute returns on windows of 1000 observations '
              'moved by 21 observations, dated at the last day of each window, for the S&P 500, the BET and Bitcoin, with '
              'the 95% band of i.i.d. Normal series of length 1000 (as in Cajueiro and Tabak 2004; Bariviera 2017).',
         keywords='rolling window, Hurst exponent, DFA, time-varying efficiency, adaptive markets, S&P 500, BET, Bitcoin',
         consts=CONSTS, funcs=EST + [g.mc_null, g.rolling_hurst, g.fig_rolling], run="print(fig_rolling())",
         charts=['sfm_ch11_rolling']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar11 as s
    QUANTLETS.append(dict(
        name='SFM_ch11_seminar',
        desc='Seminar 11 of Statistics of Financial Markets: R/S step by step and the Hurst exponent from an R/S plot, '
             'fractional Gaussian noise and 10-day VaR 1%, ARFIMA(0,d,0) weights and autocorrelations on paper; R/S, DFA, '
             'GPH and Lo\'s test with Monte Carlo bands for S&P 500 returns; long memory in S&P 500 volatility with a shuffle '
             'test; rolling DFA exponents of the S&P 500.',
        keywords='seminar, Hurst exponent, R/S, DFA, GPH, Lo test, Monte Carlo band, ARFIMA, fractional Gaussian noise, '
                 'volatility, rolling window',
        consts=CONSTS + [f'SEM_REPS = {s.SEM_REPS!r}', f'A1_X = {s.A1_X!r}', f'A2_X = {s.A2_X!r}', f'POINTS = {s.POINTS!r}',
                         f'CRISES = {s.CRISES!r}'],
        funcs=EST + SIM + [g.mc_null, g.rolling_hurst, g.rs_steps, s.rs_by_hand, s.hurst_from_points, s.rs_points,
                           s.fgn_numbers, s.arfima_numbers, s.b1_memory, s.b3_volatility, s.b5_rolling],
        run="print(rs_by_hand(A1_X))\nprint(hurst_from_points(POINTS, rs_points('sp500')))\nprint(fgn_numbers(0.7))\n"
            "print(arfima_numbers(0.3))\nprint(b1_memory('sp500'))\nprint(b3_volatility('sp500'))\nprint(b5_rolling('sp500'))",
        charts=['ch11_sem_b1', 'ch11_sem_b3', 'ch11_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 11, 'Fractal markets hypothesis and long memory', HERE, data=DATA, submitted=SUBMITTED)
