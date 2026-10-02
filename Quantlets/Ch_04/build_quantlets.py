"""
build_quantlets.py -- Quantlet folders of Chapter 4 (SFM): Probability for finance
=================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The older folders of Quantlets/Ch_04 (efficient markets: SFM_ch4_acf_sp500, SFM_ch4_vr_profile, ...) belong to the
efficient-markets chapter and are kept unchanged.
Run:  python3 Quantlets/Ch_04/generate_all_charts.py && python3 Quantlets/Ch_04/seminar4.py
      python3 Quantlets/Ch_04/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 4
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
DATA = ('Daily market data from EODHD (S&P 500, DAX, BET, Bitcoin), data/market of the SFM repository; simulated '
        'random numbers (NumPy PCG64 and the RANDU generator)')
CONSTS = [f'START = {g.START!r}', 'ASSETS = ' + repr(g.ASSETS), 'COLORS = ' + repr(g.COLORS), 'SHORT = ' + repr(g.SHORT),
          'PAL = ' + repr(g.PAL), f'RANDU_A, RANDU_M = {g.RANDU_A}, {g.RANDU_M}']
CORE = [g.returns, g.joint_returns, g.acf, g.max_drawdown, g.gbm_params, g.simulate_gbm, g.lcg, g.lcg_period, g.randu,
        g.binomial_crr, g.ar1]

QUANTLETS = [
    dict(name='SFM_ch4_random_variables',
         desc='Random variables and distributions on real data, 2000-2026: the number of up days in a five-day week of '
              'the S&P 500 (a discrete random variable) against the binomial distribution, the histogram of daily log '
              'returns (a continuous random variable) against the Normal density, the empirical CDF against the Normal '
              'CDF with the left tail on a log scale and the 1% quantile (VaR 1%); conditional probabilities of down days.',
         keywords='random variable, probability mass function, density, cumulative distribution function, quantile, '
                  'VaR 1%, binomial distribution, conditional probability',
         consts=CONSTS, funcs=CORE + [g.down_days, g.up_days_per_week, g.fig_random_variables, g.fig_cdf_quantile],
         run="print(down_days())\nprint(fig_random_variables())\nprint(fig_cdf_quantile())",
         charts=['sfm_ch4_random_variables', 'sfm_ch4_cdf_quantile']),
    dict(name='SFM_ch4_dependence',
         desc='Joint distributions and dependence: scatter plots of daily returns (S&P 500 vs DAX and vs BET, common '
              'trading days), covariance, correlation and the volatility of a 50/50 portfolio; uncorrelated but dependent '
              'variables (X and X^2; returns and squared returns of four markets); conditional mean and conditional '
              'standard deviation of today\'s return given the quintile of yesterday\'s absolute return, with the law of '
              'total variance.',
         keywords='joint distribution, covariance, correlation, portfolio variance, independence, conditional '
                  'expectation, conditional variance, law of total variance, volatility clustering',
         consts=CONSTS, funcs=CORE + [g.pair_stats, g.fig_joint, g.lag_dependence, g.fig_uncorrelated_dependent,
                                      g.conditional_moments, g.fig_conditional],
         run="print(fig_joint())\nprint(fig_uncorrelated_dependent())\n"
             "res = fig_conditional()\nprint({k: {'sd': v['sd'].round(3), 'ratio': round(v['ratio'], 2)} for k, v in res.items()})",
         charts=['sfm_ch4_joint', 'sfm_ch4_uncorrelated_dependent', 'sfm_ch4_conditional']),
    dict(name='SFM_ch4_random_numbers',
         desc='Random number generation: linear congruential generators x_{k+1} = (a x_k + c) mod M and their period, '
              'the RANDU generator whose triples lie on 15 planes (9u_k - 6u_{k+1} + u_{k+2} is an integer) against '
              'NumPy PCG64, and the inverse transform method X = F^{-1}(U) for the exponential, the Normal and the '
              'empirical distribution of S&P 500 returns (historical simulation). Based on the SFE Quantlets SFErandu, '
              'SFErangen1 and SFErangen2.',
         keywords='pseudo-random numbers, linear congruential generator, RANDU, PCG64, inverse transform method, '
                  'exponential distribution, historical simulation',
         consts=CONSTS, funcs=CORE + [g.fig_randu, g.fig_inverse_transform],
         run="print(fig_randu())\nprint(fig_inverse_transform())",
         charts=['sfm_ch4_randu', 'sfm_ch4_inverse_transform']),
    dict(name='SFM_ch4_monte_carlo',
         desc='The law of large numbers and Monte Carlo: the running mean of S&P 500 daily returns against simulated '
              'i.i.d. Normal returns with the band +/- 1.96 sd/sqrt(n); the Monte Carlo estimate of the probability of '
              'a 21-day loss beyond 10% under i.i.d. Normal returns as a function of the number of simulations, with '
              'its standard error sqrt(p(1-p)/N).',
         keywords='law of large numbers, Monte Carlo, standard error, simulation, loss probability',
         consts=CONSTS, funcs=CORE + [g.fig_lln, g.mc_loss_probability, g.fig_monte_carlo],
         run="print(fig_lln())\nprint(fig_monte_carlo())",
         charts=['sfm_ch4_lln', 'sfm_ch4_monte_carlo']),
    dict(name='SFM_ch4_binomial',
         desc='The binomial process (Cox-Ross-Rubinstein) calibrated to the drift and volatility of the S&P 500, '
              '2000-2026: sample paths over one year of daily steps and the distribution of the terminal price against '
              'its lognormal limit. Based on the SFE Quantlet SFEBinomp.',
         keywords='binomial process, binomial tree, Cox-Ross-Rubinstein, lognormal distribution, random walk',
         consts=CONSTS, funcs=CORE + [g.fig_binomial],
         run="print(fig_binomial())",
         charts=['sfm_ch4_binomial']),
    dict(name='SFM_ch4_processes',
         desc='Discrete-time stochastic processes: white noise, AR(1) with phi = 0.5 and 0.95 and the random walk driven '
              'by the same shocks, with simulated variances and autocorrelations against theory; the S&P 500 '
              '(2000-2026) hidden among three GBM paths with the same drift and volatility (prices and daily returns).',
         keywords='stochastic process, white noise, random walk, AR(1), stationarity, martingale, geometric Brownian motion',
         consts=CONSTS, funcs=CORE + [g.fig_processes, g.fig_spot_the_real],
         run="print(fig_processes())\nprint(fig_spot_the_real())",
         charts=['sfm_ch4_processes', 'sfm_ch4_spot_the_real']),
    dict(name='SFM_ch4_wiener_gbm',
         desc='The Wiener process as the limit of scaled random walks, Wiener paths with +/- sqrt(t) bands and the '
              'quadratic variation; geometric Brownian motion calibrated to the S&P 500 with its mean and median; real '
              'S&P 500, BET and Bitcoin paths in the fan of a GBM with their own drift and volatility; what GBM misses '
              '(excess kurtosis, autocorrelation of squared returns, maximum drawdown) against 300 simulations. Based on '
              'the SFE Quantlets SFEWienerProcess and SFEsimGBM.',
         keywords='Wiener process, Brownian motion, quadratic variation, geometric Brownian motion, lognormal, '
                  'simulation, maximum drawdown, stylised facts',
         consts=CONSTS, funcs=CORE + [g.scaled_random_walk, g.fig_wiener, g.fig_gbm_paths, g.fig_gbm_fan, g.gbm_check,
                                      g.fig_gbm_check],
         run="print(fig_wiener())\nprint(fig_gbm_paths())\nprint(fig_gbm_fan())\n"
             "res = fig_gbm_check()\nprint({k: v['real'] for k, v in res.items()})",
         charts=['sfm_ch4_wiener', 'sfm_ch4_gbm_paths', 'sfm_ch4_gbm_fan', 'sfm_ch4_gbm_check']),
    dict(name='SFM_ch4_drawdown_open',
         desc='Open question of Chapter 4: does GBM get drawdown risk right? Maximum drawdowns within each complete '
              'calendar year of the S&P 500, the BET and Bitcoin against the distribution of the one-year maximum '
              'drawdown of a GBM calibrated to each series; shares of years beyond 20% and 40%.',
         keywords='maximum drawdown, geometric Brownian motion, simulation, volatility clustering, tail risk',
         consts=CONSTS, funcs=CORE + [g.yearly_drawdowns, g.gbm_yearly_drawdown, g.fig_drawdown_years],
         run="res = fig_drawdown_years()\nprint({k: {a: b for a, b in v.items()} for k, v in res.items()})",
         charts=['sfm_ch4_drawdown_years']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar4 as s
    QUANTLETS.append(dict(
        name='SFM_ch4_seminar',
        desc='Seminar 4 of Statistics of Financial Markets: moments of a discrete return, portfolio volatility, a '
             'binomial tree and the inverse transform on paper; dependence of S&P 500 returns (lag-1 correlations of '
             'returns and squared returns, conditional moments by quintile of yesterday\'s move), the probability of a '
             '21-day loss beyond 10% by Monte Carlo and bootstrap, and GBM against the BET (the solved tasks).',
        keywords='seminar, random variables, covariance, conditional variance, binomial tree, inverse transform, '
                 'Monte Carlo, bootstrap, geometric Brownian motion',
        consts=CONSTS + [f'SEM_START = {s.SEM_START!r}'],
        funcs=CORE + [g.lag_dependence, g.conditional_moments, g.pair_stats, g.yearly_drawdowns,
                      s.discrete_moments, s.joint_table, s.portfolio_sd, s.mixture_moments, s.binomial_tree,
                      s.ar1_moments, s.inverse_transform_examples, s.gbm_by_hand, s.b1_dependence, s.b3_monte_carlo,
                      s.b5_gbm_check],
        run="print(discrete_moments([-3, 0, 2], [0.2, 0.3, 0.5]))\nprint(portfolio_sd(0.6, 20, 30, 0.3))\n"
            "print(binomial_tree(100, 1.1, 0.9, 0.55, 3))\nprint(inverse_transform_examples())\n"
            "print(b1_dependence())\nprint(b3_monte_carlo())\nprint(b5_gbm_check())",
        charts=['ch4_sem_b1', 'ch4_sem_b3', 'ch4_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 4, 'Probability', HERE, data=DATA, submitted=SUBMITTED)
