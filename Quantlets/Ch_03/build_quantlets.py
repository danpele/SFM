"""
build_quantlets.py -- Quantlet folders of Chapter 3 (SFM): alpha-stable distributions
====================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The older folders of Quantlets/Ch_03 (SFM_ch3_emh_tests, SFM_ch3_variance_ratio, ...) belong to the efficient
markets chapter and are kept unchanged.
Run:  python3 Quantlets/Ch_03/generate_all_charts.py && python3 Quantlets/Ch_03/seminar3.py
      python3 Quantlets/Ch_03/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 3
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
DATA = ('Daily market data from EODHD (BET, S&P 500 and DAX since 2000, Bitcoin since 2014), data/market of the SFM '
        'repository')
CONSTS = [f'START = {g.START!r}', 'ASSETS = ' + repr(g.ASSETS), 'COLORS = ' + repr(g.COLORS),
          'MODEL_COL = ' + repr(g.MODEL_COL), f'SEED = {g.SEED!r}']
LAW = [g.stable_cf, g.s1_to_s0, g.s0_to_s1, g.StableGrid, g.stable_pdf]
SIM = [g.rstable]
EST = [g.mcculloch, g.mcculloch_quantiles, g.stable_nll, g.stable_mle]
FIT = [g.fit_models, g.model_cdf, g.model_ppf, g.returns, g.fit_all]
NODATA = 'Simulated data only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch3_densities',
         desc='Densities of alpha-stable laws computed by FFT inversion of the characteristic function (Nolan S0 and S1 '
              'parameterisations): the effect of alpha (tails and peak, linear and log scale), of beta (skewness) and of '
              'gamma (scale); S1 vs S0 near alpha = 1 with beta = 0.8; the three closed-form cases Normal, Cauchy and Levy.',
         keywords='stable distribution, characteristic function, parameterisation S0, parameterisation S1, Cauchy, Levy, '
                  'Normal distribution, fast Fourier transform',
         consts=CONSTS, funcs=LAW + [g.fig_density_alpha, g.fig_density_beta, g.fig_s0_s1, g.fig_special_cases],
         run="print(fig_density_alpha())\nfig_density_beta()\nprint(fig_s0_s1())\nfig_special_cases()",
         charts=['sfm_ch3_density_alpha', 'sfm_ch3_density_beta', 'sfm_ch3_s0_s1', 'sfm_ch3_special_cases'], data=NODATA),
    dict(name='SFM_ch3_stability_gclt',
         desc='Stability under summation and the generalised central limit theorem: QQ plot of sums of 10 i.i.d. '
              'S(1.7, 0, 1, 0) draws rescaled by 10^(1/alpha) and by sqrt(10) against single draws; normalised sums of '
              'Student-t variables with 1.5 degrees of freedom (infinite variance) approaching the stable limit '
              'S(1.5, 0, gamma, 0), against a Normal law with the same interquartile range.',
         keywords='stability, generalised central limit theorem, domain of attraction, power-law tails, Student-t, '
                  'simulation',
         consts=CONSTS, funcs=LAW + SIM + [g.fig_stability_qq, g.fig_gclt],
         run="print(fig_stability_qq())\nprint(fig_gclt())",
         charts=['sfm_ch3_stability_qq', 'sfm_ch3_gclt'], data=NODATA),
    dict(name='SFM_ch3_tails_moments',
         desc='Tails and moments of alpha-stable laws: P(X > x) on log-log axes with the power-law approximation '
              'c_alpha x^(-alpha); P(|X| > k) for the stable law with alpha = 1.7 against the Normal law with the same '
              'scale; the running sample variance of 20,000 Normal and stable (alpha = 1.7) draws.',
         keywords='power-law tails, tail index, infinite variance, moments, sample variance, stable distribution',
         consts=CONSTS, funcs=LAW + SIM + [g.fig_tails_loglog, g.tail_table, g.fig_running_variance],
         run="print(fig_tails_loglog())\nprint(tail_table())\nprint(fig_running_variance())",
         charts=['sfm_ch3_tails_loglog', 'sfm_ch3_running_var_sim'], data=NODATA),
    dict(name='SFM_ch3_sim_stable',
         desc='Simulation of alpha-stable random variables by the Chambers-Mallows-Stuck (1976) method with the '
              'correction of Weron (1996), in the S1 and S0 parameterisations: 100,000 draws of S(1.5, 0.5, 1, 0; 1) '
              'against the exact density and a two-sample Kolmogorov-Smirnov test against scipy.stats.levy_stable.rvs. '
              'Based on the Quantlet sim_stable.',
         keywords='simulation, Chambers-Mallows-Stuck, stable distribution, random number generation, Kolmogorov-Smirnov test',
         consts=CONSTS, funcs=LAW + SIM + [g.fig_cms],
         run="print(fig_cms())",
         charts=['sfm_ch3_cms'], data=NODATA),
    dict(name='SFM_ch3_estimation',
         desc='Estimation of alpha-stable parameters: the quantile method of McCulloch (1986) with Tables III, IV, V and '
              'VII, and maximum likelihood in the S0 parameterisation with an FFT density and standard errors from the '
              'numerical Hessian; McCulloch\'s map from nu_alpha to alpha with the daily log returns of the BET, '
              'S&P 500, DAX and Bitcoin; a Monte Carlo comparison of both estimators (200 samples of S(1.7, 0, 1, 0), '
              'n = 500 and 2500).',
         keywords='McCulloch quantile estimator, maximum likelihood, stable distribution, Monte Carlo, estimation, '
                  'BET, S&P 500, DAX, Bitcoin',
         consts=CONSTS, funcs=LAW + SIM + EST + [g.returns, g.fig_mcculloch_map, g.estimator_mc, g.fig_estimator_mc],
         run="real = {}\nfor k in ASSETS:\n    m = mcculloch(returns(k).values)\n    real[k] = (m['nu_alpha'], m['alpha'])\n"
             "    print(LABELS[k], {c: round(m[c], 3) for c in ['nu_alpha', 'nu_beta', 'alpha', 'beta', 'gamma', 'delta0']})\n"
             "fig_mcculloch_map(real)\nprint(fig_estimator_mc(estimator_mc()))",
         charts=['sfm_ch3_mcculloch_map', 'sfm_ch3_estimators_mc']),
    dict(name='SFM_ch3_fit_returns',
         desc='Stable (ML, S0), Student-t and Normal fits of the daily log returns of the BET, S&P 500 and DAX since 2000 '
              'and of Bitcoin since 2014: parameter table with standard errors and AIC, fitted densities on a log scale, '
              'QQ plots, the left tail on log-log axes, the observed and expected numbers of daily losses above 3-20%, '
              'VaR 1% and VaR 0.1% of the data and of each model.',
         keywords='stable distribution, Student-t, maximum likelihood, QQ plot, tail probability, VaR, AIC, BET, S&P 500, '
                  'DAX, Bitcoin',
         consts=CONSTS, funcs=LAW + EST + FIT + [g.fits_table, g.fig_fit_density, g.fig_qq_real, g.fig_tails_real, g.tail_counts],
         run="fits = fit_all()\nprint(fits_table(fits).round(3).T)\nfig_fit_density(fits)\nfig_qq_real(fits)\nfig_tails_real(fits)\n"
             "print(pd.DataFrame(tail_counts(fits)).round(2))",
         charts=['sfm_ch3_fit_density', 'sfm_ch3_qq_real', 'sfm_ch3_tails_real'], extra=['ch3_fits.csv']),
    dict(name='SFM_ch3_aggregation',
         desc='The critique of the stable model: ML estimates of alpha (95% intervals) for daily, weekly and monthly log '
              'returns of the BET, S&P 500, DAX and Bitcoin (a stable i.i.d. series would keep the same alpha), and the '
              'running sample variance of the S&P 500 against paths simulated from its fitted stable law.',
         keywords='aggregation, temporal scaling, finite variance, stable distribution, sample variance, critique',
         consts=CONSTS, funcs=LAW + SIM + EST + FIT + [g.aggregate, g.aggregation_alpha, g.fig_aggregation, g.fig_running_var_real],
         run="agg = aggregation_alpha()\nprint(pd.DataFrame({(k, h): v for k, a in agg.items() for h, v in a.items()}).T.round(3))\n"
             "fig_aggregation(agg)\nfits = fit_all(['sp500'])\nprint(fig_running_var_real(fits))",
         charts=['sfm_ch3_aggregation', 'sfm_ch3_running_var_real']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar3 as s
    QUANTLETS.append(dict(
        name='SFM_ch3_seminar',
        desc='Seminar 3 of Statistics of Financial Markets: stability constants, characteristic functions in S1 and S0, '
             'tail frequencies, McCulloch estimates from five quantiles and one Chambers-Mallows-Stuck draw on paper; '
             'McCulloch estimates with bootstrap intervals for the BET, maximum likelihood for the S&P 500 with a QQ plot '
             'and AIC, and the McCulloch alpha of daily, weekly and monthly S&P 500 returns.',
        keywords='seminar, stable distribution, McCulloch estimator, bootstrap, maximum likelihood, QQ plot, aggregation',
        consts=CONSTS + [f'SEM_START = {s.SEM_START!r}'],
        funcs=LAW + SIM + EST + FIT + [g.aggregate, s.sem_returns, s.stability_constant, s.cf_value, s.cms_draw,
                                       s.bootstrap_mcculloch, s.b1_mcculloch, s.b3_ml, s.b5_aggregation],
        run="print({a: round(stability_constant(3, 4, a), 2) for a in (2.0, 1.5, 1.0)})\n"
            "print(mcculloch_quantiles(-2.10, -0.55, 0.05, 0.62, 2.06))\nprint(cms_draw(0.8, 1.2, 1.5))\n"
            "print(b1_mcculloch('bet'))\nprint(b3_ml('sp500'))\nprint(b5_aggregation('sp500'))",
        charts=['ch3_sem_b1', 'ch3_sem_b3', 'ch3_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 3, 'alpha-stable distributions', HERE, data=DATA, submitted=SUBMITTED)
