"""
build_quantlets.py -- Quantlet folders of Chapter 2 (SFM): Classical distributions and stylised facts
====================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The older Chapter 2 folders (SFM_ch2_normal_distribution, SFM_ch2_student_t, SFM_ch2_stylized_facts, ...) are kept
unchanged.
Run:  python3 Quantlets/Ch_02/generate_all_charts.py && python3 Quantlets/Ch_02/seminar2.py
      python3 Quantlets/Ch_02/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 2
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
DATA = ('Daily market data from EODHD (S&P 500, DAX, BET, Bitcoin, OMV Petrom, BRD, Nuclearelectrica, Romgaz, Banca '
        'Transilvania), data/market of the SFM repository')
CONSTS = [f'START = {g.START!r}', 'ASSETS = ' + repr(g.ASSETS), 'INDICES = ' + repr(g.INDICES),
          'COLORS = ' + repr(g.COLORS), 'PAL = ' + repr(g.PAL), 'SHORT = ' + repr(g.SHORT), 'HORIZONS = ' + repr(g.HORIZONS)]
CORE = [g.returns, g.moments, g.t_fit, g.sigma_days, g.normal_tail_table, g.acf, g.ljung_box, g.leverage, g.aggregate,
        g.qq_points]

QUANTLETS = [
    dict(name='SFM_ch2_normal_lognormal',
         desc='The Normal and lognormal distributions: the standard Normal density with the areas within 1, 2 and 3 '
              'standard deviations, the two-sided tail probabilities P(|Z| > k) for k = 1..6 with the waiting time they '
              'imply in trading days and years, and lognormal densities LN(0, sigma^2) for sigma = 0.25, 0.5, 1 with '
              'their means, medians and modes. Based on the SFE Quantlet SFElognormal.',
         keywords='Normal distribution, lognormal distribution, tail probability, quantile, mean, median, mode, '
                  'geometric random walk',
         consts=CONSTS, funcs=[g.normal_tail_table, g.fig_normal_pdf, g.fig_lognormal],
         run="print(pd.DataFrame(fig_normal_pdf()).T)\nprint(pd.DataFrame(fig_lognormal()).T)",
         charts=['sfm_ch2_normal_pdf', 'sfm_ch2_lognormal']),
    dict(name='SFM_ch2_clt_normal_approx',
         desc='The central limit theorem: standardised means of n Bernoulli(0.2) draws (n = 5, 30, 300; 20000 '
              'simulations) against the standard Normal density, and the Normal approximation of the binomial '
              'distribution B(n, p) for four (n, p), with the largest error of the cumulative probabilities under the '
              'continuity correction. Based on the SFE Quantlets SFEclt and SFENormalApprox1-4.',
         keywords='central limit theorem, Bernoulli distribution, binomial distribution, Normal approximation, '
                  'continuity correction, simulation',
         consts=CONSTS, funcs=[g.fig_clt, g.fig_normal_approx],
         run="print(fig_clt())\nprint(fig_normal_approx())",
         charts=['sfm_ch2_clt', 'sfm_ch2_normal_approx']),
    dict(name='SFM_ch2_moments_jarque_bera',
         desc='Moments of daily log returns, 2010-2026 (S&P 500, DAX, BET, Bitcoin, OMV Petrom, BRD, Nuclearelectrica, '
              'Romgaz): mean, standard deviation, skewness, excess kurtosis with their standard errors under normality, '
              'the Jarque-Bera test, extremes with dates; densities with different skewness and kurtosis; the kernel '
              'density of DAX returns against the Normal density on a linear and a log scale; the effect of two '
              'misplaced adjustment days of Banca Transilvania (May 2016) on the kurtosis. Based on the SFE Quantlet '
              'SFEDaxReturnDistribution.',
         keywords='skewness, kurtosis, excess kurtosis, Jarque-Bera test, kernel density, heavy tails, data check',
         consts=CONSTS, funcs=CORE + [g.moments_table, g.fig_shapes, g.fig_dax_density, g.tlv_check],
         run="tab = moments_table()\nprint(tab[['n', 'mean', 'sd', 'skew', 'exkurt', 'se_kurt', 'jb', 'jb_p', 'min', "
             "'min_date']].round(3))\nprint(fig_shapes())\nprint(fig_dax_density())\nprint(tlv_check())",
         charts=['sfm_ch2_shapes', 'sfm_ch2_dax_density'], extra=['ch2_moments.csv']),
    dict(name='SFM_ch2_qq_plots',
         desc='QQ plots of the daily log returns of the S&P 500, the BET and Bitcoin, 2010-2026: standardised returns '
              'against the standard Normal distribution and raw returns against the Student-t distribution fitted by '
              'maximum likelihood.',
         keywords='QQ plot, quantiles, Normal distribution, Student-t distribution, heavy tails, goodness of fit',
         consts=CONSTS, funcs=CORE + [g.fig_qq],
         run="print(fig_qq(dist='norm'))\nprint(fig_qq(dist='t'))",
         charts=['sfm_ch2_qq_normal', 'sfm_ch2_qq_t']),
    dict(name='SFM_ch2_student_t_fit',
         desc='The Student-t distribution: densities of t(3), t(5), t(10) scaled to unit variance against the standard '
              'Normal on a linear and a log scale, P(|X| > 4) for each; maximum likelihood fit of a location-scale '
              'Student-t to daily log returns, log-likelihood and AIC against the Normal distribution, for eight series '
              '2010-2026; the S&P 500 histogram with both fits.',
         keywords='Student-t distribution, degrees of freedom, maximum likelihood, AIC, heavy tails, power-law tails',
         consts=CONSTS, funcs=CORE + [g.moments_table, g.fig_t_densities, g.fig_t_fit],
         run="tab = moments_table()\nprint(tab[['nu', 'loc', 'scale', 'll_n', 'll_t', 'aic_n', 'aic_t']].round(2))\n"
             "print(fig_t_densities())\nprint(fig_t_fit())",
         charts=['sfm_ch2_t_densities', 'sfm_ch2_t_fit_sp500'], extra=['ch2_moments.csv']),
    dict(name='SFM_ch2_sigma_days',
         desc='Heavy tails counted: days beyond 3, 4, 5 and 6 standard deviations in eight daily return series, '
              '2010-2026, against the number expected under the Normal distribution, with the Poisson probability of '
              'the observed count; days with losses beyond 3%, 5% and 10%; the gain/loss asymmetry (1% and 99% '
              'quantiles, days below -4 sd and above +4 sd).',
         keywords='heavy tails, k-sigma events, Poisson distribution, extreme returns, gain/loss asymmetry, stylised facts',
         consts=CONSTS, funcs=CORE + [g.sigma_table, g.loss_days, g.fig_sigma_days, g.fig_gain_loss],
         run="print(sigma_table().round(3).T)\nprint(loss_days())\nfig_sigma_days()\nprint(fig_gain_loss())",
         charts=['sfm_ch2_sigma_days', 'sfm_ch2_gain_loss']),
    dict(name='SFM_ch2_cont_stylised_facts',
         desc='The stylised facts of Cont (2001) on real data, 2010-2026: aggregational Gaussianity (excess kurtosis of '
              'non-overlapping h-day returns, h = 1..63; daily vs monthly QQ plots of the S&P 500), absence of linear '
              'autocorrelation and volatility clustering (ACF of returns and of absolute returns, Ljung-Box tests), '
              'the leverage effect (corr(r_t, |r_{t+k}|)), and a summary table of the facts for eight series.',
         keywords='stylised facts, aggregational Gaussianity, autocorrelation, Ljung-Box test, volatility clustering, '
                  'leverage effect, gain/loss asymmetry',
         consts=CONSTS, funcs=CORE + [g.facts_table, g.fig_aggregation, g.fig_qq_horizons, g.fig_clustering, g.fig_acf,
                                      g.fig_leverage],
         run="print(facts_table().round(3).T)\nprint(fig_aggregation())\nprint(fig_qq_horizons())\nfig_clustering()\n"
             "res = fig_acf()\nprint({k: (v['lb_r'], v['lb_abs']) for k, v in res.items()})\nprint(fig_leverage())",
         charts=['sfm_ch2_aggregation', 'sfm_ch2_qq_horizons', 'sfm_ch2_clustering', 'sfm_ch2_acf', 'sfm_ch2_leverage']),
    dict(name='SFM_ch2_tails_by_year',
         desc='Open question of Chapter 2: are the tails of Bitcoin getting thinner? Excess kurtosis and the fitted '
              'Student-t degrees of freedom of daily log returns of Bitcoin and the S&P 500 for each calendar year, '
              '2015-2026.',
         keywords='Bitcoin, tail index, Student-t, degrees of freedom, market maturity, rolling estimation',
         consts=CONSTS, funcs=CORE + [g.fig_tails_by_year],
         run="print(pd.DataFrame({k: {y: v['nu'] for y, v in d.items()} for k, d in fig_tails_by_year().items()}).round(2))",
         charts=['sfm_ch2_tails_by_year']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar2 as s
    QUANTLETS.append(dict(
        name='SFM_ch2_seminar',
        desc='Seminar 2 of Statistics of Financial Markets: Normal, lognormal and Student-t computations, the Normal '
             'approximation of the binomial and the Jarque-Bera statistic on paper; moments of S&P 500 returns with '
             'bootstrap intervals for skewness and kurtosis, the Student-t fit and QQ plots of the BET, aggregational '
             'Gaussianity of the S&P 500 (the solved tasks).',
        keywords='seminar, Normal distribution, lognormal distribution, Jarque-Bera test, bootstrap, QQ plot, '
                 'Student-t, stylised facts',
        consts=CONSTS + [f'SEM_START = {s.SEM_START!r}'],
        funcs=CORE + [s.normal_day, s.lognormal_price, s.binomial_normal, s.sum_of_returns, s.jb_by_hand, s.t_moments,
                      s.bootstrap_moments, s.b1_moments, s.b3_qq, s.b5_aggregation],
        run="print(normal_day(0.05, 1.1, 2.0, 252))\nprint(lognormal_price(100, 0.07, 0.18, 1))\n"
            "print(binomial_normal(252, 0.53, 140))\nprint(jb_by_hand(4000, -0.6, 12.0))\n"
            "print(b1_moments())\nprint(b3_qq())\nprint(b5_aggregation())",
        charts=['ch2_sem_b1', 'ch2_sem_b3', 'ch2_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 2, 'Classical distributions and stylised facts', HERE, data=DATA, submitted=SUBMITTED)
