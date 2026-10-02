"""
build_quantlets.py -- Quantlet folders of Chapter 6 (SFM): model selection and risk management
=============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The alpha-stable law (density by FFT, McCulloch and maximum likelihood estimators) comes from Chapter 3
(Quantlets/Ch_03/generate_all_charts.py); its code is copied into each notebook that needs it.
Run:  python3 Quantlets/Ch_06/generate_all_charts.py && python3 Quantlets/Ch_06/seminar6.py
      python3 Quantlets/Ch_06/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 6
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
DATA = ('Daily market data from EODHD (BET, S&P 500 and DAX since 2000, Bitcoin since 2014, Banca Transilvania, OMV Petrom '
        'and BRD since 2010), data/market of the SFM repository')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['from scipy import optimize', 'from scipy.special import gammaln',
          f'START = {g.START!r}', f'STOCK_START = {g.STOCK_START!r}', f'ASSETS = {g.ASSETS!r}', f'STOCKS = {g.STOCKS!r}',
          f'BAD_DAYS = {g.BAD_DAYS!r}', f'MODELS = {g.MODELS!r}', f'K = {g.K!r}', f'MODEL_COL = {g.MODEL_COL!r}',
          f'SHORT = {g.SHORT!r}', f'COLORS = {g.COLORS!r}', f'SEED = {g.SEED!r}', f'SPLIT = {g.SPLIT!r}',
          f'ALPHA_VAR = {g.ALPHA_VAR!r}', f'PANEL = {g.PANEL!r}', f'ALPHA_ES = {g.ALPHA_ES!r}', f'BLOCK = {g.BLOCK!r}']
CORE = g.STABLE + [g.returns, g.skewt_consts, g.skewt_logpdf, g.skewt_cdf, g.skewt_ppf, g.mixture_em, g.mixture_logpdf,
                   g.mixture_cdf, g.mixture_ppf, g.nig_fit, g.skewt_fit, g.ged_fit, g.t_fit, g.fit_model, g.logpdf, g.cdf,
                   g.ppf, g.var_es]
FITS = [g.fit_all, g.fits_table]
KW = 'model selection, maximum likelihood, Student-t, skewed-t, GED, normal inverse Gaussian, Normal mixture, stable distribution'

QUANTLETS = [
    dict(name='SFM_ch6_candidates',
         desc='The seven candidate distributions for daily returns, all with mean 0 and variance 1 (the alpha-stable law with '
              'the interquartile range of the standard Normal law): Normal, Student-t, the skewed-t of Hansen (1994), the '
              'generalised error distribution (GED), a two-component Normal mixture, the normal inverse Gaussian (NIG) and '
              'the alpha-stable law; densities on a linear and a log scale and P(X < -4) under each law.',
         keywords=KW + ', heavy tails, density',
         consts=CONSTS, funcs=CORE + [g.standard_candidates, g.fig_candidates],
         run="print(fig_candidates())", charts=['sfm_ch6_candidates'], data=NODATA),
    dict(name='SFM_ch6_summary',
         desc='Port of the Quantlet SFESumm: minimum, maximum, mean, median, standard deviation, annual volatility, skewness '
              'and kurtosis of the DAX monthly log returns between first trading days, July 2004 - May 2014 and extended to '
              '2026; the same statistics with the Jarque-Bera test for the daily log returns of seven series; the two '
              'erroneous Banca Transilvania days of May 2016 and their effect on the kurtosis and on the fitted Student-t.',
         keywords='summary statistics, skewness, kurtosis, Jarque-Bera, data quality, DAX, BET, Bitcoin, Banca Transilvania',
         consts=CONSTS, funcs=CORE + [g.monthly_first_day, g.summary_stats, g.fig_dax_monthly, g.daily_summary, g.tlv_check],
         run="print(fig_dax_monthly())\nprint(pd.DataFrame(daily_summary()).T)\nprint(tlv_check())",
         charts=['sfm_ch6_dax_monthly']),
    dict(name='SFM_ch6_ml_fits',
         desc='Maximum likelihood fits of seven candidate models (Normal, Student-t, skewed-t, GED, Normal mixture by the EM '
              'algorithm, NIG and alpha-stable) to the daily log returns of the BET, S&P 500, DAX, Bitcoin, Banca '
              'Transilvania, OMV Petrom and BRD: parameters and maximised log-likelihoods; the profile log-likelihood of the '
              'Student-t degrees of freedom of the S&P 500 with its 95% likelihood-ratio interval.',
         keywords=KW + ', profile likelihood, EM algorithm',
         consts=CONSTS, funcs=CORE + FITS + [g.profile_nu],
         run="fits = fit_all()\nfits_table(fits).to_csv('ch6_fits.csv', index=False)\nprint(fits_table(fits)[['series', 'model', 'k', 'loglik']])\n"
             "print(profile_nu('sp500'))",
         charts=['sfm_ch6_profile_nu'], extra=['ch6_fits.csv']),
    dict(name='SFM_ch6_lr_aic',
         desc='Comparison of the seven fitted models: likelihood-ratio tests for nested pairs (Student-t inside skewed-t, '
              'Normal inside GED, Normal inside Student-t on the boundary with the 50:50 chi-square mixture of Self and '
              'Liang, 1987), the Vuong (1989) test for non-nested pairs, AIC, BIC, Delta AIC and Akaike weights for seven '
              'series, and a heat map of Delta AIC.',
         keywords='likelihood-ratio test, Wilks theorem, boundary, Vuong test, AIC, BIC, Akaike weights, model selection',
         consts=CONSTS, funcs=CORE + FITS + [g.lr_test, g.vuong, g.tests_all, g.fig_delta_aic],
         run="fits = fit_all()\nprint(fits_table(fits)[['series', 'model', 'aic', 'bic', 'daic', 'weight']].round(3))\n"
             "print(tests_all(fits))\nfig_delta_aic(fits)",
         charts=['sfm_ch6_delta_aic']),
    dict(name='SFM_ch6_gof_tests',
         desc='Goodness-of-fit tests based on the empirical distribution function: Kolmogorov-Smirnov, Cramer-von Mises and '
              'Anderson-Darling statistics of every fitted model, parametric bootstrap p-values (Stute et al., 1993) for the '
              'Normal, Student-t and skewed-t fits of the S&P 500, simulated Lilliefors critical values, the power of the '
              'three tests with estimated parameters against Student-t and tail-contaminated Normal data, and where the KS '
              'and Anderson-Darling statistics look along the distribution.',
         keywords='goodness of fit, Kolmogorov-Smirnov, Lilliefors, Cramer-von Mises, Anderson-Darling, parametric bootstrap, power',
         consts=CONSTS, funcs=CORE + FITS + [g.edf_stats, g.gof_bootstrap, g.gof_all, g.lilliefors_crit, g.gof_power, g.fig_ks_tail],
         run="x = returns('sp500').values\nfor m in ['Normal', 'Student-t']:\n    print(m, gof_bootstrap(m, x, B=99))\n"
             "print({n: lilliefors_crit(n) for n in (5, 100, 1000)})\nprint(gof_power())\nprint(fig_ks_tail('sp500'))",
         charts=['sfm_ch6_gof_power', 'sfm_ch6_ks_tail']),
    dict(name='SFM_ch6_pp_qq',
         desc='PP plots and QQ plots of the Normal, Student-t, skewed-t and alpha-stable fits of the S&P 500 daily log returns, '
              'and QQ plots of the Normal, Student-t and skewed-t fits of the BET, Bitcoin and Banca Transilvania.',
         keywords='PP plot, QQ plot, goodness of fit, tails, S&P 500, BET, Bitcoin, Banca Transilvania',
         consts=CONSTS, funcs=CORE + FITS + [g.fig_pp_qq, g.fig_qq_assets],
         run="fits = fit_all(['sp500', 'bet', 'btc', 'tlv'])\nprint(fig_pp_qq(fits, 'sp500'))\nfig_qq_assets(fits)",
         charts=['sfm_ch6_pp', 'sfm_ch6_qq', 'sfm_ch6_qq_assets']),
    dict(name='SFM_ch6_tail_var',
         desc='VaR 1% and ES 2.5% of the data and of each fitted model for seven series, the in-sample exceedances of each '
              'model VaR 1% with the Kupiec (1995) test, the left tail of the S&P 500 on log-log axes against five models, '
              'and the ratio of each model VaR 1% to the empirical one.',
         keywords='Value at Risk, expected shortfall, VaR 1%, ES 2.5%, exceedances, Kupiec test, tails',
         consts=CONSTS, funcs=CORE + FITS + [g.kupiec, g.tail_table, g.fig_left_tail, g.fig_var_models],
         run="fits = fit_all()\ntails = tail_table(fits)\nprint(pd.DataFrame({k: {m: v['var1'] for m, v in t.items()} for k, t in tails.items()}).round(2))\n"
             "fig_left_tail(fits, 'sp500')\nfig_var_models(tails)",
         charts=['sfm_ch6_left_tail', 'sfm_ch6_var_models']),
    dict(name='SFM_ch6_out_of_sample',
         desc='Out-of-sample validation: the seven models fitted on 2000-2019 (Bitcoin 2014-2019) and evaluated on 2020-2026 by '
              'the mean log score, the VaR 1% exceedances (Kupiec test) and the pinball loss, with historical simulation; '
              'overfitting with Normal mixtures of 1 to 6 components (S&P 500, estimation 2017-2018); a rolling VaR 1% '
              'backtest (1000-day window, refitted every 20 days) of the Normal, Student-t and skewed-t models and historical '
              'simulation for the S&P 500 and the BET.',
         keywords='out-of-sample, log score, pinball loss, overfitting, AIC, BIC, rolling window, backtesting, VaR 1%',
         consts=CONSTS, funcs=CORE + [g.kupiec, g.pinball, g.out_of_sample, g.fig_oos, g.overfit_mixtures, g.rolling_var, g.fig_rolling_var],
         run="oos = out_of_sample()\nfig_oos(oos)\nprint(overfit_mixtures('sp500'))\ndf, res = rolling_var('sp500')\nprint(res)\n"
             "fig_rolling_var(df, 'sp500')",
         charts=['sfm_ch6_oos', 'sfm_ch6_overfit', 'sfm_ch6_rolling_var']),
    dict(name='SFM_ch6_var_qqplot',
         desc='Port of the Quantlet SFEVaRqqplot: VaR 1% of the DAX from a Normal model with the volatility of the previous 250 '
              'days, as a rectangular moving average (RMA) or an exponential moving average (EMA, lambda = 0.96); the VaR '
              'reliability plot, a QQ plot of the return divided by the VaR forecast against Normal quantiles, and the '
              'exceedances with the Kupiec test.',
         keywords='VaR reliability plot, QQ plot, RMA, EMA, exponential smoothing, Value at Risk, backtesting, DAX',
         consts=CONSTS, funcs=CORE + [g.kupiec, g.var_rma_ema, g.fig_var_qqplot],
         run="print(fig_var_qqplot('dax'))", charts=['sfm_ch6_var_qqplot']),
    dict(name='SFM_ch6_model_uncertainty',
         desc='Model uncertainty: moving-block bootstrap (blocks of 20 days, 200 resamples) of the S&P 500 daily log returns, '
              'the share of resamples in which each of six models has the lowest AIC and the bootstrap spread of VaR 1% '
              'under each model; VaR 1% averaged with the Akaike weights and the range of VaR 1% across the heavy-tailed '
              'models for seven series.',
         keywords='model uncertainty, model risk, block bootstrap, Akaike weights, model averaging, VaR 1%',
         consts=CONSTS, funcs=CORE + FITS + [g.kupiec, g.tail_table, g.averaged_var, g.block_bootstrap, g.model_uncertainty],
         run="print(model_uncertainty('sp500'))\nfits = fit_all()\nprint(pd.DataFrame(averaged_var(fits, tail_table(fits))).T.round(2))",
         charts=['sfm_ch6_model_uncertainty']),
]
try:                                   # seminar code (instructor files, see .gitignore)
    import seminar6 as s
    QUANTLETS.append(dict(
        name='SFM_ch6_seminar',
        desc='Seminar 6 of Statistics of Financial Markets: AIC, BIC, Akaike weights, likelihood-ratio tests (also on the '
             'boundary), a Kolmogorov-Smirnov test with known and estimated parameters and VaR exceedances with the Kupiec '
             'test on paper; the ranking of six candidate models for the BET, DAX and Bitcoin; goodness-of-fit tests with '
             'bootstrap p-values and QQ plots for the S&P 500; crash or data error for Banca Transilvania; out-of-sample VaR 1% '
             'for the DAX and Bitcoin; the AIC winner in two-year windows; the checks of an AI answer.',
        keywords='seminar, model selection, AIC, BIC, likelihood-ratio test, Kolmogorov-Smirnov, Kupiec test, out-of-sample, data quality',
        consts=CONSTS + [f'SEM_MODELS = {s.SEM_MODELS!r}'],
        funcs=CORE + [g.kupiec, g.pinball, g.lr_test, g.vuong, g.edf_stats, g.lilliefors_crit,
                      s.ic_table, s.fit_table, s.part_a, s.b1_ranking, s.b3_gof, s.b4_tlv, s.b5_oos, s.c1_windows, s.c2_check],
        run="print(part_a())\nprint(b1_ranking('bet', chart='ch6_sem_b1')['tab'])\nprint(b3_gof('sp500'))\nprint(b4_tlv()['top'])\n"
            "print(b5_oos('dax'))\nprint(c1_windows())\nprint(c2_check())",
        charts=['ch6_sem_b1', 'ch6_sem_b3', 'ch6_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 6, 'model selection and risk management', HERE, data=DATA, submitted=SUBMITTED)
