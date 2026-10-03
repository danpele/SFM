"""
build_quantlets.py -- Quantlet folders of Chapter 9 (SFM): ARCH and GARCH models and their extensions
=====================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The notebooks install the arch package first (pip install arch), as Google Colab does not include it.
Run:  python3 Quantlets/Ch_09/generate_all_charts.py && python3 Quantlets/Ch_09/seminar9.py
      python3 Quantlets/Ch_09/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 9
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
DATA = ('Daily market data from EODHD (S&P 500, DAX and BET since 2000, Bitcoin since 2014, Banca Transilvania and '
        'OMV Petrom since 2010), data/market of the SFM repository')
CONSTS = ['import statsmodels.api as sm', 'from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch',
          'from arch import arch_model',
          f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}', f'BAD_DAYS = {g.BAD_DAYS!r}',
          f'COLORS = {g.COLORS!r}', f'EPISODES = {g.EPISODES!r}', f'OOS_START = {g.OOS_START!r}', f'REFIT = {g.REFIT!r}',
          f'LAMBDA = {g.LAMBDA!r}', f'ALPHA_VAR = {g.ALPHA_VAR!r}', f'SEED = {g.SEED!r}', f'TERM_DATES = {g.TERM_DATES!r}']
CORE = [g.returns, g.fit, g.persistence, g.half_life, g.summary]
NODATA = 'Simulated data only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch9_simulation_likelihood',
         desc='Simulated i.i.d. Normal, ARCH(1) (omega = alpha = 0.5) and GARCH(1,1) (omega = 0.02, alpha = 0.10, '
              'beta = 0.88) paths with unconditional variance 1 (the idea of SFEtimegarch), with their sample kurtosis; '
              'the conditional log-likelihood of a simulated ARCH(1) as a function of alpha for n = 100 and n = 1000 (as '
              'in SFElikarch1); contour plots of the log-likelihood of a simulated GARCH(1,1) (omega = 0.1, alpha = 0.1, '
              'beta = 0.8) over a grid of (alpha, beta) with variance targeting, for n = 500 (as in SFElikgarch) and n = 2000.',
         keywords='ARCH, GARCH, simulation, conditional heteroskedasticity, volatility clustering, kurtosis, likelihood, '
                  'log-likelihood, maximum likelihood, contour plot',
         consts=CONSTS, funcs=[g.simulate_garch, g.fig_simulated, g.arch1_loglik, g.fig_lik_arch1, g.garch_loglik_vt,
                               g.fig_lik_garch],
         run="print(fig_simulated())\nprint(fig_lik_arch1())\nprint(fig_lik_garch())",
         charts=['sfm_ch9_simulated', 'sfm_ch9_lik_arch1', 'sfm_ch9_lik_garch'], data=NODATA),
    dict(name='SFM_ch9_garch_estimation',
         desc='Maximum-likelihood estimation on daily S&P 500 log returns since 2000 (as in SFEgarchest): ARCH(1), '
              'ARCH(5), ARCH(10) and GARCH(1,1) compared by log-likelihood, AIC and BIC; the Gaussian GARCH(1,1) '
              'estimated step by step (scipy) and with the arch package, with classic and robust (Bollerslev-Wooldridge) '
              'standard errors; GARCH(1,1) with Normal, Student-t and skewed-t innovations; QQ plots of the '
              'standardised residuals.',
         keywords='GARCH, ARCH, maximum likelihood, quasi-maximum likelihood, robust standard errors, Student-t, '
                  'skewed t, standardised residuals, QQ plot, S&P 500, arch package',
         consts=CONSTS, funcs=CORE + [g.garch11_negloglik, g.fit_step_by_step, g.arch_q_table, g.estimation_sp500, g.fig_qq],
         run="print(pd.DataFrame(arch_q_table()).T.round(3))\nest = estimation_sp500()\n"
             "print(pd.DataFrame({'step by step': est['step']['params'], 'arch': est['normal']['params'], "
             "'classic SE': est['normal']['se_classic'], 'robust SE': est['normal']['se']}).round(4))\n"
             "print(pd.DataFrame({d: est[d]['params'] for d in ['normal', 't', 'skewt']}).round(4))\nprint(fig_qq())",
         charts=['sfm_ch9_qq']),
    dict(name='SFM_ch9_volatility_markets',
         desc='GARCH(1,1) with Student-t innovations for the S&P 500, DAX, BET (since 2000), Bitcoin (since 2014), Banca '
              'Transilvania and OMV Petrom (since 2010): parameters with robust standard errors, persistence, half-life, '
              'long-run and sample volatility; daily S&P 500 returns with the 2008 and 2020 episodes; annualised '
              'conditional volatility (as in SFEvolgarchest); the decay of a variance shock.',
         keywords='GARCH, conditional volatility, persistence, half-life, IGARCH, volatility clustering, global financial '
                  'crisis, COVID-19, S&P 500, DAX, BET, Bitcoin, BVB',
         consts=CONSTS, funcs=CORE + [g.markets_table, g.fig_returns, g.fig_vol, g.fig_persistence],
         run="T = markets_table()\nprint(pd.DataFrame({NAME[k]: {**v['params'], 'persistence': v['pers'], 'half-life': v['hl'], "
             "'long-run vol': v.get('vol_lr'), 'sample vol': v['vol_sample']} for k, v in T.items()}).T.round(4))\n"
             "print(fig_returns())\nfig_persistence(T)\nprint(fig_vol(('sp500', 'dax'), 'sfm_ch9_vol_sp500_dax'))\n"
             "print(fig_vol(('bet', 'btc'), 'sfm_ch9_vol_bet_btc'))",
         charts=['sfm_ch9_returns', 'sfm_ch9_vol_sp500_dax', 'sfm_ch9_vol_bet_btc', 'sfm_ch9_persistence'],
         extra=['ch9_garch_table.csv']),
    dict(name='SFM_ch9_asymmetry',
         desc='Asymmetric GARCH models with Student-t innovations: GJR-GARCH(1,1) (Glosten, Jagannathan and Runkle, 1993) '
              'and EGARCH(1,1) (Nelson, 1991) for six series; the likelihood-ratio test of GJR against GARCH; the '
              'sign-bias test of Engle and Ng (1993); news impact curves of GARCH, GJR and EGARCH for the S&P 500 and '
              'Bitcoin, with a Nadaraya-Watson kernel estimate of E[r(t)^2 | r(t-1)] (the idea of SFENewsImpactCurve).',
         keywords='leverage effect, asymmetry, GJR-GARCH, EGARCH, news impact curve, kernel regression, sign-bias test, '
                  'likelihood ratio test, S&P 500, Bitcoin, BET',
         consts=CONSTS, funcs=CORE + [g.news_impact, g.kernel_nic, g.fig_nic, g.sign_bias_test, g.asym_table],
         run="print(fig_nic())\nat = asym_table()\nprint(pd.DataFrame({NAME[k]: {c: v[c] for c in ['gjr_alpha', 'gjr_gamma', "
             "'gjr_t', 'lr', 'lr_p', 'eg_gamma', 'eg_t']} for k, v in at.items()}).T.round(3))",
         charts=['sfm_ch9_nic'], extra=['ch9_asym_table.csv']),
    dict(name='SFM_ch9_diagnostics',
         desc='Diagnostics of GARCH models: Ljung-Box Q(10) of returns, squared returns, standardised residuals and their '
              'squares, ARCH-LM(5), for GARCH(1,1) with Normal and Student-t innovations (S&P 500) and for six series; '
              'the ACF of squared returns against the ACF of squared standardised residuals; nine models (GARCH, GJR, '
              'EGARCH with Normal, Student-t and skewed-t innovations) compared by AIC and BIC.',
         keywords='diagnostics, standardised residuals, Ljung-Box test, ARCH-LM test, autocorrelation function, model '
                  'selection, AIC, BIC, GARCH, EGARCH, GJR-GARCH',
         consts=CONSTS, funcs=CORE + [g.ljung_box, g.diagnostics, g.diag_markets, g.fig_acf_diag, g.model_selection],
         run="print(diagnostics())\nprint(diag_markets())\nprint(fig_acf_diag())\nprint(model_selection().round(1))",
         charts=['sfm_ch9_acf_diag'], extra=['ch9_model_selection.csv']),
    dict(name='SFM_ch9_forecasts',
         desc='Volatility forecasts: multi-step GARCH(1,1)-t forecasts and the term structure of volatility for the S&P '
              '500 on a calm day, at the COVID-19 peak and on the last day; out-of-sample one-day variance forecasts since '
              '2015 (GARCH-t and GJR-t re-estimated every 250 days, EWMA with lambda = 0.94) compared by QLIKE (Patton, '
              '2011) and the Diebold-Mariano test for the S&P 500, DAX, BET and Bitcoin; one-day VaR 1% from GARCH-t and '
              'EWMA-Normal with their exceedances.',
         keywords='volatility forecasting, term structure, QLIKE, Diebold-Mariano test, EWMA, out-of-sample, value at '
                  'risk, VaR 1%, exceedances, backtesting, GARCH',
         consts=CONSTS, funcs=CORE + [g.garch_path_forecast, g.fig_term_structure, g.ewma_variance, g.oos_forecasts,
                                      g.qlike, g.dm_test, g.forecast_eval, g.fig_forecast_eval, g.fig_var],
         run="print(fig_term_structure())\nfe, frames = forecast_eval()\n"
             "print(pd.DataFrame({NAME[k]: {**{f'QLIKE {m}': v['qlike'][m] for m in v['qlike']}, 'DM t': v['dm_garch_ewma']['t'], "
             "'exceed. GARCH-t': v['exc_garch'], 'exceed. EWMA': v['exc_ewma']} for k, v in fe.items()}).T.round(4))\n"
             "fig_forecast_eval(frames)\nprint(fig_var(frames))",
         charts=['sfm_ch9_term_structure', 'sfm_ch9_forecast_eval', 'sfm_ch9_var'], extra=['ch9_forecast_eval.csv']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar9 as s
    QUANTLETS.append(dict(
        name='SFM_ch9_seminar',
        desc='Seminar 9 of Statistics of Financial Markets: GARCH(1,1) algebra, multi-step forecasts and VaR 1%, the ARCH(1) '
             'likelihood and GJR and EGARCH news impact on paper; GARCH(1,1)-t for the DAX with robust standard errors; '
             'the leverage effect in the S&P 500 (GJR-GARCH, likelihood-ratio and sign-bias tests, news impact curves); '
             'out-of-sample QLIKE and VaR 1% exceedances of GARCH-t and EWMA for the S&P 500.',
        keywords='seminar, GARCH, half-life, volatility forecast, VaR, GJR-GARCH, leverage effect, QLIKE, EWMA',
        consts=CONSTS, funcs=CORE + [g.news_impact, g.sign_bias_test, g.ewma_variance, g.oos_forecasts, g.qlike, g.dm_test,
                                     s.garch_algebra, s.garch_forecasts, s.arch1_likelihood, s.asym_news, s.b1_estimate,
                                     s.b3_asymmetry, s.b5_oos],
        run="print(garch_algebra(0.02, 0.08, 0.90, 1.5, -3.0))\nprint(garch_forecasts(0.02, 0.08, 0.90, 2.09))\n"
            "print(arch1_likelihood(0.5, 0.5, [0.5, -1.2, 2.0, -0.3]))\nprint(b1_estimate('dax'))\nprint(b3_asymmetry('sp500'))\n"
            "print(b5_oos('sp500'))",
        charts=['ch9_sem_b1', 'ch9_sem_b3', 'ch9_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 9, 'ARCH and GARCH models', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
