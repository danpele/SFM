"""
build_quantlets.py -- Quantlet folders of Chapter 15 (SFM): systemic risk
=========================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Run:  python3 Quantlets/Ch_15/generate_all_charts.py && python3 Quantlets/Ch_15/seminar15.py
      python3 Quantlets/Ch_15/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 15
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
DATA = ('Daily market data from EODHD: adjusted closes of JPMorgan Chase, Bank of America, Citigroup, Goldman Sachs, '
        'Morgan Stanley, Wells Fargo, Deutsche Bank, BNP Paribas, Santander, ING, HSBC (since 2005), Banca Transilvania '
        'and BRD (since 2010); closes of the S&P 500, Euro Stoxx 50, BET and VIX; data/market of the SFM repository')
CONSTS = ['import matplotlib.dates as mdates', 'import statsmodels.api as sm',
          'from statsmodels.regression.quantile_regression import QuantReg', 'from scipy.special import gammaln',
          f'START, END = {g.START!r}, {g.END!r}', f'BANKS = {g.BANKS!r}', f'REGIONS = {g.REGIONS!r}',
          f'ALL = {g.ALL!r}', f'INTL = {g.INTL!r}', f'REGION_START = {g.REGION_START!r}', f'INDEX = {g.INDEX!r}',
          f'INDEX_NAME = {g.INDEX_NAME!r}', f'REG_COL = {g.REG_COL!r}', f'REG_LAB = {g.REG_LAB!r}',
          f'BAD_DAYS = {g.BAD_DAYS!r}', f'EVENTS = {g.EVENTS!r}', f'EPISODES = {g.EPISODES!r}', f'ALPHA = {g.ALPHA!r}',
          f'ALPHA_MES = {g.ALPHA_MES!r}', f'K_CAP = {g.K_CAP!r}', f'B_BOOT, BLOCK = {g.B_BOOT!r}, {g.BLOCK!r}',
          f'GC_WINDOW = {g.GC_WINDOW!r}', f'DY_H, DY_WINDOW, DY_STEP = {g.DY_H!r}, {g.DY_WINDOW!r}, {g.DY_STEP!r}',
          f'SEED = {g.SEED!r}']
BASE = [g.bank_price, g.prices, g.returns, g.region_returns, g.bank_portfolio]
COPULA = [g.pseudo_obs, g.empirical_tail_dep, g.t_copula_fit]
COVAR = [g.qreg, g.covar, g.block_idx, g.covar_ci]
MES = [g.mes, g.mes_threshold, g.lrmes, g.srisk, g.srisk_ratio, g.breakeven_leverage]
NET = [g.var_fit, g.var_bic, g.gfevd, g.connectedness]

QUANTLETS = [
    dict(name='SFM_ch15_crises',
         desc='Equal-weighted portfolios (daily rebalancing) of six US banks, five European banks (since 2005) and the two '
              'Romanian banks Banca Transilvania and BRD (since 2010): value of 100 invested, maximum drawdown, and the '
              'cumulative returns of the bank portfolios and of the S&P 500 in four systemic episodes (Lehman 2008, the '
              'euro-area crisis 2011-2012, March 2020, March 2023).',
         keywords='systemic risk, banks, financial crisis, Lehman Brothers, euro-area crisis, COVID-19, Silicon Valley Bank, '
                  'Credit Suisse, drawdown, bank portfolio',
         consts=CONSTS, funcs=BASE + [g.fig_bank_indices, g.episode_paths, g.fig_episodes],
         run="print(fig_bank_indices())\nprint(fig_episodes())",
         charts=['sfm_ch15_bank_indices', 'sfm_ch15_episodes']),
    dict(name='SFM_ch15_dependence',
         desc='Correlation and tail dependence of banks in crises: 120-day rolling average pairwise correlation of bank '
              'returns (US, Europe, Romania); correlation of the US and European bank portfolios in a calm period and after '
              'Lehman with the Forbes-Rigobon (2002) correction; empirical lower tail dependence at 5% and 1% and the t '
              'copula (rho from Kendall tau, nu by maximum likelihood) of each bank portfolio and its index; joint bad days '
              'on the uniform scale.',
         keywords='correlation, contagion, interdependence, Forbes-Rigobon correction, tail dependence, t copula, Kendall tau, '
                  'pseudo-observations, banks, systemic risk',
         consts=CONSTS, funcs=BASE + COPULA + [g.avg_pairwise_corr, g.forbes_rigobon, g.fig_rolling_corr, g.contagion_test,
                                               g.tail_table, g.fig_tail_scatter],
         run="print(fig_rolling_corr())\nprint(contagion_test())\nprint(pd.DataFrame(tail_table()).T.round(3))\n"
             "print(fig_tail_scatter())",
         charts=['sfm_ch15_rolling_corr', 'sfm_ch15_tail_scatter']),
    dict(name='SFM_ch15_covar',
         desc='CoVaR and Delta-CoVaR (Adrian and Brunnermeier, 2016) at 1% and 5% by linear quantile regression (Koenker '
              'and Bassett, 1978) of the regional equity index (S&P 500, Euro Stoxx 50, BET) on the return of each of 13 '
              'banks, with 95% moving-block bootstrap intervals (blocks of 20 days, 200 resamples); the quantile '
              'regression lines of JPMorgan Chase and the S&P 500; VaR 1% of the bank against Delta-CoVaR 1%; a '
              'time-varying Delta-CoVaR with the lagged VIX level and index return as state variables.',
         keywords='CoVaR, Delta-CoVaR, quantile regression, systemic risk, state variables, VIX, block bootstrap, VaR, banks',
         consts=CONSTS, funcs=BASE + COVAR + [g.covar_table, g.fig_covar_qr, g.fig_dcovar, g.fig_var_vs_dcovar,
                                              g.dynamic_dcovar, g.fig_dcovar_dynamic],
         run="print(fig_covar_qr())\nCT = covar_table()\nprint(pd.DataFrame(CT).T[['region', 'b', 'var_i', 'covar', "
             "'dcovar', 'dcovar5', 'ci']])\nfig_dcovar(CT)\nprint(fig_var_vs_dcovar(CT))\nprint(fig_dcovar_dynamic())",
         charts=['sfm_ch15_covar_qr', 'sfm_ch15_dcovar', 'sfm_ch15_var_vs_dcovar', 'sfm_ch15_dcovar_dynamic'],
         extra=['ch15_covar_table.csv']),
    dict(name='SFM_ch15_mes_srisk',
         desc='MES 5% (Acharya, Pedersen, Philippon and Richardson, 2017) of 13 banks against their regional index with '
              'moving-block bootstrap intervals; MES on the days when the market falls by more than 2%, the LRMES '
              'approximation 1 - exp(-18 MES) (Acharya, Engle and Richardson, 2012), SRISK per unit of market equity as a '
              'function of leverage and the break-even leverage (Brownlees and Engle, 2017, k = 8%); MES measured from June '
              '2006 to June 2007 against the bank returns from July 2007 to December 2008.',
         keywords='MES, marginal expected shortfall, LRMES, SRISK, capital shortfall, leverage, systemic risk, banks, '
                  'global financial crisis',
         consts=CONSTS, funcs=BASE + MES + [g.block_idx, g.mes_table, g.fig_mes, g.fig_srisk, g.mes_vs_crisis,
                                            g.fig_mes_crisis],
         run="MT = mes_table()\nprint(pd.DataFrame(MT).T.round(3))\nfig_mes(MT)\nprint(fig_srisk(MT))\nprint(fig_mes_crisis())",
         charts=['sfm_ch15_mes', 'sfm_ch15_srisk', 'sfm_ch15_mes_crisis'], extra=['ch15_mes_table.csv']),
    dict(name='SFM_ch15_networks',
         desc='Networks of 11 US and European banks: pairwise Granger-causality tests (lag 1, 5% level) on monthly returns '
              'in 36-month rolling windows and the degree of Granger causality (Billio, Getmansky, Lo and Pelizzon, 2012); '
              'the networks of 2007-2009 and 2017-2019; the Diebold-Yilmaz connectedness table of daily returns '
              '(generalized forecast error variance decomposition, H = 10 days, VAR order by BIC) and the total '
              'connectedness on 200-day rolling windows.',
         keywords='network, Granger causality, degree of Granger causality, connectedness, Diebold-Yilmaz, spillover, '
                  'variance decomposition, VAR, banks, systemic risk',
         consts=CONSTS, funcs=BASE + NET + [g.monthly_returns, g.granger_p, g.granger_network, g.rolling_dgc, g.fig_dgc,
                                            g.fig_granger_net, g.dy_static, g.fig_dy_table, g.rolling_total,
                                            g.fig_dy_rolling],
         run="print(fig_dgc())\nprint(fig_granger_net())\nc, p, n = dy_static()\nprint(p, n, round(c['total'], 1))\n"
             "print(c['table'].round(1))\nfig_dy_table(c)\nprint(fig_dy_rolling())",
         charts=['sfm_ch15_dgc', 'sfm_ch15_granger_net', 'sfm_ch15_dy_table', 'sfm_ch15_dy_rolling']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar15 as s
    QUANTLETS.append(dict(
        name='SFM_ch15_seminar',
        desc='Seminar 15 of Statistics of Financial Markets: CoVaR and Delta-CoVaR from a linear quantile regression '
             '(JPMorgan Chase, Bank of America), MES and LRMES from 20 days of March 2020, SRISK of one and of two banks on '
             'paper; MES 5% and Delta-CoVaR 1% with block-bootstrap intervals for US, European and Romanian banks; '
             'correlation in crises with the Forbes-Rigobon correction and Fisher z tests; MES before a crisis against the '
             'losses in March 2020 and March 2023; connectedness of Romanian and euro-area banks.',
        keywords='seminar, systemic risk, CoVaR, Delta-CoVaR, MES, SRISK, contagion, Forbes-Rigobon, connectedness',
        consts=CONSTS, funcs=BASE + COVAR + MES + NET + [g.forbes_rigobon, s.covar_from_qr, s.qr_inputs,
                                                         s.march2020_table, s.mes_by_hand, s.srisk_example,
                                                         s.two_banks, s.region_measures, s.b1_us, s.fisher_z, s.contagion,
                                                         s.b3_contagion, s.mes_predict, s.b5_mes2020],
        run="print(qr_inputs('JPM'))\nT = march2020_table()\nprint(T)\nprint(mes_by_hand(T, 'JPM'))\n"
            "print(srisk_example(900, 100, 0.55))\nprint(b1_us())\nr = b3_contagion()\nprint(r)\nprint(b5_mes2020())",
        charts=['ch15_sem_b1', 'ch15_sem_b3', 'ch15_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 15, 'Systemic risk', HERE, data=DATA, submitted=SUBMITTED)
