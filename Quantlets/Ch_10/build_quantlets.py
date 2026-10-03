"""
build_quantlets.py -- Quantlet folders of Chapter 10 (SFM): VaR, ES and backtesting
===================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The notebooks install the arch package first (pip install arch), as Google Colab does not include it.
Run:  python3 Quantlets/Ch_10/generate_all_charts.py && python3 Quantlets/Ch_10/seminar10.py
      python3 Quantlets/Ch_10/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 10
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
CONSTS = ['from arch import arch_model',
          f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}', f'BT_ASSETS = {g.BT_ASSETS!r}',
          f'BAD_DAYS = {g.BAD_DAYS!r}', f'COLORS = {g.COLORS!r}', f'EPISODES = {g.EPISODES!r}',
          f'ALPHA_VAR = {g.ALPHA_VAR!r}', f'ALPHA_ES = {g.ALPHA_ES!r}', f'WINDOW = {g.WINDOW!r}', f'REFIT = {g.REFIT!r}',
          f'OOS_START = {g.OOS_START!r}', f'POT_Q = {g.POT_Q!r}', f'METHODS = {g.METHODS!r}', f'MCOL = {g.MCOL!r}',
          f'PORT = {g.PORT!r}', f'WEIGHTS = {g.WEIGHTS!r}', f'N_SIM = {g.N_SIM!r}', f'N_Z2 = {g.N_Z2!r}',
          f'SEED = {g.SEED!r}', f'PLUS = {g.PLUS!r}']
BASE = [g.returns, g.hs_var, g.hs_es, g.normal_var, g.normal_es, g.t_es_factor, g.std_t_q, g.std_t_es, g.t_fit,
        g.t_var_es, g.cf_z, g.cf_var_es, g.pot_fit, g.pot_var, g.pot_es, g.garch_fit]
BT = [g.rolling_forecasts, g.kupiec, g.christoffersen, g.basel_zone, g.traffic_table, g.z2_stat, g.z2_pvalue,
      g.quantile_loss, g.backtest, g.stress_counts, g.backtest_all]

QUANTLETS = [
    dict(name='SFM_ch10_var_methods',
         desc='VaR 1% and ES 2.5% of daily log returns (VaR_alpha = -q_alpha, the loss exceeded with probability alpha) '
              'by historical simulation, Normal, Student-t (maximum likelihood), Cornish-Fisher and EVT (peaks over '
              'threshold, GPD above the 90% quantile of the losses) for the S&P 500, DAX, BET, Bitcoin, Banca '
              'Transilvania and OMV Petrom (as in the SFE Quantlets VaRest and SFEVaRqqplot); VaR as a function of the '
              'tail probability from 5% to 0.1%; bootstrap intervals of the historical VaR and ES; the square-root-of-time '
              'rule against 10-day historical VaR; GARCH(1,1)-t and filtered historical simulation VaR for the next day.',
         keywords='value at risk, VaR, expected shortfall, ES, historical simulation, Student-t, Cornish-Fisher, extreme '
                  'value theory, peaks over threshold, GPD, bootstrap, square-root-of-time rule, S&P 500, BET',
         consts=CONSTS, funcs=BASE + [g.conditional_next_day, g.methods_table, g.fig_var_es_def, g.var_curve,
                                      g.fig_var_curves, g.precision, g.horizon_check],
         run="print(fig_var_es_def())\nMT = methods_table()\n"
             "print(pd.DataFrame({(NAME[k], m): {'VaR 1%': MT[k][m]['var'], 'ES 2.5%': MT[k][m]['es']} for k in ASSETS "
             "for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT', 'GARCH-t', 'FHS']}).T.round(2))\n"
             "fig_var_curves()\nprint(precision())\nprint(horizon_check())",
         charts=['sfm_ch10_var_es_def', 'sfm_ch10_var_curves'], extra=['ch10_methods_table.csv']),
    dict(name='SFM_ch10_conditional_var',
         desc='Conditional VaR: GARCH(1,1) with standardised Student-t innovations and filtered historical simulation '
              '(Hull and White, 1998; Barone-Adesi et al., 1999) for the next day; one-day-ahead VaR 1%, VaR 2.5% and ES '
              '2.5% forecasts since 2005 from historical simulation and the Normal method on 500-day windows and from '
              'GARCH-t and FHS re-estimated every 250 days (as in SFEVaRtimeplot and SFEVaRbank); the S&P 500 in the '
              '2008 and 2020 stress periods.',
         keywords='conditional VaR, GARCH, Student-t, filtered historical simulation, rolling window, out-of-sample, '
                  'ghost effect, global financial crisis, COVID-19, S&P 500',
         consts=CONSTS, funcs=BASE + [g.conditional_next_day, g.rolling_forecasts, g.fig_rolling_var],
         run="print(conditional_next_day('sp500'))\ndf = rolling_forecasts('sp500')\nfig_rolling_var(df)",
         charts=['sfm_ch10_rolling_var']),
    dict(name='SFM_ch10_backtesting',
         desc='Backtesting VaR 1% forecasts of four methods (historical simulation, Normal, GARCH-t, FHS) for the S&P 500, '
              'DAX, BET (since 2005) and Bitcoin (since 2017): exceedance counts, the Kupiec (1995) proportion-of-failures '
              'test, the Christoffersen (1998) independence and conditional coverage tests, the Basel traffic light '
              '(Binomial(250, 1%) zones and plus factors of 1996) on rolling 250-day windows, the average quantile loss, '
              'the 2008 and 2020 stress periods, and the Acerbi-Szekely (2014) Z2 test of ES 2.5% with simulated p-values '
              '(as in SFEvar_pot_backtesting).',
         keywords='backtesting, VaR exceedances, Kupiec test, proportion of failures, Christoffersen test, conditional '
                  'coverage, Basel traffic light, binomial distribution, quantile loss, expected shortfall backtest, Z2',
         consts=CONSTS, funcs=BASE + BT + [g.fig_binomial, g.fig_hits, g.fig_traffic_time],
         run="print(pd.DataFrame(fig_binomial()))\nbt, frames = backtest_all()\n"
             "print(pd.DataFrame({(NAME[k], m): {c: bt[k][m][c] for c in ['n', 'x', 'rate', 'p_uc', 'p_ind', 'p_cc', "
             "'max250', 'qloss', 'z2', 'p_z2']} for k in BT_ASSETS for m in METHODS}).T.round(4))\n"
             "fig_hits(frames['sp500'])\nfig_traffic_time(frames['sp500'])",
         charts=['sfm_ch10_binomial', 'sfm_ch10_hits', 'sfm_ch10_traffic_time'], extra=['ch10_backtest_table.csv']),
    dict(name='SFM_ch10_portfolio_copula',
         desc='A 50/50 portfolio of the BET and the S&P 500 (prices joined on common days, simple returns): '
              'variance-covariance VaR 1% and ES 2.5%, diversification benefit and Euler contributions; 250-day rolling '
              'correlation; Kendall tau, empirical lower tail dependence; Gaussian and t copulas (rho from Kendall tau, '
              'nu by maximum likelihood), their tail dependence; portfolio VaR and ES by copula simulation with empirical '
              'margins against historical simulation.',
         keywords='portfolio VaR, variance-covariance, diversification, correlation, tail dependence, copula, Gaussian '
                  'copula, t copula, Sklar theorem, Kendall tau, pseudo-observations, BET, S&P 500',
         consts=CONSTS, funcs=[g.returns, g.hs_var, g.hs_es, g.normal_var, g.normal_es, g.portfolio_data, g.pseudo_obs,
                               g.t_copula_loglik, g.gauss_copula_loglik, g.t_tail_dependence, g.copula_fit,
                               g.empirical_tail_dep, g.simulate_copula, g.portfolio_analysis, g.fig_rolling_corr,
                               g.fig_copula],
         run="R = portfolio_data()\nP = portfolio_analysis(R)\nprint({k: v for k, v in P.items() if k != 'copula'})\n"
             "print(P['copula'])\nprint(fig_rolling_corr(R))\nprint(fig_copula(R, P['copula']))",
         charts=['sfm_ch10_rolling_corr', 'sfm_ch10_copula']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar10 as s
    QUANTLETS.append(dict(
        name='SFM_ch10_seminar',
        desc='Seminar 10 of Statistics of Financial Markets: Normal and Student-t VaR 1% and ES 2.5% of a position, '
             'historical simulation from the worst returns of the BET, a subadditivity counterexample, the Kupiec test, '
             'the Basel traffic light and the Christoffersen test on paper; VaR 1% and ES 2.5% of the BET by five '
             'methods; a yearly backtest of the S&P 500 (historical simulation against GARCH-t); a portfolio of Banca '
             'Transilvania and OMV Petrom (variance-covariance, historical simulation, tail dependence, copulas).',
        keywords='seminar, VaR, expected shortfall, historical simulation, Kupiec test, Christoffersen test, traffic light, '
                 'subadditivity, portfolio, copula',
        consts=CONSTS, funcs=BASE + [g.kupiec, g.christoffersen, g.basel_zone, g.portfolio_data, g.pseudo_obs,
                                     g.t_copula_loglik, g.gauss_copula_loglik, g.t_tail_dependence, g.copula_fit,
                                     g.simulate_copula, g.empirical_tail_dep,
                                     s.normal_position, s.t_position, s.worst_returns, s.two_bonds, s.kupiec_traffic,
                                     s.christoffersen_counts, s.five_methods, s.b1_methods, s.yearly_backtest,
                                     s.b3_backtest, s.two_asset_portfolio, s.b5_portfolio],
        run="print(normal_position(1_000_000, 0.05, 1.4))\nprint(worst_returns('bet'))\nprint(kupiec_traffic(7))\n"
            "print(b1_methods('bet'))\nprint(b3_backtest('sp500'))\nprint(b5_portfolio(('tlv', 'snp')))",
        charts=['ch10_sem_b1', 'ch10_sem_b3', 'ch10_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 10, 'VaR, ES and backtesting', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
