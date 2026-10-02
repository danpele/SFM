"""
build_quantlets.py -- Quantlet folders of Chapter 5 (SFM): heavy tails and extreme value theory
==============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
Run:  python3 Quantlets/Ch_05/generate_all_charts.py && python3 Quantlets/Ch_05/seminar5.py
      python3 Quantlets/Ch_05/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 5
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
DATA = ('Daily market data from EODHD (S&P 500 and DAX since 1990, BET since 1997, Bitcoin since 2014, OMV Petrom, BRD, '
        'Banca Transilvania and Transgaz since 2010), data/market of the SFM repository')
CONSTS = ['from scipy import optimize', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}', f'STOCKS = {g.STOCKS!r}', f'BAD_DAYS = {g.BAD_DAYS!r}',
          f'COLORS = {g.COLORS!r}', f'MODEL_COL = {g.MODEL_COL!r}', f'SEED = {g.SEED!r}', f'HILL_FRAC = {g.HILL_FRAC!r}',
          f'POT_Q = {g.POT_Q!r}', f'SPLIT = {g.SPLIT!r}', f'BLOCK = {g.BLOCK!r}']
DATAF = [g.returns, g.losses]
HILL = [g.hill, g.hill_quantile, g.ls_tail_index, g.block_indices, g.hill_inference]
ML = [g.num_hessian]
GEV = [g.gev_cdf, g.gev_ppf, g.gev_pdf, g.fit_gev, g.return_level]
GPD = [g.gpd_sf, g.fit_gpd, g.pot, g.pot_var, g.pot_es, g.pot_tail_prob]
RISK = [g.normal_var, g.normal_es, g.hist_var, g.hist_es, g.risk_table]
NODATA = 'Simulated data and closed-form laws only (no market data)'

QUANTLETS = [
    dict(name='SFM_ch5_crashes',
         desc='The largest daily losses of the S&P 500 and DAX (since 1990), the BET (since 1997) and Bitcoin (since 2014): '
              'dates, z-scores and waiting times under the Normal distribution fitted to each series, the number of days '
              'below -4 standard deviations against the Normal expectation, the 19 October 1987 crash (about -20%, '
              'Carlson, 2007) as a z-score, and the daily log returns of the S&P 500 and the BET with the Normal '
              '1-in-100-years loss level.',
         keywords='crash, extreme loss, Normal distribution, z-score, waiting time, heavy tails, S&P 500, DAX, BET, Bitcoin',
         consts=CONSTS, funcs=DATAF + [g.crash_table, g.worst_days, g.crash_1987, g.fig_crash_days],
         run="print(pd.DataFrame(crash_table()).T)\nprint(worst_days())\nprint(crash_1987())\nprint(fig_crash_days())",
         charts=['sfm_ch5_crash_days']),
    dict(name='SFM_ch5_tails_loglog',
         desc='Light against heavy tails: P(X > x) of the Normal, Exponential, Student-t (3 degrees of freedom) and Pareto '
              '(alpha = 3) laws on log-log axes; the empirical tail of daily losses of the S&P 500, DAX, BET and Bitcoin on '
              'log-log axes with the least-squares line through the 2.5% largest losses (slope = -alpha, the least-squares '
              'tail index estimator of SFElshill) and the Normal tail.',
         keywords='heavy tails, regular variation, power law, tail index, least squares, log-log plot, Pareto, Student-t',
         consts=CONSTS, funcs=DATAF + [g.ls_tail_index, g.fig_light_heavy, g.fig_loglog_real],
         run="print(fig_light_heavy())\nprint(fig_loglog_real())",
         charts=['sfm_ch5_light_heavy', 'sfm_ch5_loglog_real']),
    dict(name='SFM_ch5_hill',
         desc='The Hill (1975) estimator of the tail index of daily losses: Hill plots for k between 0.5% and 10% of n '
              '(S&P 500, DAX, BET, Bitcoin; OMV Petrom, BRD, Banca Transilvania and Transgaz since 2010), estimates at '
              'k = 2.5% of n with i.i.d. standard errors, 95% intervals and moving-block bootstrap standard errors (blocks '
              'of 20 days), the Hill index of gains, the Hill (Weissman, 1978) quantile, and the BET before and after 2010. '
              'Based on the Quantlets SFElshill and SFEhillquantile.',
         keywords='Hill estimator, tail index, Hill plot, bootstrap, moving block bootstrap, Weissman quantile, VaR, '
                  'BET, BVB, S&P 500, DAX, Bitcoin',
         consts=CONSTS, funcs=DATAF + HILL + [g.fig_hill_plot, g.hill_table, g.bet_periods],
         run="print(pd.DataFrame(hill_table()).T[['n', 'k', 'alpha', 'lo', 'hi', 'se_iid', 'se_block', 'alpha_gain']].round(2))\n"
             "fig_hill_plot()\nfig_hill_plot(STOCKS, 'sfm_ch5_hill_bvb', band='tlv')\n"
             "x = losses('sp500').values\nk = int(HILL_FRAC * len(x))\nprint('Hill VaR 0.1%:', round(hill_quantile(x, k, 0.001), 2))\n"
             "print(bet_periods())",
         charts=['sfm_ch5_hill_plot', 'sfm_ch5_hill_bvb']),
    dict(name='SFM_ch5_mean_excess',
         desc='The empirical mean excess function e(u) = E[L - u | L > u] of daily losses of the S&P 500, DAX, BET and '
              'Bitcoin, with the threshold u at the 90% quantile and the straight line implied by the generalised Pareto '
              'distribution fitted above u. Based on the Quantlet SFEMeanExcessFun.',
         keywords='mean excess function, threshold, generalised Pareto distribution, peaks over threshold, heavy tails',
         consts=CONSTS, funcs=DATAF + ML + GPD + [g.mean_excess, g.fig_mean_excess],
         run="print(fig_mean_excess())",
         charts=['sfm_ch5_mean_excess']),
    dict(name='SFM_ch5_gev_block_maxima',
         desc='Block maxima and the generalised extreme value (GEV) distribution: densities and distribution functions of '
              'the Gumbel, Frechet and Weibull laws (as in SFEevt1); the Fisher-Tippett-Gnedenko theorem in a simulation '
              '(maxima of Exponential, Pareto and Uniform samples); maximum likelihood GEV fits to the monthly maxima of daily '
              'losses of the S&P 500, DAX, BET and Bitcoin with standard errors; PP and QQ plots (as in SFEtailGEV_pp and '
              'SFEtailGEV_qq); return levels for 1, 10 and 50 years with bootstrap intervals (as in block_max).',
         keywords='extreme value theory, block maxima, GEV, Gumbel, Frechet, Weibull, return level, PP plot, QQ plot, '
                  'maximum likelihood',
         consts=CONSTS, funcs=DATAF + ML + GEV + [g.fig_gev_types, g.fig_maxima_sim, g.monthly_maxima, g.gev_table,
                                                  g.fig_gev_ppqq, g.fig_return_levels],
         run="fig_gev_types()\nprint(fig_maxima_sim())\ngt = gev_table()\n"
             "print(pd.DataFrame(gt).T[['months', 'xi', 'se_xi', 'mu', 'sigma', 'rl1', 'rl10', 'rl10_lo', 'rl10_hi', 'rl50', 'max']].round(2))\n"
             "fig_gev_ppqq('sp500')\nfig_return_levels(gt)",
         charts=['sfm_ch5_gev_types', 'sfm_ch5_maxima_sim', 'sfm_ch5_gev_ppqq', 'sfm_ch5_return_levels']),
    dict(name='SFM_ch5_pot_gpd',
         desc='Peaks over threshold and the generalised Pareto distribution (GPD): GPD densities and tails; the parameter '
              'stability of xi-hat and of the EVT VaR 1% for thresholds between the 80% and 99% quantiles; maximum '
              'likelihood GPD fits to the losses above the 90% quantile (McNeil and Frey, 2000) for the S&P 500, DAX, BET '
              'and Bitcoin; PP and QQ plots of the excesses (as in SFEtailGPareto_pp and SFEtailGPareto_qq); the fitted '
              'tails against the data and the Normal tail; waiting times of the largest losses under the EVT tail.',
         keywords='peaks over threshold, generalised Pareto distribution, threshold choice, parameter stability, PP plot, '
                  'QQ plot, tail estimator, extreme value theory',
         consts=CONSTS, funcs=DATAF + ML + GPD + RISK + [g.crash_table, g.crash_1987, g.fig_gpd_densities, g.threshold_stability,
                                                       g.fig_threshold_stability, g.fig_gpd_ppqq, g.fig_tail_fit, g.crash_waiting],
         run="fig_gpd_densities()\nfig_threshold_stability()\nprint(fig_gpd_ppqq('sp500'))\nrt = risk_table()\n"
             "print(pd.DataFrame({k: v['pot'] for k, v in rt.items()}).T.round(3))\nfig_tail_fit(rt)\n"
             "print(crash_waiting(rt, crash_table(), crash_1987()))",
         charts=['sfm_ch5_gpd_densities', 'sfm_ch5_threshold_stability', 'sfm_ch5_gpd_ppqq', 'sfm_ch5_tail_fit']),
    dict(name='SFM_ch5_evt_var',
         desc='VaR 1%, ES 2.5% and VaR 0.1% of daily losses by extreme value theory (peaks over threshold, u at the 90% '
              'quantile), by historical simulation, by the Normal distribution and by the Hill quantile, for the S&P 500, '
              'DAX, BET, Bitcoin and four BVB stocks; an out-of-sample check: VaR estimated until 2019 and the number of '
              'exceedances in 2020-2026. Based on the Quantlet var_pot.',
         keywords='VaR, expected shortfall, peaks over threshold, historical simulation, Normal distribution, '
                  'out-of-sample, exceedances, extreme value theory',
         consts=CONSTS, funcs=DATAF + HILL + ML + GPD + RISK + [g.out_of_sample, g.fig_oos],
         run="rt = risk_table(ASSETS + STOCKS)\n"
             "print(pd.DataFrame({k: {f'{l} {m}': v[l][m] for l in ['var1', 'es25', 'var01'] for m in v[l]} for k, v in rt.items()}).T.round(2))\n"
             "oos = out_of_sample()\nprint(pd.DataFrame({k: {f'{l} {m}': o[l][m]['exc'] for l in ['var1', 'var01'] for m in ['EVT', 'Historical', 'Normal']} for k, o in oos.items()}).T)\n"
             "fig_oos()",
         charts=['sfm_ch5_oos'], extra=['ch5_tail_table.csv']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar5 as s
    QUANTLETS.append(dict(
        name='SFM_ch5_seminar',
        desc='Seminar 5 of Statistics of Financial Markets: Hill estimates and Hill quantiles from a few large losses, '
             'Pareto tails, GEV return levels and POT risk measures on paper; the Hill plot and Hill inference for DAX '
             'losses, POT for the BET (mean excess, GPD fit, QQ plot, VaR 1%, ES 2.5%, VaR 0.1% against historical '
             'simulation and the Normal distribution), and a GEV fit to quarterly maxima of S&P 500 losses with the '
             '10-year return level.',
        keywords='seminar, Hill estimator, peaks over threshold, generalised Pareto distribution, GEV, return level, VaR, '
                 'expected shortfall',
        consts=CONSTS, funcs=DATAF + HILL + ML + GEV + GPD + RISK[:-1] + [g.mean_excess, s.hill_from_list, s.pareto_tail,
                                                                       s.normal_same_tail, s.gev_paper, s.pot_paper,
                                                                       s.b1_hill, s.b3_pot, s.quarterly_maxima, s.b5_quarterly],
        run="print(hill_from_list([12.4, 9.8, 8.1, 7.5, 6.9, 6.2, 5.8, 5.5], 2500))\n"
            "print(pareto_tail(2.5, 0.02, 3.0, u=4.0))\nprint(gev_paper(0.25, 1.5, 0.8))\nprint(pot_paper(5000, 250, 2.0, 0.2, 0.8))\n"
            "print(b1_hill('dax'))\nprint(b3_pot('bet'))\nprint(b5_quarterly('sp500'))",
        charts=['ch5_sem_b1', 'ch5_sem_b3', 'ch5_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 5, 'heavy tails and extreme value theory', HERE, data=DATA, submitted=SUBMITTED)
