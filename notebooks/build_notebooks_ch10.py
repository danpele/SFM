"""
build_notebooks_ch10.py -- lecture and seminar notebooks of Chapter 10 (SFM): VaR, ES and backtesting
=====================================================================================================
Output: notebooks/EN/chapter10_lecture_notebook.ipynb, notebooks/EN/chapter10_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_10/generate_all_charts.py and seminar10.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Both notebooks install the arch package first (not part of Colab).
Run:  python3 notebooks/build_notebooks_ch10.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter10_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter10_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 10
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(10)
import generate_all_charts as g   # noqa: E402
import seminar10 as s             # noqa: E402

INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
CONSTS = ('from arch import arch_model\n'
          f'NAME = {g.NAME!r}\nSTART = {g.START!r}\nASSETS = {g.ASSETS!r}\nBT_ASSETS = {g.BT_ASSETS!r}\n'
          f'BAD_DAYS = {g.BAD_DAYS!r}\nCOLORS = {g.COLORS!r}\nEPISODES = {g.EPISODES!r}\n'
          f'ALPHA_VAR = {g.ALPHA_VAR!r}\nALPHA_ES = {g.ALPHA_ES!r}\nWINDOW = {g.WINDOW!r}\nREFIT = {g.REFIT!r}\n'
          f'OOS_START = {g.OOS_START!r}\nPOT_Q = {g.POT_Q!r}\nMETHODS = {g.METHODS!r}\nMCOL = {g.MCOL!r}\n'
          f'PORT = {g.PORT!r}\nWEIGHTS = {g.WEIGHTS!r}\nN_SIM = {g.N_SIM!r}\nN_Z2 = {g.N_Z2!r}\nSEED = {g.SEED!r}\n'
          f'PLUS = {g.PLUS!r}')
BASE = [g.returns, g.hs_var, g.hs_es, g.normal_var, g.normal_es, g.t_es_factor, g.std_t_q, g.std_t_es, g.t_fit,
        g.t_var_es, g.cf_z, g.cf_var_es, g.pot_fit, g.pot_var, g.pot_es, g.garch_fit]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 10: VaR, ES and backtesting\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- VaR 1% and ES 2.5% of daily returns: definitions, the S&P 500 since 2000.\n"
       "- Five unconditional methods: historical simulation, Normal, Student-t, Cornish–Fisher, EVT (VaRest, SFEVaRqqplot).\n"
       "- Conditional VaR: GARCH(1,1)-t and filtered historical simulation; rolling forecasts since 2005 "
       "(SFEVaRtimeplot, SFEVaRbank).\n"
       "- Backtesting: Kupiec, Christoffersen, the Basel traffic light, the Acerbi–Székely test of ES "
       "(SFEvar_pot_backtesting).\n"
       "- Portfolio VaR for the BET and the S&P 500: variance–covariance, tail dependence, Gaussian and t copulas.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 16–18; "
       "McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, Ch. 2 and 7; Artzner et al. (1999); "
       "Kupiec (1995); Christoffersen (1998)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; S&P 500, DAX and BET since 2000, Bitcoin since 2014, "
       "Banca Transilvania and OMV Petrom since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Level $\\alpha$ = the probability of the tail. $\\mathrm{VaR}_\\alpha = -q_\\alpha$: the loss exceeded with "
       "probability $\\alpha$ (VaR 1%); $\\mathrm{ES}_\\alpha$: the average loss on the worst $\\alpha$ of days (ES 2.5%). "
       "Both are positive numbers (losses).\n"
       "- Historical simulation: $k = \\lceil n\\alpha \\rceil$, VaR $= -r_{(k)}$, ES $= -$ the mean of the $k$ smallest returns."),
    code(CONSTS + '\n\n\n' + src(*BASE)),
    md("## 1. VaR 1% and ES 2.5% of the S&P 500\n\n- Histogram of daily returns with minus VaR 1% and minus ES 2.5%."),
    code(src(g.fig_var_es_def)),
    code("fig_var_es_def(save=False)"),
    md("## 2. Five unconditional methods on six series\n\n"
       "- Historical simulation, Normal, Student-t (maximum likelihood), Cornish–Fisher, EVT (GPD above the 90% quantile "
       "of the losses); GARCH-t and FHS for the next day."),
    code(src(g.conditional_next_day, g.methods_table)),
    code("MT = methods_table()\n"
         "pd.DataFrame({(NAME[k], m): {'VaR 1%': MT[k][m]['var'], 'ES 2.5%': MT[k][m]['es']} for k in ASSETS "
         "for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT', 'GARCH-t', 'FHS']}).T.round(2)"),
    md("## 3. VaR across tail probabilities, precision and horizon\n\n"
       "- $\\mathrm{VaR}_\\alpha$ for $\\alpha$ from 5% to 0.1%; bootstrap intervals of the historical VaR and ES; the "
       "square-root-of-time rule."),
    code(src(g.var_curve, g.fig_var_curves, g.precision, g.horizon_check)),
    code("fig_var_curves(save=False)"),
    code("precision()"),
    code("horizon_check()"),
    md("## 4. Rolling forecasts and backtests\n\n"
       "- One-day VaR 1%, VaR 2.5% and ES 2.5% since 2005 (Bitcoin since 2017): HS and Normal on 500 days, GARCH-t and FHS "
       "re-estimated every 250 days.\n"
       "- Kupiec, Christoffersen, the traffic light, the quantile loss and the Acerbi–Székely $Z_2$ test."),
    code(src(g.rolling_forecasts, g.kupiec, g.christoffersen, g.basel_zone, g.traffic_table, g.z2_stat, g.z2_pvalue,
             g.quantile_loss, g.backtest, g.stress_counts, g.backtest_all)),
    code("bt, frames = backtest_all()\n"
         "pd.DataFrame({(NAME[k], m): {c: bt[k][m][c] for c in ['n', 'x', 'rate', 'p_uc', 'p_ind', 'p_cc', 'max250', "
         "'qloss', 'z2', 'p_z2']} for k in BT_ASSETS for m in METHODS}).T.round(4)"),
    code("pd.DataFrame({(NAME[k], m): {e: bt[k][m]['stress'][e]['x'] for e in EPISODES} for k in ['sp500', 'dax', 'bet'] "
         "for m in METHODS}).T"),
    md("## 5. Charts of the backtests\n\n- VaR 1% in 2008 and 2020; exceedance dates; Binomial(250, 1%) and the zones; "
       "the traffic light through time."),
    code(src(g.fig_rolling_var, g.fig_hits, g.fig_binomial, g.fig_traffic_time)),
    code("fig_rolling_var(frames['sp500'], save=False)"),
    code("fig_hits(frames['sp500'], save=False)"),
    code("pd.DataFrame(fig_binomial(save=False))"),
    code("fig_traffic_time(frames['sp500'], save=False)"),
    md("## 6. Portfolio VaR, tail dependence and copulas\n\n"
       "- A 50/50 BET and S&P 500 portfolio: variance–covariance, historical simulation, Gaussian and t copulas with "
       "empirical margins."),
    code(src(g.portfolio_data, g.pseudo_obs, g.t_copula_loglik, g.gauss_copula_loglik, g.t_tail_dependence, g.copula_fit,
             g.empirical_tail_dep, g.simulate_copula, g.portfolio_analysis, g.fig_rolling_corr, g.fig_copula)),
    code("R = portfolio_data()\nP = portfolio_analysis(R)\n{k: v for k, v in P.items() if k != 'copula'}"),
    code("P['copula']"),
    code("fig_rolling_corr(R, save=False)"),
    code("fig_copula(R, P['copula'], save=False)"),
    md("## 7. AI for scientific discovery: does tail dependence jump in crises?\n\n"
       "- Open question: does the tail dependence between the BVB and world markets rise in crises and come back?\n"
       "- Starter result: the t copula fitted on separate calendar periods.\n"
       "- Check before trusting any answer, yours or an AI's: the VaR convention (VaR 1% is a positive loss), prices "
       "joined on common days, pseudo-observations divided by $n + 1$, the uncertainty of tail estimates, overlapping windows."),
    code("for a, b in [('2005-01-01', '2007-12-31'), ('2008-01-01', '2009-12-31'), ('2017-01-01', '2019-12-31'), "
         "('2020-01-01', '2021-12-31'), ('2023-01-01', '2026-09-18')]:\n"
         "    cf = copula_fit(R.loc[a:b])\n"
         "    print(a[:4], b[:4], 'rho', round(cf['rho'], 3), 'nu', cf['nu'], 'lambda', round(cf['lambda_t'], 3))"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [g.kupiec, g.christoffersen, g.basel_zone, g.portfolio_data, g.pseudo_obs, g.t_copula_loglik,
             g.gauss_copula_loglik, g.t_tail_dependence, g.copula_fit, g.empirical_tail_dep, g.simulate_copula]
SEM_HELPERS = [s.normal_position, s.t_position, s.worst_returns, s.two_bonds, s.kupiec_traffic, s.christoffersen_counts,
               s.five_methods, s.yearly_backtest, s.two_asset_portfolio]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 10: VaR, ES and backtesting\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 10: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n"
       "- Returns; VaR and ES by historical simulation, Normal, Student-t, Cornish–Fisher and EVT; GARCH(1,1)-t; the "
       "Kupiec and Christoffersen tests and the Basel zones; portfolio data, pseudo-observations and copulas.\n"
       "- Helpers for Part A and Part B: a Normal or Student-t position, the worst returns of a series, two bonds, the "
       "Kupiec test with the traffic light, the Christoffersen test from transition counts, five methods, a yearly "
       "backtest, a two-asset portfolio."),
    code(CONSTS + '\n\n\n' + src(*BASE) + '\n\n\n' + src(*SEM_FUNCS) + '\n\n\n' + src(*SEM_HELPERS)),
    code("# [solutions only]\n" + src(s.c1_ratio, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: VaR and ES of a Normal position\n\n"
       "**Context.** A position of 1,000,000 RON; daily returns $N(\\mu, \\sigma^2)$ with $\\mu = 0.05\\%$, $\\sigma = 1.4\\%$.\n\n"
       "1. Compute VaR 1% in % and in RON.\n"
       "2. Compute ES 2.5% in % and in RON.\n"
       "3. Compute the 10-day VaR 1% with the square-root-of-time rule ($\\mu$ ignored).\n"
       "4. Compute the ratio ES 2.5% / VaR 1%.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nnormal_position(1_000_000, 0.05, 1.4)"),
    md("**Interpretation of the result.** For Normal returns ES 2.5% and VaR 1% are almost the same number (ratio about 1.005)."),
    md("## A2 [Proposed]: the same position with Student-t returns\n\n"
       "**Context.** As in A1, but the standardised returns are Student-t with $\\nu = 4$. Model: A1.\n\n"
       "1. Compute $q_{0.01}(z)$ and VaR 1% in % and in RON.\n"
       "2. Compute $\\mathrm{ES}_{2.5\\%}(z)$ and ES 2.5% in % and in RON.\n"
       "3. Compare both numbers with A1.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nt_position(1_000_000, 0.05, 1.4, 4)"),
    md("## A3 [Solved]: historical simulation from the worst days of the BET\n\n"
       "**Context.** The last 500 daily BET returns and their 15 smallest values.\n\n"
       "1. Compute VaR 1% and VaR 2.5% by historical simulation.\n"
       "2. Compute ES 2.5%.\n"
       "3. Translate VaR 1% and ES 2.5% into RON for a position of 1,000,000 RON.\n\n"
       "**Report:** five numbers and one sentence."),
    code("# Solution\nw = worst_returns('bet')\nprint(w['worst'])\n{c: w[c] for c in ['k1', 'k25', 'var1', 'es25']}"),
    md("**Interpretation of the result.** Here ES 2.5% is below VaR 1%: ES is larger than VaR only at the same level "
       "(ES 2.5% is above VaR 2.5%)."),
    md("## A4 [Proposed]: two bonds and subadditivity\n\n"
       "**Context.** Two independent bonds; each loses 100 with probability 3% and 0 otherwise; level 5%. Model: A3.\n\n"
       "1. Compute VaR 5% and ES 5% of one bond.\n"
       "2. Write the distribution of the total loss of the two bonds.\n"
       "3. Compute VaR 5% and ES 5% of the total loss.\n"
       "4. Check subadditivity for VaR and for ES.\n\n"
       "**Report:** four numbers and two sentences."),
    code("# Solution\ntwo_bonds(0.03)"),
    md("## A5 [Solved]: the Kupiec test and the traffic light\n\n"
       "**Context.** A bank's VaR 1% was exceeded on 7 of the last 250 days; a second model on 16 of 1000 days.\n\n"
       "1. Compute $\\ln L_0$, $\\ln L_1$ and $LR_{uc}$ for the bank, and decide at 5%.\n"
       "2. Give the Basel zone, the plus factor and the capital multiplier.\n"
       "3. Compute $LR_{uc}$ for the second model and decide at 5%.\n\n"
       "**Report:** five numbers, two decisions and one sentence."),
    code("# Solution\nprint(kupiec_traffic(7))\nprint(kupiec_traffic(16, 1000))"),
    md("**Interpretation of the result.** A rate of 2.8% is rejected with 250 days (yellow zone, multiplier 3.65); "
       "a rate of 1.6% is not rejected with 1000 days."),
    md("## A6 [Proposed]: the Christoffersen test\n\n"
       "**Context.** 1000 days of VaR 1% forecasts give $n_{00} = 978$, $n_{01} = 8$, $n_{10} = 8$, $n_{11} = 5$. Model: A5.\n\n"
       "1. Compute $\\hat\\pi_{01}$, $\\hat\\pi_{11}$ and $\\hat\\pi$.\n"
       "2. Compute $LR_{ind}$ and decide at 5%.\n"
       "3. Compute $LR_{uc}$ (with $x = 13$, $n = 999$) and $LR_{cc}$, and decide at 5%.\n\n"
       "**Report:** six numbers, three decisions and one sentence."),
    code("# Solution\nchristoffersen_counts(978, 8, 8, 5)"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: five methods for the BET\n\n"
       "**Question.** How much do VaR 1% and ES 2.5% of the BET depend on the estimation method?\n\n"
       "1. Compute VaR 1% and ES 2.5% by historical simulation, Normal and Student-t (maximum likelihood).\n"
       "2. Compute the skewness and the excess kurtosis, and the Cornish–Fisher VaR 1%.\n"
       "3. Fit a GPD to the losses above their 90% quantile and compute the EVT VaR 1% and ES 2.5%.\n"
       "4. Draw the left tail with the five values of minus VaR 1%.\n"
       "5. Interpretation: which number would you report to a risk committee?\n\n"
       "**Report:** a table of ten numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b1_methods) + "\n\n\nb1 = b1_methods('bet', save=False)\n"
         "pd.DataFrame({m: {'VaR 1%': b1[m][0], 'ES 2.5%': b1[m][1]} for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT']}).round(2)"),
    md("**Interpretation of the result.** Historical simulation and EVT agree; the Normal VaR 1% is about a quarter too "
       "low, and Cornish–Fisher is outside its range with an excess kurtosis above 10."),
    md("## B2 [Proposed]: TLV, SNP and Bitcoin\n\n"
       "**Question.** Does the ranking of the methods found for the BET hold for two BVB stocks and for Bitcoin? Model: B1.\n\n"
       "1. Compute VaR 1% and ES 2.5% of each series by the five methods of B1.\n"
       "2. Compute by how much the Normal VaR 1% is below the historical VaR 1%.\n"
       "3. Report $\\hat\\nu$, $\\hat\\xi$ and the excess kurtosis of each series.\n"
       "4. Interpretation: for which series does the Normal understate VaR 1% most?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = {k: five_methods(k) for k in ('tlv', 'snp', 'btc')}\n"
         "pd.DataFrame({NAME[k]: {**{f'VaR {m}': d[m][0] for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT']}, "
         "'gap Normal (%)': 100 * (1 - d['Normal'][0] / d['HS'][0]), 'nu': d['nu'], 'xi': d['xi'], 'excess kurtosis': d['exkurt']} "
         "for k, d in b2.items()}).T.round(2)"),
    md("## B3 [Solved]: backtesting the S&P 500 year by year\n\n"
       "**Question.** Which of two VaR 1% models would a supervisor have penalised, year by year, since 2015?\n\n"
       "1. Compute the one-day VaR 1% by historical simulation on the last 250 days.\n"
       "2. Compute the one-day GARCH(1,1)-t VaR 1%, re-estimating the model at the start of every year on all earlier data.\n"
       "3. Count the exceptions of each model in every calendar year and give the Basel zone of each year.\n"
       "4. Run the Kupiec and Christoffersen tests on the whole period.\n"
       "5. Interpretation: which model would a supervisor have penalised in 2022?\n\n"
       "**Report:** the chart, six statistics with p-values and two sentences."),
    code("# Solution\n" + src(s.b3_backtest) + "\n\n\nb3 = b3_backtest('sp500', save=False)\n"
         "print(pd.DataFrame(b3['years']).T)\n{m: b3[m] for m in ['HS', 'GARCH-t']}"),
    md("**Interpretation of the result.** In 2022 historical simulation had 10 exceptions (red zone) while GARCH-t had 2: "
       "the slow window is penalised when volatility rises; GARCH-t paid in calmer years such as 2021."),
    md("## B4 [Proposed]: backtesting the BET and Bitcoin\n\n"
       "**Question.** Does the conclusion of B3 hold on the BVB and for Bitcoin? Model: B3.\n\n"
       "1. Compute both VaR 1% forecasts as in B3 (BET from 2015, Bitcoin from 2016).\n"
       "2. Count the exceptions per calendar year and give the zones.\n"
       "3. Run the Kupiec and Christoffersen tests on the whole period.\n"
       "4. Interpretation: which model would you keep for each series?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb4 = {k: yearly_backtest(k, start='2016-01-01' if k == 'btc' else '2015-01-01') for k in ('bet', 'btc')}\n"
         "pd.DataFrame({(NAME[k], m): d[m] for k, d in b4.items() for m in ['HS', 'GARCH-t']}).T.round(4)"),
    md("## B5 [Solved]: a portfolio of two BVB stocks\n\n"
       "**Question.** How much does diversification between Banca Transilvania and OMV Petrom reduce VaR 1%, and does it "
       "survive a crisis?\n\n"
       "1. Compute the variance–covariance VaR 1% of the portfolio, the two individual VaRs and the diversification benefit.\n"
       "2. Compute the historical VaR 1% and ES 2.5% of the portfolio.\n"
       "3. Compute the correlation in 2017 and in February–May 2020, and the tail dependence at $q = 5\\%$ and $q = 1\\%$.\n"
       "4. Fit a t copula and compute the copula VaR 1% with empirical margins.\n"
       "5. Interpretation: why does the variance–covariance VaR understate the risk of this portfolio?\n\n"
       "**Report:** eight numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_portfolio) + "\n\n\nb5 = b5_portfolio(('tlv', 'snp'), save=False)\nb5"),
    md("**Interpretation of the result.** Both margins have heavy tails and the two stocks fall together on the worst "
       "days, when the correlation is highest: the Normal variance–covariance VaR is well below the historical one."),
    md("## B6 [Proposed]: DAX and S&P 500\n\n"
       "**Question.** Is the Gaussian copula good enough for a 50/50 DAX and S&P 500 portfolio? Model: B5.\n\n"
       "1. Compute the variance–covariance and the historical VaR 1% and ES 2.5%.\n"
       "2. Fit the Gaussian and the t copula and compare them with the LR statistic.\n"
       "3. Compute the copula VaR 1% and ES 2.5% with empirical margins for both copulas.\n"
       "4. Interpretation: is the Gaussian copula good enough for this pair?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6 = two_asset_portfolio(('dax', 'sp500'))\nb6.pop('R')\nb6"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: is ES 2.5% as large as VaR 1%?\n\n"
       "**Question.** Basel replaced VaR 1% by ES 2.5% because the two are equal for Normal returns; how close are they on "
       "real data and in crises? Models: B1, A5.\n\n"
       "1. Compute the ratio ES 2.5% / VaR 1% by historical simulation on the whole sample of each series.\n"
       "2. Compute the same ratio on the 500 days that end in March 2009 and in May 2020.\n"
       "3. Compute the ratio implied by a fitted Student-t distribution.\n"
       "4. Interpretation: is ES 2.5% a stricter rule than VaR 1% on these markets?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\nc1 = c1_ratio()\npd.DataFrame({k: v for k, v in c1.items() if k != 'normal'}).T.round(3)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the risk of the BET over the last 500 days. It answered:\n\n"
       "- (a) the VaR 99% of the BET is a negative number, the 1% quantile;\n"
       "- (b) ES is always at least as large as VaR, so ES 2.5% is above VaR 1%;\n"
       "- (c) the 10-day VaR 1% is 10 times the one-day VaR 1%;\n"
       "- (d) with 8 exceptions in 250 days the model stays in the green zone;\n"
       "- (e) VaR is subadditive, so the VaR of a portfolio never exceeds the sum of the VaRs of its parts;\n"
       "- (f) if the exceptions cluster, the Christoffersen test can reject even when their number is right.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2 = c2_check()\nc2"),
]

if __name__ == '__main__':
    build(LECTURE, 10, 'lecture')
    build(SEMINAR, 10, 'seminar')
