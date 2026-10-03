"""
build_notebooks_ch15.py -- lecture and seminar notebooks of Chapter 15 (SFM): systemic risk
==========================================================================================
Output: notebooks/EN/chapter15_lecture_notebook.ipynb, notebooks/EN/chapter15_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_15/generate_all_charts.py and seminar15.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch15.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter15_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter15_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 15
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(15)
import generate_all_charts as g   # noqa: E402
import seminar15 as s             # noqa: E402

CONSTS = ('import matplotlib.dates as mdates\nimport statsmodels.api as sm\n'
          'from statsmodels.regression.quantile_regression import QuantReg\nfrom scipy.special import gammaln\n'
          f'START, END = {g.START!r}, {g.END!r}\nBANKS = {g.BANKS!r}\nREGIONS = {g.REGIONS!r}\nALL = {g.ALL!r}\n'
          f'INTL = {g.INTL!r}\nREGION_START = {g.REGION_START!r}\nINDEX = {g.INDEX!r}\nINDEX_NAME = {g.INDEX_NAME!r}\n'
          f'REG_COL = {g.REG_COL!r}\nREG_LAB = {g.REG_LAB!r}\nBAD_DAYS = {g.BAD_DAYS!r}\nEVENTS = {g.EVENTS!r}\n'
          f'EPISODES = {g.EPISODES!r}\nALPHA = {g.ALPHA!r}\nALPHA_MES = {g.ALPHA_MES!r}\nK_CAP = {g.K_CAP!r}\n'
          f'B_BOOT, BLOCK = {g.B_BOOT!r}, {g.BLOCK!r}\nGC_WINDOW = {g.GC_WINDOW!r}\n'
          f'DY_H, DY_WINDOW, DY_STEP = {g.DY_H!r}, {g.DY_WINDOW!r}, {g.DY_STEP!r}\nSEED = {g.SEED!r}')
BASE = [g.bank_price, g.prices, g.returns, g.region_returns, g.bank_portfolio]
COPULA = [g.pseudo_obs, g.empirical_tail_dep, g.t_copula_fit]
COVAR = [g.qreg, g.covar, g.block_idx, g.covar_ci]
MES = [g.mes, g.mes_threshold, g.lrmes, g.srisk, g.srisk_ratio, g.breakeven_leverage]
NET = [g.var_fit, g.var_bic, g.gfevd, g.connectedness]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 15: Systemic risk\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Bank portfolios of the US, Europe and Romania; four systemic episodes (2008, 2011–2012, March 2020, March 2023).\n"
       "- Correlation and tail dependence in crises; the Forbes–Rigobon correction.\n"
       "- CoVaR and Delta-CoVaR by quantile regression, static and with state variables.\n"
       "- MES, LRMES and SRISK; MES before 2008 against the losses in 2008.\n"
       "- Networks: Granger causality and Diebold–Yilmaz connectedness.\n"
       "- References: Adrian and Brunnermeier (2016), *American Economic Review*; Acharya, Pedersen, Philippon and "
       "Richardson (2017) and Brownlees and Engle (2017), *Review of Financial Studies*; Billio, Getmansky, Lo and "
       "Pelizzon (2012); Diebold and Yilmaz (2014); Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, "
       "5th ed., Ch. 16–17."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %, computed after joining the prices on common days.\n"
       "- Banks: 6 US and 5 European banks since 2005, Banca Transilvania (TLV) and BRD since 2010 (BVB days without "
       "trades removed; TLV without 30–31 May 2016, a data error). Markets: S&P 500, Euro Stoxx 50, BET.\n"
       "- Level $\\alpha$ = the probability of the tail: VaR 1%, CoVaR 1%, MES 5%; losses are positive numbers.\n"
       "- CoVaR: quantile regression of the system on the bank at level $\\alpha$; "
       "$\\Delta\\mathrm{CoVaR} = \\hat b\\,(q_{50\\%} - q_\\alpha)$ of the bank.\n"
       "- MES: minus the average bank return on the market's worst $\\alpha$ of days; "
       "SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$, $k = 8\\%$."),
    code(CONSTS + '\n\n\n' + src(*BASE)),
    md("## 1. Bank portfolios and four crises\n\n- Equal-weighted portfolios with daily rebalancing; cumulative returns in "
       "four episodes."),
    code(src(g.fig_bank_indices, g.episode_paths, g.fig_episodes)),
    code("fig_bank_indices(save=False)"),
    code("fig_episodes(save=False)"),
    md("## 2. Correlation and tail dependence in crises\n\n"
       "- 120-day average pairwise correlation; the Forbes–Rigobon correction; empirical tail dependence and the t copula."),
    code(src(*COPULA, g.avg_pairwise_corr, g.forbes_rigobon, g.fig_rolling_corr, g.contagion_test, g.tail_table,
             g.fig_tail_scatter)),
    code("fig_rolling_corr(save=False)"),
    code("contagion_test()"),
    code("pd.DataFrame(tail_table()).T.round(3)"),
    code("fig_tail_scatter(save=False)"),
    md("## 3. CoVaR and Delta-CoVaR\n\n"
       "- Quantile regression of the regional index on each bank at 1% (and 5%); moving-block bootstrap intervals."),
    code(src(*COVAR, g.covar_table, g.fig_covar_qr, g.fig_dcovar, g.fig_var_vs_dcovar)),
    code("fig_covar_qr(save=False)"),
    code("CT = covar_table()\npd.DataFrame(CT).T[['region', 'b', 'var_i', 'covar', 'dcovar', 'dcovar5', 'ci']]"),
    code("fig_dcovar(CT, save=False)\nfig_var_vs_dcovar(CT, save=False)"),
    md("## 4. Time-varying Delta-CoVaR\n\n- State variables: the VIX level and the index return of the previous day."),
    code(src(g.dynamic_dcovar, g.fig_dcovar_dynamic)),
    code("fig_dcovar_dynamic(save=False)"),
    md("## 5. MES, LRMES and SRISK\n\n"
       "- MES 5% with bootstrap intervals; LRMES $= 1 - \\exp(-18\\,\\mathrm{MES}_{2\\%})$; SRISK as a function of leverage; "
       "MES from June 2006 to June 2007 against the returns from July 2007 to December 2008."),
    code(src(*MES, g.mes_table, g.fig_mes, g.fig_srisk, g.mes_vs_crisis, g.fig_mes_crisis)),
    code("MT = mes_table()\npd.DataFrame(MT).T.round(3)"),
    code("fig_mes(MT, save=False)\nfig_srisk(MT, save=False)"),
    code("fig_mes_crisis(save=False)"),
    md("## 6. Networks\n\n"
       "- Granger causality on monthly returns, 36-month windows, the degree of Granger causality (DGC).\n"
       "- Diebold–Yilmaz connectedness of daily returns: the table and the total on 200-day windows."),
    code(src(g.monthly_returns, g.granger_p, g.granger_network, g.rolling_dgc, g.fig_dgc, g.fig_granger_net)),
    code("fig_dgc(save=False)"),
    code("fig_granger_net(save=False)"),
    code(src(*NET, g.dy_static, g.fig_dy_table, g.rolling_total, g.fig_dy_rolling)),
    code("c, p, n = dy_static()\nprint('VAR order', p, 'days', n, 'total connectedness', round(c['total'], 1))\n"
         "fig_dy_table(c, save=False)\npd.DataFrame({'TO': c['to'], 'FROM': c['from'], 'NET': c['net']}).round(1)"),
    code("fig_dy_rolling(save=False)"),
    md("## 7. AI for scientific discovery: is the systemic risk of the Romanian banks imported?\n\n"
       "- Open question: does the euro-area banking sector raise the tail risk of TLV and BRD in stress periods?\n"
       "- Starter result: Delta-CoVaR 1% of each Romanian bank conditional on the European bank portfolio.\n"
       "- Check before trusting any answer, yours or an AI's: the level convention (CoVaR 1% is a positive loss), the "
       "direction of conditioning, Delta-CoVaR against the median, prices joined on common days, bootstrap intervals."),
    code("R = returns(REGIONS['EU'] + REGIONS['RO'], start='2010-01-01')\n"
         "eu = bank_portfolio(R, REGIONS['EU'])\n"
         "for k in REGIONS['RO']:\n"
         "    c = covar(R[k], eu, 0.01)\n"
         "    print(k, 'Delta-CoVaR 1% given the European banks:', round(c['dcovar'], 2), 'slope', round(c['b'], 2))"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.covar_from_qr, s.qr_inputs, s.march2020_table, s.mes_by_hand, s.srisk_example, s.two_banks,
               s.region_measures, s.fisher_z, s.contagion, s.mes_predict]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 15: Systemic risk\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 15: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Data: bank prices, returns on common days, equal-weighted bank portfolios.\n"
       "- Measures: quantile regression, CoVaR and Delta-CoVaR, moving-block bootstrap, MES, LRMES, SRISK, the "
       "Forbes–Rigobon correction, VAR and connectedness.\n"
       "- Helpers for Part A and Part B: CoVaR from a regression, the inputs of A1 and A2, the table of March 2020, MES "
       "by hand, SRISK of one and of several banks, the measures of a region, the Fisher z test, a contagion test, MES "
       "before a crisis against the losses in it."),
    code(CONSTS + '\n\n\n' + src(*BASE, *COVAR, *MES, *NET, g.forbes_rigobon) + '\n\n\n' + src(*SEM_HELPERS)),
    code("# [solutions only]\n" + src(s.b2_other, s.b4_contagion, s.b6_mes2023, s.c1_ro_eu, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: CoVaR of the S&P 500 given JPMorgan Chase\n\n"
       "**Context.** Quantile regression at 1% of the S&P 500 on JPMorgan Chase (daily log returns since 2005): "
       "$\\hat a = -2.35$, $\\hat b = 0.39$; JPM: $q_{1\\%} = -6.27$, $q_{50\\%} = 0.07$; VaR 1% of the S&P 500: 3.53%.\n\n"
       "1. Compute CoVaR 1% of the S&P 500 given that JPM is at its 1% quantile.\n"
       "2. Compute CoVaR 1% at the median of JPM.\n"
       "3. Compute Delta-CoVaR 1%.\n"
       "4. Compare CoVaR 1% with the unconditional VaR 1% of the S&P 500.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nprint(qr_inputs('JPM'))\nc = covar_from_qr(-2.35, 0.39, -6.27, 0.07)\nprint(c, 'ratio', round(c['covar'] / 3.53, 2))"),
    md("**Interpretation of the result.** When JPM is in distress, the VaR 1% of the index is about 1.36 times its "
       "unconditional value."),
    md("## A2 [Proposed]: Bank of America\n\n"
       "**Context.** Same data; quantile regression at 1% of the S&P 500 on BAC: $\\hat a = -2.65$, $\\hat b = 0.26$; "
       "BAC: $q_{1\\%} = -8.13$, $q_{50\\%} = 0.06$. Model: A1.\n\n"
       "1. Compute VaR 1% of BAC, CoVaR 1% and CoVaR 1% at the median.\n"
       "2. Compute Delta-CoVaR 1%.\n"
       "3. Rank JPM and BAC by VaR 1% and by Delta-CoVaR 1%.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nprint(qr_inputs('BAC'))\ncovar_from_qr(-2.65, 0.26, -8.13, 0.06)"),
    md("## A3 [Solved]: MES and LRMES of JPMorgan Chase\n\n"
       "**Context.** Daily log returns (%) of the S&P 500, JPMorgan Chase and Goldman Sachs, 2–27 March 2020 (20 days).\n\n"
       "1. Find the $k = \\lceil 20 \\times 0.10 \\rceil$ worst market days and compute MES 10% of JPM.\n"
       "2. Compute ES 10% of the market on the same days and the ratio MES/ES.\n"
       "3. Compute $\\mathrm{MES}_{2\\%}$: the average loss of JPM on the days when the S&P 500 fell by more than 2%.\n"
       "4. Compute LRMES.\n\n"
       "**Report:** five numbers and one sentence."),
    code("T = march2020_table()\nT"),
    code("# Solution\nmes_by_hand(T, 'JPM')"),
    md("**Interpretation of the result.** On the market's worst days JPM lost more than the market; measured in this "
       "crisis month, LRMES is very high (about 76%)."),
    md("## A4 [Proposed]: MES and LRMES of Goldman Sachs\n\n"
       "**Context.** The same table, for GS. Model: A3.\n\n"
       "1. Compute MES 10% of GS and the ratio MES/ES of the market.\n"
       "2. Compute $\\mathrm{MES}_{2\\%}$ and LRMES of GS.\n"
       "3. Compare GS with JPM on both measures.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nmes_by_hand(T, 'GS')"),
    md("## A5 [Solved]: SRISK of one bank\n\n"
       "**Context.** Debt $D = 900$, market value of equity $W = 100$ (billion RON), LRMES = 55%, $k = 8\\%$.\n\n"
       "1. Compute the leverage $L$ and SRISK.\n"
       "2. Compute the break-even leverage $L^*$.\n"
       "3. Recompute $L$ and SRISK after a fall of 30% in $W$, with $D$ unchanged.\n\n"
       "**Report:** five numbers and one sentence."),
    code("# Solution\nsrisk_example(900, 100, 0.55)"),
    md("**Interpretation of the result.** A fall in the share price raises leverage and SRISK at the same time: SRISK "
       "grows in a crisis before any loan defaults."),
    md("## A6 [Proposed]: SRISK of two banks\n\n"
       "**Context.** Bank X: $D = 570$, $W = 30$, LRMES = 50%; bank Y: $D = 180$, $W = 60$, LRMES = 65% (billion RON). "
       "Model: A5.\n\n"
       "1. Compute the leverage and SRISK of each bank with $k = 8\\%$.\n"
       "2. Compute the aggregate SRISK and the share of each bank.\n"
       "3. Recompute both SRISK values with $k = 5.5\\%$.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\ntwo_banks({'X': (570, 30, 0.50), 'Y': (180, 60, 0.65)})"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: MES and Delta-CoVaR of the US banks\n\n"
       "**Question.** Do MES and Delta-CoVaR point to the same most systemic US bank?\n\n"
       "1. Compute MES 5% of each bank with the S&P 500 as the market.\n"
       "2. Compute Delta-CoVaR 1% of the S&P 500 given each bank by quantile regression.\n"
       "3. Compute 95% moving-block bootstrap intervals for both measures (blocks of 20 days).\n"
       "4. Compute the Spearman correlation between the two rankings.\n"
       "5. Interpretation: do the two measures agree on the most systemic bank?\n\n"
       "**Report:** a table of twelve numbers with intervals, the chart and two sentences."),
    code("# Solution\n" + src(s.b1_us) + "\n\n\nb1 = b1_us(save=False)\n"
         "print({k: b1[k] for k in ['spearman', 'top_mes', 'top_dcovar']})\n"
         "pd.DataFrame(b1['banks']).T[['mes', 'mes_lo', 'mes_hi', 'dcovar', 'ci']]"),
    md("**Interpretation of the result.** No: MES ranks first the banks that fall most with the market (Citigroup, Bank "
       "of America), Delta-CoVaR the banks whose distress moves the market most; the intervals overlap."),
    md("## B2 [Proposed]: European and Romanian banks\n\n"
       "**Question.** Does the disagreement between MES and Delta-CoVaR found in B1 also appear in Europe and in Romania? "
       "Model: B1.\n\n"
       "1. Compute MES 5% and Delta-CoVaR 1% with bootstrap intervals for each bank, against its own market.\n"
       "2. Compute the Spearman correlation of the two rankings for the European banks.\n"
       "3. Compare the two Romanian banks on both measures.\n"
       "4. Interpretation: is the disagreement between MES and Delta-CoVaR as strong in Europe as in the US?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = b2_other()\nfor r, d in b2.items():\n    print(r, {k: d[k] for k in ['spearman', 'top_mes', 'top_dcovar']})\n"
         "    print(pd.DataFrame(d['banks']).T[['mes', 'mes_lo', 'mes_hi', 'dcovar', 'ci']])"),
    md("## B3 [Solved]: US and European banks in 2008\n\n"
       "**Question.** Did the link between US and European banks become stronger after Lehman: contagion or "
       "interdependence?\n\n"
       "1. Compute the correlation of the two portfolios in the calm period (January 2005 – June 2007) and in the crisis "
       "(15 September – 31 December 2008).\n"
       "2. Test the rise in correlation with the one-sided Fisher z test at 5%.\n"
       "3. Compute $\\delta$ from the variance of the US portfolio and the Forbes–Rigobon corrected correlation.\n"
       "4. Repeat the test with the corrected correlation.\n"
       "5. Interpretation: contagion or interdependence?\n\n"
       "**Report:** six numbers, two decisions, the chart and two sentences."),
    code("# Solution\n" + src(s.b3_contagion) + "\n\n\nb3_contagion(save=False)"),
    md("**Interpretation of the result.** Interdependence: the raw correlation rose significantly, but the variance of the "
       "US banks rose about 85 times; after the correction the link is not stronger."),
    md("## B4 [Proposed]: European and Romanian banks in 2020\n\n"
       "**Question.** Did the link between European and Romanian banks become stronger in the COVID-19 crash? Model: B3.\n\n"
       "1. Compute the correlation in the calm period (2019) and in the crisis (20 February – 30 April 2020) and test the "
       "rise with the Fisher z test.\n"
       "2. Compute $\\delta$ from the variance of the European portfolio and the corrected correlation.\n"
       "3. Repeat the test with the corrected correlation.\n"
       "4. Interpretation: contagion or interdependence?\n\n"
       "**Report:** six numbers, two decisions and two sentences."),
    code("# Solution\nb4_contagion()"),
    md("## B5 [Solved]: did MES in 2019 rank the losses of March 2020?\n\n"
       "**Question.** Do the banks with a high MES before a crisis lose more in it, as in Acharya et al. (2017)?\n\n"
       "1. Compute MES 5% of each of the 11 US and European banks in 2019 against its own market.\n"
       "2. Compute the return of each bank from 19 February to 23 March 2020.\n"
       "3. Compute the Spearman correlation between MES and the return and its p-value.\n"
       "4. Interpretation: did MES rank the March 2020 losses?\n\n"
       "**Report:** the chart, the correlation with its p-value and two sentences."),
    code("# Solution\n" + src(s.b5_mes2020) + "\n\n\nb5 = b5_mes2020(save=False)\n"
         "print({k: b5[k] for k in ['spearman', 'p', 'n', 'worst', 'top_mes']})\npd.DataFrame(b5['banks']).T"),
    md("**Interpretation of the result.** Partly: the sign is the one of the paper and the extremes match (Citigroup, "
       "HSBC), but with 11 banks the correlation is not significant at 5%."),
    md("## B6 [Proposed]: did MES in 2022 rank the losses of March 2023?\n\n"
       "**Question.** Does the result of B5 hold in the banking turmoil of March 2023? Model: B5.\n\n"
       "1. Compute MES 5% of each of the 13 banks in 2022 against its own market.\n"
       "2. Compute the return of each bank from 28 February to 31 March 2023.\n"
       "3. Compute the Spearman correlation and its p-value.\n"
       "4. Interpretation: why may MES fail to rank the losses of March 2023?\n\n"
       "**Report:** one table, the correlation with its p-value and two sentences."),
    code("# Solution\nb6 = b6_mes2023()\nprint({k: b6[k] for k in ['spearman', 'p', 'n', 'worst', 'top_mes']})\n"
         "pd.DataFrame(b6['banks']).T"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: how connected are Romanian and euro-area banks?\n\n"
       "**Question.** What share of the risk of TLV and BRD comes from the European banks, and does it rise in crises? "
       "Model: B1 and the connectedness slide.\n\n"
       "1. Fit a VAR to the seven return series (order by BIC) and compute the connectedness table for $H = 10$ days.\n"
       "2. Compute the share of the forecast error variance of TLV and BRD that comes from the five European banks.\n"
       "3. Repeat it on rolling windows of 250 days and compare 2019 with 2020.\n"
       "4. Interpretation: is the systemic risk of the Romanian banks imported?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\nc1_ro_eu()"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the systemic risk of JPMorgan Chase. It answered:\n\n"
       "- (a) the CoVaR 99% of the S&P 500 given JPM is a negative number;\n"
       "- (b) Delta-CoVaR = CoVaR minus the VaR of JPM, a negative number, so JPM reduces systemic risk;\n"
       "- (c) MES 5% is the average loss of the S&P 500 on the worst 5% of days of JPM;\n"
       "- (d) the bank with the largest VaR 1% is always the most systemic one;\n"
       "- (e) the correlation of US and European banks rose in autumn 2008, which proves contagion;\n"
       "- (f) a positive SRISK means that the bank has more capital than it needs in a crisis;\n"
       "- (g) Granger causality from bank A to bank B means that past returns of A help to forecast B; it does not "
       "prove a causal channel.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** seven verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 15, 'lecture')
    build(SEMINAR, 15, 'seminar')
