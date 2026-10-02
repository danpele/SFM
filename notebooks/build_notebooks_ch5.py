"""
build_notebooks_ch5.py -- lecture and seminar notebooks of Chapter 5 (SFM): heavy tails and extreme value theory
================================================================================================================
Output: notebooks/EN/chapter5_lecture_notebook.ipynb, notebooks/EN/chapter5_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_05/generate_all_charts.py and seminar5.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch5.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter5_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter5_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 5
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(5)
import generate_all_charts as g   # noqa: E402
import seminar5 as s              # noqa: E402

CONSTS = ('from scipy import optimize\n'
          f'START = {g.START!r}\nASSETS = {g.ASSETS!r}\nSTOCKS = {g.STOCKS!r}\nBAD_DAYS = {g.BAD_DAYS!r}\n'
          f'COLORS = {g.COLORS!r}\nMODEL_COL = {g.MODEL_COL!r}\nSEED = {g.SEED!r}\nHILL_FRAC = {g.HILL_FRAC!r}\n'
          f'POT_Q = {g.POT_Q!r}\nSPLIT = {g.SPLIT!r}\nBLOCK = {g.BLOCK!r}')
CORE = [g.returns, g.losses, g.hill, g.hill_quantile, g.ls_tail_index, g.block_indices, g.hill_inference,
        g.gev_cdf, g.gev_ppf, g.gev_pdf, g.gpd_sf, g.num_hessian, g.fit_gev, g.fit_gpd, g.return_level,
        g.pot, g.pot_var, g.pot_es, g.pot_tail_prob, g.normal_var, g.normal_es, g.hist_var, g.hist_es, g.mean_excess]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 5: Heavy tails and extreme value theory\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Crash days and the Normal distribution; light and heavy tails; regular variation and the tail index.\n"
       "- The Hill estimator, the Hill plot, standard errors and the Hill quantile; the mean excess function.\n"
       "- Block maxima and the GEV distribution (Fisher–Tippett–Gnedenko); peaks over threshold and the GPD "
       "(Pickands–Balkema–de Haan); return levels.\n"
       "- VaR 1%, ES 2.5% and VaR 0.1% by EVT, historical simulation and the Normal distribution; an out-of-sample check.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 18; "
       "McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, Ch. 5; McNeil and Frey (2000)."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily losses $L_t = -r_t$ in %, with $r_t = 100(\\ln P_t - \\ln P_{t-1})$; S&P 500 and DAX since 1990, BET since "
       "1997, Bitcoin since 2014, BVB stocks since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Hill estimator: $\\hat\\alpha_k = [\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})]^{-1}$, here $k = 2.5\\%$ of $n$.\n"
       "- POT: threshold $u$ = the 90% quantile of the losses; GPD for the excesses; "
       "$\\mathrm{VaR}_p = u + \\frac{\\beta}{\\xi}[(np/N_u)^{-\\xi} - 1]$, $\\mathrm{ES}_p = (\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$.\n"
       "- scipy: `genextreme` uses the shape $c = -\\xi$; `genpareto` uses $c = \\xi$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Why tails matter: crash days\n\n"
       "- The largest daily loss of each series, its z-score and its waiting time under the Normal distribution.\n"
       "- 19 October 1987: the S&P 500 fell about 20% (Carlson, 2007)."),
    code(src(g.crash_table, g.worst_days, g.crash_1987, g.fig_crash_days)),
    code("pd.DataFrame(crash_table()).T[['date', 'ret', 'z', 'log_years', 'n_below_4sd', 'exp_below_4sd']]"),
    code("print(crash_1987())\nfig_crash_days(save=False)"),
    md("## 2. Light and heavy tails; regular variation\n\n"
       "- On log-log axes a power law $P(L > x) \\approx Cx^{-\\alpha}$ is a straight line with slope $-\\alpha$.\n"
       "- The least-squares slope through the 2.5% largest losses (as in the SFE Quantlet SFElshill)."),
    code(src(g.fig_light_heavy, g.fig_loglog_real)),
    code("fig_light_heavy(save=False)"),
    code("fig_loglog_real(save=False)"),
    md("## 3. The Hill estimator\n\n"
       "- Hill plot: $\\hat\\alpha_k$ against $k/n$; inference at $k = 2.5\\%$ of $n$: i.i.d. SE $\\hat\\alpha/\\sqrt{k}$ and "
       "moving-block bootstrap SE (blocks of 20 days).\n"
       "- Hill (Weissman) quantile $\\hat x_p = L_{(k+1)}(k/(np))^{1/\\hat\\alpha}$."),
    code(src(g.fig_hill_plot, g.hill_table)),
    code("fig_hill_plot(save=False)"),
    code("ht = hill_table()\npd.DataFrame(ht).T[['n', 'k', 'alpha', 'lo', 'hi', 'se_iid', 'se_block', 'alpha_1', 'alpha_5', 'alpha_gain']].round(2)"),
    code("fig_hill_plot(STOCKS, 'sfm_ch5_hill_bvb', band='tlv', save=False)"),
    code("x = losses('sp500').values\nk = int(HILL_FRAC * len(x))\n"
         "print('Hill VaR 0.1% of the S&P 500:', round(hill_quantile(x, k, 0.001), 2), '| empirical:', round(hist_var(x, 0.001), 2))"),
    md("## 4. The mean excess function\n\n"
       "- $e(u) = E[L - u \\mid L > u]$: constant for the Exponential law, a rising line for Pareto-type tails."),
    code(src(g.fig_mean_excess)),
    code("fig_mean_excess(save=False)"),
    md("## 5. Block maxima and the GEV distribution\n\n"
       "- The three extreme value laws; the Fisher–Tippett–Gnedenko theorem in a simulation.\n"
       "- GEV fits to monthly maxima of daily losses; PP and QQ plots; return levels with bootstrap intervals."),
    code(src(g.fig_gev_types, g.fig_maxima_sim, g.monthly_maxima, g.gev_table, g.fig_gev_ppqq, g.fig_return_levels)),
    code("fig_gev_types(save=False)"),
    code("fig_maxima_sim(save=False)"),
    code("gt = gev_table()\npd.DataFrame(gt).T[['months', 'xi', 'se_xi', 'mu', 'sigma', 'rl1', 'rl10', 'rl10_lo', 'rl10_hi', 'rl50', 'max']].round(2)"),
    code("fig_gev_ppqq('sp500', save=False)"),
    code("fig_return_levels(gt, save=False)"),
    md("## 6. Peaks over threshold and the GPD\n\n"
       "- GPD densities; threshold choice by parameter stability; GPD fits above the 90% quantile; PP and QQ plots; "
       "the fitted tails; waiting times of the crash days under the EVT tail."),
    code(src(g.fig_gpd_densities, g.threshold_stability, g.fig_threshold_stability, g.fig_gpd_ppqq, g.risk_table,
             g.fig_tail_fit, g.crash_waiting)),
    code("fig_gpd_densities(save=False)"),
    code("stab = fig_threshold_stability(save=False)"),
    code("fig_gpd_ppqq('sp500', save=False)"),
    code("rt = risk_table(ASSETS + STOCKS)\npd.DataFrame({k: v['pot'] for k, v in rt.items()}).T[['u', 'n_u', 'xi', 'se_xi', 'beta']].round(3)"),
    code("fig_tail_fit(rt, save=False)"),
    code("pd.DataFrame(crash_waiting(rt, crash_table(), crash_1987())).T.round(4)"),
    md("## 7. VaR and ES by EVT\n\n"
       "- VaR 1%, ES 2.5% and VaR 0.1%: EVT (POT), historical simulation, the Normal distribution and the Hill quantile.\n"
       "- Out of sample: VaR estimated until 2019, exceedances counted in 2020–2026."),
    code(src(g.out_of_sample, g.fig_oos)),
    code("pd.DataFrame({k: {f'{l} {m}': v[l][m] for l in ['var1', 'es25', 'var01'] for m in v[l]} for k, v in rt.items()}).T.round(2)"),
    code("oos = out_of_sample()\n"
         "pd.DataFrame({k: {f'{l} {m}': o[l][m]['exc'] for l in ['var1', 'var01'] for m in ['EVT', 'Historical', 'Normal']} "
         "| {'expected 1%': o['var1']['expected'], 'expected 0.1%': o['var01']['expected']} for k, o in oos.items()}).T.round(1)"),
    code("fig_oos(save=False)"),
    md("## 8. AI for scientific discovery: has the tail of the BET changed?\n\n"
       "- Open question: is the tail of BET losses lighter or heavier after 2010 than before, once volatility is removed?\n"
       "- Starter result: the Hill index before and after 1 January 2010, with block-bootstrap standard errors.\n"
       "- Check before trusting any answer, yours or an AI's: the sign of the losses, the rule for $k$ fixed in advance, "
       "overlapping windows, and whether the change survives returns divided by a rolling volatility."),
    code(src(g.bet_periods)),
    code("bp = bet_periods()\nprint('z =', round(bp['z'], 2))\npd.DataFrame({p: bp[p] for p in ['before', 'after']}).T[['first', 'last', 'n', 'k', 'u', 'alpha', 'se_block']]"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.hill_from_list, s.pareto_tail, s.normal_same_tail, s.gev_paper, s.pot_paper, s.quarterly_maxima]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 5: Heavy tails and extreme value theory\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 5: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with a measure of precision and an "
       "interpretation question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Losses $L_t = -r_t$ (%), the Hill estimator with inference, GEV and GPD fits by maximum likelihood, "
       "POT VaR and ES, historical and Normal VaR and ES, the mean excess function.\n"
       "- Helpers for Part A: Hill from a list of losses, Pareto tails, GEV probabilities and return levels, POT formulas."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: Hill from eight losses\n\n"
       "**Context.** The eight largest daily losses (%) of a stock, in $n = 2500$ days: 12.4, 9.8, 8.1, 7.5, 6.9, 6.2, 5.8, 5.5.\n\n"
       "1. Compute the Hill estimate with $k = 7$.\n"
       "2. Compute its standard error and the 95% interval.\n"
       "3. Compute the Hill quantile for $p = 0.1\\%$.\n\n"
       "**Report:** $\\hat\\alpha$, SE, the interval, VaR 0.1% and one sentence on the moments."),
    code("# Solution\npd.Series(hill_from_list([12.4, 9.8, 8.1, 7.5, 6.9, 6.2, 5.8, 5.5], 2500))"),
    md("**Interpretation of the result.** $\\hat\\alpha \\approx 2.8$: finite variance, infinite kurtosis; with $k = 7$ "
       "the interval is very wide and also allows $\\alpha < 2$."),
    md("## A2 [Proposed]: Hill for a crypto asset\n\n"
       "**Context.** The nine largest daily losses (%) in $n = 3000$ days: 15.2, 11.0, 9.4, 8.0, 7.7, 7.1, 6.6, 6.3, 6.0. Model: A1.\n\n"
       "1. Compute the Hill estimate with $k = 8$, its standard error and the 95% interval.\n"
       "2. Compute the Hill quantile for $p = 0.1\\%$.\n"
       "3. Explain why the interval is so wide and what would make it narrower.\n\n"
       "**Report:** $\\hat\\alpha$, SE, the interval, VaR 0.1% and one sentence."),
    code("# Solution\npd.Series(hill_from_list([15.2, 11.0, 9.4, 8.0, 7.7, 7.1, 6.6, 6.3, 6.0], 3000))"),
    md("## A3 [Solved]: a Pareto tail\n\n"
       "**Context.** $P(L > x) = 2\\% \\times (x/2.5)^{-3}$ for $x \\ge 2.5\\%$.\n\n"
       "1. Compute $P(L > 5\\%)$ and $P(L > 10\\%)$.\n"
       "2. Compute VaR 1%, VaR 0.1% and ES 1%.\n"
       "3. Compute the mean excess at $u = 4\\%$.\n"
       "4. Compare with a Normal law $N(0, \\sigma^2)$ that has the same $P(L > 2.5\\%) = 2\\%$.\n\n"
       "**Report:** four probabilities, three risk numbers, $e(4)$ and one sentence."),
    code("# Solution\nprint(pareto_tail(2.5, 0.02, 3.0, xs=(5, 10), ps=(0.01, 0.001), u=4.0))\nprint(normal_same_tail(2.5, 0.02, xs=(5, 10)))"),
    md("**Interpretation of the result.** The Pareto tail keeps 5% and 10% losses possible; the Normal law with the same "
       "2% tail at 2.5% makes them practically impossible."),
    md("## A4 [Proposed]: a heavier Pareto tail\n\n"
       "**Context.** $P(L > x) = 1.5\\% \\times (x/3)^{-2.5}$ for $x \\ge 3\\%$. Model: A3.\n\n"
       "1. Compute $P(L > 6\\%)$ and $P(L > 12\\%)$.\n"
       "2. Compute VaR 0.5%, VaR 0.1%, ES 0.5% and the mean excess at $u = 5\\%$.\n"
       "3. Say which moments of $L$ exist.\n\n"
       "**Report:** two probabilities, four numbers and one sentence."),
    code("# Solution\nprint(pareto_tail(3.0, 0.015, 2.5, xs=(6, 12), ps=(0.005, 0.001), u=5.0))"),
    md("## A5 [Solved]: a GEV return level\n\n"
       "**Context.** Monthly maxima of daily losses (%) follow a GEV with $\\xi = 0.25$, $\\mu = 1.5$, $\\sigma = 0.8$.\n\n"
       "1. Compute the probability that the largest daily loss of a month exceeds 6%.\n"
       "2. Compute the return period of a 6% loss in years.\n"
       "3. Compute the 10-year return level.\n"
       "4. Name the type of the GEV and its tail index.\n\n"
       "**Report:** one probability, one period, one level and one sentence."),
    code("# Solution\npd.Series(gev_paper(0.25, 1.5, 0.8, x=6.0, years=10))"),
    md("**Interpretation of the result.** $\\xi > 0$: Fréchet type with tail index $1/\\xi = 4$; a 6% day comes back "
       "about every three years."),
    md("## A6 [Proposed]: the same question with a Gumbel law\n\n"
       "**Context.** Monthly maxima follow a Gumbel law ($\\xi = 0$) with $\\mu = 1.5$, $\\sigma = 0.8$. Model: A5.\n\n"
       "1. Compute $P(M > 6\\%)$ and the return period of 6% in years.\n"
       "2. Compute the 10-year return level.\n"
       "3. Compare with A5: what does $\\xi = 0.25$ change?\n\n"
       "**Report:** one probability, one period, one level and one sentence."),
    code("# Solution\npd.Series(gev_paper(0.0, 1.5, 0.8, x=6.0, years=10))"),
    md("## A7 [Solved]: POT risk measures, step by step\n\n"
       "**Context.** $n = 5000$ daily losses; $u = 2\\%$; $N_u = 250$; GPD fit: $\\xi = 0.2$, $\\beta = 0.8$.\n\n"
       "1. Compute VaR 1% and VaR 0.1%.\n"
       "2. Compute VaR 2.5% and ES 2.5%.\n"
       "3. Compute the mean excess at $u$ and the tail index.\n\n"
       "**Report:** four risk numbers, $e(u)$, $\\alpha$ and one sentence."),
    code("# Solution\npot_paper(5000, 250, 2.0, 0.2, 0.8)"),
    md("**Interpretation of the result.** VaR 0.1% is almost twice VaR 1%; with $\\alpha = 1/\\xi = 5$ the tail is heavy, "
       "but the kurtosis is finite."),
    md("## A8 [Proposed]: POT for a crypto asset\n\n"
       "**Context.** $n = 4000$; $u = 3.1\\%$; $N_u = 200$; GPD fit: $\\xi = 0.3$, $\\beta = 1.5$. Model: A7.\n\n"
       "1. Compute VaR 1%, VaR 0.1% and ES 2.5%.\n"
       "2. Compute the ratio ES 1% / VaR 1% and compare it with $1/(1 - \\xi)$.\n"
       "3. Say whether the formulas can be used for VaR 10%, and why.\n\n"
       "**Report:** three risk numbers, one ratio and two sentences."),
    code("# Solution\nr = pot_paper(4000, 200, 3.1, 0.3, 1.5)\nprint(r)\nprint('ES/VaR at 1%:', round(r['es'][0.01] / r['var'][0.01], 3), '| 1/(1 - xi):', round(1 / 0.7, 3))"),
    # ---------------- Part B
    md("# Part B: real data, precision and interpretation"),
    md("## B1 [Solved]: how heavy is the tail of DAX losses?\n\n"
       "**Question.** What is the tail index of daily DAX losses, how precise is it, and does the kurtosis of DAX returns exist?\n\n"
       "1. Draw the Hill plot of the losses for $k$ between 0.5% and 10% of $n$, with the i.i.d. 95% band.\n"
       "2. Compute $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ with the i.i.d. SE, the 95% interval and the moving-block bootstrap SE "
       "(blocks of 20 days, 500 samples).\n"
       "3. Compute the Hill estimate of the gains with the same $k$.\n"
       "4. Interpretation: is $\\alpha > 4$ (a finite kurtosis) compatible with the data?\n\n"
       "**Report:** the chart, four numbers and one sentence."),
    code("# Solution\n" + src(s.b1_hill) + "\n\n\npd.Series(b1_hill('dax', save=False))"),
    md("**Interpretation of the result.** $\\alpha = 4$ lies about five block-bootstrap standard errors above $\\hat\\alpha$: "
       "the kurtosis of DAX returns does not exist, so the sample kurtosis is not a stable number."),
    md("## B2 [Proposed]: BET, Bitcoin and two BVB stocks\n\n"
       "**Question.** Which of the BET, Bitcoin, OMV Petrom and Banca Transilvania has the heaviest tail of losses? Model: B1.\n\n"
       "1. Compute $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ for each series, with the 95% interval and the block-bootstrap SE.\n"
       "2. Add the estimates at $k = 1\\%$ and $5\\%$ of $n$.\n"
       "3. Test the difference between the BET and Bitcoin with $z = (\\hat\\alpha_1 - \\hat\\alpha_2)/\\sqrt{SE_1^2 + SE_2^2}$.\n"
       "4. Interpretation: is Bitcoin's tail heavier than the BET's once the scale of the losses is set aside?\n\n"
       "**Report:** one table, one $z$ and two sentences."),
    code("# Solution\n" + src(s.b2_table) + "\n\n\nb2 = b2_table()\nprint('z (BET - Bitcoin) =', round(b2.pop('z_bet_btc'), 2))\n"
         "pd.DataFrame(b2).T[['n', 'k', 'u', 'alpha', 'lo', 'hi', 'se_block', 'alpha_1', 'alpha_5']].round(2)"),
    md("## B3 [Solved]: POT for the BET\n\n"
       "**Question.** How large are VaR 1%, ES 2.5% and VaR 0.1% of daily BET losses by EVT, compared with historical "
       "simulation and the Normal distribution?\n\n"
       "1. Draw the empirical mean excess function and mark $u$ (the 90% quantile of the losses).\n"
       "2. Fit a GPD to the excesses over $u$ by maximum likelihood; report $\\hat\\xi$ with its SE and $\\hat\\beta$.\n"
       "3. Draw the QQ plot of the excesses against the fitted GPD.\n"
       "4. Compute VaR 1%, ES 2.5% and VaR 0.1% by EVT, by historical simulation and by the Normal distribution.\n"
       "5. Interpretation: which method would you use for the capital of a bank holding the BET, and why?\n\n"
       "**Report:** the two charts, a table of nine numbers and two sentences."),
    code("# Solution\n" + src(s.b3_pot) + "\n\n\nb3 = b3_pot('bet', save=False)\nprint({c: round(b3['pot'][c], 3) for c in ['u', 'n_u', 'xi', 'se_xi', 'beta']})\n"
         "pd.DataFrame({lab: b3[lab] for lab in ['var1', 'es25', 'var01']}).round(2)"),
    md("**Interpretation of the result.** EVT agrees with the data at 1% and gives a smooth VaR 0.1%; the Normal VaR 0.1% "
       "is less than half of it."),
    md("## B4 [Proposed]: does the threshold matter? Bitcoin\n\n"
       "**Question.** How much do $\\hat\\xi$ and the EVT risk measures of Bitcoin change with the threshold? Model: B3.\n\n"
       "1. For $u$ at the 85%, 90% and 95% quantiles, fit the GPD and report $u$, $N_u$, $\\hat\\xi$ with its SE and $\\hat\\beta$.\n"
       "2. Compute VaR 1%, ES 2.5% and VaR 0.1% for each threshold, and by historical simulation.\n"
       "3. Interpretation: which quantity is robust to the choice of $u$, $\\hat\\xi$ or the VaR?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b4_threshold) + "\n\n\npd.DataFrame(b4_threshold('btc')).T.round(2)"),
    md("## B5 [Solved]: quarterly maxima of the S&P 500\n\n"
       "**Question.** How large is the daily loss that the S&P 500 exceeds once in 10 years, according to a GEV fit to "
       "quarterly maxima?\n\n"
       "1. Build the quarterly maxima of the daily losses.\n"
       "2. Fit a GEV by maximum likelihood; report $\\hat\\xi$ with its 95% interval, $\\hat\\mu$ and $\\hat\\sigma$.\n"
       "3. Draw the QQ plot and the return level plot.\n"
       "4. Compute the 10-year return level (40 quarters) and count the quarters above it.\n"
       "5. Interpretation: is the S&P 500 tail Fréchet or Gumbel?\n\n"
       "**Report:** three parameters, one level, the charts and one sentence."),
    code("# Solution\n" + src(s.b5_quarterly) + "\n\n\npd.Series(b5_quarterly('sp500', save=False)).round(3)"),
    md("**Interpretation of the result.** $\\xi = 0$ lies outside the 95% interval: a Fréchet-type, heavy tail."),
    md("## B6 [Proposed]: does VaR 1% survive a new period?\n\n"
       "**Question.** Estimated until 2014, is the VaR 1% of DAX and BET losses exceeded on about 1% of the days of "
       "2015–2026? Models: B3, B5.\n\n"
       "1. Estimate VaR 1% on the first period by EVT (POT, $u$ at the 90% quantile), historical simulation and the Normal distribution.\n"
       "2. Count the test days with a loss above each VaR and compare with $np$.\n"
       "3. Compute the two-sided binomial p-value of each count.\n"
       "4. Interpretation: why can all three methods fail in the same direction?\n\n"
       "**Report:** a table with six counts and p-values, and two sentences."),
    code("# Solution\n" + src(s.b6_oos) + "\n\n\nb6 = b6_oos()\n"
         "pd.DataFrame({(k, m): b6[k][m] for k in b6 for m in ['EVT', 'Historical', 'Normal']}).T.round(3)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: has the tail of Bitcoin changed?\n\n"
       "**Question.** Is the tail index of Bitcoin losses different in 2014–2017, 2018–2021 and 2022–2026? Models: B1, B2.\n\n"
       "1. Compute the Hill $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ in each period, with the block-bootstrap SE and a 95% interval.\n"
       "2. Say whether any two periods differ significantly.\n"
       "3. Propose one change to the method that would give a sharper answer.\n"
       "4. Interpretation: what can a sample of about 1,500 days say about a tail index?\n\n"
       "**Report:** a table, one sentence per comparison and a plan for a project."),
    code("# Reference analysis\n" + src(s.c1_btc_periods) + "\n\n\npd.DataFrame(c1_btc_periods()).T.round(2)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant estimated the tail of daily S&P 500 losses since 1990 by POT and claimed:\n\n"
       "- (a) the GPD gives $\\xi \\approx 0.16$, so the tail index is $\\alpha = 0.16$ and the variance is infinite;\n"
       "- (b) VaR 1% $= u + (\\beta/\\xi)[(np/N_u)^{\\xi} - 1]$;\n"
       "- (c) ES 2.5% $=$ VaR 2.5% $\\times (1 - \\xi)$;\n"
       "- (d) the same formula gives a reliable VaR 20%;\n"
       "- (e) above $u$, the mean excess function of the fitted GPD is a straight line in the threshold.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\npd.Series(c2_check()).round(3)"),
]

if __name__ == '__main__':
    build(LECTURE, 5, 'lecture')
    build(SEMINAR, 5, 'seminar')
