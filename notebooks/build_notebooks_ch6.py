"""
build_notebooks_ch6.py -- lecture and seminar notebooks of Chapter 6 (SFM): model selection and risk management
==============================================================================================================
Output: notebooks/EN/chapter6_lecture_notebook.ipynb, notebooks/EN/chapter6_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_06/generate_all_charts.py and seminar6.py (and the alpha-stable law of
Quantlets/Ch_03) with inspect.getsource, so the notebooks stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch6.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter6_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter6_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 6
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(6)
import generate_all_charts as g   # noqa: E402
import seminar6 as s              # noqa: E402
from build_quantlets import CONSTS, CORE   # noqa: E402

import json   # noqa: E402
C2 = json.load(open(os.path.join(os.path.dirname(g.__file__), 'sem6_results.json')))['C2']
SETUP = '\n'.join(CONSTS) + '\n\n\n' + src(*CORE)

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 6: Model selection and risk management\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Seven candidate distributions for daily returns: Normal, Student-t, skewed-t (Hansen, 1994), GED, a two-component "
       "Normal mixture, NIG and the α-stable law (Chapter 3).\n"
       "- Maximum likelihood, likelihood-ratio and Vuong tests, AIC and BIC, Akaike weights.\n"
       "- Goodness of fit: Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling; PP and QQ plots.\n"
       "- The purpose decides: VaR 1% in sample and out of sample, overfitting, model uncertainty.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 16 and 18; "
       "Burnham and Anderson (2002); Stephens (1974)."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns in %: indices since 2000, Bitcoin since 2014, Bucharest stocks since 2010; the two erroneous "
       "Banca Transilvania days of May 2016 are removed.\n"
       "- Log-likelihood $\\ell(\\theta) = \\sum_t \\ln f(r_t; \\theta)$; $\\mathrm{AIC} = -2\\ell + 2k$, $\\mathrm{BIC} = -2\\ell + k\\ln n$.\n"
       "- `fit_model(name, x)` returns the ML parameters and the maximised log-likelihood; `logpdf`, `cdf`, `ppf` evaluate a fitted model.\n"
       "- The skewed-t of Hansen (1994) is parameterised by its mean μ, standard deviation σ, degrees of freedom ν > 2 and skewness λ.\n"
       "- VaR 1% $= -q_{1\\%}$ and ES 2.5% $= -\\frac{1}{0.025}\\int_0^{0.025} q_u\\,du$."),
    code(SETUP),
    md("## 1. The candidates and the data\n\n"
       "- Densities with variance 1; the summary statistics of the Quantlet SFESumm (DAX monthly log returns); "
       "the two erroneous Banca Transilvania days."),
    code(src(g.standard_candidates, g.fig_candidates, g.monthly_first_day, g.summary_stats, g.fig_dax_monthly,
             g.daily_summary, g.tlv_check)),
    code("fig_candidates(save=False)"),
    code("fig_dax_monthly(save=False)"),
    code("pd.DataFrame(daily_summary()).T[['n', 'mean', 'sd', 'skew', 'kurt', 'jb']].round(3)"),
    code("tlv_check()"),
    md("## 2. Maximum likelihood fits\n\n"
       "- Seven models for seven series (the stable fits take about a minute); the profile log-likelihood of ν."),
    code(src(g.fit_all, g.fits_table, g.profile_nu)),
    code("fits = fit_all()\ntab = fits_table(fits)\ntab[tab.series == 'sp500'][['model', 'k', 'loglik']]"),
    code("profile_nu('sp500', save=False)"),
    md("## 3. Likelihood-ratio and Vuong tests, AIC and BIC\n\n"
       "- Nested pairs: $LR = 2(\\ell_1 - \\ell_0) \\approx \\chi^2(k_1 - k_0)$; on the boundary (Normal inside Student-t): 50:50 mixture.\n"
       "- Non-nested pairs: Vuong (1989). Information criteria and Akaike weights for every series."),
    code(src(g.lr_test, g.vuong, g.tests_all, g.fig_delta_aic)),
    code("pd.DataFrame({k: {t: round(v['lr'] if 'lr' in v else v['v'], 2) for t, v in d.items()} for k, d in tests_all(fits).items()})"),
    code("tab.pivot(index='series', columns='model', values='daic').round(1)"),
    code("fig_delta_aic(fits, save=False)"),
    md("## 4. Goodness-of-fit tests\n\n"
       "- EDF statistics of every model; parametric bootstrap p-values for the S&P 500 (199 resamples per model, a few minutes); "
       "Lilliefors critical values; power of the three tests; where KS and AD look."),
    code(src(g.edf_stats, g.gof_bootstrap, g.gof_all, g.lilliefors_crit, g.gof_power, g.fig_ks_tail)),
    code("pd.DataFrame(gof_all(fits)['sp500']).T.round(3)"),
    code("x = returns('sp500').values\n{m: gof_bootstrap(m, x, B=199)['p_boot'] for m in ['Normal', 'Student-t', 'Skewed-t']}"),
    code("pd.DataFrame({n: lilliefors_crit(n) for n in (5, 100, 1000)}).round(3)"),
    code("gof_power(save=False)"),
    code("fig_ks_tail('sp500', save=False)"),
    md("## 5. PP and QQ plots"),
    code(src(g.fig_pp_qq, g.fig_qq_assets)),
    code("fig_pp_qq(fits, 'sp500', save=False)"),
    code("fig_qq_assets(fits, save=False)"),
    md("## 6. Tails: VaR 1%, ES 2.5% and exceedances\n\n"
       "- The Kupiec (1995) test: $LR_{uc} = -2\\ln[(1-p)^{n-x}p^x / ((1-x/n)^{n-x}(x/n)^x)] \\approx \\chi^2(1)$."),
    code(src(g.kupiec, g.tail_table, g.fig_left_tail, g.fig_var_models)),
    code("tails = tail_table(fits)\npd.DataFrame({k: {m: round(v['var1'], 2) for m, v in t.items()} for k, t in tails.items()})"),
    code("pd.DataFrame({k: {m: v['kupiec']['x'] for m, v in t.items() if m != 'emp'} for k, t in tails.items()})"),
    code("fig_left_tail(fits, 'sp500', save=False)"),
    code("fig_var_models(tails, save=False)"),
    md("## 7. Out of sample, overfitting, rolling VaR\n\n"
       "- Estimation until 31 December 2019, test 2020–2026; Normal mixtures with 1–6 components; a rolling 1000-day window "
       "refitted every 20 days; the VaR reliability plot of the Quantlet SFEVaRqqplot."),
    code(src(g.pinball, g.out_of_sample, g.fig_oos, g.overfit_mixtures, g.rolling_var, g.fig_rolling_var, g.var_rma_ema, g.fig_var_qqplot)),
    code("oos = out_of_sample()\npd.DataFrame({k: {m: (round(v['logscore'], 4) if 'logscore' in v else None, v['kupiec']['x']) "
         "for m, v in o.items() if isinstance(v, dict)} for k, o in oos.items()})"),
    code("fig_oos(oos, save=False)"),
    code("of = overfit_mixtures('sp500', save=False)\npd.DataFrame(of['res']).T.round(3)"),
    code("df, res = rolling_var('sp500')\nfig_rolling_var(df, 'sp500', save=False)\npd.DataFrame({m: v for m, v in res.items() if m != 'first'}).T"),
    code("fig_var_qqplot('dax', save=False)"),
    md("## 8. Model uncertainty\n\n"
       "- Moving-block bootstrap of the AIC winner and of VaR 1% (200 resamples, several minutes); VaR averaged with the Akaike weights."),
    code(src(g.block_bootstrap, g.model_uncertainty, g.averaged_var)),
    code("model_uncertainty('sp500', save=False)"),
    code("pd.DataFrame(averaged_var(fits, tails)).T.round(2)"),
    md("## 9. AI for scientific discovery: does NIG still win after removing volatility clustering?\n\n"
       "- Open question: which candidate fits returns divided by a volatility forecast best?\n"
       "- Starter code: divide the S&P 500 returns by the EMA volatility (λ = 0.96, previous 250 days only), refit five models "
       "and compare the Akaike weights with those of the raw returns.\n"
       "- Check before trusting any answer, yours or an AI's: no look-ahead in the volatility, the same sample for all models, "
       "the parameterisation of each law."),
    code("r = returns('sp500').values / 100\n_, ema = var_rma_ema(r)\nz = r[250:] / (ema / stats.norm.ppf(0.01))\n"
         "res = {m: fit_model(m, z) for m in ['Normal', 'Student-t', 'Skewed-t', 'GED', 'NIG']}\n"
         "aic = {m: 2 * K[m] - 2 * p['loglik'] for m, p in res.items()}\n"
         "pd.Series({m: a - min(aic.values()) for m, a in aic.items()}).round(1)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [g.kupiec, g.pinball, g.lr_test, g.vuong, g.edf_stats, g.lilliefors_crit, s.ic_table, s.fit_table]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 6: Model selection and risk management\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 6: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with an interpretation question. "
       "Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- The candidate models: `fit_model`, `logpdf`, `cdf`, `ppf`, `var_es` (Normal, Student-t, skewed-t, GED, Normal mixture, NIG, stable).\n"
       "- Tests and scores: `lr_test`, `vuong`, `edf_stats` (KS, CvM, AD), `lilliefors_crit`, `kupiec`, `pinball`.\n"
       "- `ic_table` (AIC, BIC, Δ, Akaike weights) and `fit_table` (fit several models and tabulate them)."),
    code(SETUP + f"\nSEM_MODELS = {s.SEM_MODELS!r}\n\n\n" + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: four models for the S&P 500\n\n"
       "**Context.** S&P 500, the last 1000 trading days; the maximised log-likelihoods are computed below (the slides give them rounded).\n\n"
       "1. Compute AIC and BIC ($\\ln 1000 = 6.908$) of each model.\n"
       "2. Compute ΔAIC and the Akaike weights.\n"
       "3. Say which model each criterion selects and how strong the evidence is.\n\n"
       "**Report:** a table with four rows and two sentences."),
    code("# Solution\nx = returns('sp500').values[-1000:]\nll = {m: round(fit_model(m, x)['loglik'], 1) for m in ['Normal', 'Student-t', 'Skewed-t', 'GED']}\n"
         "pd.DataFrame(ic_table(ll, {m: K[m] for m in ll}, 1000)).T.round(3)"),
    md("**Interpretation of the result.** AIC finds the Student-t and the skewed-t tied (Δ ≤ 2); BIC prefers the Student-t, "
       "which has one parameter less."),
    md("## A2 [Proposed]: four models for Bitcoin\n\n"
       "**Context.** Bitcoin, the last 1000 days; models Normal, Student-t, NIG, Normal mixture. Model: A1.\n\n"
       "1. Compute AIC, BIC, ΔAIC and the Akaike weights.\n"
       "2. Say whether AIC and BIC select the same model.\n"
       "3. Explain why the Normal mixture, with the most parameters, does not win.\n\n"
       "**Report:** a table with four rows and two sentences."),
    code("# Solution\nx = returns('btc').values[-1000:]\nll = {m: round(fit_model(m, x)['loglik'], 1) for m in ['Normal', 'Student-t', 'NIG', 'Mixture']}\n"
         "pd.DataFrame(ic_table(ll, {m: K[m] for m in ll}, 1000)).T.round(3)"),
    md("## A3 [Solved]: is skewness needed? Is the Normal law enough?\n\n"
       "**Context.** The log-likelihoods of A1.\n\n"
       "1. Test the Student-t ($H_0$: λ = 0) against the skewed-t with a likelihood-ratio test at 5%.\n"
       "2. Test the Normal law ($H_0$: β = 2) against the GED.\n"
       "3. Compare the LR decision of step 1 with the AIC verdict of A1.\n\n"
       "**Report:** two LR values, two decisions and one sentence."),
    code("# Solution\nx = returns('sp500').values[-1000:]\nll = {m: round(fit_model(m, x)['loglik'], 1) for m in ['Normal', 'Student-t', 'Skewed-t', 'GED']}\n"
         "print('t vs skewed-t:', lr_test(ll['Student-t'], ll['Skewed-t'], 1))\nprint('Normal vs GED:', lr_test(ll['Normal'], ll['GED'], 1))"),
    md("**Interpretation of the result.** The skewness parameter is not significant, in line with the AIC tie; the Normal law is rejected."),
    md("## A4 [Proposed]: a test on the boundary\n\n"
       "**Context.** The log-likelihoods of A2. Model: A3.\n\n"
       "1. Compute LR for the Normal law inside the Student-t.\n"
       "2. Explain why the null law is not χ²(1), and give the correct 5% critical value.\n"
       "3. Say whether an LR test can compare the Student-t with the NIG.\n\n"
       "**Report:** one LR value, one critical value and two sentences."),
    code("# Solution\nx = returns('btc').values[-1000:]\nll = {m: round(fit_model(m, x)['loglik'], 1) for m in ['Normal', 'Student-t']}\n"
         "print(lr_test(ll['Normal'], ll['Student-t'], 1, boundary=True), 'critical value', round(stats.chi2.ppf(0.90, 1), 3))"),
    md("## A5 [Solved]: KS on five returns\n\n"
       "**Context.** Daily returns (%): −2.1, −0.4, 0.3, 0.9, 1.6.\n\n"
       "1. Compute $D_5$ against N(0, 1) and compare it with the exact 5% critical value.\n"
       "2. Estimate μ and σ (divisor n), compute $D_5$ against $N(\\hat\\mu, \\hat\\sigma^2)$ and compare it with the Lilliefors value.\n"
       "3. Explain why the Lilliefors critical value is smaller.\n\n"
       "**Report:** two values of $D_5$, two decisions, one sentence."),
    code("# Solution\nx = np.array([-2.1, -0.4, 0.3, 0.9, 1.6])\nprint('known F :', round(edf_stats(stats.norm.cdf(x))['ks'], 3), ' critical', round(stats.kstwo.ppf(0.95, 5), 3))\n"
         "print('estimated:', round(edf_stats(stats.norm.cdf(x, x.mean(), x.std()))['ks'], 3), ' Lilliefors', round(lilliefors_crit(5, reps=20000)['ks'], 3))"),
    md("**Interpretation of the result.** Neither test rejects: five observations carry almost no information about the shape."),
    md("## A6 [Proposed]: KS with one large loss\n\n"
       "**Context.** Daily returns (%): −3.0, −0.6, −0.2, 0.1, 0.5, 1.0. Model: A5.\n\n"
       "1. Compute $D_6$ against N(0, 1).\n"
       "2. Compute $D_6$ against $N(\\hat\\mu, \\hat\\sigma^2)$.\n"
       "3. Explain why KS hardly reacts to the −3% day, while AD would.\n\n"
       "**Report:** two values of $D_6$, two decisions, one sentence."),
    code("# Solution\nx = np.array([-3.0, -0.6, -0.2, 0.1, 0.5, 1.0])\nprint(edf_stats(stats.norm.cdf(x)), round(stats.kstwo.ppf(0.95, 6), 3))\n"
         "print(edf_stats(stats.norm.cdf(x, x.mean(), x.std())), round(lilliefors_crit(6, reps=20000)['ks'], 3))"),
    md("## A7 [Solved]: VaR 1% under two models\n\n"
       "**Context.** Normal with μ = 0.05, σ = 1.2; Student-t with ν = 4, location 0.06, scale 0.8.\n\n"
       "1. Compute VaR 1% under each model.\n"
       "2. A bank uses the Normal VaR and observes 20 exceedances in 1000 days; compute the Kupiec statistic.\n"
       "3. Decide at 5% and interpret.\n\n"
       "**Report:** two VaR values, one statistic, one decision."),
    code("# Solution\nprint('Normal VaR 1%:', round(-(0.05 + 1.2 * stats.norm.ppf(0.01)), 3), ' Student-t VaR 1%:', round(-(0.06 + 0.8 * stats.t.ppf(0.01, 4)), 3))\n"
         "print(kupiec(20, 1000))"),
    md("**Interpretation of the result.** Twice the expected number of exceedances: the Normal VaR is too low."),
    md("## A8 [Proposed]: too few exceedances, and ES\n\n"
       "**Context.** Another bank observes 4 exceedances of its VaR 1% in 1000 days. Model: A7.\n\n"
       "1. Compute the Kupiec statistic and decide at 5%.\n"
       "2. Explain why too few exceedances is also a failure.\n"
       "3. Compute ES 2.5% of the Normal model of A7.\n\n"
       "**Report:** one statistic, one decision, one ES value and one sentence."),
    code("# Solution\nprint(kupiec(4, 1000))\nprint('ES 2.5%:', round(-0.05 + 1.2 * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025, 3))"),
    # ---------------- Part B
    md("# Part B: real data, precision and interpretation"),
    md("## B1 [Solved]: which distribution for the BET?\n\n"
       "**Question.** Which of the six candidates fits the daily BET returns best, and is the skewness parameter needed?\n\n"
       "1. Fit the six candidates by maximum likelihood and report $\\ell(\\hat\\theta)$ of each.\n"
       "2. Compute AIC, BIC, ΔAIC and the Akaike weights.\n"
       "3. Test the Student-t against the skewed-t with a likelihood-ratio test.\n"
       "4. Interpretation: what do the winner and the LR test say about the tails and the skewness of the BET?\n\n"
       "**Report:** a table with six rows, one LR test and two sentences."),
    code("# Solution\n" + src(s.b1_ranking) + "\n\n\nb = b1_ranking('bet', save=False)\nprint(b['best_aic'], b['best_bic'], b['lr_t_skt'])\npd.DataFrame(b['tab']).T.round(3)"),
    md("**Interpretation of the result.** NIG wins on both criteria; the skewness parameter of the skewed-t is not significant; "
       "the Normal law is thousands of AIC points behind."),
    md("## B2 [Proposed]: DAX and Bitcoin\n\n"
       "**Question.** Do the DAX and Bitcoin choose the same distribution as the BET? Model: B1.\n\n"
       "1. Repeat steps 1–3 of B1 for each series.\n"
       "2. Say whether AIC and BIC agree for each series.\n"
       "3. Interpretation: why might the best model differ between a stock index and a crypto asset?\n\n"
       "**Report:** two tables and three sentences."),
    code("# Solution\nfor k in ['dax', 'btc']:\n    b = b1_ranking(k, save=False)\n    print(k, b['best_aic'], b['best_bic'], b['lr_t_skt'])\n"
         "    print(pd.DataFrame(b['tab']).T[['daic', 'dbic', 'w']].round(3))"),
    md("## B3 [Solved]: do the best models pass?\n\n"
       "**Question.** Do the Normal and Student-t fits of the S&P 500 pass the KS, Cramér–von Mises and Anderson–Darling tests?\n\n"
       "1. Fit both models and compute the three statistics.\n"
       "2. Compute the naive KS p-value, which treats the parameters as known.\n"
       "3. Compute parametric bootstrap p-values with B = 99 (simulate, refit, recompute).\n"
       "4. Draw the QQ plot of the data against both fits.\n"
       "5. Interpretation: if both models are rejected, what is the test still good for?\n\n"
       "**Report:** a table of six statistics with p-values, the QQ plot and two sentences."),
    code("# Solution\n" + src(s.b3_gof) + "\n\n\nb3_gof('sp500', save=False)"),
    md("**Interpretation of the result.** With more than 6000 days every i.i.d. model is rejected; the statistics still rank "
       "the models, and the QQ plot shows where each one fails."),
    md("## B4 [Proposed]: crash or data error? Banca Transilvania\n\n"
       "**Question.** Which of the largest TLV returns are real, and does removing the wrong ones change the chosen model? Model: B1.\n\n"
       "1. List the three largest absolute returns with their dates.\n"
       "2. For each, compare the close and the adjusted close around the date, and the BET return of the same day.\n"
       "3. Decide which days are data errors and remove only those.\n"
       "4. Refit five candidates; compare the kurtosis, the Student-t ν̂, the AD statistic and the AIC winner before and after.\n"
       "5. Interpretation: what would have happened if you had removed all three days?\n\n"
       "**Report:** a table of three days with your verdict, a before/after table and two sentences."),
    code("# Solution\n" + src(s.b4_tlv) + "\n\n\nb = b4_tlv()\nprint(b['top'], b['bet_20181219'])\nprint(pd.DataFrame(b['prices']).T)\n"
         "pd.DataFrame({lab: {c: b[lab][c] for c in ['n', 'kurt', 'sd', 'nu', 'ad_t', 'best_aic']} for lab in ['raw', 'clean']})"),
    md("## B5 [Solved]: VaR 1% out of sample for the DAX\n\n"
       "**Question.** Does the model chosen by AIC on 2000–2019 also give the best VaR 1% on 2020–2026?\n\n"
       "1. Fit the six candidates on the estimation days and rank them by AIC.\n"
       "2. Compute each model's mean log score on the test days.\n"
       "3. Compute each model's VaR 1%, count its exceedances on the test days and run the Kupiec test.\n"
       "4. Add historical simulation and the VaR averaged with the Akaike weights.\n"
       "5. Interpretation: which model would you choose for VaR 1%, and why?\n\n"
       "**Report:** one table, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_oos) + "\n\n\nb = b5_oos('dax', save=False)\nprint(b['best_aic'], b['best_logscore'], b['best_pinball'])\n"
         "pd.DataFrame({m: {'logscore': b[m].get('logscore'), 'var1': b[m]['var1'], 'x': b[m]['kupiec']['x'], 'p': b[m]['kupiec']['p'], "
         "'pinball': b[m]['pinball']} for m in SEM_MODELS + ['Historical', 'Averaged']}).T.round(4)"),
    md("**Interpretation of the result.** The AIC winner gives the best density but too few exceedances; for VaR 1% alone "
       "the GED and the Student-t were closer to the 1% target."),
    md("## B6 [Proposed]: VaR 1% out of sample for Bitcoin\n\n"
       "**Question.** Does the B5 conclusion hold for Bitcoin? Model: B5.\n\n"
       "1. Repeat steps 1–4 of B5.\n"
       "2. Compare the empirical VaR 1% of the test days with that of the estimation days.\n"
       "3. Interpretation: why do all heavy-tailed models fail the Kupiec test here, and in which direction?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb = b5_oos('btc', save=False)\nprint(b['best_aic'], b['best_logscore'], b['best_pinball'], 'test VaR 1%:', round(b['test_var1'], 2))\n"
         "pd.DataFrame({m: {'var1': b[m]['var1'], 'x': b[m]['kupiec']['x'], 'p': b[m]['kupiec']['p']} for m in SEM_MODELS + ['Historical', 'Averaged']}).T.round(4)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: does the best model change over time?\n\n"
       "**Question.** Is the AIC winner of the S&P 500 and the BET the same in every two-year window? Models: B1, A1.\n\n"
       "1. In each window, find the AIC winner, its Akaike weight and the ΔAIC of the runner-up.\n"
       "2. Count how often each model wins, for each index.\n"
       "3. Propose one explanation for the changes and one way to test it.\n"
       "4. Interpretation: should a risk model be re-selected every two years?\n\n"
       "**Report:** a table, a count of winners and a plan for a project."),
    code("# Reference analysis\n" + src(s.c1_windows) + "\n\n\nres = c1_windows()\n"
         "pd.DataFrame({(k, p): v for k, a in res.items() for p, v in a.items()}).T"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant chose a distribution for the daily S&P 500 log returns (%) since 2000 and claimed:\n\n"
       "- (a) the skewed-t has the lowest AIC, so it is significantly better than the Student-t;\n"
       f"- (b) the KS test of the Student-t fit with `scipy.stats.kstest` gives p = {C2['p_naive_t']:.3f}, so the Student-t fits at the 1% level;\n"
       "- (c) the LR test of Normal vs Student-t uses the χ²(1) critical value 3.84;\n"
       f"- (d) BIC penalises extra parameters less than AIC, because ln(n) is only {C2['lnn']:.2f};\n"
       "- (e) the Anderson–Darling test is more sensitive than KS to the fit in the tails;\n"
       "- (f) since the skewed-t has the best fit, its VaR 1% is the most accurate one.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\nc = c2_check()\n{k: v for k, v in c.items() if k != 'tab'}"),
]

if __name__ == '__main__':
    build(LECTURE, 6, 'lecture')
    build(SEMINAR, 6, 'seminar')
