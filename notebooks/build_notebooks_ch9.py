"""
build_notebooks_ch9.py -- lecture and seminar notebooks of Chapter 9 (SFM): ARCH and GARCH models and their extensions
=====================================================================================================================
Output: notebooks/EN/chapter9_lecture_notebook.ipynb, notebooks/EN/chapter9_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_09/generate_all_charts.py and seminar9.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Both notebooks install the arch package first (not part of Colab).
Run:  python3 notebooks/build_notebooks_ch9.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter9_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter9_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 9
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(9)
import generate_all_charts as g   # noqa: E402
import seminar9 as s              # noqa: E402

INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
CONSTS = ('import statsmodels.api as sm\nfrom statsmodels.stats.diagnostic import acorr_ljungbox, het_arch\n'
          'from arch import arch_model\n'
          f'NAME = {g.NAME!r}\nSTART = {g.START!r}\nASSETS = {g.ASSETS!r}\nBAD_DAYS = {g.BAD_DAYS!r}\nCOLORS = {g.COLORS!r}\n'
          f'EPISODES = {g.EPISODES!r}\nOOS_START = {g.OOS_START!r}\nREFIT = {g.REFIT!r}\nLAMBDA = {g.LAMBDA!r}\n'
          f'ALPHA_VAR = {g.ALPHA_VAR!r}\nSEED = {g.SEED!r}\nTERM_DATES = {g.TERM_DATES!r}')
CORE = [g.returns, g.fit, g.persistence, g.half_life, g.summary]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 9: ARCH and GARCH models\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Volatility clustering in the S&P 500 since 2000; the 2008 and 2020 episodes.\n"
       "- Simulated ARCH(1) and GARCH(1,1) paths (SFEtimegarch); the likelihood of ARCH(1) and GARCH(1,1) "
       "(SFElikarch1, SFElikgarch).\n"
       "- Maximum-likelihood estimation step by step and with the `arch` package (SFEgarchest); robust standard errors; "
       "Student-t and skewed-t innovations.\n"
       "- GARCH(1,1)-t in six markets; conditional volatility (SFEvolgarchest); persistence and half-life.\n"
       "- Asymmetry: GJR-GARCH, EGARCH, the news impact curve (SFENewsImpactCurve); diagnostics; AIC and BIC.\n"
       "- Multi-step forecasts; out-of-sample comparison with EWMA (QLIKE, Diebold–Mariano); VaR 1%.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 13; "
       "Tsay (2010), *Analysis of Financial Time Series*, 3rd ed., Ch. 3; Engle (1982); Bollerslev (1986)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; S&P 500, DAX and BET since 2000, Bitcoin since 2014, "
       "Banca Transilvania and OMV Petrom since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Model: $r_t = \\mu + \\varepsilon_t$, $\\varepsilon_t = \\sigma_t z_t$, $z_t$ i.i.d. with mean 0 and variance 1.\n"
       "- GARCH(1,1): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$; persistence $\\alpha + \\beta$; "
       "long-run variance $\\omega/(1 - \\alpha - \\beta)$; half-life $\\ln 0.5/\\ln(\\alpha + \\beta)$.\n"
       "- `fit(r, vol, dist, o)`: maximum likelihood with the `arch` package (robust standard errors by default)."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Volatility comes in bursts\n\n- S&P 500 daily returns since 2000 with the 2008 and 2020 episodes."),
    code(src(g.fig_returns)),
    code("fig_returns(save=False)"),
    md("## 2. Simulated ARCH(1) and GARCH(1,1) paths\n\n"
       "- Three series with unconditional variance 1: i.i.d. Normal, ARCH(1) and GARCH(1,1); their sample kurtosis."),
    code(src(g.simulate_garch, g.fig_simulated)),
    code("fig_simulated(save=False)"),
    md("## 3. The likelihood\n\n"
       "- ARCH(1): $\\ell(\\alpha)$ with $\\omega = 1 - \\alpha$ for $n = 100$ and $n = 1000$ (SFElikarch1).\n"
       "- GARCH(1,1): contours of $\\ell(\\alpha, \\beta)$ with variance targeting, $n = 500$ (SFElikgarch) and $n = 2000$."),
    code(src(g.arch1_loglik, g.fig_lik_arch1, g.garch_loglik_vt, g.fig_lik_garch)),
    code("fig_lik_arch1(save=False)"),
    code("fig_lik_garch(save=False)"),
    md("## 4. Estimation: ARCH(q) against GARCH(1,1); step by step against the `arch` package\n\n"
       "- ARCH(1), ARCH(5), ARCH(10) and GARCH(1,1) with Normal innovations: log-likelihood, AIC, BIC.\n"
       "- The Gaussian GARCH(1,1) estimated by minimising $-\\ell(\\theta)$ with scipy, and with `arch`; classic and robust "
       "standard errors.\n"
       "- Normal, Student-t and skewed-t innovations; QQ plots of the standardised residuals."),
    code(src(g.arch_q_table, g.garch11_negloglik, g.fit_step_by_step, g.estimation_sp500, g.fig_qq)),
    code("pd.DataFrame(arch_q_table()).T.round(3)"),
    code("est = estimation_sp500()\n"
         "pd.DataFrame({'step by step': est['step']['params'], 'SE (step by step)': est['step']['se'], 'arch': est['normal']['params'], "
         "'classic SE': est['normal']['se_classic'], 'robust SE': est['normal']['se']}).round(4)"),
    code("pd.DataFrame({d: {**est[d]['params'], 'loglik': est[d]['loglik'], 'persistence': est[d]['pers']} "
         "for d in ['normal', 't', 'skewt']}).round(4)"),
    code("fig_qq(save=False)"),
    md("## 5. GARCH(1,1)-t in six markets\n\n"
       "- Parameters, persistence, half-life, long-run and sample volatility; Bitcoin is estimated on the boundary "
       "$\\alpha + \\beta = 1$ (IGARCH).\n"
       "- Annualised conditional volatility (SFEvolgarchest) and the decay of a variance shock."),
    code(src(g.markets_table, g.fig_vol, g.fig_persistence)),
    code("T = markets_table()\n"
         "pd.DataFrame({NAME[k]: {**v['params'], 'persistence': v['pers'], 'half-life': v['hl'], 'long-run vol': v.get('vol_lr'), "
         "'sample vol': v['vol_sample']} for k, v in T.items()}).T.round(4)"),
    code("fig_vol(('sp500', 'dax'), 'sfm_ch9_vol_sp500_dax', save=False)"),
    code("fig_vol(('bet', 'btc'), 'sfm_ch9_vol_bet_btc', save=False)"),
    code("fig_persistence(T, save=False)"),
    md("## 6. Asymmetry: GJR-GARCH, EGARCH and the news impact curve\n\n"
       "- News impact curves of GARCH, GJR and EGARCH (Student-t) and a kernel estimate of $E[r_t^2 \\mid r_{t-1}]$.\n"
       "- GJR $\\gamma$, the likelihood-ratio test against GARCH, EGARCH $\\gamma$, the sign-bias test."),
    code(src(g.news_impact, g.kernel_nic, g.fig_nic, g.sign_bias_test, g.asym_table)),
    code("fig_nic(save=False)"),
    code("at = asym_table()\npd.DataFrame({NAME[k]: {c: v[c] for c in ['gjr_alpha', 'gjr_gamma', 'gjr_t', 'lr', 'lr_p', 'eg_gamma', 'eg_t']} "
         "for k, v in at.items()}).T.round(3)"),
    md("## 7. Diagnostics and model selection\n\n"
       "- Ljung–Box and ARCH-LM tests on the standardised residuals; the ACF of $r_t^2$ against that of $\\hat z_t^2$.\n"
       "- Nine models compared by AIC and BIC."),
    code(src(g.ljung_box, g.diagnostics, g.diag_markets, g.fig_acf_diag, g.model_selection)),
    code("diagnostics()"),
    code("fig_acf_diag(save=False)"),
    code("model_selection().round(1)"),
    md("## 8. Forecasts and the term structure of volatility\n\n"
       "- $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; forecasts made on a "
       "calm day, at the COVID-19 peak and on the last day."),
    code(src(g.garch_path_forecast, g.fig_term_structure)),
    code("fig_term_structure(save=False)"),
    md("## 9. Out-of-sample comparison with EWMA, and VaR 1%\n\n"
       "- One-day forecasts since 2015: GARCH-t and GJR-t re-estimated every 250 days, EWMA with $\\lambda = 0.94$.\n"
       "- QLIKE loss $r_t^2/h_t + \\ln h_t$ (Patton, 2011) and the Diebold–Mariano test; VaR 1% exceedances "
       "(a preview of Chapter 10)."),
    code(src(g.ewma_variance, g.oos_forecasts, g.qlike, g.dm_test, g.forecast_eval, g.fig_forecast_eval, g.fig_var)),
    code("fe, frames = forecast_eval()\n"
         "pd.DataFrame({NAME[k]: {**{f'QLIKE {m}': v['qlike'][m] for m in v['qlike']}, 'DM t': v['dm_garch_ewma']['t'], "
         "'DM p': v['dm_garch_ewma']['p'], 'exceed. GARCH-t': v['exc_garch'], 'exceed. EWMA': v['exc_ewma']} for k, v in fe.items()}).T.round(4)"),
    code("fig_forecast_eval(frames, save=False)"),
    code("fig_var(frames, save=False)"),
    md("## 10. AI for scientific discovery: is Bitcoin volatility becoming equity-like?\n\n"
       "- Open question: has Bitcoin volatility moved closer to the volatility of the S&P 500 since 2021?\n"
       "- Starter result: GJR-GARCH-t on 2015–2019 and 2021–2026 for both series.\n"
       "- Check before trusting any answer, yours or an AI's: returns in %, the GJR persistence $\\alpha + \\beta + \\gamma/2$, "
       "estimates on the boundary, different calendars, overlapping rolling windows."),
    code("for a, b in [('2015-01-01', '2019-12-31'), ('2021-01-01', '2026-09-18')]:\n"
         "    for k in ('btc', 'sp500'):\n"
         "        res = fit(returns(k).loc[a:b], o=1, dist='t')\n"
         "        print(a[:4], b[:4], NAME[k], {n: round(v, 3) for n, v in res.params.items()}, 'persistence', round(persistence(res), 4))"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [g.news_impact, g.sign_bias_test, g.ljung_box, g.ewma_variance, g.oos_forecasts, g.qlike, g.dm_test]
SEM_HELPERS = [s.garch_algebra, s.garch_forecasts, s.arch1_likelihood, s.asym_news]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 9: ARCH and GARCH models\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 9: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n"
       "- Returns; `fit` (maximum likelihood with `arch`), persistence, half-life, a summary of a fit; the news impact "
       "curve; the sign-bias test; Ljung–Box; EWMA; out-of-sample forecasts; QLIKE and the Diebold–Mariano test.\n"
       "- Helpers for Part A: GARCH(1,1) algebra, multi-step forecasts and VaR, the ARCH(1) likelihood step by step, "
       "GJR and EGARCH news impact."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: GARCH(1,1) algebra\n\n"
       "**Context.** Daily returns in %: $\\omega = 0.02$, $\\alpha = 0.08$, $\\beta = 0.90$, $\\mu = 0$, 252 trading days per year.\n\n"
       "1. Compute the persistence and the long-run variance.\n"
       "2. Compute the annualised long-run volatility.\n"
       "3. Compute the half-life of a variance shock.\n"
       "4. Today $\\sigma_t^2 = 1.5$ and $r_t = -3\\%$; compute $\\sigma_{t+1}^2$ and $\\sigma_{t+1}$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\ngarch_algebra(0.02, 0.08, 0.90, 1.5, -3.0)"),
    md("**Interpretation of the result.** A fall of 3% raises the variance by about 40%; half of the excess is gone after "
       "about 34 trading days."),
    md("## A2 [Proposed]: a more reactive market\n\n"
       "**Context.** $\\omega = 0.05$, $\\alpha = 0.12$, $\\beta = 0.85$, $\\mu = 0$; today $\\sigma_t^2 = 2.0$ and $r_t = +1.5\\%$. Model: A1.\n\n"
       "1. Compute the persistence, the long-run variance and the annualised long-run volatility.\n"
       "2. Compute the half-life.\n"
       "3. Compute $\\sigma_{t+1}^2$.\n"
       "4. Write EWMA with $\\lambda = 0.94$ as a GARCH(1,1) and say which of the quantities in 1–2 exist for it.\n\n"
       "**Report:** four numbers and two sentences."),
    code("# Solution\ngarch_algebra(0.05, 0.12, 0.85, 2.0, 1.5)"),
    md("## A3 [Solved]: forecasts and VaR 1%\n\n"
       "**Context.** The model of A1, with $\\sigma_{t+1}^2 = 2.09$ and Normal innovations.\n\n"
       "1. Compute $E_t[\\sigma_{t+5}^2]$ and $E_t[\\sigma_{t+10}^2]$.\n"
       "2. Compute the 10-day variance and volatility.\n"
       "3. Compare with the square-root-of-time rule $\\sqrt{10}\\,\\sigma_{t+1}$ and with $\\sqrt{10\\,\\bar\\sigma^2}$.\n"
       "4. Compute the one-day and the 10-day VaR 1%.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\ngarch_forecasts(0.02, 0.08, 0.90, 2.09)"),
    md("**Interpretation of the result.** The high variance decays slowly: the 10-day risk lies between the two simple "
       "rules, closer to the square-root-of-time rule."),
    md("## A4 [Proposed]: forecasts with Student-t innovations\n\n"
       "**Context.** The model of A2, with $\\sigma_{t+1}^2 = 2.02$ and standardised Student-t innovations with $\\nu = 5$. Model: A3.\n\n"
       "1. Compute $E_t[\\sigma_{t+10}^2]$ and the 10-day variance.\n"
       "2. Compute the quantile $q_{0.01}$ of the standardised t.\n"
       "3. Compute the one-day and the 10-day VaR 1% with t and with Normal innovations.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nprint(garch_forecasts(0.05, 0.12, 0.85, 2.02, nu=5))\nprint(garch_forecasts(0.05, 0.12, 0.85, 2.02))"),
    md("## A5 [Solved]: the ARCH(1) likelihood, step by step\n\n"
       "**Context.** ARCH(1) with $\\omega = 0.5$, $\\alpha = 0.5$, $\\mu = 0$, Normal $z_t$; observed $r_1, \\dots, r_4 = 0.5, -1.2, 2.0, -0.3$.\n\n"
       "1. Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$.\n"
       "2. Compute $\\ell_2$, $\\ell_3$, $\\ell_4$ and the conditional log-likelihood.\n"
       "3. Compute the unconditional variance and the kurtosis of this ARCH(1).\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\narch1_likelihood(0.5, 0.5, [0.5, -1.2, 2.0, -0.3])"),
    md("**Interpretation of the result.** Day 3 (a return of 2 after a calm day) contributes least to the likelihood: a "
       "large return when the conditional variance is small is unlikely under the model."),
    md("## A6 [Proposed]: good and bad news\n\n"
       "**Context.** GJR-GARCH: $\\omega = 0.02$, $\\alpha = 0.03$, $\\gamma = 0.12$, $\\beta = 0.90$; EGARCH: $\\omega = 0$, "
       "$\\alpha = 0.12$, $\\gamma = -0.08$, $\\beta = 0.98$; in both $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.\n\n"
       "1. Compute the GJR $\\sigma_{t+1}^2$ after $\\varepsilon_t = -2$ and after $\\varepsilon_t = +2$, and their ratio.\n"
       "2. Compute the GJR persistence and half-life.\n"
       "3. Compute the EGARCH $\\sigma_{t+1}^2$ for the same two shocks.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nasym_news(0.02, 0.03, 0.12, 0.90, 1.0, eg=(0.0, 0.12, -0.08, 0.98))"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: GARCH(1,1)-t for the DAX\n\n"
       "**Question.** How persistent is DAX volatility, and how far is the DAX from its long-run volatility today?\n\n"
       "1. Estimate GARCH(1,1) with Student-t innovations and report the parameters with robust standard errors.\n"
       "2. Compute the persistence, the half-life and the annualised long-run volatility, and compare it with the sample volatility.\n"
       "3. Report the peaks of the annualised conditional volatility in 2008 and in 2020, and its value on the last day.\n"
       "4. Draw the returns since 2018 with $\\pm 2\\hat\\sigma_t$ bands.\n"
       "5. Interpretation: should a risk manager use the long-run volatility or the current one for tomorrow?\n\n"
       "**Report:** a table of five parameters, five numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b1_estimate) + "\n\n\nb1 = b1_estimate('dax', save=False)\n"
         "print(pd.DataFrame({'estimate': b1['params'], 'robust SE': b1['se']}).round(4))\n"
         "{c: b1[c] for c in ['pers', 'hl', 'vol_lr', 'vol_sample', 'peak2008', 'peak2020', 'last_vol', 'share_out']}"),
    md("**Interpretation of the result.** For tomorrow the current conditional volatility matters; the long-run level "
       "matters only for long horizons, and it is imprecise because $1 - \\alpha - \\beta$ is small."),
    md("## B2 [Proposed]: BET, Banca Transilvania and Bitcoin\n\n"
       "**Question.** Which of the BET, Banca Transilvania and Bitcoin has the most persistent volatility? Model: B1.\n\n"
       "1. Estimate GARCH(1,1)-t for each series and report $\\alpha$, $\\beta$ and $\\nu$.\n"
       "2. Compute the persistence, the half-life and the long-run volatility where they exist.\n"
       "3. Compare the long-run with the sample volatility (annualise Bitcoin with 365 days).\n"
       "4. Interpretation: what does an estimate $\\hat\\alpha + \\hat\\beta = 1$ tell a risk manager?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = {k: b1_estimate(k, save=False) for k in ('bet', 'tlv', 'btc')}\n"
         "pd.DataFrame({NAME[k]: {**{c: d['params'][c] for c in ['alpha[1]', 'beta[1]', 'nu']}, 'persistence': d['pers'], "
         "'half-life': d['hl'], 'long-run vol': d['vol_lr'], 'sample vol': d['vol_sample']} for k, d in b2.items()}).T.round(4)"),
    md("## B3 [Solved]: the leverage effect in the S&P 500\n\n"
       "**Question.** Do falls raise the volatility of the S&P 500 more than rises of the same size?\n\n"
       "1. Estimate GARCH(1,1)-t and GJR-GARCH(1,1)-t; report $\\gamma$ with its robust $t$ statistic.\n"
       "2. Compute the LR statistic of GJR against GARCH, its p-value and the two BIC values.\n"
       "3. Run the sign-bias test on the standardised residuals of GARCH-t.\n"
       "4. Draw both news impact curves and compute the GJR variance after shocks of −2% and +2%.\n"
       "5. Interpretation: how does the leverage effect change the VaR on the day after a fall?\n\n"
       "**Report:** four statistics with p-values, two variances, the chart and two sentences."),
    code("# Solution\n" + src(s.b3_asymmetry) + "\n\n\nb3 = b3_asymmetry('sp500', save=False)\nb3"),
    md("**Interpretation of the result.** Only falls raise the variance ($\\hat\\alpha = 0$ in GJR); after a fall the "
       "VaR is about $\\sqrt{1.6}$ times the VaR after an equal rise; a symmetric GARCH understates risk after bad days."),
    md("## B4 [Proposed]: asymmetry on the BVB and in Bitcoin\n\n"
       "**Question.** Is there a leverage effect in the BET, in Banca Transilvania and in Bitcoin? Model: B3.\n\n"
       "1. Estimate GJR-GARCH(1,1)-t and EGARCH(1,1)-t; report both $\\gamma$ with robust $t$ statistics.\n"
       "2. Compute the LR test of GJR against GARCH and the change in BIC.\n"
       "3. Compute the GJR ratio of the variances after −2% and +2%.\n"
       "4. Interpretation: would you use an asymmetric model for each of the three? Why?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b4_asymmetry) + "\n\n\nb4 = b4_asymmetry()\n"
         "pd.DataFrame({NAME[k]: {c: d[c] for c in ['gamma', 't_gamma', 'lr', 'p_lr', 'bic_garch', 'bic_gjr', 'eg_gamma', 'eg_t', 'ratio']} "
         "for k, d in b4.items()}).T.round(3)"),
    md("## B5 [Solved]: GARCH against EWMA for the S&P 500\n\n"
       "**Question.** Does GARCH(1,1)-t forecast tomorrow's variance of the S&P 500 better than EWMA, and is its VaR 1% "
       "exceeded on 1% of days?\n\n"
       "1. Re-estimate GARCH(1,1)-t every 250 days on all data up to that day and compute the one-day forecasts; compute EWMA "
       "with $\\lambda = 0.94$.\n"
       "2. Compute the mean QLIKE of both models and the Diebold–Mariano $t$ statistic.\n"
       "3. Count the exceedances of the GARCH-t VaR 1% and of the EWMA-Normal VaR 1% and compare with the expected number.\n"
       "4. Draw the cumulative difference of the losses.\n"
       "5. Interpretation: which model would you give to a risk manager, and what would you still fix?\n\n"
       "**Report:** four numbers, two counts, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_oos) + "\n\n\nb5 = b5_oos('sp500', save=False)\nb5"),
    md("**Interpretation of the result.** GARCH-t has the smaller QLIKE (significant by Diebold–Mariano), but both VaR "
       "models are exceeded more often than on 1% of the days: add the leverage effect (B3) and test the exceedances "
       "formally (Chapter 10)."),
    md("## B6 [Proposed]: GARCH against EWMA for the DAX and the BET\n\n"
       "**Question.** Does the result of B5 hold for the DAX and for the BET? Model: B5.\n\n"
       "1. Compute the out-of-sample GARCH(1,1)-t and EWMA forecasts as in B5.\n"
       "2. Compute the mean QLIKE of both models and the DM test.\n"
       "3. Count the VaR 1% exceedances of both models.\n"
       "4. Interpretation: for which index does GARCH help most, and why?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6 = {k: b5_oos(k, save=False) for k in ('dax', 'bet')}\n"
         "pd.DataFrame({NAME[k]: {'QLIKE GARCH-t': d['q_garch'], 'QLIKE EWMA': d['q_ewma'], 'DM t': d['dm']['t'], 'DM p': d['dm']['p'], "
         "'expected': d['expected'], 'exceed. GARCH-t': d['nexc_garch'], 'exceed. EWMA': d['nexc_ewma']} for k, d in b6.items()}).T.round(4)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: is Bitcoin volatility becoming equity-like?\n\n"
       "**Question.** Has Bitcoin volatility moved closer to the volatility of the S&P 500 since 2021? Models: B1, B3.\n\n"
       "1. Estimate GJR-GARCH(1,1)-t for both series in 2015–2019 and in 2021–2026 and report the persistence, $\\gamma$ and $\\nu$.\n"
       "2. Compute the correlation of the logarithms of the two GARCH-t volatilities on common days in each period.\n"
       "3. List the events of 2020–2024 that could explain a change.\n"
       "4. Interpretation: how would you separate a lasting change from a change caused by one crisis?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\n" + src(s.c1_bitcoin) + "\n\n\nc1 = c1_bitcoin()\nc1"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant interpreted a GARCH(1,1)-t fit of the S&P 500 (daily, since 2000). It answered:\n\n"
       "- (a) $\\alpha + \\beta$ is close to 1, so a volatility shock disappears within about a week;\n"
       "- (b) the long-run variance is $\\omega/(1 - \\alpha)$;\n"
       "- (c) a GJR fit gives $\\gamma > 0$: good news raises volatility more than bad news;\n"
       "- (d) the Ljung–Box test of the squared standardised residuals does not reject, so the GARCH-t model is correct;\n"
       "- (e) with Student-t innovations the one-day VaR 1% is 2.326 times $\\sigma_{t+1}$;\n"
       "- (f) EWMA with $\\lambda = 0.94$ is a GARCH(1,1) with $\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\nc2 = c2_check()\nc2"),
]

if __name__ == '__main__':
    build(LECTURE, 9, 'lecture')
    build(SEMINAR, 9, 'seminar')
