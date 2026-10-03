"""
build_notebooks_ch8.py -- lecture and seminar notebooks of Chapter 8 (SFM): volatility estimators and volatility
clustering
================================================================================================================
Output: notebooks/EN/chapter8_lecture_notebook.ipynb, notebooks/EN/chapter8_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_08/generate_all_charts.py and seminar8.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch8.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter8_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter8_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 8
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(8)
import generate_all_charts as g   # noqa: E402
import seminar8 as s              # noqa: E402

CONSTS = (f'NAME = {g.NAME!r}\nSTART = {g.START!r}\nOHLC_START = {g.OHLC_START!r}\nBAD_DAYS = {g.BAD_DAYS!r}\n'
          f'ASSETS = {g.ASSETS!r}\nSTOCKS = {g.STOCKS!r}\nBVB = {g.BVB!r}\nOHLC_ASSETS = {g.OHLC_ASSETS!r}\n'
          f'COLORS = {g.COLORS!r}\nEST_COLORS = {g.EST_COLORS!r}\nEST_LABEL = {g.EST_LABEL!r}\nWINDOWS = {g.WINDOWS!r}\n'
          f'LAMBDAS = {g.LAMBDAS!r}\nLB_LAGS = {g.LB_LAGS!r}\nARCH_LAGS = {g.ARCH_LAGS!r}\nOOS_START = {g.OOS_START!r}\n'
          f'SEED = {g.SEED!r}\nGHOST = {g.GHOST!r}')
CORE = [g.returns, g.ohlc, g.ann, g.rolling_vol, g.ewma_var, g.ewma_series, g.ewma_facts, g.range_parts,
        g.daily_estimators, g.yz_k, g.window_estimators, g.rolling_estimators, g.acf, g.ljung_box, g.arch_lm, g.ols,
        g.forecasts, g.losses, g.mz_table, g.dm_test, g.lambda_qlike]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 8: Volatility estimators and volatility clustering\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Volatility as the standard deviation of returns; annualisation with the actual frequency; why volatility is latent.\n"
       "- Historical (rolling) volatility and the choice of the window; EWMA (RiskMetrics) and the ghost effect.\n"
       "- Range-based estimators: Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang; efficiency by simulation; real data.\n"
       "- Realised volatility from (simulated) intraday prices; microstructure noise.\n"
       "- Volatility clustering: ACF of squared and absolute returns, Ljung–Box (McLeod–Li) and ARCH-LM tests (SFEkurgarch).\n"
       "- Long-run behaviour, the VIX, the leverage effect (SFEvolnonparest); evaluation of volatility forecasts with MSE and QLIKE.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 13; Tsay (2010), "
       "*Analysis of Financial Time Series*, Ch. 3; J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; S&P 500 and DAX since 1990, BET since 1997, Bitcoin since "
       "2014, BVB stocks since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Volatility = standard deviation of returns; annualised with $\\sqrt{a}$, $a$ = observations per year.\n"
       "- EWMA: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1-\\lambda)r_{t-1}^2$.\n"
       "- OHLC notation (in %): $o_t = \\ln(O_t/C_{t-1})$, $u_t = \\ln(H_t/O_t)$, $d_t = \\ln(L_t/O_t)$, $c_t = \\ln(C_t/O_t)$.\n"
       "- Losses of a variance forecast $h_t$ with proxy $\\hat\\sigma_t^2$: MSE $(\\hat\\sigma_t^2 - h_t)^2$, QLIKE "
       "$\\hat\\sigma_t^2/h_t + \\ln h_t$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Volatility in four markets\n\n"
       "- Daily returns of the S&P 500, DAX, BET and Bitcoin; annual volatility, largest move, excess kurtosis."),
    code(src(g.fig_returns_bursts)),
    code("pd.DataFrame(fig_returns_bursts(save=False)).T"),
    md("## 2. Historical volatility and the window\n\n"
       "- Rolling 21-, 63- and 252-day volatility of the S&P 500; the standard error of a sample volatility."),
    code(src(g.fig_rolling_windows, g.vol_standard_error)),
    code("w = fig_rolling_windows(save=False)\n{k: v for k, v in w.items() if k in ('21', '63', '252')}"),
    code("vol_standard_error()"),
    md("## 3. EWMA and the ghost effect\n\n"
       "- EWMA weights for $\\lambda = 0.94$ and $0.97$ against a 63-day window; half-life $\\ln 0.5/\\ln\\lambda$.\n"
       "- The 63-day rolling volatility of the S&P 500 drops abruptly in June 2020, when the crash days leave the window."),
    code(src(g.fig_ewma_weights, g.fig_ghost)),
    code("fig_ewma_weights(save=False)"),
    code("fig_ghost(save=False)"),
    md("## 4. Range-based estimators\n\n"
       "- One simulated trading day with its open, high, low and close.\n"
       "- Whole-sample estimators for five assets; rolling 21-day estimates of the S&P 500.\n"
       "- Efficiency by simulation: continuous trading, an overnight jump, discrete trading."),
    code(src(g.fig_ohlc_path, g.estimators_table, g.fig_range_rolling, g.simulate_ohlc, g.efficiency_sim, g.fig_efficiency)),
    code("fig_ohlc_path(save=False)"),
    code("pd.DataFrame(estimators_table()).T"),
    code("fig_range_rolling(save=False)"),
    code("eff, draws = efficiency_sim()\nfig_efficiency(draws, save=False)\n"
         "pd.DataFrame({(sc, e): v for sc, d in eff.items() for e, v in d.items()}).T.round(3)"),
    md("## 5. Realised volatility (simulated one-minute prices)\n\n"
       "- Realised variance $\\mathrm{RV}_t = \\sum_i r_{t,i}^2$ tracks the true daily variance; microstructure noise "
       "inflates RV at high sampling frequencies (signature plot)."),
    code(src(g.simulate_intraday, g.realised_var, g.fig_realised, g.fig_signature)),
    code("fig_realised(save=False)"),
    code("fig_signature(save=False)"),
    md("## 6. Volatility clustering\n\n"
       "- ACF of returns, squared returns and absolute returns; Ljung–Box and ARCH-LM tests; shuffled returns.\n"
       "- Kurtosis of a variance mixture and of a GARCH(1,1) process (SFEkurgarch)."),
    code(src(g.clustering_table, g.fig_acf_clustering, g.fig_shuffle, g.mixture_kurtosis, g.garch_kurtosis, g.fig_kurtosis)),
    code("ct = clustering_table()\n"
         "pd.DataFrame({NAME[k]: {'rho1 r': v['rho_r'], 'rho1 r^2': v['rho_r2'], 'LB(10) r': v['lb_r']['q'], "
         "'LB(10) r^2': v['lb_r2']['q'], 'LB(10) |r|': v['lb_abs']['q'], 'ARCH-LM(5)': v['arch']['lm'], "
         "'R^2': v['arch']['r2'], 'excess kurtosis': v['exkurt']} for k, v in ct.items()}).T.round(3)"),
    code("fig_acf_clustering(save=False)"),
    code("fig_shuffle(save=False)"),
    code("fig_kurtosis(save=False)"),
    md("## 7. Long-run behaviour, the VIX and the leverage effect\n\n"
       "- Annual volatility since 1990; persistence of monthly realised volatility.\n"
       "- VIX against the realised volatility of the next 21 days.\n"
       "- corr$(r_t, |r_{t+j}|)$ and the nonparametric volatility given yesterday's return (SFEvolnonparest)."),
    code(src(g.annual_vol, g.monthly_rv, g.fig_long_term, g.vix_data, g.fig_vix, g.leverage, g.fig_leverage, g.quadk,
             g.lpregest, g.nonparametric_vol, g.fig_news_impact)),
    code("fig_long_term(save=False)['ar']"),
    code("vx = fig_vix(save=False)\n{k: vx[k] for k in ['mean_vix', 'mean_rv', 'share_above', 'corr', 'mz_vix', 'mz_past', 'corr_dvix_r']}"),
    code("lv = fig_leverage(save=False)\n{k: (round(v['cc'][1], 3), round(v['ratio'], 2)) for k, v in lv.items()}"),
    code("fig_news_impact(save=False)"),
    md("## 8. Evaluating volatility forecasts\n\n"
       "- Next-day variance forecasts of the S&P 500 from 2010: rolling windows, EWMA, VIX; MSE, QLIKE, Mincer–Zarnowitz, "
       "Diebold–Mariano.\n"
       "- The best $\\lambda$ by QLIKE for the S&P 500, BET and Bitcoin."),
    code(src(g.forecast_table, g.fig_lambda, g.fig_forecast_paths)),
    code("ft = forecast_table()\nprint(ft['dm_vix_ewma'], ft['dm_ewma_hist21'])\n"
         "pd.DataFrame({m: {'MSE': ft['r2'][m]['mse'], 'QLIKE': ft['r2'][m]['qlike'], 'MZ R^2 (r^2)': ft['mz'][m]['r2']['r2'], "
         "'MZ R^2 (range)': ft['mz'][m]['range']['r2']} for m in ft['r2']}).T.round(4)"),
    code("fig_lambda(save=False)"),
    code("fig_forecast_paths(save=False)"),
    md("## 9. AI for scientific discovery: range-based estimators on the BVB\n\n"
       "- Open question: do range-based estimators help on the Bucharest Stock Exchange, where many stocks trade thinly?\n"
       "- Starter result: share of days with an unchanged close, ratios to close-to-close, QLIKE of next-day forecasts.\n"
       "- Check before trusting any answer, yours or an AI's: the constants of each estimator, adjusted OHLC prices, the "
       "annualisation, no look-ahead in the forecasts."),
    code(src(g.bvb_range)),
    code("pd.DataFrame({k: {'unchanged close': v['flat'], 'P/CC': v['park_cc'], 'YZ/CC': v['yz_cc'], "
         "**{f'QLIKE {e}': q for e, q in v['qlike'].items()}} for k, v in bvb_range().items()}).T.round(3)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.hist_vol, s.ewma_steps, s.ohlc_one_day, s.ohlc_days, s.clustering_from_acf, s.losses_by_hand]


def task(head, context, steps, report, model=None):
    t = f"## {head}\n\n**Context.** {context}" + (f" Model: {model}." if model else '') + '\n\n'
    return md(t + '\n'.join(f'{i}. {x}' for i, x in enumerate(steps, 1)) + f'\n\n**Report:** {report}')


SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 8: Volatility estimators and volatility clustering\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 8: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Returns and adjusted OHLC prices; rolling and EWMA volatility; the five range-based estimators; ACF, Ljung–Box and "
       "ARCH-LM tests; next-day variance forecasts, MSE and QLIKE losses, Mincer–Zarnowitz and Diebold–Mariano.\n"
       "- Helpers for Part A: historical volatility from a few returns, EWMA step by step, estimators from OHLC prices, "
       "Ljung–Box and ARCH-LM from given numbers, losses by hand."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    task('A1 [Solved]: historical volatility from five days', 'Daily returns (%) of a stock: 1.2, −0.8, 0.5, −1.5, 0.6.',
         ['Compute the mean and the sample standard deviation (divide by $n - 1$).', 'Annualise the standard deviation with $a = 252$.',
          'Compute the zero-mean volatility $\\sqrt{\\frac{1}{n}\\sum r_t^2}$.',
          'Compute $1/\\sqrt{2n}$ and an approximate 95% interval for the annual volatility.'],
         'the mean, two daily volatilities, one annual volatility, one interval and one sentence.'),
    code("# Solution\npd.Series(hist_vol([1.2, -0.8, 0.5, -1.5, 0.6]))"),
    md("**Interpretation of the result.** Five days say almost nothing: the 95% interval for the annual volatility is very wide."),
    task('A2 [Proposed]: Bitcoin over six days', 'Daily Bitcoin returns (%): 3.1, −4.2, 1.0, 5.5, −2.4, −0.9.',
         ['Compute the mean and the sample standard deviation.', 'Annualise it with $a = 365$ and, wrongly, with $a = 252$.',
          'Compute the approximate 95% interval for the annual volatility ($a = 365$).'],
         'two annual volatilities, one interval and one sentence.', model='A1'),
    code("# Solution\nr = [3.1, -4.2, 1.0, 5.5, -2.4, -0.9]\npd.DataFrame({'a = 365': hist_vol(r, 365), 'a = 252': hist_vol(r, 252)})"),
    task('A3 [Solved]: EWMA step by step', '$\\lambda = 0.94$; today\'s EWMA variance is $\\sigma_1^2 = 1.00$ (%²); the next '
         'three returns are $r_1 = -2.0\\%$, $r_2 = 0.5\\%$, $r_3 = 3.0\\%$.',
         ['Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$.', 'Annualise $\\sigma_4$ with $a = 252$.',
          'Compute the half-life and the weight of $r_3^2$ in $\\sigma_4^2$.'],
         'three variances, one annual volatility, the half-life and one sentence.'),
    code("# Solution\newma_steps([-2.0, 0.5, 3.0], 0.94, 1.0)"),
    md("**Interpretation of the result.** One 3% day raises the variance by about 42%; a quiet day lowers it by only 6%."),
    task('A4 [Proposed]: a slower EWMA', 'The same data as A3, with $\\lambda = 0.97$ (the RiskMetrics monthly value).',
         ['Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$ and annualise $\\sigma_4$.',
          'Compute the half-life and the number of days that carry 99% of the weight.',
          'Compare with A3: which $\\lambda$ reacts faster to the 3% day?'],
         'three variances, one annual volatility, two numbers of days and one sentence.', model='A3'),
    code("# Solution\newma_steps([-2.0, 0.5, 3.0], 0.97, 1.0)"),
    task('A5 [Solved]: four estimators from one day', 'Previous close 100.0; open 101.0, high 103.5, low 99.8, close 102.2.',
         ['Compute $o$, $u$, $d$, $c$ (in %) and $r = o + c$.',
          'Compute the one-day CC, Parkinson, Garman–Klass and Rogers–Satchell variances.',
          'Annualise each with $a = 252$, as if every day looked like this one.', 'Say which estimators see the overnight move $o$.'],
         'five log returns, four variances, four annual volatilities and one sentence.'),
    code("# Solution\nohlc_one_day(100.0, 101.0, 103.5, 99.8, 102.2)"),
    md("**Interpretation of the result.** Only close-to-close (and Yang–Zhang, on several days) include the overnight move; "
       "Parkinson, Garman–Klass and Rogers–Satchell see only the trading session."),
    task('A6 [Proposed]: three days and the Yang–Zhang estimator', 'Previous close 50.0; open, high, low, close of three days: '
         '(50.2, 51.0, 49.6, 50.8), (50.5, 51.9, 50.3, 51.6), (51.9, 52.4, 50.9, 51.1).',
         ['Compute $o$, $u$, $d$, $c$ for each day.', 'Compute the Parkinson and Rogers–Satchell terms of each day and their means.',
          'Compute $k$, $\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$ and the Yang–Zhang variance for $n = 3$.',
          'Compare with the CC sample variance and annualise all three with $a = 252$.'],
         'a small table, three variances, three annual volatilities and one sentence.', model='A5'),
    code("# Solution\nohlc_days([(50.2, 51.0, 49.6, 50.8), (50.5, 51.9, 50.3, 51.6), (51.9, 52.4, 50.9, 51.1)], 50.0)"),
    task('A7 [Solved]: two tests for clustering', '$T = 1000$ daily returns; the ACF of the squared returns at lags 1–5: 0.20, '
         '0.15, 0.12, 0.10, 0.08; the ARCH-LM regression with $q = 5$ lags on $n = 995$ observations has $R^2 = 0.08$.',
         ['Compute the Ljung–Box statistic $Q(5)$ of the squared returns.', 'Compute $\\mathrm{LM} = nR^2$.',
          'Compare both with $\\chi^2_{0.95}(5) = 11.07$ and decide.'],
         'two statistics, two decisions and one sentence.'),
    code("# Solution\nclustering_from_acf([0.20, 0.15, 0.12, 0.10, 0.08], 1000, r2=0.08, n=995)"),
    md("**Interpretation of the result.** Both statistics are far above the critical value: the variance is predictable from "
       "the recent past."),
    task('A8 [Proposed]: when the two tests disagree', '$T = 500$; ACF of the squared returns at lags 1–5: 0.12, 0.08, 0.06, '
         '0.05, 0.04; ARCH-LM with $q = 5$ on $n = 495$ observations: $R^2 = 0.018$.',
         ['Compute $Q(5)$ and decide at 5%.', 'Compute LM and decide at 5%.', 'Give one reason why the two tests can disagree.'],
         'two statistics, two decisions and one sentence.', model='A7'),
    code("# Solution\nclustering_from_acf([0.12, 0.08, 0.06, 0.05, 0.04], 500, r2=0.018, n=495)"),
    task('A9 [Solved]: MSE and QLIKE by hand', 'Over four days the proxy $r_t^2$ is 1.0, 4.0, 0.25, 2.25 (%²). Forecast A is '
         'flat at 1.5; forecast B is 1.0, 2.5, 0.8, 1.8.',
         ['Compute the MSE of each forecast.', 'Compute the QLIKE terms $r_t^2/h_t + \\ln h_t$ and their mean for each forecast.',
          'Say which forecast is better under each loss.'],
         'two MSE values, two QLIKE values and one sentence.'),
    code("# Solution\nlosses_by_hand([1.0, 4.0, 0.25, 2.25], {'A': [1.5] * 4, 'B': [1.0, 2.5, 0.8, 1.8]})"),
    md("**Interpretation of the result.** B is better under both losses: it follows the changes of the variance."),
    task('A10 [Proposed]: under- and over-prediction', 'The true variance and the proxy are both 1. Two forecasts: $h = 0.5$ '
         '(half) and $h = 2$ (double).',
         ['Compute the MSE of each forecast.', 'Compute the QLIKE of each forecast.', 'Say which loss punishes under-prediction more.'],
         'four numbers and one sentence on why this matters for VaR.', model='A9'),
    code("# Solution\nlosses_by_hand([1.0], {'h = 0.5': [0.5], 'h = 2': [2.0]})"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: historical and EWMA volatility of the S&P 500\n\n"
       "**Question.** How different are the 21-, 63- and 252-day and EWMA volatilities of the S&P 500?\n\n"
       "1. Compute the rolling 21-, 63- and 252-day volatility and the EWMA volatility ($\\lambda = 0.94$), annualised with the actual frequency.\n"
       "2. Draw the four series since 2018.\n"
       "3. Report the four values at the end of the sample and their maxima in 2020, with dates.\n"
       "4. Find the day of June 2020 when the 63-day volatility falls most, and report the change of EWMA on that day.\n"
       "5. Interpretation: which estimate would you report to a risk committee on that day?\n\n"
       "**Report:** the chart, twelve numbers and two sentences."),
    code("# Solution\n" + src(s.b1_hist) + "\n\n\nb1 = b1_hist(save=False)\nb1"),
    md("**Interpretation of the result.** EWMA: the drop of the 63-day estimate in June 2020 is the ghost effect of the crash "
       "days leaving the window, not news."),
    md("## B2 [Proposed]: BET and Bitcoin\n\n"
       "**Question.** How volatile are the BET and Bitcoin compared with the S&P 500, today and at the end of 2017? Model: B1.\n\n"
       "1. Compute the number of observations per year and the whole-sample annual volatility of each series.\n"
       "2. Annualise Bitcoin also with 252 and give the error in %.\n"
       "3. Report the 252-day volatility at the end of 2017 and at the end of the sample, and the maximum of the 63-day "
       "volatility with its year.\n"
       "4. Interpretation: has Bitcoin become an asset with equity-like volatility?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b2_hist) + "\n\n\npd.DataFrame(b2_hist()).T"),
    md("## B3 [Solved]: range-based estimators for the S&P 500\n\n"
       "**Question.** Do range-based estimators give the same volatility as closing prices for the S&P 500?\n\n"
       "1. Compute $o_t$, $u_t$, $d_t$, $c_t$ and the five estimators on the whole sample, annualised.\n"
       "2. Decompose $\\mathrm{Var}(r_t) = \\mathrm{Var}(o_t) + \\mathrm{Var}(c_t) + 2\\,\\mathrm{Cov}(o_t, c_t)$.\n"
       "3. Draw the rolling 21-day estimates since 2025.\n"
       "4. Interpretation: why are Parkinson, Garman–Klass and Rogers–Satchell below close-to-close?\n\n"
       "**Report:** the chart, five volatilities, a decomposition and two sentences."),
    code("# Solution\n" + src(s.b3_range) + "\n\n\nb3 = b3_range(save=False)\nb3"),
    md("**Interpretation of the result.** They see only the trading session: they miss the overnight variance and the "
       "covariance of overnight and open-to-close returns; Yang–Zhang adds the night but not the covariance."),
    md("## B4 [Proposed]: DAX, Bitcoin and TLV\n\n"
       "**Question.** For which of the DAX, Bitcoin and Banca Transilvania do range-based estimators agree with close-to-close? Model: B3.\n\n"
       "1. Compute the five annualised estimators for each asset.\n"
       "2. Compute the share of the night and $\\mathrm{corr}(o_t, c_t)$.\n"
       "3. Rank the assets by the ratio Parkinson / CC.\n"
       "4. Interpretation: for which asset could you replace close-to-close by Parkinson?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b4_range) + "\n\n\nb4 = b4_range()\n"
         "pd.DataFrame({NAME[k]: {**d['ann'], 'night share': d['night_share'], 'corr(o, c)': d['corr_oc']} for k, d in b4.items()}).T.round(3)"),
    md("## B5 [Solved]: clustering in the S&P 500\n\n"
       "**Question.** Are the size and the sign of S&P 500 returns equally predictable?\n\n"
       "1. Draw the ACF of $r_t$, $r_t^2$ and $|r_t|$ for lags 1–50 with the i.i.d. 95% band.\n"
       "2. Report $\\hat\\rho_1, \\dots, \\hat\\rho_5$ of $r_t$ and of $r_t^2$.\n"
       "3. Compute the Ljung–Box $Q(10)$ of $r_t$, $r_t^2$ and $|r_t|$, and the ARCH-LM(5) statistic with its $R^2$.\n"
       "4. Interpretation: does volatility clustering contradict weak-form market efficiency?\n\n"
       "**Report:** the chart, ten autocorrelations, four statistics and two sentences."),
    code("# Solution\n" + src(s.b5_clustering) + "\n\n\nb5 = b5_clustering(save=False)\nb5"),
    md("**Interpretation of the result.** No: the size of returns is predictable, the sign hardly; efficiency is about "
       "predictable returns, not about predictable risk (Chapter 7)."),
    md("## B6 [Proposed]: four more series\n\n"
       "**Question.** Does volatility clustering in the BET, Bitcoin, Banca Transilvania and OMV Petrom depend on the "
       "order of the days? Model: B5.\n\n"
       "1. Compute $\\hat\\rho_1(r^2)$, $Q(10)$ of $r_t^2$, ARCH-LM(5) and the excess kurtosis of each series.\n"
       "2. Shuffle the returns at random (seed 2026) and compute $Q(10)$ of $r_t^2$ and ARCH-LM(5) again.\n"
       "3. Interpretation: why does shuffling remove the clustering but not the heavy tails?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b6_clustering) + "\n\n\nb6 = b6_clustering()\n"
         "pd.DataFrame({NAME[k]: {'rho1 r^2': d['rho_r2'], 'LB(10) r^2': d['lb_r2']['q'], 'ARCH-LM': d['arch']['lm'], "
         "'excess kurtosis': d['exkurt'], 'LB shuffled': d['lb_r2_shuffled']['q'], 'LM shuffled': d['arch_shuffled']['lm']} "
         "for k, d in b6.items()}).T.round(3)"),
    md("## B7 [Solved]: which forecast is best?\n\n"
       "**Question.** Which simple forecast of tomorrow's S&P 500 variance has the smallest loss since 2010?\n\n"
       "1. Build next-day forecasts: rolling means of $r_t^2$ on 21, 63 and 252 days, EWMA with $\\lambda = 0.94$ and $0.97$, and $\\mathrm{VIX}^2/a$.\n"
       "2. Compute MSE and QLIKE with the proxy $r_t^2$.\n"
       "3. Run the Diebold–Mariano test for EWMA 0.94 against the 21-day window and for the VIX against EWMA 0.94.\n"
       "4. Draw the QLIKE of EWMA for $\\lambda$ from 0.80 to 0.995.\n"
       "5. Interpretation: is the VIX a better forecast of tomorrow's variance than EWMA?\n\n"
       "**Report:** a table of twelve numbers, two tests, the chart and two sentences."),
    code("# Solution\n" + src(s.b7_forecast) + "\n\n\nb7 = b7_forecast(save=False)\n"
         "print({c: b7[c] for c in ['best_lambda', 'dm_ewma_hist21', 'dm_vix_ewma']})\npd.DataFrame(b7['loss']).T.round(4)"),
    md("**Interpretation of the result.** No: the VIX has slightly lower losses than EWMA, but the difference is not "
       "significant; both beat the rolling windows."),
    md("## B8 [Proposed]: the best $\\lambda$ for the BET and Bitcoin\n\n"
       "**Question.** Is the RiskMetrics $\\lambda = 0.94$ a good choice for the BET and for Bitcoin? Model: B7.\n\n"
       "1. Compute the QLIKE of next-day EWMA forecasts for $\\lambda$ from 0.80 to 0.995 and find the best $\\lambda$.\n"
       "2. Report the QLIKE at the best $\\lambda$, at 0.94 and at 0.97, and the half-life of the best $\\lambda$.\n"
       "3. Run the Diebold–Mariano test of the best $\\lambda$ against 0.94.\n"
       "4. Interpretation: would you use $\\lambda = 0.94$ for both series?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b8_lambda) + "\n\n\npd.DataFrame(b8_lambda()).T"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: which estimator for BVB stocks?\n\n"
       "**Question.** Do range-based estimators improve volatility forecasts for thinly traded stocks of the Bucharest Stock "
       "Exchange? Models: B3, B7.\n\n"
       "1. Compute the share of trading days with an unchanged close, and the ratios Parkinson / CC and YZ / CC of the whole-sample volatility.\n"
       "2. Compute the QLIKE (proxy $r_t^2$) of next-day forecasts by the 21-day CC, Parkinson and YZ variances, from 2010.\n"
       "3. Compare the four stocks with the S&P 500.\n"
       "4. List two reasons why the best estimator may differ across stocks.\n"
       "5. Interpretation: what data would you need to explain the differences?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\n" + src(g.bvb_range, s.c1_bvb) + "\n\n\nc1 = c1_bvb()\n"
         "pd.DataFrame({k: {'unchanged close': v['flat'], 'P/CC': v['park_cc'], 'YZ/CC': v['yz_cc'], "
         "**{f'QLIKE {e}': q for e, q in v['qlike'].items()}} for k, v in c1.items()}).T.round(3)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant was asked to summarise the volatility of the S&P 500 and of Bitcoin. It answered:\n\n"
       "- (a) since 2008 the Parkinson volatility of the S&P 500 is below the close-to-close volatility; Parkinson is about five "
       "times more efficient, so the Parkinson value is the true volatility;\n"
       "- (b) the Ljung–Box $Q(10)$ of S&P 500 returns has a p-value below 0.001, so the S&P 500 shows volatility clustering;\n"
       "- (c) EWMA with $\\lambda = 0.94$ has a half-life of about 94 days;\n"
       "- (d) Bitcoin's annual volatility is its daily standard deviation times $\\sqrt{252}$;\n"
       "- (e) since 1990 the VIX was above the realised volatility of the next 21 days on most days.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(g.vix_data, s.c2_check) + "\n\n\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 8, 'lecture')
    build(SEMINAR, 8, 'seminar')
