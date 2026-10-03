"""
build_notebooks_ch16.py -- lecture and seminar notebooks of Chapter 16 (SFM): review, the course in one notebook
=================================================================================================================
Output: notebooks/EN/chapter16_lecture_notebook.ipynb, notebooks/EN/chapter16_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_16/generate_all_charts.py and seminar16.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Both notebooks install the arch package first (not part of Colab).
Run:  python3 notebooks/build_notebooks_ch16.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter16_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter16_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 16
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(16)
import generate_all_charts as g   # noqa: E402
import seminar16 as s             # noqa: E402

INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
CONSTS = ('from arch import arch_model\n'
          f'NAME = {g.NAME!r}\nASSETS = {g.ASSETS!r}\nCOLORS = {g.COLORS!r}\nSTART = {g.START!r}\n'
          f'LB_LAGS = {g.LB_LAGS!r}\nVR_Q = {g.VR_Q!r}\nHILL_FRAC = {g.HILL_FRAC!r}\nLAMBDA = {g.LAMBDA!r}\n'
          f'ALPHA_VAR = {g.ALPHA_VAR!r}\nALPHA_ES = {g.ALPHA_ES!r}\nWINDOW = {g.WINDOW!r}\nOOS_START = {g.OOS_START!r}\n'
          f'NMIN = {g.NMIN!r}')
BASE = [g.returns, g.performance, g.moments, g.hill, g.tail_index, g.acf, g.ljung_box, g.variance_ratio, g.ewma_var,
        g.garch_t, g.hs_var, g.hs_es, g.normal_var, g.normal_es, g.kupiec, g.var_backtest, g.block_sizes, g.hurst_rs]

# =============================================================================
# LECTURE: THE COURSE IN ONE NOTEBOOK
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 16: Review\n\n"
       "*Lecture notebook: the course in one notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Three series, one common window (2 January 2015 to 18 September 2026): the S&P 500, the BET and Bitcoin.\n"
       "- One section per block of chapters: returns and performance (Chapters 0–1), distribution and tails (2–6), "
       "dependence and efficiency (7–8), GARCH (9), VaR, ES and backtesting (10), long memory (11).\n"
       "- Every function uses the definition of its chapter; the last cell prints the course in one table.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed.; Cont (2001); "
       "Lo and MacKinlay (1988); Bollerslev (1986); Kupiec (1995); Hurst (1951)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %, each series on its own calendar (Bitcoin: 7 days "
       "a week); annualisation with the actual number of observations per year $q$.\n"
       "- VaR at level $\\alpha$: $\\mathrm{VaR}_\\alpha = -q_\\alpha$, the loss exceeded with probability $\\alpha$ (VaR 1%); "
       "ES 2.5%: the average loss on the worst 2.5% of days. Both are positive numbers."),
    code(CONSTS + '\n\n\n' + src(*BASE)),
    md("## 1. Returns and performance (Chapters 0–1)\n\n"
       "- CAGR, annualised mean and volatility, the Sharpe ratio with its standard error, the maximum drawdown."),
    code("perf = {NAME[k]: performance(k) for k in ASSETS}\n"
         "pd.DataFrame(perf).T[['n', 'q', 'mean_log', 'mean_arith', 'cagr', 'vol', 'sharpe', 'sharpe_se', 'mdd', 'mdd_date']]"),
    code(src(g.fig_three_series) + "\n\n\nfig_three_series(save=False)"),
    md("- Bitcoin: the arithmetic mean is far above the CAGR, the volatility drag $\\approx \\sigma^2/2$.\n"
       "- The Sharpe ratios differ by less than their standard errors."),
    md("## 2. Distribution and tails (Chapters 2–6)\n\n"
       "- Skewness, excess kurtosis and Jarque–Bera; the Hill tail index of the losses with $k = 2.5\\%$ of $n$."),
    code("pd.DataFrame({NAME[k]: {**moments(returns(k)), **{'hill_' + c: v for c, v in tail_index(returns(k)).items()}} "
         "for k in ASSETS}).T.round(3)"),
    md("- Excess kurtosis above 10 and tail indices between 2 and 3: heavy tails, finite variance, infinite kurtosis."),
    md("## 3. Dependence and efficiency (Chapters 7–8)\n\n"
       "- Autocorrelation of $r_t$ and $|r_t|$; Ljung–Box $Q(10)$ on $r_t$ and $r_t^2$; the robust variance ratio $Z^*(5)$."),
    code(src(g.fig_stylised) + "\n\n\nfig_stylised(save=False)"),
    code("pd.DataFrame({NAME[k]: {'LB r': ljung_box(returns(k))['q'], 'LB r^2': ljung_box(returns(k) ** 2)['q'],\n"
         "                         'VR(5)': variance_ratio(returns(k))['vr'], 'Z*(5)': variance_ratio(returns(k))['zs'],\n"
         "                         'p(Z*)': variance_ratio(returns(k))['p_zs']} for k in ASSETS}).T.round(3)"),
    md("- The squared returns are far more autocorrelated than the returns: volatility clustering.\n"
       "- The robust VR test does not reject the random walk at 5% on this window."),
    md("## 4. GARCH(1,1) with Student-t innovations (Chapter 9)"),
    code("pd.DataFrame({NAME[k]: garch_t(returns(k)) for k in ASSETS}).T"),
    md("- Persistence $\\alpha + \\beta$ close to 1; Bitcoin is an integrated GARCH (no finite half-life)."),
    md("## 5. VaR, ES and backtesting (Chapter 10)\n\n"
       "- VaR 1% and ES 2.5% by historical simulation and with the Normal distribution; one-day VaR 1% forecasts since 2017 "
       "by historical simulation (500 days) and EWMA-Normal, with the Kupiec test."),
    code("pd.DataFrame({NAME[k]: {'VaR 1% HS': hs_var(returns(k)), 'VaR 1% Normal': normal_var(returns(k)),\n"
         "                         'ES 2.5% HS': hs_es(returns(k)), 'ES 2.5% Normal': normal_es(returns(k))} for k in ASSETS}).T.round(2)"),
    code(src(g.fig_backtest) + "\n\n\nbt = fig_backtest('bet', save=False)\nbt"),
    md("- EWMA-Normal follows the storms but has more than twice the target rate of exceptions: the Normal quantile is too small.\n"
       "- Historical simulation has the right rate but reacts slowly: its exceptions come in clusters."),
    md("## 6. Long memory (Chapter 11)"),
    code("pd.DataFrame({NAME[k]: {'H (R/S) of r': hurst_rs(returns(k)), 'H (R/S) of |r|': hurst_rs(np.abs(returns(k)))} "
         "for k in ASSETS}).T.round(3)"),
    md("- $H$ of $|r_t|$ is far above 0.5: the long memory is in the volatility, not in the returns."),
    md("## 7. The course in one table"),
    code(src(g.summary, g.summary_table) + "\n\n\nS = {k: summary(k) for k in ASSETS}\nsummary_table(S).round(3)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_VISIBLE = [s.a1_returns, s.a3_vr, s.b1_checkup]
SEM_SOLUTIONS = [s.a2_tails, s.a4_vol, s.a5_var, s.a6_credit, s.b2_backtests, s.b3_memory, s.c2_check]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 16: Review\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 16: the slides \"What You Need for Today\" are a formula sheet for all chapters.\n"
       "- Part A: six exam-type problems on paper, checked in code. Part B: three data tasks with an interpretation "
       "question. Part C: an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n"
       "- The functions of the course: returns, performance, moments, Hill, autocorrelation, Ljung–Box, variance ratio, "
       "EWMA, GARCH(1,1)-t, VaR and ES, Kupiec, rolling VaR forecasts, Hurst (R/S).\n"
       "- Helpers for the [Solved] tasks: A1, A3 and B1."),
    code(CONSTS + "\nZ01, Z025 = stats.norm.ppf(0.01), stats.norm.ppf(0.025)\n\n\n" + src(*BASE) + '\n\n\n' + src(*SEM_VISIBLE)),
    code("# [solutions only]\n" + src(*SEM_SOLUTIONS)),
    # ---------------- Part A
    md("# Part A: problems on paper\n\n- Solve each problem on paper first; then check it with the code."),
    md("## A1 [Solved]: returns, drawdown and the Sharpe ratio\n\n"
       "**Context.** A share closes at 50.0, 51.5, 49.8, 52.3, 47.9 and 50.6 RON on six consecutive days.\n\n"
       "1. Compute the five simple and the five log returns, in %.\n"
       "2. Compute the total return from the prices and from $\\sum_t r_t$.\n"
       "3. Compute the maximum drawdown and the gain needed to return to the peak.\n"
       "4. A fund has a daily mean return of 0.04% and a daily standard deviation of 1.3%; with $r_f = 3\\%$ a year, compute "
       "its annual mean, its annual volatility, its Sharpe ratio and the SE of the Sharpe ratio over 10 years.\n\n"
       "**Report:** fourteen numbers and one sentence."),
    code("# Solution\na1_returns()"),
    md("**Interpretation of the result.** The Sharpe ratio is about one standard error from zero: ten years of data cannot "
       "show that this fund beats the risk-free rate."),
    md("## A2 [Proposed]: normality, tails and stable scaling\n\n"
       "**Context.** $n = 2000$ daily returns with skewness $-0.8$ and excess kurtosis 9; the six largest losses, in %: "
       "11.2, 8.4, 7.7, 6.0, 5.5, 4.9. Model: Seminar 2, A7 and Seminar 5, A1.\n\n"
       "1. Compute JB, its skewness and kurtosis parts, and decide at 5%.\n"
       "2. Compute $\\nu$ of a Student-t distribution with the same excess kurtosis.\n"
       "3. Compute the Hill estimate with $k = 5$ and its SE.\n"
       "4. A stable law with $\\alpha = 1.6$ describes daily returns: compute the 20-day scale factor and compare it with $\\sqrt{20}$.\n\n"
       "**Report:** seven numbers, one decision and one sentence."),
    code("# Solution\na2_tails()"),
    md("## A3 [Solved]: autocorrelation and the variance ratio\n\n"
       "**Context.** $T = 1500$ daily returns; $\\hat\\rho_1 = -0.05$, $\\hat\\rho_2 = 0.03$, $\\hat\\rho_3 = 0.02$, "
       "$\\hat\\rho_4 = -0.01$; the robust SE of $\\widehat{\\mathrm{VR}}(5)$ is 0.055.\n\n"
       "1. Compute the Ljung–Box statistic $Q(4)$ and decide at 5%.\n"
       "2. Compute $\\mathrm{VR}(5)$.\n"
       "3. Compute $Z(5)$ and $Z^*(5)$ and decide at 5%.\n\n"
       "**Report:** four numbers, two decisions and one sentence."),
    code("# Solution\na3_vr()"),
    md("**Interpretation of the result.** No evidence against the random walk at one week; weak-form efficiency is not rejected."),
    md("## A4 [Proposed]: volatility forecasts\n\n"
       "**Context.** GARCH(1,1) for returns in %: $\\omega = 0.05$, $\\alpha = 0.10$, $\\beta = 0.85$; today $\\sigma_t^2 = 2.5$ "
       "and $\\varepsilon_t = 1.0$. One day of an index: high 105.2, low 101.6. Model: Seminar 9, A1 and A3; Seminar 8, A3 and A5.\n\n"
       "1. Compute $\\sigma_{t+1}^2$ and the EWMA forecast with $\\lambda = 0.94$.\n"
       "2. Compute the persistence, the half-life, the long-run variance and the long-run annual volatility.\n"
       "3. Compute $E_t[\\sigma_{t+5}^2]$.\n"
       "4. Compute the Parkinson variance of the day and the annualised Parkinson volatility.\n\n"
       "**Report:** eight numbers and one sentence."),
    code("# Solution\na4_vol()"),
    md("## A5 [Proposed]: VaR, ES and backtesting\n\n"
       "**Context.** A position of 2,000,000 RON; daily returns with mean 0.05% and standard deviation 1.2%. "
       "Model: Seminar 10, A1, A3 and A5.\n\n"
       "1. Compute the Normal VaR 1% and ES 2.5%, in % and in RON.\n"
       "2. Compute VaR 1% if the standardised returns are Student-t with $\\nu = 5$.\n"
       "3. A model has 17 exceptions of VaR 1% in 1000 days: compute $LR_{uc}$ and decide at 5%.\n"
       "4. Give the Basel zone of 11 exceptions in 250 days.\n\n"
       "**Report:** seven numbers, two decisions and one sentence."),
    code("# Solution\na5_var()"),
    md("## A6 [Proposed]: credit scoring and CoVaR\n\n"
       "**Context.** Logit $\\eta = -1.8 + 0.9x_1 - 0.6x_2$, applicant with $x_1 = 1.2$, $x_2 = 2.5$; LGD 40%, EAD 150,000 RON. "
       "A 1% quantile regression of the system return on a bank: $a = -1.8$, $b = 0.45$; bank: $q_{0.01} = -6.0\\%$, "
       "median 0.05%. Model: Seminar 12, A1 and A5; Seminar 15, A1.\n\n"
       "1. Compute the PD of the applicant and the odds ratio of $x_1$.\n"
       "2. Compute the expected loss of the loan.\n"
       "3. A scorecard has Gini 0.56: compute its AUC.\n"
       "4. Compute CoVaR 1% at the bank's VaR and at its median, and $\\Delta$CoVaR.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\na6_credit()"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: a full check-up of the BET\n\n"
       "**Question.** What do Chapters 1–11 say about the BET since 2015?\n\n"
       "1. Compute the annual mean, the volatility, the Sharpe ratio with its SE and the maximum drawdown.\n"
       "2. Compute the skewness, the excess kurtosis, JB and the Hill tail index of the losses with $k = 2.5\\%$ of $n$.\n"
       "3. Compute Ljung–Box $Q(10)$ on $r_t$ and on $r_t^2$, and fit a GARCH(1,1)-t.\n"
       "4. Compute VaR 1% and ES 2.5% by historical simulation and with the Normal distribution.\n"
       "5. Interpretation: which three facts would you write in an exam answer about the BET?\n\n"
       "**Report:** a table of about fifteen numbers, the chart and three sentences."),
    code("# Solution\nb1 = b1_checkup('bet', save=False)\n"
         "{'perf': {c: b1['perf'][c] for c in ['mean_log', 'vol', 'sharpe', 'sharpe_se', 'mdd']}, 'moments': b1['mom'],\n"
         " 'hill': b1['hill'], 'LB r': b1['lb_r']['q'], 'LB r^2': b1['lb_r2']['q'], 'garch': b1['garch'],\n"
         " 'VaR 1% HS': b1['var_hs'], 'VaR 1% Normal': b1['var_n'], 'ES 2.5% HS': b1['es_hs'], 'ES 2.5% Normal': b1['es_n']}"),
    md("**Interpretation of the result.** (1) Heavy tails: the Normal VaR 1% is far too low. (2) Returns are weakly "
       "autocorrelated, squared returns strongly: volatility clustering. (3) The Sharpe ratio is less than three standard "
       "errors from zero."),
    md("## B2 [Proposed]: backtesting VaR 1% for the S&P 500 and Bitcoin\n\n"
       "**Question.** Does a heavy-tailed method beat a Normal method with fast volatility? Model: B1 and Seminar 10, B3.\n\n"
       "1. Compute the one-day VaR 1% by historical simulation on the last 500 returns.\n"
       "2. Compute the one-day EWMA-Normal VaR 1% with $\\lambda = 0.94$.\n"
       "3. Count the exceptions of each method and run the Kupiec test.\n"
       "4. Find the largest number of exceptions in any 250 consecutive days.\n"
       "5. Interpretation: which method would a supervisor accept for each series?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = b2_backtests()\n"
         "pd.DataFrame({(NAME[k], m): {c: d[m][c] for c in ['x', 'n', 'rate', 'lr', 'p', 'max250']} "
         "for k, d in b2.items() for m in ['HS', 'EWMA']}).T.round(3)"),
    md("## B3 [Proposed]: efficiency and memory in two windows\n\n"
       "**Question.** Did the BET and the S&P 500 become more efficient after 2015? Model: A3; Seminar 7, B1 and Seminar 11, B1.\n\n"
       "1. Compute VR(5) and the robust $Z^*(5)$ in each window (2000–2014 and 2015–2026).\n"
       "2. Compute the Hurst exponent (R/S) of $r_t$ and of $|r_t|$ in each window.\n"
       "3. Interpretation: did the BET become more efficient after 2015?\n\n"
       "**Report:** one table of sixteen numbers and two sentences."),
    code("# Solution\npd.DataFrame(b3_memory()).T.round(3)"),
    # ---------------- Part C
    md("# Part C: AI critique"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the course for the exam, with the course data. It answered:\n\n"
       "- (a) the VaR 99% of the BET since 2015 is a negative number, the 1% quantile;\n"
       "- (b) the BET returns are close to Normal, because the skewness is small;\n"
       "- (c) Bitcoin's annual volatility is its daily standard deviation times $\\sqrt{252}$;\n"
       "- (d) for the S&P 500 a volatility shock halves in $1/(1 - \\alpha - \\beta)$ days;\n"
       "- (e) the Hurst exponent of $|r_t|$ is above 0.8 for the S&P 500, so its returns are predictable;\n"
       "- (f) for Normal returns, ES 2.5% and VaR 1% are almost the same number.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 16, 'lecture')
    build(SEMINAR, 16, 'seminar')
