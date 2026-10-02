"""
build_notebooks_ch4.py -- lecture and seminar notebooks of Chapter 4 (SFM): Probability
======================================================================================
Output: notebooks/EN/chapter4_lecture_notebook.ipynb, notebooks/EN/chapter4_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_04/generate_all_charts.py and seminar4.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch4.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter4_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter4_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 4
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(4)
import generate_all_charts as g   # noqa: E402
import seminar4 as s              # noqa: E402

CONSTS = (f'START = {g.START!r}\nASSETS = {g.ASSETS!r}\nCOLORS = {g.COLORS!r}\nSHORT = {g.SHORT!r}\nPAL = {g.PAL!r}\n'
          f'RANDU_A, RANDU_M = {g.RANDU_A}, {g.RANDU_M}')
CORE = [g.returns, g.joint_returns, g.acf, g.max_drawdown, g.gbm_params, g.simulate_gbm, g.lcg, g.lcg_period, g.randu,
        g.binomial_crr, g.ar1, g.lag_dependence, g.conditional_moments, g.pair_stats, g.yearly_drawdowns]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 4: Probability\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Random variables, distributions, expectation, variance, covariance and correlation on real returns.\n"
       "- Independence vs uncorrelatedness, conditional expectation and conditional variance (the basis of GARCH).\n"
       "- Random numbers, the inverse transform and Monte Carlo; the binomial process; white noise, random walk, AR(1).\n"
       "- The Wiener process and geometric Brownian motion (GBM) against real paths of the S&P 500, the BET and Bitcoin.\n"
       "- Textbook: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 3-5."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- `returns(name)`: daily log returns in %, from 2000 (or from the first day of the series).\n"
       "- `joint_returns(a, b)`: returns of two series on their common trading days (prices joined first).\n"
       "- `gbm_params(r)`: annual log drift, volatility and arithmetic drift $\\mu = \\mu_{\\log} + \\sigma^2/2$; "
       "`simulate_gbm`: exact GBM simulation.\n"
       "- `lcg`, `randu`: linear congruential generators; `binomial_crr`: the Cox-Ross-Rubinstein tree; `ar1`: AR(1).\n"
       "- `max_drawdown`: the largest fall from a running peak, $\\min_t(P_t/\\max_{s\\le t}P_s - 1)$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Events and random variables\n\n"
       "- Conditional probability of a down day given a down day yesterday.\n"
       "- A discrete random variable (up days in a week) and a continuous one (daily returns); the empirical CDF and "
       "the 1% quantile (VaR 1% = minus the 1% quantile)."),
    code(src(g.down_days, g.up_days_per_week, g.fig_random_variables, g.fig_cdf_quantile)),
    code("pd.DataFrame(down_days()).round(4)"),
    code("fig_random_variables(save=False)"),
    code("fig_cdf_quantile(save=False)"),
    md("## 2. Joint distributions and dependence\n\n"
       "- Covariance, correlation and the volatility of a 50/50 portfolio.\n"
       "- Uncorrelated is not independent: $X$ and $X^2$; returns and squared returns."),
    code(src(g.fig_joint, g.fig_uncorrelated_dependent)),
    code("pd.DataFrame(fig_joint(save=False)).round(4)"),
    code("fig_uncorrelated_dependent(save=False)"),
    md("## 3. Conditional expectation and conditional variance\n\n"
       "- Mean and standard deviation of today's return by quintile of yesterday's absolute return.\n"
       "- The law of total variance: within-quintile plus between-quintile variance."),
    code(src(g.fig_conditional)),
    code("res = fig_conditional(save=False)\n{k: {'sd by quintile': v['sd'].round(3), 'Q5/Q1': round(v['ratio'], 2), "
         "'within': round(v['within'], 4), 'between': round(v['between'], 5)} for k, v in res.items()}"),
    md("## 4. The law of large numbers and Monte Carlo\n\n"
       "- The running mean of S&P 500 returns: the expected return is estimated very imprecisely.\n"
       "- Monte Carlo estimate of a 21-day loss probability and its standard error $\\sqrt{p(1-p)/N}$."),
    code(src(g.fig_lln, g.mc_loss_probability, g.fig_monte_carlo)),
    code("fig_lln(save=False)"),
    code("fig_monte_carlo(save=False)"),
    md("## 5. Random number generators and the inverse transform\n\n"
       "- Linear congruential generators and RANDU (ports of SFErangen1, SFErangen2 and SFErandu).\n"
       "- The inverse transform: exponential, Normal and empirical (historical simulation)."),
    code(src(g.fig_randu, g.fig_inverse_transform)),
    code("fig_randu(save=False)"),
    code("fig_inverse_transform(save=False)"),
    md("## 6. The binomial process\n\n- Cox-Ross-Rubinstein tree calibrated to the S&P 500 (port of SFEBinomp)."),
    code(src(g.fig_binomial)),
    code("fig_binomial(save=False)"),
    md("## 7. Discrete-time processes\n\n"
       "- White noise, AR(1) and random walk driven by the same shocks.\n"
       "- Which one is real? The S&P 500 among three GBM paths."),
    code(src(g.fig_processes, g.fig_spot_the_real)),
    code("fig_processes(save=False)"),
    code("out = fig_spot_the_real(save=False)\nprint('the real series is column', out['real'])\n"
         "pd.DataFrame({'excess kurtosis': out['kurt'], 'Corr(r_t^2, r_t-1^2)': out['acf2']}, index=list('ABCD')).round(3)"),
    md("## 8. The Wiener process and geometric Brownian motion\n\n"
       "- Scaled random walks converge to the Wiener process (port of SFEWienerProcess).\n"
       "- GBM calibrated to the S&P 500 (port of SFEsimGBM); real paths in GBM fans; what GBM misses."),
    code(src(g.scaled_random_walk, g.fig_wiener, g.fig_gbm_paths, g.fig_gbm_fan, g.gbm_check, g.fig_gbm_check)),
    code("fig_wiener(save=False)"),
    code("fig_gbm_paths(save=False)"),
    code("fig_gbm_fan(save=False)"),
    code("res = fig_gbm_check(save=False)\npd.DataFrame({k: v['real'] for k, v in res.items()}).round(3)"),
    md("## 9. AI for scientific discovery: does GBM get drawdown risk right?\n\n"
       "- Open question: how often does a calendar year contain a drawdown beyond 20% or 40%, in data and under GBM?\n"
       "- Starter code: within-year maximum drawdowns and their GBM distribution.\n"
       "- Things to check before trusting any answer, yours or an AI's: the definition of the drawdown, the calibration, "
       "the small number of years, and the effect of volatility clustering."),
    code(src(g.gbm_yearly_drawdown, g.fig_drawdown_years)),
    code("res = fig_drawdown_years(save=False)\n"
         "pd.DataFrame({k: {c: v[c] for c in ['n_years', 'share_real', 'p_gbm', 'n_hit40', 'p40_gbm', 'binom_p']} "
         "for k, v in res.items()}).round(4)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.discrete_moments, s.joint_table, s.portfolio_sd, s.mixture_moments, s.binomial_tree, s.ar1_moments,
               s.inverse_transform_examples, s.gbm_by_hand]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 4: Probability\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 4: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations and derivations on paper, checked in code. Part B: simulation and real data, each with "
       "an interpretation question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Daily log returns, joint returns, ACF, maximum drawdown, GBM calibration and simulation, LCG and RANDU, the "
       "binomial tree, AR(1), lag-1 dependence, conditional moments by quintile, pair statistics, yearly drawdowns.\n"
       "- Helpers for Part A: discrete moments, joint tables, portfolio volatility, a two-regime mixture, binomial "
       "trees, AR(1) moments, inverse transform, GBM by hand."),
    code(CONSTS + f"\nSEM_START = {s.SEM_START!r}\n\n\n" + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations and derivations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: moments of a discrete return\n\n"
       "**Context.** A daily return $X$ (in %) takes the values $-3$, $0$, $2$ with probabilities $0.2$, $0.3$, $0.5$.\n\n"
       "1. Compute $E[X]$, $E[X^2]$, $\\mathrm{Var}(X)$ and the standard deviation.\n"
       "2. Write the CDF $F(x)$ for every $x$.\n"
       "3. Compute $P(X < 0)$ and the median.\n\n"
       "**Report:** four moments, the CDF and two numbers."),
    code("# Solution\npd.Series(discrete_moments([-3, 0, 2], [0.2, 0.3, 0.5]))"),
    md("**Interpretation of the result.** The mean is positive although a loss is possible: risk and expected return are "
       "different questions."),
    md("## A2 [Proposed]: uncorrelated but dependent\n\n"
       "**Context.** $X \\in \\{-1, 0, 1\\}$, $Y \\in \\{0, 1\\}$ with $P(-1, 1) = 0.25$, $P(0, 0) = 0.5$, $P(1, 1) = 0.25$, "
       "other cells 0. Model: A1.\n\n"
       "1. Compute the marginal distributions of $X$ and $Y$ and their means.\n"
       "2. Compute $\\mathrm{Cov}(X, Y)$ and $\\rho$.\n"
       "3. Decide whether $X$ and $Y$ are independent, using one cell of the table.\n"
       "4. Compute $\\mathrm{Var}(X + Y)$.\n\n"
       "**Report:** two marginals, $\\rho$, a verdict and a variance."),
    code("# Solution\npd.Series(joint_table([-1, 0, 1], [0, 1], [[0, 0.25], [0.5, 0], [0, 0.25]]))"),
    md("## A3 [Solved]: volatility of a two-asset portfolio\n\n"
       "**Context.** Volatilities 20% and 30%, weights 0.6 and 0.4.\n\n"
       "1. Compute the portfolio volatility for $\\rho = 0.3$.\n"
       "2. Repeat for $\\rho = 0$, $-0.5$ and $1$.\n"
       "3. Explain when the portfolio is less risky than the weighted average of the volatilities.\n\n"
       "**Report:** four volatilities and one sentence."),
    code("# Solution\npd.Series({rho: portfolio_sd(0.6, 20, 30, rho) for rho in (0.3, 0.0, -0.5, 1.0)}).round(2)"),
    md("**Interpretation of the result.** Only $\\rho = 1$ gives the weighted average 24%; every $\\rho < 1$ diversifies."),
    md("## A4 [Proposed]: two regimes and the law of total variance\n\n"
       "**Context.** Calm with probability 0.8, $r \\mid \\text{calm} \\sim N(0, 0.8^2)$; turbulent otherwise, "
       "$r \\mid \\text{turbulent} \\sim N(0, 2^2)$. Model: A3.\n\n"
       "1. Use the law of total variance to compute $\\mathrm{Var}(r)$ and the standard deviation.\n"
       "2. Compute $E[r^4]$ and the kurtosis $E[r^4]/\\mathrm{Var}(r)^2$.\n"
       "3. Explain why the mixture has heavy tails although each regime is Normal.\n\n"
       "**Report:** a variance, a kurtosis and one sentence."),
    code("# Solution\npd.Series(mixture_moments(0.8, 0.8, 2.0))"),
    md("## A5 [Solved]: a three-step binomial tree\n\n"
       "**Context.** $S_0 = 100$, $u = 1.1$, $d = 0.9$, $p = 0.55$, three independent steps.\n\n"
       "1. List the possible values of $S_3$ and their probabilities.\n"
       "2. Compute $E[S_3]$ from the table and from $S_0(pu + (1-p)d)^3$.\n"
       "3. Compute $P(S_3 < 100)$.\n"
       "4. Find the $p^*$ that makes $S_t$ a martingale.\n\n"
       "**Report:** a table, two expectations, a probability, $p^*$."),
    code("# Solution\nbinomial_tree(100, 1.1, 0.9, 0.55, 3)"),
    md("**Interpretation of the result.** With $p = 0.55 > p^* = 0.5$ the price drifts up; under $p^*$ it is a fair game."),
    md("## A6 [Proposed]: random walk, martingale and AR(1)\n\n"
       "**Context.** $S_t = S_{t-1} + \\varepsilon_t$; $X_t = 0.2 + 0.6X_{t-1} + \\varepsilon_t$ with $\\sigma = 1$. Model: A5.\n\n"
       "1. Compute $E[S_t]$ and $\\mathrm{Var}(S_t)$, and show that $S_t$ is a martingale.\n"
       "2. Show that $S_t^2 - t\\sigma^2$ is also a martingale.\n"
       "3. Compute the mean, the variance and $\\rho(1), \\rho(2), \\rho(5)$ of the stationary $X_t$.\n"
       "4. Compute $E[X_{t+1} \\mid X_t = 2]$ and $\\mathrm{Var}(X_{t+1} \\mid X_t = 2)$, and check the moments by simulation.\n\n"
       "**Report:** two derivations, five moments, a forecast."),
    code("# Solution\nprint(ar1_moments(0.2, 0.6, 1.0, x_now=2.0))\n"
         "x = ar1(0.6, np.random.default_rng(1).standard_normal(200000)) + 0.2 / 0.4\n"
         "print('simulated mean %.3f, variance %.3f, rho(1) %.3f' % (x[1000:].mean(), x[1000:].var(), acf(x[1000:], 1)[0]))"),
    md("## A7 [Solved]: the inverse transform step by step\n\n"
       "**Context.** Uniform numbers $u = 0.3$, $0.9$ (and $0.15$, $0.45$, $0.8$).\n\n"
       "1. Turn $u = 0.3$ and $u = 0.9$ into exponential draws with rate $\\lambda = 0.5$.\n"
       "2. Turn $u = 0.15, 0.45, 0.8$ into draws of the return $X$ of A1.\n"
       "3. Turn $u = 0.01$ into a draw of $N(0.05, 1.2^2)$.\n\n"
       "**Report:** six simulated values."),
    code("# Solution\nprint(inverse_transform_examples())\n"
         "F = np.cumsum([0.2, 0.3, 0.5]); vals = np.array([-3, 0, 2])\n"
         "print('A1 draws:', [vals[np.searchsorted(F, u)] for u in (0.15, 0.45, 0.8)])"),
    md("**Interpretation of the result.** A large $u$ gives a large draw: the inverse transform keeps the order of the "
       "uniforms."),
    md("## A8 [Proposed]: GBM and the Wiener process\n\n"
       "**Context.** GBM with $\\mu = 8\\%$, $\\sigma = 20\\%$, $S_0 = 100$; $W$ a Wiener process. Model: A7.\n\n"
       "1. Compute the mean and the median of $S_1$ and of $S_5$.\n"
       "2. Compute $P(S_T < 100)$ for $T = 1$ and $T = 5$.\n"
       "3. Compute the 5% quantile of $S_1$ and of $S_5$.\n"
       "4. Compute $\\mathrm{Var}(W(2) - W(0.5))$ and $\\mathrm{Cov}(W(1), W(3))$.\n\n"
       "**Report:** eight GBM numbers and two Wiener moments."),
    code("# Solution\npd.DataFrame({'T = 1': gbm_by_hand(100, 0.08, 0.20, 1), 'T = 5': gbm_by_hand(100, 0.08, 0.20, 5)}).round(4)"),
    # ---------------- Part B
    md("# Part B: simulation, real data and interpretation"),
    md("## B1 [Solved]: are S&P 500 returns independent?\n\n"
       "**Question.** Are the daily returns of the S&P 500 (2000-2026) independent, and what does yesterday tell us "
       "about today?\n\n"
       "1. Compute $\\mathrm{Corr}(r_t, r_{t-1})$ and $\\mathrm{Corr}(r_t^2, r_{t-1}^2)$ and compare them with the band "
       "$\\pm 1.96/\\sqrt{n}$.\n"
       "2. Split the days into quintiles of $|r_{t-1}|$ and compute the mean and the standard deviation of $r_t$ in each.\n"
       "3. Check the law of total variance with the quintiles as groups.\n"
       "4. Interpretation: which is predictable from yesterday, the mean or the variance of today's return?\n\n"
       "**Report:** two correlations, a table of five quintiles, one decomposition, one sentence."),
    code("# Solution\n" + src(s.b1_dependence) + "\n\n\nres = b1_dependence(save=False)\n"
         "print({k: res[k] for k in ['n', 'rho_r', 'rho_r2', 'band', 'ratio', 'within', 'between', 'total']})\n"
         "pd.DataFrame({'mean': res['mean'], 'sd': res['sd']}, index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).round(3)"),
    md("**Interpretation of the result.** The variance is predictable from yesterday, the mean almost not: returns are "
       "close to uncorrelated but not independent."),
    md("## B2 [Proposed]: the same for DAX, BET and Bitcoin\n\n"
       "**Question.** Do the DAX, the BET and Bitcoin show the same pattern as the S&P 500? Model: B1.\n\n"
       "1. Compute both lag-1 correlations and the band for each series.\n"
       "2. Compute the conditional standard deviation in Q1 and Q5 and their ratio.\n"
       "3. Compute the share of the variance explained by the quintile means.\n"
       "4. Interpretation: why is the BET the only series with a clearly positive $\\mathrm{Corr}(r_t, r_{t-1})$?\n\n"
       "**Report:** one table and one sentence."),
    code("# Solution\n" + src(s.b2_dependence) + "\n\n\npd.DataFrame(b2_dependence()).round(4)"),
    md("## B3 [Solved]: a monthly loss probability by Monte Carlo\n\n"
       "**Question.** How likely is a 21-day loss of more than 10% for the S&P 500, and does the answer depend on the model?\n\n"
       "1. Compute the exact probability under i.i.d. Normal daily returns.\n"
       "2. Estimate it by Monte Carlo under the same model and give the standard error.\n"
       "3. Estimate it by resampling 21 observed days with replacement (the empirical inverse transform).\n"
       "4. Compute the share of overlapping 21-day windows in the data with a loss beyond 10%.\n"
       "5. Interpretation: which differences are simulation error and which are model error?\n\n"
       "**Report:** four probabilities, two standard errors, one sentence."),
    code("# Solution\n" + src(s.b3_monte_carlo) + "\n\n\npd.Series(b3_monte_carlo(save=False))"),
    md("**Interpretation of the result.** Monte Carlo and the exact value differ by less than one standard error "
       "(simulation error); the bootstrap differs from the Normal model by several standard errors (model error)."),
    md("## B4 [Proposed]: test a random number generator\n\n"
       "**Question.** Can a generator pass simple tests and still be unusable? Model: B3.\n\n"
       "1. Compute the mean, the variance and the lag-1 correlation of RANDU and PCG64 (100,000 numbers, seed 1).\n"
       "2. Count the distinct values of $9u_k - 6u_{k+1} + u_{k+2}$ for both generators.\n"
       "3. Find the period of the LCGs $(5, 1, 16)$, $(5, 0, 16)$ and $(4, 1, 16)$ started at 7.\n"
       "4. Interpretation: why do one-dimensional tests not detect the flaw of RANDU?\n\n"
       "**Report:** a table of six statistics, two counts, three periods, one sentence."),
    code("# Solution\n" + src(s.b4_randu) + "\n\n\npd.Series(b4_randu())"),
    md("## B5 [Solved]: is the BET a geometric Brownian motion?\n\n"
       "**Question.** Which features of the BET can a GBM with the same drift and volatility reproduce?\n\n"
       "1. Calibrate GBM: annual log drift and volatility from the daily log returns.\n"
       "2. Simulate 500 paths of the same length and compute the excess kurtosis, $\\mathrm{Corr}(r_t^2, r_{t-1}^2)$ and "
       "the maximum drawdown of each.\n"
       "3. Locate the real values in the simulated distributions (5% and 95% quantiles).\n"
       "4. Interpretation: which stylised facts does GBM miss, and which risk number is most affected?\n\n"
       "**Report:** two parameters, three comparisons, one sentence."),
    code("# Solution\n" + src(s.b5_gbm_check) + "\n\n\nres = b5_gbm_check(save=False)\n"
         "print({k: res[k] for k in ['mu_log', 'sigma', 'n']})\n"
         "pd.DataFrame({k: {'real': res['real'][k], **res[k]} for k in ['kurt', 'acf2', 'dd']}).round(3)"),
    md("**Interpretation of the result.** GBM misses heavy tails and volatility clustering completely; the 2008 crash "
       "makes the real drawdown deeper than every simulated path."),
    md("## B6 [Proposed]: does variance grow linearly with the horizon?\n\n"
       "**Question.** Does $\\mathrm{Var}(h\\text{-day return}) = h\\,\\mathrm{Var}(1\\text{-day return})$ hold for the "
       "S&P 500, the BET and Bitcoin? Model: B5.\n\n"
       "1. Compute the ratio for $h = 5, 21, 63$ with non-overlapping blocks.\n"
       "2. Use the approximate standard error $\\sqrt{2/(m-1)}$ with $m$ blocks to judge the distance from 1.\n"
       "3. Interpretation: what does a ratio above or below 1 say about the autocorrelation of returns?\n\n"
       "**Report:** a 3 x 3 table and two sentences."),
    code("# Solution\n" + src(s.b6_variance_scaling) + "\n\n\n"
         "pd.DataFrame({k: {h: v[h]['ratio'] for h in v} for k, v in b6_variance_scaling().items()}).round(3)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: how often is there a bad year?\n\n"
       "**Question.** Does GBM give the right probability of a drawdown beyond 20% (or 40%) within one calendar year? "
       "Models: B3, B5.\n\n"
       "1. Compute the maximum drawdown within each calendar year and the share of years beyond 20% and 40%.\n"
       "2. Simulate 10,000 one-year paths under GBM and under the i.i.d. bootstrap and compute the same probabilities.\n"
       "3. Compare data and models with a binomial test.\n"
       "4. Propose one model that could reproduce both numbers.\n"
       "5. Interpretation: what does the gap between the bootstrap and the data say about volatility clustering?\n\n"
       "**Report:** a table, two binomial $p$-values, a short project plan."),
    code("# Reference analysis\n" + src(s.c1_drawdowns) + "\n\n\nres = c1_drawdowns()\n"
         "print({k: stats.binomtest(v['n20'], v['n_years'], v['gbm20']).pvalue for k, v in res.items()})\n"
         "pd.DataFrame(res).round(4)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the probability facts about the S&P 500 for a risk report:\n\n"
       "- (a) the lag-1 correlation of returns is close to zero, so returns are independent and tomorrow's variance "
       "cannot be predicted;\n"
       "- (b) the log price is a random walk, so its variance is the same at every date and the process is stationary;\n"
       "- (c) under GBM with the S&P 500 parameters, the median value of 100 after 10 years is $100e^{10\\mu}$;\n"
       "- (d) a Monte Carlo probability of about 3% from 10,000 paths has a standard error of about 0.0017;\n"
       "- (e) a 50/50 portfolio of S&P 500 and DAX has volatility $\\sqrt{0.25\\sigma_1^2 + 0.25\\sigma_2^2}$.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\npd.Series(c2_check())"),
]

if __name__ == '__main__':
    build(LECTURE, 4, 'lecture')
    build(SEMINAR, 4, 'seminar')
