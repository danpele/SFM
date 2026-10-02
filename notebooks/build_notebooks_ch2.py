"""
build_notebooks_ch2.py -- lecture and seminar notebooks of Chapter 2 (SFM): Classical distributions and stylised facts
====================================================================================================================
Output: notebooks/EN/chapter2_lecture_notebook.ipynb, notebooks/EN/chapter2_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_02/generate_all_charts.py and seminar2.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch2.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter2_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter2_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 2
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(2)
import generate_all_charts as g   # noqa: E402
import seminar2 as s              # noqa: E402

CONSTS = (f'START = {g.START!r}\nASSETS = {g.ASSETS!r}\nINDICES = {g.INDICES!r}\nCOLORS = {g.COLORS!r}\n'
          f'PAL = {g.PAL!r}\nSHORT = {g.SHORT!r}\nHORIZONS = {g.HORIZONS!r}')
CORE = [g.returns, g.moments, g.t_fit, g.sigma_days, g.normal_tail_table, g.acf, g.ljung_box, g.leverage, g.aggregate,
        g.qq_points]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 2: Classical distributions and stylised facts\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The Normal and lognormal distributions, the central limit theorem (CLT) and the Normal approximation.\n"
       "- Moments (skewness, excess kurtosis), the Jarque-Bera test, QQ plots and the Student-t distribution.\n"
       "- Six stylised facts of returns (Cont, 2001) on the S&P 500, DAX, BET, Bitcoin and four BVB stocks, 2010-2026.\n"
       "- Textbook: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Sec. 3.3 and 11.2."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- `returns(name)`: daily log returns in %, from 2010 (or from the first day of the series).\n"
       "- Sample skewness $S = m_3/m_2^{3/2}$, excess kurtosis $K = m_4/m_2^2 - 3$, Jarque-Bera "
       "$JB = \\frac{n}{6}(S^2 + K^2/4)$.\n"
       "- Student-t fit by maximum likelihood (`scipy.stats.t.fit`), AIC $= 2k - 2\\ell$.\n"
       "- $k$-sigma days: $|r_t - \\bar r| > k s$; ACF $\\hat\\rho(h)$, Ljung-Box $Q(m)$, leverage $L(k) = "
       "\\mathrm{Corr}(r_t, |r_{t+k}|)$; non-overlapping $h$-day returns."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. The Normal distribution\n\n"
       "- The 68-95-99.7 rule and the two-sided tail probabilities $P(|Z| > k)$.\n"
       "- Under the Normal distribution a 4-sigma day happens once in about 63 years of trading."),
    code(src(g.fig_normal_pdf)),
    code("pd.DataFrame(fig_normal_pdf(save=False)).T"),
    code(src(g.loss_days) + "\n\n\nloss_days()"),
    md("## 2. The lognormal distribution\n\n"
       "- If $\\ln Y \\sim N(m, s^2)$: median $e^m$, mean $e^{m + s^2/2}$, mode $e^{m - s^2}$ (port of SFElognormal)."),
    code(src(g.fig_lognormal)),
    code("pd.DataFrame(fig_lognormal(save=False)).T"),
    md("## 3. The central limit theorem\n\n"
       "- Standardised means of Bernoulli(0.2) draws (port of SFEclt).\n"
       "- Normal approximation of the binomial distribution (port of SFENormalApprox1-4)."),
    code(src(g.fig_clt, g.fig_normal_approx)),
    code("fig_clt(save=False)"),
    code("fig_normal_approx(save=False)"),
    md("## 4. Moments, the Jarque-Bera test and a data check\n\n"
       "- Moments of eight return series; the kernel density of DAX returns (port of SFEDaxReturnDistribution).\n"
       "- Banca Transilvania, May 2016: two misplaced adjustment days and their effect on the kurtosis."),
    code(src(g.moments_table, g.fig_shapes, g.fig_dax_density, g.tlv_check)),
    code("tab = moments_table()\ntab[['n', 'mean', 'sd', 'skew', 'exkurt', 'se_kurt', 'jb', 'jb_p', 'min', 'min_date']].round(3)"),
    code("fig_shapes(save=False)"),
    code("fig_dax_density(save=False)"),
    code("tlv_check()"),
    md("## 5. QQ plots and the Student-t distribution\n\n"
       "- QQ plots against the Normal distribution and against the fitted Student-t.\n"
       "- Student-t densities with unit variance; the fitted degrees of freedom and the AIC of both models."),
    code(src(g.fig_qq, g.fig_t_densities, g.fig_t_fit)),
    code("fig_qq(dist='norm', save=False)"),
    code("fig_t_densities(save=False)"),
    code("tab[['nu', 'loc', 'scale', 'll_n', 'll_t', 'aic_n', 'aic_t']].round(2)"),
    code("fig_t_fit(save=False)"),
    code("fig_qq(dist='t', save=False)"),
    md("## 6. Stylised facts (Cont, 2001)\n\n"
       "- Heavy tails: $k$-sigma days against the Normal expectation.\n"
       "- Aggregational Gaussianity, absence of linear autocorrelation, volatility clustering, leverage effect, "
       "gain/loss asymmetry; a summary table."),
    code(src(g.sigma_table, g.fig_sigma_days, g.fig_aggregation, g.fig_qq_horizons, g.fig_clustering, g.fig_acf,
             g.fig_leverage, g.fig_gain_loss, g.facts_table)),
    code("sigma_table().round(3).T"),
    code("fig_sigma_days(save=False);"),
    code("fig_aggregation(save=False);"),
    code("fig_qq_horizons(save=False)"),
    code("fig_clustering(save=False)"),
    code("res = fig_acf(save=False)\n{k: {'Q(10) returns': round(v['lb_r']['Q'], 1), 'Q(10) |returns|': round(v['lb_abs']['Q'], 1)} "
         "for k, v in res.items()}"),
    code("fig_leverage(save=False);"),
    code("fig_gain_loss(save=False);"),
    code("facts_table().round(3).T"),
    md("## 7. AI for scientific discovery: are Bitcoin's tails getting thinner?\n\n"
       "- Open question: does the tail of a crypto asset thin out as its market matures?\n"
       "- Starter code: excess kurtosis and Student-t degrees of freedom by calendar year.\n"
       "- Things to check before trusting any answer, yours or an AI's: the largest moves, the convergence of the "
       "optimiser, the noise of yearly estimates, calm vs turbulent years."),
    code(src(g.fig_tails_by_year)),
    code("yrs = fig_tails_by_year(save=False)\n"
         "nu = pd.Series({y: v['nu'] for y, v in yrs['btc'].items()})\n"
         "print('Spearman rank correlation of nu with the year: %.2f (p = %.2f)' % stats.spearmanr(nu.index, nu.values))\n"
         "pd.DataFrame({k: {y: v['nu'] for y, v in d.items()} for k, d in yrs.items()}).round(2)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.normal_day, s.lognormal_price, s.binomial_normal, s.sum_of_returns, s.jb_by_hand, s.t_moments,
               s.bootstrap_moments]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 2: Classical distributions and stylised facts\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 2: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations and derivations on paper, checked in code. Part B: real data with inference and an "
       "interpretation question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Daily log returns, moments, Jarque-Bera, Student-t fit, $k$-sigma days, ACF, Ljung-Box, leverage, $h$-day returns, QQ points.\n"
       "- Helpers for Part A (Normal, lognormal, binomial, Jarque-Bera, Student-t) and the bootstrap of skewness and kurtosis."),
    code(CONSTS + f"\nSEM_START = {s.SEM_START!r}\n\n\n" + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations and derivations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: a large daily loss under the Normal distribution\n\n"
       "**Context.** Daily log returns of an index are $N(0.05\\%, 1.1\\%^2)$; 252 trading days a year.\n\n"
       "1. Compute the probability of a daily loss larger than 2% and the expected number of such days per year.\n"
       "2. Compute the 1% quantile of the daily return.\n"
       "3. Compute the expected number of 4-sigma days in 10 years.\n\n"
       "**Report:** a probability, a number of days, a quantile and an expected count."),
    code("# Solution\npd.Series(normal_day(0.05, 1.1, 2.0, 252))"),
    md("**Interpretation of the result.** Under the Normal distribution most decades would have no 4-sigma day at all."),
    md("## A2 [Proposed]: the same questions for Bitcoin\n\n"
       "**Context.** Daily log returns of Bitcoin are $N(0.12\\%, 3.6\\%^2)$; 365 trading days a year. Model: A1.\n\n"
       "1. Compute the probability of a daily loss larger than 10% and the expected number of such days per year.\n"
       "2. Compute the 1% quantile of the daily return.\n"
       "3. Compute the expected number of 4-sigma days in 10 years.\n\n"
       "**Report:** a probability, a number of days, a quantile and an expected count."),
    code("# Solution\npd.Series(normal_day(0.12, 3.6, 10.0, 365))"),
    md("## A3 [Solved]: lognormal prices after one and ten years\n\n"
       "**Context.** A stock costs 100 lei; its yearly log returns are i.i.d. $N(0.07, 0.18^2)$.\n\n"
       "1. Compute the median and the mean of the price after one year.\n"
       "2. Compute the probability that the price after one year is below 100 lei.\n"
       "3. Compute the 5% quantile of the price after one year.\n"
       "4. Repeat 1-3 for ten years.\n\n"
       "**Report:** eight numbers and one sentence on the horizon."),
    code("# Solution\npd.DataFrame({'1 year': lognormal_price(100, 0.07, 0.18, 1), '10 years': lognormal_price(100, 0.07, 0.18, 10)}).round(4)"),
    md("**Interpretation of the result.** A loss becomes less likely with time, but the bad 5% case barely improves."),
    md("## A4 [Proposed]: why the mean is $e^{m + s^2/2}$\n\n"
       "**Context.** $X \\sim N(m, s^2)$, $Y = e^X$. Model: A3.\n\n"
       "1. Complete the square in $\\int e^x f(x)\\,dx$ to show that $E[e^X] = e^{m + s^2/2}$.\n"
       "2. Derive $\\mathrm{Var}(Y) = (e^{s^2} - 1)e^{2m + s^2}$.\n"
       "3. For $m = 0.07$, $s = 0.18$, compute the mean and the median of $Y$ and check them by simulation.\n\n"
       "**Report:** the two derivations and one sentence linking the gap to the volatility drag."),
    code("# Solution\nrng = np.random.default_rng(0)\ny = np.exp(rng.normal(0.07, 0.18, 1_000_000))\n"
         "print('mean: formula %.4f, simulation %.4f' % (np.exp(0.07 + 0.18**2 / 2), y.mean()))\n"
         "print('median: formula %.4f, simulation %.4f' % (np.exp(0.07), np.median(y)))\n"
         "print('variance: formula %.4f, simulation %.4f' % ((np.exp(0.18**2) - 1) * np.exp(2 * 0.07 + 0.18**2), y.var()))"),
    md("## A5 [Solved]: up days in a year (Normal approximation)\n\n"
       "**Context.** An index rises on a day with probability 0.53, independently; 252 trading days.\n\n"
       "1. Give the distribution of the number $X$ of up days, its mean and standard deviation.\n"
       "2. Approximate $P(X \\ge 140)$ with the Normal distribution and the continuity correction.\n"
       "3. Compare with the exact binomial value.\n\n"
       "**Report:** two moments, two probabilities, one sentence."),
    code("# Solution\npd.Series(binomial_normal(252, 0.53, 140))"),
    md("**Interpretation of the result.** $np(1-p) \\approx 63$ is large: the approximation error is in the fourth decimal."),
    md("## A6 [Proposed]: is a monthly return Normal?\n\n"
       "**Context.** Daily log returns are i.i.d. with mean 0.04% and standard deviation 1.2%. Model: A5.\n\n"
       "1. Give the approximate distribution of the 21-day log return.\n"
       "2. Compute the probability of a 21-day loss larger than 10%, and of a 5-day loss larger than 10%.\n"
       "3. Say whether the CLT still applies if the daily returns are Student-t with $\\nu = 3$.\n"
       "4. Say whether it applies with $\\nu = 1.5$.\n\n"
       "**Report:** two distributions, two probabilities, two answers with a reason each."),
    code("# Solution\nprint(sum_of_returns(0.04, 1.2, 21, 10.0))\nprint(sum_of_returns(0.04, 1.2, 5, 10.0))\n"
         "# nu = 3: finite variance, the CLT applies (slowly); nu = 1.5: infinite variance, it does not"),
    md("## A7 [Solved]: Jarque-Bera on paper\n\n"
       "**Context.** $n = 4000$ daily returns with skewness $-0.6$ and excess kurtosis $12$.\n\n"
       "1. Compute the standard errors of $S$ and $K$ under normality and the two $z$ statistics.\n"
       "2. Compute JB and the share of it that comes from the kurtosis.\n"
       "3. Decide at the 5% level.\n\n"
       "**Report:** two standard errors, two $z$ values, JB, the decision."),
    code("# Solution\npd.Series(jb_by_hand(4000, -0.6, 12.0))"),
    md("**Interpretation of the result.** JB is the sum of the two squared $z$ statistics; here almost all of it comes "
       "from the kurtosis."),
    md("## A8 [Proposed]: moments of the Student-t\n\n"
       "**Context.** $T \\sim t(\\nu)$; a model for daily returns is $r = m + sT$. Model: A7.\n\n"
       "1. Compute the variance and the excess kurtosis of $t(5)$ and $t(10)$.\n"
       "2. Find $\\nu$ such that the excess kurtosis is 3.\n"
       "3. Find the scale $s$ that gives a daily standard deviation of 1.2% for $\\nu = 5$ and $\\nu = 6$.\n"
       "4. Compare $P(|X| > 4)$ for a unit-variance $t(5)$ with the Normal value.\n\n"
       "**Report:** four moments, one $\\nu$, two scales and one ratio."),
    code("# Solution\npd.DataFrame({f't({nu})': t_moments(nu, 1.2) for nu in (5, 6, 10)}).round(4)"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: how non-Normal is the S&P 500?\n\n"
       "**Question.** How far are the daily log returns of the S&P 500 (2010-2026) from the Normal distribution, and how "
       "precisely do we know it?\n\n"
       "1. Compute the mean, standard deviation, skewness, excess kurtosis and the worst day with its date.\n"
       "2. Compute the Normal-theory standard errors of $S$ and $K$, and the JB statistic.\n"
       "3. Bootstrap $S$ and $K$ with 2000 resamples of the days and take the 2.5% and 97.5% percentiles.\n"
       "4. Interpretation: why is the bootstrap interval of $K$ so much wider than the Normal-theory one?\n\n"
       "**Report:** a table of moments, JB, two intervals for $K$, one sentence."),
    code("# Solution\n" + src(s.b1_moments) + "\n\n\npd.Series(b1_moments(save=False))"),
    md("**Interpretation of the result.** $\\sqrt{24/n}$ assumes Normal data; with heavy tails the kurtosis depends on a "
       "few days, so its sampling variability is many times larger."),
    md("## B2 [Proposed]: three more series\n\n"
       "**Question.** Which of the BET, Bitcoin and OMV Petrom is farthest from the Normal distribution, and how much does "
       "one day matter? Model: B1.\n\n"
       "1. Compute $n$, mean, standard deviation, $S$, $K$ and JB for each series.\n"
       "2. Remove the single largest absolute return of each series and recompute $K$.\n"
       "3. Interpretation: what does the change in $K$ tell you about kurtosis as a summary of risk?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b2_table) + "\n\n\npd.DataFrame(b2_table()).T[['n', 'mean', 'sd', 'skew', 'exkurt', 'jb', "
         "'drop_date', 'drop_r', 'exkurt_drop1']]"),
    md("## B3 [Solved]: Normal or Student-t for the BET?\n\n"
       "**Question.** Does a Student-t describe the daily returns of the BET better than the Normal distribution, and "
       "where does each fail?\n\n"
       "1. Draw the QQ plot of the standardised returns against $N(0,1)$.\n"
       "2. Fit a Student-t by maximum likelihood and draw the QQ plot against the fitted t.\n"
       "3. Compare the log-likelihoods and the AIC of the two models.\n"
       "4. Interpretation: where does the Normal fail, and where does the fitted t fail?\n\n"
       "**Report:** $\\hat\\nu$, $\\hat m$, $\\hat s$, two AIC values, two QQ plots, one sentence."),
    code("# Solution\n" + src(s.b3_qq) + "\n\n\npd.Series(b3_qq(save=False))"),
    md("**Interpretation of the result.** The Normal misses both tails (S-shape); the t fits the centre and most of the "
       "tails, but its most extreme quantiles go slightly beyond the data."),
    md("## B4 [Proposed]: how many 4-sigma days?\n\n"
       "**Question.** How many 4-sigma days do the S&P 500, DAX, BET and Bitcoin have, and which model predicts the "
       "count? Model: B3.\n\n"
       "1. Count the days with $|r_t - \\bar r| > 4s$ in each series.\n"
       "2. Compute the expected count under the Normal distribution and the Poisson probability of the observed count.\n"
       "3. Compute the expected count under the fitted Student-t (same threshold).\n"
       "4. Interpretation: is the Student-t too heavy, too light or about right for each series?\n\n"
       "**Report:** one table and one sentence per series."),
    code("# Solution\n" + src(s.b4_sigma) + "\n\n\npd.DataFrame(b4_sigma()).T"),
    md("## B5 [Solved]: does the S&P 500 become Normal at longer horizons?\n\n"
       "**Question.** Are 5-day and 21-day returns of the S&P 500 closer to the Normal distribution than daily returns?\n\n"
       "1. Build the 5-day and 21-day log returns (non-overlapping blocks).\n"
       "2. Compute $n$, $S$, $K$, $\\sqrt{24/n}$ and JB with its $p$-value at $h = 1, 5, 21$.\n"
       "3. Draw the three QQ plots against $N(0,1)$.\n"
       "4. Interpretation: is the monthly distribution Normal, and why does the test have less power at $h = 21$?\n\n"
       "**Report:** a table with three rows, three QQ plots, two sentences."),
    code("# Solution\n" + src(s.b5_aggregation) + "\n\n\npd.DataFrame(b5_aggregation(save=False)).T"),
    md("**Interpretation of the result.** The kurtosis falls with the horizon, but monthly returns are still not Normal "
       "(March 2020); with 200 months the test detects only large departures."),
    md("## B6 [Proposed]: are BET returns independent?\n\n"
       "**Question.** Are the daily returns of the BET uncorrelated, and are they independent? Model: B5.\n\n"
       "1. Compute the ACF of $r_t$ and of $|r_t|$ at lags 1-10 and the band $\\pm 1.96/\\sqrt n$.\n"
       "2. Compute the Ljung-Box statistic $Q(10)$ for $r_t$ and for $|r_t|$ and compare with $\\chi^2_{0.95}(10)$.\n"
       "3. Interpretation: can returns be uncorrelated but not independent?\n\n"
       "**Report:** two ACF columns, two $Q$ values, one sentence."),
    code("# Solution\n" + src(s.b6_clustering) + "\n\n\nres = b6_clustering()\n"
         "print('band', round(res['band'], 3), '| Q(10) returns', res['lb_r'], '| Q(10) |returns|', res['lb_abs'])\n"
         "pd.DataFrame({'ACF r': res['rho_r'], 'ACF |r|': res['rho_abs']}, index=range(1, 11)).round(3)"),
    md("## B7 [Proposed]: is Bitcoin like a stock index?\n\n"
       "**Question.** Do the S&P 500 and Bitcoin both show the leverage effect and the gain/loss asymmetry? Model: B5.\n\n"
       "1. Compute $L(k) = \\mathrm{Corr}(r_t, |r_{t+k}|)$ for $k = 1..5$ and their mean, with the band $\\pm 1.96/\\sqrt n$.\n"
       "2. Count the days below $\\bar r - 4s$ and above $\\bar r + 4s$; binomial test of equal probabilities.\n"
       "3. Compare $|q_{0.01}|$ and $q_{0.99}$ and the skewness.\n"
       "4. Interpretation: which stylised facts of stock indices does Bitcoin share?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b7_asymmetry) + "\n\n\npd.DataFrame(b7_asymmetry()).T"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: are Bitcoin's tails getting thinner?\n\n"
       "**Question.** Has the tail of Bitcoin's daily returns become thinner since 2015? Models: B3, B5.\n\n"
       "1. Fit a Student-t and compute the excess kurtosis for each calendar year, for Bitcoin and the S&P 500.\n"
       "2. Compute the Spearman rank correlation between the year and $\\hat\\nu$.\n"
       "3. Fit one t on 2015-2020 and one on 2021-2026 and compare $\\hat\\nu$.\n"
       "4. Propose one way to make the answer more reliable.\n"
       "5. Interpretation: what can a dozen yearly estimates tell us about a trend?\n\n"
       "**Report:** a table, the rank correlation with its $p$-value, a short project plan."),
    code("# Reference analysis\n" + src(s.c1_tails_by_year) + "\n\n\nres = c1_tails_by_year()\n"
         "print({k: (round(v['spearman'], 2), round(v['p'], 2), round(v['nu_first'], 2), round(v['nu_second'], 2)) for k, v in res.items()})\n"
         "pd.DataFrame({k: {y: r['nu'] for y, r in v['rows'].items()} for k, v in res.items()}).round(2)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant described the daily S&P 500 returns of 2010-2026 and claimed:\n\n"
       "- (a) since the Normal distribution has kurtosis 0, the kurtosis of the S&P 500 equals its excess kurtosis;\n"
       "- (b) a JB $p$-value of 0.000 proves that the returns follow a Student-t distribution;\n"
       "- (c) the observed number of 4-sigma days is just bad luck under the Normal distribution;\n"
       "- (d) the lag-1 autocorrelation is small, so returns are independent and volatility cannot be predicted;\n"
       "- (e) with yearly log returns $N(7\\%, 18\\%^2)$, the expected value of 100 after one year is $100e^{0.07}$.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\npd.Series(c2_check())"),
]

if __name__ == '__main__':
    build(LECTURE, 2, 'lecture')
    build(SEMINAR, 2, 'seminar')
