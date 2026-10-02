"""
build_notebooks_ch3.py -- lecture and seminar notebooks of Chapter 3 (SFM): alpha-stable distributions
=====================================================================================================
Output: notebooks/EN/chapter3_lecture_notebook.ipynb, notebooks/EN/chapter3_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_03/generate_all_charts.py and seminar3.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch3.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter3_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter3_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 3
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(3)
import generate_all_charts as g   # noqa: E402
import seminar3 as s              # noqa: E402

CONSTS = (f'START = {g.START!r}\nASSETS = {g.ASSETS!r}\nCOLORS = {g.COLORS!r}\nMODEL_COL = {g.MODEL_COL!r}\nSEED = {g.SEED!r}')
CORE = [g.stable_cf, g.s1_to_s0, g.s0_to_s1, g.StableGrid, g.stable_pdf, g.rstable, g.mcculloch, g.mcculloch_quantiles,
        g.stable_nll, g.stable_mle, g.fit_models, g.model_cdf, g.model_ppf, g.returns, g.aggregate]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 3: α-stable distributions\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Stability under summation and the generalised central limit theorem.\n"
       "- Parameters (α, β, γ, δ), the characteristic function, the S0 and S1 parameterisations (Nolan, 2020).\n"
       "- Power-law tails, infinite variance, simulation (Chambers–Mallows–Stuck), estimation (McCulloch, maximum likelihood).\n"
       "- Fits to daily log returns of the BET, S&P 500, DAX and Bitcoin, and the critique of the stable model.\n"
       "- References: Nolan (2020), *Univariate Stable Distributions*; Borak, Härdle and Weron (2005); "
       "Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- $X \\sim S(\\alpha, \\beta, \\gamma, \\delta; 1)$ (S1, $\\alpha \\ne 1$): "
       "$\\ln E e^{itX} = -\\gamma^\\alpha|t|^\\alpha[1 - i\\beta\\,\\mathrm{sign}(t)\\tan(\\pi\\alpha/2)] + i\\delta_1 t$.\n"
       "- S0: the same law shifted, $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$; the density is continuous in $\\alpha$.\n"
       "- `scipy.stats.levy_stable` uses S1 by default (`levy_stable.parameterization = 'S1'`).\n"
       "- Densities: FFT inversion of the characteristic function on a grid (`StableGrid`); "
       "simulation: `rstable` (Chambers–Mallows–Stuck); estimation: `mcculloch` and `stable_mle` (S0).\n"
       "- Returns: daily log returns in %, indices since 2000, Bitcoin since 2014."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Densities and parameters\n\n"
       "- The effect of α and β; the scale γ; S1 vs S0 near α = 1; the closed-form cases Normal, Cauchy, Lévy."),
    code(src(g.fig_density_alpha, g.fig_density_beta, g.fig_s0_s1, g.fig_special_cases)),
    code("fig_density_alpha(save=False)"),
    code("fig_density_beta(save=False)"),
    code("fig_s0_s1(save=False)"),
    code("fig_special_cases(save=False)"),
    md("## 2. Stability and the generalised CLT\n\n"
       "- Sums of $n$ i.i.d. stable draws divided by $n^{1/\\alpha}$ have the law of one draw.\n"
       "- Sums of Student-t(1.5) draws (infinite variance) approach a stable law, not the Normal distribution."),
    code(src(g.fig_stability_qq, g.fig_gclt)),
    code("fig_stability_qq(save=False)"),
    code("fig_gclt(save=False)"),
    md("## 3. Tails and moments\n\n"
       "- $P(X > x) \\sim \\gamma^\\alpha c_\\alpha (1 + \\beta) x^{-\\alpha}$, $c_\\alpha = \\Gamma(\\alpha)\\sin(\\pi\\alpha/2)/\\pi$.\n"
       "- $E|X|^p < \\infty$ only for $p < \\alpha$: the sample variance of a stable sample does not settle."),
    code(src(g.fig_tails_loglog, g.tail_table, g.fig_running_variance)),
    code("fig_tails_loglog(save=False)"),
    code("pd.DataFrame(tail_table()).T"),
    code("fig_running_variance(save=False)"),
    md("## 4. Simulation: Chambers–Mallows–Stuck\n\n"
       "- 100,000 draws of $S(1.5, 0.5, 1, 0; 1)$ against the exact density; a two-sample KS test against `levy_stable.rvs`."),
    code(src(g.fig_cms)),
    code("fig_cms(save=False)"),
    md("## 5. Estimation: McCulloch and maximum likelihood\n\n"
       "- McCulloch (1986): $\\nu_\\alpha = (q_{0.95} - q_{0.05})/(q_{0.75} - q_{0.25})$ and $\\nu_\\beta$, then the tables.\n"
       "- ML in S0 with an FFT density, standard errors from the numerical Hessian.\n"
       "- Monte Carlo: 200 samples of $S(1.7, 0, 1, 0)$ for $n = 500$ and $n = 2500$ (a few minutes)."),
    code(src(g.fig_mcculloch_map, g.estimator_mc, g.fig_estimator_mc)),
    code("real = {}\nfor k in ASSETS:\n    m = mcculloch(returns(k).values)\n    real[k] = (m['nu_alpha'], m['alpha'])\n"
         "    print(LABELS[k], {c: round(m[c], 3) for c in ['nu_alpha', 'nu_beta', 'alpha', 'beta', 'gamma', 'delta0']})\n"
         "fig_mcculloch_map(real, save=False)"),
    code("fig_estimator_mc(estimator_mc(), save=False)"),
    md("## 6. Fitting real returns\n\n"
       "- Stable (ML), Student-t and Normal fits; densities on a log scale, QQ plots, the left tail.\n"
       "- Observed vs expected numbers of large daily losses; VaR 1% and VaR 0.1% of the data and of each model."),
    code(src(g.fit_all, g.fits_table, g.fig_fit_density, g.fig_qq_real, g.fig_tails_real, g.tail_counts)),
    code("fits = fit_all()\nfits_table(fits).round(3).T"),
    code("fig_fit_density(fits, save=False)"),
    code("fig_qq_real(fits, save=False)"),
    code("fig_tails_real(fits, save=False)"),
    code("pd.DataFrame(tail_counts(fits)).round(2)"),
    md("## 7. The critique: does α stay the same when returns are summed?\n\n"
       "- Under i.i.d. stable returns, weekly and monthly sums keep the same α.\n"
       "- The running sample variance of the S&P 500 against paths simulated from its fitted stable law."),
    code(src(g.aggregation_alpha, g.fig_aggregation, g.fig_running_var_real)),
    code("agg = aggregation_alpha()\npd.DataFrame({(k, h): v for k, a in agg.items() for h, v in a.items()}).T.round(3)"),
    code("fig_aggregation(agg, save=False)"),
    code("fig_running_var_real(fits, save=False)"),
    md("## 8. AI for scientific discovery: have the tails of Bitcoin become lighter?\n\n"
       "- Open question: is the tail of Bitcoin returns lighter now than in its first years?\n"
       "- Starter code: the McCulloch α of Bitcoin daily log returns for each calendar year, with bootstrap intervals.\n"
       "- Check before trusting any answer, yours or an AI's: the parameterisation (S0 or S1), the units, the window "
       "length, and whether the change survives returns divided by a rolling volatility."),
    code("r = returns('btc')\nrng = np.random.default_rng(SEED)\nrows = {}\n"
         "for y, x in r.groupby(r.index.year):\n"
         "    if len(x) < 300:\n        continue\n"
         "    a = mcculloch(x.values)['alpha']\n"
         "    bs = [mcculloch(rng.choice(x.values, len(x)))['alpha'] for _ in range(300)]\n"
         "    rows[y] = {'n': len(x), 'alpha': a, 'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5)}\n"
         "pd.DataFrame(rows).T.round(3)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.sem_returns, s.stability_constant, s.cf_value, s.cms_draw, s.bootstrap_mcculloch]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 3: α-stable distributions\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 3: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with a measure of precision and an "
       "interpretation question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- The stable law: characteristic function, FFT density (`StableGrid`), simulation (`rstable`, Chambers–Mallows–Stuck).\n"
       "- Estimation: `mcculloch` (McCulloch, 1986), `stable_mle` (maximum likelihood, S0), `fit_models` (Normal, Student-t, stable).\n"
       "- Helpers for Part A and the bootstrap of the McCulloch α."),
    code(CONSTS + f"\nSEM_START = {s.SEM_START!r}\n\n\n" + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: the scale of a sum\n\n"
       "**Context.** $X_1, X_2$ are independent copies of $S(\\alpha, 0, 1, 0)$; $a = 3$, $b = 4$.\n\n"
       "1. Compute $c$ with $3X_1 + 4X_2 \\overset{d}{=} cX$ for $\\alpha = 2$, $1.5$ and $1$.\n"
       "2. For $\\alpha = 2$, check the result with variances.\n"
       "3. Explain why $c$ grows as $\\alpha$ falls.\n\n"
       "**Report:** three values of $c$ and one sentence."),
    code("# Solution\n{a: round(stability_constant(3, 4, a), 2) for a in (2.0, 1.5, 1.0)}"),
    md("**Interpretation of the result.** With small α, one large term can dominate the sum: the scale of the sum "
       "approaches the sum of the scales."),
    md("## A2 [Proposed]: horizons and diversification\n\n"
       "**Context.** Daily returns are i.i.d. $S(\\alpha, 0, \\gamma, 0)$. Model: A1.\n\n"
       "1. Compute the scale of the 20-day sum, as a multiple of γ, for $\\alpha = 2$, $1.7$ and $1.5$.\n"
       "2. Compute the scale of an equally weighted portfolio of $N = 10$ i.i.d. assets for $\\alpha = 2$, $1.5$ and $1$.\n"
       "3. Explain what happens to diversification when $\\alpha = 1$.\n\n"
       "**Report:** six multiples of γ and one sentence."),
    code("# Solution\nprint({a: round(20 ** (1 / a), 2) for a in (2.0, 1.7, 1.5)})\nprint({a: round(10 ** (1 / a - 1), 2) for a in (2.0, 1.5, 1.0)})"),
    md("## A3 [Solved]: S1 and S0 at $t = 1$\n\n"
       "**Context.** $X$ has $\\alpha = 1.5$, $\\beta = 0.3$, $\\gamma = 1$, $\\delta = 0$.\n\n"
       "1. Compute $\\varphi(1)$ in the S1 parameterisation.\n"
       "2. Compute $\\varphi(1)$ in the S0 parameterisation.\n"
       "3. Explain the difference using $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$.\n\n"
       "**Report:** two complex numbers and one sentence."),
    code("# Solution\nprint('S1:', np.round(cf_value(1.0, 1.5, 0.3, 1), 3), ' S0:', np.round(cf_value(1.0, 1.5, 0.3, 0), 3))\n"
         "print('shift beta * tan(pi alpha / 2) =', round(0.3 * np.tan(np.pi * 0.75), 3))"),
    md("**Interpretation of the result.** Same modulus $e^{-1}$: the two laws differ only by a shift of the location."),
    md("## A4 [Proposed]: special cases and a location conversion\n\n"
       "**Context.** Model: A3.\n\n"
       "1. Write $\\varphi(t)$ for $\\alpha = 2$ and compare it with $\\exp(i\\mu t - \\sigma^2 t^2/2)$, the characteristic function of $N(\\mu, \\sigma^2)$.\n"
       "2. Write $\\varphi(t)$ for $\\alpha = 1$, $\\beta = 0$ and name the law.\n"
       "3. scipy reports $\\alpha = 1.7$, $\\beta = -0.2$, $\\gamma = 0.6$, `loc` $= 0.05$; compute $\\delta_0$.\n\n"
       "**Report:** μ, σ², the name of the law and $\\delta_0$."),
    code("# Solution\nprint('delta0 =', round(s1_to_s0(1.7, -0.2, 0.6, 0.05), 4))"),
    md("## A5 [Solved]: how fast do the tails fall?\n\n"
       "**Context.** Daily losses have a power-law tail with $\\alpha = 1.7$; a loss above $x$ happens on 1 day in 50.\n\n"
       "1. Compute $P(L > 2x)/P(L > x)$.\n"
       "2. How often does a loss above $2x$ happen? And above $4x$?\n"
       "3. Compare with a Normal law that has the same frequency of losses above $x$.\n\n"
       "**Report:** one ratio, two frequencies and one sentence."),
    code("# Solution\nprint('ratio', round(2 ** -1.7, 3), '| above 2x: 1 in', round(50 * 2 ** 1.7), '| above 4x: 1 in', round(50 * 4 ** 1.7))\n"
         "z = stats.norm.isf(1 / 50)\nprint('Normal, above 2x: 1 in', round(1 / stats.norm.sf(2 * z)))"),
    md("**Interpretation of the result.** The power law keeps large losses frequent; under the Normal law they vanish."),
    md("## A6 [Proposed]: tails and moments for other α\n\n"
       "**Context.** A loss above $x$ happens on 1 day in 100. Model: A5.\n\n"
       "1. For $\\alpha = 1.3$ and $1.9$, compute how often losses above $2x$ and above $3x$ happen.\n"
       "2. For $\\alpha = 1.9$, $1.3$ and $0.8$, say whether the mean and the variance exist.\n"
       "3. Explain why a sample variance can be computed even when the variance does not exist.\n\n"
       "**Report:** four frequencies, a small table of moments and one sentence."),
    code("# Solution\n{a: {'2x': round(100 * 2 ** a), '3x': round(100 * 3 ** a), 'mean': a > 1, 'variance': a >= 2} for a in (1.3, 1.9, 0.8)}"),
    md("## A7 [Solved]: McCulloch from five quantiles\n\n"
       "**Context.** Quantiles of 2000 daily returns (%): $q_{0.05} = -2.10$, $q_{0.25} = -0.55$, $q_{0.5} = 0.05$, "
       "$q_{0.75} = 0.62$, $q_{0.95} = 2.06$.\n\n"
       "1. Compute $\\nu_\\alpha$ and $\\nu_\\beta$.\n"
       "2. Read $\\hat\\alpha$ from the table excerpt (column $\\nu_\\beta = 0$).\n"
       "3. Compute $\\hat\\gamma$ with Table V.\n\n"
       "**Report:** two ratios, $\\hat\\alpha$, $\\hat\\gamma$ and one sentence on the tails."),
    code("# Solution\npd.Series(mcculloch_quantiles(-2.10, -0.55, 0.05, 0.62, 2.06)).round(3)"),
    md("**Interpretation of the result.** $\\nu_\\alpha$ is far above 2.439 (the Normal value): heavy tails, $\\hat\\alpha \\approx 1.38$."),
    md("## A8 [Proposed]: McCulloch for a crypto asset\n\n"
       "**Context.** Quantiles (%): $q_{0.05} = -6.90$, $q_{0.25} = -1.45$, $q_{0.5} = 0.12$, $q_{0.75} = 1.75$, "
       "$q_{0.95} = 6.40$. Model: A7.\n\n"
       "1. Compute $\\nu_\\alpha$ and $\\nu_\\beta$.\n"
       "2. Read $\\hat\\alpha$ and compute $\\hat\\gamma$ from the table excerpts.\n"
       "3. Compare with A7: which asset has the heavier tails?\n\n"
       "**Report:** two ratios, $\\hat\\alpha$, $\\hat\\gamma$ and one sentence."),
    code("# Solution\npd.Series(mcculloch_quantiles(-6.90, -1.45, 0.12, 1.75, 6.40)).round(3)"),
    md("## A9 [Solved]: one stable draw by CMS\n\n"
       "**Context.** $\\alpha = 1.5$, $\\beta = 0$; a uniform draw $U = 0.8$ and an exponential draw $W = 1.2$.\n\n"
       "1. Compute $V = \\pi(U - 1/2)$.\n"
       "2. Compute the three factors of the CMS formula.\n"
       "3. Compute the draw $X$ and the draw of $S(1.5, 0, 0.6, 0.05)$.\n\n"
       "**Report:** V, three factors, two draws."),
    code("# Solution\nV = np.pi * (0.8 - 0.5)\nf1, f2, f3 = np.sin(1.5 * V), np.cos(V) ** (1 / 1.5), (np.cos(-0.5 * V) / 1.2) ** (-1 / 3)\n"
         "x = cms_draw(0.8, 1.2, 1.5)\nprint(round(V, 3), round(f1, 3), round(f2, 3), round(f3, 3), round(x, 3), round(0.6 * x + 0.05, 3))"),
    md("## A10 [Proposed]: CMS for the Normal and the Cauchy laws\n\n"
       "**Context.** Model: A9.\n\n"
       "1. Show that for $\\alpha = 2$ the CMS formula becomes $X = 2\\sin V\\sqrt{W}$.\n"
       "2. Show that for $\\alpha = 1$ it becomes $X = \\tan V$.\n"
       "3. Compute both draws for $U = 0.8$, $W = 1.2$.\n\n"
       "**Report:** two short derivations and two numbers."),
    code("# Solution\nV = np.pi * 0.3\nprint(round(cms_draw(0.8, 1.2, 2.0), 4), round(2 * np.sin(V) * np.sqrt(1.2), 4), round(cms_draw(0.8, 1.2, 1.0), 4))"),
    # ---------------- Part B
    md("# Part B: real data, precision and interpretation"),
    md("## B1 [Solved]: how heavy are the tails of the BET?\n\n"
       "**Question.** Which stable law do the McCulloch estimates give for the daily BET returns, and how precise is $\\hat\\alpha$?\n\n"
       "1. Compute the 5%, 25%, 50%, 75% and 95% quantiles and the ratios $\\nu_\\alpha$ and $\\nu_\\beta$.\n"
       "2. Compute the McCulloch estimates $\\hat\\alpha$, $\\hat\\beta$, $\\hat\\gamma$ and $\\hat\\delta_0$ with the full tables.\n"
       "3. Bootstrap $\\hat\\alpha$ with 1000 resamples of the days; report the standard error and the 95% percentile interval.\n"
       "4. Interpretation: is $\\alpha = 2$ (the Normal distribution) compatible with the data?\n\n"
       "**Report:** five quantiles, two ratios, four estimates, the interval and one sentence."),
    code("# Solution\n" + src(s.b1_mcculloch) + "\n\n\npd.Series(b1_mcculloch('bet', save=False))"),
    md("**Interpretation of the result.** α = 2 lies far outside the bootstrap interval: the tails of the BET are much "
       "heavier than the Normal distribution allows."),
    md("## B2 [Proposed]: three more markets\n\n"
       "**Question.** Which of the S&P 500, DAX and Bitcoin has the heaviest tails by McCulloch's method? Model: B1.\n\n"
       "1. Repeat steps 1-3 of B1 for each series.\n"
       "2. Put $\\nu_\\alpha$, $\\hat\\alpha$ with its interval, $\\hat\\beta$ and $\\hat\\gamma$ of the four series (with the BET) in one table.\n"
       "3. Interpretation: is the difference between the $\\hat\\alpha$ of Bitcoin and of the S&P 500 larger than the estimation error?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b2_table) + "\n\n\npd.DataFrame(b2_table()).T[['nu_alpha', 'alpha', 'lo', 'hi', 'beta', 'gamma']].round(3)"),
    md("## B3 [Solved]: maximum likelihood for the S&P 500\n\n"
       "**Question.** Does the ML stable fit describe the S&P 500 better than the Normal distribution, in the body and in the tails?\n\n"
       "1. Fit the stable law by ML (S0); report $\\hat\\alpha$ and $\\hat\\beta$ with standard errors, $\\hat\\gamma$, $\\hat\\delta_0$ and $\\hat\\delta_1$.\n"
       "2. Compare $\\hat\\alpha$ with the McCulloch estimate.\n"
       "3. Fit the Normal and the Student-t distributions and compare the AIC ($2k - 2\\ell$) of the three models.\n"
       "4. Draw the QQ plot of the data against the Normal and the stable fits.\n"
       "5. Interpretation: where does each model fail?\n\n"
       "**Report:** the estimates, three AIC values, the QQ plot and two sentences."),
    code("# Solution\n" + src(s.b3_ml) + "\n\n\npd.Series(b3_ml('sp500', save=False)).round(3)"),
    md("**Interpretation of the result.** The Normal fit misses the tails; the stable fit overshoots them; the "
       "Student-t has the lowest AIC."),
    md("## B4 [Proposed]: large Bitcoin losses under three models\n\n"
       "**Question.** Which model reproduces the number of large daily Bitcoin losses best? Model: B3.\n\n"
       "1. Fit the Normal, Student-t and stable (ML) laws.\n"
       "2. Count the days with a loss above 10% and above 20%.\n"
       "3. Compute the expected number of such days under each model, $n \\times P(r < -x)$.\n"
       "4. Compute VaR 1% of the data and of each model.\n"
       "5. Interpretation: which model would you use for a margin requirement on a Bitcoin position, and why?\n\n"
       "**Report:** one table of counts, four VaR values and two sentences."),
    code("# Solution\n" + src(s.b4_tails) + "\n\n\npd.Series(b4_tails('btc')).round(2)"),
    md("## B5 [Solved]: does α stay the same when returns are summed?\n\n"
       "**Question.** If S&P 500 daily returns were i.i.d. stable, weekly and monthly returns would have the same α; do they?\n\n"
       "1. Compute the weekly and monthly log returns as sums of the daily ones.\n"
       "2. Compute the McCulloch $\\hat\\alpha$ at the three horizons.\n"
       "3. Bootstrap each $\\hat\\alpha$ with 500 resamples and report the 95% intervals.\n"
       "4. Interpretation: is the pattern consistent with i.i.d. stable returns?\n\n"
       "**Report:** three estimates with intervals, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_aggregation) + "\n\n\npd.DataFrame(b5_aggregation('sp500', save=False)).T.round(3)"),
    md("**Interpretation of the result.** α̂ is higher for weekly than for daily returns, a warning sign against i.i.d. "
       "stable returns; the monthly sample is too short for a firm answer."),
    md("## B6 [Proposed]: does the sample variance settle?\n\n"
       "**Question.** Is the sample variance of BET and Bitcoin returns as large and as unstable as a stable law with the "
       "fitted α implies? Models: B1, B5.\n\n"
       "1. Compute the sample variance of each series over the full sample and over its first half.\n"
       "2. Simulate 200 paths of the same length from the McCulloch fit (CMS) and compute the sample variance of each path.\n"
       "3. Report the median and the 5% and 95% percentiles of the simulated variances, and the share below the observed one.\n"
       "4. Interpretation: what does the comparison say about the infinite-variance hypothesis?\n\n"
       "**Report:** a table with two rows and one sentence."),
    code("# Solution\n" + src(s.b6_running_variance) + "\n\n\npd.DataFrame(b6_running_variance()).T.round(2)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: is α constant over time?\n\n"
       "**Question.** Do the tails of the BET and the S&P 500 change from one five-year period to the next? Models: B1, B3.\n\n"
       "1. Compute the McCulloch $\\hat\\alpha$ with a bootstrap interval in each period (2000-2004, ..., 2020-2026), for each index.\n"
       "2. Say which periods differ significantly.\n"
       "3. Propose one explanation for the changes and one way to test it.\n"
       "4. Interpretation: what would a changing α mean for a risk model estimated on old data?\n\n"
       "**Report:** a table, a short list of significant changes and a plan for a project."),
    code("# Reference analysis\n" + src(s.c1_periods) + "\n\n\nres = c1_periods()\n"
         "pd.DataFrame({(k, p): v for k, a in res.items() for p, v in a.items()}).T.round(3)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant fitted a stable law to the daily S&P 500 log returns (%) since 2000 and claimed:\n\n"
       "- (a) scipy's `levy_stable` uses S0 by default, so `loc` is the S0 location;\n"
       "- (b) the variance of daily returns is $2\\gamma^2$;\n"
       "- (c) the scale of 20-day returns is $\\sqrt{20}\\,\\gamma$;\n"
       "- (d) since α < 2, the VaR 1% of the model does not exist;\n"
       "- (e) since α > 1, the mean daily return exists.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\npd.Series(c2_check())"),
]

if __name__ == '__main__':
    build(LECTURE, 3, 'lecture')
    build(SEMINAR, 3, 'seminar')
