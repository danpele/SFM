"""
build_notebooks_ch11.py -- lecture and seminar notebooks of Chapter 11 (SFM): fractal markets hypothesis and long memory
=====================================================================================================================
Output: notebooks/EN/chapter11_lecture_notebook.ipynb, notebooks/EN/chapter11_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_11/generate_fmh_charts.py and seminar11.py (inspect.getsource), so the
notebooks stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch11.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter11_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter11_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 11
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(11)
import generate_fmh_charts as g   # noqa: E402
import seminar11 as s             # noqa: E402

CONSTS = ('from scipy import special\n'
          f'NAME = {g.NAME!r}\nSTART = {g.START!r}\nASSETS = {g.ASSETS!r}\nBAD_DAYS = {g.BAD_DAYS!r}\nCOLORS = {g.COLORS!r}\n'
          f'SEED = {g.SEED!r}\nNMIN = {g.NMIN!r}\nWINDOW = {g.WINDOW!r}\nSTEP = {g.STEP!r}\nMC_REPS = {g.MC_REPS!r}\n'
          f'LO_CRIT = {g.LO_CRIT!r}\nGPH_POWER = {g.GPH_POWER!r}\nRS_EXAMPLE = {g.RS_EXAMPLE!r}\nNILE_BREAK = {g.NILE_BREAK!r}\n'
          f'H_LIST = {g.H_LIST!r}\nMC_N = {g.MC_N!r}\nPHIS = {g.PHIS!r}\nHS_POWER = {g.HS_POWER!r}\nDELTAS = {g.DELTAS!r}\nPERS = {g.PERS!r}')
CORE = [g.returns, g.block_sizes, g.rs_block, g.rs_curve, g.hurst_rs, g.dfa_curve, g.hurst_dfa, g.periodogram, g.gph,
        g.lo_test, g.acf, g.all_estimates, g.fgn_acvf, g.arfima_acf, g.arfima_var, g.circulant_gaussian, g.sim_fgn,
        g.sim_arfima, g.sim_garch, g.mc_null, g.rolling_hurst, g.rs_steps]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 11: Fractal markets hypothesis and long memory\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Self-similarity: the scaling of $h$-day returns (S&P 500, BET, Bitcoin).\n"
       "- The R/S statistic step by step; fractional Brownian motion (SFEfbmplot), fractional Gaussian noise (SFEfgnacf) "
       "and ARFIMA(0,d,0) (SFEarfima), simulated exactly by circulant embedding.\n"
       "- Estimators: R/S, Lo's modified R/S, DFA, GPH; Monte Carlo bands; size and power of the R/S tests.\n"
       "- Spurious long memory: the Nile and its 1898 break, mean shifts, GARCH volatility clustering.\n"
       "- Long memory in volatility on six markets; rolling Hurst exponents.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 14; Peters (1994); "
       "Hurst (1951); Lo (1991); Peng et al. (1994); Geweke and Porter-Hudak (1983); Weron (2002)."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; S&P 500, DAX and BET since 2000, Bitcoin since 2014, "
       "Banca Transilvania and OMV Petrom since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Hurst exponent $H$: $H = 0.5$ no memory, $H > 0.5$ persistence, $H < 0.5$ anti-persistence; memory parameter $d = H - 0.5$.\n"
       "- R/S: $(R/S)_n$ = range of the cumulative deviations divided by the standard deviation, averaged over blocks of size $n$; "
       "$\\hat H$ = slope of $\\log (R/S)_n$ on $\\log n$.\n"
       "- DFA: slope of $\\log F(n)$ on $\\log n$, $F(n)$ the root mean square deviation of the profile from local straight lines.\n"
       "- GPH: OLS slope $\\hat d$ of $\\log I(\\lambda_j)$ on $-\\log(4\\sin^2(\\lambda_j/2))$, $j \\le m = [T^{0.5}]$.\n"
       "- Lo's test: $V(q) = R/(\\sqrt{N}\\hat\\sigma(q))$; short memory rejected at 5% outside $[0.809, 1.862]$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Self-similarity: the scaling of $h$-day returns\n\n"
       "- Standard deviation of non-overlapping $h$-day returns against $h$; the square-root-of-time rule has slope 0.5."),
    code(src(g.scaling, g.fig_scaling)),
    code("fig_scaling(save=False)"),
    md("## 2. The R/S statistic step by step\n\n- Eight returns: mean, cumulative deviations, range, standard deviation."),
    code(src(g.fig_rs_example)),
    code("fig_rs_example(save=False)"),
    md("## 3. Fractional Brownian motion and fractional Gaussian noise\n\n"
       "- Exact simulation of fGn by circulant embedding; fBm paths for $H = 0.3, 0.5, 0.7$ (SFEfbmplot).\n"
       "- Sample and theoretical ACF for $H = 0.2$ and $0.8$ (SFEfgnacf); hyperbolic against exponential decay."),
    code(src(g.fig_fbm_paths, g.fig_fgn_acf)),
    code("fig_fbm_paths(save=False)"),
    code("fig_fgn_acf(save=False)"),
    md("## 4. ARFIMA(0,d,0)\n\n- Paths and ACF for $d = 0.4$ and $d = -0.4$ (SFEarfima)."),
    code(src(g.fig_arfima)),
    code("fig_arfima(save=False)"),
    md("## 5. Estimators on the S&P 500: R/S, DFA, GPH and Lo's test\n\n"
       "- Log-log plots for returns and absolute returns, with the mean curve of i.i.d. Normal series."),
    code(src(g.fig_rs_dfa, g.fig_gph)),
    code("fig_rs_dfa(save=False)"),
    code("fig_gph(save=False)"),
    code("r = returns('sp500')\npd.DataFrame({'returns': all_estimates(r.values), '|returns|': all_estimates(np.abs(r.values))}).T"),
    md("## 6. Monte Carlo: small-sample bias, bands, size and power of the R/S tests\n\n"
       "- 500 i.i.d. Normal series for each $N$ (Weron, 2002).\n"
       "- Rejection rates of the classical and modified R/S tests under AR(1) and fractional Gaussian noise."),
    code(src(g.fig_mc_bias, g.lo_rejections)),
    code("mc = fig_mc_bias(save=False)\npd.DataFrame({(N, k): v for N, d in mc.items() for k, v in d.items()}).T.round(3)"),
    code("lo = lo_rejections(save=False)\npd.DataFrame(lo['ar']).T, pd.DataFrame(lo['fgn']).T"),
    md("## 7. Spurious long memory\n\n"
       "- The Nile at Aswan, 1871–1970 (Cobb, 1978): R/S of the raw series and after removing the two regime means.\n"
       "- Simulations: a mean shift in i.i.d. noise; absolute returns of GARCH(1,1), a short-memory model."),
    code(src(g.nile, g.fig_nile, g.spurious)),
    code("fig_nile(save=False)"),
    code("sp = spurious(save=False)\npd.DataFrame(sp['break']).T, pd.DataFrame(sp['garch']).T"),
    md("## 8. Long memory in volatility on six markets\n\n"
       "- ACF of $r_t$, $|r_t|$ and $r_t^2$ (Ding, Granger and Engle, 1993).\n"
       "- All estimators for $r_t$ and $|r_t|$; the DFA exponent of $|r_t|$ after a random shuffle."),
    code(src(g.fig_vol_acf, g.markets, g.fig_markets)),
    code("fig_vol_acf(save=False)"),
    code("M = markets()\nfig_markets(M, save=False)\n"
         "pd.DataFrame({NAME[k]: {'R/S r': v['r']['rs'], 'DFA r': v['r']['dfa'], 'GPH d r': v['r']['gph_d'], 'Lo V r': v['r']['lo_V'], "
         "'DFA |r|': v['abs']['dfa'], 'GPH d |r|': v['abs']['gph_d'], 'DFA band 2.5%': v['mc']['dfa']['q025'], "
         "'DFA band 97.5%': v['mc']['dfa']['q975'], 'DFA |r| shuffled': v['abs_shuffled_dfa']} for k, v in M.items()}).T.round(3)"),
    md("## 9. Rolling Hurst exponents\n\n"
       "- DFA on windows of 1000 observations moved by 21, dated at the window end; the i.i.d. 95% band for $N = 1000$."),
    code(src(g.fig_rolling)),
    code("ro = fig_rolling(save=False)\npd.DataFrame({k: v for k, v in ro.items() if k != 'band'}).T"),
    md("## 10. AI for scientific discovery: does the Hurst exponent change before a crisis?\n\n"
       "- Open question: does the rolling exponent of $|r_t|$ rise before crises, or only after them?\n"
       "- Starter result: rolling DFA of S&P 500 $|r_t|$ on windows of 500 days, around 2008 and 2020.\n"
       "- Check before trusting any answer, yours or an AI's: dating at the window end, $d = H - 0.5$, a band for each $N$, "
       "returns against volatility, the GPH bandwidth."),
    code("ra = rolling_hurst(np.abs(returns('sp500')), window=500)\n"
         "for a, b in [('2007-06-01', '2009-06-30'), ('2019-09-01', '2020-12-31')]:\n"
         "    print(a, b)\n    print(ra.loc[a:b].round(3).to_string())"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.rs_by_hand, s.hurst_from_points, s.rs_points, s.fgn_numbers, s.arfima_numbers]
SEM_CONSTS = (f'SEM_REPS = {s.SEM_REPS!r}\nA1_X = {s.A1_X!r}\nA2_X = {s.A2_X!r}\nPOINTS = {s.POINTS!r}\nCRISES = {s.CRISES!r}')

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 11: Fractal markets hypothesis and long memory\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 11: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Returns; R/S, DFA, GPH and Lo's test; the ACF; Monte Carlo bands under i.i.d. Normal series; rolling exponents; "
       "simulation of fractional Gaussian noise and ARFIMA.\n"
       "- Helpers for Part A: R/S step by step, $H$ from points of an R/S plot, fGn and ARFIMA numbers."),
    code(CONSTS + '\n' + SEM_CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: R/S step by step\n\n"
       "**Context.** Eight daily returns (%): 0.5, −0.3, 0.8, −0.2, 0.6, −0.1, 0.4, −0.7.\n\n"
       "1. Compute the mean, the deviations and the cumulative deviations $Y_1, \\dots, Y_8$.\n"
       "2. Compute $R$, $S$ (divisor $n$) and $R/S$.\n"
       "3. S&P 500 returns since 2000 give $(R/S)_n$ at $n = 16, 64, 256$ (computed below); compute $\\hat H$.\n\n"
       "**Report:** $R$, $S$, $R/S$, $\\hat H$ and one sentence."),
    code("# Solution\nprint(rs_by_hand(A1_X))\nprint(hurst_from_points(POINTS, rs_points('sp500')))"),
    md("**Interpretation of the result.** Alternating signs keep the cumulative deviations small; an $\\hat H$ slightly above 0.5 "
       "is the small-sample bias of R/S, not memory (B1)."),
    md("## A2 [Proposed]: grouped signs\n\n"
       "**Context.** Eight returns (%): 0.6, 0.4, 0.5, 0.3, −0.4, −0.6, −0.2, −0.5. Model: A1.\n\n"
       "1. Compute the mean and $Y_1, \\dots, Y_8$.\n"
       "2. Compute $R$, $S$ and $R/S$, and compare with A1.\n"
       "3. BET returns give $(R/S)_n$ at $n = 16, 64, 256$; compute $\\hat H$.\n\n"
       "**Report:** $R/S$, $\\hat H$ and two sentences."),
    code("# Solution\nprint(rs_by_hand(A2_X))\nprint(hurst_from_points(POINTS, rs_points('bet')))"),
    md("## A3 [Solved]: fractional Gaussian noise and 10-day risk\n\n"
       "**Context.** Daily returns follow an fGn with $H = 0.7$ and daily volatility $\\sigma = 1.2\\%$; $z_{0.99} = 2.326$.\n\n"
       "1. Compute $\\rho(1)$ and $\\rho(10)$ exactly and $\\rho(10)$ with the approximation $H(2H - 1)k^{2H - 2}$.\n"
       "2. Compute the scaling factor $10^H$ and compare it with $\\sqrt{10}$.\n"
       "3. Compute the one-day and the 10-day VaR 1% (Normal) with $h^H$ and with $\\sqrt{h}$.\n"
       "4. Compute the fractal dimension of the price path.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nfgn_numbers(0.7)"),
    md("**Interpretation of the result.** Persistent returns make multi-day losses add up: the $\\sqrt{h}$ rule understates "
       "the 10-day risk by more than a third."),
    md("## A4 [Proposed]: anti-persistence and a small deviation from 0.5\n\n"
       "**Context.** The setting of A3 with $H = 0.4$ and with $H = 0.55$. Model: A3.\n\n"
       "1. Compute $\\rho(1)$ and $\\rho(10)$ for both values of $H$.\n"
       "2. Compute the 10-day VaR 1% for both and its difference from the $\\sqrt{h}$ rule, in %.\n"
       "3. Compute $D$ for both.\n\n"
       "**Report:** eight numbers and one sentence."),
    code("# Solution\nprint(fgn_numbers(0.4))\nprint(fgn_numbers(0.55))"),
    md("## A5 [Solved]: ARFIMA(0, 0.3, 0) against an AR(1)\n\n"
       "**Context.** $(1 - L)^{0.3}X_t = \\varepsilon_t$, $\\varepsilon_t$ i.i.d. $N(0, 1)$.\n\n"
       "1. Compute $\\pi_1$, $\\pi_2$ and the MA weights $\\psi_1$, $\\psi_2$.\n"
       "2. Compute $\\rho(1)$, $\\rho(2)$, $\\rho(10)$ and $\\rho(50)$.\n"
       "3. For an AR(1) with $\\phi = \\rho(1)$, compute $\\rho(10)$ and $\\rho(50)$.\n"
       "4. Give $H$ and the first lag with $\\rho(k) < 0.05$ for both models.\n\n"
       "**Report:** ten numbers and one sentence."),
    code("# Solution\narfima_numbers(0.3)"),
    md("**Interpretation of the result.** The same $\\rho(1)$, very different memory: the AR(1) forgets in days, the ARFIMA in months."),
    md("## A6 [Proposed]: strong memory and anti-persistence\n\n"
       "**Context.** ARFIMA(0,d,0) with $d = 0.45$ and with $d = -0.2$. Model: A5.\n\n"
       "1. Compute $\\pi_1$, $\\pi_2$, $\\rho(1)$, $\\rho(2)$ and $\\rho(10)$ for both.\n"
       "2. For $d = 0.45$, compare $\\rho(10)$ and $\\rho(50)$ with an AR(1) with $\\phi = \\rho(1)$.\n"
       "3. Give $H$ for both and say which one is stationary.\n\n"
       "**Report:** a small table and one sentence."),
    code("# Solution\nprint(arfima_numbers(0.45))\nprint(arfima_numbers(-0.2))"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: long memory in S&P 500 returns?\n\n"
       "**Question.** Do daily S&P 500 returns have long memory?\n\n"
       "1. Estimate $H$ with R/S and DFA (block sizes from 10 to $N/4$) and $d$ with GPH ($m = [T^{0.5}]$), with the standard error of $\\hat d$.\n"
       "2. Compute Lo's modified statistic $V(q)$ with Andrews' $q$, and the classical statistic $V(0)$.\n"
       "3. Simulate 300 i.i.d. Normal series of the same length and give the 95% band of each estimator.\n"
       "4. Draw the R/S plot inside the Monte Carlo envelope.\n"
       "5. Interpretation: is the R/S estimate above 0.5 evidence of long memory?\n\n"
       "**Report:** a table of four estimates with their bands, the chart and two sentences."),
    code("# Solution\n" + src(s.b1_memory) + "\n\n\nb1 = b1_memory('sp500', save=False)\n"
         "print(pd.DataFrame({'estimate': {k: b1['est'][k] for k in ['rs', 'dfa', 'gph_H']}, "
         "'2.5%': {k: b1['mc'][m]['q025'] for k, m in [('rs', 'rs'), ('dfa', 'dfa'), ('gph_H', 'gph')]}, "
         "'97.5%': {k: b1['mc'][m]['q975'] for k, m in [('rs', 'rs'), ('dfa', 'dfa'), ('gph_H', 'gph')]}}).round(3))\n"
         "{k: b1['est'][k] for k in ['gph_d', 'gph_se', 'lo_V', 'lo_q', 'lo_V0']}"),
    md("**Interpretation of the result.** No: i.i.d. series of this length give an R/S slope above 0.5 on average; every "
       "estimate lies inside its band and Lo's test does not reject short memory."),
    md("## B2 [Proposed]: BET and Banca Transilvania\n\n"
       "**Question.** Does the BET or Banca Transilvania show long memory in returns? Model: B1.\n\n"
       "1. Estimate $H$ with R/S, DFA and GPH for each series.\n"
       "2. Compute Lo's $V(q)$ and say whether short memory is rejected at 5%.\n"
       "3. Compute the Monte Carlo band of each estimator for each length.\n"
       "4. Interpretation: what could explain the result for the BET?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = {k: b1_memory(k, save=False) for k in ('bet', 'tlv')}\n"
         "pd.DataFrame({NAME[k]: {**{c: d['est'][c] for c in ['n', 'rs', 'dfa', 'gph_d', 'gph_se', 'lo_V', 'lo_q']}, "
         "'R/S band': (round(d['mc']['rs']['q025'], 3), round(d['mc']['rs']['q975'], 3)), "
         "'DFA band': (round(d['mc']['dfa']['q025'], 3), round(d['mc']['dfa']['q975'], 3))} for k, d in b2.items()}).T"),
    md("## B3 [Solved]: long memory in S&P 500 volatility\n\n"
       "**Question.** Do the absolute and the squared S&P 500 returns have long memory, and is it in the order of the days?\n\n"
       "1. Estimate $d$ with GPH for $m = [T^{0.5}]$ and $m = [T^{0.6}]$, and $H$ with DFA, for $|r_t|$ and $r_t^2$.\n"
       "2. Report the ACF of $|r_t|$ at lags 1, 50 and 250 and the 95% band $\\pm 1.96/\\sqrt{T}$.\n"
       "3. Shuffle $|r_t|$ at random and repeat the estimates.\n"
       "4. Draw the ACF of $|r_t|$ before and after the shuffle.\n"
       "5. Interpretation: does long memory in $|r_t|$ contradict the weak form of the EMH?\n\n"
       "**Report:** a table, the chart and two sentences."),
    code("# Solution\n" + src(s.b3_volatility) + "\n\n\nb3 = b3_volatility('sp500', save=False)\n"
         "pd.DataFrame({k: b3[k] for k in ['abs', 'sq', 'abs_shuffled']}).T.round(3)"),
    md("**Interpretation of the result.** No: the weak form concerns the direction of returns (B1); long memory in $|r_t|$ "
       "means predictable risk, which needs a volatility model (Chapter 9)."),
    md("## B4 [Proposed]: Bitcoin returns and volatility\n\n"
       "**Question.** Does Bitcoin show long memory in returns, in volatility, or in both? Models: B1, B3.\n\n"
       "1. Estimate R/S, DFA, GPH and Lo's $V(q)$ for the returns, with the Monte Carlo bands.\n"
       "2. Estimate GPH ($T^{0.5}$, $T^{0.6}$) and DFA for $|r_t|$ and $r_t^2$.\n"
       "3. Repeat the volatility estimates after a random shuffle of $|r_t|$.\n"
       "4. Interpretation: is Bitcoin closer to the S&P 500 or to the BET?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb4r = b1_memory('btc', save=False)\nprint({k: round(v, 3) for k, v in b4r['est'].items() if isinstance(v, float)})\n"
         "print({m: (round(b4r['mc'][m]['q025'], 3), round(b4r['mc'][m]['q975'], 3)) for m in b4r['mc']})\n"
         "b4 = b3_volatility('btc', save=False)\npd.DataFrame({k: b4[k] for k in ['abs', 'sq', 'abs_shuffled']}).T.round(3)"),
    md("## B5 [Solved]: rolling Hurst exponents of the S&P 500\n\n"
       "**Question.** Has the memory of S&P 500 returns changed since 2000?\n\n"
       "1. Estimate the DFA exponent on windows of 1000 days moved by 21 days, dated at the last day of each window.\n"
       "2. Compute the 95% Monte Carlo band for $N = 1000$.\n"
       "3. Report the minimum and the maximum with their dates, and the share of windows below and above the band.\n"
       "4. Draw the rolling exponent with the band.\n"
       "5. Interpretation: does a window outside the band prove that the market was inefficient at that time?\n\n"
       "**Report:** six numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_rolling) + "\n\n\nb5 = b5_rolling('sp500', save=False)\nb5"),
    md("**Interpretation of the result.** No: about 5% of the windows fall outside by chance and consecutive windows overlap; "
       "only long runs count. Here the runs are below the band (anti-persistence after 2008 and in 2016), never above."),
    md("## B6 [Proposed]: rolling exponents of the DAX and of Banca Transilvania\n\n"
       "**Question.** Does the result of B5 hold for the DAX and for Banca Transilvania? Model: B5.\n\n"
       "1. Estimate the rolling DFA exponent as in B5.\n"
       "2. Report the share of windows below and above the band, and the minimum and maximum with their dates.\n"
       "3. Compare the mean exponent in the first and in the last third of the windows.\n"
       "4. Interpretation: which market looks more stable over time?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6 = {k: b5_rolling(k, save=False) for k in ('dax', 'tlv')}\npd.DataFrame(b6).T"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: does market memory change in crises?\n\n"
       "**Question.** Does the Hurst exponent of returns or of volatility change in a crisis, as the fractal market hypothesis "
       "suggests? Models: B1, B3.\n\n"
       "1. Estimate the DFA exponent of $r_t$ and of $|r_t|$ and the annualised volatility in 2017, in September 2008–August 2009 "
       "and in February 2020–February 2021, for the S&P 500 and the BET.\n"
       "2. Compute the Monte Carlo band for windows of about 250 days.\n"
       "3. List three reasons why one crisis year cannot settle the question.\n"
       "4. Interpretation: which design would separate a change of memory from a change of volatility?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\n" + src(s.c1_crisis) + "\n\n\nc1 = c1_crisis()\nc1"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant interpreted long-memory estimates for the S&P 500 (daily, since 2000). It answered:\n\n"
       "- (a) R/S gives $H$ above 0.5 for the returns, so returns have long memory and trends can be exploited;\n"
       "- (b) $H = 0.53$ means a 53% probability that tomorrow has the same sign as today;\n"
       "- (c) GPH gives $d = 0.47$ for $|r|$, so $H = d - 0.5$: absolute returns are anti-persistent;\n"
       "- (d) long memory in $|r|$ contradicts the weak form of the EMH;\n"
       "- (e) Lo's modified statistic lies inside $[0.809, 1.862]$, so short memory is not rejected at 5%;\n"
       "- (f) for more precision, use all $T/2$ frequencies in the GPH regression.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\nc2 = c2_check()\nc2"),
]

if __name__ == '__main__':
    build(LECTURE, 11, 'lecture')
    build(SEMINAR, 11, 'seminar')
