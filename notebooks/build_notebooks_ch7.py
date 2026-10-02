"""
build_notebooks_ch7.py -- lecture and seminar notebooks of Chapter 7 (SFM): efficient markets, random walk, unit roots
and variance-ratio tests
====================================================================================================================
Output: notebooks/EN/chapter7_lecture_notebook.ipynb, notebooks/EN/chapter7_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_07/generate_all_charts.py and seminar7.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch7.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter7_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter7_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 7
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(7)
import generate_all_charts as g   # noqa: E402
import seminar7 as s              # noqa: E402

CONSTS = ('import statsmodels.api as sm\nfrom statsmodels.tsa.stattools import adfuller, kpss\n'
          'from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp\n'
          f'EXTRA_SERIES = {g.EXTRA_SERIES!r}\nNAME = {g.NAME!r}\nSTART = {g.START!r}\nASSETS = {g.ASSETS!r}\n'
          f'STOCKS = {g.STOCKS!r}\nDEVELOPED = {g.DEVELOPED!r}\nEMERGING = {g.EMERGING!r}\nCRYPTO = {g.CRYPTO!r}\n'
          f'MARKETS_START = {g.MARKETS_START!r}\nBAD_DAYS = {g.BAD_DAYS!r}\nCOLORS = {g.COLORS!r}\nQS = {g.QS!r}\n'
          f'LB_LAGS = {g.LB_LAGS!r}\nWINDOW = {g.WINDOW!r}\nSTEP = {g.STEP!r}\nSEED = {g.SEED!r}\nB_BOOT = {g.B_BOOT!r}\n'
          f'FTSE_UPGRADE = {g.FTSE_UPGRADE!r}\nSUBPERIODS = {g.SUBPERIODS!r}\nDAYS = {g.DAYS!r}\nACF_CASES = {g.ACF_CASES!r}')
CORE = [g.returns, g.log_price, g.acf, g.acf_fft, g.robust_var_acf, g.portmanteau, g.runs_test, g.adf_test, g.pp_test,
        g.kpss_test, g.variance_ratio, g.chow_denning, g.qs_kernel, g.auto_vr_stat, g.auto_vr, g.vr_profile]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 7: Efficient markets, random walk and variance-ratio tests\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Can past prices predict returns? Today's return against yesterday's; random walks against the S&P 500.\n"
       "- White noise and the autocorrelation function (SFEtimewn, SFEacfar1, SFEacfar2, SFEacfma1, SFEacfma2).\n"
       "- Box–Pierce, Ljung–Box and robust portmanteau tests; the runs test.\n"
       "- Unit roots: ADF, Phillips–Perron and KPSS on log prices and returns; spurious regression.\n"
       "- Variance-ratio tests: Lo–MacKinlay Z and Z*, Chow–Denning, the automatic VR test with wild bootstrap.\n"
       "- Efficiency across markets and over time (adaptive markets); calendar anomalies.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 11; "
       "Campbell, Lo and MacKinlay (1997), *The Econometrics of Financial Markets*, Ch. 2; Lo and MacKinlay (1988)."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; S&P 500 and DAX since 1990, BET since 1997, Bitcoin since "
       "2014, BVB stocks since 2010 (Banca Transilvania without 30–31 May 2016, a data error).\n"
       "- Sample ACF $\\hat\\rho(k)$; i.i.d. band $\\pm 1.96/\\sqrt{T}$; robust band $\\pm 1.96\\sqrt{\\hat\\delta(k)/T}$ with "
       "$\\hat\\delta(k) = T\\sum e_t^2e_{t-k}^2/(\\sum e_t^2)^2$.\n"
       "- Variance ratio $\\mathrm{VR}(q) = \\mathrm{Var}(r_t(q))/(q\\,\\mathrm{Var}(r_t))$; $Z(q)$ assumes i.i.d. returns, "
       "$Z^*(q)$ allows for volatility clustering.\n"
       "- Unit-root tests: ADF and PP ($H_0$: unit root), KPSS ($H_0$: stationarity)."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Can prices be predicted?\n\n"
       "- Scatter of $r_t$ against $r_{t-1}$: the slope is the first autocorrelation.\n"
       "- Four random walks with i.i.d. Normal steps of the same mean and standard deviation as the S&P 500."),
    code(src(g.fig_scatter_lag, g.fig_rw_paths)),
    code("fig_scatter_lag(save=False)"),
    code("fig_rw_paths(save=False)"),
    md("## 2. Random walk: how risk grows with the horizon\n\n"
       "- Under a random walk, $\\mathrm{sd}(r_t(q)) = \\sigma\\sqrt{q}$ (the square-root-of-time rule)."),
    code("r = returns('sp500')\nsigma = r.std()\n"
         "pd.Series({f'{q} days': sigma * np.sqrt(q) for q in (1, 5, 20, 250)}).round(2)"),
    md("## 3. White noise and the ACF\n\n"
       "- Gaussian white noise and its sample ACF (SFEtimewn).\n"
       "- Theoretical and sample ACF of AR(1), AR(2), MA(1) and MA(2) processes (SFEacfar1, SFEacfar2, SFEacfma1, SFEacfma2).\n"
       "- The ACF of daily returns with the i.i.d. and the robust bands; the ACF of absolute returns."),
    code(src(g.fig_white_noise, g.arma_acf, g.simulate_arma, g.fig_acf_processes, g.fig_acf_returns, g.fig_acf_abs)),
    code("fig_white_noise(save=False)"),
    code("fig_acf_processes(save=False)"),
    code("fig_acf_returns(save=False)"),
    code("fig_acf_abs(save=False)"),
    md("## 4. Autocorrelation tests\n\n"
       "- $Q_{BP}(10)$, $Q_{LB}(10)$ and the robust $\\tilde Q(10)$ on returns and squared returns; the runs test."),
    code(src(g.tests_table)),
    code("tt = tests_table()\n"
         "pd.DataFrame({NAME[k]: {'rho1': v['ret']['rho1'], 'LB(10)': v['ret']['lb'], 'p LB': v['ret']['p_lb'], "
         "'robust Q(10)': v['ret']['rob'], 'p robust': v['ret']['p_rob'], 'LB(10) squared': v['sq']['lb'], "
         "'runs z': v['runs']['z'], 'p runs': v['runs']['p']} for k, v in tt.items()}).T.round(3)"),
    md("## 5. Unit roots: prices against returns\n\n"
       "- ADF (lag order by AIC), Phillips–Perron and KPSS on log prices (constant and trend) and on returns (constant).\n"
       "- The spurious regression of two independent random walks (Granger and Newbold, 1974)."),
    code(src(g.unit_root_table, g.fig_unit_root, g.spurious, g.fig_spurious)),
    code("ur = unit_root_table()\n"
         "pd.DataFrame({NAME[k]: {f'{s_} {t}': v[s_][t]['stat'] for s_ in ['price', 'ret'] for t in ['adf', 'pp', 'kpss']} "
         "for k, v in ur.items()}).T.round(2)"),
    code("fig_unit_root('bet', save=False)"),
    code("sp = spurious()\nprint({lab: (sp[lab]['reject'], round(sp[lab]['r2_med'], 3)) for lab in ['level', 'diff']})\nfig_spurious(sp, save=False)"),
    md("## 6. Variance-ratio tests\n\n"
       "- Lo–MacKinlay VR(q) with $Z(q)$ and $Z^*(q)$; the Chow–Denning test; Choi's automatic VR test with the wild "
       "bootstrap of Kim (2009).\n"
       "- A Monte Carlo study: how often do $Z$ and $Z^*$ reject a true null under GARCH returns?"),
    code(src(g.vr_table, g.fig_vr_profile, g.simulate_garch, g.vr_size, g.fig_vr_size)),
    code("vt = vr_table()\n"
         "pd.DataFrame({NAME[k]: {**{f'VR({q})': v['vr'][q]['vr'] for q in QS}, **{f'Z*({q})': v['vr'][q]['zs'] for q in QS}, "
         "'CD': v['cd']['cd'], 'p CD': v['cd']['p'], 'AVR': v['avr']['stat'], 'k': v['avr']['k'], 'p AVR': v['avr']['p_boot']} "
         "for k, v in vt.items()}).T.round(3)"),
    code("fig_vr_profile(save=False)"),
    code("sz = vr_size()\nfig_vr_size(sz, save=False)\npd.DataFrame({(lab, q): r for lab, d in sz.items() for q, r in d.items()}).T"),
    md("## 7. Efficiency across markets and over time\n\n"
       "- VR(5) over the last ten years for developed, emerging and crypto markets.\n"
       "- Rolling VR(5) on 500-day windows (the adaptive markets hypothesis); three sub-periods per market."),
    code(src(g.markets_table, g.fig_vr_markets, g.rolling_vr, g.fig_rolling_vr, g.subperiods)),
    code("mt = markets_table()\nfig_vr_markets(mt, save=False)\npd.DataFrame(mt).T.round(3)"),
    code("fig_rolling_vr(save=False)"),
    code("sub = subperiods()\npd.DataFrame([dict(market=k, **r) for k, rows in sub.items() for r in rows]).round(3)"),
    md("## 8. Calendar anomalies\n\n"
       "- Mean daily return by weekday and January against the other months, with Newey–West (HAC) standard errors."),
    code(src(g.calendar, g.fig_calendar)),
    code("cal = fig_calendar(save=False)\n{k: {c: v[c] for c in ['wald_p', 'jan_mean', 'other_mean', 'jan_t', 'jan_p']} for k, v in cal.items()}"),
    md("## 9. AI for scientific discovery: has the BET become more efficient?\n\n"
       "- Open question: did the BET become more efficient after Romania's upgrade to Secondary Emerging market "
       "(FTSE Russell, September 2020)?\n"
       "- Starter result: VR(5) and $Z^*(5)$ in the five years before and after 21 September 2020.\n"
       "- Check before trusting any answer, yours or an AI's: robust $Z^*$, not $Z$; overlapping windows are not independent "
       "tests; the event date is fixed in advance; other changes happened at the same time."),
    code(src(g.bet_upgrade)),
    code("up = bet_upgrade()\nprint('z of the difference:', round(up['z_diff'], 2))\npd.DataFrame({p: up[p] for p in ['before', 'after']}).T"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.vr_from_acf, s.rw_horizon, s.portmanteau_from_acf, s.runs_from_signs, s.adf_decision]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 7: Efficient markets, random walk and variance-ratio tests\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 7: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Returns and log prices; sample ACF, robust variance of the ACF, Box–Pierce, Ljung–Box and robust portmanteau "
       "statistics; the runs test; ADF, Phillips–Perron and KPSS; Lo–MacKinlay VR with $Z$ and $Z^*$, Chow–Denning, "
       "the automatic VR test with wild bootstrap.\n"
       "- Helpers for Part A: VR from autocorrelations, the square-root-of-time rule, portmanteau statistics from given "
       "autocorrelations, the runs test from a string of signs, Dickey–Fuller and KPSS decisions."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: variance ratios from autocorrelations\n\n"
       "**Context.** Daily returns have a standard deviation $\\sigma = 1.2\\%$.\n\n"
       "1. Under RW1, compute the standard deviation of the 5-day and of the 20-day return.\n"
       "2. Suppose $\\rho(1) = 0.10$, $\\rho(2) = 0.05$ and $\\rho(k) = 0$ for $k \\ge 3$; compute VR(2) and VR(5).\n"
       "3. Treat these as estimates from $T = 2500$ i.i.d. returns and compute $Z(2)$ and $Z(5)$.\n"
       "4. Compute the 5-day standard deviation implied by VR(5).\n\n"
       "**Report:** two standard deviations, two VR, two $Z$ and one sentence."),
    code("# Solution\nprint(rw_horizon(1.2, rho=[0.10, 0.05]))\nprint(vr_from_acf([0.10, 0.05], 2, 2500))\nprint(vr_from_acf([0.10, 0.05], 5, 2500))"),
    md("**Interpretation of the result.** Positive autocorrelation (momentum) makes the weekly risk larger than the "
       "square-root-of-time rule says."),
    md("## A2 [Proposed]: mean reversion in a volatile asset\n\n"
       "**Context.** $\\sigma = 2\\%$, $\\rho(1) = -0.08$, $\\rho(2) = 0.03$, $\\rho(k) = 0$ for $k \\ge 3$, $T = 1000$. Model: A1.\n\n"
       "1. Compute VR(2) and VR(5).\n"
       "2. Compute $Z(2)$ and $Z(5)$ and decide at 5%.\n"
       "3. Compare the 5-day standard deviation implied by VR(5) with $2\\sqrt5$.\n\n"
       "**Report:** two VR, two $Z$, two standard deviations and one sentence."),
    code("# Solution\nprint(vr_from_acf([-0.08, 0.03], 2, 1000))\nprint(vr_from_acf([-0.08, 0.03], 5, 1000))\nprint(rw_horizon(2.0, rho=[-0.08, 0.03]))"),
    md("## A3 [Solved]: Box–Pierce and Ljung–Box, step by step\n\n"
       "**Context.** $T = 500$ daily returns; $\\hat\\rho(1), \\dots, \\hat\\rho(5) = 0.12, -0.05, 0.04, 0.03, -0.06$.\n\n"
       "1. Compute $Q_{BP}(5)$.\n"
       "2. Compute $Q_{LB}(5)$.\n"
       "3. Compare with $\\chi^2_{0.95}(5) = 11.07$ and decide.\n"
       "4. Say which lag drives the result.\n\n"
       "**Report:** two statistics, a decision and one sentence."),
    code("# Solution\npd.Series(portmanteau_from_acf([0.12, -0.05, 0.04, 0.03, -0.06], 500))"),
    md("**Interpretation of the result.** $Q_{LB}(5)$ is just above the critical value; lag 1 alone gives most of it."),
    md("## A4 [Proposed]: Ljung–Box on squared returns\n\n"
       "**Context.** $T = 750$; the ACF of the squared returns: 0.25, 0.20, 0.18, 0.15, 0.12 at lags 1–5. Model: A3.\n\n"
       "1. Compute $Q_{LB}(5)$ and decide at 5%.\n"
       "2. Say which of RW1, RW3 and the martingale this rejects.\n"
       "3. Say whether weak-form efficiency is rejected.\n\n"
       "**Report:** one statistic and two sentences."),
    code("# Solution\npd.Series(portmanteau_from_acf([0.25, 0.20, 0.18, 0.15, 0.12], 750))"),
    md("## A5 [Solved]: the runs test on 20 days\n\n"
       "**Context.** Signs of 20 daily returns: `++-+++--+---++-+--++`.\n\n"
       "1. Count $n_1$, $n_2$ and the number of runs $R$.\n"
       "2. Compute $E[R]$ and $\\mathrm{Var}(R)$.\n"
       "3. Compute $z$ and decide at 5%.\n\n"
       "**Report:** $R$, $E[R]$, $z$ and one sentence."),
    code("# Solution\npd.Series(runs_from_signs('++-+++--+---++-+--++'))"),
    md("**Interpretation of the result.** $R$ is almost exactly its expected value: no evidence against independence; with "
       "20 days the test has little power."),
    md("## A6 [Proposed]: a sequence with long runs\n\n"
       "**Context.** Signs of 24 daily returns: `+++---+++---++++----++--`. Model: A5.\n\n"
       "1. Count $n_1$, $n_2$ and $R$.\n"
       "2. Compute $E[R]$, $\\mathrm{Var}(R)$ and $z$.\n"
       "3. Decide at 5% and say what kind of dependence the signs show.\n\n"
       "**Report:** $R$, $E[R]$, $z$ and one sentence."),
    code("# Solution\npd.Series(runs_from_signs('+++---+++---++++----++--'))"),
    md("## A7 [Solved]: ADF and KPSS from regression output\n\n"
       "**Context.** Log price, $T = 2500$: $\\Delta p_t = 0.021 - 0.0028\\,p_{t-1} + 0.05\\,\\Delta p_{t-1} + e_t$, "
       "$\\mathrm{SE}(\\hat\\gamma) = 0.0019$, KPSS (constant) $= 2.10$. Returns: $\\Delta r_t = 0.03 - 0.92\\,r_{t-1} + e_t$, "
       "$\\mathrm{SE}(\\hat\\gamma) = 0.020$, KPSS (constant) $= 0.09$.\n\n"
       "1. Compute $\\tau$ for the price and for the returns.\n"
       "2. Decide with the 5% critical value for a constant.\n"
       "3. Decide with KPSS and give the order of integration of each series.\n\n"
       "**Report:** two $\\tau$, four decisions and one sentence."),
    code("# Solution\nprint('price  :', adf_decision(-0.0028, 0.0019, 'c', 2.10))\nprint('returns:', adf_decision(-0.92, 0.020, 'c', 0.09))"),
    md("**Interpretation of the result.** The price is $I(1)$ and the returns are $I(0)$: the picture of a random walk, "
       "but not yet a proof of it."),
    md("## A8 [Proposed]: when the tests disagree\n\n"
       "**Context.** Log price of a stock, $T = 1500$, constant and trend: $\\hat\\gamma = -0.0105$, "
       "$\\mathrm{SE}(\\hat\\gamma) = 0.0029$; KPSS (trend) $= 0.21$. Model: A7.\n\n"
       "1. Compute $\\tau$ and decide with the 5% critical value for a constant and a trend.\n"
       "2. Decide with KPSS.\n"
       "3. Combine the two decisions and give two possible reasons for the conflict.\n\n"
       "**Report:** $\\tau$, two decisions and two sentences."),
    code("# Solution\nadf_decision(-0.0105, 0.0029, 'ct', 0.21)"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: autocorrelation in the S&P 500\n\n"
       "**Question.** Does the verdict on the autocorrelation of daily S&P 500 returns depend on whether we allow for "
       "volatility clustering?\n\n"
       "1. Draw the ACF of the returns (lags 1–20) with the i.i.d. and the robust 95% bands, and the ACF of the squared returns.\n"
       "2. Report $\\hat\\rho(1), \\dots, \\hat\\rho(5)$ of the returns and of the squared returns.\n"
       "3. Compute $Q_{LB}(10)$ and the robust $\\tilde Q(10)$, with p-values, for the returns and for the squared returns.\n"
       "4. Say which random walk hypothesis each test rejects.\n"
       "5. Interpretation: is weak-form efficiency rejected?\n\n"
       "**Report:** the chart, ten autocorrelations, four statistics with p-values and two sentences."),
    code("# Solution\n" + src(s.b1_acf) + "\n\n\nb1 = b1_acf('sp500', save=False)\n"
         "print('rho returns:', np.round(b1['rho'], 3), '| rho squared:', np.round(b1['rho_sq'], 3))\n"
         "pd.DataFrame({lab: {c: b1[lab][c] for c in ['lb', 'p_lb', 'rob', 'p_rob']} for lab in ['ret', 'sq']}).T.round(4)"),
    md("**Interpretation of the result.** Squared returns reject RW1 (volatility clustering); the robust test rejects RW3 "
       "only barely, because of a small negative first autocorrelation."),
    md("## B2 [Proposed]: BET, Bitcoin and two BVB stocks\n\n"
       "**Question.** For which of the BET, Bitcoin, Banca Transilvania and OMV Petrom does the verdict on autocorrelation "
       "change between the classic and the robust test? Model: B1.\n\n"
       "1. Compute $\\hat\\rho(1)$, $Q_{LB}(10)$ and $\\tilde Q(10)$ with p-values for each series.\n"
       "2. Compute the runs test $z$ and its p-value for each series.\n"
       "3. Mark the series for which the classic and the robust test disagree at 5%.\n"
       "4. Interpretation: why is the BET different from the other three?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\n" + src(s.b2_table) + "\n\n\nb2 = b2_table()\n"
         "pd.DataFrame({NAME[k]: {'n': d['n'], 'rho1': d['port']['rho1'], 'LB(10)': d['port']['lb'], 'p LB': d['port']['p_lb'], "
         "'robust Q(10)': d['port']['rob'], 'p robust': d['port']['p_rob'], 'runs z': d['runs']['z'], 'p runs': d['runs']['p']} "
         "for k, d in b2.items()}).T.round(3)"),
    md("## B3 [Solved]: unit roots in the DAX\n\n"
       "**Question.** What is the order of integration of the DAX log price and of its returns?\n\n"
       "1. Draw the log price with a fitted linear trend, and the returns.\n"
       "2. Run ADF (lag order by AIC), PP and KPSS on the log price with a constant and a trend.\n"
       "3. Run the same three tests on the returns with a constant.\n"
       "4. Interpretation: does a unit root in the DAX price prove that the DAX is weak-form efficient?\n\n"
       "**Report:** the chart, six statistics with p-values and two sentences."),
    code("# Solution\n" + src(s.b3_unit_root) + "\n\n\nb3 = b3_unit_root('dax', save=False)\n"
         "pd.DataFrame({(s_, t): b3[s_][t] for s_ in ['price', 'ret'] for t in ['adf', 'pp', 'kpss']}).T.round(3)"),
    md("**Interpretation of the result.** The log price is $I(1)$ and the returns are $I(0)$; a unit root is necessary for "
       "a random walk but does not prove efficiency: that is tested on the returns (B1, B5)."),
    md("## B4 [Proposed]: BET, Bitcoin and TLV\n\n"
       "**Question.** Do the three unit-root tests agree for the BET, Bitcoin and Banca Transilvania? Model: B3.\n\n"
       "1. Run ADF, PP and KPSS on each log price (constant and trend) and on each return series (constant).\n"
       "2. Classify each series as $I(1)$, $I(0)$ or inconclusive.\n"
       "3. For an inconclusive case, propose one check that could settle it.\n"
       "4. Interpretation: would you trade a \"mean-reverting\" TLV price on this evidence?\n\n"
       "**Report:** one table, three classifications and two sentences."),
    code("# Solution\n" + src(s.b4_unit_roots) + "\n\n\nb4 = b4_unit_roots()\n"
         "pd.DataFrame({NAME[k]: {f'{s_} {t} ({c})': d[s_][t][c] for s_ in ['price', 'ret'] for t in ['adf', 'pp', 'kpss'] for c in ['stat', 'p']} "
         "for k, d in b4.items()}).T.round(3)"),
    md("## B5 [Solved]: variance ratios of the S&P 500\n\n"
       "**Question.** Is the S&P 500 mean-reverting over horizons of 2 to 20 days, once volatility clustering is taken into "
       "account?\n\n"
       "1. Compute VR($q$), $Z(q)$ and $Z^*(q)$ for $q = 2, 5, 10, 20$.\n"
       "2. Compute the Chow–Denning statistic and compare it with its 5% critical value.\n"
       "3. Draw VR($q$) for $q = 2, \\dots, 40$ with the i.i.d. and the robust 95% bands.\n"
       "4. Interpretation: is the mean reversion large enough to matter for an investor?\n\n"
       "**Report:** a table of twelve numbers, one statistic, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_vr) + "\n\n\nb5 = b5_vr('sp500', save=False)\nprint(b5['cd'])\n"
         "pd.DataFrame(b5['vr']).loc[['vr', 'z', 'zs', 'p_zs']].round(3)"),
    md("**Interpretation of the result.** Significant at every horizon with the robust test, but small: VR(5) is only "
       "modestly below 1, and $Z$ exaggerates the evidence compared with $Z^*$."),
    md("## B6 [Proposed]: five emerging markets\n\n"
       "**Question.** Which of the WIG20, BUX, PX, BET and BIST 100 departs from a random walk over the last ten years? Model: B5.\n\n"
       "1. Compute VR(5), $Z^*(5)$ and its p-value for each market.\n"
       "2. Compute the Chow–Denning p-value ($q = 2, 5, 10, 20$) and the automatic VR test with its wild-bootstrap p-value.\n"
       "3. Compute the Šidák critical value for five markets tested together at 5%.\n"
       "4. Interpretation: which markets would you call inefficient once you account for testing five markets?\n\n"
       "**Report:** one table, one critical value and two sentences."),
    code("# Solution\n" + src(s.b6_emerging) + "\n\n\nb6 = b6_emerging()\nprint(b6.pop('sidak'))\n"
         "pd.DataFrame({NAME[k]: {'n': d['n'], 'VR(5)': d['vr'], 'Z*(5)': d['zs'], 'p': d['p_zs'], 'CD p': d['cd_p'], "
         "'AVR': d['avr']['stat'], 'p AVR': d['avr']['p_boot']} for k, d in b6.items()}).T.round(3)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: has the BET become more efficient?\n\n"
       "**Question.** How has the predictability of the BET evolved since 1997, including around the FTSE Russell upgrade "
       "of September 2020? Models: B1, B5.\n\n"
       "1. Compute VR(5) and $Z^*(5)$ on rolling windows of 500 days, one every 21 days, and the share of windows with "
       "$|Z^*(5)| > 1.96$ in the first and in the last five years.\n"
       "2. Compute $\\hat\\rho(1)$, VR(5), $Z^*(5)$ and the Chow–Denning p-value in 1997–2006, 2007–2016 and 2017–2026.\n"
       "3. Compare VR(5) in the five years before and after 21 September 2020.\n"
       "4. List what else changed in the Romanian market at the same time.\n"
       "5. Interpretation: how would you separate these changes from the effect of the upgrade?\n\n"
       "**Report:** a table, one chart and a plan for a project."),
    code("# Reference analysis\n" + src(g.rolling_vr, g.subperiods, g.bet_upgrade, s.c1_bet_efficiency) + "\n\n\nc1 = c1_bet_efficiency()\n"
         "print({c: c1[c] for c in ['share_rej_first5y', 'share_rej_last5y']})\nprint(pd.DataFrame(c1['sub']).round(3))\n"
         "print(pd.DataFrame({p: c1['upgrade'][p] for p in ['before', 'after']}).T)\nprint('z of the difference:', round(c1['upgrade']['z_diff'], 2))"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant was asked whether the DAX (daily, since 1990) is weak-form efficient. It answered:\n\n"
       "- (a) ADF on the DAX log price does not reject a unit root, so the DAX is weak-form efficient;\n"
       "- (b) Ljung–Box $Q(10)$ has a p-value below 5%, so DAX returns are predictable and the DAX is not efficient;\n"
       "- (c) the Ljung–Box statistic of the squared returns is huge, so DAX returns are not a martingale;\n"
       "- (d) VR(10) < 1 shows positive autocorrelation (momentum) in the DAX;\n"
       "- (e) under RW1, VR(q) = 1 at every horizon and the variance of q-day returns is q times the daily variance.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\nc2 = c2_check()\nc2"),
]

if __name__ == '__main__':
    build(LECTURE, 7, 'lecture')
    build(SEMINAR, 7, 'seminar')
