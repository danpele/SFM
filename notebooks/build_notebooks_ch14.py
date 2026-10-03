"""
build_notebooks_ch14.py -- lecture and seminar notebooks of Chapter 14 (SFM): crypto assets and stablecoins
===========================================================================================================
Output: notebooks/EN/chapter14_lecture_notebook.ipynb, notebooks/EN/chapter14_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_14/generate_all_charts.py and seminar14.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Both notebooks install the arch package first (not part of Colab).
Run:  python3 notebooks/build_notebooks_ch14.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter14_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter14_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 14
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(14)
import generate_all_charts as g   # noqa: E402
import seminar14 as s             # noqa: E402

INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
CONSTS = ('import matplotlib.dates as mdates\nimport statsmodels.api as sm\nfrom arch import arch_model\n'
          f'EXTRA = {g.EXTRA!r}\nNAME = {g.NAME!r}\nSTART = {g.START!r}\nASSETS = {g.ASSETS!r}\nCOLORS = {g.COLORS!r}\n'
          f'HALVINGS = {g.HALVINGS!r}\nETF_DATE = {g.ETF_DATE!r}\nEVENTS = {g.EVENTS!r}\nEPISODES = {g.EPISODES!r}\n'
          f'HILL_FRAC = {g.HILL_FRAC!r}\nVR_Q = {g.VR_Q!r}\nVR_WINDOW, VR_STEP = {g.VR_WINDOW!r}, {g.VR_STEP!r}\n'
          f'SUBPERIODS = {g.SUBPERIODS!r}\nCORR_WINDOW = {g.CORR_WINDOW!r}\nALPHA_VAR, ALPHA_ES = {g.ALPHA_VAR!r}, {g.ALPHA_ES!r}\n'
          f'POSITION = {g.POSITION!r}\nBT_START = {g.BT_START!r}\nBT_WINDOW = {g.BT_WINDOW!r}\nPEG_START = {g.PEG_START!r}\n'
          f'SVB = {g.SVB!r}\nLLAMA = {g.LLAMA!r}\nSTABLE_ID = {g.STABLE_ID!r}\nDAYS = {g.DAYS!r}\nWD_PERIODS = {g.WD_PERIODS!r}\n'
          f"SEED = {g.SEED!r}\n_LLAMA = {{}}\nTABLE_DIR = '.'")
BASE = [g.register_extra, g.close, g.returns, g.joint_returns, g.hac_ols, g.hill, g.describe, g.acf, g.ljung_box,
        g.garch_fit, g.hs_var, g.hs_es, g.std_t_q, g.std_t_es, g.kupiec]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 14: Crypto assets\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Bitcoin's supply rule and the halvings; prices since 2014.\n"
       "- Crypto as an asset class: moments, annualisation with 365 days, Hill tail indices, the weekday effect.\n"
       "- Volatility clustering and GARCH(1,1)-t; variance-ratio tests of weak-form efficiency, on rolling windows.\n"
       "- Correlation with equities and gold; the COVID-19, Terra/Luna and FTX episodes; hedge or safe haven.\n"
       "- Drawdowns; Bitcoin before and after the spot ETFs of January 2024.\n"
       "- Stablecoins: supply (DefiLlama), peg deviations in basis points, USDC in March 2023, the collapse of TerraUSD.\n"
       "- VaR 1% and ES 2.5% of a crypto position and a backtest.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 23; Liu and Tsyvinski "
       "(2021); Makarov and Schoar (2020); Griffin and Shams (2020); Lyons and Viswanath-Natraj (2023); Gorton et al. (2022)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in %; crypto trades 7 days a week, equities and gold 5.\n"
       "- Annualisation with the actual frequency $m$ of each series: mean $m\\bar r$, volatility $\\sqrt{m}\\,s$ "
       "($m \\approx 365$ for crypto, $\\approx 252$ for equities).\n"
       "- Two series together: join the prices on common days, then compute returns.\n"
       "- Peg deviation of a stablecoin: $d_t = 10\\,000\\,(P_t - 1)$ basis points.\n"
       "- $\\mathrm{VaR}_\\alpha = -q_\\alpha$, the loss exceeded with probability $\\alpha$ (VaR 1%); ES 2.5%: the average "
       "loss on the worst 2.5% of days."),
    code(CONSTS + '\n\n\n' + src(*BASE)),
    md("## 1. Bitcoin: the supply rule and the halvings\n\n- 50 BTC per block, halved every 210 000 blocks: 21 million in total."),
    code(src(g.btc_supply_schedule, g.fig_halving)),
    code("fig_halving(save=False)"),
    md("## 2. Crypto as an asset class\n\n- Moments, annualisation with the actual frequency and with 252 days, Hill tail indices "
       "($k = 2.5\\%$ of $n$)."),
    code("T = {k: describe(k) for k in ASSETS}\npd.DataFrame(T).T[['n', 'mean', 'sd', 'skew', 'exkurt', 'ann_vol', 'ann_vol_252', 'hill_left', "
         "'hill_left_lo', 'hill_left_hi', 'hill_right']].round(3)"),
    code(src(g.fig_rolling_vol)),
    code("fig_rolling_vol(save=False)"),
    md("## 3. The weekday effect\n\n- Weekend dummy with HAC standard errors (mean); Brown–Forsythe test (variance)."),
    code(src(g.weekday_effect, g.fig_weekday)),
    code("W = fig_weekday(save=False)\npd.DataFrame({k: {'b': v['b'], 'p': v['p'], 'var ratio': v['var_ratio'], 'p BF': v['p_bf']} "
         "for k, v in W.items()}).T.round(4)"),
    md("## 4. Volatility clustering and GARCH(1,1)-t\n\n- ACF of returns and squared returns; GARCH(1,1)-t and GJR for four assets."),
    code(src(g.fig_acf, g.garch_table, g.fig_garch_vol)),
    code("fig_acf(save=False)"),
    code("G = garch_table()\npd.DataFrame(G).T[['omega', 'alpha', 'beta', 'pers', 'half_life', 'nu', 'gamma', 't_gamma']].round(3)"),
    code("fig_garch_vol(save=False)"),
    md("## 5. Weak-form efficiency: variance ratios\n\n- Lo–MacKinlay VR(q) with the robust $Z^*(q)$; sub-periods; rolling windows "
       "of 500 observations."),
    code(src(g.variance_ratio, g.efficiency_table, g.rolling_vr, g.fig_rolling_vr)),
    code("E = efficiency_table()\npd.DataFrame({k: {f'VR({q})': v['vr'] for q, v in e['vr'].items()} | "
         "{f'Z*({q})': v['zs'] for q, v in e['vr'].items()} for k, e in E.items()}).T.round(3)"),
    code("fig_rolling_vr(save=False)"),
    md("## 6. Correlation with equities and gold\n\n- Rolling 250-day correlation; crisis episodes; returns on the worst 5% of "
       "S&P 500 days."),
    code(src(g.fig_rolling_corr, g.crisis_table, g.worst_days)),
    code("C = fig_rolling_corr(save=False)\npd.DataFrame({k: {c: v[c] for c in ['all', 'weekly', 'pre2020', 'post2020']} for k, v in C.items()}).T.round(3)"),
    code("pd.DataFrame(crisis_table()).T"),
    code("worst_days()"),
    md("## 7. Drawdowns and the spot Bitcoin ETFs"),
    code(src(g.drawdown, g.drawdown_episodes, g.fig_drawdowns, g.etf_effect, g.fig_etf)),
    code("D = fig_drawdowns(save=False)\npd.DataFrame(D['btc_episodes'])"),
    code("etf_effect()"),
    code("fig_etf(save=False)"),
    md("## 8. Stablecoins\n\n- Supply (DefiLlama), peg deviations, March 2023, TerraUSD."),
    code(src(g.stablecoin_chart, g.stablecoin_prices, g.fig_stablecoin_supply, g.peg_stats, g.fig_peg, g.fig_svb, g.fig_ust)),
    code("fig_stablecoin_supply(save=False)"),
    code("pd.DataFrame(fig_peg(save=False)).T.round(3)"),
    code("fig_svb(save=False)"),
    code("fig_ust(save=False)"),
    md("## 9. VaR 1% and ES 2.5% of a crypto position"),
    code(src(g.var_table, g.backtest_btc, g.fig_btc_var)),
    code("V = var_table()\npd.DataFrame(V).T[['hs_v', 'hs_e', 'n_v', 'n_e', 't_v', 't_e', 'g_v', 'g_e', 'hs_v_usd', 'hs10', 'sqrt10']].round(2)"),
    code("bt, df = backtest_btc()\n{m: {c: bt[m][c] for c in ['x', 'rate', 'exp', 'lr', 'p']} for m in ['HS', 'GARCH-t']}"),
    code("fig_btc_var(df, save=False)"),
    md("## 10. AI for scientific discovery: do stablecoin flows move crypto prices?\n\n"
       "- Open question: does the weekly change of the stablecoin supply predict Bitcoin's next-week return?\n"
       "- Starter result: same-week and next-week correlation and an HAC regression since 2020.\n"
       "- Check before trusting any answer, yours or an AI's: 365-day annualisation, prices joined on common days, the same "
       "week definition for both series, basis points against percent, no look-ahead in the supply data."),
    code(src(g.stablecoin_flows_btc)),
    code("stablecoin_flows_btc()"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.annualise, s.drawdown_path, s.peg_bp, s.normal_crypto, s.worst_returns, s.hill_plot, s.fisher_diff,
               s.var_position]
SEM_CONSTS = f'\nBINS = {s.BINS!r}\nBIN_LAB = {s.BIN_LAB!r}'

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 14: Crypto assets\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 14: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n"
       "- Returns and joint returns; moments and Hill tail indices; ACF and Ljung–Box; GARCH(1,1)-t; VaR and ES by "
       "historical simulation and Student-t; the Kupiec test; peg statistics; a rolling VaR backtest; crisis episodes.\n"
       "- Helpers for Part A and Part B: annualisation, drawdown of a price path, peg deviations in bp, a Normal crypto "
       "position, the worst returns of a series, the Hill plot, the Fisher $z$ test, VaR of a position."),
    code(CONSTS + SEM_CONSTS + '\n\n\n' + src(*BASE) + '\n\n\n' + src(g.peg_stats, g.backtest_btc, g.crisis_table, g.worst_days)
         + '\n\n\n' + src(*SEM_HELPERS)),
    code("# [solutions only]\n" + src(g.etf_effect, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: annualising Bitcoin\n\n"
       "**Context.** Bitcoin daily log returns: mean 0.12%, standard deviation 3.5%; an analyst uses 252 days.\n\n"
       "1. Compute the annual mean and volatility with 365 days.\n"
       "2. Compute them with 252 days and say by how much they are understated.\n"
       "3. Compute the Sharpe ratio (risk-free rate 0) correctly and with the mean × 252 and the volatility × √365.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nannualise(0.12, 3.5)"),
    md("**Interpretation of the result.** The 252-day convention understates the volatility by 17% and the mean by 31%; "
       "mixing the conventions lowers the Sharpe ratio."),
    md("## A2 [Proposed]: annualising Ethereum\n\n"
       "**Context.** Ethereum daily log returns: mean 0.20%, standard deviation 5.0%. Model: A1.\n\n"
       "1. Compute the annual mean and volatility with 365 and with 252 days.\n"
       "2. Compute the ratio of the two volatilities.\n"
       "3. Compute the correct and the mixed Sharpe ratio.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\nannualise(0.20, 5.0)"),
    md("## A3 [Solved]: the maximum drawdown of a price path\n\n"
       "**Context.** Monthly closes in thousand USD: 20, 35, 69, 50, 30, 16, 25, 45, 73.\n\n"
       "1. Write the running maximum and the drawdown of each month.\n"
       "2. Give the maximum drawdown and the gain needed to recover from the trough.\n"
       "3. Say in which month the asset recovers.\n\n"
       "**Report:** two numbers, a list and one sentence."),
    code("# Solution\ndrawdown_path([20, 35, 69, 50, 30, 16, 25, 45, 73])"),
    md("**Interpretation of the result.** A fall of 77% needs a gain of 331% to recover; the record is regained in month 9."),
    md("## A4 [Proposed]: a second price path\n\n"
       "**Context.** Quarterly closes of Ethereum in thousand USD: 1.4, 4.8, 3.2, 0.9, 1.2, 2.5, 4.9. Model: A3.\n\n"
       "1. Write the running maximum and the drawdowns.\n"
       "2. Give the maximum drawdown and the gain needed to recover.\n"
       "3. Say in which quarter the asset recovers.\n\n"
       "**Report:** two numbers, a list and one sentence."),
    code("# Solution\ndrawdown_path([1.4, 4.8, 3.2, 0.9, 1.2, 2.5, 4.9])"),
    md("## A5 [Solved]: USDC in March 2023\n\n"
       "**Context.** USDC daily closes from 10 to 13 March 2023 and the lowest daily low.\n\n"
       "1. Compute the peg deviation of each close in basis points.\n"
       "2. Compute the deviation of the lowest low.\n"
       "3. Compute the loss of a holder of 1,000,000 USDC who sold at the worst close, and at the lowest low.\n\n"
       "**Report:** five deviations, two losses and one sentence."),
    code("# Solution\npeg_bp('usdc')"),
    md("**Interpretation of the result.** The worst close was 285 bp below the peg and the low more than 12% below; who held "
       "until Monday lost almost nothing."),
    md("## A6 [Proposed]: DAI in March 2023\n\n"
       "**Context.** DAI daily closes from 10 to 13 March 2023 and the lowest daily low. Model: A5.\n\n"
       "1. Compute the four closing deviations and the deviation of the low in bp.\n"
       "2. Compute the loss on 1,000,000 DAI sold at the worst close and at the low.\n"
       "3. Explain why DAI, a crypto-backed coin, followed USDC.\n\n"
       "**Report:** five deviations, two losses and one sentence."),
    code("# Solution\npeg_bp('dai')"),
    md("## A7 [Solved]: VaR of a Bitcoin position\n\n"
       "**Context.** A position of 100,000 USD in Bitcoin.\n\n"
       "1. With Normal simple returns ($\\mu = 0.10\\%$, $\\sigma = 3.5\\%$), compute VaR 1%, ES 2.5% and the 10-day VaR 1% "
       "in % and in USD.\n"
       "2. From the 10 smallest of the last 365 log returns, compute the historical VaR 1% and ES 2.5% in % and in USD.\n\n"
       "**Report:** ten numbers and one sentence."),
    code("# Solution\nprint(normal_crypto(100_000, 0.10, 3.5))\nworst_returns('btc')"),
    md("**Interpretation of the result.** With 365 days, $k = 4$ for VaR 1% and $k = 10$ for ES 2.5%; ES 2.5% can be below "
       "VaR 1%, because ES is larger than VaR only at the same level."),
    md("## A8 [Proposed]: VaR of an Ethereum position\n\n"
       "**Context.** A position of 100,000 USD in Ethereum. Model: A7.\n\n"
       "1. With Normal simple returns ($\\mu = 0.20\\%$, $\\sigma = 5.0\\%$), compute VaR 1%, ES 2.5% and the 10-day VaR 1%.\n"
       "2. From the 10 smallest of the last 365 log returns, compute the historical VaR 1% and ES 2.5%.\n\n"
       "**Report:** ten numbers and one sentence."),
    code("# Solution\nprint(normal_crypto(100_000, 0.20, 5.0))\nworst_returns('eth')"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: Bitcoin against the S&P 500\n\n"
       "**Question.** Is Bitcoin's distribution different from that of the S&P 500 only in scale, or also in the shape of "
       "its tails?\n\n"
       "1. Compute the mean, the standard deviation, the skewness and the excess kurtosis of each series.\n"
       "2. Annualise the mean and the volatility with the actual frequency and with 252 days.\n"
       "3. Compute the Hill index of the left and of the right tail with $k = 2.5\\%$ of $n$ and its 95% interval.\n"
       "4. Draw the Hill plot of the losses for both series.\n"
       "5. Interpretation: are Bitcoin's tails heavier than those of the S&P 500?\n\n"
       "**Report:** a table of 12 numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b1_tails) + "\n\n\nb1 = b1_tails(save=False)\n"
         "pd.DataFrame(b1).T[['n', 'mean', 'sd', 'skew', 'exkurt', 'ann_vol', 'ann_vol_252', 'hill_left', 'hill_left_lo', "
         "'hill_left_hi', 'hill_right']].round(3)"),
    md("**Interpretation of the result.** The left-tail indices of Bitcoin and the S&P 500 are almost equal and their "
       "intervals overlap: Bitcoin is about four times more volatile, but its tail has the same shape."),
    md("## B2 [Proposed]: Ethereum and Solana\n\n"
       "**Question.** Do the conclusions of B1 hold for two other large crypto assets? Model: B1.\n\n"
       "1. Compute the four moments of each series.\n"
       "2. Annualise the volatility with 365 and with 252 days.\n"
       "3. Compute both Hill indices with their 95% intervals.\n"
       "4. Interpretation: which of the assets of B1 and B2 has the heaviest left tail?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\npd.DataFrame({k: describe(k) for k in ('eth', 'sol')}).T[['n', 'sd', 'skew', 'exkurt', 'ann_vol', "
         "'ann_vol_252', 'hill_left', 'hill_left_lo', 'hill_left_hi', 'hill_right']].round(3)"),
    md("## B3 [Solved]: Bitcoin and the S&P 500 in calm and crisis\n\n"
       "**Question.** Did the correlation between Bitcoin and equities change after 2020, and what happened in the 2020 "
       "and 2022 crises?\n\n"
       "1. Compute the correlation before 2020 and from 2020 on, and test their equality with the Fisher $z$ test.\n"
       "2. Compute the correlation of daily and of weekly returns year by year and draw both.\n"
       "3. Compute the cumulative returns of both series in the COVID-19, Terra/Luna and FTX episodes.\n"
       "4. Compute Bitcoin's mean return on the worst 5% of S&P 500 days.\n"
       "5. Interpretation: does Bitcoin diversify an equity portfolio when it is most needed?\n\n"
       "**Report:** six numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b3_corr) + "\n\n\nb3 = b3_corr(save=False)\n"
         "{k: b3[k] for k in ['pre', 'post', 'z', 'p', 'ci_lo', 'ci_hi', 'daily_all', 'weekly_all', 'bad_mean', 'crisis']}"),
    md("**Interpretation of the result.** The correlation rose from about 0 to about 0.4 after 2020 (Fisher $z$ above 10), "
       "and on the worst S&P 500 days Bitcoin falls as much as the index: no diversification when it is most needed."),
    md("## B4 [Proposed]: Bitcoin and gold, Ethereum and the S&P 500\n\n"
       "**Question.** Is Bitcoin linked with gold, and does Ethereum behave like Bitcoin towards equities? Model: B3.\n\n"
       "1. Compute both correlations before and after 2020 and test their equality.\n"
       "2. Compute the correlations of 2022 and 2025, daily and weekly.\n"
       "3. Compute the mean return of Bitcoin on the worst 5% of gold days, and of Ethereum on the worst 5% of S&P 500 days.\n"
       "4. Interpretation: is Bitcoin \"digital gold\"?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb4 = {'btc_gold': b3_corr('btc', 'gold', save=False), 'eth_sp500': b3_corr('eth', 'sp500', save=False)}\n"
         "pd.DataFrame({k: {c: v[c] for c in ['pre', 'post', 'z', 'p', 'bad_mean']} | {'2022': v['year']['2022'], "
         "'2025': v['year']['2025'], '2022 w': v['week_year']['2022'], '2025 w': v['week_year']['2025']} for k, v in b4.items()}).T.round(3)"),
    md("## B5 [Solved]: how stable is USDC?\n\n"
       "**Question.** How far and for how long does USDC move away from 1 USD?\n\n"
       "1. Compute the peg deviations in bp and their standard deviation and median absolute value.\n"
       "2. Compute the share of days in the classes below 1, 1–5, 5–10, 10–50 and above 50 bp, and draw them.\n"
       "3. Fit an AR(1) to the deviation and compute the half-life of a deviation.\n"
       "4. Compare the standard deviation before 2023 and after April 2023.\n"
       "5. Interpretation: can a treasurer treat USDC as cash?\n\n"
       "**Report:** six numbers, the chart and two sentences."),
    code("# Solution\n" + src(s.b5_peg) + "\n\n\nb5 = b5_peg('usdc', save=False)\nb5"),
    md("**Interpretation of the result.** On a typical day USDC is within a few bp of the peg and a deviation halves within "
       "a day; but the tail is real: one weekend in March 2023 cost up to 3% at the close and 12% at the low."),
    md("## B6 [Proposed]: USDT and DAI\n\n"
       "**Question.** Are USDT and DAI as stable as USDC? Model: B5.\n\n"
       "1. Compute the standard deviation, the median absolute deviation and the shares beyond 10 and 50 bp.\n"
       "2. Find the worst close and the worst low of each coin, with their dates.\n"
       "3. Fit the AR(1) and compute the half-lives.\n"
       "4. Interpretation: which of the three coins keeps the peg best in normal times, and which in a crisis?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6 = {k: b5_peg(k, save=False) for k in ('usdt', 'dai', 'usdc')}\n"
         "pd.DataFrame({k: {c: v[c] for c in ['sd', 'mad', 'gt10', 'gt50', 'close_min', 'close_min_date', 'low_min', 'ar', 'ar_hl']} "
         "for k, v in b6.items()}).T"),
    md("## B7 [Solved]: risk of a Bitcoin position\n\n"
       "**Question.** How much can a position of 100,000 USD in Bitcoin lose in one day, and which model would have passed "
       "a backtest?\n\n"
       "1. Compute VaR 1% and ES 2.5% by historical simulation, Normal, Student-t (maximum likelihood) and GARCH(1,1)-t for "
       "the next day.\n"
       "2. Convert them into USD with $W(1 - e^{-v/100})$.\n"
       "3. Backtest the one-day VaR 1% since 2018 (historical simulation on 365 days, GARCH(1,1)-t re-estimated every 365 "
       "days); count the exceptions and run the Kupiec test.\n"
       "4. Interpretation: which number would you report to a risk committee?\n\n"
       "**Report:** a table of eight numbers, the chart, two test results and two sentences."),
    code("# Solution\n" + src(s.b7_var) + "\n\n\nb7 = b7_var('btc', save=False)\n"
         "print(pd.DataFrame({m: b7['var'][m] for m in ['HS', 'Normal', 'Student-t', 'GARCH-t']}).round(2))\n"
         "{m: {c: b7['bt'][m][c] for c in ['x', 'rate', 'exp', 'lr', 'p']} for m in ['HS', 'GARCH-t']}"),
    md("**Interpretation of the result.** Report the historical VaR 1% as the long-run number and the GARCH-t VaR as today's "
       "number; the Normal VaR is about a quarter too low; both backtested models pass the Kupiec test."),
    md("## B8 [Proposed]: risk of an Ethereum position\n\n"
       "**Question.** Does the conclusion of B7 hold for Ethereum? Model: B7.\n\n"
       "1. Compute VaR 1% and ES 2.5% by the four methods of B7, in % and in USD.\n"
       "2. Backtest the two one-day VaR 1% forecasts since 2019 and run the Kupiec test.\n"
       "3. Find the largest number of exceptions in one calendar year for each model.\n"
       "4. Interpretation: which model would you keep for Ethereum?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb8 = b7_var('eth', start='2019-01-01', save=False)\n"
         "print(pd.DataFrame({m: b8['var'][m] for m in ['HS', 'Normal', 'Student-t', 'GARCH-t']}).round(2))\n"
         "{m: {c: b8['bt'][m][c] for c in ['x', 'rate', 'exp', 'lr', 'p', 'year']} for m in ['HS', 'GARCH-t']}"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: did the spot ETFs change Bitcoin?\n\n"
       "**Question.** Since 11 January 2024 US investors can hold Bitcoin through spot ETFs traded on weekdays; did "
       "Bitcoin's volatility, its weekend pattern or its link with equities change? Models: B1, B3.\n\n"
       "1. Compute the annualised volatility two years before and two years after the launch, and test the equality of "
       "variances (Brown–Forsythe).\n"
       "2. Compute the ratio of weekend to weekday variance in both windows.\n"
       "3. Compute the correlation with the S&P 500 and the GARCH(1,1)-t persistence in both windows.\n"
       "4. Interpretation: can a before–after comparison prove an effect of the ETFs?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\nc1 = etf_effect()\nc1"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the risks of Bitcoin and stablecoins. It answered:\n\n"
       "- (a) Bitcoin's annual volatility is the daily standard deviation times √252;\n"
       "- (b) the \"VaR 99%\" of 100,000 USD in Bitcoin is a negative number of dollars equal to 1,000 times the VaR in %;\n"
       "- (c) on 11 March 2023 USDC lost 2.85 basis points;\n"
       "- (d) TerraUSD was a fiat-backed stablecoin with US Treasury bills as reserves;\n"
       "- (e) Bitcoin is a safe haven because its correlation with the S&P 500 is about zero;\n"
       "- (f) Bitcoin returns are less volatile at weekends than on weekdays.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2 = c2_check()\nc2"),
]

if __name__ == '__main__':
    build(LECTURE, 14, 'lecture')
    build(SEMINAR, 14, 'seminar')
