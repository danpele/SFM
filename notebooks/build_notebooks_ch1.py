"""
build_notebooks_ch1.py -- lecture and seminar notebooks of Chapter 1 (SFM): Data, returns and indicators
=======================================================================================================
Output: notebooks/EN/chapter1_lecture_notebook.ipynb, notebooks/EN/chapter1_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_01/generate_all_charts.py and seminar1.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch1.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter1_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter1_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 1
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(1)
import generate_all_charts as g   # noqa: E402
import seminar1 as s              # noqa: E402

CONSTS = (f'START = {g.START!r}\nASSETS = {g.ASSETS!r}\nCOLORS = {g.COLORS!r}\nSCATTER = {g.SCATTER!r}')
CORE = [g.simple_ret, g.log_ret, g.cagr, g.drawdown, g.max_drawdown, g.downside_deviation, g.sharpe_se,
        g.performance, g.describe]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 1: Data, returns and indicators\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Data sources and data quality: EUR/RON from two sources, adjusted prices, price vs total return index.\n"
       "- Simple and log returns, multi-period and portfolio returns, annualisation.\n"
       "- Descriptive statistics and performance indicators: CAGR, volatility, Sharpe, Sortino, maximum drawdown, "
       "Calmar, volatility drag.\n"
       "- Textbook: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Chapter 11."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Simple return $R_t = P_t/P_{t-1} - 1$; log return $r_t = \\ln(P_t/P_{t-1})$.\n"
       "- CAGR $= (P_T/P_0)^{1/Y} - 1$; drawdown $DD_t = P_t/\\max_{s\\le t}P_s - 1$; MDD $= \\min_t DD_t$.\n"
       "- Sharpe ratio $= \\mu/\\sigma$ (risk-free rate 0), annualised with $\\sqrt{q}$; standard error of Lo (2002).\n"
       "- Sortino ratio $= \\mu/\\sigma_D$ with the downside deviation $\\sigma_D$; Calmar ratio $=$ CAGR$/|$MDD$|$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Data quality\n\n"
       "- Two EUR/RON series: the EODHD series and the official BNR reference rate.\n"
       "- Banca Transilvania (TLV): the close jumps at corporate actions, the adjusted close does not.\n"
       "- BET (price index) vs BET-TR (dividends reinvested).\n"
       "- The last five daily bars (OHLCV) of TLV: look for suspicious rows."),
    code(src(g.fig_eurron, g.fig_tlv_adjusted, g.fig_bet_bettr, g.ohlc_last)),
    code("fig_eurron(save=False)"),
    code("fig_tlv_adjusted(save=False)"),
    code("fig_bet_bettr(save=False)"),
    code("pd.DataFrame(ohlc_last()).T"),
    md("## 2. Prices vs returns\n\n"
       "- Prices trend; returns fluctuate around a stable level.\n"
       "- The ADF (augmented Dickey-Fuller) test: $H_0$ unit root. We do not reject it for the log price, "
       "we reject it for the returns."),
    code(src(g.fig_price_returns)),
    code("fig_price_returns(save=False)"),
    md("## 3. Simple vs log returns\n\n"
       "- $r = \\ln(1+R) < R$; the gap is about $R^2/2$, negligible for daily data, large for crashes."),
    code(src(g.fig_simple_log)),
    code("fig_simple_log(save=False)"),
    md("## 4. Multi-period and portfolio returns\n\n"
       "- Log returns add over time; simple returns add across assets.\n"
       "- Growth of 100 since 2015 (log scale) and a 50/50 TLV-SNP portfolio rebalanced daily."),
    code(src(g.fig_growth, g.portfolio_example)),
    code("fig_growth(save=False)"),
    code("portfolio_example()"),
    md("## 5. Annualisation and descriptive statistics\n\n"
       "- Annual mean $= q\\times$ mean, annual volatility $= \\sqrt{q}\\times$ s.d., $q$ = actual observations per year.\n"
       "- Standard error of the annual mean $= \\sigma_{year}/\\sqrt{Y}$: it depends on the number of years only.\n"
       "- Skewness, excess kurtosis and the worst day of each series."),
    code(src(g.descriptive_table, g.fig_hist, g.fig_rolling_vol)),
    code("desc = descriptive_table()\ndesc[['n', 'obs_per_year', 'mean', 'sd', 'skew', 'exkurt', 'min', 'min_date', "
         "'ann_mean', 'ann_vol', 'se_ann_mean']].round(3)"),
    code("fig_hist(save=False)"),
    code("fig_rolling_vol(save=False)"),
    md("## 6. Performance indicators\n\n"
       "- CAGR, volatility, Sharpe (with its 95% interval), Sortino, MDD with dates, Calmar, volatility drag."),
    code(src(g.performance_table, g.fig_drawdowns, g.fig_risk_return, g.fig_sharpe_ci, g.fig_vol_drag)),
    code("perf = performance_table()\nperf[['cagr', 'vol', 'sharpe', 'sharpe_se', 'sortino', 'mdd', 'mdd_peak', "
         "'mdd_trough', 'calmar', 'drag', 'half_var']].round(3)"),
    code("fig_drawdowns(save=False)"),
    code("fig_sharpe_ci(perf, save=False)"),
    code("fig_risk_return(perf, save=False)"),
    code("fig_vol_drag(perf, save=False)"),
    md("## 7. AI for scientific discovery: do Sharpe ratios persist?\n\n"
       "- Open question: is a stock with a high Sharpe ratio in one period also good in the next?\n"
       "- Starter code: Sharpe ratios of six BVB stocks in two halves of the sample and the rank correlation.\n"
       "- Things to check before trusting any answer, yours or an AI's: adjusted prices, the annualisation factor, "
       "the standard errors, how many windows you tried."),
    code("rows = {}\nfor k in ['tlv', 'snp', 'brd', 'tgn', 'sng', 'snn']:\n"
         "    p = load_close(k, START)\n"
         "    a, b = performance(p.loc[:'2020-12-31']), performance(p.loc['2020-12-31':])\n"
         "    rows[LABELS[k]] = {'SR 2015-2020': a['sharpe'], 'SE 1': a['sharpe_se'], 'SR 2021-2026': b['sharpe'], 'SE 2': b['sharpe_se']}\n"
         "tab = pd.DataFrame(rows).T\n"
         "print('Spearman rank correlation:', round(stats.spearmanr(tab.iloc[:, 0], tab.iloc[:, 2])[0], 2))\n"
         "tab.round(2)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.returns_table, s.adjusted_prices, s.path_indicators, s.bootstrap_sharpe]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 1: Data, returns and indicators\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 1: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with a measure of precision and an "
       "interpretation question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(),
    md("## Seminar functions\n\n"
       "- Returns, CAGR, drawdown, Sharpe ratio with the Lo (2002) standard error, Sortino ratio, descriptive statistics.\n"
       "- Helpers for Part A (short price paths) and the bootstrap of the Sharpe ratio."),
    code(CONSTS + f"\nSEM_START = {s.SEM_START!r}\n\n\n" + src(*CORE) + '\n\n\n' + src(*SEM_HELPERS)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: simple and log returns over two days\n\n"
       "**Context.** A share closes at 50, 55 and 48 lei on three consecutive days.\n\n"
       "1. Compute the two daily simple returns and the two daily log returns.\n"
       "2. Compute the two-day simple return from the prices and by compounding.\n"
       "3. Check whether the sum of the simple returns and the sum of the log returns equal the two-day returns.\n\n"
       "**Report:** four daily returns, two two-day returns, one sentence on which returns add up."),
    code("# Solution\nreturns_table([50, 55, 48])"),
    md("**Interpretation of the result.** The log returns add up to $\\ln(48/50)$; the simple returns must be compounded."),
    md("## A2 [Proposed]: a three-day path\n\n"
       "**Context.** A share closes at 80, 100, 75 and 90 lei. Model: A1.\n\n"
       "1. Compute the three daily simple returns and the three daily log returns.\n"
       "2. Compute the three-day simple return from the first and last price and by compounding.\n"
       "3. Explain why the sum of the simple returns overstates the three-day return.\n\n"
       "**Report:** six daily returns, the three-day simple and log returns, one sentence."),
    code("# Solution\nreturns_table([80, 100, 75, 90])"),
    md("## A3 [Solved]: a dividend and a split\n\n"
       "**Context.** Closes on days 0-6: 30, 31, 30.5, 29.2, 29.8, 10.1, 10.3. Dividend of 1.5 on day 3; 3-for-1 split on day 5.\n\n"
       "1. Compute the adjustment factor of each event.\n"
       "2. Compute the adjusted prices of days 0-6.\n"
       "3. Compare the raw and the adjusted return of day 5.\n\n"
       "**Report:** two factors, seven adjusted prices, two returns."),
    code("# Solution\nprices = [30, 31, 30.5, 29.2, 29.8, 10.1, 10.3]\n"
         "factors = {3: 1 - 1.5 / 30.5, 5: 1 / 3}\n"
         "adj = adjusted_prices(prices, factors)\n"
         "print('factors:', {k: round(v, 4) for k, v in factors.items()})\n"
         "print('adjusted:', np.round(adj, 2))\n"
         "print('day 5, raw: %.1f%%, adjusted: %.1f%%' % (100 * (prices[5] / prices[4] - 1), 100 * (adj[5] / adj[4] - 1)))"),
    md("## A4 [Proposed]: a dividend and a consolidation\n\n"
       "**Context.** Closes on days 0-5: 2.05, 2.10, 1.95, 1.98, 19.90, 20.40. Dividend of 0.18 on day 2; ten old shares "
       "consolidated into one on day 4. Model: A3.\n\n"
       "1. Compute the adjustment factor of each event.\n"
       "2. Compute the adjusted prices of days 0-5.\n"
       "3. Compute the raw log return of day 4 and say what it would do to a volatility estimate.\n\n"
       "**Report:** two factors, six adjusted prices, one return and one sentence."),
    code("# Solution\nprices = [2.05, 2.10, 1.95, 1.98, 19.90, 20.40]\n"
         "adj = adjusted_prices(prices, {2: 1 - 0.18 / 2.10, 4: 10.0})\n"
         "print(np.round(adj, 2), 'raw log return of day 4: %.0f%%' % (100 * np.log(prices[4] / prices[3])))"),
    md("## A5 [Solved]: from daily moments to annual indicators\n\n"
       "**Context.** Daily log returns with mean 0.06% and standard deviation 1.5%; 250 trading days a year; 10 years of data.\n\n"
       "1. Compute the annual mean log return and the annual volatility.\n"
       "2. Compute the CAGR and the annual mean of simple returns implied by the volatility drag.\n"
       "3. Compute the standard error and the 95% CI of the annual mean log return.\n\n"
       "**Report:** five numbers and one sentence."),
    code("# Solution\nm, s, q, Y = 0.0006, 0.015, 250, 10\n"
         "mu, vol = m * q, s * np.sqrt(q)\n"
         "se = vol / np.sqrt(Y)\n"
         "print('annual mean log return %.1f%%, volatility %.1f%%' % (100 * mu, 100 * vol))\n"
         "print('CAGR %.1f%%, arithmetic mean %.1f%% (drag %.2f%%)' % (100 * (np.exp(mu) - 1), 100 * (mu + vol**2 / 2), 100 * vol**2 / 2))\n"
         "print('SE %.1f%%, 95%% CI [%.1f%%, %.1f%%]' % (100 * se, 100 * (mu - 1.96 * se), 100 * (mu + 1.96 * se)))"),
    md("## A6 [Solved]: indicators of a short price path\n\n"
       "**Context.** Month-end values of a fund: 100, 104, 101, 107, 99, 103, 110 ($q = 12$, $r_f = 0$).\n\n"
       "1. Compute the monthly simple returns, their mean and standard deviation.\n"
       "2. Compute the annualised Sharpe and Sortino ratios.\n"
       "3. Compute the MDD, the CAGR and the Calmar ratio.\n\n"
       "**Report:** a table of returns, five indicators, one sentence on why Sortino exceeds Sharpe."),
    code("# Solution\npd.Series(path_indicators([100, 104, 101, 107, 99, 103, 110], q=12))"),
    md("## A7 [Proposed]: indicators of another fund\n\n"
       "**Context.** Month-end values: 200, 190, 205, 215, 180, 196, 210, 222 ($q = 12$, $r_f = 0$). Model: A6.\n\n"
       "1. Compute the monthly simple returns, their mean and standard deviation.\n"
       "2. Compute the annualised Sharpe and Sortino ratios.\n"
       "3. Compute the MDD, the CAGR and the Calmar ratio.\n"
       "4. Compare with A6: which indicators rank the two funds differently?\n\n"
       "**Report:** five indicators and one sentence."),
    code("# Solution\npd.DataFrame({'A6': path_indicators([100, 104, 101, 107, 99, 103, 110], q=12),\n"
         "              'A7': path_indicators([200, 190, 205, 215, 180, 196, 210, 222], q=12)}).drop('R')"),
    # ---------------- Part B
    md("# Part B: real data, precision and interpretation"),
    md("## B1 [Solved]: can we trust a EUR/RON series?\n\n"
       "**Question.** Do the EUR/RON series from EODHD and the BNR reference rate describe the same exchange rate?\n\n"
       "1. Compute the daily log returns of both series on the common days.\n"
       "2. Count the days on which the two returns differ by more than one percentage point.\n"
       "3. Compute the annualised volatility of each series, and of the EODHD series without those days.\n"
       "4. Interpretation: which series would you use to measure the currency risk of a EUR loan, and why?\n\n"
       "**Report:** the number of days, three volatilities, one sentence."),
    code("# Solution\n" + src(s.b1_eurron) + "\n\n\nres = b1_eurron(save=False)\n"
         "{k: v for k, v in res.items() if k != 'top'}"),
    md("**Interpretation of the result.** Use the BNR reference rate, the official owner of the number: the EODHD "
       "series has erroneous quotes that inflate the volatility."),
    md("## B2 [Solved]: is the mean return of the BET positive?\n\n"
       "**Question.** Is the mean daily return of the BET significantly different from zero over 2015-2026?\n\n"
       "1. Compute the mean, standard deviation, skewness, excess kurtosis, minimum (with its date) and maximum of the daily log returns.\n"
       "2. Annualise the mean and the standard deviation with the actual number of observations per year.\n"
       "3. Compute the standard error of the annual mean, its 95% CI and the $z$ statistic of $H_0$: mean $= 0$.\n"
       "4. Interpretation: does the histogram look like the Normal distribution, and what does the CI tell an investor?\n\n"
       "**Report:** a table of statistics, the CI, $z$ and $p$, two sentences."),
    code("# Solution\n" + src(s.b2_mean_ci, s.b2_chart) + "\n\n\nb2_chart(save=False)\npd.Series(b2_mean_ci('bet'))"),
    md("**Interpretation of the result.** A taller centre and heavier tails than the Normal distribution; the mean is "
       "positive, but the interval is wide."),
    md("## B3 [Proposed]: four more markets\n\n"
       "**Question.** Which of the S&P 500, DAX, Bitcoin and Banca Transilvania have a mean return significantly above zero? Model: B2.\n\n"
       "1. Repeat steps 1-3 of B2 for each of the four series.\n"
       "2. Put the annual mean, the volatility, the CI and $p$ of all five series (with the BET) in one table.\n"
       "3. Interpretation: why is the interval of Bitcoin so much wider than that of the S&P 500?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\npd.DataFrame({LABELS[k]: b2_mean_ci(k) for k in ['bet', 'sp500', 'dax', 'btc', 'tlv']}).T"
         "[['ann_mean', 'ann_vol', 'se_ann_mean', 'ci_lo', 'ci_hi', 'p']].round(3)"),
    md("## B4 [Solved]: how precise is a Sharpe ratio?\n\n"
       "**Question.** How precisely is the Sharpe ratio of the S&P 500 estimated from 2015-2026 data?\n\n"
       "1. Compute the daily mean, standard deviation and Sharpe ratio, and annualise the Sharpe ratio.\n"
       "2. Compute the standard error of Lo (2002) and the 95% CI.\n"
       "3. Bootstrap the Sharpe ratio with 2000 resamples of the days and take the 2.5% and 97.5% percentiles.\n"
       "4. Interpretation: could the true Sharpe ratio of the S&P 500 be below 0.25?\n\n"
       "**Report:** the Sharpe ratio, two intervals and one sentence."),
    code("# Solution\n" + src(s.b4_sharpe) + "\n\n\npd.Series(b4_sharpe(save=False))"),
    md("**Interpretation of the result.** Yes: both intervals are wide; a decade of daily data does not pin down a Sharpe ratio."),
    md("## B5 [Proposed]: is Banca Transilvania better than the BET?\n\n"
       "**Question.** Is the Sharpe ratio of TLV significantly higher than that of the BET? Model: B4.\n\n"
       "1. Join the two price series on their common days, then compute daily simple returns.\n"
       "2. Compute both annualised Sharpe ratios with their Lo standard errors.\n"
       "3. Bootstrap the difference of the two Sharpe ratios, resampling the same days for both series.\n"
       "4. Interpretation: why must the days be resampled in pairs?\n\n"
       "**Report:** two Sharpe ratios, the difference with its interval, one sentence."),
    code("# Solution\n" + src(s.b5_compare) + "\n\n\nb5_compare()"),
    md("## B6 [Solved]: the deepest fall of the BET\n\n"
       "**Question.** How deep was the worst fall of the BET since its launch, and how long did the recovery take?\n\n"
       "1. Compute the drawdown series and the maximum drawdown with the dates of the peak, the trough and the recovery.\n"
       "2. Compute the gain needed to recover and the share of days spent more than 20% below the peak.\n"
       "3. Compute the CAGR, the MDD and the Calmar ratio over 2015-2026.\n"
       "4. Interpretation: what does the drawdown add to what the volatility already tells us?\n\n"
       "**Report:** the MDD with three dates, two percentages, three indicators, one sentence."),
    code("# Solution\n" + src(s.b6_drawdown) + "\n\n\npd.Series(b6_drawdown(save=False))"),
    md("**Interpretation of the result.** Volatility ignores the order of returns; the drawdown shows how long an "
       "investor stays below the starting level."),
    md("## B7 [Proposed]: Bitcoin, OMV Petrom and the volatility drag\n\n"
       "**Question.** How deep are the drawdowns of Bitcoin and SNP, and does the volatility-drag formula hold? Model: B6.\n\n"
       "1. Compute the MDD of each series with the dates of the peak, the trough and the recovery.\n"
       "2. Compute the CAGR and the Calmar ratio of each series.\n"
       "3. Compute the arithmetic annual mean of simple returns, the annual mean of log returns and their difference.\n"
       "4. Compare the difference with $\\sigma^2/2$.\n"
       "5. Interpretation: for which asset does the drag matter for a long-term investor?\n\n"
       "**Report:** one table and one sentence."),
    code("# Solution\n" + src(s.b7_drag) + "\n\n\npd.DataFrame(b7_drag()).T"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: do Sharpe ratios persist?\n\n"
       "**Question.** Does a BVB stock with a high Sharpe ratio in 2015-2020 also have a high one in 2021-2026? Models: B4, B6.\n\n"
       "1. Compute the annualised Sharpe ratio and its standard error for each stock in each half.\n"
       "2. Compute the Spearman rank correlation between the two halves.\n"
       "3. Propose one way to make the answer more reliable.\n"
       "4. Interpretation: what can six stocks tell us about persistence?\n\n"
       "**Report:** a table, the rank correlation with its $p$-value, a short project plan."),
    code("# Reference analysis\n" + src(s.c1_persistence) + "\n\n\nres = c1_persistence()\n"
         "print('Spearman %.2f (p = %.2f)' % (res['spearman'], res['p']))\npd.DataFrame(res['table']).T.round(2)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant compared Bitcoin and the BET over 2015-2026 and claimed:\n\n"
       "- (a) the annual volatility of Bitcoin is its daily volatility times $\\sqrt{252}$;\n"
       "- (b) Bitcoin's higher Sharpe ratio proves that it is the better investment;\n"
       "- (c) Bitcoin's maximum drawdown is its worst daily return;\n"
       "- (d) the log return of a 50/50 portfolio is exactly the average of the two log returns;\n"
       "- (e) for TLV the close column is the right price.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** five verdicts with one line of justification each."),
    code("# Solution\n" + src(s.c2_check) + "\n\n\npd.Series(c2_check())"),
]

if __name__ == '__main__':
    build(LECTURE, 1, 'lecture')
    build(SEMINAR, 1, 'seminar')
