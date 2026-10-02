"""
build_notebooks_ch0.py -- the lecture and seminar notebooks of Chapter 0 (SFM): Introduction
============================================================================================
Output: notebooks/EN/chapter0_lecture_notebook.ipynb, notebooks/EN/chapter0_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_00/generate_all_charts.py and Quantlets/Ch_00/seminar0.py (inspect.getsource),
so the notebooks stay in sync with the Quantlets and the slides. seminar0.py is an instructor file: the seminar
notebook is built in its full version and then split with  python3 notebooks/split_seminar_notebooks.py 0.
Run:  python3 notebooks/build_notebooks_ch0.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter0_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter0_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 0
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(0)
import generate_all_charts as g   # noqa: E402

PALETTE = code("# the palette names as attributes of the style namespace `st` (st.MainBlue, st.Teal, ...)\n"
               "for _c in ('MainBlue', 'IDAred', 'Forest', 'Amber', 'Purple', 'Orange', 'Teal', 'Crimson', 'DarkText'):\n"
               "    setattr(st, _c, globals()[_c])")

LECTURE = [
    md("# Statistics of Financial Markets — Chapter 0: Introduction\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Reproduces every chart of the Chapter 0 lecture from the course data (daily data from EODHD, to 18 September 2026).\n"
       "- Sections: five markets since 2015; risk and return; drawdowns; the VIX; the BET index; daily returns and "
       "heavy tails; Bitcoin volatility per year (the AI-discovery mini-case)."),
    *common_cells(), PALETTE,
    md("## Chapter constants and helpers\n\n"
       "- Annualisation uses the actual number of observations per year of each series "
       "(about 252 for exchanges, 365 for Bitcoin)."),
    code('MARKETS_CH0 = ' + repr(g.MARKETS_CH0) + f'\nSTART_CH0 = {g.START_CH0!r}\nSTART_LONG = {g.START_LONG!r}\n'
         + 'RISK_RETURN = ' + repr(g.RISK_RETURN) + '\nCOLOURS_RR = ' + repr(g.COLOURS_RR)
         + '\nMARKERS_RR = ' + repr(g.MARKERS_RR) + '\nOFFSETS_RR = ' + repr(g.OFFSETS_RR) + '\n\n\n'
         + src(g.market_table, g.drawdown, g.annual_vol)),
    md("## 1. Five markets since 2015\n\n"
       "- Growth of 100 invested in January 2015, log scale: equal vertical distances are equal percentage changes.\n"
       "- Table: final over initial price, annualised mean log return and annualised volatility (%)."),
    code(src(g.fig_markets)),
    code("tab = market_table()\ntab.round(2)"),
    code("fig_markets(save=False)\nplt.show()"),
    md("## 2. Risk and return, 2015–2026\n\n"
       "- Each point is one market: annualised volatility (log scale) against annualised mean log return.\n"
       "- One sample of eleven years: a different window gives a different ranking."),
    code(src(g.fig_risk_return) + "\n\n\nrr = fig_risk_return(save=False)\nplt.show()\nrr[['ann_return', 'ann_vol']].round(1)"),
    md("## 3. Crises and drawdowns, 2000–2026\n\n"
       "- Drawdown $DD_t = P_t / \\max_{s \\le t} P_s - 1$: the loss from the highest previous price."),
    code(src(g.fig_drawdowns) + "\n\n\ndd = fig_drawdowns(save=False)\nplt.show()\n"
         "for k, d in dd.items():\n    print(k, 'maximum drawdown', round(d.min(), 1), '% on', d.idxmin().date())"),
    md("## 4. The VIX, 2000–2026\n\n"
       "- The VIX is the volatility of the S&P 500 over the next 30 days expected by option traders, in % a year."),
    code(src(g.fig_vix) + "\n\n\nv, top = fig_vix(save=False)\nplt.show()\n"
         "print('mean', round(v.mean(), 1), 'median', round(v.median(), 1))"),
    md("## 5. The BET index since 1997\n\n- Launched on 19 September 1997 at 1000 points; log scale; without dividends."),
    code(src(g.fig_bet_history) + "\n\n\np = fig_bet_history(save=False)\nplt.show()\n"
         "print(p.index[0].date(), p.iloc[0], p.index[-1].date(), round(p.iloc[-1], 2))"),
    md("## 6. Daily returns and heavy tails\n\n"
       "- Daily log returns of the S&P 500 since 2000: calm and turbulent periods alternate (volatility clustering).\n"
       "- Histogram against the Normal density with the same mean and variance; kurtosis of the Normal distribution: 3."),
    code(src(g.fig_returns, g.fig_histogram) + "\n\n\nr = fig_returns(save=False)\nplt.show()\nfig_histogram(save=False)\nplt.show()"),
    code("z = (r - r.mean()) / r.std()\n"
         "print('observations:', len(r))\n"
         "print('kurtosis:', round(stats.kurtosis(r, fisher=False), 2))\n"
         "print('days beyond 4 standard deviations:', int((z.abs() > 4).sum()),\n"
         "      '; expected under the Normal distribution:', round(len(r) * 2 * stats.norm.sf(4), 2))"),
    md("## 7. AI for scientific discovery: has Bitcoin become less volatile?\n\n"
       "- Annualised volatility per calendar year: Bitcoin with 365 days, the S&P 500 with its own frequency.\n"
       "- Before drawing a conclusion: fix the break date (January 2024, the spot ETFs) in advance, compare with the "
       "S&P 500, and remember that one year is a short sample."),
    code(src(g.fig_btc_vol) + "\n\n\nbv = fig_btc_vol(save=False)\nplt.show()\n"
         "bv['ratio'] = bv['btc'] / bv['sp500']\nbv.round(2)"),
    code("print('Bitcoin, mean annual volatility 2015-2023:', round(bv.loc[2015:2023, 'btc'].mean(), 1))\n"
         "print('Bitcoin, mean annual volatility 2024-2026:', round(bv.loc[2024:2026, 'btc'].mean(), 1))"),
]


def seminar_cells():
    import seminar0 as s0
    fun = src(s0.paper_returns, s0.paper_drawdown, s0.annualise, s0.describe, s0.fig_growth, s0.fig_drawdown,
              s0.eurron_check, s0.fig_eurron)
    return [
        md("# Statistics of Financial Markets — Seminar 0: Prices, Returns, Volatility and Drawdown\n\n"
           "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
           "- The seminar comes before the Chapter 0 lecture: the definitions are in the primer below.\n"
           "- **[Solved]** exercises show the full code and output: use them as models. **[Proposed]** exercises have an "
           "empty code cell: solve them yourself; the solutions are discussed in class. Nothing is handed in."),
        *common_cells(), PALETTE,
        md("## Setup\n\n"
           "### Primer\n\n"
           "- Simple return $R_t = P_t/P_{t-1} - 1$; log return $r_t = \\ln(P_t/P_{t-1})$.\n"
           "- Simple returns compound, log returns add: $\\ln(P_T/P_0) = \\sum_t r_t$.\n"
           "- Annualised mean $= q\\,\\bar r$, annualised volatility $= \\sqrt{q}\\, s$, with $q$ observations per year "
           "(about 252 for exchanges, 365 for Bitcoin).\n"
           "- Drawdown $DD_t = P_t / \\max_{s\\le t} P_s - 1$; the maximum drawdown is its most negative value.\n\n"
           "### Seminar functions"),
        code(f"START = {s0.START!r}\n\n\n" + fun),
        md("### Setup check\n\n- Expected: 2944 prices from 2015-01-02 to 2026-09-18; first close 2058.20."),
        code("p = load_close('sp500', start='2015-01-02')\nprint(len(p), p.index[0].date(), p.index[-1].date())\n"
             "print(p.head(3).round(2))"),
        # ---------------------------------------------------------------- A1
        md("## A1 [Solved]: Simple and log returns\n\n"
           "**Question:** can two daily returns simply be added to get the two-day return? **Inputs:** $P_0 = 100$, "
           "$P_1 = 110$, $P_2 = 99$.\n\n"
           "1. Compute the simple returns $R_1, R_2$ and the log returns $r_1, r_2$.\n"
           "2. Add the two simple returns, and then add the two log returns.\n"
           "3. Compute the two-day returns and compare them with the sums.\n\n"
           "**Report:** the table and one sentence on which returns add up."),
        code("# Solution\na1 = paper_returns([100, 110, 99])\n"
             "print('simple returns, %:', a1['R'].round(2), ' log returns, %:', a1['r'].round(2))\n"
             "print('sums, %:', round(a1['sum_R'], 2), round(a1['sum_r'], 3))\n"
             "print('two-day returns, %:', round(a1['total_R'], 2), round(a1['total_r'], 3))"),
        md("**Interpretation:** the sum of the simple returns is 0% for a path that lost 1%; log returns add up, "
           "simple returns compound."),
        # ---------------------------------------------------------------- A2
        md("## A2 [Solved]: Annualisation\n\n"
           "**Inputs:** a stock index with daily mean log return 0.05% and daily volatility 1.2%; Bitcoin with daily "
           "volatility 3%.\n\n"
           "1. Annualise the mean and the volatility of the index with $q = 252$.\n"
           "2. Annualise the volatility of Bitcoin with $q = 365$, and then with $q = 252$.\n"
           "3. Explain which $q$ is right for Bitcoin."),
        code("# Solution\nprint('index, mean and volatility per year (%):', [round(x, 2) for x in annualise(0.05, 1.2, 252)])\n"
             "print('Bitcoin, volatility with 365 and 252 days (%):', round(3 * np.sqrt(365), 2), round(3 * np.sqrt(252), 2))"),
        md("**Interpretation:** $q$ is the number of observations per year of the series; Bitcoin trades every day, "
           "so $q = 365$."),
        # ---------------------------------------------------------------- A3
        md("## A3 [Solved]: Drawdown on paper\n\n"
           "**Inputs:** prices on days 0–5: 100, 120, 90, 110, 130, 104.\n\n"
           "1. Compute the running peak $M_t$.\n2. Compute the drawdown $DD_t$.\n3. Find the maximum drawdown."),
        code("# Solution\na3 = paper_drawdown([100, 120, 90, 110, 130, 104])\n"
             "print(pd.DataFrame({'P': [100, 120, 90, 110, 130, 104], 'peak': a3['peak'], 'DD %': a3['dd'].round(1)}))\n"
             "print('maximum drawdown, %:', a3['mdd'], 'on day', a3['mdd_day'])"),
        # ---------------------------------------------------------------- B1
        md("## B1 [Solved]: The S&P 500, 2015–2026\n\n"
           "**Question:** how much did the S&P 500 return, and how risky was it?\n\n"
           "1. Compute the daily simple and log returns; check the first two step by step.\n"
           "2. Compute $q$, the annualised mean log return and the annualised volatility.\n"
           "3. Compute the cumulative return $P_T/P_0 - 1$ and check it against $\\exp(\\sum_t r_t) - 1$.\n"
           "4. Plot the value of 100 invested and the drawdown; find the maximum drawdown with its dates.\n\n"
           "**Report:** $q$, annual mean and volatility, cumulative return, maximum drawdown with dates, and one sentence."),
        code("# Solution\nd = describe('sp500')\n"
             "for k in ('P0', 'P1', 'P2', 'R1', 'r1', 'R2', 'r2', 'n', 'q', 'ann_mean', 'ann_vol', 'cum', 'cum_from_r',\n"
             "          'mdd', 'mdd_peak', 'mdd_date'):\n    print(f'{k:12s}', round(d[k], 3) if isinstance(d[k], float) else d[k])"),
        code("fig_growth(d, 'ch0_sem_b1_growth', 'S&P 500', st.MainBlue, save=False)\nplt.show()\n"
             "fig_drawdown(d, 'ch0_sem_b1_drawdown', 'S&P 500', st.MainBlue, save=False)\nplt.show()"),
        md("**Interpretation:** a good average year hides a fall of about a third in one month (March 2020); report "
           "the volatility and the drawdown next to the mean."),
        # ---------------------------------------------------------------- B2
        md("## B2 [Solved]: One suspicious quote\n\n"
           "**Question:** is the largest EUR/RON move in the EODHD series a real price? **Inputs:** the EUR/RON series "
           "from EODHD (`eurron_eodhd`) and the BNR reference rate (`eurron`).\n\n"
           "1. Compute the gap between the EODHD quote and the BNR rate of the same day.\n"
           "2. Find the day with the largest absolute gap; print the quotes of the day before and after.\n"
           "3. Compute the log returns into and out of that day.\n4. Plot both series in the month around that day."),
        code("# Solution\nc = eurron_check()\n"
             "for k in ('date', 'prev', 'quote', 'next', 'bnr', 'gap', 'move_in', 'move_out', 'bnr_max_move'):\n"
             "    print(f'{k:14s}', round(c[k], 4) if isinstance(c[k], float) else c[k])\nfig_eurron(c, save=False)\nplt.show()"),
        md("**Interpretation:** a jump that fully reverses the next day and is absent from the official rate is a bad "
           "tick; the course uses the BNR rate for EUR/RON."),
        # ---------------------------------------------------------------- A4
        md("## A4 [Proposed]: New numbers\n\n"
           "**Model:** A1, A2 and A3 [Solved]. **Inputs:** a share at 50, 55, 44; a fund at 200, 180, 220, 165, 198, 230 "
           "on days 0–5; daily mean 0.03% and daily volatility 1.5%.\n\n"
           "1. Compute the simple and log returns of the share, their sums and the two-day returns.\n"
           "2. Compute the drawdown of the fund for each day and its maximum drawdown.\n"
           "3. Annualise the daily mean and volatility with $q = 252$."),
        code("# Solution\na4 = paper_returns([50, 55, 44])\nprint(a4['R'].round(2), a4['r'].round(2), round(a4['sum_R'], 2), "
             "round(a4['sum_r'], 2), round(a4['total_R'], 2), round(a4['total_r'], 2))\n"
             "a4d = paper_drawdown([200, 180, 220, 165, 198, 230])\nprint(a4d['dd'].round(1), a4d['mdd'])\n"
             "print([round(x, 2) for x in annualise(0.03, 1.5, 252)])"),
        # ---------------------------------------------------------------- B3
        md("## B3 [Proposed]: The BET-TR, 2015–2026\n\n"
           "**Model:** B1 [Solved]; reuse its code with `describe('bettr')`.\n\n"
           "1. Compute $q$, the annualised mean log return and the annualised volatility.\n"
           "2. Compute the cumulative return and the maximum drawdown with its dates.\n"
           "3. Compare each number with the S&P 500 result of B1.\n\n"
           "**Report:** a two-column table and one sentence: why is the comparison not fully fair?"),
        code("# Solution\nd3 = describe('bettr')\nfor k in ('q', 'ann_mean', 'ann_vol', 'cum', 'mdd', 'mdd_peak', 'mdd_date'):\n"
             "    print(f'{k:10s}', round(d3[k], 2) if isinstance(d3[k], float) else d3[k], '  S&P 500:', "
             "round(d[k], 2) if isinstance(d[k], float) else d[k])"),
        # ---------------------------------------------------------------- B4
        md("## B4 [Proposed]: Bitcoin, 365 or 252 days?\n\n"
           "**Model:** A2 and B1 [Solved].\n\n"
           "1. Compute $q$ from the data.\n2. Compute the annualised volatility with $q = 365$ and with $q = 252$.\n"
           "3. Compute the cumulative return and the maximum drawdown with its dates."),
        code("# Solution\nd4 = describe('btc')\nfor k in ('n', 'q', 'ann_vol_365', 'ann_vol_252', 'ann_mean', 'cum', 'mdd', 'mdd_peak', 'mdd_date'):\n"
             "    print(f'{k:12s}', round(d4[k], 2) if isinstance(d4[k], float) else d4[k])"),
        # ---------------------------------------------------------------- C2
        md("## C2 [Proposed]: Critique an AI answer\n\n"
           "**Prompt** sent to an AI assistant: *From the course data, compute the annualised volatility and the "
           "cumulative return of the S&P 500 and of Bitcoin since 2 January 2015, from daily simple returns.* "
           "The AI's code is below; it runs without an error message. Its conclusion: *Bitcoin earned only about five "
           "times as much as the S&P 500.*\n\n"
           "**Model:** A1, A2 and B1 [Solved].\n\n"
           "1. Run the AI's code and read its output.\n"
           "2. Find the three errors planted in the code.\n"
           "3. For each error, say how you would detect it.\n"
           "4. Write the corrected code and report the corrected numbers.\n"
           "5. Say whether the conclusion survives."),
        code("# The AI's answer (do not edit this cell)\n"
             "for name, sym in [('S&P 500', 'GSPC.INDX'), ('Bitcoin', 'BTC-USD.CC')]:\n"
             "    p_ai = read_market(sym)['close'].loc['2015-01-02':]   # course data, closing prices\n"
             "    R = p_ai.pct_change().dropna()          # daily simple returns\n"
             "    vol = R.std() * 252                     # annualised volatility\n"
             "    cum = R.sum()                           # cumulative return\n"
             "    print(name, round(100 * vol, 1), round(100 * cum, 1))"),
        code("# Solution\nfor name, key, q in [('S&P 500', 'sp500', 252), ('Bitcoin', 'btc', 365)]:\n"
             "    px = load_close(key, '2015-01-02')\n"
             "    R = px.pct_change().dropna()\n"
             "    vol = R.std() * np.sqrt(q)               # error 1: sqrt(q), not q; error 2: q = 365 for Bitcoin\n"
             "    cum = px.iloc[-1] / px.iloc[0] - 1       # error 3: compound, do not add simple returns\n"
             "    print(name, round(100 * vol, 1), round(100 * cum, 1))"),
        md("**Solution discussion:** volatility scales with $\\sqrt{q}$; Bitcoin needs $q = 365$; the cumulative "
           "return is $P_T/P_0 - 1$. With the corrections, the conclusion of the AI fails."),
    ]


if __name__ == '__main__':
    build(LECTURE, 0, 'lecture')
    try:
        build(seminar_cells(), 0, 'seminar')
    except ImportError:
        print('seminar0.py (instructor file) not found: seminar notebook not rebuilt')
