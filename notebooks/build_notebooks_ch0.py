"""
build_notebooks_ch0.py -- the lecture notebook of Chapter 0 (SFM): Introduction
===============================================================================
Output: notebooks/EN/chapter0_lecture_notebook.ipynb   (Chapter 0 has no seminar)
The code is taken from Quantlets/Ch_00/generate_all_charts.py (inspect.getsource), so the notebook stays in sync
with the Quantlet and the charts.
Run:  python3 notebooks/build_notebooks_ch0.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter0_lecture_notebook.ipynb   (optional)
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(0)
import generate_all_charts as g   # noqa: E402

LECTURE = [
    md("# Statistics of Financial Markets — Chapter 0: Introduction\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Markets used throughout the course: S&P 500, DAX, BET-TR, Bitcoin and gold.\n"
       "- Growth of 100 invested since January 2015, and annualised return and volatility of each series."),
    *common_cells(),
    md("## 1. Five markets since 2015\n\n"
       "- Annualisation uses the actual number of observations per year of each series "
       "(about 252 for exchanges, 365 for Bitcoin).\n"
       "- The chart uses a log scale: equal vertical distances are equal percentage changes."),
    code('MARKETS_CH0 = ' + repr(g.MARKETS_CH0) + f'\nSTART_CH0 = {g.START_CH0!r}\n\n\n'
         + src(g.market_table, g.fig_markets)),
    code("tab = market_table()\ntab.round(2)"),
    code("fig_markets(save=False)\nplt.show()"),
]

if __name__ == '__main__':
    build(LECTURE, 0, 'lecture')
