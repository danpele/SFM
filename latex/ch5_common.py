r"""
ch5_common.py -- shared helpers of the Chapter 5 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_05/ch5_numbers.json (generate_all_charts.py) and sem5_results.json (seminar5.py);
the clickable citations of Chapter 5 (DOIs checked against Crossref, 2 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_05')
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'snp': 'OMV Petrom (SNP)', 'brd': 'BRD',
         'tlv': 'Banca Transilvania (TLV)', 'tgn': 'Transgaz (TGN)'}
SHORT = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'snp': 'SNP', 'brd': 'BRD', 'tlv': 'TLV',
         'tgn': 'TGN'}
ASSETS = ['sp500', 'dax', 'bet', 'btc']
STOCKS = ['snp', 'brd', 'tlv', 'tgn']


def load():
    with open(os.path.join(QL, 'ch5_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem5_results.json')) as f:
        return json.load(f)


def sci(x, d=1):
    """x as m \\times 10^{e} (math mode content), the mantissa marked for the RO decimal comma."""
    m, e = f'{x:.{d}e}'.split('e')
    return '⁅' + m + '⁆\\times 10^{' + str(int(e)) + '}'


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    V.raw('y1', N['end'][:4])
    for k, c in N['crash'].items():
        V.put(f'c.{k}.ret', c['ret'], 1)
        put_date(V, f'c.{k}.date', c['date'])
        V.put(f'c.{k}.z', c['z'], 1)
        V.put(f'c.{k}.sd', c['sd'], 2)
        V.raw(f'c.{k}.ly', str(int(round(c['log_years']))))
        V.raw(f'c.{k}.n4', str(c['n_below_4sd']))
        V.put(f'c.{k}.e4', c['exp_below_4sd'], 2)
        V.int(f'c.{k}.n', c['n'])
        V.raw(f'c.{k}.y0', c['first'][:4])
    for k, w in N['worst'].items():
        for i, d in enumerate(w, 1):
            V.put(f'wd.{k}.{i}', d['ret'], 1)
            put_date(V, f'wd.{k}.{i}.d', d['date'])
    C = N['c87']
    V.put('c87.lr', C['logret'], 1)
    V.put('c87.z', C['z'], 1)
    V.raw('c87.lp', str(int(round(C['log10p']))))
    for k, w in N['wait'].items():
        y = w['years_evt']
        if y >= 1000:
            V.int(f'w.{k}', round(y, -2))
        else:
            V.put(f'w.{k}', y, 0)
    for k, d in N['crash_days'].items():
        V.put(f'cd.{k}.lvl', d['level100'], 1)
        V.raw(f'cd.{k}.nb', str(d['n_below']))
    L = N['light_heavy']
    V.raw('lh.n10', sci(L['n10']))
    V.raw('lh.e10', sci(L['e10']))
    V.raw('lh.t10', sci(L['t10']))
    V.put('lh.tr', L['t_ratio'], 3)
    for k, a in N['ls'].items():
        V.put(f'ls.{k}', a, 2)
    for k, h in N['hill'].items():
        for c in ['alpha', 'lo', 'hi', 'se_iid', 'se_block', 'alpha_gain', 'alpha_1', 'alpha_5', 'ls', 'u']:
            V.put(f'h.{k}.{c}', h[c], 2)
        V.int(f'h.{k}.n', h['n'])
        V.raw(f'h.{k}.k', str(h['k']))
        V.raw(f'h.{k}.y0', h['first'][:4])
    H = N['hill']
    V.put('h.min', min(H[k]['alpha'] for k in ASSETS), 2)
    V.put('h.max', max(H[k]['alpha'] for k in ASSETS), 2)
    for k, m in N['me'].items():
        V.put(f'me.{k}.u', m['u'], 2)
        V.put(f'me.{k}.e', m['e_u'], 2)
    for k, g in N['gev'].items():
        for c in ['xi', 'se_xi', 'mu', 'sigma', 'xi_lo', 'xi_hi']:
            V.put(f'g.{k}.{c}', g[c], 2)
        for c in ['rl1', 'rl10', 'rl10_lo', 'rl10_hi', 'rl50', 'max']:
            V.put(f'g.{k}.{c}', g[c], 1)
        V.raw(f'g.{k}.months', str(g['months']))
        V.put(f'g.{k}.years', g['years'], 0 if abs(g['years'] - round(g['years'])) < 0.05 else 1)
        V.raw(f'g.{k}.nab', str(g['n_above_rl10']))
        V.put(f'g.{k}.alpha', 1 / g['xi'], 1)
    for k, r in N['risk'].items():
        p = r['pot']
        for c in ['xi', 'se_xi', 'beta', 'u']:
            V.put(f'r.{k}.{c}', p[c], 2 if c != 'beta' else 3)
        V.raw(f'r.{k}.nu', str(int(p['n_u'])))
        V.int(f'r.{k}.n', r['n'])
        V.put(f'r.{k}.alpha', 1 / p['xi'], 1)
        for lab in ['var1', 'var01', 'es25']:
            for m, v in r[lab].items():
                V.put(f'r.{k}.{lab}.{m}', v, 2)
        V.put(f'r.{k}.max', r['max'], 1)
        V.put(f'r.{k}.gap1', 100 * (r['var1']['EVT'] / r['var1']['Normal'] - 1), 0)
        V.put(f'r.{k}.gap01', r['var01']['EVT'] / r['var01']['Normal'], 1)
    for k, o in N['oos'].items():
        V.int(f'o.{k}.n', o['n_test'])
        V.int(f'o.{k}.ne', o['n_est'])
        for lab in ['var1', 'var01']:
            V.put(f'o.{k}.{lab}.exp', o[lab]['expected'], 1)
            for m in ['EVT', 'Historical', 'Normal']:
                V.put(f'o.{k}.{lab}.{m}', o[lab][m]['var'], 2)
                V.raw(f'o.{k}.{lab}.{m}.x', str(o[lab][m]['exc']))
    B = N['bet_periods']
    for p in ['before', 'after']:
        for c in ['alpha', 'se_block', 'lo', 'hi', 'u']:
            V.put(f'bp.{p}.{c}', B[p][c], 2)
        V.int(f'bp.{p}.n', B[p]['n'])
        V.raw(f'bp.{p}.k', str(B[p]['k']))
        V.raw(f'bp.{p}.y0', B[p]['first'][:4])
        V.raw(f'bp.{p}.y1', B[p]['last'][:4])
    V.put('bp.z', B['z'], 2)
    for k, rows in N['stab'].items():
        mid = [r for r in rows if 0.85 <= r['q'] <= 0.95]
        V.put(f'st.{k}.xmin', min(r['xi'] for r in mid), 2)
        V.put(f'st.{k}.xmax', max(r['xi'] for r in mid), 2)
        V.put(f'st.{k}.vmin', min(r['var1'] for r in mid), 2)
        V.put(f'st.{k}.vmax', max(r['var1'] for r in mid), 2)
    V.put('gq.q', N['gpd_qq']['q_top'], 1)
    V.put('gq.y', N['gpd_qq']['y_top'], 1)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refHill}{\href{https://doi.org/10.1214/aos/1176343247}{Hill (1975)}}
\newcommand{\refFT}{\href{https://doi.org/10.1017/S0305004100015681}{Fisher \& Tippett (1928)}}
\newcommand{\refGnedenko}{\href{https://doi.org/10.2307/1968974}{Gnedenko (1943)}}
\newcommand{\refPickands}{\href{https://doi.org/10.1214/aos/1176343003}{Pickands (1975)}}
\newcommand{\refBdH}{\href{https://doi.org/10.1214/aop/1176996548}{Balkema \& de Haan (1974)}}
\newcommand{\refMF}{\href{https://doi.org/10.1016/S0927-5398(00)00012-8}{McNeil \& Frey (2000)}}
\newcommand{\refQRM}{\href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{McNeil, Frey \& Embrechts (2015)}}
\newcommand{\refEKM}{\href{https://doi.org/10.1007/978-3-642-33483-2}{Embrechts, Klüppelberg \& Mikosch (1997)}}
\newcommand{\refGumbel}{\href{https://doi.org/10.7312/gumb92958}{Gumbel (1958)}}
\newcommand{\refWeissman}{\href{https://doi.org/10.1080/01621459.1978.10480104}{Weissman (1978)}}
\newcommand{\refDS}{\href{https://doi.org/10.1111/j.2517-6161.1990.tb01796.x}{Davison \& Smith (1990)}}
\newcommand{\refColes}{\href{https://doi.org/10.1007/978-1-4471-3675-0}{Coles (2001)}}
\newcommand{\refGopi}{\href{https://doi.org/10.1103/PhysRevE.60.5305}{Gopikrishnan et al.\ (1999)}}
\newcommand{\refDdV}{\href{https://doi.org/10.1016/S0927-5398(97)00008-X}{Daníelsson \& de Vries (1997)}}
\newcommand{\refLongin}{\href{https://doi.org/10.1086/209695}{Longin (1996)}}
\newcommand{\refJdV}{\href{https://doi.org/10.2307/2109682}{Jansen \& de Vries (1991)}}
\newcommand{\refDRH}{\href{https://doi.org/10.1214/aos/1016120372}{Drees, Resnick \& de Haan (2000)}}
\newcommand{\refRocco}{\href{https://doi.org/10.1111/j.1467-6419.2012.00744.x}{Rocco (2014)}}
\newcommand{\refAT}{\href{https://doi.org/10.1016/S0378-4266(02)00283-2}{Acerbi \& Tasche (2002)}}
\newcommand{\refResnick}{\href{https://doi.org/10.1007/978-0-387-45024-7}{Resnick (2007)}}
\newcommand{\refJenkinson}{\href{https://doi.org/10.1002/qj.49708134804}{Jenkinson (1955)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.3390/e21121204}{Pele, Lazar \& Mazurencu-Marinescu-Pele (2019)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refCarlson}{\href{https://www.federalreserve.gov/pubs/feds/2007/200713/200713pap.pdf}{Carlson (2007)}}
\newcommand{\refBasel}{\href{https://www.bis.org/basel_framework/chapter/MAR/33.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (MAR33)}}
"""

_BIB = {
    'AT': r"Acerbi, C., Tasche, D. (2002). \href{https://doi.org/10.1016/S0378-4266(02)00283-2}{On the coherence of expected shortfall}. \textit{Journal of Banking \& Finance}, 26(7), 1487--1503.",
    'BdH': r"Balkema, A.A., de Haan, L. (1974). \href{https://doi.org/10.1214/aop/1176996548}{Residual life time at great age}. \textit{Annals of Probability}, 2(5), 792--804.",
    'Basel': r"Basel Committee on Banking Supervision. \href{https://www.bis.org/basel_framework/chapter/MAR/33.htm}{MAR33: Internal models approach -- capital requirements calculation}. \textit{The Basel Framework}. Bank for International Settlements.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'Carlson': r"Carlson, M. (2007). \href{https://www.federalreserve.gov/pubs/feds/2007/200713/200713pap.pdf}{A brief history of the 1987 stock market crash with a discussion of the Federal Reserve response}. \textit{Finance and Economics Discussion Series} 2007-13, Board of Governors of the Federal Reserve System.",
    'Coles': r"Coles, S. (2001). \href{https://doi.org/10.1007/978-1-4471-3675-0}{\textit{An Introduction to Statistical Modeling of Extreme Values}}. Springer.",
    'Cont': r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    'DdV': r"Daníelsson, J., de Vries, C.G. (1997). \href{https://doi.org/10.1016/S0927-5398(97)00008-X}{Tail index and quantile estimation with very high frequency data}. \textit{Journal of Empirical Finance}, 4(2--3), 241--257.",
    'DS': r"Davison, A.C., Smith, R.L. (1990). \href{https://doi.org/10.1111/j.2517-6161.1990.tb01796.x}{Models for exceedances over high thresholds}. \textit{Journal of the Royal Statistical Society, Series B}, 52(3), 393--425.",
    'DRH': r"Drees, H., Resnick, S., de Haan, L. (2000). \href{https://doi.org/10.1214/aos/1016120372}{How to make a Hill plot}. \textit{Annals of Statistics}, 28(1), 254--274.",
    'EKM': r"Embrechts, P., Klüppelberg, C., Mikosch, T. (1997). \href{https://doi.org/10.1007/978-3-642-33483-2}{\textit{Modelling Extremal Events for Insurance and Finance}}. Springer.",
    'FT': r"Fisher, R.A., Tippett, L.H.C. (1928). \href{https://doi.org/10.1017/S0305004100015681}{Limiting forms of the frequency distribution of the largest or smallest member of a sample}. \textit{Mathematical Proceedings of the Cambridge Philosophical Society}, 24(2), 180--190.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'Gnedenko': r"Gnedenko, B. (1943). \href{https://doi.org/10.2307/1968974}{Sur la distribution limite du terme maximum d'une série aléatoire}. \textit{Annals of Mathematics}, 44(3), 423--453.",
    'Gopi': r"Gopikrishnan, P., Plerou, V., Amaral, L.A.N., Meyer, M., Stanley, H.E. (1999). \href{https://doi.org/10.1103/PhysRevE.60.5305}{Scaling of the distribution of fluctuations of financial market indices}. \textit{Physical Review E}, 60(5), 5305--5316.",
    'Gumbel': r"Gumbel, E.J. (1958). \href{https://doi.org/10.7312/gumb92958}{\textit{Statistics of Extremes}}. Columbia University Press.",
    'Hill': r"Hill, B.M. (1975). \href{https://doi.org/10.1214/aos/1176343247}{A simple general approach to inference about the tail of a distribution}. \textit{Annals of Statistics}, 3(5), 1163--1174.",
    'JdV': r"Jansen, D.W., de Vries, C.G. (1991). \href{https://doi.org/10.2307/2109682}{On the frequency of large stock returns: putting booms and busts into perspective}. \textit{Review of Economics and Statistics}, 73(1), 18--24.",
    'Jenkinson': r"Jenkinson, A.F. (1955). \href{https://doi.org/10.1002/qj.49708134804}{The frequency distribution of the annual maximum (or minimum) values of meteorological elements}. \textit{Quarterly Journal of the Royal Meteorological Society}, 81(348), 158--171.",
    'Longin': r"Longin, F.M. (1996). \href{https://doi.org/10.1086/209695}{The asymptotic distribution of extreme stock market returns}. \textit{Journal of Business}, 69(3), 383--408.",
    'Mandelbrot': r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    'MF': r"McNeil, A.J., Frey, R. (2000). \href{https://doi.org/10.1016/S0927-5398(00)00012-8}{Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach}. \textit{Journal of Empirical Finance}, 7(3--4), 271--300.",
    'QRM': r"McNeil, A.J., Frey, R., Embrechts, P. (2015). \href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{\textit{Quantitative Risk Management: Concepts, Techniques and Tools}}, revised ed. Princeton University Press.",
    'PeleA': r"Pele, D.T., Lazar, E., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21121204}{Modeling expected shortfall using tail entropy}. \textit{Entropy}, 21(12), 1204.",
    'PeleB': r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    'Pickands': r"Pickands, J. (1975). \href{https://doi.org/10.1214/aos/1176343003}{Statistical inference using extreme order statistics}. \textit{Annals of Statistics}, 3(1), 119--131.",
    'Resnick': r"Resnick, S.I. (2007). \href{https://doi.org/10.1007/978-0-387-45024-7}{\textit{Heavy-Tail Phenomena: Probabilistic and Statistical Modeling}}. Springer.",
    'Rocco': r"Rocco, M. (2014). \href{https://doi.org/10.1111/j.1467-6419.2012.00744.x}{Extreme value theory in finance: a survey}. \textit{Journal of Economic Surveys}, 28(1), 82--108.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'Weissman': r"Weissman, I. (1978). \href{https://doi.org/10.1080/01621459.1978.10480104}{Estimation of parameters and large quantiles based on the $k$ largest observations}. \textit{Journal of the American Statistical Association}, 73(364), 812--815.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', ''))


BIB = bib()
