r"""
ch3_common.py -- shared helpers of the Chapter 3 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_03/ch3_numbers.json (generate_all_charts.py) and sem3_results.json (seminar3.py);
the clickable citations of Chapter 3 (DOIs checked against Crossref, 2 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_03')
# the chapter title in assets/course-data.js contains the Greek letter alpha (used in \subtitle and the header)
ALPHA = '\n\\DeclareUnicodeCharacter{03B1}{\\ensuremath{\\alpha}}\n'
NAMES = {'bet': 'BET', 'sp500': 'S\\&P 500', 'dax': 'DAX', 'btc': 'Bitcoin'}
ASSETS = ['bet', 'sp500', 'dax', 'btc']


def ro_tuples(path):
    """RO decks: decimals use a comma, so the parameters of S(...) are separated by semicolons,
    S(1{,}5; 0; 1; 0), and the parameterisation is written after a bar, S(alpha; beta; 1; 0 | 1)."""
    if '/RO/' not in path.replace(os.sep, '/'):
        return
    import re
    tex = open(path, encoding='utf-8').read()

    def fix(m):
        inner = m.group(1)
        main, _, par = inner.partition('; ')
        main = main.replace(', ', '; ')
        return 'S(' + main + (' \\mid ' + par if par else '') + ')'
    tex = re.sub(r'(?<![A-Za-z\\])S\(((?:\\alpha|\\beta|\\gamma|\\delta|[-\d{},.]|\s|;|_|\\hat|\{|\})*?)\)', fix, tex)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)


def load():
    with open(os.path.join(QL, 'ch3_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem3_results.json')) as f:
        return json.load(f)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    for k, f in N['fits'].items():
        m, s, t = f['mcculloch'], f['stable'], f['t']
        V.int(f'f.{k}.n', f['n'])
        V.raw(f'f.{k}.y0', f['first'][:4])
        V.put(f'f.{k}.min', f['min'], 1)
        put_date(V, f'f.{k}.mindate', f['min_date'], short=True)
        V.put(f'f.{k}.sd', f['sd'], 2)
        for c in ['alpha', 'beta', 'gamma', 'delta0', 'delta1', 'nu_alpha', 'nu_beta']:
            V.put(f'f.{k}.m.{c}', m[c], 3 if c != 'gamma' else 2)
        for c in ['q05', 'q25', 'q50', 'q75', 'q95']:
            V.put(f'f.{k}.m.{c}', m[c], 2)
        for c in ['alpha', 'se_alpha', 'beta', 'se_beta', 'gamma', 'delta0', 'delta1']:
            V.put(f'f.{k}.s.{c}', s[c], 3 if c != 'gamma' else 2)
        V.put(f'f.{k}.s.z', (2 - s['alpha']) / s['se_alpha'], 0)
        V.put(f'f.{k}.t.nu', t['nu'], 2)
        V.put(f'f.{k}.t.scale', t['scale'], 2)
        V.put(f'f.{k}.n.sigma', f['normal']['sigma'], 2)
        V.int(f'f.{k}.aic.n', round(f['normal']['aic']))
        V.int(f'f.{k}.aic.t', round(t['aic']))
        V.int(f'f.{k}.aic.s', round(s['aic']))
        V.int(f'f.{k}.aic.ts', round(s['aic'] - t['aic']))
        V.int(f'f.{k}.aic.ns', round(f['normal']['aic'] - s['aic']))
        V.put(f'f.{k}.sig2', 2 * s['gamma'] ** 2, 2)
    for k, c in N['tail_counts'].items():
        for x in [3, 5, 7, 10, 15, 20]:
            V.raw(f'tc.{k}.obs{x}', str(c[f'obs_{x}']))
            for m, lab in [('Normal', 'n'), ('Student-t', 't'), ('Stable', 's')]:
                v = c[f'{m}_{x}']
                if v < 0.05:
                    V.raw(f'tc.{k}.{lab}{x}', '\\approx 0')
                else:
                    V.put(f'tc.{k}.{lab}{x}', v, 1)
        for a in ['1', '01']:
            V.put(f'tc.{k}.var{a}.e', c[f'var{a}_emp'], 1)
            for m, lab in [('Normal', 'n'), ('Student-t', 't'), ('Stable', 's')]:
                V.put(f'tc.{k}.var{a}.{lab}', c[f'var{a}_{m}'], 1)
    for k, a in N['aggregation'].items():
        for h, d in a.items():
            V.put(f'ag.{k}.{h}', d['alpha'], 2)
            V.int(f'ag.{k}.{h}.n', d['n'])
            V.put(f'ag.{k}.{h}.se', d['se'] if d['se'] == d['se'] else 0.0, 2)
            V.put(f'ag.{k}.{h}.beta', d['beta'], 2)
    for n_, r in N['mc'].items():
        for k, d in r.items():
            V.put(f'mc.{n_}.{k}.mean', d['mean'], 3)
            V.put(f'mc.{n_}.{k}.sd', d['sd'], 3)
    V.put('mc.ratio500', N['mc']['500']['mcculloch']['sd'] / N['mc']['500']['ml']['sd'], 1)
    TT = N['tail_table']
    for k, d in TT.items():
        pn = 100 * d['normal']
        V.put(f'tt.{k}.n', pn, 2 if pn >= 1 else 3)
        V.put(f'tt.{k}.s', 100 * d['stable'], 2)
        r = d['stable'] / d['normal']
        if r < 100:
            V.put(f'tt.{k}.r', r, 1)
        elif r < 1e6:
            V.raw(f'tt.{k}.r', f'{r:,.0f}'.replace(',', '\\,'))
        else:
            m_, e_ = f'{r:.1e}'.split('e+')
            V.raw(f'tt.{k}.r', '⁅' + m_ + '⁆\\times 10^{' + str(int(e_)) + '}')
    m_, e_ = f"{100 * TT['10']['normal']:.1e}".split('e-')
    V.raw('tt.10.n.sci', '⁅' + m_ + '⁆\\times 10^{-' + str(int(e_)) + '}')
    S = N['stability']
    for c in ['q99_one', 'q99_alpha', 'q99_sqrt', 'factor', 'sqrtn']:
        V.put(f'stab.{c}', S[c], 2)
    G = N['gclt']
    V.put('gclt.gamma', G['gamma_limit'], 2)
    for c in ['n1_tail', 'n10_tail', 'n100_tail', 'tail_stable']:
        V.put(f'gclt.{c}', 100 * G[c], 2)
    V.put('gclt.tail_normal', 100 * G['tail_normal'], 4)
    C = N['cms']
    V.put('cms.ks', C['ks_stat'], 4)
    V.put('cms.p', C['ks_p'], 2)
    for c in ['q01', 'q99', 'q01_exact', 'q99_exact']:
        V.put(f'cms.{c}', C[c], 2)
    for a, d in N['tails'].items():
        if 'c' in d:
            V.put(f'tl.{a}.c', d['c'], 3)
            V.put(f'tl.{a}.sf10', 100 * d['sf10'], 3)
            V.put(f'tl.{a}.sf20', 100 * d['sf20'], 3)
    m_, e_ = f"{100 * N['tails']['2.0']['sf10']:.0e}".split('e-')
    V.raw('tl.2.sf10', m_ + '\\times 10^{-' + str(int(e_)) + '}')
    R = N['runvar_sim']
    for k, d in R.items():
        V.put(f'rv.{k}.v1000', d['v1000'], 1)
        V.put(f'rv.{k}.vend', d['vend'], 1)
    RR = N['runvar_real']
    for k in ['sim1', 'sim2', 'sim3', 'data', 'data_2010']:
        V.put(f'rvr.{k}', RR[k], 2)
    V.put('rvr.simmin', min(RR['sim1'], RR['sim2'], RR['sim3']), 1)
    V.put('rvr.simmax', max(RR['sim1'], RR['sim2'], RR['sim3']), 1)
    put_date(V, 'end', N['end'])
    V.raw('y1', N['end'][:4])
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refFama}{\href{https://doi.org/10.1086/294743}{Fama (1965)}}
\newcommand{\refFamaRoll}{\href{https://doi.org/10.1080/01621459.1968.11009311}{Fama \& Roll (1968)}}
\newcommand{\refFamaRollB}{\href{https://doi.org/10.1080/01621459.1971.10482264}{Fama \& Roll (1971)}}
\newcommand{\refCMS}{\href{https://doi.org/10.1080/01621459.1976.10480344}{Chambers, Mallows \& Stuck (1976)}}
\newcommand{\refWeron}{\href{https://doi.org/10.1016/0167-7152(95)00113-1}{Weron (1996)}}
\newcommand{\refMcCulloch}{\href{https://doi.org/10.1080/03610918608812563}{McCulloch (1986)}}
\newcommand{\refNolanA}{\href{https://doi.org/10.1080/15326349708807450}{Nolan (1997)}}
\newcommand{\refNolan}{\href{https://doi.org/10.1007/978-3-030-52915-4}{Nolan (2020)}}
\newcommand{\refNolanML}{\href{https://doi.org/10.1007/978-1-4612-0197-7_17}{Nolan (2001)}}
\newcommand{\refST}{\href{https://doi.org/10.1201/9780203738818}{Samorodnitsky \& Taqqu (1994)}}
\newcommand{\refBHW}{\href{https://doi.org/10.1007/3-540-27395-6_1}{Borak, Härdle \& Weron (2005)}}
\newcommand{\refBMW}{\href{https://doi.org/10.1007/978-3-642-18062-0_1}{Borak, Misiorek \& Weron (2011)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refOfficer}{\href{https://doi.org/10.1080/01621459.1972.10481297}{Officer (1972)}}
\newcommand{\refBG}{\href{https://doi.org/10.1086/295634}{Blattberg \& Gonedes (1974)}}
\newcommand{\refAB}{\href{https://doi.org/10.1080/07350015.1988.10509636}{Akgiray \& Booth (1988)}}
\newcommand{\refLLW}{\href{https://doi.org/10.1080/07350015.1990.10509793}{Lau, Lau \& Wingender (1990)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refRM}{\href{https://www.wiley.com/en-us/Stable+Paretian+Models+in+Finance-p-9780471953142}{Rachev \& Mittnik (2000)}}
\newcommand{\refMDC}{\href{https://doi.org/10.1016/S0895-7177(99)00106-5}{Mittnik, Doganoglu \& Chenyao (1999)}}
\newcommand{\refMRDC}{\href{https://doi.org/10.1016/S0895-7177(99)00110-7}{Mittnik et al.\ (1999)}}
\newcommand{\refKoutrouvelis}{\href{https://doi.org/10.1080/01621459.1980.10477573}{Koutrouvelis (1980)}}
\newcommand{\refMS}{\href{https://doi.org/10.1038/376046a0}{Mantegna \& Stanley (1995)}}
\newcommand{\refGopi}{\href{https://doi.org/10.1103/PhysRevE.60.5305}{Gopikrishnan et al.\ (1999)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refGS}{\href{https://doi.org/10.1080/14697680903540381}{Grabchak \& Samorodnitsky (2010)}}
\newcommand{\refHill}{\href{https://doi.org/10.1214/aos/1176343247}{Hill (1975)}}
\newcommand{\refGK}{\href{https://archive.org/details/limitdistributio00gned_0}{Gnedenko \& Kolmogorov (1954)}}
\newcommand{\refPele}{\href{https://doi.org/10.1016/S2212-5671(14)00279-2}{Pele (2014)}}
"""

_BIB = {
    'AB': r"Akgiray, V., Booth, G.G. (1988). \href{https://doi.org/10.1080/07350015.1988.10509636}{The stable-law model of stock returns}. \textit{Journal of Business \& Economic Statistics}, 6(1), 51--57.",
    'BG': r"Blattberg, R.C., Gonedes, N.J. (1974). \href{https://doi.org/10.1086/295634}{A comparison of the stable and Student distributions as statistical models for stock prices}. \textit{Journal of Business}, 47(2), 244--280.",
    'BHW': r"Borak, S., Härdle, W., Weron, R. (2005). \href{https://doi.org/10.1007/3-540-27395-6_1}{Stable distributions}. In Čížek, P., Härdle, W., Weron, R. (eds.), \textit{Statistical Tools for Finance and Insurance}, 21--44. Springer.",
    'BMW': r"Borak, S., Misiorek, A., Weron, R. (2011). \href{https://doi.org/10.1007/978-3-642-18062-0_1}{Models for heavy-tailed asset returns}. In Čížek, P., Härdle, W.K., Weron, R. (eds.), \textit{Statistical Tools for Finance and Insurance}, 2nd ed., 21--55. Springer.",
    'CMS': r"Chambers, J.M., Mallows, C.L., Stuck, B.W. (1976). \href{https://doi.org/10.1080/01621459.1976.10480344}{A method for simulating stable random variables}. \textit{Journal of the American Statistical Association}, 71(354), 340--344.",
    'Cont': r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    'Fama': r"Fama, E.F. (1965). \href{https://doi.org/10.1086/294743}{The behavior of stock-market prices}. \textit{Journal of Business}, 38(1), 34--105.",
    'FamaRoll': r"Fama, E.F., Roll, R. (1968). \href{https://doi.org/10.1080/01621459.1968.11009311}{Some properties of symmetric stable distributions}. \textit{Journal of the American Statistical Association}, 63(323), 817--836.",
    'FamaRollB': r"Fama, E.F., Roll, R. (1971). \href{https://doi.org/10.1080/01621459.1971.10482264}{Parameter estimates for symmetric stable distributions}. \textit{Journal of the American Statistical Association}, 66(334), 331--338.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GK': r"Gnedenko, B.V., Kolmogorov, A.N. (1954). \href{https://archive.org/details/limitdistributio00gned_0}{\textit{Limit Distributions for Sums of Independent Random Variables}}. Addison-Wesley.",
    'Gopi': r"Gopikrishnan, P., Plerou, V., Amaral, L.A.N., Meyer, M., Stanley, H.E. (1999). \href{https://doi.org/10.1103/PhysRevE.60.5305}{Scaling of the distribution of fluctuations of financial market indices}. \textit{Physical Review E}, 60(5), 5305--5316.",
    'GS': r"Grabchak, M., Samorodnitsky, G. (2010). \href{https://doi.org/10.1080/14697680903540381}{Do financial returns have finite or infinite variance? A paradox and an explanation}. \textit{Quantitative Finance}, 10(8), 883--893.",
    'Hill': r"Hill, B.M. (1975). \href{https://doi.org/10.1214/aos/1176343247}{A simple general approach to inference about the tail of a distribution}. \textit{Annals of Statistics}, 3(5), 1163--1174.",
    'Koutrouvelis': r"Koutrouvelis, I.A. (1980). \href{https://doi.org/10.1080/01621459.1980.10477573}{Regression-type estimation of the parameters of stable laws}. \textit{Journal of the American Statistical Association}, 75(372), 918--928.",
    'LLW': r"Lau, A.H.-L., Lau, H.-S., Wingender, J.R. (1990). \href{https://doi.org/10.1080/07350015.1990.10509793}{The distribution of stock returns: new evidence against the stable model}. \textit{Journal of Business \& Economic Statistics}, 8(2), 217--223.",
    'Mandelbrot': r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    'MS': r"Mantegna, R.N., Stanley, H.E. (1995). \href{https://doi.org/10.1038/376046a0}{Scaling behaviour in the dynamics of an economic index}. \textit{Nature}, 376(6535), 46--49.",
    'McCulloch': r"McCulloch, J.H. (1986). \href{https://doi.org/10.1080/03610918608812563}{Simple consistent estimators of stable distribution parameters}. \textit{Communications in Statistics -- Simulation and Computation}, 15(4), 1109--1136.",
    'MDC': r"Mittnik, S., Doganoglu, T., Chenyao, D. (1999). \href{https://doi.org/10.1016/S0895-7177(99)00106-5}{Computing the probability density function of the stable Paretian distribution}. \textit{Mathematical and Computer Modelling}, 29(10--12), 235--240.",
    'MRDC': r"Mittnik, S., Rachev, S.T., Doganoglu, T., Chenyao, D. (1999). \href{https://doi.org/10.1016/S0895-7177(99)00110-7}{Maximum likelihood estimation of stable Paretian models}. \textit{Mathematical and Computer Modelling}, 29(10--12), 275--293.",
    'NolanA': r"Nolan, J.P. (1997). \href{https://doi.org/10.1080/15326349708807450}{Numerical calculation of stable densities and distribution functions}. \textit{Communications in Statistics. Stochastic Models}, 13(4), 759--774.",
    'NolanML': r"Nolan, J.P. (2001). \href{https://doi.org/10.1007/978-1-4612-0197-7_17}{Maximum likelihood estimation and diagnostics for stable distributions}. In Barndorff-Nielsen, O.E., Mikosch, T., Resnick, S.I. (eds.), \textit{Lévy Processes}, 379--400. Birkhäuser.",
    'Nolan': r"Nolan, J.P. (2020). \href{https://doi.org/10.1007/978-3-030-52915-4}{\textit{Univariate Stable Distributions: Models for Heavy Tailed Data}}. Springer.",
    'Officer': r"Officer, R.R. (1972). \href{https://doi.org/10.1080/01621459.1972.10481297}{The distribution of stock returns}. \textit{Journal of the American Statistical Association}, 67(340), 807--812.",
    'Pele': r"Pele, D.T. (2014). \href{https://doi.org/10.1016/S2212-5671(14)00279-2}{A SAS approach for estimating the parameters of an alpha-stable distribution}. \textit{Procedia Economics and Finance}, 10, 68--77.",
    'RM': r"Rachev, S.T., Mittnik, S. (2000). \href{https://www.wiley.com/en-us/Stable+Paretian+Models+in+Finance-p-9780471953142}{\textit{Stable Paretian Models in Finance}}. Wiley.",
    'ST': r"Samorodnitsky, G., Taqqu, M.S. (1994). \href{https://doi.org/10.1201/9780203738818}{\textit{Stable Non-Gaussian Random Processes: Stochastic Models with Infinite Variance}}. Chapman \& Hall.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'Weron': r"Weron, R. (1996). \href{https://doi.org/10.1016/0167-7152(95)00113-1}{On the Chambers--Mallows--Stuck method for simulating skewed stable random variables}. \textit{Statistics \& Probability Letters}, 28(2), 165--171.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower())


BIB = bib()
