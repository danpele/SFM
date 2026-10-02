r"""
ch2_common.py -- shared helpers of the Chapter 2 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_02/ch2_numbers.json (generate_all_charts.py) and sem2_results.json (seminar2.py); the
bilingual date helpers of Chapter 1; the clickable citations of Chapter 2 (DOIs checked against Crossref,
2 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values                       # noqa: E402
from ch1_common import T, date, put_date                 # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_02')

NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'snp': 'OMV Petrom (SNP)',
         'brd': 'BRD', 'snn': 'Nuclearelectrica (SNN)', 'sng': 'Romgaz (SNG)', 'tlv': 'Banca Transilvania (TLV)'}
SHORTNAME = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'snp': 'SNP', 'brd': 'BRD',
             'snn': 'SNN', 'sng': 'SNG', 'tlv': 'TLV'}
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'snp', 'brd', 'snn', 'sng']
INDICES = ['sp500', 'dax', 'bet', 'btc']


def load():
    with open(os.path.join(QL, 'ch2_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem2_results.json')) as f:
        return json.load(f)


def sci(x, d=1):
    """A small probability in LaTeX scientific notation, marked for the RO decimal comma: 6.3 \\times 10^{-5}."""
    import math
    if x == 0:
        return '0'
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return '⁅' + f'{m:.{d}f}' + '⁆' + f' \\times 10^{{{e}}}'


def big(x):
    """A large integer with a thin space for thousands (15\\,787)."""
    return f'{int(round(x)):,}'.replace(',', '\\,')


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    for k, d in N['mom'].items():
        V.int(f'm.{k}.n', d['n'])
        V.put(f'm.{k}.mean', d['mean'], 3)
        V.put(f'm.{k}.sd', d['sd'], 2)
        V.put(f'm.{k}.skew', d['skew'], 2)
        V.put(f'm.{k}.exkurt', d['exkurt'], 1)
        V.put(f'm.{k}.kurt', d['kurt'], 1)
        V.put(f'm.{k}.min', d['min'], 1)
        V.put(f'm.{k}.max', d['max'], 1)
        V.put(f'm.{k}.q01', d['q01'], 2)
        V.put(f'm.{k}.q99', d['q99'], 2)
        V.raw(f'm.{k}.jb', big(d['jb']))
        V.put(f'm.{k}.nu', d['nu'], 2)
        V.put(f'm.{k}.daic', d['aic_n'] - d['aic_t'], 0)
        V.put(f'm.{k}.sesk', d['se_skew'], 3)
        V.put(f'm.{k}.seku', d['se_kurt'], 3)
        put_date(V, f'm.{k}.mindate', d['min_date'])
        put_date(V, f'm.{k}.maxdate', d['max_date'])
        put_date(V, f'm.{k}.first', d['first'], short=True)
        V.raw(f'm.{k}.y0', d['first'][:4])
        V.put(f'm.{k}.zmin', (d['min'] - d['mean']) / d['sd'], 1)
    put_date(V, 'end', N['end'])
    V.raw('y0', N['start'][:4])
    V.raw('y1', N['end'][:4])
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refFHHex}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refBachelier}{\href{https://doi.org/10.24033/asens.476}{Bachelier (1900)}}
\newcommand{\refOsborne}{\href{https://doi.org/10.1287/opre.7.2.145}{Osborne (1959)}}
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refFama}{\href{https://doi.org/10.1086/294743}{Fama (1965)}}
\newcommand{\refStudent}{\href{https://doi.org/10.1093/biomet/6.1.1}{Student (1908)}}
\newcommand{\refPraetz}{\href{https://doi.org/10.1086/295425}{Praetz (1972)}}
\newcommand{\refBG}{\href{https://doi.org/10.1086/295634}{Blattberg \& Gonedes (1974)}}
\newcommand{\refJBa}{\href{https://doi.org/10.1016/0165-1765(80)90024-5}{Jarque \& Bera (1980)}}
\newcommand{\refJBb}{\href{https://doi.org/10.2307/1403192}{Jarque \& Bera (1987)}}
\newcommand{\refWG}{\href{https://doi.org/10.1093/biomet/55.1.1}{Wilk \& Gnanadesikan (1968)}}
\newcommand{\refChristie}{\href{https://doi.org/10.1016/0304-405X(82)90018-6}{Christie (1982)}}
\newcommand{\refBollerslev}{\href{https://doi.org/10.2307/1925546}{Bollerslev (1987)}}
\newcommand{\refLB}{\href{https://doi.org/10.1093/biomet/65.2.297}{Ljung \& Box (1978)}}
\newcommand{\refAkaike}{\href{https://doi.org/10.1109/TAC.1974.1100705}{Akaike (1974)}}
\newcommand{\refPeleCrypto}{\href{https://doi.org/10.1080/1351847X.2021.1960403}{Pele et al.\ (2023)}}
\newcommand{\refPeleEntropy}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
"""

BIB = [
    r"Akaike, H. (1974). \href{https://doi.org/10.1109/TAC.1974.1100705}{A new look at the statistical model identification}. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723.",
    r"Bachelier, L. (1900). \href{https://doi.org/10.24033/asens.476}{Théorie de la spéculation}. \textit{Annales scientifiques de l'École Normale Supérieure}, 17, 21--86.",
    r"Blattberg, R.C., Gonedes, N.J. (1974). \href{https://doi.org/10.1086/295634}{A comparison of the stable and Student distributions as statistical models for stock prices}. \textit{Journal of Business}, 47(2), 244--280.",
    r"Bollerslev, T. (1987). \href{https://doi.org/10.2307/1925546}{A conditionally heteroskedastic time series model for speculative prices and rates of return}. \textit{Review of Economics and Statistics}, 69(3), 542--547.",
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Christie, A.A. (1982). \href{https://doi.org/10.1016/0304-405X(82)90018-6}{The stochastic behavior of common stock variances: value, leverage and interest rate effects}. \textit{Journal of Financial Economics}, 10(4), 407--432.",
    r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    r"Fama, E.F. (1965). \href{https://doi.org/10.1086/294743}{The behavior of stock-market prices}. \textit{Journal of Business}, 38(1), 34--105.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Jarque, C.M., Bera, A.K. (1980). \href{https://doi.org/10.1016/0165-1765(80)90024-5}{Efficient tests for normality, homoscedasticity and serial independence of regression residuals}. \textit{Economics Letters}, 6(3), 255--259.",
    r"Jarque, C.M., Bera, A.K. (1987). \href{https://doi.org/10.2307/1403192}{A test for normality of observations and regression residuals}. \textit{International Statistical Review}, 55(2), 163--172.",
    r"Ljung, G.M., Box, G.E.P. (1978). \href{https://doi.org/10.1093/biomet/65.2.297}{On a measure of lack of fit in time series models}. \textit{Biometrika}, 65(2), 297--303.",
    r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    r"Osborne, M.F.M. (1959). \href{https://doi.org/10.1287/opre.7.2.145}{Brownian motion in the stock market}. \textit{Operations Research}, 7(2), 145--173.",
    r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847X.2021.1960403}{Are cryptos becoming alternative assets?} \textit{European Journal of Finance}, 29(10), 1064--1105.",
    r"Praetz, P.D. (1972). \href{https://doi.org/10.1086/295425}{The distribution of share price changes}. \textit{Journal of Business}, 45(1), 49--55.",
    r"Student (1908). \href{https://doi.org/10.1093/biomet/6.1.1}{The probable error of a mean}. \textit{Biometrika}, 6(1), 1--25.",
    r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    r"Wilk, M.B., Gnanadesikan, R. (1968). \href{https://doi.org/10.1093/biomet/55.1.1}{Probability plotting methods for the analysis of data}. \textit{Biometrika}, 55(1), 1--17.",
]
