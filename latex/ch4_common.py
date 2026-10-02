r"""
ch4_common.py -- shared helpers of the Chapter 4 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_04/ch4_numbers.json (generate_all_charts.py) and sem4_results.json (seminar4.py); the
bilingual date helpers of Chapter 1; the clickable citations of Chapter 4 (DOIs checked against Crossref,
2 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values                       # noqa: E402,F401
from ch1_common import T, date, put_date                 # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_04')


def load():
    with open(os.path.join(QL, 'ch4_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem4_results.json')) as f:
        return json.load(f)


def sci(x, d=1):
    """A small probability in LaTeX scientific notation, marked for the RO decimal comma: 6.3 \\times 10^{-5}."""
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


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refFHHex}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refKolmogorov}{\href{https://doi.org/10.1007/978-3-642-49888-6}{Kolmogoroff (1933)}}
\newcommand{\refBachelier}{\href{https://doi.org/10.24033/asens.476}{Bachelier (1900)}}
\newcommand{\refEinstein}{\href{https://doi.org/10.1002/andp.19053220806}{Einstein (1905)}}
\newcommand{\refWiener}{\href{https://doi.org/10.1002/sapm192321131}{Wiener (1923)}}
\newcommand{\refOsborne}{\href{https://doi.org/10.1287/opre.7.2.145}{Osborne (1959)}}
\newcommand{\refBS}{\href{https://doi.org/10.1086/260062}{Black \& Scholes (1973)}}
\newcommand{\refCRR}{\href{https://doi.org/10.1016/0304-405X(79)90015-1}{Cox, Ross \& Rubinstein (1979)}}
\newcommand{\refMU}{\href{https://doi.org/10.1080/01621459.1949.10483310}{Metropolis \& Ulam (1949)}}
\newcommand{\refMarsaglia}{\href{https://doi.org/10.1073/pnas.61.1.25}{Marsaglia (1968)}}
\newcommand{\refMT}{\href{https://doi.org/10.1145/272991.272995}{Matsumoto \& Nishimura (1998)}}
\newcommand{\refGlasserman}{\href{https://doi.org/10.1007/978-0-387-21617-1}{Glasserman (2003)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refBollerslev}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refFama}{\href{https://doi.org/10.2307/2325486}{Fama (1970)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refPeleCrypto}{\href{https://doi.org/10.1080/1351847X.2021.1960403}{Pele et al.\ (2023)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
"""

BIB = [
    r"Bachelier, L. (1900). \href{https://doi.org/10.24033/asens.476}{Théorie de la spéculation}. \textit{Annales scientifiques de l'École Normale Supérieure}, 17, 21--86.",
    r"Black, F., Scholes, M. (1973). \href{https://doi.org/10.1086/260062}{The pricing of options and corporate liabilities}. \textit{Journal of Political Economy}, 81(3), 637--654.",
    r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    r"Cox, J.C., Ross, S.A., Rubinstein, M. (1979). \href{https://doi.org/10.1016/0304-405X(79)90015-1}{Option pricing: a simplified approach}. \textit{Journal of Financial Economics}, 7(3), 229--263.",
    r"Einstein, A. (1905). \href{https://doi.org/10.1002/andp.19053220806}{Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen}. \textit{Annalen der Physik}, 322(8), 549--560.",
    r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \textit{Econometrica}, 50(4), 987--1007.",
    r"Fama, E.F. (1970). \href{https://doi.org/10.2307/2325486}{Efficient capital markets: a review of theory and empirical work}. \textit{Journal of Finance}, 25(2), 383--417.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Glasserman, P. (2003). \href{https://doi.org/10.1007/978-0-387-21617-1}{\textit{Monte Carlo Methods in Financial Engineering}}. Springer.",
    r"Kolmogoroff, A. (1933). \href{https://doi.org/10.1007/978-3-642-49888-6}{\textit{Grundbegriffe der Wahrscheinlichkeitsrechnung}}. Springer, Berlin.",
    r"Marsaglia, G. (1968). \href{https://doi.org/10.1073/pnas.61.1.25}{Random numbers fall mainly in the planes}. \textit{Proceedings of the National Academy of Sciences}, 61(1), 25--28.",
    r"Matsumoto, M., Nishimura, T. (1998). \href{https://doi.org/10.1145/272991.272995}{Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator}. \textit{ACM Transactions on Modeling and Computer Simulation}, 8(1), 3--30.",
    r"Metropolis, N., Ulam, S. (1949). \href{https://doi.org/10.1080/01621459.1949.10483310}{The Monte Carlo method}. \textit{Journal of the American Statistical Association}, 44(247), 335--341.",
    r"Osborne, M.F.M. (1959). \href{https://doi.org/10.1287/opre.7.2.145}{Brownian motion in the stock market}. \textit{Operations Research}, 7(2), 145--173.",
    r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847X.2021.1960403}{Are cryptos becoming alternative assets?} \textit{European Journal of Finance}, 29(10), 1064--1105.",
    r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    r"Wiener, N. (1923). \href{https://doi.org/10.1002/sapm192321131}{Differential-space}. \textit{Journal of Mathematics and Physics}, 2(1--4), 131--174.",
]
