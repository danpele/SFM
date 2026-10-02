r"""
ch6_common.py -- shared helpers of the Chapter 6 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_06/ch6_numbers.json (generate_all_charts.py) and sem6_results.json (seminar6.py); the
bilingual date helpers of Chapter 1; the clickable citations of Chapter 6 (DOIs checked against Crossref,
2 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values                       # noqa: E402,F401
from ch1_common import DATEKEYS, T, date, put_date       # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_06')
NAMES = {'bet': 'BET', 'sp500': 'S\\&P 500', 'dax': 'DAX', 'btc': 'Bitcoin', 'tlv': 'TLV', 'snp': 'SNP', 'brd': 'BRD'}
LONG = {'tlv': 'Banca Transilvania (TLV)', 'snp': 'OMV Petrom (SNP)', 'brd': 'BRD -- Groupe Société Générale (BRD)'}
ASSETS = ['bet', 'sp500', 'dax', 'btc']
STOCKS = ['tlv', 'snp', 'brd']
MODELS = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG', 'Stable']
MNAME = {'Normal': '⟦Normal||Normală⟧', 'Student-t': 'Student-t', 'Skewed-t': '⟦skewed-t||skewed-t⟧', 'GED': 'GED',
         'Mixture': '⟦Normal mixture||amestec Normal⟧', 'NIG': 'NIG', 'Stable': '⟦stable||stabilă⟧',
         'Historical': '⟦historical||istorică⟧', 'Averaged': '⟦averaged||mediată⟧'}


def put_name(V, key, m):
    """A model name as a bilingual value: @{key} inside T(en, ro) becomes the EN or the RO name (no nested ⟦..⟧)."""
    import re
    en, ro = re.match(r'⟦(.*)\|\|(.*)⟧', MNAME[m]).groups() if MNAME[m].startswith('⟦') else (MNAME[m], MNAME[m])
    V.raw(key + '.en', en)
    V.raw(key + '.ro', ro)
    DATEKEYS.add(key)


def load():
    with open(os.path.join(QL, 'ch6_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem6_results.json')) as f:
        return json.load(f)


def sci(x, d=1):
    """A small probability in LaTeX scientific notation, marked for the RO decimal comma."""
    if x == 0:
        return '0'
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return '⁅' + f'{m:.{d}f}' + '⁆' + f' \\times 10^{{{e}}}'


def big(x):
    """A (possibly negative) large number rounded to an integer, with a thin space for thousands."""
    s = f'{abs(int(round(x))):,}'.replace(',', '\\,')
    return ('-' if round(x) < 0 else '') + s


def pval(p):
    """A p-value for the slides: three decimals, or < 0.001."""
    return '<⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.3f}' + '⁆'


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refFHHex}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refBox}{\href{https://doi.org/10.1080/01621459.1976.10480949}{Box (1976)}}
\newcommand{\refAkaike}{\href{https://doi.org/10.1109/TAC.1974.1100705}{Akaike (1974)}}
\newcommand{\refSchwarz}{\href{https://doi.org/10.1214/aos/1176344136}{Schwarz (1978)}}
\newcommand{\refKL}{\href{https://doi.org/10.1214/aoms/1177729694}{Kullback \& Leibler (1951)}}
\newcommand{\refBA}{\href{https://doi.org/10.1177/0049124104268644}{Burnham \& Anderson (2004)}}
\newcommand{\refBAbook}{\href{https://doi.org/10.1007/b97636}{Burnham \& Anderson (2002)}}
\newcommand{\refStone}{\href{https://doi.org/10.1111/j.2517-6161.1977.tb01603.x}{Stone (1977)}}
\newcommand{\refHT}{\href{https://doi.org/10.1093/biomet/76.2.297}{Hurvich \& Tsai (1989)}}
\newcommand{\refWilks}{\href{https://doi.org/10.1214/aoms/1177732360}{Wilks (1938)}}
\newcommand{\refSL}{\href{https://doi.org/10.1080/01621459.1987.10478472}{Self \& Liang (1987)}}
\newcommand{\refVuong}{\href{https://doi.org/10.2307/1912557}{Vuong (1989)}}
\newcommand{\refPearson}{\href{https://doi.org/10.1080/14786440009463897}{Pearson (1900)}}
\newcommand{\refSmirnov}{\href{https://doi.org/10.1214/aoms/1177730256}{Smirnov (1948)}}
\newcommand{\refLilliefors}{\href{https://doi.org/10.1080/01621459.1967.10482916}{Lilliefors (1967)}}
\newcommand{\refAD}{\href{https://doi.org/10.1214/aoms/1177729437}{Anderson \& Darling (1952)}}
\newcommand{\refADb}{\href{https://doi.org/10.1080/01621459.1954.10501232}{Anderson \& Darling (1954)}}
\newcommand{\refCramer}{\href{https://doi.org/10.1080/03461238.1928.10416862}{Cramér (1928)}}
\newcommand{\refStephens}{\href{https://doi.org/10.1080/01621459.1974.10480196}{Stephens (1974)}}
\newcommand{\refSGQ}{\href{https://doi.org/10.1007/BF02613687}{Stute et al.\ (1993)}}
\newcommand{\refHansen}{\href{https://doi.org/10.2307/2527081}{Hansen (1994)}}
\newcommand{\refNelson}{\href{https://doi.org/10.2307/2938260}{Nelson (1991)}}
\newcommand{\refKon}{\href{https://doi.org/10.1111/j.1540-6261.1984.tb03865.x}{Kon (1984)}}
\newcommand{\refBN}{\href{https://doi.org/10.1111/1467-9469.00045}{Barndorff-Nielsen (1997)}}
\newcommand{\refEK}{\href{https://doi.org/10.2307/3318481}{Eberlein \& Keller (1995)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refGR}{\href{https://doi.org/10.1198/016214506000001437}{Gneiting \& Raftery (2007)}}
\newcommand{\refAG}{\href{https://doi.org/10.1198/073500106000000332}{Amisano \& Giacomini (2007)}}
\newcommand{\refDPD}{\href{https://doi.org/10.1016/j.jeconom.2011.04.001}{Diks, Panchenko \& van Dijk (2011)}}
\newcommand{\refHoeting}{\href{https://doi.org/10.1214/ss/1009212519}{Hoeting et al.\ (1999)}}
\newcommand{\refDanielsson}{\href{https://doi.org/10.1016/j.jfs.2016.02.002}{Danielsson et al.\ (2016)}}
\newcommand{\refKerkhof}{\href{https://doi.org/10.1016/j.jbankfin.2009.07.025}{Kerkhof, Melenberg \& Schumacher (2010)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e21121204}{Pele, Lazar \& Mazurencu-Marinescu-Pele (2019)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
"""

_BIB = {
    'Akaike': r"Akaike, H. (1974). \href{https://doi.org/10.1109/TAC.1974.1100705}{A new look at the statistical model identification}. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723.",
    'AG': r"Amisano, G., Giacomini, R. (2007). \href{https://doi.org/10.1198/073500106000000332}{Comparing density forecasts via weighted likelihood ratio tests}. \textit{Journal of Business \& Economic Statistics}, 25(2), 177--190.",
    'AD': r"Anderson, T.W., Darling, D.A. (1952). \href{https://doi.org/10.1214/aoms/1177729437}{Asymptotic theory of certain ``goodness of fit'' criteria based on stochastic processes}. \textit{Annals of Mathematical Statistics}, 23(2), 193--212.",
    'ADb': r"Anderson, T.W., Darling, D.A. (1954). \href{https://doi.org/10.1080/01621459.1954.10501232}{A test of goodness of fit}. \textit{Journal of the American Statistical Association}, 49(268), 765--769.",
    'BN': r"Barndorff-Nielsen, O.E. (1997). \href{https://doi.org/10.1111/1467-9469.00045}{Normal inverse Gaussian distributions and stochastic volatility modelling}. \textit{Scandinavian Journal of Statistics}, 24(1), 1--13.",
    'FHHex': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'Box': r"Box, G.E.P. (1976). \href{https://doi.org/10.1080/01621459.1976.10480949}{Science and statistics}. \textit{Journal of the American Statistical Association}, 71(356), 791--799.",
    'BAbook': r"Burnham, K.P., Anderson, D.R. (2002). \href{https://doi.org/10.1007/b97636}{\textit{Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach}}, 2nd ed. Springer.",
    'BA': r"Burnham, K.P., Anderson, D.R. (2004). \href{https://doi.org/10.1177/0049124104268644}{Multimodel inference: understanding AIC and BIC in model selection}. \textit{Sociological Methods \& Research}, 33(2), 261--304.",
    'Cont': r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    'Cramer': r"Cramér, H. (1928). \href{https://doi.org/10.1080/03461238.1928.10416862}{On the composition of elementary errors}. \textit{Scandinavian Actuarial Journal}, 1928(1), 13--74.",
    'Danielsson': r"Danielsson, J., James, K.R., Valenzuela, M., Zer, I. (2016). \href{https://doi.org/10.1016/j.jfs.2016.02.002}{Model risk of risk models}. \textit{Journal of Financial Stability}, 23, 79--91.",
    'DPD': r"Diks, C., Panchenko, V., van Dijk, D. (2011). \href{https://doi.org/10.1016/j.jeconom.2011.04.001}{Likelihood-based scoring rules for comparing density forecasts in tails}. \textit{Journal of Econometrics}, 163(2), 215--230.",
    'EK': r"Eberlein, E., Keller, U. (1995). \href{https://doi.org/10.2307/3318481}{Hyperbolic distributions in finance}. \textit{Bernoulli}, 1(3), 281--299.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GR': r"Gneiting, T., Raftery, A.E. (2007). \href{https://doi.org/10.1198/016214506000001437}{Strictly proper scoring rules, prediction, and estimation}. \textit{Journal of the American Statistical Association}, 102(477), 359--378.",
    'Hansen': r"Hansen, B.E. (1994). \href{https://doi.org/10.2307/2527081}{Autoregressive conditional density estimation}. \textit{International Economic Review}, 35(3), 705--730.",
    'Hoeting': r"Hoeting, J.A., Madigan, D., Raftery, A.E., Volinsky, C.T. (1999). \href{https://doi.org/10.1214/ss/1009212519}{Bayesian model averaging: a tutorial}. \textit{Statistical Science}, 14(4), 382--417.",
    'HT': r"Hurvich, C.M., Tsai, C.-L. (1989). \href{https://doi.org/10.1093/biomet/76.2.297}{Regression and time series model selection in small samples}. \textit{Biometrika}, 76(2), 297--307.",
    'Kerkhof': r"Kerkhof, J., Melenberg, B., Schumacher, H. (2010). \href{https://doi.org/10.1016/j.jbankfin.2009.07.025}{Model risk and capital reserves}. \textit{Journal of Banking \& Finance}, 34(1), 267--279.",
    'Kon': r"Kon, S.J. (1984). \href{https://doi.org/10.1111/j.1540-6261.1984.tb03865.x}{Models of stock returns -- a comparison}. \textit{Journal of Finance}, 39(1), 147--165.",
    'KL': r"Kullback, S., Leibler, R.A. (1951). \href{https://doi.org/10.1214/aoms/1177729694}{On information and sufficiency}. \textit{Annals of Mathematical Statistics}, 22(1), 79--86.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'Lilliefors': r"Lilliefors, H.W. (1967). \href{https://doi.org/10.1080/01621459.1967.10482916}{On the Kolmogorov--Smirnov test for normality with mean and variance unknown}. \textit{Journal of the American Statistical Association}, 62(318), 399--402.",
    'Nelson': r"Nelson, D.B. (1991). \href{https://doi.org/10.2307/2938260}{Conditional heteroskedasticity in asset returns: a new approach}. \textit{Econometrica}, 59(2), 347--370.",
    'Pearson': r"Pearson, K. (1900). \href{https://doi.org/10.1080/14786440009463897}{On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling}. \textit{Philosophical Magazine}, 50(302), 157--175.",
    'PeleA': r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    'PeleB': r"Pele, D.T., Lazar, E., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21121204}{Modeling expected shortfall using tail entropy}. \textit{Entropy}, 21(12), 1204.",
    'Schwarz': r"Schwarz, G. (1978). \href{https://doi.org/10.1214/aos/1176344136}{Estimating the dimension of a model}. \textit{Annals of Statistics}, 6(2), 461--464.",
    'SL': r"Self, S.G., Liang, K.-Y. (1987). \href{https://doi.org/10.1080/01621459.1987.10478472}{Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions}. \textit{Journal of the American Statistical Association}, 82(398), 605--610.",
    'Smirnov': r"Smirnov, N. (1948). \href{https://doi.org/10.1214/aoms/1177730256}{Table for estimating the goodness of fit of empirical distributions}. \textit{Annals of Mathematical Statistics}, 19(2), 279--281.",
    'Stephens': r"Stephens, M.A. (1974). \href{https://doi.org/10.1080/01621459.1974.10480196}{EDF statistics for goodness of fit and some comparisons}. \textit{Journal of the American Statistical Association}, 69(347), 730--737.",
    'Stone': r"Stone, M. (1977). \href{https://doi.org/10.1111/j.2517-6161.1977.tb01603.x}{An asymptotic equivalence of choice of model by cross-validation and Akaike's criterion}. \textit{Journal of the Royal Statistical Society, Series B}, 39(1), 44--47.",
    'SGQ': r"Stute, W., González Manteiga, W., Presedo Quindimil, M. (1993). \href{https://doi.org/10.1007/BF02613687}{Bootstrap based goodness-of-fit-tests}. \textit{Metrika}, 40(1), 243--256.",
    'Tsay': r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    'Vuong': r"Vuong, Q.H. (1989). \href{https://doi.org/10.2307/1912557}{Likelihood ratio tests for model selection and non-nested hypotheses}. \textit{Econometrica}, 57(2), 307--333.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'Wilks': r"Wilks, S.S. (1938). \href{https://doi.org/10.1214/aoms/1177732360}{The large-sample distribution of the likelihood ratio for testing composite hypotheses}. \textit{Annals of Mathematical Statistics}, 9(1), 60--62.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower())


BIB = bib()
