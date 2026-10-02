r"""
ch1_common.py -- shared helpers of the Chapter 1 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_01/ch1_numbers.json and the CSV tables (generate_all_charts.py); dates in EN/RO;
the clickable citations of Chapter 1 (DOIs checked against Crossref).
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402

QL = os.path.join(ROOT, 'Quantlets', 'Ch_01')

MONTHS_EN = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']
MONTHS_RO = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
             'octombrie', 'noiembrie', 'decembrie']
SHORT_EN = [m[:3] for m in MONTHS_EN]
SHORT_RO = ['ian.', 'feb.', 'mar.', 'apr.', 'mai', 'iun.', 'iul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.']

NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'bettr': 'BET-TR', 'btc': 'Bitcoin',
         'tlv': 'Banca Transilvania (TLV)', 'snp': 'OMV Petrom (SNP)', 'brd': 'BRD',
         'snn': 'Nuclearelectrica (SNN)', 'tgn': 'Transgaz (TGN)'}
SHORTNAME = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'bettr': 'BET-TR', 'btc': 'Bitcoin', 'tlv': 'TLV',
             'snp': 'SNP', 'brd': 'BRD', 'snn': 'SNN', 'tgn': 'TGN'}
INDICES = ['sp500', 'dax', 'bet', 'bettr', 'btc']
STOCKS = ['tlv', 'snp', 'brd', 'snn', 'tgn']


def date(iso, short=False):
    """'2020-03-16' -> ⟦16 March 2020||16 martie 2020⟧ (short: 16 Mar 2020 / 16 mar. 2020)."""
    if iso is None:
        return '⟦not yet||încă nu⟧'
    y, m, d = (int(x) for x in iso.split('-'))
    en = (SHORT_EN if short else MONTHS_EN)[m - 1]
    ro = (SHORT_RO if short else MONTHS_RO)[m - 1]
    return f'⟦{d} {en} {y}||{d} {ro} {y}⟧'


DATEKEYS = set()


def date_lang(iso, lang, short=False):
    if iso is None:
        return 'not yet' if lang == 'en' else 'încă nu'
    y, m, d = (int(x) for x in iso.split('-'))
    mon = (SHORT_EN if short else MONTHS_EN) if lang == 'en' else (SHORT_RO if short else MONTHS_RO)
    return f'{d} {mon[m - 1]} {y}'


def put_date(V, key, iso, short=False):
    """A date value in both languages: @{key} inside T(en, ro) becomes the EN or the RO form."""
    V.raw(key + '.en', date_lang(iso, 'en', short))
    V.raw(key + '.ro', date_lang(iso, 'ro', short))
    DATEKEYS.add(key)


def T(en, ro):
    """Bilingual text; date keys (put_date) are switched to their EN / RO form."""
    def sub(text, lang):
        return re.sub(r'@\{([\w.]+)\}', lambda m: '@{' + m.group(1) + '.' + lang + '}' if m.group(1) in DATEKEYS else m.group(0), text)
    return f'⟦{sub(en, "en")}||{sub(ro, "ro")}⟧'


def load():
    with open(os.path.join(QL, 'ch1_numbers.json')) as f:
        return json.load(f)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    for k, d in N['desc'].items():
        V.int(f'd.{k}.n', d['n'])
        V.put(f'd.{k}.ppy', d['obs_per_year'], 0)
        V.put(f'd.{k}.mean', d['mean'], 3)
        V.put(f'd.{k}.sd', d['sd'], 2)
        V.put(f'd.{k}.skew', d['skew'], 2)
        V.put(f'd.{k}.exkurt', d['exkurt'], 1)
        V.put(f'd.{k}.min', d['min'], 1)
        V.put(f'd.{k}.max', d['max'], 1)
        put_date(V, f'd.{k}.mindate', d['min_date'], short=True)
        put_date(V, f'd.{k}.maxdate', d['max_date'], short=True)
        V.put(f'd.{k}.annmean', d['ann_mean'], 1)
        V.put(f'd.{k}.annvol', d['ann_vol'], 1)
        V.put(f'd.{k}.se', d['se_ann_mean'], 1)
        V.put(f'd.{k}.cilo', d['ann_mean'] - 1.96 * d['se_ann_mean'], 1)
        V.put(f'd.{k}.cihi', d['ann_mean'] + 1.96 * d['se_ann_mean'], 1)
        V.put(f'd.{k}.years', d['n'] / d['obs_per_year'], 1)
    for k, p in N['perf'].items():
        for c, dd in [('cagr', 1), ('vol', 1), ('mdd', 1), ('mean_arith', 1), ('mean_log', 1)]:
            V.put(f'p.{k}.{c}', p[c], dd, pct=True)
        for c, dd in [('drag', 2), ('half_var', 2)]:
            V.put(f'p.{k}.{c}', p[c], dd, pct=True)
        for c in ['sharpe', 'sharpe_se', 'sharpe_lo', 'sharpe_hi', 'sortino', 'calmar']:
            V.put(f'p.{k}.{c}', p[c], 2)
        V.put(f'p.{k}.years', p['years'], 1)
        put_date(V, f'p.{k}.peak', p['mdd_peak'], short=True)
        put_date(V, f'p.{k}.trough', p['mdd_trough'], short=True)
        put_date(V, f'p.{k}.rec', p['mdd_recovery'], short=True)
    put_date(V, 'start', N['start'])
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
\newcommand{\refCLM}{\href{https://doi.org/10.1515/9781400830213}{Campbell, Lo \& MacKinlay (1997)}}
\newcommand{\refSharpeA}{\href{https://doi.org/10.1086/294846}{Sharpe (1966)}}
\newcommand{\refSharpeB}{\href{https://doi.org/10.3905/jpm.1994.409501}{Sharpe (1994)}}
\newcommand{\refLo}{\href{https://doi.org/10.2469/faj.v58.n4.2453}{Lo (2002)}}
\newcommand{\refSortino}{\href{https://doi.org/10.3905/jpm.1991.409343}{Sortino \& van der Meer (1991)}}
\newcommand{\refMagdon}{\href{https://doi.org/10.1239/jap/1077134674}{Magdon-Ismail et al.\ (2004)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refBoothFama}{\href{https://doi.org/10.2469/faj.v48.n3.26}{Booth \& Fama (1992)}}
\newcommand{\refBGIR}{\href{https://doi.org/10.1093/rfs/5.4.553}{Brown et al.\ (1992)}}
\newcommand{\refHLZ}{\href{https://doi.org/10.1093/rfs/hhv059}{Harvey, Liu \& Zhu (2016)}}
\newcommand{\refDSR}{\href{https://doi.org/10.3905/jpm.2014.40.5.094}{Bailey \& López de Prado (2014)}}
\newcommand{\refBBLZ}{\href{https://doi.org/10.1090/noti1105}{Bailey et al.\ (2014)}}
\newcommand{\refPele}{\href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Pele \& Mazurencu-Marinescu (2012)}}
\newcommand{\refDF}{\href{https://doi.org/10.1080/01621459.1979.10482531}{Dickey \& Fuller (1979)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refGKX}{\href{https://doi.org/10.1093/rfs/hhaa009}{Gu, Kelly \& Xiu (2020)}}
"""

BIB = [
    r"Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J. (2014). \href{https://doi.org/10.1090/noti1105}{Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance}. \textit{Notices of the American Mathematical Society}, 61(5), 458--471.",
    r"Bailey, D.H., López de Prado, M. (2014). \href{https://doi.org/10.3905/jpm.2014.40.5.094}{The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality}. \textit{Journal of Portfolio Management}, 40(5), 94--107.",
    r"Booth, D.G., Fama, E.F. (1992). \href{https://doi.org/10.2469/faj.v48.n3.26}{Diversification returns and asset contributions}. \textit{Financial Analysts Journal}, 48(3), 26--32.",
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Brown, S.J., Goetzmann, W., Ibbotson, R.G., Ross, S.A. (1992). \href{https://doi.org/10.1093/rfs/5.4.553}{Survivorship bias in performance studies}. \textit{Review of Financial Studies}, 5(4), 553--580.",
    r"Campbell, J.Y., Lo, A.W., MacKinlay, A.C. (1997). \href{https://doi.org/10.1515/9781400830213}{\textit{The Econometrics of Financial Markets}}. Princeton University Press.",
    r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    r"Dickey, D.A., Fuller, W.A. (1979). \href{https://doi.org/10.1080/01621459.1979.10482531}{Distribution of the estimators for autoregressive time series with a unit root}. \textit{Journal of the American Statistical Association}, 74(366), 427--431.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Harvey, C.R., Liu, Y., Zhu, H. (2016). \href{https://doi.org/10.1093/rfs/hhv059}{\ldots and the cross-section of expected returns}. \textit{Review of Financial Studies}, 29(1), 5--68.",
    r"Lo, A.W. (2002). \href{https://doi.org/10.2469/faj.v58.n4.2453}{The statistics of Sharpe ratios}. \textit{Financial Analysts Journal}, 58(4), 36--52.",
    r"Magdon-Ismail, M., Atiya, A.F., Pratap, A., Abu-Mostafa, Y.S. (2004). \href{https://doi.org/10.1239/jap/1077134674}{On the maximum drawdown of a Brownian motion}. \textit{Journal of Applied Probability}, 41(1), 147--161.",
    r"Pele, D.T., Mazurencu-Marinescu, M. (2012). \href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Modelling stock market crashes: the case of Bucharest Stock Exchange}. \textit{Procedia -- Social and Behavioral Sciences}, 58, 533--542.",
    r"Sharpe, W.F. (1966). \href{https://doi.org/10.1086/294846}{Mutual fund performance}. \textit{Journal of Business}, 39(S1), 119--138.",
    r"Sharpe, W.F. (1994). \href{https://doi.org/10.3905/jpm.1994.409501}{The Sharpe ratio}. \textit{Journal of Portfolio Management}, 21(1), 49--58.",
    r"Sortino, F.A., van der Meer, R. (1991). \href{https://doi.org/10.3905/jpm.1991.409343}{Downside risk}. \textit{Journal of Portfolio Management}, 17(4), 27--31.",
    r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
]
