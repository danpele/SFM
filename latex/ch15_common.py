r"""
ch15_common.py -- shared helpers of the Chapter 15 generators (lecture and seminar), SFM
========================================================================================
Numbers from Quantlets/Ch_15/ch15_numbers.json (generate_all_charts.py) and sem15_results.json (seminar15.py);
the clickable citations of Chapter 15 (DOIs checked against Crossref, official pages checked, 3 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_15')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_15'
NAMES = {'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America', 'C': 'Citigroup', 'GS': 'Goldman Sachs',
         'MS': 'Morgan Stanley', 'WFC': 'Wells Fargo', 'DBK': 'Deutsche Bank', 'BNP': 'BNP Paribas',
         'SAN': 'Santander', 'INGA': 'ING', 'HSBA': 'HSBC', 'TLV': 'Banca Transilvania', 'BRD': 'BRD'}
US = ['JPM', 'BAC', 'C', 'GS', 'MS', 'WFC']
EU = ['DBK', 'BNP', 'SAN', 'INGA', 'HSBA']
RO = ['TLV', 'BRD']
ALL = US + EU + RO


def load():
    with open(os.path.join(QL, 'ch15_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem15_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    for reg, d in N['indices'].items():
        V.put(f'ix.{reg}.last', d['last'], 0)
        V.put(f'ix.{reg}.dd', -d['maxdd'], 0)
        put_date(V, f'ix.{reg}.ddd', d['date_dd'])
        V.int(f'ix.{reg}.n', d['n'])
    for e, d in N['episodes'].items():
        key = {'Lehman 2008': 'leh', 'Euro area 2011-2012': 'euro', 'March 2020': 'c20', 'March 2023': 'c23'}[e]
        for s, v in d.items():
            V.put(f'ep.{key}.{s}.min', -v['min'], 0)
            V.put(f'ep.{key}.{s}.end', v['end'], 0)
            put_date(V, f'ep.{key}.{s}.dmin', v['date_min'])
    for reg, d in N['corr'].items():
        for c in ('calm', 'y2017', 'max', 'last', 'c2008', 'c2020'):
            if d.get(c) is not None:
                V.put(f'co.{reg}.{c}', d[c], 2)
        put_date(V, f'co.{reg}.dmax', d['date_max'])
    F = N['fr']
    for c in ('rho_calm', 'rho_crisis', 'rho_adj'):
        V.put(f'fr.{c}', F[c], 2)
    V.put('fr.vc', F['var_calm'], 2)
    V.put('fr.vk', F['var_crisis'], 1)
    V.put('fr.delta', F['delta'], 1)
    V.raw('fr.n0', str(F['n_calm']))
    V.raw('fr.n1', str(F['n_crisis']))
    for reg, d in N['tail'].items():
        V.int(f'td.{reg}.n', d['n'])
        for c in ('corr', 'td5', 'td1', 'tau', 'rho', 'lambda'):
            V.put(f'td.{reg}.{c}', d[c], 2)
        V.put(f'td.{reg}.nu', d['nu'], 1)
    for reg, d in N['tail_fig'].items():
        V.raw(f'tf.{reg}.both', str(d['both']))
        V.put(f'tf.{reg}.exp', d['exp'], 0)
        V.put(f'tf.{reg}.ratio', d['both'] / d['exp'], 0)
    Q = N['qr']
    for c in ('a', 'b', 'q_a', 'q_m', 'var_i', 'covar', 'covar_med', 'dcovar', 'var_sys', 'a50', 'b50'):
        V.put(f'qr.{c}', Q[c], 2)
    V.put('qr.mqa', -Q['q_a'], 2)
    V.put('qr.ma', -Q['a'], 2)
    V.put('qr.bq', Q['b'] * Q['q_a'], 2)
    V.put('qr.bqm', -Q['b'] * Q['q_a'], 2)
    V.put('qr.diff', Q['q_m'] - Q['q_a'], 2)
    V.int('qr.n', Q['n'])
    for k, d in N['covar'].items():
        V.put(f'cv.{k}.b', d['b'], 2)
        V.int(f'cvn.{k}', d['n'])
        V.put(f'cvm.{k}', d['q_m'], 2)
        V.put(f'cv.{k}.var', d['var_i'], 2)
        V.put(f'cv.{k}.covar', d['covar'], 2)
        V.put(f'cv.{k}.d', d['dcovar'], 2)
        V.put(f'cv.{k}.d5', d['dcovar5'], 2)
        V.put(f'cv.{k}.lo', d['ci'][0], 2)
        V.put(f'cv.{k}.hi', d['ci'][1], 2)
    V.put('sp.vd', N['spearman_var_dcovar'], 2)
    V.put('sp.15', N['spearman_1_5'], 2)
    V.put('sp.md', N['spearman_mes_dcovar'], 2)
    for k, d in N['dyn'].items():
        for c in ('mean', 'max', 'b', 'y2017', 'last'):
            V.put(f'dy.{k}.{c}', d[c], 2)
        put_date(V, f'dy.{k}.dmax', d['date_max'])
    for k, d in N['mes'].items():
        V.put(f'me.{k}.m', d['mes5'], 2)
        V.put(f'me.{k}.lo', d['lo'], 2)
        V.put(f'me.{k}.hi', d['hi'], 2)
        V.put(f'me.{k}.m2', d['mes2'], 2)
        V.raw(f'me.{k}.n2', str(d['n2']))
        V.put(f'me.{k}.lr', 100 * d['lrmes'], 0)
        V.put(f'me.{k}.L', d['Lstar'], 1)
        V.put(f'me.{k}.beta', d['beta'], 2)
        V.put(f'me.{k}.esm', d['es5_m'], 2)
    MC = N['mes_crisis']
    V.put('mc.sp', MC['spearman'], 2)
    V.put('mc.p', MC['p'], 2)
    V.raw('mc.n', str(MC['n']))
    for k, d in MC['banks'].items():
        V.put(f'mc.{k}.m', d['mes'], 2)
        V.put(f'mc.{k}.r', d['ret'], 0)
    G = N['dgc']
    for c in ('max', 'min', 'mean', 'last', 'd2009', 'd2019'):
        V.put(f'gc.{c}', G[c], 0)
    put_date(V, 'gc.dmax', G['date_max'])
    put_date(V, 'gc.dmin', G['date_min'])
    V.raw('gc.nm', str(G['n_months']))
    for e, d in N['gnet'].items():
        V.raw(f'gn.{e[:4]}.links', str(d['links']))
        V.put(f'gn.{e[:4]}.dgc', d['dgc'], 0)
    Y = N['dy']
    V.put('dyt.total', Y['total'], 1)
    V.raw('dyt.p', str(Y['p']))
    V.int('dyt.n', Y['n'])
    for k in Y['to']:
        V.put(f'dyt.{k}.to', Y['to'][k], 0)
        V.put(f'dyt.{k}.from', Y['from'][k], 0)
        V.put(f'dyt.{k}.net', Y['net'][k], 1, sign=True)
    R = N['dy_roll']
    for c in ('min', 'max', 'last', 'mean', 'y2017', 'y2008'):
        V.put(f'dr.{c}', R[c], 0)
    put_date(V, 'dr.dmin', R['date_min'])
    put_date(V, 'dr.dmax', R['date_max'])
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, official pages verified, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refAB}{\href{https://doi.org/10.1257/aer.20120555}{Adrian \& Brunnermeier (2016)}}
\newcommand{\refAPPR}{\href{https://doi.org/10.1093/rfs/hhw088}{Acharya, Pedersen, Philippon \& Richardson (2017)}}
\newcommand{\refAER}{\href{https://doi.org/10.1257/aer.102.3.59}{Acharya, Engle \& Richardson (2012)}}
\newcommand{\refBE}{\href{https://doi.org/10.1093/rfs/hhw060}{Brownlees \& Engle (2017)}}
\newcommand{\refEJR}{\href{https://doi.org/10.1093/rof/rfu012}{Engle, Jondeau \& Rockinger (2015)}}
\newcommand{\refDYa}{\href{https://doi.org/10.1111/j.1468-0297.2008.02208.x}{Diebold \& Yilmaz (2009)}}
\newcommand{\refDYb}{\href{https://doi.org/10.1016/j.ijforecast.2011.02.006}{Diebold \& Yilmaz (2012)}}
\newcommand{\refDYc}{\href{https://doi.org/10.1016/j.jeconom.2014.04.012}{Diebold \& Yilmaz (2014)}}
\newcommand{\refBGLP}{\href{https://doi.org/10.1016/j.jfineco.2011.12.010}{Billio, Getmansky, Lo \& Pelizzon (2012)}}
\newcommand{\refFR}{\href{https://doi.org/10.1111/0022-1082.00494}{Forbes \& Rigobon (2002)}}
\newcommand{\refKB}{\href{https://doi.org/10.2307/1913643}{Koenker \& Bassett (1978)}}
\newcommand{\refBP}{\href{https://doi.org/10.1093/rfs/hhn098}{Brunnermeier \& Pedersen (2009)}}
\newcommand{\refSV}{\href{https://doi.org/10.1257/jep.25.1.29}{Shleifer \& Vishny (2011)}}
\newcommand{\refAG}{\href{https://doi.org/10.1086/262109}{Allen \& Gale (2000)}}
\newcommand{\refDD}{\href{https://doi.org/10.1086/261155}{Diamond \& Dybvig (1983)}}
\newcommand{\refBFLV}{\href{https://doi.org/10.1146/annurev-financial-110311-101754}{Bisias, Flood, Lo \& Valavanis (2012)}}
\newcommand{\refBCHP}{\href{https://doi.org/10.1093/rof/rfw026}{Benoit, Colliard, Hurlin \& Pérignon (2017)}}
\newcommand{\refPS}{\href{https://doi.org/10.1016/S0165-1765(97)00214-0}{Pesaran \& Shin (1998)}}
\newcommand{\refGranger}{\href{https://doi.org/10.2307/1912791}{Granger (1969)}}
\newcommand{\refHM}{\href{https://doi.org/10.1038/nature09659}{Haldane \& May (2011)}}
\newcommand{\refFRM}{\href{https://doi.org/10.1108/s0731-905320200000042016}{Mihoci, Althof, Chen \& Härdle (2020)}}
\newcommand{\refFRMem}{\href{https://doi.org/10.1016/j.ribaf.2021.101594}{Ben Amor, Althof \& Härdle (2022)}}
\newcommand{\refTENET}{\href{https://doi.org/10.1016/j.jeconom.2016.02.013}{Härdle, Wang \& Yu (2016)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refDM}{\href{https://doi.org/10.1111/j.1751-5823.2005.tb00254.x}{Demarta \& McNeil (2005)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refFSB}{\href{https://www.bis.org/publ/othp07.htm}{IMF, BIS \& FSB (2009)}}
\newcommand{\refBIII}{\href{https://www.bis.org/publ/bcbs189.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (2011)}}
\newcommand{\refGSIB}{\href{https://www.bis.org/bcbs/publ/d445.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (2018)}}
\newcommand{\refRBC}{\href{https://www.bis.org/basel_framework/chapter/RBC/30.htm}{⟦Basel Framework||Cadrul Basel⟧, RBC30}}
\newcommand{\refCCyB}{\href{https://www.bis.org/basel_framework/chapter/RBC/30.htm}{RBC30}}
\newcommand{\refESRB}{\href{https://www.esrb.europa.eu/about/background/html/index.en.html}{⟦(ESRB website)||(site-ul ESRB)⟧}}
\newcommand{\refCNSM}{\href{https://www.cnsmro.ro/}{⟦(CNSM website)||(site-ul CNSM)⟧}}
\newcommand{\refFed}{\href{https://www.federalreserve.gov/publications/files/svb-review-20230428.pdf}{Federal Reserve (2023)}}
\newcommand{\refFINMA}{\href{https://www.finma.ch/en/news/2023/03/20230319-mm-cs-ubs/}{FINMA (2023)}}
\newcommand{\refDraghi}{\href{https://www.ecb.europa.eu/press/key/date/2012/html/sp120726.en.html}{Draghi (2012)}}
\newcommand{\refFDIC}{\href{https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/silicon-valley.html}{FDIC (2023)}}
\newcommand{\refVLab}{\href{https://vlab.stern.nyu.edu/srisk}{V-Lab SRISK}}
"""

_BIB = {
    'AER': r"Acharya, V.V., Engle, R., Richardson, M. (2012). \href{https://doi.org/10.1257/aer.102.3.59}{Capital shortfall: a new approach to ranking and regulating systemic risks}. \textit{American Economic Review}, 102(3), 59--64.",
    'APPR': r"Acharya, V.V., Pedersen, L.H., Philippon, T., Richardson, M. (2017). \href{https://doi.org/10.1093/rfs/hhw088}{Measuring systemic risk}. \textit{Review of Financial Studies}, 30(1), 2--47.",
    'AB': r"Adrian, T., Brunnermeier, M.K. (2016). \href{https://doi.org/10.1257/aer.20120555}{CoVaR}. \textit{American Economic Review}, 106(7), 1705--1741.",
    'AG': r"Allen, F., Gale, D. (2000). \href{https://doi.org/10.1086/262109}{Financial contagion}. \textit{Journal of Political Economy}, 108(1), 1--33.",
    'BIII': r"Basel Committee on Banking Supervision (2011). \href{https://www.bis.org/publ/bcbs189.htm}{Basel III: a global regulatory framework for more resilient banks and banking systems}, revised June 2011. Bank for International Settlements.",
    'GSIB': r"Basel Committee on Banking Supervision (2018). \href{https://www.bis.org/bcbs/publ/d445.htm}{Global systemically important banks: revised assessment methodology and the higher loss absorbency requirement}. Bank for International Settlements.",
    'FRMem': r"Ben Amor, S., Althof, M., Härdle, W.K. (2022). \href{https://doi.org/10.1016/j.ribaf.2021.101594}{Financial Risk Meter for emerging markets}. \textit{Research in International Business and Finance}, 60, 101594.",
    'BCHP': r"Benoit, S., Colliard, J.-E., Hurlin, C., Pérignon, C. (2017). \href{https://doi.org/10.1093/rof/rfw026}{Where the risks lie: a survey on systemic risk}. \textit{Review of Finance}, 21(1), 109--152.",
    'BGLP': r"Billio, M., Getmansky, M., Lo, A.W., Pelizzon, L. (2012). \href{https://doi.org/10.1016/j.jfineco.2011.12.010}{Econometric measures of connectedness and systemic risk in the finance and insurance sectors}. \textit{Journal of Financial Economics}, 104(3), 535--559.",
    'BFLV': r"Bisias, D., Flood, M., Lo, A.W., Valavanis, S. (2012). \href{https://doi.org/10.1146/annurev-financial-110311-101754}{A survey of systemic risk analytics}. \textit{Annual Review of Financial Economics}, 4, 255--296.",
    'BE': r"Brownlees, C., Engle, R.F. (2017). \href{https://doi.org/10.1093/rfs/hhw060}{SRISK: a conditional capital shortfall measure of systemic risk}. \textit{Review of Financial Studies}, 30(1), 48--79.",
    'BP': r"Brunnermeier, M.K., Pedersen, L.H. (2009). \href{https://doi.org/10.1093/rfs/hhn098}{Market liquidity and funding liquidity}. \textit{Review of Financial Studies}, 22(6), 2201--2238.",
    'DM': r"Demarta, S., McNeil, A.J. (2005). \href{https://doi.org/10.1111/j.1751-5823.2005.tb00254.x}{The t copula and related copulas}. \textit{International Statistical Review}, 73(1), 111--129.",
    'DD': r"Diamond, D.W., Dybvig, P.H. (1983). \href{https://doi.org/10.1086/261155}{Bank runs, deposit insurance, and liquidity}. \textit{Journal of Political Economy}, 91(3), 401--419.",
    'DYa': r"Diebold, F.X., Yilmaz, K. (2009). \href{https://doi.org/10.1111/j.1468-0297.2008.02208.x}{Measuring financial asset return and volatility spillovers, with application to global equity markets}. \textit{Economic Journal}, 119(534), 158--171.",
    'DYb': r"Diebold, F.X., Yilmaz, K. (2012). \href{https://doi.org/10.1016/j.ijforecast.2011.02.006}{Better to give than to receive: predictive directional measurement of volatility spillovers}. \textit{International Journal of Forecasting}, 28(1), 57--66.",
    'DYc': r"Diebold, F.X., Yilmaz, K. (2014). \href{https://doi.org/10.1016/j.jeconom.2014.04.012}{On the network topology of variance decompositions: measuring the connectedness of financial firms}. \textit{Journal of Econometrics}, 182(1), 119--134.",
    'Draghi': r"Draghi, M. (2012). \href{https://www.ecb.europa.eu/press/key/date/2012/html/sp120726.en.html}{Speech at the Global Investment Conference}, London, 26 July 2012. European Central Bank.",
    'EJR': r"Engle, R., Jondeau, E., Rockinger, M. (2015). \href{https://doi.org/10.1093/rof/rfu012}{Systemic risk in Europe}. \textit{Review of Finance}, 19(1), 145--190.",
    'ESRB': r"European Systemic Risk Board (2026). \href{https://www.esrb.europa.eu/about/background/html/index.en.html}{Background and tasks}. Frankfurt am Main.",
    'FDIC': r"Federal Deposit Insurance Corporation (2023). \href{https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/silicon-valley.html}{Silicon Valley Bank, Santa Clara, CA}. Failed bank information.",
    'Fed': r"Federal Reserve Board (2023). \href{https://www.federalreserve.gov/publications/files/svb-review-20230428.pdf}{Review of the Federal Reserve's supervision and regulation of Silicon Valley Bank}. Washington, DC, April 2023.",
    'FINMA': r"FINMA (2023). \href{https://www.finma.ch/en/news/2023/03/20230319-mm-cs-ubs/}{FINMA approves merger of Credit Suisse and UBS}. Press release, 19 March 2023.",
    'FR': r"Forbes, K.J., Rigobon, R. (2002). \href{https://doi.org/10.1111/0022-1082.00494}{No contagion, only interdependence: measuring stock market comovements}. \textit{Journal of Finance}, 57(5), 2223--2261.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'Granger': r"Granger, C.W.J. (1969). \href{https://doi.org/10.2307/1912791}{Investigating causal relations by econometric models and cross-spectral methods}. \textit{Econometrica}, 37(3), 424--438.",
    'HM': r"Haldane, A.G., May, R.M. (2011). \href{https://doi.org/10.1038/nature09659}{Systemic risk in banking ecosystems}. \textit{Nature}, 469, 351--355.",
    'TENET': r"Härdle, W.K., Wang, W., Yu, L. (2016). \href{https://doi.org/10.1016/j.jeconom.2016.02.013}{TENET: Tail-Event driven NETwork risk}. \textit{Journal of Econometrics}, 192(2), 499--513.",
    'FSB': r"IMF, BIS, FSB (2009). \href{https://www.bis.org/publ/othp07.htm}{Guidance to assess the systemic importance of financial institutions, markets and instruments: initial considerations}. Report to the G-20 Finance Ministers and Central Bank Governors.",
    'KB': r"Koenker, R., Bassett, G. (1978). \href{https://doi.org/10.2307/1913643}{Regression quantiles}. \textit{Econometrica}, 46(1), 33--50.",
    'FRM': r"Mihoci, A., Althof, M., Chen, C.Y.-H., Härdle, W.K. (2020). \href{https://doi.org/10.1108/s0731-905320200000042016}{FRM Financial Risk Meter}. In \textit{The Econometrics of Networks}, Advances in Econometrics, 42, 335--368. Emerald.",
    'CNSM': r"National Committee for Macroprudential Oversight (CNSM) (2026). \href{https://www.cnsmro.ro/}{Comitetul Național pentru Supravegherea Macroprudențială}. Bucharest.",
    'PS': r"Pesaran, H.H., Shin, Y. (1998). \href{https://doi.org/10.1016/S0165-1765(97)00214-0}{Generalized impulse response analysis in linear multivariate models}. \textit{Economics Letters}, 58(1), 17--29.",
    'SV': r"Shleifer, A., Vishny, R. (2011). \href{https://doi.org/10.1257/jep.25.1.29}{Fire sales in finance and macroeconomics}. \textit{Journal of Economic Perspectives}, 25(1), 29--48.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
