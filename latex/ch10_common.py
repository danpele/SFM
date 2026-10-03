r"""
ch10_common.py -- shared helpers of the Chapter 10 generators (lecture and seminar), SFM
========================================================================================
Numbers from Quantlets/Ch_10/ch10_numbers.json (generate_all_charts.py) and sem10_results.json (seminar10.py);
the clickable citations of Chapter 10 (DOIs checked against Crossref, BIS pages checked, 3 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_10')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_10'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania (TLV)',
         'snp': 'OMV Petrom (SNP)', 'brd': 'BRD'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP')
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']
BT = ['sp500', 'dax', 'bet', 'btc']
METHODS = ['HS', 'Normal', 'GARCH-t', 'FHS']
MKEY = {'HS': 'hs', 'Normal': 'n', 'Student-t': 't', 'Cornish-Fisher': 'cf', 'EVT': 'evt', 'GARCH-t': 'g', 'FHS': 'f'}


def load():
    with open(os.path.join(QL, 'ch10_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem10_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    d = N['def']
    V.int('def.n', d['n'])
    V.put('def.var1', d['var1'], 2)
    V.put('def.q25', -d['q25'], 2)
    V.put('def.es25', d['es25'], 2)
    V.raw('def.ntail', str(d['n_tail1']))
    V.put('def.min', d['min'], 1)
    put_date(V, 'def.dmin', d['date_min'])
    for k, m in N['methods'].items():
        V.int(f'm.{k}.n', m['n'])
        V.raw(f'm.{k}.y0', m['first'][:4])
        V.put(f'm.{k}.mu', m['mu'], 3)
        V.put(f'm.{k}.sd', m['sd'], 2)
        V.put(f'm.{k}.skew', m['skew'], 2)
        V.put(f'm.{k}.k', m['exkurt'], 1)
        V.put(f'm.{k}.nu', m['nu'], 2)
        V.put(f'm.{k}.xi', m['xi'], 2)
        for meth, key in MKEY.items():
            V.put(f'm.{k}.{key}.v', m[meth]['var'], 2)
            V.put(f'm.{k}.{key}.e', m[meth]['es'], 2)
        V.put(f'm.{k}.cfz', m['cf_z'], 2)
        f = m['pot']
        V.put(f'm.{k}.u', f['u'], 2)
        V.raw(f'm.{k}.Nu', str(f['Nu']))
        V.put(f'm.{k}.beta', f['beta'], 3)
        V.put(f'm.{k}.xi3', f['xi'], 3)
        c = m['cond']
        V.put(f'c.{k}.mu', c['mu'], 3)
        V.put(f'c.{k}.sig', c['sigma'], 3)
        V.put(f'c.{k}.nu', c['nu'], 2)
        V.put(f'c.{k}.q', c['q_t'], 3)
        V.put(f'c.{k}.qa', -c['q_t'], 3)
        V.put(f'c.{k}.est', c['es_t'], 3)
        V.put(f'c.{k}.qz', c['q_z'], 3)
        V.put(f'c.{k}.qza', -c['q_z'], 3)
        V.put(f'c.{k}.esz', c['es_z'], 3)
        V.put(f'c.{k}.vg', c['var_g'], 2)
        V.put(f'c.{k}.eg', c['es_g'], 2)
        V.put(f'c.{k}.vf', c['var_f'], 2)
        V.put(f'c.{k}.ef', c['es_f'], 2)
        V.put(f'c.{k}.a', c['alpha'], 3)
        V.put(f'c.{k}.b', c['beta'], 3)
        put_date(V, f'c.{k}.last', c['last'])
    for lab, d in N['precision'].items():
        V.int(f'pr.{lab}.n', d['n'])
        for c in ['var', 'es', 'var_lo', 'var_hi', 'es_lo', 'es_hi', 'var_se', 'es_se']:
            V.put(f'pr.{lab}.{c}', d[c], 2)
    H = N['horizon']
    for c in ['var1', 'var_sqrt', 'var_h', 'var_n1', 'var_nh']:
        V.put(f'hz.{c}', H[c], 2)
    for row in N['traffic']:
        V.put(f'tl.{row["x"]}.p', 100 * row['p'], 2)
        V.put(f'tl.{row["x"]}.c', 100 * row['cum'], 2)
    for k, b in N['bt'].items():
        put_date(V, f'bt.{k}.start', b['start'])
        for m in METHODS:
            r = b[m]
            key = f'bt.{k}.{MKEY[m]}'
            V.int(f'{key}.n', r['n'])
            V.raw(f'{key}.x', str(r['x']))
            V.put(f'{key}.exp', r['exp'], 0)
            V.put(f'{key}.rate', 100 * r['rate'], 2)
            V.put(f'{key}.luc', r['lr_uc'], 2)
            V.raw(f'{key}.puc', pv(r['p_uc']))
            V.put(f'{key}.lind', r['lr_ind'], 2)
            V.raw(f'{key}.pind', pv(r['p_ind']))
            V.put(f'{key}.lcc', r['lr_cc'], 2)
            V.raw(f'{key}.pcc', pv(r['p_cc']))
            V.raw(f'{key}.n11', str(r['n11']))
            V.put(f'{key}.p01', 100 * r['p01'], 2)
            V.put(f'{key}.p11', 100 * r['p11'], 1)
            V.raw(f'{key}.max', str(r['max250']))
            V.put(f'{key}.z2', r['z2'], 2)
            V.raw(f'{key}.pz2', pv(r['p_z2']))
            V.put(f'{key}.red', 100 * r['zones']['red'] / r['n_windows'], 0)
            V.put(f'{key}.green', 100 * r['zones']['green'] / r['n_windows'], 0)
            V.put(f'{key}.mv', r['mean_var'], 2)
            V.put(f'{key}.ql', 100 * r['qloss'], 2)
            V.raw(f'{key}.l250', str(r['last250']))
            for e in ('2008', '2020'):
                s = r['stress'][e]
                V.raw(f'{key}.s{e}', str(s['x']))
                V.raw(f'{key}.s{e}n', str(s['n']))
                V.put(f'{key}.s{e}e', s['exp'], 1)
    P = N['port']
    V.int('p.n', P['n'])
    V.put('p.mu1', P['mu'][0], 3)
    V.put('p.mu2', P['mu'][1], 3)
    V.put('p.sd1', P['sd'][0], 2)
    V.put('p.sd2', P['sd'][1], 2)
    V.put('p.corr', P['corr'], 2)
    V.put('p.mp', P['mp'], 3)
    V.put('p.sp', P['sp'], 3)
    V.put('p.sp2', P['sp'] ** 2, 3)
    for c in ['var_n', 'es_n', 'var_hs', 'es_hs', 'var_n_sum', 'var_hs_sum', 'var_gauss', 'es_gauss', 'var_t', 'es_t']:
        V.put(f'p.{c}', P[c], 2)
    V.put('p.vn1', P['var_n_single'][0], 2)
    V.put('p.vn2', P['var_n_single'][1], 2)
    V.put('p.vh1', P['var_hs_single'][0], 2)
    V.put('p.vh2', P['var_hs_single'][1], 2)
    V.put('p.ben', 100 * (1 - P['var_n'] / P['var_n_sum']), 0)
    V.put('p.benh', 100 * (1 - P['var_hs'] / P['var_hs_sum']), 0)
    V.put('p.c1', 100 * P['contrib'][0] / P['sp'], 0)
    V.put('p.c2', 100 * P['contrib'][1] / P['sp'], 0)
    for e in ('2008', '2020', 'calm'):
        V.put(f'p.corr{e}', P[f'corr_{e}'], 2)
    V.put('p.td5', P['tail_emp']['0.05'], 2)
    V.put('p.td1', P['tail_emp']['0.01'], 2)
    for c in ['emp', 'gauss', 't', 'ind']:
        V.put(f'p.j.{c}', 100 * P[f'joint1_{c}'], 2)
    V.put('p.jratio', P['joint1_emp'] / P['joint1_gauss'], 1)
    C = P['copula']
    V.put('cop.tau', C['tau'], 3)
    V.put('cop.rho', C['rho'], 3)
    V.put('cop.nu', C['nu'], 2)
    V.put('cop.llt', C['ll_t'], 1)
    V.put('cop.llg', C['ll_g'], 1)
    V.put('cop.lr', 2 * (C['ll_t'] - C['ll_g']), 1)
    V.put('cop.lam', C['lambda_t'], 3)
    V.put('cop.sp', C['spearman'], 2)
    R = N['corr']
    V.put('rc.max', R['max'], 2)
    put_date(V, 'rc.dmax', R['date_max'])
    V.put('rc.min', R['min'], 2)
    V.put('rc.last', R['last'], 2)
    F = N['copula_fig']
    for c in ['corner_data', 'corner_gauss', 'corner_t']:
        V.raw(f'cf.{c}', str(F[c]))
    V.put('cf.exp', F['expected_ind'], 0)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, BIS pages verified, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refArtzner}{\href{https://doi.org/10.1111/1467-9965.00068}{Artzner et al.\ (1999)}}
\newcommand{\refAT}{\href{https://doi.org/10.1016/S0378-4266(02)00283-2}{Acerbi \& Tasche (2002)}}
\newcommand{\refAS}{\href{https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b}{Acerbi \& Székely (2014)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refChris}{\href{https://doi.org/10.2307/2527341}{Christoffersen (1998)}}
\newcommand{\refGneiting}{\href{https://doi.org/10.1198/jasa.2011.r10138}{Gneiting (2011)}}
\newcommand{\refFZ}{\href{https://doi.org/10.1214/16-AOS1439}{Fissler \& Ziegel (2016)}}
\newcommand{\refHW}{\href{https://doi.org/10.21314/JOR.1998.001}{Hull \& White (1998)}}
\newcommand{\refBGV}{\href{https://doi.org/10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S}{Barone-Adesi, Giannopoulos \& Vosper (1999)}}
\newcommand{\refMF}{\href{https://doi.org/10.1016/S0927-5398(00)00012-8}{McNeil \& Frey (2000)}}
\newcommand{\refCF}{\href{https://doi.org/10.2307/1400905}{Cornish \& Fisher (1938)}}
\newcommand{\refEMS}{\href{https://doi.org/10.1017/CBO9780511615337.008}{Embrechts, McNeil \& Straumann (2002)}}
\newcommand{\refDM}{\href{https://doi.org/10.1111/j.1751-5823.2005.tb00254.x}{Demarta \& McNeil (2005)}}
\newcommand{\refBO}{\href{https://doi.org/10.1111/1540-6261.00455}{Berkowitz \& O'Brien (2002)}}
\newcommand{\refCAViaR}{\href{https://doi.org/10.1198/073500104000000370}{Engle \& Manganelli (2004)}}
\newcommand{\refDJSS}{\href{https://doi.org/10.1016/j.jeconom.2012.08.011}{Daníelsson et al.\ (2013)}}
\newcommand{\refRM}{\href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{J.P.\ Morgan/Reuters (1996)}}
\newcommand{\refBCBSa}{\href{https://www.bis.org/publ/bcbs24.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (1996a)}}
\newcommand{\refBCBSb}{\href{https://www.bis.org/publ/bcbs22.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (1996b)}}
\newcommand{\refFRTB}{\href{https://www.bis.org/bcbs/publ/d457.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (2019)}}
\newcommand{\refMAR}{\href{https://www.bis.org/basel_framework/chapter/MAR/33.htm}{⟦Basel Framework||Cadrul Basel⟧, MAR33}}
\newcommand{\refMARb}{\href{https://www.bis.org/basel_framework/chapter/MAR/32.htm}{⟦Basel Framework||Cadrul Basel⟧, MAR32}}
\newcommand{\refBCBSh}{\href{https://www.bis.org/bcbs/history.htm}{⟦history of the Basel Committee||istoria Comitetului de la Basel⟧}}
\newcommand{\refQRM}{\href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{McNeil, Frey \& Embrechts (2015)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refBoll}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e21020102}{Pele \& Mazurencu-Marinescu-Pele (2019)}}
"""

_BIB = {
    'AT': r"Acerbi, C., Tasche, D. (2002). \href{https://doi.org/10.1016/S0378-4266(02)00283-2}{On the coherence of expected shortfall}. \textit{Journal of Banking \& Finance}, 26(7), 1487--1503.",
    'AS': r"Acerbi, C., Székely, B. (2014). \href{https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b}{Backtesting expected shortfall}. MSCI Research; published in \textit{Risk}, December 2014.",
    'Artzner': r"Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). \href{https://doi.org/10.1111/1467-9965.00068}{Coherent measures of risk}. \textit{Mathematical Finance}, 9(3), 203--228.",
    'BGV': r"Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999). \href{https://doi.org/10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S}{VaR without correlations for portfolios of derivative securities}. \textit{Journal of Futures Markets}, 19(5), 583--602.",
    'BCBSa': r"Basel Committee on Banking Supervision (1996a). \href{https://www.bis.org/publ/bcbs24.htm}{Amendment to the capital accord to incorporate market risks}. Bank for International Settlements.",
    'BCBSb': r"Basel Committee on Banking Supervision (1996b). \href{https://www.bis.org/publ/bcbs22.htm}{Supervisory framework for the use of ``backtesting'' in conjunction with the internal models approach to market risk capital requirements}. Bank for International Settlements.",
    'FRTB': r"Basel Committee on Banking Supervision (2019). \href{https://www.bis.org/bcbs/publ/d457.htm}{Minimum capital requirements for market risk}. Bank for International Settlements.",
    'BO': r"Berkowitz, J., O'Brien, J. (2002). \href{https://doi.org/10.1111/1540-6261.00455}{How accurate are value-at-risk models at commercial banks?} \textit{Journal of Finance}, 57(3), 1093--1111.",
    'Boll': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'Chris': r"Christoffersen, P.F. (1998). \href{https://doi.org/10.2307/2527341}{Evaluating interval forecasts}. \textit{International Economic Review}, 39(4), 841--862.",
    'CF': r"Cornish, E.A., Fisher, R.A. (1938). \href{https://doi.org/10.2307/1400905}{Moments and cumulants in the specification of distributions}. \textit{Revue de l'Institut International de Statistique}, 5(4), 307--320.",
    'DJSS': r"Daníelsson, J., Jorgensen, B.N., Samorodnitsky, G., Sarma, M., de Vries, C.G. (2013). \href{https://doi.org/10.1016/j.jeconom.2012.08.011}{Fat tails, VaR and subadditivity}. \textit{Journal of Econometrics}, 172(2), 283--291.",
    'DM': r"Demarta, S., McNeil, A.J. (2005). \href{https://doi.org/10.1111/j.1751-5823.2005.tb00254.x}{The t copula and related copulas}. \textit{International Statistical Review}, 73(1), 111--129.",
    'EMS': r"Embrechts, P., McNeil, A.J., Straumann, D. (2002). \href{https://doi.org/10.1017/CBO9780511615337.008}{Correlation and dependence in risk management: properties and pitfalls}. In M. Dempster (ed.), \textit{Risk Management: Value at Risk and Beyond}, 176--223. Cambridge University Press.",
    'CAViaR': r"Engle, R.F., Manganelli, S. (2004). \href{https://doi.org/10.1198/073500104000000370}{CAViaR: conditional autoregressive value at risk by regression quantiles}. \textit{Journal of Business \& Economic Statistics}, 22(4), 367--381.",
    'FZ': r"Fissler, T., Ziegel, J.F. (2016). \href{https://doi.org/10.1214/16-AOS1439}{Higher order elicitability and Osband's principle}. \textit{Annals of Statistics}, 44(4), 1680--1707.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'Gneiting': r"Gneiting, T. (2011). \href{https://doi.org/10.1198/jasa.2011.r10138}{Making and evaluating point forecasts}. \textit{Journal of the American Statistical Association}, 106(494), 746--762.",
    'HW': r"Hull, J., White, A. (1998). \href{https://doi.org/10.21314/JOR.1998.001}{Incorporating volatility updating into the historical simulation method for value-at-risk}. \textit{Journal of Risk}, 1(1), 5--19.",
    'RM': r"J.P. Morgan/Reuters (1996). \href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{\textit{RiskMetrics -- Technical Document}}, 4th ed. New York.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'MF': r"McNeil, A.J., Frey, R. (2000). \href{https://doi.org/10.1016/S0927-5398(00)00012-8}{Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach}. \textit{Journal of Empirical Finance}, 7(3--4), 271--300.",
    'QRM': r"McNeil, A.J., Frey, R., Embrechts, P. (2015). \href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{\textit{Quantitative Risk Management: Concepts, Techniques and Tools}}, revised ed. Princeton University Press.",
    'PeleA': r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21020102}{Using high-frequency entropy to forecast Bitcoin's daily value at risk}. \textit{Entropy}, 21(2), 102.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
