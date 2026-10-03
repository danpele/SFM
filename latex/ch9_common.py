r"""
ch9_common.py -- shared helpers of the Chapter 9 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_09/ch9_numbers.json (generate_all_charts.py) and sem9_results.json (seminar9.py);
the clickable citations of Chapter 9 (DOIs checked against Crossref, 3 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_09')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_09'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania (TLV)',
         'snp': 'OMV Petrom (SNP)'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP')
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']
IG = 0.9995          # persistence at or above this value: IGARCH (alpha + beta = 1 on the boundary)


def load():
    with open(os.path.join(QL, 'ch9_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem9_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def put_fit(V, key, s, ig=True):
    """Parameters, robust standard errors, persistence, half-life and volatilities of one fit (summary())."""
    p, se = s['params'], s['se']
    for a, b in [('mu', 'mu'), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]'), ('g', 'gamma[1]'),
                 ('nu', 'nu'), ('eta', 'eta'), ('lam', 'lambda')]:
        if b in p:
            d = 2 if a in ('nu', 'eta') else (4 if a == 'om' else 3)
            V.put(f'{key}.{a}', p[b], d)
            V.put(f'{key}.{a}.se', se[b], d if a != 'om' else 4)
    V.put(f'{key}.ll', s['loglik'], 1)
    V.put(f'{key}.aic', s['aic'], 1)
    V.put(f'{key}.bic', s['bic'], 1)
    V.int(f'{key}.n', s['n'])
    V.raw(f'{key}.y0', s['first'][:4])
    V.put(f'{key}.vs', s['vol_sample'], 1)
    if s['pers'] >= IG and ig:
        V.raw(f'{key}.pers', '⁅1.000⁆')
        V.raw(f'{key}.hl', '--')
        V.raw(f'{key}.vlr', '--')
    else:
        V.put(f'{key}.pers', s['pers'], 3)
        V.put(f'{key}.hl', s['hl'], 0)
        if 'vol_lr' in s:
            V.put(f'{key}.vlr', s['vol_lr'], 1)
            V.put(f'{key}.uv', s['uv'], 2)
    V.put(f'{key}.1mp', 1 - s['pers'], 4)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    R = N['returns']
    V.int('ret.n', R['n'])
    V.put('ret.min', R['min'], 1)
    V.put('ret.max', R['max'], 1)
    put_date(V, 'ret.dmin', R['date_min'])
    put_date(V, 'ret.dmax', R['date_max'])
    for c in ['sd2008', 'sd2020', 'sd_all', 'sd2017']:
        V.put(f'ret.{c}', R[c], 2)
    V.put('ret.ratio', R['sd2020'] / R['sd2017'], 1)
    S = N['sim']
    for k, lab in [('iid', 'i.i.d. Normal(0, 1)'), ('arch', 'ARCH(1)'), ('garch', 'GARCH(1,1)')]:
        V.put(f'sim.{k}.k', S[lab]['kurt'], 2)
        V.put(f'sim.{k}.max', S[lab]['max_abs'], 2)
    V.put('sim.garch.kth', 3 * (1 - 0.98 ** 2) / (1 - 0.98 ** 2 - 2 * 0.1 ** 2), 2)
    for n_ in ['100', '1000']:
        V.put(f'la.{n_}', N['lik_arch1'][n_]['alpha_hat'], 3)
        V.put(f'la.{n_}.se', N['lik_arch1'][n_]['se'], 3)
    for n_ in ['500', '2000']:
        V.put(f'lg.{n_}.a', N['lik_garch'][n_]['a_hat'], 2)
        V.put(f'lg.{n_}.b', N['lik_garch'][n_]['b_hat'], 2)
    for lab, k in [('ARCH(1)', 'a1'), ('ARCH(5)', 'a5'), ('ARCH(10)', 'a10'), ('GARCH(1,1)', 'g11')]:
        d = N['archq'][lab]
        V.put(f'aq.{k}.ll', d['loglik'], 1)
        V.put(f'aq.{k}.aic', d['aic'], 1)
        V.put(f'aq.{k}.bic', d['bic'], 1)
        V.put(f'aq.{k}.sa', d['sum_alpha'], 3)
        V.put(f'aq.{k}.pers', d['pers'], 3)
        V.raw(f'aq.{k}.k', str(d['k']))
    E = N['est']
    st_ = E['step']
    for a, b in [('mu', 'mu'), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]')]:
        d = 4
        V.put(f'step.{a}', st_['params'][b], d)
        V.put(f'step.{a}.se', st_['se'][b], d)
        V.put(f'arch.{a}', E['normal']['params'][b], d)
        V.put(f'arch.{a}.sec', E['normal']['se_classic'][b], d)
        V.put(f'arch.{a}.ser', E['normal']['se'][b], d)
    V.put('step.ll', st_['loglik'], 1)
    put_fit(V, 'en', E['normal'])
    put_fit(V, 'et', E['t'])
    put_fit(V, 'es', E['skewt'])
    V.put('kurt.r', E['kurt_r'], 1)
    V.put('kurt.z', E['kurt_z'], 1)
    V.put('skew.z', E['skew_z'], 2)
    V.put('lr.tn', 2 * (E['t']['loglik'] - E['normal']['loglik']), 1)
    V.put('lr.st', 2 * (E['skewt']['loglik'] - E['t']['loglik']), 1)
    for k, s in N['markets'].items():
        put_fit(V, f'm.{k}', s)
        V.put(f'm.{k}.ab', s['params']['alpha[1]'] + s['params']['beta[1]'], 4)
    V.put('m.btc.1mp', 1 - N['markets']['btc']['pers'], 6)
    for grp in ['vol1', 'vol2']:
        for k, d in N[grp].items():
            V.put(f'v.{k}.last', d['last'], 0)
            V.put(f'v.{k}.med', d['median'], 0)
            for e in ['2008', '2020']:
                if f'peak{e}' in d:
                    V.put(f'v.{k}.p{e}', d[f'peak{e}'], 0)
                    put_date(V, f'v.{k}.d{e}', d[f'date{e}'])
    Q = N['qq']
    V.put('qq.nu', Q['nu'], 1)
    V.put('qq.n001', Q['q']['GARCH(1,1)-N']['q001'], 2)
    V.put('qq.n001th', Q['q']['GARCH(1,1)-N']['th001'], 2)
    V.put('qq.t001', Q['q']['GARCH(1,1)-t']['q001'], 2)
    V.put('qq.t001th', Q['q']['GARCH(1,1)-t']['th001'], 2)
    for k, d in N['nic'].items():
        V.put(f'nic.{k}.ratio', d['ratio_gjr'], 2)
    for k, d in N['asym'].items():
        V.put(f'as.{k}.g', d['gjr_gamma'], 3)
        V.put(f'as.{k}.gt', d['gjr_t'], 1)
        V.put(f'as.{k}.a', d['gjr_alpha'], 3)
        V.put(f'as.{k}.lr', d['lr'], 1)
        V.raw(f'as.{k}.lrp', pv(d['lr_p']))
        V.put(f'as.{k}.eg', d['eg_gamma'], 3)
        V.put(f'as.{k}.egt', d['eg_t'], 1)
        V.put(f'as.{k}.eb', d['eg_beta'], 3)
        V.put(f'as.{k}.sb', d['sb']['joint'], 1)
        V.raw(f'as.{k}.sbp', pv(d['sb']['joint_p']))
        V.raw(f'as.{k}.sbp.e', pv(d['sb']['joint_p']) if d['sb']['joint_p'] < 0.001 else '= ' + pv(d['sb']['joint_p']))
        V.put(f'as.{k}.sbt', d['sb']['sign_t'], 2)
        V.put(f'as.{k}.negt', d['sb']['neg_t'], 2)
        V.put(f'as.{k}.post', d['sb']['pos_t'], 2)
        V.put(f'as.{k}.pers', d['gjr_pers'], 3)
    D = N['diag']
    V.put('dg.r', D['r'][0], 1)
    V.raw('dg.r.p', pv(D['r'][1]))
    V.put('dg.r2', D['r2'][0], 0)
    V.raw('dg.r2.p', pv(D['r2'][1]))
    V.put('dg.lm.r', D['archlm_r'][0], 0)
    for d in ['normal', 't']:
        for c in ['z', 'z2']:
            V.put(f'dg.{d}.{c}', D[d][c][0], 1)
            V.raw(f'dg.{d}.{c}.p', pv(D[d][c][1]))
        V.put(f'dg.{d}.lm', D[d]['archlm'][0], 1)
        V.raw(f'dg.{d}.lm.p', pv(D[d]['archlm'][1]))
    for k, d in N['diag_m'].items():
        V.put(f'dm.{k}.r2', d['r2'][0], 0)
        V.put(f'dm.{k}.z2', d['z2'][0], 1)
        V.raw(f'dm.{k}.z2p', pv(d['z2'][1]))
        V.put(f'dm.{k}.z', d['z'][0], 1)
        V.raw(f'dm.{k}.zp', pv(d['z'][1]))
    A = N['acf_diag']
    V.put('acf.r1', A['a1_1'], 2)
    V.put('acf.r50', A['a1_50'], 3)
    V.put('acf.z1', A['a2_1'], 3)
    V.put('acf.zmax', A['max_a2'], 3)
    V.put('acf.band', A['band'], 3)
    for r in N['ms']:
        key = f"ms.{r['model']}.{r['dist'].replace(' ', '')}"
        V.put(key + '.ll', r['loglik'], 1)
        V.put(key + '.aic', r['aic'], 1)
        V.put(key + '.bic', r['bic'], 1)
        V.put(key + '.daic', r['d_aic'], 1)
        V.put(key + '.dbic', r['d_bic'], 1)
        V.raw(key + '.k', str(r['k']))
    TS = N['term']
    V.put('ts.lr', TS['lr'], 1)
    for d, lab in zip(sorted(k for k in TS if k[:2] == '20'), ['calm', 'covid', 'last']):
        put_date(V, f'ts.{lab}.d', d)
        for c in ['h1', 'h10', 'h22', 'h250', 'avg10', 'avg22', 'avg250']:
            V.put(f'ts.{lab}.{c}', TS[d][c], 1)
        V.put(f'ts.{lab}.s2n', TS[d]['s2next'], 3)
        V.put(f'ts.{lab}.sum10', TS[d]['sum10'], 2)
    V.put('ts.s2bar', TS['s2bar'], 2)
    for k, d in N['fe'].items():
        V.int(f'fe.{k}.n', d['n'])
        for m in ['garch', 'gjr', 'ewma']:
            V.put(f'fe.{k}.q.{m}', d['qlike'][m], 3)
        V.put(f'fe.{k}.dm', d['dm_garch_ewma']['t'], 2)
        V.raw(f'fe.{k}.dmp', pv(d['dm_garch_ewma']['p']))
        V.put(f'fe.{k}.dmj', d['dm_gjr_garch']['t'], 2)
        V.raw(f'fe.{k}.dmjp', pv(d['dm_gjr_garch']['p']))
        V.put(f'fe.{k}.eg', 100 * d['exc_garch'], 2)
        V.put(f'fe.{k}.ee', 100 * d['exc_ewma'], 2)
        V.raw(f'fe.{k}.neg', str(d['nexc_garch']))
        V.raw(f'fe.{k}.nee', str(d['nexc_ewma']))
        V.raw(f'fe.{k}.nexp', str(round(0.01 * d['n'])))
    VF = N['var_fig']
    V.raw('vf.n', str(VF['n']))
    V.raw('vf.eg', str(VF['exc_garch']))
    V.raw('vf.ee', str(VF['exc_ewma']))
    V.raw('vf.exp', str(round(0.01 * VF['n'])))
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refBoll}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refBollT}{\href{https://doi.org/10.2307/1925546}{Bollerslev (1987)}}
\newcommand{\refEB}{\href{https://doi.org/10.1080/07474938608800095}{Engle \& Bollerslev (1986)}}
\newcommand{\refNelson}{\href{https://doi.org/10.2307/2938260}{Nelson (1991)}}
\newcommand{\refGJR}{\href{https://doi.org/10.1111/j.1540-6261.1993.tb05128.x}{Glosten, Jagannathan \& Runkle (1993)}}
\newcommand{\refEN}{\href{https://doi.org/10.1111/j.1540-6261.1993.tb05127.x}{Engle \& Ng (1993)}}
\newcommand{\refBW}{\href{https://doi.org/10.1080/07474939208800229}{Bollerslev \& Wooldridge (1992)}}
\newcommand{\refHansen}{\href{https://doi.org/10.2307/2527081}{Hansen (1994)}}
\newcommand{\refPatton}{\href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Patton (2011)}}
\newcommand{\refHL}{\href{https://doi.org/10.1002/jae.800}{Hansen \& Lunde (2005)}}
\newcommand{\refAB}{\href{https://doi.org/10.2307/2527343}{Andersen \& Bollerslev (1998)}}
\newcommand{\refChristie}{\href{https://doi.org/10.1016/0304-405X(82)90018-6}{Christie (1982)}}
\newcommand{\refEngleN}{\href{https://doi.org/10.1257/0002828041464597}{Engle (2004)}}
\newcommand{\refDM}{\href{https://doi.org/10.1080/07350015.1995.10524599}{Diebold \& Mariano (1995)}}
\newcommand{\refAkaike}{\href{https://doi.org/10.1109/TAC.1974.1100705}{Akaike (1974)}}
\newcommand{\refSchwarz}{\href{https://doi.org/10.1214/aos/1176344136}{Schwarz (1978)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refLjung}{\href{https://doi.org/10.1093/biomet/65.2.297}{Ljung \& Box (1978)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e21020102}{Pele \& Mazurencu-Marinescu-Pele (2019)}}
\newcommand{\refPeleC}{\href{https://doi.org/10.1080/1351847x.2021.1960403}{Pele et al.\ (2023)}}
"""

_BIB = {
    'Akaike': r"Akaike, H. (1974). \href{https://doi.org/10.1109/TAC.1974.1100705}{A new look at the statistical model identification}. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723.",
    'AB': r"Andersen, T.G., Bollerslev, T. (1998). \href{https://doi.org/10.2307/2527343}{Answering the skeptics: yes, standard volatility models do provide accurate forecasts}. \textit{International Economic Review}, 39(4), 885--905.",
    'Boll': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'BollT': r"Bollerslev, T. (1987). \href{https://doi.org/10.2307/1925546}{A conditionally heteroskedastic time series model for speculative prices and rates of return}. \textit{Review of Economics and Statistics}, 69(3), 542--547.",
    'BW': r"Bollerslev, T., Wooldridge, J.M. (1992). \href{https://doi.org/10.1080/07474939208800229}{Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances}. \textit{Econometric Reviews}, 11(2), 143--172.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'Christie': r"Christie, A.A. (1982). \href{https://doi.org/10.1016/0304-405X(82)90018-6}{The stochastic behavior of common stock variances: value, leverage and interest rate effects}. \textit{Journal of Financial Economics}, 10(4), 407--432.",
    'DM': r"Diebold, F.X., Mariano, R.S. (1995). \href{https://doi.org/10.1080/07350015.1995.10524599}{Comparing predictive accuracy}. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263.",
    'Engle': r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \textit{Econometrica}, 50(4), 987--1007.",
    'EngleN': r"Engle, R.F. (2004). \href{https://doi.org/10.1257/0002828041464597}{Risk and volatility: econometric models and financial practice}. \textit{American Economic Review}, 94(3), 405--420.",
    'EB': r"Engle, R.F., Bollerslev, T. (1986). \href{https://doi.org/10.1080/07474938608800095}{Modelling the persistence of conditional variances}. \textit{Econometric Reviews}, 5(1), 1--50.",
    'EN': r"Engle, R.F., Ng, V.K. (1993). \href{https://doi.org/10.1111/j.1540-6261.1993.tb05127.x}{Measuring and testing the impact of news on volatility}. \textit{Journal of Finance}, 48(5), 1749--1778.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GJR': r"Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). \href{https://doi.org/10.1111/j.1540-6261.1993.tb05128.x}{On the relation between the expected value and the volatility of the nominal excess return on stocks}. \textit{Journal of Finance}, 48(5), 1779--1801.",
    'Hansen': r"Hansen, B.E. (1994). \href{https://doi.org/10.2307/2527081}{Autoregressive conditional density estimation}. \textit{International Economic Review}, 35(3), 705--730.",
    'HL': r"Hansen, P.R., Lunde, A. (2005). \href{https://doi.org/10.1002/jae.800}{A forecast comparison of volatility models: does anything beat a GARCH(1,1)?} \textit{Journal of Applied Econometrics}, 20(7), 873--889.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'Ljung': r"Ljung, G.M., Box, G.E.P. (1978). \href{https://doi.org/10.1093/biomet/65.2.297}{On a measure of lack of fit in time series models}. \textit{Biometrika}, 65(2), 297--303.",
    'Nelson': r"Nelson, D.B. (1991). \href{https://doi.org/10.2307/2938260}{Conditional heteroskedasticity in asset returns: a new approach}. \textit{Econometrica}, 59(2), 347--370.",
    'Patton': r"Patton, A.J. (2011). \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Volatility forecast comparison using imperfect volatility proxies}. \textit{Journal of Econometrics}, 160(1), 246--256.",
    'PeleA': r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21020102}{Using high-frequency entropy to forecast Bitcoin's daily value at risk}. \textit{Entropy}, 21(2), 102.",
    'PeleC': r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847x.2021.1960403}{Are cryptos becoming alternative assets?} \textit{European Journal of Finance}, 29(10), 1064--1105.",
    'Schwarz': r"Schwarz, G. (1978). \href{https://doi.org/10.1214/aos/1176344136}{Estimating the dimension of a model}. \textit{Annals of Statistics}, 6(2), 461--464.",
    'Tsay': r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
