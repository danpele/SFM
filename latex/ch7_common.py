r"""
ch7_common.py -- shared helpers of the Chapter 7 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_07/ch7_numbers.json (generate_all_charts.py) and sem7_results.json (seminar7.py);
the clickable citations of Chapter 7 (DOIs checked against Crossref, 2 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_07')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_07'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'eth': 'Ethereum', 'stoxx50': 'Euro Stoxx 50',
         'nikkei': 'Nikkei 225', 'tlv': 'Banca Transilvania (TLV)', 'snp': 'OMV Petrom (SNP)', 'brd': 'BRD',
         'tgn': 'Transgaz (TGN)', 'wig20': 'WIG20', 'bux': 'BUX', 'px': 'PX', 'bvsp': 'Bovespa', 'nifty': 'Nifty 50',
         'bist': 'BIST 100', 'ssec': 'Shanghai Composite', 'kospi': 'KOSPI', 'ipc': 'IPC Mexico'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP', tgn='TGN')
ASSETS = ['sp500', 'dax', 'bet', 'btc']
STOCKS = ['tlv', 'snp', 'brd', 'tgn']
QS = ['2', '5', '10', '20']


def load():
    with open(os.path.join(QL, 'ch7_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem7_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def kp(p):
    """KPSS p-value (tabulated only between 0.01 and 0.10)."""
    if p <= 0.0100001:
        return '$<$\\,⁅0.01⁆'
    if p >= 0.0999999:
        return '$>$\\,⁅0.10⁆'
    return '⁅' + f'{p:.3f}' + '⁆'


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    V.raw('y1', N['end'][:4])
    for k, s in N['scatter'].items():
        V.put(f'sc.{k}.rho', s['rho1'], 3)
        V.int(f'sc.{k}.n', s['n'])
    R = N['rw']
    V.put('rw.mu', 100 * R['mu'], 3)
    V.put('rw.sd', 100 * R['sd'], 2)
    V.put('rw.final', R['final'], 2)
    V.put('rw.smin', min(R['sim_final']), 2)
    V.put('rw.smax', max(R['sim_final']), 2)
    V.put('wn.band', N['wn']['band'], 3)
    V.raw('wn.out', str(N['wn']['outside']))
    for k, a in N['acf'].items():
        V.put(f'acf.{k}.rho1', a['rho'][0], 3)
        V.put(f'acf.{k}.biid', a['band_iid'], 3)
        V.put(f'acf.{k}.brob', a['band_rob1'], 3)
        V.raw(f'acf.{k}.oiid', str(a['out_iid']))
        V.raw(f'acf.{k}.orob', str(a['out_rob']))
    for k, a in N['acf_abs'].items():
        for c in ['abs1', 'abs20', 'abs100', 'ret1']:
            V.put(f'aa.{k}.{c}', a[c], 2)
    for k, t in N['tests'].items():
        V.int(f't.{k}.n', t['n'])
        V.put(f't.{k}.rho1', t['ret']['rho1'], 3)
        for c in ['bp', 'lb', 'rob']:
            V.put(f't.{k}.{c}', t['ret'][c], 1)
            V.raw(f't.{k}.p_{c}', pv(t['ret'][f'p_{c}']))
        V.put(f't.{k}.lbsq', t['sq']['lb'], 0)
        V.put(f't.{k}.robsq', t['sq']['rob'], 1)
        V.put(f't.{k}.rz', t['runs']['z'], 2)
        V.raw(f't.{k}.rp', pv(t['runs']['p']))
        V.int(f't.{k}.runs', t['runs']['runs'])
        V.int(f't.{k}.rmean', round(t['runs']['mean']))
    V.put('lb.crit', N['tests']['sp500']['ret']['crit'], 2)
    for k, u in N['ur'].items():
        for s in ['price', 'ret']:
            for t in ['adf', 'pp']:
                V.put(f'ur.{k}.{s}.{t}', u[s][t]['stat'], 2)
                V.raw(f'ur.{k}.{s}.{t}.p', pv(u[s][t]['p']))
            V.put(f'ur.{k}.{s}.kpss', u[s]['kpss']['stat'], 2)
            V.raw(f'ur.{k}.{s}.kpss.p', kp(u[s]['kpss']['p']))
        V.raw(f'ur.{k}.lags', str(u['price']['adf']['lags']))
    V.put('df.c', N['df_crit']['c'], 2)
    V.put('df.ct', N['df_crit']['ct'], 2)
    S = N['spurious']
    V.put('sp.lev', 100 * S['level']['reject'], 0)
    V.put('sp.r2', S['level']['r2_med'], 2)
    V.put('sp.dif', 100 * S['diff']['reject'], 1)
    V.put('sp.b40', 100 * S['level']['beyond40'], 1)
    for k, v in N['vr'].items():
        V.int(f'vr.{k}.n', v['n'])
        V.raw(f'vr.{k}.y0', v['first'][:4])
        for q in QS:
            a = v['vr'][q]
            V.put(f'vr.{k}.{q}', a['vr'], 3)
            V.put(f'vr.{k}.z{q}', a['z'], 2)
            V.put(f'vr.{k}.zs{q}', a['zs'], 2)
            V.raw(f'vr.{k}.p{q}', pv(a['p_zs']))
        V.put(f'vr.{k}.cd', v['cd']['cd'], 2)
        V.raw(f'vr.{k}.cdp', pv(v['cd']['p']))
        V.put(f'vr.{k}.avr', v['avr']['stat'], 2)
        V.put(f'vr.{k}.k', v['avr']['k'], 1)
        V.raw(f'vr.{k}.avrp', pv(v['avr']['p_boot']))
    V.put('cd.crit', N['cd_crit'], 2)
    for lab, d in N['vr_size'].items():
        for q, r in d.items():
            V.put(f'size.{lab}.{q}.z', 100 * r['z'], 1)
            V.put(f'size.{lab}.{q}.zs', 100 * r['zs'], 1)
    for k, m in N['markets'].items():
        V.put(f'mk.{k}.vr', m['vr'], 2)
        V.put(f'mk.{k}.zs', m['zs'], 2)
        V.put(f'mk.{k}.rho1', m['rho1'], 3)
        V.raw(f'mk.{k}.cdp', pv(m['cd_p']))
    for k, r in N['rolling'].items():
        V.put(f'ro.{k}.rej', 100 * r['share_rej'], 0)
        V.put(f'ro.{k}.pos', 100 * r['share_pos'], 0)
        V.put(f'ro.{k}.neg', 100 * r['share_neg'], 0)
        V.put(f'ro.{k}.min', r['vr_min'], 2)
        V.put(f'ro.{k}.max', r['vr_max'], 2)
        V.raw(f'ro.{k}.ymax', r['date_max'][:4])
        V.raw(f'ro.{k}.ymin', r['date_min'][:4])
        V.raw(f'ro.{k}.n', str(r['n_win']))
    for k, rows in N['sub'].items():
        for i, r in enumerate(rows, 1):
            V.raw(f'sub.{k}.{i}.y0', r['from'][:4])
            V.raw(f'sub.{k}.{i}.y1', r['to'][:4])
            V.put(f'sub.{k}.{i}.rho1', r['rho1'], 3)
            V.put(f'sub.{k}.{i}.vr', r['vr'], 2)
            V.put(f'sub.{k}.{i}.zs', r['zs'], 2)
            V.raw(f'sub.{k}.{i}.cdp', pv(r['cd_p']))
            V.int(f'sub.{k}.{i}.n', r['n'])
    for k, c in N['calendar'].items():
        for i, m in enumerate(c['means']):
            V.put(f'cal.{k}.d{i}', m, 3)
        V.raw(f'cal.{k}.wp', pv(c['wald_p'], 2))
        V.put(f'cal.{k}.jan', c['jan_mean'], 3)
        V.put(f'cal.{k}.oth', c['other_mean'], 3)
        V.put(f'cal.{k}.jt', c['jan_t'], 2)
        V.raw(f'cal.{k}.jp', pv(c['jan_p'], 2))
    U = N['upgrade']
    for p in ['before', 'after']:
        for c in ['rho1']:
            V.put(f'up.{p}.{c}', U[p][c], 3)
        V.put(f'up.{p}.vr', U[p]['vr'], 2)
        V.put(f'up.{p}.zs', U[p]['zs'], 2)
        V.put(f'up.{p}.se', U[p]['se'], 2)
        V.raw(f'up.{p}.cdp', pv(U[p]['cd_p']))
        V.int(f'up.{p}.n', U[p]['n'])
    V.put('up.z', U['z_diff'], 2)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 2 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFamaA}{\href{https://doi.org/10.1111/j.1540-6261.1970.tb00518.x}{Fama (1970)}}
\newcommand{\refFamaB}{\href{https://doi.org/10.1111/j.1540-6261.1991.tb04636.x}{Fama (1991)}}
\newcommand{\refFamaC}{\href{https://doi.org/10.1086/294743}{Fama (1965)}}
\newcommand{\refSamuelson}{\href{https://doi.org/10.1142/9789814566926_0002}{Samuelson (1965)}}
\newcommand{\refBachelier}{\href{https://doi.org/10.24033/asens.476}{Bachelier (1900)}}
\newcommand{\refKendall}{\href{https://doi.org/10.2307/2980947}{Kendall (1953)}}
\newcommand{\refRoberts}{\href{https://doi.org/10.1111/j.1540-6261.1959.tb00481.x}{Roberts (1959)}}
\newcommand{\refLM}{\href{https://doi.org/10.1093/rfs/1.1.41}{Lo \& MacKinlay (1988)}}
\newcommand{\refLo}{\href{https://doi.org/10.3905/jpm.2004.442611}{Lo (2004)}}
\newcommand{\refLoBook}{\href{https://doi.org/10.1515/9781400887767}{Lo (2017)}}
\newcommand{\refCLM}{\href{https://doi.org/10.1515/9781400830213}{Campbell, Lo \& MacKinlay (1997)}}
\newcommand{\refCD}{\href{https://doi.org/10.1016/0304-4076(93)90051-6}{Chow \& Denning (1993)}}
\newcommand{\refChoi}{\href{https://doi.org/10.1002/(SICI)1099-1255(199905/06)14:3<293::AID-JAE503>3.0.CO;2-5}{Choi (1999)}}
\newcommand{\refKimB}{\href{https://doi.org/10.1016/j.frl.2009.04.003}{Kim (2009)}}
\newcommand{\refAndrews}{\href{https://doi.org/10.2307/2938229}{Andrews (1991)}}
\newcommand{\refLjung}{\href{https://doi.org/10.1093/biomet/65.2.297}{Ljung \& Box (1978)}}
\newcommand{\refBP}{\href{https://doi.org/10.1080/01621459.1970.10481180}{Box \& Pierce (1970)}}
\newcommand{\refWW}{\href{https://doi.org/10.1214/aoms/1177731909}{Wald \& Wolfowitz (1940)}}
\newcommand{\refEL}{\href{https://doi.org/10.1016/j.jeconom.2009.03.001}{Escanciano \& Lobato (2009)}}
\newcommand{\refDF}{\href{https://doi.org/10.1080/01621459.1979.10482531}{Dickey \& Fuller (1979)}}
\newcommand{\refSD}{\href{https://doi.org/10.1093/biomet/71.3.599}{Said \& Dickey (1984)}}
\newcommand{\refPP}{\href{https://doi.org/10.1093/biomet/75.2.335}{Phillips \& Perron (1988)}}
\newcommand{\refKPSS}{\href{https://doi.org/10.1016/0304-4076(92)90104-Y}{Kwiatkowski et al.\ (1992)}}
\newcommand{\refMacKinnon}{\href{https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<601::AID-JAE417>3.0.CO;2-T}{MacKinnon (1996)}}
\newcommand{\refNW}{\href{https://doi.org/10.2307/1913610}{Newey \& West (1987)}}
\newcommand{\refGN}{\href{https://doi.org/10.1016/0304-4076(74)90034-7}{Granger \& Newbold (1974)}}
\newcommand{\refUrquhart}{\href{https://doi.org/10.1016/j.econlet.2016.09.019}{Urquhart (2016)}}
\newcommand{\refUM}{\href{https://doi.org/10.1016/j.irfa.2016.06.011}{Urquhart \& McGroarty (2016)}}
\newcommand{\refKSL}{\href{https://doi.org/10.1016/j.jempfin.2011.08.002}{Kim, Shamsuddin \& Lim (2011)}}
\newcommand{\refLB}{\href{https://doi.org/10.1111/j.1467-6419.2009.00611.x}{Lim \& Brooks (2011)}}
\newcommand{\refGS}{\href{https://ideas.repec.org/a/aea/aecrev/v70y1980i3p393-408.html}{Grossman \& Stiglitz (1980)}}
\newcommand{\refFrench}{\href{https://doi.org/10.1016/0304-405X(80)90021-5}{French (1980)}}
\newcommand{\refRK}{\href{https://doi.org/10.1016/0304-405X(76)90028-3}{Rozeff \& Kinney (1976)}}
\newcommand{\refAriel}{\href{https://doi.org/10.1016/0304-405X(87)90066-3}{Ariel (1987)}}
\newcommand{\refMP}{\href{https://doi.org/10.1111/jofi.12365}{McLean \& Pontiff (2016)}}
\newcommand{\refHLZ}{\href{https://doi.org/10.1093/rfs/hhv059}{Harvey, Liu \& Zhu (2016)}}
\newcommand{\refMacKinlay}{\href{https://ideas.repec.org/a/aea/jeclit/v35y1997i1p13-39.html}{MacKinlay (1997)}}
\newcommand{\refJensen}{\href{https://doi.org/10.1111/j.1540-6261.1968.tb00815.x}{Jensen (1968)}}
\newcommand{\refFF}{\href{https://doi.org/10.1111/j.1540-6261.2010.01598.x}{Fama \& French (2010)}}
\newcommand{\refMalkiel}{\href{https://doi.org/10.1257/089533003321164958}{Malkiel (2003)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Pele \& Mazurencu-Marinescu (2012)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.2139/ssrn.7157232}{Tak, Schlamp, Osterrieder \& Pele (2026)}}
\newcommand{\refPeleC}{\href{https://doi.org/10.2139/ssrn.5176758}{Lin, Găman, Wang \& Pele (2025)}}
\newcommand{\refFTSE}{\href{https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2019/ftse-russell-announces-results-country-classification-review-equities-and-fixed-income}{FTSE Russell (2019)}}
"""

_BIB = {
    'Andrews': r"Andrews, D.W.K. (1991). \href{https://doi.org/10.2307/2938229}{Heteroskedasticity and autocorrelation consistent covariance matrix estimation}. \textit{Econometrica}, 59(3), 817--858.",
    'Ariel': r"Ariel, R.A. (1987). \href{https://doi.org/10.1016/0304-405X(87)90066-3}{A monthly effect in stock returns}. \textit{Journal of Financial Economics}, 18(1), 161--174.",
    'Bachelier': r"Bachelier, L. (1900). \href{https://doi.org/10.24033/asens.476}{Théorie de la spéculation}. \textit{Annales scientifiques de l'École normale supérieure}, 17, 21--86.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'BP': r"Box, G.E.P., Pierce, D.A. (1970). \href{https://doi.org/10.1080/01621459.1970.10481180}{Distribution of residual autocorrelations in autoregressive-integrated moving average time series models}. \textit{Journal of the American Statistical Association}, 65(332), 1509--1526.",
    'CLM': r"Campbell, J.Y., Lo, A.W., MacKinlay, A.C. (1997). \href{https://doi.org/10.1515/9781400830213}{\textit{The Econometrics of Financial Markets}}. Princeton University Press.",
    'Choi': r"Choi, I. (1999). \href{https://doi.org/10.1002/(SICI)1099-1255(199905/06)14:3<293::AID-JAE503>3.0.CO;2-5}{Testing the random walk hypothesis for real exchange rates}. \textit{Journal of Applied Econometrics}, 14(3), 293--308.",
    'CD': r"Chow, K.V., Denning, K.C. (1993). \href{https://doi.org/10.1016/0304-4076(93)90051-6}{A simple multiple variance ratio test}. \textit{Journal of Econometrics}, 58(3), 385--401.",
    'DF': r"Dickey, D.A., Fuller, W.A. (1979). \href{https://doi.org/10.1080/01621459.1979.10482531}{Distribution of the estimators for autoregressive time series with a unit root}. \textit{Journal of the American Statistical Association}, 74(366), 427--431.",
    'EL': r"Escanciano, J.C., Lobato, I.N. (2009). \href{https://doi.org/10.1016/j.jeconom.2009.03.001}{An automatic Portmanteau test for serial correlation}. \textit{Journal of Econometrics}, 151(2), 140--149.",
    'FamaC': r"Fama, E.F. (1965). \href{https://doi.org/10.1086/294743}{The behavior of stock-market prices}. \textit{Journal of Business}, 38(1), 34--105.",
    'FamaA': r"Fama, E.F. (1970). \href{https://doi.org/10.1111/j.1540-6261.1970.tb00518.x}{Efficient capital markets: a review of theory and empirical work}. \textit{Journal of Finance}, 25(2), 383--417.",
    'FamaB': r"Fama, E.F. (1991). \href{https://doi.org/10.1111/j.1540-6261.1991.tb04636.x}{Efficient capital markets: II}. \textit{Journal of Finance}, 46(5), 1575--1617.",
    'FF': r"Fama, E.F., French, K.R. (2010). \href{https://doi.org/10.1111/j.1540-6261.2010.01598.x}{Luck versus skill in the cross-section of mutual fund returns}. \textit{Journal of Finance}, 65(5), 1915--1947.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'French': r"French, K.R. (1980). \href{https://doi.org/10.1016/0304-405X(80)90021-5}{Stock returns and the weekend effect}. \textit{Journal of Financial Economics}, 8(1), 55--69.",
    'FTSE': r"FTSE Russell (2019). \href{https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2019/ftse-russell-announces-results-country-classification-review-equities-and-fixed-income}{FTSE Russell announces results of country classification review}. Press release, London Stock Exchange Group.",
    'GN': r"Granger, C.W.J., Newbold, P. (1974). \href{https://doi.org/10.1016/0304-4076(74)90034-7}{Spurious regressions in econometrics}. \textit{Journal of Econometrics}, 2(2), 111--120.",
    'GS': r"Grossman, S.J., Stiglitz, J.E. (1980). \href{https://ideas.repec.org/a/aea/aecrev/v70y1980i3p393-408.html}{On the impossibility of informationally efficient markets}. \textit{American Economic Review}, 70(3), 393--408.",
    'HLZ': r"Harvey, C.R., Liu, Y., Zhu, H. (2016). \href{https://doi.org/10.1093/rfs/hhv059}{\ldots and the cross-section of expected returns}. \textit{Review of Financial Studies}, 29(1), 5--68.",
    'Jensen': r"Jensen, M.C. (1968). \href{https://doi.org/10.1111/j.1540-6261.1968.tb00815.x}{The performance of mutual funds in the period 1945--1964}. \textit{Journal of Finance}, 23(2), 389--416.",
    'Kendall': r"Kendall, M.G. (1953). \href{https://doi.org/10.2307/2980947}{The analysis of economic time-series, Part I: prices}. \textit{Journal of the Royal Statistical Society, Series A}, 116(1), 11--34.",
    'KimB': r"Kim, J.H. (2009). \href{https://doi.org/10.1016/j.frl.2009.04.003}{Automatic variance ratio test under conditional heteroskedasticity}. \textit{Finance Research Letters}, 6(3), 179--185.",
    'KSL': r"Kim, J.H., Shamsuddin, A., Lim, K.-P. (2011). \href{https://doi.org/10.1016/j.jempfin.2011.08.002}{Stock return predictability and the adaptive markets hypothesis: evidence from century-long U.S. data}. \textit{Journal of Empirical Finance}, 18(5), 868--879.",
    'KPSS': r"Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., Shin, Y. (1992). \href{https://doi.org/10.1016/0304-4076(92)90104-Y}{Testing the null hypothesis of stationarity against the alternative of a unit root}. \textit{Journal of Econometrics}, 54(1--3), 159--178.",
    'LB': r"Lim, K.-P., Brooks, R. (2011). \href{https://doi.org/10.1111/j.1467-6419.2009.00611.x}{The evolution of stock market efficiency over time: a survey of the empirical literature}. \textit{Journal of Economic Surveys}, 25(1), 69--108.",
    'PeleC': r"Lin, Y., Găman, A., Wang, Y., Pele, D.T. (2025). \href{https://doi.org/10.2139/ssrn.5176758}{Market responses to Ethereum development milestones: an event study approach}. SSRN Working Paper 5176758.",
    'Ljung': r"Ljung, G.M., Box, G.E.P. (1978). \href{https://doi.org/10.1093/biomet/65.2.297}{On a measure of lack of fit in time series models}. \textit{Biometrika}, 65(2), 297--303.",
    'Lo': r"Lo, A.W. (2004). \href{https://doi.org/10.3905/jpm.2004.442611}{The adaptive markets hypothesis}. \textit{Journal of Portfolio Management}, 30(5), 15--29.",
    'LoBook': r"Lo, A.W. (2017). \href{https://doi.org/10.1515/9781400887767}{\textit{Adaptive Markets: Financial Evolution at the Speed of Thought}}. Princeton University Press.",
    'LM': r"Lo, A.W., MacKinlay, A.C. (1988). \href{https://doi.org/10.1093/rfs/1.1.41}{Stock market prices do not follow random walks: evidence from a simple specification test}. \textit{Review of Financial Studies}, 1(1), 41--66.",
    'MacKinlay': r"MacKinlay, A.C. (1997). \href{https://ideas.repec.org/a/aea/jeclit/v35y1997i1p13-39.html}{Event studies in economics and finance}. \textit{Journal of Economic Literature}, 35(1), 13--39.",
    'MacKinnon': r"MacKinnon, J.G. (1996). \href{https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<601::AID-JAE417>3.0.CO;2-T}{Numerical distribution functions for unit root and cointegration tests}. \textit{Journal of Applied Econometrics}, 11(6), 601--618.",
    'Malkiel': r"Malkiel, B.G. (2003). \href{https://doi.org/10.1257/089533003321164958}{The efficient market hypothesis and its critics}. \textit{Journal of Economic Perspectives}, 17(1), 59--82.",
    'MP': r"McLean, R.D., Pontiff, J. (2016). \href{https://doi.org/10.1111/jofi.12365}{Does academic research destroy stock return predictability?} \textit{Journal of Finance}, 71(1), 5--32.",
    'NW': r"Newey, W.K., West, K.D. (1987). \href{https://doi.org/10.2307/1913610}{A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix}. \textit{Econometrica}, 55(3), 703--708.",
    'PeleA': r"Pele, D.T., Mazurencu-Marinescu, M. (2012). \href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Modelling stock market crashes: the case of Bucharest Stock Exchange}. \textit{Procedia -- Social and Behavioral Sciences}, 58, 533--542.",
    'PP': r"Phillips, P.C.B., Perron, P. (1988). \href{https://doi.org/10.1093/biomet/75.2.335}{Testing for a unit root in time series regression}. \textit{Biometrika}, 75(2), 335--346.",
    'Roberts': r"Roberts, H.V. (1959). \href{https://doi.org/10.1111/j.1540-6261.1959.tb00481.x}{Stock-market ``patterns'' and financial analysis: methodological suggestions}. \textit{Journal of Finance}, 14(1), 1--10.",
    'RK': r"Rozeff, M.S., Kinney, W.R. (1976). \href{https://doi.org/10.1016/0304-405X(76)90028-3}{Capital market seasonality: the case of stock returns}. \textit{Journal of Financial Economics}, 3(4), 379--402.",
    'SD': r"Said, S.E., Dickey, D.A. (1984). \href{https://doi.org/10.1093/biomet/71.3.599}{Testing for unit roots in autoregressive-moving average models of unknown order}. \textit{Biometrika}, 71(3), 599--607.",
    'Samuelson': r"Samuelson, P.A. (1965). \href{https://doi.org/10.1142/9789814566926_0002}{Proof that properly anticipated prices fluctuate randomly}. \textit{Industrial Management Review}, 6(2), 41--49; reprinted in the \textit{World Scientific Handbook in Financial Economics Series}, 25--38.",
    'PeleB': r"Tak, A., Schlamp, S., Osterrieder, J., Pele, D.T. (2026). \href{https://doi.org/10.2139/ssrn.7157232}{Racing the news: nanosecond market efficiency at Eurex and CME}. SSRN Working Paper 7157232.",
    'Tsay': r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    'Urquhart': r"Urquhart, A. (2016). \href{https://doi.org/10.1016/j.econlet.2016.09.019}{The inefficiency of Bitcoin}. \textit{Economics Letters}, 148, 80--82.",
    'UM': r"Urquhart, A., McGroarty, F. (2016). \href{https://doi.org/10.1016/j.irfa.2016.06.011}{Are stock markets really efficient? Evidence of the adaptive market hypothesis}. \textit{International Review of Financial Analysis}, 47, 39--49.",
    'WW': r"Wald, A., Wolfowitz, J. (1940). \href{https://doi.org/10.1214/aoms/1177731909}{On a test whether two samples are from the same population}. \textit{Annals of Mathematical Statistics}, 11(2), 147--162.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
