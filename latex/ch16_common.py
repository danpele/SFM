r"""
ch16_common.py -- shared helpers of the Chapter 16 generators (review: lecture and seminar), SFM
================================================================================================
Numbers:
  * the key empirical facts of Chapters 0--15, read from the numbers files of each chapter
    (Quantlets/Ch_NN/chN_numbers.json), so the review quotes exactly the numbers of the chapter slides;
  * the course in one table (S&P 500, BET, Bitcoin on one common window) from Quantlets/Ch_16/ch16_numbers.json
    (generate_all_charts.py);
  * the seminar numbers from Quantlets/Ch_16/sem16_results.json (seminar16.py).
Clickable citations: DOIs and pages already verified for the earlier chapters (Crossref, 2-3 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_16')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_16'
NAMES = {'sp500': 'S\\&P 500', 'bet': 'BET', 'btc': 'Bitcoin'}
ASSETS = ['sp500', 'bet', 'btc']


def _json(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def load():
    return _json('Quantlets/Ch_16/ch16_numbers.json')


def load_sem():
    return _json('Quantlets/Ch_16/sem16_results.json')


def chapter_numbers():
    """The numbers files of Chapters 0--15."""
    out = {0: _json('Quantlets/Ch_00/ch0_values.json')}
    for c in range(1, 16):
        out[c] = _json(f'Quantlets/Ch_{c:02d}/ch{c}_numbers.json')
    return out


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def year(iso):
    return iso[:4]


def facts(V):
    """The key facts of every chapter as @{f.N.name} values (numbers from the chapter files)."""
    C = chapter_numbers()
    # Chapter 0: volatility since 2015; Chapter 1: drawdown of the BET since 1997
    V.put('f0.btcvol', C[0]['btc_vol'], 1)
    V.put('f0.spvol', C[0]['sp500_vol'], 1)
    d = C[1]['dd']['bet']
    V.put('f0.betmdd', 100 * d['mdd'], 1)
    put_date(V, 'f0.betpeak', d['peak'])
    put_date(V, 'f0.bettrough', d['trough'])
    put_date(V, 'f0.betrec', d['recovery'])
    # Chapter 1: BET since 2015
    p = C[1]['perf']['bet']
    V.put('f1.cagr', 100 * p['cagr'], 1)
    V.put('f1.vol', 100 * p['vol'], 1)
    V.put('f1.sh', p['sharpe'], 2)
    V.put('f1.shlo', p['sharpe_lo'], 2)
    V.put('f1.shhi', p['sharpe_hi'], 2)
    b = C[1]['perf']['btc']
    V.put('f1.btcar', 100 * b['mean_arith'], 1)
    V.put('f1.btccagr', 100 * b['cagr'], 1)
    V.raw('f1.y0', year(C[1]['start']))
    put_date(V, 'f1.betmin', C[1]['desc']['bet']['min_date'])
    V.put('f1.betminr', C[1]['desc']['bet']['min'], 1)
    # Chapter 2: BET since 2010
    m = C[2]['mom']['bet']
    V.raw('f2.y0', year(m['first']))
    V.put('f2.skew', m['skew'], 2)
    V.put('f2.k', m['exkurt'], 1)
    V.int('f2.jb', round(m['jb']))
    V.put('f2.nu', m['nu'], 2)
    fb = C[2]['facts']['bet']
    V.put('f2.rho1', fb['rho1'], 2)
    V.put('f2.rho1a', fb['rho1_abs'], 2)
    # Chapter 3: stable fits since 2000
    for k in ASSETS:
        V.put(f'f3.a.{k}', C[3]['fits'][k]['stable']['alpha'], 2)
    V.put('f3.am', C[3]['aggregation']['bet']['monthly']['alpha'], 2)
    V.raw('f3.y0', year(C[3]['start']))
    # Chapter 4: S&P 500 since 2000
    cf = C[4]['cdf']
    V.put('f4.p5e', 100 * cf['p5_emp'], 2)
    V.put('f4.p5n', 100 * cf['p5_norm'], 4)
    V.put('f4.ratio', cf['p5_emp'] / cf['p5_norm'], 0)
    L = C[4]['lln']
    V.put('f4.mean', L['mean_ann'], 1)
    V.put('f4.se', L['se_ann'], 1)
    V.put('f4.yrs', L['years_for_t3'], 0)
    V.raw('f4.y0', year(C[4]['start']))
    # Chapter 5: Hill tail index of the losses
    for k in ASSETS:
        h = C[5]['hill'][k]
        V.put(f'f5.a.{k}', h['alpha'], 2)
        V.raw(f'f5.y0.{k}', year(h['first']))
    V.put('f5.lo', C[5]['hill']['bet']['lo'], 2)
    V.put('f5.hi', C[5]['hill']['bet']['hi'], 2)
    r = C[5]['risk']['bet']
    V.put('f5.v01e', r['var01']['EVT'], 1)
    V.put('f5.v01n', r['var01']['Normal'], 1)
    # Chapter 6: BET since 2000
    g = C[6]['gof']['bet']
    V.put('f6.adn', g['Normal']['ad'], 0)
    V.put('f6.adt', g['Student-t']['ad'], 2)
    V.put('f6.adnig', g['NIG']['ad'], 2)
    a = C[6]['avg_var']['bet']
    V.put('f6.vn', a['normal'], 2)
    V.put('f6.vlo', a['min_heavy'], 2)
    V.put('f6.vhi', a['max_heavy'], 2)
    V.raw('f6.y0', year(C[6]['daily']['bet']['first']))
    # Chapter 7: variance ratio VR(5)
    for k in ASSETS:
        v = C[7]['vr'][k]
        V.put(f'f7.vr.{k}', v['vr']['5']['vr'], 2)
        V.put(f'f7.zs.{k}', v['vr']['5']['zs'], 2)
        V.raw(f'f7.y0.{k}', year(v['first']))
    # Chapter 8: S&P 500 clustering; EWMA half-life
    c = C[8]['clust']['sp500']
    V.put('f8.lbr', c['lb_r']['q'], 0)
    V.put('f8.lbr2', c['lb_r2']['q'], 0)
    V.put('f8.crit', c['lb_r']['crit'], 2)
    V.put('f8.hl', C[8]['ewma']['0.94']['half_life'], 1)
    # Chapter 9: GARCH(1,1)-t since 2000
    for k in ('sp500', 'bet'):
        mk = C[9]['markets'][k]
        V.put(f'f9.p.{k}', mk['pers'], 3)
        V.put(f'f9.hl.{k}', mk['hl'], 0)
    V.raw('f9.y0', year(C[9]['markets']['sp500']['first']))
    V.put('f9.p.btc', C[14]['garch']['btc']['pers'], 3)
    # Chapter 10: S&P 500 since 2000, backtests since 2005
    me = C[10]['methods']['sp500']
    V.put('f10.hs', me['HS']['var'], 2)
    V.put('f10.n', me['Normal']['var'], 2)
    V.put('f10.gap', 100 * (1 - me['Normal']['var'] / me['HS']['var']), 0)
    bt = C[10]['bt']['sp500']
    for mm, kk in (('Normal', 'n'), ('HS', 'hs'), ('GARCH-t', 'g'), ('FHS', 'f')):
        V.put(f'f10.r.{kk}', 100 * bt[mm]['rate'], 2)
    V.raw('f10.bt0', year(bt['start']))
    # Chapter 11: Hurst exponents of the BET since 2000
    hb = C[11]['markets']['bet']
    V.put('f11.hr', hb['r']['rs'], 2)
    V.put('f11.ha', hb['abs']['rs'], 2)
    V.put('f11.mc', hb['mc']['rs']['q975'], 2)
    assert hb['r']['lo_reject'], 'the slide says that the Lo test rejects for the BET returns'
    V.put('f11.hrs', C[11]['markets']['sp500']['r']['rs'], 2)
    V.put('f11.has', C[11]['markets']['sp500']['abs']['rs'], 2)
    # Chapter 12: German credit data, test sample
    vv = C[12]['val']
    V.put('f12.auc', vv['auc'], 3)
    V.put('f12.lo', vv['delong']['lo'], 2)
    V.put('f12.hi', vv['delong']['hi'], 2)
    V.put('f12.gini', vv['gini'], 2)
    V.raw('f12.n', str(C[12]['n']))
    V.raw('f12.ntest', str(vv['n_test']))
    # Chapter 13: sign of S&P 500 returns since 2008; realised volatility
    s = C[13]['sign']['sp500']
    V.put('f13.base', 100 * s['base_acc'], 1)
    V.put('f13.logit', 100 * s['Logit']['acc'], 1)
    V.put('f13.gb', 100 * s['GB']['acc'], 1)
    V.raw('f13.y0', year(s['first']))
    rv = C[13]['rv']['sp500']
    V.put('f13.har', rv['HAR']['r2_mean'], 2)
    V.put('f13.lasso', 100 * rv['Lasso']['r2_har'], 1)
    # Chapter 14: Bitcoin
    sb = C[14]['stats']['btc']
    V.put('f14.vol', sb['ann_vol'], 1)
    V.put('f14.vol252', sb['ann_vol_252'], 1)
    V.put('f14.pre', C[14]['corr']['btc_sp500']['pre2020'], 2)
    V.put('f14.post', C[14]['corr']['btc_sp500']['post2020'], 2)
    V.put('f14.mdd', C[14]['dd']['btc']['mdd'], 1)
    # Chapter 15: Banca Transilvania and the BET; bank connectedness
    tv = C[15]['covar']['TLV']
    V.put('f15.dcv', tv['dcovar'], 2)
    V.put('f15.lo', tv['ci'][0], 2)
    V.put('f15.hi', tv['ci'][1], 2)
    V.put('f15.var', tv['var_i'], 2)
    V.put('f15.dy', C[15]['dy']['total'], 0)
    return V


def summary_values(V, N):
    """The course in one table: @{s.<asset>.<quantity>}."""
    put_date(V, 's.first', N['summary']['sp500']['perf']['first'])
    put_date(V, 's.last', N['summary']['sp500']['perf']['last'])
    for k in ASSETS:
        s = N['summary'][k]
        p = s['perf']
        V.int(f's.{k}.n', p['n'])
        V.put(f's.{k}.q', p['q'], 0)
        V.put(f's.{k}.mu', p['mean_log'], 1)
        V.put(f's.{k}.vol', p['vol'], 1)
        V.put(f's.{k}.sh', p['sharpe'], 2)
        V.put(f's.{k}.shse', p['sharpe_se'], 2)
        V.put(f's.{k}.mdd', p['mdd'], 1)
        V.put(f's.{k}.skew', s['mom']['skew'], 2)
        V.put(f's.{k}.k', s['mom']['exkurt'], 1)
        V.int(f's.{k}.jb', round(s['mom']['jb']))
        V.put(f's.{k}.hill', s['hill']['alpha'], 2)
        V.put(f's.{k}.lbr', s['lb_r']['q'], 0)
        V.put(f's.{k}.lbr2', s['lb_r2']['q'], 0)
        V.put(f's.{k}.vr', s['vr']['vr'], 2)
        V.put(f's.{k}.zs', s['vr']['zs'], 2)
        V.put(f's.{k}.pers', s['garch']['pers'], 3)
        V.raw(f's.{k}.hl', '--' if s['garch']['half_life'] is None else '⁅' + f"{s['garch']['half_life']:.0f}" + '⁆')
        V.put(f's.{k}.vhs', s['var_hs'], 2)
        V.put(f's.{k}.vn', s['var_n'], 2)
        V.put(f's.{k}.ehs', s['es_hs'], 2)
        V.put(f's.{k}.gap', s['gap_n'], 0)
        V.put(f's.{k}.hr', s['H_r'], 2)
        V.put(f's.{k}.ha', s['H_abs'], 2)
        V.put(f's.{k}.a1', s['acf1_abs'], 2)
    V.put('s.crit', N['summary']['bet']['lb_r']['crit'], 2)
    V.put('s.ewhl', N['summary']['bet']['ewma_half'], 1)
    for k, b in N['backtest'].items():
        V.int(f'bt.{k}.n', b['n'])
        put_date(V, f'bt.{k}.first', b['first'])
        for m, mk in (('HS', 'hs'), ('EWMA', 'ew')):
            V.raw(f'bt.{k}.{mk}.x', str(b[m]['x']))
            V.put(f'bt.{k}.{mk}.rate', b[m]['rate'], 2)
            V.put(f'bt.{k}.{mk}.lr', b[m]['lr'], 2)
            V.raw(f'bt.{k}.{mk}.p', pv(b[m]['p']))
            V.raw(f'bt.{k}.{mk}.max', str(b[m]['max250']))
        V.put(f'bt.{k}.exp', 0.01 * b['n'], 1)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref for the earlier chapters)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refCLM}{\href{https://doi.org/10.1515/9781400830213}{Campbell, Lo \& MacKinlay (1997)}}
\newcommand{\refQRM}{\href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{McNeil, Frey \& Embrechts (2015)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refNolan}{\href{https://doi.org/10.1007/978-3-030-52915-4}{Nolan (2020)}}
\newcommand{\refJB}{\href{https://doi.org/10.1016/0165-1765(80)90024-5}{Jarque \& Bera (1980)}}
\newcommand{\refHill}{\href{https://doi.org/10.1214/aos/1176343247}{Hill (1975)}}
\newcommand{\refFama}{\href{https://doi.org/10.2307/2325486}{Fama (1970)}}
\newcommand{\refLM}{\href{https://doi.org/10.1093/rfs/1.1.41}{Lo \& MacKinlay (1988)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refBoll}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refRM}{\href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{J.P.\ Morgan/Reuters (1996)}}
\newcommand{\refArtzner}{\href{https://doi.org/10.1111/1467-9965.00068}{Artzner et al.\ (1999)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refHurst}{\href{https://doi.org/10.1061/TACEAT.0006518}{Hurst (1951)}}
\newcommand{\refPeters}{\href{https://www.wiley.com/en-us/Fractal+Market+Analysis\%3A+Applying+Chaos+Theory+to+Investment+and+Economics-p-9780471585244}{Peters (1994)}}
\newcommand{\refAltman}{\href{https://doi.org/10.1111/j.1540-6261.1968.tb00843.x}{Altman (1968)}}
\newcommand{\refGKX}{\href{https://doi.org/10.1093/rfs/hhaa009}{Gu, Kelly \& Xiu (2020)}}
\newcommand{\refNakamoto}{\href{https://bitcoin.org/bitcoin.pdf}{Nakamoto (2008)}}
\newcommand{\refAB}{\href{https://doi.org/10.1257/aer.20120555}{Adrian \& Brunnermeier (2016)}}
\newcommand{\refDY}{\href{https://doi.org/10.1016/j.ijforecast.2011.02.006}{Diebold \& Yilmaz (2012)}}
\newcommand{\refMenk}{\href{https://doi.org/10.1111/jofi.13337}{Menkveld et al.\ (2024)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
"""

_BIB = {
    'AB': r"Adrian, T., Brunnermeier, M.K. (2016). \href{https://doi.org/10.1257/aer.20120555}{CoVaR}. \textit{American Economic Review}, 106(7), 1705--1741.",
    'Altman': r"Altman, E.I. (1968). \href{https://doi.org/10.1111/j.1540-6261.1968.tb00843.x}{Financial ratios, discriminant analysis and the prediction of corporate bankruptcy}. \textit{Journal of Finance}, 23(4), 589--609.",
    'Artzner': r"Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). \href{https://doi.org/10.1111/1467-9965.00068}{Coherent measures of risk}. \textit{Mathematical Finance}, 9(3), 203--228.",
    'Boll': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'CLM': r"Campbell, J.Y., Lo, A.W., MacKinlay, A.C. (1997). \href{https://doi.org/10.1515/9781400830213}{\textit{The Econometrics of Financial Markets}}. Princeton University Press.",
    'Cont': r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    'DY': r"Diebold, F.X., Yilmaz, K. (2012). \href{https://doi.org/10.1016/j.ijforecast.2011.02.006}{Better to give than to receive: predictive directional measurement of volatility spillovers}. \textit{International Journal of Forecasting}, 28(1), 57--66.",
    'Engle': r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \textit{Econometrica}, 50(4), 987--1007.",
    'Fama': r"Fama, E.F. (1970). \href{https://doi.org/10.2307/2325486}{Efficient capital markets: a review of theory and empirical work}. \textit{Journal of Finance}, 25(2), 383--417.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GKX': r"Gu, S., Kelly, B., Xiu, D. (2020). \href{https://doi.org/10.1093/rfs/hhaa009}{Empirical asset pricing via machine learning}. \textit{Review of Financial Studies}, 33(5), 2223--2273.",
    'Hill': r"Hill, B.M. (1975). \href{https://doi.org/10.1214/aos/1176343247}{A simple general approach to inference about the tail of a distribution}. \textit{Annals of Statistics}, 3(5), 1163--1174.",
    'Hurst': r"Hurst, H.E. (1951). \href{https://doi.org/10.1061/TACEAT.0006518}{Long-term storage capacity of reservoirs}. \textit{Transactions of the American Society of Civil Engineers}, 116(1), 770--799.",
    'RM': r"J.P. Morgan/Reuters (1996). \href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{\textit{RiskMetrics -- Technical Document}}, 4th ed. New York.",
    'JB': r"Jarque, C.M., Bera, A.K. (1980). \href{https://doi.org/10.1016/0165-1765(80)90024-5}{Efficient tests for normality, homoscedasticity and serial independence of regression residuals}. \textit{Economics Letters}, 6(3), 255--259.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'LM': r"Lo, A.W., MacKinlay, A.C. (1988). \href{https://doi.org/10.1093/rfs/1.1.41}{Stock market prices do not follow random walks: evidence from a simple specification test}. \textit{Review of Financial Studies}, 1(1), 41--66.",
    'Mandelbrot': r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    'QRM': r"McNeil, A.J., Frey, R., Embrechts, P. (2015). \href{https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management}{\textit{Quantitative Risk Management: Concepts, Techniques and Tools}}, revised ed. Princeton University Press.",
    'Menk': r"Menkveld, A.J., Dreber, A., Holzmeister, F., Huber, J., Johannesson, M., et al. (2024). \href{https://doi.org/10.1111/jofi.13337}{Nonstandard errors}. \textit{Journal of Finance}, 79(3), 2339--2390.",
    'Nakamoto': r"Nakamoto, S. (2008). \href{https://bitcoin.org/bitcoin.pdf}{Bitcoin: a peer-to-peer electronic cash system}. White paper.",
    'Nolan': r"Nolan, J.P. (2020). \href{https://doi.org/10.1007/978-3-030-52915-4}{\textit{Univariate Stable Distributions: Models for Heavy Tailed Data}}. Springer.",
    'Peters': r"Peters, E.E. (1994). \href{https://www.wiley.com/en-us/Fractal+Market+Analysis\%3A+Applying+Chaos+Theory+to+Investment+and+Economics-p-9780471585244}{\textit{Fractal Market Analysis: Applying Chaos Theory to Investment and Economics}}. Wiley.",
    'Tsay': r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()


def money(x):
    return f'{x:,.0f}'.replace(',', '\\,')
