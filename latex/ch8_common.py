r"""
ch8_common.py -- shared helpers of the Chapter 8 generators (lecture and seminar), SFM
======================================================================================
Numbers from Quantlets/Ch_08/ch8_numbers.json (generate_all_charts.py) and sem8_results.json (seminar8.py);
the clickable citations of Chapter 8 (DOIs checked against Crossref, 3 October 2026).
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_08')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_08'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania (TLV)',
         'snp': 'OMV Petrom (SNP)', 'brd': 'BRD', 'tgn': 'Transgaz (TGN)'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP', tgn='TGN')
ASSETS = ['sp500', 'dax', 'bet', 'btc']
STOCKS = ['tlv', 'snp']
OHLC_ASSETS = ['sp500', 'dax', 'btc', 'tlv', 'snp']
EST = ['cc', 'park', 'gk', 'rs', 'yz']
METHODS = ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']


def load():
    with open(os.path.join(QL, 'ch8_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem8_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def fix_refs(D):
    """A citation macro followed by a space and a word would swallow the space (\\refX text): add {} after it."""
    D.FR = [re.sub(r'(\\ref[A-Z][A-Za-z]*)(?= [^\s])', r'\1{}', fr) for fr in D.FR]


def nz(x):
    """Avoid a printed '-0' after rounding to an integer."""
    return 0.0 if abs(x) < 0.5 else x


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    for k, b in N['bursts'].items():
        V.put(f'b.{k}.vol', b['vol'], 1)
        V.put(f'b.{k}.sd', b['sd'], 2)
        V.raw(f'b.{k}.ppy', str(round(b['ppy'])))
        V.put(f'b.{k}.max', b['max_abs'], 1)
        put_date(V, f'b.{k}.maxd', b['max_date'])
        V.put(f'b.{k}.kurt', b['kurt'], 1)
        V.raw(f'b.{k}.y0', b['first'][:4])
        V.int(f'b.{k}.n', b['n'])
    W = N['windows']
    for n in ['21', '63', '252']:
        V.put(f'w.{n}.max', W[n]['max'], 1)
        put_date(V, f'w.{n}.maxd', W[n]['max_date'])
        V.put(f'w.{n}.min', W[n]['min'], 1)
        V.put(f'w.{n}.last', W[n]['last'], 1)
        V.put(f'w.{n}.sdc', 100 * W[n]['sd_change'], 1)
    V.put('w.whole', W['whole'], 1)
    S = N['se']
    V.put('se.kappa', S['kappa'], 1)
    for n in ['21', '63', '252']:
        V.put(f'se.{n}.n', 100 * S[n]['normal'], 0)
        V.put(f'se.{n}.h', 100 * S[n]['heavy'], 0)
    for l, e in N['ewma'].items():
        t = l[2:]
        V.put(f'ew.{t}.hl', e['half_life'], 1)
        V.put(f'ew.{t}.lag', e['mean_lag'], 1)
        V.raw(f'ew.{t}.d99', str(round(e['days99'])))
    G = N['ghost']
    V.put('gh.rpeak', G['roll_peak'], 1)
    put_date(V, 'gh.rpeakd', G['roll_peak_date'])
    V.put('gh.epeak', G['ew_peak'], 1)
    put_date(V, 'gh.epeakd', G['ew_peak_date'])
    put_date(V, 'gh.dropd', G['drop_day'])
    V.put('gh.before', G['roll_before'], 1)
    V.put('gh.after', G['roll_after'], 1)
    put_date(V, 'gh.left', G['left_day'])
    V.put('gh.leftr', G['left_ret'], 1)
    V.put('gh.edrop', G['ew_drop_day'], 1)
    for s, d in N['eff'].items():
        for e, x in d.items():
            V.put(f'ef.{s}.{e}.bias', nz(100 * x['bias']), 0)
            V.put(f'ef.{s}.{e}.eff', x['eff'], 1)
            V.put(f'ef.{s}.{e}.mse', x['mse_ratio'], 1)
    for k, d in N['est'].items():
        for e in ['cc', 'park', 'gk', 'rs', 'yz']:
            V.put(f'es.{k}.{e}', d[e], 1)
        V.put(f'es.{k}.night', 100 * d['night_share'], 0)
        V.put(f'es.{k}.corr', d['corr_oc'], 2)
        V.raw(f'es.{k}.y0', d['first'][:4])
        V.int(f'es.{k}.n', d['n'])
    for e, x in N['range_rolling']['mean'].items():
        V.put(f'rr.{e}', x, 1)
        V.put(f'rr.rough.{e}', 100 * N['range_rolling']['roughness'][e], 1)
    R = N['rv']
    V.put('rv.mae5', 100 * R['mae_rv5'], 0)
    V.put('rv.mae1', 100 * R['mae_rv1'], 0)
    V.put('rv.maer', 100 * R['mae_r2'], 0)
    V.put('rv.maep', 100 * R['mae_park'], 0)
    V.put('rv.c5', R['corr_rv5'], 3)
    V.put('rv.cr', R['corr_r'], 2)
    V.put('rv.cp', R['corr_park'], 2)
    SG = N['signature']
    V.put('sg.n1', SG['noisy']['1'], 2)
    V.put('sg.n5', SG['noisy']['5'], 2)
    V.put('sg.n30', SG['noisy']['30'], 2)
    V.put('sg.c1', SG['clean']['1'], 2)
    V.put('sg.c390', SG['clean']['390'], 2)
    V.put('sg.noise', SG['noise'], 3)
    for k, c in N['clust'].items():
        V.int(f'cl.{k}.n', c['n'])
        V.put(f'cl.{k}.rr', c['rho_r'], 3)
        V.put(f'cl.{k}.r2', c['rho_r2'], 3)
        V.put(f'cl.{k}.abs', c['rho_abs'], 3)
        V.put(f'cl.{k}.lbr', c['lb_r']['q'], 1)
        V.raw(f'cl.{k}.plbr', pv(c['lb_r']['p']))
        V.int(f'cl.{k}.lbr2', round(c['lb_r2']['q']))
        V.int(f'cl.{k}.lbabs', round(c['lb_abs']['q']))
        V.int(f'cl.{k}.lm', round(c['arch']['lm']))
        V.put(f'cl.{k}.lmr2', c['arch']['r2'], 3)
        V.int(f'cl.{k}.lmn', c['arch']['n'])
        V.put(f'cl.{k}.kurt', c['exkurt'], 1)
    V.put('cl.crit10', N['clust']['sp500']['lb_r']['crit'], 2)
    V.put('cl.crit5', N['clust']['sp500']['arch']['crit'], 2)
    for k, a in N['acf'].items():
        V.put(f'ac.{k}.band', a['band'], 3)
        for j in ['1', '10', '50', '100']:
            V.put(f'ac.{k}.abs{j}', a['abs'][j], 2)
            V.put(f'ac.{k}.sq{j}', a['sq'][j], 2)
        V.put(f'ac.{k}.taylor', 100 * a['taylor'], 0)
        V.raw(f'ac.{k}.outsq', str(a['out_sq']))
        V.raw(f'ac.{k}.outabs', str(a['out_abs']))
        V.raw(f'ac.{k}.outr', str(a['out_r']))
    SH = N['shuffle']
    V.put('sh.a1', SH['acf1_abs'], 2)
    V.put('sh.a1s', SH['acf1_abs_shuffled'], 2)
    V.int('sh.lb', round(SH['lb_sq']))
    V.put('sh.lbs', SH['lb_sq_shuffled'], 1)
    V.put('sh.k', SH['kurt'], 1)
    K = N['kurt']
    V.put('ku.m022', K['mix_02_2'], 2)
    V.put('ku.m053', K['mix_05_3'], 2)
    V.put('ku.g', K['garch_01_085'], 2)
    L = N['long']
    V.put('lt.phi', L['ar']['phi'], 2)
    V.put('lt.se', L['ar']['se'], 2)
    V.put('lt.hl', L['ar']['half_life'], 1)
    V.put('lt.skr', L['ar']['skew_rv'], 1)
    V.put('lt.skl', L['ar']['skew_log'], 1)
    V.put('lt.lr', L['ar']['long_run'], 1)
    V.put('lt.mean', L['ar']['mean_rv'], 1)
    V.put('lt.med', L['ar']['median_rv'], 1)
    V.raw('lt.nm', str(L['ar']['n']))
    for k in ['sp500', 'dax', 'bet', 'btc']:
        V.put(f'lt.{k}.max', L[k]['max'], 0)
        V.raw(f'lt.{k}.maxy', str(L[k]['max_year']))
        V.put(f'lt.{k}.min', L[k]['min'], 0)
        V.raw(f'lt.{k}.miny', str(L[k]['min_year']))
        V.put(f'lt.{k}.med', L[k]['median'], 0)
        V.put(f'lt.{k}.y2020', L[k]['y2020'], 0)
    X = N['vix']
    V.put('vx.mv', X['mean_vix'], 1)
    V.put('vx.mr', X['mean_rv'], 1)
    V.put('vx.share', 100 * X['share_above'], 0)
    V.put('vx.gap', X['gap'], 1)
    V.put('vx.corr', X['corr'], 2)
    V.put('vx.mza', X['mz_vix']['a'], 2)
    V.put('vx.mzb', X['mz_vix']['b'], 2)
    V.put('vx.mzr', X['mz_vix']['r2'], 2)
    V.put('vx.pa', X['mz_past']['a'], 2)
    V.put('vx.pb', X['mz_past']['b'], 2)
    V.put('vx.pr', X['mz_past']['r2'], 2)
    V.put('vx.dvr', X['corr_dvix_r'], 2)
    V.put('vx.max', X['max_vix'], 1)
    put_date(V, 'vx.maxd', X['max_vix_date'])
    V.put('vx.last', X['last_vix'], 1)
    V.int('vx.n', X['n'])
    for k, d in N['lev'].items():
        V.put(f'lv.{k}.c1', d['cc']['1'], 3)
        V.put(f'lv.{k}.c2', d['cc']['2'], 3)
        V.put(f'lv.{k}.cm1', d['cc']['-1'], 3)
        V.put(f'lv.{k}.ratio', d['ratio'], 2)
        V.put(f'lv.{k}.neg', d['after_neg'], 2)
        V.put(f'lv.{k}.pos', d['after_pos'], 2)
        V.put(f'lv.{k}.bneg', d['after_big_neg'], 2)
        V.put(f'lv.{k}.bpos', d['after_big_pos'], 2)
    for k, d in N['npvol'].items():
        V.put(f'np.{k}.m2', d['m2'], 2)
        V.put(f'np.{k}.p2', d['p2'], 2)
        V.put(f'np.{k}.zero', d['zero'], 2)
        V.put(f'np.{k}.min', d['min_x'], 1)
    F = N['fc']
    V.int('fc.n', F['n'])
    for m in METHODS:
        V.put(f'fc.{m}.mse', F['r2'][m]['mse'], 2)
        V.put(f'fc.{m}.ql', F['r2'][m]['qlike'], 3)
        V.put(f'fc.{m}.mzr', F['mz'][m]['r2']['r2'], 2)
        V.put(f'fc.{m}.mzrr', F['mz'][m]['range']['r2'], 2)
        V.put(f'fc.{m}.mzb', F['mz'][m]['r2']['b'], 2)
    for c in ['dm_vix_ewma', 'dm_ewma_hist21', 'dm_ewma_hist252']:
        V.put(f'fc.{c}.t', F[c]['t'], 2)
        V.raw(f'fc.{c}.p', pv(F[c]['p']))
    for k, d in N['lambda'].items():
        V.put(f'la.{k}.best', d['best'], 3)
        V.put(f'la.{k}.q94', d['q094'], 3)
    for k, d in N['bvb'].items():
        V.put(f'bv.{k}.zr', 100 * d['zero_range'], 1)
        V.put(f'bv.{k}.flat', 100 * d['flat'], 0)
        V.put(f'bv.{k}.pc', d['park_cc'], 2)
        V.put(f'bv.{k}.yc', d['yz_cc'], 2)
        for e in ['cc', 'park', 'yz']:
            V.put(f'bv.{k}.q{e}', d['qlike'][e], 3)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 3 October 2026; web pages with HTTP 200)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refEngleN}{\href{https://doi.org/10.1257/0002828041464597}{Engle (2004)}}
\newcommand{\refNobel}{\href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{Nobel Prize (2003)}}
\newcommand{\refBollerslev}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refRM}{\href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{J.P. Morgan/Reuters (1996)}}
\newcommand{\refParkinson}{\href{https://doi.org/10.1086/296071}{Parkinson (1980)}}
\newcommand{\refGK}{\href{https://doi.org/10.1086/296072}{Garman \& Klass (1980)}}
\newcommand{\refRS}{\href{https://doi.org/10.1214/aoap/1177005835}{Rogers \& Satchell (1991)}}
\newcommand{\refYZ}{\href{https://doi.org/10.1086/209650}{Yang \& Zhang (2000)}}
\newcommand{\refMolnar}{\href{https://doi.org/10.1016/j.irfa.2011.06.012}{Molnár (2012)}}
\newcommand{\refABD}{\href{https://doi.org/10.1111/1540-6261.00454}{Alizadeh, Brandt \& Diebold (2002)}}
\newcommand{\refABDL}{\href{https://doi.org/10.1111/1468-0262.00418}{Andersen, Bollerslev, Diebold \& Labys (2003)}}
\newcommand{\refABDE}{\href{https://doi.org/10.1016/S0304-405X(01)00055-1}{Andersen, Bollerslev, Diebold \& Ebens (2001)}}
\newcommand{\refBNS}{\href{https://doi.org/10.1111/1467-9868.00336}{Barndorff-Nielsen \& Shephard (2002)}}
\newcommand{\refAB}{\href{https://doi.org/10.2307/2527343}{Andersen \& Bollerslev (1998)}}
\newcommand{\refLPS}{\href{https://doi.org/10.1016/j.jeconom.2015.02.008}{Liu, Patton \& Sheppard (2015)}}
\newcommand{\refCorsi}{\href{https://doi.org/10.1093/jjfinec/nbp001}{Corsi (2009)}}
\newcommand{\refML}{\href{https://doi.org/10.1111/j.1467-9892.1983.tb00373.x}{McLeod \& Li (1983)}}
\newcommand{\refLjung}{\href{https://doi.org/10.1093/biomet/65.2.297}{Ljung \& Box (1978)}}
\newcommand{\refDGE}{\href{https://doi.org/10.1016/0927-5398(93)90006-D}{Ding, Granger \& Engle (1993)}}
\newcommand{\refSchwert}{\href{https://doi.org/10.1111/j.1540-6261.1989.tb02647.x}{Schwert (1989)}}
\newcommand{\refChristie}{\href{https://doi.org/10.1016/0304-405X(82)90018-6}{Christie (1982)}}
\newcommand{\refFSS}{\href{https://doi.org/10.1016/0304-405X(87)90026-2}{French, Schwert \& Stambaugh (1987)}}
\newcommand{\refBW}{\href{https://doi.org/10.1093/rfs/13.1.1}{Bekaert \& Wu (2000)}}
\newcommand{\refWhaley}{\href{https://doi.org/10.3905/jod.1993.407868}{Whaley (1993)}}
\newcommand{\refWhaleyB}{\href{https://doi.org/10.3905/JPM.2009.35.3.098}{Whaley (2009)}}
\newcommand{\refCboe}{\href{https://www.cboe.com/tradable_products/vix/}{Cboe (2026)}}
\newcommand{\refCW}{\href{https://doi.org/10.1093/rfs/hhn038}{Carr \& Wu (2009)}}
\newcommand{\refBTZ}{\href{https://doi.org/10.1093/rfs/hhp008}{Bollerslev, Tauchen \& Zhou (2009)}}
\newcommand{\refPatton}{\href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Patton (2011)}}
\newcommand{\refHL}{\href{https://doi.org/10.1002/jae.800}{Hansen \& Lunde (2005)}}
\newcommand{\refPG}{\href{https://doi.org/10.1257/002205103765762743}{Poon \& Granger (2003)}}
\newcommand{\refDM}{\href{https://doi.org/10.1080/07350015.1995.10524599}{Diebold \& Mariano (1995)}}
\newcommand{\refMZ}{\href{https://www.nber.org/books-and-chapters/economic-forecasts-and-expectations-analysis-forecasting-behavior-and-performance/evaluation-economic-forecasts}{Mincer \& Zarnowitz (1969)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.3390/e19050226}{Pele, Lazar \& Dufour (2017)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e21020102}{Pele \& Mazurencu-Marinescu-Pele (2019)}}
"""

_BIB = {
    'AB': r"Andersen, T.G., Bollerslev, T. (1998). \href{https://doi.org/10.2307/2527343}{Answering the skeptics: yes, standard volatility models do provide accurate forecasts}. \textit{International Economic Review}, 39(4), 885--905.",
    'ABDE': r"Andersen, T.G., Bollerslev, T., Diebold, F.X., Ebens, H. (2001). \href{https://doi.org/10.1016/S0304-405X(01)00055-1}{The distribution of realized stock return volatility}. \textit{Journal of Financial Economics}, 61(1), 43--76.",
    'ABDL': r"Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P. (2003). \href{https://doi.org/10.1111/1468-0262.00418}{Modeling and forecasting realized volatility}. \textit{Econometrica}, 71(2), 579--625.",
    'ABD': r"Alizadeh, S., Brandt, M.W., Diebold, F.X. (2002). \href{https://doi.org/10.1111/1540-6261.00454}{Range-based estimation of stochastic volatility models}. \textit{Journal of Finance}, 57(3), 1047--1091.",
    'BNS': r"Barndorff-Nielsen, O.E., Shephard, N. (2002). \href{https://doi.org/10.1111/1467-9868.00336}{Econometric analysis of realized volatility and its use in estimating stochastic volatility models}. \textit{Journal of the Royal Statistical Society, Series B}, 64(2), 253--280.",
    'BW': r"Bekaert, G., Wu, G. (2000). \href{https://doi.org/10.1093/rfs/13.1.1}{Asymmetric volatility and risk in equity markets}. \textit{Review of Financial Studies}, 13(1), 1--42.",
    'Bollerslev': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'BTZ': r"Bollerslev, T., Tauchen, G., Zhou, H. (2009). \href{https://doi.org/10.1093/rfs/hhp008}{Expected stock returns and variance risk premia}. \textit{Review of Financial Studies}, 22(11), 4463--4492.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'CW': r"Carr, P., Wu, L. (2009). \href{https://doi.org/10.1093/rfs/hhn038}{Variance risk premiums}. \textit{Review of Financial Studies}, 22(3), 1311--1341.",
    'Cboe': r"Cboe Global Markets (2026). \href{https://www.cboe.com/tradable_products/vix/}{Cboe Volatility Index (VIX)}. Product page and methodology.",
    'Christie': r"Christie, A.A. (1982). \href{https://doi.org/10.1016/0304-405X(82)90018-6}{The stochastic behavior of common stock variances: value, leverage and interest rate effects}. \textit{Journal of Financial Economics}, 10(4), 407--432.",
    'Cont': r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    'Corsi': r"Corsi, F. (2009). \href{https://doi.org/10.1093/jjfinec/nbp001}{A simple approximate long-memory model of realized volatility}. \textit{Journal of Financial Econometrics}, 7(2), 174--196.",
    'DM': r"Diebold, F.X., Mariano, R.S. (1995). \href{https://doi.org/10.1080/07350015.1995.10524599}{Comparing predictive accuracy}. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263.",
    'DGE': r"Ding, Z., Granger, C.W.J., Engle, R.F. (1993). \href{https://doi.org/10.1016/0927-5398(93)90006-D}{A long memory property of stock market returns and a new model}. \textit{Journal of Empirical Finance}, 1(1), 83--106.",
    'Engle': r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \textit{Econometrica}, 50(4), 987--1007.",
    'EngleN': r"Engle, R.F. (2004). \href{https://doi.org/10.1257/0002828041464597}{Risk and volatility: econometric models and financial practice}. \textit{American Economic Review}, 94(3), 405--420.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'FSS': r"French, K.R., Schwert, G.W., Stambaugh, R.F. (1987). \href{https://doi.org/10.1016/0304-405X(87)90026-2}{Expected stock returns and volatility}. \textit{Journal of Financial Economics}, 19(1), 3--29.",
    'GK': r"Garman, M.B., Klass, M.J. (1980). \href{https://doi.org/10.1086/296072}{On the estimation of security price volatilities from historical data}. \textit{Journal of Business}, 53(1), 67--78.",
    'HL': r"Hansen, P.R., Lunde, A. (2005). \href{https://doi.org/10.1002/jae.800}{A forecast comparison of volatility models: does anything beat a GARCH(1,1)?} \textit{Journal of Applied Econometrics}, 20(7), 873--889.",
    'RM': r"J.P. Morgan/Reuters (1996). \href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{\textit{RiskMetrics -- Technical Document}}, 4th ed. New York: Morgan Guaranty Trust Company.",
    'LPS': r"Liu, L.Y., Patton, A.J., Sheppard, K. (2015). \href{https://doi.org/10.1016/j.jeconom.2015.02.008}{Does anything beat 5-minute RV? A comparison of realized measures across multiple asset classes}. \textit{Journal of Econometrics}, 187(1), 293--311.",
    'Ljung': r"Ljung, G.M., Box, G.E.P. (1978). \href{https://doi.org/10.1093/biomet/65.2.297}{On a measure of lack of fit in time series models}. \textit{Biometrika}, 65(2), 297--303.",
    'Mandelbrot': r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    'ML': r"McLeod, A.I., Li, W.K. (1983). \href{https://doi.org/10.1111/j.1467-9892.1983.tb00373.x}{Diagnostic checking ARMA time series models using squared-residual autocorrelations}. \textit{Journal of Time Series Analysis}, 4(4), 269--273.",
    'MZ': r"Mincer, J.A., Zarnowitz, V. (1969). \href{https://www.nber.org/books-and-chapters/economic-forecasts-and-expectations-analysis-forecasting-behavior-and-performance/evaluation-economic-forecasts}{The evaluation of economic forecasts}. In J.A. Mincer (ed.), \textit{Economic Forecasts and Expectations}, 3--46. NBER.",
    'Molnar': r"Molnár, P. (2012). \href{https://doi.org/10.1016/j.irfa.2011.06.012}{Properties of range-based volatility estimators}. \textit{International Review of Financial Analysis}, 23, 20--29.",
    'Nobel': r"Nobel Prize Outreach (2003). \href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel 2003: Robert F. Engle III and Clive W.J. Granger}. nobelprize.org.",
    'Parkinson': r"Parkinson, M. (1980). \href{https://doi.org/10.1086/296071}{The extreme value method for estimating the variance of the rate of return}. \textit{Journal of Business}, 53(1), 61--65.",
    'Patton': r"Patton, A.J. (2011). \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Volatility forecast comparison using imperfect volatility proxies}. \textit{Journal of Econometrics}, 160(1), 246--256.",
    'PeleA': r"Pele, D.T., Lazar, E., Dufour, A. (2017). \href{https://doi.org/10.3390/e19050226}{Information entropy and measures of market risk}. \textit{Entropy}, 19(5), 226.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21020102}{Using high-frequency entropy to forecast Bitcoin's daily Value at Risk}. \textit{Entropy}, 21(2), 102.",
    'PG': r"Poon, S.-H., Granger, C.W.J. (2003). \href{https://doi.org/10.1257/002205103765762743}{Forecasting volatility in financial markets: a review}. \textit{Journal of Economic Literature}, 41(2), 478--539.",
    'RS': r"Rogers, L.C.G., Satchell, S.E. (1991). \href{https://doi.org/10.1214/aoap/1177005835}{Estimating variance from high, low and closing prices}. \textit{Annals of Applied Probability}, 1(4), 504--512.",
    'Schwert': r"Schwert, G.W. (1989). \href{https://doi.org/10.1111/j.1540-6261.1989.tb02647.x}{Why does stock market volatility change over time?} \textit{Journal of Finance}, 44(5), 1115--1153.",
    'Tsay': r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\textit{Analysis of Financial Time Series}}, 3rd ed. Wiley.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'Whaley': r"Whaley, R.E. (1993). \href{https://doi.org/10.3905/jod.1993.407868}{Derivatives on market volatility}. \textit{Journal of Derivatives}, 1(1), 71--84.",
    'WhaleyB': r"Whaley, R.E. (2009). \href{https://doi.org/10.3905/JPM.2009.35.3.098}{Understanding the VIX}. \textit{Journal of Portfolio Management}, 35(3), 98--105.",
    'YZ': r"Yang, D., Zhang, Q. (2000). \href{https://doi.org/10.1086/209650}{Drift-independent volatility estimation based on high, low, open, and close prices}. \textit{Journal of Business}, 73(3), 477--492.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
