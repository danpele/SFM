r"""
ch13_common.py -- shared helpers of the Chapter 13 generators (lecture and seminar), SFM
=======================================================================================
Numbers from Quantlets/Ch_13/ch13_numbers.json (generate_all_charts.py) and sem13_results.json (seminar13.py);
the clickable citations of Chapter 13 (DOIs checked against Crossref, 3 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_13')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_13'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania (TLV)',
         'snp': 'OMV Petrom (SNP)'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP')
RV_MODELS = ['HAR', 'Lasso', 'RF', 'GB', 'MLP']
FEATS = ['d', 'w', 'm', 'd1', 'd2', 'd3', 'd4', 'q', 'r', 'r_neg', 'abs_r']


def load():
    with open(os.path.join(QL, 'ch13_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem13_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def put_rv(V, key, M, models):
    """Out-of-sample metrics of the volatility forecasts: R^2 against the mean and against HAR (%), QLIKE, DM."""
    V.int(f'{key}.n', M['n'])
    put_date(V, f'{key}.first', M['first'])
    put_date(V, f'{key}.last', M['last'])
    for m in models:
        d = M[m]
        V.put(f'{key}.{m}.r2m', 100 * d['r2_mean'], 1)
        V.put(f'{key}.{m}.r2h', 100 * d['r2_har'], 1, sign=True)
        V.put(f'{key}.{m}.ql', d['qlike'], 3)
        V.put(f'{key}.{m}.mse', d['mse'], 3)
        if m != 'HAR' and d['dm_t'] == d['dm_t']:
            V.put(f'{key}.{m}.dm', d['dm_t'], 2, sign=True)
            V.raw(f'{key}.{m}.dmp', pv(d['dm_p']))


def put_sign(V, key, S, models):
    V.int(f'{key}.n', S['n'])
    V.put(f'{key}.up', 100 * S['up'], 1)
    V.put(f'{key}.base', 100 * S['base_acc'], 1)
    for m in models:
        d = S[m]
        V.put(f'{key}.{m}.acc', 100 * d['acc'], 1)
        V.put(f'{key}.{m}.diff', 100 * d['diff'], 1, sign=True)
        V.put(f'{key}.{m}.z', d['z'], 2, sign=True)
        V.raw(f'{key}.{m}.p', pv(d['p']))
        V.put(f'{key}.{m}.auc', d['auc'], 3)
        V.put(f'{key}.{m}.upred', 100 * d['share_up_pred'], 0)


def put_var(V, key, B, methods):
    V.int(f'{key}.n', B['n'])
    put_date(V, f'{key}.first', B['first'])
    for m in methods:
        d = B[m]
        V.raw(f'{key}.{m}.x', str(d['x']))
        V.put(f'{key}.{m}.rate', 100 * d['rate'], 2)
        V.raw(f'{key}.{m}.puc', pv(d['p_uc']))
        V.raw(f'{key}.{m}.pind', pv(d['p_ind']))
        V.put(f'{key}.{m}.ql', 100 * d['qloss'], 2)
        V.put(f'{key}.{m}.mv', d['mean_var'], 2)
        V.raw(f'{key}.{m}.x20', str(d['x2020']))
        V.raw(f'{key}.{m}.n11', str(d['n11']))


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    B = N['bv']
    V.put('bv.s2', B['sigma2'], 2)
    V.raw('bv.best', str(B['best_depth']))
    V.raw('bv.n', str(B['n']))
    for d in ('1', '3', '10'):
        for c in ('bias2', 'var', 'test', 'train'):
            V.put(f'bv.{d}.{c}', B[f'd{d}'][c], 3)
    L = N['leak']
    for s in ('sim', 'sp500'):
        V.int(f'lk.{s}.n', L[s]['n'])
        for c in ('kfold', 'wf', 'purged'):
            V.put(f'lk.{s}.{c}', 100 * L[s][c], 1)
    S = N['shrink']
    V.put('sh.alpha', S['cv_alpha'], 4)
    V.raw('sh.nz', str(S['nonzero']))
    V.int('sh.n', S['n'])
    for f in FEATS:
        V.put(f'sh.cv.{f}', S['cv_coef'][f], 3, sign=True)
        V.put(f'sh.ols.{f}', S['ols'][f], 3, sign=True)
    R = N['tree']
    V.int('tr.n', R['n'])
    V.put('tr.thr', R['root_threshold'], 2)
    V.put('tr.vol', R['root_vol'], 1)
    V.put('tr.mean', R['mean_y'], 2)
    for i, (v, c) in enumerate(zip(R['leaves'], R['leaf_n']), 1):
        V.put(f'tr.l{i}', v, 2)
        V.raw(f'tr.n{i}', str(c))
    E = N['ens']
    for c in ('har', 'tree_best', 'rf_best', 'tree14', 'rf14'):
        V.put(f'en.{c}', E[c], 3)
    V.raw('en.td', str(E['tree_best_depth']))
    V.raw('en.rd', str(E['rf_best_depth']))
    for lr in ('0.3', '0.1', '0.03'):
        k = lr.replace('0.', '')
        V.put(f'en.gb{k}', E['gb_best'][lr], 3)
        V.raw(f'en.gb{k}.n', str(E['gb_best_n'][lr]))
        V.put(f'en.gb{k}.last', E['gb_last'][lr], 3)
    V.int('en.ntr', E['n_train'])
    V.int('en.nva', E['n_val'])
    A = N['nnarch']
    V.put('na.r2', 100 * A['r2_mean'], 2)
    V.put('na.corr', A['corr_rbf_garch'], 2)
    V.put('na.corrm', A['corr_mlp_garch'], 2)
    for c in ('mean_rbf', 'mean_garch', 'max_rbf', 'max_garch'):
        V.put(f'na.{c}', A[c], 1)
    put_date(V, 'na.dmax', A['date_max_garch'])
    put_date(V, 'na.first', A['first'])
    V.int('na.n', A['n'])
    V.put('na.alpha', A['alpha'], 3)
    V.put('na.beta', A['beta'], 3)
    J = N['nnjpy']
    for c in ('rmse_tr', 'rmse_te', 'rmse_rw_tr', 'rmse_rw_te'):
        V.put(f'jp.{c}', J[c], 2)
    V.put('jp.max_tr', J['max_tr'], 1)
    V.put('jp.max_te', J['max_te'], 1)
    put_date(V, 'jp.split', J['split'])
    V.int('jp.n', J['n'])
    V.put('jp.r2ret', 100 * J['r2_ret_te'], 2)
    V.put('jp.ratio', J['rmse_te'] / J['rmse_rw_te'], 0)
    for k in ('sp500', 'btc'):
        put_rv(V, f'rv.{k}', N['rv'][k], RV_MODELS)
    I = N['imp']
    for f in FEATS:
        V.put(f'im.{f}', I['perm'][f], 3)
    for f in ('d', 'w', 'm'):
        V.put(f'im.phi.{f}', I['mean_abs_phi'][f], 2)
    V.put('im.base', I['base'], 2)
    V.int('im.n', I['n_test'])
    for k in ('sp500', 'bet', 'btc'):
        put_sign(V, f'sg.{k}', N['sign'][k], ['Logit', 'RF', 'GB'])
    for k in ('sp500', 'btc'):
        put_var(V, f'qv.{k}', N['qvar'][k], ['QR', 'QGB', 'HS', 'GARCH-t'])
    Sn = N['snoop']
    b = Sn['best']
    V.raw('sn.N', str(Sn['N']))
    V.raw('sn.s', str(b['s']))
    V.raw('sn.l', str(b['l']))
    V.put('sn.is', b['is'], 2)
    V.put('sn.oos', b['oos'], 2)
    V.put('sn.ispp', b['is_pp'], 4)
    V.int('sn.T', b['T'])
    V.put('sn.skew', b['skew'], 2)
    V.put('sn.kurt', b['kurt'], 1)
    V.put('sn.sdpp', Sn['sd_pp'], 4)
    V.put('sn.sr0', Sn['dsr']['sr0'], 4)
    V.put('sn.dsr', Sn['dsr']['dsr'], 2)
    V.put('sn.psr', Sn['dsr']['psr'], 3)
    V.put('sn.z', Sn['dsr']['z'], 2)
    for c in ('bh_is', 'bh_oos', 'median_is', 'median_oos', 'corr'):
        V.put(f'sn.{c}', Sn[c], 2)
    V.raw('sn.rank', str(Sn['rank_oos']))
    M = N['maxsr']
    for n in ('10', '100', '1000'):
        V.put(f'ms.{n}', M[n]['mean'], 2)
        V.put(f'ms.{n}.f', M[n]['formula'], 2)
        V.put(f'ms.{n}.q95', M[n]['q95'], 2)
    G = N['gkx']
    for m, r, t, s in zip(G['models'], G['r2'], G['top'], G['sr']):
        k = m.replace('+H', '').replace('-', '')
        V.put(f'gkx.{k}.r2', r, 2)
        V.put(f'gkx.{k}.top', t, 2)
        V.put(f'gkx.{k}.sr', s, 2)
    D = N['design']
    V.raw('wf.folds', str(D['n_folds']))
    put_date(V, 'wf.first', D['first'])
    r0, r1 = D['rows'][0], D['rows'][-1]
    V.int('wf.ntr0', r0['n_train'])
    V.int('wf.nte0', r0['n_test'])
    put_date(V, 'wf.last0', r0['last_train'])
    put_date(V, 'wf.purge0', r0['first_purged'])
    V.int('wf.ntr1', r1['n_train'])
    V.raw('wf.h', str(D['h']))
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refISL}{\href{https://doi.org/10.1007/978-3-031-38747-0}{James et al.\ (2023)}}
\newcommand{\refHTF}{\href{https://doi.org/10.1007/978-0-387-84858-7}{Hastie, Tibshirani \& Friedman (2009)}}
\newcommand{\refBreimanTC}{\href{https://doi.org/10.1214/ss/1009213726}{Breiman (2001a)}}
\newcommand{\refBreiman}{\href{https://doi.org/10.1023/A:1010933404324}{Breiman (2001b)}}
\newcommand{\refFriedman}{\href{https://doi.org/10.1214/aos/1013203451}{Friedman (2001)}}
\newcommand{\refHK}{\href{https://doi.org/10.1080/00401706.1970.10488634}{Hoerl \& Kennard (1970)}}
\newcommand{\refTib}{\href{https://doi.org/10.1111/j.2517-6161.1996.tb02080.x}{Tibshirani (1996)}}
\newcommand{\refRosenblatt}{\href{https://doi.org/10.1037/h0042519}{Rosenblatt (1958)}}
\newcommand{\refHSW}{\href{https://doi.org/10.1016/0893-6080(89)90020-8}{Hornik, Stinchcombe \& White (1989)}}
\newcommand{\refRHW}{\href{https://doi.org/10.1038/323533a0}{Rumelhart, Hinton \& Williams (1986)}}
\newcommand{\refCorsi}{\href{https://doi.org/10.1093/jjfinec/nbp001}{Corsi (2009)}}
\newcommand{\refGK}{\href{https://doi.org/10.1086/296072}{Garman \& Klass (1980)}}
\newcommand{\refPatton}{\href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Patton (2011)}}
\newcommand{\refDM}{\href{https://doi.org/10.1080/07350015.1995.10524599}{Diebold \& Mariano (1995)}}
\newcommand{\refCT}{\href{https://doi.org/10.1093/rfs/hhm055}{Campbell \& Thompson (2008)}}
\newcommand{\refWG}{\href{https://doi.org/10.1093/rfs/hhm014}{Welch \& Goyal (2008)}}
\newcommand{\refKB}{\href{https://doi.org/10.2307/1913643}{Koenker \& Bassett (1978)}}
\newcommand{\refCAViaR}{\href{https://doi.org/10.1198/073500104000000370}{Engle \& Manganelli (2004)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refChris}{\href{https://doi.org/10.2307/2527341}{Christoffersen (1998)}}
\newcommand{\refBLdP}{\href{https://doi.org/10.3905/jpm.2014.40.5.094}{Bailey \& López de Prado (2014)}}
\newcommand{\refHLZ}{\href{https://doi.org/10.1093/rfs/hhv059}{Harvey, Liu \& Zhu (2016)}}
\newcommand{\refWhite}{\href{https://doi.org/10.1111/1468-0262.00152}{White (2000)}}
\newcommand{\refLdP}{\href{https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086}{López de Prado (2018)}}
\newcommand{\refBHK}{\href{https://doi.org/10.1016/j.csda.2017.11.003}{Bergmeir, Hyndman \& Koo (2018)}}
\newcommand{\refGKX}{\href{https://doi.org/10.1093/rfs/hhaa009}{Gu, Kelly \& Xiu (2020)}}
\newcommand{\refKX}{\href{https://doi.org/10.1561/0500000064}{Kelly \& Xiu (2023)}}
\newcommand{\refCSV}{\href{https://doi.org/10.1093/jjfinec/nbac020}{Christensen, Siggaard \& Veliyev (2023)}}
\newcommand{\refLL}{\href{https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html}{Lundberg \& Lee (2017)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refBoll}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.1080/1351847X.2021.1960403}{Pele et al.\ (2023)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/math14132316}{Pele \& Mazurencu-Marinescu-Pele (2026)}}
\newcommand{\refSFEa}{\href{https://github.com/QuantLet/SFE/tree/master/QID-3386-SFEnnarch}{SFEnnarch}}
\newcommand{\refSFEb}{\href{https://github.com/QuantLet/SFE/tree/master/QID-3387-SFEnnjpyusd}{SFEnnjpyusd}}
"""

_BIB = {
    'BLdP': r"Bailey, D.H., López de Prado, M. (2014). \href{https://doi.org/10.3905/jpm.2014.40.5.094}{The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality}. \textit{Journal of Portfolio Management}, 40(5), 94--107.",
    'BHK': r"Bergmeir, C., Hyndman, R.J., Koo, B. (2018). \href{https://doi.org/10.1016/j.csda.2017.11.003}{A note on the validity of cross-validation for evaluating autoregressive time series prediction}. \textit{Computational Statistics \& Data Analysis}, 120, 70--83.",
    'Boll': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'BHL': r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    'BreimanTC': r"Breiman, L. (2001a). \href{https://doi.org/10.1214/ss/1009213726}{Statistical modeling: the two cultures}. \textit{Statistical Science}, 16(3), 199--231.",
    'Breiman': r"Breiman, L. (2001b). \href{https://doi.org/10.1023/A:1010933404324}{Random forests}. \textit{Machine Learning}, 45(1), 5--32.",
    'CT': r"Campbell, J.Y., Thompson, S.B. (2008). \href{https://doi.org/10.1093/rfs/hhm055}{Predicting excess stock returns out of sample: can anything beat the historical average?} \textit{Review of Financial Studies}, 21(4), 1509--1531.",
    'Chris': r"Christoffersen, P.F. (1998). \href{https://doi.org/10.2307/2527341}{Evaluating interval forecasts}. \textit{International Economic Review}, 39(4), 841--862.",
    'CSV': r"Christensen, K., Siggaard, M., Veliyev, B. (2023). \href{https://doi.org/10.1093/jjfinec/nbac020}{A machine learning approach to volatility forecasting}. \textit{Journal of Financial Econometrics}, 21(5), 1680--1727.",
    'Corsi': r"Corsi, F. (2009). \href{https://doi.org/10.1093/jjfinec/nbp001}{A simple approximate long-memory model of realized volatility}. \textit{Journal of Financial Econometrics}, 7(2), 174--196.",
    'DM': r"Diebold, F.X., Mariano, R.S. (1995). \href{https://doi.org/10.1080/07350015.1995.10524599}{Comparing predictive accuracy}. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263.",
    'Engle': r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \textit{Econometrica}, 50(4), 987--1007.",
    'CAViaR': r"Engle, R.F., Manganelli, S. (2004). \href{https://doi.org/10.1198/073500104000000370}{CAViaR: conditional autoregressive value at risk by regression quantiles}. \textit{Journal of Business \& Economic Statistics}, 22(4), 367--381.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'Friedman': r"Friedman, J.H. (2001). \href{https://doi.org/10.1214/aos/1013203451}{Greedy function approximation: a gradient boosting machine}. \textit{Annals of Statistics}, 29(5), 1189--1232.",
    'GK': r"Garman, M.B., Klass, M.J. (1980). \href{https://doi.org/10.1086/296072}{On the estimation of security price volatilities from historical data}. \textit{Journal of Business}, 53(1), 67--78.",
    'GKX': r"Gu, S., Kelly, B., Xiu, D. (2020). \href{https://doi.org/10.1093/rfs/hhaa009}{Empirical asset pricing via machine learning}. \textit{Review of Financial Studies}, 33(5), 2223--2273.",
    'HLZ': r"Harvey, C.R., Liu, Y., Zhu, H. (2016). \href{https://doi.org/10.1093/rfs/hhv059}{\ldots and the cross-section of expected returns}. \textit{Review of Financial Studies}, 29(1), 5--68.",
    'HTF': r"Hastie, T., Tibshirani, R., Friedman, J. (2009). \href{https://doi.org/10.1007/978-0-387-84858-7}{\textit{The Elements of Statistical Learning}}, 2nd ed. Springer.",
    'HK': r"Hoerl, A.E., Kennard, R.W. (1970). \href{https://doi.org/10.1080/00401706.1970.10488634}{Ridge regression: biased estimation for nonorthogonal problems}. \textit{Technometrics}, 12(1), 55--67.",
    'HSW': r"Hornik, K., Stinchcombe, M., White, H. (1989). \href{https://doi.org/10.1016/0893-6080(89)90020-8}{Multilayer feedforward networks are universal approximators}. \textit{Neural Networks}, 2(5), 359--366.",
    'ISL': r"James, G., Witten, D., Hastie, T., Tibshirani, R., Taylor, J. (2023). \href{https://doi.org/10.1007/978-3-031-38747-0}{\textit{An Introduction to Statistical Learning with Applications in Python}}. Springer.",
    'KX': r"Kelly, B., Xiu, D. (2023). \href{https://doi.org/10.1561/0500000064}{Financial machine learning}. \textit{Foundations and Trends in Finance}, 13(3--4), 205--363.",
    'KB': r"Koenker, R., Bassett, G. (1978). \href{https://doi.org/10.2307/1913643}{Regression quantiles}. \textit{Econometrica}, 46(1), 33--50.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'LdP': r"López de Prado, M. (2018). \href{https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086}{\textit{Advances in Financial Machine Learning}}. Wiley.",
    'LL': r"Lundberg, S.M., Lee, S.-I. (2017). \href{https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html}{A unified approach to interpreting model predictions}. \textit{Advances in Neural Information Processing Systems}, 30.",
    'Patton': r"Patton, A.J. (2011). \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Volatility forecast comparison using imperfect volatility proxies}. \textit{Journal of Econometrics}, 160(1), 246--256.",
    'PeleA': r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847X.2021.1960403}{Are cryptos becoming alternative assets?} \textit{European Journal of Finance}, 29(10), 1064--1105.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2026). \href{https://doi.org/10.3390/math14132316}{Finite-sample precision limits for expected shortfall forecast comparisons}. \textit{Mathematics}, 14(13), 2316.",
    'RHW': r"Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). \href{https://doi.org/10.1038/323533a0}{Learning representations by back-propagating errors}. \textit{Nature}, 323, 533--536.",
    'Rosenblatt': r"Rosenblatt, F. (1958). \href{https://doi.org/10.1037/h0042519}{The perceptron: a probabilistic model for information storage and organization in the brain}. \textit{Psychological Review}, 65(6), 386--408.",
    'Tib': r"Tibshirani, R. (1996). \href{https://doi.org/10.1111/j.2517-6161.1996.tb02080.x}{Regression shrinkage and selection via the lasso}. \textit{Journal of the Royal Statistical Society, Series B}, 58(1), 267--288.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'WG': r"Welch, I., Goyal, A. (2008). \href{https://doi.org/10.1093/rfs/hhm014}{A comprehensive look at the empirical performance of equity premium prediction}. \textit{Review of Financial Studies}, 21(4), 1455--1508.",
    'White': r"White, H. (2000). \href{https://doi.org/10.1111/1468-0262.00152}{A reality check for data snooping}. \textit{Econometrica}, 68(5), 1097--1126.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
