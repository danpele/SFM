r"""
ch12_common.py -- shared helpers of the Chapter 12 generators (lecture and seminar), SFM
========================================================================================
Numbers from Quantlets/Ch_12/ch12_numbers.json (generate_all_charts.py) and sem12_results.json (seminar12.py);
the clickable citations of Chapter 12 (DOIs checked against Crossref and DataCite, web pages checked, 3 October 2026).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_12')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_12'
VARNAME = {'status': ('checking account', 'contul curent'), 'credit_history': ('credit history', 'istoricul de credit'),
           'purpose': ('purpose', 'destinația creditului'), 'savings': ('savings', 'economiile'),
           'duration': ('duration', 'durata'), 'employment_duration': ('years with the employer', 'vechimea la angajator'),
           'age': ('age', 'vîrsta'), 'amount': ('amount', 'suma'), 'property': ('property', 'proprietatea')}


def load():
    with open(os.path.join(QL, 'ch12_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem12_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def vname(v):
    en, ro = VARNAME.get(v, (v.replace('_', ' '), v.replace('_', ' ')))
    return T(en, ro)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    V.int('n', N['n'])
    V.int('bad', N['bad'])
    V.put('badpct', 100 * N['bad'] / N['n'], 0)
    for c, d in N['status'].items():
        V.int(f'st{c}.n', d['n'])
        V.put(f'st{c}.r', 100 * d['rate'], 1)
    for c, d in N['status_woe'].items():
        V.int(f'sw{c}.g', d['good'])
        V.int(f'sw{c}.b', d['bad'])
        V.put(f'sw{c}.dg', d['dist_good'], 3)
        V.put(f'sw{c}.db', d['dist_bad'], 3)
        V.put(f'sw{c}.w', d['woe'], 3)
        V.put(f'sw{c}.iv', d['iv'], 3)
    V.put('sw.iv', sum(d['iv'] for d in N['status_woe'].values()), 3)
    dw = N['duration_woe']
    for i, (w, r, nn) in enumerate(zip(dw['woe'], dw['bad_rate'], dw['n'])):
        V.put(f'dw{i}.w', w, 2)
        V.put(f'dw{i}.r', 100 * r, 0)
        V.int(f'dw{i}.n', nn)
    V.put('dw.iv', dw['iv'], 3)
    for k, v in N['iv'].items():
        V.put(f'iv.{k}', v, 3)
    for k, v in N['iv_train'].items():
        V.put(f'ivt.{k}', v, 3)
    V.raw('nsel', str(len(N['sel'])))
    c = N['curve']
    V.put('cv.b0', c['b0'], 3)
    V.put('cv.b1', c['b1'], 4)
    V.put('cv.se1', c['se1'], 4)
    V.put('cv.or', math.exp(c['b1']), 3)
    V.put('cv.or12', math.exp(12 * c['b1']), 2)
    V.put('cv.p12', 100 / (1 + math.exp(-(c['b0'] + 12 * c['b1']))), 1)
    V.put('cv.p48', 100 / (1 + math.exp(-(c['b0'] + 48 * c['b1']))), 1)
    s = N['small']['table']
    for k, d in s.items():
        V.put(f'sm.{k}.b', d['b'], 3)
        V.put(f'sm.{k}.se', d['se'], 3)
        V.put(f'sm.{k}.z', d['z'], 2)
        V.raw(f'sm.{k}.p', pv(d['p']))
        V.put(f'sm.{k}.or', d['or'], 2)
        V.put(f'sm.{k}.lo', d['or_lo'], 2)
        V.put(f'sm.{k}.hi', d['or_hi'], 2)
        V.put(f'sm.{k}.pct', 100 * (d['or'] - 1), 0)
    V.put('sm.ll', N['small']['ll'], 1)
    V.put('sm.ll0', N['small']['ll0'], 1)
    V.put('sm.lr', N['small']['lr'], 1)
    L = N['lda2']
    V.put('l2.m0d', L['m0'][0], 2)
    V.put('l2.m0a', L['m0'][1], 2)
    V.put('l2.m1d', L['m1'][0], 2)
    V.put('l2.m1a', L['m1'][1], 2)
    V.put('l2.dd', L['m1'][0] - L['m0'][0], 2)
    V.put('l2.da', L['m1'][1] - L['m0'][1], 2)
    V.put('l2.s11', L['S'][0][0], 1)
    V.put('l2.s12', L['S'][0][1], 2)
    V.put('l2.s22', L['S'][1][1], 1)
    V.put('l2.w1', L['w'][0], 4)
    V.put('l2.w2', L['w'][1], 4)
    V.put('l2.lb1', L['logit_b'][1], 4)
    V.put('l2.lb2', L['logit_b'][2], 4)
    V.put('l2.ratio', L['w'][0] / L['w'][1], 2)
    V.put('l2.lratio', L['logit_b'][1] / L['logit_b'][2], 2)
    V.raw('l2.n0', str(L['n0']))
    V.raw('l2.n1', str(L['n1']))
    W = N['woe_logit']
    for i, v in enumerate(['const'] + W['vars']):
        V.put(f'wl.{v}.b', W['b'][i], 3)
        V.put(f'wl.{v}.se', W['se'][i], 3)
    V.put('wl.ll', W['ll'], 1)
    va = N['val']
    V.int('ntest', va['n_test'])
    V.int('ntrain', va['n_train'])
    V.int('badtest', va['bad_test'])
    for k in ['auc_train', 'auc', 'auc_lda', 'auc_single', 'gini', 'ar', 'ks', 'brier', 'brier_ref', 'corr_logit_lda',
              'hl', 'hl_p']:
        V.put(f'v.{k}', va[k], 3)
    V.put('v.ks_score', va['ks_score'], 0)
    V.put('v.auc.lo', va['delong']['lo'], 3)
    V.put('v.auc.hi', va['delong']['hi'], 3)
    V.put('v.auc.se', va['delong']['se'], 3)
    V.put('v.dl.z', va['delong_lda']['z'], 2)
    V.put('v.dl.p', va['delong_lda']['p'], 2)
    V.put('v.ds.z', va['delong_single']['z'], 2)
    V.put('v.ds.p', va['delong_single']['p'], 3)
    V.put('v.bss', 100 * (1 - va['brier'] / va['brier_ref']), 0)
    for cm in ('cm50', 'cm_cost'):
        d = va[cm]
        for k in ('TP', 'FN', 'FP', 'TN'):
            V.raw(f'{cm}.{k}', str(d[k]))
        for k in ('acc', 'tpr', 'tnr', 'fpr', 'precision', 'cost', 'reject'):
            V.put(f'{cm}.{k}', 100 * d[k] if k not in ('cost',) else d[k], 1 if k != 'cost' else 3)
    V.put('allgood', 100 * va['all_good'], 0)
    sc = N['scaling']
    V.put('sc.f', sc['factor'], 2)
    V.put('sc.o', sc['offset'], 2)
    V.put('prior', N['prior_shift'], 3)
    V.put('prior.pts', -N['prior_shift'] * sc['factor'], 1)
    d = N['score_dist']
    V.put('sd.cut', d['cut'], 0)
    V.put('sd.g', d['mean_good'], 0)
    V.put('sd.b', d['mean_bad'], 0)
    V.put('sd.min', d['min'], 0)
    V.put('sd.max', d['max'], 0)
    V.put('ks.at', N['ks_fig']['at'], 0)
    V.put('ks.ks', N['ks_fig']['ks'], 3)
    c = N['cost']
    V.put('cost.best', c['best_cut'], 2)
    V.put('cost.bestc', c['best_cost'], 3)
    V.put('cost.acc', c['cost_accept_all'], 2)
    V.put('cost.rej', c['cost_reject_all'], 2)
    for m, d in N['cv'].items():
        k = {'WoE logit': 'wl', 'LDA': 'lda', 'small logit': 'sm', 'status only': 'st'}[m]
        for part in ('train', 'test'):
            V.put(f'cv.{k}.{part}', d[part]['mean'], 3)
            V.put(f'cv.{k}.{part}.sd', d[part]['sd'], 3)
        V.put(f'cv.{k}.gap', d['train']['mean'] - d['test']['mean'], 3)
    tw = N['tw']
    V.int('tw.n', tw['n'])
    V.put('tw.rate', 100 * tw['rate'], 1)
    V.put('tw.allgood', 100 * (1 - tw['rate']), 1)
    t = N['tw_cal']
    for k in ['hl', 'brier', 'brier_ref', 'auc', 'auc_train']:
        V.put(f'tc.{k}', t[k], 3)
    V.raw('tc.hlp', pv(t['hl_p']))
    V.int('tc.n', t['n_test'])
    g = t['groups']
    V.put('tc.g0p', 100 * g['0']['pd_mean'], 1)
    V.put('tc.g0r', 100 * g['0']['rate'], 1)
    V.put('tc.g9p', 100 * g['9']['pd_mean'], 1)
    V.put('tc.g9r', 100 * g['9']['rate'], 1)
    V.put('tc.bss', 100 * (1 - t['brier'] / t['brier_ref']), 0)
    cm = N['tw_cm']
    V.put('twcm.acc', 100 * cm['acc'], 1)
    V.put('twcm.tpr', 100 * cm['tpr'], 1)
    r = N['reject']
    for k in ['bad_acc', 'bad_rej', 'bad_all', 'rate_rejected', 'rate_accepted_test', 'mean_pd_rej_acc_model',
              'mean_pd_rej_all_model']:
        V.put(f'ri.{k}', 100 * r[k], 1)
    for k in ['auc_acc_model_on_acc', 'auc_acc_model_on_all', 'auc_all_model_on_all', 'auc_all_model_on_acc',
              'auc_old_on_all']:
        V.put(f'ri.{k}', r[k], 3)
    V.int('ri.nacc', r['n_acc_train'])
    F = N['fair']['groups']
    for k, d in F.items():
        kk = k.replace('age ', 'a').replace('<= ', 'le').replace('> ', 'gt').replace('-', '_')
        for c in ('rate', 'pd', 'approve', 'tpr_good'):
            V.put(f'fr.{kk}.{c}', 100 * d[c], 1)
        V.int(f'fr.{kk}.n', d['n'])
    V.put('fr.dp', 100 * (F['women']['approve'] - F['men']['approve']), 1)
    V.put('fr.eo', 100 * (F['women']['tpr_good'] - F['men']['tpr_good']), 1)
    for p, d in N['irb'].items():
        key = str(round(float(p) * 1000))
        for kind, kk in [('other retail', 'or'), ('mortgage', 'mo'), ('revolving', 'rv')]:
            V.put(f'irb.{key}.{kk}', 100 * d[kind], 2)
            V.put(f'irb.{key}.{kk}.rw', 1250 * d[kind], 0)
        V.put(f'irb.{key}.R', N['irb_R'][p], 3)
    m = N['merton']
    V.put('mt.dd', m['dd'], 3)
    V.put('mt.pd', 100 * m['pd'], 2)
    me = N['merton_eq']
    V.put('me.V', me['V'], 1)
    V.put('me.s', 100 * me['sigma_V'], 1)
    V.put('me.dd', me['dd'], 2)
    V.put('me.pd', 100 * me['pd'], 2)
    V.put('alt.z', N['altman']['z'], 3)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref and DataCite, web pages verified, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refAltman}{\href{https://doi.org/10.1111/j.1540-6261.1968.tb00843.x}{Altman (1968)}}
\newcommand{\refFisher}{\href{https://doi.org/10.1111/j.1469-1809.1936.tb02137.x}{Fisher (1936)}}
\newcommand{\refMerton}{\href{https://doi.org/10.1111/j.1540-6261.1974.tb03058.x}{Merton (1974)}}
\newcommand{\refHM}{\href{https://doi.org/10.1148/radiology.143.1.7063747}{Hanley \& McNeil (1982)}}
\newcommand{\refDeLong}{\href{https://doi.org/10.2307/2531595}{DeLong et al.\ (1988)}}
\newcommand{\refBrier}{\href{https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2}{Brier (1950)}}
\newcommand{\refGordy}{\href{https://doi.org/10.1016/S1042-9573(03)00040-8}{Gordy (2003)}}
\newcommand{\refTCE}{\href{https://doi.org/10.1137/1.9781611974560}{Thomas, Crook \& Edelman (2017)}}
\newcommand{\refLessmann}{\href{https://doi.org/10.1016/j.ejor.2015.05.030}{Lessmann et al.\ (2015)}}
\newcommand{\refHH}{\href{https://doi.org/10.1111/j.1467-985X.1997.00078.x}{Hand \& Henley (1997)}}
\newcommand{\refOhlson}{\href{https://doi.org/10.2307/2490395}{Ohlson (1980)}}
\newcommand{\refShumway}{\href{https://doi.org/10.1086/209665}{Shumway (2001)}}
\newcommand{\refMVA}{\href{https://doi.org/10.1007/978-3-030-26006-4}{Härdle \& Simar (2019)}}
\newcommand{\refFuster}{\href{https://doi.org/10.1111/jofi.13090}{Fuster et al.\ (2022)}}
\newcommand{\refBS}{\href{https://doi.org/10.1093/rfs/hhn044}{Bharath \& Shumway (2008)}}
\newcommand{\refHL}{\href{https://doi.org/10.1080/03610928008827941}{Hosmer \& Lemeshow (1980)}}
\newcommand{\refCox}{\href{https://doi.org/10.1111/j.2517-6161.1958.tb00292.x}{Cox (1958)}}
\newcommand{\refSGC}{\href{https://doi.org/10.24432/C5QG88}{South German Credit (2020)}}
\newcommand{\refGroemping}{\href{http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf}{Grömping (2019)}}
\newcommand{\refTW}{\href{https://doi.org/10.24432/C55S3H}{Default of Credit Card Clients (2009)}}
\newcommand{\refYL}{\href{https://doi.org/10.1016/j.eswa.2007.12.020}{Yeh \& Lien (2009)}}
\newcommand{\refCHM}{\href{https://doi.org/10.1080/14697680903410015}{Chen, Härdle \& Moro (2011)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refBCT}{\href{https://doi.org/10.1057/palgrave.jors.2601578}{Banasik, Crook \& Thomas (2003)}}
\newcommand{\refSiddiqi}{\href{https://doi.org/10.1002/9781119201731}{Siddiqi (2006)}}
\newcommand{\refDurand}{\href{https://www.nber.org/books-and-chapters/risk-elements-consumer-instalment-financing-technical-edition}{Durand (1941)}}
\newcommand{\refBCBS}{\href{https://www.bis.org/publ/bcbs128.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (2006)}}
\newcommand{\refIRBnote}{\href{https://www.bis.org/bcbs/irbriskweight.htm}{⟦Basel Committee||Comitetul de la Basel⟧ (2005)}}
\newcommand{\refHPS}{\href{https://arxiv.org/abs/1610.02413}{Hardt, Price \& Srebro (2016)}}
\newcommand{\refECOA}{\href{https://www.ftc.gov/legal-library/browse/statutes/equal-credit-opportunity-act}{⟦Equal Credit Opportunity Act (FTC)||Equal Credit Opportunity Act (FTC)⟧}}
\newcommand{\refAIAct}{\href{https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai}{⟦European Commission, AI Act||Comisia Europeană, AI Act⟧}}
\newcommand{\refFICO}{\href{https://www.myfico.com/credit-education/blog/history-of-the-fico-score}{myFICO, ⟦history of the FICO score||istoria scorului FICO⟧}}
\newcommand{\refPeleZ}{\href{https://doi.org/10.1016/j.najef.2025.102543}{Będowska-Sójka, Wójcik \& Pele (2026)}}
"""

_BIB = {
    'Altman': r"Altman, E.I. (1968). \href{https://doi.org/10.1111/j.1540-6261.1968.tb00843.x}{Financial ratios, discriminant analysis and the prediction of corporate bankruptcy}. \textit{Journal of Finance}, 23(4), 589--609.",
    'BCT': r"Banasik, J., Crook, J., Thomas, L. (2003). \href{https://doi.org/10.1057/palgrave.jors.2601578}{Sample selection bias in credit scoring models}. \textit{Journal of the Operational Research Society}, 54(8), 822--832.",
    'IRBnote': r"Basel Committee on Banking Supervision (2005). \href{https://www.bis.org/bcbs/irbriskweight.htm}{An explanatory note on the Basel II IRB risk weight functions}. Bank for International Settlements.",
    'BCBS': r"Basel Committee on Banking Supervision (2006). \href{https://www.bis.org/publ/bcbs128.htm}{International convergence of capital measurement and capital standards: a revised framework (comprehensive version)}. Bank for International Settlements.",
    'PeleZ': r"Będowska-Sójka, B., Wójcik, P., Pele, D.T. (2026). \href{https://doi.org/10.1016/j.najef.2025.102543}{Early warning systems for cryptocurrency markets: predicting `zombie' assets using machine learning}. \textit{North American Journal of Economics and Finance}, 102543.",
    'BS': r"Bharath, S.T., Shumway, T. (2008). \href{https://doi.org/10.1093/rfs/hhn044}{Forecasting default with the Merton distance to default model}. \textit{Review of Financial Studies}, 21(3), 1339--1369.",
    'Brier': r"Brier, G.W. (1950). \href{https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2}{Verification of forecasts expressed in terms of probability}. \textit{Monthly Weather Review}, 78(1), 1--3.",
    'CHM': r"Chen, S., Härdle, W.K., Moro, R.A. (2011). \href{https://doi.org/10.1080/14697680903410015}{Modeling default risk with support vector machines}. \textit{Quantitative Finance}, 11(1), 135--154.",
    'Cox': r"Cox, D.R. (1958). \href{https://doi.org/10.1111/j.2517-6161.1958.tb00292.x}{The regression analysis of binary sequences}. \textit{Journal of the Royal Statistical Society, Series B}, 20(2), 215--232.",
    'TW': r"Default of Credit Card Clients [data set] (2009). \href{https://doi.org/10.24432/C55S3H}{UCI Machine Learning Repository}, CC BY 4.0.",
    'DeLong': r"DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). \href{https://doi.org/10.2307/2531595}{Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach}. \textit{Biometrics}, 44(3), 837--845.",
    'Durand': r"Durand, D. (1941). \href{https://www.nber.org/books-and-chapters/risk-elements-consumer-instalment-financing-technical-edition}{\textit{Risk Elements in Consumer Instalment Financing}}, technical edition. National Bureau of Economic Research.",
    'AIAct': r"European Commission (2024). \href{https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai}{AI Act: regulatory framework on artificial intelligence}. Shaping Europe's digital future.",
    'ECOA': r"Federal Trade Commission. \href{https://www.ftc.gov/legal-library/browse/statutes/equal-credit-opportunity-act}{Equal Credit Opportunity Act}, 15 U.S.C. §§ 1691--1691f.",
    'Fisher': r"Fisher, R.A. (1936). \href{https://doi.org/10.1111/j.1469-1809.1936.tb02137.x}{The use of multiple measurements in taxonomic problems}. \textit{Annals of Eugenics}, 7(2), 179--188.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'Fuster': r"Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., Walther, A. (2022). \href{https://doi.org/10.1111/jofi.13090}{Predictably unequal? The effects of machine learning on credit markets}. \textit{Journal of Finance}, 77(1), 5--47.",
    'Gordy': r"Gordy, M.B. (2003). \href{https://doi.org/10.1016/S1042-9573(03)00040-8}{A risk-factor model foundation for ratings-based bank capital rules}. \textit{Journal of Financial Intermediation}, 12(3), 199--232.",
    'Groemping': r"Grömping, U. (2019). \href{http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf}{South German credit data: correcting a widely used data set}. Reports in Mathematics, Physics and Chemistry 4/2019, Beuth University of Applied Sciences Berlin.",
    'HH': r"Hand, D.J., Henley, W.E. (1997). \href{https://doi.org/10.1111/j.1467-985X.1997.00078.x}{Statistical classification methods in consumer credit scoring: a review}. \textit{Journal of the Royal Statistical Society, Series A}, 160(3), 523--541.",
    'HM': r"Hanley, J.A., McNeil, B.J. (1982). \href{https://doi.org/10.1148/radiology.143.1.7063747}{The meaning and use of the area under a receiver operating characteristic (ROC) curve}. \textit{Radiology}, 143(1), 29--36.",
    'MVA': r"Härdle, W.K., Simar, L. (2019). \href{https://doi.org/10.1007/978-3-030-26006-4}{\textit{Applied Multivariate Statistical Analysis}}, 5th ed. Springer.",
    'HPS': r"Hardt, M., Price, E., Srebro, N. (2016). \href{https://arxiv.org/abs/1610.02413}{Equality of opportunity in supervised learning}. \textit{Advances in Neural Information Processing Systems} 29; arXiv:1610.02413.",
    'HL': r"Hosmer, D.W., Lemeshow, S. (1980). \href{https://doi.org/10.1080/03610928008827941}{Goodness of fit tests for the multiple logistic regression model}. \textit{Communications in Statistics -- Theory and Methods}, 9(10), 1043--1069.",
    'Lessmann': r"Lessmann, S., Baesens, B., Seow, H.-V., Thomas, L.C. (2015). \href{https://doi.org/10.1016/j.ejor.2015.05.030}{Benchmarking state-of-the-art classification algorithms for credit scoring: an update of research}. \textit{European Journal of Operational Research}, 247(1), 124--136.",
    'Merton': r"Merton, R.C. (1974). \href{https://doi.org/10.1111/j.1540-6261.1974.tb03058.x}{On the pricing of corporate debt: the risk structure of interest rates}. \textit{Journal of Finance}, 29(2), 449--470.",
    'FICO': r"myFICO. \href{https://www.myfico.com/credit-education/blog/history-of-the-fico-score}{The history of the FICO score}. Fair Isaac Corporation.",
    'Ohlson': r"Ohlson, J.A. (1980). \href{https://doi.org/10.2307/2490395}{Financial ratios and the probabilistic prediction of bankruptcy}. \textit{Journal of Accounting Research}, 18(1), 109--131.",
    'Shumway': r"Shumway, T. (2001). \href{https://doi.org/10.1086/209665}{Forecasting bankruptcy more accurately: a simple hazard model}. \textit{Journal of Business}, 74(1), 101--124.",
    'Siddiqi': r"Siddiqi, N. (2006). \href{https://doi.org/10.1002/9781119201731}{\textit{Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring}}. Wiley.",
    'SGC': r"South German Credit [data set] (2020). \href{https://doi.org/10.24432/C5QG88}{UCI Machine Learning Repository}, CC BY 4.0.",
    'TCE': r"Thomas, L.C., Crook, J., Edelman, D. (2017). \href{https://doi.org/10.1137/1.9781611974560}{\textit{Credit Scoring and Its Applications}}, 2nd ed. SIAM.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'YL': r"Yeh, I.-C., Lien, C.-H. (2009). \href{https://doi.org/10.1016/j.eswa.2007.12.020}{The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients}. \textit{Expert Systems with Applications}, 36(2), 2473--2480.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', '').replace('ę', 'e').replace('ö', 'o').replace('ä', 'a'))


BIB = bib()
