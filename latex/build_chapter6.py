r"""
build_chapter6.py -- Capitolul 6 (Selecția modelului și managementul riscului), EN + RO dintr-o singură sursă
==========================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_06/ch6_numbers.json (generate_all_charts.py) și din
sem6_results.json (seminar6.py, exemplul KS pe hîrtie) sau sînt calculate aici, în Python, pentru exemplele lucrate.
Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter6_model_selection_risk_management.tex
  RO/Cursuri/capitol6_selectia_modelului_managementul_riscului.tex
Rulare:
  python3 Quantlets/Ch_06/generate_all_charts.py && python3 Quantlets/Ch_06/seminar6.py
  python3 latex/build_chapter6.py && python3 latex/sfm_build.py compile 6
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, photo, ql   # noqa: E402
from ch6_common import ASSETS, BIB, LONG, MNAME, MODELS, NAMES, REFS, STOCKS, T, big, load, load_sem, pval, put_date, sci   # noqa: E402

N = load()
S = load_sem()
V = Values()
D = Deck(6, 'lecture', refs=REFS)
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_06'
ALL = ASSETS + STOCKS


def chart(title, fig, folder, bullets, h='0.62\\textheight', size='footnotesize'):
    D.chart(title, fig, folder, bullets, height=h, size=size)


def pic(file, cap_en, cap_ro, url, cred_en, cred_ro, h='0.50\\textheight'):
    return photo(file, T(cap_en, cap_ro), url, T(cred_en, cred_ro), h=h)


# =============================================================================
# CIFRE
# =============================================================================
F = N['fits']
put_date(V, 'end', N['end'])
V.raw('y1', N['end'][:4])
for k in ALL:
    f = F[k]
    V.int(f'n.{k}', f['n'])
    V.raw(f'y0.{k}', f['first'][:4])
    V.raw(f'best.{k}', MNAME[f['best_aic']])
    V.raw(f'bestb.{k}', MNAME[f['best_bic']])
    for m in MODELS:
        v = f['models'][m]
        V.raw(f'll.{k}.{m}', big(v['loglik']))
        V.raw(f'aic.{k}.{m}', big(v['aic']))
        V.raw(f'bic.{k}.{m}', big(v['bic']))
        V.raw(f'daic.{k}.{m}', big(v['daic']) if v['daic'] >= 0.5 else '0')
        V.put(f'daic1.{k}.{m}', v['daic'], 1)
        V.put(f'dbic1.{k}.{m}', v['dbic'], 1)
        V.put(f'w.{k}.{m}', v['weight'], 3)
    P = {m: f['models'][m]['params'] for m in MODELS}
    V.put(f'nu.{k}', P['Student-t']['nu'], 2)
    V.put(f'lam.{k}', P['Skewed-t']['lam'], 3)
    V.put(f'ged.{k}', P['GED']['beta'], 2)
# S&P 500 parametri
P = {m: F['sp500']['models'][m]['params'] for m in MODELS}
V.put('sp.mu', P['Normal']['mu'], 3)
V.put('sp.sig', P['Normal']['sigma'], 3)
V.put('sp.t.nu', P['Student-t']['nu'], 2)
V.put('sp.t.loc', P['Student-t']['loc'], 3)
V.put('sp.t.sc', P['Student-t']['scale'], 3)
V.put('sp.skt.mu', P['Skewed-t']['mu'], 3)
V.put('sp.skt.sig', P['Skewed-t']['sigma'], 2)
V.put('sp.skt.nu', P['Skewed-t']['nu'], 2)
V.put('sp.skt.lam', P['Skewed-t']['lam'], 3)
V.put('sp.ged.b', P['GED']['beta'], 2)
V.put('sp.ged.loc', P['GED']['loc'], 3)
V.put('sp.ged.sc', P['GED']['scale'], 3)
V.put('sp.mix.w1', 100 * P['Mixture']['w'][0], 0)
V.put('sp.mix.w2', 100 * P['Mixture']['w'][1], 0)
V.put('sp.mix.s1', P['Mixture']['sd'][0], 2)
V.put('sp.mix.s2', P['Mixture']['sd'][1], 2)
V.put('sp.mix.m1', P['Mixture']['mu'][0], 3)
V.put('sp.mix.m2', P['Mixture']['mu'][1], 3)
V.put('sp.nig.a', P['NIG']['a'], 3)
V.put('sp.nig.b', P['NIG']['b'], 3)
V.put('sp.nig.loc', P['NIG']['loc'], 3)
V.put('sp.nig.sc', P['NIG']['scale'], 3)
V.put('sp.st.a', P['Stable']['alpha'], 3)
V.put('sp.st.b', P['Stable']['beta'], 3)
V.put('sp.st.g', P['Stable']['gamma'], 3)
V.put('sp.st.d', P['Stable']['delta0'], 3)
# Normal ML pas cu pas (S&P 500)
n_sp = F['sp500']['n']
sig = P['Normal']['sigma']
llN = -n_sp / 2 * (math.log(2 * math.pi * sig ** 2) + 1)
V.put('sp.ln2pis', math.log(2 * math.pi * sig ** 2), 4)
V.raw('sp.llN', big(llN))
V.put('lnn.sp', math.log(n_sp), 2)
# profil nu
Pr = N['profile']
V.put('pr.nu', Pr['nu'], 2)
V.put('pr.lo', Pr['lo'], 1)
V.put('pr.hi', Pr['hi'], 1)
V.put('pr.l3', Pr['ll_3'], 1)
V.put('pr.l5', Pr['ll_5'], 0)
# teste LR și Vuong
TS = N['tests']
for k in ASSETS:
    t = TS[k]
    V.put(f'lr.tskt.{k}', t['t_vs_skt']['lr'], 2)
    V.raw(f'lrp.tskt.{k}', pval(t['t_vs_skt']['p']))
    V.raw(f'lr.nged.{k}', big(t['n_vs_ged']['lr']))
    V.raw(f'lr.nt.{k}', big(t['n_vs_t']['lr']))
    for c in ['vu_t_ged', 'vu_t_stable', 'vu_skt_nig']:
        V.put(f'{c}.{k}', t[c]['v'], 2)
        V.raw(f'{c}p.{k}', pval(t[c]['p']))
V.put('chi95', 3.841, 2)
V.put('chi90', 2.706, 2)
V.put('chi2_95', 5.991, 2)
# candidați
CA = N['candidates']
for m in MODELS:
    V.raw(f'ca.{m}', sci(CA[m]['p_below_m4'], 1))
V.put('ca.ratio.t', CA['Student-t']['p_below_m4'] / CA['Normal']['p_below_m4'], 0)
# SFESumm
DM = N['dax_monthly']
for per in ['book', 'full']:
    s = DM[per]
    V.int(f'dm.{per}.n', s['n'])
    for c in ['min', 'max', 'mean', 'median', 'sd', 'ann_vol']:
        V.put(f'dm.{per}.{c}', 100 * s[c], 2)
    V.put(f'dm.{per}.skew', s['skew'], 2)
    V.put(f'dm.{per}.kurt', s['kurt'], 2)
    V.put(f'dm.{per}.jb', s['jb'], 1)
    V.raw(f'dm.{per}.jbp', pval(s['jb_p']))
put_date(V, 'dm.full.last', DM['full_last'], short=True)
DS = N['daily']
for k in ALL:
    s = DS[k]
    V.put(f'ds.{k}.sd', s['sd'], 2)
    V.put(f'ds.{k}.skew', s['skew'], 2)
    V.put(f'ds.{k}.kurt', s['kurt'], 1)
    V.put(f'ds.{k}.min', s['min'], 1)
    V.raw(f'ds.{k}.jb', big(s['jb']))
# TLV
TL = N['tlv']
V.put('tlv.bad1', TL['r_bad'][0], 1)
V.put('tlv.bad2', TL['r_bad'][1], 1)
for lab in ['raw', 'clean']:
    V.put(f'tlv.{lab}.kurt', TL[lab]['kurt'], 1)
    V.put(f'tlv.{lab}.nu', TL[lab]['nu'], 2)
    V.put(f'tlv.{lab}.var', TL[lab]['var1_t'], 2)
    V.raw(f'tlv.{lab}.best', MNAME[TL[lab]['best']])
# GoF
GB = N['gof_boot']
for m in ['Normal', 'Student-t', 'Skewed-t']:
    for s_ in ['ks', 'cvm', 'ad']:
        V.put(f'gb.{m}.{s_}', GB[m]['stat'][s_], 3 if s_ != 'ad' else 2)
        V.put(f'gb.{m}.{s_}.crit', GB[m]['crit_boot'][s_], 3 if s_ != 'ad' else 2)
    V.raw(f'gb.{m}.naive', pval(GB[m]['p_naive_ks']) if GB[m]['p_naive_ks'] >= 1e-3 else sci(GB[m]['p_naive_ks'], 1))
V.put('gb.pmin', 1 / (GB['Normal']['B'] + 1), 3)
V.int('gb.B', GB['Normal']['B'])
G = N['gof']['sp500']
for m in MODELS:
    V.put(f'g.{m}.ks', G[m]['ks'], 3)
    V.put(f'g.{m}.cvm', G[m]['cvm'], 2)
    V.put(f'g.{m}.ad', G[m]['ad'], 2)
LI = N['lillie']
for nn in ['5', '100', '1000']:
    V.put(f'li.{nn}', LI[nn]['ks'], 3)
V.put('ks.5', 0.5633, 3)
V.put('ks.asy.1000', 1.358 / math.sqrt(1000), 3)
V.put('ks.asy.sp', 1.358 / math.sqrt(n_sp), 3)
PW = N['power']
for a in ['t5', 'contam']:
    for nn in ['100', '250', '1000']:
        for s_ in ['ks', 'cvm', 'ad']:
            V.put(f'pw.{a}.{nn}.{s_}', 100 * PW[a][nn][s_], 0)
KT = N['ks_tail']
V.put('kt.x', KT['ks_x'], 2)
V.put('kt.d', abs(KT['ks_d']), 3)
V.put('kt.F', 100 * KT['ks_F'], 0)
V.put('kt.wx', KT['w_x'], 1)
# exemplul KS pe hîrtie (seminarul A5)
A5 = S['A']['a5']
V.put('a5.d', A5['d_known'], 3)
V.put('a5.dest', A5['d_est'], 3)
V.put('a5.crit', A5['crit_known'], 3)
V.put('a5.lil', A5['crit_lillie'], 3)
for i, u in enumerate(A5['u_known']):
    V.put(f'a5.u{i}', u, 3)
# PP/QQ
PQ = N['ppqq']
V.put('pq.xmin', PQ['x_min'], 1)
for m in ['Normal', 'Student-t', 'Skewed-t', 'Stable']:
    V.put(f'pq.{m}.q', PQ[m]['q0001'], 1)
    V.put(f'pq.{m}.pp', PQ[m]['pp_maxdev'], 3)
# cozi
TA = N['tails']
for k in ALL:
    V.put(f'v.{k}.emp', TA[k]['emp']['var1'], 2)
    V.put(f'e.{k}.emp', TA[k]['emp']['es25'], 2)
    V.raw(f'x.{k}.exp', f"{F[k]['n'] / 100:.0f}")
    for m in MODELS:
        V.put(f'v.{k}.{m}', TA[k][m]['var1'], 2)
        V.put(f'e.{k}.{m}', TA[k][m]['es25'], 2)
        V.raw(f'x.{k}.{m}', str(TA[k][m]['kupiec']['x']))
        V.raw(f'xp.{k}.{m}', pval(TA[k][m]['kupiec']['p']))
V.put('z01', 2.326, 3)
V.put('sp.varN', P['Normal']['sigma'] * 2.3263478740408408 - P['Normal']['mu'], 2)
V.put('sp.zsig', 2.3263478740408408 * P['Normal']['sigma'], 3)
AV = N['avg_var']
for k in ALL:
    for c in ['avg', 'min_heavy', 'max_heavy', 'normal', 'emp']:
        V.put(f'av.{k}.{c}', AV[k][c], 2)
    V.put(f'av.{k}.range', 100 * (AV[k]['max_heavy'] / AV[k]['min_heavy'] - 1), 0)
# out-of-sample
O = N['oos']
for k in ASSETS:
    o = O[k]
    V.int(f'o.{k}.nest', o['n_est'])
    V.int(f'o.{k}.ntest', o['n_test'])
    V.put(f'o.{k}.exp', o['n_test'] / 100, 1)
    V.raw(f'o.{k}.baic', MNAME[o['best_aic_est']])
    V.raw(f'o.{k}.bls', MNAME[o['best_logscore']])
    V.raw(f'o.{k}.bpin', MNAME[o['best_pinball']])
    for m in MODELS + ['Historical']:
        V.raw(f'o.{k}.{m}.x', str(o[m]['kupiec']['x']))
        V.raw(f'o.{k}.{m}.p', pval(o[m]['kupiec']['p']))
        V.put(f'o.{k}.{m}.var', o[m]['var1'], 2)
        if 'logscore' in o[m]:
            V.put(f'o.{k}.{m}.ls', o[m]['logscore'], 3)
OF = N['overfit']
for c in ['in', 'out']:
    for kk, r in OF['res'].items():
        V.put(f'of.{kk}.{c}', r[c], 3)
for kk, r in OF['res'].items():
    V.raw(f'of.{kk}.k', str(r['k']))
V.int('of.nin', OF['n_in'])
V.int('of.nout', OF['n_out'])
for c in ['best_in', 'best_out', 'best_aic', 'best_bic']:
    V.raw(f'of.{c}', str(OF[c]))
RO_ = N['rolling']
RB = N['rolling_bet']
for lab, R in [('sp', RO_), ('bet', RB)]:
    for m in ['Normal', 'Student-t', 'Skewed-t', 'Historical']:
        V.put(f'rv.{lab}.{m}.rate', 100 * R[m]['rate'], 2)
        V.raw(f'rv.{lab}.{m}.x', str(R[m]['x']))
        V.raw(f'rv.{lab}.{m}.p', pval(R[m]['p']))
        V.raw(f'rv.{lab}.{m}.my', str(R[m]['max_year']))
    V.int(f'rv.{lab}.n', R['Normal']['n'])
    V.raw(f'rv.{lab}.exp', f"{R['Normal']['n'] / 100:.0f}")
    V.raw(f'rv.{lab}.y0', R['first'][:4])
VQ = N['var_qq']
for c in ['rma', 'ema']:
    V.put(f'vq.{c}.rate', 100 * VQ[c]['rate'], 2)
    V.raw(f'vq.{c}.x', str(VQ[c]['x']))
    V.raw(f'vq.{c}.p', pval(VQ[c]['p']))
    V.put(f'vq.{c}.kurt', VQ[c]['kurt_std'], 1)
V.int('vq.n', VQ['n'])
V.raw('vq.y0', VQ['first'][:4])
U = N['uncertainty']
for m, w in U['wins'].items():
    V.put(f'u.w.{m}', 100 * w, 1 if 0 < 100 * w < 10 else 0)
for m, v in U['var'].items():
    V.put(f'u.v.{m}.med', v['median'], 2)
    V.put(f'u.v.{m}.lo', v['q05'], 2)
    V.put(f'u.v.{m}.hi', v['q95'], 2)
V.int('u.B', U['B'])
# rapoarte VaR model / VaR empiric (toate seriile)
rat = {m: [TA[k][m]['var1'] / TA[k]['emp']['var1'] for k in ALL] for m in MODELS}
V.put('rat.n.lo', 100 * (1 - max(rat['Normal'])), 0)
V.put('rat.n.hi', 100 * (1 - min(rat['Normal'])), 0)
V.put('rat.s.hi', 100 * (max(rat['Stable']) - 1), 0)
V.put('rat.good', 100 * max(abs(1 - v) for m in ['Student-t', 'Skewed-t', 'NIG'] for v in rat[m]), 0)
lsd = [O[k][m]['logscore'] - O[k]['Normal']['logscore'] for k in ASSETS for m in MODELS if m != 'Normal']
V.put('ls.lo', min(lsd), 2)
V.put('ls.hi', max(lsd), 2)
V.put('lnn.brd', math.log(F['brd']['n']), 1)
V.put('tlv.kdrop', 100 * (1 - TL['clean']['kurt'] / TL['raw']['kurt']), 0)
# exemplu Akaike weights (S&P 500): ΔAIC al celor mai apropiate două modele
dsp = F['sp500']['models']
V.put('ex.e.skt', math.exp(-0.5 * dsp['Skewed-t']['daic']), 6)
V.raw('ex.e.skt.sci', sci(math.exp(-0.5 * dsp['Skewed-t']['daic']), 1))
V.put('ex.d2.w', math.exp(-1) / (1 + math.exp(-1)), 2)


# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: we have seven candidate distributions for daily returns; which one should a risk manager use, and how sure can we be?',
       '\\textbf{Întrebarea}: avem șapte distribuții candidate pentru randamentele zilnice; pe care să o folosească un manager de risc și cît de siguri putem fi?'),
     [T('Chapters 2, 3 and 5: the Normal distribution fails; Student-t, stable laws and extreme value theory each describe part of the picture',
        'Capitolele 2, 3 și 5: distribuția Normală eșuează; Student-t, distribuțiile stabile și teoria valorilor extreme descriu fiecare doar o parte a fenomenului')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the candidates and what they must reproduce; maximum likelihood, recalled', 'candidații și proprietățile pe care trebuie să le reproducă; recapitulare: metoda verosimilității maxime'),
      T('comparing models: likelihood-ratio tests, AIC and BIC', 'compararea modelelor: teste ale raportului de verosimilitate, AIC și BIC'),
      T('checking one model: goodness-of-fit tests, PP and QQ plots', 'verificarea unui model: teste de concordanță, grafice PP și QQ'),
      T('the purpose decides: which model gets VaR 1\\% right, in sample and out of sample', 'scopul decide: ce model estimează corect VaR 1\\%, în eșantion și în afara lui'),
      T('overfitting and model uncertainty', 'supraajustarea și incertitudinea de model')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Write the log-likelihood of a candidate model and explain how maximum likelihood estimates are found',
      'Scrieți log-verosimilitatea unui model candidat și explicați cum se obțin estimările de verosimilitate maximă'),
    T('Test a restricted model against a larger one with the likelihood-ratio test, and know when the test does not apply',
      'Testați un model restrîns față de unul mai larg cu testul raportului de verosimilitate și știți cînd testul nu se aplică'),
    T('Compute AIC, BIC, $\\Delta$AIC and Akaike weights and interpret them', 'Calculați AIC, BIC, $\\Delta$AIC și ponderile Akaike și interpretați-le'),
    T('Run and read KS, Cramér--von Mises and Anderson--Darling tests, PP and QQ plots, and explain why KS is weak in the tails',
      'Aplicați și interpretați testele KS, Cramér--von Mises și Anderson--Darling, graficele PP și QQ, și explicați de ce KS este slab în cozi'),
    T('Judge a model by its VaR 1\\% out of sample, recognise overfitting, and report model uncertainty',
      'Judecați un model după VaR 1\\% în afara eșantionului, recunoașteți supraajustarea și raportați incertitudinea de model')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~16 (Value at Risk and backtesting) and Ch.~18 (statistics of extreme risks); exercises: \\refFHHex',
       'Manual: \\refFHH, cap.~16 (Value at Risk și backtesting) și cap.~18 (statistica riscurilor extreme); exerciții: \\refFHHex'),
     [T('distributional properties of returns: \\refTsay, Ch.~1', 'proprietățile distribuției randamentelor: \\refTsay, cap.~1')]),
    T('Model selection: \\refBAbook; short version \\refBA', 'Selecția modelului: \\refBAbook; versiunea scurtă \\refBA'),
    T('Goodness of fit: \\refStephens; the original papers \\refAD, \\refLilliefors', 'Concordanța: \\refStephens; lucrările originale \\refAD, \\refLilliefors'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_06}',
       'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_06}'),
     [T('ported from the Quantlets SFESumm and SFEVaRqqplot; each chart links to the Quantlet that draws it',
        'adaptate după Quantlet-urile SFESumm și SFEVaRqqplot; fiecare grafic are un link către Quantlet-ul care îl generează')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter6_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter6_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}',
      'Curs video: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}')))

# =============================================================================
# 1. CANDIDAȚII
# =============================================================================
D.section('The Candidates', 'Candidații')

D.frame(T('``All Models Are Wrong\'\'', '„Toate modelele sînt greșite”'), cols(items(
    (T('George Box: a statistician should look for a model that is useful, not one that is true \\refBox',
       'George Box: un statistician caută un model util, nu unul adevărat \\refBox'),
     [T('``all models are wrong, but some are useful\'\' is the short form of his argument', '„toate modelele sînt greșite, dar unele sînt utile” este forma scurtă a argumentului său')]),
    (T('For returns, ``useful\'\' depends on the purpose', 'Pentru randamente, „util” depinde de scop'),
     [T('pricing and portfolio choice: the body of the distribution (typical days)', 'evaluarea activelor și alegerea portofoliului: corpul distribuției (zilele obișnuite)'),
      T('risk management: the left tail (the worst 1\\% of days)', 'managementul riscului: coada stîngă (cele mai slabe zile, 1\\% din total)')]),
    T('This chapter: a toolbox to compare models, and the habit of asking ``useful for what?\'\'',
      'Acest capitol: instrumentele de comparare a modelelor și obiceiul de a întreba „util pentru ce?”')),
    pic('ch6_box.jpg', 'George E. P. Box (1919--2013)', 'George E. P. Box (1919--2013)',
        'https://commons.wikimedia.org/wiki/File:GeorgeEPBox.jpg',
        'Photo: DavidMCEddy (2011); CC BY-SA 3.0; Wikimedia Commons', 'Foto: DavidMCEddy (2011); CC BY-SA 3.0; Wikimedia Commons',
        h='0.46\\textheight'), wl='0.62', wr='0.34'))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Seven Candidate Distributions', 'Șapte distribuții candidate'), table(
    'l' + TB + 'p{0.8cm}' + TB + 'p{3.2cm}' + TB + 'p{2.4cm}' + TB + 'p{2.6cm}',
    T('\\textbf{Model} & \\textbf{$k$} & \\textbf{Tails} & \\textbf{Skewness} & \\textbf{Reference}',
      '\\textbf{Model} & \\textbf{$k$} & \\textbf{Cozi} & \\textbf{Asimetrie} & \\textbf{Referință}'),
    [T('Normal & 2 & thin, $e^{-x^2/2}$ & no & Chapter 2', 'Normală & 2 & subțiri, $e^{-x^2/2}$ & nu & Capitolul 2'),
     T('Student-t & 3 & power, $x^{-\\nu}$ & no & Chapter 2', 'Student-t & 3 & putere, $x^{-\\nu}$ & nu & Capitolul 2'),
     T('skewed-t & 4 & power, $x^{-\\nu}$ & yes ($\\lambda$) & \\refHansen', 'skewed-t & 4 & putere, $x^{-\\nu}$ & da ($\\lambda$) & \\refHansen'),
     T('GED & 3 & $e^{-|x|^\\beta}$, heavy if $\\beta < 2$ & no & \\refNelson', 'GED & 3 & $e^{-|x|^\\beta}$, groase dacă $\\beta < 2$ & nu & \\refNelson'),
     T('Normal mixture & 5 & thin, but a wide component & yes & \\refKon', 'amestec Normal & 5 & subțiri, dar o componentă largă & da & \\refKon'),
     T('NIG & 4 & semi-heavy, $|x|^{-3/2}e^{-c|x|}$ & yes & \\refBN', 'NIG & 4 & semigroase, $|x|^{-3/2}e^{-c|x|}$ & da & \\refBN'),
     T('stable & 4 & power, $x^{-\\alpha}$, $\\alpha < 2$ & yes ($\\beta$) & Chapter 3', 'stabilă & 4 & putere, $x^{-\\alpha}$, $\\alpha < 2$ & da ($\\beta$) & Capitolul 3')],
    size='scriptsize') + items(
    T('$k$: number of parameters; GED: generalised error distribution; NIG: normal inverse Gaussian, a member of the generalised hyperbolic family \\refEK',
      '$k$: numărul de parametri; GED: generalised error distribution (distribuția erorii generalizate); NIG: normal inverse Gaussian (Normală invers Gaussiană), din familia hiperbolică generalizată \\refEK'),
    T('Normal mixture: with probability $w$ a calm-day $N(\\mu_1, \\sigma_1^2)$, otherwise a turbulent-day $N(\\mu_2, \\sigma_2^2)$',
      'Amestecul Normal: cu probabilitatea $w$, o zi calmă $N(\\mu_1, \\sigma_1^2)$, altfel o zi agitată $N(\\mu_2, \\sigma_2^2)$')), size='footnotesize')

chart(T('The Candidates Side by Side', 'Candidații comparați'), 'sfm_ch6_candidates', 'SFM_ch6_candidates', [
    T('All with mean 0 and variance 1 (stable: the same interquartile range as the Normal law); right panel on a log scale',
      'Toate au media 0 și varianța 1 (distribuția stabilă: același interval intercuartilic ca distribuția Normală); panoul din dreapta folosește scara logaritmică'),
    T('Near the centre they look alike; in the tails they differ by orders of magnitude', 'În centru arată aproape la fel; în cozi diferă cu cîteva ordine de mărime')], h='0.58\\textheight')

D.frame(T('Same Variance, Very Different Tails', 'Aceeași varianță, cozi foarte diferite'), items(
    (T('$P(X < -4)$, a move of four standard deviations, under each candidate of the chart:', '$P(X < -4)$, o mișcare de patru abateri standard, pentru fiecare candidat din grafic:'),
     [T('Normal $@{ca.Normal}$; GED $@{ca.GED}$; Student-t $@{ca.Student-t}$; Normal mixture $@{ca.Mixture}$',
        'Normală $@{ca.Normal}$; GED $@{ca.GED}$; Student-t $@{ca.Student-t}$; amestec Normal $@{ca.Mixture}$'),
      T('NIG $@{ca.NIG}$; skewed-t $@{ca.Skewed-t}$; stable $@{ca.Stable}$', 'NIG $@{ca.NIG}$; skewed-t $@{ca.Skewed-t}$; stabilă $@{ca.Stable}$')]),
    T('The Student-t gives this loss about $@{ca.ratio.t}$ times the Normal probability', 'Distribuția Student-t atribuie acestei pierderi o probabilitate de circa $@{ca.ratio.t}$ de ori mai mare decît distribuția Normală'),
    T('A risk measure such as VaR 1\\% lives exactly where the candidates disagree', 'O măsură de risc precum VaR 1\\% se află exact în zona în care candidații diferă'),
    T('\\textbf{Question for the room}: if all fit the centre well, why not keep the simplest one?', '\\textbf{Întrebare pentru sală}: dacă toate se potrivesc bine în centru, de ce nu îl păstrăm pe cel mai simplu?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('the simplest one (Normal) is wrong precisely in the region that drives capital and margin requirements',
        'cel mai simplu model (distribuția Normală) greșește exact în zona care determină cerințele de capital și de marjă')])))

chart(T('What Every Model Must Reproduce: the SFESumm Statistics', 'Statisticile SFESumm: reperele oricărui model'), 'sfm_ch6_dax_monthly', 'SFM_ch6_summary', [
    T('Quantlet SFESumm: DAX monthly log returns between first trading days, July 2004 -- May 2014 ($n = @{dm.book.n}$), extended here to @{dm.full.last}',
      'Quantlet-ul SFESumm: randamentele logaritmice lunare DAX între primele zile de tranzacționare, iulie 2004 -- mai 2014 ($n = @{dm.book.n}$), extinse aici pînă la @{dm.full.last}'),
    T('Even monthly returns are skewed and heavy-tailed: the next slide gives the numbers', 'Chiar și randamentele lunare sînt asimetrice și au cozi groase: cifrele sînt pe slide-ul următor')], h='0.56\\textheight')

D.frame(T('Summary Statistics: Monthly DAX and Daily Returns', 'Statistici descriptive: DAX lunar și randamente zilnice'), table(
    'lrrrrrrr', T('& $n$ & min & max & st.\\ dev. & ann.\\ vol. & skew. & kurt.', '& $n$ & min & max & ab.\\ std. & vol.\\ anuală & asim. & boltire'),
    ['DAX 2004--2014 & @{dm.book.n} & $@{dm.book.min}$ & $@{dm.book.max}$ & $@{dm.book.sd}$ & $@{dm.book.ann_vol}$ & $@{dm.book.skew}$ & $@{dm.book.kurt}$',
     'DAX 2004--@{y1} & @{dm.full.n} & $@{dm.full.min}$ & $@{dm.full.max}$ & $@{dm.full.sd}$ & $@{dm.full.ann_vol}$ & $@{dm.full.skew}$ & $@{dm.full.kurt}$'],
    size='footnotesize') + items(
    (T('Monthly, in \\%; st.\\ dev.: standard deviation $\\sigma$; ann.\\ vol.: $\\sigma\\sqrt{12}$', 'Lunar, în \\%; ab.\\ std.: abaterea standard $\\sigma$; vol.\\ anuală: $\\sigma\\sqrt{12}$'),
     [T('skewness $E(X - \\mu)^3/\\sigma^3$ and kurtosis $E(X - \\mu)^4/\\sigma^4$, with $\\mu$ the mean; 0 and 3 for the Normal distribution', 'asimetria $E(X - \\mu)^3/\\sigma^3$ și boltirea (kurtosis) $E(X - \\mu)^4/\\sigma^4$, cu $\\mu$ media; 0 și 3 pentru distribuția Normală'),
      T('Jarque--Bera test of Normality (Chapter 2): $p$-value $@{dm.full.jbp}$', 'testul de normalitate Jarque--Bera (Capitolul 2): p-value $@{dm.full.jbp}$')]),
    (T('Daily log returns since 2000 (Bitcoin since 2014, BVB stocks since 2010): kurtosis', 'Randamente logaritmice zilnice din 2000 (Bitcoin din 2014, acțiunile BVB din 2010): boltirea'),
     [T('BET $@{ds.bet.kurt}$, S\\&P 500 $@{ds.sp500.kurt}$, DAX $@{ds.dax.kurt}$, Bitcoin $@{ds.btc.kurt}$, TLV $@{ds.tlv.kurt}$, SNP $@{ds.snp.kurt}$, BRD $@{ds.brd.kurt}$',
        'BET $@{ds.bet.kurt}$, S\\&P 500 $@{ds.sp500.kurt}$, DAX $@{ds.dax.kurt}$, Bitcoin $@{ds.btc.kurt}$, TLV $@{ds.tlv.kurt}$, SNP $@{ds.snp.kurt}$, BRD $@{ds.brd.kurt}$'),
      T('skewness: S\\&P 500 $@{ds.sp500.skew}$, BET $@{ds.bet.skew}$, Bitcoin $@{ds.btc.skew}$', 'asimetria: S\\&P 500 $@{ds.sp500.skew}$, BET $@{ds.bet.skew}$, Bitcoin $@{ds.btc.skew}$')])) + ql('SFM_ch6_summary'), size='footnotesize')

D.frame(T('Data Before Models: Two Bad Days', 'Datele înaintea modelelor: două zile greșite'), items(
    (T('Banca Transilvania (TLV), adjusted close from EODHD: log returns of $@{tlv.bad1}\\%$ on 30 May 2016 and $@{tlv.bad2}\\%$ on 31 May 2016',
       'Banca Transilvania (TLV), prețul ajustat de la EODHD: randamente logaritmice de $@{tlv.bad1}\\%$ pe 30 mai 2016 și $@{tlv.bad2}\\%$ pe 31 mai 2016'),
     [T('the adjustment for the bonus shares is applied one day late (Chapter 2): a fake crash followed by a fake rally',
        'ajustarea pentru acțiunile gratuite este aplicată cu o zi întîrziere (Capitolul 2): un crah fals urmat de o creștere falsă')]),
    (T('Effect of the two days on the fitted model', 'Efectul celor două zile asupra modelului estimat'),
     [T('kurtosis $@{tlv.raw.kurt}$ with them, $@{tlv.clean.kurt}$ without; Student-t $\\hat\\nu$: $@{tlv.raw.nu}$ vs $@{tlv.clean.nu}$',
        'boltirea $@{tlv.raw.kurt}$ cu ele, $@{tlv.clean.kurt}$ fără; $\\hat\\nu$ Student-t: $@{tlv.raw.nu}$ față de $@{tlv.clean.nu}$'),
      T('the AIC ranking does not change here (best: @{tlv.clean.best}), but the moments do', 'clasamentul AIC nu se schimbă aici (cel mai bun: @{tlv.clean.best}), dar momentele se schimbă')]),
    T('Rule: check the largest returns against the close and the news before any model selection; the two days are removed in the rest of the chapter',
      'Regula: înainte de orice selecție de model, comparați cele mai mari randamente cu prețurile de închidere și cu știrile; în restul capitolului, cele două zile sînt eliminate')))

# =============================================================================
# 2. VEROSIMILITATEA MAXIMĂ
# =============================================================================
D.section('Maximum Likelihood, Recalled', 'Recapitulare: metoda verosimilității maxime')

D.frame(T('Likelihood and Log-Likelihood', 'Verosimilitate și log-verosimilitate'), items(
    (T('Data $r_1, \\dots, r_n$, assumed i.i.d.\\ (independent, identically distributed) with density $f(r; \\theta)$', 'Datele $r_1, \\dots, r_n$, presupuse i.i.d.\\ (independente și identic distribuite), cu densitatea $f(r; \\theta)$'),
     [T('$\\theta$: the vector of model parameters (for example $\\mu, \\sigma$); $f$: the density of the model', '$\\theta$: vectorul parametrilor modelului (de exemplu $\\mu, \\sigma$); $f$: densitatea modelului'),
      T('\\textbf{likelihood}: $L(\\theta) = \\prod_{t=1}^n f(r_t; \\theta)$, the probability density of the observed sample as a function of $\\theta$',
        '\\textbf{verosimilitatea}: $L(\\theta) = \\prod_{t=1}^n f(r_t; \\theta)$, densitatea eșantionului observat, ca funcție de $\\theta$'),
      T('\\textbf{log-likelihood}: $\\ell(\\theta) = \\sum_{t=1}^n \\ln f(r_t; \\theta)$; sums are easier than products', '\\textbf{log-verosimilitatea}: $\\ell(\\theta) = \\sum_{t=1}^n \\ln f(r_t; \\theta)$; sumele se manipulează mai ușor decît produsele')]),
    (T('\\textbf{Maximum likelihood (ML) estimate}: $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$', '\\textbf{Estimarea de verosimilitate maximă (ML, maximum likelihood)}: $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$'),
     [T('$\\ell(\\hat\\theta)$, the maximised log-likelihood, is the raw material of every comparison in this chapter',
        '$\\ell(\\hat\\theta)$, log-verosimilitatea maximizată, stă la baza tuturor comparațiilor din acest capitol')]),
    (T('Large $n$: $\\hat\\theta$ is approximately Normal around the true $\\theta$', 'Pentru $n$ mare, $\\hat\\theta$ are o distribuție aproximativ Normală în jurul valorii adevărate $\\theta$'),
     [T('covariance $\\approx [-\\partial^2\\ell/\\partial\\theta\\,\\partial\\theta^\\top]^{-1}$, the inverse of the observed information', 'covarianța $\\approx [-\\partial^2\\ell/\\partial\\theta\\,\\partial\\theta^\\top]^{-1}$, inversa informației observate'),
      T('a sharply curved $\\ell$ (large second derivative) means a precise estimate', 'o funcție $\\ell$ puternic curbată (derivata a doua mare) înseamnă o estimare precisă')])), 'footnotesize')

D.frame(T('Worked Example: the Normal Model, Step by Step', 'Exemplu lucrat: modelul Normal, pas cu pas'), items(
    (T('$\\ell(\\mu, \\sigma) = -\\dfrac{n}{2}\\ln(2\\pi\\sigma^2) - \\dfrac{1}{2\\sigma^2}\\sum_t (r_t - \\mu)^2$', '$\\ell(\\mu, \\sigma) = -\\dfrac{n}{2}\\ln(2\\pi\\sigma^2) - \\dfrac{1}{2\\sigma^2}\\sum_t (r_t - \\mu)^2$'),
     [T('setting the derivatives to zero: $\\hat\\mu = \\bar r$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t (r_t - \\bar r)^2$ (divisor $n$, not $n - 1$)',
        'anulînd derivatele: $\\hat\\mu = \\bar r$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t (r_t - \\bar r)^2$ (numitorul $n$, nu $n - 1$)'),
      T('plugging back: $\\ell(\\hat\\theta) = -\\dfrac{n}{2}\\big[\\ln(2\\pi\\hat\\sigma^2) + 1\\big]$', 'înlocuind în $\\ell$: $\\ell(\\hat\\theta) = -\\dfrac{n}{2}\\big[\\ln(2\\pi\\hat\\sigma^2) + 1\\big]$')]),
    (T('S\\&P 500, daily log returns in \\%, @{y0.sp500}--@{y1}: $n = @{n.sp500}$, $\\hat\\mu = @{sp.mu}$, $\\hat\\sigma = @{sp.sig}$',
       'S\\&P 500, randamente logaritmice zilnice în \\%, @{y0.sp500}--@{y1}: $n = @{n.sp500}$, $\\hat\\mu = @{sp.mu}$, $\\hat\\sigma = @{sp.sig}$'),
     [T('$\\ln(2\\pi\\hat\\sigma^2) = @{sp.ln2pis}$, so $\\ell(\\hat\\theta) = -\\frac{@{n.sp500}}{2}(@{sp.ln2pis} + 1) = @{sp.llN}$',
        '$\\ln(2\\pi\\hat\\sigma^2) = @{sp.ln2pis}$, deci $\\ell(\\hat\\theta) = -\\frac{@{n.sp500}}{2}(@{sp.ln2pis} + 1) = @{sp.llN}$')]),
    T('Only the Normal model has a closed form; the others are maximised numerically', 'Doar modelul Normal are soluție analitică; celelalte se maximizează numeric')))

D.frame(T('Numerical ML for the Other Candidates (S\\&P 500)', 'Estimarea ML numerică pentru ceilalți candidați (S\\&P 500)'), table(
    'l' + TB + 'p{6.6cm}r', T('\\textbf{Model} & \\textbf{Estimates} & $\\ell(\\hat\\theta)$', '\\textbf{Model} & \\textbf{Estimări} & $\\ell(\\hat\\theta)$'),
    [T('Normal', 'Normală') + ' & $\\hat\\mu = @{sp.mu}$, $\\hat\\sigma = @{sp.sig}$ & $@{ll.sp500.Normal}$',
     'Student-t & $\\hat\\nu = @{sp.t.nu}$, ' + T('location', 'locație') + ' $@{sp.t.loc}$, ' + T('scale', 'scală') + ' $@{sp.t.sc}$ & $@{ll.sp500.Student-t}$',
     'skewed-t & $\\hat\\nu = @{sp.skt.nu}$, $\\hat\\lambda = @{sp.skt.lam}$, $\\hat\\mu = @{sp.skt.mu}$, $\\hat\\sigma = @{sp.skt.sig}$ & $@{ll.sp500.Skewed-t}$',
     'GED & $\\hat\\beta = @{sp.ged.b}$, ' + T('location', 'locație') + ' $@{sp.ged.loc}$, ' + T('scale', 'scală') + ' $@{sp.ged.sc}$ & $@{ll.sp500.GED}$',
     T('Normal mixture', 'amestec Normal') + ' & $\\hat w = @{sp.mix.w1}\\%$: $N(@{sp.mix.m1}, @{sp.mix.s1}^2)$; $@{sp.mix.w2}\\%$: $N(@{sp.mix.m2}, @{sp.mix.s2}^2)$ & $@{ll.sp500.Mixture}$',
     'NIG & $\\hat a = @{sp.nig.a}$, $\\hat b = @{sp.nig.b}$, ' + T('location', 'locație') + ' $@{sp.nig.loc}$, ' + T('scale', 'scală') + ' $@{sp.nig.sc}$ & $@{ll.sp500.NIG}$',
     T('stable', 'stabilă') + ' & $\\hat\\alpha = @{sp.st.a}$, $\\hat\\beta = @{sp.st.b}$, $\\hat\\gamma = @{sp.st.g}$, $\\hat\\delta_0 = @{sp.st.d}$ & $@{ll.sp500.Stable}$'],
    size='scriptsize') + items(
    T('Optimiser: Nelder--Mead on transformed parameters (for example $\\ln\\sigma$, $\\ln(\\nu - 2)$); the Normal mixture by the EM (expectation--maximisation) algorithm; the stable density by FFT (Chapter 3)',
      'Algoritmul de optimizare: Nelder--Mead aplicat parametrilor transformați (de exemplu $\\ln\\sigma$, $\\ln(\\nu - 2)$); amestecul Normal prin algoritmul EM (expectation--maximisation); densitatea stabilă prin FFT (Capitolul 3)'),
    T('Skewed-t: $\\mu$ and $\\sigma$ are the mean and the standard deviation (Hansen\'s parameterisation)', 'Skewed-t: $\\mu$ și $\\sigma$ sînt media și abaterea standard (parametrizarea lui Hansen)')) + ql('SFM_ch6_ml_fits'), size='footnotesize')

D.frame(T('Reading the Estimates', 'Interpretarea estimărilor'), items(
    (T('Student-t $\\hat\\nu = @{sp.t.nu}$: tails like $x^{-@{sp.t.nu}}$; the variance exists ($\\nu > 2$), the kurtosis does not ($\\nu < 4$)',
       'Student-t $\\hat\\nu = @{sp.t.nu}$: cozi de tip $x^{-@{sp.t.nu}}$; varianța există ($\\nu > 2$), boltirea nu există ($\\nu < 4$)'),
     [T('the sample kurtosis $@{ds.sp500.kurt}$ estimates a quantity that the fitted model says is infinite',
        'boltirea de selecție $@{ds.sp500.kurt}$ estimează o mărime care, conform modelului estimat, este infinită')]),
    T('Skewed-t $\\hat\\lambda = @{sp.skt.lam} < 0$: a slightly longer left tail (crashes)', 'Skewed-t $\\hat\\lambda = @{sp.skt.lam} < 0$: o coadă stîngă puțin mai lungă (crahuri)'),
    T('GED $\\hat\\beta = @{sp.ged.b}$: below 2 (Normal) and even below 1 (Laplace): a sharp peak', 'GED $\\hat\\beta = @{sp.ged.b}$: sub 2 (Normală) și chiar sub 1 (Laplace): un vîrf ascuțit'),
    T('Normal mixture: $@{sp.mix.w1}\\%$ calm days with standard deviation $@{sp.mix.s1}\\%$, $@{sp.mix.w2}\\%$ turbulent days with $@{sp.mix.s2}\\%$ and a negative mean',
      'Amestecul Normal: $@{sp.mix.w1}\\%$ zile calme cu abaterea standard $@{sp.mix.s1}\\%$, $@{sp.mix.w2}\\%$ zile agitate cu $@{sp.mix.s2}\\%$ și medie negativă'),
    T('Stable $\\hat\\alpha = @{sp.st.a}$ (Chapter 3): heavier tails than any other candidate', 'Distribuția stabilă, $\\hat\\alpha = @{sp.st.a}$ (Capitolul 3): cozi mai groase decît orice alt candidat')))

chart(T('How Precise Is $\\hat\\nu$? The Profile Log-Likelihood', 'Precizia lui $\\hat\\nu$: log-verosimilitatea profil'), 'sfm_ch6_profile_nu', 'SFM_ch6_ml_fits', [
    T('For each $\\nu$, maximise $\\ell$ over location and scale; the curve falls by $@{pr.l5}$ at $\\nu = 5$',
      'Pentru fiecare $\\nu$, maximizăm $\\ell$ în raport cu locația și scala; la $\\nu = 5$, curba scade cu $@{pr.l5}$'),
    T('95\\% interval: all $\\nu$ with $2[\\ell(\\hat\\nu) - \\ell(\\nu)] \\le @{chi95}$, here $[@{pr.lo}, @{pr.hi}]$ around $\\hat\\nu = @{pr.nu}$: the tails are pinned down well',
      'Intervalul de încredere de 95\\%: toate valorile $\\nu$ pentru care $2[\\ell(\\hat\\nu) - \\ell(\\nu)] \\le @{chi95}$, aici $[@{pr.lo}; @{pr.hi}]$ în jurul lui $\\hat\\nu = @{pr.nu}$: grosimea cozilor este estimată precis')], h='0.56\\textheight')

D.frame(T('Pitfalls of Numerical ML', 'Capcanele ML numerice'), items(
    (T('\\textbf{Local maxima}: the optimiser can stop on a hill that is not the highest', '\\textbf{Maxime locale}: optimizatorul se poate opri pe un deal care nu este cel mai înalt'),
     [T('remedy: several starting values (the mixtures in this chapter use 4 to 12 starts)', 'remediu: mai multe valori inițiale (amestecurile din acest capitol folosesc între 4 și 12 puncte de pornire)')]),
    (T('\\textbf{Boundaries}: $\\nu \\to 2$, a mixture weight $\\to 0$, $|\\lambda| \\to 1$', '\\textbf{Frontiere}: $\\nu \\to 2$, o pondere a amestecului $\\to 0$, $|\\lambda| \\to 1$'),
     [T('the usual standard errors and tests are not valid at a boundary (next section)', 'erorile standard și testele obișnuite nu sînt valide la frontieră (secțiunea următoare)')]),
    (T('\\textbf{Parameterisation}: the same law, different parameters', '\\textbf{Parametrizarea}: aceeași distribuție, parametrizări diferite'),
     [T('scipy\'s Student-t scale is not the standard deviation: $\\mathrm{sd} = \\mathrm{scale}\\sqrt{\\nu/(\\nu - 2)}$', 'scala Student-t din scipy nu este abaterea standard: $\\mathrm{sd} = \\mathrm{scale}\\sqrt{\\nu/(\\nu - 2)}$')]),
    T('\\textbf{Check}: compare $\\ell(\\hat\\theta)$ of nested models; a larger model can never have a lower maximum',
      '\\textbf{Verificare}: comparați $\\ell(\\hat\\theta)$ pentru modelele imbricate; un model mai general nu poate avea un maxim mai mic')))

# =============================================================================
# 3. TESTUL RAPORTULUI DE VEROSIMILITATE
# =============================================================================
D.section('Nested Models: the Likelihood-Ratio Test', 'Modele imbricate: testul raportului de verosimilitate')

D.frame(T('Nested Models', 'Modele imbricate'), items(
    (T('Model 0 is \\textbf{nested} in model 1 if model 0 is model 1 with some parameters fixed', 'Modelul 0 este \\textbf{imbricat} în modelul 1 dacă modelul 0 este modelul 1 cu unii parametri fixați'),
     [T('Normal inside GED: $\\beta = 2$', 'Normală în GED: $\\beta = 2$'),
      T('Student-t inside skewed-t: $\\lambda = 0$', 'Student-t în skewed-t: $\\lambda = 0$'),
      T('Normal inside Student-t: $\\nu = \\infty$, i.e.\\ $1/\\nu = 0$, on the \\textbf{edge} of the parameter space', 'Normală în Student-t: $\\nu = \\infty$, adică $1/\\nu = 0$, la \\textbf{marginea} spațiului parametrilor')]),
    (T('Not nested: Student-t and GED, Student-t and stable, skewed-t and NIG', 'Neimbricate: Student-t și GED, Student-t și stabilă, skewed-t și NIG'),
     [T('neither is a special case of the other: they need other tools (Vuong test, AIC)', 'niciunul nu este caz particular al celuilalt: sînt necesare alte instrumente (testul Vuong, AIC)')]),
    T('Nested: $\\ell_1(\\hat\\theta_1) \\ge \\ell_0(\\hat\\theta_0)$ always; the question is whether the gain is larger than chance',
      'Modele imbricate: întotdeauna $\\ell_1(\\hat\\theta_1) \\ge \\ell_0(\\hat\\theta_0)$; întrebarea este dacă cîștigul depășește ce s-ar obține din întîmplare')))

D.frame(T('The Likelihood-Ratio Test', 'Testul raportului de verosimilitate'), items(
    (T('$H_0$: model 0 (restricted) is true; $H_1$: model 1 (larger)', '$H_0$: modelul 0 (restrîns) este adevărat; $H_1$: modelul 1 (mai general)'),
     [T('$LR = 2\\,[\\ell_1(\\hat\\theta_1) - \\ell_0(\\hat\\theta_0)] \\ge 0$: twice the gain in maximised log-likelihood', '$LR = 2\\,[\\ell_1(\\hat\\theta_1) - \\ell_0(\\hat\\theta_0)] \\ge 0$: de două ori cîștigul de log-verosimilitate maximizată'),
      T('$\\ell_0, \\ell_1$: the log-likelihoods of the two models; $k_0, k_1$: their numbers of parameters', '$\\ell_0, \\ell_1$: log-verosimilitățile celor două modele; $k_0, k_1$: numerele lor de parametri')]),
    (T('\\textbf{Wilks\' theorem} (\\refWilks): under $H_0$, $LR \\approx \\chi^2(d)$, with $d = k_1 - k_0$ restrictions', '\\textbf{Teorema lui Wilks} (\\refWilks): sub $H_0$, $LR \\approx \\chi^2(d)$, cu $d = k_1 - k_0$ restricții'),
     [T('$\\chi^2(d)$: the chi-square distribution with $d$ degrees of freedom', '$\\chi^2(d)$: distribuția hi-pătrat cu $d$ grade de libertate'),
      T('valid if the restricted values are inside the parameter space', 'valabilă dacă valorile fixate se află în interiorul spațiului parametrilor')]),
    (T('Decision at 5\\%: reject $H_0$ if $LR > \\chi^2_{0.95}(d)$', 'Decizia la 5\\%: respingem $H_0$ dacă $LR > \\chi^2_{0.95}(d)$'),
     [T('$\\chi^2_{0.95}(1) = @{chi95}$, $\\chi^2_{0.95}(2) = @{chi2_95}$', '$\\chi^2_{0.95}(1) = @{chi95}$; $\\chi^2_{0.95}(2) = @{chi2_95}$')]),
    T('Intuition: twice the log of how many times more likely the data are under the larger model', 'Intuiția: de două ori logaritmul raportului dintre verosimilitățile datelor sub cele două modele')))

D.frame(T('Worked Example: Is the Skewness Parameter Needed?', 'Exemplu lucrat: este necesar parametrul de asimetrie?'), items(
    (T('S\\&P 500: $\\ell$(Student-t) $= @{ll.sp500.Student-t}$, $\\ell$(skewed-t) $= @{ll.sp500.Skewed-t}$', 'S\\&P 500: $\\ell$(Student-t) $= @{ll.sp500.Student-t}$, $\\ell$(skewed-t) $= @{ll.sp500.Skewed-t}$'),
     [T('$LR = 2(@{ll.sp500.Skewed-t} - (@{ll.sp500.Student-t})) = @{lr.tskt.sp500} > @{chi95}$; $p = @{lrp.tskt.sp500}$: reject $\\lambda = 0$',
        '$LR = 2(@{ll.sp500.Skewed-t} - (@{ll.sp500.Student-t})) = @{lr.tskt.sp500} > @{chi95}$; $p = @{lrp.tskt.sp500}$: respingem $\\lambda = 0$')]),
    (T('The same test elsewhere', 'Același test pentru alte serii'),
     [T('DAX: $LR = @{lr.tskt.dax}$, $p = @{lrp.tskt.dax}$: skewness needed', 'DAX: $LR = @{lr.tskt.dax}$, $p = @{lrp.tskt.dax}$: asimetria este necesară'),
      T('BET: $LR = @{lr.tskt.bet}$, $p = @{lrp.tskt.bet}$; Bitcoin: $LR = @{lr.tskt.btc}$, $p = @{lrp.tskt.btc}$: no evidence of skewness',
        'BET: $LR = @{lr.tskt.bet}$, $p = @{lrp.tskt.bet}$; Bitcoin: $LR = @{lr.tskt.btc}$, $p = @{lrp.tskt.btc}$: nicio dovadă de asimetrie')]),
    T('Normal inside GED ($\\beta = 2$): $LR$ between @{lr.nged.btc} (Bitcoin) and @{lr.nged.bet} (BET); the Normal model is rejected everywhere',
      'Normală în GED ($\\beta = 2$): $LR$ între @{lr.nged.btc} (Bitcoin) și @{lr.nged.bet} (BET); modelul Normal este respins peste tot'),
    T('Interpretation: the two developed indices have a longer left tail; the BET and Bitcoin do not', 'Interpretare: cei doi indici dezvoltați au o coadă stîngă mai lungă; BET și Bitcoin nu')) + ql('SFM_ch6_lr_aic'))

D.frame(T('A Boundary Case: Normal inside Student-t', 'Un caz de frontieră: Normală în Student-t'), items(
    (T('$H_0$: $1/\\nu = 0$ is on the edge of the space $1/\\nu \\ge 0$: Wilks\' theorem does not apply', '$H_0$: $1/\\nu = 0$ este pe marginea spațiului $1/\\nu \\ge 0$: teorema lui Wilks nu se aplică'),
     [T('\\refSL: the null law is a 50:50 mixture of $\\chi^2(0)$ (a point mass at 0) and $\\chi^2(1)$', '\\refSL: distribuția statisticii sub $H_0$ este un amestec 50:50 de $\\chi^2(0)$ (o masă de probabilitate concentrată în 0) și $\\chi^2(1)$'),
      T('5\\% critical value: $\\chi^2_{0.90}(1) = @{chi90}$ instead of $@{chi95}$; the naive $p$-value is twice the correct one',
        'valoarea critică la 5\\%: $\\chi^2_{0.90}(1) = @{chi90}$ în loc de $@{chi95}$; p-value-ul naiv este dublul celui corect')]),
    T('Real data: $LR$ = @{lr.nt.sp500} (S\\&P 500), @{lr.nt.btc} (Bitcoin): the Normal model is rejected with either critical value',
      'Date reale: $LR$ = @{lr.nt.sp500} (S\\&P 500), @{lr.nt.btc} (Bitcoin): modelul Normal este respins cu oricare valoare critică'),
    T('The same issue appears for a mixture weight $w = 0$ and for variance parameters equal to 0', 'Aceeași problemă apare pentru o pondere a amestecului $w = 0$ și pentru parametri de varianță egali cu 0'),
    T('\\textbf{What do you think?} Does the boundary matter when $LR$ is in the thousands?', '\\textbf{Ce credeți?} Contează frontiera cînd $LR$ este de ordinul miilor?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('no for the decision; yes for small samples and borderline $LR$ values, where it halves the $p$-value', 'nu pentru decizia de aici; contează însă în eșantioane mici și pentru valori $LR$ la limită, unde înjumătățește p-value-ul')])))

D.frame(T('Non-Nested Models: the Vuong Test', 'Modele neimbricate: testul Vuong'), table(
    'lrrr', T('& Student-t vs GED & Student-t vs stable & skewed-t vs NIG', '& Student-t față de GED & Student-t față de stabilă & skewed-t față de NIG'),
    [f'{NAMES[k]} & $@{{vu_t_ged.{k}}}$ ($@{{vu_t_gedp.{k}}}$) & $@{{vu_t_stable.{k}}}$ ($@{{vu_t_stablep.{k}}}$) & $@{{vu_skt_nig.{k}}}$ ($@{{vu_skt_nigp.{k}}}$)' for k in ASSETS],
    size='footnotesize') + items(
    (T('\\refVuong: $m_t = \\ln f_1(r_t) - \\ln f_2(r_t)$, $V = \\sqrt{n}\\,\\bar m / s_m \\approx N(0, 1)$ if both models are equally close to the truth',
      '\\refVuong: $m_t = \\ln f_1(r_t) - \\ln f_2(r_t)$, $V = \\sqrt{n}\\,\\bar m / s_m \\approx N(0, 1)$ dacă ambele modele sînt la fel de aproape de adevăr'),
     [T('$f_1, f_2$: the two fitted densities; $m_t$: by how much model 1 explains day $t$ better; $\\bar m$, $s_m$: mean and standard deviation of the $m_t$', '$f_1, f_2$: cele două densități estimate; $m_t$: cu cît explică modelul 1 mai bine ziua $t$; $\\bar m$, $s_m$: media și abaterea standard a valorilor $m_t$'),
      T('$V > 1.96$: the first model is closer; $V < -1.96$: the second; ($p$-value) in brackets', '$V > 1{,}96$: primul model este mai aproape; $V < -1{,}96$: al doilea; p-value-ul este în paranteză')]),
    T('Student-t beats the stable law everywhere; NIG beats the skewed-t on the S\\&P 500 and the DAX; Student-t vs GED splits by market',
      'Student-t este preferat distribuției stabile pe toate piețele; NIG este preferat distribuției skewed-t pentru S\\&P 500 și DAX; comparația Student-t--GED diferă de la o piață la alta')) + ql('SFM_ch6_lr_aic'), size='footnotesize')

# =============================================================================
# 4. CRITERII INFORMAȚIONALE
# =============================================================================
D.section('Information Criteria: AIC and BIC', 'Criterii informaționale: AIC și BIC')

D.frame(T('Akaike\'s Idea', 'Ideea lui Akaike'), cols(items(
    (T('Hirotugu Akaike asked: which model will predict \\textbf{new} data best?', 'Hirotugu Akaike și-a pus întrebarea: ce model va descrie cel mai bine date \\textbf{noi}?'),
     [T('the distance from the truth $g$ to a model $f$: the Kullback--Leibler divergence \\refKL', 'distanța de la distribuția adevărată $g$ la un model $f$: divergența Kullback--Leibler \\refKL'),
      T('$KL(g, f) = E_g[\\ln g(R) - \\ln f(R)] \\ge 0$, zero only if $f = g$', '$KL(g, f) = E_g[\\ln g(R) - \\ln f(R)] \\ge 0$, zero doar dacă $f = g$'),
      T('$R$: a new return drawn from the truth $g$; $E_g$: the mean under $g$', '$R$: un randament nou, generat de distribuția adevărată $g$; $E_g$: media calculată sub $g$')]),
    (T('$\\ell(\\hat\\theta)$ is too optimistic: the same data choose $\\hat\\theta$ and judge it', '$\\ell(\\hat\\theta)$ este prea optimistă: aceleași date servesc și la estimarea lui $\\hat\\theta$, și la evaluarea lui'),
     [T('the optimism is about $k$, the number of parameters \\refAkaike', 'supraestimarea este, în medie, de circa $k$, numărul de parametri \\refAkaike')]),
    T('Result: a penalty of 2 per parameter corrects the optimism; the criterion, AIC, is on the next slide',
      'Rezultatul: o penalizare de 2 pentru fiecare parametru corectează supraestimarea; criteriul obținut, AIC, este pe slide-ul următor')),
    pic('ch6_akaike.jpg', 'Hirotugu Akaike (1927--2009)', 'Hirotugu Akaike (1927--2009)',
        'https://commons.wikimedia.org/wiki/File:Akaike.jpg',
        'Photo: The Institute of Statistical Mathematics; CC BY-SA 4.0; Wikimedia Commons',
        'Foto: The Institute of Statistical Mathematics; CC BY-SA 4.0; Wikimedia Commons', h='0.44\\textheight'), wl='0.64', wr='0.32'))

D.frame(T('AIC and BIC', 'AIC și BIC'), items(
    (T('\\textbf{AIC} (Akaike information criterion): $\\mathrm{AIC} = -2\\ell(\\hat\\theta) + 2k$', '\\textbf{AIC} (Akaike information criterion, criteriul informațional Akaike): $\\mathrm{AIC} = -2\\ell(\\hat\\theta) + 2k$'),
     [T('aim: the best predictive model; asymptotically equivalent to leave-one-out cross-validation \\refStone',
        'scopul: cel mai bun model predictiv; asimptotic echivalent cu validarea încrucișată leave-one-out \\refStone'),
      T('small samples: AICc $= \\mathrm{AIC} + 2k(k+1)/(n - k - 1)$ \\refHT', 'eșantioane mici: AICc $= \\mathrm{AIC} + 2k(k+1)/(n - k - 1)$ \\refHT')]),
    (T('\\textbf{BIC} (Bayesian information criterion) \\refSchwarz: $\\mathrm{BIC} = -2\\ell(\\hat\\theta) + k\\ln n$', '\\textbf{BIC} (Bayesian information criterion, criteriul informațional bayesian) \\refSchwarz: $\\mathrm{BIC} = -2\\ell(\\hat\\theta) + k\\ln n$'),
     [T('aim: the true model, if it is among the candidates (consistency)', 'scopul: modelul adevărat, dacă este printre candidați (consistență)'),
      T('penalty per parameter $\\ln n$: $@{lnn.sp}$ for $n = @{n.sp500}$, more than four times the AIC penalty 2', 'penalizarea pe parametru $\\ln n$: $@{lnn.sp}$ pentru $n = @{n.sp500}$, de peste patru ori mai mare decît penalizarea AIC, egală cu 2')]),
    T('Both: lower is better; only differences between models on the \\textbf{same data} mean anything', 'Pentru ambele criterii, valoarea mai mică este preferată; doar diferențele dintre modele estimate pe \\textbf{aceleași date} sînt interpretabile')))

D.frame(T('$\\Delta$AIC and Akaike Weights', '$\\Delta$AIC și ponderile Akaike'), items(
    (T('$\\Delta_i = \\mathrm{AIC}_i - \\min_j \\mathrm{AIC}_j$; rules of thumb of \\refBA:', '$\\Delta_i = \\mathrm{AIC}_i - \\min_j \\mathrm{AIC}_j$; regulile practice din \\refBA:'),
     [T('$\\Delta \\le 2$: substantial support; $4 \\le \\Delta \\le 7$: considerably less; $\\Delta > 10$: essentially none',
        '$\\Delta \\le 2$: sprijin substanțial; $4 \\le \\Delta \\le 7$: considerabil mai puțin; $\\Delta > 10$: practic niciunul')]),
    (T('\\textbf{Akaike weight}: $w_i = \\dfrac{e^{-\\Delta_i/2}}{\\sum_j e^{-\\Delta_j/2}}$', '\\textbf{Ponderea Akaike}: $w_i = \\dfrac{e^{-\\Delta_i/2}}{\\sum_j e^{-\\Delta_j/2}}$'),
     [T('the weight of evidence for model $i$ among the candidates; the weights sum to 1', 'ponderea dovezilor în favoarea modelului $i$ dintre candidați; suma ponderilor este 1'),
      T('two models with $\\Delta = 0$ and $\\Delta = 2$: $w = 1/(1 + e^{-1}) \\approx 0.73$ and $@{ex.d2.w}$', 'două modele cu $\\Delta = 0$ și $\\Delta = 2$: $w = 1/(1 + e^{-1}) \\approx 0{,}73$ și $@{ex.d2.w}$')]),
    T('\\textbf{What do you think?} Is $\\Delta = 2$ a ``significant\'\' difference?', '\\textbf{Ce credeți?} Este $\\Delta = 2$ o diferență „semnificativă”?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('no: AIC is not a test; a gap of 2 means both models remain plausible (the weights above)', 'nu: AIC nu este un test; o diferență de 2 înseamnă că ambele modele rămîn plauzibile (ponderile de mai sus)')])))


def icrow(k, m):
    lab = {'Mixture': T('Normal mixture', 'amestec Normal'), 'Normal': T('Normal', 'Normală'), 'Stable': T('stable', 'stabilă'), 'Skewed-t': 'skewed-t'}.get(m, m)
    return (f'{lab} & {F[k]["models"][m]["k"]} & $@{{ll.{k}.{m}}}$ & $@{{aic.{k}.{m}}}$ & $@{{bic.{k}.{m}}}$ & '
            f'$@{{daic1.{k}.{m}}}$ & $@{{dbic1.{k}.{m}}}$ & $@{{w.{k}.{m}}}$')


D.frame(T('Worked Example: Seven Models for the S\\&P 500', 'Exemplu lucrat: șapte modele pentru S\\&P 500'), table(
    'lrrrrrrr', '& $k$ & $\\ell(\\hat\\theta)$ & AIC & BIC & $\\Delta$AIC & $\\Delta$BIC & $w$', [icrow('sp500', m) for m in MODELS], size='scriptsize') + items(
    T('Example: NIG, $\\mathrm{AIC} = -2(@{ll.sp500.NIG}) + 2 \\cdot 4 = @{aic.sp500.NIG}$', 'Exemplu: NIG, $\\mathrm{AIC} = -2(@{ll.sp500.NIG}) + 2 \\cdot 4 = @{aic.sp500.NIG}$'),
    T('NIG wins on both criteria; the next model, skewed-t, has $\\Delta = @{daic1.sp500.Skewed-t}$, so $e^{-\\Delta/2} \\approx @{ex.e.skt.sci}$: weight $w \\approx 1$ for NIG',
      'NIG cîștigă pe ambele criterii; următorul model, skewed-t, are $\\Delta = @{daic1.sp500.Skewed-t}$, deci $e^{-\\Delta/2} \\approx @{ex.e.skt.sci}$: ponderea $w \\approx 1$ pentru NIG'),
    T('Normal: $\\Delta$AIC of @{daic.sp500.Normal}; the stable law: @{daic.sp500.Stable}', 'Modelul Normal: $\\Delta$AIC de @{daic.sp500.Normal}; distribuția stabilă: @{daic.sp500.Stable}')) + ql('SFM_ch6_lr_aic'), size='footnotesize')

chart(T('$\\Delta$AIC across Seven Series', '$\\Delta$AIC pentru șapte serii'), 'sfm_ch6_delta_aic', 'SFM_ch6_lr_aic', [
    T('Each row: one series; 0 marks the best model; darker cells: worse models (log colour scale)', 'Fiecare rînd: o serie; 0 marchează cel mai bun model; celule mai închise: modele mai slabe (scară logaritmică a culorii)')],
    h='0.62\\textheight')

D.frame(T('Reading the $\\Delta$AIC Map', 'Interpretarea hărții $\\Delta$AIC'), items(
    (T('No single winner: the best model depends on the market', 'Niciun cîștigător unic: cel mai bun model depinde de piață'),
     [T('indices (BET, S\\&P 500, DAX): NIG; Bitcoin: GED; BVB stocks: Student-t (TLV, SNP) and skewed-t (BRD)', 'indicii (BET, S\\&P 500, DAX): NIG; Bitcoin: GED; acțiunile BVB: Student-t (TLV, SNP) și skewed-t (BRD)')]),
    (T('Robust conclusions', 'Concluzii robuste'),
     [T('the Normal model is last everywhere, by @{daic.brd.Normal} (BRD) to @{daic.bet.Normal} (BET) points', 'modelul Normal este ultimul peste tot, cu $\\Delta$AIC între @{daic.brd.Normal} (BRD) și @{daic.bet.Normal} (BET)'),
      T('the stable law ($\\Delta$ from @{daic.brd.Stable} to @{daic.btc.Stable}) and the Normal mixture are never competitive', 'distribuția stabilă ($\\Delta$ de la @{daic.brd.Stable} la @{daic.btc.Stable}) și amestecul Normal nu sînt niciodată competitive')]),
    T('\\textbf{Question for the room}: for BRD, AIC prefers the skewed-t ($\\Delta$ of the Student-t: $@{daic1.brd.Student-t}$); which model does BIC prefer?',
      '\\textbf{Întrebare pentru sală}: pentru BRD, AIC preferă skewed-t ($\\Delta$ al Student-t: $@{daic1.brd.Student-t}$); ce model preferă BIC?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('the Student-t: one parameter less saves $\\ln n \\approx @{lnn.brd}$ BIC points, more than the gain in fit; with $\\Delta \\approx 1$ the two are practically tied',
        'Student-t: un parametru în minus reduce BIC cu $\\ln n \\approx @{lnn.brd}$, mai mult decît cîștigul de ajustare; cu $\\Delta \\approx 1$, cele două sînt practic la egalitate')])))

# =============================================================================
# 5. TESTE DE CONCORDANȚĂ
# =============================================================================
D.section('Goodness-of-Fit Tests', 'Teste de concordanță')

D.frame(T('From Pearson to the Empirical Distribution Function', 'De la Pearson la funcția de repartiție empirică'), cols(items(
    (T('\\refPearson: the first goodness-of-fit test, the $\\chi^2$ test', '\\refPearson: primul test de concordanță, testul $\\chi^2$'),
     [T('group the data into bins and compare observed with expected counts', 'grupăm datele în clase și comparăm frecvențele observate cu cele așteptate'),
      T('the result depends on the bins, and the tails hold few observations', 'rezultatul depinde de clase, iar cozile conțin puține observații')]),
    (T('\\textbf{Empirical distribution function} (EDF): $F_n(x) = \\frac{1}{n}\\#\\{t: r_t \\le x\\}$', '\\textbf{Funcția de repartiție empirică} (EDF, empirical distribution function): $F_n(x) = \\frac{1}{n}\\#\\{t: r_t \\le x\\}$'),
     [T('no bins: compare $F_n$ with the model distribution function $F$', 'fără clase: comparăm $F_n$ cu funcția de repartiție a modelului $F$')]),
    T('Three EDF tests: Kolmogorov--Smirnov, Cramér--von Mises, Anderson--Darling \\refStephens', 'Trei teste EDF: Kolmogorov--Smirnov, Cramér--von Mises, Anderson--Darling \\refStephens')),
    pic('ch6_pearson.jpg', 'Karl Pearson (1857--1936)', 'Karl Pearson (1857--1936)',
        'https://commons.wikimedia.org/wiki/File:Karl_Pearson.jpg',
        'Photogravure (1910); public domain; Wikimedia Commons', 'Fotogravură (1910); domeniu public; Wikimedia Commons', h='0.44\\textheight'),
    wl='0.64', wr='0.32'))

D.frame(T('The Kolmogorov--Smirnov Test', 'Testul Kolmogorov--Smirnov'), cols(items(
    (T('$D_n = \\sup_x |F_n(x) - F(x)|$, the largest vertical gap between the EDF and the model CDF ($\\sup$: the largest value over all $x$)', '$D_n = \\sup_x |F_n(x) - F(x)|$, cea mai mare distanță pe verticală dintre EDF și funcția de repartiție a modelului ($\\sup$: cea mai mare valoare pe toate $x$)'),
     [T('$r_{(i)}$: the $i$-th smallest return; $u_{(i)} = F(r_{(i)})$: its model probability', '$r_{(i)}$: al $i$-lea cel mai mic randament; $u_{(i)} = F(r_{(i)})$: probabilitatea lui în model'),
      T('computing formula: $D_n = \\max_i \\max\\{i/n - u_{(i)},\\ u_{(i)} - (i-1)/n\\}$', 'formula de calcul: $D_n = \\max_i \\max\\{i/n - u_{(i)},\\ u_{(i)} - (i-1)/n\\}$')]),
    (T('$F$ fully known: the law of $D_n$ does not depend on $F$ (Kolmogorov, 1933)', '$F$ complet cunoscută: distribuția lui $D_n$ nu depinde de $F$ (Kolmogorov, 1933)'),
     [T('large $n$: reject at 5\\% if $D_n > 1.358/\\sqrt{n}$; tables for small $n$: \\refSmirnov', '$n$ mare: respingem la 5\\% dacă $D_n > 1{,}358/\\sqrt{n}$; tabele pentru $n$ mic: \\refSmirnov'),
      T('$n = @{n.sp500}$: critical value $@{ks.asy.sp}$', '$n = @{n.sp500}$: valoarea critică $@{ks.asy.sp}$')]),
    T('Andrey Kolmogorov also gave probability its axioms (Chapter 4)', 'Andrei Kolmogorov este și autorul axiomaticii probabilităților (Capitolul 4)')),
    pic('ch4_kolmogorov_1963.jpg', 'Andrey Kolmogorov (1903--1987) lecturing, 1963', 'Andrei Kolmogorov (1903--1987) la curs, 1963',
        'https://commons.wikimedia.org/wiki/File:Математик_Андрей_Колмогоров_в_аудитории.jpg',
        'Photo: Vsevolod Tarasevich (1963--1964); CC BY 4.0; Wikimedia Commons', 'Foto: Vsevolod Tarasevich (1963--1964); CC BY 4.0; Wikimedia Commons',
        h='0.38\\textheight'), wl='0.6', wr='0.36'))

D.frame(T('Worked Example: KS on Five Returns', 'Exemplu lucrat: KS pe cinci randamente'), items(
    (T('Returns (\\%): $-2.1, -0.4, 0.3, 0.9, 1.6$; $H_0$: $N(0, 1)$, fully specified', 'Randamente (\\%): $-2{,}1;\\ -0{,}4;\\ 0{,}3;\\ 0{,}9;\\ 1{,}6$; $H_0$: $N(0, 1)$, complet specificată'),
     [T('$u_{(i)} = \\Phi(r_{(i)})$: $@{a5.u0}$, $@{a5.u1}$, $@{a5.u2}$, $@{a5.u3}$, $@{a5.u4}$ ($\\Phi$: the standard Normal distribution function)',
        '$u_{(i)} = \\Phi(r_{(i)})$: $@{a5.u0}$; $@{a5.u1}$; $@{a5.u2}$; $@{a5.u3}$; $@{a5.u4}$ ($\\Phi$: funcția de repartiție Normală standard)'),
      T('$i/n$: $0.2, 0.4, 0.6, 0.8, 1.0$; the largest gap is $u_{(3)} - 2/5 = @{a5.d}$', '$i/n$: $0{,}2;\\ 0{,}4;\\ 0{,}6;\\ 0{,}8;\\ 1{,}0$; cea mai mare distanță este $u_{(3)} - 2/5 = @{a5.d}$')]),
    T('$D_5 = @{a5.d} < @{a5.crit}$, the exact 5\\% critical value for $n = 5$: do not reject', '$D_5 = @{a5.d} < @{a5.crit}$, valoarea critică exactă la 5\\% pentru $n = 5$: nu respingem'),
    T('With five observations, almost any law passes: the test has little power in small samples', 'Cu cinci observații, aproape orice distribuție trece testul: în eșantioane mici, puterea testului este redusă')) + ql('SFM_ch6_gof_tests'))

D.frame(T('Estimated Parameters: the Lilliefors Problem', 'Parametri estimați: problema Lilliefors'), items(
    (T('In practice $F$ has estimated parameters: $F(x; \\hat\\theta)$ is fitted to the same data', 'În practică, $F$ are parametri estimați: $F(x; \\hat\\theta)$ este estimată pe aceleași date'),
     [T('the fit pulls $F$ towards $F_n$: $D_n$ is smaller than under a known $F$', 'estimarea apropie $F$ de $F_n$: $D_n$ este mai mic decît pentru o $F$ cunoscută'),
      T('the usual critical values are too large: the test almost never rejects', 'valorile critice obișnuite sînt prea mari: testul aproape că nu mai respinge')]),
    (T('\\refLilliefors: critical values by simulation for the Normal law with $\\hat\\mu$, $\\hat\\sigma$ (5\\% level)', '\\refLilliefors: valori critice obținute prin simulare pentru distribuția Normală cu $\\hat\\mu$, $\\hat\\sigma$ (nivelul 5\\%)'),
     [T('$n = 5$: $@{a5.lil}$ instead of $@{a5.crit}$; $n = 1000$: $@{li.1000}$ instead of $@{ks.asy.1000}$', '$n = 5$: $@{a5.lil}$ în loc de $@{a5.crit}$; $n = 1000$: $@{li.1000}$ în loc de $@{ks.asy.1000}$')]),
    (T('Any model: \\textbf{parametric bootstrap} \\refSGQ', 'Orice model: \\textbf{bootstrap parametric} \\refSGQ'),
     [T('simulate $n$ draws from $F(\\cdot; \\hat\\theta)$, refit, recompute the statistic; repeat $B$ times', 'simulăm $n$ extrageri din $F(\\cdot; \\hat\\theta)$, reestimăm parametrii, recalculăm statistica; repetăm de $B$ ori'),
      T('$p$-value $= (1 + \\#\\{\\text{bootstrap statistics} \\ge \\text{observed}\\})/(B + 1)$: the share of simulated statistics at least as large as the observed one', 'p-value $= (1 + \\#\\{\\text{statistici bootstrap} \\ge \\text{cea observată}\\})/(B + 1)$: proporția statisticilor simulate cel puțin egale cu cea observată')])))

D.frame(T('Cramér--von Mises and Anderson--Darling', 'Cramér--von Mises și Anderson--Darling'), cols(items(
    (T('Both integrate the squared gap $[F_n(x) - F(x)]^2$ over all $x$ instead of taking its maximum', 'Ambele integrează pătratul distanței $[F_n(x) - F(x)]^2$ pe toate valorile $x$ în loc să ia maximul'),
     [T('\\textbf{Cramér--von Mises}: $W^2 = n\\int [F_n - F]^2\\, dF = \\frac{1}{12n} + \\sum_i \\big(u_{(i)} - \\frac{2i - 1}{2n}\\big)^2$ \\refCramer',
        '\\textbf{Cramér--von Mises}: $W^2 = n\\int [F_n - F]^2\\, dF = \\frac{1}{12n} + \\sum_i \\big(u_{(i)} - \\frac{2i - 1}{2n}\\big)^2$ \\refCramer'),
      T('\\textbf{Anderson--Darling}: $A^2 = n\\int \\dfrac{[F_n - F]^2}{F(1 - F)}\\, dF$ \\refAD', '\\textbf{Anderson--Darling}: $A^2 = n\\int \\dfrac{[F_n - F]^2}{F(1 - F)}\\, dF$ \\refAD')]),
    (T('Computing formula \\refADb: $A^2 = -n - \\frac{1}{n}\\sum_i (2i - 1)\\big[\\ln u_{(i)} + \\ln(1 - u_{(n + 1 - i)})\\big]$',
       'Formula de calcul \\refADb: $A^2 = -n - \\frac{1}{n}\\sum_i (2i - 1)\\big[\\ln u_{(i)} + \\ln(1 - u_{(n + 1 - i)})\\big]$'),
     [T('$u_{(i)} = F(r_{(i)})$ as in KS; large $W^2$ or $A^2$: a poor fit', '$u_{(i)} = F(r_{(i)})$, ca la KS; valori mari ale lui $W^2$ sau $A^2$: o ajustare slabă'),
      T('the weight $1/[F(1 - F)]$ is huge where $F$ is near 0 or 1: the tails', 'ponderea $1/[F(1 - F)]$ este foarte mare acolo unde $F$ este aproape de 0 sau 1, adică în cozi')])),
    pic('ch6_cramer.jpg', 'Harald Cramér (1893--1985)', 'Harald Cramér (1893--1985)',
        'https://commons.wikimedia.org/wiki/File:Harald_Cramér.jpg',
        'Photo: unknown photographer (1951); public domain; Wikimedia Commons', 'Foto: fotograf necunoscut (1951); domeniu public; Wikimedia Commons',
        h='0.42\\textheight'), wl='0.66', wr='0.30'))

chart(T('Why KS Is Weak in the Tails', 'Slăbiciunea testului KS în cozi'), 'sfm_ch6_ks_tail', 'SFM_ch6_gof_tests', [
    T('S\\&P 500 against its Normal fit. Left: the KS gap peaks at $x = @{kt.x}\\%$, near the centre ($F = @{kt.F}\\%$), with $D_n = @{kt.d}$',
      'S\\&P 500 comparat cu distribuția Normală estimată. Stînga: distanța KS are maximul la $x = @{kt.x}\\%$, aproape de centru ($F = @{kt.F}\\%$), cu $D_n = @{kt.d}$'),
    (T('In the tails $F_n$ and $F$ are both close to 0 or 1, so the raw gap is small even when the model is badly wrong', 'În cozi, $F_n$ și $F$ sînt amîndouă aproape de 0 sau 1, deci distanța brută este mică chiar dacă modelul greșește mult'),
     [T('the Anderson--Darling weight (right) puts the largest gap at $x = @{kt.wx}\\%$', 'cu ponderea Anderson--Darling (dreapta), distanța maximă apare la $x = @{kt.wx}\\%$')])],
    h='0.50\\textheight')

chart(T('Power of the Three Tests', 'Puterea celor trei teste'), 'sfm_ch6_gof_power', 'SFM_ch6_gof_tests', [
    T('Share of 1000 simulated samples in which Normality (estimated parameters) is rejected at 5\\%; power = probability of rejecting a false $H_0$',
      'Proporția din 1000 de eșantioane simulate în care normalitatea (parametri estimați) este respinsă la 5\\%; puterea = probabilitatea de a respinge un $H_0$ fals'),
    (T('Student-t data, $n = 250$: KS $@{pw.t5.250.ks}\\%$, Cramér--von Mises $@{pw.t5.250.cvm}\\%$, Anderson--Darling $@{pw.t5.250.ad}\\%$', 'Date Student-t, $n = 250$: KS $@{pw.t5.250.ks}\\%$, Cramér--von Mises $@{pw.t5.250.cvm}\\%$, Anderson--Darling $@{pw.t5.250.ad}\\%$'),
     [T('tail-only departures ($n = 250$): $@{pw.contam.250.ks}\\%$, $@{pw.contam.250.cvm}\\%$, $@{pw.contam.250.ad}\\%$', 'abateri doar în cozi ($n = 250$): $@{pw.contam.250.ks}\\%$, $@{pw.contam.250.cvm}\\%$, $@{pw.contam.250.ad}\\%$')])],
    h='0.50\\textheight')


def gofrow(m):
    lab = {'Mixture': T('Normal mixture', 'amestec Normal'), 'Normal': T('Normal', 'Normală'), 'Stable': T('stable', 'stabilă'), 'Skewed-t': 'skewed-t'}.get(m, m)
    return f'{lab} & $@{{g.{m}.ks}}$ & $@{{g.{m}.cvm}}$ & $@{{g.{m}.ad}}$'


D.frame(T('EDF Tests on the S\\&P 500', 'Teste EDF pe S\\&P 500'), cols(table(
    'lrrr', T('& KS & CvM & AD', '& KS & CvM & AD'), [gofrow(m) for m in MODELS], size='scriptsize'), items(
    (T('Parametric bootstrap, $B = @{gb.B}$, 95\\% critical values', 'Bootstrap parametric, $B = @{gb.B}$, valori critice (cuantila de 95\\%)'),
     [T('Student-t: KS $@{gb.Student-t.ks.crit}$, AD $@{gb.Student-t.ad.crit}$', 'Student-t: KS $@{gb.Student-t.ks.crit}$, AD $@{gb.Student-t.ad.crit}$'),
      T('skewed-t: KS $@{gb.Skewed-t.ks.crit}$, AD $@{gb.Skewed-t.ad.crit}$', 'skewed-t: KS $@{gb.Skewed-t.ks.crit}$, AD $@{gb.Skewed-t.ad.crit}$')]),
    T('Normal, Student-t, skewed-t: bootstrap $p = @{gb.pmin}$, the smallest possible: all rejected', 'Normală, Student-t, skewed-t: $p$ bootstrap $= @{gb.pmin}$, cel mai mic posibil: toate respinse'),
    T('Student-t, naive KS $p$-value (as if $\\theta$ were known): $@{gb.Student-t.naive}$', 'Student-t, p-value-ul KS naiv (ca și cum $\\theta$ ar fi cunoscut): $@{gb.Student-t.naive}$')),
    wl='0.44', wr='0.52') + items(
    T('Smaller statistics mean a better fit', 'Valorile mai mici ale statisticilor indică o ajustare mai bună')) + ql('SFM_ch6_gof_tests'), size='footnotesize')

D.frame(T('Statistical vs Practical Significance', 'Semnificație statistică și semnificație practică'), items(
    (T('With $n = @{n.sp500}$ every model is rejected: tests detect tiny departures', 'Cu $n = @{n.sp500}$, toate modelele sînt respinse: testele detectează abateri minuscule'),
     [T('and real returns are not i.i.d.\\ (volatility clustering), so no i.i.d.\\ law can pass', 'în plus, randamentele reale nu sînt i.i.d.\\ (volatility clustering), deci nicio distribuție i.i.d.\\ nu poate trece testul')]),
    (T('Use the statistics to \\textbf{rank} the models', 'Folosiți statisticile pentru a \\textbf{ordona} modelele'),
     [T('the AD column of the previous table gives the AIC order: NIG first, the Normal model far behind', 'coloana AD din tabelul anterior dă ordinea AIC: NIG pe primul loc, modelul Normal mult în urmă')]),
    T('\\textbf{Question for the room}: a KS test on 50 daily returns does not reject the Normal law; is the Normal law then fine for VaR 1\\%?',
      '\\textbf{Întrebare pentru sală}: un test KS pe 50 de randamente zilnice nu respinge distribuția Normală; este atunci distribuția Normală potrivită pentru VaR 1\\%?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('no: with $n = 50$ the test rarely rejects even Student-t data ($@{pw.t5.100.ks}\\%$ of the time at $n = 100$), and KS barely looks at the tail',
        'nu: cu $n = 50$, testul respinge rar chiar și date Student-t (în $@{pw.t5.100.ks}\\%$ din cazuri la $n = 100$), iar KS aproape ignoră coada')])))

# =============================================================================
# 6. GRAFICE PP ȘI QQ
# =============================================================================
D.section('PP and QQ Plots', 'Grafice PP și QQ')

D.frame(T('Two Ways to Look at a Fit', 'Două moduri de a evalua vizual ajustarea'), items(
    (T('\\textbf{PP plot} (probability--probability): points $\\big(p_i, F(r_{(i)})\\big)$ with $p_i = (i - 0.5)/n$', '\\textbf{Grafic PP} (probabilitate--probabilitate): punctele $\\big(p_i, F(r_{(i)})\\big)$ cu $p_i = (i - 0{,}5)/n$'),
     [T('both axes in $[0, 1]$: differences in the centre show; the tails are squeezed into the corners', 'ambele axe în $[0, 1]$: se văd diferențele din centru; cozile sînt înghesuite în colțuri'),
      T('the visual twin of KS: the largest vertical distance to the diagonal is close to $D_n$', 'echivalentul grafic al testului KS: cea mai mare distanță pe verticală față de diagonală este apropiată de $D_n$')]),
    (T('\\textbf{QQ plot} (quantile--quantile): points $\\big(F^{-1}(p_i), r_{(i)}\\big)$', '\\textbf{Grafic QQ} (cuantilă--cuantilă): punctele $\\big(F^{-1}(p_i), r_{(i)}\\big)$'),
     [T('in units of returns: the tails are stretched out, each extreme day is visible', 'în unități de randament: cozile sînt dilatate, iar fiecare zi extremă este vizibilă'),
      T('ends above/below the line: the model tails are too short; ends flattened: the model tails are too long', 'capetele se îndepărtează de dreaptă: cozile modelului sînt prea scurte; capetele aplatizate: cozile modelului sînt prea lungi')]),
    T('Risk management reads QQ plots; PP plots and KS share the same blind spot', 'În managementul riscului se folosesc graficele QQ; graficele PP și testul KS au același punct orb')))

chart(T('PP Plots: S\\&P 500', 'Grafice PP: S\\&P 500'), 'sfm_ch6_pp', 'SFM_ch6_pp_qq', [
    T('Normal: an S-shape in the centre (too few small returns, too many medium ones), largest gap $@{pq.Normal.pp}$', 'Normală: o formă de S în centru (prea puține randamente mici, prea multe de mărime medie), cea mai mare distanță $@{pq.Normal.pp}$'),
    T('Student-t, skewed-t, stable: on the diagonal; in a PP plot they are indistinguishable', 'Student-t, skewed-t, stabilă: pe diagonală; într-un grafic PP nu se pot deosebi')], h='0.5\\textheight')

chart(T('QQ Plots: S\\&P 500', 'Grafice QQ: S\\&P 500'), 'sfm_ch6_qq', 'SFM_ch6_pp_qq', [
    (T('The worst day was $@{pq.xmin}\\%$; the model quantile at the same probability:', 'Cel mai mic randament zilnic a fost $@{pq.xmin}\\%$; cuantila modelului la aceeași probabilitate:'),
     [T('Normal $@{pq.Normal.q}\\%$, Student-t $@{pq.Student-t.q}\\%$, skewed-t $@{pq.Skewed-t.q}\\%$, stable $@{pq.Stable.q}\\%$', 'Normală $@{pq.Normal.q}\\%$, Student-t $@{pq.Student-t.q}\\%$, skewed-t $@{pq.Skewed-t.q}\\%$, stabilă $@{pq.Stable.q}\\%$')]),
    T('The models that looked identical in the PP plot are far apart in the far tail', 'Modelele care păreau identice în graficul PP diferă puternic în extremitatea cozii')], h='0.5\\textheight')

chart(T('QQ Plots for BET, Bitcoin and Banca Transilvania', 'Grafice QQ pentru BET, Bitcoin și Banca Transilvania'), 'sfm_ch6_qq_assets', 'SFM_ch6_pp_qq', [
    T('Normal (red): the steep ends of all three series; the Student-t and skewed-t follow the data far into the tails',
      'Normală (roșu): capete abrupte pentru toate cele trei serii; Student-t și skewed-t urmăresc datele pînă departe în cozi'),
    T('The last few points are single days (the largest crashes): there the fitted tails disagree most and the uncertainty is widest',
      'Ultimele cîteva puncte sînt zile izolate (cele mai mari crahuri): acolo cozile estimate diferă cel mai mult, iar incertitudinea este cea mai mare')], h='0.54\\textheight')

# =============================================================================
# 7. COZILE: VaR 1%
# =============================================================================
D.section('The Purpose Decides: Which Model Gets VaR 1\\% Right?', 'Scopul decide: ce model estimează corect VaR 1\\%?')

D.frame(T('VaR and ES', 'VaR și ES'), items(
    (T('Return $X$, level $\\alpha$ (here $\\alpha = 1\\%$ or $2.5\\%$, the probability of the tail)', 'Randamentul $X$, nivelul $\\alpha$ (aici $\\alpha = 1\\%$ sau $2{,}5\\%$, probabilitatea cozii)'),
     [T('\\textbf{VaR} (Value at Risk): $\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$, the loss exceeded with probability $\\alpha$', '\\textbf{VaR} (Value at Risk, valoarea expusă la risc): $\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$, pierderea depășită cu probabilitatea $\\alpha$'),
      T('\\textbf{ES} (expected shortfall): $\\mathrm{ES}_\\alpha = -\\frac{1}{\\alpha}\\int_0^\\alpha q_u\\, du = -E[X \\mid X \\le q_\\alpha]$, the average loss in the tail',
        '\\textbf{ES} (expected shortfall, pierderea așteptată în coadă): $\\mathrm{ES}_\\alpha = -\\frac{1}{\\alpha}\\int_0^\\alpha q_u\\, du = -E[X \\mid X \\le q_\\alpha]$, pierderea medie din coadă')]),
    (T('Normal model: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, $z_{0.01} = -@{z01}$', 'Modelul Normal: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, $z_{0{,}01} = -@{z01}$'),
     [T('S\\&P 500: $-(@{sp.mu} - @{z01} \\times @{sp.sig}) = @{sp.varN}\\%$, against the empirical VaR 1\\% of $@{v.sp500.emp}\\%$',
        'S\\&P 500: $-(@{sp.mu} - @{z01} \\times @{sp.sig}) = @{sp.varN}\\%$, față de VaR 1\\% empiric de $@{v.sp500.emp}\\%$')]),
    T('Basel rules: ES 2.5\\% for market risk, VaR 1\\% for backtesting (Chapter 10)', 'Regulile Basel: ES 2,5\\% pentru riscul de piață, VaR 1\\% pentru backtesting (Capitolul 10)')))


def varrow(k):
    return (f'{NAMES[k]} & $@{{v.{k}.emp}}$ & $@{{v.{k}.Normal}}$ & $@{{v.{k}.Student-t}}$ & $@{{v.{k}.Skewed-t}}$ & $@{{v.{k}.GED}}$ & '
            f'$@{{v.{k}.NIG}}$ & $@{{v.{k}.Stable}}$')


def esrow(k):
    return (f'{NAMES[k]} & $@{{e.{k}.emp}}$ & $@{{e.{k}.Normal}}$ & $@{{e.{k}.Student-t}}$ & $@{{e.{k}.Skewed-t}}$ & $@{{e.{k}.GED}}$ & '
            f'$@{{e.{k}.NIG}}$ & $@{{e.{k}.Stable}}$')


HDR = T('& data & Normal & Student-t & skewed-t & GED & NIG & stable', '& date & Normală & Student-t & skewed-t & GED & NIG & stabilă')
D.frame(T('VaR 1\\% and ES 2.5\\% under Each Model (In Sample)', 'VaR 1\\% și ES 2,5\\% pentru fiecare model (în eșantion)'),
        T('\\textbf{VaR 1\\%} (\\%)', '\\textbf{VaR 1\\%} (\\%)') + table('lrrrrrrr', HDR, [varrow(k) for k in ALL], size='scriptsize') +
        T('\\textbf{ES 2.5\\%} (\\%)', '\\textbf{ES 2,5\\%} (\\%)') + table('lrrrrrrr', HDR, [esrow(k) for k in ['bet', 'sp500', 'btc']], size='scriptsize')
        + ql('SFM_ch6_tail_var'), size='footnotesize')

chart(T('Model VaR 1\\% Relative to the Data', 'VaR 1\\% al modelelor raportat la date'), 'sfm_ch6_var_models', 'SFM_ch6_tail_var', [
    T('Normal: $@{rat.n.lo}$--$@{rat.n.hi}\\%$ too low everywhere; stable: up to $@{rat.s.hi}\\%$ too high', 'Normală: prea mic cu $@{rat.n.lo}$--$@{rat.n.hi}\\%$ pe toate seriile; stabilă: prea mare cu pînă la $@{rat.s.hi}\\%$'),
    T('Student-t, skewed-t and NIG stay within $@{rat.good}\\%$ of the empirical VaR 1\\%; GED is a little low', 'Student-t, skewed-t și NIG se abat cu cel mult $@{rat.good}\\%$ de la VaR 1\\% empiric; GED este puțin prea mic')], h='0.55\\textheight')

chart(T('The Left Tail on Log-Log Axes (S\\&P 500)', 'Coada stîngă pe axe log-log (S\\&P 500)'), 'sfm_ch6_left_tail', 'SFM_ch6_tail_var', [
    T('Points: share of days with a loss above $x$; lines: the same probability under each fitted model', 'Punctele: proporția zilelor cu o pierdere peste $x$; liniile: aceeași probabilitate pentru fiecare model estimat'),
    T('At 1\\% all heavy-tailed models are close; beyond $5\\%$ losses the stable line is far too high and the Normal one collapses',
      'La 1\\%, toate modelele cu cozi groase sînt apropiate; pentru pierderi de peste $5\\%$, curba distribuției stabile este mult prea sus, iar cea Normală scade abrupt')], h='0.58\\textheight')


def xrow(k):
    return (f'{NAMES[k]} & @{{x.{k}.exp}} & @{{x.{k}.Normal}} & @{{x.{k}.Student-t}} & @{{x.{k}.Skewed-t}} & @{{x.{k}.GED}} & '
            f'@{{x.{k}.NIG}} & @{{x.{k}.Stable}}')


D.frame(T('Counting Exceedances: the Kupiec Test', 'Numărarea depășirilor: testul Kupiec'), table(
    'lrrrrrrr', T('& expected & Normal & Student-t & skewed-t & GED & NIG & stable', '& așteptat & Normală & Student-t & skewed-t & GED & NIG & stabilă'),
    [xrow(k) for k in ['bet', 'sp500', 'dax', 'btc']], size='scriptsize') + items(
    T('Exceedance: a day with $r_t < -\\mathrm{VaR}_{1\\%}$; under a correct model their number is $\\mathrm{Binomial}(n, 0.01)$',
      'Depășire: o zi cu $r_t < -\\mathrm{VaR}_{1\\%}$; pentru un model corect, numărul lor este $\\mathrm{Binomial}(n; 0{,}01)$'),
    (T('\\refKupiec: $LR_{uc} = -2\\ln\\dfrac{(1-\\alpha)^{n - x}\\alpha^x}{(1 - x/n)^{n - x}(x/n)^x} \\approx \\chi^2(1)$, $x$ = number of exceedances',
      '\\refKupiec: $LR_{uc} = -2\\ln\\dfrac{(1-\\alpha)^{n - x}\\alpha^x}{(1 - x/n)^{n - x}(x/n)^x} \\approx \\chi^2(1)$, $x$ = numărul de depășiri'),
     [T('an LR test comparing the target rate $\\alpha$ with the observed rate $x/n$; large $LR_{uc}$: wrong number of exceedances', 'un test LR care compară rata țintă $\\alpha$ cu rata observată $x/n$; $LR_{uc}$ mare: număr greșit de depășiri')]),
    (T('S\\&P 500: Student-t @{x.sp500.Student-t} exceedances for @{x.sp500.exp} expected ($p = @{xp.sp500.Student-t}$)', 'S\\&P 500: Student-t, @{x.sp500.Student-t} de depășiri față de @{x.sp500.exp} așteptate ($p = @{xp.sp500.Student-t}$)'),
     [T('NIG, the AIC winner, @{x.sp500.NIG} ($p = @{xp.sp500.NIG}$)', 'NIG, cîștigătorul AIC: @{x.sp500.NIG} ($p = @{xp.sp500.NIG}$)')]),
    T('The best model for the whole density is not always the best for one quantile; tail-focused scores exist \\refDPD',
      'Cel mai bun model pentru întreaga densitate nu este întotdeauna cel mai bun pentru o cuantilă; există scoruri axate pe coadă \\refDPD')) + ql('SFM_ch6_tail_var'), size='footnotesize')

# =============================================================================
# 8. ÎN AFARA EȘANTIONULUI ȘI SUPRAAJUSTARE
# =============================================================================
D.section('Out of Sample: Validation and Overfitting', 'În afara eșantionului: validare și supraajustare')

D.frame(T('Why Out of Sample?', 'Rolul validării în afara eșantionului'), items(
    (T('In sample, a larger model always fits at least as well: $\\ell$ never decreases when parameters are added', 'În eșantion, un model mai general se ajustează cel puțin la fel de bine: $\\ell$ nu scade niciodată cînd adăugăm parametri'),
     [T('\\textbf{overfitting}: the extra parameters fit noise of the estimation sample, not features of future returns',
        '\\textbf{supraajustare} (overfitting): parametrii suplimentari modelează zgomotul eșantionului de estimare, nu trăsături ale randamentelor viitoare')]),
    (T('The honest check: estimate until 31 December 2019, evaluate on 2020--@{y1} (COVID-19 crash, 2022 inflation shock, 2025 tariffs)',
       'Verificarea onestă: estimăm pînă la 31 decembrie 2019 și evaluăm pe 2020--@{y1} (crahul COVID-19, șocul inflației din 2022, taxele vamale din 2025)'),
     [T('\\textbf{log score}: the mean of $\\ln f(r_t; \\hat\\theta)$ over the test days; higher is better, a proper scoring rule \\refGR',
        '\\textbf{scorul logaritmic}: media lui $\\ln f(r_t; \\hat\\theta)$ pe zilele de test; valoarea mai mare este preferată; este o regulă de scor proprie (proper scoring rule) \\refGR'),
      T('\\textbf{exceedances} of VaR 1\\% (Kupiec) and the \\textbf{pinball loss} $\\frac{1}{n}\\sum_t (\\alpha - \\mathbf{1}\\{r_t < q\\})(r_t - q)$ of the 1\\% quantile $q$',
        '\\textbf{depășirile} VaR 1\\% (Kupiec) și \\textbf{pierderea pinball} $\\frac{1}{n}\\sum_t (\\alpha - \\mathbf{1}\\{r_t < q\\})(r_t - q)$ a cuantilei de 1\\% $q$'),
      T('pinball: $\\mathbf{1}\\{r_t < q\\} = 1$ on a day below $q$, 0 otherwise; it penalises misses asymmetrically, lower is better', 'pinball: $\\mathbf{1}\\{r_t < q\\} = 1$ într-o zi sub $q$, 0 altfel; penalizează asimetric abaterile, valoarea mai mică este preferată')]),
    T('A score is \\textbf{proper} if the true distribution gets the best expected score: forecasters cannot gain by distorting',
      'Un scor este \\textbf{propriu} dacă distribuția adevărată obține cel mai bun scor așteptat: prognozistul nu are nimic de cîștigat dacă își distorsionează prognoza')))

chart(T('Out-of-Sample Results, 2020--@{y1}', 'Rezultate în afara eșantionului, 2020--@{y1}'), 'sfm_ch6_oos', 'SFM_ch6_out_of_sample', [
    T('Left: every heavy-tailed model beats the Normal log score by $@{ls.lo}$ to $@{ls.hi}$ per day; the differences among them are small',
      'Stînga: fiecare model cu cozi groase depășește scorul logaritmic al modelului Normal cu $@{ls.lo}$ pînă la $@{ls.hi}$ pe zi; diferențele dintre ele sînt mici'),
    T('Right: VaR 1\\% exceedance rates; the target is the dashed line at 1\\%', 'Dreapta: ratele de depășire ale VaR 1\\%; ținta este linia punctată de la 1\\%')], h='0.55\\textheight')


def oosrow(k):
    return (f'{NAMES[k]} & @{{o.{k}.exp}} & @{{o.{k}.Normal.x}} & @{{o.{k}.Student-t.x}} & @{{o.{k}.Skewed-t.x}} & @{{o.{k}.NIG.x}} & '
            f'@{{o.{k}.Stable.x}} & @{{o.{k}.Historical.x}} & @{{o.{k}.baic}} & @{{o.{k}.bls}}')


D.frame(T('Reading the Out-of-Sample Table', 'Interpretarea tabelului din afara eșantionului'), table(
    'lrrrrrrrll', T('& exp. & Normal & t & skew-t & NIG & stable & hist. & best AIC & best log score',
                    '& aștept. & Normală & t & skew-t & NIG & stabilă & ist. & cel mai bun AIC & cel mai bun scor'),
    [oosrow(k) for k in ASSETS], size='scriptsize') + items(
    T('VaR 1\\% exceedances on the test days; exp.: expected number; hist.: historical simulation (the 1\\% quantile of the estimation sample)',
      'Depășiri ale VaR 1\\% pe zilele de test; aștept.: numărul așteptat; ist.: simularea istorică (cuantila de 1\\% a eșantionului de estimare)'),
    T('The AIC winner of 2000--2019 has the best log score only for the DAX; elsewhere the best log score belongs to a close neighbour',
      'Cîștigătorul AIC din 2000--2019 are cel mai bun scor logaritmic doar pentru DAX; în rest, cel mai bun scor aparține unui model apropiat în clasament'),
    (T('BET: the Normal model has @{o.bet.Normal.x} exceedances for @{o.bet.exp} expected, the heavy-tailed models only @{o.bet.Student-t.x}', 'BET: modelul Normal are @{o.bet.Normal.x} depășiri față de @{o.bet.exp} așteptate, modelele cu cozi groase doar @{o.bet.Student-t.x}'),
     [T('2020--@{y1} was calmer for the BET than 2000--2019', 'perioada 2020--@{y1} a fost mai calmă pentru BET decît 2000--2019')]),
    T('The right count for the wrong reason: a static model cannot follow changes in volatility', 'Numărul potrivit, dar din motive greșite: un model static nu poate urmări schimbările de volatilitate')) + ql('SFM_ch6_out_of_sample'), size='footnotesize')

chart(T('Overfitting: Normal Mixtures with 1 to 6 Components', 'Supraajustare: amestecuri Normale cu 1 pînă la 6 componente'), 'sfm_ch6_overfit', 'SFM_ch6_out_of_sample', [
    T('S\\&P 500, estimation 2017--2018 ($n = @{of.nin}$), test 2019--@{y1} ($n = @{of.nout}$); $c$ components have $3c - 1$ parameters',
      'S\\&P 500, estimare 2017--2018 ($n = @{of.nin}$), test 2019--@{y1} ($n = @{of.nout}$); $c$ componente au $3c - 1$ parametri'),
    T('In sample (blue) the fit improves with every component; out of sample (red) it peaks at $c = @{of.best_out}$ and then declines',
      'În eșantion (albastru), calitatea ajustării crește cu fiecare componentă; în afara eșantionului (roșu) atinge maximul la $c = @{of.best_out}$ și apoi scade')], h='0.52\\textheight')

D.frame(T('Which Criterion Warned Us?', 'Criteriul care a semnalat supraajustarea'), items(
    (T('Mean log-likelihood per day, in sample $\\to$ out of sample', 'Log-verosimilitatea medie pe zi, în eșantion $\\to$ în afara eșantionului'),
     [T('$c = 1$ (Normal): $@{of.1.in} \\to @{of.1.out}$; $c = 3$: $@{of.3.in} \\to @{of.3.out}$; $c = 6$ (@{of.6.k} parameters): $@{of.6.in} \\to @{of.6.out}$',
        '$c = 1$ (Normală): $@{of.1.in} \\to @{of.1.out}$; $c = 3$: $@{of.3.in} \\to @{of.3.out}$; $c = 6$ (@{of.6.k} parametri): $@{of.6.in} \\to @{of.6.out}$')]),
    (T('BIC chose $c = @{of.best_bic}$, the best model out of sample; AIC chose $c = @{of.best_aic}$', 'BIC a ales $c = @{of.best_bic}$, cel mai bun model în afara eșantionului; AIC a ales $c = @{of.best_aic}$'),
     [T('with $n = @{of.nin}$, AIC\'s penalty of 2 per parameter is too weak to stop extra components', 'cu $n = @{of.nin}$, penalizarea AIC de 2 pe parametru este prea slabă pentru a opri componentele în plus')]),
    T('The gap between in-sample and out-of-sample fit grows with the number of parameters: that gap is the overfitting',
      'Diferența dintre ajustarea în eșantion și cea din afara lui crește cu numărul de parametri: această diferență este supraajustarea'),
    T('Rule: short samples, many parameters $\\Rightarrow$ prefer BIC or AICc, and always keep a test period', 'Regula: eșantioane scurte, mulți parametri $\\Rightarrow$ preferați BIC sau AICc și păstrați întotdeauna o perioadă de test')) + ql('SFM_ch6_out_of_sample'))

chart(T('Rolling VaR 1\\%: Refit Every Month', 'VaR 1\\% pe fereastră mobilă: reestimare lunară'), 'sfm_ch6_rolling_var', 'SFM_ch6_out_of_sample', [
    T('S\\&P 500: each model refitted every 20 trading days on the previous 1000 days; the VaR 1\\% line is drawn as a loss threshold $-\\mathrm{VaR}$',
      'S\\&P 500: fiecare model este reestimat la fiecare 20 de zile de tranzacționare pe cele 1000 de zile anterioare; linia VaR 1\\% este desenată ca prag de pierdere $-\\mathrm{VaR}$'),
    T('Black triangles: exceedances of the Student-t VaR; they come in clusters (March 2020)', 'Triunghiurile negre: depășiri ale VaR Student-t; apar grupat (martie 2020)')], h='0.55\\textheight')

D.frame(T('Rolling Backtest: the Results', 'Backtesting pe fereastră mobilă: rezultatele'), table(
    'lrrrr', T('& Normal & Student-t & skewed-t & historical', '& Normală & Student-t & skewed-t & istorică'),
    [T('S\\&P 500, rate (\\%)', 'S\\&P 500, rata (\\%)') + ' & $@{rv.sp.Normal.rate}$ & $@{rv.sp.Student-t.rate}$ & $@{rv.sp.Skewed-t.rate}$ & $@{rv.sp.Historical.rate}$',
     T('S\\&P 500, Kupiec $p$', 'S\\&P 500, Kupiec $p$') + ' & $@{rv.sp.Normal.p}$ & $@{rv.sp.Student-t.p}$ & $@{rv.sp.Skewed-t.p}$ & $@{rv.sp.Historical.p}$',
     T('S\\&P 500, most in one year', 'S\\&P 500, maximul anual') + ' & @{rv.sp.Normal.my} & @{rv.sp.Student-t.my} & @{rv.sp.Skewed-t.my} & @{rv.sp.Historical.my}',
     T('BET, rate (\\%)', 'BET, rata (\\%)') + ' & $@{rv.bet.Normal.rate}$ & $@{rv.bet.Student-t.rate}$ & $@{rv.bet.Skewed-t.rate}$ & $@{rv.bet.Historical.rate}$',
     T('BET, Kupiec $p$', 'BET, Kupiec $p$') + ' & $@{rv.bet.Normal.p}$ & $@{rv.bet.Student-t.p}$ & $@{rv.bet.Skewed-t.p}$ & $@{rv.bet.Historical.p}$'],
    size='footnotesize') + items(
    T('Backtest @{rv.sp.y0}--@{y1}, $n = @{rv.sp.n}$ days for the S\\&P 500; target rate 1\\%; the worst year is 2008 for every model and both series',
      'Backtesting @{rv.sp.y0}--@{y1}, $n = @{rv.sp.n}$ zile pentru S\\&P 500; rata țintă 1\\%; cel mai slab an este 2008, pentru toate modelele și pentru ambele serii'),
    (T('Heavy tails halve the excess of exceedances, but no unconditional model reaches 1\\%', 'Cozile groase înjumătățesc excesul de depășiri, dar niciun model necondiționat nu ajunge la 1\\%'),
     [T('the volatility changes faster than a 1000-day window can adapt', 'volatilitatea se schimbă mai repede decît se poate adapta o fereastră de 1000 de zile')]),
    T('The remedy is a model for the volatility itself: GARCH, Chapters 8 and 9', 'Remediul este un model pentru volatilitatea însăși: GARCH, Capitolele 8 și 9')) + ql('SFM_ch6_out_of_sample'), size='footnotesize')

chart(T('The VaR Reliability Plot (SFEVaRqqplot)', 'Graficul de fiabilitate VaR (SFEVaRqqplot)'), 'sfm_ch6_var_qqplot', 'SFM_ch6_var_qqplot', [
    (T('DAX since @{vq.y0}: Normal VaR 1\\% $= -z_{0.01}\\hat\\sigma_t = 2.326\\,\\hat\\sigma_t$, with $\\hat\\sigma_t$ from the previous 250 days', 'DAX din @{vq.y0}: VaR 1\\% Normal $= -z_{0.01}\\hat\\sigma_t = 2.326\\,\\hat\\sigma_t$, cu $\\hat\\sigma_t$ din ultimele 250 de zile'),
     [T('RMA (rectangular moving average): $\\hat\\sigma_t^2 = \\frac{1}{250}\\sum_{j=1}^{250} r_{t-j}^2$, equal weights', 'RMA (rectangular moving average, medie mobilă simplă): $\\hat\\sigma_t^2 = \\frac{1}{250}\\sum_{j=1}^{250} r_{t-j}^2$, ponderi egale'),
      T('EMA (exponential moving average): $\\hat\\sigma_t^2 = (1 - \\lambda)\\sum_{j \\ge 0} \\lambda^j r_{t-1-j}^2$, $\\lambda = 0.96$: each older day weighs 4\\% less', 'EMA (exponential moving average, medie mobilă exponențială): $\\hat\\sigma_t^2 = (1 - \\lambda)\\sum_{j \\ge 0} \\lambda^j r_{t-1-j}^2$, $\\lambda = 0{,}96$: fiecare zi mai veche are o pondere cu 4\\% mai mică'),
      T('QQ plot of $L/\\mathrm{VaR}$ (loss divided by VaR) against Normal quantiles', 'graficul QQ al lui $L/\\mathrm{VaR}$ (pierderea împărțită la VaR) față de cuantilele Normale')]),
    (T('A straight line would mean a reliable VaR; both curve upwards at the right end', 'O dreaptă ar însemna un VaR fiabil; ambele se curbează în sus la capătul drept'),
     [T('the losses beyond VaR are too large and too frequent ($@{vq.rma.rate}\\%$ and $@{vq.ema.rate}\\%$ exceedances)', 'pierderile dincolo de VaR sînt prea mari și prea frecvente ($@{vq.rma.rate}\\%$ și $@{vq.ema.rate}\\%$ depășiri)')])], h='0.46\\textheight', size='scriptsize')

# =============================================================================
# 9. INCERTITUDINEA DE MODEL
# =============================================================================
D.section('Model Uncertainty', 'Incertitudinea de model')

chart(T('How Sure Are We of the Winner?', 'Stabilitatea cîștigătorului'), 'sfm_ch6_model_uncertainty', 'SFM_ch6_model_uncertainty', [
    T('S\\&P 500, @{u.B} moving-block bootstrap resamples (blocks of 20 days keep the volatility clusters); in each, six models refitted',
      'S\\&P 500, @{u.B} de reeșantionări bootstrap pe blocuri mobile (blocurile de 20 de zile păstrează volatility clustering); în fiecare reeșantionare, cele șase modele sînt reestimate'),
    T('Left: NIG has the lowest AIC in $@{u.w.NIG}\\%$ of the resamples; right: the bootstrap spread of VaR 1\\% under each model',
      'Stînga: NIG are cel mai mic AIC în $@{u.w.NIG}\\%$ din reeșantionări; dreapta: dispersia bootstrap a VaR 1\\% pentru fiecare model')], h='0.52\\textheight')

D.frame(T('Two Sources of Uncertainty', 'Două surse de incertitudine'), items(
    (T('\\textbf{Parameter uncertainty}: the same model, re-estimated on resampled data', '\\textbf{Incertitudinea parametrilor}: același model, reestimat pe date reeșantionate'),
     [T('Student-t VaR 1\\%: 90\\% bootstrap band $[@{u.v.Student-t.lo}, @{u.v.Student-t.hi}]\\%$; NIG: $[@{u.v.NIG.lo}, @{u.v.NIG.hi}]\\%$',
        'VaR 1\\% Student-t: intervalul bootstrap de 90\\% $[@{u.v.Student-t.lo}; @{u.v.Student-t.hi}]\\%$; NIG: $[@{u.v.NIG.lo}; @{u.v.NIG.hi}]\\%$')]),
    (T('\\textbf{Model uncertainty}: different models on the same data', '\\textbf{Incertitudinea de model}: modele diferite pe aceleași date'),
     [T('median VaR 1\\%: GED $@{u.v.GED.med}\\%$, Student-t $@{u.v.Student-t.med}\\%$, NIG $@{u.v.NIG.med}\\%$, Normal mixture $@{u.v.Mixture.med}\\%$',
        'VaR 1\\% median: GED $@{u.v.GED.med}\\%$, Student-t $@{u.v.Student-t.med}\\%$, NIG $@{u.v.NIG.med}\\%$, amestec Normal $@{u.v.Mixture.med}\\%$'),
      T('the spread between models is comparable to the spread within one model', 'dispersia dintre modele este comparabilă cu dispersia din interiorul unui model')]),
    T('Even when the AIC winner is clear, the number that matters for risk is not', 'Chiar cînd cîștigătorul AIC este clar, cifra relevantă pentru risc rămîne incertă'),
    T('Entropy-based tail measures are another way to summarise this uncertainty: \\refPeleA, \\refPeleB', 'Măsurile de coadă construite pe baza entropiei oferă o altă cale de a rezuma această incertitudine: \\refPeleA, \\refPeleB')))

D.frame(T('Model Averaging and Model Risk', 'Medierea modelelor și riscul de model'), table(
    'lrrrrr', T('& data & Normal & heavy-tailed range & averaged & range (\\%)', '& date & Normală & interval (cozi groase) & mediat & interval (\\%)'),
    [f'{NAMES[k]} & $@{{av.{k}.emp}}$ & $@{{av.{k}.normal}}$ & $[@{{av.{k}.min_heavy}}, @{{av.{k}.max_heavy}}]$ & $@{{av.{k}.avg}}$ & $@{{av.{k}.range}}$' for k in ALL],
    size='scriptsize') + items(
    (T('VaR 1\\% in \\%; heavy-tailed range: the smallest and largest VaR 1\\% of the six non-Normal models', 'VaR 1\\% în \\%; intervalul cozilor groase: cel mai mic și cel mai mare VaR 1\\% dintre cele șase modele nenormale'),
     [T('range (\\%): max/min $- 1$, the relative distance between the extreme models', 'intervalul (\\%): max/min $- 1$, distanța relativă dintre modelele extreme')]),
    (T('Averaged: $\\sum_i w_i\\,\\mathrm{VaR}_i$ with the Akaike weights $w_i$', 'Mediat: $\\sum_i w_i\\,\\mathrm{VaR}_i$, cu ponderile Akaike $w_i$'),
     [T('a frequentist cousin of Bayesian model averaging \\refHoeting', 'un analog frecventist al medierii bayesiene a modelelor \\refHoeting')]),
    T('\\textbf{Model risk}: the range across plausible models; regulators ask banks to measure it \\refKerkhof, \\refDanielsson',
      '\\textbf{Riscul de model}: amplitudinea rezultatelor între modelele plauzibile; autoritățile de supraveghere cer băncilor să îl măsoare \\refKerkhof, \\refDanielsson')) + ql('SFM_ch6_model_uncertainty'), size='footnotesize')

D.frame(T('A Model-Selection Workflow', 'Etapele selecției modelului'), items(
    T('\\textbf{1. Data}: check the largest returns against prices and news; remove only proven errors', '\\textbf{1. Datele}: comparați cele mai mari randamente cu prețurile și cu știrile; eliminați doar erorile dovedite'),
    T('\\textbf{2. Candidates}: choose them for the stylised facts (tails, skewness), not for convenience', '\\textbf{2. Candidații}: alegeți-i după faptele stilizate (cozi, asimetrie), nu după comoditate'),
    T('\\textbf{3. Fit and rank}: ML, then AIC and BIC; LR tests for nested pairs, Vuong for the others', '\\textbf{3. Ajustare și clasament}: ML, apoi AIC și BIC; teste LR pentru perechile imbricate, Vuong pentru celelalte'),
    T('\\textbf{4. Check}: AD statistic and QQ plot of the leaders; the region you care about must fit', '\\textbf{4. Verificare}: statistica AD și graficul QQ pentru primele modele din clasament; modelul trebuie să se potrivească bine în zona care vă interesează'),
    T('\\textbf{5. Validate}: out of sample, with the loss that matches the purpose (exceedances for VaR)', '\\textbf{5. Validare}: în afara eșantionului, cu funcția de pierdere potrivită scopului (depășiri pentru VaR)'),
    T('\\textbf{6. Report}: the winner, the runners-up and the range of the risk number across them', '\\textbf{6. Raportare}: cîștigătorul, modelele de pe locurile următoare și intervalul valorilor de risc între acestea')))

# =============================================================================
# 10. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Does NIG still win after we remove volatility clustering?}', '\\textbf{Mai cîștigă NIG după ce eliminăm efectul de volatility clustering?}'),
     [T('part of the heavy tails of daily returns comes from changing volatility (Chapter 2)', 'o parte din cozile groase ale randamentelor zilnice provine din volatilitatea variabilă în timp (Capitolul 2)'),
      T('returns divided by a volatility forecast (for example EMA, $\\lambda = 0.96$) have lighter tails; which candidate fits them best?',
        'randamentele împărțite la o prognoză de volatilitate (de exemplu EMA, $\\lambda = 0{,}96$) au cozi mai subțiri; care candidat li se potrivește cel mai bine?')]),
    T('Why it is open: the answer differs across markets, and the volatility model itself is a choice', 'Motivul pentru care este deschisă: răspunsul diferă de la o piață la alta, iar modelul de volatilitate este el însuși o alegere'),
    T('Starting point: on raw returns the winner depends on the market (the $\\Delta$AIC map of Section 4)', 'Punctul de plecare: pe randamentele brute, modelul cîștigător depinde de piață (harta $\\Delta$AIC din Secțiunea 4)'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea rezultatelor \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: list studies that compare return distributions after GARCH filtering, with their samples and criteria',
      '\\textbf{Literatura}: lista studiilor care compară distribuțiile randamentelor după filtrarea GARCH, cu eșantioanele și criteriile lor'),
    T('\\textbf{Code}: draft the loop ``standardise, fit seven models, rank by AIC and BIC, test out of sample\'\'', '\\textbf{Cod}: o primă versiune a buclei „standardizează, ajustează șapte modele, ordonează după AIC și BIC, testează în afara eșantionului”'),
    T('\\textbf{Robustness}: propose other volatility filters, windows and markets', '\\textbf{Robustețe}: alte filtre de volatilitate, alte ferestre și alte piețe'),
    T('\\textbf{Explanation}: a first draft of why the ranking changes after filtering', '\\textbf{Explicație}: o primă explicație a schimbării clasamentului după filtrare'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that divides daily log returns by an EMA volatility (lambda = 0.96), fits Normal, Student-t, skewed-t, GED and NIG by maximum likelihood, and reports AIC, BIC and Akaike weights.}',
        '\\aiprompt{Write Python code that divides daily log returns by an EMA volatility (lambda = 0.96), fits Normal, Student-t, skewed-t, GED and NIG by maximum likelihood, and reports AIC, BIC and Akaike weights.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Look-ahead: the volatility of day $t$ must use only returns up to $t - 1$', 'Informația din viitor (look-ahead bias): volatilitatea zilei $t$ trebuie să folosească doar randamentele pînă la $t - 1$'),
    T('Parameter counts: $k$ in AIC must include every estimated parameter, also those of the volatility filter if they were estimated',
      'Numărul de parametri: $k$ din AIC trebuie să includă fiecare parametru estimat, inclusiv pe cei ai filtrului de volatilitate, dacă au fost estimați'),
    T('Same data for all models: AIC values computed on different samples cannot be compared', 'Aceleași date pentru toate modelele: valorile AIC calculate pe eșantioane diferite nu se pot compara'),
    T('Parameterisations: scale vs standard deviation in the Student-t; S0 vs S1 for the stable law', 'Parametrizările: scală sau abatere standard la Student-t; S0 sau S1 la distribuția stabilă'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Seed', 'Idee de proiect'), items(
    (T('\\textbf{Question}: does the best distribution for standardised returns differ from the best one for raw returns?',
       '\\textbf{Întrebarea}: diferă cea mai bună distribuție pentru randamentele standardizate de cea pentru randamentele brute?'),
     [T('data: BET, S\\&P 500, DAX, Bitcoin and BVB stocks from the course data (EODHD)', 'date: BET, S\\&P 500, DAX, Bitcoin și acțiunile BVB din datele cursului (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('standardise by EMA volatility; fit the seven candidates; table of $\\Delta$AIC and Akaike weights', 'standardizați cu volatilitatea EMA; ajustați cei șapte candidați; tabel cu $\\Delta$AIC și ponderile Akaike'),
      T('repeat the out-of-sample VaR 1\\% backtest of this chapter with the standardised model', 'repetați backtesting-ul VaR 1\\% în afara eșantionului din acest capitol cu modelul standardizat'),
      T('compare the exceedance clusters of 2008 and 2020 with those of the unconditional models', 'comparați grupurile de depășiri din 2008 și 2020 cu cele ale modelelor necondiționate')]),
    T('Deliverable: one table, one chart and a paragraph on what changes and why', 'Rezultat: un tabel, un grafic și un paragraf despre ce se schimbă și de ce'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice folosire a AI și enumerați erorile AI pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Model choice starts with clean data and a purpose: the body or the tail?', 'Alegerea modelului începe cu date curate și un scop: corpul sau coada?'),
    T('ML fits; LR tests compare nested models (watch boundaries); Vuong compares non-nested ones', 'ML estimează parametrii; testele LR compară modele imbricate (atenție la frontiere); Vuong le compară pe cele neimbricate'),
    T('AIC and BIC rank all candidates; read $\\Delta$ and the Akaike weights', 'AIC și BIC ordonează toți candidații; citiți $\\Delta$ și ponderile Akaike'),
    T('EDF tests: AD looks at the tails, KS does not; with large $n$ all are rejected, so rank and plot', 'Testele EDF: AD pune accentul pe cozi, KS nu; cu $n$ mare toate modelele sînt respinse, deci ordonați modelele și examinați graficele'),
    T('Real returns: NIG, GED and Student-t win; the Normal and stable laws lose everywhere', 'Pe randamentele reale cîștigă NIG, GED și Student-t; distribuțiile Normală și stabilă pierd peste tot'),
    T('Validate out of sample and report the range of VaR across plausible models', 'Validați în afara eșantionului și raportați intervalul VaR între modelele plauzibile')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Log-likelihood', 'Log-verosimilitatea') + ' & $\\ell(\\theta) = \\sum_t \\ln f(r_t; \\theta)$',
     T('Likelihood ratio', 'Raportul de verosimilitate') + ' & $LR = 2(\\ell_1 - \\ell_0) \\approx \\chi^2(k_1 - k_0)$',
     'AIC, BIC & $-2\\ell(\\hat\\theta) + 2k$, \\quad $-2\\ell(\\hat\\theta) + k\\ln n$',
     T('Akaike weight', 'Ponderea Akaike') + ' & $w_i = e^{-\\Delta_i/2} / \\sum_j e^{-\\Delta_j/2}$',
     'Vuong & $V = \\sqrt{n}\\,\\bar m / s_m$, \\quad $m_t = \\ln f_1(r_t) - \\ln f_2(r_t)$',
     'KS & $D_n = \\sup_x |F_n(x) - F(x)|$',
     'Anderson--Darling & $A^2 = n\\int [F_n - F]^2 / [F(1 - F)]\\, dF$',
     'Kupiec & $LR_{uc} = -2\\ln\\frac{(1-\\alpha)^{n - x}\\alpha^x}{(1 - x/n)^{n - x}(x/n)^x}$',
     T('Pinball loss', 'Pierderea pinball') + ' & $\\frac{1}{n}\\sum_t (\\alpha - \\mathbf{1}\\{r_t < q\\})(r_t - q)$'],
    size='footnotesize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: model A has $\\ell = -1000$ with $k = 3$, model B has $\\ell = -998$ with $k = 5$; which has the lower AIC?',
       '\\textbf{Întrebare}: modelul A are $\\ell = -1000$ cu $k = 3$, modelul B are $\\ell = -998$ cu $k = 5$; care are AIC mai mic?'),
     [T('\\textbf{Answer}: A: $2006$; B: $2006$; a tie ($\\Delta = 0$): the two extra parameters buy exactly their penalty',
        '\\textbf{Răspuns}: A: $2006$; B: $2006$; egalitate ($\\Delta = 0$): cei doi parametri în plus aduc un cîștig egal exact cu penalizarea lor')]),
    (T('\\textbf{Question}: A is nested in B; is B significantly better at 5\\%?', '\\textbf{Întrebare}: A este imbricat în B; este B semnificativ mai bun la 5\\%?'),
     [T('\\textbf{Answer}: $LR = 2 \\cdot 2 = 4 < \\chi^2_{0.95}(2) = @{chi2_95}$: no', '\\textbf{Răspuns}: $LR = 2 \\cdot 2 = 4 < \\chi^2_{0{,}95}(2) = @{chi2_95}$: nu')]),
    (T('\\textbf{Question}: why can a KS test with estimated parameters not use the textbook KS table?', '\\textbf{Întrebare}: de ce nu poate un test KS cu parametri estimați să folosească tabelul KS clasic?'),
     [T('\\textbf{Answer}: the fit pulls $F$ towards the data, so $D_n$ is too small; use Lilliefors or a parametric bootstrap',
        '\\textbf{Răspuns}: estimarea apropie $F$ de date, deci $D_n$ este prea mic; folosiți Lilliefors sau un bootstrap parametric')]),
    T('Next: Chapter 7, efficient markets, the random walk and variance-ratio tests', 'Urmează: Capitolul 7, piețele eficiente, mersul aleator și testele variance ratio')))

D.references(BIB)



def dehyphen(D, V):
    """Value keys may contain model names with a hyphen (Student-t); the @{key} tokens allow only [\\w.]: drop the hyphens."""
    import re
    D.FR = [re.sub(r'@\{([\w.\-]+)\}', lambda m: '@{' + m.group(1).replace('-', '') + '}', x) for x in D.FR]
    W = Values()
    W.update({k.replace('-', ''): v for k, v in V.items()})
    return W


if __name__ == '__main__':
    import re
    for path in D.write(dehyphen(D, V)):
        tex = open(path, encoding='utf-8').read()
        tex = re.sub(r'=\s*<', '<', tex)                 # p = <0.001  ->  p < 0.001
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tex)
