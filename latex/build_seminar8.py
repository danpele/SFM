r"""
build_seminar8.py -- Seminarul 8 (Estimatori de volatilitate și volatility clustering), EN + RO dintr-o singură sursă
==================================================================================================================
Seminarul are loc ÎNAINTEA cursului 8: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_08/sem8_results.json (seminar8.py).
Ieșire:
  EN/Seminars/seminar8_volatility_estimators.tex          (+ _solutions.tex)
  RO/Seminarii/seminar8_estimatori_volatilitate_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_08/seminar8.py && python3 latex/build_seminar8.py && python3 latex/sfm_build.py compile 8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig, n   # noqa: E402
from ch8_common import NAMES, REFS, SHORT, T, bib, fix_refs, load_sem, put_date, pv   # noqa: E402

S = load_sem()
V = Values()
D = Deck(8, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_08}{SFM_ch8_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
a1 = A['a1']
for i, x in enumerate(a1['dev2'], 1):
    V.put(f'a1.d{i}', x, 2)
V.put('a1.ss', a1['ss'], 2)
V.put('a1.var', a1['var'], 3)
V.put('a1.sd', a1['sd'], 3)
V.put('a1.sd0', a1['sd0'], 3)
V.put('a1.ann', a1['ann'], 1)
V.put('a1.rse', 100 * a1['rse'], 1)
V.put('a1.rsed', a1['rse'], 3)
V.put('a1.lo', a1['lo'], 1)
V.put('a1.hi', a1['hi'], 1)
for lab in ['a365', 'a252']:
    d = A['a2'][lab]
    V.put(f'a2.{lab}.ann', d['ann'], 1)
    V.put(f'a2.{lab}.lo', d['lo'], 1)
    V.put(f'a2.{lab}.hi', d['hi'], 1)
d = A['a2']['a365']
V.put('a2.mean', d['mean'], 2)
V.put('a2.ss', d['ss'], 2)
V.put('a2.var', d['var'], 3)
V.put('a2.sd', d['sd'], 3)
V.put('a2.rse', 100 * d['rse'], 1)
V.put('a2.ratio', 100 * (A['a2']['a365']['ann'] / A['a2']['a252']['ann'] - 1), 1)
for a in ['a3', 'a4']:
    d = A[a]
    for i, x in enumerate(d['s2'], 1):
        V.put(f'{a}.s{i}', x, 4)
    V.put(f'{a}.vol', d['vol'][-1], 3)
    V.put(f'{a}.ann', d['ann_last'], 1)
    V.put(f'{a}.hl', d['half_life'], 1)
    V.raw(f'{a}.d99', str(round(d['days99'])))
a5 = A['a5']
for c in ['o', 'u', 'd', 'c', 'cc', 'hl']:
    V.put(f'a5.{c}', a5[c], 3)
for e in ['cc', 'park', 'gk', 'rs']:
    V.put(f'a5.e.{e}', a5['est'][e], 3)
    V.put(f'a5.a.{e}', a5['ann'][e], 1)
a6 = A['a6']
for c in ['o', 'u', 'd', 'c', 'cc']:
    for i, x in enumerate(a6['parts'][c], 1):
        V.put(f'a6.{c}{i}', x, 3)
for e in ['park', 'rs']:
    for i, x in enumerate(a6['daily'][e], 1):
        V.put(f'a6.{e}{i}', x, 3)
V.put('a6.k', a6['k'], 4)
V.put('a6.so', a6['s2_o'], 3)
V.put('a6.sc', a6['s2_c'], 3)
for e in ['cc', 'park', 'gk', 'rs', 'yz']:
    V.put(f'a6.e.{e}', a6['est'][e], 3)
    V.put(f'a6.a.{e}', a6['ann'][e], 1)
for a in ['a7', 'a8']:
    d = A[a]
    for i, x in enumerate(d['terms'], 1):
        V.put(f'{a}.t{i}', x, 2)
    V.put(f'{a}.lb', d['lb'], 2)
    V.raw(f'{a}.plb', pv(d['p_lb']))
    V.put(f'{a}.lm', d['lm'], 2)
    V.raw(f'{a}.plm', pv(d['p_lm']))
    V.put(f'{a}.crit', d['crit_lb'], 2)
a9 = A['a9']
for f in ['A', 'B']:
    for i, (x, y) in enumerate(zip(a9[f]['se'], a9[f]['ql']), 1):
        V.put(f'a9.{f}.se{i}', x, 4)
        V.put(f'a9.{f}.ql{i}', y, 3)
    V.put(f'a9.{f}.mse', a9[f]['mse'], 3)
    V.put(f'a9.{f}.ql', a9[f]['qlike'], 3)
V.put('a10.q05', A['a10']['h05']['qlike'], 3)
V.put('a3.inc', 100 * (A['a3']['s2'][3] / A['a3']['s2'][2] - 1), 0)
V.put('a10.q2', A['a10']['h2']['qlike'], 3)

B1 = S['B1']
V.put('b1.whole', B1['whole'], 1)
V.raw('b1.ppy', str(round(B1['ppy'])))
cols = ['21 days', '63 days', '252 days', 'EWMA 0.94']
keys = ['w21', 'w63', 'w252', 'ewma']
for c, k in zip(cols, keys):
    V.put(f'b1.last.{k}', B1['last'][c], 1)
    V.put(f'b1.max.{k}', B1['max2020'][c], 1)
    put_date(V, f'b1.maxd.{k}', B1['max2020_date'][c])
put_date(V, 'b1.end', B1['last_date'])
put_date(V, 'b1.ghost', B1['ghost_day'])
V.put('b1.g63b', B1['ghost_63_before'], 1)
V.put('b1.g63a', B1['ghost_63_after'], 1)
V.put('b1.geb', B1['ghost_ewma_before'], 1)
V.put('b1.gea', B1['ghost_ewma_after'], 1)
for k, d in S['B2'].items():
    V.raw(f'b2.{k}.ppy', str(round(d['ppy'])))
    V.put(f'b2.{k}.vol', d['vol'], 1)
    V.put(f'b2.{k}.v252d', d['vol252days'], 1)
    V.put(f'b2.{k}.v17', d['v252_2017'], 1)
    V.put(f'b2.{k}.vl', d['v252_last'], 1)
    V.put(f'b2.{k}.max', d['max63'], 1)
    V.raw(f'b2.{k}.maxy', d['max63_date'][:4])
    V.put(f'b2.{k}.ew', d['ewma_last'], 1)
V.put('b2.err', 100 * (S['B2']['btc']['vol'] / S['B2']['btc']['vol252days'] - 1), 1)
V.put('b2.rsp', S['B2']['btc']['v252_last'] / S['B2']['sp500']['v252_last'], 1)
V.put('b2.rbet', S['B2']['btc']['v252_last'] / S['B2']['bet']['v252_last'], 1)


def put_range(prefix, d):
    for e in ['cc', 'park', 'gk', 'rs', 'yz']:
        V.put(f'{prefix}.{e}', d['ann'][e], 1)
    V.put(f'{prefix}.night', 100 * d['night_share'], 0)
    V.put(f'{prefix}.corr', d['corr_oc'], 2)
    V.put(f'{prefix}.vcc', d['var_cc'], 3)
    V.put(f'{prefix}.vo', d['var_o'], 3)
    V.put(f'{prefix}.vc', d['var_c'], 3)
    V.put(f'{prefix}.cov2', d['cov_oc2'], 3)
    V.int(f'{prefix}.n', d['n'])
    V.raw(f'{prefix}.y0', d['first'][:4])


put_range('b3', S['B3'])
for k, d in S['B4'].items():
    put_range(f'b4.{k}', d)
B5 = S['B5']
V.int('b5.T', B5['T'])
for lab in ['rho_r', 'rho_r2', 'rho_abs']:
    for i, x in enumerate(B5[lab], 1):
        V.put(f'b5.{lab}{i}', x, 3)
V.put('b5.lbr', B5['lb_r']['q'], 1)
V.raw('b5.plbr', pv(B5['lb_r']['p']))
V.int('b5.lbr2', round(B5['lb_r2']['q']))
V.int('b5.lbabs', round(B5['lb_abs']['q']))
V.put('b5.crit', B5['lb_r']['crit'], 2)
V.int('b5.lm', round(B5['arch']['lm']))
V.put('b5.r2', B5['arch']['r2'], 3)
V.int('b5.n', B5['arch']['n'])
V.put('b5.critlm', B5['arch']['crit'], 2)
for k, d in S['B6'].items():
    V.int(f'b6.{k}.T', d['T'])
    V.put(f'b6.{k}.rho', d['rho_r2'], 3)
    V.int(f'b6.{k}.lb', round(d['lb_r2']['q']))
    V.int(f'b6.{k}.lm', round(d['arch']['lm']))
    V.put(f'b6.{k}.r2', d['arch']['r2'], 3)
    V.put(f'b6.{k}.k', d['exkurt'], 1)
    V.put(f'b6.{k}.lbs', d['lb_r2_shuffled']['q'], 1)
    V.raw(f'b6.{k}.plbs', pv(d['lb_r2_shuffled']['p']))
    V.put(f'b6.{k}.lms', d['arch_shuffled']['lm'], 1)
B7 = S['B7']
V.int('b7.n', B7['n'])
for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']:
    V.put(f'b7.{m}.mse', B7['loss'][m]['mse'], 2)
    V.put(f'b7.{m}.ql', B7['loss'][m]['qlike'], 3)
V.put('b7.best', B7['best_lambda'], 3)
V.put('b7.bestq', B7['best_q'], 3)
for c in ['dm_vix_ewma', 'dm_ewma_hist21', 'dm_ewma97_ewma94']:
    V.put(f'b7.{c}.t', B7[c]['t'], 2)
    V.raw(f'b7.{c}.p', pv(B7[c]['p']))
for k, d in S['B8'].items():
    V.put(f'b8.{k}.best', d['best'], 3)
    V.put(f'b8.{k}.qb', d['q_best'], 3)
    V.put(f'b8.{k}.q94', d['q094'], 3)
    V.put(f'b8.{k}.q97', d['q097'], 3)
    V.put(f'b8.{k}.hl', d['half_life'], 1)
    V.put(f'b8.{k}.t', d['dm']['t'], 2)
    V.raw(f'b8.{k}.p', pv(d['dm']['p']))
    V.raw(f'b8.{k}.y0', d['first'][:4])
for k, d in S['C1'].items():
    V.put(f'c1.{k}.flat', 100 * d['flat'], 1)
    V.put(f'c1.{k}.pc', d['park_cc'], 2)
    V.put(f'c1.{k}.yc', d['yz_cc'], 2)
    for e in ['cc', 'park', 'yz']:
        V.put(f'c1.{k}.q{e}', d['qlike'][e], 3)
C2 = S['C2']
for c in ['park', 'cc', 'yz', 'half_life', 'btc_sd', 'btc_252', 'btc_ann']:
    V.put(f'c2.{c}', C2[c], 2 if c == 'btc_sd' else 1)
V.put('c2.night', 100 * C2['night'], 0)
V.put('c2.corr', C2['corr_oc'], 2)
V.put('c2.lbr', C2['lb_r'], 1)
V.int('c2.lbr2', round(C2['lb_r2']))
V.put('c2.vix', 100 * C2['vix_above'], 0)
V.raw('c2.ppy', str(round(C2['btc_ppy'])))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how can we measure, from prices, a volatility that we never observe?',
       '\\textbf{Întrebarea}: cum putem măsura, din prețuri, o volatilitate pe care nu o observăm niciodată?'),
     [T('this seminar comes \\textbf{before} Lecture 8: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 8: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: historical and EWMA volatility, range-based estimators, clustering tests and forecast losses, on paper',
        'Partea A: volatilitatea istorică și EWMA, estimatorii de amplitudine, testele de volatility clustering și pierderile prognozelor, pe hîrtie'),
      T('Part B: the S\\&P 500, BET, DAX, Bitcoin and BVB stocks, each task with an interpretation question',
        'Partea B: S\\&P 500, BET, DAX, Bitcoin și acțiuni BVB, fiecare cerință cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('historical volatility, annualisation and its standard error', 'volatilitatea istorică, anualizarea și eroarea ei standard') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('EWMA step by step; half-life', 'EWMA pas cu pas; timpul de înjumătățire') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('Parkinson, Garman--Klass, Rogers--Satchell and Yang--Zhang from OHLC prices', 'Parkinson, Garman--Klass, Rogers--Satchell și Yang--Zhang din prețurile OHLC') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('Ljung--Box on squared returns and the ARCH-LM test', 'Ljung--Box pe randamentele la pătrat și testul ARCH-LM') + ' & ' + SP + ' & A7',
     'A9, A10 & ' + T('MSE and QLIKE losses of volatility forecasts', 'pierderile MSE și QLIKE ale prognozelor de volatilitate') + ' & ' + SP + ' & A9',
     'B1, B2 & ' + T('rolling and EWMA volatility: S\\&P 500; BET, Bitcoin', 'volatilitatea pe ferestre mobile și EWMA: S\\&P 500; BET, Bitcoin') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('range-based estimators: S\\&P 500; DAX, Bitcoin, TLV', 'estimatori de amplitudine: S\\&P 500; DAX, Bitcoin, TLV') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('clustering tests: S\\&P 500; BET, Bitcoin, TLV, SNP', 'teste de volatility clustering: S\\&P 500; BET, Bitcoin, TLV, SNP') + ' & ' + SP + ' & B5',
     'B7, B8 & ' + T('forecast evaluation with the VIX; the best $\\lambda$', 'evaluarea prognozelor cu VIX; cel mai bun $\\lambda$') + ' & ' + SP + ' & B7',
     'C1, C2 & ' + T('which estimator for BVB stocks? what is wrong in an AI answer?', 'ce estimator pentru acțiunile BVB? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3, B7'],
    size='scriptsize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Prices used}', '\\textbf{Prețurile folosite}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500 & EODHD & ' + T('close; OHLC from 2008', 'închidere; OHLC din 2008') + ' & 1990--2026',
     'DAX & EODHD & ' + T('close; OHLC from 2006', 'închidere; OHLC din 2006') + ' & 1990--2026',
     'BET & EODHD & ' + T('close only', 'doar închidere') + ' & 1997--2026',
     'Bitcoin & EODHD & ' + T('OHLC, 7 days a week', 'OHLC, 7 zile pe săptămînă') + ' & 2014--2026',
     T('Banca Transilvania (TLV), OMV Petrom (SNP)', 'Banca Transilvania (TLV), OMV Petrom (SNP)') + ' & EODHD & ' + T('adjusted OHLC', 'OHLC ajustat') + ' & 2010--2026',
     'VIX & EODHD & ' + T('close', 'închidere') + ' & 1990--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin); the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); ultima zi este 18 septembrie 2026'),
    T('Stocks: open, high and low multiplied by the same adjustment factor as the close (dividends, splits); TLV without 30--31 May 2016 (Chapter 2)',
      'Acțiuni: deschiderea, maximul și minimul se înmulțesc cu același factor de ajustare ca închiderea (dividende, splituri); TLV fără 30--31 mai 2016 (Capitolul 2)'),
    T('In the notebook: \\texttt{returns(\'sp500\')}, \\texttt{ohlc(\'btc\')}; no account or key is needed',
      'În notebook: \\texttt{returns(\'sp500\')}, \\texttt{ohlc(\'btc\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Volatility and Its Precision', 'Noțiuni necesare azi (1/5): volatilitatea și precizia ei'), items(
    (T('\\textbf{Volatility}: the standard deviation of returns; with $n$ returns, $\\hat\\sigma = \\sqrt{\\frac{1}{n-1}\\sum_{t=1}^n (r_t - \\bar r)^2}$ (\\% per day)',
       '\\textbf{Volatilitatea}: abaterea standard a randamentelor; cu $n$ randamente, $\\hat\\sigma = \\sqrt{\\frac{1}{n-1}\\sum_{t=1}^n (r_t - \\bar r)^2}$ (\\% pe zi)'),
     [T('zero-mean version for daily data: $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t r_t^2$', 'varianta cu media zero pentru date zilnice: $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t r_t^2$'),
      T('\\textbf{historical volatility}: the same formula on a rolling window of the last $n$ days ($n = 21, 63, 252$)', '\\textbf{volatilitatea istorică}: aceeași formulă pe o fereastră mobilă cu ultimele $n$ zile ($n = 21, 63, 252$)')]),
    (T('\\textbf{Annualisation}: $\\sigma_{\\text{year}} = \\sigma_{\\text{day}}\\sqrt{a}$, $a$ = observations per year', '\\textbf{Anualizarea}: $\\sigma_{\\text{an}} = \\sigma_{\\text{zi}}\\sqrt{a}$, $a$ = numărul de observații pe an'),
     [T('$a \\approx 252$ for stock markets, $a = 365$ for Bitcoin; it assumes uncorrelated returns', '$a \\approx 252$ pentru piețele de acțiuni, $a = 365$ pentru Bitcoin; presupune randamente necorelate')]),
    (T('\\textbf{Precision}: with $n$ i.i.d.\\ Normal returns, $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx 1/\\sqrt{2n}$', '\\textbf{Precizia}: cu $n$ randamente Normale i.i.d., $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx 1/\\sqrt{2n}$'),
     [T('approximate 95\\% interval: $\\hat\\sigma\\,(1 \\pm 1.96/\\sqrt{2n})$', 'interval aproximativ de 95\\%: $\\hat\\sigma\\,(1 \\pm 1{,}96/\\sqrt{2n})$')])))

D.frame(T('What You Need for Today (2/5): EWMA', 'Noțiuni necesare azi (2/5): EWMA'), items(
    (T('\\textbf{EWMA} (exponentially weighted moving average) variance, RiskMetrics \\refRM:', 'Varianța \\textbf{EWMA} (exponentially weighted moving average, medie mobilă ponderată exponențial), RiskMetrics \\refRM:'),
     [T('$\\sigma_t^2 = \\lambda\\,\\sigma_{t-1}^2 + (1-\\lambda)\\,r_{t-1}^2$, with the \\textbf{decay factor} $0 < \\lambda < 1$; $\\lambda = 0.94$ for daily data',
        '$\\sigma_t^2 = \\lambda\\,\\sigma_{t-1}^2 + (1-\\lambda)\\,r_{t-1}^2$, cu \\textbf{factorul de atenuare} $0 < \\lambda < 1$; $\\lambda = 0{,}94$ pentru date zilnice'),
      T('$\\sigma_t^2$ uses returns up to day $t-1$: it is the forecast for day $t$', '$\\sigma_t^2$ folosește randamentele pînă în ziua $t-1$: este prognoza pentru ziua $t$')]),
    (T('Weights: $\\sigma_t^2 = (1-\\lambda)\\sum_{j \\ge 1}\\lambda^{j-1}r_{t-j}^2$', 'Ponderile: $\\sigma_t^2 = (1-\\lambda)\\sum_{j \\ge 1}\\lambda^{j-1}r_{t-j}^2$'),
     [T('\\textbf{half-life} $\\ln 0.5/\\ln\\lambda$: the lag at which the weight halves', '\\textbf{timpul de înjumătățire} $\\ln 0{,}5/\\ln\\lambda$: decalajul la care ponderea se înjumătățește'),
      T('days that carry 99\\% of the weight: $\\ln 0.01/\\ln\\lambda$', 'zilele care cumulează 99\\% din pondere: $\\ln 0{,}01/\\ln\\lambda$')]),
    T('A rolling window gives weight $1/n$ to each of the last $n$ days: when a big day leaves the window, the estimate drops abruptly (the \\textbf{ghost effect})',
      'O fereastră mobilă dă ponderea $1/n$ fiecăreia dintre ultimele $n$ zile: cînd o zi cu o variație mare iese din fereastră, estimarea scade brusc (\\textbf{ghost effect})')))

D.frame(T('What You Need for Today (3/5): Range-Based Estimators', 'Noțiuni necesare azi (3/5): estimatorii de amplitudine'), items(
    (T('\\textbf{OHLC} prices of day $t$: open $O_t$, high $H_t$, low $L_t$, close $C_t$; logs in \\%:', 'Prețurile \\textbf{OHLC} ale zilei $t$: deschiderea $O_t$, maximul $H_t$, minimul $L_t$, închiderea $C_t$; logaritmi în \\%:'),
     [T('overnight $o_t = \\ln(O_t/C_{t-1})$, $u_t = \\ln(H_t/O_t)$, $d_t = \\ln(L_t/O_t)$, $c_t = \\ln(C_t/O_t)$; close-to-close (CC) $r_t = o_t + c_t$',
        'overnight $o_t = \\ln(O_t/C_{t-1})$, $u_t = \\ln(H_t/O_t)$, $d_t = \\ln(L_t/O_t)$, $c_t = \\ln(C_t/O_t)$; închidere--închidere (CC) $r_t = o_t + c_t$')]),
    (T('One-day variance estimates; on $n$ days, take their mean', 'Estimări ale varianței pentru o zi; pe $n$ zile se calculează media lor'),
     [T('Parkinson \\refParkinson: $(u_t - d_t)^2/(4\\ln 2)$; Garman--Klass \\refGK: $0.5(u_t - d_t)^2 - (2\\ln 2 - 1)c_t^2$',
        'Parkinson \\refParkinson: $(u_t - d_t)^2/(4\\ln 2)$; Garman--Klass \\refGK: $0{,}5(u_t - d_t)^2 - (2\\ln 2 - 1)c_t^2$'),
      T('Rogers--Satchell \\refRS: $u_t(u_t - c_t) + d_t(d_t - c_t)$, valid with a drift', 'Rogers--Satchell \\refRS: $u_t(u_t - c_t) + d_t(d_t - c_t)$, valabil și cu tendință (drift)')]),
    (T('Yang--Zhang \\refYZ on $n \\ge 2$ days: $\\hat\\sigma_O^2 + k\\hat\\sigma_C^2 + (1-k)\\hat\\sigma_{RS}^2$, $k = 0.34/(1.34 + (n+1)/(n-1))$',
       'Yang--Zhang \\refYZ pe $n \\ge 2$ zile: $\\hat\\sigma_O^2 + k\\hat\\sigma_C^2 + (1-k)\\hat\\sigma_{RS}^2$, $k = 0{,}34/(1{,}34 + (n+1)/(n-1))$'),
     [T('$\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$: sample variances of $o_t$ and $c_t$; $\\hat\\sigma_{RS}^2$: mean Rogers--Satchell term; Yang--Zhang is the only estimator that includes the night',
        '$\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$: varianțele de selecție ale lui $o_t$ și $c_t$; $\\hat\\sigma_{RS}^2$: media termenului Rogers--Satchell; Yang--Zhang este singurul estimator care include noaptea')])))

D.frame(T('What You Need for Today (4/5): Volatility Clustering', 'Noțiuni necesare azi (4/5): volatility clustering'), items(
    (T('\\textbf{Volatility clustering}: large moves are followed by large moves; returns are almost uncorrelated, but $r_t^2$ and $|r_t|$ are autocorrelated',
       '\\textbf{Volatility clustering}: variațiile mari sînt urmate de variații mari; randamentele sînt aproape necorelate, dar $r_t^2$ și $|r_t|$ sînt autocorelate'),
     [T('sample \\textbf{ACF} (autocorrelation function) $\\hat\\rho_k$; 95\\% band for i.i.d.\\ data: $\\pm 1.96/\\sqrt{T}$',
        '\\textbf{ACF} (autocorrelation function, funcția de autocorelație) de selecție $\\hat\\rho_k$; banda de 95\\% pentru date i.i.d.: $\\pm 1{,}96/\\sqrt{T}$')]),
    (T('\\textbf{McLeod--Li test} \\refML: Ljung--Box \\refLjung on $r_t^2$, $Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k^2/(T-k) \\sim \\chi^2(m)$',
       '\\textbf{Testul McLeod--Li} \\refML: Ljung--Box \\refLjung pe $r_t^2$, $Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k^2/(T-k) \\sim \\chi^2(m)$'),
     [T('$H_0$: no clustering (constant conditional variance); reject if $Q(m) > \\chi^2_{0.95}(m)$', '$H_0$: fără volatility clustering (varianță condiționată constantă); respingem dacă $Q(m) > \\chi^2_{0{,}95}(m)$')]),
    (T('\\textbf{ARCH-LM test} \\refEngle: regress $e_t^2$ ($e_t = r_t - \\bar r$) on a constant and $e_{t-1}^2, \\dots, e_{t-q}^2$', '\\textbf{Testul ARCH-LM} \\refEngle: regresia lui $e_t^2$ ($e_t = r_t - \\bar r$) pe o constantă și pe $e_{t-1}^2, \\dots, e_{t-q}^2$'),
     [T('$\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$, $n$ = observations in the regression; 5\\% critical values: $\\chi^2_{0.95}(5) = 11.07$, $\\chi^2_{0.95}(10) = 18.31$',
        '$\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$, $n$ = numărul de observații din regresie; valori critice de 5\\%: $\\chi^2_{0{,}95}(5) = 11{,}07$, $\\chi^2_{0{,}95}(10) = 18{,}31$')])))

D.frame(T('What You Need for Today (5/5): Evaluating Forecasts', 'Noțiuni necesare azi (5/5): evaluarea prognozelor'), items(
    (T('A forecast $h_t$ of tomorrow\'s variance is compared with a \\textbf{proxy}, usually $r_t^2$ (unbiased but noisy)', 'O prognoză $h_t$ a varianței de mîine se compară cu o variabilă \\textbf{proxy}, de obicei $r_t^2$ (nedeplasată, dar zgomotoasă)'),
     [T('\\textbf{MSE} (mean squared error): mean of $(r_t^2 - h_t)^2$; \\textbf{QLIKE}: mean of $r_t^2/h_t + \\ln h_t$; lower is better \\refPatton',
        '\\textbf{MSE} (mean squared error, eroarea pătratică medie): media lui $(r_t^2 - h_t)^2$; \\textbf{QLIKE}: media lui $r_t^2/h_t + \\ln h_t$; o valoare mai mică este mai bună \\refPatton'),
      T('\\textbf{DM} (Diebold--Mariano) test \\refDM: $t$ statistic of the mean loss difference of two forecasts; $|t| > 1.96$: they differ at 5\\%',
        'testul \\textbf{DM} (Diebold--Mariano) \\refDM: statistica $t$ a diferenței medii a pierderilor a două prognoze; $|t| > 1{,}96$: diferă la 5\\%')]),
    (T('\\textbf{VIX} \\refCboe: the expected volatility of the S\\&P 500 over the next 30 days, implied by option prices, in annualised \\%',
       '\\textbf{VIX} \\refCboe: volatilitatea așteptată a S\\&P 500 pentru următoarele 30 de zile, implicită în prețurile opțiunilor, exprimată în procente anualizate'),
     [T('as a forecast of the daily variance: $h_t = \\mathrm{VIX}_{t-1}^2/a$', 'ca prognoză a varianței zilnice: $h_t = \\mathrm{VIX}_{t-1}^2/a$')]),
    T('Forecasts must use only information available at the close of day $t-1$', 'Prognozele trebuie să folosească doar informația disponibilă la închiderea zilei $t-1$')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Historical Volatility from Five Days', 'A1: volatilitatea istorică din cinci zile'),
         items(T('Daily returns (\\%) of a stock: $1.2$, $-0.8$, $0.5$, $-1.5$, $0.6$.', 'Randamentele zilnice (\\%) ale unei acțiuni: $1{,}2$; $-0{,}8$; $0{,}5$; $-1{,}5$; $0{,}6$.'),
               T('1. Compute the mean and the sample standard deviation (divide by $n - 1$).', '1. Calculați media și abaterea standard de selecție (împărțiți la $n - 1$).'),
               T('2. Annualise the standard deviation with $a = 252$.', '2. Anualizați abaterea standard cu $a = 252$.'),
               T('3. Compute the zero-mean volatility $\\sqrt{\\frac{1}{n}\\sum r_t^2}$.', '3. Calculați volatilitatea cu media zero, $\\sqrt{\\frac{1}{n}\\sum r_t^2}$.'),
               T('4. Compute $1/\\sqrt{2n}$ and an approximate 95\\% interval for the annual volatility.', '4. Calculați $1/\\sqrt{2n}$ și un interval aproximativ de 95\\% pentru volatilitatea anuală.'),
               T('Report: the mean, two daily volatilities, one annual volatility, one interval and one sentence.', 'Raportați: media, două volatilități zilnice, o volatilitate anuală, un interval și o frază.')),
         items(T('1. Mean $= 0/5 = 0$; squared deviations $@{a1.d1}$, $@{a1.d2}$, $@{a1.d3}$, $@{a1.d4}$, $@{a1.d5}$, sum $@{a1.ss}$; $s^2 = @{a1.ss}/4 = @{a1.var}$, $s = @{a1.sd}\\%$',
                 '1. Media $= 0/5 = 0$; abaterile la pătrat $@{a1.d1}$; $@{a1.d2}$; $@{a1.d3}$; $@{a1.d4}$; $@{a1.d5}$, suma $@{a1.ss}$; $s^2 = @{a1.ss}/4 = @{a1.var}$, $s = @{a1.sd}\\%$'),
               T('2. $@{a1.sd} \\times \\sqrt{252} = @{a1.ann}\\%$', '2. $@{a1.sd} \\times \\sqrt{252} = @{a1.ann}\\%$'),
               T('3. $\\sqrt{@{a1.ss}/5} = @{a1.sd0}\\%$: with a zero mean, only the divisor changes', '3. $\\sqrt{@{a1.ss}/5} = @{a1.sd0}\\%$: cu media zero, se schimbă doar numitorul'),
               T('4. $1/\\sqrt{10} = @{a1.rse}\\%$; $@{a1.ann} \\times (1 \\pm 1.96 \\times @{a1.rsed})$: $[@{a1.lo}\\%; @{a1.hi}\\%]$',
                 '4. $1/\\sqrt{10} = @{a1.rse}\\%$; $@{a1.ann} \\times (1 \\pm 1{,}96 \\times @{a1.rsed})$: $[@{a1.lo}\\%; @{a1.hi}\\%]$'),
               T('Five days say almost nothing: the annual volatility could be anywhere between @{a1.lo}\\% and @{a1.hi}\\%.', 'Cinci zile nu spun aproape nimic: volatilitatea anuală poate fi oriunde între @{a1.lo}\\% și @{a1.hi}\\%.')),
         size='scriptsize')

D.proposed(T('A2: Bitcoin over Six Days', 'A2: Bitcoin în șase zile'),
           items(T('Daily Bitcoin returns (\\%): $3.1$, $-4.2$, $1.0$, $5.5$, $-2.4$, $-0.9$. Model: A1.', 'Randamente zilnice Bitcoin (\\%): $3{,}1$; $-4{,}2$; $1{,}0$; $5{,}5$; $-2{,}4$; $-0{,}9$. Model: A1.'),
                 T('1. Compute the mean and the sample standard deviation.', '1. Calculați media și abaterea standard de selecție.'),
                 T('2. Annualise it with $a = 365$ and, wrongly, with $a = 252$.', '2. Anualizați-o cu $a = 365$ și, greșit, cu $a = 252$.'),
                 T('3. Compute the approximate 95\\% interval for the annual volatility ($a = 365$).', '3. Calculați intervalul aproximativ de 95\\% pentru volatilitatea anuală ($a = 365$).'),
                 T('Report: two annual volatilities, one interval and one sentence.', 'Raportați: două volatilități anuale, un interval și o frază.')),
           items(T('1. Mean $@{a2.mean}\\%$; sum of squared deviations $@{a2.ss}$; $s^2 = @{a2.var}$, $s = @{a2.sd}\\%$', '1. Media $@{a2.mean}\\%$; suma abaterilor la pătrat $@{a2.ss}$; $s^2 = @{a2.var}$, $s = @{a2.sd}\\%$'),
                 T('2. $\\sqrt{365}$: $@{a2.a365.ann}\\%$; $\\sqrt{252}$: $@{a2.a252.ann}\\%$, too low: the correct value is $\\sqrt{365/252}$ times larger (by $@{a2.ratio}\\%$)', '2. $\\sqrt{365}$: $@{a2.a365.ann}\\%$; $\\sqrt{252}$: $@{a2.a252.ann}\\%$, prea mică: valoarea corectă este de $\\sqrt{365/252}$ ori mai mare (cu $@{a2.ratio}\\%$)'),
                 T('3. $1/\\sqrt{12} = @{a2.rse}\\%$: $[@{a2.a365.lo}\\%; @{a2.a365.hi}\\%]$; six days cannot pin down the volatility of Bitcoin.', '3. $1/\\sqrt{12} = @{a2.rse}\\%$: $[@{a2.a365.lo}\\%; @{a2.a365.hi}\\%]$; șase zile nu pot fixa volatilitatea Bitcoin.')),
           size='scriptsize')

D.solved(T('A3: EWMA Step by Step', 'A3: EWMA pas cu pas'),
         items(T('$\\lambda = 0.94$; today\'s EWMA variance is $\\sigma_1^2 = 1.00$ (\\%$^2$); the next three returns are $r_1 = -2.0\\%$, $r_2 = 0.5\\%$, $r_3 = 3.0\\%$.',
                 '$\\lambda = 0{,}94$; varianța EWMA de azi este $\\sigma_1^2 = 1{,}00$ (\\%$^2$); următoarele trei randamente sînt $r_1 = -2{,}0\\%$, $r_2 = 0{,}5\\%$, $r_3 = 3{,}0\\%$.'),
               T('1. Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$.', '1. Calculați $\\sigma_2^2$, $\\sigma_3^2$ și $\\sigma_4^2$.'),
               T('2. Annualise $\\sigma_4$ with $a = 252$.', '2. Anualizați $\\sigma_4$ cu $a = 252$.'),
               T('3. Compute the half-life and the weight of $r_3^2$ in $\\sigma_4^2$.', '3. Calculați timpul de înjumătățire și ponderea lui $r_3^2$ în $\\sigma_4^2$.'),
               T('Report: three variances, one annual volatility, the half-life and one sentence.', 'Raportați: trei varianțe, o volatilitate anuală, timpul de înjumătățire și o frază.')),
         items(T('1. $\\sigma_2^2 = 0.94 \\times 1.00 + 0.06 \\times 4.00 = @{a3.s2}$', '1. $\\sigma_2^2 = 0{,}94 \\times 1{,}00 + 0{,}06 \\times 4{,}00 = @{a3.s2}$'),
               T('$\\sigma_3^2 = 0.94 \\times @{a3.s2} + 0.06 \\times 0.25 = @{a3.s3}$; $\\sigma_4^2 = 0.94 \\times @{a3.s3} + 0.06 \\times 9 = @{a3.s4}$',
                 '$\\sigma_3^2 = 0{,}94 \\times @{a3.s2} + 0{,}06 \\times 0{,}25 = @{a3.s3}$; $\\sigma_4^2 = 0{,}94 \\times @{a3.s3} + 0{,}06 \\times 9 = @{a3.s4}$'),
               T('2. $\\sigma_4 = @{a3.vol}\\%$; $@{a3.vol} \\times \\sqrt{252} = @{a3.ann}\\%$', '2. $\\sigma_4 = @{a3.vol}\\%$; $@{a3.vol} \\times \\sqrt{252} = @{a3.ann}\\%$'),
               T('3. $\\ln 0.5/\\ln 0.94 = @{a3.hl}$ days; the newest squared return gets weight $1 - \\lambda = 0.06$', '3. $\\ln 0{,}5/\\ln 0{,}94 = @{a3.hl}$ zile; cel mai nou randament la pătrat primește ponderea $1 - \\lambda = 0{,}06$'),
               T('One 3\\% day raises the variance by @{a3.inc}\\%; a quiet day lowers it by only 6\\%.', 'O zi cu 3\\% crește varianța cu @{a3.inc}\\%; o zi liniștită o scade doar cu 6\\%.')),
         size='scriptsize')

D.proposed(T('A4: A Slower EWMA', 'A4: un EWMA mai lent'),
           items(T('The same data as A3, with $\\lambda = 0.97$ (the RiskMetrics monthly value). Model: A3.', 'Aceleași date ca în A3, cu $\\lambda = 0{,}97$ (valoarea lunară RiskMetrics). Model: A3.'),
                 T('1. Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$ and annualise $\\sigma_4$.', '1. Calculați $\\sigma_2^2$, $\\sigma_3^2$ și $\\sigma_4^2$ și anualizați $\\sigma_4$.'),
                 T('2. Compute the half-life and the number of days that carry 99\\% of the weight.', '2. Calculați timpul de înjumătățire și numărul de zile care cumulează 99\\% din pondere.'),
                 T('3. Compare with A3: which $\\lambda$ reacts faster to the 3\\% day?', '3. Comparați cu A3: care $\\lambda$ reacționează mai repede la ziua cu 3\\%?'),
                 T('Report: three variances, one annual volatility, two numbers of days and one sentence.', 'Raportați: trei varianțe, o volatilitate anuală, două numere de zile și o frază.')),
           items(T('1. $@{a4.s2}$, $@{a4.s3}$, $@{a4.s4}$; $\\sigma_4 = @{a4.vol}\\%$, annual $@{a4.ann}\\%$ (A3: $@{a3.ann}\\%$)', '1. $@{a4.s2}$; $@{a4.s3}$; $@{a4.s4}$; $\\sigma_4 = @{a4.vol}\\%$, anual $@{a4.ann}\\%$ (A3: $@{a3.ann}\\%$)'),
                 T('2. Half-life @{a4.hl} days (A3: @{a3.hl}); 99\\% of the weight in @{a4.d99} days (A3: @{a3.d99})', '2. Timp de înjumătățire @{a4.hl} zile (A3: @{a3.hl}); 99\\% din pondere în @{a4.d99} de zile (A3: @{a3.d99})'),
                 T('3. $\\lambda = 0.94$ reacts faster; $\\lambda = 0.97$ is smoother and remembers longer.', '3. EWMA cu $\\lambda = 0{,}94$ reacționează mai repede; cu $\\lambda = 0{,}97$ este mai neted și are o memorie mai lungă.')),
           size='scriptsize')

D.solved(T('A5: Four Estimators from One Day', 'A5: patru estimatori dintr-o singură zi'),
         items(T('Previous close $100.0$; open $101.0$, high $103.5$, low $99.8$, close $102.2$.', 'Închiderea precedentă $100{,}0$; deschiderea $101{,}0$, maximul $103{,}5$, minimul $99{,}8$, închiderea $102{,}2$.'),
               T('1. Compute $o$, $u$, $d$, $c$ (in \\%) and $r = o + c$.', '1. Calculați $o$, $u$, $d$, $c$ (în \\%) și $r = o + c$.'),
               T('2. Compute the one-day CC, Parkinson, Garman--Klass and Rogers--Satchell variances.', '2. Calculați varianțele pentru o zi CC, Parkinson, Garman--Klass și Rogers--Satchell.'),
               T('3. Annualise each with $a = 252$, as if every day looked like this one.', '3. Anualizați-le cu $a = 252$, ca și cum toate zilele ar arăta ca aceasta.'),
               T('4. Say which estimators see the overnight move $o$.', '4. Precizați care estimatori văd mișcarea overnight $o$.'),
               T('Report: five log returns, four variances, four annual volatilities and one sentence.', 'Raportați: cinci randamente logaritmice, patru varianțe, patru volatilități anuale și o frază.')),
         items(T('1. $o = 100\\ln(101/100) = @{a5.o}$, $u = @{a5.u}$, $d = @{a5.d}$, $c = @{a5.c}$, $r = @{a5.cc}$', '1. $o = 100\\ln(101/100) = @{a5.o}$, $u = @{a5.u}$, $d = @{a5.d}$, $c = @{a5.c}$, $r = @{a5.cc}$'),
               T('2. CC $r^2 = @{a5.e.cc}$; Parkinson $@{a5.hl}^2/2.773 = @{a5.e.park}$; GK $0.5 \\times @{a5.hl}^2 - 0.386 \\times @{a5.c}^2 = @{a5.e.gk}$; RS $@{a5.e.rs}$',
                 '2. CC $r^2 = @{a5.e.cc}$; Parkinson $@{a5.hl}^2/2{,}773 = @{a5.e.park}$; GK $0{,}5 \\times @{a5.hl}^2 - 0{,}386 \\times @{a5.c}^2 = @{a5.e.gk}$; RS $@{a5.e.rs}$'),
               T('3. $\\sqrt{252\\,\\hat\\sigma^2}$: CC $@{a5.a.cc}\\%$, Parkinson $@{a5.a.park}\\%$, GK $@{a5.a.gk}\\%$, RS $@{a5.a.rs}\\%$', '3. $\\sqrt{252\\,\\hat\\sigma^2}$: CC $@{a5.a.cc}\\%$, Parkinson $@{a5.a.park}\\%$, GK $@{a5.a.gk}\\%$, RS $@{a5.a.rs}\\%$'),
               T('4. Only CC (and Yang--Zhang, on several days) include $o$; Parkinson, GK and RS see only the session.', '4. Doar CC (și Yang--Zhang, pe mai multe zile) includ $o$; Parkinson, GK și RS văd doar ședința.')),
         size='scriptsize')

D.proposed(T('A6: Three Days and the Yang--Zhang Estimator', 'A6: trei zile și estimatorul Yang--Zhang'),
           table('lrrrr', T('\\textbf{Day} & \\textbf{Open} & \\textbf{High} & \\textbf{Low} & \\textbf{Close}', '\\textbf{Ziua} & \\textbf{Deschidere} & \\textbf{Maxim} & \\textbf{Minim} & \\textbf{Închidere}'),
                 [f'{i} & ' + ' & '.join(n(x, 1) for x in row) for i, row in enumerate([(50.2, 51.0, 49.6, 50.8), (50.5, 51.9, 50.3, 51.6), (51.9, 52.4, 50.9, 51.1)], 1)],
                 size='scriptsize') + items(
               T('Previous close: $50.0$. Model: A5.', 'Închiderea precedentă: $50{,}0$. Model: A5.'),
               T('1. Compute $o$, $u$, $d$, $c$ for each day.', '1. Calculați $o$, $u$, $d$, $c$ pentru fiecare zi.'),
               T('2. Compute the Parkinson and Rogers--Satchell terms of each day and their means.', '2. Calculați termenii Parkinson și Rogers--Satchell ai fiecărei zile și mediile lor.'),
               T('3. Compute $k$, $\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$ and the Yang--Zhang variance for $n = 3$.', '3. Calculați $k$, $\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$ și varianța Yang--Zhang pentru $n = 3$.'),
               T('4. Compare with the CC sample variance and annualise all three with $a = 252$.', '4. Comparați cu varianța de selecție CC și anualizați toate trei cu $a = 252$.'),
               T('Report: a small table, three variances, three annual volatilities and one sentence.', 'Raportați: un tabel mic, trei varianțe, trei volatilități anuale și o frază.')),
           items(T('1. $o$: $@{a6.o1}$, $@{a6.o2}$, $@{a6.o3}$; $u$: $@{a6.u1}$, $@{a6.u2}$, $@{a6.u3}$; $d$: $@{a6.d1}$, $@{a6.d2}$, $@{a6.d3}$; $c$: $@{a6.c1}$, $@{a6.c2}$, $@{a6.c3}$',
                   '1. $o$: $@{a6.o1}$; $@{a6.o2}$; $@{a6.o3}$; $u$: $@{a6.u1}$; $@{a6.u2}$; $@{a6.u3}$; $d$: $@{a6.d1}$; $@{a6.d2}$; $@{a6.d3}$; $c$: $@{a6.c1}$; $@{a6.c2}$; $@{a6.c3}$'),
                 T('2. Parkinson: $@{a6.park1}$, $@{a6.park2}$, $@{a6.park3}$, mean $@{a6.e.park}$; RS: $@{a6.rs1}$, $@{a6.rs2}$, $@{a6.rs3}$, mean $@{a6.e.rs}$',
                   '2. Parkinson: $@{a6.park1}$; $@{a6.park2}$; $@{a6.park3}$, media $@{a6.e.park}$; RS: $@{a6.rs1}$; $@{a6.rs2}$; $@{a6.rs3}$, media $@{a6.e.rs}$'),
                 T('3. $k = 0.34/(1.34 + 2) = @{a6.k}$; $\\hat\\sigma_O^2 = @{a6.so}$, $\\hat\\sigma_C^2 = @{a6.sc}$; YZ $= @{a6.so} + @{a6.k} \\times @{a6.sc} + (1 - @{a6.k}) \\times @{a6.e.rs} = @{a6.e.yz}$',
                   '3. $k = 0{,}34/(1{,}34 + 2) = @{a6.k}$; $\\hat\\sigma_O^2 = @{a6.so}$, $\\hat\\sigma_C^2 = @{a6.sc}$; YZ $= @{a6.so} + @{a6.k} \\times @{a6.sc} + (1 - @{a6.k}) \\times @{a6.e.rs} = @{a6.e.yz}$'),
                 T('4. CC $@{a6.e.cc}$; annual: CC $@{a6.a.cc}\\%$, Parkinson $@{a6.a.park}\\%$, YZ $@{a6.a.yz}\\%$; with three days CC is very noisy: the ranges carry more information.',
                   '4. CC $@{a6.e.cc}$; anual: CC $@{a6.a.cc}\\%$, Parkinson $@{a6.a.park}\\%$, YZ $@{a6.a.yz}\\%$; cu trei zile CC este foarte zgomotos: amplitudinile conțin mai multă informație.')),
           size='scriptsize')

D.solved(T('A7: Two Tests for Clustering', 'A7: două teste de volatility clustering'),
         items(T('$T = 1000$ daily returns; the ACF of the \\textbf{squared} returns at lags 1--5: $0.20$, $0.15$, $0.12$, $0.10$, $0.08$; the ARCH-LM regression with $q = 5$ lags on $n = 995$ observations has $R^2 = 0.08$.',
                 '$T = 1000$ de randamente zilnice; ACF-ul randamentelor \\textbf{la pătrat} la decalajele 1--5: $0{,}20$; $0{,}15$; $0{,}12$; $0{,}10$; $0{,}08$; regresia ARCH-LM cu $q = 5$ decalaje pe $n = 995$ de observații are $R^2 = 0{,}08$.'),
               T('1. Compute the Ljung--Box statistic $Q(5)$ of the squared returns.', '1. Calculați statistica Ljung--Box $Q(5)$ a randamentelor la pătrat.'),
               T('2. Compute $\\mathrm{LM} = nR^2$.', '2. Calculați $\\mathrm{LM} = nR^2$.'),
               T('3. Compare both with $\\chi^2_{0.95}(5) = 11.07$ and decide.', '3. Comparați ambele statistici cu $\\chi^2_{0{,}95}(5) = 11{,}07$ și decideți.'),
               T('Report: two statistics, two decisions and one sentence.', 'Raportați: două statistici, două decizii și o frază.')),
         items(T('1. Terms $1000 \\cdot 1002\\,\\hat\\rho_k^2/(1000 - k)$: $@{a7.t1}$, $@{a7.t2}$, $@{a7.t3}$, $@{a7.t4}$, $@{a7.t5}$; $Q(5) = @{a7.lb}$',
                 '1. Termenii $1000 \\cdot 1002\\,\\hat\\rho_k^2/(1000 - k)$: $@{a7.t1}$; $@{a7.t2}$; $@{a7.t3}$; $@{a7.t4}$; $@{a7.t5}$; $Q(5) = @{a7.lb}$'),
               T('2. $\\mathrm{LM} = 995 \\times 0.08 = @{a7.lm}$', '2. $\\mathrm{LM} = 995 \\times 0{,}08 = @{a7.lm}$'),
               T('3. Both far above $@{a7.crit}$ (both p @{a7.plb}): reject $H_0$; the variance is predictable from the recent past.',
                 '3. Ambele mult peste $@{a7.crit}$ (ambele cu p @{a7.plb}): respingem $H_0$; varianța poate fi anticipată din trecutul recent.')),
         size='scriptsize')

D.proposed(T('A8: When the Two Tests Disagree', 'A8: cînd cele două teste nu sînt de acord'),
           items(T('$T = 500$; ACF of the squared returns at lags 1--5: $0.12$, $0.08$, $0.06$, $0.05$, $0.04$; ARCH-LM with $q = 5$ on $n = 495$ observations: $R^2 = 0.018$. Model: A7.',
                   '$T = 500$; ACF-ul randamentelor la pătrat la decalajele 1--5: $0{,}12$; $0{,}08$; $0{,}06$; $0{,}05$; $0{,}04$; ARCH-LM cu $q = 5$ pe $n = 495$ de observații: $R^2 = 0{,}018$. Model: A7.'),
                 T('1. Compute $Q(5)$ and decide at 5\\%.', '1. Calculați $Q(5)$ și decideți la 5\\%.'),
                 T('2. Compute LM and decide at 5\\%.', '2. Calculați LM și decideți la 5\\%.'),
                 T('3. Give one reason why the two tests can disagree.', '3. Indicați un motiv pentru care cele două teste pot da rezultate diferite.'),
                 T('Report: two statistics, two decisions and one sentence.', 'Raportați: două statistici, două decizii și o frază.')),
           items(T('1. Terms $@{a8.t1}$, $@{a8.t2}$, $@{a8.t3}$, $@{a8.t4}$, $@{a8.t5}$; $Q(5) = @{a8.lb} > @{a8.crit}$ (p = @{a8.plb}): reject',
                   '1. Termenii $@{a8.t1}$; $@{a8.t2}$; $@{a8.t3}$; $@{a8.t4}$; $@{a8.t5}$; $Q(5) = @{a8.lb} > @{a8.crit}$ (p = @{a8.plb}): respingem'),
                 T('2. $\\mathrm{LM} = 495 \\times 0.018 = @{a8.lm} < @{a8.crit}$ (p = @{a8.plm}): do not reject', '2. $\\mathrm{LM} = 495 \\times 0{,}018 = @{a8.lm} < @{a8.crit}$ (p = @{a8.plm}): nu respingem'),
                 T('3. Borderline evidence: LB adds up single autocorrelations, LM measures what the five lags explain together (they overlap); with $T = 500$ both have limited power.',
                   '3. Dovezi la limită: LB adună autocorelațiile individuale, LM măsoară ce explică împreună cele cinci decalaje (care se suprapun); cu $T = 500$ ambele au putere limitată.')),
           size='scriptsize')

D.solved(T('A9: MSE and QLIKE by Hand', 'A9: MSE și QLIKE pas cu pas'),
         items(T('Over four days the proxy $r_t^2$ is $1.0$, $4.0$, $0.25$, $2.25$ (\\%$^2$). Forecast A is flat at $1.5$; forecast B is $1.0$, $2.5$, $0.8$, $1.8$.',
                 'În patru zile variabila proxy $r_t^2$ ia valorile $1{,}0$; $4{,}0$; $0{,}25$; $2{,}25$ (\\%$^2$). Prognoza A este constantă, $1{,}5$; prognoza B este $1{,}0$; $2{,}5$; $0{,}8$; $1{,}8$.'),
               T('1. Compute the MSE of each forecast.', '1. Calculați MSE pentru fiecare prognoză.'),
               T('2. Compute the QLIKE terms $r_t^2/h_t + \\ln h_t$ and their mean for each forecast.', '2. Calculați termenii QLIKE $r_t^2/h_t + \\ln h_t$ și media lor pentru fiecare prognoză.'),
               T('3. Say which forecast is better under each loss.', '3. Precizați care prognoză este mai bună după fiecare funcție de pierdere.'),
               T('Report: two MSE values, two QLIKE values and one sentence.', 'Raportați: două valori MSE, două valori QLIKE și o frază.')),
         items(T('1. A: $(1 - 1.5)^2 = @{a9.A.se1}$, $@{a9.A.se2}$, $@{a9.A.se3}$, $@{a9.A.se4}$, MSE $@{a9.A.mse}$; B: $@{a9.B.se1}$, $@{a9.B.se2}$, $@{a9.B.se3}$, $@{a9.B.se4}$, MSE $@{a9.B.mse}$',
                 '1. A: $(1 - 1{,}5)^2 = @{a9.A.se1}$; $@{a9.A.se2}$; $@{a9.A.se3}$; $@{a9.A.se4}$, MSE $@{a9.A.mse}$; B: $@{a9.B.se1}$; $@{a9.B.se2}$; $@{a9.B.se3}$; $@{a9.B.se4}$, MSE $@{a9.B.mse}$'),
               T('2. A: $1/1.5 + \\ln 1.5 = @{a9.A.ql1}$, $@{a9.A.ql2}$, $@{a9.A.ql3}$, $@{a9.A.ql4}$, mean $@{a9.A.ql}$; B: $@{a9.B.ql1}$, $@{a9.B.ql2}$, $@{a9.B.ql3}$, $@{a9.B.ql4}$, mean $@{a9.B.ql}$',
                 '2. A: $1/1{,}5 + \\ln 1{,}5 = @{a9.A.ql1}$; $@{a9.A.ql2}$; $@{a9.A.ql3}$; $@{a9.A.ql4}$, media $@{a9.A.ql}$; B: $@{a9.B.ql1}$; $@{a9.B.ql2}$; $@{a9.B.ql3}$; $@{a9.B.ql4}$, media $@{a9.B.ql}$'),
               T('3. B is better under both losses: it follows the changes of the variance; a flat forecast misses them.', '3. B este mai bună după ambele pierderi: urmărește schimbările varianței; o prognoză constantă le ratează.')),
         size='scriptsize')

D.proposed(T('A10: Under- and Over-Prediction', 'A10: subestimare și supraestimare'),
           items(T('The true variance and the proxy are both $1$. Two forecasts: $h = 0.5$ (half) and $h = 2$ (double). Model: A9.', 'Varianța reală și variabila proxy sînt ambele egale cu $1$. Două prognoze: $h = 0{,}5$ (jumătate) și $h = 2$ (dublu). Model: A9.'),
                 T('1. Compute the MSE of each forecast.', '1. Calculați MSE pentru fiecare prognoză.'),
                 T('2. Compute the QLIKE of each forecast.', '2. Calculați QLIKE pentru fiecare prognoză.'),
                 T('3. Say which loss punishes under-prediction more.', '3. Precizați care funcție de pierdere penalizează mai mult subestimarea.'),
                 T('Report: four numbers and one sentence on why this matters for VaR.', 'Raportați: patru valori și o frază despre importanța acestui fapt pentru VaR.')),
           items(T('1. MSE: $0.25$ for $h = 0.5$, $1$ for $h = 2$', '1. MSE: $0{,}25$ pentru $h = 0{,}5$, $1$ pentru $h = 2$'),
                 T('2. QLIKE: $2 + \\ln 0.5 = @{a10.q05}$ for $h = 0.5$; $0.5 + \\ln 2 = @{a10.q2}$ for $h = 2$', '2. QLIKE: $2 + \\ln 0{,}5 = @{a10.q05}$ pentru $h = 0{,}5$; $0{,}5 + \\ln 2 = @{a10.q2}$ pentru $h = 2$'),
                 T('3. QLIKE punishes under-prediction more; an under-predicted variance gives a VaR that is too small, the costly error for a bank.', '3. QLIKE penalizează mai mult subestimarea; o varianță subestimată dă un VaR prea mic, eroarea costisitoare pentru o bancă.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Historical and EWMA Volatility of the S\\&P 500 [Solved]', 'B1: volatilitatea istorică și volatilitatea EWMA pentru S\\&P 500 [Rezolvat]'),
       T('how different are the 21-, 63- and 252-day and EWMA volatilities of the S\\&P 500?',
         'cît de diferite sînt volatilitățile S\\&P 500 pe 21, 63 și 252 de zile și volatilitatea EWMA?'),
       T('S\\&P 500 closes since 1990, daily log returns in \\%', 'închiderile S\\&P 500 din 1990, randamente logaritmice zilnice în \\%'),
       [T('Compute the rolling 21-, 63- and 252-day volatility and the EWMA volatility ($\\lambda = 0.94$), annualised with the actual frequency.',
          'Calculați volatilitatea pe ferestre mobile de 21, 63 și 252 de zile și volatilitatea EWMA ($\\lambda = 0{,}94$), anualizate cu frecvența reală.'),
        T('Draw the four series since 2018.', 'Desenați cele patru serii din 2018.'),
        T('Report the four values at the end of the sample and their maxima in 2020, with dates.', 'Raportați cele patru valori de la sfîrșitul eșantionului și maximele lor din 2020, cu datele.'),
        T('Find the day of June 2020 when the 63-day volatility falls most, and report the change of EWMA on that day.', 'Găsiți ziua din iunie 2020 în care volatilitatea pe 63 de zile scade cel mai mult și raportați variația EWMA din acea zi.'),
        T('Interpretation: which estimate would you report to a risk committee on that day?', 'Interpretare: ce estimare ați prezenta unui comitet de risc în acea zi?')],
       T('the chart, twelve numbers and two sentences', 'graficul, douăsprezece valori și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch8_sem_b1', h='0.34') + items(
    T('$a = @{b1.ppy}$; whole sample @{b1.whole}\\%; on @{b1.end}: 21 days @{b1.last.w21}\\%, 63 days @{b1.last.w63}\\%, 252 days @{b1.last.w252}\\%, EWMA @{b1.last.ewma}\\%',
      '$a = @{b1.ppy}$; întregul eșantion @{b1.whole}\\%; pe @{b1.end}: 21 de zile @{b1.last.w21}\\%, 63 de zile @{b1.last.w63}\\%, 252 de zile @{b1.last.w252}\\%, EWMA @{b1.last.ewma}\\%'),
    T('2020 maxima: @{b1.max.w21}\\% (@{b1.maxd.w21}), @{b1.max.w63}\\% (@{b1.maxd.w63}), @{b1.max.w252}\\% (@{b1.maxd.w252}), EWMA @{b1.max.ewma}\\% (@{b1.maxd.ewma})',
      'Maximele din 2020: @{b1.max.w21}\\% (@{b1.maxd.w21}), @{b1.max.w63}\\% (@{b1.maxd.w63}), @{b1.max.w252}\\% (@{b1.maxd.w252}), EWMA @{b1.max.ewma}\\% (@{b1.maxd.ewma})'),
    T('On @{b1.ghost} the 63-day volatility fell from @{b1.g63b}\\% to @{b1.g63a}\\% (the crash day of 16 March left the window); EWMA moved from @{b1.geb}\\% to @{b1.gea}\\%',
      'Pe @{b1.ghost}, volatilitatea pe 63 de zile a scăzut de la @{b1.g63b}\\% la @{b1.g63a}\\% (ziua crahului din 16 martie a ieșit din fereastră); EWMA a trecut de la @{b1.geb}\\% la @{b1.gea}\\%'),
    T('Interpretation: EWMA; the drop of the 63-day estimate is the ghost effect, not news; the 252-day value still reflects March', 'Interpretare: EWMA; scăderea estimării pe 63 de zile este ghost effect, nu o știre; valoarea pe 252 de zile reflectă încă luna martie')) + qlsem(), 'scriptsize')

D.task(T('B2: BET and Bitcoin [Proposed]', 'B2: BET și Bitcoin [Propus]'),
       T('how volatile are the BET and Bitcoin compared with the S\\&P 500, today and at the end of 2017? Model: B1.',
         'cît de volatile sînt BET și Bitcoin în comparație cu S\\&P 500, azi și la sfîrșitul lui 2017? Model: B1.'),
       T('BET since 1997, Bitcoin since 2014, S\\&P 500 since 1990, daily log returns in \\%', 'BET din 1997, Bitcoin din 2014, S\\&P 500 din 1990, randamente logaritmice zilnice în \\%'),
       [T('Compute the number of observations per year and the whole-sample annual volatility of each series.', 'Calculați numărul de observații pe an și volatilitatea anuală pe întregul eșantion pentru fiecare serie.'),
        T('Annualise Bitcoin also with 252 and give the error in \\%.', 'Anualizați Bitcoin și cu 252 și dați eroarea în \\%.'),
        T('Report the 252-day volatility at the end of 2017 and at the end of the sample, and the maximum of the 63-day volatility with its year.',
          'Raportați volatilitatea pe 252 de zile la sfîrșitul lui 2017 și la sfîrșitul eșantionului, precum și maximul volatilității pe 63 de zile, cu anul lui.'),
        T('Interpretation: has Bitcoin become an asset with equity-like volatility?', 'Interpretare: a devenit Bitcoin un activ cu o volatilitate asemănătoare acțiunilor?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrr', T('& $a$ & \\textbf{Annual vol.} & \\textbf{252 days, 2017} & \\textbf{252 days, end} & \\textbf{Max 63 days} & \\textbf{Year}',
                 '& $a$ & \\textbf{Vol. anuală} & \\textbf{252 de zile, 2017} & \\textbf{252 de zile, final} & \\textbf{Max. 63 de zile} & \\textbf{Anul}'),
    [f'{NAMES[k]} & @{{b2.{k}.ppy}} & @{{b2.{k}.vol}}\\% & @{{b2.{k}.v17}}\\% & @{{b2.{k}.vl}}\\% & @{{b2.{k}.max}}\\% & @{{b2.{k}.maxy}}' for k in ['sp500', 'bet', 'btc']],
    size='footnotesize') + items(
    T('Bitcoin with $\\sqrt{252}$: @{b2.btc.v252d}\\% instead of @{b2.btc.vol}\\%: the correct value is @{b2.err}\\% higher', 'Bitcoin cu $\\sqrt{252}$: @{b2.btc.v252d}\\% în loc de @{b2.btc.vol}\\%: valoarea corectă este cu @{b2.err}\\% mai mare'),
    T('Interpretation: no; Bitcoin\'s 252-day volatility fell from @{b2.btc.v17}\\% (end of 2017) to @{b2.btc.vl}\\%, still @{b2.rsp} times that of the S\\&P 500 and @{b2.rbet} times that of the BET',
      'Interpretare: nu; volatilitatea Bitcoin pe 252 de zile a scăzut de la @{b2.btc.v17}\\% (sfîrșitul lui 2017) la @{b2.btc.vl}\\%, încă de @{b2.rsp} ori mai mare decît cea a S\\&P 500 și de @{b2.rbet} ori mai mare decît cea a BET')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Range-Based Estimators for the S\\&P 500 [Solved]', 'B3: estimatori de amplitudine pentru S\\&P 500 [Rezolvat]'),
       T('do range-based estimators give the same volatility as closing prices for the S\\&P 500?', 'dau estimatorii de amplitudine aceeași volatilitate ca prețurile de închidere pentru S\\&P 500?'),
       T('S\\&P 500 OHLC since @{b3.y0} ($T = @{b3.n}$ days)', 'OHLC S\\&P 500 din @{b3.y0} ($T = @{b3.n}$ de zile)'),
       [T('Compute $o_t$, $u_t$, $d_t$, $c_t$ and the five estimators on the whole sample, annualised.', 'Calculați $o_t$, $u_t$, $d_t$, $c_t$ și cei cinci estimatori pe întregul eșantion, anualizați.'),
        T('Decompose $\\mathrm{Var}(r_t) = \\mathrm{Var}(o_t) + \\mathrm{Var}(c_t) + 2\\,\\mathrm{Cov}(o_t, c_t)$.', 'Descompuneți $\\mathrm{Var}(r_t) = \\mathrm{Var}(o_t) + \\mathrm{Var}(c_t) + 2\\,\\mathrm{Cov}(o_t, c_t)$.'),
        T('Draw the rolling 21-day estimates since 2025.', 'Desenați estimările pe ferestre mobile de 21 de zile din 2025.'),
        T('Interpretation: why are Parkinson, Garman--Klass and Rogers--Satchell below close-to-close?', 'Interpretare: de ce sînt Parkinson, Garman--Klass și Rogers--Satchell sub estimatorul închidere--închidere?')],
       T('the chart, five volatilities, a decomposition and two sentences', 'graficul, cinci volatilități, o descompunere și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch8_sem_b3', h='0.34') + items(
    T('Annual volatility: CC @{b3.cc}\\%, Parkinson @{b3.park}\\%, GK @{b3.gk}\\%, RS @{b3.rs}\\%, YZ @{b3.yz}\\%', 'Volatilitatea anuală: CC @{b3.cc}\\%, Parkinson @{b3.park}\\%, GK @{b3.gk}\\%, RS @{b3.rs}\\%, YZ @{b3.yz}\\%'),
    T('Daily variances (\\%$^2$): $@{b3.vcc} = @{b3.vo} + @{b3.vc} + @{b3.cov2}$; the night is @{b3.night}\\% of $\\mathrm{Var}(o) + \\mathrm{Var}(c)$; $\\mathrm{corr}(o, c) = @{b3.corr}$',
      'Varianțe zilnice (\\%$^2$): $@{b3.vcc} = @{b3.vo} + @{b3.vc} + @{b3.cov2}$; noaptea reprezintă @{b3.night}\\% din $\\mathrm{Var}(o) + \\mathrm{Var}(c)$; $\\mathrm{corr}(o, c) = @{b3.corr}$'),
    T('Interpretation: these estimators see only the trading session, so they miss the overnight variance and the covariance term; YZ adds the night but not the covariance',
      'Interpretare: acești estimatori văd doar ședința de tranzacționare, deci ratează varianța overnight și termenul de covarianță; YZ adaugă noaptea, dar nu și covarianța')) + qlsem(), 'scriptsize')

D.task(T('B4: DAX, Bitcoin and TLV [Proposed]', 'B4: DAX, Bitcoin și TLV [Propus]'),
       T('for which of the DAX, Bitcoin and Banca Transilvania do range-based estimators agree with close-to-close? Model: B3.',
         'pentru care dintre DAX, Bitcoin și Banca Transilvania sînt estimatorii de amplitudine apropiați de estimatorul închidere--închidere? Model: B3.'),
       T('DAX OHLC since 2006, Bitcoin since 2014, TLV adjusted OHLC since 2010', 'OHLC DAX din 2006, Bitcoin din 2014, OHLC ajustat TLV din 2010'),
       [T('Compute the five annualised estimators for each asset.', 'Calculați cei cinci estimatori anualizați pentru fiecare activ.'),
        T('Compute the share of the night and $\\mathrm{corr}(o_t, c_t)$.', 'Calculați ponderea nopții și $\\mathrm{corr}(o_t, c_t)$.'),
        T('Rank the assets by the ratio Parkinson / CC.', 'Ordonați activele după raportul Parkinson / CC.'),
        T('Interpretation: for which asset could you replace close-to-close by Parkinson?', 'Interpretare: pentru ce activ ați putea înlocui estimatorul închidere--închidere cu Parkinson?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(k):
    return (f'{SHORT[k]} & @{{b4.{k}.cc}} & @{{b4.{k}.park}} & @{{b4.{k}.gk}} & @{{b4.{k}.rs}} & @{{b4.{k}.yz}} & @{{b4.{k}.night}}\\% & ${{@{{b4.{k}.corr}}}}$')


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& CC & P & GK & RS & YZ & \\textbf{Night share} & $\\mathrm{corr}(o, c)$', '& CC & P & GK & RS & YZ & \\textbf{Ponderea nopții} & $\\mathrm{corr}(o, c)$'),
    [b4row(k) for k in ['dax', 'btc', 'tlv']], size='footnotesize') + items(
    T('Bitcoin: no night (24-hour market), all estimators within a few percentage points of CC', 'Bitcoin: fără noapte (piață deschisă 24 de ore), toți estimatorii diferă de CC cu cel mult cîteva puncte procentuale'),
    T('DAX and TLV: a large night share; Parkinson, GK and RS far below CC, YZ close to it', 'DAX și TLV: o pondere mare a nopții; Parkinson, GK și RS mult sub CC, YZ apropiat de CC'),
    T('Interpretation: Bitcoin; for markets that close at night use Yang--Zhang, never Parkinson alone', 'Interpretare: Bitcoin; pentru piețele care se închid noaptea folosiți Yang--Zhang, niciodată doar Parkinson')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Clustering in the S\\&P 500 [Solved]', 'B5: volatility clustering la S\\&P 500 [Rezolvat]'),
       T('are the size and the sign of S\\&P 500 returns equally predictable?', 'sînt mărimea și semnul randamentelor S\\&P 500 la fel de previzibile?'),
       T('S\\&P 500 closes since 1990, daily log returns in \\%', 'închiderile S\\&P 500 din 1990, randamente logaritmice zilnice în \\%'),
       [T('Draw the ACF of $r_t$, $r_t^2$ and $|r_t|$ for lags 1--50 with the i.i.d.\\ 95\\% band.', 'Desenați ACF-ul lui $r_t$, $r_t^2$ și $|r_t|$ pentru decalajele 1--50, cu banda de 95\\% pentru date i.i.d.'),
        T('Report $\\hat\\rho_1, \\dots, \\hat\\rho_5$ of $r_t$ and of $r_t^2$.', 'Raportați $\\hat\\rho_1, \\dots, \\hat\\rho_5$ pentru $r_t$ și pentru $r_t^2$.'),
        T('Compute the Ljung--Box $Q(10)$ of $r_t$, $r_t^2$ and $|r_t|$, and the ARCH-LM(5) statistic with its $R^2$.', 'Calculați statistica Ljung--Box $Q(10)$ pentru $r_t$, $r_t^2$ și $|r_t|$, precum și statistica ARCH-LM(5) cu $R^2$-ul ei.'),
        T('Interpretation: does volatility clustering contradict weak-form market efficiency?', 'Interpretare: contrazice volatility clustering eficiența pieței în formă slabă?')],
       T('the chart, ten autocorrelations, four statistics and two sentences', 'graficul, zece autocorelații, patru statistici și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch8_sem_b5', h='0.34') + items(
    T('$\\hat\\rho_{1..5}(r)$: $@{b5.rho_r1}$, $@{b5.rho_r2}$, $@{b5.rho_r3}$, $@{b5.rho_r4}$, $@{b5.rho_r5}$; $\\hat\\rho_{1..5}(r^2)$: $@{b5.rho_r21}$, $@{b5.rho_r22}$, $@{b5.rho_r23}$, $@{b5.rho_r24}$, $@{b5.rho_r25}$; $T = @{b5.T}$',
      '$\\hat\\rho_{1..5}(r)$: $@{b5.rho_r1}$; $@{b5.rho_r2}$; $@{b5.rho_r3}$; $@{b5.rho_r4}$; $@{b5.rho_r5}$; $\\hat\\rho_{1..5}(r^2)$: $@{b5.rho_r21}$; $@{b5.rho_r22}$; $@{b5.rho_r23}$; $@{b5.rho_r24}$; $@{b5.rho_r25}$; $T = @{b5.T}$'),
    T('$Q(10)$: $r$ @{b5.lbr} (p @{b5.plbr}), $r^2$ @{b5.lbr2}, $|r|$ @{b5.lbabs} (critical value @{b5.crit}); ARCH-LM(5) $= @{b5.n} \\times @{b5.r2} = @{b5.lm}$ (critical value @{b5.critlm})',
      '$Q(10)$: $r$ @{b5.lbr} (p @{b5.plbr}), $r^2$ @{b5.lbr2}, $|r|$ @{b5.lbabs} (valoarea critică @{b5.crit}); ARCH-LM(5) $= @{b5.n} \\times @{b5.r2} = @{b5.lm}$ (valoarea critică @{b5.critlm})'),
    T('Interpretation: no; the size is predictable, the sign hardly; efficiency is about predictable returns, not about predictable risk (Chapter 7)',
      'Interpretare: nu; mărimea este previzibilă, semnul aproape deloc; eficiența privește randamentele previzibile, nu riscul previzibil (Capitolul 7)')) + qlsem(), 'scriptsize')

D.task(T('B6: Four More Series [Proposed]', 'B6: încă patru serii [Propus]'),
       T('does volatility clustering in the BET, Bitcoin, Banca Transilvania and OMV Petrom depend on the order of the days? Model: B5.',
         'depinde volatility clustering de la BET, Bitcoin, Banca Transilvania și OMV Petrom de ordinea zilelor? Model: B5.'),
       T('BET since 1997, Bitcoin since 2014, TLV and SNP since 2010', 'BET din 1997, Bitcoin din 2014, TLV și SNP din 2010'),
       [T('Compute $\\hat\\rho_1(r^2)$, $Q(10)$ of $r_t^2$, ARCH-LM(5) and the excess kurtosis of each series.', 'Calculați $\\hat\\rho_1(r^2)$, $Q(10)$ pentru $r_t^2$, ARCH-LM(5) și excesul de boltire pentru fiecare serie.'),
        T('Shuffle the returns at random (seed 2026) and compute $Q(10)$ of $r_t^2$ and ARCH-LM(5) again.', 'Amestecați aleator randamentele (seed 2026) și calculați din nou $Q(10)$ pentru $r_t^2$ și ARCH-LM(5).'),
        T('Interpretation: why does shuffling remove the clustering but not the heavy tails?', 'Interpretare: de ce amestecarea elimină volatility clustering, dar nu și cozile groase?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{SHORT[k]} & @{{b6.{k}.T}} & $@{{b6.{k}.rho}}$ & @{{b6.{k}.lb}} & @{{b6.{k}.lm}} & @{{b6.{k}.k}} & @{{b6.{k}.lbs}} (@{{b6.{k}.plbs}}) & @{{b6.{k}.lms}}')


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& $T$ & $\\hat\\rho_1(r^2)$ & $Q(10)$ & ARCH-LM & \\textbf{Exc.\\ kurt.} & $Q(10)$, \\textbf{shuffled} (p) & \\textbf{LM, shuffled}',
                  '& $T$ & $\\hat\\rho_1(r^2)$ & $Q(10)$ & ARCH-LM & \\textbf{Exc.\\ boltire} & $Q(10)$, \\textbf{amestecat} (p) & \\textbf{LM, amestecat}'),
    [b6row(k) for k in ['bet', 'btc', 'tlv', 'snp']], size='footnotesize') + items(
    T('All tests reject strongly for the four series in the actual order; after shuffling no test rejects', 'Toate testele resping puternic pentru cele patru serii în ordinea reală; după amestecare niciun test nu respinge'),
    T('Interpretation: clustering is a property of the order of the days, while kurtosis depends only on the values, which a shuffle keeps',
      'Interpretare: volatility clustering este o proprietate a ordinii zilelor, iar boltirea depinde doar de valori, pe care amestecarea le păstrează')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B7: Which Forecast Is Best? [Solved]', 'B7: care prognoză este cea mai bună? [Rezolvat]'),
       T('which simple forecast of tomorrow\'s S\\&P 500 variance has the smallest loss since 2010?', 'ce prognoză simplă a varianței de mîine a S\\&P 500 are cea mai mică pierdere din 2010?'),
       T('S\\&P 500 returns since 1990, VIX closes; forecasts for @{b7.n} days from 2010', 'randamentele S\\&P 500 din 1990, închiderile VIX; prognoze pentru @{b7.n} de zile din 2010'),
       [T('Build next-day forecasts: rolling means of $r_t^2$ on 21, 63 and 252 days, EWMA with $\\lambda = 0.94$ and $0.97$, and $\\mathrm{VIX}^2/a$.',
          'Construiți prognoze pentru ziua următoare: mediile mobile ale lui $r_t^2$ pe 21, 63 și 252 de zile, EWMA cu $\\lambda = 0{,}94$ și $0{,}97$ și $\\mathrm{VIX}^2/a$.'),
        T('Compute MSE and QLIKE with the proxy $r_t^2$.', 'Calculați MSE și QLIKE cu variabila proxy $r_t^2$.'),
        T('Run the Diebold--Mariano test for EWMA 0.94 against the 21-day window and for the VIX against EWMA 0.94.', 'Aplicați testul Diebold--Mariano pentru EWMA 0,94 față de fereastra de 21 de zile și pentru VIX față de EWMA 0,94.'),
        T('Draw the QLIKE of EWMA for $\\lambda$ from 0.80 to 0.995.', 'Desenați QLIKE pentru EWMA, cu $\\lambda$ între 0,80 și 0,995.'),
        T('Interpretation: is the VIX a better forecast of tomorrow\'s variance than EWMA?', 'Interpretare: este VIX o prognoză mai bună a varianței de mîine decît EWMA?')],
       T('a table of twelve numbers, two tests, the chart and two sentences', 'un tabel cu douăsprezece valori, două teste, graficul și două fraze'), size='footnotesize', nb='B7')

D.frame(T('B7: Solution [Solved]', 'B7: rezolvare [Rezolvat]'), fig('ch8_sem_b7', h='0.30') + table(
    'lrrrrrr', T('& 21 & 63 & 252 & EWMA 0.94 & EWMA 0.97 & VIX', '& 21 & 63 & 252 & EWMA 0,94 & EWMA 0,97 & VIX'),
    ['MSE & ' + ' & '.join(f'@{{b7.{m}.mse}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']),
     'QLIKE & ' + ' & '.join(f'@{{b7.{m}.ql}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix'])], size='scriptsize') + items(
    T('DM: EWMA 0.94 against 21 days $t = @{b7.dm_ewma_hist21.t}$ (p @{b7.dm_ewma_hist21.p}); VIX against EWMA 0.94 $t = @{b7.dm_vix_ewma.t}$ (p = @{b7.dm_vix_ewma.p}); best $\\lambda = @{b7.best}$',
      'DM: EWMA 0,94 față de 21 de zile $t = @{b7.dm_ewma_hist21.t}$ (p @{b7.dm_ewma_hist21.p}); VIX față de EWMA 0,94 $t = @{b7.dm_vix_ewma.t}$ (p = @{b7.dm_vix_ewma.p}); cel mai bun $\\lambda = @{b7.best}$'),
    T('Interpretation: no; the VIX has slightly lower losses, but the difference is not significant; both beat the rolling windows',
      'Interpretare: nu; VIX are pierderi puțin mai mici, dar diferența nu este semnificativă; ambele sînt mai bune decît ferestrele mobile')) + qlsem(), 'scriptsize')

D.task(T('B8: The Best $\\lambda$ for the BET and Bitcoin [Proposed]', 'B8: cel mai bun $\\lambda$ pentru BET și Bitcoin [Propus]'),
       T('is the RiskMetrics $\\lambda = 0.94$ a good choice for the BET and for Bitcoin? Model: B7.', 'este $\\lambda = 0{,}94$ din RiskMetrics o alegere bună pentru BET și pentru Bitcoin? Model: B7.'),
       T('BET since 1997 and Bitcoin since 2014; forecasts from 2010 (BET) and from 2016 (Bitcoin)', 'BET din 1997 și Bitcoin din 2014; prognoze din 2010 (BET) și din 2016 (Bitcoin)'),
       [T('Compute the QLIKE of next-day EWMA forecasts for $\\lambda$ from 0.80 to 0.995 and find the best $\\lambda$.', 'Calculați QLIKE pentru prognozele EWMA ale zilei următoare, cu $\\lambda$ între 0,80 și 0,995, și găsiți cel mai bun $\\lambda$.'),
        T('Report the QLIKE at the best $\\lambda$, at 0.94 and at 0.97, and the half-life of the best $\\lambda$.', 'Raportați QLIKE pentru cel mai bun $\\lambda$, pentru 0,94 și pentru 0,97, precum și timpul de înjumătățire al celui mai bun $\\lambda$.'),
        T('Run the Diebold--Mariano test of the best $\\lambda$ against 0.94.', 'Aplicați testul Diebold--Mariano pentru cel mai bun $\\lambda$ față de 0,94.'),
        T('Interpretation: would you use $\\lambda = 0.94$ for both series?', 'Interpretare: ați folosi $\\lambda = 0{,}94$ pentru ambele serii?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B8')

D.frame(T('B8: Solution [Proposed]', 'B8: rezolvare [Propus]'), table(
    'lrrrrrr', T('& \\textbf{Best} $\\lambda$ & \\textbf{Half-life} & QLIKE, \\textbf{best} & QLIKE, 0.94 & QLIKE, 0.97 & DM $t$ (p)',
                 '& \\textbf{Cel mai bun} $\\lambda$ & \\textbf{Timp de înjumătățire} & QLIKE, \\textbf{optim} & QLIKE, 0,94 & QLIKE, 0,97 & DM $t$ (p)'),
    [f'{NAMES[k]} & @{{b8.{k}.best}} & @{{b8.{k}.hl}} & @{{b8.{k}.qb}} & @{{b8.{k}.q94}} & @{{b8.{k}.q97}} & ${{@{{b8.{k}.t}}}}$ (@{{b8.{k}.p}})' for k in ['bet', 'btc']],
    size='footnotesize') + items(
    T('BET: a faster EWMA is best; its gain over 0.94 is borderline (p = @{b8.bet.p}); Bitcoin: a slower EWMA is best, but not significantly better than 0.94',
      'BET: cel mai bun este un EWMA mai rapid; cîștigul față de 0,94 este la limită (p = @{b8.bet.p}); Bitcoin: cel mai bun este un EWMA mai lent, dar nu semnificativ mai bun decît 0,94'),
    T('Interpretation: yes, as a simple common rule; the losses near the optimum are flat, and one $\\lambda$ for all assets is easier to defend',
      'Interpretare: da, ca regulă simplă comună; pierderile din jurul optimului variază puțin, iar un singur $\\lambda$ pentru toate activele este mai ușor de justificat')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Which Estimator for BVB Stocks? [Proposed]', 'C1: ce estimator pentru acțiunile BVB? [Propus]'),
       T('do range-based estimators improve volatility forecasts for thinly traded stocks of the Bucharest Stock Exchange?',
         'îmbunătățesc estimatorii de amplitudine prognozele de volatilitate pentru acțiunile puțin lichide de la Bursa de Valori București?'),
       T('adjusted OHLC of TLV, SNP, BRD and TGN since 2010; S\\&P 500 since 2008 as a benchmark; models: B3, B7', 'OHLC ajustat pentru TLV, SNP, BRD și TGN din 2010; S\\&P 500 din 2008 ca termen de comparație; modele: B3, B7'),
       [T('Compute the share of trading days with an unchanged close, and the ratios Parkinson / CC and YZ / CC of the whole-sample volatility.',
          'Calculați ponderea zilelor de tranzacționare cu închiderea neschimbată, precum și rapoartele Parkinson / CC și YZ / CC ale volatilității pe întregul eșantion.'),
        T('Compute the QLIKE (proxy $r_t^2$) of next-day forecasts by the 21-day CC, Parkinson and YZ variances, from 2010.', 'Calculați QLIKE (proxy $r_t^2$) pentru prognozele zilei următoare date de varianțele CC, Parkinson și YZ pe 21 de zile, din 2010.'),
        T('Compare the four stocks with the S\\&P 500.', 'Comparați cele patru acțiuni cu S\\&P 500.'),
        T('List two reasons why the best estimator may differ across stocks.', 'Enumerați două motive pentru care cel mai bun estimator poate diferi de la o acțiune la alta.'),
        T('Interpretation: what data would you need to explain the differences?', 'Interpretare: de ce date ați avea nevoie pentru a explica diferențele?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')


def c1row(k):
    return (f'{SHORT[k]} & @{{c1.{k}.flat}}\\% & @{{c1.{k}.pc}} & @{{c1.{k}.yc}} & @{{c1.{k}.qcc}} & @{{c1.{k}.qpark}} & @{{c1.{k}.qyz}}')


D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrrr', T('& \\textbf{Unchanged close} & P/CC & YZ/CC & QLIKE CC & QLIKE P & QLIKE YZ', '& \\textbf{Închidere neschimbată} & P/CC & YZ/CC & QLIKE CC & QLIKE P & QLIKE YZ'),
    [c1row(k) for k in ['tlv', 'snp', 'brd', 'tgn', 'sp500']], size='footnotesize') + items(
    T('YZ gives the lowest QLIKE for SNP, BRD and TGN; CC is best for TLV and for the S\\&P 500; Parkinson alone is never clearly best',
      'YZ dă cel mai mic QLIKE pentru SNP, BRD și TGN; CC este cel mai bun pentru TLV și pentru S\\&P 500; Parkinson singur nu este niciodată clar cel mai bun'),
    T('Reasons: the overnight share, discrete trading that shrinks the range, the opening auction, stale opens; differences need DM tests before any conclusion',
      'Motive: ponderea nopții, tranzacționarea discretă care micșorează amplitudinea, licitația de deschidere, deschideri cu prețuri vechi; diferențele cer teste DM înainte de orice concluzie'),
    T('Data for a project: intraday trades or volumes, the number of trades per day, the free float of each stock',
      'Date pentru un proiect: tranzacțiile sau volumele intrazilnice, numărul de tranzacții pe zi, free float-ul fiecărei acțiuni')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the volatility of the S\\&P 500 and of Bitcoin. The answer:',
      'Un student a cerut unui asistent AI să rezume volatilitatea S\\&P 500 și a Bitcoin. Răspunsul:'),
    T('\\aiprompt{(a) Since 2008 the Parkinson volatility of the S\\&P 500 is @{c2.park}\\% and the close-to-close volatility @{c2.cc}\\%; Parkinson is about five times more efficient, so the true volatility is @{c2.park}\\%.}',
      '\\aiprompt{(a) Din 2008 volatilitatea Parkinson a S\\&P 500 este @{c2.park}\\%, iar cea închidere--închidere @{c2.cc}\\%; Parkinson este de circa cinci ori mai eficient, deci volatilitatea reală este @{c2.park}\\%.}'),
    T('\\aiprompt{(b) The Ljung-Box Q(10) of S\\&P 500 returns is @{c2.lbr} (p < 0.001), so the S\\&P 500 shows volatility clustering.}',
      '\\aiprompt{(b) Statistica Ljung-Box Q(10) a randamentelor S\\&P 500 este @{c2.lbr} (p < 0,001), deci S\\&P 500 prezintă volatility clustering.}'),
    T('\\aiprompt{(c) EWMA with lambda = 0.94 has a half-life of about 94 days.}', '\\aiprompt{(c) EWMA cu lambda = 0,94 are un timp de înjumătățire de circa 94 de zile.}'),
    T('\\aiprompt{(d) The daily standard deviation of Bitcoin is @{c2.btc_sd}\\%, so its annual volatility is @{c2.btc_sd} x sqrt(252) = @{c2.btc_252}\\%.}',
      '\\aiprompt{(d) Abaterea standard zilnică a Bitcoin este @{c2.btc_sd}\\%, deci volatilitatea anuală este @{c2.btc_sd} x sqrt(252) = @{c2.btc_252}\\%.}'),
    T('\\aiprompt{(e) Since 1990 the VIX was above the realised volatility of the next 21 days on about @{c2.vix}\\% of days.}',
      '\\aiprompt{(e) Din 1990, VIX a fost peste volatilitatea realizată din următoarele 21 de zile în circa @{c2.vix}\\% din zile.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: efficiency is about variance, not bias; Parkinson misses the night (@{c2.night}\\% of the variance) and the covariance of night and day ($\\mathrm{corr} = @{c2.corr}$); YZ gives @{c2.yz}\\%',
      '(a) Greșit: eficiența privește varianța estimatorului, nu deplasarea; Parkinson ratează noaptea (@{c2.night}\\% din varianță) și covarianța dintre noapte și zi ($\\mathrm{corr} = @{c2.corr}$); YZ dă @{c2.yz}\\%'),
    T('(b) Wrong: $Q(10)$ on returns tests the autocorrelation of the mean; clustering is tested on $r_t^2$: $Q(10) = @{c2.lbr2}$',
      '(b) Greșit: $Q(10)$ pe randamente testează autocorelația mediei; volatility clustering se testează pe $r_t^2$: $Q(10) = @{c2.lbr2}$'),
    T('(c) Wrong: $\\ln 0.5/\\ln 0.94 = @{c2.half_life}$ days', '(c) Greșit: $\\ln 0{,}5/\\ln 0{,}94 = @{c2.half_life}$ zile'),
    T('(d) Wrong: Bitcoin trades @{c2.ppy} days a year: $@{c2.btc_sd} \\times \\sqrt{365} = @{c2.btc_ann}\\%$', '(d) Greșit: Bitcoin se tranzacționează @{c2.ppy} de zile pe an: $@{c2.btc_sd} \\times \\sqrt{365} = @{c2.btc_ann}\\%$'),
    T('(e) Correct: the variance risk premium makes implied volatility usually higher than realised', '(e) Corect: prima de risc a varianței face ca volatilitatea implicită să fie de obicei mai mare decît cea realizată')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Annualise with the actual frequency: 252 for stock markets, 365 for Bitcoin', 'Anualizați cu frecvența reală: 252 pentru piețele de acțiuni, 365 pentru Bitcoin'),
    T('Short windows react fast, long windows smooth the estimate; EWMA reacts without ghost effects', 'Ferestrele scurte reacționează repede, cele lungi netezesc estimarea; EWMA reacționează fără ghost effect'),
    T('Range-based estimators are precise but see only the session: use Yang--Zhang when the market closes at night', 'Estimatorii de amplitudine sînt preciși, dar văd doar ședința: folosiți Yang--Zhang cînd piața se închide noaptea'),
    T('Clustering is tested on $r_t^2$ (Ljung--Box, ARCH-LM), not on $r_t$', 'Volatility clustering se testează pe $r_t^2$ (Ljung--Box, ARCH-LM), nu pe $r_t$'),
    T('Compare forecasts with QLIKE and test the difference; an AI answer is a draft to check', 'Comparați prognozele cu QLIKE și testați diferența; un răspuns AI este o ciornă de verificat')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 8 develops each topic of today: historical and EWMA volatility, range-based and realised volatility, clustering, the VIX, forecast evaluation',
      'Cursul 8 dezvoltă fiecare temă de azi: volatilitatea istorică și EWMA, volatilitatea de amplitudine și cea realizată, volatility clustering, VIX, evaluarea prognozelor'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: all BET stocks, liquidity groups, Diebold--Mariano tests', 'C1 poate deveni un proiect de echipă: toate acțiunile din BET, grupe de lichiditate, teste Diebold--Mariano'),
    T('Reading: \\refFHH, Ch.~13; exercises in \\refBHL, Ch.~13; \\refTsay, Ch.~3.1', 'Lectură: \\refFHH, cap.~13; exerciții în \\refBHL, cap.~13; \\refTsay, cap.~3.1')))

D.references(bib(['BHL', 'Cboe', 'DM', 'Engle', 'FHH', 'GK', 'Ljung', 'ML', 'Parkinson', 'Patton', 'RM', 'RS', 'Tsay', 'YZ']), per=16)

if __name__ == '__main__':
    fix_refs(D)
    D.write(V)
