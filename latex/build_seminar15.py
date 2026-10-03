r"""
build_seminar15.py -- Seminarul 15 (Risc sistemic), EN + RO dintr-o singură sursă
================================================================================
Seminarul are loc ÎNAINTEA cursului 15: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_15/sem15_results.json (seminar15.py).
Ieșire:
  EN/Seminars/seminar15_systemic_risk.tex          (+ _solutions.tex)
  RO/Seminarii/seminar15_risc_sistemic_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_15/seminar15.py && python3 latex/build_seminar15.py && python3 latex/sfm_build.py compile 15
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch15_common import EU, NAMES, REFS, RO, US, T, bib, load_sem, put_date   # noqa: E402

S = load_sem()
V = Values()
D = Deck(15, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_15}{SFM_ch15_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for key in ('a1', 'a2'):
    a = A[key]
    for c in ('a', 'b', 'q_a', 'q_m', 'var_sys', 'var_i'):
        V.put(f'{key}.{c}', a[c], 2)
    V.put(f'{key}.ma', -a['a'], 2)
    V.put(f'{key}.bq', -a['b'] * a['q_a'], 4)
    V.put(f'{key}.bm', a['b'] * a['q_m'], 4)
    V.put(f'{key}.diff', a['q_m'] - a['q_a'], 2)
    V.put(f'{key}.covar', a['covar'], 2)
    V.put(f'{key}.covar4', a['covar'], 4)
    V.put(f'{key}.med', a['covar_med'], 2)
    V.put(f'{key}.med4', a['covar_med'], 4)
    V.put(f'{key}.d', a['dcovar'], 2)
    V.put(f'{key}.d4', a['dcovar'], 4)
    V.put(f'{key}.ratio', a['covar'] / a['var_sys'], 2)
    V.int(f'{key}.n', a['n'])
TB = A['table']
days = list(TB)
for key in ('a3', 'a4'):
    a = A[key]
    V.put(f'{key}.mes', a['mes'], 3)
    V.put(f'{key}.es', a['es_m'], 2)
    V.raw(f'{key}.n2', str(a['n2']))
    V.put(f'{key}.sum2', a['sum2'], 2)
    V.put(f'{key}.mes2', a['mes2'], 3)
    V.put(f'{key}.x18', 18 * a['mes2'] / 100, 4)
    V.put(f'{key}.exp', math.exp(-18 * a['mes2'] / 100), 4)
    V.put(f'{key}.lr', 100 * a['lrmes'], 2)
    V.put(f'{key}.b1', a['bank'][0], 2)
    V.put(f'{key}.b2', a['bank'][1], 2)
V.put('a3.m1', A['a3']['mkt'][0], 2)
V.put('a3.m2', A['a3']['mkt'][1], 2)
V.put('a3.ratio', A['a3']['mes'] / A['a3']['es_m'], 2)
V.put('a4.ratio', A['a4']['mes'] / A['a4']['es_m'], 2)
a5 = A['a5']
for c in ('srisk', 'kD', 'cap', 'srisk_fall'):
    V.put(f'a5.{c}', a5[c], 2)
V.put('a5.L', a5['L'], 0)
V.put('a5.Lstar', a5['Lstar'], 3)
V.put('a5.Lf', a5['L_fall'], 2)
V.put('a5.capf', 0.92 * 0.45 * 70, 2)
a6 = A['a6']
for kk in ('k8', 'k55'):
    for b in ('X', 'Y'):
        V.put(f'a6.{kk}.{b}', a6[kk]['srisk'][b], 2 if kk == 'k8' else 3)
    V.put(f'a6.{kk}.tot', a6[kk]['total'], 2 if kk == 'k8' else 3)
V.put('a6.k8.kdx', 0.08 * 570, 1)
V.put('a6.k8.cx', 0.92 * 0.5 * 30, 1)
V.put('a6.k8.kdy', 0.08 * 180, 1)
V.put('a6.k8.cy', 0.92 * 0.35 * 60, 2)
V.put('a6.k55.kdx', 0.055 * 570, 2)
V.put('a6.k55.cx', 0.945 * 0.5 * 30, 3)
V.put('a6.k55.kdy', 0.055 * 180, 1)
V.put('a6.k55.cy', 0.945 * 0.35 * 60, 3)
V.put('a6.Lx', (570 + 30) / 30, 0)
V.put('a6.Ly', (180 + 60) / 60, 0)


def put_region(prefix, d):
    for k, v in d['banks'].items():
        V.put(f'{prefix}.{k}.m', v['mes'], 2)
        V.put(f'{prefix}.{k}.mlo', v['mes_lo'], 2)
        V.put(f'{prefix}.{k}.mhi', v['mes_hi'], 2)
        V.put(f'{prefix}.{k}.d', v['dcovar'], 2)
        V.put(f'{prefix}.{k}.dlo', v['ci'][0], 2)
        V.put(f'{prefix}.{k}.dhi', v['ci'][1], 2)
        V.put(f'{prefix}.{k}.var', v['var_i'], 2)
    if d.get('spearman') == d.get('spearman'):
        V.put(f'{prefix}.sp', d['spearman'], 2)
    V.int(f'{prefix}.n', d['n'])
    V.raw(f'{prefix}.topm', NAMES[d['top_mes']])
    V.raw(f'{prefix}.topd', NAMES[d['top_dcovar']])


put_region('b1', S['B1'])
put_region('b2eu', S['B2']['EU'])
put_region('b2ro', S['B2']['RO'])
for key in ('B3', 'B4'):
    c = S[key]
    k = key.lower()
    for x in ('r0', 'r1', 'radj'):
        V.put(f'{k}.{x}', c[x], 2)
    V.put(f'{k}.v0', c['v0'], 2)
    V.put(f'{k}.v1', c['v1'], 2)
    V.put(f'{k}.delta', c['delta'], 2)
    V.put(f'{k}.vr', c['v1'] / c['v0'], 0)
    V.raw(f'{k}.n0', str(c['n0']))
    V.raw(f'{k}.n1', str(c['n1']))
    V.put(f'{k}.z', c['z'], 2)
    V.raw(f'{k}.p', '< ⁅0.001⁆' if c['p'] < 0.001 else f"= ⁅{c['p']:.3f}⁆")
    V.put(f'{k}.za', c['za'], 2)
    V.raw(f'{k}.pa', '< ⁅0.001⁆' if c['pa'] < 0.001 else f"= ⁅{c['pa']:.3f}⁆")
    V.put(f'{k}.r2', 1 - c['r1'] ** 2, 3)
    V.put(f'{k}.inside', 1 + c['delta'] * (1 - c['r1'] ** 2), 2)
    V.put(f'{k}.at0', math.atanh(c['r0']), 3)
    V.put(f'{k}.at1', math.atanh(c['r1']), 3)
    V.put(f'{k}.se', math.sqrt(1 / (c['n1'] - 3) + 1 / (c['n0'] - 3)), 3)
for key in ('B5', 'B6'):
    c = S[key]
    k = key.lower()
    V.put(f'{k}.sp', c['spearman'], 2)
    V.put(f'{k}.p', c['p'], 3)
    V.raw(f'{k}.n', str(c['n']))
    V.raw(f'{k}.worst', NAMES[c['worst']])
    V.raw(f'{k}.topm', NAMES[c['top_mes']])
    for b, v in c['banks'].items():
        V.put(f'{k}.{b}.m', v['mes'], 2)
        V.put(f'{k}.{b}.r', v['ret'], 1)
C1 = S['C1']
V.put('c1.total', C1['total'], 1)
V.put('c1.tlv', C1['from_eu']['TLV'], 1)
V.put('c1.brd', C1['from_eu']['BRD'], 1)
V.put('c1.mean', C1['roll_mean'], 1)
V.put('c1.max', C1['roll_max'], 1)
V.put('c1.min', C1['roll_min'], 1)
V.put('c1.y19', C1['roll_2019'], 1)
V.put('c1.y20', C1['roll_2020'], 1)
V.raw('c1.p', str(C1['p']))
V.int('c1.n', C1['n'])
put_date(V, 'c1.dmax', C1['roll_date_max'])
C2 = S['C2']
V.put('c2.covar', C2['covar'], 2)
V.put('c2.d', C2['dcovar'], 2)
V.put('c2.var', C2['var_i'], 2)
V.put('c2.wrong', C2['wrong_d'], 2)
V.put('z95', stats.norm.ppf(0.95), 3)


def b5row(c, k):
    return f'{k} & $@{{{c}.{k}.m}}$ & $@{{{c}.{k}.r}}$'


# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: which bank matters most for the stability of the system, and can we tell it from market prices?',
       '\\textbf{Întrebarea}: care bancă contează cel mai mult pentru stabilitatea sistemului și putem afla acest lucru din prețurile de piață?'),
     [T('this seminar comes \\textbf{before} Lecture 15: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 15: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: CoVaR from a quantile regression, MES from a table of 20 days, SRISK of one and of two banks, on paper',
        'Partea A: CoVaR dintr-o regresie cuantilică, MES dintr-un tabel de 20 de zile, SRISK pentru una și pentru două bănci, pe hîrtie'),
      T('Part B: MES and $\\Delta$CoVaR of real banks, correlation in crises, MES before a crisis against the losses in it, each with an interpretation question',
        'Partea B: MES și $\\Delta$CoVaR pentru bănci reale, corelația în crize, MES înaintea unei crize comparat cu pierderile din criză, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TBL = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TBL + 'p{1.1cm}' + TBL + 'p{7.5cm}' + TBL + 'p{1.9cm}' + TBL + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('CoVaR 1\\% and $\\Delta$CoVaR from a quantile regression: JPMorgan Chase; Bank of America', 'CoVaR 1\\% și $\\Delta$CoVaR dintr-o regresie cuantilică: JPMorgan Chase; Bank of America') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('MES and LRMES from 20 days of March 2020: JPMorgan Chase; Goldman Sachs', 'MES și LRMES din 20 de zile din martie 2020: JPMorgan Chase; Goldman Sachs') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('SRISK of one bank; SRISK of two banks and the aggregate SRISK', 'SRISK pentru o bancă; SRISK pentru două bănci și SRISK agregat') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('MES 5\\% and $\\Delta$CoVaR 1\\% with bootstrap intervals: US banks; European and Romanian banks', 'MES 5\\% și $\\Delta$CoVaR 1\\% cu intervale bootstrap: băncile americane; băncile europene și cele românești') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('contagion or interdependence: US and European banks in 2008; European and Romanian banks in 2020', 'contagiune sau interdependență: băncile americane și europene în 2008; băncile europene și românești în 2020') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('does MES before a crisis rank the losses in it? March 2020; March 2023', 'ordonează MES dinaintea unei crize pierderile din criză? martie 2020; martie 2023') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('how connected are the Romanian and the euro-area banks? what is wrong in an AI answer?', 'cît de conectate sînt băncile românești cu cele din zona euro? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A1'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('6 US banks, 5 European banks', '6 bănci americane, 5 bănci europene') + ' & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2005--2026',
     'Banca Transilvania (TLV), BRD & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2010--2026',
     'S\\&P 500, Euro Stoxx 50, BET & EODHD & ' + T('close', 'închidere') + ' & 2005--2026'],
    size='footnotesize') + items(
    T('US banks: JPMorgan Chase (JPM), Bank of America (BAC), Citigroup (C), Goldman Sachs (GS), Morgan Stanley (MS), Wells Fargo (WFC); European banks: Deutsche Bank (DBK), BNP Paribas (BNP), Santander (SAN), ING (INGA), HSBC (HSBA)',
      'Băncile americane: JPMorgan Chase (JPM), Bank of America (BAC), Citigroup (C), Goldman Sachs (GS), Morgan Stanley (MS), Wells Fargo (WFC); băncile europene: Deutsche Bank (DBK), BNP Paribas (BNP), Santander (SAN), ING (INGA), HSBC (HSBA)'),
    T('Prices joined on common days first, then daily log returns in \\%; BVB days without trades removed; TLV without 30--31 May 2016; the last day is 18 September 2026',
      'Prețurile se aliniază întîi pe zilele comune, apoi se calculează randamentele logaritmice zilnice în \\%; zilele BVB fără tranzacții se elimină; TLV fără 30--31 mai 2016; ultima zi este 18 septembrie 2026'),
    T('Market of each bank: S\\&P 500 (US), Euro Stoxx 50 (Europe), BET (Romania); a bank portfolio: equal weights, rebalanced daily',
      'Piața fiecărei bănci: S\\&P 500 (SUA), Euro Stoxx 50 (Europa), BET (România); un portofoliu de bănci: ponderi egale, reechilibrate zilnic')), 'footnotesize')

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Systemic Risk', 'Noțiuni necesare azi (1/5): riscul sistemic'), items(
    (T('\\textbf{Systemic risk}: the risk that the financial system as a whole stops working, with large costs for the real economy \\refFSB',
       '\\textbf{Riscul sistemic}: riscul ca sistemul financiar în ansamblu să nu mai funcționeze, cu costuri mari pentru economia reală \\refFSB'),
     [T('channels: direct contagion (exposures between banks), common exposures, fire sales, runs on deposits and funding', 'canale: contagiunea directă (expuneri între bănci), expunerile comune, vînzările forțate, retragerea depozitelor și a finanțării')]),
    (T('The VaR of a bank (Chapter 10) looks at the bank alone; systemic measures look at the \\textbf{joint tail} of a bank and the system', 'VaR-ul unei bănci (Capitolul 10) privește doar banca; măsurile sistemice privesc \\textbf{coada comună} a băncii și a sistemului'),
     [T('\\textbf{CoVaR}: the system given that the bank is in distress \\refAB', '\\textbf{CoVaR}: sistemul, dat fiind că banca este în dificultate \\refAB'),
      T('\\textbf{MES} and \\textbf{SRISK}: the bank given that the market crashes \\refAPPR; \\refBE', '\\textbf{MES} și \\textbf{SRISK}: banca, dat fiind că piața se prăbușește \\refAPPR; \\refBE')]),
    T('Level $\\alpha$ = the probability of the tail: VaR 1\\%, CoVaR 1\\%, MES 5\\%; losses are reported as positive numbers', 'Nivelul $\\alpha$ = probabilitatea cozii: VaR 1\\%, CoVaR 1\\%, MES 5\\%; pierderile se raportează ca numere pozitive')))

D.frame(T('What You Need for Today (2/5): CoVaR by Quantile Regression', 'Noțiuni necesare azi (2/5): CoVaR prin regresie cuantilică'), items(
    (T('$q_\\alpha(X)$: the $\\alpha$-quantile; $\\mathrm{VaR}_\\alpha = -q_\\alpha$ (Chapter 10); linear \\textbf{quantile regression} \\refKB: $q_\\alpha(Y \\mid X = x) = a + bx$',
       '$q_\\alpha(X)$: cuantila de ordin $\\alpha$; $\\mathrm{VaR}_\\alpha = -q_\\alpha$ (Capitolul 10); \\textbf{regresia cuantilică} liniară \\refKB: $q_\\alpha(Y \\mid X = x) = a + bx$'),
     [T('$a$: the intercept (constant term), $b$: the slope; estimated by minimising $\\sum_t (y_t - a - bx_t)(\\alpha - \\mathbf{1}\\{y_t < a + bx_t\\})$',
        '$a$: termenul liber, $b$: panta; estimate prin minimizarea $\\sum_t (y_t - a - bx_t)(\\alpha - \\mathbf{1}\\{y_t < a + bx_t\\})$')]),
    (T('$X^i$: return of bank $i$; $X^{sys}$: return of the system; regression of $X^{sys}$ on $X^i$ at level $\\alpha$: $\\hat a$, $\\hat b$ \\refAB',
       '$X^i$: randamentul băncii $i$; $X^{sys}$: randamentul sistemului; regresia lui $X^{sys}$ pe $X^i$ la nivelul $\\alpha$: $\\hat a$, $\\hat b$ \\refAB'),
     [T('\\textbf{CoVaR}$_\\alpha = -(\\hat a + \\hat b\\,q_\\alpha(X^i))$: the VaR of the system on a day when bank $i$ is at its own VaR', '\\textbf{CoVaR}$_\\alpha = -(\\hat a + \\hat b\\,q_\\alpha(X^i))$: VaR-ul sistemului într-o zi în care banca $i$ este la propriul VaR'),
      T('CoVaR at the median: $-(\\hat a + \\hat b\\,q_{50\\%}(X^i))$; \\textbf{$\\Delta$CoVaR}$_\\alpha = \\hat b\\,(q_{50\\%}(X^i) - q_\\alpha(X^i))$', 'CoVaR la mediană: $-(\\hat a + \\hat b\\,q_{50\\%}(X^i))$; \\textbf{$\\Delta$CoVaR}$_\\alpha = \\hat b\\,(q_{50\\%}(X^i) - q_\\alpha(X^i))$')]),
    T('$\\Delta$CoVaR: how much the VaR of the system rises when the bank moves from a normal day to distress: its contribution to systemic risk',
      '$\\Delta$CoVaR: cu cît crește VaR-ul sistemului cînd banca trece de la o zi obișnuită la una de criză: contribuția ei la riscul sistemic')))

D.frame(T('What You Need for Today (3/5): MES, LRMES and SRISK', 'Noțiuni necesare azi (3/5): MES, LRMES și SRISK'), items(
    (T('\\textbf{MES} \\refAPPR: $\\mathrm{MES}_\\alpha = -E[X^i \\mid X^m \\le q_\\alpha(X^m)]$, the average loss of the bank on the market\'s worst $\\alpha$ of days',
       '\\textbf{MES} \\refAPPR: $\\mathrm{MES}_\\alpha = -E[X^i \\mid X^m \\le q_\\alpha(X^m)]$, pierderea medie a băncii în cele mai proaste $\\alpha$ dintre zilele pieței'),
     [T('estimate: the $k = \\lceil n\\alpha \\rceil$ days with the smallest market returns; MES $= -$ the average bank return on those days', 'estimare: cele $k = \\lceil n\\alpha \\rceil$ zile cu cele mai mici randamente ale pieței; MES $= -$ media randamentelor băncii în acele zile')]),
    (T('\\textbf{LRMES} \\refAER: $\\mathrm{LRMES} = 1 - \\exp(-18\\,\\mathrm{MES}_{2\\%})$, the loss of the bank if the market falls 40\\% in six months',
       '\\textbf{LRMES} \\refAER: $\\mathrm{LRMES} = 1 - \\exp(-18\\,\\mathrm{MES}_{2\\%})$, pierderea băncii dacă piața scade cu 40\\% în șase luni'),
     [T('$\\mathrm{MES}_{2\\%}$: the average loss of the bank (as a fraction) on the days when the market falls by more than 2\\%', '$\\mathrm{MES}_{2\\%}$: pierderea medie a băncii (ca fracție) în zilele în care piața scade cu peste 2\\%')]),
    (T('\\textbf{SRISK} \\refBE: $\\mathrm{SRISK} = kD - (1 - k)(1 - \\mathrm{LRMES})\\,W$, with $k = 8\\%$', '\\textbf{SRISK} \\refBE: $\\mathrm{SRISK} = kD - (1 - k)(1 - \\mathrm{LRMES})\\,W$, cu $k = 8\\%$'),
     [T('$D$: debt (book value), $W$: market value of equity, leverage $L = (D + W)/W$; SRISK $> 0$: capital missing in a crisis', '$D$: datoriile (valoarea contabilă), $W$: valoarea de piață a capitalului propriu, efectul de levier $L = (D + W)/W$; SRISK $> 0$: capital lipsă într-o criză'),
      T('SRISK $> 0$ exactly when $L > L^* = 1 + (1 - k)(1 - \\mathrm{LRMES})/k$; aggregate SRISK: the sum of the positive values', 'SRISK $> 0$ exact cînd $L > L^* = 1 + (1 - k)(1 - \\mathrm{LRMES})/k$; SRISK agregat: suma valorilor pozitive')])), 'footnotesize')

D.frame(T('What You Need for Today (4/5): Correlation in Crises', 'Noțiuni necesare azi (4/5): corelația în crize'), items(
    (T('If $y = \\beta x + e$ with fixed $\\beta$, the correlation rises when the variance of $x$ rises \\refFR', 'Dacă $y = \\beta x + e$ cu $\\beta$ fix, corelația crește cînd crește varianța lui $x$ \\refFR'),
     [T('\\textbf{Forbes--Rigobon correction}: $\\rho^* = \\rho/\\sqrt{1 + \\delta(1 - \\rho^2)}$, $\\delta = \\sigma^2_{\\text{crisis}}/\\sigma^2_{\\text{calm}} - 1$ of the source market', '\\textbf{corecția Forbes--Rigobon}: $\\rho^* = \\rho/\\sqrt{1 + \\delta(1 - \\rho^2)}$, $\\delta = \\sigma^2_{\\text{criză}}/\\sigma^2_{\\text{calm}} - 1$ pentru piața-sursă'),
      T('\\textbf{contagion}: the corrected correlation rises; \\textbf{interdependence}: only the raw correlation rises', '\\textbf{contagiune}: corelația corectată crește; \\textbf{interdependență}: crește doar corelația necorectată')]),
    (T('\\textbf{Fisher $z$ test} of $H_0: \\rho_{\\text{crisis}} = \\rho_{\\text{calm}}$ against a rise (independent samples of sizes $n_1$, $n_0$)', '\\textbf{Testul $z$ al lui Fisher} pentru $H_0: \\rho_{\\text{criză}} = \\rho_{\\text{calm}}$ față de o creștere (eșantioane independente de mărimi $n_1$, $n_0$)'),
     [T('$z = (\\mathrm{atanh}\\,r_1 - \\mathrm{atanh}\\,r_0)/\\sqrt{1/(n_1 - 3) + 1/(n_0 - 3)}$; reject at 5\\% if $z > 1.645$', '$z = (\\mathrm{atanh}\\,r_1 - \\mathrm{atanh}\\,r_0)/\\sqrt{1/(n_1 - 3) + 1/(n_0 - 3)}$; respingem la 5\\% dacă $z > 1{,}645$')]),
    T('\\textbf{Spearman correlation}: the correlation of the ranks; it measures whether two orderings agree', '\\textbf{Corelația Spearman}: corelația rangurilor; măsoară dacă două ordonări sînt de acord')))

D.frame(T('What You Need for Today (5/5): Bootstrap and Connectedness', 'Noțiuni necesare azi (5/5): bootstrap și conectivitate'), items(
    (T('\\textbf{Moving-block bootstrap}: resample blocks of 20 consecutive days, recompute the measure, take the 2.5\\% and 97.5\\% percentiles', '\\textbf{Bootstrap pe blocuri mobile}: reeșantionăm blocuri de 20 de zile consecutive, recalculăm măsura și luăm percentilele de 2,5\\% și 97,5\\%'),
     [T('blocks keep the volatility clustering of the returns (Chapter 8)', 'blocurile păstrează volatility clustering din randamente (Capitolul 8)')]),
    (T('\\textbf{Connectedness} \\refDYb: a VAR (vector autoregression) for $N$ return series and the share $d_{ij}$ of the $H$-day forecast error variance of $i$ due to shocks to $j$',
       '\\textbf{Conectivitatea} \\refDYb: un model VAR (vector autoregresiv) pentru $N$ serii de randamente și proporția $d_{ij}$ din varianța erorii de prognoză pe $H$ zile a lui $i$ datorată șocurilor lui $j$'),
     [T('rows sum to 100\\%; FROM others $= \\sum_{j \\ne i} d_{ij}$; total $= \\frac1N\\sum_{i \\ne j} d_{ij}$', 'rîndurile însumează 100\\%; FROM (de la celelalte) $= \\sum_{j \\ne i} d_{ij}$; totalul $= \\frac1N\\sum_{i \\ne j} d_{ij}$')]),
    T('\\textbf{Granger causality} \\refGranger: past values of $x$ improve the forecast of $y$; a statistical link, not a proof of a causal channel',
      '\\textbf{Cauzalitatea Granger} \\refGranger: valorile trecute ale lui $x$ îmbunătățesc prognoza lui $y$; o legătură statistică, nu dovada unui canal cauzal')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: CoVaR of the S\\&P 500 Given JPMorgan Chase', 'A1: CoVaR al S\\&P 500 condiționat de JPMorgan Chase'),
         items(T('Daily log returns (\\%) since 2005, $n = @{a1.n}$. Quantile regression at 1\\% of the S\\&P 500 on JPM: $\\hat a = @{a1.a}$, $\\hat b = @{a1.b}$; JPM: $q_{1\\%} = @{a1.q_a}$, $q_{50\\%} = @{a1.q_m}$; VaR 1\\% of the S\\&P 500: $@{a1.var_sys}\\%$.',
                 'Randamente logaritmice zilnice (\\%) din 2005, $n = @{a1.n}$. Regresia cuantilică la 1\\% a S\\&P 500 pe JPM: $\\hat a = @{a1.a}$, $\\hat b = @{a1.b}$; JPM: $q_{1\\%} = @{a1.q_a}$, $q_{50\\%} = @{a1.q_m}$; VaR 1\\% al S\\&P 500: $@{a1.var_sys}\\%$.'),
               T('1. Compute CoVaR 1\\% of the S\\&P 500 given that JPM is at its 1\\% quantile.', '1. Calculați CoVaR 1\\% al S\\&P 500, dat fiind că JPM se află la cuantila sa de 1\\%.'),
               T('2. Compute CoVaR 1\\% at the median of JPM.', '2. Calculați CoVaR 1\\% la mediana JPM.'),
               T('3. Compute $\\Delta$CoVaR 1\\%.', '3. Calculați $\\Delta$CoVaR 1\\%.'),
               T('4. Compare CoVaR 1\\% with the unconditional VaR 1\\% of the S\\&P 500.', '4. Comparați CoVaR 1\\% cu VaR 1\\% necondiționat al S\\&P 500.'),
               T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
         items(T('1. $-(@{a1.a} + @{a1.b} \\times (@{a1.q_a})) = @{a1.ma} + @{a1.bq} = @{a1.covar4} \\approx @{a1.covar}\\%$', '1. $-(@{a1.a} + @{a1.b} \\times (@{a1.q_a})) = @{a1.ma} + @{a1.bq} = @{a1.covar4} \\approx @{a1.covar}\\%$'),
               T('2. $-(@{a1.a} + @{a1.b} \\times @{a1.q_m}) = @{a1.ma} - @{a1.bm} = @{a1.med4} \\approx @{a1.med}\\%$', '2. $-(@{a1.a} + @{a1.b} \\times @{a1.q_m}) = @{a1.ma} - @{a1.bm} = @{a1.med4} \\approx @{a1.med}\\%$'),
               T('3. $@{a1.b} \\times (@{a1.q_m} - (@{a1.q_a})) = @{a1.b} \\times @{a1.diff} = @{a1.d4} \\approx @{a1.d}\\%$ (the difference of 1 and 2)', '3. $@{a1.b} \\times (@{a1.q_m} - (@{a1.q_a})) = @{a1.b} \\times @{a1.diff} = @{a1.d4} \\approx @{a1.d}\\%$ (diferența dintre 1 și 2)'),
               T('4. $@{a1.covar}/@{a1.var_sys} = @{a1.ratio}$', '4. $@{a1.covar}/@{a1.var_sys} = @{a1.ratio}$'),
               T('When JPM is in distress, the VaR 1\\% of the index is about @{a1.ratio} times its unconditional value.', 'Cînd JPM este în dificultate, VaR 1\\% al indicelui este de aproximativ @{a1.ratio} ori mai mare decît valoarea sa necondiționată.')),
         size='scriptsize')

D.proposed(T('A2: Bank of America', 'A2: Bank of America'),
           items(T('Same data as A1. Quantile regression at 1\\% of the S\\&P 500 on BAC: $\\hat a = @{a2.a}$, $\\hat b = @{a2.b}$; BAC: $q_{1\\%} = @{a2.q_a}$, $q_{50\\%} = @{a2.q_m}$. Model: A1.',
                   'Aceleași date ca în A1. Regresia cuantilică la 1\\% a S\\&P 500 pe BAC: $\\hat a = @{a2.a}$, $\\hat b = @{a2.b}$; BAC: $q_{1\\%} = @{a2.q_a}$, $q_{50\\%} = @{a2.q_m}$. Model: A1.'),
                 T('1. Compute VaR 1\\% of BAC, CoVaR 1\\% and CoVaR 1\\% at the median.', '1. Calculați VaR 1\\% al BAC, CoVaR 1\\% și CoVaR 1\\% la mediană.'),
                 T('2. Compute $\\Delta$CoVaR 1\\%.', '2. Calculați $\\Delta$CoVaR 1\\%.'),
                 T('3. Rank JPM and BAC by VaR 1\\% and by $\\Delta$CoVaR 1\\%.', '3. Ordonați JPM și BAC după VaR 1\\% și după $\\Delta$CoVaR 1\\%.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. VaR 1\\% $= @{a2.var_i}\\%$; CoVaR 1\\% $= @{a2.ma} + @{a2.bq} = @{a2.covar}\\%$; at the median: $@{a2.ma} - @{a2.bm} = @{a2.med}\\%$', '1. VaR 1\\% $= @{a2.var_i}\\%$; CoVaR 1\\% $= @{a2.ma} + @{a2.bq} = @{a2.covar}\\%$; la mediană: $@{a2.ma} - @{a2.bm} = @{a2.med}\\%$'),
                 T('2. $@{a2.b} \\times @{a2.diff} = @{a2.d}\\%$', '2. $@{a2.b} \\times @{a2.diff} = @{a2.d}\\%$'),
                 T('3. VaR: BAC ($@{a2.var_i}$) $>$ JPM ($@{a1.var_i}$); $\\Delta$CoVaR: JPM ($@{a1.d}$) $>$ BAC ($@{a2.d}$).', '3. VaR: BAC ($@{a2.var_i}$) $>$ JPM ($@{a1.var_i}$); $\\Delta$CoVaR: JPM ($@{a1.d}$) $>$ BAC ($@{a2.d}$).'),
                 T('The riskier bank on its own is not the larger contributor: the slope $\\hat b$ of BAC is much smaller.', 'Banca mai riscantă luată separat nu este cea care contribuie mai mult: panta $\\hat b$ a BAC este mult mai mică.')),
           size='scriptsize')


def a_table():
    rows = []
    half = len(days) // 2
    for i in range(half):
        l, r = days[i], days[i + half]
        f = lambda d: (f'{int(d[8:])}.{d[5:7]} & ' + ' & '.join(f'${TB[d][c]:.2f}$' for c in ('sp500', 'JPM', 'GS')))   # noqa: E731
        rows.append(f(l) + ' & ' + f(r))
    return table('lrrr|lrrr', T('Day & S\\&P 500 & JPM & GS & Day & S\\&P 500 & JPM & GS', 'Ziua & S\\&P 500 & JPM & GS & Ziua & S\\&P 500 & JPM & GS'),
                 [r.replace('$-', '$-') for r in rows], size='scriptsize')


D.frame(T('Data for A3 and A4: 20 Days of March 2020', 'Datele pentru A3 și A4: 20 de zile din martie 2020'), a_table() + items(
    T('Daily log returns in \\%, 2--27 March 2020 (day.month); JPM: JPMorgan Chase, GS: Goldman Sachs', 'Randamente logaritmice zilnice în \\%, 2--27 martie 2020 (ziua.luna); JPM: JPMorgan Chase, GS: Goldman Sachs'),
    T('The same table is in the notebook (section A3)', 'Același tabel se află în notebook (secțiunea A3)')), 'footnotesize')

D.solved(T('A3: MES and LRMES of JPMorgan Chase', 'A3: MES și LRMES pentru JPMorgan Chase'),
         items(T('Use the 20 days of the table (market: S\\&P 500).', 'Folosiți cele 20 de zile din tabel (piața: S\\&P 500).'),
               T('1. Find the $k = \\lceil 20 \\times 0.10 \\rceil$ worst market days and compute MES 10\\% of JPM.', '1. Găsiți cele $k = \\lceil 20 \\times 0{,}10 \\rceil$ cele mai proaste zile ale pieței și calculați MES 10\\% pentru JPM.'),
               T('2. Compute ES 10\\% of the market on the same days and the ratio MES/ES.', '2. Calculați ES 10\\% al pieței în aceleași zile și raportul MES/ES.'),
               T('3. Compute $\\mathrm{MES}_{2\\%}$: the average loss of JPM on the days when the S\\&P 500 fell by more than 2\\%.', '3. Calculați $\\mathrm{MES}_{2\\%}$: pierderea medie a JPM în zilele în care S\\&P 500 a scăzut cu peste 2\\%.'),
               T('4. Compute LRMES.', '4. Calculați LRMES.'),
               T('Report: five numbers and one sentence.', 'Raportați: cinci valori și o frază.')),
         items(T('1. $k = 2$: 16.03 ($@{a3.m1}$) and 12.03 ($@{a3.m2}$); MES 10\\% $= -(@{a3.b1} + (@{a3.b2}))/2 = @{a3.mes}\\%$', '1. $k = 2$: 16.03 ($@{a3.m1}$) și 12.03 ($@{a3.m2}$); MES 10\\% $= -(@{a3.b1} + (@{a3.b2}))/2 = @{a3.mes}\\%$'),
               T('2. ES 10\\% $= @{a3.es}\\%$; MES/ES $= @{a3.ratio}$', '2. ES 10\\% $= @{a3.es}\\%$; MES/ES $= @{a3.ratio}$'),
               T('3. @{a3.n2} days below $-2\\%$; the JPM returns on them sum to $@{a3.sum2}$: $\\mathrm{MES}_{2\\%} = @{a3.mes2}\\%$', '3. @{a3.n2} zile sub $-2\\%$; randamentele JPM din aceste zile însumează $@{a3.sum2}$: $\\mathrm{MES}_{2\\%} = @{a3.mes2}\\%$'),
               T('4. $1 - \\exp(-18 \\times @{a3.mes2}/100) = 1 - \\exp(-@{a3.x18}) = 1 - @{a3.exp} = @{a3.lr}\\%$', '4. $1 - \\exp(-18 \\times @{a3.mes2}/100) = 1 - \\exp(-@{a3.x18}) = 1 - @{a3.exp} = @{a3.lr}\\%$'),
               T('On the market\'s worst days JPM lost more than the market; measured in this crisis month, LRMES is very high.', 'În cele mai proaste zile ale pieței, JPM a pierdut mai mult decît piața; măsurat în această lună de criză, LRMES este foarte mare.')),
         size='scriptsize')

D.proposed(T('A4: MES and LRMES of Goldman Sachs', 'A4: MES și LRMES pentru Goldman Sachs'),
           items(T('Use the same table for GS. Model: A3.', 'Folosiți același tabel pentru GS. Model: A3.'),
                 T('1. Compute MES 10\\% of GS and the ratio MES/ES of the market.', '1. Calculați MES 10\\% pentru GS și raportul MES/ES al pieței.'),
                 T('2. Compute $\\mathrm{MES}_{2\\%}$ and LRMES of GS.', '2. Calculați $\\mathrm{MES}_{2\\%}$ și LRMES pentru GS.'),
                 T('3. Compare GS with JPM on both measures.', '3. Comparați GS cu JPM după ambele măsuri.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. MES 10\\% $= -(@{a4.b1} + (@{a4.b2}))/2 = @{a4.mes}\\%$; MES/ES $= @{a4.ratio}$', '1. MES 10\\% $= -(@{a4.b1} + (@{a4.b2}))/2 = @{a4.mes}\\%$; MES/ES $= @{a4.ratio}$'),
                 T('2. $\\mathrm{MES}_{2\\%} = -(@{a4.sum2})/@{a4.n2} = @{a4.mes2}\\%$; LRMES $= 1 - \\exp(-@{a4.x18}) = @{a4.lr}\\%$', '2. $\\mathrm{MES}_{2\\%} = -(@{a4.sum2})/@{a4.n2} = @{a4.mes2}\\%$; LRMES $= 1 - \\exp(-@{a4.x18}) = @{a4.lr}\\%$'),
                 T('3. GS: MES 10\\% higher ($@{a4.mes}$ against $@{a3.mes}$), LRMES almost the same ($@{a4.lr}$ against $@{a3.lr}$).', '3. GS: MES 10\\% mai mare ($@{a4.mes}$ față de $@{a3.mes}$), LRMES aproape același ($@{a4.lr}$ față de $@{a3.lr}$).'),
                 T('With two tail days, the ranking by MES 10\\% rests on two numbers: it is fragile.', 'Cu două zile în coadă, ordonarea după MES 10\\% se sprijină pe două numere: este fragilă.')),
           size='scriptsize')

D.solved(T('A5: SRISK of One Bank', 'A5: SRISK pentru o bancă'),
         items(T('A bank has debt $D = 900$ and market value of equity $W = 100$ (billion RON); LRMES $= 55\\%$; $k = 8\\%$.', 'O bancă are datorii $D = 900$ și valoarea de piață a capitalului propriu $W = 100$ (miliarde de lei); LRMES $= 55\\%$; $k = 8\\%$.'),
               T('1. Compute the leverage $L$ and SRISK.', '1. Calculați efectul de levier $L$ și SRISK.'),
               T('2. Compute the break-even leverage $L^*$.', '2. Calculați efectul de levier critic $L^*$.'),
               T('3. Recompute $L$ and SRISK after a fall of 30\\% in $W$, with $D$ unchanged.', '3. Recalculați $L$ și SRISK după o scădere de 30\\% a lui $W$, cu $D$ neschimbat.'),
               T('Report: five numbers and one sentence.', 'Raportați: cinci valori și o frază.')),
         items(T('1. $L = 1000/100 = @{a5.L}$; SRISK $= 0.08 \\times 900 - 0.92 \\times 0.45 \\times 100 = @{a5.kD} - @{a5.cap} = @{a5.srisk}$ billion', '1. $L = 1000/100 = @{a5.L}$; SRISK $= 0{,}08 \\times 900 - 0{,}92 \\times 0{,}45 \\times 100 = @{a5.kD} - @{a5.cap} = @{a5.srisk}$ miliarde'),
               T('2. $L^* = 1 + 0.92 \\times 0.45/0.08 = @{a5.Lstar}$; $L = @{a5.L} > L^*$, so SRISK $> 0$', '2. $L^* = 1 + 0{,}92 \\times 0{,}45/0{,}08 = @{a5.Lstar}$; $L = @{a5.L} > L^*$, deci SRISK $> 0$'),
               T('3. $W = 70$: $L = 970/70 = @{a5.Lf}$; SRISK $= @{a5.kD} - @{a5.capf} = @{a5.srisk_fall}$ billion', '3. $W = 70$: $L = 970/70 = @{a5.Lf}$; SRISK $= @{a5.kD} - @{a5.capf} = @{a5.srisk_fall}$ miliarde'),
               T('A fall in the share price raises leverage and SRISK at the same time: SRISK grows in a crisis before any loan defaults.', 'O scădere a prețului acțiunii crește simultan efectul de levier și SRISK: SRISK crește într-o criză înainte ca vreun credit să devină neperformant.')),
         size='scriptsize')

D.proposed(T('A6: SRISK of Two Banks', 'A6: SRISK pentru două bănci'),
           items(T('Bank X: $D = 570$, $W = 30$, LRMES $= 50\\%$; bank Y: $D = 180$, $W = 60$, LRMES $= 65\\%$ (billion RON). Model: A5.', 'Banca X: $D = 570$, $W = 30$, LRMES $= 50\\%$; banca Y: $D = 180$, $W = 60$, LRMES $= 65\\%$ (miliarde de lei). Model: A5.'),
                 T('1. Compute the leverage and SRISK of each bank with $k = 8\\%$.', '1. Calculați efectul de levier și SRISK pentru fiecare bancă, cu $k = 8\\%$.'),
                 T('2. Compute the aggregate SRISK and the share of each bank.', '2. Calculați SRISK agregat și ponderea fiecărei bănci.'),
                 T('3. Recompute both SRISK values with $k = 5.5\\%$.', '3. Recalculați cele două valori SRISK cu $k = 5{,}5\\%$.'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $L_X = @{a6.Lx}$, $L_Y = @{a6.Ly}$; $\\mathrm{SRISK}_X = @{a6.k8.kdx} - @{a6.k8.cx} = @{a6.k8.X}$; $\\mathrm{SRISK}_Y = @{a6.k8.kdy} - @{a6.k8.cy} = @{a6.k8.Y}$', '1. $L_X = @{a6.Lx}$, $L_Y = @{a6.Ly}$; $\\mathrm{SRISK}_X = @{a6.k8.kdx} - @{a6.k8.cx} = @{a6.k8.X}$; $\\mathrm{SRISK}_Y = @{a6.k8.kdy} - @{a6.k8.cy} = @{a6.k8.Y}$'),
                 T('2. Aggregate $= \\max(@{a6.k8.X}, 0) + \\max(@{a6.k8.Y}, 0) = @{a6.k8.tot}$: X 100\\%, Y 0\\%', '2. Agregat $= \\max(@{a6.k8.X}; 0) + \\max(@{a6.k8.Y}; 0) = @{a6.k8.tot}$: X 100\\%, Y 0\\%'),
                 T('3. $\\mathrm{SRISK}_X = @{a6.k55.kdx} - @{a6.k55.cx} = @{a6.k55.X}$; $\\mathrm{SRISK}_Y = @{a6.k55.kdy} - @{a6.k55.cy} = @{a6.k55.Y}$', '3. $\\mathrm{SRISK}_X = @{a6.k55.kdx} - @{a6.k55.cx} = @{a6.k55.X}$; $\\mathrm{SRISK}_Y = @{a6.k55.kdy} - @{a6.k55.cy} = @{a6.k55.Y}$'),
                 T('Y has the larger LRMES, yet X is the systemic one: leverage dominates; the ranking does not change with $k$.', 'Y are LRMES mai mare, dar X este banca sistemică: efectul de levier domină; ordinea nu se schimbă cu $k$.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: MES and $\\Delta$CoVaR of the US Banks [Solved]', 'B1: MES și $\\Delta$CoVaR pentru băncile americane [Rezolvat]'),
       T('do MES and $\\Delta$CoVaR point to the same most systemic US bank?', 'indică MES și $\\Delta$CoVaR aceeași bancă americană ca fiind cea mai importantă sistemic?'),
       T('six US banks and the S\\&P 500, daily log returns since 2005', 'cele șase bănci americane și S\\&P 500, randamente logaritmice zilnice din 2005'),
       [T('Compute MES 5\\% of each bank with the S\\&P 500 as the market.', 'Calculați MES 5\\% pentru fiecare bancă, cu S\\&P 500 ca piață.'),
        T('Compute $\\Delta$CoVaR 1\\% of the S\\&P 500 given each bank by quantile regression.', 'Calculați $\\Delta$CoVaR 1\\% al S\\&P 500 condiționat de fiecare bancă, prin regresie cuantilică.'),
        T('Compute 95\\% moving-block bootstrap intervals for both measures (blocks of 20 days).', 'Calculați intervale bootstrap pe blocuri mobile de 95\\% pentru ambele măsuri (blocuri de 20 de zile).'),
        T('Compute the Spearman correlation between the two rankings.', 'Calculați corelația Spearman dintre cele două ordonări.'),
        T('Interpretation: do the two measures agree on the most systemic bank?', 'Interpretare: sînt cele două măsuri de acord asupra celei mai importante bănci sistemice?')],
       T('a table of twelve numbers with intervals, the chart and two sentences', 'un tabel cu douăsprezece valori și intervalele lor, graficul și două fraze'), size='footnotesize', nb='B1')


def brow(p, k):
    return f'{k} & $@{{{p}.{k}.m}}$ & [@{{{p}.{k}.mlo}}, @{{{p}.{k}.mhi}}] & $@{{{p}.{k}.d}}$ & [@{{{p}.{k}.dlo}}, @{{{p}.{k}.dhi}}]'


D.frame(T('B1: Solution (1/2) [Solved]', 'B1: rezolvare (1/2) [Rezolvat]'), table(
    'lrrrr', T('& MES 5\\% & 95\\% interval & $\\Delta$CoVaR 1\\% & 95\\% interval', '& MES 5\\% & interval 95\\% & $\\Delta$CoVaR 1\\% & interval 95\\%'),
    [brow('b1', k) for k in US], size='footnotesize') + items(
    T('MES 5\\%: average loss of the bank on the 5\\% worst days of the S\\&P 500; $\\Delta$CoVaR 1\\%: quantile regression of the S\\&P 500 on the bank; intervals: moving-block bootstrap, blocks of 20 days',
      'MES 5\\%: pierderea medie a băncii în cele mai proaste 5\\% dintre zilele S\\&P 500; $\\Delta$CoVaR 1\\%: regresia cuantilică a S\\&P 500 pe bancă; intervale: bootstrap pe blocuri mobile, blocuri de 20 de zile'),
    T('MES is about twice as large as $\\Delta$CoVaR: the two measures are on different scales and answer different questions', 'MES este de aproximativ două ori mai mare decît $\\Delta$CoVaR: cele două măsuri au scări diferite și răspund la întrebări diferite')) + qlsem(), 'footnotesize')

D.frame(T('B1: Solution (2/2) [Solved]', 'B1: rezolvare (2/2) [Rezolvat]'), fig('ch15_sem_b1', h='0.52') + items(
    T('$n = @{b1.n}$ days; largest MES: @{b1.topm}; largest $\\Delta$CoVaR: @{b1.topd}; Spearman correlation of the rankings $@{b1.sp}$', '$n = @{b1.n}$ zile; cel mai mare MES: @{b1.topm}; cel mai mare $\\Delta$CoVaR: @{b1.topd}; corelația Spearman a ordonărilor $@{b1.sp}$'),
    T('Interpretation: no: MES ranks first the banks that fall most with the market (Citigroup, Bank of America), $\\Delta$CoVaR the banks whose distress moves the market most; the intervals overlap, so neither ranking is sharp',
      'Interpretare: nu: MES pune pe primele locuri băncile care scad cel mai mult odată cu piața (Citigroup, Bank of America), $\\Delta$CoVaR băncile a căror criză mișcă cel mai mult piața; intervalele se suprapun, deci niciuna dintre ordonări nu este clară')) + qlsem(), 'scriptsize')

D.task(T('B2: European and Romanian Banks [Proposed]', 'B2: băncile europene și cele românești [Propus]'),
       T('does the disagreement between MES and $\\Delta$CoVaR found in B1 also appear in Europe and in Romania? Model: B1.', 'apare și în Europa și în România dezacordul dintre MES și $\\Delta$CoVaR găsit în B1? Model: B1.'),
       T('five European banks and the Euro Stoxx 50 since 2005; TLV, BRD and the BET since 2010', 'cele cinci bănci europene și Euro Stoxx 50 din 2005; TLV, BRD și BET din 2010'),
       [T('Compute MES 5\\% and $\\Delta$CoVaR 1\\% with bootstrap intervals for each bank, against its own market.', 'Calculați MES 5\\% și $\\Delta$CoVaR 1\\% cu intervale bootstrap pentru fiecare bancă, față de propria piață.'),
        T('Compute the Spearman correlation of the two rankings for the European banks.', 'Calculați corelația Spearman a celor două ordonări pentru băncile europene.'),
        T('Compare the two Romanian banks on both measures.', 'Comparați cele două bănci românești după ambele măsuri.'),
        T('Interpretation: is the disagreement between MES and $\\Delta$CoVaR as strong in Europe as in the US?', 'Interpretare: este dezacordul dintre MES și $\\Delta$CoVaR la fel de puternic în Europa ca în SUA?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrr', T('& MES 5\\% & 95\\% interval & $\\Delta$CoVaR 1\\% & 95\\% interval', '& MES 5\\% & interval 95\\% & $\\Delta$CoVaR 1\\% & interval 95\\%'),
    [brow('b2eu', k) for k in EU] + ['\\midrule'] + [brow('b2ro', k) for k in RO], size='scriptsize').replace('\\midrule \\\\', '\\midrule') + items(
    T('Europe: largest MES @{b2eu.topm}, largest $\\Delta$CoVaR @{b2eu.topd}; Spearman $@{b2eu.sp}$; HSBC has by far the smallest MES', 'Europa: cel mai mare MES @{b2eu.topm}, cel mai mare $\\Delta$CoVaR @{b2eu.topd}; Spearman $@{b2eu.sp}$; HSBC are de departe cel mai mic MES'),
    T('Romania: TLV has the larger MES and $\\Delta$CoVaR, but the intervals of the two banks overlap widely', 'România: TLV are MES și $\\Delta$CoVaR mai mari, dar intervalele celor două bănci se suprapun mult'),
    T('Interpretation: no: in Europe the two rankings agree much more than in the US; the banks are larger relative to the Euro Stoxx 50, so falling with the market and moving the market go together',
      'Interpretare: nu: în Europa, cele două ordonări sînt mult mai apropiate decît în SUA; băncile sînt mai mari în raport cu Euro Stoxx 50, deci a scădea odată cu piața și a mișca piața merg împreună')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: US and European Banks in 2008 [Solved]', 'B3: băncile americane și europene în 2008 [Rezolvat]'),
       T('did the link between US and European banks become stronger after Lehman: contagion or interdependence?', 'a devenit mai puternică legătura dintre băncile americane și cele europene după Lehman: contagiune sau interdependență?'),
       T('equal-weighted portfolios of the six US and the five European banks, common days; calm: January 2005 -- June 2007; crisis: 15 September -- 31 December 2008',
         'portofoliile cu ponderi egale ale celor șase bănci americane și cinci bănci europene, zilele comune; perioada calmă: ianuarie 2005 -- iunie 2007; criza: 15 septembrie -- 31 decembrie 2008'),
       [T('Compute the correlation of the two portfolios in the calm and in the crisis period.', 'Calculați corelația dintre cele două portofolii în perioada calmă și în criză.'),
        T('Test the rise in correlation with the one-sided Fisher $z$ test at 5\\%.', 'Testați creșterea corelației cu testul $z$ unilateral al lui Fisher, la 5\\%.'),
        T('Compute $\\delta$ from the variance of the US portfolio and the Forbes--Rigobon corrected correlation.', 'Calculați $\\delta$ din varianța portofoliului american și corelația corectată Forbes--Rigobon.'),
        T('Repeat the test with the corrected correlation.', 'Repetați testul cu corelația corectată.'),
        T('Interpretation: contagion or interdependence?', 'Interpretare: contagiune sau interdependență?')],
       T('six numbers, two decisions, the chart and two sentences', 'șase valori, două decizii, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch15_sem_b3', h='0.44') + items(
    T('$r_0 = @{b3.r0}$ ($n_0 = @{b3.n0}$), $r_1 = @{b3.r1}$ ($n_1 = @{b3.n1}$); $z = (@{b3.at1} - @{b3.at0})/@{b3.se} = @{b3.z}$, p $@{b3.p}$: the raw rise is significant',
      '$r_0 = @{b3.r0}$ ($n_0 = @{b3.n0}$), $r_1 = @{b3.r1}$ ($n_1 = @{b3.n1}$); $z = (@{b3.at1} - @{b3.at0})/@{b3.se} = @{b3.z}$, p $@{b3.p}$: creșterea necorectată este semnificativă'),
    T('Variance of the US portfolio: $@{b3.v0}$ $\\to$ $@{b3.v1}$, $\\delta = @{b3.delta}$; $\\rho^* = @{b3.r1}/\\sqrt{1 + @{b3.delta} \\times @{b3.r2}} = @{b3.radj}$; $z = @{b3.za}$, p $@{b3.pa}$: no rise',
      'Varianța portofoliului american: $@{b3.v0}$ $\\to$ $@{b3.v1}$, $\\delta = @{b3.delta}$; $\\rho^* = @{b3.r1}/\\sqrt{1 + @{b3.delta} \\times @{b3.r2}} = @{b3.radj}$; $z = @{b3.za}$, p $@{b3.pa}$: nicio creștere'),
    T('Interpretation: interdependence: the correlation rose because the variance of the US banks rose about @{b3.vr} times; after the correction the link is not stronger \\refFR',
      'Interpretare: interdependență: corelația a crescut deoarece varianța băncilor americane a crescut de aproximativ @{b3.vr} ori; după corecție, legătura nu este mai puternică \\refFR')) + qlsem(), 'scriptsize')

D.task(T('B4: European and Romanian Banks in 2020 [Proposed]', 'B4: băncile europene și cele românești în 2020 [Propus]'),
       T('did the link between European and Romanian banks become stronger in the COVID-19 crash? Model: B3.', 'a devenit mai puternică legătura dintre băncile europene și cele românești în prăbușirea din perioada COVID-19? Model: B3.'),
       T('equal-weighted portfolios of the five European banks and of TLV and BRD, common days since 2010; calm: 2019; crisis: 20 February -- 30 April 2020',
         'portofoliile cu ponderi egale ale celor cinci bănci europene și ale TLV și BRD, zilele comune din 2010; perioada calmă: 2019; criza: 20 februarie -- 30 aprilie 2020'),
       [T('Compute the correlation in the calm and in the crisis period and test the rise with the Fisher $z$ test.', 'Calculați corelația în perioada calmă și în criză și testați creșterea cu testul $z$ al lui Fisher.'),
        T('Compute $\\delta$ from the variance of the European portfolio and the corrected correlation.', 'Calculați $\\delta$ din varianța portofoliului european și corelația corectată.'),
        T('Repeat the test with the corrected correlation.', 'Repetați testul cu corelația corectată.'),
        T('Interpretation: contagion or interdependence?', 'Interpretare: contagiune sau interdependență?')],
       T('six numbers, two decisions and two sentences', 'șase valori, două decizii și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), items(
    T('$r_0 = @{b4.r0}$ ($n_0 = @{b4.n0}$), $r_1 = @{b4.r1}$ ($n_1 = @{b4.n1}$); $z = @{b4.z}$, p $@{b4.p}$: the raw rise is significant',
      '$r_0 = @{b4.r0}$ ($n_0 = @{b4.n0}$), $r_1 = @{b4.r1}$ ($n_1 = @{b4.n1}$); $z = @{b4.z}$, p $@{b4.p}$: creșterea necorectată este semnificativă'),
    T('Variance of the European portfolio: $@{b4.v0}$ $\\to$ $@{b4.v1}$, $\\delta = @{b4.delta}$; $\\rho^* = @{b4.r1}/\\sqrt{@{b4.inside}} = @{b4.radj}$; $z = @{b4.za}$, p $@{b4.pa}$: not significant',
      'Varianța portofoliului european: $@{b4.v0}$ $\\to$ $@{b4.v1}$, $\\delta = @{b4.delta}$; $\\rho^* = @{b4.r1}/\\sqrt{@{b4.inside}} = @{b4.radj}$; $z = @{b4.za}$, p $@{b4.pa}$: nesemnificativ'),
    T('Interpretation: the evidence points to interdependence; with 2019 as the calm period the correlation was unusually low, and @{b4.n1} crisis days give a wide interval, so the test has little power',
      'Interpretare: datele indică interdependență; cu 2019 ca perioadă calmă, corelația a fost neobișnuit de mică, iar cele @{b4.n1} zile de criză dau un interval larg, deci testul are o putere redusă')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Did MES in 2019 Rank the Losses of March 2020? [Solved]', 'B5: a ordonat MES din 2019 pierderile din martie 2020? [Rezolvat]'),
       T('do the banks with a high MES before a crisis lose more in it, as in \\refAPPR?', 'pierd mai mult într-o criză băncile cu MES mare înaintea crizei, ca în \\refAPPR?'),
       T('11 US and European banks; MES 5\\% on the daily returns of 2019; buy-and-hold return from 19 February to 23 March 2020', '11 bănci americane și europene; MES 5\\% pe randamentele zilnice din 2019; randamentul cumpără-și-păstrează din 19 februarie pînă în 23 martie 2020'),
       [T('Compute MES 5\\% of each bank in 2019 against its own market.', 'Calculați MES 5\\% pentru fiecare bancă în 2019, față de propria piață.'),
        T('Compute the return of each bank from 19 February to 23 March 2020.', 'Calculați randamentul fiecărei bănci din 19 februarie pînă în 23 martie 2020.'),
        T('Compute the Spearman correlation between MES and the return and its p-value.', 'Calculați corelația Spearman dintre MES și randament și p-valoarea ei.'),
        T('Interpretation: did MES rank the March 2020 losses?', 'Interpretare: a ordonat MES pierderile din martie 2020?')],
       T('the chart, the correlation with its p-value and two sentences', 'graficul, corelația cu p-valoarea ei și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch15_sem_b5', h='0.40') + items(
    T('Highest MES in 2019: @{b5.topm} ($@{b5.C.m}\\%$), also the worst return ($@{b5.C.r}\\%$); lowest MES: HSBC ($@{b5.HSBA.m}\\%$), also the smallest loss ($@{b5.HSBA.r}\\%$)',
      'Cel mai mare MES în 2019: @{b5.topm} ($@{b5.C.m}\\%$), și cel mai prost randament ($@{b5.C.r}\\%$); cel mai mic MES: HSBC ($@{b5.HSBA.m}\\%$), și cea mai mică pierdere ($@{b5.HSBA.r}\\%$)'),
    T('Spearman correlation $@{b5.sp}$, p $= @{b5.p}$ ($n = @{b5.n}$)', 'Corelația Spearman $@{b5.sp}$, p $= @{b5.p}$ ($n = @{b5.n}$)'),
    T('Interpretation: partly: the sign is the one of \\refAPPR and the extremes match, but with 11 banks the correlation is not significant at 5\\%; the middle of the ranking is noise',
      'Interpretare: parțial: semnul este cel din \\refAPPR, iar extremele coincid, dar cu 11 bănci corelația nu este semnificativă la 5\\%; mijlocul ordonării este zgomot')) + qlsem(), 'scriptsize')

D.task(T('B6: Did MES in 2022 Rank the Losses of March 2023? [Proposed]', 'B6: a ordonat MES din 2022 pierderile din martie 2023? [Propus]'),
       T('does the result of B5 hold in the banking turmoil of March 2023? Model: B5.', 'se păstrează rezultatul din B5 în tulburările bancare din martie 2023? Model: B5.'),
       T('13 banks (with TLV and BRD); MES 5\\% on the daily returns of 2022; return from 28 February to 31 March 2023', '13 bănci (cu TLV și BRD); MES 5\\% pe randamentele zilnice din 2022; randamentul din 28 februarie pînă în 31 martie 2023'),
       [T('Compute MES 5\\% of each bank in 2022 against its own market.', 'Calculați MES 5\\% pentru fiecare bancă în 2022, față de propria piață.'),
        T('Compute the return of each bank from 28 February to 31 March 2023.', 'Calculați randamentul fiecărei bănci din 28 februarie pînă în 31 martie 2023.'),
        T('Compute the Spearman correlation and its p-value.', 'Calculați corelația Spearman și p-valoarea ei.'),
        T('Interpretation: why may MES fail to rank the losses of March 2023?', 'Interpretare: de ce poate MES să nu ordoneze pierderile din martie 2023?')],
       T('one table, the correlation with its p-value and two sentences', 'un tabel, corelația cu p-valoarea ei și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrr|lrr', T('Bank & MES 5\\% & return (\\%) & Bank & MES 5\\% & return (\\%)', 'Banca & MES 5\\% & randament (\\%) & Banca & MES 5\\% & randament (\\%)'),
    [b5row('b6', a) + ' & ' + (b5row('b6', b) if b else '& &') for a, b in zip(['JPM', 'BAC', 'C', 'GS', 'MS', 'WFC', 'DBK'], ['BNP', 'SAN', 'INGA', 'HSBA', 'TLV', 'BRD', None])],
    size='scriptsize') + items(
    T('Spearman $@{b6.sp}$, p $= @{b6.p}$ ($n = @{b6.n}$); highest MES: @{b6.topm}; worst return: @{b6.worst}', 'Spearman $@{b6.sp}$, p $= @{b6.p}$ ($n = @{b6.n}$); cel mai mare MES: @{b6.topm}; cel mai prost randament: @{b6.worst}'),
    T('Interpretation: the 2023 shock came from deposits and interest-rate risk at a few banks (SVB, Credit Suisse), not from a market crash: MES, built on market co-movement, ranks such a shock only weakly',
      'Interpretare: șocul din 2023 a venit din depozite și din riscul de dobîndă al cîtorva bănci (SVB, Credit Suisse), nu dintr-o prăbușire a pieței: MES, construit pe mișcarea comună cu piața, ordonează doar slab un astfel de șoc')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: How Connected Are Romanian and Euro-Area Banks? [Proposed]', 'C1: cît de conectate sînt băncile românești cu cele din zona euro? [Propus]'),
       T('what share of the risk of TLV and BRD comes from the European banks, and does it rise in crises?', 'ce parte din riscul TLV și BRD vine de la băncile europene și crește această parte în crize?'),
       T('the five European banks, TLV and BRD, daily log returns on common days since 2010; models: B1, the connectedness slide', 'cele cinci bănci europene, TLV și BRD, randamente logaritmice zilnice în zilele comune din 2010; modele: B1, slide-ul despre conectivitate'),
       [T('Fit a VAR to the seven return series (order by BIC) and compute the connectedness table for $H = 10$ days.', 'Estimați un model VAR pe cele șapte serii de randamente (ordinul ales prin BIC) și calculați tabelul de conectivitate pentru $H = 10$ zile.'),
        T('Compute the share of the forecast error variance of TLV and BRD that comes from the five European banks.', 'Calculați proporția din varianța erorii de prognoză a TLV și BRD care vine de la cele cinci bănci europene.'),
        T('Repeat it on rolling windows of 250 days and compare 2019 with 2020.', 'Repetați calculul pe ferestre mobile de 250 de zile și comparați 2019 cu 2020.'),
        T('Interpretation: is the systemic risk of the Romanian banks imported?', 'Interpretare: este riscul sistemic al băncilor românești importat?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), items(
    T('$n = @{c1.n}$ common days, VAR(@{c1.p}); total connectedness of the seven banks @{c1.total}\\%', '$n = @{c1.n}$ zile comune, VAR(@{c1.p}); conectivitatea totală a celor șapte bănci @{c1.total}\\%'),
    T('Share from the European banks, whole sample: TLV @{c1.tlv}\\%, BRD @{c1.brd}\\%', 'Proporția venită de la băncile europene, eșantionul complet: TLV @{c1.tlv}\\%, BRD @{c1.brd}\\%'),
    T('Rolling 250-day windows (average of TLV and BRD): mean @{c1.mean}\\%, 2019 @{c1.y19}\\%, 2020 @{c1.y20}\\%, maximum @{c1.max}\\% (window ending @{c1.dmax})',
      'Ferestre mobile de 250 de zile (media TLV și BRD): media @{c1.mean}\\%, 2019 @{c1.y19}\\%, 2020 @{c1.y20}\\%, maximul @{c1.max}\\% (fereastra care se încheie la @{c1.dmax})'),
    T('Reading: in calm years almost all the risk of the Romanian banks is local; in crises a large share comes from the euro area',
      'Lectura rezultatelor: în anii liniștiți, aproape tot riscul băncilor românești este local; în crize, o parte mare vine din zona euro'),
    T('Project design: different trading hours (two-day returns), $\\Delta$CoVaR with the European bank portfolio as the conditioning variable, bootstrap intervals for the rolling shares',
      'Designul proiectului: ore de tranzacționare diferite (randamente pe două zile), $\\Delta$CoVaR cu portofoliul băncilor europene ca variabilă de condiționare, intervale bootstrap pentru proporțiile pe ferestre mobile')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the systemic risk of JPMorgan Chase. The answer:', 'Un student a cerut unui asistent AI să rezume riscul sistemic al JPMorgan Chase. Răspunsul:'),
    T('\\aiprompt{(a) The CoVaR 99\\% of the S\\&P 500 given JPM is -@{c2.covar}\\%.}', '\\aiprompt{(a) CoVaR 99\\% al S\\&P 500 condiționat de JPM este -@{c2.covar}\\%.}'),
    T('\\aiprompt{(b) Delta-CoVaR = CoVaR - VaR of JPM = @{c2.covar} - @{c2.var} = @{c2.wrong}\\%, so JPM reduces systemic risk.}', '\\aiprompt{(b) Delta-CoVaR = CoVaR - VaR-ul JPM = @{c2.covar} - @{c2.var} = @{c2.wrong}\\%, deci JPM reduce riscul sistemic.}'),
    T('\\aiprompt{(c) MES 5\\% is the average loss of the S\\&P 500 on the worst 5\\% of days of JPM.}', '\\aiprompt{(c) MES 5\\% este pierderea medie a S\\&P 500 în cele mai proaste 5\\% dintre zilele JPM.}'),
    T('\\aiprompt{(d) The bank with the largest VaR 1\\% is always the most systemic one.}', '\\aiprompt{(d) Banca cu cel mai mare VaR 1\\% este întotdeauna cea mai importantă sistemic.}'),
    T('\\aiprompt{(e) The correlation of US and European banks rose from @{b3.r0} to @{b3.r1} in autumn 2008, which proves contagion.}', '\\aiprompt{(e) Corelația dintre băncile americane și cele europene a crescut de la @{b3.r0} la @{b3.r1} în toamna lui 2008, ceea ce dovedește contagiunea.}'),
    T('\\aiprompt{(f) A positive SRISK means that the bank has more capital than it needs in a crisis.}', '\\aiprompt{(f) Un SRISK pozitiv înseamnă că banca are mai mult capital decît îi trebuie într-o criză.}'),
    T('\\aiprompt{(g) Granger causality from bank A to bank B means that past returns of A help to forecast B; it does not prove a causal channel.}', '\\aiprompt{(g) Cauzalitatea Granger de la banca A la banca B înseamnă că randamentele trecute ale lui A ajută la prognoza lui B; nu dovedește un canal cauzal.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of seven verdicts with one line of justification each.', '2. Raportați: o listă de șapte verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong twice: the level is the tail probability (CoVaR 1\\%), and CoVaR is a loss, a positive number: CoVaR 1\\% $= @{c2.covar}\\%$', '(a) Greșit de două ori: nivelul este probabilitatea cozii (CoVaR 1\\%), iar CoVaR este o pierdere, un număr pozitiv: CoVaR 1\\% $= @{c2.covar}\\%$'),
    T('(b) Wrong: $\\Delta$CoVaR compares CoVaR in distress with CoVaR at the median of JPM: $\\Delta$CoVaR 1\\% $= @{c2.d}\\% > 0$ (A1)', '(b) Greșit: $\\Delta$CoVaR compară CoVaR în dificultate cu CoVaR la mediana JPM: $\\Delta$CoVaR 1\\% $= @{c2.d}\\% > 0$ (A1)'),
    T('(c) Wrong: the conditioning is reversed: MES is the average loss of the bank on the market\'s worst days (A3)', '(c) Greșit: condiționarea este inversată: MES este pierderea medie a băncii în cele mai proaste zile ale pieței (A3)'),
    T('(d) Wrong: BAC has a larger VaR 1\\% than JPM but a smaller $\\Delta$CoVaR (A2)', '(d) Greșit: BAC are un VaR 1\\% mai mare decît JPM, dar un $\\Delta$CoVaR mai mic (A2)'),
    T('(e) Wrong: after the Forbes--Rigobon correction the crisis correlation is $@{b3.radj}$: interdependence, not proven contagion (B3)', '(e) Greșit: după corecția Forbes--Rigobon, corelația din criză este $@{b3.radj}$: interdependență, nu o contagiune dovedită (B3)'),
    T('(f) Wrong: SRISK $> 0$ is capital \\textbf{missing} in a crisis; a surplus gives SRISK $< 0$ (A5)', '(f) Greșit: SRISK $> 0$ înseamnă capital \\textbf{lipsă} într-o criză; un surplus dă SRISK $< 0$ (A5)'),
    T('(g) Correct', '(g) Corect')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('CoVaR 1\\% $= -(\\hat a + \\hat b\\,q_{1\\%}(X^i))$ and $\\Delta$CoVaR $= \\hat b\\,(q_{50\\%} - q_{1\\%})$: the system given the bank', 'CoVaR 1\\% $= -(\\hat a + \\hat b\\,q_{1\\%}(X^i))$ și $\\Delta$CoVaR $= \\hat b\\,(q_{50\\%} - q_{1\\%})$: sistemul condiționat de bancă'),
    T('MES: the bank given the market; SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$ grows with leverage', 'MES: banca condiționată de piață; SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$ crește odată cu efectul de levier'),
    T('VaR, MES and $\\Delta$CoVaR rank banks differently; rankings need bootstrap intervals', 'VaR, MES și $\\Delta$CoVaR ordonează diferit băncile; ordonările au nevoie de intervale bootstrap'),
    T('A rise in correlation in a crisis is not proof of contagion: correct for volatility (Forbes--Rigobon)', 'O creștere a corelației într-o criză nu este o dovadă de contagiune: corectați pentru volatilitate (Forbes--Rigobon)'),
    T('An AI answer is a draft: check the level, the sign, the direction of conditioning and the definition of $\\Delta$CoVaR', 'Un răspuns AI este o ciornă: verificați nivelul, semnul, direcția condiționării și definiția $\\Delta$CoVaR')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 15 develops each topic of today: the channels and the crises, CoVaR with state variables, MES and SRISK, networks, macroprudential policy',
      'Cursul 15 dezvoltă fiecare temă de azi: canalele și crizele, CoVaR cu variabile de stare, MES și SRISK, rețele, politica macroprudențială'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: Romanian and euro-area banks, connectedness and $\\Delta$CoVaR in 2011--2012, 2020 and 2023', 'C1 poate deveni un proiect de echipă: băncile românești și cele din zona euro, conectivitatea și $\\Delta$CoVaR în 2011--2012, 2020 și 2023'),
    T('Reading: \\refAB; \\refAPPR; \\refBE; survey: \\refBCHP', 'Lectură: \\refAB; \\refAPPR; \\refBE; sinteză: \\refBCHP')))

D.references(bib(['AB', 'AER', 'APPR', 'BCHP', 'BE', 'DYb', 'FR', 'FSB', 'Granger', 'KB', 'FHH']), per=16)

if __name__ == '__main__':
    D.write(V)
