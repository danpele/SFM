r"""
build_seminar10.py -- Seminarul 10 (VaR, ES și backtesting), EN + RO dintr-o singură sursă
=========================================================================================
Seminarul are loc ÎNAINTEA cursului 10: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_10/sem10_results.json (seminar10.py).
Ieșire:
  EN/Seminars/seminar10_var_es_backtesting.tex          (+ _solutions.tex)
  RO/Seminarii/seminar10_var_es_backtesting_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_10/seminar10.py && python3 latex/build_seminar10.py && python3 latex/sfm_build.py compile 10
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch10_common import NAMES, REFS, SHORT, T, bib, load_sem, pv, put_date   # noqa: E402

S = load_sem()
V = Values()
D = Deck(10, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_10}{SFM_ch10_seminar}'


def money(x):
    return f'{x:,.0f}'.replace(',', '\\,')


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
a1 = A['a1']
for c in ['var1', 'es25', 'var_h', 'ratio']:
    V.put(f'a1.{c}', a1[c], 3 if c == 'ratio' else 2)
V.raw('a1.var1c', money(a1['var1_cur']))
V.raw('a1.es25c', money(a1['es25_cur']))
V.raw('a1.varhc', money(a1['var_h_cur']))
V.put('a1.zs', 2.326 * 1.4, 3)
V.put('a1.fs', 0.0584 / 0.025 * 1.4, 3)
V.put('a1.sq', 2.326 * 1.4 * math.sqrt(10), 2)
a2 = A['a2']
for c in ['q', 'ez', 'fac', 'var1', 'es25', 'ratio']:
    V.put(f'a2.{c}', a2[c], 3)
V.put('a2.var1r', a2['var1'], 2)
V.put('a2.es25r', a2['es25'], 2)
V.raw('a2.var1c', money(a2['var1_cur']))
V.raw('a2.es25c', money(a2['es25_cur']))
V.put('a2.inc', 100 * (a2['var1'] / a1['var1'] - 1), 0)
V.put('a2.ince', 100 * (a2['es25'] / a1['es25'] - 1), 0)
a3 = A['a3']
put_date(V, 'a3.first', a3['first'])
put_date(V, 'a3.last', a3['last'])
w = a3['worst']
V.raw('a3.list1', '; '.join(f'$⁅{x:.2f}⁆$' for x in w[:8]))
V.raw('a3.list2', '; '.join(f'$⁅{x:.2f}⁆$' for x in w[8:]))
V.put('a3.var1', a3['var1'], 2)
V.put('a3.var25', -w[a3['k25'] - 1], 2)
V.put('a3.es25', a3['es25'], 2)
V.put('a3.sum13', -sum(w[:a3['k25']]), 2)
V.raw('a3.var1c', money(1e4 * a3['var1']))
V.raw('a3.es25c', money(1e4 * a3['es25']))
a4 = A['a4']
for c in ['p_any', 'p_both', 'p_one']:
    V.put(f'a4.{c}', 100 * a4[c], 2)
V.put('a4.es1', a4['es1'], 0)
V.put('a4.es2', a4['es2'], 1)
V.put('a4.var2', a4['var2'], 0)
a5 = A['a5']
V.put('a5.l0', a5['l0'], 2)
V.put('a5.l1', a5['l1'], 2)
V.put('a5.lr', a5['lr'], 2)
V.put('a5.p', a5['p'], 3)
V.put('a5.cum', 100 * a5['cum'], 2)
V.put('a5.cump', 100 * a5['cum_prev'], 2)
b5 = A['a5b']
V.put('a5b.l0', b5['l0'], 2)
V.put('a5b.l1', b5['l1'], 2)
V.put('a5b.lr', b5['lr'], 2)
V.put('a5b.p', b5['p'], 3)
a6 = A['a6']
V.put('a6.p01', a6['p01'], 4)
V.put('a6.p11', a6['p11'], 3)
V.put('a6.p', a6['p'], 4)
V.put('a6.ll0', a6['ll0'], 2)
V.put('a6.ll1', a6['ll1'], 2)
for c in ['lr_ind', 'lr_uc', 'lr_cc']:
    V.put(f'a6.{c}', a6[c], 2)
for c in ['p_ind', 'p_uc', 'p_cc']:
    V.raw(f'a6.{c}', pv(a6[c]))
V.put('chi1', stats.chi2.ppf(0.95, 1), 2)
V.put('chi2', stats.chi2.ppf(0.95, 2), 2)
MK = {'HS': 'hs', 'Normal': 'n', 'Student-t': 't', 'Cornish-Fisher': 'cf', 'EVT': 'evt'}


def put5(key, d):
    V.int(f'{key}.n', d['n'])
    V.raw(f'{key}.y0', d['first'][:4])
    V.put(f'{key}.mu', d['mu'], 3)
    V.put(f'{key}.sd', d['sd'], 2)
    V.put(f'{key}.skew', d['skew'], 2)
    V.put(f'{key}.k', d['exkurt'], 1)
    V.put(f'{key}.nu', d['nu'], 2)
    V.put(f'{key}.xi', d['xi'], 2)
    V.put(f'{key}.u', d['u'], 2)
    V.raw(f'{key}.Nu', str(d['Nu']))
    V.put(f'{key}.cfz', d['cf_z'], 2)
    for m, k in MK.items():
        V.put(f'{key}.{k}.v', d[m][0], 2)
        V.put(f'{key}.{k}.e', d[m][1], 2)
    V.put(f'{key}.gap', 100 * (1 - d['Normal'][0] / d['HS'][0]), 0)


put5('b1', S['B1'])
for k, d in S['B2'].items():
    put5(f'b2.{k}', d)


def put_bt(key, d):
    for m, mk in [('HS', 'hs'), ('GARCH-t', 'g')]:
        r = d[m]
        V.int(f'{key}.{mk}.n', r['n'])
        V.raw(f'{key}.{mk}.x', str(r['x']))
        V.put(f'{key}.{mk}.rate', 100 * r['rate'], 2)
        V.put(f'{key}.{mk}.exp', 0.01 * r['n'], 1)
        V.put(f'{key}.{mk}.luc', r['lr_uc'], 2)
        V.raw(f'{key}.{mk}.puc', pv(r['p_uc']))
        V.put(f'{key}.{mk}.lind', r['lr_ind'], 2)
        V.raw(f'{key}.{mk}.pind', pv(r['p_ind']))
        V.put(f'{key}.{mk}.lcc', r['lr_cc'], 2)
        V.raw(f'{key}.{mk}.pcc', pv(r['p_cc']))
        V.raw(f'{key}.{mk}.n11', str(r['n11']))
        V.raw(f'{key}.{mk}.red', str(r['red']))
        V.raw(f'{key}.{mk}.yel', str(r['yellow']))
    for y, v in d['years'].items():
        V.raw(f'{key}.y{y}.hs', str(v['HS']))
        V.raw(f'{key}.y{y}.g', str(v['GARCH-t']))


put_bt('b3', S['B3'])
for k, d in S['B4'].items():
    put_bt(f'b4.{k}', d)


def put_port(key, d):
    V.int(f'{key}.n', d['n'])
    V.raw(f'{key}.y0', d['first'][:4])
    V.put(f'{key}.mu1', d['mu'][0], 3)
    V.put(f'{key}.mu2', d['mu'][1], 3)
    V.put(f'{key}.sd1', d['sd'][0], 2)
    V.put(f'{key}.sd2', d['sd'][1], 2)
    V.put(f'{key}.corr', d['corr'], 2)
    V.put(f'{key}.sp', d['sp'], 3)
    V.put(f'{key}.mp', d['mp'], 3)
    for c in ['var_n', 'es_n', 'var_hs', 'es_hs', 'var_sum', 'var_hs_sum', 'var_gauss', 'es_gauss', 'var_t', 'es_t']:
        V.put(f'{key}.{c}', d[c], 2)
    V.put(f'{key}.v1', d['var_single'][0], 2)
    V.put(f'{key}.v2', d['var_single'][1], 2)
    V.put(f'{key}.ben', 100 * d['benefit'], 0)
    V.put(f'{key}.benh', 100 * (1 - d['var_hs'] / d['var_hs_sum']), 0)
    V.put(f'{key}.c17', d['corr_2017'], 2)
    V.put(f'{key}.c20', d['corr_2020'], 2)
    V.put(f'{key}.td5', d['tail']['0.05'], 2)
    V.put(f'{key}.td1', d['tail']['0.01'], 2)
    c = d['copula']
    V.put(f'{key}.tau', c['tau'], 3)
    V.put(f'{key}.rho', c['rho'], 3)
    V.put(f'{key}.nu', c['nu'], 2)
    V.put(f'{key}.lr', 2 * (c['ll_t'] - c['ll_g']), 1)
    V.put(f'{key}.lam', c['lambda_t'], 3)
    V.put(f'{key}.under', 100 * (1 - d['var_n'] / d['var_hs']), 0)


put_port('b5', S['B5'])
V.raw('b5.nboth', str(S['B5']['n_both']))
V.put('b5.exp', S['B5']['exp_ind'], 2)
put_port('b6', S['B6'])
C1 = S['C1']
V.put('c1.norm', C1['normal'], 3)
c1rows = []
for k in ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp', 'brd']:
    r = C1[k]
    f = lambda key: f'$⁅{r[key]:.2f}⁆$' if key in r else '--'   # noqa: E731
    c1rows.append(f'{SHORT[k]} & {f("all")} & {f("2009")} & {f("2020")} & {f("t")} & $⁅{r["nu"]:.1f}⁆$')
V.put('c1.min', min(min(v for kk, v in C1[k].items() if kk in ('all', '2009', '2020')) for k in C1 if k != 'normal'), 2)
V.put('c1.max', max(max(v for kk, v in C1[k].items() if kk in ('all', '2009', '2020')) for k in C1 if k != 'normal'), 2)
C2 = S['C2']
V.put('c2.var1', C2['var1'], 2)
V.put('c2.es25', C2['es25'], 2)
V.put('c2.v10', C2['var10_sqrt'], 2)
V.put('c2.v10w', C2['var10_wrong'], 1)
V.put('c2.p8', C2['x8']['p'], 3)
V.put('c2.lr8', C2['x8']['lr'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how large can tomorrow\'s loss be, and how do we check a risk number afterwards?',
       '\\textbf{Întrebarea}: cît de mare poate fi pierderea de mîine și cum verificăm ulterior o măsură de risc?'),
     [T('this seminar comes \\textbf{before} Lecture 10: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 10: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: VaR and ES of a position, historical simulation, subadditivity, Kupiec and Christoffersen tests, on paper',
        'Partea A: VaR și ES ale unei poziții, simularea istorică, subaditivitatea, testele Kupiec și Christoffersen, pe hîrtie'),
      T('Part B: five estimation methods, backtests per calendar year and two-asset portfolios on real data, each with an interpretation question',
        'Partea B: cinci metode de estimare, backtesting pe ani calendaristici și portofolii cu două active, pe date reale, fiecare cu o întrebare de interpretare'),
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
    ['A1, A2 & ' + T('VaR 1\\% and ES 2.5\\% of a position: Normal and Student-t', 'VaR 1\\% și ES 2,5\\% ale unei poziții: distribuția Normală și Student-t') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('historical simulation from the worst days of the BET; a subadditivity counterexample', 'simularea istorică din cele mai proaste zile ale BET; un contraexemplu pentru subaditivitate') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('the Kupiec test and the traffic light; the Christoffersen test', 'testul Kupiec și semaforul Basel; testul Christoffersen') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('five estimation methods: BET; TLV, SNP, Bitcoin', 'cinci metode de estimare: BET; TLV, SNP, Bitcoin') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('backtests per calendar year: S\\&P 500; BET, Bitcoin', 'backtesting pe ani calendaristici: S\\&P 500; BET, Bitcoin') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('two-asset portfolios: TLV and SNP; DAX and S\\&P 500', 'portofolii cu două active: TLV și SNP; DAX și S\\&P 500') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('is ES 2.5\\% as large as VaR 1\\% on real data? what is wrong in an AI answer?', 'este ES 2,5\\% la fel de mare ca VaR 1\\% pe date reale? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX, BET & EODHD & ' + T('close', 'închidere') + ' & 2000--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'închidere, 7 zile pe săptămînă') + ' & 2014--2026',
     T('Banca Transilvania (TLV), OMV Petrom (SNP), BRD', 'Banca Transilvania (TLV), OMV Petrom (SNP), BRD') + ' & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2010--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; for portfolios, simple returns $R_t = 100(P_t/P_{t-1} - 1)$ on the common days; the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; pentru portofolii, randamente simple $R_t = 100(P_t/P_{t-1} - 1)$ în zilele comune; ultima zi este 18 septembrie 2026'),
    T('TLV: the days 30 and 31 May 2016 are removed (a price adjustment applied one day late, Chapter 2)',
      'TLV: zilele de 30 și 31 mai 2016 se elimină (o ajustare de preț aplicată cu o zi mai tîrziu, Capitolul 2)'),
    T('In the notebook: \\texttt{returns(\'bet\')}, \\texttt{hs\\_var(x)}, \\texttt{kupiec(x, n)} (package \\texttt{arch} for GARCH); no account or key is needed',
      'În notebook: \\texttt{returns(\'bet\')}, \\texttt{hs\\_var(x)}, \\texttt{kupiec(x, n)} (pachetul \\texttt{arch} pentru GARCH); nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): VaR and ES', 'Noțiuni necesare azi (1/5): VaR și ES'), items(
    (T('$X$: the daily return (P\\&L) of a position, in \\%; level $\\alpha$ = the probability of the tail (1\\% or 2.5\\%)', '$X$: randamentul zilnic (P\\&L) al unei poziții, în \\%; nivelul $\\alpha$ = probabilitatea cozii (1\\% sau 2,5\\%)'),
     [T('$q_\\alpha$: the $\\alpha$-quantile of $X$, a negative number for small $\\alpha$', '$q_\\alpha$: cuantila de ordin $\\alpha$ a lui $X$, un număr negativ pentru $\\alpha$ mic')]),
    (T('\\textbf{VaR} (value at risk): $\\mathrm{VaR}_\\alpha = -q_\\alpha$, the loss exceeded with probability $\\alpha$; we write \\textbf{VaR 1\\%}', '\\textbf{VaR} (value at risk, valoarea expusă la risc): $\\mathrm{VaR}_\\alpha = -q_\\alpha$, pierderea depășită cu probabilitatea $\\alpha$; scriem \\textbf{VaR 1\\%}'),
     [T('a positive number: ``VaR 1\\% = 3\\%\'\' means a loss of more than 3\\% on one day in a hundred', 'un număr pozitiv: „VaR 1\\% = 3\\%” înseamnă o pierdere de peste 3\\% într-o zi din o sută')]),
    (T('\\textbf{ES} (expected shortfall): $\\mathrm{ES}_\\alpha = -E[X \\mid X \\le q_\\alpha]$, the average loss on the worst $\\alpha$ of days; we write \\textbf{ES 2.5\\%}',
       '\\textbf{ES} (expected shortfall): $\\mathrm{ES}_\\alpha = -E[X \\mid X \\le q_\\alpha]$, pierderea medie din cele mai proaste $\\alpha$ dintre zile; scriem \\textbf{ES 2,5\\%}'),
     [T('$\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ at the \\textbf{same} level $\\alpha$', '$\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ la \\textbf{același} nivel $\\alpha$')]),
    (T('\\textbf{Historical simulation} (HS): sort the $n$ returns, $r_{(1)} \\le \\dots \\le r_{(n)}$, $k = \\lceil n\\alpha \\rceil$ (rounded up)', '\\textbf{Simularea istorică} (HS): ordonăm cele $n$ randamente, $r_{(1)} \\le \\dots \\le r_{(n)}$, $k = \\lceil n\\alpha \\rceil$ (rotunjit în sus)'),
     [T('$\\widehat{\\mathrm{VaR}}_\\alpha = -r_{(k)}$, \\quad $\\widehat{\\mathrm{ES}}_\\alpha = -(r_{(1)} + \\dots + r_{(k)})/k$; in money: $W \\times \\mathrm{VaR}/100$ for a position of value $W$',
        '$\\widehat{\\mathrm{VaR}}_\\alpha = -r_{(k)}$, \\quad $\\widehat{\\mathrm{ES}}_\\alpha = -(r_{(1)} + \\dots + r_{(k)})/k$; în bani: $W \\times \\mathrm{VaR}/100$ pentru o poziție cu valoarea $W$')])))

D.frame(T('What You Need for Today (2/5): Parametric VaR and ES', 'Noțiuni necesare azi (2/5): VaR și ES parametrice'), items(
    (T('\\textbf{Normal} $N(\\mu, \\sigma^2)$: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$ ($\\varphi$: the $N(0,1)$ density)',
       '\\textbf{Distribuția Normală} $N(\\mu, \\sigma^2)$: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$ ($\\varphi$: densitatea $N(0,1)$)'),
     [T('$z_{0.01} = -2.326$, $z_{0.025} = -1.960$, $\\varphi(-1.960) = 0.0584$', '$z_{0.01} = -2.326$, $z_{0.025} = -1.960$, $\\varphi(-1.960) = 0.0584$'),
      T('$h$ days (i.i.d., mean 0): $\\mathrm{VaR}^{(h)} = \\sqrt{h}\\,\\mathrm{VaR}^{(1)}$ (the square-root-of-time rule)', '$h$ zile (i.i.d., media 0): $\\mathrm{VaR}^{(h)} = \\sqrt{h}\\,\\mathrm{VaR}^{(1)}$ (regula rădăcinii pătrate a timpului)')]),
    (T('\\textbf{Standardised Student-t} with $\\nu$ degrees of freedom (variance 1), $t_\\alpha = t_\\nu^{-1}(\\alpha)$, $f_\\nu$ the density of $t_\\nu$',
       '\\textbf{Student-t standardizată} cu $\\nu$ grade de libertate (varianța 1), $t_\\alpha = t_\\nu^{-1}(\\alpha)$, $f_\\nu$ densitatea lui $t_\\nu$'),
     [T('$q_\\alpha(z) = t_\\alpha\\sqrt{(\\nu - 2)/\\nu}$; \\quad $\\mathrm{ES}_\\alpha(z) = \\sqrt{(\\nu - 2)/\\nu}\\;\\dfrac{f_\\nu(t_\\alpha)}{\\alpha}\\,\\dfrac{\\nu + t_\\alpha^2}{\\nu - 1}$',
        '$q_\\alpha(z) = t_\\alpha\\sqrt{(\\nu - 2)/\\nu}$; \\quad $\\mathrm{ES}_\\alpha(z) = \\sqrt{(\\nu - 2)/\\nu}\\;\\dfrac{f_\\nu(t_\\alpha)}{\\alpha}\\,\\dfrac{\\nu + t_\\alpha^2}{\\nu - 1}$'),
      T('with $X = \\mu + \\sigma z$: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma q_\\alpha(z))$, $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\,\\mathrm{ES}_\\alpha(z)$', 'cu $X = \\mu + \\sigma z$: $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma q_\\alpha(z))$, $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\,\\mathrm{ES}_\\alpha(z)$')]),
    (T('\\textbf{Cornish--Fisher}: $z_{CF} = z + (z^2 - 1)S/6 + (z^3 - 3z)K/24 - (2z^3 - 5z)S^2/36$ ($S$: skewness, $K$: excess kurtosis)',
       '\\textbf{Cornish--Fisher}: $z_{CF} = z + (z^2 - 1)S/6 + (z^3 - 3z)K/24 - (2z^3 - 5z)S^2/36$ ($S$: asimetria, $K$: excesul de boltire)'),
     [T('\\textbf{EVT} (POT, Chapter 5): $\\mathrm{VaR}_\\alpha = u + \\frac{\\beta}{\\xi}[(n\\alpha/N_u)^{-\\xi} - 1]$, GPD fitted to the losses above the threshold $u$',
        '\\textbf{EVT} (POT, Capitolul 5): $\\mathrm{VaR}_\\alpha = u + \\frac{\\beta}{\\xi}[(n\\alpha/N_u)^{-\\xi} - 1]$, distribuția GPD estimată pe pierderile de peste pragul $u$')])))

D.frame(T('What You Need for Today (3/5): Coherence', 'Noțiuni necesare azi (3/5): coerența'), items(
    (T('A risk measure $\\rho$ is \\textbf{coherent} \\refArtzner if it is monotone, translation invariant, positively homogeneous and subadditive',
       'O măsură de risc $\\rho$ este \\textbf{coerentă} \\refArtzner dacă este monotonă, invariantă la translație, pozitiv omogenă și subaditivă'),
     [T('\\textbf{subadditivity}: $\\rho(L_1 + L_2) \\le \\rho(L_1) + \\rho(L_2)$: diversification never adds risk', '\\textbf{subaditivitatea}: $\\rho(L_1 + L_2) \\le \\rho(L_1) + \\rho(L_2)$: diversificarea nu adaugă niciodată risc')]),
    (T('ES is coherent \\refAT; VaR is not always subadditive', 'ES este coerent \\refAT; VaR nu este întotdeauna subaditiv'),
     [T('for a discrete loss $L$ (default or not): $\\mathrm{VaR}_\\alpha$ is the smallest $l$ with $P(L > l) \\le \\alpha$', 'pentru o pierdere discretă $L$ (neplată sau nu): $\\mathrm{VaR}_\\alpha$ este cel mai mic $l$ cu $P(L > l) \\le \\alpha$'),
      T('$\\mathrm{ES}_\\alpha$ = the average loss over the worst $\\alpha$ of the probability mass (a part of an atom counts with its share)', '$\\mathrm{ES}_\\alpha$ = pierderea medie pe cele mai proaste $\\alpha$ din masa de probabilitate (o parte dintr-un atom contează cu ponderea ei)')]),
    T('Two independent events with probabilities $p$: both happen with $p^2$, exactly one with $2p(1 - p)$', 'Două evenimente independente cu probabilitățile $p$: ambele au loc cu $p^2$, exact unul cu $2p(1 - p)$')))

D.frame(T('What You Need for Today (4/5): Backtesting', 'Noțiuni necesare azi (4/5): backtesting'), items(
    (T('\\textbf{Exception}: $r_t < -\\mathrm{VaR}_t$ (the VaR forecast made the day before); $I_t = 1$ on such days, else 0', '\\textbf{Depășire}: $r_t < -\\mathrm{VaR}_t$ (prognoza VaR făcută cu o zi înainte); $I_t = 1$ în astfel de zile, altfel 0'),
     [T('correct VaR 1\\%: the $I_t$ are i.i.d.\\ Bernoulli(0.01); $x$ exceptions in $n$ days: $x \\sim \\mathrm{Binomial}(n, 0.01)$', 'VaR 1\\% corect: $I_t$ sînt i.i.d.\\ Bernoulli(0,01); $x$ depășiri în $n$ zile: $x \\sim \\mathrm{Binomial}(n; 0{,}01)$')]),
    (T('\\textbf{Kupiec} \\refKupiec, $\\hat\\pi = x/n$: $LR_{uc} = -2[\\ln L_0 - \\ln L_1] \\sim \\chi^2(1)$', '\\textbf{Kupiec} \\refKupiec, $\\hat\\pi = x/n$: $LR_{uc} = -2[\\ln L_0 - \\ln L_1] \\sim \\chi^2(1)$'),
     [T('$\\ln L_0 = (n - x)\\ln 0.99 + x\\ln 0.01$, \\quad $\\ln L_1 = (n - x)\\ln(1 - \\hat\\pi) + x\\ln\\hat\\pi$', '$\\ln L_0 = (n - x)\\ln 0.99 + x\\ln 0.01$, \\quad $\\ln L_1 = (n - x)\\ln(1 - \\hat\\pi) + x\\ln\\hat\\pi$')]),
    (T('\\textbf{Christoffersen} \\refChris: $n_{ij}$ = days with $I_{t-1} = i$, $I_t = j$; $\\hat\\pi_{01} = n_{01}/(n_{00} + n_{01})$, $\\hat\\pi_{11} = n_{11}/(n_{10} + n_{11})$',
       '\\textbf{Christoffersen} \\refChris: $n_{ij}$ = zilele cu $I_{t-1} = i$, $I_t = j$; $\\hat\\pi_{01} = n_{01}/(n_{00} + n_{01})$, $\\hat\\pi_{11} = n_{11}/(n_{10} + n_{11})$'),
     [T('$LR_{ind} = -2[\\ln L(\\hat\\pi) - \\ln L(\\hat\\pi_{01}, \\hat\\pi_{11})] \\sim \\chi^2(1)$; $LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)$; critical values at 5\\%: 3.84 and 5.99',
        '$LR_{ind} = -2[\\ln L(\\hat\\pi) - \\ln L(\\hat\\pi_{01}, \\hat\\pi_{11})] \\sim \\chi^2(1)$; $LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)$; valori critice la 5\\%: 3,84 și 5,99')]),
    (T('\\textbf{Basel traffic light} \\refBCBSb, 250 days of VaR 1\\%: green 0--4, yellow 5--9, red 10 or more exceptions', '\\textbf{Semaforul Basel} \\refBCBSb, 250 de zile de VaR 1\\%: verde 0--4, galben 5--9, roșu 10 sau mai multe depășiri'),
     [T('capital multiplier $3 + $ plus factor; plus factors for 5, 6, 7, 8, 9 exceptions: 0.40, 0.50, 0.65, 0.75, 0.85; red: 1', 'factorul de capital $3 + $ adaosul; adaosurile pentru 5, 6, 7, 8, 9 depășiri: 0,40; 0,50; 0,65; 0,75; 0,85; zona roșie: 1')])), 'footnotesize')

D.frame(T('What You Need for Today (5/5): Portfolios and Dependence', 'Noțiuni necesare azi (5/5): portofolii și dependență'), items(
    (T('Weights $w_1, w_2$: $R_p = w_1R_1 + w_2R_2$, $\\sigma_p^2 = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\sigma_1\\sigma_2$', 'Ponderile $w_1, w_2$: $R_p = w_1R_1 + w_2R_2$, $\\sigma_p^2 = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\sigma_1\\sigma_2$'),
     [T('\\textbf{variance--covariance} VaR: $-(\\mu_p + z_\\alpha\\sigma_p)$; diversification benefit: $1 - \\mathrm{VaR}_p/(w_1\\mathrm{VaR}_1 + w_2\\mathrm{VaR}_2)$',
        'VaR \\textbf{varianță--covarianță}: $-(\\mu_p + z_\\alpha\\sigma_p)$; beneficiul diversificării: $1 - \\mathrm{VaR}_p/(w_1\\mathrm{VaR}_1 + w_2\\mathrm{VaR}_2)$')]),
    (T('\\textbf{Tail dependence}: $P(U_1 \\le q \\mid U_2 \\le q)$ for small $q$, with $U_i$ the ranks divided by $n + 1$ (pseudo-observations)',
       '\\textbf{Dependența în cozi}: $P(U_1 \\le q \\mid U_2 \\le q)$ pentru $q$ mic, cu $U_i$ rangurile împărțite la $n + 1$ (pseudo-observații)'),
     [T('equals $q$ under independence; correlation alone does not determine it', 'este egală cu $q$ la independență; corelația singură nu o determină')]),
    (T('\\textbf{Copula}: the joint distribution of $(U_1, U_2)$; Sklar: $F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$', '\\textbf{Copula}: distribuția comună a lui $(U_1, U_2)$; Sklar: $F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$'),
     [T('Gaussian copula: no tail dependence; t copula with $\\nu$: $\\lambda = 2t_{\\nu+1}(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}) > 0$ \\refDM',
        'copula Gaussiană: fără dependență în cozi; copula t cu $\\nu$: $\\lambda = 2t_{\\nu+1}(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}) > 0$ \\refDM'),
      T('$\\rho$ from Kendall\'s $\\tau$: $\\rho = \\sin(\\pi\\tau/2)$', '$\\rho$ din $\\tau$ al lui Kendall: $\\rho = \\sin(\\pi\\tau/2)$')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: VaR and ES of a Normal Position', 'A1: VaR și ES ale unei poziții Normale'),
         items(T('A position of 1\\,000\\,000 RON; daily returns $N(\\mu, \\sigma^2)$ with $\\mu = 0.05\\%$, $\\sigma = 1.4\\%$.',
                 'O poziție de 1\\,000\\,000 de lei; randamente zilnice $N(\\mu, \\sigma^2)$ cu $\\mu = 0{,}05\\%$, $\\sigma = 1{,}4\\%$.'),
               T('1. Compute VaR 1\\% in \\% and in RON.', '1. Calculați VaR 1\\% în \\% și în lei.'),
               T('2. Compute ES 2.5\\% in \\% and in RON.', '2. Calculați ES 2,5\\% în \\% și în lei.'),
               T('3. Compute the 10-day VaR 1\\% with the square-root-of-time rule ($\\mu$ ignored).', '3. Calculați VaR 1\\% pe 10 zile cu regula rădăcinii pătrate a timpului (fără $\\mu$).'),
               T('4. Compute the ratio ES 2.5\\% / VaR 1\\%.', '4. Calculați raportul ES 2,5\\% / VaR 1\\%.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. $-(0.05 - 2.326 \\times 1.4) = -0.05 + @{a1.zs} = @{a1.var1}\\%$, i.e.\\ @{a1.var1c} RON', '1. $-(0.05 - 2.326 \\times 1.4) = -0.05 + @{a1.zs} = @{a1.var1}\\%$, adică @{a1.var1c} lei'),
               T('2. $-0.05 + 1.4 \\times 0.0584/0.025 = -0.05 + @{a1.fs} = @{a1.es25}\\%$, i.e.\\ @{a1.es25c} RON', '2. $-0.05 + 1.4 \\times 0.0584/0.025 = -0.05 + @{a1.fs} = @{a1.es25}\\%$, adică @{a1.es25c} lei'),
               T('3. $\\sqrt{10} \\times 2.326 \\times 1.4 = @{a1.var_h}\\%$, i.e.\\ @{a1.varhc} RON', '3. $\\sqrt{10} \\times 2.326 \\times 1.4 = @{a1.var_h}\\%$, adică @{a1.varhc} lei'),
               T('4. $@{a1.es25}/@{a1.var1} = @{a1.ratio}$', '4. $@{a1.es25}/@{a1.var1} = @{a1.ratio}$'),
               T('For Normal returns ES 2.5\\% and VaR 1\\% are almost the same number.', 'Pentru randamente Normale, ES 2,5\\% și VaR 1\\% sînt practic același număr.')),
         size='scriptsize')

D.proposed(T('A2: The Same Position with Student-t Returns', 'A2: aceeași poziție, cu randamente Student-t'),
           items(T('As in A1, but the standardised returns are Student-t with $\\nu = 4$: $X = \\mu + \\sigma z$; $t_4^{-1}(0.01) = -3.747$, $t_4^{-1}(0.025) = -2.776$, $f_4(-2.776) = 0.0256$. Model: A1.',
                   'Ca în A1, dar randamentele standardizate sînt Student-t cu $\\nu = 4$: $X = \\mu + \\sigma z$; $t_4^{-1}(0{,}01) = -3{,}747$, $t_4^{-1}(0{,}025) = -2{,}776$, $f_4(-2{,}776) = 0{,}0256$. Model: A1.'),
                 T('1. Compute $q_{0.01}(z)$ and VaR 1\\% in \\% and in RON.', '1. Calculați $q_{0{,}01}(z)$ și VaR 1\\% în \\% și în lei.'),
                 T('2. Compute $\\mathrm{ES}_{2.5\\%}(z)$ and ES 2.5\\% in \\% and in RON.', '2. Calculați $\\mathrm{ES}_{2,5\\%}(z)$ și ES 2,5\\% în \\% și în lei.'),
                 T('3. Compare both numbers with A1.', '3. Comparați ambele valori cu A1.'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $q = -3.747 \\times \\sqrt{2/4} = @{a2.q}$; VaR 1\\% $= -(0.05 + 1.4 \\times (@{a2.q})) = @{a2.var1r}\\%$ (@{a2.var1c} RON)', '1. $q = -3.747 \\times \\sqrt{2/4} = @{a2.q}$; VaR 1\\% $= -(0.05 + 1.4 \\times (@{a2.q})) = @{a2.var1r}\\%$ (@{a2.var1c} lei)'),
                 T('2. $\\sqrt{0.5} \\times (0.0256/0.025) \\times (4 + 2.776^2)/3 = @{a2.ez}$; ES 2.5\\% $= -0.05 + 1.4 \\times @{a2.ez} = @{a2.es25r}\\%$ (@{a2.es25c} RON)',
                   '2. $\\sqrt{0.5} \\times (0.0256/0.025) \\times (4 + 2.776^2)/3 = @{a2.ez}$; ES 2,5\\% $= -0.05 + 1.4 \\times @{a2.ez} = @{a2.es25r}\\%$ (@{a2.es25c} lei)'),
                 T('3. VaR 1\\% is @{a2.inc}\\% higher and ES 2.5\\% @{a2.ince}\\% higher than under the Normal; ES/VaR $= @{a2.ratio}$.', '3. VaR 1\\% este cu @{a2.inc}\\% mai mare, iar ES 2,5\\% cu @{a2.ince}\\% mai mare decît pentru distribuția Normală; ES/VaR $= @{a2.ratio}$.'),
                 T('With the same variance, heavy tails raise ES more than VaR.', 'La aceeași varianță, cozile groase cresc ES mai mult decît VaR.')),
           size='scriptsize')

D.solved(T('A3: Historical Simulation from the Worst Days of the BET', 'A3: simularea istorică din cele mai proaste zile ale BET'),
         items(T('The last 500 daily BET returns (@{a3.first} -- @{a3.last}); the 15 smallest, in \\%: @{a3.list1};', 'Ultimele 500 de randamente zilnice ale BET (@{a3.first} -- @{a3.last}); cele mai mici 15, în \\%: @{a3.list1};'),
               T('@{a3.list2}.', '@{a3.list2}.'),
               T('1. Compute VaR 1\\% and VaR 2.5\\% by historical simulation.', '1. Calculați VaR 1\\% și VaR 2,5\\% prin simulare istorică.'),
               T('2. Compute ES 2.5\\%.', '2. Calculați ES 2,5\\%.'),
               T('3. Translate VaR 1\\% and ES 2.5\\% into RON for a position of 1\\,000\\,000 RON.', '3. Exprimați VaR 1\\% și ES 2,5\\% în lei, pentru o poziție de 1\\,000\\,000 de lei.'),
               T('Report: five numbers and one sentence.', 'Raportați: cinci valori și o frază.')),
         items(T('1. $k = \\lceil 500 \\times 0.01 \\rceil = 5$: VaR 1\\% $= @{a3.var1}\\%$; $k = \\lceil 12.5 \\rceil = 13$: VaR 2.5\\% $= @{a3.var25}\\%$', '1. $k = \\lceil 500 \\times 0.01 \\rceil = 5$: VaR 1\\% $= @{a3.var1}\\%$; $k = \\lceil 12.5 \\rceil = 13$: VaR 2,5\\% $= @{a3.var25}\\%$'),
               T('2. ES 2.5\\% $= @{a3.sum13}/13 = @{a3.es25}\\%$ (the average of the 13 smallest)', '2. ES 2,5\\% $= @{a3.sum13}/13 = @{a3.es25}\\%$ (media celor mai mici 13)'),
               T('3. VaR 1\\%: @{a3.var1c} RON; ES 2.5\\%: @{a3.es25c} RON', '3. VaR 1\\%: @{a3.var1c} lei; ES 2,5\\%: @{a3.es25c} lei'),
               T('Here ES 2.5\\% is below VaR 1\\%: ES is larger than VaR only at the same level; ES 2.5\\% $\\ge$ VaR 2.5\\% holds.', 'Aici ES 2,5\\% este sub VaR 1\\%: ES este mai mare decît VaR doar la același nivel; ES 2,5\\% $\\ge$ VaR 2,5\\% este îndeplinită.')),
         size='scriptsize')

D.proposed(T('A4: Two Bonds and Subadditivity', 'A4: două obligațiuni și subaditivitatea'),
           items(T('Two independent bonds; each loses 100 with probability 3\\% and 0 otherwise; level 5\\%. Model: A3 and slide 3/5.', 'Două obligațiuni independente; fiecare pierde 100 cu probabilitatea 3\\% și 0 în rest; nivelul 5\\%. Model: A3 și slide-ul 3/5.'),
                 T('1. Compute VaR 5\\% and ES 5\\% of one bond.', '1. Calculați VaR 5\\% și ES 5\\% pentru o obligațiune.'),
                 T('2. Write the distribution of the total loss of the two bonds.', '2. Scrieți distribuția pierderii totale a celor două obligațiuni.'),
                 T('3. Compute VaR 5\\% and ES 5\\% of the total loss.', '3. Calculați VaR 5\\% și ES 5\\% ale pierderii totale.'),
                 T('4. Check subadditivity for VaR and for ES.', '4. Verificați subaditivitatea pentru VaR și pentru ES.'),
                 T('Report: four numbers and two sentences.', 'Raportați: patru valori și două fraze.')),
           items(T('1. $P(L > 0) = 3\\% \\le 5\\%$: VaR 5\\% $= 0$; ES 5\\% $= (3\\% \\times 100 + 2\\% \\times 0)/5\\% = @{a4.es1}$', '1. $P(L > 0) = 3\\% \\le 5\\%$: VaR 5\\% $= 0$; ES 5\\% $= (3\\% \\times 100 + 2\\% \\times 0)/5\\% = @{a4.es1}$'),
                 T('2. $P(200) = @{a4.p_both}\\%$, $P(100) = @{a4.p_one}\\%$, $P(0) = 100 - @{a4.p_any}\\%$', '2. $P(200) = @{a4.p_both}\\%$, $P(100) = @{a4.p_one}\\%$, $P(0) = 100 - @{a4.p_any}\\%$'),
                 T('3. $P(L > 0) = @{a4.p_any}\\% > 5\\%$: VaR 5\\% $= @{a4.var2}$; ES 5\\% $= (@{a4.p_both}\\% \\times 200 + (5 - @{a4.p_both})\\% \\times 100)/5\\% = @{a4.es2}$', '3. $P(L > 0) = @{a4.p_any}\\% > 5\\%$: VaR 5\\% $= @{a4.var2}$; ES 5\\% $= (@{a4.p_both}\\% \\times 200 + (5 - @{a4.p_both})\\% \\times 100)/5\\% = @{a4.es2}$'),
                 T('4. VaR: $@{a4.var2} > 0 + 0$, not subadditive; ES: $@{a4.es2} \\le @{a4.es1} + @{a4.es1}$, subadditive.', '4. VaR: $@{a4.var2} > 0 + 0$, nu este subaditiv; ES: $@{a4.es2} \\le @{a4.es1} + @{a4.es1}$, subaditiv.')),
           size='scriptsize')

D.solved(T('A5: The Kupiec Test and the Traffic Light', 'A5: testul Kupiec și semaforul Basel'),
         items(T('A bank\'s VaR 1\\% was exceeded on 7 of the last 250 days. A second model was exceeded on 16 of 1000 days.', 'VaR 1\\% al unei bănci a fost depășit în 7 din ultimele 250 de zile. Un al doilea model a fost depășit în 16 din 1000 de zile.'),
               T('1. Compute $\\ln L_0$, $\\ln L_1$ and $LR_{uc}$ for the bank, and decide at 5\\%.', '1. Calculați $\\ln L_0$, $\\ln L_1$ și $LR_{uc}$ pentru bancă și decideți la 5\\%.'),
               T('2. Give the Basel zone, the plus factor and the capital multiplier.', '2. Precizați zona Basel, adaosul și factorul de capital.'),
               T('3. Compute $LR_{uc}$ for the second model and decide at 5\\%.', '3. Calculați $LR_{uc}$ pentru al doilea model și decideți la 5\\%.'),
               T('Report: five numbers, two decisions and one sentence.', 'Raportați: cinci valori, două decizii și o frază.')),
         items(T('1. $243\\ln 0.99 + 7\\ln 0.01 = @{a5.l0}$; $243\\ln 0.972 + 7\\ln 0.028 = @{a5.l1}$; $LR_{uc} = @{a5.lr} > 3.84$ (p $= @{a5.p}$): rejected', '1. $243\\ln 0.99 + 7\\ln 0.01 = @{a5.l0}$; $243\\ln 0.972 + 7\\ln 0.028 = @{a5.l1}$; $LR_{uc} = @{a5.lr} > 3.84$ (p $= @{a5.p}$): respins'),
               T('2. Yellow zone ($P(X \\le 6) = @{a5.cump}\\%$, $P(X \\le 7) = @{a5.cum}\\%$); plus 0.65, multiplier 3.65', '2. Zona galbenă ($P(X \\le 6) = @{a5.cump}\\%$, $P(X \\le 7) = @{a5.cum}\\%$); adaosul 0,65, factorul 3,65'),
               T('3. $984\\ln 0.99 + 16\\ln 0.01 = @{a5b.l0}$; $984\\ln 0.984 + 16\\ln 0.016 = @{a5b.l1}$; $LR_{uc} = @{a5b.lr}$ (p $= @{a5b.p}$): not rejected', '3. $984\\ln 0.99 + 16\\ln 0.01 = @{a5b.l0}$; $984\\ln 0.984 + 16\\ln 0.016 = @{a5b.l1}$; $LR_{uc} = @{a5b.lr}$ (p $= @{a5b.p}$): nerespins'),
               T('A rate of 1.6\\% is not rejected with 1000 days; 2.8\\% is rejected with 250: the test needs a large deviation or many days.', 'O rată de 1,6\\% nu este respinsă cu 1000 de zile; 2,8\\% este respinsă cu 250: testul are nevoie de o abatere mare sau de multe zile.')),
         size='scriptsize')

D.proposed(T('A6: The Christoffersen Test', 'A6: testul Christoffersen'),
           items(T('1000 days of VaR 1\\% forecasts give the transition counts $n_{00} = 978$, $n_{01} = 8$, $n_{10} = 8$, $n_{11} = 5$. Model: A5.', '1000 de zile de prognoze VaR 1\\% dau numerele de tranziții $n_{00} = 978$, $n_{01} = 8$, $n_{10} = 8$, $n_{11} = 5$. Model: A5.'),
                 T('1. Compute $\\hat\\pi_{01}$, $\\hat\\pi_{11}$ and $\\hat\\pi$.', '1. Calculați $\\hat\\pi_{01}$, $\\hat\\pi_{11}$ și $\\hat\\pi$.'),
                 T('2. Compute $LR_{ind}$ and decide at 5\\%.', '2. Calculați $LR_{ind}$ și decideți la 5\\%.'),
                 T('3. Compute $LR_{uc}$ (with $x = 13$, $n = 999$) and $LR_{cc}$, and decide at 5\\%.', '3. Calculați $LR_{uc}$ (cu $x = 13$, $n = 999$) și $LR_{cc}$ și decideți la 5\\%.'),
                 T('Report: six numbers, three decisions and one sentence.', 'Raportați: șase valori, trei decizii și o frază.')),
           items(T('1. $\\hat\\pi_{01} = 8/986 = @{a6.p01}$; $\\hat\\pi_{11} = 5/13 = @{a6.p11}$; $\\hat\\pi = 13/999 = @{a6.p}$', '1. $\\hat\\pi_{01} = 8/986 = @{a6.p01}$; $\\hat\\pi_{11} = 5/13 = @{a6.p11}$; $\\hat\\pi = 13/999 = @{a6.p}$'),
                 T('2. $\\ln L(\\hat\\pi) = @{a6.ll0}$, $\\ln L(\\hat\\pi_{01}, \\hat\\pi_{11}) = @{a6.ll1}$; $LR_{ind} = @{a6.lr_ind} > 3.84$ (p @{a6.p_ind}): rejected', '2. $\\ln L(\\hat\\pi) = @{a6.ll0}$, $\\ln L(\\hat\\pi_{01}, \\hat\\pi_{11}) = @{a6.ll1}$; $LR_{ind} = @{a6.lr_ind} > 3.84$ (p @{a6.p_ind}): respins'),
                 T('3. $LR_{uc} = @{a6.lr_uc}$ (p @{a6.p_uc}): not rejected; $LR_{cc} = @{a6.lr_cc} > 5.99$ (p @{a6.p_cc}): rejected', '3. $LR_{uc} = @{a6.lr_uc}$ (p @{a6.p_uc}): nerespins; $LR_{cc} = @{a6.lr_cc} > 5.99$ (p @{a6.p_cc}): respins'),
                 T('The number of exceptions is right, but they come in clusters: the model reacts too slowly.', 'Numărul de depășiri este corect, dar ele apar grupat: modelul reacționează prea lent.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Five Methods for the BET [Solved]', 'B1: cinci metode pentru BET [Rezolvat]'),
       T('how much do VaR 1\\% and ES 2.5\\% of the BET depend on the estimation method?', 'cît de mult depind VaR 1\\% și ES 2,5\\% ale BET de metoda de estimare?'),
       T('BET closes since 2000, daily log returns in \\%', 'închiderile BET din 2000, randamente logaritmice zilnice în \\%'),
       [T('Compute VaR 1\\% and ES 2.5\\% by historical simulation, Normal and Student-t (maximum likelihood).', 'Calculați VaR 1\\% și ES 2,5\\% prin simulare istorică, cu distribuția Normală și cu Student-t (verosimilitate maximă).'),
        T('Compute the skewness and the excess kurtosis, and the Cornish--Fisher VaR 1\\%.', 'Calculați asimetria și excesul de boltire, precum și VaR 1\\% Cornish--Fisher.'),
        T('Fit a GPD to the losses above their 90\\% quantile and compute the EVT VaR 1\\% and ES 2.5\\%.', 'Estimați o distribuție GPD pe pierderile de peste cuantila lor de 90\\% și calculați VaR 1\\% și ES 2,5\\% prin EVT.'),
        T('Draw the left tail with the five values of minus VaR 1\\%.', 'Desenați coada stîngă cu cele cinci valori ale lui minus VaR 1\\%.'),
        T('Interpretation: which number would you report to a risk committee?', 'Interpretare: ce valoare ați raporta unui comitet de risc?')],
       T('a table of ten numbers, the chart and two sentences', 'un tabel cu zece valori, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch10_sem_b1', h='0.30') + table(
    'lrrrrr', T('& HS & Normal & Student-t & Cornish--Fisher & EVT', '& HS & Normală & Student-t & Cornish--Fisher & EVT'),
    ['VaR 1\\% & $@{b1.hs.v}$ & $@{b1.n.v}$ & $@{b1.t.v}$ & $@{b1.cf.v}$ & $@{b1.evt.v}$',
     'ES 2.5\\% & $@{b1.hs.e}$ & $@{b1.n.e}$ & $@{b1.t.e}$ & $@{b1.cf.e}$ & $@{b1.evt.e}$'], size='scriptsize') + items(
    T('$n = @{b1.n}$; $\\hat\\mu = @{b1.mu}$, $\\hat\\sigma = @{b1.sd}$; $S = @{b1.skew}$, $K = @{b1.k}$; $\\hat\\nu = @{b1.nu}$; EVT: $u = @{b1.u}\\%$, $N_u = @{b1.Nu}$, $\\hat\\xi = @{b1.xi}$',
      '$n = @{b1.n}$; $\\hat\\mu = @{b1.mu}$, $\\hat\\sigma = @{b1.sd}$; $S = @{b1.skew}$, $K = @{b1.k}$; $\\hat\\nu = @{b1.nu}$; EVT: $u = @{b1.u}\\%$, $N_u = @{b1.Nu}$, $\\hat\\xi = @{b1.xi}$'),
    T('The Normal VaR 1\\% is @{b1.gap}\\% below the data; Cornish--Fisher ($z_{CF} = @{b1.cfz}$) is far above all others', 'VaR 1\\% Normal este cu @{b1.gap}\\% sub date; Cornish--Fisher ($z_{CF} = @{b1.cfz}$) este mult peste toate celelalte'),
    T('Interpretation: report the historical or the EVT value (about @{b1.hs.v}\\% and @{b1.hs.e}\\%), which agree; mention that the Normal understates and Cornish--Fisher is outside its range with $K > 10$',
      'Interpretare: raportăm valoarea istorică sau pe cea EVT (circa @{b1.hs.v}\\% și @{b1.hs.e}\\%), care sînt de acord; menționăm că distribuția Normală subestimează, iar Cornish--Fisher este în afara domeniului de valabilitate cu $K > 10$')) + qlsem(), 'scriptsize')

D.task(T('B2: TLV, SNP and Bitcoin [Proposed]', 'B2: TLV, SNP și Bitcoin [Propus]'),
       T('does the ranking of the methods found for the BET hold for two BVB stocks and for Bitcoin? Model: B1.', 'se păstrează ordinea metodelor găsită pentru BET la două acțiuni BVB și la Bitcoin? Model: B1.'),
       T('TLV and SNP since 2010 (TLV without 30--31 May 2016), Bitcoin since 2014; daily log returns in \\%', 'TLV și SNP din 2010 (TLV fără 30--31 mai 2016), Bitcoin din 2014; randamente logaritmice zilnice în \\%'),
       [T('Compute VaR 1\\% and ES 2.5\\% of each series by the five methods of B1.', 'Calculați VaR 1\\% și ES 2,5\\% pentru fiecare serie, cu cele cinci metode din B1.'),
        T('Compute by how much the Normal VaR 1\\% is below the historical VaR 1\\%.', 'Calculați cu cît este VaR 1\\% Normal sub VaR 1\\% istoric.'),
        T('Report $\\hat\\nu$, $\\hat\\xi$ and the excess kurtosis of each series.', 'Raportați $\\hat\\nu$, $\\hat\\xi$ și excesul de boltire pentru fiecare serie.'),
        T('Interpretation: for which series does the Normal understate VaR 1\\% most?', 'Interpretare: pentru care serie subestimează cel mai mult distribuția Normală VaR 1\\%?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{SHORT[k]} & ' + ' & '.join(f'$@{{b2.{k}.{m}.v}}$' for m in ['hs', 'n', 't', 'cf', 'evt']) + ' & ' +
            ' & '.join(f'$@{{b2.{k}.{m}.e}}$' for m in ['hs', 'n', 'evt']) + f' & @{{b2.{k}.gap}}\\% & ${{@{{b2.{k}.k}}}}$')


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'l|rrrrr|rrr|rr', T('& \\multicolumn{5}{c|}{VaR 1\\%} & \\multicolumn{3}{c|}{ES 2.5\\%} & & \\\\ & HS & N & t & CF & EVT & HS & N & EVT & gap N & $K$',
                        '& \\multicolumn{5}{c|}{VaR 1\\%} & \\multicolumn{3}{c|}{ES 2,5\\%} & & \\\\ & HS & N & t & CF & EVT & HS & N & EVT & abatere N & $K$'),
    [b2row(k) for k in ['tlv', 'snp', 'btc']], size='scriptsize') + items(
    T('N: Normal, t: Student-t, CF: Cornish--Fisher; gap N: how much the Normal VaR 1\\% is below HS', 'N: Normală, t: Student-t, CF: Cornish--Fisher; abatere N: cu cît este VaR 1\\% Normal sub HS'),
    T('The ranking of B1 holds: Normal too low, CF far too high, HS and EVT close; $\\hat\\nu$: TLV $@{b2.tlv.nu}$, SNP $@{b2.snp.nu}$, Bitcoin $@{b2.btc.nu}$; $\\hat\\xi$: $@{b2.tlv.xi}$, $@{b2.snp.xi}$, $@{b2.btc.xi}$',
      'Ordinea din B1 se păstrează: distribuția Normală dă valori prea mici, CF mult prea mari, HS și EVT sînt apropiate; $\\hat\\nu$: TLV $@{b2.tlv.nu}$, SNP $@{b2.snp.nu}$, Bitcoin $@{b2.btc.nu}$; $\\hat\\xi$: $@{b2.tlv.xi}$, $@{b2.snp.xi}$, $@{b2.btc.xi}$'),
    T('Interpretation: Bitcoin (@{b2.btc.gap}\\%) and TLV (@{b2.tlv.gap}\\%): the heaviest tails relative to the variance; SNP has the smallest gap',
      'Interpretare: Bitcoin (@{b2.btc.gap}\\%) și TLV (@{b2.tlv.gap}\\%): cele mai groase cozi în raport cu varianța; SNP are cea mai mică abatere')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Backtesting the S\\&P 500 Year by Year [Solved]', 'B3: backtesting pentru S\\&P 500, an cu an [Rezolvat]'),
       T('which of two VaR 1\\% models would a supervisor have penalised, year by year, since 2015?', 'pe care dintre două modele VaR 1\\% l-ar fi penalizat o autoritate de supraveghere, an cu an, din 2015?'),
       T('S\\&P 500 since 2000; forecasts for every day since 2 January 2015', 'S\\&P 500 din 2000; prognoze pentru fiecare zi de la 2 ianuarie 2015'),
       [T('Compute the one-day VaR 1\\% by historical simulation on the last 250 days.', 'Calculați VaR 1\\% pe o zi prin simulare istorică pe ultimele 250 de zile.'),
        T('Compute the one-day GARCH(1,1)-t VaR 1\\%, re-estimating the model at the start of every year on all earlier data.', 'Calculați VaR 1\\% pe o zi din GARCH(1,1)-t, reestimînd modelul la începutul fiecărui an pe toate datele anterioare.'),
        T('Count the exceptions of each model in every calendar year and give the Basel zone of each year.', 'Numărați depășirile fiecărui model în fiecare an calendaristic și precizați zona Basel a fiecărui an.'),
        T('Run the Kupiec and Christoffersen tests on the whole period.', 'Aplicați testele Kupiec și Christoffersen pe întreaga perioadă.'),
        T('Interpretation: which model would a supervisor have penalised in 2022?', 'Interpretare: ce model ar fi penalizat o autoritate de supraveghere în 2022?')],
       T('the chart, six statistics with p-values and two sentences', 'graficul, șase statistici cu p-valori și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch10_sem_b3', h='0.34') + table(
    'lrrrrrrr', T('& $n$ & $x$ & rate (\\%) & $LR_{uc}$ (p) & $LR_{ind}$ (p) & $LR_{cc}$ (p) & red/yellow years', '& $n$ & $x$ & rata (\\%) & $LR_{uc}$ (p) & $LR_{ind}$ (p) & $LR_{cc}$ (p) & ani roșii/galbeni'),
    [f'HS 250 & @{{b3.hs.n}} & @{{b3.hs.x}} & $@{{b3.hs.rate}}$ & $@{{b3.hs.luc}}$ (@{{b3.hs.puc}}) & $@{{b3.hs.lind}}$ (@{{b3.hs.pind}}) & $@{{b3.hs.lcc}}$ (@{{b3.hs.pcc}}) & @{{b3.hs.red}}/@{{b3.hs.yel}}',
     f'GARCH-t & @{{b3.g.n}} & @{{b3.g.x}} & $@{{b3.g.rate}}$ & $@{{b3.g.luc}}$ (@{{b3.g.puc}}) & $@{{b3.g.lind}}$ (@{{b3.g.pind}}) & $@{{b3.g.lcc}}$ (@{{b3.g.pcc}}) & @{{b3.g.red}}/@{{b3.g.yel}}'],
    size='scriptsize') + items(
    T('Both models are exceeded too often (expected about @{b3.hs.exp}); HS exceptions cluster strongly ($n_{11} = @{b3.hs.n11}$, p ind.\\ @{b3.hs.pind}), GARCH-t less ($n_{11} = @{b3.g.n11}$)',
      'Ambele modele sînt depășite prea des (așteptat circa @{b3.hs.exp}); depășirile HS apar puternic grupat ($n_{11} = @{b3.hs.n11}$, p ind.\\ @{b3.hs.pind}), cele GARCH-t mai puțin ($n_{11} = @{b3.g.n11}$)'),
    T('Interpretation: in 2022 HS had @{b3.y2022.hs} exceptions (red zone), GARCH-t @{b3.y2022.g}: the slow window was penalised when volatility rose; GARCH-t paid in calmer years (2021: @{b3.y2021.g})',
      'Interpretare: în 2022, HS a avut @{b3.y2022.hs} depășiri (zona roșie), GARCH-t @{b3.y2022.g}: fereastra lentă a fost penalizată cînd volatilitatea a crescut; GARCH-t a fost penalizat în anii mai liniștiți (2021: @{b3.y2021.g})')) + qlsem(), 'scriptsize')

D.task(T('B4: Backtesting the BET and Bitcoin [Proposed]', 'B4: backtesting pentru BET și Bitcoin [Propus]'),
       T('does the conclusion of B3 hold on the BVB and for Bitcoin? Model: B3.', 'se păstrează concluzia din B3 la BVB și pentru Bitcoin? Model: B3.'),
       T('BET since 2000, forecasts since 2015; Bitcoin since 2014, forecasts since 2016', 'BET din 2000, prognoze din 2015; Bitcoin din 2014, prognoze din 2016'),
       [T('Compute both VaR 1\\% forecasts as in B3.', 'Calculați ambele prognoze VaR 1\\% ca în B3.'),
        T('Count the exceptions per calendar year and give the zones.', 'Numărați depășirile pe ani calendaristici și precizați zonele.'),
        T('Run the Kupiec and Christoffersen tests on the whole period.', 'Aplicați testele Kupiec și Christoffersen pe întreaga perioadă.'),
        T('Interpretation: which model would you keep for each series?', 'Interpretare: ce model ați păstra pentru fiecare serie?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(k, m, mk):
    key = f'b4.{k}.{mk}'
    return (f'{SHORT[k] if mk == "hs" else ""} & {m} & @{{{key}.n}} & @{{{key}.x}} & $@{{{key}.rate}}$ & @{{{key}.puc}} & @{{{key}.pind}} & @{{{key}.pcc}} & @{{{key}.red}}/@{{{key}.yel}}')


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'llrrrrrrr', T('& & $n$ & $x$ & rate (\\%) & p POF & p ind. & p c.c. & red/yellow years', '& & $n$ & $x$ & rata (\\%) & p POF & p ind. & p ac.c. & ani roșii/galbeni'),
    [b4row('bet', 'HS 250', 'hs'), b4row('bet', 'GARCH-t', 'g'), b4row('btc', 'HS 250', 'hs'), b4row('btc', 'GARCH-t', 'g')], size='scriptsize') + items(
    T('BET: GARCH-t passes all three tests (rate @{b4.bet.g.rate}\\%); HS is exceeded too often and its exceptions cluster', 'BET: GARCH-t trece toate cele trei teste (rata @{b4.bet.g.rate}\\%); HS este depășit prea des, iar depășirile lui apar grupat'),
    T('Bitcoin: HS passes (rate @{b4.btc.hs.rate}\\%); GARCH-t is borderline (@{b4.btc.g.rate}\\%, p POF @{b4.btc.g.puc}) with @{b4.btc.g.yel} yellow years', 'Bitcoin: HS trece testele (rata @{b4.btc.hs.rate}\\%); GARCH-t este la limită (@{b4.btc.g.rate}\\%, p POF @{b4.btc.g.puc}), cu @{b4.btc.g.yel} ani în zona galbenă'),
    T('Interpretation: keep GARCH-t for the BET and HS (or FHS) for Bitcoin: the best method depends on the market, so every model needs its own backtest',
      'Interpretare: păstrăm GARCH-t pentru BET și HS (sau FHS) pentru Bitcoin: cea mai bună metodă depinde de piață, deci fiecare model are nevoie de propriul backtesting')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: A Portfolio of Two BVB Stocks [Solved]', 'B5: un portofoliu cu două acțiuni BVB [Rezolvat]'),
       T('how much does diversification between Banca Transilvania and OMV Petrom reduce VaR 1\\%, and does it survive a crisis?', 'cît reduce diversificarea între Banca Transilvania și OMV Petrom VaR 1\\% și rezistă ea într-o criză?'),
       T('TLV and SNP adjusted closes since 2010, joined on common days; simple returns in \\%; weights 50/50', 'închiderile ajustate TLV și SNP din 2010, aliniate pe zilele comune; randamente simple în \\%; ponderi 50/50'),
       [T('Compute the variance--covariance VaR 1\\% of the portfolio, the two individual VaRs and the diversification benefit.', 'Calculați VaR 1\\% varianță--covarianță al portofoliului, cele două VaR individuale și beneficiul diversificării.'),
        T('Compute the historical VaR 1\\% and ES 2.5\\% of the portfolio.', 'Calculați VaR 1\\% și ES 2,5\\% istorice ale portofoliului.'),
        T('Compute the correlation in 2017 and in February--May 2020, and the tail dependence at $q = 5\\%$ and $q = 1\\%$.', 'Calculați corelația din 2017 și din februarie--mai 2020, precum și dependența în cozi la $q = 5\\%$ și $q = 1\\%$.'),
        T('Fit a t copula and compute the copula VaR 1\\% with empirical margins.', 'Estimați o copulă t și calculați VaR 1\\% prin copulă, cu marginale empirice.'),
        T('Interpretation: why does the variance--covariance VaR understate the risk of this portfolio?', 'Interpretare: de ce subestimează VaR varianță--covarianță riscul acestui portofoliu?')],
       T('eight numbers, the chart and two sentences', 'opt valori, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch10_sem_b5', h='0.32') + items(
    T('$n = @{b5.n}$; $\\hat\\sigma$: TLV $@{b5.sd1}$, SNP $@{b5.sd2}$; $\\hat\\rho = @{b5.corr}$; $\\sigma_p = @{b5.sp}$; VaR 1\\%: TLV $@{b5.v1}$, SNP $@{b5.v2}$, portfolio $@{b5.var_n}\\%$; benefit @{b5.ben}\\%',
      '$n = @{b5.n}$; $\\hat\\sigma$: TLV $@{b5.sd1}$, SNP $@{b5.sd2}$; $\\hat\\rho = @{b5.corr}$; $\\sigma_p = @{b5.sp}$; VaR 1\\%: TLV $@{b5.v1}$, SNP $@{b5.v2}$, portofoliu $@{b5.var_n}\\%$; beneficiu @{b5.ben}\\%'),
    T('Historical: VaR 1\\% $= @{b5.var_hs}\\%$, ES 2.5\\% $= @{b5.es_hs}\\%$; t copula ($\\hat\\nu = @{b5.nu}$, $\\lambda = @{b5.lam}$): VaR 1\\% $= @{b5.var_t}\\%$',
      'Istoric: VaR 1\\% $= @{b5.var_hs}\\%$, ES 2,5\\% $= @{b5.es_hs}\\%$; copula t ($\\hat\\nu = @{b5.nu}$, $\\lambda = @{b5.lam}$): VaR 1\\% $= @{b5.var_t}\\%$'),
    T('Correlation: 2017 $@{b5.c17}$, spring 2020 $@{b5.c20}$; tail dependence $@{b5.td5}$ ($q = 5\\%$), $@{b5.td1}$ ($q = 1\\%$); joint worst-1\\% days: @{b5.nboth}, against @{b5.exp} under independence',
      'Corelația: 2017 $@{b5.c17}$, primăvara 2020 $@{b5.c20}$; dependența în cozi $@{b5.td5}$ ($q = 5\\%$), $@{b5.td1}$ ($q = 1\\%$); zile comune în cel mai prost 1\\%: @{b5.nboth}, față de @{b5.exp} la independență'),
    T('Interpretation: the Normal VaR is @{b5.under}\\% below the historical one: both margins have heavy tails and the two stocks fall together on the worst days, when the correlation is highest',
      'Interpretare: VaR Normal este cu @{b5.under}\\% sub cel istoric: ambele marginale au cozi groase, iar cele două acțiuni scad împreună în zilele cele mai proaste, cînd corelația este maximă')) + qlsem(), 'scriptsize')

D.task(T('B6: DAX and S\\&P 500 [Proposed]', 'B6: DAX și S\\&P 500 [Propus]'),
       T('is the Gaussian copula good enough for a 50/50 DAX and S\\&P 500 portfolio? Model: B5.', 'este copula Gaussiană suficient de bună pentru un portofoliu 50/50 DAX și S\\&P 500? Model: B5.'),
       T('DAX and S\\&P 500 closes since 2000, joined on common days; simple returns in \\%', 'închiderile DAX și S\\&P 500 din 2000, aliniate pe zilele comune; randamente simple în \\%'),
       [T('Compute the variance--covariance and the historical VaR 1\\% and ES 2.5\\%.', 'Calculați VaR 1\\% și ES 2,5\\% prin metoda varianță--covarianță și prin simulare istorică.'),
        T('Fit the Gaussian and the t copula and compare them with the LR statistic.', 'Estimați copula Gaussiană și copula t și comparați-le prin statistica LR.'),
        T('Compute the copula VaR 1\\% and ES 2.5\\% with empirical margins for both copulas.', 'Calculați VaR 1\\% și ES 2,5\\% prin copule, cu marginale empirice, pentru ambele copule.'),
        T('Interpretation: is the Gaussian copula good enough for this pair?', 'Interpretare: este copula Gaussiană suficient de bună pentru această pereche?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrr', T('Method & VaR 1\\% & ES 2.5\\%', 'Metoda & VaR 1\\% & ES 2,5\\%'),
    [T('variance--covariance', 'varianță--covarianță') + ' & $@{b6.var_n}$ & $@{b6.es_n}$',
     T('Gaussian copula', 'copula Gaussiană') + ' & $@{b6.var_gauss}$ & $@{b6.es_gauss}$',
     T('t copula', 'copula t') + ' & $@{b6.var_t}$ & $@{b6.es_t}$',
     T('historical simulation', 'simulare istorică') + ' & $@{b6.var_hs}$ & $@{b6.es_hs}$'], size='footnotesize') + items(
    T('$n = @{b6.n}$; $\\hat\\rho = @{b6.corr}$, $\\hat\\tau = @{b6.tau}$, copula $\\rho = @{b6.rho}$; t copula $\\hat\\nu = @{b6.nu}$, LR $= @{b6.lr}$ against the Gaussian copula, $\\lambda = @{b6.lam}$',
      '$n = @{b6.n}$; $\\hat\\rho = @{b6.corr}$, $\\hat\\tau = @{b6.tau}$, $\\rho$ al copulei $= @{b6.rho}$; copula t $\\hat\\nu = @{b6.nu}$, LR $= @{b6.lr}$ față de copula Gaussiană, $\\lambda = @{b6.lam}$'),
    T('The t copula reproduces the historical VaR 1\\% almost exactly; the Gaussian copula is lower, the variance--covariance VaR lower still', 'Copula t reproduce aproape exact VaR 1\\% istoric; copula Gaussiană dă o valoare mai mică, iar VaR varianță--covarianță una și mai mică'),
    T('Interpretation: no: two developed markets that trade at overlapping hours have strong tail dependence ($\\lambda = @{b6.lam}$), which only the t copula captures',
      'Interpretare: nu: două piețe dezvoltate cu ore de tranzacționare suprapuse au o dependență în cozi puternică ($\\lambda = @{b6.lam}$), pe care doar copula t o surprinde')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Is ES 2.5\\% as Large as VaR 1\\%? [Proposed]', 'C1: este ES 2,5\\% la fel de mare ca VaR 1\\%? [Propus]'),
       T('Basel replaced VaR 1\\% by ES 2.5\\% because the two are equal for Normal returns; how close are they on real data and in crises?',
         'Basel a înlocuit VaR 1\\% cu ES 2,5\\% deoarece cele două sînt egale pentru randamente Normale; cît de apropiate sînt pe date reale și în crize?'),
       T('S\\&P 500, DAX, BET, Bitcoin, TLV, SNP, BRD; models: B1, A5', 'S\\&P 500, DAX, BET, Bitcoin, TLV, SNP, BRD; modele: B1, A5'),
       [T('Compute the ratio ES 2.5\\% / VaR 1\\% by historical simulation on the whole sample of each series.', 'Calculați raportul ES 2,5\\% / VaR 1\\% prin simulare istorică pe eșantionul complet al fiecărei serii.'),
        T('Compute the same ratio on the 500 days that end in March 2009 and in May 2020.', 'Calculați același raport pe cele 500 de zile care se încheie în martie 2009 și în mai 2020.'),
        T('Compute the ratio implied by a fitted Student-t distribution.', 'Calculați raportul implicat de o distribuție Student-t estimată.'),
        T('Interpretation: is ES 2.5\\% a stricter rule than VaR 1\\% on these markets?', 'Interpretare: este ES 2,5\\% o regulă mai strictă decît VaR 1\\% pe aceste piețe?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrr', T('& all data & 500 d. to 3/2009 & 500 d. to 5/2020 & Student-t & $\\hat\\nu$', '& toate datele & 500 z. pînă la 3/2009 & 500 z. pînă la 5/2020 & Student-t & $\\hat\\nu$'),
    c1rows, size='scriptsize') + items(
    T('Normal: $@{c1.norm}$; the data: from $@{c1.min}$ to $@{c1.max}$; Bitcoin and the BVB stocks have no data before 2009', 'Distribuția Normală: $@{c1.norm}$; datele: de la $@{c1.min}$ la $@{c1.max}$; Bitcoin și acțiunile BVB nu au date înainte de 2009'),
    T('The ratio can be below 1 (windows ending in 2009): one huge loss pushes VaR 1\\% up more than the average ES 2.5\\%', 'Raportul poate fi sub 1 (ferestrele care se încheie în 2009): o pierdere foarte mare împinge VaR 1\\% în sus mai mult decît media ES 2,5\\%'),
    T('Project design: rolling ratios for many assets, bootstrap intervals (the ratio is noisy), the link with the tail index $\\xi$ (Chapter 5), and the capital that each rule would require',
      'Designul proiectului: rapoarte pe ferestre mobile pentru multe active, intervale bootstrap (raportul este zgomotos), legătura cu indicele de coadă $\\xi$ (Capitolul 5) și capitalul cerut de fiecare regulă')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the risk of the BET over the last 500 days. The answer:', 'Un student a cerut unui asistent AI să rezume riscul BET în ultimele 500 de zile. Răspunsul:'),
    T('\\aiprompt{(a) The VaR 99\\% of the BET is -@{c2.var1}\\%.}', '\\aiprompt{(a) VaR 99\\% al BET este -@{c2.var1}\\%.}'),
    T('\\aiprompt{(b) ES is always at least as large as VaR, so ES 2.5\\% is above @{c2.var1}\\%.}', '\\aiprompt{(b) ES este întotdeauna cel puțin la fel de mare ca VaR, deci ES 2,5\\% este peste @{c2.var1}\\%.}'),
    T('\\aiprompt{(c) The 10-day VaR 1\\% is 10 x @{c2.var1} = @{c2.v10w}\\%.}', '\\aiprompt{(c) VaR 1\\% pe 10 zile este 10 x @{c2.var1} = @{c2.v10w}\\%.}'),
    T('\\aiprompt{(d) With 8 exceptions in 250 days the Kupiec p-value is @{c2.p8}, but the model stays in the green zone.}', '\\aiprompt{(d) Cu 8 depășiri în 250 de zile, p-valoarea Kupiec este @{c2.p8}, dar modelul rămîne în zona verde.}'),
    T('\\aiprompt{(e) VaR is subadditive, so the VaR of a portfolio never exceeds the sum of the VaRs of its parts.}', '\\aiprompt{(e) VaR este subaditiv, deci VaR-ul unui portofoliu nu depășește niciodată suma VaR-urilor componentelor.}'),
    T('\\aiprompt{(f) If the exceptions cluster, the Christoffersen test can reject even when their number is right.}', '\\aiprompt{(f) Dacă depășirile apar grupat, testul Christoffersen poate respinge modelul chiar dacă numărul lor este corect.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong twice: the level is the tail probability (VaR 1\\%), and VaR is a loss, a positive number: VaR 1\\% $= @{c2.var1}\\%$', '(a) Greșit de două ori: nivelul este probabilitatea cozii (VaR 1\\%), iar VaR este o pierdere, un număr pozitiv: VaR 1\\% $= @{c2.var1}\\%$'),
    T('(b) Wrong: $\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ only at the same level; here ES 2.5\\% $= @{c2.es25}\\%$ is below VaR 1\\% (A3)', '(b) Greșit: $\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ doar la același nivel; aici ES 2,5\\% $= @{c2.es25}\\%$ este sub VaR 1\\% (A3)'),
    T('(c) Wrong: the square-root-of-time rule gives $\\sqrt{10} \\times @{c2.var1} = @{c2.v10}\\%$, and even that holds only for i.i.d.\\ Normal returns', '(c) Greșit: regula rădăcinii pătrate a timpului dă $\\sqrt{10} \\times @{c2.var1} = @{c2.v10}\\%$ și chiar și aceasta este valabilă doar pentru randamente Normale i.i.d.'),
    T('(d) Wrong: 8 exceptions are in the yellow zone (plus factor 0.75); $LR_{uc} = @{c2.lr8}$ also rejects the model', '(d) Greșit: 8 depășiri înseamnă zona galbenă (adaosul 0,75); $LR_{uc} = @{c2.lr8}$ respinge și el modelul'),
    T('(e) Wrong: VaR can fail subadditivity (the two bonds of A4); ES is the coherent measure', '(e) Greșit: VaR poate să nu fie subaditiv (cele două obligațiuni din A4); ES este măsura coerentă'),
    T('(f) Correct: this is the independence part of the conditional coverage test (A6)', '(f) Corect: aceasta este componenta de independență a testului de acoperire condiționată (A6)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('VaR 1\\% $= -q_{1\\%}$, a positive loss; ES 2.5\\% averages the worst 2.5\\% of days; ES $\\ge$ VaR only at the same level', 'VaR 1\\% $= -q_{1\\%}$, o pierdere pozitivă; ES 2,5\\% face media celor mai proaste 2,5\\% dintre zile; ES $\\ge$ VaR doar la același nivel'),
    T('On real data the Normal method understates VaR; Cornish--Fisher fails for large kurtosis; HS and EVT agree at 1\\%', 'Pe date reale, metoda Normală subestimează VaR; Cornish--Fisher nu funcționează pentru boltire mare; HS și EVT sînt de acord la 1\\%'),
    T('Backtesting needs two checks: the number of exceptions (Kupiec) and their independence (Christoffersen)', 'Backtesting-ul are nevoie de două verificări: numărul depășirilor (Kupiec) și independența lor (Christoffersen)'),
    T('Diversification is weakest in crises: correlation and tail dependence rise together', 'Diversificarea este cea mai slabă în crize: corelația și dependența în cozi cresc împreună'),
    T('An AI answer is a draft: check the sign and the level of VaR, the horizon rule and the Basel zone', 'Un răspuns AI este o ciornă: verificați semnul și nivelul VaR, regula pentru orizont și zona Basel')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 10 develops each topic of today: definitions and coherence, estimation methods, conditional VaR, portfolios and copulas, backtesting',
      'Cursul 10 dezvoltă fiecare temă de azi: definițiile și coerența, metodele de estimare, VaR condiționat, portofoliile și copulele, backtesting-ul'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: rolling ratios, bootstrap intervals, the capital under each rule', 'C1 poate deveni un proiect de echipă: rapoarte pe ferestre mobile, intervale bootstrap, capitalul cerut de fiecare regulă'),
    T('Reading: \\refFHH, Ch.~16; exercises in \\refBHL, Ch.~16; \\refQRM, Ch.~2', 'Lectură: \\refFHH, cap.~16; exerciții în \\refBHL, cap.~16; \\refQRM, cap.~2')))

D.references(bib(['AT', 'Artzner', 'BCBSb', 'BHL', 'Chris', 'CF', 'DM', 'FHH', 'Kupiec', 'MF', 'QRM']), per=16)

if __name__ == '__main__':
    D.write(V)
