r"""
build_seminar3.py -- Seminarul 3 (Distribuții α-stabile), EN + RO dintr-o singură sursă
======================================================================================
Seminarul are loc ÎNAINTEA cursului 3: secțiunea „Ce vă trebuie azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_03/sem3_results.json (seminar3.py).
Ieșire:
  EN/Seminars/seminar3_alpha_stable_distributions.tex          (+ _solutions.tex)
  RO/Seminarii/seminar3_distributii_alfa_stabile_ro.tex        (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_03/seminar3.py && python3 latex/build_seminar3.py && python3 latex/sfm_build.py compile 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, fig   # noqa: E402
from ch3_common import ALPHA, NAMES, REFS, T, bib, load_sem, put_date, ro_tuples   # noqa: E402

S = load_sem()
V = Values()
D = Deck(3, 'seminar', refs=REFS, macros=ALPHA)


def qlsem():
    return '\\quantlet{SFM\\_ch3\\_seminar}{\\qlurl{SFM_ch3_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a, c in A['a1'].items():
    V.put(f'a1.{float(a):g}', c, 2)
V.put('a1.var', 2 * 25, 0)
for a, c in A['a2']['horizon'].items():
    V.put(f'a2.h.{float(a):g}', c, 2)
for a, c in A['a2']['portfolio'].items():
    V.put(f'a2.p.{float(a):g}', c, 2)
for c in ['s1_re', 's1_abs', 's0_re', 'tan', 'shift']:
    V.put(f'a3.{c}', A['a3'][c], 3 if c not in ('tan', 'shift') else 1)
V.put('a3.s1_im', -A['a3']['s1_im'], 3)
V.put('a4.tan', A['a4']['tan'], 4)
V.put('a4.add', -0.2 * 0.6 * A['a4']['tan'], 4)
V.put('a4.delta0', A['a4']['delta0'], 3)
V.put('a5.ratio2', A['a5']['ratio2'], 3)
V.put('a5.one6', A['a5']['one_in_6'], 0)
V.put('a5.one12', A['a5']['one_in_12'], 0)
V.int('a5.normal', round(50 / A['a5']['normal_ratio']))
V.put('a5.normal.ratio', A['a5']['normal_ratio'], 4)
for a in ['1.3', '1.9']:
    V.put(f'a6.{a}.ratio2', A['a6'][a]['ratio2'], 3)
    V.put(f'a6.{a}.x2', A['a6'][a]['one_in_x2'], 0)
    V.put(f'a6.{a}.x3', A['a6'][a]['one_in_x3'], 0)
# McCulloch pe hîrtie: coloana nu_beta = 0 a Tabelului III și coloana beta = 0 a Tabelului V (interpolare liniară)
T3 = {3.2: 1.484, 3.5: 1.391, 4.0: 1.279, 5.0: 1.128}
T5 = {1.2: 1.965, 1.3: 1.955, 1.4: 1.946}


def lin(x, tab):
    ks = sorted(tab)
    for lo, hi in zip(ks[:-1], ks[1:]):
        if lo <= x <= hi:
            return tab[lo] + (x - lo) / (hi - lo) * (tab[hi] - tab[lo])
    raise ValueError(x)


for k in ['a7', 'a8']:
    m = A[k]
    V.put(f'{k}.range', m['q95'] - m['q05'], 2)
    V.put(f'{k}.iqr', m['q75'] - m['q25'], 2)
    V.put(f'{k}.nua', m['nu_alpha'], 3)
    V.put(f'{k}.nub', m['nu_beta'], 3)
    ap = lin(m['nu_alpha'], T3)
    V.put(f'{k}.alpha.paper', ap, 3)
    ph = lin(ap, T5)
    V.put(f'{k}.phi3', ph, 3)
    V.put(f'{k}.gamma.paper', (m['q75'] - m['q25']) / ph, 3)
    for c in ['alpha', 'beta', 'gamma', 'delta0']:
        V.put(f'{k}.{c}', m[c], 3)
V.put('a9.V', A['a9']['V'], 3)
V.put('a9.x', A['a9']['x'], 3)
V.put('a10.normal', A['a10']['normal'], 3)
V.put('a10.cauchy', A['a10']['cauchy'], 3)
import math   # noqa: E402
V.put('a9.sin', math.sin(1.5 * A['a9']['V']), 3)
V.put('a9.cos', math.cos(A['a9']['V']) ** (1 / 1.5), 3)
V.put('a9.last', (math.cos(-0.5 * A['a9']['V']) / 1.2) ** (-1 / 3), 3)
V.put('a10.sinV', math.sin(A['a9']['V']), 3)
V.put('a10.sqrtW', math.sqrt(1.2), 3)

B1 = S['B1']
V.int('b1.n', B1['n'])
V.raw('b1.y0', B1['first'][:4])
V.raw('b1.y1', B1['last'][:4])
for c in ['q05', 'q25', 'q50', 'q75', 'q95']:
    V.put(f'b1.{c}', B1[c], 2)
for c in ['nu_alpha', 'nu_beta', 'alpha', 'beta', 'delta0', 'delta1', 'se', 'lo', 'hi']:
    V.put(f'b1.{c}', B1[c], 3)
V.put('b1.gamma', B1['gamma'], 2)
for k, b in S['B2'].items():
    V.int(f'b2.{k}.n', b['n'])
    V.raw(f'b2.{k}.y0', b['first'][:4])
    for c in ['nu_alpha', 'alpha', 'beta', 'delta0', 'se', 'lo', 'hi']:
        V.put(f'b2.{k}.{c}', b[c], 3)
    V.put(f'b2.{k}.gamma', b['gamma'], 2)
B3 = S['B3']
V.int('b3.n', B3['n'])
for c in ['alpha', 'se_alpha', 'beta', 'se_beta', 'delta0', 'delta1', 'mcc_alpha', 'mcc_beta']:
    V.put(f'b3.{c}', B3[c], 3)
for c in ['gamma', 'mcc_gamma', 't_nu']:
    V.put(f'b3.{c}', B3[c], 2)
for c in ['aic_normal', 'aic_t', 'aic_stable']:
    V.int(f'b3.{c}', round(B3[c]))
V.int('b3.aic.ns', round(B3['aic_normal'] - B3['aic_stable']))
V.int('b3.aic.st', round(B3['aic_stable'] - B3['aic_t']))
for c in ['q001_emp', 'q001_normal', 'q001_stable']:
    V.put(f'b3.{c}', B3[c], 2)
V.put('b3.lo', B3['alpha'] - 1.96 * B3['se_alpha'], 3)
V.put('b3.hi', B3['alpha'] + 1.96 * B3['se_alpha'], 3)
B4 = S['B4']
V.int('b4.n', B4['n'])
for c in ['alpha', 'se_alpha', 'beta', 'delta0', 'delta1']:
    V.put(f'b4.{c}', B4[c], 3)
V.put('b4.gamma', B4['gamma'], 2)
V.put('b4.t_nu', B4['t_nu'], 2)
for x in [10, 20]:
    V.raw(f'b4.obs{x}', str(B4[f'obs_{x}']))
    for m, lab in [('Normal', 'n'), ('Student-t', 't'), ('Stable', 's')]:
        v = B4[f'{m}_{x}']
        if v < 0.05:
            V.raw(f'b4.{lab}{x}', '\\approx 0')
        else:
            V.put(f'b4.{lab}{x}', v, 1)
for m, lab in [('emp', 'e'), ('Normal', 'n'), ('Student-t', 't'), ('Stable', 's')]:
    V.put(f'b4.var.{lab}', B4[f'var1_{m}'], 1)
for h, b in S['B5'].items():
    V.int(f'b5.{h}.n', b['n'])
    for c in ['alpha', 'lo', 'hi']:
        V.put(f'b5.{h}.{c}', b[c], 2)
for k, b in S['B6'].items():
    V.put(f'b6.{k}.alpha', b['alpha'], 3)
    for c in ['var_data', 'var_first_half', 'sim_median', 'sim_q05', 'sim_q95']:
        V.put(f'b6.{k}.{c}', b[c], 1)
    V.put(f'b6.{k}.share', 100 * b['share_below'], 0)
C2 = S['C2']
for c in ['alpha', 'beta', 'delta0', 'delta1', 'tan']:
    V.put(f'c2.{c}', C2[c], 3)
for c in ['gamma', 'sample_var', 'two_gamma2', 'scale20_wrong', 'scale20', 'var1_stable']:
    V.put(f'c2.{c}', C2[c], 2)
V.put('c2.sqrt20', math.sqrt(20), 2)
V.put('c2.f20', 20 ** (1 / C2['alpha']), 2)
c1rows = []
for k, per in S['C1'].items():
    for p_, r in per.items():
        c1rows.append(f'{NAMES[k]} & {p_.replace("-", "--")} & {r["n"]} & $⁅{r["alpha"]:.2f}⁆$ & $[⁅{r["lo"]:.2f}⁆, ⁅{r["hi"]:.2f}⁆]$')
c1 = S['C1']
V.put('c1.sp.min', min(r['alpha'] for r in c1['sp500'].values()), 2)
V.put('c1.sp.max', max(r['alpha'] for r in c1['sp500'].values()), 2)
V.put('c1.bet.min', min(r['alpha'] for r in c1['bet'].values()), 2)
V.put('c1.bet.max', max(r['alpha'] for r in c1['bet'].values()), 2)
lowp = min(c1['sp500'], key=lambda p: c1['sp500'][p]['alpha'])
V.raw('c1.sp.low', lowp.replace('-', '--'))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can a stable law with $\\alpha < 2$ describe daily returns, and what does it imply about their sums?',
       '\\textbf{Întrebarea}: poate o lege stabilă cu $\\alpha < 2$ să descrie randamentele zilnice și ce implică ea despre sumele lor?'),
     [T('this seminar comes \\textbf{before} Lecture 3: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 3: secțiunea „Ce vă trebuie azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: stability, characteristic functions, tails, McCulloch tables and one simulated draw on paper',
        'Partea A: stabilitate, funcții caracteristice, cozi, tabelele McCulloch și o extragere simulată, pe hîrtie'),
      T('Part B: stable fits of BET, S\\&P 500, DAX and Bitcoin returns, each with a measure of precision and an interpretation question',
        'Partea B: ajustări stabile pe randamentele BET, S\\&P 500, DAX și Bitcoin, fiecare cu o măsură a preciziei și o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Nu se predă nimic: seminarul este pentru exercițiu; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('how does the scale of a sum grow with the number of terms?', 'cum crește scala unei sume cu numărul de termeni?') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('what changes between the S1 and S0 parameterisations?', 'ce se schimbă între parametrizările S1 și S0?') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('how fast do power-law tails fall?', 'cît de repede scad cozile de tip putere?') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('McCulloch estimates from five quantiles', 'estimări McCulloch din cinci cuantile') + ' & ' + SP + ' & A7',
     'A9, A10 & ' + T('one stable draw by the CMS method', 'o extragere stabilă prin metoda CMS') + ' & ' + SP + ' & A9',
     'B1, B2 & ' + T('how heavy are the tails of BET, S\\&P 500, DAX and Bitcoin returns?', 'cît de groase sînt cozile randamentelor BET, S\\&P 500, DAX și Bitcoin?') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('ML fit, QQ plot, large losses: stable vs Normal vs Student-t', 'ajustare ML, grafic QQ, pierderi mari: stabilă vs Normală vs Student-t') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('does $\\alpha$ stay the same for weekly and monthly returns? does the variance settle?', 'rămîne $\\alpha$ la fel pentru randamentele săptămînale și lunare? se stabilizează varianța?') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('is $\\alpha$ constant over time? what is wrong in an AI answer?', 'este $\\alpha$ constant în timp? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500, DAX & EODHD & close & @{b1.y0}--@{b1.y1}',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & @{b2.btc.y0}--@{b1.y1}'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin)',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin)'),
    T('Weekly and monthly log returns: sums of the daily log returns of each calendar week or month',
      'Randamente logaritmice săptămînale și lunare: sumele randamentelor logaritmice zilnice din fiecare săptămînă sau lună calendaristică'),
    T('In the notebook: \\texttt{log\\_returns(\'bet\', \'2000-01-01\')}; no account or key is needed',
      'În notebook: \\texttt{log\\_returns(\'bet\', \'2000-01-01\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Ce vă trebuie azi')

D.frame(T('What You Need for Today (1/5): Stable Laws and Sums', 'Ce vă trebuie azi (1/5): legi stabile și sume'), items(
    (T('$X$ is \\textbf{stable} if for independent copies $X_1, X_2$ and any $a, b > 0$: $aX_1 + bX_2 \\overset{d}{=} cX + d$ ($\\overset{d}{=}$: same distribution)',
       '$X$ este \\textbf{stabilă} dacă, pentru copii independente $X_1, X_2$ și orice $a, b > 0$: $aX_1 + bX_2 \\overset{d}{=} cX + d$ ($\\overset{d}{=}$: aceeași distribuție)'),
     [T('the scale rule: $c^\\alpha = a^\\alpha + b^\\alpha$, with the \\textbf{index of stability} $0 < \\alpha \\le 2$',
        'regula scalei: $c^\\alpha = a^\\alpha + b^\\alpha$, cu \\textbf{indicele de stabilitate} $0 < \\alpha \\le 2$'),
      T('$n$ independent copies: $X_1 + \\dots + X_n \\overset{d}{=} n^{1/\\alpha}X + d_n$', '$n$ copii independente: $X_1 + \\dots + X_n \\overset{d}{=} n^{1/\\alpha}X + d_n$')]),
    (T('Examples', 'Exemple'),
     [T('Normal ($\\alpha = 2$): variances add, $c^2 = a^2 + b^2$', 'Normală ($\\alpha = 2$): varianțele se adună, $c^2 = a^2 + b^2$'),
      T('Cauchy ($\\alpha = 1$, density $1/(\\pi(1 + x^2))$): scales add, $c = a + b$', 'Cauchy ($\\alpha = 1$, densitatea $1/(\\pi(1 + x^2))$): scalele se adună, $c = a + b$')]),
    T('\\textbf{GCLT} (generalised central limit theorem): stable laws are the only possible limits of normalised sums of i.i.d.\\ (independent, identically distributed) variables; with finite variance the limit is Normal',
      '\\textbf{GCLT} (teorema limită centrală generalizată): legile stabile sînt singurele limite posibile ale sumelor normalizate de variabile i.i.d.\\ (independente și identic distribuite); cu varianță finită, limita este Normală')))

D.frame(T('What You Need for Today (2/5): Parameters and Parameterisations', 'Ce vă trebuie azi (2/5): parametri și parametrizări'), items(
    (T('\\textbf{Characteristic function} $\\varphi(t) = E[e^{itX}]$; it always exists and defines the law', '\\textbf{Funcția caracteristică} $\\varphi(t) = E[e^{itX}]$; există întotdeauna și definește legea'),
     [T('S1, $\\alpha \\ne 1$: $\\ln\\varphi(t) = -\\gamma^\\alpha|t|^\\alpha\\big[1 - i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big] + i\\delta_1 t$',
        'S1, $\\alpha \\ne 1$: $\\ln\\varphi(t) = -\\gamma^\\alpha|t|^\\alpha\\big[1 - i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big] + i\\delta_1 t$'),
      T('S0, $\\alpha \\ne 1$: $\\ln\\varphi(t) = -\\gamma^\\alpha|t|^\\alpha\\big[1 + i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}(|\\gamma t|^{1-\\alpha} - 1)\\big] + i\\delta_0 t$',
        'S0, $\\alpha \\ne 1$: $\\ln\\varphi(t) = -\\gamma^\\alpha|t|^\\alpha\\big[1 + i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}(|\\gamma t|^{1-\\alpha} - 1)\\big] + i\\delta_0 t$')]),
    T('Parameters: $\\alpha$ tails, $\\beta \\in [-1, 1]$ skewness, $\\gamma > 0$ scale, $\\delta$ location; only the location differs: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$',
      'Parametri: $\\alpha$ cozi, $\\beta \\in [-1, 1]$ asimetrie, $\\gamma > 0$ scală, $\\delta$ locație; diferă doar locația: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$'),
    (T('Special cases and traps', 'Cazuri particulare și capcane'),
     [T('$\\alpha = 2$: $N(\\delta, 2\\gamma^2)$, variance $2\\gamma^2$; $\\alpha = 1$, $\\beta = 0$: Cauchy, $\\varphi(t) = e^{-\\gamma|t| + i\\delta t}$',
        '$\\alpha = 2$: $N(\\delta, 2\\gamma^2)$, varianța $2\\gamma^2$; $\\alpha = 1$, $\\beta = 0$: Cauchy, $\\varphi(t) = e^{-\\gamma|t| + i\\delta t}$'),
      T('\\texttt{scipy.stats.levy\\_stable} uses S1 by default (\\texttt{loc} $= \\delta_1$, \\texttt{scale} $= \\gamma$)',
        '\\texttt{scipy.stats.levy\\_stable} folosește implicit S1 (\\texttt{loc} $= \\delta_1$, \\texttt{scale} $= \\gamma$)')])))

D.frame(T('What You Need for Today (3/5): Tails and Moments', 'Ce vă trebuie azi (3/5): cozi și momente'), items(
    (T('For $\\alpha < 2$, power-law tails: $P(X > x) \\approx C x^{-\\alpha}$ for large $x$', 'Pentru $\\alpha < 2$, cozi de tip putere: $P(X > x) \\approx C x^{-\\alpha}$ pentru $x$ mare'),
     [T('so $P(X > kx)/P(X > x) \\approx k^{-\\alpha}$: doubling the threshold divides the probability by $2^\\alpha$',
        'deci $P(X > kx)/P(X > x) \\approx k^{-\\alpha}$: dublarea pragului împarte probabilitatea la $2^\\alpha$'),
      T('the Normal tail falls much faster, like $e^{-x^2/(2\\sigma^2)}$', 'coada distribuției Normale scade mult mai repede, ca $e^{-x^2/(2\\sigma^2)}$')]),
    (T('Moments: for $\\alpha < 2$, $E|X|^p < \\infty$ only if $p < \\alpha$', 'Momente: pentru $\\alpha < 2$, $E|X|^p < \\infty$ doar dacă $p < \\alpha$'),
     [T('$\\alpha < 2$: infinite variance; $\\alpha \\le 1$: no mean; in S1 the mean ($\\alpha > 1$) is $\\delta_1$',
        '$\\alpha < 2$: varianță infinită; $\\alpha \\le 1$: nu există media; în S1, media ($\\alpha > 1$) este $\\delta_1$'),
      T('the sample variance of a stable sample keeps jumping as the sample grows', 'varianța de selecție a unui eșantion stabil continuă să sară pe măsură ce eșantionul crește')]),
    T('Quantiles always exist: $\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, the loss exceeded with probability 1\\%; ES (expected shortfall) needs $\\alpha > 1$',
      'Cuantilele există întotdeauna: $\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, pierderea depășită cu probabilitatea 1\\%; ES (expected shortfall) cere $\\alpha > 1$')))

D.frame(T('What You Need for Today (4/5): Simulation and Estimation', 'Ce vă trebuie azi (4/5): simulare și estimare'), items(
    (T('\\textbf{CMS} (Chambers--Mallows--Stuck) for $S(\\alpha, 0, 1, 0)$: $V = \\pi(U - \\frac12)$ with $U \\sim U(0, 1)$, $W \\sim \\mathrm{Exp}(1)$',
       '\\textbf{CMS} (Chambers--Mallows--Stuck) pentru $S(\\alpha, 0, 1, 0)$: $V = \\pi(U - \\frac12)$ cu $U \\sim U(0, 1)$, $W \\sim \\mathrm{Exp}(1)$'),
     [T('$X = \\dfrac{\\sin(\\alpha V)}{(\\cos V)^{1/\\alpha}}\\Big(\\dfrac{\\cos((1-\\alpha)V)}{W}\\Big)^{(1-\\alpha)/\\alpha}$; then $\\gamma X + \\delta$',
        '$X = \\dfrac{\\sin(\\alpha V)}{(\\cos V)^{1/\\alpha}}\\Big(\\dfrac{\\cos((1-\\alpha)V)}{W}\\Big)^{(1-\\alpha)/\\alpha}$; apoi $\\gamma X + \\delta$')]),
    (T('\\textbf{McCulloch} (1986): $\\nu_\\alpha = \\dfrac{q_{0.95} - q_{0.05}}{q_{0.75} - q_{0.25}}$, $\\nu_\\beta = \\dfrac{q_{0.95} + q_{0.05} - 2q_{0.5}}{q_{0.95} - q_{0.05}}$',
       '\\textbf{McCulloch} (1986): $\\nu_\\alpha = \\dfrac{q_{0.95} - q_{0.05}}{q_{0.75} - q_{0.25}}$, $\\nu_\\beta = \\dfrac{q_{0.95} + q_{0.05} - 2q_{0.5}}{q_{0.95} - q_{0.05}}$'),
     [T('tables give $\\hat\\alpha$, $\\hat\\beta$; then $\\hat\\gamma = (q_{0.75} - q_{0.25})/\\phi_3(\\hat\\alpha, \\hat\\beta)$; Normal: $\\nu_\\alpha = 2.439$',
        'tabelele dau $\\hat\\alpha$, $\\hat\\beta$; apoi $\\hat\\gamma = (q_{0.75} - q_{0.25})/\\phi_3(\\hat\\alpha, \\hat\\beta)$; Normală: $\\nu_\\alpha = 2.439$')]),
    T('\\textbf{ML} (maximum likelihood): maximise $\\sum_t \\ln f(r_t; \\alpha, \\beta, \\gamma, \\delta)$, with $f$ computed numerically from $\\varphi$; standard errors from the curvature',
      '\\textbf{ML} (verosimilitate maximă): maximizați $\\sum_t \\ln f(r_t; \\alpha, \\beta, \\gamma, \\delta)$, cu $f$ calculat numeric din $\\varphi$; erorile standard din curbură'),
    T('\\textbf{Bootstrap}: resample the days with replacement $B$ times, re-estimate each time; the 2.5\\% and 97.5\\% percentiles give a 95\\% interval',
      '\\textbf{Bootstrap}: reeșantionați zilele cu întoarcere de $B$ ori, reestimați de fiecare dată; percentilele de 2,5\\% și 97,5\\% dau un interval de 95\\%')))

D.frame(T('What You Need for Today (5/5): McCulloch Tables (Excerpt)', 'Ce vă trebuie azi (5/5): tabelele McCulloch (extras)'), cols(
    table('rr', '$\\nu_\\alpha$ & $\\alpha$', ['$2.439$ & $2.000$', '$2.6$ & $1.808$', '$2.8$ & $1.664$', '$3.0$ & $1.563$', '$3.2$ & $1.484$',
                                              '$3.5$ & $1.391$', '$4.0$ & $1.279$', '$5.0$ & $1.128$', '$6.0$ & $1.029$'], size='footnotesize'),
    table('rr', '$\\alpha$ & $\\phi_3(\\alpha, 0)$', ['$2.0$ & $1.908$', '$1.8$ & $1.921$', '$1.6$ & $1.933$', '$1.5$ & $1.939$', '$1.4$ & $1.946$',
                                                     '$1.3$ & $1.955$', '$1.2$ & $1.965$', '$1.1$ & $1.980$', '$1.0$ & $2.000$'], size='footnotesize') + items(
        T('Left: Table III, column $\\nu_\\beta = 0$; right: Table V, column $\\beta = 0$ \\refMcCulloch', 'Stînga: Tabelul III, coloana $\\nu_\\beta = 0$; dreapta: Tabelul V, coloana $\\beta = 0$ \\refMcCulloch'),
        T('Between two rows, interpolate linearly', 'Între două rînduri, interpolați liniar'),
        T('For $|\\nu_\\beta| < 0.05$ the $\\beta = 0$ columns are close enough on paper', 'Pentru $|\\nu_\\beta| < 0.05$, coloanele cu $\\beta = 0$ sînt suficient de apropiate pe hîrtie')),
    wl='0.40', wr='0.56'), size='footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: The Scale of a Sum', 'A1: scala unei sume'),
         items(T('$X_1, X_2$ are independent copies of $S(\\alpha, 0, 1, 0)$; $a = 3$, $b = 4$.', '$X_1, X_2$ sînt copii independente ale lui $S(\\alpha, 0, 1, 0)$; $a = 3$, $b = 4$.'),
               T('1. Compute $c$ with $3X_1 + 4X_2 \\overset{d}{=} cX$ for $\\alpha = 2$, $\\alpha = 1.5$ and $\\alpha = 1$.', '1. Calculați $c$ cu $3X_1 + 4X_2 \\overset{d}{=} cX$ pentru $\\alpha = 2$, $\\alpha = 1.5$ și $\\alpha = 1$.'),
               T('2. For $\\alpha = 2$, check the result with variances.', '2. Pentru $\\alpha = 2$, verificați rezultatul cu varianțe.'),
               T('3. Explain why $c$ grows as $\\alpha$ falls.', '3. Explicați de ce $c$ crește cînd $\\alpha$ scade.'),
               T('Report: three values of $c$ and one sentence.', 'Raportați: trei valori ale lui $c$ și o frază.')),
         items(T('1. $c = (3^\\alpha + 4^\\alpha)^{1/\\alpha}$: $\\alpha = 2$: $@{a1.2}$; $\\alpha = 1.5$: $@{a1.1.5}$; $\\alpha = 1$: $@{a1.1}$',
                 '1. $c = (3^\\alpha + 4^\\alpha)^{1/\\alpha}$: $\\alpha = 2$: $@{a1.2}$; $\\alpha = 1.5$: $@{a1.1.5}$; $\\alpha = 1$: $@{a1.1}$'),
               T('2. $S(2, 0, 1, 0) = N(0, 2)$: $\\mathrm{Var}(3X_1 + 4X_2) = (9 + 16) \\cdot 2 = @{a1.var} = \\mathrm{Var}(5X)$', '2. $S(2, 0, 1, 0) = N(0, 2)$: $\\mathrm{Var}(3X_1 + 4X_2) = (9 + 16) \\cdot 2 = @{a1.var} = \\mathrm{Var}(5X)$'),
               T('3. With small $\\alpha$, large values are frequent and one term can dominate: the scale of the sum approaches the sum of the scales.',
                 '3. Cu $\\alpha$ mic, valorile mari sînt frecvente și un singur termen poate domina: scala sumei se apropie de suma scalelor.')),
         size='footnotesize')

D.proposed(T('A2: Horizons and Diversification', 'A2: orizonturi și diversificare'),
           items(T('Daily returns are i.i.d.\\ $S(\\alpha, 0, \\gamma, 0)$. Model: A1.', 'Randamentele zilnice sînt i.i.d.\\ $S(\\alpha, 0, \\gamma, 0)$. Model: A1.'),
                 T('1. Compute the scale of the 20-day sum, as a multiple of $\\gamma$, for $\\alpha = 2$, $1.7$ and $1.5$.', '1. Calculați scala sumei pe 20 de zile, ca multiplu al lui $\\gamma$, pentru $\\alpha = 2$, $1.7$ și $1.5$.'),
                 T('2. Compute the scale of an equally weighted portfolio of $N = 10$ i.i.d.\\ assets for $\\alpha = 2$, $1.5$ and $1$.',
                   '2. Calculați scala unui portofoliu cu ponderi egale din $N = 10$ active i.i.d.\\ pentru $\\alpha = 2$, $1.5$ și $1$.'),
                 T('3. Explain what happens to diversification when $\\alpha = 1$.', '3. Explicați ce se întîmplă cu diversificarea cînd $\\alpha = 1$.'),
                 T('Report: six multiples of $\\gamma$ and one sentence.', 'Raportați: șase multipli ai lui $\\gamma$ și o frază.')),
           items(T('1. $20^{1/\\alpha}$: $@{a2.h.2}$, $@{a2.h.1.7}$, $@{a2.h.1.5}$', '1. $20^{1/\\alpha}$: $@{a2.h.2}$, $@{a2.h.1.7}$, $@{a2.h.1.5}$'),
                 T('2. $\\frac{1}{N}\\sum_i X_i$ has scale $N^{1/\\alpha - 1}\\gamma$: $@{a2.p.2}$, $@{a2.p.1.5}$, $@{a2.p.1}$', '2. $\\frac{1}{N}\\sum_i X_i$ are scala $N^{1/\\alpha - 1}\\gamma$: $@{a2.p.2}$, $@{a2.p.1.5}$, $@{a2.p.1}$'),
                 T('3. With $\\alpha = 1$ the portfolio has the same scale as one asset: diversification does not reduce risk.',
                   '3. Cu $\\alpha = 1$, portofoliul are aceeași scală ca un singur activ: diversificarea nu reduce riscul.')),
           size='footnotesize')

D.solved(T('A3: S1 and S0 at $t = 1$', 'A3: S1 și S0 în $t = 1$'),
         items(T('$X$ has $\\alpha = 1.5$, $\\beta = 0.3$, $\\gamma = 1$, $\\delta = 0$.', '$X$ are $\\alpha = 1.5$, $\\beta = 0.3$, $\\gamma = 1$, $\\delta = 0$.'),
               T('1. Compute $\\varphi(1)$ in the S1 parameterisation.', '1. Calculați $\\varphi(1)$ în parametrizarea S1.'),
               T('2. Compute $\\varphi(1)$ in the S0 parameterisation.', '2. Calculați $\\varphi(1)$ în parametrizarea S0.'),
               T('3. Explain the difference using $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$.', '3. Explicați diferența folosind $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$.'),
               T('Report: two complex numbers and one sentence.', 'Raportați: două numere complexe și o frază.')),
         items(T('1. $\\tan\\frac{3\\pi}{4} = @{a3.tan}$; $\\ln\\varphi(1) = -[1 - i(0.3)(-1)] = -1 - 0.3i$; $\\varphi(1) = e^{-1}(\\cos 0.3 - i\\sin 0.3) = @{a3.s1_re} - @{a3.s1_im}i$',
                 '1. $\\tan\\frac{3\\pi}{4} = @{a3.tan}$; $\\ln\\varphi(1) = -[1 - i(0.3)(-1)] = -1 - 0.3i$; $\\varphi(1) = e^{-1}(\\cos 0.3 - i\\sin 0.3) = @{a3.s1_re} - @{a3.s1_im}i$'),
               T('2. $|t|^{1-\\alpha} - 1 = 0$ at $t = 1$: $\\varphi(1) = e^{-1} = @{a3.s0_re}$', '2. $|t|^{1-\\alpha} - 1 = 0$ în $t = 1$: $\\varphi(1) = e^{-1} = @{a3.s0_re}$'),
               T('3. Same modulus $e^{-1}$; the laws differ by a shift: S1 with $\\delta_1 = 0$ is S0 with $\\delta_0 = 0.3 \\cdot (@{a3.tan}) = @{a3.shift}$.',
                 '3. Același modul $e^{-1}$; legile diferă printr-o deplasare: S1 cu $\\delta_1 = 0$ este S0 cu $\\delta_0 = 0.3 \\cdot (@{a3.tan}) = @{a3.shift}$.')),
         size='footnotesize')

D.proposed(T('A4: Special Cases and a Location Conversion', 'A4: cazuri particulare și o conversie a locației'),
           items(T('Model: A3.', 'Model: A3.'),
                 T('1. Write $\\varphi(t)$ for $\\alpha = 2$ and compare it with $\\exp(i\\mu t - \\sigma^2 t^2/2)$, the characteristic function of $N(\\mu, \\sigma^2)$: what are $\\mu$ and $\\sigma^2$?',
                   '1. Scrieți $\\varphi(t)$ pentru $\\alpha = 2$ și comparați cu $\\exp(i\\mu t - \\sigma^2 t^2/2)$, funcția caracteristică a $N(\\mu, \\sigma^2)$: cît sînt $\\mu$ și $\\sigma^2$?'),
                 T('2. Write $\\varphi(t)$ for $\\alpha = 1$, $\\beta = 0$ and name the law.', '2. Scrieți $\\varphi(t)$ pentru $\\alpha = 1$, $\\beta = 0$ și numiți legea.'),
                 T('3. scipy reports $\\alpha = 1.7$, $\\beta = -0.2$, $\\gamma = 0.6$, \\texttt{loc} $= 0.05$; compute $\\delta_0$.',
                   '3. scipy raportează $\\alpha = 1.7$, $\\beta = -0.2$, $\\gamma = 0.6$, \\texttt{loc} $= 0.05$; calculați $\\delta_0$.'),
                 T('Report: $\\mu$, $\\sigma^2$, the name of the law and $\\delta_0$.', 'Raportați: $\\mu$, $\\sigma^2$, numele legii și $\\delta_0$.')),
           items(T('1. $\\varphi(t) = \\exp(-\\gamma^2 t^2 + i\\delta t)$: $\\mu = \\delta$, $\\sigma^2 = 2\\gamma^2$ ($\\beta$ plays no role, $\\tan\\pi = 0$)',
                   '1. $\\varphi(t) = \\exp(-\\gamma^2 t^2 + i\\delta t)$: $\\mu = \\delta$, $\\sigma^2 = 2\\gamma^2$ ($\\beta$ nu contează, $\\tan\\pi = 0$)'),
                 T('2. $\\varphi(t) = \\exp(-\\gamma|t| + i\\delta t)$: the Cauchy law with location $\\delta$ and scale $\\gamma$', '2. $\\varphi(t) = \\exp(-\\gamma|t| + i\\delta t)$: legea Cauchy cu locația $\\delta$ și scala $\\gamma$'),
                 T('3. scipy uses S1: $\\delta_0 = 0.05 + (-0.2)(0.6)\\tan(0.85\\pi)$, $\\tan(0.85\\pi) = @{a4.tan}$, so $\\delta_0 = 0.05 + @{a4.add} = @{a4.delta0}$',
                   '3. scipy folosește S1: $\\delta_0 = 0.05 + (-0.2)(0.6)\\tan(0.85\\pi)$, $\\tan(0.85\\pi) = @{a4.tan}$, deci $\\delta_0 = 0.05 + @{a4.add} = @{a4.delta0}$')),
           size='footnotesize')

D.solved(T('A5: How Fast Do the Tails Fall?', 'A5: cît de repede scad cozile?'),
         items(T('Daily losses have a power-law tail with $\\alpha = 1.7$; a loss above $x$ happens on 1 day in 50.', 'Pierderile zilnice au o coadă de tip putere cu $\\alpha = 1.7$; o pierdere peste $x$ apare într-o zi din 50.'),
               T('1. Compute $P(L > 2x)/P(L > x)$.', '1. Calculați $P(L > 2x)/P(L > x)$.'),
               T('2. How often does a loss above $2x$ happen? And above $4x$?', '2. Cît de des apare o pierdere peste $2x$? Dar peste $4x$?'),
               T('3. Compare with a Normal law that has the same frequency of losses above $x$.', '3. Comparați cu o lege Normală care are aceeași frecvență a pierderilor peste $x$.'),
               T('Report: one ratio, two frequencies and one sentence.', 'Raportați: un raport, două frecvențe și o frază.')),
         items(T('1. $2^{-1.7} = @{a5.ratio2}$', '1. $2^{-1.7} = @{a5.ratio2}$'),
               T('2. Above $2x$: 1 day in $50 \\cdot 2^{1.7} \\approx @{a5.one6}$; above $4x$: 1 day in $50 \\cdot 4^{1.7} \\approx @{a5.one12}$',
                 '2. Peste $2x$: o zi din $50 \\cdot 2^{1.7} \\approx @{a5.one6}$; peste $4x$: o zi din $50 \\cdot 4^{1.7} \\approx @{a5.one12}$'),
               T('3. Normal: $x = z_{0.98}\\sigma$; $P(L > 2x) = P(Z > 2z_{0.98})$, i.e.\\ 1 day in about $@{a5.normal}$: the power law keeps large losses frequent.',
                 '3. Normală: $x = z_{0.98}\\sigma$; $P(L > 2x) = P(Z > 2z_{0.98})$, adică o zi din circa $@{a5.normal}$: legea putere păstrează pierderile mari frecvente.')),
         size='footnotesize')

D.proposed(T('A6: Tails and Moments for Other $\\alpha$', 'A6: cozi și momente pentru alte valori ale lui $\\alpha$'),
           items(T('A loss above $x$ happens on 1 day in 100. Model: A5.', 'O pierdere peste $x$ apare într-o zi din 100. Model: A5.'),
                 T('1. For $\\alpha = 1.3$ and $\\alpha = 1.9$, compute how often losses above $2x$ and above $3x$ happen.',
                   '1. Pentru $\\alpha = 1.3$ și $\\alpha = 1.9$, calculați cît de des apar pierderi peste $2x$ și peste $3x$.'),
                 T('2. For $\\alpha = 1.9$, $1.3$ and $0.8$, say whether the mean and the variance exist.', '2. Pentru $\\alpha = 1.9$, $1.3$ și $0.8$, spuneți dacă există media și varianța.'),
                 T('3. Explain why a sample variance can be computed even when the variance does not exist.', '3. Explicați de ce se poate calcula o varianță de selecție chiar dacă varianța nu există.'),
                 T('Report: four frequencies, a small table of moments and one sentence.', 'Raportați: patru frecvențe, un mic tabel al momentelor și o frază.')),
           items(T('1. $\\alpha = 1.3$: 1 in $@{a6.1.3.x2}$ ($2x$), 1 in $@{a6.1.3.x3}$ ($3x$); $\\alpha = 1.9$: 1 in $@{a6.1.9.x2}$, 1 in $@{a6.1.9.x3}$',
                   '1. $\\alpha = 1.3$: 1 din $@{a6.1.3.x2}$ ($2x$), 1 din $@{a6.1.3.x3}$ ($3x$); $\\alpha = 1.9$: 1 din $@{a6.1.9.x2}$, 1 din $@{a6.1.9.x3}$'),
                 T('2. $\\alpha = 1.9$ and $1.3$: mean yes, variance no; $\\alpha = 0.8$: neither', '2. $\\alpha = 1.9$ și $1.3$: media da, varianța nu; $\\alpha = 0.8$: niciuna'),
                 T('3. Any finite sample has a finite sample variance; it just does not converge to a number as $n$ grows.',
                   '3. Orice eșantion finit are o varianță de selecție finită; doar că ea nu converge către un număr cînd $n$ crește.')),
           size='footnotesize')

D.solved(T('A7: McCulloch from Five Quantiles', 'A7: McCulloch din cinci cuantile'),
         items(T('Quantiles of 2000 daily returns (\\%): $q_{0.05} = -2.10$, $q_{0.25} = -0.55$, $q_{0.5} = 0.05$, $q_{0.75} = 0.62$, $q_{0.95} = 2.06$.',
                 'Cuantilele a 2000 de randamente zilnice (\\%): $q_{0.05} = -2.10$, $q_{0.25} = -0.55$, $q_{0.5} = 0.05$, $q_{0.75} = 0.62$, $q_{0.95} = 2.06$.'),
               T('1. Compute $\\nu_\\alpha$ and $\\nu_\\beta$.', '1. Calculați $\\nu_\\alpha$ și $\\nu_\\beta$.'),
               T('2. Read $\\hat\\alpha$ from the table excerpt (column $\\nu_\\beta = 0$).', '2. Citiți $\\hat\\alpha$ din extrasul de tabel (coloana $\\nu_\\beta = 0$).'),
               T('3. Compute $\\hat\\gamma$ with Table V.', '3. Calculați $\\hat\\gamma$ cu Tabelul V.'),
               T('Report: two ratios, $\\hat\\alpha$, $\\hat\\gamma$ and one sentence on the tails.', 'Raportați: două rapoarte, $\\hat\\alpha$, $\\hat\\gamma$ și o frază despre cozi.')),
         items(T('1. $\\nu_\\alpha = @{a7.range}/@{a7.iqr} = @{a7.nua}$; $\\nu_\\beta = (2.06 - 2.10 - 0.10)/@{a7.range} = @{a7.nub}$, almost symmetric',
                 '1. $\\nu_\\alpha = @{a7.range}/@{a7.iqr} = @{a7.nua}$; $\\nu_\\beta = (2.06 - 2.10 - 0.10)/@{a7.range} = @{a7.nub}$, aproape simetric'),
               T('2. Between $3.5 \\mapsto 1.391$ and $4.0 \\mapsto 1.279$: $\\hat\\alpha \\approx @{a7.alpha.paper}$', '2. Între $3.5 \\mapsto 1.391$ și $4.0 \\mapsto 1.279$: $\\hat\\alpha \\approx @{a7.alpha.paper}$'),
               T('3. $\\phi_3(\\hat\\alpha, 0) \\approx @{a7.phi3}$, so $\\hat\\gamma = @{a7.iqr}/@{a7.phi3} = @{a7.gamma.paper}$', '3. $\\phi_3(\\hat\\alpha, 0) \\approx @{a7.phi3}$, deci $\\hat\\gamma = @{a7.iqr}/@{a7.phi3} = @{a7.gamma.paper}$'),
               T('With the full tables ($\\nu_\\beta \\ne 0$): $\\hat\\alpha = @{a7.alpha}$, $\\hat\\beta = @{a7.beta}$, $\\hat\\gamma = @{a7.gamma}$; tails far heavier than for the Normal distribution ($\\nu_\\alpha = 2.439$).',
                 'Cu tabelele complete ($\\nu_\\beta \\ne 0$): $\\hat\\alpha = @{a7.alpha}$, $\\hat\\beta = @{a7.beta}$, $\\hat\\gamma = @{a7.gamma}$; cozi mult mai groase decît la distribuția Normală ($\\nu_\\alpha = 2.439$).')),
         size='scriptsize')

D.proposed(T('A8: McCulloch for a Crypto Asset', 'A8: McCulloch pentru un activ cripto'),
           items(T('Quantiles of daily returns (\\%): $q_{0.05} = -6.90$, $q_{0.25} = -1.45$, $q_{0.5} = 0.12$, $q_{0.75} = 1.75$, $q_{0.95} = 6.40$. Model: A7.',
                   'Cuantilele randamentelor zilnice (\\%): $q_{0.05} = -6.90$, $q_{0.25} = -1.45$, $q_{0.5} = 0.12$, $q_{0.75} = 1.75$, $q_{0.95} = 6.40$. Model: A7.'),
                 T('1. Compute $\\nu_\\alpha$ and $\\nu_\\beta$.', '1. Calculați $\\nu_\\alpha$ și $\\nu_\\beta$.'),
                 T('2. Read $\\hat\\alpha$ and compute $\\hat\\gamma$ from the table excerpts.', '2. Citiți $\\hat\\alpha$ și calculați $\\hat\\gamma$ din extrasele de tabel.'),
                 T('3. Compare with A7: which asset has the heavier tails?', '3. Comparați cu A7: care activ are cozile mai groase?'),
                 T('Report: two ratios, $\\hat\\alpha$, $\\hat\\gamma$ and one sentence.', 'Raportați: două rapoarte, $\\hat\\alpha$, $\\hat\\gamma$ și o frază.')),
           items(T('1. $\\nu_\\alpha = @{a8.range}/@{a8.iqr} = @{a8.nua}$; $\\nu_\\beta = @{a8.nub}$', '1. $\\nu_\\alpha = @{a8.range}/@{a8.iqr} = @{a8.nua}$; $\\nu_\\beta = @{a8.nub}$'),
                 T('2. Between $4.0 \\mapsto 1.279$ and $5.0 \\mapsto 1.128$: $\\hat\\alpha \\approx @{a8.alpha.paper}$; $\\phi_3 \\approx @{a8.phi3}$, $\\hat\\gamma \\approx @{a8.gamma.paper}$ (full tables: $@{a8.alpha}$, $@{a8.gamma}$)',
                   '2. Între $4.0 \\mapsto 1.279$ și $5.0 \\mapsto 1.128$: $\\hat\\alpha \\approx @{a8.alpha.paper}$; $\\phi_3 \\approx @{a8.phi3}$, $\\hat\\gamma \\approx @{a8.gamma.paper}$ (tabelele complete: $@{a8.alpha}$, $@{a8.gamma}$)'),
                 T('3. The crypto asset: larger $\\nu_\\alpha$, smaller $\\hat\\alpha$, heavier tails, and a scale almost three times larger.',
                   '3. Activul cripto: $\\nu_\\alpha$ mai mare, $\\hat\\alpha$ mai mic, cozi mai groase și o scală de aproape trei ori mai mare.')),
           size='scriptsize')

D.solved(T('A9: One Stable Draw by CMS', 'A9: o extragere stabilă prin CMS'),
         items(T('$\\alpha = 1.5$, $\\beta = 0$; a uniform draw $U = 0.8$ and an exponential draw $W = 1.2$.', '$\\alpha = 1.5$, $\\beta = 0$; o extragere uniformă $U = 0.8$ și o extragere exponențială $W = 1.2$.'),
               T('1. Compute $V = \\pi(U - 1/2)$.', '1. Calculați $V = \\pi(U - 1/2)$.'),
               T('2. Compute the three factors of the CMS formula.', '2. Calculați cei trei factori ai formulei CMS.'),
               T('3. Compute the draw $X$ and the draw of $S(1.5, 0, 0.6, 0.05)$.', '3. Calculați extragerea $X$ și extragerea din $S(1.5, 0, 0.6, 0.05)$.'),
               T('Report: $V$, three factors, two draws.', 'Raportați: $V$, trei factori, două extrageri.')),
         items(T('1. $V = 0.3\\pi = @{a9.V}$', '1. $V = 0.3\\pi = @{a9.V}$'),
               T('2. $\\sin(1.5V) = @{a9.sin}$; $(\\cos V)^{1/1.5} = @{a9.cos}$; $(\\cos(-0.5V)/1.2)^{-1/3} = @{a9.last}$',
                 '2. $\\sin(1.5V) = @{a9.sin}$; $(\\cos V)^{1/1.5} = @{a9.cos}$; $(\\cos(-0.5V)/1.2)^{-1/3} = @{a9.last}$'),
               T('3. $X = @{a9.sin}/@{a9.cos} \\times @{a9.last} = @{a9.x}$; then $0.6X + 0.05$', '3. $X = @{a9.sin}/@{a9.cos} \\times @{a9.last} = @{a9.x}$; apoi $0.6X + 0.05$')),
         size='footnotesize')

D.proposed(T('A10: CMS for the Normal and the Cauchy Laws', 'A10: CMS pentru legea Normală și legea Cauchy'),
           items(T('Model: A9.', 'Model: A9.'),
                 T('1. Show that for $\\alpha = 2$ the CMS formula becomes $X = 2\\sin V\\sqrt{W}$.', '1. Arătați că pentru $\\alpha = 2$ formula CMS devine $X = 2\\sin V\\sqrt{W}$.'),
                 T('2. Show that for $\\alpha = 1$ it becomes $X = \\tan V$.', '2. Arătați că pentru $\\alpha = 1$ devine $X = \\tan V$.'),
                 T('3. Compute both draws for $U = 0.8$, $W = 1.2$.', '3. Calculați ambele extrageri pentru $U = 0.8$, $W = 1.2$.'),
                 T('Report: two short derivations and two numbers.', 'Raportați: două derivări scurte și două numere.')),
           items(T('1. $\\sin 2V = 2\\sin V\\cos V$, $(\\cos V)^{1/2}$ and $(\\cos(-V)/W)^{-1/2} = \\sqrt{W}/(\\cos V)^{1/2}$: the cosines cancel',
                   '1. $\\sin 2V = 2\\sin V\\cos V$, $(\\cos V)^{1/2}$ și $(\\cos(-V)/W)^{-1/2} = \\sqrt{W}/(\\cos V)^{1/2}$: cosinusurile se simplifică'),
                 T('2. The exponent $(1 - \\alpha)/\\alpha = 0$ and $\\sin V/\\cos V = \\tan V$', '2. Exponentul $(1 - \\alpha)/\\alpha = 0$ și $\\sin V/\\cos V = \\tan V$'),
                 T('3. Normal: $2 \\cdot @{a10.sinV} \\cdot @{a10.sqrtW} = @{a10.normal}$; Cauchy: $\\tan(@{a9.V}) = @{a10.cauchy}$',
                   '3. Normală: $2 \\cdot @{a10.sinV} \\cdot @{a10.sqrtW} = @{a10.normal}$; Cauchy: $\\tan(@{a9.V}) = @{a10.cauchy}$')),
           size='footnotesize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Precision and Interpretation', 'Partea B: date reale, precizie și interpretare')

D.task(T('B1: How Heavy Are the Tails of the BET? [Solved]', 'B1: cît de groase sînt cozile BET? [Rezolvat]'),
       T('which stable law do the McCulloch quantile estimates give for the daily BET returns, and how precise is $\\hat\\alpha$?',
         'ce lege stabilă dau estimările McCulloch pentru randamentele zilnice BET și cît de precis este $\\hat\\alpha$?'),
       T('BET closes, @{b1.y0}--@{b1.y1}, daily log returns in \\%', 'închiderile BET, @{b1.y0}--@{b1.y1}, randamente logaritmice zilnice în \\%'),
       [T('Compute the 5\\%, 25\\%, 50\\%, 75\\% and 95\\% quantiles and the ratios $\\nu_\\alpha$ and $\\nu_\\beta$.', 'Calculați cuantilele de 5\\%, 25\\%, 50\\%, 75\\% și 95\\% și rapoartele $\\nu_\\alpha$ și $\\nu_\\beta$.'),
        T('Compute the McCulloch estimates $\\hat\\alpha$, $\\hat\\beta$, $\\hat\\gamma$ and $\\hat\\delta_0$ with the full tables.', 'Calculați estimările McCulloch $\\hat\\alpha$, $\\hat\\beta$, $\\hat\\gamma$ și $\\hat\\delta_0$ cu tabelele complete.'),
        T('Bootstrap $\\hat\\alpha$ with 1000 resamples of the days and report the standard error and the 95\\% percentile interval.',
          'Faceți bootstrap pentru $\\hat\\alpha$ cu 1000 de reeșantionări ale zilelor și raportați eroarea standard și intervalul percentil de 95\\%.'),
        T('Interpretation: is $\\alpha = 2$ (the Normal distribution) compatible with the data?', 'Interpretare: este $\\alpha = 2$ (distribuția Normală) compatibil cu datele?')],
       T('five quantiles, two ratios, four estimates, the interval and one sentence', 'cinci cuantile, două rapoarte, patru estimări, intervalul și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch3_sem_b1', h='0.42') + items(
    T('$n = @{b1.n}$; quantiles $@{b1.q05}$, $@{b1.q25}$, $@{b1.q50}$, $@{b1.q75}$, $@{b1.q95}$; $\\nu_\\alpha = @{b1.nu_alpha}$, $\\nu_\\beta = @{b1.nu_beta}$',
      '$n = @{b1.n}$; cuantile $@{b1.q05}$, $@{b1.q25}$, $@{b1.q50}$, $@{b1.q75}$, $@{b1.q95}$; $\\nu_\\alpha = @{b1.nu_alpha}$, $\\nu_\\beta = @{b1.nu_beta}$'),
    T('$\\hat\\alpha = @{b1.alpha}$, $\\hat\\beta = @{b1.beta}$, $\\hat\\gamma = @{b1.gamma}\\%$, $\\hat\\delta_0 = @{b1.delta0}\\%$; bootstrap SE $@{b1.se}$, 95\\% interval $[@{b1.lo}, @{b1.hi}]$',
      '$\\hat\\alpha = @{b1.alpha}$, $\\hat\\beta = @{b1.beta}$, $\\hat\\gamma = @{b1.gamma}\\%$, $\\hat\\delta_0 = @{b1.delta0}\\%$; SE bootstrap $@{b1.se}$, interval de 95\\% $[@{b1.lo}, @{b1.hi}]$'),
    T('Interpretation: $\\alpha = 2$ lies far outside the interval; the BET body and tails are much wider than the Normal distribution allows',
      'Interpretare: $\\alpha = 2$ este mult în afara intervalului; corpul și cozile BET sînt mult mai largi decît permite distribuția Normală')) + qlsem(), 'footnotesize')

D.task(T('B2: Three More Markets [Proposed]', 'B2: încă trei piețe [Propus]'),
       T('which of the S\\&P 500, DAX and Bitcoin has the heaviest tails by McCulloch\'s method? Model: B1.', 'care dintre S\\&P 500, DAX și Bitcoin are cele mai groase cozi după metoda McCulloch? Model: B1.'),
       T('S\\&P 500 and DAX since @{b2.sp500.y0}, Bitcoin since @{b2.btc.y0}, daily log returns in \\%', 'S\\&P 500 și DAX din @{b2.sp500.y0}, Bitcoin din @{b2.btc.y0}, randamente logaritmice zilnice în \\%'),
       [T('Repeat steps 1--3 of B1 for each series.', 'Repetați pașii 1--3 din B1 pentru fiecare serie.'),
        T('Put $\\nu_\\alpha$, $\\hat\\alpha$ with its interval, $\\hat\\beta$ and $\\hat\\gamma$ of the four series (with the BET) in one table.',
          'Puneți $\\nu_\\alpha$, $\\hat\\alpha$ cu intervalul lui, $\\hat\\beta$ și $\\hat\\gamma$ pentru cele patru serii (cu BET) într-un tabel.'),
        T('Interpretation: is the difference between the $\\hat\\alpha$ of Bitcoin and of the S\\&P 500 larger than the estimation error?',
          'Interpretare: este diferența dintre $\\hat\\alpha$ al Bitcoin și cel al S\\&P 500 mai mare decît eroarea de estimare?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{NAMES[k]} & $@{{b2.{k}.nu_alpha}}$ & $@{{b2.{k}.alpha}}$ & $[@{{b2.{k}.lo}}, @{{b2.{k}.hi}}]$ & $@{{b2.{k}.beta}}$ & $@{{b2.{k}.gamma}}$')


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrr', T('& $\\nu_\\alpha$ & $\\hat\\alpha$ & 95\\% interval & $\\hat\\beta$ & $\\hat\\gamma$ (\\%)', '& $\\nu_\\alpha$ & $\\hat\\alpha$ & interval 95\\% & $\\hat\\beta$ & $\\hat\\gamma$ (\\%)'),
    ['BET & $@{b1.nu_alpha}$ & $@{b1.alpha}$ & $[@{b1.lo}, @{b1.hi}]$ & $@{b1.beta}$ & $@{b1.gamma}$'] + [b2row(k) for k in ['sp500', 'dax', 'btc']],
    size='footnotesize') + items(
    T('Heaviest tails: Bitcoin, $\\hat\\alpha = @{b2.btc.alpha}$; its interval does not overlap that of the S\\&P 500', 'Cele mai groase cozi: Bitcoin, $\\hat\\alpha = @{b2.btc.alpha}$; intervalul lui nu se suprapune cu cel al S\\&P 500'),
    T('BET, S\\&P 500 and DAX: overlapping intervals, no clear ranking', 'BET, S\\&P 500 și DAX: intervale care se suprapun, fără un clasament clar'),
    T('$\\hat\\gamma$ of Bitcoin is more than twice that of the indices', '$\\hat\\gamma$ al Bitcoin este de peste două ori mai mare decît cel al indicilor')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Maximum Likelihood for the S\\&P 500 [Solved]', 'B3: verosimilitate maximă pentru S\\&P 500 [Rezolvat]'),
       T('does the ML stable fit describe the S\\&P 500 better than the Normal distribution, in the body and in the tails?',
         'descrie ajustarea stabilă ML randamentele S\\&P 500 mai bine decît distribuția Normală, în corp și în cozi?'),
       T('S\\&P 500 closes since @{b2.sp500.y0}, daily log returns in \\%', 'închiderile S\\&P 500 din @{b2.sp500.y0}, randamente logaritmice zilnice în \\%'),
       [T('Fit the stable law by ML (S0) and report $\\hat\\alpha$ and $\\hat\\beta$ with standard errors, $\\hat\\gamma$, $\\hat\\delta_0$ and $\\hat\\delta_1$.',
          'Ajustați legea stabilă prin ML (S0) și raportați $\\hat\\alpha$ și $\\hat\\beta$ cu erorile standard, $\\hat\\gamma$, $\\hat\\delta_0$ și $\\hat\\delta_1$.'),
        T('Compare $\\hat\\alpha$ with the McCulloch estimate.', 'Comparați $\\hat\\alpha$ cu estimarea McCulloch.'),
        T('Fit the Normal and the Student-t distributions and compare the AIC (Akaike information criterion, $2k - 2\\ell$) of the three models.',
          'Ajustați distribuția Normală și distribuția Student-t și comparați AIC (Akaike information criterion, criteriul informațional Akaike, $2k - 2\\ell$) al celor trei modele.'),
        T('Draw the QQ plot of the data against the Normal and the stable fits.', 'Desenați graficul QQ al datelor față de ajustările Normală și stabilă.'),
        T('Interpretation: where does each model fail?', 'Interpretare: unde eșuează fiecare model?')],
       T('the estimates, three AIC values, the QQ plot and two sentences', 'estimările, trei valori AIC, graficul QQ și două fraze'), size='scriptsize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch3_sem_b3', h='0.40') + items(
    T('ML: $\\hat\\alpha = @{b3.alpha}$ (SE $@{b3.se_alpha}$, 95\\% CI $[@{b3.lo}, @{b3.hi}]$), $\\hat\\beta = @{b3.beta}$ ($@{b3.se_beta}$), $\\hat\\gamma = @{b3.gamma}$, $\\hat\\delta_0 = @{b3.delta0}$, $\\hat\\delta_1 = @{b3.delta1}$; McCulloch $\\hat\\alpha = @{b3.mcc_alpha}$',
      'ML: $\\hat\\alpha = @{b3.alpha}$ (SE $@{b3.se_alpha}$, CI 95\\% $[@{b3.lo}, @{b3.hi}]$), $\\hat\\beta = @{b3.beta}$ ($@{b3.se_beta}$), $\\hat\\gamma = @{b3.gamma}$, $\\hat\\delta_0 = @{b3.delta0}$, $\\hat\\delta_1 = @{b3.delta1}$; McCulloch $\\hat\\alpha = @{b3.mcc_alpha}$'),
    T('AIC: Normal @{b3.aic_normal}, Student-t @{b3.aic_t}, stable @{b3.aic_stable}: the stable law beats the Normal fit by @{b3.aic.ns} points, the Student-t beats the stable law by @{b3.aic.st}',
      'AIC: Normală @{b3.aic_normal}, Student-t @{b3.aic_t}, stabilă @{b3.aic_stable}: legea stabilă bate distribuția Normală la @{b3.aic.ns} de puncte, Student-t bate legea stabilă la @{b3.aic.st} de puncte'),
    T('Interpretation: the Normal fit misses the tails (0.1\\% quantile $@{b3.q001_normal}$ vs $@{b3.q001_emp}$ in the data); the stable fit overshoots them ($@{b3.q001_stable}$)',
      'Interpretare: ajustarea Normală ratează cozile (cuantila de 0,1\\% $@{b3.q001_normal}$ vs $@{b3.q001_emp}$ în date); ajustarea stabilă le depășește ($@{b3.q001_stable}$)')) + qlsem(), 'scriptsize')

D.task(T('B4: Large Bitcoin Losses under Three Models [Proposed]', 'B4: pierderi mari Bitcoin în trei modele [Propus]'),
       T('which model predicts the number of large daily Bitcoin losses best? Model: B3.', 'care model reproduce cel mai bine numărul pierderilor zilnice mari ale Bitcoin? Model: B3.'),
       T('Bitcoin closes since @{b2.btc.y0}, 7 days a week, daily log returns in \\%', 'închiderile Bitcoin din @{b2.btc.y0}, 7 zile pe săptămînă, randamente logaritmice zilnice în \\%'),
       [T('Fit the Normal, Student-t and stable (ML) laws.', 'Ajustați legile Normală, Student-t și stabilă (ML).'),
        T('Count the days with a loss above 10\\% and above 20\\%.', 'Numărați zilele cu o pierdere peste 10\\% și peste 20\\%.'),
        T('Compute the expected number of such days under each model, $n \\times P(r < -x)$.', 'Calculați numărul așteptat de astfel de zile în fiecare model, $n \\times P(r < -x)$.'),
        T('Compute VaR 1\\% of the data and of each model.', 'Calculați VaR 1\\% al datelor și al fiecărui model.'),
        T('Interpretation: which model would you use for a margin requirement on a Bitcoin position, and why?', 'Interpretare: ce model ați folosi pentru o cerință de marjă pe o poziție în Bitcoin și de ce?')],
       T('one table of counts, four VaR values and two sentences', 'un tabel de numărători, patru valori VaR și două fraze'), size='scriptsize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrr', T('& observed & Normal & Student-t & stable', '& observat & Normală & Student-t & stabilă'),
    [T('loss above 10\\% (days)', 'pierdere peste 10\\% (zile)') + ' & @{b4.obs10} & $@{b4.n10}$ & $@{b4.t10}$ & $@{b4.s10}$',
     T('loss above 20\\% (days)', 'pierdere peste 20\\% (zile)') + ' & @{b4.obs20} & $@{b4.n20}$ & $@{b4.t20}$ & $@{b4.s20}$',
     'VaR 1\\% (\\%) & $@{b4.var.e}$ & $@{b4.var.n}$ & $@{b4.var.t}$ & $@{b4.var.s}$'], size='footnotesize') + items(
    T('$n = @{b4.n}$; stable ML: $\\hat\\alpha = @{b4.alpha}$ ($@{b4.se_alpha}$), $\\hat\\beta = @{b4.beta}$, $\\hat\\gamma = @{b4.gamma}$; Student-t $\\hat\\nu = @{b4.t_nu}$',
      '$n = @{b4.n}$; stabilă ML: $\\hat\\alpha = @{b4.alpha}$ ($@{b4.se_alpha}$), $\\hat\\beta = @{b4.beta}$, $\\hat\\gamma = @{b4.gamma}$; Student-t $\\hat\\nu = @{b4.t_nu}$'),
    T('Normal: far too few large losses; stable: too many beyond 20\\%; Student-t: closest at 10\\%, too many at 20\\%',
      'Normală: mult prea puține pierderi mari; stabilă: prea multe peste 20\\%; Student-t: cea mai apropiată la 10\\%, prea multe la 20\\%'),
    T('Interpretation: Student-t, or the stable law as a conservative bound; never the Normal distribution', 'Interpretare: Student-t sau legea stabilă ca limită prudentă; niciodată distribuția Normală')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Does $\\alpha$ Stay the Same When Returns Are Summed? [Solved]', 'B5: rămîne $\\alpha$ la fel cînd randamentele se adună? [Rezolvat]'),
       T('if S\\&P 500 daily returns were i.i.d.\\ stable, weekly and monthly returns would have the same $\\alpha$; do they?',
         'dacă randamentele zilnice S\\&P 500 ar fi i.i.d.\\ stabile, randamentele săptămînale și lunare ar avea același $\\alpha$; îl au?'),
       T('S\\&P 500 daily log returns since @{b2.sp500.y0}; weekly and monthly sums', 'randamentele logaritmice zilnice S\\&P 500 din @{b2.sp500.y0}; sumele săptămînale și lunare'),
       [T('Compute the weekly and monthly log returns as sums of the daily ones.', 'Calculați randamentele logaritmice săptămînale și lunare ca sume ale celor zilnice.'),
        T('Compute the McCulloch $\\hat\\alpha$ at the three horizons.', 'Calculați $\\hat\\alpha$ McCulloch la cele trei orizonturi.'),
        T('Bootstrap each $\\hat\\alpha$ with 500 resamples and report the 95\\% intervals.', 'Faceți bootstrap pentru fiecare $\\hat\\alpha$ cu 500 de reeșantionări și raportați intervalele de 95\\%.'),
        T('Interpretation: is the pattern consistent with i.i.d.\\ stable returns?', 'Interpretare: este tiparul compatibil cu randamente stabile i.i.d.?')],
       T('three estimates with intervals, the chart and two sentences', 'trei estimări cu intervale, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch3_sem_b5', h='0.42') + items(
    T('Daily ($n = @{b5.daily.n}$): $@{b5.daily.alpha}$ $[@{b5.daily.lo}, @{b5.daily.hi}]$; weekly ($@{b5.weekly.n}$): $@{b5.weekly.alpha}$ $[@{b5.weekly.lo}, @{b5.weekly.hi}]$; monthly ($@{b5.monthly.n}$): $@{b5.monthly.alpha}$ $[@{b5.monthly.lo}, @{b5.monthly.hi}]$',
      'Zilnic ($n = @{b5.daily.n}$): $@{b5.daily.alpha}$ $[@{b5.daily.lo}, @{b5.daily.hi}]$; săptămînal ($@{b5.weekly.n}$): $@{b5.weekly.alpha}$ $[@{b5.weekly.lo}, @{b5.weekly.hi}]$; lunar ($@{b5.monthly.n}$): $@{b5.monthly.alpha}$ $[@{b5.monthly.lo}, @{b5.monthly.hi}]$'),
    T('$\\hat\\alpha$ is higher for weekly than for daily returns; the monthly interval is wide and reaches up to $@{b5.monthly.hi}$',
      '$\\hat\\alpha$ este mai mare pentru randamentele săptămînale decît pentru cele zilnice; intervalul lunar este larg și urcă pînă la $@{b5.monthly.hi}$'),
    T('Interpretation: a rising $\\alpha$ is a warning sign against i.i.d.\\ stable returns; monthly samples are too short for a firm answer (Lecture 3 repeats the test with ML)',
      'Interpretare: un $\\alpha$ crescător este un semnal împotriva randamentelor stabile i.i.d.; eșantioanele lunare sînt prea scurte pentru un răspuns ferm (Cursul 3 repetă testul cu ML)')) + qlsem(), 'footnotesize')

D.task(T('B6: Does the Sample Variance Settle? [Proposed]', 'B6: se stabilizează varianța de selecție? [Propus]'),
       T('is the sample variance of BET and Bitcoin returns as large and as unstable as a stable law with the fitted $\\alpha$ implies? Models: B1, B5.',
         'este varianța de selecție a randamentelor BET și Bitcoin atît de mare și de instabilă cît implică o lege stabilă cu $\\alpha$ estimat? Modele: B1, B5.'),
       T('BET since @{b1.y0}, Bitcoin since @{b2.btc.y0}, daily log returns in \\%', 'BET din @{b1.y0}, Bitcoin din @{b2.btc.y0}, randamente logaritmice zilnice în \\%'),
       [T('Compute the sample variance of each series over the full sample and over its first half.', 'Calculați varianța de selecție a fiecărei serii pe tot eșantionul și pe prima jumătate.'),
        T('Simulate 200 paths of the same length from the McCulloch fit (CMS) and compute the sample variance of each path.',
          'Simulați 200 de traiectorii de aceeași lungime din ajustarea McCulloch (CMS) și calculați varianța de selecție a fiecărei traiectorii.'),
        T('Report the median and the 5\\% and 95\\% percentiles of the simulated variances, and the share below the observed one.',
          'Raportați mediana și percentilele de 5\\% și 95\\% ale varianțelor simulate, precum și proporția celor sub varianța observată.'),
        T('Interpretation: what does the comparison say about the infinite-variance hypothesis?', 'Interpretare: ce spune comparația despre ipoteza varianței infinite?')],
       T('a table with two rows and one sentence', 'un tabel cu două rînduri și o frază'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{NAMES[k]} & $@{{b6.{k}.alpha}}$ & $@{{b6.{k}.var_data}}$ & $@{{b6.{k}.var_first_half}}$ & $@{{b6.{k}.sim_median}}$ & '
            f'$[@{{b6.{k}.sim_q05}}, @{{b6.{k}.sim_q95}}]$ & $@{{b6.{k}.share}}\\%$')


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrr', T('& $\\hat\\alpha$ & var. data & first half & sim. median & sim. 5--95\\% & share below',
                 '& $\\hat\\alpha$ & var. date & prima jumătate & mediana sim. & sim. 5--95\\% & proporția sub'),
    [b6row('bet'), b6row('btc')], size='footnotesize') + items(
    T('The observed variance is below every simulated one: real returns are far less extreme than the fitted stable law implies',
      'Varianța observată este sub toate varianțele simulate: randamentele reale sînt mult mai puțin extreme decît implică legea stabilă ajustată'),
    T('Interpretation: evidence against infinite variance; the variance of the first half differs from that of the full sample because volatility changes over time',
      'Interpretare: dovezi împotriva varianței infinite; varianța primei jumătăți diferă de cea a eșantionului complet pentru că volatilitatea se schimbă în timp')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Is $\\alpha$ Constant over Time? [Proposed]', 'C1: este $\\alpha$ constant în timp? [Propus]'),
       T('do the tails of the BET and the S\\&P 500 change from one five-year period to the next?', 'se schimbă cozile BET și S\\&P 500 de la o perioadă de cinci ani la alta?'),
       T('daily log returns since @{b1.y0}, split into 2000--2004, 2005--2009, 2010--2014, 2015--2019, 2020--@{b1.y1}; models: B1, B3',
         'randamente logaritmice zilnice din @{b1.y0}, împărțite în 2000--2004, 2005--2009, 2010--2014, 2015--2019, 2020--@{b1.y1}; modele: B1, B3'),
       [T('Compute the McCulloch $\\hat\\alpha$ with a bootstrap interval in each period, for each index.', 'Calculați $\\hat\\alpha$ McCulloch cu un interval bootstrap în fiecare perioadă, pentru fiecare indice.'),
        T('Say which periods differ significantly.', 'Spuneți care perioade diferă semnificativ.'),
        T('Propose one explanation for the changes and one way to test it (for example, returns divided by a rolling volatility).',
          'Propuneți o explicație a schimbărilor și o cale de a o testa (de exemplu, randamente împărțite la o volatilitate mobilă).'),
        T('Interpretation: what would a changing $\\alpha$ mean for a risk model estimated on old data?', 'Interpretare: ce ar însemna un $\\alpha$ variabil pentru un model de risc estimat pe date vechi?')],
       T('a table, a short list of significant changes and a plan for a project', 'un tabel, o scurtă listă de schimbări semnificative și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'llrrr', T('Index & period & $n$ & $\\hat\\alpha$ & 95\\% interval', 'Indice & perioada & $n$ & $\\hat\\alpha$ & interval 95\\%'), c1rows, size='scriptsize') + items(
    T('S\\&P 500: $\\hat\\alpha$ between $@{c1.sp.min}$ and $@{c1.sp.max}$, lowest in @{c1.sp.low} (the global financial crisis); several intervals do not overlap',
      'S\\&P 500: $\\hat\\alpha$ între $@{c1.sp.min}$ și $@{c1.sp.max}$, cel mai mic în @{c1.sp.low} (criza financiară globală); mai multe intervale nu se suprapun'),
    T('BET: between $@{c1.bet.min}$ and $@{c1.bet.max}$; the changes follow volatility regimes, a sign of volatility clustering (Chapter 9)',
      'BET: între $@{c1.bet.min}$ și $@{c1.bet.max}$; schimbările urmează regimurile de volatilitate, un semn al grupării volatilității (Capitolul 9)'),
    T('Project directions: rolling windows, ML, returns standardised by GARCH volatility, more BVB stocks', 'Direcții de proiect: ferestre mobile, ML, randamente standardizate cu volatilitatea GARCH, mai multe acțiuni BVB')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to fit a stable law to the daily S\\&P 500 log returns (\\%) since 2000. The answer:',
      'Un student a cerut unui asistent AI să ajusteze o lege stabilă pe randamentele logaritmice zilnice S\\&P 500 (\\%) din 2000. Răspunsul:'),
    T('\\aiprompt{(a) scipy\'s levy\\_stable uses S0 by default, so loc = @{c2.delta1} is the S0 location delta0.}',
      '\\aiprompt{(a) levy\\_stable din scipy folosește implicit S0, deci loc = @{c2.delta1} este locația S0 delta0.}'),
    T('\\aiprompt{(b) With alpha = @{c2.alpha} and gamma = @{c2.gamma}, the variance of daily returns is 2 gamma\\textasciicircum 2 = @{c2.two_gamma2}.}',
      '\\aiprompt{(b) Cu alpha = @{c2.alpha} și gamma = @{c2.gamma}, varianța randamentelor zilnice este 2 gamma\\textasciicircum 2 = @{c2.two_gamma2}.}'),
    T('\\aiprompt{(c) The scale of 20-day returns is sqrt(20) x gamma = @{c2.scale20_wrong}\\%.}',
      '\\aiprompt{(c) Scala randamentelor pe 20 de zile este sqrt(20) x gamma = @{c2.scale20_wrong}\\%.}'),
    T('\\aiprompt{(d) Since alpha < 2, the VaR 1\\% of this model does not exist.}', '\\aiprompt{(d) Deoarece alpha < 2, VaR 1\\% al acestui model nu există.}'),
    T('\\aiprompt{(e) Since alpha > 1, the mean daily return exists.}', '\\aiprompt{(e) Deoarece alpha > 1, randamentul zilnic mediu există.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație spuneți dacă este corectă; dacă nu, dați afirmația corectă și, unde se poate, cifra corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: scipy uses S1 by default; $\\hat\\delta_1 = @{c2.delta1}$ and $\\hat\\delta_0 = \\hat\\delta_1 + \\hat\\beta\\hat\\gamma\\tan\\frac{\\pi\\hat\\alpha}{2} = @{c2.delta0}$',
      '(a) Greșit: scipy folosește implicit S1; $\\hat\\delta_1 = @{c2.delta1}$ și $\\hat\\delta_0 = \\hat\\delta_1 + \\hat\\beta\\hat\\gamma\\tan\\frac{\\pi\\hat\\alpha}{2} = @{c2.delta0}$'),
    T('(b) Wrong: $2\\gamma^2$ is the variance only for $\\alpha = 2$; with $\\alpha < 2$ the model variance is infinite; the sample variance is $@{c2.sample_var}$',
      '(b) Greșit: $2\\gamma^2$ este varianța doar pentru $\\alpha = 2$; cu $\\alpha < 2$, varianța modelului este infinită; varianța de selecție este $@{c2.sample_var}$'),
    T('(c) Wrong: under the stable model the scale grows like $20^{1/\\alpha}$: $@{c2.f20} \\times @{c2.gamma} = @{c2.scale20}\\%$, not $\\sqrt{20} = @{c2.sqrt20}$ times',
      '(c) Greșit: în modelul stabil, scala crește ca $20^{1/\\alpha}$: $@{c2.f20} \\times @{c2.gamma} = @{c2.scale20}\\%$, nu de $\\sqrt{20} = @{c2.sqrt20}$ ori'),
    T('(d) Wrong: quantiles always exist; VaR 1\\% of the fitted stable law is $@{c2.var1_stable}\\%$ (it is ES that needs $\\alpha > 1$, which holds here)',
      '(d) Greșit: cuantilele există întotdeauna; VaR 1\\% al legii stabile ajustate este $@{c2.var1_stable}\\%$ (ES este cel care cere $\\alpha > 1$, ceea ce este adevărat aici)'),
    T('(e) Correct: $E|X| < \\infty$ because $1 < \\alpha$; in S1 the mean equals $\\delta_1$', '(e) Corect: $E|X| < \\infty$ pentru că $1 < \\alpha$; în S1, media este egală cu $\\delta_1$')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Ce rămîne de azi'), items(
    T('A stable sum of $n$ terms scales like $n^{1/\\alpha}$, faster than $\\sqrt{n}$ when $\\alpha < 2$', 'O sumă stabilă de $n$ termeni se scalează ca $n^{1/\\alpha}$, mai repede decît $\\sqrt{n}$ cînd $\\alpha < 2$'),
    T('Know the parameterisation: S0 and S1 differ in the location; scipy uses S1', 'Cunoașteți parametrizarea: S0 și S1 diferă prin locație; scipy folosește S1'),
    T('McCulloch: five quantiles and a table; ML: more precise; bootstrap for intervals', 'McCulloch: cinci cuantile și un tabel; ML: mai precisă; bootstrap pentru intervale'),
    T('Real returns: $\\hat\\alpha$ well below 2 at the daily horizon, yet the variance settles and $\\hat\\alpha$ rises with aggregation',
      'Randamentele reale: $\\hat\\alpha$ mult sub 2 la orizont zilnic, dar varianța se stabilizează, iar $\\hat\\alpha$ crește prin agregare'),
    T('An AI answer is a draft: check the parameterisation, the scale rule and what exists for $\\alpha < 2$',
      'Un răspuns AI este o ciornă: verificați parametrizarea, regula scalei și ce există pentru $\\alpha < 2$')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 3 develops each topic of today: stability, the generalised CLT, parameters, tails, simulation, estimation and the critique',
      'Cursul 3 dezvoltă fiecare temă de azi: stabilitatea, CLT generalizată, parametrii, cozile, simularea, estimarea și critica'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: rolling windows, ML, more markets', 'C1 poate deveni un proiect de echipă: ferestre mobile, ML, mai multe piețe'),
    T('Reading: \\refNolan, Ch.~1; \\refBHW; \\refMcCulloch', 'Lectură: \\refNolan, cap.~1; \\refBHW; \\refMcCulloch')))

D.references(bib(['BHW', 'CMS', 'FHH', 'Mandelbrot', 'McCulloch', 'Nolan', 'NolanML', 'Weron']))

if __name__ == '__main__':
    for path in D.write(V):
        ro_tuples(path)
