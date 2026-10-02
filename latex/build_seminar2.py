r"""
build_seminar2.py -- Seminarul 2 (Distribuții clasice și fapte stilizate), EN + RO dintr-o singură sursă
======================================================================================================
Seminarul are loc ÎNAINTEA cursului 2: secțiunea „Ce vă trebuie azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A derivări și calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o
întrebare deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea
doar în versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_02/sem2_results.json (seminar2.py).
Ieșire:
  EN/Seminars/seminar2_classical_distributions_stylised_facts.tex          (+ _solutions.tex)
  RO/Seminarii/seminar2_distributii_clasice_fapte_stilizate_ro.tex         (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_02/seminar2.py && python3 latex/build_seminar2.py && python3 latex/sfm_build.py compile 2
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, fig   # noqa: E402
from ch2_common import REFS, T, big, load_sem, put_date, sci   # noqa: E402

S = load_sem()
V = Values()
D = Deck(2, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_02}{SFM_ch2_seminar}'


def p_or_sci(x, d=2):
    if x < 1e-300:
        return '\\approx 0'
    return sci(x, 1) if x < 0.001 else '⁅' + f'{x:.{d}f}' + '⁆'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for k in ['a1', 'a2']:
    a = A[k]
    V.put(f'{k}.z', a['z'], 2)
    V.put(f'{k}.p', 100 * a['p_loss'], 2)
    V.put(f'{k}.dy', a['days_loss_year'], 1)
    V.put(f'{k}.q01', a['q01'], 2)
    V.raw(f'{k}.p4', sci(a['p4'], 1))
    V.put(f'{k}.exp4', a['exp4'], 2)
    V.raw(f'{k}.nd', big(a['n_days']))
    V.put(f'{k}.z01', a['z01'], 3)
for T_ in ['T1', 'T10']:
    a = A['a3'][T_]
    for c, d in [('median', 2), ('mean', 2), ('q', 2), ('sT', 3)]:
        V.put(f'a3.{T_}.{c}', a[c], d)
    V.put(f'a3.{T_}.ploss', 100 * a['p_loss'], 1)
    V.put(f'a3.{T_}.zl', -a['mT'] / a['sT'], 3)
V.put('a3.z05', A['a3']['T1']['z'], 3)
V.put('a4.mean', A['a4']['mean'], 4)
V.put('a4.median', A['a4']['median'], 4)
V.put('a4.drag', 100 * A['a4']['drag'], 2)
V.put('a4.var', A['a4']['var'], 4)
a5 = A['a5']
for c, d in [('mu', 2), ('sd', 2), ('z', 3)]:
    V.put(f'a5.{c}', a5[c], d)
V.put('a5.approx', 100 * a5['approx'], 2)
V.put('a5.exact', 100 * a5['exact'], 2)
a6 = A['a6']
for h in ['h21', 'h5']:
    V.put(f'a6.{h}.mean', a6[h]['mean'], 2)
    V.put(f'a6.{h}.sd', a6[h]['sd'], 2)
    V.put(f'a6.{h}.z', a6[h]['z'], 2)
V.put('a6.h21.p', 100 * a6['h21']['p'], 2)
V.raw('a6.h5.p', sci(a6['h5']['p'], 1))
a7 = A['a7']
V.raw('a7.jb', big(a7['jb']))
V.put('a7.crit', a7['crit'], 2)
V.put('a7.se_s', a7['se_s'], 4)
V.put('a7.se_k', a7['se_k'], 4)
V.put('a7.z_s', a7['z_s'], 1)
V.put('a7.z_k', a7['z_k'], 1)
V.raw('a7.ps', big(a7['part_s']))
V.raw('a7.pk', big(a7['part_k']))
a8 = A['a8']
for k in ['t5', 't6', 't10']:
    V.put(f'a8.{k}.var', a8[k]['var'], 3)
    V.put(f'a8.{k}.ek', a8[k]['exkurt'], 1)
    V.put(f'a8.{k}.scale', a8[k]['scale'], 3)
    V.put(f'a8.{k}.p4', 100 * a8[k]['p4'], 3)
V.put('a8.nu3', a8['nu_for_3'], 0)
V.put('a8.normal.p4', 100 * A['a1']['p4'], 4)
V.put('a8.ratio', a8['t5']['p4'] / A['a1']['p4'], 0)

B1 = S['B1']
V.int('b1.n', B1['n'])
for c, d in [('mean', 3), ('sd', 2), ('skew', 2), ('exkurt', 2), ('se_skew', 3), ('se_kurt', 3), ('min', 1), ('max', 1),
             ('bs_s_lo', 2), ('bs_s_hi', 2), ('bs_k_lo', 1), ('bs_k_hi', 1), ('n_lo_k', 2), ('n_hi_k', 2), ('bs_k_sd', 2)]:
    V.put(f'b1.{c}', B1[c], d)
V.raw('b1.jb', big(B1['jb']))
put_date(V, 'b1.mindate', B1['min_date'])
V.put('b1.ratio', B1['bs_k_sd'] / B1['se_kurt'], 0)
V.put('b1.zs', B1['skew'] / B1['se_skew'], 1)
V.put('b1.zk', B1['exkurt'] / B1['se_kurt'], 0)
B2 = S['B2']
for k, b in B2.items():
    V.int(f'b2.{k}.n', b['n'])
    for c, d in [('mean', 3), ('sd', 2), ('skew', 2), ('exkurt', 1), ('exkurt_drop1', 1), ('drop_r', 1)]:
        V.put(f'b2.{k}.{c}', b[c], d)
    V.raw(f'b2.{k}.jb', big(b['jb']))
    put_date(V, f'b2.{k}.drop', b['drop_date'])
B3 = S['B3']
for c, d in [('nu', 2), ('loc', 3), ('scale', 3), ('sd_t', 2), ('min', 1), ('q_t_min', 1), ('q_n_min', 1), ('daic', 0), ('ll_t', 0), ('ll_n', 0)]:
    V.put(f'b3.{c}', B3[c], d)
V.int('b3.n', B3['n'])
B4 = S['B4']
for k, b in B4.items():
    V.raw(f'b4.{k}.obs', str(b['obs']))
    V.put(f'b4.{k}.en', b['exp_n'], 2)
    V.put(f'b4.{k}.et', b['exp_t'], 1)
    V.put(f'b4.{k}.nu', b['nu'], 2)
    V.raw(f'b4.{k}.pp', sci(b['poisson_p'], 0))
    V.int(f'b4.{k}.n', b['n'])
B5 = S['B5']
for h in ['1', '5', '21']:
    V.raw(f'b5.{h}.n', str(B5[h]['n']))
    V.put(f'b5.{h}.skew', B5[h]['skew'], 2)
    V.put(f'b5.{h}.ek', B5[h]['exkurt'], 2)
    V.raw(f'b5.{h}.jb', big(B5[h]['jb']))
    V.raw(f'b5.{h}.p', p_or_sci(B5[h]['jb_p']))
    V.put(f'b5.{h}.sek', B5[h]['se_k'], 2)
B6 = S['B6']
V.int('b6.n', B6['n'])
V.put('b6.band', B6['band'], 3)
for i in range(3):
    V.put(f'b6.r{i + 1}', B6['rho_r'][i], 3)
    V.put(f'b6.a{i + 1}', B6['rho_abs'][i], 3)
V.put('b6.a10', B6['rho_abs'][9], 3)
V.put('b6.qr', B6['lb_r']['Q'], 1)
V.put('b6.qa', B6['lb_abs']['Q'], 0)
V.raw('b6.pr', p_or_sci(B6['lb_r']['p'], 3))
V.put('b6.crit', B6['crit'], 2)
V.raw('b6.nout', str(sum(abs(x) > B6['band'] for x in B6['rho_r'])))
B7 = S['B7']
for k, b in B7.items():
    V.put(f'b7.{k}.L1', b['L'][0], 3)
    V.put(f'b7.{k}.Lm', b['Lmean'], 3)
    V.put(f'b7.{k}.band', b['band'], 3)
    V.raw(f'b7.{k}.d', str(b['down']))
    V.raw(f'b7.{k}.u', str(b['up']))
    V.put(f'b7.{k}.bp', b['binom_p'], 2)
    V.put(f'b7.{k}.skew', b['skew'], 2)
    V.put(f'b7.{k}.q01', -b['q01'], 2)
    V.put(f'b7.{k}.q99', b['q99'], 2)
C1 = S['C1']
for k, c in C1.items():
    V.put(f'c1.{k}.rho', c['spearman'], 2)
    V.put(f'c1.{k}.p', c['p'], 2)
    V.put(f'c1.{k}.nu1', c['nu_first'], 2)
    V.put(f'c1.{k}.nu2', c['nu_second'], 2)
    V.raw(f'c1.{k}.ny', str(len(c['rows'])))
nub = [r['nu'] for r in C1['btc']['rows'].values()]
V.put('c1.btc.numin', min(nub), 1)
V.put('c1.btc.numax', max(nub), 1)
C2 = S['C2']
V.put('c2.exkurt', C2['exkurt'], 1)
V.put('c2.kurt', C2['kurt'], 1)
V.raw('c2.jb', big(C2['jb']))
V.raw('c2.obs4', str(C2['obs4']))
V.put('c2.exp4', C2['exp4'], 2)
V.int('c2.n', C2['n'])
V.raw('c2.every', big(C2['every_normal']))
V.raw('c2.pp', sci(C2['poisson_p'], 0))
V.put('c2.r1', C2['rho1_r'], 2)
V.put('c2.a1', C2['rho1_abs'], 2)
V.put('c2.a10', C2['rho10_abs'], 2)
V.put('c2.lba', C2['lb_abs'], 0)
V.put('c2.med', C2['median1'], 2)
V.put('c2.mean', C2['mean1'], 2)
V.put('c2.nu', C2['nu'], 2)

# cifre derivate folosite în interpretări
rat = [b['obs'] / b['exp_n'] for b in B4.values()]
V.put('b4.rmin', min(rat), 0)
V.put('b4.rmax', max(rat), 0)
V.put('b7.sp500.times', abs(B7['sp500']['Lmean']) / B7['sp500']['band'], 0)
V.put('b6.max', max(abs(x) for x in B6['rho_r']), 2)
V.put('a5.v', 252 * 0.53 * 0.47, 0)
V.put('a3.half', 0.18 ** 2 / 2, 4)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how far are real daily returns from the Normal distribution, and how do we measure it?',
       '\\textbf{Întrebarea}: cît de departe sînt randamentele zilnice reale de distribuția Normală și cum măsurăm acest lucru?'),
     [T('this seminar comes \\textbf{before} Lecture 2: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 2: secțiunea „Ce vă trebuie azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: Normal, lognormal and Student-$t$ computations and derivations on paper', 'Partea A: calcule și derivări pe hîrtie cu distribuțiile Normală, lognormală și Student-$t$'),
      T('Part B: moments, tests, QQ plots and stylised facts on real data, each with an interpretation question',
        'Partea B: momente, teste, QQ plots și fapte stilizate pe date reale, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Nu se predă nimic: seminarul este pentru exercițiu; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.0cm}' + TB + 'p{7.6cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('how likely is a large daily loss under the Normal distribution?', 'cît de probabilă este o pierdere zilnică mare în distribuția Normală?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A1',
     'A3, A4 & ' + T('what is the price after one and ten years under lognormal prices?', 'care este prețul după unu și zece ani cu prețuri lognormale?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A3',
     'A5, A6 & ' + T('when is a sum approximately Normal (CLT)?', 'cînd este o sumă aproximativ Normală (CLT)?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A5',
     'A7, A8 & ' + T('the Jarque--Bera statistic and the moments of a Student-$t$', 'statistica Jarque--Bera și momentele unei distribuții Student-$t$') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A7',
     'B1, B2 & ' + T('how non-Normal are daily returns, and how precise is the kurtosis?', 'cît de departe de normalitate sînt randamentele zilnice și cît de precisă este aplatizarea?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B1',
     'B3, B4 & ' + T('does a Student-$t$ fit better? how many 4-sigma days?', 'se potrivește mai bine o distribuție Student-$t$? cîte zile de 4 sigma?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B3',
     'B5--B7 & ' + T('aggregational Gaussianity, volatility clustering, leverage and asymmetry', 'gaussianitate agregată, volatility clustering, efect de levier și asimetrie') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B5',
     'C1, C2 & ' + T('are Bitcoin\'s tails getting thinner? what is wrong in an AI answer?', 'devin mai subțiri cozile Bitcoin? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3, B5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX, BET & EODHD & close & 2010--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & 2014--2026',
     T('OMV Petrom (SNP)', 'OMV Petrom (SNP)') + ' & EODHD & adjusted close & 2010--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$', 'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$'),
    T('Weekends and repeated holiday closes are dropped (except Bitcoin); each series on its own calendar',
      'Weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); fiecare serie pe calendarul ei'),
    T('In the notebook: \\texttt{returns(\'sp500\')}, \\texttt{moments(r)}, \\texttt{t\\_fit(r)}; no account or key is needed',
      'În notebook: \\texttt{returns(\'sp500\')}, \\texttt{moments(r)}, \\texttt{t\\_fit(r)}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Ce vă trebuie azi')

D.frame(T('What You Need for Today (1/4): the Normal distribution Distribution', 'Ce vă trebuie azi (1/4): distribuția Normală'), items(
    T('$X \\sim N(\\mu, \\sigma^2)$: density $f(x) = (\\sigma\\sqrt{2\\pi})^{-1}\\exp\\big(-(x-\\mu)^2/(2\\sigma^2)\\big)$, mean $\\mu$, variance $\\sigma^2$',
      '$X \\sim N(\\mu, \\sigma^2)$: densitatea $f(x) = (\\sigma\\sqrt{2\\pi})^{-1}\\exp\\big(-(x-\\mu)^2/(2\\sigma^2)\\big)$, media $\\mu$, varianța $\\sigma^2$'),
    (T('Standardise: $Z = (X - \\mu)/\\sigma \\sim N(0,1)$; $P(X \\le x) = \\Phi\\big((x - \\mu)/\\sigma\\big)$, $\\Phi$ = standard Normal CDF (cumulative distribution function)',
       'Standardizare: $Z = (X - \\mu)/\\sigma \\sim N(0,1)$; $P(X \\le x) = \\Phi\\big((x - \\mu)/\\sigma\\big)$, $\\Phi$ = CDF (cumulative distribution function, funcția de repartiție) Normală standard'),
     [T('useful values: $\\Phi(-1.645) = 0.05$, $\\Phi(-1.96) = 0.025$, $\\Phi(-2.326) = 0.01$; $P(|Z| > 4) = @{a1.p4}$', 'valori utile: $\\Phi(-1.645) = 0.05$, $\\Phi(-1.96) = 0.025$, $\\Phi(-2.326) = 0.01$; $P(|Z| > 4) = @{a1.p4}$')]),
    T('Quantile of order $\\alpha$: $x_\\alpha = \\mu + \\sigma z_\\alpha$, with $z_\\alpha = \\Phi^{-1}(\\alpha)$', 'Cuantila de ordin $\\alpha$: $x_\\alpha = \\mu + \\sigma z_\\alpha$, cu $z_\\alpha = \\Phi^{-1}(\\alpha)$'),
    T('A \\textbf{$k$-sigma day}: $|r_t - \\bar r| > k\\,s$, with $\\bar r$ and $s$ the sample mean and standard deviation', 'O \\textbf{zi de $k$ sigma}: $|r_t - \\bar r| > k\\,s$, cu $\\bar r$ și $s$ media și abaterea standard de selecție'),
    T('Expected number of $k$-sigma days in $n$ days: $n \\cdot P(|Z| > k)$; the count is approximately Poisson with this mean',
      'Numărul așteptat de zile de $k$ sigma în $n$ zile: $n \\cdot P(|Z| > k)$; numărul este aproximativ Poisson cu această medie')), 'footnotesize')

D.frame(T('What You Need for Today (2/4): Lognormal Prices and the CLT', 'Ce vă trebuie azi (2/4): prețuri lognormale și CLT'), items(
    (T('$Y$ is \\textbf{lognormal}, $Y \\sim LN(m, s^2)$, if $\\ln Y \\sim N(m, s^2)$', '$Y$ este \\textbf{lognormală}, $Y \\sim LN(m, s^2)$, dacă $\\ln Y \\sim N(m, s^2)$'),
     [T('median $e^m$, mean $e^{m + s^2/2}$, $P(Y \\le y) = \\Phi\\big((\\ln y - m)/s\\big)$', 'mediana $e^m$, media $e^{m + s^2/2}$, $P(Y \\le y) = \\Phi\\big((\\ln y - m)/s\\big)$')]),
    T('If yearly log returns are i.i.d. $N(\\mu, \\sigma^2)$: $P_T = P_0 e^{r_1 + \\dots + r_T}$ with $\\ln(P_T/P_0) \\sim N(T\\mu, T\\sigma^2)$',
      'Dacă randamentele logaritmice anuale sînt i.i.d. $N(\\mu, \\sigma^2)$: $P_T = P_0 e^{r_1 + \\dots + r_T}$, cu $\\ln(P_T/P_0) \\sim N(T\\mu, T\\sigma^2)$'),
    (T('\\textbf{CLT} (central limit theorem): for i.i.d. $X_i$ with mean $\\mu$ and finite variance $\\sigma^2$, $X_1 + \\dots + X_n \\approx N(n\\mu, n\\sigma^2)$',
       '\\textbf{CLT} (central limit theorem, teorema limită centrală): pentru $X_i$ i.i.d. cu media $\\mu$ și varianța finită $\\sigma^2$, $X_1 + \\dots + X_n \\approx N(n\\mu, n\\sigma^2)$'),
     [T('binomial $B(n, p)$: $P(X \\ge k) \\approx 1 - \\Phi\\big((k - 0.5 - np)/\\sqrt{np(1-p)}\\big)$ (continuity correction)',
        'binomiala $B(n, p)$: $P(X \\ge k) \\approx 1 - \\Phi\\big((k - 0.5 - np)/\\sqrt{np(1-p)}\\big)$ (corecția de continuitate)')]),
    T('The three conditions: independence, identical distribution, finite variance', 'Cele trei condiții: independență, aceeași distribuție, varianță finită')), 'footnotesize')

D.frame(T('What You Need for Today (3/4): Shape, Tests and the Student-$t$', 'Ce vă trebuie azi (3/4): formă, teste și distribuția Student-$t$'), items(
    (T('Sample skewness $S = m_3/m_2^{3/2}$, sample excess kurtosis $K = m_4/m_2^2 - 3$, with $m_k = \\frac1n\\sum_t (r_t - \\bar r)^k$; Normal distribution: $S = K = 0$',
       'Asimetria de selecție $S = m_3/m_2^{3/2}$, excesul de aplatizare $K = m_4/m_2^2 - 3$, cu $m_k = \\frac1n\\sum_t (r_t - \\bar r)^k$; distribuția Normală: $S = K = 0$'),
     [T('under normality, standard errors $\\sqrt{6/n}$ and $\\sqrt{24/n}$', 'în ipoteza de normalitate, erorile standard $\\sqrt{6/n}$ și $\\sqrt{24/n}$')]),
    T('\\textbf{Jarque--Bera}: $\\text{JB} = \\frac{n}{6}(S^2 + K^2/4)$; under normality $\\chi^2(2)$; reject at 5\\% if $\\text{JB} > 5.99$',
      '\\textbf{Jarque--Bera}: $\\text{JB} = \\frac{n}{6}(S^2 + K^2/4)$; în ipoteza de normalitate $\\chi^2(2)$; respingem la 5\\% dacă $\\text{JB} > 5.99$'),
    T('\\textbf{QQ plot}: sorted data $r_{(i)}$ against model quantiles $F^{-1}((i - 0.5)/n)$; an S-shape means heavier tails than the model',
      '\\textbf{QQ plot}: datele ordonate $r_{(i)}$ față de cuantilele modelului $F^{-1}((i - 0.5)/n)$; o formă de S înseamnă cozi mai groase decît ale modelului'),
    (T('\\textbf{Student-$t(\\nu)$}: variance $\\nu/(\\nu-2)$ for $\\nu > 2$, excess kurtosis $6/(\\nu - 4)$ for $\\nu > 4$; for returns $r = m + s\\,T$',
       '\\textbf{Student-$t(\\nu)$}: varianța $\\nu/(\\nu-2)$ pentru $\\nu > 2$, excesul de aplatizare $6/(\\nu - 4)$ pentru $\\nu > 4$; pentru randamente $r = m + s\\,T$'),
     [T('fitted by MLE (maximum likelihood estimation); compared with the Normal distribution by AIC $= 2k - 2\\ell$ (lower is better)',
        'estimată prin MLE (maximum likelihood estimation, verosimilitate maximă); comparată cu distribuția Normală prin AIC $= 2k - 2\\ell$ (mai mic este mai bine)')])), 'footnotesize')

D.frame(T('What You Need for Today (4/4): Stylised Facts', 'Ce vă trebuie azi (4/4): fapte stilizate'), items(
    T('\\textbf{Stylised facts} \\refCont: statistical properties shared by most assets and periods', '\\textbf{Faptele stilizate} \\refCont: proprietăți statistice comune majorității activelor și perioadelor'),
    (T('\\textbf{ACF}: $\\hat\\rho(h) = \\text{Corr}(r_t, r_{t-h})$; 95\\% band under independence $\\pm 1.96/\\sqrt n$', '\\textbf{ACF}: $\\hat\\rho(h) = \\text{Corr}(r_t, r_{t-h})$; banda de 95\\% în ipoteza de independență $\\pm 1.96/\\sqrt n$'),
     [T('\\textbf{Ljung--Box}: $Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n-h)$, compared with $\\chi^2(m)$; $\\chi^2_{0.95}(10) = 18.31$',
        '\\textbf{Ljung--Box}: $Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n-h)$, comparat cu $\\chi^2(m)$; $\\chi^2_{0.95}(10) = 18.31$')]),
    T('No linear autocorrelation: $\\hat\\rho(h)$ of $r_t$ close to 0; \\textbf{volatility clustering}: $\\hat\\rho(h)$ of $|r_t|$ positive and slowly decaying',
      'Fără autocorelație liniară: $\\hat\\rho(h)$ pentru $r_t$ aproape de 0; \\textbf{volatility clustering}: $\\hat\\rho(h)$ pentru $|r_t|$ pozitivă și cu scădere lentă'),
    T('\\textbf{Aggregational Gaussianity}: $h$-day returns (sums of $h$ daily log returns) are closer to the Normal distribution as $h$ grows',
      '\\textbf{Gaussianitatea agregată}: randamentele pe $h$ zile (sume de $h$ randamente zilnice) sînt mai apropiate de distribuția Normală cînd $h$ crește'),
    T('\\textbf{Leverage effect}: $L(k) = \\text{Corr}(r_t, |r_{t+k}|) < 0$; \\textbf{gain/loss asymmetry}: large losses bigger and more frequent than large gains',
      '\\textbf{Efectul de levier}: $L(k) = \\text{Corr}(r_t, |r_{t+k}|) < 0$; \\textbf{asimetria cîștig/pierdere}: pierderile mari mai mari și mai frecvente decît cîștigurile mari'),
    T('\\textbf{Bootstrap}: resample the days with replacement many times, recompute the statistic, take the 2.5\\% and 97.5\\% percentiles',
      '\\textbf{Bootstrap}: reeșantionăm zilele cu întoarcere de multe ori, recalculăm statistica, luăm percentilele 2,5\\% și 97,5\\%')), 'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations and Derivations on Paper', 'Partea A: calcule și derivări pe hîrtie')

D.solved(T('A1: A Large Daily Loss under the Normal distribution Distribution', 'A1: o pierdere zilnică mare în distribuția Normală'),
         items(T('Daily log returns of an index are $N(\\mu, \\sigma^2)$ with $\\mu = 0.05\\%$ and $\\sigma = 1.1\\%$; there are 252 trading days a year.',
                 'Randamentele logaritmice zilnice ale unui indice sînt $N(\\mu, \\sigma^2)$, cu $\\mu = 0.05\\%$ și $\\sigma = 1.1\\%$; un an are 252 de zile de tranzacționare.'),
               T('1. Compute the probability of a daily loss larger than 2\\% and the expected number of such days per year.', '1. Calculați probabilitatea unei pierderi zilnice mai mari de 2\\% și numărul așteptat de asemenea zile pe an.'),
               T('2. Compute the 1\\% quantile of the daily return.', '2. Calculați cuantila de 1\\% a randamentului zilnic.'),
               T('3. Compute the expected number of 4-sigma days in 10 years.', '3. Calculați numărul așteptat de zile de 4 sigma în 10 ani.'),
               T('Report: a probability, a number of days, a quantile and an expected count.', 'Raportați: o probabilitate, un număr de zile, o cuantilă și un număr așteptat.')),
         items(T('1. $z = (-2 - 0.05)/1.1 = @{a1.z}$; $P = \\Phi(@{a1.z}) = @{a1.p}\\%$; $252 \\times @{a1.p}\\% = @{a1.dy}$ days a year',
                 '1. $z = (-2 - 0.05)/1.1 = @{a1.z}$; $P = \\Phi(@{a1.z}) = @{a1.p}\\%$; $252 \\times @{a1.p}\\% = @{a1.dy}$ zile pe an'),
               T('2. $x_{0.01} = 0.05 + 1.1 \\times (@{a1.z01}) = @{a1.q01}\\%$', '2. $x_{0.01} = 0.05 + 1.1 \\times (@{a1.z01}) = @{a1.q01}\\%$'),
               T('3. $P(|Z| > 4) = @{a1.p4}$; in @{a1.nd} days: $@{a1.exp4}$ days, so most decades would see none',
                 '3. $P(|Z| > 4) = @{a1.p4}$; în @{a1.nd} zile: $@{a1.exp4}$ zile, deci majoritatea deceniilor n-ar avea niciuna'),
               T('Lecture 2 shows that real indices have one such day every few months.', 'Cursul 2 arată că indicii reali au o asemenea zi la cîteva luni.')),
         size='footnotesize')

D.proposed(T('A2: The Same Questions for Bitcoin', 'A2: aceleași întrebări pentru Bitcoin'),
           items(T('Daily log returns of Bitcoin are $N(0.12\\%, 3.6\\%^2)$; Bitcoin trades 365 days a year. Model: A1.', 'Randamentele logaritmice zilnice ale Bitcoin sînt $N(0.12\\%, 3.6\\%^2)$; Bitcoin se tranzacționează 365 de zile pe an. Model: A1.'),
                 T('1. Compute the probability of a daily loss larger than 10\\% and the expected number of such days per year.', '1. Calculați probabilitatea unei pierderi zilnice mai mari de 10\\% și numărul așteptat de asemenea zile pe an.'),
                 T('2. Compute the 1\\% quantile of the daily return.', '2. Calculați cuantila de 1\\% a randamentului zilnic.'),
                 T('3. Compute the expected number of 4-sigma days in 10 years.', '3. Calculați numărul așteptat de zile de 4 sigma în 10 ani.'),
                 T('Report: a probability, a number of days, a quantile and an expected count.', 'Raportați: o probabilitate, un număr de zile, o cuantilă și un număr așteptat.')),
           items(T('1. $z = (-10 - 0.12)/3.6 = @{a2.z}$; $P = @{a2.p}\\%$; $365 \\times @{a2.p}\\% = @{a2.dy}$ days a year', '1. $z = (-10 - 0.12)/3.6 = @{a2.z}$; $P = @{a2.p}\\%$; $365 \\times @{a2.p}\\% = @{a2.dy}$ zile pe an'),
                 T('2. $x_{0.01} = 0.12 + 3.6 \\times (@{a2.z01}) = @{a2.q01}\\%$', '2. $x_{0.01} = 0.12 + 3.6 \\times (@{a2.z01}) = @{a2.q01}\\%$'),
                 T('3. $@{a2.nd} \\times @{a2.p4} = @{a2.exp4}$ days', '3. $@{a2.nd} \\times @{a2.p4} = @{a2.exp4}$ zile'),
                 T('In the data (Lecture 2), Bitcoin had losses beyond 10\\% far more often than this.', 'În date (Cursul 2), Bitcoin a avut pierderi de peste 10\\% mult mai des.')),
           size='footnotesize')

D.solved(T('A3: Lognormal Prices after One and Ten Years', 'A3: prețuri lognormale după unu și zece ani'),
         items(T('A stock costs 100 lei; its yearly log returns are i.i.d. $N(0.07, 0.18^2)$.', 'O acțiune costă 100 de lei; randamentele ei logaritmice anuale sînt i.i.d. $N(0.07, 0.18^2)$.'),
               T('1. Compute the median and the mean of the price after one year.', '1. Calculați mediana și media prețului după un an.'),
               T('2. Compute the probability that the price after one year is below 100 lei.', '2. Calculați probabilitatea ca prețul după un an să fie sub 100 de lei.'),
               T('3. Compute the 5\\% quantile of the price after one year.', '3. Calculați cuantila de 5\\% a prețului după un an.'),
               T('4. Repeat 1--3 for ten years.', '4. Repetați 1--3 pentru zece ani.'),
               T('Report: eight numbers and one sentence on the horizon.', 'Raportați: opt cifre și o frază despre orizont.')),
         items(T('1. Median $100e^{0.07} = @{a3.T1.median}$; mean $100e^{0.07 + @{a3.half}} = @{a3.T1.mean}$', '1. Mediana $100e^{0.07} = @{a3.T1.median}$; media $100e^{0.07 + @{a3.half}} = @{a3.T1.mean}$'),
               T('2. $P = \\Phi(-0.07/0.18) = \\Phi(@{a3.T1.zl}) = @{a3.T1.ploss}\\%$', '2. $P = \\Phi(-0.07/0.18) = \\Phi(@{a3.T1.zl}) = @{a3.T1.ploss}\\%$'),
               T('3. $100e^{0.07 + 0.18 \\times (@{a3.z05})} = @{a3.T1.q}$', '3. $100e^{0.07 + 0.18 \\times (@{a3.z05})} = @{a3.T1.q}$'),
               T('4. $T = 10$: $10\\mu = 0.7$, $\\sqrt{10}\\sigma = @{a3.T10.sT}$; median @{a3.T10.median}, mean @{a3.T10.mean}, $P$(loss) $= @{a3.T10.ploss}\\%$, 5\\% quantile @{a3.T10.q}',
                 '4. $T = 10$: $10\\mu = 0.7$, $\\sqrt{10}\\sigma = @{a3.T10.sT}$; mediana @{a3.T10.median}, media @{a3.T10.mean}, $P$(pierdere) $= @{a3.T10.ploss}\\%$, cuantila de 5\\% @{a3.T10.q}'),
               T('A loss becomes less likely with time, but the bad 5\\% case barely improves.', 'O pierdere devine mai puțin probabilă în timp, dar cazul prost de 5\\% abia se îmbunătățește.')),
         size='scriptsize')

D.proposed(T('A4: Why the Mean Is $e^{m + s^2/2}$', 'A4: de ce media este $e^{m + s^2/2}$'),
           items(T('Let $X \\sim N(m, s^2)$ and $Y = e^X$. Model: A3; exercise of the type in \\refFHHex.', 'Fie $X \\sim N(m, s^2)$ și $Y = e^X$. Model: A3; exercițiu de tipul celor din \\refFHHex.'),
                 T('1. Write $E[e^X] = \\int e^x f(x)\\,dx$ and complete the square in the exponent to show that $E[e^X] = e^{m + s^2/2}$.',
                   '1. Scrieți $E[e^X] = \\int e^x f(x)\\,dx$ și formați pătratul perfect în exponent pentru a arăta că $E[e^X] = e^{m + s^2/2}$.'),
                 T('2. Use the same idea for $E[e^{2X}]$ and derive $\\text{Var}(Y) = (e^{s^2} - 1)e^{2m + s^2}$.', '2. Folosiți aceeași idee pentru $E[e^{2X}]$ și deduceți $\\text{Var}(Y) = (e^{s^2} - 1)e^{2m + s^2}$.'),
                 T('3. For $m = 0.07$, $s = 0.18$, compute the mean and the median of $Y$ and their gap.', '3. Pentru $m = 0.07$, $s = 0.18$, calculați media și mediana lui $Y$ și diferența lor.'),
                 T('Report: the two derivations and one sentence linking the gap to the volatility drag of Chapter 1.', 'Raportați: cele două derivări și o frază care leagă diferența de volatility drag din Capitolul 1.')),
           items(T('1. $x - (x - m)^2/(2s^2) = -\\big(x - (m + s^2)\\big)^2/(2s^2) + m + s^2/2$; the remaining integral is that of a Normal density, equal to 1',
                   '1. $x - (x - m)^2/(2s^2) = -\\big(x - (m + s^2)\\big)^2/(2s^2) + m + s^2/2$; integrala rămasă este cea a unei densități Normale, egală cu 1'),
                 T('2. $2X \\sim N(2m, 4s^2)$, so $E[e^{2X}] = e^{2m + 2s^2}$ and $\\text{Var}(Y) = e^{2m + 2s^2} - e^{2m + s^2}$', '2. $2X \\sim N(2m, 4s^2)$, deci $E[e^{2X}] = e^{2m + 2s^2}$ și $\\text{Var}(Y) = e^{2m + 2s^2} - e^{2m + s^2}$'),
                 T('3. Mean $@{a4.mean}$, median $@{a4.median}$, variance $@{a4.var}$; in log terms the gap is $s^2/2 = @{a4.drag}\\%$',
                   '3. Media $@{a4.mean}$, mediana $@{a4.median}$, varianța $@{a4.var}$; în termeni logaritmici diferența este $s^2/2 = @{a4.drag}\\%$'),
                 T('The same $s^2/2$ separates the arithmetic mean of simple returns from the mean log return: the volatility drag.',
                   'Același $s^2/2$ separă media aritmetică a randamentelor simple de media randamentelor logaritmice: volatility drag.')),
           size='scriptsize')

D.solved(T('A5: Up Days in a Year (Normal Approximation)', 'A5: zilele de creștere dintr-un an (aproximarea Normală)'),
         items(T('An index rises on a given day with probability $p = 0.53$, independently across days; a year has 252 trading days.',
                 'Un indice crește într-o zi dată cu probabilitatea $p = 0.53$, independent de la o zi la alta; un an are 252 de zile de tranzacționare.'),
               T('1. Give the distribution of the number $X$ of up days, its mean and its standard deviation.', '1. Dați distribuția numărului $X$ de zile de creștere, media și abaterea ei standard.'),
               T('2. Approximate $P(X \\ge 140)$ with the Normal distribution and the continuity correction.', '2. Aproximați $P(X \\ge 140)$ cu distribuția Normală și corecția de continuitate.'),
               T('3. Compare with the exact binomial value from the notebook.', '3. Comparați cu valoarea binomială exactă din notebook.'),
               T('Report: two moments, two probabilities, one sentence on the quality of the approximation.', 'Raportați: două momente, două probabilități, o frază despre calitatea aproximării.')),
         items(T('1. $X \\sim B(252, 0.53)$; $E[X] = @{a5.mu}$, $\\text{sd} = \\sqrt{252 \\times 0.53 \\times 0.47} = @{a5.sd}$', '1. $X \\sim B(252, 0.53)$; $E[X] = @{a5.mu}$, $\\text{ab. std.} = \\sqrt{252 \\times 0.53 \\times 0.47} = @{a5.sd}$'),
               T('2. $z = (139.5 - @{a5.mu})/@{a5.sd} = @{a5.z}$; $P \\approx 1 - \\Phi(@{a5.z}) = @{a5.approx}\\%$', '2. $z = (139.5 - @{a5.mu})/@{a5.sd} = @{a5.z}$; $P \\approx 1 - \\Phi(@{a5.z}) = @{a5.approx}\\%$'),
               T('3. Exact: $@{a5.exact}\\%$', '3. Exact: $@{a5.exact}\\%$'),
               T('$np(1-p) \\approx @{a5.v}$ is large, so the CLT works very well here.', '$np(1-p) \\approx @{a5.v}$ este mare, deci CLT funcționează foarte bine aici.')),
         size='footnotesize')

D.proposed(T('A6: Is a Monthly Return Normal?', 'A6: este Normal un randament lunar?'),
           items(T('Daily log returns are i.i.d. with mean $0.04\\%$ and standard deviation $1.2\\%$. Model: A5.', 'Randamentele logaritmice zilnice sînt i.i.d., cu media $0.04\\%$ și abaterea standard $1.2\\%$. Model: A5.'),
                 T('1. Give the approximate distribution of the 21-day log return and its two parameters.', '1. Dați distribuția aproximativă a randamentului logaritmic pe 21 de zile și cei doi parametri ai ei.'),
                 T('2. Compute the probability of a 21-day loss larger than 10\\%, and of a 5-day loss larger than 10\\%.', '2. Calculați probabilitatea unei pierderi pe 21 de zile mai mari de 10\\% și a unei pierderi pe 5 zile mai mari de 10\\%.'),
                 T('3. Say whether the CLT still applies if the daily returns are Student-$t$ with $\\nu = 3$.', '3. Spuneți dacă CLT se mai aplică atunci cînd randamentele zilnice sînt Student-$t$ cu $\\nu = 3$.'),
                 T('4. Say whether it applies with $\\nu = 1.5$.', '4. Spuneți dacă se aplică pentru $\\nu = 1.5$.'),
                 T('Report: two distributions, two probabilities, two answers with a reason each.', 'Raportați: două distribuții, două probabilități, două răspunsuri, fiecare cu o justificare.')),
           items(T('1. $N(21 \\times 0.04, 21 \\times 1.2^2) = N(@{a6.h21.mean}, @{a6.h21.sd}^2)$', '1. $N(21 \\times 0.04, 21 \\times 1.2^2) = N(@{a6.h21.mean}, @{a6.h21.sd}^2)$'),
                 T('2. $z = @{a6.h21.z}$, $P = @{a6.h21.p}\\%$; 5 days: $N(@{a6.h5.mean}, @{a6.h5.sd}^2)$, $z = @{a6.h5.z}$, $P = @{a6.h5.p}$',
                   '2. $z = @{a6.h21.z}$, $P = @{a6.h21.p}\\%$; 5 zile: $N(@{a6.h5.mean}, @{a6.h5.sd}^2)$, $z = @{a6.h5.z}$, $P = @{a6.h5.p}$'),
                 T('3. Yes: $t(3)$ has finite variance ($\\nu > 2$); convergence is slower because the kurtosis is infinite', '3. Da: $t(3)$ are varianță finită ($\\nu > 2$); convergența este mai lentă pentru că aplatizarea este infinită'),
                 T('4. No: with $\\nu \\le 2$ the variance is infinite; sums converge to an $\\alpha$-stable law (Chapter 3)', '4. Nu: cu $\\nu \\le 2$ varianța este infinită; sumele converg către o lege $\\alpha$-stabilă (Capitolul 3)')),
           size='scriptsize')

D.solved(T('A7: Jarque--Bera on Paper', 'A7: Jarque--Bera pe hîrtie'),
         items(T('A series of $n = 4000$ daily returns has sample skewness $S = -0.6$ and excess kurtosis $K = 12$.', 'O serie de $n = 4000$ de randamente zilnice are asimetria $S = -0.6$ și excesul de aplatizare $K = 12$.'),
               T('1. Compute the standard errors of $S$ and $K$ under normality and the two $z$ statistics.', '1. Calculați erorile standard ale lui $S$ și $K$ în ipoteza de normalitate și cele două statistici $z$.'),
               T('2. Compute JB and the share of it that comes from the kurtosis.', '2. Calculați JB și partea care vine din aplatizare.'),
               T('3. Decide at the 5\\% level.', '3. Decideți la nivelul de 5\\%.'),
               T('Report: two standard errors, two $z$ values, JB, the decision.', 'Raportați: două erori standard, două valori $z$, JB, decizia.')),
         items(T('1. $\\sqrt{6/4000} = @{a7.se_s}$, $\\sqrt{24/4000} = @{a7.se_k}$; $z_S = @{a7.z_s}$, $z_K = @{a7.z_k}$', '1. $\\sqrt{6/4000} = @{a7.se_s}$, $\\sqrt{24/4000} = @{a7.se_k}$; $z_S = @{a7.z_s}$, $z_K = @{a7.z_k}$'),
               T('2. $\\text{JB} = \\frac{4000}{6}(0.36 + 36) = @{a7.ps} + @{a7.pk} = @{a7.jb}$, almost all from $K$', '2. $\\text{JB} = \\frac{4000}{6}(0.36 + 36) = @{a7.ps} + @{a7.pk} = @{a7.jb}$, aproape tot din $K$'),
               T('3. $\\text{JB} > @{a7.crit}$: reject normality', '3. $\\text{JB} > @{a7.crit}$: respingem normalitatea'),
               T('JB is the sum of the two squared $z$ statistics: $z_S^2 + z_K^2$.', 'JB este suma celor două statistici $z$ la pătrat: $z_S^2 + z_K^2$.')),
         size='footnotesize')

D.proposed(T('A8: Moments of the Student-$t$', 'A8: momentele distribuției Student-$t$'),
           items(T('$T \\sim t(\\nu)$; a model for daily returns is $r = m + s\\,T$. Model: A7.', '$T \\sim t(\\nu)$; un model pentru randamentele zilnice este $r = m + s\\,T$. Model: A7.'),
                 T('1. Compute the variance and the excess kurtosis of $t(5)$ and $t(10)$.', '1. Calculați varianța și excesul de aplatizare pentru $t(5)$ și $t(10)$.'),
                 T('2. Find $\\nu$ such that the excess kurtosis is 3.', '2. Găsiți $\\nu$ pentru care excesul de aplatizare este 3.'),
                 T('3. Find the scale $s$ that gives a daily standard deviation of $1.2\\%$ for $\\nu = 5$ and for $\\nu = 6$.', '3. Găsiți scara $s$ care dă o abatere standard zilnică de $1.2\\%$ pentru $\\nu = 5$ și pentru $\\nu = 6$.'),
                 T('4. Compare $P(|X| > 4)$ for a unit-variance $t(5)$ (value in the notebook) with the Normal value.', '4. Comparați $P(|X| > 4)$ pentru o $t(5)$ cu varianța 1 (valoarea din notebook) cu valoarea Normală.'),
                 T('Report: four moments, one $\\nu$, two scales and one ratio.', 'Raportați: patru momente, un $\\nu$, două scări și un raport.')),
           items(T('1. $t(5)$: variance $@{a8.t5.var}$, excess kurtosis $@{a8.t5.ek}$; $t(10)$: $@{a8.t10.var}$ and $@{a8.t10.ek}$', '1. $t(5)$: varianța $@{a8.t5.var}$, excesul de aplatizare $@{a8.t5.ek}$; $t(10)$: $@{a8.t10.var}$ și $@{a8.t10.ek}$'),
                 T('2. $6/(\\nu - 4) = 3 \\Rightarrow \\nu = @{a8.nu3}$', '2. $6/(\\nu - 4) = 3 \\Rightarrow \\nu = @{a8.nu3}$'),
                 T('3. $s = 1.2\\sqrt{(\\nu - 2)/\\nu}$: $@{a8.t5.scale}\\%$ for $\\nu = 5$, $@{a8.t6.scale}\\%$ for $\\nu = 6$', '3. $s = 1.2\\sqrt{(\\nu - 2)/\\nu}$: $@{a8.t5.scale}\\%$ pentru $\\nu = 5$, $@{a8.t6.scale}\\%$ pentru $\\nu = 6$'),
                 T('4. $@{a8.t5.p4}\\%$ vs $@{a8.normal.p4}\\%$: @{a8.ratio} times more likely', '4. $@{a8.t5.p4}\\%$ față de $@{a8.normal.p4}\\%$: de @{a8.ratio} de ori mai probabil')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: How Non-Normal Is the S\\&P 500? [Solved]', 'B1: cît de departe de normalitate este S\\&P 500? [Rezolvat]'),
       T('how far are the daily log returns of the S\\&P 500 from the Normal distribution, and how precisely do we know it?',
         'cît de departe sînt randamentele logaritmice zilnice ale S\\&P 500 de distribuția Normală și cît de precis știm acest lucru?'),
       T('S\\&P 500 closes, 2010--2026, daily log returns in \\%', 'închiderile S\\&P 500, 2010--2026, randamente logaritmice zilnice în \\%'),
       [T('Compute the mean, standard deviation, skewness, excess kurtosis and the worst day with its date.', 'Calculați media, abaterea standard, asimetria, excesul de aplatizare și cea mai proastă zi, cu data ei.'),
        T('Compute the Normal-theory standard errors of $S$ and $K$, and the JB statistic.', 'Calculați erorile standard ale lui $S$ și $K$ în ipoteza de normalitate și statistica JB.'),
        T('Bootstrap $S$ and $K$ with 2000 resamples of the days and take the 2.5\\% and 97.5\\% percentiles.', 'Faceți bootstrap pentru $S$ și $K$ cu 2000 de reeșantionări ale zilelor și luați percentilele 2,5\\% și 97,5\\%.'),
        T('Interpretation: why is the bootstrap interval of $K$ so much wider than the Normal-theory one?', 'Interpretare: de ce este intervalul bootstrap al lui $K$ mult mai larg decît cel din teoria Normală?')],
       T('a table of moments, JB, two intervals for $K$, one sentence', 'un tabel de momente, JB, două intervale pentru $K$, o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), cols(fig('ch2_sem_b1', h='0.55', w='1.0'), items(
    T('$n = @{b1.n}$; mean $@{b1.mean}\\%$, s.d. $@{b1.sd}\\%$; worst day $@{b1.min}\\%$ on @{b1.mindate}', '$n = @{b1.n}$; media $@{b1.mean}\\%$, abaterea std. $@{b1.sd}\\%$; cea mai proastă zi $@{b1.min}\\%$, pe @{b1.mindate}'),
    T('$S = @{b1.skew}$ (SE $@{b1.se_skew}$), $K = @{b1.exkurt}$ (SE $@{b1.se_kurt}$); $\\text{JB} = @{b1.jb}$: normality rejected',
      '$S = @{b1.skew}$ (SE $@{b1.se_skew}$), $K = @{b1.exkurt}$ (SE $@{b1.se_kurt}$); $\\text{JB} = @{b1.jb}$: normalitatea respinsă'),
    T('Normal theory: $K \\in [@{b1.n_lo_k}, @{b1.n_hi_k}]$; bootstrap: $[@{b1.bs_k_lo}, @{b1.bs_k_hi}]$; for $S$, bootstrap $[@{b1.bs_s_lo}, @{b1.bs_s_hi}]$',
      'Teoria Normală: $K \\in [@{b1.n_lo_k}, @{b1.n_hi_k}]$; bootstrap: $[@{b1.bs_k_lo}, @{b1.bs_k_hi}]$; pentru $S$, bootstrap $[@{b1.bs_s_lo}, @{b1.bs_s_hi}]$'),
    T('Interpretation: $\\sqrt{24/n}$ assumes Normal data; with heavy tails $K$ depends on a few days, and whether they are resampled moves it a lot; the bootstrap s.d. of $K$ is @{b1.ratio} times larger',
      'Interpretare: $\\sqrt{24/n}$ presupune date Normale; cu cozi groase, $K$ depinde de cîteva zile, iar prezența lor în reeșantionare îl mută mult; abaterea standard bootstrap a lui $K$ este de @{b1.ratio} de ori mai mare'),
    T('The skewness interval contains 0: the sign of $S$ is not certain', 'Intervalul asimetriei conține 0: semnul lui $S$ nu este sigur')),
    wl='0.48', wr='0.50') + qlsem(), 'scriptsize')

D.task(T('B2: Three More Series [Proposed]', 'B2: încă trei serii [Propus]'),
       T('which of the BET, Bitcoin and OMV Petrom is farthest from the Normal distribution, and how much does one day matter?',
         'care dintre BET, Bitcoin și OMV Petrom este cel mai departe de distribuția Normală și cît contează o singură zi?'),
       T('daily log returns, 2010--2026 (Bitcoin from 2014), each on its own calendar; model: B1', 'randamente logaritmice zilnice, 2010--2026 (Bitcoin din 2014), fiecare pe calendarul ei; model: B1'),
       [T('Compute $n$, mean, standard deviation, $S$, $K$ and JB for each series.', 'Calculați $n$, media, abaterea standard, $S$, $K$ și JB pentru fiecare serie.'),
        T('Remove the single largest absolute return of each series and recompute $K$.', 'Eliminați cel mai mare randament în valoare absolută din fiecare serie și recalculați $K$.'),
        T('Interpretation: what does the change in $K$ tell you about kurtosis as a summary of risk?', 'Interpretare: ce vă spune schimbarea lui $K$ despre aplatizare ca rezumat al riscului?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrr', T('Series', 'Seria') + ' & $n$ & ' + T('Mean', 'Media') + ' & ' + T('S.d.', 'Ab. std.') + ' & $S$ & $K$ & JB & $K$ ' + T('without max', 'fără max'),
    [f'{lab} & @{{b2.{k}.n}} & $@{{b2.{k}.mean}}$ & $@{{b2.{k}.sd}}$ & $@{{b2.{k}.skew}}$ & $@{{b2.{k}.exkurt}}$ & @{{b2.{k}.jb}} & $@{{b2.{k}.exkurt_drop1}}$'
     for lab, k in [('BET', 'bet'), ('Bitcoin', 'btc'), ('OMV Petrom', 'snp')]], size='footnotesize') + items(
    T('Removed days: BET @{b2.bet.drop} ($@{b2.bet.drop_r}\\%$), Bitcoin @{b2.btc.drop} ($@{b2.btc.drop_r}\\%$), SNP @{b2.snp.drop} ($@{b2.snp.drop_r}\\%$)',
      'Zilele eliminate: BET @{b2.bet.drop} ($@{b2.bet.drop_r}\\%$), Bitcoin @{b2.btc.drop} ($@{b2.btc.drop_r}\\%$), SNP @{b2.snp.drop} ($@{b2.snp.drop_r}\\%$)'),
    T('Farthest from the Normal distribution by $K$ and JB: the BET; all three rejected', 'Cel mai departe de distribuția Normală după $K$ și JB: BET; toate trei respinse'),
    T('Interpretation: one day out of thousands moves the kurtosis of Bitcoin from @{b2.btc.exkurt} to @{b2.btc.exkurt_drop1}; $K$ is a fragile summary, report it with the extremes',
      'Interpretare: o zi din mii mută aplatizarea Bitcoin de la @{b2.btc.exkurt} la @{b2.btc.exkurt_drop1}; $K$ este un rezumat fragil, raportați-l împreună cu extremele')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Normal or Student-$t$ for the BET? [Solved]', 'B3: distribuția Normală sau Student-$t$ pentru BET? [Rezolvat]'),
       T('does a Student-$t$ describe the daily returns of the BET better than the Normal distribution, and where does each fail?',
         'descrie o distribuție Student-$t$ randamentele zilnice ale BET mai bine decît distribuția Normală și unde greșește fiecare?'),
       T('BET closes, 2010--2026, daily log returns in \\%', 'închiderile BET, 2010--2026, randamente logaritmice zilnice în \\%'),
       [T('Draw the QQ plot of the standardised returns against $N(0,1)$.', 'Desenați QQ plot-ul randamentelor standardizate față de $N(0,1)$.'),
        T('Fit a Student-$t$ by maximum likelihood and draw the QQ plot against the fitted $t$.', 'Estimați o distribuție Student-$t$ prin verosimilitate maximă și desenați QQ plot-ul față de $t$ estimată.'),
        T('Compare the log-likelihoods and the AIC of the two models.', 'Comparați log-verosimilitățile și AIC ale celor două modele.'),
        T('Interpretation: where does the Normal distribution fail, and where does the fitted $t$ fail?', 'Interpretare: unde greșește distribuția Normală și unde greșește $t$ estimată?')],
       T('$\\hat\\nu$, $\\hat m$, $\\hat s$, two AIC values, two QQ plots, one sentence', '$\\hat\\nu$, $\\hat m$, $\\hat s$, două valori AIC, două QQ plots, o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), cols(fig('ch2_sem_b3', h='0.55', w='1.0'), items(
    T('$\\hat\\nu = @{b3.nu}$, $\\hat m = @{b3.loc}\\%$, $\\hat s = @{b3.scale}\\%$; implied s.d. $\\hat s\\sqrt{\\hat\\nu/(\\hat\\nu - 2)} = @{b3.sd_t}\\%$',
      '$\\hat\\nu = @{b3.nu}$, $\\hat m = @{b3.loc}\\%$, $\\hat s = @{b3.scale}\\%$; abaterea std. implicată $\\hat s\\sqrt{\\hat\\nu/(\\hat\\nu - 2)} = @{b3.sd_t}\\%$'),
    T('$\\ell_t = @{b3.ll_t}$ vs $\\ell_N = @{b3.ll_n}$; $\\text{AIC}_N - \\text{AIC}_t = @{b3.daic}$: the $t$ wins clearly',
      '$\\ell_t = @{b3.ll_t}$ față de $\\ell_N = @{b3.ll_n}$; $\\text{AIC}_N - \\text{AIC}_t = @{b3.daic}$: $t$ cîștigă clar'),
    T('Lowest return $@{b3.min}\\%$; matching quantile: Normal distribution $@{b3.q_n_min}\\%$, fitted $t$ $@{b3.q_t_min}\\%$',
      'Cel mai mic randament $@{b3.min}\\%$; cuantila corespunzătoare: distribuția Normală $@{b3.q_n_min}\\%$, $t$ estimată $@{b3.q_t_min}\\%$'),
    T('Interpretation: the Normal distribution misses both tails (S-shape); the $t$ fits the centre and most of the tails, but its extreme quantiles go slightly beyond the data',
      'Interpretare: distribuția Normală ratează ambele cozi (forma de S); $t$ se potrivește centrului și majorității cozilor, dar cuantilele ei extreme trec puțin dincolo de date')),
    wl='0.50', wr='0.48') + qlsem(), 'scriptsize')

D.task(T('B4: How Many 4-Sigma Days? [Proposed]', 'B4: cîte zile de 4 sigma? [Propus]'),
       T('how many 4-sigma days do the S\\&P 500, DAX, BET and Bitcoin have, and which model predicts the count?',
         'cîte zile de 4 sigma au S\\&P 500, DAX, BET și Bitcoin și ce model dă numărul corect?'),
       T('daily log returns, 2010--2026 (Bitcoin from 2014); model: B3', 'randamente logaritmice zilnice, 2010--2026 (Bitcoin din 2014); model: B3'),
       [T('Count the days with $|r_t - \\bar r| > 4s$ in each series.', 'Numărați zilele cu $|r_t - \\bar r| > 4s$ din fiecare serie.'),
        T('Compute the expected count under the Normal distribution and the Poisson probability of seeing at least the observed count.',
          'Calculați numărul așteptat în distribuția Normală și probabilitatea Poisson de a vedea cel puțin numărul observat.'),
        T('Compute the expected count under the fitted Student-$t$ (same threshold $\\bar r \\pm 4s$).', 'Calculați numărul așteptat cu distribuția Student-$t$ estimată (același prag $\\bar r \\pm 4s$).'),
        T('Interpretation: is the Student-$t$ too heavy, too light or about right for each series?', 'Interpretare: este distribuția Student-$t$ prea grea, prea ușoară sau potrivită pentru fiecare serie?')],
       T('one table and one sentence per series', 'un tabel și o frază pentru fiecare serie'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrr', T('Series', 'Seria') + ' & $n$ & ' + T('observed', 'observate') + ' & Normal & $p$ (Poisson) & $\\hat\\nu$ & Student-$t$',
    [f'{lab} & @{{b4.{k}.n}} & @{{b4.{k}.obs}} & $@{{b4.{k}.en}}$ & ${{@{{b4.{k}.pp}}}}$ & $@{{b4.{k}.nu}}$ & $@{{b4.{k}.et}}$'
     for lab, k in [('S\\&P 500', 'sp500'), ('DAX', 'dax'), ('BET', 'bet'), ('Bitcoin', 'btc')]], size='footnotesize') + items(
    T('The Normal distribution is rejected for every series: the observed counts are @{b4.rmin}--@{b4.rmax} times the expected ones', 'Distribuția Normală este respinsă pentru toate seriile: numerele observate sînt de @{b4.rmin}--@{b4.rmax} ori mai mari decît cele așteptate'),
    T('Interpretation: the $t$ is about right for the BET, somewhat too heavy for the S\\&P 500 and the DAX, and much too heavy for Bitcoin ($\\hat\\nu$ close to 2)',
      'Interpretare: $t$ este potrivită pentru BET, ceva prea grea pentru S\\&P 500 și DAX și mult prea grea pentru Bitcoin ($\\hat\\nu$ aproape de 2)'),
    T('One $\\nu$ fits the whole distribution by likelihood, which is dominated by the many ordinary days, not by the few extreme ones',
      'Un singur $\\nu$ ajustează întreaga distribuție prin verosimilitate, dominată de multele zile obișnuite, nu de puținele zile extreme')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Does the S\\&P 500 Become Normal at Longer Horizons? [Solved]', 'B5: devine S\\&P 500 Normal pe orizonturi mai lungi? [Rezolvat]'),
       T('are 5-day and 21-day returns of the S\\&P 500 closer to the Normal distribution than daily returns?', 'sînt randamentele pe 5 și pe 21 de zile ale S\\&P 500 mai apropiate de distribuția Normală decît cele zilnice?'),
       T('S\\&P 500 daily log returns, 2010--2026; $h$-day returns are sums over non-overlapping blocks of $h$ days', 'randamentele logaritmice zilnice ale S\\&P 500, 2010--2026; randamentele pe $h$ zile sînt sume pe blocuri de $h$ zile care nu se suprapun'),
       [T('Build the 5-day and 21-day log returns.', 'Construiți randamentele logaritmice pe 5 și pe 21 de zile.'),
        T('Compute $n$, $S$, $K$, the standard error $\\sqrt{24/n}$ and JB with its $p$-value at $h = 1, 5, 21$.', 'Calculați $n$, $S$, $K$, eroarea standard $\\sqrt{24/n}$ și JB cu valoarea $p$ pentru $h = 1, 5, 21$.'),
        T('Draw the three QQ plots against $N(0,1)$.', 'Desenați cele trei QQ plots față de $N(0,1)$.'),
        T('Interpretation: is the monthly distribution Normal, and why does the test have less power at $h = 21$?', 'Interpretare: este distribuția lunară Normală și de ce are testul mai puțină putere la $h = 21$?')],
       T('a table with three rows, three QQ plots, two sentences', 'un tabel cu trei rînduri, trei QQ plots, două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch2_sem_b5', h='0.42') + table(
    'rrrrrr', '$h$ & $n$ & $S$ & $K$ & $\\sqrt{24/n}$ & JB ($p$)',
    [f'{h} & @{{b5.{h}.n}} & $@{{b5.{h}.skew}}$ & $@{{b5.{h}.ek}}$ & $@{{b5.{h}.sek}}$ & @{{b5.{h}.jb}} (${{@{{b5.{h}.p}}}}$)' for h in ['1', '5', '21']],
    size='scriptsize') + items(
    T('Interpretation: $K$ falls from @{b5.1.ek} to @{b5.21.ek}, but monthly returns are still not Normal (skewness $@{b5.21.skew}$, March 2020); with only @{b5.21.n} months, the standard error of $K$ is @{b5.21.sek}, so smaller departures would go undetected',
      'Interpretare: $K$ scade de la @{b5.1.ek} la @{b5.21.ek}, dar randamentele lunare tot nu sînt Normale (asimetria $@{b5.21.skew}$, martie 2020); cu doar @{b5.21.n} luni, eroarea standard a lui $K$ este @{b5.21.sek}, deci abaterile mai mici n-ar fi detectate')) + qlsem(),
    'scriptsize')

D.task(T('B6: Are BET Returns Independent? [Proposed]', 'B6: sînt independente randamentele BET? [Propus]'),
       T('are the daily returns of the BET uncorrelated, and are they independent?', 'sînt randamentele zilnice ale BET necorelate și sînt ele independente?'),
       T('BET daily log returns, 2010--2026; model: B5', 'randamentele logaritmice zilnice ale BET, 2010--2026; model: B5'),
       [T('Compute the ACF of $r_t$ and of $|r_t|$ at lags 1--10 and the band $\\pm 1.96/\\sqrt n$.', 'Calculați ACF pentru $r_t$ și pentru $|r_t|$ la decalajele 1--10 și banda $\\pm 1.96/\\sqrt n$.'),
        T('Compute the Ljung--Box statistic $Q(10)$ for $r_t$ and for $|r_t|$ and compare with $\\chi^2_{0.95}(10)$.', 'Calculați statistica Ljung--Box $Q(10)$ pentru $r_t$ și pentru $|r_t|$ și comparați cu $\\chi^2_{0.95}(10)$.'),
        T('Interpretation: can returns be uncorrelated but not independent?', 'Interpretare: pot fi randamentele necorelate, dar nu independente?')],
       T('two ACF columns, two $Q$ values, one sentence', 'două coloane ACF, două valori $Q$, o frază'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), items(
    T('$n = @{b6.n}$, band $\\pm @{b6.band}$', '$n = @{b6.n}$, banda $\\pm @{b6.band}$'),
    T('$r_t$: $\\hat\\rho(1), \\hat\\rho(2), \\hat\\rho(3) = @{b6.r1}, @{b6.r2}, @{b6.r3}$; @{b6.nout} of 10 lags outside the band; $Q(10) = @{b6.qr}$ ($p = @{b6.pr}$)',
      '$r_t$: $\\hat\\rho(1), \\hat\\rho(2), \\hat\\rho(3) = @{b6.r1}, @{b6.r2}, @{b6.r3}$; @{b6.nout} din 10 decalaje în afara benzii; $Q(10) = @{b6.qr}$ ($p = @{b6.pr}$)'),
    T('$|r_t|$: $@{b6.a1}, @{b6.a2}, @{b6.a3}$, \\dots, $@{b6.a10}$ at lag 10; $Q(10) = @{b6.qa}$, far above $@{b6.crit}$',
      '$|r_t|$: $@{b6.a1}, @{b6.a2}, @{b6.a3}$, \\dots, $@{b6.a10}$ la decalajul 10; $Q(10) = @{b6.qa}$, mult peste $@{b6.crit}$'),
    T('The test on $r_t$ also rejects, but the autocorrelations are at most @{b6.max} in absolute value and the band assumes i.i.d. data, which volatility clustering violates',
      'Testul pentru $r_t$ respinge și el, dar autocorelațiile sînt cel mult @{b6.max} în valoare absolută, iar banda presupune date i.i.d., ceea ce volatility clustering încalcă'),
    T('Interpretation: yes; independence would make $|r_t|$ uncorrelated too; the strong autocorrelation of $|r_t|$ shows that the size of returns is predictable, their sign much less',
      'Interpretare: da; independența ar face și $|r_t|$ necorelate; autocorelația puternică a lui $|r_t|$ arată că mărimea randamentelor este previzibilă, semnul lor mult mai puțin')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B7: Is Bitcoin Like a Stock Index? [Proposed]', 'B7: seamănă Bitcoin cu un indice bursier? [Propus]'),
       T('do the S\\&P 500 and Bitcoin both show the leverage effect and the gain/loss asymmetry?', 'arată atît S\\&P 500, cît și Bitcoin efectul de levier și asimetria cîștig/pierdere?'),
       T('daily log returns, S\\&P 500 2010--2026, Bitcoin 2014--2026; model: B5', 'randamente logaritmice zilnice, S\\&P 500 2010--2026, Bitcoin 2014--2026; model: B5'),
       [T('Compute $L(k) = \\text{Corr}(r_t, |r_{t+k}|)$ for $k = 1, \\dots, 5$ and their mean, with the band $\\pm 1.96/\\sqrt n$.', 'Calculați $L(k) = \\text{Corr}(r_t, |r_{t+k}|)$ pentru $k = 1, \\dots, 5$ și media lor, cu banda $\\pm 1.96/\\sqrt n$.'),
        T('Count the days below $\\bar r - 4s$ and above $\\bar r + 4s$; test whether falls and rises are equally likely (binomial test, $p = 0.5$).',
          'Numărați zilele sub $\\bar r - 4s$ și peste $\\bar r + 4s$; testați dacă scăderile și creșterile sînt la fel de probabile (testul binomial, $p = 0.5$).'),
        T('Compare $|q_{0.01}|$ and $q_{0.99}$ and the skewness.', 'Comparați $|q_{0.01}|$ cu $q_{0.99}$ și asimetria.'),
        T('Interpretation: which stylised facts of stock indices does Bitcoin share?', 'Interpretare: ce fapte stilizate ale indicilor bursieri are și Bitcoin?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B7')

D.frame(T('B7: Solution [Proposed]', 'B7: rezolvare [Propus]'), table(
    'lrrrrrrrr', T('Series', 'Seria') + ' & $L(1)$ & $\\bar L(1..5)$ & ' + T('band', 'banda') + ' & $< -4s$ & $> +4s$ & $p$ & $|q_{0.01}|$ & $q_{0.99}$',
    [f'{lab} & $@{{b7.{k}.L1}}$ & $@{{b7.{k}.Lm}}$ & $@{{b7.{k}.band}}$ & @{{b7.{k}.d}} & @{{b7.{k}.u}} & $@{{b7.{k}.bp}}$ & $@{{b7.{k}.q01}}$ & $@{{b7.{k}.q99}}$'
     for lab, k in [('S\\&P 500', 'sp500'), ('Bitcoin', 'btc')]], size='footnotesize') + items(
    T('Leverage: clear for the S\\&P 500 ($\\bar L = @{b7.sp500.Lm}$, @{b7.sp500.times} times the band); for Bitcoin only at lag 1, the mean is at the edge of the band',
      'Levier: clar pentru S\\&P 500 ($\\bar L = @{b7.sp500.Lm}$, de @{b7.sp500.times} ori banda); pentru Bitcoin doar la decalajul 1, media este la marginea benzii'),
    T('More extreme falls than rises in both, but with so few extreme days the binomial test cannot reject equality ($p = @{b7.sp500.bp}$ and $@{b7.btc.bp}$)',
      'Mai multe căderi extreme decît creșteri la ambele, dar cu atît de puține zile extreme testul binomial nu poate respinge egalitatea ($p = @{b7.sp500.bp}$ și $@{b7.btc.bp}$)'),
    T('Interpretation: Bitcoin shares heavy tails, clustering and negative skewness ($@{b7.btc.skew}$), but the leverage effect is weak: there is no firm with debt behind it',
      'Interpretare: Bitcoin are cozi groase, clustering și asimetrie negativă ($@{b7.btc.skew}$), dar efectul de levier este slab: în spatele lui nu există o firmă cu datorii')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Are Bitcoin\'s Tails Getting Thinner? [Proposed]', 'C1: devin mai subțiri cozile Bitcoin? [Propus]'),
       T('has the tail of Bitcoin\'s daily returns become thinner since 2015, as its market matured?', 'a devenit mai subțire coada randamentelor zilnice ale Bitcoin din 2015, pe măsură ce piața s-a maturizat?'),
       T('Bitcoin and S\\&P 500 daily log returns, 2015--2026; models: B3, B5', 'randamentele logaritmice zilnice ale Bitcoin și S\\&P 500, 2015--2026; modele: B3, B5'),
       [T('Fit a Student-$t$ and compute the excess kurtosis for each calendar year and each series.', 'Estimați o distribuție Student-$t$ și calculați excesul de aplatizare pentru fiecare an calendaristic și fiecare serie.'),
        T('Compute the Spearman rank correlation between the year and $\\hat\\nu$.', 'Calculați corelația rangurilor Spearman între an și $\\hat\\nu$.'),
        T('Fit one $t$ on 2015--2020 and one on 2021--2026 and compare $\\hat\\nu$.', 'Estimați o distribuție $t$ pe 2015--2020 și una pe 2021--2026 și comparați $\\hat\\nu$.'),
        T('Propose one way to make the answer more reliable (intervals, other windows, other crypto assets).', 'Propuneți o cale de a face răspunsul mai sigur (intervale, alte ferestre, alte active cripto).'),
        T('Interpretation: what can @{c1.btc.ny} yearly estimates tell us about a trend?', 'Interpretare: ce ne pot spune @{c1.btc.ny} estimări anuale despre un trend?')],
       T('a table, the rank correlation with its $p$-value, a short plan for a project', 'un tabel, corelația rangurilor cu valoarea $p$, un scurt plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), items(
    T('Bitcoin: yearly $\\hat\\nu$ between @{c1.btc.numin} and @{c1.btc.numax}; Spearman with the year $@{c1.btc.rho}$ ($p = @{c1.btc.p}$, @{c1.btc.ny} years)',
      'Bitcoin: $\\hat\\nu$ anual între @{c1.btc.numin} și @{c1.btc.numax}; Spearman cu anul $@{c1.btc.rho}$ ($p = @{c1.btc.p}$, @{c1.btc.ny} ani)'),
    T('Two halves: $\\hat\\nu$ = @{c1.btc.nu1} (2015--2020) and @{c1.btc.nu2} (2021--2026)', 'Două jumătăți: $\\hat\\nu$ = @{c1.btc.nu1} (2015--2020) și @{c1.btc.nu2} (2021--2026)'),
    T('S\\&P 500: Spearman $@{c1.sp500.rho}$ ($p = @{c1.sp500.p}$); $\\hat\\nu$ = @{c1.sp500.nu1} and @{c1.sp500.nu2}: the benchmark also changed', 'S\\&P 500: Spearman $@{c1.sp500.rho}$ ($p = @{c1.sp500.p}$); $\\hat\\nu$ = @{c1.sp500.nu1} și @{c1.sp500.nu2}: și reperul s-a schimbat'),
    T('Weak evidence of thinner Bitcoin tails, but one test on @{c1.btc.ny} noisy points, chosen after seeing the data; calm years alone can raise $\\hat\\nu$',
      'Dovezi slabe de cozi mai subțiri pentru Bitcoin, dar un singur test pe @{c1.btc.ny} puncte zgomotoase, ales după ce am văzut datele; doar anii calmi pot crește $\\hat\\nu$'),
    T('Project directions: rolling windows with bootstrap intervals, returns scaled by a rolling volatility, Ethereum and Solana, the Hill estimator of Chapter 5',
      'Direcții de proiect: ferestre mobile cu intervale bootstrap, randamente împărțite la o volatilitate mobilă, Ethereum și Solana, estimatorul Hill din Capitolul 5')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant: ``Describe the distribution of daily S\\&P 500 returns in 2010--2026.\'\' The answer:',
      'Un student a întrebat un asistent AI: „Descrie distribuția randamentelor zilnice ale S\\&P 500 în 2010--2026.” Răspunsul:'),
    T('\\aiprompt{(a) The excess kurtosis is @{c2.exkurt}; since the Normal distribution has kurtosis 0, the kurtosis of the S\\&P 500 is @{c2.exkurt}.}',
      '\\aiprompt{(a) Excesul de aplatizare este @{c2.exkurt}; cum distribuția Normală are aplatizarea 0, aplatizarea S\\&P 500 este @{c2.exkurt}.}'),
    T('\\aiprompt{(b) JB = @{c2.jb} with p = 0.000, which proves that the returns follow a Student-t distribution.}',
      '\\aiprompt{(b) JB = @{c2.jb} cu p = 0,000, ceea ce dovedește că randamentele urmează o distribuție Student-t.}'),
    T('\\aiprompt{(c) A Normal 4-sigma day happens once every @{c2.every} days; the @{c2.obs4} such days in @{c2.n} days are just bad luck.}',
      '\\aiprompt{(c) O zi Normală de 4 sigma apare o dată la @{c2.every} zile; cele @{c2.obs4} zile de acest fel din @{c2.n} sînt doar ghinion.}'),
    T('\\aiprompt{(d) The lag-1 autocorrelation is only @{c2.r1}, so returns are independent and volatility cannot be predicted.}',
      '\\aiprompt{(d) Autocorelația la decalajul 1 este doar @{c2.r1}, deci randamentele sînt independente și volatilitatea nu poate fi prevăzută.}'),
    T('\\aiprompt{(e) With yearly log returns N(7\\%, 18\\%\\textasciicircum 2), the expected value of 100 invested after one year is 100 exp(0.07) = @{c2.med}.}',
      '\\aiprompt{(e) Cu randamente logaritmice anuale N(7\\%, 18\\%\\textasciicircum 2), valoarea așteptată a 100 investiți după un an este 100 exp(0,07) = @{c2.med}.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație spuneți dacă este corectă; dacă nu, dați afirmația corectă și cifra corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: the Normal kurtosis is 3 (excess 0); kurtosis $= @{c2.exkurt} + 3 = @{c2.kurt}$', '(a) Greșit: aplatizarea Normală este 3 (excesul 0); aplatizarea $= @{c2.exkurt} + 3 = @{c2.kurt}$'),
    T('(b) Wrong: JB only rejects normality; the fitted $t$ ($\\hat\\nu = @{c2.nu}$) is better by AIC, but its QQ plot still misses the extremes and it is symmetric',
      '(b) Greșit: JB doar respinge normalitatea; $t$ estimată ($\\hat\\nu = @{c2.nu}$) este mai bună după AIC, dar QQ plot-ul ei tot ratează extremele și este simetrică'),
    T('(c) Wrong: the Normal expectation is $@{c2.exp4}$ days; seeing @{c2.obs4} has Poisson probability about $@{c2.pp}$: the model is wrong, not the luck',
      '(c) Greșit: așteptarea Normală este de $@{c2.exp4}$ zile; a vedea @{c2.obs4} are probabilitatea Poisson de circa $@{c2.pp}$: greșit este modelul, nu norocul'),
    T('(d) Wrong: uncorrelated is not independent; the ACF of $|r_t|$ is $@{c2.a1}$ at lag 1 and $@{c2.a10}$ at lag 10, Ljung--Box $Q(10) = @{c2.lba}$: volatility is predictable',
      '(d) Greșit: necorelat nu înseamnă independent; ACF a lui $|r_t|$ este $@{c2.a1}$ la decalajul 1 și $@{c2.a10}$ la decalajul 10, Ljung--Box $Q(10) = @{c2.lba}$: volatilitatea este previzibilă'),
    T('(e) Wrong: $100e^{0.07} = @{c2.med}$ is the median; the mean is $100e^{0.07 + 0.18^2/2} = @{c2.mean}$', '(e) Greșit: $100e^{0.07} = @{c2.med}$ este mediana; media este $100e^{0.07 + 0.18^2/2} = @{c2.mean}$')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Ce rămîne de azi'), items(
    T('The Normal distribution makes 4-sigma days almost impossible; real markets have them every few months', 'Distribuția Normală face zilele de 4 sigma aproape imposibile; piețele reale le au la cîteva luni'),
    T('Lognormal prices: the median is below the mean by the volatility drag', 'Prețuri lognormale: mediana este sub medie cu volatility drag'),
    T('Skewness and kurtosis are fragile: report bootstrap intervals and the extremes', 'Asimetria și aplatizarea sînt fragile: raportați intervale bootstrap și extremele'),
    T('The Student-$t$ fits much better than the Normal distribution, but not perfectly', 'Distribuția Student-$t$ se potrivește mult mai bine decît distribuția Normală, dar nu perfect'),
    T('Returns are nearly uncorrelated but not independent: their size is predictable', 'Randamentele sînt aproape necorelate, dar nu independente: mărimea lor este previzibilă'),
    T('An AI answer is a draft: check every definition, model and number', 'Un răspuns AI este o ciornă: verificați fiecare definiție, model și cifră')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 2 develops each topic of today: the Normal distribution and lognormal models, the CLT, moments and tests, the Student-$t$ and six stylised facts',
      'Cursul 2 dezvoltă fiecare temă de azi: modelele Normal și lognormal, CLT, momente și teste, distribuția Student-$t$ și șase fapte stilizate'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: rolling windows, bootstrap intervals, more crypto assets', 'C1 poate deveni un proiect de echipă: ferestre mobile, intervale bootstrap, mai multe active cripto'),
    T('Reading: \\refFHH, Sec.~3.3 and Sec.~11.2; exercises with solutions: \\refFHHex', 'Lectură: \\refFHH, secț.~3.3 și secț.~11.2; exerciții rezolvate: \\refFHHex')))

D.references([
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \textit{Quantitative Finance}, 1(2), 223--236.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Jarque, C.M., Bera, A.K. (1987). \href{https://doi.org/10.2307/1403192}{A test for normality of observations and regression residuals}. \textit{International Statistical Review}, 55(2), 163--172.",
    r"Ljung, G.M., Box, G.E.P. (1978). \href{https://doi.org/10.1093/biomet/65.2.297}{On a measure of lack of fit in time series models}. \textit{Biometrika}, 65(2), 297--303.",
    r"Student (1908). \href{https://doi.org/10.1093/biomet/6.1.1}{The probable error of a mean}. \textit{Biometrika}, 6(1), 1--25.",
])

if __name__ == '__main__':
    D.write(V)
