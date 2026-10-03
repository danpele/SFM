r"""
build_seminar16.py -- Seminarul 16 (Recapitulare: exercițiu pentru examen), EN + RO dintr-o singură sursă
========================================================================================================
Seminarul are loc ÎNAINTEA cursului 16: secțiunea „Noțiuni necesare azi” este o fișă compactă de formule pentru toate
capitolele. Zece probleme mixte: Partea A pe hîrtie (randamente, cozi, eficiență, volatilitate, risc, credit și risc
sistemic), Partea B pe date reale (BET, S&P 500, Bitcoin), Partea C critica unui răspuns AI.
[Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea profesorului (*_solutions.tex,
exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_16/sem16_results.json (seminar16.py).
Ieșire:
  EN/Seminars/seminar16_review.tex           (+ _solutions.tex)
  RO/Seminarii/seminar16_recapitulare_ro.tex (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_16/seminar16.py && python3 latex/build_seminar16.py && python3 latex/sfm_build.py compile 16
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch16_common import REFS, T, bib, load_sem, money, pv, put_date   # noqa: E402

S = load_sem()
V = Values()
D = Deck(16, 'seminar', refs=REFS)
CLOSE = T('The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class',
          'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')


def qlsem():
    return '\\sfmquantlet{Ch_16}{SFM_ch16_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
a1 = A['a1']
for i in range(5):
    V.put(f'a1.R{i + 1}', a1['R'][i], 2)
    V.put(f'a1.r{i + 1}', a1['r'][i], 2)
for c, d in [('sum_r', 2), ('tot', 2), ('tot_from_r', 2), ('mdd', 2), ('gain', 2), ('mu_a', 2), ('sd_a', 2), ('sr', 3), ('sr_se', 2)]:
    V.put(f'a1.{c}', a1[c], d)
a2 = A['a2']
for c, d in [('jb_s', 1), ('jb_k', 0), ('jb', 1), ('nu', 2), ('mean_log', 4), ('alpha', 2), ('se', 2), ('scale', 2), ('sqrt', 2)]:
    V.put(f'a2.{c}', a2[c], d)
for i, x in enumerate(a2['logs']):
    V.put(f'a2.l{i + 1}', x, 3)
a3 = A['a3']
for c, d in [('lb', 2), ('crit', 2), ('p_lb', 3), ('sum_sq', 4), ('inner', 3), ('vr', 3), ('se', 4), ('z', 2), ('zs', 2)]:
    V.put(f'a3.{c}', a3[c], d)
a4 = A['a4']
for c, d in [('a_eps', 2), ('b_s2', 3), ('s2n', 3), ('pers', 2), ('hl', 1), ('lv', 2), ('ann', 2), ('ph', 4), ('sh', 3),
             ('ewma', 2), ('range', 3), ('park', 3), ('park_sd', 2), ('park_ann', 1)]:
    V.put(f'a4.{c}', a4[c], d)
a5 = A['a5']
for c, d in [('var_n', 2), ('es_n', 2), ('esf', 3), ('qz', 3), ('var_t', 2), ('l0', 2), ('l1', 2), ('lr', 2), ('p', 3),
             ('crit', 2), ('ci_lo', 1), ('ci_hi', 1)]:
    V.put(f'a5.{c}', a5[c], d)
for c in ('var_n_m', 'es_n_m', 'var_t_m'):
    V.raw(f'a5.{c}', money(a5[c]))
a6 = A['a6']
for c, d in [('eta', 2), ('pd', 2), ('odds', 3), ('or1', 2), ('auc', 2), ('covar', 2), ('cmed', 2), ('dcv', 2)]:
    V.put(f'a6.{c}', a6[c], d)
V.raw('a6.el', money(a6['el']))

b = S['B1']
p = b['perf']
put_date(V, 'b1.first', p['first'])
put_date(V, 'b1.last', p['last'])
V.int('b1.n', p['n'])
for c, d in [('mean_log', 1), ('vol', 1), ('sharpe', 2), ('sharpe_se', 2), ('mdd', 1), ('cagr', 1)]:
    V.put(f'b1.{c}', p[c], d)
put_date(V, 'b1.mddd', p['mdd_date'])
V.put('b1.skew', b['mom']['skew'], 2)
V.put('b1.k', b['mom']['exkurt'], 1)
V.int('b1.jb', round(b['mom']['jb']))
V.put('b1.hill', b['hill']['alpha'], 2)
V.put('b1.hillse', b['hill']['se'], 2)
V.raw('b1.hillk', str(b['hill']['k']))
V.put('b1.lbr', b['lb_r']['q'], 1)
V.put('b1.lbr2', b['lb_r2']['q'], 0)
V.put('b1.crit', b['lb_r']['crit'], 2)
V.put('b1.pers', b['garch']['pers'], 3)
V.put('b1.hl', b['garch']['half_life'], 1)
V.put('b1.nu', b['garch']['nu'], 2)
for c in ('var_hs', 'var_n', 'es_hs', 'es_n'):
    V.put(f'b1.{c}', b[c], 2)
V.put('b1.gap', 100 * (1 - b['var_n'] / b['var_hs']), 0)
V.put('b1.ewmax', b['ewma_max'], 1)
put_date(V, 'b1.ewmaxd', b['ewma_max_date'])
V.put('b1.ewlast', b['ewma_last'], 1)

for k, d in S['B2'].items():
    V.int(f'b2.{k}.n', d['n'])
    V.put(f'b2.{k}.exp', 0.01 * d['n'], 1)
    for m, mk in (('HS', 'hs'), ('EWMA', 'ew')):
        V.raw(f'b2.{k}.{mk}.x', str(d[m]['x']))
        V.put(f'b2.{k}.{mk}.rate', d[m]['rate'], 2)
        V.put(f'b2.{k}.{mk}.lr', d[m]['lr'], 2)
        V.raw(f'b2.{k}.{mk}.p', pv(d[m]['p']))
        V.raw(f'b2.{k}.{mk}.max', str(d[m]['max250']))
        V.put(f'b2.{k}.{mk}.mv', d[m]['mean_var'], 2)

for k, d in S['B3'].items():
    V.int(f'b3.{k}.n', d['n'])
    V.put(f'b3.{k}.vr', d['vr'], 3)
    V.put(f'b3.{k}.zs', d['zs'], 2)
    V.raw(f'b3.{k}.p', pv(d['p_zs']))
    V.put(f'b3.{k}.hr', d['H_r'], 2)
    V.put(f'b3.{k}.ha', d['H_abs'], 2)

c2 = S['C2']
for c, d in [('bet_var1', 2), ('bet_skew', 2), ('bet_k', 1), ('btc_sd', 2), ('btc_252', 1), ('btc_365', 1), ('sp_pers', 3),
             ('sp_hl', 0), ('sp_hl_wrong', 0), ('sp_H_r', 2), ('sp_H_abs', 2), ('sp_vr', 2), ('sp_zs', 2), ('norm_ratio', 3)]:
    V.put(f'c2.{c}', c2[c], d)
V.int('c2.bet_jb', round(c2['bet_jb']))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can you solve, on paper and on data, a problem from any chapter of the course?',
       '\\textbf{Întrebarea}: puteți rezolva, pe hîrtie și pe date, o problemă din orice capitol al cursului?'),
     [T('this seminar comes \\textbf{before} Lecture 16: the section ``What You Need for Today\'\' is a formula sheet for all chapters',
        'seminarul are loc \\textbf{înaintea} Cursului 16: secțiunea „Noțiuni necesare azi” este o fișă de formule pentru toate capitolele')]),
    (T('Route', 'Traseul'),
     [T('Part A: six exam-type problems on paper, from returns to systemic risk', 'Partea A: șase probleme de tip examen, pe hîrtie, de la randamente la riscul sistemic'),
      T('Part B: three data tasks on the BET, the S\\&P 500 and Bitcoin, each with an interpretation question', 'Partea B: trei cerințe pe datele BET, S\\&P 500 și Bitcoin, fiecare cu o întrebare de interpretare'),
      T('Part C: an AI answer to audit', 'Partea C: un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    CLOSE))

TB = '>{\\raggedright\\arraybackslash}'
SOL, PRO = T('Solved', 'Rezolvat'), T('Proposed', 'Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{0.8cm}' + TB + 'p{6.6cm}' + TB + 'p{1.3cm}' + TB + 'p{2.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Topic}', '\\textbf{Tema}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & ' + T('\\textbf{Chapters}', '\\textbf{Capitole}'),
    ['A1 & ' + T('returns, drawdown, annualisation, Sharpe ratio', 'randamente, drawdown, anualizare, raportul Sharpe') + f' & {SOL} & 1',
     'A2 & ' + T('Jarque--Bera, Student-$t$, Hill, stable scaling', 'Jarque--Bera, Student-$t$, Hill, scalarea stabilă') + f' & {PRO} & 2, 3, 5',
     'A3 & ' + T('Ljung--Box, variance ratio, robust test', 'Ljung--Box, raportul varianțelor, testul robust') + f' & {SOL} & 7',
     'A4 & ' + T('GARCH, EWMA and Parkinson volatility', 'volatilitatea GARCH, EWMA și Parkinson') + f' & {PRO} & 8, 9',
     'A5 & ' + T('VaR 1\\%, ES 2.5\\%, Kupiec, traffic light', 'VaR 1\\%, ES 2,5\\%, Kupiec, semaforul Basel') + f' & {PRO} & 10',
     'A6 & ' + T('logit PD, expected loss, AUC, $\\Delta$CoVaR', 'PD logit, pierderea așteptată, AUC, $\\Delta$CoVaR') + f' & {PRO} & 12, 15',
     'B1 & ' + T('a full check-up of the BET since 2015', 'o analiză completă a BET din 2015') + f' & {SOL} & 1--11',
     'B2 & ' + T('VaR 1\\% backtests: S\\&P 500 and Bitcoin', 'backtesting VaR 1\\%: S\\&P 500 și Bitcoin') + f' & {PRO} & 8, 10, 14',
     'B3 & ' + T('efficiency and memory in two windows', 'eficiență și memorie în două ferestre') + f' & {PRO} & 7, 11',
     'C2 & ' + T('audit an AI answer', 'verificarea unui răspuns AI') + f' & {PRO} & 0--15'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500 & EODHD & ' + T('close', 'închidere') + ' & 2000--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'închidere, 7 zile pe săptămînă') + ' & 2014--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, each series on its own calendar; the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, fiecare serie pe calendarul ei; ultima zi este 18 septembrie 2026'),
    T('Part A needs only a calculator; Part B uses the notebook, with the functions of the course already written (\\texttt{returns}, \\texttt{moments}, \\texttt{ljung\\_box}, \\texttt{variance\\_ratio}, \\texttt{garch\\_t}, \\texttt{kupiec}, \\texttt{hurst\\_rs})',
      'Partea A are nevoie doar de un calculator; Partea B folosește notebook-ul, cu funcțiile cursului deja scrise (\\texttt{returns}, \\texttt{moments}, \\texttt{ljung\\_box}, \\texttt{variance\\_ratio}, \\texttt{garch\\_t}, \\texttt{kupiec}, \\texttt{hurst\\_rs})')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')
FH = T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}')

D.frame(T('What You Need for Today (1/3): Returns, Distributions, Tails', 'Noțiuni necesare azi (1/3): randamente, distribuții, cozi'), '{\\renewcommand{\\arraystretch}{1.3}' + table(
    'll', FH,
    [T('Returns', 'Randamente') + ' & $R_t = P_t/P_{t-1} - 1$, \\ $r_t = \\ln(1 + R_t)$; ' + T('total', 'total') + ': $e^{\\sum r_t} - 1$',
     T('Annualisation', 'Anualizare') + ' & ' + T('mean $\\times q$, volatility $\\times \\sqrt{q}$; $q = 252$ (shares), 365 (crypto)', 'media $\\times q$, volatilitatea $\\times \\sqrt{q}$; $q = 252$ (acțiuni), 365 (cripto)'),
     T('Sharpe, drawdown', 'Sharpe, drawdown') + ' & $(\\mu - r_f)/\\sigma$, SE $\\sqrt{(1 + \\mathrm{SR}^2/2)/Y}$; \\ $D_t = P_t/\\max_{s \\le t}P_s - 1$, ' + T('recovery', 'revenire') + ' $d/(1 - d)$',
     'Jarque--Bera & $\\frac{n}{6}S^2 + \\frac{n}{24}K^2 \\sim \\chi^2(2)$, \\ $\\chi^2_{0.95}(2) = 5.99$',
     'Student-$t(\\nu)$ & ' + T('excess kurtosis $6/(\\nu - 4)$; unit-variance quantile $t_\\nu^{-1}(\\alpha)\\sqrt{(\\nu - 2)/\\nu}$', 'excesul de boltire $6/(\\nu - 4)$; cuantila cu varianță unitară $t_\\nu^{-1}(\\alpha)\\sqrt{(\\nu - 2)/\\nu}$'),
     'Hill & $\\hat\\alpha = [\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})]^{-1}$, SE $\\hat\\alpha/\\sqrt{k}$; ' + T('moments exist for $p < \\alpha$', 'momentele există pentru $p < \\alpha$'),
     T('Stable law', 'Lege stabilă') + ' & ' + T('sum of $h$ copies: scale $\\times h^{1/\\alpha}$ (Normal: $\\sqrt{h}$)', 'suma a $h$ copii: scala $\\times h^{1/\\alpha}$ (distribuția Normală: $\\sqrt{h}$)')],
    size='footnotesize') + '}')

D.frame(T('What You Need for Today (2/3): Efficiency, Volatility, Memory', 'Noțiuni necesare azi (2/3): eficiență, volatilitate, memorie'), '{\\renewcommand{\\arraystretch}{1.3}' + table(
    'll', FH,
    ['Ljung--Box & $Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k^2/(T - k) \\sim \\chi^2(m)$, \\ $\\chi^2_{0.95}(4) = 9.49$',
     T('Variance ratio', 'Raportul varianțelor') + ' & $\\mathrm{VR}(q) = 1 + 2\\sum_{k=1}^{q-1}(1 - k/q)\\hat\\rho_k$; \\ $Z = (\\mathrm{VR} - 1)/\\sqrt{2(2q-1)(q-1)/(3qT)}$',
     T('Robust VR test', 'Testul VR robust') + ' & $Z^* = (\\mathrm{VR} - 1)/\\mathrm{SE}_{rob}$; ' + T('reject at 5\\% if $|Z^*| > 1.96$', 'respingem la 5\\% dacă $|Z^*| > 1{,}96$'),
     'EWMA & $\\sigma_{t+1}^2 = \\lambda\\sigma_t^2 + (1 - \\lambda)r_t^2$, \\ $\\lambda = 0.94$',
     'GARCH(1,1) & $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$, \\ $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$, \\ $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$',
     T('GARCH forecast', 'Prognoza GARCH') + ' & $E_t\\sigma_{t+h}^2 = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
     'Parkinson & $\\sigma^2 = (\\ln H - \\ln L)^2/(4\\ln 2)$',
     'Hurst & $(R/S)_n \\propto n^H$; \\ ' + T('risk over $h$ days $\\times h^H$', 'riscul pe $h$ zile $\\times h^H$')],
    size='footnotesize') + '}')

D.frame(T('What You Need for Today (3/3): Risk, Backtesting, Credit, Systemic Risk', 'Noțiuni necesare azi (3/3): risc, backtesting, credit, risc sistemic'), '{\\renewcommand{\\arraystretch}{1.3}' + table(
    'll', FH,
    ['VaR, ES & $\\mathrm{VaR}_\\alpha = -q_\\alpha > 0$; \\ ' + T('Normal', 'Normal') + ': $-(\\mu + \\sigma z_\\alpha)$, \\ ES $-\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$',
     T('Values', 'Valori') + ' & $z_{0.01} = -2.326$, $z_{0.025} = -1.960$, $\\varphi(1.96) = 0.0584$, $t_5^{-1}(0.01) = -3.365$',
     'Kupiec & $LR_{uc} = -2[(n - x)\\ln(1 - \\alpha) + x\\ln\\alpha - (n - x)\\ln(1 - \\hat\\pi) - x\\ln\\hat\\pi] \\sim \\chi^2(1)$, \\ $\\chi^2_{0.95}(1) = 3.84$',
     T('Traffic light (250 days)', 'Semaforul Basel (250 de zile)') + ' & ' + T('green 0--4, yellow 5--9, red 10 or more exceptions', 'verde 0--4, galben 5--9, roșu 10 sau mai multe depășiri'),
     'Logit, EL & $\\mathrm{PD} = 1/(1 + e^{-\\eta})$, ' + T('odds ratio', 'raportul șanselor') + ' $e^{\\beta_j}$; \\ EL $=$ PD $\\times$ LGD $\\times$ EAD',
     'AUC, Gini & Gini $= 2\\,\\mathrm{AUC} - 1$',
     'CoVaR & ' + T('1\\% quantile regression $X_{sys} = a + bX_i$', 'regresia cuantilică la 1\\% $X_{sys} = a + bX_i$') + ': CoVaR $= -(a + b\\,q_\\alpha^i)$, \\ $\\Delta$CoVaR $= b\\,(q_{50}^i - q_\\alpha^i)$'],
    size='footnotesize') + '}')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Problems on Paper', 'Partea A: probleme pe hîrtie')

D.solved(T('A1: Returns, Drawdown and the Sharpe Ratio', 'A1: randamente, drawdown și raportul Sharpe'),
         items(T('A share closes at 50.0, 51.5, 49.8, 52.3, 47.9 and 50.6 RON on six consecutive days.', 'O acțiune se închide la 50,0; 51,5; 49,8; 52,3; 47,9 și 50,6 lei în șase zile consecutive.'),
               T('1. Compute the five simple and the five log returns, in \\%.', '1. Calculați cele cinci randamente simple și cele cinci randamente logaritmice, în \\%.'),
               T('2. Compute the total return from the prices and from $\\sum_t r_t$.', '2. Calculați randamentul total din prețuri și din $\\sum_t r_t$.'),
               T('3. Compute the maximum drawdown and the gain needed to return to the peak.', '3. Calculați drawdown-ul maxim și cîștigul necesar pentru revenirea la vîrf.'),
               T('4. A fund has a daily mean return of 0.04\\% and a daily standard deviation of 1.3\\%; with $r_f = 3\\%$ a year, compute its annual mean, its annual volatility, its Sharpe ratio and the SE of the Sharpe ratio over 10 years.',
                 '4. Un fond are randamentul mediu zilnic 0,04\\% și abaterea standard zilnică 1,3\\%; cu $r_f = 3\\%$ pe an, calculați media anuală, volatilitatea anuală, raportul Sharpe și SE a raportului Sharpe pe 10 ani.'),
               T('Report: fourteen numbers and one sentence.', 'Raportați: paisprezece valori și o frază.')),
         items(T('1. $R$: $@{a1.R1}$, $@{a1.R2}$, $@{a1.R3}$, $@{a1.R4}$, $@{a1.R5}$; $r$: $@{a1.r1}$, $@{a1.r2}$, $@{a1.r3}$, $@{a1.r4}$, $@{a1.r5}$', '1. $R$: $@{a1.R1}$; $@{a1.R2}$; $@{a1.R3}$; $@{a1.R4}$; $@{a1.R5}$; $r$: $@{a1.r1}$; $@{a1.r2}$; $@{a1.r3}$; $@{a1.r4}$; $@{a1.r5}$'),
               T('2. $50.6/50 - 1 = @{a1.tot}\\%$; $\\sum r_t = @{a1.sum_r}\\%$ and $e^{@{a1.sum_r}/100} - 1 = @{a1.tot_from_r}\\%$', '2. $50{,}6/50 - 1 = @{a1.tot}\\%$; $\\sum r_t = @{a1.sum_r}\\%$ și $e^{@{a1.sum_r}/100} - 1 = @{a1.tot_from_r}\\%$'),
               T('3. MDD $= 47.9/52.3 - 1 = @{a1.mdd}\\%$; gain $52.3/47.9 - 1 = @{a1.gain}\\%$', '3. MDD $= 47{,}9/52{,}3 - 1 = @{a1.mdd}\\%$; cîștigul $52{,}3/47{,}9 - 1 = @{a1.gain}\\%$'),
               T('4. $252 \\times 0.04 = @{a1.mu_a}\\%$; $\\sqrt{252} \\times 1.3 = @{a1.sd_a}\\%$; SR $= (@{a1.mu_a} - 3)/@{a1.sd_a} = @{a1.sr}$; SE $= \\sqrt{(1 + @{a1.sr}^2/2)/10} = @{a1.sr_se}$',
                 '4. $252 \\times 0{,}04 = @{a1.mu_a}\\%$; $\\sqrt{252} \\times 1{,}3 = @{a1.sd_a}\\%$; SR $= (@{a1.mu_a} - 3)/@{a1.sd_a} = @{a1.sr}$; SE $= \\sqrt{(1 + @{a1.sr}^2/2)/10} = @{a1.sr_se}$'),
               T('Interpretation: the Sharpe ratio is about one SE from zero: ten years of data cannot show that this fund beats the risk-free rate.', 'Interpretare: raportul Sharpe se află la circa o eroare standard de zero: zece ani de date nu pot arăta că fondul bate rata fără risc.')),
         size='scriptsize')

D.proposed(T('A2: Normality, Tails and Stable Scaling', 'A2: normalitate, cozi și scalare stabilă'),
           items(T('$n = 2000$ daily returns: skewness $S = -0.8$, excess kurtosis $K = 9$. The six largest losses, in \\%: 11.2, 8.4, 7.7, 6.0, 5.5, 4.9. Model: Seminar 2, A7 and Seminar 5, A1.',
                   '$n = 2000$ de randamente zilnice: asimetria $S = -0{,}8$, excesul de boltire $K = 9$. Cele mai mari șase pierderi, în \\%: 11,2; 8,4; 7,7; 6,0; 5,5; 4,9. Model: Seminarul 2, A7 și Seminarul 5, A1.'),
                 T('1. Compute JB, its skewness and kurtosis parts, and decide at 5\\%.', '1. Calculați JB, componentele asimetriei și boltirii, apoi decideți la 5\\%.'),
                 T('2. Compute $\\nu$ of a Student-$t$ distribution with the same excess kurtosis.', '2. Calculați $\\nu$ pentru o distribuție Student-$t$ cu același exces de boltire.'),
                 T('3. Compute the Hill estimate $\\hat\\alpha$ with $k = 5$ and its SE.', '3. Calculați estimarea Hill $\\hat\\alpha$ cu $k = 5$ și eroarea ei standard.'),
                 T('4. A stable law with $\\alpha = 1.6$ describes daily returns: compute the 20-day scale factor and compare it with $\\sqrt{20}$.', '4. O lege stabilă cu $\\alpha = 1{,}6$ descrie randamentele zilnice: calculați factorul de scală pe 20 de zile și comparați-l cu $\\sqrt{20}$.'),
                 T('Report: seven numbers, one decision and one sentence.', 'Raportați: șapte valori, o decizie și o frază.')),
           items(T('1. $\\frac{2000}{6}(0.64) = @{a2.jb_s}$, $\\frac{2000}{24}(81) = @{a2.jb_k}$, JB $= @{a2.jb} > 5.99$: normality is rejected', '1. $\\frac{2000}{6}(0{,}64) = @{a2.jb_s}$, $\\frac{2000}{24}(81) = @{a2.jb_k}$, JB $= @{a2.jb} > 5{,}99$: normalitatea se respinge'),
                 T('2. $6/(\\nu - 4) = 9 \\Rightarrow \\nu = @{a2.nu}$', '2. $6/(\\nu - 4) = 9 \\Rightarrow \\nu = @{a2.nu}$'),
                 T('3. logs over 4.9: $@{a2.l1}$, $@{a2.l2}$, $@{a2.l3}$, $@{a2.l4}$, $@{a2.l5}$; mean $@{a2.mean_log}$; $\\hat\\alpha = @{a2.alpha}$, SE $= @{a2.se}$', '3. logaritmii față de 4,9: $@{a2.l1}$; $@{a2.l2}$; $@{a2.l3}$; $@{a2.l4}$; $@{a2.l5}$; media $@{a2.mean_log}$; $\\hat\\alpha = @{a2.alpha}$, SE $= @{a2.se}$'),
                 T('4. $20^{1/1.6} = @{a2.scale}$ against $\\sqrt{20} = @{a2.sqrt}$', '4. $20^{1/1{,}6} = @{a2.scale}$, față de $\\sqrt{20} = @{a2.sqrt}$'),
                 T('The kurtosis part dominates; with $k = 5$ the Hill estimate is too noisy to decide whether the variance is finite.', 'Componenta boltirii domină; cu $k = 5$, estimarea Hill este prea zgomotoasă pentru a decide dacă varianța este finită.')),
           size='scriptsize')

D.solved(T('A3: Autocorrelation and the Variance Ratio', 'A3: autocorelație și raportul varianțelor'),
         items(T('$T = 1500$ daily returns; $\\hat\\rho_1 = -0.05$, $\\hat\\rho_2 = 0.03$, $\\hat\\rho_3 = 0.02$, $\\hat\\rho_4 = -0.01$; the robust SE of $\\widehat{\\mathrm{VR}}(5)$ is $0.055$.',
                 '$T = 1500$ de randamente zilnice; $\\hat\\rho_1 = -0{,}05$, $\\hat\\rho_2 = 0{,}03$, $\\hat\\rho_3 = 0{,}02$, $\\hat\\rho_4 = -0{,}01$; eroarea standard robustă a lui $\\widehat{\\mathrm{VR}}(5)$ este $0{,}055$.'),
               T('1. Compute the Ljung--Box statistic $Q(4)$ and decide at 5\\%.', '1. Calculați statistica Ljung--Box $Q(4)$ și decideți la 5\\%.'),
               T('2. Compute $\\mathrm{VR}(5)$.', '2. Calculați $\\mathrm{VR}(5)$.'),
               T('3. Compute $Z(5)$ and $Z^*(5)$ and decide at 5\\%.', '3. Calculați $Z(5)$ și $Z^*(5)$, apoi decideți la 5\\%.'),
               T('Report: four numbers, two decisions and one sentence.', 'Raportați: patru valori, două decizii și o frază.')),
         items(T('1. $\\sum\\hat\\rho_k^2 = @{a3.sum_sq}$; $Q(4) = @{a3.lb} < @{a3.crit}$ ($p = @{a3.p_lb}$): not rejected', '1. $\\sum\\hat\\rho_k^2 = @{a3.sum_sq}$; $Q(4) = @{a3.lb} < @{a3.crit}$ ($p = @{a3.p_lb}$): nu se respinge'),
               T('2. $1 + 2(-0.040 + 0.018 + 0.008 - 0.002) = 1 + 2 \\times (@{a3.inner}) = @{a3.vr}$', '2. $1 + 2(-0{,}040 + 0{,}018 + 0{,}008 - 0{,}002) = 1 + 2 \\times (@{a3.inner}) = @{a3.vr}$'),
               T('3. SE $= \\sqrt{72/22\\,500} = @{a3.se}$: $Z = @{a3.z}$; $Z^* = (@{a3.vr} - 1)/0.055 = @{a3.zs}$; both $|Z| < 1.96$: not rejected', '3. SE $= \\sqrt{72/22\\,500} = @{a3.se}$: $Z = @{a3.z}$; $Z^* = (@{a3.vr} - 1)/0{,}055 = @{a3.zs}$; ambele $|Z| < 1{,}96$: nu se respinge'),
               T('Interpretation: no evidence against the random walk at one week; weak-form efficiency is not rejected.', 'Interpretare: nicio dovadă împotriva mersului aleator pe o săptămînă; eficiența în formă slabă nu se respinge.')),
         size='scriptsize')

D.proposed(T('A4: Volatility Forecasts', 'A4: prognoze de volatilitate'),
           items(T('GARCH(1,1), returns in \\%: $\\omega = 0.05$, $\\alpha = 0.10$, $\\beta = 0.85$; today $\\sigma_t^2 = 2.5$ and $\\varepsilon_t = 1.0$. One day of an index: high 105.2, low 101.6. Model: Seminar 9, A1 and A3; Seminar 8, A3 and A5.',
                   'GARCH(1,1), randamente în \\%: $\\omega = 0{,}05$, $\\alpha = 0{,}10$, $\\beta = 0{,}85$; azi $\\sigma_t^2 = 2{,}5$ și $\\varepsilon_t = 1{,}0$. O zi a unui indice: maximul 105,2, minimul 101,6. Model: Seminarul 9, A1 și A3; Seminarul 8, A3 și A5.'),
                 T('1. Compute $\\sigma_{t+1}^2$ and the EWMA forecast with $\\lambda = 0.94$.', '1. Calculați $\\sigma_{t+1}^2$ și prognoza EWMA cu $\\lambda = 0{,}94$.'),
                 T('2. Compute the persistence, the half-life, the long-run variance and the long-run annual volatility.', '2. Calculați persistența, timpul de înjumătățire, varianța pe termen lung și volatilitatea anuală pe termen lung.'),
                 T('3. Compute $E_t[\\sigma_{t+5}^2]$.', '3. Calculați $E_t[\\sigma_{t+5}^2]$.'),
                 T('4. Compute the Parkinson variance of the day and the annualised Parkinson volatility.', '4. Calculați varianța Parkinson a zilei și volatilitatea Parkinson anualizată.'),
                 T('Report: eight numbers and one sentence.', 'Raportați: opt valori și o frază.')),
           items(T('1. $0.05 + @{a4.a_eps} + @{a4.b_s2} = @{a4.s2n}$; EWMA $0.94 \\times 2.5 + 0.06 \\times 1 = @{a4.ewma}$', '1. $0{,}05 + @{a4.a_eps} + @{a4.b_s2} = @{a4.s2n}$; EWMA $0{,}94 \\times 2{,}5 + 0{,}06 \\times 1 = @{a4.ewma}$'),
                 T('2. $\\alpha + \\beta = @{a4.pers}$; $h_{1/2} = \\ln 0.5/\\ln 0.95 = @{a4.hl}$ days; $\\bar\\sigma^2 = 0.05/0.05 = @{a4.lv}$; $\\sqrt{252} \\times 1 = @{a4.ann}\\%$', '2. $\\alpha + \\beta = @{a4.pers}$; $h_{1/2} = \\ln 0{,}5/\\ln 0{,}95 = @{a4.hl}$ zile; $\\bar\\sigma^2 = 0{,}05/0{,}05 = @{a4.lv}$; $\\sqrt{252} \\times 1 = @{a4.ann}\\%$'),
                 T('3. $1 + 0.95^4(@{a4.s2n} - 1) = 1 + @{a4.ph} \\times (@{a4.s2n} - 1) = @{a4.sh}$', '3. $1 + 0{,}95^4(@{a4.s2n} - 1) = 1 + @{a4.ph} \\times (@{a4.s2n} - 1) = @{a4.sh}$'),
                 T('4. $100\\ln(105.2/101.6) = @{a4.range}$; $@{a4.range}^2/(4\\ln 2) = @{a4.park}$; $\\sqrt{252 \\times @{a4.park}} = @{a4.park_ann}\\%$', '4. $100\\ln(105{,}2/101{,}6) = @{a4.range}$; $@{a4.range}^2/(4\\ln 2) = @{a4.park}$; $\\sqrt{252 \\times @{a4.park}} = @{a4.park_ann}\\%$'),
                 T('A calm day lowers the variance, which then moves back to its long-run level of 1; a single wide day says little about annual risk.', 'O zi calmă reduce varianța, care revine apoi spre nivelul pe termen lung de 1; o singură zi cu amplitudine mare spune puțin despre riscul anual.')),
           size='scriptsize')

D.proposed(T('A5: VaR, ES and Backtesting', 'A5: VaR, ES și backtesting'),
           items(T('A position of 2\\,000\\,000 RON; daily returns with mean 0.05\\% and standard deviation 1.2\\%. Model: Seminar 10, A1, A3 and A5.',
                   'O poziție de 2\\,000\\,000 de lei; randamente zilnice cu media 0,05\\% și abaterea standard 1,2\\%. Model: Seminarul 10, A1, A3 și A5.'),
                 T('1. Compute the Normal VaR 1\\% and ES 2.5\\%, in \\% and in RON.', '1. Calculați VaR 1\\% și ES 2,5\\% Normale, în \\% și în lei.'),
                 T('2. Compute VaR 1\\% if the standardised returns are Student-$t$ with $\\nu = 5$.', '2. Calculați VaR 1\\% dacă randamentele standardizate sînt Student-$t$ cu $\\nu = 5$.'),
                 T('3. A model has 17 exceptions of VaR 1\\% in 1000 days: compute $LR_{uc}$ and decide at 5\\%.', '3. Un model are 17 depășiri ale VaR 1\\% în 1000 de zile: calculați $LR_{uc}$ și decideți la 5\\%.'),
                 T('4. Give the Basel zone of 11 exceptions in 250 days.', '4. Precizați zona Basel pentru 11 depășiri în 250 de zile.'),
                 T('Report: seven numbers, two decisions and one sentence.', 'Raportați: șapte valori, două decizii și o frază.')),
           items(T('1. VaR $= -(0.05 - 2.326 \\times 1.2) = @{a5.var_n}\\%$ (@{a5.var_n_m} RON); ES $= -0.05 + 1.2 \\times @{a5.esf} = @{a5.es_n}\\%$ (@{a5.es_n_m} RON)', '1. VaR $= -(0{,}05 - 2{,}326 \\times 1{,}2) = @{a5.var_n}\\%$ (@{a5.var_n_m} de lei); ES $= -0{,}05 + 1{,}2 \\times @{a5.esf} = @{a5.es_n}\\%$ (@{a5.es_n_m} de lei)'),
                 T('2. $q = -3.365\\sqrt{3/5} = @{a5.qz}$; VaR $= -(0.05 + 1.2 \\times (@{a5.qz})) = @{a5.var_t}\\%$ (@{a5.var_t_m} RON)', '2. $q = -3{,}365\\sqrt{3/5} = @{a5.qz}$; VaR $= -(0{,}05 + 1{,}2 \\times (@{a5.qz})) = @{a5.var_t}\\%$ (@{a5.var_t_m} de lei)'),
                 T('3. $\\ell_0 = @{a5.l0}$, $\\ell_1 = @{a5.l1}$, $LR_{uc} = @{a5.lr} > @{a5.crit}$ ($p = @{a5.p}$): rejected; 17 is outside $10 \\pm 1.96\\sqrt{9.9} = [@{a5.ci_lo}, @{a5.ci_hi}]$', '3. $\\ell_0 = @{a5.l0}$, $\\ell_1 = @{a5.l1}$, $LR_{uc} = @{a5.lr} > @{a5.crit}$ ($p = @{a5.p}$): se respinge; 17 este în afara intervalului $10 \\pm 1{,}96\\sqrt{9{,}9} = [@{a5.ci_lo}; @{a5.ci_hi}]$'),
                 T('4. 11 exceptions: red zone (10 or more)', '4. 11 depășiri: zona roșie (10 sau mai multe)'),
                 T('The Student-$t$ tail raises VaR 1\\%; a rate of 1.7\\% is too high for a VaR 1\\% model.', 'Coada Student-$t$ crește VaR 1\\%; o rată de 1,7\\% este prea mare pentru un model VaR 1\\%.')),
           size='scriptsize')

D.proposed(T('A6: Credit Scoring and CoVaR', 'A6: scoring de credit și CoVaR'),
           items(T('Logit: $\\eta = -1.8 + 0.9x_1 - 0.6x_2$, applicant with $x_1 = 1.2$, $x_2 = 2.5$; LGD $= 40\\%$, EAD $= 150\\,000$ RON. A 1\\% quantile regression of the system return on a bank: $a = -1.8$, $b = 0.45$; bank: $q_{0.01} = -6.0\\%$, median $0.05\\%$. Model: Seminar 12, A1 and A5; Seminar 15, A1.',
                   'Logit: $\\eta = -1{,}8 + 0{,}9x_1 - 0{,}6x_2$, solicitant cu $x_1 = 1{,}2$, $x_2 = 2{,}5$; LGD $= 40\\%$, EAD $= 150\\,000$ de lei. Regresia cuantilică la 1\\% a randamentului sistemului în funcție de o bancă: $a = -1{,}8$, $b = 0{,}45$; banca: $q_{0,01} = -6{,}0\\%$, mediana $0{,}05\\%$. Model: Seminarul 12, A1 și A5; Seminarul 15, A1.'),
                 T('1. Compute the PD of the applicant and the odds ratio of $x_1$.', '1. Calculați PD pentru acest solicitant și raportul șanselor pentru $x_1$.'),
                 T('2. Compute the expected loss of the loan.', '2. Calculați pierderea așteptată a creditului.'),
                 T('3. A scorecard has Gini $= 0.56$: compute its AUC.', '3. Un scorecard are Gini $= 0{,}56$: calculați AUC.'),
                 T('4. Compute CoVaR 1\\% at the bank\'s VaR and at its median, and $\\Delta$CoVaR.', '4. Calculați CoVaR 1\\% cînd banca este la VaR-ul ei și la mediană, precum și $\\Delta$CoVaR.'),
                 T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
           items(T('1. $\\eta = -1.8 + 1.08 - 1.5 = @{a6.eta}$; PD $= 1/(1 + e^{2.22}) = @{a6.pd}\\%$; odds ratio $e^{0.9} = @{a6.or1}$', '1. $\\eta = -1{,}8 + 1{,}08 - 1{,}5 = @{a6.eta}$; PD $= 1/(1 + e^{2{,}22}) = @{a6.pd}\\%$; raportul șanselor $e^{0{,}9} = @{a6.or1}$'),
                 T('2. EL $= @{a6.pd}\\% \\times 0.40 \\times 150\\,000 = @{a6.el}$ RON', '2. EL $= @{a6.pd}\\% \\times 0{,}40 \\times 150\\,000 = @{a6.el}$ de lei'),
                 T('3. AUC $= (1 + 0.56)/2 = @{a6.auc}$', '3. AUC $= (1 + 0{,}56)/2 = @{a6.auc}$'),
                 T('4. CoVaR $= -(-1.8 + 0.45 \\times (-6.0)) = @{a6.covar}\\%$; median: $@{a6.cmed}\\%$; $\\Delta$CoVaR $= 0.45 \\times 6.05 = @{a6.dcv}$ pp', '4. CoVaR $= -(-1{,}8 + 0{,}45 \\times (-6{,}0)) = @{a6.covar}\\%$; la mediană: $@{a6.cmed}\\%$; $\\Delta$CoVaR $= 0{,}45 \\times 6{,}05 = @{a6.dcv}$ pp'),
                 T('A one-unit rise in $x_1$ multiplies the odds of default by @{a6.or1}; the distress of the bank raises the VaR of the system by @{a6.dcv} percentage points.', 'O creștere cu o unitate a lui $x_1$ multiplică șansa de nerambursare cu @{a6.or1}; dificultățile băncii cresc VaR-ul sistemului cu @{a6.dcv} puncte procentuale.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: A Full Check-Up of the BET [Solved]', 'B1: o analiză completă a BET [Rezolvat]'),
       T('what do Chapters 1--11 say about the BET since 2015?', 'ce spun Capitolele 1--11 despre BET din 2015?'),
       T('BET closes since 2 January 2015, daily log returns in \\%', 'închiderile BET din 2 ianuarie 2015, randamente logaritmice zilnice în \\%'),
       [T('Compute the annual mean, the volatility, the Sharpe ratio with its SE and the maximum drawdown.', 'Calculați media anuală, volatilitatea, raportul Sharpe cu SE și drawdown-ul maxim.'),
        T('Compute the skewness, the excess kurtosis, JB and the Hill tail index of the losses with $k = 2.5\\%$ of $n$.', 'Calculați asimetria, excesul de boltire, JB și indicele de coadă Hill al pierderilor cu $k = 2{,}5\\%$ din $n$.'),
        T('Compute Ljung--Box $Q(10)$ on $r_t$ and on $r_t^2$, and fit a GARCH(1,1)-$t$.', 'Calculați Ljung--Box $Q(10)$ pe $r_t$ și pe $r_t^2$, apoi estimați un GARCH(1,1)-$t$.'),
        T('Compute VaR 1\\% and ES 2.5\\% by historical simulation and with the Normal distribution.', 'Calculați VaR 1\\% și ES 2,5\\% prin simulare istorică și cu distribuția Normală.'),
        T('Interpretation: which three facts would you write in an exam answer about the BET?', 'Interpretare: ce trei fapte ați scrie într-un răspuns de examen despre BET?')],
       T('a table of about fifteen numbers, the chart and three sentences', 'un tabel cu circa cincisprezece valori, graficul și trei fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch16_sem_b1', h='0.32') + table(
    'llll', T('\\textbf{Performance}', '\\textbf{Performanță}') + ' & & ' + T('\\textbf{Distribution, dependence}', '\\textbf{Distribuție, dependență}') + ' &',
    [T('mean, \\% a year', 'media, \\% pe an') + ' & $@{b1.mean_log}$ & ' + T('skewness; excess kurtosis', 'asimetrie; exces de boltire') + ' & $@{b1.skew}$; $@{b1.k}$',
     T('volatility, \\% a year', 'volatilitate, \\% pe an') + ' & $@{b1.vol}$ & JB; Hill $\\hat\\alpha$ (SE) & @{b1.jb}; $@{b1.hill}$ ($@{b1.hillse}$)',
     T('Sharpe (SE)', 'Sharpe (SE)') + ' & $@{b1.sharpe}$ ($@{b1.sharpe_se}$) & $Q(10)$: $r_t$; $r_t^2$ & $@{b1.lbr}$; $@{b1.lbr2}$',
     'MDD & $@{b1.mdd}\\%$ & GARCH-$t$: $\\alpha + \\beta$; $h_{1/2}$ & $@{b1.pers}$; $@{b1.hl}$',
     T('VaR 1\\%: HS; Normal', 'VaR 1\\%: HS; Normal') + ' & $@{b1.var_hs}$; $@{b1.var_n}$ & ES 2.5\\%: HS; Normal & $@{b1.es_hs}$; $@{b1.es_n}$'],
    size='scriptsize') + items(
    T('$n = @{b1.n}$, @{b1.first} -- @{b1.last}; critical value $\\chi^2_{0.95}(10) = @{b1.crit}$; EWMA volatility peak @{b1.ewmax}\\% on @{b1.ewmaxd}',
      '$n = @{b1.n}$, @{b1.first} -- @{b1.last}; valoarea critică $\\chi^2_{0,95}(10) = @{b1.crit}$; vîrful volatilității EWMA @{b1.ewmax}\\% la @{b1.ewmaxd}'),
    T('Interpretation: (1) heavy tails: the Normal VaR 1\\% is @{b1.gap}\\% too low; (2) returns are weakly autocorrelated, squared returns strongly: volatility clustering; (3) the Sharpe ratio is less than three SE from zero',
      'Interpretare: (1) cozi groase: VaR 1\\% Normal este prea mic cu @{b1.gap}\\%; (2) randamentele sînt slab autocorelate, pătratele lor puternic: volatility clustering; (3) raportul Sharpe se află la mai puțin de trei erori standard de zero')) + qlsem(), 'scriptsize')

D.task(T('B2: Backtesting VaR 1\\% for the S\\&P 500 and Bitcoin [Proposed]', 'B2: backtesting VaR 1\\% pentru S\\&P 500 și Bitcoin [Propus]'),
       T('does a heavy-tailed method beat a Normal method with fast volatility? Model: B1 and Seminar 10, B3.', 'este mai bună o metodă cu cozi groase decît o metodă Normală cu volatilitate rapidă? Model: B1 și Seminarul 10, B3.'),
       T('S\\&P 500 and Bitcoin, all data; forecasts for every day since 1 January 2017', 'S\\&P 500 și Bitcoin, toate datele; prognoze pentru fiecare zi de la 1 ianuarie 2017'),
       [T('Compute the one-day VaR 1\\% by historical simulation on the last 500 returns.', 'Calculați VaR 1\\% pe o zi prin simulare istorică pe ultimele 500 de randamente.'),
        T('Compute the one-day EWMA-Normal VaR 1\\% with $\\lambda = 0.94$.', 'Calculați VaR 1\\% pe o zi EWMA-Normal, cu $\\lambda = 0{,}94$.'),
        T('Count the exceptions of each method and run the Kupiec test.', 'Numărați depășirile fiecărei metode și aplicați testul Kupiec.'),
        T('Find the largest number of exceptions in any 250 consecutive days.', 'Determinați cel mai mare număr de depășiri din orice 250 de zile consecutive.'),
        T('Interpretation: which method would a supervisor accept for each series?', 'Interpretare: ce metodă ar accepta o autoritate de supraveghere pentru fiecare serie?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k, name):
    return (f'{name} & ' + ' & '.join(f'@{{b2.{k}.{m}.x}} (@{{b2.{k}.{m}.rate}}\\%) & ${{@{{b2.{k}.{m}.lr}}}}$ & @{{b2.{k}.{m}.p}} & @{{b2.{k}.{m}.max}}' for m in ('hs', 'ew')))


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'l|rrrr|rrrr', T('& \\multicolumn{4}{c|}{HS (500 days)} & \\multicolumn{4}{c}{EWMA-Normal} \\\\ & exc. & $LR_{uc}$ & $p$ & max 250 & exc. & $LR_{uc}$ & $p$ & max 250',
                     '& \\multicolumn{4}{c|}{HS (500 de zile)} & \\multicolumn{4}{c}{EWMA-Normal} \\\\ & dep. & $LR_{uc}$ & $p$ & max 250 & dep. & $LR_{uc}$ & $p$ & max 250'),
    [b2row('sp500', 'S\\&P 500'), b2row('btc', 'Bitcoin')], size='scriptsize') + items(
    T('Expected exceptions: @{b2.sp500.exp} (S\\&P 500, @{b2.sp500.n} days) and @{b2.btc.exp} (Bitcoin, @{b2.btc.n} days)', 'Depășiri așteptate: @{b2.sp500.exp} (S\\&P 500, @{b2.sp500.n} de zile) și @{b2.btc.exp} (Bitcoin, @{b2.btc.n} de zile)'),
    T('EWMA-Normal has more than twice the target rate on both series: the Normal quantile is too small for heavy tails', 'EWMA-Normal are de peste două ori rata-țintă pe ambele serii: cuantila Normală este prea mică pentru cozi groase'),
    T('HS passes Kupiec, but on the S\\&P 500 it has up to @{b2.sp500.hs.max} exceptions in 250 days (red zone): it reacts slowly', 'HS trece testul Kupiec, dar pe S\\&P 500 are pînă la @{b2.sp500.hs.max} depășiri în 250 de zile (zona roșie): reacționează lent'),
    T('Interpretation: neither is acceptable on its own for the S\\&P 500; a model with both a volatility filter and heavy tails (GARCH-$t$, FHS, Chapter 10) is needed', 'Interpretare: niciuna nu este acceptabilă singură pentru S\\&P 500; este nevoie de un model cu filtru de volatilitate și cozi groase (GARCH-$t$, FHS, Capitolul 10)')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Efficiency and Memory in Two Windows [Proposed]', 'B3: eficiență și memorie în două ferestre [Propus]'),
       T('did the BET and the S\\&P 500 become more efficient after 2015? Model: A3; Seminar 7, B1 and Seminar 11, B1.', 'au devenit BET și S\\&P 500 mai eficiente după 2015? Model: A3; Seminarul 7, B1 și Seminarul 11, B1.'),
       T('BET and S\\&P 500, two windows: 2000--2014 and 2015--2026', 'BET și S\\&P 500, două ferestre: 2000--2014 și 2015--2026'),
       [T('Compute VR(5) and the robust $Z^*(5)$ in each window.', 'Calculați VR(5) și $Z^*(5)$ robust în fiecare fereastră.'),
        T('Compute the Hurst exponent (R/S) of $r_t$ and of $|r_t|$ in each window.', 'Calculați exponentul Hurst (R/S) pentru $r_t$ și pentru $|r_t|$ în fiecare fereastră.'),
        T('Interpretation: did the BET become more efficient after 2015?', 'Interpretare: a devenit BET mai eficient după 2015?')],
       T('one table of sixteen numbers and two sentences', 'un tabel cu șaisprezece valori și două fraze'), size='footnotesize', nb='B3')


def b3row(k, name, y):
    w = '2000--2014' if y == '2000' else '2015--2026'
    key = f'{k}_{y}'
    return f'{name}, {w} & @{{b3.{key}.n}} & ${{@{{b3.{key}.vr}}}}$ & ${{@{{b3.{key}.zs}}}}$ & @{{b3.{key}.p}} & ${{@{{b3.{key}.hr}}}}$ & ${{@{{b3.{key}.ha}}}}$'


D.frame(T('B3: Solution [Proposed]', 'B3: rezolvare [Propus]'), table(
    'lrrrrrr', T('& $n$ & VR(5) & $Z^*(5)$ & $p$ & $H$ of $r_t$ & $H$ of $|r_t|$', '& $n$ & VR(5) & $Z^*(5)$ & $p$ & $H$ pentru $r_t$ & $H$ pentru $|r_t|$'),
    [b3row('bet', 'BET', '2000'), b3row('bet', 'BET', '2015'), b3row('sp500', 'S\\&P 500', '2000'), b3row('sp500', 'S\\&P 500', '2015')],
    size='scriptsize') + items(
    T('BET: VR(5) above 1 in both windows (momentum); rejected at 5\\% before 2015, only borderline after', 'BET: VR(5) peste 1 în ambele ferestre (momentum); respins la 5\\% înainte de 2015, la limită după'),
    T('S\\&P 500: VR(5) below 1 (mean reversion), rejected before 2015, not after', 'S\\&P 500: VR(5) sub 1 (revenire la medie), respins înainte de 2015, nu și după'),
    T('$H$ of $|r_t|$ is about 0.75--0.84 in every window: the memory is in the volatility, not in the returns', '$H$ pentru $|r_t|$ este de circa 0,75--0,84 în fiecare fereastră: memoria se află în volatilitate, nu în randamente'),
    T('Interpretation: the BET is closer to weak-form efficiency after 2015, but one test at the 5\\% border is weak evidence; rolling windows and Monte Carlo bands would be the next step',
      'Interpretare: BET este mai aproape de eficiența în formă slabă după 2015, dar un test aflat la limita de 5\\% este o dovadă slabă; următorul pas ar fi ferestrele mobile și benzile Monte Carlo')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: AI Critique', 'Partea C: critica unui răspuns AI')

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the course for the exam, with the course data. The answer:', 'Un student a cerut unui asistent AI să rezume cursul pentru examen, pe datele cursului. Răspunsul:'),
    T('\\aiprompt{(a) The VaR 99\\% of the BET since 2015 is -@{c2.bet_var1}\\%.}', '\\aiprompt{(a) VaR 99\\% al BET din 2015 este -@{c2.bet_var1}\\%.}'),
    T('\\aiprompt{(b) The BET returns are close to Normal: the skewness is only @{c2.bet_skew}.}', '\\aiprompt{(b) Randamentele BET sînt apropiate de distribuția Normală: asimetria este doar @{c2.bet_skew}.}'),
    T('\\aiprompt{(c) Bitcoin has a daily standard deviation of @{c2.btc_sd}\\%, so its annual volatility is @{c2.btc_252}\\%.}', '\\aiprompt{(c) Bitcoin are o abatere standard zilnică de @{c2.btc_sd}\\%, deci volatilitatea anuală este @{c2.btc_252}\\%.}'),
    T('\\aiprompt{(d) For the S\\&P 500, alpha + beta = @{c2.sp_pers}, so a volatility shock halves in 1/(1 - @{c2.sp_pers}) = @{c2.sp_hl_wrong} days.}', '\\aiprompt{(d) Pentru S\\&P 500, alpha + beta = @{c2.sp_pers}, deci un șoc de volatilitate se înjumătățește în 1/(1 - @{c2.sp_pers}) = @{c2.sp_hl_wrong} de zile.}'),
    T('\\aiprompt{(e) The Hurst exponent of |r| is @{c2.sp_H_abs} for the S\\&P 500, so its returns are predictable and the market is inefficient.}', '\\aiprompt{(e) Exponentul Hurst pentru |r| este @{c2.sp_H_abs} la S\\&P 500, deci randamentele sînt previzibile, iar piața este ineficientă.}'),
    T('\\aiprompt{(f) For Normal returns, ES 2.5\\% and VaR 1\\% are almost the same number.}', '\\aiprompt{(f) Pentru randamente Normale, ES 2,5\\% și VaR 1\\% sînt practic același număr.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong twice: the level is the tail probability, VaR 1\\%, and VaR is a positive loss: VaR 1\\% $= @{c2.bet_var1}\\%$', '(a) Greșit de două ori: nivelul este probabilitatea cozii, VaR 1\\%, iar VaR este o pierdere pozitivă: VaR 1\\% $= @{c2.bet_var1}\\%$'),
    T('(b) Wrong: $S = @{c2.bet_skew}$ is far from 0 for daily data, the excess kurtosis is $@{c2.bet_k}$ and JB $=$ @{c2.bet_jb}', '(b) Greșit: $S = @{c2.bet_skew}$ este departe de 0 pentru date zilnice, excesul de boltire este $@{c2.bet_k}$, iar JB $=$ @{c2.bet_jb}'),
    T('(c) Wrong: Bitcoin trades 365 days a year: $\\sqrt{365} \\times @{c2.btc_sd} = @{c2.btc_365}\\%$', '(c) Greșit: Bitcoin se tranzacționează 365 de zile pe an: $\\sqrt{365} \\times @{c2.btc_sd} = @{c2.btc_365}\\%$'),
    T('(d) Wrong: the half-life is $\\ln 0.5/\\ln @{c2.sp_pers} = @{c2.sp_hl}$ days; $1/(1 - \\alpha - \\beta)$ is not a half-life', '(d) Greșit: timpul de înjumătățire este $\\ln 0{,}5/\\ln @{c2.sp_pers} = @{c2.sp_hl}$ zile; $1/(1 - \\alpha - \\beta)$ nu este un timp de înjumătățire'),
    T('(e) Wrong: memory in $|r_t|$ is memory of the volatility; for $r_t$, $H = @{c2.sp_H_r}$ and the robust VR test gives $Z^* = @{c2.sp_zs}$ (not rejected)', '(e) Greșit: memoria lui $|r_t|$ este memoria volatilității; pentru $r_t$, $H = @{c2.sp_H_r}$, iar testul VR robust dă $Z^* = @{c2.sp_zs}$ (nu se respinge)'),
    T('(f) Correct: $\\varphi(1.96)/0.025 = 2.338$ against $2.326$, a ratio of @{c2.norm_ratio}', '(f) Corect: $\\varphi(1{,}96)/0{,}025 = 2{,}338$, față de $2{,}326$, un raport de @{c2.norm_ratio}')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Every exam problem has three parts: the formula with the numbers, the result with its unit and sign, the interpretation', 'Orice problemă de examen are trei părți: formula cu cifrele înlocuite, rezultatul cu unitate și semn, interpretarea'),
    T('Tests: state $H_0$, the statistic, the critical value, the decision; prefer the robust version', 'Testele: precizați $H_0$, statistica, valoarea critică, decizia; preferați varianta robustă'),
    T('Risk: VaR 1\\% is a positive loss; the Normal distribution understates it on real data', 'Riscul: VaR 1\\% este o pierdere pozitivă; distribuția Normală îl subestimează pe date reale'),
    T('Volatility is predictable, the sign of returns almost not', 'Volatilitatea este previzibilă, semnul randamentelor aproape deloc'),
    T('An AI answer is a draft: check the level and the sign of VaR, the annualisation factor and every formula', 'Un răspuns AI este o ciornă: verificați nivelul și semnul VaR, factorul de anualizare și fiecare formulă')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 16 gathers the course: the course map, one recap slide per chapter, the toolbox, eight solved exam-type problems, the project',
      'Cursul 16 adună întreaga materie: harta cursului, cîte un slide de recapitulare pentru fiecare capitol, trusa de instrumente, opt probleme de tip examen rezolvate, proiectul'),
    T('Try the [Proposed] tasks on paper first, then check them in the notebook', 'Încercați cerințele [Propus] întîi pe hîrtie, apoi verificați-le în notebook'),
    T('B3 can grow into a team project: rolling windows, Monte Carlo bands, more emerging markets', 'B3 poate deveni un proiect de echipă: ferestre mobile, benzi Monte Carlo, mai multe piețe emergente'),
    T('Reading: \\refFHH; exercises in \\refBHL', 'Lectură: \\refFHH; exerciții în \\refBHL'),
    CLOSE))

D.references(bib(['BHL', 'FHH', 'Hill', 'JB', 'Kupiec', 'LM', 'Boll', 'AB', 'Hurst']), per=16)

if __name__ == '__main__':
    D.write(V)
