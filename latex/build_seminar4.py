r"""
build_seminar4.py -- Seminarul 4 (Probabilitate pentru finanțe), EN + RO dintr-o singură sursă
==============================================================================================
Seminarul are loc ÎNAINTEA cursului 4: secțiunea „Noțiuni necesare azi” conține tot ce folosesc cerințele.
Formatul A/B/C: A derivări și calcule pe hîrtie, B simulări și date reale, fiecare cu o întrebare de interpretare,
C o întrebare deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea
doar în versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_04/sem4_results.json (seminar4.py).
Ieșire:
  EN/Seminars/seminar4_probability.tex          (+ _solutions.tex)
  RO/Seminarii/seminar4_probabilitate_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_04/seminar4.py && python3 latex/build_seminar4.py && python3 latex/sfm_build.py compile 4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, fig   # noqa: E402
from ch4_common import REFS, T, load_sem                       # noqa: E402

S = load_sem()
V = Values()
D = Deck(4, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_04}{SFM_ch4_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
a1 = A['a1']
for c, d in [('mean', 2), ('m2', 2), ('var', 2), ('sd', 3)]:
    V.put(f'a1.{c}', a1[c], d)
a2 = A['a2']
for c, d in [('ex', 2), ('ey', 2), ('exy', 2), ('vx', 2), ('vy', 2), ('cov', 2), ('var_sum', 2)]:
    V.put(f'a2.{c}', a2[c], d)
for rho, k in [('0.3', 'p3'), ('0.0', 'z'), ('-0.5', 'm5'), ('1.0', 'one')]:
    V.put(f'a3.{k}', A['a3'][rho], 2)
V.put('a3.v', A['a3']['0.3'] ** 2, 1)
a4 = A['a4']
for c, d in [('var', 3), ('sd', 3), ('m4', 3), ('kurt', 2), ('exkurt', 2)]:
    V.put(f'a4.{c}', a4[c], d)
V.put('a4.p4', 100 * a4['p4'], 3)
a5 = A['a5']
for i in range(4):
    V.put(f'a5.s{i}', a5['S'][i], 1)
    V.put(f'a5.p{i}', a5['prob'][i], 4)
V.put('a5.E', a5['ES'], 2)
V.put('a5.pl', a5['ploss'], 4)
V.put('a5.ps', a5['p_star'], 2)
a6 = A['a6']
V.put('a6.mean', a6['mean'], 2)
V.put('a6.var', a6['var'], 4)
V.put('a6.r1', a6['rho'][0], 2)
V.put('a6.r2', a6['rho'][1], 2)
V.put('a6.r5', a6['rho'][2], 4)
V.put('a6.cm', a6['cond_mean'], 2)
a7 = A['a7']
V.put('a7.e1', a7['exp'][0], 3)
V.put('a7.e2', a7['exp'][1], 3)
V.put('a7.z01', a7['z01'], 3)
V.put('a7.n01', a7['norm01'], 3)
a8 = A['a8']
for T_ in ['T1', 'T5']:
    for c, d in [('mean', 2), ('median', 2), ('q', 2), ('z', 3), ('s', 3)]:
        V.put(f'a8.{T_}.{c}', a8[T_][c], d)
    V.put(f'a8.{T_}.pl', 100 * a8[T_]['ploss'], 2)

B1 = S['B1']
V.int('b1.n', B1['n'])
V.put('b1.r', B1['rho_r'], 3)
V.put('b1.r2', B1['rho_r2'], 3)
V.put('b1.band', B1['band'], 3)
V.put('b1.ratio', B1['ratio'], 2)
for i in range(5):
    V.put(f'b1.m{i + 1}', B1['mean'][i], 3)
    V.put(f'b1.s{i + 1}', B1['sd'][i], 2)
V.put('b1.within', B1['within'], 4)
V.put('b1.between', B1['between'], 5)
V.put('b1.total', B1['total'], 4)
V.put('b1.share', 100 * B1['between'] / B1['total'], 2)
B2 = S['B2']
for k, b in B2.items():
    V.put(f'b2.{k}.r', b['rho_r'], 3)
    V.put(f'b2.{k}.r2', b['rho_r2'], 3)
    V.put(f'b2.{k}.band', b['band'], 3)
    V.put(f'b2.{k}.sd1', b['sd1'], 2)
    V.put(f'b2.{k}.sd5', b['sd5'], 2)
    V.put(f'b2.{k}.ratio', b['ratio'], 2)
    V.put(f'b2.{k}.bs', 100 * b['between_share'], 2)
B3 = S['B3']
V.put('b3.exact', 100 * B3['exact'], 3)
V.put('b3.norm', 100 * B3['mc_norm'], 3)
V.put('b3.boot', 100 * B3['mc_boot'], 3)
V.put('b3.emp', 100 * B3['emp'], 2)
V.put('b3.se', 100 * B3['se'], 3)
V.put('b3.seb', 100 * B3['se_boot'], 3)
V.int('b3.nover', B3['n_over'])
V.raw('b3.hits', str(B3['n_hits_emp']))
V.put('b3.nonover', 100 * B3['emp_nonover'], 2)
V.raw('b3.nn', str(B3['n_nonover']))
V.put('b3.zb', (B3['mc_boot'] - B3['mc_norm']) / (2 ** 0.5 * B3['se']), 1)
B4 = S['B4']
for c in ['randu_mean', 'randu_var', 'pcg_mean', 'pcg_var']:
    V.put(f'b4.{c}', B4[c], 4)
V.put('b4.randu_rho', B4['randu_rho'], 4)
V.put('b4.pcg_rho', B4['pcg_rho'], 4)
V.raw('b4.rd', str(B4['randu_distinct']))
V.int('b4.pd', B4['pcg_distinct'])
for c in ['period_good', 'period_bad', 'period_mid']:
    V.raw(f'b4.{c}', str(B4[c]))
B5 = S['B5']
V.put('b5.sigma', 100 * B5['sigma'], 1)
V.put('b5.mulog', 100 * B5['mu_log'], 1)
V.int('b5.n', B5['n'])
V.put('b5.k', B5['real']['kurt'], 2)
V.put('b5.klo', B5['kurt']['q05'], 2)
V.put('b5.khi', B5['kurt']['q95'], 2)
V.put('b5.a', B5['real']['acf2'], 3)
V.put('b5.alo', B5['acf2']['q05'], 3)
V.put('b5.ahi', B5['acf2']['q95'], 3)
V.put('b5.dd', 100 * B5['real']['dd'], 1)
V.put('b5.ddlo', 100 * B5['dd']['q05'], 1)
V.put('b5.ddhi', 100 * B5['dd']['q95'], 1)
V.put('b5.ddmed', 100 * B5['dd']['q50'], 1)
B6 = S['B6']
for k in B6:
    for h in ['5', '21', '63']:
        V.put(f'b6.{k}.{h}', B6[k][h]['ratio'], 2)
        V.put(f'b6.{k}.{h}.se', B6[k][h]['se'], 2)
C1 = S['C1']
for k, c in C1.items():
    for f in ['real20', 'real40', 'gbm20', 'gbm40', 'boot20', 'boot40']:
        V.put(f'c1.{k}.{f}', 100 * c[f], 1)
    V.raw(f'c1.{k}.ny', str(c['n_years']))
C2 = S['C2']
V.put('c2.r', C2['rho_r'], 3)
V.put('c2.r2', C2['rho_r2'], 2)
V.put('c2.ratio', C2['ratio'], 2)
V.put('c2.mu', 100 * C2['mu'], 0)
V.put('c2.sigma', 100 * C2['sigma'], 0)
V.put('c2.mean10', C2['mean10'], 1)
V.put('c2.median10', C2['median10'], 1)
V.put('c2.se', C2['se_mc'], 4)
V.put('c2.corr', C2['corr_dax'], 2)
V.put('c2.sdsp', C2['sd_sp'], 2)
V.put('c2.sddax', C2['sd_dax'], 2)
V.put('c2.sdp', C2['sd_p'], 2)
V.put('c2.sdw', C2['sd_p_wrong'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we describe, condition and simulate random returns and prices?',
       '\\textbf{Întrebarea}: cum descriem, condiționăm și simulăm randamente și prețuri aleatoare?'),
     [T('this seminar comes \\textbf{before} Lecture 4: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 4: secțiunea „Noțiuni necesare azi” conține toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: moments, joint tables, conditioning, binomial trees, AR(1), inverse transform and GBM on paper',
        'Partea A: momente, tabele comune, condiționare, arbori binomiali, AR(1), transformare inversă și GBM pe hîrtie'),
      T('Part B: dependence in real returns, Monte Carlo, random number generators and GBM against data, each with an interpretation question',
        'Partea B: dependența în randamente reale, Monte Carlo, generatoare de numere aleatoare și GBM comparat cu datele, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.0cm}' + TB + 'p{7.6cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('moments of a discrete return; is an uncorrelated pair independent?', 'momentele unui randament discret; este independentă o pereche necorelată?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A1',
     'A3, A4 & ' + T('portfolio volatility; a two-regime model and the law of total variance', 'volatilitatea unui portofoliu; un model cu două regimuri și legea varianței totale') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A3',
     'A5, A6 & ' + T('a binomial tree; random walk, martingale and AR(1)', 'un arbore binomial; mers aleator, martingal și AR(1)') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A5',
     'A7, A8 & ' + T('inverse transform by hand; GBM and the Wiener process', 'transformarea inversă pas cu pas; GBM și procesul Wiener') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A7',
     'B1, B2 & ' + T('are returns independent? conditional mean and variance', 'sînt randamentele independente? media și varianța condiționate') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B1',
     'B3, B4 & ' + T('a loss probability by Monte Carlo; testing a random number generator', 'probabilitatea unei pierderi prin Monte Carlo; testarea unui generator de numere aleatoare') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B3',
     'B5, B6 & ' + T('GBM against the BET; does variance grow linearly with the horizon?', 'GBM față de BET; crește varianța liniar cu orizontul?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B5',
     'C1, C2 & ' + T('how often is there a bad year? what is wrong in an AI answer?', 'cît de des apare un an nefavorabil? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3, B5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX, BET & EODHD & close & 2000--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & 2014--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$', 'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$'),
    T('Weekends and repeated holiday closes are dropped (except Bitcoin); each series on its own calendar',
      'Weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); fiecare serie își păstrează propriul calendar'),
    T('Simulations use NumPy\'s default generator with fixed seeds, so every number can be reproduced',
      'Simulările folosesc generatorul implicit din NumPy cu semințe fixe, astfel încît fiecare rezultat numeric poate fi reprodus')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/4): Random Variables and Moments', 'Noțiuni necesare azi (1/4): variabile aleatoare și momente'), items(
    (T('Discrete $X$: \\textbf{PMF} (probability mass function) $p(x_i) = P(X = x_i)$; continuous $X$: \\textbf{PDF} (probability density function) $f$',
       '$X$ discretă: \\textbf{PMF} (probability mass function, funcția de masă de probabilitate) $p(x_i) = P(X = x_i)$; $X$ continuă: \\textbf{PDF} (probability density function, densitatea de probabilitate) $f$'),
     [T('\\textbf{CDF} (cumulative distribution function): $F(x) = P(X \\le x)$; quantile $q_\\alpha = F^{-1}(\\alpha)$',
        '\\textbf{CDF} (cumulative distribution function, funcția de repartiție): $F(x) = P(X \\le x)$; cuantila $q_\\alpha = F^{-1}(\\alpha)$')]),
    T('$E[X] = \\sum x_i p(x_i)$ or $\\int x f(x)\\,dx$; $\\text{Var}(X) = E[X^2] - (E[X])^2$', '$E[X] = \\sum x_i p(x_i)$ sau $\\int x f(x)\\,dx$; $\\text{Var}(X) = E[X^2] - (E[X])^2$'),
    T('$\\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$; $\\rho = \\text{Cov}/(\\sigma_X\\sigma_Y)$; $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}$',
      '$\\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$; $\\rho = \\text{Cov}/(\\sigma_X\\sigma_Y)$; $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}$'),
    (T('\\textbf{Independent}: $P(X = x, Y = y) = P(X = x)P(Y = y)$ for every cell; independent $\\Rightarrow$ uncorrelated, not conversely',
       '\\textbf{Independente}: $P(X = x, Y = y) = P(X = x)P(Y = y)$ pentru fiecare celulă; independente $\\Rightarrow$ necorelate, nu și invers'),
     [T('the i.i.d. band for a sample correlation: $\\pm 1.96/\\sqrt{n}$', 'banda de 95\\% pentru o corelație de selecție, în ipoteza i.i.d.: $\\pm 1.96/\\sqrt{n}$')])), 'footnotesize')

D.frame(T('What You Need for Today (2/4): Conditioning', 'Noțiuni necesare azi (2/4): condiționare'), items(
    T('$P(A \\mid B) = P(A \\cap B)/P(B)$; $E[Y \\mid X]$: the mean of $Y$ once $X$ is known, the best forecast of $Y$ from $X$',
      '$P(A \\mid B) = P(A \\cap B)/P(B)$; $E[Y \\mid X]$: media lui $Y$ după ce aflăm $X$, cea mai bună prognoză a lui $Y$ pe baza lui $X$'),
    T('\\textbf{Tower property}: $E[E[Y \\mid X]] = E[Y]$', '\\textbf{Proprietatea turnului}: $E[E[Y \\mid X]] = E[Y]$'),
    T('\\textbf{Law of total variance}: $\\text{Var}(Y) = E[\\text{Var}(Y \\mid X)] + \\text{Var}(E[Y \\mid X])$', '\\textbf{Legea varianței totale}: $\\text{Var}(Y) = E[\\text{Var}(Y \\mid X)] + \\text{Var}(E[Y \\mid X])$'),
    (T('For returns: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$ and $\\sigma_t^2 = \\text{Var}(r_t \\mid \\mathcal{F}_{t-1})$, with $\\mathcal{F}_{t-1}$ the information up to yesterday',
       'Pentru randamente: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$ și $\\sigma_t^2 = \\text{Var}(r_t \\mid \\mathcal{F}_{t-1})$, cu $\\mathcal{F}_{t-1}$ informația pînă ieri'),
     [T('a changing $\\sigma_t$ is what GARCH models (Chapter 9)', 'GARCH modelează tocmai variația în timp a lui $\\sigma_t$ (Capitolul 9)')]),
    T('A mixture of Normals with different variances has kurtosis above 3: $E[X^4] = 3\\sigma^4$ for $N(0, \\sigma^2)$', 'Un amestec de distribuții Normale cu varianțe diferite are aplatizarea peste 3: $E[X^4] = 3\\sigma^4$ pentru $N(0, \\sigma^2)$')), 'footnotesize')

D.frame(T('What You Need for Today (3/4): Processes', 'Noțiuni necesare azi (3/4): procese'), items(
    T('\\textbf{White noise}: mean 0, variance $\\sigma^2$, uncorrelated over time; \\textbf{i.i.d.} noise: independent and identically distributed',
      '\\textbf{Zgomot alb}: media 0, varianța $\\sigma^2$, necorelat în timp; zgomot \\textbf{i.i.d.}: independent și identic distribuit'),
    T('\\textbf{Random walk}: $S_t = S_{t-1} + \\varepsilon_t$, $E[S_t] = S_0$, $\\text{Var}(S_t) = t\\sigma^2$', '\\textbf{Mers aleator}: $S_t = S_{t-1} + \\varepsilon_t$, $E[S_t] = S_0$, $\\text{Var}(S_t) = t\\sigma^2$'),
    T('\\textbf{Martingale}: $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$ (a fair game)', '\\textbf{Martingal}: $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$ (un joc echitabil)'),
    T('\\textbf{AR(1)}: $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $|\\phi| < 1$: mean $c/(1 - \\phi)$, variance $\\sigma^2/(1 - \\phi^2)$, $\\rho(h) = \\phi^h$',
      '\\textbf{AR(1)}: $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $|\\phi| < 1$: media $c/(1 - \\phi)$, varianța $\\sigma^2/(1 - \\phi^2)$, $\\rho(h) = \\phi^h$'),
    (T('\\textbf{Binomial tree}: up by $u$ with probability $p$, down by $d$; after $n$ steps $S_n = S_0u^Kd^{n-K}$, $K \\sim B(n, p)$',
       '\\textbf{Arbore binomial}: prețul se înmulțește cu $u$ cu probabilitatea $p$ și cu $d$ în rest; după $n$ pași $S_n = S_0u^Kd^{n-K}$, $K \\sim B(n, p)$'),
     [T('$E[S_n] = S_0(pu + (1 - p)d)^n$; fair game for $p^* = (1 - d)/(u - d)$', '$E[S_n] = S_0(pu + (1 - p)d)^n$; joc echitabil pentru $p^* = (1 - d)/(u - d)$')])), 'footnotesize')

D.frame(T('What You Need for Today (4/4): Simulation, Wiener and GBM', 'Noțiuni necesare azi (4/4): simulare, Wiener și GBM'), items(
    T('\\textbf{Inverse transform}: $U \\sim U(0, 1)$ $\\Rightarrow$ $F^{-1}(U)$ has CDF $F$; exponential: $-\\ln(1 - U)/\\lambda$; Normal: $\\mu + \\sigma\\Phi^{-1}(U)$',
      '\\textbf{Transformarea inversă}: $U \\sim U(0, 1)$ $\\Rightarrow$ $F^{-1}(U)$ are CDF $F$; exponențiala: $-\\ln(1 - U)/\\lambda$; distribuția Normală: $\\mu + \\sigma\\Phi^{-1}(U)$'),
    T('\\textbf{LCG} (linear congruential generator): $x_{k+1} = (ax_k + c) \\bmod M$, $u_k = x_k/M$', '\\textbf{LCG} (linear congruential generator, generator congruențial liniar): $x_{k+1} = (ax_k + c) \\bmod M$, $u_k = x_k/M$'),
    T('\\textbf{Monte Carlo}: $\\hat p = $ share of simulations in the event; standard error $\\sqrt{p(1 - p)/N}$', '\\textbf{Monte Carlo}: $\\hat p = $ ponderea simulărilor în care are loc evenimentul; eroarea standard $\\sqrt{p(1 - p)/N}$'),
    T('\\textbf{Wiener process}: $W(0) = 0$, independent increments, $W(t) - W(s) \\sim N(0, t - s)$; $\\text{Cov}(W(s), W(t)) = \\min(s, t)$',
      '\\textbf{Procesul Wiener}: $W(0) = 0$, creșteri independente, $W(t) - W(s) \\sim N(0, t - s)$; $\\text{Cov}(W(s), W(t)) = \\min(s, t)$'),
    (T('\\textbf{GBM} (geometric Brownian motion): $S_T = S_0\\exp((\\mu - \\sigma^2/2)T + \\sigma W_T)$', '\\textbf{GBM} (geometric Brownian motion, mișcarea browniană geometrică): $S_T = S_0\\exp((\\mu - \\sigma^2/2)T + \\sigma W_T)$'),
     [T('mean $S_0e^{\\mu T}$, median $S_0e^{(\\mu - \\sigma^2/2)T}$; daily log returns i.i.d. Normal', 'media $S_0e^{\\mu T}$, mediana $S_0e^{(\\mu - \\sigma^2/2)T}$; randamente logaritmice zilnice Normale i.i.d.'),
      T('maximum drawdown: $\\min_t(P_t/\\max_{s \\le t}P_s - 1)$, the largest fall from a running peak', 'drawdown-ul maxim: $\\min_t(P_t/\\max_{s \\le t}P_s - 1)$, cea mai mare scădere față de maximul anterior')])), 'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations and Derivations on Paper', 'Partea A: calcule și derivări pe hîrtie')

D.solved(T('A1: Moments of a Discrete Return', 'A1: momentele unui randament discret'),
         items(T('A stock\'s daily return $X$ (in \\%) takes the values $-3$, $0$ and $2$ with probabilities $0.2$, $0.3$ and $0.5$.',
                 'Randamentul zilnic $X$ (în \\%) al unei acțiuni ia valorile $-3$, $0$ și $2$ cu probabilitățile $0.2$, $0.3$ și $0.5$.'),
               T('1. Compute $E[X]$, $E[X^2]$, $\\text{Var}(X)$ and the standard deviation.', '1. Calculați $E[X]$, $E[X^2]$, $\\text{Var}(X)$ și abaterea standard.'),
               T('2. Write the CDF $F(x)$ for every $x$.', '2. Scrieți CDF $F(x)$ pentru orice $x$.'),
               T('3. Compute $P(X < 0)$ and the median.', '3. Calculați $P(X < 0)$ și mediana.'),
               T('Report: four moments, the CDF and two numbers.', 'Raportați: patru momente, CDF și două valori.')),
         items(T('1. $E[X] = -0.6 + 0 + 1 = @{a1.mean}$; $E[X^2] = 1.8 + 2 = @{a1.m2}$; $\\text{Var} = @{a1.m2} - @{a1.mean}^2 = @{a1.var}$; s.d. $@{a1.sd}$',
                 '1. $E[X] = -0.6 + 0 + 1 = @{a1.mean}$; $E[X^2] = 1.8 + 2 = @{a1.m2}$; $\\text{Var} = @{a1.m2} - @{a1.mean}^2 = @{a1.var}$; abaterea standard $@{a1.sd}$'),
               T('2. $F(x) = 0$ for $x < -3$; $0.2$ on $[-3, 0)$; $0.5$ on $[0, 2)$; $1$ for $x \\ge 2$', '2. $F(x) = 0$ pentru $x < -3$; $0.2$ pe $[-3, 0)$; $0.5$ pe $[0, 2)$; $1$ pentru $x \\ge 2$'),
               T('3. $P(X < 0) = 0.2$; median $= \\inf\\{x : F(x) \\ge 0.5\\} = 0$', '3. $P(X < 0) = 0.2$; mediana $= \\inf\\{x : F(x) \\ge 0.5\\} = 0$'),
               T('The mean is positive although a loss is possible: risk and expected return are different questions.', 'Media este pozitivă deși o pierdere este posibilă: riscul și randamentul așteptat sînt întrebări diferite.')),
         size='footnotesize')

D.proposed(T('A2: Uncorrelated but Dependent', 'A2: necorelate, dar dependente'),
           items(T('$X$ = market move ($-1$ fall, $0$ flat, $+1$ rise); $Y = 1$ if the day is ``large\'\'. Joint probabilities: $P(-1, 1) = 0.25$, $P(0, 0) = 0.5$, $P(1, 1) = 0.25$, all other cells 0. Model: A1.',
                   '$X$ = mișcarea pieței ($-1$ scădere, $0$ stagnare, $+1$ creștere); $Y = 1$ dacă ziua este „mare”. Probabilitățile comune: $P(-1, 1) = 0.25$, $P(0, 0) = 0.5$, $P(1, 1) = 0.25$, celelalte celule au probabilitatea 0. Model: A1.'),
                 T('1. Compute the marginal distributions of $X$ and $Y$ and their means.', '1. Calculați distribuțiile marginale ale lui $X$ și $Y$ și mediile lor.'),
                 T('2. Compute $\\text{Cov}(X, Y)$ and $\\rho$.', '2. Calculați $\\text{Cov}(X, Y)$ și $\\rho$.'),
                 T('3. Decide whether $X$ and $Y$ are independent, using one cell of the table.', '3. Decideți dacă $X$ și $Y$ sînt independente, folosind o celulă a tabelului.'),
                 T('4. Compute $\\text{Var}(X + Y)$.', '4. Calculați $\\text{Var}(X + Y)$.'),
                 T('Report: two marginals, $\\rho$, a verdict and a variance.', 'Raportați: două distribuții marginale, $\\rho$, o concluzie și o varianță.')),
           items(T('1. $X$: $0.25, 0.5, 0.25$, $E[X] = @{a2.ex}$; $Y$: $0.5, 0.5$, $E[Y] = @{a2.ey}$', '1. $X$: $0.25, 0.5, 0.25$, $E[X] = @{a2.ex}$; $Y$: $0.5, 0.5$, $E[Y] = @{a2.ey}$'),
                 T('2. $E[XY] = -0.25 + 0.25 = @{a2.exy}$, $\\text{Cov} = @{a2.cov}$, $\\rho = 0$', '2. $E[XY] = -0.25 + 0.25 = @{a2.exy}$, $\\text{Cov} = @{a2.cov}$, $\\rho = 0$'),
                 T('3. $P(X = 0, Y = 0) = 0.5 \\ne P(X = 0)P(Y = 0) = 0.25$: dependent ($Y = |X|$)', '3. $P(X = 0, Y = 0) = 0.5 \\ne P(X = 0)P(Y = 0) = 0.25$: dependente ($Y = |X|$)'),
                 T('4. $\\text{Var}(X) + \\text{Var}(Y) + 2\\,\\text{Cov} = @{a2.vx} + @{a2.vy} + 0 = @{a2.var_sum}$', '4. $\\text{Var}(X) + \\text{Var}(Y) + 2\\,\\text{Cov} = @{a2.vx} + @{a2.vy} + 0 = @{a2.var_sum}$'),
                 T('The same structure as returns and their absolute values.', 'Aceeași structură o au randamentele și valorile lor absolute.')),
           size='scriptsize')

D.solved(T('A3: Volatility of a Two-Asset Portfolio', 'A3: volatilitatea unui portofoliu cu două active'),
         items(T('Two assets with annual volatilities $\\sigma_1 = 20\\%$ and $\\sigma_2 = 30\\%$; weights $0.6$ and $0.4$.', 'Două active cu volatilitățile anuale $\\sigma_1 = 20\\%$ și $\\sigma_2 = 30\\%$; ponderile $0.6$ și $0.4$.'),
               T('1. Compute the portfolio volatility for $\\rho = 0.3$.', '1. Calculați volatilitatea portofoliului pentru $\\rho = 0.3$.'),
               T('2. Repeat for $\\rho = 0$, $\\rho = -0.5$ and $\\rho = 1$.', '2. Repetați pentru $\\rho = 0$, $\\rho = -0.5$ și $\\rho = 1$.'),
               T('3. Explain when the portfolio is less risky than the weighted average of the two volatilities.', '3. Explicați cînd portofoliul este mai puțin riscant decît media ponderată a celor două volatilități.'),
               T('Report: four volatilities and one sentence.', 'Raportați: patru volatilități și o frază.')),
         items(T('1. $\\sigma_p^2 = 0.36 \\times 400 + 0.16 \\times 900 + 2 \\times 0.24 \\times 0.3 \\times 600 = @{a3.v}$; $\\sigma_p = @{a3.p3}\\%$',
                 '1. $\\sigma_p^2 = 0.36 \\times 400 + 0.16 \\times 900 + 2 \\times 0.24 \\times 0.3 \\times 600 = @{a3.v}$; $\\sigma_p = @{a3.p3}\\%$'),
               T('2. $\\rho = 0$: $@{a3.z}\\%$; $\\rho = -0.5$: $@{a3.m5}\\%$; $\\rho = 1$: $@{a3.one}\\%$', '2. $\\rho = 0$: $@{a3.z}\\%$; $\\rho = -0.5$: $@{a3.m5}\\%$; $\\rho = 1$: $@{a3.one}\\%$'),
               T('3. Only $\\rho = 1$ gives the weighted average $0.6 \\times 20 + 0.4 \\times 30 = 24\\%$; every $\\rho < 1$ diversifies.', '3. Doar $\\rho = 1$ dă media ponderată $0.6 \\times 20 + 0.4 \\times 30 = 24\\%$; orice $\\rho < 1$ aduce un cîștig din diversificare.')),
         size='footnotesize')

D.proposed(T('A4: Two Regimes and the Law of Total Variance', 'A4: două regimuri și legea varianței totale'),
           items(T('A day is calm with probability $0.8$, $r \\mid \\text{calm} \\sim N(0, 0.8^2)$, and turbulent otherwise, $r \\mid \\text{turbulent} \\sim N(0, 2^2)$ (in \\%). Model: A3.',
                   'O zi este calmă cu probabilitatea $0.8$, $r \\mid \\text{calm} \\sim N(0, 0.8^2)$, și agitată în rest, $r \\mid \\text{agitat} \\sim N(0, 2^2)$ (în \\%). Model: A3.'),
                 T('1. Use the law of total variance to compute $\\text{Var}(r)$ and the standard deviation.', '1. Folosiți legea varianței totale pentru a calcula $\\text{Var}(r)$ și abaterea standard.'),
                 T('2. Compute $E[r^4]$ with $E[Z^4] = 3\\sigma^4$ in each regime, then the kurtosis $E[r^4]/\\text{Var}(r)^2$.', '2. Calculați $E[r^4]$ cu $E[Z^4] = 3\\sigma^4$ în fiecare regim, apoi aplatizarea $E[r^4]/\\text{Var}(r)^2$.'),
                 T('3. Explain why the mixture has heavy tails although each regime is Normal.', '3. Explicați de ce amestecul are cozi groase, deși fiecare regim este Normal.'),
                 T('Report: a variance, a kurtosis and one sentence.', 'Raportați: o varianță, o aplatizare și o frază.')),
           items(T('1. $E[\\text{Var}] = 0.8 \\times 0.64 + 0.2 \\times 4 = @{a4.var}$, $\\text{Var}(E) = 0$; s.d. $@{a4.sd}\\%$', '1. $E[\\text{Var}] = 0.8 \\times 0.64 + 0.2 \\times 4 = @{a4.var}$, $\\text{Var}(E) = 0$; abaterea standard $@{a4.sd}\\%$'),
                 T('2. $E[r^4] = 3(0.8 \\times 0.8^4 + 0.2 \\times 2^4) = @{a4.m4}$; kurtosis $@{a4.kurt}$, excess $@{a4.exkurt}$', '2. $E[r^4] = 3(0.8 \\times 0.8^4 + 0.2 \\times 2^4) = @{a4.m4}$; aplatizarea $@{a4.kurt}$, excesul $@{a4.exkurt}$'),
                 T('3. Large moves come almost only from the turbulent regime; $P(|r| > 4\\sigma) = @{a4.p4}\\%$ vs $0.006\\%$ for one Normal', '3. Mișcările mari provin aproape numai din regimul agitat; $P(|r| > 4\\sigma) = @{a4.p4}\\%$ față de $0.006\\%$ pentru o singură distribuție Normală'),
                 T('This is the GARCH mechanism in its simplest form.', 'Acesta este mecanismul GARCH în forma lui cea mai simplă.')),
           size='scriptsize')

D.solved(T('A5: A Three-Step Binomial Tree', 'A5: un arbore binomial cu trei pași'),
         items(T('$S_0 = 100$, $u = 1.1$, $d = 0.9$, $p = 0.55$, three independent steps.', '$S_0 = 100$, $u = 1.1$, $d = 0.9$, $p = 0.55$, trei pași independenți.'),
               T('1. List the possible values of $S_3$ and their probabilities.', '1. Listați valorile posibile ale lui $S_3$ și probabilitățile lor.'),
               T('2. Compute $E[S_3]$ in two ways: from the table and from $S_0(pu + (1 - p)d)^3$.', '2. Calculați $E[S_3]$ în două moduri: din tabel și din $S_0(pu + (1 - p)d)^3$.'),
               T('3. Compute $P(S_3 < 100)$.', '3. Calculați $P(S_3 < 100)$.'),
               T('4. Find the $p^*$ that makes $S_t$ a martingale.', '4. Găsiți $p^*$ care face din $S_t$ un martingal.'),
               T('Report: a table, two expectations, a probability, $p^*$.', 'Raportați: un tabel, două speranțe, o probabilitate, $p^*$.')),
         items(T('1. $K = 0..3$ up moves: $@{a5.s0}, @{a5.s1}, @{a5.s2}, @{a5.s3}$ with $@{a5.p0}, @{a5.p1}, @{a5.p2}, @{a5.p3}$ ($\\binom{3}{k}p^k(1-p)^{3-k}$)',
                 '1. $K = 0..3$ creșteri: $@{a5.s0}, @{a5.s1}, @{a5.s2}, @{a5.s3}$ cu $@{a5.p0}, @{a5.p1}, @{a5.p2}, @{a5.p3}$ ($\\binom{3}{k}p^k(1-p)^{3-k}$)'),
               T('2. $E[S_3] = 100 \\times 1.01^3 = @{a5.E}$ both ways', '2. $E[S_3] = 100 \\times 1.01^3 = @{a5.E}$ în ambele moduri'),
               T('3. $P(K \\le 1) = @{a5.pl}$', '3. $P(K \\le 1) = @{a5.pl}$'),
               T('4. $p^* = (1 - 0.9)/(1.1 - 0.9) = @{a5.ps}$: with $p = 0.55 > p^*$ the price drifts up', '4. $p^* = (1 - 0.9)/(1.1 - 0.9) = @{a5.ps}$: cu $p = 0.55 > p^*$ prețul are o tendință de creștere')),
         size='footnotesize')

D.proposed(T('A6: Random Walk, Martingale and AR(1)', 'A6: mers aleator, martingal și AR(1)'),
           items(T('$S_t = S_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ i.i.d. with mean 0 and variance $\\sigma^2$; $X_t = 0.2 + 0.6X_{t-1} + \\varepsilon_t$ with $\\sigma = 1$. Model: A5.',
                   '$S_t = S_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ i.i.d. cu media 0 și varianța $\\sigma^2$; $X_t = 0.2 + 0.6X_{t-1} + \\varepsilon_t$ cu $\\sigma = 1$. Model: A5.'),
                 T('1. Compute $E[S_t]$ and $\\text{Var}(S_t)$, and show that $S_t$ is a martingale.', '1. Calculați $E[S_t]$ și $\\text{Var}(S_t)$ și arătați că $S_t$ este un martingal.'),
                 T('2. Show that $S_t^2 - t\\sigma^2$ is also a martingale.', '2. Arătați că și $S_t^2 - t\\sigma^2$ este un martingal.'),
                 T('3. Compute the mean, the variance and $\\rho(1), \\rho(2), \\rho(5)$ of the stationary $X_t$.', '3. Calculați media, varianța și $\\rho(1), \\rho(2), \\rho(5)$ pentru $X_t$ staționar.'),
                 T('4. Compute $E[X_{t+1} \\mid X_t = 2]$ and $\\text{Var}(X_{t+1} \\mid X_t = 2)$.', '4. Calculați $E[X_{t+1} \\mid X_t = 2]$ și $\\text{Var}(X_{t+1} \\mid X_t = 2)$.'),
                 T('Report: two derivations, five moments, a forecast.', 'Raportați: două demonstrații, cinci momente, o prognoză.')),
           items(T('1. $E[S_t] = S_0$, $\\text{Var}(S_t) = t\\sigma^2$; $E[S_{t+1} \\mid \\mathcal{F}_t] = S_t + E[\\varepsilon_{t+1}] = S_t$', '1. $E[S_t] = S_0$, $\\text{Var}(S_t) = t\\sigma^2$; $E[S_{t+1} \\mid \\mathcal{F}_t] = S_t + E[\\varepsilon_{t+1}] = S_t$'),
                 T('2. $E[(S_t + \\varepsilon_{t+1})^2 \\mid \\mathcal{F}_t] = S_t^2 + \\sigma^2$, so $E[S_{t+1}^2 - (t+1)\\sigma^2 \\mid \\mathcal{F}_t] = S_t^2 - t\\sigma^2$', '2. $E[(S_t + \\varepsilon_{t+1})^2 \\mid \\mathcal{F}_t] = S_t^2 + \\sigma^2$, deci $E[S_{t+1}^2 - (t+1)\\sigma^2 \\mid \\mathcal{F}_t] = S_t^2 - t\\sigma^2$'),
                 T('3. Mean $0.2/0.4 = @{a6.mean}$; variance $1/0.64 = @{a6.var}$; $\\rho = @{a6.r1}, @{a6.r2}, @{a6.r5}$', '3. Media $0.2/0.4 = @{a6.mean}$; varianța $1/0.64 = @{a6.var}$; $\\rho = @{a6.r1}, @{a6.r2}, @{a6.r5}$'),
                 T('4. $0.2 + 0.6 \\times 2 = @{a6.cm}$, conditional variance 1: the forecast is pulled back towards the mean @{a6.mean}', '4. $0.2 + 0.6 \\times 2 = @{a6.cm}$, varianța condiționată 1: prognoza revine spre media @{a6.mean}')),
           size='scriptsize')

D.solved(T('A7: The Inverse Transform Step by Step', 'A7: transformarea inversă pas cu pas'),
         items(T('You have the uniform numbers $u = 0.3$ and $u = 0.9$ (and $u = 0.15, 0.45, 0.8$ for question 2).', 'Aveți numerele uniforme $u = 0.3$ și $u = 0.9$ (și $u = 0.15, 0.45, 0.8$ pentru punctul 2).'),
               T('1. Turn $u = 0.3$ and $u = 0.9$ into exponential draws with rate $\\lambda = 0.5$.', '1. Transformați $u = 0.3$ și $u = 0.9$ în extrageri din distribuția exponențială cu rata $\\lambda = 0.5$.'),
               T('2. Turn $u = 0.15, 0.45, 0.8$ into draws of the return $X$ of A1.', '2. Transformați $u = 0.15, 0.45, 0.8$ în extrageri ale randamentului $X$ din A1.'),
               T('3. Turn $u = 0.01$ into a draw of $N(0.05, 1.2^2)$.', '3. Transformați $u = 0.01$ într-o extragere din $N(0.05, 1.2^2)$.'),
               T('Report: six simulated values.', 'Raportați: șase valori simulate.')),
         items(T('1. $x = -\\ln(1 - u)/0.5$: $@{a7.e1}$ and $@{a7.e2}$', '1. $x = -\\ln(1 - u)/0.5$: $@{a7.e1}$ și $@{a7.e2}$'),
               T('2. $F^{-1}(u) = \\inf\\{x : F(x) \\ge u\\}$ with $F = 0.2, 0.5, 1$: $-3$, $0$ and $2$', '2. $F^{-1}(u) = \\inf\\{x : F(x) \\ge u\\}$ cu $F = 0.2, 0.5, 1$: $-3$, $0$ și $2$'),
               T('3. $0.05 + 1.2 \\times (@{a7.z01}) = @{a7.n01}$: the 1\\% quantile', '3. $0.05 + 1.2 \\times (@{a7.z01}) = @{a7.n01}$: cuantila de 1\\%'),
               T('A large $u$ gives a large draw: the inverse transform keeps the order of the uniforms.', 'Un $u$ mare dă o extragere mare: transformarea inversă păstrează ordinea numerelor uniforme.')),
         size='footnotesize')

D.proposed(T('A8: GBM and the Wiener Process', 'A8: GBM și procesul Wiener'),
           items(T('An index follows GBM with $\\mu = 8\\%$, $\\sigma = 20\\%$ a year, $S_0 = 100$; $W$ is a Wiener process. Model: A7.', 'Un indice urmează GBM cu $\\mu = 8\\%$, $\\sigma = 20\\%$ pe an, $S_0 = 100$; $W$ este un proces Wiener. Model: A7.'),
                 T('1. Compute the mean and the median of $S_1$ and of $S_5$.', '1. Calculați media și mediana lui $S_1$ și ale lui $S_5$.'),
                 T('2. Compute $P(S_T < 100)$ for $T = 1$ and $T = 5$.', '2. Calculați $P(S_T < 100)$ pentru $T = 1$ și $T = 5$.'),
                 T('3. Compute the 5\\% quantile of $S_1$ and of $S_5$.', '3. Calculați cuantila de 5\\% a lui $S_1$ și a lui $S_5$.'),
                 T('4. Compute $\\text{Var}(W(2) - W(0.5))$ and $\\text{Cov}(W(1), W(3))$.', '4. Calculați $\\text{Var}(W(2) - W(0.5))$ și $\\text{Cov}(W(1), W(3))$.'),
                 T('Report: eight GBM numbers and two Wiener moments.', 'Raportați: opt valori pentru GBM și două momente ale procesului Wiener.')),
           items(T('1. $T = 1$: mean $100e^{0.08} = @{a8.T1.mean}$, median $100e^{0.06} = @{a8.T1.median}$; $T = 5$: @{a8.T5.mean} and @{a8.T5.median}', '1. $T = 1$: media $100e^{0.08} = @{a8.T1.mean}$, mediana $100e^{0.06} = @{a8.T1.median}$; $T = 5$: @{a8.T5.mean} și @{a8.T5.median}'),
                 T('2. $\\Phi(-0.06/0.2) = \\Phi(@{a8.T1.z}) = @{a8.T1.pl}\\%$; $\\Phi(-0.3/@{a8.T5.s}) = @{a8.T5.pl}\\%$', '2. $\\Phi(-0.06/0.2) = \\Phi(@{a8.T1.z}) = @{a8.T1.pl}\\%$; $\\Phi(-0.3/@{a8.T5.s}) = @{a8.T5.pl}\\%$'),
                 T('3. $100e^{0.06 - 1.645 \\times 0.2} = @{a8.T1.q}$; $100e^{0.3 - 1.645 \\times @{a8.T5.s}} = @{a8.T5.q}$', '3. $100e^{0.06 - 1.645 \\times 0.2} = @{a8.T1.q}$; $100e^{0.3 - 1.645 \\times @{a8.T5.s}} = @{a8.T5.q}$'),
                 T('4. $2 - 0.5 = 1.5$; $\\min(1, 3) = 1$', '4. $2 - 0.5 = 1.5$; $\\min(1, 3) = 1$')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Simulation, Real Data and Interpretation', 'Partea B: simulare, date reale și interpretare')

D.task(T('B1: Are S\\&P 500 Returns Independent? [Solved]', 'B1: sînt independente randamentele S\\&P 500? [Rezolvat]'),
       T('are the daily returns of the S\\&P 500 independent, and what does yesterday tell us about today?', 'sînt independente randamentele zilnice ale S\\&P 500 și ce ne spune ziua de ieri despre cea de azi?'),
       T('S\\&P 500 closes, 2000--2026, daily log returns in \\%', 'închiderile S\\&P 500, 2000--2026, randamente logaritmice zilnice în \\%'),
       [T('Compute $\\text{Corr}(r_t, r_{t-1})$ and $\\text{Corr}(r_t^2, r_{t-1}^2)$ and compare them with the band $\\pm 1.96/\\sqrt{n}$.', 'Calculați $\\text{Corr}(r_t, r_{t-1})$ și $\\text{Corr}(r_t^2, r_{t-1}^2)$ și comparați-le cu banda $\\pm 1.96/\\sqrt{n}$.'),
        T('Split the days into quintiles of $|r_{t-1}|$ and compute the mean and the standard deviation of $r_t$ in each quintile.', 'Împărțiți zilele în chintile ale lui $|r_{t-1}|$ și calculați media și abaterea standard a lui $r_t$ în fiecare chintilă.'),
        T('Check the law of total variance with the quintiles as groups.', 'Verificați legea varianței totale, cu chintilele ca grupuri.'),
        T('Interpretation: which is predictable from yesterday, the mean or the variance of today\'s return?', 'Interpretare: ce este previzibil pe baza zilei de ieri, media sau varianța randamentului de azi?')],
       T('two correlations, a table of five quintiles, one decomposition, one sentence', 'două corelații, un tabel cu cinci chintile, o descompunere, o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), cols(fig('ch4_sem_b1', h='0.52', w='1.0'), items(
    T('$n = @{b1.n}$, band $\\pm @{b1.band}$; $\\text{Corr}(r_t, r_{t-1}) = @{b1.r}$; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{b1.r2}$', '$n = @{b1.n}$, banda $\\pm @{b1.band}$; $\\text{Corr}(r_t, r_{t-1}) = @{b1.r}$; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{b1.r2}$'),
    T('Conditional means Q1--Q5: $@{b1.m1}, @{b1.m2}, @{b1.m3}, @{b1.m4}, @{b1.m5}$ (\\%)', 'Mediile condiționate Q1--Q5: $@{b1.m1}, @{b1.m2}, @{b1.m3}, @{b1.m4}, @{b1.m5}$ (\\%)'),
    T('Conditional s.d.: $@{b1.s1}, @{b1.s2}, @{b1.s3}, @{b1.s4}, @{b1.s5}$; Q5/Q1 $= @{b1.ratio}$', 'Abaterile standard condiționate: $@{b1.s1}, @{b1.s2}, @{b1.s3}, @{b1.s4}, @{b1.s5}$; Q5/Q1 $= @{b1.ratio}$'),
    T('Total variance $@{b1.total} = @{b1.within}$ (within) $+ @{b1.between}$ (between, @{b1.share}\\%)', 'Varianța totală $@{b1.total} = @{b1.within}$ (în interiorul grupurilor) $+ @{b1.between}$ (între grupuri, @{b1.share}\\%)'),
    T('Interpretation: the variance is predictable, the mean almost not; the negative lag-1 correlation comes from crisis days with large reversals, so returns are not independent',
      'Interpretare: varianța este previzibilă, media aproape deloc; corelația negativă la decalajul 1 vine din zilele de criză cu reveniri mari, deci randamentele nu sînt independente')),
    wl='0.48', wr='0.50') + qlsem(), 'scriptsize')

D.task(T('B2: The Same for DAX, BET and Bitcoin [Proposed]', 'B2: aceeași analiză pentru DAX, BET și Bitcoin [Propus]'),
       T('do the DAX, the BET and Bitcoin show the same pattern as the S\\&P 500?', 'arată DAX, BET și Bitcoin același tipar ca S\\&P 500?'),
       T('daily log returns, 2000--2026 (Bitcoin from 2014); model: B1', 'randamente logaritmice zilnice, 2000--2026 (Bitcoin din 2014); model: B1'),
       [T('Compute both lag-1 correlations and the band for each series.', 'Calculați ambele corelații la decalajul 1 și banda pentru fiecare serie.'),
        T('Compute the conditional standard deviation in Q1 and Q5 and their ratio.', 'Calculați abaterea standard condiționată în Q1 și Q5 și raportul lor.'),
        T('Compute the share of the variance explained by the quintile means.', 'Calculați ponderea varianței explicate de mediile chintilelor.'),
        T('Interpretation: why is the BET the only series with a clearly positive $\\text{Corr}(r_t, r_{t-1})$?', 'Interpretare: de ce este BET singura serie cu $\\text{Corr}(r_t, r_{t-1})$ clar pozitivă?')],
       T('one table and one sentence', 'un tabel și o frază'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrr', T('Series', 'Seria') + ' & $\\rho(r)$ & $\\rho(r^2)$ & ' + T('band', 'banda') + ' & sd Q1 & sd Q5 & Q5/Q1 & ' + T('between \\%', 'între \\%'),
    [f'{lab} & $@{{b2.{k}.r}}$ & $@{{b2.{k}.r2}}$ & $@{{b2.{k}.band}}$ & $@{{b2.{k}.sd1}}$ & $@{{b2.{k}.sd5}}$ & $@{{b2.{k}.ratio}}$ & $@{{b2.{k}.bs}}$'
     for lab, k in [('DAX', 'dax'), ('BET', 'bet'), ('Bitcoin', 'btc')]], size='footnotesize') + items(
    T('All three: squared returns correlated far beyond the band, variance larger after large moves, quintile means explain almost nothing',
      'Toate cele trei serii: corelația pătratelor randamentelor depășește mult banda, varianța este mai mare după mișcări mari, iar mediile chintilelor nu explică aproape nimic'),
    T('BET: the strongest volatility dependence (Q5/Q1 = @{b2.bet.ratio})', 'BET: cea mai puternică dependență a volatilității (Q5/Q1 = @{b2.bet.ratio})'),
    T('Interpretation: positive $\\rho(r)$ for the BET reflects thin trading: prices of illiquid stocks react with a delay, so the index inherits yesterday\'s news',
      'Interpretare: $\\rho(r)$ pozitivă pentru BET reflectă lichiditatea redusă: prețurile acțiunilor nelichide reacționează cu întîrziere, deci indicele preia știrile de ieri')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: A Monthly Loss Probability by Monte Carlo [Solved]', 'B3: probabilitatea unei pierderi lunare prin Monte Carlo [Rezolvat]'),
       T('how likely is a 21-day loss of more than 10\\% for the S\\&P 500, and does the answer depend on the model?', 'cît de probabilă este o pierdere de peste 10\\% în 21 de zile pentru S\\&P 500 și depinde răspunsul de model?'),
       T('S\\&P 500 daily log returns, 2000--2026; 100\\,000 simulated months', 'randamentele logaritmice zilnice ale S\\&P 500, 2000--2026; 100\\,000 de luni simulate'),
       [T('Compute the exact probability under i.i.d. Normal daily returns with the sample mean and standard deviation.', 'Calculați probabilitatea exactă în ipoteza unor randamente zilnice Normale i.i.d., cu media și abaterea standard de selecție.'),
        T('Estimate it by Monte Carlo under the same model and give the standard error.', 'Estimați-o prin Monte Carlo în același model și raportați eroarea standard.'),
        T('Estimate it by resampling 21 observed days with replacement (the empirical inverse transform).', 'Estimați-o prin reeșantionarea cu întoarcere a 21 de zile observate (transformarea inversă empirică).'),
        T('Compute the share of overlapping 21-day windows in the data with a loss beyond 10\\%.', 'Calculați ponderea ferestrelor de 21 de zile suprapuse din date cu o pierdere de peste 10\\%.'),
        T('Interpretation: which differences are simulation error and which are model error?', 'Interpretare: care diferențe se datorează erorii de simulare și care erorii de model?')],
       T('four probabilities, two standard errors, one sentence', 'patru probabilități, două erori standard, o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch4_sem_b3', h='0.40') + items(
    T('Exact Normal: @{b3.exact}\\%; Monte Carlo Normal: @{b3.norm}\\% (s.e. @{b3.se}\\%); bootstrap: @{b3.boot}\\% (s.e. @{b3.seb}\\%)', 'Distribuția Normală, exact: @{b3.exact}\\%; Monte Carlo cu distribuția Normală: @{b3.norm}\\% (eroarea standard @{b3.se}\\%); bootstrap: @{b3.boot}\\% (eroarea standard @{b3.seb}\\%)'),
    T('Data: @{b3.emp}\\% of @{b3.nover} overlapping windows (@{b3.hits} windows, from a few episodes); non-overlapping months: @{b3.nonover}\\% of @{b3.nn}', 'În date: @{b3.emp}\\% din cele @{b3.nover} ferestre suprapuse (@{b3.hits} ferestre, concentrate în cîteva episoade); luni care nu se suprapun: @{b3.nonover}\\% din @{b3.nn}'),
    T('Interpretation: Monte Carlo vs exact differs by less than one s.e. (simulation error); bootstrap vs Normal differs by about @{b3.zb} s.e. (model error: heavier daily tails); the data are noisy because overlapping windows are dependent',
      'Interpretare: Monte Carlo și valoarea exactă diferă cu mai puțin de o eroare standard (eroare de simulare); bootstrap-ul și modelul Normal diferă cu circa @{b3.zb} erori standard (eroare de model: cozi zilnice mai groase); datele sînt zgomotoase pentru că ferestrele suprapuse sînt dependente')) + qlsem(),
    'scriptsize')

D.task(T('B4: Test a Random Number Generator [Proposed]', 'B4: testați un generator de numere aleatoare [Propus]'),
       T('can a generator pass simple tests and still be unusable?', 'poate un generator să treacă testele simple și să fie totuși inutilizabil?'),
       T('RANDU, $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$, seed 1, 100\\,000 numbers; NumPy PCG64 with the same seed; model: B3', 'RANDU, $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$, sămînța 1, 100\\,000 de numere; NumPy PCG64 cu aceeași sămînță; model: B3'),
       [T('Compute the mean, the variance and the lag-1 correlation of both sequences and compare with $1/2$, $1/12$ and 0.', 'Calculați media, varianța și corelația la decalajul 1 pentru ambele șiruri și comparați-le cu $1/2$, $1/12$ și 0.'),
        T('Count the distinct values of $9u_k - 6u_{k+1} + u_{k+2}$ for both generators.', 'Numărați valorile distincte ale lui $9u_k - 6u_{k+1} + u_{k+2}$ pentru ambele generatoare.'),
        T('Find the period of the LCGs $(a, c, M) = (5, 1, 16)$, $(5, 0, 16)$ and $(4, 1, 16)$ started at 7.', 'Găsiți perioada generatoarelor LCG $(a, c, M) = (5, 1, 16)$, $(5, 0, 16)$ și $(4, 1, 16)$ pornite din 7.'),
        T('Interpretation: why do one-dimensional tests not detect the flaw of RANDU?', 'Interpretare: de ce nu detectează testele unidimensionale defectul RANDU?')],
       T('a table of six statistics, two counts, three periods, one sentence', 'un tabel cu șase statistici, două numărători, trei perioade, o frază'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrr', T('Generator', 'Generator') + ' & ' + T('mean', 'media') + ' & ' + T('variance', 'varianța') + ' & $\\rho(1)$ & ' + T('distinct values', 'valori distincte'),
    ['RANDU & $@{b4.randu_mean}$ & $@{b4.randu_var}$ & $@{b4.randu_rho}$ & @{b4.rd}',
     'PCG64 & $@{b4.pcg_mean}$ & $@{b4.pcg_var}$ & $@{b4.pcg_rho}$ & @{b4.pd}'], size='footnotesize') + items(
    T('Both pass the moment and lag-1 tests ($1/12 = 0.0833$); RANDU\'s combination takes only @{b4.rd} values: its triples lie on @{b4.rd} planes', 'Ambele trec testele pentru momente și pentru corelația la decalajul 1 ($1/12 = 0.0833$); combinația RANDU ia doar @{b4.rd} valori: tripletele lui stau pe @{b4.rd} plane'),
    T('Periods: (5, 1, 16): @{b4.period_good}; (5, 0, 16): @{b4.period_mid}; (4, 1, 16): @{b4.period_bad}', 'Perioade: (5, 1, 16): @{b4.period_good}; (5, 0, 16): @{b4.period_mid}; (4, 1, 16): @{b4.period_bad}'),
    T('Interpretation: the flaw is in the joint distribution of consecutive numbers; tests of the marginal distribution or of a single lag cannot see it; any 3-dimensional simulation with RANDU is biased',
      'Interpretare: defectul este în distribuția comună a numerelor consecutive; testele distribuției marginale sau ale unui singur decalaj nu îl pot detecta; orice simulare tridimensională cu RANDU este deformată')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Is the BET a Geometric Brownian Motion? [Solved]', 'B5: este BET o mișcare browniană geometrică? [Rezolvat]'),
       T('which features of the BET can a GBM with the same drift and volatility reproduce?', 'ce trăsături ale BET poate reproduce un GBM cu aceeași tendință și volatilitate?'),
       T('BET closes, 2000--2026; 500 GBM paths of the same length', 'închiderile BET, 2000--2026; 500 de traiectorii GBM de aceeași lungime'),
       [T('Calibrate GBM: annual log drift and volatility from the daily log returns.', 'Calibrați GBM: tendința logaritmică anuală și volatilitatea din randamentele logaritmice zilnice.'),
        T('Simulate 500 paths and compute, for each, the excess kurtosis, $\\text{Corr}(r_t^2, r_{t-1}^2)$ and the maximum drawdown.', 'Simulați 500 de traiectorii și calculați, pentru fiecare, excesul de aplatizare, $\\text{Corr}(r_t^2, r_{t-1}^2)$ și drawdown-ul maxim.'),
        T('Locate the real values in the simulated distributions (5\\% and 95\\% quantiles).', 'Plasați valorile reale în distribuțiile simulate (cuantilele de 5\\% și 95\\%).'),
        T('Interpretation: which stylised facts does GBM miss, and which risk number is most affected?', 'Interpretare: ce fapte stilizate ratează GBM și care indicator de risc este cel mai afectat?')],
       T('two parameters, three comparisons, one sentence', 'doi parametri, trei comparații, o frază'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch4_sem_b5', h='0.36') + items(
    T('$n = @{b5.n}$ days; $\\sigma = @{b5.sigma}\\%$, log drift @{b5.mulog}\\% a year', '$n = @{b5.n}$ zile; $\\sigma = @{b5.sigma}\\%$, tendința logaritmică @{b5.mulog}\\% pe an'),
    T('Excess kurtosis: real $@{b5.k}$, GBM 90\\% range $[@{b5.klo}, @{b5.khi}]$; $\\text{Corr}(r_t^2, r_{t-1}^2)$: real $@{b5.a}$, GBM $[@{b5.alo}, @{b5.ahi}]$',
      'Excesul de aplatizare: real $@{b5.k}$, intervalul GBM de 90\\% $[@{b5.klo}, @{b5.khi}]$; $\\text{Corr}(r_t^2, r_{t-1}^2)$: real $@{b5.a}$, GBM $[@{b5.alo}, @{b5.ahi}]$'),
    T('Maximum drawdown: real $@{b5.dd}\\%$, GBM median $@{b5.ddmed}\\%$, range $[@{b5.ddlo}\\%, @{b5.ddhi}\\%]$', 'Drawdown-ul maxim: real $@{b5.dd}\\%$, mediana GBM $@{b5.ddmed}\\%$, intervalul $[@{b5.ddlo}\\%; @{b5.ddhi}\\%]$'),
    T('Interpretation: GBM misses heavy tails and volatility clustering completely; the 2008 crash makes the real drawdown deeper than every simulated path, so a GBM-based risk limit would have been far too loose',
      'Interpretare: GBM nu reproduce deloc cozile groase și volatility clustering; crahul din 2008 face drawdown-ul real mai adînc decît al oricărei traiectorii simulate, deci o limită de risc calculată cu GBM ar fi fost mult prea largă')) + qlsem(),
    'scriptsize')

D.task(T('B6: Does Variance Grow Linearly with the Horizon? [Proposed]', 'B6: crește varianța liniar cu orizontul? [Propus]'),
       T('does $\\text{Var}(h\\text{-day return}) = h\\,\\text{Var}(1\\text{-day return})$ hold, as a random walk with i.i.d. increments predicts?', 'este valabilă relația $\\text{Var}(\\text{randament pe } h \\text{ zile}) = h\\,\\text{Var}(\\text{randament zilnic})$, așa cum implică un mers aleator cu creșteri i.i.d.?'),
       T('S\\&P 500, BET, Bitcoin daily log returns; non-overlapping blocks of $h = 5, 21, 63$ days; model: B5', 'randamentele logaritmice zilnice ale S\\&P 500, BET, Bitcoin; blocuri disjuncte de $h = 5$, 21 și 63 de zile; model: B5'),
       [T('Compute the ratio $\\text{Var}(h\\text{-day})/(h\\,\\text{Var}(1\\text{-day}))$ for each series and horizon.', 'Calculați raportul $\\text{Var}(h \\text{ zile})/(h\\,\\text{Var}(1 \\text{ zi}))$ pentru fiecare serie și orizont.'),
        T('Use the approximate standard error $\\sqrt{2/(m - 1)}$ of a variance ratio with $m$ blocks to judge the distance from 1.', 'Folosiți eroarea standard aproximativă $\\sqrt{2/(m - 1)}$ a unui raport de varianțe cu $m$ blocuri pentru a evalua cît de departe este raportul de 1.'),
        T('Interpretation: what does a ratio above 1 or below 1 say about the autocorrelation of returns?', 'Interpretare: ce spune un raport peste 1 sau sub 1 despre autocorelația randamentelor?')],
       T('a $3 \\times 3$ table and two sentences', 'un tabel $3 \\times 3$ și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrr', T('Series', 'Seria') + ' & $h = 5$ & $h = 21$ & $h = 63$',
    [f'{lab} & $@{{b6.{k}.5}}$ & $@{{b6.{k}.21}}$ & $@{{b6.{k}.63}}$' for lab, k in [('S\\&P 500', 'sp500'), ('BET', 'bet'), ('Bitcoin', 'btc')]]
    + [T('approx. s.e. (S\\&P 500)', 'eroarea standard aprox. (S\\&P 500)') + ' & $@{b6.sp500.5.se}$ & $@{b6.sp500.21.se}$ & $@{b6.sp500.63.se}$'], size='footnotesize') + items(
    T('S\\&P 500 below 1 (mean reversion over weeks, mostly from crisis rebounds); BET above 1 and growing (positive autocorrelation, thin trading); Bitcoin within about two s.e. of 1',
      'S\\&P 500 sub 1 (revenire la medie pe cîteva săptămîni, mai ales din revenirile de după crize); BET peste 1 și în creștere (autocorelație pozitivă, lichiditate redusă); Bitcoin, la circa două erori standard de 1'),
    T('Interpretation: $\\text{Var}(\\sum r) = h\\sigma^2 + 2\\sum_{i<j}\\text{Cov}(r_i, r_j)$; positive autocorrelations push the ratio above 1, negative ones below 1',
      'Interpretare: $\\text{Var}(\\sum r) = h\\sigma^2 + 2\\sum_{i<j}\\text{Cov}(r_i, r_j)$; autocorelațiile pozitive împing raportul peste 1, cele negative sub 1'),
    T('This is the variance-ratio test of Chapter 7, with its proper standard errors', 'Acesta este testul raportului varianțelor din Capitolul 7, unde se folosesc erorile standard exacte ale testului')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: How Often Is There a Bad Year? [Proposed]', 'C1: cît de des apare un an nefavorabil? [Propus]'),
       T('does GBM give the right probability of a drawdown beyond 20\\% (or 40\\%) within one calendar year?', 'dă GBM probabilitatea corectă a unui drawdown de peste 20\\% (sau 40\\%) în cursul unui an calendaristic?'),
       T('S\\&P 500, DAX, BET, Bitcoin, complete calendar years 2000--2025 (Bitcoin 2015--2025); models: B3, B5', 'S\\&P 500, DAX, BET, Bitcoin, ani calendaristici compleți 2000--2025 (Bitcoin 2015--2025); modele: B3, B5'),
       [T('Compute the maximum drawdown within each calendar year and the share of years beyond 20\\% and beyond 40\\%.', 'Calculați drawdown-ul maxim din fiecare an calendaristic și ponderea anilor cu peste 20\\% și cu peste 40\\%.'),
        T('Simulate 10\\,000 one-year paths under GBM and under the i.i.d. bootstrap of the daily returns and compute the same two probabilities.', 'Simulați 10\\,000 de traiectorii de un an în GBM și în bootstrap-ul i.i.d. al randamentelor zilnice și calculați aceleași două probabilități.'),
        T('Compare data and models with a binomial test.', 'Comparați datele și modelele cu un test binomial.'),
        T('Propose one model that could reproduce both numbers.', 'Propuneți un model care ar putea reproduce ambele probabilități.'),
        T('Interpretation: what does the gap between the bootstrap and the data say about volatility clustering?', 'Interpretare: ce spune diferența dintre bootstrap și date despre volatility clustering?')],
       T('a table, two binomial $p$-values, a short project plan', 'un tabel, două valori $p$ binomiale, un scurt plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrrrr', T('Series', 'Seria') + ' & ' + T('years', 'ani') + ' & ' + T('data 20\\%', 'date 20\\%') + ' & GBM & boot. & ' + T('data 40\\%', 'date 40\\%') + ' & GBM & boot.',
    [f'{lab} & @{{c1.{k}.ny}} & @{{c1.{k}.real20}} & @{{c1.{k}.gbm20}} & @{{c1.{k}.boot20}} & @{{c1.{k}.real40}} & @{{c1.{k}.gbm40}} & @{{c1.{k}.boot40}}'
     for lab, k in [('S\\&P 500', 'sp500'), ('DAX', 'dax'), ('BET', 'bet'), ('Bitcoin', 'btc')]], size='footnotesize') + items(
    T('20\\%: GBM and bootstrap agree with each other; the S\\&P 500 had fewer bad years than both, the BET more', '20\\%: GBM și bootstrap-ul dau rezultate apropiate; S\\&P 500 a avut mai puțini ani nefavorabili decît prevăd ambele modele, BET mai mulți'),
    T('40\\%: both models give well under 3\\% for the indices, but 2008 (and for the DAX 2001--2002) happened: deep drawdowns need clustered volatility, not only heavy tails',
      '40\\%: ambele modele dau probabilități mult sub 3\\% pentru indici, totuși 2008 (și, pentru DAX, 2001--2002) s-a produs: drawdown-urile adînci cer volatility clustering, nu doar cozi groase'),
    T('Project directions: GARCH(1,1) and regime-switching simulations, block bootstrap, more indices, a formal test on 26 years', 'Direcții de proiect: simulări GARCH(1,1) și cu schimbare de regim, block bootstrap, mai mulți indici, un test formal pe 26 de ani')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to ``summarise the probability facts about the S\\&P 500 for a risk report\'\'. The answer:',
      'Un student a cerut unui asistent AI să „rezume faptele de probabilitate despre S\\&P 500 pentru un raport de risc”. Răspunsul:'),
    T('\\aiprompt{(a) Corr(r\\_t, r\\_\\{t-1\\}) = @{c2.r}, so daily returns are independent and tomorrow\'s variance cannot be predicted.}',
      '\\aiprompt{(a) Corr(r\\_t, r\\_\\{t-1\\}) = @{c2.r}, deci randamentele zilnice sînt independente și varianța de mîine nu poate fi prevăzută.}'),
    T('\\aiprompt{(b) The log price is a random walk, so its variance is the same at every date and the process is stationary.}',
      '\\aiprompt{(b) Prețul logaritmic este un mers aleator, deci varianța lui este aceeași la fiecare dată și procesul este staționar.}'),
    T('\\aiprompt{(c) Under GBM with mu = @{c2.mu}\\% and sigma = @{c2.sigma}\\%, the median value of 100 after 10 years is 100 exp(0.8) = @{c2.mean10}.}',
      '\\aiprompt{(c) În GBM cu mu = @{c2.mu}\\% și sigma = @{c2.sigma}\\%, valoarea mediană a unei investiții de 100 după 10 ani este 100 exp(0,8) = @{c2.mean10}.}'),
    T('\\aiprompt{(d) A Monte Carlo probability of about 3\\% from 10,000 paths has a standard error of about @{c2.se}.}',
      '\\aiprompt{(d) O probabilitate Monte Carlo de circa 3\\% din 10.000 de traiectorii are o eroare standard de circa @{c2.se}.}'),
    T('\\aiprompt{(e) A 50/50 portfolio of S\\&P 500 and DAX has daily volatility sqrt(0.25 x @{c2.sdsp}\\textasciicircum 2 + 0.25 x @{c2.sddax}\\textasciicircum 2) = @{c2.sdw}\\%.}',
      '\\aiprompt{(e) Un portofoliu 50/50 din S\\&P 500 și DAX are volatilitatea zilnică sqrt(0,25 x @{c2.sdsp}\\textasciicircum 2 + 0,25 x @{c2.sddax}\\textasciicircum 2) = @{c2.sdw}\\%.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație stabiliți dacă este corectă; dacă nu este, formulați afirmația corectă și dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: uncorrelated is not independent; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{c2.r2}$ and the conditional s.d. after the most turbulent days is @{c2.ratio} times that after the calmest',
      '(a) Greșit: necorelat nu înseamnă independent; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{c2.r2}$, iar abaterea standard condiționată după zilele cele mai agitate este de @{c2.ratio} ori mai mare decît cea de după zilele cele mai calme'),
    T('(b) Wrong: $\\text{Var}(S_t) = t\\sigma^2$ grows with $t$; a random walk is not stationary (its increments are)', '(b) Greșit: $\\text{Var}(S_t) = t\\sigma^2$ crește cu $t$; un mers aleator nu este staționar (creșterile lui sînt)'),
    T('(c) Wrong: $100e^{\\mu T} = @{c2.mean10}$ is the mean; the median is $100e^{(\\mu - \\sigma^2/2)T} = @{c2.median10}$', '(c) Greșit: $100e^{\\mu T} = @{c2.mean10}$ este media; mediana este $100e^{(\\mu - \\sigma^2/2)T} = @{c2.median10}$'),
    T('(d) Correct: $\\sqrt{0.03 \\times 0.97/10\\,000} = @{c2.se}$; the error falls like $1/\\sqrt{N}$, not $1/N$', '(d) Corect: $\\sqrt{0.03 \\times 0.97/10\\,000} = @{c2.se}$; eroarea scade ca $1/\\sqrt{N}$, nu ca $1/N$'),
    T('(e) Wrong: the covariance term is missing; with $\\rho = @{c2.corr}$ the volatility is $@{c2.sdp}\\%$, not $@{c2.sdw}\\%$: the AI understates the risk', '(e) Greșit: lipsește termenul de covarianță; cu $\\rho = @{c2.corr}$ volatilitatea este $@{c2.sdp}\\%$, nu $@{c2.sdw}\\%$: răspunsul AI subestimează riscul')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Zero correlation does not mean independence: check squares or absolute values too', 'Corelația zero nu înseamnă independență: verificați și pătratele sau valorile absolute'),
    T('The conditional variance of returns moves with yesterday\'s news; the conditional mean hardly moves', 'Varianța condiționată a randamentelor reacționează la știrile de ieri; media condiționată aproape deloc'),
    T('Mixtures of Normal regimes give heavy tails', 'Amestecurile de regimuri Normale dau cozi groase'),
    T('Monte Carlo: report the standard error; separate simulation error from model error', 'Monte Carlo: raportați eroarea standard; separați eroarea de simulare de eroarea de model'),
    T('GBM reproduces the path of an index roughly, but not its tails, clusters or deepest drawdowns', 'GBM reproduce aproximativ traiectoria unui indice, dar nu și cozile, volatility clustering sau cele mai adînci drawdown-uri'),
    T('An AI answer is a draft: check every definition, model and number', 'Un răspuns AI este o ciornă: verificați fiecare definiție, fiecare model și fiecare valoare numerică')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 4 develops each topic of today: distributions, dependence, conditioning, Monte Carlo, binomial trees, random walks, Wiener process and GBM',
      'Cursul 4 dezvoltă fiecare temă de azi: distribuții, dependență, condiționare, Monte Carlo, arbori binomiali, mers aleator, procesul Wiener și GBM'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: GARCH and regime-switching simulations, block bootstrap, more indices', 'C1 poate deveni un proiect de echipă: simulări GARCH și cu schimbare de regim, block bootstrap, mai mulți indici'),
    T('Reading: \\refFHH, Ch.~3--5; exercises with solutions: \\refFHHex', 'Lectură: \\refFHH, cap.~3--5; exerciții rezolvate: \\refFHHex')))

D.references([
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Cox, J.C., Ross, S.A., Rubinstein, M. (1979). \href{https://doi.org/10.1016/0304-405X(79)90015-1}{Option pricing: a simplified approach}. \textit{Journal of Financial Economics}, 7(3), 229--263.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Glasserman, P. (2003). \href{https://doi.org/10.1007/978-0-387-21617-1}{\textit{Monte Carlo Methods in Financial Engineering}}. Springer.",
    r"Marsaglia, G. (1968). \href{https://doi.org/10.1073/pnas.61.1.25}{Random numbers fall mainly in the planes}. \textit{Proceedings of the National Academy of Sciences}, 61(1), 25--28.",
])

if __name__ == '__main__':
    D.write(V)
