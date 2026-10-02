r"""
build_seminar7.py -- Seminarul 7 (Ipoteza piețelor eficiente, mersul aleator și testele VR), EN + RO dintr-o singură
sursă
==================================================================================================================
Seminarul are loc ÎNAINTEA cursului 7: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_07/sem7_results.json (seminar7.py).
Ieșire:
  EN/Seminars/seminar7_efficient_markets_random_walk_vr.tex          (+ _solutions.tex)
  RO/Seminarii/seminar7_piete_eficiente_mers_aleator_vr_ro.tex       (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_07/seminar7.py && python3 latex/build_seminar7.py && python3 latex/sfm_build.py compile 7
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch7_common import NAMES, REFS, SHORT, T, bib, kp, load_sem, pv   # noqa: E402

S = load_sem()
V = Values()
D = Deck(7, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_07}{SFM_ch7_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    d = A[a]
    V.put(f'{a}.vr2', d['vr'], 2)
    V.put(f'{a}.se2', d['se'], 3)
    V.put(f'{a}.z2', d['z'], 2)
    V.put(f'{a}.vr5', d['vr5']['vr'], 3)
    V.put(f'{a}.se5', d['vr5']['se'], 4)
    V.put(f'{a}.z5', d['vr5']['z'], 2)
    for c in ['sd5', 'sd20', 'sd5_vr', 'sd20_vr']:
        V.put(f'{a}.{c}', d[c], 2)
for a in ['a3', 'a4']:
    d = A[a]
    for i, (b, l) in enumerate(zip(d['terms_bp'], d['terms_lb']), 1):
        V.put(f'{a}.b{i}', b, 2)
        V.put(f'{a}.l{i}', l, 2)
    V.put(f'{a}.bp', d['bp'], 2)
    V.put(f'{a}.lb', d['lb'], 2)
    V.put(f'{a}.crit', d['crit'], 2)
    V.raw(f'{a}.p', pv(d['p_lb']))
for a in ['a5', 'a6']:
    d = A[a]
    V.raw(f'{a}.n1', str(d['n1']))
    V.raw(f'{a}.n2', str(d['n2']))
    V.raw(f'{a}.r', str(d['runs']))
    V.put(f'{a}.mean', d['mean'], 2)
    V.put(f'{a}.sd', d['sd'], 2)
    V.put(f'{a}.var', d['sd'] ** 2, 2)
    V.put(f'{a}.z', d['z'], 2)
    V.raw(f'{a}.p', pv(d['p']))
A7 = A['a7']
V.put('a7.taup', A7['price']['tau'], 2)
V.put('a7.taur', A7['ret']['tau'], 1)
V.put('a7.crit', A7['price']['crit'], 2)
V.put('a8.tau', A['a8']['tau'], 2)
V.put('a8.crit', A['a8']['crit'], 2)

B1 = S['B1']
V.int('b1.n', B1['n'])
V.raw('b1.y0', B1['first'][:4])
for i, r in enumerate(B1['rho'], 1):
    V.put(f'b1.r{i}', r, 3)
for i, r in enumerate(B1['rho_sq'], 1):
    V.put(f'b1.s{i}', r, 3)
for lab in ['ret', 'sq']:
    d = B1[lab]
    V.put(f'b1.{lab}.lb', d['lb'], 1 if d['lb'] < 1000 else 0)
    V.raw(f'b1.{lab}.plb', pv(d['p_lb']))
    V.put(f'b1.{lab}.rob', d['rob'], 1)
    V.raw(f'b1.{lab}.prob', pv(d['p_rob']))
V.put('b1.crit', B1['ret']['crit'], 2)
B2 = S['B2']
for k, d in B2.items():
    V.int(f'b2.{k}.n', d['n'])
    V.raw(f'b2.{k}.y0', d['first'][:4])
    V.put(f'b2.{k}.rho1', d['port']['rho1'], 3)
    V.put(f'b2.{k}.lb', d['port']['lb'], 1)
    V.raw(f'b2.{k}.plb', pv(d['port']['p_lb']))
    V.put(f'b2.{k}.rob', d['port']['rob'], 1)
    V.raw(f'b2.{k}.prob', pv(d['port']['p_rob']))
    V.put(f'b2.{k}.rz', d['runs']['z'], 2)
    V.raw(f'b2.{k}.rp', pv(d['runs']['p']))


def put_ur(prefix, d):
    for s_ in ['price', 'ret']:
        for t in ['adf', 'pp']:
            V.put(f'{prefix}.{s_}.{t}', d[s_][t]['stat'], 2)
            V.raw(f'{prefix}.{s_}.{t}.p', pv(d[s_][t]['p']))
        V.put(f'{prefix}.{s_}.kpss', d[s_]['kpss']['stat'], 2)
        V.raw(f'{prefix}.{s_}.kpss.p', kp(d[s_]['kpss']['p']))
    if 'lags' in d['price']['adf']:
        V.raw(f'{prefix}.lags', str(d['price']['adf']['lags']))


B3 = S['B3']
put_ur('b3', B3)
V.raw('b3.lagsr', str(B3['ret']['adf']['lags']))
V.raw('b3.pplags', str(B3['price']['pp']['lags']))
V.int('b3.n', B3['n'])
for k, d in S['B4'].items():
    put_ur(f'b4.{k}', d)
B5 = S['B5']
V.int('b5.n', B5['n'])
for q, d in B5['vr'].items():
    V.put(f'b5.vr{q}', d['vr'], 3)
    V.put(f'b5.z{q}', d['z'], 2)
    V.put(f'b5.zs{q}', d['zs'], 2)
    V.put(f'b5.ser{q}', d['se_rob'], 3)
    V.put(f'b5.sei{q}', d['se_iid'], 3)
V.put('b5.cd', B5['cd']['cd'], 2)
V.put('b5.crit', B5['cd']['crit'], 2)
V.raw('b5.cdp', pv(B5['cd']['p']))
V.put('b5.ratio', B5['vr']['2']['z'] / B5['vr']['2']['zs'], 1)
V.put('b5.pct', 100 * B5['vr']['5']['vr'], 0)
B6 = S['B6']
EM = ['wig20', 'bux', 'px', 'bet', 'bist']
for k in EM:
    d = B6[k]
    V.int(f'b6.{k}.n', d['n'])
    V.put(f'b6.{k}.vr', d['vr'], 3)
    V.put(f'b6.{k}.zs', d['zs'], 2)
    V.raw(f'b6.{k}.pzs', pv(d['p_zs']))
    V.raw(f'b6.{k}.cdp', pv(d['cd_p']))
    V.put(f'b6.{k}.avr', d['avr']['stat'], 2)
    V.raw(f'b6.{k}.avrp', pv(d['avr']['p_boot']))
V.put('b6.sidak', B6['sidak']['crit'], 2)
V.put('b6.alpha', 100 * B6['sidak']['alpha_each'], 2)
V.put('b6.any', 100 * B6['sidak']['p_any_false'], 1)
C1 = S['C1']
V.put('c1.first', 100 * C1['share_rej_first5y'], 0)
V.put('c1.last', 100 * C1['share_rej_last5y'], 0)
c1rows = []
for r in C1['sub']:
    c1rows.append(f'{r["from"][:4]}--{r["to"][:4]} & {r["n"]} & $⁅{r["rho1"]:.3f}⁆$ & $⁅{r["vr"]:.2f}⁆$ & $⁅{r["zs"]:.2f}⁆$ & ' + pv(r['cd_p']))
U = C1['upgrade']
for p in ['before', 'after']:
    V.put(f'c1.{p}.vr', U[p]['vr'], 2)
    V.put(f'c1.{p}.zs', U[p]['zs'], 2)
    V.put(f'c1.{p}.rho1', U[p]['rho1'], 3)
V.put('c1.z', U['z_diff'], 2)
C2 = S['C2']
V.put('c2.adf', C2['adf_price']['stat'], 2)
V.put('c2.adfp', C2['adf_price']['p'], 2)
V.put('c2.lb', C2['port']['lb'], 2)
V.put('c2.plb', C2['port']['p_lb'], 3)
V.put('c2.rob', C2['port']['rob'], 2)
V.put('c2.prob', C2['port']['p_rob'], 3)
V.put('c2.lbsq', C2['port_sq']['lb'], 0)
V.put('c2.robsq', C2['port_sq']['rob'], 1)
V.put('c2.vr10', C2['vr10'], 2)
V.put('c2.zs10', C2['zs10'], 2)
V.put('c2.z10', C2['z10'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how can we test whether past prices predict future returns without fooling ourselves?',
       '\\textbf{Întrebarea}: cum testăm dacă prețurile trecute anticipează randamentele viitoare, fără să ne păcălim singuri?'),
     [T('this seminar comes \\textbf{before} Lecture 7: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 7: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: variance ratios, Ljung--Box statistics, runs and Dickey--Fuller decisions, on paper',
        'Partea A: rapoarte ale varianțelor, statistici Ljung--Box, testul runs și decizii Dickey--Fuller, pe hîrtie'),
      T('Part B: autocorrelation, unit-root and variance-ratio tests on the S\\&P 500, DAX, BET, Bitcoin, BVB stocks and emerging markets, each with an interpretation question',
        'Partea B: teste de autocorelație, de rădăcină unitară și variance ratio pe S\\&P 500, DAX, BET, Bitcoin, acțiuni BVB și piețe emergente, fiecare cu o întrebare de interpretare'),
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
    ['A1, A2 & ' + T('the square-root-of-time rule and variance ratios from autocorrelations', 'regula rădăcinii pătrate a timpului și rapoarte ale varianțelor din autocorelații') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('Box--Pierce and Ljung--Box statistics, step by step', 'statisticile Box--Pierce și Ljung--Box, pas cu pas') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('the runs test on a sequence of signs', 'testul runs pe un șir de semne') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('ADF and KPSS decisions from regression output', 'decizii ADF și KPSS din rezultatele unei regresii') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('ACF, Ljung--Box and robust tests, runs test: S\\&P 500; BET, Bitcoin, BVB stocks', 'ACF, Ljung--Box și teste robuste, testul runs: S\\&P 500; BET, Bitcoin, acțiuni BVB') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('ADF, Phillips--Perron and KPSS on prices and returns: DAX; BET, Bitcoin, TLV', 'ADF, Phillips--Perron și KPSS pe prețuri și randamente: DAX; BET, Bitcoin, TLV') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('variance-ratio tests: S\\&P 500; five emerging markets', 'testele variance ratio: S\\&P 500; cinci piețe emergente') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('has the BET become more efficient? what is wrong in an AI answer?', 'a devenit BET mai eficient? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX & EODHD & close & 1990--2026',
     'BET & EODHD & close & 1997--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & 2014--2026',
     T('Banca Transilvania (TLV), OMV Petrom (SNP)', 'Banca Transilvania (TLV), OMV Petrom (SNP)') + ' & EODHD & adjusted close & 2010--2026',
     'WIG20, BUX, PX, BIST 100 & EODHD & close & ' + T('last ten years', 'ultimii zece ani')],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin); the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); ultima zi este 18 septembrie 2026'),
    T('TLV: the days 30 and 31 May 2016 are removed (a price adjustment applied one day late, Chapter 2)',
      'TLV: zilele de 30 și 31 mai 2016 se elimină (o ajustare de preț aplicată cu o zi mai tîrziu, Capitolul 2)'),
    T('In the notebook: \\texttt{returns(\'sp500\')}, \\texttt{log\\_price(\'dax\')}; no account or key is needed',
      'În notebook: \\texttt{returns(\'sp500\')}, \\texttt{log\\_price(\'dax\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Efficiency and Random Walks', 'Noțiuni necesare azi (1/5): eficiență și mers aleator'), items(
    (T('\\textbf{Weak-form efficiency} \\refFamaA: past prices cannot be used to earn returns above the return that pays for risk',
       '\\textbf{Eficiența în formă slabă} \\refFamaA: prețurile trecute nu pot fi folosite pentru a obține randamente peste randamentul care plătește riscul'),
     [T('excess returns are a \\textbf{martingale difference}: $E[r_{t+1} - \\mu \\mid \\text{past}] = 0$', 'randamentele în exces sînt o \\textbf{diferență de martingal}: $E[r_{t+1} - \\mu \\mid \\text{trecut}] = 0$')]),
    (T('Log price $p_t = \\ln P_t$, \\textbf{random walk}: $p_t = \\mu + p_{t-1} + \\varepsilon_t$, so $r_t = \\mu + \\varepsilon_t$ \\refCLM',
       'Prețul logaritmic $p_t = \\ln P_t$, \\textbf{mers aleator}: $p_t = \\mu + p_{t-1} + \\varepsilon_t$, deci $r_t = \\mu + \\varepsilon_t$ \\refCLM'),
     [T('\\textbf{RW1}: $\\varepsilon_t$ i.i.d.\\ (independent, identically distributed); \\textbf{RW2}: independent; \\textbf{RW3}: uncorrelated',
        '\\textbf{RW1}: $\\varepsilon_t$ i.i.d.\\ (independente și identic distribuite); \\textbf{RW2}: independente; \\textbf{RW3}: necorelate'),
      T('RW1 $\\Rightarrow$ RW2 $\\Rightarrow$ RW3; volatility clustering is allowed only by RW3 and by the martingale', 'RW1 $\\Rightarrow$ RW2 $\\Rightarrow$ RW3; gruparea volatilității este permisă doar de RW3 și de martingal')]),
    (T('Under any of them, $\\mathrm{Var}(r_t(q)) = q\\sigma^2$ for the $q$-day return $r_t(q) = r_t + \\dots + r_{t-q+1}$',
       'În oricare dintre ele, $\\mathrm{Var}(r_t(q)) = q\\sigma^2$ pentru randamentul pe $q$ zile $r_t(q) = r_t + \\dots + r_{t-q+1}$'),
     [T('``square-root-of-time rule\'\': $\\mathrm{sd}(r_t(q)) = \\sigma\\sqrt{q}$', '„regula rădăcinii pătrate a timpului”: $\\mathrm{sd}(r_t(q)) = \\sigma\\sqrt{q}$')])))

D.frame(T('What You Need for Today (2/5): ACF and Portmanteau Tests', 'Noțiuni necesare azi (2/5): ACF și teste portmanteau'), items(
    (T('Sample \\textbf{ACF} (autocorrelation function), with $e_t = r_t - \\bar r$: $\\hat\\rho(k) = \\sum_{t>k} e_te_{t-k}\\big/\\sum_t e_t^2$',
       '\\textbf{ACF} (autocorrelation function, funcția de autocorelație) de selecție, cu $e_t = r_t - \\bar r$: $\\hat\\rho(k) = \\sum_{t>k} e_te_{t-k}\\big/\\sum_t e_t^2$'),
     [T('i.i.d.\\ returns: $\\mathrm{Var}(\\hat\\rho(k)) \\approx 1/T$, 95\\% band $\\pm 1.96/\\sqrt{T}$', 'randamente i.i.d.: $\\mathrm{Var}(\\hat\\rho(k)) \\approx 1/T$, banda de 95\\% $\\pm 1{,}96/\\sqrt{T}$'),
      T('robust (volatility clustering): $\\mathrm{Var}(\\hat\\rho(k)) \\approx \\hat\\delta(k)/T$, $\\hat\\delta(k) = T\\sum_t e_t^2e_{t-k}^2/(\\sum_t e_t^2)^2$ \\refLM',
        'robust (gruparea volatilității): $\\mathrm{Var}(\\hat\\rho(k)) \\approx \\hat\\delta(k)/T$, $\\hat\\delta(k) = T\\sum_t e_t^2e_{t-k}^2/(\\sum_t e_t^2)^2$ \\refLM')]),
    (T('$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; three statistics, each $\\approx \\chi^2(m)$ under its null', '$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; trei statistici, fiecare $\\approx \\chi^2(m)$ în ipoteza ei nulă'),
     [T('Box--Pierce $Q_{BP} = T\\sum_{k=1}^m \\hat\\rho(k)^2$ \\refBP; \\textbf{LB} (Ljung--Box) $Q_{LB} = T(T+2)\\sum_{k=1}^m \\hat\\rho(k)^2/(T-k)$ \\refLjung',
        'Box--Pierce $Q_{BP} = T\\sum_{k=1}^m \\hat\\rho(k)^2$ \\refBP; \\textbf{LB} (Ljung--Box, testul Ljung--Box) $Q_{LB} = T(T+2)\\sum_{k=1}^m \\hat\\rho(k)^2/(T-k)$ \\refLjung'),
      T('robust $\\tilde Q = \\sum_{k=1}^m T\\hat\\rho(k)^2/\\hat\\delta(k)$ \\refEL: valid for a martingale with volatility clustering', '$\\tilde Q$ robust $= \\sum_{k=1}^m T\\hat\\rho(k)^2/\\hat\\delta(k)$ \\refEL: valabil pentru un martingal cu gruparea volatilității')]),
    T('5\\% critical values: $\\chi^2_{0.95}(3) = 7.81$, $\\chi^2_{0.95}(5) = 11.07$, $\\chi^2_{0.95}(10) = 18.31$', 'Valori critice de 5\\%: $\\chi^2_{0.95}(3) = 7{,}81$, $\\chi^2_{0.95}(5) = 11{,}07$, $\\chi^2_{0.95}(10) = 18{,}31$')))

D.frame(T('What You Need for Today (3/5): the Runs Test', 'Noțiuni necesare azi (3/5): testul runs'), items(
    (T('A \\textbf{run}: a maximal sequence of returns with the same sign \\refWW; zero returns are dropped', 'Un \\textbf{run} (secvență): un șir maximal de randamente cu același semn \\refWW; randamentele nule se elimină'),
     [T('$+ + - - - + -$: 4 runs; $n_1 = 3$ positive, $n_2 = 4$ negative, $n = n_1 + n_2 = 7$', '$+ + - - - + -$: 4 secvențe; $n_1 = 3$ pozitive, $n_2 = 4$ negative, $n = n_1 + n_2 = 7$')]),
    (T('Under independence of the signs:', 'Dacă semnele sînt independente:'),
     [T('$E[R] = \\dfrac{2n_1n_2}{n} + 1$, \\quad $\\mathrm{Var}(R) = \\dfrac{2n_1n_2(2n_1n_2 - n)}{n^2(n-1)}$', '$E[R] = \\dfrac{2n_1n_2}{n} + 1$, \\quad $\\mathrm{Var}(R) = \\dfrac{2n_1n_2(2n_1n_2 - n)}{n^2(n-1)}$'),
      T('$z = (R - E[R])/\\sqrt{\\mathrm{Var}(R)} \\approx N(0, 1)$; reject at 5\\% if $|z| > 1.96$', '$z = (R - E[R])/\\sqrt{\\mathrm{Var}(R)} \\approx N(0, 1)$; respingem la 5\\% dacă $|z| > 1{,}96$')]),
    (T('Reading $z$', 'Interpretarea lui $z$'),
     [T('$z < 0$: too few runs, signs persist (positive dependence, trends)', '$z < 0$: prea puține secvențe, semnele persistă (dependență pozitivă, tendințe)'),
      T('$z > 0$: too many runs, signs alternate (negative dependence, reversals)', '$z > 0$: prea multe secvențe, semnele alternează (dependență negativă, reveniri)')]),
    T('Signs only: the test ignores the size of the returns, so heavy tails and volatility do not affect it', 'Doar semnele: testul ignoră mărimea randamentelor, deci cozile groase și volatilitatea nu îl afectează')))

D.frame(T('What You Need for Today (4/5): Unit Roots', 'Noțiuni necesare azi (4/5): rădăcini unitare'), items(
    T('$I(0)$: stationary (constant mean and variance, shocks fade); $I(1)$: the first difference is $I(0)$; a random walk is $I(1)$',
       '$I(0)$: staționar (medie și varianță constante, șocurile se sting); $I(1)$: prima diferență este $I(0)$; un mers aleator este $I(1)$'),
    (T('Dickey--Fuller regression \\refDF: $\\Delta p_t = c\\ [+\\,bt] + \\gamma\\, p_{t-1}\\ [+ \\sum_j \\delta_j\\Delta p_{t-j}] + \\varepsilon_t$',
       'Regresia Dickey--Fuller \\refDF: $\\Delta p_t = c\\ [+\\,bt] + \\gamma\\, p_{t-1}\\ [+ \\sum_j \\delta_j\\Delta p_{t-j}] + \\varepsilon_t$'),
     [T('$H_0$: $\\gamma = 0$ (unit root); $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; reject if $\\tau <$ the critical value',
        '$H_0$: $\\gamma = 0$ (rădăcină unitară); $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; respingem dacă $\\tau <$ valoarea critică'),
      T('5\\% critical values \\refMacKinnon: $-2.86$ (constant), $-3.41$ (constant and trend); not $-1.645$',
        'valori critice de 5\\% \\refMacKinnon: $-2{,}86$ (constantă), $-3{,}41$ (constantă și tendință); nu $-1{,}645$'),
      T('\\textbf{ADF} (augmented Dickey--Fuller): with the lagged differences \\refSD; \\textbf{PP} (Phillips--Perron): without them, corrected $\\tau$ \\refPP',
        '\\textbf{ADF} (augmented Dickey--Fuller): cu diferențele decalate \\refSD; \\textbf{PP} (Phillips--Perron): fără ele, cu $\\tau$ corectat \\refPP')]),
    (T('\\textbf{KPSS} \\refKPSS: $H_0$ stationarity; reject if the statistic exceeds $0.463$ (constant) or $0.146$ (trend)',
       '\\textbf{KPSS} \\refKPSS: $H_0$ staționaritate; respingem dacă statistica depășește $0{,}463$ (constantă) sau $0{,}146$ (tendință)'),
     [T('ADF does not reject + KPSS rejects: $I(1)$; ADF rejects + KPSS does not: $I(0)$; otherwise: inconclusive',
        'ADF nu respinge + KPSS respinge: $I(1)$; ADF respinge + KPSS nu: $I(0)$; altfel: neconcludent')])))

D.frame(T('What You Need for Today (5/5): Variance Ratios', 'Noțiuni necesare azi (5/5): rapoarte ale varianțelor'), items(
    (T('\\textbf{VR} (variance ratio): $\\mathrm{VR}(q) = \\dfrac{\\mathrm{Var}(r_t(q))}{q\\,\\mathrm{Var}(r_t)} = 1 + 2\\sum_{k=1}^{q-1}\\Big(1 - \\dfrac{k}{q}\\Big)\\rho(k)$ \\refLM',
       '\\textbf{VR} (variance ratio, raportul varianțelor): $\\mathrm{VR}(q) = \\dfrac{\\mathrm{Var}(r_t(q))}{q\\,\\mathrm{Var}(r_t)} = 1 + 2\\sum_{k=1}^{q-1}\\Big(1 - \\dfrac{k}{q}\\Big)\\rho(k)$ \\refLM'),
     [T('$= 1$: random walk; $> 1$: momentum (positive autocorrelation); $< 1$: mean reversion (negative)', '$= 1$: mers aleator; $> 1$: momentum (autocorelație pozitivă); $< 1$: mean reversion (revenire la medie, negativă)')]),
    (T('Tests with $T$ returns', 'Teste cu $T$ randamente'),
     [T('i.i.d.: $Z(q) = (\\widehat{\\mathrm{VR}}(q) - 1)\\big/\\sqrt{2(2q-1)(q-1)/(3qT)}$', 'i.i.d.: $Z(q) = (\\widehat{\\mathrm{VR}}(q) - 1)\\big/\\sqrt{2(2q-1)(q-1)/(3qT)}$'),
      T('robust: $Z^*(q) = (\\widehat{\\mathrm{VR}}(q) - 1)\\big/\\sqrt{\\hat\\theta(q)/T}$, $\\hat\\theta(q) = \\sum_{j=1}^{q-1}[2(q-j)/q]^2\\hat\\delta(j)$',
        'robust: $Z^*(q) = (\\widehat{\\mathrm{VR}}(q) - 1)\\big/\\sqrt{\\hat\\theta(q)/T}$, $\\hat\\theta(q) = \\sum_{j=1}^{q-1}[2(q-j)/q]^2\\hat\\delta(j)$')]),
    (T('Several horizons: \\textbf{CD} (Chow--Denning) $= \\max_i|Z^*(q_i)|$ \\refCD; 5\\% critical value $2.49$ for $m = 4$ horizons',
       'Mai multe orizonturi: \\textbf{CD} (Chow--Denning) $= \\max_i|Z^*(q_i)|$ \\refCD; valoarea critică de 5\\% $2{,}49$ pentru $m = 4$ orizonturi'),
     [T('the same rule for $m$ markets: each test at level $1 - 0.95^{1/m}$ (Šidák)', 'aceeași regulă pentru $m$ piețe: fiecare test la nivelul $1 - 0{,}95^{1/m}$ (Šidák)'),
      T('automatic VR \\refChoi: the horizon is chosen from the data; p-value by wild bootstrap \\refKimB', 'VR automat \\refChoi: orizontul este ales din date; p-valoarea prin wild bootstrap \\refKimB')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Variance Ratios from Autocorrelations', 'A1: rapoarte ale varianțelor din autocorelații'),
         items(T('Daily returns have a standard deviation $\\sigma = 1.2\\%$.', 'Randamentele zilnice au abaterea standard $\\sigma = 1{,}2\\%$.'),
               T('1. Under RW1, compute the standard deviation of the 5-day and of the 20-day return.', '1. Pentru RW1, calculați abaterea standard a randamentului pe 5 zile și pe 20 de zile.'),
               T('2. Suppose $\\rho(1) = 0.10$, $\\rho(2) = 0.05$ and $\\rho(k) = 0$ for $k \\ge 3$; compute VR(2) and VR(5).', '2. Presupunem $\\rho(1) = 0{,}10$, $\\rho(2) = 0{,}05$ și $\\rho(k) = 0$ pentru $k \\ge 3$; calculați VR(2) și VR(5).'),
               T('3. Treat these as estimates from $T = 2500$ i.i.d.\\ returns and compute $Z(2)$ and $Z(5)$.', '3. Tratați-le ca estimări din $T = 2500$ de randamente i.i.d.\\ și calculați $Z(2)$ și $Z(5)$.'),
               T('4. Compute the 5-day standard deviation implied by VR(5).', '4. Calculați abaterea standard pe 5 zile implicată de VR(5).'),
               T('Report: two standard deviations, two VR, two $Z$ and one sentence.', 'Raportați: două abateri standard, două VR, două $Z$ și o frază.')),
         items(T('1. $1.2\\sqrt5 = @{a1.sd5}\\%$; $1.2\\sqrt{20} = @{a1.sd20}\\%$', '1. $1.2\\sqrt5 = @{a1.sd5}\\%$; $1.2\\sqrt{20} = @{a1.sd20}\\%$'),
               T('2. VR(2) $= 1 + 0.10 = @{a1.vr2}$; VR(5) $= 1 + 2(0.8 \\times 0.10 + 0.6 \\times 0.05) = @{a1.vr5}$', '2. VR(2) $= 1 + 0.10 = @{a1.vr2}$; VR(5) $= 1 + 2(0.8 \\times 0.10 + 0.6 \\times 0.05) = @{a1.vr5}$'),
               T('3. $\\sqrt{2 \\cdot 3 \\cdot 1/(6 \\cdot 2500)} = @{a1.se2}$, $Z(2) = @{a1.z2}$; $\\sqrt{2 \\cdot 9 \\cdot 4/(15 \\cdot 2500)} = @{a1.se5}$, $Z(5) = @{a1.z5}$',
                 '3. $\\sqrt{2 \\cdot 3 \\cdot 1/(6 \\cdot 2500)} = @{a1.se2}$, $Z(2) = @{a1.z2}$; $\\sqrt{2 \\cdot 9 \\cdot 4/(15 \\cdot 2500)} = @{a1.se5}$, $Z(5) = @{a1.z5}$'),
               T('4. $1.2\\sqrt{5 \\times @{a1.vr5}} = @{a1.sd5_vr}\\%$ instead of $@{a1.sd5}\\%$: momentum makes weekly risk larger than the $\\sqrt{q}$ rule says.',
                 '4. $1.2\\sqrt{5 \\times @{a1.vr5}} = @{a1.sd5_vr}\\%$ în loc de $@{a1.sd5}\\%$: momentumul face riscul săptămînal mai mare decît spune regula $\\sqrt{q}$.')),
         size='scriptsize')

D.proposed(T('A2: Mean Reversion in a Volatile Asset', 'A2: revenire la medie la un activ volatil'),
           items(T('Daily returns: $\\sigma = 2\\%$, $\\rho(1) = -0.08$, $\\rho(2) = 0.03$, $\\rho(k) = 0$ for $k \\ge 3$, $T = 1000$. Model: A1.',
                   'Randamente zilnice: $\\sigma = 2\\%$, $\\rho(1) = -0{,}08$, $\\rho(2) = 0{,}03$, $\\rho(k) = 0$ pentru $k \\ge 3$, $T = 1000$. Model: A1.'),
                 T('1. Compute VR(2) and VR(5).', '1. Calculați VR(2) și VR(5).'),
                 T('2. Compute $Z(2)$ and $Z(5)$ and decide at 5\\%.', '2. Calculați $Z(2)$ și $Z(5)$ și decideți la 5\\%.'),
                 T('3. Compare the 5-day standard deviation implied by VR(5) with $2\\sqrt5$.', '3. Comparați abaterea standard pe 5 zile implicată de VR(5) cu $2\\sqrt5$.'),
                 T('Report: two VR, two $Z$, two standard deviations and one sentence.', 'Raportați: două VR, două $Z$, două abateri standard și o frază.')),
           items(T('1. VR(2) $= @{a2.vr2}$; VR(5) $= 1 + 2(0.8 \\times (-0.08) + 0.6 \\times 0.03) = @{a2.vr5}$', '1. VR(2) $= @{a2.vr2}$; VR(5) $= 1 + 2(0.8 \\times (-0.08) + 0.6 \\times 0.03) = @{a2.vr5}$'),
                 T('2. $Z(2) = @{a2.z2}$: reject at 5\\%; $Z(5) = @{a2.z5}$: do not reject', '2. $Z(2) = @{a2.z2}$: respingem la 5\\%; $Z(5) = @{a2.z5}$: nu respingem'),
                 T('3. $2\\sqrt{5 \\times @{a2.vr5}} = @{a2.sd5_vr}\\%$ against $@{a2.sd5}\\%$: reversals make weekly risk smaller; one horizon rejects, the other does not.',
                   '3. $2\\sqrt{5 \\times @{a2.vr5}} = @{a2.sd5_vr}\\%$ față de $@{a2.sd5}\\%$: revenirile fac riscul săptămînal mai mic; un orizont respinge, celălalt nu.')),
           size='scriptsize')

D.solved(T('A3: Box--Pierce and Ljung--Box, Step by Step', 'A3: Box--Pierce și Ljung--Box, pas cu pas'),
         items(T('$T = 500$ daily returns; $\\hat\\rho(1), \\dots, \\hat\\rho(5) = 0.12, -0.05, 0.04, 0.03, -0.06$.', '$T = 500$ de randamente zilnice; $\\hat\\rho(1), \\dots, \\hat\\rho(5) = 0{,}12;\\ -0{,}05;\\ 0{,}04;\\ 0{,}03;\\ -0{,}06$.'),
               T('1. Compute $Q_{BP}(5)$.', '1. Calculați $Q_{BP}(5)$.'),
               T('2. Compute $Q_{LB}(5)$.', '2. Calculați $Q_{LB}(5)$.'),
               T('3. Compare with $\\chi^2_{0.95}(5) = 11.07$ and decide.', '3. Comparați cu $\\chi^2_{0.95}(5) = 11{,}07$ și decideți.'),
               T('4. Say which lag drives the result.', '4. Spuneți ce decalaj determină rezultatul.'),
               T('Report: two statistics, a decision and one sentence.', 'Raportați: două statistici, o decizie și o frază.')),
         items(T('1. $T\\hat\\rho(k)^2$: $@{a3.b1}$, $@{a3.b2}$, $@{a3.b3}$, $@{a3.b4}$, $@{a3.b5}$; $Q_{BP}(5) = @{a3.bp}$', '1. $T\\hat\\rho(k)^2$: $@{a3.b1}$; $@{a3.b2}$; $@{a3.b3}$; $@{a3.b4}$; $@{a3.b5}$; $Q_{BP}(5) = @{a3.bp}$'),
               T('2. $500 \\cdot 502\\,\\hat\\rho(k)^2/(500 - k)$: $@{a3.l1}$, $@{a3.l2}$, $@{a3.l3}$, $@{a3.l4}$, $@{a3.l5}$; $Q_{LB}(5) = @{a3.lb}$',
                 '2. $500 \\cdot 502\\,\\hat\\rho(k)^2/(500 - k)$: $@{a3.l1}$; $@{a3.l2}$; $@{a3.l3}$; $@{a3.l4}$; $@{a3.l5}$; $Q_{LB}(5) = @{a3.lb}$'),
               T('3. $@{a3.lb} > @{a3.crit}$: reject $H_0$ at 5\\% (p = @{a3.p})', '3. $@{a3.lb} > @{a3.crit}$: respingem $H_0$ la 5\\% (p = @{a3.p})'),
               T('4. Lag 1 alone gives $@{a3.b1}$ of $@{a3.bp}$: one autocorrelation of $0.12$ drives the rejection.', '4. Decalajul 1 dă singur $@{a3.b1}$ din $@{a3.bp}$: o singură autocorelație de $0{,}12$ determină respingerea.')),
         size='scriptsize')

D.proposed(T('A4: Ljung--Box on Squared Returns', 'A4: Ljung--Box pe randamentele la pătrat'),
           items(T('$T = 750$; the ACF of the \\textbf{squared} returns: $0.25, 0.20, 0.18, 0.15, 0.12$ at lags 1--5. Model: A3.',
                   '$T = 750$; ACF-ul randamentelor \\textbf{la pătrat}: $0{,}25;\\ 0{,}20;\\ 0{,}18;\\ 0{,}15;\\ 0{,}12$ la decalajele 1--5. Model: A3.'),
                 T('1. Compute $Q_{LB}(5)$ and decide at 5\\%.', '1. Calculați $Q_{LB}(5)$ și decideți la 5\\%.'),
                 T('2. Say which of RW1, RW3 and the martingale this rejects.', '2. Spuneți pe care dintre RW1, RW3 și martingal îl respinge.'),
                 T('3. Say whether weak-form efficiency is rejected.', '3. Spuneți dacă eficiența în formă slabă este respinsă.'),
                 T('Report: one statistic and two sentences.', 'Raportați: o statistică și două fraze.')),
           items(T('1. Terms $@{a4.l1}$, $@{a4.l2}$, $@{a4.l3}$, $@{a4.l4}$, $@{a4.l5}$: $Q_{LB}(5) = @{a4.lb} \\gg @{a4.crit}$: reject', '1. Termenii $@{a4.l1}$; $@{a4.l2}$; $@{a4.l3}$; $@{a4.l4}$; $@{a4.l5}$: $Q_{LB}(5) = @{a4.lb} \\gg @{a4.crit}$: respingem'),
                 T('2. Squared returns are correlated: volatility clustering; RW1 (and RW2) is rejected; RW3 and the martingale are not tested', '2. Randamentele la pătrat sînt corelate: gruparea volatilității; RW1 (și RW2) este respins; RW3 și martingalul nu sînt testate'),
                 T('3. No: a predictable variance does not give a predictable return.', '3. Nu: o varianță previzibilă nu dă un randament previzibil.')),
           size='scriptsize')

D.solved(T('A5: The Runs Test on 20 Days', 'A5: testul runs pe 20 de zile'),
         items(T('Signs of 20 daily returns: $+ + - + + + - - + - - - + + - + - - + +$.', 'Semnele a 20 de randamente zilnice: $+ + - + + + - - + - - - + + - + - - + +$.'),
               T('1. Count $n_1$, $n_2$ and the number of runs $R$.', '1. Numărați $n_1$, $n_2$ și numărul de secvențe $R$.'),
               T('2. Compute $E[R]$ and $\\mathrm{Var}(R)$.', '2. Calculați $E[R]$ și $\\mathrm{Var}(R)$.'),
               T('3. Compute $z$ and decide at 5\\%.', '3. Calculați $z$ și decideți la 5\\%.'),
               T('Report: $R$, $E[R]$, $z$ and one sentence.', 'Raportați: $R$, $E[R]$, $z$ și o frază.')),
         items(T('1. $n_1 = @{a5.n1}$, $n_2 = @{a5.n2}$, $R = @{a5.r}$', '1. $n_1 = @{a5.n1}$, $n_2 = @{a5.n2}$, $R = @{a5.r}$'),
               T('2. $E[R] = 2 \\cdot 11 \\cdot 9/20 + 1 = @{a5.mean}$; $\\mathrm{Var}(R) = 198 \\cdot 178/(400 \\cdot 19) = @{a5.var}$', '2. $E[R] = 2 \\cdot 11 \\cdot 9/20 + 1 = @{a5.mean}$; $\\mathrm{Var}(R) = 198 \\cdot 178/(400 \\cdot 19) = @{a5.var}$'),
               T('3. $z = (11 - @{a5.mean})/@{a5.sd} = @{a5.z}$ (p = @{a5.p}): do not reject independence.', '3. $z = (11 - @{a5.mean})/@{a5.sd} = @{a5.z}$ (p = @{a5.p}): nu respingem independența.'),
               T('With 20 days the test has little power: only large departures can be detected.', 'Cu 20 de zile testul are putere mică: doar abaterile mari pot fi detectate.')),
         size='scriptsize')

D.proposed(T('A6: A Sequence with Long Runs', 'A6: un șir cu secvențe lungi'),
           items(T('Signs of 24 daily returns: $+ + + - - - + + + - - - + + + + - - - - + + - -$. Model: A5.', 'Semnele a 24 de randamente zilnice: $+ + + - - - + + + - - - + + + + - - - - + + - -$. Model: A5.'),
                 T('1. Count $n_1$, $n_2$ and $R$.', '1. Numărați $n_1$, $n_2$ și $R$.'),
                 T('2. Compute $E[R]$, $\\mathrm{Var}(R)$ and $z$.', '2. Calculați $E[R]$, $\\mathrm{Var}(R)$ și $z$.'),
                 T('3. Decide at 5\\% and say what kind of dependence the signs show.', '3. Decideți la 5\\% și spuneți ce fel de dependență arată semnele.'),
                 T('Report: $R$, $E[R]$, $z$ and one sentence.', 'Raportați: $R$, $E[R]$, $z$ și o frază.')),
           items(T('1. $n_1 = @{a6.n1}$, $n_2 = @{a6.n2}$, $R = @{a6.r}$', '1. $n_1 = @{a6.n1}$, $n_2 = @{a6.n2}$, $R = @{a6.r}$'),
                 T('2. $E[R] = @{a6.mean}$, $\\mathrm{Var}(R) = @{a6.var}$, $z = @{a6.z}$', '2. $E[R] = @{a6.mean}$, $\\mathrm{Var}(R) = @{a6.var}$, $z = @{a6.z}$'),
                 T('3. p = @{a6.p}: reject at 5\\%; too few runs: signs persist, a sign of trends (positive dependence).', '3. p = @{a6.p}: respingem la 5\\%; prea puține secvențe: semnele persistă, semn de tendințe (dependență pozitivă).')),
           size='scriptsize')

D.solved(T('A7: ADF and KPSS from Regression Output', 'A7: ADF și KPSS din rezultatele regresiei'),
         items(T('Log price, $T = 2500$: $\\Delta p_t = 0.021 - 0.0028\\,p_{t-1} + 0.05\\,\\Delta p_{t-1} + e_t$, $\\mathrm{SE}(\\hat\\gamma) = 0.0019$; KPSS (constant) $= 2.10$.',
                 'Preț logaritmic, $T = 2500$: $\\Delta p_t = 0{,}021 - 0{,}0028\\,p_{t-1} + 0{,}05\\,\\Delta p_{t-1} + e_t$, $\\mathrm{SE}(\\hat\\gamma) = 0{,}0019$; KPSS (constantă) $= 2{,}10$.'),
               T('Returns: $\\Delta r_t = 0.03 - 0.92\\,r_{t-1} + e_t$, $\\mathrm{SE}(\\hat\\gamma) = 0.020$; KPSS (constant) $= 0.09$.', 'Randamente: $\\Delta r_t = 0{,}03 - 0{,}92\\,r_{t-1} + e_t$, $\\mathrm{SE}(\\hat\\gamma) = 0{,}020$; KPSS (constantă) $= 0{,}09$.'),
               T('1. Compute $\\tau$ for the price and for the returns.', '1. Calculați $\\tau$ pentru preț și pentru randamente.'),
               T('2. Decide with the 5\\% critical value for a constant.', '2. Decideți cu valoarea critică de 5\\% pentru constantă.'),
               T('3. Decide with KPSS and give the order of integration of each series.', '3. Decideți cu KPSS și dați ordinul de integrare al fiecărei serii.'),
               T('Report: two $\\tau$, four decisions and one sentence.', 'Raportați: doi $\\tau$, patru decizii și o frază.')),
         items(T('1. Price: $\\tau = -0.0028/0.0019 = @{a7.taup}$; returns: $\\tau = -0.92/0.020 = @{a7.taur}$', '1. Preț: $\\tau = -0.0028/0.0019 = @{a7.taup}$; randamente: $\\tau = -0.92/0.020 = @{a7.taur}$'),
               T('2. Price: $@{a7.taup} > @{a7.crit}$: do not reject a unit root; returns: $@{a7.taur} < @{a7.crit}$: reject', '2. Preț: $@{a7.taup} > @{a7.crit}$: nu respingem rădăcina unitară; randamente: $@{a7.taur} < @{a7.crit}$: respingem'),
               T('3. KPSS: price $2.10 > 0.463$: reject stationarity; returns $0.09 < 0.463$: do not reject', '3. KPSS: preț $2.10 > 0.463$: respingem staționaritatea; randamente $0.09 < 0.463$: nu respingem'),
               T('The price is $I(1)$, the returns $I(0)$: the picture of a random walk, but not yet a proof of it.', 'Prețul este $I(1)$, randamentele $I(0)$: imaginea unui mers aleator, dar încă nu o dovadă a lui.')),
         size='scriptsize')

D.proposed(T('A8: When the Tests Disagree', 'A8: cînd testele nu sînt de acord'),
           items(T('Log price of a stock, $T = 1500$, constant and trend: $\\hat\\gamma = -0.0105$, $\\mathrm{SE}(\\hat\\gamma) = 0.0029$; KPSS (trend) $= 0.21$. Model: A7.',
                   'Prețul logaritmic al unei acțiuni, $T = 1500$, constantă și tendință: $\\hat\\gamma = -0{,}0105$, $\\mathrm{SE}(\\hat\\gamma) = 0{,}0029$; KPSS (tendință) $= 0{,}21$. Model: A7.'),
                 T('1. Compute $\\tau$ and decide with the 5\\% critical value for a constant and a trend.', '1. Calculați $\\tau$ și decideți cu valoarea critică de 5\\% pentru constantă și tendință.'),
                 T('2. Decide with KPSS.', '2. Decideți cu KPSS.'),
                 T('3. Combine the two decisions and give two possible reasons for the conflict.', '3. Combinați cele două decizii și dați două motive posibile pentru conflict.'),
                 T('Report: $\\tau$, two decisions and two sentences.', 'Raportați: $\\tau$, două decizii și două fraze.')),
           items(T('1. $\\tau = -0.0105/0.0029 = @{a8.tau} < @{a8.crit}$: reject the unit root at 5\\%', '1. $\\tau = -0.0105/0.0029 = @{a8.tau} < @{a8.crit}$: respingem rădăcina unitară la 5\\%'),
                 T('2. KPSS $0.21 > 0.146$: reject trend-stationarity', '2. KPSS $0.21 > 0.146$: respingem staționaritatea în jurul tendinței'),
                 T('3. Both reject: inconclusive. Reasons: a structural break (a one-off level shift), $\\phi$ close to but below 1, heteroskedasticity, or a 5\\% false rejection by one test.',
                   '3. Ambele resping: neconcludent. Motive: o ruptură structurală (o schimbare bruscă de nivel), un $\\phi$ apropiat de 1 dar sub 1, heteroscedasticitate sau o respingere falsă de 5\\% a unuia dintre teste.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Autocorrelation in the S\\&P 500 [Solved]', 'B1: autocorelația S\\&P 500 [Rezolvat]'),
       T('does the verdict on the autocorrelation of daily S\\&P 500 returns depend on whether we allow for volatility clustering?',
         'depinde concluzia privind autocorelația randamentelor zilnice S\\&P 500 de faptul că ținem cont sau nu de gruparea volatilității?'),
       T('S\\&P 500 closes since @{b1.y0}, daily log returns in \\%', 'închiderile S\\&P 500 din @{b1.y0}, randamente logaritmice zilnice în \\%'),
       [T('Draw the ACF of the returns (lags 1--20) with the i.i.d.\\ and the robust 95\\% bands, and the ACF of the squared returns.',
          'Desenați ACF-ul randamentelor (decalajele 1--20) cu benzile de 95\\% i.i.d.\\ și robustă, și ACF-ul randamentelor la pătrat.'),
        T('Report $\\hat\\rho(1), \\dots, \\hat\\rho(5)$ of the returns and of the squared returns.', 'Raportați $\\hat\\rho(1), \\dots, \\hat\\rho(5)$ pentru randamente și pentru randamentele la pătrat.'),
        T('Compute $Q_{LB}(10)$ and the robust $\\tilde Q(10)$, with p-values, for the returns and for the squared returns.', 'Calculați $Q_{LB}(10)$ și $\\tilde Q(10)$ robust, cu p-valori, pentru randamente și pentru randamentele la pătrat.'),
        T('Say which random walk hypothesis each test rejects.', 'Spuneți ce ipoteză de mers aleator respinge fiecare test.'),
        T('Interpretation: is weak-form efficiency rejected?', 'Interpretare: este respinsă eficiența în formă slabă?')],
       T('the chart, ten autocorrelations, four statistics with p-values and two sentences', 'graficul, zece autocorelații, patru statistici cu p-valori și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch7_sem_b1', h='0.36') + items(
    T('$\\hat\\rho(1..5)$, returns: $@{b1.r1}$, $@{b1.r2}$, $@{b1.r3}$, $@{b1.r4}$, $@{b1.r5}$; squared: $@{b1.s1}$, $@{b1.s2}$, $@{b1.s3}$, $@{b1.s4}$, $@{b1.s5}$; $T = @{b1.n}$',
      '$\\hat\\rho(1..5)$, randamente: $@{b1.r1}$; $@{b1.r2}$; $@{b1.r3}$; $@{b1.r4}$; $@{b1.r5}$; la pătrat: $@{b1.s1}$; $@{b1.s2}$; $@{b1.s3}$; $@{b1.s4}$; $@{b1.s5}$; $T = @{b1.n}$'),
    T('Returns: $Q_{LB}(10) = @{b1.ret.lb}$ (p @{b1.ret.plb}), $\\tilde Q(10) = @{b1.ret.rob}$ (p = @{b1.ret.prob}); squared: $Q_{LB}(10) = @{b1.sq.lb}$, $\\tilde Q(10) = @{b1.sq.rob}$ (both p @{b1.sq.prob})',
      'Randamente: $Q_{LB}(10) = @{b1.ret.lb}$ (p @{b1.ret.plb}), $\\tilde Q(10) = @{b1.ret.rob}$ (p = @{b1.ret.prob}); la pătrat: $Q_{LB}(10) = @{b1.sq.lb}$, $\\tilde Q(10) = @{b1.sq.rob}$ (ambele p @{b1.sq.prob})'),
    T('Interpretation: squared returns reject RW1 (volatility clustering); the robust test rejects RW3 only barely, because of a small negative $\\hat\\rho(1)$: a weak, short-lived reversal',
      'Interpretare: randamentele la pătrat resping RW1 (gruparea volatilității); testul robust respinge RW3 doar la limită, din cauza unui $\\hat\\rho(1)$ negativ mic: o revenire slabă, de scurtă durată')) + qlsem(), 'scriptsize')

D.task(T('B2: BET, Bitcoin and Two BVB Stocks [Proposed]', 'B2: BET, Bitcoin și două acțiuni BVB [Propus]'),
       T('for which of the BET, Bitcoin, Banca Transilvania and OMV Petrom does the verdict on autocorrelation change between the classic and the robust test? Model: B1.',
         'pentru care dintre BET, Bitcoin, Banca Transilvania și OMV Petrom se schimbă verdictul privind autocorelația între testul clasic și cel robust? Model: B1.'),
       T('BET since 1997, Bitcoin since 2014, TLV and SNP since 2010 (TLV without 30--31 May 2016), daily log returns in \\%',
         'BET din 1997, Bitcoin din 2014, TLV și SNP din 2010 (TLV fără 30--31 mai 2016), randamente logaritmice zilnice în \\%'),
       [T('Compute $\\hat\\rho(1)$, $Q_{LB}(10)$ and $\\tilde Q(10)$ with p-values for each series.', 'Calculați $\\hat\\rho(1)$, $Q_{LB}(10)$ și $\\tilde Q(10)$ cu p-valori pentru fiecare serie.'),
        T('Compute the runs test $z$ and its p-value for each series.', 'Calculați statistica $z$ a testului runs și p-valoarea ei pentru fiecare serie.'),
        T('Mark the series for which the classic and the robust test disagree at 5\\%.', 'Marcați seriile pentru care testul clasic și cel robust nu sînt de acord la 5\\%.'),
        T('Interpretation: why is the BET different from the other three?', 'Interpretare: de ce este BET diferit de celelalte trei?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{SHORT[k]} & @{{b2.{k}.n}} & ${{@{{b2.{k}.rho1}}}}$ & $@{{b2.{k}.lb}}$ & @{{b2.{k}.plb}} & $@{{b2.{k}.rob}}$ & @{{b2.{k}.prob}} & '
            f'${{@{{b2.{k}.rz}}}}$ & @{{b2.{k}.rp}}')


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrrr', '& $T$ & $\\hat\\rho(1)$ & $Q_{LB}(10)$ & p & $\\tilde Q(10)$ & p & runs $z$ & p',
    [b2row(k) for k in ['bet', 'btc', 'tlv', 'snp']], size='footnotesize') + items(
    T('Classic LB rejects for all four; the robust test only for the BET: the verdict changes for Bitcoin, TLV and SNP',
      'LB clasic respinge pentru toate patru; testul robust doar pentru BET: verdictul se schimbă pentru Bitcoin, TLV și SNP'),
    T('Runs: the BET has far too few runs ($z = @{b2.bet.rz}$); Bitcoin has too many ($z = @{b2.btc.rz}$): signs alternate slightly more than under independence',
      'Runs: BET are mult prea puține secvențe ($z = @{b2.bet.rz}$); Bitcoin are prea multe ($z = @{b2.btc.rz}$): semnele alternează puțin mai des decît în cazul independenței'),
    T('Interpretation: the BET is an index of stocks with uneven trading: stale closing prices create positive autocorrelation (thin trading); the single stocks and Bitcoin do not show it',
      'Interpretare: BET este un indice de acțiuni tranzacționate inegal: prețurile de închidere vechi creează autocorelație pozitivă (thin trading); acțiunile individuale și Bitcoin nu o arată')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Unit Roots in the DAX [Solved]', 'B3: rădăcini unitare în DAX [Rezolvat]'),
       T('what is the order of integration of the DAX log price and of its returns?', 'care este ordinul de integrare al prețului logaritmic DAX și al randamentelor lui?'),
       T('DAX closes since 1990; log price and daily log returns', 'închiderile DAX din 1990; prețul logaritmic și randamentele logaritmice zilnice'),
       [T('Draw the log price with a fitted linear trend, and the returns.', 'Desenați prețul logaritmic cu o tendință liniară ajustată, precum și randamentele.'),
        T('Run ADF (lag order by AIC), PP and KPSS on the log price with a constant and a trend.', 'Aplicați ADF (ordinul decalajelor ales prin AIC), PP și KPSS pe prețul logaritmic, cu constantă și tendință.'),
        T('Run the same three tests on the returns with a constant.', 'Aplicați aceleași trei teste pe randamente, cu constantă.'),
        T('Interpretation: does a unit root in the DAX price prove that the DAX is weak-form efficient?', 'Interpretare: dovedește o rădăcină unitară în prețul DAX că DAX este eficient în formă slabă?')],
       T('the chart, six statistics with p-values and two sentences', 'graficul, șase statistici cu p-valori și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch7_sem_b3', h='0.36') + table(
    'lrrr', T('& ADF & PP & KPSS', '& ADF & PP & KPSS'),
    [T('log price (trend)', 'preț logaritmic (tendință)') + ' & $@{b3.price.adf}$ (p = @{b3.price.adf.p}) & $@{b3.price.pp}$ (p = @{b3.price.pp.p}) & $@{b3.price.kpss}$ (p @{b3.price.kpss.p})',
     T('returns (constant)', 'randamente (constantă)') + ' & $@{b3.ret.adf}$ (p @{b3.ret.adf.p}) & $@{b3.ret.pp}$ (p @{b3.ret.pp.p}) & $@{b3.ret.kpss}$ (p @{b3.ret.kpss.p})'],
    size='scriptsize') + items(
    T('ADF used @{b3.lags} lags for the price (AIC); PP and KPSS use @{b3.pplags} Bartlett lags; $T = @{b3.n}$', 'ADF a folosit @{b3.lags} decalaje pentru preț (AIC); PP și KPSS folosesc @{b3.pplags} decalaje Bartlett; $T = @{b3.n}$'),
    T('Price: no rejection by ADF or PP, rejection by KPSS: $I(1)$; returns: the opposite: $I(0)$', 'Prețul: nicio respingere ADF sau PP, respingere KPSS: $I(1)$; randamentele: invers: $I(0)$'),
    T('Interpretation: no; a unit root is necessary for a random walk but allows predictable increments; efficiency is tested on the returns (B1, B5)',
      'Interpretare: nu; rădăcina unitară este necesară pentru un mers aleator, dar permite creșteri previzibile; eficiența se testează pe randamente (B1, B5)')) + qlsem(), 'scriptsize')

D.task(T('B4: BET, Bitcoin and TLV [Proposed]', 'B4: BET, Bitcoin și TLV [Propus]'),
       T('do the three unit-root tests agree for the BET, Bitcoin and Banca Transilvania? Model: B3.', 'sînt de acord cele trei teste de rădăcină unitară pentru BET, Bitcoin și Banca Transilvania? Model: B3.'),
       T('BET since 1997, Bitcoin since 2014, TLV since 2010 (without 30--31 May 2016)', 'BET din 1997, Bitcoin din 2014, TLV din 2010 (fără 30--31 mai 2016)'),
       [T('Run ADF, PP and KPSS on each log price (constant and trend) and on each return series (constant).', 'Aplicați ADF, PP și KPSS pe fiecare preț logaritmic (constantă și tendință) și pe fiecare serie de randamente (constantă).'),
        T('Classify each series as $I(1)$, $I(0)$ or inconclusive.', 'Clasificați fiecare serie ca $I(1)$, $I(0)$ sau neconcludentă.'),
        T('For an inconclusive case, propose one check that could settle it.', 'Pentru un caz neconcludent, propuneți o verificare care l-ar putea lămuri.'),
        T('Interpretation: would you trade a ``mean-reverting\'\' TLV price on this evidence?', 'Interpretare: ați tranzacționa pe baza acestei dovezi un preț TLV „care revine la medie”?')],
       T('one table, three classifications and two sentences', 'un tabel, trei clasificări și două fraze'), size='footnotesize', nb='B4')


def b4row(k):
    return (f'{SHORT[k]} & ${{@{{b4.{k}.price.adf}}}}$ (@{{b4.{k}.price.adf.p}}) & ${{@{{b4.{k}.price.pp}}}}$ (@{{b4.{k}.price.pp.p}}) & '
            f'$@{{b4.{k}.price.kpss}}$ (@{{b4.{k}.price.kpss.p}}) & ${{@{{b4.{k}.ret.adf}}}}$ & ${{@{{b4.{k}.ret.pp}}}}$ & $@{{b4.{k}.ret.kpss}}$ (@{{b4.{k}.ret.kpss.p}})')


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrr', T('& ADF price (p) & PP price (p) & KPSS price (p) & ADF ret. & PP ret. & KPSS ret. (p)',
                 '& ADF preț (p) & PP preț (p) & KPSS preț (p) & ADF rand. & PP rand. & KPSS rand. (p)'),
    [b4row(k) for k in ['bet', 'btc', 'tlv']], size='scriptsize') + items(
    T('BET and Bitcoin: log price $I(1)$, returns $I(0)$', 'BET și Bitcoin: preț logaritmic $I(1)$, randamente $I(0)$'),
    T('TLV price: ADF rejects at 5\\% (p = @{b4.tlv.price.adf.p}), PP does not (p = @{b4.tlv.price.pp.p}), KPSS rejects stationarity: inconclusive; check: test for a structural break, or split the sample',
      'Prețul TLV: ADF respinge la 5\\% (p = @{b4.tlv.price.adf.p}), PP nu (p = @{b4.tlv.price.pp.p}), KPSS respinge staționaritatea: neconcludent; verificare: testați o ruptură structurală sau împărțiți eșantionul'),
    T('Interpretation: no; a borderline p-value, contradicted by two other tests, on one of many series tested, is not a trading signal',
      'Interpretare: nu; o p-valoare la limită, contrazisă de alte două teste, pentru una dintre multe serii testate, nu este un semnal de tranzacționare')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Variance Ratios of the S\\&P 500 [Solved]', 'B5: rapoartele varianțelor pentru S\\&P 500 [Rezolvat]'),
       T('is the S\\&P 500 mean-reverting over horizons of 2 to 20 days, once volatility clustering is taken into account?',
         'revine S\\&P 500 la medie pe orizonturi de 2--20 de zile, dacă ținem cont de gruparea volatilității?'),
       T('S\\&P 500 closes since 1990, daily log returns in \\%', 'închiderile S\\&P 500 din 1990, randamente logaritmice zilnice în \\%'),
       [T('Compute VR($q$), $Z(q)$ and $Z^*(q)$ for $q = 2, 5, 10, 20$.', 'Calculați VR($q$), $Z(q)$ și $Z^*(q)$ pentru $q = 2, 5, 10, 20$.'),
        T('Compute the Chow--Denning statistic and compare it with its 5\\% critical value.', 'Calculați statistica Chow--Denning și comparați-o cu valoarea ei critică de 5\\%.'),
        T('Draw VR($q$) for $q = 2, \\dots, 40$ with the i.i.d.\\ and the robust 95\\% bands.', 'Desenați VR($q$) pentru $q = 2, \\dots, 40$ cu benzile de 95\\% i.i.d.\\ și robustă.'),
        T('Interpretation: is the mean reversion large enough to matter for an investor?', 'Interpretare: este revenirea la medie suficient de mare pentru a conta pentru un investitor?')],
       T('a table of twelve numbers, one statistic, the chart and two sentences', 'un tabel cu douăsprezece cifre, o statistică, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch7_sem_b5', h='0.32') + table(
    'lrrrr', '& $q = 2$ & $q = 5$ & $q = 10$ & $q = 20$',
    ['VR($q$) & ' + ' & '.join(f'$@{{b5.vr{q}}}$' for q in ['2', '5', '10', '20']),
     '$Z(q)$ & ' + ' & '.join(f'${{@{{b5.z{q}}}}}$' for q in ['2', '5', '10', '20']),
     '$Z^*(q)$ & ' + ' & '.join(f'${{@{{b5.zs{q}}}}}$' for q in ['2', '5', '10', '20'])], size='scriptsize') + items(
    T('$T = @{b5.n}$; Chow--Denning: $\\max|Z^*| = @{b5.cd} > @{b5.crit}$ (p = @{b5.cdp}): reject the random walk', '$T = @{b5.n}$; Chow--Denning: $\\max|Z^*| = @{b5.cd} > @{b5.crit}$ (p = @{b5.cdp}): respingem mersul aleator'),
    T('$Z(2)$ is about @{b5.ratio} times $Z^*(2)$: ignoring volatility clustering exaggerates the evidence', '$Z(2)$ este de circa @{b5.ratio} ori $Z^*(2)$: ignorarea grupării volatilității exagerează dovezile'),
    T('Interpretation: significant, but small: the weekly variance is about @{b5.pct}\\% of five daily variances; after trading costs this is hardly a profit',
      'Interpretare: semnificativă, dar mică: varianța săptămînală este circa @{b5.pct}\\% din cinci varianțe zilnice; după costurile de tranzacționare, cu greu un profit')) + qlsem(), 'scriptsize')

D.task(T('B6: Five Emerging Markets [Proposed]', 'B6: cinci piețe emergente [Propus]'),
       T('which of the WIG20, BUX, PX, BET and BIST 100 departs from a random walk over the last ten years? Model: B5.',
         'care dintre WIG20, BUX, PX, BET și BIST 100 se abate de la un mers aleator în ultimii zece ani? Model: B5.'),
       T('daily log returns from 19 September 2016 to 18 September 2026', 'randamente logaritmice zilnice de la 19 septembrie 2016 pînă la 18 septembrie 2026'),
       [T('Compute VR(5), $Z^*(5)$ and its p-value for each market.', 'Calculați VR(5), $Z^*(5)$ și p-valoarea lui pentru fiecare piață.'),
        T('Compute the Chow--Denning p-value ($q = 2, 5, 10, 20$) and the automatic VR test with its wild-bootstrap p-value.', 'Calculați p-valoarea Chow--Denning ($q = 2, 5, 10, 20$) și testul VR automat cu p-valoarea wild bootstrap.'),
        T('Compute the Šidák critical value for five markets tested together at 5\\%.', 'Calculați valoarea critică Šidák pentru cinci piețe testate împreună la 5\\%.'),
        T('Interpretation: which markets would you call inefficient once you account for testing five markets?', 'Interpretare: ce piețe ați numi ineficiente, după ce țineți cont de testarea a cinci piețe?')],
       T('one table, one critical value and two sentences', 'un tabel, o valoare critică și două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{NAMES[k]} & @{{b6.{k}.n}} & $@{{b6.{k}.vr}}$ & ${{@{{b6.{k}.zs}}}}$ & @{{b6.{k}.pzs}} & @{{b6.{k}.cdp}} & ${{@{{b6.{k}.avr}}}}$ & @{{b6.{k}.avrp}}')


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& $T$ & VR(5) & $Z^*(5)$ & p & CD p & automatic VR & p (bootstrap)', '& $T$ & VR(5) & $Z^*(5)$ & p & CD p & VR automat & p (bootstrap)'),
    [b6row(k) for k in EM], size='footnotesize') + items(
    T('Šidák: each test at $1 - 0.95^{1/5} = @{b6.alpha}\\%$, critical value $@{b6.sidak}$; with five independent tests at 5\\%, the chance of at least one false rejection is @{b6.any}\\%',
      'Šidák: fiecare test la $1 - 0{,}95^{1/5} = @{b6.alpha}\\%$, valoarea critică $@{b6.sidak}$; cu cinci teste independente la 5\\%, șansa a cel puțin unei respingeri false este @{b6.any}\\%'),
    T('PX and BET: $Z^*(5)$ just above 1.96, below $@{b6.sidak}$; Chow--Denning does not reject; only the automatic VR rejects for the BET',
      'PX și BET: $Z^*(5)$ puțin peste 1,96, sub $@{b6.sidak}$; Chow--Denning nu respinge; doar VR automat respinge pentru BET'),
    T('Interpretation: no market is clearly inefficient; the BET shows weak, test-dependent momentum', 'Interpretare: nicio piață nu este clar ineficientă; BET arată un momentum slab, care depinde de test')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Has the BET Become More Efficient? [Proposed]', 'C1: a devenit BET mai eficient? [Propus]'),
       T('how has the predictability of the BET evolved since 1997, including around the FTSE Russell upgrade of September 2020?',
         'cum a evoluat previzibilitatea BET din 1997 pînă azi, inclusiv în jurul trecerii la Secondary Emerging de către FTSE Russell în septembrie 2020?'),
       T('BET daily log returns since 1997; models: B1, B5', 'randamentele logaritmice zilnice BET din 1997; modele: B1, B5'),
       [T('Compute VR(5) and $Z^*(5)$ on rolling windows of 500 days, one every 21 days, and the share of windows with $|Z^*(5)| > 1.96$ in the first and in the last five years.',
          'Calculați VR(5) și $Z^*(5)$ pe ferestre mobile de 500 de zile, una la fiecare 21 de zile, și ponderea ferestrelor cu $|Z^*(5)| > 1{,}96$ în primii și în ultimii cinci ani.'),
        T('Compute $\\hat\\rho(1)$, VR(5), $Z^*(5)$ and the Chow--Denning p-value in 1997--2006, 2007--2016 and 2017--2026.', 'Calculați $\\hat\\rho(1)$, VR(5), $Z^*(5)$ și p-valoarea Chow--Denning în 1997--2006, 2007--2016 și 2017--2026.'),
        T('Compare VR(5) in the five years before and after 21 September 2020.', 'Comparați VR(5) în cei cinci ani dinainte și de după 21 septembrie 2020.'),
        T('List what else changed in the Romanian market at the same time.', 'Enumerați ce altceva s-a schimbat pe piața românească în același timp.'),
        T('Interpretation: how would you separate these changes from the effect of the upgrade?', 'Interpretare: cum ați separa aceste schimbări de efectul trecerii la noul statut?')],
       T('a table, one chart and a plan for a project', 'un tabel, un grafic și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrr', T('Period & $T$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ & CD p', 'Perioada & $T$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ & CD p'), c1rows, size='footnotesize') + items(
    T('Rolling windows rejecting at 5\\%: @{c1.first}\\% in the first five years, @{c1.last}\\% in the last five years', 'Ferestre mobile care resping la 5\\%: @{c1.first}\\% în primii cinci ani, @{c1.last}\\% în ultimii cinci ani'),
    T('Five years before and after the upgrade: VR(5) $@{c1.before.vr}$ ($Z^* = @{c1.before.zs}$) and $@{c1.after.vr}$ ($Z^* = @{c1.after.zs}$); difference $z = @{c1.z}$: no evidence',
      'Cinci ani înainte și după: VR(5) $@{c1.before.vr}$ ($Z^* = @{c1.before.zs}$) și $@{c1.after.vr}$ ($Z^* = @{c1.after.zs}$); diferența $z = @{c1.z}$: nicio dovadă'),
    T('Confounders: COVID-19 (2020), Hidroelectrica listing (2023), volumes, index composition; designs: the same test on BVB stocks, WIG20 and BUX as controls',
      'Factori care se suprapun: COVID-19 (2020), listarea Hidroelectrica (2023), volumele, componența indicelui; variante: același test pe acțiunile BVB, WIG20 și BUX drept comparație')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant whether the DAX (daily, since 1990) is weak-form efficient. The answer:',
      'Un student a cerut unui asistent AI să spună dacă DAX (zilnic, din 1990) este eficient în formă slabă. Răspunsul:'),
    T('\\aiprompt{(a) ADF on the DAX log price gives tau = @{c2.adf}, p = @{c2.adfp}: the unit root is not rejected, so the DAX is weak-form efficient.}',
      '\\aiprompt{(a) ADF pe prețul logaritmic DAX dă tau = @{c2.adf}, p = @{c2.adfp}: rădăcina unitară nu este respinsă, deci DAX este eficient în formă slabă.}'),
    T('\\aiprompt{(b) Ljung-Box Q(10) = @{c2.lb}, p = @{c2.plb}: DAX returns are predictable, so the DAX is not efficient.}',
      '\\aiprompt{(b) Ljung-Box Q(10) = @{c2.lb}, p = @{c2.plb}: randamentele DAX sînt previzibile, deci DAX nu este eficient.}'),
    T('\\aiprompt{(c) For squared returns Q(10) = @{c2.lbsq}: DAX returns are not a martingale.}',
      '\\aiprompt{(c) Pentru randamentele la pătrat Q(10) = @{c2.lbsq}: randamentele DAX nu sînt un martingal.}'),
    T('\\aiprompt{(d) VR(10) = @{c2.vr10} < 1 shows positive autocorrelation (momentum) in the DAX.}',
      '\\aiprompt{(d) VR(10) = @{c2.vr10} < 1 arată autocorelație pozitivă (momentum) în DAX.}'),
    T('\\aiprompt{(e) Under RW1, VR(q) = 1 at every horizon and the variance of q-day returns is q times the daily variance.}',
      '\\aiprompt{(e) Pentru RW1, VR(q) = 1 la orice orizont, iar varianța randamentelor pe q zile este de q ori varianța zilnică.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație spuneți dacă este corectă; dacă nu, dați afirmația corectă și, unde se poate, cifra corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: a unit root is necessary, not sufficient; $I(1)$ prices can have predictable increments; test the returns', '(a) Greșit: rădăcina unitară este necesară, nu suficientă; prețurile $I(1)$ pot avea creșteri previzibile; testați randamentele'),
    T('(b) Wrong: $Q_{LB}$ assumes i.i.d.\\ returns; the robust $\\tilde Q(10) = @{c2.rob}$ (p = @{c2.prob}) does not reject', '(b) Greșit: $Q_{LB}$ presupune randamente i.i.d.; $\\tilde Q(10)$ robust $= @{c2.rob}$ (p = @{c2.prob}) nu respinge'),
    T('(c) Wrong: correlated squared returns are volatility clustering; they reject RW1, not the martingale (robust $\\tilde Q$ of the squares: $@{c2.robsq}$, still RW1 only)',
      '(c) Greșit: randamentele la pătrat corelate înseamnă gruparea volatilității; resping RW1, nu martingalul ($\\tilde Q$ robust al pătratelor: $@{c2.robsq}$, tot doar RW1)'),
    T('(d) Wrong: VR $< 1$ means negative autocorrelation (mean reversion), and $Z^*(10) = @{c2.zs10}$ is not significant ($Z(10) = @{c2.z10}$)',
      '(d) Greșit: VR $< 1$ înseamnă autocorelație negativă (revenire la medie), iar $Z^*(10) = @{c2.zs10}$ nu este semnificativ ($Z(10) = @{c2.z10}$)'),
    T('(e) Correct: $\\mathrm{Var}(r_t(q)) = q\\sigma^2$ under RW1 (indeed under RW3 as well)', '(e) Corect: $\\mathrm{Var}(r_t(q)) = q\\sigma^2$ pentru RW1 (de fapt și pentru RW3)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('A unit root in prices is the starting point, not the answer: efficiency is about the returns', 'O rădăcină unitară în prețuri este punctul de plecare, nu răspunsul: eficiența privește randamentele'),
    T('Test the martingale with robust statistics ($\\tilde Q$, $Z^*$): classic tests mistake volatility clustering for predictability',
      'Testați martingalul cu statistici robuste ($\\tilde Q$, $Z^*$): testele clasice confundă gruparea volatilității cu previzibilitatea'),
    T('VR $> 1$: momentum; VR $< 1$: mean reversion; the S\\&P 500 reverts slightly, the old BET trended', 'VR $> 1$: momentum; VR $< 1$: revenire la medie; S\\&P 500 revine ușor, vechiul BET urma tendința'),
    T('Many horizons or many markets: use Chow--Denning or a Šidák correction', 'Multe orizonturi sau multe piețe: folosiți Chow--Denning sau o corecție Šidák'),
    T('An AI answer is a draft: check the null of each test, the sign of VR $- 1$ and the robust version', 'Un răspuns AI este o ciornă: verificați ipoteza nulă a fiecărui test, semnul lui VR $- 1$ și versiunea robustă')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 7 develops each topic of today: efficiency, random walks, ACF tests, unit roots, variance ratios, adaptive markets',
      'Cursul 7 dezvoltă fiecare temă de azi: eficiența, mersul aleator, testele ACF, rădăcinile unitare, rapoartele varianțelor, piețele adaptive'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: BET stocks, WIG20 and BUX as controls, a block bootstrap of the difference', 'C1 poate deveni un proiect de echipă: acțiunile din BET, WIG20 și BUX drept comparație, un bootstrap pe blocuri al diferenței'),
    T('Reading: \\refFHH, Ch.~11; exercises in \\refBHL, Ch.~11; \\refCLM, Ch.~2', 'Lectură: \\refFHH, cap.~11; exerciții în \\refBHL, cap.~11; \\refCLM, cap.~2')))

D.references(bib(['BHL', 'BP', 'Choi', 'CD', 'CLM', 'DF', 'EL', 'FamaA', 'FHH', 'KimB', 'KPSS', 'Ljung', 'LM', 'MacKinnon', 'PP', 'SD', 'WW']), per=17)

if __name__ == '__main__':
    D.write(V)
