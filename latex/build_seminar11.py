r"""
build_seminar11.py -- Seminarul 11 (Ipoteza piețelor fractale și memoria lungă), EN + RO dintr-o singură sursă
=============================================================================================================
Seminarul are loc ÎNAINTEA cursului 11: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_11/sem11_results.json (seminar11.py).
Ieșire:
  EN/Seminars/seminar11_fractal_markets_long_memory.tex              (+ _solutions.tex)
  RO/Seminarii/seminar11_piete_fractale_memorie_lunga_ro.tex         (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_11/seminar11.py && python3 latex/build_seminar11.py && python3 latex/sfm_build.py compile 11
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch11_common import NAMES, REFS, SHORT, T, bib, load_sem, put_date   # noqa: E402

S = load_sem()
V = Values()
D = Deck(11, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_11}{SFM_ch11_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    d = A[a]
    V.put(f'{a}.mean', d['mean'], 4 if a == 'a2' else 3)
    for i, (dv, y) in enumerate(zip(d['dev'], d['Y']), 1):
        V.put(f'{a}.dev{i}', dv, 4 if a == 'a2' else 3, sign=True)
        V.put(f'{a}.y{i}', round(y, 10) + 0.0, 4 if a == 'a2' else 3)
    V.put(f'{a}.max', d['max'], 4 if a == 'a2' else 3)
    V.put(f'{a}.min', d['min'] + 0.0, 4 if a == 'a2' else 3)
    V.raw(f'{a}.kmax', str(d['kmax']))
    V.raw(f'{a}.kmin', str(d['kmin']))
    V.put(f'{a}.R', d['R'], 3)
    V.put(f'{a}.S', d['S'], 3)
    V.put(f'{a}.RS', d['RS'], 3)
    V.put(f'{a}.ss', d['sumsq'], 4 if a == 'a2' else 3)
    P = A[f'{a}_points']
    for j, (rs, lr, ln) in enumerate(zip(P['rs'], P['log_rs'], P['log_n']), 1):
        V.put(f'{a}.p{j}', rs, 2)
        V.put(f'{a}.lr{j}', lr, 3)
        V.put(f'{a}.ln{j}', ln, 3)
    V.put(f'{a}.H', P['H'], 3)
    V.put(f'{a}.dlr', P['log_rs'][2] - P['log_rs'][0], 3)
    V.put(f'{a}.dln', P['log_n'][2] - P['log_n'][0], 3)
for a in ['a3', 'a4', 'a4b']:
    d = A[a]
    V.put(f'{a}.r1', d['rho1'], 3)
    V.put(f'{a}.r10', d['rho10'], 4)
    V.put(f'{a}.r10a', d['rho10_asy'], 4)
    V.put(f'{a}.sc', d['scale'], 3)
    V.put(f'{a}.v1', d['var1'], 2)
    V.put(f'{a}.vh', d['varh'], 2)
    V.put(f'{a}.vs', d['varh_sqrt'], 2)
    V.put(f'{a}.diff', d['diff_pct'], 1)
    V.put(f'{a}.D', d['D'], 2)
V.put('sqrt10', math.sqrt(10), 3)
for a in ['a5', 'a6', 'a6n']:
    d = A[a]
    for c in ['pi1', 'pi2', 'psi1', 'psi2', 'rho1', 'rho2', 'rho10', 'rho50', 'phi', 'ar10', 'ar50']:
        V.put(f'{a}.{c}', d[c], 3 if c not in ('ar10', 'ar50') else 4)
    V.raw(f'{a}.fa', str(d['first_arfima']) if d['first_arfima'] else '2000')
    V.raw(f'{a}.far', str(d['first_ar']))
V.put('a6.ar50', A['a6']['ar50'], 5)


def put_mem(key, d):
    e, mc = d['est'], d['mc']
    V.int(f'{key}.n', e['n'])
    V.raw(f'{key}.y0', d['first'][:4])
    for c in ['rs', 'dfa', 'gph_H']:
        V.put(f'{key}.{c}', e[c], 3)
    V.put(f'{key}.d', e['gph_d'], 3)
    V.put(f'{key}.dse', e['gph_se'], 3)
    V.raw(f'{key}.m', str(e['gph_m']))
    V.put(f'{key}.V', e['lo_V'], 3)
    V.put(f'{key}.V0', e['lo_V0'], 3)
    V.raw(f'{key}.q', str(e['lo_q']))
    for m in ['rs', 'dfa', 'gph']:
        V.put(f'{key}.{m}.lo', mc[m]['q025'], 3)
        V.put(f'{key}.{m}.hi', mc[m]['q975'], 3)
        V.put(f'{key}.{m}.mean', mc[m]['mean'], 3)
    V.put(f'{key}.z', e['gph_d'] / e['gph_se'], 2)


B = S['B']
put_mem('b1', B['b1'])
for k, d in B['b2'].items():
    put_mem(f'b2.{k}', d)
put_mem('b4r', B['b4_r'])


def put_vol(key, d):
    for part in ['abs', 'sq', 'abs_shuffled']:
        e = d[part]
        k = {'abs': 'a', 'sq': 's', 'abs_shuffled': 'sh'}[part]
        V.put(f'{key}.{k}.g5', e['gph05'], 3)
        V.put(f'{key}.{k}.g6', e['gph06'], 3)
        V.put(f'{key}.{k}.se5', e['se05'], 3)
        V.put(f'{key}.{k}.se6', e['se06'], 3)
        V.raw(f'{key}.{k}.m5', str(e['m05']))
        V.raw(f'{key}.{k}.m6', str(e['m06']))
        V.put(f'{key}.{k}.dfa', e['dfa'], 3)
        for lag in ['1', '50', '250']:
            V.put(f'{key}.{k}.acf{lag}', e[f'acf{lag}'], 3)
    V.put(f'{key}.band', d['band'], 3)
    V.int(f'{key}.n', d['n'])


put_vol('b3', B['b3'])
put_vol('b4', B['b4'])


def put_roll(key, d):
    V.raw(f'{key}.nw', str(d['n_windows']))
    for c in ['q025', 'q975', 'mean', 'min', 'max', 'first_third', 'last_third', 'last']:
        V.put(f'{key}.{c}', d[c], 3)
    V.put(f'{key}.above', 100 * d['above'], 1)
    V.put(f'{key}.below', 100 * d['below'], 1)
    put_date(V, f'{key}.dmin', d['date_min'])
    put_date(V, f'{key}.dmax', d['date_max'])
    put_date(V, f'{key}.e0', d['first_end'])
    V.put(f'{key}.fa', 0.05 * d['n_windows'], 0)


put_roll('b5', B['b5'])
for k, d in B['b6'].items():
    put_roll(f'b6.{k}', d)
C1 = S['C']['c1']
for k in ['sp500', 'bet']:
    for lab, kk in [('calm 2017', 'calm'), ('GFC 2008-09', 'gfc'), ('COVID 2020-21', 'covid')]:
        d = C1[k][lab]
        V.put(f'c1.{k}.{kk}.r', d['dfa_r'], 2)
        V.put(f'c1.{k}.{kk}.a', d['dfa_abs'], 2)
        V.put(f'c1.{k}.{kk}.v', d['vol'], 1)
        V.raw(f'c1.{k}.{kk}.n', str(d['n']))
V.put('c1.lo', C1['band250']['q025'], 2)
V.put('c1.hi', C1['band250']['q975'], 2)
C2 = S['C']['c2']
V.put('c2.rs', C2['rs'], 3)
V.put('c2.rshi', C2['rs_q975'], 3)
V.put('c2.rsm', C2['rs_mean_iid'], 3)
V.put('c2.dfa', C2['dfa'], 3)
V.put('c2.V', C2['lo_V'], 2)
V.put('c2.d', C2['abs_d'], 3)
V.put('c2.H', C2['abs_H'], 3)
V.put('c2.Hw', C2['H_from_wrong_d'], 3)
V.raw('c2.mfull', str(C2['gph_m_full']))
V.put('c2.rsp', 100 * C2['rs'], 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how can we measure the long memory of a market without seeing memory where there is none?',
       '\\textbf{Întrebarea}: cum putem măsura memoria lungă a unei piețe fără să vedem memorie acolo unde nu există?'),
     [T('this seminar comes \\textbf{before} Lecture 11: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 11: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: R/S step by step, the Hurst exponent from an R/S plot, fractional Gaussian noise and risk scaling, ARFIMA weights and autocorrelations, on paper',
        'Partea A: R/S pas cu pas, exponentul Hurst dintr-un grafic R/S, zgomotul gaussian fracționar și scalarea riscului, ponderile și autocorelațiile ARFIMA, pe hîrtie'),
      T('Part B: R/S, DFA, GPH and Lo\'s test with Monte Carlo bands on the S\\&P 500, BET, Banca Transilvania and Bitcoin; long memory in volatility; rolling exponents; each with an interpretation question',
        'Partea B: R/S, DFA, GPH și testul lui Lo cu benzi Monte Carlo pe S\\&P 500, BET, Banca Transilvania și Bitcoin; memoria lungă a volatilității; exponenți pe ferestre mobile; fiecare cu o întrebare de interpretare'),
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
    ['A1, A2 & ' + T('R/S of eight returns step by step; $H$ from three points of an R/S plot', 'R/S pentru opt randamente, pas cu pas; $H$ din trei puncte ale unui grafic R/S') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('fractional Gaussian noise: autocorrelations, 10-day risk, VaR 1\\% (value at risk)', 'zgomot gaussian fracționar: autocorelații, riscul pe 10 zile, VaR 1\\% (value at risk, valoarea expusă la risc)') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('ARFIMA(0,$d$,0): weights and autocorrelations against an AR(1)', 'ARFIMA(0,$d$,0): ponderi și autocorelații comparate cu un AR(1)') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('R/S, DFA, GPH and Lo\'s test with Monte Carlo bands: S\\&P 500; BET, TLV', 'R/S, DFA, GPH și testul lui Lo cu benzi Monte Carlo: S\\&P 500; BET, TLV') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('long memory in $|r_t|$ and $r_t^2$, the shuffle test: S\\&P 500; Bitcoin', 'memoria lungă a lui $|r_t|$ și $r_t^2$, testul permutării: S\\&P 500; Bitcoin') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('rolling DFA exponents with a Monte Carlo band: S\\&P 500; DAX, TLV', 'exponenți DFA pe ferestre mobile, cu o bandă Monte Carlo: S\\&P 500; DAX, TLV') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('does memory change in crises? what is wrong in an AI answer?', 'se schimbă memoria în crize? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX, BET & EODHD & ' + T('close', 'închidere') + ' & 2000--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'închidere, 7 zile pe săptămînă') + ' & 2014--2026',
     T('Banca Transilvania (TLV)', 'Banca Transilvania (TLV)') + ' & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2010--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin); the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); ultima zi este 18 septembrie 2026'),
    T('TLV: the days 30 and 31 May 2016 are removed (a price adjustment applied one day late, Chapter 2)',
      'TLV: zilele de 30 și 31 mai 2016 se elimină (o ajustare de preț aplicată cu o zi mai tîrziu, Capitolul 2)'),
    T('In the notebook: \\texttt{returns(\'bet\')}, \\texttt{hurst\\_rs(x)}, \\texttt{hurst\\_dfa(x)}, \\texttt{gph(x)}, \\texttt{lo\\_test(x)}, \\texttt{mc\\_null(N)}; no account or key is needed',
      'În notebook: \\texttt{returns(\'bet\')}, \\texttt{hurst\\_rs(x)}, \\texttt{hurst\\_dfa(x)}, \\texttt{gph(x)}, \\texttt{lo\\_test(x)}, \\texttt{mc\\_null(N)}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): the Fractal Market Hypothesis and $H$', 'Noțiuni necesare azi (1/5): ipoteza pieței fractale și $H$'), items(
    (T('\\textbf{FMH} (fractal market hypothesis) \\refPeters: a market is stable when investors with many horizons (minutes to decades) trade with each other',
       '\\textbf{FMH} (fractal market hypothesis, ipoteza pieței fractale) \\refPeters: o piață este stabilă cînd investitori cu multe orizonturi (de la minute la decenii) tranzacționează între ei'),
     [T('in a crisis, all horizons shrink to the shortest one: no buyers, no liquidity', 'într-o criză, toate orizonturile se reduc la cel mai scurt: fără cumpărători, fără lichiditate'),
      T('testable through scaling and memory across horizons', 'testabilă prin scalare și memorie pe orizonturi diferite')]),
    (T('\\textbf{Self-similarity}: $X_{ct} \\overset{d}{=} c^H X_t$; the $h$-day standard deviation is $h^H$ times the daily one', '\\textbf{Autosimilaritate}: $X_{ct} \\overset{d}{=} c^H X_t$; abaterea standard pe $h$ zile este de $h^H$ ori cea zilnică'),
     [T('\\textbf{Hurst exponent} $H \\in (0, 1)$: $H = 0.5$ independent increments (random walk, EMH); $H > 0.5$ persistence; $H < 0.5$ anti-persistence',
        '\\textbf{Exponentul Hurst} $H \\in (0, 1)$: $H = 0.5$ creșteri independente (mers aleator, EMH); $H > 0.5$ persistență; $H < 0.5$ antipersistență'),
      T('$H$ is not a probability; the fractal dimension of a price path is $D = 2 - H$', '$H$ nu este o probabilitate; dimensiunea fractală a unei traiectorii de preț este $D = 2 - H$')])))

D.frame(T('What You Need for Today (2/5): the R/S Statistic', 'Noțiuni necesare azi (2/5): statistica R/S'), items(
    (T('Block $x_1, \\dots, x_n$, mean $\\bar x$; cumulative deviations $Y_k = \\sum_{j=1}^{k}(x_j - \\bar x)$', 'Blocul $x_1, \\dots, x_n$, media $\\bar x$; abaterile cumulate $Y_k = \\sum_{j=1}^{k}(x_j - \\bar x)$'),
     [T('range $R_n = \\max_k Y_k - \\min_k Y_k$; standard deviation $S_n = \\sqrt{\\tfrac1n\\sum_j(x_j - \\bar x)^2}$; $(R/S)_n = R_n/S_n$ \\refHurst',
        'amplitudinea $R_n = \\max_k Y_k - \\min_k Y_k$; abaterea standard $S_n = \\sqrt{\\tfrac1n\\sum_j(x_j - \\bar x)^2}$; $(R/S)_n = R_n/S_n$ \\refHurst')]),
    (T('$(R/S)_n \\approx c\\,n^H$: average $R/S$ over non-overlapping blocks of size $n$, then regress $\\log(R/S)_n$ on $\\log n$; the slope is $\\hat H$',
       '$(R/S)_n \\approx c\\,n^H$: facem media $R/S$ pe blocuri fără suprapunere de mărime $n$, apoi estimăm regresia lui $\\log(R/S)_n$ pe $\\log n$; panta este $\\hat H$'),
     [T('three block sizes equally spaced on a log scale: the OLS slope equals the slope between the outer points', 'trei mărimi de bloc la distanțe egale pe scară logaritmică: panta OLS este egală cu panta dintre punctele extreme')]),
    T('Small samples: for i.i.d.\\ data, the R/S slope is above 0.5 (about $@{b1.rs.mean}$ for $N = @{b1.n}$) \\refAL', 'Eșantioane mici: pentru date i.i.d., panta R/S este peste 0,5 (circa $@{b1.rs.mean}$ pentru $N = @{b1.n}$) \\refAL')))

D.frame(T('What You Need for Today (3/5): Long Memory, fGn and ARFIMA', 'Noțiuni necesare azi (3/5): memoria lungă, fGn și ARFIMA'), items(
    T('\\textbf{Long memory}: $\\rho(k) \\sim C\\,k^{2d - 1}$, $0 < d < 0.5$; the autocorrelations decay hyperbolically and are not summable (\\refFHH, 14.1); short memory (AR): $\\rho(k) = \\phi^k$',
      '\\textbf{Memoria lungă}: $\\rho(k) \\sim C\\,k^{2d - 1}$, $0 < d < 0.5$; autocorelațiile scad hiperbolic și nu sînt sumabile (\\refFHH, 14.1); memoria scurtă (AR): $\\rho(k) = \\phi^k$'),
    (T('\\textbf{Fractional Gaussian noise} (fGn), the increments of fractional Brownian motion \\refMVN:', '\\textbf{Zgomotul gaussian fracționar} (fGn, fractional Gaussian noise), creșterile mișcării browniene fracționare \\refMVN:'),
     [T('$\\rho(k) = \\tfrac12\\left(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}\\right) \\approx H(2H - 1)k^{2H - 2}$; $\\rho(1) = 2^{2H - 1} - 1$', '$\\rho(k) = \\tfrac12\\left(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}\\right) \\approx H(2H - 1)k^{2H - 2}$; $\\rho(1) = 2^{2H - 1} - 1$')]),
    (T('\\textbf{ARFIMA}(0,$d$,0) \\refHosking: $(1 - L)^d X_t = \\varepsilon_t$, $d = H - 0.5$', '\\textbf{ARFIMA}(0,$d$,0) \\refHosking: $(1 - L)^d X_t = \\varepsilon_t$, $d = H - 0.5$'),
     [T('$(1 - L)^d = \\sum_j \\pi_j L^j$: $\\pi_0 = 1$, $\\pi_j = \\pi_{j-1}(j - 1 - d)/j$; MA weights $\\psi_j = \\psi_{j-1}(j - 1 + d)/j$', '$(1 - L)^d = \\sum_j \\pi_j L^j$: $\\pi_0 = 1$, $\\pi_j = \\pi_{j-1}(j - 1 - d)/j$; ponderile MA $\\psi_j = \\psi_{j-1}(j - 1 + d)/j$'),
      T('$\\rho(1) = d/(1 - d)$, $\\rho(k) = \\rho(k - 1)\\,(k - 1 + d)/(k - d)$', '$\\rho(1) = d/(1 - d)$, $\\rho(k) = \\rho(k - 1)\\,(k - 1 + d)/(k - d)$')])), 'footnotesize')

D.frame(T('What You Need for Today (4/5): DFA, GPH, Lo\'s Test and Monte Carlo Bands', 'Noțiuni necesare azi (4/5): DFA, GPH, testul lui Lo și benzile Monte Carlo'), items(
    T('\\textbf{DFA} \\refPeng: profile $Y_k$; straight line fitted in each block of size $n$; $F(n)$ = root mean square of the deviations; slope of $\\log F$ on $\\log n$ = $\\hat H$',
      '\\textbf{DFA} \\refPeng: profilul $Y_k$; o dreaptă estimată în fiecare bloc de mărime $n$; $F(n)$ = rădăcina mediei pătratelor abaterilor; panta lui $\\log F$ în funcție de $\\log n$ = $\\hat H$'),
    T('\\textbf{GPH} \\refGPH: OLS of $\\ln I(\\lambda_j)$ (periodogram) on $-\\ln(4\\sin^2(\\lambda_j/2))$, $j \\le m = \\lfloor T^{0.5}\\rfloor$; slope $\\hat d$, SE $\\pi/\\sqrt{24m}$',
      '\\textbf{GPH} \\refGPH: OLS pentru $\\ln I(\\lambda_j)$ (periodograma) în funcție de $-\\ln(4\\sin^2(\\lambda_j/2))$, $j \\le m = \\lfloor T^{0.5}\\rfloor$; panta $\\hat d$, SE $\\pi/\\sqrt{24m}$'),
    T("\\textbf{Lo's modified R/S} \\refLo: $V = R/(\\sqrt{N}\\hat\\sigma(q))$ with a Newey--West variance; $H_0$: short memory; reject at 5\\% if $V \\notin [0.809, 1.862]$",
      '\\textbf{R/S modificat al lui Lo} \\refLo: $V = R/(\\sqrt{N}\\hat\\sigma(q))$, cu o varianță Newey--West; $H_0$: memorie scurtă; respingem la 5\\% dacă $V \\notin [0.809, 1.862]$'),
    (T('\\textbf{Monte Carlo band} \\refWeron: simulate many i.i.d.\\ Normal series of the same length, apply the same estimator, take the 2.5\\% and 97.5\\% quantiles',
       '\\textbf{Banda Monte Carlo} \\refWeron: simulăm multe serii i.i.d.\\ Normale de aceeași lungime, aplicăm același estimator, luăm cuantilele de 2,5\\% și 97,5\\%'),
     [T('reject $H = 0.5$ only if $\\hat H$ lies outside the band', 'respingem $H = 0.5$ doar dacă $\\hat H$ se află în afara benzii')])), 'footnotesize')

D.frame(T('What You Need for Today (5/5): Spurious Memory, Volatility, Rolling Windows', 'Noțiuni necesare azi (5/5): memorie aparentă, volatilitate, ferestre mobile'), items(
    T('\\textbf{Spurious long memory}: mean shifts \\refDI\\ and persistent GARCH volatility (Chapter 9) produce slowly decaying sample ACFs',
      '\\textbf{Memorie lungă aparentă}: schimbările mediei \\refDI\\ și volatilitatea GARCH persistentă (Capitolul 9) produc ACF de selecție care scad lent'),
    T('\\textbf{Shuffle test}: a random permutation keeps the distribution and destroys the order; if $\\hat H$ falls to 0.5, the memory was in the order of the days',
      '\\textbf{Testul permutării}: o permutare aleatoare păstrează distribuția și distruge ordinea; dacă $\\hat H$ scade la 0,5, memoria era în ordinea zilelor'),
    T('\\textbf{Volatility}: long memory of $|r_t|$ and $r_t^2$ \\refDGE\\ concerns the size of returns, not their direction', '\\textbf{Volatilitatea}: memoria lungă a lui $|r_t|$ și $r_t^2$ \\refDGE\\ privește mărimea randamentelor, nu direcția lor'),
    T('\\textbf{Rolling windows}: $\\hat H$ on windows of 1000 days moved by 21, dated at the window end; with 5\\% tests, about 5\\% of windows fall outside the band by chance',
      '\\textbf{Ferestre mobile}: $\\hat H$ pe ferestre de 1000 de zile mutate cu 21, datat la sfîrșitul ferestrei; cu teste la 5\\%, circa 5\\% dintre ferestre ies din bandă din întîmplare')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: R/S Step by Step', 'A1: R/S pas cu pas'),
         items(T('Eight daily returns (\\%): $0.5;\\ -0.3;\\ 0.8;\\ -0.2;\\ 0.6;\\ -0.1;\\ 0.4;\\ -0.7$.', 'Opt randamente zilnice (\\%): $0.5;\\ -0.3;\\ 0.8;\\ -0.2;\\ 0.6;\\ -0.1;\\ 0.4;\\ -0.7$.'),
               T('1. Compute the mean, the deviations and the cumulative deviations $Y_1, \\dots, Y_8$.', '1. Calculați media, abaterile și abaterile cumulate $Y_1, \\dots, Y_8$.'),
               T('2. Compute $R$, $S$ (divisor $n$) and $R/S$.', '2. Calculați $R$, $S$ (cu numitorul $n$) și $R/S$.'),
               T('3. S\\&P 500 returns since 2000 give $(R/S)_{16} = @{a1.p1}$, $(R/S)_{64} = @{a1.p2}$, $(R/S)_{256} = @{a1.p3}$; compute $\\hat H$.', '3. Randamentele S\\&P 500 din 2000 dau $(R/S)_{16} = @{a1.p1}$, $(R/S)_{64} = @{a1.p2}$, $(R/S)_{256} = @{a1.p3}$; calculați $\\hat H$.'),
               T('Report: $R$, $S$, $R/S$, $\\hat H$ and one sentence.', 'Raportați: $R$, $S$, $R/S$, $\\hat H$ și o frază.')),
         items(T('1. $\\bar x = @{a1.mean}$; $Y_k$: $@{a1.y1}$, $@{a1.y2}$, $@{a1.y3}$, $@{a1.y4}$, $@{a1.y5}$, $@{a1.y6}$, $@{a1.y7}$, $@{a1.y8}$',
                 '1. $\\bar x = @{a1.mean}$; $Y_k$: $@{a1.y1}$; $@{a1.y2}$; $@{a1.y3}$; $@{a1.y4}$; $@{a1.y5}$; $@{a1.y6}$; $@{a1.y7}$; $@{a1.y8}$'),
               T('2. $R = @{a1.max} - (@{a1.min}) = @{a1.R}$; $S = \\sqrt{@{a1.ss}/8} = @{a1.S}$; $R/S = @{a1.RS}$', '2. $R = @{a1.max} - (@{a1.min}) = @{a1.R}$; $S = \\sqrt{@{a1.ss}/8} = @{a1.S}$; $R/S = @{a1.RS}$'),
               T('3. $\\log_{10}$: $n$: $@{a1.ln1}$, $@{a1.ln2}$, $@{a1.ln3}$; $R/S$: $@{a1.lr1}$, $@{a1.lr2}$, $@{a1.lr3}$; $\\hat H = @{a1.dlr}/@{a1.dln} = @{a1.H}$',
                 '3. $\\log_{10}$: $n$: $@{a1.ln1}$; $@{a1.ln2}$; $@{a1.ln3}$; $R/S$: $@{a1.lr1}$; $@{a1.lr2}$; $@{a1.lr3}$; $\\hat H = @{a1.dlr}/@{a1.dln} = @{a1.H}$'),
               T('Alternating signs keep the cumulative deviations small; $\\hat H$ slightly above 0.5 is the small-sample bias of R/S, not memory (B1).',
                 'Semnele alternante țin abaterile cumulate mici; un $\\hat H$ puțin peste 0,5 reflectă deplasarea R/S în eșantioane mici, nu memoria (B1).')),
         size='scriptsize')

D.proposed(T('A2: Grouped Signs', 'A2: semne grupate'),
           items(T('Eight returns (\\%): $0.6;\\ 0.4;\\ 0.5;\\ 0.3;\\ -0.4;\\ -0.6;\\ -0.2;\\ -0.5$. Model: A1.', 'Opt randamente (\\%): $0.6;\\ 0.4;\\ 0.5;\\ 0.3;\\ -0.4;\\ -0.6;\\ -0.2;\\ -0.5$. Model: A1.'),
                 T('1. Compute the mean and $Y_1, \\dots, Y_8$.', '1. Calculați media și $Y_1, \\dots, Y_8$.'),
                 T('2. Compute $R$, $S$ and $R/S$, and compare with A1.', '2. Calculați $R$, $S$ și $R/S$ și comparați cu A1.'),
                 T('3. BET returns give $(R/S)_{16} = @{a2.p1}$, $(R/S)_{64} = @{a2.p2}$, $(R/S)_{256} = @{a2.p3}$; compute $\\hat H$.', '3. Randamentele BET dau $(R/S)_{16} = @{a2.p1}$, $(R/S)_{64} = @{a2.p2}$, $(R/S)_{256} = @{a2.p3}$; calculați $\\hat H$.'),
                 T('Report: $R/S$, $\\hat H$ and two sentences.', 'Raportați: $R/S$, $\\hat H$ și două fraze.')),
           items(T('1. $\\bar x = @{a2.mean}$; $Y_k$: $@{a2.y1}$, $@{a2.y2}$, $@{a2.y3}$, $@{a2.y4}$, $@{a2.y5}$, $@{a2.y6}$, $@{a2.y7}$, $@{a2.y8}$',
                   '1. $\\bar x = @{a2.mean}$; $Y_k$: $@{a2.y1}$; $@{a2.y2}$; $@{a2.y3}$; $@{a2.y4}$; $@{a2.y5}$; $@{a2.y6}$; $@{a2.y7}$; $@{a2.y8}$'),
                 T('2. $R = @{a2.R}$, $S = @{a2.S}$, $R/S = @{a2.RS}$, about twice A1 ($@{a1.RS}$)', '2. $R = @{a2.R}$, $S = @{a2.S}$, $R/S = @{a2.RS}$, aproape dublu față de A1 ($@{a1.RS}$)'),
                 T('3. $\\hat H = (@{a2.lr3} - @{a2.lr1})/@{a2.dln} = @{a2.H}$', '3. $\\hat H = (@{a2.lr3} - @{a2.lr1})/@{a2.dln} = @{a2.H}$'),
                 T('Runs of equal signs make the cumulative deviations drift: a large $R/S$ is the signature of persistence; BET is above the S\\&P 500 (A1).',
                   'Seriile de semne egale fac ca abaterile cumulate să se îndepărteze de zero: un $R/S$ mare este semnul persistenței; BET este peste S\\&P 500 (A1).')),
           size='scriptsize')

D.solved(T('A3: Fractional Gaussian Noise and 10-Day Risk', 'A3: zgomot gaussian fracționar și riscul pe 10 zile'),
         items(T('Daily returns follow an fGn with $H = 0.7$ and daily volatility $\\sigma = 1.2\\%$; $z_{0.99} = 2.326$.', 'Randamentele zilnice urmează un fGn cu $H = 0.7$ și volatilitatea zilnică $\\sigma = 1.2\\%$; $z_{0.99} = 2.326$.'),
               T('1. Compute $\\rho(1)$ and $\\rho(10)$ exactly and $\\rho(10)$ with the approximation $H(2H - 1)k^{2H - 2}$.', '1. Calculați exact $\\rho(1)$ și $\\rho(10)$, precum și $\\rho(10)$ cu aproximarea $H(2H - 1)k^{2H - 2}$.'),
               T('2. Compute the scaling factor $10^H$ and compare it with $\\sqrt{10}$.', '2. Calculați factorul de scalare $10^H$ și comparați-l cu $\\sqrt{10}$.'),
               T('3. Compute the one-day and the 10-day VaR 1\\% (Normal) with $h^H$ and with $\\sqrt{h}$.', '3. Calculați VaR 1\\% pe o zi și pe 10 zile (cazul Normal), cu $h^H$ și cu $\\sqrt{h}$.'),
               T('4. Compute the fractal dimension of the price path.', '4. Calculați dimensiunea fractală a traiectoriei prețului.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. $\\rho(1) = 2^{0.4} - 1 = @{a3.r1}$; $\\rho(10) = \\tfrac12(11^{1.4} - 2 \\cdot 10^{1.4} + 9^{1.4}) = @{a3.r10}$; approximation $0.7 \\cdot 0.4 \\cdot 10^{-0.6} = @{a3.r10a}$',
                 '1. $\\rho(1) = 2^{0.4} - 1 = @{a3.r1}$; $\\rho(10) = \\tfrac12(11^{1.4} - 2 \\cdot 10^{1.4} + 9^{1.4}) = @{a3.r10}$; aproximarea $0.7 \\cdot 0.4 \\cdot 10^{-0.6} = @{a3.r10a}$'),
               T('2. $10^{0.7} = @{a3.sc}$ against $\\sqrt{10} = @{sqrt10}$', '2. $10^{0.7} = @{a3.sc}$, față de $\\sqrt{10} = @{sqrt10}$'),
               T('3. $2.326 \\times 1.2 = @{a3.v1}\\%$; 10 days: $@{a3.v1} \\times @{a3.sc} = @{a3.vh}\\%$ against $@{a3.vs}\\%$ ($+@{a3.diff}\\%$)', '3. $2.326 \\times 1.2 = @{a3.v1}\\%$; 10 zile: $@{a3.v1} \\times @{a3.sc} = @{a3.vh}\\%$, față de $@{a3.vs}\\%$ ($+@{a3.diff}\\%$)'),
               T('4. $D = 2 - H = @{a3.D}$', '4. $D = 2 - H = @{a3.D}$'),
               T('Persistent returns make multi-day losses add up: the $\\sqrt{h}$ rule understates 10-day risk by more than a third.', 'Randamentele persistente fac ca pierderile pe mai multe zile să se adune: regula $\\sqrt{h}$ subestimează riscul pe 10 zile cu peste o treime.')),
         size='scriptsize')

D.proposed(T('A4: Anti-Persistence and a Small Deviation from 0.5', 'A4: antipersistență și o abatere mică de la 0,5'),
           items(T('The setting of A3 with $H = 0.4$ and with $H = 0.55$. Model: A3.', 'Datele din A3, cu $H = 0.4$ și cu $H = 0.55$. Model: A3.'),
                 T('1. Compute $\\rho(1)$ and $\\rho(10)$ for both values of $H$.', '1. Calculați $\\rho(1)$ și $\\rho(10)$ pentru ambele valori ale lui $H$.'),
                 T('2. Compute the 10-day VaR 1\\% for both and its difference from the $\\sqrt{h}$ rule, in \\%.', '2. Calculați VaR 1\\% pe 10 zile pentru ambele valori și diferența față de regula $\\sqrt{h}$, în \\%.'),
                 T('3. Compute $D$ for both.', '3. Calculați $D$ pentru ambele valori.'),
                 T('Report: eight numbers and one sentence.', 'Raportați: opt valori și o frază.')),
           items(T('1. $H = 0.4$: $\\rho(1) = @{a4.r1}$, $\\rho(10) = @{a4.r10}$; $H = 0.55$: $@{a4b.r1}$, $@{a4b.r10}$', '1. $H = 0.4$: $\\rho(1) = @{a4.r1}$, $\\rho(10) = @{a4.r10}$; $H = 0.55$: $@{a4b.r1}$, $@{a4b.r10}$'),
                 T('2. $H = 0.4$: $@{a4.vh}\\%$ ($@{a4.diff}\\%$); $H = 0.55$: $@{a4b.vh}\\%$ ($+@{a4b.diff}\\%$); $\\sqrt{h}$ rule: $@{a4.vs}\\%$', '2. $H = 0.4$: $@{a4.vh}\\%$ ($@{a4.diff}\\%$); $H = 0.55$: $@{a4b.vh}\\%$ ($+@{a4b.diff}\\%$); regula $\\sqrt{h}$: $@{a4.vs}\\%$'),
                 T('3. $D = @{a4.D}$ and $@{a4b.D}$', '3. $D = @{a4.D}$ și $@{a4b.D}$'),
                 T('A daily autocorrelation of only $@{a4b.r1}$ already changes the 10-day risk by $@{a4b.diff}\\%$.', 'O autocorelație zilnică de doar $@{a4b.r1}$ schimbă deja riscul pe 10 zile cu $@{a4b.diff}\\%$.')),
           size='scriptsize')

D.solved(T('A5: ARFIMA(0, 0.3, 0) against an AR(1)', 'A5: ARFIMA(0; 0,3; 0) comparat cu un AR(1)'),
         items(T('$(1 - L)^{0.3}X_t = \\varepsilon_t$, $\\varepsilon_t$ i.i.d.\\ $N(0, 1)$.', '$(1 - L)^{0.3}X_t = \\varepsilon_t$, $\\varepsilon_t$ i.i.d.\\ $N(0, 1)$.'),
               T('1. Compute $\\pi_1$, $\\pi_2$ and the MA weights $\\psi_1$, $\\psi_2$.', '1. Calculați $\\pi_1$, $\\pi_2$ și ponderile MA $\\psi_1$, $\\psi_2$.'),
               T('2. Compute $\\rho(1)$, $\\rho(2)$, $\\rho(10)$ and $\\rho(50)$.', '2. Calculați $\\rho(1)$, $\\rho(2)$, $\\rho(10)$ și $\\rho(50)$.'),
               T('3. For an AR(1) with $\\phi = \\rho(1)$, compute $\\rho(10)$ and $\\rho(50)$.', '3. Pentru un AR(1) cu $\\phi = \\rho(1)$, calculați $\\rho(10)$ și $\\rho(50)$.'),
               T('4. Give $H$ and the first lag with $\\rho(k) < 0.05$ for both models.', '4. Dați $H$ și primul decalaj cu $\\rho(k) < 0.05$ pentru ambele modele.'),
               T('Report: ten numbers and one sentence.', 'Raportați: zece valori și o frază.')),
         items(T('1. $\\pi_1 = -0.3$, $\\pi_2 = -0.3 \\times 0.7/2 = @{a5.pi2}$; $\\psi_1 = 0.3$, $\\psi_2 = 0.3 \\times 1.3/2 = @{a5.psi2}$', '1. $\\pi_1 = -0.3$, $\\pi_2 = -0.3 \\times 0.7/2 = @{a5.pi2}$; $\\psi_1 = 0.3$, $\\psi_2 = 0.3 \\times 1.3/2 = @{a5.psi2}$'),
               T('2. $\\rho(1) = 0.3/0.7 = @{a5.rho1}$; $\\rho(2) = @{a5.rho1} \\times 1.3/1.7 = @{a5.rho2}$; $\\rho(10) = @{a5.rho10}$; $\\rho(50) = @{a5.rho50}$',
                 '2. $\\rho(1) = 0.3/0.7 = @{a5.rho1}$; $\\rho(2) = @{a5.rho1} \\times 1.3/1.7 = @{a5.rho2}$; $\\rho(10) = @{a5.rho10}$; $\\rho(50) = @{a5.rho50}$'),
               T('3. $@{a5.phi}^{10} = @{a5.ar10}$; $@{a5.phi}^{50} \\approx 0$', '3. $@{a5.phi}^{10} = @{a5.ar10}$; $@{a5.phi}^{50} \\approx 0$'),
               T('4. $H = 0.8$; below 0.05 after @{a5.far} lags (AR) and @{a5.fa} lags (ARFIMA)', '4. $H = 0.8$; sub 0,05 după @{a5.far} decalaje (AR) și @{a5.fa} decalaje (ARFIMA)'),
               T('Same $\\rho(1)$, very different memory: the AR(1) forgets in days, the ARFIMA in months.', 'Același $\\rho(1)$, memorie foarte diferită: AR(1) uită în cîteva zile, ARFIMA în cîteva luni.')),
         size='scriptsize')

D.proposed(T('A6: Strong Memory and Anti-Persistence', 'A6: memorie puternică și antipersistență'),
           items(T('ARFIMA(0,$d$,0) with $d = 0.45$ and with $d = -0.2$. Model: A5.', 'ARFIMA(0,$d$,0) cu $d = 0.45$ și cu $d = -0.2$. Model: A5.'),
                 T('1. Compute $\\pi_1$, $\\pi_2$, $\\rho(1)$, $\\rho(2)$ and $\\rho(10)$ for both.', '1. Calculați $\\pi_1$, $\\pi_2$, $\\rho(1)$, $\\rho(2)$ și $\\rho(10)$ pentru ambele valori.'),
                 T('2. For $d = 0.45$, compare $\\rho(10)$ and $\\rho(50)$ with an AR(1) with $\\phi = \\rho(1)$.', '2. Pentru $d = 0.45$, comparați $\\rho(10)$ și $\\rho(50)$ cu un AR(1) cu $\\phi = \\rho(1)$.'),
                 T('3. Give $H$ for both and say which one is stationary.', '3. Dați $H$ pentru ambele valori și precizați care este staționar.'),
                 T('Report: a small table and one sentence.', 'Raportați: un tabel mic și o frază.')),
           items(T('1. $d = 0.45$: $\\pi_1 = @{a6.pi1}$, $\\pi_2 = @{a6.pi2}$, $\\rho(1) = @{a6.rho1}$, $\\rho(2) = @{a6.rho2}$, $\\rho(10) = @{a6.rho10}$', '1. $d = 0.45$: $\\pi_1 = @{a6.pi1}$, $\\pi_2 = @{a6.pi2}$, $\\rho(1) = @{a6.rho1}$, $\\rho(2) = @{a6.rho2}$, $\\rho(10) = @{a6.rho10}$'),
                 T('$d = -0.2$: $\\pi_1 = @{a6n.pi1}$, $\\pi_2 = @{a6n.pi2}$, $\\rho(1) = @{a6n.rho1}$, $\\rho(2) = @{a6n.rho2}$, $\\rho(10) = @{a6n.rho10}$', '$d = -0.2$: $\\pi_1 = @{a6n.pi1}$, $\\pi_2 = @{a6n.pi2}$, $\\rho(1) = @{a6n.rho1}$, $\\rho(2) = @{a6n.rho2}$, $\\rho(10) = @{a6n.rho10}$'),
                 T('2. ARFIMA: $\\rho(10) = @{a6.rho10}$, $\\rho(50) = @{a6.rho50}$; AR(1): $@{a6.ar10}$, $@{a6.ar50}$; below 0.05 after @{a6.far} lags (AR); still above 0.05 at lag @{a6.fa} (ARFIMA)',
                   '2. ARFIMA: $\\rho(10) = @{a6.rho10}$, $\\rho(50) = @{a6.rho50}$; AR(1): $@{a6.ar10}$, $@{a6.ar50}$; sub 0,05 după @{a6.far} decalaje (AR); încă peste 0,05 la decalajul @{a6.fa} (ARFIMA)'),
                 T('3. $H = 0.95$ and $H = 0.3$; both stationary ($|d| < 0.5$); $d = -0.2$ is anti-persistent.', '3. $H = 0.95$ și $H = 0.3$; ambele staționare ($|d| < 0.5$); $d = -0.2$ este antipersistent.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Long Memory in S\\&P 500 Returns? [Solved]', 'B1: memorie lungă în randamentele S\\&P 500? [Rezolvat]'),
       T('do daily S\\&P 500 returns have long memory?', 'au randamentele zilnice S\\&P 500 memorie lungă?'),
       T('S\\&P 500 closes since 2000, daily log returns in \\%', 'închiderile S\\&P 500 din 2000, randamente logaritmice zilnice în \\%'),
       [T('Estimate $H$ with R/S and DFA (block sizes from 10 to $N/4$) and $d$ with GPH ($m = \\lfloor T^{0.5}\\rfloor$), with the standard error of $\\hat d$.', 'Estimați $H$ cu R/S și DFA (mărimi de bloc de la 10 la $N/4$) și $d$ cu GPH ($m = \\lfloor T^{0.5}\\rfloor$), cu eroarea standard a lui $\\hat d$.'),
        T("Compute Lo's modified statistic $V(q)$ with Andrews' $q$, and the classical statistic $V(0)$.", 'Calculați statistica modificată a lui Lo, $V(q)$, cu $q$ ales prin regula lui Andrews, și statistica clasică $V(0)$.'),
        T('Simulate 300 i.i.d.\\ Normal series of the same length and give the 95\\% band of each estimator.', 'Simulați 300 de serii i.i.d.\\ Normale de aceeași lungime și dați banda de 95\\% a fiecărui estimator.'),
        T('Draw the R/S plot inside the Monte Carlo envelope.', 'Desenați graficul R/S în interiorul anvelopei Monte Carlo.'),
        T('Interpretation: is the R/S estimate above 0.5 evidence of long memory?', 'Interpretare: este estimarea R/S de peste 0,5 o dovadă de memorie lungă?')],
       T('a table of four estimates with their bands, the chart and two sentences', 'un tabel cu patru estimări și benzile lor, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch11_sem_b1', h='0.30') + table(
    'lrrrr', T(' & R/S & DFA & GPH $\\hat H = \\hat d + 0.5$ & Lo $V(q)$', ' & R/S & DFA & GPH $\\hat H = \\hat d + 0.5$ & Lo $V(q)$'),
    [T('estimate', 'estimare') + ' & $@{b1.rs}$ & $@{b1.dfa}$ & $@{b1.gph_H}$ & $@{b1.V}$ ($q = @{b1.q}$)',
     T('95\\% band', 'banda de 95\\%') + ' & $[@{b1.rs.lo}, @{b1.rs.hi}]$ & $[@{b1.dfa.lo}, @{b1.dfa.hi}]$ & $[@{b1.gph.lo}, @{b1.gph.hi}]$ & $[0.809, 1.862]$'], size='scriptsize') + items(
    T('$N = @{b1.n}$; GPH: $\\hat d = @{b1.d}$, SE $@{b1.dse}$ ($m = @{b1.m}$), $t = @{b1.z}$; classical $V(0) = @{b1.V0}$',
      '$N = @{b1.n}$; GPH: $\\hat d = @{b1.d}$, SE $@{b1.dse}$ ($m = @{b1.m}$), $t = @{b1.z}$; statistica clasică $V(0) = @{b1.V0}$'),
    T('Interpretation: no: i.i.d.\\ series of this length give R/S $@{b1.rs.mean}$ on average; every estimate lies inside its band and Lo does not reject short memory',
      'Interpretare: nu: seriile i.i.d.\\ de această lungime dau în medie R/S $@{b1.rs.mean}$; fiecare estimare se află în banda ei, iar testul lui Lo nu respinge memoria scurtă')) + qlsem(), 'scriptsize')

D.task(T('B2: BET and Banca Transilvania [Proposed]', 'B2: BET și Banca Transilvania [Propus]'),
       T('does the BET or Banca Transilvania show long memory in returns? Model: B1.', 'arată BET sau Banca Transilvania memorie lungă în randamente? Model: B1.'),
       T('BET since 2000, TLV since 2010 (without 30--31 May 2016); daily log returns in \\%', 'BET din 2000, TLV din 2010 (fără 30--31 mai 2016); randamente logaritmice zilnice în \\%'),
       [T('Estimate $H$ with R/S, DFA and GPH for each series.', 'Estimați $H$ cu R/S, DFA și GPH pentru fiecare serie.'),
        T("Compute Lo's $V(q)$ and say whether short memory is rejected at 5\\%.", 'Calculați $V(q)$ al lui Lo și precizați dacă memoria scurtă este respinsă la 5\\%.'),
        T('Compute the Monte Carlo band of each estimator for each length.', 'Calculați banda Monte Carlo a fiecărui estimator pentru fiecare lungime.'),
        T('Interpretation: what could explain the result for the BET?', 'Interpretare: ce ar putea explica rezultatul pentru BET?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f"{SHORT[k]} & @{{b2.{k}.n}} & $@{{b2.{k}.rs}}$ $[@{{b2.{k}.rs.lo}}, @{{b2.{k}.rs.hi}}]$ & $@{{b2.{k}.dfa}}$ $[@{{b2.{k}.dfa.lo}}, @{{b2.{k}.dfa.hi}}]$ & "
            f"$@{{b2.{k}.d}}$ ($@{{b2.{k}.dse}}$) & $@{{b2.{k}.V}}$")


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrr', T(' & $N$ & R/S [band] & DFA [band] & GPH $\\hat d$ (SE) & Lo $V(q)$', ' & $N$ & R/S [banda] & DFA [banda] & GPH $\\hat d$ (SE) & Lo $V(q)$'),
    [b2row('bet'), b2row('tlv')], size='scriptsize') + items(
    T('BET: R/S at the edge of its band, DFA above it, GPH $\\hat d$ about @{b2.bet.z} standard errors from 0, Lo rejects ($@{b2.bet.V} > 1.862$): persistence',
      'BET: R/S la limita benzii, DFA peste ea, $\\hat d$ GPH la circa @{b2.bet.z} erori standard de 0, testul lui Lo respinge ($@{b2.bet.V} > 1.862$): persistență'),
    T('TLV: every estimate inside its band; Lo does not reject', 'TLV: fiecare estimare se află în banda ei; testul lui Lo nu respinge'),
    T('Interpretation: thin trading and slow price adjustment in the early years of the BVB (Chapter 7, variance-ratio tests); the rolling exponents of the lecture show the BET moving towards 0.5',
      'Interpretare: tranzacționarea redusă și ajustarea lentă a prețurilor în primii ani ai BVB (Capitolul 7, testele raportului varianțelor); exponenții pe ferestre mobile din curs arată că BET se apropie de 0,5')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Long Memory in S\\&P 500 Volatility [Solved]', 'B3: memoria lungă a volatilității S\\&P 500 [Rezolvat]'),
       T('do the absolute and the squared S\\&P 500 returns have long memory?', 'au randamentele absolute și pătratele randamentelor S\\&P 500 memorie lungă?'),
       T('S\\&P 500 daily log returns since 2000; $|r_t|$ and $r_t^2$', 'randamentele logaritmice zilnice S\\&P 500 din 2000; $|r_t|$ și $r_t^2$'),
       [T('Estimate $d$ with GPH for $m = \\lfloor T^{0.5}\\rfloor$ and $m = \\lfloor T^{0.6}\\rfloor$, and $H$ with DFA, for $|r_t|$ and $r_t^2$.', 'Estimați $d$ cu GPH pentru $m = \\lfloor T^{0.5}\\rfloor$ și $m = \\lfloor T^{0.6}\\rfloor$ și $H$ cu DFA, pentru $|r_t|$ și $r_t^2$.'),
        T('Report the ACF of $|r_t|$ at lags 1, 50 and 250 and the 95\\% band $\\pm 1.96/\\sqrt{T}$.', 'Raportați ACF a lui $|r_t|$ la decalajele 1, 50 și 250 și banda de 95\\% $\\pm 1.96/\\sqrt{T}$.'),
        T('Shuffle $|r_t|$ at random and repeat the estimates.', 'Permutați aleator $|r_t|$ și repetați estimările.'),
        T('Draw the ACF of $|r_t|$ before and after the shuffle.', 'Desenați ACF a lui $|r_t|$ înainte și după permutare.'),
        T('Interpretation: does long memory in $|r_t|$ contradict the weak form of the EMH?', 'Interpretare: contrazice memoria lungă a lui $|r_t|$ forma slabă a EMH?')],
       T('a table, the chart and two sentences', 'un tabel, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch11_sem_b3', h='0.28') + table(
    'lrrrrrr', T(' & GPH $T^{0.5}$ & GPH $T^{0.6}$ & DFA & $\\rho(1)$ & $\\rho(50)$ & $\\rho(250)$', ' & GPH $T^{0.5}$ & GPH $T^{0.6}$ & DFA & $\\rho(1)$ & $\\rho(50)$ & $\\rho(250)$'),
    ['$|r_t|$ & $@{b3.a.g5}$ & $@{b3.a.g6}$ & $@{b3.a.dfa}$ & $@{b3.a.acf1}$ & $@{b3.a.acf50}$ & $@{b3.a.acf250}$',
     '$r_t^2$ & $@{b3.s.g5}$ & $@{b3.s.g6}$ & $@{b3.s.dfa}$ & $@{b3.s.acf1}$ & $@{b3.s.acf50}$ & $@{b3.s.acf250}$',
     T('$|r_t|$ shuffled', '$|r_t|$ permutat') + ' & $@{b3.sh.g5}$ & $@{b3.sh.g6}$ & $@{b3.sh.dfa}$ & $@{b3.sh.acf1}$ & $@{b3.sh.acf50}$ & $@{b3.sh.acf250}$'],
    size='scriptsize') + items(
    T('$m = @{b3.a.m5}$ and @{b3.a.m6}; SE $@{b3.a.se5}$ and $@{b3.a.se6}$; ACF band $\\pm @{b3.band}$; after the shuffle all estimates fall to 0 ($d$) or 0.5 ($H$)',
      '$m = @{b3.a.m5}$ și @{b3.a.m6}; SE $@{b3.a.se5}$ și $@{b3.a.se6}$; banda ACF $\\pm @{b3.band}$; după permutare, toate estimările scad la 0 ($d$) sau la 0,5 ($H$)'),
    T('Interpretation: no: the weak form concerns the direction of returns (B1); long memory in $|r_t|$ means predictable risk, which needs a volatility model (Chapter 9)',
      'Interpretare: nu: forma slabă privește direcția randamentelor (B1); memoria lungă a lui $|r_t|$ înseamnă risc previzibil, care cere un model de volatilitate (Capitolul 9)')) + qlsem(), 'scriptsize')

D.task(T('B4: Bitcoin Returns and Volatility [Proposed]', 'B4: randamentele și volatilitatea Bitcoin [Propus]'),
       T('does Bitcoin show long memory in returns, in volatility, or in both? Models: B1, B3.', 'arată Bitcoin memorie lungă în randamente, în volatilitate sau în ambele? Modele: B1, B3.'),
       T('Bitcoin daily log returns since 2014 (7 days a week)', 'randamentele logaritmice zilnice Bitcoin din 2014 (7 zile pe săptămînă)'),
       [T('Estimate R/S, DFA, GPH and Lo\'s $V(q)$ for the returns, with the Monte Carlo bands.', 'Estimați R/S, DFA, GPH și $V(q)$ al lui Lo pentru randamente, cu benzile Monte Carlo.'),
        T('Estimate GPH ($T^{0.5}$, $T^{0.6}$) and DFA for $|r_t|$ and $r_t^2$.', 'Estimați GPH ($T^{0.5}$, $T^{0.6}$) și DFA pentru $|r_t|$ și $r_t^2$.'),
        T('Repeat the volatility estimates after a random shuffle of $|r_t|$.', 'Repetați estimările volatilității după o permutare aleatoare a lui $|r_t|$.'),
        T('Interpretation: is Bitcoin closer to the S\\&P 500 or to the BET?', 'Interpretare: este Bitcoin mai aproape de S\\&P 500 sau de BET?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrr', T('Returns & R/S & DFA & GPH $\\hat d$ (SE) & Lo $V(q)$', 'Randamente & R/S & DFA & GPH $\\hat d$ (SE) & Lo $V(q)$'),
    [T('estimate', 'estimare') + ' & $@{b4r.rs}$ & $@{b4r.dfa}$ & $@{b4r.d}$ ($@{b4r.dse}$) & $@{b4r.V}$',
     T('95\\% band', 'banda de 95\\%') + ' & $[@{b4r.rs.lo}, @{b4r.rs.hi}]$ & $[@{b4r.dfa.lo}, @{b4r.dfa.hi}]$ & $[@{b4r.gph.lo}, @{b4r.gph.hi}]$ ($\\hat H$) & $[0.809, 1.862]$'],
    size='scriptsize') + table(
    'lrrr', T('Volatility & GPH $T^{0.5}$ & GPH $T^{0.6}$ & DFA', 'Volatilitate & GPH $T^{0.5}$ & GPH $T^{0.6}$ & DFA'),
    ['$|r_t|$ & $@{b4.a.g5}$ & $@{b4.a.g6}$ & $@{b4.a.dfa}$', '$r_t^2$ & $@{b4.s.g5}$ & $@{b4.s.g6}$ & $@{b4.s.dfa}$',
     T('$|r_t|$ shuffled', '$|r_t|$ permutat') + ' & $@{b4.sh.g5}$ & $@{b4.sh.g6}$ & $@{b4.sh.dfa}$'], size='scriptsize') + items(
    T('Returns: R/S and DFA slightly above their bands, GPH and Lo do not reject: weak evidence, sensitive to the estimator',
      'Randamente: R/S și DFA puțin peste benzile lor, GPH și testul lui Lo nu resping: dovezi slabe, sensibile la estimator'),
    T('Volatility: long memory ($\\hat d$ around $@{b4.a.g5}$), weaker than for the S\\&P 500 (B3); gone after the shuffle',
      'Volatilitate: memorie lungă ($\\hat d$ în jur de $@{b4.a.g5}$), mai slabă decît la S\\&P 500 (B3); dispare după permutare'),
    T('Interpretation: between the two: closer to the S\\&P 500 in returns, with weaker volatility memory; Lecture 11 shows the rolling estimates', 'Interpretare: între cele două: mai aproape de S\\&P 500 la randamente, cu o memorie a volatilității mai slabă; Cursul 11 arată estimările pe ferestre mobile')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B5: Rolling Hurst Exponents of the S\\&P 500 [Solved]', 'B5: exponenți Hurst pe ferestre mobile pentru S\\&P 500 [Rezolvat]'),
       T('has the memory of S\\&P 500 returns changed since 2000?', 's-a schimbat memoria randamentelor S\\&P 500 din 2000 încoace?'),
       T('S\\&P 500 daily log returns since 2000', 'randamentele logaritmice zilnice S\\&P 500 din 2000'),
       [T('Estimate the DFA exponent on windows of 1000 days moved by 21 days, dated at the last day of each window.', 'Estimați exponentul DFA pe ferestre de 1000 de zile mutate cu 21 de zile, datate la ultima zi a fiecărei ferestre.'),
        T('Compute the 95\\% Monte Carlo band for $N = 1000$.', 'Calculați banda Monte Carlo de 95\\% pentru $N = 1000$.'),
        T('Report the minimum and the maximum with their dates, and the share of windows below and above the band.', 'Raportați minimul și maximul, cu datele lor, precum și ponderea ferestrelor sub bandă și peste bandă.'),
        T('Draw the rolling exponent with the band.', 'Desenați exponentul pe ferestre mobile, cu banda.'),
        T('Interpretation: does a window outside the band prove that the market was inefficient at that time?', 'Interpretare: dovedește o fereastră din afara benzii că piața era ineficientă în acel moment?')],
       T('six numbers, the chart and two sentences', 'șase valori, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch11_sem_b5', h='0.34') + items(
    T('@{b5.nw} windows from @{b5.e0}; band $[@{b5.q025}, @{b5.q975}]$; mean $\\hat H = @{b5.mean}$', '@{b5.nw} ferestre din @{b5.e0}; banda $[@{b5.q025}, @{b5.q975}]$; $\\hat H$ mediu $= @{b5.mean}$'),
    T('Minimum $@{b5.min}$ (@{b5.dmin}), maximum $@{b5.max}$ (@{b5.dmax}); @{b5.below}\\% of the windows below the band, @{b5.above}\\% above',
      'Minimul $@{b5.min}$ (@{b5.dmin}), maximul $@{b5.max}$ (@{b5.dmax}); @{b5.below}\\% dintre ferestre sub bandă, @{b5.above}\\% peste'),
    T('Interpretation: no: by chance about @{b5.fa} of @{b5.nw} windows fall outside, and consecutive windows overlap; only long runs count; here the runs are below the band (anti-persistence after 2008 and in 2016), never above',
      'Interpretare: nu: din întîmplare, circa @{b5.fa} dintre @{b5.nw} de ferestre ies din bandă, iar ferestrele consecutive se suprapun; contează doar perioadele lungi; aici perioadele sînt sub bandă (antipersistență după 2008 și în 2016), niciodată peste')) + qlsem(), 'scriptsize')

D.task(T('B6: Rolling Exponents of the DAX and of Banca Transilvania [Proposed]', 'B6: exponenți pe ferestre mobile pentru DAX și Banca Transilvania [Propus]'),
       T('does the result of B5 hold for the DAX and for Banca Transilvania? Model: B5.', 'se confirmă rezultatul din B5 pentru DAX și Banca Transilvania? Model: B5.'),
       T('DAX since 2000, TLV since 2010; daily log returns in \\%', 'DAX din 2000, TLV din 2010; randamente logaritmice zilnice în \\%'),
       [T('Estimate the rolling DFA exponent as in B5.', 'Estimați exponentul DFA pe ferestre mobile ca în B5.'),
        T('Report the share of windows below and above the band, and the minimum and maximum with their dates.', 'Raportați ponderea ferestrelor sub bandă și peste bandă, precum și minimul și maximul, cu datele lor.'),
        T('Compare the mean exponent in the first and in the last third of the windows.', 'Comparați exponentul mediu din prima și din ultima treime a ferestrelor.'),
        T('Interpretation: which market looks more stable over time?', 'Interpretare: care piață pare mai stabilă în timp?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return T(*[(f"{SHORT[k]} & @{{b6.{k}.nw}} & @{{b6.{k}.below}}\\% & @{{b6.{k}.above}}\\% & $@{{b6.{k}.min}}$ (@{{b6.{k}.dmin}}) & $@{{b6.{k}.max}}$ (@{{b6.{k}.dmax}}) & "
            f"$@{{b6.{k}.first_third}}$ / $@{{b6.{k}.last_third}}$")] * 2)


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrr', T(' & windows & below & above & minimum & maximum & first / last third', ' & ferestre & sub & peste & minimul & maximul & prima / ultima treime'),
    [b6row('dax'), b6row('tlv')], size='scriptsize') + items(
    T('Band for $N = 1000$: $[@{b6.dax.q025}, @{b6.dax.q975}]$', 'Banda pentru $N = 1000$: $[@{b6.dax.q025}, @{b6.dax.q975}]$'),
    T('DAX: a few windows outside on both sides, around the 2009 and 2021 extremes; TLV: almost always inside', 'DAX: cîteva ferestre în afara benzii, de ambele părți, în jurul extremelor din 2009 și 2021; TLV: aproape mereu în interior'),
    T('Interpretation: both are close to 0.5 most of the time; neither shows a lasting departure, unlike the BET in its early years', 'Interpretare: ambele sînt aproape de 0,5 în cea mai mare parte a timpului; niciuna nu arată o abatere durabilă, spre deosebire de BET în primii ani')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Does Market Memory Change in Crises? [Proposed]', 'C1: se schimbă memoria pieței în crize? [Propus]'),
       T('does the Hurst exponent of returns or of volatility change in a crisis, as the fractal market hypothesis suggests?', 'se schimbă exponentul Hurst al randamentelor sau al volatilității într-o criză, așa cum sugerează ipoteza pieței fractale?'),
       T('S\\&P 500 and BET daily log returns; a calm year (2017) and two crisis years (September 2008--August 2009, February 2020--February 2021); models: B1, B3',
         'randamentele logaritmice zilnice S\\&P 500 și BET; un an liniștit (2017) și doi ani de criză (septembrie 2008--august 2009, februarie 2020--februarie 2021); modele: B1, B3'),
       [T('Estimate the DFA exponent of $r_t$ and of $|r_t|$ and the annualised volatility in each of the three years, for both indices.', 'Estimați exponentul DFA pentru $r_t$ și $|r_t|$ și volatilitatea anualizată în fiecare dintre cei trei ani, pentru ambii indici.'),
        T('Compute the Monte Carlo band for windows of about 250 days.', 'Calculați banda Monte Carlo pentru ferestre de circa 250 de zile.'),
        T('List three reasons why one crisis year cannot settle the question.', 'Enumerați trei motive pentru care un singur an de criză nu poate lămuri întrebarea.'),
        T('Interpretation: which design would separate a change of memory from a change of volatility?', 'Interpretare: ce abordare ar separa o schimbare a memoriei de o schimbare a volatilității?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

c1rows = []
for k in ['sp500', 'bet']:
    for kk, lab in [('calm', T('calm 2017', 'an liniștit 2017')), ('gfc', T('crisis 2008--2009', 'criza 2008--2009')), ('covid', T('crisis 2020--2021', 'criza 2020--2021'))]:
        c1rows.append(f"{SHORT[k]} & {lab} & @{{c1.{k}.{kk}.n}} & $@{{c1.{k}.{kk}.r}}$ & $@{{c1.{k}.{kk}.a}}$ & @{{c1.{k}.{kk}.v}}\\%")
D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'llrrrr', T('Index & period & $N$ & DFA $r_t$ & DFA $|r_t|$ & volatility', 'Indice & perioada & $N$ & DFA $r_t$ & DFA $|r_t|$ & volatilitate'),
    c1rows, size='scriptsize') + items(
    T('Band for $N = 250$: $[@{c1.lo}, @{c1.hi}]$; only the 2020--2021 returns of both indices lie above it', 'Banda pentru $N = 250$: $[@{c1.lo}, @{c1.hi}]$; doar randamentele din 2020--2021 ale ambilor indici sînt peste ea'),
    T('Reasons: one year gives a wide band; the volatility level changes the estimate of $|r_t|$; the rebound after March 2020 is a trend inside the window',
      'Motive: un singur an dă o bandă largă; nivelul volatilității schimbă estimarea pentru $|r_t|$; revenirea de după martie 2020 este o tendință în interiorul ferestrei'),
    T('Design: standardise $r_t$ by a GARCH volatility (Chapter 9) before estimating; use many crises and markets, event windows, and a GARCH-based band; compare with the wavelet approach of \\refKrisB',
      'Abordare: standardizăm $r_t$ cu o volatilitate GARCH (Capitolul 9) înainte de estimare; folosim multe crize și piețe, ferestre de eveniment și o bandă pe baza unui GARCH; comparăm cu abordarea wavelet din \\refKrisB')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to interpret long-memory estimates for the S\\&P 500 (daily, since 2000). The answer:',
      'Un student a cerut unui asistent AI să interpreteze estimările memoriei lungi pentru S\\&P 500 (date zilnice, din 2000). Răspunsul:'),
    T('\\aiprompt{(a) R/S gives H = @{c2.rs} for the returns: above 0.5, so returns have long memory and trends can be exploited.}', '\\aiprompt{(a) R/S dă H = @{c2.rs} pentru randamente: peste 0,5, deci randamentele au memorie lungă și tendințele pot fi exploatate.}'),
    T('\\aiprompt{(b) H = @{c2.rs} means a @{c2.rsp}\\% probability that tomorrow has the same sign as today.}', '\\aiprompt{(b) H = @{c2.rs} înseamnă o probabilitate de @{c2.rsp}\\% ca mîine să aibă același semn ca azi.}'),
    T('\\aiprompt{(c) GPH gives d = @{c2.d} for |r|, so H = d - 0.5 = @{c2.Hw}: absolute returns are anti-persistent.}', '\\aiprompt{(c) GPH dă d = @{c2.d} pentru |r|, deci H = d - 0,5 = @{c2.Hw}: randamentele absolute sînt antipersistente.}'),
    T('\\aiprompt{(d) Long memory in |r| contradicts the weak form of the EMH.}', '\\aiprompt{(d) Memoria lungă a lui |r| contrazice forma slabă a EMH.}'),
    T("\\aiprompt{(e) Lo's modified statistic V = @{c2.V} lies inside [0.809, 1.862], so short memory is not rejected at 5\\%.}", '\\aiprompt{(e) Statistica modificată a lui Lo, V = @{c2.V}, se află în [0,809; 1,862], deci memoria scurtă nu este respinsă la 5\\%.}'),
    T('\\aiprompt{(f) For more precision, use all T/2 = @{c2.mfull} frequencies in the GPH regression.}', '\\aiprompt{(f) Pentru mai multă precizie, folosiți toate cele T/2 = @{c2.mfull} de frecvențe în regresia GPH.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: i.i.d.\\ series of this length give R/S $@{c2.rsm}$ on average, band up to $@{c2.rshi}$; DFA gives $@{c2.dfa}$: no evidence of long memory (B1)',
      '(a) Greșit: seriile i.i.d.\\ de această lungime dau în medie R/S $@{c2.rsm}$, cu banda pînă la $@{c2.rshi}$; DFA dă $@{c2.dfa}$: nicio dovadă de memorie lungă (B1)'),
    T('(b) Wrong: $H$ is a scaling exponent, not a probability', '(b) Greșit: $H$ este un exponent de scalare, nu o probabilitate'),
    T('(c) Wrong: $H = d + 0.5 = @{c2.H}$: strong persistence of volatility', '(c) Greșit: $H = d + 0.5 = @{c2.H}$: persistență puternică a volatilității'),
    T('(d) Wrong: the weak form concerns the direction of returns; predictable volatility is compatible with it (B3, Chapter 9)', '(d) Greșit: forma slabă privește direcția randamentelor; volatilitatea previzibilă este compatibilă cu ea (B3, Capitolul 9)'),
    T('(e) Correct', '(e) Corect'),
    T('(f) Wrong: high frequencies carry the short-run dynamics and bias $\\hat d$; GPH needs $m/T \\to 0$; report $m = T^{0.5}$ and $T^{0.6}$', '(f) Greșit: frecvențele înalte poartă dinamica de termen scurt și deplasează $\\hat d$; GPH are nevoie de $m/T \\to 0$; raportăm $m = T^{0.5}$ și $T^{0.6}$')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('R/S, DFA and GPH estimate the same $H$ in different ways; report several, each with a Monte Carlo band for the same $N$',
      'R/S, DFA și GPH estimează același $H$ în moduri diferite; raportăm mai mulți estimatori, fiecare cu o bandă Monte Carlo pentru același $N$'),
    T('$d = H - 0.5$; $H$ is a scaling exponent, not a probability', '$d = H - 0.5$; $H$ este un exponent de scalare, nu o probabilitate'),
    T('Returns of large markets: no long memory; volatility: long memory everywhere; the shuffle test locates it in the order of the days',
      'Randamentele piețelor mari: fără memorie lungă; volatilitatea: memorie lungă peste tot; testul permutării o localizează în ordinea zilelor'),
    T('Rolling estimates: read long runs outside the band, not single windows', 'Estimările pe ferestre mobile: interpretăm perioadele lungi din afara benzii, nu ferestrele izolate'),
    T('An AI answer is a draft: check the band, the conversion between $d$ and $H$, and the bandwidth', 'Un răspuns AI este o ciornă: verificați banda, conversia dintre $d$ și $H$ și lățimea de bandă')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 11 develops each topic of today: the FMH, self-similarity, fBm and ARFIMA, the estimators, spurious memory, rolling exponents, risk scaling',
      'Cursul 11 dezvoltă fiecare temă de azi: FMH, autosimilaritatea, fBm și ARFIMA, estimatorii, memoria aparentă, exponenții pe ferestre mobile, scalarea riscului'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: GARCH-standardised returns, many crises, a GARCH-based band', 'C1 poate deveni un proiect de echipă: randamente standardizate cu GARCH, multe crize, o bandă pe baza unui GARCH'),
    T('Reading: \\refFHH, Ch.~14; \\refLo; \\refWeron', 'Lectură: \\refFHH, cap.~14; \\refLo; \\refWeron')))

D.references(bib(['AL', 'DGE', 'DI', 'FHH', 'GPH', 'Hosking', 'Hurst', 'KrisB', 'Lo', 'MVN', 'Peng', 'Peters', 'Weron']), per=16)

if __name__ == '__main__':
    D.write(V)
