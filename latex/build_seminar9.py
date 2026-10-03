r"""
build_seminar9.py -- Seminarul 9 (Modele ARCH și GARCH), EN + RO dintr-o singură sursă
====================================================================================
Seminarul are loc ÎNAINTEA cursului 9: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_09/sem9_results.json (seminar9.py).
Ieșire:
  EN/Seminars/seminar9_arch_garch_models.tex          (+ _solutions.tex)
  RO/Seminarii/seminar9_modele_arch_garch_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_09/seminar9.py && python3 latex/build_seminar9.py && python3 latex/sfm_build.py compile 9
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch9_common import IG, NAMES, REFS, SHORT, T, bib, load_sem, pv   # noqa: E402

S = load_sem()
V = Values()
D = Deck(9, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_09}{SFM_ch9_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    d = A[a]
    V.put(f'{a}.pers', d['pers'], 2)
    V.put(f'{a}.uv', d['uv'], 3)
    V.put(f'{a}.vol', d['vol_lr'], 2)
    V.put(f'{a}.hl', d['hl'], 1)
    V.put(f'{a}.lnp', math.log(d['pers']), 5)
    V.put(f'{a}.s2', d['s2_next'], 3)
    V.put(f'{a}.s', math.sqrt(d['s2_next']), 3)
for a in ['a3', 'a4', 'a4n']:
    d = A[a]
    V.put(f'{a}.s2bar', d['s2bar'], 3)
    for h, v in d['f'].items():
        V.put(f'{a}.f{h}', v, 3)
    V.put(f'{a}.sum', d['sumH'], 2)
    V.put(f'{a}.vol', d['volH'], 2)
    V.put(f'{a}.volsq', d['volH_sqrt'], 2)
    V.put(f'{a}.vollr', d['volH_lr'], 2)
    V.put(f'{a}.q', d['q'], 3)
    V.put(f'{a}.qa', -d['q'], 3)
    V.put(f'{a}.v1', d['var1'], 2)
    V.put(f'{a}.vH', d['varH'], 2)
    V.put(f'{a}.fac', d['factor'], 3)
V.put('a3.p4', 0.98 ** 4, 4)
V.put('a3.p9', 0.98 ** 9, 4)
V.put('a3.p10', 0.98 ** 10, 4)
V.put('a4.p9', 0.97 ** 9, 4)
V.put('a4.inc', 100 * (A['a4']['var1'] / A['a4n']['var1'] - 1), 0)
V.put('a1.inc', 100 * (A['a1']['s2_next'] / 1.5 - 1), 0)
V.put('a1.wk', A['a1']['hl'] / 5, 0)
d = A['a5']
for i, (s2, lt) in enumerate(zip(d['s2'], d['lt']), 2):
    V.put(f'a5.s{i}', s2, 3)
    V.put(f'a5.l{i}', lt, 3)
V.put('a5.ll', d['loglik'], 3)
V.put('a5.k', d['kurt'], 1)
V.put('a5.uv', d['uv'], 1)
d = A['a6']
V.put('a6.neg', d['gjr']['-2.0'], 2)
V.put('a6.pos', d['gjr']['2.0'], 2)
V.put('a6.ratio', d['ratio'], 2)
V.put('a6.pers', d['pers'], 2)
V.put('a6.hl', d['hl'], 1)
V.put('a6.egneg.ln', d['eg']['-2.0']['ln'], 4)
V.put('a6.egneg', d['eg']['-2.0']['s2'], 3)
V.put('a6.egpos.ln', d['eg']['2.0']['ln'], 4)
V.put('a6.egpos', d['eg']['2.0']['s2'], 3)
V.put('a6.ez', math.sqrt(2 / math.pi), 4)


def put_b1(key, d):
    p, se = d['params'], d['se']
    for a, b in [('mu', 'mu'), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]'), ('nu', 'nu')]:
        dd = 2 if a == 'nu' else (4 if a == 'om' else 3)
        V.put(f'{key}.{a}', p[b], dd)
        V.put(f'{key}.{a}.se', se[b], dd)
    V.int(f'{key}.n', d['n'])
    V.raw(f'{key}.y0', d['first'][:4])
    V.put(f'{key}.vs', d['vol_sample'], 1)
    V.put(f'{key}.last', d['last_vol'], 1)
    V.put(f'{key}.out', 100 * d['share_out'], 1)
    V.put(f'{key}.ab', p['alpha[1]'] + p['beta[1]'], 4)
    if d['pers'] >= IG:
        V.raw(f'{key}.pers', '⁅1.000⁆')
        V.raw(f'{key}.hl', '--')
        V.raw(f'{key}.vlr', '--')
    else:
        V.put(f'{key}.pers', d['pers'], 3)
        V.put(f'{key}.hl', d['hl'], 1)
        V.put(f'{key}.vlr', d['vol_lr'], 1)
        V.put(f'{key}.uv', d['uv'], 3)
        V.put(f'{key}.1mp', 1 - d['pers'], 4)
    for e in ['2008', '2020']:
        if f'peak{e}' in d:
            V.put(f'{key}.p{e}', d[f'peak{e}'], 0)
        else:
            V.raw(f'{key}.p{e}', '--')


put_b1('b1', S['B1'])
V.put('b1.lnp', math.log(S['B1']['pers']), 5)
for k, d in S['B2'].items():
    put_b1(f'b2.{k}', d)


def put_b3(key, d):
    V.put(f'{key}.g', d['gamma'], 3)
    V.put(f'{key}.gt', d['t_gamma'], 2)
    V.put(f'{key}.a', d['alpha_gjr'], 3)
    V.put(f'{key}.b', d['beta_gjr'], 3)
    V.put(f'{key}.lr', d['lr'], 1)
    V.raw(f'{key}.lrp', pv(d['p_lr']))
    V.put(f'{key}.bg', d['bic_garch'], 1)
    V.put(f'{key}.bj', d['bic_gjr'], 1)
    V.put(f'{key}.db', d['bic_gjr'] - d['bic_garch'], 1)
    V.put(f'{key}.sb', d['sb']['joint'], 1)
    V.raw(f'{key}.sbp', pv(d['sb']['joint_p']))
    V.put(f'{key}.sbt', d['sb']['sign_t'], 2)
    V.put(f'{key}.neg', d['nic_neg'], 3)
    V.put(f'{key}.pos', d['nic_pos'], 3)
    V.put(f'{key}.ratio', d['ratio'], 2)
    V.put(f'{key}.pj', d['pers_gjr'], 3)
    V.put(f'{key}.pg', d['pers_garch'], 3)
    if 'eg_gamma' in d:
        V.put(f'{key}.eg', d['eg_gamma'], 3)
        V.put(f'{key}.egt', d['eg_t'], 2)


put_b3('b3', S['B3'])
for k, d in S['B4'].items():
    put_b3(f'b4.{k}', d)


def put_b5(key, d):
    V.int(f'{key}.n', d['n'])
    V.put(f'{key}.qg', d['q_garch'], 3)
    V.put(f'{key}.qe', d['q_ewma'], 3)
    V.put(f'{key}.dm', d['dm']['t'], 2)
    V.put(f'{key}.dmm', d['dm']['mean'], 3)
    V.raw(f'{key}.dmp', pv(d['dm']['p']))
    V.raw(f'{key}.ng', str(d['nexc_garch']))
    V.raw(f'{key}.ne', str(d['nexc_ewma']))
    V.put(f'{key}.eg', 100 * d['exc_garch'], 2)
    V.put(f'{key}.ee', 100 * d['exc_ewma'], 2)
    V.raw(f'{key}.exp', str(round(d['expected'])))


put_b5('b5', S['B5'])
for k, d in S['B6'].items():
    put_b5(f'b6.{k}', d)
c1rows = []
for r in S['C1']:
    b, s_ = r['btc'], r['sp500']
    bp = '⁅1.000⁆' if b['pers'] >= IG else f'⁅{b["pers"]:.3f}⁆'
    c1rows.append(f'{r["from"][:4]}--{r["to"][:4]} & {bp} & $⁅{b["gamma"]:.3f}⁆$ ($⁅{b["t_gamma"]:.1f}⁆$) & $⁅{b["nu"]:.2f}⁆$ & '
                  f'$⁅{s_["pers"]:.3f}⁆$ & $⁅{s_["gamma"]:.3f}⁆$ ($⁅{s_["t_gamma"]:.1f}⁆$) & $⁅{r["corr_vol"]:.2f}⁆$')
C1 = S['C1']
V.put('c1.c0', C1[0]['corr_vol'], 2)
V.put('c1.c1', C1[1]['corr_vol'], 2)
V.put('c1.g0', C1[0]['btc']['gamma'], 3)
V.put('c1.g0t', C1[0]['btc']['t_gamma'], 1)
V.put('c1.g1', C1[1]['btc']['gamma'], 3)
V.put('c1.g1t', C1[1]['btc']['t_gamma'], 1)
V.put('c1.p1', C1[1]['btc']['pers'], 4)
V.put('c1.v0', C1[0]['btc']['vol_sample'], 1)
V.put('c1.v1', C1[1]['btc']['vol_sample'], 1)
C2 = S['C2']
V.put('c2.a', C2['alpha'], 3)
V.put('c2.b', C2['beta'], 3)
V.put('c2.om', C2['omega'], 4)
V.put('c2.pers', C2['pers'], 3)
V.put('c2.hl', C2['hl'], 0)
V.put('c2.uvw', C2['uv_wrong'], 3)
V.put('c2.volw', C2['vol_wrong'], 1)
V.put('c2.uv', C2['uv'], 2)
V.put('c2.vol', C2['vol_lr'], 1)
V.put('c2.nu', C2['nu'], 2)
V.put('c2.qt', -C2['q_t'], 3)
V.put('c2.qn', -C2['q_n'], 3)
V.put('c2.lb', C2['lb_z2'][0], 1)
V.put('c2.lbp', C2['lb_z2'][1], 2)
V.put('c2.g', C2['gamma'], 3)
V.put('c2.sb', S['B3']['sb']['joint'], 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how can we model, estimate and forecast a volatility that changes every day?',
       '\\textbf{Întrebarea}: cum modelăm, estimăm și prognozăm o volatilitate care se schimbă în fiecare zi?'),
     [T('this seminar comes \\textbf{before} Lecture 9: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 9: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: GARCH algebra, forecasts and VaR, the ARCH likelihood, asymmetric news impact, on paper',
        'Partea A: algebra GARCH, prognoze și VaR, verosimilitatea ARCH, impactul asimetric al știrilor, pe hîrtie'),
      T('Part B: GARCH estimation, asymmetry and out-of-sample forecasts on the S\\&P 500, DAX, BET, Bitcoin and BVB stocks, each with an interpretation question',
        'Partea B: estimarea GARCH, asimetria și prognozele în afara eșantionului pe S\\&P 500, DAX, BET, Bitcoin și acțiuni BVB, fiecare cu o întrebare de interpretare'),
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
    ['A1, A2 & ' + T('persistence, long-run variance, half-life and one GARCH(1,1) step', 'persistența, varianța de lungă durată, timpul de înjumătățire și un pas GARCH(1,1)') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('multi-step forecasts, 10-day volatility and VaR 1\\%', 'prognoze pe mai mulți pași, volatilitatea pe 10 zile și VaR 1\\%') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('the ARCH(1) likelihood and kurtosis; news impact in GJR-GARCH and EGARCH', 'verosimilitatea și boltirea ARCH(1); impactul știrilor în GJR-GARCH și EGARCH') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('GARCH(1,1)-t estimates: DAX; BET, Banca Transilvania, Bitcoin', 'estimări GARCH(1,1)-t: DAX; BET, Banca Transilvania, Bitcoin') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('asymmetry and the leverage effect: S\\&P 500; BET, Bitcoin, TLV', 'asimetria și efectul de levier: S\\&P 500; BET, Bitcoin, TLV') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('GARCH against EWMA out of sample, VaR 1\\% exceedances: S\\&P 500; DAX, BET', 'GARCH față de EWMA în afara eșantionului, depășirile VaR 1\\%: S\\&P 500; DAX, BET') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('is Bitcoin volatility becoming more like equity volatility? what is wrong in an AI answer?', 'devine volatilitatea Bitcoin asemănătoare cu volatilitatea acțiunilor? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
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
    T('In the notebook: \\texttt{returns(\'dax\')}, \\texttt{fit(r, dist=\'t\')} (package \\texttt{arch}); no account or key is needed',
      'În notebook: \\texttt{returns(\'dax\')}, \\texttt{fit(r, dist=\'t\')} (pachetul \\texttt{arch}); nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Conditional Variance, ARCH and GARCH', 'Noțiuni necesare azi (1/5): varianța condiționată, ARCH și GARCH'), items(
    (T('$r_t = \\mu + \\varepsilon_t$, $\\varepsilon_t = \\sigma_t z_t$; $z_t$ i.i.d., mean 0, variance 1 (the \\textbf{innovations})',
       '$r_t = \\mu + \\varepsilon_t$, $\\varepsilon_t = \\sigma_t z_t$; $z_t$ i.i.d., media 0, varianța 1 (\\textbf{inovațiile})'),
     [T('$\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\text{past})$: the \\textbf{conditional variance}, known one day ahead', '$\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\text{trecut})$: \\textbf{varianța condiționată}, cunoscută cu o zi înainte'),
      T('the shocks $\\varepsilon_t$ are uncorrelated, but their squares are correlated (volatility clustering, Chapter 8)', 'șocurile $\\varepsilon_t$ sînt necorelate, dar pătratele lor sînt corelate (volatility clustering, Capitolul 8)')]),
    (T('\\textbf{ARCH(1)} \\refEngle: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$, $\\omega > 0$, $0 \\le \\alpha < 1$', '\\textbf{ARCH(1)} \\refEngle: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$, $\\omega > 0$, $0 \\le \\alpha < 1$'),
     [T('unconditional variance $\\omega/(1 - \\alpha)$; with Normal $z_t$ and $3\\alpha^2 < 1$: kurtosis $3(1 - \\alpha^2)/(1 - 3\\alpha^2)$',
        'varianța necondiționată $\\omega/(1 - \\alpha)$; cu $z_t$ Normale și $3\\alpha^2 < 1$: coeficientul de boltire $3(1 - \\alpha^2)/(1 - 3\\alpha^2)$')]),
    (T('\\textbf{GARCH(1,1)} \\refBoll: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\omega > 0$, $\\alpha, \\beta \\ge 0$',
       '\\textbf{GARCH(1,1)} \\refBoll: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\omega > 0$, $\\alpha, \\beta \\ge 0$'),
     [T('$\\alpha$: reaction to news; $\\beta$: memory; $\\alpha + \\beta < 1$: stationary', '$\\alpha$: reacția la știri; $\\beta$: memoria; $\\alpha + \\beta < 1$: staționar'),
      T('\\textbf{EWMA} (Chapter 8): $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$, a GARCH with $\\omega = 0$, $\\alpha + \\beta = 1$ (\\textbf{IGARCH})',
        '\\textbf{EWMA} (Capitolul 8): $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$, un GARCH cu $\\omega = 0$, $\\alpha + \\beta = 1$ (\\textbf{IGARCH})')])))

D.frame(T('What You Need for Today (2/5): Persistence and Forecasts', 'Noțiuni necesare azi (2/5): persistență și prognoze'), items(
    (T('\\textbf{Persistence} $\\alpha + \\beta$; \\textbf{long-run variance} $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$; annual volatility $\\sqrt{252\\,\\bar\\sigma^2}$',
       '\\textbf{Persistența} $\\alpha + \\beta$; \\textbf{varianța de lungă durată} $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$; volatilitatea anuală $\\sqrt{252\\,\\bar\\sigma^2}$'),
     [T('\\textbf{half-life}: $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$ days, the time for half of a variance shock to disappear',
        '\\textbf{timpul de înjumătățire}: $h_{1/2} = \\ln 0{,}5/\\ln(\\alpha + \\beta)$ zile, timpul în care dispare jumătate dintr-un șoc de varianță')]),
    (T('Forecasts made at the end of day $t$', 'Prognoze făcute la sfîrșitul zilei $t$'),
     [T('one day: $\\sigma_{t+1}^2 = \\omega + \\alpha(r_t - \\mu)^2 + \\beta\\sigma_t^2$', 'o zi: $\\sigma_{t+1}^2 = \\omega + \\alpha(r_t - \\mu)^2 + \\beta\\sigma_t^2$'),
      T('$h$ days: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$', '$h$ zile: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
      T('$H$-day variance: $\\sum_{h=1}^{H}E_t[\\sigma_{t+h}^2] = H\\bar\\sigma^2 + (\\sigma_{t+1}^2 - \\bar\\sigma^2)\\dfrac{1 - (\\alpha + \\beta)^H}{1 - \\alpha - \\beta}$',
        'varianța pe $H$ zile: $\\sum_{h=1}^{H}E_t[\\sigma_{t+h}^2] = H\\bar\\sigma^2 + (\\sigma_{t+1}^2 - \\bar\\sigma^2)\\dfrac{1 - (\\alpha + \\beta)^H}{1 - \\alpha - \\beta}$')]),
    (T('\\textbf{VaR 1\\%} (value at risk, Chapter 6): $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_{0.01}(z))$',
       '\\textbf{VaR 1\\%} (value at risk, valoarea expusă la risc, Capitolul 6): $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_{0{,}01}(z))$'),
     [T('Normal: $q_{0.01} = -2.326$; standardised Student-t: $q_{0.01} = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$, for example $t_5^{-1}(0.01) = -3.365$',
        'Normală: $q_{0{,}01} = -2{,}326$; Student-t standardizată: $q_{0{,}01} = t_\\nu^{-1}(0{,}01)\\sqrt{(\\nu - 2)/\\nu}$, de exemplu $t_5^{-1}(0{,}01) = -3{,}365$')])))

D.frame(T('What You Need for Today (3/5): Maximum Likelihood', 'Noțiuni necesare azi (3/5): verosimilitatea maximă'), items(
    (T('Conditional Gaussian log-likelihood, with $\\sigma_t^2$ computed recursively:', 'Log-verosimilitatea Gaussiană condiționată, cu $\\sigma_t^2$ calculat recursiv:'),
     [T('$\\ell(\\theta) = \\sum_{t=2}^{T}\\ell_t$, \\quad $\\ell_t = -\\dfrac12\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2 + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2}\\Big]$',
        '$\\ell(\\theta) = \\sum_{t=2}^{T}\\ell_t$, \\quad $\\ell_t = -\\dfrac12\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2 + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2}\\Big]$'),
      T('\\textbf{MLE}: $\\hat\\theta$ maximises $\\ell$; numerically, with $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$',
        '\\textbf{MLE}: $\\hat\\theta$ maximizează $\\ell$; numeric, cu $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$')]),
    (T('Standard errors (SE)', 'Erori standard (SE)'),
     [T('classic: from the curvature of $\\ell$; robust \\refBW: valid when $z_t$ is not Normal; \\texttt{arch} reports the robust ones',
        'clasice: din curbura lui $\\ell$; robuste \\refBW: valide cînd $z_t$ nu este Normal; \\texttt{arch} le raportează pe cele robuste')]),
    (T('Heavy-tailed innovations (Chapter 2): standardised Student-t with $\\nu$ degrees of freedom \\refBollT; skewed t \\refHansen',
       'Inovații cu cozi groase (Capitolul 2): Student-t standardizată cu $\\nu$ grade de libertate \\refBollT; t asimetrică \\refHansen'),
     [T('compare models with the \\textbf{LR} (likelihood ratio) test $2(\\ell_1 - \\ell_0) \\sim \\chi^2(r)$, $r$ = number of restrictions, and with AIC/BIC (Chapter 6)',
        'comparăm modelele prin testul \\textbf{LR} (likelihood ratio, raportul de verosimilitate) $2(\\ell_1 - \\ell_0) \\sim \\chi^2(r)$, $r$ = numărul de restricții, și prin AIC/BIC (Capitolul 6)'),
      T('$\\chi^2_{0.95}(1) = 3.84$', '$\\chi^2_{0{,}95}(1) = 3{,}84$')])))

D.frame(T('What You Need for Today (4/5): Asymmetry', 'Noțiuni necesare azi (4/5): asimetria'), items(
    T('\\textbf{Leverage effect}: falls raise volatility more than rises of the same size \\refChristie', '\\textbf{Efectul de levier}: scăderile cresc volatilitatea mai mult decît creșterile de aceeași mărime \\refChristie'),
    (T('\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ if $\\varepsilon_{t-1} < 0$',
       '\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ dacă $\\varepsilon_{t-1} < 0$'),
     [T('leverage effect: $\\gamma > 0$; persistence $\\alpha + \\beta + \\gamma/2$ for symmetric $z_t$', 'efect de levier: $\\gamma > 0$; persistența $\\alpha + \\beta + \\gamma/2$ pentru $z_t$ simetric')]),
    (T('\\textbf{EGARCH} \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$',
       '\\textbf{EGARCH} \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$'),
     [T('$E|z| = \\sqrt{2/\\pi} = 0.7979$ for Normal $z$; leverage effect: $\\gamma < 0$', '$E|z| = \\sqrt{2/\\pi} = 0{,}7979$ pentru $z$ Normal; efect de levier: $\\gamma < 0$')]),
    (T('\\textbf{News impact curve} \\refEN: $\\sigma_t^2$ as a function of $\\varepsilon_{t-1}$, with $\\sigma_{t-1}^2$ fixed',
       '\\textbf{Curba de impact a știrilor} \\refEN: $\\sigma_t^2$ ca funcție de $\\varepsilon_{t-1}$, cu $\\sigma_{t-1}^2$ fixat'),
     [T('symmetric parabola for GARCH; steeper on the left for GJR and EGARCH with leverage', 'parabolă simetrică pentru GARCH; mai abruptă la stînga pentru GJR și EGARCH cu efect de levier')])))

D.frame(T('What You Need for Today (5/5): Checking and Comparing Forecasts', 'Noțiuni necesare azi (5/5): verificarea și compararea prognozelor'), items(
    (T('Standardised residuals $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$: close to i.i.d.\\ if the model is right', 'Reziduurile standardizate $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$: aproape i.i.d.\\ dacă modelul este corect'),
     [T('Ljung--Box $Q(10)$ on $\\hat z_t^2$ (Chapter 7) \\refLjung: no ARCH effect left if not rejected', 'Ljung--Box $Q(10)$ pe $\\hat z_t^2$ (Capitolul 7) \\refLjung: nu a rămas niciun efect ARCH dacă nu respinge'),
      T('sign-bias test \\refEN: $\\chi^2(3)$; a rejection means a missing asymmetry', 'testul de asimetrie (sign bias) \\refEN: $\\chi^2(3)$; o respingere înseamnă că lipsește asimetria')]),
    (T('\\textbf{Out of sample}: the forecast $h_t$ for day $t$ uses only data up to $t-1$; proxy of the true variance: $r_t^2$',
       '\\textbf{În afara eșantionului}: prognoza $h_t$ pentru ziua $t$ folosește doar date pînă la $t-1$; aproximarea varianței adevărate: $r_t^2$'),
     [T('\\textbf{QLIKE} loss \\refPatton: $L_t = r_t^2/h_t + \\ln h_t$; the smaller the mean, the better the forecasts', 'pierderea \\textbf{QLIKE} \\refPatton: $L_t = r_t^2/h_t + \\ln h_t$; cu cît media este mai mică, cu atît prognozele sînt mai bune'),
      T('\\textbf{DM} (Diebold--Mariano) test \\refDM: $t$ statistic of the mean of $L_t^A - L_t^B$ with HAC standard errors; $t < -1.96$: A is better',
        'testul \\textbf{DM} (Diebold--Mariano) \\refDM: statistica $t$ a mediei lui $L_t^A - L_t^B$, cu erori standard HAC; $t < -1{,}96$: A este mai bun')]),
    T('\\textbf{VaR exceedance}: a day with $r_t < -\\mathrm{VaR}_t$; a correct VaR 1\\% is exceeded on about 1\\% of days', '\\textbf{Depășire VaR}: o zi cu $r_t < -\\mathrm{VaR}_t$; un VaR 1\\% corect este depășit în circa 1\\% dintre zile')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: GARCH(1,1) Algebra', 'A1: algebra GARCH(1,1)'),
         items(T('Daily returns in \\%: $\\omega = 0.02$, $\\alpha = 0.08$, $\\beta = 0.90$, $\\mu = 0$, 252 trading days per year.',
                 'Randamente zilnice în \\%: $\\omega = 0{,}02$, $\\alpha = 0{,}08$, $\\beta = 0{,}90$, $\\mu = 0$, 252 de zile de tranzacționare pe an.'),
               T('1. Compute the persistence and the long-run variance.', '1. Calculați persistența și varianța de lungă durată.'),
               T('2. Compute the annualised long-run volatility.', '2. Calculați volatilitatea de lungă durată anualizată.'),
               T('3. Compute the half-life of a variance shock.', '3. Calculați timpul de înjumătățire al unui șoc de varianță.'),
               T('4. Today $\\sigma_t^2 = 1.5$ and $r_t = -3\\%$; compute $\\sigma_{t+1}^2$ and $\\sigma_{t+1}$.', '4. Azi $\\sigma_t^2 = 1{,}5$ și $r_t = -3\\%$; calculați $\\sigma_{t+1}^2$ și $\\sigma_{t+1}$.'),
               T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
         items(T('1. $\\alpha + \\beta = @{a1.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{a1.pers}) = @{a1.uv}$', '1. $\\alpha + \\beta = @{a1.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{a1.pers}) = @{a1.uv}$'),
               T('2. $\\sqrt{252 \\times @{a1.uv}} = @{a1.vol}\\%$ per year', '2. $\\sqrt{252 \\times @{a1.uv}} = @{a1.vol}\\%$ pe an'),
               T('3. $\\ln 0.5/\\ln @{a1.pers} = -0.69315/(@{a1.lnp}) = @{a1.hl}$ days', '3. $\\ln 0.5/\\ln @{a1.pers} = -0.69315/(@{a1.lnp}) = @{a1.hl}$ zile'),
               T('4. $0.02 + 0.08 \\times 9 + 0.90 \\times 1.5 = @{a1.s2}$; $\\sigma_{t+1} = @{a1.s}\\%$', '4. $0.02 + 0.08 \\times 9 + 0.90 \\times 1.5 = @{a1.s2}$; $\\sigma_{t+1} = @{a1.s}\\%$'),
               T('A fall of 3\\% raises the variance by @{a1.inc}\\%; half of the excess is gone after @{a1.hl} trading days, about @{a1.wk} weeks.', 'O scădere de 3\\% crește varianța cu @{a1.inc}\\%; jumătate din excedent dispare după @{a1.hl} zile de tranzacționare, circa @{a1.wk} săptămîni.')),
         size='scriptsize')

D.proposed(T('A2: A More Reactive Market', 'A2: o piață mai reactivă'),
           items(T('$\\omega = 0.05$, $\\alpha = 0.12$, $\\beta = 0.85$, $\\mu = 0$; today $\\sigma_t^2 = 2.0$ and $r_t = +1.5\\%$. Model: A1.',
                   '$\\omega = 0{,}05$, $\\alpha = 0{,}12$, $\\beta = 0{,}85$, $\\mu = 0$; azi $\\sigma_t^2 = 2{,}0$ și $r_t = +1{,}5\\%$. Model: A1.'),
                 T('1. Compute the persistence, the long-run variance and the annualised long-run volatility.', '1. Calculați persistența, varianța de lungă durată și volatilitatea de lungă durată anualizată.'),
                 T('2. Compute the half-life.', '2. Calculați timpul de înjumătățire.'),
                 T('3. Compute $\\sigma_{t+1}^2$.', '3. Calculați $\\sigma_{t+1}^2$.'),
                 T('4. Write EWMA with $\\lambda = 0.94$ as a GARCH(1,1) and say which of the quantities in 1--2 exist for it.', '4. Scrieți EWMA cu $\\lambda = 0{,}94$ ca un GARCH(1,1) și precizați care dintre mărimile de la 1--2 există pentru el.'),
                 T('Report: four numbers and two sentences.', 'Raportați: patru valori și două fraze.')),
           items(T('1. $@{a2.pers}$; $\\bar\\sigma^2 = 0.05/0.03 = @{a2.uv}$; $\\sqrt{252 \\times @{a2.uv}} = @{a2.vol}\\%$', '1. $@{a2.pers}$; $\\bar\\sigma^2 = 0.05/0.03 = @{a2.uv}$; $\\sqrt{252 \\times @{a2.uv}} = @{a2.vol}\\%$'),
                 T('2. $\\ln 0.5/\\ln @{a2.pers} = @{a2.hl}$ days', '2. $\\ln 0.5/\\ln @{a2.pers} = @{a2.hl}$ zile'),
                 T('3. $0.05 + 0.12 \\times 2.25 + 0.85 \\times 2 = @{a2.s2}$: a rise also raises the variance; here it stays almost unchanged', '3. $0.05 + 0.12 \\times 2.25 + 0.85 \\times 2 = @{a2.s2}$: și o creștere mărește varianța; aici ea rămîne aproape neschimbată'),
                 T('4. $\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$: IGARCH; no long-run variance and no half-life (the factor is 1).', '4. $\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$: IGARCH; nu există varianță de lungă durată și nici timp de înjumătățire (factorul este 1).')),
           size='scriptsize')

D.solved(T('A3: Forecasts and VaR 1\\%', 'A3: prognoze și VaR 1\\%'),
         items(T('The model of A1, with $\\sigma_{t+1}^2 = @{a1.s2}$ and Normal innovations.', 'Modelul din A1, cu $\\sigma_{t+1}^2 = @{a1.s2}$ și inovații Normale.'),
               T('1. Compute $E_t[\\sigma_{t+5}^2]$ and $E_t[\\sigma_{t+10}^2]$.', '1. Calculați $E_t[\\sigma_{t+5}^2]$ și $E_t[\\sigma_{t+10}^2]$.'),
               T('2. Compute the 10-day variance and volatility.', '2. Calculați varianța și volatilitatea pe 10 zile.'),
               T('3. Compare with the square-root-of-time rule $\\sqrt{10}\\,\\sigma_{t+1}$ and with $\\sqrt{10\\,\\bar\\sigma^2}$.', '3. Comparați cu regula rădăcinii pătrate a timpului, $\\sqrt{10}\\,\\sigma_{t+1}$, și cu $\\sqrt{10\\,\\bar\\sigma^2}$.'),
               T('4. Compute the one-day and the 10-day VaR 1\\%.', '4. Calculați VaR 1\\% pe o zi și pe 10 zile.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. $1 + 0.98^{4}(@{a1.s2} - 1) = 1 + @{a3.p4} \\times 1.09 = @{a3.f5}$; $1 + 0.98^{9} \\times 1.09 = @{a3.f10}$', '1. $1 + 0.98^{4}(@{a1.s2} - 1) = 1 + @{a3.p4} \\times 1.09 = @{a3.f5}$; $1 + 0.98^{9} \\times 1.09 = @{a3.f10}$'),
               T('2. $10 \\times 1 + 1.09 \\times (1 - 0.98^{10})/0.02 = 10 + 1.09 \\times @{a3.fac} = @{a3.sum}$; volatility $@{a3.vol}\\%$', '2. $10 \\times 1 + 1.09 \\times (1 - 0.98^{10})/0.02 = 10 + 1.09 \\times @{a3.fac} = @{a3.sum}$; volatilitatea $@{a3.vol}\\%$'),
               T('3. $\\sqrt{10 \\times @{a1.s2}} = @{a3.volsq}\\%$ (too high), $\\sqrt{10} = @{a3.vollr}\\%$ (too low)', '3. $\\sqrt{10 \\times @{a1.s2}} = @{a3.volsq}\\%$ (prea mare), $\\sqrt{10} = @{a3.vollr}\\%$ (prea mică)'),
               T('4. $2.326 \\times @{a1.s} = @{a3.v1}\\%$; $2.326 \\times @{a3.vol} = @{a3.vH}\\%$', '4. $2.326 \\times @{a1.s} = @{a3.v1}\\%$; $2.326 \\times @{a3.vol} = @{a3.vH}\\%$'),
               T('The high variance decays slowly: the 10-day risk lies between the two simple rules, closer to the first.', 'Varianța ridicată scade lent: riscul pe 10 zile este între cele două reguli simple, mai aproape de prima.')),
         size='scriptsize')

D.proposed(T('A4: Forecasts with Student-t Innovations', 'A4: prognoze cu inovații Student-t'),
           items(T('The model of A2, with $\\sigma_{t+1}^2 = @{a2.s2}$ and standardised Student-t innovations with $\\nu = 5$ ($t_5^{-1}(0.01) = -3.365$). Model: A3.',
                   'Modelul din A2, cu $\\sigma_{t+1}^2 = @{a2.s2}$ și inovații Student-t standardizate cu $\\nu = 5$ ($t_5^{-1}(0{,}01) = -3{,}365$). Model: A3.'),
                 T('1. Compute $E_t[\\sigma_{t+10}^2]$ and the 10-day variance.', '1. Calculați $E_t[\\sigma_{t+10}^2]$ și varianța pe 10 zile.'),
                 T('2. Compute the quantile $q_{0.01}$ of the standardised t.', '2. Calculați cuantila $q_{0{,}01}$ a distribuției t standardizate.'),
                 T('3. Compute the one-day and the 10-day VaR 1\\% with t and with Normal innovations.', '3. Calculați VaR 1\\% pe o zi și pe 10 zile, cu inovații t și cu inovații Normale.'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $@{a4.s2bar} + 0.97^{9}(@{a2.s2} - @{a4.s2bar}) = @{a4.f10}$; sum $= @{a4.sum}$, volatility $@{a4.vol}\\%$', '1. $@{a4.s2bar} + 0.97^{9}(@{a2.s2} - @{a4.s2bar}) = @{a4.f10}$; suma $= @{a4.sum}$, volatilitatea $@{a4.vol}\\%$'),
                 T('2. $q = -3.365 \\times \\sqrt{3/5} = @{a4.q}$', '2. $q = -3.365 \\times \\sqrt{3/5} = @{a4.q}$'),
                 T('3. t: $@{a4.qa} \\times @{a2.s} = @{a4.v1}\\%$, 10 days $@{a4.vH}\\%$; Normal: $@{a4n.v1}\\%$ and $@{a4n.vH}\\%$', '3. t: $@{a4.qa} \\times @{a2.s} = @{a4.v1}\\%$, 10 zile $@{a4.vH}\\%$; Normală: $@{a4n.v1}\\%$ și $@{a4n.vH}\\%$'),
                 T('With the same variance, heavy tails raise VaR 1\\% by about @{a4.inc}\\%.', 'La aceeași varianță, cozile groase cresc VaR 1\\% cu circa @{a4.inc}\\%.')),
           size='scriptsize')

D.solved(T('A5: The ARCH(1) Likelihood, Step by Step', 'A5: verosimilitatea ARCH(1), pas cu pas'),
         items(T('ARCH(1) with $\\omega = 0.5$, $\\alpha = 0.5$, $\\mu = 0$, Normal $z_t$; observed: $r_1, \\dots, r_4 = 0.5;\\ -1.2;\\ 2.0;\\ -0.3$.',
                 'ARCH(1) cu $\\omega = 0{,}5$, $\\alpha = 0{,}5$, $\\mu = 0$, $z_t$ Normale; observat: $r_1, \\dots, r_4 = 0{,}5;\\ -1{,}2;\\ 2{,}0;\\ -0{,}3$.'),
               T('1. Compute $\\sigma_2^2$, $\\sigma_3^2$ and $\\sigma_4^2$.', '1. Calculați $\\sigma_2^2$, $\\sigma_3^2$ și $\\sigma_4^2$.'),
               T('2. Compute $\\ell_2$, $\\ell_3$, $\\ell_4$ and the conditional log-likelihood.', '2. Calculați $\\ell_2$, $\\ell_3$, $\\ell_4$ și log-verosimilitatea condiționată.'),
               T('3. Compute the unconditional variance and the kurtosis of this ARCH(1).', '3. Calculați varianța necondiționată și coeficientul de boltire pentru acest ARCH(1).'),
               T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
         items(T('1. $0.5 + 0.5 \\times 0.25 = @{a5.s2}$; $0.5 + 0.5 \\times 1.44 = @{a5.s3}$; $0.5 + 0.5 \\times 4 = @{a5.s4}$', '1. $0.5 + 0.5 \\times 0.25 = @{a5.s2}$; $0.5 + 0.5 \\times 1.44 = @{a5.s3}$; $0.5 + 0.5 \\times 4 = @{a5.s4}$'),
               T('2. $\\ell_2 = -\\frac12[1.8379 + \\ln @{a5.s2} + 1.44/@{a5.s2}] = @{a5.l2}$; $\\ell_3 = @{a5.l3}$; $\\ell_4 = @{a5.l4}$; sum $@{a5.ll}$',
                 '2. $\\ell_2 = -\\frac12[1.8379 + \\ln @{a5.s2} + 1.44/@{a5.s2}] = @{a5.l2}$; $\\ell_3 = @{a5.l3}$; $\\ell_4 = @{a5.l4}$; suma $@{a5.ll}$'),
               T('3. $0.5/(1 - 0.5) = @{a5.uv}$; $K = 3 \\times 0.75/0.25 = @{a5.k}$', '3. $0.5/(1 - 0.5) = @{a5.uv}$; $K = 3 \\times 0.75/0.25 = @{a5.k}$'),
               T('Day 3 ($r_3 = 2$ after a calm day) contributes least: a large return when $\\sigma_t^2$ is small is ``unlikely\'\'.', 'Ziua 3 ($r_3 = 2$ după o zi liniștită) contribuie cel mai puțin: un randament mare cînd $\\sigma_t^2$ este mic este „puțin probabil”.')),
         size='scriptsize')

D.proposed(T('A6: Good and Bad News', 'A6: știri bune și știri proaste'),
           items(T('GJR-GARCH: $\\omega = 0.02$, $\\alpha = 0.03$, $\\gamma = 0.12$, $\\beta = 0.90$; EGARCH: $\\omega = 0$, $\\alpha = 0.12$, $\\gamma = -0.08$, $\\beta = 0.98$; in both $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.',
                   'GJR-GARCH: $\\omega = 0{,}02$, $\\alpha = 0{,}03$, $\\gamma = 0{,}12$, $\\beta = 0{,}90$; EGARCH: $\\omega = 0$, $\\alpha = 0{,}12$, $\\gamma = -0{,}08$, $\\beta = 0{,}98$; în ambele $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.'),
                 T('1. Compute the GJR $\\sigma_{t+1}^2$ after $\\varepsilon_t = -2$ and after $\\varepsilon_t = +2$, and their ratio.', '1. Calculați $\\sigma_{t+1}^2$ GJR după $\\varepsilon_t = -2$ și după $\\varepsilon_t = +2$, precum și raportul lor.'),
                 T('2. Compute the GJR persistence and half-life.', '2. Calculați persistența și timpul de înjumătățire GJR.'),
                 T('3. Compute the EGARCH $\\sigma_{t+1}^2$ for the same two shocks (use $E|z| = 0.7979$).', '3. Calculați $\\sigma_{t+1}^2$ EGARCH pentru aceleași două șocuri (folosiți $E|z| = 0{,}7979$).'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $0.02 + 0.15 \\times 4 + 0.90 = @{a6.neg}$; $0.02 + 0.03 \\times 4 + 0.90 = @{a6.pos}$; ratio $@{a6.ratio}$', '1. $0.02 + 0.15 \\times 4 + 0.90 = @{a6.neg}$; $0.02 + 0.03 \\times 4 + 0.90 = @{a6.pos}$; raportul $@{a6.ratio}$'),
                 T('2. $0.03 + 0.90 + 0.06 = @{a6.pers}$; $\\ln 0.5/\\ln @{a6.pers} = @{a6.hl}$ days', '2. $0.03 + 0.90 + 0.06 = @{a6.pers}$; $\\ln 0.5/\\ln @{a6.pers} = @{a6.hl}$ zile'),
                 T('3. $z = -2$: $0.12(2 - 0.7979) + 0.16 = @{a6.egneg.ln}$, $e^{@{a6.egneg.ln}} = @{a6.egneg}$; $z = +2$: $@{a6.egpos.ln}$, $\\sigma_{t+1}^2 = @{a6.egpos}$',
                   '3. $z = -2$: $0.12(2 - 0.7979) + 0.16 = @{a6.egneg.ln}$, $e^{@{a6.egneg.ln}} = @{a6.egneg}$; $z = +2$: $@{a6.egpos.ln}$, $\\sigma_{t+1}^2 = @{a6.egpos}$'),
                 T('In EGARCH a good shock of 2 can even lower the variance: the sign term dominates.', 'În EGARCH, un șoc pozitiv de 2 poate chiar să scadă varianța: termenul de semn domină.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: GARCH(1,1)-t for the DAX [Solved]', 'B1: GARCH(1,1)-t pentru DAX [Rezolvat]'),
       T('how persistent is DAX volatility, and how far is the DAX from its long-run volatility today?', 'cît de persistentă este volatilitatea DAX și cît de departe este DAX azi de volatilitatea lui de lungă durată?'),
       T('DAX closes since 2000, daily log returns in \\%', 'închiderile DAX din 2000, randamente logaritmice zilnice în \\%'),
       [T('Estimate GARCH(1,1) with Student-t innovations and report the parameters with robust standard errors.', 'Estimați GARCH(1,1) cu inovații Student-t și raportați parametrii cu erorile standard robuste.'),
        T('Compute the persistence, the half-life and the annualised long-run volatility, and compare it with the sample volatility.', 'Calculați persistența, timpul de înjumătățire și volatilitatea de lungă durată anualizată și comparați-o cu volatilitatea de selecție.'),
        T('Report the peaks of the annualised conditional volatility in 2008 and in 2020, and its value on the last day.', 'Raportați vîrfurile volatilității condiționate anualizate din 2008 și din 2020, precum și valoarea ei în ultima zi.'),
        T('Draw the returns since 2018 with $\\pm 2\\hat\\sigma_t$ bands.', 'Desenați randamentele din 2018 cu benzile $\\pm 2\\hat\\sigma_t$.'),
        T('Interpretation: should a risk manager use the long-run volatility or the current one for tomorrow?', 'Interpretare: ar trebui un manager de risc să folosească pentru ziua de mîine volatilitatea de lungă durată sau pe cea curentă?')],
       T('a table of five parameters, five numbers, the chart and two sentences', 'un tabel cu cinci parametri, cinci valori, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch9_sem_b1', h='0.30') + table(
    'lrrrrr', '& $\\mu$ & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$',
    [T('estimate', 'estimare') + ' & $@{b1.mu}$ & $@{b1.om}$ & $@{b1.a}$ & $@{b1.b}$ & $@{b1.nu}$',
     T('robust SE', 'SE robustă') + ' & $@{b1.mu.se}$ & $@{b1.om.se}$ & $@{b1.a.se}$ & $@{b1.b.se}$ & $@{b1.nu.se}$'], size='scriptsize') + items(
    T('$T = @{b1.n}$; $\\alpha + \\beta = @{b1.pers}$, half-life $\\ln 0.5/(@{b1.lnp}) = @{b1.hl}$ days; long-run volatility @{b1.vlr}\\% against @{b1.vs}\\% in the sample',
      '$T = @{b1.n}$; $\\alpha + \\beta = @{b1.pers}$, timpul de înjumătățire $\\ln 0{,}5/(@{b1.lnp}) = @{b1.hl}$ zile; volatilitatea de lungă durată @{b1.vlr}\\%, față de @{b1.vs}\\% în eșantion'),
    T('Peaks: @{b1.p2008}\\% (2008), @{b1.p2020}\\% (2020); last day: @{b1.last}\\%; @{b1.out}\\% of the days fall outside $\\pm 2\\hat\\sigma_t$',
      'Vîrfuri: @{b1.p2008}\\% (2008), @{b1.p2020}\\% (2020); ultima zi: @{b1.last}\\%; @{b1.out}\\% dintre zile ies din banda $\\pm 2\\hat\\sigma_t$'),
    T('Interpretation: the current one: VaR for tomorrow needs $\\sigma_{t+1}$; the long-run level matters only for long horizons, and it is imprecise because $1 - \\alpha - \\beta = @{b1.1mp}$ is small',
      'Interpretare: pe cea curentă: VaR pentru mîine are nevoie de $\\sigma_{t+1}$; nivelul de lungă durată contează doar pentru orizonturi lungi și este imprecis, deoarece $1 - \\alpha - \\beta = @{b1.1mp}$ este mic')) + qlsem(), 'scriptsize')

D.task(T('B2: BET, Banca Transilvania and Bitcoin [Proposed]', 'B2: BET, Banca Transilvania și Bitcoin [Propus]'),
       T('which of the BET, Banca Transilvania and Bitcoin has the most persistent volatility? Model: B1.', 'care dintre BET, Banca Transilvania și Bitcoin are volatilitatea cea mai persistentă? Model: B1.'),
       T('BET since 2000, TLV since 2010 (without 30--31 May 2016), Bitcoin since 2014; daily log returns in \\%', 'BET din 2000, TLV din 2010 (fără 30--31 mai 2016), Bitcoin din 2014; randamente logaritmice zilnice în \\%'),
       [T('Estimate GARCH(1,1)-t for each series and report $\\alpha$, $\\beta$ and $\\nu$.', 'Estimați GARCH(1,1)-t pentru fiecare serie și raportați $\\alpha$, $\\beta$ și $\\nu$.'),
        T('Compute the persistence, the half-life and the long-run volatility where they exist.', 'Calculați persistența, timpul de înjumătățire și volatilitatea de lungă durată, acolo unde există.'),
        T('Compare the long-run with the sample volatility (annualise Bitcoin with 365 days).', 'Comparați volatilitatea de lungă durată cu volatilitatea de selecție (anualizați Bitcoin cu 365 de zile).'),
        T('Interpretation: what does an estimate $\\hat\\alpha + \\hat\\beta = 1$ tell a risk manager?', 'Interpretare: ce îi spune unui manager de risc o estimare $\\hat\\alpha + \\hat\\beta = 1$?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{SHORT[k]} & @{{b2.{k}.n}} & $@{{b2.{k}.a}}$ & $@{{b2.{k}.b}}$ & $@{{b2.{k}.nu}}$ & $@{{b2.{k}.pers}}$ & @{{b2.{k}.hl}} & '
            f'@{{b2.{k}.vlr}} & @{{b2.{k}.vs}} & @{{b2.{k}.p2020}}')


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrrrr', T('& $T$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & half-life & long-run vol. & sample vol. & peak 2020',
                    '& $T$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & înjumătățire & vol. lungă durată & vol. selecție & vîrf 2020'),
    [b2row(k) for k in ['bet', 'tlv', 'btc']], size='scriptsize') + items(
    T('BET: the strongest reaction ($\\alpha = @{b2.bet.a}$) and a half-life of @{b2.bet.hl} days; its long-run volatility (@{b2.bet.vlr}\\%) is far above the sample value: an imprecise number',
      'BET: cea mai puternică reacție ($\\alpha = @{b2.bet.a}$) și un timp de înjumătățire de @{b2.bet.hl} zile; volatilitatea lui de lungă durată (@{b2.bet.vlr}\\%) este mult peste valoarea de selecție: un număr imprecis'),
    T('TLV: shorter memory (@{b2.tlv.hl} days); long-run and sample volatility agree (@{b2.tlv.vlr}\\% and @{b2.tlv.vs}\\%)', 'TLV: memorie mai scurtă (@{b2.tlv.hl} zile); volatilitatea de lungă durată și cea de selecție sînt apropiate (@{b2.tlv.vlr}\\% și @{b2.tlv.vs}\\%)'),
    T('Bitcoin: $\\hat\\alpha + \\hat\\beta = @{b2.btc.ab}$: an IGARCH, the most persistent; $\\hat\\nu = @{b2.btc.nu}$', 'Bitcoin: $\\hat\\alpha + \\hat\\beta = @{b2.btc.ab}$: un IGARCH, cel mai persistent; $\\hat\\nu = @{b2.btc.nu}$'),
    T('Interpretation: no mean reversion: forecasts for every horizon equal today\'s variance; the model behaves like an EWMA, and a long-run volatility should not be reported',
      'Interpretare: nu există revenire la medie: prognozele pentru orice orizont sînt egale cu varianța de azi; modelul se comportă ca un EWMA, iar o volatilitate de lungă durată nu ar trebui raportată')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: The Leverage Effect in the S\\&P 500 [Solved]', 'B3: efectul de levier la S\\&P 500 [Rezolvat]'),
       T('do falls raise the volatility of the S\\&P 500 more than rises of the same size?', 'cresc scăderile volatilitatea S\\&P 500 mai mult decît creșterile de aceeași mărime?'),
       T('S\\&P 500 closes since 2000, daily log returns in \\%', 'închiderile S\\&P 500 din 2000, randamente logaritmice zilnice în \\%'),
       [T('Estimate GARCH(1,1)-t and GJR-GARCH(1,1)-t; report $\\gamma$ with its robust $t$ statistic.', 'Estimați GARCH(1,1)-t și GJR-GARCH(1,1)-t; raportați $\\gamma$ cu statistica $t$ robustă.'),
        T('Compute the LR statistic of GJR against GARCH, its p-value and the two BIC values.', 'Calculați statistica LR pentru GJR față de GARCH, p-valoarea ei și cele două valori BIC.'),
        T('Run the sign-bias test on the standardised residuals of GARCH-t.', 'Aplicați testul de asimetrie (sign bias) pe reziduurile standardizate ale GARCH-t.'),
        T('Draw both news impact curves and compute the GJR variance after shocks of $-2\\%$ and $+2\\%$.', 'Desenați ambele curbe de impact ale știrilor și calculați varianța GJR după șocuri de $-2\\%$ și $+2\\%$.'),
        T('Interpretation: how does the leverage effect change the VaR on the day after a fall?', 'Interpretare: cum schimbă efectul de levier VaR-ul din ziua de după o scădere?')],
       T('four statistics with p-values, two variances, the chart and two sentences', 'patru statistici cu p-valori, două varianțe, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch9_sem_b3', h='0.34') + items(
    T('GJR: $\\hat\\alpha = @{b3.a}$, $\\hat\\gamma = @{b3.g}$ ($t = @{b3.gt}$), $\\hat\\beta = @{b3.b}$; LR $= @{b3.lr}$ (p @{b3.lrp}); BIC $@{b3.bg} \\to @{b3.bj}$',
      'GJR: $\\hat\\alpha = @{b3.a}$, $\\hat\\gamma = @{b3.g}$ ($t = @{b3.gt}$), $\\hat\\beta = @{b3.b}$; LR $= @{b3.lr}$ (p @{b3.lrp}); BIC $@{b3.bg} \\to @{b3.bj}$'),
    T('Sign bias on GARCH-t residuals: $\\chi^2(3) = @{b3.sb}$ (p @{b3.sbp}), sign $t = @{b3.sbt}$', 'Testul de asimetrie pe reziduurile GARCH-t: $\\chi^2(3) = @{b3.sb}$ (p @{b3.sbp}), $t$ al semnului $= @{b3.sbt}$'),
    T('GJR variance after $-2\\%$: $@{b3.neg}$; after $+2\\%$: $@{b3.pos}$; ratio $@{b3.ratio}$', 'Varianța GJR după $-2\\%$: $@{b3.neg}$; după $+2\\%$: $@{b3.pos}$; raportul $@{b3.ratio}$'),
    T('Interpretation: only falls raise the variance ($\\hat\\alpha = 0$); after a fall the VaR is about $\\sqrt{@{b3.ratio}}$ times the VaR after an equal rise; a symmetric GARCH understates risk after bad days',
      'Interpretare: doar scăderile cresc varianța ($\\hat\\alpha = 0$); după o scădere, VaR este de circa $\\sqrt{@{b3.ratio}}$ ori VaR-ul de după o creștere egală; un GARCH simetric subestimează riscul după zilele proaste')) + qlsem(), 'scriptsize')

D.task(T('B4: Asymmetry on the BVB and in Bitcoin [Proposed]', 'B4: asimetria la BVB și la Bitcoin [Propus]'),
       T('is there a leverage effect in the BET, in Banca Transilvania and in Bitcoin? Model: B3.', 'există un efect de levier la BET, la Banca Transilvania și la Bitcoin? Model: B3.'),
       T('BET since 2000, TLV since 2010, Bitcoin since 2014; daily log returns in \\%', 'BET din 2000, TLV din 2010, Bitcoin din 2014; randamente logaritmice zilnice în \\%'),
       [T('Estimate GJR-GARCH(1,1)-t and EGARCH(1,1)-t; report both $\\gamma$ with robust $t$ statistics.', 'Estimați GJR-GARCH(1,1)-t și EGARCH(1,1)-t; raportați ambele valori $\\gamma$, cu statisticile $t$ robuste.'),
        T('Compute the LR test of GJR against GARCH and the change in BIC.', 'Calculați testul LR pentru GJR față de GARCH și modificarea BIC.'),
        T('Compute the GJR ratio of the variances after $-2\\%$ and $+2\\%$.', 'Calculați raportul GJR dintre varianțele de după $-2\\%$ și $+2\\%$.'),
        T('Interpretation: would you use an asymmetric model for each of the three? Why?', 'Interpretare: ați folosi un model asimetric pentru fiecare dintre cele trei serii? De ce?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(k):
    return (f'{SHORT[k]} & $@{{b4.{k}.g}}$ & ${{@{{b4.{k}.gt}}}}$ & $@{{b4.{k}.lr}}$ & @{{b4.{k}.lrp}} & ${{@{{b4.{k}.db}}}}$ & ${{@{{b4.{k}.eg}}}}$ & ${{@{{b4.{k}.egt}}}}$ & $@{{b4.{k}.ratio}}$')


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrrrr', T('& GJR $\\gamma$ & $t$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t$ & ratio $-2/+2$', '& GJR $\\gamma$ & $t$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t$ & raport $-2/+2$'),
    [b4row(k) for k in ['bet', 'tlv', 'btc']], size='footnotesize') + items(
    T('BET and TLV: $\\gamma > 0$ (EGARCH $\\gamma < 0$), significant by LR at 5\\%, but BIC rises ($\\Delta$BIC $> 0$) and the ratio is only @{b4.bet.ratio} and @{b4.tlv.ratio}',
      'BET și TLV: $\\gamma > 0$ (în EGARCH $\\gamma < 0$), semnificativ după testul LR la 5\\%, dar BIC crește ($\\Delta$BIC $> 0$), iar raportul este doar @{b4.bet.ratio} și, respectiv, @{b4.tlv.ratio}'),
    T('Bitcoin: no asymmetry at all ($t = @{b4.btc.gt}$; ratio @{b4.btc.ratio})', 'Bitcoin: nicio asimetrie ($t = @{b4.btc.gt}$; raportul @{b4.btc.ratio})'),
    T('Interpretation: for the BVB the effect is real but small: GARCH-t is enough for most uses; for Bitcoin a symmetric model is right; for the S\\&P 500 (B3) asymmetry is essential',
      'Interpretare: la BVB efectul este real, dar mic: GARCH-t este suficient pentru majoritatea aplicațiilor; pentru Bitcoin un model simetric este potrivit; pentru S\\&P 500 (B3) asimetria este esențială')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: GARCH against EWMA for the S\\&P 500 [Solved]', 'B5: GARCH față de EWMA pentru S\\&P 500 [Rezolvat]'),
       T('does GARCH(1,1)-t forecast tomorrow\'s variance of the S\\&P 500 better than EWMA, and is its VaR 1\\% exceeded on 1\\% of days?',
         'prognozează GARCH(1,1)-t varianța de mîine a S\\&P 500 mai bine decît EWMA și este VaR 1\\% al lui depășit în 1\\% dintre zile?'),
       T('S\\&P 500 since 2000; forecasts for every day since 2 January 2015', 'S\\&P 500 din 2000; prognoze pentru fiecare zi de la 2 ianuarie 2015'),
       [T('Re-estimate GARCH(1,1)-t every 250 days on all data up to that day and compute the one-day forecasts $h_t$; compute EWMA with $\\lambda = 0.94$.',
          'Reestimați GARCH(1,1)-t la fiecare 250 de zile pe toate datele pînă în acea zi și calculați prognozele pe o zi $h_t$; calculați EWMA cu $\\lambda = 0{,}94$.'),
        T('Compute the mean QLIKE of both models and the Diebold--Mariano $t$ statistic.', 'Calculați media QLIKE pentru ambele modele și statistica $t$ Diebold--Mariano.'),
        T('Count the exceedances of the GARCH-t VaR 1\\% and of the EWMA-Normal VaR 1\\% and compare with the expected number.', 'Numărați depășirile VaR 1\\% GARCH-t și VaR 1\\% EWMA-Normal și comparați-le cu numărul așteptat.'),
        T('Draw the cumulative difference of the losses.', 'Desenați diferența cumulată a pierderilor.'),
        T('Interpretation: which model would you give to a risk manager, and what would you still fix?', 'Interpretare: ce model i-ați da unui manager de risc și ce ați mai corecta?')],
       T('four numbers, two counts, the chart and two sentences', 'patru valori, două numărători, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch9_sem_b5', h='0.32') + items(
    T('@{b5.n} days; mean QLIKE: GARCH-t $@{b5.qg}$, EWMA $@{b5.qe}$; DM: mean difference $@{b5.dmm}$, $t = @{b5.dm}$ (p @{b5.dmp}): GARCH-t is better',
      '@{b5.n} zile; media QLIKE: GARCH-t $@{b5.qg}$, EWMA $@{b5.qe}$; DM: diferența medie $@{b5.dmm}$, $t = @{b5.dm}$ (p @{b5.dmp}): GARCH-t este mai bun'),
    T('VaR 1\\% exceedances: GARCH-t @{b5.ng} (@{b5.eg}\\%), EWMA-Normal @{b5.ne} (@{b5.ee}\\%); expected about @{b5.exp}',
      'Depășiri VaR 1\\%: GARCH-t @{b5.ng} (@{b5.eg}\\%), EWMA-Normal @{b5.ne} (@{b5.ee}\\%); așteptat circa @{b5.exp}'),
    T('Interpretation: GARCH-t, but both VaR models are exceeded too often; add the leverage effect (B3) and test the exceedances formally (Chapter 10)',
      'Interpretare: GARCH-t, dar ambele modele VaR sînt depășite prea des; adăugăm efectul de levier (B3) și testăm formal depășirile (Capitolul 10)')) + qlsem(), 'scriptsize')

D.task(T('B6: GARCH against EWMA for the DAX and the BET [Proposed]', 'B6: GARCH față de EWMA pentru DAX și BET [Propus]'),
       T('does the result of B5 hold for the DAX and for the BET? Model: B5.', 'se păstrează rezultatul din B5 pentru DAX și pentru BET? Model: B5.'),
       T('DAX and BET since 2000; forecasts for every day since January 2015', 'DAX și BET din 2000; prognoze pentru fiecare zi din ianuarie 2015'),
       [T('Compute the out-of-sample GARCH(1,1)-t and EWMA forecasts as in B5.', 'Calculați prognozele GARCH(1,1)-t și EWMA în afara eșantionului, ca în B5.'),
        T('Compute the mean QLIKE of both models and the DM test.', 'Calculați media QLIKE pentru ambele modele și testul DM.'),
        T('Count the VaR 1\\% exceedances of both models.', 'Numărați depășirile VaR 1\\% pentru ambele modele.'),
        T('Interpretation: for which index does GARCH help most, and why?', 'Interpretare: pentru care indice ajută cel mai mult GARCH și de ce?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{NAMES[k]} & @{{b6.{k}.n}} & $@{{b6.{k}.qg}}$ & $@{{b6.{k}.qe}}$ & ${{@{{b6.{k}.dm}}}}$ (@{{b6.{k}.dmp}}) & @{{b6.{k}.exp}} & '
            f'@{{b6.{k}.ng}} (@{{b6.{k}.eg}}\\%) & @{{b6.{k}.ne}} (@{{b6.{k}.ee}}\\%)')


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& days & QLIKE GARCH-t & QLIKE EWMA & DM $t$ (p) & expected & exc. GARCH-t & exc. EWMA',
                  '& zile & QLIKE GARCH-t & QLIKE EWMA & DM $t$ (p) & așteptat & dep. GARCH-t & dep. EWMA'),
    [b6row(k) for k in ['dax', 'bet']], size='scriptsize') + items(
    T('Both indices: GARCH-t has the smaller QLIKE, significant at 5\\% by DM', 'Ambii indici: GARCH-t are QLIKE mai mic, semnificativ la 5\\% după testul DM'),
    T('BET: the largest QLIKE gain and an almost exact VaR (@{b6.bet.eg}\\% against 1\\%); EWMA-Normal is exceeded about twice as often as it should be',
      'BET: cel mai mare cîștig QLIKE și un VaR aproape exact (@{b6.bet.eg}\\% față de 1\\%); EWMA-Normal este depășit de circa două ori mai des decît ar trebui'),
    T('Interpretation: the BET has heavy tails ($\\nu \\approx 5$) and sudden jumps of volatility: the Student-t quantile and the mean reversion of GARCH both help',
      'Interpretare: BET are cozi groase ($\\nu \\approx 5$) și salturi bruște ale volatilității: ajută atît cuantila Student-t, cît și revenirea la medie din GARCH')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Is Bitcoin Volatility Becoming Equity-Like? [Proposed]', 'C1: devine volatilitatea Bitcoin asemănătoare cu cea a acțiunilor? [Propus]'),
       T('has Bitcoin volatility moved closer to the volatility of the S\\&P 500 since 2021?', 's-a apropiat volatilitatea Bitcoin de volatilitatea S\\&P 500 după 2021?'),
       T('Bitcoin and S\\&P 500 daily log returns, 2015--2019 and 2021--2026; models: B1, B3', 'randamentele logaritmice zilnice Bitcoin și S\\&P 500, 2015--2019 și 2021--2026; modele: B1, B3'),
       [T('Estimate GJR-GARCH(1,1)-t for both series in both periods and report the persistence, $\\gamma$ and $\\nu$.', 'Estimați GJR-GARCH(1,1)-t pentru ambele serii, în ambele perioade, și raportați persistența, $\\gamma$ și $\\nu$.'),
        T('Compute the correlation of the logarithms of the two GARCH-t volatilities on common days in each period.', 'Calculați corelația logaritmilor celor două volatilități GARCH-t în zilele comune, în fiecare perioadă.'),
        T('List the events of 2020--2024 that could explain a change.', 'Enumerați evenimentele din 2020--2024 care ar putea explica o schimbare.'),
        T('Interpretation: how would you separate a lasting change from a change caused by one crisis?', 'Interpretare: cum ați deosebi o schimbare durabilă de o schimbare produsă de o singură criză?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrrr', T('Period & BTC $\\alpha + \\beta + \\gamma/2$ & BTC $\\gamma$ ($t$) & BTC $\\nu$ & S\\&P persistence & S\\&P $\\gamma$ ($t$) & corr.\\ of log-vol.',
                 'Perioada & BTC $\\alpha + \\beta + \\gamma/2$ & BTC $\\gamma$ ($t$) & BTC $\\nu$ & persistență S\\&P & S\\&P $\\gamma$ ($t$) & corel.\\ log-vol.'),
    c1rows, size='scriptsize') + items(
    T('Bitcoin 2015--2019: $\\gamma = @{c1.g0}$ ($t = @{c1.g0t}$): rises raised volatility more than falls (an inverse leverage effect); 2021--2026: $\\gamma = @{c1.g1}$ ($t = @{c1.g1t}$), not significant',
      'Bitcoin 2015--2019: $\\gamma = @{c1.g0}$ ($t = @{c1.g0t}$): creșterile au mărit volatilitatea mai mult decît scăderile (un efect de levier invers); 2021--2026: $\\gamma = @{c1.g1}$ ($t = @{c1.g1t}$), nesemnificativ'),
    T('Correlation of the two volatilities: from $@{c1.c0}$ to $@{c1.c1}$; sample volatility of Bitcoin: from @{c1.v0}\\% to @{c1.v1}\\%; still IGARCH-like',
      'Corelația celor două volatilități: de la $@{c1.c0}$ la $@{c1.c1}$; volatilitatea de selecție a Bitcoin: de la @{c1.v0}\\% la @{c1.v1}\\%; tot de tip IGARCH'),
    T('Events: COVID-19 (2020), institutional buying (2021), the crypto crash of 2022, spot Bitcoin ETFs (2024); design: rolling windows, Ethereum and gold as controls, a test of the change in the correlation',
      'Evenimente: pandemia COVID-19 (2020), cumpărările instituționale (2021), prăbușirea pieței cripto din 2022, ETF-urile spot pe Bitcoin (2024); abordare: ferestre mobile, Ethereum și aurul ca termeni de comparație, un test al schimbării corelației')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to interpret a GARCH(1,1)-t fit of the S\\&P 500 (daily, since 2000): $\\hat\\omega = @{c2.om}$, $\\hat\\alpha = @{c2.a}$, $\\hat\\beta = @{c2.b}$, $\\hat\\nu = @{c2.nu}$. The answer:',
      'Un student a cerut unui asistent AI să interpreteze un GARCH(1,1)-t estimat pentru S\\&P 500 (date zilnice, din 2000): $\\hat\\omega = @{c2.om}$, $\\hat\\alpha = @{c2.a}$, $\\hat\\beta = @{c2.b}$, $\\hat\\nu = @{c2.nu}$. Răspunsul:'),
    T('\\aiprompt{(a) alpha + beta = @{c2.pers}, so a volatility shock disappears within about a week.}', '\\aiprompt{(a) alpha + beta = @{c2.pers}, deci un șoc de volatilitate dispare în aproximativ o săptămînă.}'),
    T('\\aiprompt{(b) The long-run variance is omega/(1 - alpha) = @{c2.uvw}, an annual volatility of @{c2.volw}\\%.}', '\\aiprompt{(b) Varianța de lungă durată este omega/(1 - alpha) = @{c2.uvw}, o volatilitate anuală de @{c2.volw}\\%.}'),
    T('\\aiprompt{(c) A GJR fit gives gamma = @{c2.g} > 0: good news raises volatility more than bad news.}', '\\aiprompt{(c) Un GJR estimat dă gamma = @{c2.g} > 0: știrile bune cresc volatilitatea mai mult decît știrile proaste.}'),
    T('\\aiprompt{(d) Ljung-Box Q(10) of the squared standardised residuals = @{c2.lb}, p = @{c2.lbp}: the GARCH-t model is correct.}', '\\aiprompt{(d) Ljung-Box Q(10) al pătratelor reziduurilor standardizate = @{c2.lb}, p = @{c2.lbp}: modelul GARCH-t este corect.}'),
    T('\\aiprompt{(e) With Student-t innovations the one-day VaR 1\\% is 2.326 times sigma(t+1).}', '\\aiprompt{(e) Cu inovații Student-t, VaR 1\\% pe o zi este de 2,326 ori sigma(t+1).}'),
    T('\\aiprompt{(f) EWMA with lambda = 0.94 is a GARCH(1,1) with omega = 0, alpha = 0.06, beta = 0.94.}', '\\aiprompt{(f) EWMA cu lambda = 0,94 este un GARCH(1,1) cu omega = 0, alpha = 0,06, beta = 0,94.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: the half-life is $\\ln 0.5/\\ln @{c2.pers} = @{c2.hl}$ days, about half a year', '(a) Greșit: timpul de înjumătățire este $\\ln 0{,}5/\\ln @{c2.pers} = @{c2.hl}$ zile, circa jumătate de an'),
    T('(b) Wrong: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = @{c2.uv}$, a long-run volatility of @{c2.vol}\\% (imprecise, since $1 - \\alpha - \\beta$ is small)',
      '(b) Greșit: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = @{c2.uv}$, o volatilitate de lungă durată de @{c2.vol}\\% (imprecisă, deoarece $1 - \\alpha - \\beta$ este mic)'),
    T('(c) Wrong: $\\gamma > 0$ adds to the slope of \\textbf{negative} shocks: bad news raises volatility more (the leverage effect)', '(c) Greșit: $\\gamma > 0$ se adaugă la panta șocurilor \\textbf{negative}: știrile proaste cresc volatilitatea mai mult (efectul de levier)'),
    T('(d) Wrong: no ARCH effect is left, but this does not prove the model correct; the sign-bias test rejects ($\\chi^2(3) = @{c2.sb}$): the asymmetry is missing',
      '(d) Greșit: nu a rămas niciun efect ARCH, dar asta nu dovedește că modelul este corect; testul de asimetrie respinge ($\\chi^2(3) = @{c2.sb}$): lipsește asimetria'),
    T('(e) Wrong: 2.326 is the Normal quantile; for the standardised $t$ with $\\nu = @{c2.nu}$ the factor is $@{c2.qt}$', '(e) Greșit: 2,326 este cuantila Normală; pentru $t$ standardizată cu $\\nu = @{c2.nu}$ factorul este $@{c2.qt}$'),
    T('(f) Correct: EWMA is an IGARCH without a constant', '(f) Corect: EWMA este un IGARCH fără constantă')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('$\\alpha + \\beta$ decides everything long-run: half-life, long-run variance, the shape of the forecasts', '$\\alpha + \\beta$ decide tot ce ține de termenul lung: timpul de înjumătățire, varianța de lungă durată, forma prognozelor'),
    T('Multi-day risk is the sum of the daily variance forecasts, not $\\sqrt{H}$ times today\'s volatility', 'Riscul pe mai multe zile este suma prognozelor zilnice ale varianței, nu $\\sqrt{H}$ ori volatilitatea de azi'),
    T('Heavy-tailed innovations change the VaR quantile; asymmetry changes the VaR after bad days', 'Inovațiile cu cozi groase schimbă cuantila VaR; asimetria schimbă VaR-ul după zilele proaste'),
    T('Compare forecasts out of sample (QLIKE, DM), not only in sample (AIC, BIC)', 'Comparăm prognozele în afara eșantionului (QLIKE, DM), nu doar în eșantion (AIC, BIC)'),
    T('An AI answer is a draft: check the formula of the long-run variance, the sign of $\\gamma$ and the quantile', 'Un răspuns AI este o ciornă: verificați formula varianței de lungă durată, semnul lui $\\gamma$ și cuantila')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 9 develops each topic of today: ARCH and GARCH, estimation, innovations, asymmetry, diagnostics, forecasts, VaR',
      'Cursul 9 dezvoltă fiecare temă de azi: ARCH și GARCH, estimarea, inovațiile, asimetria, diagnosticul, prognozele, VaR'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: rolling windows, Ethereum and gold as controls', 'C1 poate deveni un proiect de echipă: ferestre mobile, Ethereum și aurul ca termeni de comparație'),
    T('Reading: \\refFHH, Ch.~13; exercises in \\refBHL, Ch.~13; \\refTsay, Ch.~3', 'Lectură: \\refFHH, cap.~13; exerciții în \\refBHL, cap.~13; \\refTsay, cap.~3')))

D.references(bib(['BHL', 'Boll', 'BollT', 'BW', 'Christie', 'DM', 'Engle', 'EN', 'FHH', 'GJR', 'Hansen', 'Ljung', 'Nelson', 'Patton', 'Tsay']), per=16)

if __name__ == '__main__':
    D.write(V)
