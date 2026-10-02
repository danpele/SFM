r"""
build_seminar5.py -- Seminarul 5 (Cozi groase și teoria valorilor extreme), EN + RO dintr-o singură sursă
=======================================================================================================
Seminarul are loc ÎNAINTEA cursului 5: secțiunea „Noțiuni necesare azi” conține tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_05/sem5_results.json (seminar5.py).
Ieșire:
  EN/Seminars/seminar5_heavy_tails_evt.tex          (+ _solutions.tex)
  RO/Seminarii/seminar5_cozi_groase_evt_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_05/seminar5.py && python3 latex/build_seminar5.py && python3 latex/sfm_build.py compile 5
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch5_common import NAMES, REFS, T, bib, load_sem, sci   # noqa: E402

S = load_sem()
V = Values()
D = Deck(5, 'seminar', refs=REFS)


def qlsem():
    return '\\quantlet{SFM\\_ch5\\_seminar}{\\qlurl{SFM_ch5_seminar}}'


def put(prefix, d, keys, dec=3):
    for k in keys:
        V.put(f'{prefix}.{k}', d[k], dec)


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    h = A[a]
    for i, l in enumerate(h['logs'], 1):
        V.put(f'{a}.l{i}', l, 3)
    put(a, h, ['mean_log'], 3)
    put(a, h, ['alpha', 'se', 'lo', 'hi', 'factor', 'xp'], 2)
    V.put(f'{a}.ratio', h['ratio'], 3)
    V.raw(f'{a}.k', str(h['k']))
for a, pk, ek in [('a3', ['5', '10'], None), ('a4', ['6', '12'], None)]:
    d = A[a]
    for x in pk:
        V.put(f'{a}.p{x}', 100 * d['p'][x], 3 if 100 * d['p'][x] >= 0.01 else 4)
    for p, v in d['var'].items():
        V.put(f'{a}.var{p}', v, 2)
    for p, v in d['es'].items():
        V.put(f'{a}.es{p}', v, 2)
    V.put(f'{a}.eu', d['e_u'], 2)
    n = A[a + 'n']
    V.put(f'{a}.sig', n['sigma'], 2)
    for x in pk:
        V.raw(f'{a}.np{x}', sci(100 * n['p'][x]))
for a in ['a5', 'a6']:
    d = A[a]
    V.put(f'{a}.p', 100 * d['p'], 2)
    V.put(f'{a}.months', d['months'], 0)
    V.put(f'{a}.years', d['years'], 1)
    V.put(f'{a}.y', d['y'], 5)
    V.put(f'{a}.ypow', d['ypow'], 3)
    V.put(f'{a}.z', d['z'], 2)
    V.put(f'{a}.t', d['t'], 3)
V.put('a5.tpow', A['a5']['t'] ** (-1 / 0.25), 4)
V.put('a6.ez', math.exp(-A['a6']['t']), 5)
for a in ['a7', 'a8']:
    d = A[a]
    for p in ['0.01', '0.001', '0.025']:
        V.put(f'{a}.r{p}', d['r'][p], 3)
        V.put(f'{a}.pow{p}', d['pow'][p], 3)
        V.put(f'{a}.var{p}', d['var'][p], 2)
        V.put(f'{a}.es{p}', d['es'][p], 2)
    V.put(f'{a}.eu', d['e_u'], 2)
    V.put(f'{a}.boxi', d['boxi'], 2)

B1 = S['B1']
V.int('b1.n', B1['n'])
V.raw('b1.k', str(B1['k']))
V.raw('b1.y0', B1['first'][:4])
V.raw('b1.y1', B1['last'][:4])
put('b1', B1, ['alpha', 'se_iid', 'lo', 'hi', 'se_block', 'alpha_1', 'alpha_5', 'alpha_gain', 'u', 'ls'], 2)
V.put('b1.kurt', B1['kurt'], 1)
V.put('b1.tz', (B1['alpha'] - 4) / B1['se_block'], 1)
B2 = S['B2']
for k in ['bet', 'btc', 'snp', 'tlv']:
    put(f'b2.{k}', B2[k], ['alpha', 'lo', 'hi', 'se_block', 'alpha_1', 'alpha_5'], 2)
    V.int(f'b2.{k}.n', B2[k]['n'])
    V.raw(f'b2.{k}.k', str(B2[k]['k']))
    V.raw(f'b2.{k}.y0', B2[k]['first'][:4])
V.put('b2.z', B2['z_bet_btc'], 2)
B3 = S['B3']
V.int('b3.n', B3['n'])
V.raw('b3.y0', B3['first'][:4])
P3 = B3['pot']
put('b3', P3, ['u', 'xi', 'se_xi', 'beta'], 3)
V.raw('b3.nu', str(P3['n_u']))
V.put('b3.eu', B3['e_u'], 2)
V.put('b3.alpha', 1 / P3['xi'], 1)
for lab in ['var1', 'var01', 'es25']:
    for m, v in B3[lab].items():
        V.put(f'b3.{lab}.{m}', v, 2)
B4 = S['B4']
for q in ['0.85', '0.90', '0.95']:
    d = B4[q]
    put(f'b4.{q}', d, ['u', 'xi', 'se', 'beta', 'var1', 'var01', 'es25'], 2)
    V.raw(f'b4.{q}.nu', str(d['n_u']))
put('b4.h', B4['hist'], ['var1', 'var01', 'es25'], 2)
V.int('b4.n', B4['hist']['n'])
B5 = S['B5']
put('b5', B5, ['xi', 'se_xi', 'xi_lo', 'xi_hi', 'mu', 'sigma'], 2)
put('b5', B5, ['rl10', 'max'], 1)
V.raw('b5.q', str(B5['quarters']))
V.raw('b5.nab', str(B5['n_above']))
V.put('b5.years', B5['years'], 1)
B6 = S['B6']
for k in ['dax', 'bet']:
    V.int(f'b6.{k}.n', B6[k]['n_test'])
    V.int(f'b6.{k}.ne', B6[k]['n_est'])
    V.put(f'b6.{k}.exp', B6[k]['expected'], 1)
    for m in ['EVT', 'Historical', 'Normal']:
        V.put(f'b6.{k}.{m}', B6[k][m]['var'], 2)
        V.raw(f'b6.{k}.{m}.x', str(B6[k][m]['exc']))
        pv = B6[k][m]['pvalue']
        V.raw(f'b6.{k}.{m}.p', '⁅' + (f'{pv:.3f}' if pv >= 0.001 else '<0.001') + '⁆' if pv >= 0.001 else '< ⁅0.001⁆')
c1rows = []
V.put('b5.expn', B5['quarters'] / 40, 1)
V.put('b5.alpha', 1 / B5['xi'], 1)
V.put('b2.min', min(S['B2'][k]['alpha'] for k in ['bet', 'btc', 'snp', 'tlv']), 2)
V.put('b2.max', max(S['B2'][k]['alpha'] for k in ['bet', 'btc', 'snp', 'tlv']), 2)
V.raw('c1.kmin', str(min(r['k'] for r in S['C1'].values())))
V.raw('c1.kmax', str(max(r['k'] for r in S['C1'].values())))
V.put('c1.semin', min(r['se'] for r in S['C1'].values()), 2)
V.put('c1.semax', max(r['se'] for r in S['C1'].values()), 2)
V.put('a8.ratio', A['a8']['es']['0.01'] / A['a8']['var']['0.01'], 2)
V.put('a8.lim', 1 / 0.7, 2)
V.put('a56.ratio', A['a5']['p'] / A['a6']['p'], 0)
for per, r in S['C1'].items():
    c1rows.append(f'{per.replace("-", "--")} & {r["n"]} & {r["k"]} & $⁅{r["alpha"]:.2f}⁆$ & $⁅{r["se"]:.2f}⁆$ & '
                  f'$[⁅{r["lo"]:.2f}⁆⟦, ||; ⟧⁅{r["hi"]:.2f}⁆]$')
C2 = S['C2']
put('c2', C2, ['u', 'xi', 'beta', 'var1', 'var1_wrong', 'var25', 'es25', 'es25_wrong'], 2)
V.put('c2.xi3', C2['xi'], 3)
V.put('c2.beta3', C2['beta'], 3)
V.put('c2.u3', C2['u'], 3)
V.put('c2.alpha', C2['alpha'], 1)
V.int('c2.n', C2['n'])
V.raw('c2.nu', str(C2['n_u']))
V.put('c2.share', 100 * C2['share'], 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how heavy are the tails of daily losses, and how large is the loss we should expect once in 1\\,000 days or once in 10 years?',
       '\\textbf{Întrebarea}: cît de groase sînt cozile pierderilor zilnice și cît de mare este pierderea la care trebuie să ne așteptăm o dată la 1\\,000 de zile sau o dată la 10 ani?'),
     [T('this seminar comes \\textbf{before} Lecture 5: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 5: secțiunea „Noțiuni necesare azi” conține toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: Hill estimates, Pareto tails, GEV return levels and POT risk measures, on paper',
        'Partea A: estimări Hill, cozi Pareto, return levels GEV și măsuri de risc POT, pe hîrtie'),
      T('Part B: Hill, POT and block maxima on DAX, BET, Bitcoin, BVB stocks and the S\\&P 500, each with a measure of precision and an interpretation question',
        'Partea B: Hill, POT și block maxima pe DAX, BET, Bitcoin, acțiuni BVB și S\\&P 500, fiecare cu o măsură a preciziei și o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('the Hill estimator and the Hill quantile from a few large losses', 'estimatorul Hill și cuantila Hill din cîteva pierderi mari') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('tail probabilities, VaR, ES and mean excess of a Pareto tail', 'probabilități în coadă, VaR, ES și mean excess pentru o coadă Pareto') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('GEV probabilities and return levels', 'probabilități GEV și return levels') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('VaR and ES from a POT fit, step by step', 'VaR și ES dintr-o ajustare POT, pas cu pas') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('Hill plots and Hill estimates with standard errors for DAX, BET, Bitcoin and BVB stocks', 'grafice Hill și estimări Hill cu erori standard pentru DAX, BET, Bitcoin și acțiuni BVB') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('POT for the BET and for Bitcoin: EVT vs historical vs Normal; the choice of the threshold', 'POT pentru BET și Bitcoin: EVT, simularea istorică și distribuția Normală; alegerea pragului') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('quarterly maxima of S\\&P 500 losses; an out-of-sample check of VaR 1\\%', 'maximele trimestriale ale pierderilor S\\&P 500; o verificare a VaR 1\\% în afara eșantionului') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('has the tail of Bitcoin changed? what is wrong in an AI answer?', 's-a schimbat coada Bitcoin? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, DAX & EODHD & close & @{b1.y0}--@{b1.y1}',
     'BET & EODHD & close & @{b3.y0}--@{b1.y1}',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & @{b2.btc.y0}--@{b1.y1}',
     T('OMV Petrom (SNP), Banca Transilvania (TLV)', 'OMV Petrom (SNP), Banca Transilvania (TLV)') + ' & EODHD & adjusted close & 2010--@{b1.y1}'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; daily losses $L_t = -r_t$; weekends and repeated holiday closes are dropped (except Bitcoin)',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; pierderi zilnice $L_t = -r_t$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin)'),
    T('TLV: the days 30 and 31 May 2016 are removed (a price adjustment applied one day late, Chapter 2)',
      'TLV: zilele de 30 și 31 mai 2016 se elimină (o ajustare de preț înregistrată cu o zi întîrziere, Capitolul 2)'),
    T('In the notebook: \\texttt{losses(\'dax\')}; no account or key is needed',
      'În notebook: \\texttt{losses(\'dax\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Heavy Tails', 'Noțiuni necesare azi (1/5): cozi groase'), items(
    (T('Tail of the losses: $\\bar F(x) = P(L > x)$; \\textbf{heavy (power) tail} with \\textbf{tail index} $\\alpha$: $\\bar F(x) \\approx C x^{-\\alpha}$ for large $x$',
       'Coada pierderilor: $\\bar F(x) = P(L > x)$; \\textbf{coadă groasă (de tip putere)} cu \\textbf{indicele de coadă} $\\alpha$: $\\bar F(x) \\approx C x^{-\\alpha}$ pentru $x$ mare'),
     [T('doubling the loss divides its probability by $2^\\alpha$; on log-log axes: a line with slope $-\\alpha$',
        'dublarea pierderii împarte probabilitatea ei la $2^\\alpha$; pe axe log-log: o dreaptă cu panta $-\\alpha$'),
      T('light tail (Normal, Exponential): $\\bar F(x)$ falls faster than any power', 'coadă subțire (distribuția Normală, distribuția exponențială): $\\bar F(x)$ scade mai repede decît orice putere')]),
    (T('\\textbf{Pareto tail}: $P(L > x) = p_0\\,(x/x_0)^{-\\alpha}$ for $x \\ge x_0$', '\\textbf{Coadă Pareto}: $P(L > x) = p_0\\,(x/x_0)^{-\\alpha}$ pentru $x \\ge x_0$'),
     [T('VaR at level $p < p_0$: $\\mathrm{VaR}_p = x_0\\,(p_0/p)^{1/\\alpha}$; $\\mathrm{ES}_p = \\mathrm{VaR}_p\\,\\alpha/(\\alpha - 1)$ for $\\alpha > 1$',
        'VaR la nivelul $p < p_0$: $\\mathrm{VaR}_p = x_0\\,(p_0/p)^{1/\\alpha}$; $\\mathrm{ES}_p = \\mathrm{VaR}_p\\,\\alpha/(\\alpha - 1)$ pentru $\\alpha > 1$')]),
    T('Moments: $E|L|^m < \\infty$ only for $m < \\alpha$ ($\\alpha \\le 2$: no variance; $\\alpha \\le 4$: no kurtosis)',
      'Momente: $E|L|^m < \\infty$ doar pentru $m < \\alpha$ ($\\alpha \\le 2$: fără varianță; $\\alpha \\le 4$: aplatizarea nu există)')))

D.frame(T('What You Need for Today (2/5): the Hill Estimator', 'Noțiuni necesare azi (2/5): estimatorul Hill'), items(
    (T('Order the losses $L_{(1)} \\ge L_{(2)} \\ge \\dots$; use the $k$ largest: $\\hat\\alpha_k = \\Big[\\dfrac1k\\sum_{i=1}^{k}\\ln\\dfrac{L_{(i)}}{L_{(k+1)}}\\Big]^{-1}$ \\refHill',
       'Ordonăm pierderile $L_{(1)} \\ge L_{(2)} \\ge \\dots$; folosim cele mai mari $k$: $\\hat\\alpha_k = \\Big[\\dfrac1k\\sum_{i=1}^{k}\\ln\\dfrac{L_{(i)}}{L_{(k+1)}}\\Big]^{-1}$ \\refHill'),
     [T('i.i.d.\\ (independent, identically distributed) losses: $\\mathrm{SE} \\approx \\hat\\alpha/\\sqrt{k}$; 95\\% CI $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$',
        'pierderi i.i.d.\\ (independente și identic distribuite): $\\mathrm{SE} \\approx \\hat\\alpha/\\sqrt{k}$; CI de 95\\% $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$'),
      T('\\textbf{moving-block bootstrap}: resample blocks of 20 consecutive days; keeps the clusters of large losses',
        '\\textbf{bootstrap pe blocuri mobile}: reeșantionăm blocuri de 20 de zile consecutive; blocurile păstrează episoadele de pierderi mari grupate')]),
    T('\\textbf{Hill plot}: $\\hat\\alpha_k$ against $k$; choose $k$ in a flat region; here $k = 2.5\\%$ of $n$',
      '\\textbf{Graficul Hill (Hill plot)}: $\\hat\\alpha_k$ în funcție de $k$; alegem $k$ într-o zonă plată; aici $k = 2.5\\%$ din $n$'),
    T('\\textbf{Hill quantile} \\refWeissman: $\\hat x_p = L_{(k+1)}\\,(k/(np))^{1/\\hat\\alpha}$, the loss exceeded with probability $p$',
      '\\textbf{Cuantila Hill} \\refWeissman: $\\hat x_p = L_{(k+1)}\\,(k/(np))^{1/\\hat\\alpha}$, pierderea depășită cu probabilitatea $p$')))

D.frame(T('What You Need for Today (3/5): Peaks over Threshold', 'Noțiuni necesare azi (3/5): peaks over threshold'), items(
    (T('\\textbf{POT} (peaks over threshold): the excesses $Y = L - u$ over a high threshold $u$ follow a \\textbf{GPD} (generalised Pareto distribution) \\refPickands, \\refBdH',
       '\\textbf{POT} (peaks over threshold, vîrfuri peste prag): excesele $Y = L - u$ peste un prag înalt $u$ urmează o \\textbf{GPD} (distribuția Pareto generalizată) \\refPickands, \\refBdH'),
     [T('$P(Y \\le y) = 1 - (1 + \\xi y/\\beta)^{-1/\\xi}$; shape $\\xi$ ($\\xi > 0$: $\\alpha = 1/\\xi$), scale $\\beta > 0$',
        '$P(Y \\le y) = 1 - (1 + \\xi y/\\beta)^{-1/\\xi}$; forma $\\xi$ ($\\xi > 0$: $\\alpha = 1/\\xi$), scala $\\beta > 0$'),
      T('\\textbf{mean excess} $e(u) = E[L - u \\mid L > u]$; for a GPD: $(\\beta + \\xi(v - u))/(1 - \\xi)$, a rising line', '\\textbf{mean excess} (excesul mediu) $e(u) = E[L - u \\mid L > u]$; pentru o GPD: $(\\beta + \\xi(v - u))/(1 - \\xi)$, o dreaptă crescătoare')]),
    (T('With $n$ days and $N_u$ losses above $u$, for $p < N_u/n$ \\refMF:', 'Cu $n$ zile și $N_u$ pierderi peste $u$, pentru $p < N_u/n$ \\refMF:'),
     [T('$\\mathrm{VaR}_p = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{np}{N_u}\\Big)^{-\\xi} - 1\\Big]$, \\quad $\\mathrm{ES}_p = \\dfrac{\\mathrm{VaR}_p + \\beta - \\xi u}{1 - \\xi}$',
        '$\\mathrm{VaR}_p = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{np}{N_u}\\Big)^{-\\xi} - 1\\Big]$, \\quad $\\mathrm{ES}_p = \\dfrac{\\mathrm{VaR}_p + \\beta - \\xi u}{1 - \\xi}$')]),
    T('Threshold: $u =$ the 90\\% quantile of the losses (the 10\\% largest losses, as in \\refMF); check with the mean excess and the QQ plot of the excesses',
      'Pragul: $u =$ cuantila de 90\\% a pierderilor (cele mai mari 10\\% dintre pierderi, ca în \\refMF); verificare cu mean excess și graficul QQ al exceselor')))

D.frame(T('What You Need for Today (4/5): Block Maxima and the GEV', 'Noțiuni necesare azi (4/5): block maxima și GEV'), items(
    (T('$M$ = the largest daily loss in a block (a month, a quarter); for long blocks $M$ follows a \\textbf{GEV} (generalised extreme value) law \\refFT, \\refGnedenko',
       '$M$ = cea mai mare pierdere zilnică dintr-un bloc (o lună, un trimestru); pentru blocuri lungi, $M$ urmează o lege \\textbf{GEV} (distribuția generalizată a valorilor extreme) \\refFT, \\refGnedenko'),
     [T('$P(M \\le x) = \\exp\\{-[1 + \\xi(x - \\mu)/\\sigma]^{-1/\\xi}\\}$; $\\xi = 0$: Gumbel, $\\exp\\{-e^{-(x - \\mu)/\\sigma}\\}$',
        '$P(M \\le x) = \\exp\\{-[1 + \\xi(x - \\mu)/\\sigma]^{-1/\\xi}\\}$; $\\xi = 0$: Gumbel, $\\exp\\{-e^{-(x - \\mu)/\\sigma}\\}$'),
      T('$\\xi > 0$ Fréchet (heavy tails), $\\xi = 0$ Gumbel (light), $\\xi < 0$ Weibull (bounded)', '$\\xi > 0$ Fréchet (cozi groase), $\\xi = 0$ Gumbel (subțiri), $\\xi < 0$ Weibull (mărginite)')]),
    (T('\\textbf{Return level} for $m$ blocks: the level exceeded on average once in $m$ blocks', '\\textbf{Return level} pentru $m$ blocuri: nivelul depășit, în medie, o dată la $m$ blocuri'),
     [T('$z = \\mu + \\dfrac{\\sigma}{\\xi}\\big[(-\\ln(1 - 1/m))^{-\\xi} - 1\\big]$; 10 years = 120 months = 40 quarters',
        '$z = \\mu + \\dfrac{\\sigma}{\\xi}\\big[(-\\ln(1 - 1/m))^{-\\xi} - 1\\big]$; 10 ani = 120 de luni = 40 de trimestre'),
      T('return period of a level $x$: $1/P(M > x)$ blocks', 'perioada de revenire a unui nivel $x$: $1/P(M > x)$ blocuri')])))

D.frame(T('What You Need for Today (5/5): VaR, ES and Their Checks', 'Noțiuni necesare azi (5/5): VaR, ES și verificarea lor'), items(
    (T('$\\mathrm{VaR}_p$: the loss exceeded with probability $p$ (VaR 1\\%, VaR 0.1\\%); $\\mathrm{ES}_p$: the mean loss beyond $\\mathrm{VaR}_p$ (ES 2.5\\%)',
       '$\\mathrm{VaR}_p$: pierderea depășită cu probabilitatea $p$ (VaR 1\\%, VaR 0,1\\%); $\\mathrm{ES}_p$: pierderea medie dincolo de $\\mathrm{VaR}_p$ (ES 2,5\\%)'),
     [T('historical: the empirical $(1 - p)$ quantile of the losses and the mean of the losses beyond it', 'simularea istorică: cuantila empirică de ordin $(1 - p)$ a pierderilor și media pierderilor de dincolo de ea'),
      T('Normal: $\\hat\\mu_L + \\hat\\sigma z_{1-p}$ and $\\hat\\mu_L + \\hat\\sigma\\varphi(z_{1-p})/p$ ($\\varphi$: standard Normal density)', 'distribuția Normală: $\\hat\\mu_L + \\hat\\sigma z_{1-p}$ și $\\hat\\mu_L + \\hat\\sigma\\varphi(z_{1-p})/p$ ($\\varphi$: densitatea distribuției Normale standard)')]),
    (T('\\textbf{Out-of-sample check}: estimate VaR on a first period, count the days of a second period with a loss above it',
       '\\textbf{Verificarea în afara eșantionului}: estimăm VaR pe o primă perioadă, apoi numărăm zilele din a doua perioadă în care pierderea îl depășește'),
     [T('if VaR is right, the count $X \\sim \\mathrm{Binomial}(n, p)$, expected $np$; two-sided p-value $2\\min\\{P(X \\le x), P(X \\ge x)\\}$',
        'dacă VaR este corect, numărul de depășiri $X \\sim \\mathrm{Binomial}(n, p)$, cu media $np$; valoarea $p$ bilaterală $2\\min\\{P(X \\le x), P(X \\ge x)\\}$')]),
    T('Quick checks: VaR 0.1\\% $>$ VaR 1\\%; ES 2.5\\% $>$ VaR 2.5\\%; a GPD with $\\xi < 1/2$ has a finite variance',
      'Verificări rapide: VaR 0,1\\% $>$ VaR 1\\%; ES 2,5\\% $>$ VaR 2,5\\%; o GPD cu $\\xi < 1/2$ are varianță finită')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Hill from Eight Losses', 'A1: Hill din opt pierderi'),
         items(T('The eight largest daily losses (\\%) of a stock, in $n = 2500$ days: $12.4, 9.8, 8.1, 7.5, 6.9, 6.2, 5.8, 5.5$.',
                 'Cele mai mari opt pierderi zilnice (\\%) ale unei acțiuni, în $n = 2500$ de zile: $12.4$; $9.8$; $8.1$; $7.5$; $6.9$; $6.2$; $5.8$; $5.5$.'),
               T('1. Compute the Hill estimate with $k = 7$.', '1. Calculați estimarea Hill cu $k = 7$.'),
               T('2. Compute its standard error and the 95\\% interval.', '2. Calculați eroarea standard și intervalul de 95\\%.'),
               T('3. Compute the Hill quantile for $p = 0.1\\%$.', '3. Calculați cuantila Hill pentru $p = 0{,}1\\%$.'),
               T('Report: $\\hat\\alpha$, SE, the interval, VaR 0.1\\% and one sentence on the moments.', 'Raportați: $\\hat\\alpha$, SE, intervalul, VaR 0,1\\% și o frază despre momente.')),
         items(T('1. $\\ln(L_{(i)}/5.5)$: $@{a1.l1}, @{a1.l2}, @{a1.l3}, @{a1.l4}, @{a1.l5}, @{a1.l6}, @{a1.l7}$; mean $@{a1.mean_log}$; $\\hat\\alpha = @{a1.alpha}$',
                 '1. $\\ln(L_{(i)}/5.5)$: $@{a1.l1}$; $@{a1.l2}$; $@{a1.l3}$; $@{a1.l4}$; $@{a1.l5}$; $@{a1.l6}$; $@{a1.l7}$; media $@{a1.mean_log}$; $\\hat\\alpha = @{a1.alpha}$'),
               T('2. $\\mathrm{SE} = @{a1.alpha}/\\sqrt7 = @{a1.se}$; interval $[@{a1.lo}, @{a1.hi}]$: very wide with $k = 7$', '2. $\\mathrm{SE} = @{a1.alpha}/\\sqrt7 = @{a1.se}$; interval $[@{a1.lo}; @{a1.hi}]$: foarte larg cu $k = 7$'),
               T('3. $k/(np) = 7/2.5 = @{a1.ratio}$; $@{a1.ratio}^{1/@{a1.alpha}} = @{a1.factor}$; $\\hat x_{0.001} = 5.5 \\times @{a1.factor} = @{a1.xp}\\%$',
                 '3. $k/(np) = 7/2.5 = @{a1.ratio}$; $@{a1.ratio}^{1/@{a1.alpha}} = @{a1.factor}$; $\\hat x_{0.001} = 5.5 \\times @{a1.factor} = @{a1.xp}\\%$'),
               T('$\\hat\\alpha \\approx 2.8$: variance finite, kurtosis infinite; but the interval also allows $\\alpha < 2$.', '$\\hat\\alpha \\approx 2{,}8$: varianță finită, aplatizare infinită; dar intervalul permite și $\\alpha < 2$.')),
         size='scriptsize')

D.proposed(T('A2: Hill for a Crypto Asset', 'A2: Hill pentru un activ cripto'),
           items(T('The nine largest daily losses (\\%) in $n = 3000$ days: $15.2, 11.0, 9.4, 8.0, 7.7, 7.1, 6.6, 6.3, 6.0$. Model: A1.',
                   'Cele mai mari nouă pierderi zilnice (\\%) în $n = 3000$ de zile: $15.2$; $11.0$; $9.4$; $8.0$; $7.7$; $7.1$; $6.6$; $6.3$; $6.0$. Model: A1.'),
                 T('1. Compute the Hill estimate with $k = 8$, its standard error and the 95\\% interval.', '1. Calculați estimarea Hill cu $k = 8$, eroarea standard și intervalul de 95\\%.'),
                 T('2. Compute the Hill quantile for $p = 0.1\\%$.', '2. Calculați cuantila Hill pentru $p = 0{,}1\\%$.'),
                 T('3. Explain why the interval is so wide and what would make it narrower.', '3. Explicați de ce intervalul este atît de larg și ce l-ar îngusta.'),
                 T('Report: $\\hat\\alpha$, SE, the interval, VaR 0.1\\% and one sentence.', 'Raportați: $\\hat\\alpha$, SE, intervalul, VaR 0,1\\% și o frază.')),
           items(T('1. mean of the 8 logs $= @{a2.mean_log}$, $\\hat\\alpha = @{a2.alpha}$, $\\mathrm{SE} = @{a2.se}$, interval $[@{a2.lo}, @{a2.hi}]$',
                   '1. media celor 8 logaritmi $= @{a2.mean_log}$, $\\hat\\alpha = @{a2.alpha}$, $\\mathrm{SE} = @{a2.se}$, interval $[@{a2.lo}; @{a2.hi}]$'),
                 T('2. $k/(np) = 8/3 = @{a2.ratio}$; factor $@{a2.factor}$; $\\hat x_{0.001} = 6.0 \\times @{a2.factor} = @{a2.xp}\\%$', '2. $k/(np) = 8/3 = @{a2.ratio}$; factorul $@{a2.factor}$; $\\hat x_{0.001} = 6.0 \\times @{a2.factor} = @{a2.xp}\\%$'),
                 T('3. SE $\\propto 1/\\sqrt{k}$: only 8 losses; more data (a longer sample) allows a larger $k$ at the same tail depth.', '3. SE $\\propto 1/\\sqrt{k}$: doar 8 pierderi; mai multe date (un eșantion mai lung) permit un $k$ mai mare la aceeași adîncime a cozii.')),
           size='scriptsize')

D.solved(T('A3: A Pareto Tail', 'A3: o coadă Pareto'),
         items(T('Daily losses: $P(L > x) = 2\\% \\times (x/2.5)^{-3}$ for $x \\ge 2.5\\%$.', 'Pierderile zilnice: $P(L > x) = 2\\% \\times (x/2.5)^{-3}$ pentru $x \\ge 2{,}5\\%$.'),
               T('1. Compute $P(L > 5\\%)$ and $P(L > 10\\%)$.', '1. Calculați $P(L > 5\\%)$ și $P(L > 10\\%)$.'),
               T('2. Compute VaR 1\\%, VaR 0.1\\% and ES 1\\%.', '2. Calculați VaR 1\\%, VaR 0,1\\% și ES 1\\%.'),
               T('3. Compute the mean excess at $u = 4\\%$.', '3. Calculați mean excess-ul la $u = 4\\%$.'),
               T('4. Compare $P(L > 5\\%)$ and $P(L > 10\\%)$ with a Normal law $N(0, \\sigma^2)$ that has the same $P(L > 2.5\\%) = 2\\%$.', '4. Comparați $P(L > 5\\%)$ și $P(L > 10\\%)$ cu o distribuție Normală $N(0, \\sigma^2)$ care are același $P(L > 2{,}5\\%) = 2\\%$.'),
               T('Report: four probabilities, three risk numbers, $e(4)$ and one sentence.', 'Raportați: patru probabilități, trei măsuri de risc, $e(4)$ și o frază.')),
         items(T('1. $2\\% \\times 2^{-3} = @{a3.p5}\\%$; $2\\% \\times 4^{-3} = @{a3.p10}\\%$', '1. $2\\% \\times 2^{-3} = @{a3.p5}\\%$; $2\\% \\times 4^{-3} = @{a3.p10}\\%$'),
               T('2. $\\mathrm{VaR}_p = 2.5\\,(2\\%/p)^{1/3}$: VaR 1\\% $= @{a3.var0.01}\\%$, VaR 0.1\\% $= @{a3.var0.001}\\%$; ES 1\\% $= @{a3.var0.01} \\times 3/2 = @{a3.es0.01}\\%$',
                 '2. $\\mathrm{VaR}_p = 2.5\\,(2\\%/p)^{1/3}$: VaR 1\\% $= @{a3.var0.01}\\%$, VaR 0,1\\% $= @{a3.var0.001}\\%$; ES 1\\% $= @{a3.var0.01} \\times 3/2 = @{a3.es0.01}\\%$'),
               T('3. $e(u) = u/(\\alpha - 1) = 4/2 = @{a3.eu}\\%$', '3. $e(u) = u/(\\alpha - 1) = 4/2 = @{a3.eu}\\%$'),
               T('4. $\\sigma = 2.5/z_{0.98} = @{a3.sig}$; Normal: $P(L > 5) = @{a3.np5}\\%$, $P(L > 10) = @{a3.np10}\\%$: the Normal tail vanishes', '4. $\\sigma = 2.5/z_{0.98} = @{a3.sig}$; distribuția Normală: $P(L > 5) = @{a3.np5}\\%$, $P(L > 10) = @{a3.np10}\\%$: coada distribuției Normale practic dispare')),
         size='scriptsize')

D.proposed(T('A4: A Heavier Pareto Tail', 'A4: o coadă Pareto mai groasă'),
           items(T('$P(L > x) = 1.5\\% \\times (x/3)^{-2.5}$ for $x \\ge 3\\%$. Model: A3.', '$P(L > x) = 1{,}5\\% \\times (x/3)^{-2.5}$ pentru $x \\ge 3\\%$. Model: A3.'),
                 T('1. Compute $P(L > 6\\%)$ and $P(L > 12\\%)$.', '1. Calculați $P(L > 6\\%)$ și $P(L > 12\\%)$.'),
                 T('2. Compute VaR 0.5\\%, VaR 0.1\\%, ES 0.5\\% and the mean excess at $u = 5\\%$.', '2. Calculați VaR 0,5\\%, VaR 0,1\\%, ES 0,5\\% și mean excess-ul la $u = 5\\%$.'),
                 T('3. Say which moments of $L$ exist.', '3. Stabiliți ce momente ale lui $L$ există.'),
                 T('Report: two probabilities, four numbers and one sentence.', 'Raportați: două probabilități, patru valori și o frază.')),
           items(T('1. $@{a4.p6}\\%$ and $@{a4.p12}\\%$ (Normal with the same $P(L > 3\\%)$: $@{a4.np6}\\%$ and $@{a4.np12}\\%$)',
                   '1. $@{a4.p6}\\%$ și $@{a4.p12}\\%$ (distribuția Normală cu același $P(L > 3\\%)$: $@{a4.np6}\\%$ și $@{a4.np12}\\%$)'),
                 T('2. VaR 0.5\\% $= 3 \\times 3^{0.4} = @{a4.var0.005}\\%$; VaR 0.1\\% $= @{a4.var0.001}\\%$; ES 0.5\\% $= @{a4.var0.005} \\times 2.5/1.5 = @{a4.es0.005}\\%$; $e(5) = 5/1.5 = @{a4.eu}\\%$',
                   '2. VaR 0,5\\% $= 3 \\times 3^{0.4} = @{a4.var0.005}\\%$; VaR 0,1\\% $= @{a4.var0.001}\\%$; ES 0,5\\% $= @{a4.var0.005} \\times 2.5/1.5 = @{a4.es0.005}\\%$; $e(5) = 5/1.5 = @{a4.eu}\\%$'),
                 T('3. $\\alpha = 2.5$: mean and variance exist; skewness ($m = 3$) and kurtosis do not.', '3. $\\alpha = 2{,}5$: media și varianța există; asimetria ($m = 3$) și aplatizarea nu.')),
           size='scriptsize')

D.solved(T('A5: A GEV Return Level', 'A5: un return level GEV'),
         items(T('Monthly maxima of daily losses (\\%) follow a GEV with $\\xi = 0.25$, $\\mu = 1.5$, $\\sigma = 0.8$.', 'Maximele lunare ale pierderilor zilnice (\\%) urmează o GEV cu $\\xi = 0{,}25$, $\\mu = 1{,}5$, $\\sigma = 0{,}8$.'),
               T('1. Compute the probability that the largest daily loss of a month exceeds 6\\%.', '1. Calculați probabilitatea ca cea mai mare pierdere zilnică dintr-o lună să depășească 6\\%.'),
               T('2. Compute the return period of a 6\\% loss in years.', '2. Calculați perioada de revenire a unei pierderi de 6\\%, în ani.'),
               T('3. Compute the 10-year return level.', '3. Calculați return level-ul de 10 ani.'),
               T('4. Name the type of the GEV and its tail index.', '4. Numiți tipul GEV și indicele ei de coadă.'),
               T('Report: one probability, one period, one level and one sentence.', 'Raportați: o probabilitate, o perioadă, un nivel și o frază.')),
         items(T('1. $1 + 0.25(6 - 1.5)/0.8 = @{a5.t}$; $P = 1 - \\exp\\{-@{a5.t}^{-4}\\} = @{a5.p}\\%$', '1. $1 + 0.25(6 - 1.5)/0.8 = @{a5.t}$; $P = 1 - \\exp\\{-@{a5.t}^{-4}\\} = @{a5.p}\\%$'),
               T('2. $1/P = @{a5.months}$ months, i.e.\\ about @{a5.years} years', '2. $1/P = @{a5.months}$ luni, adică circa @{a5.years} ani'),
               T('3. $m = 120$: $-\\ln(1 - 1/120) = @{a5.y}$; $@{a5.y}^{-0.25} = @{a5.ypow}$; $z = 1.5 + (0.8/0.25)(@{a5.ypow} - 1) = @{a5.z}\\%$',
                 '3. $m = 120$: $-\\ln(1 - 1/120) = @{a5.y}$; $@{a5.y}^{-0.25} = @{a5.ypow}$; $z = 1.5 + (0.8/0.25)(@{a5.ypow} - 1) = @{a5.z}\\%$'),
               T('4. $\\xi > 0$: Fréchet, tail index $1/\\xi = 4$.', '4. $\\xi > 0$: Fréchet, indicele de coadă $1/\\xi = 4$.')),
         size='scriptsize')

D.proposed(T('A6: The Same Question with a Gumbel Law', 'A6: aceeași întrebare pentru o distribuție Gumbel'),
           items(T('Monthly maxima follow a Gumbel law ($\\xi = 0$) with $\\mu = 1.5$, $\\sigma = 0.8$. Model: A5.', 'Maximele lunare urmează o distribuție Gumbel ($\\xi = 0$) cu $\\mu = 1{,}5$, $\\sigma = 0{,}8$. Model: A5.'),
                 T('1. Compute $P(M > 6\\%)$ and the return period of 6\\% in years.', '1. Calculați $P(M > 6\\%)$ și perioada de revenire a lui 6\\%, în ani.'),
                 T('2. Compute the 10-year return level, $z = \\mu - \\sigma\\ln(-\\ln(1 - 1/120))$.', '2. Calculați return level-ul de 10 ani, $z = \\mu - \\sigma\\ln(-\\ln(1 - 1/120))$.'),
                 T('3. Compare with A5: what does $\\xi = 0.25$ change?', '3. Comparați cu A5: ce schimbă $\\xi = 0{,}25$?'),
                 T('Report: one probability, one period, one level and one sentence.', 'Raportați: o probabilitate, o perioadă, un nivel și o frază.')),
           items(T('1. $(6 - 1.5)/0.8 = @{a6.t}$; $P = 1 - \\exp\\{-e^{-@{a6.t}}\\} = @{a6.p}\\%$; $1/P = @{a6.months}$ months $\\approx @{a6.years}$ years',
                   '1. $(6 - 1.5)/0.8 = @{a6.t}$; $P = 1 - \\exp\\{-e^{-@{a6.t}}\\} = @{a6.p}\\%$; $1/P = @{a6.months}$ de luni $\\approx @{a6.years}$ ani'),
                 T('2. $-\\ln(@{a6.y}) = @{a6.ypow}$; $z = 1.5 + 0.8 \\times @{a6.ypow} = @{a6.z}\\%$', '2. $-\\ln(@{a6.y}) = @{a6.ypow}$; $z = 1.5 + 0.8 \\times @{a6.ypow} = @{a6.z}\\%$'),
                 T('3. With $\\xi = 0.25$ a 6\\% loss is about @{a56.ratio} times more frequent and the 10-year level rises from $@{a6.z}\\%$ to $@{a5.z}\\%$: the heavy tail matters far out.',
                   '3. Cu $\\xi = 0{,}25$, o pierdere de 6\\% este de circa @{a56.ratio} ori mai frecventă, iar nivelul de 10 ani crește de la $@{a6.z}\\%$ la $@{a5.z}\\%$: coada groasă contează mult în zona extremelor.')),
           size='scriptsize')

D.solved(T('A7: POT Risk Measures, Step by Step', 'A7: măsuri de risc POT, pas cu pas'),
         items(T('$n = 5000$ daily losses; $u = 2\\%$; $N_u = 250$ losses above $u$; GPD fit: $\\xi = 0.2$, $\\beta = 0.8$.', '$n = 5000$ de pierderi zilnice; $u = 2\\%$; $N_u = 250$ de pierderi peste $u$; ajustarea GPD: $\\xi = 0{,}2$, $\\beta = 0{,}8$.'),
               T('1. Compute VaR 1\\% and VaR 0.1\\%.', '1. Calculați VaR 1\\% și VaR 0,1\\%.'),
               T('2. Compute VaR 2.5\\% and ES 2.5\\%.', '2. Calculați VaR 2,5\\% și ES 2,5\\%.'),
               T('3. Compute the mean excess at $u$ and the tail index.', '3. Calculați mean excess-ul la $u$ și indicele de coadă.'),
               T('Report: four risk numbers, $e(u)$, $\\alpha$ and one sentence.', 'Raportați: patru măsuri de risc, $e(u)$, $\\alpha$ și o frază.')),
         items(T('1. $\\beta/\\xi = @{a7.boxi}$; $p = 1\\%$: $np/N_u = @{a7.r0.01}$, $@{a7.r0.01}^{-0.2} = @{a7.pow0.01}$, VaR $= 2 + 4(@{a7.pow0.01} - 1) = @{a7.var0.01}\\%$; $p = 0.1\\%$: $@{a7.pow0.001}$, VaR $= @{a7.var0.001}\\%$',
                 '1. $\\beta/\\xi = @{a7.boxi}$; $p = 1\\%$: $np/N_u = @{a7.r0.01}$, $@{a7.r0.01}^{-0.2} = @{a7.pow0.01}$, VaR $= 2 + 4(@{a7.pow0.01} - 1) = @{a7.var0.01}\\%$; $p = 0{,}1\\%$: $@{a7.pow0.001}$, VaR $= @{a7.var0.001}\\%$'),
               T('2. $np/N_u = @{a7.r0.025}$, VaR 2.5\\% $= @{a7.var0.025}\\%$; ES $= (@{a7.var0.025} + 0.8 - 0.4)/0.8 = @{a7.es0.025}\\%$', '2. $np/N_u = @{a7.r0.025}$, VaR 2,5\\% $= @{a7.var0.025}\\%$; ES $= (@{a7.var0.025} + 0.8 - 0.4)/0.8 = @{a7.es0.025}\\%$'),
               T('3. $e(u) = \\beta/(1 - \\xi) = @{a7.eu}\\%$; $\\alpha = 1/\\xi = 5$: a heavy tail, yet with a finite kurtosis.', '3. $e(u) = \\beta/(1 - \\xi) = @{a7.eu}\\%$; $\\alpha = 1/\\xi = 5$: o coadă groasă, dar cu aplatizare finită.')),
         size='scriptsize')

D.proposed(T('A8: POT for a Crypto Asset', 'A8: POT pentru un activ cripto'),
           items(T('$n = 4000$; $u = 3.1\\%$; $N_u = 200$; GPD fit: $\\xi = 0.3$, $\\beta = 1.5$. Model: A7.', '$n = 4000$; $u = 3{,}1\\%$; $N_u = 200$; ajustarea GPD: $\\xi = 0{,}3$, $\\beta = 1{,}5$. Model: A7.'),
                 T('1. Compute VaR 1\\%, VaR 0.1\\% and ES 2.5\\%.', '1. Calculați VaR 1\\%, VaR 0,1\\% și ES 2,5\\%.'),
                 T('2. Compute the ratio ES 1\\% / VaR 1\\% and compare it with $1/(1 - \\xi)$.', '2. Calculați raportul ES 1\\% / VaR 1\\% și comparați-l cu $1/(1 - \\xi)$.'),
                 T('3. Say whether the formulas can be used for VaR 10\\%, and why.', '3. Stabiliți dacă formulele pot fi folosite pentru VaR 10\\% și de ce.'),
                 T('Report: three risk numbers, one ratio and two sentences.', 'Raportați: trei măsuri de risc, un raport și două fraze.')),
           items(T('1. $\\beta/\\xi = @{a8.boxi}$; VaR 1\\% $= @{a8.var0.01}\\%$; VaR 0.1\\% $= @{a8.var0.001}\\%$; VaR 2.5\\% $= @{a8.var0.025}\\%$, ES 2.5\\% $= @{a8.es0.025}\\%$',
                   '1. $\\beta/\\xi = @{a8.boxi}$; VaR 1\\% $= @{a8.var0.01}\\%$; VaR 0,1\\% $= @{a8.var0.001}\\%$; VaR 2,5\\% $= @{a8.var0.025}\\%$, ES 2,5\\% $= @{a8.es0.025}\\%$'),
                 T('2. ES 1\\% $= @{a8.es0.01}\\%$; ratio $@{a8.es0.01}/@{a8.var0.01} = @{a8.ratio}$, above $1/(1 - \\xi) = 1/0.7 = @{a8.lim}$, the limit it approaches as $p \\to 0$', '2. ES 1\\% $= @{a8.es0.01}\\%$; raportul $@{a8.es0.01}/@{a8.var0.01} = @{a8.ratio}$, peste $1/(1 - \\xi) = 1/0.7 = @{a8.lim}$, limita de care se apropie cînd $p \\to 0$'),
                 T('3. No: $N_u/n = 5\\%$; the GPD describes only losses above $u$, so $p$ must be below 5\\%.', '3. Nu: $N_u/n = 5\\%$; GPD descrie doar pierderile peste $u$, deci $p$ trebuie să fie sub 5\\%.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Precision and Interpretation', 'Partea B: date reale, precizie și interpretare')

D.task(T('B1: How Heavy Is the Tail of DAX Losses? [Solved]', 'B1: cît de groasă este coada pierderilor DAX? [Rezolvat]'),
       T('what is the tail index of daily DAX losses, how precise is it, and does the kurtosis of DAX returns exist?',
         'care este indicele de coadă al pierderilor zilnice DAX, cît de precis este și există aplatizarea (kurtosis) randamentelor DAX?'),
       T('DAX closes, @{b1.y0}--@{b1.y1}, daily losses $L_t = -r_t$ in \\%', 'închiderile DAX, @{b1.y0}--@{b1.y1}, pierderi zilnice $L_t = -r_t$ în \\%'),
       [T('Draw the Hill plot of the losses for $k$ between 0.5\\% and 10\\% of $n$, with the i.i.d.\\ 95\\% band.', 'Desenați graficul Hill al pierderilor pentru $k$ între 0,5\\% și 10\\% din $n$, cu banda i.i.d.\\ de 95\\%.'),
        T('Compute $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ with the i.i.d.\\ SE, the 95\\% interval and the moving-block bootstrap SE (blocks of 20 days, 500 samples).',
          'Calculați $\\hat\\alpha$ la $k = 2.5\\%$ din $n$ cu SE i.i.d., intervalul de 95\\% și SE din bootstrap pe blocuri mobile (blocuri de 20 de zile, 500 de eșantioane).'),
        T('Compute the Hill estimate of the gains with the same $k$.', 'Calculați estimarea Hill a cîștigurilor cu același $k$.'),
        T('Interpretation: is $\\alpha > 4$ (a finite kurtosis) compatible with the data?', 'Interpretare: este $\\alpha > 4$ (aplatizare finită) compatibil cu datele?')],
       T('the chart, four numbers and one sentence', 'graficul, patru valori și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch5_sem_b1', h='0.42') + items(
    T('$n = @{b1.n}$, $k = @{b1.k}$, $L_{(k+1)} = @{b1.u}\\%$: $\\hat\\alpha = @{b1.alpha}$, SE i.i.d.\\ $@{b1.se_iid}$, interval $[@{b1.lo}, @{b1.hi}]$, SE block $@{b1.se_block}$; gains: $@{b1.alpha_gain}$',
      '$n = @{b1.n}$, $k = @{b1.k}$, $L_{(k+1)} = @{b1.u}\\%$: $\\hat\\alpha = @{b1.alpha}$, SE i.i.d.\\ $@{b1.se_iid}$, interval $[@{b1.lo}; @{b1.hi}]$, SE blocuri $@{b1.se_block}$; cîștiguri: $@{b1.alpha_gain}$'),
    T('Between $k = 1\\%$ and $5\\%$ of $n$ the estimate falls slowly, from $@{b1.alpha_1}$ to $@{b1.alpha_5}$; the gains have a lighter tail',
      'Între $k = 1\\%$ și $5\\%$ din $n$, estimarea scade lent, de la $@{b1.alpha_1}$ la $@{b1.alpha_5}$; cîștigurile au o coadă mai subțire'),
    T('Interpretation: $\\alpha = 4$ is $@{b1.tz}$ block SE away: the kurtosis of DAX returns does not exist, so the sample kurtosis ($@{b1.kurt}$) is not a stable number',
      'Interpretare: pentru $\\alpha = 4$, statistica $z$ (cu SE pe blocuri) este $@{b1.tz}$: aplatizarea randamentelor DAX nu există, deci aplatizarea de selecție ($@{b1.kurt}$) nu este o mărime stabilă')) + qlsem(), 'footnotesize')

D.task(T('B2: BET, Bitcoin and Two BVB Stocks [Proposed]', 'B2: BET, Bitcoin și două acțiuni BVB [Propus]'),
       T('which of the BET, Bitcoin, OMV Petrom and Banca Transilvania has the heaviest tail of losses? Model: B1.',
         'care dintre BET, Bitcoin, OMV Petrom și Banca Transilvania are cea mai groasă coadă a pierderilor? Model: B1.'),
       T('BET since @{b2.bet.y0}, Bitcoin since @{b2.btc.y0}, SNP and TLV since 2010 (TLV without 30--31 May 2016), daily losses in \\%',
         'BET din @{b2.bet.y0}, Bitcoin din @{b2.btc.y0}, SNP și TLV din 2010 (TLV fără 30--31 mai 2016), pierderi zilnice în \\%'),
       [T('Compute $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ for each series, with the 95\\% interval and the block-bootstrap SE.', 'Calculați $\\hat\\alpha$ la $k = 2.5\\%$ din $n$ pentru fiecare serie, cu intervalul de 95\\% și SE din bootstrap pe blocuri.'),
        T('Add the estimates at $k = 1\\%$ and $5\\%$ of $n$.', 'Adăugați estimările la $k = 1\\%$ și $5\\%$ din $n$.'),
        T('Test the difference between the BET and Bitcoin with $z = (\\hat\\alpha_1 - \\hat\\alpha_2)/\\sqrt{\\mathrm{SE}_1^2 + \\mathrm{SE}_2^2}$.', 'Testați diferența dintre BET și Bitcoin cu $z = (\\hat\\alpha_1 - \\hat\\alpha_2)/\\sqrt{\\mathrm{SE}_1^2 + \\mathrm{SE}_2^2}$.'),
        T('Interpretation: is Bitcoin\'s tail heavier than the BET\'s once the scale of the losses is set aside?', 'Interpretare: este coada Bitcoin mai groasă decît cea a BET, dacă lăsăm deoparte scala pierderilor?')],
       T('one table, one $z$ and two sentences', 'un tabel, un $z$ și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{NAMES[k]} & @{{b2.{k}.n}} & @{{b2.{k}.k}} & $@{{b2.{k}.alpha}}$ & $[@{{b2.{k}.lo}}⟦, ||; ⟧@{{b2.{k}.hi}}]$ & $@{{b2.{k}.se_block}}$ & '
            f'$@{{b2.{k}.alpha_1}}$ / $@{{b2.{k}.alpha_5}}$')


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrr', T('& $n$ & $k$ & $\\hat\\alpha$ & 95\\% interval & SE block & $k = 1\\%$ / $5\\%$', '& $n$ & $k$ & $\\hat\\alpha$ & interval 95\\% & SE blocuri & $k = 1\\%$ / $5\\%$'),
    [b2row(k) for k in ['bet', 'btc', 'snp', 'tlv']], size='footnotesize') + items(
    T('BET vs Bitcoin: $z = @{b2.z}$: no significant difference in the tail index', 'BET față de Bitcoin: $z = @{b2.z}$: nicio diferență semnificativă a indicelui de coadă'),
    T('All four estimates lie between $@{b2.min}$ and $@{b2.max}$; TLV has the lowest $\\hat\\alpha$, BET the second lowest', 'Toate cele patru estimări sînt între $@{b2.min}$ și $@{b2.max}$; TLV are cel mai mic $\\hat\\alpha$, BET al doilea'),
    T('Interpretation: Bitcoin\'s losses are much larger (the threshold $L_{(k+1)}$ is two to three times higher), but the shape of the tail is similar: scale and tail index are different things',
      'Interpretare: pierderile Bitcoin sînt mult mai mari (pragul $L_{(k+1)}$ este de două-trei ori mai mare), dar forma cozii este asemănătoare: scala și indicele de coadă sînt lucruri diferite')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: POT for the BET [Solved]', 'B3: POT pentru BET [Rezolvat]'),
       T('how large are VaR 1\\%, ES 2.5\\% and VaR 0.1\\% of daily BET losses by EVT, compared with historical simulation and the Normal distribution?',
         'cît de mari sînt VaR 1\\%, ES 2,5\\% și VaR 0,1\\% ale pierderilor zilnice BET prin EVT, comparate cu simularea istorică și cu distribuția Normală?'),
       T('BET closes since @{b3.y0}, daily losses in \\%; $u$ = the 90\\% quantile of the losses', 'închiderile BET din @{b3.y0}, pierderi zilnice în \\%; $u$ = cuantila de 90\\% a pierderilor'),
       [T('Draw the empirical mean excess function and mark $u$.', 'Desenați funcția mean excess empirică și marcați $u$.'),
        T('Fit a GPD to the excesses over $u$ by maximum likelihood; report $\\hat\\xi$ with its SE and $\\hat\\beta$.', 'Ajustați o GPD la excesele peste $u$ prin verosimilitate maximă; raportați $\\hat\\xi$ cu SE și $\\hat\\beta$.'),
        T('Draw the QQ plot of the excesses against the fitted GPD.', 'Desenați graficul QQ al exceselor față de distribuția GPD ajustată.'),
        T('Compute VaR 1\\%, ES 2.5\\% and VaR 0.1\\% by EVT, by historical simulation and by the Normal distribution.', 'Calculați VaR 1\\%, ES 2,5\\% și VaR 0,1\\% prin EVT, prin simulare istorică și cu distribuția Normală.'),
        T('Interpretation: which method would you use for the capital of a bank holding the BET, and why?', 'Interpretare: ce metodă ați folosi pentru capitalul unei bănci care deține BET și de ce?')],
       T('the two charts, a table of nine numbers and two sentences', 'cele două grafice, un tabel cu nouă valori și două fraze'), size='scriptsize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch5_sem_b3', h='0.34') + table(
    'lrrr', T('(\\%) & EVT & historical & Normal', '(\\%) & EVT & istoric & Normal'),
    ['VaR 1\\% & $@{b3.var1.EVT}$ & $@{b3.var1.Historical}$ & $@{b3.var1.Normal}$',
     T('ES 2.5\\%', 'ES 2,5\\%') + ' & $@{b3.es25.EVT}$ & $@{b3.es25.Historical}$ & $@{b3.es25.Normal}$',
     T('VaR 0.1\\%', 'VaR 0,1\\%') + ' & $@{b3.var01.EVT}$ & $@{b3.var01.Historical}$ & $@{b3.var01.Normal}$'], size='scriptsize') + items(
    T('$n = @{b3.n}$, $u = @{b3.u}\\%$, $N_u = @{b3.nu}$; $\\hat\\xi = @{b3.xi}$ (SE $@{b3.se_xi}$), $\\hat\\beta = @{b3.beta}$; the mean excess rises linearly above $u$; the QQ plot is on the line except the largest excesses',
      '$n = @{b3.n}$, $u = @{b3.u}\\%$, $N_u = @{b3.nu}$; $\\hat\\xi = @{b3.xi}$ (SE $@{b3.se_xi}$), $\\hat\\beta = @{b3.beta}$; funcția mean excess crește liniar peste $u$; punctele graficului QQ stau pe dreaptă, cu excepția celor mai mari excese'),
    T('Interpretation: EVT, which agrees with the data at 1\\% and gives a smooth VaR 0.1\\%; the Normal VaR 0.1\\% is less than half', 'Interpretare: aș folosi EVT, care concordă cu datele la 1\\% și dă un VaR 0,1\\% stabil; VaR 0,1\\% Normal este mai mic decît jumătate din cel EVT')) + qlsem(), 'scriptsize')

D.task(T('B4: Does the Threshold Matter? Bitcoin [Proposed]', 'B4: contează pragul? Bitcoin [Propus]'),
       T('how much do $\\hat\\xi$ and the EVT risk measures of Bitcoin change with the threshold? Model: B3.', 'cît de mult se schimbă $\\hat\\xi$ și măsurile de risc EVT pentru Bitcoin odată cu pragul? Model: B3.'),
       T('Bitcoin since @{b2.btc.y0}, daily losses in \\%; $u$ at the 85\\%, 90\\% and 95\\% quantiles', 'Bitcoin din @{b2.btc.y0}, pierderi zilnice în \\%; $u$ la cuantilele de 85\\%, 90\\% și 95\\%'),
       [T('For each threshold, fit the GPD and report $u$, $N_u$, $\\hat\\xi$ with its SE and $\\hat\\beta$.', 'Pentru fiecare prag, ajustați GPD și raportați $u$, $N_u$, $\\hat\\xi$ cu SE și $\\hat\\beta$.'),
        T('Compute VaR 1\\%, ES 2.5\\% and VaR 0.1\\% for each threshold, and by historical simulation.', 'Calculați VaR 1\\%, ES 2,5\\% și VaR 0,1\\% pentru fiecare prag și prin simulare istorică.'),
        T('Interpretation: which quantity is robust to the choice of $u$, $\\hat\\xi$ or the VaR?', 'Interpretare: care mărime este robustă la alegerea lui $u$, $\\hat\\xi$ sau VaR?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(q, lab):
    return (f'{lab} & $@{{b4.{q}.u}}$ & @{{b4.{q}.nu}} & $@{{b4.{q}.xi}}$ ($@{{b4.{q}.se}}$) & $@{{b4.{q}.beta}}$ & $@{{b4.{q}.var1}}$ & '
            f'$@{{b4.{q}.es25}}$ & $@{{b4.{q}.var01}}$')


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrrr', T('$u$ at & $u$ (\\%) & $N_u$ & $\\hat\\xi$ (SE) & $\\hat\\beta$ & VaR 1\\% & ES 2.5\\% & VaR 0.1\\%',
                  '$u$ la & $u$ (\\%) & $N_u$ & $\\hat\\xi$ (SE) & $\\hat\\beta$ & VaR 1\\% & ES 2,5\\% & VaR 0,1\\%'),
    [b4row('0.85', '85\\%'), b4row('0.90', '90\\%'), b4row('0.95', '95\\%'),
     T('historical', 'istoric') + ' & & & & & $@{b4.h.var1}$ & $@{b4.h.es25}$ & $@{b4.h.var01}$'], size='footnotesize') + items(
    T('$\\hat\\xi$ moves between thresholds by about one SE; the risk measures move by a few tenths of a percentage point only',
      '$\\hat\\xi$ variază între praguri cu aproximativ o SE; măsurile de risc variază doar cu cîteva zecimi de punct procentual'),
    T('Interpretation: VaR and ES are robust to $u$ (the fit adjusts $\\beta$); $\\hat\\xi$, and every extrapolation far beyond the data, is not',
      'Interpretare: VaR și ES sînt robuste la $u$ (ajustarea modifică $\\beta$); $\\hat\\xi$, și orice extrapolare mult dincolo de date, nu')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: Quarterly Maxima of the S\\&P 500 [Solved]', 'B5: maximele trimestriale ale S\\&P 500 [Rezolvat]'),
       T('how large is the daily loss that the S\\&P 500 exceeds once in 10 years, according to a GEV fit to quarterly maxima?',
         'cît de mare este pierderea zilnică pe care S\\&P 500 o depășește o dată la 10 ani, conform unei ajustări GEV la maximele trimestriale?'),
       T('S\\&P 500 closes since @{b1.y0}, the largest daily loss of each calendar quarter', 'închiderile S\\&P 500 din @{b1.y0}, cea mai mare pierdere zilnică din fiecare trimestru calendaristic'),
       [T('Build the quarterly maxima of the daily losses.', 'Construiți maximele trimestriale ale pierderilor zilnice.'),
        T('Fit a GEV by maximum likelihood; report $\\hat\\xi$ with its 95\\% interval, $\\hat\\mu$ and $\\hat\\sigma$.', 'Ajustați o GEV prin verosimilitate maximă; raportați $\\hat\\xi$ cu intervalul de 95\\%, $\\hat\\mu$ și $\\hat\\sigma$.'),
        T('Draw the QQ plot and the return level plot.', 'Desenați graficul QQ și graficul return levels.'),
        T('Compute the 10-year return level (40 quarters) and count the quarters above it.', 'Calculați return level-ul de 10 ani (40 de trimestre) și numărați trimestrele în care a fost depășit.'),
        T('Interpretation: is the S\\&P 500 tail Fréchet or Gumbel?', 'Interpretare: este coada S\\&P 500 de tip Fréchet sau Gumbel?')],
       T('three parameters, one level, the charts and one sentence', 'trei parametri, un nivel, graficele și o frază'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch5_sem_b5', h='0.42') + items(
    T('@{b5.q} quarters: $\\hat\\xi = @{b5.xi}$, 95\\% interval $[@{b5.xi_lo}, @{b5.xi_hi}]$; $\\hat\\mu = @{b5.mu}\\%$, $\\hat\\sigma = @{b5.sigma}\\%$',
      '@{b5.q} trimestre: $\\hat\\xi = @{b5.xi}$, interval de 95\\% $[@{b5.xi_lo}; @{b5.xi_hi}]$; $\\hat\\mu = @{b5.mu}\\%$, $\\hat\\sigma = @{b5.sigma}\\%$'),
    T('10-year return level: $@{b5.rl10}\\%$; exceeded in @{b5.nab} of @{b5.q} quarters ($@{b5.expn}$ expected in @{b5.years} years); largest loss $@{b5.max}\\%$',
      'Return level-ul de 10 ani: $@{b5.rl10}\\%$; depășit în @{b5.nab} din @{b5.q} trimestre ($@{b5.expn}$ așteptate în @{b5.years} ani); cea mai mare pierdere $@{b5.max}\\%$'),
    T('Interpretation: $\\xi = 0$ lies outside the interval: Fréchet, a heavy tail with $\\alpha = 1/\\hat\\xi = @{b5.alpha}$', 'Interpretare: $\\xi = 0$ este în afara intervalului: Fréchet, o coadă groasă cu $\\alpha = 1/\\hat\\xi = @{b5.alpha}$')) + qlsem(), 'footnotesize')

D.task(T('B6: Does VaR 1\\% Survive a New Period? [Proposed]', 'B6: rezistă VaR 1\\% într-o perioadă nouă? [Propus]'),
       T('estimated until 2014, is the VaR 1\\% of DAX and BET losses exceeded on about 1\\% of the days of 2015--@{b1.y1}? Models: B3, B5.',
         'estimat pînă în 2014, este VaR 1\\% al pierderilor DAX și BET depășit în circa 1\\% din zilele din 2015--@{b1.y1}? Modele: B3, B5.'),
       T('DAX since 1990, BET since 1997, daily losses in \\%; estimation until 31 December 2014, test afterwards', 'DAX din 1990, BET din 1997, pierderi zilnice în \\%; estimare pînă la 31 decembrie 2014, test după aceea'),
       [T('Estimate VaR 1\\% on the first period by EVT (POT, $u$ at the 90\\% quantile), historical simulation and the Normal distribution.', 'Estimați VaR 1\\% pe prima perioadă prin EVT (POT, $u$ la cuantila de 90\\%), simulare istorică și distribuția Normală.'),
        T('Count the test days with a loss above each VaR and compare with $np$.', 'Numărați zilele de test cu o pierdere peste fiecare VaR și comparați cu $np$.'),
        T('Compute the two-sided binomial p-value of each count.', 'Calculați valoarea $p$ binomială bilaterală a fiecărui număr de depășiri.'),
        T('Interpretation: why can all three methods fail in the same direction?', 'Interpretare: de ce pot eșua toate cele trei metode în aceeași direcție?')],
       T('a table with six counts and p-values, and two sentences', 'un tabel cu șase numere de depășiri și valorile $p$, plus două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{NAMES[k]} & @{{b6.{k}.n}} & $@{{b6.{k}.exp}}$ & @{{b6.{k}.EVT.x}} ($@{{b6.{k}.EVT.p}}$) & '
            f'@{{b6.{k}.Historical.x}} ($@{{b6.{k}.Historical.p}}$) & @{{b6.{k}.Normal.x}} ($@{{b6.{k}.Normal.p}}$)')


D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrr', T('& test days & expected & EVT (p-value) & historical & Normal', '& zile de test & așteptat & EVT (valoarea $p$) & istoric & Normal'),
    [b6row('dax'), b6row('bet')], size='footnotesize') + items(
    T('VaR 1\\% (\\%): DAX: EVT $@{b6.dax.EVT}$, historical $@{b6.dax.Historical}$, Normal $@{b6.dax.Normal}$; BET: $@{b6.bet.EVT}$, $@{b6.bet.Historical}$, $@{b6.bet.Normal}$',
      'VaR 1\\% (\\%): DAX: EVT $@{b6.dax.EVT}$, istoric $@{b6.dax.Historical}$, Normal $@{b6.dax.Normal}$; BET: $@{b6.bet.EVT}$, $@{b6.bet.Historical}$, $@{b6.bet.Normal}$'),
    T('EVT and historical: far fewer exceedances than expected (too conservative); the Normal VaR passes for the DAX by chance: a too-thin tail on a too-high variance',
      'EVT și simularea istorică: mult mai puține depășiri decît se așteaptă (estimări prea prudente); VaR Normal trece testul pentru DAX din întîmplare: o coadă prea subțire compensată de o varianță prea mare'),
    T('Interpretation: 2015--@{b1.y1} was calmer than 1990--2014; an unconditional VaR does not adapt to the volatility regime (Chapters 9 and 10)',
      'Interpretare: perioada 2015--@{b1.y1} a fost mai liniștită decît 1990--2014; un VaR necondiționat nu se adaptează regimului de volatilitate (Capitolele 9 și 10)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Has the Tail of Bitcoin Changed? [Proposed]', 'C1: s-a schimbat coada Bitcoin? [Propus]'),
       T('is the tail index of Bitcoin losses different in 2014--2017, 2018--2021 and 2022--@{b1.y1}?', 'este indicele de coadă al pierderilor Bitcoin diferit în 2014--2017, 2018--2021 și 2022--@{b1.y1}?'),
       T('Bitcoin daily losses since @{b2.btc.y0}, split into three periods; models: B1, B2', 'pierderile zilnice Bitcoin din @{b2.btc.y0}, împărțite în trei perioade; modele: B1, B2'),
       [T('Compute the Hill $\\hat\\alpha$ at $k = 2.5\\%$ of $n$ in each period, with the block-bootstrap SE and a 95\\% interval.', 'Calculați $\\hat\\alpha$ Hill la $k = 2.5\\%$ din $n$ în fiecare perioadă, cu SE din bootstrap pe blocuri și un interval de 95\\%.'),
        T('Say whether any two periods differ significantly.', 'Stabiliți dacă vreo pereche de perioade diferă semnificativ.'),
        T('Propose one change to the method that would give a sharper answer.', 'Propuneți o schimbare a metodei care ar da un răspuns mai clar.'),
        T('Interpretation: what can a sample of about 1\\,500 days say about a tail index?', 'Interpretare: ce poate spune un eșantion de circa 1\\,500 de zile despre un indice de coadă?')],
       T('a table, one sentence per comparison and a plan for a project', 'un tabel, o frază pentru fiecare comparație și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrr', T('Period & $n$ & $k$ & $\\hat\\alpha$ & SE block & 95\\% interval', 'Perioada & $n$ & $k$ & $\\hat\\alpha$ & SE blocuri & interval 95\\%'), c1rows, size='footnotesize') + items(
    T('All intervals overlap: no significant change; with $k$ between @{c1.kmin} and @{c1.kmax} the SE is $@{c1.semin}$--$@{c1.semax}$', 'Toate intervalele se suprapun: nicio schimbare semnificativă; cu $k$ între @{c1.kmin} și @{c1.kmax}, SE este $@{c1.semin}$--$@{c1.semax}$'),
    T('Sharper designs: GARCH-standardised losses (Chapter 9), POT with a common threshold, more crypto assets pooled, longer windows',
      'Variante mai clare: pierderi standardizate cu GARCH (Capitolul 9), POT cu un prag comun, mai multe active cripto împreună, ferestre mai lungi'),
    T('Interpretation: about 1\\,500 days leave about 40 tail observations: enough for a rough $\\alpha$, not for detecting a change of 0.5',
      'Interpretare: circa 1\\,500 de zile oferă circa 40 de observații în coadă: suficient pentru un $\\alpha$ aproximativ, nu pentru a detecta o schimbare de 0,5')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to estimate the tail of daily S\\&P 500 losses since 1990 by POT. The answer:',
      'Un student a cerut unui asistent AI să estimeze coada pierderilor zilnice S\\&P 500 din 1990 prin POT. Răspunsul:'),
    T('\\aiprompt{(a) With u = @{c2.u3}\\%, N\\_u = @{c2.nu} of n = @{c2.n}, the GPD gives xi = @{c2.xi3}, so the tail index is alpha = @{c2.xi3} and the variance is infinite.}',
      '\\aiprompt{(a) Cu u = @{c2.u3}\\%, N\\_u = @{c2.nu} din n = @{c2.n}, GPD dă xi = @{c2.xi3}, deci indicele de coadă este alpha = @{c2.xi3}, iar varianța este infinită.}'),
    T('\\aiprompt{(b) VaR 1\\% = u + (beta/xi) [(n p / N\\_u)\\textasciicircum xi - 1] = @{c2.var1_wrong}\\%.}',
      '\\aiprompt{(b) VaR 1\\% = u + (beta/xi) [(n p / N\\_u)\\textasciicircum xi - 1] = @{c2.var1_wrong}\\%.}'),
    T('\\aiprompt{(c) ES 2.5\\% = VaR 2.5\\% x (1 - xi) = @{c2.es25_wrong}\\%.}', '\\aiprompt{(c) ES 2,5\\% = VaR 2,5\\% x (1 - xi) = @{c2.es25_wrong}\\%.}'),
    T('\\aiprompt{(d) The same formula gives a reliable VaR 20\\%.}', '\\aiprompt{(d) Aceeași formulă dă un VaR 20\\% fiabil.}'),
    T('\\aiprompt{(e) Above u, the mean excess function of the fitted GPD is a straight line in the threshold.}', '\\aiprompt{(e) Peste u, funcția mean excess a GPD ajustate este o dreaptă în funcție de prag.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație stabiliți dacă este corectă; dacă nu este, formulați afirmația corectă și, unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: $\\alpha = 1/\\xi = @{c2.alpha}$, not $\\xi$; the variance of a GPD tail is finite for $\\xi < 1/2$', '(a) Greșit: $\\alpha = 1/\\xi = @{c2.alpha}$, nu $\\xi$; varianța unei cozi GPD este finită pentru $\\xi < 1/2$'),
    T('(b) Wrong: the exponent is $-\\xi$; VaR 1\\% $= @{c2.var1}\\%$ (a negative VaR, $@{c2.var1_wrong}\\%$, below $u$, is a red flag)',
      '(b) Greșit: exponentul este $-\\xi$; VaR 1\\% $= @{c2.var1}\\%$ (un VaR negativ, $@{c2.var1_wrong}\\%$, sub $u$, este un semnal de alarmă)'),
    T('(c) Wrong: $\\mathrm{ES}_p = (\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$: VaR 2.5\\% $= @{c2.var25}\\%$, ES 2.5\\% $= @{c2.es25}\\%$; ES can never be below VaR',
      '(c) Greșit: $\\mathrm{ES}_p = (\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$: VaR 2,5\\% $= @{c2.var25}\\%$, ES 2,5\\% $= @{c2.es25}\\%$; ES nu poate fi niciodată sub VaR'),
    T('(d) Wrong: the GPD models only the $N_u/n = @{c2.share}\\%$ largest losses; it is valid only for $p < @{c2.share}\\%$',
      '(d) Greșit: GPD modelează doar cele mai mari $N_u/n = @{c2.share}\\%$ dintre pierderi; este valabilă doar pentru $p < @{c2.share}\\%$'),
    T('(e) Correct: $e(v) = (\\beta + \\xi(v - u))/(1 - \\xi)$ for $v > u$, linear with slope $\\xi/(1 - \\xi)$', '(e) Corect: $e(v) = (\\beta + \\xi(v - u))/(1 - \\xi)$ pentru $v > u$, liniară, cu panta $\\xi/(1 - \\xi)$')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Daily losses have power tails with $\\alpha$ between 2.5 and 3.5: finite variance, no finite kurtosis', 'Pierderile zilnice au cozi de tip putere cu $\\alpha$ între 2,5 și 3,5: varianță finită, aplatizare infinită'),
    T('Hill: fix $k$ in a flat region of the Hill plot; report a standard error that allows for clustering', 'Hill: fixați $k$ într-o zonă plată a graficului Hill; raportați o eroare standard care ține seama de volatility clustering'),
    T('POT: $u$ at a high quantile, GPD for the excesses, VaR and ES by closed formulas; check with mean excess and QQ plots',
      'POT: $u$ la o cuantilă înaltă, GPD pentru excese, VaR și ES prin formule explicite; verificați cu mean excess și grafice QQ'),
    T('Block maxima: GEV, Fréchet for financial losses; return levels beyond the sample are extrapolation', 'Block maxima: GEV, Fréchet pentru pierderile financiare; return levels dincolo de eșantion sînt extrapolare'),
    T('An AI answer is a draft: check $\\alpha = 1/\\xi$, the sign of the exponent, ES $\\ge$ VaR and the range $p < N_u/n$',
      'Un răspuns AI este o ciornă: verificați $\\alpha = 1/\\xi$, semnul exponentului, ES $\\ge$ VaR și domeniul $p < N_u/n$')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 5 develops each topic of today: crashes, regular variation, Hill, mean excess, GEV, GPD, EVT VaR and ES',
      'Cursul 5 dezvoltă fiecare temă de azi: crahuri, variație regulată, Hill, mean excess, GEV, GPD, VaR și ES prin EVT'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: GARCH-standardised losses, POT with a common threshold, more crypto assets', 'C1 poate deveni un proiect de echipă: pierderi standardizate cu GARCH, POT cu prag comun, mai multe active cripto'),
    T('Reading: \\refFHH, Ch.~18; exercises in \\refBHL, Ch.~18; \\refQRM, Ch.~5', 'Lectură: \\refFHH, cap.~18; exerciții în \\refBHL, cap.~18; \\refQRM, cap.~5')))

D.references(bib(['BdH', 'BHL', 'FHH', 'FT', 'Gnedenko', 'Hill', 'MF', 'Pickands', 'QRM', 'Weissman']))

if __name__ == '__main__':
    D.write(V)
