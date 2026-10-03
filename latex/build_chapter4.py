r"""
build_chapter4.py -- Capitolul 4 (Probabilitate pentru finanțe), EN + RO dintr-o singură sursă
==============================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_04/ch4_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter4_probability.tex
  RO/Cursuri/capitol4_probabilitate.tex
Rulare:
  python3 Quantlets/Ch_04/generate_all_charts.py && python3 Quantlets/Ch_04/seminar4.py
  python3 latex/build_chapter4.py && python3 latex/sfm_build.py compile 4
"""

import math
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats   # noqa: E402

from sfm_build import Deck, Values, cols, items, table, photo   # noqa: E402
from ch4_common import BIB, REFS, T, big, load, put_date, sci   # noqa: E402

N = load()
V = Values()
D = Deck(4, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_04}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def commons(name):
    return C + urllib.parse.quote(name.replace(' ', '_')).replace('%', '\\%')


PH = {
    'pascal': ('ch4_pascal.jpg', C + 'Blaise_Pascal_Versailles.JPG',
               '⟦Portrait||Portret⟧: ⟦unknown author, after François II Quesnel||autor necunoscut, după François II Quesnel⟧ (c. 1690); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'grund': ('ch4_kolmogorov_grundbegriffe.jpg', C + 'Kolmogoroff_Grundbegriffe.png',
              '⟦Photo||Foto⟧: KurtSchwitters (2012); CC BY-SA 3.0; Wikimedia Commons'),
    'kolmogorov': ('ch4_kolmogorov_1963.jpg', commons('Математик Андрей Колмогоров в аудитории.jpg'),
                   '⟦Photo||Foto⟧: Vsevolod Tarasevich (1963--1964); CC BY 4.0; Wikimedia Commons'),
    'bachelier': ('ch0_bachelier.jpg', C + 'LouisBachelier.jpg',
                  '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧; ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'brown': ('ch4_brown_1855.jpg', C + 'Robert_Brown_(botanist).jpg',
              '⟦Photo||Foto⟧: Maull \\& Polyblank (1855); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'wiener': ('ch4_wiener.jpg', C + 'Norbert_wiener.jpg',
               '⟦Photo||Foto⟧: Konrad Jacobs; CC BY-SA 2.0 DE; Wikimedia Commons'),
}


def ph(key, cap, h='0.5\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE
# =============================================================================
put_date(V, 'end', N['end'])
V.raw('y0', N['start'][:4])
V.raw('y1', N['end'][:4])
# evenimente
for k in ['sp500', 'bet', 'btc']:
    d = N['down'][k]
    for c in ['p', 'p_cond', 'p_both', 'p_indep', 'p_cond_up']:
        V.put(f'dn.{k}.{c}', 100 * d[c], 1)
# problema punctelor (A are nevoie de 1 cîștig, B de 2; jocuri echitabile)
V.put('pp.a', 3 / 4 * 100, 0)
# variabile aleatoare
R = N['rv']
V.put('rv.p', R['p'], 3)
V.int('rv.weeks', R['n_weeks'])
V.put('rv.mean', R['mean_emp'], 2)
V.put('rv.var', R['var_emp'], 2)
V.put('rv.mth', R['mean_th'], 2)
V.put('rv.vth', R['var_th'], 2)
for k_ in range(6):
    V.put(f'rv.e{k_}', R['emp'][k_], 3)
    V.put(f'rv.b{k_}', R['binom'][k_], 3)
CF = N['cdf']
V.int('cdf.n', CF['n'])
for c, d in [('mean', 3), ('sd', 2), ('q01_emp', 2), ('q01_norm', 2), ('median', 3)]:
    V.put(f'cdf.{c}', CF[c], d)
V.put('cdf.var', -CF['q01_emp'], 2)
V.put('cdf.varn', -CF['q01_norm'], 2)
V.put('cdf.p3e', 100 * CF['p3_emp'], 2)
V.put('cdf.p3n', 100 * CF['p3_norm'], 2)
V.put('cdf.p5e', 100 * CF['p5_emp'], 2)
V.raw('cdf.p5n', sci(100 * CF['p5_norm'], 1))
V.put('cdf.F0', 100 * CF['F0'], 1)
V.put('cdf.ratio5', CF['p5_emp'] / CF['p5_norm'], 0)
# medie, varianță, Cebîșev
L = N['lln']
V.put('an.m', L['mean'], 4)
V.put('an.s', L['sd'], 3)
V.put('an.ppy', L['ppy'], 0)
V.put('an.ma', L['mean_ann'], 1)
V.put('an.sa', L['sd'] * math.sqrt(L['ppy']), 1)
CK = N['check']['sp500']
V.raw('ch.four', str(CK['real']['four']))
V.put('ch.share', 100 * CK['real']['four'] / CF['n'], 2)
V.put('ch.normal', 100 * 2 * stats.norm.sf(4), 4)
V.put('ch.bound', 100 / 16, 2)
# perechi
J = N['joint']['sp500_dax']
for c, d in [('corr', 2), ('cov', 3), ('s1', 2), ('s2', 2), ('sd_p', 2), ('sd_avg', 2), ('var_sum', 3), ('cov_term', 3)]:
    V.put(f'j.{c}', J[c], d)
V.int('j.n', J['n'])
V.put('j.varp', J['sd_p'] ** 2, 3)
JB = N['joint']['sp500_bet']
V.put('jb.corr', JB['corr'], 2)
V.put('jb.sd_p', JB['sd_p'], 2)
V.put('jb.sd_avg', JB['sd_avg'], 2)
DP = N['dep']
V.put('dep.xy', DP['xy_corr'], 3)
for k in ['sp500', 'dax', 'bet', 'btc']:
    V.put(f'dep.{k}.r', DP[k]['rho_r'], 3)
    V.put(f'dep.{k}.r2', DP[k]['rho_r2'], 2)
    V.put(f'dep.{k}.band', DP[k]['band'], 3)
# condiționare
CD = N['cond']
for k in ['sp500', 'btc']:
    V.put(f'cd.{k}.sd1', CD[k]['sd'][0], 2)
    V.put(f'cd.{k}.sd5', CD[k]['sd'][4], 2)
    V.put(f'cd.{k}.ratio', CD[k]['ratio'], 2)
    V.put(f'cd.{k}.sd', CD[k]['sd_all'], 2)
    V.put(f'cd.{k}.mmin', min(CD[k]['mean']), 3)
    V.put(f'cd.{k}.mmax', max(CD[k]['mean']), 3)
    V.put(f'cd.{k}.within', CD[k]['within'], 3)
    V.put(f'cd.{k}.between', CD[k]['between'], 5)
    V.put(f'cd.{k}.total', CD[k]['total'], 3)
    V.put(f'cd.{k}.share', 100 * CD[k]['between'] / CD[k]['total'], 2)
# exemplul cu două regimuri
pc, mc_, sc_, mt_, stt = 0.7, 0.05, 0.8, -0.10, 2.0
Em = pc * mc_ + (1 - pc) * mt_
within = pc * sc_ ** 2 + (1 - pc) * stt ** 2
between = pc * (mc_ - Em) ** 2 + (1 - pc) * (mt_ - Em) ** 2
V.put('rg.E', Em, 3)
V.put('rg.w', within, 3)
V.put('rg.b', between, 4)
V.put('rg.v', within + between, 3)
V.put('rg.sd', math.sqrt(within + between), 3)
# LLN
V.put('lln.se', L['se'], 4)
V.put('lln.t', L['t'], 2)
V.put('lln.years', L['years'], 1)
V.put('lln.sea', L['se_ann'], 1)
V.put('lln.lo', L['mean_ann'] - 1.96 * L['se_ann'], 1)
V.put('lln.hi', L['mean_ann'] + 1.96 * L['se_ann'], 1)
V.put('lln.y3', L['years_for_t3'], 0)
# numere aleatoare
RU = N['randu']
V.raw('ru.nv', str(RU['n_values']))
V.raw('ru.vmin', str(RU['vmin']))
V.raw('ru.vmax', str(RU['vmax']))
V.put('ru.corr', RU['corr_pairs'], 3)
V.raw('ru.lcg', ', '.join(str(x) for x in RU['lcg16'][:17]))
V.raw('ru.per', str(RU['period16']))
V.raw('ru.perbad', str(RU['period_bad']))
IV = N['inv']
V.put('iv.p5n', 100 * IV['p5_norm'], 4)
V.put('iv.p5h', 100 * IV['p5_hist'], 2)
V.put('iv.p5d', 100 * IV['p5_data'], 2)
V.put('iv.minh', IV['min_hist'], 1)
V.put('iv.minn', IV['min_norm'], 1)
V.put('iv.meanE', IV['mean_E'], 3)
V.put('iv.u', 0.9, 1)
V.put('iv.x', -math.log(1 - 0.9) / 2, 3)
MC = N['mc']
V.put('mc.p', 100 * MC['p'], 3)
V.put('mc.z', (-10 - 21 * MC['m']) / (MC['s'] * math.sqrt(21)), 3)
V.put('mc.mh', 21 * MC['m'], 3)
V.put('mc.sh', MC['s'] * math.sqrt(21), 2)
for Nn in ['100', '1000', '10000', '100000', '1000000']:
    V.put(f'mc.e{Nn}', 100 * MC[f'est_{Nn}'], 3)
    V.put(f'mc.s{Nn}', 100 * MC[f'se_{Nn}'], 3)
V.int('mc.nrel', MC['N_rel10'])
# binomial
BN = N['binom']
for c, d in [('u', 4), ('d', 4), ('p', 4), ('ES', 2), ('ES_th', 2), ('q05_bin', 1), ('q05_ln', 1)]:
    V.put(f'bn.{c}', BN[c], d)
V.put('bn.mu', 100 * BN['mu'], 2)
V.put('bn.mulog', 100 * BN['mu_log'], 2)
V.put('bn.sigma', 100 * BN['sigma'], 2)
V.put('bn.pl', 100 * BN['ploss_bin'], 1)
V.put('bn.pln', 100 * BN['ploss_ln'], 1)
# arborele cu doi pași
S0, u2, d2, p2 = 100, 1.2, 0.8, 0.6
V.put('t2.uu', S0 * u2 * u2, 0)
V.put('t2.ud', S0 * u2 * d2, 0)
V.put('t2.dd', S0 * d2 * d2, 0)
V.put('t2.puu', p2 * p2, 2)
V.put('t2.pud', 2 * p2 * (1 - p2), 2)
V.put('t2.pdd', (1 - p2) ** 2, 2)
V.put('t2.E', S0 * (p2 * u2 + (1 - p2) * d2) ** 2, 2)
V.put('t2.g', p2 * u2 + (1 - p2) * d2, 2)
V.put('t2.pstar', (1 - d2) / (u2 - d2), 2)
# procese
PR = N['proc']
for ph_ in ['0.5', '0.95']:
    k = ph_.replace('.', '')
    V.put(f'pr.{k}.var', PR[ph_]['var'], 2)
    V.put(f'pr.{k}.vth', PR[ph_]['var_th'], 2)
    V.put(f'pr.{k}.r5', PR[ph_]['rho5'], 3)
    V.put(f'pr.{k}.r5th', PR[ph_]['rho5_th'], 3)
    V.put(f'pr.{k}.hl', PR[ph_]['half_life'], 1)
for t_ in ['10', '50', '100']:
    V.put(f'pr.rw{t_}', PR['rw_var'][t_], 1)
SP = N['spot']
V.raw('sp.real', SP['real'])
ri = 'ABCD'.index(SP['real'])
V.put('sp.kr', SP['kurt'][ri], 1)
V.put('sp.ks', max(abs(x) for j, x in enumerate(SP['kurt']) if j != ri), 2)
V.put('sp.ar', SP['acf2'][ri], 2)
V.put('sp.as', max(abs(x) for j, x in enumerate(SP['acf2']) if j != ri), 3)
V.put('sp.mr', SP['max_abs_real'], 1)
V.put('sp.ms', SP['max_abs_sim'], 1)
# Wiener
W = N['wiener']
V.put('w.v1', W['var_W1'], 3)
V.put('w.vh', W['var_Wh'], 3)
V.put('w.qv', W['qv_mean'], 3)
V.put('w.qvsd', W['qv_sd'], 3)
V.put('w.in1', 100 * W['share_in_1'], 1)
V.put('w.in1th', 100 * (1 - 2 * stats.norm.sf(1)), 1)
V.put('w.cov', W['cov_half_one'], 3)
# GBM
G = N['gbm']
V.put('g.mulog', 100 * G['mu_log'], 2)
V.put('g.sigma', 100 * G['sigma'], 2)
V.put('g.mu', 100 * G['mu'], 2)
V.put('g.mean', G['mean_T'], 1)
V.put('g.median', G['median_T'], 1)
V.put('g.below', 100 * G['share_below_mean'], 1)
V.put('g.belowth', 100 * G['share_below_mean_th'], 1)
V.put('g.ploss', 100 * G['ploss_T'], 1)
F = N['fan']
for k in ['sp500', 'bet', 'btc']:
    V.put(f'f.{k}.out', 100 * F[k]['outside90'], 0)
    V.put(f'f.{k}.minr', 100 * F[k]['min_rank'], 1)
    put_date(V, f'f.{k}.mind', F[k]['min_rank_date'])
    V.put(f'f.{k}.mu', 100 * F[k]['mu_log'], 1)
    V.put(f'f.{k}.sigma', 100 * F[k]['sigma'], 1)
CH = N['check']
for k in ['sp500', 'bet', 'btc']:
    c = CH[k]
    V.put(f'ck.{k}.k', c['real']['kurt'], 1)
    V.put(f'ck.{k}.klo', c['kurt']['q05'], 2)
    V.put(f'ck.{k}.khi', c['kurt']['q95'], 2)
    V.put(f'ck.{k}.a', c['real']['acf2'], 2)
    V.put(f'ck.{k}.ahi', c['acf2']['q95'], 2)
    V.put(f'ck.{k}.dd', 100 * c['real']['dd'], 0)
    V.put(f'ck.{k}.ddlo', 100 * c['dd']['q05'], 0)
    V.put(f'ck.{k}.ddhi', 100 * c['dd']['q95'], 0)
    V.raw(f'ck.{k}.four', str(c['real']['four']))
    V.put(f'ck.{k}.fourhi', c['four']['q95'], 0)
# drawdown-uri
DD = N['dd']
for k in ['sp500', 'bet', 'btc']:
    d = DD[k]
    V.put(f'dd.{k}.pg', 100 * d['p_gbm'], 0)
    V.put(f'dd.{k}.sr', 100 * d['share_real'], 0)
    V.raw(f'dd.{k}.nh', str(d['n_hit']))
    V.raw(f'dd.{k}.ny', str(d['n_years']))
    V.put(f'dd.{k}.w', 100 * d['worst'], 0)
    V.raw(f'dd.{k}.wy', str(d['worst_year']))
    V.put(f'dd.{k}.p40', 100 * d['p40_gbm'], 2)
    V.raw(f'dd.{k}.n40', str(d['n_hit40']))
    V.put(f'dd.{k}.e40', d['exp40'], 2)
    V.put(f'dd.{k}.med', 100 * d['median_real'], 0)
    V.put(f'dd.{k}.medg', 100 * d['median_gbm'], 0)
V.raw('dd.sp500.years', ', '.join(str(y) for y in DD['sp500']['years_hit']))
# verificați-vă
V.put('cy.var', 0.5 ** 2 * 4 + 0.5 ** 2 * 9 + 2 * 0.25 * 0.5 * 2 * 3, 2)
V.put('cy.sd', math.sqrt(0.5 ** 2 * 4 + 0.5 ** 2 * 9 + 2 * 0.25 * 0.5 * 2 * 3), 2)
V.put('cy.med', 100 * math.exp((0.08 - 0.2 ** 2 / 2) * 1), 2)
V.put('cy.mean', 100 * math.exp(0.08), 2)
V.put('cy.ar', 1 / (1 - 0.8 ** 2), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: which probability tools do we need to describe returns and to simulate prices?',
       '\\textbf{Întrebarea}: de ce instrumente de probabilitate avem nevoie pentru a descrie randamentele și pentru a simula prețuri?'),
     [T('every later chapter (tails, volatility, VaR, efficiency) is built on them', 'toate capitolele următoare (cozi, volatilitate, VaR, eficiență) se sprijină pe ele')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('random variables, distributions, expectation, variance, covariance', 'variabile aleatoare, distribuții, speranță matematică, varianță, covarianță'),
      T('independence vs uncorrelatedness, conditional expectation and conditional variance', 'independență și necorelare, speranța condiționată și varianța condiționată'),
      T('random numbers and Monte Carlo; the binomial model', 'numere aleatoare și Monte Carlo; modelul binomial'),
      T('random walk, martingale, white noise, AR(1); Wiener process and GBM', 'mers aleator, martingal, zgomot alb, AR(1); procesul Wiener și GBM')]),
    T('Real data, @{y0}--@{y1}: S\\&P 500, DAX, BET and Bitcoin (from 2014)', 'Date reale, @{y0}--@{y1}: S\\&P 500, DAX, BET și Bitcoin (din 2014)')))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Compute expectations, variances, covariances and correlations, and the volatility of a portfolio', 'Calculați speranțe matematice, varianțe, covarianțe și corelații, precum și volatilitatea unui portofoliu'),
    T('Explain why uncorrelated returns can still be dependent, and measure it', 'Explicați de ce randamente necorelate pot fi totuși dependente și măsurați acest lucru'),
    T('Use conditional expectation and conditional variance, the building blocks of GARCH', 'Folosiți speranța condiționată și varianța condiționată, cărămizile modelelor GARCH'),
    T('Generate random numbers with the inverse transform and estimate a probability by Monte Carlo, with its standard error', 'Generați numere aleatoare prin metoda transformării inverse și estimați o probabilitate prin Monte Carlo, cu eroarea ei standard'),
    T('Define a random walk, a martingale, white noise, the Wiener process and GBM, and simulate them', 'Definiți mersul aleator, martingalul, zgomotul alb, procesul Wiener și GBM și simulați-le')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, \\textit{Statistics of Financial Markets}, 5th ed., Ch.~3 (Sec.~3.1--3.5) and Ch.~4 (discrete-time stochastic processes)',
       'Manual: \\refFHH, \\textit{Statistics of Financial Markets}, ed. a 5-a, cap.~3 (secț.~3.1--3.5) și cap.~4 (procese stochastice în timp discret)'),
     [T('exercises with solutions: \\refFHHex, Ch.~3--4', 'exerciții rezolvate: \\refFHHex, cap.~3--4')]),
    T('Monte Carlo methods in finance: \\refGlasserman', 'Metode Monte Carlo în finanțe: \\refGlasserman'),
    (T('Python Quantlets of this chapter: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_04}{Quantlets/Ch\\_04}',
       'Quantlet-urile Python ale capitolului: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_04}{Quantlets/Ch\\_04}'),
     [T('ported from the SFE Quantlets of the textbook (SFErandu, SFErangen1, SFErangen2, SFEBinomp, SFEWienerProcess, SFEsimGBM)',
        'portate din Quantlet-urile SFE ale manualului (SFErandu, SFErangen1, SFErangen2, SFEBinomp, SFEWienerProcess, SFEsimGBM)'),
      T('each chart in these slides links to the Quantlet that draws it', 'fiecare grafic din aceste slide-uri are legătură către Quantlet-ul care îl desenează')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter4_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter4_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}',
      'Curs video: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}')))

# =============================================================================
# 1. DE LA JOCURI DE NOROC LA AXIOME
# =============================================================================
D.section('From Games of Chance to Axioms', 'De la jocuri de noroc la axiome')

D.frame(T('1654: the Problem of Points', '1654: problema punctelor'), cols(items(
    T('Two players stop a fair game before the end: how should the stake be split?', 'Doi jucători opresc un joc echitabil înainte de final: cum se împarte miza?'),
    T('Blaise Pascal and Pierre de Fermat solved it in letters (1654): split by the probability of winning, not by the score',
      'Blaise Pascal și Pierre de Fermat au rezolvat-o prin scrisori (1654): împărțim după probabilitatea de a cîștiga, nu după scor'),
    (T('Example: A needs 1 more win, B needs 2; at most two more rounds', 'Exemplu: lui A îi mai trebuie 1 victorie, lui B 2; cel mult încă două runde'),
     [T('B wins only if B wins both rounds: probability $1/4$', 'B cîștigă doar dacă cîștigă ambele runde: probabilitatea $1/4$'),
      T('A should get @{pp.a}\\% of the stake: its \\textbf{expected} gain', 'A trebuie să primească @{pp.a}\\% din miză: cîștigul lui \\textbf{așteptat}')]),
    T('The same idea prices a derivative today: a fair price is an expected payoff', 'Aceeași idee stă azi la baza evaluării derivatelor: prețul corect este o plată așteptată')),
    ph('pascal', 'Blaise Pascal (1623--1662)', h='0.55\\textheight'), wl='0.60', wr='0.36'))

D.frame(T('Kolmogorov\'s Axioms (1933)', 'Axiomele lui Kolmogorov (1933)'), cols(items(
    (T('A \\textbf{probability space} $(\\Omega, \\mathcal{F}, P)$ \\refKolmogorov', 'Un \\textbf{spațiu de probabilitate} $(\\Omega, \\mathcal{F}, P)$ \\refKolmogorov'),
     [T('$\\Omega$: all possible outcomes (all possible paths of tomorrow\'s market)', '$\\Omega$: toate rezultatele posibile (toate traiectoriile posibile ale pieței de mîine)'),
      T('$\\mathcal{F}$: the events we can assign a probability to (a $\\sigma$-algebra)', '$\\mathcal{F}$: evenimentele cărora le putem atribui o probabilitate (o $\\sigma$-algebră)'),
      T('$P$: a function from events to $[0, 1]$', '$P$: o funcție de la evenimente la $[0, 1]$')]),
    (T('Axioms', 'Axiome'),
     [T('$P(A) \\ge 0$; $P(\\Omega) = 1$', '$P(A) \\ge 0$; $P(\\Omega) = 1$'),
      T('for disjoint $A_1, A_2, \\dots$: $P(\\cup_i A_i) = \\sum_i P(A_i)$', 'pentru $A_1, A_2, \\dots$ disjuncte: $P(\\cup_i A_i) = \\sum_i P(A_i)$')]),
    T('Everything else (expectation, independence, processes) is built on these three rules', 'Tot restul (speranță, independență, procese) se construiește pe aceste trei reguli')),
    ph('grund', T('Title page of the \\textit{Grundbegriffe}, Springer 1933', 'Pagina de titlu a lucrării \\textit{Grundbegriffe}, Springer 1933'), h='0.56\\textheight'),
    wl='0.62', wr='0.34'), 'footnotesize')

D.frame(T('Andrey Kolmogorov', 'Andrei Kolmogorov'), cols(items(
    T('Andrey Nikolaevich Kolmogorov (1903--1987), Moscow State University', 'Andrei Nikolaevici Kolmogorov (1903--1987), Universitatea de Stat din Moscova'),
    T('Made probability a branch of measure theory: events are sets, probability is a measure', 'A făcut din probabilitate o ramură a teoriei măsurii: evenimentele sînt mulțimi, probabilitatea este o măsură'),
    T('Also: conditional expectation in its modern form, Markov processes, turbulence, complexity', 'Alte contribuții: speranța condiționată în forma ei modernă, procese Markov, turbulență, complexitate'),
    T('The Kolmogorov--Smirnov test (Chapter 6) carries his name', 'Testul Kolmogorov--Smirnov (Capitolul 6) îi poartă numele')),
    ph('kolmogorov', T('Kolmogorov lecturing in Moscow, 1963--1964', 'Kolmogorov la un curs la Moscova, 1963--1964'), h='0.40\\textheight'),
    wl='0.44', wr='0.52'))

D.frame(T('Events, Conditional Probability, Independence', 'Evenimente, probabilitate condiționată, independență'), items(
    T('\\textbf{Conditional probability}: $P(A \\mid B) = P(A \\cap B)/P(B)$, for $P(B) > 0$', '\\textbf{Probabilitatea condiționată}: $P(A \\mid B) = P(A \\cap B)/P(B)$, pentru $P(B) > 0$'),
    (T('\\textbf{Bayes\' rule}: $P(B \\mid A) = P(A \\mid B)P(B)/P(A)$', '\\textbf{Regula lui Bayes}: $P(B \\mid A) = P(A \\mid B)P(B)/P(A)$'),
     [T('reverses the direction of conditioning: from ``signal given crisis\'\' to ``crisis given signal\'\'', 'inversează sensul condiționării: de la „semnal dată fiind criza” la „criză dat fiind semnalul”')]),
    T('\\textbf{Independent} events: $P(A \\cap B) = P(A)P(B)$, i.e. $P(A \\mid B) = P(A)$', 'Evenimente \\textbf{independente}: $P(A \\cap B) = P(A)P(B)$, adică $P(A \\mid B) = P(A)$'),
    (T('S\\&P 500, @{y0}--@{y1}: $P(\\text{down day}) = @{dn.sp500.p}\\%$; $P(\\text{down} \\mid \\text{down yesterday}) = @{dn.sp500.p_cond}\\%$',
       'S\\&P 500, @{y0}--@{y1}: $P(\\text{zi de scădere}) = @{dn.sp500.p}\\%$; $P(\\text{scădere} \\mid \\text{scădere ieri}) = @{dn.sp500.p_cond}\\%$'),
     [T('two down days in a row: @{dn.sp500.p_both}\\%, against $@{dn.sp500.p_indep}\\%$ under independence', 'două zile de scădere la rînd: @{dn.sp500.p_both}\\%, față de $@{dn.sp500.p_indep}\\%$ în ipoteza de independență'),
      T('BET: @{dn.bet.p_cond}\\% after a down day vs @{dn.bet.p_cond_up}\\% after an up day: thin trading makes signs persist',
        'BET: @{dn.bet.p_cond}\\% după o zi de scădere față de @{dn.bet.p_cond_up}\\% după o zi de creștere: din cauza lichidității reduse, semnul randamentului persistă')])), 'footnotesize')

D.recap(('From Games to Axioms', 'De la jocuri la axiome'), [
    T('Probability space $(\\Omega, \\mathcal{F}, P)$ and three axioms', 'Spațiul de probabilitate $(\\Omega, \\mathcal{F}, P)$ și trei axiome'),
    T('Fair value = expected value: the idea of Pascal and Fermat', 'Valoare corectă = valoare așteptată: ideea lui Pascal și Fermat'),
    T('Conditional probability measures how information changes the odds', 'Probabilitatea condiționată măsoară cum schimbă informația șansele')])

# =============================================================================
# 2. VARIABILE ALEATOARE ȘI DISTRIBUȚII
# =============================================================================
D.section('Random Variables and Distributions', 'Variabile aleatoare și distribuții')

D.frame(T('Random Variables', 'Variabile aleatoare'), items(
    (T('A \\textbf{random variable} $X: \\Omega \\to \\mathbb{R}$ assigns a number to each outcome', 'O \\textbf{variabilă aleatoare} $X: \\Omega \\to \\mathbb{R}$ asociază un număr fiecărui rezultat'),
     [T('tomorrow\'s log return, the number of up days in a week, the loss of a portfolio', 'randamentul logaritmic de mîine, numărul zilelor de creștere dintr-o săptămînă, pierderea unui portofoliu')]),
    (T('\\textbf{Discrete}: finitely or countably many values $x_i$', '\\textbf{Discretă}: un număr finit sau numărabil de valori $x_i$'),
     [T('\\textbf{PMF} (probability mass function): $p(x_i) = P(X = x_i)$, with $\\sum_i p(x_i) = 1$', '\\textbf{PMF} (probability mass function, funcția de masă de probabilitate): $p(x_i) = P(X = x_i)$, cu $\\sum_i p(x_i) = 1$')]),
    (T('\\textbf{Continuous}: $P(a < X \\le b) = \\int_a^b f(x)\\,dx$', '\\textbf{Continuă}: $P(a < X \\le b) = \\int_a^b f(x)\\,dx$'),
     [T('$f$ is the \\textbf{PDF} (probability density function), $f \\ge 0$, $\\int f = 1$; $P(X = x) = 0$ for every single $x$',
        '$f$ este \\textbf{PDF} (probability density function, densitatea de probabilitate), $f \\ge 0$, $\\int f = 1$; $P(X = x) = 0$ pentru orice $x$')]),
    T('Textbook: \\refFHH, Sec.~3.1', 'Manual: \\refFHH, secț.~3.1')))

D.frame(T('The Distribution Function and the Quantile Function', 'Funcția de repartiție și funcția cuantilă'), items(
    (T('\\textbf{CDF} (cumulative distribution function): $F(x) = P(X \\le x)$', '\\textbf{CDF} (cumulative distribution function, funcția de repartiție): $F(x) = P(X \\le x)$'),
     [T('non-decreasing, right-continuous, $F(-\\infty) = 0$, $F(\\infty) = 1$; $f = F\'$ for a continuous $X$', 'nedescrescătoare, continuă la dreapta, $F(-\\infty) = 0$, $F(\\infty) = 1$; $f = F\'$ pentru $X$ continuă')]),
    (T('\\textbf{Quantile} of order $\\alpha$: $q_\\alpha = F^{-1}(\\alpha) = \\inf\\{x : F(x) \\ge \\alpha\\}$', '\\textbf{Cuantila} de ordin $\\alpha$: $q_\\alpha = F^{-1}(\\alpha) = \\inf\\{x : F(x) \\ge \\alpha\\}$'),
     [T('median $= q_{0.5}$; the 1\\% quantile is the return beaten 99\\% of the days', 'mediana $= q_{0.5}$; cuantila de 1\\% este randamentul depășit în 99\\% din zile')]),
    (T('\\textbf{VaR} (Value at Risk) 1\\%: $\\text{VaR}_{1\\%} = -q_{0.01}$, the loss exceeded with probability 1\\%', '\\textbf{VaR} (Value at Risk, valoarea expusă la risc) 1\\%: $\\text{VaR}_{1\\%} = -q_{0.01}$, pierderea depășită cu probabilitatea 1\\%'),
     [T('a quantile of the return distribution; estimation and backtesting in Chapter 10', 'o cuantilă a distribuției randamentelor; estimare și backtesting în Capitolul 10')]),
    T('The \\textbf{empirical CDF} of a sample: $\\hat F(x) = \\frac1n \\#\\{t : r_t \\le x\\}$', '\\textbf{CDF empirică} a unui eșantion: $\\hat F(x) = \\frac1n \\#\\{t : r_t \\le x\\}$')))

chart(T('A Discrete and a Continuous Random Variable', 'O variabilă aleatoare discretă și una continuă'), 'sfm_ch4_random_variables', 'SFM_ch4_random_variables', [
    T('Left: $K$ = up days in a full week of the S\\&P 500 (@{rv.weeks} weeks) vs $B(5, p)$ with $p = @{rv.p}$: mean @{rv.mean} vs @{rv.mth}, variance @{rv.var} vs @{rv.vth}',
      'Stînga: $K$ = zilele de creștere dintr-o săptămînă completă a S\\&P 500 (@{rv.weeks} de săptămîni), comparat cu $B(5, p)$, $p = @{rv.p}$: media @{rv.mean} față de @{rv.mth}, varianța @{rv.var} față de @{rv.vth}'),
    T('Right: daily log returns, a continuous random variable; the histogram estimates the PDF', 'Dreapta: randamentele logaritmice zilnice, o variabilă aleatoare continuă; histograma estimează PDF'),
    T('The binomial fits $K$ well; the Normal density misses the peak and the tails of the returns (Chapter 2)', 'Distribuția binomială se potrivește bine lui $K$; densitatea distribuției Normale nu surprinde nici vîrful, nici cozile randamentelor (Capitolul 2)')], h='0.56\\textheight')

chart(T('Empirical CDF and the 1\\% Quantile of the S\\&P 500', 'CDF empirică și cuantila de 1\\% a S\\&P 500'), 'sfm_ch4_cdf_quantile', 'SFM_ch4_random_variables', [
    T('$n = @{cdf.n}$ days; median $@{cdf.median}\\%$; $\\hat F(0) = @{cdf.F0}\\%$ of the days are not up days', '$n = @{cdf.n}$ zile; mediana $@{cdf.median}\\%$; în $\\hat F(0) = @{cdf.F0}\\%$ din zile randamentul nu a fost pozitiv'),
    T('VaR 1\\%: empirical @{cdf.var}\\%, Normal @{cdf.varn}\\%; $P(r < -3\\%)$: @{cdf.p3e}\\% vs @{cdf.p3n}\\%',
      'VaR 1\\%: empiric @{cdf.var}\\%, în distribuția Normală @{cdf.varn}\\%; $P(r < -3\\%)$: @{cdf.p3e}\\% față de @{cdf.p3n}\\%'),
    T('$P(r < -5\\%)$: @{cdf.p5e}\\% in the data, @{cdf.ratio5} times the value under the Normal distribution ($@{cdf.p5n}\\%$)', '$P(r < -5\\%)$: @{cdf.p5e}\\% în date, de @{cdf.ratio5} de ori mai mult decît în distribuția Normală ($@{cdf.p5n}\\%$)')], h='0.56\\textheight')

D.recap(('Random Variables', 'Variabile aleatoare'), [
    T('PMF for discrete, PDF for continuous variables; the CDF works for both', 'PMF pentru variabile discrete, PDF pentru cele continue; CDF este definită pentru ambele'),
    T('Quantiles invert the CDF; VaR 1\\% is minus the 1\\% quantile', 'Cuantilele inversează CDF; VaR 1\\% este minus cuantila de 1\\%'),
    T('The tails of the empirical CDF are much heavier than the Normal ones', 'Cozile CDF empirice sînt mult mai groase decît cele ale distribuției Normale')])

# =============================================================================
# 3. SPERANȚĂ, VARIANȚĂ, MOMENTE
# =============================================================================
D.section('Expectation and Variance', 'Speranța matematică și varianța')

D.frame(T('Expectation', 'Speranța matematică'), items(
    (T('\\textbf{Expectation} (mean): $E[X] = \\sum_i x_i p(x_i)$ or $E[X] = \\int x f(x)\\,dx$', '\\textbf{Speranța matematică} (media): $E[X] = \\sum_i x_i p(x_i)$ sau $E[X] = \\int x f(x)\\,dx$'),
     [T('the centre of mass of the distribution; it exists only if $E|X| < \\infty$ (not for Cauchy, Chapter 3)', 'centrul de greutate al distribuției; există doar dacă $E|X| < \\infty$ (nu și pentru Cauchy, Capitolul 3)')]),
    T('Of a function: $E[g(X)] = \\int g(x) f(x)\\,dx$, without finding the distribution of $g(X)$', 'A unei funcții: $E[g(X)] = \\int g(x) f(x)\\,dx$, fără a afla distribuția lui $g(X)$'),
    (T('\\textbf{Linearity}: $E[aX + bY + c] = aE[X] + bE[Y] + c$, always (no independence needed)', '\\textbf{Liniaritate}: $E[aX + bY + c] = aE[X] + bE[Y] + c$, întotdeauna (fără independență)'),
     [T('the expected return of a portfolio is the weighted mean of the expected returns', 'randamentul așteptat al unui portofoliu este media ponderată a randamentelor așteptate')]),
    T('In general $E[g(X)] \\ne g(E[X])$: $E[e^X] > e^{E[X]}$ (Jensen), the volatility drag of Chapter 1', 'În general $E[g(X)] \\ne g(E[X])$: $E[e^X] > e^{E[X]}$ (Jensen), volatility drag din Capitolul 1')))

D.frame(T('Variance and Moments', 'Varianța și momentele'), items(
    T('\\textbf{Variance}: $\\text{Var}(X) = E[(X - \\mu)^2] = E[X^2] - \\mu^2$; standard deviation $\\sigma = \\sqrt{\\text{Var}(X)}$ (volatility)',
      '\\textbf{Varianța}: $\\text{Var}(X) = E[(X - \\mu)^2] = E[X^2] - \\mu^2$; abaterea standard $\\sigma = \\sqrt{\\text{Var}(X)}$ (volatilitatea)'),
    T('$\\text{Var}(aX + b) = a^2\\,\\text{Var}(X)$: a shift does not change risk, a scale does', '$\\text{Var}(aX + b) = a^2\\,\\text{Var}(X)$: o translație nu schimbă riscul, o scalare da'),
    T('Moments $E[X^k]$; skewness and kurtosis use $k = 3, 4$ (Chapter 2)', 'Momentele $E[X^k]$; asimetria și boltirea folosesc $k = 3, 4$ (Capitolul 2)'),
    (T('Worked example: S\\&P 500 daily mean $@{an.m}\\%$, s.d. $@{an.s}\\%$, @{an.ppy} days a year', 'Exemplu lucrat: S\\&P 500, media zilnică $@{an.m}\\%$, abaterea standard $@{an.s}\\%$, @{an.ppy} de zile pe an'),
     [T('for i.i.d. daily returns (independent and identically distributed): $E[\\sum r_t] = @{an.ppy} \\times @{an.m} = @{an.ma}\\%$ a year',
        'pentru randamente zilnice i.i.d. (independente și identic distribuite): $E[\\sum r_t] = @{an.ppy} \\times @{an.m} = @{an.ma}\\%$ pe an'),
      T('$\\text{Var}(\\sum r_t) = @{an.ppy}\\,\\sigma^2$, so annual volatility $@{an.s}\\sqrt{@{an.ppy}} = @{an.sa}\\%$: the square-root-of-time rule', '$\\text{Var}(\\sum r_t) = @{an.ppy}\\,\\sigma^2$, deci volatilitatea anuală $@{an.s}\\sqrt{@{an.ppy}} = @{an.sa}\\%$: regula rădăcinii pătrate a timpului')])), 'footnotesize')

D.frame(T('How Rare Is a 4-Sigma Day? Chebyshev\'s Bound', 'Cît de rară este o zi de 4 sigma? Marginea lui Cebîșev'), items(
    (T('\\textbf{Chebyshev\'s inequality}: for any $X$ with finite variance, $P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2$', '\\textbf{Inegalitatea lui Cebîșev}: pentru orice $X$ cu varianță finită, $P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2$'),
     [T('it uses only the mean and the variance, so it holds for every distribution', 'folosește doar media și varianța, deci este valabilă pentru orice distribuție')]),
    (T('$k = 4$, S\\&P 500, @{y0}--@{y1}', '$k = 4$, S\\&P 500, @{y0}--@{y1}'),
     [T('Chebyshev bound: at most @{ch.bound}\\% of the days', 'marginea Cebîșev: cel mult @{ch.bound}\\% din zile'),
      T('Normal distribution: @{ch.normal}\\%', 'distribuția Normală: @{ch.normal}\\%'),
      T('data: @{ch.four} days, @{ch.share}\\%', 'în date: @{ch.four} zile, @{ch.share}\\%')]),
    T('Reality is between the two: the Normal distribution is far too optimistic, the worst case of Chebyshev too pessimistic', 'Realitatea este între cele două: distribuția Normală este mult prea optimistă, cazul cel mai rău al lui Cebîșev prea pesimist')))

D.recap(('Expectation and Variance', 'Speranța și varianța'), [
    T('Expectation is linear; variance scales with the square', 'Speranța este liniară; varianța se scalează cu pătratul'),
    T('For i.i.d. returns the mean grows with $h$ and the volatility with $\\sqrt{h}$', 'Pentru randamente i.i.d., media crește cu $h$, iar volatilitatea cu $\\sqrt{h}$'),
    T('Chebyshev bounds tail probabilities for any distribution', 'Cebîșev mărginește probabilitățile din cozi pentru orice distribuție')])

# =============================================================================
# 4. DISTRIBUȚII COMUNE, COVARIANȚĂ, INDEPENDENȚĂ
# =============================================================================
D.section('Joint Distributions and Dependence', 'Distribuții comune și dependență')

D.frame(T('Joint Distribution, Covariance, Correlation', 'Distribuția comună, covarianța, corelația'), items(
    (T('\\textbf{Joint CDF}: $F(x, y) = P(X \\le x, Y \\le y)$; marginals $F_X(x) = F(x, \\infty)$', '\\textbf{CDF comună}: $F(x, y) = P(X \\le x, Y \\le y)$; marginalele $F_X(x) = F(x, \\infty)$'),
     [T('joint PDF $f(x, y)$; marginal $f_X(x) = \\int f(x, y)\\,dy$', 'PDF comună $f(x, y)$; marginala $f_X(x) = \\int f(x, y)\\,dy$')]),
    T('\\textbf{Covariance}: $\\text{Cov}(X, Y) = E[(X - \\mu_X)(Y - \\mu_Y)] = E[XY] - \\mu_X\\mu_Y$', '\\textbf{Covarianța}: $\\text{Cov}(X, Y) = E[(X - \\mu_X)(Y - \\mu_Y)] = E[XY] - \\mu_X\\mu_Y$'),
    (T('\\textbf{Correlation}: $\\rho = \\text{Cov}(X, Y)/(\\sigma_X\\sigma_Y) \\in [-1, 1]$', '\\textbf{Corelația}: $\\rho = \\text{Cov}(X, Y)/(\\sigma_X\\sigma_Y) \\in [-1, 1]$'),
     [T('measures \\textbf{linear} dependence only; $|\\rho| = 1$ iff $Y = a + bX$', 'măsoară doar dependența \\textbf{liniară}; $|\\rho| = 1$ dacă și numai dacă $Y = a + bX$')]),
    (T('\\textbf{Variance of a sum}: $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}(X, Y)$', '\\textbf{Varianța unei sume}: $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}(X, Y)$'),
     [T('with weights $w$ and covariance matrix $\\Sigma$: $\\sigma_p^2 = w^\\top \\Sigma w$', 'cu ponderile $w$ și matricea de covarianță $\\Sigma$: $\\sigma_p^2 = w^\\top \\Sigma w$')])), 'footnotesize')

chart(T('Joint Distributions of Daily Returns', 'Distribuții comune ale randamentelor zilnice'), 'sfm_ch4_joint', 'SFM_ch4_dependence', [
    T('Common trading days only: prices are joined first, then differenced ($n = @{j.n}$ days for S\\&P 500 and DAX)', 'Doar zilele de tranzacționare comune: întîi se unesc prețurile, apoi se calculează randamentele ($n = @{j.n}$ zile pentru S\\&P 500 și DAX)'),
    T('S\\&P 500 and DAX: $\\rho = @{j.corr}$; S\\&P 500 and BET: $\\rho = @{jb.corr}$', 'S\\&P 500 și DAX: $\\rho = @{j.corr}$; S\\&P 500 și BET: $\\rho = @{jb.corr}$'),
    T('The cloud is elliptic in the centre, but extreme days are joint: crises hit all markets together', 'Norul este eliptic în centru, dar zilele extreme apar simultan: crizele lovesc toate piețele deodată')], h='0.56\\textheight')

D.frame(T('Worked Example: a 50/50 Portfolio of S\\&P 500 and DAX', 'Exemplu lucrat: un portofoliu 50/50 din S\\&P 500 și DAX'), items(
    T('Daily s.d.: $\\sigma_1 = @{j.s1}\\%$, $\\sigma_2 = @{j.s2}\\%$; covariance $@{j.cov}$; correlation $@{j.corr}$', 'Abaterile standard zilnice: $\\sigma_1 = @{j.s1}\\%$, $\\sigma_2 = @{j.s2}\\%$; covarianța $@{j.cov}$; corelația $@{j.corr}$'),
    (T('$\\sigma_p^2 = 0.25\\sigma_1^2 + 0.25\\sigma_2^2 + 2 \\times 0.25\\,\\text{Cov} = @{j.var_sum} + @{j.cov_term} = @{j.varp}$', '$\\sigma_p^2 = 0.25\\sigma_1^2 + 0.25\\sigma_2^2 + 2 \\times 0.25\\,\\text{Cov} = @{j.var_sum} + @{j.cov_term} = @{j.varp}$'),
     [T('$\\sigma_p = @{j.sd_p}\\%$, below the average volatility $@{j.sd_avg}\\%$: diversification', '$\\sigma_p = @{j.sd_p}\\%$, sub volatilitatea medie $@{j.sd_avg}\\%$: diversificare')]),
    T('S\\&P 500 and BET, lower correlation: $\\sigma_p = @{jb.sd_p}\\%$ vs average $@{jb.sd_avg}\\%$, a larger gain', 'S\\&P 500 și BET, corelație mai mică: $\\sigma_p = @{jb.sd_p}\\%$ față de media $@{jb.sd_avg}\\%$, un cîștig mai mare'),
    T('Diversification works only through $\\rho < 1$; in a crash correlations rise, exactly when it is needed', 'Diversificarea funcționează doar prin $\\rho < 1$; într-un crah corelațiile cresc, exact cînd este nevoie de ea')))

D.frame(T('Independence and Uncorrelatedness', 'Independență și necorelare'), items(
    (T('$X$ and $Y$ are \\textbf{independent} if $F(x, y) = F_X(x)F_Y(y)$ for all $x, y$', '$X$ și $Y$ sînt \\textbf{independente} dacă $F(x, y) = F_X(x)F_Y(y)$ pentru orice $x, y$'),
     [T('then $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ for \\textbf{all} functions $g, h$', 'atunci $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ pentru \\textbf{toate} funcțiile $g, h$')]),
    T('Independent $\\Rightarrow$ uncorrelated ($g, h$ = identity); the converse is false', 'Independente $\\Rightarrow$ necorelate ($g, h$ = identitatea); reciproca este falsă'),
    (T('Example: $X \\sim N(0, 1)$, $Y = X^2$', 'Exemplu: $X \\sim N(0, 1)$, $Y = X^2$'),
     [T('$\\text{Cov}(X, X^2) = E[X^3] = 0$: uncorrelated', '$\\text{Cov}(X, X^2) = E[X^3] = 0$: necorelate'),
      T('but $Y$ is a function of $X$: completely dependent', 'dar $Y$ este o funcție de $X$: complet dependente')]),
    T('Exception: for jointly Normal variables, uncorrelated $\\Leftrightarrow$ independent', 'Excepție: pentru variabile cu distribuție Normală comună (bivariată), necorelate $\\Leftrightarrow$ independente')))

chart(T('Returns Are Uncorrelated, Squared Returns Are Not', 'Randamentele sînt necorelate, pătratele lor nu'), 'sfm_ch4_uncorrelated_dependent', 'SFM_ch4_dependence', [
    T('Left: simulated $Y = X^2$, sample correlation $@{dep.xy}$, yet $Y$ is determined by $X$', 'Stînga: $Y = X^2$ simulat, corelația de selecție $@{dep.xy}$, deși $Y$ este determinat de $X$'),
    T('Right: $\\text{Corr}(r_t, r_{t-1})$ is small (S\\&P 500 $@{dep.sp500.r}$, DAX $@{dep.dax.r}$, Bitcoin $@{dep.btc.r}$); $\\text{Corr}(r_t^2, r_{t-1}^2)$ is far outside the band: $@{dep.sp500.r2}$, $@{dep.dax.r2}$, $@{dep.btc.r2}$',
      'Dreapta: $\\text{Corr}(r_t, r_{t-1})$ este mică (S\\&P 500 $@{dep.sp500.r}$, DAX $@{dep.dax.r}$, Bitcoin $@{dep.btc.r}$); $\\text{Corr}(r_t^2, r_{t-1}^2)$ iese mult din bandă: $@{dep.sp500.r2}$, $@{dep.dax.r2}$, $@{dep.btc.r2}$'),
    T('BET: $\\text{Corr}(r_t, r_{t-1}) = @{dep.bet.r}$, a trace of thin trading; the negative S\\&P 500 value comes mostly from crisis days', 'BET: $\\text{Corr}(r_t, r_{t-1}) = @{dep.bet.r}$, o urmă a lichidității reduse; valoarea negativă pentru S\\&P 500 provine mai ales din zilele de criză')], h='0.52\\textheight')

D.recap(('Joint Distributions', 'Distribuții comune'), [
    T('Covariance enters the variance of every portfolio', 'Covarianța intră în varianța oricărui portofoliu'),
    T('Independent implies uncorrelated, not the other way round', 'Independența implică necorelarea, nu invers'),
    T('Returns: nearly uncorrelated, but their squares are correlated, so they are not independent', 'Randamentele: aproape necorelate, dar pătratele lor sînt corelate, deci nu sînt independente')])

# =============================================================================
# 5. SPERANȚA ȘI VARIANȚA CONDIȚIONATĂ
# =============================================================================
D.section('Conditional Expectation and Conditional Variance', 'Speranța condiționată și varianța condiționată')

D.frame(T('Conditional Expectation', 'Speranța condiționată'), items(
    (T('Conditional density: $f(y \\mid x) = f(x, y)/f_X(x)$', 'Densitatea condiționată: $f(y \\mid x) = f(x, y)/f_X(x)$'),
     [T('the distribution of $Y$ once we know that $X = x$', 'distribuția lui $Y$ după ce aflăm că $X = x$')]),
    T('\\textbf{Conditional expectation}: $E[Y \\mid X = x] = \\int y f(y \\mid x)\\,dy$; $E[Y \\mid X]$ is a random variable, a function of $X$',
      '\\textbf{Speranța condiționată}: $E[Y \\mid X = x] = \\int y f(y \\mid x)\\,dy$; $E[Y \\mid X]$ este o variabilă aleatoare, o funcție de $X$'),
    (T('It is the best forecast of $Y$ from $X$: it minimises $E[(Y - g(X))^2]$ over all functions $g$', 'Este cea mai bună prognoză a lui $Y$ pe baza lui $X$: minimizează $E[(Y - g(X))^2]$ dintre toate funcțiile $g$'),
     [T('for returns: $E[r_t \\mid \\mathcal{F}_{t-1}]$, the forecast given all information $\\mathcal{F}_{t-1}$ up to yesterday', 'pentru randamente: $E[r_t \\mid \\mathcal{F}_{t-1}]$, prognoza dată fiind toată informația $\\mathcal{F}_{t-1}$ pînă ieri')]),
    T('\\textbf{Tower property} (law of iterated expectations): $E\\big[E[Y \\mid X]\\big] = E[Y]$', '\\textbf{Proprietatea turnului} (legea speranțelor iterate): $E\\big[E[Y \\mid X]\\big] = E[Y]$')))

D.frame(T('Conditional Variance and the Law of Total Variance', 'Varianța condiționată și legea varianței totale'), items(
    T('\\textbf{Conditional variance}: $\\text{Var}(Y \\mid X) = E\\big[(Y - E[Y \\mid X])^2 \\mid X\\big]$', '\\textbf{Varianța condiționată}: $\\text{Var}(Y \\mid X) = E\\big[(Y - E[Y \\mid X])^2 \\mid X\\big]$'),
    (T('\\textbf{Law of total variance}: $\\text{Var}(Y) = E\\big[\\text{Var}(Y \\mid X)\\big] + \\text{Var}\\big(E[Y \\mid X]\\big)$', '\\textbf{Legea varianței totale}: $\\text{Var}(Y) = E\\big[\\text{Var}(Y \\mid X)\\big] + \\text{Var}\\big(E[Y \\mid X]\\big)$'),
     [T('within-group variance plus between-group variance', 'varianța din interiorul grupurilor plus varianța dintre grupuri')]),
    (T('Worked example, two regimes: calm (probability $0.7$, mean $0.05\\%$, s.d. $0.8\\%$), turbulent ($0.3$, $-0.10\\%$, $2\\%$)',
       'Exemplu lucrat, două regimuri: calm (probabilitatea $0.7$, media $0.05\\%$, abaterea standard $0.8\\%$), agitat ($0.3$, $-0.10\\%$, $2\\%$)'),
     [T('$E[r] = 0.7 \\times 0.05 + 0.3 \\times (-0.10) = @{rg.E}\\%$', '$E[r] = 0.7 \\times 0.05 + 0.3 \\times (-0.10) = @{rg.E}\\%$'),
      T('$E[\\text{Var}] = 0.7 \\times 0.64 + 0.3 \\times 4 = @{rg.w}$; $\\text{Var}(E) = @{rg.b}$', '$E[\\text{Var}] = 0.7 \\times 0.64 + 0.3 \\times 4 = @{rg.w}$; $\\text{Var}(E) = @{rg.b}$'),
      T('$\\text{Var}(r) = @{rg.v}$, s.d. $@{rg.sd}\\%$: almost all risk comes from the changing variance, not from the changing mean',
        '$\\text{Var}(r) = @{rg.v}$, abaterea standard $@{rg.sd}\\%$: aproape tot riscul provine din diferența de varianță între regimuri, nu din diferența de medie')])), 'footnotesize')

chart(T('Today\'s Return Given Yesterday\'s Move', 'Randamentul de azi dată fiind mișcarea de ieri'), 'sfm_ch4_conditional', 'SFM_ch4_dependence', [
    T('Quintiles of $|r_{t-1}|$: Q1 = the calmest 20\\% of yesterdays, Q5 = the most turbulent 20\\%', 'Chintilele lui $|r_{t-1}|$: Q1 = cele mai calme 20\\% dintre zilele de ieri, Q5 = cele mai agitate 20\\%'),
    T('Conditional mean: between $@{cd.sp500.mmin}\\%$ and $@{cd.sp500.mmax}\\%$ for the S\\&P 500, no usable pattern', 'Media condiționată: între $@{cd.sp500.mmin}\\%$ și $@{cd.sp500.mmax}\\%$ pentru S\\&P 500, fără un tipar util'),
    T('Conditional s.d.: S\\&P 500 from @{cd.sp500.sd1}\\% (Q1) to @{cd.sp500.sd5}\\% (Q5), ratio @{cd.sp500.ratio}; Bitcoin ratio @{cd.btc.ratio}',
      'Abaterea standard condiționată: S\\&P 500 de la @{cd.sp500.sd1}\\% (Q1) la @{cd.sp500.sd5}\\% (Q5), raportul Q5/Q1 @{cd.sp500.ratio}; Bitcoin, raportul @{cd.btc.ratio}'),
    T('Law of total variance, S\\&P 500: @{cd.sp500.total} = @{cd.sp500.within} + @{cd.sp500.between}; the between part is @{cd.sp500.share}\\%',
      'Legea varianței totale, S\\&P 500: @{cd.sp500.total} = @{cd.sp500.within} + @{cd.sp500.between}; componenta dintre grupuri reprezintă @{cd.sp500.share}\\% din total')], h='0.50\\textheight')

D.frame(T('From Conditional Variance to GARCH', 'De la varianța condiționată la GARCH'), items(
    (T('Write the return as $r_t = \\mu_t + \\sigma_t z_t$, with $z_t$ i.i.d., mean 0, variance 1', 'Scriem randamentul ca $r_t = \\mu_t + \\sigma_t z_t$, cu $z_t$ i.i.d., media 0, varianța 1'),
     [T('$\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$: the conditional mean, close to constant', '$\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$: media condiționată, aproape constantă'),
      T('$\\sigma_t^2 = \\text{Var}(r_t \\mid \\mathcal{F}_{t-1})$: the conditional variance, which changes with the news', '$\\sigma_t^2 = \\text{Var}(r_t \\mid \\mathcal{F}_{t-1})$: varianța condiționată, care se schimbă cu știrile')]),
    (T('ARCH (autoregressive conditional heteroskedasticity) \\refEngle: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2$', 'ARCH \\refEngle: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2$'),
     [T('GARCH \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2 + \\beta\\sigma_{t-1}^2$ (Chapter 9)', 'GARCH \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2 + \\beta\\sigma_{t-1}^2$ (Capitolul 9)')]),
    T('Exactly the pattern in the chart: a large $|r_{t-1}|$ raises today\'s conditional s.d.', 'Este exact tiparul din grafic: un $|r_{t-1}|$ mare crește abaterea standard condiționată de azi'),
    T('The unconditional distribution is a mixture over $\\sigma_t$: heavier tails than the Normal distribution, even if $z_t$ is Normal', 'Distribuția necondiționată este un amestec peste $\\sigma_t$: cozi mai groase decît distribuția Normală, chiar dacă $z_t$ este Normal')))

D.recap(('Conditioning', 'Condiționare'), [
    T('$E[Y \\mid X]$ is the best forecast; the tower property averages it back to $E[Y]$', '$E[Y \\mid X]$ este cea mai bună prognoză; proprietatea turnului o readuce, în medie, la $E[Y]$'),
    T('Returns: conditional mean almost flat, conditional variance strongly time-varying', 'Randamente: media condiționată aproape constantă, varianța condiționată puternic variabilă în timp'),
    T('Modelling $\\sigma_t^2$ is the job of ARCH and GARCH', 'Modelele ARCH și GARCH descriu dinamica lui $\\sigma_t^2$')])

# =============================================================================
# 6. LEGEA NUMERELOR MARI ȘI CLT
# =============================================================================
D.section('The Law of Large Numbers and the CLT', 'Legea numerelor mari și CLT')

D.frame(T('Two Limit Theorems', 'Două teoreme limită'), items(
    (T('\\textbf{LLN} (law of large numbers): for i.i.d. $X_i$ with $E|X| < \\infty$, $\\bar X_n \\to \\mu$ as $n \\to \\infty$', '\\textbf{LLN} (law of large numbers, legea numerelor mari): pentru $X_i$ i.i.d. cu $E|X| < \\infty$, $\\bar X_n \\to \\mu$ cînd $n \\to \\infty$'),
     [T('the reason why averages, frequencies and Monte Carlo estimates work', 'motivul pentru care funcționează mediile, frecvențele și estimările Monte Carlo')]),
    (T('\\textbf{CLT}: with finite variance, $\\sqrt{n}(\\bar X_n - \\mu)/\\sigma \\to N(0, 1)$ (Chapter 2)', '\\textbf{CLT}: cu varianță finită, $\\sqrt{n}(\\bar X_n - \\mu)/\\sigma \\to N(0, 1)$ (Capitolul 2)'),
     [T('gives the speed: the error of $\\bar X_n$ is about $\\sigma/\\sqrt{n}$', 'dă viteza de convergență: eroarea lui $\\bar X_n$ este de circa $\\sigma/\\sqrt{n}$'),
      T('infinite variance: sums converge to $\\alpha$-stable laws instead (Chapter 3)', 'varianță infinită: sumele converg, în schimb, către legi $\\alpha$-stabile (Capitolul 3)')]),
    T('Both need some form of independence; dependent returns converge more slowly', 'Ambele au nevoie de o formă de independență; randamentele dependente converg mai încet')))

chart(T('The LLN for the Mean Return: Slow', 'LLN pentru randamentul mediu: convergență lentă'), 'sfm_ch4_lln', 'SFM_ch4_monte_carlo', [
    T('S\\&P 500, @{lln.years} years: mean $@{an.m}\\%$ a day, standard error $\\sigma/\\sqrt{n} = @{lln.se}\\%$, $t = @{lln.t}$', 'S\\&P 500, @{lln.years} ani: media $@{an.m}\\%$ pe zi, eroarea standard $\\sigma/\\sqrt{n} = @{lln.se}\\%$, $t = @{lln.t}$'),
    T('Annual mean @{an.ma}\\%, 95\\% interval $[@{lln.lo}\\%, @{lln.hi}\\%]$: even 26 years do not pin down the expected return',
      'Media anuală @{an.ma}\\%, intervalul de 95\\% $[@{lln.lo}\\%; @{lln.hi}\\%]$: nici 26 de ani de date nu ajung pentru a estima precis randamentul așteptat'),
    T('For $t = 3$ at this mean and volatility we would need about @{lln.y3} years of data', 'Pentru $t = 3$, la această medie și volatilitate, ar fi nevoie de circa @{lln.y3} de ani de date')], h='0.56\\textheight')

# =============================================================================
# 7. NUMERE ALEATOARE ȘI MONTE CARLO
# =============================================================================
D.section('Random Numbers and Monte Carlo', 'Numere aleatoare și Monte Carlo')

D.frame(T('The Monte Carlo Idea', 'Ideea Monte Carlo'), items(
    (T('Estimate $\\theta = E[g(X)]$ by $\\hat\\theta_N = \\frac1N\\sum_{i=1}^N g(X_i)$ with simulated $X_i$', 'Estimăm $\\theta = E[g(X)]$ prin $\\hat\\theta_N = \\frac1N\\sum_{i=1}^N g(X_i)$, cu $X_i$ simulate'),
     [T('a probability is an expectation: $P(A) = E[\\mathbf{1}_A]$', 'o probabilitate este o speranță: $P(A) = E[\\mathbf{1}_A]$'),
      T('the LLN makes $\\hat\\theta_N \\to \\theta$; the CLT gives the error $\\text{sd}(g(X))/\\sqrt{N}$', 'LLN face ca $\\hat\\theta_N \\to \\theta$; CLT dă eroarea $\\text{sd}(g(X))/\\sqrt{N}$')]),
    T('Named after the Monaco casino by Ulam and von Neumann, Los Alamos, 1940s \\refMU', 'Ulam și von Neumann au numit metoda după cazinoul din Monaco (Los Alamos, anii 1940) \\refMU'),
    (T('In finance \\refGlasserman', 'În finanțe \\refGlasserman'),
     [T('prices of path-dependent options, VaR and ES of portfolios, stress scenarios', 'prețuri de opțiuni dependente de traiectorie, VaR și ES ale portofoliilor, scenarii de stres'),
      T('any expectation without a closed form', 'orice speranță matematică fără formulă analitică')]),
    T('Two ingredients: uniform random numbers and a way to turn them into any distribution', 'Două ingrediente: numere aleatoare uniforme și o metodă de a le transforma în extrageri din orice distribuție')))

D.frame(T('Pseudo-Random Numbers: Linear Congruential Generators', 'Numere pseudo-aleatoare: generatoare congruențiale liniare'), items(
    (T('\\textbf{LCG} (linear congruential generator): $x_{k+1} = (a x_k + c) \\bmod M$, $u_k = x_k/M \\in [0, 1)$', '\\textbf{LCG} (linear congruential generator, generator congruențial liniar): $x_{k+1} = (a x_k + c) \\bmod M$, $u_k = x_k/M \\in [0, 1)$'),
     [T('deterministic: the same seed $x_0$ gives the same sequence, which makes results reproducible', 'determinist: aceeași sămînță $x_0$ dă același șir, ceea ce face rezultatele reproductibile')]),
    (T('Worked example (SFErangen1): $a = 5$, $c = 1$, $M = 16$, $x_0 = 7$', 'Exemplu lucrat (SFErangen1): $a = 5$, $c = 1$, $M = 16$, $x_0 = 7$'),
     [T('$x$: @{ru.lcg}', '$x$: @{ru.lcg}'),
      T('period @{ru.per}, the maximum $M$; with $a = 4$ the sequence sticks after one step (period @{ru.perbad})', 'perioada @{ru.per}, maximul posibil, $M$; cu $a = 4$ șirul se blochează după un pas (perioada @{ru.perbad})')]),
    T('Real generators use huge $M$; the period must be far longer than any simulation', 'Generatoarele reale folosesc $M$ uriaș; perioada trebuie să fie mult mai lungă decît orice simulare'),
    T('Good uniform marginals are not enough: consecutive numbers must also look independent', 'Nu ajunge ca distribuțiile marginale să fie uniforme: numerele consecutive trebuie să pară și independente')), 'footnotesize')

chart(T('RANDU: Random Numbers Fall Mainly in the Planes (SFErandu)', 'RANDU: numerele aleatoare cad mai ales în plane (SFErandu)'), 'sfm_ch4_randu', 'SFM_ch4_random_numbers', [
    T('RANDU, IBM, 1960s: $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$; pairs look uniform (left), correlation $@{ru.corr}$', 'RANDU, IBM, anii 1960: $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$; perechile par uniforme (stînga), corelația $@{ru.corr}$'),
    T('But $9u_k - 6u_{k+1} + u_{k+2}$ takes only @{ru.nv} integer values, from @{ru.vmin} to @{ru.vmax}: all triples lie on @{ru.nv} planes \\refMarsaglia',
      'Dar $9u_k - 6u_{k+1} + u_{k+2}$ ia doar @{ru.nv} valori întregi, de la $@{ru.vmin}$ la $@{ru.vmax}$: toate tripletele stau pe @{ru.nv} plane \\refMarsaglia'),
    T('Modern generators: Mersenne Twister \\refMT, PCG64 (the NumPy default, right panel)', 'Generatoare moderne: Mersenne Twister \\refMT, PCG64 (implicit în NumPy, panoul din dreapta)')], h='0.52\\textheight')

D.frame(T('The Inverse Transform Method', 'Metoda transformării inverse'), items(
    (T('\\textbf{Theorem}: if $U \\sim U(0, 1)$ and $F$ is a CDF, then $X = F^{-1}(U)$ has CDF $F$', '\\textbf{Teoremă}: dacă $U \\sim U(0, 1)$ și $F$ este o CDF, atunci $X = F^{-1}(U)$ are CDF $F$'),
     [T('proof: $P(X \\le x) = P(F^{-1}(U) \\le x) = P(U \\le F(x)) = F(x)$', 'demonstrație: $P(X \\le x) = P(F^{-1}(U) \\le x) = P(U \\le F(x)) = F(x)$')]),
    (T('Exponential with rate $\\lambda$: $F(x) = 1 - e^{-\\lambda x}$, so $X = -\\ln(1 - U)/\\lambda$', 'Distribuția exponențială cu rata $\\lambda$: $F(x) = 1 - e^{-\\lambda x}$, deci $X = -\\ln(1 - U)/\\lambda$'),
     [T('$\\lambda = 2$, $u = @{iv.u}$: $x = -\\ln(0.1)/2 = @{iv.x}$', '$\\lambda = 2$, $u = @{iv.u}$: $x = -\\ln(0.1)/2 = @{iv.x}$')]),
    T('Normal: $X = \\mu + \\sigma\\,\\Phi^{-1}(U)$', 'Distribuția Normală: $X = \\mu + \\sigma\\,\\Phi^{-1}(U)$'),
    (T('Data: use the empirical quantile function $\\hat F^{-1}$', 'Pentru date reale: folosim funcția cuantilă empirică $\\hat F^{-1}$'),
     [T('this is resampling past days: historical simulation and the bootstrap', 'adică reeșantionăm zilele trecute: simularea istorică și bootstrap-ul')])))

chart(T('Inverse Transform: Exponential, Normal and Historical', 'Transformarea inversă: extrageri exponențiale, Normale și istorice'), 'sfm_ch4_inverse_transform', 'SFM_ch4_random_numbers', [
    T('Left: 200\\,000 draws $-\\ln(1 - U)/2$ match the exponential density; sample mean @{iv.meanE} (theory $1/\\lambda = 0.5$)', 'Stînga: 200\\,000 de extrageri $-\\ln(1 - U)/2$ urmează densitatea exponențială; media de selecție @{iv.meanE} (teoretic $1/\\lambda = 0.5$)'),
    T('Right: the same uniforms through the Normal and the empirical S\\&P 500 quantile functions', 'Dreapta: aceleași numere uniforme trecute prin funcția cuantilă a distribuției Normale și prin cea empirică a S\\&P 500'),
    T('Draws below $-5\\%$: @{iv.p5n}\\% Normal vs @{iv.p5h}\\% historical (data: @{iv.p5d}\\%); worst draw @{iv.minn}\\% vs @{iv.minh}\\%',
      'Extrageri sub $-5\\%$: @{iv.p5n}\\% din cele Normale față de @{iv.p5h}\\% din cele istorice (în date: @{iv.p5d}\\%); cea mai mică extragere: $@{iv.minn}\\%$ față de $@{iv.minh}\\%$')], h='0.52\\textheight')

D.frame(T('Worked Example: a Monthly Loss Probability by Monte Carlo', 'Exemplu lucrat: probabilitatea unei pierderi lunare prin Monte Carlo'), items(
    T('Model: S\\&P 500 daily log returns i.i.d. $N(@{an.m}, @{an.s}^2)$ (in \\%); event: 21-day log return below $-10\\%$', 'Modelul: randamentele logaritmice zilnice ale S\\&P 500 i.i.d. $N(@{an.m}, @{an.s}^2)$ (în \\%); evenimentul: randamentul logaritmic pe 21 de zile sub $-10\\%$'),
    T('Exact: the sum is $N(@{mc.mh}, @{mc.sh}^2)$, so $p = \\Phi(@{mc.z}) = @{mc.p}\\%$', 'Exact: suma este $N(@{mc.mh}, @{mc.sh}^2)$, deci $p = \\Phi(@{mc.z}) = @{mc.p}\\%$'),
    (T('Monte Carlo: $\\hat p_N$ = share of simulated months below $-10\\%$; standard error $\\sqrt{p(1 - p)/N}$', 'Monte Carlo: $\\hat p_N$ = ponderea lunilor simulate sub $-10\\%$; eroarea standard $\\sqrt{p(1 - p)/N}$'),
     [T('$N = 1\\,000$: $@{mc.e1000}\\%$ (s.e. @{mc.s1000}\\%); $N = 100\\,000$: $@{mc.e100000}\\%$ (s.e. @{mc.s100000}\\%)', '$N = 1\\,000$: $@{mc.e1000}\\%$ (eroarea standard @{mc.s1000}\\%); $N = 100\\,000$: $@{mc.e100000}\\%$ (eroarea standard @{mc.s100000}\\%)'),
      T('10 times more precision needs 100 times more simulations', 'o precizie de 10 ori mai mare cere de 100 de ori mai multe simulări')]),
    T('For a relative error of 10\\% (95\\% level) we need $N \\approx 1.96^2(1 - p)/(0.1^2 p) = @{mc.nrel}$', 'Pentru o eroare relativă de 10\\% (la nivelul de încredere de 95\\%) avem nevoie de $N \\approx 1.96^2(1 - p)/(0.1^2 p) = @{mc.nrel}$')), 'footnotesize')

chart(T('Monte Carlo Converges at Rate $1/\\sqrt{N}$', 'Monte Carlo converge cu viteza $1/\\sqrt{N}$'), 'sfm_ch4_monte_carlo', 'SFM_ch4_monte_carlo', [
    T('The estimate wanders for small $N$ and settles inside the 95\\% band, which shrinks like $1/\\sqrt{N}$', 'Pentru $N$ mic estimarea fluctuează mult, apoi intră în banda de 95\\%, care se îngustează ca $1/\\sqrt{N}$'),
    T('$N = 10^6$: $@{mc.e1000000}\\%$ vs exact @{mc.p}\\%, s.e. @{mc.s1000000}\\%', '$N = 10^6$: $@{mc.e1000000}\\%$ față de valoarea exactă @{mc.p}\\%, eroarea standard @{mc.s1000000}\\%'),
    T('Monte Carlo removes the simulation error, not the model error: Seminar 4 (B3) repeats it with resampled days', 'Un $N$ mare reduce eroarea de simulare, nu și eroarea de model: Seminarul 4 (B3) repetă calculul cu zile reeșantionate')], h='0.56\\textheight')

D.recap(('Random Numbers and Monte Carlo', 'Numere aleatoare și Monte Carlo'), [
    T('Pseudo-random generators are deterministic; check them in several dimensions', 'Generatoarele pseudo-aleatoare sînt deterministe; verificați-le în mai multe dimensiuni'),
    T('$F^{-1}(U)$ turns uniforms into any distribution, including the empirical one', '$F^{-1}(U)$ transformă numerele uniforme în orice distribuție, inclusiv cea empirică'),
    T('Monte Carlo error $\\propto 1/\\sqrt{N}$; always report it', 'Eroarea Monte Carlo $\\propto 1/\\sqrt{N}$; raportați-o întotdeauna')])

# =============================================================================
# 8. MODELUL BINOMIAL
# =============================================================================
D.section('The Binomial Model', 'Modelul binomial')

D.frame(T('The Binomial Process', 'Procesul binomial'), items(
    (T('Each period the price moves up by a factor $u$ with probability $p$ or down by $d < u$ with probability $1 - p$, independently',
       'În fiecare perioadă prețul crește cu factorul $u$ cu probabilitatea $p$ sau scade cu factorul $d < u$ cu probabilitatea $1 - p$, independent'),
     [T('after $n$ steps with $K$ up moves: $S_n = S_0 u^K d^{n - K}$, $K \\sim B(n, p)$', 'după $n$ pași cu $K$ creșteri: $S_n = S_0 u^K d^{n - K}$, $K \\sim B(n, p)$')]),
    T('$\\ln S_n = \\ln S_0 + K\\ln u + (n - K)\\ln d$: a random walk in log prices (Section 9)', '$\\ln S_n = \\ln S_0 + K\\ln u + (n - K)\\ln d$: un mers aleator în prețurile logaritmice (secțiunea 9)'),
    (T('$E[S_n] = S_0\\,(pu + (1 - p)d)^n$, because the steps are independent', '$E[S_n] = S_0\\,(pu + (1 - p)d)^n$, pentru că pașii sînt independenți'),
     [T('the basis of the binomial option pricing model \\refCRR; textbook: \\refFHH, Ch.~7', 'baza modelului binomial de evaluare a opțiunilor \\refCRR; manual: \\refFHH, cap.~7')])))

D.frame(T('Worked Example: a Two-Step Tree', 'Exemplu lucrat: un arbore cu doi pași'), items(
    T('$S_0 = 100$, $u = 1.2$, $d = 0.8$, $p = 0.6$', '$S_0 = 100$, $u = 1.2$, $d = 0.8$, $p = 0.6$'),
    (T('Terminal values and probabilities', 'Valorile finale și probabilitățile lor'),
     [T('$uu$: @{t2.uu} with $p^2 = @{t2.puu}$; $ud$ or $du$: @{t2.ud} with $2p(1-p) = @{t2.pud}$; $dd$: @{t2.dd} with $(1-p)^2 = @{t2.pdd}$',
        '$uu$: @{t2.uu} cu $p^2 = @{t2.puu}$; $ud$ sau $du$: @{t2.ud} cu $2p(1-p) = @{t2.pud}$; $dd$: @{t2.dd} cu $(1-p)^2 = @{t2.pdd}$')]),
    T('$E[S_2] = 100 \\times (0.6 \\times 1.2 + 0.4 \\times 0.8)^2 = 100 \\times @{t2.g}^2 = @{t2.E}$', '$E[S_2] = 100 \\times (0.6 \\times 1.2 + 0.4 \\times 0.8)^2 = 100 \\times @{t2.g}^2 = @{t2.E}$'),
    T('$P(S_2 < 100) = @{t2.pud} + @{t2.pdd}$: an up and a down move lose money, because $ud = 0.96 < 1$', '$P(S_2 < 100) = @{t2.pud} + @{t2.pdd}$: o creștere urmată de o scădere (sau invers) aduce o pierdere, pentru că $ud = 0.96 < 1$'),
    T('With $p^* = (1 - d)/(u - d) = @{t2.pstar}$ the price is a fair game, $E[S_{t+1} \\mid S_t] = S_t$ (Section 9)', 'Cu $p^* = (1 - d)/(u - d) = @{t2.pstar}$ prețul este un joc echitabil, $E[S_{t+1} \\mid S_t] = S_t$ (secțiunea 9)')))

D.frame(T('Calibrating the Tree to a Market (Cox--Ross--Rubinstein)', 'Calibrarea arborelui pe o piață (Cox--Ross--Rubinstein)'), items(
    (T('\\textbf{CRR} (Cox--Ross--Rubinstein) with step $\\Delta t$: $u = e^{\\sigma\\sqrt{\\Delta t}}$, $d = 1/u$, $p = (e^{\\mu\\Delta t} - d)/(u - d)$', '\\textbf{CRR} (Cox--Ross--Rubinstein) cu pasul $\\Delta t$: $u = e^{\\sigma\\sqrt{\\Delta t}}$, $d = 1/u$, $p = (e^{\\mu\\Delta t} - d)/(u - d)$'),
     [T('so that the mean and the variance of each step match a market with drift $\\mu$ and volatility $\\sigma$', 'astfel încît media și varianța fiecărui pas să corespundă unei piețe cu tendința $\\mu$ și volatilitatea $\\sigma$')]),
    (T('S\\&P 500, @{y0}--@{y1}: $\\sigma = @{bn.sigma}\\%$, $\\mu = @{bn.mu}\\%$ a year; daily steps, $\\Delta t = 1/252$', 'S\\&P 500, @{y0}--@{y1}: $\\sigma = @{bn.sigma}\\%$, $\\mu = @{bn.mu}\\%$ pe an; pași zilnici, $\\Delta t = 1/252$'),
     [T('$u = @{bn.u}$, $d = @{bn.d}$, $p = @{bn.p}$', '$u = @{bn.u}$, $d = @{bn.d}$, $p = @{bn.p}$')]),
    T('As $\\Delta t \\to 0$, $\\ln S_T$ becomes Normal (CLT for $K$): the tree converges to GBM (Section 10)', 'Cînd $\\Delta t \\to 0$, $\\ln S_T$ devine Normal (CLT pentru $K$): arborele converge către GBM (secțiunea 10)')))

chart(T('Binomial Paths and the Lognormal Limit (SFEBinomp)', 'Traiectorii binomiale și limita lognormală (SFEBinomp)'), 'sfm_ch4_binomial', 'SFM_ch4_binomial', [
    T('Left: 12 paths over one year of daily steps, $S_0 = 100$', 'Stînga: 12 traiectorii pe un an, cu pași zilnici, $S_0 = 100$'),
    T('Right: distribution of $S_T$ with 252 steps; $E[S_T] = @{bn.ES}$ $= 100e^{\\mu}$; 5\\% quantile @{bn.q05_bin} vs lognormal @{bn.q05_ln}',
      'Dreapta: distribuția lui $S_T$ cu 252 de pași; $E[S_T] = @{bn.ES}$ $= 100e^{\\mu}$; cuantila de 5\\%: @{bn.q05_bin}, față de @{bn.q05_ln} în distribuția lognormală'),
    T('$P(S_T < 100)$: @{bn.pl}\\% binomial vs @{bn.pln}\\% lognormal ($S_T = 100$ exactly has a probability mass, which the binomial counts on one side)',
      '$P(S_T < 100)$: @{bn.pl}\\% în modelul binomial, față de @{bn.pln}\\% în cel lognormal (valoarea exactă $S_T = 100$ are probabilitate pozitivă, pe care modelul binomial o plasează de o singură parte)')], h='0.52\\textheight')

D.recap(('The Binomial Model', 'Modelul binomial'), [
    T('Up or down each period: $S_n = S_0 u^K d^{n-K}$ with binomial $K$', 'Sus sau jos în fiecare perioadă: $S_n = S_0 u^K d^{n-K}$ cu $K$ binomial'),
    T('CRR matches $\\mu$ and $\\sigma$; the limit is lognormal', 'CRR reproduce $\\mu$ și $\\sigma$; limita este lognormală'),
    T('A special $p^*$ turns the price into a fair game: the key to option pricing', 'Un $p^*$ special transformă prețul într-un joc echitabil: cheia evaluării opțiunilor')])

# =============================================================================
# 9. PROCESE ÎN TIMP DISCRET
# =============================================================================
D.section('Discrete-Time Stochastic Processes', 'Procese stochastice în timp discret')

D.frame(T('Stochastic Processes and Information', 'Procese stochastice și informație'), items(
    (T('A \\textbf{stochastic process} $\\{X_t\\}$: a family of random variables indexed by time', 'Un \\textbf{proces stochastic} $\\{X_t\\}$: o familie de variabile aleatoare indexate după timp'),
     [T('one realisation = one path; a price series is a single path of the process', 'o realizare = o traiectorie; o serie de prețuri este o singură traiectorie a procesului')]),
    T('$\\mathcal{F}_t$: the information available at time $t$ (past prices, news); it grows with $t$ (a \\textbf{filtration})', '$\\mathcal{F}_t$: informația disponibilă la momentul $t$ (prețuri trecute, știri); crește cu $t$ (o \\textbf{filtrare})'),
    (T('\\textbf{(Weakly) stationary}: $E[X_t]$ and $\\text{Cov}(X_t, X_{t+h})$ do not depend on $t$', '\\textbf{Staționar (slab)}: $E[X_t]$ și $\\text{Cov}(X_t, X_{t+h})$ nu depind de $t$'),
     [T('returns: roughly yes; prices: no', 'randamentele: aproximativ da; prețurile: nu')]),
    T('Textbook: \\refFHH, Ch.~4 and Sec.~11.3', 'Manual: \\refFHH, cap.~4 și secț.~11.3')))

D.frame(T('White Noise and the Random Walk', 'Zgomotul alb și mersul aleator'), items(
    (T('\\textbf{White noise}: $E[\\varepsilon_t] = 0$, $\\text{Var}(\\varepsilon_t) = \\sigma^2$, $\\text{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ for $t \\ne s$', '\\textbf{Zgomot alb}: $E[\\varepsilon_t] = 0$, $\\text{Var}(\\varepsilon_t) = \\sigma^2$, $\\text{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ pentru $t \\ne s$'),
     [T('i.i.d. noise is stronger: the $\\varepsilon_t$ are independent, not only uncorrelated', 'zgomotul i.i.d. este o condiție mai tare: $\\varepsilon_t$ sînt independente, nu doar necorelate')]),
    (T('\\textbf{Random walk}: $S_t = S_{t-1} + \\mu + \\varepsilon_t = S_0 + \\mu t + \\sum_{s=1}^t \\varepsilon_s$', '\\textbf{Mers aleator}: $S_t = S_{t-1} + \\mu + \\varepsilon_t = S_0 + \\mu t + \\sum_{s=1}^t \\varepsilon_s$'),
     [T('$E[S_t] = S_0 + \\mu t$; $\\text{Var}(S_t) = t\\sigma^2$: not stationary', '$E[S_t] = S_0 + \\mu t$; $\\text{Var}(S_t) = t\\sigma^2$: nestaționar'),
      T('simulated with $\\sigma = 1$: variance @{pr.rw10}, @{pr.rw50}, @{pr.rw100} at $t = 10, 50, 100$', 'simulat cu $\\sigma = 1$: varianța @{pr.rw10}, @{pr.rw50}, @{pr.rw100} la $t = 10, 50, 100$')]),
    T('Log prices as a random walk = returns as white noise: the benchmark of Chapter 7 (efficient markets)', 'Dacă prețurile logaritmice urmează un mers aleator, randamentele sînt zgomot alb: ipoteza de referință din Capitolul 7 (piețe eficiente)')))

D.frame(T('Martingales', 'Martingale'), items(
    (T('$\\{X_t\\}$ is a \\textbf{martingale} if $E|X_t| < \\infty$ and $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$', '$\\{X_t\\}$ este un \\textbf{martingal} dacă $E|X_t| < \\infty$ și $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$'),
     [T('a fair game: the best forecast of tomorrow is today', 'un joc echitabil: cea mai bună prognoză pentru mîine este valoarea de azi')]),
    T('A \\textbf{martingale difference}: $\\varepsilon_t = X_t - X_{t-1}$ with $E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = 0$', 'O \\textbf{diferență de martingal}: $\\varepsilon_t = X_t - X_{t-1}$ cu $E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = 0$'),
    (T('Examples', 'Exemple'),
     [T('a random walk without drift; $S_t^2 - t\\sigma^2$ for the same walk; the binomial price under $p^*$', 'un mers aleator fără tendință; $S_t^2 - t\\sigma^2$ pentru același mers; prețul binomial cu $p^*$')]),
    (T('Martingale $\\ne$ i.i.d. increments: the conditional \\textbf{variance} may change, as in GARCH', 'Martingal $\\ne$ creșteri i.i.d.: \\textbf{varianța} condiționată se poate schimba, ca în GARCH'),
     [T('the efficient-market idea in its weak form \\refFama: excess returns are a martingale difference', 'forma slabă a ipotezei pieței eficiente \\refFama: randamentele în exces sînt o diferență de martingal')])))

D.frame(T('The AR(1) Process', 'Procesul AR(1)'), items(
    (T('\\textbf{AR(1)} (autoregressive of order 1): $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ white noise', '\\textbf{AR(1)} (autoregresiv de ordinul 1): $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ zgomot alb'),
     [T('$\\phi = 0$: white noise; $\\phi = 1$, $c = \\mu$: random walk', '$\\phi = 0$: zgomot alb; $\\phi = 1$, $c = \\mu$: mers aleator')]),
    (T('Stationary iff $|\\phi| < 1$; then', 'Staționar dacă și numai dacă $|\\phi| < 1$; atunci'),
     [T('mean $c/(1 - \\phi)$, variance $\\sigma^2/(1 - \\phi^2)$, autocorrelation $\\rho(h) = \\phi^h$', 'media $c/(1 - \\phi)$, varianța $\\sigma^2/(1 - \\phi^2)$, autocorelația $\\rho(h) = \\phi^h$'),
      T('half-life of a shock: $\\ln 0.5/\\ln\\phi$ periods', 'timpul de înjumătățire al unui șoc: $\\ln 0.5/\\ln\\phi$ perioade')]),
    (T('Simulated, $\\sigma = 1$', 'Simulat, $\\sigma = 1$'),
     [T('$\\phi = 0.5$: variance @{pr.05.var} (theory @{pr.05.vth}); $\\rho(5) = @{pr.05.r5}$ (theory @{pr.05.r5th}); half-life @{pr.05.hl}',
        '$\\phi = 0.5$: varianța @{pr.05.var} (teoretic @{pr.05.vth}); $\\rho(5) = @{pr.05.r5}$ (teoretic @{pr.05.r5th}); timpul de înjumătățire @{pr.05.hl}'),
      T('$\\phi = 0.95$: variance @{pr.095.var} (theory @{pr.095.vth}); $\\rho(5) = @{pr.095.r5}$; half-life @{pr.095.hl}', '$\\phi = 0.95$: varianța @{pr.095.var} (teoretic @{pr.095.vth}); $\\rho(5) = @{pr.095.r5}$; timpul de înjumătățire @{pr.095.hl}')]),
    T('Uses: interest rates and spreads (mean reversion); the squared returns behave like a persistent AR, which leads to GARCH', 'Utilizări: dobînzi și spread-uri (revenire la medie); pătratele randamentelor se comportă ca un AR persistent, ceea ce duce la GARCH')), 'footnotesize')

chart(T('White Noise, AR(1) and Random Walk, Same Shocks', 'Zgomot alb, AR(1) și mers aleator, aceleași șocuri'), 'sfm_ch4_processes', 'SFM_ch4_processes', [
    T('The larger $\\phi$, the longer a shock lives: the AR(1) with $\\phi = 0.95$ wanders, but returns to 0', 'Cu cît $\\phi$ este mai mare, cu atît un șoc persistă mai mult: AR(1) cu $\\phi = 0.95$ se abate mult timp de la 0, dar revine'),
    T('The random walk never returns on purpose: shocks are permanent and its variance grows with $t$', 'Mersul aleator nu revine în mod sistematic: șocurile sînt permanente, iar varianța lui crește cu $t$')], h='0.58\\textheight')

chart(T('Question for the Room: Which One Is Real?', 'Întrebare pentru sală: care serie este cea reală?'), 'sfm_ch4_spot_the_real', 'SFM_ch4_processes', [
    T('One column is the S\\&P 500, @{y0}--@{y1}; three are GBM paths (i.i.d. Normal returns) with the same mean and volatility', 'O coloană este S\\&P 500, @{y0}--@{y1}; trei sînt traiectorii GBM (randamente Normale i.i.d.) cu aceeași medie și volatilitate'),
    T('\\textbf{What do you think?} Which column is real, from the prices alone?', '\\textbf{Ce credeți?} Care coloană este reală, doar după prețuri?'),
    T('\\textbf{What do you think?} Which column is real, from the returns?', '\\textbf{Ce credeți?} Care coloană este reală, după randamente?')], h='0.56\\textheight')

D.frame(T('Answer: Column @{sp.real}', 'Răspuns: coloana @{sp.real}'), items(
    T('\\textbf{Answer}: column @{sp.real} is the S\\&P 500', '\\textbf{Răspuns}: coloana @{sp.real} este S\\&P 500'),
    (T('Prices: hard to tell; random walks with drift produce booms and long falls too', 'Prețurile: greu de spus; mersurile aleatoare cu tendință produc și ele avînturi și căderi lungi'),
     [T('this is why \\refBachelier{} and \\refOsborne{} modelled prices as random walks', 'de aceea \\refBachelier{} și \\refOsborne{} au modelat prețurile ca mers aleator')]),
    (T('Returns: obvious; the real series has quiet years and violent clusters (2008, 2020)', 'Randamentele: diferența este evidentă; seria reală are ani liniștiți și episoade violente grupate (2008, 2020)'),
     [T('excess kurtosis @{sp.kr} vs at most @{sp.ks} in the simulations; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{sp.ar}$ vs at most @{sp.as}', 'exces de boltire @{sp.kr} față de cel mult @{sp.ks} în simulări; $\\text{Corr}(r_t^2, r_{t-1}^2) = @{sp.ar}$ față de cel mult @{sp.as}'),
      T('largest absolute daily move @{sp.mr}\\% vs @{sp.ms}\\%', 'cea mai mare mișcare zilnică în valoare absolută @{sp.mr}\\% față de @{sp.ms}\\%')]),
    T('Lesson: a model can fit the price path and still be wrong about risk', 'Lecția: un model poate reproduce traiectoria prețului și totuși să greșească în privința riscului')))

D.recap(('Discrete-Time Processes', 'Procese în timp discret'), [
    T('White noise: uncorrelated; i.i.d. noise: independent; a martingale difference: unpredictable mean', 'Zgomot alb: necorelat; zgomot i.i.d.: independent; diferență de martingal: media imprevizibilă'),
    T('Random walk: permanent shocks, variance $t\\sigma^2$; AR(1) with $|\\phi| < 1$: shocks fade at rate $\\phi^h$', 'Mers aleator: șocuri permanente, varianța $t\\sigma^2$; AR(1) cu $|\\phi| < 1$: șocurile se sting în ritmul $\\phi^h$'),
    T('Prices look like random walks; returns are not i.i.d.', 'Prețurile arată ca mersuri aleatoare; randamentele nu sînt i.i.d.')])

# =============================================================================
# 10. PROCESUL WIENER ȘI GBM
# =============================================================================
D.section('The Wiener Process and Geometric Brownian Motion', 'Procesul Wiener și mișcarea browniană geometrică')

D.frame(T('From Pollen to Prices: Brownian Motion', 'De la polen la prețuri: mișcarea browniană'), cols(items(
    T('1827: the botanist Robert Brown sees pollen particles jiggling in water', '1827: botanistul Robert Brown observă particule de polen care tremură în apă'),
    T('1900: Louis Bachelier models the Paris bond market with the same motion, five years before physics \\refBachelier', '1900: Louis Bachelier modelează piața obligațiunilor de la Paris cu aceeași mișcare, cu cinci ani înaintea lui Einstein \\refBachelier'),
    T('1905: Albert Einstein explains it by molecular collisions \\refEinstein', '1905: Albert Einstein o explică prin ciocniri moleculare \\refEinstein'),
    T('1923: Norbert Wiener builds the rigorous mathematical object \\refWiener', '1923: Norbert Wiener construiește obiectul matematic riguros \\refWiener'),
    T('1959: \\refOsborne{} applies it to log prices: geometric Brownian motion', '1959: \\refOsborne{} o aplică prețurilor logaritmice: mișcarea browniană geometrică')),
    '\\begin{center}\n' + ph('brown', 'Robert Brown (1773--1858)', h='0.20\\textheight') + '\\\\[2mm]\n' +
    ph('bachelier', 'Louis Bachelier (1870--1946)', h='0.20\\textheight') + '\n\\end{center}', wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('The Wiener Process', 'Procesul Wiener'), cols(items(
    (T('$\\{W(t), t \\ge 0\\}$ is a \\textbf{Wiener process} (standard Brownian motion) if', '$\\{W(t), t \\ge 0\\}$ este un \\textbf{proces Wiener} (mișcare browniană standard) dacă'),
     [T('$W(0) = 0$', '$W(0) = 0$'),
      T('increments over disjoint intervals are independent', 'creșterile pe intervale disjuncte sînt independente'),
      T('$W(t) - W(s) \\sim N(0, t - s)$ for $s < t$', '$W(t) - W(s) \\sim N(0, t - s)$ pentru $s < t$'),
      T('the paths are continuous', 'traiectoriile sînt continue')]),
    T('It is the limit of the scaled random walk $W_n(t) = (\\varepsilon_1 + \\dots + \\varepsilon_{\\lfloor nt\\rfloor})/\\sqrt{n}$, $\\varepsilon = \\pm 1$ (Donsker)',
      'Este limita mersului aleator scalat $W_n(t) = (\\varepsilon_1 + \\dots + \\varepsilon_{\\lfloor nt\\rfloor})/\\sqrt{n}$, $\\varepsilon = \\pm 1$ (Donsker)'),
    T('Textbook: \\refFHH, Ch.~5', 'Manual: \\refFHH, cap.~5')),
    ph('wiener', 'Norbert Wiener (1894--1964)', h='0.50\\textheight'), wl='0.64', wr='0.32'), 'footnotesize')

chart(T('From Random Walks to the Wiener Process (SFEWienerProcess)', 'De la mersul aleator la procesul Wiener (SFEWienerProcess)'), 'sfm_ch4_wiener', 'SFM_ch4_wiener_gbm', [
    T('Left: scaled random walks with 10, 100 and 10\\,000 steps; the steps vanish, the randomness stays', 'Stînga: mersuri aleatoare scalate cu 10, 100 și 10\\,000 de pași; treptele dispar, caracterul aleator rămîne'),
    T('Right: 20 Wiener paths; $\\text{Var}(W(t)) = t$, so they spread like $\\pm\\sqrt{t}$; simulated $\\text{Var}(W(1)) = @{w.v1}$; @{w.in1}\\% of $W(1)$ inside $\\pm 1$ (theory @{w.in1th}\\%)',
      'Dreapta: 20 de traiectorii Wiener; $\\text{Var}(W(t)) = t$, deci dispersia crește ca $\\pm\\sqrt{t}$; $\\text{Var}(W(1))$ simulată $= @{w.v1}$; @{w.in1}\\% din $W(1)$ în $\\pm 1$ (teoretic @{w.in1th}\\%)')], h='0.54\\textheight')

D.frame(T('Strange Properties of Wiener Paths', 'Proprietăți neobișnuite ale traiectoriilor Wiener'), items(
    (T('Continuous everywhere, differentiable nowhere', 'Continue peste tot, nicăieri derivabile'),
     [T('$(W(t + h) - W(t))/h \\sim N(0, 1/h)$ explodes as $h \\to 0$', '$(W(t + h) - W(t))/h \\sim N(0, 1/h)$ explodează cînd $h \\to 0$')]),
    (T('\\textbf{Quadratic variation}: $\\sum_k (W(t_{k+1}) - W(t_k))^2 \\to t$ on a finer and finer grid', '\\textbf{Variația pătratică}: $\\sum_k (W(t_{k+1}) - W(t_k))^2 \\to t$ pe o grilă tot mai fină'),
     [T('1000 steps on $[0, 1]$: mean @{w.qv}, s.d. @{w.qvsd} across 10\\,000 paths', '1000 de pași pe $[0, 1]$: media @{w.qv}, abaterea standard @{w.qvsd} pe 10\\,000 de traiectorii'),
      T('the reason why $(dW)^2 = dt$ in It\\^o calculus, and why realised variance measures volatility (Chapter 8)', 'motivul pentru care $(dW)^2 = dt$ în calculul It\\^o și pentru care varianța realizată măsoară volatilitatea (Capitolul 8)')]),
    T('$\\text{Cov}(W(s), W(t)) = \\min(s, t)$: simulated $\\text{Cov}(W(0.5), W(1)) = @{w.cov}$', '$\\text{Cov}(W(s), W(t)) = \\min(s, t)$: $\\text{Cov}(W(0.5), W(1))$ simulată $= @{w.cov}$'),
    T('$W$ is a martingale; so is $W(t)^2 - t$', '$W$ este un martingal; la fel și $W(t)^2 - t$')))

D.frame(T('Geometric Brownian Motion', 'Mișcarea browniană geometrică'), items(
    (T('\\textbf{GBM} (geometric Brownian motion): $dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$', '\\textbf{GBM} (geometric Brownian motion, mișcarea browniană geometrică): $dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$'),
     [T('the relative change $dS/S$ has drift $\\mu$ and volatility $\\sigma$', 'variația relativă $dS/S$ are tendința $\\mu$ și volatilitatea $\\sigma$')]),
    (T('Solution: $S_t = S_0\\exp\\big((\\mu - \\sigma^2/2)t + \\sigma W_t\\big)$', 'Soluția: $S_t = S_0\\exp\\big((\\mu - \\sigma^2/2)t + \\sigma W_t\\big)$'),
     [T('$\\ln(S_t/S_0) \\sim N\\big((\\mu - \\sigma^2/2)t, \\sigma^2 t\\big)$: lognormal prices, i.i.d. Normal log returns', '$\\ln(S_t/S_0) \\sim N\\big((\\mu - \\sigma^2/2)t, \\sigma^2 t\\big)$: prețuri lognormale, randamente logaritmice Normale i.i.d.')]),
    T('Mean $E[S_t] = S_0e^{\\mu t}$; median $S_0e^{(\\mu - \\sigma^2/2)t}$: the gap is the volatility drag', 'Media $E[S_t] = S_0e^{\\mu t}$; mediana $S_0e^{(\\mu - \\sigma^2/2)t}$: diferența este volatility drag'),
    T('Exact simulation on a grid: $S_{t + \\Delta t} = S_t\\exp\\big((\\mu - \\sigma^2/2)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\big)$, $Z \\sim N(0, 1)$', 'Simulare exactă pe o grilă: $S_{t + \\Delta t} = S_t\\exp\\big((\\mu - \\sigma^2/2)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\big)$, $Z \\sim N(0, 1)$'),
    T('The model behind \\refBS{} (Chapter 6 of the textbook)', 'Modelul din spatele formulei \\refBS{} (capitolul 6 din manual)')))

chart(T('GBM Calibrated to the S\\&P 500 (SFEsimGBM)', 'GBM calibrat pe S\\&P 500 (SFEsimGBM)'), 'sfm_ch4_gbm_paths', 'SFM_ch4_wiener_gbm', [
    T('Calibration, @{y0}--@{y1}: $\\sigma = @{g.sigma}\\%$, log drift $\\mu - \\sigma^2/2 = @{g.mulog}\\%$, so $\\mu = @{g.mu}\\%$ a year', 'Calibrare, @{y0}--@{y1}: $\\sigma = @{g.sigma}\\%$, tendința logaritmică $\\mu - \\sigma^2/2 = @{g.mulog}\\%$, deci $\\mu = @{g.mu}\\%$ pe an'),
    T('After 10 years: mean @{g.mean}, median @{g.median}; @{g.below}\\% of the paths end below the mean (theory $\\Phi(\\sigma\\sqrt{T}/2) = @{g.belowth}\\%$)',
      'După 10 ani: media @{g.mean}, mediana @{g.median}; @{g.below}\\% din traiectorii se termină sub medie (teoretic $\\Phi(\\sigma\\sqrt{T}/2) = @{g.belowth}\\%$)'),
    T('Probability of a loss after 10 years: @{g.ploss}\\%', 'Probabilitatea unei pierderi după 10 ani: @{g.ploss}\\%')], h='0.54\\textheight')

chart(T('Real Paths in the GBM Fan', 'Traiectorii reale în evantaiul GBM'), 'sfm_ch4_gbm_fan', 'SFM_ch4_wiener_gbm', [
    T('Each fan: 4000 GBM paths with the drift and volatility of the series itself, started on its first day', 'Fiecare evantai: 4000 de traiectorii GBM cu tendința și volatilitatea seriei respective, pornite din prima zi a seriei'),
    T('S\\&P 500: the path reached the @{f.sp500.minr}\\% quantile on @{f.sp500.mind}, then the median at the end; outside the 90\\% band @{f.sp500.out}\\% of the time',
      'S\\&P 500: traiectoria reală a coborît pînă la cuantila de @{f.sp500.minr}\\% pe @{f.sp500.mind} și a încheiat perioada în jurul medianei; în afara benzii de 90\\% în @{f.sp500.out}\\% din timp'),
    T('BET: outside the band @{f.bet.out}\\% of the time; the 2000--2007 boom and the 2008 crash are not GBM-like; Bitcoin: $\\sigma = @{f.btc.sigma}\\%$ a year makes the fan enormous',
      'BET: în afara benzii @{f.bet.out}\\% din timp; avîntul din 2000--2007 și crahul din 2008 nu seamănă cu GBM; Bitcoin: cu $\\sigma = @{f.btc.sigma}\\%$ pe an, evantaiul este foarte larg')], h='0.54\\textheight')

chart(T('What GBM Misses', 'Limitele modelului GBM'), 'sfm_ch4_gbm_check', 'SFM_ch4_wiener_gbm', [
    T('Excess kurtosis: S\\&P 500 @{ck.sp500.k}, BET @{ck.bet.k}, Bitcoin @{ck.btc.k}; 90\\% of 300 GBM simulations lie within $[@{ck.sp500.klo}, @{ck.sp500.khi}]$',
      'Excesul de boltire: S\\&P 500 @{ck.sp500.k}, BET @{ck.bet.k}, Bitcoin @{ck.btc.k}; 90\\% din 300 de simulări GBM sînt în $[@{ck.sp500.klo}, @{ck.sp500.khi}]$'),
    T('Volatility clustering: $\\text{Corr}(r_t^2, r_{t-1}^2) = @{ck.sp500.a}$, $@{ck.bet.a}$, $@{ck.btc.a}$ vs at most $@{ck.btc.ahi}$ under GBM',
      'Volatility clustering: $\\text{Corr}(r_t^2, r_{t-1}^2) = @{ck.sp500.a}$, $@{ck.bet.a}$, $@{ck.btc.a}$ față de cel mult $@{ck.btc.ahi}$ în GBM'),
    T('Maximum drawdown: S\\&P 500 @{ck.sp500.dd}\\% is ordinary for GBM; BET @{ck.bet.dd}\\% is beyond all but 5\\% of the paths (5\\% quantile @{ck.bet.ddlo}\\%)',
      'Drawdown-ul maxim: $@{ck.sp500.dd}\\%$ la S\\&P 500 este obișnuit în GBM; $@{ck.bet.dd}\\%$ la BET depășește 95\\% din traiectorii (cuantila de 5\\%: $@{ck.bet.ddlo}\\%$)'),
    T('4-sigma days: @{ck.sp500.four} (S\\&P 500) vs at most @{ck.sp500.fourhi} in 95\\% of the GBM paths; crypto has the same problem \\refPeleCrypto',
      'Zile de 4 sigma: @{ck.sp500.four} (S\\&P 500) față de cel mult @{ck.sp500.fourhi} în 95\\% din traiectoriile GBM; criptomonedele au aceeași problemă \\refPeleCrypto')], h='0.46\\textheight', size='scriptsize')

D.recap(('Wiener Process and GBM', 'Procesul Wiener și GBM'), [
    T('Wiener process: independent Normal increments with variance equal to the elapsed time', 'Procesul Wiener: creșteri Normale independente, cu varianța egală cu timpul scurs'),
    T('GBM: lognormal prices, median below mean, the benchmark of option pricing', 'GBM: prețuri lognormale, mediana sub medie, reperul evaluării opțiunilor'),
    T('Real returns have heavy tails and volatility clusters that GBM cannot produce', 'Randamentele reale au cozi groase și volatility clustering, pe care GBM nu le poate genera')])

# =============================================================================
# 11. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

chart(T('An Open Question: Does GBM Get Drawdown Risk Right?', 'O întrebare deschisă: estimează GBM corect riscul de drawdown?'), 'sfm_ch4_drawdown_years', 'SFM_ch4_drawdown_open', [
    T('Drawdown within a calendar year beyond 20\\%: S\\&P 500 in @{dd.sp500.nh} of @{dd.sp500.ny} years (@{dd.sp500.sr}\\%) vs @{dd.sp500.pg}\\% under GBM; BET @{dd.bet.sr}\\% vs @{dd.bet.pg}\\%',
      'Drawdown de peste 20\\% în cursul unui an calendaristic: S\\&P 500 în @{dd.sp500.nh} din @{dd.sp500.ny} ani (@{dd.sp500.sr}\\%) față de @{dd.sp500.pg}\\% în GBM; BET @{dd.bet.sr}\\% față de @{dd.bet.pg}\\%'),
    T('Beyond 40\\%: GBM expects @{dd.sp500.e40} such years for the S\\&P 500 and @{dd.bet.e40} for the BET; both had @{dd.sp500.n40} (2008: @{dd.sp500.w}\\% and @{dd.bet.w}\\%)',
      'Peste 40\\%: GBM prevede, în medie, @{dd.sp500.e40} asemenea ani pentru S\\&P 500 și @{dd.bet.e40} pentru BET; fiecare indice a avut @{dd.sp500.n40} (2008: $@{dd.sp500.w}\\%$ și $@{dd.bet.w}\\%$)'),
    T('Open: does GBM overstate moderate drawdowns and understate extreme ones because calm and crisis years cluster?', 'Întrebare deschisă: supraestimează GBM drawdown-urile moderate și le subestimează pe cele extreme pentru că anii calmi și cei de criză apar grupat?')], h='0.48\\textheight')

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: find studies of drawdown distributions and of their link to volatility clustering, and summarise them', '\\textbf{Literatura}: identificarea și rezumarea studiilor despre distribuția drawdown-urilor și legătura lor cu volatility clustering'),
    T('\\textbf{Code}: draft simulations of yearly drawdowns under GBM, the i.i.d. bootstrap and a GARCH model', '\\textbf{Cod}: o primă versiune a simulărilor drawdown-urilor anuale în GBM, în bootstrap i.i.d. și într-un model GARCH'),
    T('\\textbf{Design}: propose a test that compares the observed number of bad years with the model probability', '\\textbf{Design}: propunerea unui test care compară numărul observat de ani nefavorabili cu probabilitatea din model'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that simulates 10,000 one-year paths of daily S\\&P 500 returns under GBM and under an i.i.d. bootstrap of 2000-2026 returns, and returns the probability of a drawdown beyond 20\\% and 40\\% within the year.}',
        '\\aiprompt{Write Python code that simulates 10,000 one-year paths of daily S\\&P 500 returns under GBM and under an i.i.d. bootstrap of 2000-2026 returns, and returns the probability of a drawdown beyond 20\\% and 40\\% within the year.}')])), 'footnotesize')

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Definitions: drawdown from the running peak within the year, not from the first day of the year', 'Definițiile: drawdown-ul se calculează față de maximul atins pînă atunci în cursul anului, nu față de prima zi a anului'),
    T('Calibration: the same $\\mu$ and $\\sigma$, the same number of trading days a year as the data', 'Calibrarea: aceleași $\\mu$ și $\\sigma$, același număr de zile de tranzacționare pe an ca în date'),
    T('Inference: @{dd.sp500.ny} years is a small sample; compute a binomial $p$-value, not only shares', 'Inferența: @{dd.sp500.ny} de ani reprezintă un eșantion mic; calculați o valoare $p$ binomială, nu doar proporții'),
    T('Mechanism: the i.i.d. bootstrap keeps heavy tails but removes clustering; the difference isolates the effect of clustering', 'Mecanismul: bootstrap-ul i.i.d. păstrează cozile groase, dar elimină clustering-ul; diferența izolează efectul clustering-ului'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Seed', 'Idee de proiect'), items(
    (T('\\textbf{Question}: how wrong is GBM about the probability of a bad year, and why?', '\\textbf{Întrebarea}: cît de mult greșește GBM probabilitatea unui an nefavorabil și de ce?'),
     [T('data: S\\&P 500, DAX, BET and Bitcoin from the course data', 'date: S\\&P 500, DAX, BET și Bitcoin din datele cursului')]),
    (T('Steps', 'Pași'),
     [T('compute within-year maximum drawdowns for every calendar year', 'calculați drawdown-ul maxim din fiecare an calendaristic'),
      T('simulate the same statistic under GBM, the i.i.d. bootstrap and a GARCH(1,1) model (Chapter 9)', 'simulați aceeași statistică în GBM, în bootstrap-ul i.i.d. și într-un model GARCH(1,1) (Capitolul 9)'),
      T('compare the shares beyond 20\\% and 40\\% with binomial tests', 'comparați ponderile peste 20\\% și 40\\% cu teste binomiale')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Rezultat: un tabel, un grafic și un paragraf despre concluziile pe care datele le susțin și pe care nu le susțin'),
    T('Declare any AI use, and list the errors of the AI that you corrected \\refWang', 'Declarați orice folosire a AI și listați erorile AI pe care le-ați corectat \\refWang')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei principale'), items(
    T('Distributions: PMF, PDF, CDF and quantiles; VaR 1\\% is minus a quantile', 'Distribuții: PMF, PDF, CDF și cuantile; VaR 1\\% este minus o cuantilă'),
    T('Covariance drives portfolio risk; uncorrelated returns are not independent', 'Covarianța determină riscul portofoliului; randamentele necorelate nu sînt independente'),
    T('Conditional variance changes with yesterday\'s news: the starting point of GARCH', 'Varianța condiționată se schimbă cu știrile de ieri: punctul de plecare al GARCH'),
    T('Monte Carlo: inverse transform plus the LLN; error $\\propto 1/\\sqrt{N}$', 'Monte Carlo: transformarea inversă plus LLN; eroarea $\\propto 1/\\sqrt{N}$'),
    T('Binomial tree $\\to$ GBM; random walk, martingale, white noise, AR(1)', 'Arborele binomial $\\to$ GBM; mers aleator, martingal, zgomot alb, AR(1)'),
    T('GBM fits price paths roughly, but misses heavy tails, clustering and extreme drawdowns', 'GBM reproduce aproximativ traiectoriile prețurilor, dar nu și cozile groase, volatility clustering și drawdown-urile extreme')))

D.frame(T('Key Formulas', 'Formule de reținut'), table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Variance of a sum', 'Varianța unei sume') + ' & $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}(X, Y)$',
     T('Correlation', 'Corelația') + ' & $\\rho = \\text{Cov}(X, Y)/(\\sigma_X\\sigma_Y)$',
     T('Tower property', 'Proprietatea turnului') + ' & $E[E[Y \\mid X]] = E[Y]$',
     T('Total variance', 'Varianța totală') + ' & $\\text{Var}(Y) = E[\\text{Var}(Y \\mid X)] + \\text{Var}(E[Y \\mid X])$',
     T('Inverse transform', 'Transformarea inversă') + ' & $X = F^{-1}(U)$, $U \\sim U(0, 1)$',
     T('Monte Carlo error', 'Eroarea Monte Carlo') + ' & $\\sqrt{p(1 - p)/N}$',
     T('Random walk', 'Mers aleator') + ' & $S_t = S_{t-1} + \\mu + \\varepsilon_t$, $\\text{Var}(S_t) = t\\sigma^2$',
     'AR(1) & $\\text{Var}(X_t) = \\sigma^2/(1 - \\phi^2)$, $\\rho(h) = \\phi^h$',
     'GBM & $S_t = S_0\\exp((\\mu - \\sigma^2/2)t + \\sigma W_t)$, $E[S_t] = S_0e^{\\mu t}$'],
    size='footnotesize'))

D.frame(T('Check Yourself', 'Verificați-vă'), items(
    (T('\\textbf{Question}: $\\sigma_1 = 2\\%$, $\\sigma_2 = 3\\%$, $\\rho = 0.5$; what is the volatility of the 50/50 portfolio?', '\\textbf{Întrebare}: $\\sigma_1 = 2\\%$, $\\sigma_2 = 3\\%$, $\\rho = 0.5$; cît este volatilitatea portofoliului 50/50?'),
     [T('\\textbf{Answer}: $\\sqrt{0.25 \\times 4 + 0.25 \\times 9 + 2 \\times 0.25 \\times 0.5 \\times 2 \\times 3} = \\sqrt{@{cy.var}} = @{cy.sd}\\%$', '\\textbf{Răspuns}: $\\sqrt{0.25 \\times 4 + 0.25 \\times 9 + 2 \\times 0.25 \\times 0.5 \\times 2 \\times 3} = \\sqrt{@{cy.var}} = @{cy.sd}\\%$')]),
    (T('\\textbf{Question}: GBM with $\\mu = 8\\%$, $\\sigma = 20\\%$, $S_0 = 100$; what are the mean and the median of $S_1$?', '\\textbf{Întrebare}: GBM cu $\\mu = 8\\%$, $\\sigma = 20\\%$, $S_0 = 100$; cît sînt media și mediana lui $S_1$?'),
     [T('\\textbf{Answer}: mean $100e^{0.08} = @{cy.mean}$, median $100e^{0.06} = @{cy.med}$', '\\textbf{Răspuns}: media $100e^{0.08} = @{cy.mean}$, mediana $100e^{0.06} = @{cy.med}$')]),
    (T('\\textbf{Question}: AR(1) with $\\phi = 0.8$ and $\\sigma = 1$; what is its variance?', '\\textbf{Întrebare}: AR(1) cu $\\phi = 0.8$ și $\\sigma = 1$; cît este varianța lui?'),
     [T('\\textbf{Answer}: $1/(1 - 0.64) = @{cy.ar}$', '\\textbf{Răspuns}: $1/(1 - 0.64) = @{cy.ar}$')]),
    T('Next: Chapter 5, heavy tails and extreme value theory', 'Urmează: Capitolul 5, cozi groase și teoria valorilor extreme')))

D.references(BIB, per=10)

if __name__ == '__main__':
    D.write(V)
