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
V.put('t2.ploss', 2 * p2 * (1 - p2) + (1 - p2) ** 2, 2)
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
D.frame(T("Today's Question, Route and Learning Outcomes", 'Întrebarea de azi, traseul și rezultatele învățării'), items(
    (T('\\textbf{Question}: which probability tools do we need to describe returns and to simulate prices?',
       '\\textbf{Întrebarea}: de ce instrumente de probabilitate avem nevoie pentru a descrie randamentele și pentru a simula prețuri?'),
     [T('every later chapter (tails, volatility, VaR, efficiency) is built on them', 'toate capitolele următoare (cozi, volatilitate, VaR, eficiență) se sprijină pe ele')]),
    (T('\\textbf{Route and outcomes}: after this chapter you can', '\\textbf{Traseul și rezultatele învățării}: la finalul capitolului veți putea'),
     [T('compute expectations, variances, covariances and correlations, and the volatility of a portfolio', 'calcula speranțe matematice, varianțe, covarianțe și corelații, precum și volatilitatea unui portofoliu'),
      T('explain why uncorrelated returns can still be dependent, and measure it', 'explica de ce randamente necorelate pot fi totuși dependente și măsura această dependență'),
      T('use conditional expectation and conditional variance, the building blocks of GARCH', 'folosi speranța condiționată și varianța condiționată, elementele de bază ale modelelor GARCH'),
      T('generate random numbers with the inverse transform and estimate a probability by Monte Carlo, with its standard error', 'genera numere aleatoare prin metoda transformării inverse și estima o probabilitate prin Monte Carlo, cu eroarea ei standard'),
      T('define and simulate the binomial model, a random walk, a martingale, white noise, AR(1), the Wiener process and GBM', 'defini și simula modelul binomial, mersul aleator, martingalul, zgomotul alb, AR(1), procesul Wiener și GBM')]),
    T('Real data, @{y0}--@{y1}: S\\&P 500, DAX, BET and Bitcoin (from 2014)', 'Date reale, @{y0}--@{y1}: S\\&P 500, DAX, BET și Bitcoin (din 2014)')))

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
    (T('Axioms, for any event $A \\in \\mathcal{F}$', 'Axiomele, pentru orice eveniment $A \\in \\mathcal{F}$'),
     [T('$P(A) \\ge 0$: no probability is negative', '$P(A) \\ge 0$: nicio probabilitate nu este negativă'),
      T('$P(\\Omega) = 1$: some outcome certainly occurs', '$P(\\Omega) = 1$: unul dintre rezultate are loc sigur'),
      T('$P(\\cup_i A_i) = \\sum_i P(A_i)$ for disjoint $A_1, A_2, \\dots$ (no two can occur together): the probability of ``one of them\'\' is the sum',
        '$P(\\cup_i A_i) = \\sum_i P(A_i)$ pentru $A_1, A_2, \\dots$ disjuncte (oricare două nu pot avea loc simultan): probabilitatea ca unul dintre ele să aibă loc este suma probabilităților')]),
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
    (T('\\textbf{Conditional probability}: $P(A \\mid B) = P(A \\cap B)/P(B)$, for $P(B) > 0$', '\\textbf{Probabilitatea condiționată}: $P(A \\mid B) = P(A \\cap B)/P(B)$, pentru $P(B) > 0$'),
     [T('$A, B$: two events; $A \\cap B$: both occur', '$A, B$: două evenimente; $A \\cap B$: au loc amîndouă'),
      T('in words: among the cases in which $B$ occurs, the share in which $A$ also occurs', 'în cuvinte: dintre cazurile în care are loc $B$, proporția celor în care are loc și $A$')]),
    (T('\\textbf{Bayes\' rule}: $P(B \\mid A) = P(A \\mid B)P(B)/P(A)$', '\\textbf{Regula lui Bayes}: $P(B \\mid A) = P(A \\mid B)P(B)/P(A)$'),
     [T('$P(B)$: the probability of $B$ before we learn $A$; $P(B \\mid A)$: the updated probability once $A$ is observed', '$P(B)$: probabilitatea lui $B$ înainte de a afla $A$; $P(B \\mid A)$: probabilitatea actualizată după ce observăm $A$'),
      T('it reverses the direction of conditioning: from ``signal given crisis\'\' to ``crisis given signal\'\'', 'inversează sensul condiționării: de la „semnal dată fiind criza” la „criză dat fiind semnalul”')]),
    (T('\\textbf{Independent} events: $P(A \\cap B) = P(A)P(B)$, i.e. $P(A \\mid B) = P(A)$', 'Evenimente \\textbf{independente}: $P(A \\cap B) = P(A)P(B)$, adică $P(A \\mid B) = P(A)$'),
     [T('learning that $B$ occurred does not change the probability of $A$', 'faptul că $B$ a avut loc nu schimbă probabilitatea lui $A$')])), 'footnotesize')

D.frame(T('Example: Down Days in a Row', 'Exemplu: zile de scădere consecutive'), items(
    (T('S\\&P 500, @{y0}--@{y1}: $P(\\text{down day}) = @{dn.sp500.p}\\%$', 'S\\&P 500, @{y0}--@{y1}: $P(\\text{zi de scădere}) = @{dn.sp500.p}\\%$'),
     [T('after a down day: $P(\\text{down} \\mid \\text{down yesterday}) = @{dn.sp500.p_cond}\\%$', 'după o zi de scădere: $P(\\text{scădere} \\mid \\text{scădere ieri}) = @{dn.sp500.p_cond}\\%$'),
      T('two down days in a row: @{dn.sp500.p_both}\\% in the data', 'două zile de scădere la rînd: @{dn.sp500.p_both}\\% în date'),
      T('under independence: $P(\\text{down})^2 = @{dn.sp500.p_indep}\\%$, close to the data', 'în ipoteza de independență: $P(\\text{scădere})^2 = @{dn.sp500.p_indep}\\%$, apropiat de date')]),
    (T('BET: @{dn.bet.p_cond}\\% after a down day vs @{dn.bet.p_cond_up}\\% after an up day', 'BET: @{dn.bet.p_cond}\\% după o zi de scădere, față de @{dn.bet.p_cond_up}\\% după o zi de creștere'),
     [T('the sign of the return persists from one day to the next', 'semnul randamentului persistă de la o zi la alta'),
      T('a consequence of thin trading: prices adjust to news over several days', 'o consecință a lichidității reduse: prețurile se ajustează la știri în mai multe zile')])))

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
     [T('\\textbf{PMF} (probability mass function): $p(x_i) = P(X = x_i)$, with $\\sum_i p(x_i) = 1$', '\\textbf{PMF} (probability mass function, funcția de masă de probabilitate): $p(x_i) = P(X = x_i)$, cu $\\sum_i p(x_i) = 1$'),
      T('$p(x_i)$: the probability of the value $x_i$; the probabilities of all values add up to 1', '$p(x_i)$: probabilitatea valorii $x_i$; probabilitățile tuturor valorilor însumează 1')]),
    (T('\\textbf{Continuous}: $P(a < X \\le b) = \\int_a^b f(x)\\,dx$', '\\textbf{Continuă}: $P(a < X \\le b) = \\int_a^b f(x)\\,dx$'),
     [T('the probability of an interval $(a, b]$ is the area under $f$ between $a$ and $b$', 'probabilitatea unui interval $(a, b]$ este aria de sub $f$ între $a$ și $b$'),
      T('$f$ is the \\textbf{PDF} (probability density function): $f \\ge 0$ and the total area $\\int f = 1$', '$f$ este \\textbf{PDF} (probability density function, densitatea de probabilitate): $f \\ge 0$, iar aria totală $\\int f = 1$'),
      T('a single point has no area: $P(X = x) = 0$ for every $x$', 'un singur punct nu are arie: $P(X = x) = 0$ pentru orice $x$')]),
    T('Textbook: \\refFHH, Sec.~3.1', 'Manual: \\refFHH, secț.~3.1')), 'footnotesize')

D.frame(T('The Distribution Function and the Quantile Function', 'Funcția de repartiție și funcția cuantilă'), items(
    (T('\\textbf{CDF} (cumulative distribution function): $F(x) = P(X \\le x)$', '\\textbf{CDF} (cumulative distribution function, funcția de repartiție): $F(x) = P(X \\le x)$'),
     [T('the probability that $X$ does not exceed the threshold $x$', 'probabilitatea ca $X$ să nu depășească pragul $x$'),
      T('non-decreasing, right-continuous, from $F(-\\infty) = 0$ to $F(\\infty) = 1$', 'nedescrescătoare, continuă la dreapta, de la $F(-\\infty) = 0$ la $F(\\infty) = 1$'),
      T('for a continuous $X$ the density is its derivative: $f = F\'$', 'pentru $X$ continuă, densitatea este derivata ei: $f = F\'$')]),
    (T('\\textbf{Quantile} of order $\\alpha$: $q_\\alpha = F^{-1}(\\alpha) = \\inf\\{x : F(x) \\ge \\alpha\\}$', '\\textbf{Cuantila} de ordin $\\alpha$: $q_\\alpha = F^{-1}(\\alpha) = \\inf\\{x : F(x) \\ge \\alpha\\}$'),
     [T('$\\alpha \\in (0, 1)$: a probability; $q_\\alpha$ is the smallest $x$ with $P(X \\le x) \\ge \\alpha$', '$\\alpha \\in (0, 1)$: o probabilitate; $q_\\alpha$ este cel mai mic $x$ pentru care $P(X \\le x) \\ge \\alpha$'),
      T('$F^{-1}$: the quantile function, the inverse of the CDF; $\\inf$: the smallest value of the set', '$F^{-1}$: funcția cuantilă, inversa funcției de repartiție; $\\inf$: cea mai mică valoare a mulțimii'),
      T('median $= q_{0.5}$; the 1\\% quantile is the return exceeded on 99\\% of the days', 'mediana $= q_{0.5}$; cuantila de 1\\% este randamentul depășit în 99\\% din zile')])))

D.frame(T('Value at Risk and the Empirical CDF', 'VaR și funcția de repartiție empirică'), items(
    (T('\\textbf{VaR} (Value at Risk) 1\\%: $\\text{VaR}_{1\\%} = -q_{0.01}$', '\\textbf{VaR} (Value at Risk, valoarea expusă la risc) 1\\%: $\\text{VaR}_{1\\%} = -q_{0.01}$'),
     [T('$q_{0.01}$: the 1\\% quantile of the returns, a negative number', '$q_{0.01}$: cuantila de 1\\% a randamentelor, un număr negativ'),
      T('the minus sign turns it into a loss: the loss exceeded with probability 1\\%', 'semnul minus o transformă în pierdere: pierderea depășită cu probabilitatea 1\\%'),
      T('estimation and backtesting in Chapter 10', 'estimarea și backtesting-ul: Capitolul 10')]),
    (T('The \\textbf{empirical CDF} of a sample: $\\hat F(x) = \\frac1n \\#\\{t : r_t \\le x\\}$', '\\textbf{CDF empirică} a unui eșantion: $\\hat F(x) = \\frac1n \\#\\{t : r_t \\le x\\}$'),
     [T('$r_1, \\dots, r_n$: the observed returns; $n$: the sample size', '$r_1, \\dots, r_n$: randamentele observate; $n$: volumul eșantionului'),
      T('$\\#\\{\\cdot\\}$: the number of days in the set; $\\hat F(x)$ is the share of days with a return at most $x$', '$\\#\\{\\cdot\\}$: numărul de zile din mulțime; $\\hat F(x)$ este proporția zilelor cu randamentul cel mult $x$'),
      T('the hat marks an estimate computed from data', 'notația $\hat{\ }$ indică o estimare calculată din date')])))

chart(T('A Discrete and a Continuous Random Variable', 'O variabilă aleatoare discretă și una continuă'), 'sfm_ch4_random_variables', 'SFM_ch4_random_variables', [
    (T('Left: $K$ = number of up days in a full week of the S\\&P 500 (@{rv.weeks} weeks)', 'Stînga: $K$ = numărul zilelor de creștere dintr-o săptămînă completă a S\\&P 500 (@{rv.weeks} de săptămîni)'),
     [T('compared with $B(5, p)$, the binomial distribution: 5 days, each up with probability $p = @{rv.p}$', 'comparat cu $B(5, p)$, distribuția binomială: 5 zile, fiecare de creștere cu probabilitatea $p = @{rv.p}$'),
      T('mean @{rv.mean} vs @{rv.mth}; variance @{rv.var} vs @{rv.vth}', 'media @{rv.mean} față de @{rv.mth}; varianța @{rv.var} față de @{rv.vth}')]),
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
     [T('$x_i$, $p(x_i)$: the values and their probabilities; $f$: the density', '$x_i$, $p(x_i)$: valorile și probabilitățile lor; $f$: densitatea'),
      T('the centre of mass of the distribution; it exists only if $E|X| < \\infty$ (not for Cauchy, Chapter 3)', 'centrul de greutate al distribuției; există doar dacă $E|X| < \\infty$ (nu și pentru Cauchy, Capitolul 3)')]),
    T('Of a function: $E[g(X)] = \\int g(x) f(x)\\,dx$, without finding the distribution of $g(X)$', 'A unei funcții: $E[g(X)] = \\int g(x) f(x)\\,dx$, fără a afla distribuția lui $g(X)$'),
    (T('\\textbf{Linearity}: $E[aX + bY + c] = aE[X] + bE[Y] + c$, always (no independence needed)', '\\textbf{Liniaritate}: $E[aX + bY + c] = aE[X] + bE[Y] + c$, întotdeauna (fără independență)'),
     [T('$a, b, c$: constants; $X, Y$: any two random variables', '$a, b, c$: constante; $X, Y$: oricare două variabile aleatoare'),
      T('the expected return of a portfolio is the weighted mean of the expected returns', 'randamentul așteptat al unui portofoliu este media ponderată a randamentelor așteptate')]),
    (T('In general $E[g(X)] \\ne g(E[X])$', 'În general, $E[g(X)] \\ne g(E[X])$'),
     [T('Jensen: for a convex $g$, $E[g(X)] \\ge g(E[X])$; for example $E[e^X] > e^{E[X]}$', 'Jensen: pentru $g$ convexă, $E[g(X)] \\ge g(E[X])$; de exemplu, $E[e^X] > e^{E[X]}$'),
      T('the source of the volatility drag of Chapter 1', 'sursa volatility drag din Capitolul 1')])), 'footnotesize')

D.frame(T('Variance and Moments', 'Varianța și momentele'), items(
    (T('\\textbf{Variance}: $\\text{Var}(X) = E[(X - \\mu)^2] = E[X^2] - \\mu^2$', '\\textbf{Varianța}: $\\text{Var}(X) = E[(X - \\mu)^2] = E[X^2] - \\mu^2$'),
     [T('$\\mu = E[X]$; the variance is the mean squared distance from the mean', '$\\mu = E[X]$; varianța este distanța pătratică medie față de medie'),
      T('standard deviation $\\sigma = \\sqrt{\\text{Var}(X)}$, in the units of $X$; for returns it is the volatility', 'abaterea standard $\\sigma = \\sqrt{\\text{Var}(X)}$, în unitățile lui $X$; pentru randamente, este volatilitatea')]),
    (T('$\\text{Var}(aX + b) = a^2\\,\\text{Var}(X)$', '$\\text{Var}(aX + b) = a^2\\,\\text{Var}(X)$'),
     [T('adding a constant $b$ does not change risk; multiplying by $a$ multiplies the variance by $a^2$', 'adăugarea unei constante $b$ nu schimbă riscul; înmulțirea cu $a$ înmulțește varianța cu $a^2$')]),
    (T('Moments: $E[X^k]$ for $k = 1, 2, \\dots$', 'Momentele: $E[X^k]$, pentru $k = 1, 2, \\dots$'),
     [T('skewness $E[(X - \\mu)^3]/\\sigma^3$: the asymmetry of the distribution (0 for the Normal distribution)', 'asimetria $E[(X - \\mu)^3]/\\sigma^3$: lipsa de simetrie a distribuției (0 pentru distribuția Normală)'),
      T('kurtosis $E[(X - \\mu)^4]/\\sigma^4$: equals 3 for the Normal distribution', 'boltirea $E[(X - \\mu)^4]/\\sigma^4$: egală cu 3 pentru distribuția Normală'),
      T('excess kurtosis = kurtosis $-\\,3$; positive for heavy tails (Chapter 2)', 'excesul de boltire = boltirea $-\\,3$; pozitiv pentru cozi groase (Capitolul 2)')])))

D.frame(T('Worked Example: Annual Mean and Volatility', 'Exemplu lucrat: media și volatilitatea anuale'), items(
    (T('S\\&P 500, daily log returns $r_t$', 'S\\&P 500, randamentele logaritmice zilnice $r_t$'),
     [T('daily mean $@{an.m}\\%$, daily standard deviation $\\sigma = @{an.s}\\%$', 'media zilnică $@{an.m}\\%$, abaterea standard zilnică $\\sigma = @{an.s}\\%$'),
      T('$A = @{an.ppy}$: the number of trading days per year', '$A = @{an.ppy}$: numărul de zile de tranzacționare pe an')]),
    (T('Assume i.i.d. daily returns (independent and identically distributed)', 'Presupunem randamente zilnice i.i.d. (independente și identic distribuite)'),
     [T('the annual log return is the sum of the $A$ daily ones: $\\sum_{t=1}^{A} r_t$', 'randamentul logaritmic anual este suma celor $A$ randamente zilnice: $\\sum_{t=1}^{A} r_t$')]),
    (T('Mean: $E[\\sum r_t] = A \\times @{an.m} = @{an.ma}\\%$ a year', 'Media: $E[\\sum r_t] = A \\times @{an.m} = @{an.ma}\\%$ pe an'),
     [T('linearity of expectation', 'din liniaritatea speranței')]),
    (T('Variance: $\\text{Var}(\\sum r_t) = A\\,\\sigma^2$, because the covariances are zero', 'Varianța: $\\text{Var}(\\sum r_t) = A\\,\\sigma^2$, pentru că toate covarianțele sînt zero'),
     [T('annual volatility $\\sigma\\sqrt{A} = @{an.s}\\sqrt{@{an.ppy}} = @{an.sa}\\%$', 'volatilitatea anuală $\\sigma\\sqrt{A} = @{an.s}\\sqrt{@{an.ppy}} = @{an.sa}\\%$'),
      T('the square-root-of-time rule: volatility grows with $\\sqrt{h}$ over $h$ periods', 'regula rădăcinii pătrate a timpului: pe $h$ perioade, volatilitatea crește cu $\\sqrt{h}$')])))

D.frame(T('How Rare Is a 4-Sigma Day? Chebyshev\'s Bound', 'Frecvența zilelor de 4 sigma: marginea lui Cebîșev'), items(
    (T('\\textbf{Chebyshev\'s inequality}: for any $X$ with finite variance, $P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2$', '\\textbf{Inegalitatea lui Cebîșev}: pentru orice $X$ cu varianță finită, $P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2$'),
     [T('$\\mu$, $\\sigma$: the mean and the standard deviation; $k > 1$: the distance from the mean, in standard deviations', '$\\mu$, $\\sigma$: media și abaterea standard; $k > 1$: distanța față de medie, în abateri standard'),
      T('it uses only the mean and the variance, so it holds for every distribution', 'folosește doar media și varianța, deci este valabilă pentru orice distribuție')]),
    (T('$k = 4$, S\\&P 500, @{y0}--@{y1}', '$k = 4$, S\\&P 500, @{y0}--@{y1}'),
     [T('Chebyshev bound: at most @{ch.bound}\\% of the days', 'marginea Cebîșev: cel mult @{ch.bound}\\% din zile'),
      T('Normal distribution: @{ch.normal}\\%', 'distribuția Normală: @{ch.normal}\\%'),
      T('data: @{ch.four} days, @{ch.share}\\%', 'în date: @{ch.four} zile, @{ch.share}\\%')]),
    (T('The data lie between the two', 'Frecvența observată se află între cele două valori'),
     [T('the Normal distribution is far too optimistic; the worst case of Chebyshev is too pessimistic', 'distribuția Normală este mult prea optimistă; cazul cel mai nefavorabil al lui Cebîșev este prea pesimist')])))

D.recap(('Expectation and Variance', 'Speranța și varianța'), [
    T('Expectation is linear; variance scales with the square', 'Speranța este liniară; varianța se scalează cu pătratul'),
    T('For i.i.d. returns the mean grows with $h$ and the volatility with $\\sqrt{h}$', 'Pentru randamente i.i.d., media crește cu $h$, iar volatilitatea cu $\\sqrt{h}$'),
    T('Chebyshev bounds tail probabilities for any distribution', 'Cebîșev mărginește probabilitățile din cozi pentru orice distribuție')])

# =============================================================================
# 4. DISTRIBUȚII COMUNE, COVARIANȚĂ, INDEPENDENȚĂ
# =============================================================================
D.section('Joint Distributions and Dependence', 'Distribuții comune și dependență')

D.frame(T('Joint Distribution and Marginals', 'Distribuția comună și distribuțiile marginale'), items(
    (T('\\textbf{Joint CDF}: $F(x, y) = P(X \\le x, Y \\le y)$', '\\textbf{CDF comună}: $F(x, y) = P(X \\le x, Y \\le y)$'),
     [T('the probability that both variables are below their thresholds at the same time', 'probabilitatea ca ambele variabile să fie simultan sub pragurile lor'),
      T('for two returns: both markets fall below given levels on the same day', 'pentru două randamente: ambele piețe scad sub nivelurile date în aceeași zi')]),
    (T('\\textbf{Marginal} CDF: $F_X(x) = F(x, \\infty)$', 'CDF \\textbf{marginală}: $F_X(x) = F(x, \\infty)$'),
     [T('the distribution of $X$ alone, whatever the value of $Y$', 'distribuția lui $X$ singură, oricare ar fi valoarea lui $Y$')]),
    (T('\\textbf{Joint PDF} $f(x, y)$: $P((X, Y) \\in B) = \\iint_B f(x, y)\\,dx\\,dy$', '\\textbf{PDF comună} $f(x, y)$: $P((X, Y) \\in B) = \\iint_B f(x, y)\\,dx\\,dy$'),
     [T('$B$: a region of the plane; the probability is the volume under $f$ above $B$', '$B$: o regiune din plan; probabilitatea este volumul de sub $f$ deasupra lui $B$'),
      T('marginal density: $f_X(x) = \\int f(x, y)\\,dy$, integrating $Y$ out', 'densitatea marginală: $f_X(x) = \\int f(x, y)\\,dy$, prin integrare după $y$')])))

D.frame(T('Covariance, Correlation and the Variance of a Sum', 'Covarianța, corelația și varianța unei sume'), items(
    (T('\\textbf{Covariance}: $\\text{Cov}(X, Y) = E[(X - \\mu_X)(Y - \\mu_Y)] = E[XY] - \\mu_X\\mu_Y$', '\\textbf{Covarianța}: $\\text{Cov}(X, Y) = E[(X - \\mu_X)(Y - \\mu_Y)] = E[XY] - \\mu_X\\mu_Y$'),
     [T('$\\mu_X$, $\\mu_Y$: the means; positive when $X$ and $Y$ tend to be above their means together', '$\\mu_X$, $\\mu_Y$: mediile; pozitivă cînd $X$ și $Y$ tind să fie simultan peste mediile lor')]),
    (T('\\textbf{Correlation}: $\\rho = \\text{Cov}(X, Y)/(\\sigma_X\\sigma_Y) \\in [-1, 1]$', '\\textbf{Corelația}: $\\rho = \\text{Cov}(X, Y)/(\\sigma_X\\sigma_Y) \\in [-1, 1]$'),
     [T('the covariance divided by the two standard deviations: a number without units', 'covarianța împărțită la cele două abateri standard: un număr fără unitate de măsură'),
      T('measures \\textbf{linear} dependence only; $|\\rho| = 1$ iff $Y = a + bX$', 'măsoară doar dependența \\textbf{liniară}; $|\\rho| = 1$ dacă și numai dacă $Y = a + bX$')]),
    (T('\\textbf{Variance of a sum}: $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}(X, Y)$', '\\textbf{Varianța unei sume}: $\\text{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\text{Cov}(X, Y)$'),
     [T('$a, b$: constants (in a portfolio, the weights); the covariance term can raise or lower the risk', '$a, b$: constante (într-un portofoliu, ponderile); termenul de covarianță poate crește sau reduce riscul'),
      T('many assets: $\\sigma_p^2 = w^\\top \\Sigma w$, with $w$ the vector of weights and $\\Sigma$ the covariance matrix', 'mai multe active: $\\sigma_p^2 = w^\\top \\Sigma w$, cu $w$ vectorul ponderilor și $\\Sigma$ matricea de covarianță'),
      T('$\\sigma_p^2$: the variance of the portfolio; $w^\\top$: the transposed vector', '$\\sigma_p^2$: varianța portofoliului; $w^\\top$: vectorul transpus')])), 'footnotesize')

chart(T('Joint Distributions of Daily Returns', 'Distribuții comune ale randamentelor zilnice'), 'sfm_ch4_joint', 'SFM_ch4_dependence', [
    T('Common trading days only: prices are joined first, then differenced ($n = @{j.n}$ days for S\\&P 500 and DAX)', 'Doar zilele de tranzacționare comune: întîi se unesc prețurile, apoi se calculează randamentele ($n = @{j.n}$ zile pentru S\\&P 500 și DAX)'),
    T('S\\&P 500 and DAX: $\\rho = @{j.corr}$; S\\&P 500 and BET: $\\rho = @{jb.corr}$', 'S\\&P 500 și DAX: $\\rho = @{j.corr}$; S\\&P 500 și BET: $\\rho = @{jb.corr}$'),
    T('The cloud is elliptic in the centre, but extreme days are joint: crises hit all markets together', 'Norul este eliptic în centru, dar zilele extreme apar simultan: crizele lovesc toate piețele deodată')], h='0.56\\textheight')

D.frame(T('Worked Example: a 50/50 Portfolio of S\\&P 500 and DAX', 'Exemplu lucrat: un portofoliu 50/50 din S\\&P 500 și DAX'), items(
    (T('Daily standard deviations: $\\sigma_1 = @{j.s1}\\%$ (S\\&P 500), $\\sigma_2 = @{j.s2}\\%$ (DAX)', 'Abaterile standard zilnice: $\\sigma_1 = @{j.s1}\\%$ (S\\&P 500), $\\sigma_2 = @{j.s2}\\%$ (DAX)'),
     [T('covariance $@{j.cov}$; correlation $@{j.corr}$', 'covarianța $@{j.cov}$; corelația $@{j.corr}$'),
      T('weights $a = b = 0.5$, so $a^2 = b^2 = 0.25$ and $2ab = 0.5$', 'ponderile $a = b = 0.5$, deci $a^2 = b^2 = 0.25$ și $2ab = 0.5$')]),
    (T('$\\sigma_p^2 = 0.25\\sigma_1^2 + 0.25\\sigma_2^2 + 2 \\times 0.25\\,\\text{Cov} = @{j.var_sum} + @{j.cov_term} = @{j.varp}$', '$\\sigma_p^2 = 0.25\\sigma_1^2 + 0.25\\sigma_2^2 + 2 \\times 0.25\\,\\text{Cov} = @{j.var_sum} + @{j.cov_term} = @{j.varp}$'),
     [T('$\\sigma_p = @{j.sd_p}\\%$, below the average volatility $@{j.sd_avg}\\%$: diversification', '$\\sigma_p = @{j.sd_p}\\%$, sub volatilitatea medie $@{j.sd_avg}\\%$: efectul diversificării')]),
    (T('S\\&P 500 and BET, lower correlation: $\\sigma_p = @{jb.sd_p}\\%$ vs average $@{jb.sd_avg}\\%$', 'S\\&P 500 și BET, corelație mai mică: $\\sigma_p = @{jb.sd_p}\\%$ față de media $@{jb.sd_avg}\\%$'),
     [T('the lower the correlation, the larger the reduction in risk', 'cu cît corelația este mai mică, cu atît reducerea riscului este mai mare')]),
    (T('Diversification works only through $\\rho < 1$', 'Diversificarea funcționează doar prin $\\rho < 1$'),
     [T('in a crash correlations rise, exactly when diversification is needed', 'într-un crah corelațiile cresc, exact cînd diversificarea ar fi necesară')])), 'footnotesize')

D.frame(T('Independence and Uncorrelatedness', 'Independență și necorelare'), items(
    (T('$X$ and $Y$ are \\textbf{independent} if $F(x, y) = F_X(x)F_Y(y)$ for all $x, y$', '$X$ și $Y$ sînt \\textbf{independente} dacă $F(x, y) = F_X(x)F_Y(y)$ pentru orice $x, y$'),
     [T('the joint CDF is the product of the marginal CDFs', 'CDF comună este produsul CDF marginale'),
      T('then $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ for \\textbf{all} functions $g, h$ (for example squares or absolute values)', 'atunci $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ pentru \\textbf{toate} funcțiile $g, h$ (de exemplu, pătrate sau valori absolute)')]),
    T('Independent $\\Rightarrow$ uncorrelated ($g, h$ = identity); the converse is false', 'Independente $\\Rightarrow$ necorelate ($g, h$ = identitatea); reciproca este falsă'),
    (T('Example: $X \\sim N(0, 1)$, $Y = X^2$', 'Exemplu: $X \\sim N(0, 1)$, $Y = X^2$'),
     [T('$\\text{Cov}(X, X^2) = E[X^3] = 0$: uncorrelated', '$\\text{Cov}(X, X^2) = E[X^3] = 0$: necorelate'),
      T('but $Y$ is a function of $X$: completely dependent', 'dar $Y$ este o funcție de $X$: complet dependente')]),
    T('Exception: for jointly Normal variables, uncorrelated $\\Leftrightarrow$ independent', 'Excepție: pentru variabile cu distribuție Normală comună (bivariată), necorelate $\\Leftrightarrow$ independente')))

chart(T('Returns Are Uncorrelated, Squared Returns Are Not', 'Randamentele sînt necorelate, pătratele lor nu'), 'sfm_ch4_uncorrelated_dependent', 'SFM_ch4_dependence', [
    T('Left: simulated $Y = X^2$, sample correlation $@{dep.xy}$, yet $Y$ is determined by $X$', 'Stînga: $Y = X^2$ simulat, corelația de selecție $@{dep.xy}$, deși $Y$ este determinat de $X$'),
    (T('Right: $\\text{Corr}(r_t, r_{t-1})$ is small: S\\&P 500 $@{dep.sp500.r}$, DAX $@{dep.dax.r}$, Bitcoin $@{dep.btc.r}$', 'Dreapta: $\\text{Corr}(r_t, r_{t-1})$ este mică: S\\&P 500 $@{dep.sp500.r}$, DAX $@{dep.dax.r}$, Bitcoin $@{dep.btc.r}$'),
     [T('$\\text{Corr}(r_t^2, r_{t-1}^2)$ is far outside the band: $@{dep.sp500.r2}$, $@{dep.dax.r2}$, $@{dep.btc.r2}$', '$\\text{Corr}(r_t^2, r_{t-1}^2)$ iese mult din bandă: $@{dep.sp500.r2}$, $@{dep.dax.r2}$, $@{dep.btc.r2}$'),
      T('band $\\pm 1.96/\\sqrt{n}$: the values compatible with zero correlation at the 5\\% level', 'banda $\\pm 1.96/\\sqrt{n}$: valorile compatibile cu o corelație nulă, la pragul de 5\\%'),
      T('the negative S\\&P 500 value comes mostly from crisis days', 'valoarea negativă pentru S\\&P 500 provine mai ales din zilele de criză')]),
    T('BET: $\\text{Corr}(r_t, r_{t-1}) = @{dep.bet.r}$, an effect of thin trading', 'BET: $\\text{Corr}(r_t, r_{t-1}) = @{dep.bet.r}$, un efect al lichidității reduse')], h='0.46\\textheight', size='scriptsize')

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
     [T('the distribution of $Y$ once we know that $X = x$: the joint density rescaled by the marginal one', 'distribuția lui $Y$ după ce aflăm că $X = x$: densitatea comună rescalată prin cea marginală')]),
    (T('\\textbf{Conditional expectation}: $E[Y \\mid X = x] = \\int y f(y \\mid x)\\,dy$', '\\textbf{Speranța condiționată}: $E[Y \\mid X = x] = \\int y f(y \\mid x)\\,dy$'),
     [T('the mean of $Y$ computed with the conditional density', 'media lui $Y$ calculată cu densitatea condiționată'),
      T('$E[Y \\mid X]$ is a random variable, a function of $X$', '$E[Y \\mid X]$ este o variabilă aleatoare, o funcție de $X$')]),
    (T('It is the best forecast of $Y$ from $X$: it minimises $E[(Y - g(X))^2]$ over all functions $g$', 'Este cea mai bună prognoză a lui $Y$ pe baza lui $X$: minimizează $E[(Y - g(X))^2]$ dintre toate funcțiile $g$'),
     [T('$E[(Y - g(X))^2]$: the mean squared error of the forecast $g(X)$', '$E[(Y - g(X))^2]$: eroarea pătratică medie a prognozei $g(X)$'),
      T('for returns: $E[r_t \\mid \\mathcal{F}_{t-1}]$, the forecast given all information $\\mathcal{F}_{t-1}$ up to yesterday', 'pentru randamente: $E[r_t \\mid \\mathcal{F}_{t-1}]$, prognoza dată fiind toată informația $\\mathcal{F}_{t-1}$ pînă ieri')]),
    (T('\\textbf{Tower property} (law of iterated expectations): $E\\big[E[Y \\mid X]\\big] = E[Y]$', '\\textbf{Proprietatea turnului} (legea speranțelor iterate): $E\\big[E[Y \\mid X]\\big] = E[Y]$'),
     [T('averaging the conditional forecasts over all values of $X$ gives back the unconditional mean', 'media prognozelor condiționate, pe toate valorile lui $X$, este media necondiționată')])), 'footnotesize')

D.frame(T('Conditional Variance and the Law of Total Variance', 'Varianța condiționată și legea varianței totale'), items(
    (T('\\textbf{Conditional variance}: $\\text{Var}(Y \\mid X) = E\\big[(Y - E[Y \\mid X])^2 \\mid X\\big]$', '\\textbf{Varianța condiționată}: $\\text{Var}(Y \\mid X) = E\\big[(Y - E[Y \\mid X])^2 \\mid X\\big]$'),
     [T('the spread of $Y$ around its conditional mean, once $X$ is known', 'dispersia lui $Y$ în jurul mediei condiționate, după ce $X$ este cunoscut')]),
    (T('\\textbf{Law of total variance}: $\\text{Var}(Y) = E\\big[\\text{Var}(Y \\mid X)\\big] + \\text{Var}\\big(E[Y \\mid X]\\big)$', '\\textbf{Legea varianței totale}: $\\text{Var}(Y) = E\\big[\\text{Var}(Y \\mid X)\\big] + \\text{Var}\\big(E[Y \\mid X]\\big)$'),
     [T('$E[\\text{Var}(Y \\mid X)]$: the average variance inside the groups defined by $X$ (within)', '$E[\\text{Var}(Y \\mid X)]$: varianța medie din interiorul grupurilor definite de $X$'),
      T('$\\text{Var}(E[Y \\mid X])$: how much the group means differ from each other (between)', '$\\text{Var}(E[Y \\mid X])$: cît de mult diferă mediile grupurilor între ele'),
      T('total risk = within-group variance + between-group variance', 'riscul total = varianța din interiorul grupurilor + varianța dintre grupuri')])))

D.frame(T('Worked Example: Two Regimes', 'Exemplu lucrat: două regimuri'), items(
    (T('$X$ = the market regime, $Y = r$ = the daily return (in \\%)', '$X$ = regimul pieței, $Y = r$ = randamentul zilnic (în \\%)'),
     [T('calm: probability $0.7$, mean $0.05\\%$, standard deviation $0.8\\%$', 'calm: probabilitatea $0.7$, media $0.05\\%$, abaterea standard $0.8\\%$'),
      T('turbulent: probability $0.3$, mean $-0.10\\%$, standard deviation $2\\%$', 'agitat: probabilitatea $0.3$, media $-0.10\\%$, abaterea standard $2\\%$')]),
    (T('Mean (tower property): $E[r] = 0.7 \\times 0.05 + 0.3 \\times (-0.10) = @{rg.E}\\%$', 'Media (proprietatea turnului): $E[r] = 0.7 \\times 0.05 + 0.3 \\times (-0.10) = @{rg.E}\\%$')),
    (T('The two parts of the variance', 'Cele două componente ale varianței'),
     [T('within: $E[\\text{Var}(r \\mid X)] = 0.7 \\times 0.8^2 + 0.3 \\times 2^2 = @{rg.w}$', 'în interiorul regimurilor: $E[\\text{Var}(r \\mid X)] = 0.7 \\times 0.8^2 + 0.3 \\times 2^2 = @{rg.w}$'),
      T('between: $\\text{Var}(E[r \\mid X]) = 0.7(0.05 - E[r])^2 + 0.3(-0.10 - E[r])^2 = @{rg.b}$', 'între regimuri: $\\text{Var}(E[r \\mid X]) = 0.7(0.05 - E[r])^2 + 0.3(-0.10 - E[r])^2 = @{rg.b}$')]),
    (T('$\\text{Var}(r) = @{rg.v}$, standard deviation $@{rg.sd}\\%$', '$\\text{Var}(r) = @{rg.v}$, abaterea standard $@{rg.sd}\\%$'),
     [T('almost all risk comes from the different variances of the regimes, not from their different means', 'aproape tot riscul provine din diferența de varianță dintre regimuri, nu din diferența de medie')])), 'footnotesize')

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
    (T('ARCH (autoregressive conditional heteroskedasticity) \\refEngle: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2$', 'ARCH (autoregressive conditional heteroskedasticity) \\refEngle: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2$'),
     [T('today\'s variance is a base level plus a share of yesterday\'s squared return', 'varianța de azi este un nivel de bază plus o parte din pătratul randamentului de ieri'),
      T('$\\omega > 0$: the base level; $\\alpha \\ge 0$: how strongly a large move yesterday raises today\'s variance', '$\\omega > 0$: nivelul de bază; $\\alpha \\ge 0$: cît de mult crește varianța de azi după o mișcare mare ieri')]),
    (T('GARCH \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2 + \\beta\\sigma_{t-1}^2$ (Chapter 9)', 'GARCH \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha r_{t-1}^2 + \\beta\\sigma_{t-1}^2$ (Capitolul 9)'),
     [T('$\\beta \\ge 0$: the persistence; a large $\\beta$ keeps a high variance high for many days', '$\\beta \\ge 0$: persistența; un $\\beta$ mare menține mult timp o varianță ridicată')]),
    T('Exactly the pattern in the chart: a large $|r_{t-1}|$ raises today\'s conditional s.d.', 'Este exact tiparul din grafic: un $|r_{t-1}|$ mare crește abaterea standard condiționată de azi'),
    (T('The unconditional distribution is a mixture over $\\sigma_t$', 'Distribuția necondiționată este un amestec de distribuții cu diferite $\\sigma_t$'),
     [T('it has heavier tails than the Normal distribution, even if $z_t$ is Normal', 'are cozi mai groase decît distribuția Normală, chiar dacă $z_t$ are distribuție Normală')])), 'footnotesize')

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
     [T('$\\bar X_n = \\frac1n\\sum_{i=1}^n X_i$: the sample mean; $\\mu = E[X]$: the true mean', '$\\bar X_n = \\frac1n\\sum_{i=1}^n X_i$: media de selecție; $\\mu = E[X]$: media adevărată'),
      T('the reason why averages, frequencies and Monte Carlo estimates work', 'motivul pentru care funcționează mediile, frecvențele și estimările Monte Carlo')]),
    (T('\\textbf{CLT} (central limit theorem): with finite variance, $\\sqrt{n}(\\bar X_n - \\mu)/\\sigma \\to N(0, 1)$ (Chapter 2)', '\\textbf{CLT} (central limit theorem, teorema limită centrală): cu varianță finită, $\\sqrt{n}(\\bar X_n - \\mu)/\\sigma \\to N(0, 1)$ (Capitolul 2)'),
     [T('the standardised sample mean has, for large $n$, approximately the standard Normal distribution', 'media de selecție standardizată are, pentru $n$ mare, aproximativ distribuția Normală standard'),
      T('gives the speed: the error of $\\bar X_n$ is about $\\sigma/\\sqrt{n}$', 'dă viteza de convergență: eroarea lui $\\bar X_n$ este de circa $\\sigma/\\sqrt{n}$'),
      T('infinite variance: sums converge to $\\alpha$-stable laws instead (Chapter 3)', 'varianță infinită: sumele converg, în schimb, către legi $\\alpha$-stabile (Capitolul 3)')]),
    T('Both need some form of independence; dependent returns converge more slowly', 'Ambele au nevoie de o formă de independență; randamentele dependente converg mai încet')))

chart(T('The LLN for the Mean Return: Slow', 'LLN pentru randamentul mediu: convergență lentă'), 'sfm_ch4_lln', 'SFM_ch4_monte_carlo', [
    (T('S\\&P 500, @{lln.years} years: mean $@{an.m}\\%$ a day, standard error $\\sigma/\\sqrt{n} = @{lln.se}\\%$', 'S\\&P 500, @{lln.years} ani: media $@{an.m}\\%$ pe zi, eroarea standard $\\sigma/\\sqrt{n} = @{lln.se}\\%$'),
     [T('$t$ = mean / standard error $= @{lln.t}$; $|t| > 1.96$ would mean a mean significantly different from 0 at 5\\%', '$t$ = media / eroarea standard $= @{lln.t}$; $|t| > 1.96$ ar indica o medie semnificativ diferită de 0, la pragul de 5\\%')]),
    (T('Annual mean @{an.ma}\\%, 95\\% interval $[@{lln.lo}\\%, @{lln.hi}\\%]$', 'Media anuală @{an.ma}\\%, intervalul de încredere de 95\\%: $[@{lln.lo}\\%; @{lln.hi}\\%]$'),
     [T('even 26 years do not pin down the expected return', 'nici 26 de ani de date nu ajung pentru a estima precis randamentul așteptat'),
      T('for $t = 3$ at this mean and volatility we would need about @{lln.y3} years of data', 'pentru $t = 3$, la această medie și volatilitate, ar fi nevoie de circa @{lln.y3} de ani de date')])], h='0.50\\textheight')

# =============================================================================
# 7. NUMERE ALEATOARE ȘI MONTE CARLO
# =============================================================================
D.section('Random Numbers and Monte Carlo', 'Numere aleatoare și Monte Carlo')

D.frame(T('The Monte Carlo Idea', 'Ideea Monte Carlo'), items(
    (T('Estimate $\\theta = E[g(X)]$ by $\\hat\\theta_N = \\frac1N\\sum_{i=1}^N g(X_i)$ with simulated $X_i$', 'Estimăm $\\theta = E[g(X)]$ prin $\\hat\\theta_N = \\frac1N\\sum_{i=1}^N g(X_i)$, cu $X_i$ simulate'),
     [T('$\\theta$: the unknown quantity; $g$: a function of the simulated variable; $N$: the number of simulations', '$\\theta$: mărimea necunoscută; $g$: o funcție de variabila simulată; $N$: numărul de simulări'),
      T('a probability is an expectation: $P(A) = E[\\mathbf{1}_A]$, with $\\mathbf{1}_A = 1$ if $A$ occurs and 0 otherwise', 'o probabilitate este o speranță: $P(A) = E[\\mathbf{1}_A]$, cu $\\mathbf{1}_A = 1$ dacă $A$ are loc și 0 altfel'),
      T('the LLN makes $\\hat\\theta_N \\to \\theta$; the CLT gives the error $\\text{sd}(g(X))/\\sqrt{N}$', 'LLN face ca $\\hat\\theta_N \\to \\theta$; CLT dă eroarea $\\text{sd}(g(X))/\\sqrt{N}$, unde $\\text{sd}$ este abaterea standard')]),
    T('Named after the Monaco casino by Ulam and von Neumann, Los Alamos, 1940s \\refMU', 'Ulam și von Neumann au numit metoda după cazinoul din Monaco (Los Alamos, anii 1940) \\refMU'),
    (T('In finance \\refGlasserman', 'În finanțe \\refGlasserman'),
     [T('prices of path-dependent options, portfolio VaR (Chapter 10), stress scenarios', 'prețuri de opțiuni dependente de traiectorie, VaR-ul portofoliilor (Capitolul 10), scenarii de stres'),
      T('any expectation without a closed form', 'orice speranță matematică fără formulă analitică')]),
    T('Two ingredients: uniform random numbers and a way to turn them into any distribution', 'Două ingrediente: numere aleatoare uniforme și o metodă de a le transforma în extrageri din orice distribuție')), 'footnotesize')

D.frame(T('Pseudo-Random Numbers: Linear Congruential Generators', 'Numere pseudo-aleatoare: generatoare congruențiale liniare'), items(
    (T('\\textbf{LCG} (linear congruential generator): $x_{k+1} = (a x_k + c) \\bmod M$, $u_k = x_k/M \\in [0, 1)$', '\\textbf{LCG} (linear congruential generator, generator congruențial liniar): $x_{k+1} = (a x_k + c) \\bmod M$, $u_k = x_k/M \\in [0, 1)$'),
     [T('each integer $x_{k+1}$ is computed from the previous one $x_k$', 'fiecare număr întreg $x_{k+1}$ se calculează din precedentul, $x_k$'),
      T('$a$: the multiplier; $c$: the increment; $M$: the modulus; $\\bmod M$: the remainder of the division by $M$', '$a$: multiplicatorul; $c$: incrementul; $M$: modulul; $\\bmod M$: restul împărțirii la $M$'),
      T('$u_k$: the uniform number in $[0, 1)$ obtained by dividing by $M$', '$u_k$: numărul uniform din $[0, 1)$ obținut prin împărțirea la $M$')]),
    (T('Deterministic: the same seed $x_0$ gives the same sequence', 'Determinist: aceeași sămînță $x_0$ dă același șir'),
     [T('this makes results reproducible', 'acest lucru face rezultatele reproductibile'),
      T('the \\textbf{period}: the number of values before the sequence repeats; at most $M$', '\\textbf{perioada}: numărul de valori după care șirul se repetă; cel mult $M$')]),
    T('Real generators use huge $M$; the period must be far longer than any simulation', 'Generatoarele reale folosesc $M$ uriaș; perioada trebuie să fie mult mai lungă decît orice simulare')))

D.frame(T('Worked Example: an LCG with $M = 16$', 'Exemplu lucrat: un LCG cu $M = 16$'), items(
    (T('SFErangen1: $a = 5$, $c = 1$, $M = 16$, $x_0 = 7$', 'SFErangen1: $a = 5$, $c = 1$, $M = 16$, $x_0 = 7$'),
     [T('first step: $x_1 = (5 \\times 7 + 1) \\bmod 16 = 36 \\bmod 16 = 4$', 'primul pas: $x_1 = (5 \\times 7 + 1) \\bmod 16 = 36 \\bmod 16 = 4$'),
      T('the sequence $x$: @{ru.lcg}', 'șirul $x$: @{ru.lcg}')]),
    (T('Period @{ru.per}, the maximum possible, $M$', 'Perioada @{ru.per}, maximul posibil, $M$'),
     [T('with $a = 4$ the sequence sticks after one step (period @{ru.perbad})', 'cu $a = 4$ șirul se blochează după un pas (perioada @{ru.perbad})')]),
    (T('Good uniform marginals are not enough', 'Nu ajunge ca distribuțiile marginale să fie uniforme'),
     [T('consecutive numbers must also look independent: the next chart', 'numerele consecutive trebuie să pară și independente: graficul următor')])))

chart(T('RANDU: Random Numbers Fall Mainly in the Planes (SFErandu)', 'RANDU: numerele aleatoare cad mai ales în plane (SFErandu)'), 'sfm_ch4_randu', 'SFM_ch4_random_numbers', [
    T('RANDU, IBM, 1960s: $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$; pairs look uniform (left), correlation $@{ru.corr}$', 'RANDU, IBM, anii 1960: $x_{k+1} = 65539\\,x_k \\bmod 2^{31}$; perechile par uniforme (stînga), corelația $@{ru.corr}$'),
    T('But $9u_k - 6u_{k+1} + u_{k+2}$ takes only @{ru.nv} integer values, from @{ru.vmin} to @{ru.vmax}: all triples lie on @{ru.nv} planes \\refMarsaglia',
      'Dar $9u_k - 6u_{k+1} + u_{k+2}$ ia doar @{ru.nv} valori întregi, de la $@{ru.vmin}$ la $@{ru.vmax}$: toate tripletele stau pe @{ru.nv} plane \\refMarsaglia'),
    T('Modern generators: Mersenne Twister \\refMT, PCG64 (the NumPy default, right panel)', 'Generatoare moderne: Mersenne Twister \\refMT, PCG64 (implicit în NumPy, panoul din dreapta)')], h='0.52\\textheight')

D.frame(T('The Inverse Transform Method', 'Metoda transformării inverse'), items(
    (T('\\textbf{Theorem}: if $U \\sim U(0, 1)$ and $F$ is a CDF, then $X = F^{-1}(U)$ has CDF $F$', '\\textbf{Teoremă}: dacă $U \\sim U(0, 1)$ și $F$ este o CDF, atunci $X = F^{-1}(U)$ are CDF $F$'),
     [T('$U(0, 1)$: the uniform distribution on $[0, 1]$, so $P(U \\le u) = u$', '$U(0, 1)$: distribuția uniformă pe $[0, 1]$, deci $P(U \\le u) = u$'),
      T('in words: a uniform number, read as a probability, is turned into the quantile of that order', 'în cuvinte: un număr uniform, interpretat ca probabilitate, este transformat în cuantila de acel ordin'),
      T('proof: $P(X \\le x) = P(F^{-1}(U) \\le x) = P(U \\le F(x)) = F(x)$', 'demonstrație: $P(X \\le x) = P(F^{-1}(U) \\le x) = P(U \\le F(x)) = F(x)$')]),
    (T('Exponential with rate $\\lambda > 0$: $F(x) = 1 - e^{-\\lambda x}$, so $X = -\\ln(1 - U)/\\lambda$', 'Distribuția exponențială cu rata $\\lambda > 0$: $F(x) = 1 - e^{-\\lambda x}$, deci $X = -\\ln(1 - U)/\\lambda$'),
     [T('obtained by solving $u = F(x)$ for $x$; the mean is $1/\\lambda$', 'se obține rezolvînd $u = F(x)$ în raport cu $x$; media este $1/\\lambda$'),
      T('$\\lambda = 2$, $u = @{iv.u}$: $x = -\\ln(0.1)/2 = @{iv.x}$', '$\\lambda = 2$, $u = @{iv.u}$: $x = -\\ln(0.1)/2 = @{iv.x}$')]),
    (T('Normal: $X = \\mu + \\sigma\\,\\Phi^{-1}(U)$', 'Distribuția Normală: $X = \\mu + \\sigma\\,\\Phi^{-1}(U)$'),
     [T('$\\Phi^{-1}$: the quantile function of $N(0, 1)$; $\\mu$, $\\sigma$: the desired mean and standard deviation', '$\\Phi^{-1}$: funcția cuantilă a distribuției $N(0, 1)$; $\\mu$, $\\sigma$: media și abaterea standard dorite')]),
    (T('Data: use the empirical quantile function $\\hat F^{-1}$', 'Pentru date reale: folosim funcția cuantilă empirică $\\hat F^{-1}$'),
     [T('this is resampling past days: historical simulation and the bootstrap', 'adică reeșantionăm zilele trecute: simularea istorică și bootstrap-ul')])), 'footnotesize')

chart(T('Inverse Transform: Exponential, Normal and Historical', 'Transformarea inversă: extrageri exponențiale, Normale și istorice'), 'sfm_ch4_inverse_transform', 'SFM_ch4_random_numbers', [
    T('Left: 200\\,000 draws $-\\ln(1 - U)/2$ match the exponential density; sample mean @{iv.meanE} (theory $1/\\lambda = 0.5$)', 'Stînga: 200\\,000 de extrageri $-\\ln(1 - U)/2$ urmează densitatea exponențială; media de selecție @{iv.meanE} (teoretic $1/\\lambda = 0.5$)'),
    T('Right: the same uniforms through the Normal and the empirical S\\&P 500 quantile functions', 'Dreapta: aceleași numere uniforme trecute prin funcția cuantilă a distribuției Normale și prin cea empirică a S\\&P 500'),
    T('Draws below $-5\\%$: @{iv.p5n}\\% Normal vs @{iv.p5h}\\% historical (data: @{iv.p5d}\\%); worst draw @{iv.minn}\\% vs @{iv.minh}\\%',
      'Extrageri sub $-5\\%$: @{iv.p5n}\\% din cele Normale față de @{iv.p5h}\\% din cele istorice (în date: @{iv.p5d}\\%); cea mai mică extragere: $@{iv.minn}\\%$ față de $@{iv.minh}\\%$')], h='0.52\\textheight')

D.frame(T('Worked Example: a Monthly Loss Probability by Monte Carlo (1/2)', 'Exemplu lucrat: probabilitatea unei pierderi lunare prin Monte Carlo (1/2)'), items(
    (T('Model: S\\&P 500 daily log returns i.i.d. $N(@{an.m}, @{an.s}^2)$, in \\%', 'Modelul: randamentele logaritmice zilnice ale S\\&P 500, i.i.d. $N(@{an.m}, @{an.s}^2)$, în \\%'),
     [T('$N(m, s^2)$: the Normal distribution with mean $m$ and variance $s^2$', '$N(m, s^2)$: distribuția Normală cu media $m$ și varianța $s^2$')]),
    (T('Event: the 21-day log return (one month) is below $-10\\%$', 'Evenimentul: randamentul logaritmic pe 21 de zile (o lună) este sub $-10\\%$'),
     [T('$p$: its probability, the quantity we want', '$p$: probabilitatea lui, mărimea căutată')]),
    (T('Exact: the sum of 21 i.i.d. Normal returns is $N(@{mc.mh}, @{mc.sh}^2)$', 'Exact: suma a 21 de randamente Normale i.i.d. este $N(@{mc.mh}, @{mc.sh}^2)$'),
     [T('mean $21 \\times @{an.m}$, standard deviation $@{an.s}\\sqrt{21}$', 'media $21 \\times @{an.m}$, abaterea standard $@{an.s}\\sqrt{21}$'),
      T('standardise: $z = (-10 - @{mc.mh})/@{mc.sh} = @{mc.z}$', 'standardizăm: $z = (-10 - @{mc.mh})/@{mc.sh} = @{mc.z}$'),
      T('$p = \\Phi(z) = @{mc.p}\\%$, with $\\Phi$ the CDF of $N(0, 1)$', '$p = \\Phi(z) = @{mc.p}\\%$, cu $\\Phi$ funcția de repartiție a distribuției $N(0, 1)$')])))

D.frame(T('Worked Example: a Monthly Loss Probability by Monte Carlo (2/2)', 'Exemplu lucrat: probabilitatea unei pierderi lunare prin Monte Carlo (2/2)'), items(
    (T('Monte Carlo: simulate $N$ months; $\\hat p_N$ = share of simulated months below $-10\\%$', 'Monte Carlo: simulăm $N$ luni; $\\hat p_N$ = ponderea lunilor simulate sub $-10\\%$'),
     [T('standard error $\\sqrt{p(1 - p)/N}$: the typical distance between $\\hat p_N$ and $p$', 'eroarea standard $\\sqrt{p(1 - p)/N}$: distanța tipică dintre $\\hat p_N$ și $p$')]),
    (T('Results', 'Rezultate'),
     [T('$N = 1\\,000$: $@{mc.e1000}\\%$ (s.e. @{mc.s1000}\\%)', '$N = 1\\,000$: $@{mc.e1000}\\%$ (eroarea standard @{mc.s1000}\\%)'),
      T('$N = 100\\,000$: $@{mc.e100000}\\%$ (s.e. @{mc.s100000}\\%)', '$N = 100\\,000$: $@{mc.e100000}\\%$ (eroarea standard @{mc.s100000}\\%)'),
      T('10 times more precision needs 100 times more simulations', 'o precizie de 10 ori mai mare cere de 100 de ori mai multe simulări')]),
    (T('For a relative error of 10\\% at the 95\\% level: $N \\approx 1.96^2(1 - p)/(0.1^2 p) = @{mc.nrel}$', 'Pentru o eroare relativă de 10\\%, la nivelul de încredere de 95\\%: $N \\approx 1.96^2(1 - p)/(0.1^2 p) = @{mc.nrel}$'),
     [T('obtained from $1.96 \\times \\sqrt{p(1 - p)/N} = 0.1\\,p$: the half-width of the interval is 10\\% of $p$', 'se obține din $1.96 \\times \\sqrt{p(1 - p)/N} = 0.1\\,p$: semilățimea intervalului este 10\\% din $p$'),
      T('a rare event needs many simulations', 'un eveniment rar cere multe simulări')])))

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
    (T('Each period the price is multiplied by $u$ (up) with probability $p$, or by $d < u$ (down) with probability $1 - p$', 'În fiecare perioadă prețul se înmulțește cu $u$ (creștere), cu probabilitatea $p$, sau cu $d < u$ (scădere), cu probabilitatea $1 - p$'),
     [T('$u$, $d$: the up and down factors (e.g. $u = 1.2$: $+20\\%$); the periods are independent', '$u$, $d$: factorii de creștere și de scădere (de exemplu, $u = 1.2$: $+20\\%$); perioadele sînt independente')]),
    (T('After $n$ steps with $K$ up moves: $S_n = S_0 u^K d^{n - K}$', 'După $n$ pași, dintre care $K$ creșteri: $S_n = S_0 u^K d^{n - K}$'),
     [T('$S_0$: the initial price; $K \\sim B(n, p)$: the number of up moves is binomial', '$S_0$: prețul inițial; $K \\sim B(n, p)$: numărul de creșteri are distribuție binomială'),
      T('$\\ln S_n = \\ln S_0 + K\\ln u + (n - K)\\ln d$: a random walk in log prices (Section 9)', '$\\ln S_n = \\ln S_0 + K\\ln u + (n - K)\\ln d$: un mers aleator în prețurile logaritmice (secțiunea 9)')]),
    (T('$E[S_n] = S_0\\,(pu + (1 - p)d)^n$, because the steps are independent', '$E[S_n] = S_0\\,(pu + (1 - p)d)^n$, pentru că pașii sînt independenți'),
     [T('$pu + (1 - p)d$: the expected growth factor of one step', '$pu + (1 - p)d$: factorul de creștere așteptat al unui pas'),
      T('the basis of the binomial option pricing model \\refCRR; textbook: \\refFHH, Ch.~7', 'baza modelului binomial de evaluare a opțiunilor \\refCRR; manual: \\refFHH, cap.~7')])), 'footnotesize')

D.frame(T('Worked Example: a Two-Step Tree', 'Exemplu lucrat: un arbore cu doi pași'), items(
    T('$S_0 = 100$, $u = 1.2$, $d = 0.8$, $p = 0.6$', '$S_0 = 100$, $u = 1.2$, $d = 0.8$, $p = 0.6$'),
    (T('Terminal values and probabilities', 'Valorile finale și probabilitățile lor'),
     [T('$uu$: @{t2.uu} with $p^2 = @{t2.puu}$; $ud$ or $du$: @{t2.ud} with $2p(1-p) = @{t2.pud}$; $dd$: @{t2.dd} with $(1-p)^2 = @{t2.pdd}$',
        '$uu$: @{t2.uu} cu $p^2 = @{t2.puu}$; $ud$ sau $du$: @{t2.ud} cu $2p(1-p) = @{t2.pud}$; $dd$: @{t2.dd} cu $(1-p)^2 = @{t2.pdd}$')]),
    T('$E[S_2] = 100 \\times (0.6 \\times 1.2 + 0.4 \\times 0.8)^2 = 100 \\times @{t2.g}^2 = @{t2.E}$', '$E[S_2] = 100 \\times (0.6 \\times 1.2 + 0.4 \\times 0.8)^2 = 100 \\times @{t2.g}^2 = @{t2.E}$'),
    (T('$P(S_2 < 100) = @{t2.pud} + @{t2.pdd} = @{t2.ploss}$', '$P(S_2 < 100) = @{t2.pud} + @{t2.pdd} = @{t2.ploss}$'),
     [T('one up and one down move lose money, because $ud = 0.96 < 1$', 'o creștere și o scădere, în orice ordine, aduc o pierdere, pentru că $ud = 0.96 < 1$')]),
    (T('With $p^* = (1 - d)/(u - d) = @{t2.pstar}$ the price is a fair game: $E[S_{t+1} \\mid S_t] = S_t$ (Section 9)', 'Cu $p^* = (1 - d)/(u - d) = @{t2.pstar}$ prețul este un joc echitabil: $E[S_{t+1} \\mid S_t] = S_t$ (secțiunea 9)'),
     [T('$p^*$ solves $p^*u + (1 - p^*)d = 1$: the expected growth factor of a step equals 1', '$p^*$ este soluția ecuației $p^*u + (1 - p^*)d = 1$: factorul de creștere așteptat al unui pas este 1')])), 'footnotesize')

D.frame(T('Calibrating the Tree to a Market (Cox--Ross--Rubinstein)', 'Calibrarea arborelui pe o piață (Cox--Ross--Rubinstein)'), items(
    (T('\\textbf{CRR} (Cox--Ross--Rubinstein) with step $\\Delta t$: $u = e^{\\sigma\\sqrt{\\Delta t}}$, $d = 1/u$, $p = (e^{\\mu\\Delta t} - d)/(u - d)$', '\\textbf{CRR} (Cox--Ross--Rubinstein) cu pasul $\\Delta t$: $u = e^{\\sigma\\sqrt{\\Delta t}}$, $d = 1/u$, $p = (e^{\\mu\\Delta t} - d)/(u - d)$'),
     [T('$\\Delta t$: the length of one step, in years; $\\mu$: the annual drift; $\\sigma$: the annual volatility', '$\\Delta t$: lungimea unui pas, în ani; $\\mu$: tendința anuală (drift); $\\sigma$: volatilitatea anuală'),
      T('$d = 1/u$: an up move followed by a down move returns to the start', '$d = 1/u$: o creștere urmată de o scădere readuce prețul la valoarea inițială'),
      T('chosen so that the mean and the variance of each step match a market with drift $\\mu$ and volatility $\\sigma$', 'alese astfel încît media și varianța fiecărui pas să corespundă unei piețe cu tendința $\\mu$ și volatilitatea $\\sigma$')]),
    (T('S\\&P 500, @{y0}--@{y1}: $\\sigma = @{bn.sigma}\\%$, $\\mu = @{bn.mu}\\%$ a year; daily steps, $\\Delta t = 1/252$', 'S\\&P 500, @{y0}--@{y1}: $\\sigma = @{bn.sigma}\\%$, $\\mu = @{bn.mu}\\%$ pe an; pași zilnici, $\\Delta t = 1/252$'),
     [T('$u = @{bn.u}$, $d = @{bn.d}$, $p = @{bn.p}$', '$u = @{bn.u}$, $d = @{bn.d}$, $p = @{bn.p}$')]),
    (T('As $\\Delta t \\to 0$, $\\ln S_T$ becomes Normal (CLT for $K$)', 'Cînd $\\Delta t \\to 0$, $\\ln S_T$ devine Normal (CLT aplicată lui $K$)'),
     [T('$T$: the horizon; the tree converges to GBM (Section 10)', '$T$: orizontul; arborele converge către GBM (secțiunea 10)')])), 'footnotesize')

chart(T('Binomial Paths and the Lognormal Limit (SFEBinomp)', 'Traiectorii binomiale și limita lognormală (SFEBinomp)'), 'sfm_ch4_binomial', 'SFM_ch4_binomial', [
    T('Left: 12 paths over one year of daily steps, $S_0 = 100$', 'Stînga: 12 traiectorii pe un an, cu pași zilnici, $S_0 = 100$'),
    (T('Right: distribution of $S_T$ after 252 steps ($T = 1$ year); $E[S_T] = @{bn.ES}$ $= 100e^{\\mu}$', 'Dreapta: distribuția lui $S_T$ după 252 de pași ($T = 1$ an); $E[S_T] = @{bn.ES}$ $= 100e^{\\mu}$'),
     [T('5\\% quantile @{bn.q05_bin} vs @{bn.q05_ln} in the lognormal limit', 'cuantila de 5\\%: @{bn.q05_bin}, față de @{bn.q05_ln} în limita lognormală')]),
    (T('$P(S_T < 100)$: @{bn.pl}\\% binomial vs @{bn.pln}\\% lognormal', '$P(S_T < 100)$: @{bn.pl}\\% în modelul binomial, față de @{bn.pln}\\% în cel lognormal'),
     [T('the exact value $S_T = 100$ has a positive probability, which the binomial counts on one side only', 'valoarea exactă $S_T = 100$ are probabilitate pozitivă, pe care modelul binomial o plasează de o singură parte')])], h='0.46\\textheight')

D.recap(('The Binomial Model', 'Modelul binomial'), [
    T('Up or down each period: $S_n = S_0 u^K d^{n-K}$ with binomial $K$', 'Creștere sau scădere în fiecare perioadă: $S_n = S_0 u^K d^{n-K}$ cu $K$ binomial'),
    T('CRR matches $\\mu$ and $\\sigma$; the limit is lognormal', 'CRR reproduce $\\mu$ și $\\sigma$; limita este lognormală'),
    T('A special $p^*$ turns the price into a fair game: the key to option pricing', 'Un $p^*$ special transformă prețul într-un joc echitabil: cheia evaluării opțiunilor')])

# =============================================================================
# 9. PROCESE ÎN TIMP DISCRET
# =============================================================================
D.section('Discrete-Time Stochastic Processes', 'Procese stochastice în timp discret')

D.frame(T('Stochastic Processes and Information', 'Procese stochastice și informație'), items(
    (T('A \\textbf{stochastic process} $\\{X_t\\}$: a family of random variables indexed by time', 'Un \\textbf{proces stochastic} $\\{X_t\\}$: o familie de variabile aleatoare indexate după timp'),
     [T('one realisation = one path; a price series is a single path of the process', 'o realizare = o traiectorie; o serie de prețuri este o singură traiectorie a procesului')]),
    (T('$\\mathcal{F}_t$: the information available at time $t$ (past prices, news)', '$\\mathcal{F}_t$: informația disponibilă la momentul $t$ (prețuri trecute, știri)'),
     [T('it grows with $t$: nothing is forgotten (a \\textbf{filtration})', 'crește cu $t$: nimic nu se uită (o \\textbf{filtrare})')]),
    (T('\\textbf{(Weakly) stationary}: $E[X_t]$ and $\\text{Cov}(X_t, X_{t+h})$ do not depend on $t$', '\\textbf{Staționar (slab)}: $E[X_t]$ și $\\text{Cov}(X_t, X_{t+h})$ nu depind de $t$'),
     [T('$h$: the lag between two dates; the mean and the dependence look the same at every date', '$h$: lagul dintre două momente; media și dependența arată la fel la orice moment'),
      T('returns: roughly yes; prices: no', 'randamentele: aproximativ da; prețurile: nu')]),
    T('Textbook: \\refFHH, Ch.~4 and Sec.~11.3', 'Manual: \\refFHH, cap.~4 și secț.~11.3')))

D.frame(T('White Noise and the Random Walk', 'Zgomotul alb și mersul aleator'), items(
    (T('\\textbf{White noise}: $E[\\varepsilon_t] = 0$, $\\text{Var}(\\varepsilon_t) = \\sigma^2$, $\\text{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ for $t \\ne s$', '\\textbf{Zgomot alb}: $E[\\varepsilon_t] = 0$, $\\text{Var}(\\varepsilon_t) = \\sigma^2$, $\\text{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ pentru $t \\ne s$'),
     [T('$\\varepsilon_t$: the shock at time $t$; mean zero, constant variance $\\sigma^2$, no correlation across dates', '$\\varepsilon_t$: șocul de la momentul $t$; medie zero, varianță constantă $\\sigma^2$, necorelat între momente diferite'),
      T('i.i.d. noise is stronger: the $\\varepsilon_t$ are independent, not only uncorrelated', 'zgomotul i.i.d. este o condiție mai tare: $\\varepsilon_t$ sînt independente, nu doar necorelate')]),
    (T('\\textbf{Random walk}: $S_t = S_{t-1} + \\mu + \\varepsilon_t = S_0 + \\mu t + \\sum_{s=1}^t \\varepsilon_s$', '\\textbf{Mers aleator}: $S_t = S_{t-1} + \\mu + \\varepsilon_t = S_0 + \\mu t + \\sum_{s=1}^t \\varepsilon_s$'),
     [T('each step adds the drift $\\mu$ and a new shock; $S_0$: the starting value', 'fiecare pas adaugă tendința (drift-ul) $\\mu$ și un șoc nou; $S_0$: valoarea de pornire'),
      T('$E[S_t] = S_0 + \\mu t$; $\\text{Var}(S_t) = t\\sigma^2$ grows with $t$: not stationary', '$E[S_t] = S_0 + \\mu t$; $\\text{Var}(S_t) = t\\sigma^2$ crește cu $t$: nestaționar'),
      T('simulated with $\\sigma = 1$: variance @{pr.rw10}, @{pr.rw50}, @{pr.rw100} at $t = 10, 50, 100$', 'simulat cu $\\sigma = 1$: varianța @{pr.rw10}, @{pr.rw50}, @{pr.rw100} la $t = 10, 50, 100$')]),
    (T('Log prices as a random walk = returns as white noise', 'Dacă prețurile logaritmice urmează un mers aleator, randamentele sînt zgomot alb'),
     [T('the benchmark of Chapter 7 (efficient markets)', 'ipoteza de referință din Capitolul 7 (piețe eficiente)')])), 'footnotesize')

D.frame(T('Martingales', 'Martingale'), items(
    (T('$\\{X_t\\}$ is a \\textbf{martingale} if $E|X_t| < \\infty$ and $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$', '$\\{X_t\\}$ este un \\textbf{martingal} dacă $E|X_t| < \\infty$ și $E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$'),
     [T('$E|X_t| < \\infty$: the mean exists; $\\mathcal{F}_t$: the information up to $t$', '$E|X_t| < \\infty$: media există; $\\mathcal{F}_t$: informația pînă la momentul $t$'),
      T('a fair game: the best forecast of tomorrow is today', 'un joc echitabil: cea mai bună prognoză pentru mîine este valoarea de azi')]),
    (T('A \\textbf{martingale difference}: $\\varepsilon_t = X_t - X_{t-1}$ with $E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = 0$', 'O \\textbf{diferență de martingal}: $\\varepsilon_t = X_t - X_{t-1}$ cu $E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = 0$'),
     [T('the change cannot be predicted from the past', 'variația nu poate fi prevăzută pe baza trecutului')]),
    (T('Examples', 'Exemple'),
     [T('a random walk without drift; $S_t^2 - t\\sigma^2$ for the same walk; the binomial price under $p^*$', 'un mers aleator fără tendință; $S_t^2 - t\\sigma^2$ pentru același mers; prețul binomial cu $p^*$')]),
    (T('Martingale $\\ne$ i.i.d. increments: the conditional \\textbf{variance} may change, as in GARCH', 'Martingal $\\ne$ creșteri i.i.d.: \\textbf{varianța} condiționată se poate schimba, ca în GARCH'),
     [T('the efficient-market idea in its weak form \\refFama: excess returns (above the risk-free rate) are a martingale difference', 'forma slabă a ipotezei pieței eficiente \\refFama: randamentele în exces (peste dobînda fără risc) sînt o diferență de martingal')])), 'footnotesize')

D.frame(T('The AR(1) Process', 'Procesul AR(1)'), items(
    (T('\\textbf{AR(1)} (autoregressive of order 1): $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ white noise', '\\textbf{AR(1)} (autoregresiv de ordinul 1): $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, $\\varepsilon_t$ zgomot alb'),
     [T('today\'s value is a constant, plus a share $\\phi$ of yesterday\'s value, plus a new shock', 'valoarea de azi este o constantă, plus o fracțiune $\\phi$ din valoarea de ieri, plus un șoc nou'),
      T('$c$: the constant; $\\phi$: the persistence; $\\sigma^2$: the variance of $\\varepsilon_t$', '$c$: constanta; $\\phi$: persistența; $\\sigma^2$: varianța lui $\\varepsilon_t$'),
      T('$\\phi = 0$: white noise; $\\phi = 1$, $c = \\mu$: random walk', '$\\phi = 0$: zgomot alb; $\\phi = 1$, $c = \\mu$: mers aleator')]),
    (T('Stationary iff $|\\phi| < 1$; then', 'Staționar dacă și numai dacă $|\\phi| < 1$; atunci'),
     [T('mean $c/(1 - \\phi)$, variance $\\sigma^2/(1 - \\phi^2)$', 'media $c/(1 - \\phi)$, varianța $\\sigma^2/(1 - \\phi^2)$'),
      T('autocorrelation at lag $h$: $\\rho(h) = \\text{Corr}(X_t, X_{t-h}) = \\phi^h$, decaying geometrically', 'autocorelația la lagul $h$: $\\rho(h) = \\text{Corr}(X_t, X_{t-h}) = \\phi^h$, cu scădere geometrică'),
      T('half-life of a shock: $\\ln 0.5/\\ln\\phi$ periods, the time until half of a shock is gone', 'timpul de înjumătățire al unui șoc: $\\ln 0.5/\\ln\\phi$ perioade, timpul după care jumătate din șoc a dispărut')]),
    (T('Uses', 'Utilizări'),
     [T('interest rates and spreads (mean reversion)', 'dobînzi și spread-uri (revenire la medie)'),
      T('the squared returns behave like a persistent AR, which leads to GARCH', 'pătratele randamentelor se comportă ca un AR persistent, ceea ce duce la GARCH')])), 'footnotesize')

D.frame(T('The AR(1) Process: Simulation', 'Procesul AR(1): simulare'), items(
    (T('Simulated with $\\sigma = 1$, $\\phi = 0.5$', 'Simulat cu $\\sigma = 1$, $\\phi = 0.5$'),
     [T('variance @{pr.05.var} (theory $1/(1 - 0.5^2) = @{pr.05.vth}$)', 'varianța @{pr.05.var} (teoretic $1/(1 - 0.5^2) = @{pr.05.vth}$)'),
      T('$\\rho(5) = @{pr.05.r5}$ (theory $0.5^5 = @{pr.05.r5th}$); half-life @{pr.05.hl} periods', '$\\rho(5) = @{pr.05.r5}$ (teoretic $0.5^5 = @{pr.05.r5th}$); timpul de înjumătățire @{pr.05.hl} perioade')]),
    (T('Simulated with $\\sigma = 1$, $\\phi = 0.95$', 'Simulat cu $\\sigma = 1$, $\\phi = 0.95$'),
     [T('variance @{pr.095.var} (theory @{pr.095.vth})', 'varianța @{pr.095.var} (teoretic @{pr.095.vth})'),
      T('$\\rho(5) = @{pr.095.r5}$; half-life @{pr.095.hl} periods', '$\\rho(5) = @{pr.095.r5}$; timpul de înjumătățire @{pr.095.hl} perioade')]),
    T('The closer $\\phi$ is to 1, the larger the variance and the slower a shock fades', 'Cu cît $\\phi$ este mai aproape de 1, cu atît varianța este mai mare și un șoc se stinge mai încet')))

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
    T('1827: the botanist Robert Brown sees pollen particles jiggling in water', '1827: botanistul Robert Brown observă mișcarea neregulată a particulelor de polen în apă'),
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
      T('$W(t) - W(s) \\sim N(0, t - s)$ for $s < t$: the variance equals the elapsed time', '$W(t) - W(s) \\sim N(0, t - s)$ pentru $s < t$: varianța este egală cu timpul scurs'),
      T('the paths are continuous', 'traiectoriile sînt continue')]),
    (T('Limit of the scaled random walk $W_n(t) = (\\varepsilon_1 + \\dots + \\varepsilon_{\\lfloor nt\\rfloor})/\\sqrt{n}$ (Donsker)',
       'Limita mersului aleator scalat $W_n(t) = (\\varepsilon_1 + \\dots + \\varepsilon_{\\lfloor nt\\rfloor})/\\sqrt{n}$ (Donsker)'),
     [T('$\\varepsilon_i = \\pm 1$ with probability $1/2$ each; $n$ steps per unit of time', '$\\varepsilon_i = \\pm 1$, fiecare cu probabilitatea $1/2$; $n$ pași pe unitatea de timp'),
      T('$\\lfloor nt\\rfloor$: the number of steps up to time $t$ (integer part); dividing by $\\sqrt{n}$ keeps the variance equal to $t$', '$\\lfloor nt\\rfloor$: numărul de pași pînă la momentul $t$ (partea întreagă); împărțirea la $\\sqrt{n}$ menține varianța egală cu $t$')]),
    T('Textbook: \\refFHH, Ch.~5', 'Manual: \\refFHH, cap.~5')),
    ph('wiener', 'Norbert Wiener (1894--1964)', h='0.50\\textheight'), wl='0.64', wr='0.32'), 'footnotesize')

chart(T('From Random Walks to the Wiener Process (SFEWienerProcess)', 'De la mersul aleator la procesul Wiener (SFEWienerProcess)'), 'sfm_ch4_wiener', 'SFM_ch4_wiener_gbm', [
    T('Left: scaled random walks with 10, 100 and 10\\,000 steps; the steps vanish, the randomness stays', 'Stînga: mersuri aleatoare scalate cu 10, 100 și 10\\,000 de pași; treptele dispar, caracterul aleator rămîne'),
    (T('Right: 20 Wiener paths; $\\text{Var}(W(t)) = t$, so they spread like $\\pm\\sqrt{t}$', 'Dreapta: 20 de traiectorii Wiener; $\\text{Var}(W(t)) = t$, deci dispersia crește ca $\\pm\\sqrt{t}$'),
     [T('simulated $\\text{Var}(W(1)) = @{w.v1}$; @{w.in1}\\% of $W(1)$ inside $\\pm 1$ (theory @{w.in1th}\\%)', '$\\text{Var}(W(1))$ simulată $= @{w.v1}$; @{w.in1}\\% din valorile $W(1)$ în $\\pm 1$ (teoretic @{w.in1th}\\%)')])], h='0.50\\textheight')

D.frame(T('Strange Properties of Wiener Paths', 'Proprietăți neobișnuite ale traiectoriilor Wiener'), items(
    (T('Continuous everywhere, differentiable nowhere', 'Continue peste tot, nicăieri derivabile'),
     [T('the slope $(W(t + h) - W(t))/h \\sim N(0, 1/h)$ has a variance $1/h$ that grows without bound as $h \\to 0$', 'panta $(W(t + h) - W(t))/h \\sim N(0, 1/h)$ are varianța $1/h$, care crește nelimitat cînd $h \\to 0$')]),
    (T('\\textbf{Quadratic variation}: $\\sum_k (W(t_{k+1}) - W(t_k))^2 \\to t$ on a finer and finer grid', '\\textbf{Variația pătratică}: $\\sum_k (W(t_{k+1}) - W(t_k))^2 \\to t$ pe o grilă tot mai fină'),
     [T('$0 = t_0 < t_1 < \\dots < t$: the grid; the sum of squared increments tends to the elapsed time $t$', '$0 = t_0 < t_1 < \\dots < t$: grila; suma pătratelor creșterilor tinde către timpul scurs $t$'),
      T('1000 steps on $[0, 1]$: mean @{w.qv}, s.d. @{w.qvsd} across 10\\,000 paths', '1000 de pași pe $[0, 1]$: media @{w.qv}, abaterea standard @{w.qvsd} pe 10\\,000 de traiectorii'),
      T('the reason why $(dW)^2 = dt$ in It\\^o calculus, and why realised variance measures volatility (Chapter 8)', 'motivul pentru care $(dW)^2 = dt$ în calculul It\\^o și pentru care varianța realizată măsoară volatilitatea (Capitolul 8)')]),
    T('$\\text{Cov}(W(s), W(t)) = \\min(s, t)$: simulated $\\text{Cov}(W(0.5), W(1)) = @{w.cov}$', '$\\text{Cov}(W(s), W(t)) = \\min(s, t)$: $\\text{Cov}(W(0.5), W(1))$ simulată $= @{w.cov}$'),
    T('$W$ is a martingale; so is $W(t)^2 - t$', '$W$ este un martingal; la fel și $W(t)^2 - t$')), 'footnotesize')

D.frame(T('Geometric Brownian Motion', 'Mișcarea browniană geometrică'), items(
    (T('\\textbf{GBM} (geometric Brownian motion): $dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$', '\\textbf{GBM} (geometric Brownian motion, mișcarea browniană geometrică): $dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$'),
     [T('$dS_t$: the change of the price over a very short interval $dt$', '$dS_t$: variația prețului pe un interval foarte scurt $dt$'),
      T('$dW_t$: the increment of the Wiener process over $dt$, distributed $N(0, dt)$', '$dW_t$: creșterea procesului Wiener pe intervalul $dt$, cu distribuția $N(0, dt)$'),
      T('the relative change $dS/S$ has drift $\\mu$ and volatility $\\sigma$, both per year', 'variația relativă $dS/S$ are tendința (drift-ul) $\\mu$ și volatilitatea $\\sigma$, ambele anuale')]),
    (T('Solution: $S_t = S_0\\exp\\big((\\mu - \\sigma^2/2)t + \\sigma W_t\\big)$', 'Soluția: $S_t = S_0\\exp\\big((\\mu - \\sigma^2/2)t + \\sigma W_t\\big)$'),
     [T('$\\ln(S_t/S_0) \\sim N\\big((\\mu - \\sigma^2/2)t, \\sigma^2 t\\big)$', '$\\ln(S_t/S_0) \\sim N\\big((\\mu - \\sigma^2/2)t, \\sigma^2 t\\big)$'),
      T('lognormal prices, i.i.d. Normal log returns', 'prețuri lognormale, randamente logaritmice Normale i.i.d.')]),
    T('The model behind \\refBS{} (Chapter 6 of the textbook)', 'Modelul din spatele formulei \\refBS{} (capitolul 6 din manual)')), 'footnotesize')

D.frame(T('GBM: Mean, Median and Simulation', 'GBM: media, mediana și simularea'), items(
    (T('Mean $E[S_t] = S_0e^{\\mu t}$; median $S_0e^{(\\mu - \\sigma^2/2)t}$', 'Media $E[S_t] = S_0e^{\\mu t}$; mediana $S_0e^{(\\mu - \\sigma^2/2)t}$'),
     [T('the median is lower: half of the paths end below $S_0e^{(\\mu - \\sigma^2/2)t}$', 'mediana este mai mică: jumătate din traiectorii se termină sub $S_0e^{(\\mu - \\sigma^2/2)t}$'),
      T('the gap, $\\sigma^2/2$ a year in log terms, is the volatility drag', 'diferența, $\\sigma^2/2$ pe an în termeni logaritmici, este volatility drag')]),
    (T('Exact simulation on a grid: $S_{t + \\Delta t} = S_t\\exp\\big((\\mu - \\sigma^2/2)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\big)$', 'Simulare exactă pe o grilă: $S_{t + \\Delta t} = S_t\\exp\\big((\\mu - \\sigma^2/2)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\big)$'),
     [T('$\\Delta t$: the grid step, in years; $Z \\sim N(0, 1)$, a new independent draw at each step', '$\\Delta t$: pasul grilei, în ani; $Z \\sim N(0, 1)$, o extragere nouă și independentă la fiecare pas'),
      T('exact because each log increment is Normal with the right mean and variance', 'exactă, pentru că fiecare creștere logaritmică este Normală, cu media și varianța corecte')])))

chart(T('GBM Calibrated to the S\\&P 500 (SFEsimGBM)', 'GBM calibrat pe S\\&P 500 (SFEsimGBM)'), 'sfm_ch4_gbm_paths', 'SFM_ch4_wiener_gbm', [
    T('Calibration, @{y0}--@{y1}: $\\sigma = @{g.sigma}\\%$, log drift $\\mu - \\sigma^2/2 = @{g.mulog}\\%$, so $\\mu = @{g.mu}\\%$ a year', 'Calibrare, @{y0}--@{y1}: $\\sigma = @{g.sigma}\\%$, tendința logaritmică $\\mu - \\sigma^2/2 = @{g.mulog}\\%$, deci $\\mu = @{g.mu}\\%$ pe an'),
    (T('After $T = 10$ years: mean @{g.mean}, median @{g.median}', 'După $T = 10$ ani: media @{g.mean}, mediana @{g.median}'),
     [T('@{g.below}\\% of the paths end below the mean; theory: $\\Phi(\\sigma\\sqrt{T}/2) = @{g.belowth}\\%$, with $\\Phi$ the $N(0, 1)$ CDF', '@{g.below}\\% din traiectorii se termină sub medie; teoretic: $\\Phi(\\sigma\\sqrt{T}/2) = @{g.belowth}\\%$, cu $\\Phi$ funcția de repartiție $N(0, 1)$')]),
    T('Probability of a loss after 10 years: @{g.ploss}\\%', 'Probabilitatea unei pierderi după 10 ani: @{g.ploss}\\%')], h='0.54\\textheight')

chart(T('Real Paths in the GBM Fan', 'Traiectorii reale în evantaiul GBM'), 'sfm_ch4_gbm_fan', 'SFM_ch4_wiener_gbm', [
    T('Each fan: 4000 GBM paths with the drift and volatility of the series itself, started on its first day', 'Fiecare evantai: 4000 de traiectorii GBM cu tendința și volatilitatea seriei respective, pornite din prima zi a seriei'),
    (T('S\\&P 500: the path reached the @{f.sp500.minr}\\% quantile on @{f.sp500.mind}, then the median at the end', 'S\\&P 500: traiectoria reală a coborît pînă la cuantila de @{f.sp500.minr}\\% pe @{f.sp500.mind} și a încheiat perioada în jurul medianei'),
     [T('outside the 90\\% band @{f.sp500.out}\\% of the time', 'în afara benzii de 90\\% în @{f.sp500.out}\\% din timp')]),
    (T('BET: outside the band @{f.bet.out}\\% of the time', 'BET: în afara benzii @{f.bet.out}\\% din timp'),
     [T('the 2000--2007 boom and the 2008 crash are not GBM-like', 'avîntul din 2000--2007 și crahul din 2008 nu seamănă cu GBM')]),
    (T('Bitcoin: $\\sigma = @{f.btc.sigma}\\%$ a year makes the fan enormous', 'Bitcoin: cu $\\sigma = @{f.btc.sigma}\\%$ pe an, evantaiul este foarte larg'),
     [T('annualised with $A = 365$: crypto trades every calendar day ($A = 252$ for the indices)', 'anualizată cu $A = 365$: activele cripto se tranzacționează în fiecare zi calendaristică ($A = 252$ pentru indici)')])], h='0.46\\textheight', size='scriptsize')

chart(T('What GBM Misses', 'Limitele modelului GBM'), 'sfm_ch4_gbm_check', 'SFM_ch4_wiener_gbm', [
    T('Excess kurtosis: S\\&P 500 @{ck.sp500.k}, BET @{ck.bet.k}, Bitcoin @{ck.btc.k}; 90\\% of 300 GBM simulations lie within $[@{ck.sp500.klo}, @{ck.sp500.khi}]$',
      'Excesul de boltire: S\\&P 500 @{ck.sp500.k}, BET @{ck.bet.k}, Bitcoin @{ck.btc.k}; 90\\% din 300 de simulări GBM sînt în $[@{ck.sp500.klo}, @{ck.sp500.khi}]$'),
    T('Volatility clustering: $\\text{Corr}(r_t^2, r_{t-1}^2) = @{ck.sp500.a}$, $@{ck.bet.a}$, $@{ck.btc.a}$ vs at most $@{ck.btc.ahi}$ under GBM',
      'Volatility clustering: $\\text{Corr}(r_t^2, r_{t-1}^2) = @{ck.sp500.a}$, $@{ck.bet.a}$, $@{ck.btc.a}$ față de cel mult $@{ck.btc.ahi}$ în GBM'),
    (T('\\hypertarget{ch4-dd}{}Maximum drawdown $\\min_t \\left(S_t/\\max_{s \\le t} S_s - 1\\right)$: the largest fall from a previous peak', '\\hypertarget{ch4-dd}{}Drawdown-ul maxim $\\min_t \\left(S_t/\\max_{s \\le t} S_s - 1\\right)$: cea mai mare scădere față de un maxim anterior'),
     [T('S\\&P 500 @{ck.sp500.dd}\\% is ordinary for GBM; BET @{ck.bet.dd}\\% is beyond all but 5\\% of the paths (5\\% quantile @{ck.bet.ddlo}\\%)',
        '$@{ck.sp500.dd}\\%$ la S\\&P 500 este obișnuit în GBM; $@{ck.bet.dd}\\%$ la BET depășește 95\\% din traiectorii (cuantila de 5\\%: $@{ck.bet.ddlo}\\%$)')]),
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
    (T('Drawdown (\\hyperlink{ch4-dd}{\\textcolor{MainBlue}{What GBM Misses}}), here within a calendar year', 'Drawdown (\\hyperlink{ch4-dd}{\\textcolor{MainBlue}{Limitele modelului GBM}}), aici în cursul unui an calendaristic'),
     [T('beyond 20\\%: S\\&P 500 in @{dd.sp500.nh} of @{dd.sp500.ny} years (@{dd.sp500.sr}\\%) vs @{dd.sp500.pg}\\% under GBM; BET @{dd.bet.sr}\\% vs @{dd.bet.pg}\\%', 'peste 20\\%: S\\&P 500 în @{dd.sp500.nh} din @{dd.sp500.ny} ani (@{dd.sp500.sr}\\%), față de @{dd.sp500.pg}\\% în GBM; BET @{dd.bet.sr}\\% față de @{dd.bet.pg}\\%')]),
    (T('Beyond 40\\%: GBM expects @{dd.sp500.e40} such years for the S\\&P 500 and @{dd.bet.e40} for the BET', 'Peste 40\\%: GBM prevede, în medie, @{dd.sp500.e40} asemenea ani pentru S\\&P 500 și @{dd.bet.e40} pentru BET'),
     [T('both had @{dd.sp500.n40} (2008: @{dd.sp500.w}\\% and @{dd.bet.w}\\%)', 'fiecare indice a avut @{dd.sp500.n40} (2008: $@{dd.sp500.w}\\%$ și $@{dd.bet.w}\\%$)')]),
    (T('Open question', 'Întrebare deschisă'),
     [T('does GBM overstate moderate drawdowns and understate extreme ones because calm and crisis years cluster?', 'supraestimează GBM drawdown-urile moderate și le subestimează pe cele extreme pentru că anii calmi și cei de criză apar grupat?')])], h='0.42\\textheight', size='scriptsize')

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: find studies of drawdown distributions and of their link to volatility clustering, and summarise them', '\\textbf{Literatura}: identificarea și rezumarea studiilor despre distribuția drawdown-urilor și legătura lor cu volatility clustering'),
    T('\\textbf{Code}: draft simulations of yearly drawdowns under GBM, the i.i.d. bootstrap and a GARCH model', '\\textbf{Cod}: o primă versiune a simulărilor drawdown-urilor anuale în GBM, în bootstrap i.i.d. și într-un model GARCH'),
    T('\\textbf{Design}: propose a test that compares the observed number of bad years with the model probability', '\\textbf{Design}: propunerea unui test care compară numărul observat de ani nefavorabili cu probabilitatea din model'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that simulates 10,000 one-year paths of daily S\\&P 500 returns under GBM and under an i.i.d.\\ bootstrap of 2000-2026 returns, and returns the probability of a drawdown beyond 20\\% and 40\\% within the year.}',
        '\\aiprompt{Write Python code that simulates 10,000 one-year paths of daily S\\&P 500 returns under GBM and under an i.i.d.\\ bootstrap of 2000-2026 returns, and returns the probability of a drawdown beyond 20\\% and 40\\% within the year.}')])), 'footnotesize')

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Definitions: drawdown from the running peak within the year, not from the first day of the year', 'Definițiile: drawdown-ul se calculează față de maximul atins pînă atunci în cursul anului, nu față de prima zi a anului'),
    T('Calibration: the same $\\mu$ and $\\sigma$, the same number of trading days a year as the data', 'Calibrarea: aceleași $\\mu$ și $\\sigma$, același număr de zile de tranzacționare pe an ca în date'),
    T('Inference: @{dd.sp500.ny} years is a small sample; compute a binomial p-value, not only shares', 'Inferența: @{dd.sp500.ny} de ani reprezintă un eșantion mic; calculați un p-value binomial, nu doar proporții'),
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

D.frame(T('Check Yourself', 'Autoevaluare'), items(
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
