r"""
build_chapter5.py -- Capitolul 5 (Cozi groase și teoria valorilor extreme), EN + RO dintr-o singură sursă
=======================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_05/ch5_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter5_heavy_tails_evt.tex
  RO/Cursuri/capitol5_cozi_groase_evt.tex
Rulare:
  python3 Quantlets/Ch_05/generate_all_charts.py
  python3 latex/build_chapter5.py && python3 latex/sfm_build.py compile 5
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo, ql   # noqa: E402
from ch5_common import ASSETS, BIB, NAMES, REFS, SHORT, STOCKS, T, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(5, 'lecture', refs=REFS)
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_05'


def side(title, fig, folder, bullets, w=0.56, size='footnotesize', h='0.78\\textheight'):
    """Chart on the left, bullets on the right, Quantlet link below."""
    left = f'\\begin{{center}}\n\\includegraphics[width=\\linewidth,height={h},keepaspectratio]{{{fig}.pdf}}\n\\end{{center}}'
    D.frame(title, cols(left, items(*bullets), wl=f'{w:.2f}', wr=f'{0.96 - w:.2f}') + '\n' + ql(folder), size)


def chart(title, fig, folder, bullets, h='0.62\\textheight', size='footnotesize'):
    D.chart(title, fig, folder, bullets, height=h, size=size)


def pic(file, cap_en, cap_ro, url, cred_en, cred_ro, h='0.50\\textheight'):
    return photo(file, T(cap_en, cap_ro), url, T(cred_en, cred_ro), h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
# Pareto tail: P(L > 3%) = 1%, alpha = 3
V.put('ex.p6', 1.0 * 2 ** -3, 3)
V.put('ex.p9', 1.0 * 3 ** -3, 3)
V.put('ex.v01', 3 * 10 ** (1 / 3), 2)
V.put('ex.cuberoot', 10 ** (1 / 3), 2)
# Hill on six numbers (k = 5)
HX = [9.5, 7.2, 6.1, 5.4, 4.8, 4.0]
logs = [math.log(x / HX[5]) for x in HX[:5]]
for i, l in enumerate(logs, 1):
    V.put(f'hx.l{i}', l, 3)
V.put('hx.mean', sum(logs) / 5, 3)
V.put('hx.alpha', 5 / sum(logs), 2)
V.put('hx.se', 5 / sum(logs) / math.sqrt(5), 2)
# Hill quantile for the S&P 500, VaR 0.1%
H = N['hill']['sp500']
fac = (H['k'] / (H['n'] * 0.001)) ** (1 / H['alpha'])
V.put('hq.ratio', H['k'] / (H['n'] * 0.001), 1)
V.put('hq.fac', fac, 2)
V.put('hq.var', H['u'] * fac, 2)
V.put('hq.np', H['n'] * 0.001, 2)
# GEV return level, S&P 500, 10 years = 120 months
G = N['gev']['sp500']
y120 = -math.log(1 - 1 / 120)
V.put('rl.y', y120, 5)
V.put('ge.xi', G['xi'], 3)
V.put('ge.mu', G['mu'], 3)
V.put('ge.sigma', G['sigma'], 3)

V.put('rl.pow', y120 ** (-G['xi']), 3)
V.put('rl.check', G['mu'] + G['sigma'] / G['xi'] * (y120 ** (-G['xi']) - 1), 1)
# POT VaR and ES, S&P 500
P = N['risk']['sp500']['pot']
n, nu, u, xi, be = P['n'], P['n_u'], P['u'], P['xi'], P['beta']
for lab, p in [('1', 0.01), ('25', 0.025), ('01', 0.001)]:
    r = n * p / nu
    V.put(f'pe.r{lab}', r, 3)
    V.put(f'pe.pow{lab}', r ** (-xi), 3)
    V.put(f'pe.var{lab}', u + be / xi * (r ** (-xi) - 1), 2)
V.put('pe.es25', (u + be / xi * ((n * 0.025 / nu) ** (-xi) - 1)) / (1 - xi) + (be - xi * u) / (1 - xi), 2)
V.put('pe.boxi', be / xi, 3)
V.put('pe.u', u, 3)
V.put('pe.xi', xi, 3)
V.put('pe.beta', be, 3)
V.put('pe.share', 100 * nu / n, 1)
# mean excess of a Pareto tail with alpha = 3
V.put('me.e2', 2 / 2, 1)
V.put('me.e4', 4 / 2, 1)
# 1987: Normal probability as a power of ten
V.raw('n4.exp', '⁅' + f'{N["crash"]["sp500"]["exp_below_4sd"]:.2f}' + '⁆')
RK = N['risk']
C = N['crash']
V.raw('n4.min', str(min(C[k]['n_below_4sd'] for k in ASSETS)))
V.raw('n4.max', str(max(C[k]['n_below_4sd'] for k in ASSETS)))
V.raw('c87.ly', str(int(round(-N['c87']['log10p'] - math.log10(252)))))
V.raw('nh.min', str(int(min(RK[k]['n'] for k in ASSETS + STOCKS) * 0.001)))
V.raw('nh.max', str(int(max(RK[k]['n'] for k in ASSETS + STOCKS) * 0.001)))
V.put('sexi.min', min(RK[k]['pot']['se_xi'] for k in ASSETS + STOCKS), 2)
V.put('sexi.max', max(RK[k]['pot']['se_xi'] for k in ASSETS + STOCKS), 2)
V.put('xi.min', min(RK[k]['pot']['xi'] for k in ASSETS), 2)
V.put('xi.max', max(RK[k]['pot']['xi'] for k in ASSETS), 2)
V.put('gxi.min', min(N['gev'][k]['xi'] for k in ASSETS), 2)
V.put('gxi.max', max(N['gev'][k]['xi'] for k in ASSETS), 2)
V.put('ha.min', min(N['hill'][k]['alpha'] for k in ASSETS + STOCKS), 2)
V.put('ha.max', max(N['hill'][k]['alpha'] for k in ASSETS + STOCKS), 2)
V.put('dmax', max(abs(RK[k][l]['EVT'] - RK[k][l]['Historical']) for k in ASSETS for l in ('var1', 'es25')), 2)
V.put('hs.min', min(N['hill'][k]['alpha'] for k in STOCKS), 2)
V.put('hs.max', max(N['hill'][k]['alpha'] for k in STOCKS), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how large can a daily loss be, and how often should we expect it?',
       '\\textbf{Întrebarea}: cît de mare poate fi o pierdere zilnică și cît de des ar trebui să ne așteptăm la ea?'),
     [T('Chapters 2 and 3: returns have heavier tails than the Normal distribution', 'Capitolele 2 și 3: randamentele au cozi mai groase decît distribuția Normală'),
      T('risk managers and regulators care about the tail, not about the centre', 'managerii de risc și autoritățile de reglementare se uită la coadă, nu la centru')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('three crashes: 1987, 2008, 2020', 'trei crahuri: 1987, 2008, 2020'),
      T('heavy and light tails; regular variation and the tail index $\\alpha$', 'cozi groase și cozi subțiri; variația regulată și indicele de coadă $\\alpha$'),
      T('the Hill estimator and the mean excess function', 'estimatorul Hill și funcția mean excess'),
      T('block maxima and the GEV distribution; peaks over threshold and the GPD', 'block maxima și distribuția GEV; peaks over threshold și distribuția GPD'),
      T('VaR 1\\%, ES 2.5\\% and VaR 0.1\\% by EVT, compared with historical simulation and the Normal distribution',
        'VaR 1\\%, ES 2,5\\% și VaR 0,1\\% prin EVT, comparate cu simularea istorică și cu distribuția Normală')])))

D.frame(T('Learning Outcomes', 'Ce veți ști la final'), items(
    T('Tell a heavy tail from a light tail and read a tail on log-log axes', 'Deosebiți o coadă groasă de una subțire și citiți o coadă pe axe log-log'),
    T('Estimate the tail index with the Hill estimator, with a standard error and a Hill plot',
      'Estimați indicele de coadă cu estimatorul Hill, cu eroare standard și cu graficul Hill (Hill plot)'),
    T('State the Fisher--Tippett--Gnedenko and the Pickands--Balkema--de Haan theorems in words and in formulas',
      'Enunțați teoremele Fisher--Tippett--Gnedenko și Pickands--Balkema--de Haan în cuvinte și în formule'),
    T('Fit a GEV to block maxima and a GPD to the losses above a threshold; check the fit with PP and QQ plots',
      'Ajustați o GEV pe maximele pe blocuri și o GPD pe pierderile peste un prag; verificați ajustarea cu grafice PP și QQ'),
    T('Compute return levels, VaR and ES from an EVT fit, step by step', 'Calculați return levels, VaR și ES dintr-o ajustare EVT, pas cu pas'),
    T('Judge, on real data, when EVT improves on historical simulation and on the Normal distribution',
      'Judecați, pe date reale, cînd EVT aduce ceva în plus față de simularea istorică și de distribuția Normală')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~18 (statistics of extreme risks)', 'Manual: \\refFHH, cap.~18 (statistica riscurilor extreme)'),
     [T('exercises with solutions: \\refBHL, Ch.~18', 'exerciții rezolvate: \\refBHL, cap.~18')]),
    T('Risk management view: \\refQRM, Ch.~5 (extreme value theory)', 'Perspectiva managementului riscului: \\refQRM, cap.~5 (teoria valorilor extreme)'),
    T('Reference books: \\refEKM; \\refColes', 'Cărți de referință: \\refEKM; \\refColes'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_05}',
       'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_05}'),
     [T('ported from the SFE Quantlets SFElshill, SFEhillquantile, SFEMeanExcessFun, SFEevt1, SFEtailGEV, SFEtailGPareto, block\\_max and var\\_pot',
        'portate din Quantlet-urile SFE SFElshill, SFEhillquantile, SFEMeanExcessFun, SFEevt1, SFEtailGEV, SFEtailGPareto, block\\_max și var\\_pot')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter5_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter5_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}',
      'Curs video: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}')))

# =============================================================================
# 1. DE CE CONTEAZĂ COZILE
# =============================================================================
D.section('Why Tails Matter', 'De ce contează cozile')

D.frame(T('19 October 1987: Black Monday', '19 octombrie 1987: Lunea Neagră'), cols(items(
    (T('On 19 October 1987 the S\\&P 500 fell about 20\\% in one day \\refCarlson', 'Pe 19 octombrie 1987, S\\&P 500 a scăzut cu aproximativ 20\\% într-o singură zi \\refCarlson'),
     [T('as a log return: $\\ln(1 - 0.20) = @{c87.lr}\\%$', 'ca randament logaritmic: $\\ln(1 - 0.20) = @{c87.lr}\\%$'),
      T('larger than any daily loss in our data since 1990 (the worst: $@{c.sp500.ret}\\%$ on @{c.sp500.date})', 'mai mare decît orice pierdere zilnică din datele noastre din 1990 (cea mai mare: $@{c.sp500.ret}\\%$ pe @{c.sp500.date})')]),
    (T('Measure it with the Normal distribution fitted to the S\\&P 500, @{c.sp500.y0}--@{y1}', 'Să o măsurăm cu distribuția Normală ajustată pe S\\&P 500, @{c.sp500.y0}--@{y1}'),
     [T('daily standard deviation $\\hat\\sigma = @{c.sp500.sd}\\%$, so $z = @{c87.z}$', 'abaterea standard zilnică $\\hat\\sigma = @{c.sp500.sd}\\%$, deci $z = @{c87.z}$'),
      T('$P(Z \\le @{c87.z}) \\approx 10^{@{c87.lp}}$: impossible, yet it happened', '$P(Z \\le @{c87.z}) \\approx 10^{@{c87.lp}}$: imposibil, și totuși s-a întîmplat')]),
    T('The Normal distribution says nothing useful about such days: we need a model of the tail',
      'Distribuția Normală nu spune nimic util despre astfel de zile: ne trebuie un model al cozii')),
    pic('ch5_sp500_1987_fed.png', 'The S\\&P 500, 14--21 October 1987, 5-minute data (Federal Reserve)',
        'S\\&P 500, 14--21 octombrie 1987, date la 5 minute (Federal Reserve)',
        'https://commons.wikimedia.org/wiki/File:S\\%26P_500_index_around_the_time_of_the_crash.png',
        'Chart: M. Carlson, Federal Reserve Board (2006); public domain; Wikimedia Commons',
        'Grafic: M. Carlson, Federal Reserve Board (2006); domeniu public; Wikimedia Commons', h='0.38\\textheight'), wl='0.50', wr='0.46'))

D.frame(T('September 2008: Lehman Brothers', 'Septembrie 2008: Lehman Brothers'), cols(items(
    (T('Lehman Brothers filed for bankruptcy on 15 September 2008; the global financial crisis followed',
       'Lehman Brothers a intrat în faliment pe 15 septembrie 2008; a urmat criza financiară globală'),
     [T('the S\\&P 500 fell $@{wd.sp500.2}\\%$ on @{wd.sp500.2.d} and $@{wd.sp500.3}\\%$ on @{wd.sp500.3.d}',
        'S\\&P 500 a scăzut cu $@{wd.sp500.2}\\%$ pe @{wd.sp500.2.d} și cu $@{wd.sp500.3}\\%$ pe @{wd.sp500.3.d}'),
      T('the BET lost $@{wd.bet.1}\\%$ on @{wd.bet.1.d}', 'BET a pierdut $@{wd.bet.1}\\%$ pe @{wd.bet.1.d}')]),
    (T('Large losses came in clusters, not one at a time', 'Pierderile mari au venit în grupuri, nu una cîte una'),
     [T('volatility clustering (Chapter 2) concentrates the extremes in a few months', 'gruparea volatilității (Capitolul 2) concentrează extremele în cîteva luni')]),
    T('A VaR model based on the Normal distribution puts almost zero probability on such days',
      'Un model VaR bazat pe distribuția Normală dă o probabilitate aproape nulă acestor zile')),
    pic('ch5_lehman_2007.jpg', 'Lehman Brothers headquarters, New York, 2007', 'Sediul Lehman Brothers, New York, 2007',
        'https://commons.wikimedia.org/wiki/File:Lehman_Brothers_Times_Square_by_David_Shankbone.jpg',
        'Photo: David Shankbone (2007); CC BY-SA 3.0; Wikimedia Commons', 'Foto: David Shankbone (2007); CC BY-SA 3.0; Wikimedia Commons',
        h='0.46\\textheight'), wl='0.60', wr='0.36'))

D.frame(T('March 2020: the COVID-19 Crash', 'Martie 2020: crahul COVID-19'), cols(items(
    (T('In one week of March 2020 three of our series had their largest daily loss', 'Într-o singură săptămînă din martie 2020, trei dintre seriile noastre au avut cea mai mare pierdere zilnică'),
     [T('S\\&P 500: $@{c.sp500.ret}\\%$ on @{c.sp500.date}', 'S\\&P 500: $@{c.sp500.ret}\\%$ pe @{c.sp500.date}'),
      T('DAX: $@{c.dax.ret}\\%$ on @{c.dax.date}', 'DAX: $@{c.dax.ret}\\%$ pe @{c.dax.date}'),
      T('Bitcoin: $@{c.btc.ret}\\%$ on @{c.btc.date}', 'Bitcoin: $@{c.btc.ret}\\%$ pe @{c.btc.date}')]),
    (T('The data of this chapter', 'Datele acestui capitol'),
     [T('the data: daily closes from EODHD; S\\&P 500 and DAX since 1990, BET since 1997, Bitcoin since 2014',
        'datele: închideri zilnice de la EODHD; S\\&P 500 și DAX din 1990, BET din 1997, Bitcoin din 2014')]),
    T('Question of the chapter: how often should we expect such a day?', 'Întrebarea capitolului: cît de des ar trebui să ne așteptăm la o astfel de zi?')),
    pic('ch5_fidi_2020.jpg', 'The empty Financial District of New York, 29 March 2020', 'Districtul financiar gol din New York, 29 martie 2020',
        'https://commons.wikimedia.org/wiki/File:Solitude_(50073346382).jpg',
        'Photo: Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons', 'Foto: Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons',
        h='0.36\\textheight'), wl='0.54', wr='0.42'))

chart(T('Crash Days against the Normal Distribution', 'Zilele de crah față de distribuția Normală'), 'sfm_ch5_crash_days', 'SFM_ch5_crashes', [
    T('Dashed line: the daily loss that a Normal law fitted to each series exceeds once in 100 years (S\\&P 500: $@{cd.sp500.lvl}\\%$; BET: $@{cd.bet.lvl}\\%$)',
      'Linia punctată: pierderea zilnică depășită o dată la 100 de ani de legea Normală ajustată pe fiecare serie (S\\&P 500: $@{cd.sp500.lvl}\\%$; BET: $@{cd.bet.lvl}\\%$)'),
    T('Observed: @{cd.sp500.nb} such days for the S\\&P 500 since 1990 and @{cd.bet.nb} for the BET since 1997',
      'Observat: @{cd.sp500.nb} astfel de zile pentru S\\&P 500 din 1990 și @{cd.bet.nb} pentru BET din 1997')], h='0.60\\textheight')


def crow(k):
    return (f'{NAMES[k]} & ' + T(f'@{{c.{k}.date}}', f'@{{c.{k}.date}}') + f' & ${{@{{c.{k}.ret}}}}$ & ${{@{{c.{k}.z}}}}$ & $10^{{@{{c.{k}.ly}}}}$ & '
            f'@{{c.{k}.n4}} & $@{{c.{k}.e4}}$')


D.frame(T('How Rare Are the Worst Days under the Normal Law?', 'Cît de rare sînt cele mai proaste zile pentru legea Normală?'), table(
    'llrrrrr', T('& worst day & loss (\\%) & $z$ & Normal: once in (years) & days below $-4\\sigma$ & expected',
                 '& cea mai proastă zi & pierdere (\\%) & $z$ & Normală: o dată la (ani) & zile sub $-4\\sigma$ & așteptat'),
    [crow(k) for k in ASSETS], size='scriptsize') + items(
    T('$z = (r - \\hat\\mu)/\\hat\\sigma$ with the mean and the standard deviation of the whole sample; once in $1/(p \\times 252)$ years (365 days for Bitcoin)',
      '$z = (r - \\hat\\mu)/\\hat\\sigma$ cu media și abaterea standard a întregului eșantion; o dată la $1/(p \\times 252)$ ani (365 de zile pentru Bitcoin)'),
    T('The Normal waiting times are absurd: far longer than the history of any market',
      'Timpii de așteptare ai legii Normale sînt absurzi: mult mai lungi decît istoria oricărei piețe'),
    T('Losses beyond $4\\sigma$: the Normal law expects fewer than one day per series; we see between @{n4.min} and @{n4.max}',
      'Pierderi dincolo de $4\\sigma$: legea Normală se așteaptă la mai puțin de o zi pe serie; vedem între @{n4.min} și @{n4.max}')) + ql('SFM_ch5_crashes'), 'footnotesize')

D.frame(T('Where Extreme Value Theory Comes From', 'De unde vine teoria valorilor extreme'), cols(items(
    (T('\\textbf{EVT} (extreme value theory): the statistics of the largest values of a sample', '\\textbf{EVT} (extreme value theory, teoria valorilor extreme): statistica celor mai mari valori dintr-un eșantion'),
     [T('first used for floods, storms and the strength of materials', 'folosită întîi pentru inundații, furtuni și rezistența materialelor'),
      T('\\refGumbel: the yearly maximum of a river decides the height of a dam', '\\refGumbel: maximul anual al unui rîu decide înălțimea unui baraj')]),
    (T('In finance the same questions appear', 'În finanțe apar aceleași întrebări'),
     [T('how large is the loss exceeded once in 10 years?', 'cît de mare este pierderea depășită o dată la 10 ani?'),
      T('how much capital covers the worst 1\\% of days?', 'cît capital acoperă cele mai proaste 1\\% dintre zile?')]),
    T('EVT models only the tail and lets the data speak there; early financial uses: \\refJdV, \\refLongin',
      'EVT modelează doar coada și lasă datele să vorbească acolo; primele aplicații în finanțe: \\refJdV, \\refLongin')),
    pic('ch5_gumbel_1931.jpg', 'Emil Julius Gumbel (1891--1966)', 'Emil Julius Gumbel (1891--1966)',
        'https://commons.wikimedia.org/wiki/File:Emil_Julius_Gumbel.jpg',
        'Photo: unknown author (c. 1931); public domain; Wikimedia Commons', 'Foto: autor necunoscut (c. 1931); domeniu public; Wikimedia Commons',
        h='0.44\\textheight'), wl='0.62', wr='0.34'))

D.frame(T('Why the Tail Matters for Risk', 'De ce contează coada pentru risc'), items(
    (T('\\textbf{VaR} (Value at Risk) at level $p$: the loss exceeded with probability $p$', '\\textbf{VaR} (Value at Risk, valoarea la risc) la nivelul $p$: pierderea depășită cu probabilitatea $p$'),
     [T('$\\mathrm{VaR}_p = -q_p(r)$, with $q_p$ the $p$-quantile of the return; for example VaR 1\\%', '$\\mathrm{VaR}_p = -q_p(r)$, cu $q_p$ cuantila de ordin $p$ a randamentului; de exemplu VaR 1\\%')]),
    (T('\\textbf{ES} (expected shortfall) at level $p$: the average loss on the worst $p$ share of days',
       '\\textbf{ES} (expected shortfall) la nivelul $p$: pierderea medie din cele mai proaste $p$ dintre zile'),
     [T('$\\mathrm{ES}_p = -E[r \\mid r \\le q_p(r)]$; banks report ES 2.5\\% under the \\refBasel', '$\\mathrm{ES}_p = -E[r \\mid r \\le q_p(r)]$; băncile raportează ES 2,5\\% conform \\refBasel'),
      T('ES looks inside the tail and is a coherent risk measure \\refAT', 'ES privește în interiorul cozii și este o măsură de risc coerentă \\refAT')]),
    T('Both live in the tail: a model that misses the tail misses the risk', 'Ambele trăiesc în coadă: un model care ratează coada ratează riscul'),
    T('\\textbf{Question for the room}: with 250 trading days a year, how many years of data contain about 10 days below VaR 0.1\\%?',
      '\\textbf{Întrebare pentru sală}: cu 250 de zile de tranzacționare pe an, cîți ani de date conțin circa 10 zile sub VaR 0,1\\%?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('$10 / (0.001 \\times 250) = 40$ years: historical data alone are too thin; EVT extrapolates the tail with a model',
        '$10 / (0.001 \\times 250) = 40$ de ani: datele istorice singure sînt prea puține; EVT extrapolează coada cu un model')])))

D.recap(('Why Tails Matter', 'De ce contează cozile'), [
    T('1987, 2008, 2020: daily losses of 10--20\\% for indices, almost 50\\% for Bitcoin', '1987, 2008, 2020: pierderi zilnice de 10--20\\% pentru indici, aproape 50\\% pentru Bitcoin'),
    T('Under the Normal distribution these days should never happen', 'Pentru distribuția Normală, aceste zile nu ar trebui să se întîmple niciodată'),
    T('VaR and ES are tail quantities; few observations lie that far out', 'VaR și ES sînt mărimi din coadă; puține observații se află atît de departe'),
    T('EVT: a theory for the shape of the tail, from hydrology to finance', 'EVT: o teorie pentru forma cozii, de la hidrologie la finanțe')])

# =============================================================================
# 2. COZI GROASE ȘI VARIAȚIA REGULATĂ
# =============================================================================
D.section('Heavy Tails and Regular Variation', 'Cozi groase și variația regulată')

D.frame(T('Light and Heavy Tails', 'Cozi subțiri și cozi groase'), items(
    (T('Work with losses $L = -r$ (in \\%): a large positive $L$ is a large loss', 'Lucrăm cu pierderile $L = -r$ (în \\%): un $L$ mare și pozitiv este o pierdere mare'),
     [T('the \\textbf{tail} of $L$ is described by the survival function $\\bar F(x) = P(L > x)$', '\\textbf{coada} lui $L$ este descrisă de funcția de supraviețuire $\\bar F(x) = P(L > x)$')]),
    (T('\\textbf{Light tail}: $\\bar F(x)$ falls at least as fast as an exponential, $e^{-cx}$', '\\textbf{Coadă subțire}: $\\bar F(x)$ scade cel puțin la fel de repede ca o exponențială, $e^{-cx}$'),
     [T('examples: Normal ($\\bar F(x) \\approx e^{-x^2/2}$), Exponential ($e^{-x}$); all moments are finite',
        'exemple: Normală ($\\bar F(x) \\approx e^{-x^2/2}$), Exponențială ($e^{-x}$); toate momentele sînt finite')]),
    (T('\\textbf{Heavy (power-law) tail}: $\\bar F(x)$ falls like a power, $x^{-\\alpha}$', '\\textbf{Coadă groasă (de tip putere)}: $\\bar F(x)$ scade ca o putere, $x^{-\\alpha}$'),
     [T('examples: Pareto, Student-t, the $\\alpha$-stable laws of Chapter 3', 'exemple: Pareto, Student-t, legile $\\alpha$-stabile din Capitolul 3'),
      T('$\\alpha$ is the \\textbf{tail index}: the smaller $\\alpha$, the heavier the tail', '$\\alpha$ este \\textbf{indicele de coadă}: cu cît $\\alpha$ este mai mic, cu atît coada este mai groasă')])))

chart(T('Power Laws Fall Much More Slowly', 'Legile putere scad mult mai încet'), 'sfm_ch5_light_heavy', 'SFM_ch5_tails_loglog', [
    T('On log-log axes a power law is a straight line with slope $-\\alpha$; light tails bend down and vanish',
      'Pe axe log-log, o lege putere este o dreaptă cu panta $-\\alpha$; cozile subțiri se curbează în jos și dispar'),
    T('$P(X > 10)$: Normal $@{lh.n10}$, Exponential $@{lh.e10}$, Student-t with 3 degrees of freedom $@{lh.t10}$',
      '$P(X > 10)$: Normală $@{lh.n10}$, Exponențială $@{lh.e10}$, Student-t cu 3 grade de libertate $@{lh.t10}$')], h='0.58\\textheight')

D.frame(T('Regular Variation and the Tail Index', 'Variația regulată și indicele de coadă'), items(
    (T('$L$ has a \\textbf{regularly varying} tail with index $\\alpha > 0$ if $\\bar F(x) = x^{-\\alpha}\\ell(x)$',
       '$L$ are o coadă \\textbf{cu variație regulată}, de indice $\\alpha > 0$, dacă $\\bar F(x) = x^{-\\alpha}\\ell(x)$'),
     [T('$\\ell$ is \\textbf{slowly varying}: $\\ell(tx)/\\ell(x) \\to 1$ as $x \\to \\infty$, for every $t > 0$ (e.g.\\ a constant, $\\ln x$)',
        '$\\ell$ este \\textbf{cu variație lentă}: $\\ell(tx)/\\ell(x) \\to 1$ cînd $x \\to \\infty$, pentru orice $t > 0$ (de ex.\\ o constantă, $\\ln x$)')]),
    (T('Equivalent: $\\dfrac{P(L > tx)}{P(L > x)} \\to t^{-\\alpha}$ for large $x$', 'Echivalent: $\\dfrac{P(L > tx)}{P(L > x)} \\to t^{-\\alpha}$ pentru $x$ mare'),
     [T('doubling the loss divides its probability by $2^\\alpha$, whatever the level', 'dublarea pierderii împarte probabilitatea ei la $2^\\alpha$, oricare ar fi nivelul'),
      T('Student-t with 3 degrees of freedom: $P(X > 20)/P(X > 10) = @{lh.tr}$, close to $2^{-3} = 0.125$', 'Student-t cu 3 grade de libertate: $P(X > 20)/P(X > 10) = @{lh.tr}$, aproape de $2^{-3} = 0.125$')]),
    (T('Tail indices of common laws', 'Indicii de coadă ai legilor uzuale'),
     [T('Pareto with $\\bar F(x) = (x/x_0)^{-\\alpha}$: index $\\alpha$; Student-t with $\\nu$ degrees of freedom: $\\alpha = \\nu$',
        'Pareto cu $\\bar F(x) = (x/x_0)^{-\\alpha}$: indicele $\\alpha$; Student-t cu $\\nu$ grade de libertate: $\\alpha = \\nu$'),
      T('$\\alpha$-stable with $\\alpha < 2$ (Chapter 3): the same $\\alpha$; Normal: no power tail ($\\alpha = \\infty$)',
        '$\\alpha$-stabilă cu $\\alpha < 2$ (Capitolul 3): același $\\alpha$; Normală: fără coadă putere ($\\alpha = \\infty$)')])))

D.frame(T('The Tail Index Decides Which Moments Exist', 'Indicele de coadă decide ce momente există'), items(
    (T('For a regularly varying tail: $E|L|^m < \\infty$ if $m < \\alpha$ and $E|L|^m = \\infty$ if $m > \\alpha$',
       'Pentru o coadă cu variație regulată: $E|L|^m < \\infty$ dacă $m < \\alpha$ și $E|L|^m = \\infty$ dacă $m > \\alpha$'),
     [T('$\\alpha \\le 1$: no mean; $\\alpha \\le 2$: no variance; $\\alpha \\le 4$: no kurtosis', '$\\alpha \\le 1$: nu există media; $\\alpha \\le 2$: nu există varianța; $\\alpha \\le 4$: nu există kurtosis-ul')]),
    (T('The ``inverse cubic law\'\' of stock returns: $\\alpha \\approx 3$ \\refGopi', '„Legea cubică inversă” a randamentelor acțiunilor: $\\alpha \\approx 3$ \\refGopi'),
     [T('variance finite, kurtosis infinite: the sample kurtosis of Chapter 2 does not settle', 'varianța finită, kurtosis-ul infinit: kurtosis-ul de selecție din Capitolul 2 nu se stabilizează')]),
    (T('Link to Chapter 3', 'Legătura cu Capitolul 3'),
     [T('the stable $\\alpha$ ($\\approx 1.5$) is fitted to the whole distribution; the tail index here uses only the tail',
        '$\\alpha$ stabil ($\\approx 1{,}5$) se ajustează pe întreaga distribuție; indicele de coadă de aici folosește doar coada'),
      T('a tail index near 3 is the evidence against infinite variance announced in Chapter 3 \\refMandelbrot',
        'un indice de coadă în jur de 3 este dovada împotriva varianței infinite anunțată în Capitolul 3 \\refMandelbrot')])))

chart(T('Real Losses on Log-Log Axes', 'Pierderi reale pe axe log-log'), 'sfm_ch5_loglog_real', 'SFM_ch5_tails_loglog', [
    T('Dots: the share of days with a loss above $x$; black: least-squares line through the 2.5\\% largest losses; red: Normal',
      'Puncte: proporția zilelor cu o pierdere peste $x$; negru: dreapta celor mai mici pătrate prin cele mai mari 2,5\\% dintre pierderi; roșu: Normală'),
    T('Slopes: S\\&P 500 $-@{ls.sp500}$, DAX $-@{ls.dax}$, BET $-@{ls.bet}$, Bitcoin $-@{ls.btc}$: straight lines, power tails',
      'Pante: S\\&P 500 $-@{ls.sp500}$, DAX $-@{ls.dax}$, BET $-@{ls.bet}$, Bitcoin $-@{ls.btc}$: drepte, deci cozi de tip putere')], h='0.64\\textheight')

D.frame(T('Reading a Log-Log Tail Plot', 'Cum citim un grafic log-log al cozii'), items(
    (T('If $\\bar F(x) \\approx C x^{-\\alpha}$, then $\\ln \\bar F(x) \\approx \\ln C - \\alpha \\ln x$: a line with slope $-\\alpha$',
       'Dacă $\\bar F(x) \\approx C x^{-\\alpha}$, atunci $\\ln \\bar F(x) \\approx \\ln C - \\alpha \\ln x$: o dreaptă cu panta $-\\alpha$'),
     [T('\\textbf{least-squares estimator}: regress $\\ln(i/n)$ on $\\ln L_{(i)}$ for the $k$ largest losses $L_{(1)} \\ge \\dots \\ge L_{(k)}$',
        '\\textbf{estimatorul celor mai mici pătrate}: regresia lui $\\ln(i/n)$ pe $\\ln L_{(i)}$ pentru cele mai mari $k$ pierderi $L_{(1)} \\ge \\dots \\ge L_{(k)}$'),
      T('$L_{(i)}$ is the $i$-th largest loss, an \\textbf{order statistic}', '$L_{(i)}$ este a $i$-a cea mai mare pierdere, o \\textbf{statistică de ordine}')]),
    T('The body of the distribution (small losses) bends away from the line: the power law holds only in the tail',
      'Corpul distribuției (pierderile mici) se depărtează de dreaptă: legea putere este valabilă doar în coadă'),
    T('The Normal tail drops below the data already at 3--4\\% for the indices', 'Coada Normală coboară sub date deja la 3--4\\% pentru indici'),
    T('The least-squares slope is simple but its points are not independent; the Hill estimator is the standard tool',
      'Panta celor mai mici pătrate este simplă, dar punctele ei nu sînt independente; estimatorul Hill este instrumentul standard')) + ql('SFM_ch5_tails_loglog'))

D.frame(T('Worked Example: a Pareto Tail', 'Exemplu lucrat: o coadă Pareto'), items(
    (T('Daily losses have a Pareto tail with $\\alpha = 3$ and $P(L > 3\\%) = 1\\%$', 'Pierderile zilnice au o coadă Pareto cu $\\alpha = 3$ și $P(L > 3\\%) = 1\\%$'),
     [T('$P(L > x) = 1\\% \\times (x/3)^{-3}$ for $x \\ge 3\\%$', '$P(L > x) = 1\\% \\times (x/3)^{-3}$ pentru $x \\ge 3\\%$')]),
    (T('Loss above 6\\%: $1\\% \\times 2^{-3} = @{ex.p6}\\%$; above 9\\%: $1\\% \\times 3^{-3} = @{ex.p9}\\%$', 'Pierdere peste 6\\%: $1\\% \\times 2^{-3} = @{ex.p6}\\%$; peste 9\\%: $1\\% \\times 3^{-3} = @{ex.p9}\\%$'),
     [T('doubling the loss divides the probability by $2^3 = 8$', 'dublarea pierderii împarte probabilitatea la $2^3 = 8$')]),
    T('\\textbf{What do you think?} Which loss is exceeded with probability 0.1\\% (VaR 0.1\\%)?',
      '\\textbf{Ce credeți?} Ce pierdere este depășită cu probabilitatea 0,1\\% (VaR 0,1\\%)?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('solve $1\\% \\times (x/3)^{-3} = 0.1\\%$: $x = 3 \\times 10^{1/3} = 3 \\times @{ex.cuberoot} = @{ex.v01}\\%$',
        'rezolvăm $1\\% \\times (x/3)^{-3} = 0.1\\%$: $x = 3 \\times 10^{1/3} = 3 \\times @{ex.cuberoot} = @{ex.v01}\\%$'),
      T('a probability ten times smaller only moves the loss by a factor $10^{1/\\alpha}$', 'o probabilitate de zece ori mai mică mută pierderea doar cu un factor $10^{1/\\alpha}$')])))

D.recap(('Heavy Tails', 'Cozi groase'), [
    T('Heavy tail: $P(L > x) = x^{-\\alpha}\\ell(x)$, a straight line on log-log axes', 'Coadă groasă: $P(L > x) = x^{-\\alpha}\\ell(x)$, o dreaptă pe axe log-log'),
    T('The tail index $\\alpha$ decides which moments exist: moments of order below $\\alpha$', 'Indicele de coadă $\\alpha$ decide ce momente există: momentele de ordin sub $\\alpha$'),
    T('Daily index losses: slopes near 3, the inverse cubic law', 'Pierderile zilnice ale indicilor: pante în jur de 3, legea cubică inversă'),
    T('Power law: the VaR moves only by $10^{1/\\alpha}$ when the probability falls tenfold', 'Legea putere: VaR se mută doar cu $10^{1/\\alpha}$ cînd probabilitatea scade de zece ori')])

# =============================================================================
# 3. ESTIMATORUL HILL
# =============================================================================
D.section('The Hill Estimator', 'Estimatorul Hill')

D.frame(T('The Hill Estimator', 'Estimatorul Hill'), items(
    (T('Sort the losses: $L_{(1)} \\ge L_{(2)} \\ge \\dots \\ge L_{(n)}$; keep the $k$ largest', 'Ordonăm pierderile: $L_{(1)} \\ge L_{(2)} \\ge \\dots \\ge L_{(n)}$; păstrăm cele mai mari $k$'),
     [T('\\textbf{Hill estimator} \\refHill: $\\hat\\alpha_k = \\Big[\\dfrac{1}{k}\\sum_{i=1}^{k}\\ln\\dfrac{L_{(i)}}{L_{(k+1)}}\\Big]^{-1}$',
        '\\textbf{Estimatorul Hill} \\refHill: $\\hat\\alpha_k = \\Big[\\dfrac{1}{k}\\sum_{i=1}^{k}\\ln\\dfrac{L_{(i)}}{L_{(k+1)}}\\Big]^{-1}$'),
      T('the threshold is the $(k+1)$-th largest loss, $u = L_{(k+1)}$', 'pragul este a $(k+1)$-a cea mai mare pierdere, $u = L_{(k+1)}$')]),
    (T('Why it works: if $P(L > x \\mid L > u) = (x/u)^{-\\alpha}$ (Pareto above $u$), then $\\ln(L/u)$ given $L > u$ is Exponential with mean $1/\\alpha$',
       'De ce funcționează: dacă $P(L > x \\mid L > u) = (x/u)^{-\\alpha}$ (Pareto peste $u$), atunci $\\ln(L/u)$, condiționat de $L > u$, este Exponențială cu media $1/\\alpha$'),
     [T('the mean of the $k$ log excesses estimates $1/\\alpha$; its inverse estimates $\\alpha$', 'media celor $k$ excese logaritmice estimează $1/\\alpha$; inversa ei estimează $\\alpha$'),
      T('Hill $=$ the maximum likelihood estimator of a Pareto tail above $u$', 'Hill $=$ estimatorul de verosimilitate maximă al unei cozi Pareto peste $u$')]),
    T('Ported from the SFE Quantlet SFElshill (least-squares and Hill estimators of the tail index) \\refFHH',
      'Portat din Quantlet-ul SFE SFElshill (estimatorii celor mai mici pătrate și Hill ai indicelui de coadă) \\refFHH')))

D.frame(T('Worked Example: Hill on Six Losses', 'Exemplu lucrat: Hill pe șase pierderi'), items(
    T('The six largest daily losses (\\%): $9.5, 7.2, 6.1, 5.4, 4.8, 4.0$; take $k = 5$, so $L_{(6)} = 4.0$',
      'Cele mai mari șase pierderi zilnice (\\%): $9.5$; $7.2$; $6.1$; $5.4$; $4.8$; $4.0$; luăm $k = 5$, deci $L_{(6)} = 4.0$'),
    (T('Log excesses $\\ln(L_{(i)}/4.0)$', 'Excesele logaritmice $\\ln(L_{(i)}/4.0)$'),
     [T('$@{hx.l1}$, $@{hx.l2}$, $@{hx.l3}$, $@{hx.l4}$, $@{hx.l5}$; mean $@{hx.mean}$', '$@{hx.l1}$; $@{hx.l2}$; $@{hx.l3}$; $@{hx.l4}$; $@{hx.l5}$; media $@{hx.mean}$')]),
    T('$\\hat\\alpha_5 = 1/@{hx.mean} = @{hx.alpha}$', '$\\hat\\alpha_5 = 1/@{hx.mean} = @{hx.alpha}$'),
    (T('Standard error under independence: $\\mathrm{SE}(\\hat\\alpha) \\approx \\hat\\alpha/\\sqrt{k} = @{hx.se}$', 'Eroarea standard sub independență: $\\mathrm{SE}(\\hat\\alpha) \\approx \\hat\\alpha/\\sqrt{k} = @{hx.se}$'),
     [T('with $k = 5$ the estimate is very imprecise: real applications use hundreds of losses', 'cu $k = 5$, estimarea este foarte imprecisă: aplicațiile reale folosesc sute de pierderi')])))

D.frame(T('Choosing $k$: the Hill Plot', 'Alegerea lui $k$: graficul Hill'), items(
    T('\\textbf{Small $k$}: only the far tail, few points: high variance', '\\textbf{$k$ mic}: doar coada îndepărtată, puține puncte: varianță mare'),
    T('\\textbf{Large $k$}: the body of the distribution enters, where the power law fails: bias', '\\textbf{$k$ mare}: intră corpul distribuției, unde legea putere nu mai este valabilă: deplasare (bias)'),
    (T('The \\textbf{Hill plot}: $\\hat\\alpha_k$ against $k$; look for a region where it is flat \\refDRH',
       '\\textbf{Graficul Hill (Hill plot)}: $\\hat\\alpha_k$ în funcție de $k$; căutăm o zonă în care este plat \\refDRH'),
     [T('here $k$ is shown as a share of the sample, $k/n$ between 0.5\\% and 10\\%', 'aici $k$ apare ca proporție din eșantion, $k/n$ între 0,5\\% și 10\\%')]),
    (T('Fix the rule before looking at the result: here $k = 2.5\\%$ of $n$', 'Fixăm regula înainte de a vedea rezultatul: aici $k = 2.5\\%$ din $n$'),
     [T('and report the estimates at $k = 1\\%$ and $5\\%$ of $n$ as a check', 'și raportăm estimările pentru $k = 1\\%$ și $5\\%$ din $n$, ca verificare')])))

chart(T('Hill Plots of Daily Losses', 'Grafice Hill ale pierderilor zilnice'), 'sfm_ch5_hill_plot', 'SFM_ch5_hill', [
    T('Between 1\\% and 5\\% of $n$ the curves are fairly flat, mostly between 2 and 3.5; band: 95\\% interval for the S\\&P 500',
      'Între 1\\% și 5\\% din $n$, curbele sînt destul de plate, în mare parte între 2 și 3,5; banda: intervalul de 95\\% pentru S\\&P 500'),
    T('Beyond 5\\% the estimates drift down: the body of the distribution biases $\\hat\\alpha$', 'Peste 5\\%, estimările coboară: corpul distribuției deplasează $\\hat\\alpha$')], h='0.60\\textheight')

D.frame(T('Inference for the Hill Estimator', 'Inferența pentru estimatorul Hill'), items(
    (T('For i.i.d.\\ data with a Pareto-type tail: $\\sqrt{k}\\,(\\hat\\alpha_k - \\alpha) \\to N(0, \\alpha^2)$, for $k \\to \\infty$, $k/n \\to 0$',
       'Pentru date i.i.d.\\ cu coadă de tip Pareto: $\\sqrt{k}\\,(\\hat\\alpha_k - \\alpha) \\to N(0, \\alpha^2)$, pentru $k \\to \\infty$, $k/n \\to 0$'),
     [T('i.i.d.: independent and identically distributed; $\\mathrm{SE} = \\hat\\alpha/\\sqrt{k}$, 95\\% CI (confidence interval) $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$',
        'i.i.d.: independente și identic distribuite; $\\mathrm{SE} = \\hat\\alpha/\\sqrt{k}$, CI de 95\\% $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$')]),
    (T('Returns are not independent: large losses come in clusters (volatility clustering)', 'Randamentele nu sînt independente: pierderile mari vin în grupuri (gruparea volatilității)'),
     [T('\\textbf{moving-block bootstrap}: resample blocks of 20 consecutive days, recompute $\\hat\\alpha$, take the standard deviation',
        '\\textbf{bootstrap pe blocuri mobile}: reeșantionăm blocuri de 20 de zile consecutive, recalculăm $\\hat\\alpha$, luăm abaterea standard'),
      T('the blocks keep the clusters together, so the standard error is honest about the dependence',
        'blocurile păstrează grupurile împreună, deci eroarea standard ține cont de dependență')]),
    T('Same rule for every series: $k = 2.5\\%$ of $n$, 500 bootstrap samples', 'Aceeași regulă pentru fiecare serie: $k = 2.5\\%$ din $n$, 500 de eșantioane bootstrap')))


def hrow(k):
    return (f'{NAMES[k]} & @{{h.{k}.n}} & @{{h.{k}.k}} & $@{{h.{k}.alpha}}$ & $[@{{h.{k}.lo}}⟦, ||; ⟧@{{h.{k}.hi}}]$ & $@{{h.{k}.se_iid}}$ & '
            f'$@{{h.{k}.se_block}}$ & $@{{h.{k}.alpha_1}}$ / $@{{h.{k}.alpha_5}}$ & $@{{h.{k}.alpha_gain}}$')


D.frame(T('Hill Estimates for Four Markets', 'Estimări Hill pentru patru piețe'), table(
    'lrrrrrrrr', T('& $n$ & $k$ & $\\hat\\alpha$ & 95\\% CI & SE i.i.d. & SE block & $k = 1\\%$ / $5\\%$ & gains',
                   '& $n$ & $k$ & $\\hat\\alpha$ & CI 95\\% & SE i.i.d. & SE blocuri & $k = 1\\%$ / $5\\%$ & cîștiguri'),
    [hrow(k) for k in ASSETS], size='scriptsize') + items(
    T('Losses: $\\hat\\alpha$ between $@{h.min}$ and $@{h.max}$; the intervals exclude 2 (finite variance) and lie below 4 (no finite kurtosis)',
      'Pierderi: $\\hat\\alpha$ între $@{h.min}$ și $@{h.max}$; intervalele exclud 2 (varianță finită) și sînt sub 4 (kurtosis infinit)'),
    T('The block-bootstrap SE is close to the i.i.d.\\ one here; the estimates move with $k$ by more than one SE',
      'SE din bootstrap pe blocuri este apropiată aici de cea i.i.d.; estimările se mișcă cu $k$ cu mai mult de o SE'),
    T('Last column: the Hill index of gains ($r_t > 0$), same $k$', 'Ultima coloană: indicele Hill al cîștigurilor ($r_t > 0$), același $k$')) + ql('SFM_ch5_hill'), 'footnotesize')

D.frame(T('From the Tail Index to a Quantile', 'De la indicele de coadă la o cuantilă'), items(
    (T('\\textbf{Hill (Weissman) quantile} \\refWeissman: $\\hat x_p = L_{(k+1)}\\Big(\\dfrac{k}{n p}\\Big)^{1/\\hat\\alpha}$', '\\textbf{Cuantila Hill (Weissman)} \\refWeissman: $\\hat x_p = L_{(k+1)}\\Big(\\dfrac{k}{n p}\\Big)^{1/\\hat\\alpha}$'),
     [T('the loss exceeded with probability $p$, extrapolated from the threshold with the power law; SFE Quantlet SFEhillquantile',
        'pierderea depășită cu probabilitatea $p$, extrapolată de la prag cu legea putere; Quantlet-ul SFE SFEhillquantile')]),
    (T('\\textbf{Worked example}, S\\&P 500, VaR 0.1\\%: $n = @{h.sp500.n}$, $k = @{h.sp500.k}$, $L_{(k+1)} = @{h.sp500.u}\\%$, $\\hat\\alpha = @{h.sp500.alpha}$',
       '\\textbf{Exemplu lucrat}, S\\&P 500, VaR 0,1\\%: $n = @{h.sp500.n}$, $k = @{h.sp500.k}$, $L_{(k+1)} = @{h.sp500.u}\\%$, $\\hat\\alpha = @{h.sp500.alpha}$'),
     [T('$k/(np) = @{h.sp500.k}/@{hq.np} = @{hq.ratio}$; $@{hq.ratio}^{1/@{h.sp500.alpha}} = @{hq.fac}$', '$k/(np) = @{h.sp500.k}/@{hq.np} = @{hq.ratio}$; $@{hq.ratio}^{1/@{h.sp500.alpha}} = @{hq.fac}$'),
      T('$\\hat x_{0.001} = @{h.sp500.u} \\times @{hq.fac} = @{hq.var}\\%$; empirical 0.1\\% quantile: $@{r.sp500.var01.Historical}\\%$',
        '$\\hat x_{0.001} = @{h.sp500.u} \\times @{hq.fac} = @{hq.var}\\%$; cuantila empirică de 0,1\\%: $@{r.sp500.var01.Historical}\\%$')]),
    T('The extrapolation is only as good as $\\hat\\alpha$: an error of one SE in $\\hat\\alpha$ moves $\\hat x_p$ noticeably',
      'Extrapolarea este atît de bună cît este $\\hat\\alpha$: o eroare de o SE în $\\hat\\alpha$ mută vizibil $\\hat x_p$')) + ql('SFM_ch5_hill'))


def brow(k):
    return (f'{NAMES[k]} & @{{h.{k}.n}} & $@{{h.{k}.alpha}}$ & $[@{{h.{k}.lo}}⟦, ||; ⟧@{{h.{k}.hi}}]$ & $@{{h.{k}.se_block}}$ & $@{{h.{k}.alpha_gain}}$')


D.frame(T('Four Stocks of the Bucharest Stock Exchange', 'Patru acțiuni de la Bursa de Valori București'), cols(
    table('lrrrrr', T('& $n$ & $\\hat\\alpha$ & 95\\% CI & SE block & gains', '& $n$ & $\\hat\\alpha$ & CI 95\\% & SE blocuri & cîștiguri'),
          [brow(k) for k in STOCKS], size='scriptsize') + items(
        T('Daily log returns of the adjusted close since 2010, $k = 2.5\\%$ of $n$', 'Randamente logaritmice zilnice ale prețului ajustat din 2010, $k = 2.5\\%$ din $n$'),
        T('TLV: the two days 30--31 May 2016 are removed, an adjustment applied one day late (Chapter 2)',
          'TLV: zilele de 30--31 mai 2016 sînt eliminate, o ajustare aplicată cu o zi mai tîrziu (Capitolul 2)'),
        T('All four: $\\hat\\alpha$ between $@{hs.min}$ and $@{hs.max}$, as for the indices; the losses have heavier tails than the gains', 'Toate patru: $\\hat\\alpha$ între $@{hs.min}$ și $@{hs.max}$, ca la indici; pierderile au cozi mai groase decît cîștigurile')),
    pic('ch5_bvb_palace_2019.jpg', 'The Stock Exchange Palace, Bucharest', 'Palatul Bursei, București',
        'https://commons.wikimedia.org/wiki/File:Stock_Exchange_Palace_(Bucharest).jpg',
        'Photo: Neoclassicism Enthusiast (2019); CC BY-SA 4.0; Wikimedia Commons', 'Foto: Neoclassicism Enthusiast (2019); CC BY-SA 4.0; Wikimedia Commons',
        h='0.32\\textheight'), wl='0.60', wr='0.36') + ql('SFM_ch5_hill'), 'footnotesize')

chart(T('Hill Plots of the BVB Stocks', 'Grafice Hill pentru acțiunile BVB'), 'sfm_ch5_hill_bvb', 'SFM_ch5_hill', [
    T('Shorter samples ($n \\approx 3\\,800$): wider bands (TLV shown) and more wiggles at small $k$', 'Eșantioane mai scurte ($n \\approx 3\\,800$): benzi mai largi (arătată pentru TLV) și mai multe oscilații la $k$ mic'),
    T('No stock has $\\hat\\alpha$ below 2 in the flat region: variances are finite', 'Nicio acțiune nu are $\\hat\\alpha$ sub 2 în zona plată: varianțele sînt finite')], h='0.58\\textheight')

D.recap(('The Hill Estimator', 'Estimatorul Hill'), [
    T('$\\hat\\alpha_k = [\\frac1k\\sum_{i \\le k}\\ln(L_{(i)}/L_{(k+1)})]^{-1}$, the ML estimator of a Pareto tail', '$\\hat\\alpha_k = [\\frac1k\\sum_{i \\le k}\\ln(L_{(i)}/L_{(k+1)})]^{-1}$, estimatorul ML al unei cozi Pareto'),
    T('Choose $k$ where the Hill plot is flat; fix the rule in advance', 'Alegeți $k$ acolo unde graficul Hill este plat; fixați regula dinainte'),
    T('SE $\\approx \\hat\\alpha/\\sqrt{k}$; block bootstrap for dependent returns', 'SE $\\approx \\hat\\alpha/\\sqrt{k}$; bootstrap pe blocuri pentru randamente dependente'),
    T('Indices and BVB stocks: $\\hat\\alpha$ between $@{ha.min}$ and $@{ha.max}$, finite variance, infinite kurtosis', 'Indici și acțiuni BVB: $\\hat\\alpha$ între $@{ha.min}$ și $@{ha.max}$, varianță finită, kurtosis infinit')])

# =============================================================================
# 4. FUNCȚIA MEAN EXCESS
# =============================================================================
D.section('The Mean Excess Function', 'Funcția mean excess')

D.frame(T('The Mean Excess Function', 'Funcția mean excess'), items(
    (T('\\textbf{Mean excess function}: $e(u) = E[L - u \\mid L > u]$, the average loss beyond $u$, given that $u$ is exceeded',
       '\\textbf{Funcția mean excess} (excesul mediu): $e(u) = E[L - u \\mid L > u]$, pierderea medie peste $u$, știind că $u$ este depășit'),
     [T('empirical version: the average of $L_i - u$ over the losses with $L_i > u$', 'versiunea empirică: media lui $L_i - u$ peste pierderile cu $L_i > u$')]),
    (T('Its shape tells the type of tail', 'Forma ei arată tipul de coadă'),
     [T('Exponential: $e(u)$ constant (no memory)', 'Exponențială: $e(u)$ constant (fără memorie)'),
      T('Pareto with $\\alpha > 1$: $e(u) = u/(\\alpha - 1)$, a rising straight line', 'Pareto cu $\\alpha > 1$: $e(u) = u/(\\alpha - 1)$, o dreaptă crescătoare'),
      T('Normal: $e(u)$ falls towards 0 (roughly $1/u$ for a standard Normal)', 'Normală: $e(u)$ scade spre 0 (aproximativ $1/u$ pentru o Normală standard)')]),
    (T('\\textbf{Example}: Pareto tail with $\\alpha = 3$: $e(u) = u/2$', '\\textbf{Exemplu}: coadă Pareto cu $\\alpha = 3$: $e(u) = u/2$'),
     [T('beyond a 2\\% loss the average extra loss is $@{me.e2}\\%$; beyond 4\\% it is $@{me.e4}\\%$: the deeper, the worse',
        'peste o pierdere de 2\\%, pierderea suplimentară medie este $@{me.e2}\\%$; peste 4\\% este $@{me.e4}\\%$: cu cît mai adînc, cu atît mai rău')])))

chart(T('Empirical Mean Excess of Daily Losses', 'Mean excess empiric al pierderilor zilnice'), 'sfm_ch5_mean_excess', 'SFM_ch5_mean_excess', [
    T('All four functions rise above the 90\\% quantile $u$ (dotted): a heavy, Pareto-type tail; black: the line implied by the GPD fitted above $u$',
      'Toate cele patru funcții cresc peste cuantila de 90\\% $u$ (punctat): o coadă groasă, de tip Pareto; negru: dreapta implicată de GPD ajustată peste $u$'),
    T('The last points rest on very few losses and jump around: do not read them literally', 'Ultimele puncte se bazează pe foarte puține pierderi și sar: nu le citiți literal')], h='0.62\\textheight')

D.frame(T('Reading the Mean Excess Plot', 'Cum citim graficul mean excess'), items(
    (T('Look for the threshold above which $e(u)$ becomes roughly a straight line', 'Căutăm pragul peste care $e(u)$ devine aproximativ o dreaptă'),
     [T('a rising line means a heavy tail; its slope gives a first guess of the shape of the tail', 'o dreaptă crescătoare înseamnă o coadă groasă; panta ei dă o primă idee despre forma cozii')]),
    (T('At the 90\\% quantile: $e(u)$ is $@{me.sp500.e}\\%$ for the S\\&P 500 ($u = @{me.sp500.u}\\%$) and $@{me.bet.e}\\%$ for the BET ($u = @{me.bet.u}\\%$)',
       'La cuantila de 90\\%: $e(u)$ este $@{me.sp500.e}\\%$ pentru S\\&P 500 ($u = @{me.sp500.u}\\%$) și $@{me.bet.e}\\%$ pentru BET ($u = @{me.bet.u}\\%$)'),
     [T('on a day with a loss above $u$, the loss beyond $u$ is about as large as $u$ itself', 'într-o zi cu o pierdere peste $u$, pierderea dincolo de $u$ este cam cît $u$')]),
    T('The mean excess function is one of the tools for choosing the threshold of the peaks over threshold method (Section 6)',
      'Funcția mean excess este unul dintre instrumentele pentru alegerea pragului în metoda peaks over threshold (Secțiunea 6)'),
    T('Ported from the SFE Quantlet SFEMeanExcessFun', 'Portată din Quantlet-ul SFE SFEMeanExcessFun')) + ql('SFM_ch5_mean_excess'))

# =============================================================================
# 5. BLOCK MAXIMA ȘI GEV
# =============================================================================
D.section('Block Maxima and the GEV Distribution', 'Block maxima și distribuția GEV')

D.frame(T('Block Maxima', 'Maxime pe blocuri (block maxima)'), cols(items(
    (T('Split the losses into blocks of equal length and keep the largest loss of each block', 'Împărțim pierderile în blocuri de aceeași lungime și păstrăm cea mai mare pierdere din fiecare bloc'),
     [T('$M_n = \\max(L_1, \\dots, L_n)$, here the largest daily loss of each calendar month ($n \\approx 21$)',
        '$M_n = \\max(L_1, \\dots, L_n)$, aici cea mai mare pierdere zilnică din fiecare lună calendaristică ($n \\approx 21$)')]),
    (T('Question: what is the law of $M_n$ for large $n$?', 'Întrebarea: care este legea lui $M_n$ pentru $n$ mare?'),
     [T('the analogue of the central limit theorem, for maxima instead of sums', 'analogul teoremei limită centrale, pentru maxime în loc de sume')]),
    T('\\refFT\\ found the three possible limit laws, studying the largest of a sample', '\\refFT\\ au găsit cele trei legi limită posibile, studiind cel mai mare element al unui eșantion')),
    pic('ch5_fisher_1913.jpg', 'Ronald A. Fisher (1890--1962), in 1913', 'Ronald A. Fisher (1890--1962), în 1913',
        'https://commons.wikimedia.org/wiki/File:Youngronaldfisher2.JPG',
        'Photo: unknown author (1913); public domain; Wikimedia Commons', 'Foto: autor necunoscut (1913); domeniu public; Wikimedia Commons',
        h='0.42\\textheight'), wl='0.62', wr='0.34'))

D.frame(T('The Fisher--Tippett--Gnedenko Theorem', 'Teorema Fisher--Tippett--Gnedenko'), items(
    (T('$L_1, L_2, \\dots$ i.i.d. If there are constants $a_n > 0$, $b_n$ such that $P\\Big(\\dfrac{M_n - b_n}{a_n} \\le x\\Big) \\to H(x)$, non-degenerate,',
       '$L_1, L_2, \\dots$ i.i.d. Dacă există constante $a_n > 0$, $b_n$ astfel încît $P\\Big(\\dfrac{M_n - b_n}{a_n} \\le x\\Big) \\to H(x)$, nedegenerată,'),
     [T('then $H$ is a \\textbf{GEV} (generalised extreme value) distribution \\refFT, \\refGnedenko', 'atunci $H$ este o distribuție \\textbf{GEV} (generalised extreme value, distribuția generalizată a valorilor extreme) \\refFT, \\refGnedenko')]),
    (T('\\textbf{GEV} with shape $\\xi$, location $\\mu$, scale $\\sigma > 0$ (the unified form of \\refJenkinson)', '\\textbf{GEV} cu forma $\\xi$, locația $\\mu$, scala $\\sigma > 0$ (forma unificată a lui \\refJenkinson)'),
     [T('$H(x) = \\exp\\Big\\{-\\Big[1 + \\xi\\,\\dfrac{x - \\mu}{\\sigma}\\Big]^{-1/\\xi}\\Big\\}$ for $1 + \\xi(x - \\mu)/\\sigma > 0$',
        '$H(x) = \\exp\\Big\\{-\\Big[1 + \\xi\\,\\dfrac{x - \\mu}{\\sigma}\\Big]^{-1/\\xi}\\Big\\}$ pentru $1 + \\xi(x - \\mu)/\\sigma > 0$'),
      T('$\\xi = 0$: $H(x) = \\exp\\{-e^{-(x - \\mu)/\\sigma}\\}$, the limit as $\\xi \\to 0$', '$\\xi = 0$: $H(x) = \\exp\\{-e^{-(x - \\mu)/\\sigma}\\}$, limita pentru $\\xi \\to 0$')]),
    T('Whatever the law of the losses, the normalised maximum has only one family of possible limits',
      'Oricare ar fi legea pierderilor, maximul normalizat are o singură familie de limite posibile')))

D.frame(T('Three Types in One Family', 'Trei tipuri într-o singură familie'), cols(items(
    (T('\\textbf{Fréchet} ($\\xi > 0$): heavy tail, $1 - H(x) \\sim x^{-1/\\xi}$', '\\textbf{Fréchet} ($\\xi > 0$): coadă groasă, $1 - H(x) \\sim x^{-1/\\xi}$'),
     [T('tail index $\\alpha = 1/\\xi$; the case of financial losses', 'indicele de coadă $\\alpha = 1/\\xi$; cazul pierderilor financiare')]),
    (T('\\textbf{Gumbel} ($\\xi = 0$): light, exponential tail', '\\textbf{Gumbel} ($\\xi = 0$): coadă subțire, exponențială'),
     [T('maxima of Normal, lognormal and Exponential variables', 'maximele variabilelor Normale, lognormale și Exponențiale')]),
    (T('\\textbf{Weibull} ($\\xi < 0$): a finite upper end point $\\mu - \\sigma/\\xi$', '\\textbf{Weibull} ($\\xi < 0$): un capăt superior finit $\\mu - \\sigma/\\xi$'),
     [T('maxima of bounded variables (Uniform, Beta)', 'maximele variabilelor mărginite (Uniformă, Beta)')]),
    T('The set of laws whose maxima converge to a given $H$ is its \\textbf{maximum domain of attraction}',
      'Mulțimea legilor ale căror maxime converg către un $H$ dat este \\textbf{domeniul de atracție al maximului} său')),
    pic('ch5_frechet.jpg', 'Maurice Fréchet (1878--1973)', 'Maurice Fréchet (1878--1973)',
        'https://commons.wikimedia.org/wiki/File:Frechet.jpeg',
        'Photo: unknown author; public domain; Wikimedia Commons', 'Foto: autor necunoscut; domeniu public; Wikimedia Commons',
        h='0.40\\textheight'), wl='0.64', wr='0.32'))

chart(T('The Three Extreme Value Laws', 'Cele trei legi ale valorilor extreme'), 'sfm_ch5_gev_types', 'SFM_ch5_gev_block_maxima', [
    T('Standard location and scale ($\\mu = 0$, $\\sigma = 1$): Fréchet has the longest right tail, Weibull stops at $x = 2$',
      'Locația și scala standard ($\\mu = 0$, $\\sigma = 1$): Fréchet are cea mai lungă coadă dreaptă, Weibull se oprește la $x = 2$'),
    T('Ported from the SFE Quantlet SFEevt1', 'Portat din Quantlet-ul SFE SFEevt1')], h='0.56\\textheight')

chart(T('The Theorem in a Simulation', 'Teorema într-o simulare'), 'sfm_ch5_maxima_sim', 'SFM_ch5_gev_block_maxima', [
    T('5,000 samples of $n = 250$; Exponential maxima minus $\\ln n$ $\\to$ Gumbel; Pareto(3) maxima over $n^{1/3}$ $\\to$ Fréchet with $\\xi = 1/3$; Uniform: $n(M_n - 1)$ $\\to$ Weibull',
      '5\\,000 de eșantioane de $n = 250$; maximele Exponențiale minus $\\ln n$ $\\to$ Gumbel; maximele Pareto(3) împărțite la $n^{1/3}$ $\\to$ Fréchet cu $\\xi = 1/3$; Uniformă: $n(M_n - 1)$ $\\to$ Weibull'),
    T('\\textbf{Question for the room}: daily losses have a tail index near 3; which type do their monthly maxima follow?',
      '\\textbf{Întrebare pentru sală}: pierderile zilnice au un indice de coadă în jur de 3; ce tip urmează maximele lor lunare?'),
    T('\\textbf{Answer}: Fréchet, with $\\xi = 1/\\alpha \\approx 0.3$', '\\textbf{Răspuns}: Fréchet, cu $\\xi = 1/\\alpha \\approx 0{,}3$')], h='0.44\\textheight')


def grow(k):
    return (f'{NAMES[k]} & @{{g.{k}.months}} & $@{{g.{k}.xi}}$ & $[@{{g.{k}.xi_lo}}⟦, ||; ⟧@{{g.{k}.xi_hi}}]$ & $@{{g.{k}.mu}}$ & $@{{g.{k}.sigma}}$ & $@{{g.{k}.alpha}}$')


D.frame(T('GEV Fits to Monthly Maxima', 'Ajustări GEV pe maximele lunare'), table(
    'lrrrrrr', T('& months & $\\hat\\xi$ & 95\\% CI & $\\hat\\mu$ (\\%) & $\\hat\\sigma$ (\\%) & $1/\\hat\\xi$',
                 '& luni & $\\hat\\xi$ & CI 95\\% & $\\hat\\mu$ (\\%) & $\\hat\\sigma$ (\\%) & $1/\\hat\\xi$'),
    [grow(k) for k in ASSETS], size='footnotesize') + items(
    T('Maximum likelihood on the monthly maxima of daily losses; SE from the curvature of the log-likelihood',
      'Verosimilitate maximă pe maximele lunare ale pierderilor zilnice; SE din curbura log-verosimilității'),
    T('$\\hat\\xi > 0$ for all four series: Fréchet type, heavy tails; the intervals exclude 0 (Gumbel)',
      '$\\hat\\xi > 0$ pentru toate cele patru serii: tip Fréchet, cozi groase; intervalele exclud 0 (Gumbel)'),
    T('$1/\\hat\\xi$ is a second estimate of the tail index; with only one loss per month it is less precise than Hill',
      '$1/\\hat\\xi$ este o a doua estimare a indicelui de coadă; cu o singură pierdere pe lună este mai puțin precisă decît Hill')) + ql('SFM_ch5_gev_block_maxima'), 'footnotesize')

chart(T('Does the GEV Fit? PP and QQ Plots', 'Se potrivește GEV? Grafice PP și QQ'), 'sfm_ch5_gev_ppqq', 'SFM_ch5_gev_block_maxima', [
    T('\\textbf{PP plot}: fitted probability $H(m_{(i)})$ against the empirical $(i - 0.5)/n$; \\textbf{QQ plot}: observed maxima against fitted quantiles; S\\&P 500',
      '\\textbf{Graficul PP}: probabilitatea ajustată $H(m_{(i)})$ față de cea empirică $(i - 0{,}5)/n$; \\textbf{graficul QQ}: maximele observate față de cuantilele ajustate; S\\&P 500'),
    T('Both close to the 45-degree line; the QQ plot shows the few largest months (2008, 2020) a little above: the hardest part to fit',
      'Ambele aproape de dreapta de 45 de grade; graficul QQ arată cele cîteva luni extreme (2008, 2020) puțin deasupra: partea cea mai greu de ajustat')], h='0.54\\textheight')

D.frame(T('Return Levels', 'Return levels (niveluri de revenire)'), items(
    (T('\\textbf{Return level} for a return period of $T$ years: the level that the monthly maximum exceeds on average once every $T$ years',
       '\\textbf{Return level} pentru o perioadă de revenire de $T$ ani: nivelul pe care maximul lunar îl depășește, în medie, o dată la $T$ ani'),
     [T('$z_T = H^{-1}(1 - 1/m)$ with $m = 12T$ months: $z_T = \\mu + \\dfrac{\\sigma}{\\xi}\\big[(-\\ln(1 - 1/m))^{-\\xi} - 1\\big]$',
        '$z_T = H^{-1}(1 - 1/m)$ cu $m = 12T$ luni: $z_T = \\mu + \\dfrac{\\sigma}{\\xi}\\big[(-\\ln(1 - 1/m))^{-\\xi} - 1\\big]$')]),
    (T('\\textbf{Worked example}, S\\&P 500, $T = 10$ years: $\\hat\\xi = @{ge.xi}$, $\\hat\\mu = @{ge.mu}$, $\\hat\\sigma = @{ge.sigma}$',
       '\\textbf{Exemplu lucrat}, S\\&P 500, $T = 10$ ani: $\\hat\\xi = @{ge.xi}$, $\\hat\\mu = @{ge.mu}$, $\\hat\\sigma = @{ge.sigma}$'),
     [T('$m = 120$; $-\\ln(1 - 1/120) = @{rl.y}$; raised to $-\\hat\\xi$: $@{rl.pow}$', '$m = 120$; $-\\ln(1 - 1/120) = @{rl.y}$; ridicat la $-\\hat\\xi$: $@{rl.pow}$'),
      T('$z_{10} = @{ge.mu} + (@{ge.sigma}/@{ge.xi})(@{rl.pow} - 1) = @{rl.check}\\%$', '$z_{10} = @{ge.mu} + (@{ge.sigma}/@{ge.xi})(@{rl.pow} - 1) = @{rl.check}\\%$')]),
    T('A daily loss of about $@{g.sp500.rl10}\\%$ in the S\\&P 500 is a ``10-year event\'\'; it was exceeded in @{g.sp500.nab} months of @{g.sp500.years} years',
      'O pierdere zilnică de circa $@{g.sp500.rl10}\\%$ în S\\&P 500 este un „eveniment de 10 ani”; a fost depășită în @{g.sp500.nab} luni din @{g.sp500.years} de ani')))

chart(T('Return Level Plot', 'Graficul return levels'), 'sfm_ch5_return_levels', 'SFM_ch5_gev_block_maxima', [
    T('Lines: GEV return levels; dots: observed monthly maxima at their empirical return periods',
      'Linii: return levels GEV; puncte: maximele lunare observate, la perioadele lor empirice de revenire'),
    T('Beyond the length of the sample (@{g.sp500.years} years for the S\\&P 500, @{g.btc.years} for Bitcoin) the curves are extrapolation', 'Dincolo de lungimea eșantionului (@{g.sp500.years} de ani pentru S\\&P 500, @{g.btc.years} pentru Bitcoin), curbele sînt extrapolare')], h='0.58\\textheight')


def rrow(k):
    return (f'{NAMES[k]} & $@{{g.{k}.rl1}}$ & $@{{g.{k}.rl10}}$ & $[@{{g.{k}.rl10_lo}}⟦, ||; ⟧@{{g.{k}.rl10_hi}}]$ & $@{{g.{k}.rl50}}$ & $@{{g.{k}.max}}$ & $@{{g.{k}.years}}$')


D.frame(T('How Large Is a 10-Year Loss?', 'Cît de mare este o pierdere de 10 ani?'), table(
    'lrrrrrr', T('& 1 year & 10 years & 95\\% interval & 50 years & largest observed & years of data',
                 '& 1 an & 10 ani & interval 95\\% & 50 de ani & cea mai mare observată & ani de date'),
    [rrow(k) for k in ASSETS], size='footnotesize') + items(
    T('Daily loss (\\%) exceeded on average once in $T$ years; interval: bootstrap of the monthly maxima (200 samples)',
      'Pierderea zilnică (\\%) depășită în medie o dată la $T$ ani; intervalul: bootstrap al maximelor lunare (200 de eșantioane)'),
    T('BET: $\\hat\\xi = @{g.bet.xi}$ makes the 50-year level $@{g.bet.rl50}\\%$, almost twice the largest loss seen: a small change in $\\hat\\xi$, a large change far out',
      'BET: $\\hat\\xi = @{g.bet.xi}$ face nivelul de 50 de ani $@{g.bet.rl50}\\%$, aproape dublul celei mai mari pierderi observate: o mică schimbare în $\\hat\\xi$, o schimbare mare în extrem'),
    T('Block maxima use one loss per month: the peaks over threshold method uses all large losses', 'Block maxima folosesc o singură pierdere pe lună: metoda peaks over threshold folosește toate pierderile mari')) + ql('SFM_ch5_gev_block_maxima'), 'footnotesize')

D.recap(('Block Maxima', 'Block maxima'), [
    T('Maxima of i.i.d.\\ variables can only converge to a GEV law (Fisher--Tippett--Gnedenko)', 'Maximele variabilelor i.i.d.\\ pot converge doar către o lege GEV (Fisher--Tippett--Gnedenko)'),
    T('$\\xi > 0$ Fréchet (heavy, $\\alpha = 1/\\xi$), $\\xi = 0$ Gumbel, $\\xi < 0$ Weibull', '$\\xi > 0$ Fréchet (groasă, $\\alpha = 1/\\xi$), $\\xi = 0$ Gumbel, $\\xi < 0$ Weibull'),
    T('Monthly maxima of daily losses: Fréchet, $\\hat\\xi$ between $@{gxi.min}$ and $@{gxi.max}$', 'Maximele lunare ale pierderilor zilnice: Fréchet, $\\hat\\xi$ între $@{gxi.min}$ și $@{gxi.max}$'),
    T('Return level: the loss exceeded once in $T$ years; far beyond the sample it is extrapolation', 'Return level: pierderea depășită o dată la $T$ ani; mult dincolo de eșantion este extrapolare')])

# =============================================================================
# 6. PEAKS OVER THRESHOLD ȘI GPD
# =============================================================================
D.section('Peaks over Threshold and the GPD', 'Peaks over threshold și distribuția GPD')

D.frame(T('Peaks over Threshold', 'Peaks over threshold (vîrfuri peste prag)'), cols(items(
    (T('\\textbf{POT} (peaks over threshold): fix a high threshold $u$ and keep every loss above it', '\\textbf{POT} (peaks over threshold, vîrfuri peste prag): fixăm un prag înalt $u$ și păstrăm fiecare pierdere de peste el'),
     [T('the \\textbf{excess} over $u$: $Y = L - u$, given $L > u$', '\\textbf{excesul} peste $u$: $Y = L - u$, știind că $L > u$'),
      T('excess distribution: $F_u(y) = P(L - u \\le y \\mid L > u)$', 'distribuția excesului: $F_u(y) = P(L - u \\le y \\mid L > u)$')]),
    T('Uses all large losses, not one per block: more data in the tail', 'Folosește toate pierderile mari, nu una pe bloc: mai multe date în coadă'),
    T('The method behind most EVT estimates of VaR and ES \\refMF, \\refQRM', 'Metoda din spatele majorității estimărilor EVT pentru VaR și ES \\refMF, \\refQRM')),
    pic('ch5_dehaan_1987.jpg', 'Laurens de Haan, Oberwolfach, 1987', 'Laurens de Haan, Oberwolfach, 1987',
        'https://commons.wikimedia.org/wiki/File:Laurens_de_haan.jpg',
        'Photo: Konrad Jacobs (1987), Oberwolfach Photo Collection; CC BY-SA 2.0 de; Wikimedia Commons',
        'Foto: Konrad Jacobs (1987), Oberwolfach Photo Collection; CC BY-SA 2.0 de; Wikimedia Commons', h='0.36\\textheight'), wl='0.58', wr='0.38'))

D.frame(T('The Pickands--Balkema--de Haan Theorem', 'Teorema Pickands--Balkema--de Haan'), items(
    (T('If the maxima of $L$ converge to a GEV with shape $\\xi$, then for high thresholds', 'Dacă maximele lui $L$ converg către o GEV cu forma $\\xi$, atunci pentru praguri înalte'),
     [T('$F_u(y) \\approx G_{\\xi, \\beta(u)}(y)$, the \\textbf{GPD} (generalised Pareto distribution) \\refPickands, \\refBdH',
        '$F_u(y) \\approx G_{\\xi, \\beta(u)}(y)$, \\textbf{GPD} (generalised Pareto distribution, distribuția Pareto generalizată) \\refPickands, \\refBdH')]),
    (T('$G_{\\xi, \\beta}(y) = 1 - \\Big(1 + \\xi\\,\\dfrac{y}{\\beta}\\Big)^{-1/\\xi}$, $y \\ge 0$ (for $\\xi < 0$: $y \\le -\\beta/\\xi$); $\\xi = 0$: $1 - e^{-y/\\beta}$',
       '$G_{\\xi, \\beta}(y) = 1 - \\Big(1 + \\xi\\,\\dfrac{y}{\\beta}\\Big)^{-1/\\xi}$, $y \\ge 0$ (pentru $\\xi < 0$: $y \\le -\\beta/\\xi$); $\\xi = 0$: $1 - e^{-y/\\beta}$'),
     [T('shape $\\xi$: the \\textbf{same} as in the GEV; scale $\\beta > 0$ depends on $u$', 'forma $\\xi$: \\textbf{aceeași} ca la GEV; scala $\\beta > 0$ depinde de $u$'),
      T('$\\xi > 0$: Pareto-type tail with $\\alpha = 1/\\xi$; $\\xi = 0$: Exponential; $\\xi < 0$: bounded', '$\\xi > 0$: coadă de tip Pareto cu $\\alpha = 1/\\xi$; $\\xi = 0$: Exponențială; $\\xi < 0$: mărginită')]),
    (T('Moments of the GPD: mean $\\beta/(1 - \\xi)$ if $\\xi < 1$; variance finite if $\\xi < 1/2$', 'Momentele GPD: media $\\beta/(1 - \\xi)$ dacă $\\xi < 1$; varianța finită dacă $\\xi < 1/2$'),
     [T('mean excess of a GPD: $e(v) = \\dfrac{\\beta + \\xi(v - u)}{1 - \\xi}$, a straight line in $v > u$, as seen in Section 4',
        'mean excess-ul unei GPD: $e(v) = \\dfrac{\\beta + \\xi(v - u)}{1 - \\xi}$, o dreaptă în $v > u$, ca în Secțiunea 4')])))

chart(T('GPD Densities and Tails', 'Densități și cozi GPD'), 'sfm_ch5_gpd_densities', 'SFM_ch5_pot_gpd', [
    T('$\\beta = 1$; the densities look alike near 0, the tails do not: on log-log axes only $\\xi > 0$ gives straight lines',
      '$\\beta = 1$; densitățile seamănă lîngă 0, cozile nu: pe axe log-log, doar $\\xi > 0$ dă drepte'),
    T('$\\xi = 0.3$ corresponds to a tail index $\\alpha \\approx 3.3$, close to stock returns', '$\\xi = 0{,}3$ corespunde unui indice de coadă $\\alpha \\approx 3{,}3$, apropiat de randamentele acțiunilor')], h='0.56\\textheight')

D.frame(T('Choosing the Threshold', 'Alegerea pragului'), items(
    (T('The same trade-off as $k$ in the Hill estimator', 'Același compromis ca $k$ la estimatorul Hill'),
     [T('$u$ too low: the GPD approximation is poor (bias); $u$ too high: few excesses (variance)',
        '$u$ prea jos: aproximarea GPD este slabă (deplasare); $u$ prea sus: puține excese (varianță)')]),
    (T('Graphical tools', 'Instrumente grafice'),
     [T('mean excess plot: roughly linear above $u$', 'graficul mean excess: aproximativ liniar peste $u$'),
      T('parameter stability plot: $\\hat\\xi$ should not change much when $u$ is raised further', 'graficul stabilității parametrilor: $\\hat\\xi$ nu ar trebui să se schimbe mult cînd $u$ crește în continuare')]),
    (T('Rule used here: $u =$ the 90\\% quantile of the losses, i.e.\\ the 10\\% largest losses', 'Regula folosită aici: $u =$ cuantila de 90\\% a pierderilor, adică cele mai mari 10\\% dintre pierderi'),
     [T('\\refMF\\ use $k = 100$ excesses in windows of $n = 1000$ days, the same share', '\\refMF\\ folosesc $k = 100$ de excese în ferestre de $n = 1000$ de zile, aceeași proporție'),
      T('for the S\\&P 500: $u = @{r.sp500.u}\\%$ and $N_u = @{r.sp500.nu}$ excesses', 'pentru S\\&P 500: $u = @{r.sp500.u}\\%$ și $N_u = @{r.sp500.nu}$ excese')])))

chart(T('Parameter Stability', 'Stabilitatea parametrilor'), 'sfm_ch5_threshold_stability', 'SFM_ch5_pot_gpd', [
    T('Left: $\\hat\\xi$ with 95\\% interval for $u$ from the 80\\% to the 99\\% quantile; right: the EVT VaR 1\\% for the same thresholds',
      'Stînga: $\\hat\\xi$ cu interval de 95\\% pentru $u$ de la cuantila de 80\\% la cea de 99\\%; dreapta: VaR 1\\% EVT pentru aceleași praguri'),
    T('Between the 85\\% and 95\\% quantiles: S\\&P 500 $\\hat\\xi \\in [@{st.sp500.xmin}, @{st.sp500.xmax}]$, VaR 1\\% $\\in [@{st.sp500.vmin}, @{st.sp500.vmax}]\\%$; the VaR barely moves',
      'Între cuantilele de 85\\% și 95\\%: S\\&P 500 $\\hat\\xi \\in [@{st.sp500.xmin}; @{st.sp500.xmax}]$, VaR 1\\% $\\in [@{st.sp500.vmin}; @{st.sp500.vmax}]\\%$; VaR aproape nu se mișcă')], h='0.54\\textheight')


def prow(k):
    return (f'{NAMES[k]} & @{{r.{k}.n}} & $@{{r.{k}.u}}$ & @{{r.{k}.nu}} & $@{{r.{k}.xi}}$ & $@{{r.{k}.se_xi}}$ & $@{{r.{k}.beta}}$ & $@{{r.{k}.alpha}}$')


D.frame(T('GPD Fits above the 90\\% Quantile', 'Ajustări GPD peste cuantila de 90\\%'), table(
    'lrrrrrrr', T('& $n$ & $u$ (\\%) & $N_u$ & $\\hat\\xi$ & SE & $\\hat\\beta$ & $1/\\hat\\xi$', '& $n$ & $u$ (\\%) & $N_u$ & $\\hat\\xi$ & SE & $\\hat\\beta$ & $1/\\hat\\xi$'),
    [prow(k) for k in ASSETS], size='footnotesize') + items(
    T('Maximum likelihood on the excesses $L - u$ \\refDS; SE from the curvature of the log-likelihood',
      'Verosimilitate maximă pe excesele $L - u$ \\refDS; SE din curbura log-verosimilității'),
    T('$\\hat\\xi$ between $@{xi.min}$ and $@{xi.max}$: heavy tails with a finite variance ($\\xi < 0.5$) for all four', '$\\hat\\xi$ între $@{xi.min}$ și $@{xi.max}$: cozi groase cu varianță finită ($\\xi < 0.5$) pentru toate patru'),
    T('$1/\\hat\\xi$ is larger than the Hill $\\hat\\alpha$ (2.5\\% of the losses): above the 90\\% quantile, a GPD with a free scale $\\beta$ needs a smaller $\\xi$; the tail index is hard to pin down, so report more than one estimator',
      '$1/\\hat\\xi$ este mai mare decît $\\hat\\alpha$ Hill (2,5\\% din pierderi): peste cuantila de 90\\%, o GPD cu scala liberă $\\beta$ are nevoie de un $\\xi$ mai mic; indicele de coadă este greu de fixat, deci raportați mai mulți estimatori')) + ql('SFM_ch5_pot_gpd'), 'footnotesize')

chart(T('Does the GPD Fit? PP and QQ Plots', 'Se potrivește GPD? Grafice PP și QQ'), 'sfm_ch5_gpd_ppqq', 'SFM_ch5_pot_gpd', [
    T('S\\&P 500 excesses over $u$: the PP plot is on the line; the QQ plot is on the line up to excesses of about 4\\%',
      'Excesele S\\&P 500 peste $u$: graficul PP este pe dreaptă; graficul QQ este pe dreaptă pînă la excese de circa 4\\%'),
    T('The largest excess ($@{gq.y}\\%$, March 2020) against a fitted quantile of $@{gq.q}\\%$; ported from the SFE Quantlets SFEtailGPareto\\_pp and \\_qq',
      'Cel mai mare exces ($@{gq.y}\\%$, martie 2020) față de o cuantilă ajustată de $@{gq.q}\\%$; portat din Quantlet-urile SFE SFEtailGPareto\\_pp și \\_qq')], h='0.54\\textheight')

D.frame(T('From the GPD to the Tail of the Losses', 'De la GPD la coada pierderilor'), items(
    (T('For $x > u$: $P(L > x) = P(L > u)\\,P(L > x \\mid L > u)$', 'Pentru $x > u$: $P(L > x) = P(L > u)\\,P(L > x \\mid L > u)$'),
     [T('estimate $P(L > u)$ by $N_u/n$ and $P(L > x \\mid L > u)$ by the GPD', 'estimăm $P(L > u)$ prin $N_u/n$ și $P(L > x \\mid L > u)$ prin GPD')]),
    T('\\textbf{Tail estimator}: $\\hat{\\bar F}(x) = \\dfrac{N_u}{n}\\Big(1 + \\hat\\xi\\,\\dfrac{x - u}{\\hat\\beta}\\Big)^{-1/\\hat\\xi}$, $x > u$',
      '\\textbf{Estimatorul cozii}: $\\hat{\\bar F}(x) = \\dfrac{N_u}{n}\\Big(1 + \\hat\\xi\\,\\dfrac{x - u}{\\hat\\beta}\\Big)^{-1/\\hat\\xi}$, $x > u$'),
    (T('Historical data inside the sample, a fitted power law beyond it', 'Datele istorice în interiorul eșantionului, o lege putere ajustată dincolo de el'),
     [T('the model is used only where it is justified: above $u$', 'modelul se folosește doar acolo unde este justificat: peste $u$')]),
    T('Inverting $\\hat{\\bar F}(x) = p$ gives VaR; integrating beyond it gives ES (Section 7)', 'Inversarea lui $\\hat{\\bar F}(x) = p$ dă VaR; integrarea dincolo de el dă ES (Secțiunea 7)')))

chart(T('The Fitted Tails', 'Cozile ajustate'), 'sfm_ch5_tail_fit', 'SFM_ch5_pot_gpd', [
    T('Dots: share of days with a loss above $x$ (above $u$); blue: EVT tail estimator; red: Normal; dotted lines at $p = 1\\%$ and $0.1\\%$',
      'Puncte: proporția zilelor cu o pierdere peste $x$ (peste $u$); albastru: estimatorul EVT al cozii; roșu: Normală; linii punctate la $p = 1\\%$ și $0{,}1\\%$'),
    T('The GPD follows the data down to $10^{-4}$; the Normal tail leaves them before $p = 1\\%$', 'GPD urmează datele pînă la $10^{-4}$; coada Normală le părăsește înainte de $p = 1\\%$')], h='0.66\\textheight')


def wrow(k):
    return f'{NAMES[k]} & ' + T(f'@{{c.{k}.date}}', f'@{{c.{k}.date}}') + f' & ${{@{{c.{k}.ret}}}}$ & $10^{{@{{c.{k}.ly}}}}$ & $@{{w.{k}}}$'


D.frame(T('Back to the Crashes: How Often?', 'Înapoi la crahuri: cît de des?'), table(
    'llrrr', T('& day & loss (\\%) & Normal: once in (years) & EVT: once in (years)', '& ziua & pierdere (\\%) & Normală: o dată la (ani) & EVT: o dată la (ani)'),
    [wrow(k) for k in ASSETS] + [T('S\\&P 500, Black Monday', 'S\\&P 500, Lunea Neagră') + ' & 19.10.1987 & $@{c87.lr}$ & $10^{@{c87.ly}}$ & $@{w.1987}$'],
    size='footnotesize') + items(
    T('EVT waiting time: $1/(\\hat{\\bar F}(x) \\times 252)$ years (365 for Bitcoin), with the POT fit of the whole sample',
      'Timpul de așteptare EVT: $1/(\\hat{\\bar F}(x) \\times 252)$ ani (365 pentru Bitcoin), cu ajustarea POT pe întregul eșantion'),
    T('EVT makes the COVID-19 days rare but plausible events of a lifetime; the Normal law makes them impossible',
      'EVT face din zilele COVID-19 evenimente rare, dar plauzibile, de o viață; legea Normală le face imposibile'),
    T('1987 is outside our data: EVT extrapolates it to a waiting time of about @{w.1987} years, with a wide uncertainty',
      '1987 este în afara datelor noastre: EVT îl extrapolează la un timp de așteptare de circa @{w.1987} de ani, cu o incertitudine mare')) + ql('SFM_ch5_pot_gpd'), 'footnotesize')

D.recap(('Peaks over Threshold', 'Peaks over threshold'), [
    T('Excesses over a high threshold follow approximately a GPD (Pickands--Balkema--de Haan)', 'Excesele peste un prag înalt urmează aproximativ o GPD (Pickands--Balkema--de Haan)'),
    T('Same shape $\\xi$ as the GEV; $\\xi > 0$: power tail with $\\alpha = 1/\\xi$', 'Aceeași formă $\\xi$ ca la GEV; $\\xi > 0$: coadă putere cu $\\alpha = 1/\\xi$'),
    T('Threshold: mean excess and parameter stability; here the 90\\% quantile', 'Pragul: mean excess și stabilitatea parametrilor; aici cuantila de 90\\%'),
    T('Tail estimator: data up to $u$, GPD beyond; check with PP and QQ plots', 'Estimatorul cozii: datele pînă la $u$, GPD dincolo; verificare cu grafice PP și QQ')])

# =============================================================================
# 7. VaR ȘI ES PRIN EVT
# =============================================================================
D.section('VaR and ES by Extreme Value Theory', 'VaR și ES prin teoria valorilor extreme')

D.frame(T('VaR and ES: Definitions', 'VaR și ES: definiții'), items(
    (T('Losses $L = -r$ with distribution function $F_L$; level $p$ (e.g.\\ $p = 1\\%$)', 'Pierderile $L = -r$ cu funcția de repartiție $F_L$; nivelul $p$ (de ex.\\ $p = 1\\%$)'),
     [T('$\\mathrm{VaR}_p = F_L^{-1}(1 - p) = -q_p(r)$: the loss exceeded with probability $p$', '$\\mathrm{VaR}_p = F_L^{-1}(1 - p) = -q_p(r)$: pierderea depășită cu probabilitatea $p$'),
      T('$\\mathrm{ES}_p = \\dfrac{1}{p}\\displaystyle\\int_0^p \\mathrm{VaR}_s\\,ds = E[L \\mid L \\ge \\mathrm{VaR}_p]$ for continuous laws', '$\\mathrm{ES}_p = \\dfrac{1}{p}\\displaystyle\\int_0^p \\mathrm{VaR}_s\\,ds = E[L \\mid L \\ge \\mathrm{VaR}_p]$ pentru legi continue')]),
    (T('Three ways to estimate them from daily losses', 'Trei moduri de a le estima din pierderile zilnice'),
     [T('\\textbf{historical simulation}: the empirical quantile and the mean of the losses beyond it', '\\textbf{simularea istorică}: cuantila empirică și media pierderilor de dincolo de ea'),
      T('\\textbf{Normal}: $\\mathrm{VaR}_p = \\hat\\mu_L + \\hat\\sigma z_{1-p}$, $\\mathrm{ES}_p = \\hat\\mu_L + \\hat\\sigma\\,\\varphi(z_{1-p})/p$ ($\\varphi$: the standard Normal density)',
        '\\textbf{Normală}: $\\mathrm{VaR}_p = \\hat\\mu_L + \\hat\\sigma z_{1-p}$, $\\mathrm{ES}_p = \\hat\\mu_L + \\hat\\sigma\\,\\varphi(z_{1-p})/p$ ($\\varphi$: densitatea Normalei standard)'),
      T('\\textbf{EVT (POT)}: invert the tail estimator', '\\textbf{EVT (POT)}: inversăm estimatorul cozii')]),
    T('Levels used in practice: VaR 1\\% (daily, Basel backtesting), ES 2.5\\% \\refBasel, VaR 0.1\\% (stress)',
      'Nivelurile folosite în practică: VaR 1\\% (zilnic, backtesting Basel), ES 2,5\\% \\refBasel, VaR 0,1\\% (stres)')))

D.frame(T('EVT Formulas for VaR and ES', 'Formulele EVT pentru VaR și ES'), items(
    (T('Solve $\\dfrac{N_u}{n}\\Big(1 + \\xi\\,\\dfrac{x - u}{\\beta}\\Big)^{-1/\\xi} = p$ for $x$', 'Rezolvăm $\\dfrac{N_u}{n}\\Big(1 + \\xi\\,\\dfrac{x - u}{\\beta}\\Big)^{-1/\\xi} = p$ în raport cu $x$'),
     [T('$\\mathrm{VaR}_p = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{n p}{N_u}\\Big)^{-\\xi} - 1\\Big]$, valid for $p < N_u/n$', '$\\mathrm{VaR}_p = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{n p}{N_u}\\Big)^{-\\xi} - 1\\Big]$, valabil pentru $p < N_u/n$')]),
    (T('The excess over $\\mathrm{VaR}_p$ is again GPD with the same $\\xi$: its mean is $(\\beta + \\xi(\\mathrm{VaR}_p - u))/(1 - \\xi)$',
       'Excesul peste $\\mathrm{VaR}_p$ este tot GPD, cu același $\\xi$: media lui este $(\\beta + \\xi(\\mathrm{VaR}_p - u))/(1 - \\xi)$'),
     [T('$\\mathrm{ES}_p = \\dfrac{\\mathrm{VaR}_p}{1 - \\xi} + \\dfrac{\\beta - \\xi u}{1 - \\xi}$, valid for $\\xi < 1$ \\refMF', '$\\mathrm{ES}_p = \\dfrac{\\mathrm{VaR}_p}{1 - \\xi} + \\dfrac{\\beta - \\xi u}{1 - \\xi}$, valabil pentru $\\xi < 1$ \\refMF')]),
    T('With $\\xi > 0$, ES is a fixed multiple of VaR far in the tail: $\\mathrm{ES}_p/\\mathrm{VaR}_p \\to 1/(1 - \\xi)$',
      'Cu $\\xi > 0$, ES este un multiplu fix al VaR în extrem: $\\mathrm{ES}_p/\\mathrm{VaR}_p \\to 1/(1 - \\xi)$'),
    T('Ported from the SFE Quantlet var\\_pot (VaR estimates with the POT model)', 'Portat din Quantlet-ul SFE var\\_pot (estimări VaR cu modelul POT)')))

D.frame(T('Worked Example: S\\&P 500, Step by Step', 'Exemplu lucrat: S\\&P 500, pas cu pas'), items(
    T('Fit: $n = @{r.sp500.n}$, $u = @{pe.u}\\%$, $N_u = @{r.sp500.nu}$ ($@{pe.share}\\%$), $\\hat\\xi = @{pe.xi}$, $\\hat\\beta = @{pe.beta}$, $\\hat\\beta/\\hat\\xi = @{pe.boxi}$',
      'Ajustarea: $n = @{r.sp500.n}$, $u = @{pe.u}\\%$, $N_u = @{r.sp500.nu}$ ($@{pe.share}\\%$), $\\hat\\xi = @{pe.xi}$, $\\hat\\beta = @{pe.beta}$, $\\hat\\beta/\\hat\\xi = @{pe.boxi}$'),
    (T('\\textbf{VaR 1\\%}: $np/N_u = @{pe.r1}$; $@{pe.r1}^{-\\hat\\xi} = @{pe.pow1}$', '\\textbf{VaR 1\\%}: $np/N_u = @{pe.r1}$; $@{pe.r1}^{-\\hat\\xi} = @{pe.pow1}$'),
     [T('$\\mathrm{VaR}_{1\\%} = @{pe.u} + @{pe.boxi}\\,(@{pe.pow1} - 1) = @{pe.var1}\\%$', '$\\mathrm{VaR}_{1\\%} = @{pe.u} + @{pe.boxi}\\,(@{pe.pow1} - 1) = @{pe.var1}\\%$')]),
    (T('\\textbf{ES 2.5\\%}: $np/N_u = @{pe.r25}$, $\\mathrm{VaR}_{2.5\\%} = @{pe.u} + @{pe.boxi}\\,(@{pe.pow25} - 1) = @{pe.var25}\\%$',
       '\\textbf{ES 2,5\\%}: $np/N_u = @{pe.r25}$, $\\mathrm{VaR}_{2,5\\%} = @{pe.u} + @{pe.boxi}\\,(@{pe.pow25} - 1) = @{pe.var25}\\%$'),
     [T('$\\mathrm{ES}_{2.5\\%} = \\dfrac{@{pe.var25} + @{pe.beta} - @{pe.xi} \\times @{pe.u}}{1 - @{pe.xi}} = @{pe.es25}\\%$',
        '$\\mathrm{ES}_{2,5\\%} = \\dfrac{@{pe.var25} + @{pe.beta} - @{pe.xi} \\times @{pe.u}}{1 - @{pe.xi}} = @{pe.es25}\\%$')]),
    T('\\textbf{VaR 0.1\\%}: $np/N_u = @{pe.r01}$, $@{pe.r01}^{-\\hat\\xi} = @{pe.pow01}$, $\\mathrm{VaR}_{0.1\\%} = @{pe.var01}\\%$',
      '\\textbf{VaR 0,1\\%}: $np/N_u = @{pe.r01}$, $@{pe.r01}^{-\\hat\\xi} = @{pe.pow01}$, $\\mathrm{VaR}_{0,1\\%} = @{pe.var01}\\%$')))


def vrow(k):
    return (f'{NAMES[k]} & $@{{r.{k}.var1.EVT}}$ & $@{{r.{k}.var1.Historical}}$ & $@{{r.{k}.var1.Normal}}$ & '
            f'$@{{r.{k}.es25.EVT}}$ & $@{{r.{k}.es25.Historical}}$ & $@{{r.{k}.es25.Normal}}$')


D.frame(T('VaR 1\\% and ES 2.5\\%: Three Methods', 'VaR 1\\% și ES 2,5\\%: trei metode'), table(
    'l|rrr|rrr', T('& \\multicolumn{3}{c|}{VaR 1\\% (\\%)} & \\multicolumn{3}{c}{ES 2.5\\% (\\%)} \\\\ & EVT & historical & Normal & EVT & historical & Normal',
                   '& \\multicolumn{3}{c|}{VaR 1\\% (\\%)} & \\multicolumn{3}{c}{ES 2,5\\% (\\%)} \\\\ & EVT & istoric & Normală & EVT & istoric & Normală'),
    [vrow(k) for k in ASSETS], size='footnotesize') + items(
    T('At these levels EVT and historical simulation differ by at most $@{dmax}$ percentage points: there are enough data at 1\\% and 2.5\\%',
      'La aceste niveluri, EVT și simularea istorică diferă cu cel mult $@{dmax}$ puncte procentuale: există destule date la 1\\% și 2,5\\%'),
    T('The Normal law is too low: the EVT VaR 1\\% is $@{r.sp500.gap1}\\%$ higher for the S\\&P 500 and $@{r.bet.gap1}\\%$ higher for the BET; ES 2.5\\% by a similar margin',
      'Legea Normală este prea jos: VaR 1\\% EVT este cu $@{r.sp500.gap1}\\%$ mai mare pentru S\\&P 500 și cu $@{r.bet.gap1}\\%$ mai mare pentru BET; ES 2,5\\% cu o marjă asemănătoare')) + ql('SFM_ch5_evt_var'), 'footnotesize')


def v01row(k):
    return (f'{NAMES[k]} & $@{{r.{k}.var01.EVT}}$ & $@{{r.{k}.var01.Hill}}$ & $@{{r.{k}.var01.Historical}}$ & $@{{r.{k}.var01.Normal}}$ & $@{{r.{k}.max}}$')


D.frame(T('Far in the Tail: VaR 0.1\\%', 'Departe în coadă: VaR 0,1\\%'), table(
    'lrrrrr', T('& EVT (POT) & Hill & historical & Normal & largest loss', '& EVT (POT) & Hill & istoric & Normală & cea mai mare pierdere'),
    [v01row(k) for k in ASSETS + STOCKS], size='scriptsize') + items(
    T('VaR 0.1\\% (\\%): the loss exceeded on one day in 1\\,000; BVB stocks since 2010',
      'VaR 0,1\\% (\\%): pierderea depășită într-o zi din 1\\,000; acțiunile BVB din 2010'),
    T('EVT is about $@{r.sp500.gap01}$ times the Normal value for the S\\&P 500 and $@{r.bet.gap01}$ times for the BET',
      'EVT este de circa $@{r.sp500.gap01}$ ori valoarea Normală pentru S\\&P 500 și de $@{r.bet.gap01}$ ori pentru BET'),
    T('Historical VaR 0.1\\% rests on the @{nh.min}--@{nh.max} largest losses only; EVT and Hill smooth them with a model of the tail',
      'VaR 0,1\\% istoric se bazează doar pe cele mai mari @{nh.min}--@{nh.max} pierderi; EVT și Hill le netezesc cu un model al cozii')) + ql('SFM_ch5_evt_var'), 'footnotesize')

chart(T('Out of Sample: Estimate until 2019, Test 2020--@{y1}', 'În afara eșantionului: estimare pînă în 2019, test 2020--@{y1}'), 'sfm_ch5_oos', 'SFM_ch5_evt_var', [
    T('S\\&P 500 daily losses after 2019 with the VaR 1\\% of the three methods, all estimated on 1990--2019',
      'Pierderile zilnice S\\&P 500 după 2019 cu VaR 1\\% al celor trei metode, toate estimate pe 1990--2019'),
    T('A good VaR 1\\% is exceeded on about 1\\% of the test days, here $@{o.sp500.var1.exp}$ days', 'Un VaR 1\\% bun este depășit în circa 1\\% din zilele de test, aici $@{o.sp500.var1.exp}$ zile')], h='0.58\\textheight')


def orow(k):
    return (f'{NAMES[k]} & @{{o.{k}.n}} & $@{{o.{k}.var1.exp}}$ & @{{o.{k}.var1.EVT.x}} & @{{o.{k}.var1.Historical.x}} & @{{o.{k}.var1.Normal.x}} & '
            f'$@{{o.{k}.var01.exp}}$ & @{{o.{k}.var01.EVT.x}} & @{{o.{k}.var01.Historical.x}} & @{{o.{k}.var01.Normal.x}}')


D.frame(T('Counting the Exceedances', 'Numărarea depășirilor'), table(
    'lr|rrrr|rrrr', T('& & \\multicolumn{4}{c|}{VaR 1\\%} & \\multicolumn{4}{c}{VaR 0.1\\%} \\\\ & test days & expected & EVT & hist. & Normal & expected & EVT & hist. & Normal',
                      '& & \\multicolumn{4}{c|}{VaR 1\\%} & \\multicolumn{4}{c}{VaR 0,1\\%} \\\\ & zile de test & așteptat & EVT & ist. & Normală & așteptat & EVT & ist. & Normală'),
    [orow(k) for k in ASSETS], size='scriptsize') + items(
    T('The Normal VaR is exceeded far too often, especially at 0.1\\%: @{o.sp500.var01.Normal.x} days instead of $@{o.sp500.var01.exp}$ for the S\\&P 500',
      'VaR Normal este depășit mult prea des, mai ales la 0,1\\%: @{o.sp500.var01.Normal.x} zile în loc de $@{o.sp500.var01.exp}$ pentru S\\&P 500'),
    T('EVT and historical VaR are close to each other; S\\&P 500 above the target (the 2020 cluster), BET below it (calm years)',
      'VaR EVT și VaR istoric sînt apropiate între ele; S\\&P 500 peste țintă (grupul din 2020), BET sub ea (ani liniștiți)'),
    T('A VaR estimated once ignores the changing volatility: conditional EVT (GARCH + POT) \\refMF, Chapters 9 and 10',
      'Un VaR estimat o singură dată ignoră volatilitatea variabilă: EVT condiționată (GARCH + POT) \\refMF, Capitolele 9 și 10')) + ql('SFM_ch5_evt_var'), 'footnotesize')

D.frame(T('What EVT Can and Cannot Do', 'Ce poate și ce nu poate face EVT'), items(
    (T('\\textbf{Can}: give a principled model of the tail and extrapolate beyond the largest observed loss', '\\textbf{Poate}: să dea un model fundamentat al cozii și să extrapoleze dincolo de cea mai mare pierdere observată'),
     [T('stable VaR and ES at 0.1\\% where historical simulation has only a handful of points', 'VaR și ES stabile la 0,1\\%, acolo unde simularea istorică are doar cîteva puncte')]),
    (T('\\textbf{Cannot}: remove the uncertainty of the far tail', '\\textbf{Nu poate}: să elimine incertitudinea cozii îndepărtate'),
     [T('$\\hat\\xi$ has a standard error of $@{sexi.min}$--$@{sexi.max}$ here; return levels beyond the sample have wide intervals', '$\\hat\\xi$ are aici o eroare standard de $@{sexi.min}$--$@{sexi.max}$; return levels dincolo de eșantion au intervale largi')]),
    (T('\\textbf{Assumes}: i.i.d.\\ losses, or at least a stable tail over time', '\\textbf{Presupune}: pierderi i.i.d., sau măcar o coadă stabilă în timp'),
     [T('apply EVT to returns divided by a GARCH volatility (Chapter 9) for daily risk; see also entropy-based ES \\refPeleA',
        'aplicați EVT pe randamente împărțite la o volatilitate GARCH (Capitolul 9) pentru riscul zilnic; vedeți și ES bazat pe entropie \\refPeleA')]),
    T('Surveys of EVT in finance: \\refRocco; \\refQRM, Ch.~5', 'Sinteze despre EVT în finanțe: \\refRocco; \\refQRM, cap.~5')))

D.recap(('VaR and ES by EVT', 'VaR și ES prin EVT'), [
    T('$\\mathrm{VaR}_p = u + \\frac{\\beta}{\\xi}[(np/N_u)^{-\\xi} - 1]$; $\\mathrm{ES}_p = (\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$',
      '$\\mathrm{VaR}_p = u + \\frac{\\beta}{\\xi}[(np/N_u)^{-\\xi} - 1]$; $\\mathrm{ES}_p = (\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$'),
    T('VaR 1\\% and ES 2.5\\%: EVT $\\approx$ historical, Normal too low', 'VaR 1\\% și ES 2,5\\%: EVT $\\approx$ istoric, Normală prea jos'),
    T('VaR 0.1\\%: EVT about twice the Normal value; historical simulation rests on a few days', 'VaR 0,1\\%: EVT de circa două ori valoarea Normală; simularea istorică se bazează pe cîteva zile'),
    T('Out of sample: the Normal VaR fails; unconditional EVT ignores volatility regimes', 'În afara eșantionului: VaR Normal eșuează; EVT necondiționată ignoră regimurile de volatilitate')])

# =============================================================================
# 8. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Has the tail of BET losses changed as the Bucharest market grew?}', '\\textbf{S-a schimbat coada pierderilor BET pe măsură ce piața din București a crescut?}'),
     [T('Hill index, $k = 2.5\\%$ of $n$: @{bp.before.y0}--@{bp.before.y1}: $\\hat\\alpha = @{bp.before.alpha}$ (block SE $@{bp.before.se_block}$); @{bp.after.y0}--@{bp.after.y1}: $\\hat\\alpha = @{bp.after.alpha}$ ($@{bp.after.se_block}$)',
        'indicele Hill, $k = 2.5\\%$ din $n$: @{bp.before.y0}--@{bp.before.y1}: $\\hat\\alpha = @{bp.before.alpha}$ (SE pe blocuri $@{bp.before.se_block}$); @{bp.after.y0}--@{bp.after.y1}: $\\hat\\alpha = @{bp.after.alpha}$ ($@{bp.after.se_block}$)'),
      T('$z = @{bp.z}$: a heavier tail after 2010, but not significant at 5\\%', '$z = @{bp.z}$: o coadă mai groasă după 2010, dar nesemnificativă la 5\\%')]),
    T('Why it is open: the threshold fell from $@{bp.before.u}\\%$ to $@{bp.after.u}\\%$ with the volatility; is the change in the tail or in the volatility?',
      'De ce este deschisă: pragul a scăzut de la $@{bp.before.u}\\%$ la $@{bp.after.u}\\%$ odată cu volatilitatea; schimbarea este în coadă sau în volatilitate?'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')) + ql('SFM_ch5_hill'))

D.frame(T('How AI Could Help', 'Cum ar putea ajuta AI'), items(
    T('\\textbf{Literature}: list studies of tail indices in emerging markets and summarise their methods',
      '\\textbf{Literatura}: lista studiilor despre indicii de coadă pe piețele emergente și rezumatul metodelor lor'),
    T('\\textbf{Code}: a first draft of a Hill and POT analysis on rolling windows, with block-bootstrap intervals',
      '\\textbf{Cod}: o primă versiune a unei analize Hill și POT pe ferestre mobile, cu intervale bootstrap pe blocuri'),
    T('\\textbf{Robustness}: other thresholds, GARCH-standardised returns, other BVB indices', '\\textbf{Robustețe}: alte praguri, randamente standardizate cu GARCH, alți indici BVB'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that computes the Hill tail index of daily BET losses with k = 2.5\\% of n on rolling 5-year windows, with moving-block bootstrap standard errors (blocks of 20 days).}',
        '\\aiprompt{Write a Python function that computes the Hill tail index of daily BET losses with k = 2.5\\% of n on rolling 5-year windows, with moving-block bootstrap standard errors (blocks of 20 days).}')])))

D.frame(T('What to Check', 'Ce trebuie verificat'), items(
    T('Sign: losses are $-r$; a Hill estimate on returns instead of losses measures the gains', 'Semnul: pierderile sînt $-r$; o estimare Hill pe randamente, nu pe pierderi, măsoară cîștigurile'),
    T('Parameterisation: scipy\'s \\texttt{genextreme} uses $c = -\\xi$; \\texttt{genpareto} uses $c = \\xi$', 'Parametrizarea: \\texttt{genextreme} din scipy folosește $c = -\\xi$; \\texttt{genpareto} folosește $c = \\xi$'),
    T('The rule for $k$ or $u$ is fixed before seeing the result, not tuned to get a significant change', 'Regula pentru $k$ sau $u$ se fixează înainte de a vedea rezultatul, nu se ajustează pentru a obține o schimbare semnificativă'),
    T('Rolling windows overlap: their estimates are not independent tests', 'Ferestrele mobile se suprapun: estimările lor nu sînt teste independente'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Seed', 'Sămînță de proiect'), items(
    (T('\\textbf{Question}: is the tail of BET losses lighter or heavier today than before 2010, once volatility is removed?',
       '\\textbf{Întrebarea}: este coada pierderilor BET mai subțire sau mai groasă azi decît înainte de 2010, după ce eliminăm volatilitatea?'),
     [T('data: BET daily closes since 1997 (EODHD); optionally BET-TR since 2014', 'date: închiderile zilnice BET din 1997 (EODHD); opțional BET-TR din 2014')]),
    (T('Steps', 'Pași'),
     [T('Hill and POT ($u$ at the 90\\% quantile) for the two periods, with block-bootstrap intervals', 'Hill și POT ($u$ la cuantila de 90\\%) pentru cele două perioade, cu intervale bootstrap pe blocuri'),
      T('repeat on returns divided by a rolling 60-day volatility: does the difference survive?', 'repetați pe randamente împărțite la o volatilitate mobilă pe 60 de zile: rezistă diferența?'),
      T('compare with the S\\&P 500 over the same periods', 'comparați cu S\\&P 500 pe aceleași perioade')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show',
      'Rezultat: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice folosire a AI și listați erorile AI pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei principale'), items(
    T('Daily losses have power-law tails with index $\\alpha \\approx 3$: finite variance, infinite kurtosis', 'Pierderile zilnice au cozi de tip putere cu indicele $\\alpha \\approx 3$: varianță finită, kurtosis infinit'),
    T('Hill: the tail index from the $k$ largest losses; choose $k$ on the Hill plot; block-bootstrap SE', 'Hill: indicele de coadă din cele mai mari $k$ pierderi; $k$ ales pe graficul Hill; SE prin bootstrap pe blocuri'),
    T('Block maxima $\\to$ GEV; excesses over a threshold $\\to$ GPD; the same shape $\\xi = 1/\\alpha$', 'Block maxima $\\to$ GEV; excesele peste un prag $\\to$ GPD; aceeași formă $\\xi = 1/\\alpha$'),
    T('Check every fit with mean excess, parameter stability, PP and QQ plots', 'Verificați fiecare ajustare cu mean excess, stabilitatea parametrilor, grafice PP și QQ'),
    T('EVT VaR and ES: close to historical at 1\\% and 2.5\\%, far above the Normal values at 0.1\\%', 'VaR și ES EVT: apropiate de cele istorice la 1\\% și 2,5\\%, mult peste valorile Normale la 0,1\\%'),
    T('Extrapolation far beyond the data remains uncertain: report intervals', 'Extrapolarea mult dincolo de date rămîne incertă: raportați intervale')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Regular variation', 'Variație regulată') + ' & $P(L > x) = x^{-\\alpha}\\ell(x)$, \\quad $P(L > tx)/P(L > x) \\to t^{-\\alpha}$',
     'Hill & $\\hat\\alpha_k = [\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})]^{-1}$, \\quad $\\mathrm{SE} \\approx \\hat\\alpha/\\sqrt{k}$',
     T('Hill quantile', 'Cuantila Hill') + ' & $\\hat x_p = L_{(k+1)}(k/(np))^{1/\\hat\\alpha}$',
     'Mean excess & $e(u) = E[L - u \\mid L > u]$; \\quad GPD: $(\\beta + \\xi(v - u))/(1 - \\xi)$',
     'GEV & $H(x) = \\exp\\{-[1 + \\xi(x - \\mu)/\\sigma]^{-1/\\xi}\\}$',
     'Return level & $z_T = \\mu + \\frac{\\sigma}{\\xi}[(-\\ln(1 - \\frac{1}{12T}))^{-\\xi} - 1]$',
     'GPD & $G(y) = 1 - (1 + \\xi y/\\beta)^{-1/\\xi}$',
     'VaR (POT) & $u + \\frac{\\beta}{\\xi}[(np/N_u)^{-\\xi} - 1]$',
     'ES (POT) & $(\\mathrm{VaR}_p + \\beta - \\xi u)/(1 - \\xi)$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Verificați-vă'), items(
    (T('\\textbf{Question}: a GPD fit gives $\\hat\\xi = 0.25$; which moments of the losses exist?', '\\textbf{Întrebare}: o ajustare GPD dă $\\hat\\xi = 0{,}25$; ce momente ale pierderilor există?'),
     [T('\\textbf{Answer}: $\\alpha = 1/\\xi = 4$: mean, variance and skewness exist; the kurtosis does not ($m < 4$ only)',
        '\\textbf{Răspuns}: $\\alpha = 1/\\xi = 4$: există media, varianța și asimetria; kurtosis-ul nu (doar $m < 4$)')]),
    (T('\\textbf{Question}: why use the 2.5\\% largest losses and not all losses for the Hill estimator?', '\\textbf{Întrebare}: de ce folosim cele mai mari 2,5\\% dintre pierderi și nu toate pierderile pentru estimatorul Hill?'),
     [T('\\textbf{Answer}: the power law holds only in the tail; with the body included $\\hat\\alpha$ is biased', '\\textbf{Răspuns}: legea putere este valabilă doar în coadă; cu tot corpul inclus, $\\hat\\alpha$ este deplasat')]),
    (T('\\textbf{Question}: is VaR 0.1\\% by historical simulation reliable with 10 years of data?', '\\textbf{Întrebare}: este VaR 0,1\\% prin simulare istorică fiabil cu 10 ani de date?'),
     [T('\\textbf{Answer}: no: about 2\\,500 days leave only 2--3 losses beyond it; EVT uses the whole tail', '\\textbf{Răspuns}: nu: circa 2\\,500 de zile lasă doar 2--3 pierderi dincolo de el; EVT folosește întreaga coadă')]),
    T('Next: Chapter 6, model selection and risk management', 'Urmează: Capitolul 6, selecția modelului și managementul riscului')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
