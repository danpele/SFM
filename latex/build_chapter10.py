r"""
build_chapter10.py -- Capitolul 10 (VaR, ES și backtesting), EN + RO dintr-o singură sursă
========================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_10/ch10_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Convenția nivelului: VaR 1% (VaR_alpha = -q_alpha), ES 2,5%; niciodată „VaR 99%”.
Ieșire:
  EN/Courses/chapter10_var_es_backtesting.tex
  RO/Cursuri/capitol10_var_es_backtesting.tex
Rulare:
  python3 Quantlets/Ch_10/generate_all_charts.py
  python3 latex/build_chapter10.py && python3 latex/sfm_build.py compile 10
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch10_common import ASSETS, BIB, BT, METHODS, MKEY, NAMES, QLURL, REFS, SHORT, T, load, pv, values   # noqa: E402

N = load()
V = values(N)
D = Deck(10, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_10}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'jpm': ('ch8_jpmorgan_23wall.jpg', C + 'J._P._Morgan_\\%26_Company_Building_23_Wall_Street.jpg',
            '⟦Photo||Foto⟧: Beyond My Ken (2012); CC BY-SA 4.0; Wikimedia Commons'),
    'bis': ('ch10_bis_tower_2014.jpg', C + 'Basel_Committee_on_Banking_Supervision_-_BCBS.jpg',
            '⟦Photo||Foto⟧: Taxiarchos228 (2014); CC BY-SA 4.0; Wikimedia Commons'),
    'delbaen': ('ch10_delbaen_1997.jpg', C + 'ETH-BIB-Delbaen,_Freddy_(1946-)-Portr_16468.tif',
                '⟦Photo||Foto⟧: ETH-Bibliothek Zürich, Bildarchiv (1997); CC BY-SA 4.0; Wikimedia Commons'),
    'lehman': ('ch0_lehman_2008.jpg', C + 'Lehman_Brothers-NYC-20080915.jpg',
               '⟦Photo||Foto⟧: Robert Scoble (2008); CC BY 2.0; Wikimedia Commons'),
    'fidi': ('ch0_fidi_march_2020.jpg', C + 'Subdued_FiDi_(50063555551).jpg',
             '⟦Photo||Foto⟧: Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
z1, z25 = stats.norm.ppf(0.01), stats.norm.ppf(0.025)
V.put('z1', -z1, 3)
V.put('z25', -z25, 3)
V.put('phi25', stats.norm.pdf(z25), 4)
V.put('esn.f', stats.norm.pdf(z25) / 0.025, 3)
V.put('esn.ratio', stats.norm.pdf(z25) / 0.025 / -z1, 3)
# a toy sample of 100 days: the five worst returns
TOY = [-6.1, -4.8, -4.2, -3.9, -3.5]
V.put('toy.es', -sum(TOY[:3]) / 3, 2)
# historical simulation on the S&P 500: k = ceil(n alpha)
M = N['methods']
n_sp = M['sp500']['n']
V.raw('hs.k1', str(math.ceil(n_sp * 0.01 - 1e-9)))
V.raw('hs.k25', str(math.ceil(n_sp * 0.025 - 1e-9)))
V.put('hs.n1', n_sp * 0.01, 2)
V.put('hs.n25', n_sp * 0.025, 2)
# Normal worked example on the S&P 500
sp = M['sp500']
V.put('sp.zsd', -z1 * sp['sd'], 3)
V.put('sp.fsd', stats.norm.pdf(z25) / 0.025 * sp['sd'], 3)
V.put('sp.gapN', 100 * (1 - sp['Normal']['var'] / sp['HS']['var']), 0)
# Student-t worked example
tt = sp['t']
V.put('t.m', tt['m'], 3)
V.put('t.s', tt['s'], 3)
V.put('t.q', stats.t.ppf(0.01, tt['nu']), 3)
V.put('t.fac', stats.t.pdf(stats.t.ppf(0.025, tt['nu']), tt['nu']) * (tt['nu'] + stats.t.ppf(0.025, tt['nu']) ** 2) / ((tt['nu'] - 1) * 0.025), 3)
# position of 1 million RON in the S&P 500
V.raw('pos.var', f"{1e6 * sp['HS']['var'] / 100:,.0f}".replace(',', '\\,'))
V.raw('pos.es', f"{1e6 * sp['HS']['es'] / 100:,.0f}".replace(',', '\\,'))
V.put('hz.ratio', N['horizon']['var_h'] / N['horizon']['var1'], 2)
V.put('sqrt10', math.sqrt(10), 3)
# two bonds (Artzner et al., 1999, style counterexample): default probability 4%, loss 100, level 5%
p = 0.04
V.put('bd.pany', 100 * (1 - (1 - p) ** 2), 2)
V.put('bd.pboth', 100 * p ** 2, 2)
V.put('bd.pone', 100 * 2 * p * (1 - p), 2)
V.put('bd.es1', (p * 100) / 0.05, 0)
V.put('bd.es2', (p ** 2 * 200 + (0.05 - p ** 2) * 100) / 0.05, 1)
# Cornish-Fisher for the S&P 500
V.put('cf.z1', z1, 3)
# EVT worked example
f = sp['pot']
V.put('evt.r1', f['n'] * 0.01 / f['Nu'], 4)
V.put('evt.pow1', (f['n'] * 0.01 / f['Nu']) ** (-f['xi']), 3)
V.put('evt.boxi', f['beta'] / f['xi'], 3)
V.put('evt.share', 100 * f['Nu'] / f['n'], 1)
# backtesting: worked Kupiec examples
def kup(x, n, a=0.01):
    ph_ = x / n
    lr = -2 * ((n - x) * math.log(1 - a) + x * math.log(a) - (n - x) * math.log(1 - ph_) - x * math.log(ph_))
    return lr, stats.chi2.sf(lr, 1)


lr7, p7 = kup(7, 250)
V.put('k7.lr', lr7, 2)
V.put('k7.p', p7, 3)
V.put('k7.l0', 243 * math.log(0.99) + 7 * math.log(0.01), 2)
V.put('k7.l1', 243 * math.log(1 - 7 / 250) + 7 * math.log(7 / 250), 2)
V.put('chi1', stats.chi2.ppf(0.95, 1), 2)
V.put('chi2', stats.chi2.ppf(0.95, 2), 2)
# the smallest and largest counts not rejected by Kupiec at 5% in 250 days
acc = [x for x in range(0, 30) if (kup(x, 250)[1] if x > 0 else stats.chi2.sf(-2 * 250 * math.log(0.99), 1)) > 0.05]
V.raw('k.acc.hi', str(max(acc)))
V.raw('k.acc.lo', str(min(acc)))
B = N['bt']
b = B['sp500']['HS']
V.raw('ch.n00', str(b['n00']))
V.raw('ch.n01', str(b['n01']))
V.raw('ch.n10', str(b['n10']))
V.raw('ch.n11', str(b['n11']))
V.put('ch.ratio', b['p11'] / b['p01'], 0)
g = B['sp500']['GARCH-t']
nn, xx = g['n'], g['x']
V.put('kg.pi', xx / nn, 4)
V.put('kg.l0', (nn - xx) * math.log(0.99) + xx * math.log(0.01), 1)
V.put('kg.l1', (nn - xx) * math.log(1 - xx / nn) + xx * math.log(xx / nn), 1)
V.raw('kg.nx', str(nn - xx))
V.put('kg.ci.lo', 100 * stats.binom.ppf(0.025, nn, 0.01) / nn, 2)
V.put('kg.ci.hi', 100 * stats.binom.ppf(0.975, nn, 0.01) / nn, 2)
# portfolio worked example
P = N['port']
V.put('p.w1sd', 0.5 * P['sd'][0], 3)
V.put('p.w2sd', 0.5 * P['sd'][1], 3)
V.put('p.cov', P['corr'] * P['sd'][0] * P['sd'][1], 3)
# ranges quoted in the text (computed, never typed)
gapN = [100 * (1 - M[k]['Normal']['var'] / M[k]['HS']['var']) for k in ASSETS]
V.put('rg.gapN.lo', min(gapN), 0)
V.put('rg.gapN.hi', max(gapN), 0)
dif = [max(M[k][m]['var'] for m in ('HS', 'Student-t', 'EVT')) - min(M[k][m]['var'] for m in ('HS', 'Student-t', 'EVT')) for k in ASSETS]
V.put('rg.dif.max', max(dif), 1)
cfr = [M[k]['Cornish-Fisher']['var'] / M[k]['HS']['var'] for k in ASSETS]
V.put('rg.cf.lo', min(cfr), 1)
V.put('rg.cf.hi', max(cfr), 1)
kk = [M[k]['exkurt'] for k in ASSETS]
V.put('rg.k.lo', min(kk), 0)
V.put('rg.k.hi', max(kk), 0)
er = [100 * (M[k]['HS']['es'] / M[k]['HS']['var'] - 1) for k in ASSETS]
V.put('rg.es.lo', min(er), 1)
V.put('rg.es.hi', max(er), 1)
gr = [100 * B[k][m]['rate'] for k in ('sp500', 'dax') for m in ('GARCH-t', 'FHS')]
V.put('rg.g.lo', min(gr), 1)
V.put('rg.g.hi', max(gr), 1)
V.put('k.p2', 100 * stats.binom.cdf(max(acc), 250, 0.02) - 100 * stats.binom.cdf(min(acc) - 1, 250, 0.02), 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how much can a portfolio lose tomorrow, and how do we know that the answer was right?',
       '\\textbf{Întrebarea}: cît poate pierde un portofoliu mîine și de unde știm că răspunsul a fost corect?'),
     [T('Chapter 5 described the tails, Chapter 9 the volatility of tomorrow; this chapter turns both into risk numbers and tests them',
        'Capitolul 5 a descris cozile, Capitolul 9 volatilitatea de mîine; acest capitol le transformă în măsuri de risc și le testează')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('VaR (value at risk) and ES (expected shortfall): definitions, coherence, elicitability', 'VaR (value at risk, valoarea expusă la risc) și ES: definiții, coerență, elicitabilitate'),
      T('estimation: historical simulation, Normal, Student-t, Cornish--Fisher, EVT, GARCH and filtered historical simulation', 'estimarea: simularea istorică, distribuția Normală, Student-t, Cornish--Fisher, EVT, GARCH și simularea istorică filtrată'),
      T('portfolios: variance--covariance, tail dependence, a first look at copulas', 'portofolii: metoda varianță--covarianță, dependența în cozi, o primă privire asupra copulelor'),
      T('backtesting: Kupiec, Christoffersen, the Basel traffic light, ES backtesting', 'backtesting: testele Kupiec și Christoffersen, semaforul Basel, backtesting pentru ES')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Define VaR and ES at a tail probability $\\alpha$ (VaR 1\\%, ES 2.5\\%) and compute them for the Normal and Student-t distributions',
      'Definiți VaR și ES la o probabilitate a cozii $\\alpha$ (VaR 1\\%, ES 2,5\\%) și calculați-le pentru distribuția Normală și pentru Student-t'),
    T('State the axioms of a coherent risk measure and show with a counterexample that VaR is not subadditive',
      'Enunțați axiomele unei măsuri de risc coerente și arătați printr-un contraexemplu că VaR nu este subaditiv'),
    T('Estimate VaR and ES by historical simulation, parametric methods, Cornish--Fisher, EVT, GARCH and filtered historical simulation',
      'Estimați VaR și ES prin simulare istorică, metode parametrice, Cornish--Fisher, EVT, GARCH și simulare istorică filtrată'),
    T('Compute a portfolio VaR and explain why correlation does not measure the risk of joint crashes', 'Calculați VaR-ul unui portofoliu și explicați de ce corelația nu măsoară riscul prăbușirilor simultane'),
    T('Backtest a VaR model with the Kupiec and Christoffersen tests and the Basel traffic light', 'Testați un model VaR (backtesting) cu testele Kupiec și Christoffersen și cu semaforul Basel')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~16 (VaR and backtesting), Ch.~17 (copulae and VaR), Sections 18.1 and 18.4 (extreme risks)',
       'Manual: \\refFHH, cap.~16 (VaR și backtesting), cap.~17 (copule și VaR), secțiunile 18.1 și 18.4 (riscuri extreme)'),
     [T('exercises with solutions: \\refBHL, Ch.~16', 'exerciții rezolvate: \\refBHL, cap.~16'),
      T('risk measures and copulas: \\refQRM, Ch.~2 and 7', 'măsuri de risc și copule: \\refQRM, cap.~2 și 7')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_10}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_10}'),
     [T('ported from the SFE Quantlets VaRest, SFEVaRbank, SFEVaRtimeplot, SFEVaRqqplot and SFEvar\\_pot\\_backtesting',
        'portate din Quantlet-urile SFE VaRest, SFEVaRbank, SFEVaRtimeplot, SFEVaRqqplot și SFEvar\\_pot\\_backtesting')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter10_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter10_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}',
      'Curs video: \\quantinar{Measuring Statistical Risk}{https://quantinar.com/course/100080/measuring-statistical-risk}')))

# =============================================================================
# 1. MOTIVAȚIE
# =============================================================================
D.section('Measuring Tail Risk', 'Măsurarea riscului din coadă')

D.frame(T('The 4:15 Report and RiskMetrics', 'Raportul de la ora 16:15 și RiskMetrics'), cols(items(
    (T('Around 1990: J.P. Morgan\'s chairman, Dennis Weatherstone, asks for one number, every day at 4:15 pm, summarising how much the bank could lose the next day',
       'În jurul anului 1990: președintele J.P. Morgan, Dennis Weatherstone, cere un singur număr, în fiecare zi la ora 16:15, care să rezume cît poate pierde banca a doua zi'),
     [T('the answer became the \\textbf{value at risk} (VaR): a loss that is exceeded only with a small probability', 'răspunsul a devenit \\textbf{valoarea expusă la risc} (VaR): o pierdere depășită doar cu o probabilitate mică')]),
    (T('October 1994: the methods and data are published free of charge as RiskMetrics \\refRM', 'Octombrie 1994: metodele și datele sînt publicate gratuit sub numele RiskMetrics \\refRM'),
     [T('Normal returns, EWMA volatility (Chapter 8), VaR at a 5\\% tail probability over one day', 'randamente Normale, volatilitate EWMA (Capitolul 8), VaR la o probabilitate a cozii de 5\\% pe o zi'),
      T('VaR became the common language of banks, regulators and fund managers', 'VaR a devenit limbajul comun al băncilor, al autorităților de reglementare și al administratorilor de fonduri')])),
    ph('jpm', T('The J.P. Morgan building at 23 Wall Street, New York', 'Clădirea J.P. Morgan de la 23 Wall Street, New York'), h='0.40\\textheight'),
    wl='0.56', wr='0.40'), 'footnotesize')

D.frame(T('From RiskMetrics to Basel', 'De la RiskMetrics la Basel'), cols(items(
    (T('The \\textbf{BCBS} (Basel Committee on Banking Supervision) sits at the \\textbf{BIS} (Bank for International Settlements) in Basel \\refBCBSh',
       '\\textbf{BCBS} (Basel Committee on Banking Supervision, Comitetul de la Basel pentru supraveghere bancară) are sediul la \\textbf{BIS}, în Basel \\refBCBSh'),
     [T('1996: banks may compute market-risk capital with their own VaR models \\refBCBSa', '1996: băncile pot calcula capitalul pentru riscul de piață cu propriile modele VaR \\refBCBSa'),
      T('rule: VaR 1\\% over 10 days, times a multiplier of at least 3; the multiplier grows when the model fails the backtest \\refBCBSb',
        'regula: VaR 1\\% pe 10 zile, înmulțit cu un factor de cel puțin 3; factorul crește dacă modelul nu trece testul (backtesting) \\refBCBSb')]),
    (T('2019: the \\textbf{FRTB} (Fundamental Review of the Trading Book) \\refFRTB', '2019: \\textbf{FRTB} (Fundamental Review of the Trading Book, revizuirea fundamentală a portofoliului de tranzacționare) \\refFRTB'),
     [T('capital from ES 2.5\\% instead of VaR 1\\% \\refMAR; backtesting still counts VaR exceptions \\refMARb',
        'capitalul se calculează din ES 2,5\\% în loc de VaR 1\\% \\refMAR; backtesting-ul numără în continuare depășirile VaR \\refMARb')])),
    ph('bis', T('The BIS tower in Basel, seat of the Basel Committee', 'Turnul BIS din Basel, sediul Comitetului de la Basel'), h='0.42\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.frame(T('Two Storms: 2008 and 2020', 'Două crize: 2008 și 2020'), cols(
    ph('lehman', T('Lehman Brothers headquarters, New York, 15 September 2008', 'Sediul Lehman Brothers, New York, 15 septembrie 2008'), h='0.30\\textheight'),
    ph('fidi', T('The Financial District of New York, 25 March 2020', 'Districtul financiar din New York, 25 martie 2020'), h='0.30\\textheight'),
    wl='0.48', wr='0.48') + items(
    T('The worst day of the S\\&P 500 since 2000: $@{def.min}\\%$ on @{def.dmin}; a Normal model with the sample volatility gives such a day a probability of practically zero',
      'Cea mai proastă zi a S\\&P 500 din 2000: $@{def.min}\\%$ pe @{def.dmin}; un model Normal cu volatilitatea de selecție îi dă unei asemenea zile o probabilitate practic nulă'),
    T('A risk measure must be judged in such periods: we use 2008 and 2020 as stress tests throughout the chapter',
      'O măsură de risc trebuie judecată în astfel de perioade: folosim 2008 și 2020 ca teste de stres în tot capitolul')), 'footnotesize')

# =============================================================================
# 2. DEFINIȚII
# =============================================================================
D.section('VaR and ES: Definitions', 'VaR și ES: definiții')

D.frame(T('Losses, Horizon and Level', 'Pierderi, orizont și nivel'), items(
    (T('$X$: the \\textbf{P\\&L} (profit and loss) or the return of a position over a horizon $h$ (here one day, in \\%)',
       '$X$: \\textbf{P\\&L} (profit and loss, profitul sau pierderea) ori randamentul unei poziții pe un orizont $h$ (aici o zi, în \\%)'),
     [T('the loss is $L = -X$: a positive number when the position loses money', 'pierderea este $L = -X$: un număr pozitiv cînd poziția pierde bani')]),
    (T('The \\textbf{level} $\\alpha$ is the probability of the tail: $\\alpha = 1\\%$ or $\\alpha = 2.5\\%$', '\\textbf{Nivelul} $\\alpha$ este probabilitatea cozii: $\\alpha = 1\\%$ sau $\\alpha = 2{,}5\\%$'),
     [T('we write \\textbf{VaR 1\\%} and \\textbf{ES 2.5\\%}: the number in the name is always the tail probability', 'scriem \\textbf{VaR 1\\%} și \\textbf{ES 2,5\\%}: numărul din denumire este întotdeauna probabilitatea cozii')]),
    (T('$q_\\alpha(X) = \\inf\\{x : P(X \\le x) > \\alpha\\}$: the $\\alpha$-\\textbf{quantile} of $X$ (Chapter 2)', '$q_\\alpha(X) = \\inf\\{x : P(X \\le x) > \\alpha\\}$: \\textbf{cuantila} de ordin $\\alpha$ a lui $X$ (Capitolul 2)'),
     [T('for a continuous distribution function $F$: $q_\\alpha = F^{-1}(\\alpha)$, a negative number for small $\\alpha$', 'pentru o funcție de repartiție continuă $F$: $q_\\alpha = F^{-1}(\\alpha)$, un număr negativ pentru $\\alpha$ mic')])))

D.frame(T('Value at Risk', 'Valoarea expusă la risc (VaR)'), items(
    (T('\\textbf{Definition}: $\\mathrm{VaR}_\\alpha(X) = -q_\\alpha(X)$', '\\textbf{Definiție}: $\\mathrm{VaR}_\\alpha(X) = -q_\\alpha(X)$'),
     [T('the loss that is exceeded with probability $\\alpha$: $P(L > \\mathrm{VaR}_\\alpha) \\le \\alpha$', 'pierderea depășită cu probabilitatea $\\alpha$: $P(L > \\mathrm{VaR}_\\alpha) \\le \\alpha$'),
      T('the minus sign makes VaR a positive loss; a VaR of 3\\% means ``a loss of more than 3\\% happens on 1 day in 100\'\'',
        'semnul minus face din VaR o pierdere pozitivă; un VaR de 3\\% înseamnă „o pierdere de peste 3\\% apare într-o zi din 100”')]),
    (T('What VaR does \\textbf{not} say', 'Ce \\textbf{nu} spune VaR'),
     [T('how large the loss is on the bad days: a loss of 4\\% and a loss of 40\\% beyond VaR give the same VaR',
        'cît de mare este pierderea în zilele proaste: o pierdere de 4\\% și una de 40\\% dincolo de VaR dau același VaR'),
      T('VaR is not the maximum loss: by construction it is exceeded on about $\\alpha$ of the days', 'VaR nu este pierderea maximă: prin construcție este depășit în aproximativ $\\alpha$ dintre zile')]),
    T('In money: a position of value $W$ has $\\mathrm{VaR} = W \\times \\mathrm{VaR}(\\%)/100$', 'În bani: o poziție cu valoarea $W$ are $\\mathrm{VaR} = W \\times \\mathrm{VaR}(\\%)/100$')))

D.frame(T('Expected Shortfall', 'Expected shortfall (ES)'), items(
    (T('\\textbf{Definition}: $\\mathrm{ES}_\\alpha(X) = -\\dfrac{1}{\\alpha}\\displaystyle\\int_0^\\alpha q_u(X)\\,du$: the average of the VaRs at all levels below $\\alpha$',
       '\\textbf{Definiție}: $\\mathrm{ES}_\\alpha(X) = -\\dfrac{1}{\\alpha}\\displaystyle\\int_0^\\alpha q_u(X)\\,du$: media valorilor VaR la toate nivelurile sub $\\alpha$'),
     [T('for a continuous distribution: $\\mathrm{ES}_\\alpha = -E[X \\mid X \\le q_\\alpha(X)]$, the average loss on the worst $\\alpha$ of the days',
        'pentru o distribuție continuă: $\\mathrm{ES}_\\alpha = -E[X \\mid X \\le q_\\alpha(X)]$, pierderea medie din cele mai proaste $\\alpha$ dintre zile'),
      T('$u$: the integration variable, a tail level running from 0 to $\\alpha$', '$u$: variabila de integrare, un nivel al cozii care parcurge valorile de la 0 la $\\alpha$'),
      T('other names: CVaR (conditional VaR), TVaR (tail VaR), expected tail loss', 'alte denumiri: CVaR (conditional VaR), TVaR (tail VaR), expected tail loss')]),
    (T('Properties', 'Proprietăți'),
     [T('$\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ at the same level $\\alpha$', '$\\mathrm{ES}_\\alpha \\ge \\mathrm{VaR}_\\alpha$ la același nivel $\\alpha$'),
      T('ES looks inside the tail: two distributions with the same VaR but different tails have different ES', 'ES privește în interiorul cozii: două distribuții cu același VaR, dar cu cozi diferite, au ES diferit')])))

D.frame(T('Worked Example: VaR and ES from 100 Days', 'Exemplu rezolvat: VaR și ES din 100 de zile'), items(
    (T('A position has 100 daily returns; the five worst are $-6.1$, $-4.8$, $-4.2$, $-3.9$, $-3.5$ (in \\%)', 'O poziție are 100 de randamente zilnice; cele mai proaste cinci sînt $-6.1$, $-4.8$, $-4.2$, $-3.9$, $-3.5$ (în \\%)'),
     [T('empirical quantile at level $\\alpha$: the $k$-th smallest return, $k = \\lceil n\\alpha \\rceil$ ($\\lceil\\cdot\\rceil$: rounding up)',
        'cuantila empirică de nivel $\\alpha$: al $k$-lea cel mai mic randament, $k = \\lceil n\\alpha \\rceil$ ($\\lceil\\cdot\\rceil$: rotunjire în sus)')]),
    (T('\\textbf{VaR 1\\%}: $k = \\lceil 100 \\times 0.01 \\rceil = 1$, so $\\mathrm{VaR}_{1\\%} = 6.1\\%$', '\\textbf{VaR 1\\%}: $k = \\lceil 100 \\times 0.01 \\rceil = 1$, deci $\\mathrm{VaR}_{1\\%} = 6{,}1\\%$'),
     [T('one observation decides VaR 1\\%: with 100 days the estimate is very imprecise', 'o singură observație decide VaR 1\\%: cu 100 de zile, estimarea este foarte imprecisă')]),
    (T('\\textbf{ES 2.5\\%}: $k = \\lceil 2.5 \\rceil = 3$; the average of the three worst losses', '\\textbf{ES 2,5\\%}: $k = \\lceil 2{,}5 \\rceil = 3$; media celor mai mari trei pierderi'),
     [T('$\\mathrm{ES}_{2.5\\%} = (6.1 + 4.8 + 4.2)/3 = @{toy.es}\\%$; $\\mathrm{VaR}_{2.5\\%} = 4.2\\%$', '$\\mathrm{ES}_{2.5\\%} = (6.1 + 4.8 + 4.2)/3 = @{toy.es}\\%$; $\\mathrm{VaR}_{2.5\\%} = 4{,}2\\%$')]),
    T('A position of 1 million RON: VaR 1\\% = 61\\,000 RON; on one day in a hundred the loss is larger', 'O poziție de 1 milion de lei: VaR 1\\% = 61\\,000 de lei; într-o zi din o sută, pierderea este mai mare')))

chart(T('VaR 1\\% and ES 2.5\\% of the S\\&P 500', 'VaR 1\\% și ES 2,5\\% pentru S\\&P 500'), 'sfm_ch10_var_es_def', 'SFM_ch10_var_methods', [
    T('Daily log returns since 2000 (@{def.n} days); historical estimates: VaR 1\\% $= @{def.var1}\\%$, VaR 2.5\\% $= @{def.q25}\\%$, ES 2.5\\% $= @{def.es25}\\%$',
      'Randamente logaritmice zilnice din 2000 (@{def.n} zile); estimări istorice: VaR 1\\% $= @{def.var1}\\%$, VaR 2,5\\% $= @{def.q25}\\%$, ES 2,5\\% $= @{def.es25}\\%$'),
    T('Interpretation: ES 2.5\\% averages the whole shaded tail, including the extreme days of 2008 and 2020; VaR 1\\% is one point in that tail',
      'Interpretare: ES 2,5\\% face media întregii cozi hașurate, inclusiv a zilelor extreme din 2008 și 2020; VaR 1\\% este un singur punct din această coadă'),
    T('For a position of 1 million RON in the index: VaR 1\\% about @{pos.var} RON, ES 2.5\\% about @{pos.es} RON', 'Pentru o poziție de 1 milion de lei în indice: VaR 1\\% circa @{pos.var} de lei, ES 2,5\\% circa @{pos.es} lei')],
    h='0.50\\textheight')

D.frame(T('The Normal Distribution: Closed Forms', 'Distribuția Normală: formule explicite'), items(
    (T('$X \\sim N(\\mu, \\sigma^2)$, $z_\\alpha = \\Phi^{-1}(\\alpha)$ ($\\Phi$, $\\varphi$: distribution function and density of $N(0,1)$)', '$X \\sim N(\\mu, \\sigma^2)$, $z_\\alpha = \\Phi^{-1}(\\alpha)$ ($\\Phi$, $\\varphi$: funcția de repartiție și densitatea lui $N(0,1)$)'),
     [T('$\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, \\quad $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\,\\dfrac{\\varphi(z_\\alpha)}{\\alpha}$', '$\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_\\alpha)$, \\quad $\\mathrm{ES}_\\alpha = -\\mu + \\sigma\\,\\dfrac{\\varphi(z_\\alpha)}{\\alpha}$'),
      T('derivation: $E[Z \\mid Z \\le z] = -\\varphi(z)/\\Phi(z)$ because $\\varphi\'(z) = -z\\varphi(z)$', 'derivare: $E[Z \\mid Z \\le z] = -\\varphi(z)/\\Phi(z)$, deoarece $\\varphi\'(z) = -z\\varphi(z)$')]),
    (T('Numbers: $z_{0.01} = -@{z1}$; $z_{0.025} = -@{z25}$, $\\varphi(z_{0.025}) = @{phi25}$, $\\varphi(z_{0.025})/0.025 = @{esn.f}$', 'Valori: $z_{0.01} = -@{z1}$; $z_{0.025} = -@{z25}$, $\\varphi(z_{0.025}) = @{phi25}$, $\\varphi(z_{0.025})/0.025 = @{esn.f}$'),
     [T('ES 2.5\\% / VaR 1\\% $= @{esn.f}/@{z1} = @{esn.ratio}$ (with $\\mu = 0$): for Normal returns the two measures are almost equal',
        'ES 2,5\\% / VaR 1\\% $= @{esn.f}/@{z1} = @{esn.ratio}$ (cu $\\mu = 0$): pentru randamente Normale, cele două măsuri sînt aproape egale'),
      T('this is why ES 2.5\\% replaced VaR 1\\% in Basel: same size for Normal data, larger for heavy tails', 'de aceea ES 2,5\\% a înlocuit VaR 1\\% în Basel: aceeași mărime pentru date Normale, mai mare pentru cozi groase')])), 'footnotesize')

D.frame(T('Worked Example: Normal VaR and ES of the S\\&P 500', 'Exemplu rezolvat: VaR și ES Normale pentru S\\&P 500'), items(
    (T('Sample moments of the daily returns since 2000: $\\hat\\mu = @{m.sp500.mu}$, $\\hat\\sigma = @{m.sp500.sd}$ (in \\%)', 'Momentele de selecție ale randamentelor zilnice din 2000: $\\hat\\mu = @{m.sp500.mu}$, $\\hat\\sigma = @{m.sp500.sd}$ (în \\%)'),
     [T('VaR 1\\% $= -(@{m.sp500.mu} - @{z1} \\times @{m.sp500.sd}) = -@{m.sp500.mu} + @{sp.zsd} = @{m.sp500.n.v}\\%$', 'VaR 1\\% $= -(@{m.sp500.mu} - @{z1} \\times @{m.sp500.sd}) = -@{m.sp500.mu} + @{sp.zsd} = @{m.sp500.n.v}\\%$'),
      T('ES 2.5\\% $= -@{m.sp500.mu} + @{esn.f} \\times @{m.sp500.sd} = -@{m.sp500.mu} + @{sp.fsd} = @{m.sp500.n.e}\\%$', 'ES 2,5\\% $= -@{m.sp500.mu} + @{esn.f} \\times @{m.sp500.sd} = -@{m.sp500.mu} + @{sp.fsd} = @{m.sp500.n.e}\\%$')]),
    (T('Compare with the data (historical simulation): VaR 1\\% $= @{m.sp500.hs.v}\\%$, ES 2.5\\% $= @{m.sp500.hs.e}\\%$', 'Comparăm cu datele (simulare istorică): VaR 1\\% $= @{m.sp500.hs.v}\\%$, ES 2,5\\% $= @{m.sp500.hs.e}\\%$'),
     [T('the Normal VaR 1\\% is @{sp.gapN}\\% too low: the Normal tail is too thin (Chapter 2)', 'VaR 1\\% Normal este prea mic cu @{sp.gapN}\\%: coada distribuției Normale este prea subțire (Capitolul 2)'),
      T('the Normal ES 2.5\\% is even further from the data: it ignores the extreme days', 'ES 2,5\\% Normal este și mai departe de date: ignoră zilele extreme')]),
    T('Interpretation: two parameters cannot describe the tail; the next methods change either the distribution or the data used', 'Interpretare: doi parametri nu pot descrie coada; metodele următoare schimbă fie distribuția, fie datele folosite')))

D.frame(T('The Student-t Distribution: Closed Forms', 'Distribuția Student-t: formule explicite'), items(
    (T('$X = m + s\\,T$, $T \\sim t_\\nu$ (location $m$, scale $s$, $\\nu$ degrees of freedom, Chapter 2); $t_\\alpha = t_\\nu^{-1}(\\alpha)$, $f_\\nu$ the density of $t_\\nu$',
       '$X = m + s\\,T$, $T \\sim t_\\nu$ (poziția $m$, scala $s$, $\\nu$ grade de libertate, Capitolul 2); $t_\\alpha = t_\\nu^{-1}(\\alpha)$, $f_\\nu$ densitatea lui $t_\\nu$'),
     [T('$\\mathrm{VaR}_\\alpha = -(m + s\\,t_\\alpha)$, \\quad $\\mathrm{ES}_\\alpha = -m + s\\,\\dfrac{f_\\nu(t_\\alpha)}{\\alpha}\\,\\dfrac{\\nu + t_\\alpha^2}{\\nu - 1}$ (for $\\nu > 1$)',
        '$\\mathrm{VaR}_\\alpha = -(m + s\\,t_\\alpha)$, \\quad $\\mathrm{ES}_\\alpha = -m + s\\,\\dfrac{f_\\nu(t_\\alpha)}{\\alpha}\\,\\dfrac{\\nu + t_\\alpha^2}{\\nu - 1}$ (pentru $\\nu > 1$)'),
      T('the variance is $s^2\\nu/(\\nu - 2)$: a \\textbf{standardised} t (variance 1) has $s = \\sqrt{(\\nu - 2)/\\nu}$', 'varianța este $s^2\\nu/(\\nu - 2)$: o t \\textbf{standardizată} (varianța 1) are $s = \\sqrt{(\\nu - 2)/\\nu}$')]),
    (T('\\textbf{Worked example}: S\\&P 500, maximum likelihood: $\\hat m = @{t.m}$, $\\hat s = @{t.s}$, $\\hat\\nu = @{m.sp500.nu}$', '\\textbf{Exemplu rezolvat}: S\\&P 500, verosimilitate maximă: $\\hat m = @{t.m}$, $\\hat s = @{t.s}$, $\\hat\\nu = @{m.sp500.nu}$'),
     [T('$t_{0.01} = @{t.q}$: VaR 1\\% $= -(@{t.m} + @{t.s} \\times (@{t.q})) = @{m.sp500.t.v}\\%$', '$t_{0.01} = @{t.q}$: VaR 1\\% $= -(@{t.m} + @{t.s} \\times (@{t.q})) = @{m.sp500.t.v}\\%$'),
      T('ES factor $@{t.fac}$: ES 2.5\\% $= -@{t.m} + @{t.s} \\times @{t.fac} = @{m.sp500.t.e}\\%$', 'factorul ES $@{t.fac}$: ES 2,5\\% $= -@{t.m} + @{t.s} \\times @{t.fac} = @{m.sp500.t.e}\\%$')]),
    T('Interpretation: with $\\nu < 3$ the fitted tail is very heavy: VaR 1\\% matches the data, ES 2.5\\% is above the historical $@{m.sp500.hs.e}\\%$',
      'Interpretare: cu $\\nu < 3$, coada estimată este foarte groasă: VaR 1\\% se potrivește cu datele, ES 2,5\\% este peste valoarea istorică de $@{m.sp500.hs.e}\\%$')), 'footnotesize')

D.frame(T('From One Day to Ten Days', 'De la o zi la zece zile'), items(
    (T('\\textbf{Square-root-of-time rule}: $\\mathrm{VaR}^{(h)} \\approx \\sqrt{h}\\,\\mathrm{VaR}^{(1)}$', '\\textbf{Regula rădăcinii pătrate a timpului}: $\\mathrm{VaR}^{(h)} \\approx \\sqrt{h}\\,\\mathrm{VaR}^{(1)}$'),
     [T('exact for i.i.d.\\ Normal returns with mean 0: the $h$-day return has standard deviation $\\sigma\\sqrt{h}$', 'exactă pentru randamente Normale i.i.d.\\ cu media 0: randamentul pe $h$ zile are abaterea standard $\\sigma\\sqrt{h}$'),
      T('Basel 1996: 10-day VaR, usually scaled from one day with $\\sqrt{10} = @{sqrt10}$', 'Basel 1996: VaR pe 10 zile, de obicei scalat de la o zi cu $\\sqrt{10} = @{sqrt10}$')]),
    (T('S\\&P 500, historical simulation', 'S\\&P 500, simulare istorică'),
     [T('one day: @{hz.var1}\\%; scaled: $@{sqrt10} \\times @{hz.var1} = @{hz.var_sqrt}\\%$', 'o zi: @{hz.var1}\\%; scalat: $@{sqrt10} \\times @{hz.var1} = @{hz.var_sqrt}\\%$'),
      T('VaR 1\\% of overlapping 10-day returns: @{hz.var_h}\\%', 'VaR 1\\% al randamentelor pe 10 zile suprapuse: @{hz.var_h}\\%')]),
    (T('Why the rule fails in general', 'De ce regula nu funcționează în general'),
     [T('heavy tails: a sum of 10 heavy-tailed returns is closer to Normal (CLT, central limit theorem), so the ratio of quantiles is below $\\sqrt{10}$',
        'cozi groase: suma a 10 randamente cu cozi groase este mai aproape de Normală (CLT, teorema limită centrală), deci raportul cuantilelor este sub $\\sqrt{10}$'),
      T('volatility clustering: after a calm day the 10-day risk is larger than $\\sqrt{10}$ times today\'s, after a storm smaller (Chapter 9)',
        'volatility clustering: după o zi liniștită, riscul pe 10 zile este mai mare decît $\\sqrt{10}$ ori riscul de azi, după o perioadă turbulentă este mai mic (Capitolul 9)')])), 'footnotesize')

# =============================================================================
# 3. COERENȚĂ
# =============================================================================
D.section('Coherent Risk Measures', 'Măsuri de risc coerente')

D.frame(T('1999: Four Axioms for a Risk Measure', '1999: patru axiome pentru o măsură de risc'), cols(items(
    (T('\\refArtzner: a risk measure $\\rho$ maps a loss $L$ to the capital that makes it acceptable; $\\rho$ is \\textbf{coherent} if',
       '\\refArtzner: o măsură de risc $\\rho$ asociază unei pierderi $L$ capitalul care o face acceptabilă; $\\rho$ este \\textbf{coerentă} dacă'),
     [T('\\textbf{monotonicity}: $L_1 \\le L_2$ always $\\Rightarrow \\rho(L_1) \\le \\rho(L_2)$', '\\textbf{monotonie}: $L_1 \\le L_2$ întotdeauna $\\Rightarrow \\rho(L_1) \\le \\rho(L_2)$'),
      T('\\textbf{translation invariance}: $\\rho(L + c) = \\rho(L) + c$ for a sure amount $c$', '\\textbf{invarianță la translație}: $\\rho(L + c) = \\rho(L) + c$ pentru o sumă sigură $c$'),
      T('\\textbf{positive homogeneity}: $\\rho(\\lambda L) = \\lambda\\rho(L)$, $\\lambda > 0$', '\\textbf{omogenitate pozitivă}: $\\rho(\\lambda L) = \\lambda\\rho(L)$, $\\lambda > 0$'),
      T('\\textbf{subadditivity}: $\\rho(L_1 + L_2) \\le \\rho(L_1) + \\rho(L_2)$', '\\textbf{subaditivitate}: $\\rho(L_1 + L_2) \\le \\rho(L_1) + \\rho(L_2)$')]),
    T('Subadditivity: diversification never adds risk; a bank can add the capital of its desks as an upper bound', 'Subaditivitatea: diversificarea nu adaugă niciodată risc; o bancă poate aduna capitalul diviziilor ca margine superioară')),
    ph('delbaen', T('Freddy Delbaen, co-author of the axioms (1997)', 'Freddy Delbaen, coautor al axiomelor (1997)'), h='0.40\\textheight'),
    wl='0.64', wr='0.32'), 'footnotesize')

D.frame(T('Counterexample: Two Bonds, Step by Step', 'Contraexemplu: două obligațiuni, pas cu pas'), items(
    (T('Two independent bonds; each loses 100 with probability 4\\% (default) and 0 otherwise; level $\\alpha = 5\\%$',
       'Două obligațiuni independente; fiecare pierde 100 cu probabilitatea 4\\% (neplată) și 0 în rest; nivelul $\\alpha = 5\\%$'),
     [T('one bond: $P(L > 0) = 4\\% \\le 5\\%$, so $\\mathrm{VaR}_{5\\%} = 0$ for each', 'o obligațiune: $P(L > 0) = 4\\% \\le 5\\%$, deci $\\mathrm{VaR}_{5\\%} = 0$ pentru fiecare')]),
    (T('Both bonds: $P(L = 200) = @{bd.pboth}\\%$, $P(L = 100) = @{bd.pone}\\%$, so $P(L > 0) = @{bd.pany}\\% > 5\\%$', 'Ambele obligațiuni: $P(L = 200) = @{bd.pboth}\\%$, $P(L = 100) = @{bd.pone}\\%$, deci $P(L > 0) = @{bd.pany}\\% > 5\\%$'),
     [T('$\\mathrm{VaR}_{5\\%}(L_1 + L_2) = 100 > 0 + 0$: VaR \\textbf{punishes} diversification', '$\\mathrm{VaR}_{5\\%}(L_1 + L_2) = 100 > 0 + 0$: VaR \\textbf{penalizează} diversificarea')]),
    (T('ES 5\\% (the average of the worst 5\\% of outcomes)', 'ES 5\\% (media celor mai proaste 5\\% dintre rezultate)'),
     [T('one bond: $(4\\% \\times 100 + 1\\% \\times 0)/5\\% = @{bd.es1}$', 'o obligațiune: $(4\\% \\times 100 + 1\\% \\times 0)/5\\% = @{bd.es1}$'),
      T('both: $(@{bd.pboth}\\% \\times 200 + (5 - @{bd.pboth})\\% \\times 100)/5\\% = @{bd.es2} \\le @{bd.es1} + @{bd.es1}$: subadditive', 'ambele: $(@{bd.pboth}\\% \\times 200 + (5 - @{bd.pboth})\\% \\times 100)/5\\% = @{bd.es2} \\le @{bd.es1} + @{bd.es1}$: subaditiv')])), 'footnotesize')

D.frame(T('ES Is Coherent; VaR Only Sometimes', 'ES este coerent; VaR doar uneori'), items(
    (T('\\textbf{ES} satisfies all four axioms \\refAT', '\\textbf{ES} satisface toate cele patru axiome \\refAT'),
     [T('reason: ES is an average of quantiles over the tail, and averaging respects sums', 'motivul: ES este o medie a cuantilelor pe coadă, iar media respectă sumele')]),
    (T('\\textbf{VaR} satisfies the first three; subadditivity can fail', '\\textbf{VaR} le satisface pe primele trei; subaditivitatea poate să nu fie îndeplinită'),
     [T('fails for discrete losses (credit, the bonds above) and for extremely heavy tails (tail index below 1, Chapter 5)', 'nu este îndeplinită pentru pierderi discrete (credit, obligațiunile de mai sus) și pentru cozi extrem de groase (tail index-ul sub 1, Capitolul 5)'),
      T('holds for elliptical distributions (Normal, Student-t) and, in practice, for most equity returns \\refDJSS', 'este îndeplinită pentru distribuțiile eliptice (Normală, Student-t) și, în practică, pentru majoritatea randamentelor de acțiuni \\refDJSS')]),
    T('ES needs a finite mean of the losses; VaR always exists', 'ES are nevoie de o medie finită a pierderilor; VaR există întotdeauna')))

D.frame(T('Elicitability: Can a Forecast Be Scored?', 'Elicitabilitatea: evaluarea prognozelor'), items(
    (T('A statistic is \\textbf{elicitable} if some loss (scoring) function is minimised, on average, by its true value \\refGneiting',
       'O statistică este \\textbf{elicitabilă} dacă o funcție de pierdere (de scor) este minimizată, în medie, de valoarea ei adevărată \\refGneiting'),
     [T('the mean: squared error; the median: absolute error', 'media: eroarea pătratică; mediana: eroarea absolută')]),
    (T('\\textbf{VaR is elicitable}: the quantile (pinball) loss $S(v, x) = (\\alpha - \\mathbf{1}\\{x < -v\\})(x + v)$', '\\textbf{VaR este elicitabil}: pierderea cuantilă (pinball) $S(v, x) = (\\alpha - \\mathbf{1}\\{x < -v\\})(x + v)$'),
     [T('$v$: the VaR forecast; $x$: the realised return; $\\mathbf{1}\\{\\cdot\\}$: 1 if the event occurs, 0 otherwise', '$v$: prognoza VaR; $x$: randamentul realizat; $\\mathbf{1}\\{\\cdot\\}$: 1 dacă evenimentul are loc, altfel 0'),
      T('two VaR models can be compared by their average quantile loss', 'două modele VaR pot fi comparate prin media pierderii cuantile')]),
    (T('\\textbf{ES alone is not elicitable} \\refGneiting; the pair (VaR, ES) is \\refFZ', '\\textbf{ES singur nu este elicitabil} \\refGneiting; perechea (VaR, ES) este elicitabilă \\refFZ'),
     [T('consequence: ES forecasts are tested together with VaR forecasts (Section 7)', 'consecința: prognozele ES se testează împreună cu prognozele VaR (secțiunea 7)')])))

# =============================================================================
# 4. METODE NECONDIȚIONATE
# =============================================================================
D.section('Estimating VaR and ES', 'Estimarea VaR și ES')

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Five Unconditional Methods at a Glance', 'Cinci metode necondiționate, pe scurt'), table(
    TB + 'p{2.3cm}' + TB + 'p{4.0cm}' + TB + 'p{4.6cm}',
    T('\\textbf{Method}', '\\textbf{Metoda}') + ' & ' + T('\\textbf{Assumption}', '\\textbf{Ipoteza}') + ' & ' + T('\\textbf{Weak point}', '\\textbf{Punctul slab}'),
    [T('historical simulation (HS)', 'simulare istorică (HS)') + ' & ' + T('the past window is a sample of tomorrow', 'fereastra trecută este un eșantion din ziua de mîine') + ' & ' + T('few tail points; nothing beyond the worst day', 'puține puncte în coadă; nimic dincolo de cea mai proastă zi'),
     T('Normal', 'Normală') + ' & ' + T('returns are Normal', 'randamentele sînt Normale') + ' & ' + T('tails far too thin', 'cozi mult prea subțiri'),
     'Student-t & ' + T('returns are Student-t', 'randamentele urmează o distribuție Student-t') + ' & ' + T('one tail shape for both tails', 'o singură formă pentru ambele cozi'),
     'Cornish--Fisher & ' + T('Normal quantile corrected by skewness and kurtosis', 'cuantila Normală corectată prin asimetrie și boltire') + ' & ' + T('breaks down for large kurtosis', 'nu mai funcționează pentru boltire mare'),
     'EVT (POT) & ' + T('GPD above a high threshold (Chapter 5)', 'distribuția GPD peste un prag ridicat (Capitolul 5)') + ' & ' + T('choice of the threshold', 'alegerea pragului')],
    size='footnotesize') + items(
    T('``Unconditional\'\': the same VaR every day of the window; conditional methods (Section 5) use today\'s volatility', '„Necondiționat”: același VaR în fiecare zi a ferestrei; metodele condiționate (secțiunea 5) folosesc volatilitatea de azi'),
    T('All methods: daily log returns in \\% (Chapter 1), VaR and ES as positive losses', 'Toate metodele: randamente logaritmice zilnice în \\% (Capitolul 1), VaR și ES ca pierderi pozitive')), 'footnotesize')

D.frame(T('Historical Simulation', 'Simularea istorică'), items(
    (T('\\textbf{Algorithm}: take the last $n$ returns $r_{t-n+1}, \\dots, r_t$, sort them: $r_{(1)} \\le \\dots \\le r_{(n)}$', '\\textbf{Algoritmul}: luăm ultimele $n$ randamente $r_{t-n+1}, \\dots, r_t$ și le ordonăm: $r_{(1)} \\le \\dots \\le r_{(n)}$'),
     [T('$\\widehat{\\mathrm{VaR}}_\\alpha = -r_{(k)}$, $k = \\lceil n\\alpha \\rceil$; \\quad $\\widehat{\\mathrm{ES}}_\\alpha = -\\frac{1}{k}\\sum_{i=1}^{k} r_{(i)}$', '$\\widehat{\\mathrm{VaR}}_\\alpha = -r_{(k)}$, $k = \\lceil n\\alpha \\rceil$; \\quad $\\widehat{\\mathrm{ES}}_\\alpha = -\\frac{1}{k}\\sum_{i=1}^{k} r_{(i)}$'),
      T('no distribution is assumed: the empirical distribution of the window is the forecast', 'nu presupunem nicio distribuție: distribuția empirică a ferestrei este prognoza')]),
    (T('\\textbf{Worked example}: S\\&P 500 since 2000, $n = @{m.sp500.n}$', '\\textbf{Exemplu rezolvat}: S\\&P 500 din 2000, $n = @{m.sp500.n}$'),
     [T('VaR 1\\%: $k = \\lceil @{hs.n1} \\rceil = @{hs.k1}$, the @{hs.k1}-th worst day: $@{m.sp500.hs.v}\\%$', 'VaR 1\\%: $k = \\lceil @{hs.n1} \\rceil = @{hs.k1}$, a @{hs.k1}-a cea mai proastă zi: $@{m.sp500.hs.v}\\%$'),
      T('ES 2.5\\%: the average of the $\\lceil @{hs.n25} \\rceil = @{hs.k25}$ worst days: $@{m.sp500.hs.e}\\%$', 'ES 2,5\\%: media celor mai proaste $\\lceil @{hs.n25} \\rceil = @{hs.k25}$ de zile: $@{m.sp500.hs.e}\\%$')]),
    (T('The window $n$ is a trade-off', 'Fereastra $n$ este un compromis'),
     [T('short (250 days): reacts, but VaR 1\\% rests on 3 observations; long (all data): stable, but blind to today\'s volatility', 'scurtă (250 de zile): reacționează, dar VaR 1\\% se bazează pe 3 observații; lungă (toate datele): stabilă, dar insensibilă la volatilitatea de azi'),
      T('\\textbf{ghost effect}: a crash day changes VaR when it enters the window and again, abruptly, $n$ days later when it leaves', '\\textbf{ghost effect}: o zi de crah schimbă VaR cînd intră în fereastră și din nou, brusc, după $n$ zile, cînd iese din ea')])), 'footnotesize')

D.frame(T('How Precise Is a Historical VaR?', 'Precizia unui VaR istoric'), items(
    (T('A VaR estimate is a statistic: it has a sampling error; we measure it with the bootstrap (Chapter 2)', 'O estimare VaR este o statistică: are o eroare de eșantionare; o măsurăm prin bootstrap (Capitolul 2)'),
     [T('resample the $n$ returns with replacement 2000 times, recompute VaR and ES, read the 2.5\\% and 97.5\\% quantiles', 'reeșantionăm cele $n$ randamente cu întoarcere de 2000 de ori, recalculăm VaR și ES și citim cuantilele de 2,5\\% și 97,5\\%')]),
    (T('S\\&P 500, all @{pr.all.n} days: VaR 1\\% $= @{pr.all.var}\\%$, 95\\% interval [@{pr.all.var_lo}, @{pr.all.var_hi}]; ES 2.5\\% $= @{pr.all.es}\\%$, [@{pr.all.es_lo}, @{pr.all.es_hi}]',
       'S\\&P 500, toate cele @{pr.all.n} zile: VaR 1\\% $= @{pr.all.var}\\%$, intervalul de 95\\% [@{pr.all.var_lo}, @{pr.all.var_hi}]; ES 2,5\\% $= @{pr.all.es}\\%$, [@{pr.all.es_lo}, @{pr.all.es_hi}]'),
     [T('last @{pr.last.n} days: VaR 1\\% $= @{pr.last.var}\\%$, [@{pr.last.var_lo}, @{pr.last.var_hi}]; ES 2.5\\% $= @{pr.last.es}\\%$, [@{pr.last.es_lo}, @{pr.last.es_hi}]',
        'ultimele @{pr.last.n} zile: VaR 1\\% $= @{pr.last.var}\\%$, [@{pr.last.var_lo}, @{pr.last.var_hi}]; ES 2,5\\% $= @{pr.last.es}\\%$, [@{pr.last.es_lo}, @{pr.last.es_hi}]')]),
    T('With two years of data the interval of VaR 1\\% is wider than 2 percentage points: VaR 1\\% rests on 5 observations', 'Cu doi ani de date, intervalul pentru VaR 1\\% are o lățime de peste 2 puncte procentuale: VaR 1\\% se bazează pe 5 observații'),
    T('Interpretation: differences between methods smaller than the bootstrap interval are not evidence that one method is better', 'Interpretare: diferențele dintre metode mai mici decît intervalul bootstrap nu dovedesc că o metodă este mai bună')) + ql('SFM_ch10_var_methods'), 'footnotesize')

D.frame(T('The Cornish--Fisher Expansion', 'Dezvoltarea Cornish--Fisher'), items(
    (T('\\refCF: correct the Normal quantile $z = z_\\alpha$ with the skewness $S$ and the excess kurtosis $K$ (kurtosis minus 3)',
       '\\refCF: corectăm cuantila Normală $z = z_\\alpha$ cu asimetria $S$ și excesul de boltire $K$ (coeficientul de boltire minus 3)'),
     [T('$z_{CF} = z + \\dfrac{(z^2 - 1)S}{6} + \\dfrac{(z^3 - 3z)K}{24} - \\dfrac{(2z^3 - 5z)S^2}{36}$, \\quad $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_{CF})$',
        '$z_{CF} = z + \\dfrac{(z^2 - 1)S}{6} + \\dfrac{(z^3 - 3z)K}{24} - \\dfrac{(2z^3 - 5z)S^2}{36}$, \\quad $\\mathrm{VaR}_\\alpha = -(\\mu + \\sigma z_{CF})$'),
      T('negative $S$ and positive $K$ push the quantile further into the left tail', '$S$ negativ și $K$ pozitiv împing cuantila mai departe în coada stîngă')]),
    (T('S\\&P 500: $S = @{m.sp500.skew}$, $K = @{m.sp500.k}$, $z_{0.01} = @{cf.z1}$', 'S\\&P 500: $S = @{m.sp500.skew}$, $K = @{m.sp500.k}$, $z_{0.01} = @{cf.z1}$'),
     [T('$z_{CF} = @{m.sp500.cfz}$, so VaR 1\\% $= @{m.sp500.cf.v}\\%$, against the historical $@{m.sp500.hs.v}\\%$', '$z_{CF} = @{m.sp500.cfz}$, deci VaR 1\\% $= @{m.sp500.cf.v}\\%$, față de valoarea istorică de $@{m.sp500.hs.v}\\%$')]),
    (T('Interpretation: the expansion is a small correction around the Normal; with an excess kurtosis above 10 it overshoots badly', 'Interpretare: dezvoltarea este o corecție mică în jurul distribuției Normale; cu un exces de boltire de peste 10, ea depășește mult valoarea corectă'),
     [T('use it only when $|S|$ and $K$ are moderate (for example monthly returns or diversified portfolios)', 'o folosim doar cînd $|S|$ și $K$ sînt moderate (de exemplu, pentru randamente lunare sau portofolii diversificate)')])), 'footnotesize')

D.frame(T('EVT: Peaks over Threshold (Chapter 5)', 'EVT: peaks over threshold (Capitolul 5)'), items(
    (T('Losses $L = -r$; threshold $u$ = the 90\\% quantile of the losses \\refMF; excesses $L - u$ above $u$ follow a GPD (generalised Pareto distribution) with shape $\\xi$ and scale $\\beta$',
       'Pierderile $L = -r$; pragul $u$ = cuantila de 90\\% a pierderilor \\refMF; excesele $L - u$ peste $u$ urmează o distribuție GPD (Pareto generalizată) cu forma $\\xi$ și scala $\\beta$'),
     [T('$\\mathrm{VaR}_\\alpha = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{n\\alpha}{N_u}\\Big)^{-\\xi} - 1\\Big]$, \\quad $\\mathrm{ES}_\\alpha = \\dfrac{\\mathrm{VaR}_\\alpha + \\beta - \\xi u}{1 - \\xi}$ ($N_u$: number of excesses)',
        '$\\mathrm{VaR}_\\alpha = u + \\dfrac{\\beta}{\\xi}\\Big[\\Big(\\dfrac{n\\alpha}{N_u}\\Big)^{-\\xi} - 1\\Big]$, \\quad $\\mathrm{ES}_\\alpha = \\dfrac{\\mathrm{VaR}_\\alpha + \\beta - \\xi u}{1 - \\xi}$ ($N_u$: numărul de excese)'),
      T('$n$: the number of days; $n\\alpha/N_u$: the tail probability relative to the share of excesses; $\\xi > 0$: a heavy tail', '$n$: numărul de zile; $n\\alpha/N_u$: probabilitatea cozii raportată la ponderea exceselor; $\\xi > 0$: o coadă groasă')]),
    (T('S\\&P 500: $u = @{m.sp500.u}\\%$, $N_u = @{m.sp500.Nu}$ (@{evt.share}\\% of the days), $\\hat\\xi = @{m.sp500.xi3}$, $\\hat\\beta = @{m.sp500.beta}$',
       'S\\&P 500: $u = @{m.sp500.u}\\%$, $N_u = @{m.sp500.Nu}$ (@{evt.share}\\% dintre zile), $\\hat\\xi = @{m.sp500.xi3}$, $\\hat\\beta = @{m.sp500.beta}$'),
     [T('VaR 1\\%: $n\\alpha/N_u = @{evt.r1}$, $@{evt.r1}^{-\\hat\\xi} = @{evt.pow1}$; $\\hat\\beta/\\hat\\xi = @{evt.boxi}$; $@{m.sp500.u} + @{evt.boxi} \\times (@{evt.pow1} - 1) = @{m.sp500.evt.v}\\%$',
        'VaR 1\\%: $n\\alpha/N_u = @{evt.r1}$, $@{evt.r1}^{-\\hat\\xi} = @{evt.pow1}$; $\\hat\\beta/\\hat\\xi = @{evt.boxi}$; $@{m.sp500.u} + @{evt.boxi} \\times (@{evt.pow1} - 1) = @{m.sp500.evt.v}\\%$'),
      T('ES 2.5\\% $= @{m.sp500.evt.e}\\%$; $\\hat\\xi > 0$: a heavy (Pareto-type) tail', 'ES 2,5\\% $= @{m.sp500.evt.e}\\%$; $\\hat\\xi > 0$: o coadă groasă (de tip Pareto)')]),
    T('EVT smooths the tail and extrapolates beyond the worst observed day: useful for VaR 0.1\\% and for stress tests', 'EVT netezește coada și extrapolează dincolo de cea mai proastă zi observată: util pentru VaR 0,1\\% și pentru testele de stres')), 'footnotesize')

chart(T('VaR across Tail Probabilities', 'VaR pentru diferite probabilități ale cozii'), 'sfm_ch10_var_curves', 'SFM_ch10_var_methods', [
    T('Each line: $\\mathrm{VaR}_\\alpha$ of one method for $\\alpha$ from 5\\% to 0.1\\%; points: historical simulation', 'Fiecare linie: $\\mathrm{VaR}_\\alpha$ al unei metode pentru $\\alpha$ de la 5\\% la 0,1\\%; punctele: simularea istorică'),
    T('Interpretation: at 5\\% all methods agree; at 1\\% the Normal is too low; at 0.1\\% Student-t and EVT stay close to the data, while Cornish--Fisher explodes',
      'Interpretare: la 5\\% toate metodele sînt de acord; la 1\\% distribuția Normală dă valori prea mici; la 0,1\\% Student-t și EVT rămîn aproape de date, iar Cornish--Fisher crește exploziv')],
    h='0.52\\textheight')


def mrow(k, kind):
    return (f'{SHORT[k]} & ' + ' & '.join(f'$@{{m.{k}.{MKEY[m]}.{kind}}}$' for m in ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT'])
            + f' & ${{@{{m.{k}.k}}}}$ & $@{{m.{k}.xi}}$')


MH = T('& HS & Normal & Student-t & Cornish--Fisher & EVT & $K$ & $\\hat\\xi$', '& HS & Normală & Student-t & Cornish--Fisher & EVT & $K$ & $\\hat\\xi$')
D.frame(T('VaR 1\\% by Method on Six Series', 'VaR 1\\% pe metode, pentru șase serii'), table(
    'lrrrrrrr', MH, [mrow(k, 'v') for k in ASSETS], size='footnotesize') + items(
    T('Whole samples (S\\&P 500, DAX, BET since 2000; Bitcoin since 2014; TLV and SNP since 2010), in \\%; $K$: excess kurtosis', 'Eșantioanele complete (S\\&P 500, DAX, BET din 2000; Bitcoin din 2014; TLV și SNP din 2010), în \\%; $K$: excesul de boltire'),
    T('HS, Student-t and EVT differ by at most @{rg.dif.max} percentage points; the Normal VaR is @{rg.gapN.lo}--@{rg.gapN.hi}\\% too low everywhere', 'HS, Student-t și EVT diferă cu cel mult @{rg.dif.max} puncte procentuale; VaR-ul Normal este peste tot prea mic cu @{rg.gapN.lo}--@{rg.gapN.hi}\\%'),
    T('Cornish--Fisher gives @{rg.cf.lo} to @{rg.cf.hi} times the historical VaR: with $K$ between @{rg.k.lo} and @{rg.k.hi} the expansion is outside its range', 'Cornish--Fisher dă de @{rg.cf.lo} pînă la @{rg.cf.hi} ori VaR-ul istoric: cu $K$ între @{rg.k.lo} și @{rg.k.hi}, dezvoltarea este în afara domeniului ei de valabilitate')) + ql('SFM_ch10_var_methods'), 'footnotesize')

D.frame(T('ES 2.5\\% by Method on Six Series', 'ES 2,5\\% pe metode, pentru șase serii'), table(
    'lrrrrrrr', MH, [mrow(k, 'e') for k in ASSETS], size='footnotesize') + items(
    T('Normal: ES 2.5\\% equals its VaR 1\\%; the data: the historical ES 2.5\\% is @{rg.es.lo}--@{rg.es.hi}\\% above the historical VaR 1\\%', 'Distribuția Normală: ES 2,5\\% este egal cu VaR 1\\%; datele: ES 2,5\\% istoric este cu @{rg.es.lo}--@{rg.es.hi}\\% peste VaR 1\\% istoric'),
    T('For the S\\&P 500, DAX, BET and Bitcoin, Student-t gives the largest ES among the sensible methods (Bitcoin: $@{m.btc.t.e}\\%$ against $@{m.btc.hs.e}\\%$): with $\\hat\\nu < 3$ its tail is heavier than the data beyond the threshold',
      'Pentru S\\&P 500, DAX, BET și Bitcoin, Student-t dă cel mai mare ES dintre metodele rezonabile (Bitcoin: $@{m.btc.t.e}\\%$ față de $@{m.btc.hs.e}\\%$): cu $\\hat\\nu < 3$, coada ei este mai groasă decît cea a datelor de dincolo de prag'),
    T('Interpretation: EVT and HS agree at 2.5\\%; the method matters more for ES than for VaR, because ES depends on the whole tail',
      'Interpretare: EVT și HS sînt de acord la 2,5\\%; metoda contează mai mult pentru ES decît pentru VaR, deoarece ES depinde de întreaga coadă')) + ql('SFM_ch10_var_methods'), 'footnotesize')

# =============================================================================
# 5. VaR CONDIȚIONAT
# =============================================================================
D.section('Conditional VaR: GARCH and Filtered Historical Simulation', 'VaR condiționat: GARCH și simularea istorică filtrată')

D.frame(T('Why Condition on Today\'s Volatility?', 'Rolul volatilității de azi'), items(
    (T('Volatility clustering (Chapters 8 and 9): a calm day is followed by calm days, a storm by storms', 'Volatility clustering (Capitolele 8 și 9): după o zi liniștită urmează zile liniștite, după o zi agitată urmează zile agitate'),
     [T('an unconditional VaR is too high in calm years and too low at the start of a crisis', 'un VaR necondiționat este prea mare în anii liniștiți și prea mic la începutul unei crize')]),
    (T('Conditional model: $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$, $z$ i.i.d.\\ with mean 0 and variance 1', 'Modelul condiționat: $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$, $z$ i.i.d.\\ cu media 0 și varianța 1'),
     [T('$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$, \\quad $\\mathrm{ES}_{t+1} = -\\mu + \\sigma_{t+1}\\,\\mathrm{ES}_\\alpha(z)$', '$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$, \\quad $\\mathrm{ES}_{t+1} = -\\mu + \\sigma_{t+1}\\,\\mathrm{ES}_\\alpha(z)$'),
      T('$q_\\alpha(z)$, $\\mathrm{ES}_\\alpha(z)$: the quantile and the ES of the standardised innovation $z$', '$q_\\alpha(z)$, $\\mathrm{ES}_\\alpha(z)$: cuantila și ES ale inovației standardizate $z$'),
      T('two ingredients: the volatility forecast $\\sigma_{t+1}$ and the distribution of $z$', 'două ingrediente: prognoza volatilității $\\sigma_{t+1}$ și distribuția lui $z$')]),
    (T('Two choices for the distribution of $z$', 'Două variante pentru distribuția lui $z$'),
     [T('\\textbf{GARCH-t}: standardised Student-t with $\\nu$ estimated with the GARCH(1,1) \\refBoll', '\\textbf{GARCH-t}: Student-t standardizată, cu $\\nu$ estimat împreună cu GARCH(1,1) \\refBoll'),
      T('\\textbf{FHS} (filtered historical simulation): the empirical distribution of the standardised residuals $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$ \\refHW; \\refBGV',
        '\\textbf{FHS} (filtered historical simulation, simularea istorică filtrată): distribuția empirică a reziduurilor standardizate $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$ \\refHW; \\refBGV')])), 'footnotesize')

D.frame(T('Worked Example: the S\\&P 500 on @{c.sp500.last}', 'Exemplu rezolvat: S\\&P 500 la @{c.sp500.last}'), items(
    T('GARCH(1,1)-t on all data: $\\hat\\mu = @{c.sp500.mu}$, $\\hat\\alpha = @{c.sp500.a}$, $\\hat\\beta = @{c.sp500.b}$, $\\hat\\nu = @{c.sp500.nu}$; forecast for the next day $\\sigma_{t+1} = @{c.sp500.sig}\\%$',
      'GARCH(1,1)-t pe toate datele: $\\hat\\mu = @{c.sp500.mu}$, $\\hat\\alpha = @{c.sp500.a}$, $\\hat\\beta = @{c.sp500.b}$, $\\hat\\nu = @{c.sp500.nu}$; prognoza pentru ziua următoare $\\sigma_{t+1} = @{c.sp500.sig}\\%$'),
    (T('\\textbf{GARCH-t}: $q_{0.01}(z) = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu} = @{c.sp500.q}$; $\\mathrm{ES}_{2.5\\%}(z) = @{c.sp500.est}$', '\\textbf{GARCH-t}: $q_{0.01}(z) = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu} = @{c.sp500.q}$; $\\mathrm{ES}_{2.5\\%}(z) = @{c.sp500.est}$'),
     [T('VaR 1\\% $= -(@{c.sp500.mu} + @{c.sp500.sig} \\times (@{c.sp500.q})) = @{c.sp500.vg}\\%$; ES 2.5\\% $= -@{c.sp500.mu} + @{c.sp500.sig} \\times @{c.sp500.est} = @{c.sp500.eg}\\%$',
        'VaR 1\\% $= -(@{c.sp500.mu} + @{c.sp500.sig} \\times (@{c.sp500.q})) = @{c.sp500.vg}\\%$; ES 2,5\\% $= -@{c.sp500.mu} + @{c.sp500.sig} \\times @{c.sp500.est} = @{c.sp500.eg}\\%$')]),
    (T('\\textbf{FHS}: the 1\\% quantile of the @{m.sp500.n} standardised residuals is $@{c.sp500.qz}$; their ES 2.5\\% is $@{c.sp500.esz}$', '\\textbf{FHS}: cuantila de 1\\% a celor @{m.sp500.n} reziduuri standardizate este $@{c.sp500.qz}$; ES 2,5\\% al lor este $@{c.sp500.esz}$'),
     [T('VaR 1\\% $= -(@{c.sp500.mu} + @{c.sp500.sig} \\times (@{c.sp500.qz})) = @{c.sp500.vf}\\%$; ES 2.5\\% $= @{c.sp500.ef}\\%$', 'VaR 1\\% $= -(@{c.sp500.mu} + @{c.sp500.sig} \\times (@{c.sp500.qz})) = @{c.sp500.vf}\\%$; ES 2,5\\% $= @{c.sp500.ef}\\%$')]),
    T('Interpretation: a calm market: the conditional VaR 1\\% is about half of the unconditional $@{m.sp500.hs.v}\\%$; the standardised residuals have a heavier left tail than the fitted t',
      'Interpretare: o piață liniștită: VaR 1\\% condiționat este cam jumătate din valoarea necondiționată de $@{m.sp500.hs.v}\\%$; reziduurile standardizate au o coadă stîngă mai groasă decît t estimată')) + ql('SFM_ch10_conditional_var'), 'footnotesize')


def crow(k):
    return f'{SHORT[k]} & $@{{m.{k}.hs.v}}$ & $@{{c.{k}.vg}}$ & $@{{c.{k}.vf}}$ & $@{{m.{k}.hs.e}}$ & $@{{c.{k}.eg}}$ & $@{{c.{k}.ef}}$ & $@{{c.{k}.sig}}$'


D.frame(T('Next-Day VaR and ES on Six Series', 'VaR și ES pentru ziua următoare, pe șase serii'), table(
    'lrrrrrrr', T('& \\multicolumn{3}{c}{VaR 1\\%} & \\multicolumn{3}{c}{ES 2.5\\%} & \\\\ & HS (all) & GARCH-t & FHS & HS (all) & GARCH-t & FHS & $\\sigma_{t+1}$',
                  '& \\multicolumn{3}{c}{VaR 1\\%} & \\multicolumn{3}{c}{ES 2,5\\%} & \\\\ & HS (tot) & GARCH-t & FHS & HS (tot) & GARCH-t & FHS & $\\sigma_{t+1}$'),
    [crow(k) for k in ASSETS], size='footnotesize') + items(
    T('Forecasts for the day after @{end}, in \\%; HS (all): the unconditional historical values of the previous section', 'Prognoze pentru ziua de după @{end}, în \\%; HS (tot): valorile istorice necondiționate din secțiunea precedentă'),
    T('S\\&P 500, DAX, Bitcoin: calm, the conditional values are below the unconditional ones', 'S\\&P 500, DAX, Bitcoin: perioadă liniștită, valorile condiționate sînt sub cele necondiționate'),
    T('Interpretation: BET, TLV and SNP: the current volatility is high, so the conditional VaR is above the long-run historical VaR; the same method gives opposite messages on different markets',
      'Interpretare: BET, TLV și SNP: volatilitatea curentă este mare, deci VaR condiționat este peste VaR istoric pe termen lung; aceeași metodă transmite mesaje opuse pe piețe diferite')) + ql('SFM_ch10_conditional_var'), 'footnotesize')

D.frame(T('Rolling Forecasts: the Design', 'Prognoze pe fereastră mobilă: schema de evaluare'), items(
    (T('\\textbf{Out of sample}: the forecast for day $t$ uses only data up to day $t-1$', '\\textbf{În afara eșantionului}: prognoza pentru ziua $t$ folosește doar date pînă în ziua $t-1$'),
     [T('S\\&P 500, DAX, BET: every day since January 2005 (2008 included); Bitcoin: since January 2017', 'S\\&P 500, DAX, BET: fiecare zi din ianuarie 2005 (inclusiv 2008); Bitcoin: din ianuarie 2017')]),
    (T('Four methods', 'Patru metode'),
     [T('\\textbf{HS} and \\textbf{Normal}: the last 500 days (about two years)', '\\textbf{HS} și \\textbf{Normală}: ultimele 500 de zile (circa doi ani)'),
      T('\\textbf{GARCH-t} and \\textbf{FHS}: GARCH(1,1)-t re-estimated every 250 days on all past data; $\\sigma_t$ updated every day', '\\textbf{GARCH-t} și \\textbf{FHS}: GARCH(1,1)-t reestimat la fiecare 250 de zile pe toate datele trecute; $\\sigma_t$ actualizat zilnic')]),
    T('Each day: VaR 1\\% (for the backtests), VaR 2.5\\% and ES 2.5\\% (for the ES backtest)', 'În fiecare zi: VaR 1\\% (pentru backtesting), VaR 2,5\\% și ES 2,5\\% (pentru backtesting-ul ES)')) + ql('SFM_ch10_conditional_var'))

chart(T('VaR 1\\% through 2008 and 2020', 'VaR 1\\% în 2008 și în 2020'), 'sfm_ch10_rolling_var', 'SFM_ch10_conditional_var', [
    T('S\\&P 500: daily returns and minus the VaR 1\\% forecasts of the four methods; dots: exceptions of the HS VaR', 'S\\&P 500: randamentele zilnice și minus prognozele VaR 1\\% ale celor patru metode; punctele: depășirile VaR HS'),
    T('HS and Normal react months late: the 500-day window changes slowly; GARCH-t and FHS jump within days', 'HS și metoda Normală reacționează cu luni de întîrziere: fereastra de 500 de zile se schimbă încet; GARCH-t și FHS cresc brusc în cîteva zile'),
    T('Interpretation: after the crisis HS stays high for two years (the ghost effect), while GARCH-t returns to normal levels', 'Interpretare: după criză, HS rămîne ridicat timp de doi ani (ghost effect), în timp ce GARCH-t revine la niveluri normale')],
    h='0.52\\textheight')


def srow(k):
    return (f'{SHORT[k]} & ' + ' & '.join(f'@{{bt.{k}.{MKEY[m]}.s2008}}' for m in METHODS) + f' & @{{bt.{k}.hs.s2008e}} & '
            + ' & '.join(f'@{{bt.{k}.{MKEY[m]}.s2020}}' for m in METHODS) + f' & @{{bt.{k}.hs.s2020e}}')


D.frame(T('Stress Periods: Exceptions in 2008 and 2020', 'Perioade de stres: depășiri în 2008 și 2020'), table(
    'l|rrrr|r|rrrr|r', T('& \\multicolumn{5}{c|}{Sep 2008 -- Mar 2009} & \\multicolumn{5}{c}{Feb -- May 2020} \\\\ & HS & Normal & GARCH-t & FHS & exp. & HS & Normal & GARCH-t & FHS & exp.',
                         '& \\multicolumn{5}{c|}{sep. 2008 -- mar. 2009} & \\multicolumn{5}{c}{feb. -- mai 2020} \\\\ & HS & Normală & GARCH-t & FHS & aștept. & HS & Normală & GARCH-t & FHS & aștept.'),
    [srow(k) for k in ['sp500', 'dax', 'bet']], size='footnotesize') + items(
    T('Exceptions of the VaR 1\\%: days with $r_t < -\\mathrm{VaR}_t$; exp.: 1\\% of the days in the period', 'Depășiri ale VaR 1\\%: zile cu $r_t < -\\mathrm{VaR}_t$; aștept.: 1\\% din zilele perioadei'),
    T('In 2008 the S\\&P 500 Normal VaR was exceeded on @{bt.sp500.n.s2008} of @{bt.sp500.n.s2008n} days, HS on @{bt.sp500.hs.s2008}; GARCH-t and FHS on @{bt.sp500.g.s2008}',
      'În 2008, VaR Normal pentru S\\&P 500 a fost depășit în @{bt.sp500.n.s2008} din @{bt.sp500.n.s2008n} zile, HS în @{bt.sp500.hs.s2008}; GARCH-t și FHS în @{bt.sp500.g.s2008}'),
    T('Interpretation: no method is perfect in a crisis, but the conditional ones fail far less: the volatility forecast matters more than the tail shape',
      'Interpretare: nicio metodă nu este perfectă într-o criză, dar cele condiționate greșesc mult mai rar: prognoza volatilității contează mai mult decît forma cozii')) + ql('SFM_ch10_backtesting'), 'footnotesize')

D.frame(T('Which Method When?', 'Alegerea metodei'), table(
    TB + 'p{3.6cm}' + TB + 'p{3.3cm}' + TB + 'p{4.0cm}',
    T('\\textbf{Situation}', '\\textbf{Situația}') + ' & ' + T('\\textbf{Method}', '\\textbf{Metoda}') + ' & ' + T('\\textbf{Reason}', '\\textbf{Motivul}'),
    [T('daily VaR of a liquid position', 'VaR zilnic pentru o poziție lichidă') + ' & GARCH-t, FHS & ' + T('follow the volatility; FHS keeps the empirical tail', 'urmăresc volatilitatea; FHS păstrează coada empirică'),
     T('long history, quick check', 'istoric lung, verificare rapidă') + ' & HS & ' + T('no model, easy to explain', 'fără model, ușor de explicat'),
     T('very small $\\alpha$ (0.1\\%), stress tests', '$\\alpha$ foarte mic (0,1\\%), teste de stres') + ' & EVT & ' + T('extrapolates beyond the worst day', 'extrapolează dincolo de cea mai proastă zi'),
     T('portfolio of many assets', 'portofoliu cu multe active') + ' & ' + T('variance--covariance, copula', 'varianță--covarianță, copulă') + ' & ' + T('dependence; the t copula for joint crashes', 'dependența; copula t pentru prăbușirile simultane'),
     T('moderate skewness and kurtosis', 'asimetrie și boltire moderate') + ' & Cornish--Fisher & ' + T('a quick correction of the Normal', 'o corecție rapidă a distribuției Normale')],
    size='footnotesize') + items(
    T('Never only the Normal for daily data; always report the method, the window and the level', 'Niciodată doar distribuția Normală pentru date zilnice; raportăm întotdeauna metoda, fereastra și nivelul'),
    T('Whatever the method: backtest it (Section 7)', 'Indiferent de metodă: o testăm prin backtesting (secțiunea 7)')), 'footnotesize')

# =============================================================================
# 6. PORTOFOLIU
# =============================================================================
D.section('Portfolio VaR and Dependence', 'VaR de portofoliu și dependența')

D.frame(T('A BET and S\\&P 500 Portfolio', 'Un portofoliu BET și S\\&P 500'), items(
    (T('Weights $w = (0.5, 0.5)$; simple returns $R_t$ in \\%, so that $R_{p,t} = w_1R_{1,t} + w_2R_{2,t}$ exactly', 'Ponderile $w = (0{,}5;\\ 0{,}5)$; randamente simple $R_t$ în \\%, astfel încît $R_{p,t} = w_1R_{1,t} + w_2R_{2,t}$ exact'),
     [T('prices joined on the @{p.n} common trading days since 2000, then returns', 'prețurile se aliniază pe cele @{p.n} zile comune de tranzacționare din 2000, apoi se calculează randamentele'),
      T('each index in its own currency: the currency risk of a RON investor is left out', 'fiecare indice în moneda lui: riscul valutar al unui investitor în lei este ignorat')]),
    (T('Daily data: BET $\\hat\\mu = @{p.mu1}$, $\\hat\\sigma = @{p.sd1}$; S\\&P 500 $\\hat\\mu = @{p.mu2}$, $\\hat\\sigma = @{p.sd2}$; correlation $\\hat\\rho = @{p.corr}$',
       'Date zilnice: BET $\\hat\\mu = @{p.mu1}$, $\\hat\\sigma = @{p.sd1}$; S\\&P 500 $\\hat\\mu = @{p.mu2}$, $\\hat\\sigma = @{p.sd2}$; corelația $\\hat\\rho = @{p.corr}$'),
     [T('the BVB closes before New York opens: part of the joint reaction appears one day later, so same-day correlation understates the link',
        'BVB se închide înainte de deschiderea bursei din New York: o parte din reacția comună apare a doua zi, deci corelația din aceeași zi subestimează legătura')]),
    T('Question: is the portfolio VaR smaller than the average of the two VaRs, and by how much?', 'Întrebarea: este VaR-ul portofoliului mai mic decît media celor două VaR și cu cît?')))

D.frame(T('The Variance--Covariance Method', 'Metoda varianță--covarianță'), items(
    (T('Portfolio: $\\mu_p = w^\\top\\mu$, $\\sigma_p^2 = w^\\top\\Sigma w = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\sigma_1\\sigma_2$ ($\\Sigma$: covariance matrix)',
       'Portofoliul: $\\mu_p = w^\\top\\mu$, $\\sigma_p^2 = w^\\top\\Sigma w = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\sigma_1\\sigma_2$ ($\\Sigma$: matricea de covarianță)'),
     [T('$w$: the vector of weights; $\\mu$: the vector of mean returns; $\\rho$: the correlation of the two assets', '$w$: vectorul ponderilor; $\\mu$: vectorul randamentelor medii; $\\rho$: corelația celor două active'),
      T('if returns are jointly Normal: $\\mathrm{VaR}_p = -(\\mu_p + z_\\alpha\\sigma_p)$ (the RiskMetrics method \\refRM)', 'dacă randamentele sînt Normale în ansamblu: $\\mathrm{VaR}_p = -(\\mu_p + z_\\alpha\\sigma_p)$ (metoda RiskMetrics \\refRM)')]),
    (T('\\textbf{Step by step}: $\\sigma_p^2 = @{p.w1sd}^2 + @{p.w2sd}^2 + 2 \\times 0.5 \\times 0.5 \\times @{p.cov} = @{p.sp2}$, $\\sigma_p = @{p.sp}$',
       '\\textbf{Pas cu pas}: $\\sigma_p^2 = @{p.w1sd}^2 + @{p.w2sd}^2 + 2 \\times 0.5 \\times 0.5 \\times @{p.cov} = @{p.sp2}$, $\\sigma_p = @{p.sp}$'),
     [T('VaR 1\\% $= -(@{p.mp} - @{z1} \\times @{p.sp}) = @{p.var_n}\\%$; the weighted sum of the two Normal VaRs: $0.5 \\times @{p.vn1} + 0.5 \\times @{p.vn2} = @{p.var_n_sum}\\%$',
        'VaR 1\\% $= -(@{p.mp} - @{z1} \\times @{p.sp}) = @{p.var_n}\\%$; suma ponderată a celor două VaR Normale: $0.5 \\times @{p.vn1} + 0.5 \\times @{p.vn2} = @{p.var_n_sum}\\%$'),
      T('\\textbf{diversification benefit}: @{p.ben}\\% of the risk disappears; with $\\rho = 1$ it would be zero', '\\textbf{beneficiul diversificării}: @{p.ben}\\% din risc dispare; cu $\\rho = 1$ ar fi zero')]),
    T('Euler contributions $w_i(\\Sigma w)_i/\\sigma_p$ ($(\\Sigma w)_i$: the $i$-th entry of $\\Sigma w$) add up to $\\sigma_p$: BET @{p.c1}\\%, S\\&P 500 @{p.c2}\\% of the portfolio risk', 'Contribuțiile Euler $w_i(\\Sigma w)_i/\\sigma_p$ ($(\\Sigma w)_i$: componenta $i$ a vectorului $\\Sigma w$) au suma $\\sigma_p$: BET @{p.c1}\\%, S\\&P 500 @{p.c2}\\% din riscul portofoliului')), 'footnotesize')

chart(T('Correlation Rises in Crises', 'Corelația crește în crize'), 'sfm_ch10_rolling_corr', 'SFM_ch10_portfolio_copula', [
    T('250-day correlation of BET and S\\&P 500 daily returns; shaded: the 2008 and 2020 stress periods', 'Corelația pe 250 de zile a randamentelor zilnice BET și S\\&P 500; hașurat: perioadele de stres 2008 și 2020'),
    T('Correlation within each period: 2017 (calm) $@{p.corrcalm}$; Sep. 2008 -- Mar. 2009 $@{p.corr2008}$; Feb. -- May 2020 $@{p.corr2020}$',
      'Corelația în fiecare perioadă: 2017 (an liniștit) $@{p.corrcalm}$; sep. 2008 -- mar. 2009 $@{p.corr2008}$; feb. -- mai 2020 $@{p.corr2020}$'),
    T('Maximum of the 250-day correlation: $@{rc.max}$ on @{rc.dmax}', 'Maximul corelației pe 250 de zile: $@{rc.max}$ la @{rc.dmax}'),
    T('Interpretation: diversification is weakest exactly when it is needed: a VaR built on the average correlation is too optimistic in a crisis',
      'Interpretare: diversificarea este cea mai slabă exact cînd este nevoie de ea: un VaR construit pe corelația medie este prea optimist într-o criză')],
    h='0.45\\textheight')

D.frame(T('Correlation Is Not Tail Dependence', 'Corelația nu este dependență în cozi'), items(
    (T('Pearson correlation measures linear co-movement over all days; it is dominated by ordinary days \\refEMS', 'Corelația Pearson măsoară co-mișcarea liniară pe toate zilele; este dominată de zilele obișnuite \\refEMS'),
     [T('rank correlations (Spearman, Kendall\'s $\\tau$) use only the order of the returns: Kendall $\\hat\\tau = @{cop.tau}$, Spearman $@{cop.sp}$', 'corelațiile de rang (Spearman, $\\tau$ al lui Kendall) folosesc doar ordinea randamentelor: Kendall $\\hat\\tau = @{cop.tau}$, Spearman $@{cop.sp}$')]),
    (T('\\textbf{Lower tail dependence}: $\\lambda_L = \\lim_{q \\to 0} P(U_1 \\le q \\mid U_2 \\le q)$, $U_i = F_i(X_i)$ (the returns on a uniform scale)',
       '\\textbf{Dependența în lower tail}: $\\lambda_L = \\lim_{q \\to 0} P(U_1 \\le q \\mid U_2 \\le q)$, $U_i = F_i(X_i)$ (randamentele pe o scală uniformă)'),
     [T('$q$: a small probability; $F_i$: the distribution function of asset $i$; $\\lambda_L \\in [0, 1]$', '$q$: o probabilitate mică; $F_i$: funcția de repartiție a activului $i$; $\\lambda_L \\in [0, 1]$'),
      T('the chance that BET has one of its worst days given that the S\\&P 500 has one of its worst days', 'șansa ca BET să aibă una dintre cele mai proaste zile ale lui, dat fiind că S\\&P 500 are una dintre cele mai proaste zile ale lui'),
      T('data: $q = 5\\%$: $@{p.td5}$; $q = 1\\%$: $@{p.td1}$; under independence both would equal $q$', 'datele: $q = 5\\%$: $@{p.td5}$; $q = 1\\%$: $@{p.td1}$; la independență, ambele ar fi egale cu $q$')]),
    T('Joint crash (both below their own 1\\% quantile): @{p.j.emp}\\% of the days, against @{p.j.ind}\\% under independence', 'Prăbușire simultană (ambele sub propria cuantilă de 1\\%): @{p.j.emp}\\% dintre zile, față de @{p.j.ind}\\% la independență')))

D.frame(T('Copulas: a Light Introduction', 'Copule: o introducere'), items(
    (T('\\textbf{Sklar\'s theorem}: every joint distribution function can be written as $F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$ \\refQRM',
       '\\textbf{Teorema lui Sklar}: orice funcție de repartiție comună se poate scrie $F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$ \\refQRM'),
     [T('$F_1$, $F_2$: the marginal distributions (each asset alone); $C$: the \\textbf{copula}, a joint distribution of uniforms on $[0,1]^2$',
        '$F_1$, $F_2$: distribuțiile marginale (fiecare activ separat); $C$: \\textbf{copula}, o distribuție comună a unor variabile uniforme pe $[0,1]^2$'),
      T('the copula holds all the dependence, the margins all the individual tails', 'copula conține toată dependența, iar marginalele toate cozile individuale')]),
    (T('Practical recipe', 'Rețeta practică'),
     [T('margins: empirical or fitted (Student-t, EVT); dependence: a copula fitted to the \\textbf{pseudo-observations} $\\hat U_{i,t} = \\mathrm{rank}(X_{i,t})/(n + 1)$',
        'marginalele: empirice sau estimate (Student-t, EVT); dependența: o copulă estimată pe \\textbf{pseudo-observațiile} $\\hat U_{i,t} = \\mathrm{rang}(X_{i,t})/(n + 1)$'),
      T('simulate $(U_1, U_2)$ from the copula, map back with $F_i^{-1}$, compute the portfolio return, read VaR and ES', 'simulăm $(U_1, U_2)$ din copulă, revenim la randamente cu $F_i^{-1}$, calculăm randamentul portofoliului și citim VaR și ES')]),
    T('Textbook: \\refFHH, Ch.~17', 'Manual: \\refFHH, cap.~17')))

D.frame(T('Gaussian and t Copulas', 'Copula Gaussiană și copula t'), items(
    (T('\\textbf{Gaussian copula}: the dependence of a bivariate Normal with correlation $\\rho$; $\\lambda_L = 0$ for $\\rho < 1$', '\\textbf{Copula Gaussiană}: dependența unei distribuții Normale bivariate cu corelația $\\rho$; $\\lambda_L = 0$ pentru $\\rho < 1$'),
     [T('extreme days become independent in the limit: joint crashes are rare', 'zilele extreme devin independente la limită: prăbușirile simultane sînt rare')]),
    (T('\\textbf{t copula}: the dependence of a bivariate Student-t with $\\rho$ and $\\nu$ \\refDM', '\\textbf{Copula t}: dependența unei distribuții Student-t bivariate cu $\\rho$ și $\\nu$ \\refDM'),
     [T('$\\lambda_L = 2\\,t_{\\nu+1}\\Big(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}\\Big) > 0$: crashes cluster across assets', '$\\lambda_L = 2\\,t_{\\nu+1}\\Big(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}\\Big) > 0$: prăbușirile apar împreună pe mai multe active'),
      T('$t_{\\nu+1}(\\cdot)$: the distribution function of a Student-t with $\\nu + 1$ degrees of freedom; a smaller $\\nu$ or a larger $\\rho$ gives a larger $\\lambda_L$', '$t_{\\nu+1}(\\cdot)$: funcția de repartiție Student-t cu $\\nu + 1$ grade de libertate; un $\\nu$ mai mic sau un $\\rho$ mai mare dau un $\\lambda_L$ mai mare')]),
    (T('Fit on BET and S\\&P 500: $\\rho = \\sin(\\pi\\hat\\tau/2) = @{cop.rho}$ (valid for both copulas); $\\hat\\nu = @{cop.nu}$ by maximum likelihood',
       'Estimarea pe BET și S\\&P 500: $\\rho = \\sin(\\pi\\hat\\tau/2) = @{cop.rho}$ (valabil pentru ambele copule); $\\hat\\nu = @{cop.nu}$ prin verosimilitate maximă'),
     [T('log-likelihood: Gaussian $@{cop.llg}$, t $@{cop.llt}$; LR $= @{cop.lr}$, far above $\\chi^2_{0.95}(1) = @{chi1}$: the t copula wins', 'log-verosimilitatea: Gaussiană $@{cop.llg}$, t $@{cop.llt}$; LR $= @{cop.lr}$, mult peste $\\chi^2_{0.95}(1) = @{chi1}$: copula t este mai bună'),
      T('implied tail dependence $\\lambda_L = @{cop.lam}$, against 0 for the Gaussian copula', 'dependența în coadă implicată $\\lambda_L = @{cop.lam}$, față de 0 pentru copula Gaussiană')])), 'footnotesize')

chart(T('Joint Bad Days: Data and Two Copulas', 'Zile proaste comune: datele și două copule'), 'sfm_ch10_copula', 'SFM_ch10_portfolio_copula', [
    T('Lower-left corner of the uniform scale: pseudo-observations and samples of the same size (@{p.n}) from the fitted copulas', 'Colțul din stînga jos al scalei uniforme: pseudo-observațiile și eșantioane de aceeași mărime (@{p.n}) din copulele estimate'),
    T('Days on which both indices are among their worst 5\\%: data @{cf.corner_data}, Gaussian @{cf.corner_gauss}, t @{cf.corner_t}; independence: about @{cf.exp}',
      'Zile în care ambii indici sînt printre cele mai proaste 5\\% ale lor: datele @{cf.corner_data}, copula Gaussiană @{cf.corner_gauss}, copula t @{cf.corner_t}; independența: circa @{cf.exp}'),
    T('Interpretation: the Gaussian copula halves the number of joint bad days; the t copula is much closer to the data', 'Interpretare: copula Gaussiană înjumătățește numărul zilelor proaste comune; copula t este mult mai aproape de date')],
    h='0.48\\textheight')

D.frame(T('Copula VaR: the Simulation Step by Step', 'VaR prin copule: simularea pas cu pas'), items(
    (T('\\textbf{1.} Draw $(Z_1, Z_2)$ from a bivariate Normal with correlation $\\rho$', '\\textbf{1.} Extragem $(Z_1, Z_2)$ dintr-o distribuție Normală bivariată cu corelația $\\rho$'),
     [T('Gaussian copula: $U_i = \\Phi(Z_i)$', 'copula Gaussiană: $U_i = \\Phi(Z_i)$'),
      T('t copula: draw $W \\sim \\chi^2_\\nu$, set $U_i = t_\\nu(Z_i\\sqrt{\\nu/W})$: the common factor $\\sqrt{\\nu/W}$ makes extremes happen together',
        'copula t: extragem $W \\sim \\chi^2_\\nu$ și luăm $U_i = t_\\nu(Z_i\\sqrt{\\nu/W})$: factorul comun $\\sqrt{\\nu/W}$ face ca extremele să apară împreună')]),
    T('\\textbf{2.} Map to returns with the empirical quantile functions: $R_i = \\hat F_i^{-1}(U_i)$', '\\textbf{2.} Revenim la randamente cu funcțiile cuantilă empirice: $R_i = \\hat F_i^{-1}(U_i)$'),
    T('\\textbf{3.} Portfolio return $R_p = 0.5R_1 + 0.5R_2$; repeat 200\\,000 times', '\\textbf{3.} Randamentul portofoliului $R_p = 0{,}5R_1 + 0{,}5R_2$; repetăm de 200\\,000 de ori'),
    T('\\textbf{4.} VaR 1\\% and ES 2.5\\% by historical simulation on the simulated $R_p$', '\\textbf{4.} VaR 1\\% și ES 2,5\\% prin simulare istorică pe valorile simulate ale lui $R_p$'),
    T('Same margins in both copulas: any difference in VaR comes from the dependence alone', 'Aceleași marginale în ambele copule: orice diferență de VaR vine numai din dependență')) + ql('SFM_ch10_portfolio_copula'))

D.frame(T('Portfolio VaR 1\\% and ES 2.5\\% by Method', 'VaR 1\\% și ES 2,5\\% ale portofoliului, pe metode'), table(
    'lrrr', T('Method & VaR 1\\% & ES 2.5\\% & P(joint crash at 1\\%)', 'Metoda & VaR 1\\% & ES 2,5\\% & P(prăbușire simultană la 1\\%)'),
    [T('variance--covariance (Normal)', 'varianță--covarianță (Normală)') + ' & $@{p.var_n}$ & $@{p.es_n}$ & --',
     T('Gaussian copula, empirical margins', 'copula Gaussiană, marginale empirice') + ' & $@{p.var_gauss}$ & $@{p.es_gauss}$ & @{p.j.gauss}\\%',
     T('t copula, empirical margins', 'copula t, marginale empirice') + ' & $@{p.var_t}$ & $@{p.es_t}$ & @{p.j.t}\\%',
     T('historical simulation (the data)', 'simulare istorică (datele)') + ' & $@{p.var_hs}$ & $@{p.es_hs}$ & @{p.j.emp}\\%'],
    size='footnotesize') + items(
    T('In \\%, 50/50 portfolio, @{p.n} common days; copulas: 200\\,000 simulated days', 'În \\%, portofoliul 50/50, @{p.n} zile comune; copule: 200\\,000 de zile simulate'),
    T('The Normal method misses both the heavy margins and the tail dependence; the copulas fix the margins, the t copula also the joint crashes', 'Metoda Normală ignoră atît marginalele cu cozi groase, cît și dependența în cozi; copulele corectează marginalele, iar copula t și prăbușirile simultane'),
    T('Interpretation: the t copula still understates the historical VaR: the BET--S\\&P 500 link is stronger in crises than any constant copula can describe',
      'Interpretare: copula t subestimează în continuare VaR istoric: legătura BET--S\\&P 500 este mai puternică în crize decît poate descrie o copulă constantă')) + ql('SFM_ch10_portfolio_copula'), 'footnotesize')

# =============================================================================
# 7. BACKTESTING
# =============================================================================
D.section('Backtesting', 'Backtesting')

D.frame(T('The Idea: the Hit Sequence', 'Ideea: secvența depășirilor'), items(
    (T('\\textbf{Exception} (hit): a day with $r_t < -\\mathrm{VaR}_t$; $I_t = \\mathbf{1}\\{r_t < -\\mathrm{VaR}_t\\}$', '\\textbf{Depășire} (excepție): o zi cu $r_t < -\\mathrm{VaR}_t$; $I_t = \\mathbf{1}\\{r_t < -\\mathrm{VaR}_t\\}$'),
     [T('the VaR must be the forecast made the day before: no look-ahead', 'VaR trebuie să fie prognoza făcută cu o zi înainte: fără informații din viitor')]),
    (T('A correct VaR at level $\\alpha$: the $I_t$ are i.i.d.\\ Bernoulli($\\alpha$) \\refChris', 'Un VaR corect la nivelul $\\alpha$: $I_t$ sînt i.i.d.\\ Bernoulli($\\alpha$) \\refChris'),
     [T('\\textbf{unconditional coverage}: $P(I_t = 1) = \\alpha$: the right number of exceptions', '\\textbf{acoperire necondiționată}: $P(I_t = 1) = \\alpha$: numărul corect de depășiri'),
      T('\\textbf{independence}: an exception today does not make one tomorrow more likely', '\\textbf{independență}: o depășire azi nu face mai probabilă o depășire mîine')]),
    T('Number of exceptions in $n$ days: $x = \\sum_t I_t \\sim \\mathrm{Binomial}(n, \\alpha)$, mean $n\\alpha$, variance $n\\alpha(1 - \\alpha)$', 'Numărul de depășiri în $n$ zile: $x = \\sum_t I_t \\sim \\mathrm{Binomial}(n, \\alpha)$, media $n\\alpha$, varianța $n\\alpha(1 - \\alpha)$')))

chart(T('Exceptions in 250 Days under a Correct VaR 1\\%', 'Depășiri în 250 de zile pentru un VaR 1\\% corect'), 'sfm_ch10_binomial', 'SFM_ch10_backtesting', [
    T('$\\mathrm{Binomial}(250, 0.01)$: expected 2.5 exceptions; 0 exceptions with probability @{tl.0.p}\\%, 5 or more with probability $100\\% - @{tl.4.c}\\%$, about 11\\%',
      '$\\mathrm{Binomial}(250; 0{,}01)$: 2,5 depășiri așteptate; 0 depășiri cu probabilitatea @{tl.0.p}\\%, 5 sau mai multe cu probabilitatea $100\\% - @{tl.4.c}\\%$, adică circa 11\\%'),
    T('Interpretation: one year of data separates a good model from a bad one only roughly: a correct model lands in the yellow zone about one year in nine',
      'Interpretare: un an de date separă doar aproximativ un model bun de unul prost: un model corect ajunge în zona galbenă cam un an din nouă')],
    h='0.48\\textheight')

D.frame(T('Kupiec\'s POF Test', 'Testul POF al lui Kupiec'), items(
    (T('\\textbf{POF} (proportion of failures) \\refKupiec: $H_0$: $P(I_t = 1) = \\alpha$; $\\hat\\pi = x/n$', '\\textbf{POF} (proportion of failures, proporția depășirilor) \\refKupiec: $H_0$: $P(I_t = 1) = \\alpha$; $\\hat\\pi = x/n$'),
     [T('$LR_{uc} = -2\\big[(n - x)\\ln(1 - \\alpha) + x\\ln\\alpha - (n - x)\\ln(1 - \\hat\\pi) - x\\ln\\hat\\pi\\big] \\sim \\chi^2(1)$ under $H_0$',
        '$LR_{uc} = -2\\big[(n - x)\\ln(1 - \\alpha) + x\\ln\\alpha - (n - x)\\ln(1 - \\hat\\pi) - x\\ln\\hat\\pi\\big] \\sim \\chi^2(1)$ în ipoteza $H_0$'),
      T('$x$: exceptions in $n$ days; $\\hat\\pi$: the observed rate; $LR_{uc}$ compares the likelihood under the rate $\\alpha$ with that under $\\hat\\pi$', '$x$: depășirile în $n$ zile; $\\hat\\pi$: rata observată; $LR_{uc}$ compară verosimilitatea cu rata $\\alpha$ cu cea cu rata $\\hat\\pi$'),
      T('$LR_{uc} \\ge 0$; a likelihood-ratio test (Chapter 6); reject at 5\\% if $LR_{uc} > @{chi1}$', '$LR_{uc} \\ge 0$; un test al raportului de verosimilitate (Capitolul 6); respingem la 5\\% dacă $LR_{uc} > @{chi1}$')]),
    (T('\\textbf{Step by step}: $n = 250$, $x = 7$, $\\hat\\pi = 0.028$', '\\textbf{Pas cu pas}: $n = 250$, $x = 7$, $\\hat\\pi = 0{,}028$'),
     [T('$\\ln L_0 = 243\\ln 0.99 + 7\\ln 0.01 = @{k7.l0}$; $\\ln L_1 = 243\\ln 0.972 + 7\\ln 0.028 = @{k7.l1}$', '$\\ln L_0 = 243\\ln 0.99 + 7\\ln 0.01 = @{k7.l0}$; $\\ln L_1 = 243\\ln 0.972 + 7\\ln 0.028 = @{k7.l1}$'),
      T('$LR_{uc} = -2(@{k7.l0} - (@{k7.l1})) = @{k7.lr}$, p $= @{k7.p}$: rejected at 5\\%', '$LR_{uc} = -2(@{k7.l0} - (@{k7.l1})) = @{k7.lr}$, p $= @{k7.p}$: respins la 5\\%')]),
    (T('Low power with one year of data', 'Putere mică cu un an de date'),
     [T('in 250 days, any count from @{k.acc.lo} to @{k.acc.hi} is accepted at 5\\%: a model with a true rate of 2\\% passes with probability @{k.p2}\\%', 'în 250 de zile, orice număr între @{k.acc.lo} și @{k.acc.hi} este acceptat la 5\\%: un model cu o rată reală de 2\\% trece testul cu probabilitatea @{k.p2}\\%'),
      T('too few exceptions are also rejected: an overly prudent VaR wastes capital', 'și prea puține depășiri duc la respingere: un VaR exagerat de prudent irosește capital')])), 'footnotesize')

D.frame(T('Kupiec on 21 Years of S\\&P 500 Forecasts', 'Testul Kupiec pe 21 de ani de prognoze pentru S\\&P 500'), items(
    (T('GARCH-t VaR 1\\%, @{bt.sp500.start} -- @{end}: $n = @{bt.sp500.g.n}$ days, $x = @{bt.sp500.g.x}$ exceptions, $\\hat\\pi = @{kg.pi}$', 'VaR 1\\% GARCH-t, @{bt.sp500.start} -- @{end}: $n = @{bt.sp500.g.n}$ de zile, $x = @{bt.sp500.g.x}$ de depășiri, $\\hat\\pi = @{kg.pi}$'),
     [T('expected $n\\alpha = @{bt.sp500.g.exp}$; a 95\\% binomial range for the rate under $H_0$: @{kg.ci.lo}--@{kg.ci.hi}\\%', 'așteptat $n\\alpha = @{bt.sp500.g.exp}$; un interval binomial de 95\\% pentru rată sub $H_0$: @{kg.ci.lo}--@{kg.ci.hi}\\%')]),
    (T('\\textbf{Step by step}', '\\textbf{Pas cu pas}'),
     [T('$\\ln L_0 = @{kg.nx}\\ln 0.99 + @{bt.sp500.g.x}\\ln 0.01 = @{kg.l0}$', '$\\ln L_0 = @{kg.nx}\\ln 0.99 + @{bt.sp500.g.x}\\ln 0.01 = @{kg.l0}$'),
      T('$\\ln L_1 = @{kg.nx}\\ln(1 - @{kg.pi}) + @{bt.sp500.g.x}\\ln @{kg.pi} = @{kg.l1}$', '$\\ln L_1 = @{kg.nx}\\ln(1 - @{kg.pi}) + @{bt.sp500.g.x}\\ln @{kg.pi} = @{kg.l1}$'),
      T('$LR_{uc} = -2(\\ln L_0 - \\ln L_1) = @{bt.sp500.g.luc}$, p @{bt.sp500.g.puc}', '$LR_{uc} = -2(\\ln L_0 - \\ln L_1) = @{bt.sp500.g.luc}$, p @{bt.sp500.g.puc}')]),
    T('Interpretation: with 21 years of data even a rate of @{bt.sp500.g.rate}\\% instead of 1\\% is clearly rejected; the missing leverage effect (Chapter 9) is a likely cause',
      'Interpretare: cu 21 de ani de date, chiar și o rată de @{bt.sp500.g.rate}\\% în loc de 1\\% este respinsă clar; o cauză probabilă este lipsa efectului de levier (Capitolul 9)')) + ql('SFM_ch10_backtesting'), 'footnotesize')

D.frame(T('Christoffersen\'s Independence Test', 'Testul de independență al lui Christoffersen'), items(
    (T('Count the transitions: $n_{ij}$ = number of days with $I_{t-1} = i$ and $I_t = j$; $\\hat\\pi_{01} = \\dfrac{n_{01}}{n_{00} + n_{01}}$, $\\hat\\pi_{11} = \\dfrac{n_{11}}{n_{10} + n_{11}}$',
       'Numărăm tranzițiile: $n_{ij}$ = numărul zilelor cu $I_{t-1} = i$ și $I_t = j$; $\\hat\\pi_{01} = \\dfrac{n_{01}}{n_{00} + n_{01}}$, $\\hat\\pi_{11} = \\dfrac{n_{11}}{n_{10} + n_{11}}$'),
     [T('$\\hat\\pi_{01}$: the rate of exceptions after a day without one; $\\hat\\pi_{11}$: after a day with one', '$\\hat\\pi_{01}$: rata depășirilor după o zi fără depășire; $\\hat\\pi_{11}$: după o zi cu depășire'),
      T('$H_0$: $\\pi_{01} = \\pi_{11}$ (an exception today does not change the chance of one tomorrow) \\refChris', '$H_0$: $\\pi_{01} = \\pi_{11}$ (o depășire azi nu schimbă șansa unei depășiri mîine) \\refChris'),
      T('$LR_{ind} = -2\\ln\\dfrac{(1 - \\hat\\pi)^{n_{00} + n_{10}}\\hat\\pi^{n_{01} + n_{11}}}{(1 - \\hat\\pi_{01})^{n_{00}}\\hat\\pi_{01}^{n_{01}}(1 - \\hat\\pi_{11})^{n_{10}}\\hat\\pi_{11}^{n_{11}}} \\sim \\chi^2(1)$',
        '$LR_{ind} = -2\\ln\\dfrac{(1 - \\hat\\pi)^{n_{00} + n_{10}}\\hat\\pi^{n_{01} + n_{11}}}{(1 - \\hat\\pi_{01})^{n_{00}}\\hat\\pi_{01}^{n_{01}}(1 - \\hat\\pi_{11})^{n_{10}}\\hat\\pi_{11}^{n_{11}}} \\sim \\chi^2(1)$')]),
    T('\\textbf{Conditional coverage}: $LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)$; reject at 5\\% if $LR_{cc} > @{chi2}$', '\\textbf{Acoperire condiționată}: $LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)$; respingem la 5\\% dacă $LR_{cc} > @{chi2}$'),
    (T('\\textbf{Example}: S\\&P 500, HS VaR 1\\%, @{bt.sp500.hs.n} days: $n_{00} = @{ch.n00}$, $n_{01} = @{ch.n01}$, $n_{10} = @{ch.n10}$, $n_{11} = @{ch.n11}$',
       '\\textbf{Exemplu}: S\\&P 500, VaR 1\\% HS, @{bt.sp500.hs.n} zile: $n_{00} = @{ch.n00}$, $n_{01} = @{ch.n01}$, $n_{10} = @{ch.n10}$, $n_{11} = @{ch.n11}$'),
     [T('$\\hat\\pi_{01} = @{bt.sp500.hs.p01}\\%$, $\\hat\\pi_{11} = @{bt.sp500.hs.p11}\\%$: after an exception the next one is @{ch.ratio} times more likely',
        '$\\hat\\pi_{01} = @{bt.sp500.hs.p01}\\%$, $\\hat\\pi_{11} = @{bt.sp500.hs.p11}\\%$: după o depășire, următoarea este de @{ch.ratio} ori mai probabilă'),
      T('$LR_{ind} = @{bt.sp500.hs.lind}$ (p @{bt.sp500.hs.pind}): the exceptions cluster', '$LR_{ind} = @{bt.sp500.hs.lind}$ (p @{bt.sp500.hs.pind}): depășirile apar grupat')])), 'footnotesize')


def trow(x):
    zone = T('green', 'verde') if x <= 4 else (T('yellow', 'galbenă') if x <= 9 else T('red', 'roșie'))
    plus = {5: '0.40', 6: '0.50', 7: '0.65', 8: '0.75', 9: '0.85'}.get(x, '0.00' if x <= 4 else '1.00')
    lab = str(x) if x < 10 else '$\\ge 10$'
    return f'{lab} & {zone} & $@{{tl.{x}.c}}$ & ${plus}$ & ${3 + float(plus):.2f}$'


D.frame(T('The Basel Traffic Light', 'Semaforul Basel'), cols(table(
    'lcrrr', T('$x$ & Zone & Cum.\\ (\\%) & Plus & Factor', '$x$ & Zona & Cumul.\\ (\\%) & Adaos & Factor'),
    [trow(x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], size='scriptsize'), items(
    (T('\\refBCBSb: count the exceptions of the one-day VaR 1\\% over the last 250 days', '\\refBCBSb: se numără depășirile VaR 1\\% pe o zi în ultimele 250 de zile'),
     [T('green: no action; yellow: the multiplier grows by the plus factor; red: the model is presumed wrong', 'verde: nicio măsură; galbenă: factorul crește cu adaosul; roșie: modelul este considerat greșit')]),
    (T('Capital $= (3 + \\text{plus}) \\times$ the 60-day average of the 10-day VaR 1\\% (or yesterday\'s value, if larger)', 'Capitalul $= (3 + \\text{adaos}) \\times$ media pe 60 de zile a VaR 1\\% pe 10 zile (sau valoarea de ieri, dacă este mai mare)'),
     [T('zones from the binomial: yellow starts where the cumulative probability first exceeds 95\\%, red where it reaches 99.99\\%', 'zonele vin din distribuția binomială: zona galbenă începe unde probabilitatea cumulată trece de 95\\%, zona roșie unde atinge 99,99\\%')]),
    T('$x$: exceptions in 250 days; Cumul.: $P(X \\le x)$ under $\\mathrm{Binomial}(250, 0.01)$', '$x$: depășiri în 250 de zile; Cumul.: $P(X \\le x)$ pentru $\\mathrm{Binomial}(250; 0{,}01)$'),
    T('FRTB keeps the count of VaR 1\\% exceptions at bank and desk level \\refMARb', 'FRTB păstrează numărarea depășirilor VaR 1\\% la nivelul băncii și al diviziilor \\refMARb')), wl='0.44', wr='0.54'), 'footnotesize')

chart(T('When Do the Exceptions Happen?', 'Depășirile în timp'), 'sfm_ch10_hits', 'SFM_ch10_backtesting', [
    T('S\\&P 500, 2005--2026: each mark is a day on which the VaR 1\\% of the method was exceeded; expected about @{bt.sp500.hs.exp} per method',
      'S\\&P 500, 2005--2026: fiecare marcaj este o zi în care VaR 1\\% al metodei a fost depășit; aproximativ @{bt.sp500.hs.exp} așteptate pentru fiecare metodă'),
    T('Interpretation: HS and Normal exceptions come in bursts (2007--2008, 2011, 2022); GARCH-t and FHS exceptions are spread over time, but there are more of them in calm years',
      'Interpretare: depășirile HS și Normale apar grupat (2007--2008, 2011, 2022); depășirile GARCH-t și FHS sînt răspîndite în timp, dar sînt mai multe în anii liniștiți')],
    h='0.48\\textheight')


def brow(k, m):
    key = f'bt.{k}.{MKEY[m]}'
    return (f'{SHORT[k] if m == "HS" else ""} & {m} & @{{{key}.x}} & $@{{{key}.rate}}$ & @{{{key}.puc}} & @{{{key}.pind}} & @{{{key}.pcc}} & @{{{key}.max}} & '
            f'${{@{{{key}.z2}}}}$ (@{{{key}.pz2}})')


BH = T('& Method & $x$ & rate (\\%) & p POF & p ind. & p c.c. & max 250 d. & $Z_2$ (p)', '& Metoda & $x$ & rata (\\%) & p POF & p ind. & p ac.c. & max. 250 z. & $Z_2$ (p)')
for pair, en, ro in [(['sp500', 'dax'], 'S\\&P 500 and DAX', 'S\\&P 500 și DAX'), (['bet', 'btc'], 'BET and Bitcoin', 'BET și Bitcoin')]:
    rows = []
    for k in pair:
        rows += [brow(k, m) for m in METHODS]
        if k != pair[-1]:
            rows.append('\\midrule')
    tab = table('llrrrrrrr', BH, rows, size='scriptsize').replace('\\midrule \\\\', '\\midrule')
    D.frame(T(f'Backtest Results: {en}', f'Rezultatele backtesting-ului: {ro}'), tab + items(
        (T('VaR 1\\% since @{bt.' + pair[0] + '.start} (Bitcoin since @{bt.btc.start}); the columns as on the previous slide',
           'VaR 1\\% din @{bt.' + pair[0] + '.start} (Bitcoin din @{bt.btc.start}); coloanele ca pe slide-ul precedent') if 'btc' in pair else
         T('VaR 1\\% since @{bt.' + pair[0] + '.start} (about 5\\,500 days); $x$: exceptions; p ind.: Christoffersen independence; p c.c.: conditional coverage; max 250 d.: the worst 250-day count; $Z_2$: the ES backtest at the end of this section',
           'VaR 1\\% din @{bt.' + pair[0] + '.start} (circa 5\\,500 de zile); $x$: depășirile; p ind.: independența Christoffersen; p ac.c.: acoperirea condiționată; max. 250 z.: cel mai mare număr pe 250 de zile; $Z_2$: backtesting-ul ES de la sfîrșitul acestei secțiuni')),
        (T('Normal: rejected everywhere; HS: exceptions cluster; GARCH-t and FHS: independent exceptions, but a rate of @{rg.g.lo}--@{rg.g.hi}\\%', 'Distribuția Normală: respinsă peste tot; HS: depășirile apar grupat; GARCH-t și FHS: depășiri independente, dar o rată de @{rg.g.lo}--@{rg.g.hi}\\%')
         if 'sp500' in pair else
         T('BET: GARCH-t and FHS pass all three tests; Bitcoin: HS passes, FHS is too prudent ($@{bt.btc.f.rate}\\%$)', 'BET: GARCH-t și FHS trec toate cele trei teste; Bitcoin: HS trece testele, FHS este prea prudent ($@{bt.btc.f.rate}\\%$)'))) + ql('SFM_ch10_backtesting'), 'footnotesize')



def qrow(k):
    return f'{SHORT[k]} & ' + ' & '.join(f'$@{{bt.{k}.{MKEY[m]}.ql}}$' for m in METHODS) + ' & ' + ' & '.join(f'@{{bt.{k}.{MKEY[m]}.l250}}' for m in METHODS)


D.frame(T('Ranking the Forecasts, and Today\'s Zone', 'Ordonarea prognozelor și zona de azi'), table(
    'l|rrrr|rrrr', T('& \\multicolumn{4}{c|}{average quantile loss $\\times 100$} & \\multicolumn{4}{c}{exceptions, last 250 days} \\\\ & HS & Normal & GARCH-t & FHS & HS & Normal & GARCH-t & FHS',
                     '& \\multicolumn{4}{c|}{pierderea cuantilă medie $\\times 100$} & \\multicolumn{4}{c}{depășiri, ultimele 250 de zile} \\\\ & HS & Normală & GARCH-t & FHS & HS & Normală & GARCH-t & FHS'),
    [qrow(k) for k in BT], size='footnotesize') + items(
    (T('Tests say whether a model is acceptable; a \\textbf{scoring function} says which of two models is better', 'Testele spun dacă un model este acceptabil; o \\textbf{funcție de scor} spune care dintre două modele este mai bun'),
     [T('VaR is elicitable: the average quantile loss $\\frac1n\\sum_t(\\alpha - I_t)(r_t + \\mathrm{VaR}_t)$ ranks the forecasts \\refGneiting', 'VaR este elicitabil: media pierderii cuantile $\\frac1n\\sum_t(\\alpha - I_t)(r_t + \\mathrm{VaR}_t)$ ordonează prognozele \\refGneiting')]),
    T('The equity indices: GARCH-t and FHS have the smallest loss, the Normal the largest; Bitcoin: the four methods are close', 'Indicii de acțiuni: GARCH-t și FHS au pierderea cea mai mică, metoda Normală cea mai mare; Bitcoin: cele patru metode sînt apropiate'),
    T('Interpretation: on the last 250 days every model is in the green zone except the Normal VaR of the BET (@{bt.bet.n.l250}, yellow): a recent window says little about a model',
      'Interpretare: în ultimele 250 de zile, toate modelele sînt în zona verde, cu excepția VaR Normal pentru BET (@{bt.bet.n.l250}, zona galbenă): o fereastră recentă spune puțin despre un model')) + ql('SFM_ch10_backtesting'), 'footnotesize')

chart(T('The Traffic Light through Time', 'Semaforul Basel în timp'), 'sfm_ch10_traffic_time', 'SFM_ch10_backtesting', [
    T('S\\&P 500: number of VaR 1\\% exceptions in the last 250 days, for every day since 2006; dashed: the zone limits', 'S\\&P 500: numărul depășirilor VaR 1\\% în ultimele 250 de zile, pentru fiecare zi din 2006; linii întrerupte: limitele zonelor'),
    T('Share of days in the red zone: Normal @{bt.sp500.n.red}\\%, HS @{bt.sp500.hs.red}\\%, GARCH-t @{bt.sp500.g.red}\\%; worst count: @{bt.sp500.n.max}, @{bt.sp500.hs.max} and @{bt.sp500.g.max}',
      'Ponderea zilelor în zona roșie: Normală @{bt.sp500.n.red}\\%, HS @{bt.sp500.hs.red}\\%, GARCH-t @{bt.sp500.g.red}\\%; cel mai mare număr: @{bt.sp500.n.max}, @{bt.sp500.hs.max} și @{bt.sp500.g.max}'),
    T('Interpretation: a bank using the Normal or HS VaR would have paid the red-zone penalty for most of 2008--2009; GARCH-t almost never reaches the red zone',
      'Interpretare: o bancă ce ar fi folosit VaR Normal sau HS ar fi plătit penalizarea zonei roșii în cea mai mare parte din 2008--2009; GARCH-t aproape că nu ajunge niciodată în zona roșie')],
    h='0.48\\textheight')

D.frame(T('Backtesting ES: Acerbi and Székely', 'Backtesting pentru ES: Acerbi și Székely'), items(
    (T('ES is not elicitable alone, but it can be tested together with VaR \\refAS', 'ES nu este elicitabil singur, dar poate fi testat împreună cu VaR \\refAS'),
     [T('$Z_2 = \\dfrac{1}{T\\alpha}\\displaystyle\\sum_{t=1}^{T}\\dfrac{r_t I_t}{\\mathrm{ES}_t} + 1$, \\quad $I_t = \\mathbf{1}\\{r_t < -\\mathrm{VaR}_t\\}$, $\\alpha = 2.5\\%$',
        '$Z_2 = \\dfrac{1}{T\\alpha}\\displaystyle\\sum_{t=1}^{T}\\dfrac{r_t I_t}{\\mathrm{ES}_t} + 1$, \\quad $I_t = \\mathbf{1}\\{r_t < -\\mathrm{VaR}_t\\}$, $\\alpha = 2{,}5\\%$'),
      T('$T$: the number of test days; $\\mathrm{VaR}_t$, $\\mathrm{ES}_t$: the forecasts for day $t$; each exception adds $r_t/\\mathrm{ES}_t$', '$T$: numărul zilelor de test; $\\mathrm{VaR}_t$, $\\mathrm{ES}_t$: prognozele pentru ziua $t$; fiecare depășire adaugă $r_t/\\mathrm{ES}_t$'),
      T('if VaR and ES are right, $E[Z_2] = 0$; $Z_2 < 0$: the losses beyond VaR are larger than the forecast ES, or too frequent', 'dacă VaR și ES sînt corecte, $E[Z_2] = 0$; $Z_2 < 0$: pierderile de dincolo de VaR sînt mai mari decît ES prognozat sau prea dese')]),
    (T('p-value by simulation: draw returns from each day\'s forecast distribution, recompute $Z_2$ 1000 times', 'p-value-ul prin simulare: extragem randamente din distribuția prognozată a fiecărei zile și recalculăm $Z_2$ de 1000 de ori'),
     [T('the values of $Z_2$ and their p-values: the last column of the two backtest tables', 'valorile $Z_2$ și p-value-urile lor: ultima coloană din cele două tabele de backtesting')]),
    T('Interpretation: all four ES 2.5\\% forecasts are too small for the equity indices; FHS is closest, because it takes the tail of $z$ from the data',
      'Interpretare: toate cele patru prognoze ES 2,5\\% sînt prea mici pentru indicii de acțiuni; FHS este cel mai aproape, deoarece preia coada lui $z$ din date')) + ql('SFM_ch10_backtesting'), 'footnotesize')

D.frame(T('Case Study: VaR Models at Commercial Banks', 'Studiu de caz: modelele VaR ale băncilor comerciale'), items(
    (T('\\refBO: the first study of the VaR forecasts that large US banks actually reported, with their daily trading P\\&L',
       '\\refBO: primul studiu al prognozelor VaR raportate efectiv de mari bănci americane, comparate cu P\\&L-ul zilnic din tranzacționare'),
     [T('design: the banks\' VaR 1\\% against the realised P\\&L; a simple GARCH model of each bank\'s P\\&L as the benchmark', 'schema studiului: VaR 1\\% al băncilor față de P\\&L-ul realizat; un model GARCH simplu al P\\&L-ului fiecărei bănci ca reper')]),
    (T('Findings', 'Rezultatele'),
     [T('the banks\' VaR was conservative on average, yet the exceptions that did occur came in clusters', 'VaR-ul băncilor a fost în medie prudent, dar depășirile care au avut loc au apărut grupat'),
      T('the GARCH benchmark gave lower VaRs with comparable coverage: it followed changes in volatility better', 'reperul GARCH a dat valori VaR mai mici, cu o acoperire comparabilă: a urmărit mai bine schimbările volatilității')]),
    T('The same pattern as in our data: window-based VaR is either too high in calm times or too slow in storms; conditional VaR adapts',
      'Același tipar ca în datele noastre: VaR-ul pe fereastră este fie prea mare în perioadele liniștite, fie prea lent în perioadele turbulente; VaR condiționat se adaptează'),
    T('Further reading on forecasting the quantile directly: CAViaR \\refCAViaR', 'Lectură suplimentară despre prognoza directă a cuantilei: CAViaR \\refCAViaR')))

# =============================================================================
# 8. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Does the tail dependence between the BVB and world markets jump in crises, and does it come back?}', '\\textbf{Crește brusc dependența în cozi dintre BVB și piețele mondiale în crize și revine apoi la nivelul anterior?}'),
     [T('today: correlation $@{p.corrcalm}$ in a calm year, $@{p.corr2020}$ in spring 2020; a constant t copula gives $\\lambda_L = @{cop.lam}$', 'azi: corelația $@{p.corrcalm}$ într-un an liniștit, $@{p.corr2020}$ în primăvara lui 2020; o copulă t constantă dă $\\lambda_L = @{cop.lam}$'),
      T('a portfolio VaR with a constant copula is still below the historical one (Section 6)', 'un VaR de portofoliu cu o copulă constantă rămîne sub cel istoric (secțiunea 6)')]),
    T('Why it is open: crises are few, the BVB trades at other hours, and tail estimates need many observations', 'Întrebarea rămîne deschisă: crizele sînt puține, BVB se tranzacționează la alte ore, iar estimările din coadă au nevoie de multe observații'),
    T('Related work on entropy as a risk measure, as further reading: \\refPeleA; \\refPeleB', 'Lucrări înrudite despre entropie ca măsură de risc, ca lectură suplimentară: \\refPeleA; \\refPeleB'),
    T('AI tools can speed up such a study, but its results must still be checked \\refWang', 'Instrumentele AI pot accelera un astfel de studiu, dar rezultatele lui trebuie verificate \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on time-varying copulas and on contagion between emerging and developed markets', '\\textbf{Literatura}: o listă a studiilor despre copule variabile în timp și despre contagiunea dintre piețele emergente și cele dezvoltate'),
    T('\\textbf{Code}: a first draft of rolling t-copula estimates (250 days) for BET and S\\&P 500, with $\\rho$, $\\nu$ and $\\lambda_L$ over time', '\\textbf{Cod}: o primă versiune a estimărilor copulei t pe ferestre mobile (250 de zile) pentru BET și S\\&P 500, cu $\\rho$, $\\nu$ și $\\lambda_L$ în timp'),
    T('\\textbf{Robustness}: two-day returns (different trading hours), DAX instead of S\\&P 500, other windows', '\\textbf{Robustețe}: randamente pe două zile (ore de tranzacționare diferite), DAX în loc de S\\&P 500, alte ferestre'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that fits a bivariate t copula by maximum likelihood to the pseudo-observations of daily BET and S\\&P 500 returns on rolling windows of 250 days moved by 21 days, and plots rho, nu and the lower tail dependence over time.}',
        '\\aiprompt{Write Python code that fits a bivariate t copula by maximum likelihood to the pseudo-observations of daily BET and S\\&P 500 returns on rolling windows of 250 days moved by 21 days, and plots rho, nu and the lower tail dependence over time.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The VaR convention: ``VaR 1\\%\'\' is the loss exceeded with probability 1\\%, a positive number; an answer that names the complementary probability or reports a negative VaR mixes conventions',
      'Convenția VaR: „VaR 1\\%” este pierderea depășită cu probabilitatea 1\\%, un număr pozitiv; un răspuns care folosește probabilitatea complementară sau raportează un VaR negativ amestecă convențiile'),
    T('Prices joined on common days before computing returns; the BVB and New York trade at different hours', 'Prețurile se aliniază pe zilele comune înainte de calculul randamentelor; BVB și New York se tranzacționează la ore diferite'),
    T('Pseudo-observations: ranks divided by $n + 1$, not by $n$ (otherwise the copula density is infinite at 1)', 'Pseudo-observațiile: rangurile împărțite la $n + 1$, nu la $n$ (altfel densitatea copulei este infinită în 1)'),
    T('Tail dependence from 250 days rests on a handful of joint bad days: report its uncertainty (bootstrap, Chapter 2)', 'Dependența în cozi din 250 de zile se bazează pe cîteva zile proaste comune: raportați incertitudinea ei (bootstrap, Capitolul 2)'),
    T('Overlapping windows: consecutive estimates are not independent evidence', 'Ferestrele suprapuse: estimările consecutive nu sînt dovezi independente'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: does a time-varying copula give a better portfolio VaR for a BVB and world-equity portfolio than a constant one?',
       '\\textbf{Întrebarea}: dă o copulă variabilă în timp un VaR de portofoliu mai bun decît o copulă constantă, pentru un portofoliu BVB și acțiuni mondiale?'),
     [T('data: BET, BET-TR, S\\&P 500, DAX since 2005 (EODHD)', 'date: BET, BET-TR, S\\&P 500, DAX din 2005 (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('GARCH-t margins for each index (Chapter 9); constant and rolling t copula for the dependence', 'marginale GARCH-t pentru fiecare indice (Capitolul 9); copulă t constantă și pe fereastră mobilă pentru dependență'),
      T('one-day VaR 1\\% and ES 2.5\\% of a 50/50 portfolio, out of sample since 2008', 'VaR 1\\% și ES 2,5\\% pe o zi pentru un portofoliu 50/50, în afara eșantionului, din 2008'),
      T('Kupiec, Christoffersen and $Z_2$ for both versions, with special attention to 2008, 2020 and 2022', 'testele Kupiec, Christoffersen și $Z_2$ pentru ambele variante, cu atenție specială pentru 2008, 2020 și 2022')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('VaR 1\\% $= -q_{1\\%}$: the loss exceeded on one day in a hundred; ES 2.5\\%: the average loss on the worst 2.5\\% of days', 'VaR 1\\% $= -q_{1\\%}$: pierderea depășită într-o zi din o sută; ES 2,5\\%: pierderea medie din cele mai proaste 2,5\\% dintre zile'),
    T('ES is coherent; VaR can penalise diversification; VaR is elicitable, ES only together with VaR', 'ES este coerent; VaR poate penaliza diversificarea; VaR este elicitabil, ES doar împreună cu VaR'),
    T('The Normal VaR is @{rg.gapN.lo}--@{rg.gapN.hi}\\% too low on daily data; HS, Student-t and EVT agree at 1\\%; Cornish--Fisher fails for large kurtosis', 'VaR-ul Normal este prea mic cu @{rg.gapN.lo}--@{rg.gapN.hi}\\% pe date zilnice; HS, Student-t și EVT sînt de acord la 1\\%; Cornish--Fisher nu funcționează pentru boltire mare'),
    T('Conditional VaR (GARCH-t, FHS) follows the storms and passes backtests that window methods fail', 'VaR condiționat (GARCH-t, FHS) urmărește perioadele turbulente și trece testele pe care metodele pe fereastră nu le trec'),
    T('Portfolio risk: correlation rises in crises; tail dependence (t copula) matters for joint crashes', 'Riscul de portofoliu: corelația crește în crize; dependența în cozi (copula t) contează pentru prăbușirile simultane'),
    T('Backtest with Kupiec (rate), Christoffersen (clustering), the traffic light (regulation) and $Z_2$ (ES)', 'Backtesting cu Kupiec (rata), Christoffersen (gruparea), semaforul Basel (reglementarea) și $Z_2$ (ES)')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['VaR & $\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$; \\ HS: $-r_{(\\lceil n\\alpha\\rceil)}$',
     'ES & $\\mathrm{ES}_\\alpha = -\\frac{1}{\\alpha}\\int_0^\\alpha q_u\\,du$; \\ HS: $-\\frac1k\\sum_{i \\le k} r_{(i)}$',
     T('Normal', 'Normală') + ' & $-(\\mu + \\sigma z_\\alpha)$, \\ $-\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$',
     'EVT (POT) & $u + \\frac{\\beta}{\\xi}[(n\\alpha/N_u)^{-\\xi} - 1]$, \\ ES $= (\\mathrm{VaR} + \\beta - \\xi u)/(1 - \\xi)$',
     T('Conditional', 'Condiționat') + ' & $-(\\mu + \\sigma_{t+1}q_\\alpha(z))$',
     T('Portfolio', 'Portofoliu') + ' & $\\sigma_p^2 = w^\\top\\Sigma w$',
     T('t copula', 'copula t') + ' & $\\lambda_L = 2t_{\\nu+1}(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)})$',
     'Kupiec & $LR_{uc} = -2\\ln[(1-\\alpha)^{n-x}\\alpha^x/((1-\\hat\\pi)^{n-x}\\hat\\pi^x)] \\sim \\chi^2(1)$',
     'Christoffersen & $LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)$',
     'Acerbi--Székely & $Z_2 = \\frac{1}{T\\alpha}\\sum_t r_tI_t/\\mathrm{ES}_t + 1$'],
    size='small') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a desk reports ``VaR 1\\% = $-2.4\\%$\'\'. What is wrong?', '\\textbf{Întrebare}: o divizie raportează „VaR 1\\% = $-2{,}4\\%$”. Ce este greșit?'),
     [T('\\textbf{Answer}: the sign: $-2.4\\%$ is the 1\\% quantile $q_{0.01}$; VaR 1\\% $= -q_{0.01} = 2.4\\%$, a loss', '\\textbf{Răspuns}: semnul: $-2{,}4\\%$ este cuantila $q_{0{,}01}$; VaR 1\\% $= -q_{0{,}01} = 2{,}4\\%$, o pierdere')]),
    (T('\\textbf{Question}: a model has 3 exceptions in 250 days, all in the same week. Which test detects the problem?', '\\textbf{Întrebare}: un model are 3 depășiri în 250 de zile, toate în aceeași săptămînă. Ce test detectează problema?'),
     [T('\\textbf{Answer}: not Kupiec (3 is close to 2.5), but Christoffersen\'s independence test', '\\textbf{Răspuns}: nu testul Kupiec (3 este aproape de 2,5), ci testul de independență al lui Christoffersen')]),
    (T('\\textbf{Question}: two portfolios have the same VaR 1\\%. Can their ES 2.5\\% differ?', '\\textbf{Întrebare}: două portofolii au același VaR 1\\%. Pot avea ES 2,5\\% diferit?'),
     [T('\\textbf{Answer}: yes: ES depends on the whole tail beyond the quantile, VaR only on one point', '\\textbf{Răspuns}: da: ES depinde de întreaga coadă de dincolo de cuantilă, VaR doar de un punct')]),
    T('Next: Chapter 11, the fractal markets hypothesis and long memory', 'Urmează: Capitolul 11, ipoteza piețelor fractale și memoria lungă')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
