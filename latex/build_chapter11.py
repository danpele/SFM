r"""
build_chapter11.py -- Capitolul 11 (Ipoteza piețelor fractale și memoria lungă), EN + RO dintr-o singură sursă
==============================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_11/ch11_numbers.json (generate_fmh_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter11_fractal_markets_long_memory.tex
  RO/Cursuri/capitol11_piete_fractale_memorie_lunga.tex
Rulare:
  python3 Quantlets/Ch_11/generate_fmh_charts.py
  python3 latex/build_chapter11.py && python3 latex/sfm_build.py compile 11
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch11_common import ASSETS, BIB, NAMES, QLURL, REFS, SHORT, T, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(11, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_11}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.58\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'hurst': ('ch11_hurst_1953.jpg', C + 'Harold_Edwin_Hurst_in_1953.jpg', '⟦Photo||Foto⟧: Elliott \\& Fry (1953); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'nilometer': ('ch11_nilometer_roda.jpg', C + 'Nilometer_(8590204613).jpg', '⟦Photo||Foto⟧: David Stanley (2013); CC BY 2.0; Wikimedia Commons'),
    'aswan': ('ch11_aswan_low_dam.jpg', C + 'Aswan_Low_Dam_Egypt_1.jpg', '⟦Photo||Foto⟧: Karelj (2010); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'mandelbrot': ('ch11_mandelbrot_2006.jpg', C + 'Mandelbrot_p1130876.jpg', '⟦Photo||Foto⟧: David Monniaux (2006); CC BY-SA 3.0; Wikimedia Commons'),
    'mset': ('ch11_mandelbrot_set.jpg', C + 'Mandel_zoom_00_mandelbrot_set.jpg', '⟦Image||Imagine⟧: Wolfgang Beyer; CC BY-SA 3.0; Wikimedia Commons'),
    'btc': ('ch11_bitcoin_atm_prague.jpg', C + 'Bitcoin_ATM_Prague.jpg', '⟦Photo||Foto⟧: Perituss (2016); CC0; Wikimedia Commons'),
    'crash87': ('ch11_sp500_1987_fed.png', C + 'S\\%26P_500_index_around_the_time_of_the_crash.png',
                '⟦Chart||Grafic⟧: Mark Carlson, Federal Reserve Board (2006); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
# fractal dimensions
V.put('dim.koch', math.log(4) / math.log(3), 3)
V.put('dim.sier', math.log(3) / math.log(2), 3)
# ARFIMA(0, 0.3, 0): weights and autocorrelations against an AR(1) with the same rho(1)
d0 = 0.3
rho = [1.0]
for k in range(1, 2001):
    rho.append(rho[-1] * (k - 1 + d0) / (k - d0))
V.put('af3.pi1', -d0, 3)
V.put('af3.pi2', -d0 * (1 - d0) / 2, 3)
V.put('af3.psi1', d0, 3)
V.put('af3.psi2', d0 * (1 + d0) / 2, 3)
V.put('af3.r1', rho[1], 3)
V.put('af3.r2', rho[2], 3)
V.put('af3.r10', rho[10], 3)
V.put('af3.r100', rho[100], 3)
V.put('af3.ar10', rho[1] ** 10, 4)
first_af = next(k for k in range(1, 2001) if rho[k] < 0.05)
first_ar = next(k for k in range(1, 200) if rho[1] ** k < 0.05)
V.raw('af3.first', str(first_af))
V.raw('af3.firstar', str(first_ar))
# risk scaling with h^H (Normal VaR 1%, daily volatility 1.2%)
z, sig = 2.326, 1.2
V.put('vs.var1', z * sig, 2)
for H in (0.45, 0.5, 0.55):
    k = str(int(round(100 * H)))
    V.put(f'vs.f10.{k}', 10 ** H, 2)
    V.put(f'vs.v10.{k}', z * sig * 10 ** H, 2)
    V.put(f'vs.f250.{k}', 250 ** H, 1)
V.put('vs.diff', 100 * (10 ** 0.55 / 10 ** 0.5 - 1), 0)
V.put('vs.diff250', 100 * (250 ** 0.55 / 250 ** 0.5 - 1), 0)
# fGn rho(1) = 2^(2H-1) - 1 at H = 0.8
V.put('fgn.r1f', 2 ** 0.6 - 1, 3)
M = N['markets']
V.put('m.absd.min', min(M[k]['abs']['gph_d'] for k in ASSETS), 2)
V.put('m.absd.max', max(M[k]['abs']['gph_d'] for k in ASSETS), 2)
V.put('m.absdfa.min', min(M[k]['abs']['dfa'] for k in ASSETS), 2)
V.put('m.absdfa.max', max(M[k]['abs']['dfa'] for k in ASSETS), 2)
V.put('m.shuf.min', min(M[k]['abs_shuffled_dfa'] for k in ASSETS), 2)
V.put('m.shuf.max', max(M[k]['abs_shuffled_dfa'] for k in ASSETS), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: do today\'s returns remember what happened months ago?',
       '\\textbf{Întrebarea}: își amintesc randamentele de azi ce s-a întîmplat acum cîteva luni?'),
     [T('the fractal market hypothesis adds a second theme: who trades, and at which horizon', 'ipoteza pieței fractale adaugă o a doua temă: cine tranzacționează și pe ce orizont'),
      T('Chapter 7: daily returns are almost uncorrelated at short lags; Chapters 8 and 9: volatility is persistent',
        'Capitolul 7: randamentele zilnice sînt aproape necorelate la laguri mici; Capitolele 8 și 9: volatilitatea este persistentă'),
      T('this chapter asks about dependence over \\textbf{long} lags and over \\textbf{many} time scales', 'acest capitol se ocupă de dependența la laguri \\textbf{mari} și pe \\textbf{multe} scale de timp')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the fractal market hypothesis of Peters; self-similarity and fractals', 'ipoteza pieței fractale a lui Peters; autosimilaritate și fractali'),
      T('Hurst and the Nile; the R/S statistic; long memory, fractional Brownian motion, ARFIMA', 'Hurst și Nilul; statistica R/S; memoria lungă, mișcarea browniană fracționară, ARFIMA'),
      T('estimators (R/S, Lo, DFA, GPH), Monte Carlo bands, spurious long memory', 'estimatori (R/S, Lo, DFA, GPH), benzi Monte Carlo, memorie lungă aparentă'),
      T('long memory in volatility; rolling Hurst exponents on six markets; consequences for risk', 'memoria lungă a volatilității; exponenți Hurst pe ferestre mobile pentru șase piețe; consecințe pentru risc')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('State the fractal market hypothesis and compare it with the efficient and the adaptive market hypotheses',
      'Enunțați ipoteza pieței fractale și comparați-o cu ipoteza pieței eficiente și cu ipoteza piețelor adaptive'),
    T('Define self-similarity, the Hurst exponent, fractional Brownian motion, fractional Gaussian noise and ARFIMA(0,$d$,0)',
      'Definiți autosimilaritatea, exponentul Hurst, mișcarea browniană fracționară, zgomotul gaussian fracționar și ARFIMA(0,$d$,0)'),
    T('Compute the R/S statistic step by step and estimate $H$ with R/S, DFA and the GPH regression; apply Lo\'s modified R/S test',
      'Calculați statistica R/S pas cu pas și estimați $H$ cu R/S, DFA și regresia GPH; aplicați testul R/S modificat al lui Lo'),
    T('Judge an estimate against Monte Carlo bands and recognise spurious long memory (breaks, volatility clustering)',
      'Evaluați o estimare față de benzile Monte Carlo și recunoașteți memoria lungă aparentă (rupturi structurale, volatility clustering)'),
    T('Separate long memory in returns from long memory in volatility on real data, with rolling estimates',
      'Deosebiți memoria lungă a randamentelor de memoria lungă a volatilității pe date reale, cu estimări pe ferestre mobile'),
    T('Explain how $H$ changes the scaling of risk with the horizon', 'Explicați cum schimbă $H$ scalarea riscului cu orizontul')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~14 (14.1--14.5 definition, fractional integration, self-similarity, tests and estimators of long memory; 14.6 long memory models)',
       'Manual: \\refFHH, cap.~14 (14.1--14.5 definiție, integrare fracționară, autosimilaritate, teste și estimatori ai memoriei lungi; 14.6 modele cu memorie lungă)'),
     [T('the fractal market hypothesis: \\refPeters; a survey of long memory: \\refBaillie', 'ipoteza pieței fractale: \\refPeters; o sinteză despre memoria lungă: \\refBaillie')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_11}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_11}'),
     [T('ported from the SFE Quantlets SFEfbmplot, SFEfgnacf and SFEarfima', 'portate din Quantlet-urile SFE SFEfbmplot, SFEfgnacf și SFEarfima')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter11_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter11_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video courses: \\quantinar{Efficiency of cryptocurrency markets}{https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis}; \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Cursuri video: \\quantinar{Efficiency of cryptocurrency markets}{https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis}; \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}'),
    T('Further reading on crashes and power laws in Romanian and crypto markets: \\refPeleA; \\refPeleB',
      'Lecturi suplimentare despre crahuri și legi de putere pe piața românească și pe piața cripto: \\refPeleA; \\refPeleB')))

# =============================================================================
# 1. DE LA PIEȚE EFICIENTE LA PIEȚE FRACTALE
# =============================================================================
D.section('From Efficient to Fractal Markets', 'De la piețe eficiente la piețe fractale')

D.frame(T('What the EMH Leaves Open', 'Întrebări lăsate deschise de EMH'), items(
    (T('EMH (efficient market hypothesis, \\refFama; Chapter 7): prices reflect the available information, so returns cannot be forecast from past returns',
       'EMH (efficient market hypothesis, ipoteza pieței eficiente, \\refFama; Capitolul 7): prețurile reflectă informația disponibilă, deci randamentele nu pot fi anticipate din randamentele trecute'),
     [T('the random walk of prices implies independent increments: in this chapter, $H = 0.5$', 'mersul aleator al prețurilor implică creșteri independente: în acest capitol, $H = 0.5$')]),
    (T('Facts that the random walk with Normal increments does not explain', 'Fapte pe care mersul aleator cu creșteri Normale nu le explică'),
     [T('heavy tails (Chapters 2, 3, 5); volatility clustering (Chapters 8, 9)', 'cozi groase (Capitolele 2, 3, 5); volatility clustering (Capitolele 8, 9)'),
      T('crashes such as that of 19 October 1987 (a few slides ahead)', 'crahuri precum cel din 19 octombrie 1987 (cîteva slide-uri mai departe)')]),
    T('The EMH says nothing about \\textbf{who} trades and at which \\textbf{horizon}: all investors are treated as one representative investor',
      'EMH nu spune nimic despre \\textbf{cine} tranzacționează și pe ce \\textbf{orizont}: toți investitorii sînt tratați ca un singur investitor reprezentativ')))

D.frame(T('Edgar Peters and the Fractal Market Hypothesis', 'Edgar Peters și ipoteza pieței fractale'), items(
    T('\\textbf{FMH} (fractal market hypothesis) \\refPeters: the market is made of investors with \\textbf{many investment horizons}, from minutes to decades',
      '\\textbf{FMH} (fractal market hypothesis, ipoteza pieței fractale) \\refPeters: piața este formată din investitori cu \\textbf{multe orizonturi investiționale}, de la minute la decenii'),
    (T('The five statements of Peters', 'Cele cinci afirmații ale lui Peters'),
     [T('the market is stable when it contains investors with many different horizons: this ensures liquidity',
        'piața este stabilă cînd conține investitori cu orizonturi foarte diferite: astfel se asigură lichiditatea'),
      T('short horizons react to sentiment and technical information; long horizons react to fundamental information',
        'orizonturile scurte reacționează la sentiment și la informația tehnică; orizonturile lungi reacționează la informația fundamentală'),
      T('if the fundamental information becomes doubtful, long-term investors leave or start to trade short term: horizons become uniform and the market becomes unstable',
        'dacă informația fundamentală devine îndoielnică, investitorii pe termen lung se retrag sau încep să tranzacționeze pe termen scurt: orizonturile devin uniforme, iar piața devine instabilă'),
      T('prices combine short-term technical trading and long-term fundamental valuation', 'prețurile combină tranzacționarea tehnică pe termen scurt și evaluarea fundamentală pe termen lung'),
      T('an asset without a link to the economic cycle has no long-term trend: short-term trading and liquidity dominate',
        'un activ fără legătură cu ciclul economic nu are tendință pe termen lung: domină tranzacționarea pe termen scurt și lichiditatea')])), 'footnotesize')

D.frame(T('Investment Horizons and Liquidity: an Example', 'Orizonturi investiționale și lichiditate: un exemplu'), items(
    (T('A share falls by 3\\% in one morning', 'O acțiune scade cu 3\\% într-o dimineață'),
     [T('a day trader (horizon: hours) sees a large move and sells', 'un day trader (orizont: ore) vede o mișcare mare și vinde'),
      T('a pension fund (horizon: ten years) sees a small move relative to its horizon and buys at a lower price',
        'un fond de pensii (orizont: zece ani) vede o mișcare mică în raport cu orizontul lui și cumpără la un preț mai mic')]),
    T('The long-horizon investor supplies the liquidity that the short-horizon investor demands: the price stays orderly',
      'Investitorul cu orizont lung oferă lichiditatea cerută de investitorul cu orizont scurt: prețul evoluează ordonat'),
    T('If the pension fund also starts to fear the next hours, nobody buys: there is no liquidity and the fall accelerates',
      'Dacă și fondul de pensii începe să se teamă de orele următoare, nu mai cumpără nimeni: nu mai există lichiditate, iar scăderea se accelerează'),
    T('\\textbf{Fractal}: the same pattern of behaviour repeats at every horizon, with statistically similar returns at different scales',
      '\\textbf{Fractal}: același tipar de comportament se repetă pe fiecare orizont, cu randamente asemănătoare statistic la scale diferite')))

D.frame(T('1987: a Crash as a Collapse of Horizons', '1987: un crah ca prăbușire a orizonturilor'), cols(items(
    T('19 October 1987: the S\\&P 500 fell by about 20\\% in one day \\refCarlson', '19 octombrie 1987: S\\&P 500 a scăzut cu circa 20\\% într-o singură zi \\refCarlson'),
    (T('The FMH reading', 'Interpretarea FMH'),
     [T('portfolio insurance and program trading made long-term portfolios sell like short-term traders', 'asigurarea de portofoliu și tranzacționarea automată au făcut ca portofoliile pe termen lung să vîndă ca traderii pe termen scurt'),
      T('all horizons became one horizon: buyers disappeared', 'toate orizonturile s-au redus la unul singur: cumpărătorii au dispărut')]),
    T('Testable consequence: in crises, short horizons dominate and the statistical structure across scales changes',
      'Consecință testabilă: în crize domină orizonturile scurte, iar structura statistică pe scale diferite se schimbă')),
    ph('crash87', T('The S\\&P 500 around the crash of October 1987', 'S\\&P 500 în jurul crahului din octombrie 1987'), h='0.36\\textheight'),
    wl='0.52', wr='0.44'), 'footnotesize')

D.frame(T('Testing the FMH: Scaling, Horizons and Liquidity', 'Testarea FMH: scalare, orizonturi și lichiditate'), items(
    (T('The FMH is a verbal theory; its statistical consequences can be tested', 'FMH este o teorie formulată verbal; consecințele ei statistice pot fi testate'),
     [T('\\textbf{self-similarity}: the distribution of $h$-day returns looks the same after rescaling by $h^H$ (Section 2)', '\\textbf{autosimilaritate}: distribuția randamentelor pe $h$ zile arată la fel după rescalarea cu $h^H$ (secțiunea 2)'),
      T('\\textbf{memory}: the Hurst exponent $H$ measures dependence over long horizons (Sections 3--5)', '\\textbf{memorie}: exponentul Hurst $H$ măsoară dependența pe orizonturi lungi (secțiunile 3--5)'),
      T('\\textbf{change}: in crises, $H$ and the share of short horizons should change (Section 8)', '\\textbf{schimbare}: în crize, $H$ și ponderea orizonturilor scurte ar trebui să se schimbe (secțiunea 8)')]),
    (T('Landmark evidence: \\refKrisA\\ and \\refKrisB\\ on the 2008 crisis', 'Studii de referință: \\refKrisA\\ și \\refKrisB\\ despre criza din 2008'),
     [T('local Hurst exponents and wavelet power of US and European indices', 'exponenți Hurst locali și puterea wavelet a indicilor din SUA și Europa'),
      T('the most turbulent periods coincide with a dominance of short investment horizons, as the FMH predicts', 'perioadele cele mai turbulente coincid cu dominația orizonturilor investiționale scurte, așa cum prezice FMH')])))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('EMH, FMH and AMH', 'EMH, FMH și AMH'), table(
    TB + 'p{2.2cm}' + TB + 'p{2.9cm}' + TB + 'p{2.9cm}' + TB + 'p{2.9cm}',
    ' & \\textbf{EMH} & \\textbf{FMH} & \\textbf{AMH}',
    [T('Source', 'Sursa') + ' & \\refFama & \\refPeters & \\refLoAMH',
     T('Investors', 'Investitorii') + ' & ' + T('one representative investor', 'un investitor reprezentativ') + ' & ' + T('many horizons', 'multe orizonturi') + ' & ' + T('species that compete and adapt', 'specii care concurează și se adaptează'),
     T('Stability', 'Stabilitatea') + ' & ' + T('prices always fair', 'prețuri mereu corecte') + ' & ' + T('diversity of horizons gives liquidity', 'diversitatea orizonturilor aduce lichiditate') + ' & ' + T('changes with the environment', 'se schimbă odată cu mediul'),
     T('Memory', 'Memoria') + ' & $H = 0.5$ & ' + T('$H$ may differ from 0.5', '$H$ poate fi diferit de 0,5') + ' & ' + T('$H$ varies over time', '$H$ variază în timp'),
     T('Crises', 'Crizele') + ' & ' + T('outside the theory', 'în afara teoriei') + ' & ' + T('collapse of horizons', 'prăbușirea orizonturilor') + ' & ' + T('mass extinction of strategies', 'dispariția în masă a strategiilor')],
    size='footnotesize') + items(
    T('The three are not rival tests: the FMH and the AMH explain when and why the EMH approximation fails',
      'Cele trei nu sînt teste rivale: FMH și AMH explică unde și de ce aproximarea EMH nu mai funcționează')))

# =============================================================================
# 2. AUTOSIMILARITATE ȘI FRACTALI
# =============================================================================
D.section('Self-Similarity and Fractals', 'Autosimilaritate și fractali')

D.frame(T('Benoît Mandelbrot: Roughness as a Measure', 'Benoît Mandelbrot: măsurarea neregularității'), cols(items(
    (T('\\refMan: cotton prices have heavy tails and look alike at daily, monthly and yearly scales', '\\refMan: prețurile bumbacului au cozi groase și arată asemănător la scară zilnică, lunară și anuală'),
     [T('the $\\alpha$-stable distributions of Chapter 3', 'distribuțiile $\\alpha$-stabile din Capitolul 3')]),
    T('\\refCoast: the measured length of a coast grows as the ruler shrinks; the growth rate defines a \\textbf{fractal dimension}',
      '\\refCoast: lungimea măsurată a unei coaste crește cînd rigla devine mai mică; rata de creștere definește o \\textbf{dimensiune fractală}'),
    T('\\refMVN: fractional Brownian motion, the model of this chapter', '\\refMVN: mișcarea browniană fracționară, modelul acestui capitol'),
    T('\\textbf{Fractal}: an object whose parts resemble the whole, exactly or in distribution', '\\textbf{Fractal}: un obiect ale cărui părți seamănă cu întregul, exact sau în distribuție')),
    ph('mandelbrot', T('Benoît Mandelbrot (1924--2010), Paris, 2006', 'Benoît Mandelbrot (1924--2010), Paris, 2006'), h='0.25\\textheight') + '\\\\[1mm]\n' +
    ph('mset', T('The Mandelbrot set', 'Mulțimea Mandelbrot'), h='0.15\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.frame(T('Fractal Dimension', 'Dimensiunea fractală'), items(
    (T('\\textbf{Similarity dimension}: an object made of $N$ copies of itself, each scaled by $1/s$, has $D = \\ln N/\\ln s$',
       '\\textbf{Dimensiunea de similaritate}: un obiect format din $N$ copii ale lui însuși, fiecare scalată cu $1/s$, are $D = \\ln N/\\ln s$'),
     [T('a segment: 2 halves, $D = \\ln 2/\\ln 2 = 1$; a square: 4 quarters of side $1/2$, $D = \\ln 4/\\ln 2 = 2$', 'un segment: 2 jumătăți, $D = \\ln 2/\\ln 2 = 1$; un pătrat: 4 sferturi cu latura $1/2$, $D = \\ln 4/\\ln 2 = 2$'),
      T('the Koch curve: 4 copies scaled by $1/3$, $D = \\ln 4/\\ln 3 = @{dim.koch}$', 'curba Koch: 4 copii scalate cu $1/3$, $D = \\ln 4/\\ln 3 = @{dim.koch}$'),
      T('the Sierpinski triangle: 3 copies scaled by $1/2$, $D = \\ln 3/\\ln 2 = @{dim.sier}$', 'triunghiul Sierpinski: 3 copii scalate cu $1/2$, $D = \\ln 3/\\ln 2 = @{dim.sier}$')]),
    T('The west coast of Britain: $D \\approx 1.25$ \\refCoast', 'Coasta de vest a Marii Britanii: $D \\approx 1.25$ \\refCoast'),
    (T('The graph of a price path: $D = 2 - H$', 'Graficul unei traiectorii de preț: $D = 2 - H$'),
     [T('Brownian motion: $D = 1.5$; a smoother, trending path ($H > 0.5$) has $D < 1.5$', 'mișcarea browniană: $D = 1.5$; o traiectorie mai netedă, cu tendințe ($H > 0.5$), are $D < 1.5$')])))

D.frame(T('Self-Similar Processes', 'Procese autosimilare'), items(
    (T('\\textbf{Self-similar process} \\refFHH, Def.~14.1: $X_{ct} \\overset{d}{=} c^H X_t$ for every $c > 0$; $H$ is the \\textbf{Hurst exponent}',
       '\\textbf{Proces autosimilar} \\refFHH, Def.~14.1: $X_{ct} \\overset{d}{=} c^H X_t$ pentru orice $c > 0$; $H$ este \\textbf{exponentul Hurst}'),
     [T('$\\overset{d}{=}$: equal in distribution; stretching time by $c$ equals stretching values by $c^H$', '$\\overset{d}{=}$: egalitate în distribuție; dilatarea timpului cu $c$ echivalează cu dilatarea valorilor cu $c^H$')]),
    T('Brownian motion (Chapter 4) is self-similar with $H = 1/2$: $W_{ct} \\overset{d}{=} \\sqrt{c}\\,W_t$', 'Mișcarea browniană (Capitolul 4) este autosimilară cu $H = 1/2$: $W_{ct} \\overset{d}{=} \\sqrt{c}\\,W_t$'),
    (T('Consequence for log prices with self-similar increments', 'Consecință pentru prețurile logaritmice cu creșteri autosimilare'),
     [T('$\\mathrm{sd}(r_t^{(h)}) = h^H\\,\\mathrm{sd}(r_t)$, where $r_t^{(h)}$ is the $h$-day return and sd the standard deviation', '$\\mathrm{sd}(r_t^{(h)}) = h^H\\,\\mathrm{sd}(r_t)$, unde $r_t^{(h)}$ este randamentul pe $h$ zile, iar sd abaterea standard'),
      T('the square-root-of-time rule $\\sqrt{h}$ is the special case $H = 1/2$', 'regula rădăcinii pătrate a timpului, $\\sqrt{h}$, este cazul particular $H = 1/2$')]),
    T('A first estimator of $H$: the slope of $\\ln \\mathrm{sd}(r^{(h)})$ on $\\ln h$', 'Un prim estimator al lui $H$: panta dreptei lui $\\ln \\mathrm{sd}(r^{(h)})$ în funcție de $\\ln h$')))

chart(T('The Scaling of $h$-Day Returns', 'Scalarea randamentelor pe $h$ zile'), 'sfm_ch11_scaling', 'SFM_ch11_self_similarity', [
    T('Standard deviation of non-overlapping $h$-day returns divided by the daily one, $h = 1, \\dots, 250$, log scales; data to @{end}',
      'Abaterea standard a randamentelor pe $h$ zile, fără suprapunere, împărțită la cea zilnică, $h = 1, \\dots, 250$, scale logaritmice; date pînă la @{end}'),
    T('Slopes: S\\&P 500 $@{sc.sp500.H}$, BET $@{sc.bet.H}$, Bitcoin $@{sc.btc.H}$; the square-root rule has slope 0.5',
      'Pante: S\\&P 500 $@{sc.sp500.H}$, BET $@{sc.bet.H}$, Bitcoin $@{sc.btc.H}$; regula rădăcinii pătrate are panta 0,5')],
    h='0.50\\textheight')

D.frame(T('Interpretation: Scaling Exponents', 'Interpretarea exponenților de scalare'), items(
    (T('S\\&P 500: 10-day risk is $@{sc.sp500.r10}$ times the daily risk, below $\\sqrt{10} = 3.16$', 'S\\&P 500: riscul pe 10 zile este de $@{sc.sp500.r10}$ ori riscul zilnic, sub $\\sqrt{10} = 3.16$'),
     [T('slight mean reversion: daily returns have a negative first-order autocorrelation, $@{va.r.1}$', 'ușoară revenire la medie: randamentele zilnice au autocorelația de ordinul întîi negativă, $@{va.r.1}$')]),
    (T('BET: $@{sc.bet.r10}$ times, above $\\sqrt{10}$: positive autocorrelation', 'BET: de $@{sc.bet.r10}$ ori, peste $\\sqrt{10}$: autocorelație pozitivă'),
     [T('thin trading in the early years makes the index adjust slowly (Chapter 7)', 'tranzacționarea redusă din primii ani face ca indicele să se ajusteze lent (Capitolul 7)')]),
    T('Bitcoin: $@{sc.btc.r10}$ times, close to $\\sqrt{10}$', 'Bitcoin: de $@{sc.btc.r10}$ ori, aproape de $\\sqrt{10}$'),
    T('Caution: there are only a few dozen non-overlapping 250-day blocks; the slope is imprecise, so it needs a proper estimator and a band',
      'Atenție: există doar cîteva zeci de blocuri de 250 de zile fără suprapunere; panta este imprecisă, deci avem nevoie de un estimator adecvat și de o bandă de încredere')))

# =============================================================================
# 3. HURST, NILUL ȘI STATISTICA R/S
# =============================================================================
D.section('Hurst, the Nile and the R/S Statistic', 'Hurst, Nilul și statistica R/S')

D.frame(T('Harold Edwin Hurst and the Nile', 'Harold Edwin Hurst și Nilul'), cols(items(
    (T('Hurst (1880--1978), a British hydrologist who worked in Egypt from 1906, designed reservoirs on the Nile', 'Hurst (1880--1978), hidrolog britanic care a lucrat în Egipt din 1906, a proiectat rezervoare pe Nil'),
     [T('the capacity needed to deliver the average flow every year equals the range of the cumulative deviations of the inflows from their mean',
        'capacitatea necesară pentru a livra în fiecare an debitul mediu este egală cu amplitudinea abaterilor cumulate ale debitelor de la medie')]),
    (T('\\refHurst: for independent inflows this range grows like $n^{0.5}$; in river flows, rainfall and tree rings it grows like $n^{K}$ with $K$ about $0.73$ on average',
       '\\refHurst: pentru debite independente, această amplitudine crește ca $n^{0.5}$; în seriile de debite ale rîurilor, de precipitații și de inele ale copacilor crește ca $n^{K}$, cu $K$ în medie de circa $0.73$'),
     [T('years of high flow follow years of high flow: \\textbf{long memory}', 'anii cu debit mare urmează după ani cu debit mare: \\textbf{memorie lungă}')])),
    ph('hurst', T('H.E. Hurst (1953)', 'H.E. Hurst (1953)'), h='0.24\\textheight') + '\\\\[1mm]\n' +
    ph('nilometer', T('The Nilometer on Roda Island, Cairo, built in 861', 'Nilometrul de pe insula Roda, Cairo, construit în 861'), h='0.17\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('The R/S Statistic', 'Statistica R/S'), items(
    (T('Block $x_1, \\dots, x_n$ with mean $\\bar x$; \\textbf{cumulative deviations} $Y_k = \\sum_{j=1}^{k}(x_j - \\bar x)$, $k = 1, \\dots, n$',
       'Blocul $x_1, \\dots, x_n$, cu media $\\bar x$; \\textbf{abaterile cumulate} $Y_k = \\sum_{j=1}^{k}(x_j - \\bar x)$, $k = 1, \\dots, n$'),
     [T('$Y_n = 0$ by construction', '$Y_n = 0$ prin construcție')]),
    T('\\textbf{Range}: $R_n = \\max_k Y_k - \\min_k Y_k$; \\textbf{standard deviation}: $S_n = \\sqrt{\\tfrac1n\\sum_j (x_j - \\bar x)^2}$',
      '\\textbf{Amplitudinea}: $R_n = \\max_k Y_k - \\min_k Y_k$; \\textbf{abaterea standard}: $S_n = \\sqrt{\\tfrac1n\\sum_j (x_j - \\bar x)^2}$'),
    (T('\\textbf{R/S statistic} (rescaled range) \\refFHH, 14.4.1: $(R/S)_n = R_n/S_n$', '\\textbf{Statistica R/S} (rescaled range, amplitudinea rescalată) \\refFHH, 14.4.1: $(R/S)_n = R_n/S_n$'),
     [T('dividing by $S_n$ removes the unit: $(R/S)_n$ is comparable across assets', 'împărțirea la $S_n$ elimină unitatea de măsură: $(R/S)_n$ este comparabilă între active'),
      T('for i.i.d.\\ data with finite variance, $(R/S)_n$ grows like $n^{1/2}$; under long memory like $n^H$, $H > 1/2$', 'pentru date i.i.d.\\ cu varianță finită, $(R/S)_n$ crește ca $n^{1/2}$; în cazul memoriei lungi crește ca $n^H$, $H > 1/2$')]),
    T('Robust to heavy tails: $R$ and $S$ are inflated by the same extreme values \\refMW', 'Robustă la cozi groase: $R$ și $S$ sînt mărite de aceleași valori extreme \\refMW')))

D.frame(T('Worked Example: R/S of Eight Returns', 'Exemplu rezolvat: R/S pentru opt randamente'), cols(items(
    T('Returns (\\%): $0.5;\\ -0.3;\\ 0.8;\\ -0.2;\\ 0.6;\\ -0.1;\\ 0.4;\\ -0.7$', 'Randamente (\\%): $0.5;\\ -0.3;\\ 0.8;\\ -0.2;\\ 0.6;\\ -0.1;\\ 0.4;\\ -0.7$'),
    T('Mean $\\bar x = 1.0/8 = @{ex.mean}$', 'Media $\\bar x = 1.0/8 = @{ex.mean}$'),
    T('Deviations: $@{ex.dev1}$, $@{ex.dev2}$, $@{ex.dev3}$, $@{ex.dev4}$, $@{ex.dev5}$, $@{ex.dev6}$, $@{ex.dev7}$, $@{ex.dev8}$',
      'Abateri: $@{ex.dev1}$; $@{ex.dev2}$; $@{ex.dev3}$; $@{ex.dev4}$; $@{ex.dev5}$; $@{ex.dev6}$; $@{ex.dev7}$; $@{ex.dev8}$'),
    T('$Y_k$: $@{ex.y1}$, $@{ex.y2}$, $@{ex.y3}$, $@{ex.y4}$, $@{ex.y5}$, $@{ex.y6}$, $@{ex.y7}$, $@{ex.y8}$',
      '$Y_k$: $@{ex.y1}$; $@{ex.y2}$; $@{ex.y3}$; $@{ex.y4}$; $@{ex.y5}$; $@{ex.y6}$; $@{ex.y7}$; $@{ex.y8}$'),
    T('$R = @{ex.max} - (@{ex.min}) = @{ex.R}$; $S = \\sqrt{@{ex.ss}/8} = @{ex.S}$', '$R = @{ex.max} - (@{ex.min}) = @{ex.R}$; $S = \\sqrt{@{ex.ss}/8} = @{ex.S}$'),
    T('$(R/S)_8 = @{ex.R}/@{ex.S} = @{ex.RS}$; one block gives one point, $H$ needs many block sizes', '$(R/S)_8 = @{ex.R}/@{ex.S} = @{ex.RS}$; un bloc dă un singur punct, iar $H$ are nevoie de multe mărimi de bloc')),
    '\\includegraphics[width=\\textwidth]{sfm_ch11_rs_example.pdf}\n' + ql('SFM_ch11_rs_estimators'), wl='0.52', wr='0.46'), 'scriptsize')

D.frame(T('From R/S to the Hurst Exponent', 'De la R/S la exponentul Hurst'), items(
    (T('Steps \\refMW', 'Pașii \\refMW'),
     [T('choose block sizes $n$ between 10 and $N/4$, evenly spaced on a log scale', 'alegem mărimi de bloc $n$ între 10 și $N/4$, la distanțe egale pe scară logaritmică'),
      T('split the $N$ returns into $\\lfloor N/n\\rfloor$ non-overlapping blocks; compute $R/S$ in each block and average: $(R/S)_n$', 'împărțim cele $N$ randamente în $\\lfloor N/n\\rfloor$ blocuri fără suprapunere; calculăm $R/S$ în fiecare bloc și facem media: $(R/S)_n$'),
      T('regress $\\ln (R/S)_n = \\ln c + H \\ln n + e_n$ by OLS (ordinary least squares): the slope $\\hat H$ estimates $H$', 'estimăm prin OLS (ordinary least squares, metoda celor mai mici pătrate) $\\ln (R/S)_n = \\ln c + H \\ln n + e_n$: panta $\\hat H$ estimează $H$'),
      T('$N$: the number of returns; $n$: the block size; $c$: a constant; $e_n$: the regression error', '$N$: numărul de randamente; $n$: mărimea blocului; $c$: o constantă; $e_n$: eroarea regresiei')]),
    T('The log-log plot is the main diagnostic: a straight line supports a single scaling exponent', 'Graficul log-log este principalul instrument de diagnostic: o dreaptă susține existența unui singur exponent de scalare'),
    T('Applied to returns (stationary), not to prices: for prices, $(R/S)_n$ grows like $n$', 'Se aplică randamentelor (staționare), nu prețurilor: pentru prețuri, $(R/S)_n$ crește ca $n$')))

D.frame(T('Interpreting $H$', 'Interpretarea exponentului $H$'), table(
    'llll', T('$H$ & increments & memory & path ($D = 2 - H$)', '$H$ & creșteri & memorie & traiectorie ($D = 2 - H$)'),
    [T('$0 < H < 0.5$ & negatively correlated & anti-persistent: reversals & rough, $D > 1.5$', '$0 < H < 0.5$ & corelate negativ & antipersistentă: inversări & neregulată, $D > 1.5$'),
     T('$H = 0.5$ & uncorrelated & none (random walk) & $D = 1.5$', '$H = 0.5$ & necorelate & fără memorie (mers aleator) & $D = 1.5$'),
     T('$0.5 < H < 1$ & positively correlated & persistent: trends & smoother, $D < 1.5$', '$0.5 < H < 1$ & corelate pozitiv & persistentă: tendințe & mai netedă, $D < 1.5$')],
    size='footnotesize') + items(
    T('$H$ is not a probability: $H = 0.7$ does not mean ``70\\% chance that the sign repeats\'\'', '$H$ nu este o probabilitate: $H = 0.7$ nu înseamnă „70\\% șanse ca semnul să se repete”'),
    (T('Two effects named by \\refNoah', 'Două efecte numite de \\refNoah'),
     [T('\\textbf{Noah effect}: extreme values (heavy tails, the flood); \\textbf{Joseph effect}: long runs of high or low values (``seven years of plenty, seven years of famine\'\')',
        '\\textbf{efectul Noe}: valori extreme (cozi groase, potopul); \\textbf{efectul Iosif}: serii lungi de valori mari sau mici („șapte ani de belșug, șapte ani de foamete”)'),
      T('$H$ measures the Joseph effect; the tail index of Chapter 5 measures the Noah effect', '$H$ măsoară efectul Iosif; tail index-ul din Capitolul 5 măsoară efectul Noe')])))

D.frame(T('Small Samples: the Bias of R/S', 'Eșantioane mici: deplasarea estimatorului R/S'), items(
    (T('For i.i.d.\\ data, $(R/S)_n$ approaches $c\\,n^{1/2}$ only for large $n$; for small $n$ it grows faster \\refAL', 'Pentru date i.i.d., $(R/S)_n$ se apropie de $c\\,n^{1/2}$ doar pentru $n$ mare; pentru $n$ mic crește mai repede \\refAL'),
     [T('the fitted slope over $n = 10, \\dots, N/4$ is above 0.5 even without any memory', 'panta estimată pe $n = 10, \\dots, N/4$ este peste 0,5 chiar fără nicio memorie')]),
    (T('Monte Carlo, 500 i.i.d.\\ Normal series', 'Monte Carlo, 500 de serii i.i.d.\\ Normale'),
     [T('mean $\\hat H_{R/S}$: $@{mc.250.rs.mean}$ for $N = 250$, $@{mc.1000.rs.mean}$ for $N = 1000$, $@{mc.5000.rs.mean}$ for $N = 5000$',
        '$\\hat H_{R/S}$ mediu: $@{mc.250.rs.mean}$ pentru $N = 250$, $@{mc.1000.rs.mean}$ pentru $N = 1000$, $@{mc.5000.rs.mean}$ pentru $N = 5000$'),
      T('95\\% band for $N = 1000$: $[@{mc.1000.rs.lo}, @{mc.1000.rs.hi}]$', 'banda de 95\\% pentru $N = 1000$: $[@{mc.1000.rs.lo}, @{mc.1000.rs.hi}]$')]),
    T('Rule: compare $\\hat H$ with the distribution of $\\hat H$ under the null hypothesis for the same $N$, not with 0.5 (Section 5)',
      'Regulă: comparăm $\\hat H$ cu distribuția lui $\\hat H$ în ipoteza nulă, pentru același $N$, nu cu 0,5 (secțiunea 5)')))

# =============================================================================
# 4. MEMORIA LUNGĂ
# =============================================================================
D.section('Long Memory: Definitions and Models', 'Memoria lungă: definiții și modele')

D.frame(T('Short and Long Memory', 'Memorie scurtă și memorie lungă'), items(
    (T('Stationary $X_t$ with autocorrelations $\\rho(k)$ (ACF, autocorrelation function); \\textbf{long memory} \\refFHH, 14.1:',
       '$X_t$ staționar, cu autocorelațiile $\\rho(k)$ (ACF, autocorrelation function, funcția de autocorelație); \\textbf{memorie lungă} \\refFHH, 14.1:'),
     [T('$\\sum_{k} |\\rho(k)| = \\infty$: the autocorrelations are not summable', '$\\sum_{k} |\\rho(k)| = \\infty$: autocorelațiile nu sînt sumabile'),
      T('equivalently, hyperbolic decay $\\rho(k) \\sim C\\,k^{2d - 1}$ as $k \\to \\infty$, with the \\textbf{memory parameter} $0 < d < 0.5$', 'echivalent, scădere hiperbolică $\\rho(k) \\sim C\\,k^{2d - 1}$ cînd $k \\to \\infty$, cu \\textbf{parametrul de memorie} $0 < d < 0.5$'),
      T('in frequency: the spectral density has a pole at zero, $f(\\lambda) \\sim C\\,\\lambda^{-2d}$ as $\\lambda \\to 0$', 'în frecvență: densitatea spectrală are un pol în zero, $f(\\lambda) \\sim C\\,\\lambda^{-2d}$ cînd $\\lambda \\to 0$'),
      T('$C > 0$: a constant; $\\sim$: the ratio of the two sides tends to 1; $f(\\lambda)$: the share of variance at frequency $\\lambda$ (long cycles: $\\lambda$ near 0)', '$C > 0$: o constantă; $\\sim$: raportul celor doi membri tinde la 1; $f(\\lambda)$: partea din varianță la frecvența $\\lambda$ (ciclurile lungi: $\\lambda$ aproape de 0)')]),
    (T('\\textbf{Short memory}: ARMA (autoregressive moving average) models, e.g.\\ AR(1) with $\\rho(k) = \\phi^k$', '\\textbf{Memorie scurtă}: modelele ARMA (autoregressive moving average, autoregresive cu medie mobilă), de exemplu AR(1) cu $\\rho(k) = \\phi^k$'),
     [T('exponential decay: $\\sum_k |\\phi|^k = 1/(1 - |\\phi|) < \\infty$', 'scădere exponențială: $\\sum_k |\\phi|^k = 1/(1 - |\\phi|) < \\infty$')]),
    T('Link with the Hurst exponent: $d = H - 1/2$', 'Legătura cu exponentul Hurst: $d = H - 1/2$')))

D.frame(T('Worked Example: How Fast Does Memory Fade?', 'Exemplu rezolvat: cît de repede se stinge memoria?'), items(
    (T('Long memory: fractional Gaussian noise (next slides) with $H = 0.8$', 'Memorie lungă: zgomot gaussian fracționar (slide-urile următoare) cu $H = 0.8$'),
     [T('$\\rho(1) = 2^{2H - 1} - 1 = 2^{0.6} - 1 = @{fgn.r1f}$', '$\\rho(1) = 2^{2H - 1} - 1 = 2^{0.6} - 1 = @{fgn.r1f}$'),
      T('$\\rho(10) = @{fgn.8.r10}$, $\\rho(50) = @{fgn.8.r50}$, $\\rho(100) = @{fgn.8.r100}$', '$\\rho(10) = @{fgn.8.r10}$, $\\rho(50) = @{fgn.8.r50}$, $\\rho(100) = @{fgn.8.r100}$')]),
    (T('Short memory: AR(1) with the same first autocorrelation, $\\phi = @{ar.phi}$', 'Memorie scurtă: AR(1) cu aceeași autocorelație de ordinul întîi, $\\phi = @{ar.phi}$'),
     [T('$\\rho(10) = \\phi^{10} = @{ar.r10}$; $\\rho(100) = \\phi^{100}$, practically zero', '$\\rho(10) = \\phi^{10} = @{ar.r10}$; $\\rho(100) = \\phi^{100}$, practic zero')]),
    T('Same short-run behaviour, very different long run: after 100 days, a shock still matters only under long memory',
      'Același comportament pe termen scurt, dar un termen lung foarte diferit: după 100 de zile, un șoc mai contează doar în cazul memoriei lungi')))

D.frame(T('Fractional Brownian Motion', 'Mișcarea browniană fracționară'), items(
    T('\\textbf{Fractional Brownian motion} (fBm) $B_H(t)$, $0 < H < 1$ \\refMVN; \\refFHH, Def.~14.2: a Gaussian process with continuous paths, self-similar with exponent $H$ and with stationary increments',
       '\\textbf{Mișcarea browniană fracționară} (fBm, fractional Brownian motion) $B_H(t)$, $0 < H < 1$ \\refMVN; \\refFHH, Def.~14.2: proces gaussian cu traiectorii continue, autosimilar cu exponentul $H$ și cu creșteri staționare'),
    (T('Properties (\\refFHH, Th.~14.1)', 'Proprietăți (\\refFHH, Teorema 14.1)'),
     [T('$B_H(0) = 0$, $E[B_H(t)] = 0$, $E[B_H(t)^2] = \\sigma^2 |t|^{2H}$ ($\\sigma^2$: the variance at $t = 1$)', '$B_H(0) = 0$, $E[B_H(t)] = 0$, $E[B_H(t)^2] = \\sigma^2 |t|^{2H}$ ($\\sigma^2$: varianța la $t = 1$)'),
      T('$\\mathrm{Cov}(B_H(t), B_H(s)) = \\tfrac{\\sigma^2}{2}\\left(|t|^{2H} + |s|^{2H} - |t - s|^{2H}\\right)$', '$\\mathrm{Cov}(B_H(t), B_H(s)) = \\tfrac{\\sigma^2}{2}\\left(|t|^{2H} + |s|^{2H} - |t - s|^{2H}\\right)$')]),
    T('$H = 1/2$: Brownian motion, independent increments; $H \\neq 1/2$: the increments are correlated at all lags',
      '$H = 1/2$: mișcarea browniană, creșteri independente; $H \\neq 1/2$: creșterile sînt corelate la toate lagurile'),
    T('Simulation: exact, from the covariance matrix of the increments (circulant embedding, as in the notebook)',
      'Simulare: exactă, din matricea de covarianță a creșterilor (scufundare circulantă, ca în notebook)')))

chart(T('fBm Paths for Three Hurst Exponents', 'Traiectorii fBm pentru trei exponenți Hurst'), 'sfm_ch11_fbm_paths', 'SFM_ch11_fbm_fgn', [
    T('The same random numbers, $H = 0.3$, $0.5$, $0.7$ (as in SFEfbmplot); right: the first 150 increments',
      'Aceleași numere aleatoare, $H = 0.3$, $0.5$, $0.7$ (ca în SFEfbmplot); dreapta: primele 150 de creșteri'),
    T('$H = 0.7$: long excursions and a wide range; $H = 0.3$: frequent reversals, a narrow range; lag-1 autocorrelation of the increments $@{fbm.07.a1}$ and $@{fbm.03.a1}$ (theory $@{fbm.07.a1th}$ and $@{fbm.03.a1th}$)',
      '$H = 0.7$: excursii lungi și amplitudine mare; $H = 0.3$: inversări frecvente, amplitudine mică; autocorelația de ordinul 1 a creșterilor $@{fbm.07.a1}$ și $@{fbm.03.a1}$ (teoretic $@{fbm.07.a1th}$ și $@{fbm.03.a1th}$)')],
    h='0.52\\textheight')

D.frame(T('Fractional Gaussian Noise', 'Zgomotul gaussian fracționar'), items(
    (T('\\textbf{Fractional Gaussian noise} (fGn): the increments $Y_k = B_H(k + 1) - B_H(k)$, a stationary series \\refFHH, 14.3',
       '\\textbf{Zgomotul gaussian fracționar} (fGn, fractional Gaussian noise): creșterile $Y_k = B_H(k + 1) - B_H(k)$, o serie staționară \\refFHH, 14.3'),
     [T('$\\rho(k) = \\tfrac12\\left(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}\\right)$', '$\\rho(k) = \\tfrac12\\left(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}\\right)$'),
      T('for large $k$: $\\rho(k) \\approx H(2H - 1)\\,k^{2H - 2}$', 'pentru $k$ mare: $\\rho(k) \\approx H(2H - 1)\\,k^{2H - 2}$')]),
    (T('Three regimes', 'Trei regimuri'),
     [T('$1/2 < H < 1$: $\\rho(k) > 0$ and not summable: long memory with $d = H - 1/2 \\in (0, 1/2)$', '$1/2 < H < 1$: $\\rho(k) > 0$ și nesumabile: memorie lungă cu $d = H - 1/2 \\in (0, 1/2)$'),
      T('$H = 1/2$: white noise', '$H = 1/2$: zgomot alb'),
      T('$0 < H < 1/2$: $\\rho(k) < 0$, summable: anti-persistence', '$0 < H < 1/2$: $\\rho(k) < 0$, sumabile: antipersistență')]),
    T('fGn is the model behind R/S and DFA: for fGn data, both estimate $H$', 'fGn este modelul din spatele R/S și DFA: pentru date fGn, ambele estimează $H$')))

chart(T('The ACF of Fractional Gaussian Noise', 'ACF a zgomotului gaussian fracționar'), 'sfm_ch11_fgn_acf', 'SFM_ch11_fbm_fgn', [
    T('Left: sample and theoretical ACF for $H = 0.8$ and $H = 0.2$ (as in SFEfgnacf); $\\rho(1) = @{fgn.8.r1}$ and $@{fgn.2.r1}$',
      'Stînga: ACF de selecție și teoretică pentru $H = 0.8$ și $H = 0.2$ (ca în SFEfgnacf); $\\rho(1) = @{fgn.8.r1}$ și $@{fgn.2.r1}$'),
    T('Right, log scales: hyperbolic decay (a straight line) against the exponential decay of an AR(1) with the same $\\rho(1)$',
      'Dreapta, scale logaritmice: scădere hiperbolică (o dreaptă) față de scăderea exponențială a unui AR(1) cu același $\\rho(1)$'),
    T('The sample ACF of a long-memory series lies below the theoretical one: estimating the mean removes part of the slow component',
      'ACF de selecție a unei serii cu memorie lungă este sub cea teoretică: estimarea mediei elimină o parte din componenta lentă')],
    h='0.48\\textheight')

D.frame(T('ARFIMA(0,$d$,0): Fractional Differencing', 'ARFIMA(0,$d$,0): diferențierea fracționară'), items(
    (T('\\textbf{ARFIMA}(0,$d$,0) \\refGJ; \\refHosking: $(1 - L)^d X_t = \\varepsilon_t$, $\\varepsilon_t$ i.i.d.\\ $(0, \\sigma^2)$, $L$ the lag operator ($L X_t = X_{t-1}$)',
       '\\textbf{ARFIMA}(0,$d$,0) \\refGJ; \\refHosking: $(1 - L)^d X_t = \\varepsilon_t$, $\\varepsilon_t$ i.i.d.\\ $(0, \\sigma^2)$, $L$ operatorul lag ($L X_t = X_{t-1}$)'),
     [T('ARFIMA: autoregressive fractionally integrated moving average; $d = 0$: white noise, $d = 1$: random walk (Chapter 7)',
        'ARFIMA: autoregressive fractionally integrated moving average, model autoregresiv fracționar integrat cu medie mobilă; $d = 0$: zgomot alb, $d = 1$: mers aleator (Capitolul 7)')]),
    (T('Binomial expansion \\refFHH, 14.2: $(1 - L)^d = \\sum_{j \\ge 0} \\pi_j L^j$, $\\pi_0 = 1$, $\\pi_j = \\pi_{j-1}\\,(j - 1 - d)/j$',
       'Dezvoltarea binomială \\refFHH, 14.2: $(1 - L)^d = \\sum_{j \\ge 0} \\pi_j L^j$, $\\pi_0 = 1$, $\\pi_j = \\pi_{j-1}\\,(j - 1 - d)/j$'),
     [T('$X_t = \\varepsilon_t - \\sum_{j \\ge 1}\\pi_j X_{t-j}$: the whole past matters, with slowly decaying weights', '$X_t = \\varepsilon_t - \\sum_{j \\ge 1}\\pi_j X_{t-j}$: contează tot trecutul, cu ponderi care scad lent'),
      T('MA($\\infty$) form: $X_t = \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$, $\\psi_0 = 1$, $\\psi_j = \\psi_{j-1}\\,(j - 1 + d)/j$', 'forma MA($\\infty$): $X_t = \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$, $\\psi_0 = 1$, $\\psi_j = \\psi_{j-1}\\,(j - 1 + d)/j$')]),
    (T('Autocorrelations: $\\rho(k) = \\dfrac{\\Gamma(1 - d)\\,\\Gamma(k + d)}{\\Gamma(d)\\,\\Gamma(k + 1 - d)}$, i.e.\\ $\\rho(k) = \\rho(k - 1)\\,\\dfrac{k - 1 + d}{k - d}$',
       'Autocorelațiile: $\\rho(k) = \\dfrac{\\Gamma(1 - d)\\,\\Gamma(k + d)}{\\Gamma(d)\\,\\Gamma(k + 1 - d)}$, adică $\\rho(k) = \\rho(k - 1)\\,\\dfrac{k - 1 + d}{k - d}$'),
     [T('$\\Gamma(\\cdot)$: the gamma function, $\\Gamma(n) = (n - 1)!$ for integers; $\\rho(1) = d/(1 - d)$', '$\\Gamma(\\cdot)$: funcția gamma, $\\Gamma(n) = (n - 1)!$ pentru numere întregi; $\\rho(1) = d/(1 - d)$'),
      T('for large $k$, $\\rho(k) \\sim C k^{2d - 1}$: long memory for $0 < d < 1/2$', 'pentru $k$ mare, $\\rho(k) \\sim C k^{2d - 1}$: memorie lungă pentru $0 < d < 1/2$')])), 'footnotesize')

D.frame(T('The Memory Parameter $d$', 'Parametrul de memorie $d$'), table(
    'lllll', T('$d$ & memory & mean reversion & variance & ACF', '$d$ & memorie & revenire la medie & varianță & ACF'),
    [T('$(-0.5, 0)$ & anti-persistent & yes & finite & hyperbolic, negative', '$(-0{,}5; 0)$ & antipersistentă & da & finită & hiperbolică, negativă'),
     T('$0$ & short & yes & finite & exponential (ARMA)', '$0$ & scurtă & da & finită & exponențială (ARMA)'),
     T('$(0, 0.5)$ & long, stationary & yes & finite & hyperbolic', '$(0; 0{,}5)$ & lungă, staționară & da & finită & hiperbolică'),
     T('$[0.5, 1)$ & long, non-stationary & yes & infinite & hyperbolic', '$[0{,}5; 1)$ & lungă, nestaționară & da & infinită & hiperbolică'),
     T('$1$ & unit root & no & infinite & does not decay', '$1$ & rădăcină unitară & nu & infinită & nu scade')],
    size='footnotesize') + items(
    T('Source: \\refFHH, Table 14.1; $H = d + 1/2$ for the stationary range', 'Sursa: \\refFHH, tabelul 14.1; $H = d + 1/2$ pe domeniul staționar'),
    T('ARFIMA($p$,$d$,$q$): ARMA($p$,$q$) dynamics for the short run, $d$ for the long run \\refFHH, 14.6.1', 'ARFIMA($p$,$d$,$q$): dinamica ARMA($p$,$q$) pentru termen scurt, $d$ pentru termen lung \\refFHH, 14.6.1'),
    T('Fractional integration fills the gap between the stationary $d = 0$ and the unit root $d = 1$ of Chapter 7', 'Integrarea fracționară acoperă spațiul dintre cazul staționar $d = 0$ și rădăcina unitară $d = 1$ din Capitolul 7')))

chart(T('ARFIMA(0,$d$,0) with $d = 0.4$ and $d = -0.4$', 'ARFIMA(0,$d$,0) cu $d = 0.4$ și $d = -0.4$'), 'sfm_ch11_arfima', 'SFM_ch11_arfima', [
    T('Left: 1000 simulated values (as in SFEarfima); right: sample ACF (20000 values) and theoretical ACF',
      'Stînga: 1000 de valori simulate (ca în SFEarfima); dreapta: ACF de selecție (20000 de valori) și ACF teoretică'),
    T('$d = 0.4$: $\\rho(1) = @{af.p.rho1}$, $\\rho(10) = @{af.p.rho10}$, $\\rho(50) = @{af.p.rho50}$: slow waves around the mean; $d = -0.4$: $\\rho(1) = @{af.n.rho1}$, then almost zero',
      '$d = 0.4$: $\\rho(1) = @{af.p.rho1}$, $\\rho(10) = @{af.p.rho10}$, $\\rho(50) = @{af.p.rho50}$: valuri lente în jurul mediei; $d = -0.4$: $\\rho(1) = @{af.n.rho1}$, apoi aproape zero')],
    h='0.52\\textheight')

D.frame(T('Worked Example: ARFIMA(0, 0.3, 0)', 'Exemplu rezolvat: ARFIMA(0; 0,3; 0)'), items(
    (T('Weights of $(1 - L)^{0.3}$: $\\pi_1 = -d = @{af3.pi1}$, $\\pi_2 = \\pi_1(1 - d)/2 = @{af3.pi2}$', 'Ponderile lui $(1 - L)^{0.3}$: $\\pi_1 = -d = @{af3.pi1}$, $\\pi_2 = \\pi_1(1 - d)/2 = @{af3.pi2}$'),
     [T('MA weights: $\\psi_1 = d = @{af3.psi1}$, $\\psi_2 = \\psi_1(1 + d)/2 = @{af3.psi2}$', 'ponderile MA: $\\psi_1 = d = @{af3.psi1}$, $\\psi_2 = \\psi_1(1 + d)/2 = @{af3.psi2}$')]),
    (T('Autocorrelations', 'Autocorelațiile'),
     [T('$\\rho(1) = 0.3/0.7 = @{af3.r1}$; $\\rho(2) = \\rho(1) \\times 1.3/1.7 = @{af3.r2}$', '$\\rho(1) = 0.3/0.7 = @{af3.r1}$; $\\rho(2) = \\rho(1) \\times 1.3/1.7 = @{af3.r2}$'),
      T('$\\rho(10) = @{af3.r10}$, $\\rho(100) = @{af3.r100}$', '$\\rho(10) = @{af3.r10}$, $\\rho(100) = @{af3.r100}$')]),
    T('AR(1) with $\\phi = @{af3.r1}$: $\\rho(10) = @{af3.ar10}$', 'AR(1) cu $\\phi = @{af3.r1}$: $\\rho(10) = @{af3.ar10}$'),
    T('The autocorrelation falls below 0.05 after @{af3.firstar} lags for the AR(1) and after @{af3.first} lags for ARFIMA; $H = d + 0.5 = 0.8$',
      'Autocorelația scade sub 0,05 după @{af3.firstar} laguri pentru AR(1) și după @{af3.first} de laguri pentru ARFIMA; $H = d + 0.5 = 0.8$')))

# =============================================================================
# 5. ESTIMAREA LUI H ȘI d
# =============================================================================
D.section('Estimators of Long Memory', 'Estimatori ai memoriei lungi')

D.frame(T('Four Tools', 'Patru instrumente'), table(
    TB + 'p{2.0cm}' + TB + 'p{1.7cm}' + TB + 'p{3.4cm}' + TB + 'p{3.6cm}',
    T('\\textbf{Tool} & \\textbf{Domain} & \\textbf{What it gives} & \\textbf{Weak point}', '\\textbf{Instrument} & \\textbf{Domeniu} & \\textbf{Rezultat} & \\textbf{Punct slab}'),
    [T('R/S & time & $\\hat H$, slope of a log-log plot & biased upwards in small samples; sensitive to short memory', 'R/S & timp & $\\hat H$, panta unui grafic log-log & deplasat în sus în eșantioane mici; sensibil la memoria scurtă'),
     T("Lo's modified R/S & time & a test of short memory & low power against long memory", 'R/S modificat (Lo) & timp & un test al memoriei scurte & putere mică față de memoria lungă'),
     T('DFA & time & $\\hat H$ (exponent $\\alpha$) & needs block sizes; affected by breaks', 'DFA & timp & $\\hat H$ (exponentul $\\alpha$) & necesită mărimi de bloc; afectat de rupturi'),
     T('GPH & frequency & $\\hat d$ with a standard error & large variance; choice of bandwidth $m$', 'GPH & frecvență & $\\hat d$ cu eroare standard & varianță mare; alegerea lățimii de bandă $m$')],
    size='scriptsize') + items(
    T('Other estimators: the local Whittle (Gaussian semiparametric) estimator \\refRob, wavelets, exact maximum likelihood for ARFIMA (\\refFHH, 14.5)',
      'Alți estimatori: estimatorul Whittle local (gaussian semiparametric) \\refRob, wavelets, verosimilitatea maximă exactă pentru ARFIMA (\\refFHH, 14.5)'),
    T('Good practice: report several estimators, each with its uncertainty', 'Bună practică: raportăm mai mulți estimatori, fiecare cu incertitudinea lui')))

D.frame(T('Detrended Fluctuation Analysis (DFA)', 'Analiza fluctuațiilor fără tendință (DFA)'), items(
    (T('\\textbf{DFA} \\refPeng: designed for signals with slowly changing trends', '\\textbf{DFA} \\refPeng: concepută pentru semnale cu tendințe care se schimbă lent'),
     [T('1. profile $Y_k = \\sum_{j \\le k}(x_j - \\bar x)$', '1. profilul $Y_k = \\sum_{j \\le k}(x_j - \\bar x)$'),
      T('2. split $Y$ into blocks of size $n$; in each block, fit a straight line by OLS (linear detrending)', '2. împărțim $Y$ în blocuri de mărime $n$; în fiecare bloc estimăm o dreaptă prin OLS (eliminarea tendinței liniare)'),
      T('3. $F(n)$ = root mean square of the deviations from the lines, over all blocks', '3. $F(n)$ = rădăcina mediei pătratelor abaterilor de la drepte, pe toate blocurile'),
      T('4. $F(n) \\propto n^{\\alpha}$: the slope of $\\ln F(n)$ on $\\ln n$ is $\\hat\\alpha$', '4. $F(n) \\propto n^{\\alpha}$: panta lui $\\ln F(n)$ în funcție de $\\ln n$ este $\\hat\\alpha$')]),
    (T('Interpretation', 'Interpretare'),
     [T('for fGn, $\\alpha = H$: $\\alpha = 0.5$ white noise, $0.5 < \\alpha < 1$ long memory; for fBm, $\\alpha = H + 1$', 'pentru fGn, $\\alpha = H$: $\\alpha = 0.5$ zgomot alb, $0.5 < \\alpha < 1$ memorie lungă; pentru fBm, $\\alpha = H + 1$'),
      T('the local line removes linear trends inside each block; $F(n)$ is unreliable for $n$ below about 10 \\refKan', 'dreapta locală elimină tendințele liniare din fiecare bloc; $F(n)$ nu este de încredere pentru $n$ sub circa 10 \\refKan')])))

chart(T('R/S and DFA: S\\&P 500 since 2000', 'R/S și DFA: S\\&P 500 din 2000'), 'sfm_ch11_rs_dfa', 'SFM_ch11_rs_estimators', [
    T('@{rd.ns} block sizes from 10 to @{rd.nmax} days; dashed: mean curve of i.i.d.\\ Normal series of the same length',
      '@{rd.ns} de mărimi de bloc, de la 10 la @{rd.nmax} zile; linia întreruptă: curba medie a seriilor i.i.d.\\ Normale de aceeași lungime'),
    T('Returns: R/S slope $@{rd.rs_r}$ (i.i.d.\\ series: $@{rd.rs_iid}$), DFA $@{rd.dfa_r}$ (i.i.d.: $@{rd.dfa_iid}$): no long memory',
      'Randamente: panta R/S $@{rd.rs_r}$ (serii i.i.d.: $@{rd.rs_iid}$), DFA $@{rd.dfa_r}$ (i.i.d.: $@{rd.dfa_iid}$): fără memorie lungă'),
    T('Absolute returns: R/S $@{rd.rs_abs}$, DFA $@{rd.dfa_abs}$: strong long memory in volatility (Section 7)',
      'Randamente absolute: R/S $@{rd.rs_abs}$, DFA $@{rd.dfa_abs}$: memorie lungă puternică a volatilității (secțiunea 7)')],
    h='0.48\\textheight')

D.frame(T('The GPH Log-Periodogram Regression', 'Regresia GPH pe log-periodogramă'), items(
    (T('\\textbf{Periodogram} at the Fourier frequencies $\\lambda_j = 2\\pi j/T$: $I(\\lambda_j) = \\dfrac{1}{2\\pi T}\\left|\\sum_{t=1}^{T} x_t e^{-i\\lambda_j t}\\right|^2$',
       '\\textbf{Periodograma} la frecvențele Fourier $\\lambda_j = 2\\pi j/T$: $I(\\lambda_j) = \\dfrac{1}{2\\pi T}\\left|\\sum_{t=1}^{T} x_t e^{-i\\lambda_j t}\\right|^2$'),
     [T('$T$: the number of observations; $i$: the imaginary unit; $|\\cdot|^2$: the squared modulus', '$T$: numărul de observații; $i$: unitatea imaginară; $|\\cdot|^2$: pătratul modulului'),
      T('it estimates the spectral density, which behaves like $C\\lambda^{-2d}$ near zero', 'estimează densitatea spectrală, care se comportă ca $C\\lambda^{-2d}$ lîngă zero')]),
    (T('\\textbf{GPH} \\refGPH; \\refFHH, 14.5.2: $\\ln I(\\lambda_j) = c - d\\,\\ln\\!\\left(4\\sin^2(\\lambda_j/2)\\right) + e_j$, $j = 1, \\dots, m$',
       '\\textbf{GPH} \\refGPH; \\refFHH, 14.5.2: $\\ln I(\\lambda_j) = c - d\\,\\ln\\!\\left(4\\sin^2(\\lambda_j/2)\\right) + e_j$, $j = 1, \\dots, m$'),
     [T('$m$: the number of low frequencies used; $c$: a constant; $e_j$: the error', '$m$: numărul de frecvențe joase folosite; $c$: o constantă; $e_j$: eroarea'),
      T('$\\hat d$ = OLS slope on $-\\ln(4\\sin^2(\\lambda_j/2))$; $\\hat H = \\hat d + 0.5$', '$\\hat d$ = panta OLS în funcție de $-\\ln(4\\sin^2(\\lambda_j/2))$; $\\hat H = \\hat d + 0.5$'),
      T('asymptotic standard error $\\pi/\\sqrt{24m}$; the bandwidth $m = \\lfloor T^{0.5}\\rfloor$', 'eroarea standard asimptotică $\\pi/\\sqrt{24m}$; lățimea de bandă $m = \\lfloor T^{0.5}\\rfloor$')]),
    T('Trade-off: a small $m$ keeps only the long-run frequencies (small bias, large variance); a large $m$ lets short-run dynamics in (bias)',
      'Compromis: un $m$ mic păstrează doar frecvențele de termen lung (deplasare mică, varianță mare); un $m$ mare lasă să intre dinamica de termen scurt (deplasare)')))

chart(T('GPH: S\\&P 500 Returns and Absolute Returns', 'GPH: randamentele și randamentele absolute S\\&P 500'), 'sfm_ch11_gph', 'SFM_ch11_rs_estimators', [
    T('$m = @{gph.r.m}$ lowest frequencies; each point is one frequency', 'Se folosesc cele mai joase $m = @{gph.r.m}$ frecvențe; fiecare punct este o frecvență'),
    T('Returns: $\\hat d = @{gph.r.d}$ (SE $@{gph.r.se}$): not different from 0; absolute returns: $\\hat d = @{gph.abs.d}$ (SE $@{gph.abs.se}$), $\\hat H = @{gph.abs.H}$',
      'Randamente: $\\hat d = @{gph.r.d}$ (SE $@{gph.r.se}$): nu diferă de 0; randamente absolute: $\\hat d = @{gph.abs.d}$ (SE $@{gph.abs.se}$), $\\hat H = @{gph.abs.H}$')],
    h='0.52\\textheight')

D.frame(T("Lo's Modified R/S Test", 'Testul R/S modificat al lui Lo'), items(
    T('Problem: short memory (e.g.\\ an AR(1)) also raises R/S, so classical R/S mistakes short memory for long memory', 'Problema: și memoria scurtă (de exemplu un AR(1)) mărește R/S, deci R/S clasic confundă memoria scurtă cu memoria lungă'),
    (T('\\refLo: $V_N(q) = \\dfrac{R_N}{\\sqrt{N}\\,\\hat\\sigma_N(q)}$, $\\hat\\sigma^2_N(q) = \\hat\\gamma_0 + 2\\sum_{j=1}^{q}\\left(1 - \\tfrac{j}{q + 1}\\right)\\hat\\gamma_j$',
       '\\refLo: $V_N(q) = \\dfrac{R_N}{\\sqrt{N}\\,\\hat\\sigma_N(q)}$, $\\hat\\sigma^2_N(q) = \\hat\\gamma_0 + 2\\sum_{j=1}^{q}\\left(1 - \\tfrac{j}{q + 1}\\right)\\hat\\gamma_j$'),
     [T('$N$: the number of returns; $R_N$: the range of the cumulative deviations over the whole sample', '$N$: numărul de randamente; $R_N$: amplitudinea abaterilor cumulate pe întregul eșantion'),
      T('$\\hat\\gamma_j$: sample autocovariances; $\\hat\\sigma^2_N(q)$: the Newey--West long-run variance \\refNW; $q = 0$: classical R/S', '$\\hat\\gamma_j$: autocovarianțele de selecție; $\\hat\\sigma^2_N(q)$: varianța pe termen lung Newey--West \\refNW; $q = 0$: R/S clasic'),
      T('$q = \\lfloor (3N/2)^{1/3}\\,(2\\hat\\rho_1/(1 - \\hat\\rho_1^2))^{2/3}\\rfloor$ ($\\hat\\rho_1$: first-order autocorrelation), data-dependent (Andrews\' rule, used by Lo)', '$q = \\lfloor (3N/2)^{1/3}\\,(2\\hat\\rho_1/(1 - \\hat\\rho_1^2))^{2/3}\\rfloor$ ($\\hat\\rho_1$: autocorelația de ordinul 1), dependent de date (regula lui Andrews, folosită de Lo)')]),
    (T('$H_0$: short memory; at 5\\%, reject if $V_N(q) \\notin [⁅0.809⁆, ⁅1.862⁆]$', '$H_0$: memorie scurtă; la 5\\% respingem dacă $V_N(q) \\notin [⁅0.809⁆, ⁅1.862⁆]$'),
     [T('Lo\'s finding: no long memory in US stock index returns once short memory is allowed for', 'Concluzia lui Lo: nu există memorie lungă în randamentele indicilor bursieri din SUA, odată ce se ține cont de memoria scurtă')])), 'footnotesize')

chart(T("Classical against Modified R/S: Size and Power", 'R/S clasic față de R/S modificat: mărimea și puterea testului'), 'sfm_ch11_lo_test', 'SFM_ch11_monte_carlo', [
    T('500 series of 1000 values. Left, AR(1) (short memory, $H_0$ true): with $\\phi = 0.5$ the classical test rejects in @{lo.ar05.c}\\% of cases, the modified test in @{lo.ar05.m}\\%',
      '500 de serii de 1000 de valori. Stînga, AR(1) (memorie scurtă, $H_0$ adevărată): pentru $\\phi = 0.5$, testul clasic respinge în @{lo.ar05.c}\\% din cazuri, testul modificat în @{lo.ar05.m}\\%'),
    T('Right, fGn (long memory if $H > 0.5$): with $H = 0.7$, rejection rates @{lo.h07.c}\\% and @{lo.h07.m}\\%: the correction costs power \\refTTW',
      'Dreapta, fGn (memorie lungă dacă $H > 0.5$): pentru $H = 0.7$, ratele de respingere sînt @{lo.h07.c}\\% și @{lo.h07.m}\\%: corecția costă putere \\refTTW')],
    h='0.50\\textheight')

D.frame(T('Small Samples and Monte Carlo Bands', 'Eșantioane mici și benzi Monte Carlo'), items(
    (T('Asymptotic results are unreliable for R/S and DFA at the sample sizes of finance; use simulation \\refWeron', 'Rezultatele asimptotice nu sînt de încredere pentru R/S și DFA la mărimile de eșantion din finanțe; folosim simularea \\refWeron'),
     [T('1. simulate many i.i.d.\\ Normal series with the same length $N$ as the data', '1. simulăm multe serii i.i.d.\\ Normale, de aceeași lungime $N$ ca datele'),
      T('2. apply exactly the same estimator (same block sizes, same bandwidth)', '2. aplicăm exact același estimator (aceleași mărimi de bloc, aceeași lățime de bandă)'),
      T('3. the 2.5\\% and 97.5\\% quantiles of $\\hat H$ form the 95\\% band under $H = 0.5$', '3. cuantilele de 2,5\\% și 97,5\\% ale lui $\\hat H$ formează banda de 95\\% în ipoteza $H = 0.5$')]),
    T('Reject $H = 0.5$ only if $\\hat H$ lies outside the band', 'Respingem $H = 0.5$ doar dacă $\\hat H$ se află în afara benzii'),
    T('i.i.d.\\ Normal is the simplest null; a fitted short-memory model (AR, GARCH) gives a stricter null (Section 6)', 'Seria i.i.d.\\ Normală este cea mai simplă ipoteză nulă; un model estimat cu memorie scurtă (AR, GARCH) dă o ipoteză nulă mai exigentă (secțiunea 6)')))

chart(T('Monte Carlo: Bias and Spread of the Estimators', 'Monte Carlo: deplasarea și dispersia estimatorilor'), 'sfm_ch11_mc_bias', 'SFM_ch11_monte_carlo', [
    T('500 i.i.d.\\ Normal series for each $N$; true $H = 0.5$', '500 de serii i.i.d.\\ Normale pentru fiecare $N$; valoarea adevărată este $H = 0.5$'),
    T('R/S: mean $@{mc.1000.rs.mean}$ at $N = 1000$ (biased upwards); DFA: $@{mc.1000.dfa.mean}$; GPH: $@{mc.1000.gph.mean}$, unbiased but wide',
      'R/S: media $@{mc.1000.rs.mean}$ la $N = 1000$ (deplasată în sus); DFA: $@{mc.1000.dfa.mean}$; GPH: $@{mc.1000.gph.mean}$, nedeplasat, dar cu dispersie mare'),
    T('Width of the 95\\% band at $N = 1000$: R/S $@{mc.1000.rs.w}$, DFA $@{mc.1000.dfa.w}$, GPH $@{mc.1000.gph.w}$; at $N = 250$: DFA $@{mc.250.dfa.w}$',
      'Lățimea benzii de 95\\% la $N = 1000$: R/S $@{mc.1000.rs.w}$, DFA $@{mc.1000.dfa.w}$, GPH $@{mc.1000.gph.w}$; la $N = 250$: DFA $@{mc.250.dfa.w}$')],
    h='0.50\\textheight')

D.frame(T('S\\&P 500: All Estimators', 'S\\&P 500: toți estimatorii'), table(
    'lrrrrr', T(' & R/S & DFA & GPH $\\hat d$ (SE) & Lo $V_N(q)$ & $q$', ' & R/S & DFA & GPH $\\hat d$ (SE) & Lo $V_N(q)$ & $q$'),
    [T('returns $r_t$', 'randamente $r_t$') + ' & $@{m.sp500.r.rs}$ & $@{m.sp500.r.dfa}$ & $@{m.sp500.r.d}$ ($@{m.sp500.r.dse}$) & $@{m.sp500.r.V}$ & @{m.sp500.r.q}',
     T('absolute returns $|r_t|$', 'randamente absolute $|r_t|$') + ' & $@{m.sp500.abs.rs}$ & $@{m.sp500.abs.dfa}$ & $@{m.sp500.abs.d}$ ($@{m.sp500.abs.dse}$) & $@{m.sp500.abs.V}$ & @{m.sp500.abs.q}',
     T('95\\% band, i.i.d.', 'banda de 95\\%, i.i.d.') + ' & $[@{m.sp500.mc.rs.lo}, @{m.sp500.mc.rs.hi}]$ & $[@{m.sp500.mc.dfa.lo}, @{m.sp500.mc.dfa.hi}]$ & $[@{m.sp500.mc.gph.lo}, @{m.sp500.mc.gph.hi}]$ & $[⁅0.809⁆, ⁅1.862⁆]$ & '],
    size='footnotesize') + items(
    T('$N = @{m.sp500.n}$ daily returns, 2000 to @{end}; GPH band given for $\\hat H = \\hat d + 0.5$', '$N = @{m.sp500.n}$ randamente zilnice, din 2000 pînă la @{end}; banda GPH este dată pentru $\\hat H = \\hat d + 0.5$'),
    T('Returns: every estimate inside its band, $V_N(q)$ inside $[⁅0.809⁆, ⁅1.862⁆]$', 'Randamente: fiecare estimare se află în banda ei, $V_N(q)$ se află în $[⁅0.809⁆, ⁅1.862⁆]$'),
    T('Absolute returns: every estimate far above its band, Lo rejects short memory', 'Randamente absolute: fiecare estimare este mult peste banda ei, iar testul lui Lo respinge memoria scurtă')))

D.frame(T('Interpretation: S\\&P 500', 'Interpretarea rezultatelor: S\\&P 500'), items(
    (T('The direction of returns has no long memory since 2000', 'Direcția randamentelor nu are memorie lungă din 2000 încoace'),
     [T('consistent with the weak form of the EMH (Chapter 7) and with \\refLo', 'în acord cu forma slabă a EMH (Capitolul 7) și cu \\refLo'),
      T('R/S alone ($@{m.sp500.r.rs}$) would look like ``$H > 0.5$\'\'; its band starts at $@{m.sp500.mc.rs.lo}$', 'doar R/S ($@{m.sp500.r.rs}$) ar părea să indice „$H > 0.5$”; banda lui începe de la $@{m.sp500.mc.rs.lo}$')]),
    (T('The size of returns has long memory: $\\hat d = @{m.sp500.abs.d}$ for $|r_t|$', 'Mărimea randamentelor are memorie lungă: $\\hat d = @{m.sp500.abs.d}$ pentru $|r_t|$'),
     [T('the risk of a calm or turbulent period persists for months: the long-run version of volatility clustering', 'riscul unei perioade liniștite sau agitate persistă luni de zile: versiunea pe termen lung a volatility clustering')]),
    T('Before concluding ``long memory\'\', rule out spurious long memory: next section', 'Înainte de a concluziona că există memorie lungă, eliminăm memoria lungă aparentă: secțiunea următoare')))

# =============================================================================
# 6. MEMORIE LUNGĂ APARENTĂ
# =============================================================================
D.section('Spurious Long Memory', 'Memorie lungă aparentă')

D.frame(T('Breaks Look Like Memory', 'Rupturile structurale par memorie'), items(
    (T('A series with a few shifts in the mean has a sample ACF that decays slowly, like long memory', 'O serie cu cîteva schimbări ale mediei are o ACF de selecție care scade lent, ca la memoria lungă'),
     [T('two regimes with means $\\mu_1 \\neq \\mu_2$: every pair of observations from the same regime shares the deviation $\\mu_i - \\bar x$', 'două regimuri cu mediile $\\mu_1 \\neq \\mu_2$: orice pereche de observații din același regim are în comun abaterea $\\mu_i - \\bar x$')]),
    (T('Landmark results', 'Rezultate de referință'),
     [T('\\refDI: rare regime switches produce estimates of $d$ indistinguishable from true long memory', '\\refDI: schimbările rare de regim produc estimări ale lui $d$ care nu se pot deosebi de memoria lungă reală'),
      T('\\refGH: part of the long memory of S\\&P 500 absolute returns comes from occasional breaks', '\\refGH: o parte din memoria lungă a randamentelor absolute S\\&P 500 provine din rupturi ocazionale'),
      T('\\refLS: long memory in squared S\\&P 500 returns survives the checks; in returns, it is not found', '\\refLS: memoria lungă a pătratelor randamentelor S\\&P 500 rezistă verificărilor; în randamente nu apare')]),
    T('Tests of true against spurious long memory: \\refFHH, 14.4.3', 'Teste ale memoriei lungi reale față de cea aparentă: \\refFHH, 14.4.3')))

D.frame(T('The Nile in 1898: a Break, Not Only Memory', 'Nilul în 1898: o ruptură, nu doar memorie'), cols(items(
    T('\\refCobb: the annual flow of the Nile at Aswan, 1871--1970, changes its mean after 1898', '\\refCobb: debitul anual al Nilului la Aswan, 1871--1970, își schimbă media după 1898'),
    T('Mean flow: @{nile.m0} before, @{nile.m1} after (in $10^8$ m$^3$), a drop of @{nile.drop}\\%', 'Debitul mediu: @{nile.m0} înainte, @{nile.m1} după (în $10^8$ m$^3$), o scădere de @{nile.drop}\\%'),
    T('The change coincides with the building of the Aswan Low Dam (1898--1902)', 'Schimbarea coincide cu construcția barajului Aswan Low Dam (1898--1902)'),
    T('A river record with a break is the same problem as a market with a change of regime', 'O serie hidrologică cu o ruptură ridică aceeași problemă ca o piață care își schimbă regimul')),
    ph('aswan', T('The Aswan Low Dam on the Nile', 'Barajul Aswan Low Dam de pe Nil'), h='0.36\\textheight'),
    wl='0.55', wr='0.41'), 'footnotesize')

chart(T('The Nile: R/S before and after Removing the Break', 'Nilul: R/S înainte și după eliminarea rupturii'), 'sfm_ch11_nile', 'SFM_ch11_spurious', [
    T('Raw series: $\\hat H_{R/S} = @{nile.H}$, DFA $@{nile.D}$; after removing the two regime means: $@{nile.Ha}$ and $@{nile.Da}$',
      'Seria brută: $\\hat H_{R/S} = @{nile.H}$, DFA $@{nile.D}$; după eliminarea celor două medii de regim: $@{nile.Ha}$ și $@{nile.Da}$'),
    T('With $N = 100$ the i.i.d.\\ band for R/S is wide, $[@{nile.q025}, @{nile.q975}]$, centred at $@{nile.nm}$: most of the apparent memory was the break',
      'Cu $N = 100$, banda i.i.d.\\ pentru R/S este largă, $[@{nile.q025}, @{nile.q975}]$, centrată în $@{nile.nm}$: cea mai mare parte a memoriei aparente era ruptura')],
    h='0.50\\textheight')

chart(T('Simulated Spurious Long Memory', 'Memorie lungă aparentă în simulări'), 'sfm_ch11_spurious', 'SFM_ch11_spurious', [
    T('@{sp.reps} series of @{sp.n} values. Left: i.i.d.\\ $N(0,1)$ with a mean shift at the middle; a shift of 0.5 standard deviations gives GPH $\\hat H = @{br.05.gph}$, DFA $@{br.05.dfa}$',
      '@{sp.reps} serii de @{sp.n} valori. Stînga: i.i.d.\\ $N(0,1)$ cu o schimbare a mediei la mijloc; o schimbare de 0,5 abateri standard dă GPH $\\hat H = @{br.05.gph}$, DFA $@{br.05.dfa}$'),
    T('Right: GARCH(1,1), a short-memory model (Chapter 9); for $|r_t|$, DFA gives $@{ga.095.adfa}$ at $\\alpha + \\beta = 0.95$ and $@{ga.099.adfa}$ at $0.99$; the returns stay at $@{ga.099.rdfa}$',
      'Dreapta: GARCH(1,1), un model cu memorie scurtă (Capitolul 9); pentru $|r_t|$, DFA dă $@{ga.095.adfa}$ la $\\alpha + \\beta = 0.95$ și $@{ga.099.adfa}$ la $0.99$; randamentele rămîn la $@{ga.099.rdfa}$')],
    h='0.48\\textheight')

D.frame(T('Volatility Clustering Mimics Long Memory', 'Volatility clustering imită memoria lungă'), items(
    (T('In GARCH(1,1), the ACF of $r_t^2$ decays like $(\\alpha + \\beta)^k$: short memory', 'În GARCH(1,1), ACF a lui $r_t^2$ scade ca $(\\alpha + \\beta)^k$: memorie scurtă'),
     [T('with $\\alpha + \\beta$ close to 1, the decay is so slow that a sample of a few thousand days cannot separate it from $k^{2d - 1}$', 'cu $\\alpha + \\beta$ apropiat de 1, scăderea este atît de lentă încît un eșantion de cîteva mii de zile nu o poate deosebi de $k^{2d - 1}$')]),
    T('Consequence: a high $\\hat H$ for $|r_t|$ is compatible with a persistent GARCH; it is not by itself proof of fractional dynamics',
      'Consecință: un $\\hat H$ mare pentru $|r_t|$ este compatibil cu un GARCH persistent; nu este, singur, o dovadă de dinamică fracționară'),
    T('For returns, GARCH does not create memory: the estimates stay at 0.5 (right panel)', 'Pentru randamente, GARCH nu creează memorie: estimările rămîn la 0,5 (panoul din dreapta)'),
    T('Better null hypothesis: simulate the fitted GARCH and compare $\\hat H$ of the data with the GARCH band', 'O ipoteză nulă mai bună: simulăm GARCH-ul estimat și comparăm $\\hat H$ al datelor cu banda GARCH')))

D.frame(T('Checks against Spurious Long Memory', 'Verificări împotriva memoriei lungi aparente'), items(
    T('\\textbf{Shuffle test}: permute the series at random; the distribution stays, the order disappears; $\\hat H$ should fall to 0.5',
      '\\textbf{Testul permutării}: permutăm aleator seria; distribuția rămîne, ordinea dispare; $\\hat H$ ar trebui să scadă la 0,5'),
    T('\\textbf{Subsamples}: estimate before and after known breaks (2008, 2020); true long memory appears in each subsample',
      '\\textbf{Subeșantioane}: estimăm înainte și după rupturile cunoscute (2008, 2020); memoria lungă reală apare în fiecare subeșantion'),
    T('\\textbf{Several estimators and bandwidths}: R/S, DFA, GPH with $m = T^{0.5}$ and $T^{0.6}$ should agree', '\\textbf{Mai mulți estimatori și lățimi de bandă}: R/S, DFA, GPH cu $m = T^{0.5}$ și $T^{0.6}$ ar trebui să concorde'),
    T("\\textbf{Lo's modified R/S}: removes the effect of short memory", '\\textbf{R/S modificat al lui Lo}: elimină efectul memoriei scurte'),
    T('\\textbf{Model-based null}: Monte Carlo bands from a fitted AR or GARCH instead of i.i.d.\\ noise', '\\textbf{Ipoteză nulă pe baza unui model}: benzi Monte Carlo dintr-un AR sau GARCH estimat, în locul zgomotului i.i.d.')))

# =============================================================================
# 7. MEMORIA LUNGĂ A VOLATILITĂȚII
# =============================================================================
D.section('Long Memory in Volatility', 'Memoria lungă a volatilității')

chart(T('S\\&P 500: the ACF of $r_t$, $|r_t|$ and $r_t^2$', 'S\\&P 500: ACF pentru $r_t$, $|r_t|$ și $r_t^2$'), 'sfm_ch11_vol_acf', 'SFM_ch11_volatility_memory', [
    T('$|r_t|$: $\\rho(1) = @{va.a.1}$, $\\rho(50) = @{va.a.50}$, $\\rho(100) = @{va.a.100}$; @{va.a.npos} of the first 250 lags above the band $@{va.band}$',
      '$|r_t|$: $\\rho(1) = @{va.a.1}$, $\\rho(50) = @{va.a.50}$, $\\rho(100) = @{va.a.100}$; @{va.a.npos} dintre primele 250 de laguri sînt peste banda $@{va.band}$'),
    T('$r_t^2$: $\\rho(1) = @{va.s.1}$, $\\rho(50) = @{va.s.50}$; $r_t$: $\\rho(1) = @{va.r.1}$, then inside the band',
      '$r_t^2$: $\\rho(1) = @{va.s.1}$, $\\rho(50) = @{va.s.50}$; $r_t$: $\\rho(1) = @{va.r.1}$, apoi în interiorul benzii'),
    T('\\refDGE: $|r_t|$ is more persistent than $r_t^2$, which is more affected by single extreme days', '\\refDGE: $|r_t|$ este mai persistent decît $r_t^2$, care este mai afectat de zilele extreme izolate')],
    h='0.48\\textheight')

D.frame(T('Six Markets: Returns and Absolute Returns', 'Șase piețe: randamente și randamente absolute'), table(
    'lrrrrrrrr', T(' & $N$ & \\multicolumn{3}{c}{returns $r_t$} & \\multicolumn{3}{c}{absolute returns $|r_t|$} & DFA band', ' & $N$ & \\multicolumn{3}{c}{randamente $r_t$} & \\multicolumn{3}{c}{randamente absolute $|r_t|$} & banda DFA') +
    ' \\\\\n & & R/S & DFA & GPH $\\hat d$ & DFA & GPH $\\hat d$ & Lo $V$ & ',
    [f'{SHORT[k]} & @{{m.{k}.n}} & $@{{m.{k}.r.rs}}$ & $@{{m.{k}.r.dfa}}$ & $@{{m.{k}.r.d}}$ & $@{{m.{k}.abs.dfa}}$ & $@{{m.{k}.abs.d}}$ & $@{{m.{k}.abs.V}}$ & $[@{{m.{k}.mc.dfa.lo}}, @{{m.{k}.mc.dfa.hi}}]$' for k in ASSETS],
    size='scriptsize') + items(
    T('Daily data to @{end}: S\\&P 500, DAX, BET since 2000, Bitcoin since 2014, TLV and SNP since 2010; standard error of GPH about $@{m.sp500.r.dse}$--$@{m.tlv.r.dse}$',
      'Date zilnice pînă la @{end}: S\\&P 500, DAX, BET din 2000, Bitcoin din 2014, TLV și SNP din 2010; eroarea standard GPH este de circa $@{m.sp500.r.dse}$--$@{m.tlv.r.dse}$'),
    T('BET returns: DFA $@{m.bet.r.dfa}$ above its band, GPH $\\hat d = @{m.bet.r.d}$, Lo $V = @{m.bet.r.V}$ rejects short memory', 'Randamentele BET: DFA $@{m.bet.r.dfa}$ peste banda sa, GPH $\\hat d = @{m.bet.r.d}$, testul lui Lo ($V = @{m.bet.r.V}$) respinge memoria scurtă'),
    T('Absolute returns: $\\hat d$ from $@{m.absd.min}$ to $@{m.absd.max}$, Lo rejects for all six', 'Randamente absolute: $\\hat d$ între $@{m.absd.min}$ și $@{m.absd.max}$; testul lui Lo respinge pentru toate cele șase serii')), 'footnotesize')

chart(T('Six Markets: DFA Exponents and the Shuffle Test', 'Șase piețe: exponenții DFA și testul permutării'), 'sfm_ch11_markets', 'SFM_ch11_volatility_memory', [
    T('Shaded: 95\\% band of i.i.d.\\ series of the same length; circles: returns; squares: $|r_t|$; crosses: $|r_t|$ after a random shuffle',
      'Zona colorată: banda de 95\\% a seriilor i.i.d.\\ de aceeași lungime; cercuri: randamente; pătrate: $|r_t|$; cruciulițe: $|r_t|$ după o permutare aleatoare'),
    T('$|r_t|$: DFA from $@{m.absdfa.min}$ to $@{m.absdfa.max}$; after shuffling: from $@{m.shuf.min}$ to $@{m.shuf.max}$, inside the bands: the memory lies in the order of the days',
      '$|r_t|$: DFA între $@{m.absdfa.min}$ și $@{m.absdfa.max}$; după permutare: între $@{m.shuf.min}$ și $@{m.shuf.max}$, în interiorul benzilor: memoria se află în ordinea zilelor')],
    h='0.50\\textheight')

D.frame(T('Interpretation: What Is Predictable?', 'Interpretarea rezultatelor: ce este previzibil?'), items(
    (T('\\textbf{Returns}: no long memory in the S\\&P 500, DAX, TLV and SNP; Bitcoin at the edge of its band', '\\textbf{Randamentele}: fără memorie lungă la S\\&P 500, DAX, TLV și SNP; Bitcoin la limita benzii'),
     [T('BET: persistence, as in the variance-ratio tests of Chapter 7; partly the thin trading of the early years (Section 8)', 'BET: persistență, ca în testele raportului varianțelor din Capitolul 7; în parte, efectul tranzacționării reduse din primii ani (secțiunea 8)')]),
    (T('\\textbf{Volatility}: long memory everywhere, strongest in the large equity indices', '\\textbf{Volatilitatea}: memorie lungă peste tot, cea mai puternică la marii indici de acțiuni'),
     [T('the risk regime of today is informative for months: useful for VaR (value at risk) and option pricing', 'regimul de risc de azi este informativ luni de zile: util pentru VaR (value at risk, valoarea expusă la risc) și pentru evaluarea opțiunilor')]),
    T('Weak-form efficiency (unpredictable direction) and predictable risk coexist: the FMH adds a reason, investors with different horizons react to volatility on different time scales',
      'Eficiența în formă slabă (direcție imprevizibilă) și riscul previzibil coexistă: FMH adaugă o explicație, investitorii cu orizonturi diferite reacționează la volatilitate pe scale de timp diferite')))

D.frame(T('Models for Long Memory in Volatility', 'Modele pentru memoria lungă a volatilității'), items(
    (T('\\textbf{FIGARCH} (fractionally integrated GARCH) \\refBBM; \\refFHH, 14.6.2: $(1 - L)^d$ applied to $\\varepsilon_t^2$', '\\textbf{FIGARCH} (fractionally integrated GARCH, GARCH fracționar integrat) \\refBBM; \\refFHH, 14.6.2: $(1 - L)^d$ aplicat lui $\\varepsilon_t^2$'),
     [T('shocks to volatility decay hyperbolically; $d = 0$ gives GARCH, $d = 1$ gives IGARCH (integrated GARCH, Chapter 9)', 'șocurile de volatilitate scad hiperbolic; $d = 0$ dă GARCH, $d = 1$ dă IGARCH (integrated GARCH, GARCH integrat, Capitolul 9)')]),
    (T('\\textbf{Heterogeneous horizons}, the FMH idea in volatility models', '\\textbf{Orizonturi eterogene}, ideea FMH în modelele de volatilitate'),
     [T('\\refMuller: market components with different time resolutions (intraday traders to long-term investors)', '\\refMuller: componente ale pieței cu rezoluții de timp diferite (de la traderii intraday la investitorii pe termen lung)'),
      T('\\refAB: many volatility components with different persistence add up to long memory', '\\refAB: multe componente de volatilitate cu persistențe diferite se însumează și dau memorie lungă'),
      T('\\refCorsi: HAR (heterogeneous autoregressive) model, tomorrow\'s volatility on the daily, weekly and monthly volatility: three horizons, approximate long memory',
        '\\refCorsi: modelul HAR (heterogeneous autoregressive, autoregresiv heterogen), volatilitatea de mîine în funcție de volatilitatea zilnică, săptămînală și lunară: trei orizonturi, memorie lungă aproximativă')])))

# =============================================================================
# 8. EXPONENȚI HURST PE FERESTRE MOBILE
# =============================================================================
D.section('Rolling Hurst Exponents', 'Exponenți Hurst pe ferestre mobile')

D.frame(T('Does Efficiency Change over Time?', 'Se schimbă eficiența în timp?'), items(
    (T('The AMH and the FMH both predict that market memory changes over time', 'AMH și FMH prezic, amîndouă, că memoria pieței se schimbă în timp'),
     [T('\\refCT: rolling Hurst exponents of emerging stock markets, to test whether they become more efficient', '\\refCT: exponenți Hurst pe ferestre mobile pentru piețe emergente, pentru a testa dacă devin mai eficiente'),
      T('\\refBar: rolling DFA on Bitcoin returns and volatility', '\\refBar: DFA pe ferestre mobile pentru randamentele și volatilitatea Bitcoin')]),
    (T('Design used here', 'Abordarea folosită aici'),
     [T('windows of 1000 observations (about four years of trading days), moved by 21 observations (about one month)', 'ferestre de 1000 de observații (circa patru ani de zile de tranzacționare), mutate cu 21 de observații (circa o lună)'),
      T('DFA on $r_t$ and on $|r_t|$; each estimate dated at the last day of its window, so it uses only past data', 'DFA pe $r_t$ și pe $|r_t|$; fiecare estimare este datată la ultima zi a ferestrei, deci folosește doar date trecute'),
      T('95\\% Monte Carlo band for $N = 1000$: $[@{rl.lo}, @{rl.hi}]$', 'banda Monte Carlo de 95\\% pentru $N = 1000$: $[@{rl.lo}, @{rl.hi}]$')])))

chart(T('Rolling DFA Exponents: S\\&P 500, BET, Bitcoin', 'Exponenți DFA pe ferestre mobile: S\\&P 500, BET, Bitcoin'), 'sfm_ch11_rolling', 'SFM_ch11_rolling_hurst', [
    T('Top to bottom: S\\&P 500, BET, Bitcoin; coloured line: $r_t$; purple line: $|r_t|$; shaded: i.i.d.\\ band', 'De sus în jos: S\\&P 500, BET, Bitcoin; linia colorată: $r_t$; linia mov: $|r_t|$; zona colorată: banda i.i.d.'),
    T('$|r_t|$ always far above the band; its exponent rises in and after crises (2008--2009, 2020)', '$|r_t|$ mereu mult peste bandă; exponentul lui crește în timpul și după crize (2008--2009, 2020)')],
    h='0.60\\textheight')

D.frame(T('Interpretation: Rolling Exponents', 'Interpretarea exponenților pe ferestre mobile'), items(
    (T('S\\&P 500: @{rl.sp500.nw} windows; $\\hat H$ of returns between $@{rl.sp500.min}$ and $@{rl.sp500.max}$; @{rl.sp500.below}\\% of windows below the band, none above',
       'S\\&P 500: @{rl.sp500.nw} ferestre; $\\hat H$ al randamentelor între $@{rl.sp500.min}$ și $@{rl.sp500.max}$; @{rl.sp500.below}\\% dintre ferestre sub bandă, niciuna peste'),
     [T('the minimum, @{rl.sp500.dmin}, follows the 2008 crash: daily reversals (anti-persistence) in a panic', 'minimul, la @{rl.sp500.dmin}, vine după crahul din 2008: inversări zilnice (antipersistență) în panică')]),
    (T('BET: @{rl.bet.above}\\% of windows above the band; mean $\\hat H$ $@{rl.bet.first_third}$ in the first third of the windows, $@{rl.bet.last_third}$ in the last third',
       'BET: @{rl.bet.above}\\% dintre ferestre peste bandă; $\\hat H$ mediu $@{rl.bet.first_third}$ în prima treime a ferestrelor, $@{rl.bet.last_third}$ în ultima treime'),
     [T('the maximum, @{rl.bet.dmax}; a market that becomes more efficient as liquidity grows', 'maximul este atins la @{rl.bet.dmax}; apoi piața devine mai eficientă, pe măsură ce lichiditatea crește')]),
    T('Bitcoin: windows from @{rl.btc.e0}; $\\hat H$ between $@{rl.btc.min}$ and $@{rl.btc.max}$, @{rl.btc.above}\\% of windows above the band',
      'Bitcoin: ferestre din @{rl.btc.e0}; $\\hat H$ între $@{rl.btc.min}$ și $@{rl.btc.max}$, @{rl.btc.above}\\% dintre ferestre peste bandă')))

D.frame(T('Case Study: Is Bitcoin Efficient?', 'Studiu de caz: este Bitcoin eficient?'), cols(items(
    (T('\\refUrq: several tests, among them R/S, on Bitcoin 2010--2016', '\\refUrq: mai multe teste, printre care R/S, pe Bitcoin, 2010--2016'),
     [T('inefficient over the full sample, moving towards efficiency in the later years', 'ineficient pe întregul eșantion, cu o evoluție spre eficiență în ultimii ani')]),
    T('\\refNC: the same data after a power transformation of the returns: consistent with weak-form efficiency', '\\refNC: aceleași date după o transformare de tip putere a randamentelor: în acord cu eficiența în formă slabă'),
    T('\\refBar: rolling DFA: persistent returns until 2014, close to $H = 0.5$ afterwards; long memory in volatility throughout',
      '\\refBar: DFA pe ferestre mobile: randamente persistente pînă în 2014, apropiate de $H = 0.5$ după aceea; memorie lungă a volatilității pe toată perioada'),
    T('Our windows start in 2017 and agree: returns near the band, $|r_t|$ far above it', 'Ferestrele noastre încep în 2017 și confirmă: randamente lîngă bandă, $|r_t|$ mult peste ea')),
    ph('btc', T('A Bitcoin cash machine in Prague (2016)', 'Un bancomat Bitcoin la Praga (2016)'), h='0.40\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('Pitfalls of Rolling Estimates', 'Capcanele estimărilor pe ferestre mobile'), items(
    (T('\\textbf{Multiple testing}: @{rl.sp500.nw} windows at 5\\% each give about @{rl.sp500.fa} windows outside the band by chance', '\\textbf{Testarea multiplă}: @{rl.sp500.nw} ferestre, fiecare la 5\\%, dau din întîmplare circa @{rl.sp500.fa} ferestre în afara benzii'),
     [T('look for long runs outside the band, not for isolated windows', 'căutăm perioade lungi în afara benzii, nu ferestre izolate')]),
    T('\\textbf{Overlap}: consecutive windows share 979 of their 1000 observations; the path is smooth by construction', '\\textbf{Suprapunerea}: ferestrele consecutive au în comun 979 dintre cele 1000 de observații; traiectoria este netedă prin construcție'),
    T('\\textbf{Window length}: shorter windows react faster but have wider bands (DFA, $N = 250$: width $@{mc.250.dfa.w}$)', '\\textbf{Lungimea ferestrei}: ferestrele mai scurte reacționează mai repede, dar au benzi mai largi (DFA, $N = 250$: lățimea $@{mc.250.dfa.w}$)'),
    T('\\textbf{Volatility regimes}: a crisis inside a window shifts the estimate for four years', '\\textbf{Regimurile de volatilitate}: o criză din interiorul unei ferestre deplasează estimarea timp de patru ani'),
    T('\\textbf{Dating}: date each estimate at the end of its window, never at the middle (no look-ahead)', '\\textbf{Datarea}: datăm fiecare estimare la sfîrșitul ferestrei, niciodată la mijloc (fără informație din viitor)')))

# =============================================================================
# 9. MEMORIA LUNGĂ ȘI RISCUL
# =============================================================================
D.section('Long Memory and Risk', 'Memoria lungă și riscul')

D.frame(T('Scaling Risk with $h^H$', 'Scalarea riscului cu $h^H$'), items(
    (T('If returns are self-similar: $h$-day VaR 1\\% $= -z_{0.01}\\,\\sigma\\,h^H$, with $z_{0.01} = -2.326$ (Normal case, mean 0)', 'Dacă randamentele sînt autosimilare: VaR 1\\% pe $h$ zile $= -z_{0.01}\\,\\sigma\\,h^H$, cu $z_{0.01} = -2.326$ (cazul Normal, media 0)'),
     [T('$z_{0.01}$: the 1\\% quantile of $N(0,1)$; $\\sigma$: the daily volatility; $h^H$ replaces the factor $\\sqrt{h}$', '$z_{0.01}$: cuantila de 1\\% a lui $N(0,1)$; $\\sigma$: volatilitatea zilnică; $h^H$ înlocuiește factorul $\\sqrt{h}$'),
      T('daily volatility $\\sigma = 1.2\\%$: one-day VaR 1\\% $= @{vs.var1}\\%$', 'volatilitatea zilnică $\\sigma = 1.2\\%$: VaR 1\\% pe o zi $= @{vs.var1}\\%$')]),
    (T('Ten days', 'Zece zile'),
     [T('$H = 0.45$: factor $@{vs.f10.45}$, VaR $@{vs.v10.45}\\%$; $H = 0.5$: $@{vs.f10.50}$, $@{vs.v10.50}\\%$; $H = 0.55$: $@{vs.f10.55}$, $@{vs.v10.55}\\%$',
        '$H = 0.45$: factorul $@{vs.f10.45}$, VaR $@{vs.v10.45}\\%$; $H = 0.5$: $@{vs.f10.50}$, $@{vs.v10.50}\\%$; $H = 0.55$: $@{vs.f10.55}$, $@{vs.v10.55}\\%$'),
      T('$H = 0.55$ instead of 0.5: @{vs.diff}\\% more risk at 10 days, @{vs.diff250}\\% more at 250 days', '$H = 0.55$ în loc de 0,5: cu @{vs.diff}\\% mai mult risc la 10 zile și cu @{vs.diff250}\\% mai mult la 250 de zile')]),
    T('Data: the 10-day ratio is $@{sc.sp500.r10}$ for the S\\&P 500 and $@{sc.bet.r10}$ for the BET, against $\\sqrt{10} = 3.16$: the $\\sqrt{h}$ rule overstates S\\&P 500 risk and understates BET risk (Chapter 10)',
      'Datele: raportul pe 10 zile este $@{sc.sp500.r10}$ pentru S\\&P 500 și $@{sc.bet.r10}$ pentru BET, față de $\\sqrt{10} = 3.16$: regula $\\sqrt{h}$ supraestimează riscul S\\&P 500 și subestimează riscul BET (Capitolul 10)'),
    T('Pricing: with fBm prices and $H \\neq 1/2$, a frictionless market admits arbitrage \\refRogers; long memory enters option models through volatility instead',
      'Evaluarea: cu prețuri fBm și $H \\neq 1/2$, o piață fără costuri de tranzacționare admite arbitraj \\refRogers; de aceea memoria lungă intră în modelele de opțiuni prin volatilitate')), 'footnotesize')

# =============================================================================
# 10. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Does a change in the Hurst exponent come before a crisis, or only after it?}', '\\textbf{Apare schimbarea exponentului Hurst înaintea unei crize sau doar după ea?}'),
     [T('the FMH predicts a collapse of horizons; in our data, $\\hat H$ of $|r_t|$ rises in and after 2008 and 2020', 'FMH prezice o prăbușire a orizonturilor; în datele noastre, $\\hat H$ al lui $|r_t|$ crește în timpul și după 2008 și 2020'),
      T('a rolling window contains the crisis only after it starts: is there any signal before?', 'o fereastră mobilă conține criza doar după ce aceasta începe: există vreun semnal înainte?')]),
    T('Earlier evidence: \\refKrisB\\ (horizons in 2008); \\refCT\\ (emerging markets)', 'Dovezi anterioare: \\refKrisB\\ (orizonturile în 2008); \\refCT\\ (piețe emergente)'),
    T('Why it is open: few crises, overlapping windows, volatility regimes, many possible estimators', 'Întrebarea rămîne deschisă: puține crize, ferestre suprapuse, regimuri de volatilitate, mulți estimatori posibili'),
    T('AI tools can speed up such a study, but its results must still be checked \\refWang', 'Instrumentele AI pot accelera un astfel de studiu, dar rezultatele lui trebuie verificate \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on Hurst exponents as early-warning indicators, with their data and estimators', '\\textbf{Literatura}: o listă a studiilor despre exponenții Hurst ca indicatori de avertizare timpurie, cu datele și estimatorii folosiți'),
    T('\\textbf{Code}: a first draft of rolling R/S, DFA and GPH with Monte Carlo bands', '\\textbf{Cod}: o primă versiune pentru R/S, DFA și GPH pe ferestre mobile, cu benzi Monte Carlo'),
    T('\\textbf{Robustness}: other window lengths, wavelet estimators, a GARCH-based null, other crises (2011, 2022)', '\\textbf{Robustețe}: alte lungimi ale ferestrei, estimatori wavelet, o ipoteză nulă pe baza unui GARCH, alte crize (2011, 2022)'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that computes the DFA exponent of daily S\\&P 500 absolute log returns on rolling windows of 500 and 1000 days, dated at the window end, adds a 95\\% band from 500 simulated GARCH(1,1) paths, and marks the start of the 2008 and 2020 crises.}',
        '\\aiprompt{Write Python code that computes the DFA exponent of daily S\\&P 500 absolute log returns on rolling windows of 500 and 1000 days, dated at the window end, adds a 95\\% band from 500 simulated GARCH(1,1) paths, and marks the start of the 2008 and 2020 crises.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Dating: an estimate dated at the middle of its window uses future data', 'Datarea: o estimare datată la mijlocul ferestrei folosește date din viitor'),
    T('Conversions: $d = H - 0.5$, not $H + 0.5$; DFA on prices gives $H + 1$, not $H$', 'Conversiile: $d = H - 0.5$, nu $H + 0.5$; DFA aplicat prețurilor dă $H + 1$, nu $H$'),
    T('Bands: every $\\hat H$ needs a band for its own $N$; R/S is biased upwards', 'Benzile: fiecare $\\hat H$ are nevoie de o bandă pentru propriul $N$; R/S este deplasat în sus'),
    T('Returns or volatility: memory in $|r_t|$ does not make the direction of returns predictable', 'Randamente sau volatilitate: memoria lui $|r_t|$ nu face previzibilă direcția randamentelor'),
    T('GPH bandwidth: $m = T/2$ (all frequencies) is not ``more accurate\'\'; report $T^{0.5}$ and $T^{0.6}$', 'Lățimea de bandă GPH: $m = T/2$ (toate frecvențele) nu este „mai precis”; raportăm $T^{0.5}$ și $T^{0.6}$'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: does the rolling Hurst exponent of volatility rise before crises on the BVB and in Bitcoin?', '\\textbf{Întrebarea}: crește exponentul Hurst al volatilității, pe ferestre mobile, înaintea crizelor de pe BVB și din piața Bitcoin?'),
     [T('data: BET, BET-TR, TLV, SNP and Bitcoin (EODHD), daily', 'date: BET, BET-TR, TLV, SNP și Bitcoin (EODHD), zilnice')]),
    (T('Steps', 'Pași'),
     [T('rolling DFA and GPH of $r_t$ and $|r_t|$, windows of 500 and 1000 days, dated at the end', 'DFA și GPH pe ferestre mobile pentru $r_t$ și $|r_t|$, ferestre de 500 și 1000 de zile, datate la sfîrșit'),
      T('bands from i.i.d.\\ and from fitted GARCH(1,1) simulations', 'benzi din simulări i.i.d.\\ și din simulări ale GARCH(1,1) estimat'),
      T('event windows around 2008, 2020 and 2022: the exponent in the 60 days before and after', 'ferestre de eveniment în jurul anilor 2008, 2020 și 2022: exponentul în cele 60 de zile dinainte și de după')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('FMH: stability needs investors with many horizons; crises are collapses of horizons', 'FMH: stabilitatea are nevoie de investitori cu multe orizonturi; crizele sînt prăbușiri ale orizonturilor'),
    T('$H$ measures memory and scaling: $H = 0.5$ none, $H > 0.5$ persistence, $H < 0.5$ anti-persistence; $d = H - 0.5$', '$H$ măsoară memoria și scalarea: $H = 0.5$ fără memorie, $H > 0.5$ persistență, $H < 0.5$ antipersistență; $d = H - 0.5$'),
    T('R/S, DFA, GPH and Lo\'s test, always with a Monte Carlo band for the same $N$', 'R/S, DFA, GPH și testul lui Lo, mereu cu o bandă Monte Carlo pentru același $N$'),
    T('Breaks and persistent GARCH volatility create spurious long memory: shuffle, split, compare', 'Rupturile și volatilitatea GARCH persistentă creează memorie lungă aparentă: permutăm, împărțim eșantionul, comparăm'),
    T('Returns: no long memory in large markets (BET: persistence); volatility: long memory everywhere', 'Randamentele: fără memorie lungă pe piețele mari (BET: persistență); volatilitatea: memorie lungă peste tot'),
    T('Multi-day risk scales like $h^H$', 'Riscul pe mai multe zile se scalează ca $h^H$')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Self-similarity', 'Autosimilaritate') + ' & $X_{ct} \\overset{d}{=} c^H X_t$; \\quad $\\mathrm{sd}(r^{(h)}) = h^H \\mathrm{sd}(r)$',
     'R/S & $(R/S)_n = \\left[\\max_k Y_k - \\min_k Y_k\\right]/S_n$, \\quad $Y_k = \\sum_{j \\le k}(x_j - \\bar x)$; \\quad $(R/S)_n \\propto n^H$',
     'fGn & $\\rho(k) = \\tfrac12(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}) \\approx H(2H - 1)k^{2H - 2}$',
     'ARFIMA(0,$d$,0) & $(1 - L)^d X_t = \\varepsilon_t$, \\quad $\\rho(k) = \\rho(k - 1)\\frac{k - 1 + d}{k - d}$, \\quad $d = H - 0.5$',
     'DFA & $F(n) \\propto n^{\\alpha}$, \\quad $\\alpha = H$ ' + T('for fGn', 'pentru fGn'),
     'GPH & $\\ln I(\\lambda_j) = c - d\\ln(4\\sin^2(\\lambda_j/2)) + e_j$, \\quad $j \\le m = T^{0.5}$, \\quad SE $= \\pi/\\sqrt{24m}$',
     'Lo & $V_N(q) = R_N/(\\sqrt{N}\\hat\\sigma_N(q))$; ' + T('reject at 5\\% outside', 'respingem la 5\\% în afara') + ' $[⁅0.809⁆, ⁅1.862⁆]$',
     'VaR 1\\% & $-z_{0.01}\\,\\sigma\\,h^H$'],
    size='small') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: an R/S estimate on 1000 i.i.d.\\ returns is $\\hat H = 0.56$. Is this evidence of long memory?', '\\textbf{Întrebare}: o estimare R/S pe 1000 de randamente i.i.d.\\ este $\\hat H = 0{,}56$. Este o dovadă de memorie lungă?'),
     [T('\\textbf{Answer}: no: the i.i.d.\\ band for $N = 1000$ is $[@{mc.1000.rs.lo}, @{mc.1000.rs.hi}]$, centred near $@{mc.1000.rs.mean}$', '\\textbf{Răspuns}: nu: banda i.i.d.\\ pentru $N = 1000$ este $[@{mc.1000.rs.lo}, @{mc.1000.rs.hi}]$, centrată în jur de $@{mc.1000.rs.mean}$')]),
    (T('\\textbf{Question}: GPH gives $\\hat d = 0.35$ for $|r_t|$. What is $\\hat H$?', '\\textbf{Întrebare}: GPH dă $\\hat d = 0{,}35$ pentru $|r_t|$. Cît este $\\hat H$?'),
     [T('\\textbf{Answer}: $\\hat H = \\hat d + 0.5 = 0.85$: long memory in volatility, not in returns', '\\textbf{Răspuns}: $\\hat H = \\hat d + 0.5 = 0.85$: memorie lungă a volatilității, nu a randamentelor')]),
    (T('\\textbf{Question}: after a random shuffle of $|r_t|$, DFA falls from 0.9 to 0.5. What does this show?', '\\textbf{Întrebare}: după o permutare aleatoare a lui $|r_t|$, DFA scade de la 0,9 la 0,5. Ce arată acest lucru?'),
     [T('\\textbf{Answer}: the memory comes from the order of the days, not from the distribution (heavy tails)', '\\textbf{Răspuns}: memoria provine din ordinea zilelor, nu din distribuție (cozi groase)')]),
    T('Next: Chapter 12, scoring models', 'Urmează: Capitolul 12, modele de scoring')), 'footnotesize')

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
