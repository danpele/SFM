r"""
build_chapter3.py -- Capitolul 3 (Distribuții α-stabile), EN + RO dintr-o singură sursă
======================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_03/ch3_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter3_alpha_stable_distributions.tex
  RO/Cursuri/capitol3_distributii_alfa_stabile.tex
Rulare:
  python3 Quantlets/Ch_03/generate_all_charts.py
  python3 latex/build_chapter3.py && python3 latex/sfm_build.py compile 3
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo, ql   # noqa: E402
from ch3_common import ALPHA, ASSETS, BIB, NAMES, REFS, T, load, ro_tuples, values   # noqa: E402

N = load()
V = values(N)
D = Deck(3, 'lecture', refs=REFS, macros=ALPHA)
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_03'


def side(title, fig, folder, bullets, w=0.56, size='footnotesize', h='0.78\\textheight'):
    """Chart on the left, bullets on the right, Quantlet link below."""
    left = f'\\begin{{center}}\n\\includegraphics[width=\\linewidth,height={h},keepaspectratio]{{{fig}.pdf}}\n\\end{{center}}'
    D.frame(title, cols(left, items(*bullets), wl=f'{w:.2f}', wr=f'{0.96 - w:.2f}') + '\n' + ql(folder), size)


def chart(title, fig, folder, bullets, h='0.64\\textheight', size='footnotesize'):
    D.chart(title, fig, folder, bullets, height=h, size=size)


def pic(file, cap_en, cap_ro, url, cred_en, cred_ro, h='0.50\\textheight'):
    return photo(file, T(cap_en, cap_ro), url, T(cred_en, cred_ro), h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
for a in (2.0, 1.5, 1.0):
    V.put(f'ex.c.{a:g}', (3 ** a + 4 ** a) ** (1 / a), 2)
for a in (2.0, 1.7, 1.5):
    V.put(f'ex.h.{a:g}', 20 ** (1 / a), 2)
for a in (2.0, 1.5, 1.0):
    V.put(f'ex.pf.{a:g}', 10 ** (1 / a - 1), 2)
# funcția caracteristică a S(1.5, 0.3, 1, 0) în t = 1
tan15 = math.tan(math.pi * 0.75)
z1 = complex(math.exp(-1) * math.cos(0.3), -math.exp(-1) * math.sin(0.3))
V.put('cf.re', z1.real, 3)
V.put('cf.im', -z1.imag, 3)
V.put('cf.abs', math.exp(-1), 3)
V.put('cf.shift', 0.3 * tan15, 1)
# CMS: U = 0.8, W = 1.2, alpha = 1.5
Vv = math.pi * 0.3
X = math.sin(1.5 * Vv) / math.cos(Vv) ** (1 / 1.5) * (math.cos(-0.5 * Vv) / 1.2) ** (-0.5 / 1.5)
V.put('cms.V', Vv, 3)
V.put('cms.X', X, 3)
V.put('cms.sin', math.sin(1.5 * Vv), 3)
V.put('cms.cos', math.cos(Vv) ** (1 / 1.5), 3)
V.put('cms.last', (math.cos(-0.5 * Vv) / 1.2) ** (-0.5 / 1.5), 3)
# McCulloch, BET: interpolare în Tabelul III (nu_beta = 0) între 3.2 și 3.5
nuB = N['fits']['bet']['mcculloch']['nu_alpha']
V.put('mc.bet.interp', 1.484 + (nuB - 3.2) / 0.3 * (1.391 - 1.484), 3)
V.put('mc.bet.iqr', N['fits']['bet']['mcculloch']['q75'] - N['fits']['bet']['mcculloch']['q25'], 2)
V.put('mc.bet.range', N['fits']['bet']['mcculloch']['q95'] - N['fits']['bet']['mcculloch']['q05'], 2)
# tail ratio 2^(-alpha)
V.put('tr.17', 2 ** -1.7, 2)
V.put('tr.15', 2 ** -1.5, 2)
# variance check
V.put('chk.c', 2 ** (1 / 1.5), 2)
V.put('chk.h', 20 ** (1 / 1.5) / 20 ** 0.5, 2)
V.put('ex.d0.add', -0.2 * 0.6 * math.tan(0.85 * math.pi), 3)
V.put('ex.d0', 0.05 - 0.2 * 0.6 * math.tan(0.85 * math.pi), 3)
V.put('sqrt2', math.sqrt(2), 2)
fits = N['fits']
best_alpha = min(fits, key=lambda k: fits[k]['stable']['alpha'])
V.raw('low.alpha', NAMES[best_alpha])

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: which probability laws can describe daily returns and also their weekly and monthly sums?',
       '\\textbf{Întrebarea}: ce legi de probabilitate pot descrie randamentele zilnice și, în același timp, sumele lor săptămînale și lunare?'),
     [T('Chapter 2: daily returns have heavier tails than the Normal distribution',
        'Capitolul 2: randamentele zilnice au cozi mai groase decît distribuția Normală'),
      T('log returns add up over time (Chapter 1): the law of a sum matters', 'randamentele logaritmice se adună în timp (Capitolul 1): legea unei sume contează')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('Mandelbrot and the cotton prices: where the idea comes from', 'Mandelbrot și prețurile bumbacului: de unde vine ideea'),
      T('stability under summation and the generalised central limit theorem', 'stabilitatea la adunare și teorema limită centrală generalizată'),
      T('parameters, characteristic function, tails and moments', 'parametri, funcția caracteristică, cozi și momente'),
      T('simulation and estimation; fits to BET, S\\&P 500, DAX and Bitcoin', 'simulare și estimare; aplicații pe BET, S\\&P 500, DAX și Bitcoin'),
      T('the critique: do returns really have infinite variance?', 'critica: au randamentele chiar varianță infinită?')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('State the stability property and compute the scale of a sum of stable variables',
      'Enunțați proprietatea de stabilitate și calculați scala unei sume de variabile stabile'),
    T('Explain the generalised central limit theorem and when sums do not approach the Normal distribution',
      'Explicați teorema limită centrală generalizată și cînd sumele nu se apropie de distribuția Normală'),
    T('Read the four parameters $(\\alpha, \\beta, \\gamma, \\delta)$ and convert between the S0 and S1 parameterisations',
      'Interpretați cei patru parametri $(\\alpha, \\beta, \\gamma, \\delta)$ și faceți conversia între parametrizările S0 și S1'),
    T('Say which moments exist for a given $\\alpha$ and compare tail probabilities with the Normal distribution',
      'Stabiliți ce momente există pentru un $\\alpha$ dat și comparați probabilitățile din cozi cu cele ale distribuției Normale'),
    T('Simulate stable variables and estimate their parameters by quantiles and by maximum likelihood',
      'Simulați variabile stabile și estimați parametrii prin cuantile și prin verosimilitate maximă'),
    T('Judge, on real returns, where the stable model works and where it fails',
      'Evaluați, pe randamente reale, unde funcționează modelul stabil și unde eșuează')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Main reference: \\refNolan, Ch.~1 (definitions, parameterisations, tails) and Ch.~4 (estimation)',
       'Referința principală: \\refNolan, cap.~1 (definiții, parametrizări, cozi) și cap.~4 (estimare)'),
     [T('short survey with code: \\refBHW; updated in \\refBMW', 'sinteză scurtă, cu cod: \\refBHW; actualizată în \\refBMW')]),
    T('Textbook: \\refFHH, Ch.~3 (probability) and Ch.~18 (heavy tails and extreme risks)',
      'Manual: \\refFHH, cap.~3 (probabilități) și cap.~18 (cozi groase și riscuri extreme)'),
    T('The original papers: \\refMandelbrot, \\refFama', 'Lucrările originale: \\refMandelbrot, \\refFama'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_03}',
       'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_03}'),
     [T('ported from the Quantlet sim\\_stable; each chart links to the Quantlet that draws it',
        'adaptate după Quantlet-ul sim\\_stable; fiecare grafic are un link către Quantlet-ul care îl generează')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter3_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter3_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Stable Distribution}{https://quantinar.com/course/980/stable-distribution}',
      'Curs video: \\quantinar{Stable Distribution}{https://quantinar.com/course/980/stable-distribution}')))

# =============================================================================
# 1. MANDELBROT ȘI BUMBACUL
# =============================================================================
D.section('Mandelbrot and the Cotton Prices', 'Mandelbrot și prețurile bumbacului')

D.frame(T('Cotton: an Old Market with Long Price Records', 'Bumbacul: o piață veche, cu serii lungi de prețuri'), cols(items(
    (T('In the 19th century cotton was one of the most traded commodities in the world',
       'În secolul al XIX-lea, bumbacul era una dintre cele mai tranzacționate mărfuri din lume'),
     [T('exchanges in New Orleans, New York and Liverpool quoted prices every day',
        'bursele din New Orleans, New York și Liverpool cotau prețuri în fiecare zi'),
      T('the result: some of the longest daily and monthly price series in economics',
        'rezultatul: unele dintre cele mai lungi serii zilnice și lunare de prețuri din economie')]),
    (T('In the early 1960s these series reached Benoit Mandelbrot, then at IBM',
       'La începutul anilor 1960, aceste serii au ajuns la Benoit Mandelbrot, pe atunci la IBM'),
     [T('he looked at the \\textbf{changes} of the logarithm of the price, i.e.\\ log returns',
        'el a studiat \\textbf{variațiile} logaritmului prețului, adică randamentele logaritmice')]),
    T('His paper \\refMandelbrot\\ started the study of heavy tails in finance',
      'Lucrarea lui, \\refMandelbrot, a deschis studiul cozilor groase în finanțe')),
    pic('ch3_cotton_office.jpg', 'Edgar Degas, A Cotton Office in New Orleans (1873)', 'Edgar Degas, Un birou de bumbac din New Orleans (1873)',
        'https://commons.wikimedia.org/wiki/File:Edgar_Germain_Hilaire_Degas_016.jpg',
        'Image: Edgar Degas (1873); public domain; Wikimedia Commons', 'Imagine: Edgar Degas (1873); domeniu public; Wikimedia Commons',
        h='0.42\\textheight'), wl='0.52', wr='0.44'))

D.frame(T('What Mandelbrot Saw', 'Observația lui Mandelbrot'), cols(items(
    (T('\\textbf{Too many large changes}: far more big moves than the Normal distribution allows',
       '\\textbf{Prea multe variații mari}: mult mai multe variații mari decît permite distribuția Normală'),
     [T('the sample variance did not settle as the sample grew', 'varianța de selecție nu se stabiliza pe măsură ce creștea eșantionul')]),
    (T('\\textbf{The same shape at every horizon}: daily and monthly changes looked alike after rescaling',
       '\\textbf{Aceeași formă la orice orizont}: variațiile zilnice și cele lunare arătau la fel după rescalare'),
     [T('a sum of many daily changes kept the shape of one daily change', 'o sumă de multe variații zilnice păstra forma unei singure variații zilnice')]),
    (T('\\textbf{His proposal}: stable Paretian laws with $\\alpha \\approx 1.7$ \\refMandelbrot',
       '\\textbf{Propunerea lui}: legi stabile de tip Pareto, cu $\\alpha \\approx 1.7$ \\refMandelbrot'),
     [T('``Paretian\'\': the tails decay like a power of $x$, as in Pareto\'s law of incomes',
        '„de tip Pareto”: cozile scad ca o putere a lui $x$, ca în legea veniturilor a lui Pareto'),
      T('consequence: the variance of price changes would be infinite', 'consecință: varianța variațiilor de preț ar fi infinită')])),
    pic('ch3_mandelbrot.jpg', 'Benoit Mandelbrot (1924--2010)', 'Benoit Mandelbrot (1924--2010)',
        'https://commons.wikimedia.org/wiki/File:Benoit_Mandelbrot_mg_1804-d.jpg',
        'Photo: Rama (2007); CC BY-SA 2.0 fr; Wikimedia Commons', 'Foto: Rama (2007); CC BY-SA 2.0 fr; Wikimedia Commons',
        h='0.45\\textheight'), wl='0.62', wr='0.34'))

D.frame(T('Fama Tests the Idea on Stocks', 'Fama testează ideea pe acțiuni'), cols(items(
    (T('\\refFama: daily returns of the 30 stocks of the Dow Jones Industrial Average',
       '\\refFama: randamentele zilnice ale celor 30 de acțiuni din Dow Jones Industrial Average'),
     [T('more large returns than the Normal distribution predicts, for every stock',
        'mai multe randamente mari decît permite distribuția Normală, pentru fiecare acțiune'),
      T('estimates of $\\alpha$ below 2: support for Mandelbrot\'s hypothesis', 'estimări ale lui $\\alpha$ sub 2: sprijin pentru ipoteza lui Mandelbrot')]),
    (T('Estimation methods for symmetric stable laws followed', 'Au urmat metode de estimare pentru legile stabile simetrice'),
     [T('\\refFamaRoll, \\refFamaRollB: quantile-based estimates and tables', '\\refFamaRoll, \\refFamaRollB: estimări pe baza cuantilelor și tabele')]),
    T('Within ten years the evidence started to turn: see the critique at the end of the chapter',
      'În mai puțin de zece ani, dovezile au început să se schimbe: vedeți critica de la finalul capitolului')),
    pic('ch3_fama.jpg', 'Eugene Fama at the Nobel Prize ceremony, 2013', 'Eugene Fama la ceremonia Premiului Nobel, 2013',
        'https://commons.wikimedia.org/wiki/File:Eugene_Fama_at_Nobel_Prize,_2013.jpg',
        'Photo: Bengt Nyman (2013); CC BY 2.0; Wikimedia Commons', 'Foto: Bengt Nyman (2013); CC BY 2.0; Wikimedia Commons',
        h='0.45\\textheight'), wl='0.62', wr='0.34'))

D.frame(T('Why the Law of a Sum Matters', 'Importanța legii unei sume'), items(
    (T('A monthly log return is the sum of about 21 daily log returns', 'Un randament logaritmic lunar este suma a circa 21 de randamente logaritmice zilnice'),
     [T('a model for daily returns implies a model for weekly, monthly and yearly returns',
        'un model pentru randamentele zilnice implică un model pentru randamentele săptămînale, lunare și anuale')]),
    (T('The Normal distribution keeps its shape under summation (Chapter 2)', 'Distribuția Normală își păstrează forma la adunare (Capitolul 2)'),
     [T('but it has thin tails: it misses the large daily moves', 'dar are cozi subțiri: nu surprinde variațiile zilnice mari')]),
    T('\\textbf{Question for the room}: the Student-t distribution has heavy tails; is a sum of 21 independent Student-t variables again Student-t?',
      '\\textbf{Întrebare pentru sală}: distribuția Student-t are cozi groase; este suma a 21 de variabile Student-t independente tot Student-t?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('no: the Student-t family is not closed under summation', 'nu: familia Student-t nu este închisă la adunare'),
      T('if its variance is finite, the sum moves towards the Normal distribution (the central limit theorem)',
        'dacă varianța ei este finită, suma se apropie de distribuția Normală (teorema limită centrală)')]),
    T('\\textbf{Goal}: find every law whose sums keep the same shape', '\\textbf{Obiectivul}: identificarea tuturor legilor ale căror sume își păstrează forma')))

# =============================================================================
# 2. STABILITATEA LA ADUNARE
# =============================================================================
D.section('Stability under Summation', 'Stabilitatea la adunare')

D.frame(T('Definition: Stable Laws', 'Definiție: legile stabile'), items(
    (T('$X$ is \\textbf{stable} if, for $X_1, X_2$ independent copies of $X$ and any $a, b > 0$, there are $c > 0$ and $d$ with',
       '$X$ este \\textbf{stabilă} dacă, pentru $X_1, X_2$ copii independente ale lui $X$ și orice $a, b > 0$, există $c > 0$ și $d$ astfel încît'),
     [T('$aX_1 + bX_2 \\overset{d}{=} cX + d$, where $\\overset{d}{=}$ means ``has the same distribution as\'\'',
        '$aX_1 + bX_2 \\overset{d}{=} cX + d$, unde $\\overset{d}{=}$ înseamnă „are aceeași distribuție ca”'),
      T('$a, b$: the weights of the two copies; $c$: the scale and $d$: the shift of the result', '$a, b$: ponderile celor două copii; $c$: scala și $d$: deplasarea rezultatului'),
      T('\\textbf{strictly stable} if $d = 0$ for all $a, b$', '\\textbf{strict stabilă} dacă $d = 0$ pentru orice $a, b$')]),
    (T('In words: a weighted sum of two copies has the same shape as one copy', 'Altfel spus: o sumă ponderată a două copii are aceeași formă ca o copie'),
     [T('only the scale ($c$) and the location ($d$) change', 'se schimbă doar scala ($c$) și poziția ($d$)')]),
    (T('By induction, for $n$ independent copies: $X_1 + \\dots + X_n \\overset{d}{=} c_n X + d_n$',
       'Prin inducție, pentru $n$ copii independente: $X_1 + \\dots + X_n \\overset{d}{=} c_n X + d_n$'),
     [T('the only possible scale factor is $c_n = n^{1/\\alpha}$ with $0 < \\alpha \\le 2$ \\refNolan',
        'singurul factor de scală posibil este $c_n = n^{1/\\alpha}$, cu $0 < \\alpha \\le 2$ \\refNolan')]),
    T('$\\alpha$ is called the \\textbf{index of stability} (also: characteristic exponent, tail index)',
      '$\\alpha$ se numește \\textbf{indicele de stabilitate} (sau exponentul caracteristic, sau tail index)')))

D.frame(T('Two Examples: Normal and Cauchy', 'Două exemple: distribuția Normală și distribuția Cauchy'), cols(items(
    (T('\\textbf{Normal}: $X_1, X_2 \\sim N(0, \\sigma^2)$ independent', '\\textbf{Distribuția Normală}: $X_1, X_2 \\sim N(0, \\sigma^2)$ independente'),
     [T('$aX_1 + bX_2 \\sim N(0, (a^2 + b^2)\\sigma^2)$, so $c = \\sqrt{a^2 + b^2}$', '$aX_1 + bX_2 \\sim N(0, (a^2 + b^2)\\sigma^2)$, deci $c = \\sqrt{a^2 + b^2}$'),
      T('variances add: $c^2 = a^2 + b^2$, the case $\\alpha = 2$', 'varianțele se adună: $c^2 = a^2 + b^2$, cazul $\\alpha = 2$')]),
    (T('\\textbf{Cauchy}: density $f(x) = \\dfrac{1}{\\pi(1 + x^2)}$', '\\textbf{Cauchy}: densitatea $f(x) = \\dfrac{1}{\\pi(1 + x^2)}$'),
     [T('$aX_1 + bX_2$ is Cauchy with scale $c = a + b$', '$aX_1 + bX_2$ este Cauchy cu scala $c = a + b$'),
      T('scales add linearly: $c^1 = a^1 + b^1$, the case $\\alpha = 1$', 'scalele se adună liniar: $c^1 = a^1 + b^1$, cazul $\\alpha = 1$'),
      T('the average of $n$ Cauchy draws is again standard Cauchy: averaging does not help',
        'media a $n$ extrageri Cauchy este tot Cauchy standard: medierea nu reduce dispersia')])),
    pic('ch3_cauchy.jpg', 'Augustin-Louis Cauchy (1789--1857)', 'Augustin-Louis Cauchy (1789--1857)',
        'https://commons.wikimedia.org/wiki/File:Augstin_Louis,_Baron_Cauchy._Lithograph_by_J._Boilly,_1821._Wellcome_V0001034.jpg',
        'Lithograph: J. Boilly (1821), Wellcome Collection; CC BY 2.0; Wikimedia Commons',
        'Litografie: J. Boilly (1821), Wellcome Collection; CC BY 2.0; Wikimedia Commons', h='0.46\\textheight'), wl='0.64', wr='0.32'))

D.frame(T('The General Rule and a Worked Example', 'Regula generală și un exemplu lucrat'), items(
    (T('For $X_1, X_2$ independent copies of a strictly stable $X$ with index $\\alpha$:', 'Pentru $X_1, X_2$ copii independente ale unei $X$ strict stabile cu indicele $\\alpha$:'),
     [T('$c^\\alpha = a^\\alpha + b^\\alpha$, i.e.\\ $c = (a^\\alpha + b^\\alpha)^{1/\\alpha}$', '$c^\\alpha = a^\\alpha + b^\\alpha$, adică $c = (a^\\alpha + b^\\alpha)^{1/\\alpha}$')]),
    (T('\\textbf{Example}: $a = 3$, $b = 4$', '\\textbf{Exemplu}: $a = 3$, $b = 4$'),
     [T('$\\alpha = 2$: $c = \\sqrt{9 + 16} = @{ex.c.2}$', '$\\alpha = 2$: $c = \\sqrt{9 + 16} = @{ex.c.2}$'),
      T('$\\alpha = 1.5$: $c = (3^{1.5} + 4^{1.5})^{1/1.5} = @{ex.c.1.5}$', '$\\alpha = 1.5$: $c = (3^{1.5} + 4^{1.5})^{1/1.5} = @{ex.c.1.5}$'),
      T('$\\alpha = 1$: $c = 3 + 4 = @{ex.c.1}$', '$\\alpha = 1$: $c = 3 + 4 = @{ex.c.1}$')]),
    T('The smaller $\\alpha$, the larger the scale of the sum: one large term can dominate the sum',
      'Cu cît $\\alpha$ este mai mic, cu atît scala sumei este mai mare: un singur termen mare poate domina suma'),
    T('\\textbf{What do you think?} For $\\alpha = 1.5$, what is the scale factor of a sum of two copies?',
      '\\textbf{Ce credeți?} Pentru $\\alpha = 1.5$, care este factorul de scală al sumei a două copii?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('$c_2 = 2^{1/1.5} = 2^{2/3} = @{chk.c}$, larger than $\\sqrt{2} \\approx @{sqrt2}$ for the Normal distribution',
        '$c_2 = 2^{1/1.5} = 2^{2/3} = @{chk.c}$, mai mare decît $\\sqrt{2} \\approx @{sqrt2}$ pentru distribuția Normală')])))

chart(T('Stability in a Simulation', 'Stabilitatea într-o simulare'), 'sfm_ch3_stability_qq', 'SFM_ch3_stability_gclt', [
    T('Notation (Section 4): $S(\\alpha, \\beta, \\gamma, \\delta)$ is the stable law with index $\\alpha$, skewness $\\beta$, scale $\\gamma$ and location $\\delta$',
      'Notație (secțiunea 4): $S(\\alpha, \\beta, \\gamma, \\delta)$ este legea stabilă cu indicele $\\alpha$, asimetria $\\beta$, scala $\\gamma$ și poziția $\\delta$'),
    T('Blue: sums of 10 draws of $S(1.7, 0, 1, 0)$ divided by $10^{1/1.7} = @{stab.factor}$ lie on the 45-degree line: same law as one draw',
      'Albastru: sumele a 10 extrageri din $S(1.7, 0, 1, 0)$ împărțite la $10^{1/1.7} = @{stab.factor}$ se află pe prima bisectoare: aceeași lege ca o singură extragere'),
    T('Red: dividing by $\\sqrt{10} = @{stab.sqrtn}$ (the Normal rule) leaves the tails too wide: 99\\% quantile $@{stab.q99_sqrt}$ vs $@{stab.q99_one}$',
      'Roșu: împărțirea la $\\sqrt{10} = @{stab.sqrtn}$ (regula distribuției Normale) lasă cozile prea largi: cuantila de 99\\% este $@{stab.q99_sqrt}$, față de $@{stab.q99_one}$')])

D.frame(T('Consequences: Horizons and Diversification', 'Consecințe: orizonturi și diversificare'), items(
    (T('\\textbf{Horizon}: if daily returns are i.i.d.\\ (independent, identically distributed) $S(\\alpha, 0, \\gamma, 0)$, the 20-day sum has scale $20^{1/\\alpha}\\gamma$',
       '\\textbf{Orizontul}: dacă randamentele zilnice sînt i.i.d.\\ (independente și identic distribuite) $S(\\alpha, 0, \\gamma, 0)$, suma pe 20 de zile are scala $20^{1/\\alpha}\\gamma$'),
     [T('$\\alpha = 2$: $@{ex.h.2}\\,\\gamma$ (the square-root-of-time rule); $\\alpha = 1.7$: $@{ex.h.1.7}\\,\\gamma$; $\\alpha = 1.5$: $@{ex.h.1.5}\\,\\gamma$',
        '$\\alpha = 2$: $@{ex.h.2}\\,\\gamma$ (regula rădăcinii pătrate a timpului); $\\alpha = 1.7$: $@{ex.h.1.7}\\,\\gamma$; $\\alpha = 1.5$: $@{ex.h.1.5}\\,\\gamma$'),
      T('with $\\alpha < 2$, the risk of long horizons grows faster than $\\sqrt{n}$', 'cu $\\alpha < 2$, riscul pe orizonturi lungi crește mai repede decît $\\sqrt{n}$')]),
    (T('\\textbf{Diversification}: an equally weighted portfolio of $N$ i.i.d.\\ stable assets has scale $N^{1/\\alpha - 1}\\gamma$',
       '\\textbf{Diversificarea}: un portofoliu cu ponderi egale din $N$ active stabile i.i.d.\\ are scala $N^{1/\\alpha - 1}\\gamma$'),
     [T('$N = 10$: $\\alpha = 2$ gives $@{ex.pf.2}\\,\\gamma$; $\\alpha = 1.5$ gives $@{ex.pf.1.5}\\,\\gamma$; $\\alpha = 1$ gives $@{ex.pf.1}\\,\\gamma$',
        '$N = 10$: $\\alpha = 2$ dă $@{ex.pf.2}\\,\\gamma$; $\\alpha = 1.5$ dă $@{ex.pf.1.5}\\,\\gamma$; $\\alpha = 1$ dă $@{ex.pf.1}\\,\\gamma$'),
      T('with $\\alpha = 1$ diversification does not reduce the scale at all; \\refFama\\ discussed this point',
        'cu $\\alpha = 1$, diversificarea nu reduce deloc scala; \\refFama\\ a discutat această consecință')])))

D.recap(('Stability', 'Stabilitatea'), [
    T('Stable law: a sum of independent copies has the same shape, only rescaled and shifted',
      'Lege stabilă: o sumă de copii independente are aceeași formă, doar rescalată și deplasată'),
    T('Scale rule: $c^\\alpha = a^\\alpha + b^\\alpha$; $n$ copies: factor $n^{1/\\alpha}$, $0 < \\alpha \\le 2$',
      'Regula scalei: $c^\\alpha = a^\\alpha + b^\\alpha$; $n$ copii: factorul $n^{1/\\alpha}$, $0 < \\alpha \\le 2$'),
    T('Normal: $\\alpha = 2$; Cauchy: $\\alpha = 1$', 'Normală: $\\alpha = 2$; Cauchy: $\\alpha = 1$'),
    T('Small $\\alpha$: risk grows faster with the horizon and diversification helps less',
      '$\\alpha$ mic: riscul crește mai repede cu orizontul, iar diversificarea ajută mai puțin')])

# =============================================================================
# 3. TEOREMA LIMITĂ CENTRALĂ GENERALIZATĂ
# =============================================================================
D.section('The Generalised Central Limit Theorem', 'Teorema limită centrală generalizată')

D.frame(T('From the Classical to the Generalised CLT', 'De la CLT clasică la cea generalizată'), items(
    (T('\\textbf{Classical CLT} (central limit theorem, Chapter 2): $X_i$ i.i.d.\\ with mean $\\mu$ and finite variance $\\sigma^2$',
       '\\textbf{CLT clasică} (teorema limită centrală, Capitolul 2): $X_i$ i.i.d.\\ cu media $\\mu$ și varianța finită $\\sigma^2$'),
     [T('$\\dfrac{X_1 + \\dots + X_n - n\\mu}{\\sigma\\sqrt{n}} \\to N(0, 1)$ in distribution', '$\\dfrac{X_1 + \\dots + X_n - n\\mu}{\\sigma\\sqrt{n}} \\to N(0, 1)$ în distribuție')]),
    T('What if the variance is infinite? The normalisation $\\sqrt{n}$ is then wrong',
      'Ce se întîmplă dacă varianța este infinită? Normalizarea cu $\\sqrt{n}$ nu mai este potrivită'),
    (T('\\textbf{GCLT} (generalised central limit theorem) \\refGK: if $(X_1 + \\dots + X_n - b_n)/a_n$ converges to a non-degenerate law, that law is stable',
       '\\textbf{GCLT} (teorema limită centrală generalizată) \\refGK: dacă $(X_1 + \\dots + X_n - b_n)/a_n$ converge către o lege nedegenerată, acea lege este stabilă'),
     [T('$a_n > 0$, $b_n$: normalising constants that rescale and recentre the sum (for the classical CLT, $a_n = \\sigma\\sqrt{n}$, $b_n = n\\mu$)',
        '$a_n > 0$, $b_n$: constante de normalizare care rescalează și recentrează suma (pentru CLT clasică, $a_n = \\sigma\\sqrt{n}$, $b_n = n\\mu$)'),
      T('stable laws are the \\textbf{only} possible limits of normalised sums of i.i.d.\\ variables',
        'legile stabile sînt \\textbf{singurele} limite posibile ale sumelor normalizate de variabile i.i.d.'),
      T('the Normal distribution is the special case $\\alpha = 2$', 'distribuția Normală este cazul particular $\\alpha = 2$')])))

D.frame(T('Domains of Attraction', 'Domenii de atracție'), cols(items(
    (T('The \\textbf{domain of attraction} of a stable law: all laws whose normalised sums converge to it',
       '\\textbf{Domeniul de atracție} al unei legi stabile: toate legile ale căror sume normalizate converg către ea'),
     [T('finite variance: domain of attraction of the Normal distribution, $a_n \\propto \\sqrt{n}$',
        'varianță finită: domeniul de atracție al distribuției Normale, $a_n \\propto \\sqrt{n}$'),
      T('power-law tails $P(|X| > x) \\approx C x^{-\\alpha}$ with $0 < \\alpha < 2$: the stable law with the same $\\alpha$, $a_n \\propto n^{1/\\alpha}$',
        'cozi de tip putere $P(|X| > x) \\approx C x^{-\\alpha}$ cu $0 < \\alpha < 2$: legea stabilă cu același $\\alpha$, $a_n \\propto n^{1/\\alpha}$'),
      T('$C > 0$: a constant; $\\propto$: proportional to', '$C > 0$: o constantă; $\\propto$: proporțional cu')]),
    T('So the \\textbf{tail} of one return decides the law of long sums', 'Prin urmare, \\textbf{coada} distribuției unui randament determină legea sumelor cu mulți termeni'),
    T('Paul Lévy characterised the stable laws in the 1920s; the full theory is in \\refGK',
      'Paul Lévy a caracterizat legile stabile în anii 1920; teoria completă se găsește în \\refGK')),
    pic('ch3_levy.jpg', 'Paul Lévy (1886--1971)', 'Paul Lévy (1886--1971)',
        'https://commons.wikimedia.org/wiki/File:Paul_Pierre_Levy_1886-1971.jpg',
        'Photo: Konrad Jacobs, Oberwolfach Photo Collection; CC BY-SA 2.0 de; Wikimedia Commons',
        'Foto: Konrad Jacobs, Oberwolfach Photo Collection; CC BY-SA 2.0 de; Wikimedia Commons', h='0.42\\textheight'), wl='0.64', wr='0.32'))

chart(T('The GCLT in a Simulation', 'GCLT într-o simulare'), 'sfm_ch3_gclt', 'SFM_ch3_stability_gclt', [
    T('Student-t with $\\nu = 1.5$ degrees of freedom: infinite variance, tails $\\propto x^{-1.5}$; sums of $n$ draws divided by $n^{1/1.5}$',
      'Student-t cu $\\nu = 1.5$ grade de libertate: varianță infinită, cozi $\\propto x^{-1.5}$; sumele a $n$ extrageri împărțite la $n^{1/1.5}$'),
    T('$n = 1, 10, 100$: the histograms approach the stable limit $S(1.5, 0, @{gclt.gamma}, 0)$; $P(|\\cdot| > 10)$: $@{gclt.n100_tail}\\%$ for $n = 100$, $@{gclt.tail_stable}\\%$ in the limit',
      '$n = 1, 10, 100$: histogramele se apropie de limita stabilă $S(1.5, 0, @{gclt.gamma}, 0)$; $P(|\\cdot| > 10)$: $@{gclt.n100_tail}\\%$ pentru $n = 100$, $@{gclt.tail_stable}\\%$ la limită'),
    T('A Normal law with the same interquartile range gives $@{gclt.tail_normal}\\%$: it misses the tails completely',
      'O distribuție Normală cu același interval intercuartilic dă $@{gclt.tail_normal}\\%$: nu surprinde deloc cozile')])

D.frame(T('What the GCLT Means for Returns', 'Implicațiile GCLT pentru randamente'), items(
    (T('Two possible worlds for daily returns', 'Două lumi posibile pentru randamentele zilnice'),
     [T('\\textbf{tail index below 2} (infinite variance): monthly returns are stable with the \\textbf{same} $\\alpha$',
        '\\textbf{tail index sub 2} (varianță infinită): randamentele lunare sînt stabile cu \\textbf{același} $\\alpha$'),
      T('\\textbf{tail index above 2} (finite variance): monthly returns move towards the Normal distribution',
        '\\textbf{tail index peste 2} (varianță finită): randamentele lunare se apropie de distribuția Normală')]),
    T('The two worlds give different testable predictions about returns at longer horizons',
      'Cele două lumi conduc la predicții diferite și testabile despre randamentele pe orizonturi mai lungi'),
    T('\\textbf{Question for the room}: in which world do the S\\&P 500 returns live?',
      '\\textbf{Întrebare pentru sală}: căreia dintre cele două lumi îi aparțin randamentele S\\&P 500?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('we test it with real data in the last sections: estimates of $\\alpha$ at daily, weekly and monthly horizons',
        'o testăm cu date reale în ultimele secțiuni: estimări ale lui $\\alpha$ la orizont zilnic, săptămînal și lunar')])))

D.recap(('The Generalised CLT', 'Teorema limită centrală generalizată'), [
    T('Classical CLT: finite variance, normalisation $\\sqrt{n}$, Normal limit', 'CLT clasică: varianță finită, normalizare $\\sqrt{n}$, limită Normală'),
    T('GCLT: stable laws are the only limits of normalised sums', 'GCLT: legile stabile sînt singurele limite ale sumelor normalizate'),
    T('Power-law tails with index $\\alpha < 2$: normalisation $n^{1/\\alpha}$, stable limit with the same $\\alpha$',
      'Cozi de tip putere cu indicele $\\alpha < 2$: normalizare $n^{1/\\alpha}$, limită stabilă cu același $\\alpha$'),
    T('The tail of a single return decides the law of long sums', 'Coada distribuției unui singur randament determină legea sumelor cu mulți termeni')])

# =============================================================================
# 4. PARAMETRI ȘI FUNCȚIA CARACTERISTICĂ
# =============================================================================
D.section('Parameters and the Characteristic Function', 'Parametrii și funcția caracteristică')

D.frame(T('The Characteristic Function', 'Funcția caracteristică'), items(
    (T('\\textbf{Characteristic function} of $X$: $\\varphi_X(t) = E[e^{itX}] = E[\\cos(tX)] + i\\,E[\\sin(tX)]$, $t \\in \\mathbb{R}$',
       '\\textbf{Funcția caracteristică} a lui $X$: $\\varphi_X(t) = E[e^{itX}] = E[\\cos(tX)] + i\\,E[\\sin(tX)]$, $t \\in \\mathbb{R}$'),
     [T('$i = \\sqrt{-1}$: the imaginary unit; $t$: a real argument, not time', '$i = \\sqrt{-1}$: unitatea imaginară; $t$: un argument real, nu timpul'),
      T('it always exists, because $|e^{itX}| = 1$, even when no moment exists', 'există întotdeauna, pentru că $|e^{itX}| = 1$, chiar și cînd nu există niciun moment'),
      T('it determines the distribution uniquely', 'determină distribuția în mod unic')]),
    (T('Two properties used in this chapter', 'Două proprietăți folosite în capitol'),
     [T('independent $X, Y$: $\\varphi_{X+Y}(t) = \\varphi_X(t)\\,\\varphi_Y(t)$: sums become products', '$X, Y$ independente: $\\varphi_{X+Y}(t) = \\varphi_X(t)\\,\\varphi_Y(t)$: sumele devin produse'),
      T('$\\varphi_{aX+b}(t) = e^{ibt}\\varphi_X(at)$', '$\\varphi_{aX+b}(t) = e^{ibt}\\varphi_X(at)$')]),
    T('Normal $N(\\mu, \\sigma^2)$: $\\varphi(t) = \\exp(i\\mu t - \\sigma^2 t^2/2)$', 'Normală $N(\\mu, \\sigma^2)$: $\\varphi(t) = \\exp(i\\mu t - \\sigma^2 t^2/2)$'),
    T('Why we need it: apart from three cases, stable laws have \\textbf{no closed-form density}; they are defined by $\\varphi$',
      'Utilitatea ei: cu excepția a trei cazuri, legile stabile \\textbf{nu au densitate în formă închisă}; sînt definite prin $\\varphi$')))

D.frame(T('The Stable Characteristic Function (S1)', 'Funcția caracteristică stabilă (S1)'), items(
    (T('$X \\sim S(\\alpha, \\beta, \\gamma, \\delta; 1)$ if, for $\\alpha \\ne 1$, \\refST, \\refNolan:',
       '$X \\sim S(\\alpha, \\beta, \\gamma, \\delta; 1)$ dacă, pentru $\\alpha \\ne 1$, \\refST, \\refNolan:'),
     [T('$\\varphi(t) = \\exp\\Big\\{-\\gamma^\\alpha |t|^\\alpha \\big[1 - i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big] + i\\delta t\\Big\\}$',
        '$\\varphi(t) = \\exp\\Big\\{-\\gamma^\\alpha |t|^\\alpha \\big[1 - i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big] + i\\delta t\\Big\\}$'),
      T('for $\\alpha = 1$: $\\varphi(t) = \\exp\\Big\\{-\\gamma |t| \\big[1 + i\\beta\\,\\frac{2}{\\pi}\\mathrm{sign}(t)\\ln|t|\\big] + i\\delta t\\Big\\}$',
        'pentru $\\alpha = 1$: $\\varphi(t) = \\exp\\Big\\{-\\gamma |t| \\big[1 + i\\beta\\,\\frac{2}{\\pi}\\mathrm{sign}(t)\\ln|t|\\big] + i\\delta t\\Big\\}$')]),
    (T('The four parameters', 'Cei patru parametri'),
     [T('$\\alpha \\in (0, 2]$: \\textbf{index of stability}; smaller $\\alpha$, heavier tails', '$\\alpha \\in (0, 2]$: \\textbf{indicele de stabilitate}; $\\alpha$ mai mic, cozi mai groase'),
      T('$\\beta \\in [-1, 1]$: \\textbf{skewness}; $\\beta > 0$ heavier right tail, $\\beta < 0$ heavier left tail',
        '$\\beta \\in [-1, 1]$: \\textbf{asimetria}; $\\beta > 0$ coada dreaptă mai groasă, $\\beta < 0$ coada stîngă mai groasă'),
      T('$\\gamma > 0$: \\textbf{scale} (it plays the role of $\\sigma$); $\\delta \\in \\mathbb{R}$: \\textbf{location}',
        '$\\gamma > 0$: \\textbf{scala} (joacă rolul lui $\\sigma$); $\\delta \\in \\mathbb{R}$: \\textbf{poziția}')]),
    T('$\\mathrm{sign}(t)$: $+1$ for $t > 0$, $-1$ for $t < 0$; the term with $\\beta$ makes the law asymmetric',
      '$\\mathrm{sign}(t)$: $+1$ pentru $t > 0$, $-1$ pentru $t < 0$; termenul cu $\\beta$ face legea asimetrică'),
    T('Stability is visible: $\\varphi(t)^n$ has the same form, with $\\gamma^\\alpha$ replaced by $n\\gamma^\\alpha$',
      'Stabilitatea este vizibilă: $\\varphi(t)^n$ are aceeași formă, cu $\\gamma^\\alpha$ înlocuit de $n\\gamma^\\alpha$')))

chart(T('The Effect of $\\alpha$', 'Efectul lui $\\alpha$'), 'sfm_ch3_density_alpha', 'SFM_ch3_densities', [
    T('Symmetric laws $S(\\alpha, 0, 1, 0)$: smaller $\\alpha$ gives a taller, narrower peak and much heavier tails',
      'Legi simetrice $S(\\alpha, 0, 1, 0)$: $\\alpha$ mai mic dă un vîrf mai înalt și mai îngust și cozi mult mai groase'),
    T('Log scale: the Normal tail ($\\alpha = 2$) falls like $e^{-x^2/4}$; the others fall like a power of $x$',
      'Scara logaritmică: coada distribuției Normale ($\\alpha = 2$) scade ca $e^{-x^2/4}$; celelalte scad ca o putere a lui $x$')])

chart(T('The Effect of $\\beta$ and $\\gamma$', 'Efectul lui $\\beta$ și $\\gamma$'), 'sfm_ch3_density_beta', 'SFM_ch3_densities', [
    T('$\\beta$ moves weight between the tails: $\\beta = 1$ (green) has the heavier right tail, $\\beta = -1$ (red) the heavier left tail',
      '$\\beta$ redistribuie masa între cozi: $\\beta = 1$ (verde) are coada dreaptă mai groasă, $\\beta = -1$ (roșu) coada stîngă mai groasă'),
    T('$\\gamma$ stretches the density: doubling $\\gamma$ doubles every quantile around $\\delta$',
      '$\\gamma$ întinde densitatea: dublarea lui $\\gamma$ dublează fiecare cuantilă în jurul lui $\\delta$')])

D.frame(T('Two Parameterisations: S0 and S1', 'Două parametrizări: S0 și S1'), items(
    (T('\\textbf{S0} (Nolan\'s recommendation for data analysis), $\\alpha \\ne 1$:', '\\textbf{S0} (recomandarea lui Nolan pentru analiza datelor), $\\alpha \\ne 1$:'),
     [T('$\\varphi(t) = \\exp\\Big\\{-\\gamma^\\alpha |t|^\\alpha \\big[1 + i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big(|\\gamma t|^{1-\\alpha} - 1\\big)\\big] + i\\delta_0 t\\Big\\}$',
        '$\\varphi(t) = \\exp\\Big\\{-\\gamma^\\alpha |t|^\\alpha \\big[1 + i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}\\big(|\\gamma t|^{1-\\alpha} - 1\\big)\\big] + i\\delta_0 t\\Big\\}$')]),
    (T('Only the location differs: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$ ($\\alpha \\ne 1$)',
       'Diferă doar poziția: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$ ($\\alpha \\ne 1$)'),
     [T('$\\delta_0$, $\\delta_1$: the location parameter in S0 and in S1; $\\alpha$, $\\beta$ and $\\gamma$ are the same in both', '$\\delta_0$, $\\delta_1$: parametrul de poziție în S0 și în S1; $\\alpha$, $\\beta$ și $\\gamma$ sînt aceleași în ambele'),
      T('for $\\beta = 0$ the two coincide', 'pentru $\\beta = 0$ cele două coincid')]),
    (T('Why two?', 'Rolul fiecăreia'),
     [T('S1: simpler algebra for sums; used in most theory books \\refST', 'S1: algebră mai simplă pentru sume; folosită în majoritatea cărților de teorie \\refST'),
      T('S0: the density changes continuously with $\\alpha$ and $\\beta$; better for estimation and plots \\refNolan',
        'S0: densitatea se schimbă continuu cu $\\alpha$ și $\\beta$; mai bună pentru estimare și grafice \\refNolan')])))

chart(T('S1 vs S0 near $\\alpha = 1$', 'S1 și S0 în apropiere de $\\alpha = 1$'), 'sfm_ch3_s0_s1', 'SFM_ch3_densities', [
    T('S1 (left): with $\\beta = 0.8$, the density runs away to $+\\infty$ as $\\alpha \\uparrow 1$ and comes back from $-\\infty$ as $\\alpha \\downarrow 1$, because $\\tan\\frac{\\pi\\alpha}{2}$ explodes',
      'S1 (stînga): cu $\\beta = 0.8$, densitatea se deplasează spre $+\\infty$ cînd $\\alpha \\uparrow 1$ și revine dinspre $-\\infty$ cînd $\\alpha \\downarrow 1$, pentru că $\\tan\\frac{\\pi\\alpha}{2}$ tinde la infinit'),
    T('S0 (right): the same four laws, shifted; the mode stays near 0 and changes smoothly',
      'S0 (dreapta): aceleași patru legi, deplasate; modul rămîne lîngă 0 și se schimbă lin')])

D.frame(T('Mind the Parameterisation in Software', 'Atenție la parametrizare în software'), items(
    (T('\\texttt{scipy.stats.levy\\_stable} uses \\textbf{S1} by default', '\\texttt{scipy.stats.levy\\_stable} folosește implicit \\textbf{S1}'),
     [T('switch with \\texttt{levy\\_stable.parameterization = \'S0\'}', 'se schimbă cu \\texttt{levy\\_stable.parameterization = \'S0\'}'),
      T('arguments: \\texttt{levy\\_stable.pdf(x, alpha, beta, loc=delta, scale=gamma)}', 'argumente: \\texttt{levy\\_stable.pdf(x, alpha, beta, loc=delta, scale=gamma)}')]),
    T('Papers and programs report $\\delta$ in S0, in S1 or in other parameterisations: check which one before comparing',
      'Lucrările și programele raportează $\\delta$ în S0, în S1 sau în alte parametrizări: verificați parametrizarea înainte de a compara rezultatele'),
    (T('Scale trap: $S(2, 0, \\gamma, \\delta)$ is $N(\\delta, 2\\gamma^2)$, not $N(\\delta, \\gamma^2)$', 'Capcana scalei: $S(2, 0, \\gamma, \\delta)$ este $N(\\delta, 2\\gamma^2)$, nu $N(\\delta, \\gamma^2)$'),
     [T('so $\\gamma = \\sigma/\\sqrt{2}$ for the Normal distribution', 'deci $\\gamma = \\sigma/\\sqrt{2}$ pentru distribuția Normală')]),
    T('In this course: estimates in S0, with $\\delta_1$ also reported', 'În acest curs: estimările sînt în S0, iar $\\delta_1$ este raportat separat')))

D.frame(T('Three Laws with a Closed-Form Density', 'Trei legi cu densitate în formă închisă'), items(
    (T('\\textbf{Normal}: $\\alpha = 2$ ($\\beta$ plays no role)', '\\textbf{Distribuția Normală}: $\\alpha = 2$ ($\\beta$ nu are niciun rol)'),
     [T('$\\varphi(t) = \\exp(-\\gamma^2 t^2 + i\\delta t)$, i.e.\\ $N(\\delta, 2\\gamma^2)$', '$\\varphi(t) = \\exp(-\\gamma^2 t^2 + i\\delta t)$, adică $N(\\delta, 2\\gamma^2)$')]),
    (T('\\textbf{Cauchy}: $\\alpha = 1$, $\\beta = 0$', '\\textbf{Cauchy}: $\\alpha = 1$, $\\beta = 0$'),
     [T('$f(x) = \\dfrac{\\gamma}{\\pi\\,(\\gamma^2 + (x - \\delta)^2)}$; $\\varphi(t) = \\exp(-\\gamma|t| + i\\delta t)$',
        '$f(x) = \\dfrac{\\gamma}{\\pi\\,(\\gamma^2 + (x - \\delta)^2)}$; $\\varphi(t) = \\exp(-\\gamma|t| + i\\delta t)$')]),
    (T('\\textbf{Lévy}: $\\alpha = 1/2$, $\\beta = 1$ (S1); totally skewed to the right', '\\textbf{Lévy}: $\\alpha = 1/2$, $\\beta = 1$ (S1); complet asimetrică la dreapta'),
     [T('$f(x) = \\sqrt{\\dfrac{\\gamma}{2\\pi}}\\,(x - \\delta)^{-3/2} \\exp\\Big(-\\dfrac{\\gamma}{2(x - \\delta)}\\Big)$ for $x > \\delta$',
        '$f(x) = \\sqrt{\\dfrac{\\gamma}{2\\pi}}\\,(x - \\delta)^{-3/2} \\exp\\Big(-\\dfrac{\\gamma}{2(x - \\delta)}\\Big)$ pentru $x > \\delta$'),
      T('the first time a Brownian motion hits a level has a Lévy law', 'momentul în care o mișcare browniană atinge prima dată un nivel are o lege Lévy')]),
    T('All other stable densities are computed numerically from $\\varphi$ \\refNolanA, \\refMDC',
      'Toate celelalte densități stabile se calculează numeric din $\\varphi$ \\refNolanA, \\refMDC')))

chart(T('Normal, Cauchy and Lévy', 'Distribuțiile Normală, Cauchy și Lévy'), 'sfm_ch3_special_cases', 'SFM_ch3_densities', [
    T('Cauchy: lower peak, much heavier tails than the Normal distribution with variance 1', 'Cauchy: vîrf mai jos, cozi mult mai groase decît distribuția Normală cu varianța 1'),
    T('Lévy: only positive values, a long right tail; its mean is infinite', 'Lévy: doar valori pozitive, o coadă dreaptă lungă; media ei este infinită')], h='0.60\\textheight')

D.frame(T('Worked Example: S1 vs S0 at $t = 1$', 'Exemplu lucrat: S1 și S0 în $t = 1$'), items(
    T('$X$ with $\\alpha = 1.5$, $\\beta = 0.3$, $\\gamma = 1$, $\\delta = 0$; note $\\tan\\frac{3\\pi}{4} = -1$',
      '$X$ cu $\\alpha = 1.5$, $\\beta = 0.3$, $\\gamma = 1$, $\\delta = 0$; observați că $\\tan\\frac{3\\pi}{4} = -1$'),
    (T('\\textbf{S1}: $\\ln\\varphi(1) = -1\\cdot[1 - i \\cdot 0.3 \\cdot (-1)] = -1 - 0.3i$', '\\textbf{S1}: $\\ln\\varphi(1) = -1\\cdot[1 - i \\cdot 0.3 \\cdot (-1)] = -1 - 0.3i$'),
     [T('$\\varphi(1) = e^{-1}(\\cos 0.3 - i \\sin 0.3) = @{cf.re} - @{cf.im}i$; $|\\varphi(1)| = e^{-1} = @{cf.abs}$',
        '$\\varphi(1) = e^{-1}(\\cos 0.3 - i \\sin 0.3) = @{cf.re} - @{cf.im}i$; $|\\varphi(1)| = e^{-1} = @{cf.abs}$')]),
    (T('\\textbf{S0}: the factor $|t|^{1-\\alpha} - 1$ is 0 at $t = 1$, so $\\varphi(1) = e^{-1} = @{cf.abs}$, a real number',
       '\\textbf{S0}: factorul $|t|^{1-\\alpha} - 1$ este 0 în $t = 1$, deci $\\varphi(1) = e^{-1} = @{cf.abs}$, un număr real'),
     [T('same modulus: the two laws differ only by a shift', 'același modul: cele două legi diferă doar printr-o deplasare'),
      T('shift $= \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2} = @{cf.shift}$: the S1 law with $\\delta_1 = 0$ is the S0 law with $\\delta_0 = @{cf.shift}$',
        'deplasarea $= \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2} = @{cf.shift}$: legea S1 cu $\\delta_1 = 0$ este legea S0 cu $\\delta_0 = @{cf.shift}$')])))

D.recap(('Parameters and the Characteristic Function', 'Parametrii și funcția caracteristică'), [
    T('Stable laws are defined by $\\varphi(t)$; the density is computed numerically', 'Legile stabile se definesc prin $\\varphi(t)$; densitatea se calculează numeric'),
    T('$\\alpha$ tails, $\\beta$ skewness, $\\gamma$ scale, $\\delta$ location', '$\\alpha$ cozi, $\\beta$ asimetrie, $\\gamma$ scală, $\\delta$ poziție'),
    T('S0 vs S1: only $\\delta$ differs, $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$; scipy uses S1 by default',
      'S0 și S1: diferă doar $\\delta$, $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$; scipy folosește implicit S1'),
    T('Closed forms: Normal ($\\alpha = 2$, variance $2\\gamma^2$), Cauchy, Lévy', 'Forme închise: Normală ($\\alpha = 2$, varianța $2\\gamma^2$), Cauchy, Lévy')])

# =============================================================================
# 5. COZI ȘI MOMENTE
# =============================================================================
D.section('Tails and Moments', 'Cozi și momente')

D.frame(T('Power-Law Tails', 'Cozi de tip putere'), items(
    (T('For $\\alpha < 2$, as $x \\to \\infty$ \\refNolan:', 'Pentru $\\alpha < 2$, cînd $x \\to \\infty$ \\refNolan:'),
     [T('$P(X > x) \\sim \\gamma^\\alpha c_\\alpha (1 + \\beta)\\, x^{-\\alpha}$ and $P(X < -x) \\sim \\gamma^\\alpha c_\\alpha (1 - \\beta)\\, x^{-\\alpha}$',
        '$P(X > x) \\sim \\gamma^\\alpha c_\\alpha (1 + \\beta)\\, x^{-\\alpha}$ și $P(X < -x) \\sim \\gamma^\\alpha c_\\alpha (1 - \\beta)\\, x^{-\\alpha}$'),
      T('$c_\\alpha = \\Gamma(\\alpha)\\sin(\\pi\\alpha/2)/\\pi$, with $\\Gamma$ the gamma function; $f \\sim g$ means $f/g \\to 1$',
        '$c_\\alpha = \\Gamma(\\alpha)\\sin(\\pi\\alpha/2)/\\pi$, cu $\\Gamma$ funcția gamma; $f \\sim g$ înseamnă $f/g \\to 1$')]),
    (T('On log-log axes the tail is a straight line with slope $-\\alpha$', 'Pe axe log-log, coada este o dreaptă cu panta $-\\alpha$'),
     [T('doubling the threshold divides the tail probability by $2^\\alpha$: $\\times @{tr.17}$ for $\\alpha = 1.7$',
        'dublarea pragului împarte probabilitatea din coadă la $2^\\alpha$: $\\times @{tr.17}$ pentru $\\alpha = 1.7$')]),
    T('The Normal tail falls like $e^{-x^2/(4\\gamma^2)}$: doubling the threshold makes it vanish',
      'Coada distribuției Normale scade ca $e^{-x^2/(4\\gamma^2)}$: dublarea pragului o face practic nulă'),
    T('$\\beta = \\pm 1$ with $\\alpha < 1$: one tail is missing (the law lives on a half-line)',
      '$\\beta = \\pm 1$ cu $\\alpha < 1$: una dintre cozi lipsește (suportul legii este o semidreaptă)')))

chart(T('Tails on Log-Log Axes', 'Cozile pe axe log-log'), 'sfm_ch3_tails_loglog', 'SFM_ch3_tails_moments', [
    T('Solid: $P(X > x)$ of $S(\\alpha, 0, 1, 0)$; dotted: the power law $c_\\alpha x^{-\\alpha}$, reached already near $x = 5$',
      'Linie continuă: $P(X > x)$ pentru $S(\\alpha, 0, 1, 0)$; linie punctată: legea de tip putere $c_\\alpha x^{-\\alpha}$, atinsă deja în jurul lui $x = 5$'),
    T('$P(X > 10)$: $@{tl.1.7.sf10}\\%$ for $\\alpha = 1.7$, $@{tl.1.5.sf10}\\%$ for $\\alpha = 1.5$, about $@{tl.2.sf10}\\%$ for the Normal distribution',
      '$P(X > 10)$: $@{tl.1.7.sf10}\\%$ pentru $\\alpha = 1.7$, $@{tl.1.5.sf10}\\%$ pentru $\\alpha = 1.5$, circa $@{tl.2.sf10}\\%$ pentru distribuția Normală')])

D.frame(T('Which Moments Exist?', 'Existența momentelor'), items(
    (T('For $0 < \\alpha < 2$: $E|X|^p < \\infty$ if and only if $p < \\alpha$', 'Pentru $0 < \\alpha < 2$: $E|X|^p < \\infty$ dacă și numai dacă $p < \\alpha$'),
     [T('$E|X|^p$: the absolute moment of order $p > 0$; $p = 1$ gives the mean, $p = 2$ the variance', '$E|X|^p$: momentul absolut de ordin $p > 0$; $p = 1$ corespunde mediei, $p = 2$ varianței'),
      T('reason: $|x|^p$ times a density of order $|x|^{-\\alpha-1}$ is integrable only if $p < \\alpha$',
        'motivul: $|x|^p$ înmulțit cu o densitate de ordinul $|x|^{-\\alpha-1}$ este integrabil doar dacă $p < \\alpha$')]),
    (T('Consequences', 'Consecințe'),
     [T('$\\alpha < 2$: \\textbf{infinite variance}; skewness and kurtosis are not defined', '$\\alpha < 2$: \\textbf{varianță infinită}; asimetria și boltirea nu sînt definite'),
      T('$1 < \\alpha < 2$: the mean exists; in S1 it equals $\\delta_1$', '$1 < \\alpha < 2$: media există; în S1 este egală cu $\\delta_1$'),
      T('$\\alpha \\le 1$: not even the mean exists (Cauchy, Lévy)', '$\\alpha \\le 1$: nici media nu există (Cauchy, Lévy)'),
      T('$\\alpha = 2$: all moments exist', '$\\alpha = 2$: toate momentele există')]),
    T('\\textbf{What do you think?} Mandelbrot found $\\alpha \\approx 1.7$ for cotton. Does the mean daily change exist?',
      '\\textbf{Ce credeți?} Mandelbrot a găsit $\\alpha \\approx 1.7$ pentru bumbac. Există media variației zilnice?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('yes, because $1 < 1.7$; but the variance does not, because $2 > 1.7$', 'da, pentru că $1 < 1.7$; dar varianța nu există, pentru că $2 > 1.7$')])))

chart(T('The Sample Variance Does Not Settle', 'Varianța de selecție nu se stabilizează'), 'sfm_ch3_running_var_sim', 'SFM_ch3_tails_moments', [
    T('Normal (blue): the sample variance converges to the true value 2', 'Distribuția Normală (albastru): varianța de selecție converge la valoarea adevărată 2'),
    T('Stable $\\alpha = 1.7$: each large draw makes the variance jump; after 20\\,000 draws the three paths end at $@{rv.path1.vend}$, $@{rv.path2.vend}$, $@{rv.path3.vend}$',
      'Stabilă $\\alpha = 1.7$: fiecare extragere mare produce un salt al varianței; după 20\\,000 de extrageri, cele trei traiectorii ajung la $@{rv.path1.vend}$, $@{rv.path2.vend}$, $@{rv.path3.vend}$'),
    T('Mandelbrot saw exactly this in cotton prices', 'Același fenomen l-a observat Mandelbrot în prețurile bumbacului')])


def ttrow(k):
    return f'${k}$ & $@{{tt.{k}.n}}$ & $@{{tt.{k}.s}}$ & $@{{tt.{k}.r}}$'


D.frame(T('Tail Probabilities: Stable vs Normal', 'Probabilitățile din cozi: distribuția stabilă și distribuția Normală'), table(
    'rrrr', T('$k$ & $P(|X| > k)$, Normal \\% & $P(|X| > k)$, $\\alpha = 1.7$ \\% & ratio', '$k$ & $P(|X| > k)$, Normală \\% & $P(|X| > k)$, $\\alpha = 1.7$ \\% & raport'),
    [ttrow(2), ttrow(3), ttrow(5), f'$10$ & $@{{tt.10.n.sci}}$ & $@{{tt.10.s}}$ & $@{{tt.10.r}}$'], size='footnotesize') + items(
    T('Both laws with $\\beta = 0$, $\\gamma = 1$, $\\delta = 0$: the Normal law is $N(0, 2)$', 'Ambele legi au $\\beta = 0$, $\\gamma = 1$, $\\delta = 0$; legea Normală este deci $N(0, 2)$'),
    T('Near the centre the two laws are close; far in the tails they differ by orders of magnitude',
      'În jurul centrului, cele două legi sînt apropiate; departe în cozi, diferă prin ordine de mărime'),
    T('A risk model is judged in the tails, where the choice of law matters most', 'Un model de risc se evaluează în cozi, unde alegerea legii contează cel mai mult')) + ql('SFM_ch3_tails_moments'))

D.frame(T('Infinite Variance and Risk Measures', 'Varianța infinită și măsurile de risc'), items(
    (T('Tools that need a finite variance', 'Instrumente care au nevoie de varianță finită'),
     [T('volatility, the Sharpe ratio and its standard error (Chapter 1); mean--variance portfolios; confidence intervals based on the CLT',
        'volatilitatea, raportul Sharpe și eroarea lui standard (Capitolul 1); portofoliile medie--varianță; intervalele de încredere obținute din CLT'),
      T('with $\\alpha < 2$ these numbers exist in every sample but estimate nothing', 'cu $\\alpha < 2$, aceste mărimi se pot calcula pe orice eșantion, dar nu estimează nimic')]),
    (T('Tools that survive', 'Instrumente care rămîn valabile'),
     [T('quantiles always exist: $\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, the loss exceeded with probability 1\\%',
        'cuantilele există întotdeauna: $\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, pierderea depășită cu probabilitatea 1\\%'),
      T('$\\mathrm{ES}_{2.5\\%}$ (expected shortfall, the mean loss beyond the VaR) exists only if $\\alpha > 1$',
        '$\\mathrm{ES}_{2.5\\%}$ (expected shortfall, pierderea medie dincolo de VaR) există doar dacă $\\alpha > 1$'),
      T('the scale $\\gamma$ and the interquartile range replace the standard deviation', 'scala $\\gamma$ și intervalul intercuartilic înlocuiesc abaterea standard')]),
    T('VaR and ES are developed in Chapter 10', 'VaR și ES sînt tratate pe larg în Capitolul 10')))

D.recap(('Tails and Moments', 'Cozi și momente'), [
    T('Tails: $P(|X| > x) \\propto x^{-\\alpha}$; a straight line on log-log axes', 'Cozi: $P(|X| > x) \\propto x^{-\\alpha}$; o dreaptă pe axe log-log'),
    T('Moments: $E|X|^p < \\infty$ iff $p < \\alpha$; $\\alpha < 2$ means infinite variance', 'Momente: $E|X|^p < \\infty$ dacă și numai dacă $p < \\alpha$; $\\alpha < 2$ înseamnă varianță infinită'),
    T('The sample variance of a stable sample jumps and does not settle', 'Varianța de selecție a unui eșantion stabil are salturi și nu se stabilizează'),
    T('Quantiles, VaR and the scale $\\gamma$ remain meaningful', 'Cuantilele, VaR și scala $\\gamma$ își păstrează sensul')])

# =============================================================================
# 6. SIMULARE
# =============================================================================
D.section('Simulation', 'Simulare')

D.frame(T('The Chambers--Mallows--Stuck Method (Symmetric Case)', 'Metoda Chambers--Mallows--Stuck (cazul simetric)'), items(
    T('No closed-form density, yet stable variables are easy to simulate \\refCMS', 'Nu există densitate în formă închisă, dar variabilele stabile se simulează ușor \\refCMS'),
    (T('\\textbf{CMS} (Chambers--Mallows--Stuck) for $S(\\alpha, 0, 1, 0)$:', '\\textbf{CMS} (Chambers--Mallows--Stuck) pentru $S(\\alpha, 0, 1, 0)$:'),
     [T('draw $V \\sim U(-\\pi/2, \\pi/2)$, a uniform angle, and $W \\sim \\mathrm{Exp}(1)$, an exponential variable with mean 1, independent', 'extrageți $V \\sim U(-\\pi/2, \\pi/2)$, un unghi uniform, și $W \\sim \\mathrm{Exp}(1)$, o variabilă exponențială cu media 1, independente'),
      T('$X = \\dfrac{\\sin(\\alpha V)}{(\\cos V)^{1/\\alpha}} \\left(\\dfrac{\\cos((1 - \\alpha)V)}{W}\\right)^{(1 - \\alpha)/\\alpha}$',
        '$X = \\dfrac{\\sin(\\alpha V)}{(\\cos V)^{1/\\alpha}} \\left(\\dfrac{\\cos((1 - \\alpha)V)}{W}\\right)^{(1 - \\alpha)/\\alpha}$'),
      T('then $\\gamma X + \\delta \\sim S(\\alpha, 0, \\gamma, \\delta)$', 'apoi $\\gamma X + \\delta \\sim S(\\alpha, 0, \\gamma, \\delta)$')]),
    (T('Checks on special cases', 'Verificări pe cazuri particulare'),
     [T('$\\alpha = 1$: $X = \\tan V$, a Cauchy variable', '$\\alpha = 1$: $X = \\tan V$, o variabilă Cauchy'),
      T('$\\alpha = 2$: $X = 2\\sin V\\sqrt{W}$, a $N(0, 2)$ variable', '$\\alpha = 2$: $X = 2\\sin V\\sqrt{W}$, o variabilă $N(0, 2)$')])))

D.frame(T('The General Case ($\\beta \\ne 0$)', 'Cazul general ($\\beta \\ne 0$)'), items(
    (T('For $\\alpha \\ne 1$, S1 parameterisation \\refWeron:', 'Pentru $\\alpha \\ne 1$, parametrizarea S1 \\refWeron:'),
     [T('$B = \\dfrac{\\arctan(\\beta\\tan\\frac{\\pi\\alpha}{2})}{\\alpha}$, \\quad $S = \\Big(1 + \\beta^2\\tan^2\\frac{\\pi\\alpha}{2}\\Big)^{1/(2\\alpha)}$',
        '$B = \\dfrac{\\arctan(\\beta\\tan\\frac{\\pi\\alpha}{2})}{\\alpha}$, \\quad $S = \\Big(1 + \\beta^2\\tan^2\\frac{\\pi\\alpha}{2}\\Big)^{1/(2\\alpha)}$'),
      T('$X = S\\,\\dfrac{\\sin(\\alpha(V + B))}{(\\cos V)^{1/\\alpha}} \\left(\\dfrac{\\cos(V - \\alpha(V + B))}{W}\\right)^{(1 - \\alpha)/\\alpha} \\sim S(\\alpha, \\beta, 1, 0; 1)$',
        '$X = S\\,\\dfrac{\\sin(\\alpha(V + B))}{(\\cos V)^{1/\\alpha}} \\left(\\dfrac{\\cos(V - \\alpha(V + B))}{W}\\right)^{(1 - \\alpha)/\\alpha} \\sim S(\\alpha, \\beta, 1, 0; 1)$')]),
    T('$B$, $S$: auxiliary constants that depend only on $\\alpha$ and $\\beta$; for $\\beta = 0$, $B = 0$ and $S = 1$', '$B$, $S$: constante auxiliare care depind doar de $\\alpha$ și $\\beta$; pentru $\\beta = 0$, $B = 0$ și $S = 1$'),
    (T('From the standard draw to any parameters', 'De la extragerea standard la orice parametri'),
     [T('S1: $\\gamma X + \\delta_1$; S0: $\\gamma X + \\delta_0 - \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$', 'S1: $\\gamma X + \\delta_1$; S0: $\\gamma X + \\delta_0 - \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$')]),
    T('Weron (1996) corrected a formula of the original paper for $\\beta \\ne 0$; \\texttt{scipy.stats.levy\\_stable.rvs} uses the same method',
      'Weron (1996) a corectat o formulă din lucrarea originală pentru $\\beta \\ne 0$; \\texttt{scipy.stats.levy\\_stable.rvs} folosește aceeași metodă')))

D.frame(T('Worked Example: One Draw Step by Step', 'Exemplu lucrat: o extragere pas cu pas'), items(
    T('$\\alpha = 1.5$, $\\beta = 0$; a uniform draw $U = 0.8$ and an exponential draw $W = 1.2$',
      '$\\alpha = 1.5$, $\\beta = 0$; o extragere uniformă $U = 0.8$ și o extragere exponențială $W = 1.2$'),
    (T('Steps', 'Pași'),
     [T('$V = \\pi(U - 1/2) = @{cms.V}$: turns $U$, uniform on $(0, 1)$, into a uniform angle on $(-\\pi/2, \\pi/2)$', '$V = \\pi(U - 1/2) = @{cms.V}$: transformă $U$, uniformă pe $(0, 1)$, într-un unghi uniform pe $(-\\pi/2, \\pi/2)$'),
      T('$\\sin(\\alpha V) = @{cms.sin}$; $(\\cos V)^{1/\\alpha} = @{cms.cos}$', '$\\sin(\\alpha V) = @{cms.sin}$; $(\\cos V)^{1/\\alpha} = @{cms.cos}$'),
      T('$\\big(\\cos((1 - \\alpha)V)/W\\big)^{(1 - \\alpha)/\\alpha} = @{cms.last}$', '$\\big(\\cos((1 - \\alpha)V)/W\\big)^{(1 - \\alpha)/\\alpha} = @{cms.last}$'),
      T('$X = @{cms.sin}/@{cms.cos} \\times @{cms.last} = @{cms.X}$', '$X = @{cms.sin}/@{cms.cos} \\times @{cms.last} = @{cms.X}$')]),
    T('Repeat with new $(U, W)$ pairs to obtain a sample of any size', 'Repetați cu noi perechi $(U, W)$ pentru un eșantion de orice mărime')))

chart(T('CMS Draws against the Exact Density', 'Extrageri CMS față de densitatea exactă'), 'sfm_ch3_cms', 'SFM_ch3_sim_stable', [
    T('100\\,000 CMS draws of $S(1.5, 0.5, 1, 0; 1)$: the histogram follows the density computed from $\\varphi$',
      '100\\,000 de extrageri CMS din $S(1.5, 0.5, 1, 0; 1)$: histograma urmează densitatea calculată din $\\varphi$'),
    T('Quantiles 1\\% and 99\\%: sample $@{cms.q01}$ and $@{cms.q99}$, exact $@{cms.q01_exact}$ and $@{cms.q99_exact}$',
      'Cuantilele de 1\\% și 99\\%: eșantion $@{cms.q01}$ și $@{cms.q99}$, exact $@{cms.q01_exact}$ și $@{cms.q99_exact}$'),
    T('Two-sample KS (Kolmogorov--Smirnov) test against \\texttt{levy\\_stable.rvs}: statistic $@{cms.ks}$, p-value $@{cms.p}$, no difference',
      'Testul KS (Kolmogorov--Smirnov) cu două eșantioane față de \\texttt{levy\\_stable.rvs}: statistica $@{cms.ks}$, p-value $@{cms.p}$, nicio diferență')])

D.recap(('Simulation', 'Simularea'), [
    T('CMS: a uniform angle $V$ and an exponential $W$ give one stable draw', 'CMS: un unghi uniform $V$ și un $W$ exponențial dau o extragere stabilă'),
    T('Special cases check the formula: $\\tan V$ (Cauchy), $2\\sin V\\sqrt{W}$ (Normal)', 'Cazurile particulare verifică formula: $\\tan V$ (Cauchy), $2\\sin V\\sqrt{W}$ (Normală)'),
    T('Skewed case: the formula of \\refWeron; shift by $\\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$ for S0', 'Cazul asimetric: formula din \\refWeron; deplasare cu $\\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$ pentru S0')])

# =============================================================================
# 7. ESTIMARE
# =============================================================================
D.section('Estimation', 'Estimare')

D.frame(T('Three Families of Estimators', 'Trei familii de estimatori'), items(
    (T('\\textbf{Quantile methods}: match sample quantiles to those of the stable law', '\\textbf{Metodele cuantilelor}: potrivesc cuantilele de selecție cu cele ale legii stabile'),
     [T('\\refFamaRollB\\ for symmetric laws; \\refMcCulloch\\ for all four parameters', '\\refFamaRollB\\ pentru legi simetrice; \\refMcCulloch\\ pentru toți cei patru parametri')]),
    (T('\\textbf{Characteristic-function methods}: regress the empirical $\\ln(-\\ln|\\hat\\varphi(t)|^2)$ on $\\ln|t|$',
       '\\textbf{Metodele funcției caracteristice}: regresia lui $\\ln(-\\ln|\\hat\\varphi(t)|^2)$ empiric pe $\\ln|t|$'),
     [T('$\\hat\\varphi(t) = \\frac1n\\sum_{j=1}^n e^{itr_j}$: the empirical characteristic function; $|\\varphi(t)|^2 = e^{-2\\gamma^\\alpha|t|^\\alpha}$, so the log-log relation is linear in $\\ln|t|$',
        '$\\hat\\varphi(t) = \\frac1n\\sum_{j=1}^n e^{itr_j}$: funcția caracteristică empirică; $|\\varphi(t)|^2 = e^{-2\\gamma^\\alpha|t|^\\alpha}$, deci relația dublu logaritmică este liniară în $\\ln|t|$'),
      T('\\refKoutrouvelis; the slope estimates $\\alpha$', '\\refKoutrouvelis; panta estimează $\\alpha$')]),
    (T('\\textbf{Maximum likelihood} (ML): the most precise, once the density can be computed', '\\textbf{Verosimilitatea maximă} (ML): cea mai precisă, dacă densitatea poate fi calculată'),
     [T('\\refNolanML, \\refMRDC', '\\refNolanML, \\refMRDC')]),
    T('A SAS implementation of these estimators is described in \\refPele', 'O implementare SAS a acestor estimatori este descrisă în \\refPele')))

D.frame(T("McCulloch's Quantile Method", 'Metoda cuantilelor a lui McCulloch'), items(
    (T('Two ratios of sample quantiles $\\hat q_p$, free of location and scale \\refMcCulloch:', 'Două rapoarte de cuantile de selecție $\\hat q_p$, independente de poziție și de scală \\refMcCulloch:'),
     [T('$\\nu_\\alpha = \\dfrac{\\hat q_{0.95} - \\hat q_{0.05}}{\\hat q_{0.75} - \\hat q_{0.25}}$ (tail width relative to the body)',
        '$\\nu_\\alpha = \\dfrac{\\hat q_{0.95} - \\hat q_{0.05}}{\\hat q_{0.75} - \\hat q_{0.25}}$ (lățimea cozilor față de corp)'),
      T('$\\nu_\\beta = \\dfrac{\\hat q_{0.95} + \\hat q_{0.05} - 2\\hat q_{0.5}}{\\hat q_{0.95} - \\hat q_{0.05}}$ (asymmetry)',
        '$\\nu_\\beta = \\dfrac{\\hat q_{0.95} + \\hat q_{0.05} - 2\\hat q_{0.5}}{\\hat q_{0.95} - \\hat q_{0.05}}$ (asimetria)')]),
    T('$\\hat q_p$: the sample quantile of order $p$, the value below which a share $p$ of the returns lies', '$\\hat q_p$: cuantila de selecție de ordin $p$, valoarea sub care se află o proporție $p$ din randamente'),
    (T('Tables III and IV of the paper map $(\\nu_\\alpha, \\nu_\\beta)$ to $(\\hat\\alpha, \\hat\\beta)$', 'Tabelele III și IV din lucrare transformă $(\\nu_\\alpha, \\nu_\\beta)$ în $(\\hat\\alpha, \\hat\\beta)$'),
     [T('Normal distribution: $\\nu_\\alpha = 2.439$; larger $\\nu_\\alpha$ means smaller $\\alpha$', 'distribuția Normală: $\\nu_\\alpha = 2.439$; un $\\nu_\\alpha$ mai mare înseamnă un $\\alpha$ mai mic')]),
    T('Then $\\hat\\gamma = (\\hat q_{0.75} - \\hat q_{0.25})/\\phi_3(\\hat\\alpha, \\hat\\beta)$ (Table V) and $\\hat\\delta_0$ from Table VII; $\\phi_3$: the interquartile range of $S(\\alpha, \\beta, 1, 0)$',
      'Apoi $\\hat\\gamma = (\\hat q_{0.75} - \\hat q_{0.25})/\\phi_3(\\hat\\alpha, \\hat\\beta)$ (Tabelul V) și $\\hat\\delta_0$ din Tabelul VII; $\\phi_3$: intervalul intercuartilic al lui $S(\\alpha, \\beta, 1, 0)$'),
    T('Fast, consistent, valid for $0.6 \\le \\alpha \\le 2$; less precise than ML; a good starting point for ML',
      'Rapidă, consistentă, valabilă pentru $0.6 \\le \\alpha \\le 2$; mai puțin precisă decît ML; un bun punct de plecare pentru ML')))

side(T("McCulloch's Map from $\\nu_\\alpha$ to $\\alpha$", 'Corespondența McCulloch între $\\nu_\\alpha$ și $\\alpha$'), 'sfm_ch3_mcculloch_map', 'SFM_ch3_estimation', [
    T('Line: Table III for a symmetric sample ($\\nu_\\beta = 0$)', 'Linia: Tabelul III pentru un eșantion simetric ($\\nu_\\beta = 0$)'),
    T('Stars: daily log returns since @{f.bet.y0} (Bitcoin since @{f.btc.y0})', 'Stelele: randamente logaritmice zilnice din @{f.bet.y0} (Bitcoin din @{f.btc.y0})'),
    T('$\\nu_\\alpha$: BET $@{f.bet.m.nu_alpha}$, S\\&P 500 $@{f.sp500.m.nu_alpha}$, DAX $@{f.dax.m.nu_alpha}$, Bitcoin $@{f.btc.m.nu_alpha}$',
      '$\\nu_\\alpha$: BET $@{f.bet.m.nu_alpha}$, S\\&P 500 $@{f.sp500.m.nu_alpha}$, DAX $@{f.dax.m.nu_alpha}$, Bitcoin $@{f.btc.m.nu_alpha}$'),
    T('All well above 2.439: tails much wider than the Normal distribution allows', 'Toate mult peste 2,439: cozi mult mai largi decît permite distribuția Normală')], w=0.58)

D.frame(T('Worked Example: McCulloch on the BET', 'Exemplu lucrat: McCulloch pe BET'), items(
    T('BET daily log returns (\\%), @{f.bet.y0}--@{y1}, $n = @{f.bet.n}$', 'Randamentele logaritmice zilnice BET (\\%), @{f.bet.y0}--@{y1}, $n = @{f.bet.n}$'),
    (T('Step 1: quantiles', 'Pasul 1: cuantilele'),
     [T('$\\hat q_{0.05} = @{f.bet.m.q05}$, $\\hat q_{0.25} = @{f.bet.m.q25}$, $\\hat q_{0.5} = @{f.bet.m.q50}$, $\\hat q_{0.75} = @{f.bet.m.q75}$, $\\hat q_{0.95} = @{f.bet.m.q95}$',
        '$\\hat q_{0.05} = @{f.bet.m.q05}$, $\\hat q_{0.25} = @{f.bet.m.q25}$, $\\hat q_{0.5} = @{f.bet.m.q50}$, $\\hat q_{0.75} = @{f.bet.m.q75}$, $\\hat q_{0.95} = @{f.bet.m.q95}$')]),
    (T('Step 2: ratios', 'Pasul 2: rapoartele'),
     [T('$\\nu_\\alpha = @{mc.bet.range}/@{mc.bet.iqr} = @{f.bet.m.nu_alpha}$; $\\nu_\\beta = @{f.bet.m.nu_beta}$, practically symmetric',
        '$\\nu_\\alpha = @{mc.bet.range}/@{mc.bet.iqr} = @{f.bet.m.nu_alpha}$; $\\nu_\\beta = @{f.bet.m.nu_beta}$, practic simetric')]),
    (T('Step 3: Table III, column $\\nu_\\beta = 0$: $\\nu_\\alpha = 3.2 \\mapsto 1.484$, $\\nu_\\alpha = 3.5 \\mapsto 1.391$',
       'Pasul 3: Tabelul III, coloana $\\nu_\\beta = 0$: $\\nu_\\alpha = 3.2 \\mapsto 1.484$, $\\nu_\\alpha = 3.5 \\mapsto 1.391$'),
     [T('linear interpolation: $\\hat\\alpha \\approx @{mc.bet.interp}$', 'interpolare liniară: $\\hat\\alpha \\approx @{mc.bet.interp}$')]),
    T('Step 4: $\\hat\\gamma = @{f.bet.m.gamma}$ (\\% a day), $\\hat\\beta = @{f.bet.m.beta}$, $\\hat\\delta_0 = @{f.bet.m.delta0}$',
      'Pasul 4: $\\hat\\gamma = @{f.bet.m.gamma}$ (\\% pe zi), $\\hat\\beta = @{f.bet.m.beta}$, $\\hat\\delta_0 = @{f.bet.m.delta0}$')) + ql('SFM_ch3_estimation'))

D.frame(T('Maximum Likelihood', 'Verosimilitatea maximă'), items(
    (T('Log-likelihood of a sample $r_1, \\dots, r_n$: $\\ell(\\theta) = \\sum_{t=1}^n \\ln f(r_t; \\theta)$, $\\theta = (\\alpha, \\beta, \\gamma, \\delta_0)$',
       'Log-verosimilitatea unui eșantion $r_1, \\dots, r_n$: $\\ell(\\theta) = \\sum_{t=1}^n \\ln f(r_t; \\theta)$, $\\theta = (\\alpha, \\beta, \\gamma, \\delta_0)$'),
     [T('$\\theta$: the vector of the four parameters; $f(r_t; \\theta)$: the stable density at $r_t$; $\\hat\\theta$ maximises $\\ell$ numerically, starting from the McCulloch estimate',
        '$\\theta$: vectorul celor patru parametri; $f(r_t; \\theta)$: densitatea stabilă în $r_t$; $\\hat\\theta$ maximizează numeric $\\ell$, pornind de la estimarea McCulloch'),
      T('standard errors from the curvature of $\\ell$: the inverse of the Hessian, the matrix of second derivatives of $\\ell$', 'erorile standard din curbura lui $\\ell$: inversa hessianei, matricea derivatelor a doua ale lui $\\ell$')]),
    (T('The density $f$ is computed from $\\varphi$', 'Densitatea $f$ se calculează din $\\varphi$'),
     [T('$f(x) = \\frac{1}{2\\pi}\\int e^{-itx}\\varphi(t)\\,dt$, evaluated on a grid with the FFT (fast Fourier transform) \\refMDC',
        '$f(x) = \\frac{1}{2\\pi}\\int e^{-itx}\\varphi(t)\\,dt$, evaluată pe o grilă cu FFT (fast Fourier transform, transformata Fourier rapidă) \\refMDC'),
      T('or by direct integration \\refNolanA; \\texttt{levy\\_stable.fit} does this and needs minutes for a few thousand returns',
        'sau prin integrare directă \\refNolanA; \\texttt{levy\\_stable.fit} face acest calcul și durează cîteva minute pentru cîteva mii de randamente')]),
    T('ML in S0 is regular for $\\alpha \\in (0, 2)$: standard errors shrink like $1/\\sqrt{n}$ \\refNolanML',
      'Estimarea ML în S0 este regulată pentru $\\alpha \\in (0, 2)$: erorile standard scad ca $1/\\sqrt{n}$ \\refNolanML')))

chart(T('McCulloch vs ML in a Monte Carlo Study', 'McCulloch și ML într-un studiu Monte Carlo'), 'sfm_ch3_estimators_mc', 'SFM_ch3_estimation', [
    T('200 samples of $S(1.7, 0, 1, 0)$ for each size; both estimators are centred on 1.7',
      '200 de eșantioane din $S(1.7, 0, 1, 0)$ pentru fiecare mărime; ambii estimatori sînt centrați pe 1,7'),
    T('Standard deviation of $\\hat\\alpha$: $n = 500$: McCulloch $@{mc.500.mcculloch.sd}$, ML $@{mc.500.ml.sd}$; $n = 2500$: $@{mc.2500.mcculloch.sd}$ vs $@{mc.2500.ml.sd}$',
      'Abaterea standard a lui $\\hat\\alpha$: $n = 500$: McCulloch $@{mc.500.mcculloch.sd}$, ML $@{mc.500.ml.sd}$; $n = 2500$: $@{mc.2500.mcculloch.sd}$ față de $@{mc.2500.ml.sd}$'),
    T('ML is about $@{mc.ratio500}$ times more precise for the same data', 'ML este de circa $@{mc.ratio500}$ ori mai precisă pe aceleași date')], h='0.60\\textheight')

D.recap(('Estimation', 'Estimarea'), [
    T('McCulloch: five quantiles, two ratios, four tables; fast and robust', 'McCulloch: cinci cuantile, două rapoarte, patru tabele; rapidă și robustă'),
    T('ML: numerical density from $\\varphi$, numerical maximisation; the most precise', 'ML: densitate numerică din $\\varphi$, maximizare numerică; cea mai precisă'),
    T('Report the parameterisation (S0 or S1) and the units of $\\gamma$ and $\\delta$', 'Raportați parametrizarea (S0 sau S1) și unitățile lui $\\gamma$ și $\\delta$')])

# =============================================================================
# 8. AJUSTARE PE RANDAMENTE REALE
# =============================================================================
D.section('Fitting Real Returns', 'Estimarea pe randamente reale')


def frow(k):
    return (f'{NAMES[k]} & @{{f.{k}.n}} & $@{{f.{k}.m.alpha}}$ & $@{{f.{k}.s.alpha}}$ ($@{{f.{k}.s.se_alpha}}$) & '
            f'$@{{f.{k}.s.beta}}$ ($@{{f.{k}.s.se_beta}}$) & $@{{f.{k}.s.gamma}}$ & $@{{f.{k}.s.delta0}}$ & $@{{f.{k}.s.delta1}}$')


D.frame(T('Stable Fits of Daily Log Returns', 'Legi stabile estimate pe randamentele logaritmice zilnice'), table(
    'lrrrrrrr', T('& $n$ & $\\hat\\alpha$ McC. & $\\hat\\alpha$ ML (SE) & $\\hat\\beta$ ML (SE) & $\\hat\\gamma$ & $\\hat\\delta_0$ & $\\hat\\delta_1$',
                  '& $n$ & $\\hat\\alpha$ McC. & $\\hat\\alpha$ ML (SE) & $\\hat\\beta$ ML (SE) & $\\hat\\gamma$ & $\\hat\\delta_0$ & $\\hat\\delta_1$'),
    [frow(k) for k in ASSETS], size='footnotesize') + items(
    T('Daily log returns in \\%; indices @{f.bet.y0}--@{y1}, Bitcoin @{f.btc.y0}--@{y1} (7 days a week); data: EODHD',
      'Randamente logaritmice zilnice în \\%; indici @{f.bet.y0}--@{y1}, Bitcoin @{f.btc.y0}--@{y1} (7 zile pe săptămînă); date: EODHD'),
    T('McC.: McCulloch; ML in S0 with standard errors; $\\hat\\gamma$, $\\hat\\delta_0$, $\\hat\\delta_1$ in \\% a day',
      'McC.: McCulloch; ML în S0, cu erori standard; $\\hat\\gamma$, $\\hat\\delta_0$, $\\hat\\delta_1$ în \\% pe zi')) + ql('SFM_ch3_fit_returns'))

D.frame(T('Reading the Estimates', 'Interpretarea estimărilor'), items(
    (T('$\\hat\\alpha$ well below 2 for all four series', '$\\hat\\alpha$ mult sub 2 pentru toate cele patru serii'),
     [T('S\\&P 500: $(2 - @{f.sp500.s.alpha})/@{f.sp500.s.se_alpha} \\approx @{f.sp500.s.z}$ standard errors from the Normal case',
        'S\\&P 500: $(2 - @{f.sp500.s.alpha})/@{f.sp500.s.se_alpha} \\approx @{f.sp500.s.z}$ erori standard față de cazul Normal'),
      T('the heaviest tails: @{low.alpha} ($\\hat\\alpha = @{f.btc.s.alpha}$)', 'cele mai groase cozi: @{low.alpha} ($\\hat\\alpha = @{f.btc.s.alpha}$)')]),
    (T('$\\hat\\beta < 0$ for the S\\&P 500 and the DAX: the left tail is heavier (crashes)', '$\\hat\\beta < 0$ pentru S\\&P 500 și DAX: coada stîngă este mai groasă (crahuri)'),
     [T('BET and Bitcoin: $\\hat\\beta$ within two standard errors of 0', 'BET și Bitcoin: $\\hat\\beta$ la mai puțin de două erori standard de 0')]),
    T('$\\hat\\gamma$: Bitcoin $@{f.btc.s.gamma}\\%$ vs S\\&P 500 $@{f.sp500.s.gamma}\\%$ a day; compare the sample standard deviations $@{f.btc.sd}\\%$ and $@{f.sp500.sd}\\%$',
      '$\\hat\\gamma$: Bitcoin $@{f.btc.s.gamma}\\%$ pe zi, față de $@{f.sp500.s.gamma}\\%$ pentru S\\&P 500; comparați abaterile standard de selecție $@{f.btc.sd}\\%$ și $@{f.sp500.sd}\\%$'),
    T('McCulloch gives lower $\\hat\\alpha$ than ML for every series: the two methods weigh the tails differently',
      'McCulloch dă un $\\hat\\alpha$ mai mic decît ML pentru fiecare serie: cele două metode ponderează diferit cozile')))

chart(T('Fitted Densities on a Log Scale (1/2)', 'Densități estimate, pe scară logaritmică (1/2)'), 'sfm_ch3_fit_density_1', 'SFM_ch3_fit_returns', [
    T('Normal (red): far too few large returns; Student-t (green) and stable (blue) both follow the body',
      'Distribuția Normală (roșu): mult prea puține randamente mari; distribuțiile Student-t (verde) și stabilă (albastru) descriu bine corpul')],
    h='0.70\\textheight')

chart(T('Fitted Densities on a Log Scale (2/2)', 'Densități estimate, pe scară logaritmică (2/2)'), 'sfm_ch3_fit_density_2', 'SFM_ch3_fit_returns', [
    T('Far tails: the stable density stays above the data points; the Student-t is closer', 'Cozile îndepărtate: densitatea stabilă rămîne deasupra punctelor; distribuția Student-t este mai aproape de ele')],
    h='0.70\\textheight')

chart(T('QQ Plots against the Three Models (1/2)', 'QQ plots față de cele trei modele (1/2)'), 'sfm_ch3_qq_real_1', 'SFM_ch3_fit_returns', [
    T('A QQ (quantile--quantile) plot puts the empirical quantiles against the model quantiles; a good model lies on the 45-degree line',
      'Un QQ plot (graficul cuantilă--cuantilă) reprezintă cuantilele empirice în funcție de cele ale modelului; pentru un model bun, punctele se află pe prima bisectoare')],
    h='0.70\\textheight')

chart(T('QQ Plots against the Three Models (2/2)', 'QQ plots față de cele trei modele (2/2)'), 'sfm_ch3_qq_real_2', 'SFM_ch3_fit_returns', [
    T('Normal: too-short tails (steep ends); stable: too-long tails (flat ends, model quantiles far beyond the data); Student-t: the closest',
      'Distribuția Normală: cozi prea scurte (capete abrupte); distribuția stabilă: cozi prea lungi (capete plate, cuantile ale modelului mult dincolo de date); Student-t: cea mai apropiată')],
    h='0.70\\textheight')

chart(T('The Left Tail on Log-Log Axes (1/2)', 'Coada stîngă pe axe log-log (1/2)'), 'sfm_ch3_tails_real_1', 'SFM_ch3_fit_returns', [
    T('Points: share of days with a loss above $x$; lines: the same probability under each fitted model',
      'Punctele: proporția zilelor cu o pierdere peste $x$; liniile: aceeași probabilitate conform fiecărui model estimat')],
    h='0.70\\textheight')

chart(T('The Left Tail on Log-Log Axes (2/2)', 'Coada stîngă pe axe log-log (2/2)'), 'sfm_ch3_tails_real_2', 'SFM_ch3_fit_returns', [
    T('The empirical tail bends down faster than the stable line: the tail index of the data is larger than $\\hat\\alpha$',
      'Coada empirică se curbează în jos mai repede decît dreapta stabilă: tail index-ul datelor este mai mare decît $\\hat\\alpha$')],
    h='0.70\\textheight')


def tcrow(k, x):
    return (f'{NAMES[k]} & ${x}\\%$ & @{{tc.{k}.obs{x}}} & $@{{tc.{k}.n{x}}}$ & $@{{tc.{k}.t{x}}}$ & $@{{tc.{k}.s{x}}}$')


D.frame(T('Counting Large Losses', 'Numărarea pierderilor mari'), table(
    'lrrrrr', T('& loss above & observed & Normal & Student-t & stable', '& pierdere peste & observate & Normală & Student-t & stabilă'),
    [tcrow('sp500', 5), tcrow('sp500', 10), tcrow('sp500', 15), tcrow('bet', 10), tcrow('btc', 10), tcrow('btc', 20)], size='footnotesize') + items(
    T('Expected number of days $= n \\times P(r < -x)$ under each fitted model', 'Numărul așteptat de zile $= n \\times P(r < -x)$ conform fiecărui model estimat'),
    T('Normal: almost no large losses; stable: too many, and the gap grows with the threshold; Student-t: closest to the counts',
      'Distribuția Normală: aproape nicio pierdere mare; distribuția stabilă: prea multe, iar diferența crește odată cu pragul; Student-t: cea mai apropiată de valorile observate'),
    T('S\\&P 500: @{tc.sp500.obs10} day below $-10\\%$ in @{f.sp500.n} days; the stable fit expects $@{tc.sp500.s10}$',
      'S\\&P 500, @{f.sp500.n} zile: zile sub $-10\\%$ observate @{tc.sp500.obs10}, prevăzute de legea stabilă estimată $@{tc.sp500.s10}$')) + ql('SFM_ch3_fit_returns'))


def varrow(k):
    return (f'{NAMES[k]} & $@{{tc.{k}.var1.e}}$ & $@{{tc.{k}.var1.n}}$ & $@{{tc.{k}.var1.t}}$ & $@{{tc.{k}.var1.s}}$ & '
            f'$@{{tc.{k}.var01.e}}$ & $@{{tc.{k}.var01.n}}$ & $@{{tc.{k}.var01.t}}$ & $@{{tc.{k}.var01.s}}$')


D.frame(T('VaR 1\\% and VaR 0.1\\% under the Three Models', 'VaR 1\\% și VaR 0,1\\% conform celor trei modele'), table(
    'l|rrrr|rrrr', T('& \\multicolumn{4}{c|}{VaR 1\\% (\\%)} & \\multicolumn{4}{c}{VaR 0.1\\% (\\%)} \\\\ & data & Normal & t & stable & data & Normal & t & stable',
                     '& \\multicolumn{4}{c|}{VaR 1\\% (\\%)} & \\multicolumn{4}{c}{VaR 0,1\\% (\\%)} \\\\ & date & Normală & t & stabilă & date & Normală & t & stabilă'),
    [varrow(k) for k in ASSETS], size='footnotesize') + items(
    T('VaR at level $p$: $\\mathrm{VaR}_p = -q_p$, the daily loss exceeded with probability $p$ (Chapter 10); here $p$, because $\\alpha$ is the stable index',
      'VaR la nivelul $p$: $\\mathrm{VaR}_p = -q_p$, pierderea zilnică depășită cu probabilitatea $p$ (Capitolul 10); aici notăm nivelul cu $p$, deoarece $\\alpha$ desemnează indicele de stabilitate'),
    T('VaR 1\\%: the Normal model is too low, the stable model too high', 'VaR 1\\%: modelul Normal dă valori prea mici, modelul stabil valori prea mari'),
    T('VaR 0.1\\%: the stable model gives roughly two to four times the empirical quantile', 'VaR 0,1\\%: modelul stabil dă valori de aproximativ două pînă la patru ori mai mari decît cuantila empirică')) + ql('SFM_ch3_fit_returns'))


def aicrow(k):
    return f'{NAMES[k]} & @{{f.{k}.aic.n}} & @{{f.{k}.aic.t}} & @{{f.{k}.aic.s}} & $@{{f.{k}.t.nu}}$'


D.frame(T('Which Model Fits Best?', 'Compararea celor trei modele'), table(
    'lrrrr', T('& AIC Normal & AIC Student-t & AIC stable & $\\hat\\nu$ (Student-t)', '& AIC Normală & AIC Student-t & AIC stabilă & $\\hat\\nu$ (Student-t)'),
    [aicrow(k) for k in ASSETS], size='footnotesize') + items(
    T('AIC (Akaike information criterion) $= 2k - 2\\ell(\\hat\\theta)$, with $k$ the number of parameters; lower is better',
      'AIC (Akaike information criterion, criteriul informațional Akaike) $= 2k - 2\\ell(\\hat\\theta)$, cu $k$ numărul de parametri; se preferă valoarea mai mică'),
    T('Both heavy-tailed models beat the Normal distribution by thousands of AIC points', 'Ambele modele cu cozi groase sînt preferate distribuției Normale, cu diferențe de mii de puncte AIC'),
    T('Student-t wins for every series, with fewer parameters; S\\&P 500: by @{f.sp500.aic.ts} points',
      'Student-t este preferată pentru fiecare serie, deși are mai puțini parametri; la S\\&P 500, diferența este de @{f.sp500.aic.ts} de puncte'),
    T('$\\hat\\nu$ between 2 and 3.1: finite variance, but few finite moments beyond it', '$\\hat\\nu$ între 2 și 3,1: varianță finită, dar aproape niciun moment finit de ordin superior')) + ql('SFM_ch3_fit_returns'))

D.recap(('Fitting Real Returns', 'Estimarea pe randamente reale'), [
    T('All four series: $\\hat\\alpha$ between @{f.btc.s.alpha} and @{f.dax.s.alpha}, far from 2', 'Toate cele patru serii: $\\hat\\alpha$ între @{f.btc.s.alpha} și @{f.dax.s.alpha}, departe de 2'),
    T('The stable law fits the body much better than the Normal distribution', 'Legea stabilă descrie corpul distribuției mult mai bine decît distribuția Normală'),
    T('It overstates the far tails: too many extreme losses, VaR 0.1\\% too high', 'Supraestimează cozile îndepărtate: prea multe pierderi extreme, VaR 0,1\\% prea mare'),
    T('By AIC, the Student-t distribution fits better', 'După AIC, distribuția Student-t este preferată')])

# =============================================================================
# 9. CRITICA
# =============================================================================
D.section('The Critique: Finite or Infinite Variance?', 'Critica: varianță finită sau infinită?')

chart(T('Test 1: Does $\\alpha$ Stay the Same When Returns Are Summed?', 'Testul 1: rămîne $\\alpha$ la fel cînd randamentele se adună?'), 'sfm_ch3_aggregation', 'SFM_ch3_aggregation', [
    T('If daily returns were i.i.d.\\ stable, weekly and monthly sums would have the \\textbf{same} $\\alpha$ (stability)',
      'Dacă randamentele zilnice ar fi i.i.d.\\ stabile, sumele săptămînale și lunare ar avea \\textbf{același} $\\alpha$ (stabilitatea)'),
    T('ML estimates: S\\&P 500 $@{ag.sp500.daily} \\to @{ag.sp500.weekly} \\to @{ag.sp500.monthly}$; Bitcoin $@{ag.btc.daily} \\to @{ag.btc.weekly} \\to @{ag.btc.monthly}$ (at the bound 2)',
      'Estimări ML: S\\&P 500 $@{ag.sp500.daily} \\to @{ag.sp500.weekly} \\to @{ag.sp500.monthly}$; Bitcoin $@{ag.btc.daily} \\to @{ag.btc.weekly} \\to @{ag.btc.monthly}$ (la limita 2)')])

D.frame(T('Reading the Aggregation Test', 'Interpretarea testului de agregare'), items(
    (T('$\\hat\\alpha$ rises with the horizon for the S\\&P 500, DAX and Bitcoin', '$\\hat\\alpha$ crește cu orizontul pentru S\\&P 500, DAX și Bitcoin'),
     [T('DAX: $@{ag.dax.daily} \\to @{ag.dax.weekly} \\to @{ag.dax.monthly}$; the daily and monthly intervals do not overlap for the S\\&P 500',
        'DAX: $@{ag.dax.daily} \\to @{ag.dax.weekly} \\to @{ag.dax.monthly}$; pentru S\\&P 500, intervalele de încredere ale orizontului zilnic și lunar nu se suprapun')]),
    (T('BET: $@{ag.bet.daily} \\to @{ag.bet.weekly} \\to @{ag.bet.monthly}$, no clear rise', 'BET: $@{ag.bet.daily} \\to @{ag.bet.weekly} \\to @{ag.bet.monthly}$, fără o creștere clară'),
     [T('monthly samples are short (@{ag.bet.monthly.n} months): wide intervals', 'eșantioanele lunare sînt scurte (@{ag.bet.monthly.n} de luni): intervale largi')]),
    T('The same pattern in the classic studies: \\refOfficer, \\refAB', 'Același tipar în studiile clasice: \\refOfficer, \\refAB'),
    T('A rising $\\alpha$ is what the GCLT predicts for sums of finite-variance returns that converge slowly to the Normal distribution',
      'Un $\\alpha$ crescător este exact ceea ce prevede GCLT pentru sume de randamente cu varianță finită care converg lent la distribuția Normală')))

chart(T('Test 2: Does the Sample Variance Settle?', 'Testul 2: se stabilizează varianța de selecție?'), 'sfm_ch3_running_var_real', 'SFM_ch3_aggregation', [
    T('S\\&P 500 (blue): after the jumps of 2008 and 2020 the running variance settles near $@{rvr.data}$',
      'S\\&P 500 (albastru): după salturile din 2008 și 2020, varianța cumulată se stabilizează în jur de $@{rvr.data}$'),
    T('Paths simulated from the fitted stable law, same length: final variances between $@{rvr.simmin}$ and $@{rvr.simmax}$, far above the data',
      'Traiectorii de aceeași lungime simulate din legea stabilă estimată: varianțe finale între $@{rvr.simmin}$ și $@{rvr.simmax}$, mult peste date')])

D.frame(T('Test 3: How Heavy Is the Far Tail?', 'Testul 3: cît de groasă este coada îndepărtată?'), items(
    (T('The stable $\\alpha$ is driven by the whole distribution, mostly by the body', '$\\alpha$ stabil este determinat de întreaga distribuție, mai ales de corp'),
     [T('the tail index of the far tail alone is estimated with the Hill estimator \\refHill\\ (Chapter 5)',
        'tail index-ul cozii îndepărtate se estimează separat, cu estimatorul Hill \\refHill\\ (Capitolul 5)')]),
    (T('For stock indices the far tail decays roughly like $x^{-3}$, the ``inverse cubic law\'\' \\refGopi',
       'Pentru indicii bursieri, coada îndepărtată scade aproximativ ca $x^{-3}$, „legea cubică inversă” \\refGopi'),
     [T('a tail index near 3 means finite variance, and no stable law has it', 'un tail index în jur de 3 înseamnă varianță finită, iar nicio lege stabilă nu îl are')]),
    T('Our counts agree: the stable fits expect many more losses beyond $10\\%$ than observed', 'Numărul pierderilor observate confirmă: legile stabile estimate prevăd mult mai multe pierderi peste $10\\%$ decît s-au observat'),
    T('\\refLLW: further evidence against the stable model from the behaviour of sample moments',
      '\\refLLW: alte dovezi împotriva modelului stabil, din comportamentul momentelor de selecție')))

D.frame(T('Why Do Daily Returns Look Stable?', 'Originea aspectului stabil al randamentelor zilnice'), items(
    (T('\\textbf{Volatility clustering} (Chapters 2 and 9): calm and turbulent periods alternate', '\\textbf{Volatility clustering} (Capitolele 2 și 9): perioadele calme și cele agitate alternează'),
     [T('a mixture of Normal laws with changing variance has heavy tails and a finite variance \\refBG',
        'un amestec de legi Normale cu varianță variabilă are cozi groase și varianță finită \\refBG'),
      T('fitted to one stable law, the mixture gives $\\hat\\alpha < 2$', 'dacă pe datele generate de amestec se estimează o singură lege stabilă, se obține $\\hat\\alpha < 2$')]),
    (T('\\textbf{Truncated and tempered stable laws}: stable at moderate scales, lighter tails far out',
       '\\textbf{Legi stabile trunchiate și temperate}: stabile la scări moderate, cu cozi mai subțiri în zona extremă'),
     [T('\\refMS\\ fitted a truncated Lévy flight to the S\\&P 500', '\\refMS\\ au estimat un zbor Lévy trunchiat (truncated Lévy flight) pe S\\&P 500'),
      T('\\refGS: such laws explain why returns look stable at short horizons and Normal at long ones',
        '\\refGS: astfel de legi explică de ce randamentele par stabile pe orizonturi scurte și Normale pe orizonturi lungi')])))

D.frame(T('Verdict', 'Verdictul'), items(
    (T('\\textbf{Mandelbrot was right} about the facts', '\\textbf{Mandelbrot a avut dreptate} în privința faptelor'),
     [T('daily returns have heavy tails; the Normal distribution fails badly', 'randamentele zilnice au cozi groase; distribuția Normală descrie foarte slab datele')]),
    (T('\\textbf{The infinite variance is not supported}', '\\textbf{Varianța infinită nu este susținută}'),
     [T('$\\hat\\alpha$ rises with aggregation; the sample variance settles; the far tail is lighter than any stable tail',
        '$\\hat\\alpha$ crește prin agregare; varianța de selecție se stabilizează; coada îndepărtată este mai subțire decît orice coadă stabilă')]),
    (T('Where stable laws remain useful', 'Unde rămîn utile legile stabile'),
     [T('a benchmark for heavy tails; simulation; the scaling idea; data with truly infinite variance (some insurance losses, some crypto tokens)',
        'un etalon pentru cozi groase; simulare; ideea de scalare; date cu varianță cu adevărat infinită (unele daune de asigurări, unele tokenuri cripto)'),
      T('the GCLT: a reminder that $\\sqrt{n}$-rules need a finite variance', 'GCLT: o reamintire că regulile cu $\\sqrt{n}$ cer varianță finită')]),
    T('Next: Chapter 5 estimates the tail index directly with extreme value theory', 'Urmează: Capitolul 5 estimează direct tail index-ul cu teoria valorilor extreme')))

D.recap(('The Critique', 'Critica'), [
    T('Stability predicts the same $\\alpha$ at every horizon: the data show a rising $\\alpha$', 'Stabilitatea implică același $\\alpha$ la orice orizont: datele arată un $\\alpha$ crescător'),
    T('The sample variance of real returns settles; that of stable samples does not', 'Varianța de selecție a randamentelor reale se stabilizează; cea a eșantioanelor stabile, nu'),
    T('Volatility clustering and tempered tails explain the apparent $\\alpha < 2$', 'Volatility clustering și cozile temperate explică valoarea aparentă $\\alpha < 2$'),
    T('Heavy tails: yes; infinite variance: no', 'Cozi groase: da; varianță infinită: nu')])

# =============================================================================
# 10. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Have the tails of Bitcoin become lighter as the market matured?}', '\\textbf{Au devenit mai subțiri cozile Bitcoin pe măsură ce piața s-a maturizat?}'),
     [T('more participants, futures and ETFs (exchange-traded funds) since 2017--2024', 'mai mulți participanți, contracte futures și ETF-uri (exchange-traded funds, fonduri tranzacționate la bursă) în perioada 2017--2024'),
      T('if the tails got lighter, risk models fitted on old data overstate today\'s risk', 'dacă cozile au devenit mai subțiri, modelele de risc estimate pe date vechi supraestimează riscul de azi')]),
    T('Why it is open: $\\hat\\alpha$ moves with volatility regimes, and yearly samples are short', 'De ce rămîne deschisă: $\\hat\\alpha$ variază odată cu regimurile de volatilitate, iar eșantioanele anuale sînt scurte'),
    T('Full sample: $\\hat\\alpha = @{f.btc.s.alpha}$ (SE $@{f.btc.s.se_alpha}$), the lowest of our four series', 'Eșantionul complet: $\\hat\\alpha = @{f.btc.s.alpha}$ (SE $@{f.btc.s.se_alpha}$), cel mai mic dintre cele patru serii'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: list papers on tail estimates for crypto assets and summarise their methods',
      '\\textbf{Literatura}: inventarierea lucrărilor despre estimarea cozilor activelor cripto și rezumarea metodelor lor'),
    T('\\textbf{Code}: draft a rolling-window McCulloch estimator with bootstrap intervals', '\\textbf{Cod}: o primă versiune a unui estimator McCulloch pe ferestre mobile, cu intervale bootstrap'),
    T('\\textbf{Robustness}: propose other windows, ML instead of quantiles, Student-t $\\nu$ instead of $\\alpha$', '\\textbf{Robustețe}: propunerea altor ferestre, ML în loc de cuantile, $\\nu$ din Student-t în loc de $\\alpha$'),
    T('\\textbf{Explanation}: a first draft of the interpretation of a table of yearly estimates', '\\textbf{Explicație}: o primă versiune a interpretării unui tabel de estimări anuale'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that estimates the stable alpha of daily log returns on rolling 365-day windows with McCulloch (1986), and adds 95\\% bootstrap intervals.}',
        '\\aiprompt{Write a Python function that estimates the stable alpha of daily log returns on rolling 365-day windows with McCulloch (1986), and adds 95\\% bootstrap intervals.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Parameterisation: S0 or S1; scipy uses S1 by default; the location differs by $\\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$',
      'Parametrizarea: S0 sau S1; scipy folosește implicit S1; poziția diferă cu $\\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$'),
    T('Units: returns in \\% or in decimals; $\\gamma$ and $\\delta$ change, $\\alpha$ and $\\beta$ do not', 'Unitățile: randamente în \\% sau în zecimale; $\\gamma$ și $\\delta$ se schimbă, $\\alpha$ și $\\beta$ nu'),
    T('Overlapping windows are not independent: do not read the rolling estimates as separate tests',
      'Ferestrele suprapuse nu sînt independente: nu citiți estimările mobile ca teste separate'),
    T('Numbers: recompute one window with a second method (ML or the quantile tables)', 'Rezultatele numerice: recalculați o fereastră cu o a doua metodă (ML sau tabelele de cuantile)'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Seed', 'Idee de proiect'), items(
    (T('\\textbf{Question}: is the tail of Bitcoin returns lighter now than in 2014--2018?', '\\textbf{Întrebarea}: este coada randamentelor Bitcoin mai subțire acum decît în 2014--2018?'),
     [T('data: Bitcoin daily closes from the course data (EODHD), 2014--2026', 'date: prețurile de închidere zilnice ale Bitcoin din datele cursului (EODHD), 2014--2026')]),
    (T('Steps', 'Pași'),
     [T('estimate $\\alpha$ by McCulloch and by ML for each calendar year, with bootstrap intervals', 'estimați $\\alpha$ prin McCulloch și prin ML pentru fiecare an calendaristic, cu intervale bootstrap'),
      T('repeat on returns divided by a rolling volatility: does the trend survive?', 'repetați pe randamente împărțite la o volatilitate mobilă: se menține tendința?'),
      T('compare with the Student-t $\\hat\\nu$ of each year', 'comparați cu $\\hat\\nu$ din Student-t pentru fiecare an')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show',
      'Livrabile: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice folosire a AI și listați erorile AI pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei principale'), items(
    T('Stable laws are the only laws whose sums keep the same shape, and the only limits of normalised sums (GCLT)',
      'Legile stabile sînt singurele legi ale căror sume își păstrează forma și singurele limite ale sumelor normalizate (GCLT)'),
    T('Four parameters: $\\alpha$ tails, $\\beta$ skewness, $\\gamma$ scale, $\\delta$ location; know your parameterisation (S0 or S1)',
      'Patru parametri: $\\alpha$ cozi, $\\beta$ asimetrie, $\\gamma$ scală, $\\delta$ poziție; precizați întotdeauna parametrizarea (S0 sau S1)'),
    T('$\\alpha < 2$: power-law tails and infinite variance; $\\alpha \\le 1$: no mean', '$\\alpha < 2$: cozi de tip putere și varianță infinită; $\\alpha \\le 1$: nu există media'),
    T('Simulate with CMS; estimate with McCulloch quantiles or ML', 'Simulați cu CMS; estimați cu cuantilele McCulloch sau cu ML'),
    T('Daily returns: $\\hat\\alpha \\approx 1.4$--$1.6$, much better than the Normal distribution in the body', 'Randamentele zilnice: $\\hat\\alpha \\approx 1{,}4$--$1{,}6$; legea stabilă descrie corpul distribuției mult mai bine decît distribuția Normală'),
    T('But $\\hat\\alpha$ rises with aggregation and the variance settles: heavy tails, finite variance',
      'Dar $\\hat\\alpha$ crește prin agregare, iar varianța se stabilizează: cozi groase, varianță finită')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Stability', 'Stabilitate') + ' & $aX_1 + bX_2 \\overset{d}{=} cX + d$, \\quad $c^\\alpha = a^\\alpha + b^\\alpha$',
     T('Sum of $n$ copies', 'Suma a $n$ copii') + ' & $X_1 + \\dots + X_n \\overset{d}{=} n^{1/\\alpha}X + d_n$',
     T('Characteristic function (S1)', 'Funcția caracteristică (S1)') + ' & $\\exp\\{-\\gamma^\\alpha|t|^\\alpha[1 - i\\beta\\,\\mathrm{sign}(t)\\tan\\frac{\\pi\\alpha}{2}] + i\\delta_1 t\\}$',
     T('S0 vs S1', 'S0 și S1') + ' & $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan\\frac{\\pi\\alpha}{2}$',
     T('Normal case', 'Cazul Normal') + ' & $S(2, \\beta, \\gamma, \\delta) = N(\\delta, 2\\gamma^2)$',
     T('Tails', 'Cozi') + ' & $P(X > x) \\sim \\gamma^\\alpha c_\\alpha(1 + \\beta)x^{-\\alpha}$',
     T('Moments', 'Momente') + ' & $E|X|^p < \\infty \\iff p < \\alpha$ \\quad ($\\alpha < 2$)',
     'CMS ($\\beta = 0$) & $\\dfrac{\\sin(\\alpha V)}{(\\cos V)^{1/\\alpha}}\\Big(\\dfrac{\\cos((1-\\alpha)V)}{W}\\Big)^{(1-\\alpha)/\\alpha}$',
     'McCulloch & $\\nu_\\alpha = \\dfrac{q_{0.95} - q_{0.05}}{q_{0.75} - q_{0.25}}$, \\quad $\\nu_\\beta = \\dfrac{q_{0.95} + q_{0.05} - 2q_{0.5}}{q_{0.95} - q_{0.05}}$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: daily returns are i.i.d.\\ $S(1.5, 0, \\gamma, 0)$; by what factor does the scale grow from 1 day to 20 days?',
       '\\textbf{Întrebare}: randamentele zilnice sînt i.i.d.\\ $S(1.5, 0, \\gamma, 0)$; de cîte ori crește scala de la 1 zi la 20 de zile?'),
     [T('\\textbf{Answer}: $20^{1/1.5} = @{ex.h.1.5}$, i.e.\\ $@{chk.h}$ times the square-root-of-time factor $\\sqrt{20}$',
        '\\textbf{Răspuns}: $20^{1/1.5} = @{ex.h.1.5}$, adică de $@{chk.h}$ ori mai mult decît factorul $\\sqrt{20}$ al regulii rădăcinii pătrate a timpului')]),
    (T('\\textbf{Question}: a fitted law has $\\alpha = 1.2$; do the mean and the variance exist?', '\\textbf{Întrebare}: o lege estimată are $\\alpha = 1.2$; există media și varianța?'),
     [T('\\textbf{Answer}: the mean exists ($1 < 1.2$), the variance does not ($2 > 1.2$)', '\\textbf{Răspuns}: media există ($1 < 1.2$), varianța nu ($2 > 1.2$)')]),
    (T('\\textbf{Question}: scipy reports $\\delta = 0.05$ with $\\alpha = 1.7$, $\\beta = -0.2$, $\\gamma = 0.6$; what is $\\delta_0$?',
       '\\textbf{Întrebare}: scipy raportează $\\delta = 0.05$ cu $\\alpha = 1.7$, $\\beta = -0.2$, $\\gamma = 0.6$; cît este $\\delta_0$?'),
     [T('\\textbf{Answer}: scipy uses S1, so $\\delta_0 = 0.05 + (-0.2)(0.6)\\tan(0.85\\pi) = 0.05 + @{ex.d0.add} = @{ex.d0}$',
        '\\textbf{Răspuns}: scipy folosește S1, deci $\\delta_0 = 0.05 + (-0.2)(0.6)\\tan(0.85\\pi) = 0.05 + @{ex.d0.add} = @{ex.d0}$')]),
    T('Next: Chapter 4, probability for finance', 'Urmează: Capitolul 4, probabilități pentru finanțe')))

D.references(BIB)

if __name__ == '__main__':
    for path in D.write(V):
        ro_tuples(path)
