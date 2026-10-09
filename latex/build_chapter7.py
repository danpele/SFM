r"""
build_chapter7.py -- Capitolul 7 (Ipoteza piețelor eficiente, mersul aleator și testele VR), EN + RO dintr-o singură
sursă
====================================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_07/ch7_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter7_efficient_markets_random_walk_vr.tex
  RO/Cursuri/capitol7_piete_eficiente_mers_aleator_vr.tex
Rulare:
  python3 Quantlets/Ch_07/generate_all_charts.py
  python3 latex/build_chapter7.py && python3 latex/sfm_build.py compile 7
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Quantlets', 'common'))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch7_common import ASSETS, BIB, NAMES, QLURL, REFS, SHORT, STOCKS, T, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(7, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_07}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def side(title, fig, folder, bullets, w=0.56, size='footnotesize', h='0.74\\textheight'):
    """Chart on the left, bullets on the right, Quantlet link below."""
    left = f'\\begin{{center}}\n\\includegraphics[width=\\linewidth,height={h},keepaspectratio]{{{fig}.pdf}}\n\\end{{center}}'
    D.frame(title, cols(left, items(*bullets), wl=f'{w:.2f}', wr=f'{0.96 - w:.2f}') + '\n' + ql(folder), size)


PH = {
    'nyse': ('ch0_nyse_floor_1963.jpg', C + 'NY_stock_exchange_traders_floor_LC-U9-10548-6.jpg',
             "⟦Photo||Foto⟧: Thomas J. O'Halloran (1963), Library of Congress; ⟦public domain||domeniu public⟧; Wikimedia Commons"),
    'bachelier': ('ch0_bachelier.jpg', C + 'LouisBachelier.jpg',
                  '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧; ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'samuelson': ('ch7_samuelson.jpg', C + 'Paul_A._Samuelson,_economist.jpg',
                  '⟦Photo||Foto⟧: Bernard Gotfryd (1970--1975); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'fama': ('ch0_fama_2013.jpg', C + 'Eugene_Fama_at_Nobel_Prize,_2013.jpg',
             '⟦Photo||Foto⟧: Bengt Nyman (2013); CC BY 2.0; Wikimedia Commons'),
    'shiller': ('ch7_shiller_2013.jpg', C + 'Robert_J._Shiller_(50372668186).jpg',
                '⟦Photo||Foto⟧: Bengt Nyman (2013); CC BY 2.0; Wikimedia Commons'),
    'lo': ('ch7_lo_2012.jpg', C + 'Andrew_Lo_2012_Shankbone_2.JPG',
           '⟦Photo||Foto⟧: David Shankbone (2012); CC BY 3.0; Wikimedia Commons'),
    'bvb': ('ch5_bvb_palace_2019.jpg', C + 'Stock_Exchange_Palace_(Bucharest).jpg',
            '⟦Photo||Foto⟧: Neoclassicism Enthusiast (2019); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
sd = 100 * N['rw']['sd']
for q in (5, 20, 250):
    V.put(f'ex.sd{q}', sd * math.sqrt(q), 2)
    V.put(f'ex.sq{q}', math.sqrt(q), 2)
# ACF of AR(1), AR(2), MA(1)
V.put('ex.ma1', 0.5 / 1.25, 2)
V.put('ex.ar3', 0.9 ** 3, 3)
V.put('ex.ar2r1', 0.5 / 0.6, 3)
V.put('ex.ar2r2', 0.5 * 0.5 / 0.6 + 0.4, 3)
# Ljung-Box, step by step: T = 1000, rho = 0.06, -0.04, 0.03
rho_ex = [0.06, -0.04, 0.03]
V.put('ex.bp', 1000 * sum(r ** 2 for r in rho_ex), 2)
V.put('ex.lb', 1000 * 1002 * sum(r ** 2 / (1000 - k) for k, r in enumerate(rho_ex, 1)), 2)
V.put('ex.c3', 7.8147, 2)
for i, r in enumerate(rho_ex, 1):
    V.put(f'ex.r2.{i}', 1000 * r ** 2, 1)
# VR(5) from the first autocorrelations of the S&P 500
rs = N['acf']['sp500']['rho']
V.put('ex.vr2', 1 + rs[0], 3)
V.put('ex.vr5', 1 + 2 * sum((1 - k / 5) * rs[k - 1] for k in range(1, 5)), 3)
for i in range(4):
    V.put(f'ex.rs{i + 1}', rs[i], 3)
# Dickey-Fuller step by step: BET log price, constant, no lags
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Quantlets', 'Ch_07'))
import generate_all_charts as g   # noqa: E402
lp = g.log_price('bet').values
dy, yl = np.diff(lp), lp[:-1]
X = np.column_stack([np.ones(len(yl)), yl])
b, *_ = np.linalg.lstsq(X, dy, rcond=None)
u = dy - X @ b
se = math.sqrt(np.sum(u ** 2) / (len(dy) - 2) * np.linalg.inv(X.T @ X)[1, 1])
V.put('df.g', b[1], 5)
V.put('df.se', se, 5)
V.put('df.tau', b[1] / se, 2)
V.int('df.n', len(dy))
# size of a multiple test: m independent tests at 5%
V.put('mt.16', 100 * (1 - 0.95 ** 16), 0)
V.put('mt.exp', 16 * 0.05, 1)
V.put('mt.4', 100 * (1 - 0.95 ** 4), 1)
V.put('rw.share', 100 * np.mean(np.array(N['rw']['sim_final']) > N['rw']['final']), 0)
V.put('aa.ratio', N['acf_abs']['sp500']['abs1'] / abs(N['acf_abs']['sp500']['ret1']), 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can yesterday\'s prices tell us anything about tomorrow\'s return?',
       '\\textbf{Întrebarea}: ne pot spune prețurile de ieri ceva despre randamentul de mîine?'),
     [T('if yes, a simple rule could beat the market; if no, the market is efficient in the weak sense',
        'dacă da, o regulă simplă ar putea bate piața; dacă nu, piața este eficientă în sens slab'),
      T('Chapter 4 defined the random walk and the martingale; here we test them on real prices',
        'Capitolul 4 a definit mersul aleator și martingalul; aici le testăm pe prețuri reale')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('efficient markets: three forms, the joint-hypothesis problem', 'piețele eficiente: trei forme, problema ipotezei comune'),
      T('random walk hypotheses RW1, RW2, RW3 and the martingale', 'ipotezele de mers aleator RW1, RW2, RW3 și martingalul'),
      T('white noise and the autocorrelation function; Ljung--Box and runs tests', 'zgomotul alb și funcția de autocorelație; testele Ljung--Box și runs'),
      T('unit roots: ADF, Phillips--Perron, KPSS; prices against returns', 'rădăcini unitare: ADF, Phillips--Perron, KPSS; prețuri față de randamente'),
      T('variance-ratio tests; efficiency across markets and over time; anomalies', 'testele variance ratio; eficiența pe piețe diferite și în timp; anomalii')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('State the three forms of market efficiency and explain why every test is a joint test', 'Enunțați cele trei forme ale eficienței pieței și explicați de ce orice test este un test comun'),
    T('Tell RW1, RW2, RW3 and the martingale apart, and say which of them volatility clustering violates',
      'Deosebiți RW1, RW2, RW3 și martingalul și precizați care dintre aceste ipoteze sînt încălcate de volatility clustering'),
    T('Read a sample ACF with the right confidence band and compute Box--Pierce, Ljung--Box and runs statistics',
      'Citiți un ACF de selecție cu banda de încredere potrivită și calculați statisticile Box--Pierce, Ljung--Box și runs'),
    T('Run and combine ADF, Phillips--Perron and KPSS tests on prices and on returns', 'Aplicați și combinați testele ADF, Phillips--Perron și KPSS pe prețuri și pe randamente'),
    T('Compute a variance ratio and its robust test, for one horizon and for several horizons together',
      'Calculați un raport al varianțelor și testul lui robust, pentru un orizont și pentru mai multe orizonturi împreună'),
    T('Judge, on real data, which markets look efficient, and whether this changes over time',
      'Judecați, pe date reale, care piețe par eficiente și dacă acest lucru se schimbă în timp')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~11 (11.3 expectations and efficient markets, 11.5 random walk tests, 11.6 unit root tests)',
       'Manual: \\refFHH, cap.~11 (11.3 așteptări și piețe eficiente, 11.5 teste de mers aleator, 11.6 teste de rădăcină unitară)'),
     [T('exercises with solutions: \\refBHL, Ch.~11', 'exerciții rezolvate: \\refBHL, cap.~11')]),
    T('Econometrics of predictability: \\refCLM, Ch.~2 (random walks, variance ratios) and Ch.~4 (event studies)',
      'Econometria previzibilității: \\refCLM, cap.~2 (mers aleator, variance ratio) și cap.~4 (studii de eveniment)'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_07}',
       'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_07}'),
     [T('ported from the SFE Quantlets SFEtimewn, SFEacfar1, SFEacfar2, SFEacfma1 and SFEacfma2',
        'portate din Quantlet-urile SFE SFEtimewn, SFEacfar1, SFEacfar2, SFEacfma1 și SFEacfma2')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter7_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter7_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Efficiency of cryptocurrency markets -- A GMM-based analysis}{https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis}',
      'Curs video: \\quantinar{Efficiency of cryptocurrency markets -- A GMM-based analysis}{https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis}')))

# =============================================================================
# 1. POT FI PREVĂZUTE PREȚURILE?
# =============================================================================
D.section('Can Prices Be Predicted?', 'Previzibilitatea prețurilor')

D.frame(T('Can You Beat the Market?', 'Randamente peste cele ale pieței'), cols(items(
    (T('If past prices predicted future returns, a simple rule would earn money with no risk', 'Dacă prețurile trecute ar anticipa randamentele viitoare, o regulă simplă ar aduce bani fără risc'),
     [T('traders would use the rule; their trades would move prices until the pattern disappears',
        'investitorii ar folosi regula; tranzacțiile lor ar muta prețurile pînă cînd tiparul dispare')]),
    (T('The evidence on professional investors', 'Dovezile despre investitorii profesioniști'),
     [T('mutual funds, on average, did not beat a buy-and-hold portfolio after costs \\refJensen',
        'fondurile de investiții nu au obținut, în medie, randamente mai mari decît o strategie buy-and-hold, după deducerea costurilor \\refJensen'),
      T('few funds show skill large enough to cover their costs \\refFF', 'puține fonduri au o abilitate suficient de mare încît să își acopere costurile \\refFF')]),
    T('A survey of the arguments for and against: \\refMalkiel', 'O sinteză a argumentelor pro și contra: \\refMalkiel')),
    ph('nyse', T('The trading floor of the New York Stock Exchange, 1963', 'Ringul de tranzacționare al Bursei din New York, 1963'),
       h='0.36\\textheight'), wl='0.54', wr='0.42'))

D.frame(T('1900: Prices as a Random Walk', '1900: prețurile ca mers aleator'), cols(items(
    (T('Louis Bachelier, \\textit{Théorie de la spéculation} \\refBachelier', 'Louis Bachelier, \\textit{Théorie de la spéculation} \\refBachelier'),
     [T('the first mathematical model of prices: the change of a price is a sum of many small random shocks',
        'primul model matematic al prețurilor: variația unui preț este o sumă de multe șocuri mici aleatoare'),
      T('the expected gain of the speculator is zero', 'cîștigul așteptat al speculatorului este zero')]),
    (T('\\refKendall: 22 British price series behave like a ``wandering\'\' series', '\\refKendall: 22 de serii de prețuri britanice se comportă ca o serie „rătăcitoare”'),
     [T('the changes from one week to the next look like independent draws', 'variațiile de la o săptămînă la alta seamănă cu extrageri independente')]),
    T('\\refRoberts: simulated random walks show ``patterns\'\' that chartists see in real prices',
      '\\refRoberts: mersurile aleatoare simulate arată „tipare” pe care analiștii tehnici le văd în prețurile reale')),
    ph('bachelier', T('Louis Bachelier (1870--1946)', 'Louis Bachelier (1870--1946)'), h='0.42\\textheight'), wl='0.64', wr='0.32'))

chart(T('Which One Is the S\\&P 500?', 'Întrebare pentru sală: care traiectorie este S\\&P 500?'), 'sfm_ch7_rw_paths', 'SFM_ch7_random_walk', [
    T('Four random walks with i.i.d.\\ Normal steps (mean $@{rw.mu}\\%$, standard deviation $@{rw.sd}\\%$ per day, as the S\\&P 500) and the S\\&P 500 log price since 1990',
      'Patru mersuri aleatoare cu pași Normali i.i.d.\\ (media $@{rw.mu}\\%$, abaterea standard $@{rw.sd}\\%$ pe zi, ca la S\\&P 500) și prețul logaritmic S\\&P 500 din 1990'),
    T('\\textbf{Question for the room}: without the legend, could you tell which path is real?', '\\textbf{Întrebare pentru sală}: fără legendă, ați putea spune care traiectorie este cea reală?'),
    T('\\textbf{Answer}: hardly; booms and long falls appear in pure noise too (end values: $@{rw.smin}$ to $@{rw.smax}$; S\\&P 500: $@{rw.final}$)',
      '\\textbf{Răspuns}: greu; avînturile și căderile lungi apar și în zgomotul pur (valori finale: $@{rw.smin}$--$@{rw.smax}$; S\\&P 500: $@{rw.final}$)')],
    h='0.56\\textheight')

D.frame(T('1965: Properly Anticipated Prices Fluctuate Randomly', '1965: prețurile anticipate corect fluctuează aleator'), cols(items(
    (T('Paul Samuelson \\refSamuelson: if prices already contain all that traders expect, the next change is unpredictable',
       'Paul Samuelson \\refSamuelson: dacă prețurile conțin deja tot ce anticipează investitorii, următoarea variație este imprevizibilă'),
     [T('formally: the price is a \\textbf{martingale}, $E[P_{t+1} \\mid \\mathcal{F}_t] = P_t$ (Chapter 4)',
        'formal: prețul este un \\textbf{martingal}, $E[P_{t+1} \\mid \\mathcal{F}_t] = P_t$ (Capitolul 4)'),
      T('$\\mathcal{F}_t$: the information available at time $t$', '$\\mathcal{F}_t$: informația disponibilă la momentul $t$')]),
    T('Randomness of price changes is a sign of a market that works, not of chaos', 'Caracterul aleator al variațiilor de preț este semnul unei piețe care funcționează, nu al haosului'),
    T('\\refFamaC: daily returns of the 30 Dow Jones stocks show almost no autocorrelation', '\\refFamaC: randamentele zilnice ale celor 30 de acțiuni Dow Jones nu arată aproape deloc autocorelație')),
    ph('samuelson', T('Paul A. Samuelson (1915--2009), Nobel Prize 1970', 'Paul A. Samuelson (1915--2009), Premiul Nobel 1970'),
       h='0.34\\textheight'), wl='0.54', wr='0.42'))

chart(T('Does Yesterday Predict Today?', 'Randamentul de ieri și randamentul de azi'), 'sfm_ch7_scatter_lag', 'SFM_ch7_random_walk', [
    T('Each point: (return yesterday, return today); the line: least squares; its slope is the first autocorrelation $\\hat\\rho(1)$',
      'Fiecare punct: (randamentul de ieri, randamentul de azi); dreapta: cele mai mici pătrate; panta ei este prima autocorelație $\\hat\\rho(1)$'),
    T('S\\&P 500: $\\hat\\rho(1) = @{sc.sp500.rho}$ (a slight reversal); BET: $\\hat\\rho(1) = @{sc.bet.rho}$ (a slight continuation); are these numbers different from zero?',
      'S\\&P 500: $\\hat\\rho(1) = @{sc.sp500.rho}$ (o ușoară revenire); BET: $\\hat\\rho(1) = @{sc.bet.rho}$ (o ușoară continuare); sînt aceste valori diferite de zero?')],
    h='0.58\\textheight')

# =============================================================================
# 2. PIEȚE EFICIENTE
# =============================================================================
D.section('Efficient Markets', 'Piețe eficiente')

D.frame(T('What Is an Efficient Market?', 'Piața eficientă: definiția'), cols(items(
    (T('\\textbf{EMH} (efficient market hypothesis) \\refFamaA: prices ``fully reflect\'\' the available information $\\Omega_t$',
       '\\textbf{EMH} (efficient market hypothesis, ipoteza pieței eficiente) \\refFamaA: prețurile „reflectă pe deplin” informația disponibilă $\\Omega_t$'),
     [T('no trading rule based on $\\Omega_t$ earns more than the return that compensates its risk',
        'nicio regulă de tranzacționare care folosește $\\Omega_t$ nu cîștigă mai mult decît randamentul care îi compensează riscul'),
      T('prices move only on \\textbf{news}: information that was not in $\\Omega_t$', 'prețurile se mișcă doar la \\textbf{știri}: informație care nu era în $\\Omega_t$')]),
    (T('Efficiency is relative to an information set', 'Eficiența este relativă la o mulțime de informații'),
     [T('the smaller $\\Omega_t$, the weaker the claim and the easier the test', 'cu cît $\\Omega_t$ este mai mică, cu atît afirmația este mai slabă și testul mai ușor')]),
    T('Efficient does not mean ``returns are zero\'\': the expected return pays for risk', 'Eficient nu înseamnă „randamente zero”: randamentul așteptat remunerează riscul')),
    ph('fama', T('Eugene F. Fama, Nobel Prize 2013', 'Eugene F. Fama, Premiul Nobel 2013'), h='0.42\\textheight'), wl='0.60', wr='0.36'))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Three Forms of Efficiency', 'Trei forme ale eficienței'), table(
    TB + 'p{1.8cm}' + TB + 'p{3.2cm}' + TB + 'p{3.0cm}' + TB + 'p{2.9cm}',
    T('\\textbf{Form}', '\\textbf{Forma}') + ' & ' + T('\\textbf{Information set $\\Omega_t$}', '\\textbf{Mulțimea de informații $\\Omega_t$}') + ' & '
    + T('\\textbf{What cannot earn excess returns}', '\\textbf{Strategii fără randamente în exces}') + ' & ' + T('\\textbf{Typical test}', '\\textbf{Testul tipic}'),
    [T('Weak', 'Slabă') + ' & ' + T('past prices and returns', 'prețurile și randamentele trecute') + ' & ' + T('technical analysis, chart patterns', 'analiza tehnică, tiparele grafice') + ' & '
     + T('autocorrelation, runs, variance ratio', 'autocorelație, runs, variance ratio'),
     T('Semi-strong', 'Semi-tare') + ' & ' + T('all public information', 'toată informația publică') + ' & ' + T('fundamental analysis of public data', 'analiza fundamentală a datelor publice') + ' & '
     + T('event studies', 'studii de eveniment'),
     T('Strong', 'Tare') + ' & ' + T('all information, also private', 'toată informația, inclusiv cea privată') + ' & ' + T('insider information', 'informația privilegiată') + ' & '
     + T('returns of insiders and fund managers', 'randamentele insiderilor și ale administratorilor de fonduri')], size='footnotesize') + items(
    T('The forms are nested: strong $\\Rightarrow$ semi-strong $\\Rightarrow$ weak; a rejection of the weak form rejects all three',
      'Formele sînt incluse una în alta: tare $\\Rightarrow$ semi-tare $\\Rightarrow$ slabă; respingerea formei slabe le respinge pe toate trei'),
    T('This chapter tests mainly the weak form: past prices are free data for everyone \\refFamaA, \\refFamaB',
      'Acest capitol testează mai ales forma slabă: prețurile trecute sînt date gratuite pentru oricine \\refFamaA, \\refFamaB')))

D.frame(T('The Fair-Game Model', 'Modelul jocului echitabil'), items(
    (T('Let $r_{t+1}$ be the return of an asset and $E[r_{t+1} \\mid \\Omega_t]$ its \\textbf{equilibrium expected return} (the return that pays for its risk)',
       'Fie $r_{t+1}$ randamentul unui activ și $E[r_{t+1} \\mid \\Omega_t]$ \\textbf{randamentul așteptat de echilibru} (randamentul care îi remunerează riscul)'),
     [T('the \\textbf{excess return} (surprise): $z_{t+1} = r_{t+1} - E[r_{t+1} \\mid \\Omega_t]$', '\\textbf{randamentul în exces} (surpriza): $z_{t+1} = r_{t+1} - E[r_{t+1} \\mid \\Omega_t]$')]),
    (T('\\textbf{Fair game}: $E[z_{t+1} \\mid \\Omega_t] = 0$', '\\textbf{Joc echitabil}: $E[z_{t+1} \\mid \\Omega_t] = 0$'),
     [T('forecast errors are zero on average whatever we know at time $t$', 'erorile de prognoză sînt nule în medie, orice am ști la momentul $t$'),
      T('a rule that invests $w(\\Omega_t)$ earns $E[w(\\Omega_t)\\, z_{t+1}] = 0$ in excess of equilibrium', 'o regulă care investește $w(\\Omega_t)$ cîștigă $E[w(\\Omega_t)\\, z_{t+1}] = 0$ peste echilibru')]),
    T('With a constant expected return $\\mu$: $E[r_{t+1} \\mid \\Omega_t] = \\mu$; returns minus $\\mu$ are a \\textbf{martingale difference}',
      'Cu un randament așteptat constant $\\mu$: $E[r_{t+1} \\mid \\Omega_t] = \\mu$; randamentele minus $\\mu$ sînt o \\textbf{diferență de martingal}'),
    T('Nothing here says the variance is constant or the distribution is Normal', 'Definiția nu cere nici varianță constantă, nici distribuția Normală')))

D.frame(T('The Joint-Hypothesis Problem', 'Problema ipotezei comune'), items(
    (T('To test efficiency we must know the equilibrium expected return $E[r_{t+1} \\mid \\Omega_t]$', 'Pentru a testa eficiența trebuie să știm randamentul așteptat de echilibru $E[r_{t+1} \\mid \\Omega_t]$'),
     [T('this needs a model of equilibrium: constant mean, CAPM (capital asset pricing model), factor models',
        'pentru asta ne trebuie un model de echilibru: medie constantă, CAPM (capital asset pricing model, modelul de evaluare a activelor financiare), modele cu factori')]),
    (T('Every test of efficiency is a \\textbf{joint test} of efficiency and of the model \\refFamaB', 'Orice test al eficienței este un \\textbf{test comun} al eficienței și al modelului \\refFamaB'),
     [T('a rejection may mean an inefficient market, or a wrong model of expected returns', 'o respingere poate însemna o piață ineficientă sau un model greșit al randamentelor așteptate'),
      T('efficiency alone can never be rejected', 'eficiența, luată separat, nu poate fi respinsă niciodată')]),
    T('For daily returns the problem is small: daily expected returns are tiny (about $@{rw.mu}\\%$ for the S\\&P 500), so any reasonable model gives almost the same answer',
      'Pentru randamentele zilnice problema contează puțin: randamentele așteptate zilnice sînt foarte mici (circa $@{rw.mu}\\%$ pentru S\\&P 500), deci orice model rezonabil dă aproape același răspuns')))

D.frame(T('The Grossman--Stiglitz Paradox', 'Paradoxul Grossman--Stiglitz'), items(
    (T('Information is costly to collect and to analyse \\refGS', 'Informația costă: trebuie colectată și analizată \\refGS'),
     [T('if prices reflected all information perfectly, nobody would be paid for collecting it', 'dacă prețurile ar reflecta perfect toată informația, nimeni nu ar fi plătit pentru a o colecta'),
      T('then nobody would collect it, and prices could not reflect it', 'atunci nimeni nu ar mai colecta-o, iar prețurile nu ar mai putea s-o reflecte')]),
    (T('Conclusion: a market can only be \\textbf{nearly} efficient', 'Concluzie: o piață poate fi doar \\textbf{aproape} eficientă'),
     [T('small inefficiencies pay informed traders for their costs', 'micile ineficiențe acoperă costurile investitorilor informați'),
      T('the question becomes ``how efficient, where and when?\'\', not ``yes or no?\'\'', 'întrebarea devine „cît de eficientă, unde și cînd?”, nu „da sau nu?”')]),
    T('This view leads to the adaptive markets hypothesis of Section 8', 'Această perspectivă duce la ipoteza piețelor adaptive din secțiunea 8')))

D.frame(T('2013: One Nobel Prize, Two Views', '2013: un Premiu Nobel, două perspective'), cols(
    ph('fama', T('Eugene F. Fama: prices are hard to predict over days and weeks', 'Eugene F. Fama: prețurile sînt greu de anticipat pe zile și săptămîni'), h='0.38\\textheight'),
    ph('shiller', T('Robert J. Shiller: prices swing too much compared with fundamentals', 'Robert J. Shiller: prețurile oscilează prea mult față de valorile fundamentale'), h='0.38\\textheight'),
    wl='0.48', wr='0.48') + items(
    T('Both are right, at different horizons: short-run returns are almost unpredictable; long-run returns are partly predictable from valuation ratios',
      'Amîndoi au dreptate, pe orizonturi diferite: randamentele pe termen scurt sînt aproape imprevizibile; cele pe termen lung sînt parțial previzibile din indicatorii de evaluare')), 'footnotesize')

# =============================================================================
# 3. IPOTEZELE MERSULUI ALEATOR
# =============================================================================
D.section('Random Walk Hypotheses', 'Ipotezele mersului aleator')

D.frame(T('The Random Walk Model of Log Prices', 'Modelul mersului aleator pentru prețurile logaritmice'), items(
    (T('Log price $p_t = \\ln P_t$; \\textbf{random walk with drift} $\\mu$: $p_t = \\mu + p_{t-1} + \\varepsilon_t$',
       'Prețul logaritmic $p_t = \\ln P_t$; \\textbf{mers aleator cu tendință (drift)} $\\mu$: $p_t = \\mu + p_{t-1} + \\varepsilon_t$'),
     [T('the return $r_t = p_t - p_{t-1} = \\mu + \\varepsilon_t$: the increments $\\varepsilon_t$ have mean 0', 'randamentul $r_t = p_t - p_{t-1} = \\mu + \\varepsilon_t$: creșterile $\\varepsilon_t$ au media 0'),
      T('by recursion: $p_t = p_0 + \\mu t + \\sum_{s=1}^{t} \\varepsilon_s$: every shock is \\textbf{permanent}',
        'prin recurență: $p_t = p_0 + \\mu t + \\sum_{s=1}^{t} \\varepsilon_s$: fiecare șoc este \\textbf{permanent}')]),
    (T('If the $\\varepsilon_t$ are uncorrelated with variance $\\sigma^2$:', 'Dacă $\\varepsilon_t$ sînt necorelate, cu varianța $\\sigma^2$:'),
     [T('$E[p_t] = p_0 + \\mu t$ and $\\mathrm{Var}(p_t - p_0) = t\\sigma^2$: the uncertainty grows with the horizon',
        '$E[p_t] = p_0 + \\mu t$ și $\\mathrm{Var}(p_t - p_0) = t\\sigma^2$: incertitudinea crește cu orizontul'),
      T('the $q$-day return $r_t(q) = r_t + \\dots + r_{t-q+1}$ has $\\mathrm{Var}(r_t(q)) = q\\sigma^2$', 'randamentul pe $q$ zile $r_t(q) = r_t + \\dots + r_{t-q+1}$ are $\\mathrm{Var}(r_t(q)) = q\\sigma^2$')]),
    T('Log prices, not prices: $P_t$ cannot become negative, and log returns add over time (Chapter 1)',
      'Prețuri logaritmice, nu prețuri: $P_t$ nu poate deveni negativ, iar randamentele logaritmice se adună în timp (Capitolul 1)')))

D.frame(T('Three Random Walk Hypotheses', 'Trei ipoteze de mers aleator'), items(
    (T('\\textbf{RW1} (i.i.d.\\ increments): $\\varepsilon_t$ independent and identically distributed, $\\varepsilon_t \\sim \\mathrm{IID}(0, \\sigma^2)$ \\refCLM',
       '\\textbf{RW1} (creșteri i.i.d.): $\\varepsilon_t$ independente și identic distribuite, $\\varepsilon_t \\sim \\mathrm{IID}(0, \\sigma^2)$ \\refCLM'),
     [T('the strongest: no function of the past predicts any function of the future', 'cea mai puternică: nicio funcție a trecutului nu anticipează vreo funcție a viitorului')]),
    (T('\\textbf{RW2} (independent increments): $\\varepsilon_t$ independent, but not identically distributed', '\\textbf{RW2} (creșteri independente): $\\varepsilon_t$ independente, dar nu identic distribuite'),
     [T('allows the variance to change over time in a way that is not predictable from the past (regimes)', 'permite varianței să se schimbe în timp într-un mod care nu poate fi anticipat din trecut (regimuri)')]),
    (T('\\textbf{RW3} (uncorrelated increments): $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ for all $k \\ne 0$',
       '\\textbf{RW3} (creșteri necorelate): $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ pentru orice $k \\ne 0$'),
     [T('the weakest: allows $\\mathrm{Cov}(\\varepsilon_t^2, \\varepsilon_{t-k}^2) \\ne 0$, that is, volatility clustering', 'cea mai slabă: permite $\\mathrm{Cov}(\\varepsilon_t^2, \\varepsilon_{t-k}^2) \\ne 0$, adică volatility clustering')]),
    T('Nested: RW1 $\\Rightarrow$ RW2 $\\Rightarrow$ RW3; tests of RW3 use only autocorrelations of returns', 'Incluse una în alta: RW1 $\\Rightarrow$ RW2 $\\Rightarrow$ RW3; testele RW3 folosesc doar autocorelațiile randamentelor')))

D.frame(T('What Each Hypothesis Allows', 'Comparația ipotezelor de mers aleator'), table(
    'lcccc', T('\\textbf{Property of the returns}', '\\textbf{Proprietatea randamentelor}') + ' & RW1 & RW2 & RW3 & ' + T('\\textbf{Martingale}', '\\textbf{Martingal}'),
    [T('$\\mathrm{Corr}(r_t, r_{t-k}) = 0$', '$\\mathrm{Corr}(r_t, r_{t-k}) = 0$') + ' & ' + ' & '.join([T('yes', 'da')] * 4),
     T('$E[r_t \\mid \\text{past}] = \\mu$ (unpredictable mean)', '$E[r_t \\mid \\text{trecut}] = \\mu$ (media imprevizibilă)') + ' & ' + T('yes', 'da') + ' & ' + T('yes', 'da') + ' & ' + T('not always', 'nu mereu') + ' & ' + T('yes', 'da'),
     T('constant variance', 'varianță constantă') + ' & ' + T('yes', 'da') + ' & ' + T('no', 'nu') + ' & ' + T('no', 'nu') + ' & ' + T('no', 'nu'),
     T('volatility clustering allowed', 'admite volatility clustering') + ' & ' + T('no', 'nu') + ' & ' + T('no', 'nu') + ' & ' + T('yes', 'da') + ' & ' + T('yes', 'da'),
     T('GARCH returns (Chapter 9) satisfy it', 'randamentele GARCH (Capitolul 9) o satisfac') + ' & ' + T('no', 'nu') + ' & ' + T('no', 'nu') + ' & ' + T('yes', 'da') + ' & ' + T('yes', 'da')],
    size='footnotesize') + items(
    T('\\textbf{GARCH} (generalised autoregressive conditional heteroskedasticity): the variance of tomorrow depends on today\'s shocks; the mean stays unpredictable',
      '\\textbf{GARCH} (generalised autoregressive conditional heteroskedasticity, heteroscedasticitate condiționată autoregresivă generalizată): varianța de mîine depinde de șocurile de azi; media rămîne imprevizibilă'),
    T('Weak-form efficiency needs only a martingale (or RW3), not RW1', 'Eficiența în formă slabă cere doar un martingal (sau RW3), nu RW1')))

D.frame(T('Martingale and Random Walk', 'Martingal și mers aleator'), items(
    (T('$\\{p_t - \\mu t\\}$ is a \\textbf{martingale} if $E[p_{t+1} - \\mu(t+1) \\mid \\mathcal{F}_t] = p_t - \\mu t$ \\refSamuelson',
       '$\\{p_t - \\mu t\\}$ este un \\textbf{martingal} dacă $E[p_{t+1} - \\mu(t+1) \\mid \\mathcal{F}_t] = p_t - \\mu t$ \\refSamuelson'),
     [T('equivalently, $E[r_{t+1} - \\mu \\mid \\mathcal{F}_t] = 0$: the returns minus $\\mu$ are a martingale difference', 'echivalent, $E[r_{t+1} - \\mu \\mid \\mathcal{F}_t] = 0$: randamentele minus $\\mu$ sînt o diferență de martingal'),
      T('a martingale difference with finite variance is uncorrelated: martingale $\\Rightarrow$ RW3', 'o diferență de martingal cu varianță finită este necorelată: martingal $\\Rightarrow$ RW3')]),
    T('RW1 and RW2 imply the martingale; the converse fails: a martingale may have predictable variance', 'RW1 și RW2 implică martingalul; reciproca nu este adevărată: un martingal poate avea varianța previzibilă'),
    (T('\\textbf{Question for the room}: does volatility clustering contradict weak-form efficiency?', '\\textbf{Întrebare pentru sală}: este volatility clustering în contradicție cu eficiența în formă slabă?'),
     [T('\\textbf{Answer}: no; it rejects RW1 and RW2, but a predictable variance gives no predictable return; the martingale and RW3 survive',
        '\\textbf{Răspuns}: nu; respinge RW1 și RW2, dar o varianță previzibilă nu face randamentul previzibil; martingalul și RW3 rămîn valabile')])))

D.frame(T('Worked Example: How Risk Grows with the Horizon', 'Exemplu lucrat: cum crește riscul cu orizontul'), items(
    (T('S\\&P 500 since 1990: daily standard deviation $\\hat\\sigma = @{rw.sd}\\%$', 'S\\&P 500 din 1990: abaterea standard zilnică $\\hat\\sigma = @{rw.sd}\\%$'),
     [T('under a random walk, $\\mathrm{sd}(r_t(q)) = \\sigma\\sqrt{q}$', 'în ipoteza de mers aleator, $\\mathrm{sd}(r_t(q)) = \\sigma\\sqrt{q}$')]),
    (T('Step by step', 'Pas cu pas'),
     [T('one week, $q = 5$: $@{rw.sd} \\times \\sqrt{5} = @{rw.sd} \\times @{ex.sq5} = @{ex.sd5}\\%$', 'o săptămînă, $q = 5$: $@{rw.sd} \\times \\sqrt{5} = @{rw.sd} \\times @{ex.sq5} = @{ex.sd5}\\%$'),
      T('one month, $q = 20$: $@{rw.sd} \\times @{ex.sq20} = @{ex.sd20}\\%$', 'o lună, $q = 20$: $@{rw.sd} \\times @{ex.sq20} = @{ex.sd20}\\%$'),
      T('one year, $q = 250$: $@{rw.sd} \\times @{ex.sq250} = @{ex.sd250}\\%$', 'un an, $q = 250$: $@{rw.sd} \\times @{ex.sq250} = @{ex.sd250}\\%$')]),
    T('This is the ``square-root-of-time rule\'\' used to scale VaR and volatility', 'Aceasta este „regula rădăcinii pătrate a timpului”, folosită pentru a scala VaR și volatilitatea'),
    T('If returns are autocorrelated, the rule fails: the variance-ratio test of Section 7 measures by how much',
      'Dacă randamentele sînt autocorelate, regula nu mai funcționează: testul variance ratio din secțiunea 7 măsoară cu cît')))

# =============================================================================
# 4. ZGOMOT ALB ȘI ACF
# =============================================================================
D.section('White Noise and the Autocorrelation Function', 'Zgomotul alb și funcția de autocorelație')

D.frame(T('White Noise and Autocorrelation', 'Zgomot alb și autocorelație'), items(
    (T('\\textbf{Autocovariance} at lag $k$: $\\gamma(k) = \\mathrm{Cov}(x_t, x_{t-k})$; \\textbf{autocorrelation}: $\\rho(k) = \\gamma(k)/\\gamma(0)$',
       '\\textbf{Autocovarianța} la lagul $k$: $\\gamma(k) = \\mathrm{Cov}(x_t, x_{t-k})$; \\textbf{autocorelația}: $\\rho(k) = \\gamma(k)/\\gamma(0)$'),
     [T('the \\textbf{ACF} (autocorrelation function): $k \\mapsto \\rho(k)$, $k = 1, 2, \\dots$', '\\textbf{ACF} (autocorrelation function, funcția de autocorelație): $k \\mapsto \\rho(k)$, $k = 1, 2, \\dots$')]),
    (T('\\textbf{White noise}: mean 0, constant variance $\\sigma^2$, $\\rho(k) = 0$ for every $k \\ne 0$', '\\textbf{Zgomot alb}: media 0, varianța constantă $\\sigma^2$, $\\rho(k) = 0$ pentru orice $k \\ne 0$'),
     [T('i.i.d.\\ noise is white noise; Gaussian white noise is i.i.d.\\ $N(0, \\sigma^2)$', 'zgomotul i.i.d.\\ este zgomot alb; zgomotul alb gaussian este i.i.d.\\ $N(0, \\sigma^2)$'),
      T('white noise need not be independent: GARCH returns are white noise but not i.i.d.', 'zgomotul alb nu trebuie să fie independent: randamentele GARCH sînt zgomot alb, dar nu i.i.d.')]),
    (T('\\textbf{Sample ACF}: $\\hat\\rho(k) = \\sum_{t=k+1}^{T}(x_t - \\bar x)(x_{t-k} - \\bar x)\\big/\\sum_{t=1}^{T}(x_t - \\bar x)^2$',
       '\\textbf{ACF de selecție}: $\\hat\\rho(k) = \\sum_{t=k+1}^{T}(x_t - \\bar x)(x_{t-k} - \\bar x)\\big/\\sum_{t=1}^{T}(x_t - \\bar x)^2$'),
     [T('for i.i.d.\\ data, $\\hat\\rho(k) \\approx N(0, 1/T)$: 95\\% band $\\pm 1.96/\\sqrt{T}$', 'pentru date i.i.d., $\\hat\\rho(k) \\approx N(0, 1/T)$: banda de 95\\% $\\pm 1{,}96/\\sqrt{T}$')])))

chart(T('Gaussian White Noise and Its Sample ACF', 'Zgomotul alb gaussian și ACF-ul lui de selecție'), 'sfm_ch7_white_noise', 'SFM_ch7_white_noise_acf', [
    T('Left: 1\\,000 draws from $N(0, 1)$ (as in SFEtimewn); right: $\\hat\\rho(1), \\dots, \\hat\\rho(30)$ with the band $\\pm 1.96/\\sqrt{1000} = \\pm @{wn.band}$',
      'Stînga: 1\\,000 de extrageri din $N(0, 1)$ (ca în SFEtimewn); dreapta: $\\hat\\rho(1), \\dots, \\hat\\rho(30)$ cu banda $\\pm 1{,}96/\\sqrt{1000} = \\pm @{wn.band}$'),
    T('Here @{wn.out} of 30 autocorrelations fall outside the band; with 30 lags we expect about 1.5 by chance alone',
      'Aici @{wn.out} din 30 de autocorelații ies din bandă; cu 30 de laguri ne așteptăm la circa 1,5 doar din întîmplare')],
    h='0.56\\textheight')

D.frame(T('The ACF of AR and MA Processes', 'ACF-ul proceselor AR și MA'), items(
    (T('\\textbf{AR(1)} (autoregressive): $x_t = \\alpha x_{t-1} + \\varepsilon_t$, $|\\alpha| < 1$: $\\rho(k) = \\alpha^k$',
       '\\textbf{AR(1)} (autoregresiv): $x_t = \\alpha x_{t-1} + \\varepsilon_t$, $|\\alpha| < 1$: $\\rho(k) = \\alpha^k$'),
     [T('geometric decay; alternating signs if $\\alpha < 0$; example: $\\alpha = 0.9 \\Rightarrow \\rho(3) = 0.9^3 = @{ex.ar3}$',
        'scădere geometrică; semne alternante dacă $\\alpha < 0$; exemplu: $\\alpha = 0.9 \\Rightarrow \\rho(3) = 0.9^3 = @{ex.ar3}$')]),
    (T('\\textbf{AR(2)}: $x_t = \\alpha_1 x_{t-1} + \\alpha_2 x_{t-2} + \\varepsilon_t$: $\\rho(1) = \\alpha_1/(1 - \\alpha_2)$, $\\rho(k) = \\alpha_1\\rho(k-1) + \\alpha_2\\rho(k-2)$',
       '\\textbf{AR(2)}: $x_t = \\alpha_1 x_{t-1} + \\alpha_2 x_{t-2} + \\varepsilon_t$: $\\rho(1) = \\alpha_1/(1 - \\alpha_2)$, $\\rho(k) = \\alpha_1\\rho(k-1) + \\alpha_2\\rho(k-2)$'),
     [T('$\\alpha_1 = 0.5$, $\\alpha_2 = 0.4$: $\\rho(1) = 0.5/0.6 = @{ex.ar2r1}$, $\\rho(2) = 0.5 \\times @{ex.ar2r1} + 0.4 = @{ex.ar2r2}$',
        '$\\alpha_1 = 0.5$, $\\alpha_2 = 0.4$: $\\rho(1) = 0.5/0.6 = @{ex.ar2r1}$, $\\rho(2) = 0.5 \\times @{ex.ar2r1} + 0.4 = @{ex.ar2r2}$')]),
    (T('\\textbf{MA(1)} (moving average): $x_t = \\varepsilon_t + \\beta\\varepsilon_{t-1}$: $\\rho(1) = \\beta/(1 + \\beta^2)$, $\\rho(k) = 0$ for $k \\ge 2$',
       '\\textbf{MA(1)} (moving average, medie mobilă): $x_t = \\varepsilon_t + \\beta\\varepsilon_{t-1}$: $\\rho(1) = \\beta/(1 + \\beta^2)$, $\\rho(k) = 0$ pentru $k \\ge 2$'),
     [T('$\\beta = 0.5$: $\\rho(1) = 0.5/1.25 = @{ex.ma1}$; an MA($q$) has an ACF that cuts off after lag $q$', '$\\beta = 0.5$: $\\rho(1) = 0.5/1.25 = @{ex.ma1}$; un MA($q$) are un ACF care se anulează după lagul $q$')]),
    T('The shape of the ACF tells the type of dependence: slow decay (AR), cut-off (MA), nothing (white noise)',
      'Forma ACF-ului arată tipul de dependență: scădere lentă (AR), anulare bruscă (MA), nimic (zgomot alb)')))

chart(T('Theoretical and Sample ACF of Six Processes', 'ACF-ul teoretic și de selecție pentru șase procese'), 'sfm_ch7_acf_processes', 'SFM_ch7_white_noise_acf', [
    T('Bars: theoretical $\\rho(k)$; points: sample ACF of one simulated path of 1\\,000 values (SFEacfar1, SFEacfar2, SFEacfma1, SFEacfma2)',
      'Bare: $\\rho(k)$ teoretic; puncte: ACF-ul de selecție al unei traiectorii simulate de 1\\,000 de valori (SFEacfar1, SFEacfar2, SFEacfma1, SFEacfma2)'),
    T('A random walk is the limit $\\alpha \\to 1$ of the AR(1): its ACF would not decay at all', 'Un mers aleator este limita $\\alpha \\to 1$ a AR(1): ACF-ul lui nu ar scădea deloc')],
    h='0.60\\textheight')

chart(T('The ACF of Daily Returns', 'ACF-ul randamentelor zilnice'), 'sfm_ch7_acf_returns', 'SFM_ch7_autocorrelation_tests', [
    T('Bars: $\\hat\\rho(1), \\dots, \\hat\\rho(20)$; dashed: the i.i.d.\\ band $\\pm 1.96/\\sqrt{T}$; shaded: the robust band (next slide)',
      'Bare: $\\hat\\rho(1), \\dots, \\hat\\rho(20)$; linia întreruptă: banda i.i.d.\\ $\\pm 1{,}96/\\sqrt{T}$; zona umbrită: banda robustă (slide-ul următor)')],
    h='0.62\\textheight')

D.frame(T('Reading the ACF of Returns', 'Interpretarea ACF-ului randamentelor'), items(
    (T('First autocorrelations: S\\&P 500 $@{acf.sp500.rho1}$, DAX $@{acf.dax.rho1}$, BET $@{acf.bet.rho1}$, Bitcoin $@{acf.btc.rho1}$',
       'Primele autocorelații: S\\&P 500 $@{acf.sp500.rho1}$, DAX $@{acf.dax.rho1}$, BET $@{acf.bet.rho1}$, Bitcoin $@{acf.btc.rho1}$'),
     [T('only the S\\&P 500 and the BET stand out at lag 1, with opposite signs', 'doar S\\&P 500 și BET ies în evidență la lagul 1, cu semne opuse')]),
    (T('The \\textbf{robust} band: $\\pm 1.96\\sqrt{\\hat\\delta(k)/T}$, $\\hat\\delta(k) = T\\sum_t e_t^2 e_{t-k}^2/(\\sum_t e_t^2)^2$, $e_t = r_t - \\bar r$ \\refLM',
       'Banda \\textbf{robustă}: $\\pm 1{,}96\\sqrt{\\hat\\delta(k)/T}$, $\\hat\\delta(k) = T\\sum_t e_t^2 e_{t-k}^2/(\\sum_t e_t^2)^2$, $e_t = r_t - \\bar r$ \\refLM'),
     [T('i.i.d.\\ returns: $\\hat\\delta(k) \\approx 1$; volatility clustering: large returns follow large returns, so $\\hat\\delta(k) > 1$',
        'randamente i.i.d.: $\\hat\\delta(k) \\approx 1$; volatility clustering: randamentele mari în modul urmează după randamente mari în modul, deci $\\hat\\delta(k) > 1$'),
      T('S\\&P 500 at lag 1: $\\pm @{acf.sp500.brob}$ instead of $\\pm @{acf.sp500.biid}$', 'S\\&P 500 la lagul 1: $\\pm @{acf.sp500.brob}$ în loc de $\\pm @{acf.sp500.biid}$')]),
    T('S\\&P 500, lags outside the band out of 20: @{acf.sp500.oiid} with the i.i.d.\\ band, @{acf.sp500.orob} with the robust band',
      'S\\&P 500, numărul de laguri (din 20) aflate în afara benzii: @{acf.sp500.oiid} cu banda i.i.d., @{acf.sp500.orob} cu banda robustă'),
    T('The i.i.d.\\ band tests RW1; the robust band tests RW3, the hypothesis that matters for efficiency',
      'Banda i.i.d.\\ testează RW1; banda robustă testează RW3, ipoteza care contează pentru eficiență')) + ql('SFM_ch7_autocorrelation_tests'))

chart(T('Returns Are Almost Uncorrelated, Their Size Is Not', 'Randamentele sînt aproape necorelate, mărimea lor nu'), 'sfm_ch7_acf_abs', 'SFM_ch7_autocorrelation_tests', [
    T('ACF of $|r_t|$: S\\&P 500 $@{aa.sp500.abs1}$ at lag 1, $@{aa.sp500.abs20}$ at lag 20, $@{aa.sp500.abs100}$ at lag 100; the returns themselves: dotted lines near 0',
      'ACF-ul lui $|r_t|$: S\\&P 500 $@{aa.sp500.abs1}$ la lagul 1, $@{aa.sp500.abs20}$ la lagul 20, $@{aa.sp500.abs100}$ la lagul 100; randamentele însele: linii punctate în jurul lui 0'),
    T('Large moves follow large moves: RW1 and RW2 are rejected by eye; RW3 and the martingale are not touched by this chart',
      'Mișcările mari urmează după mișcări mari: RW1 și RW2 sînt respinse la prima vedere; graficul nu spune nimic despre RW3 și martingal')],
    h='0.56\\textheight')

# =============================================================================
# 5. TESTE DE AUTOCORELAȚIE
# =============================================================================
D.section('Testing for Autocorrelation', 'Testarea autocorelației')

D.frame(T('Portmanteau Tests: Box--Pierce and Ljung--Box', 'Teste portmanteau: Box--Pierce și Ljung--Box'), items(
    (T('$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; $H_1$: at least one $\\rho(k) \\ne 0$; one test for $m$ lags together', '$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; $H_1$: cel puțin un $\\rho(k) \\ne 0$; un singur test pentru $m$ laguri împreună'),
     [T('``portmanteau\'\': a suitcase that carries many autocorrelations at once', '„portmanteau”: un geamantan în care încap multe autocorelații deodată')]),
    (T('Box--Pierce \\refBP: $Q_{BP}(m) = T\\sum_{k=1}^{m} \\hat\\rho(k)^2$', 'Box--Pierce \\refBP: $Q_{BP}(m) = T\\sum_{k=1}^{m} \\hat\\rho(k)^2$'),
     [T('$T$: the number of returns; $m$: the number of lags tested; squares, so positive and negative autocorrelations both count', '$T$: numărul de randamente; $m$: numărul de laguri testate; pătratele fac ca autocorelațiile pozitive și negative să conteze la fel')]),
    (T('\\textbf{LB} (Ljung--Box) \\refLjung: $Q_{LB}(m) = T(T+2)\\sum_{k=1}^{m} \\dfrac{\\hat\\rho(k)^2}{T - k}$, closer to $\\chi^2$ in small samples',
       '\\textbf{LB} (Ljung--Box) \\refLjung: $Q_{LB}(m) = T(T+2)\\sum_{k=1}^{m} \\dfrac{\\hat\\rho(k)^2}{T - k}$, mai aproape de $\\chi^2$ în eșantioane mici'),
     [T('under i.i.d.\\ returns both $\\sim \\chi^2(m)$; reject $H_0$ if $Q > \\chi^2_{0.95}(m)$; $\\chi^2_{0.95}(10) = @{lb.crit}$',
        'pentru randamente i.i.d.\\ ambele $\\sim \\chi^2(m)$; respingem $H_0$ dacă $Q > \\chi^2_{0.95}(m)$; $\\chi^2_{0.95}(10) = @{lb.crit}$')]),
    T('Choice of $m$: here $m = 10$ (two trading weeks); a larger $m$ dilutes a short-lived effect', 'Alegerea lui $m$: aici $m = 10$ (două săptămîni de tranzacționare); un $m$ mai mare diluează un efect de scurtă durată')))

D.frame(T('Worked Example: Ljung--Box Step by Step', 'Exemplu lucrat: Ljung--Box pas cu pas'), items(
    (T('$T = 1000$ daily returns, $\\hat\\rho(1) = 0.06$, $\\hat\\rho(2) = -0.04$, $\\hat\\rho(3) = 0.03$; test with $m = 3$',
       '$T = 1000$ de randamente zilnice, $\\hat\\rho(1) = 0.06$, $\\hat\\rho(2) = -0.04$, $\\hat\\rho(3) = 0.03$; testăm cu $m = 3$'),
     [T('$T\\hat\\rho(k)^2$: $@{ex.r2.1}$, $@{ex.r2.2}$, $@{ex.r2.3}$; their sum: $Q_{BP}(3) = @{ex.bp}$', '$T\\hat\\rho(k)^2$: $@{ex.r2.1}$; $@{ex.r2.2}$; $@{ex.r2.3}$; suma lor: $Q_{BP}(3) = @{ex.bp}$'),
      T('$Q_{LB}(3) = 1000 \\times 1002 \\times (0.06^2/999 + 0.04^2/998 + 0.03^2/997) = @{ex.lb}$', '$Q_{LB}(3) = 1000 \\times 1002 \\times (0.06^2/999 + 0.04^2/998 + 0.03^2/997) = @{ex.lb}$')]),
    (T('Decision: $\\chi^2_{0.95}(3) = @{ex.c3} > @{ex.lb}$: do not reject $H_0$ at 5\\%', 'Decizia: $\\chi^2_{0.95}(3) = @{ex.c3} > @{ex.lb}$: nu respingem $H_0$ la 5\\%'),
     [T('note: $\\hat\\rho(1) = 0.06$ lies just inside the band $\\pm 1.96/\\sqrt{1000} = \\pm 0.062$; the joint test does not reject either',
        'observație: $\\hat\\rho(1) = 0.06$ se află chiar în interiorul benzii $\\pm 1{,}96/\\sqrt{1000} = \\pm 0{,}062$; nici testul comun nu respinge')]),
    T('The same numbers with $T = 5000$: $Q$ five times larger, a clear rejection: small autocorrelations become significant in long samples',
      'Cu aceleași autocorelații și $T = 5000$, $Q$ este de cinci ori mai mare și respingerea este clară: în eșantioanele lungi, autocorelațiile mici devin semnificative')))

D.frame(T('A Portmanteau Test Robust to Volatility Clustering', 'Un test portmanteau robust la volatility clustering'), items(
    (T('The $\\chi^2$ law of $Q_{LB}$ needs $\\mathrm{Var}(\\hat\\rho(k)) = 1/T$: true for i.i.d.\\ returns, false with volatility clustering',
       'Distribuția $\\chi^2$ a lui $Q_{LB}$ cere $\\mathrm{Var}(\\hat\\rho(k)) = 1/T$: condiție adevărată pentru randamente i.i.d., falsă în prezența volatility clustering'),
     [T('then $Q_{LB}$ rejects a true martingale far too often', 'atunci $Q_{LB}$ respinge un martingal adevărat mult prea des')]),
    (T('\\textbf{Robust} statistic: divide each $\\hat\\rho(k)^2$ by its own variance $\\hat\\delta(k)/T$ \\refEL', 'Statistica \\textbf{robustă}: împărțim fiecare $\\hat\\rho(k)^2$ la propria varianță $\\hat\\delta(k)/T$ \\refEL'),
     [T('$\\tilde Q(m) = \\sum_{k=1}^{m} \\dfrac{T\\hat\\rho(k)^2}{\\hat\\delta(k)} \\sim \\chi^2(m)$ for a martingale difference with finite fourth moment',
        '$\\tilde Q(m) = \\sum_{k=1}^{m} \\dfrac{T\\hat\\rho(k)^2}{\\hat\\delta(k)} \\sim \\chi^2(m)$ pentru o diferență de martingal cu momentul de ordin patru finit'),
      T('with i.i.d.\\ returns $\\hat\\delta(k) \\approx 1$ and $\\tilde Q \\approx Q_{BP}$', 'cu randamente i.i.d., $\\hat\\delta(k) \\approx 1$ și $\\tilde Q \\approx Q_{BP}$')]),
    T('Rule: test RW1 with $Q_{LB}$; test the martingale (weak-form efficiency) with $\\tilde Q$', 'Regula: testăm RW1 cu $Q_{LB}$; testăm martingalul (eficiența în formă slabă) cu $\\tilde Q$')))


def trow(k):
    return (f'{SHORT[k]} & @{{t.{k}.n}} & ${{@{{t.{k}.rho1}}}}$ & $@{{t.{k}.lb}}$ & @{{t.{k}.p_lb}} & $@{{t.{k}.rob}}$ & @{{t.{k}.p_rob}} & '
            f'$@{{t.{k}.lbsq}}$ & ${{@{{t.{k}.rz}}}}$ & @{{t.{k}.rp}}')


D.frame(T('Autocorrelation Tests on Real Data', 'Teste de autocorelație pe date reale'), table(
    'lrrrrrrrrr', T('& $T$ & $\\hat\\rho(1)$ & $Q_{LB}(10)$ & p & $\\tilde Q(10)$ & p & $Q_{LB}(10)$, $r^2$ & runs $z$ & p',
                    '& $T$ & $\\hat\\rho(1)$ & $Q_{LB}(10)$ & p & $\\tilde Q(10)$ & p & $Q_{LB}(10)$, $r^2$ & runs $z$ & p'),
    [trow(k) for k in ASSETS + STOCKS], size='scriptsize') + items(
    T('Classic LB: all eight series reject at 5\\%; robust $\\tilde Q$: only the S\\&P 500 (p-value: @{t.sp500.p_rob}) and the BET (p-value: @{t.bet.p_rob})',
      'LB clasic: toate cele opt serii resping la 5\\%; $\\tilde Q$ robust: doar S\\&P 500 (p-value: @{t.sp500.p_rob}) și BET (p-value: @{t.bet.p_rob})'),
    T('Squared returns: huge $Q_{LB}$ everywhere: RW1 and RW2 are rejected for every series (volatility clustering)', 'Randamentele la pătrat: $Q_{LB}$ uriaș peste tot: RW1 și RW2 sînt respinse pentru fiecare serie (volatility clustering)'),
    T('Start of the series: S\\&P 500 and DAX 1990; BET 1997; Bitcoin 2014; BVB stocks 2010 (TLV without 30--31 May 2016, a data error, Chapter 2)',
      'Începutul seriilor: S\\&P 500 și DAX 1990; BET 1997; Bitcoin 2014; acțiuni BVB 2010 (TLV fără 30--31 mai 2016, o eroare de date, Capitolul 2)')) + ql('SFM_ch7_autocorrelation_tests'), 'footnotesize')

D.frame(T('The Runs Test', 'Testul runs'), items(
    (T('A \\textbf{run}: a sequence of returns with the same sign; $+ + - + + + - -$ has 4 runs \\refWW', 'Un \\textbf{run} (secvență): un șir de randamente cu același semn; $+ + - + + + - -$ are 4 secvențe \\refWW'),
     [T('$R$: the observed number of runs', '$R$: numărul observat de secvențe'),
      T('the test uses only the signs: it is \\textbf{nonparametric}, insensitive to heavy tails and to volatility', 'testul folosește doar semnele: este \\textbf{neparametric}, insensibil la cozile groase și la volatilitate')]),
    (T('With $n_1$ positive and $n_2$ negative returns, $n = n_1 + n_2$, under independence:', 'Cu $n_1$ randamente pozitive și $n_2$ negative, $n = n_1 + n_2$, în ipoteza de independență:'),
     [T('$E[R] = \\dfrac{2n_1n_2}{n} + 1$, \\quad $\\mathrm{Var}(R) = \\dfrac{2n_1n_2(2n_1n_2 - n)}{n^2(n-1)}$, \\quad $z = \\dfrac{R - E[R]}{\\sqrt{\\mathrm{Var}(R)}} \\approx N(0, 1)$',
        '$E[R] = \\dfrac{2n_1n_2}{n} + 1$, \\quad $\\mathrm{Var}(R) = \\dfrac{2n_1n_2(2n_1n_2 - n)}{n^2(n-1)}$, \\quad $z = \\dfrac{R - E[R]}{\\sqrt{\\mathrm{Var}(R)}} \\approx N(0, 1)$')]),
    (T('Too few runs ($z < 0$): signs persist, positive dependence; too many runs ($z > 0$): signs alternate',
       'Prea puține secvențe ($z < 0$): semnele persistă, dependență pozitivă; prea multe secvențe ($z > 0$): semnele alternează'),
     [T('BET: @{t.bet.runs} runs against @{t.bet.rmean} expected, $z = @{t.bet.rz}$; S\\&P 500: @{t.sp500.runs} against @{t.sp500.rmean}, $z = @{t.sp500.rz}$',
        'BET: @{t.bet.runs} secvențe față de @{t.bet.rmean} așteptate, $z = @{t.bet.rz}$; S\\&P 500: @{t.sp500.runs} față de @{t.sp500.rmean}, $z = @{t.sp500.rz}$'),
      T('the same directions as $\\hat\\rho(1)$: continuation in the BET, reversal in the S\\&P 500', 'aceleași direcții ca $\\hat\\rho(1)$: continuare la BET, revenire la S\\&P 500')])) + ql('SFM_ch7_autocorrelation_tests'))

# =============================================================================
# 6. RĂDĂCINI UNITARE
# =============================================================================
D.section('Unit Roots: Prices against Returns', 'Rădăcini unitare: prețuri față de randamente')

D.frame(T('Stationary and Integrated Series', 'Serii staționare și serii integrate'), items(
    (T('\\textbf{(Weakly) stationary} series: constant mean, constant variance, $\\mathrm{Cov}(x_t, x_{t-k})$ depends only on $k$',
       'Serie \\textbf{(slab) staționară}: medie constantă, varianță constantă, $\\mathrm{Cov}(x_t, x_{t-k})$ depinde doar de $k$'),
     [T('shocks fade away; the series returns to its mean; notation $I(0)$', 'șocurile se sting; seria revine la media ei; notația $I(0)$')]),
    (T('\\textbf{Integrated of order 1}, $I(1)$: not stationary, but its first difference $\\Delta x_t = x_t - x_{t-1}$ is',
       '\\textbf{Integrată de ordinul 1}, $I(1)$: nu este staționară, dar prima ei diferență $\\Delta x_t = x_t - x_{t-1}$ este'),
     [T('the random walk is $I(1)$: $p_t$ wanders, $\\Delta p_t = r_t$ is stationary', 'mersul aleator este $I(1)$: $p_t$ nu revine la o medie, iar $\\Delta p_t = r_t$ este staționar')]),
    (T('\\textbf{Unit root}: in $p_t = \\phi\\,p_{t-1} + \\varepsilon_t$ the case $\\phi = 1$; $|\\phi| < 1$ gives a stationary AR(1)',
       '\\textbf{Rădăcină unitară} (unit root): în $p_t = \\phi\\,p_{t-1} + \\varepsilon_t$, cazul $\\phi = 1$; $|\\phi| < 1$ dă un AR(1) staționar'),
     [T('the name: the root of $1 - \\phi z = 0$ is $z = 1/\\phi = 1$', 'denumirea: rădăcina ecuației $1 - \\phi z = 0$ este $z = 1/\\phi = 1$')]),
    T('Question of this section: are log prices $I(1)$ and returns $I(0)$?', 'Întrebarea acestei secțiuni: sînt prețurile logaritmice $I(1)$ și randamentele $I(0)$?')))

chart(T('Why It Matters: Spurious Regression', 'Importanța practică: regresia falsă'), 'sfm_ch7_spurious', 'SFM_ch7_unit_roots', [
    (T('Regress one random walk on another, independent one ($T = 500$, 2\\,000 repetitions) \\refGN', 'Estimăm regresia unui mers aleator pe altul, independent ($T = 500$, 2\\,000 de repetări) \\refGN'),
     [T('$|t| > 1.96$ in @{sp.lev}\\% of cases, median $R^2 = @{sp.r2}$: a ``relation\'\' that does not exist', '$|t| > 1{,}96$ în @{sp.lev}\\% din cazuri, $R^2$ median $= @{sp.r2}$: o „relație” care nu există')]),
    (T('On the increments: $|t| > 1.96$ in @{sp.dif}\\% of cases, as it should be', 'Pe diferențe: $|t| > 1{,}96$ în @{sp.dif}\\% din cazuri, cum este de așteptat'),
     [T('lesson: test for unit roots before regressing levels ($|t| > 40$ in @{sp.b40}\\% of cases, not drawn)', 'lecția: testați existența rădăcinii unitare înainte de a estima regresii în niveluri ($|t| > 40$ în @{sp.b40}\\% din cazuri, valori care nu apar în grafic)')])],
    h='0.48\\textheight')

D.frame(T('The Dickey--Fuller Test', 'Testul Dickey--Fuller'), items(
    (T('Subtract $p_{t-1}$: $\\Delta p_t = c + \\gamma\\, p_{t-1} + \\varepsilon_t$ with $\\gamma = \\phi - 1$ \\refDF',
       'Scădem $p_{t-1}$: $\\Delta p_t = c + \\gamma\\, p_{t-1} + \\varepsilon_t$ cu $\\gamma = \\phi - 1$ \\refDF'),
     [T('$\\Delta p_t = p_t - p_{t-1}$; $c$: a constant; $\\gamma < 0$ pulls $p_t$ back towards a level', '$\\Delta p_t = p_t - p_{t-1}$; $c$: o constantă; $\\gamma < 0$ readuce $p_t$ spre un nivel'),
      T('$H_0$: $\\gamma = 0$ (unit root); $H_1$: $\\gamma < 0$ (stationary); a one-sided test', '$H_0$: $\\gamma = 0$ (rădăcină unitară); $H_1$: $\\gamma < 0$ (staționară); un test unilateral'),
      T('statistic: $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$, the usual $t$-ratio of OLS (ordinary least squares)',
        'statistica: $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$, raportul $t$ obișnuit din OLS (ordinary least squares, metoda celor mai mici pătrate)')]),
    (T('Under $H_0$, $\\tau$ does \\textbf{not} follow a $t$ or Normal law: its critical values are more negative', 'În ipoteza $H_0$, $\\tau$ \\textbf{nu} urmează o lege $t$ sau Normală: valorile ei critice sînt mai negative'),
     [T('5\\% critical values \\refMacKinnon: $@{df.c}$ with a constant, $@{df.ct}$ with a constant and a trend (Normal: $-1.645$)',
        'valorile critice de 5\\% \\refMacKinnon: $@{df.c}$ cu constantă, $@{df.ct}$ cu constantă și tendință (distribuția Normală: $-1{,}645$)')]),
    (T('Three specifications: no constant; constant $c$; constant and trend $c + bt$', 'Trei specificații: fără constantă; constanta $c$; constantă și tendință $c + bt$'),
     [T('log prices drift upwards: use constant and trend; returns: constant only', 'prețurile logaritmice cresc în timp: folosim constantă și tendință; randamentele: doar constantă')])))

D.frame(T('The Augmented Test and a Worked Example', 'Testul augmentat și un exemplu lucrat'), items(
    (T('\\textbf{ADF} (augmented Dickey--Fuller) \\refSD: add lagged differences to remove autocorrelation from $\\varepsilon_t$',
       '\\textbf{ADF} (augmented Dickey--Fuller, Dickey--Fuller augmentat) \\refSD: adăugăm laguri ale diferențelor ca să eliminăm autocorelația din $\\varepsilon_t$'),
     [T('$\\Delta p_t = c + bt + \\gamma\\, p_{t-1} + \\sum_{j=1}^{L} \\delta_j \\Delta p_{t-j} + \\varepsilon_t$; same $\\tau$, same critical values',
        '$\\Delta p_t = c + bt + \\gamma\\, p_{t-1} + \\sum_{j=1}^{L} \\delta_j \\Delta p_{t-j} + \\varepsilon_t$; același $\\tau$, aceleași valori critice'),
      T('$bt$: a linear trend; $\\delta_j$: the coefficients of the $L$ lagged differences', '$bt$: o tendință liniară; $\\delta_j$: coeficienții celor $L$ laguri ale diferenței $\\Delta p_t$'),
      T('$L$ chosen by AIC (Akaike information criterion, Chapter 6), at most $12(T/100)^{1/4}$', '$L$ ales prin AIC (Akaike information criterion, criteriul informațional Akaike, Capitolul 6), cel mult $12(T/100)^{1/4}$')]),
    (T('\\textbf{Worked example}: BET log price, constant, no lags, $T = @{df.n}$', '\\textbf{Exemplu lucrat}: prețul logaritmic BET, constantă, fără laguri, $T = @{df.n}$'),
     [T('OLS: $\\hat\\gamma = @{df.g}$, $\\mathrm{SE}(\\hat\\gamma) = @{df.se}$', 'OLS: $\\hat\\gamma = @{df.g}$, $\\mathrm{SE}(\\hat\\gamma) = @{df.se}$'),
      T('$\\tau = @{df.g}/@{df.se} = @{df.tau} > @{df.c}$: do not reject the unit root', '$\\tau = @{df.g}/@{df.se} = @{df.tau} > @{df.c}$: nu respingem rădăcina unitară'),
      T('a usual $t$-test (critical value $-1.645$) would have rejected it: the wrong distribution gives the wrong answer',
        'un test $t$ obișnuit (valoarea critică $-1{,}645$) ar fi respins-o: distribuția greșită dă răspunsul greșit')])))

D.frame(T('Phillips--Perron and KPSS', 'Phillips--Perron și KPSS'), items(
    (T('\\textbf{PP} (Phillips--Perron) \\refPP: the Dickey--Fuller regression without lags; $\\tau$ is corrected for autocorrelation and heteroskedasticity',
       '\\textbf{PP} (Phillips--Perron) \\refPP: regresia Dickey--Fuller fără laguri; $\\tau$ este corectat pentru autocorelație și heteroscedasticitate'),
     [T('the correction uses a long-run variance with Bartlett weights \\refNW; same $H_0$ and critical values as ADF',
        'corecția folosește o varianță pe termen lung cu ponderi Bartlett \\refNW; aceeași $H_0$ și aceleași valori critice ca ADF'),
      T('long-run variance $\\hat\\lambda^2 = \\hat\\gamma(0) + 2\\sum_{j=1}^{L}\\left(1 - \\frac{j}{L+1}\\right)\\hat\\gamma(j)$, with $\\hat\\gamma(j)$ the autocovariances of the residuals',
        'varianța pe termen lung $\\hat\\lambda^2 = \\hat\\gamma(0) + 2\\sum_{j=1}^{L}\\left(1 - \\frac{j}{L+1}\\right)\\hat\\gamma(j)$, cu $\\hat\\gamma(j)$ autocovarianțele reziduurilor')]),
    (T('\\textbf{KPSS} (Kwiatkowski--Phillips--Schmidt--Shin) \\refKPSS: the hypotheses are \\textbf{reversed}', '\\textbf{KPSS} (Kwiatkowski--Phillips--Schmidt--Shin) \\refKPSS: ipotezele sînt \\textbf{inversate}'),
     [T('$H_0$: stationary (around a constant or a trend); $H_1$: unit root', '$H_0$: staționară (în jurul unei constante sau al unei tendințe); $H_1$: rădăcină unitară'),
      T('$\\mathrm{KPSS} = \\sum_{t=1}^{T} S_t^2/(T^2\\hat\\lambda^2)$, $S_t$ the partial sums of the residuals, $\\hat\\lambda^2$ their long-run variance',
        '$\\mathrm{KPSS} = \\sum_{t=1}^{T} S_t^2/(T^2\\hat\\lambda^2)$, $S_t$ sumele parțiale ale reziduurilor, $\\hat\\lambda^2$ varianța lor pe termen lung'),
      T('reject stationarity if $\\mathrm{KPSS} > 0.463$ (constant) or $> 0.146$ (trend), at 5\\%', 'respingem staționaritatea dacă $\\mathrm{KPSS} > 0{,}463$ (constantă) sau $> 0{,}146$ (tendință), la 5\\%')]),
    (T('Use both: ADF/PP ask ``is there evidence against a unit root?\'\'; KPSS asks ``is there evidence against stationarity?\'\'',
       'Folosiți-le pe amîndouă: ADF/PP întreabă „există dovezi împotriva rădăcinii unitare?”; KPSS întreabă „există dovezi împotriva staționarității?”'),
     [T('ADF does not reject and KPSS rejects: $I(1)$; ADF rejects and KPSS does not: $I(0)$; both or neither reject: inconclusive',
        'ADF nu respinge și KPSS respinge: $I(1)$; ADF respinge și KPSS nu: $I(0)$; ambele resping sau niciunul nu respinge: neconcludent')])))

chart(T('The BET: Log Price and Returns', 'BET: prețul logaritmic și randamentele'), 'sfm_ch7_unit_root_bet', 'SFM_ch7_unit_roots', [
    T('Log price: ADF does not reject a unit root (p = @{ur.bet.price.adf.p}), KPSS rejects stationarity: $I(1)$', 'Prețul logaritmic: ADF nu respinge rădăcina unitară (p = @{ur.bet.price.adf.p}), KPSS respinge staționaritatea: $I(1)$'),
    T('Returns: ADF rejects, KPSS does not: stationary, $I(0)$; all statistics on the next slide', 'Randamentele: ADF respinge, KPSS nu: staționare, $I(0)$; toate statisticile sînt pe slide-ul următor'),
    T('The 2008 fall looks like a temporary deviation, but the tests say: permanent shocks, no return to a trend line',
      'Căderea din 2008 pare o abatere temporară, dar testele indică șocuri permanente, fără revenire la o dreaptă de tendință')],
    h='0.52\\textheight')


def urow(k):
    return (f'{SHORT[k]} & ${{@{{ur.{k}.price.adf}}}}$ & ${{@{{ur.{k}.price.pp}}}}$ & ${{@{{ur.{k}.price.kpss}}}}$ & '
            f'${{@{{ur.{k}.ret.adf}}}}$ & ${{@{{ur.{k}.ret.pp}}}}$ & ${{@{{ur.{k}.ret.kpss}}}}$')


D.frame(T('Unit-Root Tests on Prices and Returns', 'Teste de rădăcină unitară pe prețuri și randamente'), table(
    'lrrrrrr', T('& \\multicolumn{3}{c}{\\textbf{log price} (constant, trend)} & \\multicolumn{3}{c}{\\textbf{returns} (constant)} \\\\ & ADF & PP & KPSS & ADF & PP & KPSS',
                 '& \\multicolumn{3}{c}{\\textbf{preț logaritmic} (constantă, tendință)} & \\multicolumn{3}{c}{\\textbf{randamente} (constantă)} \\\\ & ADF & PP & KPSS & ADF & PP & KPSS'),
    [urow(k) for k in ASSETS + STOCKS], size='scriptsize') + items(
    T('5\\% critical values: ADF and PP $@{df.ct}$ (trend), $@{df.c}$ (constant); KPSS $0.146$ (trend), $0.463$ (constant)',
      'Valori critice de 5\\%: ADF și PP $@{df.ct}$ (tendință), $@{df.c}$ (constantă); KPSS $0{,}146$ (tendință), $0{,}463$ (constantă)'),
    T('Log prices: no ADF/PP rejection except TLV (ADF p = @{ur.tlv.price.adf.p}); KPSS rejects stationarity for all: prices are $I(1)$',
      'Prețurile logaritmice: nicio respingere ADF/PP cu excepția TLV (ADF p = @{ur.tlv.price.adf.p}); KPSS respinge staționaritatea peste tot: prețurile sînt $I(1)$'),
    T('Returns: ADF and PP reject strongly; KPSS never rejects (p @{ur.sp500.ret.kpss.p}): returns are $I(0)$',
      'Randamentele: ADF și PP resping puternic; KPSS nu respinge niciodată (p @{ur.sp500.ret.kpss.p}): randamentele sînt $I(0)$')) + ql('SFM_ch7_unit_roots'), 'footnotesize')

D.frame(T('A Unit Root Is Not Proof of Efficiency', 'O rădăcină unitară nu dovedește eficiența'), items(
    (T('A unit root in prices is \\textbf{necessary} for a random walk, but \\textbf{not sufficient}', 'O rădăcină unitară în prețuri este \\textbf{necesară} pentru un mers aleator, dar \\textbf{nu suficientă}'),
     [T('example: $r_t = 0.3\\,r_{t-1} + \\varepsilon_t$ makes $p_t$ an $I(1)$ series whose increments are predictable',
        'exemplu: $r_t = 0{,}3\\,r_{t-1} + \\varepsilon_t$ face din $p_t$ o serie $I(1)$ ale cărei creșteri sînt previzibile'),
      T('ADF cannot distinguish it from a random walk; the BET has a unit root and autocorrelated returns', 'ADF nu o poate deosebi de un mers aleator; BET are rădăcină unitară și randamente autocorelate')]),
    (T('A stationary price (no unit root) would contradict efficiency: prices would return to a predictable level',
       'Un preț staționar (fără rădăcină unitară) ar contrazice eficiența: prețurile ar reveni la un nivel previzibil'),
     [T('TLV: ADF rejects at 5\\% with a trend, KPSS also rejects: an inconclusive case, not a profitable rule', 'TLV: ADF respinge la 5\\% cu tendință, iar KPSS respinge și el: un caz neconcludent, nu o regulă profitabilă')]),
    T('Unit-root tests have low power against $\\phi$ close to 1: the efficiency question is about the increments, and needs tests on returns',
      'Testele de rădăcină unitară au putere mică împotriva unui $\\phi$ apropiat de 1: întrebarea eficienței privește creșterile și cere teste pe randamente')))

# =============================================================================
# 7. TESTUL VARIANCE RATIO
# =============================================================================
D.section('The Variance-Ratio Test', 'Testul variance ratio')

D.frame(T('The Idea of the Variance Ratio', 'Ideea raportului varianțelor'), items(
    (T('Under a random walk, $\\mathrm{Var}(r_t(q)) = q\\,\\mathrm{Var}(r_t)$; the \\textbf{VR} (variance ratio) compares the two sides',
       'În ipoteza de mers aleator, $\\mathrm{Var}(r_t(q)) = q\\,\\mathrm{Var}(r_t)$; \\textbf{VR} (variance ratio, raportul varianțelor) compară cei doi termeni'),
     [T('$\\mathrm{VR}(q) = \\dfrac{\\mathrm{Var}(r_t(q))}{q\\,\\mathrm{Var}(r_t)} = 1 + 2\\sum_{k=1}^{q-1}\\Big(1 - \\dfrac{k}{q}\\Big)\\rho(k)$ \\refLM, \\refCLM',
        '$\\mathrm{VR}(q) = \\dfrac{\\mathrm{Var}(r_t(q))}{q\\,\\mathrm{Var}(r_t)} = 1 + 2\\sum_{k=1}^{q-1}\\Big(1 - \\dfrac{k}{q}\\Big)\\rho(k)$ \\refLM, \\refCLM')]),
    (T('Reading the number', 'Interpretarea valorii'),
     [T('$\\mathrm{VR}(q) = 1$: random walk (RW1, RW2, RW3 all give 1)', '$\\mathrm{VR}(q) = 1$: mers aleator (RW1, RW2, RW3 dau toate 1)'),
      T('$\\mathrm{VR}(q) > 1$: positive autocorrelation, \\textbf{momentum} (moves continue)', '$\\mathrm{VR}(q) > 1$: autocorelație pozitivă, \\textbf{momentum} (mișcările continuă)'),
      T('$\\mathrm{VR}(q) < 1$: negative autocorrelation, \\textbf{mean reversion} (moves are partly undone)', '$\\mathrm{VR}(q) < 1$: autocorelație negativă, \\textbf{mean reversion} (revenire la medie: mișcările sînt parțial anulate)')]),
    T('One number summarises $q - 1$ autocorrelations with declining weights: more power against persistent, small autocorrelations',
      'Un singur număr rezumă $q - 1$ autocorelații cu ponderi descrescătoare: mai multă putere împotriva autocorelațiilor mici și persistente')))

D.frame(T('Worked Example: VR from the ACF of the S\\&P 500', 'Exemplu lucrat: VR din ACF-ul S\\&P 500'), items(
    (T('$\\hat\\rho(1), \\dots, \\hat\\rho(4)$ of the S\\&P 500: $@{ex.rs1}$, $@{ex.rs2}$, $@{ex.rs3}$, $@{ex.rs4}$', '$\\hat\\rho(1), \\dots, \\hat\\rho(4)$ pentru S\\&P 500: $@{ex.rs1}$; $@{ex.rs2}$; $@{ex.rs3}$; $@{ex.rs4}$'),
     [T('$q = 2$: $\\mathrm{VR}(2) = 1 + \\hat\\rho(1) = @{ex.vr2}$', '$q = 2$: $\\mathrm{VR}(2) = 1 + \\hat\\rho(1) = @{ex.vr2}$'),
      T('$q = 5$: weights $2(1 - k/5) = 1.6, 1.2, 0.8, 0.4$: $\\mathrm{VR}(5) = 1 + 1.6\\hat\\rho(1) + 1.2\\hat\\rho(2) + 0.8\\hat\\rho(3) + 0.4\\hat\\rho(4) = @{ex.vr5}$',
        '$q = 5$: ponderile $2(1 - k/5) = 1{,}6;\\ 1{,}2;\\ 0{,}8;\\ 0{,}4$: $\\mathrm{VR}(5) = 1 + 1{,}6\\hat\\rho(1) + 1{,}2\\hat\\rho(2) + 0{,}8\\hat\\rho(3) + 0{,}4\\hat\\rho(4) = @{ex.vr5}$')]),
    (T('The Lo--MacKinlay estimator on the same data: $\\mathrm{VR}(2) = @{vr.sp500.2}$, $\\mathrm{VR}(5) = @{vr.sp500.5}$', 'Estimatorul Lo--MacKinlay pe aceleași date: $\\mathrm{VR}(2) = @{vr.sp500.2}$, $\\mathrm{VR}(5) = @{vr.sp500.5}$'),
     [T('almost the same numbers: the two forms differ only by small-sample corrections', 'aproape aceleași cifre: cele două forme diferă doar prin corecții de eșantion mic')]),
    T('Meaning: the weekly variance is about $@{vr.sp500.5}$ of five daily variances: daily moves of the S\\&P 500 are partly reversed within a week',
      'Interpretare: varianța săptămînală reprezintă circa $@{vr.sp500.5}$ din suma a cinci varianțe zilnice: mișcările zilnice ale S\\&P 500 sînt parțial anulate într-o săptămînă')))

D.frame(T('Lo and MacKinlay (1988): Estimator and Tests', 'Lo și MacKinlay (1988): estimator și teste'), cols(items(
    (T('With $T$ returns, mean $\\hat\\mu$ and overlapping $q$-day sums \\refLM:', 'Cu $T$ randamente, media $\\hat\\mu$ și sume pe $q$ zile suprapuse \\refLM:'),
     [T('$\\hat\\sigma_a^2 = \\frac{1}{T-1}\\sum_t (r_t - \\hat\\mu)^2$, \\quad $\\hat\\sigma_c^2(q) = \\frac{1}{m}\\sum_t (r_t(q) - q\\hat\\mu)^2$',
        '$\\hat\\sigma_a^2 = \\frac{1}{T-1}\\sum_t (r_t - \\hat\\mu)^2$, \\quad $\\hat\\sigma_c^2(q) = \\frac{1}{m}\\sum_t (r_t(q) - q\\hat\\mu)^2$'),
      T('$\\hat\\sigma_a^2$: the variance of one-day returns; $\\hat\\sigma_c^2(q)$: the variance of $q$-day sums, divided by $q$', '$\\hat\\sigma_a^2$: varianța randamentelor pe o zi; $\\hat\\sigma_c^2(q)$: varianța sumelor pe $q$ zile, împărțită la $q$'),
      T('$m = q(T - q + 1)(1 - q/T)$ corrects the bias; $\\widehat{\\mathrm{VR}}(q) = \\hat\\sigma_c^2(q)/\\hat\\sigma_a^2$', '$m = q(T - q + 1)(1 - q/T)$ corectează deplasarea; $\\widehat{\\mathrm{VR}}(q) = \\hat\\sigma_c^2(q)/\\hat\\sigma_a^2$')]),
    (T('\\textbf{Homoskedastic} test (RW1): $Z(q) = \\dfrac{\\widehat{\\mathrm{VR}}(q) - 1}{\\sqrt{2(2q-1)(q-1)/(3qT)}} \\approx N(0, 1)$',
       'Testul \\textbf{homoscedastic} (RW1): $Z(q) = \\dfrac{\\widehat{\\mathrm{VR}}(q) - 1}{\\sqrt{2(2q-1)(q-1)/(3qT)}} \\approx N(0, 1)$')),
    (T('\\textbf{Robust} test (RW3, martingale): $Z^*(q) = \\dfrac{\\widehat{\\mathrm{VR}}(q) - 1}{\\sqrt{\\hat\\theta(q)/T}} \\approx N(0, 1)$',
       'Testul \\textbf{robust} (RW3, martingal): $Z^*(q) = \\dfrac{\\widehat{\\mathrm{VR}}(q) - 1}{\\sqrt{\\hat\\theta(q)/T}} \\approx N(0, 1)$'),
     [T('$\\hat\\theta(q) = \\sum_{j=1}^{q-1}\\big[2(q-j)/q\\big]^2\\hat\\delta(j)$, with the $\\hat\\delta(j)$ of the robust ACF band', '$\\hat\\theta(q) = \\sum_{j=1}^{q-1}\\big[2(q-j)/q\\big]^2\\hat\\delta(j)$, cu $\\hat\\delta(j)$ din banda ACF robustă'),
      T('$\\hat\\theta(q)/T$: the variance of $\\widehat{\\mathrm{VR}}(q)$ that allows for volatility clustering', '$\\hat\\theta(q)/T$: varianța lui $\\widehat{\\mathrm{VR}}(q)$ care ține seama de volatility clustering')])),
    ph('lo', T('Andrew W. Lo, MIT', 'Andrew W. Lo, MIT'), h='0.40\\textheight'), wl='0.66', wr='0.30') + ql('SFM_ch7_variance_ratio'), 'footnotesize')

chart(T('Why the Robust Test: a Monte Carlo Check', 'Necesitatea testului robust: o verificare Monte Carlo'), 'sfm_ch7_vr_size', 'SFM_ch7_variance_ratio', [
    (T('1\\,000 samples of $T = 2000$ returns with $H_0$ true: i.i.d.\\ Normal, and GARCH(1,1) ($\\alpha = 0.12$, $\\beta = 0.86$)', '1\\,000 de eșantioane de $T = 2000$ de randamente cu $H_0$ adevărată: Normale i.i.d.\\ și GARCH(1,1) ($\\alpha = 0{,}12$, $\\beta = 0{,}86$)'),
     [T('GARCH: uncorrelated returns, with volatility clustering ($\\alpha$: reaction to yesterday\'s shock, $\\beta$: persistence)', 'GARCH: randamente necorelate, dar cu volatility clustering ($\\alpha$: reacția la șocul de ieri, $\\beta$: persistența)')]),
    T('i.i.d.: both reject about 5\\%; GARCH: $Z(2)$ rejects @{size.garch.2.z}\\% of the time, $Z^*(2)$ only @{size.garch.2.zs}\\%',
      'i.i.d.: ambele resping circa 5\\%; GARCH: $Z(2)$ respinge în @{size.garch.2.z}\\% din cazuri, $Z^*(2)$ doar în @{size.garch.2.zs}\\%'),
    T('The \\textbf{size} of a test: how often it rejects a true $H_0$; $Z$ finds ``inefficiency\'\' that is only volatility clustering',
      '\\textbf{Mărimea} unui test: cît de des respinge o $H_0$ adevărată; $Z$ „găsește” o ineficiență care nu este decît volatility clustering')],
    h='0.48\\textheight')

chart(T('Variance Ratios for Horizons of 2 to 40 Days', 'Rapoarte ale varianțelor pentru orizonturi de 2--40 de zile'), 'sfm_ch7_vr_profile', 'SFM_ch7_variance_ratio', [
    T('S\\&P 500: VR falls below 1 at every horizon (mean reversion); BET: VR rises up to about @{vr.bet.20} at 20 days and beyond (momentum)',
      'S\\&P 500: VR scade sub 1 la orice orizont (mean reversion); BET: VR crește pînă la circa @{vr.bet.20} la 20 de zile și peste (momentum)'),
    T('DAX and Bitcoin: inside the robust band (shaded) at every horizon; the i.i.d.\\ band (dashed) is much narrower',
      'DAX și Bitcoin: în interiorul benzii robuste (zona umbrită) la orice orizont; banda i.i.d.\\ (linia întreruptă) este mult mai îngustă')],
    h='0.58\\textheight')


def vrow(k):
    return (f'{SHORT[k]} & ' + ' & '.join(f'$@{{vr.{k}.{q}}}$' for q in ['2', '5', '10', '20']) + ' & '
            + ' & '.join(f'${{@{{vr.{k}.zs{q}}}}}$' for q in ['2', '5', '10', '20']))


D.frame(T('Variance Ratios and Robust Tests', 'Rapoarte ale varianțelor și teste robuste'), table(
    'lrrrrrrrr', '& VR(2) & VR(5) & VR(10) & VR(20) & $Z^*(2)$ & $Z^*(5)$ & $Z^*(10)$ & $Z^*(20)$',
    [vrow(k) for k in ASSETS + STOCKS], size='scriptsize') + items(
    T('$|Z^*| > 1.96$: reject the random walk at 5\\% for that horizon', '$|Z^*| > 1{,}96$: respingem mersul aleator la 5\\% pentru acel orizont'),
    T('S\\&P 500: significant mean reversion at all four horizons; BET: significant momentum at all four', 'S\\&P 500: mean reversion semnificativ la toate cele patru orizonturi; BET: momentum semnificativ la toate patru'),
    T('Transgaz: $Z^*(2) = @{vr.tgn.zs2}$ and $Z^*(5) = @{vr.tgn.zs5}$; DAX, Bitcoin, TLV, SNP, BRD: no rejection',
      'Transgaz: $Z^*(2) = @{vr.tgn.zs2}$ și $Z^*(5) = @{vr.tgn.zs5}$; DAX, Bitcoin, TLV, SNP, BRD: nicio respingere'),
    T('But eight tests per series at once: some ``significant\'\' horizons appear by chance', 'Atenție: sînt opt teste simultane pe fiecare serie, deci unele orizonturi „semnificative” apar din întîmplare')) + ql('SFM_ch7_variance_ratio'), 'footnotesize')

D.frame(T('Several Horizons at Once: the Chow--Denning Test', 'Mai multe orizonturi deodată: testul Chow--Denning'), items(
    (T('Testing $q = 2, 5, 10, 20$ separately at 5\\% gives a much higher chance of at least one false rejection', 'Testarea separată pentru $q = 2, 5, 10, 20$ la 5\\% crește mult probabilitatea de a obține cel puțin o respingere falsă'),
     [T('for 4 independent tests: $1 - 0.95^4 = @{mt.4}\\%$', 'pentru 4 teste independente: $1 - 0{,}95^4 = @{mt.4}\\%$')]),
    (T('\\textbf{CD} (Chow--Denning) statistic \\refCD: $\\mathrm{CD} = \\max_{i} |Z^*(q_i)|$, $i = 1, \\dots, m$', 'Statistica \\textbf{CD} (Chow--Denning) \\refCD: $\\mathrm{CD} = \\max_{i} |Z^*(q_i)|$, $i = 1, \\dots, m$'),
     [T('critical value from the studentised maximum modulus law: $P(\\mathrm{CD} \\le c) = (2\\Phi(c) - 1)^m$', 'valoarea critică din distribuția modulului maxim studentizat: $P(\\mathrm{CD} \\le c) = (2\\Phi(c) - 1)^m$'),
      T('$m = 4$, 5\\%: $c = @{cd.crit}$ instead of $1.96$; $\\Phi$: the standard Normal distribution function', '$m = 4$, 5\\%: $c = @{cd.crit}$ în loc de $1{,}96$; $\\Phi$: funcția de repartiție Normală standard')]),
    T('Reject the random walk if the largest $|Z^*|$ exceeds $@{cd.crit}$: one decision for all horizons', 'Respingem mersul aleator dacă cel mai mare $|Z^*|$ depășește $@{cd.crit}$: o singură decizie pentru toate orizonturile')))

D.frame(T('An Automatic Choice of the Horizon', 'O alegere automată a orizontului'), items(
    (T('The horizon $q$ can be chosen from the data: the automatic VR test \\refChoi', 'Orizontul $q$ poate fi ales din date: testul VR automat \\refChoi'),
     [T('$\\mathrm{VR}(k) = 1 + 2\\sum_{i=1}^{T-1} w(i/k)\\,\\hat\\rho(i)$, with smoothly declining weights $w$ (quadratic spectral kernel) \\refAndrews',
        '$\\mathrm{VR}(k) = 1 + 2\\sum_{i=1}^{T-1} w(i/k)\\,\\hat\\rho(i)$, cu ponderi $w$ care scad lin (nucleul spectral pătratic) \\refAndrews'),
      T('the bandwidth $k$ grows with the first-order autocorrelation and with $T$; statistic $\\sqrt{T/k}\\,(\\mathrm{VR}(k) - 1)/\\sqrt2 \\approx N(0, 1)$',
        'lățimea de bandă $k$ crește cu autocorelația de ordinul 1 și cu $T$; statistica $\\sqrt{T/k}\\,(\\mathrm{VR}(k) - 1)/\\sqrt2 \\approx N(0, 1)$')]),
    (T('\\textbf{Wild bootstrap} p-value \\refKimB: resample $r_t^* = \\eta_t(r_t - \\bar r)$ with $\\eta_t \\sim N(0, 1)$, 500 times',
       'p-value prin \\textbf{wild bootstrap} \\refKimB: reeșantionăm $r_t^* = \\eta_t(r_t - \\bar r)$ cu $\\eta_t \\sim N(0, 1)$, de 500 de ori'),
     [T('each $r_t^*$ keeps the size of $r_t$ (its volatility) but loses any autocorrelation: a robust null distribution',
        'fiecare $r_t^*$ păstrează mărimea lui $r_t$ (volatilitatea), dar pierde orice autocorelație: o distribuție nulă robustă')]),
    T('Three tools, one question: $Z^*(q)$ for a chosen horizon, CD for a set of horizons, the automatic test for a horizon chosen by the data',
      'Trei instrumente, o întrebare: $Z^*(q)$ pentru un orizont ales, CD pentru o mulțime de orizonturi, testul automat pentru un orizont ales de date')))


def crow(k):
    return (f'{SHORT[k]} & @{{vr.{k}.n}} & $@{{vr.{k}.cd}}$ & @{{vr.{k}.cdp}} & $@{{vr.{k}.k}}$ & ${{@{{vr.{k}.avr}}}}$ & @{{vr.{k}.avrp}}')


D.frame(T('Joint and Automatic Tests on Real Data', 'Teste comune și automate pe date reale'), table(
    'lrrrrrr', T('& $T$ & CD & p & $\\hat k$ & ' + 'automatic VR & p (bootstrap)', '& $T$ & CD & p & $\\hat k$ & VR automat & p (bootstrap)'),
    [crow(k) for k in ASSETS + STOCKS], size='scriptsize') + items(
    T('Both tests reject for the S\\&P 500, the BET and Transgaz; neither rejects for DAX, Bitcoin, TLV, SNP and BRD',
      'Ambele teste resping pentru S\\&P 500, BET și Transgaz; niciunul nu respinge pentru DAX, Bitcoin, TLV, SNP și BRD'),
    T('$\\hat k$ is large where the first autocorrelation is large (BET $@{vr.bet.k}$), small where it is near 0 (DAX $@{vr.dax.k}$)',
      '$\\hat k$ este mare unde prima autocorelație este mare (BET $@{vr.bet.k}$) și mic unde ea este aproape 0 (DAX $@{vr.dax.k}$)'),
    (T('Interpretation: rejection does not mean profit', 'Interpretare: respingerea nu înseamnă profit'),
     [T('an autocorrelation of $@{sc.sp500.rho}$ explains $\\hat\\rho(1)^2$, less than 1\\%, of the variance of tomorrow\'s return', 'o autocorelație de $@{sc.sp500.rho}$ explică $\\hat\\rho(1)^2$, adică mai puțin de 1\\%, din varianța randamentului de mîine'),
      T('trading costs absorb most of it', 'costurile de tranzacționare absorb cea mai mare parte')])) + ql('SFM_ch7_variance_ratio'), 'footnotesize')

# =============================================================================
# 8. EFICIENȚA PE PIEȚE ȘI ÎN TIMP
# =============================================================================
D.section('Efficiency across Markets and over Time', 'Eficiența pe piețe și în timp')

chart(T('Sixteen Markets over the Last Ten Years', 'Șaisprezece piețe în ultimii zece ani'), 'sfm_ch7_vr_markets', 'SFM_ch7_markets_efficiency', [
    T('VR(5) with its robust 95\\% interval, from 19 September 2016 to @{end}: developed, emerging and crypto markets',
      'VR(5) cu intervalul robust de 95\\%, de la 19 septembrie 2016 pînă la @{end}: piețe dezvoltate, emergente și cripto')],
    h='0.64\\textheight')

D.frame(T('Reading the Cross-Market Comparison', 'Interpretarea comparației între piețe'), items(
    (T('Only two intervals exclude 1: the PX ($Z^*(5) = @{mk.px.zs}$) and the BET ($Z^*(5) = @{mk.bet.zs}$), both above 1',
       'Doar două intervale exclud valoarea 1: PX ($Z^*(5) = @{mk.px.zs}$) și BET ($Z^*(5) = @{mk.bet.zs}$), ambele peste 1'),
     [T('two smaller Central European markets, with less liquid index stocks', 'două piețe central-europene mai mici, cu acțiuni mai puțin lichide în indice')]),
    (T('\\textbf{Many markets, many tests}: 16 tests at 5\\% give @{mt.exp} false rejections on average', '\\textbf{Multe piețe, multe teste}: 16 teste la 5\\% dau, în medie, @{mt.exp} respingeri false'),
     [T('if the tests were independent, the chance of at least one false rejection would be $1 - 0.95^{16} = @{mt.16}\\%$', 'dacă testele ar fi independente, probabilitatea de a obține cel puțin o respingere falsă ar fi $1 - 0{,}95^{16} = @{mt.16}\\%$'),
      T('Chow--Denning over four horizons: no market rejects at 5\\% (BET p = @{mk.bet.cdp}, PX p = @{mk.px.cdp})', 'Chow--Denning pe patru orizonturi: nicio piață nu respinge la 5\\% (BET p = @{mk.bet.cdp}, PX p = @{mk.px.cdp})')]),
    T('S\\&P 500: VR(5) $= @{mk.sp500.vr}$, the lowest, but with a wide interval ($Z^* = @{mk.sp500.zs}$): the large swings of 2020 inflate $\\hat\\theta$',
      'S\\&P 500: VR(5) $= @{mk.sp500.vr}$, cel mai mic, dar cu un interval larg ($Z^* = @{mk.sp500.zs}$): oscilațiile mari din 2020 măresc $\\hat\\theta$'),
    T('Bitcoin and Ethereum: VR(5) close to 1, as for developed markets', 'Bitcoin și Ethereum: VR(5) aproape de 1, ca pe piețele dezvoltate')) + ql('SFM_ch7_markets_efficiency'))

D.frame(T('The Adaptive Markets Hypothesis', 'Ipoteza piețelor adaptive'), cols(items(
    (T('\\textbf{AMH} (adaptive markets hypothesis) \\refLo: markets are ecosystems of traders who learn and adapt',
       '\\textbf{AMH} (adaptive markets hypothesis, ipoteza piețelor adaptive) \\refLo: piețele sînt ecosisteme de investitori care învață și se adaptează'),
     [T('profit opportunities appear, are exploited and disappear; efficiency changes with the environment',
        'oportunitățile de profit apar, sînt exploatate și dispar; eficiența se schimbă odată cu mediul'),
      T('crises, new technology, new participants and regulation move the market away from efficiency or back towards it',
        'crizele, tehnologia nouă, participanții noi și reglementarea îndepărtează piața de eficiență sau o readuc spre ea')]),
    (T('Testable consequence: predictability varies over time', 'Consecința testabilă: previzibilitatea variază în timp'),
     [T('evidence: \\refKSL (U.S., a century of data); \\refUM; survey: \\refLB', 'dovezi: \\refKSL (SUA, un secol de date); \\refUM; sinteză: \\refLB')]),
    T('Tool: the same test on rolling windows', 'Instrumentul: același test pe ferestre mobile')),
    ph('lo', T('Andrew W. Lo, author of the AMH', 'Andrew W. Lo, autorul AMH'), h='0.42\\textheight'), wl='0.62', wr='0.34'))

chart(T('Rolling Variance Ratios', 'Rapoarte ale varianțelor pe ferestre mobile'), 'sfm_ch7_rolling_vr', 'SFM_ch7_adaptive_markets', [
    T('VR(5) and $Z^*(5)$ on windows of 500 trading days (about two years), a new window every 21 days, dated at the window end',
      'VR(5) și $Z^*(5)$ pe ferestre de 500 de zile de tranzacționare (circa doi ani), o fereastră nouă la fiecare 21 de zile, datată la sfîrșitul ferestrei')],
    h='0.64\\textheight')

D.frame(T('Reading the Rolling Variance Ratios', 'Interpretarea rapoartelor varianțelor pe ferestre mobile'), items(
    (T('BET: VR(5) up to @{ro.bet.max} in the window ending in @{ro.bet.ymax}; closer to 1 after 2008, apart from short episodes', 'BET: VR(5) pînă la @{ro.bet.max} în fereastra care se termină în @{ro.bet.ymax}; mai aproape de 1 după 2008, cu excepția unor episoade scurte'),
     [T('@{ro.bet.rej}\\% of the windows reject at 5\\%, all of them with VR $> 1$ (momentum)', '@{ro.bet.rej}\\% dintre ferestre resping la 5\\%, toate cu VR $> 1$ (momentum)')]),
    (T('S\\&P 500: VR(5) mostly below 1; the lowest, @{ro.sp500.min}, in the window ending in @{ro.sp500.ymin}', 'S\\&P 500: VR(5) mai ales sub 1; cel mai mic, @{ro.sp500.min}, în fereastra care se termină în @{ro.sp500.ymin}'),
     [T('@{ro.sp500.rej}\\% of the windows reject, all with VR $< 1$: reversals are strongest in crises', '@{ro.sp500.rej}\\% dintre ferestre resping, toate cu VR $< 1$: revenirile sînt cele mai puternice în crize')]),
    T('Bitcoin: no window rejects since 2016 (@{ro.btc.n} windows)', 'Bitcoin: nicio fereastră nu respinge din 2016 (@{ro.btc.n} ferestre)'),
    T('Caution: consecutive windows share 479 of 500 days; the share of rejecting windows is a description, not a test',
      'Atenție: ferestrele consecutive au în comun 479 din 500 de zile; ponderea ferestrelor care resping este o descriere, nu un test')) + ql('SFM_ch7_adaptive_markets'))


def srow(k, i):
    return (f'{SHORT[k]} & @{{sub.{k}.{i}.y0}}--@{{sub.{k}.{i}.y1}} & @{{sub.{k}.{i}.n}} & ${{@{{sub.{k}.{i}.rho1}}}}$ & $@{{sub.{k}.{i}.vr}}$ & '
            f'${{@{{sub.{k}.{i}.zs}}}}$ & @{{sub.{k}.{i}.cdp}}')


D.frame(T('Three Sub-Periods per Market', 'Trei subperioade pentru fiecare piață'), table(
    'llrrrrr', T('& period & $T$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ & CD p', '& perioada & $T$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ & CD p'),
    [srow(k, i) for k in ['bet', 'sp500', 'btc'] for i in (1, 2, 3)], size='scriptsize') + items(
    (T('BET: strong momentum in @{sub.bet.1.y0}--@{sub.bet.1.y1} ($\\hat\\rho(1) = @{sub.bet.1.rho1}$), much weaker afterwards', 'BET: momentum puternic în @{sub.bet.1.y0}--@{sub.bet.1.y1} ($\\hat\\rho(1) = @{sub.bet.1.rho1}$), mult mai slab după aceea'),
     [T('since 2017 VR(5) is still above 1, but the joint test does not reject', 'din 2017 VR(5) este tot peste 1, dar testul comun nu respinge')]),
    (T('S\\&P 500: the negative autocorrelation appears after 2002', 'S\\&P 500: autocorelația negativă apare după 2002'),
     [T('Bitcoin: no rejection in any period since 2014; \\refUrquhart found inefficiency in its earliest years', 'Bitcoin: nicio respingere în vreuna dintre subperioadele de după 2014; \\refUrquhart a găsit ineficiență în primii lui ani')])) + ql('SFM_ch7_adaptive_markets'), 'footnotesize')

D.frame(T('Why the Early BET Was Predictable', 'Previzibilitatea BET în primii ani'), cols(items(
    (T('\\textbf{Thin trading}: few trades per day in many index stocks', '\\textbf{Tranzacționare redusă} (thin trading): puține tranzacții pe zi la multe acțiuni din indice'),
     [T('a closing price may be hours old; news enters the index over several days', 'un preț de închidere poate fi stale (neactualizat de cîteva ore); știrile intră în indice pe parcursul mai multor zile'),
      T('the index then shows positive autocorrelation even if each stock is efficient', 'indicele arată atunci autocorelație pozitivă chiar dacă fiecare acțiune este eficientă')]),
    (T('Changes after 2007', 'Schimbările de după 2007'),
     [T('more liquidity, foreign investors, EU membership, the FTSE Russell upgrade to Secondary Emerging (2020) \\refFTSE',
        'mai multă lichiditate, investitori străini, aderarea la UE, reclasificarea de către FTSE Russell la statutul Secondary Emerging (2020) \\refFTSE')]),
    T('Crashes on the Bucharest Stock Exchange: \\refPeleA', 'Crahurile de la Bursa de Valori București: \\refPeleA')),
    ph('bvb', T('The Stock Exchange Palace, Bucharest', 'Palatul Bursei, București'), h='0.32\\textheight'), wl='0.60', wr='0.36'))

# =============================================================================
# 9. FORMA SEMI-TARE, FORMA TARE ȘI ANOMALII
# =============================================================================
D.section('Semi-Strong Form, Strong Form and Anomalies', 'Forma semi-tare, forma tare și anomalii')

D.frame(T('Event Studies and the Strong Form', 'Studii de eveniment și forma tare'), items(
    (T('\\textbf{Event study} (semi-strong form) \\refMacKinlay, \\refCLM, Ch.~4: how fast does a price absorb public news?',
       '\\textbf{Studiu de eveniment} (forma semi-tare) \\refMacKinlay, \\refCLM, cap.~4: cît de repede absoarbe un preț o știre publică?'),
     [T('abnormal return: the return minus the return of a model (for example the market); cumulated over a window around the announcement',
        'randamentul anormal: randamentul minus randamentul unui model (de exemplu piața); cumulat pe o fereastră în jurul anunțului'),
      T('efficient: a jump on the day of the news, no drift afterwards', 'piață eficientă: un salt în ziua știrii, fără derivă ulterioară'),
      T('recent research: how fast futures prices absorb news, measured in nanoseconds at Eurex and CME \\refPeleB; market responses to Ethereum upgrades \\refPeleC',
        'cercetări recente: cît de repede absorb prețurile futures știrile, măsurat în nanosecunde la Eurex și CME \\refPeleB; reacția pieței la actualizările Ethereum \\refPeleC')]),
    (T('\\textbf{Strong form}: can anyone with private information earn excess returns?', '\\textbf{Forma tare}: poate cineva cu informație privată să obțină randamente în exces?'),
     [T('insiders (managers trading their own shares) do: insider trading is illegal for this reason', 'insiderii (managerii care tranzacționează acțiunile propriei companii) pot: de aceea tranzacționarea pe baza informațiilor privilegiate este ilegală'),
      T('professional fund managers mostly do not \\refJensen, \\refFF', 'administratorii profesioniști de fonduri, în majoritate, nu pot \\refJensen, \\refFF')])))

D.frame(T('Calendar Anomalies', 'Anomalii calendaristice'), items(
    (T('An \\textbf{anomaly}: a pattern in returns that the equilibrium model does not explain', 'O \\textbf{anomalie}: un tipar în randamente pe care modelul de echilibru nu îl explică'),
     [T('\\textbf{weekend effect}: low or negative Monday returns \\refFrench', '\\textbf{efectul de weekend}: randamente mici sau negative lunea \\refFrench'),
      T('\\textbf{January effect}: higher returns in January, mostly for small stocks \\refRK', '\\textbf{efectul ianuarie}: randamente mai mari în ianuarie, mai ales la acțiunile mici \\refRK'),
      T('\\textbf{turn-of-the-month effect}: returns concentrated around the first trading days of the month \\refAriel',
        '\\textbf{efectul de început de lună}: randamente concentrate în jurul primelor zile de tranzacționare ale lunii \\refAriel')]),
    (T('Do anomalies last?', 'Durează anomaliile?'),
     [T('returns of published anomalies fall after publication: traders learn \\refMP', 'randamentele anomaliilor publicate scad după publicare: investitorii învață \\refMP'),
      T('hundreds of patterns have been tested: a new one needs $|t| > 3$, not 2 \\refHLZ', 'au fost testate sute de tipare: un tipar nou trebuie să aibă $|t| > 3$, nu 2 \\refHLZ')]),
    T('Data mining: with enough calendar splits, some will look significant by chance', 'Căutarea repetată în date (data mining): cu destule împărțiri calendaristice, unele vor părea semnificative din întîmplare')))

chart(T('Day-of-Week Returns: S\\&P 500 and BET', 'Randamentele pe zile ale săptămînii: S\\&P 500 și BET'), 'sfm_ch7_calendar', 'SFM_ch7_calendar_anomalies', [
    (T('Mean daily return by weekday with 95\\% intervals (Newey--West standard errors \\refNW)', 'Randamentul zilnic mediu pe zile ale săptămînii, cu intervale de 95\\% (erori standard Newey--West \\refNW)'),
     [T('Wald test of equal means across weekdays: S\\&P 500 p = @{cal.sp500.wp}, BET p = @{cal.bet.wp}', 'testul Wald al egalității mediilor pe zile: S\\&P 500 p = @{cal.sp500.wp}, BET p = @{cal.bet.wp}')]),
    (T('BET Monday: $@{cal.bet.d0}\\%$, the only negative mean, yet not significant', 'BET lunea: $@{cal.bet.d0}\\%$, singura medie negativă, dar nesemnificativă'),
     [T('January: BET $@{cal.bet.jan}\\%$ per day against $@{cal.bet.oth}\\%$ ($t = @{cal.bet.jt}$, p = @{cal.bet.jp}); S\\&P 500: p = @{cal.sp500.jp}', 'ianuarie: BET $@{cal.bet.jan}\\%$ pe zi față de $@{cal.bet.oth}\\%$ ($t = @{cal.bet.jt}$, p = @{cal.bet.jp}); S\\&P 500: p = @{cal.sp500.jp}')]),
    T('No calendar effect survives at 5\\% in our data', 'Niciun efect calendaristic nu rezistă la 5\\% în datele noastre')],
    h='0.50\\textheight')

# =============================================================================
# 10. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Did the BET become more efficient after Romania\'s upgrade to Secondary Emerging market (September 2020)?} \\refFTSE',
       '\\textbf{A devenit BET mai eficient după trecerea României la statutul de piață Secondary Emerging (septembrie 2020)?} \\refFTSE'),
     [T('five years before: $\\hat\\rho(1) = @{up.before.rho1}$, VR(5) $= @{up.before.vr}$ ($Z^* = @{up.before.zs}$)', 'cinci ani înainte: $\\hat\\rho(1) = @{up.before.rho1}$, VR(5) $= @{up.before.vr}$ ($Z^* = @{up.before.zs}$)'),
      T('five years after: $\\hat\\rho(1) = @{up.after.rho1}$, VR(5) $= @{up.after.vr}$ ($Z^* = @{up.after.zs}$)', 'cinci ani după: $\\hat\\rho(1) = @{up.after.rho1}$, VR(5) $= @{up.after.vr}$ ($Z^* = @{up.after.zs}$)'),
      T('difference of the two VR(5): $z = @{up.z}$: no evidence of a change', 'diferența celor două VR(5): $z = @{up.z}$: nicio dovadă de schimbare')]),
    (T('Why it is open: COVID-19, new large listings and higher volumes happened at the same time', 'Întrebarea rămîne deschisă: pandemia COVID-19, noile listări mari și volumele mai mari s-au produs în același timp'),
     [T('is the index or each stock the right unit?', 'care este unitatea potrivită de analiză, indicele sau fiecare acțiune?')]),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea rezultatelor \\refWang')) + ql('SFM_ch7_adaptive_markets'))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: list studies of efficiency in Central and Eastern European markets and the tests they use',
      '\\textbf{Literatura}: o listă a studiilor despre eficiența piețelor din Europa Centrală și de Est și a testelor folosite în ele'),
    T('\\textbf{Code}: a first draft of rolling Lo--MacKinlay and automatic VR tests for BET stocks, with wild-bootstrap p-values',
      '\\textbf{Cod}: o primă versiune a testelor Lo--MacKinlay și VR automat pe ferestre mobile pentru acțiunile din BET, cu p-value-uri wild bootstrap'),
    T('\\textbf{Robustness}: other windows, other horizons, BET-TR instead of BET, trading volume as a control',
      '\\textbf{Robustețe}: alte ferestre, alte orizonturi, BET-TR în loc de BET, volumul tranzacțiilor ca variabilă de control'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that computes the Lo-MacKinlay variance ratio VR(5) and the heteroskedasticity-robust z-statistic for daily BET log returns on rolling 500-day windows, and test whether the mean VR differs before and after 21 September 2020.}',
        '\\aiprompt{Write a Python function that computes the Lo-MacKinlay variance ratio VR(5) and the heteroskedasticity-robust z-statistic for daily BET log returns on rolling 500-day windows, and test whether the mean VR differs before and after 21 September 2020.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The statistic: robust $Z^*$, not the homoskedastic $Z$; a draft that uses $Z$ will ``find\'\' inefficiency', 'Statistica: $Z^*$ robust, nu $Z$ homoscedastic; o ciornă care folosește $Z$ va „găsi” ineficiență'),
    T('Log returns of the index, not price levels; no unit-root test on returns used as an efficiency test', 'Randamentele logaritmice ale indicelui, nu nivelurile prețurilor; un test de rădăcină unitară pe randamente nu este un test de eficiență'),
    T('Rolling windows overlap: a $t$-test on the mean of rolling statistics treats dependent numbers as independent', 'Ferestrele mobile se suprapun: un test $t$ pe media statisticilor mobile tratează valori dependente ca independente'),
    T('The event date and the window are fixed before looking at the result', 'Data evenimentului și fereastra se fixează înainte de a vedea rezultatul'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: is the Bucharest market more efficient today than in 2015?',
       '\\textbf{Întrebarea}: este piața de la București mai eficientă azi decît în 2015?'),
     [T('a second question: does the change appear in the index or in the individual stocks?', 'o a doua întrebare: apare schimbarea în indice sau în acțiunile individuale?'),
      T('data: BET since 1997, BET-TR since 2014, BVB stocks (TLV, SNP, BRD, TGN) since 2010 (EODHD)', 'date: BET din 1997, BET-TR din 2014, acțiuni BVB (TLV, SNP, BRD, TGN) din 2010 (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('rolling VR(2), VR(5) with $Z^*$ and the Chow--Denning test for the index and for each stock', 'VR(2), VR(5) mobile cu $Z^*$ și testul Chow--Denning pentru indice și pentru fiecare acțiune'),
      T('compare 2015--2020 with 2020--2025 by a block bootstrap of the difference', 'comparați 2015--2020 cu 2020--2025 printr-un bootstrap pe blocuri al diferenței'),
      T('repeat for the WIG20 and the BUX as controls: did all of them change at the same time?', 'repetați pentru WIG20 și BUX, ca termeni de comparație: s-au schimbat toate în același timp?')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Efficient market: prices reflect information; weak form = past prices cannot be used to earn excess returns', 'Piață eficientă: prețurile reflectă informația; forma slabă = prețurile trecute nu pot fi folosite pentru randamente în exces'),
    T('The right null is the martingale (RW3), not RW1: volatility clustering is not inefficiency', 'Ipoteza nulă potrivită este martingalul (RW3), nu RW1: volatility clustering nu înseamnă ineficiență'),
    T('Log prices are $I(1)$ and returns $I(0)$ in all our series; a unit root alone does not prove efficiency', 'Prețurile logaritmice sînt $I(1)$, iar randamentele $I(0)$ în toate seriile noastre; rădăcina unitară, în sine, nu dovedește eficiența'),
    T('Use robust tests: $\\tilde Q$ and $Z^*(q)$; join horizons with Chow--Denning or let the data choose (automatic VR)', 'Folosiți teste robuste: $\\tilde Q$ și $Z^*(q)$; combinați orizonturile cu Chow--Denning sau lăsați datele să aleagă (VR automat)'),
    T('Evidence: S\\&P 500 slightly mean-reverting, early BET trending, most markets close to efficient today; efficiency varies over time',
      'Dovezi: S\\&P 500 revine ușor la medie, BET avea momentum în primii ani, majoritatea piețelor sînt azi aproape eficiente; eficiența variază în timp'),
    T('Statistical predictability is not profit: costs and risk decide', 'Previzibilitatea statistică nu înseamnă profit: costurile și riscul decid')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Random walk', 'Mers aleator') + ' & $p_t = \\mu + p_{t-1} + \\varepsilon_t$, \\quad $\\mathrm{Var}(r_t(q)) = q\\sigma^2$',
     T('Sample ACF', 'ACF de selecție') + ' & $\\hat\\rho(k) = \\sum_t e_te_{t-k}/\\sum_t e_t^2$, \\quad band $\\pm 1.96\\sqrt{\\hat\\delta(k)/T}$',
     'Ljung--Box & $Q_{LB}(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho(k)^2/(T-k) \\sim \\chi^2(m)$',
     T('Runs test', 'Testul runs') + ' & $E[R] = 2n_1n_2/n + 1$, \\quad $z = (R - E[R])/\\mathrm{sd}(R)$',
     'ADF & $\\Delta p_t = c + bt + \\gamma p_{t-1} + \\sum_j \\delta_j\\Delta p_{t-j} + \\varepsilon_t$, \\quad $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$',
     'KPSS & $\\sum_t S_t^2/(T^2\\hat\\lambda^2)$, \\quad $H_0$: ' + T('stationarity', 'staționaritate'),
     'VR & $\\mathrm{VR}(q) = 1 + 2\\sum_{k=1}^{q-1}(1 - k/q)\\rho(k)$',
     T('Robust VR test', 'Testul VR robust') + ' & $Z^*(q) = (\\widehat{\\mathrm{VR}}(q) - 1)/\\sqrt{\\hat\\theta(q)/T}$, \\quad $\\hat\\theta(q) = \\sum_{j<q}[2(q-j)/q]^2\\hat\\delta(j)$',
     'Chow--Denning & $\\max_i |Z^*(q_i)|$, \\quad $P(\\mathrm{CD} \\le c) = (2\\Phi(c) - 1)^m$'],
    size='footnotesize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: $\\hat\\rho(1) = 0.10$ and $\\hat\\rho(k) = 0$ for $k \\ge 2$; what does VR(2) tell us?', '\\textbf{Întrebare}: $\\hat\\rho(1) = 0{,}10$ și $\\hat\\rho(k) = 0$ pentru $k \\ge 2$; ce ne spune VR(2)?'),
     [T('\\textbf{Answer}: $1 + 0.10 = 1.10$: the two-day variance is 10\\% larger than under a random walk: momentum', '\\textbf{Răspuns}: $1 + 0{,}10 = 1{,}10$: varianța pe două zile este cu 10\\% mai mare decît în cazul unui mers aleator, deci momentum')]),
    (T('\\textbf{Question}: ADF does not reject on log prices and KPSS rejects; is the market efficient?', '\\textbf{Întrebare}: ADF nu respinge pe prețurile logaritmice, iar KPSS respinge; este piața eficientă?'),
     [T('\\textbf{Answer}: we only know that prices are $I(1)$; efficiency needs tests on the increments (ACF, VR)', '\\textbf{Răspuns}: știm doar că prețurile sînt $I(1)$; eficiența cere teste pe creșteri (ACF, VR)')]),
    (T('\\textbf{Question}: for the S\\&P 500 the homoskedastic $Z(q)$ is about twice the robust $Z^*(q)$; which one should we report?', '\\textbf{Întrebare}: pentru S\\&P 500, statistica homoscedastică $Z(q)$ este de circa două ori mai mare decît statistica robustă $Z^*(q)$; pe care o raportăm?'),
     [T('\\textbf{Answer}: $Z^*(q)$: volatility clustering raises the variance of $\\widehat{\\mathrm{VR}}(q)$, which $Z(q)$ ignores, so $Z(q)$ rejects a true martingale too often', '\\textbf{Răspuns}: $Z^*(q)$: volatility clustering mărește varianța lui $\\widehat{\\mathrm{VR}}(q)$, pe care $Z(q)$ o ignoră, deci $Z(q)$ respinge prea des un martingal adevărat')]),
    T('Next: Chapter 8, volatility estimators and volatility clustering', 'Urmează: Capitolul 8, estimatori de volatilitate și volatility clustering')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
