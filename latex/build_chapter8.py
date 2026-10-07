r"""
build_chapter8.py -- Capitolul 8 (Estimatori de volatilitate și volatility clustering), EN + RO dintr-o singură sursă
===================================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_08/ch8_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele rezolvate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter8_volatility_estimators.tex
  RO/Cursuri/capitol8_estimatori_volatilitate.tex
Rulare:
  python3 Quantlets/Ch_08/generate_all_charts.py
  python3 latex/build_chapter8.py && python3 latex/sfm_build.py compile 8
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Quantlets', 'Ch_08'))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch8_common import BIB, NAMES, OHLC_ASSETS, QLURL, REFS, SHORT, T, fix_refs, load, put_date, values   # noqa: E402
import generate_all_charts as g   # noqa: E402

N = load()
V = values(N)
D = Deck(8, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_08}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.58\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'amsterdam': ('ch8_amsterdam_1987.jpg', C + 'Effectenbeurs_Amsterdam_na_koersval,_Bestanddeelnr_934-1094.jpg',
                  '⟦Photo||Foto⟧: Bart Molendijk / Anefo (1987); CC0; Wikimedia Commons'),
    'engle': ('ch8_engle_2017.png', C + 'Robert_Engle_SantiagoWEAI2017.png',
              '⟦Photo||Foto⟧: Econterms (2017); CC BY-SA 4.0; Wikimedia Commons'),
    'jpm': ('ch8_jpmorgan_23wall.jpg', C + 'J._P._Morgan_\\%26_Company_Building_23_Wall_Street.jpg',
            '⟦Photo||Foto⟧: Beyond My Ken (2012); CC BY-SA 4.0; Wikimedia Commons'),
    'cbot': ('ch8_cbot_1973.jpg', C + '20120105-OC-AMW-0425_(7042322619).jpg',
             '⟦Photo||Foto⟧: U.S. Department of Agriculture (1973); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE REZOLVATE (calculate aici)
# =============================================================================
b = N['bursts']
V.put('ex.sp.ann', b['sp500']['sd'] * math.sqrt(b['sp500']['ppy']), 1)
V.put('ex.sp.sq', math.sqrt(b['sp500']['ppy']), 2)
V.put('ex.btc.sq365', math.sqrt(b['btc']['ppy']), 2)
V.put('ex.btc.252', b['btc']['sd'] * math.sqrt(252), 1)
V.put('ex.btc.ratio', 100 * (math.sqrt(b['btc']['ppy'] / 252) - 1), 0)
# 95% interval of the last 21-day volatility of the S&P 500
w21 = N['windows']['21']['last']
for lab, rse in [('n', N['se']['21']['normal']), ('h', N['se']['21']['heavy'])]:
    V.put(f'ex.ci.{lab}.lo', w21 * (1 - 1.96 * rse), 1)
    V.put(f'ex.ci.{lab}.hi', w21 * (1 + 1.96 * rse), 1)
# EWMA update after a -3% day
V.put('ex.ew.s2', 0.94 * 1.0 + 0.06 * 9.0, 2)
V.put('ex.ew.vol', math.sqrt(0.94 + 0.06 * 9.0), 2)
V.put('ex.ew.ann', math.sqrt((0.94 + 0.06 * 9.0) * 252), 1)
V.put('ex.ew.next', 0.94 * (0.94 + 0.06 * 9.0) + 0.06 * 0.0, 2)
# S&P 500 on 16 March 2020: open, high, low, close and the previous close
t = g.ohlc('sp500').loc['2020-03-13':'2020-03-16']
pc, (o_, h_, l_, c_) = t['close'].iloc[0], t[['open', 'high', 'low', 'close']].iloc[1]
for k, x in [('pc', pc), ('o', o_), ('h', h_), ('l', l_), ('c', c_)]:
    V.int(f'm16.{k}', round(x))
oo, uu, dd, ccv = 100 * math.log(o_ / pc), 100 * math.log(h_ / o_), 100 * math.log(l_ / o_), 100 * math.log(c_ / o_)
r16 = 100 * math.log(c_ / pc)
V.put('m16.oo', oo, 2)
V.put('m16.u', uu, 2)
V.put('m16.d', dd, 2)
V.put('m16.cv', ccv, 2)
V.put('m16.r', r16, 2)
V.put('m16.r2', r16 ** 2, 1)
V.put('m16.hl', uu - dd, 2)
V.put('m16.park', (uu - dd) ** 2 / (4 * math.log(2)), 1)
V.put('m16.gk', 0.5 * (uu - dd) ** 2 - (2 * math.log(2) - 1) * ccv ** 2, 1)
V.put('m16.rs', uu * (uu - ccv) + dd * (dd - ccv), 1)
V.put('m16.on', oo ** 2, 1)
V.put('m16.rsall', oo ** 2 + uu * (uu - ccv) + dd * (dd - ccv), 1)
# McLeod-Li step by step: first three autocorrelations of squared S&P 500 returns
x = g.returns('sp500').values
rho2 = g.acf(x ** 2, 3)
Tn = len(x)
terms = [Tn * (Tn + 2) * r ** 2 / (Tn - k) for k, r in enumerate(rho2, 1)]
for i, (r, tt) in enumerate(zip(rho2, terms), 1):
    V.put(f'ml.r{i}', r, 3)
    V.int(f'ml.t{i}', round(tt))
V.int('ml.q', round(sum(terms)))
V.put('ml.crit', 7.8147, 2)
# mixture kurtosis: p = 0.2, sd ratio 2
V.put('mx.e2', 0.8 + 0.2 * 4, 1)
V.put('mx.e4', 0.8 + 0.2 * 16, 1)
# largest daily moves in standard deviations; mean daily return of the S&P 500
zs = [abs(b[k]['max_abs']) / b[k]['sd'] for k in ['sp500', 'dax', 'bet', 'btc']]
V.put('ex.zmin', math.floor(min(zs)), 0)
V.put('ex.zmax', math.floor(max(zs)), 0)
V.put('ex.sp.mean', g.returns('sp500').mean(), 2)
# efficiency range of the range estimators (continuous trading) and the overnight bias of P, GK, RS
effs = [N['eff']['continuous'][e]['eff'] for e in ['park', 'gk', 'rs', 'yz']]
V.put('ef.lo', math.floor(min(effs)), 0)
V.put('ef.hi', math.floor(max(effs)), 0)
V.put('ef.nbias', 100 * np.mean([N['eff']['overnight'][e]['bias'] for e in ['park', 'gk', 'rs']]), 0)
V.put('yz.k21', g.yz_k(21), 2)
V.put('yz.den21', 1.34 + 22 / 20, 2)
# check yourself: Parkinson from H = 102, L = 98
V.put('cy.hl', 100 * math.log(102 / 98), 2)
V.put('cy.vol', math.sqrt((100 * math.log(102 / 98)) ** 2 / (4 * math.log(2))), 2)
V.put('cy.c', 4 * math.log(2), 2)
# QLIKE against MSE for a true variance of 1
V.put('ql.h05', 1 / 0.5 + math.log(0.5), 3)
V.put('ql.h2', 1 / 2 + math.log(2), 3)
V.put('ql.ln05', math.log(0.5), 3)
V.put('ql.ln2', math.log(2), 3)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how large is the risk of an asset today, if we can never observe it directly?',
       '\\textbf{Întrebarea}: cît de mare este riscul unui activ azi, dacă nu îl putem observa niciodată direct?'),
     [T('Chapter 7: returns are almost unpredictable, but their size is not; here we measure that size',
        'Capitolul 7: randamentele sînt aproape imprevizibile, dar mărimea lor nu este; aici măsurăm această mărime'),
      T('Chapter 9 turns the measurements into a model (ARCH and GARCH)', 'Capitolul 9 transformă măsurătorile într-un model (ARCH și GARCH)')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('what volatility is and why it is latent; historical volatility and the choice of the window',
        'ce este volatilitatea și de ce este latentă; volatilitatea istorică și alegerea ferestrei'),
      T('EWMA (RiskMetrics); range-based estimators from open, high, low and close prices',
        'EWMA (RiskMetrics); estimatorii de amplitudine, care folosesc prețurile de deschidere, maxim, minim și închidere'),
      T('realised volatility; volatility clustering and its tests', 'volatilitatea realizată; volatility clustering și testele lui'),
      T('long-run behaviour, the VIX, the leverage effect; how to compare volatility forecasts',
        'comportamentul pe termen lung, indicele VIX, efectul de levier; compararea prognozelor de volatilitate')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Define volatility, annualise it with the actual trading frequency and explain why it is a latent quantity',
      'Definiți volatilitatea, anualizați-o cu frecvența reală de tranzacționare și explicați de ce este o mărime latentă'),
    T('Compute historical and EWMA volatility and choose a window or a decay factor', 'Calculați volatilitatea istorică și volatilitatea EWMA și alegeți o fereastră sau un factor de atenuare'),
    T('Compute the Parkinson, Garman--Klass, Rogers--Satchell and Yang--Zhang estimators and state their assumptions',
      'Calculați estimatorii Parkinson, Garman--Klass, Rogers--Satchell și Yang--Zhang și enunțați ipotezele lor'),
    T('Detect volatility clustering with the ACF of squared returns, the Ljung--Box test and the ARCH-LM test',
      'Detectați volatility clustering cu ACF-ul randamentelor la pătrat, testul Ljung--Box și testul ARCH-LM'),
    T('Compare implied and realised volatility, and evaluate volatility forecasts with MSE and QLIKE',
      'Comparați volatilitatea implicită cu cea realizată și evaluați prognozele de volatilitate cu MSE și QLIKE')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~13 (introduction: volatility and conditional heteroskedasticity)',
       'Manual: \\refFHH, cap.~13 (introducerea: volatilitatea și heteroscedasticitatea condiționată)'),
     [T('exercises with solutions: \\refBHL, Ch.~13', 'exerciții rezolvate: \\refBHL, cap.~13')]),
    T('Volatility and its stylised facts: \\refTsay, Ch.~3.1; a review of volatility forecasting: \\refPG',
      'Volatilitatea și faptele ei stilizate: \\refTsay, cap.~3.1; o sinteză despre prognoza volatilității: \\refPG'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_08}',
       'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_08}'),
     [T('ported from the SFE Quantlets SFEkurgarch (kurtosis of a GARCH process) and SFEvolnonparest (nonparametric volatility)',
        'portate din Quantlet-urile SFE SFEkurgarch (boltirea unui proces GARCH) și SFEvolnonparest (volatilitate neparametrică)')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter8_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter8_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. VOLATILITATEA
# =============================================================================
D.section('What Is Volatility?', 'Volatilitatea: definiție și caracter latent')

D.frame(T('Volatility Comes in Bursts', 'Volatilitatea apare în episoade'), cols(items(
    (T('``Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes\'\' \\refMandelbrot',
       '„Variațiile mari tind să fie urmate de variații mari, de orice semn, iar variațiile mici, de variații mici” \\refMandelbrot'),
     [T('after the crash of 19 October 1987, markets stayed turbulent for weeks', 'după crahul din 19 octombrie 1987, piețele au rămas agitate săptămîni la rînd')]),
    (T('Volatility enters almost every number in finance', 'Volatilitatea intervine în aproape toate mărimile din finanțe'),
     [T('risk measures: VaR and ES (Chapter 10), the Sharpe ratio (Chapter 1)', 'măsurile de risc: VaR și ES (Capitolul 10), raportul Sharpe (Capitolul 1)'),
      T('option prices, margins of exchanges, the size of positions', 'prețurile opțiunilor, marjele cerute de burse, mărimea pozițiilor')]),
    T('This chapter: how to \\textbf{measure} it from prices, and how to tell whether it clusters', 'Acest capitol: \\textbf{măsurarea} ei din prețuri și testarea volatility clustering')),
    ph('amsterdam', T('Amsterdam Stock Exchange, 21 October 1987, two days after the crash', 'Bursa din Amsterdam, 21 octombrie 1987, la două zile după crah'),
       h='0.36\\textheight'), wl='0.52', wr='0.44'))

D.frame(T('Volatility: Definition and Units', 'Volatilitatea: definiție și unități de măsură'), items(
    (T('Daily log return in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$ (Chapter 1)', 'Randamentul logaritmic zilnic în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$ (Capitolul 1)'),
     [T('\\textbf{volatility} = the standard deviation of returns, $\\sigma = \\sqrt{\\mathrm{Var}(r_t)}$', '\\textbf{volatilitatea} = abaterea standard a randamentelor, $\\sigma = \\sqrt{\\mathrm{Var}(r_t)}$'),
      T('it has the units of the return (\\% per day); the variance $\\sigma^2$ has units of \\%$^2$', 'are unitatea de măsură a randamentului (\\% pe zi); varianța $\\sigma^2$ se măsoară în \\%$^2$')]),
    (T('\\textbf{Annualisation}: with $A$ observations per year and uncorrelated returns, $\\sigma_{\\text{year}} = \\sigma_{\\text{day}}\\sqrt{A}$',
       '\\textbf{Anualizarea}: cu $A$ observații pe an și randamente necorelate, $\\sigma_{\\text{an}} = \\sigma_{\\text{zi}}\\sqrt{A}$'),
     [T('$A$ = the actual frequency: about @{b.sp500.ppy} for the S\\&P 500, @{b.bet.ppy} for the BET, @{b.btc.ppy} for Bitcoin (trades every day)',
        '$A$ = frecvența reală: circa @{b.sp500.ppy} pentru S\\&P 500, @{b.bet.ppy} pentru BET, @{b.btc.ppy} pentru Bitcoin (tranzacționat în fiecare zi)'),
      T('the rule $\\sqrt{A}$ is the square-root-of-time rule of Chapter 7: it needs uncorrelated returns', 'regula $\\sqrt{A}$ este regula rădăcinii pătrate a timpului din Capitolul 7: cere randamente necorelate')]),
    (T('Volatility measures dispersion in both directions: a large gain counts as much as a large loss', 'Volatilitatea măsoară dispersia în ambele sensuri: un cîștig mare contează la fel de mult ca o pierdere mare'),
     [T('other measures of uncertainty exist, for example the information entropy of returns \\refPeleA', 'există și alte măsuri ale incertitudinii, de exemplu entropia informațională a randamentelor \\refPeleA')])))

chart(T('Four Markets, One Pattern', 'Patru piețe, același tipar'), 'sfm_ch8_returns_bursts', 'SFM_ch8_returns_volatility', [
    T('Daily log returns in \\%: S\\&P 500 and DAX since 1990, BET since @{b.bet.y0}, Bitcoin since @{b.btc.y0}; last day @{end}',
      'Randamente logaritmice zilnice în \\%: S\\&P 500 și DAX din 1990, BET din @{b.bet.y0}, Bitcoin din @{b.btc.y0}; ultima zi: @{end}'),
    T('Quiet years alternate with turbulent months in all four series', 'Anii liniștiți alternează cu luni agitate în toate cele patru serii')],
    h='0.66\\textheight')

D.frame(T('Interpretation: Volatility in Four Markets', 'Interpretarea volatilității pe patru piețe'), table(
    'lrrrr', T('& \\textbf{Annual volatility} & \\textbf{Largest daily move} & \\textbf{Date} & \\textbf{Excess kurtosis}',
               '& \\textbf{Volatilitate anuală} & \\textbf{Cea mai mare variație zilnică} & \\textbf{Data} & \\textbf{Excesul de boltire}'),
    [T(*[f'{NAMES[k]} & @{{b.{k}.vol}}\\% & ${{@{{b.{k}.max}}}}\\%$ & @{{b.{k}.maxd}} & @{{b.{k}.kurt}}'] * 2) for k in ['sp500', 'dax', 'bet', 'btc']],
    size='footnotesize') + items(
    T('Bitcoin is about @{ex.btc.ratio}\\% more volatile per year than its daily figure suggests if one wrongly uses $\\sqrt{252}$: it trades 365 days a year',
      'Bitcoin este cu circa @{ex.btc.ratio}\\% mai volatil pe an decît ar rezulta dacă s-ar folosi greșit $\\sqrt{252}$: se tranzacționează 365 de zile pe an'),
    T('The largest moves are @{ex.zmin} to @{ex.zmax} standard deviations: practically impossible under the Normal distribution (Chapter 2)',
      'Cele mai mari variații reprezintă @{ex.zmin}--@{ex.zmax} abateri standard: valori practic imposibile în cazul distribuției Normale (Capitolul 2)'),
    T('Excess kurtosis far above 0 (the Normal value): heavy tails; Section 6 shows that clustering is one cause',
      'Excesul de boltire este mult peste 0 (valoarea pentru distribuția Normală): cozi groase; secțiunea 6 arată că volatility clustering este una dintre cauze')) + ql('SFM_ch8_returns_volatility'), 'footnotesize')

D.frame(T('Volatility Is Latent', 'Volatilitatea este latentă'), items(
    (T('Model: the return is a constant mean plus a shock whose size changes every day: $r_t = \\mu + \\sigma_t z_t$', 'Modelul: randamentul este o medie constantă plus un șoc a cărui mărime se schimbă zilnic: $r_t = \\mu + \\sigma_t z_t$'),
     [T('$\\mu$: the mean return; $z_t$: i.i.d.\\ shocks with mean 0 and variance 1', '$\\mu$: randamentul mediu; $z_t$: șocuri i.i.d., de medie 0 și varianță 1'),
      T('$\\sigma_t$ is the \\textbf{conditional volatility}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$, with $\\mathcal{F}_{t-1}$ the information at the close of day $t-1$',
        '$\\sigma_t$ este \\textbf{volatilitatea condiționată}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$, cu $\\mathcal{F}_{t-1}$ informația de la închiderea zilei $t-1$'),
      T('the \\textbf{unconditional volatility} $\\sigma = \\sqrt{\\mathrm{Var}(r_t)}$ is its long-run average level', '\\textbf{volatilitatea necondiționată} $\\sigma = \\sqrt{\\mathrm{Var}(r_t)}$ este nivelul ei mediu pe termen lung')]),
    (T('\\textbf{Latent}: we never observe $\\sigma_t$, only one return per day', '\\textbf{Latentă}: nu observăm niciodată $\\sigma_t$, ci doar un randament pe zi'),
     [T('$r_t^2$ is an unbiased but very noisy measure of $\\sigma_t^2$ (with $\\mu = 0$): $E[r_t^2 \\mid \\mathcal{F}_{t-1}] = \\sigma_t^2$',
        '$r_t^2$ este o măsură nedeplasată, dar foarte zgomotoasă, a lui $\\sigma_t^2$ (cu $\\mu = 0$): $E[r_t^2 \\mid \\mathcal{F}_{t-1}] = \\sigma_t^2$'),
      T('a quiet day can happen in a turbulent period, and the other way round', 'o zi liniștită poate apărea într-o perioadă agitată, și invers')]),
    (T('Three ways to get more information about $\\sigma_t$', 'Trei moduri de a obține mai multă informație despre $\\sigma_t$'),
     [T('average many days: historical volatility, EWMA', 'media pe multe zile: volatilitatea istorică, EWMA'),
      T('use the whole path of the day: the high and low prices', 'întreaga traiectorie a zilei: prețurile maxim și minim'),
      T('use intraday prices: realised volatility', 'prețurile din timpul zilei: volatilitatea realizată')])))

D.frame(T('Worked Example: Annualising Volatility', 'Exemplu rezolvat: anualizarea volatilității'), items(
    (T('S\\&P 500 since 1990: daily standard deviation $\\hat\\sigma = @{b.sp500.sd}\\%$, $A = @{b.sp500.ppy}$ observations per year',
       'S\\&P 500 din 1990: abaterea standard zilnică $\\hat\\sigma = @{b.sp500.sd}\\%$, $A = @{b.sp500.ppy}$ de observații pe an'),
     [T('$\\hat\\sigma_{\\text{year}} = @{b.sp500.sd} \\times \\sqrt{@{b.sp500.ppy}} = @{b.sp500.sd} \\times @{ex.sp.sq} = @{ex.sp.ann}\\%$',
        '$\\hat\\sigma_{\\text{an}} = @{b.sp500.sd} \\times \\sqrt{@{b.sp500.ppy}} = @{b.sp500.sd} \\times @{ex.sp.sq} = @{ex.sp.ann}\\%$')]),
    (T('Bitcoin since @{b.btc.y0}: $\\hat\\sigma = @{b.btc.sd}\\%$ per day, 7 days a week', 'Bitcoin din @{b.btc.y0}: $\\hat\\sigma = @{b.btc.sd}\\%$ pe zi, 7 zile pe săptămînă'),
     [T('correct: $@{b.btc.sd} \\times \\sqrt{@{b.btc.ppy}} = @{b.btc.sd} \\times @{ex.btc.sq365} = @{b.btc.vol}\\%$', 'corect: $@{b.btc.sd} \\times \\sqrt{@{b.btc.ppy}} = @{b.btc.sd} \\times @{ex.btc.sq365} = @{b.btc.vol}\\%$'),
      T('wrong, with the equity convention: $@{b.btc.sd} \\times \\sqrt{252} = @{ex.btc.252}\\%$', 'greșit, cu convenția de la acțiuni: $@{b.btc.sd} \\times \\sqrt{252} = @{ex.btc.252}\\%$')]),
    T('Comparing assets: annualise each with its own calendar, then compare', 'Pentru a compara active: anualizați fiecare serie cu propriul calendar, apoi comparați')))

D.recap(('What Is Volatility?', 'volatilitatea și caracterul ei latent'), [
    T('Volatility = standard deviation of returns; annualise with $\\sqrt{A}$, $A$ the actual number of observations per year',
      'Volatilitatea = abaterea standard a randamentelor; se anualizează cu $\\sqrt{A}$, unde $A$ este numărul real de observații pe an'),
    T('Conditional volatility $\\sigma_t$ changes over time and is never observed: we can only estimate it', 'Volatilitatea condiționată $\\sigma_t$ variază în timp și nu este observată niciodată: o putem doar estima'),
    T('All markets show quiet and turbulent periods', 'Toate piețele au perioade liniștite și perioade agitate')])

# =============================================================================
# 2. VOLATILITATEA ISTORICĂ
# =============================================================================
D.section('Historical Volatility and the Choice of the Window', 'Volatilitatea istorică și alegerea ferestrei')

D.frame(T('Historical (Rolling) Volatility', 'Volatilitatea istorică (pe fereastră mobilă)'), items(
    (T('\\textbf{Historical volatility} on a window of the last $n$ days:', '\\textbf{Volatilitatea istorică} pe o fereastră cu ultimele $n$ zile:'),
     [T('$\\hat\\sigma_t = \\sqrt{\\dfrac{1}{n-1}\\sum_{j=0}^{n-1}(r_{t-j} - \\bar r_t)^2}$, with $\\bar r_t$ the mean of the same $n$ returns',
        '$\\hat\\sigma_t = \\sqrt{\\dfrac{1}{n-1}\\sum_{j=0}^{n-1}(r_{t-j} - \\bar r_t)^2}$, cu $\\bar r_t$ media acelorași $n$ randamente'),
      T('$j$: how many days back the return lies ($j = 0$ is day $t$); $\\hat\\sigma_t$: the estimate at the close of day $t$',
        '$j$: cîte zile în urmă se află randamentul ($j = 0$ este ziua $t$); $\\hat\\sigma_t$: estimarea la închiderea zilei $t$'),
      T('\\textbf{rolling window}: each day the newest return enters and the oldest leaves', '\\textbf{fereastră mobilă}: în fiecare zi intră cel mai nou randament și iese cel mai vechi')]),
    (T('Zero-mean version: $\\hat\\sigma_t^2 = \\frac{1}{n}\\sum_{j=0}^{n-1} r_{t-j}^2$', 'Varianta cu media zero: $\\hat\\sigma_t^2 = \\frac{1}{n}\\sum_{j=0}^{n-1} r_{t-j}^2$'),
     [T('for daily data the mean is tiny compared with $\\sigma$ ($@{ex.sp.mean}\\%$ against $@{b.sp500.sd}\\%$ for the S\\&P 500); setting it to 0 removes a noisy estimate',
        'pentru datele zilnice media este foarte mică în raport cu $\\sigma$ ($@{ex.sp.mean}\\%$ față de $@{b.sp500.sd}\\%$ la S\\&P 500); dacă o fixăm la 0, eliminăm o estimare zgomotoasă')]),
    (T('Usual windows: $n = 21$ (a month), $63$ (a quarter), $252$ (a year) of trading days', 'Ferestre uzuale: $n = 21$ (o lună), $63$ (un trimestru), $252$ (un an) de zile de tranzacționare'),
     [T('each $\\hat\\sigma_t$ is annualised with $\\sqrt{A}$ and plotted at the last day of its window', 'fiecare $\\hat\\sigma_t$ se anualizează cu $\\sqrt{A}$ și se reprezintă în ultima zi a ferestrei')])))

chart(T('S\\&P 500: Three Windows, Three Answers', 'S\\&P 500: trei ferestre, trei răspunsuri'), 'sfm_ch8_rolling_windows', 'SFM_ch8_returns_volatility', [
    T('Rolling volatility of the S\\&P 500, annualised; dashed line: the whole-sample volatility, @{w.whole}\\%',
      'Volatilitatea S\\&P 500 pe ferestre mobile, anualizată; linia punctată: volatilitatea pe întregul eșantion, @{w.whole}\\%'),
    T('\\textbf{Question for the room}: which window would you trust on the day after a crash?', '\\textbf{Întrebare pentru sală}: în ce fereastră ați avea încredere în ziua de după un crah?')],
    h='0.56\\textheight')

D.frame(T('Interpretation: the Window Trade-Off', 'Interpretarea compromisului dintre ferestre'), items(
    (T('Short window ($n = 21$): reacts fast, but is noisy', 'Fereastra scurtă ($n = 21$): reacționează repede, dar este zgomotoasă'),
     [T('maximum @{w.21.max}\\% on @{w.21.maxd}; the typical daily change of $\\ln\\hat\\sigma_t$ is @{w.21.sdc}\\%',
        'maximul @{w.21.max}\\% pe @{w.21.maxd}; variația zilnică tipică a lui $\\ln\\hat\\sigma_t$ este @{w.21.sdc}\\%')]),
    (T('Long window ($n = 252$): smooth, but slow', 'Fereastra lungă ($n = 252$): netedă, dar lentă'),
     [T('maximum only @{w.252.max}\\%, reached on @{w.252.maxd}, months after the peak of the 2008 crisis; daily change @{w.252.sdc}\\%',
        'maximul doar @{w.252.max}\\%, atins pe @{w.252.maxd}, la cîteva luni după vîrful crizei din 2008; variația zilnică @{w.252.sdc}\\%')]),
    (T('\\textbf{Answer}: a short window, or EWMA (next section), because a long window still averages the calm days before the crash',
       '\\textbf{Răspuns}: o fereastră scurtă sau EWMA (secțiunea următoare), deoarece media pe o fereastră lungă include încă zilele liniștite dinaintea crahului'),
     [T('for capital planning, a long window is preferred: it changes slowly', 'pentru planificarea capitalului se preferă o fereastră lungă: se schimbă lent')]),
    T('At the end of the sample: @{w.21.last}\\% (21 days), @{w.63.last}\\% (63 days), @{w.252.last}\\% (252 days)',
      'La sfîrșitul eșantionului: @{w.21.last}\\% (21 de zile), @{w.63.last}\\% (63 de zile), @{w.252.last}\\% (252 de zile)')) + ql('SFM_ch8_returns_volatility'))

D.frame(T('How Precise Is a Sample Volatility?', 'Precizia unei volatilități de selecție'), items(
    (T('With $n$ i.i.d.\\ Normal returns, the relative standard error of $\\hat\\sigma$ is $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx 1/\\sqrt{2n}$',
       'Cu $n$ randamente Normale i.i.d., eroarea standard relativă a lui $\\hat\\sigma$ este $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx 1/\\sqrt{2n}$'),
     [T('$\\mathrm{SE}(\\hat\\sigma)$: the standard deviation of the estimate across samples; divided by $\\sigma$ it is a relative error',
        '$\\mathrm{SE}(\\hat\\sigma)$: abaterea standard a estimării de la un eșantion la altul; împărțită la $\\sigma$, este o eroare relativă'),
      T('$n = 21$: @{se.21.n}\\%; $n = 63$: @{se.63.n}\\%; $n = 252$: @{se.252.n}\\%', '$n = 21$: @{se.21.n}\\%; $n = 63$: @{se.63.n}\\%; $n = 252$: @{se.252.n}\\%')]),
    (T('With heavy tails (kurtosis $\\kappa$): $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx \\sqrt{(\\kappa - 1)/(4n)}$ (delta method)',
       'Cu cozi groase (coeficient de boltire $\\kappa$): $\\mathrm{SE}(\\hat\\sigma)/\\sigma \\approx \\sqrt{(\\kappa - 1)/(4n)}$ (metoda delta)'),
     [T('S\\&P 500: $\\kappa = @{se.kappa}$, so $n = 21$: @{se.21.h}\\%; $n = 252$: @{se.252.h}\\%', 'S\\&P 500: $\\kappa = @{se.kappa}$, deci $n = 21$: @{se.21.h}\\%; $n = 252$: @{se.252.h}\\%'),
      T('for $\\kappa = 3$ (Normal) the two formulas coincide', 'pentru $\\kappa = 3$ (distribuția Normală) cele două formule coincid')]),
    (T('\\textbf{Worked example}: the last 21-day volatility of the S\\&P 500 is @{w.21.last}\\%', '\\textbf{Exemplu rezolvat}: ultima volatilitate pe 21 de zile a S\\&P 500 este @{w.21.last}\\%'),
     [T('approximate 95\\% interval, Normal returns: $[@{ex.ci.n.lo}\\%; @{ex.ci.n.hi}\\%]$; with the observed kurtosis: $[@{ex.ci.h.lo}\\%; @{ex.ci.h.hi}\\%]$',
        'interval aproximativ de 95\\%, randamente Normale: $[@{ex.ci.n.lo}\\%; @{ex.ci.n.hi}\\%]$; cu boltirea observată: $[@{ex.ci.h.lo}\\%; @{ex.ci.h.hi}\\%]$')])))

D.recap(('Historical Volatility', 'volatilitatea istorică'), [
    T('Rolling standard deviation on $n$ days; daily data: the mean can be set to zero', 'Abaterea standard pe o fereastră mobilă de $n$ zile; pentru date zilnice media poate fi fixată la zero'),
    T('Short windows react fast but are noisy; long windows are smooth but slow', 'Ferestrele scurte reacționează repede, dar sînt zgomotoase; cele lungi sînt netede, dar lente'),
    T('A 21-day volatility is uncertain by $\\pm$@{se.21.n}\\% or more: one decimal is already too much precision', 'O volatilitate pe 21 de zile are o incertitudine de $\\pm$@{se.21.n}\\% sau mai mult: chiar și o zecimală sugerează o precizie pe care estimarea nu o are')])

# =============================================================================
# 3. EWMA ȘI RISKMETRICS
# =============================================================================
D.section('EWMA and RiskMetrics', 'EWMA și RiskMetrics')

D.frame(T('1994: RiskMetrics Makes Volatility a Daily Number', '1994: RiskMetrics transformă volatilitatea într-un indicator zilnic'), cols(items(
    (T('J.P. Morgan launched RiskMetrics in October 1994 and published its methods and data free of charge \\refRM',
       'J.P. Morgan a lansat RiskMetrics în octombrie 1994 și a publicat gratuit metodele și datele \\refRM'),
     [T('goal: one common yardstick for market risk across banks', 'scopul: un etalon comun pentru riscul de piață, valabil pentru toate băncile'),
      T('volatilities and correlations of more than 480 series, updated every day', 'volatilitățile și corelațiile a peste 480 de serii, actualizate zilnic')]),
    (T('The method: an exponentially weighted moving average (EWMA) of squared returns', 'Metoda: o medie mobilă ponderată exponențial (EWMA) a randamentelor la pătrat'),
     [T('one parameter, the \\textbf{decay factor} $\\lambda$: $0.94$ for daily forecasts, $0.97$ for monthly ones',
        'un singur parametru, \\textbf{factorul de atenuare} $\\lambda$: $0{,}94$ pentru prognozele zilnice, $0{,}97$ pentru cele lunare'),
      T('for each series, $\\lambda$ minimises the root mean squared error of its variance forecasts; 0.94 is the weighted average over all series',
        'pentru fiecare serie, $\\lambda$ minimizează rădăcina erorii pătratice medii a prognozelor de varianță; 0,94 este media ponderată pe toate seriile')])),
    ph('jpm', T('The J.P. Morgan building at 23 Wall Street, New York', 'Clădirea J.P. Morgan de la 23 Wall Street, New York'), h='0.38\\textheight'), wl='0.56', wr='0.40'))

D.frame(T('The EWMA Estimator', 'Estimatorul EWMA'), items(
    (T('\\textbf{EWMA} recursion (zero mean): $\\sigma_t^2 = \\lambda\\,\\sigma_{t-1}^2 + (1-\\lambda)\\,r_{t-1}^2$, \\ $0 < \\lambda < 1$',
       'Recurența \\textbf{EWMA} (media zero): $\\sigma_t^2 = \\lambda\\,\\sigma_{t-1}^2 + (1-\\lambda)\\,r_{t-1}^2$, \\ $0 < \\lambda < 1$'),
     [T('today\'s variance = a weighted average of yesterday\'s variance and yesterday\'s squared return', 'varianța de azi = o medie ponderată a varianței de ieri și a randamentului de ieri la pătrat'),
      T('$\\lambda$: the weight of the past; a $\\lambda$ close to 1 gives a smooth, slowly reacting estimate', '$\\lambda$: ponderea trecutului; un $\\lambda$ apropiat de 1 dă o estimare netedă, care reacționează lent'),
      T('$\\sigma_t^2$ uses returns up to day $t-1$: it is the forecast for day $t$', '$\\sigma_t^2$ folosește randamentele pînă în ziua $t-1$: este prognoza pentru ziua $t$'),
      T('start: $\\sigma_1^2$ = the mean of the first squared returns; its effect fades after a few months', 'valoarea inițială: $\\sigma_1^2$ = media primelor randamente la pătrat; efectul ei dispare după cîteva luni')]),
    (T('Unrolling the recursion: $\\sigma_t^2 = (1-\\lambda)\\sum_{j \\ge 1}\\lambda^{j-1} r_{t-j}^2$', 'Prin dezvoltarea recurenței: $\\sigma_t^2 = (1-\\lambda)\\sum_{j \\ge 1}\\lambda^{j-1} r_{t-j}^2$'),
     [T('weights $(1-\\lambda)\\lambda^{j-1}$ decline geometrically and sum to 1', 'ponderile $(1-\\lambda)\\lambda^{j-1}$ scad geometric și au suma 1'),
      T('\\textbf{half-life}: the lag at which the weight halves, $\\ln 0.5/\\ln\\lambda$', '\\textbf{timpul de înjumătățire}: lagul la care ponderea se înjumătățește, $\\ln 0{,}5/\\ln\\lambda$')]),
    T('A rolling window gives weight $1/n$ to each of the last $n$ days and 0 to all older days', 'O fereastră mobilă dă ponderea $1/n$ fiecăreia dintre ultimele $n$ zile și 0 tuturor zilelor mai vechi')))

chart(T('EWMA Weights against a Rolling Window', 'Ponderile EWMA față de ponderile unei ferestre mobile'), 'sfm_ch8_ewma_weights', 'SFM_ch8_ewma', [
    T('$\\lambda = 0.94$: half-life @{ew.94.hl} days, mean lag $1/(1-\\lambda) = @{ew.94.lag}$ days, 99\\% of the weight in the last @{ew.94.d99} days',
      '$\\lambda = 0{,}94$: timp de înjumătățire @{ew.94.hl} zile, lag mediu $1/(1-\\lambda) = @{ew.94.lag}$ zile, 99\\% din pondere în ultimele @{ew.94.d99} de zile'),
    T('$\\lambda = 0.97$: half-life @{ew.97.hl} days, 99\\% of the weight in the last @{ew.97.d99} days', '$\\lambda = 0{,}97$: timp de înjumătățire @{ew.97.hl} zile, 99\\% din pondere în ultimele @{ew.97.d99} de zile')],
    h='0.54\\textheight')

D.frame(T('Worked Example: One EWMA Update', 'Exemplu rezolvat: o actualizare EWMA'), items(
    (T('Yesterday\'s EWMA variance: $\\sigma_{t-1}^2 = 1.00$ (\\%$^2$), a daily volatility of 1\\%; yesterday\'s return: $r_{t-1} = -3\\%$',
       'Varianța EWMA de ieri: $\\sigma_{t-1}^2 = 1{,}00$ (\\%$^2$), adică o volatilitate zilnică de 1\\%; randamentul de ieri: $r_{t-1} = -3\\%$'),
     [T('step 1: $\\lambda\\sigma_{t-1}^2 = 0.94 \\times 1.00 = 0.94$', 'pasul 1: $\\lambda\\sigma_{t-1}^2 = 0{,}94 \\times 1{,}00 = 0{,}94$'),
      T('step 2: $(1-\\lambda)r_{t-1}^2 = 0.06 \\times 9 = 0.54$', 'pasul 2: $(1-\\lambda)r_{t-1}^2 = 0{,}06 \\times 9 = 0{,}54$'),
      T('step 3: $\\sigma_t^2 = 0.94 + 0.54 = @{ex.ew.s2}$, so $\\sigma_t = @{ex.ew.vol}\\%$ per day, $@{ex.ew.vol} \\times \\sqrt{252} = @{ex.ew.ann}\\%$ per year',
        'pasul 3: $\\sigma_t^2 = 0{,}94 + 0{,}54 = @{ex.ew.s2}$, deci $\\sigma_t = @{ex.ew.vol}\\%$ pe zi, $@{ex.ew.vol} \\times \\sqrt{252} = @{ex.ew.ann}\\%$ pe an')]),
    (T('If today is quiet ($r_t = 0$): $\\sigma_{t+1}^2 = 0.94 \\times @{ex.ew.s2} = @{ex.ew.next}$', 'Dacă ziua de azi este liniștită ($r_t = 0$): $\\sigma_{t+1}^2 = 0{,}94 \\times @{ex.ew.s2} = @{ex.ew.next}$'),
     [T('the shock fades by 6\\% per day; it never drops out abruptly', 'șocul se atenuează cu 6\\% pe zi; nu dispare niciodată brusc')]),
    T('A 63-day window would add $9/63 = 0.14$ to the variance today and remove this contribution at once, 63 days later', 'O fereastră de 63 de zile ar adăuga azi $9/63 = 0{,}14$ la varianță și ar elimina dintr-odată această contribuție după 63 de zile')))

chart(T('The Ghost Effect: March--June 2020', 'Ghost effect: martie--iunie 2020'), 'sfm_ch8_ghost', 'SFM_ch8_ewma', [
    T('S\\&P 500: 63-day rolling volatility and EWMA volatility ($\\lambda = 0.94$), annualised; bars: $|r_t|$ on the same annual scale',
      'S\\&P 500: volatilitatea pe fereastra mobilă de 63 de zile și volatilitatea EWMA ($\\lambda = 0{,}94$), anualizate; barele: $|r_t|$ pe aceeași scală anuală'),
    T('\\textbf{Question for the room}: why does the blue line fall sharply in mid-June 2020, on a calm day?', '\\textbf{Întrebare pentru sală}: de ce scade brusc linia albastră la mijlocul lui iunie 2020, într-o zi liniștită?')],
    h='0.54\\textheight')

D.frame(T('Interpretation: Ghosts and Fading Memory', 'Interpretarea: ghost effect și atenuarea treptată a memoriei'), items(
    (T('\\textbf{Answer}: on @{gh.dropd} the return of @{gh.left} ($@{gh.leftr}\\%$) left the 63-day window', '\\textbf{Răspuns}: pe @{gh.dropd}, randamentul din @{gh.left} ($@{gh.leftr}\\%$) a ieșit din fereastra de 63 de zile'),
     [T('the rolling volatility fell from @{gh.before}\\% to @{gh.after}\\% in one day, with no news at all', 'volatilitatea pe fereastra mobilă a scăzut de la @{gh.before}\\% la @{gh.after}\\% într-o singură zi, fără nicio știre'),
      T('this artefact is the \\textbf{ghost effect}: an old shock leaves the window abruptly', 'acest artefact este \\textbf{ghost effect}: un șoc vechi iese brusc din fereastră')]),
    (T('EWMA reacts faster and forgets smoothly', 'EWMA reacționează mai repede și uită treptat'),
     [T('EWMA peak: @{gh.epeak}\\% on @{gh.epeakd}; 63-day peak: @{gh.rpeak}\\% only on @{gh.rpeakd}', 'vîrful EWMA: @{gh.epeak}\\% pe @{gh.epeakd}; vîrful pe 63 de zile: @{gh.rpeak}\\% abia pe @{gh.rpeakd}'),
      T('on the ghost day EWMA changed by only $@{gh.edrop}$ percentage points', 'în ziua ghost effect, EWMA s-a modificat cu doar $@{gh.edrop}$ puncte procentuale')]),
    T('Price of the speed: EWMA is noisier than a long window and has no long-run level to return to', 'Costul reacției rapide: EWMA este mai zgomotos decît o fereastră lungă și nu are un nivel pe termen lung spre care să revină')) + ql('SFM_ch8_ewma'))

D.frame(T('EWMA and GARCH: a Pointer to Chapter 9', 'EWMA și GARCH: trimitere la Capitolul 9'), items(
    (T('\\textbf{ARCH} \\refEngle and \\textbf{GARCH} \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha\\, r_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$',
       '\\textbf{ARCH} \\refEngle și \\textbf{GARCH} \\refBollerslev: $\\sigma_t^2 = \\omega + \\alpha\\, r_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$'),
     [T('$\\omega > 0$: a constant; $\\alpha$: the reaction to yesterday\'s squared return; $\\beta$: the weight of yesterday\'s variance',
        '$\\omega > 0$: o constantă; $\\alpha$: reacția la randamentul de ieri la pătrat; $\\beta$: ponderea varianței de ieri'),
      T('EWMA is the special case $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$, so $\\alpha + \\beta = 1$ (``integrated\'\' GARCH, \\textbf{IGARCH})',
        'EWMA este cazul particular $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$, deci $\\alpha + \\beta = 1$ (GARCH „integrat”, \\textbf{IGARCH})')]),
    (T('Consequences for forecasts made at day $t$ for day $t + h$', 'Consecințe pentru prognozele făcute în ziua $t$ pentru ziua $t + h$'),
     [T('EWMA: flat, $E_t[\\sigma_{t+h}^2] = \\sigma_{t+1}^2$ for every $h$ ($E_t$: expectation given the information of day $t$): today\'s level persists for ever', 'EWMA: constantă, $E_t[\\sigma_{t+h}^2] = \\sigma_{t+1}^2$ pentru orice $h$ ($E_t$: media condiționată de informația din ziua $t$): nivelul de azi persistă la nesfîrșit'),
      T('GARCH with $\\alpha + \\beta < 1$: pulled towards the long-run variance $\\omega/(1 - \\alpha - \\beta)$', 'GARCH cu $\\alpha + \\beta < 1$: atrasă spre varianța pe termen lung $\\omega/(1 - \\alpha - \\beta)$')]),
    T('Chapter 9 estimates $\\omega$, $\\alpha$, $\\beta$ by maximum likelihood; here $\\lambda$ is fixed or chosen by a forecast criterion (Section 8)',
      'Capitolul 9 estimează $\\omega$, $\\alpha$, $\\beta$ prin verosimilitate maximă; aici $\\lambda$ este fixat sau ales după un criteriu de prognoză (secțiunea 8)')))

D.recap(('EWMA and RiskMetrics', 'EWMA și RiskMetrics'), [
    T('$\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1-\\lambda)r_{t-1}^2$; RiskMetrics: $\\lambda = 0.94$ for daily data', '$\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1-\\lambda)r_{t-1}^2$; RiskMetrics: $\\lambda = 0{,}94$ pentru date zilnice'),
    T('Geometric weights: half-life @{ew.94.hl} days for $\\lambda = 0.94$; no ghost effect', 'Ponderi geometrice: timp de înjumătățire @{ew.94.hl} zile pentru $\\lambda = 0{,}94$; fără ghost effect'),
    T('EWMA = IGARCH(1,1) without a constant: flat forecasts, no mean reversion', 'EWMA = IGARCH(1,1) fără constantă: prognoze constante, fără revenire la medie')])

# =============================================================================
# 4. ESTIMATORI DE AMPLITUDINE
# =============================================================================
D.section('Range-Based Estimators', 'Estimatori de amplitudine (range-based)')

D.frame(T('Open, High, Low and Close Prices', 'Prețurile de deschidere, maxim, minim și închidere'), items(
    (T('\\textbf{OHLC} data: for each day $t$, the open $O_t$, the high $H_t$, the low $L_t$ and the close $C_t$', 'Datele \\textbf{OHLC}: pentru fiecare zi $t$, deschiderea $O_t$, maximul $H_t$, minimul $L_t$ și închiderea $C_t$'),
     [T('the close-to-close return uses 2 prices a day; $H_t$ and $L_t$ carry information about the whole path of the day',
        'randamentul închidere--închidere folosește 2 prețuri pe zi; $H_t$ și $L_t$ conțin informație despre întreaga traiectorie a zilei')]),
    (T('Notation (log differences in \\%), each from the open of day $t$:', 'Notații (diferențe logaritmice în \\%), fiecare față de deschiderea zilei $t$:'),
     [T('$u_t = \\ln(H_t/O_t)$, $d_t = \\ln(L_t/O_t)$, $c_t = \\ln(C_t/O_t)$; so $d_t \\le 0 \\le u_t$ and $d_t \\le c_t \\le u_t$',
        '$u_t = \\ln(H_t/O_t)$, $d_t = \\ln(L_t/O_t)$, $c_t = \\ln(C_t/O_t)$; deci $d_t \\le 0 \\le u_t$ și $d_t \\le c_t \\le u_t$'),
      T('\\textbf{overnight} return $o_t = \\ln(O_t/C_{t-1})$; the daily return is $r_t = o_t + c_t$', 'randamentul \\textbf{overnight} $o_t = \\ln(O_t/C_{t-1})$; randamentul zilnic este $r_t = o_t + c_t$'),
      T('the \\textbf{range} $u_t - d_t = \\ln(H_t/L_t)$', '\\textbf{amplitudinea} (range) $u_t - d_t = \\ln(H_t/L_t)$')]),
    T('Data: S\\&P 500 since 2008, DAX since 2006, Bitcoin since 2014, BVB stocks since 2010; the BET file has closes only',
      'Date: S\\&P 500 din 2008, DAX din 2006, Bitcoin din 2014, acțiuni BVB din 2010; fișierul BET conține doar închideri')))

chart(T('One Trading Day: What OHLC Records', 'O zi de tranzacționare în datele OHLC'), 'sfm_ch8_ohlc_path', 'SFM_ch8_range_estimators', [
    T('Simulated log price (Brownian motion, 390 one-minute steps) after an overnight gap of $0.35\\%$: the close keeps only the end point; high and low summarise the path',
      'Preț logaritmic simulat (mișcare browniană, 390 de pași de un minut) după un salt overnight de $0{,}35\\%$: închiderea păstrează doar punctul final; maximul și minimul rezumă traiectoria'),
    T('Two days with the same close can have very different ranges, and so very different risk', 'Două zile cu aceeași închidere pot avea amplitudini foarte diferite, deci riscuri foarte diferite')],
    h='0.54\\textheight')

D.frame(T('The Parkinson Estimator (1980)', 'Estimatorul Parkinson (1980)'), items(
    (T('For a Brownian motion with no drift and variance $\\sigma^2$ per day: $E[(u_t - d_t)^2] = 4\\ln 2\\;\\sigma^2$ \\refParkinson',
       'Pentru o mișcare browniană fără tendință (drift), cu varianța $\\sigma^2$ pe zi: $E[(u_t - d_t)^2] = 4\\ln 2\\;\\sigma^2$ \\refParkinson'),
     [T('in words: the squared log range, rescaled so that on average it equals the variance', 'în cuvinte: pătratul amplitudinii logaritmice, rescalat astfel încît, în medie, să fie egal cu varianța'),
      T('one day: $\\hat\\sigma_{P,t}^2 = \\dfrac{(u_t - d_t)^2}{4\\ln 2}$; on $n$ days: the mean of these values', 'o zi: $\\hat\\sigma_{P,t}^2 = \\dfrac{(u_t - d_t)^2}{4\\ln 2}$; pe $n$ zile: media acestor valori'),
      T('$4\\ln 2 \\approx 2.77$: the range is on average larger than $\\sigma$', '$4\\ln 2 \\approx 2{,}77$: amplitudinea este, în medie, mai mare decît $\\sigma$')]),
    (T('Assumptions', 'Ipoteze'),
     [T('continuous trading during the day, so the observed high and low are the true extremes', 'tranzacționare continuă în timpul zilei, deci maximul și minimul observate sînt extremele reale'),
      T('no drift and no overnight jump: the range only sees the trading session', 'fără tendință și fără salt overnight: amplitudinea reflectă doar ședința de tranzacționare')]),
    (T('Much more precise than $r_t^2$: a simulation later in this section measures by how much', 'Mult mai precis decît $r_t^2$: o simulare din această secțiune măsoară cu cît'),
     [T('the log range is also used to estimate stochastic volatility models \\refABD', 'logaritmul amplitudinii se folosește și la estimarea modelelor de volatilitate stochastică \\refABD')])))

D.frame(T('Garman--Klass (1980) and Rogers--Satchell (1991)', 'Garman--Klass (1980) și Rogers--Satchell (1991)'), items(
    (T('\\textbf{Garman--Klass} \\refGK: adds the open-to-close return', '\\textbf{Garman--Klass} \\refGK: adaugă randamentul deschidere--închidere'),
     [T('$\\hat\\sigma_{GK,t}^2 = 0.5\\,(u_t - d_t)^2 - (2\\ln 2 - 1)\\,c_t^2$', '$\\hat\\sigma_{GK,t}^2 = 0{,}5\\,(u_t - d_t)^2 - (2\\ln 2 - 1)\\,c_t^2$'),
      T('the weights make the estimator unbiased and nearly of minimum variance among such quadratic forms, under no drift',
        'ponderile fac estimatorul nedeplasat și aproape de varianță minimă printre astfel de forme pătratice, în absența tendinței')]),
    (T('\\textbf{Rogers--Satchell} \\refRS: unbiased for any drift $\\mu$', '\\textbf{Rogers--Satchell} \\refRS: nedeplasat pentru orice tendință $\\mu$'),
     [T('$\\hat\\sigma_{RS,t}^2 = u_t(u_t - c_t) + d_t(d_t - c_t)$', '$\\hat\\sigma_{RS,t}^2 = u_t(u_t - c_t) + d_t(d_t - c_t)$'),
      T('useful for assets with a strong trend over the window (for example, Bitcoin in a strong rally)', 'util pentru active cu o tendință puternică pe fereastră (de exemplu, Bitcoin într-o perioadă de creștere puternică)')]),
    T('Both still ignore the overnight return $o_t$: they measure the variance of the trading session only',
      'Ambele ignoră însă randamentul overnight $o_t$: măsoară doar varianța ședinței de tranzacționare')))

D.frame(T('Yang--Zhang (2000): Adding the Night', 'Yang--Zhang (2000): adăugarea nopții'), items(
    (T('On a window of $n$ days \\refYZ:', 'Pe o fereastră de $n$ zile \\refYZ:'),
     [T('$\\hat\\sigma_{YZ}^2 = \\hat\\sigma_O^2 + k\\,\\hat\\sigma_C^2 + (1-k)\\,\\hat\\sigma_{RS}^2$, \\quad $k = \\dfrac{0.34}{1.34 + (n+1)/(n-1)}$',
        '$\\hat\\sigma_{YZ}^2 = \\hat\\sigma_O^2 + k\\,\\hat\\sigma_C^2 + (1-k)\\,\\hat\\sigma_{RS}^2$, \\quad $k = \\dfrac{0{,}34}{1{,}34 + (n+1)/(n-1)}$'),
      T('$\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$: sample variances of the overnight returns $o_t$ and of the open-to-close returns $c_t$; $\\hat\\sigma_{RS}^2$: the mean Rogers--Satchell term',
        '$\\hat\\sigma_O^2$, $\\hat\\sigma_C^2$: varianțele de selecție ale randamentelor overnight $o_t$ și ale randamentelor deschidere--închidere $c_t$; $\\hat\\sigma_{RS}^2$: media termenului Rogers--Satchell')]),
    (T('Properties', 'Proprietăți'),
     [T('unbiased with drift and with opening jumps; $k$ minimises the variance of the estimator', 'nedeplasat cu tendință și cu salturi la deschidere; $k$ minimizează varianța estimatorului'),
      T('$n = 21$: $k = 0.34/(1.34 + 22/20) = 0.34/@{yz.den21} = @{yz.k21}$', '$n = 21$: $k = 0{,}34/(1{,}34 + 22/20) = 0{,}34/@{yz.den21} = @{yz.k21}$')]),
    T('Needs a window of $n \\ge 2$ days (sample variances); Parkinson, Garman--Klass and Rogers--Satchell also work for a single day',
      'Cere o fereastră de $n \\ge 2$ zile (varianțe de selecție); Parkinson, Garman--Klass și Rogers--Satchell funcționează și pentru o singură zi')))

D.frame(T('Worked Example: S\\&P 500 on 16 March 2020', 'Exemplu rezolvat: S\\&P 500 pe 16 martie 2020'), items(
    (T('Previous close @{m16.pc}; open @{m16.o}, high @{m16.h}, low @{m16.l}, close @{m16.c}', 'Închiderea precedentă @{m16.pc}; deschiderea @{m16.o}, maximul @{m16.h}, minimul @{m16.l}, închiderea @{m16.c}'),
     [T('$o = @{m16.oo}$, $u = @{m16.u}$, $d = @{m16.d}$, $c = @{m16.cv}$ (in \\%); close-to-close $r = o + c = @{m16.r}$',
        '$o = @{m16.oo}$, $u = @{m16.u}$, $d = @{m16.d}$, $c = @{m16.cv}$ (în \\%); închidere--închidere $r = o + c = @{m16.r}$')]),
    (T('One-day variance estimates (\\%$^2$)', 'Estimări ale varianței pentru o zi (\\%$^2$)'),
     [T('close-to-close: $r^2 = @{m16.r2}$', 'închidere--închidere: $r^2 = @{m16.r2}$'),
      T('Parkinson: $(@{m16.hl})^2/2.773 = @{m16.park}$; Garman--Klass: @{m16.gk}; Rogers--Satchell: @{m16.rs}', 'Parkinson: $(@{m16.hl})^2/2{,}773 = @{m16.park}$; Garman--Klass: @{m16.gk}; Rogers--Satchell: @{m16.rs}'),
      T('overnight part $o^2 = @{m16.on}$; overnight plus Rogers--Satchell: @{m16.rsall}', 'partea overnight $o^2 = @{m16.on}$; overnight plus Rogers--Satchell: @{m16.rsall}')]),
    T('Most of the fall happened overnight, before the open: estimators that ignore $o_t$ miss most of the risk of this day',
      'Cea mai mare parte a scăderii s-a produs overnight, înainte de deschidere: estimatorii care ignoră $o_t$ nu surprind cea mai mare parte a riscului din această zi')) + ql('SFM_ch8_range_estimators'))

chart(T('Efficiency by Simulation', 'Eficiența estimatorilor prin simulare'), 'sfm_ch8_efficiency', 'SFM_ch8_range_efficiency', [
    T('1000 windows of 21 days; true volatility $1\\%$ per day ($15.9\\%$ per year); intraday Brownian motion on 4680 steps (one every 5 seconds)',
      '1000 de ferestre de 21 de zile; volatilitatea reală $1\\%$ pe zi ($15{,}9\\%$ pe an); mișcare browniană intrazilnică pe 4680 de pași (unul la 5 secunde)'),
    T('\\textbf{Efficiency} of an estimator: $\\mathrm{Var}(\\hat\\sigma^2_{CC})/\\mathrm{Var}(\\hat\\sigma^2)$, with CC the close-to-close estimator; right panel: 20\\% of the daily variance comes overnight',
      '\\textbf{Eficiența} unui estimator: $\\mathrm{Var}(\\hat\\sigma^2_{CC})/\\mathrm{Var}(\\hat\\sigma^2)$, CC fiind estimatorul închidere--închidere; panoul din dreapta: 20\\% din varianța zilnică apare overnight')],
    h='0.52\\textheight')

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Interpretation: Precision Is Not Accuracy', 'Interpretarea simulării: precizie și deplasare'), table(
    'lrrrrr', T('& \\textbf{CC} & \\textbf{Parkinson} & \\textbf{GK} & \\textbf{RS} & \\textbf{YZ}', '& \\textbf{CC} & \\textbf{Parkinson} & \\textbf{GK} & \\textbf{RS} & \\textbf{YZ}'),
    [T('Efficiency, continuous trading', 'Eficiența, tranzacționare continuă') + ' & ' + ' & '.join(f'@{{ef.continuous.{e}.eff}}' for e in ['cc', 'park', 'gk', 'rs', 'yz']),
     T('Bias (\\%), continuous trading', 'Deplasarea (\\%), tranzacționare continuă') + ' & ' + ' & '.join(f'${{@{{ef.continuous.{e}.bias}}}}$' for e in ['cc', 'park', 'gk', 'rs', 'yz']),
     T('Bias (\\%), overnight jump', 'Deplasarea (\\%), salt overnight') + ' & ' + ' & '.join(f'${{@{{ef.overnight.{e}.bias}}}}$' for e in ['cc', 'park', 'gk', 'rs', 'yz']),
     T('MSE ratio, overnight jump', 'Raportul MSE, salt overnight') + ' & ' + ' & '.join(f'@{{ef.overnight.{e}.mse}}' for e in ['cc', 'park', 'gk', 'rs', 'yz']),
     T('Bias (\\%), 26 prices a day', 'Deplasarea (\\%), 26 de prețuri pe zi') + ' & ' + ' & '.join(f'${{@{{ef.discrete.{e}.bias}}}}$' for e in ['cc', 'park', 'gk', 'rs', 'yz'])],
    size='footnotesize') + items(
    T('With continuous trading, range estimators are @{ef.lo} to @{ef.hi} times more efficient: one month of ranges carries as much information as @{ef.lo} to @{ef.hi} months of closes',
      'Cu tranzacționare continuă, estimatorii de amplitudine sînt de @{ef.lo}--@{ef.hi} ori mai eficienți: o lună de amplitudini conține tot atîta informație cît @{ef.lo}--@{ef.hi} luni de închideri'),
    T('With an overnight jump, Parkinson, GK and RS miss the night (bias about $@{ef.nbias}\\%$); only YZ stays unbiased; \\textbf{MSE} (mean squared error) ratio = $\\mathrm{MSE}(CC)/\\mathrm{MSE}$',
      'Cu salt overnight, Parkinson, GK și RS omit varianța nopții (deplasare de circa $@{ef.nbias}\\%$); doar YZ rămîne nedeplasat; raportul \\textbf{MSE} (mean squared error, eroarea pătratică medie) = $\\mathrm{MSE}(CC)/\\mathrm{MSE}$'),
    T('With few trades the observed range is too small: all range estimators underestimate (thin markets) \\refMolnar',
      'Cu puține tranzacții amplitudinea observată este prea mică: toți estimatorii de amplitudine subestimează (piețe puțin lichide) \\refMolnar')) + ql('SFM_ch8_range_efficiency'), 'footnotesize')


def est_row(k):
    return (f'{SHORT[k]} & @{{es.{k}.y0}} & @{{es.{k}.cc}} & @{{es.{k}.park}} & @{{es.{k}.gk}} & @{{es.{k}.rs}} & @{{es.{k}.yz}} & '
            f'@{{es.{k}.night}}\\% & ${{@{{es.{k}.corr}}}}$')


D.frame(T('Real Data: Five Estimators, Five Assets', 'Date reale: cinci estimatori, cinci active'), table(
    'lrrrrrrrr', T('& \\textbf{Since} & \\textbf{CC} & \\textbf{P} & \\textbf{GK} & \\textbf{RS} & \\textbf{YZ} & \\textbf{Night share} & $\\mathrm{corr}(o, c)$',
                   '& \\textbf{Din} & \\textbf{CC} & \\textbf{P} & \\textbf{GK} & \\textbf{RS} & \\textbf{YZ} & \\textbf{Ponderea nopții} & $\\mathrm{corr}(o, c)$'),
    [est_row(k) for k in OHLC_ASSETS], size='footnotesize') + items(
    T('Annualised volatility (\\%) on the whole OHLC sample; P = Parkinson; night share $= \\hat\\sigma_O^2/(\\hat\\sigma_O^2 + \\hat\\sigma_C^2)$',
      'Volatilitatea anualizată (\\%) pe întregul eșantion OHLC; P = Parkinson; ponderea nopții $= \\hat\\sigma_O^2/(\\hat\\sigma_O^2 + \\hat\\sigma_C^2)$'),
    T('Stocks: open, high and low adjusted with the same factor as the close; days without trades removed',
      'Acțiuni: deschiderea, maximul și minimul ajustate cu același factor ca închiderea; zilele fără tranzacții sînt eliminate')) + ql('SFM_ch8_range_estimators'), 'footnotesize')

D.frame(T('Interpretation: Why the Estimators Disagree', 'Interpretarea diferențelor dintre estimatori'), items(
    (T('Bitcoin trades 24 hours a day: the night share is almost 0, and all five estimators agree within a few points',
       'Bitcoin se tranzacționează 24 de ore din 24: ponderea nopții este aproape 0, iar toți cei cinci estimatori sînt apropiați'),
     [T('the remaining gap: the daily path is observed in discrete trades, not continuously', 'diferența rămasă: traiectoria zilei este observată prin tranzacții discrete, nu continuu')]),
    (T('DAX and the BVB stocks: a quarter to a third of the variance comes overnight', 'DAX și acțiunile BVB: între un sfert și o treime din varianță apare overnight'),
     [T('Parkinson, GK and RS are far below CC; YZ, which adds the night, is close to CC', 'Parkinson, GK și RS sînt mult sub CC; YZ, care include varianța nopții, este apropiat de CC')]),
    (T('S\\&P 500: even YZ (@{es.sp500.yz}\\%) stays below CC (@{es.sp500.cc}\\%)', 'S\\&P 500: chiar și YZ (@{es.sp500.yz}\\%) rămîne sub CC (@{es.sp500.cc}\\%)'),
     [T('overnight and open-to-close returns are positively correlated (@{es.sp500.corr}); a likely reason: the index open uses stale prices of stocks that have not traded yet',
        'randamentele overnight și cele deschidere--închidere sînt corelate pozitiv (@{es.sp500.corr}); o explicație probabilă: deschiderea indicelui folosește prețuri stale (neactualizate) ale acțiunilor care încă nu s-au tranzacționat'),
      T('$\\mathrm{Var}(r) = \\mathrm{Var}(o) + \\mathrm{Var}(c) + 2\\,\\mathrm{Cov}(o, c)$: no range estimator sees the covariance term',
        '$\\mathrm{Var}(r) = \\mathrm{Var}(o) + \\mathrm{Var}(c) + 2\\,\\mathrm{Cov}(o, c)$: niciun estimator de amplitudine nu vede termenul de covarianță')]),
    T('Lesson: check the assumptions on the data before trusting an efficiency gain', 'Lecția: verificați ipotezele pe date înainte de a avea încredere într-un cîștig de eficiență')) + ql('SFM_ch8_range_estimators'))

chart(T('S\\&P 500: Rolling 21-Day Estimates since 2019', 'S\\&P 500: estimări pe 21 de zile din 2019'), 'sfm_ch8_range_rolling', 'SFM_ch8_range_estimators', [
    T('Annualised 21-day volatility by the five estimators; means since 2019: CC @{rr.cc}\\%, Parkinson @{rr.park}\\%, YZ @{rr.yz}\\%',
      'Volatilitatea pe 21 de zile, anualizată, calculată cu cei cinci estimatori; medii din 2019: CC @{rr.cc}\\%, Parkinson @{rr.park}\\%, YZ @{rr.yz}\\%'),
    T('The range-based lines are smoother: the typical daily change of the log estimate is @{rr.rough.park}\\% for Parkinson and @{rr.rough.yz}\\% for YZ, against @{rr.rough.cc}\\% for CC',
      'Liniile estimatorilor de amplitudine sînt mai netede: variația zilnică tipică a estimării logaritmice este @{rr.rough.park}\\% pentru Parkinson și @{rr.rough.yz}\\% pentru YZ, față de @{rr.rough.cc}\\% pentru CC')],
    h='0.54\\textheight')

D.recap(('Range-Based Estimators', 'estimatori de amplitudine'), [
    T('Parkinson uses the range, Garman--Klass adds the open-to-close return, Rogers--Satchell allows a drift, Yang--Zhang adds the night',
      'Parkinson folosește amplitudinea, Garman--Klass adaugă randamentul deschidere--închidere, Rogers--Satchell permite o tendință, Yang--Zhang adaugă varianța nopții'),
    T('In theory @{ef.lo} to @{ef.hi} times more efficient than close-to-close; in practice biased down by overnight jumps and discrete trading',
      'În teorie de @{ef.lo}--@{ef.hi} ori mai eficienți decît estimatorul închidere--închidere; în practică subestimează varianța din cauza salturilor overnight și a tranzacționării discrete'),
    T('Choose Yang--Zhang for markets that close at night; any range estimator for 24-hour markets', 'Alegeți Yang--Zhang pentru piețele care se închid noaptea; orice estimator de amplitudine pentru piețele deschise 24 de ore')])

# =============================================================================
# 5. VOLATILITATEA REALIZATĂ
# =============================================================================
D.section('Realised Volatility', 'Volatilitatea realizată')

D.frame(T('Realised Variance from Intraday Returns', 'Varianța realizată din randamentele intrazilnice'), items(
    (T('Split day $t$ into $M$ intervals (for example $M = 78$ five-minute returns $r_{t,i}$ in a 6.5-hour session)',
       'Împărțim ziua $t$ în $M$ intervale (de exemplu $M = 78$ de randamente de cinci minute $r_{t,i}$ într-o ședință de 6,5 ore)'),
     [T('$r_{t,i}$: the log return of interval $i$ of day $t$, $i = 1, \\dots, M$', '$r_{t,i}$: randamentul logaritmic al intervalului $i$ din ziua $t$, $i = 1, \\dots, M$'),
      T('\\textbf{realised variance} $\\mathrm{RV}_t = \\sum_{i=1}^{M} r_{t,i}^2$; \\textbf{realised volatility} $= \\sqrt{\\mathrm{RV}_t}$',
        '\\textbf{varianța realizată} $\\mathrm{RV}_t = \\sum_{i=1}^{M} r_{t,i}^2$; \\textbf{volatilitatea realizată} $= \\sqrt{\\mathrm{RV}_t}$'),
      T('as $M \\to \\infty$, $\\mathrm{RV}_t$ converges to the \\textbf{integrated variance} of day $t$, the true variance of the day \\refABDL, \\refBNS',
        'cînd $M \\to \\infty$, $\\mathrm{RV}_t$ converge către \\textbf{varianța integrată} a zilei $t$, adică varianța reală a zilei \\refABDL, \\refBNS')]),
    (T('Volatility becomes almost observable: a nearly error-free measure for each day', 'Volatilitatea devine aproape observabilă: o măsură aproape fără eroare pentru fiecare zi'),
     [T('used to evaluate forecasts (Section 8) and to forecast volatility directly, for example with the HAR model \\refCorsi',
        'se folosește la evaluarea prognozelor (secțiunea 8) și la prognoza directă a volatilității, de exemplu cu modelul HAR \\refCorsi'),
      T('high-frequency Bitcoin prices and daily risk: \\refPeleB', 'prețuri Bitcoin de înaltă frecvență și riscul zilnic: \\refPeleB')]),
    T('Our course data are daily; the next two slides use simulated one-minute prices to show how RV behaves',
      'Datele cursului sînt zilnice; următoarele două slide-uri folosesc prețuri simulate la un minut pentru a arăta comportamentul RV')))

chart(T('Simulated Example: RV Tracks the True Volatility', 'Exemplu simulat: RV urmărește volatilitatea reală'), 'sfm_ch8_realised', 'SFM_ch8_realised_volatility', [
    T('60 simulated days of 390 one-minute returns; the daily volatility follows a log AR(1) process (persistence $0.9$)',
      '60 de zile simulate, fiecare cu 390 de randamente de un minut; logaritmul volatilității zilnice urmează un proces AR(1) (persistența $0{,}9$)'),
    T('Mean absolute relative error: 5-minute RV @{rv.mae5}\\%, Parkinson @{rv.maep}\\%, $|r_t|$ @{rv.maer}\\%; correlation with the truth: @{rv.c5}, @{rv.cp} and @{rv.cr}',
      'Eroarea relativă absolută medie: RV la 5 minute @{rv.mae5}\\%, Parkinson @{rv.maep}\\%, $|r_t|$ @{rv.maer}\\%; corelația cu valoarea reală: @{rv.c5}, @{rv.cp} și @{rv.cr}')],
    h='0.52\\textheight')

chart(T('Microstructure Noise and the Signature Plot', 'Zgomotul de microstructură și signature plot'), 'sfm_ch8_signature', 'SFM_ch8_realised_volatility', [
    T('Observed prices = true prices + \\textbf{microstructure noise} (bid--ask bounce, price ticks), here with sd $@{sg.noise}\\%$; each squared return then adds $2\\times$ the noise variance',
      'Prețurile observate = prețurile reale + \\textbf{zgomot de microstructură} (oscilația între bid și ask, pasul de cotare), aici cu abaterea standard $@{sg.noise}\\%$; fiecare randament la pătrat crește atunci, în medie, cu dublul varianței zgomotului'),
    T('With noise, one-minute RV overstates the variance by a factor of @{sg.n1}; at 5 minutes: @{sg.n5}; the \\textbf{signature plot} shows RV against the sampling interval',
      'Cu zgomot, RV la un minut supraestimează varianța de @{sg.n1} ori; la 5 minute: @{sg.n5}; \\textbf{signature plot} arată RV în funcție de intervalul de eșantionare'),
    T('In practice, 5-minute RV is hard to beat \\refLPS', 'În practică, RV la 5 minute este greu de depășit \\refLPS')],
    h='0.46\\textheight')

D.recap(('Realised Volatility', 'volatilitatea realizată'), [
    T('$\\mathrm{RV}_t = \\sum_i r_{t,i}^2$ estimates the variance of one day almost without error', '$\\mathrm{RV}_t = \\sum_i r_{t,i}^2$ estimează varianța unei zile aproape fără eroare'),
    T('Too frequent sampling picks up microstructure noise; 5 minutes is the usual compromise', 'Eșantionarea prea deasă preia zgomotul de microstructură; intervalul de 5 minute este compromisul uzual'),
    T('Ranking of daily measures: RV $>$ range estimators $>$ squared daily return', 'Ordinea după precizie a măsurilor zilnice: RV $>$ estimatorii de amplitudine $>$ randamentul zilnic la pătrat')])

# =============================================================================
# 6. VOLATILITY CLUSTERING
# =============================================================================
D.section('Volatility Clustering', 'Volatility clustering')

D.frame(T('1982: Engle and the ARCH Model', '1982: Engle și modelul ARCH'), cols(items(
    (T('\\textbf{Volatility clustering}: periods of large absolute returns alternate with periods of small ones \\refMandelbrot, \\refCont',
       '\\textbf{Volatility clustering}: perioadele cu randamente mari în valoare absolută alternează cu perioade cu randamente mici \\refMandelbrot, \\refCont'),
     [T('returns are almost uncorrelated (Chapter 7), but $|r_t|$ and $r_t^2$ are strongly autocorrelated', 'randamentele sînt aproape necorelate (Capitolul 7), dar $|r_t|$ și $r_t^2$ sînt puternic autocorelate')]),
    (T('Robert F. Engle \\refEngle: the variance of today depends on yesterday\'s squared shocks: \\textbf{ARCH}',
       'Robert F. Engle \\refEngle: varianța de azi depinde de șocurile la pătrat de ieri: \\textbf{ARCH}'),
     [T('the first model in which $\\sigma_t$ changes over time in a predictable way', 'primul model în care $\\sigma_t$ variază în timp într-un mod previzibil'),
      T('Nobel Prize in Economic Sciences 2003, shared with Clive Granger \\refNobel; Engle\'s Nobel lecture: \\refEngleN',
        'Premiul Nobel pentru economie (2003), împărțit cu Clive Granger \\refNobel; prelegerea Nobel a lui Engle: \\refEngleN')]),
    T('This section: tests for clustering; Chapter 9: the models', 'Această secțiune: testele pentru volatility clustering; Capitolul 9: modelele')),
    ph('engle', T('Robert F. Engle (b. 1942), Santiago, 2017', 'Robert F. Engle (n. 1942), Santiago, 2017'), h='0.40\\textheight'), wl='0.60', wr='0.36'))

chart(T('The Sign Is Unpredictable, the Size Is Not', 'Semnul este imprevizibil, mărimea nu'), 'sfm_ch8_acf_clustering', 'SFM_ch8_clustering', [
    T('Sample ACF (autocorrelation function, Chapter 7) of $r_t$, $r_t^2$ and $|r_t|$, lags 1--100, with the 95\\% band $\\pm 1.96/\\sqrt{T}$ for i.i.d.\\ data',
      'ACF de selecție (funcția de autocorelație, Capitolul 7) pentru $r_t$, $r_t^2$ și $|r_t|$, lagurile 1--100, cu banda de 95\\% $\\pm 1{,}96/\\sqrt{T}$ pentru date i.i.d.'),
    T('S\\&P 500: all 100 autocorrelations of $r_t^2$ and of $|r_t|$ lie above the band; at lag 100, $|r_t|$ still has $@{ac.sp500.abs100}$',
      'S\\&P 500: toate cele 100 de autocorelații ale lui $r_t^2$ și $|r_t|$ sînt deasupra benzii; la lagul 100, $|r_t|$ are încă $@{ac.sp500.abs100}$')],
    h='0.52\\textheight')

D.frame(T('Interpretation: Clustering in Three Markets', 'Interpretarea volatility clustering pe trei piețe'), table(
    'lrrrrrr', T('& $\\hat\\rho_1(|r|)$ & $\\hat\\rho_{10}(|r|)$ & $\\hat\\rho_{100}(|r|)$ & $\\hat\\rho_1(r^2)$ & $\\hat\\rho_{10}(r^2)$ & \\textbf{Lags outside, $|r|$}',
                 '& $\\hat\\rho_1(|r|)$ & $\\hat\\rho_{10}(|r|)$ & $\\hat\\rho_{100}(|r|)$ & $\\hat\\rho_1(r^2)$ & $\\hat\\rho_{10}(r^2)$ & \\textbf{Laguri în afara benzii, $|r|$}'),
    [f'{NAMES[k]} & $@{{ac.{k}.abs1}}$ & $@{{ac.{k}.abs10}}$ & $@{{ac.{k}.abs100}}$ & $@{{ac.{k}.sq1}}$ & $@{{ac.{k}.sq10}}$ & @{{ac.{k}.outabs}}/100'
     for k in ['sp500', 'bet', 'btc']], size='footnotesize') + items(
    T('Slow decay: a shock to volatility is still visible after months; Chapter 11 calls this long memory',
      'Scădere lentă: un șoc de volatilitate se vede încă după luni de zile; Capitolul 11 numește acest fenomen memorie lungă'),
    T('\\textbf{Taylor effect}: $|r_t|$ is more autocorrelated than $r_t^2$ (@{ac.sp500.taylor}\\% of lags for the S\\&P 500, all lags for the BET and Bitcoin) \\refDGE',
      '\\textbf{Efectul Taylor}: $|r_t|$ este mai autocorelat decît $r_t^2$ (@{ac.sp500.taylor}\\% dintre laguri la S\\&P 500, toate lagurile la BET și Bitcoin) \\refDGE'),
    T('Bitcoin: weaker clustering in $r_t^2$ (@{ac.btc.outsq} of 100 lags outside the band): a few huge days dominate the squares',
      'Bitcoin: volatility clustering mai slab în $r_t^2$ (@{ac.btc.outsq} din 100 de laguri în afara benzii): cîteva zile cu variații extreme domină pătratele')) + ql('SFM_ch8_clustering'), 'footnotesize')

chart(T('Is Clustering Real? Shuffle the Days', 'Amestecarea zilelor: un test pentru volatility clustering'), 'sfm_ch8_shuffle', 'SFM_ch8_clustering', [
    T('The same S\\&P 500 returns in a random order: same distribution and same kurtosis (excess @{sh.k}), but no clustering',
      'Aceleași randamente S\\&P 500 într-o ordine aleatoare: aceeași distribuție și aceeași boltire (excesul de boltire @{sh.k}), dar fără volatility clustering'),
    T('$\\hat\\rho_1(|r|)$: $@{sh.a1}$ in the actual order, $@{sh.a1s}$ after shuffling; Ljung--Box $Q(10)$ of $r_t^2$: @{sh.lb} against @{sh.lbs}',
      '$\\hat\\rho_1(|r|)$: $@{sh.a1}$ în ordinea reală, $@{sh.a1s}$ după amestecare; Ljung--Box $Q(10)$ pentru $r_t^2$: @{sh.lb} față de @{sh.lbs}'),
    T('Clustering is a property of the \\textbf{order} of the returns, not of their distribution', 'Volatility clustering este o proprietate a \\textbf{ordinii} randamentelor, nu a distribuției lor')],
    h='0.48\\textheight')

D.frame(T('Testing for Clustering: Ljung--Box on Squared Returns', 'Testarea volatility clustering: Ljung--Box pe randamentele la pătrat'), items(
    (T('$H_0$: no clustering, $\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ constant, so $r_t^2$ is uncorrelated', '$H_0$: fără volatility clustering, $\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ constantă, deci $r_t^2$ este necorelat'),
     [T('\\textbf{McLeod--Li test} \\refML: the Ljung--Box statistic \\refLjung applied to $r_t^2$', '\\textbf{Testul McLeod--Li} \\refML: statistica Ljung--Box \\refLjung aplicată lui $r_t^2$'),
      T('$Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k(r^2)^2/(T-k) \\sim \\chi^2(m)$ under $H_0$: a weighted sum of the first $m$ squared autocorrelations',
        '$Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k(r^2)^2/(T-k) \\sim \\chi^2(m)$ în ipoteza $H_0$: o sumă ponderată a pătratelor primelor $m$ autocorelații'),
      T('$T$: number of returns; $m$: number of lags tested; $\\hat\\rho_k(r^2)$: sample autocorrelation of $r_t^2$ at lag $k$',
        '$T$: numărul de randamente; $m$: numărul de laguri testate; $\\hat\\rho_k(r^2)$: autocorelația de selecție a lui $r_t^2$ la lagul $k$'),
      T('$Q(m) \\ge 0$; large values mean autocorrelated squares; reject if $Q(m)$ exceeds the 95\\% quantile $\\chi^2_{0.95}(m)$',
        '$Q(m) \\ge 0$; valorile mari indică pătrate autocorelate; respingem dacă $Q(m)$ depășește cuantila de 95\\% $\\chi^2_{0.95}(m)$')]),
    (T('\\textbf{Worked example}: S\\&P 500, $T = @{b.sp500.n}$, $m = 3$', '\\textbf{Exemplu rezolvat}: S\\&P 500, $T = @{b.sp500.n}$, $m = 3$'),
     [T('$\\hat\\rho_1, \\hat\\rho_2, \\hat\\rho_3$ of $r_t^2$: $@{ml.r1}$, $@{ml.r2}$, $@{ml.r3}$', '$\\hat\\rho_1, \\hat\\rho_2, \\hat\\rho_3$ pentru $r_t^2$: $@{ml.r1}$; $@{ml.r2}$; $@{ml.r3}$'),
      T('terms $T(T+2)\\hat\\rho_k^2/(T-k)$: @{ml.t1}, @{ml.t2}, @{ml.t3}; $Q(3) = @{ml.q} \\gg \\chi^2_{0.95}(3) = @{ml.crit}$: reject $H_0$',
        'termenii $T(T+2)\\hat\\rho_k^2/(T-k)$: @{ml.t1}; @{ml.t2}; @{ml.t3}; $Q(3) = @{ml.q} \\gg \\chi^2_{0.95}(3) = @{ml.crit}$: respingem $H_0$')]),
    T('The same test on $r_t$ (Chapter 7) asks a different question: is the \\textbf{mean} predictable?', 'Același test aplicat lui $r_t$ (Capitolul 7) pune o altă întrebare: este \\textbf{media} previzibilă?')) + ql('SFM_ch8_clustering'))

D.frame(T('The ARCH-LM Test (Engle, 1982)', 'Testul ARCH-LM (Engle, 1982)'), items(
    (T('Auxiliary regression of the squared demeaned returns $e_t^2$, $e_t = r_t - \\bar r$, on $q$ of their own lags \\refEngle:',
       'Regresia auxiliară a randamentelor centrate la pătrat $e_t^2$, $e_t = r_t - \\bar r$, pe $q$ dintre propriile lor laguri \\refEngle:'),
     [T('$e_t^2 = c_0 + c_1 e_{t-1}^2 + \\dots + c_q e_{t-q}^2 + u_t$; $H_0$: $c_1 = \\dots = c_q = 0$', '$e_t^2 = c_0 + c_1 e_{t-1}^2 + \\dots + c_q e_{t-q}^2 + u_t$; $H_0$: $c_1 = \\dots = c_q = 0$'),
      T('$\\bar r$: the sample mean; $c_0, \\dots, c_q$: regression coefficients; $u_t$: the error term', '$\\bar r$: media de selecție; $c_0, \\dots, c_q$: coeficienții regresiei; $u_t$: eroarea'),
      T('\\textbf{LM} (Lagrange multiplier) statistic: $\\mathrm{LM} = n R^2 \\sim \\chi^2(q)$, $n$ = number of observations in the regression',
        'statistica \\textbf{LM} (Lagrange multiplier, multiplicatorul Lagrange): $\\mathrm{LM} = n R^2 \\sim \\chi^2(q)$, $n$ = numărul de observații din regresie')]),
    (T('S\\&P 500, $q = 5$: $R^2 = @{cl.sp500.lmr2}$, $n = @{cl.sp500.lmn}$', 'S\\&P 500, $q = 5$: $R^2 = @{cl.sp500.lmr2}$, $n = @{cl.sp500.lmn}$'),
     [T('$R^2 \\in [0, 1]$: the share of the variance of $e_t^2$ explained by its own past', '$R^2 \\in [0, 1]$: proporția din varianța lui $e_t^2$ explicată de propriul trecut'),
      T('$\\mathrm{LM} = @{cl.sp500.lmn} \\times @{cl.sp500.lmr2} = @{cl.sp500.lm} \\gg \\chi^2_{0.95}(5) = @{cl.crit5}$: strong ARCH effects',
        '$\\mathrm{LM} = @{cl.sp500.lmn} \\times @{cl.sp500.lmr2} = @{cl.sp500.lm} \\gg \\chi^2_{0.95}(5) = @{cl.crit5}$: efecte ARCH puternice')]),
    T('Interpretation of $R^2$: about a quarter of the day-to-day variation of $e_t^2$ is predictable from the last five days',
      'Interpretarea lui $R^2$: circa un sfert din variația de la o zi la alta a lui $e_t^2$ poate fi anticipată din ultimele cinci zile'),
    T('Both tests are diagnostics: they say that clustering exists, not how to model it (Chapter 9)', 'Ambele teste sînt instrumente de diagnostic: arată că volatility clustering există, nu cum îl modelăm (Capitolul 9)')) + ql('SFM_ch8_clustering'))


def cl_row(k):
    return (f'{SHORT[k]} & @{{cl.{k}.n}} & ${{@{{cl.{k}.rr}}}}$ & $@{{cl.{k}.lbr}}$ & $@{{cl.{k}.r2}}$ & @{{cl.{k}.lbr2}} & @{{cl.{k}.lbabs}} & '
            f'@{{cl.{k}.lm}} & $@{{cl.{k}.kurt}}$')


D.frame(T('Clustering Tests in Six Series', 'Testele de volatility clustering pe șase serii'), table(
    'lrrrrrrrr', T('& $T$ & $\\hat\\rho_1(r)$ & $Q(10)$, $r$ & $\\hat\\rho_1(r^2)$ & $Q(10)$, $r^2$ & $Q(10)$, $|r|$ & ARCH-LM(5) & \\textbf{Exc.\\ kurt.}',
                   '& $T$ & $\\hat\\rho_1(r)$ & $Q(10)$, $r$ & $\\hat\\rho_1(r^2)$ & $Q(10)$, $r^2$ & $Q(10)$, $|r|$ & ARCH-LM(5) & \\textbf{Exc.\\ boltire}'),
    [cl_row(k) for k in ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']], size='scriptsize') + items(
    T('5\\% critical values: $\\chi^2_{0.95}(10) = @{cl.crit10}$ for $Q(10)$, $\\chi^2_{0.95}(5) = @{cl.crit5}$ for ARCH-LM(5); every statistic on $r^2$ and $|r|$ rejects with p $<$ 0.001',
      'Valori critice de 5\\%: $\\chi^2_{0.95}(10) = @{cl.crit10}$ pentru $Q(10)$, $\\chi^2_{0.95}(5) = @{cl.crit5}$ pentru ARCH-LM(5); toate statisticile pentru $r^2$ și $|r|$ resping, cu p $<$ 0,001'),
    T('$Q(10)$ on $r$ is far smaller: mean predictability is weak (Chapter 7), size predictability is strong everywhere',
      '$Q(10)$ pentru $r$ este mult mai mic: previzibilitatea mediei este slabă (Capitolul 7), cea a mărimii este puternică peste tot'),
    T('Bitcoin and the two BVB stocks have the smallest ARCH-LM values despite heavy tails: a few extreme days dilute the regression',
      'Bitcoin și cele două acțiuni BVB au cele mai mici valori ARCH-LM, deși au cozi groase: cîteva zile extreme domină regresia și reduc $R^2$')) + ql('SFM_ch8_clustering'), 'footnotesize')

chart(T('Clustering Creates Heavy Tails', 'Volatility clustering produce cozi groase'), 'sfm_ch8_kurtosis', 'SFM_ch8_kurgarch', [
    T('Left: calm days with sd 1, turbulent days with sd 2 (share $p = 0.2$): $E[\\sigma^2] = @{mx.e2}$, $E[\\sigma^4] = @{mx.e4}$, kurtosis $3E[\\sigma^4]/E[\\sigma^2]^2 = @{ku.m022}$, though each day is Normal',
      'Stînga: zile liniștite cu abaterea standard 1, zile agitate cu abaterea standard 2 (ponderea $p = 0{,}2$): $E[\\sigma^2] = @{mx.e2}$, $E[\\sigma^4] = @{mx.e4}$, coeficientul de boltire $3E[\\sigma^4]/E[\\sigma^2]^2 = @{ku.m022}$, deși randamentul fiecărei zile urmează distribuția Normală'),
    T('Right: GARCH(1,1) kurtosis $3 + 6\\alpha^2/(1 - \\beta^2 - 2\\alpha\\beta - 3\\alpha^2)$, as in SFEkurgarch; $\\alpha = 0.10$, $\\beta = 0.85$: @{ku.g} (Chapter 9)',
      'Dreapta: coeficientul de boltire al unui GARCH(1,1), $3 + 6\\alpha^2/(1 - \\beta^2 - 2\\alpha\\beta - 3\\alpha^2)$, ca în SFEkurgarch; $\\alpha = 0{,}10$, $\\beta = 0{,}85$: @{ku.g} (Capitolul 9)')],
    h='0.56\\textheight')

D.recap(('Volatility Clustering', 'volatility clustering'), [
    T('$r_t$ almost uncorrelated, $|r_t|$ and $r_t^2$ strongly and persistently correlated: clustering', '$r_t$ aproape necorelat, $|r_t|$ și $r_t^2$ corelate puternic și persistent: volatility clustering'),
    T('Tests: Ljung--Box on $r_t^2$ (McLeod--Li) and ARCH-LM $= nR^2$; both reject in every market', 'Testele: Ljung--Box pe $r_t^2$ (McLeod--Li) și ARCH-LM $= nR^2$; ambele resping pe toate piețele'),
    T('Clustering is about the order of returns, and it is one source of heavy tails', 'Volatility clustering ține de ordinea randamentelor și este una dintre sursele cozilor groase')])

# =============================================================================
# 7. TERMEN LUNG, VIX ȘI EFECTUL DE LEVIER
# =============================================================================
D.section('Long-Run Behaviour, the VIX and the Leverage Effect', 'Comportamentul pe termen lung, VIX și efectul de levier')

chart(T('Volatility over Three Decades', 'Volatilitatea în trei decenii'), 'sfm_ch8_long_term', 'SFM_ch8_long_term_vix', [
    T('Left: volatility of each calendar year (annualised, \\%); right: log monthly realised volatility of the S\\&P 500 (from daily returns) against the previous month',
      'Stînga: volatilitatea fiecărui an calendaristic (anualizată, \\%); dreapta: logaritmul volatilității realizate lunare a S\\&P 500 (din randamente zilnice) față de luna precedentă'),
    T('S\\&P 500: from @{lt.sp500.min}\\% (@{lt.sp500.miny}) to @{lt.sp500.max}\\% (@{lt.sp500.maxy}); BET: @{lt.bet.min}\\% (@{lt.bet.miny}) to @{lt.bet.max}\\% (@{lt.bet.maxy}); Bitcoin: @{lt.btc.min}\\% (@{lt.btc.miny}) to @{lt.btc.max}\\% (@{lt.btc.maxy})',
      'S\\&P 500: de la @{lt.sp500.min}\\% (@{lt.sp500.miny}) la @{lt.sp500.max}\\% (@{lt.sp500.maxy}); BET: de la @{lt.bet.min}\\% (@{lt.bet.miny}) la @{lt.bet.max}\\% (@{lt.bet.maxy}); Bitcoin: de la @{lt.btc.min}\\% (@{lt.btc.miny}) la @{lt.btc.max}\\% (@{lt.btc.maxy})')],
    h='0.50\\textheight')

D.frame(T('Interpretation: Persistence and Mean Reversion', 'Interpretarea persistenței și a revenirii la medie'), items(
    (T('Volatility changes over time, but it does not drift away: it returns to a normal level \\refSchwert', 'Volatilitatea se schimbă în timp, dar nu se îndepărtează la nesfîrșit: revine la un nivel normal \\refSchwert'),
     [T('AR(1) of log monthly realised volatility of the S\\&P 500 (@{lt.nm} months): $\\hat\\phi = @{lt.phi}$ (SE @{lt.se})', 'AR(1) pentru logaritmul volatilității realizate lunare a S\\&P 500 (@{lt.nm} de luni): $\\hat\\phi = @{lt.phi}$ (SE @{lt.se})'),
      T('$\\hat\\phi$: the estimated weight of last month\'s log volatility; close to 1 means slow mean reversion', '$\\hat\\phi$: ponderea estimată a logaritmului volatilității din luna precedentă; o valoare apropiată de 1 înseamnă o revenire lentă la medie'),
      T('half-life of a volatility shock: $\\ln 0.5/\\ln\\hat\\phi = @{lt.hl}$ months; long-run level about @{lt.lr}\\%', 'timpul de înjumătățire al unui șoc de volatilitate: $\\ln 0{,}5/\\ln\\hat\\phi = @{lt.hl}$ luni; nivelul pe termen lung circa @{lt.lr}\\%')]),
    (T('Monthly volatility is right-skewed (skewness @{lt.skr}), its logarithm much less (@{lt.skl}) \\refABDE', 'Volatilitatea lunară are asimetrie la dreapta (coeficientul de asimetrie @{lt.skr}), logaritmul ei mult mai puțin (@{lt.skl}) \\refABDE'),
     [T('mean @{lt.mean}\\% above the median @{lt.med}\\%: a few crisis months pull the mean up', 'media @{lt.mean}\\% este peste mediana @{lt.med}\\%: cîteva luni de criză ridică media'),
      T('models of volatility are therefore often written for $\\ln\\sigma_t$', 'de aceea modelele volatilității sînt adesea scrise pentru $\\ln\\sigma_t$')]),
    T('Mean reversion is what EWMA lacks and GARCH with $\\alpha + \\beta < 1$ has (Chapter 9)', 'Revenirea la medie este exact ce îi lipsește modelului EWMA și ce are GARCH cu $\\alpha + \\beta < 1$ (Capitolul 9)')) + ql('SFM_ch8_long_term_vix'))

D.frame(T('The VIX: Implied Volatility', 'VIX: volatilitatea implicită'), cols(items(
    (T('Options trade on exchanges since 1973, when members of the Chicago Board of Trade founded the \\textbf{CBOE} (Chicago Board Options Exchange)',
       'Opțiunile se tranzacționează la bursă din 1973, cînd membrii bursei Chicago Board of Trade au înființat \\textbf{CBOE} (Chicago Board Options Exchange)'),
     [T('an option price depends on the volatility the market expects until expiry', 'prețul unei opțiuni depinde de volatilitatea pe care piața o așteaptă pînă la scadență'),
      T('\\textbf{implied volatility}: the volatility that, put into an option-pricing formula, gives the observed price', '\\textbf{volatilitatea implicită}: volatilitatea care, introdusă într-o formulă de evaluare a opțiunilor, dă prețul observat')]),
    (T('\\textbf{VIX}, introduced in 1993 \\refWhaley: the expected volatility of the S\\&P 500 over the next 30 calendar days, from S\\&P 500 option prices \\refCboe',
       '\\textbf{VIX}, introdus în 1993 \\refWhaley: volatilitatea așteptată a S\\&P 500 pentru următoarele 30 de zile calendaristice, din prețurile opțiunilor pe S\\&P 500 \\refCboe'),
     [T('quoted in annualised \\%: VIX $= 20$ means about $20/\\sqrt{12} \\approx 5.8\\%$ over a month', 'exprimat în procente anualizate: VIX $= 20$ înseamnă circa $20/\\sqrt{12} \\approx 5{,}8\\%$ pe o lună'),
      T('called the ``fear index\'\': it jumps when stocks fall \\refWhaleyB', 'numit „indicele fricii”: crește brusc cînd acțiunile scad \\refWhaleyB')])),
    ph('cbot', T('Traders at the Chicago Board of Trade, 31 May 1973', 'Agenți de bursă la Chicago Board of Trade, 31 mai 1973'), h='0.34\\textheight'), wl='0.58', wr='0.38'))

chart(T('VIX against the Volatility That Followed', 'VIX față de volatilitatea care a urmat'), 'sfm_ch8_vix', 'SFM_ch8_long_term_vix', [
    T('VIX and the realised volatility of the S\\&P 500 over the next 21 trading days, $\\sqrt{(A/21)\\sum_{j=1}^{21} r_{t+j}^2}$; maximum VIX: @{vx.max} on @{vx.maxd}',
      'VIX și volatilitatea realizată a S\\&P 500 în următoarele 21 de zile de tranzacționare, $\\sqrt{(A/21)\\sum_{j=1}^{21} r_{t+j}^2}$; maximul VIX: @{vx.max} pe @{vx.maxd}'),
    T('\\textbf{Question for the room}: is the red line usually above or below the blue one?', '\\textbf{Întrebare pentru sală}: linia roșie este de obicei deasupra sau sub cea albastră?')],
    h='0.52\\textheight')

D.frame(T('Interpretation: Implied against Realised Volatility', 'Interpretarea relației dintre volatilitatea implicită și cea realizată'), items(
    (T('\\textbf{Answer}: above; VIX exceeded the next 21 days\' realised volatility on @{vx.share}\\% of the @{vx.n} days', '\\textbf{Răspuns}: deasupra; VIX a depășit volatilitatea realizată din următoarele 21 de zile în @{vx.share}\\% din cele @{vx.n} de zile'),
     [T('mean VIX @{vx.mv}\\% against mean realised @{vx.mr}\\%: a gap of @{vx.gap} points', 'media VIX @{vx.mv}\\% față de media volatilității realizate @{vx.mr}\\%: o diferență de @{vx.gap} puncte'),
      T('\\textbf{variance risk premium}: investors pay to be insured against volatility spikes \\refCW, \\refBTZ', '\\textbf{prima de risc a varianței}: investitorii plătesc pentru a fi asigurați împotriva salturilor de volatilitate \\refCW, \\refBTZ')]),
    (T('As a forecast, VIX is informative but biased', 'Ca prognoză, VIX este informativ, dar deplasat'),
     [T('regression of future on implied: realised $= @{vx.mza} + @{vx.mzb}\\,$VIX, $R^2 = @{vx.mzr}$', 'regresia volatilității realizate viitoare pe VIX: realizată $= @{vx.mza} + @{vx.mzb}\\,$VIX, $R^2 = @{vx.mzr}$'),
      T('the past 21-day realised volatility explains less: $R^2 = @{vx.pr}$', 'volatilitatea realizată din ultimele 21 de zile explică mai puțin: $R^2 = @{vx.pr}$')]),
    T('Daily log changes of the VIX and S\\&P 500 returns: correlation $@{vx.dvr}$: fear rises when prices fall', 'Variațiile zilnice ale logaritmului VIX și randamentele S\\&P 500: corelație $@{vx.dvr}$: frica crește cînd prețurile scad')) + ql('SFM_ch8_long_term_vix'))

D.frame(T('The Leverage Effect', 'Efectul de levier'), items(
    (T('\\textbf{Leverage effect}: negative returns are followed by higher volatility than positive returns of the same size \\refChristie',
       '\\textbf{Efectul de levier}: randamentele negative sînt urmate de o volatilitate mai mare decît randamentele pozitive de aceeași mărime \\refChristie'),
     [T('leverage story: a falling share price raises the debt-to-equity ratio, so the equity becomes riskier', 'explicația prin levier: scăderea prețului acțiunii crește raportul datorii/capital propriu, deci acțiunea devine mai riscantă'),
      T('volatility-feedback story: higher expected volatility raises the required return, so the price falls today \\refFSS, \\refBW',
        'explicația prin feedback de volatilitate: o volatilitate așteptată mai mare crește randamentul cerut, deci prețul scade azi \\refFSS, \\refBW')]),
    (T('A simple measure: $\\mathrm{corr}(r_t, |r_{t+j}|)$ for $j = 1, 2, \\dots$', 'O măsură simplă: $\\mathrm{corr}(r_t, |r_{t+j}|)$ pentru $j = 1, 2, \\dots$'),
     [T('$j$: the number of days ahead; $\\mathrm{corr}$: the sample correlation over all days $t$', '$j$: numărul de zile în avans; $\\mathrm{corr}$: corelația de selecție pe toate zilele $t$'),
      T('negative: today\'s fall announces tomorrow\'s turbulence; zero under a symmetric model', 'negativă: o scădere azi este urmată de o volatilitate mai mare mîine; zero într-un model simetric')]),
    T('Chapter 2 met the asymmetry of the distribution; here the asymmetry is in the \\textbf{dynamics} of volatility',
      'Capitolul 2 a arătat asimetria distribuției; aici asimetria apare în \\textbf{dinamica} volatilității')))

chart(T('Bad News Raises Future Volatility', 'Știrile negative cresc volatilitatea viitoare'), 'sfm_ch8_leverage', 'SFM_ch8_leverage', [
    T('$\\mathrm{corr}(r_t, |r_{t+j}|)$, $j = 1, \\dots, 10$; S\\&P 500: $@{lv.sp500.c1}$ at $j = 1$; the reverse direction $\\mathrm{corr}(|r_{t-1}|, r_t) = @{lv.sp500.cm1}$',
      '$\\mathrm{corr}(r_t, |r_{t+j}|)$, $j = 1, \\dots, 10$; S\\&P 500: $@{lv.sp500.c1}$ la $j = 1$; în sens invers, $\\mathrm{corr}(|r_{t-1}|, r_t) = @{lv.sp500.cm1}$'),
    T('Mean $r_{t+1}^2$ after a down day divided by that after an up day: S\\&P 500 @{lv.sp500.ratio}, DAX @{lv.dax.ratio}, BET @{lv.bet.ratio}, Bitcoin @{lv.btc.ratio}',
      'Media lui $r_{t+1}^2$ după o zi de scădere împărțită la cea după o zi de creștere: S\\&P 500 @{lv.sp500.ratio}, DAX @{lv.dax.ratio}, BET @{lv.bet.ratio}, Bitcoin @{lv.btc.ratio}')],
    h='0.52\\textheight')

chart(T('Nonparametric Volatility Given Yesterday\'s Return', 'Volatilitatea neparametrică în funcție de randamentul de ieri'), 'sfm_ch8_news_impact', 'SFM_ch8_leverage', [
    T('$\\hat\\sigma(x) = \\sqrt{\\hat m_2(x) - \\hat m_1(x)^2}$, $\\hat m_1$, $\\hat m_2$: local linear regressions of $r_t$ and $r_t^2$ on $r_{t-1} = x$ (quartic kernel, bandwidth $0.04$ on decimal returns, as in SFEvolnonparest)',
      '$\\hat\\sigma(x) = \\sqrt{\\hat m_2(x) - \\hat m_1(x)^2}$, $\\hat m_1$, $\\hat m_2$: regresii liniare locale ale lui $r_t$ și $r_t^2$ pe $r_{t-1} = x$ (nucleu cvartic, lățimea de bandă $0{,}04$ pe randamente zecimale, ca în SFEvolnonparest)'),
    T('S\\&P 500: $\\hat\\sigma(-2\\%) = @{np.sp500.m2}\\%$ against $\\hat\\sigma(+2\\%) = @{np.sp500.p2}\\%$; the minimum is at $x \\approx @{np.sp500.min}\\%$, not at 0: an asymmetric news impact',
      'S\\&P 500: $\\hat\\sigma(-2\\%) = @{np.sp500.m2}\\%$ față de $\\hat\\sigma(+2\\%) = @{np.sp500.p2}\\%$; minimul este la $x \\approx @{np.sp500.min}\\%$, nu la 0: un impact asimetric al știrilor')],
    h='0.48\\textheight')

D.recap(('Long-Run Behaviour, the VIX and the Leverage Effect', 'comportamentul pe termen lung, VIX și efectul de levier'), [
    T('Volatility is persistent but mean-reverting: half-life of about @{lt.hl} months for the S\\&P 500', 'Volatilitatea este persistentă, dar revine la medie: timp de înjumătățire de circa @{lt.hl} luni la S\\&P 500'),
    T('VIX (implied) is above realised volatility on @{vx.share}\\% of days: a variance risk premium', 'VIX (volatilitatea implicită) este peste volatilitatea realizată în @{vx.share}\\% din zile: o primă de risc a varianței'),
    T('Leverage effect: falls raise future volatility in equity indices; weaker and shorter-lived for Bitcoin', 'Efectul de levier: scăderile cresc volatilitatea viitoare la indicii bursieri; mai slab și de mai scurtă durată la Bitcoin')])

# =============================================================================
# 8. EVALUAREA PROGNOZELOR
# =============================================================================
D.section('Evaluating Volatility Forecasts', 'Evaluarea prognozelor de volatilitate')

D.frame(T('The Proxy Problem', 'Problema variabilei proxy'), items(
    (T('A forecast $h_t$ of $\\sigma_t^2$ is made at the close of day $t-1$; the true $\\sigma_t^2$ is never observed', 'O prognoză $h_t$ a lui $\\sigma_t^2$ se face la închiderea zilei $t-1$; valoarea reală $\\sigma_t^2$ nu este observată niciodată'),
     [T('we compare $h_t$ with a \\textbf{proxy} $\\hat\\sigma_t^2$: a measurable quantity with $E[\\hat\\sigma_t^2 \\mid \\mathcal{F}_{t-1}] = \\sigma_t^2$', 'comparăm $h_t$ cu o variabilă \\textbf{proxy} $\\hat\\sigma_t^2$: o mărime măsurabilă cu $E[\\hat\\sigma_t^2 \\mid \\mathcal{F}_{t-1}] = \\sigma_t^2$')]),
    (T('Proxies, from noisy to precise: $r_t^2$, a range estimator, realised variance', 'Variabile proxy, de la zgomotoase la precise: $r_t^2$, un estimator de amplitudine, varianța realizată'),
     [T('with $r_t^2$ even a perfect forecast looks poor: \\refAB showed that a low $R^2$ does not mean a bad model', 'cu $r_t^2$ chiar și o prognoză perfectă pare slabă: \\refAB au arătat că un $R^2$ mic nu înseamnă un model prost')]),
    (T('\\textbf{Mincer--Zarnowitz regression} \\refMZ: $\\hat\\sigma_t^2 = a + b\\,h_t + u_t$', '\\textbf{Regresia Mincer--Zarnowitz} \\refMZ: $\\hat\\sigma_t^2 = a + b\\,h_t + u_t$'),
     [T('$a$: intercept; $b$: slope; $u_t$: error; a good forecast has $a \\approx 0$, $b \\approx 1$', '$a$: interceptul; $b$: panta; $u_t$: eroarea; o prognoză bună are $a \\approx 0$, $b \\approx 1$'),
      T('the $R^2$ measures how much of the proxy the forecast explains', '$R^2$ măsoară ce parte din variația variabilei proxy este explicată de prognoză')])))

D.frame(T('Loss Functions: MSE and QLIKE', 'Funcții de pierdere: MSE și QLIKE'), items(
    (T('Average loss over the evaluation days; lower is better \\refPatton', 'Pierderea medie pe zilele de evaluare; o valoare mai mică este mai bună \\refPatton'),
     [T('$L$: the loss of one day; $h_t$: the variance forecast; $\\hat\\sigma_t^2$: the proxy', '$L$: pierderea dintr-o zi; $h_t$: prognoza varianței; $\\hat\\sigma_t^2$: variabila proxy'),
      T('MSE: $L = (\\hat\\sigma_t^2 - h_t)^2$', 'MSE: $L = (\\hat\\sigma_t^2 - h_t)^2$'),
      T('\\textbf{QLIKE} (quasi-likelihood): $L = \\hat\\sigma_t^2/h_t + \\ln h_t$, minus the Normal log-likelihood up to constants', '\\textbf{QLIKE} (quasi-likelihood, cvasi-verosimilitate): $L = \\hat\\sigma_t^2/h_t + \\ln h_t$, adică, pînă la constante, opusul log-verosimilității Normale')]),
    (T('Both are \\textbf{robust}: with a noisy but unbiased proxy they rank forecasts as the true variance would', 'Ambele sînt \\textbf{robuste}: cu o variabilă proxy zgomotoasă, dar nedeplasată, ordonează prognozele la fel ca varianța reală'),
     [T('losses on $\\sqrt{h_t}$ or on $\\ln h_t$ are not robust and can prefer the wrong forecast', 'pierderile calculate pe $\\sqrt{h_t}$ sau pe $\\ln h_t$ nu sînt robuste și pot prefera prognoza greșită')]),
    (T('\\textbf{Worked example}: true variance and proxy equal to 1; two forecasts, $h = 0.5$ and $h = 2$', '\\textbf{Exemplu rezolvat}: varianța reală și variabila proxy egale cu 1; două prognoze, $h = 0{,}5$ și $h = 2$'),
     [T('MSE: $(1 - 0.5)^2 = 0.25$ against $(1 - 2)^2 = 1$: MSE punishes over-prediction more', 'MSE: $(1 - 0{,}5)^2 = 0{,}25$ față de $(1 - 2)^2 = 1$: MSE penalizează mai mult supraestimarea'),
      T('QLIKE: $1/0.5 + (@{ql.ln05}) = @{ql.h05}$ against $1/2 + @{ql.ln2} = @{ql.h2}$: QLIKE punishes under-prediction more, the costly error for risk management',
        'QLIKE: $1/0{,}5 + (@{ql.ln05}) = @{ql.h05}$ față de $1/2 + @{ql.ln2} = @{ql.h2}$: QLIKE penalizează mai mult subestimarea, eroarea costisitoare în managementul riscului')])))

D.frame(T('S\\&P 500: Which Forecast Is Best?', 'S\\&P 500: compararea prognozelor'), table(
    'lrrrrrr', T('& \\textbf{21 days} & \\textbf{63 days} & \\textbf{252 days} & \\textbf{EWMA 0.94} & \\textbf{EWMA 0.97} & \\textbf{VIX}',
                 '& \\textbf{21 de zile} & \\textbf{63 de zile} & \\textbf{252 de zile} & \\textbf{EWMA 0,94} & \\textbf{EWMA 0,97} & \\textbf{VIX}'),
    ['MSE & ' + ' & '.join(f'@{{fc.{m}.mse}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']),
     'QLIKE & ' + ' & '.join(f'@{{fc.{m}.ql}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']),
     T('MZ $R^2$, proxy $r^2$', 'MZ $R^2$, proxy $r^2$') + ' & ' + ' & '.join(f'@{{fc.{m}.mzr}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix']),
     T('MZ $R^2$, range proxy', 'MZ $R^2$, proxy de amplitudine') + ' & ' + ' & '.join(f'@{{fc.{m}.mzrr}}' for m in ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix'])],
    size='footnotesize') + items(
    T('Next-day variance forecasts for @{fc.n} days from 2010; proxy $r_t^2$; VIX forecast: VIX$^2/A$; range proxy: $o_t^2$ plus the Garman--Klass term',
      'Prognoze ale varianței pentru ziua următoare, pentru @{fc.n} de zile, începînd din 2010; proxy $r_t^2$; prognoza VIX: VIX$^2/A$; proxy de amplitudine: $o_t^2$ plus termenul Garman--Klass'),
    T('\\textbf{DM} (Diebold--Mariano) test of equal QLIKE \\refDM: EWMA 0.94 against 21 days $t = @{fc.dm_ewma_hist21.t}$ (p @{fc.dm_ewma_hist21.p}); VIX against EWMA 0.94 $t = @{fc.dm_vix_ewma.t}$ (p = @{fc.dm_vix_ewma.p})',
      'Testul \\textbf{DM} (Diebold--Mariano) al egalității QLIKE \\refDM: EWMA 0,94 față de 21 de zile $t = @{fc.dm_ewma_hist21.t}$ (p @{fc.dm_ewma_hist21.p}); VIX față de EWMA 0,94 $t = @{fc.dm_vix_ewma.t}$ (p = @{fc.dm_vix_ewma.p})'),
    T('EWMA beats the 21-day window; VIX and EWMA are not significantly different; the long window is worst; a precise proxy raises every $R^2$, but not the ranking',
      'EWMA este semnificativ mai bun decît fereastra de 21 de zile; VIX și EWMA nu diferă semnificativ; fereastra lungă este cea mai slabă; o variabilă proxy precisă crește fiecare $R^2$, dar nu schimbă ordinea')) + ql('SFM_ch8_forecast_evaluation'), 'footnotesize')

chart(T('Forecasts in 2020', 'Prognozele din 2020'), 'sfm_ch8_forecast_paths', 'SFM_ch8_forecast_evaluation', [
    T('Next-day volatility forecasts of the S\\&P 500 in 2020 (annualised): 252-day window, EWMA ($\\lambda = 0.94$) and VIX; bars: $|r_t|$',
      'Prognozele volatilității S\\&P 500 pentru ziua următoare în 2020 (anualizate): fereastra de 252 de zile, EWMA ($\\lambda = 0{,}94$) și VIX; barele: $|r_t|$'),
    T('VIX started to rise one trading day before the first large fall of February 2020; EWMA rose right after it; the 252-day window moved slowly and stayed near its peak for the rest of the year',
      'VIX a început să crească cu o zi de tranzacționare înaintea primei scăderi mari din februarie 2020; EWMA a crescut imediat după ea; fereastra de 252 de zile s-a mișcat lent și a rămas aproape de vîrf pînă la sfîrșitul anului')],
    h='0.52\\textheight')

chart(T('Choosing $\\lambda$ by QLIKE', 'Alegerea lui $\\lambda$ după QLIKE'), 'sfm_ch8_lambda', 'SFM_ch8_forecast_evaluation', [
    T('Mean QLIKE of next-day EWMA forecasts (proxy $r_t^2$, from 2010; Bitcoin from 2016) minus its minimum, for $\\lambda$ from 0.80 to 0.995',
      'QLIKE mediu al prognozelor EWMA pentru ziua următoare (proxy $r_t^2$, din 2010; Bitcoin din 2016) minus minimul lui, pentru $\\lambda$ între 0,80 și 0,995'),
    T('Best $\\lambda$: S\\&P 500 @{la.sp500.best}, BET @{la.bet.best}, Bitcoin @{la.btc.best}; the RiskMetrics 0.94 loses only @{la.sp500.q94} for the S\\&P 500',
      'Cel mai bun $\\lambda$: S\\&P 500 @{la.sp500.best}, BET @{la.bet.best}, Bitcoin @{la.btc.best}; valoarea RiskMetrics 0,94 are un QLIKE mai mare cu doar @{la.sp500.q94} la S\\&P 500'),
    T('One $\\lambda$ for all assets is a convenient compromise, not an optimum', 'Un singur $\\lambda$ pentru toate activele este un compromis comod, nu un optim')],
    h='0.48\\textheight')

D.recap(('Evaluating Volatility Forecasts', 'evaluarea prognozelor de volatilitate'), [
    T('Compare forecasts with a proxy; $r_t^2$ is unbiased but noisy, so $R^2$ values are low even for good forecasts',
      'Comparăm prognozele cu o variabilă proxy; $r_t^2$ este nedeplasat, dar zgomotos, deci valorile $R^2$ sînt mici chiar și pentru prognoze bune'),
    T('Use robust losses (MSE, QLIKE) and test the differences (Diebold--Mariano); a large comparison of volatility models: \\refHL', 'Folosiți pierderi robuste (MSE, QLIKE) și testați diferențele (Diebold--Mariano); o comparație amplă a modelelor de volatilitate: \\refHL'),
    T('S\\&P 500: EWMA and VIX are the best simple forecasts; long windows are the worst', 'S\\&P 500: EWMA și VIX sînt cele mai bune prognoze simple; ferestrele lungi sînt cele mai slabe')])

# =============================================================================
# 9. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Do range-based estimators help on the Bucharest Stock Exchange, where many stocks trade thinly?}',
       '\\textbf{Ajută estimatorii de amplitudine la Bursa de Valori București, unde multe acțiuni se tranzacționează rar?}'),
     [T('theory: ranges are @{ef.lo} to @{ef.hi} times more efficient; but thin trading shrinks the observed range (Section 4)',
        'teoria: amplitudinile sînt de @{ef.lo}--@{ef.hi} ori mai eficiente; dar tranzacționarea rară micșorează amplitudinea observată (secțiunea 4)'),
      T('days with trades but an unchanged close, since 2010: TLV @{bv.tlv.flat}\\%, SNP @{bv.snp.flat}\\%, BRD @{bv.brd.flat}\\%, TGN @{bv.tgn.flat}\\%',
        'zile cu tranzacții, dar cu închiderea neschimbată, din 2010: TLV @{bv.tlv.flat}\\%, SNP @{bv.snp.flat}\\%, BRD @{bv.brd.flat}\\%, TGN @{bv.tgn.flat}\\%'),
      T('Parkinson / close-to-close volatility: TLV @{bv.tlv.pc}, SNP @{bv.snp.pc}, BRD @{bv.brd.pc}, TGN @{bv.tgn.pc} (S\\&P 500: @{bv.sp500.pc})',
        'raportul volatilităților Parkinson / închidere--închidere: TLV @{bv.tlv.pc}, SNP @{bv.snp.pc}, BRD @{bv.brd.pc}, TGN @{bv.tgn.pc} (S\\&P 500: @{bv.sp500.pc})'),
      T('QLIKE of next-day forecasts by the 21-day window, CC against YZ: TLV @{bv.tlv.qcc} against @{bv.tlv.qyz}; TGN @{bv.tgn.qcc} against @{bv.tgn.qyz}',
        'QLIKE al prognozelor pentru ziua următoare, cu fereastra de 21 de zile, CC față de YZ: TLV @{bv.tlv.qcc} față de @{bv.tlv.qyz}; TGN @{bv.tgn.qcc} față de @{bv.tgn.qyz}')]),
    T('Open: the answer differs by stock; is it liquidity, the opening auction, or the size of the overnight move?',
      'Întrebarea rămîne deschisă: răspunsul diferă de la o acțiune la alta; contează lichiditatea, licitația de deschidere sau mărimea mișcării overnight?'),
    T('AI tools can speed up such a study, but its results must still be checked \\refWang', 'Instrumentele AI pot accelera un astfel de studiu, dar rezultatele lui trebuie verificate \\refWang')) + ql('SFM_ch8_bvb_range'))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: list studies of range-based and realised volatility in emerging and thin markets, with their data and estimators',
      '\\textbf{Literatura}: o listă a studiilor despre volatilitatea de amplitudine și cea realizată pe piețe emergente și puțin lichide, cu datele și estimatorii folosiți'),
    T('\\textbf{Code}: a first draft of the five estimators, rolling forecasts and QLIKE comparisons for all BET stocks',
      '\\textbf{Cod}: o primă versiune a celor cinci estimatori, a prognozelor pe ferestre mobile și a comparațiilor QLIKE pentru toate acțiunile din BET'),
    T('\\textbf{Robustness}: liquidity groups by volume, other windows, the 2016--2020 and 2021--2026 halves',
      '\\textbf{Robustețe}: grupe de lichiditate după volum, alte ferestre, cele două jumătăți 2016--2020 și 2021--2026'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that computes the Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang variance on rolling 21-day windows from daily open, high, low and close prices, and compares their next-day forecasts with QLIKE against squared close-to-close returns.}',
        '\\aiprompt{Write a Python function that computes the Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang variance on rolling 21-day windows from daily open, high, low and close prices, and compares their next-day forecasts with QLIKE against squared close-to-close returns.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The constants: $4\\ln 2$ in Parkinson, $2\\ln 2 - 1$ in Garman--Klass, $k$ in Yang--Zhang; a draft with $\\ln 2$ or $k = 0.34$ is wrong',
      'Constantele: $4\\ln 2$ la Parkinson, $2\\ln 2 - 1$ la Garman--Klass, $k$ la Yang--Zhang; o primă versiune cu $\\ln 2$ sau cu $k = 0{,}34$ este greșită'),
    T('Stock prices: open, high and low must be adjusted for dividends and splits with the same factor as the close',
      'Prețurile acțiunilor: deschiderea, maximul și minimul trebuie ajustate pentru dividende și splituri cu același factor ca închiderea'),
    T('Annualisation with the actual frequency: 365 for crypto, about 250 for BVB stocks', 'Anualizarea cu frecvența reală: 365 pentru cripto, circa 250 pentru acțiunile BVB'),
    T('No look-ahead: the forecast for day $t$ uses only data up to day $t-1$', 'Fără informație din viitor: prognoza pentru ziua $t$ folosește doar datele pînă în ziua $t-1$'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: which daily volatility estimator should a Romanian risk manager use?', '\\textbf{Întrebarea}: ce estimator zilnic de volatilitate ar trebui să folosească un manager de risc din România?'),
     [T('data: BVB stocks since 2010 (TLV, SNP, BRD, TGN, and others with OHLC data), S\\&P 500 and DAX as benchmarks (EODHD)',
        'date: acțiuni BVB din 2010 (TLV, SNP, BRD, TGN și altele cu date OHLC), S\\&P 500 și DAX ca termeni de comparație (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('the five estimators on 21-day windows; EWMA versions of each; next-day forecasts', 'cei cinci estimatori pe ferestre de 21 de zile; variantele EWMA ale fiecăruia; prognoze pentru ziua următoare'),
      T('QLIKE and MSE with the proxy $r_t^2$; Diebold--Mariano tests; results by liquidity group', 'QLIKE și MSE cu proxy-ul $r_t^2$; teste Diebold--Mariano; rezultate pe grupe de lichiditate'),
      T('relate the gain of YZ over CC to the share of zero-volume days and to the overnight share', 'legați cîștigul YZ față de CC de ponderea zilelor fără volum și de ponderea nopții')]),
    T('Deliverables: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabile: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Volatility is the standard deviation of returns; it is latent and changes over time; annualise with the actual frequency',
      'Volatilitatea este abaterea standard a randamentelor; este latentă și variază în timp; se anualizează cu frecvența reală'),
    T('Historical windows trade speed against noise; EWMA ($\\lambda = 0.94$) reacts fast and avoids ghosts', 'La ferestrele istorice, viteza de reacție se plătește cu zgomot; EWMA ($\\lambda = 0{,}94$) reacționează repede și evită ghost effect'),
    T('Range estimators use the daily path: far more efficient, but biased by overnight jumps and thin trading; Yang--Zhang adds the night',
      'Estimatorii de amplitudine folosesc traiectoria zilei: mult mai eficienți, dar deplasați de salturile overnight și de tranzacționarea rară; Yang--Zhang adaugă varianța nopții'),
    T('Clustering: $|r_t|$ and $r_t^2$ are autocorrelated; detect it with Ljung--Box on $r_t^2$ and ARCH-LM', 'Volatility clustering: $|r_t|$ și $r_t^2$ sînt autocorelate; se detectează cu Ljung--Box pe $r_t^2$ și cu ARCH-LM'),
    T('Volatility mean-reverts; VIX usually exceeds realised volatility; falls raise future volatility', 'Volatilitatea revine la medie; VIX depășește de obicei volatilitatea realizată; scăderile cresc volatilitatea viitoare'),
    T('Compare forecasts with robust losses (QLIKE) and a proxy; test the differences', 'Comparați prognozele cu pierderi robuste (QLIKE) și o variabilă proxy; testați diferențele')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Annualised volatility & $\\sigma_{\\text{year}} = \\sigma_{\\text{day}}\\sqrt{A}$', 'Volatilitatea anualizată & $\\sigma_{\\text{an}} = \\sigma_{\\text{zi}}\\sqrt{A}$'),
     'EWMA & $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1-\\lambda)r_{t-1}^2$, \\quad ' + T('half-life', 'timp de înjumătățire') + ' $\\ln 0.5/\\ln\\lambda$',
     'Parkinson & $(u_t - d_t)^2/(4\\ln 2)$',
     'Garman--Klass & $0.5(u_t - d_t)^2 - (2\\ln 2 - 1)c_t^2$',
     'Rogers--Satchell & $u_t(u_t - c_t) + d_t(d_t - c_t)$',
     'Yang--Zhang & $\\hat\\sigma_O^2 + k\\hat\\sigma_C^2 + (1-k)\\hat\\sigma_{RS}^2$, \\quad $k = 0.34/(1.34 + (n+1)/(n-1))$',
     T('Realised variance', 'Varianța realizată') + ' & $\\mathrm{RV}_t = \\sum_i r_{t,i}^2$',
     'McLeod--Li & $Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k(r^2)^2/(T-k) \\sim \\chi^2(m)$',
     'ARCH-LM & $\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$',
     'MSE, QLIKE & $(\\hat\\sigma_t^2 - h_t)^2$, \\quad $\\hat\\sigma_t^2/h_t + \\ln h_t$'],
    size='small') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a stock has $H = 102$, $L = 98$ on one day; what is the Parkinson daily volatility?', '\\textbf{Întrebare}: o acțiune are $H = 102$, $L = 98$ într-o zi; care este volatilitatea zilnică Parkinson?'),
     [T('\\textbf{Answer}: $\\ln(102/98) = @{cy.hl}\\%$; $\\sqrt{@{cy.hl}^2/@{cy.c}} = @{cy.vol}\\%$', '\\textbf{Răspuns}: $\\ln(102/98) = @{cy.hl}\\%$; $\\sqrt{@{cy.hl}^2/@{cy.c}} = @{cy.vol}\\%$')]),
    (T('\\textbf{Question}: Ljung--Box on returns does not reject, on squared returns it rejects strongly; what do you conclude?',
       '\\textbf{Întrebare}: testul Ljung--Box pe randamente nu respinge, iar pe randamentele la pătrat respinge puternic; ce concluzie trageți?'),
     [T('\\textbf{Answer}: the mean is unpredictable, the variance is predictable: volatility clustering without autocorrelation',
        '\\textbf{Răspuns}: media este imprevizibilă, varianța este previzibilă: volatility clustering fără autocorelația randamentelor')]),
    (T('\\textbf{Question}: why can EWMA with $\\lambda = 0.94$ not forecast a return of volatility to its normal level?',
       '\\textbf{Întrebare}: de ce nu poate EWMA cu $\\lambda = 0{,}94$ să anticipeze revenirea volatilității la nivelul normal?'),
     [T('\\textbf{Answer}: $\\alpha + \\beta = 1$ and $\\omega = 0$: its forecasts are flat at every horizon', '\\textbf{Răspuns}: $\\alpha + \\beta = 1$ și $\\omega = 0$: prognozele lui sînt constante la orice orizont')]),
    T('Next: Chapter 9, ARCH and GARCH models', 'Urmează: Capitolul 9, modelele ARCH și GARCH')))

D.references(BIB)

if __name__ == '__main__':
    fix_refs(D)
    D.write(V)
