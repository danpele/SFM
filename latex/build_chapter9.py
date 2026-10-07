r"""
build_chapter9.py -- Capitolul 9 (Modele ARCH și GARCH), EN + RO dintr-o singură sursă
=====================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_09/ch9_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter9_arch_garch_models.tex
  RO/Cursuri/capitol9_modele_arch_garch.tex
Rulare:
  python3 Quantlets/Ch_09/generate_all_charts.py
  python3 latex/build_chapter9.py && python3 latex/sfm_build.py compile 9
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch9_common import ASSETS, BIB, NAMES, QLURL, REFS, SHORT, T, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(9, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_09}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'engle': ('ch9_engle_2022.jpg', C + '0603-Kraneshares_KRBN-RobertEngle-JonDemske-16_(cropped).jpg',
              '⟦Photo||Foto⟧: Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'stockholm': ('ch9_stockholm_concert_hall.jpg', C + 'Konserthuset_Stockholm_(Stockholm_Concert_Hall).jpg',
                  '⟦Photo||Foto⟧: Karen Zhou (2025); CC BY-SA 4.0; Wikimedia Commons'),
    'ucsd': ('ch9_ucsd_geisel_2010.jpg', C + 'Geisel_Library,_UC_San_Diego.jpg',
             '⟦Photo||Foto⟧: Stephen Bay (2010); CC BY 4.0; Wikimedia Commons'),
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
# ARCH(1): omega = 0.4, alpha = 0.6 ... kurtosis needs 3 alpha^2 < 1
w1, a1 = 0.5, 0.5
V.put('ex.a1.uv', w1 / (1 - a1), 2)
V.put('ex.a1.k', 3 * (1 - a1 ** 2) / (1 - 3 * a1 ** 2), 1)
V.put('ex.a1.s2', w1 + a1 * 2.0 ** 2, 2)
V.put('ex.a1.s', math.sqrt(w1 + a1 * 2.0 ** 2), 2)
V.put('ex.a1.lim', 1 / math.sqrt(3), 3)
# GARCH(1,1): omega = 0.02, alpha = 0.10, beta = 0.88 (daily returns in %)
w, a, b = 0.02, 0.10, 0.88
V.put('ex.g.pers', a + b, 2)
V.put('ex.g.uv', w / (1 - a - b), 2)
V.put('ex.g.vol', math.sqrt(252 * w / (1 - a - b)), 1)
V.put('ex.g.hl', math.log(0.5) / math.log(a + b), 1)
V.put('ex.g.s2', w + a * 3.0 ** 2 + b * 1.2, 3)
V.put('ex.g.s', math.sqrt(w + a * 3.0 ** 2 + b * 1.2), 2)
V.put('ex.g.k', 3 * (1 - (a + b) ** 2) / (1 - (a + b) ** 2 - 2 * a ** 2), 2)
V.put('ex.g.cond', (a + b) ** 2 + 2 * a ** 2, 4)
# from the S&P 500 estimates (GARCH(1,1)-t)
E = N['est']['t']['params']
V.put('ex.sp.pers', E['alpha[1]'] + E['beta[1]'], 4)
V.put('ex.sp.lnp', math.log(E['alpha[1]'] + E['beta[1]']), 5)
# multi-step forecast example (term structure, last day)
TS = N['term']
last = sorted(k for k in TS if k[:2] == '20')[-1]
pp = TS['params']
pers = pp['alpha[1]'] + pp['beta[1]']
V.put('fc.pers', pers, 4)
V.put('fc.pers9', pers ** 9, 3)
V.put('fc.s2n', TS[last]['s2next'], 3)
V.put('fc.s2bar', TS['s2bar'], 3)
V.put('fc.s210', TS['s2bar'] + pers ** 9 * (TS[last]['s2next'] - TS['s2bar']), 3)
V.put('fc.sum10', TS[last]['sum10'], 2)
V.put('fc.vol10', math.sqrt(TS[last]['sum10']), 2)
V.put('fc.vol10sq', math.sqrt(10 * TS[last]['s2next']), 2)
# VaR 1% example
nu = pp['nu']
qt = stats.t.ppf(0.01, nu) * math.sqrt((nu - 2) / nu)
V.put('var.nu', nu, 2)
V.put('var.t', stats.t.ppf(0.01, nu), 3)
V.put('var.sc', math.sqrt((nu - 2) / nu), 3)
V.put('var.qt', qt, 3)
V.put('var.qa', -qt, 3)
V.put('var.qn', stats.norm.ppf(0.01), 3)
V.put('var.mu', pp['mu'], 3)
V.put('var.sig', math.sqrt(TS[last]['s2next']), 3)
V.put('var.v1', -(pp['mu'] + math.sqrt(TS[last]['s2next']) * qt), 2)
V.put('var.vn', -(pp['mu'] + math.sqrt(TS[last]['s2next']) * stats.norm.ppf(0.01)), 2)
V.put('var.v10', -qt * math.sqrt(TS[last]['sum10']), 2)
# ranges and ratios used in the text
M = N['markets']
ses = [M[k]['se'][c] for k in ASSETS for c in ('alpha[1]', 'beta[1]')]
V.put('m.se.min', min(ses), 3)
V.put('m.se.max', max(ses), 3)
V.put('m.nu.min', min(M[k]['params']['nu'] for k in ASSETS), 1)
V.put('m.nu.max', max(M[k]['params']['nu'] for k in ASSETS), 1)
EN_ = N['est']['normal']
V.put('se.ratio', sum(EN_['se'][c] / EN_['se_classic'][c] for c in ('alpha[1]', 'beta[1]')) / 2, 1)
V.put('se.ratio.om', EN_['se']['omega'] / EN_['se_classic']['omega'], 1)
for pp_ in (0.90, 0.98, 0.99):
    V.put(f'hl.{int(round(100 * pp_))}', math.log(0.5) / math.log(pp_), 1)
FE = N['fe']
V.put('fe.eg.min', min(100 * FE[k]['exc_garch'] for k in FE), 1)
V.put('fe.eg.max', max(100 * FE[k]['exc_garch'] for k in FE), 1)
V.put('fe.ee.min', min(100 * FE[k]['exc_ewma'] for k in FE), 1)
V.put('fe.ee.max', max(100 * FE[k]['exc_ewma'] for k in FE), 1)
V.put('v.ratio', N['vol2']['btc']['median'] / N['vol1']['sp500']['median'], 1)
V.put('ret.r08', N['returns']['sd2008'] / N['returns']['sd2017'], 1)
MSd = {(r['model'], r['dist']): r['bic'] for r in N['ms']}
V.put('ms.gain.asym', MSd[('GARCH', 'Normal')] - MSd[('EGARCH', 'Normal')], 0)
V.put('ms.gain.dist', MSd[('GARCH', 'Normal')] - MSd[('GARCH', 'skewed t')], 0)
V.put('ms.gain.both', MSd[('GARCH', 'Normal')] - MSd[('EGARCH', 'skewed t')], 0)
V.put('ltv.calm', math.sqrt(0.5), 2)
V.put('ltv.storm', math.sqrt(4.5), 2)
# law of total variance example
V.put('ltv.v', 0.5 * 0.5 + 0.5 * 4.5, 2)
V.put('ltv.s', math.sqrt(0.5 * 0.5 + 0.5 * 4.5), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: if tomorrow\'s return cannot be predicted, can tomorrow\'s \\textbf{risk} be predicted?',
       '\\textbf{Întrebarea}: dacă randamentul de mîine nu poate fi anticipat, poate fi anticipat \\textbf{riscul} de mîine?'),
     [T('Chapter 7: daily returns are almost uncorrelated; Chapter 8: their squares are strongly correlated (volatility clustering)',
        'Capitolul 7: randamentele zilnice sînt aproape necorelate; Capitolul 8: pătratele lor sînt puternic corelate (volatility clustering)'),
      T('this chapter turns volatility clustering into a model for the conditional variance', 'acest capitol transformă volatility clustering într-un model pentru varianța condiționată')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('conditional mean and conditional variance; ARCH($q$) and GARCH(1,1)', 'media condiționată și varianța condiționată; ARCH($q$) și GARCH(1,1)'),
      T('estimation by maximum likelihood; Student-t and skewed-t innovations', 'estimarea prin verosimilitate maximă; inovații Student-t și t asimetrice'),
      T('six markets; asymmetry: GJR-GARCH, EGARCH, the news impact curve', 'șase piețe; asimetrie: GJR-GARCH, EGARCH, curba de impact a știrilor'),
      T('diagnostics, model choice, forecasts, VaR 1\\%', 'diagnostic, alegerea modelului, prognoze, VaR 1\\%')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Distinguish the conditional variance from the unconditional variance and explain why returns can be uncorrelated yet dependent',
      'Deosebiți varianța condiționată de varianța necondiționată și explicați de ce randamentele pot fi necorelate, dar dependente'),
    T('Write the ARCH($q$) and GARCH(1,1) models and compute the persistence, the unconditional variance and the half-life',
      'Scrieți modelele ARCH($q$) și GARCH(1,1) și calculați persistența, varianța necondiționată și timpul de înjumătățire'),
    T('Estimate GARCH models by maximum likelihood, read the standard errors and choose the innovation distribution',
      'Estimați modele GARCH prin verosimilitate maximă, interpretați erorile standard și alegeți distribuția inovațiilor'),
    T('Measure the leverage effect with GJR-GARCH, EGARCH and the news impact curve', 'Măsurați efectul de levier cu GJR-GARCH, EGARCH și curba de impact a știrilor'),
    T('Check a fitted model with standardised residuals and choose among models with AIC and BIC', 'Verificați un model estimat cu reziduurile standardizate și alegeți între modele cu AIC și BIC'),
    T('Forecast volatility over several horizons, compare forecasts with QLIKE and compute a one-day VaR 1\\%',
      'Prognozați volatilitatea pe mai multe orizonturi, comparați prognozele cu QLIKE și calculați un VaR 1\\% pe o zi')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~13 (13.1 ARCH and GARCH models, 13.2 extensions of the GARCH model)',
       'Manual: \\refFHH, cap.~13 (13.1 modelele ARCH și GARCH, 13.2 extensii ale modelului GARCH)'),
     [T('exercises with solutions: \\refBHL, Ch.~13', 'exerciții rezolvate: \\refBHL, cap.~13')]),
    T('Conditional heteroskedastic models: \\refTsay, Ch.~3', 'Modele cu heteroscedasticitate condiționată: \\refTsay, cap.~3'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_09}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_09}'),
     [T('ported from the SFE Quantlets SFEtimegarch, SFElikarch1, SFElikgarch, SFEgarchest, SFEvolgarchest and SFENewsImpactCurve',
        'portate din Quantlet-urile SFE SFEtimegarch, SFElikarch1, SFElikgarch, SFEgarchest, SFEvolgarchest și SFENewsImpactCurve'),
      T('estimation with the Python package \\texttt{arch}', 'estimarea cu pachetul Python \\texttt{arch}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter9_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter9_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. MOTIVAȚIE
# =============================================================================
D.section('Volatility Comes in Bursts', 'Volatilitatea apare în episoade')

chart(T('S\\&P 500: Calm Years and Storms', 'S\\&P 500: ani liniștiți și perioade turbulente'), 'sfm_ch9_returns', 'SFM_ch9_volatility_markets', [
    T('Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$, in \\%, @{ret.n} days to @{end}; the largest fall: $@{ret.min}\\%$ on @{ret.dmin}; the largest rise: $@{ret.max}\\%$ on @{ret.dmax}',
      'Randamente logaritmice zilnice $r_t = 100(\\ln P_t - \\ln P_{t-1})$, în \\%, @{ret.n} zile pînă la @{end}; cea mai mare scădere: $@{ret.min}\\%$ pe @{ret.dmin}; cea mai mare creștere: $@{ret.max}\\%$ pe @{ret.dmax}'),
    T('Standard deviation: $@{ret.sd2017}\\%$ in 2017, $@{ret.sd2008}\\%$ from September 2008 to March 2009, $@{ret.sd2020}\\%$ from February to May 2020',
      'Abaterea standard: $@{ret.sd2017}\\%$ în 2017, $@{ret.sd2008}\\%$ din septembrie 2008 pînă în martie 2009, $@{ret.sd2020}\\%$ din februarie pînă în mai 2020'),
    T('Large changes follow large changes, of either sign: volatility clustering \\refEngle', 'Variațiile mari urmează după variații mari, de orice semn: volatility clustering \\refEngle')],
    h='0.50\\textheight')

D.frame(T('Two Storms: 2008 and 2020', 'Două crize: 2008 și 2020'), cols(
    ph('lehman', T('Lehman Brothers headquarters, New York, 15 September 2008', 'Sediul Lehman Brothers, New York, 15 septembrie 2008'), h='0.30\\textheight'),
    ph('fidi', T('The Financial District of New York, 25 March 2020', 'Districtul financiar din New York, 25 martie 2020'), h='0.30\\textheight'),
    wl='0.48', wr='0.48') + items(
    T('15 September 2008: Lehman Brothers files for bankruptcy; March 2020: the COVID-19 pandemic closes economies',
      '15 septembrie 2008: Lehman Brothers intră în faliment; martie 2020: pandemia COVID-19 duce la închiderea economiilor'),
    T('In both episodes the daily standard deviation of the S\\&P 500 was @{ret.r08} and @{ret.ratio} times that of 2017; calm returned only after months',
      'În ambele episoade abaterea standard zilnică a S\\&P 500 a fost de @{ret.r08} și, respectiv, de @{ret.ratio} ori mai mare decît în 2017; calmul a revenit abia după cîteva luni')), 'footnotesize')

D.frame(T('Why Model the Conditional Variance?', 'Utilitatea modelării varianței condiționate'), items(
    (T('\\textbf{Risk management}: VaR (value at risk) and ES (expected shortfall) need tomorrow\'s volatility, not the average volatility of the last 20 years',
       '\\textbf{Managementul riscului}: VaR (value at risk, valoarea expusă la risc) și ES (expected shortfall) au nevoie de volatilitatea de mîine, nu de volatilitatea medie din ultimii 20 de ani'),
     [T('a constant-variance VaR is too large in calm years and too small in crises (Chapter 10)', 'un VaR cu varianță constantă este prea mare în anii liniștiți și prea mic în crize (Capitolul 10)')]),
    T('\\textbf{Portfolios}: weights depend on variances and covariances, which change over time', '\\textbf{Portofolii}: ponderile depind de varianțe și covarianțe, care se schimbă în timp'),
    T('\\textbf{Derivatives}: option prices depend on the volatility expected until maturity', '\\textbf{Derivate}: prețurile opțiunilor depind de volatilitatea așteptată pînă la scadență'),
    (T('\\textbf{Inference}: tests and confidence intervals that assume a constant variance can be wrong', '\\textbf{Inferență}: testele și intervalele de încredere care presupun varianță constantă pot fi greșite'),
     [T('this is why Chapter 7 needed robust tests ($\\tilde Q$, $Z^*$)', 'de aceea Capitolul 7 a avut nevoie de teste robuste ($\\tilde Q$, $Z^*$)')])))

D.frame(T('Stylised Facts a Volatility Model Must Reproduce', 'Faptele stilizate pe care un model de volatilitate trebuie să le reproducă'), items(
    (T('\\textbf{Volatility clustering}: the ACF (autocorrelation function) of $r_t^2$ is positive and decays slowly (Chapter 8)',
       '\\textbf{Volatility clustering}: ACF (autocorrelation function, funcția de autocorelație) a lui $r_t^2$ este pozitivă și scade lent (Capitolul 8)'),
     [T('S\\&P 500: autocorrelation of $r_t^2$ at lag 1: $@{acf.r1}$; at lag 50: still $@{acf.r50}$', 'S\\&P 500: autocorelația lui $r_t^2$ la lagul 1: $@{acf.r1}$; la lagul 50: încă $@{acf.r50}$')]),
    (T('\\textbf{Heavy tails}: kurtosis above 3, the value of the Normal distribution (Chapter 2)', '\\textbf{Cozi groase}: coeficient de boltire peste 3, valoarea distribuției Normale (Capitolul 2)'),
     [T('S\\&P 500 daily returns: kurtosis $@{kurt.r}$', 'randamentele zilnice S\\&P 500: coeficientul de boltire $@{kurt.r}$')]),
    T('\\textbf{Leverage effect}: falls raise volatility more than rises of the same size \\refChristie', '\\textbf{Efectul de levier}: scăderile cresc volatilitatea mai mult decît creșterile de aceeași mărime \\refChristie'),
    T('\\textbf{Mean reversion of volatility}: after a storm, volatility returns to a long-run level', '\\textbf{Revenirea volatilității la medie}: după o perioadă turbulentă, volatilitatea revine la un nivel pe termen lung'),
    T('Almost no autocorrelation in $r_t$ itself (Chapter 7)', 'Aproape nicio autocorelație în $r_t$ însuși (Capitolul 7)')))

D.recap(('Volatility Comes in Bursts', 'volatilitatea apare în episoade'), [
    T('Calm periods and storms alternate; a storm lasts for months', 'Perioadele liniștite alternează cu perioadele turbulente; o perioadă turbulentă durează luni'),
    T('Risk measures, portfolios, option prices and tests need the volatility of tomorrow', 'Măsurile de risc, portofoliile, prețurile opțiunilor și testele au nevoie de volatilitatea de mîine'),
    T('A good model: clustering, heavy tails, leverage effect, mean reversion of volatility', 'Un model bun: clustering, cozi groase, efect de levier, revenirea volatilității la medie')])

# =============================================================================
# 2. MEDIA ȘI VARIANȚA CONDIȚIONATĂ
# =============================================================================
D.section('Conditional Mean and Conditional Variance', 'Media condiționată și varianța condiționată')

D.frame(T('Conditioning on the Past', 'Condiționarea pe trecut'), items(
    (T('$\\mathcal{F}_{t-1}$: the information available at the end of day $t-1$ (past returns)', '$\\mathcal{F}_{t-1}$: informația disponibilă la sfîrșitul zilei $t-1$ (randamentele trecute)'),
     [T('conditional expectation $E[X \\mid \\mathcal{F}_{t-1}]$: the best forecast of $X$ given the past (Chapter 4)', 'speranța condiționată $E[X \\mid \\mathcal{F}_{t-1}]$: cea mai bună prognoză a lui $X$, dată fiind informația trecută (Capitolul 4)')]),
    (T('\\textbf{Conditional mean}: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$', '\\textbf{Media condiționată}: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$'),
     [T('Chapter 7: for daily returns, $\\mu_t$ is almost constant: $\\mu_t = \\mu$', 'Capitolul 7: pentru randamentele zilnice, $\\mu_t$ este aproape constantă: $\\mu_t = \\mu$')]),
    (T('\\textbf{Conditional variance}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1}) = E[(r_t - \\mu_t)^2 \\mid \\mathcal{F}_{t-1}]$',
       '\\textbf{Varianța condiționată}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1}) = E[(r_t - \\mu_t)^2 \\mid \\mathcal{F}_{t-1}]$'),
     [T('$\\sigma_t$ is the \\textbf{conditional volatility}: known at $t-1$, it changes from day to day', '$\\sigma_t$ este \\textbf{volatilitatea condiționată}: cunoscută la $t-1$, se schimbă de la o zi la alta'),
      T('\\textbf{heteroskedasticity}: a variance that is not constant', '\\textbf{heteroscedasticitate}: o varianță care nu este constantă')])))

D.frame(T('The Basic Decomposition', 'Descompunerea de bază'), items(
    (T('Every model of this chapter has the form $r_t = \\mu + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$',
       'Toate modelele din acest capitol au forma $r_t = \\mu + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$'),
     [T('$z_t$: the \\textbf{innovations}, i.i.d.\\ with mean 0 and variance 1 (Normal, Student-t, \\dots)', '$z_t$: \\textbf{inovațiile}, i.i.d.\\ cu media 0 și varianța 1 (Normale, Student-t, \\dots)'),
      T('$\\varepsilon_t$: the \\textbf{shock} (the return minus its mean); $\\sigma_t$ depends only on the past', '$\\varepsilon_t$: \\textbf{șocul} (randamentul minus media lui); $\\sigma_t$ depinde doar de trecut')]),
    (T('Consequences', 'Consecințe'),
     [T('$E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = \\sigma_t E[z_t] = 0$: the shocks are a martingale difference, hence uncorrelated', '$E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = \\sigma_t E[z_t] = 0$: șocurile sînt o diferență de martingal, deci sînt necorelate'),
      T('$\\mathrm{Var}(\\varepsilon_t \\mid \\mathcal{F}_{t-1}) = \\sigma_t^2$: the variance is predictable', '$\\mathrm{Var}(\\varepsilon_t \\mid \\mathcal{F}_{t-1}) = \\sigma_t^2$: varianța este previzibilă'),
      T('the $\\varepsilon_t$ are uncorrelated but not independent: $\\varepsilon_t^2$ is correlated with $\\varepsilon_{t-1}^2$', '$\\varepsilon_t$ sînt necorelate, dar nu independente: $\\varepsilon_t^2$ este corelat cu $\\varepsilon_{t-1}^2$')]),
    T('In the terms of Chapter 7: RW1 is rejected, the martingale (RW3) is kept', 'În termenii Capitolului 7: RW1 este respinsă, martingalul (RW3) rămîne valabil')))

D.frame(T('Worked Example: Conditional and Unconditional Variance', 'Exemplu rezolvat: varianța condiționată și varianța necondiționată'), items(
    (T('A market has calm days ($\\sigma_t^2 = 0.5$) and stormy days ($\\sigma_t^2 = 4.5$), each with probability $1/2$; the mean is 0',
       'O piață are zile liniștite ($\\sigma_t^2 = 0{,}5$) și zile agitate ($\\sigma_t^2 = 4{,}5$), fiecare cu probabilitatea $1/2$; media este 0'),
     [T('law of total variance: $\\mathrm{Var}(r_t) = E[\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})] + \\mathrm{Var}(E[r_t \\mid \\mathcal{F}_{t-1}])$',
        'legea varianței totale: $\\mathrm{Var}(r_t) = E[\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})] + \\mathrm{Var}(E[r_t \\mid \\mathcal{F}_{t-1}])$'),
      T('step by step: $0.5 \\times 0.5 + 0.5 \\times 4.5 + 0 = @{ltv.v}$, so the unconditional standard deviation is $@{ltv.s}\\%$',
        'pas cu pas: $0.5 \\times 0.5 + 0.5 \\times 4.5 + 0 = @{ltv.v}$, deci abaterea standard necondiționată este $@{ltv.s}\\%$')]),
    (T('The unconditional variance is an average; the conditional variance tells which day we are in', 'Varianța necondiționată este o medie; varianța condiționată ne spune în ce fel de zi ne aflăm'),
     [T('a calm day: $\\sigma_t = @{ltv.calm}\\%$; a stormy day: $\\sigma_t = @{ltv.storm}\\%$; the average $@{ltv.s}\\%$ fits neither', 'o zi liniștită: $\\sigma_t = @{ltv.calm}\\%$; o zi agitată: $\\sigma_t = @{ltv.storm}\\%$; media de $@{ltv.s}\\%$ nu descrie niciuna dintre ele')]),
    T('A mixture of Normal distributions with different variances has heavy tails: conditional heteroskedasticity creates kurtosis',
      'Un amestec de distribuții Normale cu varianțe diferite are cozi groase: heteroscedasticitatea condiționată creează boltire')))

D.recap(('Conditional Mean and Conditional Variance', 'media condiționată și varianța condiționată'), [
    T('$r_t = \\mu + \\sigma_t z_t$: constant mean, conditional volatility $\\sigma_t$ known one day ahead', '$r_t = \\mu + \\sigma_t z_t$: medie constantă, volatilitate condiționată $\\sigma_t$ cunoscută cu o zi înainte'),
    T('Shocks are uncorrelated but dependent through their squares', 'Șocurile sînt necorelate, dar dependente prin pătratele lor'),
    T('Unconditional variance = average of the conditional variances', 'Varianța necondiționată = media varianțelor condiționate')])

# =============================================================================
# 3. ARCH
# =============================================================================
D.section('The ARCH Model', 'Modelul ARCH')

D.frame(T('1982: Robert Engle and ARCH', '1982: Robert Engle și modelul ARCH'), cols(items(
    (T('\\refEngle: \\textbf{ARCH} (autoregressive conditional heteroskedasticity): today\'s variance depends on yesterday\'s squared shocks',
       '\\refEngle: \\textbf{ARCH} (autoregressive conditional heteroskedasticity, heteroscedasticitate condiționată autoregresivă): varianța de azi depinde de pătratele șocurilor de ieri'),
     [T('first application: the variance of inflation in the United Kingdom; written while Engle was visiting the London School of Economics',
        'prima aplicație: varianța inflației din Regatul Unit; scris în timpul vizitei lui Engle la London School of Economics')]),
    (T('\\href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{Nobel Prize 2003}: ``for methods of analyzing economic time series with time-varying volatility (ARCH)\'\'',
       '\\href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{Premiul Nobel 2003}: „pentru metode de analiză a seriilor de timp economice cu volatilitate variabilă în timp (ARCH)”'),
     [T('shared with Clive Granger (cointegration); Nobel lecture: \\refEngleN', 'împărțit cu Clive Granger (cointegrare); prelegerea Nobel: \\refEngleN')])),
    ph('engle', T('Robert F. Engle (2022)', 'Robert F. Engle (2022)'), h='0.25\\textheight') + '\\\\[1mm]\n' +
    ph('stockholm', T('Stockholm Concert Hall, venue of the Nobel Prize ceremony', 'Sala de concerte din Stockholm, unde se decernează Premiile Nobel'), h='0.16\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('ARCH(1): Definition', 'ARCH(1): definiție'), items(
    (T('$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$',
       '$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$'),
     [T('in words: today\'s variance = a constant plus a share $\\alpha$ of yesterday\'s squared shock', 'în cuvinte: varianța de azi = o constantă plus o fracțiune $\\alpha$ din pătratul șocului de ieri'),
      T('$\\omega$: the variance after a zero shock; $\\alpha$: the reaction to yesterday\'s shock; both keep the variance positive', '$\\omega$: varianța după un șoc nul; $\\alpha$: reacția la șocul de ieri; restricțiile asigură o varianță pozitivă'),
      T('a large shock yesterday, of either sign, raises today\'s variance', 'un șoc mare ieri, indiferent de semn, crește varianța de azi')]),
    (T('``Autoregressive\'\': $\\varepsilon_t^2$ follows an AR(1) (autoregressive) model', '„Autoregresiv”: $\\varepsilon_t^2$ urmează un model AR(1) (autoregresiv)'),
     [T('write $\\varepsilon_t^2 = \\sigma_t^2 + v_t$ with $v_t = \\sigma_t^2(z_t^2 - 1)$, $E[v_t \\mid \\mathcal{F}_{t-1}] = 0$', 'scriem $\\varepsilon_t^2 = \\sigma_t^2 + v_t$, cu $v_t = \\sigma_t^2(z_t^2 - 1)$, $E[v_t \\mid \\mathcal{F}_{t-1}] = 0$'),
      T('then $\\varepsilon_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + v_t$: an AR(1) for the squared shocks', 'atunci $\\varepsilon_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + v_t$: un AR(1) pentru pătratele șocurilor')]),
    T('Hence the ACF of $\\varepsilon_t^2$ is $\\rho(k) = \\alpha^k$ when the fourth moment exists: clustering, but a fast decay',
      'De aici, ACF a lui $\\varepsilon_t^2$ este $\\rho(k) = \\alpha^k$ cînd momentul de ordinul patru există: clustering, dar o scădere rapidă')))

D.frame(T('ARCH(1): Properties', 'ARCH(1): proprietăți'), items(
    T('Mean and autocorrelation: $E[\\varepsilon_t] = 0$, $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ for $k \\ne 0$', 'Media și autocorelația: $E[\\varepsilon_t] = 0$, $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ pentru $k \\ne 0$'),
    (T('\\textbf{Unconditional variance} (if $\\alpha < 1$): take expectations in $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$',
       '\\textbf{Varianța necondiționată} (dacă $\\alpha < 1$): aplicăm media în $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$'),
     [T('$\\bar\\sigma^2 = E[\\varepsilon_t^2] = \\omega + \\alpha\\bar\\sigma^2$, so $\\bar\\sigma^2 = \\omega/(1 - \\alpha)$', '$\\bar\\sigma^2 = E[\\varepsilon_t^2] = \\omega + \\alpha\\bar\\sigma^2$, deci $\\bar\\sigma^2 = \\omega/(1 - \\alpha)$')]),
    (T('\\textbf{Kurtosis} with Normal $z_t$ (if $3\\alpha^2 < 1$): $K = \\dfrac{E[\\varepsilon_t^4]}{(E[\\varepsilon_t^2])^2} = 3\\,\\dfrac{1 - \\alpha^2}{1 - 3\\alpha^2} > 3$',
       '\\textbf{Coeficientul de boltire} cu $z_t$ Normale (dacă $3\\alpha^2 < 1$): $K = \\dfrac{E[\\varepsilon_t^4]}{(E[\\varepsilon_t^2])^2} = 3\\,\\dfrac{1 - \\alpha^2}{1 - 3\\alpha^2} > 3$'),
     [T('Normal innovations, but heavy-tailed returns \\refFHH', 'inovații Normale, dar randamente cu cozi groase \\refFHH'),
      T('if $\\alpha \\ge 1/\\sqrt{3} = @{ex.a1.lim}$, the fourth moment is infinite', 'dacă $\\alpha \\ge 1/\\sqrt{3} = @{ex.a1.lim}$, momentul de ordinul patru este infinit')]),
    (T('\\textbf{Worked example}: $\\omega = 0.5$, $\\alpha = 0.5$', '\\textbf{Exemplu rezolvat}: $\\omega = 0{,}5$, $\\alpha = 0{,}5$'),
     [T('$\\bar\\sigma^2 = 0.5/0.5 = @{ex.a1.uv}$; $K = 3 \\times 0.75/0.25 = @{ex.a1.k}$; after a shock $\\varepsilon_{t-1} = 2$: $\\sigma_t^2 = 0.5 + 0.5 \\times 4 = @{ex.a1.s2}$, $\\sigma_t = @{ex.a1.s}$',
        '$\\bar\\sigma^2 = 0{,}5/0{,}5 = @{ex.a1.uv}$; $K = 3 \\times 0{,}75/0{,}25 = @{ex.a1.k}$; după un șoc $\\varepsilon_{t-1} = 2$: $\\sigma_t^2 = 0{,}5 + 0{,}5 \\times 4 = @{ex.a1.s2}$, $\\sigma_t = @{ex.a1.s}$')])))

chart(T('Simulated Paths: i.i.d., ARCH(1) and GARCH(1,1)', 'Traiectorii simulate: i.i.d., ARCH(1) și GARCH(1,1)'), 'sfm_ch9_simulated', 'SFM_ch9_simulation_likelihood', [
    T('Three series of 1000 values with unconditional variance 1; ARCH(1) with $\\omega = \\alpha = 0.5$, GARCH(1,1) with $\\omega = 0.02$, $\\alpha = 0.10$, $\\beta = 0.88$ (next section)',
      'Trei serii de 1000 de valori cu varianța necondiționată 1; ARCH(1) cu $\\omega = \\alpha = 0{,}5$, GARCH(1,1) cu $\\omega = 0{,}02$, $\\alpha = 0{,}10$, $\\beta = 0{,}88$ (secțiunea următoare)'),
    T('Sample kurtosis: i.i.d.\\ $@{sim.iid.k}$, ARCH $@{sim.arch.k}$, GARCH $@{sim.garch.k}$; theory: 3, $@{ex.a1.k}$, $@{sim.garch.kth}$; the fourth moment converges slowly',
      'Coeficientul de boltire de selecție: i.i.d.\\ $@{sim.iid.k}$, ARCH $@{sim.arch.k}$, GARCH $@{sim.garch.k}$; teoretic: 3, $@{ex.a1.k}$, $@{sim.garch.kth}$; momentul de ordinul patru converge lent'),
    T('ARCH(1) gives isolated spikes; GARCH(1,1) gives long calm and long agitated periods, as in real returns',
      'ARCH(1) produce vîrfuri izolate; GARCH(1,1) produce perioade lungi liniștite și perioade lungi agitate, ca în randamentele reale')],
    h='0.48\\textheight')

D.frame(T('ARCH($q$) and Its Limits', 'ARCH($q$) și limitele lui'), items(
    (T('\\textbf{ARCH($q$)}: $\\sigma_t^2 = \\omega + \\alpha_1\\varepsilon_{t-1}^2 + \\dots + \\alpha_q\\varepsilon_{t-q}^2$, $\\alpha_i \\ge 0$; stationary if $\\sum_i \\alpha_i < 1$',
       '\\textbf{ARCH($q$)}: $\\sigma_t^2 = \\omega + \\alpha_1\\varepsilon_{t-1}^2 + \\dots + \\alpha_q\\varepsilon_{t-q}^2$, $\\alpha_i \\ge 0$; staționar dacă $\\sum_i \\alpha_i < 1$'),
     [T('$q$: the number of lags; $\\alpha_i$: the reaction to the shock of $i$ days ago', '$q$: numărul de laguri; $\\alpha_i$: reacția la șocul de acum $i$ zile'),
      T('the effect of a shock lasts exactly $q$ days; real clustering lasts for months', 'efectul unui șoc durează exact $q$ zile; clustering-ul real durează luni')]),
    T('Testing for ARCH effects: the ARCH-LM test of Engle (regression of $\\hat\\varepsilon_t^2$ on its lags), Chapter 8', 'Testarea efectelor ARCH: testul ARCH-LM al lui Engle (regresia lui $\\hat\\varepsilon_t^2$ pe propriile laguri), Capitolul 8')) + table(
    'lrrrrr', T('S\\&P 500, Normal $z_t$ & parameters & log-likelihood & AIC & BIC & $\\sum\\alpha_i$ (+$\\beta$)',
                'S\\&P 500, $z_t$ Normale & parametri & log-verosimilitate & AIC & BIC & $\\sum\\alpha_i$ (+$\\beta$)'),
    [f'ARCH(1) & @{{aq.a1.k}} & $@{{aq.a1.ll}}$ & @{{aq.a1.aic}} & @{{aq.a1.bic}} & $@{{aq.a1.pers}}$',
     f'ARCH(5) & @{{aq.a5.k}} & $@{{aq.a5.ll}}$ & @{{aq.a5.aic}} & @{{aq.a5.bic}} & $@{{aq.a5.pers}}$',
     f'ARCH(10) & @{{aq.a10.k}} & $@{{aq.a10.ll}}$ & @{{aq.a10.aic}} & @{{aq.a10.bic}} & $@{{aq.a10.pers}}$',
     f'GARCH(1,1) & @{{aq.g11.k}} & $@{{aq.g11.ll}}$ & \\textbf{{@{{aq.g11.aic}}}} & \\textbf{{@{{aq.g11.bic}}}} & $@{{aq.g11.pers}}$'],
    size='footnotesize') + items(
    T('Interpretation: ARCH needs ten lags and still loses to GARCH(1,1), which has four parameters (AIC and BIC: smaller is better, Section 10)',
      'Interpretare: ARCH are nevoie de zece laguri și rămîne totuși inferior modelului GARCH(1,1), care are patru parametri (AIC și BIC: valoarea mai mică este mai bună, secțiunea 10)')) + ql('SFM_ch9_garch_estimation'), 'footnotesize')

D.recap(('The ARCH Model', 'modelul ARCH'), [
    T('$\\sigma_t^2 = \\omega + \\sum_i\\alpha_i\\varepsilon_{t-i}^2$: an AR model for the squared shocks', '$\\sigma_t^2 = \\omega + \\sum_i\\alpha_i\\varepsilon_{t-i}^2$: un model AR pentru pătratele șocurilor'),
    T('Unconditional variance $\\omega/(1 - \\sum\\alpha_i)$; kurtosis above 3 even with Normal innovations', 'Varianța necondiționată $\\omega/(1 - \\sum\\alpha_i)$; boltire peste 3 chiar cu inovații Normale'),
    T('Real clustering needs many lags: the motivation for GARCH', 'Clustering-ul real cere multe laguri: motivația pentru GARCH')])

# =============================================================================
# 4. GARCH(1,1)
# =============================================================================
D.section('The GARCH(1,1) Model', 'Modelul GARCH(1,1)')

D.frame(T('1986: Tim Bollerslev and GARCH', '1986: Tim Bollerslev și modelul GARCH'), cols(items(
    (T('\\refBoll, a doctoral student of Engle at the University of California, San Diego: add yesterday\'s variance to the equation',
       '\\refBoll, doctorand al lui Engle la University of California, San Diego: adaugă în ecuație varianța de ieri'),
     [T('\\textbf{GARCH} (generalised ARCH): a few parameters replace a long ARCH($q$)', '\\textbf{GARCH} (generalised ARCH, ARCH generalizat): cîțiva parametri înlocuiesc un ARCH($q$) lung')]),
    T('Student-t innovations for returns: \\refBollT', 'Inovații Student-t pentru randamente: \\refBollT'),
    T('GARCH(1,1) is still the benchmark: \\refHL compared 330 models and found it hard to beat on exchange rates',
      'GARCH(1,1) rămîne modelul de referință: \\refHL au comparat 330 de modele și l-au găsit greu de întrecut la cursurile de schimb')),
    ph('ucsd', T('Geisel Library, University of California, San Diego', 'Biblioteca Geisel, University of California, San Diego'), h='0.36\\textheight'), wl='0.56', wr='0.40'))

D.frame(T('GARCH(1,1): Definition', 'GARCH(1,1): definiție'), items(
    (T('$r_t = \\mu + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$',
       '$r_t = \\mu + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$'),
     [T('$\\omega > 0$, $\\alpha \\ge 0$, $\\beta \\ge 0$: the variance stays positive', '$\\omega > 0$, $\\alpha \\ge 0$, $\\beta \\ge 0$: varianța rămîne pozitivă')]),
    (T('Reading the parameters', 'Interpretarea parametrilor'),
     [T('$\\alpha$: the \\textbf{reaction} to yesterday\'s news (size of the jump after a shock)', '$\\alpha$: \\textbf{reacția} la știrile de ieri (mărimea saltului după un șoc)'),
      T('$\\beta$: the \\textbf{memory} of the variance (how much of yesterday\'s level is kept)', '$\\beta$: \\textbf{memoria} varianței (cît din nivelul de ieri se păstrează)'),
      T('$\\alpha + \\beta$: the \\textbf{persistence}; $\\omega$ sets the long-run level', '$\\alpha + \\beta$: \\textbf{persistența}; $\\omega$ fixează nivelul pe termen lung')]),
    (T('By recursion, an ARCH($\\infty$) with geometric weights', 'Prin recurență, un ARCH($\\infty$) cu ponderi geometrice'),
     [T('$\\sigma_t^2 = \\dfrac{\\omega}{1 - \\beta} + \\alpha\\sum_{j=0}^{\\infty}\\beta^j\\,\\varepsilon_{t-1-j}^2$: all past shocks matter, with decaying weights',
        '$\\sigma_t^2 = \\dfrac{\\omega}{1 - \\beta} + \\alpha\\sum_{j=0}^{\\infty}\\beta^j\\,\\varepsilon_{t-1-j}^2$: toate șocurile trecute contează, cu ponderi descrescătoare'),
      T('the shock of $j + 1$ days ago gets the weight $\\alpha\\beta^j$', 'șocul de acum $j + 1$ zile primește ponderea $\\alpha\\beta^j$')])))

D.frame(T('Stationarity, Long-Run Variance and Kurtosis', 'Staționaritate, varianța pe termen lung și boltire'), items(
    (T('As for ARCH: $\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\varepsilon_{t-1}^2 + v_t - \\beta v_{t-1}$, an ARMA(1,1) for the squared shocks',
       'Ca la ARCH: $\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\varepsilon_{t-1}^2 + v_t - \\beta v_{t-1}$, un ARMA(1,1) pentru pătratele șocurilor'),
     [T('$v_t = \\varepsilon_t^2 - \\sigma_t^2$: the surprise in the squared shock, as for ARCH(1)', '$v_t = \\varepsilon_t^2 - \\sigma_t^2$: surpriza din pătratul șocului, ca la ARCH(1)'),
      T('ARMA: autoregressive moving average; the ACF of $\\varepsilon_t^2$ decays like $(\\alpha + \\beta)^k$ at lag $k$', 'ARMA: model autoregresiv cu medie mobilă; ACF a lui $\\varepsilon_t^2$ scade ca $(\\alpha + \\beta)^k$ la lagul $k$')]),
    (T('\\textbf{Covariance stationary} if $\\alpha + \\beta < 1$; then the \\textbf{unconditional (long-run) variance} is', '\\textbf{Staționar în covarianță} dacă $\\alpha + \\beta < 1$; atunci \\textbf{varianța necondiționată (pe termen lung)} este'),
     [T('$\\bar\\sigma^2 = E[\\sigma_t^2] = \\dfrac{\\omega}{1 - \\alpha - \\beta}$, \\quad annualised volatility $\\sqrt{252\\,\\bar\\sigma^2}$ for 252 trading days',
        '$\\bar\\sigma^2 = E[\\sigma_t^2] = \\dfrac{\\omega}{1 - \\alpha - \\beta}$, \\quad volatilitatea anualizată $\\sqrt{252\\,\\bar\\sigma^2}$ pentru 252 de zile de tranzacționare')]),
    (T('Kurtosis with Normal $z_t$, if $(\\alpha + \\beta)^2 + 2\\alpha^2 < 1$: $K = 3\\,\\dfrac{1 - (\\alpha + \\beta)^2}{1 - (\\alpha + \\beta)^2 - 2\\alpha^2} > 3$',
       'Coeficientul de boltire cu $z_t$ Normale, dacă $(\\alpha + \\beta)^2 + 2\\alpha^2 < 1$: $K = 3\\,\\dfrac{1 - (\\alpha + \\beta)^2}{1 - (\\alpha + \\beta)^2 - 2\\alpha^2} > 3$'),
     [T('$\\alpha = 0.10$, $\\beta = 0.88$: $(\\alpha + \\beta)^2 + 2\\alpha^2 = @{ex.g.cond}$, $K = @{ex.g.k}$', '$\\alpha = 0{,}10$, $\\beta = 0{,}88$: $(\\alpha + \\beta)^2 + 2\\alpha^2 = @{ex.g.cond}$, $K = @{ex.g.k}$')])))

D.frame(T('Persistence and Half-Life', 'Persistența și timpul de înjumătățire'), items(
    (T('Forecast of the variance $h$ days ahead (derived in Section 11): $E[\\sigma_{t+h}^2 \\mid \\mathcal{F}_t] - \\bar\\sigma^2 = (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
       'Prognoza varianței peste $h$ zile (derivată în secțiunea 11): $E[\\sigma_{t+h}^2 \\mid \\mathcal{F}_t] - \\bar\\sigma^2 = (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
     [T('$E[\\cdot \\mid \\mathcal{F}_t]$: the forecast made at the end of day $t$; $\\bar\\sigma^2$: the long-run variance', '$E[\\cdot \\mid \\mathcal{F}_t]$: prognoza făcută la sfîrșitul zilei $t$; $\\bar\\sigma^2$: varianța pe termen lung'),
      T('a deviation from the long-run variance shrinks by the factor $\\alpha + \\beta$ each day', 'o abatere de la varianța pe termen lung se micșorează cu factorul $\\alpha + \\beta$ în fiecare zi')]),
    (T('\\textbf{Half-life}: the number of days after which half of a variance shock is gone', '\\textbf{Timpul de înjumătățire}: numărul de zile după care a dispărut jumătate dintr-un șoc de varianță'),
     [T('$(\\alpha + \\beta)^{h} = 1/2 \\;\\Rightarrow\\; h_{1/2} = \\dfrac{\\ln 0.5}{\\ln(\\alpha + \\beta)}$', '$(\\alpha + \\beta)^{h} = 1/2 \\;\\Rightarrow\\; h_{1/2} = \\dfrac{\\ln 0{,}5}{\\ln(\\alpha + \\beta)}$'),
      T('$\\alpha + \\beta = 0.90$: @{hl.90} days; $0.98$: @{hl.98} days; $0.99$: @{hl.99} days: small changes near 1 matter a lot',
        '$\\alpha + \\beta = 0{,}90$: @{hl.90} zile; $0{,}98$: @{hl.98} zile; $0{,}99$: @{hl.99} zile: schimbările mici în apropiere de 1 contează mult')]),
    T('Daily equity returns: $\\alpha$ around 0.05--0.15, $\\beta$ around 0.80--0.93, $\\alpha + \\beta$ close to 1 (Section 7)',
      'Randamentele zilnice ale acțiunilor: $\\alpha$ în jur de 0,05--0,15, $\\beta$ în jur de 0,80--0,93, $\\alpha + \\beta$ apropiat de 1 (secțiunea 7)')))

D.frame(T('Worked Example: One GARCH(1,1) Step', 'Exemplu rezolvat: un pas GARCH(1,1)'), items(
    T('Daily returns in \\%: $\\omega = 0.02$, $\\alpha = 0.10$, $\\beta = 0.88$, $\\mu = 0$', 'Randamente zilnice în \\%: $\\omega = 0{,}02$, $\\alpha = 0{,}10$, $\\beta = 0{,}88$, $\\mu = 0$'),
    (T('Long-run quantities', 'Mărimile pe termen lung'),
     [T('persistence $\\alpha + \\beta = @{ex.g.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{ex.g.pers}) = @{ex.g.uv}$; annualised volatility $\\sqrt{252 \\times @{ex.g.uv}} = @{ex.g.vol}\\%$',
        'persistența $\\alpha + \\beta = @{ex.g.pers}$; $\\bar\\sigma^2 = 0{,}02/(1 - @{ex.g.pers}) = @{ex.g.uv}$; volatilitatea anualizată $\\sqrt{252 \\times @{ex.g.uv}} = @{ex.g.vol}\\%$'),
      T('half-life $\\ln 0.5/\\ln @{ex.g.pers} = @{ex.g.hl}$ days', 'timpul de înjumătățire $\\ln 0{,}5/\\ln @{ex.g.pers} = @{ex.g.hl}$ zile')]),
    (T('Today: $\\sigma_t^2 = 1.2$ and the return is $r_t = -3\\%$', 'Azi: $\\sigma_t^2 = 1{,}2$, iar randamentul este $r_t = -3\\%$'),
     [T('tomorrow: $\\sigma_{t+1}^2 = 0.02 + 0.10 \\times 9 + 0.88 \\times 1.2 = @{ex.g.s2}$, $\\sigma_{t+1} = @{ex.g.s}\\%$', 'mîine: $\\sigma_{t+1}^2 = 0{,}02 + 0{,}10 \\times 9 + 0{,}88 \\times 1{,}2 = @{ex.g.s2}$, $\\sigma_{t+1} = @{ex.g.s}\\%$'),
      T('the shock raised the variance from 1.2 to almost 2; it then decays towards @{ex.g.uv} with the factor @{ex.g.pers} per day',
        'șocul a crescut varianța de la 1,2 la aproape 2; apoi ea scade spre @{ex.g.uv} cu factorul @{ex.g.pers} pe zi')])))

D.frame(T('IGARCH and the EWMA Link', 'IGARCH și legătura cu EWMA'), items(
    (T('\\textbf{IGARCH} (integrated GARCH) \\refEB: $\\alpha + \\beta = 1$', '\\textbf{IGARCH} (integrated GARCH, GARCH integrat) \\refEB: $\\alpha + \\beta = 1$'),
     [T('shocks to the variance never die out: no half-life, no finite unconditional variance', 'șocurile varianței nu se sting niciodată: nu există timp de înjumătățire și nici varianță necondiționată finită'),
      T('the forecast of the variance is the same for every horizon', 'prognoza varianței este aceeași pentru orice orizont')]),
    (T('\\textbf{EWMA} (exponentially weighted moving average, Chapter 8): $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$',
       '\\textbf{EWMA} (exponentially weighted moving average, Capitolul 8): $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$'),
     [T('this is IGARCH with $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$; RiskMetrics uses $\\lambda = 0.94$ for daily data',
        'este un IGARCH cu $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$; RiskMetrics folosește $\\lambda = 0{,}94$ pentru date zilnice'),
      T('EWMA: nothing to estimate, but no mean reversion; GARCH: three parameters, and volatility returns to $\\bar\\sigma$', 'EWMA: nimic de estimat, dar fără revenire la medie; GARCH: trei parametri, iar volatilitatea revine la $\\bar\\sigma$')]),
    (T('\\textbf{Question for the room}: an analyst forecasts the volatility one year ahead with EWMA right after a crash. What goes wrong?',
       '\\textbf{Întrebare pentru sală}: un analist prognozează volatilitatea peste un an cu EWMA imediat după un crah. Unde greșește?'),
     [T('\\textbf{Answer}: EWMA keeps today\'s crisis level for the whole year; GARCH lets it decay towards the long-run level',
        '\\textbf{Răspuns}: EWMA păstrează nivelul de criză de azi pentru tot anul; GARCH îl lasă să scadă spre nivelul pe termen lung')])))

D.recap(('The GARCH(1,1) Model', 'modelul GARCH(1,1)'), [
    T('$\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$: reaction $\\alpha$, memory $\\beta$', '$\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$: reacția $\\alpha$, memoria $\\beta$'),
    T('$\\alpha + \\beta < 1$: long-run variance $\\omega/(1 - \\alpha - \\beta)$, half-life $\\ln 0.5/\\ln(\\alpha + \\beta)$', '$\\alpha + \\beta < 1$: varianța pe termen lung $\\omega/(1 - \\alpha - \\beta)$, timpul de înjumătățire $\\ln 0{,}5/\\ln(\\alpha + \\beta)$'),
    T('$\\alpha + \\beta = 1$: IGARCH; EWMA is an IGARCH without $\\omega$', '$\\alpha + \\beta = 1$: IGARCH; EWMA este un IGARCH fără $\\omega$')])

# =============================================================================
# 5. ESTIMARE
# =============================================================================
D.section('Estimation by Maximum Likelihood', 'Estimarea prin verosimilitate maximă')

D.frame(T('The Likelihood of a GARCH Model', 'Verosimilitatea unui model GARCH'), items(
    (T('The returns are dependent, so we factor the joint density into conditional densities', 'Randamentele sînt dependente, deci descompunem densitatea comună în densități condiționate'),
     [T('$f$: a probability density; $T$: the number of returns; $\\theta$: the vector of parameters; $\\ell$: the log-likelihood', '$f$: o densitate de probabilitate; $T$: numărul de randamente; $\\theta$: vectorul parametrilor; $\\ell$: log-verosimilitatea'),
      T('$f(r_1, \\dots, r_T) = f(r_1)\\prod_{t=2}^{T} f(r_t \\mid \\mathcal{F}_{t-1})$, and $r_t \\mid \\mathcal{F}_{t-1} \\sim N(\\mu, \\sigma_t^2)$ for Normal innovations',
        '$f(r_1, \\dots, r_T) = f(r_1)\\prod_{t=2}^{T} f(r_t \\mid \\mathcal{F}_{t-1})$, iar $r_t \\mid \\mathcal{F}_{t-1} \\sim N(\\mu, \\sigma_t^2)$ pentru inovații Normale')]),
    (T('\\textbf{Conditional log-likelihood} with $\\theta = (\\mu, \\omega, \\alpha, \\beta)$:', '\\textbf{Log-verosimilitatea condiționată}, cu $\\theta = (\\mu, \\omega, \\alpha, \\beta)$:'),
     [T('$\\ell(\\theta) = -\\dfrac12\\sum_{t=2}^{T}\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2(\\theta) + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2(\\theta)}\\Big]$',
        '$\\ell(\\theta) = -\\dfrac12\\sum_{t=2}^{T}\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2(\\theta) + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2(\\theta)}\\Big]$'),
      T('in words: each day adds a penalty for a large $\\sigma_t^2$ and a penalty for a squared error that is large relative to $\\sigma_t^2$', 'în cuvinte: fiecare zi adaugă o penalizare pentru un $\\sigma_t^2$ mare și una pentru o eroare la pătrat mare în raport cu $\\sigma_t^2$'),
      T('$\\sigma_t^2(\\theta)$ is computed recursively from a starting value $\\sigma_1^2$ (for example the sample variance)', '$\\sigma_t^2(\\theta)$ se calculează recursiv, pornind de la o valoare inițială $\\sigma_1^2$ (de exemplu, varianța de selecție)')]),
    (T('\\textbf{MLE} (maximum likelihood estimation): $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$, under $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$',
       '\\textbf{MLE} (maximum likelihood estimation, estimarea prin verosimilitate maximă): $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$, cu $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$'),
     [T('no closed form: a numerical optimiser searches for the maximum \\refFHH', 'nu există o formulă explicită: un algoritm numeric de optimizare caută maximul \\refFHH')])))

chart(T('The Likelihood of ARCH(1)', 'Verosimilitatea modelului ARCH(1)'), 'sfm_ch9_lik_arch1', 'SFM_ch9_simulation_likelihood', [
    T('Simulated ARCH(1) with $\\omega = \\alpha = 0.5$; $\\ell(\\alpha)$ with $\\omega = 1 - \\alpha$, as in SFElikarch1; each curve minus its maximum',
      'ARCH(1) simulat cu $\\omega = \\alpha = 0{,}5$; $\\ell(\\alpha)$ cu $\\omega = 1 - \\alpha$, ca în SFElikarch1; fiecare curbă minus maximul ei'),
    T('$n = 100$: $\\hat\\alpha = @{la.100}$ (standard error $@{la.100.se}$); $n = 1000$: $\\hat\\alpha = @{la.1000}$ ($@{la.1000.se}$)',
      '$n = 100$: $\\hat\\alpha = @{la.100}$ (eroarea standard $@{la.100.se}$); $n = 1000$: $\\hat\\alpha = @{la.1000}$ ($@{la.1000.se}$)'),
    T('Interpretation: more data make the curve sharper; its curvature at the maximum gives the standard error', 'Interpretare: mai multe date fac curba mai ascuțită; curbura ei în punctul de maxim dă eroarea standard')],
    h='0.50\\textheight')

chart(T('The Likelihood of GARCH(1,1)', 'Verosimilitatea modelului GARCH(1,1)'), 'sfm_ch9_lik_garch', 'SFM_ch9_simulation_likelihood', [
    T('Simulated GARCH(1,1), $\\omega = 0.1$, $\\alpha = 0.1$, $\\beta = 0.8$; $\\omega$ set so that the unconditional variance equals the sample variance, as in SFElikgarch; contours: log-likelihood minus its maximum',
      'GARCH(1,1) simulat, $\\omega = 0{,}1$, $\\alpha = 0{,}1$, $\\beta = 0{,}8$; $\\omega$ fixat astfel încît varianța necondiționată să fie egală cu varianța de selecție, ca în SFElikgarch; contururi: log-verosimilitatea minus maximul ei'),
    T('$n = 500$: maximum at $(@{lg.500.a}, @{lg.500.b})$, far from the truth; $n = 2000$: $(@{lg.2000.a}, @{lg.2000.b})$',
      '$n = 500$: maximul în $(@{lg.500.a}; @{lg.500.b})$, departe de valorile adevărate; $n = 2000$: $(@{lg.2000.a}; @{lg.2000.b})$'),
    T('Interpretation: a long, flat ridge along which $\\alpha$ and $\\beta$ trade off: GARCH needs long samples', 'Interpretare: o creastă lungă și plată, de-a lungul căreia $\\alpha$ și $\\beta$ se compensează: GARCH are nevoie de eșantioane lungi')],
    h='0.48\\textheight')


def er(k, lab):
    return (f'{lab} & $@{{step.{k}}}$ ($@{{step.{k}.se}}$) & $@{{arch.{k}}}$ & $@{{arch.{k}.sec}}$ & $@{{arch.{k}.ser}}$')


D.frame(T('Estimation Step by Step and with a Package', 'Estimarea pas cu pas și cu un pachet'), table(
    'lrrrr', T('S\\&P 500, GARCH(1,1)-N & step by step (SE) & \\texttt{arch} & classic SE & robust SE',
               'S\\&P 500, GARCH(1,1)-N & pas cu pas (SE) & \\texttt{arch} & SE clasică & SE robustă'),
    [er('mu', '$\\mu$'), er('om', '$\\omega$'), er('a', '$\\alpha$'), er('b', '$\\beta$')], size='footnotesize') + items(
    (T('Step by step: write $-\\ell(\\theta)$ in Python and minimise it with \\texttt{scipy.optimize.minimize} under $\\alpha + \\beta < 1$',
       'Pas cu pas: scriem $-\\ell(\\theta)$ în Python și o minimizăm cu \\texttt{scipy.optimize.minimize}, cu restricția $\\alpha + \\beta < 1$'),
     [T('the two routes differ only in the start-up of $\\sigma_t^2$ and in the optimiser; log-likelihood $@{step.ll}$ against $@{en.ll}$',
        'cele două variante diferă doar prin valoarea inițială a lui $\\sigma_t^2$ și prin algoritmul de optimizare; log-verosimilitatea $@{step.ll}$ față de $@{en.ll}$')]),
    T('SE (standard error), classic: from the inverse Hessian (the curvature of $\\ell$); robust: valid when $z_t$ is not Normal \\refBW',
      'SE (standard error, eroarea standard) clasică: din inversa matricei hessiene (curbura lui $\\ell$); robustă: validă și cînd $z_t$ nu este Normal \\refBW'),
    T('Interpretation: the robust SE are about @{se.ratio} times larger (@{se.ratio.om} for $\\omega$): with heavy-tailed $z_t$ the classic SE overstate the precision',
      'Interpretare: SE robuste sînt de circa @{se.ratio} ori mai mari (@{se.ratio.om} pentru $\\omega$): cînd $z_t$ are cozi groase, SE clasice supraestimează precizia')) + ql('SFM_ch9_garch_estimation'), 'footnotesize')

D.frame(T('Quasi-Maximum Likelihood', 'Cvasi-verosimilitatea maximă'), items(
    (T('\\textbf{QMLE} (quasi-maximum likelihood estimation): maximise the Normal likelihood even if $z_t$ is not Normal',
       '\\textbf{QMLE} (quasi-maximum likelihood estimation, estimarea prin cvasi-verosimilitate maximă): maximizăm verosimilitatea Normală chiar dacă $z_t$ nu este Normal'),
     [T('the estimates are still consistent if $\\mu$ and $\\sigma_t^2$ are correctly specified \\refBW', 'estimările rămîn consistente dacă $\\mu$ și $\\sigma_t^2$ sînt corect specificate \\refBW'),
      T('but the classic SE are wrong: use the robust (``sandwich\'\') SE', 'dar SE clasice sînt greșite: folosim SE robuste (de tip „sandwich”)')]),
    (T('Practical rules', 'Reguli practice'),
     [T('report robust SE (the default of \\texttt{arch})', 'raportăm SE robuste (opțiunea implicită în \\texttt{arch})'),
      T('use at least 1000--2000 daily observations; returns in \\% avoid numerical problems with tiny $\\omega$', 'folosim cel puțin 1000--2000 de observații zilnice; randamentele în \\% evită problemele numerice cauzate de un $\\omega$ foarte mic'),
      T('an estimate on the boundary ($\\alpha + \\beta = 1$) has no usual standard error: report it as IGARCH', 'o estimare pe frontieră ($\\alpha + \\beta = 1$) nu are o eroare standard obișnuită: o raportăm ca IGARCH')]),
    T('A better innovation distribution gives more efficient estimates: next section', 'O distribuție mai potrivită a inovațiilor dă estimări mai eficiente: secțiunea următoare')))

D.recap(('Estimation by Maximum Likelihood', 'estimarea prin verosimilitate maximă'), [
    T('$\\ell(\\theta)$ = sum of conditional log-densities; $\\sigma_t^2(\\theta)$ computed recursively', '$\\ell(\\theta)$ = suma log-densităților condiționate; $\\sigma_t^2(\\theta)$ calculat recursiv'),
    T('The likelihood surface has a ridge in $(\\alpha, \\beta)$: long samples needed', 'Suprafața verosimilității are o creastă în $(\\alpha, \\beta)$: sînt necesare eșantioane lungi'),
    T('Report robust (Bollerslev--Wooldridge) standard errors', 'Raportăm erori standard robuste (Bollerslev--Wooldridge)')])

# =============================================================================
# 6. INOVAȚII
# =============================================================================
D.section('Heavy-Tailed Innovations', 'Inovații cu cozi groase')

D.frame(T('Student-t and Skewed-t Innovations', 'Inovații Student-t și t asimetrice'), items(
    (T('GARCH with Normal $z_t$ explains only part of the kurtosis: S\\&P 500 returns $@{kurt.r}$, standardised residuals $\\hat z_t = \\hat\\varepsilon_t/\\hat\\sigma_t$: $@{kurt.z}$',
       'GARCH cu $z_t$ Normale explică doar o parte din boltire: randamentele S\\&P 500 au $@{kurt.r}$, reziduurile standardizate $\\hat z_t = \\hat\\varepsilon_t/\\hat\\sigma_t$ au $@{kurt.z}$'),
     [T('the remaining heavy tails must come from the distribution of $z_t$', 'cozile groase rămase trebuie să vină din distribuția lui $z_t$')]),
    (T('\\textbf{Standardised Student-t} with $\\nu > 2$ degrees of freedom (Chapter 2) \\refBollT', '\\textbf{Student-t standardizată} cu $\\nu > 2$ grade de libertate (Capitolul 2) \\refBollT'),
     [T('$t_\\nu$: a Student-t variable with $\\nu$ degrees of freedom; $z_t = t_\\nu\\sqrt{(\\nu - 2)/\\nu}$ has variance 1', '$t_\\nu$: o variabilă Student-t cu $\\nu$ grade de libertate; $z_t = t_\\nu\\sqrt{(\\nu - 2)/\\nu}$ are varianța 1'),
      T('a small $\\nu$ means heavy tails: kurtosis $3 + 6/(\\nu - 4)$ for $\\nu > 4$; $\\nu \\to \\infty$: the Normal distribution', 'un $\\nu$ mic înseamnă cozi groase: coeficientul de boltire $3 + 6/(\\nu - 4)$ pentru $\\nu > 4$; $\\nu \\to \\infty$: distribuția Normală'),
      T('$\\nu$ is estimated together with $(\\mu, \\omega, \\alpha, \\beta)$', '$\\nu$ se estimează împreună cu $(\\mu, \\omega, \\alpha, \\beta)$')]),
    (T('\\textbf{Skewed t} \\refHansen: one more parameter $\\lambda \\in (-1, 1)$; $\\lambda < 0$: a longer left tail', '\\textbf{t asimetrică} \\refHansen: încă un parametru $\\lambda \\in (-1, 1)$; $\\lambda < 0$: o coadă stîngă mai lungă'),
     [T('nested models: Normal $\\subset$ t $\\subset$ skewed t; compare them with likelihood-ratio tests or AIC/BIC (Chapter 6)',
        'modele incluse unul în altul: Normal $\\subset$ t $\\subset$ t asimetrică; le comparăm prin teste ale raportului de verosimilitate sau prin AIC/BIC (Capitolul 6)')])))

D.frame(T('S\\&P 500: Three Innovation Distributions', 'S\\&P 500: trei distribuții ale inovațiilor'), table(
    'lrrrrrrr', T('GARCH(1,1) & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\lambda$ & log-lik. & $\\alpha + \\beta$',
                  'GARCH(1,1) & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\lambda$ & log-verosim. & $\\alpha + \\beta$'),
    [T('Normal', 'Normală') + ' & $@{en.om}$ & $@{en.a}$ & $@{en.b}$ & -- & -- & $@{en.ll}$ & $@{en.pers}$',
     'Student-t & $@{et.om}$ & $@{et.a}$ & $@{et.b}$ & $@{et.nu}$ & -- & $@{et.ll}$ & $@{et.pers}$',
     T('skewed t', 't asimetrică') + ' & $@{es.om}$ & $@{es.a}$ & $@{es.b}$ & $@{es.eta}$ & $@{es.lam}$ & $@{es.ll}$ & $@{es.pers}$'],
    size='footnotesize') + items(
    T('Likelihood-ratio statistic $2(\\ell_1 - \\ell_0)$ ($\\ell_1$, $\\ell_0$: maximised log-likelihoods of the larger and the smaller model): t against Normal $@{lr.tn}$ (1 restriction, $\\chi^2_{0.95}(1) = 3.84$); skewed t against t $@{lr.st}$: both reject the smaller model',
      'Statistica raportului de verosimilitate $2(\\ell_1 - \\ell_0)$ ($\\ell_1$, $\\ell_0$: log-verosimilitățile maxime ale modelului mai mare și ale celui mai mic): t față de Normal $@{lr.tn}$ (o restricție, $\\chi^2_{0.95}(1) = 3{,}84$); t asimetrică față de t $@{lr.st}$: ambele resping modelul mai mic'),
    T('$\\hat\\nu = @{et.nu}$: tails far heavier than Normal; $\\hat\\lambda = @{es.lam}$: large falls more frequent than large rises',
      '$\\hat\\nu = @{et.nu}$: cozi mult mai groase decît la distribuția Normală; $\\hat\\lambda = @{es.lam}$: scăderile mari sînt mai frecvente decît creșterile mari'),
    T('Interpretation: $\\alpha$ and $\\beta$ hardly change; the innovation distribution matters for tail quantiles (VaR), less for $\\sigma_t$',
      'Interpretare: $\\alpha$ și $\\beta$ aproape nu se schimbă; distribuția inovațiilor contează pentru cuantilele din coadă (VaR), mai puțin pentru $\\sigma_t$')) + ql('SFM_ch9_garch_estimation'), 'footnotesize')

chart(T('QQ Plots of the Standardised Residuals', 'Graficele QQ ale reziduurilor standardizate'), 'sfm_ch9_qq', 'SFM_ch9_garch_estimation', [
    T('Left: residuals of the Normal GARCH against the Normal distribution: the 0.1\\% quantile is $@{qq.n001}$, against $@{qq.n001th}$ in theory',
      'Stînga: reziduurile modelului GARCH cu inovații Normale față de distribuția Normală: cuantila de 0,1\\% este $@{qq.n001}$, față de $@{qq.n001th}$ teoretic'),
    T('Right: residuals of GARCH-t against the standardised t with $\\hat\\nu = @{qq.nu}$: $@{qq.t001}$ against $@{qq.t001th}$: much closer; the largest falls are still deeper',
      'Dreapta: reziduurile GARCH-t față de t standardizată cu $\\hat\\nu = @{qq.nu}$: $@{qq.t001}$ față de $@{qq.t001th}$, mult mai aproape; cele mai mari scăderi rămîn mai adînci'),
    T('Interpretation: the left tail is the problem: a reason for skewed innovations and for asymmetric models (Section 8)', 'Interpretare: problema este coada stîngă: un motiv pentru inovații asimetrice și pentru modele asimetrice (secțiunea 8)')],
    h='0.50\\textheight')

D.recap(('Heavy-Tailed Innovations', 'inovații cu cozi groase'), [
    T('GARCH explains part of the kurtosis; the rest comes from $z_t$', 'GARCH explică o parte din boltire; restul vine din $z_t$'),
    T('Student-t ($\\nu$) and skewed t ($\\nu$, $\\lambda$) improve the likelihood strongly', 'Student-t ($\\nu$) și t asimetrică ($\\nu$, $\\lambda$) cresc puternic verosimilitatea'),
    T('The distribution of $z_t$ decides the VaR quantile', 'Distribuția lui $z_t$ determină cuantila VaR')])

# =============================================================================
# 7. ȘASE PIEȚE
# =============================================================================
D.section('GARCH in Six Markets', 'GARCH pe șase piețe')


def mrow(k):
    return (f'{SHORT[k]} & @{{m.{k}.y0}} & $@{{m.{k}.a}}$ & $@{{m.{k}.b}}$ & $@{{m.{k}.nu}}$ & $@{{m.{k}.pers}}$ & @{{m.{k}.hl}} & '
            f'@{{m.{k}.vlr}} & @{{m.{k}.vs}}')


D.frame(T('GARCH(1,1)-t Estimates', 'Estimările GARCH(1,1)-t'), table(
    'lrrrrrrrr', T('& from & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & half-life & long-run vol. & sample vol.',
                   '& din & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & înjumătățire & vol. termen lung & vol. selecție'),
    [mrow(k) for k in ASSETS], size='scriptsize') + items(
    T('Daily log returns in \\% to @{end}; robust SE of $\\hat\\alpha$ and $\\hat\\beta$ between @{m.se.min} and @{m.se.max}; half-life in days; volatilities annualised with the actual number of observations per year, in \\%',
      'Randamente logaritmice zilnice în \\% pînă la @{end}; SE robuste pentru $\\hat\\alpha$ și $\\hat\\beta$ între @{m.se.min} și @{m.se.max}; timpul de înjumătățire în zile; volatilitățile anualizate cu numărul efectiv de observații pe an, în \\%'),
    T('Indices: $\\alpha + \\beta$ between @{m.bet.pers} and @{m.sp500.pers}, half-lives of @{m.bet.hl}--@{m.sp500.hl} trading days; BVB stocks: shorter memory (TLV @{m.tlv.hl} days, SNP @{m.snp.hl} days)',
      'Indicii: $\\alpha + \\beta$ între @{m.bet.pers} și @{m.sp500.pers}, timpi de înjumătățire de @{m.bet.hl}--@{m.sp500.hl} zile de tranzacționare; acțiunile BVB: memorie mai scurtă (TLV @{m.tlv.hl} zile, SNP @{m.snp.hl} zile)'),
    T('BET: the largest $\\alpha$ (@{m.bet.a}): a stronger reaction to news; all $\\hat\\nu$ between @{m.nu.min} and @{m.nu.max}: heavy tails everywhere',
      'BET: cel mai mare $\\alpha$ (@{m.bet.a}): o reacție mai puternică la știri; toate valorile $\\hat\\nu$ sînt între @{m.nu.min} și @{m.nu.max}: cozi groase peste tot')) + ql('SFM_ch9_volatility_markets'), 'footnotesize')

D.frame(T('Interpretation: Bitcoin and the Long-Run Volatility', 'Interpretarea rezultatelor: Bitcoin și volatilitatea pe termen lung'), items(
    (T('Bitcoin: $\\hat\\alpha + \\hat\\beta = @{m.btc.ab}$, on the boundary $\\alpha + \\beta < 1$: an IGARCH', 'Bitcoin: $\\hat\\alpha + \\hat\\beta = @{m.btc.ab}$, pe frontiera $\\alpha + \\beta < 1$: un IGARCH'),
     [T('no half-life and no long-run volatility: the model behaves like an EWMA with $\\lambda \\approx @{m.btc.b}$', 'nu există timp de înjumătățire și nici volatilitate pe termen lung: modelul se comportă ca un EWMA cu $\\lambda \\approx @{m.btc.b}$'),
      T('$\\hat\\nu = @{m.btc.nu}$: extremely heavy tails (the fourth moment of $z_t$ does not exist)', '$\\hat\\nu = @{m.btc.nu}$: cozi extrem de groase (momentul de ordinul patru al lui $z_t$ nu există)')]),
    (T('\\textbf{Question for the room}: S\\&P 500: long-run volatility @{m.sp500.vlr}\\%, sample volatility @{m.sp500.vs}\\%. Is the model wrong?',
       '\\textbf{Întrebare pentru sală}: S\\&P 500: volatilitatea pe termen lung @{m.sp500.vlr}\\%, volatilitatea de selecție @{m.sp500.vs}\\%. Este greșit modelul?'),
     [T('\\textbf{Answer}: not necessarily: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ divides by $1 - \\alpha - \\beta = @{m.sp500.1mp}$, a small and imprecise number',
        '\\textbf{Răspuns}: nu neapărat: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ se împarte la $1 - \\alpha - \\beta = @{m.sp500.1mp}$, un număr mic și imprecis'),
      T('the Normal GARCH gives @{en.vlr}\\%; the long-run level is the least reliable output of a persistent GARCH', 'GARCH cu inovații Normale dă @{en.vlr}\\%; nivelul pe termen lung este cel mai puțin sigur rezultat al unui GARCH persistent')])))

chart(T('Conditional Volatility: S\\&P 500 and DAX', 'Volatilitatea condiționată: S\\&P 500 și DAX'), 'sfm_ch9_vol_sp500_dax', 'SFM_ch9_volatility_markets', [
    T('GARCH(1,1)-t volatility $\\hat\\sigma_t$, annualised, as in SFEvolgarchest; dashed: the sample volatility',
      'Volatilitatea GARCH(1,1)-t $\\hat\\sigma_t$, anualizată, ca în SFEvolgarchest; linia întreruptă: volatilitatea de selecție'),
    T('S\\&P 500: peaks @{v.sp500.p2008}\\% (@{v.sp500.d2008}) and @{v.sp500.p2020}\\% (@{v.sp500.d2020}); median @{v.sp500.med}\\%; on @{end}: @{v.sp500.last}\\%',
      'S\\&P 500: vîrfuri de @{v.sp500.p2008}\\% (@{v.sp500.d2008}) și @{v.sp500.p2020}\\% (@{v.sp500.d2020}); mediana @{v.sp500.med}\\%; la @{end}: @{v.sp500.last}\\%')],
    h='0.56\\textheight')

chart(T('Conditional Volatility: BET and Bitcoin', 'Volatilitatea condiționată: BET și Bitcoin'), 'sfm_ch9_vol_bet_btc', 'SFM_ch9_volatility_markets', [
    T('BET: the highest peak in 2008 (@{v.bet.p2008}\\%, @{v.bet.d2008}), lower in 2020 (@{v.bet.p2020}\\%); Bitcoin: @{v.btc.p2020}\\% on @{v.btc.d2020}, median @{v.btc.med}\\%',
      'BET: cel mai înalt vîrf în 2008 (@{v.bet.p2008}\\%, @{v.bet.d2008}), mai mic în 2020 (@{v.bet.p2020}\\%); Bitcoin: @{v.btc.p2020}\\% pe @{v.btc.d2020}, mediana @{v.btc.med}\\%'),
    T('Interpretation: the 2020 shock hit all markets in the same week; the median volatility of Bitcoin is @{v.ratio} times that of the S\\&P 500 (@{v.sp500.med}\\%)',
      'Interpretare: șocul din 2020 a afectat toate piețele în aceeași săptămînă; volatilitatea mediană a Bitcoin este de @{v.ratio} ori mai mare decît cea a S\\&P 500 (@{v.sp500.med}\\%)')],
    h='0.56\\textheight')

chart(T('How Long Does a Shock Last?', 'Durata unui șoc de volatilitate'), 'sfm_ch9_persistence', 'SFM_ch9_volatility_markets', [
    T('Share of a variance shock left after $h$ days: $(\\alpha + \\beta)^h$; the half-life is where the curve crosses 1/2',
      'Ponderea unui șoc de varianță rămasă după $h$ zile: $(\\alpha + \\beta)^h$; timpul de înjumătățire este punctul în care curba trece de 1/2'),
    T('Interpretation: after a crisis, the S\\&P 500 needs about half a year to halve the excess variance; OMV Petrom about two weeks; Bitcoin never',
      'Interpretare: după o criză, S\\&P 500 are nevoie de circa jumătate de an ca să înjumătățească varianța în exces; OMV Petrom de circa două săptămîni; Bitcoin, niciodată')],
    h='0.52\\textheight')

D.recap(('GARCH in Six Markets', 'GARCH pe șase piețe'), [
    T('$\\alpha + \\beta$ close to 1 everywhere: volatility shocks last for months', '$\\alpha + \\beta$ apropiat de 1 peste tot: șocurile de volatilitate durează luni'),
    T('BVB stocks: faster decay; Bitcoin: IGARCH, extreme tails', 'Acțiunile BVB: stingere mai rapidă; Bitcoin: IGARCH, cozi extreme'),
    T('The long-run volatility is imprecise when $\\alpha + \\beta$ is near 1', 'Volatilitatea pe termen lung este imprecisă cînd $\\alpha + \\beta$ este aproape de 1')])

# =============================================================================
# 8. ASIMETRIE
# =============================================================================
D.section('Asymmetry and the Leverage Effect', 'Asimetria și efectul de levier')

D.frame(T('The Leverage Effect', 'Efectul de levier'), items(
    (T('Falls in stock prices are followed by larger increases in volatility than rises of the same size \\refChristie', 'Scăderile prețurilor acțiunilor sînt urmate de creșteri mai mari ale volatilității decît creșterile de aceeași mărime \\refChristie'),
     [T('\\textbf{leverage} explanation: a lower share price raises the debt-to-equity ratio, so the equity becomes riskier', 'explicația prin \\textbf{levier}: un preț mai mic al acțiunii crește raportul datorii/capitaluri proprii, deci acțiunea devine mai riscantă'),
      T('\\textbf{volatility feedback}: news of higher risk lowers prices at once', '\\textbf{feedback-ul volatilității}: știrile despre un risc mai mare scad imediat prețurile')]),
    (T('GARCH(1,1) cannot see it: $\\sigma_t^2$ depends on $\\varepsilon_{t-1}^2$, not on the sign of $\\varepsilon_{t-1}$', 'GARCH(1,1) nu îl poate surprinde: $\\sigma_t^2$ depinde de $\\varepsilon_{t-1}^2$, nu de semnul lui $\\varepsilon_{t-1}$'),
     [T('the QQ plot of Section 6 and the skewed t ($\\hat\\lambda < 0$) already pointed to the left tail', 'graficul QQ din secțiunea 6 și t asimetrică ($\\hat\\lambda < 0$) au arătat deja spre coada stîngă')]),
    T('\\textbf{NIC} (news impact curve) \\refEN: $\\sigma_t^2$ as a function of $\\varepsilon_{t-1}$, with $\\sigma_{t-1}^2$ fixed at the unconditional variance',
      '\\textbf{NIC} (news impact curve, curba de impact a știrilor) \\refEN: $\\sigma_t^2$ ca funcție de $\\varepsilon_{t-1}$, cu $\\sigma_{t-1}^2$ fixat la varianța necondiționată')))

D.frame(T('GJR-GARCH and EGARCH', 'GJR-GARCH și EGARCH'), items(
    (T('\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,I_{t-1})\\,\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ if $\\varepsilon_{t-1} < 0$, else 0',
       '\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,I_{t-1})\\,\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ dacă $\\varepsilon_{t-1} < 0$, altfel 0'),
     [T('$I_{t-1}$: an indicator of a negative shock yesterday; $\\gamma$: the extra reaction to bad news', '$I_{t-1}$: un indicator al unui șoc negativ ieri; $\\gamma$: reacția suplimentară la știrile proaste'),
      T('good news: slope $\\alpha$; bad news: $\\alpha + \\gamma$; leverage effect: $\\gamma > 0$', 'știri bune: panta $\\alpha$; știri proaste: $\\alpha + \\gamma$; efect de levier: $\\gamma > 0$'),
      T('persistence $\\alpha + \\beta + \\gamma/2$ for symmetric $z_t$ (half of the shocks are negative)', 'persistența $\\alpha + \\beta + \\gamma/2$ pentru $z_t$ simetric (jumătate dintre șocuri sînt negative)')]),
    (T('\\textbf{EGARCH} (exponential GARCH) \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha\\big(|z_{t-1}| - E|z_{t-1}|\\big) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$',
       '\\textbf{EGARCH} (exponential GARCH, GARCH exponențial) \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha\\big(|z_{t-1}| - E|z_{t-1}|\\big) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$'),
     [T('$|z_{t-1}| - E|z_{t-1}|$: the size of yesterday\'s standardised shock above its average ($E|z| = \\sqrt{2/\\pi}$ for Normal $z$)', '$|z_{t-1}| - E|z_{t-1}|$: mărimea șocului standardizat de ieri peste media ei ($E|z| = \\sqrt{2/\\pi}$ pentru $z$ Normal)'),
      T('the logarithm keeps $\\sigma_t^2 > 0$ without sign restrictions on the parameters', 'logaritmul păstrează $\\sigma_t^2 > 0$ fără restricții de semn asupra parametrilor'),
      T('leverage effect: $\\gamma < 0$ (a negative $z_{t-1}$ raises $\\ln\\sigma_t^2$); persistence: $\\beta$', 'efect de levier: $\\gamma < 0$ (un $z_{t-1}$ negativ crește $\\ln\\sigma_t^2$); persistența: $\\beta$')]),
    T('Textbook treatment of both, and of the threshold GARCH: \\refFHH, Section 13.2', 'Ambele modele, precum și GARCH cu prag (threshold GARCH), sînt tratate în \\refFHH, secțiunea 13.2')))

chart(T('News Impact Curves: S\\&P 500 and Bitcoin', 'Curbele de impact ale știrilor: S\\&P 500 și Bitcoin'), 'sfm_ch9_nic', 'SFM_ch9_asymmetry', [
    T('Model curves: $\\sigma_t^2$ against $\\varepsilon_{t-1}$, $\\sigma_{t-1}^2$ fixed at the sample variance; dashed: a kernel (Nadaraya--Watson) estimate of $E[r_t^2 \\mid r_{t-1}]$, the idea of SFENewsImpactCurve',
      'Curbele modelelor: $\\sigma_t^2$ în funcție de $\\varepsilon_{t-1}$, cu $\\sigma_{t-1}^2$ fixat la varianța de selecție; linia întreruptă: o estimare nucleu (Nadaraya--Watson) a lui $E[r_t^2 \\mid r_{t-1}]$, ideea din SFENewsImpactCurve'),
    T('S\\&P 500, GJR: after a fall of 2\\%, tomorrow\'s variance is @{nic.sp500.ratio} times the variance after a rise of 2\\%; Bitcoin: ratio @{nic.btc.ratio}, a symmetric curve',
      'S\\&P 500, GJR: după o scădere de 2\\%, varianța de mîine este de @{nic.sp500.ratio} ori mai mare decît după o creștere de 2\\%; Bitcoin: raportul @{nic.btc.ratio}, o curbă simetrică')],
    h='0.50\\textheight')


def arow(k):
    return (f'{SHORT[k]} & $@{{as.{k}.a}}$ & $@{{as.{k}.g}}$ & $@{{as.{k}.gt}}$ & $@{{as.{k}.lr}}$ & @{{as.{k}.lrp}} & ${{@{{as.{k}.eg}}}}$ & ${{@{{as.{k}.egt}}}}$')


D.frame(T('Asymmetry in Six Markets', 'Asimetria pe șase piețe'), table(
    'lrrrrrrr', T('& GJR $\\alpha$ & GJR $\\gamma$ & $t(\\gamma)$ & LR & p & EGARCH $\\gamma$ & $t(\\gamma)$',
                  '& GJR $\\alpha$ & GJR $\\gamma$ & $t(\\gamma)$ & LR & p & EGARCH $\\gamma$ & $t(\\gamma)$'),
    [arow(k) for k in ASSETS], size='scriptsize') + items(
    T('Student-t innovations; $t$ statistics with robust SE; LR (likelihood ratio) $= 2(\\ell_{GJR} - \\ell_{GARCH}) \\sim \\chi^2(1)$',
      'Inovații Student-t; statistici $t$ cu SE robuste; LR (likelihood ratio, raportul de verosimilitate) $= 2(\\ell_{GJR} - \\ell_{GARCH}) \\sim \\chi^2(1)$'),
    T('S\\&P 500 and DAX: GJR $\\hat\\alpha = 0$: only bad news raises volatility; a very strong leverage effect',
      'S\\&P 500 și DAX: GJR $\\hat\\alpha = 0$: doar știrile proaste cresc volatilitatea; un efect de levier foarte puternic'),
    T('BET and TLV: a weak effect, significant by LR, yet BIC prefers the symmetric GARCH; SNP: not significant at 5\\%; Bitcoin: none (EGARCH $\\gamma$ slightly positive, not significant)',
      'BET și TLV: un efect slab, semnificativ după testul LR, dar BIC preferă modelul GARCH simetric; SNP: nesemnificativ la 5\\%; Bitcoin: niciun efect (în EGARCH, $\\gamma$ este ușor pozitiv și nesemnificativ)'),
    T('Interpretation: leverage is a property of mature equity markets; in crypto, large rises are as ``frightening\'\' as large falls',
      'Interpretare: efectul de levier este o proprietate a piețelor mature de acțiuni; la cripto, creșterile mari cresc volatilitatea la fel de mult ca scăderile mari')) + ql('SFM_ch9_asymmetry'), 'footnotesize')

D.recap(('Asymmetry and the Leverage Effect', 'asimetria și efectul de levier'), [
    T('GJR: an extra slope $\\gamma$ for negative shocks; EGARCH: a sign term $\\gamma z_{t-1}$ in $\\ln\\sigma_t^2$', 'GJR: o pantă suplimentară $\\gamma$ pentru șocurile negative; EGARCH: un termen de semn $\\gamma z_{t-1}$ în $\\ln\\sigma_t^2$'),
    T('The news impact curve shows the asymmetry at a glance', 'Curba de impact a știrilor arată asimetria dintr-o privire'),
    T('Strong in the S\\&P 500 and the DAX, weak on the BVB, absent in Bitcoin', 'Puternică la S\\&P 500 și DAX, slabă la BVB, absentă la Bitcoin')])

# =============================================================================
# 9. DIAGNOSTIC
# =============================================================================
D.section('Diagnostics', 'Diagnostic')

D.frame(T('Checking a Fitted Model', 'Verificarea unui model estimat'), items(
    (T('If the model is right, the \\textbf{standardised residuals} $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$ are close to i.i.d.\\ with mean 0 and variance 1',
       'Dacă modelul este corect, \\textbf{reziduurile standardizate} $\\hat z_t = (r_t - \\hat\\mu)/\\hat\\sigma_t$ sînt aproape i.i.d., cu media 0 și varianța 1'),
     [T('\\textbf{LB} (Ljung--Box) $Q(10)$ on $\\hat z_t$: no autocorrelation left in the mean (Chapter 7) \\refLjung', '\\textbf{LB} (Ljung--Box) $Q(10)$ pe $\\hat z_t$: nu a rămas autocorelație în medie (Capitolul 7) \\refLjung'),
      T('$Q(10)$ on $\\hat z_t^2$ and the ARCH-LM test: no ARCH effects left (Chapter 8)', '$Q(10)$ pe $\\hat z_t^2$ și testul ARCH-LM: nu au rămas efecte ARCH (Capitolul 8)'),
      T('QQ plot of $\\hat z_t$: is the assumed distribution right? (Section 6)', 'graficul QQ al lui $\\hat z_t$: este corectă distribuția presupusă? (secțiunea 6)')]),
    (T('\\textbf{Sign-bias test} \\refEN: regress $\\hat z_t^2$ on $I_{t-1}$, $I_{t-1}\\hat\\varepsilon_{t-1}$ and $(1 - I_{t-1})\\hat\\varepsilon_{t-1}$',
       '\\textbf{Testul de asimetrie (sign bias)} \\refEN: regresia lui $\\hat z_t^2$ pe $I_{t-1}$, $I_{t-1}\\hat\\varepsilon_{t-1}$ și $(1 - I_{t-1})\\hat\\varepsilon_{t-1}$'),
     [T('joint test $T R^2 \\sim \\chi^2(3)$: a rejection means the sign of past shocks still predicts the variance (asymmetry is missing)',
        'testul comun $T R^2 \\sim \\chi^2(3)$: o respingere înseamnă că semnul șocurilor trecute încă anticipează varianța (lipsește asimetria)')]),
    T('A passed diagnostic does not prove the model right; a failed one shows where it is wrong', 'Un diagnostic fără respingere nu dovedește că modelul este corect; o respingere arată unde greșește')))

chart(T('What GARCH Removes', 'Autocorelațiile eliminate de GARCH'), 'sfm_ch9_acf_diag', 'SFM_ch9_diagnostics', [
    T('ACF of $r_t^2$: $@{acf.r1}$ at lag 1 and still $@{acf.r50}$ at lag 50; ACF of $\\hat z_t^2$ (GARCH(1,1)-t): $@{acf.z1}$ at lag 1, at most $@{acf.zmax}$ in absolute value; band $\\pm@{acf.band}$',
      'ACF a lui $r_t^2$: $@{acf.r1}$ la lagul 1 și încă $@{acf.r50}$ la lagul 50; ACF a lui $\\hat z_t^2$ (GARCH(1,1)-t): $@{acf.z1}$ la lagul 1, cel mult $@{acf.zmax}$ în valoare absolută; banda $\\pm@{acf.band}$'),
    T('Interpretation: three parameters absorb almost all the volatility clustering in the daily data since 2000', 'Interpretare: trei parametri absorb aproape tot volatility clustering din datele zilnice de după 2000')],
    h='0.50\\textheight')

D.frame(T('Diagnostics on Real Data', 'Diagnostic pe date reale'), table(
    'lrrrr', T('S\\&P 500 & $Q(10)$ (p) & $Q(10)$ of squares (p) & ARCH-LM(5) (p) & sign bias $\\chi^2(3)$ (p)',
               'S\\&P 500 & $Q(10)$ (p) & $Q(10)$ al pătratelor (p) & ARCH-LM(5) (p) & sign bias $\\chi^2(3)$ (p)'),
    [T('returns $r_t$', 'randamente $r_t$') + ' & $@{dg.r}$ (@{dg.r.p}) & $@{dg.r2}$ (@{dg.r2.p}) & $@{dg.lm.r}$ ($<$\\,⁅0.001⁆) & --',
     T('$\\hat z_t$, Normal GARCH', '$\\hat z_t$, GARCH Normal') + ' & $@{dg.normal.z}$ (@{dg.normal.z.p}) & $@{dg.normal.z2}$ (@{dg.normal.z2.p}) & $@{dg.normal.lm}$ (@{dg.normal.lm.p}) & --',
     T('$\\hat z_t$, GARCH-t', '$\\hat z_t$, GARCH-t') + ' & $@{dg.t.z}$ (@{dg.t.z.p}) & $@{dg.t.z2}$ (@{dg.t.z2.p}) & $@{dg.t.lm}$ (@{dg.t.lm.p}) & $@{as.sp500.sb}$ (@{as.sp500.sbp})'],
    size='scriptsize') + items(
    T('GARCH removes the ARCH effects ($Q(10)$ of squares: from $@{dg.r2}$ to $@{dg.t.z2}$); a little autocorrelation in the mean remains (constant-mean model)',
      'GARCH elimină efectele ARCH ($Q(10)$ al pătratelor: de la $@{dg.r2}$ la $@{dg.t.z2}$); rămîne puțină autocorelație în medie (modelul are media constantă)'),
    T('The sign-bias test rejects for the symmetric GARCH (sign $t = @{as.sp500.sbt}$): the asymmetry of Section 8 is missing; similar for the DAX (p @{as.dax.sbp.e}) and the BET (p = @{as.bet.sbp})',
      'Testul de asimetrie respinge pentru GARCH simetric ($t$ al semnului $= @{as.sp500.sbt}$): lipsește asimetria din secțiunea 8; la fel pentru DAX (p @{as.dax.sbp.e}) și BET (p = @{as.bet.sbp})'),
    T('Other markets, $Q(10)$ of $\\hat z_t^2$ (GARCH-t): p between @{dm.dax.z2p} (DAX) and @{dm.tlv.z2p} (TLV): no ARCH effect left anywhere',
      'Celelalte piețe, $Q(10)$ al lui $\\hat z_t^2$ (GARCH-t): p între @{dm.dax.z2p} (DAX) și @{dm.tlv.z2p} (TLV): nu a rămas niciun efect ARCH')) + ql('SFM_ch9_diagnostics'), 'footnotesize')

D.recap(('Diagnostics', 'diagnostic'), [
    T('Standardised residuals: no autocorrelation in $\\hat z_t$ and $\\hat z_t^2$, the right distribution', 'Reziduurile standardizate: fără autocorelație în $\\hat z_t$ și $\\hat z_t^2$, distribuția potrivită'),
    T('GARCH(1,1)-t removes the ARCH effects in all six series', 'GARCH(1,1)-t elimină efectele ARCH în toate cele șase serii'),
    T('The sign-bias test points to the missing asymmetry for equity indices', 'Testul de asimetrie arată asimetria care lipsește la indicii de acțiuni')])

# =============================================================================
# 10. ALEGEREA MODELULUI
# =============================================================================
D.section('Model Selection', 'Alegerea modelului')


def msrow(m, lab):
    return (f'{lab} & ' + ' & '.join(f'@{{ms.{m}.{d}.ll}} & @{{ms.{m}.{d}.dbic}}' for d in ['Normal', 't', 'skewedt']))


D.frame(T('Nine Models for the S\\&P 500', 'Nouă modele pentru S\\&P 500'), items(
    T('\\textbf{AIC} (Akaike information criterion) $= -2\\ell + 2k$ \\refAkaike; \\textbf{BIC} (Bayesian information criterion) $= -2\\ell + k\\ln T$ \\refSchwarz; $\\ell$: maximised log-likelihood, $k$ parameters, $T$ observations; smaller is better (Chapter 6)',
      '\\textbf{AIC} (Akaike information criterion, criteriul informațional Akaike) $= -2\\ell + 2k$ \\refAkaike; \\textbf{BIC} (Bayesian information criterion, criteriul informațional bayesian) $= -2\\ell + k\\ln T$ \\refSchwarz; $\\ell$: log-verosimilitatea maximă, $k$ parametri, $T$ observații; valoarea mai mică este mai bună (Capitolul 6)')) + table(
    'lrrrrrr', T('& \\multicolumn{2}{c}{Normal} & \\multicolumn{2}{c}{Student-t} & \\multicolumn{2}{c}{skewed t} \\\\ & log-lik. & $\\Delta$BIC & log-lik. & $\\Delta$BIC & log-lik. & $\\Delta$BIC',
                 '& \\multicolumn{2}{c}{Normal} & \\multicolumn{2}{c}{Student-t} & \\multicolumn{2}{c}{t asimetrică} \\\\ & log-verosim. & $\\Delta$BIC & log-verosim. & $\\Delta$BIC & log-verosim. & $\\Delta$BIC'),
    [msrow('GARCH', 'GARCH(1,1)'), msrow('GJR', 'GJR-GARCH(1,1)'), msrow('EGARCH', 'EGARCH(1,1)')], size='scriptsize') + items(
    T('$\\Delta$BIC: the difference from the best model; AIC gives the same ranking', '$\\Delta$BIC: diferența față de cel mai bun model; AIC dă aceeași ordine'),
    T('Both choices matter: starting from the Normal GARCH, asymmetry alone (Normal EGARCH) lowers BIC by @{ms.gain.asym}, heavy tails alone (GARCH-skewed t) by @{ms.gain.dist}, both together by @{ms.gain.both}',
      'Ambele alegeri contează: pornind de la GARCH cu inovații Normale, doar asimetria (EGARCH cu inovații Normale) reduce BIC cu @{ms.gain.asym}, doar cozile groase (GARCH cu t asimetrică) cu @{ms.gain.dist}, iar ambele împreună cu @{ms.gain.both}'),
    T('Interpretation: EGARCH with skewed-t innovations wins; with @{ret.n} observations even small improvements are ``significant\'\'; the forecast comparison in Section 11 is the real test',
      'Interpretare: EGARCH cu inovații t asimetrice cîștigă; cu @{ret.n} observații chiar și îmbunătățirile mici sînt „semnificative”; comparația prognozelor din secțiunea 11 este testul real')) + ql('SFM_ch9_diagnostics'), 'footnotesize')

D.recap(('Model Selection', 'alegerea modelului'), [
    T('Compare models fitted on the same data with AIC/BIC; BIC penalises parameters more', 'Comparăm modelele estimate pe aceleași date prin AIC/BIC; BIC penalizează mai mult parametrii'),
    T('S\\&P 500: asymmetry and heavy tails both needed; EGARCH-skewed t is the best', 'S\\&P 500: sînt necesare atît asimetria, cît și cozile groase; EGARCH cu t asimetrică este cel mai bun'),
    T('In-sample fit is not forecasting ability', 'Ajustarea în eșantion nu înseamnă capacitate de prognoză')])

# =============================================================================
# 11. PROGNOZE
# =============================================================================
D.section('Volatility Forecasts', 'Prognoza volatilității')

D.frame(T('Multi-Step Forecasts', 'Prognoze pe mai mulți pași'), items(
    (T('One step: $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$ is known at the end of day $t$', 'Un pas: $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$ este cunoscut la sfîrșitul zilei $t$'),
     [T('two steps: $E_t[\\sigma_{t+2}^2] = \\omega + \\alpha E_t[\\varepsilon_{t+1}^2] + \\beta\\sigma_{t+1}^2 = \\omega + (\\alpha + \\beta)\\sigma_{t+1}^2$, since $E_t[\\varepsilon_{t+1}^2] = \\sigma_{t+1}^2$',
        'doi pași: $E_t[\\sigma_{t+2}^2] = \\omega + \\alpha E_t[\\varepsilon_{t+1}^2] + \\beta\\sigma_{t+1}^2 = \\omega + (\\alpha + \\beta)\\sigma_{t+1}^2$, deoarece $E_t[\\varepsilon_{t+1}^2] = \\sigma_{t+1}^2$')]),
    (T('By induction: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$', 'Prin inducție: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
     [T('$E_t$: the expectation given the information at the end of day $t$; $h$: the horizon in days', '$E_t$: media condiționată de informația de la sfîrșitul zilei $t$; $h$: orizontul, în zile'),
      T('the forecast converges to the long-run variance; the speed is set by $\\alpha + \\beta$', 'prognoza converge spre varianța pe termen lung; viteza este dată de $\\alpha + \\beta$')]),
    (T('Variance of the $H$-day return: the sum of the daily forecasts (returns are uncorrelated)', 'Varianța randamentului pe $H$ zile: suma prognozelor zilnice (randamentele sînt necorelate)'),
     [T('$\\mathrm{Var}_t(r_{t+1} + \\dots + r_{t+H}) = \\sum_{h=1}^{H} E_t[\\sigma_{t+h}^2]$, not $H\\sigma_{t+1}^2$', '$\\mathrm{Var}_t(r_{t+1} + \\dots + r_{t+H}) = \\sum_{h=1}^{H} E_t[\\sigma_{t+h}^2]$, nu $H\\sigma_{t+1}^2$'),
      T('the square-root-of-time rule $\\sigma_{t+1}\\sqrt{H}$ is too high in a crisis and too low in a calm period', 'regula rădăcinii pătrate a timpului, $\\sigma_{t+1}\\sqrt{H}$, dă valori prea mari în criză și prea mici într-o perioadă liniștită')])))

D.frame(T('Worked Example: the S\\&P 500 on @{end}', 'Exemplu rezolvat: S\\&P 500 la @{end}'), items(
    T('GARCH(1,1)-t: $\\alpha + \\beta = @{fc.pers}$, $\\bar\\sigma^2 = @{fc.s2bar}$; forecast for the next day $\\sigma_{t+1}^2 = @{fc.s2n}$ (below the long-run level)',
      'GARCH(1,1)-t: $\\alpha + \\beta = @{fc.pers}$, $\\bar\\sigma^2 = @{fc.s2bar}$; prognoza pentru ziua următoare $\\sigma_{t+1}^2 = @{fc.s2n}$ (sub nivelul pe termen lung)'),
    (T('Step by step', 'Pas cu pas'),
     [T('day 10: $@{fc.s2bar} + @{fc.pers}^{9}(@{fc.s2n} - @{fc.s2bar}) = @{fc.s2bar} + @{fc.pers9} \\times (@{fc.s2n} - @{fc.s2bar}) = @{fc.s210}$',
        'ziua 10: $@{fc.s2bar} + @{fc.pers}^{9}(@{fc.s2n} - @{fc.s2bar}) = @{fc.s2bar} + @{fc.pers9} \\times (@{fc.s2n} - @{fc.s2bar}) = @{fc.s210}$'),
      T('10-day variance: the sum of the ten daily forecasts $= @{fc.sum10}$, so the 10-day volatility is $@{fc.vol10}\\%$', 'varianța pe 10 zile: suma celor zece prognoze zilnice $= @{fc.sum10}$, deci volatilitatea pe 10 zile este $@{fc.vol10}\\%$'),
      T('square-root-of-time rule: $\\sqrt{10 \\times @{fc.s2n}} = @{fc.vol10sq}\\%$: too low, because volatility is expected to rise towards its long-run level',
        'regula rădăcinii pătrate a timpului: $\\sqrt{10 \\times @{fc.s2n}} = @{fc.vol10sq}\\%$: prea mică, deoarece volatilitatea este așteptată să crească spre nivelul ei pe termen lung')])))

chart(T('The Term Structure of Volatility', 'Structura la termen a volatilității'), 'sfm_ch9_term_structure', 'SFM_ch9_forecasts', [
    T('S\\&P 500, GARCH(1,1)-t estimated on all data; forecasts made on a calm day (@{ts.calm.d}), at the COVID-19 peak (@{ts.covid.d}) and on @{end}',
      'S\\&P 500, GARCH(1,1)-t estimat pe toate datele; prognoze făcute într-o zi liniștită (@{ts.calm.d}), în vîrful pandemiei COVID-19 (@{ts.covid.d}) și la @{end}'),
    T('Calm day: @{ts.calm.h1}\\% tomorrow, average @{ts.calm.avg250}\\% over a year; COVID-19 peak: @{ts.covid.h1}\\% tomorrow, still @{ts.covid.avg250}\\% on average over a year',
      'Ziua liniștită: @{ts.calm.h1}\\% mîine, în medie @{ts.calm.avg250}\\% pe un an; vîrful COVID-19: @{ts.covid.h1}\\% mîine, încă @{ts.covid.avg250}\\% în medie pe un an'),
    T('Interpretation: all curves move towards the long-run level (@{ts.lr}\\%) at the same slow speed: the slope of the term structure tells whether we are in a calm or a storm',
      'Interpretare: toate curbele se îndreaptă spre nivelul pe termen lung (@{ts.lr}\\%) cu aceeași viteză redusă: panta structurii la termen arată dacă sîntem într-o perioadă liniștită sau într-una turbulentă')],
    h='0.50\\textheight')

D.frame(T('How to Evaluate a Volatility Forecast', 'Evaluarea unei prognoze de volatilitate'), items(
    (T('The true $\\sigma_t^2$ is never observed: we compare the forecast $h_t$ with a noisy \\textbf{proxy}, here $r_t^2$', 'Adevăratul $\\sigma_t^2$ nu este observat niciodată: comparăm prognoza $h_t$ cu o \\textbf{aproximare} zgomotoasă, aici $r_t^2$'),
     [T('$E_{t-1}[r_t^2] = \\sigma_t^2$ (with $\\mu \\approx 0$), but $r_t^2$ is very noisy: low $R^2$ even for a perfect forecast \\refAB',
        '$E_{t-1}[r_t^2] = \\sigma_t^2$ (cu $\\mu \\approx 0$), dar $r_t^2$ este foarte zgomotos: un $R^2$ mic chiar și pentru o prognoză perfectă \\refAB')]),
    (T('\\textbf{QLIKE} loss \\refPatton: $L(r_t^2, h_t) = \\dfrac{r_t^2}{h_t} + \\ln h_t$ (up to a constant); smaller is better',
       'Funcția de pierdere \\textbf{QLIKE} \\refPatton: $L(r_t^2, h_t) = \\dfrac{r_t^2}{h_t} + \\ln h_t$ (pînă la o constantă); valoarea mai mică este mai bună'),
     [T('$h_t$: the variance forecast for day $t$, made at the end of day $t-1$', '$h_t$: prognoza varianței pentru ziua $t$, făcută la sfîrșitul zilei $t-1$'),
      T('ranks forecasts correctly even with a noisy proxy; it is minus the Normal log-likelihood', 'ordonează corect prognozele chiar și cu o aproximare zgomotoasă; este minus log-verosimilitatea Normală'),
      T('penalises forecasts that are too low more than forecasts that are too high', 'penalizează mai mult prognozele prea mici decît pe cele prea mari')]),
    (T('\\textbf{Out of sample}: from 2015, each forecast uses only data up to the day before; the models are re-estimated every 250 days',
       '\\textbf{În afara eșantionului}: din 2015, fiecare prognoză folosește doar date pînă în ziua precedentă; modelele se reestimează la fiecare 250 de zile'),
     [T('the \\textbf{DM} (Diebold--Mariano) test \\refDM: is the mean loss difference zero? $t$ with HAC (heteroskedasticity and autocorrelation consistent) standard errors',
        'testul \\textbf{DM} (Diebold--Mariano) \\refDM: este diferența medie a pierderilor egală cu zero? $t$ cu erori standard HAC (heteroskedasticity and autocorrelation consistent, robuste la heteroscedasticitate și autocorelație)')])))


def frow(k):
    return (f'{NAMES[k] if k != "tlv" else "TLV"} & @{{fe.{k}.n}} & $@{{fe.{k}.q.garch}}$ & $@{{fe.{k}.q.gjr}}$ & $@{{fe.{k}.q.ewma}}$ & ${{@{{fe.{k}.dm}}}}$ (@{{fe.{k}.dmp}})')


D.frame(T('GARCH against EWMA, Out of Sample', 'GARCH față de EWMA, în afara eșantionului'), table(
    'lrrrrr', T('2015--2026 & days & QLIKE GARCH-t & QLIKE GJR-t & QLIKE EWMA & DM $t$, GARCH-t vs EWMA (p)',
                '2015--2026 & zile & QLIKE GARCH-t & QLIKE GJR-t & QLIKE EWMA & DM $t$, GARCH-t față de EWMA (p)'),
    [frow(k) for k in ['sp500', 'dax', 'bet', 'btc']], size='scriptsize') + '\\vspace{-1mm}\n' + \
    '\\begin{center}\\includegraphics[width=0.80\\textwidth,height=0.34\\textheight,keepaspectratio]{sfm_ch9_forecast_eval.pdf}\\end{center}\n\\vspace{-3mm}\n' + items(
    T('Equity indices: GARCH-t beats EWMA ($t < -2$); GJR-t is better still for the S\\&P 500, DAX and BET', 'Indicii de acțiuni: GARCH-t este mai bun decît EWMA ($t < -2$); GJR-t este și mai bun pentru S\\&P 500, DAX și BET'),
    T('Interpretation: Bitcoin: no difference ($t = @{fe.btc.dm}$): its GARCH is an IGARCH, which is almost an EWMA (Section 7)',
      'Interpretare: Bitcoin: nicio diferență ($t = @{fe.btc.dm}$): GARCH-ul lui este un IGARCH, adică aproape un EWMA (secțiunea 7)')) + ql('SFM_ch9_forecasts'), 'scriptsize')

D.recap(('Volatility Forecasts', 'prognoza volatilității'), [
    T('$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; $H$-day variance = sum of daily forecasts',
      '$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; varianța pe $H$ zile = suma prognozelor zilnice'),
    T('Compare forecasts out of sample with QLIKE and the Diebold--Mariano test', 'Comparăm prognozele în afara eșantionului cu QLIKE și testul Diebold--Mariano'),
    T('GARCH beats EWMA where volatility mean-reverts', 'GARCH este mai bun decît EWMA acolo unde volatilitatea revine la medie')])

# =============================================================================
# 12. VaR
# =============================================================================
D.section('VaR 1\\% from a GARCH Model', 'VaR 1\\% dintr-un model GARCH')

D.frame(T('Conditional VaR', 'VaR condiționat'), items(
    (T('\\textbf{VaR} at level $\\alpha$ (value at risk): the loss exceeded with probability $\\alpha$: $\\mathrm{VaR}_\\alpha = -q_\\alpha(r_{t+1})$ (Chapter 6)',
       '\\textbf{VaR} la nivelul $\\alpha$ (value at risk, valoarea expusă la risc): pierderea depășită cu probabilitatea $\\alpha$: $\\mathrm{VaR}_\\alpha = -q_\\alpha(r_{t+1})$ (Capitolul 6)'),
     [T('here $\\alpha$ is the VaR level, not the GARCH parameter; $q_\\alpha$: the $\\alpha$-quantile', 'aici $\\alpha$ este nivelul VaR, nu parametrul GARCH; $q_\\alpha$: cuantila de ordin $\\alpha$'),
      T('here $\\alpha = 1\\%$: VaR 1\\%, a loss exceeded on one day in a hundred', 'aici $\\alpha = 1\\%$: VaR 1\\%, o pierdere depășită într-o zi din o sută')]),
    (T('With $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$', 'Cu $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$'),
     [T('Normal: $q_{0.01} = @{var.qn}$; standardised t: $q_{0.01} = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$', 'Normală: $q_{0{,}01} = @{var.qn}$; t standardizată: $q_{0{,}01} = t_\\nu^{-1}(0{,}01)\\sqrt{(\\nu - 2)/\\nu}$'),
      T('$t_\\nu^{-1}(0.01)$: the 1\\% quantile of the Student-t with $\\nu$ degrees of freedom; the square root rescales it to variance 1', '$t_\\nu^{-1}(0{,}01)$: cuantila de 1\\% a distribuției Student-t cu $\\nu$ grade de libertate; rădăcina o rescalează la varianța 1'),
      T('the VaR moves every day with $\\sigma_{t+1}$: high in storms, low in calm periods', 'VaR se schimbă în fiecare zi odată cu $\\sigma_{t+1}$: mare în perioadele turbulente, mic în perioadele liniștite')]),
    (T('\\textbf{Worked example}: S\\&P 500 on @{end}, GARCH(1,1)-t: $\\hat\\mu = @{var.mu}$, $\\sigma_{t+1} = @{var.sig}$, $\\hat\\nu = @{var.nu}$',
       '\\textbf{Exemplu rezolvat}: S\\&P 500 la @{end}, GARCH(1,1)-t: $\\hat\\mu = @{var.mu}$, $\\sigma_{t+1} = @{var.sig}$, $\\hat\\nu = @{var.nu}$'),
     [T('$q = @{var.t} \\times @{var.sc} = @{var.qt}$; VaR 1\\% $= -(@{var.mu} + @{var.sig} \\times (@{var.qt})) = @{var.v1}\\%$', '$q = @{var.t} \\times @{var.sc} = @{var.qt}$; VaR 1\\% $= -(@{var.mu} + @{var.sig} \\times (@{var.qt})) = @{var.v1}\\%$'),
      T('with Normal $z_t$: $@{var.vn}\\%$; over 10 days (sum of variances, $\\mu$ ignored): $@{var.qa} \\times \\sqrt{@{fc.sum10}}$, about $@{var.v10}\\%$',
        'cu $z_t$ Normale: $@{var.vn}\\%$; pe 10 zile (suma varianțelor, fără $\\mu$): $@{var.qa} \\times \\sqrt{@{fc.sum10}}$, circa $@{var.v10}\\%$')])))

chart(T('VaR 1\\% through the COVID-19 Crash', 'VaR 1\\% în timpul crahului COVID-19'), 'sfm_ch9_var', 'SFM_ch9_forecasts', [
    T('S\\&P 500, July 2019 -- June 2021 (@{vf.n} days): one-day VaR 1\\% from GARCH(1,1)-t (re-estimated every 250 days) and from EWMA with Normal $z_t$',
      'S\\&P 500, iulie 2019 -- iunie 2021 (@{vf.n} zile): VaR 1\\% pe o zi din GARCH(1,1)-t (reestimat la fiecare 250 de zile) și din EWMA cu $z_t$ Normale'),
    T('Exceedances (return below $-$VaR): GARCH-t @{vf.eg}, EWMA-Normal @{vf.ee}; expected about @{vf.exp}; both models react, but only after the first large falls of March 2020',
      'Depășiri (randament sub $-$VaR): GARCH-t @{vf.eg}, EWMA-Normal @{vf.ee}; așteptat circa @{vf.exp}; ambele modele reacționează, dar abia după primele scăderi mari din martie 2020')],
    h='0.50\\textheight')


def vrow(k):
    return (f'{NAMES[k]} & @{{fe.{k}.n}} & @{{fe.{k}.nexp}} & @{{fe.{k}.neg}} (@{{fe.{k}.eg}}\\%) & @{{fe.{k}.nee}} (@{{fe.{k}.ee}}\\%)')


D.frame(T('Exceedances 2015--2026: a Preview of Backtesting', 'Depășiri 2015--2026: o primă privire asupra backtesting-ului'), table(
    'lrrrr', T('VaR 1\\% & days & expected & GARCH-t & EWMA-Normal', 'VaR 1\\% & zile & așteptat & GARCH-t & EWMA-Normal'),
    [vrow(k) for k in ['sp500', 'dax', 'bet', 'btc']], size='footnotesize') + items(
    T('A correct VaR 1\\% is exceeded on 1\\% of the days, and the exceedances do not cluster', 'Un VaR 1\\% corect este depășit în 1\\% dintre zile, iar depășirile nu apar grupat'),
    T('EWMA-Normal: @{fe.ee.min}--@{fe.ee.max}\\%: the Normal quantile is too small for heavy tails', 'EWMA-Normal: @{fe.ee.min}--@{fe.ee.max}\\%: cuantila Normală este prea mică pentru cozi groase'),
    T('GARCH-t: @{fe.eg.min}--@{fe.eg.max}\\%: much closer to 1\\%, but still above it for the S\\&P 500 and the DAX (the leverage effect is missing)',
      'GARCH-t: @{fe.eg.min}--@{fe.eg.max}\\%: mult mai aproape de 1\\%, dar tot peste 1\\% pentru S\\&P 500 și DAX (lipsește efectul de levier)'),
    T('Are these differences significant? Chapter 10: Kupiec\'s test \\refKupiec, ES 2.5\\%, backtesting', 'Sînt aceste diferențe semnificative? Capitolul 10: testul Kupiec \\refKupiec, ES 2,5\\%, backtesting'),
    T('Further reading on VaR forecasting for Bitcoin: \\refPeleB', 'Lectură suplimentară despre prognoza VaR pentru Bitcoin: \\refPeleB')) + ql('SFM_ch9_forecasts'), 'footnotesize')

D.recap(('VaR 1\\% from a GARCH Model', 'VaR 1\\% dintr-un model GARCH'), [
    T('$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_\\alpha(z))$: dynamic volatility and the right quantile', '$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_\\alpha(z))$: volatilitate dinamică și cuantila potrivită'),
    T('Normal quantiles underestimate the risk; Student-t is much closer to 1\\%', 'Cuantilele Normale subestimează riscul; Student-t este mult mai aproape de 1\\%'),
    T('Formal backtests: Chapter 10', 'Testele formale (backtesting): Capitolul 10')])

# =============================================================================
# 13. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Is Bitcoin\'s volatility becoming more like the volatility of an equity index?}', '\\textbf{Devine volatilitatea Bitcoin mai asemănătoare cu volatilitatea unui indice de acțiuni?}'),
     [T('today: IGARCH, $\\hat\\nu = @{m.btc.nu}$, no leverage effect; the S\\&P 500: mean-reverting, strong leverage', 'azi: IGARCH, $\\hat\\nu = @{m.btc.nu}$, fără efect de levier; S\\&P 500: revenire la medie, efect de levier puternic'),
      T('since 2024, spot Bitcoin ETFs (exchange-traded funds) bring equity investors to the crypto market', 'din 2024, ETF-urile (exchange-traded funds, fonduri tranzacționate la bursă) pe Bitcoin aduc investitorii din piața de acțiuni pe piața cripto')]),
    T('Earlier evidence on crypto assets as an asset class: \\refPeleC; volatility as information: \\refPeleA', 'Dovezi anterioare despre activele cripto ca o clasă de active: \\refPeleC; volatilitatea ca informație: \\refPeleA'),
    T('Why it is open: regime changes, the 2022 crypto crash and the ETF launch overlap; the answer depends on the window', 'Întrebarea rămîne deschisă: schimbările de regim, prăbușirea pieței cripto din 2022 și lansarea ETF-urilor se suprapun; răspunsul depinde de fereastra aleasă'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on GARCH models for crypto assets and the models they use', '\\textbf{Literatura}: o listă a studiilor despre modelele GARCH pentru active cripto și a modelelor folosite'),
    T('\\textbf{Code}: a first draft of rolling GJR-GARCH-t estimates for Bitcoin and the S\\&P 500 with the \\texttt{arch} package', '\\textbf{Cod}: o primă versiune a estimărilor GJR-GARCH-t pe ferestre mobile pentru Bitcoin și S\\&P 500, cu pachetul \\texttt{arch}'),
    T('\\textbf{Robustness}: other windows, Ethereum, weekend days removed, skewed-t innovations', '\\textbf{Robustețe}: alte ferestre, Ethereum, eliminarea zilelor de weekend, inovații t asimetrice'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that estimates a GJR-GARCH(1,1) model with Student-t innovations on daily Bitcoin and S\\&P 500 log returns in percent, on rolling windows of 750 days moved by 21 days, and plots the persistence, gamma and nu over time.}',
        '\\aiprompt{Write Python code that estimates a GJR-GARCH(1,1) model with Student-t innovations on daily Bitcoin and S\\&P 500 log returns in percent, on rolling windows of 750 days moved by 21 days, and plots the persistence, gamma and nu over time.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Units: returns in \\% (not decimals), otherwise the optimiser may stop at a wrong $\\omega$', 'Unitățile: randamente în \\% (nu în zecimale), altfel algoritmul de optimizare se poate opri la un $\\omega$ greșit'),
    T('The persistence of GJR is $\\alpha + \\beta + \\gamma/2$, not $\\alpha + \\beta$; for EGARCH it is $\\beta$', 'Persistența GJR este $\\alpha + \\beta + \\gamma/2$, nu $\\alpha + \\beta$; pentru EGARCH este $\\beta$'),
    T('Estimates on the boundary ($\\alpha + \\beta = 1$): no half-life, no long-run volatility; do not report infinite numbers', 'Estimările pe frontieră ($\\alpha + \\beta = 1$): fără timp de înjumătățire și fără volatilitate pe termen lung; nu raportăm valori infinite'),
    T('Calendars: Bitcoin trades 7 days a week, the S\\&P 500 five; annualise each series with its own frequency', 'Calendarele: Bitcoin se tranzacționează 7 zile pe săptămînă, S\\&P 500 cinci; anualizăm fiecare serie cu propria frecvență'),
    T('Rolling windows overlap: consecutive estimates are not independent evidence', 'Ferestrele mobile se suprapun: estimările consecutive nu sînt dovezi independente'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: has Bitcoin\'s volatility moved closer to equity volatility since 2021?', '\\textbf{Întrebarea}: s-a apropiat volatilitatea Bitcoin de volatilitatea acțiunilor după 2021?'),
     [T('data: Bitcoin since 2014, Ethereum since 2015, S\\&P 500 and BET (EODHD)', 'date: Bitcoin din 2014, Ethereum din 2015, S\\&P 500 și BET (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('GJR-GARCH-t on 2015--2019 and on 2021--2026: persistence, $\\gamma$, $\\nu$', 'GJR-GARCH-t pe 2015--2019 și pe 2021--2026: persistența, $\\gamma$, $\\nu$'),
      T('correlation of the Bitcoin and S\\&P 500 GARCH volatilities in each period', 'corelația volatilităților GARCH ale Bitcoin și S\\&P 500 în fiecare perioadă'),
      T('out-of-sample QLIKE of GARCH against EWMA in each period', 'QLIKE în afara eșantionului pentru GARCH față de EWMA, în fiecare perioadă')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Returns: unpredictable mean, predictable variance: $r_t = \\mu + \\sigma_t z_t$', 'Randamentele: media imprevizibilă, varianța previzibilă: $r_t = \\mu + \\sigma_t z_t$'),
    T('GARCH(1,1) with three parameters captures volatility clustering in all our series; $\\alpha + \\beta$ is close to 1', 'GARCH(1,1), cu trei parametri, surprinde volatility clustering în toate seriile noastre; $\\alpha + \\beta$ este apropiat de 1'),
    T('Estimate by maximum likelihood, report robust SE; Student-t or skewed-t innovations for the tails', 'Estimăm prin verosimilitate maximă și raportăm SE robuste; inovații Student-t sau t asimetrice pentru cozi'),
    T('Equity indices need asymmetry (GJR, EGARCH); Bitcoin does not, and is an IGARCH', 'Indicii de acțiuni au nevoie de asimetrie (GJR, EGARCH); Bitcoin nu, și este un IGARCH'),
    T('Forecasts revert to the long-run level; GARCH beats EWMA out of sample where volatility mean-reverts', 'Prognozele revin la nivelul pe termen lung; GARCH este mai bun decît EWMA în afara eșantionului acolo unde volatilitatea revine la medie'),
    T('VaR 1\\% = $-(\\mu + \\sigma_{t+1}q_{0.01}(z))$: the volatility model and the quantile both matter', 'VaR 1\\% = $-(\\mu + \\sigma_{t+1}q_{0{,}01}(z))$: contează atît modelul volatilității, cît și cuantila')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['ARCH($q$) & $\\sigma_t^2 = \\omega + \\sum_{i=1}^{q}\\alpha_i\\varepsilon_{t-i}^2$, \\quad $\\bar\\sigma^2 = \\omega/(1 - \\sum\\alpha_i)$',
     'GARCH(1,1) & $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, \\quad $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$',
     T('Half-life', 'Timpul de înjumătățire') + ' & $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$',
     'GJR-GARCH & $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$',
     'EGARCH & $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$',
     T('Log-likelihood (Normal)', 'Log-verosimilitatea (Normală)') + ' & $\\ell = -\\frac12\\sum_t[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu)^2/\\sigma_t^2]$',
     T('Forecast', 'Prognoza') + ' & $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
     'QLIKE & $L = r_t^2/h_t + \\ln h_t$',
     'VaR 1\\% & $-(\\mu + \\sigma_{t+1}\\,q_{0.01}(z))$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: $\\omega = 0.05$, $\\alpha = 0.05$, $\\beta = 0.90$. What are the long-run variance and the half-life?', '\\textbf{Întrebare}: $\\omega = 0{,}05$, $\\alpha = 0{,}05$, $\\beta = 0{,}90$. Care sînt varianța pe termen lung și timpul de înjumătățire?'),
     [T('\\textbf{Answer}: $0.05/0.05 = 1$; $\\ln 0.5/\\ln 0.95 = 13.5$ days', '\\textbf{Răspuns}: $0{,}05/0{,}05 = 1$; $\\ln 0{,}5/\\ln 0{,}95 = 13{,}5$ zile')]),
    (T('\\textbf{Question}: GJR gives $\\hat\\alpha = 0$, $\\hat\\gamma = 0.2$. What happens after a rise of 3\\%?', '\\textbf{Întrebare}: GJR dă $\\hat\\alpha = 0$, $\\hat\\gamma = 0{,}2$. Ce se întîmplă după o creștere de 3\\%?'),
     [T('\\textbf{Answer}: nothing beyond the decay $\\beta\\sigma_t^2$: only negative shocks raise the variance', '\\textbf{Răspuns}: nimic în afară de scăderea $\\beta\\sigma_t^2$: doar șocurile negative cresc varianța')]),
    (T('\\textbf{Question}: $Q(10)$ of $\\hat z_t^2$ has p $= 0.40$. Is the model correct?', '\\textbf{Întrebare}: $Q(10)$ al lui $\\hat z_t^2$ are p $= 0{,}40$. Este modelul corect?'),
     [T('\\textbf{Answer}: we only know that no ARCH effect is left; asymmetry or the wrong distribution may remain', '\\textbf{Răspuns}: știm doar că nu a rămas niciun efect ARCH; pot rămîne asimetria sau o distribuție greșită')]),
    T('Next: Chapter 10, VaR, ES and backtesting', 'Urmează: Capitolul 10, VaR, ES și backtesting')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
