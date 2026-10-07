r"""
build_chapter13.py -- Capitolul 13 (Machine learning în finanțe), EN + RO dintr-o singură sursă
==============================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_13/ch13_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Convenția nivelului: VaR 1% (VaR_alpha = -q_alpha); niciodată „VaR 99%”.
Ieșire:
  EN/Courses/chapter13_machine_learning.tex
  RO/Cursuri/capitol13_invatare_automata.tex
Rulare:
  python3 Quantlets/Ch_13/generate_all_charts.py
  python3 latex/build_chapter13.py && python3 latex/sfm_build.py compile 13
"""

import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch13_common import QLURL, REFS, T, _BIB as BIBKEYS, bib, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(13, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_13}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.56\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'mark1': ('ch13_mark1_perceptron_1960.png', C + 'Mark_I_Perceptron,_Figure_2_of_operator\\%27s_manual.png',
              '⟦Figure||Figură⟧: J.C. Hay, A.E. Murray (1960); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'evans': ('ch13_evans_hall_2021.jpg', C + 'Evans_Hall_(UC_Berkeley).jpg',
              '⟦Photo||Foto⟧: Gabriel Classon (2021); CC BY 2.0; Wikimedia Commons'),
    'tib': ('ch13_tibshirani_2012.jpg', C + 'Robert_tibshirani.jpg',
            '⟦Photo||Foto⟧: R. Tibshirani (2012); CC BY-SA 3.0; Wikimedia Commons'),
    'blue': ('ch13_deep_blue_2011.jpg', C + 'IBM_Deep_Blue_at_Computer_History_Museum_(9361685537).jpg',
             '⟦Photo||Foto⟧: Anton Chiang (2011); CC BY 2.0; Wikimedia Commons'),
    'hinton': ('ch13_hinton_2024.jpg', C + 'Geoffrey_E._Hinton,_2024_Nobel_Prize_Laureate_in_Physics_(cropped1).jpg',
               '⟦Photo||Foto⟧: Arthur Petron (2024); CC BY-SA 4.0; Wikimedia Commons'),
    'harper': ('ch13_harper_center_2013.jpg', C + 'University_of_Chicago_July_2013_01_(Charles_M._Harper_Center).jpg',
               '⟦Photo||Foto⟧: Michael Barera (2013); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
# a tree split step by step: six weeks, x = this week's volatility (%), y = next week's volatility (%)
TX = [0.6, 0.8, 1.0, 1.5, 2.0, 2.4]
TY = [0.7, 0.9, 0.8, 1.6, 2.1, 2.2]
V.put('ts.mean', np.mean(TY), 2)
V.put('ts.sse0', np.sum((np.array(TY) - np.mean(TY)) ** 2), 3)
split_rows = []
best = None
for i in range(1, len(TX)):
    s = (TX[i - 1] + TX[i]) / 2
    L, R = np.array(TY[:i]), np.array(TY[i:])
    sse = float(((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum())
    split_rows.append((s, L.mean(), R.mean(), sse))
    if best is None or sse < best[3]:
        best = (s, L.mean(), R.mean(), sse)
V.put('ts.best', best[0], 2)
V.put('ts.bl', best[1], 2)
V.put('ts.br', best[2], 2)
V.put('ts.bsse', best[3], 3)
V.put('ts.r2', 1 - best[3] / np.sum((np.array(TY) - np.mean(TY)) ** 2), 2)
# ridge and lasso with one centred feature: Sxx = 100, Sxy = 30
V.put('rl.ols', 30 / 100, 2)
V.put('rl.ridge', 30 / (100 + 50), 2)
V.put('rl.fac', 100 / 150, 2)
V.put('rl.lasso', (30 - 20 / 2) / 100, 2)
# variance of an average of B correlated trees: rho sigma^2 + (1 - rho) sigma^2 / B, rho = 0.3, sigma^2 = 1
for B in (1, 10, 300):
    V.put(f'rf.v{B}', 0.3 + 0.7 / B, 3)
# MLP parameters: 11 inputs, 16 hidden neurons, one output
V.raw('mlp.p1', str(11 * 16 + 16))
V.raw('mlp.p2', str(16 + 1))
V.raw('mlp.p', str(11 * 16 + 16 + 16 + 1))
# out-of-sample R^2 against HAR, step by step (S&P 500, MLP)
rv = N['rv']['sp500']
V.put('oos.mse.har', rv['HAR']['mse'], 3)
V.put('oos.mse.mlp', rv['MLP']['mse'], 3)
V.put('oos.ratio', rv['MLP']['mse'] / rv['HAR']['mse'], 3)
# QLIKE for an under- and an over-forecast of a variance of 1 (Chapter 8)
V.put('ql.u', 1 / 0.5 + math.log(0.5), 3)
V.put('ql.o', 1 / 2 + math.log(2), 3)
# the sign: z test of the random forest against the baseline (S&P 500)
sg = N['sign']['sp500']
V.put('zt.se', 100 * math.sqrt(sg['base_acc'] * (1 - sg['base_acc']) / sg['n']), 2)
V.put('zt.band', 196 * math.sqrt(sg['base_acc'] * (1 - sg['base_acc']) / sg['n']), 1)
# the expected maximum Sharpe ratio, 10 years: sd = 1/sqrt(10)
V.put('ms.sd', 1 / math.sqrt(10), 3)
g = 0.5772156649
V.put('ms.z1', stats.norm.ppf(1 - 1 / 100), 3)
V.put('ms.z2', stats.norm.ppf(1 - 1 / (100 * math.e)), 3)
# DSR: annualisation of the per-day Sharpe ratio
Sn = N['snoop']
V.put('sn.sr0a', Sn['dsr']['sr0'] * math.sqrt(252), 2)
V.put('sn.zN1', stats.norm.ppf(1 - 1 / Sn['N']), 3)
V.put('sn.zN2', stats.norm.ppf(1 - 1 / (Sn['N'] * math.e)), 3)
V.put('sqrt252', math.sqrt(252), 2)
def de_ro(x):
    """'de ' after a numeral whose last two digits are 20 or more (or 00), as in Romanian."""
    v = int(round(x))
    return 'de ' if v >= 20 and (v % 100 >= 20 or v % 100 == 0) else ''


V.raw('jp.de', de_ro(N['nnjpy']['rmse_te'] / N['nnjpy']['rmse_rw_te']))
for lr in ('0.3', '0.03'):
    V.raw('en.gb' + lr.replace('0.', '') + '.de', de_ro(N['ens']['gb_best_n'][lr]))
# GKX: OLS with all covariates (Table 1)
V.put('gkx.ols', -3.46, 2)
# ranges quoted in the text
qs = N['qvar']['sp500']

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can flexible prediction methods forecast financial risk and returns better than the models of Chapters 7--10, and how do we check it honestly?',
       '\\textbf{Întrebarea}: pot metodele flexibile de predicție să prognozeze riscul și randamentele financiare mai bine decît modelele din capitolele 7--10 și cum verificăm onest acest lucru?'),
     [T('\\textbf{machine learning} (ML): algorithms that learn a prediction rule from data, judged by their accuracy on new data',
        '\\textbf{machine learning} (ML, învățare automată): algoritmi care învață din date o regulă de predicție, judecați după acuratețea pe date noi')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('prediction against inference; overfitting, bias and variance; validation for time series without look-ahead bias',
        'predicție și inferență; overfitting, deplasare și varianță; validarea pentru serii de timp fără look-ahead bias'),
      T('ridge and lasso; trees, random forest, gradient boosting; neural networks and the neural-network ARCH of the textbook',
        'ridge și lasso; arbori de decizie, random forest, gradient boosting; rețele neuronale și modelul ARCH cu rețea neuronală din manual'),
      T('three applications: realised volatility, the sign of returns, VaR 1\\%; data snooping; a landmark paper',
        'trei aplicații: volatilitatea realizată, semnul randamentelor, VaR 1\\%; data snooping; o lucrare de referință')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Explain the difference between prediction and inference, and the bias--variance trade-off', 'Explicați diferența dintre predicție și inferență, precum și compromisul deplasare--varianță'),
    T('Design a walk-forward evaluation with purging, and recognise look-ahead bias and leakage', 'Construiți o evaluare walk-forward cu purging și recunoașteți look-ahead bias și leakage-ul'),
    T('Fit ridge, lasso, a regression tree, a random forest, gradient boosting and a small neural network', 'Estimați ridge, lasso, un arbore de regresie, un random forest, gradient boosting și o rețea neuronală mică'),
    T('Compare forecasts with out-of-sample $R^2$, QLIKE and the Diebold--Mariano test, against a proper baseline', 'Comparați prognozele prin $R^2$ în afara eșantionului, QLIKE și testul Diebold--Mariano, față de un reper adecvat'),
    T('Forecast VaR 1\\% by quantile regression and correct a Sharpe ratio for the number of strategies tried', 'Prognozați VaR 1\\% prin regresie cuantilică și corectați un raport Sharpe pentru numărul de strategii încercate')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~19 (neural networks)', 'Manual: \\refFHH, cap.~19 (rețele neuronale)'),
     [T('statistical learning in Python: \\refISL; in depth: \\refHTF', 'învățare statistică în Python: \\refISL; aprofundare: \\refHTF'),
      T('machine learning in finance: \\refKX; \\refLdP', 'machine learning în finanțe: \\refKX; \\refLdP')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_13}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_13}'),
     [T('ported from the SFE Quantlets \\refSFEa{} and \\refSFEb; models from scikit-learn (small, fixed seeds, a few minutes in Colab)',
        'portate din Quantlet-urile SFE \\refSFEa{} și \\refSFEb; modele din scikit-learn (mici, cu semințe fixe, cîteva minute în Colab)')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter13_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter13_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video courses: \\quantinar{Machine learning in Financial Risk}{https://quantinar.com/course/934/machine-learning-in-financial-risk}; \\quantinar{Random Forests}{https://quantinar.com/course/68/RF}',
      'Cursuri video: \\quantinar{Machine learning in Financial Risk}{https://quantinar.com/course/934/machine-learning-in-financial-risk}; \\quantinar{Random Forests}{https://quantinar.com/course/68/RF}')))

# =============================================================================
# 1. MACHINE LEARNING ȘI STATISTICA
# =============================================================================
D.section('What Machine Learning Adds to Statistics', 'Contribuția machine learning în statistică')

D.frame(T('The Perceptron, 1958', 'Perceptronul, 1958'), cols(items(
    (T('\\refRosenblatt: a machine that learns to classify images by adjusting weights after each error', '\\refRosenblatt: o mașină care învață să clasifice imagini ajustîndu-și ponderile după fiecare eroare'),
     [T('the \\textbf{perceptron}: output $1$ if $w_0 + w^\\top x > 0$, else $0$',
        '\\textbf{perceptronul}: ieșirea este $1$ dacă $w_0 + w^\\top x > 0$ și $0$ în caz contrar'),
      T('$x$: the inputs (photocell signals); $w$: their weights, learned from examples; $w_0$: the threshold term; $w^\\top x = \\sum_j w_jx_j$',
        '$x$: vectorul intrărilor (semnalele fotocelulelor); $w$: ponderile lor, învățate din exemple; $w_0$: termenul de prag; $w^\\top x = \\sum_j w_jx_j$'),
      T('the Mark I Perceptron (Cornell, 1960): 400 photocells, weights set by electric motors', 'Mark I Perceptron (Cornell, 1960): 400 de fotocelule, ponderi reglate de motoare electrice')]),
    (T('Three later milestones', 'Trei repere ulterioare'),
     [T('1986: back-propagation trains networks with hidden layers \\refRHW', '1986: retropropagarea erorii (back-propagation) antrenează rețele cu straturi ascunse \\refRHW'),
      T('2001: random forests \\refBreiman{} and gradient boosting \\refFriedman', '2001: random forest \\refBreiman{} și gradient boosting \\refFriedman'),
      T('2024: Nobel Prize in Physics to Hopfield and Hinton for neural networks', '2024: Premiul Nobel pentru fizică acordat lui Hopfield și Hinton pentru rețelele neuronale')])),
    ph('mark1', T('The Mark I Perceptron, from its operator\'s manual', 'Mark I Perceptron, din manualul de utilizare'), h='0.40\\textheight'),
    wl='0.56', wr='0.40'), 'footnotesize')

D.frame(T('Two Cultures of Statistical Modelling', 'Două culturi ale modelării statistice'), cols(items(
    (T('\\refBreimanTC, a statistician at Berkeley, describes two cultures', '\\refBreimanTC, statistician la Berkeley, descrie două culturi'),
     [T('\\textbf{data modelling}: assume a stochastic model (the linear regression $y = x^\\top\\beta + \\varepsilon$, GARCH), estimate it, test its parameters',
        '\\textbf{modelarea datelor}: presupunem un model stochastic (regresia liniară $y = x^\\top\\beta + \\varepsilon$, GARCH), îl estimăm și îi testăm parametrii'),
      T('\\textbf{algorithmic modelling}: treat the mechanism as unknown; find a rule $\\hat f$ that predicts $y$ well on new data',
        '\\textbf{modelarea algoritmică}: tratăm mecanismul ca necunoscut; căutăm o regulă $\\hat f$ care prezice bine $y$ pe date noi')]),
    (T('This course used the first culture in Chapters 2--11', 'Cursul a folosit prima cultură în capitolele 2--11'),
     [T('this chapter adds the second, and asks when it pays off in finance', 'acest capitol o adaugă pe a doua și analizează în ce condiții este utilă în finanțe')])),
    ph('evans', T('Evans Hall, University of California, Berkeley, home of the Department of Statistics', 'Evans Hall, Universitatea California din Berkeley, sediul Departamentului de Statistică'), h='0.46\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('Prediction against Inference', 'Predicție și inferență'), items(
    (T('\\textbf{Inference}: is $\\beta_j \\ne 0$? how large is the effect? (standard errors, tests, Chapter 6)', '\\textbf{Inferența}: este $\\beta_j \\ne 0$? cît de mare este efectul? (erori standard, teste, Capitolul 6)'),
     [T('example: does the leverage effect exist (GJR-GARCH, Chapter 9)?', 'exemplu: există efectul de levier (GJR-GARCH, Capitolul 9)?')]),
    (T('\\textbf{Prediction}: how close is $\\hat y$ to $y$ on data the model has never seen?', '\\textbf{Predicția}: cît de aproape este $\\hat y$ de $y$ pe date pe care modelul nu le-a văzut niciodată?'),
     [T('example: next week\'s volatility; tomorrow\'s VaR 1\\%', 'exemplu: volatilitatea săptămînii viitoare; VaR 1\\% de mîine'),
      T('the coefficients may be uninterpretable; what matters is the error on new data', 'coeficienții pot fi greu de interpretat; contează eroarea pe date noi')]),
    (T('The two goals can conflict', 'Cele două obiective pot intra în conflict'),
     [T('a biased estimator (ridge, lasso) can predict better than the unbiased OLS', 'un estimator deplasat (ridge, lasso) poate prezice mai bine decît estimatorul nedeplasat OLS'),
      T('a model that predicts well does not prove a causal mechanism', 'un model care prezice bine nu demonstrează un mecanism cauzal')])))

D.frame(T('Supervised Learning: Notation and Loss (1/2)', 'Învățarea supervizată: notații și funcția de pierdere (1/2)'), items(
    (T('Data $(x_t, y_t)$, $t = 1, \\dots, n$', 'Datele: perechile $(x_t, y_t)$, $t = 1, \\dots, n$'),
     [T('$t$: the index of the observation (a day, a week); $n$: the number of observations', '$t$: indicele observației (o zi, o săptămînă); $n$: numărul de observații'),
      T('$x_t \\in \\mathbb{R}^p$: the \\textbf{features}, $p$ numbers known at time $t$ (past volatilities, past returns)', '$x_t \\in \\mathbb{R}^p$: \\textbf{variabilele explicative} (features), $p$ numere cunoscute la momentul $t$ (volatilități și randamente trecute)'),
      T('$y_t$: the \\textbf{target}, the quantity we want to forecast (next week\'s volatility)', '$y_t$: \\textbf{ținta} (target), mărimea pe care vrem să o prognozăm (volatilitatea săptămînii viitoare)')]),
    (T('Two types of problems', 'Două tipuri de probleme'),
     [T('\\textbf{regression}: $y$ continuous (volatility, return)', '\\textbf{regresie}: $y$ este o variabilă continuă (volatilitate, randament)'),
      T('\\textbf{classification}: $y$ a class (up/down, default; Chapter 12)', '\\textbf{clasificare}: $y$ este o clasă (creștere/scădere, neplată; Capitolul 12)')])))

D.frame(T('Supervised Learning: Notation and Loss (2/2)', 'Învățarea supervizată: notații și funcția de pierdere (2/2)'), items(
    (T('Learning = choose $\\hat f$ in a family $\\mathcal{F}$ that minimises the average \\textbf{loss} $\\frac1n\\sum_t L(y_t, f(x_t))$',
       'Învățarea constă în alegerea regulii $\\hat f$ din familia $\\mathcal{F}$ care minimizează \\textbf{pierderea} medie $\\frac1n\\sum_t L(y_t, f(x_t))$'),
     [T('$f(x_t)$: the forecast of a rule $f$ for observation $t$; $L(y, \\hat y)$: the cost of forecasting $\\hat y$ when $y$ is observed',
        '$f(x_t)$: prognoza dată de regula $f$ pentru observația $t$; $L(y, \\hat y)$: costul prognozei $\\hat y$ cînd se observă $y$'),
      T('$\\mathcal{F}$: the candidate rules (all linear regressions, all trees of depth 3); $\\hat f$: the rule chosen from the data',
        '$\\mathcal{F}$: regulile candidate (toate regresiile liniare, toți arborii de adîncime 3); $\\hat f$: regula aleasă pe baza datelor')]),
    (T('Losses used in this course', 'Funcțiile de pierdere folosite în curs'),
     [T('squared error $(y - \\hat y)^2$ (MSE: mean squared error); QLIKE for variances (Chapter 8)', 'eroarea pătratică $(y - \\hat y)^2$ (MSE, mean squared error); QLIKE pentru varianțe (Capitolul 8)'),
      T('quantile (pinball) loss for VaR (Chapter 10); log loss for probabilities (Chapter 12)', 'pierderea cuantilică (pinball loss) pentru VaR (Capitolul 10); log loss pentru probabilități (Capitolul 12)')]),
    (T('A \\textbf{hyperparameter} is chosen before the fit', 'Un \\textbf{hiperparametru} se alege înainte de estimare'),
     [T('examples: tree depth, penalty $\\lambda$, number of neurons', 'exemple: adîncimea arborelui, penalizarea $\\lambda$, numărul de neuroni'),
      T('the parameters (coefficients, weights) are learned by the fit', 'parametrii (coeficienți, ponderi) se învață prin estimare')])))

D.frame(T('Where ML Helps in Finance, and Where It Struggles', 'Unde ajută machine learning în finanțe și unde întîmpină dificultăți'), items(
    (T('\\textbf{Helps}: targets with structure and enough data', '\\textbf{Ajută}: ținte cu structură și cu suficiente date'),
     [T('volatility is persistent (Chapters 8--9): a lot to predict, nonlinear effects possible', 'volatilitatea este persistentă (capitolele 8--9): este mult de prezis, iar efectele neliniare sînt posibile'),
      T('cross-sections with thousands of stocks and many characteristics (the case study)', 'secțiuni transversale cu mii de acțiuni și multe caracteristici (studiul de caz)')]),
    (T('\\textbf{Struggles}: a low \\textbf{signal-to-noise ratio}', '\\textbf{Dificultăți}: un \\textbf{raport semnal--zgomot} mic'),
     [T('daily returns are close to unpredictable (Chapter 7): a flexible model fits the noise', 'randamentele zilnice sînt aproape imprevizibile (Capitolul 7): un model flexibil învață zgomotul'),
      T('markets change (regimes, crises): the past is an imperfect training set', 'piețele se schimbă (regimuri, crize): trecutul este un set de antrenare imperfect'),
      T('one price history: we cannot repeat the experiment, and the observations are dependent', 'o singură istorie a prețurilor: experimentul nu poate fi repetat, iar observațiile sînt dependente')])))

D.recap(('What Machine Learning Adds', 'contribuția machine learning'), [
    T('ML judges a model by its error on new data, not by the significance of its coefficients', 'Machine learning judecă un model după eroarea pe date noi, nu după semnificația coeficienților'),
    T('Features $x_t$, target $y_t$, loss $L$, hyperparameters chosen before the fit', 'Variabile explicative $x_t$, țintă $y_t$, pierdere $L$, hiperparametri aleși înainte de estimare'),
    T('Finance: easy to predict volatility, hard to predict returns', 'Finanțe: volatilitatea se prezice ușor, randamentele greu')])

# =============================================================================
# 2. OVERFITTING
# =============================================================================
D.section('Overfitting, Bias and Variance', 'Overfitting, deplasare și varianță')

D.frame(T('Training Error and Test Error', 'Eroarea de antrenare și eroarea de test'), items(
    (T('\\textbf{Training error}: the average loss on the data used for the fit; \\textbf{test error}: on new data from the same process',
       '\\textbf{Eroarea de antrenare}: pierderea medie pe datele folosite la estimare; \\textbf{eroarea de test}: pe date noi din același proces'),
     [T('the training error always falls as the model becomes more flexible', 'eroarea de antrenare scade întotdeauna cînd modelul devine mai flexibil')]),
    (T('\\textbf{Overfitting}: the model learns the noise of the training sample', '\\textbf{Overfitting}: modelul învață zgomotul din eșantionul de antrenare'),
     [T('symptom: a small training error and a large test error', 'simptom: o eroare de antrenare mică și o eroare de test mare'),
      T('a tree with one leaf per observation has zero training error and predicts badly', 'un arbore cu cîte o frunză pentru fiecare observație are eroare de antrenare zero, dar prognoze slabe')]),
    (T('\\textbf{Underfitting}: the model is too rigid to capture the signal', '\\textbf{Underfitting}: modelul este prea rigid pentru a surprinde semnalul'),
     [T('example: a constant forecast of volatility in a world with volatility clustering', 'exemplu: o prognoză constantă a volatilității într-o lume cu volatility clustering')])))

D.frame(T('The Bias--Variance Decomposition (1/2)', 'Descompunerea deplasare--varianță (1/2)'), items(
    (T('Model: $y = f(x) + \\varepsilon$, $E[\\varepsilon] = 0$, $\\mathrm{Var}(\\varepsilon) = \\sigma^2$',
       'Modelul: $y = f(x) + \\varepsilon$, $E[\\varepsilon] = 0$, $\\mathrm{Var}(\\varepsilon) = \\sigma^2$'),
     [T('$f$: the true, unknown relation between $x$ and $y$; $\\varepsilon$: the noise, unpredictable from $x$, with variance $\\sigma^2$',
        '$f$: relația adevărată, necunoscută, dintre $x$ și $y$; $\\varepsilon$: zgomotul, imprevizibil pe baza lui $x$, cu varianța $\\sigma^2$'),
      T('$\\hat f$: the estimate of $f$ from a random training sample; another sample would give another $\\hat f$',
        '$\\hat f$: estimația lui $f$ obținută pe un eșantion de antrenare aleator; alt eșantion ar da alt $\\hat f$')]),
    (T('Expected test error at a new point $x_0$, with $y_0 = f(x_0) + \\varepsilon_0$:', 'Eroarea de test așteptată într-un punct nou $x_0$, cu $y_0 = f(x_0) + \\varepsilon_0$:') + ' \\[ E\\big[(y_0 - \\hat f(x_0))^2\\big] = \\underbrace{\\big(E[\\hat f(x_0)] - f(x_0)\\big)^2}_{\\text{' + T('bias', 'deplasare') + '}^2} + \\underbrace{\\mathrm{Var}\\big(\\hat f(x_0)\\big)}_{\\text{' + T('variance', 'varianță') + '}} + \\underbrace{\\sigma^2}_{\\text{' + T('noise', 'zgomot') + '}} \\]',
     [T('$E[\\cdot]$: the average over all possible training samples and over the noise $\\varepsilon_0$', '$E[\\cdot]$: media peste toate eșantioanele de antrenare posibile și peste zgomotul $\\varepsilon_0$'),
      T('bias: how far the average estimate is from the truth; variance: how much $\\hat f(x_0)$ moves from one sample to another',
        'deplasarea: distanța dintre estimația medie și valoarea adevărată; varianța: cît se modifică $\\hat f(x_0)$ de la un eșantion la altul')])))

D.frame(T('The Bias--Variance Decomposition (2/2)', 'Descompunerea deplasare--varianță (2/2)'), items(
    (T('Derivation: write $y_0 - \\hat f = (f - E\\hat f) + (E\\hat f - \\hat f) + \\varepsilon_0$, all at $x_0$', 'Derivarea: scriem $y_0 - \\hat f = (f - E\\hat f) + (E\\hat f - \\hat f) + \\varepsilon_0$, toate în punctul $x_0$'),
     [T('square and take the expectation: the three squares give bias$^2$, variance and $\\sigma^2$', 'ridicăm la pătrat și luăm media: cele trei pătrate dau deplasarea$^2$, varianța și $\\sigma^2$'),
      T('the cross terms have mean 0: $E[E\\hat f - \\hat f] = 0$, and $\\varepsilon_0$ is independent of the training sample', 'termenii micști au media 0: $E[E\\hat f - \\hat f] = 0$, iar $\\varepsilon_0$ este independent de eșantionul de antrenare')]),
    (T('Reading the decomposition', 'Interpretarea descompunerii'),
     [T('a rigid model: large bias, small variance', 'un model rigid: deplasare mare, varianță mică'),
      T('a flexible model: small bias, large variance', 'un model flexibil: deplasare mică, varianță mare'),
      T('$\\sigma^2$ is the floor: no model goes below it; in daily returns $\\sigma^2$ is almost everything', '$\\sigma^2$ este pragul minim: niciun model nu coboară sub el; la randamentele zilnice, $\\sigma^2$ reprezintă aproape toată eroarea')])))

chart(T('Bias and Variance of Regression Trees', 'Deplasarea și varianța arborilor de regresie'), 'sfm_ch13_bias_variance', 'SFM_ch13_validation', [
    T('Simulation: $y = \\sin(2\\pi x) + \\varepsilon$, $\\sigma^2 = @{bv.s2}$, $n = @{bv.n}$, 300 training samples; left: one sample with trees of depth 1, 3 and 10',
      'Simulare: $y = \\sin(2\\pi x) + \\varepsilon$, $\\sigma^2 = @{bv.s2}$, $n = @{bv.n}$, 300 de eșantioane de antrenare; stînga: un eșantion cu arbori de adîncime 1, 3 și 10'),
    T('Depth 1: bias$^2$ $@{bv.1.bias2}$, variance $@{bv.1.var}$; depth 10: bias$^2$ $@{bv.10.bias2}$, variance $@{bv.10.var}$; the test error is smallest at depth @{bv.best} ($@{bv.3.test}$)',
      'Adîncimea 1: deplasare$^2$ $@{bv.1.bias2}$, varianță $@{bv.1.var}$; adîncimea 10: deplasare$^2$ $@{bv.10.bias2}$, varianță $@{bv.10.var}$; eroarea de test este minimă la adîncimea @{bv.best} ($@{bv.3.test}$)')],
    h='0.50\\textheight')

D.frame(T('Interpretation: the U-Shaped Test Error', 'Interpretarea erorii de test în formă de U'), items(
    (T('The training error falls from $@{bv.1.train}$ to $@{bv.10.train}$; the test error falls, then rises', 'Eroarea de antrenare scade de la $@{bv.1.train}$ la $@{bv.10.train}$; eroarea de test scade, apoi crește'),
     [T('depth 10 has almost zero training error and a test error of $@{bv.10.test}$, worse than depth 3', 'adîncimea 10 are eroare de antrenare aproape zero și o eroare de test de $@{bv.10.test}$, mai mare decît la adîncimea 3')]),
    (T('Consequences for practice', 'Consecințe practice'),
     [T('never select a model by its training error', 'nu alegeți niciodată un model după eroarea de antrenare'),
      T('complexity is a hyperparameter: choose it on data that were not used for the fit', 'complexitatea este un hiperparametru: se alege pe date care nu au fost folosite la estimare'),
      T('in finance the noise is large, so the best model is usually simpler than in this simulation', 'în finanțe zgomotul este mare, deci cel mai bun model este de obicei mai simplu decît în această simulare')])))

D.recap(('Overfitting, Bias and Variance', 'overfitting, deplasare și varianță'), [
    T('Test error = bias$^2$ + variance + noise', 'Eroarea de test = deplasare$^2$ + varianță + zgomot'),
    T('More flexibility: less bias, more variance; the training error is always too optimistic', 'Mai multă flexibilitate: deplasare mai mică, varianță mai mare; eroarea de antrenare este întotdeauna prea optimistă'),
    T('The amount of flexibility must be chosen on new data', 'Gradul de flexibilitate se alege pe date noi')])

# =============================================================================
# 3. VALIDARE
# =============================================================================
D.section('Validation for Time Series', 'Validarea pentru serii de timp')

D.frame(T('Training, Validation and Test Samples', 'Eșantioanele de antrenare, validare și test'), items(
    (T('Three roles for the data', 'Trei roluri pentru date'),
     [T('\\textbf{training}: estimate the parameters of each candidate model', '\\textbf{antrenare}: estimăm parametrii fiecărui model candidat'),
      T('\\textbf{validation}: choose the hyperparameters and the model', '\\textbf{validare}: alegem hiperparametrii și modelul'),
      T('\\textbf{test}: measure the error of the final choice, once', '\\textbf{test}: măsurăm o singură dată eroarea alegerii finale')]),
    (T('Every look at the test sample turns it into a validation sample', 'Orice decizie luată pe baza eșantionului de test îl transformă într-un eșantion de validare'),
     [T('trying ten models on the test data and reporting the best is model selection, not testing', 'a încerca zece modele pe datele de test și a raporta cel mai bun înseamnă selecție de model, nu testare')]),
    (T('\\textbf{K-fold cross-validation}: split the data into $K$ folds; each fold is the validation set once, the others train the model',
       '\\textbf{Validarea încrucișată K-fold} (cross-validation): împărțim datele în $K$ părți; fiecare parte este o dată setul de validare, iar celelalte antrenează modelul'),
     [T('valid for i.i.d.\\ data; for a time series only under strong conditions (no dependence left in the errors) \\refBHK',
        'validă pentru date i.i.d.; pentru o serie de timp doar în condiții stricte (fără dependență rămasă în erori) \\refBHK')])))

D.frame(T('Look-Ahead Bias and Leakage', 'Look-ahead bias și leakage'), items(
    (T('\\textbf{Look-ahead bias}: the forecast for day $t$ uses information that was not available at the close of day $t - 1$',
       '\\textbf{Look-ahead bias}: prognoza pentru ziua $t$ folosește informație care nu era disponibilă la închiderea zilei $t - 1$'),
     [T('features computed with future prices (a centred moving average, a full-sample mean)', 'variabile calculate cu prețuri viitoare (o medie mobilă centrată, o medie pe tot eșantionul)'),
      T('scaling or standardising with statistics of the whole sample; revised data; today\'s index members (survivorship)', 'scalare sau standardizare cu statistici ale întregului eșantion; date revizuite; componența de azi a indicelui (survivorship)')]),
    (T('\\textbf{Leakage}: information about the test target reaches the training sample', '\\textbf{Leakage}: informația despre ținta de test ajunge în eșantionul de antrenare'),
     [T('overlapping targets: the 21-day return from day $t$ and from day $t + 1$ share 20 days', 'ținte suprapuse: randamentul pe 21 de zile de la ziua $t$ și cel de la ziua $t + 1$ au 20 de zile comune'),
      T('a random fold puts day $t + 1$ in training and day $t$ in test: the model has seen the answer', 'o partiție aleatoare pune ziua $t + 1$ în antrenare și ziua $t$ în test: modelul a văzut răspunsul')]),
    T('Both make a model look better than it can be in real time', 'Ambele fac un model să pară mai bun decît poate fi în timp real')))

chart(T('Three Validation Schemes', 'Trei scheme de validare'), 'sfm_ch13_cv_schemes', 'SFM_ch13_validation', [
    T('\\textbf{Walk-forward}: train on the past, test on the next block, move forward; the expanding window keeps all the past',
      '\\textbf{Walk-forward}: antrenăm pe trecut, testăm pe blocul următor și avansăm; fereastra extinsă păstrează tot trecutul'),
    T('\\textbf{Purged K-fold with embargo} \\refLdP: drop the training rows whose targets overlap the test block (purging) and a few rows after it (embargo)',
      '\\textbf{K-fold cu purging și embargo} \\refLdP: eliminăm rîndurile de antrenare ale căror ținte se suprapun cu blocul de test (purging) și cîteva rînduri de după el (embargo)')],
    h='0.52\\textheight')

D.frame(T('Walk-Forward Validation, Step by Step', 'Validarea walk-forward, pas cu pas'), items(
    (T('The design used in this chapter for the S\\&P 500 (data from @{wf.first})', 'Schema folosită în acest capitol pentru S\\&P 500 (date din @{wf.first})'),
     [T('one fold per calendar year from 2013: @{wf.folds} folds; the model is re-estimated each January', 'cîte o partiție pentru fiecare an calendaristic din 2013: @{wf.folds} partiții; modelul se reestimează în fiecare ianuarie'),
      T('training window: all earlier days (expanding); test: every day of the year', 'fereastra de antrenare: toate zilele anterioare (extinsă); test: fiecare zi a anului')]),
    (T('Purging: the target is the mean variance of the next @{wf.h} days', 'Purging: ținta este varianța medie din următoarele @{wf.h} zile'),
     [T('for 2013 the last @{wf.h} training rows (from @{wf.purge0}) are dropped: their targets reach into 2013', 'pentru 2013 se elimină ultimele @{wf.h} rînduri de antrenare (de la @{wf.purge0}): țintele lor ajung în 2013'),
      T('the 2013 fold trains on @{wf.ntr0} days and tests on @{wf.nte0}; the 2026 fold trains on @{wf.ntr1} days', 'partiția pentru 2013 antrenează pe @{wf.ntr0} zile și testează pe @{wf.nte0}; partiția pentru 2026 antrenează pe @{wf.ntr1} zile')]),
    (T('Inside each training window, hyperparameters are tuned by a walk-forward split of the training data only', 'În interiorul fiecărei ferestre de antrenare, hiperparametrii se aleg printr-o împărțire walk-forward doar a datelor de antrenare'),
     [T('no embargo is needed: the test block always comes after the training block', 'nu este nevoie de embargo: blocul de test vine întotdeauna după cel de antrenare')])))

chart(T('Leakage in Practice: Random Folds on Overlapping Targets', 'Leakage în practică: partiții aleatoare pe ținte suprapuse'), 'sfm_ch13_leakage', 'SFM_ch13_validation', [
    T('Random forest for the next 21-day return from past returns and volatilities; out-of-sample $R^2$ against the training mean',
      'Random forest pentru randamentul din următoarele 21 de zile, pe baza randamentelor și volatilităților trecute; $R^2$ în afara eșantionului față de media de antrenare'),
    T('Simulated random walk: random 5-fold $@{lk.sim.kfold}\\%$, walk-forward $@{lk.sim.wf}\\%$; S\\&P 500 (@{lk.sp500.n} days): $@{lk.sp500.kfold}\\%$ and $@{lk.sp500.wf}\\%$',
      'Mers aleator simulat: 5-fold aleator $@{lk.sim.kfold}\\%$, walk-forward $@{lk.sim.wf}\\%$; S\\&P 500 (@{lk.sp500.n} zile): $@{lk.sp500.kfold}\\%$ și $@{lk.sp500.wf}\\%$')],
    h='0.48\\textheight')

D.frame(T('Interpretation: a Forecast of Pure Noise with $R^2 > 30\\%$', 'Interpretare: o prognoză a zgomotului pur cu $R^2 > 30\\%$'), items(
    (T('The random walk has nothing to predict, yet random folds report $R^2 = @{lk.sim.kfold}\\%$', 'Mersul aleator nu are nimic de prezis, totuși partițiile aleatoare raportează $R^2 = @{lk.sim.kfold}\\%$'),
     [T('the forest memorises neighbouring days, whose targets share 20 of 21 returns with the test day', 'random forest memorează zilele vecine, ale căror ținte au 20 din 21 de randamente comune cu ziua de test')]),
    (T('Walk-forward gives a negative $R^2$: the forest fits noise and loses against the training mean', 'Walk-forward dă un $R^2$ negativ: random forest învață zgomotul și este inferior mediei de antrenare'),
     [T('purging (a gap of 21 days) changes little here, because in walk-forward only the boundary rows overlap', 'purging-ul (eliminarea a 21 de zile) schimbă puțin rezultatul, deoarece în walk-forward doar rîndurile de la graniță se suprapun')]),
    T('Rule: with overlapping targets or dependent data, never shuffle the time order', 'Regulă: cu ținte suprapuse sau date dependente, nu amestecați niciodată ordinea temporală')))

D.recap(('Validation for Time Series', 'validarea pentru serii de timp'), [
    T('Training, validation and test have different jobs; the test sample is used once', 'Antrenarea, validarea și testul au roluri diferite; eșantionul de test se folosește o singură dată'),
    T('Walk-forward with purging respects the arrow of time; random K-fold leaks with overlapping targets', 'Walk-forward cu purging respectă ordinea timpului; K-fold aleator produce leakage atunci cînd țintele se suprapun'),
    T('Look-ahead bias and leakage produce impressive but useless results', 'Look-ahead bias și leakage-ul produc rezultate impresionante, dar inutile')])

# =============================================================================
# 4. RIDGE ȘI LASSO
# =============================================================================
D.section('Penalised Regression: Ridge and Lasso', 'Regresia penalizată: ridge și lasso')

D.frame(T('Many Features, Unstable OLS', 'Multe variabile explicative, OLS instabil'), items(
    (T('OLS (ordinary least squares): $\\hat\\beta = \\arg\\min_\\beta \\sum_t (y_t - x_t^\\top\\beta)^2 = (X^\\top X)^{-1}X^\\top y$', 'OLS (ordinary least squares, metoda celor mai mici pătrate): $\\hat\\beta = \\arg\\min_\\beta \\sum_t (y_t - x_t^\\top\\beta)^2 = (X^\\top X)^{-1}X^\\top y$'),
     [T('$X$: the $n \\times p$ matrix whose row $t$ is $x_t^\\top$; $y$: the vector of targets; $\\arg\\min_\\beta$: the $\\beta$ that minimises the sum',
        '$X$: matricea $n \\times p$ cu rîndul $t$ egal cu $x_t^\\top$; $y$: vectorul țintelor; $\\arg\\min_\\beta$: valoarea lui $\\beta$ care minimizează suma'),
      T('unbiased, but its variance $\\sigma^2(X^\\top X)^{-1}$ ($\\sigma^2$: the error variance) explodes when features are many or strongly correlated', 'nedeplasat, dar varianța lui, $\\sigma^2(X^\\top X)^{-1}$ ($\\sigma^2$: varianța erorii), crește foarte mult cînd variabilele sînt multe sau puternic corelate')]),
    (T('Volatility features are highly correlated: today\'s, this week\'s and this month\'s realised variance', 'Variabilele de volatilitate sînt puternic corelate: varianța realizată de azi, din această săptămînă și din această lună'),
     [T('OLS may give large coefficients of opposite signs that cancel in sample and fail out of sample', 'OLS poate da coeficienți mari, de semne opuse, care se compensează în eșantion și eșuează în afara lui')]),
    (T('\\textbf{Penalised regression}: minimise the squared error plus a penalty on the size of $\\beta$', '\\textbf{Regresia penalizată}: minimizăm eroarea pătratică plus o penalizare pentru mărimea lui $\\beta$'),
     [T('accept a little bias for a large reduction in variance (the trade-off of Section 2)', 'acceptăm o deplasare mică pentru o reducere mare a varianței (compromisul din secțiunea 2)')])))

D.frame(T('Ridge Regression (1/2)', 'Regresia ridge (1/2)'), items(
    T('\\refHK: minimise the squared error plus a penalty on the squared coefficients', '\\refHK: minimizăm eroarea pătratică plus o penalizare a pătratelor coeficienților') + ' \\[ \\hat\\beta^{\\,ridge} = \\arg\\min_\\beta \\sum_t (y_t - x_t^\\top\\beta)^2 + \\lambda\\sum_j \\beta_j^2 = (X^\\top X + \\lambda I)^{-1}X^\\top y \\]',
    (T('Notation', 'Notațiile'),
     [T('$\\beta_j$: the coefficient of feature $j$; $\\sum_j \\beta_j^2$: the penalty term, large when the coefficients are large', '$\\beta_j$: coeficientul variabilei $j$; $\\sum_j \\beta_j^2$: termenul de penalizare, mare cînd coeficienții sînt mari'),
      T('$\\lambda \\ge 0$: the strength of the penalty; $\\lambda = 0$ gives OLS, $\\lambda \\to \\infty$ gives $\\hat\\beta \\to 0$', '$\\lambda \\ge 0$: intensitatea penalizării; $\\lambda = 0$ dă OLS, $\\lambda \\to \\infty$ dă $\\hat\\beta \\to 0$'),
      T('$I$: the $p \\times p$ identity matrix', '$I$: matricea identitate $p \\times p$')]),
    (T('Consequence', 'Consecința'),
     [T('$X^\\top X + \\lambda I$ is always invertible for $\\lambda > 0$: ridge works even with more features than observations', '$X^\\top X + \\lambda I$ este întotdeauna inversabilă pentru $\\lambda > 0$: ridge funcționează chiar și cu mai multe variabile decît observații')])))

D.frame(T('Ridge Regression (2/2)', 'Regresia ridge (2/2)'), items(
    (T('\\textbf{One feature}, centred (mean 0): $S_{xx} = \\sum_t x_t^2$, $S_{xy} = \\sum_t x_ty_t$', '\\textbf{O singură variabilă}, centrată (cu media 0): $S_{xx} = \\sum_t x_t^2$, $S_{xy} = \\sum_t x_ty_t$') + ' \\[ \\hat\\beta^{\\,ridge} = \\dfrac{S_{xy}}{S_{xx} + \\lambda} = \\dfrac{S_{xx}}{S_{xx} + \\lambda}\\,\\hat\\beta^{\\,OLS}, \\qquad \\hat\\beta^{\\,OLS} = \\dfrac{S_{xy}}{S_{xx}} \\]',
     [T('the factor $S_{xx}/(S_{xx} + \\lambda)$ lies in $(0, 1]$: the share of the OLS coefficient that ridge keeps', 'factorul $S_{xx}/(S_{xx} + \\lambda)$ este în $(0, 1]$: fracțiunea din coeficientul OLS păstrată de ridge')]),
    (T('Example: $S_{xx} = 100$, $S_{xy} = 30$', 'Exemplu: $S_{xx} = 100$, $S_{xy} = 30$'),
     [T('OLS: $30/100 = @{rl.ols}$', 'OLS: $30/100 = @{rl.ols}$'),
      T('ridge with $\\lambda = 50$: $30/150 = @{rl.ridge}$ (shrinkage factor $@{rl.fac}$)', 'ridge cu $\\lambda = 50$: $30/150 = @{rl.ridge}$ (factor de contracție $@{rl.fac}$)')]),
    T('Ridge shrinks all coefficients towards zero, but sets none exactly to zero', 'Ridge contractă toți coeficienții spre zero, dar nu anulează niciunul exact')))

D.frame(T('The Lasso', 'Lasso'), cols(items(
    (T('\\refTib: $\\hat\\beta^{\\,lasso} = \\arg\\min_\\beta \\sum_t (y_t - x_t^\\top\\beta)^2 + \\lambda\\sum_j |\\beta_j|$', '\\refTib: $\\hat\\beta^{\\,lasso} = \\arg\\min_\\beta \\sum_t (y_t - x_t^\\top\\beta)^2 + \\lambda\\sum_j |\\beta_j|$'),
     [T('the absolute value has a kink at 0: some coefficients become \\textbf{exactly zero} (variable selection)', 'valoarea absolută are un punct unghiular în 0: unii coeficienți devin \\textbf{exact zero} (selecția variabilelor)')]),
    (T('One feature: \\textbf{soft thresholding}', 'O singură variabilă: \\textbf{soft thresholding} (prag de contracție)'),
     [T('$\\hat\\beta^{\\,lasso} = \\mathrm{sign}(S_{xy})\\,\\max(|S_{xy}| - \\lambda/2,\\, 0)/S_{xx}$', '$\\hat\\beta^{\\,lasso} = \\mathrm{sign}(S_{xy})\\,\\max(|S_{xy}| - \\lambda/2,\\, 0)/S_{xx}$'),
      T('$\\mathrm{sign}(S_{xy})$: $+1$ or $-1$; if $|S_{xy}| \\le \\lambda/2$, the coefficient is exactly 0', '$\\mathrm{sign}(S_{xy})$: semnul, $+1$ sau $-1$; dacă $|S_{xy}| \\le \\lambda/2$, coeficientul este exact 0'),
      T('same data, $\\lambda = 20$: $(30 - 10)/100 = @{rl.lasso}$; zero for $\\lambda \\ge 60$', 'aceleași date, $\\lambda = 20$: $(30 - 10)/100 = @{rl.lasso}$; zero pentru $\\lambda \\ge 60$')]),
    (T('In practice', 'În practică'),
     [T('standardise the features first (the penalty depends on their units), with the training mean and standard deviation only',
        'standardizăm întîi variabilele (penalizarea depinde de unitățile lor), doar cu media și abaterea standard din antrenare'),
      T('choose $\\lambda$ by walk-forward cross-validation; elastic net combines both penalties', 'alegem $\\lambda$ prin validare încrucișată walk-forward; elastic net combină cele două penalizări')])),
    ph('tib', T('Robert Tibshirani, Stanford University', 'Robert Tibshirani, Universitatea Stanford'), h='0.42\\textheight'),
    wl='0.64', wr='0.32'), 'footnotesize')

chart(T('Ridge and Lasso Paths for the Volatility Features', 'Traiectoriile ridge și lasso pentru variabilele de volatilitate'), 'sfm_ch13_shrinkage', 'SFM_ch13_shrinkage_trees', [
    T('Target: log of the mean variance proxy of the next 5 days, S\\&P 500, 2008--2012 (@{sh.n} days); 11 standardised features (Section 7)',
      'Ținta: logaritmul mediei variabilei proxy a varianței din următoarele 5 zile, S\\&P 500, 2008--2012 (@{sh.n} de zile); 11 variabile standardizate (secțiunea 7)'),
    T('Lasso with $\\lambda = @{sh.alpha}$ (walk-forward CV): @{sh.nz} of 11 features kept; the week coefficient rises from $@{sh.ols.w}$ (OLS) to $@{sh.cv.w}$, the absolute return drops out',
      'Lasso cu $\\lambda = @{sh.alpha}$ (CV walk-forward): rămîn @{sh.nz} din 11 variabile; coeficientul săptămînal crește de la $@{sh.ols.w}$ (OLS) la $@{sh.cv.w}$, iar randamentul absolut iese din model')],
    h='0.50\\textheight')

D.frame(T('Interpretation of the Paths', 'Interpretarea traiectoriilor'), items(
    (T('Ridge (left): all coefficients shrink smoothly and together; correlated features share the weight', 'Ridge (stînga): toți coeficienții se contractă lin și împreună; variabilele corelate își împart ponderea'),
     [T('the negative return keeps a negative sign until the end: falling markets raise future volatility (the leverage effect, Chapter 9)', 'randamentul negativ își păstrează semnul negativ pînă la capăt: piețele în scădere cresc volatilitatea viitoare (efectul de levier, Capitolul 9)')]),
    (T('Lasso (right): the features enter one by one as $\\lambda$ falls', 'Lasso (dreapta): variabilele intră pe rînd pe măsură ce $\\lambda$ scade'),
     [T('the weekly variance enters first: the most informative single feature', 'varianța săptămînală intră prima: cea mai informativă variabilă luată singură'),
      T('at the chosen $\\lambda$ the lasso keeps the HAR structure (day, week, month) plus the negative return, $@{sh.cv.r_neg}$', 'la $\\lambda$ ales, lasso păstrează structura HAR (zi, săptămînă, lună) plus randamentul negativ, $@{sh.cv.r_neg}$')]),
    T('The lasso selects; it does not test: a zero coefficient is not a p-value', 'Lasso selectează, nu testează: un coeficient zero nu este un p-value')))

D.recap(('Ridge and Lasso', 'ridge și lasso'), [
    T('Penalties trade a little bias for less variance: $\\lambda\\sum\\beta_j^2$ (ridge), $\\lambda\\sum|\\beta_j|$ (lasso)', 'Penalizările acceptă o deplasare mică în schimbul unei varianțe mai mici: $\\lambda\\sum\\beta_j^2$ (ridge), $\\lambda\\sum|\\beta_j|$ (lasso)'),
    T('Ridge shrinks, lasso shrinks and selects; standardise first, choose $\\lambda$ by walk-forward CV', 'Ridge contractă, lasso contractă și selectează; standardizăm întîi și alegem $\\lambda$ prin CV walk-forward'),
    T('For volatility, the penalised model keeps the HAR structure and the leverage effect', 'Pentru volatilitate, modelul penalizat păstrează structura HAR și efectul de levier')])

# =============================================================================
# 5. ARBORI
# =============================================================================
D.section('Trees, Random Forest and Gradient Boosting', 'Arbori de decizie, random forest și gradient boosting')

D.frame(T('Regression Trees', 'Arborii de regresie'), items(
    (T('A \\textbf{regression tree} splits the feature space into rectangles and predicts the mean of $y$ in each \\textbf{leaf}', 'Un \\textbf{arbore de regresie} împarte spațiul variabilelor în dreptunghiuri și prezice media lui $y$ în fiecare \\textbf{frunză}'),
     [T('split rule: a feature $j$ and a threshold $s$: left $x_j \\le s$, right $x_j > s$', 'regula de împărțire: o variabilă $j$ și un prag $s$: stînga $x_j \\le s$, dreapta $x_j > s$')]),
    (T('\\textbf{Greedy fit}: at each node choose $(j, s)$ that minimises $\\mathrm{SSE} = \\sum_{\\text{left}}(y - \\bar y_L)^2 + \\sum_{\\text{right}}(y - \\bar y_R)^2$',
       '\\textbf{Estimarea greedy}: în fiecare nod alegem $(j, s)$ care minimizează $\\mathrm{SSE} = \\sum_{\\text{stînga}}(y - \\bar y_L)^2 + \\sum_{\\text{dreapta}}(y - \\bar y_R)^2$'),
     [T('SSE: sum of squared errors; $\\bar y_L$, $\\bar y_R$: the means of $y$ in the left and right groups (the two forecasts)', 'SSE: suma pătratelor erorilor; $\\bar y_L$, $\\bar y_R$: mediile lui $y$ în grupul din stînga și în cel din dreapta (cele două prognoze)'),
      T('repeat in each half until a stopping rule: maximum depth, minimum number of observations per leaf', 'repetăm în fiecare jumătate pînă la o regulă de oprire: adîncimea maximă, numărul minim de observații dintr-o frunză')]),
    (T('Strengths and weaknesses', 'Puncte tari și puncte slabe'),
     [T('nonlinear effects and interactions without specifying them; insensitive to monotone transformations of $x$', 'efecte neliniare și interacțiuni fără a le specifica; insensibil la transformările monotone ale lui $x$'),
      T('high variance: a small change in the data can change the whole tree (Section 2)', 'varianță mare: o mică schimbare a datelor poate schimba tot arborele (secțiunea 2)')])))

D.frame(T('Worked Example: One Split, Step by Step', 'Exemplu rezolvat: o împărțire, pas cu pas'), cols(table(
    'rrrr', T('$s$ & $\\bar y_L$ & $\\bar y_R$ & SSE', '$s$ & $\\bar y_L$ & $\\bar y_R$ & SSE'),
    [f'$⁅{s:.2f}⁆$ & $⁅{l:.2f}⁆$ & $⁅{r:.2f}⁆$ & $⁅{e:.3f}⁆$' for s, l, r, e in split_rows], size='footnotesize'), items(
    (T('Six weeks: $x$ = this week\'s volatility (\\%), $y$ = next week\'s', 'Șase săptămîni: $x$ = volatilitatea din această săptămînă (\\%), $y$ = cea din săptămîna următoare'),
     [T('$x$: 0.6, 0.8, 1.0, 1.5, 2.0, 2.4; $y$: 0.7, 0.9, 0.8, 1.6, 2.1, 2.2', '$x$: 0,6; 0,8; 1,0; 1,5; 2,0; 2,4; $y$: 0,7; 0,9; 0,8; 1,6; 2,1; 2,2'),
      T('no split: $\\bar y = @{ts.mean}$, SSE $= @{ts.sse0}$', 'fără împărțire: $\\bar y = @{ts.mean}$, SSE $= @{ts.sse0}$')]),
    (T('Candidates: the midpoints between consecutive $x$ values', 'Candidați: mijloacele intervalelor dintre valorile consecutive ale lui $x$'),
     [T('best split $s = @{ts.best}$: leaves $@{ts.bl}$ and $@{ts.br}$, SSE $= @{ts.bsse}$', 'cea mai bună împărțire $s = @{ts.best}$: frunzele $@{ts.bl}$ și $@{ts.br}$, SSE $= @{ts.bsse}$'),
      T('one split explains $1 - @{ts.bsse}/@{ts.sse0} = @{ts.r2}$ of the variation', 'o singură împărțire explică $1 - @{ts.bsse}/@{ts.sse0} = @{ts.r2}$ din variație')])),
    wl='0.42', wr='0.56'), 'footnotesize')

chart(T('A Depth-2 Tree for Next Week\'s Volatility', 'Un arbore de adîncime 2 pentru volatilitatea săptămînii viitoare'), 'sfm_ch13_tree', 'SFM_ch13_shrinkage_trees', [
    T('S\\&P 500, 2008--2012, target and features in log variance (\\%$^2$ per day); \\emph{value}: the mean of the target in the node',
      'S\\&P 500, 2008--2012, ținta și variabilele în logaritmul varianței (\\%$^2$ pe zi); \\emph{value}: media țintei în nod'),
    T('Interpretation: the first split, log RV week $\\le @{tr.thr}$, is an annual volatility of about @{tr.vol}\\%; all splits use the weekly variance, the feature the lasso picked first',
      'Interpretare: prima împărțire, log RV săptămînal $\\le @{tr.thr}$, corespunde unei volatilități anuale de circa @{tr.vol}\\%; toate împărțirile folosesc varianța săptămînală, variabila aleasă prima de lasso')],
    h='0.50\\textheight')

D.frame(T('Random Forest', 'Random forest'), items(
    (T('\\refBreiman: average $B$ deep trees, each grown on a bootstrap sample (Chapter 2) of the training data (\\textbf{bagging})', '\\refBreiman: media a $B$ arbori adînci, fiecare crescut pe un eșantion bootstrap (Capitolul 2) din datele de antrenare (\\textbf{bagging})'),
     [T('at each split only a random subset of the features is tried: the trees become less correlated', 'la fiecare împărțire se încearcă doar o submulțime aleatoare de variabile: arborii devin mai puțin corelați')]),
    (T('Why averaging works: $B$ trees, each with variance $\\sigma^2$ and pairwise correlation $\\rho$', 'Efectul mediei: $B$ arbori, fiecare cu varianța $\\sigma^2$ și corelația $\\rho$ între oricare doi'),
     [T('$\\mathrm{Var}(\\text{average}) = \\rho\\sigma^2 + (1 - \\rho)\\sigma^2/B$: the part $\\rho\\sigma^2$, common to all trees, does not fall with $B$', '$\\mathrm{Var}(\\text{media}) = \\rho\\sigma^2 + (1 - \\rho)\\sigma^2/B$: partea $\\rho\\sigma^2$, comună tuturor arborilor, nu scade cu $B$'),
      T('with $\\rho = 0.3$, $\\sigma^2 = 1$: $B = 1$: $@{rf.v1}$; $B = 10$: $@{rf.v10}$; $B = 300$: $@{rf.v300}$', 'cu $\\rho = 0{,}3$, $\\sigma^2 = 1$: $B = 1$: $@{rf.v1}$; $B = 10$: $@{rf.v10}$; $B = 300$: $@{rf.v300}$'),
      T('more trees never overfit; less correlated trees (fewer features per split) reduce the floor $\\rho\\sigma^2$', 'mai mulți arbori nu produc overfitting; arborii mai puțin corelați (mai puține variabile la o împărțire) coboară pragul $\\rho\\sigma^2$')]),
    T('Hyperparameters used here: 300 trees, at least 20 observations per leaf, half of the features tried at each split', 'Hiperparametrii folosiți aici: 300 de arbori, cel puțin 20 de observații într-o frunză, jumătate din variabile încercate la fiecare împărțire')))

D.frame(T('Gradient Boosting', 'Gradient boosting'), items(
    (T('\\refFriedman: build the model step by step, each new small tree fitting what the previous ones missed', '\\refFriedman: construim modelul pas cu pas, fiecare arbore mic nou învățînd ce au ratat cei anteriori'),
     [T('start with $F_0 = \\bar y$; at step $m$, fit a shallow tree $h_m$ to the residuals $y_t - F_{m-1}(x_t)$', 'pornim de la $F_0 = \\bar y$; la pasul $m$ estimăm un arbore mic $h_m$ pe reziduurile $y_t - F_{m-1}(x_t)$'),
      T('update $F_m = F_{m-1} + \\nu\\, h_m$, with the \\textbf{learning rate} $0 < \\nu \\le 1$', 'actualizăm $F_m = F_{m-1} + \\nu\\, h_m$, cu \\textbf{rata de învățare} $0 < \\nu \\le 1$'),
      T('for squared loss the residuals are minus the gradient of the loss: hence the name', 'pentru pierderea pătratică, reziduurile sînt minus gradientul pierderii: de aici numele')]),
    (T('Random forest against boosting', 'Random forest comparat cu boosting'),
     [T('forest: deep trees in parallel, averaging reduces variance', 'random forest: arbori adînci în paralel; media reduce varianța'),
      T('boosting: shallow trees in sequence, each step reduces bias; too many steps overfit', 'boosting: arbori mici în secvență; fiecare pas reduce deplasarea; prea mulți pași produc overfitting')]),
    T('Any differentiable loss works: quantile loss gives quantile boosting (Section 9)', 'Funcționează orice pierdere derivabilă: pierderea cuantilică dă quantile boosting (secțiunea 9)')))

chart(T('One Tree, a Forest and Boosting on Volatility Data', 'Un arbore, un random forest și boosting pe date de volatilitate'), 'sfm_ch13_ensembles', 'SFM_ch13_shrinkage_trees', [
    T('S\\&P 500, training 2008--2018 (@{en.ntr} days), validation 2019--2022 (@{en.nva} days); dashed right: training MSE; dotted: HAR regression ($@{en.har}$)',
      'S\\&P 500, antrenare 2008--2018 (@{en.ntr} de zile), validare 2019--2022 (@{en.nva} de zile); linii întrerupte în dreapta: MSE de antrenare; punctat: regresia HAR ($@{en.har}$)'),
    T('Single tree: best $@{en.tree_best}$ at depth @{en.td}, $@{en.tree14}$ at depth 14; forest: $@{en.rf_best}$ (depth @{en.rd}) and still $@{en.rf14}$ at depth 14',
      'Un singur arbore: minimul $@{en.tree_best}$ la adîncimea @{en.td}, $@{en.tree14}$ la adîncimea 14; random forest: $@{en.rf_best}$ (adîncimea @{en.rd}) și tot $@{en.rf14}$ la adîncimea 14')],
    h='0.48\\textheight')

D.frame(T('Interpretation: Averaging and Early Stopping', 'Interpretare: media și oprirea timpurie'), items(
    (T('The single tree overfits quickly; the forest is almost insensitive to depth', 'Arborele singur intră repede în overfitting; random forest este aproape insensibil la adîncime'),
     [T('averaging removes most of the variance of deep trees; the forest beats HAR on this validation period', 'media elimină cea mai mare parte din varianța arborilor adînci; random forest depășește HAR în această perioadă de validare')]),
    (T('Boosting: the validation error reaches its minimum, then rises while the training error keeps falling', 'Boosting: eroarea de validare atinge un minim, apoi crește, în timp ce eroarea de antrenare continuă să scadă'),
     [T('learning rate 0.3: minimum $@{en.gb3}$ after @{en.gb3.n} trees, $@{en.gb3.last}$ after 500', 'rata de învățare 0,3: minimul $@{en.gb3}$ după @{en.gb3.n} @{en.gb3.de}arbori, $@{en.gb3.last}$ după 500 de arbori'),
      T('learning rate 0.03: minimum $@{en.gb03}$ after @{en.gb03.n} trees, $@{en.gb03.last}$ after 500: smaller steps are safer', 'rata de învățare 0,03: minimul $@{en.gb03}$ după @{en.gb03.n} @{en.gb03.de}arbori, $@{en.gb03.last}$ după 500 de arbori: pașii mici sînt mai siguri')]),
    T('The number of trees of boosting is a hyperparameter; the number of trees of a forest is not', 'Numărul de arbori din boosting este un hiperparametru; numărul de arbori dintr-un random forest nu este')))

D.recap(('Trees and Ensembles', 'arbori și ansambluri'), [
    T('A tree splits greedily to minimise the SSE; alone it has high variance', 'Un arbore împarte greedy pentru a minimiza SSE; singur are varianță mare'),
    T('Random forest averages decorrelated deep trees; boosting adds shallow trees fitted to residuals', 'Random forest face media unor arbori adînci decorelați; boosting adaugă arbori mici estimați pe reziduuri'),
    T('Forest: robust to its hyperparameters; boosting: needs a small learning rate and a validated number of trees', 'Random forest: robust la hiperparametri; boosting: are nevoie de o rată de învățare mică și de un număr de arbori validat')])

# =============================================================================
# 6. REȚELE NEURONALE
# =============================================================================
D.section('Neural Networks', 'Rețele neuronale')

D.frame(T('From the Perceptron to the Multilayer Perceptron (1/2)', 'De la perceptron la perceptronul multistrat (1/2)'), items(
    (T('A \\textbf{neuron}: $h = g(b + w^\\top x)$, a weighted sum of the inputs passed through an \\textbf{activation function} $g$', 'Un \\textbf{neuron}: $h = g(b + w^\\top x)$, o sumă ponderată a intrărilor trecută printr-o \\textbf{funcție de activare} $g$'),
     [T('$x$: the inputs; $w$: their weights; $b$: a constant (the bias term of the neuron); $h$: the output of the neuron', '$x$: intrările; $w$: ponderile lor; $b$: o constantă (termenul liber al neuronului); $h$: ieșirea neuronului'),
      T('$g$: a fixed nonlinear function; without it, a network of neurons would be just a linear regression', '$g$: o funcție neliniară fixată; fără ea, o rețea de neuroni ar fi doar o regresie liniară')]),
    (T('Activation functions', 'Funcțiile de activare'),
     [T('the perceptron uses a step function (0 or 1)', 'perceptronul folosește o funcție treaptă (0 sau 1)'),
      T('modern networks use smooth or piecewise linear $g$: logistic $1/(1 + e^{-z})$, tanh, ReLU $\\max(0, z)$', 'rețelele moderne folosesc funcții $g$ netede sau liniare pe porțiuni: logistică $1/(1 + e^{-z})$, tanh, ReLU $\\max(0, z)$')])))

D.frame(T('From the Perceptron to the Multilayer Perceptron (2/2)', 'De la perceptron la perceptronul multistrat (2/2)'), items(
    (T('\\textbf{MLP} (multilayer perceptron) with one hidden layer of $K$ neurons', '\\textbf{MLP} (multilayer perceptron, perceptron multistrat) cu un strat ascuns de $K$ neuroni') + ' \\[ \\hat y = c + \\sum_{k=1}^{K} v_k\\, g(b_k + w_k^\\top x) \\]',
     [T('$w_k$, $b_k$: the weights and the constant of hidden neuron $k$; $v_k$: the weight of neuron $k$ in the output; $c$: the output constant',
        '$w_k$, $b_k$: ponderile și constanta neuronului ascuns $k$; $v_k$: ponderea neuronului $k$ în ieșire; $c$: constanta ieșirii'),
      T('reading: a linear model in $K$ nonlinear features $g(b_k + w_k^\\top x)$, which are themselves learned', 'interpretarea: un model liniar în $K$ variabile neliniare $g(b_k + w_k^\\top x)$, învățate și ele din date')]),
    (T('Number of parameters with 11 inputs and 16 neurons', 'Numărul de parametri pentru 11 intrări și 16 neuroni'),
     [T('hidden layer: $11 \\times 16$ weights $+ 16$ constants $= @{mlp.p1}$; output: $16 + 1 = @{mlp.p2}$; total @{mlp.p}', 'stratul ascuns: $11 \\times 16$ ponderi $+ 16$ constante $= @{mlp.p1}$; ieșirea: $16 + 1 = @{mlp.p2}$; în total @{mlp.p}')]),
    T('\\textbf{Universal approximation} \\refHSW: with enough neurons, one hidden layer approximates any continuous function on a compact set',
      '\\textbf{Aproximarea universală} \\refHSW: cu suficienți neuroni, un singur strat ascuns aproximează orice funcție continuă pe o mulțime compactă')))

chart(T('A Small Network and Its Activation Functions', 'O rețea mică și funcțiile ei de activare'), 'sfm_ch13_mlp', 'SFM_ch13_neural_networks', [
    T('Left: three HAR inputs, five hidden neurons, one forecast; each line is a weight', 'Stînga: trei intrări HAR, cinci neuroni ascunși, o prognoză; fiecare linie este o pondere'),
    T('Right: logistic and tanh saturate for large $|z|$; ReLU $= \\max(0, z)$ is linear for $z > 0$ and trains faster', 'Dreapta: funcțiile logistică și tanh se saturează pentru $|z|$ mare; ReLU $= \\max(0, z)$ este liniară pentru $z > 0$ și se antrenează mai repede')],
    h='0.50\\textheight')

D.frame(T('Training a Network', 'Antrenarea unei rețele'), items(
    (T('Minimise the loss by \\textbf{gradient descent}: $\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta L$', 'Minimizăm pierderea prin \\textbf{coborîre pe gradient}: $\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta L$'),
     [T('$\\theta$: all weights and constants; $\\nabla_\\theta L$: the gradient, the vector of partial derivatives of the loss; $\\eta > 0$: the learning rate (the step size)',
        '$\\theta$: toate ponderile și constantele; $\\nabla_\\theta L$: gradientul, vectorul derivatelor parțiale ale pierderii; $\\eta > 0$: rata de învățare (mărimea pasului)'),
      T('each step moves $\\theta$ in the direction in which the loss falls fastest', 'fiecare pas deplasează $\\theta$ în direcția în care pierderea scade cel mai repede'),
      T('\\textbf{back-propagation} computes $\\nabla_\\theta L$ layer by layer with the chain rule \\refRHW', '\\textbf{retropropagarea} calculează $\\nabla_\\theta L$ strat cu strat, cu regula de derivare a funcțiilor compuse \\refRHW'),
      T('one \\textbf{epoch} = one pass over the training data; Adam is a gradient method with adaptive step sizes', 'o \\textbf{epocă} = o trecere prin datele de antrenare; Adam este o metodă de gradient cu pași adaptivi')]),
    (T('The loss is not convex: different starting weights give different networks', 'Pierderea nu este convexă: ponderi inițiale diferite dau rețele diferite'),
     [T('we fix the random seeds and average five networks (an \\textbf{ensemble})', 'fixăm semințele aleatoare și facem media a cinci rețele (un \\textbf{ansamblu})')]),
    (T('Against overfitting', 'Împotriva overfitting-ului'),
     [T('standardise inputs and target with training statistics; an $L_2$ penalty on the weights (as ridge); a fixed number of epochs',
        'standardizăm intrările și ținta cu statisticile de antrenare; o penalizare $L_2$ a ponderilor (ca la ridge); un număr fix de epoci'),
      T('small data, small network: one hidden layer of 16 neurons is enough here', 'date puține, rețea mică: aici este suficient un strat ascuns cu 16 neuroni')])))

D.frame(T('From Deep Blue to Deep Learning', 'De la Deep Blue la deep learning'), cols(
    ph('blue', T('IBM Deep Blue, Computer History Museum', 'IBM Deep Blue, Computer History Museum'), h='0.36\\textheight'),
    ph('hinton', T('Geoffrey Hinton, Nobel Prize in Physics 2024', 'Geoffrey Hinton, Premiul Nobel pentru fizică 2024'), h='0.36\\textheight'),
    wl='0.48', wr='0.48') + items(
    T('1997: Deep Blue beat the world chess champion mainly by search and hand-crafted evaluation rules, not by learning from data', '1997: Deep Blue l-a învins pe campionul mondial la șah mai ales prin căutare și reguli de evaluare scrise de oameni, nu prin învățare din date'),
    T('Deep learning (many layers, huge data) now dominates images and text; on financial prices the data are few and noisy, so small networks are the norm',
      'Deep learning (multe straturi, volume uriașe de date) domină azi imaginile și textul; la prețurile financiare datele sînt puține și zgomotoase, deci rețelele mici sînt regula')), 'footnotesize')

D.frame(T('Neural-Network ARCH: the Textbook Model (1/2)', 'ARCH cu rețea neuronală: modelul din manual (1/2)'), items(
    (T('\\refFHH, Ch.~19: $r_t = f(x_{t-1}) + s(x_{t-1})\\,\\varepsilon_t$, with $f$ and $s^2$ estimated by neural networks', '\\refFHH, cap.~19: $r_t = f(x_{t-1}) + s(x_{t-1})\\,\\varepsilon_t$, cu $f$ și $s^2$ estimate prin rețele neuronale'),
     [T('$r_t$: the return of day $t$; $x_{t-1}$: the inputs known at the close of day $t - 1$', '$r_t$: randamentul zilei $t$; $x_{t-1}$: intrările cunoscute la închiderea zilei $t - 1$'),
      T('$f(x_{t-1})$: the conditional mean; $s^2(x_{t-1})$: the conditional variance; $\\varepsilon_t$: a shock with mean 0 and variance 1',
        '$f(x_{t-1})$: media condiționată; $s^2(x_{t-1})$: varianța condiționată; $\\varepsilon_t$: un șoc cu media 0 și varianța 1')]),
    (T('ARCH (\\refEngle) is the special case $s^2(x) = \\omega + \\sum_i \\alpha_i r_{t-i}^2$', 'ARCH (\\refEngle) este cazul particular $s^2(x) = \\omega + \\sum_i \\alpha_i r_{t-i}^2$'),
     [T('$\\omega > 0$, $\\alpha_i \\ge 0$: the variance is a linear function of the past squared returns', '$\\omega > 0$, $\\alpha_i \\ge 0$: varianța este o funcție liniară de pătratele randamentelor trecute'),
      T('the network imposes no shape on $s^2$', 'rețeaua nu impune nicio formă funcției $s^2$')])))

D.frame(T('Neural-Network ARCH: the Textbook Model (2/2)', 'ARCH cu rețea neuronală: modelul din manual (2/2)'), items(
    (T('The Quantlet \\refSFEa: GBP/USD, inputs $x$ = the last 3 returns, the German 10-year yield, the gold return', 'Quantlet-ul \\refSFEa: GBP/USD, intrările $x$ = ultimele 3 randamente, randamentul obligațiunilor germane pe 10 ani, randamentul aurului'),
     [T('step 1: a network $\\hat f$ for the mean; residuals $e_t = r_t - \\hat f(x_{t-1})$', 'pasul 1: o rețea $\\hat f$ pentru medie; reziduurile $e_t = r_t - \\hat f(x_{t-1})$'),
      T('step 2: a network for $e_t^2$ gives the conditional variance $\\hat s^2(x_{t-1})$', 'pasul 2: o rețea pentru $e_t^2$ dă varianța condiționată $\\hat s^2(x_{t-1})$')]),
    (T('\\textbf{RBF} (radial basis function) network with 3 units, as in the Quantlet', 'Rețea \\textbf{RBF} (radial basis function, cu funcții de bază radiale) cu 3 unități, ca în Quantlet'),
     [T('unit $j$: $\\exp(-\\|x - c_j\\|^2/s_j^2)$; $c_j$: its centre (found by clustering); $s_j$: its width; $\\|x - c_j\\|$: the distance from $x$ to $c_j$',
        'unitatea $j$: $\\exp(-\\|x - c_j\\|^2/s_j^2)$; $c_j$: centrul ei (obținut prin clustering); $s_j$: lățimea ei; $\\|x - c_j\\|$: distanța de la $x$ la $c_j$'),
      T('the unit is close to 1 when $x$ is near $c_j$ and close to 0 far from it; output: a linear combination of the 3 units', 'unitatea este aproape de 1 cînd $x$ este lîngă $c_j$ și aproape de 0 departe de el; ieșirea: o combinație liniară a celor 3 unități'),
      T('nothing forces $\\hat s^2 > 0$: a variance network needs a floor, which GARCH guarantees by its constraints', 'nimic nu impune $\\hat s^2 > 0$: o rețea pentru varianță are nevoie de un prag minim, pe care GARCH îl garantează prin restricțiile sale')])))

chart(T('Neural-Network ARCH for GBP/USD', 'ARCH cu rețea neuronală pentru GBP/USD'), 'sfm_ch13_nnarch', 'SFM_ch13_neural_networks', [
    T('Daily data from @{na.first} (@{na.n} days); annualised volatility of the RBF network and of GARCH(1,1) ($\\hat\\alpha = @{na.alpha}$, $\\hat\\beta = @{na.beta}$)',
      'Date zilnice din @{na.first} (@{na.n} de zile); volatilitatea anualizată a rețelei RBF și a modelului GARCH(1,1) ($\\hat\\alpha = @{na.alpha}$, $\\hat\\beta = @{na.beta}$)'),
    T('Average volatility: network @{na.mean_rbf}\\%, GARCH @{na.mean_garch}\\%; correlation of the two paths $@{na.corr}$ (a small MLP: $@{na.corrm}$)',
      'Volatilitatea medie: rețeaua @{na.mean_rbf}\\%, GARCH @{na.mean_garch}\\%; corelația celor două traiectorii $@{na.corr}$ (un MLP mic: $@{na.corrm}$)')],
    h='0.50\\textheight')

D.frame(T('Interpretation: Flexible Shape, Short Memory', 'Interpretare: formă flexibilă, memorie scurtă'), items(
    (T('After the Brexit vote, GARCH peaks at @{na.max_garch}\\% (@{na.dmax}); the network only at @{na.max_rbf}\\%', 'După votul pentru Brexit, GARCH atinge maximul de @{na.max_garch}\\% (@{na.dmax}); rețeaua doar @{na.max_rbf}\\%'),
     [T('the network sees only the last 3 returns: one large return raises its variance for 3 days, then the effect is gone', 'rețeaua vede doar ultimele 3 randamente: un randament mare îi crește varianța timp de 3 zile, apoi efectul dispare'),
      T('GARCH(1,1) (Chapter 9): $\\sigma_t^2 = \\omega + \\alpha\\,r_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$, with $\\sigma_{t-1}^2$ yesterday\'s conditional variance', 'GARCH(1,1) (Capitolul 9): $\\sigma_t^2 = \\omega + \\alpha\\,r_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$, cu $\\sigma_{t-1}^2$ varianța condiționată de ieri'),
      T('the term $\\beta\\,\\sigma_{t-1}^2$ carries the whole past: volatility clustering needs memory, not only flexibility', 'termenul $\\beta\\,\\sigma_{t-1}^2$ transmite tot trecutul: volatility clustering are nevoie de memorie, nu doar de flexibilitate')]),
    (T('The mean network explains $@{na.r2}\\%$ of the returns in sample: exchange rates are close to a random walk', 'Rețeaua pentru medie explică $@{na.r2}\\%$ din randamente în eșantion: cursurile de schimb sînt aproape un mers aleator'),
     [T('the useful part of NN-ARCH is the variance, as for GARCH', 'partea utilă a modelului NN-ARCH este varianța, ca la GARCH')]),
    T('Lesson: good features (HAR averages, Section 7) matter more than the type of network', 'Lecția: variabilele bune (mediile HAR, secțiunea 7) contează mai mult decît tipul de rețea')))

D.frame(T('Forecasting an Exchange Rate Level: SFEnnjpyusd', 'Prognoza nivelului unui curs de schimb: SFEnnjpyusd'), items(
    (T('The Quantlet \\refSFEb{} forecasts the yen--dollar rate from its last 3 values with an RBF network of 3 units', 'Quantlet-ul \\refSFEb{} prognozează cursul yen--dolar din ultimele 3 valori, cu o rețea RBF cu 3 unități'),
     [T('first 80\\% of the days for training, last 20\\% for testing; here USD/JPY from 2000, split on @{jp.split}', 'primele 80\\% din zile pentru antrenare, ultimele 20\\% pentru test; aici USD/JPY din 2000, cu separarea la @{jp.split}')]),
    (T('Two traps of level forecasting', 'Două capcane ale prognozei nivelului'),
     [T('a level series looks predictable because $P_t \\approx P_{t-1}$: the right benchmark is the random walk $\\hat P_t = P_{t-1}$ (Chapter 7)',
        'o serie de niveluri pare previzibilă deoarece $P_t \\approx P_{t-1}$: reperul corect este mersul aleator $\\hat P_t = P_{t-1}$ (Capitolul 7)'),
      T('a network with bounded units cannot extrapolate: the test levels go up to @{jp.max_te}, the training maximum was @{jp.max_tr}',
        'o rețea cu unități mărginite nu poate extrapola: nivelurile de test urcă pînă la @{jp.max_te}, iar maximul din antrenare a fost @{jp.max_tr}')]),
    T('The original code rescales the test forecasts with the range of the test series, which is unknown in real time; here the training range is used',
      'Codul original rescalează prognozele de test cu intervalul seriei de test, necunoscut în timp real; aici se folosește intervalul din antrenare')))

chart(T('The RBF Network against the Random Walk', 'Rețeaua RBF comparată cu mersul aleator'), 'sfm_ch13_nnjpy', 'SFM_ch13_neural_networks', [
    T('RMSE (root mean squared error) in yen: training: network @{jp.rmse_tr}, random walk @{jp.rmse_rw_tr}; test: network @{jp.rmse_te}, random walk @{jp.rmse_rw_te}',
      'RMSE (root mean squared error, rădăcina erorii pătratice medii) în yeni: antrenare: rețeaua @{jp.rmse_tr}, mersul aleator @{jp.rmse_rw_tr}; test: rețeaua @{jp.rmse_te}, mersul aleator @{jp.rmse_rw_te}'),
    T('Interpretation: out of sample the network is @{jp.ratio} times worse than ``tomorrow = today\'\'; on returns the same network has an out-of-sample $R^2$ of $@{jp.r2ret}\\%$',
      'Interpretare: în afara eșantionului rețeaua este de @{jp.ratio} @{jp.de}ori mai slabă decît „mîine = azi”; pe randamente, aceeași rețea are un $R^2$ în afara eșantionului de $@{jp.r2ret}\\%$')],
    h='0.50\\textheight')

D.recap(('Neural Networks', 'rețele neuronale'), [
    T('An MLP is a linear model in learned nonlinear features; trained by gradient descent and back-propagation', 'Un MLP este un model liniar în variabile neliniare învățate; se antrenează prin coborîre pe gradient și retropropagare'),
    T('NN-ARCH: a flexible variance function, but without the memory of GARCH and without positivity', 'NN-ARCH: o funcție a varianței flexibilă, dar fără memoria GARCH și fără garanția pozitivității'),
    T('Forecast returns, not levels, and always compare with the random walk', 'Prognozați randamente, nu niveluri, și comparați întotdeauna cu mersul aleator')])

# =============================================================================
# 7. VOLATILITATEA REALIZATĂ
# =============================================================================
D.section('Application: Forecasting Realised Volatility', 'Aplicație: prognoza volatilității realizate')

D.frame(T('Target and Features (1/2)', 'Ținta și variabilele explicative (1/2)'), items(
    (T('Daily variance proxy of Chapter 8 (no intraday data): an estimate of the variance of day $t$ from its open, high, low and close',
       'Variabila proxy zilnică a varianței din Capitolul 8 (fără date intrazilnice): o estimație a varianței zilei $t$ din prețurile de deschidere, maxim, minim și închidere') + ' \\[ v_t = o_t^2 + \\tfrac12(u_t - d_t)^2 - (2\\ln 2 - 1)c_t^2 \\]',
     [T('$o_t$: the overnight log return (from the previous close to the open)', '$o_t$: randamentul logaritmic peste noapte (de la închiderea precedentă la deschidere)'),
      T('$u_t, d_t, c_t$: log high, low and close relative to the open (the Garman--Klass term, \\refGK)', '$u_t, d_t, c_t$: logaritmii maximului, minimului și închiderii față de deschidere (termenul Garman--Klass, \\refGK)')]),
    (T('\\textbf{Target}: $y_t = \\ln\\big(\\tfrac15\\sum_{j=1}^{5} v_{t+j}\\big)$, the log realised variance of the next week', '\\textbf{Ținta}: $y_t = \\ln\\big(\\tfrac15\\sum_{j=1}^{5} v_{t+j}\\big)$, logaritmul varianței realizate din săptămîna următoare'),
     [T('$j = 1, \\dots, 5$: the next five trading days; the target is known only at the close of day $t + 5$', '$j = 1, \\dots, 5$: următoarele cinci zile de tranzacționare; ținta este cunoscută abia la închiderea zilei $t + 5$'),
      T('logs make the target closer to Normal and the forecasts positive after $\\exp$', 'logaritmul apropie ținta de distribuția Normală și face prognozele pozitive după $\\exp$')])))

D.frame(T('Target and Features (2/2)', 'Ținta și variabilele explicative (2/2)'), items(
    (T('\\textbf{Features} at the close of day $t$', '\\textbf{Variabilele explicative} la închiderea zilei $t$'),
     [T('HAR: $\\ln v_t$, $\\ln$ of the mean over 5 days (week) and over 22 days (month)', 'HAR: $\\ln v_t$, logaritmul mediei pe 5 zile (săptămîna) și pe 22 de zile (luna)'),
      T('extended (11): plus 4 lags of $\\ln v_t$ ($\\ln v_{t-1}, \\dots, \\ln v_{t-4}$), the 66-day mean, $r_t$, $\\min(r_t, 0)$, $|r_t|$', 'extinse (11): în plus 4 laguri ale lui $\\ln v_t$ ($\\ln v_{t-1}, \\dots, \\ln v_{t-4}$), media pe 66 de zile, $r_t$, $\\min(r_t, 0)$, $|r_t|$')]),
    (T('Notation', 'Notațiile'),
     [T('$r_t$: the log return of day $t$; $|r_t|$: its absolute value', '$r_t$: randamentul logaritmic al zilei $t$; $|r_t|$: valoarea lui absolută'),
      T('$\\min(r_t, 0)$: the return if negative, 0 otherwise; it lets falls and rises act differently (the leverage effect)', '$\\min(r_t, 0)$: randamentul, dacă este negativ, și 0 în rest; permite scăderilor și creșterilor să aibă efecte diferite (efectul de levier)')])))

D.frame(T('The HAR Benchmark (1/2)', 'Reperul HAR (1/2)'), items(
    (T('\\textbf{HAR} (heterogeneous autoregressive) model \\refCorsi', 'Modelul \\textbf{HAR} (heterogeneous autoregressive, autoregresiv heterogen) \\refCorsi') + ' \\[ y_t = \\beta_0 + \\beta_d\\ln v_t + \\beta_w\\ln v_t^{(w)} + \\beta_m\\ln v_t^{(m)} + u_t \\]',
     [T('$v_t^{(w)}$, $v_t^{(m)}$: the mean of $v$ over the last 5 days (week) and over the last 22 days (month)', '$v_t^{(w)}$, $v_t^{(m)}$: media lui $v$ pe ultimele 5 zile (săptămîna) și pe ultimele 22 de zile (luna)'),
      T('$\\beta_0$: the constant; $\\beta_d, \\beta_w, \\beta_m$: the weights of the three horizons; $u_t$: the error', '$\\beta_0$: constanta; $\\beta_d, \\beta_w, \\beta_m$: ponderile celor trei orizonturi; $u_t$: eroarea')]),
    (T('Interpretation', 'Interpretarea'),
     [T('three horizons of traders (daily, weekly, monthly); estimated by OLS', 'trei orizonturi ale participanților la piață (zilnic, săptămînal, lunar); estimat prin OLS'),
      T('with positive weights on the three horizons, a shock to volatility fades slowly', 'cu ponderi pozitive pe cele trei orizonturi, un șoc al volatilității se atenuează lent'),
      T('a simple linear model that mimics long memory (Chapter 11) and is hard to beat', 'un model liniar simplu care imită memoria lungă (Capitolul 11) și este greu de depășit')])))

D.frame(T('The HAR Benchmark (2/2)', 'Reperul HAR (2/2)'), items(
    (T('The question of this section', 'Întrebarea acestei secțiuni'),
     [T('do lasso, random forest, gradient boosting or an MLP add anything to HAR out of sample?', 'adaugă lasso, random forest, gradient boosting sau un MLP ceva peste HAR în afara eșantionului?'),
      T('a large study with intraday data finds modest gains, mainly from extra features \\refCSV', 'un studiu amplu cu date intrazilnice găsește cîștiguri modeste, mai ales din variabile suplimentare \\refCSV')]),
    (T('Design: walk-forward of Section 3, yearly refits; S\\&P 500 from 2013, Bitcoin from 2018', 'Schema de evaluare: walk-forward din secțiunea 3, reestimare anuală; S\\&P 500 din 2013, Bitcoin din 2018'),
     [T('variance forecast $\\hat h_t = \\exp(\\hat y_t + \\hat s^2/2)$, $\\hat s^2$ the residual variance in training (the log-Normal mean)', 'prognoza varianței $\\hat h_t = \\exp(\\hat y_t + \\hat s^2/2)$, cu $\\hat s^2$ varianța reziduală din antrenare (media log-Normală)'),
      T('$\\hat y_t$: the forecast of the log variance; $\\exp(\\hat y_t)$ alone would underestimate the mean variance, hence the term $\\hat s^2/2$',
        '$\\hat y_t$: prognoza logaritmului varianței; $\\exp(\\hat y_t)$ singur ar subestima varianța medie, de aici termenul $\\hat s^2/2$')])))

D.frame(T('Evaluation (1/3): Out-of-Sample $R^2$', 'Evaluarea (1/3): $R^2$ în afara eșantionului'), items(
    (T('\\textbf{Out-of-sample} $R^2$ against a benchmark forecast $\\tilde y_t$ \\refCT', '$R^2$ \\textbf{în afara eșantionului} față de o prognoză de referință $\\tilde y_t$ \\refCT') + ' \\[ R^2_{OOS} = 1 - \\frac{\\sum_t (y_t - \\hat y_t)^2}{\\sum_t (y_t - \\tilde y_t)^2} \\]',
     [T('$\\hat y_t$: the forecast of the model; sums over the test days only', '$\\hat y_t$: prognoza modelului; sumele se calculează doar pe zilele de test'),
      T('benchmark $\\tilde y_t$: the training mean, or HAR', 'reperul $\\tilde y_t$: media din antrenare sau HAR')]),
    (T('Reading', 'Interpretarea'),
     [T('$R^2_{OOS} \\le 1$; $R^2_{OOS} > 0$: smaller squared error than the benchmark; $R^2_{OOS} < 0$: worse than the benchmark', '$R^2_{OOS} \\le 1$; $R^2_{OOS} > 0$: eroare pătratică mai mică decît a reperului; $R^2_{OOS} < 0$: mai slab decît reperul'),
      T('unlike the in-sample $R^2$, it can be negative', 'spre deosebire de $R^2$ din eșantion, poate fi negativ')]),
    (T('Step by step, S\\&P 500, MLP against HAR', 'Pas cu pas, S\\&P 500, MLP față de HAR'),
     [T('$1 - @{oos.mse.mlp}/@{oos.mse.har} = 1 - @{oos.ratio}$, i.e.\\ $@{rv.sp500.MLP.r2h}\\%$', '$1 - @{oos.mse.mlp}/@{oos.mse.har} = 1 - @{oos.ratio}$, adică $@{rv.sp500.MLP.r2h}\\%$')])))

D.frame(T('Evaluation (2/3): QLIKE', 'Evaluarea (2/3): QLIKE'), items(
    (T('\\textbf{QLIKE} (Chapter 8): a loss for variance forecasts, robust to a noisy proxy \\refPatton', '\\textbf{QLIKE} (Capitolul 8): o funcție de pierdere pentru prognozele varianței, robustă la o variabilă proxy zgomotoasă \\refPatton') + ' \\[ L_t = \\frac{\\mathrm{RV}_t}{\\hat h_t} + \\ln\\hat h_t \\]',
     [T('$\\mathrm{RV}_t$: the realised variance of the next 5 days (the target, in levels); $\\hat h_t$: its forecast', '$\\mathrm{RV}_t$: varianța realizată din următoarele 5 zile (ținta, în niveluri); $\\hat h_t$: prognoza ei'),
      T('for a given $\\mathrm{RV}_t$, the loss is smallest at $\\hat h_t = \\mathrm{RV}_t$; lower average QLIKE is better', 'pentru un $\\mathrm{RV}_t$ dat, pierderea este minimă la $\\hat h_t = \\mathrm{RV}_t$; un QLIKE mediu mai mic este mai bun')]),
    (T('Asymmetry', 'Asimetria'),
     [T('for $\\mathrm{RV} = 1$: $\\hat h = 0.5$ costs $@{ql.u}$, $\\hat h = 2$ costs $@{ql.o}$', 'pentru $\\mathrm{RV} = 1$: $\\hat h = 0{,}5$ costă $@{ql.u}$, $\\hat h = 2$ costă $@{ql.o}$'),
      T('under-prediction is punished more: for risk management, underestimating volatility is the costly error', 'subestimarea este penalizată mai mult: în managementul riscului, subestimarea volatilității este eroarea costisitoare')])))

D.frame(T('Evaluation (3/3): the Diebold--Mariano Test', 'Evaluarea (3/3): testul Diebold--Mariano'), items(
    (T('\\textbf{Diebold--Mariano} test \\refDM: do two models $A$ and $B$ have the same expected loss?', 'Testul \\textbf{Diebold--Mariano} \\refDM: au două modele $A$ și $B$ aceeași pierdere așteptată?') + ' \\[ d_t = L_t^{A} - L_t^{B}, \\qquad \\mathrm{DM} = \\frac{\\bar d}{\\sqrt{\\widehat{\\mathrm{LRV}}/n}} \\sim N(0,1) \\text{ ' + T('under equal accuracy', 'la acuratețe egală') + '} \\]',
     [T('$L_t^{A}$, $L_t^{B}$: the losses of the two models on day $t$; $\\bar d$: the mean of $d_t$; $n$: the number of test days', '$L_t^{A}$, $L_t^{B}$: pierderile celor două modele în ziua $t$; $\\bar d$: media lui $d_t$; $n$: numărul zilelor de test'),
      T('$\\widehat{\\mathrm{LRV}}$: the Newey--West long-run variance of $d_t$ (5 lags), because overlapping targets make $d_t$ autocorrelated',
        '$\\widehat{\\mathrm{LRV}}$: varianța pe termen lung Newey--West a lui $d_t$ (5 laguri), deoarece țintele suprapuse fac $d_t$ autocorelat')]),
    (T('Reading', 'Interpretarea'),
     [T('DM $< 0$: $A$ has the smaller loss; DM $> 0$: $B$ is better', 'DM $< 0$: $A$ are pierderea mai mică; DM $> 0$: $B$ este mai bun'),
      T('$|\\mathrm{DM}| > 1.96$: the difference is significant at 5\\%', '$|\\mathrm{DM}| > 1{,}96$: diferența este semnificativă la 5\\%')])))


def rvrow(k, m):
    key = f'rv.{k}.{m}'
    dm = '--' if m == 'HAR' else f'${{@{{{key}.dm}}}}$ (@{{{key}.dmp}})'
    r2h = '--' if m == 'HAR' else f'${{@{{{key}.r2h}}}}$'
    lab = {'sp500': 'S\\&P 500', 'btc': 'Bitcoin'}[k] if m == 'HAR' else ''
    return f'{lab} & {m} & $@{{{key}.r2m}}$ & {r2h} & $@{{{key}.ql}}$ & {dm}'


RVH = T('& Model & $R^2_{OOS}$ vs.\\ mean & $R^2_{OOS}$ vs.\\ HAR & QLIKE & DM vs.\\ HAR (p)', '& Model & $R^2_{OOS}$ față de medie & $R^2_{OOS}$ față de HAR & QLIKE & DM față de HAR (p)')
D.frame(T('Results: S\\&P 500 and Bitcoin', 'Rezultatele: S\\&P 500 și Bitcoin'), table(
    'llrrrr', RVH, [rvrow('sp500', m) for m in ['HAR', 'Lasso', 'RF', 'GB', 'MLP']] + ['\\midrule'] +
    [rvrow('btc', m) for m in ['HAR', 'Lasso', 'RF', 'GB', 'MLP']], size='scriptsize').replace('\\midrule \\\\', '\\midrule') + items(
    T('Forecasts for @{rv.sp500.n} days (S\\&P 500, from @{rv.sp500.first}) and @{rv.btc.n} days (Bitcoin, from @{rv.btc.first}); $R^2_{OOS}$ in \\%, on log RV; GB: gradient boosting',
      'Prognoze pentru @{rv.sp500.n} zile (S\\&P 500, de la @{rv.sp500.first}) și @{rv.btc.n} zile (Bitcoin, de la @{rv.btc.first}); $R^2_{OOS}$ în \\%, pe log RV; GB: gradient boosting'),
    T('S\\&P 500: lasso and MLP beat HAR significantly ($@{rv.sp500.Lasso.r2h}\\%$, $@{rv.sp500.MLP.r2h}\\%$); random forest and boosting do not',
      'S\\&P 500: lasso și MLP depășesc semnificativ HAR ($@{rv.sp500.Lasso.r2h}\\%$; $@{rv.sp500.MLP.r2h}\\%$); random forest și boosting nu'),
    T('Bitcoin: no model beats HAR; boosting is significantly worse (DM $@{rv.btc.GB.dm}$)', 'Bitcoin: niciun model nu depășește HAR; boosting este semnificativ mai slab (DM $@{rv.btc.GB.dm}$)')) + ql('SFM_ch13_volatility_forecast'), 'footnotesize')

chart(T('Forecasts in Two Storms: 2020 and 2025', 'Prognozele în două furtuni: 2020 și 2025'), 'sfm_ch13_rv_forecasts', 'SFM_ch13_volatility_forecast', [
    T('S\\&P 500: annualised realised volatility of the next 5 days, $\\sqrt{252\\,\\mathrm{RV}_t}$, and the HAR, random forest and MLP forecasts made at the close of day $t$',
      'S\\&P 500: volatilitatea realizată anualizată din următoarele 5 zile, $\\sqrt{252\\,\\mathrm{RV}_t}$, și prognozele HAR, random forest și MLP făcute la închiderea zilei $t$'),
    T('Interpretation: all models react only after the first shock (March 2020, April 2025); the forest is capped by the largest values it saw in training, the MLP can overshoot',
      'Interpretare: toate modelele reacționează abia după primul șoc (martie 2020, aprilie 2025); random forest este limitat de cele mai mari valori văzute în antrenare, iar MLP poate depăși ținta')],
    h='0.50\\textheight')

D.frame(T('Feature Importance: Permutation and Shapley Values', 'Importanța variabilelor: permutare și valori Shapley'), items(
    (T('\\textbf{Permutation importance}: shuffle one feature in the test data and measure how much the test MSE rises', '\\textbf{Importanța prin permutare}: amestecăm o variabilă în datele de test și măsurăm cu cît crește MSE de test'),
     [T('large increase: the model relies on the feature; correlated features share (and hide) their importance', 'creștere mare: modelul se bazează pe variabilă; variabilele corelate își împart (și își ascund) importanța')]),
    (T('\\textbf{Shapley values} (game theory): $\\hat f(x) = \\phi_0 + \\sum_j \\phi_j(x)$, $\\phi_j$ the fair share of feature $j$ in the forecast',
       '\\textbf{Valorile Shapley} (teoria jocurilor): $\\hat f(x) = \\phi_0 + \\sum_j \\phi_j(x)$, $\\phi_j$ este contribuția echitabilă a variabilei $j$ la prognoză'),
     [T('$\\phi_0$: the average forecast; $\\phi_j(x) > 0$: feature $j$ raises this forecast above the average, $\\phi_j(x) < 0$: lowers it',
        '$\\phi_0$: prognoza medie; $\\phi_j(x) > 0$: variabila $j$ ridică această prognoză peste medie, $\\phi_j(x) < 0$: o coboară'),
      T('$\\phi_j$ averages the change of $\\hat f$ when $j$ joins every subset of the other features', '$\\phi_j$ este media modificării lui $\\hat f$ cînd $j$ se adaugă la fiecare submulțime a celorlalte variabile'),
      T('SHAP (Shapley additive explanations) \\refLL: fast versions for trees; for a linear model $\\phi_j = \\beta_j(x_j - \\bar x_j)$', 'SHAP (Shapley additive explanations) \\refLL: versiuni rapide pentru arbori; pentru un model liniar $\\phi_j = \\beta_j(x_j - \\bar x_j)$')]),
    T('Both describe the model, not the economy: importance is not causality', 'Ambele descriu modelul, nu economia: importanța nu înseamnă cauzalitate')))

chart(T('The Features Used by the Forest', 'Variabilele folosite de random forest'), 'sfm_ch13_importance', 'SFM_ch13_volatility_forecast', [
    T('Left: random forest on 11 features, trained to 2019, tested 2020--2026 (@{im.n} days): shuffling the weekly variance raises the MSE by $@{im.w}$, the monthly by $@{im.m}$, the daily by $@{im.d}$',
      'Stînga: random forest pe 11 variabile, antrenat pînă în 2019, testat în 2020--2026 (@{im.n} de zile): amestecarea varianței săptămînale crește MSE cu $@{im.w}$, a celei lunare cu $@{im.m}$, a celei zilnice cu $@{im.d}$'),
    T('Right: exact Shapley values of a forest on the 3 HAR features; mean $|\\phi|$: week $@{im.phi.w}$, month $@{im.phi.m}$, day $@{im.phi.d}$; the effect is nonlinear and grows with the level of volatility',
      'Dreapta: valorile Shapley exacte ale unui random forest pe cele 3 variabile HAR; media $|\\phi|$: săptămîna $@{im.phi.w}$, luna $@{im.phi.m}$, ziua $@{im.phi.d}$; efectul este neliniar și crește cu nivelul volatilității')],
    h='0.48\\textheight')

D.recap(('Forecasting Realised Volatility', 'prognoza volatilității realizate'), [
    T('HAR is a strong benchmark; penalised and neural models add a few per cent on the S\\&P 500, nothing on Bitcoin', 'HAR este un reper puternic; modelele penalizate și rețelele adaugă cîteva procente la S\\&P 500, nimic la Bitcoin'),
    T('Evaluate with out-of-sample $R^2$, QLIKE and Diebold--Mariano, on a walk-forward design', 'Evaluăm prin $R^2$ în afara eșantionului, QLIKE și Diebold--Mariano, într-un design walk-forward'),
    T('The weekly variance carries most of the information; importance measures describe the model, not causes', 'Varianța săptămînală conține cea mai mare parte din informație; măsurile de importanță descriu modelul, nu cauzele')])

# =============================================================================
# 8. SEMNUL RANDAMENTULUI
# =============================================================================
D.section("Application: the Sign of Tomorrow's Return", 'Aplicație: semnul randamentului de mîine')

D.frame(T('A Classification Problem', 'O problemă de clasificare'), items(
    (T('Target: $y_t = \\mathbf{1}\\{r_{t+1} > 0\\}$; features at the close of day $t$: the last 5 returns, sums over 5, 21 and 63 days, volatility over 21 and 63 days',
       'Ținta: $y_t = \\mathbf{1}\\{r_{t+1} > 0\\}$; variabilele la închiderea zilei $t$: ultimele 5 randamente, sumele pe 5, 21 și 63 de zile, volatilitatea pe 21 și 63 de zile'),
     [T('$\\mathbf{1}\\{\\cdot\\}$: the indicator, 1 if the condition holds and 0 otherwise; $y_t = 1$: tomorrow\'s return is positive', '$\\mathbf{1}\\{\\cdot\\}$: indicatorul, egal cu 1 dacă afirmația este adevărată și cu 0 în caz contrar; $y_t = 1$: randamentul de mîine este pozitiv'),
      T('models: logit (Chapter 12), random forest and gradient boosting classifiers; walk-forward, yearly refits', 'modele: logit (Capitolul 12), clasificatori random forest și gradient boosting; walk-forward, reestimare anuală'),
      T('S\\&P 500 and BET from 2008, Bitcoin from 2018', 'S\\&P 500 și BET din 2008, Bitcoin din 2018')]),
    (T('Metrics (defined in Chapter 12)', 'Indicatori (definiți în Capitolul 12)'),
     [T('accuracy: the share of days with the right sign; AUC (area under the ROC curve): 0.5 means no ranking skill', 'acuratețea: ponderea zilelor cu semnul corect; AUC (aria de sub curba ROC): 0,5 înseamnă nicio capacitate de ordonare')]),
    (T('Why expect little: weak-form efficiency (Chapter 7)', 'Motivul unor așteptări modeste: eficiența în formă slabă (Capitolul 7)'),
     [T('any reliable pattern in past returns would be traded away', 'orice tipar sigur din randamentele trecute ar dispărea prin tranzacționare')])))

D.frame(T('The Right Baseline', 'Reperul corect'), items(
    (T('\\textbf{Majority-class baseline}: always forecast the more frequent class of the training sample (here: ``up\'\')', '\\textbf{Reperul clasei majoritare}: prognozăm mereu clasa mai frecventă din eșantionul de antrenare (aici: „creștere”)'),
     [T('S\\&P 500 since 2008: @{sg.sp500.up}\\% of the days are up, so ``always up\'\' is right on @{sg.sp500.base}\\% of the days', 'S\\&P 500 din 2008: @{sg.sp500.up}\\% din zile sînt în creștere, deci „mereu creștere” are dreptate în @{sg.sp500.base}\\% din zile'),
      T('an accuracy of 53\\% sounds like skill; against this baseline it is a loss', 'o acuratețe de 53\\% pare o abilitate; față de acest reper este o pierdere')]),
    (T('Test against the baseline: $z = (\\widehat{\\mathrm{acc}} - \\mathrm{acc}_0)/\\sqrt{\\mathrm{acc}_0(1 - \\mathrm{acc}_0)/n}$', 'Testul față de reper: $z = (\\widehat{\\mathrm{acc}} - \\mathrm{acc}_0)/\\sqrt{\\mathrm{acc}_0(1 - \\mathrm{acc}_0)/n}$'),
     [T('$\\widehat{\\mathrm{acc}}$: the accuracy of the model; $\\mathrm{acc}_0$: that of the baseline; $n$: the test days; $z \\approx N(0,1)$ if the model has no extra skill',
        '$\\widehat{\\mathrm{acc}}$: acuratețea modelului; $\\mathrm{acc}_0$: acuratețea reperului; $n$: zilele de test; $z \\approx N(0,1)$ dacă modelul nu are un avantaj real'),
      T('S\\&P 500, $n = @{sg.sp500.n}$: standard error @{zt.se} points; a 95\\% band of $\\pm$@{zt.band} points around the baseline', 'S\\&P 500, $n = @{sg.sp500.n}$: eroarea standard @{zt.se} puncte; o bandă de 95\\% de $\\pm$@{zt.band} puncte în jurul reperului'),
      T('random forest: @{sg.sp500.RF.acc}\\%, $z = @{sg.sp500.RF.z}$ (p @{sg.sp500.RF.p}); boosting: @{sg.sp500.GB.acc}\\%, $z = @{sg.sp500.GB.z}$ (p @{sg.sp500.GB.p})',
        'random forest: @{sg.sp500.RF.acc}\\%, $z = @{sg.sp500.RF.z}$ (p @{sg.sp500.RF.p}); boosting: @{sg.sp500.GB.acc}\\%, $z = @{sg.sp500.GB.z}$ (p @{sg.sp500.GB.p})')]),
    T('For returns, the matching baseline is the zero or historical-mean forecast \\refWG', 'Pentru randamente, reperul corespunzător este prognoza zero sau media istorică \\refWG')))

chart(T('Accuracy against the Baseline', 'Acuratețea față de reper'), 'sfm_ch13_sign', 'SFM_ch13_sign_prediction', [
    T('Accuracy minus the baseline accuracy (percentage points); band: $\\pm 1.96$ standard errors around zero', 'Acuratețea minus acuratețea reperului (puncte procentuale); banda: $\\pm 1{,}96$ erori standard în jurul lui zero'),
    T('AUC of the logit and of the forest: S\\&P 500 @{sg.sp500.Logit.auc} and @{sg.sp500.RF.auc}, BET @{sg.bet.Logit.auc} and @{sg.bet.RF.auc}, Bitcoin @{sg.btc.Logit.auc} and @{sg.btc.RF.auc}; logit forecasts ``up\'\' on @{sg.sp500.Logit.upred}\\% of S\\&P 500 days',
      'AUC pentru logit și pentru random forest: S\\&P 500 @{sg.sp500.Logit.auc} și @{sg.sp500.RF.auc}, BET @{sg.bet.Logit.auc} și @{sg.bet.RF.auc}, Bitcoin @{sg.btc.Logit.auc} și @{sg.btc.RF.auc}; logit prognozează „creștere” în @{sg.sp500.Logit.upred}\\% din zilele S\\&P 500')],
    h='0.48\\textheight')

D.frame(T('Interpretation: Why Sign Prediction Mostly Fails', 'Interpretare: eșecul obișnuit al prognozei semnului'), items(
    (T('No model beats ``always up\'\' significantly on any market; the flexible models do worse than the logit', 'Pe nicio piață, niciun model nu este semnificativ superior regulii „mereu creștere”; modelele flexibile sînt mai slabe decît logit'),
     [T('S\\&P 500: logit $@{sg.sp500.Logit.diff}$, forest $@{sg.sp500.RF.diff}$, boosting $@{sg.sp500.GB.diff}$ points; Bitcoin: logit $@{sg.btc.Logit.diff}$, forest $@{sg.btc.RF.diff}$',
        'S\\&P 500: logit $@{sg.sp500.Logit.diff}$, random forest $@{sg.sp500.RF.diff}$, boosting $@{sg.sp500.GB.diff}$ puncte; Bitcoin: logit $@{sg.btc.Logit.diff}$, random forest $@{sg.btc.RF.diff}$')]),
    (T('The signal-to-noise ratio of daily returns is tiny', 'Raportul semnal--zgomot al randamentelor zilnice este foarte mic'),
     [T('flexible models find patterns in the noise of the training years (Section 2); the logit is close to the baseline because it hardly moves', 'modelele flexibile găsesc tipare în zgomotul anilor de antrenare (secțiunea 2); logit este aproape de reper deoarece aproape nu se mișcă')]),
    (T('Even a real edge of one point would rarely survive costs: a daily sign strategy trades very often', 'Chiar și un avantaj real de un punct procentual ar rămîne rareori profitabil după costurile de tranzacționare: o strategie zilnică bazată pe semn tranzacționează foarte des'),
     [T('ML gains in returns come from large cross-sections at monthly horizons, not from one index tomorrow (Section 11)', 'cîștigurile aduse de machine learning în prognoza randamentelor provin din secțiuni transversale mari, pe orizonturi lunare, nu dintr-un singur indice pentru ziua următoare (secțiunea 11)')])))

D.recap(("The Sign of Tomorrow's Return", 'semnul randamentului de mîine'), [
    T('Always compare accuracy with the majority-class baseline, and test the difference', 'Comparați întotdeauna acuratețea cu reperul clasei majoritare și testați diferența'),
    T('On the S\\&P 500, BET and Bitcoin no model beats ``always up\'\' significantly', 'Pe S\\&P 500, BET și Bitcoin niciun model nu este semnificativ superior regulii „mereu creștere”'),
    T('Weak-form efficiency (Chapter 7) predicts this result', 'Eficiența în formă slabă (Capitolul 7) prezice acest rezultat')])

# =============================================================================
# 9. VaR PRIN REGRESIE CUANTILICĂ
# =============================================================================
D.section('Application: VaR 1\\% by Quantile Methods', 'Aplicație: VaR 1\\% prin metode cuantilice')

D.frame(T('Quantile Regression (1/2)', 'Regresia cuantilică (1/2)'), items(
    (T('\\refKB: estimate the conditional $\\alpha$-quantile $q_\\alpha(x) = x^\\top\\beta_\\alpha$ by minimising the \\textbf{quantile (pinball) loss}', '\\refKB: estimăm cuantila condiționată de ordin $\\alpha$, $q_\\alpha(x) = x^\\top\\beta_\\alpha$, minimizînd \\textbf{pierderea cuantilică} (pinball loss)'),
     [T('$q_\\alpha(x)$: the level below which $y$ falls with probability $\\alpha$, given $x$; $\\beta_\\alpha$: coefficients specific to the level $\\alpha$',
        '$q_\\alpha(x)$: nivelul sub care $y$ se află cu probabilitatea $\\alpha$, condiționat de $x$; $\\beta_\\alpha$: coeficienți specifici nivelului $\\alpha$')]),
    (T('The loss of an error $u = y - q$:', 'Pierderea pentru o eroare $u = y - q$:') + ' \\[ \\rho_\\alpha(u) = u\\,(\\alpha - \\mathbf{1}\\{u < 0\\}) \\]',
     [T('below the quantile ($u < 0$) the error costs $(1 - \\alpha)|u|$, above it $\\alpha|u|$', 'sub cuantilă ($u < 0$) eroarea costă $(1 - \\alpha)|u|$, deasupra ei $\\alpha|u|$'),
      T('with $\\alpha = 0.01$, the minimum is reached when about 1\\% of the observations lie below $q$', 'cu $\\alpha = 0{,}01$, minimul se atinge cînd aproximativ 1\\% din observații se află sub $q$')])))

D.frame(T('Quantile Regression (2/2)', 'Regresia cuantilică (2/2)'), items(
    (T('With $\\alpha = 0.01$ and $q = -\\mathrm{VaR}$: the average quantile loss of Chapter 10', 'Cu $\\alpha = 0{,}01$ și $q = -\\mathrm{VaR}$: pierderea cuantilică medie din Capitolul 10'),
     [T('$(\\alpha - I_t)(r_t + \\mathrm{VaR}_t)$, with $I_t = 1$ if $r_t < -\\mathrm{VaR}_t$ (an exception) and 0 otherwise', '$(\\alpha - I_t)(r_t + \\mathrm{VaR}_t)$, cu $I_t = 1$ dacă $r_t < -\\mathrm{VaR}_t$ (o depășire) și 0 în rest')]),
    (T('VaR 1\\% for day $t + 1$: $\\widehat{\\mathrm{VaR}}_t = -\\hat q_{0.01}(x_t)$, with $x_t$ = daily, weekly and monthly volatility (square roots of the proxy) and $r_t$',
       'VaR 1\\% pentru ziua $t + 1$: $\\widehat{\\mathrm{VaR}}_t = -\\hat q_{0{,}01}(x_t)$, cu $x_t$ = volatilitatea zilnică, săptămînală și lunară (rădăcinile variabilei proxy) și $r_t$'),
     [T('no distribution is assumed: the data choose how volatility maps into the tail; CAViaR \\refCAViaR{} is a dynamic version', 'nu presupunem nicio distribuție: datele aleg cum se transformă volatilitatea în coadă; CAViaR \\refCAViaR{} este o versiune dinamică')]),
    (T('\\textbf{Quantile gradient boosting}: the same loss with boosted trees (Section 5)', '\\textbf{Quantile gradient boosting}: aceeași pierdere, cu arbori în boosting (secțiunea 5)'),
     [T('flexible, but at 1\\% each leaf sees few tail observations: high variance', 'flexibil, dar la 1\\% fiecare frunză vede puține observații din coadă: varianță mare')])))


def vrow(k, m):
    key = f'qv.{k}.{m}'
    lab = {'sp500': 'S\\&P 500', 'btc': 'Bitcoin'}[k] if m == 'QR' else ''
    return f'{lab} & {m} & @{{{key}.x}} & $@{{{key}.rate}}$ & @{{{key}.puc}} & @{{{key}.pind}} & $@{{{key}.ql}}$ & $@{{{key}.mv}}$'


VH = T('& Method & $x$ & rate (\\%) & p Kupiec & p ind. & loss $\\times 100$ & mean VaR', '& Metoda & $x$ & rata (\\%) & p Kupiec & p ind. & pierdere $\\times 100$ & VaR mediu')
D.frame(T('Backtest of VaR 1\\%: Quantile Methods, HS and GARCH-t (1/2)', 'Backtesting pentru VaR 1\\%: metode cuantilice, HS și GARCH-t (1/2)'), table(
    'llrrrrrr', VH, [vrow('sp500', m) for m in ['QR', 'QGB', 'HS', 'GARCH-t']] + ['\\midrule'] +
    [vrow('btc', m) for m in ['QR', 'QGB', 'HS', 'GARCH-t']], size='scriptsize').replace('\\midrule \\\\', '\\midrule') + items(
    (T('Columns', 'Coloanele'),
     [T('$x$: the number of exceptions ($r_{t+1} < -\\widehat{\\mathrm{VaR}}_t$) in $n$ test days; rate $= x/n$, target 1\\%', '$x$: numărul depășirilor ($r_{t+1} < -\\widehat{\\mathrm{VaR}}_t$) în cele $n$ zile de test; rata $= x/n$, ținta 1\\%'),
      T('p Kupiec: p-value of the Kupiec test of the rate; p ind.: p-value of the Christoffersen test of independence (Chapter 10)', 'p Kupiec: p-value-ul testului Kupiec al ratei; p ind.: p-value-ul testului de independență Christoffersen (Capitolul 10)'),
      T('loss $\\times 100$: the mean quantile loss (smaller is better); mean VaR: the average VaR 1\\%, in \\%', 'pierdere $\\times 100$: pierderea cuantilică medie (mai mică este mai bună); VaR mediu: media VaR 1\\%, în \\%')]),
    (T('Methods', 'Metodele'),
     [T('QR: linear quantile regression; QGB: quantile gradient boosting', 'QR: regresie cuantilică liniară; QGB: quantile gradient boosting'),
      T('HS: historical simulation on 500 days; GARCH-t: re-estimated each January (Chapter 10)', 'HS: simulare istorică pe 500 de zile; GARCH-t: reestimat în fiecare ianuarie (Capitolul 10)')])) + ql('SFM_ch13_quantile_var'), 'footnotesize')

D.frame(T('Backtest of VaR 1\\%: Quantile Methods, HS and GARCH-t (2/2)', 'Backtesting pentru VaR 1\\%: metode cuantilice, HS și GARCH-t (2/2)'), items(
    (T('S\\&P 500 (@{qv.sp500.n} days from @{qv.sp500.first}): QR has the right rate (@{qv.sp500.QR.rate}\\%) and the smallest loss',
       'S\\&P 500 (@{qv.sp500.n} zile de la @{qv.sp500.first}): QR are rata corectă (@{qv.sp500.QR.rate}\\%) și cea mai mică pierdere'),
     [T('its exceptions are not fully independent (p ind.\\ @{qv.sp500.QR.pind})', 'depășirile lui nu sînt complet independente (p ind.\\ @{qv.sp500.QR.pind})'),
      T('GARCH-t is exceeded too often (@{qv.sp500.GARCH-t.rate}\\%)', 'GARCH-t este depășit prea des (@{qv.sp500.GARCH-t.rate}\\%)')]),
    (T('Bitcoin (@{qv.btc.n} days): HS has the best rate (@{qv.btc.HS.rate}\\%)', 'Bitcoin (@{qv.btc.n} de zile): HS are cea mai bună rată (@{qv.btc.HS.rate}\\%)'),
     [T('QGB is rejected by the Kupiec test (@{qv.btc.QGB.rate}\\%, p @{qv.btc.QGB.puc})', 'QGB este respins de testul Kupiec (@{qv.btc.QGB.rate}\\%, p @{qv.btc.QGB.puc})'),
      T('at 1\\% each leaf of the boosted trees sees few tail days: high variance (Section 2)', 'la 1\\% fiecare frunză a arborilor din boosting vede puține zile din coadă: varianță mare (secțiunea 2)')]),
    (T('Reading the p-values', 'Interpretarea p-value-urilor'),
     [T('p Kupiec below 0.05: the exception rate differs significantly from 1\\%', 'p Kupiec sub 0,05: rata depășirilor diferă semnificativ de 1\\%'),
      T('p ind.\\ below 0.05: the exceptions come in clusters, so the VaR reacts too slowly', 'p ind.\\ sub 0,05: depășirile apar grupat, deci VaR reacționează prea lent')])), 'footnotesize')

chart(T('VaR 1\\% Forecasts in 2020 and 2025', 'Prognozele VaR 1\\% în 2020 și 2025'), 'sfm_ch13_qvar', 'SFM_ch13_quantile_var', [
    T('S\\&P 500: next-day returns and minus VaR 1\\% of the four methods; dots: exceedances of the quantile-regression VaR',
      'S\\&P 500: randamentele zilei următoare și minus VaR 1\\% pentru cele patru metode; punctele: depășirile VaR obținut prin regresie cuantilică'),
    T('Interpretation: QR and GARCH-t move with volatility; HS reacts only slowly; in 2020 HS was exceeded @{qv.sp500.HS.x20} times, QR @{qv.sp500.QR.x20} times, and the HS exceptions cluster (p ind.\\ @{qv.sp500.HS.pind})',
      'Interpretare: QR și GARCH-t urmează volatilitatea; HS reacționează lent; în 2020, HS a fost depășit de @{qv.sp500.HS.x20} ori, QR de @{qv.sp500.QR.x20} ori, iar depășirile HS apar grupat (p ind.\\ @{qv.sp500.HS.pind})')],
    h='0.48\\textheight')

D.recap(('VaR 1\\% by Quantile Methods', 'VaR 1\\% prin metode cuantilice'), [
    T('Quantile regression minimises the pinball loss: VaR without a distributional assumption', 'Regresia cuantilică minimizează pierderea cuantilică: VaR fără o ipoteză despre distribuție'),
    T('A linear quantile model on volatility features passes the Kupiec test for the S\\&P 500', 'Un model cuantilic liniar pe variabile de volatilitate trece testul Kupiec pentru S\\&P 500'),
    T('Flexible quantile models at 1\\% need much more tail data than we have', 'Modelele cuantilice flexibile la 1\\% au nevoie de mult mai multe date din coadă decît avem')])

# =============================================================================
# 10. DATA SNOOPING
# =============================================================================
D.section('Data Snooping and the Deflated Sharpe Ratio', 'Data snooping și raportul Sharpe deflatat')

D.frame(T('The Best of $N$ Strategies (1/2)', 'Cea mai bună dintre $N$ strategii (1/2)'), items(
    (T('\\textbf{Data snooping}: trying many models or rules on the same data and reporting the best \\refWhite', '\\textbf{Data snooping}: încercarea multor modele sau reguli pe aceleași date și raportarea celei mai bune \\refWhite'),
     [T('with enough trials, a rule without skill looks excellent in sample', 'cu suficiente încercări, o regulă fără nicio abilitate pare excelentă în eșantion'),
      T('hundreds of published return predictors would not survive a multiple-testing correction \\refHLZ', 'sute de predictori ai randamentelor publicați nu ar trece de o corecție pentru testare multiplă \\refHLZ')]),
    (T('The annual Sharpe ratio of a strategy without skill over $Y$ years is about $N(0, 1/Y)$', 'Raportul Sharpe anual al unei strategii fără abilitate pe $Y$ ani este aproximativ $N(0, 1/Y)$'),
     [T('Sharpe ratio: mean excess return divided by its standard deviation (Chapter 1); $Y$: the number of years of data', 'raportul Sharpe: randamentul mediu în exces împărțit la abaterea lui standard (Capitolul 1); $Y$: numărul de ani de date'),
      T('mean 0: no skill; variance $1/Y$: the estimation error, which falls with the length of the sample', 'media 0: nicio abilitate; varianța $1/Y$: eroarea de estimare, care scade odată cu lungimea eșantionului')])))

D.frame(T('The Best of $N$ Strategies (2/2)', 'Cea mai bună dintre $N$ strategii (2/2)'), items(
    (T('Expected maximum of $N$ independent zero-skill trials \\refBLdP', 'Maximul așteptat al celor $N$ încercări independente, fără abilitate \\refBLdP') + ' \\[ \\mathrm{E}[\\max] \\approx \\sigma\\Big[(1 - \\gamma)\\,\\Phi^{-1}\\big(1 - \\tfrac1N\\big) + \\gamma\\,\\Phi^{-1}\\big(1 - \\tfrac{1}{Ne}\\big)\\Big] \\]',
     [T('$\\sigma$: the standard deviation of the Sharpe ratios across trials ($1/\\sqrt{Y}$ above); $N$: the number of trials', '$\\sigma$: abaterea standard a rapoartelor Sharpe între încercări ($1/\\sqrt{Y}$ mai sus); $N$: numărul de încercări'),
      T('$\\Phi^{-1}$: the quantile function of the standard Normal distribution; $e \\approx 2.718$; $\\gamma = 0.5772$: the Euler--Mascheroni constant',
        '$\\Phi^{-1}$: funcția cuantilă a distribuției Normale standard; $e \\approx 2{,}718$; $\\gamma = 0{,}5772$: constanta Euler--Mascheroni')]),
    (T('Example: 10 years, $\\sigma = 1/\\sqrt{10} = @{ms.sd}$, $N = 100$', 'Exemplu: 10 ani, $\\sigma = 1/\\sqrt{10} = @{ms.sd}$, $N = 100$'),
     [T('$@{ms.sd}\\,[(1 - \\gamma)\\,@{ms.z1} + \\gamma\\,@{ms.z2}] = @{ms.100.f}$: a Sharpe ratio this large is expected from pure luck', '$@{ms.sd}\\,[(1 - \\gamma)\\,@{ms.z1} + \\gamma\\,@{ms.z2}] = @{ms.100.f}$: un asemenea raport Sharpe apare, în medie, doar prin noroc'),
      T('simulation: the best of 10, 100 and 1000 zero-skill strategies has an average Sharpe ratio of $@{ms.10}$, $@{ms.100}$, $@{ms.1000}$', 'simulare: cea mai bună dintre 10, 100 și 1000 de strategii fără abilitate are un raport Sharpe mediu de $@{ms.10}$; $@{ms.100}$; $@{ms.1000}$')])))

chart(T('Selection Bias in Practice', 'Distorsiunea de selecție în practică'), 'sfm_ch13_snooping', 'SFM_ch13_data_snooping', [
    T('Left: the best annual Sharpe ratio among $N$ strategies without skill (10 years); right: @{sn.N} moving-average rules on the S\\&P 500 (long or cash), 2000--2014 against 2015--2026',
      'Stînga: cel mai bun raport Sharpe anual dintre $N$ strategii fără abilitate (10 ani); dreapta: @{sn.N} de reguli cu medii mobile pe S\\&P 500 (investit sau numerar), 2000--2014 față de 2015--2026'),
    T('Interpretation: the best rule in sample (MA @{sn.s}/@{sn.l}, Sharpe $@{sn.is}$) earns $@{sn.oos}$ afterwards, rank @{sn.rank} of @{sn.N}, below buy and hold ($@{sn.bh_oos}$); in and out of sample Sharpe ratios correlate $@{sn.corr}$',
      'Interpretare: cea mai bună regulă în eșantion (MA @{sn.s}/@{sn.l}, Sharpe $@{sn.is}$) obține apoi $@{sn.oos}$, locul @{sn.rank} din @{sn.N}, sub buy and hold ($@{sn.bh_oos}$); rapoartele Sharpe din eșantion și din afara lui au corelația $@{sn.corr}$')],
    h='0.48\\textheight')

D.frame(T('The Deflated Sharpe Ratio (1/2)', 'Raportul Sharpe deflatat (1/2)'), items(
    T('\\refBLdP: the probability that the true Sharpe ratio exceeds what the best of $N$ trials would show by luck', '\\refBLdP: probabilitatea ca adevăratul raport Sharpe să depășească ceea ce ar arăta din noroc cea mai bună dintre $N$ încercări') + ' \\[ \\mathrm{DSR} = \\Phi\\Big(\\dfrac{(\\widehat{SR} - SR_0)\\sqrt{T - 1}}{\\sqrt{1 - \\hat\\gamma_3\\widehat{SR} + \\frac{\\hat\\gamma_4 - 1}{4}\\widehat{SR}^2}}\\Big) \\]',
    (T('Notation', 'Notațiile'),
     [T('$\\widehat{SR}$: the per-day Sharpe ratio of the chosen strategy; $T$: the number of days', '$\\widehat{SR}$: raportul Sharpe zilnic al strategiei alese; $T$: numărul de zile'),
      T('$SR_0$: the expected maximum of $N$ zero-skill trials (above), with $\\sigma$ = the standard deviation of the $N$ Sharpe ratios tried', '$SR_0$: maximul așteptat al celor $N$ încercări fără abilitate (mai sus), cu $\\sigma$ = abaterea standard a celor $N$ rapoarte Sharpe încercate'),
      T('$\\hat\\gamma_3$: the skewness, $\\hat\\gamma_4$: the kurtosis of the daily returns (3 for the Normal distribution); $\\Phi$: the standard Normal distribution function',
        '$\\hat\\gamma_3$: asimetria, $\\hat\\gamma_4$: boltirea randamentelor zilnice (3 pentru distribuția Normală); $\\Phi$: funcția de repartiție Normală standard')]),
    (T('Reading', 'Interpretarea'),
     [T('the DSR is a probability in $(0, 1)$; above 0.95: the Sharpe ratio is significant after the correction for $N$ trials', 'DSR este o probabilitate în $(0, 1)$; peste 0,95: raportul Sharpe este semnificativ după corecția pentru $N$ încercări'),
      T('negative skewness and fat tails enlarge the denominator and lower the DSR', 'asimetria negativă și cozile groase măresc numitorul și reduc DSR')])))

D.frame(T('The Deflated Sharpe Ratio (2/2)', 'Raportul Sharpe deflatat (2/2)'), items(
    (T('\\textbf{Step by step}: the best MA rule, $N = @{sn.N}$, $T = @{sn.T}$ days', '\\textbf{Pas cu pas}: cea mai bună regulă MA, $N = @{sn.N}$, $T = @{sn.T}$ de zile'),
     [T('$\\widehat{SR} = @{sn.ispp}$ per day ($\\times\\sqrt{252} = @{sn.is}$); $\\hat\\gamma_3 = @{sn.skew}$, $\\hat\\gamma_4 = @{sn.kurt}$', '$\\widehat{SR} = @{sn.ispp}$ pe zi ($\\times\\sqrt{252} = @{sn.is}$); $\\hat\\gamma_3 = @{sn.skew}$, $\\hat\\gamma_4 = @{sn.kurt}$'),
      T('$SR_0 = @{sn.sdpp}\\,[(1 - \\gamma)\\,@{sn.zN1} + \\gamma\\,@{sn.zN2}] = @{sn.sr0}$ (annual $@{sn.sr0a}$)', '$SR_0 = @{sn.sdpp}\\,[(1 - \\gamma)\\,@{sn.zN1} + \\gamma\\,@{sn.zN2}] = @{sn.sr0}$ (anual $@{sn.sr0a}$)'),
      T('$z$, the argument of $\\Phi$ in the DSR formula: $z = @{sn.z}$, so DSR $= \\Phi(z) = @{sn.dsr}$', '$z$, argumentul lui $\\Phi$ din formula DSR: $z = @{sn.z}$, deci DSR $= \\Phi(z) = @{sn.dsr}$'),
      T('without the correction for $N$ (PSR, probabilistic Sharpe ratio, $SR_0 = 0$): $@{sn.psr}$', 'fără corecția pentru $N$ (PSR, probabilistic Sharpe ratio, cu $SR_0 = 0$): $@{sn.psr}$')]),
    T('Interpretation: alone the rule looks significant; after @{sn.N} trials it is not (DSR below 0.95), as its later performance confirms',
      'Interpretare: luată singură, regula pare semnificativă; după @{sn.N} de încercări nu mai este (DSR sub 0,95), așa cum confirmă și rezultatele ulterioare')))

D.recap(('Data Snooping', 'data snooping'), [
    T('The best of many trials is biased upwards; the bias grows like $\\sqrt{2\\ln N}$', 'Cel mai bun rezultat dintre multe încercări este deplasat în sus; deplasarea crește proporțional cu $\\sqrt{2\\ln N}$'),
    T('Report the number of trials and deflate the Sharpe ratio', 'Raportați numărul de încercări și deflatați raportul Sharpe'),
    T('A final test sample used only once is the simplest protection', 'Un eșantion de test final folosit o singură dată este protecția cea mai simplă')])

# =============================================================================
# 11. STUDIU DE CAZ
# =============================================================================
D.section('Case Study: Gu, Kelly and Xiu (2020)', 'Studiu de caz: Gu, Kelly și Xiu (2020)')

D.frame(T('Empirical Asset Pricing via Machine Learning', 'Empirical Asset Pricing via Machine Learning'), cols(items(
    (T('\\refGKX: which methods best predict the monthly return of individual US stocks?', '\\refGKX: care metode prezic cel mai bine randamentul lunar al acțiunilor americane individuale?'),
     [T('nearly 30,000 stocks, 1957--2016; 94 firm characteristics, interacted with 8 macroeconomic variables, plus 74 industry dummies: 920 features',
        'aproape 30\\,000 de acțiuni, 1957--2016; 94 de caracteristici ale firmelor, în interacțiune cu 8 variabile macroeconomice, plus 74 de variabile indicatoare de industrie: 920 de variabile')]),
    (T('Design: 18 years of training (1957--1974), 12 of validation (1975--1986), 30 of test (1987--2016)', 'Schema de estimare: 18 ani de antrenare (1957--1974), 12 de validare (1975--1986), 30 de test (1987--2016)'),
     [T('refit once a year; the training window grows, the validation window rolls forward: a walk-forward design', 'reestimare o dată pe an; fereastra de antrenare crește, cea de validare avansează: o schemă walk-forward'),
      T('models: OLS, penalised and dimension-reduction regressions, random forest, boosted trees', 'modele: OLS, regresii penalizate și cu reducerea dimensiunii, random forest, arbori în boosting'),
      T('networks NN1--NN5: NN$k$ has $k$ hidden layers', 'rețelele NN1--NN5: NN$k$ are $k$ straturi ascunse')])),
    ph('harper', T('Charles M.\\ Harper Center, Chicago Booth, where Gu and Xiu worked', 'Charles M.\\ Harper Center, Chicago Booth, unde au lucrat Gu și Xiu'), h='0.36\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('Evaluation in the Paper', 'Evaluarea din lucrare'), items(
    (T('$R^2_{OOS} = 1 - \\sum_{(i,t)}(r_{i,t+1} - \\hat r_{i,t+1})^2 / \\sum_{(i,t)} r_{i,t+1}^2$: the benchmark is a forecast of \\textbf{zero}, not the historical mean',
       '$R^2_{OOS} = 1 - \\sum_{(i,t)}(r_{i,t+1} - \\hat r_{i,t+1})^2 / \\sum_{(i,t)} r_{i,t+1}^2$: reperul este o prognoză \\textbf{zero}, nu media istorică'),
     [T('$r_{i,t+1}$: the return of stock $i$ in month $t + 1$; $\\hat r_{i,t+1}$: its forecast; the sums run over all stock-months of the test period',
        '$r_{i,t+1}$: randamentul acțiunii $i$ în luna $t + 1$; $\\hat r_{i,t+1}$: prognoza lui; sumele parcurg toate perechile acțiune--lună din perioada de test'),
      T('the authors note that the historical mean of a stock is so noisy that it is easy to beat: a zero benchmark is harder', 'autorii arată că media istorică a unei acțiuni este atît de zgomotoasă încît este ușor de depășit: reperul zero este mai exigent')]),
    (T('Diebold--Mariano tests between all pairs of models, with a Bonferroni correction for the 12 comparisons', 'Teste Diebold--Mariano între toate perechile de modele, cu o corecție Bonferroni pentru cele 12 comparații (pragul de semnificație se împarte la numărul comparațiilor)'),
     [T('the same tool as Section 7, applied to a panel of stocks', 'același instrument ca în secțiunea 7, aplicat unui panel de acțiuni')]),
    (T('Economic value: each month, sort stocks into deciles by forecast; buy the top decile, sell the bottom one', 'Valoarea economică: în fiecare lună, sortăm acțiunile în decile după prognoză; cumpărăm decila de sus și vindem decila de jos'),
     [T('value-weighted portfolios; Sharpe ratio, drawdown and turnover (the share of the portfolio traded each month)', 'portofolii ponderate cu valoarea de piață; raportul Sharpe, drawdown și rulajul (ponderea portofoliului tranzacționată lunar)')])))

chart(T('Results: Small $R^2$, Large Economic Value', 'Rezultatele: $R^2$ mic, valoare economică mare'), 'sfm_ch13_gkx', 'SFM_ch13_case_gkx', [
    T('Left: monthly $R^2_{OOS}$ (Table 1): OLS with all 920 features $@{gkx.ols}\\%$; OLS-3 (size, value, momentum) $@{gkx.OLS3.r2}\\%$; random forest $@{gkx.RF.r2}\\%$; NN3 $@{gkx.NN3.r2}\\%$ ($@{gkx.NN3.top}\\%$ for the 1,000 largest stocks)',
      'Stînga: $R^2_{OOS}$ lunar (tabelul 1): OLS cu toate cele 920 de variabile $@{gkx.ols}\\%$; OLS-3 (mărime, valoare, momentum) $@{gkx.OLS3.r2}\\%$; random forest $@{gkx.RF.r2}\\%$; NN3 $@{gkx.NN3.r2}\\%$ ($@{gkx.NN3.top}\\%$ pentru cele mai mari 1\\,000 de acțiuni)'),
    T('Right: annual Sharpe ratio of the value-weighted decile spread (Table 7): OLS-3 $@{gkx.OLS3.sr}$, random forest $@{gkx.RF.sr}$, NN4 $@{gkx.NN4.sr}$',
      'Dreapta: raportul Sharpe anual al diferenței dintre decile, ponderată cu valoarea de piață (tabelul 7): OLS-3 $@{gkx.OLS3.sr}$, random forest $@{gkx.RF.sr}$, NN4 $@{gkx.NN4.sr}$')],
    h='0.48\\textheight')

D.frame(T('Interpretation and Limits', 'Interpretare și limite'), items(
    (T('What the paper shows', 'Rezultatele lucrării'),
     [T('nonlinear models (trees, networks) beat linear ones because they capture interactions between characteristics', 'modelele neliniare (arbori, rețele) sînt superioare celor liniare deoarece surprind interacțiunile dintre caracteristici'),
      T('all methods agree on the main signals: price trends (momentum, short-term reversal), liquidity, volatility', 'toate metodele sînt de acord asupra semnalelor principale: tendințele prețului (momentum, inversarea pe termen scurt), lichiditatea, volatilitatea'),
      T('shallow networks are enough: 3--4 layers beat 5; ``deep\'\' learning does not help with monthly returns', 'rețelele puțin adînci sînt suficiente: 3--4 straturi dau rezultate mai bune decît 5; deep learning nu ajută la randamentele lunare')]),
    (T('What to keep in mind', 'De reținut'),
     [T('an $R^2_{OOS}$ below 1\\% per month is a large result here: compare with the sign forecasts of Section 8', 'un $R^2_{OOS}$ sub 1\\% pe lună este un rezultat important în acest context: comparați cu prognozele semnului din secțiunea 8'),
      T('neural portfolios turn over more than 100\\% of their value each month: trading costs reduce the gains', 'portofoliile pe baza rețelelor își schimbă lunar peste 100\\% din valoare: costurile de tranzacționare reduc cîștigurile'),
      T('the evidence is a cross-section of thousands of stocks; it says little about timing one index', 'dovezile provin dintr-o secțiune transversală de mii de acțiuni; ele spun puțin despre alegerea momentului pentru un singur indice')])))

# =============================================================================
# 12. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Do machine-learning volatility forecasts help on the Bucharest Stock Exchange, where liquidity is low and intraday data are scarce?}',
       '\\textbf{Ajută prognozele de volatilitate prin machine learning la Bursa de Valori București, unde lichiditatea este redusă și datele intrazilnice sînt rare?}'),
     [T('today: on the S\\&P 500 the lasso gains $@{rv.sp500.Lasso.r2h}\\%$ over HAR; on Bitcoin no significant gain ($@{rv.btc.Lasso.r2h}\\%$, p @{rv.btc.Lasso.dmp})', 'azi: la S\\&P 500, lasso cîștigă $@{rv.sp500.Lasso.r2h}\\%$ față de HAR; la Bitcoin niciun cîștig semnificativ ($@{rv.btc.Lasso.r2h}\\%$, p @{rv.btc.Lasso.dmp})'),
      T('the BET has no high and low prices in our data, so the proxy must come from squared returns, which are noisier', 'BET nu are prețuri maxime și minime în datele noastre, deci variabila proxy trebuie construită din randamente la pătrat, mai zgomotoase')]),
    T('Why it is open: thin trading, a different calendar from New York, few crises in the sample', 'Întrebarea rămîne deschisă: tranzacționare rară, un calendar diferit de cel de la New York, puține crize în eșantion'),
    T('Related work, as further reading: \\refPeleA{} (statistical classification of crypto assets); \\refPeleB{} (how precise forecast comparisons can be)',
      'Lucrări înrudite, ca lectură suplimentară: \\refPeleA{} (clasificarea statistică a activelor cripto); \\refPeleB{} (cît de precise pot fi comparațiile de prognoze)'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on HAR and machine learning for emerging and frontier markets', '\\textbf{Literatura}: o listă a studiilor despre HAR și machine learning pentru piețele emergente și de frontieră'),
    T('\\textbf{Code}: a first draft of the walk-forward comparison of HAR, lasso and random forest for the BET and three BVB stocks', '\\textbf{Cod}: o primă versiune a comparației walk-forward dintre HAR, lasso și random forest pentru BET și trei acțiuni BVB'),
    T('\\textbf{Robustness}: other proxies (absolute returns, two-day returns), other horizons (1, 5, 22 days), S\\&P 500 features of the previous close', '\\textbf{Robustețe}: alte variabile proxy (randamente absolute, randamente pe două zile), alte orizonturi (1, 5, 22 de zile), variabile S\\&P 500 de la închiderea precedentă'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that compares HAR, lasso and a random forest for the weekly log variance of the BET index (proxy: squared daily returns), with yearly walk-forward refits from 2012, purging the last 5 training rows, and reports the out-of-sample R2, QLIKE and Diebold-Mariano tests.}',
        '\\aiprompt{Write Python code that compares HAR, lasso and a random forest for the weekly log variance of the BET index (proxy: squared daily returns), with yearly walk-forward refits from 2012, purging the last 5 training rows, and reports the out-of-sample R2, QLIKE and Diebold-Mariano tests.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Time order: no shuffled folds; features use only data up to the close of day $t$; the last training targets do not reach into the test period',
      'Ordinea temporală: fără partiții amestecate; variabilele folosesc doar date pînă la închiderea zilei $t$; ultimele ținte de antrenare nu ajung în perioada de test'),
    T('Scaling and tuning inside the training window only (standardisation, $\\lambda$, the number of trees)', 'Scalarea și alegerea hiperparametrilor doar în fereastra de antrenare (standardizarea, $\\lambda$, numărul de arbori)'),
    T('Calendars: the BVB closes before New York; S\\&P 500 features must come from the previous New York close', 'Calendarele: BVB se închide înaintea bursei din New York; variabilele S\\&P 500 trebuie să provină de la închiderea precedentă din New York'),
    T('A benchmark in every table (HAR, the majority class, historical simulation) and a test of the difference', 'Un reper în fiecare tabel (HAR, clasa majoritară, simularea istorică) și un test al diferenței'),
    T('The number of models tried, reported; references: every cited paper must exist; check the DOI', 'Numărul de modele încercate, raportat; referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: do machine-learning models forecast the volatility of BVB stocks better than HAR?', '\\textbf{Întrebarea}: prognozează modelele de machine learning volatilitatea acțiunilor BVB mai bine decît HAR?'),
     [T('data: BET, Banca Transilvania, OMV Petrom, BRD (EODHD); S\\&P 500 and DAX as external features', 'date: BET, Banca Transilvania, OMV Petrom, BRD (EODHD); S\\&P 500 și DAX ca variabile externe')]),
    (T('Steps', 'Pași'),
     [T('a variance proxy for each stock (range where available, squared returns otherwise), weekly target', 'o variabilă proxy a varianței pentru fiecare acțiune (amplitudinea, unde există, altfel randamentele la pătrat), țintă săptămînală'),
      T('HAR, lasso, random forest and a small MLP, walk-forward from 2014, yearly refits with purging', 'HAR, lasso, random forest și un MLP mic, walk-forward din 2014, reestimare anuală cu purging'),
      T('out-of-sample $R^2$, QLIKE and Diebold--Mariano; Shapley values for the external features', '$R^2$ în afara eșantionului, QLIKE și Diebold--Mariano; valori Shapley pentru variabilele externe')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Machine learning is judged by its error on new data; the bias--variance trade-off sets how flexible a model should be', 'Machine learning se judecă după eroarea pe date noi; compromisul deplasare--varianță stabilește cît de flexibil trebuie să fie un model'),
    T('Time series need walk-forward validation with purging; random folds on overlapping targets fake predictability', 'Seriile de timp au nevoie de validare walk-forward cu purging; partițiile aleatoare pe ținte suprapuse simulează o predictibilitate falsă'),
    T('Ridge and lasso shrink; forests average trees; boosting adds trees; networks learn nonlinear features', 'Ridge și lasso contractă coeficienții; random forest face media arborilor; boosting adaugă arbori; rețelele învață variabile neliniare'),
    T('Volatility: HAR is hard to beat, the gains are a few per cent; the sign of returns: no gain over ``always up\'\'', 'Volatilitatea: HAR este greu de depășit, cîștigurile sînt de cîteva procente; semnul randamentelor: niciun cîștig față de „mereu creștere”'),
    T('VaR 1\\%: linear quantile regression works; flexible quantile models lack tail data', 'VaR 1\\%: regresia cuantilică liniară funcționează; modelelor cuantilice flexibile le lipsesc datele din coadă'),
    T('Report the number of trials and deflate the Sharpe ratio; compare every model with a simple benchmark', 'Raportați numărul de încercări și deflatați raportul Sharpe; comparați fiecare model cu un reper simplu')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Bias--variance', 'Deplasare--varianță') + ' & $E[(y_0 - \\hat f)^2] = (E\\hat f - f)^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$',
     'Ridge & $(X^\\top X + \\lambda I)^{-1}X^\\top y$; \\ ' + T('1D', '1D') + ': $S_{xy}/(S_{xx} + \\lambda)$',
     'Lasso (1D) & $\\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$',
     T('Random forest', 'Random forest') + ' & $\\mathrm{Var} = \\rho\\sigma^2 + (1 - \\rho)\\sigma^2/B$',
     T('Boosting', 'Boosting') + ' & $F_m = F_{m-1} + \\nu h_m$',
     'MLP & $\\hat y = c + \\sum_k v_k g(b_k + w_k^\\top x)$',
     'HAR & $y_t = \\beta_0 + \\beta_d\\ln v_t + \\beta_w\\ln v_t^{(w)} + \\beta_m\\ln v_t^{(m)} + u_t$',
     '$R^2_{OOS}$ & $1 - \\sum(y - \\hat y)^2/\\sum(y - \\tilde y)^2$',
     T('Quantile loss', 'Pierderea cuantilică') + ' & $\\rho_\\alpha(u) = u(\\alpha - \\mathbf{1}\\{u < 0\\})$',
     'DSR & $\\Phi\\big((\\widehat{SR} - SR_0)\\sqrt{T - 1}/\\sqrt{1 - \\hat\\gamma_3\\widehat{SR} + (\\hat\\gamma_4 - 1)\\widehat{SR}^2/4}\\big)$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a model has a training $R^2$ of 80\\% and an out-of-sample $R^2$ of $-5\\%$. What happened?', '\\textbf{Întrebare}: un model are un $R^2$ de antrenare de 80\\% și un $R^2$ în afara eșantionului de $-5\\%$. Ce s-a întîmplat?'),
     [T('\\textbf{Answer}: overfitting: the model learned the noise of the training sample and loses against the benchmark on new data', '\\textbf{Răspuns}: overfitting: modelul a învățat zgomotul eșantionului de antrenare și pierde în fața reperului pe date noi')]),
    (T('\\textbf{Question}: why is random K-fold wrong for a 21-day return target?', '\\textbf{Întrebare}: de ce este greșit K-fold aleator pentru o țintă de tip randament pe 21 de zile?'),
     [T('\\textbf{Answer}: neighbouring targets overlap, so the test answers leak into training; use walk-forward with purging', '\\textbf{Răspuns}: țintele vecine se suprapun, deci răspunsurile de test ajung în antrenare; folosiți walk-forward cu purging')]),
    (T('\\textbf{Question}: a classifier is right on 53\\% of the days and the market rose on 54\\% of them. Is it useful?', '\\textbf{Întrebare}: un clasificator are dreptate în 53\\% din zile, iar piața a crescut în 54\\% dintre ele. Este util?'),
     [T('\\textbf{Answer}: no: ``always up\'\' is right on 54\\% of the days', '\\textbf{Răspuns}: nu: „mereu creștere” are dreptate în 54\\% din zile')]),
    T('Next: Chapter 14, crypto assets', 'Urmează: Capitolul 14, activele cripto')))

D.references(bib([k for k in BIBKEYS if k not in ('BHL', 'Boll')]), per=12)

if __name__ == '__main__':
    D.write(V)
