r"""
build_seminar13.py -- Seminarul 13 (Machine learning în finanțe), EN + RO dintr-o singură sursă
=============================================================================================
Seminarul are loc ÎNAINTEA cursului 13: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_13/sem13_results.json (seminar13.py) și ch13_numbers.json.
Ieșire:
  EN/Seminars/seminar13_machine_learning.tex          (+ _solutions.tex)
  RO/Seminarii/seminar13_invatare_automata_ro.tex     (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_13/seminar13.py && python3 latex/build_seminar13.py && python3 latex/sfm_build.py compile 13
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch13_common import REFS, T, bib, load, load_sem, pv, put_date, put_rv, put_sign, put_var   # noqa: E402

S = load_sem()
N = load()
V = Values()
D = Deck(13, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_13}{SFM_ch13_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for k in ('a1', 'a2'):
    for c in ('mean', 'bias', 'bias2', 'var', 'mse'):
        V.put(f'{k}.{c}', A[k][c], 4 if c in ('bias2', 'var') else 3)
for k in ('a3', 'a4'):
    t = A[k]
    V.put(f'{k}.mean', t['mean'], 3)
    V.put(f'{k}.sse0', t['sse0'], 3)
    b = t['best']
    V.put(f'{k}.s', b['split'], 2 if k == 'a3' else 1)
    V.put(f'{k}.l', b['mean_L'], 3)
    V.put(f'{k}.r', b['mean_R'], 3)
    V.put(f'{k}.sseL', b['sse_L'], 3)
    V.put(f'{k}.sseR', b['sse_R'], 3)
    V.put(f'{k}.sse', b['sse'], 3)
    V.put(f'{k}.r2', 1 - b['sse'] / t['sse0'], 3)
SPLITS3 = [f'$⁅{r["split"]:.2f}⁆$ & $⁅{r["mean_L"]:.3f}⁆$ & $⁅{r["mean_R"]:.3f}⁆$ & $⁅{r["sse"]:.3f}⁆$' for r in A['a3']['rows']]
SPLITS4 = [f'$⁅{r["split"]:.1f}⁆$ & $⁅{r["mean_L"]:.3f}⁆$ & $⁅{r["mean_R"]:.3f}⁆$ & $⁅{r["sse"]:.3f}⁆$' for r in A['a4']['rows']]
r5 = A['a5']
V.put('a5.ols', r5['ols'], 2)
V.put('a5.b10', r5['l10']['b'], 3)
V.put('a5.f10', r5['l10']['factor'], 3)
V.put('a5.b50', r5['l50']['b'], 2)
V.put('a5.f50', r5['l50']['factor'], 2)
r6 = A['a6']
for l in ('10', '30', '50'):
    V.put(f'a6.l{l}', r6[f'l{l}'], 2)
V.put('a6.zero', r6['zero_from'], 0)
d7 = A['a7']
V.raw('a7.folds', str(d7['n_folds']))
put_date(V, 'a7.first', d7['first'])
r0 = d7['rows'][0]
V.int('a7.ntr', r0['n_train'])
V.int('a7.nte', r0['n_test'])
put_date(V, 'a7.last', r0['last_train'])
put_date(V, 'a7.purge', r0['first_purged'])
a8 = A['a8']
for c in ('block', 'emb', 'train_inner', 'train_first', 'train_last', 'lost_inner'):
    V.int(f'a8.{c}', a8[c])
put_rv(V, 'b1', S['B1'], ['HAR', 'RF'])
V.raw('b1.yrf', str(S['B1']['years_rf_better']))
V.raw('b1.ny', str(S['B1']['n_years']))
put_rv(V, 'b2', S['B2'], ['HAR', 'Lasso', 'RF', 'GB', 'MLP'])
put_sign(V, 'b3', S['B3'], ['Logit', 'RF'])
V.raw('b3.yabove', str(S['B3']['years_rf_above']))
V.raw('b3.ny', str(S['B3']['n_years']))
for k in ('tlv', 'snp'):
    put_sign(V, f'b4.{k}', S['B4'][k], ['Logit', 'RF', 'GB'])
put_var(V, 'b5', S['B5'], ['QR', 'HS'])
for f, c in S['B5']['coef'].items():
    V.put(f'b5.c.{f}', c, 3, sign=True)
V.put('b5.c0', S['B5']['intercept'], 3, sign=True)
V.put('b5.last', S['B5']['last_var'], 2)
put_date(V, 'b5.lastd', S['B5']['last_date'])
put_var(V, 'b6', S['B6'], ['QR', 'QGB'])
c1 = {k.replace('+', ''): v for k, v in S['C1'].items()}
put_rv(V, 'c1', c1, ['HAR', 'HARSPX', 'RFSPX'])
c2 = S['C2']
V.put('c2.kfold', 100 * c2['kfold'], 1)
V.put('c2.wf', 100 * c2['wf'], 1)
V.put('c2.rfin', 100 * c2['rf_in'], 1)
V.put('c2.harin', 100 * c2['har_in'], 1)
V.put('c2.rfoos', 100 * N['rv']['sp500']['RF']['r2_har'], 1, sign=True)
V.raw('c2.rfp', pv(N['rv']['sp500']['RF']['dm_p']))
V.put('c2.sign', 100 * c2['sign_rf'], 1)
V.put('c2.base', 100 * c2['sign_base'], 1)
V.put('c2.qu', c2['ql_under'], 3)
V.put('c2.qo', c2['ql_over'], 3)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: when does a flexible prediction method beat a simple model on financial data, and how do we test it without fooling ourselves?',
       '\\textbf{Întrebarea}: cînd bate o metodă flexibilă de predicție un model simplu pe date financiare și cum o testăm fără să ne înșelăm singuri?'),
     [T('this seminar comes \\textbf{before} Lecture 13: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 13: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: bias and variance, a tree split, ridge and lasso in one dimension, walk-forward and purged designs, on paper',
        'Partea A: deplasare și varianță, o împărțire a unui arbore, ridge și lasso într-o dimensiune, designuri walk-forward și cu purging, pe hîrtie'),
      T('Part B: volatility forecasts, the sign of returns and VaR 1\\% on real data, each with an interpretation question',
        'Partea B: prognoze de volatilitate, semnul randamentelor și VaR 1\\% pe date reale, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('bias, variance and expected error from five training sets', 'deplasarea, varianța și eroarea așteptată din cinci eșantioane de antrenare') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('the best split of a regression tree, step by step', 'cea mai bună împărțire a unui arbore de regresie, pas cu pas') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('ridge and lasso with one feature', 'ridge și lasso cu o singură variabilă') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('a walk-forward design; a purged K-fold with an embargo', 'un design walk-forward; un K-fold cu purging și embargo') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('volatility forecasts: S\\&P 500 (next day), DAX (next week)', 'prognoze de volatilitate: S\\&P 500 (ziua următoare), DAX (săptămîna următoare)') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('the sign of daily returns: DAX; TLV and SNP', 'semnul randamentelor zilnice: DAX; TLV și SNP') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('VaR 1\\% of the DAX by quantile regression and quantile boosting', 'VaR 1\\% pentru DAX prin regresie cuantilică și quantile boosting') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('does the S\\&P 500 help to forecast BET volatility? what is wrong in an AI answer?', 'ajută S\\&P 500 la prognoza volatilității BET? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B2, A7'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Prices used}', '\\textbf{Prețurile folosite}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500 & EODHD & ' + T('open, high, low, close', 'deschidere, maxim, minim, închidere') + ' & 2008--2026',
     'DAX & EODHD & ' + T('open, high, low, close', 'deschidere, maxim, minim, închidere') + ' & 2006--2026',
     'BET & EODHD & ' + T('close', 'închidere') + ' & 2008--2026',
     T('Banca Transilvania (TLV), OMV Petrom (SNP)', 'Banca Transilvania (TLV), OMV Petrom (SNP)') + ' & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2010--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; daily variance proxy $v_t$ from open, high, low and close (Chapter 8); the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; variabila proxy zilnică a varianței $v_t$ din deschidere, maxim, minim și închidere (Capitolul 8); ultima zi este 18 septembrie 2026'),
    T('In the notebook: \\texttt{rv\\_frame(\'dax\')}, \\texttt{walk\\_forward(...)}, \\texttt{rv\\_metrics(...)}, \\texttt{sign\\_walk\\_forward(\'dax\')}, \\texttt{var\\_backtest(...)} (scikit-learn); no account or key is needed',
      'În notebook: \\texttt{rv\\_frame(\'dax\')}, \\texttt{walk\\_forward(...)}, \\texttt{rv\\_metrics(...)}, \\texttt{sign\\_walk\\_forward(\'dax\')}, \\texttt{var\\_backtest(...)} (scikit-learn); nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Learning from Data', 'Noțiuni necesare azi (1/5): învățarea din date'), items(
    (T('\\textbf{Machine learning} (ML): choose a prediction rule $\\hat f$ from data so that $\\hat f(x)$ is close to $y$ on \\textbf{new} data',
       '\\textbf{Machine learning} (ML, învățare automată): alegem din date o regulă de predicție $\\hat f$ astfel încît $\\hat f(x)$ să fie aproape de $y$ pe date \\textbf{noi}'),
     [T('$x_t$: the \\textbf{features} known at the close of day $t$; $y_t$: the \\textbf{target}; loss: e.g.\\ the squared error $(y - \\hat y)^2$, MSE its mean',
        '$x_t$: \\textbf{variabilele explicative} (features) cunoscute la închiderea zilei $t$; $y_t$: \\textbf{ținta}; pierderea: de exemplu eroarea pătratică $(y - \\hat y)^2$, MSE fiind media ei')]),
    (T('\\textbf{Overfitting}: a small error on the training data and a large error on new data', '\\textbf{Overfitting}: o eroare mică pe datele de antrenare și o eroare mare pe date noi'),
     [T('a \\textbf{hyperparameter} (tree depth, penalty) controls flexibility and is chosen on data not used for the fit', 'un \\textbf{hiperparametru} (adîncimea arborelui, penalizarea) controlează flexibilitatea și se alege pe date nefolosite la estimare')]),
    (T('\\textbf{Bias--variance}: if $y = f(x) + \\varepsilon$, $\\mathrm{Var}(\\varepsilon) = \\sigma^2$, and $\\hat f$ is fitted on a random training sample',
       '\\textbf{Deplasare--varianță}: dacă $y = f(x) + \\varepsilon$, $\\mathrm{Var}(\\varepsilon) = \\sigma^2$, iar $\\hat f$ este estimat pe un eșantion de antrenare aleator'),
     [T('$E[(y_0 - \\hat f(x_0))^2] = (E\\hat f(x_0) - f(x_0))^2 + \\mathrm{Var}(\\hat f(x_0)) + \\sigma^2$: bias$^2$ + variance + noise',
        '$E[(y_0 - \\hat f(x_0))^2] = (E\\hat f(x_0) - f(x_0))^2 + \\mathrm{Var}(\\hat f(x_0)) + \\sigma^2$: deplasare$^2$ + varianță + zgomot'),
      T('from $M$ training sets: bias $= \\bar f - f$, variance $= \\frac1M\\sum_m (\\hat f_m - \\bar f)^2$, with $\\bar f$ the mean forecast', 'din $M$ eșantioane de antrenare: deplasarea $= \\bar f - f$, varianța $= \\frac1M\\sum_m (\\hat f_m - \\bar f)^2$, cu $\\bar f$ prognoza medie')])))

D.frame(T('What You Need for Today (2/5): Validation for Time Series', 'Noțiuni necesare azi (2/5): validarea pentru serii de timp'), items(
    (T('\\textbf{Walk-forward}: train on the past, test on the next block (here: one calendar year), refit, move forward', '\\textbf{Walk-forward}: antrenăm pe trecut, testăm pe blocul următor (aici: un an calendaristic), reestimăm și avansăm'),
     [T('expanding window: the training sample keeps all earlier days', 'fereastra extinsă: eșantionul de antrenare păstrează toate zilele anterioare')]),
    (T('\\textbf{Purging}: a target $y_t$ built from days $t+1, \\dots, t+h$ overlaps the test block if $t$ is one of the last $h$ training days: drop them',
       '\\textbf{Purging}: o țintă $y_t$ construită din zilele $t+1, \\dots, t+h$ se suprapune cu blocul de test dacă $t$ este una dintre ultimele $h$ zile de antrenare: le eliminăm'),
     [T('\\textbf{embargo}: in K-fold, also drop a few training days right after the test block', '\\textbf{embargo}: la K-fold eliminăm și cîteva zile de antrenare imediat după blocul de test')]),
    (T('\\textbf{Look-ahead bias}: a feature uses information not yet known at time $t$ (future prices, full-sample means)', '\\textbf{Look-ahead bias}: o variabilă folosește informație încă necunoscută la momentul $t$ (prețuri viitoare, medii pe tot eșantionul)'),
     [T('\\textbf{leakage}: information about test targets reaches training; \\textbf{random K-fold} (shuffled folds) causes it with overlapping targets',
        '\\textbf{leakage} (scurgerea de informație): informația despre țintele de test ajunge în antrenare; \\textbf{K-fold aleator} (partiții amestecate) o produce cînd țintele se suprapun')])))

D.frame(T('What You Need for Today (3/5): Ridge and Lasso', 'Noțiuni necesare azi (3/5): ridge și lasso'), items(
    (T('\\textbf{Ridge}: minimise $\\sum_t (y_t - x_t^\\top\\beta)^2 + \\lambda\\sum_j \\beta_j^2$; \\textbf{lasso}: $\\dots + \\lambda\\sum_j |\\beta_j|$ ($\\lambda \\ge 0$: the penalty)',
       '\\textbf{Ridge}: minimizăm $\\sum_t (y_t - x_t^\\top\\beta)^2 + \\lambda\\sum_j \\beta_j^2$; \\textbf{lasso}: $\\dots + \\lambda\\sum_j |\\beta_j|$ ($\\lambda \\ge 0$: penalizarea)'),
     [T('both trade a little bias for a smaller variance; lasso sets some coefficients exactly to zero', 'ambele schimbă o deplasare mică pe o varianță mai mică; lasso anulează exact unii coeficienți')]),
    (T('\\textbf{One centred feature}, $S_{xx} = \\sum_t x_t^2$, $S_{xy} = \\sum_t x_ty_t$, no intercept', '\\textbf{O singură variabilă centrată}, $S_{xx} = \\sum_t x_t^2$, $S_{xy} = \\sum_t x_ty_t$, fără termen liber'),
     [T('OLS: $\\hat\\beta = S_{xy}/S_{xx}$; ridge: $\\hat\\beta = S_{xy}/(S_{xx} + \\lambda)$', 'OLS: $\\hat\\beta = S_{xy}/S_{xx}$; ridge: $\\hat\\beta = S_{xy}/(S_{xx} + \\lambda)$'),
      T('lasso: $\\hat\\beta = \\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$ (soft thresholding)', 'lasso: $\\hat\\beta = \\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$ (pragul moale, soft thresholding)')]),
    T('Standardise the features with the mean and standard deviation of the training sample before the penalty', 'Standardizăm variabilele cu media și abaterea standard din eșantionul de antrenare înainte de penalizare')))

D.frame(T('What You Need for Today (4/5): Trees, Ensembles and Networks', 'Noțiuni necesare azi (4/5): arbori, ansambluri și rețele'), items(
    (T('\\textbf{Regression tree}: split at $x_j \\le s$; each leaf predicts the mean of its $y$; choose $(j, s)$ to minimise', '\\textbf{Arborele de regresie}: împărțim la $x_j \\le s$; fiecare frunză prezice media valorilor $y$ din ea; alegem $(j, s)$ care minimizează'),
     [T('$\\mathrm{SSE}(s) = \\sum_{x_j \\le s}(y - \\bar y_L)^2 + \\sum_{x_j > s}(y - \\bar y_R)^2$; candidates: midpoints between sorted values', '$\\mathrm{SSE}(s) = \\sum_{x_j \\le s}(y - \\bar y_L)^2 + \\sum_{x_j > s}(y - \\bar y_R)^2$; candidați: mijloacele dintre valorile ordonate')]),
    (T('\\textbf{Random forest} (RF): the average of many deep trees, each grown on a bootstrap sample with random subsets of features', '\\textbf{Random forest} (RF): media multor arbori adînci, fiecare crescut pe un eșantion bootstrap, cu submulțimi aleatoare de variabile'),
     [T('\\textbf{gradient boosting} (GB): small trees added one by one, each fitted to the current residuals, scaled by a learning rate', '\\textbf{gradient boosting} (GB): arbori mici adăugați pe rînd, fiecare estimat pe reziduurile curente și scalat cu o rată de învățare')]),
    (T('\\textbf{MLP} (multilayer perceptron): $\\hat y = c + \\sum_k v_k\\, g(b_k + w_k^\\top x)$ with $g(z) = \\max(0, z)$ (ReLU)', '\\textbf{MLP} (multilayer perceptron, perceptron multistrat): $\\hat y = c + \\sum_k v_k\\, g(b_k + w_k^\\top x)$, cu $g(z) = \\max(0, z)$ (ReLU)'),
     [T('\\textbf{HAR} (heterogeneous autoregressive) benchmark: OLS of $\\ln$ future variance on $\\ln$ of the variance of the last day, week and month',
        'reperul \\textbf{HAR} (heterogeneous autoregressive, autoregresiv heterogen): OLS al logaritmului varianței viitoare pe logaritmul varianței din ultima zi, săptămînă și lună')])))

D.frame(T('What You Need for Today (5/5): Evaluation', 'Noțiuni necesare azi (5/5): evaluarea'), items(
    (T('Out-of-sample $R^2$ against a benchmark $\\tilde y$: $R^2_{OOS} = 1 - \\sum(y_t - \\hat y_t)^2/\\sum(y_t - \\tilde y_t)^2$; QLIKE: $L_t = \\mathrm{RV}_t/\\hat h_t + \\ln\\hat h_t$',
       '$R^2$ în afara eșantionului față de un reper $\\tilde y$: $R^2_{OOS} = 1 - \\sum(y_t - \\hat y_t)^2/\\sum(y_t - \\tilde y_t)^2$; QLIKE: $L_t = \\mathrm{RV}_t/\\hat h_t + \\ln\\hat h_t$'),
     [T('\\textbf{DM} (Diebold--Mariano) test of $d_t = L_t^A - L_t^B$: $\\bar d/\\mathrm{se}(\\bar d) \\sim N(0, 1)$; negative: $A$ is better', 'testul \\textbf{DM} (Diebold--Mariano) pentru $d_t = L_t^A - L_t^B$: $\\bar d/\\mathrm{se}(\\bar d) \\sim N(0, 1)$; negativ: $A$ este mai bun')]),
    (T('Classification: accuracy against the \\textbf{majority-class baseline} $\\mathrm{acc}_0$; $z = (\\widehat{\\mathrm{acc}} - \\mathrm{acc}_0)/\\sqrt{\\mathrm{acc}_0(1 - \\mathrm{acc}_0)/n}$',
       'Clasificare: acuratețea comparată cu \\textbf{reperul clasei majoritare} $\\mathrm{acc}_0$; $z = (\\widehat{\\mathrm{acc}} - \\mathrm{acc}_0)/\\sqrt{\\mathrm{acc}_0(1 - \\mathrm{acc}_0)/n}$'),
     [T('AUC: the area under the ROC curve (Chapter 12); 0.5 = no skill', 'AUC: aria de sub curba ROC (Capitolul 12); 0,5 = nicio abilitate')]),
    (T('\\textbf{Quantile regression}: $\\hat q_\\alpha(x) = x^\\top\\hat\\beta_\\alpha$ minimises $\\sum_t \\rho_\\alpha(y_t - x_t^\\top\\beta)$, $\\rho_\\alpha(u) = u(\\alpha - \\mathbf{1}\\{u < 0\\})$',
       '\\textbf{Regresia cuantilică}: $\\hat q_\\alpha(x) = x^\\top\\hat\\beta_\\alpha$ minimizează $\\sum_t \\rho_\\alpha(y_t - x_t^\\top\\beta)$, $\\rho_\\alpha(u) = u(\\alpha - \\mathbf{1}\\{u < 0\\})$'),
     [T('VaR 1\\% $= -\\hat q_{0.01}$, a positive loss; backtest with Kupiec (rate) and Christoffersen (independence), Chapter 10', 'VaR 1\\% $= -\\hat q_{0{,}01}$, o pierdere pozitivă; backtesting cu Kupiec (rata) și Christoffersen (independența), Capitolul 10')])), 'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Bias and Variance of a Shallow Tree', 'A1: deplasarea și varianța unui arbore puțin adînc'),
         items(T('At a point $x_0$ the true value is $f(x_0) = 2.0$ and $\\sigma^2 = 0.25$. A depth-1 tree fitted on five training sets forecasts 1.6, 1.7, 1.5, 1.8, 1.6.',
                 'Într-un punct $x_0$, valoarea reală este $f(x_0) = 2{,}0$ și $\\sigma^2 = 0{,}25$. Un arbore de adîncime 1 estimat pe cinci eșantioane de antrenare prognozează 1,6; 1,7; 1,5; 1,8; 1,6.'),
               T('1. Compute the mean forecast and the bias.', '1. Calculați prognoza medie și deplasarea.'),
               T('2. Compute the variance of the five forecasts (divide by 5).', '2. Calculați varianța celor cinci prognoze (împărțiți la 5).'),
               T('3. Compute the expected squared error at $x_0$.', '3. Calculați eroarea pătratică așteptată în $x_0$.'),
               T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
         items(T('1. $\\bar f = 8.2/5 = @{a1.mean}$; bias $= @{a1.mean} - 2 = @{a1.bias}$', '1. $\\bar f = 8{,}2/5 = @{a1.mean}$; deplasarea $= @{a1.mean} - 2 = @{a1.bias}$'),
               T('2. $[(-0.04)^2 + 0.06^2 + (-0.14)^2 + 0.16^2 + (-0.04)^2]/5 = @{a1.var}$', '2. $[(-0{,}04)^2 + 0{,}06^2 + (-0{,}14)^2 + 0{,}16^2 + (-0{,}04)^2]/5 = @{a1.var}$'),
               T('3. $@{a1.bias2} + @{a1.var} + 0.25 = @{a1.mse}$', '3. $@{a1.bias2} + @{a1.var} + 0{,}25 = @{a1.mse}$'),
               T('The shallow tree is stable but systematically too low: its error is mostly squared bias.', 'Arborele puțin adînc este stabil, dar sistematic prea jos: eroarea lui este în principal deplasare la pătrat.')),
         size='scriptsize')

D.proposed(T('A2: Bias and Variance of a Deep Tree', 'A2: deplasarea și varianța unui arbore adînc'),
           items(T('Same point and noise as in A1. A depth-10 tree on the same five training sets forecasts 2.6, 1.4, 2.3, 1.7, 2.2. Model: A1.',
                   'Același punct și același zgomot ca în A1. Un arbore de adîncime 10, pe aceleași cinci eșantioane, prognozează 2,6; 1,4; 2,3; 1,7; 2,2. Model: A1.'),
                 T('1. Compute the mean forecast, the bias and the variance.', '1. Calculați prognoza medie, deplasarea și varianța.'),
                 T('2. Compute the expected squared error.', '2. Calculați eroarea pătratică așteptată.'),
                 T('3. Decide which tree you would use at $x_0$.', '3. Decideți ce arbore ați folosi în $x_0$.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. $\\bar f = @{a2.mean}$; bias $= @{a2.bias}$; variance $= @{a2.var}$', '1. $\\bar f = @{a2.mean}$; deplasarea $= @{a2.bias}$; varianța $= @{a2.var}$'),
                 T('2. $@{a2.bias2} + @{a2.var} + 0.25 = @{a2.mse}$', '2. $@{a2.bias2} + @{a2.var} + 0{,}25 = @{a2.mse}$'),
                 T('3. The shallow tree: $@{a1.mse} < @{a2.mse}$.', '3. Arborele puțin adînc: $@{a1.mse} < @{a2.mse}$.'),
                 T('The deep tree is almost unbiased, but its variance costs more than the bias of the shallow tree.', 'Arborele adînc este aproape nedeplasat, dar varianța lui costă mai mult decît deplasarea arborelui puțin adînc.')),
           size='scriptsize')

D.frame(T('A3: One Split of a Regression Tree [Solved]', 'A3: o împărțire a unui arbore de regresie [Rezolvat]'),
        '\\begin{columns}[T]\n\\begin{column}{0.47\\textwidth}\n' + '\\begin{block}{' + T('Task', 'Cerință') + '}\n' + items(
            T('Eight days: $x$ = yesterday\'s absolute return (\\%), $y$ = today\'s variance proxy:', 'Opt zile: $x$ = randamentul absolut de ieri (\\%), $y$ = variabila proxy a varianței de azi:'),
            T('$x$: 0.2, 0.4, 0.5, 0.9, 1.1, 1.6, 2.0, 2.8; $y$: 0.5, 0.7, 0.6, 1.0, 1.4, 2.2, 2.6, 3.4', '$x$: 0,2; 0,4; 0,5; 0,9; 1,1; 1,6; 2,0; 2,8; $y$: 0,5; 0,7; 0,6; 1,0; 1,4; 2,2; 2,6; 3,4'),
            T('1. Compute the mean of $y$ and the SSE without a split.', '1. Calculați media lui $y$ și SSE fără împărțire.'),
            T('2. For each candidate split, compute both leaf means and the total SSE.', '2. Pentru fiecare împărțire candidată, calculați mediile celor două frunze și SSE total.'),
            T('3. Choose the best split and the share of the SSE it removes.', '3. Alegeți cea mai bună împărțire și ponderea din SSE pe care o elimină.'),
            T('Report: the table, the split and one sentence.', 'Raportați: tabelul, împărțirea și o frază.')) + '\n\\end{block}\n\\end{column}\n\\begin{column}{0.49\\textwidth}\n'
        + '\\begin{exampleblock}{' + T('Solution', 'Rezolvare') + '}\n' + table('rrrr', '$s$ & $\\bar y_L$ & $\\bar y_R$ & SSE', SPLITS3, size='tiny') + items(
            T('$\\bar y = @{a3.mean}$, SSE $= @{a3.sse0}$; best $s = @{a3.s}$: leaves $@{a3.l}$ and $@{a3.r}$, SSE $= @{a3.sseL} + @{a3.sseR} = @{a3.sse}$', '$\\bar y = @{a3.mean}$, SSE $= @{a3.sse0}$; cea mai bună $s = @{a3.s}$: frunzele $@{a3.l}$ și $@{a3.r}$, SSE $= @{a3.sseL} + @{a3.sseR} = @{a3.sse}$'),
            T('the split removes $1 - @{a3.sse}/@{a3.sse0} = @{a3.r2}$ of the SSE: large moves yesterday, high variance today', 'împărțirea elimină $1 - @{a3.sse}/@{a3.sse0} = @{a3.r2}$ din SSE: mișcări mari ieri, varianță mare azi')) + '\n\\end{exampleblock}\n\\end{column}\n\\end{columns}', 'scriptsize')

D.proposed(T('A4: A Split on the Volatility Index', 'A4: o împărțire după indicele de volatilitate'),
           items(T('Eight months: $x$ = the VIX level at the end of the month, $y$ = next month\'s realised volatility (\\% per day). Model: A3.', 'Opt luni: $x$ = nivelul VIX la sfîrșitul lunii, $y$ = volatilitatea realizată din luna următoare (\\% pe zi). Model: A3.'),
                 T('$x$: 12, 14, 15, 17, 21, 24, 30, 38; $y$: 0.8, 0.9, 0.7, 1.1, 1.5, 1.4, 2.6, 3.0', '$x$: 12, 14, 15, 17, 21, 24, 30, 38; $y$: 0,8; 0,9; 0,7; 1,1; 1,5; 1,4; 2,6; 3,0'),
                 T('1. Compute the SSE without a split and for every candidate split.', '1. Calculați SSE fără împărțire și pentru fiecare împărțire candidată.'),
                 T('2. Choose the best split and give both leaf forecasts.', '2. Alegeți cea mai bună împărțire și precizați prognozele celor două frunze.'),
                 T('3. Compute the share of the SSE that the split removes.', '3. Calculați ponderea din SSE eliminată de împărțire.'),
                 T('Report: the table, three numbers and one sentence.', 'Raportați: tabelul, trei valori și o frază.')),
           table('rrrr', '$s$ & $\\bar y_L$ & $\\bar y_R$ & SSE', SPLITS4, size='tiny') + items(
                 T('SSE without a split $@{a4.sse0}$; best $s = @{a4.s}$: leaves $@{a4.l}$ and $@{a4.r}$, SSE $= @{a4.sse}$, a share of $@{a4.r2}$ removed', 'SSE fără împărțire $@{a4.sse0}$; cea mai bună $s = @{a4.s}$: frunzele $@{a4.l}$ și $@{a4.r}$, SSE $= @{a4.sse}$, o pondere eliminată de $@{a4.r2}$'),
                 T('A high VIX announces a volatile month; the tree finds the threshold without a linear model.', 'Un VIX ridicat anunță o lună volatilă; arborele găsește pragul fără un model liniar.')),
           size='scriptsize')

D.solved(T('A5: Ridge with One Feature', 'A5: ridge cu o singură variabilă'),
         items(T('One centred feature with $S_{xx} = 50$ and $S_{xy} = 20$; no intercept.', 'O singură variabilă centrată, cu $S_{xx} = 50$ și $S_{xy} = 20$; fără termen liber.'),
               T('1. Compute the OLS coefficient.', '1. Calculați coeficientul OLS.'),
               T('2. Compute the ridge coefficient and the shrinkage factor $S_{xx}/(S_{xx} + \\lambda)$ for $\\lambda = 10$ and $\\lambda = 50$.', '2. Calculați coeficientul ridge și factorul de contracție $S_{xx}/(S_{xx} + \\lambda)$ pentru $\\lambda = 10$ și $\\lambda = 50$.'),
               T('3. Explain what happens when $\\lambda \\to \\infty$.', '3. Explicați ce se întîmplă cînd $\\lambda \\to \\infty$.'),
               T('Report: five numbers and one sentence.', 'Raportați: cinci valori și o frază.')),
         items(T('1. $\\hat\\beta^{OLS} = 20/50 = @{a5.ols}$', '1. $\\hat\\beta^{OLS} = 20/50 = @{a5.ols}$'),
               T('2. $\\lambda = 10$: $20/60 = @{a5.b10}$, factor $@{a5.f10}$; $\\lambda = 50$: $20/100 = @{a5.b50}$, factor $@{a5.f50}$', '2. $\\lambda = 10$: $20/60 = @{a5.b10}$, factorul $@{a5.f10}$; $\\lambda = 50$: $20/100 = @{a5.b50}$, factorul $@{a5.f50}$'),
               T('3. The coefficient tends to 0 but never equals 0.', '3. Coeficientul tinde la 0, dar nu devine niciodată 0.'),
               T('Ridge multiplies the OLS coefficient by a factor below 1: it shrinks, it does not select.', 'Ridge înmulțește coeficientul OLS cu un factor sub 1: contractă, nu selectează.')),
         size='scriptsize')

D.proposed(T('A6: Lasso with One Feature', 'A6: lasso cu o singură variabilă'),
           items(T('Same data as in A5: $S_{xx} = 50$, $S_{xy} = 20$. Model: A5.', 'Aceleași date ca în A5: $S_{xx} = 50$, $S_{xy} = 20$. Model: A5.'),
                 T('1. Compute the lasso coefficient for $\\lambda = 10$, 30 and 50.', '1. Calculați coeficientul lasso pentru $\\lambda = 10$, 30 și 50.'),
                 T('2. Find the smallest $\\lambda$ for which the coefficient is exactly zero.', '2. Găsiți cel mai mic $\\lambda$ pentru care coeficientul este exact zero.'),
                 T('3. Compare the lasso with the ridge results of A5 for $\\lambda = 50$.', '3. Comparați lasso cu rezultatele ridge din A5 pentru $\\lambda = 50$.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. $(20 - 5)/50 = @{a6.l10}$; $(20 - 15)/50 = @{a6.l30}$; $\\max(20 - 25, 0)/50 = @{a6.l50}$', '1. $(20 - 5)/50 = @{a6.l10}$; $(20 - 15)/50 = @{a6.l30}$; $\\max(20 - 25, 0)/50 = @{a6.l50}$'),
                 T('2. $\\lambda \\ge 2|S_{xy}| = @{a6.zero}$', '2. $\\lambda \\ge 2|S_{xy}| = @{a6.zero}$'),
                 T('3. For $\\lambda = 50$, ridge keeps $@{a5.b50}$, the lasso drops the feature.', '3. Pentru $\\lambda = 50$, ridge păstrează $@{a5.b50}$, iar lasso elimină variabila.'),
                 T('The lasso subtracts a constant from $|S_{xy}|$ (soft thresholding): weak features disappear.', 'Lasso scade o constantă din $|S_{xy}|$ (pragul moale): variabilele slabe dispar.')),
           size='scriptsize')

D.solved(T('A7: A Walk-Forward Design', 'A7: un design walk-forward'),
         items(T('Daily S\\&P 500 features from @{a7.first} to 18 September 2026; target: the mean variance of the next 5 days; refits each January from 2013.',
                 'Variabile zilnice S\\&P 500 de la @{a7.first} la 18 septembrie 2026; ținta: varianța medie din următoarele 5 zile; reestimare în fiecare ianuarie din 2013.'),
               T('1. Count the folds of the walk-forward evaluation.', '1. Numărați partițiile evaluării walk-forward.'),
               T('2. For the 2013 fold, say which training days must be purged and why.', '2. Pentru partiția din 2013, precizați ce zile de antrenare trebuie eliminate (purging) și de ce.'),
               T('3. Explain why no embargo is needed.', '3. Explicați de ce nu este nevoie de embargo.'),
               T('4. Explain what would go wrong if the 2013 model were tuned by random 5-fold on 2008--2012.', '4. Explicați ce ar merge prost dacă modelul pentru 2013 ar fi calibrat prin 5-fold aleator pe 2008--2012.'),
               T('Report: two numbers and three sentences.', 'Raportați: două valori și trei fraze.')),
         items(T('1. One fold per year 2013--2026: @{a7.folds} folds.', '1. O partiție pentru fiecare an 2013--2026: @{a7.folds} partiții.'),
               T('2. The last 5 training days (from @{a7.purge}): their targets use January 2013; the fold trains on @{a7.ntr} days, the last one @{a7.last}, and tests on @{a7.nte}.',
                 '2. Ultimele 5 zile de antrenare (de la @{a7.purge}): țintele lor folosesc ianuarie 2013; partiția antrenează pe @{a7.ntr} zile, ultima fiind @{a7.last}, și testează pe @{a7.nte}.'),
               T('3. All training days come before the test year: no training feature uses test information.', '3. Toate zilele de antrenare sînt înaintea anului de test: nicio variabilă de antrenare nu folosește informație de test.'),
               T('4. Random folds put neighbouring days with overlapping targets in training and validation: the chosen hyperparameters reward memorising, not forecasting.',
                 '4. Partițiile aleatoare pun zile vecine, cu ținte suprapuse, în antrenare și în validare: hiperparametrii aleși recompensează memorarea, nu prognoza.')),
         size='scriptsize')

D.proposed(T('A8: A Purged K-Fold with an Embargo', 'A8: K-fold cu purging și embargo'),
           items(T('2500 days, $K = 5$ blocks in time order, a target that uses the next 21 days, an embargo of 1\\% of the days. Model: A7.', '2500 de zile, $K = 5$ blocuri în ordine temporală, o țintă care folosește următoarele 21 de zile, un embargo de 1\\% din zile. Model: A7.'),
                 T('1. Compute the size of each test block and of the embargo.', '1. Calculați mărimea fiecărui bloc de test și a embargoului.'),
                 T('2. Compute the number of training days when the test block is in the middle.', '2. Calculați numărul de zile de antrenare cînd blocul de test este la mijloc.'),
                 T('3. Compute it when the test block is the first and when it is the last block.', '3. Calculați-l cînd blocul de test este primul și cînd este ultimul.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. Block $2500/5 = @{a8.block}$ days; embargo $\\lceil 25 \\rceil = @{a8.emb}$ days', '1. Blocul: $2500/5 = @{a8.block}$ de zile; embargoul: $\\lceil 25 \\rceil = @{a8.emb}$ de zile'),
                 T('2. $2500 - 500 - 21 - 25 = @{a8.train_inner}$ (21 purged before, 25 embargoed after)', '2. $2500 - 500 - 21 - 25 = @{a8.train_inner}$ (21 eliminate înainte, 25 sub embargo după)'),
                 T('3. First block: only the embargo, $@{a8.train_first}$; last block: only the purge, $@{a8.train_last}$', '3. Primul bloc: doar embargoul, $@{a8.train_first}$; ultimul bloc: doar purging-ul, $@{a8.train_last}$'),
                 T('Each inner fold loses @{a8.lost_inner} days; with daily data this is a small price for honest validation.', 'Fiecare partiție interioară pierde @{a8.lost_inner} de zile; cu date zilnice este un preț mic pentru o validare onestă.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Next-Day Volatility of the S\\&P 500 [Solved]', 'B1: volatilitatea S\\&P 500 pentru ziua următoare [Rezolvat]'),
       T('does a random forest forecast tomorrow\'s variance of the S\\&P 500 better than HAR?', 'prognozează un random forest varianța de mîine a S\\&P 500 mai bine decît HAR?'),
       T('S\\&P 500 open, high, low and close since 2008; target: $\\ln v_{t+1}$; features: the three HAR averages', 'deschiderea, maximul, minimul și închiderea S\\&P 500 din 2008; ținta: $\\ln v_{t+1}$; variabilele: cele trei medii HAR'),
       [T('Fit HAR (OLS) and a random forest (300 trees, at least 20 days per leaf) walk-forward, refitting each January from 2013.', 'Estimați HAR (OLS) și un random forest (300 de arbori, cel puțin 20 de zile într-o frunză) walk-forward, reestimînd în fiecare ianuarie din 2013.'),
        T('Compute the out-of-sample $R^2$ of both against the training mean and of the forest against HAR.', 'Calculați $R^2$ în afara eșantionului pentru ambele modele față de media de antrenare și pentru random forest față de HAR.'),
        T('Compute the mean QLIKE of both and the Diebold--Mariano test of equal QLIKE.', 'Calculați QLIKE mediu pentru ambele și testul Diebold--Mariano al egalității QLIKE.'),
        T('Draw the QLIKE difference per year.', 'Desenați diferența QLIKE pe ani.'),
        T('Interpretation: is the forest worth its complexity here?', 'Interpretare: merită aici complexitatea random forest?')],
       T('four numbers, a test with its p-value, the chart and two sentences', 'patru valori, un test cu p-valoarea lui, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch13_sem_b1', h='0.36') + items(
    T('@{b1.n} days from @{b1.first}; $R^2_{OOS}$ against the mean: HAR $@{b1.HAR.r2m}\\%$, forest $@{b1.RF.r2m}\\%$; forest against HAR: $@{b1.RF.r2h}\\%$',
      '@{b1.n} zile de la @{b1.first}; $R^2_{OOS}$ față de medie: HAR $@{b1.HAR.r2m}\\%$, random forest $@{b1.RF.r2m}\\%$; random forest față de HAR: $@{b1.RF.r2h}\\%$'),
    T('QLIKE: HAR $@{b1.HAR.ql}$, forest $@{b1.RF.ql}$; DM $= @{b1.RF.dm}$ (p @{b1.RF.dmp}); the forest is better in only @{b1.yrf} of @{b1.ny} years',
      'QLIKE: HAR $@{b1.HAR.ql}$, random forest $@{b1.RF.ql}$; DM $= @{b1.RF.dm}$ (p @{b1.RF.dmp}); random forest este mai bun doar în @{b1.yrf} din @{b1.ny} ani'),
    T('Interpretation: no: with the same three features the forest is significantly worse than the linear HAR; the relation is close to linear in logs and the forest adds variance',
      'Interpretare: nu: cu aceleași trei variabile, random forest este semnificativ mai slab decît HAR liniar; relația este aproape liniară în logaritmi, iar random forest adaugă varianță')) + qlsem(), 'scriptsize')

D.task(T('B2: Weekly Volatility of the DAX [Proposed]', 'B2: volatilitatea săptămînală a DAX [Propus]'),
       T('do lasso, random forest, gradient boosting or an MLP beat HAR for next week\'s DAX variance? Model: B1.', 'bat lasso, random forest, gradient boosting sau un MLP modelul HAR pentru varianța DAX din săptămîna următoare? Model: B1.'),
       T('DAX open, high, low and close since 2006; target: $\\ln$ of the mean $v$ over the next 5 days; 11 features (HAR, 4 lags, quarter, $r_t$, $\\min(r_t, 0)$, $|r_t|$)',
         'deschiderea, maximul, minimul și închiderea DAX din 2006; ținta: logaritmul mediei lui $v$ din următoarele 5 zile; 11 variabile (HAR, 4 decalaje, trimestrul, $r_t$, $\\min(r_t, 0)$, $|r_t|$)'),
       [T('Fit the five models walk-forward from 2012, purging the last 5 training days.', 'Estimați cele cinci modele walk-forward din 2012, eliminînd ultimele 5 zile de antrenare.'),
        T('Compute the out-of-sample $R^2$ against HAR and the mean QLIKE of each model.', 'Calculați $R^2$ în afara eșantionului față de HAR și QLIKE mediu pentru fiecare model.'),
        T('Run the Diebold--Mariano test of each model against HAR.', 'Aplicați testul Diebold--Mariano pentru fiecare model față de HAR.'),
        T('Interpretation: which model would you use for the DAX?', 'Interpretare: ce model ați folosi pentru DAX?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(m):
    key = f'b2.{m}'
    if m == 'HAR':
        return f'HAR & $@{{{key}.r2m}}$ & -- & $@{{{key}.ql}}$ & --'
    return f'{m} & $@{{{key}.r2m}}$ & ${{@{{{key}.r2h}}}}$ & $@{{{key}.ql}}$ & ${{@{{{key}.dm}}}}$ (@{{{key}.dmp}})'


D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrr', T('Model & $R^2_{OOS}$ vs.\\ mean & $R^2_{OOS}$ vs.\\ HAR & QLIKE & DM (p)', 'Model & $R^2_{OOS}$ față de medie & $R^2_{OOS}$ față de HAR & QLIKE & DM (p)'),
    [b2row(m) for m in ['HAR', 'Lasso', 'RF', 'GB', 'MLP']], size='footnotesize') + items(
    T('@{b2.n} days from @{b2.first}; $R^2_{OOS}$ in \\%; DM against HAR, negative: better than HAR', '@{b2.n} zile de la @{b2.first}; $R^2_{OOS}$ în \\%; DM față de HAR, negativ: mai bun decît HAR'),
    T('Interpretation: the lasso ($@{b2.Lasso.r2h}\\%$, p @{b2.Lasso.dmp}) or the MLP ($@{b2.MLP.r2h}\\%$, p @{b2.MLP.dmp}); the lasso is simpler and its gain is as large; trees do not beat HAR significantly',
      'Interpretare: lasso ($@{b2.Lasso.r2h}\\%$, p @{b2.Lasso.dmp}) sau MLP ($@{b2.MLP.r2h}\\%$, p @{b2.MLP.dmp}); lasso este mai simplu, iar cîștigul lui este la fel de mare; arborii nu bat semnificativ HAR')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: The Sign of DAX Returns [Solved]', 'B3: semnul randamentelor DAX [Rezolvat]'),
       T('can a logit or a random forest predict whether the DAX rises tomorrow better than ``always up\'\'?', 'pot un logit sau un random forest să prezică dacă DAX crește mîine mai bine decît „mereu creștere”?'),
       T('DAX closes since 2000; target $\\mathbf{1}\\{r_{t+1} > 0\\}$; features: the last 5 returns, sums over 5, 21 and 63 days, volatility over 21 and 63 days',
         'închiderile DAX din 2000; ținta $\\mathbf{1}\\{r_{t+1} > 0\\}$; variabilele: ultimele 5 randamente, sumele pe 5, 21 și 63 de zile, volatilitatea pe 21 și 63 de zile'),
       [T('Fit both classifiers walk-forward from 2008, refitting each January.', 'Estimați ambii clasificatori walk-forward din 2008, reestimînd în fiecare ianuarie.'),
        T('Compute the accuracy of each model and of the majority-class baseline.', 'Calculați acuratețea fiecărui model și a reperului clasei majoritare.'),
        T('Test each accuracy against the baseline with the $z$ statistic, and compute the AUC.', 'Testați fiecare acuratețe față de reper cu statistica $z$ și calculați AUC.'),
        T('Draw the accuracy by year.', 'Desenați acuratețea pe ani.'),
        T('Interpretation: does either model have skill?', 'Interpretare: are vreunul dintre modele o abilitate reală?')],
       T('six numbers, the chart and two sentences', 'șase valori, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch13_sem_b3', h='0.36') + items(
    T('$n = @{b3.n}$ days; up days @{b3.up}\\% = the baseline accuracy; logit @{b3.Logit.acc}\\% ($z = @{b3.Logit.z}$, p @{b3.Logit.p}, AUC @{b3.Logit.auc}); forest @{b3.RF.acc}\\% ($z = @{b3.RF.z}$, p @{b3.RF.p}, AUC @{b3.RF.auc})',
      '$n = @{b3.n}$ de zile; zile de creștere @{b3.up}\\% = acuratețea reperului; logit @{b3.Logit.acc}\\% ($z = @{b3.Logit.z}$, p @{b3.Logit.p}, AUC @{b3.Logit.auc}); random forest @{b3.RF.acc}\\% ($z = @{b3.RF.z}$, p @{b3.RF.p}, AUC @{b3.RF.auc})'),
    T('The logit forecasts ``up\'\' on @{b3.Logit.upred}\\% of the days; the forest beats the baseline in @{b3.yabove} of @{b3.ny} years, as often as chance would allow',
      'Logit prognozează „creștere” în @{b3.Logit.upred}\\% din zile; random forest bate reperul în @{b3.yabove} din @{b3.ny} ani, cam cît ar permite hazardul'),
    T('Interpretation: no: both accuracies are at or below the baseline, and the AUC is close to 0.5; daily DAX returns are close to unpredictable (Chapter 7)',
      'Interpretare: nu: ambele acurateți sînt la nivelul reperului sau sub el, iar AUC este aproape de 0,5; randamentele zilnice ale DAX sînt aproape imprevizibile (Capitolul 7)')) + qlsem(), 'scriptsize')

D.task(T('B4: The Sign of Two BVB Stocks [Proposed]', 'B4: semnul a două acțiuni BVB [Propus]'),
       T('is the sign of Banca Transilvania and OMV Petrom returns easier to predict than that of the DAX? Model: B3.', 'este semnul randamentelor Banca Transilvania și OMV Petrom mai ușor de prezis decît cel al DAX? Model: B3.'),
       T('TLV (without 30--31 May 2016) and SNP adjusted closes since 2010; the features of B3; forecasts from 2014', 'închiderile ajustate TLV (fără 30--31 mai 2016) și SNP din 2010; variabilele din B3; prognoze din 2014'),
       [T('Fit the logit, random forest and gradient boosting walk-forward from 2014.', 'Estimați logit, random forest și gradient boosting walk-forward din 2014.'),
        T('Compute the accuracy, the baseline accuracy, the $z$ statistic and the AUC for each stock and model.', 'Calculați acuratețea, acuratețea reperului, statistica $z$ și AUC pentru fiecare acțiune și model.'),
        T('Interpretation: is there more predictability on the BVB than on the DAX?', 'Interpretare: există mai multă predictibilitate la BVB decît la DAX?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(k, m):
    key = f'b4.{k}.{m}'
    lab = {'tlv': 'TLV', 'snp': 'SNP'}[k] if m == 'Logit' else ''
    base = f'@{{b4.{k}.base}}' if m == 'Logit' else ''
    return f'{lab} & {base} & {m} & @{{{key}.acc}} & ${{@{{{key}.z}}}}$ (@{{{key}.p}}) & @{{{key}.auc}}'


D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrlrrr', T('& baseline (\\%) & Model & accuracy (\\%) & $z$ (p) & AUC', '& reper (\\%) & Model & acuratețe (\\%) & $z$ (p) & AUC'),
    [b4row('tlv', m) for m in ['Logit', 'RF', 'GB']] + ['\\midrule'] + [b4row('snp', m) for m in ['Logit', 'RF', 'GB']], size='footnotesize').replace('\\midrule \\\\', '\\midrule') + items(
    T('TLV: @{b4.tlv.n} days, SNP: @{b4.snp.n} days, from 2014', 'TLV: @{b4.tlv.n} zile, SNP: @{b4.snp.n} zile, din 2014'),
    T('Interpretation: no: no difference is significant at 5\\%; SNP models are slightly above the baseline (forest $@{b4.snp.RF.diff}$ points, p @{b4.snp.RF.p}), TLV models below it; AUC between @{b4.tlv.Logit.auc} and @{b4.snp.GB.auc}',
      'Interpretare: nu: nicio diferență nu este semnificativă la 5\\%; modelele pentru SNP sînt puțin peste reper (random forest $@{b4.snp.RF.diff}$ puncte, p @{b4.snp.RF.p}), cele pentru TLV sub el; AUC între @{b4.tlv.Logit.auc} și @{b4.snp.GB.auc}')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: VaR 1\\% of the DAX by Quantile Regression [Solved]', 'B5: VaR 1\\% pentru DAX prin regresie cuantilică [Rezolvat]'),
       T('does a linear quantile regression on volatility features give a better VaR 1\\% than historical simulation?', 'dă o regresie cuantilică liniară pe variabile de volatilitate un VaR 1\\% mai bun decît simularea istorică?'),
       T('DAX since 2006; target $r_{t+1}$; features: $\\sqrt{v_t}$, $\\sqrt{\\text{mean } v}$ over 5 and 22 days, $r_t$', 'DAX din 2006; ținta $r_{t+1}$; variabilele: $\\sqrt{v_t}$, $\\sqrt{\\text{media lui } v}$ pe 5 și 22 de zile, $r_t$'),
       [T('Fit the 1\\% quantile regression walk-forward from 2012 and set VaR 1\\% $= -\\hat q_{0.01}$.', 'Estimați regresia cuantilică de nivel 1\\% walk-forward din 2012 și luați VaR 1\\% $= -\\hat q_{0{,}01}$.'),
        T('Compute the historical-simulation VaR 1\\% on the last 500 days.', 'Calculați VaR 1\\% prin simulare istorică pe ultimele 500 de zile.'),
        T('Count the exceptions and run the Kupiec and Christoffersen tests; compute the average quantile loss.', 'Numărați depășirile, aplicați testele Kupiec și Christoffersen și calculați pierderea cuantilică medie.'),
        T('Draw both VaR forecasts in 2020--2022.', 'Desenați ambele prognoze VaR în 2020--2022.'),
        T('Interpretation: which VaR would you report to a risk committee?', 'Interpretare: ce VaR ați raporta unui comitet de risc?')],
       T('a table, the chart and two sentences', 'un tabel, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch13_sem_b5', h='0.32') + table(
    'lrrrrr', T('Method & $x$ & rate (\\%) & p Kupiec & p ind. & loss $\\times 100$', 'Metoda & $x$ & rata (\\%) & p Kupiec & p ind. & pierdere $\\times 100$'),
    ['QR & @{b5.QR.x} & $@{b5.QR.rate}$ & @{b5.QR.puc} & @{b5.QR.pind} & $@{b5.QR.ql}$', 'HS & @{b5.HS.x} & $@{b5.HS.rate}$ & @{b5.HS.puc} & @{b5.HS.pind} & $@{b5.HS.ql}$'], size='scriptsize') + items(
    T('@{b5.n} days from @{b5.first}; last fit: $\\hat q_{0.01} = @{b5.c0} @{b5.c.sd}\\sqrt{v_t} @{b5.c.sw}\\sqrt{v^{(w)}} @{b5.c.sm}\\sqrt{v^{(m)}} @{b5.c.r} r_t$; VaR 1\\% on @{b5.lastd}: @{b5.last}\\%',
      '@{b5.n} zile de la @{b5.first}; ultima estimare: $\\hat q_{0{,}01} = @{b5.c0} @{b5.c.sd}\\sqrt{v_t} @{b5.c.sw}\\sqrt{v^{(w)}} @{b5.c.sm}\\sqrt{v^{(m)}} @{b5.c.r} r_t$; VaR 1\\% pe @{b5.lastd}: @{b5.last}\\%'),
    T('Interpretation: the quantile regression: right rate, independent exceptions, smaller loss; HS is exceeded in clusters (@{b5.HS.x20} times in 2020) because a 500-day window reacts slowly',
      'Interpretare: regresia cuantilică: rată corectă, depășiri independente, pierdere mai mică; HS este depășit grupat (de @{b5.HS.x20} ori în 2020), deoarece o fereastră de 500 de zile reacționează lent')) + qlsem(), 'scriptsize')

D.task(T('B6: Quantile Gradient Boosting for the DAX [Proposed]', 'B6: quantile gradient boosting pentru DAX [Propus]'),
       T('does a flexible quantile model improve on the linear quantile regression of B5? Model: B5.', 'îmbunătățește un model cuantilic flexibil regresia cuantilică liniară din B5? Model: B5.'),
       T('as in B5; gradient boosting with the quantile loss at 1\\%, 300 trees of depth 2, learning rate 0.05, at least 50 days per leaf', 'ca în B5; gradient boosting cu pierderea cuantilică la 1\\%, 300 de arbori de adîncime 2, rata de învățare 0,05, cel puțin 50 de zile într-o frunză'),
       [T('Fit the quantile boosting walk-forward from 2012 and compute its VaR 1\\%.', 'Estimați quantile boosting walk-forward din 2012 și calculați VaR 1\\%.'),
        T('Count the exceptions, run the Kupiec and Christoffersen tests and compute the quantile loss.', 'Numărați depășirile, aplicați testele Kupiec și Christoffersen și calculați pierderea cuantilică.'),
        T('Interpretation: is the flexible model worth it at the 1\\% level?', 'Interpretare: merită modelul flexibil la nivelul de 1\\%?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrr', T('Method & $x$ & rate (\\%) & p Kupiec & p ind. & loss $\\times 100$ & mean VaR', 'Metoda & $x$ & rata (\\%) & p Kupiec & p ind. & pierdere $\\times 100$ & VaR mediu'),
    [f'{m} & @{{b6.{m}.x}} & $@{{b6.{m}.rate}}$ & @{{b6.{m}.puc}} & @{{b6.{m}.pind}} & $@{{b6.{m}.ql}}$ & $@{{b6.{m}.mv}}$' for m in ('QR', 'QGB')], size='footnotesize') + items(
    T('@{b6.n} days; QGB: quantile gradient boosting', '@{b6.n} zile; QGB: quantile gradient boosting'),
    T('Interpretation: no: boosting is exceeded far too often (@{b6.QGB.rate}\\%, p @{b6.QGB.puc}) and its loss is larger; at 1\\% each leaf sees very few tail days, so the flexible model underestimates the risk',
      'Interpretare: nu: boosting este depășit mult prea des (@{b6.QGB.rate}\\%, p @{b6.QGB.puc}), iar pierderea lui este mai mare; la 1\\% fiecare frunză vede foarte puține zile din coadă, deci modelul flexibil subestimează riscul')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Does the S\\&P 500 Help to Forecast BET Volatility? [Proposed]', 'C1: ajută S\\&P 500 la prognoza volatilității BET? [Propus]'),
       T('the BVB closes before New York; does yesterday\'s S\\&P 500 volatility improve a HAR forecast of next week\'s BET variance?',
         'BVB se închide înaintea bursei din New York; îmbunătățește volatilitatea S\\&P 500 de ieri o prognoză HAR a varianței BET din săptămîna următoare?'),
       T('BET and S\\&P 500 closes since 2008; proxy: squared daily returns (the BET has no high and low prices in our data); models: B1, B2',
         'închiderile BET și S\\&P 500 din 2008; proxy: randamentele zilnice la pătrat (BET nu are prețuri maxime și minime în datele noastre); modele: B1, B2'),
       [T('Build the HAR features of the BET and the HAR features of the S\\&P 500 at the previous New York close.', 'Construiți variabilele HAR ale BET și variabilele HAR ale S\\&P 500 de la închiderea precedentă din New York.'),
        T('Compare HAR, HAR plus the S\\&P 500 features (OLS) and a random forest on all features, walk-forward from 2012.', 'Comparați HAR, HAR plus variabilele S\\&P 500 (OLS) și un random forest pe toate variabilele, walk-forward din 2012.'),
        T('Compute the out-of-sample $R^2$, QLIKE and the Diebold--Mariano tests against HAR.', 'Calculați $R^2$ în afara eșantionului, QLIKE și testele Diebold--Mariano față de HAR.'),
        T('Interpretation: is the S\\&P 500 information worth adding?', 'Interpretare: merită adăugată informația din S\\&P 500?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')


def c1row(m):
    key = f'c1.{m.replace("+", "")}'
    if m == 'HAR':
        return f'HAR & $@{{{key}.r2m}}$ & -- & $@{{{key}.ql}}$ & --'
    return f'{m} & $@{{{key}.r2m}}$ & ${{@{{{key}.r2h}}}}$ & $@{{{key}.ql}}$ & ${{@{{{key}.dm}}}}$ (@{{{key}.dmp}})'


D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrr', T('Model & $R^2_{OOS}$ vs.\\ mean & $R^2_{OOS}$ vs.\\ HAR & QLIKE & DM (p)', 'Model & $R^2_{OOS}$ față de medie & $R^2_{OOS}$ față de HAR & QLIKE & DM (p)'),
    [c1row(m) for m in ['HAR', 'HAR+SPX', 'RF+SPX']], size='footnotesize') + items(
    T('@{c1.n} days from @{c1.first}; SPX: the S\\&P 500 features of the previous New York close', '@{c1.n} zile de la @{c1.first}; SPX: variabilele S\\&P 500 de la închiderea precedentă din New York'),
    T('The S\\&P 500 adds $@{c1.HARSPX.r2h}\\%$ to HAR and lowers QLIKE, but not significantly (p @{c1.HARSPX.dmp}); the forest is worse than the linear model',
      'S\\&P 500 adaugă $@{c1.HARSPX.r2h}\\%$ peste HAR și scade QLIKE, dar nu semnificativ (p @{c1.HARSPX.dmp}); random forest este mai slab decît modelul liniar'),
    T('Project design: daily and monthly horizons, the DAX as a second external market, crisis periods separately, a range proxy for BVB stocks with high and low prices',
      'Designul proiectului: orizonturi zilnice și lunare, DAX ca a doua piață externă, perioadele de criză separat, un proxy de amplitudine pentru acțiunile BVB cu prețuri maxime și minime')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise machine learning for the S\\&P 500. The answer:', 'Un student a cerut unui asistent AI să rezume aplicarea machine learning la S\\&P 500. Răspunsul:'),
    T('\\aiprompt{(a) With random 5-fold cross-validation a random forest explains @{c2.kfold}\\% of the next 21-day return: the market is predictable.}', '\\aiprompt{(a) Cu validare încrucișată 5-fold aleatoare, un random forest explică @{c2.kfold}\\% din randamentul următoarelor 21 de zile: piața este previzibilă.}'),
    T('\\aiprompt{(b) The forest fits next week\'s log variance better than HAR: in-sample R2 @{c2.rfin}\\% against @{c2.harin}\\%.}', '\\aiprompt{(b) Random forest explică logaritmul varianței din săptămîna următoare mai bine decît HAR: R2 în eșantion @{c2.rfin}\\% față de @{c2.harin}\\%.}'),
    T('\\aiprompt{(c) A forest predicts the sign of tomorrow\'s return correctly on @{c2.sign}\\% of the days, more than a coin: it has skill.}', '\\aiprompt{(c) Un random forest prezice corect semnul randamentului de mîine în @{c2.sign}\\% din zile, mai mult decît o monedă: are o abilitate reală.}'),
    T('\\aiprompt{(d) For the VaR 99\\% use quantile regression at tau = 0.99 on the next-day return.}', '\\aiprompt{(d) Pentru VaR 99\\% folosiți regresia cuantilică cu tau = 0,99 pe randamentul zilei următoare.}'),
    T('\\aiprompt{(e) The lasso removes the features whose t-statistics are not significant.}', '\\aiprompt{(e) Lasso elimină variabilele ale căror statistici t nu sînt semnificative.}'),
    T('\\aiprompt{(f) QLIKE punishes an under-forecast of the variance more than an over-forecast by the same factor.}', '\\aiprompt{(f) QLIKE penalizează o subestimare a varianței mai mult decît o supraestimare cu același factor.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: overlapping targets leak through shuffled folds; walk-forward gives $R^2 = @{c2.wf}\\%$, worse than the training mean', '(a) Greșit: țintele suprapuse produc scurgeri de informație prin partițiile amestecate; walk-forward dă $R^2 = @{c2.wf}\\%$, mai slab decît media de antrenare'),
    T('(b) Wrong as evidence: an in-sample $R^2$ rewards flexibility; out of sample the forest gains only $@{c2.rfoos}\\%$ over HAR (DM p @{c2.rfp})', '(b) Greșit ca argument: $R^2$ în eșantion recompensează flexibilitatea; în afara eșantionului, random forest cîștigă doar $@{c2.rfoos}\\%$ față de HAR (DM p @{c2.rfp})'),
    T('(c) Wrong: the baseline is ``always up\'\', right on @{c2.base}\\% of the days, not a coin; the forest is below it', '(c) Greșit: reperul este „mereu creștere”, corect în @{c2.base}\\% din zile, nu o monedă; random forest este sub el'),
    T('(d) Wrong twice: the level is the tail probability, VaR 1\\%, and it needs $\\tau = 0.01$ (the lower tail); VaR 1\\% $= -\\hat q_{0.01}$, a positive loss', '(d) Greșit de două ori: nivelul este probabilitatea cozii, VaR 1\\%, și are nevoie de $\\tau = 0{,}01$ (coada inferioară); VaR 1\\% $= -\\hat q_{0{,}01}$, o pierdere pozitivă'),
    T('(e) Wrong: the lasso shrinks by a penalty; a zero coefficient is not a test, and the lasso gives no p-values', '(e) Greșit: lasso contractă prin penalizare; un coeficient zero nu este un test, iar lasso nu dă p-valori'),
    T('(f) Correct: for a variance of 1, a forecast of 0.5 costs $@{c2.qu}$ and a forecast of 2 costs $@{c2.qo}$', '(f) Corect: pentru o varianță de 1, o prognoză de 0,5 costă $@{c2.qu}$, iar o prognoză de 2 costă $@{c2.qo}$')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('The expected error is bias$^2$ + variance + noise: flexibility is not free', 'Eroarea așteptată este deplasare$^2$ + varianță + zgomot: flexibilitatea are un cost'),
    T('Time series: walk-forward with purging; never shuffle days with overlapping targets', 'Serii de timp: walk-forward cu purging; nu amestecați niciodată zilele cu ținte suprapuse'),
    T('Volatility: linear and penalised models are hard to beat; trees on the HAR features alone lose', 'Volatilitatea: modelele liniare și penalizate sînt greu de bătut; arborii doar pe variabilele HAR pierd'),
    T('The sign of daily returns: compare with ``always up\'\'; no model beats it here', 'Semnul randamentelor zilnice: comparați cu „mereu creștere”; aici niciun model nu îl bate'),
    T('VaR 1\\%: a linear quantile regression passes the backtests; a flexible quantile model at 1\\% does not', 'VaR 1\\%: o regresie cuantilică liniară trece testele; un model cuantilic flexibil la 1\\% nu le trece'),
    T('An AI answer is a draft: check the validation scheme, the baseline and the VaR convention', 'Un răspuns AI este o ciornă: verificați schema de validare, reperul și convenția VaR')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 13 develops each topic of today: bias and variance, validation, ridge and lasso, trees and ensembles, neural networks, the three applications and data snooping',
      'Cursul 13 dezvoltă fiecare temă de azi: deplasarea și varianța, validarea, ridge și lasso, arborii și ansamblurile, rețelele neuronale, cele trei aplicații și data snooping'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: BVB stocks, external markets, several horizons', 'C1 poate deveni un proiect de echipă: acțiuni BVB, piețe externe, mai multe orizonturi'),
    T('Reading: \\refISL, Ch.~2, 5, 6 and 8; \\refFHH, Ch.~19', 'Lectură: \\refISL, cap.~2, 5, 6 și 8; \\refFHH, cap.~19')))

D.references(bib(['Corsi', 'DM', 'FHH', 'Friedman', 'HK', 'ISL', 'KB', 'Kupiec', 'Chris', 'LdP', 'Tib']), per=16)

if __name__ == '__main__':
    D.write(V)
