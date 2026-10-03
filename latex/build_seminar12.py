r"""
build_seminar12.py -- Seminarul 12 (Modele de scoring), EN + RO dintr-o singură sursă
====================================================================================
Seminarul are loc ÎNAINTEA cursului 12: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_12/sem12_results.json (seminar12.py).
Ieșire:
  EN/Seminars/seminar12_scoring_models.tex          (+ _solutions.tex)
  RO/Seminarii/seminar12_modele_scoring_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_12/seminar12.py && python3 latex/build_seminar12.py && python3 latex/sfm_build.py compile 12
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch12_common import REFS, T, bib, load_sem, pv, vname   # noqa: E402

S = load_sem()
V = Values()
D = Deck(12, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_12}{SFM_ch12_seminar}'


def money(x):
    return f'{x:,.0f}'.replace(',', '\\,')


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for key in ('a1', 'a2'):
    a = A[key]
    V.put(f'{key}.eta', a['eta'], 3)
    V.put(f'{key}.odds', a['odds'], 3)
    V.put(f'{key}.pd', 100 * a['pd'], 1)
    V.put(f'{key}.or', a['or_noacc'], 2)
    V.put(f'{key}.ord', a['or_dur'], 2)
    V.put(f'{key}.p1', 100 * a['pd_plus1y'], 1)
    V.put(f'{key}.me', 100 * a['marginal'], 1)
    V.put(f'{key}.dp', 100 * (a['pd_plus1y'] - a['pd']), 1)
for key in ('a3', 'a4'):
    a = A[key]
    for i, c in enumerate(a['codes']):
        V.raw(f'{key}.g{c}', str(a['good'][i]))
        V.raw(f'{key}.b{c}', str(a['bad'][i]))
        V.put(f'{key}.dg{c}', a['dist_good'][i], 3)
        V.put(f'{key}.db{c}', a['dist_bad'][i], 3)
        V.put(f'{key}.w{c}', a['woe'][i], 3)
        V.put(f'{key}.iv{c}', a['iv_part'][i], 3)
        V.put(f'{key}.r{c}', 100 * a['bad_rate'][i], 1)
    V.put(f'{key}.iv', a['iv'], 3)
for key in ('a5', 'a6'):
    a = A[key]
    V.raw(f'{key}.wins', ' + '.join(str(w) for w in a['wins']))
    V.raw(f'{key}.sumw', str(sum(a['wins'])))
    V.raw(f'{key}.pairs', str(a['pairs']))
    V.raw(f'{key}.wrong', str(a['pairs'] - sum(a['wins'])))
    V.put(f'{key}.auc', a['auc'], 3)
    V.put(f'{key}.gini', a['gini'], 3)
    for k in ('TP', 'FN', 'FP', 'TN'):
        V.raw(f'{key}.{k}', str(a['cm'][k]))
    for k in ('acc', 'tpr', 'fpr', 'precision'):
        V.put(f'{key}.{k}', 100 * a['cm'][k], 1)
    V.put(f'{key}.ks', a['ks'], 3)
    V.put(f'{key}.ksat', a['ks_at'], 2)
    V.put(f'{key}.brier', a['brier'], 3)
    V.put(f'{key}.bsum', sum(a['brier_terms']), 4)
a7 = A['a7']
for k in ('factor', 'offset', 'score', 'ln_odds', 'odds', 'ln_odds_pd'):
    V.put(f'a7.{k}', a7[k], 3 if k in ('ln_odds', 'ln_odds_pd') else 2)
V.put('a7.pd', 100 * a7['pd_of_score'], 1)
p7 = A['a7pts']
V.put('a7.b0', p7['b0'], 3)
V.put('a7.b', p7['b'], 3)
for c in ('1', '4'):
    V.put(f'a7.w{c}', p7['woe'][c], 3)
    V.put(f'a7.pt{c}', p7['points'][c], 1)
V.put('a7.ok', p7['offset'] / p7['k'], 2)
V.put('a7.b0k', p7['b0'] / p7['k'], 4)
V.put('a7.dpt', p7['points']['4'] - p7['points']['1'], 1)
V.put('a7.dbl', (p7['points']['4'] - p7['points']['1']) / 20, 1)
a8 = A['a8']
for k in ('factor', 'offset', 'score', 'ln_odds', 'odds', 'ln_odds_pd'):
    V.put(f'a8.{k}', a8[k], 3 if k in ('ln_odds', 'ln_odds_pd') else 2)
V.put('a8.pd', 100 * a8['pd_of_score'], 1)
e8 = A['a8el']
V.put('a8.shift', e8['shift'], 3)
V.put('a8.lns', e8['ln_odds_sample'], 3)
V.put('a8.lnp', e8['ln_odds_pop'], 3)
V.put('a8.pdpop', 100 * e8['pd_pop'], 1)
V.raw('a8.el1', money(e8['el_sample']))
V.raw('a8.el2', money(e8['el_pop']))
B1 = S['B1']
V.put('b1.auc', B1['auc'], 3)
V.put('b1.auctr', B1['auc_train'], 3)
V.put('b1.lo', B1['lo'], 3)
V.put('b1.hi', B1['hi'], 3)
V.put('b1.se', B1['se_auc'], 3)
V.raw('b1.nsel', str(len(B1['sel'])))
for i, v in enumerate(['const'] + B1['sel']):
    V.put(f'b1.{v}.b', B1['b'][i], 3)
    V.put(f'b1.{v}.se', B1['se'][i], 3)
for v, x in B1['iv'].items():
    V.put(f'b1.iv.{v}', x, 3)
for i, r in enumerate(B1['rate_ch']):
    V.put(f'b1.ch{i}', 100 * r, 0)
    V.put(f'b1.chw{i}', B1['woe_ch'][i], 2)
B2 = S['B2']
for v, x in B2['iv'].items():
    V.put(f'b2.iv.{v}', x, 3)
for c, d in B2['pay0'].items():
    V.put(f'b2.p{c.replace("-", "m")}', 100 * d['rate'], 1)
B3 = S['B3']
for cm in ('cm50', 'cm_cost'):
    for k in ('TP', 'FN', 'FP', 'TN'):
        V.raw(f'b3.{cm}.{k}', str(B3[cm][k]))
    for k in ('acc', 'tpr', 'fpr', 'precision', 'reject'):
        V.put(f'b3.{cm}.{k}', 100 * B3[cm][k], 1)
    V.put(f'b3.{cm}.cost', B3[cm]['cost'], 3)
for k in ('auc', 'gini', 'ar', 'ks', 'brier', 'brier_ref', 'hl', 'hl_p'):
    V.put(f'b3.{k}', B3[k], 3)
B4 = S['B4']
for k in ('auc_logit', 'auc_lda', 'brier_logit', 'brier_lda', 'corr'):
    V.put(f'b4.{k}', B4[k], 3)
V.put('b4.z', B4['z'], 2)
V.put('b4.p', B4['p'], 3)
V.put('b4.gain', 100 * (B4['cm_logit']['acc'] - B4['acc_nobody']), 1)
V.put('b4.hll', B4['hl_logit'], 1)
V.put('b4.hld', B4['hl_lda'], 1)
for k in ('mean_pd_logit', 'mean_pd_lda', 'rate', 'acc_nobody'):
    V.put(f'b4.{k}', 100 * B4[k], 1)
for m in ('logit', 'lda'):
    V.put(f'b4.{m}.acc', 100 * B4[f'cm_{m}']['acc'], 1)
    V.put(f'b4.{m}.tpr', 100 * B4[f'cm_{m}']['tpr'], 1)
B5 = S['B5']
V.put('b5.shift', B5['shift'], 3)
for k in ('mean_pd_sample', 'mean_pd_pop', 'median_pd_pop', 'max_pd_pop', 'el_pct', 'k_pct', 'rwa_density', 'el_naive_pct'):
    V.put(f'b5.{k}', 100 * B5[k], 1)
for k in ('ead', 'el', 'k', 'rwa'):
    V.raw(f'b5.{k}', money(B5[k]))
B6 = S['B6']
V.put('b6.cut', 100 * B6['cut'], 1)
for g, kk in [('men', 'm'), ('women', 'w'), ('age <= 25', 'y'), ('age > 25', 'o')]:
    for c in ('approve', 'tpr_good', 'bad_approved', 'rate'):
        V.put(f'b6.{kk}.{c}', 100 * B6[g][c], 1)
    V.raw(f'b6.{kk}.n', money(B6[g]['n']))
for c in ('dp_sex', 'eo_sex', 'dp_age', 'eo_age'):
    V.put(f'b6.{c}', 100 * B6[c], 1)
C1 = S['C1']
for k in ('auc_gb', 'auc_logit', 'brier_gb', 'brier_logit', 'auc_gb_train'):
    V.put(f'c1.{k}', C1[k], 3)
V.put('c1.z', C1['z'], 2)
V.put('c1.p', C1['p'], 3)
C2 = S['C2']
V.put('c2.auc', C2['auc'], 2)
V.put('c2.gini', C2['gini'], 3)
V.put('c2.wrong', C2['auc'] - 0.5, 2)
V.put('c2.acc', 100 * C2['acc'], 1)
V.put('c2.s10', C2['score_pd10'], 1)
V.put('c2.s20', C2['score_pd20'], 1)
V.put('c2.ds', C2['score_pd10'] - C2['score_pd20'], 1)
V.put('c2.pdpop', 100 * C2['pd_example_pop'], 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how does a bank turn the data of an applicant into a probability of default, and how do we judge that probability?',
       '\\textbf{Întrebarea}: cum transformă o bancă datele unui solicitant într-o probabilitate de nerambursare și cum judecăm această probabilitate?'),
     [T('this seminar comes \\textbf{before} Lecture 12: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 12: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: odds and log-odds, WoE and IV, the confusion matrix and AUC, scorecard points, on paper',
        'Partea A: șansa și logaritmul șansei, WoE și IV, matricea de confuzie și AUC, punctele unui scorecard, pe hîrtie'),
      T('Part B: a scorecard on real credit data, its validation, logit against LDA, capital and fairness, each with an interpretation question',
        'Partea B: un scorecard pe date reale de credit, validarea lui, logit față de LDA, capitalul și echitatea, fiecare cu o întrebare de interpretare'),
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
    ['A1, A2 & ' + T('the PD of an applicant from a logit model; odds ratios', 'PD a unui solicitant dintr-un model logit; rapoartele șanselor') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('WoE and IV from a table of goods and bads', 'WoE și IV dintr-un tabel de bun-platnici și rău-platnici') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('confusion matrix, AUC by counting pairs, Gini, KS, Brier score', 'matricea de confuzie, AUC prin numărarea perechilor, Gini, KS, scorul Brier') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('scorecard scaling and points; prior correction and expected loss', 'scalarea scorecard-ului și punctele; corecția a priori și pierderea așteptată') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('IV and a WoE logit: South German Credit; IV of the Taiwan data', 'IV și un logit WoE: South German Credit; IV pentru datele din Taiwan') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('validation of the scorecard; logit against LDA on the Taiwan data', 'validarea scorecard-ului; logit față de LDA pe datele din Taiwan') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('expected loss and IRB capital of a portfolio; fairness by sex and age', 'pierderea așteptată și capitalul IRB ale unui portofoliu; echitatea pe sexe și vîrste') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('does gradient boosting beat the logit? what is wrong in an AI answer?', 'învinge gradient boosting logit-ul? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3, A5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    TB + 'p{2.6cm}' + TB + 'p{3.4cm}' + TB + 'p{2.6cm}' + TB + 'p{2.8cm}', T('\\textbf{Data set}', '\\textbf{Setul de date}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Size}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Outcome}', '\\textbf{Rezultatul}'),
    ['South German Credit & \\refSGC & ' + T('1000 credits, 1973--1975', '1000 de credite, 1973--1975') + ' & ' + T('300 bad, 700 good', '300 rău-platnici, 700 bun-platnici'),
     T('Taiwan credit cards', 'Carduri de credit, Taiwan') + ' & \\refTW & ' + T('30\\,000 card holders, 2005', '30\\,000 de deținători, 2005') + ' & ' + T('default in October 2005 (22.1\\%)', 'nerambursare în octombrie 2005 (22,1\\%)')],
    size='footnotesize') + items(
    T('Both under CC BY 4.0, saved once in data/credit of the course repository; South German Credit is the corrected version of the UCI German credit data \\refGroemping',
      'Ambele sub licența CC BY 4.0, salvate o singură dată în data/credit din repository-ul cursului; South German Credit este versiunea corectată a datelor germane de credit din UCI \\refGroemping'),
    T('Default $= 1$ for a bad credit or a missed payment, 0 otherwise; bad credits are oversampled in South German Credit (population rate about 5\\%)',
      'Nerambursare $= 1$ pentru un credit neperformant sau o plată ratată, 0 în rest; în South German Credit, creditele neperformante sînt suprareprezentate (rata din populație este de aproximativ 5\\%)'),
    T('In the notebook: \\texttt{load\\_credit(\'sgc\')}, \\texttt{woe\\_table(x, y, var)}, \\texttt{logit\\_fit(X, y)}, \\texttt{auc(score, y)}; no account or key is needed',
      'În notebook: \\texttt{load\\_credit(\'sgc\')}, \\texttt{woe\\_table(x, y, var)}, \\texttt{logit\\_fit(X, y)}, \\texttt{auc(score, y)}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Default and Expected Loss', 'Noțiuni necesare azi (1/5): nerambursarea și pierderea așteptată'), items(
    (T('\\textbf{Default}: the borrower does not pay as agreed (Basel: more than 90 days past due, or unlikely to pay) \\refBCBS', '\\textbf{Nerambursarea} (default): debitorul nu plătește conform contractului (Basel: peste 90 de zile de întîrziere sau probabilitate mică de plată) \\refBCBS'),
     [T('$Y = 1$ for default (a ``bad\'\'), $Y = 0$ for a good payer; a \\textbf{credit score} orders applicants by risk', '$Y = 1$ pentru nerambursare (rău-platnic), $Y = 0$ pentru bun-platnic; un \\textbf{scor de credit} ordonează solicitanții după risc')]),
    (T('\\textbf{PD}: probability of default in one year; \\textbf{LGD}: share of the exposure lost if default occurs; \\textbf{EAD}: amount owed at default',
       '\\textbf{PD}: probabilitatea de nerambursare într-un an; \\textbf{LGD}: partea din expunere pierdută în caz de nerambursare; \\textbf{EAD}: suma datorată la nerambursare'),
     [T('\\textbf{expected loss}: $\\mathrm{EL} = \\mathrm{PD} \\times \\mathrm{LGD} \\times \\mathrm{EAD}$', '\\textbf{pierderea așteptată}: $\\mathrm{EL} = \\mathrm{PD} \\times \\mathrm{LGD} \\times \\mathrm{EAD}$'),
      T('Basel IRB capital for other retail loans: $K = \\mathrm{LGD}[\\Phi((\\Phi^{-1}(\\mathrm{PD}) + \\sqrt{R}\\,\\Phi^{-1}(0.999))/\\sqrt{1 - R}) - \\mathrm{PD}]$, RWA $= 12.5\\,K\\,$EAD',
        'capitalul IRB din Basel pentru alte credite de retail: $K = \\mathrm{LGD}[\\Phi((\\Phi^{-1}(\\mathrm{PD}) + \\sqrt{R}\\,\\Phi^{-1}(0.999))/\\sqrt{1 - R}) - \\mathrm{PD}]$, RWA $= 12.5\\,K\\,$EAD'),
      T('$R$ falls from 0.16 to 0.03 as PD grows; RWA: risk-weighted assets', '$R$ scade de la 0,16 la 0,03 cînd PD crește; RWA: activele ponderate la risc')]),
    (T('\\textbf{Oversampling}: if the sample has a bad rate $\\pi_s$ and the population $\\pi$, add $\\ln\\frac{\\pi/(1 - \\pi)}{\\pi_s/(1 - \\pi_s)}$ to the log-odds of every applicant',
       '\\textbf{Suprareprezentarea}: dacă eșantionul are rata rău-platnicilor $\\pi_s$, iar populația $\\pi$, adunăm $\\ln\\frac{\\pi/(1 - \\pi)}{\\pi_s/(1 - \\pi_s)}$ la logaritmul șansei fiecărui solicitant'),
     [T('the ranking of the applicants does not change, the level of the PDs does', 'ordonarea solicitanților nu se schimbă, nivelul PD se schimbă')])), 'footnotesize')

D.frame(T('What You Need for Today (2/5): Odds and the Logit', 'Noțiuni necesare azi (2/5): șansa și modelul logit'), items(
    (T('\\textbf{Odds} of default $p/(1 - p)$; \\textbf{log-odds} $\\ln[p/(1 - p)]$; good:bad odds $(1 - p)/p$', '\\textbf{Șansa} (odds) de nerambursare $p/(1 - p)$; \\textbf{logaritmul șansei} $\\ln[p/(1 - p)]$; șansa bun:rău $(1 - p)/p$'),
     [T('$p = 0.2$: odds 0.25, log-odds $-1.386$, good:bad odds 4:1', '$p = 0.2$: șansa 0,25, logaritmul șansei $-1{,}386$, șansa bun:rău 4:1')]),
    (T('\\textbf{Logit model} \\refCox: $\\ln\\dfrac{p}{1 - p} = \\eta = \\beta_0 + \\beta_1x_1 + \\dots + \\beta_kx_k$, \\quad $p = \\dfrac{1}{1 + e^{-\\eta}}$',
       '\\textbf{Modelul logit} \\refCox: $\\ln\\dfrac{p}{1 - p} = \\eta = \\beta_0 + \\beta_1x_1 + \\dots + \\beta_kx_k$, \\quad $p = \\dfrac{1}{1 + e^{-\\eta}}$'),
     [T('$\\beta_0$: the constant (intercept); estimated by maximum likelihood', '$\\beta_0$: termenul liber; se estimează prin verosimilitate maximă'),
      T('$\\beta_j$: change in the log-odds per unit of $x_j$; $e^{\\beta_j}$: \\textbf{odds ratio}, the factor that multiplies the odds', '$\\beta_j$: modificarea logaritmului șansei la o unitate de $x_j$; $e^{\\beta_j}$: \\textbf{raportul șanselor} (odds ratio), factorul care înmulțește șansa'),
      T('effect on the probability: $\\Delta p \\approx p(1 - p)\\beta_j$', 'efectul asupra probabilității: $\\Delta p \\approx p(1 - p)\\beta_j$')]),
    (T('\\textbf{LDA} (linear discriminant analysis, \\refFisher): $w = S_W^{-1}(m_1 - m_0)$ separates the groups best', '\\textbf{LDA} (analiza discriminantă liniară, \\refFisher): $w = S_W^{-1}(m_1 - m_0)$ separă cel mai bine grupurile'),
     [T('$m_1$, $m_0$: means of bads and goods; $S_W$: their common (pooled) covariance; with Normal groups the log-odds are linear, as in the logit',
        '$m_1$, $m_0$: mediile rău-platnicilor și ale bun-platnicilor; $S_W$: covarianța lor comună; cu grupuri Normale, logaritmul șansei este liniar, ca în modelul logit')])), 'footnotesize')

D.frame(T('What You Need for Today (3/5): WoE and IV', 'Noțiuni necesare azi (3/5): WoE și IV'), items(
    (T('Split a variable into bins; $g_i$, $b_i$: goods and bads in bin $i$; $G$, $B$: their totals', 'Împărțim o variabilă în intervale (clase); $g_i$, $b_i$: bun-platnicii și rău-platnicii din intervalul $i$; $G$, $B$: totalurile'),
     [T('\\textbf{Weight of Evidence}: $\\mathrm{WoE}_i = \\ln\\dfrac{g_i/G}{b_i/B}$; positive: safer than average, negative: riskier', '\\textbf{Weight of Evidence}: $\\mathrm{WoE}_i = \\ln\\dfrac{g_i/G}{b_i/B}$; pozitiv: mai sigur decît media, negativ: mai riscant')]),
    (T('\\textbf{Information Value}: $\\mathrm{IV} = \\sum_i (g_i/G - b_i/B)\\,\\mathrm{WoE}_i$', '\\textbf{Information Value}: $\\mathrm{IV} = \\sum_i (g_i/G - b_i/B)\\,\\mathrm{WoE}_i$'),
     [T('rule of thumb \\refSiddiqi: below 0.02 useless, 0.02--0.1 weak, 0.1--0.3 medium, above 0.3 strong', 'regula practică \\refSiddiqi: sub 0,02 inutilă, 0,02--0,1 slabă, 0,1--0,3 medie, peste 0,3 puternică'),
      T('a bin without goods or without bads: add 0.5 to both counts', 'un interval fără bun-platnici sau fără rău-platnici: adăugăm 0,5 la ambele frecvențe')]),
    (T('A \\textbf{WoE logit}: every variable is replaced by the WoE of its bin; bins and WoE come from the training sample only', 'Un \\textbf{logit WoE}: fiecare variabilă este înlocuită cu WoE al intervalului ei; intervalele și WoE provin doar din eșantionul de estimare'),
     [T('the training sample is used for estimation; the test sample plays the future applicants and is used once, at the end', 'eșantionul de estimare servește la estimare; eșantionul de test joacă rolul viitorilor solicitanți și se folosește o singură dată, la final')])))

D.frame(T('What You Need for Today (4/5): Validation', 'Noțiuni necesare azi (4/5): validarea'), items(
    (T('Cut-off $c$: reject if PD $> c$. \\textbf{TP}: bad rejected, \\textbf{FN}: bad accepted, \\textbf{FP}: good rejected, \\textbf{TN}: good accepted',
       'Pragul $c$: respingem dacă PD $> c$. \\textbf{TP}: rău-platnic respins, \\textbf{FN}: rău-platnic acceptat, \\textbf{FP}: bun-platnic respins, \\textbf{TN}: bun-platnic acceptat'),
     [T('TPR $= $ TP/(TP + FN); FPR $= $ FP/(FP + TN); accuracy $=$ (TP + TN)/$n$; cost $= (C_{FN}\\,\\mathrm{FN} + C_{FP}\\,\\mathrm{FP})/n$', 'TPR $= $ TP/(TP + FN); FPR $= $ FP/(FP + TN); acuratețea $=$ (TP + TN)/$n$; costul $= (C_{FN}\\,\\mathrm{FN} + C_{FP}\\,\\mathrm{FP})/n$'),
      T('Bayes cut-off: reject if $p > C_{FP}/(C_{FP} + C_{FN})$; with $C_{FN} = 5$, $C_{FP} = 1$: $p > 1/6$', 'pragul Bayes: respingem dacă $p > C_{FP}/(C_{FP} + C_{FN})$; cu $C_{FN} = 5$, $C_{FP} = 1$: $p > 1/6$')]),
    (T('\\textbf{ROC} curve: (FPR, TPR) for all cut-offs; \\textbf{AUC} $= P(\\text{PD of a random bad} > \\text{PD of a random good})$, ties count 1/2 \\refHM',
       'Curba \\textbf{ROC}: (FPR, TPR) pentru toate pragurile; \\textbf{AUC} $= P(\\text{PD unui rău-platnic aleator} > \\text{PD unui bun-platnic aleator})$, egalitățile contează 1/2 \\refHM'),
     [T('by counting: the share of (bad, good) pairs in the right order; \\textbf{Gini} $= 2\\,\\mathrm{AUC} - 1$', 'prin numărare: ponderea perechilor (rău, bun) în ordinea corectă; \\textbf{Gini} $= 2\\,\\mathrm{AUC} - 1$'),
      T('\\textbf{KS} $= \\max_s |F_{\\text{bad}}(s) - F_{\\text{good}}(s)|$: the largest gap between the distribution functions of the PDs of bads and goods', '\\textbf{KS} $= \\max_s |F_{\\text{rău}}(s) - F_{\\text{bun}}(s)|$: cea mai mare distanță dintre funcțiile de repartiție ale PD pentru rău-platnici și bun-platnici')]),
    (T('\\textbf{Brier score} \\refBrier: $\\frac1n\\sum_i (p_i - y_i)^2$; \\textbf{calibration}: the mean PD of a group should equal its default rate',
       '\\textbf{Scorul Brier} \\refBrier: $\\frac1n\\sum_i (p_i - y_i)^2$; \\textbf{calibrarea}: PD medie a unui grup trebuie să fie egală cu rata lui de nerambursare'),
     [T('Hosmer--Lemeshow \\refHL: $\\sum_g (O_g - E_g)^2/[E_g(1 - \\bar p_g)] \\approx \\chi^2(G - 2)$; DeLong \\refDeLong: standard error and test of AUC',
        'Hosmer--Lemeshow \\refHL: $\\sum_g (O_g - E_g)^2/[E_g(1 - \\bar p_g)] \\approx \\chi^2(G - 2)$; DeLong \\refDeLong: eroarea standard și testul pentru AUC')])), 'footnotesize')

D.frame(T('What You Need for Today (5/5): Scorecards and Fairness', 'Noțiuni necesare azi (5/5): scorecard-uri și echitate'), items(
    (T('\\textbf{Scorecard}: Score $=$ Offset $+$ Factor $\\times \\ln$(good:bad odds) \\refSiddiqi', '\\textbf{Scorecard}: Scorul $=$ Offset $+$ Factor $\\times \\ln$(șansa bun:rău) \\refSiddiqi'),
     [T('\\textbf{PDO} (points to double the odds): Factor $=$ PDO$/\\ln 2$; base score at odds$_0$: Offset $=$ Base $-$ Factor $\\ln(\\text{odds}_0)$', '\\textbf{PDO} (points to double the odds): Factor $=$ PDO$/\\ln 2$; scorul de bază la șansa$_0$: Offset $=$ Baza $-$ Factor $\\ln(\\text{șansa}_0)$'),
      T('points of attribute $i$ of variable $j$ (WoE logit with $k$ variables): $-(\\beta_j\\,\\mathrm{WoE}_{ij} + \\beta_0/k)\\,$Factor $+$ Offset$/k$',
        'punctele atributului $i$ al variabilei $j$ (logit WoE cu $k$ variabile): $-(\\beta_j\\,\\mathrm{WoE}_{ij} + \\beta_0/k)\\,$Factor $+$ Offset$/k$')]),
    (T('\\textbf{Fairness} at a cut-off \\refHPS', '\\textbf{Echitatea} (fairness) la un prag \\refHPS'),
     [T('demographic parity: equal approval rates across groups', 'paritatea demografică: rate egale de aprobare între grupuri'),
      T('equal opportunity: equal approval rates of the good payers across groups', 'egalitatea de șanse: rate egale de aprobare a bun-platnicilor între grupuri'),
      T('calibration within groups: the same PD means the same default rate in every group', 'calibrarea în fiecare grup: aceeași PD înseamnă aceeași rată de nerambursare în fiecare grup')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

COEF = T('Logit for default: constant $-2.101$; duration (years) $0.385$; age (decades) $-0.159$; amount (1000 DM) $0.029$; no checking account $1.859$; balance $<$ 0 DM $1.338$.',
         'Logit pentru nerambursare: termenul liber $-2{,}101$; durata (ani) $0{,}385$; vîrsta (decenii) $-0{,}159$; suma (1000 DM) $0{,}029$; fără cont curent $1{,}859$; sold $<$ 0 DM $1{,}338$.')

D.solved(T('A1: The PD of an Applicant', 'A1: PD a unui solicitant'),
         items(COEF,
               T('Applicant: 2 years, 30 years old, 3000 DM, no checking account.', 'Solicitantul: 2 ani, 30 de ani, 3000 DM, fără cont curent.'),
               T('1. Compute the log-odds $\\eta$ and the PD.', '1. Calculați logaritmul șansei $\\eta$ și PD.'),
               T('2. Compute the odds of default and interpret $e^{1.859}$.', '2. Calculați șansa de nerambursare și interpretați $e^{1{,}859}$.'),
               T('3. Compute the PD with one more year of duration and compare it with $p(1 - p)\\beta$.', '3. Calculați PD cu un an în plus de durată și comparați-o cu $p(1 - p)\\beta$.'),
               T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
         items(T('1. $\\eta = -2.101 + 0.385 \\times 2 - 0.159 \\times 3 + 0.029 \\times 3 + 1.859 = @{a1.eta}$; PD $= 1/(1 + e^{-@{a1.eta}}) = @{a1.pd}\\%$',
                 '1. $\\eta = -2.101 + 0.385 \\times 2 - 0.159 \\times 3 + 0.029 \\times 3 + 1.859 = @{a1.eta}$; PD $= 1/(1 + e^{-@{a1.eta}}) = @{a1.pd}\\%$'),
               T('2. odds $= e^{@{a1.eta}} = @{a1.odds}$; $e^{1.859} = @{a1.or}$: without a checking account the odds are @{a1.or} times those of the reference group',
                 '2. șansa $= e^{@{a1.eta}} = @{a1.odds}$; $e^{1{,}859} = @{a1.or}$: fără cont curent, șansa este de @{a1.or} ori cea a grupului de referință'),
               T('3. $\\eta + 0.385$: PD $= @{a1.p1}\\%$ (+@{a1.dp} points); $p(1 - p)\\beta = @{a1.me}$ points', '3. $\\eta + 0{,}385$: PD $= @{a1.p1}\\%$ (+@{a1.dp} puncte procentuale); $p(1 - p)\\beta = @{a1.me}$ puncte procentuale'),
               T('The odds ratio is constant; the change in PD depends on where the applicant is.', 'Raportul șanselor este constant; modificarea PD depinde de poziția solicitantului.')),
         size='scriptsize')

D.proposed(T('A2: A Second Applicant', 'A2: un al doilea solicitant'),
           items(T('The same logit as in A1. Applicant: 1 year, 45 years old, 2000 DM, balance below 0 DM. Model: A1.', 'Același logit ca în A1. Solicitantul: 1 an, 45 de ani, 2000 DM, sold sub 0 DM. Model: A1.'),
                 T('1. Compute $\\eta$, the odds and the PD.', '1. Calculați $\\eta$, șansa și PD.'),
                 T('2. Compute the PD with one more year of duration.', '2. Calculați PD cu un an în plus de durată.'),
                 T('3. Compare the change in PD with that of A1 and explain the difference.', '3. Comparați modificarea PD cu cea din A1 și explicați diferența.'),
                 T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
           items(T('1. $\\eta = -2.101 + 0.385 - 0.159 \\times 4.5 + 0.029 \\times 2 + 1.338 = @{a2.eta}$; odds $@{a2.odds}$; PD $= @{a2.pd}\\%$',
                   '1. $\\eta = -2.101 + 0.385 - 0.159 \\times 4.5 + 0.029 \\times 2 + 1.338 = @{a2.eta}$; șansa $@{a2.odds}$; PD $= @{a2.pd}\\%$'),
                 T('2. PD $= @{a2.p1}\\%$ (+@{a2.dp} points; $p(1 - p)\\beta = @{a2.me}$)', '2. PD $= @{a2.p1}\\%$ (+@{a2.dp} puncte procentuale; $p(1 - p)\\beta = @{a2.me}$)'),
                 T('3. Smaller than in A1 (+@{a1.dp}): the same odds ratio @{a2.ord}, but $p(1 - p)$ is largest near $p = 0.5$.', '3. Mai mică decît în A1 (+@{a1.dp}): același raport al șanselor, @{a2.ord}, dar $p(1 - p)$ este maxim lîngă $p = 0{,}5$.')),
           size='scriptsize')


def woe_rows(key, codes, labels):
    return [f'{labels[c]} & @{{{key}.g{c}}} & @{{{key}.b{c}}} & $@{{{key}.dg{c}}}$ & $@{{{key}.db{c}}}$ & $@{{{key}.w{c}}}$ & $@{{{key}.iv{c}}}$'
            for c in codes]


ST = {1: T('no account', 'fără cont'), 2: T('$<$ 0 DM', '$<$ 0 DM'), 3: T('0--200 DM', '0--200 DM'), 4: T('$\\ge$ 200 DM / salary', '$\\ge$ 200 DM / salariu')}
D.solved(T('A3: WoE and IV of the Checking Account', 'A3: WoE și IV pentru contul curent'),
         items(T('South German Credit, full sample: goods and bads by checking account: no account @{a3.g1}/@{a3.b1}; below 0 DM @{a3.g2}/@{a3.b2}; 0--200 DM @{a3.g3}/@{a3.b3}; at least 200 DM or salary @{a3.g4}/@{a3.b4}.',
                 'South German Credit, eșantionul complet: bun-platnici/rău-platnici după contul curent: fără cont @{a3.g1}/@{a3.b1}; sub 0 DM @{a3.g2}/@{a3.b2}; 0--200 DM @{a3.g3}/@{a3.b3}; cel puțin 200 DM sau salariu @{a3.g4}/@{a3.b4}.'),
               T('1. Compute $g_i/G$ and $b_i/B$ for each group.', '1. Calculați $g_i/G$ și $b_i/B$ pentru fiecare grup.'),
               T('2. Compute the WoE of each group.', '2. Calculați WoE pentru fiecare grup.'),
               T('3. Compute the IV and classify the variable.', '3. Calculați IV și clasificați variabila.'),
               T('Report: a table and one sentence.', 'Raportați: un tabel și o frază.')),
         table('lrrrrrr', T('& $g$ & $b$ & $g/G$ & $b/B$ & WoE & IV', '& $g$ & $b$ & $g/G$ & $b/B$ & WoE & IV'),
               woe_rows('a3', [1, 2, 3, 4], ST) + ['total & 700 & 300 & 1 & 1 & & $@{a3.iv}$'], size='tiny') + items(
             T('e.g.\\ $\\ln(0.497/0.153) = @{a3.w4}$; IV $= @{a3.iv} > 0.3$: strong.', 'de exemplu $\\ln(0{,}497/0{,}153) = @{a3.w4}$; IV $= @{a3.iv} > 0{,}3$: puternică.'),
             T('WoE falls monotonically from the safest to the riskiest group.', 'WoE scade monoton de la grupul cel mai sigur la cel mai riscant.')),
         size='scriptsize')

SV = {1: T('none / unknown', 'fără / necunoscut'), 2: T('$<$ 100 DM', '$<$ 100 DM'), 3: T('100--500 DM', '100--500 DM'), 4: T('500--1000 DM', '500--1000 DM'), 5: T('$\\ge$ 1000 DM', '$\\ge$ 1000 DM')}
D.proposed(T('A4: WoE and IV of the Savings', 'A4: WoE și IV pentru economii'),
           items(T('Goods/bads by savings: none or unknown @{a4.g1}/@{a4.b1}; below 100 DM @{a4.g2}/@{a4.b2}; 100--500 DM @{a4.g3}/@{a4.b3}; 500--1000 DM @{a4.g4}/@{a4.b4}; at least 1000 DM @{a4.g5}/@{a4.b5}. Model: A3.',
                   'Bun-platnici/rău-platnici după economii: fără sau necunoscute @{a4.g1}/@{a4.b1}; sub 100 DM @{a4.g2}/@{a4.b2}; 100--500 DM @{a4.g3}/@{a4.b3}; 500--1000 DM @{a4.g4}/@{a4.b4}; cel puțin 1000 DM @{a4.g5}/@{a4.b5}. Model: A3.'),
                 T('1. Compute the WoE of each group.', '1. Calculați WoE pentru fiecare grup.'),
                 T('2. Compute the IV and classify the variable.', '2. Calculați IV și clasificați variabila.'),
                 T('3. Propose a merger of groups with similar WoE.', '3. Propuneți o unire a grupurilor cu WoE apropiat.'),
                 T('Report: a table and two sentences.', 'Raportați: un tabel și două fraze.')),
           table('lrrrrrr', T('& $g$ & $b$ & $g/G$ & $b/B$ & WoE & IV', '& $g$ & $b$ & $g/G$ & $b/B$ & WoE & IV'),
                 woe_rows('a4', [1, 2, 3, 4, 5], SV) + ['total & 700 & 300 & 1 & 1 & & $@{a4.iv}$'], size='tiny') + items(
               T('IV $= @{a4.iv}$: medium.', 'IV $= @{a4.iv}$: medie.'),
               T('Merge 100--500 and at least 1000 DM (WoE $@{a4.w3}$ and $@{a4.w5}$), and none with below 100 DM.', 'Unim 100--500 și cel puțin 1000 DM (WoE $@{a4.w3}$ și $@{a4.w5}$), precum și „fără” cu „sub 100 DM”.')),
           size='scriptsize')

D.solved(T('A5: Confusion Matrix and AUC by Counting', 'A5: matricea de confuzie și AUC prin numărare'),
         items(T('Ten applicants, PD and outcome ($Y = 1$: default): 0.05 (0), 0.10 (0), 0.12 (0), 0.20 (0), 0.25 (1), 0.30 (0), 0.40 (0), 0.55 (1), 0.60 (0), 0.80 (1).',
                 'Zece solicitanți, PD și rezultatul ($Y = 1$: nerambursare): 0,05 (0); 0,10 (0); 0,12 (0); 0,20 (0); 0,25 (1); 0,30 (0); 0,40 (0); 0,55 (1); 0,60 (0); 0,80 (1).'),
               T('1. Build the confusion matrix at the cut-off 0.5 and compute accuracy, TPR and FPR.', '1. Construiți matricea de confuzie la pragul 0,5 și calculați acuratețea, TPR și FPR.'),
               T('2. Compute the AUC by counting the (bad, good) pairs in the right order, and the Gini.', '2. Calculați AUC numărînd perechile (rău, bun) în ordinea corectă, precum și Gini.'),
               T('3. Compute the KS statistic and the Brier score.', '3. Calculați statistica KS și scorul Brier.'),
               T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
         items(T('1. TP @{a5.TP}, FN @{a5.FN}, FP @{a5.FP}, TN @{a5.TN}: accuracy @{a5.acc}\\%, TPR @{a5.tpr}\\%, FPR @{a5.fpr}\\%', '1. TP @{a5.TP}, FN @{a5.FN}, FP @{a5.FP}, TN @{a5.TN}: acuratețe @{a5.acc}\\%, TPR @{a5.tpr}\\%, FPR @{a5.fpr}\\%'),
               T('2. goods below each bad: @{a5.wins} $=$ @{a5.sumw} of @{a5.pairs} pairs: AUC $= @{a5.auc}$, Gini $= @{a5.gini}$', '2. bun-platnici sub fiecare rău-platnic: @{a5.wins} $=$ @{a5.sumw} din @{a5.pairs} de perechi: AUC $= @{a5.auc}$, Gini $= @{a5.gini}$'),
               T('3. at PD $\\le$ @{a5.ksat}: 4/7 of goods, 0/3 of bads: KS $= @{a5.ks}$; Brier $= @{a5.bsum}/10 = @{a5.brier}$', '3. la PD $\\le$ @{a5.ksat}: 4/7 din bun-platnici, 0/3 din rău-platnici: KS $= @{a5.ks}$; Brier $= @{a5.bsum}/10 = @{a5.brier}$'),
               T('The bad with PD 0.25 is ranked below three goods and the bad with 0.55 below one: @{a5.wrong} of @{a5.pairs} pairs are in the wrong order.', 'Rău-platnicul cu PD 0,25 este ordonat sub trei bun-platnici, iar cel cu 0,55 sub unul: @{a5.wrong} din @{a5.pairs} perechi sînt în ordinea greșită.')),
         size='scriptsize')

D.proposed(T('A6: A Second Sample', 'A6: un al doilea eșantion'),
           items(T('Eight applicants: 0.08 (0), 0.15 (0), 0.22 (1), 0.35 (0), 0.42 (0), 0.50 (1), 0.65 (1), 0.90 (0). Model: A5.', 'Opt solicitanți: 0,08 (0); 0,15 (0); 0,22 (1); 0,35 (0); 0,42 (0); 0,50 (1); 0,65 (1); 0,90 (0). Model: A5.'),
                 T('1. Build the confusion matrix at the cut-off 0.4 and compute accuracy, TPR and FPR.', '1. Construiți matricea de confuzie la pragul 0,4 și calculați acuratețea, TPR și FPR.'),
                 T('2. Compute the AUC by counting pairs, and the Gini.', '2. Calculați AUC prin numărarea perechilor, precum și Gini.'),
                 T('3. Compute the KS statistic and the Brier score.', '3. Calculați statistica KS și scorul Brier.'),
                 T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
           items(T('1. TP @{a6.TP}, FN @{a6.FN}, FP @{a6.FP}, TN @{a6.TN}: accuracy @{a6.acc}\\%, TPR @{a6.tpr}\\%, FPR @{a6.fpr}\\%', '1. TP @{a6.TP}, FN @{a6.FN}, FP @{a6.FP}, TN @{a6.TN}: acuratețe @{a6.acc}\\%, TPR @{a6.tpr}\\%, FPR @{a6.fpr}\\%'),
                 T('2. @{a6.wins} $=$ @{a6.sumw} of @{a6.pairs}: AUC $= @{a6.auc}$, Gini $= @{a6.gini}$', '2. @{a6.wins} $=$ @{a6.sumw} din @{a6.pairs}: AUC $= @{a6.auc}$, Gini $= @{a6.gini}$'),
                 T('3. KS $= @{a6.ks}$ at PD $\\le$ @{a6.ksat}; Brier $= @{a6.brier}$', '3. KS $= @{a6.ks}$ la PD $\\le$ @{a6.ksat}; Brier $= @{a6.brier}$'),
                 T('The good with PD 0.90 costs both ranking and Brier score: a confident wrong PD is penalised most.', 'Bun-platnicul cu PD 0,90 penalizează atît ordonarea, cît și scorul Brier: o PD greșită și sigură este penalizată cel mai mult.')),
           size='scriptsize')

D.solved(T('A7: Scaling a Scorecard', 'A7: scalarea unui scorecard'),
         items(T('PDO $= 20$; 600 points at good:bad odds 50:1. A WoE logit with $k = 6$ variables has constant $\\beta_0 = @{a7.b0}$ and, for the checking account, $\\beta = @{a7.b}$; WoE: no account $@{a7.w1}$, at least 200 DM $@{a7.w4}$.',
                 'PDO $= 20$; 600 de puncte la șansa bun:rău 50:1. Un logit WoE cu $k = 6$ variabile are termenul liber $\\beta_0 = @{a7.b0}$ și, pentru contul curent, $\\beta = @{a7.b}$; WoE: fără cont $@{a7.w1}$, cel puțin 200 DM $@{a7.w4}$.'),
               T('1. Compute the Factor and the Offset.', '1. Calculați Factor și Offset.'),
               T('2. Compute the score of a PD of 5\\% and the PD of a score of 560.', '2. Calculați scorul unei PD de 5\\% și PD a unui scor de 560.'),
               T('3. Compute the points of the two checking-account groups.', '3. Calculați punctele celor două grupuri de cont curent.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. Factor $= 20/\\ln 2 = @{a7.factor}$; Offset $= 600 - @{a7.factor}\\ln 50 = @{a7.offset}$', '1. Factor $= 20/\\ln 2 = @{a7.factor}$; Offset $= 600 - @{a7.factor}\\ln 50 = @{a7.offset}$'),
               T('2. $@{a7.offset} + @{a7.factor} \\times \\ln 19 = @{a7.score}$; 560: $\\ln$ odds $= @{a7.ln_odds}$, odds @{a7.odds}:1, PD $= @{a7.pd}\\%$',
                 '2. $@{a7.offset} + @{a7.factor} \\times \\ln 19 = @{a7.score}$; 560: $\\ln$ șansa $= @{a7.ln_odds}$, șansa @{a7.odds}:1, PD $= @{a7.pd}\\%$'),
               T('3. Offset$/k = @{a7.ok}$, $\\beta_0/k = @{a7.b0k}$: no account $-(@{a7.b} \\times (@{a7.w1}) + (@{a7.b0k})) \\times @{a7.factor} + @{a7.ok} = @{a7.pt1}$; at least 200 DM: @{a7.pt4}',
                 '3. Offset$/k = @{a7.ok}$, $\\beta_0/k = @{a7.b0k}$: fără cont $-(@{a7.b} \\times (@{a7.w1}) + (@{a7.b0k})) \\times @{a7.factor} + @{a7.ok} = @{a7.pt1}$; cel puțin 200 DM: @{a7.pt4}'),
               T('A well-funded account is worth @{a7.dpt} points more than no account: @{a7.dbl} doublings of the good:bad odds.', 'Un cont bine alimentat valorează cu @{a7.dpt} puncte mai mult decît lipsa contului: @{a7.dbl} dublări ale șansei bun:rău.')),
         size='scriptsize')

D.proposed(T('A8: Another Scale, Prior Correction and Expected Loss', 'A8: altă scală, corecția a priori și pierderea așteptată'),
           items(T('A bank uses PDO $= 40$ and 500 points at odds 20:1. A model built on a sample with 30\\% bads gives an applicant PD $= 30\\%$; the population bad rate is 5\\%; EAD $= 10\\,000$ DM, LGD $= 45\\%$. Model: A7.',
                   'O bancă folosește PDO $= 40$ și 500 de puncte la șansa 20:1. Un model estimat pe un eșantion cu 30\\% rău-platnici dă unui solicitant PD $= 30\\%$; rata din populație este 5\\%; EAD $= 10\\,000$ DM, LGD $= 45\\%$. Model: A7.'),
                 T('1. Compute the Factor, the Offset, the score of a PD of 10\\% and the PD of a score of 520.', '1. Calculați Factor, Offset, scorul unei PD de 10\\% și PD a unui scor de 520.'),
                 T('2. Correct the PD of 30\\% to the population.', '2. Corectați PD de 30\\% pentru populație.'),
                 T('3. Compute the expected loss with both PDs.', '3. Calculați pierderea așteptată cu ambele PD.'),
                 T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
           items(T('1. Factor $= 40/\\ln 2 = @{a8.factor}$; Offset $= 500 - @{a8.factor}\\ln 20 = @{a8.offset}$; PD 10\\%: $@{a8.offset} + @{a8.factor}\\ln 9 = @{a8.score}$; 520: odds @{a8.odds}:1, PD $= @{a8.pd}\\%$',
                   '1. Factor $= 40/\\ln 2 = @{a8.factor}$; Offset $= 500 - @{a8.factor}\\ln 20 = @{a8.offset}$; PD 10\\%: $@{a8.offset} + @{a8.factor}\\ln 9 = @{a8.score}$; 520: șansa @{a8.odds}:1, PD $= @{a8.pd}\\%$'),
                 T('2. $\\ln(0.3/0.7) = @{a8.lns}$; shift $\\ln(0.05/0.95) - \\ln(0.3/0.7) = @{a8.shift}$; $@{a8.lnp}$: PD $= @{a8.pdpop}\\%$', '2. $\\ln(0{,}3/0{,}7) = @{a8.lns}$; corecția $\\ln(0{,}05/0{,}95) - \\ln(0{,}3/0{,}7) = @{a8.shift}$; $@{a8.lnp}$: PD $= @{a8.pdpop}\\%$'),
                 T('3. EL $= 0.30 \\times 0.45 \\times 10\\,000 = @{a8.el1}$ DM without correction; $0.05 \\times 0.45 \\times 10\\,000 = @{a8.el2}$ DM with it', '3. EL $= 0{,}30 \\times 0{,}45 \\times 10\\,000 = @{a8.el1}$ DM fără corecție; $0{,}05 \\times 0{,}45 \\times 10\\,000 = @{a8.el2}$ DM cu ea'),
                 T('Without the correction the provision would be six times too large.', 'Fără corecție, provizionul ar fi de șase ori prea mare.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: A WoE Scorecard for South German Credit [Solved]', 'B1: un scorecard WoE pentru South German Credit [Rezolvat]'),
       T('which variables carry information about default, and how well does a WoE logit rank new applicants?', 'ce variabile aduc informație despre nerambursare și cît de bine ordonează un logit WoE solicitanții noi?'),
       T('South German Credit; a stratified split: 700 credits for estimation, 300 for testing (30\\% bads in both)', 'South German Credit; o împărțire stratificată: 700 de credite pentru estimare, 300 pentru test (30\\% rău-platnici în ambele)'),
       [T('Compute the WoE tables and the IV of the 20 variables on the training sample (duration, amount and age in five fixed bins).', 'Calculați tabelele WoE și IV pentru cele 20 de variabile pe eșantionul de estimare (durata, suma și vîrsta în cinci intervale fixe).'),
        T('Keep the variables with IV of at least 0.1 and estimate a logit for default on their WoE values.', 'Păstrați variabilele cu IV de cel puțin 0,1 și estimați un logit pentru nerambursare pe valorile lor WoE.'),
        T('Compute the AUC on the training and on the test sample, with a 95\\% DeLong interval for the test AUC.', 'Calculați AUC pe eșantionul de estimare și pe cel de test, cu un interval DeLong de 95\\% pentru AUC de test.'),
        T('Draw the WoE of the credit-history groups.', 'Reprezentați grafic WoE pentru grupurile de istoric de credit.'),
        T('Interpretation: does the order of the credit-history groups make economic sense?', 'Interpretare: are sens economic ordinea grupurilor de istoric de credit?')],
       T('the list of selected variables with their IV, the coefficients, two AUCs with the interval, the chart and two sentences', 'lista variabilelor selectate cu IV-ul lor, coeficienții, două valori AUC cu intervalul, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch12_sem_b1', h='0.32') + table(
    'l' + 'r' * (len(S['B1']['sel']) + 1), '& ' + T('constant', 'termen liber') + ' & ' + ' & '.join(vname(v) for v in S['B1']['sel']),
    ['IV & & ' + ' & '.join(f'$@{{b1.iv.{v}}}$' for v in S['B1']['sel']),
     '$\\hat\\beta$ & ' + ' & '.join(f'$@{{b1.{v}.b}}$' for v in ['const'] + S['B1']['sel']),
     'se & ' + ' & '.join(f'$@{{b1.{v}.se}}$' for v in ['const'] + S['B1']['sel'])], size='tiny') + items(
    T('@{b1.nsel} variables with IV $\\ge 0.1$; AUC: training $@{b1.auctr}$, test $@{b1.auc}$, 95\\% CI $[@{b1.lo}, @{b1.hi}]$', '@{b1.nsel} variabile cu IV $\\ge 0{,}1$; AUC: estimare $@{b1.auctr}$, test $@{b1.auc}$, CI 95\\% $[@{b1.lo}, @{b1.hi}]$'),
    T('Interpretation: yes: a delay in the past (@{b1.ch0}\\% bad) and a critical account (@{b1.ch1}\\%) are the riskiest, all credits at this bank paid back duly (@{b1.ch4}\\%) the safest; in the uncorrected UCI labels this safest group was called ``critical account\'\'',
      'Interpretare: da: o întîrziere în trecut (@{b1.ch0}\\% neperformante) și un cont critic (@{b1.ch1}\\%) sînt cele mai riscante, toate creditele la această bancă rambursate la timp (@{b1.ch4}\\%) cele mai sigure; în etichetele UCI necorectate, acest grup, cel mai sigur, se numea „cont critic”')) + qlsem(), 'scriptsize')

D.task(T('B2: Information Value of the Taiwan Data [Proposed]', 'B2: Information Value pentru datele din Taiwan [Propus]'),
       T('which kind of information predicts the default of a card holder: payment behaviour or personal data? Model: B1.', 'ce fel de informație prezice nerambursarea unui deținător de card: comportamentul de plată sau datele personale? Model: B1.'),
       T('Taiwan credit cards, 30\\,000 clients; the same stratified 70/30 split', 'cardurile de credit din Taiwan, 30\\,000 de clienți; aceeași împărțire stratificată 70/30'),
       [T('Compute the IV of \\texttt{PAY\\_0} and \\texttt{PAY\\_2} (repayment status in September and August; categories $-2$ to 3 or more), of \\texttt{LIMIT\\_BAL}, \\texttt{PAY\\_AMT1}, \\texttt{BILL\\_AMT1} and \\texttt{AGE} (quintiles) and of \\texttt{SEX}, \\texttt{EDUCATION} and \\texttt{MARRIAGE} (categories) on the training sample.',
          'Calculați IV pentru \\texttt{PAY\\_0} și \\texttt{PAY\\_2} (starea plăților în septembrie și august; categoriile de la $-2$ la 3 sau mai mult), pentru \\texttt{LIMIT\\_BAL}, \\texttt{PAY\\_AMT1}, \\texttt{BILL\\_AMT1} și \\texttt{AGE} (quintile) și pentru \\texttt{SEX}, \\texttt{EDUCATION} și \\texttt{MARRIAGE} (categorii) pe eșantionul de estimare.'),
        T('Compute the default rate of each \\texttt{PAY\\_0} category.', 'Calculați rata de nerambursare pentru fiecare categorie \\texttt{PAY\\_0}.'),
        T('Rank the variables by IV and classify them with the rule of thumb.', 'Ordonați variabilele după IV și clasificați-le cu regula practică.'),
        T('Interpretation: why is the last month\'s repayment status so much stronger than the personal data?', 'Interpretare: de ce este starea plății din ultima lună mult mai puternică decît datele personale?')],
       T('a table of nine IVs, six default rates and two sentences', 'un tabel cu nouă valori IV, șase rate de nerambursare și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrrrr', T('& \\texttt{PAY\\_0} & \\texttt{PAY\\_2} & \\texttt{LIMIT} & \\texttt{PAY\\_AMT1} & \\texttt{EDUC.} & \\texttt{AGE} & \\texttt{MARR.} & \\texttt{BILL\\_AMT1} & \\texttt{SEX}', '& \\texttt{PAY\\_0} & \\texttt{PAY\\_2} & \\texttt{LIMIT} & \\texttt{PAY\\_AMT1} & \\texttt{EDUC.} & \\texttt{AGE} & \\texttt{MARR.} & \\texttt{BILL\\_AMT1} & \\texttt{SEX}'),
    ['IV & ' + ' & '.join(f'$@{{b2.iv.{v}}}$' for v in ['PAY_0', 'PAY_2', 'LIMIT_BAL', 'PAY_AMT1', 'EDUCATION', 'AGE', 'MARRIAGE', 'BILL_AMT1', 'SEX'])],
    size='scriptsize') + items(
    T('Default rate by \\texttt{PAY\\_0}: $-2$: @{b2.pm2}\\%; $-1$: @{b2.pm1}\\%; 0: @{b2.p0}\\%; 1: @{b2.p1}\\%; 2: @{b2.p2}\\%; 3 or more: @{b2.p3}\\%', 'Rata de nerambursare după \\texttt{PAY\\_0}: $-2$: @{b2.pm2}\\%; $-1$: @{b2.pm1}\\%; 0: @{b2.p0}\\%; 1: @{b2.p1}\\%; 2: @{b2.p2}\\%; 3 sau mai mult: @{b2.p3}\\%'),
    T('\\texttt{PAY\\_0} and \\texttt{PAY\\_2}: strong; limit and payment: medium; education: weak; age, marital status, bill and sex: useless', '\\texttt{PAY\\_0} și \\texttt{PAY\\_2}: puternice; limita și plata: medii; educația: slabă; vîrsta, starea civilă, factura și sexul: inutile'),
    T('Interpretation: a client who is already two months late has most of the way to default behind him: behaviour measures the outcome almost directly, personal data only through averages',
      'Interpretare: un client care întîrzie deja două luni a parcurs mare parte din drumul spre nerambursare: comportamentul măsoară rezultatul aproape direct, datele personale doar prin medii')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Validating the Scorecard [Solved]', 'B3: validarea scorecard-ului [Rezolvat]'),
       T('how good is the B1 scorecard on the test sample, and which cut-off should the bank use?', 'cît de bun este scorecard-ul din B1 pe eșantionul de test și ce prag ar trebui să folosească banca?'),
       T('the 300 test credits of B1 with their PDs; costs: 5 for an accepted bad, 1 for a rejected good', 'cele 300 de credite de test din B1 cu PD-urile lor; costuri: 5 pentru un rău-platnic acceptat, 1 pentru un bun-platnic respins'),
       [T('Build the confusion matrix at the cut-offs 0.5 and 1/6, with accuracy, TPR, FPR and the cost per applicant.', 'Construiți matricea de confuzie la pragurile 0,5 și 1/6, cu acuratețea, TPR, FPR și costul pe solicitant.'),
        T('Compute the AUC, the Gini, the accuracy ratio of the CAP curve and the KS statistic.', 'Calculați AUC, Gini, raportul de acuratețe al curbei CAP și statistica KS.'),
        T('Compute the Brier score, compare it with a constant PD of 30\\%, and run the Hosmer--Lemeshow test with five groups.', 'Calculați scorul Brier, comparați-l cu o PD constantă de 30\\% și aplicați testul Hosmer--Lemeshow cu cinci grupuri.'),
        T('Draw the ROC curve with the two cut-offs and the calibration plot.', 'Reprezentați curba ROC cu cele două praguri și graficul de calibrare.'),
        T('Interpretation: which cut-off would you recommend to the bank?', 'Interpretare: ce prag i-ați recomanda băncii?')],
       T('a table of two confusion matrices, six statistics, the chart and two sentences', 'un tabel cu două matrice de confuzie, șase statistici, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch12_sem_b3', h='0.30') + table(
    'lrrrrrrrr', T('cut-off & TP & FN & FP & TN & accuracy & TPR & FPR & cost', 'prag & TP & FN & FP & TN & acuratețe & TPR & FPR & cost'),
    ['0.5 & @{b3.cm50.TP} & @{b3.cm50.FN} & @{b3.cm50.FP} & @{b3.cm50.TN} & $@{b3.cm50.acc}\\%$ & $@{b3.cm50.tpr}\\%$ & $@{b3.cm50.fpr}\\%$ & $@{b3.cm50.cost}$',
     '1/6 & @{b3.cm_cost.TP} & @{b3.cm_cost.FN} & @{b3.cm_cost.FP} & @{b3.cm_cost.TN} & $@{b3.cm_cost.acc}\\%$ & $@{b3.cm_cost.tpr}\\%$ & $@{b3.cm_cost.fpr}\\%$ & $@{b3.cm_cost.cost}$'],
    size='tiny') + items(
    T('AUC $= @{b3.auc}$, Gini $= @{b3.gini}$, AR $= @{b3.ar}$, KS $= @{b3.ks}$; Brier $= @{b3.brier}$ against $@{b3.brier_ref}$; HL $= @{b3.hl}$ (p $= @{b3.hl_p}$): no evidence of miscalibration',
      'AUC $= @{b3.auc}$, Gini $= @{b3.gini}$, AR $= @{b3.ar}$, KS $= @{b3.ks}$; Brier $= @{b3.brier}$ față de $@{b3.brier_ref}$; HL $= @{b3.hl}$ (p $= @{b3.hl_p}$): nicio dovadă de calibrare greșită'),
    T('Interpretation: the cut-off of 1/6: it rejects more applicants (@{b3.cm_cost.reject}\\%) but catches @{b3.cm_cost.tpr}\\% of the bads, and the cost falls from @{b3.cm50.cost} to @{b3.cm_cost.cost}; accuracy is the wrong criterion when errors have different costs',
      'Interpretare: pragul de 1/6: respinge mai mulți solicitanți (@{b3.cm_cost.reject}\\%), dar identifică @{b3.cm_cost.tpr}\\% din rău-platnici, iar costul scade de la @{b3.cm50.cost} la @{b3.cm_cost.cost}; acuratețea este un criteriu greșit cînd erorile au costuri diferite')) + qlsem(), 'scriptsize')

D.task(T('B4: Logit or LDA on the Taiwan Data? [Proposed]', 'B4: logit sau LDA pe datele din Taiwan? [Propus]'),
       T('do the logit and Fisher\'s LDA differ in ranking or in calibration? Model: B3.', 'diferă logit-ul și LDA a lui Fisher prin ordonare sau prin calibrare? Model: B3.'),
       T('Taiwan, 70/30 split; 13 standardised inputs: log limit, age, indicators of the September repayment status, number of late months, utilisation, log payments', 'Taiwan, împărțire 70/30; 13 variabile standardizate: logaritmul limitei, vîrsta, indicatorii stării plății din septembrie, numărul lunilor cu întîrziere, gradul de utilizare, logaritmul plăților'),
       [T('Estimate a logit and an LDA on the training sample with the same inputs.', 'Estimați un logit și o LDA pe eșantionul de estimare cu aceleași variabile.'),
        T('Compute the test AUC of both and the DeLong test of equal AUCs.', 'Calculați AUC de test pentru ambele și testul DeLong de egalitate a AUC.'),
        T('Compute the Brier score, the Hosmer--Lemeshow statistic (ten groups) and the mean PD of both, against the default rate.', 'Calculați scorul Brier, statistica Hosmer--Lemeshow (zece grupuri) și PD medie pentru ambele, față de rata de nerambursare.'),
        T('Compute the accuracy of the rule ``nobody defaults\'\' and of both models at the cut-off 0.5.', 'Calculați acuratețea regulii „nimeni nu intră în nerambursare” și a celor două modele la pragul 0,5.'),
        T('Interpretation: is the LDA worse at ranking or at calibration?', 'Interpretare: este LDA mai slabă la ordonare sau la calibrare?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrr', T('& AUC & Brier & HL & mean PD & accuracy (0.5) & TPR (0.5)', '& AUC & Brier & HL & PD medie & acuratețe (0,5) & TPR (0,5)'),
    ['logit & $@{b4.auc_logit}$ & $@{b4.brier_logit}$ & $@{b4.hll}$ & $@{b4.mean_pd_logit}\\%$ & $@{b4.logit.acc}\\%$ & $@{b4.logit.tpr}\\%$',
     'LDA & $@{b4.auc_lda}$ & $@{b4.brier_lda}$ & $@{b4.hld}$ & $@{b4.mean_pd_lda}\\%$ & $@{b4.lda.acc}\\%$ & $@{b4.lda.tpr}\\%$'],
    size='scriptsize') + items(
    T('DeLong: $z = @{b4.z}$, p $= @{b4.p}$; correlation of the log-odds $@{b4.corr}$; test default rate @{b4.rate}\\%; ``nobody defaults\'\': accuracy @{b4.acc_nobody}\\%', 'DeLong: $z = @{b4.z}$, p $= @{b4.p}$; corelația logaritmilor șansei $@{b4.corr}$; rata de nerambursare în test @{b4.rate}\\%; „nimeni nu intră în nerambursare”: acuratețe @{b4.acc_nobody}\\%'),
    T('Both models beat the trivial rule by only @{b4.gain} points of accuracy, but they catch about a third of the defaults', 'Ambele modele depășesc regula trivială cu doar @{b4.gain} puncte procentuale de acuratețe, dar identifică aproximativ o treime din nerambursări'),
    T('Interpretation: at calibration: the ranking is the same (AUC equal at 5\\%), but the LDA PDs are off (HL @{b4.hld} against @{b4.hll}) because the inputs are dummies and skewed, far from Normal',
      'Interpretare: la calibrare: ordonarea este aceeași (AUC egale la 5\\%), dar PD-urile LDA sînt deplasate (HL @{b4.hld} față de @{b4.hll}), deoarece variabilele sînt indicatori și asimetrice, departe de distribuția Normală')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: From PDs to Capital [Solved]', 'B5: de la PD la capital [Rezolvat]'),
       T('how much expected loss and Basel capital does the test portfolio of B1 carry, and how much does the prior correction matter?', 'cîtă pierdere așteptată și cît capital Basel are portofoliul de test din B1 și cît contează corecția a priori?'),
       T('the 300 test credits of B1; EAD $=$ credit amount; LGD $= 45\\%$; population bad rate 5\\%; other retail exposures', 'cele 300 de credite de test din B1; EAD $=$ suma creditului; LGD $= 45\\%$; rata din populație 5\\%; alte expuneri de retail'),
       [T('Correct the PD of every credit from the sample rate of 30\\% to the population rate of 5\\%.', 'Corectați PD a fiecărui credit de la rata de 30\\% din eșantion la rata de 5\\% din populație.'),
        T('Compute the expected loss of the portfolio, in DM and as a share of EAD, with and without the correction.', 'Calculați pierderea așteptată a portofoliului, în DM și ca pondere din EAD, cu și fără corecție.'),
        T('Compute the IRB capital $K$ and the risk-weighted assets of the portfolio with the corrected PDs.', 'Calculați capitalul IRB $K$ și activele ponderate la risc ale portofoliului cu PD corectate.'),
        T('Interpretation: why does the correction change the expected loss but not the AUC?', 'Interpretare: de ce schimbă corecția pierderea așteptată, dar nu și AUC?')],
       T('eight numbers and two sentences', 'opt valori și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), items(
    (T('Shift of the log-odds: $\\ln(0.05/0.95) - \\ln(0.30/0.70) = @{b5.shift}$', 'Corecția logaritmului șansei: $\\ln(0{,}05/0{,}95) - \\ln(0{,}30/0{,}70) = @{b5.shift}$'),
     [T('mean PD: @{b5.mean_pd_sample}\\% in the sample scale, @{b5.mean_pd_pop}\\% after the correction (median @{b5.median_pd_pop}\\%, maximum @{b5.max_pd_pop}\\%)', 'PD medie: @{b5.mean_pd_sample}\\% pe scala eșantionului, @{b5.mean_pd_pop}\\% după corecție (mediana @{b5.median_pd_pop}\\%, maximul @{b5.max_pd_pop}\\%)')]),
    (T('EAD @{b5.ead} DM; EL @{b5.el} DM $=$ @{b5.el_pct}\\% of EAD (without correction: @{b5.el_naive_pct}\\%)', 'EAD @{b5.ead} DM; EL @{b5.el} DM $=$ @{b5.el_pct}\\% din EAD (fără corecție: @{b5.el_naive_pct}\\%)'),
     [T('IRB capital $K$: @{b5.k} DM $=$ @{b5.k_pct}\\% of EAD; RWA @{b5.rwa} DM, a risk weight of @{b5.rwa_density}\\%', 'capitalul IRB $K$: @{b5.k} DM $=$ @{b5.k_pct}\\% din EAD; RWA @{b5.rwa} DM, o pondere de risc de @{b5.rwa_density}\\%')]),
    T('The mean corrected PD is above 5\\% because the correction is applied to the log-odds, not to the PD', 'PD medie corectată este peste 5\\%, deoarece corecția se aplică logaritmului șansei, nu PD'),
    T('Interpretation: the correction adds the same constant to every log-odds: the order of the applicants, hence AUC, is unchanged, while every PD shrinks and EL falls from @{b5.el_naive_pct}\\% to @{b5.el_pct}\\% of EAD',
      'Interpretare: corecția adaugă aceeași constantă la fiecare logaritm al șansei: ordinea solicitanților, deci AUC, nu se schimbă, în timp ce fiecare PD scade, iar EL coboară de la @{b5.el_naive_pct}\\% la @{b5.el_pct}\\% din EAD')) + qlsem(), 'footnotesize')

D.task(T('B6: Fairness on the Taiwan Data [Proposed]', 'B6: echitatea pe datele din Taiwan [Propus]'),
       T('does the logit of B4 treat men and women, and young and older clients, alike? Model: B5 and slide 5/5.', 'tratează logit-ul din B4 în mod egal bărbații și femeile, precum și clienții tineri și pe cei mai în vîrstă? Model: B5 și slide-ul 5/5.'),
       T('Taiwan test sample (9000 clients); sex and age are not model inputs; approve the 75\\% with the lowest PD', 'eșantionul de test din Taiwan (9000 de clienți); sexul și vîrsta nu sînt variabile ale modelului; aprobăm cei 75\\% cu PD cea mai mică'),
       [T('Compute the approval rate and the default rate of men and women, and of clients up to 25 years and older.', 'Calculați rata de aprobare și rata de nerambursare pentru bărbați și femei, precum și pentru clienții de pînă la 25 de ani și cei mai în vîrstă.'),
        T('Compute the approval rate of the good payers (equal opportunity) in each group.', 'Calculați rata de aprobare a bun-platnicilor (egalitatea de șanse) în fiecare grup.'),
        T('Compute the default rate among the approved clients of each group.', 'Calculați rata de nerambursare printre clienții aprobați din fiecare grup.'),
        T('Interpretation: which fairness criterion does the model come closest to satisfying?', 'Interpretare: de care criteriu de echitate se apropie cel mai mult modelul?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrr', T('& $n$ & default rate & approved & good payers approved & bad among approved', '& $n$ & rata de nerambursare & aprobați & bun-platnici aprobați & rău-platnici printre aprobați'),
    [T('men', 'bărbați') + ' & @{b6.m.n} & $@{b6.m.rate}\\%$ & $@{b6.m.approve}\\%$ & $@{b6.m.tpr_good}\\%$ & $@{b6.m.bad_approved}\\%$',
     T('women', 'femei') + ' & @{b6.w.n} & $@{b6.w.rate}\\%$ & $@{b6.w.approve}\\%$ & $@{b6.w.tpr_good}\\%$ & $@{b6.w.bad_approved}\\%$',
     T('age $\\le$ 25', 'vîrsta $\\le$ 25') + ' & @{b6.y.n} & $@{b6.y.rate}\\%$ & $@{b6.y.approve}\\%$ & $@{b6.y.tpr_good}\\%$ & $@{b6.y.bad_approved}\\%$',
     T('age $>$ 25', 'vîrsta $>$ 25') + ' & @{b6.o.n} & $@{b6.o.rate}\\%$ & $@{b6.o.approve}\\%$ & $@{b6.o.tpr_good}\\%$ & $@{b6.o.bad_approved}\\%$'],
    size='scriptsize') + items(
    T('Cut-off PD $= @{b6.cut}\\%$; differences women $-$ men: approval $@{b6.dp_sex}$ points, good payers $@{b6.eo_sex}$; older $-$ young: $@{b6.dp_age}$ and $@{b6.eo_age}$ points',
      'Pragul PD $= @{b6.cut}\\%$; diferențe femei $-$ bărbați: aprobare $@{b6.dp_sex}$ puncte procentuale, bun-platnici $@{b6.eo_sex}$; peste 25 de ani $-$ pînă la 25 de ani: $@{b6.dp_age}$ și $@{b6.eo_age}$ puncte procentuale'),
    T('Interpretation: equal opportunity: the gaps among good payers (@{b6.eo_sex} and @{b6.eo_age} points) are smaller than the gaps in approval (@{b6.dp_sex} and @{b6.dp_age}), which mostly reflect the different default rates; parity would require different cut-offs by group',
      'Interpretare: egalitatea de șanse: diferențele dintre bun-platnici (@{b6.eo_sex} și @{b6.eo_age} puncte procentuale) sînt mai mici decît diferențele de aprobare (@{b6.dp_sex} și @{b6.dp_age}), care reflectă mai ales ratele diferite de nerambursare; paritatea ar cere praguri diferite pe grupuri')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Does Gradient Boosting Beat the Logit? [Proposed]', 'C1: învinge gradient boosting logit-ul? [Propus]'),
       T('benchmarks report gains for flexible models \\refLessmann; how large is the gain on the Taiwan data, and is it worth the loss of a simple scorecard?',
         'comparațiile raportează cîștiguri pentru modelele flexibile \\refLessmann; cît de mare este cîștigul pe datele din Taiwan și merită pierderea unui scorecard simplu?'),
       T('the inputs and the split of B4; models: B3, B4', 'variabilele și împărțirea din B4; modele: B3, B4'),
       [T('Fit gradient boosting (scikit-learn, default settings) on the training sample.', 'Estimați gradient boosting (scikit-learn, setările implicite) pe eșantionul de estimare.'),
        T('Compare its test AUC with the logit by the DeLong test, and compare the Brier scores.', 'Comparați AUC de test cu cel al logit-ului prin testul DeLong și comparați scorurile Brier.'),
        T('Compare the training and the test AUC of gradient boosting.', 'Comparați AUC de estimare și de test pentru gradient boosting.'),
        T('Interpretation: is the gain large enough to replace the scorecard?', 'Interpretare: este cîștigul suficient de mare pentru a înlocui scorecard-ul?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrr', T('& test AUC & Brier & training AUC', '& AUC de test & Brier & AUC de estimare'),
    [f'gradient boosting & $@{{c1.auc_gb}}$ & $@{{c1.brier_gb}}$ & $@{{c1.auc_gb_train}}$',
     f'logit & $@{{c1.auc_logit}}$ & $@{{c1.brier_logit}}$ & $@{{b4.auc_logit}}$'], size='scriptsize') + items(
    T('DeLong: $z = @{c1.z}$, p $= @{c1.p}$: a significant but small gain (less than 0.01 in AUC); boosting also overfits more (training AUC @{c1.auc_gb_train})',
      'DeLong: $z = @{c1.z}$, p $= @{c1.p}$: un cîștig semnificativ, dar mic (sub 0,01 în AUC); boosting-ul supraajustează și mai mult (AUC de estimare @{c1.auc_gb_train})'),
    T('Project design: repeated cross-validation, cost-based and fairness measures by group, explanations of the boosting model, and the value of the gain in money (Lessmann et al.)',
      'Designul proiectului: validare încrucișată repetată, măsuri de cost și de echitate pe grupuri, explicații pentru modelul boosting și valoarea în bani a cîștigului (Lessmann et al.)')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the B1 scorecard. The answer:', 'Un student a cerut unui asistent AI să rezume scorecard-ul din B1. Răspunsul:'),
    T('\\aiprompt{(a) The AUC is @{c2.auc}, so the scorecard classifies @{c2.auc} of the applicants correctly.}', '\\aiprompt{(a) AUC este @{c2.auc}, deci scorecard-ul clasifică corect @{c2.auc} din solicitanți.}'),
    T('\\aiprompt{(b) The Gini coefficient is AUC - 0.5 = @{c2.wrong}.}', '\\aiprompt{(b) Coeficientul Gini este AUC - 0,5 = @{c2.wrong}.}'),
    T('\\aiprompt{(c) An applicant with a PD of 30\\% from this model has a 30\\% chance of default at the bank.}', '\\aiprompt{(c) Un solicitant cu PD de 30\\% din acest model are o probabilitate de 30\\% de nerambursare la bancă.}'),
    T('\\aiprompt{(d) With PDO = 20, a PD that halves from 20\\% to 10\\% adds exactly 20 points.}', '\\aiprompt{(d) Cu PDO = 20, o PD care scade la jumătate, de la 20\\% la 10\\%, adaugă exact 20 de puncte.}'),
    T('\\aiprompt{(e) KS is the largest vertical gap between the score distributions of bads and goods.}', '\\aiprompt{(e) KS este cea mai mare distanță verticală dintre distribuțiile scorurilor rău-platnicilor și bun-platnicilor.}'),
    T('\\aiprompt{(f) LDA may only be used when every variable is Normally distributed.}', '\\aiprompt{(f) LDA poate fi folosită doar cînd fiecare variabilă are distribuția Normală.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: AUC is the probability that a random bad has a higher PD than a random good; the accuracy at 0.5 is @{c2.acc}\\%', '(a) Greșit: AUC este probabilitatea ca un rău-platnic aleator să aibă o PD mai mare decît un bun-platnic aleator; acuratețea la 0,5 este @{c2.acc}\\%'),
    T('(b) Wrong: Gini $= 2\\,\\mathrm{AUC} - 1 = @{c2.gini}$', '(b) Greșit: Gini $= 2\\,\\mathrm{AUC} - 1 = @{c2.gini}$'),
    T('(c) Wrong: the sample has 30\\% bads, the bank about 5\\%; after the prior correction the PD is @{c2.pdpop}\\%', '(c) Greșit: eșantionul are 30\\% rău-platnici, banca aproximativ 5\\%; după corecția a priori, PD este @{c2.pdpop}\\%'),
    T('(d) Wrong: PDO doubles the good:bad odds, not the PD: from 4:1 to 9:1 the score rises from @{c2.s20} to @{c2.s10}, i.e.\\ @{c2.ds} points', '(d) Greșit: PDO dublează șansa bun:rău, nu PD: de la 4:1 la 9:1, scorul crește de la @{c2.s20} la @{c2.s10}, adică @{c2.ds} puncte'),
    T('(e) Correct: $\\max_s|F_{\\text{bad}}(s) - F_{\\text{good}}(s)|$', '(e) Corect: $\\max_s|F_{\\text{rău}}(s) - F_{\\text{bun}}(s)|$'),
    T('(f) Wrong: Fisher\'s criterion needs no distribution; Normality with a common covariance is needed only for correct posterior PDs (B4)', '(f) Greșit: criteriul Fisher nu cere nicio distribuție; normalitatea cu covarianță comună este necesară doar pentru PD a posteriori corecte (B4)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('A logit models the log-odds; $e^{\\beta}$ is an odds ratio; the effect on PD is $p(1 - p)\\beta$', 'Un logit modelează logaritmul șansei; $e^{\\beta}$ este un raport al șanselor; efectul asupra PD este $p(1 - p)\\beta$'),
    T('WoE and IV turn a table of goods and bads into evidence and a ranking of variables', 'WoE și IV transformă un tabel de bun-platnici și rău-platnici într-o dovadă și o ordonare a variabilelor'),
    T('AUC counts correctly ordered pairs; Gini $= 2\\,$AUC $- 1$; accuracy misleads when classes are imbalanced', 'AUC numără perechile ordonate corect; Gini $= 2\\,$AUC $- 1$; acuratețea înșală cînd clasele sînt dezechilibrate'),
    T('Choose the cut-off from the costs; correct oversampled PDs before any EL or capital', 'Alegem pragul din costuri; corectăm PD din eșantioane suprareprezentate înaintea oricărui calcul de EL sau capital'),
    T('An AI answer is a draft: check what AUC means, the Gini formula, the prior and the PDO scale', 'Un răspuns AI este o ciornă: verificați ce înseamnă AUC, formula Gini, corecția a priori și scala PDO')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 12 develops each topic of today: the Basel parameters, WoE and IV, logit and LDA, scorecards, validation, reject inference, fairness, Altman and Merton',
      'Cursul 12 dezvoltă fiecare temă de azi: parametrii Basel, WoE și IV, logit și LDA, scorecard-urile, validarea, reject inference, echitatea, Altman și Merton'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: cross-validated benchmark, costs and fairness by group', 'C1 poate deveni un proiect de echipă: comparație prin validare încrucișată, costuri și echitate pe grupuri'),
    T('Reading: \\refFHH, Ch.~21--22; \\refMVA, Ch.~14; \\refSiddiqi', 'Lectură: \\refFHH, cap.~21--22; \\refMVA, cap.~14; \\refSiddiqi')))

D.references(bib(['BCBS', 'Brier', 'Cox', 'DeLong', 'Fisher', 'FHH', 'Groemping', 'HM', 'HPS', 'HL', 'Lessmann', 'MVA', 'SGC', 'Siddiqi', 'TW']), per=16)

if __name__ == '__main__':
    D.write(V)
