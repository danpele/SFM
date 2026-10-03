r"""
build_seminar6.py -- Seminarul 6 (Selecția modelului și managementul riscului), EN + RO dintr-o singură sursă
==========================================================================================================
Seminarul are loc ÎNAINTEA cursului 6: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_06/sem6_results.json (seminar6.py).
Ieșire:
  EN/Seminars/seminar6_model_selection_risk_management.tex             (+ _solutions.tex)
  RO/Seminarii/seminar6_selectia_modelului_managementul_riscului_ro.tex (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_06/seminar6.py && python3 latex/build_seminar6.py && python3 latex/sfm_build.py compile 6
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, fig   # noqa: E402
from ch6_common import MNAME, NAMES, REFS, T, big, bib, load_sem, put_name, pval   # noqa: E402

S = load_sem()
V = Values()
D = Deck(6, 'seminar', refs=REFS)
SEM_MODELS = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG']


def qlsem():
    return '\\quantlet{SFM\\_ch6\\_seminar}{\\qlurl{SFM_ch6_seminar}}'


def key(m):
    return m.replace('-', '')


def lang_name(m, i):
    """The EN (i = 0) or RO (i = 1) name of a model, without the ⟦..||..⟧ marks."""
    v = MNAME[m]
    return v[1:-1].split('||')[i] if v.startswith('⟦') else v


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    t = A[a]['tab']
    for m, v in t.items():
        km = key(m)
        V.put(f'{a}.{km}.ll', v['ll'], 1)
        V.put(f'{a}.{km}.aic', v['aic'], 1)
        V.put(f'{a}.{km}.bic', v['bic'], 1)
        V.put(f'{a}.{km}.daic', v['daic'], 1)
        V.put(f'{a}.{km}.dbic', v['dbic'], 1)
        V.put(f'{a}.{km}.w', v['w'], 3)
        V.raw(f'{a}.{km}.k', str(v['k']))
        V.put(f'{a}.{km}.pen', v['k'] * math.log(1000), 2)
        V.put(f'{a}.{km}.e', math.exp(-0.5 * v['daic']), 3)
    V.raw(f'{a}.y0', A[a]['first'][:4])
    V.raw(f'{a}.y1', A[a]['last'][:4])
V.put('ln1000', math.log(1000), 3)
for c in ['t_skt', 'n_ged']:
    V.put(f'a3.{c}.lr', A['a3'][c]['lr'], 1)
    V.raw(f'a3.{c}.p', pval(A['a3'][c]['p']))
V.put('a3.p.t_skt', A['a3']['t_skt']['p'], 2)
V.put('a4.lr', A['a4']['n_t']['lr'], 1)
V.put('crit95', 3.841, 2)
V.put('crit90', 2.706, 2)
for a in ['a5', 'a6']:
    x = A[a]
    for i, u in enumerate(x['u_known']):
        V.put(f'{a}.u{i}', u, 3)
    for i, u in enumerate(x['u_est']):
        V.put(f'{a}.ue{i}', u, 3)
    for i, z in enumerate(x['z']):
        V.put(f'{a}.z{i}', z, 2)
    V.put(f'{a}.d', x['d_known'], 3)
    V.put(f'{a}.dest', x['d_est'], 3)
    V.put(f'{a}.mu', x['mu'], 3)
    V.put(f'{a}.sd', x['sd'], 3)
    V.put(f'{a}.crit', x['crit_known'], 3)
    V.put(f'{a}.lil', x['crit_lillie'], 3)
A7, A8 = A['a7'], A['a8']
V.put('a7.z', -A7['z'], 3)
V.put('a7.vn', A7['var_n'], 3)
V.put('a7.tq', -A7['tq'], 3)
V.put('a7.vt', A7['var_t'], 3)
V.put('a7.lr', A7['kup20']['lr'], 2)
V.put('a7.p', A7['kup20']['p'], 4)
V.put('a7.pge', A7['p_ge20'], 4)
V.put('a8.lr', A8['kup4']['lr'], 2)
V.put('a8.p', A8['kup4']['p'], 3)
V.put('a8.ple', A8['p_le4'], 3)
V.put('a8.es', A8['es_n'], 3)
V.put('a8.phi', A8['phi'], 4)
V.put('a8.z', -A8['z25'], 3)
# a7: termenii testului Kupiec, x = 20
V.put('a7.l0', 980 * math.log(0.99) + 20 * math.log(0.01), 2)
V.put('a7.l1', 980 * math.log(0.98) + 20 * math.log(0.02), 2)
V.put('a8.l0', 996 * math.log(0.99) + 4 * math.log(0.01), 2)
V.put('a8.l1', 996 * math.log(0.996) + 4 * math.log(0.004), 2)


def put_ic(prefix, b):
    for m, v in b['tab'].items():
        km = key(m)
        V.raw(f'{prefix}.{km}.ll', big(v['ll']))
        V.raw(f'{prefix}.{km}.aic', big(v['aic']))
        V.raw(f'{prefix}.{km}.bic', big(v['bic']))
        V.put(f'{prefix}.{km}.daic', v['daic'], 1)
        V.put(f'{prefix}.{km}.dbic', v['dbic'], 1)
        V.put(f'{prefix}.{km}.w', v['w'], 3)
    V.int(f'{prefix}.n', b['n'])
    V.raw(f'{prefix}.y0', b['first'][:4])
    put_name(V, f'{prefix}.best', b['best_aic'])
    put_name(V, f'{prefix}.bestb', b['best_bic'])
    V.put(f'{prefix}.nu', b['nu'], 2)
    V.put(f'{prefix}.lam', b['lam'], 3)
    V.put(f'{prefix}.beta', b['beta'], 2)
    V.put(f'{prefix}.lr', b['lr_t_skt']['lr'], 2)
    V.raw(f'{prefix}.lrp', pval(b['lr_t_skt']['p']))
    V.raw(f'{prefix}.lrged', big(b['lr_n_ged']['lr']))


put_ic('b1', S['B1'])
for k in ['dax', 'btc']:
    put_ic(f'b2.{k}', S['B2'][k])
B3 = S['B3']
V.int('b3.n', B3['n'])
for m in ['Normal', 'Student-t']:
    km = key(m)
    for s_ in ['ks', 'cvm', 'ad']:
        V.put(f'b3.{km}.{s_}', B3[m]['stat'][s_], 3 if s_ == 'ks' else 2)
        V.put(f'b3.{km}.{s_}.c', B3[m]['crit'][s_], 3 if s_ == 'ks' else 2)
        V.put(f'b3.{km}.{s_}.p', B3[m]['p_boot'][s_], 2)
    pn = B3[m]['p_naive']
    V.raw(f'b3.{km}.naive', pval(pn) if pn >= 1e-3 else '<⁅0.001⁆')
V.put('b3.asy', 1.358 / math.sqrt(B3['n']), 3)
B4 = S['B4']
for i, t in enumerate(B4['top']):
    V.raw(f'b4.top{i}.d', t['date'])
    V.put(f'b4.top{i}.r', t['r'], 1)
for d_, v in B4['prices'].items():
    V.put(f'b4.c.{d_.replace("-", "")}', v['close'], 2)
    V.put(f'b4.a.{d_.replace("-", "")}', v['adj'], 2)
V.put('b4.bet', B4['bet_20181219'], 1)
for lab in ['raw', 'clean']:
    b = B4[lab]
    V.int(f'b4.{lab}.n', b['n'])
    V.put(f'b4.{lab}.kurt', b['kurt'], 1)
    V.put(f'b4.{lab}.sd', b['sd'], 3)
    V.put(f'b4.{lab}.nu', b['nu'], 2)
    V.put(f'b4.{lab}.ad', b['ad_t'], 2)
    V.put(f'b4.{lab}.var', b['var1_t'], 2)
    put_name(V, f'b4.{lab}.best', b['best_aic'])
    for m, v in b['tab'].items():
        V.put(f'b4.{lab}.{key(m)}.daic', v['daic'], 1)


def put_oos(prefix, b):
    V.int(f'{prefix}.nest', b['n_est'])
    V.int(f'{prefix}.ntest', b['n_test'])
    V.put(f'{prefix}.exp', b['n_test'] / 100, 1)
    V.raw(f'{prefix}.y0', b['first'][:4])
    put_name(V, f'{prefix}.baic', b['best_aic'])
    put_name(V, f'{prefix}.bls', b['best_logscore'])
    put_name(V, f'{prefix}.bpin', b['best_pinball'])
    V.put(f'{prefix}.tv', b['test_var1'], 2)
    for m in SEM_MODELS + ['Historical', 'Averaged']:
        km = key(m)
        v = b[m]
        V.put(f'{prefix}.{km}.var', v['var1'], 2)
        V.raw(f'{prefix}.{km}.x', str(v['kupiec']['x']))
        V.raw(f'{prefix}.{km}.p', pval(v['kupiec']['p']))
        V.put(f'{prefix}.{km}.pin', 100 * v['pinball'], 2)
        if 'logscore' in v:
            V.put(f'{prefix}.{km}.ls', v['logscore'], 3)
    for m, v in b['tab'].items():
        V.put(f'{prefix}.{key(m)}.w', v['w'], 3)


put_oos('b5', S['B5'])
put_oos('b6', S['B6'])
C1 = S['C1']
c1rows = {k: [f'{p_.replace("-", "--")} & {MNAME[r["best"]]} & $⁅{r["w"]:.2f}⁆$ & {MNAME[r["second"]]} & $⁅{r["gap"]:.1f}⁆$' for p_, r in per.items()]
          for k, per in C1.items()}
nwin = {k: {} for k in C1}
for k, per in C1.items():
    for r in per.values():
        nwin[k][r['best']] = nwin[k].get(r['best'], 0) + 1
V.raw('c1.sp.ged', str(nwin['sp500'].get('GED', 0)))
V.raw('c1.bet.t', str(nwin['bet'].get('Student-t', 0)))
V.raw('c1.nper', str(len(C1['sp500'])))
C2 = S['C2']
V.int('c2.n', C2['n'])
V.put('c2.lnn', C2['lnn'], 2)
V.put('c2.ks', C2['ks_t'], 3)
V.put('c2.pn', C2['p_naive_t'], 3)
V.put('c2.vn', C2['var_n'], 2)
V.put('c2.vt', C2['var_t'], 2)
V.put('c2.vs', C2['var_skt'], 2)
V.put('c2.emp', C2['emp'], 2)
V.put('c2.nu', C2['nu'], 2)
V.raw('c2.lr', big(C2['lr_n_t']['lr']))
for m, v in C2['tab'].items():
    V.put(f'c2.{key(m)}.daic', v['daic'], 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: among several candidate distributions for daily returns, which one should we choose, and how do we know it is good enough for VaR 1\\%?',
       '\\textbf{Întrebarea}: dintre mai multe distribuții candidate pentru randamentele zilnice, pe care o alegem și de unde știm că este suficient de bună pentru VaR 1\\%?'),
     [T('this seminar comes \\textbf{before} Lecture 6: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 6: secțiunea „Noțiuni necesare azi” conține toate definițiile necesare pentru cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: AIC, BIC, Akaike weights, likelihood-ratio tests, a KS test and VaR exceedances on paper',
        'Partea A: AIC, BIC, ponderi Akaike, teste ale raportului de verosimilitate, un test KS și depășiri VaR, pe hîrtie'),
      T('Part B: model choice on BET, S\\&P 500, DAX, Bitcoin and Banca Transilvania, each with an interpretation question',
        'Partea B: alegerea modelului pe BET, S\\&P 500, DAX, Bitcoin și Banca Transilvania, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('which model has the lowest AIC and BIC, and how strong is the evidence?', 'ce model are cel mai mic AIC și BIC și cît de puternice sînt dovezile?') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('is the larger of two nested models significantly better?', 'este modelul mai general dintre două modele imbricate semnificativ mai bun?') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('does a small sample pass a KS test, with known and with estimated parameters?', 'trece un eșantion mic un test KS, cu parametri cunoscuți și cu parametri estimați?') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('VaR 1\\% under two models; are the exceedances too many or too few?', 'VaR 1\\% pentru două modele; sînt depășirile prea multe sau prea puține?') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('which distribution fits BET, DAX and Bitcoin returns best?', 'care distribuție se potrivește cel mai bine randamentelor BET, DAX și Bitcoin?') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('do the best models pass a goodness-of-fit test? are two TLV days data errors?', 'trec cele mai bune modele testele de concordanță? sînt două zile din seria TLV erori de date?') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('which model gets VaR 1\\% right out of sample?', 'ce model estimează corect VaR 1\\% în afara eșantionului?') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('does the best model change over time? what is wrong in an AI answer?', 'se schimbă în timp cel mai bun model? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500, DAX & EODHD & ' + T('close', 'închidere') + ' & @{b1.y0}--@{a1.y1}',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'închidere, 7 zile pe săptămînă') + ' & @{b2.btc.y0}--@{a1.y1}',
     T('Banca Transilvania (TLV)', 'Banca Transilvania (TLV)') + ' & EODHD & ' + T('adjusted close', 'închidere ajustată') + ' & 2010--@{a1.y1}'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin)',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin)'),
    T('Six candidate models: Normal, Student-t, skewed-t, GED, two-component Normal mixture, NIG', 'Șase modele candidate: Normală, Student-t, skewed-t, GED, amestec Normal cu două componente, NIG'),
    T('In the notebook: \\texttt{returns(\'bet\')}, \\texttt{fit\\_model(\'NIG\', x)}; no account or key is needed',
      'În notebook: \\texttt{returns(\'bet\')}, \\texttt{fit\\_model(\'NIG\', x)}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/5): Likelihood and the Candidates', 'Noțiuni necesare azi (1/5): verosimilitatea și candidații'), items(
    (T('Returns $r_1, \\dots, r_n$ i.i.d.\\ (independent, identically distributed) with density $f(r; \\theta)$', 'Randamentele $r_1, \\dots, r_n$ i.i.d.\\ (independente și identic distribuite), cu densitatea $f(r; \\theta)$'),
     [T('log-likelihood $\\ell(\\theta) = \\sum_t \\ln f(r_t; \\theta)$; the ML (maximum likelihood) estimate $\\hat\\theta$ maximises it',
        'log-verosimilitatea $\\ell(\\theta) = \\sum_t \\ln f(r_t; \\theta)$; estimarea ML (maximum likelihood, verosimilitate maximă) $\\hat\\theta$ o maximizează'),
      T('Normal: $\\ell(\\hat\\theta) = -\\frac{n}{2}[\\ln(2\\pi\\hat\\sigma^2) + 1]$ with $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t (r_t - \\bar r)^2$', 'Normală: $\\ell(\\hat\\theta) = -\\frac{n}{2}[\\ln(2\\pi\\hat\\sigma^2) + 1]$ cu $\\hat\\sigma^2 = \\frac{1}{n}\\sum_t (r_t - \\bar r)^2$')]),
    (T('Candidates and their number of parameters $k$', 'Candidații și numărul lor de parametri $k$'),
     [T('Normal (2); Student-t (3: $\\nu$, location, scale); skewed-t (4: adds the skewness $\\lambda$, $\\lambda = 0$ gives the Student-t)',
        'Normală (2); Student-t (3: $\\nu$, locație, scală); skewed-t (4: adaugă asimetria $\\lambda$; $\\lambda = 0$ dă Student-t)'),
      T('GED, generalised error distribution (3: shape $\\beta$, $\\beta = 2$ gives the Normal law); Normal mixture of two components (5); NIG, normal inverse Gaussian (4)',
        'GED, generalised error distribution (distribuția erorii generalizate) (3: forma $\\beta$; $\\beta = 2$ dă distribuția Normală); amestec Normal cu două componente (5); NIG, normal inverse Gaussian (4)')]),
    T('The heavier the tails of the data, the more the Normal model loses in $\\ell$', 'Cu cît cozile datelor sînt mai groase, cu atît modelul Normal pierde mai mult în $\\ell$')))

D.frame(T('What You Need for Today (2/5): AIC, BIC, Akaike Weights', 'Noțiuni necesare azi (2/5): AIC, BIC, ponderi Akaike'), items(
    (T('$\\mathrm{AIC} = -2\\ell(\\hat\\theta) + 2k$ (Akaike information criterion); $\\mathrm{BIC} = -2\\ell(\\hat\\theta) + k\\ln n$ (Bayesian information criterion)',
       '$\\mathrm{AIC} = -2\\ell(\\hat\\theta) + 2k$ (Akaike information criterion, criteriul informațional Akaike); $\\mathrm{BIC} = -2\\ell(\\hat\\theta) + k\\ln n$ (Bayesian information criterion, criteriul informațional bayesian)'),
     [T('lower is better; only differences on the same data matter; BIC penalises parameters more when $\\ln n > 2$, i.e.\\ $n > 7$',
        'valoarea mai mică este preferată; contează doar diferențele calculate pe aceleași date; BIC penalizează mai mult parametrii cînd $\\ln n > 2$, adică $n > 7$')]),
    (T('$\\Delta_i = \\mathrm{AIC}_i - \\min_j \\mathrm{AIC}_j$; Akaike weight $w_i = e^{-\\Delta_i/2}/\\sum_j e^{-\\Delta_j/2}$', '$\\Delta_i = \\mathrm{AIC}_i - \\min_j \\mathrm{AIC}_j$; ponderea Akaike $w_i = e^{-\\Delta_i/2}/\\sum_j e^{-\\Delta_j/2}$'),
     [T('rules of thumb \\refBA: $\\Delta \\le 2$ substantial support; $4$--$7$ considerably less; $> 10$ essentially none',
        'reguli practice \\refBA: $\\Delta \\le 2$ sprijin substanțial; $4$--$7$ considerabil mai puțin; $> 10$ practic niciunul')]),
    T('AIC is not a test: there is no $p$-value and no ``significance\'\'', 'AIC nu este un test: nu există valoare $p$ și nici „semnificație”')))

D.frame(T('What You Need for Today (3/5): the Likelihood-Ratio Test', 'Noțiuni necesare azi (3/5): testul raportului de verosimilitate'), items(
    (T('Model 0 is \\textbf{nested} in model 1 if it is model 1 with some parameters fixed (Student-t = skewed-t with $\\lambda = 0$)',
       'Modelul 0 este \\textbf{imbricat} în modelul 1 dacă este modelul 1 cu unii parametri fixați (Student-t = skewed-t cu $\\lambda = 0$)'),
     [T('$LR = 2[\\ell_1(\\hat\\theta_1) - \\ell_0(\\hat\\theta_0)] \\approx \\chi^2(k_1 - k_0)$ under $H_0$ (Wilks)', '$LR = 2[\\ell_1(\\hat\\theta_1) - \\ell_0(\\hat\\theta_0)] \\approx \\chi^2(k_1 - k_0)$ sub $H_0$ (Wilks)'),
      T('5\\% critical values: $\\chi^2_{0.95}(1) = @{crit95}$, $\\chi^2_{0.95}(2) = 5.99$', 'valori critice la 5\\%: $\\chi^2_{0{,}95}(1) = @{crit95}$; $\\chi^2_{0{,}95}(2) = 5{,}99$')]),
    (T('\\textbf{Boundary}: Normal inside Student-t means $1/\\nu = 0$, the edge of the parameter space', '\\textbf{Frontiera}: distribuția Normală ca submodel al Student-t înseamnă $1/\\nu = 0$, marginea spațiului parametrilor'),
     [T('then $LR$ follows a 50:50 mixture of 0 and $\\chi^2(1)$ \\refSL: 5\\% critical value $@{crit90}$, $p$-value = half of the $\\chi^2(1)$ one',
        'atunci $LR$ urmează un amestec 50:50 de 0 și $\\chi^2(1)$ \\refSL: valoarea critică la 5\\% $@{crit90}$, valoarea $p$ = jumătate din cea calculată cu $\\chi^2(1)$')]),
    T('Non-nested pairs (Student-t vs GED): no LR test; use AIC or the Vuong test', 'Perechi neimbricate (Student-t și GED): testul LR nu se aplică; folosiți AIC sau testul Vuong')))

D.frame(T('What You Need for Today (4/5): Goodness-of-Fit Tests', 'Noțiuni necesare azi (4/5): teste de concordanță'), items(
    (T('EDF (empirical distribution function) $F_n(x) = \\frac{1}{n}\\#\\{t: r_t \\le x\\}$; model distribution function $F$', 'EDF (empirical distribution function, funcția de repartiție empirică) $F_n(x) = \\frac{1}{n}\\#\\{t: r_t \\le x\\}$; funcția de repartiție a modelului $F$'),
     [T('\\textbf{KS} (Kolmogorov--Smirnov): $D_n = \\max_i \\max\\{i/n - u_{(i)},\\ u_{(i)} - (i-1)/n\\}$ with $u_{(i)} = F(r_{(i)})$ on the sorted data',
        '\\textbf{KS} (Kolmogorov--Smirnov): $D_n = \\max_i \\max\\{i/n - u_{(i)},\\ u_{(i)} - (i-1)/n\\}$ cu $u_{(i)} = F(r_{(i)})$ pe datele ordonate'),
      T('\\textbf{CvM} (Cramér--von Mises) and \\textbf{AD} (Anderson--Darling) integrate the squared gap; AD weights it by $1/[F(1-F)]$: it looks at the tails',
        '\\textbf{CvM} (Cramér--von Mises) și \\textbf{AD} (Anderson--Darling) integrează pătratul distanței; AD o ponderează cu $1/[F(1-F)]$, deci pune accentul pe cozi')]),
    (T('Critical values of KS at 5\\%', 'Valori critice KS la 5\\%'),
     [T('$F$ fully known: $n = 5$: $@{a5.crit}$; $n = 6$: $@{a6.crit}$; large $n$: $1.358/\\sqrt{n}$', '$F$ complet cunoscută: $n = 5$: $@{a5.crit}$; $n = 6$: $@{a6.crit}$; $n$ mare: $1{,}358/\\sqrt{n}$'),
      T('mean and variance estimated from the same data (Lilliefors): $n = 5$: $@{a5.lil}$; $n = 6$: $@{a6.lil}$; other models: parametric bootstrap',
        'media și varianța estimate din aceleași date (Lilliefors): $n = 5$: $@{a5.lil}$; $n = 6$: $@{a6.lil}$; alte modele: bootstrap parametric')]),
    T('A QQ plot puts the sorted data against the model quantiles $F^{-1}((i - 0.5)/n)$: tails that bend away from the line are wrong',
      'Un grafic QQ reprezintă datele ordonate în funcție de cuantilele modelului $F^{-1}((i - 0{,}5)/n)$: punctele din cozi care se îndepărtează de dreaptă arată că modelul greșește în cozi')))

D.frame(T('What You Need for Today (5/5): VaR, Exceedances, Out of Sample', 'Noțiuni necesare azi (5/5): VaR, depășiri, în afara eșantionului'), items(
    (T('$\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, the daily loss exceeded with probability 1\\%; Normal: $\\mathrm{VaR}_{1\\%} = -(\\mu - @{a7.z}\\sigma)$',
       '$\\mathrm{VaR}_{1\\%} = -q_{1\\%}$, pierderea zilnică depășită cu probabilitatea 1\\%; Normală: $\\mathrm{VaR}_{1\\%} = -(\\mu - @{a7.z}\\sigma)$'),
     [T('ES 2.5\\% (expected shortfall): the average loss beyond $q_{2.5\\%}$; Normal: $-\\mu + \\sigma\\varphi(z_{0.025})/0.025$', 'ES 2,5\\% (expected shortfall): pierderea medie dincolo de $q_{2{,}5\\%}$; Normală: $-\\mu + \\sigma\\varphi(z_{0{,}025})/0{,}025$')]),
    (T('\\textbf{Exceedance}: a day with $r_t < -\\mathrm{VaR}$; under a correct model $x \\sim \\mathrm{Binomial}(n, 0.01)$', '\\textbf{Depășire}: o zi cu $r_t < -\\mathrm{VaR}$; pentru un model corect, $x \\sim \\mathrm{Binomial}(n; 0{,}01)$'),
     [T('Kupiec test: $LR_{uc} = -2[\\ell_0 - \\ell_1]$, $\\ell_0 = (n - x)\\ln 0.99 + x\\ln 0.01$, $\\ell_1 = (n - x)\\ln(1 - \\hat p) + x\\ln\\hat p$, $\\hat p = x/n$; compare with $\\chi^2(1)$',
        'testul Kupiec: $LR_{uc} = -2[\\ell_0 - \\ell_1]$, $\\ell_0 = (n - x)\\ln 0{,}99 + x\\ln 0{,}01$, $\\ell_1 = (n - x)\\ln(1 - \\hat p) + x\\ln\\hat p$, $\\hat p = x/n$; comparăm cu $\\chi^2(1)$')]),
    (T('\\textbf{Out of sample}: fit until a date, evaluate afterwards', '\\textbf{În afara eșantionului}: estimăm pe datele de pînă la un moment dat și evaluăm pe datele ulterioare'),
     [T('log score: mean of $\\ln f(r_t; \\hat\\theta)$ on the test days (higher is better); pinball loss of the 1\\% quantile $q$: mean of $(0.01 - \\mathbf{1}\\{r_t < q\\})(r_t - q)$ (lower is better)',
        'scorul logaritmic: media lui $\\ln f(r_t; \\hat\\theta)$ pe zilele de test (valoarea mai mare este preferată); pierderea pinball a cuantilei de 1\\% $q$: media lui $(0{,}01 - \\mathbf{1}\\{r_t < q\\})(r_t - q)$ (valoarea mai mică este preferată)')])), size='footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')


def ic_list(a, models):
    return [T(f'{lang_name(m, 0)}: $\\ell = @{{{a}.{key(m)}.ll}}$, $k = @{{{a}.{key(m)}.k}}$', f'{lang_name(m, 1)}: $\\ell = @{{{a}.{key(m)}.ll}}$, $k = @{{{a}.{key(m)}.k}}$') for m in models]


A1M = ['Normal', 'Student-t', 'Skewed-t', 'GED']
A2M = ['Normal', 'Student-t', 'NIG', 'Mixture']
D.solved(T('A1: Four Models for the S\\&P 500', 'A1: patru modele pentru S\\&P 500'),
         items(T('S\\&P 500, the last $n = 1000$ trading days (@{a1.y0}--@{a1.y1}); maximised log-likelihoods:', 'S\\&P 500, ultimele $n = 1000$ de zile de tranzacționare (@{a1.y0}--@{a1.y1}); log-verosimilitățile maximizate:'),
               *ic_list('a1', A1M),
               T('1. Compute AIC and BIC ($\\ln 1000 = @{ln1000}$) of each model.', '1. Calculați AIC și BIC ($\\ln 1000 = @{ln1000}$) pentru fiecare model.'),
               T('2. Compute $\\Delta$AIC and the Akaike weights.', '2. Calculați $\\Delta$AIC și ponderile Akaike.'),
               T('3. Say which model each criterion selects and how strong the evidence is.', '3. Precizați ce model alege fiecare criteriu și cît de puternice sînt dovezile.'),
               T('Report: a table with four rows and two sentences.', 'Raportați: un tabel cu patru rînduri și două fraze.')),
         items(T('1. AIC: Normal $@{a1.Normal.aic}$, Student-t $@{a1.Studentt.aic}$, skewed-t $@{a1.Skewedt.aic}$, GED $@{a1.GED.aic}$; BIC: $@{a1.Normal.bic}$, $@{a1.Studentt.bic}$, $@{a1.Skewedt.bic}$, $@{a1.GED.bic}$',
                 '1. AIC: Normală $@{a1.Normal.aic}$; Student-t $@{a1.Studentt.aic}$; skewed-t $@{a1.Skewedt.aic}$; GED $@{a1.GED.aic}$. BIC, în aceeași ordine: $@{a1.Normal.bic}$; $@{a1.Studentt.bic}$; $@{a1.Skewedt.bic}$; $@{a1.GED.bic}$'),
               T('2. $\\Delta$AIC: $@{a1.Normal.daic}$, $@{a1.Studentt.daic}$, $@{a1.Skewedt.daic}$, $@{a1.GED.daic}$; $e^{-\\Delta/2}$: $@{a1.Normal.e}$, $@{a1.Studentt.e}$, $@{a1.Skewedt.e}$, $@{a1.GED.e}$; weights $@{a1.Normal.w}$, $@{a1.Studentt.w}$, $@{a1.Skewedt.w}$, $@{a1.GED.w}$',
                 '2. $\\Delta$AIC: $@{a1.Normal.daic}$; $@{a1.Studentt.daic}$; $@{a1.Skewedt.daic}$; $@{a1.GED.daic}$. $e^{-\\Delta/2}$: $@{a1.Normal.e}$; $@{a1.Studentt.e}$; $@{a1.Skewedt.e}$; $@{a1.GED.e}$. Ponderi: $@{a1.Normal.w}$; $@{a1.Studentt.w}$; $@{a1.Skewedt.w}$; $@{a1.GED.w}$'),
               T('3. AIC: Student-t and skewed-t tied ($\\Delta = @{a1.Skewedt.daic}$); BIC: Student-t, by $@{a1.Skewedt.dbic}$ points; Normal and GED have no support.',
                 '3. AIC: Student-t și skewed-t la egalitate ($\\Delta = @{a1.Skewedt.daic}$); BIC: Student-t, cu un avans de $@{a1.Skewedt.dbic}$ puncte; modelul Normal și GED nu au niciun sprijin.')),
         size='scriptsize')

D.proposed(T('A2: Four Models for Bitcoin', 'A2: patru modele pentru Bitcoin'),
           items(T('Bitcoin, the last $n = 1000$ days (@{a2.y0}--@{a2.y1}); maximised log-likelihoods: Model: A1.', 'Bitcoin, ultimele $n = 1000$ de zile (@{a2.y0}--@{a2.y1}); log-verosimilitățile maximizate: Model: A1.'),
                 *ic_list('a2', A2M),
                 T('1. Compute AIC, BIC, $\\Delta$AIC and the Akaike weights.', '1. Calculați AIC, BIC, $\\Delta$AIC și ponderile Akaike.'),
                 T('2. Say whether AIC and BIC select the same model.', '2. Precizați dacă AIC și BIC aleg același model.'),
                 T('3. Explain why the Normal mixture, with the most parameters, does not win.', '3. Explicați de ce amestecul Normal, cu cei mai mulți parametri, nu cîștigă.'),
                 T('Report: a table with four rows and two sentences.', 'Raportați: un tabel cu patru rînduri și două fraze.')),
           items(T('1. AIC: $@{a2.Normal.aic}$, $@{a2.Studentt.aic}$, $@{a2.NIG.aic}$, $@{a2.Mixture.aic}$; BIC: $@{a2.Normal.bic}$, $@{a2.Studentt.bic}$, $@{a2.NIG.bic}$, $@{a2.Mixture.bic}$; weights $@{a2.Normal.w}$, $@{a2.Studentt.w}$, $@{a2.NIG.w}$, $@{a2.Mixture.w}$',
                   '1. În ordinea Normală, Student-t, NIG, amestec Normal. AIC: $@{a2.Normal.aic}$; $@{a2.Studentt.aic}$; $@{a2.NIG.aic}$; $@{a2.Mixture.aic}$. BIC: $@{a2.Normal.bic}$; $@{a2.Studentt.bic}$; $@{a2.NIG.bic}$; $@{a2.Mixture.bic}$. Ponderi: $@{a2.Normal.w}$; $@{a2.Studentt.w}$; $@{a2.NIG.w}$; $@{a2.Mixture.w}$'),
                 T('2. AIC: NIG ($\\Delta$ of the Student-t $@{a2.Studentt.daic}$); BIC: NIG and Student-t almost tied ($\\Delta$BIC $@{a2.Studentt.dbic}$).',
                   '2. AIC: NIG ($\\Delta$ al Student-t $@{a2.Studentt.daic}$); BIC: NIG și Student-t aproape la egalitate ($\\Delta$BIC $@{a2.Studentt.dbic}$).'),
                 T('3. Its $\\ell$ is no higher than that of the Student-t, and it pays for two more parameters: a Normal mixture has Normal (thin) tails.',
                   '3. Log-verosimilitatea sa nu o depășește pe cea a modelului Student-t, iar modelul este penalizat pentru doi parametri în plus: un amestec Normal are cozi Normale (subțiri).')),
           size='scriptsize')

D.solved(T('A3: Is Skewness Needed? Is the Normal Law Enough?', 'A3: este necesară asimetria? Este suficientă distribuția Normală?'),
         items(T('Use the log-likelihoods of A1 (S\\&P 500, $n = 1000$).', 'Folosiți log-verosimilitățile din A1 (S\\&P 500, $n = 1000$).'),
               T('1. Test the Student-t ($H_0$: $\\lambda = 0$) against the skewed-t with a likelihood-ratio test at 5\\%.', '1. Testați Student-t ($H_0$: $\\lambda = 0$) față de skewed-t cu un test al raportului de verosimilitate la 5\\%.'),
               T('2. Test the Normal law ($H_0$: $\\beta = 2$) against the GED.', '2. Testați distribuția Normală ($H_0$: $\\beta = 2$) față de GED.'),
               T('3. Compare the LR decision of step 1 with the AIC verdict of A1.', '3. Comparați decizia LR de la pasul 1 cu verdictul AIC din A1.'),
               T('Report: two $LR$ values, two decisions and one sentence.', 'Raportați: două valori $LR$, două decizii și o frază.')),
         items(T('1. $LR = 2(@{a1.Skewedt.ll} - (@{a1.Studentt.ll})) = @{a3.t_skt.lr} < @{crit95}$, $p = @{a3.p.t_skt}$: do not reject $\\lambda = 0$',
                 '1. $LR = 2(@{a1.Skewedt.ll} - (@{a1.Studentt.ll})) = @{a3.t_skt.lr} < @{crit95}$, $p = @{a3.p.t_skt}$: nu respingem $\\lambda = 0$'),
               T('2. $LR = 2(@{a1.GED.ll} - (@{a1.Normal.ll})) = @{a3.n_ged.lr}$, $p @{a3.n_ged.p}$: reject the Normal law', '2. $LR = 2(@{a1.GED.ll} - (@{a1.Normal.ll})) = @{a3.n_ged.lr}$, $p @{a3.n_ged.p}$: respingem distribuția Normală'),
               T('3. Consistent: AIC finds the two t-models tied; the LR test finds no significant gain from $\\lambda$. For nested pairs, AIC prefers the larger model exactly when $LR > 2$ per extra parameter.',
                 '3. Rezultatele sînt concordante: după AIC, cele două modele t sînt la egalitate, iar testul LR nu arată un cîștig semnificativ adus de $\\lambda$. Pentru perechi imbricate, AIC preferă modelul mai general exact atunci cînd $LR > 2$ pentru fiecare parametru în plus.')),
         size='scriptsize')

D.proposed(T('A4: A Test on the Boundary', 'A4: un test la frontieră'),
           items(T('Use the log-likelihoods of A2 (Bitcoin, $n = 1000$). Model: A3.', 'Folosiți log-verosimilitățile din A2 (Bitcoin, $n = 1000$). Model: A3.'),
                 T('1. Compute $LR$ for the Normal law inside the Student-t.', '1. Calculați $LR$ pentru distribuția Normală ca submodel al Student-t.'),
                 T('2. Explain why the null law is not $\\chi^2(1)$, and give the correct 5\\% critical value.', '2. Explicați de ce distribuția statisticii sub $H_0$ nu este $\\chi^2(1)$ și dați valoarea critică corectă la 5\\%.'),
                 T('3. Say whether an LR test can compare the Student-t with the NIG.', '3. Spuneți dacă un test LR poate compara Student-t cu NIG.'),
                 T('Report: one $LR$ value, one critical value and two sentences.', 'Raportați: o valoare $LR$, o valoare critică și două fraze.')),
           items(T('1. $LR = 2(@{a2.Studentt.ll} - (@{a2.Normal.ll})) = @{a4.lr}$', '1. $LR = 2(@{a2.Studentt.ll} - (@{a2.Normal.ll})) = @{a4.lr}$'),
                 T('2. $H_0$ is $1/\\nu = 0$, on the edge of $1/\\nu \\ge 0$: the law is a 50:50 mixture of 0 and $\\chi^2(1)$, critical value $\\chi^2_{0.90}(1) = @{crit90}$; reject the Normal law either way.',
                   '2. $H_0$ este $1/\\nu = 0$, pe frontiera mulțimii $1/\\nu \\ge 0$: distribuția statisticii este un amestec 50:50 de 0 și $\\chi^2(1)$, cu valoarea critică $\\chi^2_{0{,}90}(1) = @{crit90}$; respingem distribuția Normală cu oricare dintre cele două valori critice.'),
                 T('3. No: neither model is a special case of the other; use AIC or the Vuong test.', '3. Nu: niciunul nu este caz particular al celuilalt; folosiți AIC sau testul Vuong.')),
           size='scriptsize')

D.solved(T('A5: KS on Five Returns', 'A5: KS pe cinci randamente'),
         items(T('Daily returns (\\%): $-2.1$, $-0.4$, $0.3$, $0.9$, $1.6$.', 'Randamente zilnice (\\%): $-2{,}1$; $-0{,}4$; $0{,}3$; $0{,}9$; $1{,}6$.'),
               T('1. Compute $D_5$ against $N(0, 1)$ and compare it with the 5\\% critical value $@{a5.crit}$.', '1. Calculați $D_5$ față de $N(0, 1)$ și comparați-l cu valoarea critică la 5\\% $@{a5.crit}$.'),
               T('2. Estimate $\\hat\\mu$ and $\\hat\\sigma$ (divisor $n$), compute $D_5$ against $N(\\hat\\mu, \\hat\\sigma^2)$ and compare it with the Lilliefors value $@{a5.lil}$.',
                 '2. Estimați $\\hat\\mu$ și $\\hat\\sigma$ (numitorul $n$), calculați $D_5$ față de $N(\\hat\\mu, \\hat\\sigma^2)$ și comparați-l cu valoarea Lilliefors $@{a5.lil}$.'),
               T('3. Explain why the Lilliefors critical value is smaller.', '3. Explicați de ce valoarea critică Lilliefors este mai mică.'),
               T('Report: two values of $D_5$, two decisions, one sentence.', 'Raportați: două valori $D_5$, două decizii, o frază.')),
         items(T('1. $u_{(i)} = \\Phi(r_{(i)})$: $@{a5.u0}$, $@{a5.u1}$, $@{a5.u2}$, $@{a5.u3}$, $@{a5.u4}$; the largest gap $D_5 = @{a5.d} < @{a5.crit}$: do not reject',
                 '1. $u_{(i)} = \\Phi(r_{(i)})$: $@{a5.u0}$; $@{a5.u1}$; $@{a5.u2}$; $@{a5.u3}$; $@{a5.u4}$; cea mai mare distanță $D_5 = @{a5.d} < @{a5.crit}$: nu respingem'),
               T('2. $\\hat\\mu = @{a5.mu}$, $\\hat\\sigma = @{a5.sd}$; $z_{(i)}$: $@{a5.z0}$, $@{a5.z1}$, $@{a5.z2}$, $@{a5.z3}$, $@{a5.z4}$; $u_{(i)}$: $@{a5.ue0}$, $@{a5.ue1}$, $@{a5.ue2}$, $@{a5.ue3}$, $@{a5.ue4}$; $D_5 = @{a5.dest} < @{a5.lil}$: do not reject',
                 '2. $\\hat\\mu = @{a5.mu}$, $\\hat\\sigma = @{a5.sd}$; $z_{(i)}$: $@{a5.z0}$; $@{a5.z1}$; $@{a5.z2}$; $@{a5.z3}$; $@{a5.z4}$; $u_{(i)}$: $@{a5.ue0}$; $@{a5.ue1}$; $@{a5.ue2}$; $@{a5.ue3}$; $@{a5.ue4}$; $D_5 = @{a5.dest} < @{a5.lil}$: nu respingem'),
               T('3. The fitted $F$ is pulled towards the data, so $D_n$ is smaller under $H_0$; the critical value must be smaller too.',
                 '3. $F$ estimată este apropiată de date, deci $D_n$ este mai mic sub $H_0$; prin urmare, și valoarea critică trebuie să fie mai mică.')),
         size='scriptsize')

D.proposed(T('A6: KS with One Large Loss', 'A6: KS cu o pierdere mare'),
           items(T('Daily returns (\\%): $-3.0$, $-0.6$, $-0.2$, $0.1$, $0.5$, $1.0$. Model: A5.', 'Randamente zilnice (\\%): $-3{,}0$; $-0{,}6$; $-0{,}2$; $0{,}1$; $0{,}5$; $1{,}0$. Model: A5.'),
                 T('1. Compute $D_6$ against $N(0, 1)$; the 5\\% critical value is $@{a6.crit}$.', '1. Calculați $D_6$ față de $N(0, 1)$; valoarea critică la 5\\% este $@{a6.crit}$.'),
                 T('2. Compute $D_6$ against $N(\\hat\\mu, \\hat\\sigma^2)$; the Lilliefors value is $@{a6.lil}$.', '2. Calculați $D_6$ față de $N(\\hat\\mu, \\hat\\sigma^2)$; valoarea Lilliefors este $@{a6.lil}$.'),
                 T('3. The $-3\\%$ day has $\\Phi(-3) = 0.001$; explain why KS hardly reacts to it, while AD would.', '3. Ziua de $-3\\%$ are $\\Phi(-3) = 0{,}001$; explicați de ce KS aproape nu reacționează la ea, în timp ce AD reacționează puternic.'),
                 T('Report: two values of $D_6$, two decisions, one sentence.', 'Raportați: două valori $D_6$, două decizii, o frază.')),
           items(T('1. $u_{(i)}$: $@{a6.u0}$, $@{a6.u1}$, $@{a6.u2}$, $@{a6.u3}$, $@{a6.u4}$, $@{a6.u5}$; $D_6 = @{a6.d} < @{a6.crit}$: do not reject',
                   '1. $u_{(i)}$: $@{a6.u0}$; $@{a6.u1}$; $@{a6.u2}$; $@{a6.u3}$; $@{a6.u4}$; $@{a6.u5}$; $D_6 = @{a6.d} < @{a6.crit}$: nu respingem'),
                 T('2. $\\hat\\mu = @{a6.mu}$, $\\hat\\sigma = @{a6.sd}$; $D_6 = @{a6.dest} < @{a6.lil}$: do not reject', '2. $\\hat\\mu = @{a6.mu}$, $\\hat\\sigma = @{a6.sd}$; $D_6 = @{a6.dest} < @{a6.lil}$: nu respingem'),
                 T('3. At $x = -3$ the gap is at most $1/6 - 0.001$, like any other point; AD divides the squared gap by $F(1 - F) \\approx 0.001$, so the extreme day dominates $A^2$.',
                   '3. În $x = -3$, distanța este cel mult $1/6 - 0{,}001$, ca în orice alt punct; AD împarte pătratul distanței la $F(1 - F) \\approx 0{,}001$, deci ziua extremă domină $A^2$.')),
           size='scriptsize')

D.solved(T('A7: VaR 1\\% under Two Models', 'A7: VaR 1\\% pentru două modele'),
         items(T('Daily returns (\\%): Normal with $\\mu = 0.05$, $\\sigma = 1.2$; Student-t with $\\nu = 4$, location $0.06$, scale $0.8$; $t_{0.01}(4) = -@{a7.tq}$.',
                 'Randamente zilnice (\\%): Normală cu $\\mu = 0{,}05$, $\\sigma = 1{,}2$; Student-t cu $\\nu = 4$, locația $0{,}06$, scala $0{,}8$; $t_{0{,}01}(4) = -@{a7.tq}$.'),
               T('1. Compute VaR 1\\% under each model.', '1. Calculați VaR 1\\% pentru fiecare model.'),
               T('2. A bank uses the Normal VaR and observes 20 exceedances in 1000 days; compute the Kupiec statistic.', '2. O bancă folosește VaR Normal și observă 20 de depășiri în 1000 de zile; calculați statistica Kupiec.'),
               T('3. Decide at 5\\% and interpret.', '3. Decideți la 5\\% și interpretați.'),
               T('Report: two VaR values, one statistic, one decision.', 'Raportați: două valori VaR, o statistică, o decizie.')),
         items(T('1. Normal: $-(0.05 - @{a7.z} \\times 1.2) = @{a7.vn}\\%$; Student-t: $-(0.06 - @{a7.tq} \\times 0.8) = @{a7.vt}\\%$',
                 '1. Normală: $-(0{,}05 - @{a7.z} \\times 1{,}2) = @{a7.vn}\\%$; Student-t: $-(0{,}06 - @{a7.tq} \\times 0{,}8) = @{a7.vt}\\%$'),
               T('2. $\\ell_0 = 980\\ln 0.99 + 20\\ln 0.01 = @{a7.l0}$; $\\ell_1 = 980\\ln 0.98 + 20\\ln 0.02 = @{a7.l1}$; $LR_{uc} = -2(\\ell_0 - \\ell_1) = @{a7.lr}$',
                 '2. $\\ell_0 = 980\\ln 0{,}99 + 20\\ln 0{,}01 = @{a7.l0}$; $\\ell_1 = 980\\ln 0{,}98 + 20\\ln 0{,}02 = @{a7.l1}$; $LR_{uc} = -2(\\ell_0 - \\ell_1) = @{a7.lr}$'),
               T('3. $@{a7.lr} > @{crit95}$ ($p = @{a7.p}$): reject; twice the expected 10 exceedances: the Normal VaR is too low, as the thin tail predicts.',
                 '3. $@{a7.lr} > @{crit95}$ ($p = @{a7.p}$): respingem; numărul observat este dublul celor 10 depășiri așteptate: VaR Normal este prea mic, așa cum era de așteptat pentru o coadă subțire.')),
         size='scriptsize')

D.proposed(T('A8: Too Few Exceedances, and ES', 'A8: prea puține depășiri și ES'),
           items(T('Another bank observes 4 exceedances of its VaR 1\\% in 1000 days. Model: A7.', 'O altă bancă observă 4 depășiri ale VaR 1\\% în 1000 de zile. Model: A7.'),
                 T('1. Compute the Kupiec statistic and decide at 5\\%.', '1. Calculați statistica Kupiec și decideți la 5\\%.'),
                 T('2. Explain why too few exceedances is also a failure.', '2. Explicați de ce și prea puține depășiri înseamnă un eșec.'),
                 T('3. Compute ES 2.5\\% of the Normal model of A7 ($z_{0.025} = -1.960$, $\\varphi(1.960) = @{a8.phi}$).', '3. Calculați ES 2,5\\% al modelului Normal din A7 ($z_{0{,}025} = -1{,}960$, $\\varphi(1{,}960) = @{a8.phi}$).'),
                 T('Report: one statistic, one decision, one ES value and one sentence.', 'Raportați: o statistică, o decizie, o valoare ES și o frază.')),
           items(T('1. $\\ell_0 = 996\\ln 0.99 + 4\\ln 0.01 = @{a8.l0}$, $\\ell_1 = 996\\ln 0.996 + 4\\ln 0.004 = @{a8.l1}$; $LR_{uc} = @{a8.lr} > @{crit95}$ ($p = @{a8.p}$): reject',
                   '1. $\\ell_0 = 996\\ln 0{,}99 + 4\\ln 0{,}01 = @{a8.l0}$; $\\ell_1 = 996\\ln 0{,}996 + 4\\ln 0{,}004 = @{a8.l1}$; $LR_{uc} = @{a8.lr} > @{crit95}$ ($p = @{a8.p}$): respingem'),
                 T('2. The VaR is too conservative: capital is tied up for nothing; a model is wrong in either direction.', '2. VaR este prea prudent: capitalul este blocat inutil; un model poate greși în ambele direcții.'),
                 T('3. $\\mathrm{ES} = -0.05 + 1.2 \\times @{a8.phi}/0.025 = @{a8.es}\\%$', '3. $\\mathrm{ES} = -0{,}05 + 1{,}2 \\times @{a8.phi}/0{,}025 = @{a8.es}\\%$')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Precision and Interpretation', 'Partea B: date reale, precizie și interpretare')

D.task(T('B1: Which Distribution for the BET? [Solved]', 'B1: ce distribuție pentru BET? [Rezolvat]'),
       T('which of the six candidates fits the daily BET returns best, and is the skewness parameter needed?',
         'care dintre cei șase candidați se potrivește cel mai bine randamentelor zilnice BET și este necesar parametrul de asimetrie?'),
       T('BET closes, @{b1.y0}--@{a1.y1}, daily log returns in \\% ($n = @{b1.n}$)', 'închiderile BET, @{b1.y0}--@{a1.y1}, randamente logaritmice zilnice în \\% ($n = @{b1.n}$)'),
       [T('Fit the six candidates by maximum likelihood and report $\\ell(\\hat\\theta)$ of each.', 'Estimați cei șase candidați prin metoda verosimilității maxime și raportați $\\ell(\\hat\\theta)$ pentru fiecare.'),
        T('Compute AIC, BIC, $\\Delta$AIC and the Akaike weights.', 'Calculați AIC, BIC, $\\Delta$AIC și ponderile Akaike.'),
        T('Test the Student-t against the skewed-t with a likelihood-ratio test.', 'Testați Student-t față de skewed-t cu un test al raportului de verosimilitate.'),
        T('Interpretation: what do the winner and the LR test say about the tails and the skewness of the BET?', 'Interpretare: ce spun cîștigătorul și testul LR despre cozile și asimetria BET?')],
       T('a table with six rows, one LR test and two sentences', 'un tabel cu șase rînduri, un test LR și două fraze'), size='footnotesize', nb='B1')


def b1row(p, m):
    km = key(m)
    return f'{MNAME[m]} & $@{{{p}.{km}.ll}}$ & $@{{{p}.{km}.aic}}$ & $@{{{p}.{km}.bic}}$ & $@{{{p}.{km}.daic}}$ & $@{{{p}.{km}.dbic}}$ & $@{{{p}.{km}.w}}$'


D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch6_sem_b1', h='0.30') + table(
    'lrrrrrr', '& $\\ell(\\hat\\theta)$ & AIC & BIC & $\\Delta$AIC & $\\Delta$BIC & $w$', [b1row('b1', m) for m in SEM_MODELS], size='tiny') + items(
    T('Winner: @{b1.best} on both criteria; Student-t $\\hat\\nu = @{b1.nu}$; skewed-t $\\hat\\lambda = @{b1.lam}$; $LR = @{b1.lr}$, $p = @{b1.lrp}$: skewness not needed',
      'Cîștigător: @{b1.best} pe ambele criterii; Student-t $\\hat\\nu = @{b1.nu}$; skewed-t $\\hat\\lambda = @{b1.lam}$; $LR = @{b1.lr}$, $p = @{b1.lrp}$: asimetria nu este necesară'),
    T('Interpretation: very heavy tails ($\\nu$ close to 2) and no significant skewness; the Normal law is off by thousands of AIC points',
      'Interpretare: cozi foarte groase ($\\nu$ aproape de 2) și nicio asimetrie semnificativă; distribuția Normală este în urmă cu mii de puncte AIC')) + qlsem(), 'scriptsize')

D.task(T('B2: DAX and Bitcoin [Proposed]', 'B2: DAX și Bitcoin [Propus]'),
       T('do the DAX and Bitcoin choose the same distribution as the BET? Model: B1.', 'aleg DAX și Bitcoin aceeași distribuție ca BET? Model: B1.'),
       T('DAX since @{b2.dax.y0}, Bitcoin since @{b2.btc.y0} (7 days a week), daily log returns in \\%', 'DAX din @{b2.dax.y0}, Bitcoin din @{b2.btc.y0} (7 zile pe săptămînă), randamente logaritmice zilnice în \\%'),
       [T('Repeat steps 1--3 of B1 for each series.', 'Repetați pașii 1--3 din B1 pentru fiecare serie.'),
        T('Say whether AIC and BIC agree for each series.', 'Precizați dacă AIC și BIC aleg același model pentru fiecare serie.'),
        T('Interpretation: why might the best model differ between a stock index and a crypto asset?', 'Interpretare: de ce ar putea diferi cel mai bun model între un indice bursier și un activ cripto?')],
       T('two tables and three sentences', 'două tabele și trei fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrr', T('& DAX $\\Delta$AIC & DAX $\\Delta$BIC & DAX $w$ & Bitcoin $\\Delta$AIC & Bitcoin $\\Delta$BIC & Bitcoin $w$', '& DAX $\\Delta$AIC & DAX $\\Delta$BIC & DAX $w$ & Bitcoin $\\Delta$AIC & Bitcoin $\\Delta$BIC & Bitcoin $w$'),
    [f'{MNAME[m]} & $@{{b2.dax.{key(m)}.daic}}$ & $@{{b2.dax.{key(m)}.dbic}}$ & $@{{b2.dax.{key(m)}.w}}$ & $@{{b2.btc.{key(m)}.daic}}$ & $@{{b2.btc.{key(m)}.dbic}}$ & $@{{b2.btc.{key(m)}.w}}$' for m in SEM_MODELS],
    size='scriptsize') + items(
    T('DAX: @{b2.dax.best} (AIC and BIC); $LR$(Student-t vs skewed-t) $= @{b2.dax.lr}$, $p = @{b2.dax.lrp}$: skewness needed, $\\hat\\lambda = @{b2.dax.lam}$',
      'DAX: @{b2.dax.best} (AIC și BIC); $LR$(Student-t față de skewed-t) $= @{b2.dax.lr}$, $p = @{b2.dax.lrp}$: asimetria este necesară, $\\hat\\lambda = @{b2.dax.lam}$'),
    T('Bitcoin: @{b2.btc.best} (AIC and BIC), GED $\\hat\\beta = @{b2.btc.beta}$; $LR = @{b2.btc.lr}$, $p = @{b2.btc.lrp}$: no skewness',
      'Bitcoin: @{b2.btc.best} (AIC și BIC), GED $\\hat\\beta = @{b2.btc.beta}$; $LR = @{b2.btc.lr}$, $p = @{b2.btc.lrp}$: fără asimetrie'),
    T('Interpretation: Bitcoin has a very sharp peak (many quiet days) and a different tail shape; the DAX has crash asymmetry, the BET does not',
      'Interpretare: Bitcoin are un vîrf foarte ascuțit (multe zile liniștite) și o altă formă a cozii; DAX are asimetria specifică crahurilor, BET nu')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Do the Best Models Pass? [Solved]', 'B3: trec cele mai bune modele testele? [Rezolvat]'),
       T('do the Normal and Student-t fits of the S\\&P 500 pass the KS, Cramér--von Mises and Anderson--Darling tests?',
         'trec modelele Normal și Student-t estimate pe S\\&P 500 testele KS, Cramér--von Mises și Anderson--Darling?'),
       T('S\\&P 500 daily log returns since 2000 ($n = @{b3.n}$)', 'randamentele logaritmice zilnice S\\&P 500 din 2000 ($n = @{b3.n}$)'),
       [T('Fit both models and compute the three statistics.', 'Estimați ambele modele și calculați cele trei statistici.'),
        T('Compute the naive KS $p$-value, which treats the parameters as known.', 'Calculați valoarea $p$ KS naivă, care tratează parametrii ca fiind cunoscuți.'),
        T('Compute parametric bootstrap $p$-values with $B = 99$ (simulate, refit, recompute).', 'Calculați valori $p$ prin bootstrap parametric cu $B = 99$ (simulare, reestimare, recalculare).'),
        T('Draw the QQ plot of the data against both fits.', 'Desenați graficul QQ al datelor față de ambele modele estimate.'),
        T('Interpretation: if both models are rejected, what is the test still good for?', 'Interpretare: dacă ambele modele sînt respinse, la ce mai folosește testul?')],
       T('a table of six statistics with $p$-values, the QQ plot and two sentences', 'un tabel cu șase statistici și valorile $p$, graficul QQ și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch6_sem_b3', h='0.34') + table(
    'lrrrr', T('& KS & CvM & AD & KS naive $p$', '& KS & CvM & AD & $p$ KS naiv'),
    [T('Normal', 'Normală') + ' & $@{b3.Normal.ks}$ & $@{b3.Normal.cvm}$ & $@{b3.Normal.ad}$ & $@{b3.Normal.naive}$',
     T('bootstrap 95\\% critical value', 'valoare critică bootstrap 95\\%') + ' & $@{b3.Normal.ks.c}$ & $@{b3.Normal.cvm.c}$ & $@{b3.Normal.ad.c}$ & ',
     'Student-t & $@{b3.Studentt.ks}$ & $@{b3.Studentt.cvm}$ & $@{b3.Studentt.ad}$ & $@{b3.Studentt.naive}$',
     T('bootstrap 95\\% critical value', 'valoare critică bootstrap 95\\%') + ' & $@{b3.Studentt.ks.c}$ & $@{b3.Studentt.cvm.c}$ & $@{b3.Studentt.ad.c}$ & '],
    size='tiny') + items(
    T('Bootstrap $p = @{b3.Studentt.ad.p}$ for every statistic and both models (the smallest possible with $B = 99$): both rejected; the naive KS $p$ of the Student-t is far too large',
      '$p$ bootstrap $= @{b3.Studentt.ad.p}$ pentru fiecare statistică și ambele modele (cel mai mic posibil cu $B = 99$): ambele respinse; $p$ KS naiv al Student-t este mult prea mare'),
    T('Interpretation: with $n = @{b3.n}$ any i.i.d.\\ model is rejected; the statistics still rank the models (Student-t AD $@{b3.Studentt.ad}$ vs $@{b3.Normal.ad}$), and the QQ plot shows where each fails',
      'Interpretare: cu $n = @{b3.n}$, orice model i.i.d.\\ este respins; statisticile permit totuși ordonarea modelelor (AD Student-t $@{b3.Studentt.ad}$ față de $@{b3.Normal.ad}$), iar graficul QQ arată unde eșuează fiecare')) + qlsem(), 'scriptsize')

D.task(T('B4: Crash or Data Error? Banca Transilvania [Proposed]', 'B4: crah sau eroare de date? Banca Transilvania [Propus]'),
       T('which of the largest TLV returns are real, and does removing the wrong ones change the chosen model? Model: B1.',
         'care dintre cele mai mari randamente TLV sînt reale și schimbă eliminarea celor eronate modelul ales? Model: B1.'),
       T('TLV adjusted close since 2010, daily log returns in \\%; the close (not adjusted) for comparison; the BET on the same days',
         'prețul ajustat TLV din 2010, randamente logaritmice zilnice în \\%; prețul de închidere (neajustat) pentru comparație; BET în aceleași zile'),
       [T('List the three largest absolute returns with their dates.', 'Identificați cele mai mari trei randamente în valoare absolută și datele lor.'),
        T('For each, compare the close and the adjusted close around the date, and the BET return of the same day.', 'Pentru fiecare, comparați prețul de închidere și prețul ajustat în jurul datei, precum și randamentul BET din aceeași zi.'),
        T('Decide which days are data errors and remove only those.', 'Decideți care zile sînt erori de date și eliminați doar acele zile.'),
        T('Refit five candidates; compare the kurtosis, the Student-t $\\hat\\nu$, the AD statistic and the AIC winner before and after.',
          'Reestimați cinci candidați; comparați boltirea, $\\hat\\nu$ Student-t, statistica AD și cîștigătorul AIC înainte și după.'),
        T('Interpretation: what would have happened if you had removed all three days?', 'Interpretare: ce s-ar fi întîmplat dacă ați fi eliminat toate cele trei zile?')],
       T('a table of three days with your verdict, a before/after table and two sentences', 'un tabel cu trei zile și verdictul vostru, un tabel înainte/după și două fraze'), size='scriptsize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), items(
    (T('Largest absolute returns: @{b4.top0.d} ($@{b4.top0.r}\\%$), @{b4.top1.d} ($@{b4.top1.r}\\%$), @{b4.top2.d} ($@{b4.top2.r}\\%$)',
       'Cele mai mari randamente absolute: @{b4.top0.d} ($@{b4.top0.r}\\%$), @{b4.top1.d} ($@{b4.top1.r}\\%$), @{b4.top2.d} ($@{b4.top2.r}\\%$)'),
     [T('30 May 2016: the close falls from $@{b4.c.20160527}$ to $@{b4.c.20160530}$ (bonus shares), the adjusted close only from $@{b4.a.20160527}$ to $@{b4.a.20160530}$, then jumps to $@{b4.a.20160531}$: the adjustment is one day late, \\textbf{data error}',
        '30 mai 2016: prețul de închidere scade de la $@{b4.c.20160527}$ la $@{b4.c.20160530}$ (acțiuni gratuite), prețul ajustat doar de la $@{b4.a.20160527}$ la $@{b4.a.20160530}$, apoi sare la $@{b4.a.20160531}$: ajustarea are o zi întîrziere, \\textbf{eroare de date}'),
      T('19 December 2018: close and adjusted close both fall about $20\\%$, the BET falls $@{b4.bet}\\%$ the same day (the bank tax announced in December 2018): a \\textbf{real crash}, keep it',
        '19 decembrie 2018: prețul de închidere și cel ajustat scad amîndouă cu circa $20\\%$, iar BET scade cu $@{b4.bet}\\%$ în aceeași zi (taxa pe active bancare anunțată în decembrie 2018): un \\textbf{crah real}, îl păstrăm')]),
    (T('Before $\\to$ after removing the two May 2016 days', 'Înainte $\\to$ după eliminarea celor două zile din mai 2016'),
     [T('kurtosis $@{b4.raw.kurt} \\to @{b4.clean.kurt}$; standard deviation $@{b4.raw.sd} \\to @{b4.clean.sd}$; Student-t $\\hat\\nu$ $@{b4.raw.nu} \\to @{b4.clean.nu}$; AD $@{b4.raw.ad} \\to @{b4.clean.ad}$; AIC winner @{b4.raw.best} $\\to$ @{b4.clean.best}',
        'boltirea $@{b4.raw.kurt} \\to @{b4.clean.kurt}$; abaterea standard $@{b4.raw.sd} \\to @{b4.clean.sd}$; $\\hat\\nu$ Student-t $@{b4.raw.nu} \\to @{b4.clean.nu}$; AD $@{b4.raw.ad} \\to @{b4.clean.ad}$; cîștigătorul AIC @{b4.raw.best} $\\to$ @{b4.clean.best}')]),
    T('Interpretation: removing the real crash too would hide the risk the model is built for; clean only proven errors',
      'Interpretare: eliminarea și a crahului real ar ascunde exact riscul pentru care este construit modelul; curățați doar erorile dovedite')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B5: VaR 1\\% Out of Sample for the DAX [Solved]', 'B5: VaR 1\\% în afara eșantionului pentru DAX [Rezolvat]'),
       T('does the model chosen by AIC on 2000--2019 also give the best VaR 1\\% on 2020--@{a1.y1}?', 'dă modelul ales de AIC pe 2000--2019 și cel mai bun VaR 1\\% pe 2020--@{a1.y1}?'),
       T('DAX daily log returns: estimation @{b5.y0}--2019 ($n = @{b5.nest}$), test 2020--@{a1.y1} ($n = @{b5.ntest}$)', 'randamentele logaritmice zilnice DAX: estimare @{b5.y0}--2019 ($n = @{b5.nest}$), test 2020--@{a1.y1} ($n = @{b5.ntest}$)'),
       [T('Fit the six candidates on the estimation days and rank them by AIC.', 'Estimați cei șase candidați pe zilele de estimare și ordonați-i după AIC.'),
        T('Compute each model\'s mean log score on the test days.', 'Calculați scorul logaritmic mediu al fiecărui model pe zilele de test.'),
        T('Compute each model\'s VaR 1\\%, count its exceedances on the test days and run the Kupiec test.', 'Calculați VaR 1\\% al fiecărui model, numărați depășirile pe zilele de test și aplicați testul Kupiec.'),
        T('Add historical simulation (the 1\\% quantile of the estimation days) and the VaR averaged with the Akaike weights.', 'Adăugați simularea istorică (cuantila de 1\\% a zilelor de estimare) și VaR mediat cu ponderile Akaike.'),
        T('Interpretation: which model would you choose for VaR 1\\%, and why?', 'Interpretare: ce model ați alege pentru VaR 1\\% și de ce?')],
       T('one table, the chart and two sentences', 'un tabel, graficul și două fraze'), size='scriptsize', nb='B5')


def oosrow(p, m):
    km = key(m)
    ls = f'$@{{{p}.{km}.ls}}$' if m in SEM_MODELS else '--'
    return f'{MNAME[m]} & {ls} & $@{{{p}.{km}.var}}$ & @{{{p}.{km}.x}} & $@{{{p}.{km}.p}}$ & $@{{{p}.{km}.pin}}$'


OOSH = T('& log score & VaR 1\\% & exceed. & Kupiec $p$ & pinball $\\times 100$', '& scor log. & VaR 1\\% & depășiri & $p$ Kupiec & pinball $\\times 100$')
D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch6_sem_b5', h='0.28') + table(
    'lrrrrr', OOSH, [oosrow('b5', m) for m in SEM_MODELS + ['Historical', 'Averaged']], size='tiny') + items(
    T('AIC on 2000--2019: @{b5.baic}; best test log score: @{b5.bls}; expected exceedances @{b5.exp}; lowest pinball loss: @{b5.bpin}',
      'AIC pe 2000--2019: @{b5.baic}; cel mai bun scor logaritmic de test: @{b5.bls}; depășiri așteptate @{b5.exp}; cea mai mică pierdere pinball: @{b5.bpin}'),
    T('Interpretation: the AIC winner gives the best density but too few exceedances (too prudent); for VaR 1\\% alone the GED and the Student-t were closer to 1\\%',
      'Interpretare: cîștigătorul AIC dă cea mai bună densitate, dar prea puține depășiri (prea prudent); dacă scopul este doar VaR 1\\%, GED și Student-t au rate de depășire mai apropiate de 1\\%')) + qlsem(), 'scriptsize')

D.task(T('B6: VaR 1\\% Out of Sample for Bitcoin [Proposed]', 'B6: VaR 1\\% în afara eșantionului pentru Bitcoin [Propus]'),
       T('does the B5 conclusion hold for Bitcoin? Model: B5.', 'se menține concluzia din B5 pentru Bitcoin? Model: B5.'),
       T('Bitcoin: estimation @{b6.y0}--2019 ($n = @{b6.nest}$), test 2020--@{a1.y1} ($n = @{b6.ntest}$)', 'Bitcoin: estimare @{b6.y0}--2019 ($n = @{b6.nest}$), test 2020--@{a1.y1} ($n = @{b6.ntest}$)'),
       [T('Repeat steps 1--4 of B5.', 'Repetați pașii 1--4 din B5.'),
        T('Compare the empirical VaR 1\\% of the test days with that of the estimation days.', 'Comparați VaR 1\\% empiric al zilelor de test cu cel al zilelor de estimare.'),
        T('Interpretation: why do all heavy-tailed models fail the Kupiec test here, and in which direction?', 'Interpretare: de ce eșuează aici toate modelele cu cozi groase testul Kupiec și în ce direcție?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrr', OOSH, [oosrow('b6', m) for m in SEM_MODELS + ['Historical', 'Averaged']], size='scriptsize') + items(
    T('Expected exceedances @{b6.exp}; best test log score: @{b6.bls}; lowest pinball loss: @{b6.bpin}', 'Depășiri așteptate @{b6.exp}; cel mai bun scor logaritmic de test: @{b6.bls}; cea mai mică pierdere pinball: @{b6.bpin}'),
    T('Empirical VaR 1\\% of the test days: $@{b6.tv}\\%$, against $@{b6.Historical.var}\\%$ on the estimation days: Bitcoin became calmer',
      'VaR 1\\% empiric al zilelor de test: $@{b6.tv}\\%$, față de $@{b6.Historical.var}\\%$ pe zilele de estimare: Bitcoin a devenit mai calm'),
    T('Interpretation: all heavy-tailed models are too prudent (too few exceedances); the Normal model passes by accident: a static model fitted on a turbulent past cannot follow a calmer present',
      'Interpretare: toate modelele cu cozi groase sînt prea prudente (prea puține depășiri); modelul Normal trece din întîmplare: un model static estimat pe un trecut agitat nu se poate adapta unui prezent mai calm')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Does the Best Model Change over Time? [Proposed]', 'C1: se schimbă în timp cel mai bun model? [Propus]'),
       T('is the AIC winner of the S\\&P 500 and the BET the same in every two-year window?', 'este cîștigătorul AIC pentru S\\&P 500 și BET același în fiecare fereastră de doi ani?'),
       T('daily log returns since 2000, non-overlapping two-year windows (2000--2001, 2002--2003, ...); candidates: Normal, Student-t, skewed-t, GED, NIG; models: B1, A1',
         'randamente logaritmice zilnice din 2000, ferestre de doi ani care nu se suprapun (2000--2001, 2002--2003, ...); candidați: Normală, Student-t, skewed-t, GED, NIG; modele: B1, A1'),
       [T('In each window, find the AIC winner, its Akaike weight and the $\\Delta$AIC of the runner-up.', 'În fiecare fereastră, găsiți cîștigătorul AIC, ponderea lui Akaike și $\\Delta$AIC al celui de-al doilea.'),
        T('Count how often each model wins, for each index.', 'Numărați de cîte ori cîștigă fiecare model, pentru fiecare indice.'),
        T('Propose one explanation for the changes and one way to test it.', 'Propuneți o explicație a schimbărilor și o cale de a o testa.'),
        T('Interpretation: should a risk model be re-selected every two years?', 'Interpretare: ar trebui ca modelul de risc să fie ales din nou la fiecare doi ani?')],
       T('a table, a count of winners and a plan for a project', 'un tabel, numărul de ferestre cîștigate de fiecare model și un plan de proiect'), size='footnotesize', nb='C1')

C1H = T('window & winner & $w$ & runner-up & $\\Delta$', 'fereastra & cîștigător & $w$ & al doilea & $\\Delta$')
D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), cols(
    '\\centering S\\&P 500' + table('llrlr', C1H, c1rows['sp500'], size='tiny'),
    '\\centering BET' + table('llrlr', C1H, c1rows['bet'], size='tiny')) + items(
    T('S\\&P 500: GED wins @{c1.sp.ged} of @{c1.nper} windows; BET: Student-t wins @{c1.bet.t} of @{c1.nper}; the runner-up is often within $\\Delta < 2$',
      'S\\&P 500: GED cîștigă @{c1.sp.ged} din @{c1.nper} ferestre; BET: Student-t cîștigă @{c1.bet.t} din @{c1.nper}; modelul de pe locul al doilea are adesea $\\Delta < 2$'),
    T('The full-sample winner (NIG) rarely wins in short windows: with $n \\approx 500$ the candidates are hard to separate, and volatility regimes change the shape',
      'Cîștigătorul pe tot eșantionul (NIG) cîștigă rar în ferestre scurte: cu $n \\approx 500$, candidații se deosebesc greu, iar regimurile de volatilitate schimbă forma distribuției'),
    T('Project directions: standardise by a volatility model first; longer windows; more markets', 'Direcții de proiect: standardizați întîi cu un model de volatilitate; ferestre mai lungi; mai multe piețe')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to choose a distribution for the daily S\\&P 500 log returns (\\%) since 2000 ($n = @{c2.n}$). The answer:',
      'Un student a cerut unui asistent AI să aleagă o distribuție pentru randamentele logaritmice zilnice S\\&P 500 (\\%) din 2000 ($n = @{c2.n}$). Răspunsul:'),
    T('\\aiprompt{(a) The skewed-t has the lowest AIC, so it is significantly better than the Student-t (Delta AIC = @{c2.Studentt.daic}).}',
      '\\aiprompt{(a) Skewed-t are cel mai mic AIC, deci este semnificativ mai bun decît Student-t (Delta AIC = @{c2.Studentt.daic}).}'),
    T('\\aiprompt{(b) The KS test of the Student-t fit gives p = @{c2.pn} with scipy.stats.kstest, so the Student-t fits at the 1\\% level.}',
      '\\aiprompt{(b) Testul KS al ajustării Student-t dă p = @{c2.pn} cu scipy.stats.kstest, deci Student-t se potrivește la nivelul de 1\\%.}'),
    T('\\aiprompt{(c) The LR test of Normal vs Student-t uses the chi2(1) critical value 3.84.}', '\\aiprompt{(c) Testul LR al distribuției Normale față de Student-t folosește valoarea critică chi2(1) = 3,84.}'),
    T('\\aiprompt{(d) BIC penalises extra parameters less than AIC, because ln(n) is only @{c2.lnn}.}', '\\aiprompt{(d) BIC penalizează parametrii în plus mai puțin decît AIC, pentru că ln(n) este doar @{c2.lnn}.}'),
    T('\\aiprompt{(e) The Anderson-Darling test is more sensitive than KS to the fit in the tails.}', '\\aiprompt{(e) Testul Anderson-Darling este mai sensibil decît KS la calitatea ajustării în cozi.}'),
    T('\\aiprompt{(f) Since the skewed-t has the best fit, its VaR 1\\% is the most accurate one.}', '\\aiprompt{(f) Pentru că skewed-t are cea mai bună ajustare, VaR 1\\% calculat cu el este cel mai precis.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, unde se poate, cifra corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong wording: AIC is not a test and has no ``significance\'\'; a $\\Delta$AIC of $@{c2.Studentt.daic}$ is strong evidence by the rules of thumb (the LR test of $\\lambda = 0$ agrees)',
      '(a) Formulare greșită: AIC nu este un test și nu are „semnificație”; un $\\Delta$AIC de $@{c2.Studentt.daic}$ este o dovadă puternică după regulile practice (testul LR pentru $\\lambda = 0$ duce la aceeași concluzie)'),
    T('(b) Wrong: the parameters were estimated on the same data, so the naive KS $p$ is too large; a parametric bootstrap gives $p \\le 0.01$, and $p = @{c2.pn} < 0.05$ anyway',
      '(b) Greșit: parametrii au fost estimați pe aceleași date, deci $p$ KS naiv este prea mare; un bootstrap parametric dă $p \\le 0{,}01$; în plus, chiar și $p = @{c2.pn}$ este sub $0{,}05$'),
    T('(c) Wrong: $1/\\nu = 0$ lies on the boundary; the correct 5\\% critical value is $@{crit90}$ (the decision does not change: $LR = @{c2.lr}$)',
      '(c) Greșit: $1/\\nu = 0$ este la frontieră; valoarea critică corectă la 5\\% este $@{crit90}$ (decizia nu se schimbă: $LR = @{c2.lr}$)'),
    T('(d) Wrong: $\\ln n = @{c2.lnn} > 2$, so BIC penalises each parameter more than four times as much as AIC',
      '(d) Greșit: $\\ln n = @{c2.lnn} > 2$, deci BIC penalizează fiecare parametru de peste patru ori mai mult decît AIC'),
    T('(e) Correct: AD weights the squared gap by $1/[F(1 - F)]$, large in the tails', '(e) Corect: AD ponderează pătratul distanței cu $1/[F(1 - F)]$, mare în cozi'),
    T('(f) Wrong logic: the best overall fit does not imply the best quantile; in sample the skewed-t VaR 1\\% is $@{c2.vs}\\%$, the Student-t $@{c2.vt}\\%$, the data $@{c2.emp}\\%$; judge VaR by its exceedances out of sample',
      '(f) Raționament greșit: cea mai bună ajustare globală nu implică cea mai bună cuantilă; în eșantion, VaR 1\\% skewed-t este $@{c2.vs}\\%$, Student-t $@{c2.vt}\\%$, datele $@{c2.emp}\\%$; judecați VaR după depășirile din afara eșantionului')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('AIC and BIC rank models; $\\Delta \\le 2$ is a tie; Akaike weights turn $\\Delta$ into evidence', 'AIC și BIC ordonează modelele; $\\Delta \\le 2$ înseamnă egalitate; ponderile Akaike transformă $\\Delta$ în dovezi'),
    T('LR tests need nested models and a parameter inside the space; on the boundary use the 50:50 mixture', 'Testele LR cer modele imbricate și o valoare a parametrului din interiorul spațiului parametrilor; la frontieră folosiți amestecul 50:50'),
    T('KS with estimated parameters needs Lilliefors or a bootstrap; AD sees the tails, KS does not', 'KS cu parametri estimați cere tabelele Lilliefors sau bootstrap; AD pune accentul pe cozi, KS nu'),
    T('Check the largest returns before modelling: remove errors, keep crashes', 'Verificați cele mai mari randamente înainte de modelare: eliminați erorile, păstrați crahurile'),
    T('The best density is not always the best VaR: validate out of sample', 'Cea mai bună densitate nu dă întotdeauna cel mai bun VaR: validați în afara eșantionului')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 6 develops each topic of today: likelihood, LR and Vuong tests, AIC and BIC, EDF tests, QQ plots, out-of-sample VaR, model uncertainty',
      'Cursul 6 dezvoltă fiecare temă de azi: verosimilitatea, testele LR și Vuong, AIC și BIC, testele EDF, graficele QQ, VaR în afara eșantionului, incertitudinea de model'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: standardised returns, more markets, rolling windows', 'C1 poate deveni un proiect de echipă: randamente standardizate, mai multe piețe, ferestre mobile'),
    T('Reading: \\refBA; \\refStephens; \\refKupiec', 'Lectură: \\refBA; \\refStephens; \\refKupiec')))

D.references(bib(['Akaike', 'BA', 'FHH', 'Hansen', 'Kupiec', 'Lilliefors', 'Schwarz', 'SL', 'Stephens', 'Wilks']))

if __name__ == '__main__':
    for path in D.write(V):
        for p in (path,):
            tex = open(p, encoding='utf-8').read()
            tex = re.sub(r'=\s*<', '<', tex)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(tex)
