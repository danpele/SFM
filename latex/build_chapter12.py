r"""
build_chapter12.py -- Capitolul 12 (Modele de scoring), EN + RO dintr-o singură sursă
====================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_12/ch12_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Date: South German Credit (Grömping, 2019; UCI) și Default of Credit Card Clients (Yeh și Lien, 2009; UCI),
salvate în data/credit. Convenția: default = 1 (credit neperformant, rău-platnic), 0 = bun-platnic.
Ieșire:
  EN/Courses/chapter12_scoring_models.tex
  RO/Cursuri/capitol12_modele_scoring.tex
Rulare:
  python3 Quantlets/Ch_12/generate_all_charts.py
  python3 latex/build_chapter12.py && python3 latex/sfm_build.py compile 12
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch12_common import BIB, QLURL, REFS, T, load, values, vname   # noqa: E402

N = load()
V = values(N)
D = Deck(12, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_12}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.58\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def side_chart(title, fig, folder, bullets, wl='0.50', wr='0.48', h='0.62\\textheight', size='footnotesize'):
    left = f'\\includegraphics[width=\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}'
    D.frame(title, cols(left, items(*bullets), wl=wl, wr=wr) + '\n' + ql(folder), size)


PH = {
    'fico': ('ch12_fico_hq_2020.jpg', C + 'Ficoheadquarters.jpg', '⟦Photo||Foto⟧: Coolcaesar (2020); CC BY-SA 4.0; Wikimedia Commons'),
    'fisher': ('ch5_fisher_1913.jpg', C + 'Youngronaldfisher2.JPG', '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧ (1913); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'iris': ('ch12_iris_versicolor.jpg', C + 'Iris_versicolor_-20200620-RM-100933.jpg', '⟦Photo||Foto⟧: Ermell (2020); CC BY-SA 4.0; Wikimedia Commons'),
    'stern': ('ch12_nyu_stern_2019.jpg', C + 'NYU_Stern_School_of_Business_-_Henry_Kaufman_Management_Center_(48072761732).jpg',
              '⟦Photo||Foto⟧: Ajay Suresh (2019); CC BY 2.0; Wikimedia Commons'),
    'merton': ('ch12_merton_2010.jpg', C + 'Robert_Merton_November_2010_(2x3_cropped).jpg',
               '⟦Photo||Foto⟧: Massachusetts Institute of Technology (2010); CC BY-SA 4.0; Wikimedia Commons'),
    'ecoa': ('ch12_heckler_bill_1973.jpg', C + 'A_Bill_to_amend_the_Consumer_Credit_Protection_Act.jpg',
             '⟦Photo||Foto⟧: H.R. 10162 (1973); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'bis': ('ch10_bis_tower_2014.jpg', C + 'Basel_Committee_on_Banking_Supervision_-_BCBS.jpg',
            '⟦Photo||Foto⟧: Taxiarchos228 (2014); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.48\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
# odds and log-odds of a PD of 20%
V.put('od.p', 0.20, 2)
V.put('od.o', 0.2 / 0.8, 2)
V.put('od.lo', math.log(0.25), 3)
# small logit: an applicant (36 months, 30 years, 4000 DM, no checking account), rounded coefficients
sm = {k: round(d['b'], 3) for k, d in N['small']['table'].items()}
x = {'duration_years': 3, 'age_decades': 3, 'amount_1000': 4, 'no_account': 1, 'negative_balance': 0}
eta = sm['const'] + sum(sm[k] * v for k, v in x.items())
pex = 1 / (1 + math.exp(-eta))
V.put('ex.eta', eta, 3)
V.put('ex.pd', 100 * pex, 1)
V.put('ex.odds', math.exp(eta), 2)
for k in x:
    V.put(f'ex.t.{k}', sm[k] * x[k], 3)
eta0 = eta - sm['no_account']
V.put('ex.eta0', eta0, 3)
V.put('ex.pd0', 100 / (1 + math.exp(-eta0)), 1)
V.put('ex.me', 100 * pex * (1 - pex) * sm['duration_years'], 1)
# WoE by hand: status 4 and status 1 (full sample)
sw = N['status_woe']
# expected loss and IRB capital: 100 000 RON, PD 2%, LGD 45%
K2 = N['irb']['0.02']['other retail']
V.raw('el.el', f'{100000 * 0.02 * 0.45:,.0f}'.replace(',', '\\,'))
V.raw('el.k', f'{100000 * K2:,.0f}'.replace(',', '\\,'))
V.raw('el.rwa', f'{12.5 * 100000 * K2:,.0f}'.replace(',', '\\,'))
V.put('el.rw', 1250 * K2, 0)
V.put('el.z999', stats.norm.ppf(0.999), 3)
V.put('el.zpd', stats.norm.ppf(0.02), 3)
R2 = N['irb_R']['0.02']
V.put('el.R', R2, 3)
V.put('el.arg', (stats.norm.ppf(0.02) + math.sqrt(R2) * stats.norm.ppf(0.999)) / math.sqrt(1 - R2), 3)
V.put('el.cpd', 100 * stats.norm.cdf((stats.norm.ppf(0.02) + math.sqrt(R2) * stats.norm.ppf(0.999)) / math.sqrt(1 - R2)), 2)
# scorecard scaling
f_, o_ = N['scaling']['factor'], N['scaling']['offset']
V.put('sc.ln50', math.log(50), 3)
V.put('sc.s5', o_ + f_ * math.log(19), 1)
V.put('sc.ln19', math.log(19), 3)
V.put('sc.s30', o_ + f_ * math.log(7 / 3), 1)
V.put('sc.ln73', math.log(7 / 3), 3)
lo560 = (560 - o_) / f_
V.put('sc.lo560', lo560, 3)
V.put('sc.od560', math.exp(lo560), 1)
V.put('sc.pd560', 100 / (1 + math.exp(lo560)), 1)
pts = [p for p in N['points'] if p['variable'] == 'status']
for p in pts:
    V.put(f'pt.{p["bin"]}', p['points'], 0)
    V.put(f'pt.{p["bin"]}.w', p['woe'], 3)
V.put('pt.k', len(N['sel']), 0)
V.put('pt.o6', o_ / len(N['sel']), 2)
V.put('pt.b0k', N['woe_logit']['b'][0] / len(N['sel']), 3)
# prior correction: a sample PD of 30% and 50%
sh = N['prior_shift']
V.put('pr.p50', 100 / (1 + math.exp(-(0 + sh))), 1)
V.put('prior.abs', -sh, 3)
V.put('ex.etapop', eta + sh, 3)
V.put('ex.pdpop', 100 / (1 + math.exp(-(eta + sh))), 1)
V.put('pr.ln', math.log(0.05 / 0.95), 3)
V.put('pr.ls', math.log(0.3 / 0.7), 3)
# Bayes cut-off
V.put('bayes', 1 / 6, 3)
V.put('cpd.b', 100 * N['cond_pd']['0.02']['-3'], 1)
V.put('cpd.t', 100 * N['cond_pd']['0.02']['0'], 1)
# Altman worked example
V.put('alt.t1', 1.2 * 0.15, 3)
V.put('alt.t2', 1.4 * 0.20, 3)
V.put('alt.t3', 3.3 * 0.08, 3)
V.put('alt.t4', 0.6 * 0.90, 3)
V.put('alt.t5', 1.0 * 1.10, 3)
# Merton worked example: V = 100, D = 70, sigma = 25%, mu = 5%, T = 1
V.put('mt.ln', math.log(100 / 70), 4)
V.put('mt.drift', 0.05 - 0.5 * 0.25 ** 2, 5)
V.put('mt.num', math.log(100 / 70) + 0.05 - 0.5 * 0.25 ** 2, 4)
# LDA 2x2 worked example
L = N['lda2']
S = L['S']
det = S[0][0] * S[1][1] - S[0][1] ** 2
V.put('l2.det', det, 0)
# ROC by pairs (toy): 3 bads, 7 goods -> see the seminar; here the test sample
V.raw('pairs', f"{N['val']['bad_test'] * (N['val']['n_test'] - N['val']['bad_test']):,}".replace(',', '\\,'))
# fairness ranges
V.put('ri.under', 100 * (1 - N['reject']['mean_pd_rej_acc_model'] / N['reject']['rate_rejected']), 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how does a bank turn the data of an applicant into a probability of default, and how do we know that the number is right?',
       '\\textbf{Întrebarea}: cum transformă o bancă datele unui solicitant într-o probabilitate de nerambursare și de unde știm că cifra este corectă?'),
     [T('Chapters 2--10 measured the risk of prices; this chapter measures the risk that a borrower does not pay back',
        'Capitolele 2--10 au măsurat riscul prețurilor; acest capitol măsoară riscul ca un debitor să nu-și ramburseze creditul')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('credit scores, PD, LGD, EAD and the Basel capital; default data and class imbalance', 'scorurile de credit, PD, LGD, EAD și capitalul Basel; datele de nerambursare și dezechilibrul claselor'),
      T('Weight of Evidence and Information Value; logistic regression and Fisher\'s discriminant analysis', 'Weight of Evidence și Information Value; regresia logistică și analiza discriminantă a lui Fisher'),
      T('the scorecard and its validation: confusion matrix, ROC, AUC, Gini, CAP, KS, Brier score, calibration', 'scorecard-ul și validarea lui: matricea de confuzie, ROC, AUC, Gini, CAP, KS, scorul Brier, calibrarea'),
      T('reject inference and fairness; the Altman Z-score and the Merton distance to default', 'reject inference și echitatea (fairness); scorul Z al lui Altman și distanța pînă la nerambursare a lui Merton')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Define PD, LGD, EAD and the expected loss, and explain what the Basel IRB capital adds to the expected loss',
      'Definiți PD, LGD, EAD și pierderea așteptată și explicați ce adaugă capitalul IRB din Basel peste pierderea așteptată'),
    T('Compute the Weight of Evidence and the Information Value of a variable from a table of counts', 'Calculați Weight of Evidence și Information Value ale unei variabile dintr-un tabel de frecvențe'),
    T('Estimate a logistic regression and interpret its coefficients as changes in log-odds and as odds ratios',
      'Estimați o regresie logistică și interpretați coeficienții ca modificări ale logaritmului șansei și ca rapoarte ale șanselor'),
    T('Derive Fisher\'s linear discriminant and compare it with the logit', 'Derivați discriminantul liniar al lui Fisher și comparați-l cu modelul logit'),
    T('Scale a scorecard and validate it out of sample with the confusion matrix, AUC, Gini, KS, the Brier score and a calibration plot',
      'Scalați un scorecard și validați-l în afara eșantionului cu matricea de confuzie, AUC, Gini, KS, scorul Brier și graficul de calibrare')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~21 (probability of default) and Ch.~22 (credit risk management)',
       'Manual: \\refFHH, cap.~21 (probabilitatea de nerambursare) și cap.~22 (managementul riscului de credit)'),
     [T('discriminant analysis: \\refMVA, Ch.~14', 'analiza discriminantă: \\refMVA, cap.~14'),
      T('credit scoring: \\refTCE; scorecards: \\refSiddiqi', 'scoring de credit: \\refTCE; scorecard-uri: \\refSiddiqi')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_12}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_12}'),
     [T('the conditional PD of the one-factor model is ported from the SFE Quantlet SFEdefaproba', 'PD condiționată din modelul cu un factor este portată din Quantlet-ul SFE SFEdefaproba')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter12_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter12_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{MVA Multivariate Statistical Analysis}{https://quantinar.com/course/540/multivariate-statistical-analysis}',
      'Curs video: \\quantinar{MVA Multivariate Statistical Analysis}{https://quantinar.com/course/540/multivariate-statistical-analysis}')))

# =============================================================================
# 1. CE ESTE UN SCOR DE CREDIT
# =============================================================================
D.section('What a Credit Score Is', 'Scorul de credit: definiție')

D.frame(T('The Lending Decision', 'Decizia de creditare'), items(
    (T('An applicant asks for a loan; the bank must decide \\textbf{accept} or \\textbf{reject}, and at which interest rate',
       'Un solicitant cere un credit; banca trebuie să decidă \\textbf{acceptarea} sau \\textbf{respingerea} și dobînda'),
     [T('\\textbf{default}: the borrower does not pay as agreed; Basel: the bank judges the borrower unlikely to pay, or the borrower is more than 90 days past due on a material obligation \\refBCBS, ¶452',
        '\\textbf{nerambursarea} (default): debitorul nu plătește conform contractului; Basel: banca apreciază că debitorul nu va plăti sau debitorul întîrzie peste 90 de zile o obligație semnificativă \\refBCBS, ¶452')]),
    (T('\\textbf{Credit score}: a number computed from the data of an applicant that orders applicants by their risk of default',
       '\\textbf{Scorul de credit}: un număr calculat din datele solicitantului, care ordonează solicitanții după riscul lor de nerambursare'),
     [T('\\textbf{PD} (probability of default): the probability of default within a horizon, usually one year', '\\textbf{PD} (probability of default): probabilitatea de nerambursare pe un orizont, de obicei un an'),
      T('a good score both \\textbf{ranks} (discrimination) and gives the right \\textbf{level} of PD (calibration)', 'un scor bun \\textbf{ordonează} corect (discriminare) și dă \\textbf{nivelul} corect al PD (calibrare)')]),
    (T('Kinds of scores', 'Tipuri de scoruri'),
     [T('application scores (new clients) and behavioural scores (existing clients, from their payment history)', 'scoruri de aplicare (clienți noi) și scoruri comportamentale (clienți existenți, din istoricul plăților)'),
      T('consumer scores (statistical models on many small loans) and corporate scores (accounting ratios, market prices)', 'scoruri pentru persoane fizice (modele statistice pe multe credite mici) și pentru companii (indicatori contabili, prețuri de piață)')])))

D.frame(T('From Durand to FICO', 'De la Durand la FICO'), cols(items(
    (T('1941: \\refDurand separates good and bad instalment loans of US lenders with Fisher\'s discriminant analysis',
       '1941: \\refDurand separă creditele bune de cele rele ale creditorilor americani cu analiza discriminantă a lui Fisher'),
     [T('the first statistical credit-scoring study', 'primul studiu statistic de scoring de credit')]),
    (T('1956: Bill Fair and Earl Isaac found Fair, Isaac and Company (today FICO) \\refFICO', '1956: Bill Fair și Earl Isaac înființează Fair, Isaac and Company (azi FICO) \\refFICO'),
     [T('1989: the FICO score, computed from credit-bureau reports, between 300 and 850 points', '1989: scorul FICO, calculat din rapoartele birourilor de credit, între 300 și 850 de puncte')]),
    (T('Statistical scoring replaced the judgement of a loan officer with a rule that is the same for everybody \\refHH',
       'Scoringul statistic a înlocuit judecata ofițerului de credit cu o regulă identică pentru toți \\refHH'),
     [T('cheaper, faster and testable; the question whether the rule is fair remains', 'mai ieftin, mai rapid și verificabil; rămîne însă întrebarea dacă regula este echitabilă')])),
    ph('fico', T('FICO headquarters, San Jose, California', 'Sediul FICO, San Jose, California'), h='0.40\\textheight'),
    wl='0.56', wr='0.40'), 'footnotesize')

D.frame(T('PD, LGD, EAD and the Expected Loss', 'PD, LGD, EAD și pierderea așteptată'), items(
    (T('Three risk parameters of a loan (Basel II, \\refBCBS)', 'Trei parametri de risc ai unui credit (Basel II, \\refBCBS)'),
     [T('\\textbf{PD}: probability of default within one year', '\\textbf{PD}: probabilitatea de nerambursare într-un an'),
      T('\\textbf{LGD} (loss given default): the share of the exposure lost if default occurs, after recoveries', '\\textbf{LGD} (loss given default): partea din expunere pierdută în caz de nerambursare, după recuperări'),
      T('\\textbf{EAD} (exposure at default): the amount owed at the moment of default', '\\textbf{EAD} (exposure at default): suma datorată în momentul nerambursării')]),
    (T('\\textbf{Expected loss}: $\\mathrm{EL} = \\mathrm{PD} \\times \\mathrm{LGD} \\times \\mathrm{EAD}$ (with independent PD and LGD)', '\\textbf{Pierderea așteptată}: $\\mathrm{EL} = \\mathrm{PD} \\times \\mathrm{LGD} \\times \\mathrm{EAD}$ (cu PD și LGD independente)'),
     [T('\\textbf{Example}: a loan of 100\\,000 RON with PD $= 2\\%$, LGD $= 45\\%$: EL $= 0.02 \\times 0.45 \\times 100\\,000 = @{el.el}$ RON',
        '\\textbf{Exemplu}: un credit de 100\\,000 de lei cu PD $= 2\\%$, LGD $= 45\\%$: EL $= 0.02 \\times 0.45 \\times 100\\,000 = @{el.el}$ lei'),
      T('EL is a cost of doing business: it is priced into the interest rate and covered by provisions', 'EL este un cost al activității: intră în dobîndă și este acoperită prin provizioane')]),
    T('Capital covers the \\textbf{unexpected loss}: losses above EL in a bad year', 'Capitalul acoperă \\textbf{pierderea neașteptată}: pierderile de peste EL într-un an prost')))

D.frame(T('The Basel IRB Capital Formula', 'Formula de capital IRB din Basel'), cols(items(
    (T('\\textbf{IRB} (internal ratings-based) approach: the bank estimates PD (and possibly LGD, EAD); the regulator gives the formula \\refBCBS, ¶328--330',
       'Abordarea \\textbf{IRB} (internal ratings-based): banca estimează PD (și eventual LGD, EAD); autoritatea de reglementare dă formula \\refBCBS, ¶328--330'),
     [T('$K = \\mathrm{LGD}\\left[\\Phi\\!\\left(\\dfrac{\\Phi^{-1}(\\mathrm{PD}) + \\sqrt{R}\\,\\Phi^{-1}(0.999)}{\\sqrt{1 - R}}\\right) - \\mathrm{PD}\\right]$; RWA $= 12.5\\,K\\,$EAD',
        '$K = \\mathrm{LGD}\\left[\\Phi\\!\\left(\\dfrac{\\Phi^{-1}(\\mathrm{PD}) + \\sqrt{R}\\,\\Phi^{-1}(0.999)}{\\sqrt{1 - R}}\\right) - \\mathrm{PD}\\right]$; RWA $= 12.5\\,K\\,$EAD'),
      T('$K$: the capital per unit of EAD; $\\Phi$, $\\Phi^{-1}$: the $N(0,1)$ distribution function and its inverse; 0.999: a bad year occurs once in 1000 years', '$K$: capitalul pe unitatea de EAD; $\\Phi$, $\\Phi^{-1}$: funcția de repartiție $N(0,1)$ și inversa ei; 0,999: un an prost apare o dată la 1000 de ani'),
      T('$R$: the correlation of the borrowers with one common factor; RWA: risk-weighted assets, of which the bank holds at least 8\\%', '$R$: corelația debitorilor cu un factor comun; RWA (risk-weighted assets): activele ponderate la risc, din care banca deține cel puțin 8\\%')]),
    (T('Logic \\refGordy: the term in $\\Phi(\\cdot)$ is the PD in a year whose common factor is exceeded only with probability 0.1\\%', 'Logica \\refGordy: termenul din $\\Phi(\\cdot)$ este PD într-un an al cărui factor comun este depășit doar cu probabilitatea 0,1\\%'),
     [T('so $K$ is the credit VaR 0.1\\% of the loss minus EL: the unexpected loss (Chapter 10)', 'deci $K$ este VaR 0,1\\% al pierderii din credit minus EL: pierderea neașteptată (Capitolul 10)')])),
    ph('bis', T('The BIS tower in Basel', 'Turnul BIS din Basel'), h='0.40\\textheight'), wl='0.62', wr='0.34'), 'footnotesize')

chart(T('PD in Good and Bad Years: the One-Factor Model', 'PD în ani buni și în ani proști: modelul cu un factor'), 'sfm_ch12_conditional_pd', 'SFM_ch12_validation_irb', [
    T('Borrower $i$ defaults if $\\sqrt{\\rho}\\,Y + \\sqrt{1 - \\rho}\\,\\varepsilon_i < \\Phi^{-1}(\\mathrm{PD})$; $Y$: the state of the economy; given $Y = y$: PD$(y) = \\Phi\\big((\\Phi^{-1}(\\mathrm{PD}) - \\sqrt{\\rho}\\,y)/\\sqrt{1 - \\rho}\\big)$, $\\rho = 0.2$',
      'Debitorul $i$ intră în nerambursare dacă $\\sqrt{\\rho}\\,Y + \\sqrt{1 - \\rho}\\,\\varepsilon_i < \\Phi^{-1}(\\mathrm{PD})$; $Y$: starea economiei; dat fiind $Y = y$: PD$(y) = \\Phi\\big((\\Phi^{-1}(\\mathrm{PD}) - \\sqrt{\\rho}\\,y)/\\sqrt{1 - \\rho}\\big)$, $\\rho = 0{,}2$'),
    T('$\\varepsilon_i$: the borrower\'s own shock; $Y$ and $\\varepsilon_i$ independent $N(0,1)$; $\\rho$: the weight of the common factor', '$\\varepsilon_i$: șocul propriu al debitorului; $Y$ și $\\varepsilon_i$ independente, $N(0,1)$; $\\rho$: ponderea factorului comun'),
    T('Interpretation: a loan with PD $= 2\\%$ defaults with probability @{cpd.b}\\% in a bad year ($y = -3$), @{cpd.t}\\% in a typical year and almost never in a good year; IRB uses $y = \\Phi^{-1}(0.001)$',
      'Interpretare: un credit cu PD $= 2\\%$ intră în nerambursare cu probabilitatea @{cpd.b}\\% într-un an prost ($y = -3$), @{cpd.t}\\% într-un an obișnuit și aproape niciodată într-un an bun; IRB folosește $y = \\Phi^{-1}(0{,}001)$')],
    h='0.46\\textheight')

D.frame(T('Worked Example: IRB Capital of a Retail Loan', 'Exemplu rezolvat: capitalul IRB al unui credit de retail'), items(
    (T('Other retail exposures: $R = 0.03\\,w + 0.16\\,(1 - w)$, $w = (1 - e^{-35\\,\\mathrm{PD}})/(1 - e^{-35})$ \\refBCBS, ¶330',
       'Alte expuneri de retail: $R = 0.03\\,w + 0.16\\,(1 - w)$, $w = (1 - e^{-35\\,\\mathrm{PD}})/(1 - e^{-35})$ \\refBCBS, ¶330'),
     [T('$w \\in [0, 1]$ grows with PD, so $R$ falls from 0.16 (very safe loans) to 0.03 (very risky loans)', '$w \\in [0, 1]$ crește cu PD, deci $R$ scade de la 0,16 (credite foarte sigure) la 0,03 (credite foarte riscante)'),
      T('PD $= 2\\%$: $R = @{el.R}$; $\\Phi^{-1}(0.02) = @{el.zpd}$, $\\Phi^{-1}(0.999) = @{el.z999}$', 'PD $= 2\\%$: $R = @{el.R}$; $\\Phi^{-1}(0.02) = @{el.zpd}$, $\\Phi^{-1}(0.999) = @{el.z999}$')]),
    (T('\\textbf{Step by step}', '\\textbf{Pas cu pas}'),
     [T('argument: $(@{el.zpd} + \\sqrt{@{el.R}} \\times @{el.z999})/\\sqrt{1 - @{el.R}} = @{el.arg}$; $\\Phi(@{el.arg}) = @{el.cpd}\\%$: the PD in a bad year',
        'argumentul: $(@{el.zpd} + \\sqrt{@{el.R}} \\times @{el.z999})/\\sqrt{1 - @{el.R}} = @{el.arg}$; $\\Phi(@{el.arg}) = @{el.cpd}\\%$: PD într-un an prost'),
      T('$K = 0.45 \\times (@{el.cpd}\\% - 2\\%) = @{irb.20.or}\\%$ of EAD: @{el.k} RON for a loan of 100\\,000 RON', '$K = 0.45 \\times (@{el.cpd}\\% - 2\\%) = @{irb.20.or}\\%$ din EAD: @{el.k} lei pentru un credit de 100\\,000 de lei'),
      T('RWA $= 12.5 \\times @{el.k} = @{el.rwa}$ RON: a risk weight of @{el.rw}\\%', 'RWA $= 12.5 \\times @{el.k} = @{el.rwa}$ lei: o pondere de risc de @{el.rw}\\%')]),
    T('Interpretation: EL ($@{el.el}$ RON) is priced in; the capital ($@{el.k}$ RON) absorbs a year in which defaults reach @{el.cpd}\\% instead of 2\\%',
      'Interpretare: EL ($@{el.el}$ lei) intră în preț; capitalul ($@{el.k}$ lei) absoarbe un an în care nerambursările ajung la @{el.cpd}\\% în loc de 2\\%')), 'footnotesize')

chart(T('Expected Loss and IRB Capital as Functions of PD', 'Pierderea așteptată și capitalul IRB în funcție de PD'), 'sfm_ch12_irb', 'SFM_ch12_validation_irb', [
    T('LGD $= 45\\%$; capital $K$ for other retail ($R$ from 0.16 to 0.03), residential mortgages ($R = 0.15$) and revolving retail exposures ($R = 0.04$)',
      'LGD $= 45\\%$; capitalul $K$ pentru alte expuneri de retail ($R$ de la 0,16 la 0,03), credite ipotecare ($R = 0{,}15$) și expuneri revolving ($R = 0{,}04$)'),
    T('Interpretation: EL grows linearly with PD; $K$ grows fast at low PD and then flattens, because a very risky loan is mostly an expected (priced) loss',
      'Interpretare: EL crește liniar cu PD; $K$ crește repede la PD mici și apoi se aplatizează, deoarece un credit foarte riscant este în mare parte o pierdere așteptată (inclusă în preț)')],
    h='0.50\\textheight')

D.recap(('What a Credit Score Is', 'scorul de credit: definiție'), [
    T('A score orders applicants by risk; a PD also gives the level of risk', 'Un scor ordonează solicitanții după risc; o PD dă și nivelul riscului'),
    T('EL $=$ PD $\\times$ LGD $\\times$ EAD is priced in; Basel capital covers the unexpected loss (a credit VaR 0.1\\% minus EL)', 'EL $=$ PD $\\times$ LGD $\\times$ EAD intră în preț; capitalul Basel acoperă pierderea neașteptată (VaR 0,1\\% al creditului minus EL)'),
    T('Statistical scoring started with Fisher\'s discriminant analysis (Durand, 1941)', 'Scoringul statistic a început cu analiza discriminantă a lui Fisher (Durand, 1941)')])

# =============================================================================
# 2. DATE
# =============================================================================
D.section('Default Data', 'Datele de nerambursare')

D.frame(T('The South German Credit Data', 'Datele South German Credit'), items(
    (T('@{n} consumer credits of a large regional bank in southern Germany, 1973--1975 \\refSGC; \\refGroemping',
       '@{n} de credite de consum ale unei mari bănci regionale din sudul Germaniei, 1973--1975 \\refSGC; \\refGroemping'),
     [T('outcome: the contract was complied with (good) or not (bad, \\textbf{default} $= 1$): @{bad} bad and 700 good credits',
        'rezultatul: contractul a fost respectat (bun-platnic) sau nu (rău-platnic, \\textbf{nerambursare} $= 1$): @{bad} de credite neperformante și 700 performante'),
      T('20 variables: checking account, duration, credit history, purpose, amount (DM, Deutsche Mark), savings, employment, age and others',
        '20 de variabile: contul curent, durata, istoricul de credit, destinația, suma (DM, mărci germane), economiile, vechimea în muncă, vîrsta și altele')]),
    (T('A \\textbf{stratified sample}: bad credits are heavily oversampled; the bad rate of the bank was about 5\\% \\refGroemping',
       'Un \\textbf{eșantion stratificat}: creditele neperformante sînt puternic suprareprezentate; rata lor în bancă era de aproximativ 5\\% \\refGroemping'),
     [T('all borrowers had passed the bank\'s checks: rejected applicants are missing (reject inference, Section 8)', 'toți debitorii trecuseră de verificările băncii: solicitanții respinși lipsesc (reject inference, secțiunea 8)'),
      T('the cost of accepting a bad credit is set to five times the cost of rejecting a good one', 'costul acceptării unui credit neperformant este stabilit la de cinci ori costul respingerii unuia performant')]),
    T('Licence CC BY 4.0; saved once in the course repository (data/credit)', 'Licența CC BY 4.0; salvate o singură dată în repository-ul cursului (data/credit)')))

D.frame(T('A Lesson in Data Checking', 'O lecție despre verificarea datelor'), items(
    (T('The data were used for decades in the UCI version ``Statlog (German Credit)\'\', with code labels that were wrong \\refGroemping',
       'Datele au fost folosite zeci de ani în versiunea UCI „Statlog (German Credit)”, cu etichete greșite ale codurilor \\refGroemping'),
     [T('in that version, ``no checking account\'\' was the safest group: @{st4.n} credits, @{st4.r}\\% bad',
        'în acea versiune, „fără cont curent” era grupul cel mai sigur: @{st4.n} de credite, @{st4.r}\\% neperformante'),
      T('in the corrected data, those @{st4.n} credits have a balance of at least 200 DM or a salary account; ``no checking account\'\' is the riskiest group (@{st1.r}\\% bad)',
        'în datele corectate, acele @{st4.n} de credite au un sold de cel puțin 200 DM sau un cont de salariu; „fără cont curent” este grupul cel mai riscant (@{st1.r}\\% neperformante)')]),
    (T('The numbers were right, the labels were wrong: every model was fine, every interpretation of this variable was reversed',
       'Cifrele erau corecte, etichetele greșite: fiecare model funcționa, fiecare interpretare a acestei variabile era inversată'),
     [T('checks: compare the frequency tables with the original source (here the book of Fahrmeir and Hamerle), ask whether the signs make economic sense',
        'verificări: comparăm tabelele de frecvențe cu sursa originală (aici cartea lui Fahrmeir și Hamerle) și ne întrebăm dacă semnele au sens economic')]),
    T('We use the corrected South German Credit data throughout', 'Folosim în tot capitolul datele corectate South German Credit')))

chart(T('Bad Rate by Checking Account', 'Rata creditelor neperformante după contul curent'), 'sfm_ch12_default_by_status', 'SFM_ch12_woe_iv', [
    T('Share of bad credits in each group; dashed: the sample average of @{badpct}\\%', 'Ponderea creditelor neperformante în fiecare grup; linia întreruptă: media eșantionului, @{badpct}\\%'),
    T('Interpretation: the risk falls from @{st1.r}\\% (no account) to @{st4.r}\\% (balance of at least 200 DM): one variable already separates the applicants well',
      'Interpretare: riscul scade de la @{st1.r}\\% (fără cont) la @{st4.r}\\% (sold de cel puțin 200 DM): o singură variabilă separă deja bine solicitanții')],
    h='0.50\\textheight')

D.frame(T('Class Imbalance and Oversampling', 'Dezechilibrul claselor și suprareprezentarea'), items(
    (T('Defaults are rare: a few percent of the loans of a bank; a sample with few bads tells little about them', 'Nerambursările sînt rare: cîteva procente din creditele unei bănci; un eșantion cu puțini rău-platnici spune puțin despre ei'),
     [T('banks therefore oversample bads (here @{badpct}\\% instead of about 5\\%)', 'de aceea băncile suprareprezintă rău-platnicii (aici @{badpct}\\% în loc de circa 5\\%)')]),
    (T('\\textbf{Accuracy paradox}: the rule ``everybody is good\'\' is right for @{allgood}\\% of this sample and for @{tw.allgood}\\% of the Taiwan data',
       '\\textbf{Paradoxul acurateței}: regula „toți sînt bun-platnici” are dreptate pentru @{allgood}\\% din acest eșantion și pentru @{tw.allgood}\\% din datele din Taiwan'),
     [T('so accuracy alone is a poor measure; we need measures that look at the bads and the goods separately (Section 7)',
        'deci acuratețea singură este o măsură slabă; avem nevoie de măsuri care privesc separat rău-platnicii și bun-platnicii (secțiunea 7)')]),
    (T('\\textbf{Taiwan credit card data} \\refTW; \\refYL: @{tw.n} card holders, default on the October 2005 payment, @{tw.rate}\\% defaults',
       '\\textbf{Datele cardurilor de credit din Taiwan} \\refTW; \\refYL: @{tw.n} de deținători de card, nerambursarea plății din octombrie 2005, @{tw.rate}\\% nerambursări'),
     [T('inputs: credit limit, age, the repayment status of April--September 2005, bills and payments; we use them for calibration and fairness',
        'variabile: limita de credit, vîrsta, starea plăților din aprilie--septembrie 2005, facturile și plățile; le folosim pentru calibrare și echitate')])))

D.frame(T('Training and Test Samples', 'Eșantionul de estimare și eșantionul de test'), items(
    (T('\\textbf{Stratified split}: @{ntrain} credits for estimation (training), @{ntest} for testing, with @{badpct}\\% bads in both',
       '\\textbf{Împărțire stratificată}: @{ntrain} de credite pentru estimare, @{ntest} pentru test, cu @{badpct}\\% rău-platnici în ambele'),
     [T('every choice (bins, WoE, variables, coefficients) is made on the training sample only', 'orice alegere (intervale, WoE, variabile, coeficienți) se face doar pe eșantionul de estimare'),
      T('the test sample is used once, at the end: it plays the role of the future applicants', 'eșantionul de test se folosește o singură dată, la final: joacă rolul viitorilor solicitanți')]),
    (T('\\textbf{Leakage}: information from the test sample that enters the model, e.g.\\ bins chosen on all data', '\\textbf{Leakage} (scurgerea de informație): informație din eșantionul de test care intră în model, de exemplu intervale alese pe toate datele'),
     [T('it makes the test result too optimistic; with 300 test credits one split is also noisy, so we add cross-validation (Section 7)',
        'face rezultatul testului prea optimist; cu 300 de credite de test, o singură împărțire este și zgomotoasă, așa că adăugăm validarea încrucișată (secțiunea 7)')])))

D.recap(('Default Data', 'datele de nerambursare'), [
    T('South German Credit: @{n} credits, @{badpct}\\% bad by design; the widely used version had wrong labels', 'South German Credit: @{n} de credite, @{badpct}\\% neperformante prin construcție; versiunea larg folosită avea etichete greșite'),
    T('Imbalance makes accuracy misleading; oversampling changes the level of PD, not the ranking', 'Dezechilibrul face acuratețea înșelătoare; suprareprezentarea schimbă nivelul PD, nu ordonarea'),
    T('Estimate on the training sample, judge on the test sample', 'Estimăm pe eșantionul de estimare, judecăm pe eșantionul de test')])

# =============================================================================
# 3. WOE ȘI IV
# =============================================================================
D.section('Weight of Evidence and Information Value', 'Weight of Evidence și Information Value')

D.frame(T('Weight of Evidence', 'Weight of Evidence (WoE)'), items(
    (T('Split a variable into bins $i = 1, \\dots, m$; $g_i$, $b_i$: goods and bads in bin $i$; $G$, $B$: their totals',
       'Împărțim o variabilă în intervale (clase) $i = 1, \\dots, m$; $g_i$, $b_i$: bun-platnicii și rău-platnicii din intervalul $i$; $G$, $B$: totalurile lor'),
     [T('$\\mathrm{WoE}_i = \\ln\\dfrac{g_i/G}{b_i/B}$: the log of the share of goods over the share of bads in the bin', '$\\mathrm{WoE}_i = \\ln\\dfrac{g_i/G}{b_i/B}$: logaritmul raportului dintre ponderea bun-platnicilor și ponderea rău-platnicilor în interval')]),
    (T('Meaning: $\\mathrm{WoE}_i = \\ln\\dfrac{g_i}{b_i} - \\ln\\dfrac{G}{B}$: the log-odds of being good in the bin minus the log-odds in the whole sample',
       'Sensul: $\\mathrm{WoE}_i = \\ln\\dfrac{g_i}{b_i} - \\ln\\dfrac{G}{B}$: logaritmul șansei de a fi bun-platnic în interval minus cel din întregul eșantion'),
     [T('$\\mathrm{WoE} > 0$: safer than average; $\\mathrm{WoE} < 0$: riskier; $\\mathrm{WoE} = 0$: no information', '$\\mathrm{WoE} > 0$: mai sigur decît media; $\\mathrm{WoE} < 0$: mai riscant; $\\mathrm{WoE} = 0$: nicio informație'),
      T('a bin without goods or without bads would give $\\pm\\infty$: add 0.5 to both counts', 'un interval fără bun-platnici sau fără rău-platnici ar da $\\pm\\infty$: adăugăm 0,5 la ambele frecvențe')]),
    (T('\\textbf{Information Value}: $\\mathrm{IV} = \\sum_i \\left(\\dfrac{g_i}{G} - \\dfrac{b_i}{B}\\right)\\mathrm{WoE}_i \\ge 0$', '\\textbf{Information Value}: $\\mathrm{IV} = \\sum_i \\left(\\dfrac{g_i}{G} - \\dfrac{b_i}{B}\\right)\\mathrm{WoE}_i \\ge 0$'),
     [T('a symmetric Kullback--Leibler divergence between the distributions of goods and bads across the bins', 'o divergență Kullback--Leibler simetrică între distribuțiile bun-platnicilor și ale rău-platnicilor pe intervale'),
      T('rule of thumb \\refSiddiqi: below 0.02 useless, 0.02--0.1 weak, 0.1--0.3 medium, above 0.3 strong', 'regula practică \\refSiddiqi: sub 0,02 inutilă, 0,02--0,1 slabă, 0,1--0,3 medie, peste 0,3 puternică')])), 'footnotesize')

D.frame(T('Worked Example: WoE and IV of the Checking Account', 'Exemplu rezolvat: WoE și IV pentru contul curent'), table(
    'lrrrrrr', T('Checking account & goods & bads & $g_i/G$ & $b_i/B$ & WoE & IV term', 'Contul curent & bun-pl. & rău-pl. & $g_i/G$ & $b_i/B$ & WoE & termen IV'),
    [T('no checking account', 'fără cont curent') + ' & @{sw1.g} & @{sw1.b} & $@{sw1.dg}$ & $@{sw1.db}$ & $@{sw1.w}$ & $@{sw1.iv}$',
     T('balance $<$ 0 DM', 'sold $<$ 0 DM') + ' & @{sw2.g} & @{sw2.b} & $@{sw2.dg}$ & $@{sw2.db}$ & $@{sw2.w}$ & $@{sw2.iv}$',
     T('0 to 200 DM', '0--200 DM') + ' & @{sw3.g} & @{sw3.b} & $@{sw3.dg}$ & $@{sw3.db}$ & $@{sw3.w}$ & $@{sw3.iv}$',
     T('$\\ge$ 200 DM or salary', '$\\ge$ 200 DM sau salariu') + ' & @{sw4.g} & @{sw4.b} & $@{sw4.dg}$ & $@{sw4.db}$ & $@{sw4.w}$ & $@{sw4.iv}$',
     T('total', 'total') + ' & 700 & @{bad} & 1 & 1 & & $@{sw.iv}$'], size='scriptsize') + items(
    (T('\\textbf{Step by step}, last row: $g/G = @{sw4.g}/700 = @{sw4.dg}$, $b/B = @{sw4.b}/@{bad} = @{sw4.db}$', '\\textbf{Pas cu pas}, ultimul rînd: $g/G = @{sw4.g}/700 = @{sw4.dg}$, $b/B = @{sw4.b}/@{bad} = @{sw4.db}$'),
     [T('WoE $= \\ln(@{sw4.dg}/@{sw4.db}) = @{sw4.w}$; IV term $= (@{sw4.dg} - @{sw4.db}) \\times @{sw4.w} = @{sw4.iv}$', 'WoE $= \\ln(@{sw4.dg}/@{sw4.db}) = @{sw4.w}$; termenul IV $= (@{sw4.dg} - @{sw4.db}) \\times @{sw4.w} = @{sw4.iv}$')]),
    T('Interpretation: IV $= @{sw.iv} > 0.3$: a strong variable; the WoE falls monotonically from the safest to the riskiest group',
      'Interpretare: IV $= @{sw.iv} > 0{,}3$: o variabilă puternică; WoE scade monoton de la grupul cel mai sigur la cel mai riscant')), 'footnotesize')

D.frame(T('Binning: Rules of the Trade', 'Discretizarea (binning): reguli practice'), items(
    (T('Why bins: they capture non-linear effects, limit the influence of outliers and give every applicant a small table of points',
       'Rolul intervalelor: surprind efectele neliniare, limitează influența valorilor extreme și dau fiecărui solicitant un tabel mic de puncte'),
     [T('numeric variables: a few intervals (here duration $\\le 12$, 13--18, 19--24, 25--36, $> 36$ months); categorical: the categories', 'variabile numerice: cîteva intervale (aici durata $\\le 12$, 13--18, 19--24, 25--36, $> 36$ de luni); categoriale: categoriile')]),
    (T('Rules \\refSiddiqi; \\refTCE', 'Reguli \\refSiddiqi; \\refTCE'),
     [T('at least about 5\\% of the sample in every bin; merge small categories', 'cel puțin aproximativ 5\\% din eșantion în fiecare interval; unim categoriile mici'),
      T('WoE should change in a way that makes economic sense (often monotonically)', 'WoE trebuie să varieze într-un mod cu sens economic (adesea monoton)'),
      T('bins and WoE are estimated on the training sample and applied unchanged to new applicants', 'intervalele și WoE se estimează pe eșantionul de estimare și se aplică neschimbate noilor solicitanți')]),
    T('A WoE variable enters the logit with one coefficient: the scorecard stays additive and easy to read', 'O variabilă WoE intră în modelul logit cu un singur coeficient: scorecard-ul rămîne aditiv și ușor de citit')))

chart(T('WoE of the Credit Duration', 'WoE pentru durata creditului'), 'sfm_ch12_woe_duration', 'SFM_ch12_woe_iv', [
    T('Full sample; bars: WoE of each duration bin; line: bad rate (right axis); IV $= @{dw.iv}$ (medium)', 'Eșantionul complet; bare: WoE pentru fiecare interval de durată; linia: rata creditelor neperformante (axa din dreapta); IV $= @{dw.iv}$ (medie)'),
    T('Interpretation: credits up to one year are safer (WoE $= @{dw0.w}$, @{dw0.r}\\% bad); above three years the bad rate reaches @{dw4.r}\\% (WoE $= @{dw4.w}$); 13--24 months are average',
      'Interpretare: creditele de pînă la un an sînt mai sigure (WoE $= @{dw0.w}$, @{dw0.r}\\% neperformante); peste trei ani, rata ajunge la @{dw4.r}\\% (WoE $= @{dw4.w}$); 13--24 de luni sînt în medie')],
    h='0.50\\textheight')

chart(T('Information Value of the 20 Variables', 'Information Value pentru cele 20 de variabile'), 'sfm_ch12_iv', 'SFM_ch12_woe_iv', [
    T('Full sample; dotted: the thresholds 0.02, 0.1 and 0.3; on the training sample, @{nsel} variables have IV $\\ge 0.1$ and enter the scorecard',
      'Eșantionul complet; linii punctate: pragurile 0,02; 0,1 și 0,3; pe eșantionul de estimare, @{nsel} variabile au IV $\\ge 0{,}1$ și intră în scorecard'),
    T('Interpretation: the checking account (IV $= @{iv.status}$) and the credit history ($@{iv.credit_history}$) dominate; telephone, residence and dependants carry no information',
      'Interpretare: contul curent (IV $= @{iv.status}$) și istoricul de credit ($@{iv.credit_history}$) domină; telefonul, vechimea la adresă și persoanele în întreținere nu aduc informație'),
    T('IV looks at one variable at a time: two strong but correlated variables do not add their information', 'IV privește o singură variabilă: două variabile puternice, dar corelate, nu își adună informația')],
    h='0.48\\textheight')

D.frame(T('Recap: Weight of Evidence and Information Value', 'Recapitulare: Weight of Evidence și Information Value'), items(
    T('WoE: log share of goods over share of bads in a bin; positive means safer than average', 'WoE: logaritmul raportului dintre ponderea bun-platnicilor și a rău-platnicilor într-un interval; pozitiv înseamnă mai sigur decît media'),
    T('IV ranks the variables; 0.1 and 0.3 mark medium and strong', 'IV ordonează variabilele; 0,1 și 0,3 marchează variabilele medii și puternice'),
    T('Bins, WoE and the selection of variables belong to the training sample', 'Intervalele, WoE și selecția variabilelor aparțin eșantionului de estimare')))

# =============================================================================
# 4. REGRESIA LOGISTICĂ
# =============================================================================
D.section('Logistic Regression', 'Regresia logistică')

D.frame(T('Odds and Log-Odds', 'Șansa (odds) și logaritmul șansei'), items(
    (T('A linear model $P(Y = 1) = \\beta_0 + x^\\top\\beta$ can give probabilities below 0 or above 1', 'Un model liniar $P(Y = 1) = \\beta_0 + x^\\top\\beta$ poate da probabilități sub 0 sau peste 1'),
     [T('we model a transformation of $p$ that can take any real value', 'modelăm o transformare a lui $p$ care poate lua orice valoare reală')]),
    (T('\\textbf{Odds} of default: $\\dfrac{p}{1 - p} \\in (0, \\infty)$; \\textbf{log-odds} (logit): $\\ln\\dfrac{p}{1 - p} \\in (-\\infty, \\infty)$',
       '\\textbf{Șansa} (odds) de nerambursare: $\\dfrac{p}{1 - p} \\in (0, \\infty)$; \\textbf{logaritmul șansei} (logit): $\\ln\\dfrac{p}{1 - p} \\in (-\\infty, \\infty)$'),
     [T('$p = @{od.p}$: odds $= 0.2/0.8 = @{od.o}$ (one default for four repaid loans); log-odds $= \\ln 0.25 = @{od.lo}$', '$p = @{od.p}$: șansa $= 0.2/0.8 = @{od.o}$ (o nerambursare la patru credite rambursate); logaritmul șansei $= \\ln 0.25 = @{od.lo}$'),
      T('$p = 0.5$: odds 1, log-odds 0; the logit is symmetric: $\\mathrm{logit}(1 - p) = -\\mathrm{logit}(p)$', '$p = 0.5$: șansa 1, logaritmul șansei 0; logit-ul este simetric: $\\mathrm{logit}(1 - p) = -\\mathrm{logit}(p)$')]),
    T('Bankers often speak of the \\textbf{good:bad odds} $(1 - p)/p$: here 4:1', 'Bancherii vorbesc adesea despre \\textbf{șansa bun:rău} $(1 - p)/p$: aici 4:1')))

D.frame(T('The Logit Model', 'Modelul logit'), items(
    (T('$\\ln\\dfrac{p_i}{1 - p_i} = \\beta_0 + x_i^\\top\\beta = \\eta_i$, \\quad $p_i = P(Y_i = 1 \\mid x_i) = \\dfrac{1}{1 + e^{-\\eta_i}}$ \\refCox',
       '$\\ln\\dfrac{p_i}{1 - p_i} = \\beta_0 + x_i^\\top\\beta = \\eta_i$, \\quad $p_i = P(Y_i = 1 \\mid x_i) = \\dfrac{1}{1 + e^{-\\eta_i}}$ \\refCox'),
     [T('$\\beta_0$: the constant (intercept); $\\eta_i$: the linear predictor, the log-odds of applicant $i$', '$\\beta_0$: termenul liber; $\\eta_i$: predictorul liniar, logaritmul șansei solicitantului $i$')]),
    (T('\\textbf{Maximum likelihood} (Chapter 6): $\\ell(\\beta) = \\sum_i [y_i\\ln p_i + (1 - y_i)\\ln(1 - p_i)]$', '\\textbf{Verosimilitatea maximă} (Capitolul 6): $\\ell(\\beta) = \\sum_i [y_i\\ln p_i + (1 - y_i)\\ln(1 - p_i)]$'),
     [T('first-order condition $\\sum_i (y_i - p_i)x_i = 0$: no closed form, solved by Newton--Raphson', 'condiția de ordinul întîi $\\sum_i (y_i - p_i)x_i = 0$: fără formulă explicită, se rezolvă prin Newton--Raphson'),
      T('standard errors from the inverse of $\\sum_i p_i(1 - p_i)x_ix_i^\\top$; Wald test $z = \\hat\\beta_j/\\mathrm{se}(\\hat\\beta_j)$', 'erorile standard din inversa lui $\\sum_i p_i(1 - p_i)x_ix_i^\\top$; testul Wald $z = \\hat\\beta_j/\\mathrm{se}(\\hat\\beta_j)$')]),
    (T('The logit is also used for corporate bankruptcy, starting with \\refOhlson; the probit uses $\\Phi(\\eta)$ instead of the logistic function', 'Modelul logit este folosit și pentru falimentul companiilor, începînd cu \\refOhlson; modelul probit folosește $\\Phi(\\eta)$ în locul funcției logistice'),
     [T('logit and probit give almost the same PDs; the logit has the odds interpretation', 'logit și probit dau aproape aceleași PD; logit-ul are interpretarea prin șanse')])))

chart(T('A Logistic Curve: PD and Duration', 'O curbă logistică: PD și durata'), 'sfm_ch12_logistic_curve', 'SFM_ch12_logit_lda', [
    T('Logit with duration only: $\\hat\\beta_0 = @{cv.b0}$, $\\hat\\beta_1 = @{cv.b1}$ per month (se $@{cv.se1}$); dots: observed bad rates', 'Logit doar cu durata: $\\hat\\beta_0 = @{cv.b0}$, $\\hat\\beta_1 = @{cv.b1}$ pe lună (se $@{cv.se1}$); punctele: ratele observate'),
    T('Interpretation: each extra year multiplies the odds of default by $e^{12 \\times @{cv.b1}} = @{cv.or12}$; the PD rises from @{cv.p12}\\% at 12 months to @{cv.p48}\\% at 48 months',
      'Interpretare: fiecare an în plus înmulțește șansa de nerambursare cu $e^{12 \\times @{cv.b1}} = @{cv.or12}$; PD crește de la @{cv.p12}\\% la 12 luni la @{cv.p48}\\% la 48 de luni')],
    h='0.50\\textheight')

D.frame(T('Interpreting the Coefficients', 'Interpretarea coeficienților'), items(
    (T('$\\beta_j$: the change in the \\textbf{log-odds} when $x_j$ grows by one unit, the other variables fixed', '$\\beta_j$: modificarea \\textbf{logaritmului șansei} cînd $x_j$ crește cu o unitate, celelalte variabile fiind fixe'),
     [T('$e^{\\beta_j}$: the \\textbf{odds ratio}: the odds are multiplied by $e^{\\beta_j}$; $100(e^{\\beta_j} - 1)\\%$ is the change in the odds', '$e^{\\beta_j}$: \\textbf{raportul șanselor} (odds ratio): șansa se înmulțește cu $e^{\\beta_j}$; $100(e^{\\beta_j} - 1)\\%$ este modificarea șansei'),
      T('a 95\\% confidence interval: $\\exp(\\hat\\beta_j \\pm 1.96\\,\\mathrm{se})$', 'un interval de încredere de 95\\%: $\\exp(\\hat\\beta_j \\pm 1.96\\,\\mathrm{se})$')]),
    (T('The effect on the \\textbf{probability} depends on where the applicant is', 'Efectul asupra \\textbf{probabilității} depinde de poziția solicitantului'),
     [T('$\\partial p/\\partial x_j = p(1 - p)\\beta_j$: largest at $p = 0.5$, small for very safe or very risky applicants', '$\\partial p/\\partial x_j = p(1 - p)\\beta_j$: maxim la $p = 0.5$, mic pentru solicitanții foarte siguri sau foarte riscanți')]),
    (T('A dummy variable (0/1): $e^{\\beta_j}$ compares the odds of the group with those of the reference group', 'O variabilă indicator (0/1): $e^{\\beta_j}$ compară șansa grupului cu cea a grupului de referință'),
     [T('the interpretation needs a model that contains the relevant variables: an omitted variable changes the coefficients', 'interpretarea cere un model care conține variabilele relevante: o variabilă omisă schimbă coeficienții')])))

D.frame(T('A Small Logit on the Full Sample', 'Un model logit mic pe eșantionul complet'), table(
    'lrrrrc', T('Variable & $\\hat\\beta$ & se & $z$ & p & odds ratio (95\\% CI)', 'Variabila & $\\hat\\beta$ & se & $z$ & p & raportul șanselor (CI 95\\%)'),
    [T('constant', 'termenul liber') + ' & $@{sm.const.b}$ & $@{sm.const.se}$ & $@{sm.const.z}$ & @{sm.const.p} & ',
     T('duration (years)', 'durata (ani)') + ' & $@{sm.duration_years.b}$ & $@{sm.duration_years.se}$ & $@{sm.duration_years.z}$ & @{sm.duration_years.p} & $@{sm.duration_years.or}$ ($@{sm.duration_years.lo}$--$@{sm.duration_years.hi}$)',
     T('age (decades)', 'vîrsta (decenii)') + ' & $@{sm.age_decades.b}$ & $@{sm.age_decades.se}$ & $@{sm.age_decades.z}$ & @{sm.age_decades.p} & $@{sm.age_decades.or}$ ($@{sm.age_decades.lo}$--$@{sm.age_decades.hi}$)',
     T('amount (1000 DM)', 'suma (1000 DM)') + ' & $@{sm.amount_1000.b}$ & $@{sm.amount_1000.se}$ & $@{sm.amount_1000.z}$ & @{sm.amount_1000.p} & $@{sm.amount_1000.or}$ ($@{sm.amount_1000.lo}$--$@{sm.amount_1000.hi}$)',
     T('no checking account', 'fără cont curent') + ' & $@{sm.no_account.b}$ & $@{sm.no_account.se}$ & $@{sm.no_account.z}$ & @{sm.no_account.p} & $@{sm.no_account.or}$ ($@{sm.no_account.lo}$--$@{sm.no_account.hi}$)',
     T('balance $<$ 0 DM', 'sold $<$ 0 DM') + ' & $@{sm.negative_balance.b}$ & $@{sm.negative_balance.se}$ & $@{sm.negative_balance.z}$ & @{sm.negative_balance.p} & $@{sm.negative_balance.or}$ ($@{sm.negative_balance.lo}$--$@{sm.negative_balance.hi}$)'],
    size='scriptsize') + items(
    T('$n = @{n}$; log-likelihood $@{sm.ll}$ against $@{sm.ll0}$ with the constant only: LR $= @{sm.lr}$ on 5 degrees of freedom', '$n = @{n}$; log-verosimilitatea $@{sm.ll}$ față de $@{sm.ll0}$ doar cu termenul liber: LR $= @{sm.lr}$ cu 5 grade de libertate'),
    T('Reference group of the account dummies: a non-negative balance (0--200 DM, or at least 200 DM or a salary account)', 'Grupul de referință al indicatorilor de cont: un sold nenegativ (0--200 DM, sau cel puțin 200 DM ori cont de salariu)')) + ql('SFM_ch12_logit_lda'), 'footnotesize')

side_chart(T('Odds Ratios with Confidence Intervals', 'Rapoartele șanselor cu intervale de încredere'), 'sfm_ch12_odds_ratios', 'SFM_ch12_logit_lda', [
    T('Each dot: $e^{\\hat\\beta}$; bars: 95\\% intervals; log scale; vertical line: no effect (1)', 'Fiecare punct: $e^{\\hat\\beta}$; bare: intervale de 95\\%; scală logaritmică; linia verticală: niciun efect (1)'),
    T('Interpretation: without a checking account the odds of default are @{sm.no_account.or} times those of the reference group; a balance below zero multiplies them by @{sm.negative_balance.or}',
      'Interpretare: fără cont curent, șansa de nerambursare este de @{sm.no_account.or} ori cea a grupului de referință; un sold negativ o înmulțește cu @{sm.negative_balance.or}'),
    T('One more year of duration raises the odds by @{sm.duration_years.pct}\\%; ten more years of age lower them by about 15\\%', 'Un an în plus de durată crește șansa cu @{sm.duration_years.pct}\\%; zece ani în plus de vîrstă o reduc cu aproximativ 15\\%'),
    T('The amount is not significant once the duration is in the model (the two are correlated)', 'Suma nu este semnificativă odată ce durata este în model (cele două sînt corelate)')],
    wl='0.54', wr='0.44', h='0.55\\textheight')

D.frame(T('Worked Example: the PD of an Applicant', 'Exemplu rezolvat: PD a unui solicitant'), items(
    (T('Applicant: 36 months, 30 years old, 4000 DM, no checking account; coefficients of the previous table, rounded', 'Solicitantul: 36 de luni, 30 de ani, 4000 DM, fără cont curent; coeficienții din tabelul anterior, rotunjiți'),
     [T('$\\eta = @{sm.const.b} + @{sm.duration_years.b} \\times 3 + (@{sm.age_decades.b}) \\times 3 + @{sm.amount_1000.b} \\times 4 + @{sm.no_account.b}$', '$\\eta = @{sm.const.b} + @{sm.duration_years.b} \\times 3 + (@{sm.age_decades.b}) \\times 3 + @{sm.amount_1000.b} \\times 4 + @{sm.no_account.b}$'),
      T('$\\eta = @{sm.const.b} + @{ex.t.duration_years} + (@{ex.t.age_decades}) + @{ex.t.amount_1000} + @{ex.t.no_account} = @{ex.eta}$', '$\\eta = @{sm.const.b} + @{ex.t.duration_years} + (@{ex.t.age_decades}) + @{ex.t.amount_1000} + @{ex.t.no_account} = @{ex.eta}$')]),
    (T('PD $= 1/(1 + e^{-@{ex.eta}}) = @{ex.pd}\\%$; odds $e^{@{ex.eta}} = @{ex.odds}$', 'PD $= 1/(1 + e^{-@{ex.eta}}) = @{ex.pd}\\%$; șansa $e^{@{ex.eta}} = @{ex.odds}$'),
     [T('the same applicant with a non-negative balance: $\\eta = @{ex.eta0}$, PD $= @{ex.pd0}\\%$', 'același solicitant cu un sold nenegativ: $\\eta = @{ex.eta0}$, PD $= @{ex.pd0}\\%$'),
      T('one more year of duration: $\\Delta p \\approx p(1 - p)\\beta \\approx @{ex.me}$ percentage points', 'un an în plus de durată: $\\Delta p \\approx p(1 - p)\\beta \\approx @{ex.me}$ puncte procentuale')]),
    T('Caution: this PD refers to a sample with @{badpct}\\% bads, not to the bank\'s population (prior correction, two slides ahead)', 'Atenție: această PD se referă la un eșantion cu @{badpct}\\% rău-platnici, nu la populația băncii (corecția pentru proporția a priori, peste două slide-uri)')))

D.frame(T('The WoE Logit Scorecard', 'Scorecard-ul logit pe variabile WoE'), table(
    'lrrr', T('WoE variable & IV (training) & $\\hat\\beta$ & se', 'Variabila WoE & IV (estimare) & $\\hat\\beta$ & se'),
    [T('constant', 'termenul liber') + ' & & $@{wl.const.b}$ & $@{wl.const.se}$'] +
    [vname(v) + f' & $@{{ivt.{v}}}$ & $@{{wl.{v}.b}}$ & $@{{wl.{v}.se}}$' for v in N['sel']], size='scriptsize') + items(
    (T('Training sample (@{ntrain} credits); each variable replaced by the WoE of its bin; logit for default', 'Eșantionul de estimare (@{ntrain} de credite); fiecare variabilă este înlocuită cu WoE al intervalului ei; logit pentru nerambursare'),
     [T('all coefficients are negative: a higher WoE (safer bin) lowers the log-odds of default', 'toți coeficienții sînt negativi: un WoE mai mare (interval mai sigur) reduce logaritmul șansei de nerambursare'),
      T('a coefficient of $-1$ would mean that the variable adds exactly its own WoE; values between $-0.7$ and $-1.1$ reflect the overlap between variables', 'un coeficient de $-1$ ar însemna că variabila adaugă exact propriul WoE; valorile între $-0{,}7$ și $-1{,}1$ reflectă suprapunerea dintre variabile')]),
    T('Test sample: AUC $= @{v.auc}$ (Section 7); in-sample log-likelihood $@{wl.ll}$', 'Eșantionul de test: AUC $= @{v.auc}$ (secțiunea 7); log-verosimilitatea în eșantion $@{wl.ll}$')) + ql('SFM_ch12_scorecard'), 'footnotesize')

D.frame(T('Correcting for Oversampling', 'Corecția pentru suprareprezentarea rău-platnicilor'), items(
    (T('If bads and goods are sampled with different rates, only the constant of the logit changes', 'Dacă rău-platnicii și bun-platnicii sînt eșantionați cu rate diferite, se schimbă doar termenul liber al modelului logit'),
     [T('$\\ln\\mathrm{odds}_{\\text{pop}} = \\ln\\mathrm{odds}_{\\text{sample}} + \\ln\\dfrac{\\pi/(1 - \\pi)}{\\pi_s/(1 - \\pi_s)}$; $\\pi$: population bad rate, $\\pi_s$: sample rate', '$\\ln\\mathrm{șansa}_{\\text{pop}} = \\ln\\mathrm{șansa}_{\\text{eșantion}} + \\ln\\dfrac{\\pi/(1 - \\pi)}{\\pi_s/(1 - \\pi_s)}$; $\\pi$: rata din populație, $\\pi_s$: rata din eșantion'),
      T('the slopes are unchanged: the ranking of the applicants does not depend on the sampling', 'pantele rămîn neschimbate: ordonarea solicitanților nu depinde de eșantionare')]),
    (T('\\textbf{Step by step}: $\\pi_s = 0.30$, $\\pi = 0.05$: shift $= \\ln(0.05/0.95) - \\ln(0.30/0.70) = @{pr.ln} - (@{pr.ls}) = @{prior}$', '\\textbf{Pas cu pas}: $\\pi_s = 0.30$, $\\pi = 0.05$: corecția $= \\ln(0.05/0.95) - \\ln(0.30/0.70) = @{pr.ln} - (@{pr.ls}) = @{prior}$'),
     [T('a sample PD of 30\\% becomes 5\\%; a sample PD of 50\\% becomes @{pr.p50}\\%', 'o PD de 30\\% în eșantion devine 5\\%; o PD de 50\\% devine @{pr.p50}\\%'),
      T('the applicant of the worked example: $@{ex.eta} - @{prior.abs} = @{ex.etapop}$, PD $= @{ex.pdpop}\\%$ in the bank\'s population instead of @{ex.pd}\\%', 'solicitantul din exemplul lucrat: $@{ex.eta} - @{prior.abs} = @{ex.etapop}$, PD $= @{ex.pdpop}\\%$ în populația băncii, în loc de @{ex.pd}\\%')]),
    T('Interpretation: AUC and Gini are unchanged by the correction; EL and capital are not', 'Interpretare: AUC și Gini nu sînt afectate de corecție; EL și capitalul sînt afectate')))

D.recap(('Logistic Regression', 'regresia logistică'), [
    T('The logit models the log-odds linearly; $e^{\\beta_j}$ is an odds ratio', 'Modelul logit modelează liniar logaritmul șansei; $e^{\\beta_j}$ este un raport al șanselor'),
    T('Estimated by maximum likelihood; the effect on PD is $p(1 - p)\\beta_j$', 'Se estimează prin verosimilitate maximă; efectul asupra PD este $p(1 - p)\\beta_j$'),
    T('Oversampling moves only the constant: correct it before using PDs for EL or capital', 'Suprareprezentarea mută doar termenul liber: o corectăm înainte de a folosi PD pentru EL sau capital')])

# =============================================================================
# 5. LDA
# =============================================================================
D.section('Linear Discriminant Analysis', 'Analiza discriminantă liniară')

D.frame(T('Fisher\'s Idea (1936)', 'Ideea lui Fisher (1936)'), cols(items(
    (T('\\refFisher: find the linear combination $w^\\top x$ that separates two groups best', '\\refFisher: găsim combinația liniară $w^\\top x$ care separă cel mai bine două grupuri'),
     [T('he used measurements of iris flowers; Durand applied it to loans five years later', 'a folosit măsurători ale florilor de iris; Durand a aplicat metoda creditelor cinci ani mai tîrziu')]),
    (T('Fisher criterion: $J(w) = \\dfrac{[w^\\top(m_1 - m_0)]^2}{w^\\top S_W w}$', 'Criteriul Fisher: $J(w) = \\dfrac{[w^\\top(m_1 - m_0)]^2}{w^\\top S_W w}$'),
     [T('$m_0$, $m_1$: mean vectors of goods and bads; $S_W$: the pooled within-group covariance matrix', '$m_0$, $m_1$: vectorii mediilor pentru bun-platnici și rău-platnici; $S_W$: matricea de covarianță comună din interiorul grupurilor'),
      T('maximum: $w \\propto S_W^{-1}(m_1 - m_0)$ \\refMVA, Ch.~14', 'maximul: $w \\propto S_W^{-1}(m_1 - m_0)$ \\refMVA, cap.~14')]),
    T('No distribution is needed for this criterion: it is a projection that maximises the distance between the groups relative to their spread',
      'Criteriul nu cere nicio distribuție: este o proiecție care maximizează distanța dintre grupuri raportată la dispersia lor')),
    cols(ph('fisher', T('R.A. Fisher, 1913', 'R.A. Fisher, 1913'), h='0.34\\textheight'),
         ph('iris', T('Iris versicolor', 'Iris versicolor'), h='0.30\\textheight'), wl='0.48', wr='0.48'),
    wl='0.55', wr='0.43'), 'footnotesize')

D.frame(T('LDA as a Probability Model', 'LDA ca model de probabilitate'), items(
    (T('Assume $x \\mid Y = k \\sim N(m_k, \\Sigma)$, the same $\\Sigma$ in both groups, and prior $\\pi_1 = P(Y = 1)$', 'Presupunem $x \\mid Y = k \\sim N(m_k, \\Sigma)$, aceeași $\\Sigma$ în ambele grupuri, și probabilitatea a priori $\\pi_1 = P(Y = 1)$'),
     [T('Bayes: $\\ln\\dfrac{P(Y = 1 \\mid x)}{P(Y = 0 \\mid x)} = \\underbrace{\\ln\\dfrac{\\pi_1}{1 - \\pi_1} - \\tfrac12(m_1 + m_0)^\\top w}_{c} + w^\\top x$, \\quad $w = \\Sigma^{-1}(m_1 - m_0)$',
        'Bayes: $\\ln\\dfrac{P(Y = 1 \\mid x)}{P(Y = 0 \\mid x)} = \\underbrace{\\ln\\dfrac{\\pi_1}{1 - \\pi_1} - \\tfrac12(m_1 + m_0)^\\top w}_{c} + w^\\top x$, \\quad $w = \\Sigma^{-1}(m_1 - m_0)$'),
      T('the log-odds are \\textbf{linear in} $x$, exactly as in the logit; the prior enters only the constant', 'logaritmul șansei este \\textbf{liniar în} $x$, exact ca în modelul logit; probabilitatea a priori intră doar în termenul liber')]),
    (T('The difference is in the estimation', 'Diferența este în estimare'),
     [T('LDA: plug in the sample means and the pooled covariance (a closed form)', 'LDA: folosim direct mediile de selecție și covarianța comună (formulă explicită)'),
      T('logit: maximise the conditional likelihood of $Y$ given $x$; no assumption on the distribution of $x$', 'logit: maximizăm verosimilitatea condiționată a lui $Y$ dată fiind $x$; nicio ipoteză asupra distribuției lui $x$')]),
    T('Classification rule: assign to ``bad\'\' if $c + w^\\top x > \\ln(C_{FP}/C_{FN})$ (costs, Section 7)', 'Regula de clasificare: clasificăm solicitantul ca „rău-platnic” dacă $c + w^\\top x > \\ln(C_{FP}/C_{FN})$ (costuri, secțiunea 7)')), 'footnotesize')

D.frame(T('Worked Example: Fisher\'s Direction with Two Variables', 'Exemplu rezolvat: direcția Fisher cu două variabile'), items(
    (T('$x = $ (duration in months, age in years), full sample: @{l2.n0} goods, @{l2.n1} bads', '$x = $ (durata în luni, vîrsta în ani), eșantionul complet: @{l2.n0} de bun-platnici, @{l2.n1} de rău-platnici'),
     [T('$m_0 = (@{l2.m0d}, @{l2.m0a})$, $m_1 = (@{l2.m1d}, @{l2.m1a})$, $m_1 - m_0 = (@{l2.dd}, @{l2.da})$', '$m_0 = (@{l2.m0d}, @{l2.m0a})$, $m_1 = (@{l2.m1d}, @{l2.m1a})$, $m_1 - m_0 = (@{l2.dd}, @{l2.da})$'),
      T('$S_W = \\begin{pmatrix} @{l2.s11} & @{l2.s12} \\\\ @{l2.s12} & @{l2.s22} \\end{pmatrix}$, $\\det S_W = @{l2.det}$', '$S_W = \\begin{pmatrix} @{l2.s11} & @{l2.s12} \\\\ @{l2.s12} & @{l2.s22} \\end{pmatrix}$, $\\det S_W = @{l2.det}$')]),
    (T('\\textbf{Step by step}: $S_W^{-1} = \\dfrac{1}{\\det S_W}\\begin{pmatrix} @{l2.s22} & -(@{l2.s12}) \\\\ -(@{l2.s12}) & @{l2.s11} \\end{pmatrix}$, $w = S_W^{-1}(m_1 - m_0) = (@{l2.w1}, @{l2.w2})$',
       '\\textbf{Pas cu pas}: $S_W^{-1} = \\dfrac{1}{\\det S_W}\\begin{pmatrix} @{l2.s22} & -(@{l2.s12}) \\\\ -(@{l2.s12}) & @{l2.s11} \\end{pmatrix}$, $w = S_W^{-1}(m_1 - m_0) = (@{l2.w1}, @{l2.w2})$'),
     [T('logit with the same two variables: slopes $(@{l2.lb1}, @{l2.lb2})$', 'logit cu aceleași două variabile: pantele $(@{l2.lb1}, @{l2.lb2})$'),
      T('ratio duration/age: LDA $@{l2.ratio}$, logit $@{l2.lratio}$: one month of duration weighs as much as about two years of age, in the other direction',
        'raportul durată/vîrstă: LDA $@{l2.ratio}$, logit $@{l2.lratio}$: o lună de durată cîntărește cît aproximativ doi ani de vîrstă, în sens invers')])), 'footnotesize')

chart(T('Fisher\'s Boundary and the Logit Boundary', 'Frontiera Fisher și frontiera logit'), 'sfm_ch12_lda_2d', 'SFM_ch12_logit_lda', [
    T('Duration and age of every credit (a small jitter added); lines: the points with PD $= 30\\%$ (the sample rate) for the LDA and for the logit',
      'Durata și vîrsta fiecărui credit (cu o mică perturbare); liniile: punctele cu PD $= 30\\%$ (rata din eșantion) pentru LDA și pentru logit'),
    T('Interpretation: both boundaries are lines with almost the same slope; the two clouds overlap heavily, so two variables are far from enough',
      'Interpretare: ambele frontiere sînt drepte cu aproape aceeași pantă; cei doi nori se suprapun mult, deci două variabile sînt departe de a fi suficiente')],
    h='0.50\\textheight')

D.frame(T('Logit or LDA?', 'Logit sau LDA?'), table(
    'p{3.0cm}p{3.9cm}p{3.9cm}', T('& \\textbf{Logit} & \\textbf{LDA (Fisher)}', '& \\textbf{Logit} & \\textbf{LDA (Fisher)}'),
    [T('model', 'modelul') + ' & ' + T('$P(Y \\mid x)$ only', 'doar $P(Y \\mid x)$') + ' & ' + T('$P(x \\mid Y)$ Normal, common $\\Sigma$', '$P(x \\mid Y)$ Normală, $\\Sigma$ comun'),
     T('estimation', 'estimarea') + ' & ' + T('maximum likelihood, iterative', 'verosimilitate maximă, iterativ') + ' & ' + T('means and covariance, closed form', 'medii și covarianță, formulă explicită'),
     T('if $x$ is Normal', 'dacă $x$ este Normal') + ' & ' + T('consistent, less efficient', 'consistent, mai puțin eficient') + ' & ' + T('more efficient', 'mai eficient'),
     T('dummies, WoE, skewed $x$', 'indicatori, WoE, $x$ asimetric') + ' & ' + T('fine', 'potrivit') + ' & ' + T('ranking fine, PDs may be off', 'ordonarea bună, PD pot fi greșite'),
     T('test AUC (WoE inputs)', 'AUC de test (variabile WoE)') + ' & $@{v.auc}$ & $@{v.auc_lda}$'], size='scriptsize') + items(
    T('Test sample: the correlation of the two log-odds scores is $@{v.corr_logit_lda}$; DeLong test of equal AUCs: $z = @{v.dl.z}$, p $= @{v.dl.p}$ \\refDeLong',
      'Eșantionul de test: corelația dintre cele două scoruri (logaritmul șansei) este $@{v.corr_logit_lda}$; testul DeLong de egalitate a AUC: $z = @{v.dl.z}$, p $= @{v.dl.p}$ \\refDeLong'),
    T('Interpretation: for ranking, the two methods are practically identical; the logit is preferred for PDs because it does not need Normal inputs',
      'Interpretare: pentru ordonare, cele două metode sînt practic identice; logit-ul este preferat pentru PD, deoarece nu cere variabile Normale')) + ql('SFM_ch12_logit_lda'), 'footnotesize')

D.recap(('Linear Discriminant Analysis', 'analiza discriminantă liniară'), [
    T('Fisher: $w \\propto S_W^{-1}(m_1 - m_0)$ maximises the separation of the groups', 'Fisher: $w \\propto S_W^{-1}(m_1 - m_0)$ maximizează separarea grupurilor'),
    T('With Normal classes and a common covariance, LDA has linear log-odds, like the logit', 'Cu clase Normale și covarianță comună, LDA are logaritmul șansei liniar, ca logit-ul'),
    T('On our data both rank equally well; the logit gives more reliable PDs', 'Pe datele noastre, ambele ordonează la fel de bine; logit-ul dă PD mai fiabile')])

# =============================================================================
# 6. SCORECARD
# =============================================================================
D.section('From Model to Scorecard', 'De la model la scorecard')

D.frame(T('Scaling a Scorecard', 'Scalarea unui scorecard'), items(
    (T('\\textbf{Scorecard}: a table that gives points to every attribute; the score of an applicant is the sum of the points \\refSiddiqi',
       '\\textbf{Scorecard} (grila de punctaj): un tabel care dă puncte fiecărui atribut; scorul unui solicitant este suma punctelor \\refSiddiqi'),
     [T('higher score $=$ lower risk; the score is a linear function of the good:bad log-odds', 'scor mai mare $=$ risc mai mic; scorul este o funcție liniară de logaritmul șansei bun:rău')]),
    (T('$\\text{Score} = \\text{Offset} + \\text{Factor} \\times \\ln\\dfrac{1 - p}{p}$, two conventions fix the scale', '$\\text{Scor} = \\text{Offset} + \\text{Factor} \\times \\ln\\dfrac{1 - p}{p}$, două convenții fixează scala'),
     [T('\\textbf{PDO} (points to double the odds): $+$PDO points $\\Leftrightarrow$ the good:bad odds double: Factor $=$ PDO$/\\ln 2$', '\\textbf{PDO} (points to double the odds): $+$PDO puncte $\\Leftrightarrow$ șansa bun:rău se dublează: Factor $=$ PDO$/\\ln 2$'),
      T('a base score at given odds: Offset $=$ Base $-$ Factor $\\ln(\\text{odds}_0)$', 'un scor de bază la o șansă dată: Offset $=$ Baza $-$ Factor $\\ln(\\text{șansa}_0)$')]),
    (T('Here: PDO $= 20$, 600 points at odds 50:1', 'Aici: PDO $= 20$, 600 de puncte la șansa 50:1'),
     [T('Factor $= 20/\\ln 2 = @{sc.f}$; Offset $= 600 - @{sc.f} \\times \\ln 50 = 600 - @{sc.f} \\times @{sc.ln50} = @{sc.o}$', 'Factor $= 20/\\ln 2 = @{sc.f}$; Offset $= 600 - @{sc.f} \\times \\ln 50 = 600 - @{sc.f} \\times @{sc.ln50} = @{sc.o}$')])))

D.frame(T('Worked Example: Scores and Points', 'Exemplu rezolvat: scoruri și puncte'), items(
    (T('\\textbf{PD to score}: PD $= 5\\%$, odds 19:1: Score $= @{sc.o} + @{sc.f} \\times \\ln 19 = @{sc.o} + @{sc.f} \\times @{sc.ln19} = @{sc.s5}$',
       '\\textbf{De la PD la scor}: PD $= 5\\%$, șansa 19:1: Scorul $= @{sc.o} + @{sc.f} \\times \\ln 19 = @{sc.o} + @{sc.f} \\times @{sc.ln19} = @{sc.s5}$'),
     [T('PD $= 30\\%$ (odds 7:3): $@{sc.o} + @{sc.f} \\times @{sc.ln73} = @{sc.s30}$', 'PD $= 30\\%$ (șansa 7:3): $@{sc.o} + @{sc.f} \\times @{sc.ln73} = @{sc.s30}$')]),
    T('\\textbf{Score to PD}: 560 points: $\\ln$ odds $= (560 - @{sc.o})/@{sc.f} = @{sc.lo560}$, odds $@{sc.od560}$:1, PD $= @{sc.pd560}\\%$',
       '\\textbf{De la scor la PD}: 560 de puncte: $\\ln$ șansa $= (560 - @{sc.o})/@{sc.f} = @{sc.lo560}$, șansa $@{sc.od560}$:1, PD $= @{sc.pd560}\\%$'),
    (T('\\textbf{Points of an attribute} ($k = @{pt.k}$ variables; logit for default with constant $\\beta_0$ and slope $\\beta_j$)', '\\textbf{Punctele unui atribut} ($k = @{pt.k}$ variabile; logit pentru nerambursare cu termenul liber $\\beta_0$ și panta $\\beta_j$)'),
     [T('$\\text{points}_{ij} = -(\\beta_j\\,\\mathrm{WoE}_{ij} + \\beta_0/k)\\,\\text{Factor} + \\text{Offset}/k$; the minus sign turns default log-odds into good:bad log-odds',
        '$\\text{puncte}_{ij} = -(\\beta_j\\,\\mathrm{WoE}_{ij} + \\beta_0/k)\\,\\text{Factor} + \\text{Offset}/k$; semnul minus transformă logaritmul șansei de nerambursare în logaritmul șansei bun:rău'),
      T('checking account (training WoE): no account @{pt.1} points, balance $<$ 0 @{pt.2}, 0--200 DM @{pt.3}, at least 200 DM or salary @{pt.4}',
        'contul curent (WoE din estimare): fără cont @{pt.1} de puncte, sold $<$ 0 @{pt.2}, 0--200 DM @{pt.3}, cel puțin 200 DM sau salariu @{pt.4}')]),
    T('Prior correction in points: for a population with 5\\% bads the default log-odds fall by $@{prior.abs}$, so the good:bad log-odds rise by $@{prior.abs}$ and every score rises by $@{sc.f} \\times @{prior.abs} = @{prior.pts}$ points',
      'Corecția a priori în puncte: pentru o populație cu 5\\% rău-platnici, logaritmul șansei de nerambursare scade cu $@{prior.abs}$, deci logaritmul șansei bun:rău crește cu $@{prior.abs}$ și fiecare scor crește cu $@{sc.f} \\times @{prior.abs} = @{prior.pts}$ puncte')) + ql('SFM_ch12_scorecard'), 'footnotesize')

chart(T('Scores of the Test Sample', 'Scorurile eșantionului de test'), 'sfm_ch12_score_dist', 'SFM_ch12_scorecard', [
    T('@{ntest} test credits; scores from @{sd.min} to @{sd.max} points; mean: goods @{sd.g}, bads @{sd.b}; dashed: the cut-off at PD $= 1/6$ (Section 7), @{sd.cut} points',
      '@{ntest} de credite de test; scoruri de la @{sd.min} la @{sd.max} de puncte; media: bun-platnici @{sd.g}, rău-platnici @{sd.b}; linia întreruptă: pragul la PD $= 1/6$ (secțiunea 7), @{sd.cut} de puncte'),
    T('Interpretation: the two distributions are shifted but overlap: every cut-off rejects some goods and accepts some bads; the validation measures quantify this overlap',
      'Interpretare: cele două distribuții sînt deplasate, dar se suprapun: orice prag respinge unii bun-platnici și acceptă unii rău-platnici; măsurile de validare cuantifică această suprapunere')],
    h='0.50\\textheight')

# =============================================================================
# 7. VALIDARE
# =============================================================================
D.section('Validation', 'Validarea')

D.frame(T('The Confusion Matrix', 'Matricea de confuzie'), cols(table(
    'lcc', T('& predicted bad (reject) & predicted good (accept)', '& prezis rău (respins) & prezis bun (acceptat)'),
    [T('actual bad', 'rău-platnic real') + ' & TP & FN', T('actual good', 'bun-platnic real') + ' & FP & TN'], size='scriptsize'), items(
    T('A cut-off $c$: reject if PD $> c$', 'Un prag $c$: respingem dacă PD $> c$'),
    T('TP (true positive): a bad rejected; FN (false negative): a bad accepted', 'TP (true positive): un rău-platnic respins; FN (false negative): un rău-platnic acceptat'),
    T('FP (false positive): a good rejected; TN (true negative): a good accepted', 'FP (false positive): un bun-platnic respins; TN (true negative): un bun-platnic acceptat')), wl='0.50', wr='0.48') + items(
    (T('Rates', 'Ratele'),
     [T('\\textbf{TPR} (true positive rate, sensitivity) $= \\mathrm{TP}/(\\mathrm{TP} + \\mathrm{FN})$: share of bads caught', '\\textbf{TPR} (true positive rate, sensibilitatea) $= \\mathrm{TP}/(\\mathrm{TP} + \\mathrm{FN})$: ponderea rău-platnicilor identificați'),
      T('\\textbf{FPR} (false positive rate) $= \\mathrm{FP}/(\\mathrm{FP} + \\mathrm{TN}) = 1 -$ specificity: share of goods rejected', '\\textbf{FPR} (false positive rate) $= \\mathrm{FP}/(\\mathrm{FP} + \\mathrm{TN}) = 1 -$ specificitatea: ponderea bun-platnicilor respinși'),
      T('precision $= \\mathrm{TP}/(\\mathrm{TP} + \\mathrm{FP})$; accuracy $= (\\mathrm{TP} + \\mathrm{TN})/n$', 'precizia $= \\mathrm{TP}/(\\mathrm{TP} + \\mathrm{FP})$; acuratețea $= (\\mathrm{TP} + \\mathrm{TN})/n$')]),
    T('Every rate depends on $c$; the measures that follow summarise all cut-offs at once', 'Fiecare rată depinde de $c$; măsurile următoare rezumă toate pragurile deodată')), 'footnotesize')

D.frame(T('Two Cut-Offs on the Test Sample', 'Două praguri pe eșantionul de test'), table(
    'lrrrrrrrr', T('cut-off & TP & FN & FP & TN & accuracy & TPR & FPR & cost', 'prag & TP & FN & FP & TN & acuratețe & TPR & FPR & cost'),
    ['PD $> 0.5$ & @{cm50.TP} & @{cm50.FN} & @{cm50.FP} & @{cm50.TN} & $@{cm50.acc}\\%$ & $@{cm50.tpr}\\%$ & $@{cm50.fpr}\\%$ & $@{cm50.cost}$',
     'PD $> 1/6$ & @{cm_cost.TP} & @{cm_cost.FN} & @{cm_cost.FP} & @{cm_cost.TN} & $@{cm_cost.acc}\\%$ & $@{cm_cost.tpr}\\%$ & $@{cm_cost.fpr}\\%$ & $@{cm_cost.cost}$'],
    size='scriptsize') + items(
    (T('Cost per applicant $= (5\\,\\mathrm{FN} + 1\\,\\mathrm{FP})/n$: accepting a bad costs five times as much as rejecting a good \\refGroemping',
       'Costul pe solicitant $= (5\\,\\mathrm{FN} + 1\\,\\mathrm{FP})/n$: acceptarea unui rău-platnic costă de cinci ori mai mult decît respingerea unui bun-platnic \\refGroemping'),
     [T('the Bayes cut-off minimises the expected cost: reject if $p\\,C_{FN} > (1 - p)\\,C_{FP}$, i.e.\\ $p > C_{FP}/(C_{FP} + C_{FN}) = 1/6$',
        'pragul Bayes minimizează costul așteptat: respingem dacă $p\\,C_{FN} > (1 - p)\\,C_{FP}$, adică $p > C_{FP}/(C_{FP} + C_{FN}) = 1/6$')]),
    T('Interpretation: the cut-off of 1/6 has a lower accuracy (@{cm_cost.acc}\\% against @{cm50.acc}\\%) but a much lower cost (@{cm_cost.cost} against @{cm50.cost}): it catches @{cm_cost.tpr}\\% of the bads',
      'Interpretare: pragul de 1/6 are o acuratețe mai mică (@{cm_cost.acc}\\% față de @{cm50.acc}\\%), dar un cost mult mai mic (@{cm_cost.cost} față de @{cm50.cost}): identifică @{cm_cost.tpr}\\% din rău-platnici')) + ql('SFM_ch12_validation_irb'), 'footnotesize')

chart(T('Expected Cost and the Choice of the Cut-Off', 'Costul așteptat și alegerea pragului'), 'sfm_ch12_cost_cutoff', 'SFM_ch12_validation_irb', [
    T('Test sample; red: cost per applicant; dashed: accept everybody (@{cost.acc}) or reject everybody (@{cost.rej}); blue: share rejected (right axis)',
      'Eșantionul de test; roșu: costul pe solicitant; linii întrerupte: acceptăm pe toată lumea (@{cost.acc}) sau respingem pe toată lumea (@{cost.rej}); albastru: ponderea respinșilor (axa din dreapta)'),
    T('Interpretation: the cost is flat between PD 0.1 and 0.3; the minimum on these 300 credits is at @{cost.best}, close to the Bayes cut-off of 1/6, which is chosen before seeing the test data',
      'Interpretare: costul este aproape constant între PD 0,1 și 0,3; minimul pe aceste 300 de credite este la @{cost.best}, aproape de pragul Bayes de 1/6, care se alege înainte de a vedea datele de test')],
    h='0.50\\textheight')

D.frame(T('ROC Curve and AUC', 'Curba ROC și AUC'), items(
    (T('\\textbf{ROC} (receiver operating characteristic): the points (FPR$(c)$, TPR$(c)$) for all cut-offs $c$', '\\textbf{ROC} (receiver operating characteristic): punctele (FPR$(c)$, TPR$(c)$) pentru toate pragurile $c$'),
     [T('from (0, 0) (reject nobody) to (1, 1) (reject everybody); the diagonal is a random score', 'de la (0, 0) (nu respingem pe nimeni) la (1, 1) (respingem pe toată lumea); diagonala este un scor aleator')]),
    (T('\\textbf{AUC} (area under the curve) $= P(s_{\\text{bad}} > s_{\\text{good}}) + \\tfrac12 P(s_{\\text{bad}} = s_{\\text{good}})$ \\refHM', '\\textbf{AUC} (aria de sub curbă) $= P(s_{\\text{rău}} > s_{\\text{bun}}) + \\tfrac12 P(s_{\\text{rău}} = s_{\\text{bun}})$ \\refHM'),
     [T('$s$: the risk score (here the PD); compute it by counting pairs: the share of (bad, good) pairs ordered correctly', '$s$: scorul de risc (aici PD); se calculează numărînd perechile: ponderea perechilor (rău, bun) ordonate corect'),
      T('the Mann--Whitney statistic divided by $n_1 n_0$; 0.5: no discrimination, 1: perfect', 'statistica Mann--Whitney împărțită la $n_1 n_0$; 0,5: nicio discriminare, 1: perfectă')]),
    (T('\\textbf{Gini} $= 2\\,\\mathrm{AUC} - 1$; AUC does not depend on the bad rate or on a monotone rescaling of the score', '\\textbf{Gini} $= 2\\,\\mathrm{AUC} - 1$; AUC nu depinde de rata rău-platnicilor sau de o transformare monotonă a scorului'),
     [T('standard error and tests of AUC: \\refDeLong', 'eroarea standard și testele pentru AUC: \\refDeLong')])))

side_chart(T('ROC Curves on the Test Sample', 'Curbele ROC pe eșantionul de test'), 'sfm_ch12_roc', 'SFM_ch12_validation_irb', [
    T('@{ntest} test credits, @{pairs} (bad, good) pairs', '@{ntest} de credite de test, @{pairs} de perechi (rău, bun)'),
    T('WoE logit: AUC $= @{v.auc}$, 95\\% CI $[@{v.auc.lo}, @{v.auc.hi}]$ (DeLong, se $@{v.auc.se}$); Gini $= @{v.gini}$', 'Logit WoE: AUC $= @{v.auc}$, CI 95\\% $[@{v.auc.lo}, @{v.auc.hi}]$ (DeLong, se $@{v.auc.se}$); Gini $= @{v.gini}$'),
    T('Checking account alone: AUC $= @{v.auc_single}$; DeLong test against the scorecard: $z = @{v.ds.z}$, p $= @{v.ds.p}$', 'Doar contul curent: AUC $= @{v.auc_single}$; testul DeLong față de scorecard: $z = @{v.ds.z}$, p $= @{v.ds.p}$'),
    T('Interpretation: a random bad has a higher PD than a random good in about 80\\% of the pairs; the other five variables add a significant improvement over the checking account',
      'Interpretare: un rău-platnic ales la întîmplare are o PD mai mare decît un bun-platnic ales la întîmplare în aproximativ 80\\% din perechi; celelalte cinci variabile aduc o îmbunătățire semnificativă față de contul curent')],
    wl='0.47', wr='0.51', h='0.64\\textheight')

D.frame(T('CAP Curve, Accuracy Ratio and KS', 'Curba CAP, raportul de acuratețe și KS'), items(
    (T('\\textbf{CAP} (cumulative accuracy profile): sort the applicants from the riskiest; plot the share of all bads among the first $x\\%$',
       '\\textbf{CAP} (cumulative accuracy profile): ordonăm solicitanții de la cel mai riscant; reprezentăm ponderea tuturor rău-platnicilor printre primii $x\\%$'),
     [T('perfect model: all bads first, the curve reaches 1 at $x = $ bad rate; random: the diagonal', 'modelul perfect: toți rău-platnicii primii, curba atinge 1 la $x = $ rata rău-platnicilor; aleator: diagonala'),
      T('\\textbf{AR} (accuracy ratio) $= \\dfrac{\\text{area between model CAP and diagonal}}{\\text{area between perfect CAP and diagonal}}$; for a continuous score AR $=$ Gini',
        '\\textbf{AR} (accuracy ratio, raportul de acuratețe) $= \\dfrac{\\text{aria dintre CAP-ul modelului și diagonală}}{\\text{aria dintre CAP-ul perfect și diagonală}}$; pentru un scor continuu AR $=$ Gini')]),
    (T('\\textbf{KS} (Kolmogorov--Smirnov): $\\max_s |F_{\\text{bad}}(s) - F_{\\text{good}}(s)|$, the largest gap between the score distributions of bads and goods',
       '\\textbf{KS} (Kolmogorov--Smirnov): $\\max_s |F_{\\text{rău}}(s) - F_{\\text{bun}}(s)|$, cea mai mare distanță dintre distribuțiile scorurilor rău-platnicilor și bun-platnicilor'),
     [T('$F_{\\text{bad}}$, $F_{\\text{good}}$: the empirical distribution functions of the scores in the two groups; KS $\\in [0, 1]$', '$F_{\\text{rău}}$, $F_{\\text{bun}}$: funcțiile de repartiție empirice ale scorurilor în cele două grupuri; KS $\\in [0, 1]$'),
      T('read on the ROC curve: KS $= \\max_c\\,(\\mathrm{TPR} - \\mathrm{FPR})$; the score where it occurs is a natural first cut-off', 'citit pe curba ROC: KS $= \\max_c\\,(\\mathrm{TPR} - \\mathrm{FPR})$; scorul la care apare este un prim prag natural')]),
    T('All three measure \\textbf{discrimination} (ranking), not calibration', 'Toate cele trei măsoară \\textbf{discriminarea} (ordonarea), nu calibrarea')))

D.frame(T('CAP and KS on the Test Sample', 'CAP și KS pe eșantionul de test'), cols(
    f'\\includegraphics[width=\\textwidth,height=0.50\\textheight,keepaspectratio]{{sfm_ch12_cap.pdf}}',
    f'\\includegraphics[width=\\textwidth,height=0.50\\textheight,keepaspectratio]{{sfm_ch12_ks.pdf}}', wl='0.44', wr='0.54') + items(
    T('AR $= @{v.ar}$ (Gini $= @{v.gini}$: the same up to rounding); KS $= @{ks.ks}$ at @{ks.at} points', 'AR $= @{v.ar}$ (Gini $= @{v.gini}$: același, cu excepția rotunjirii); KS $= @{ks.ks}$ la @{ks.at} de puncte'),
    T('Interpretation: rejecting the riskiest 30\\% catches about two thirds of the bads; at @{ks.at} points the scorecard separates bads and goods best',
      'Interpretare: respingînd cei mai riscanți 30\\%, identificăm aproximativ două treimi din rău-platnici; la @{ks.at} de puncte, scorecard-ul separă cel mai bine rău-platnicii de bun-platnici')) + ql('SFM_ch12_validation_irb'), 'footnotesize')

D.frame(T('Calibration and the Brier Score', 'Calibrarea și scorul Brier'), items(
    (T('\\textbf{Calibration}: among applicants with PD $\\approx p$, a share of about $p$ defaults', '\\textbf{Calibrarea}: dintre solicitanții cu PD $\\approx p$, aproximativ o pondere $p$ intră în nerambursare'),
     [T('reliability diagram: sort by PD, form groups (e.g.\\ ten), plot the mean PD against the observed default rate', 'diagrama de calibrare: ordonăm după PD, formăm grupuri (de exemplu zece), reprezentăm PD medie față de rata observată'),
      T('Hosmer--Lemeshow \\refHL: $\\sum_g \\dfrac{(O_g - n_g\\bar p_g)^2}{n_g\\bar p_g(1 - \\bar p_g)} \\approx \\chi^2(G - 2)$', 'Hosmer--Lemeshow \\refHL: $\\sum_g \\dfrac{(O_g - n_g\\bar p_g)^2}{n_g\\bar p_g(1 - \\bar p_g)} \\approx \\chi^2(G - 2)$'),
      T('$G$ groups; $n_g$: size, $O_g$: observed defaults, $\\bar p_g$: mean PD of group $g$; a large value: poor calibration', '$G$ grupuri; $n_g$: mărimea, $O_g$: nerambursările observate, $\\bar p_g$: PD medie a grupului $g$; o valoare mare: calibrare slabă')]),
    (T('\\textbf{Brier score} \\refBrier: $\\mathrm{BS} = \\frac1n\\sum_i (p_i - y_i)^2$, the mean squared error of the PDs; lower is better', '\\textbf{Scorul Brier} \\refBrier: $\\mathrm{BS} = \\frac1n\\sum_i (p_i - y_i)^2$, eroarea pătratică medie a PD; o valoare mai mică este mai bună'),
     [T('reference: a constant PD equal to the training bad rate; skill $= 1 - \\mathrm{BS}/\\mathrm{BS}_{\\text{ref}}$', 'referința: o PD constantă egală cu rata din eșantionul de estimare; cîștigul $= 1 - \\mathrm{BS}/\\mathrm{BS}_{\\text{ref}}$'),
      T('scorecard test sample: BS $= @{v.brier}$ against $@{v.brier_ref}$ (skill @{v.bss}\\%); HL with 10 groups $= @{v.hl}$, p $= @{v.hl_p}$', 'eșantionul de test al scorecard-ului: BS $= @{v.brier}$ față de $@{v.brier_ref}$ (cîștig @{v.bss}\\%); HL cu 10 grupuri $= @{v.hl}$, p $= @{v.hl_p}$')]),
    T('Brier rewards both ranking and calibration; AUC sees only the ranking', 'Scorul Brier recompensează atît ordonarea, cît și calibrarea; AUC măsoară doar ordonarea')))

side_chart(T('Calibration on the Taiwan Data', 'Calibrarea pe datele din Taiwan'), 'sfm_ch12_calibration', 'SFM_ch12_validation_irb', [
    T('Logit on @{tw.n} card holders (70\\% estimation); test sample of @{tc.n}; ten groups of 900 by PD; bars: 95\\% intervals', 'Logit pe @{tw.n} de deținători de card (70\\% estimare); eșantion de test de @{tc.n}; zece grupuri de cîte 900 după PD; bare: intervale de 95\\%'),
    T('Test AUC $= @{tc.auc}$ (estimation $@{tc.auc_train}$); Brier $= @{tc.brier}$ against $@{tc.brier_ref}$ (skill @{tc.bss}\\%)', 'AUC de test $= @{tc.auc}$ (estimare $@{tc.auc_train}$); Brier $= @{tc.brier}$ față de $@{tc.brier_ref}$ (cîștig @{tc.bss}\\%)'),
    T('Hosmer--Lemeshow $= @{tc.hl}$, p @{tc.hlp}: rejected, although the points lie close to the diagonal', 'Hosmer--Lemeshow $= @{tc.hl}$, p @{tc.hlp}: respins, deși punctele sînt aproape de diagonală'),
    T('Interpretation: with 9000 clients the test detects small deviations (the safest group: PD @{tc.g0p}\\%, observed @{tc.g0r}\\%); judge the size of the gaps, not only the p-value',
      'Interpretare: cu 9000 de clienți, testul detectează abateri mici (grupul cel mai sigur: PD @{tc.g0p}\\%, observat @{tc.g0r}\\%); judecăm mărimea abaterilor, nu doar p-value-ul')],
    wl='0.44', wr='0.54', h='0.62\\textheight')

D.frame(T('Out-of-Sample and Cross-Validation', 'Validarea în afara eșantionului și validarea încrucișată'), items(
    (T('\\textbf{In-sample} performance is optimistic: the model has adapted to the noise of the training data (overfitting)', 'Performanța \\textbf{în eșantion} este optimistă: modelul s-a adaptat la zgomotul datelor de estimare (supraajustare)'),
     [T('one test sample of 300 credits gives AUC $= @{v.auc}$ with a standard error of $@{v.auc.se}$: one split is noisy', 'un eșantion de test de 300 de credite dă AUC $= @{v.auc}$ cu o eroare standard de $@{v.auc.se}$: o singură împărțire este zgomotoasă')]),
    (T('\\textbf{$k$-fold cross-validation}: split into $k = 5$ folds; estimate on four, test on the fifth; rotate; repeat 20 times', '\\textbf{Validarea încrucișată în $k$ părți}: împărțim în $k = 5$ părți; estimăm pe patru, testăm pe a cincea; rotim; repetăm de 20 de ori'),
     [T('the whole procedure is repeated inside every fold: bins, WoE, IV selection and coefficients', 'întreaga procedură se repetă în fiecare parte: intervale, WoE, selecția după IV și coeficienții'),
      T('report the mean and the spread of the 100 test AUCs, and the gap between training and test AUC', 'raportăm media și dispersia celor 100 de AUC de test, precum și diferența dintre AUC de estimare și de test')]),
    T('\\textbf{Out-of-time} validation: estimate on older applications, test on later ones (next slide)', 'Validarea \\textbf{out-of-time}: estimăm pe cereri mai vechi, testăm pe cereri ulterioare (slide-ul următor)')))

chart(T('Training and Test AUC in Repeated Cross-Validation', 'AUC de estimare și de test în validarea încrucișată repetată'), 'sfm_ch12_cv_auc', 'SFM_ch12_validation_irb', [
    T('20 repetitions of 5 folds; dots: training folds, crosses: test folds; black lines: means', '20 de repetări a cîte 5 părți; puncte: părțile de estimare, cruci: părțile de test; liniile negre: mediile'),
    T('WoE logit: training $@{cv.wl.train}$, test $@{cv.wl.test}$ (sd $@{cv.wl.test.sd}$); LDA: $@{cv.lda.train}$, $@{cv.lda.test}$; small logit: $@{cv.sm.train}$, $@{cv.sm.test}$; checking account: $@{cv.st.test}$',
      'Logit WoE: estimare $@{cv.wl.train}$, test $@{cv.wl.test}$ (sd $@{cv.wl.test.sd}$); LDA: $@{cv.lda.train}$, $@{cv.lda.test}$; logit mic: $@{cv.sm.train}$, $@{cv.sm.test}$; contul curent: $@{cv.st.test}$'),
    T('Interpretation: the scorecard loses @{cv.wl.gap} AUC out of sample, the five-variable logit almost nothing; the single split above was lucky ($@{v.auc}$ against $@{cv.wl.test}$ on average)',
      'Interpretare: scorecard-ul pierde @{cv.wl.gap} din AUC în afara eșantionului, logit-ul cu cinci variabile aproape nimic; împărțirea unică de mai sus a fost norocoasă ($@{v.auc}$ față de $@{cv.wl.test}$ în medie)')],
    h='0.46\\textheight')

D.frame(T('Out-of-Time Validation and Drift', 'Validarea out-of-time și deriva populației'), items(
    (T('A scorecard is used for years on applicants who arrive \\textbf{after} the estimation period', 'Un scorecard este folosit ani de zile pe solicitanți care vin \\textbf{după} perioada de estimare'),
     [T('the economy, the products and the applicants change: the relation between $x$ and default drifts', 'economia, produsele și solicitanții se schimbă: relația dintre $x$ și nerambursare se modifică în timp'),
      T('out-of-time test: estimate on, say, 2019--2021, test on 2022--2023; compare AUC and calibration across years', 'testul out-of-time: estimăm, de exemplu, pe 2019--2021, testăm pe 2022--2023; comparăm AUC și calibrarea de la un an la altul')]),
    (T('Monitoring: the distribution of the scores and of each variable is compared with the training period', 'Monitorizarea: distribuția scorurilor și a fiecărei variabile se compară cu perioada de estimare'),
     [T('a shift with stable ranking calls for recalibration (the constant); a loss of ranking calls for a new model', 'o deplasare cu ordonare stabilă cere recalibrare (termenul liber); o pierdere a ordonării cere un model nou')]),
    T('Data limitation: neither data set of this chapter has application dates, so an out-of-time test is not possible here; random splits are the most we can do',
      'Limitarea datelor: niciunul dintre seturile de date ale capitolului nu conține data cererilor, deci un test out-of-time nu este posibil aici; putem folosi doar împărțiri aleatoare'),
    T('Further reading, an out-of-time design for crypto ``zombie\'\' assets: \\refPeleZ', 'Lectură suplimentară despre un design out-of-time, pentru activele cripto „zombie”: \\refPeleZ')))

D.frame(T('Case Study: Benchmarking Credit Scoring Methods', 'Studiu de caz: compararea metodelor de scoring'), items(
    (T('\\refLessmann: 41 classifiers, six accuracy measures, eight retail credit data sets (among them the German credit data of UCI)',
       '\\refLessmann: 41 de clasificatori, șase măsuri de acuratețe, opt seturi de date de retail (printre ele datele germane de credit din UCI)'),
     [T('an update of the benchmark of Baesens et al.\\ (2003), ten years later', 'o actualizare, după zece ani, a comparației lui Baesens et al.\\ (2003)')]),
    (T('Findings', 'Rezultatele'),
     [T('several classifiers predict significantly better than logistic regression, the industry standard; heterogeneous ensembles perform best', 'mai mulți clasificatori prezic semnificativ mai bine decît regresia logistică, standardul industriei; ansamblurile eterogene sînt cele mai bune'),
      T('more accurate scorecards can bring sizeable financial returns', 'scorecard-urile mai precise pot aduce cîștiguri financiare importante'),
      T('the common accuracy measures give similar signals; random forests are recommended as the benchmark for any new method', 'măsurile uzuale de acuratețe dau semnale asemănătoare; random forests sînt recomandate ca reper pentru orice metodă nouă')]),
    T('Beating the logit alone is no longer evidence of progress; on our 1000 credits, logit and LDA differ by less than one standard error of AUC; Chapter 13 adds trees and neural networks',
      'A învinge doar logit-ul nu mai este o dovadă de progres; pe cele 1000 de credite, logit și LDA diferă cu mai puțin de o eroare standard a AUC; Capitolul 13 adaugă arbori și rețele neuronale')))

D.recap(('Validation', 'validarea'), [
    T('Discrimination: ROC, AUC, Gini $=$ AR, KS; calibration: reliability diagram, Hosmer--Lemeshow, Brier score', 'Discriminarea: ROC, AUC, Gini $=$ AR, KS; calibrarea: diagrama de calibrare, Hosmer--Lemeshow, scorul Brier'),
    T('Choose the cut-off from the costs, not from the accuracy', 'Alegem pragul din costuri, nu din acuratețe'),
    T('Test out of sample, repeat with cross-validation, and test out of time whenever the data have dates', 'Testăm în afara eșantionului, repetăm cu validarea încrucișată și testăm out-of-time ori de cîte ori setul de date conține data cererilor')])

# =============================================================================
# 8. REJECT INFERENCE ȘI FAIRNESS
# =============================================================================
D.section('Reject Inference and Fairness', 'Reject inference și echitatea')

D.frame(T('Reject Inference', 'Reject inference'), items(
    (T('The outcome is known only for \\textbf{accepted} applicants; the model is used on \\textbf{all} applicants (``through the door\'\')',
       'Rezultatul este cunoscut doar pentru solicitanții \\textbf{acceptați}; modelul se folosește pe \\textbf{toți} solicitanții („through the door”)'),
     [T('a sample selected by the old score: the riskiest applicants are missing, and so are their outcomes', 'un eșantion selectat de scorul vechi: lipsesc solicitanții cei mai riscanți și rezultatele lor'),
      T('the South German Credit borrowers had all passed the bank\'s checks \\refGroemping', 'debitorii din South German Credit trecuseră toți de verificările băncii \\refGroemping')]),
    (T('\\textbf{Reject inference}: methods that try to correct for the missing outcomes \\refBCT; \\refTCE', '\\textbf{Reject inference}: metode care încearcă să corecteze lipsa rezultatelor \\refBCT; \\refTCE'),
     [T('reweighting: give accepted applicants who resemble the rejected ones a larger weight', 'reponderarea: dăm o pondere mai mare acceptaților care seamănă cu respinșii'),
      T('augmentation and parcelling: impute outcomes for the rejected from a model', 'augmentarea și parcelarea: atribuim rezultate respinșilor pe baza unui model'),
      T('the most reliable source: a small random sample of applicants accepted regardless of the score', 'sursa cea mai fiabilă: un mic eșantion aleator de solicitanți acceptați indiferent de scor')])))

D.frame(T('A Reject-Inference Experiment on the Taiwan Data', 'Un experiment de reject inference pe datele din Taiwan'), items(
    (T('An ``old score\'\' (late payment in September and the limit) accepts the best 70\\% of the training clients (@{ri.nacc}); their default rate is @{ri.bad_acc}\\%, that of the rejected @{ri.bad_rej}\\%',
       'Un „scor vechi” (întîrzierea din septembrie și limita) acceptă cei mai buni 70\\% dintre clienții de estimare (@{ri.nacc}); rata lor de nerambursare este @{ri.bad_acc}\\%, a respinșilor @{ri.bad_rej}\\%'),
     [T('among the accepted nobody was late in September: the strongest variable cannot be estimated and drops out', 'printre acceptați nimeni nu a întîrziat în septembrie: variabila cea mai puternică nu poate fi estimată și iese din model')]),
    (T('Test sample (all applicants): AUC of the model estimated on the accepted only $@{ri.auc_acc_model_on_all}$, on all applicants $@{ri.auc_all_model_on_all}$, old score $@{ri.auc_old_on_all}$',
       'Eșantionul de test (toți solicitanții): AUC al modelului estimat doar pe acceptați $@{ri.auc_acc_model_on_all}$, al celui estimat pe toți $@{ri.auc_all_model_on_all}$, al scorului vechi $@{ri.auc_old_on_all}$'),
     [T('rejected test clients: observed default rate @{ri.rate_rejected}\\%; mean PD of the accepted-only model @{ri.mean_pd_rej_acc_model}\\%, of the full model @{ri.mean_pd_rej_all_model}\\%',
        'clienții de test respinși: rata observată @{ri.rate_rejected}\\%; PD medie a modelului estimat pe acceptați @{ri.mean_pd_rej_acc_model}\\%, a modelului complet @{ri.mean_pd_rej_all_model}\\%')]),
    T('Interpretation: trained on the accepted only, the new model ranks worse than the old score and underestimates the risk of the rejected by about @{ri.under}\\%',
      'Interpretare: estimat doar pe acceptați, noul model ordonează mai slab decît scorul vechi și subestimează riscul respinșilor cu aproximativ @{ri.under}\\%')) + ql('SFM_ch12_reject_fairness'), 'footnotesize')

D.frame(T('Fairness in Credit Scoring', 'Echitatea (fairness) în scoringul de credit'), cols(items(
    (T('US law forbids discrimination in credit by sex, marital status, race, religion, national origin or age \\refECOA',
       'Legea americană interzice discriminarea în creditare după sex, stare civilă, rasă, religie, origine națională sau vîrstă \\refECOA'),
     [T('EU: credit scoring of persons is a high-risk use of AI (documentation, human oversight) \\refAIAct', 'UE: scoringul de credit al persoanelor este o utilizare cu risc ridicat a AI (documentare, supraveghere umană) \\refAIAct')]),
    (T('Leaving the protected variable out is not enough: other variables can stand in for it', 'Excluderea variabilei protejate nu este suficientă: alte variabile o pot înlocui'),
     [T('\\textbf{demographic parity}: equal approval rates across groups', '\\textbf{paritatea demografică}: rate egale de aprobare între grupuri'),
      T('\\textbf{equal opportunity}: equal approval rates of the good payers \\refHPS', '\\textbf{egalitatea de șanse}: rate egale de aprobare pentru bun-platnici \\refHPS'),
      T('\\textbf{calibration within groups}: the same PD means the same risk in every group', '\\textbf{calibrarea în fiecare grup}: aceeași PD înseamnă același risc în fiecare grup')]),
    T('When default rates differ across groups, these criteria cannot all hold at once: the choice is a matter of policy for banks and regulators', 'Cînd ratele de nerambursare diferă între grupuri, aceste criterii nu pot fi toate îndeplinite simultan: alegerea ține de politica băncii și a autorităților')),
    ph('ecoa', T('H.R. 10162 (1973), the bill that introduced the Equal Credit Opportunity Act', 'H.R. 10162 (1973), proiectul de lege care a introdus Equal Credit Opportunity Act'), h='0.46\\textheight'),
    wl='0.62', wr='0.35'), 'footnotesize')

chart(T('Approval and Default by Group: Taiwan', 'Aprobarea și nerambursarea pe grupuri: Taiwan'), 'sfm_ch12_fairness', 'SFM_ch12_reject_fairness', [
    T('Test sample; sex and age are not inputs of the model; cut-off: approve the 75\\% with the lowest PD', 'Eșantionul de test; sexul și vîrsta nu sînt variabile ale modelului; pragul: aprobăm cei 75\\% cu PD cea mai mică'),
    T('Women: default rate @{fr.women.rate}\\%, approved @{fr.women.approve}\\%; men: @{fr.men.rate}\\%, @{fr.men.approve}\\%; good payers approved: women @{fr.women.tpr_good}\\%, men @{fr.men.tpr_good}\\%',
      'Femei: rata de nerambursare @{fr.women.rate}\\%, aprobate @{fr.women.approve}\\%; bărbați: @{fr.men.rate}\\%, @{fr.men.approve}\\%; bun-platnici aprobați: femei @{fr.women.tpr_good}\\%, bărbați @{fr.men.tpr_good}\\%'),
    T('Interpretation: the model is calibrated in every group (mean PD close to the observed rate), so groups with more defaults (men, the youngest and the oldest) are approved less often',
      'Interpretare: modelul este calibrat în fiecare grup (PD medie apropiată de rata observată), deci grupurile cu mai multe nerambursări (bărbații, cei mai tineri și cei mai în vîrstă) sînt aprobate mai rar')],
    h='0.46\\textheight')

D.frame(T('Case Study: Machine Learning and Unequal Credit', 'Studiu de caz: învățarea automată și creditul inegal'), items(
    (T('\\refFuster: US mortgages; default predicted with a traditional logit and with machine-learning models', '\\refFuster: credite ipotecare din SUA; nerambursarea prezisă cu un logit tradițional și cu modele de învățare automată'),
     [T('question: who gains and who loses when the lender switches to the more accurate model?', 'întrebarea: cine cîștigă și cine pierde cînd creditorul trece la modelul mai precis?')]),
    (T('Findings', 'Rezultatele'),
     [T('the flexible models predict default better', 'modelele flexibile prezic mai bine nerambursarea'),
      T('Black and Hispanic borrowers are less likely to gain from the new technology', 'debitorii de culoare și cei hispanici au mai puține șanse să cîștige din noua tehnologie'),
      T('in an equilibrium model, the disparity in interest rates between and within groups increases, mainly because of the greater flexibility', 'într-un model de echilibru, disparitatea dobînzilor între grupuri și în interiorul lor crește, mai ales din cauza flexibilității mai mari')]),
    T('Lesson: accuracy and fairness must be measured together; a better AUC is not a sufficient argument', 'Lecția: acuratețea și echitatea trebuie măsurate împreună; un AUC mai bun nu este un argument suficient')))

D.recap(('Reject Inference and Fairness', 'reject inference și echitatea'), [
    T('Outcomes exist only for accepted applicants: a model trained on them extrapolates to the rejected', 'Rezultatele există doar pentru solicitanții acceptați: un model estimat pe ei extrapolează la cei respinși'),
    T('Fairness criteria (parity, equal opportunity, calibration) conflict when base rates differ', 'Criteriile de echitate (paritate, egalitate de șanse, calibrare) intră în conflict cînd ratele de bază diferă'),
    T('Removing a protected variable does not remove its influence', 'Eliminarea unei variabile protejate nu îi elimină influența')])

# =============================================================================
# 9. ALTMAN ȘI MERTON
# =============================================================================
D.section('Corporate Default: Altman and Merton', 'Nerambursarea companiilor: Altman și Merton')

D.frame(T('The Altman Z-Score (1968)', 'Scorul Z al lui Altman (1968)'), cols(items(
    (T('\\refAltman: 66 US manufacturing firms, 33 that filed for bankruptcy in 1946--1965 and 33 matched survivors', '\\refAltman: 66 de companii industriale americane, 33 care au intrat în faliment în 1946--1965 și 33 de companii pereche care au supraviețuit'),
     [T('Fisher\'s discriminant analysis on five accounting ratios', 'analiza discriminantă a lui Fisher pe cinci indicatori contabili')]),
    (T('$Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$', '$Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$'),
     [T('$X_1$ working capital / total assets (TA); $X_2$ retained earnings / TA; $X_3$ EBIT (earnings before interest and taxes) / TA', '$X_1$ capitalul de lucru / activele totale (TA); $X_2$ profitul reinvestit / TA; $X_3$ EBIT (profitul înainte de dobînzi și impozite) / TA'),
      T('$X_4$ market value of equity / book value of debt; $X_5$ sales / TA', '$X_4$ valoarea de piață a capitalurilor proprii / valoarea contabilă a datoriilor; $X_5$ cifra de afaceri / TA'),
      T('zones: $Z < 1.81$ distress, $1.81$--$2.99$ grey, $Z > 2.99$ safe', 'zonele: $Z < 1{,}81$ dificultate, $1{,}81$--$2{,}99$ zona gri, $Z > 2{,}99$ sigur')]),
    T('About 95\\% of the sample correctly classified one year before bankruptcy; the accuracy falls quickly for longer horizons',
      'Aproximativ 95\\% din eșantion clasificat corect cu un an înainte de faliment; acuratețea scade repede pentru orizonturi mai lungi')),
    ph('stern', T('NYU Stern School of Business, Altman\'s institution', 'NYU Stern School of Business, instituția lui Altman'), h='0.36\\textheight'),
    wl='0.60', wr='0.37'), 'footnotesize')

D.frame(T('Worked Example: a Z-Score', 'Exemplu rezolvat: un scor Z'), items(
    (T('A hypothetical firm: $X_1 = 0.15$, $X_2 = 0.20$, $X_3 = 0.08$, $X_4 = 0.90$, $X_5 = 1.10$', 'O companie ipotetică: $X_1 = 0.15$, $X_2 = 0.20$, $X_3 = 0.08$, $X_4 = 0.90$, $X_5 = 1.10$'),
     [T('$Z = 1.2 \\times 0.15 + 1.4 \\times 0.20 + 3.3 \\times 0.08 + 0.6 \\times 0.90 + 1.0 \\times 1.10$', '$Z = 1.2 \\times 0.15 + 1.4 \\times 0.20 + 3.3 \\times 0.08 + 0.6 \\times 0.90 + 1.0 \\times 1.10$'),
      T('$Z = @{alt.t1} + @{alt.t2} + @{alt.t3} + @{alt.t4} + @{alt.t5} = @{alt.z}$: the grey zone', '$Z = @{alt.t1} + @{alt.t2} + @{alt.t3} + @{alt.t4} + @{alt.t5} = @{alt.z}$: zona gri')]),
    (T('Which ratio would move the firm to the safe zone?', 'Ce indicator ar muta compania în zona sigură?'),
     [T('an EBIT/TA of 0.27 instead of 0.08 adds $3.3 \\times 0.19 \\approx 0.63$: $Z \\approx 2.99$', 'un EBIT/TA de 0,27 în loc de 0,08 adaugă $3.3 \\times 0.19 \\approx 0.63$: $Z \\approx 2{,}99$')]),
    (T('Limits', 'Limite'),
     [T('the weights come from 66 US manufacturers of 1946--1965; other sectors and countries need re-estimation', 'ponderile provin din 66 de companii industriale americane din 1946--1965; alte sectoare și țări cer reestimare'),
      T('Z is a discriminant score, not a PD; later models give PDs: logit \\refOhlson, hazard models \\refShumway', 'Z este un scor discriminant, nu o PD; modelele ulterioare dau PD: logit \\refOhlson, modele de hazard \\refShumway')])))

D.frame(T('Merton\'s Distance to Default', 'Distanța pînă la nerambursare a lui Merton'), cols(items(
    (T('\\refMerton: the value $V_t$ of the firm\'s assets follows a GBM (geometric Brownian motion, Chapter 4) with drift $\\mu$ and volatility $\\sigma_V$',
       '\\refMerton: valoarea $V_t$ a activelor companiei urmează o GBM (mișcarea browniană geometrică, Capitolul 4) cu rata de creștere (drift) $\\mu$ și volatilitatea $\\sigma_V$'),
     [T('debt with face value $D$ due at $T$; default if $V_T < D$; equity is a call option on the assets', 'datorie cu valoarea nominală $D$ scadentă la $T$; nerambursare dacă $V_T < D$; capitalurile proprii sînt o opțiune call pe active')]),
    (T('$\\ln V_T \\sim N\\big(\\ln V_0 + (\\mu - \\sigma_V^2/2)T,\\ \\sigma_V^2T\\big)$, so', '$\\ln V_T \\sim N\\big(\\ln V_0 + (\\mu - \\sigma_V^2/2)T,\\ \\sigma_V^2T\\big)$, deci'),
     [T('$\\mathrm{DD} = \\dfrac{\\ln(V_0/D) + (\\mu - \\sigma_V^2/2)T}{\\sigma_V\\sqrt{T}}$, \\quad $\\mathrm{PD} = \\Phi(-\\mathrm{DD})$', '$\\mathrm{DD} = \\dfrac{\\ln(V_0/D) + (\\mu - \\sigma_V^2/2)T}{\\sigma_V\\sqrt{T}}$, \\quad $\\mathrm{PD} = \\Phi(-\\mathrm{DD})$'),
      T('DD: the number of standard deviations between the expected asset value and the debt', 'DD: numărul de abateri standard dintre valoarea așteptată a activelor și datorie')]),
    T('$V$ and $\\sigma_V$ are not observed: they are solved from the equity price and the equity volatility (option pricing)', '$V$ și $\\sigma_V$ nu sînt observabile: se obțin din prețul și volatilitatea acțiunilor (evaluarea opțiunilor)')),
    ph('merton', T('Robert C. Merton, 2010', 'Robert C. Merton, 2010'), h='0.42\\textheight'), wl='0.66', wr='0.31'), 'footnotesize')

D.frame(T('Worked Example: Distance to Default', 'Exemplu rezolvat: distanța pînă la nerambursare'), items(
    (T('$V_0 = 100$, $D = 70$, $\\sigma_V = 25\\%$, $\\mu = 5\\%$, $T = 1$ year', '$V_0 = 100$, $D = 70$, $\\sigma_V = 25\\%$, $\\mu = 5\\%$, $T = 1$ an'),
     [T('$\\ln(100/70) = @{mt.ln}$; $\\mu - \\sigma_V^2/2 = 0.05 - 0.03125 = @{mt.drift}$', '$\\ln(100/70) = @{mt.ln}$; $\\mu - \\sigma_V^2/2 = 0.05 - 0.03125 = @{mt.drift}$'),
      T('$\\mathrm{DD} = @{mt.num}/0.25 = @{mt.dd}$; PD $= \\Phi(-@{mt.dd}) = @{mt.pd}\\%$', '$\\mathrm{DD} = @{mt.num}/0.25 = @{mt.dd}$; PD $= \\Phi(-@{mt.dd}) = @{mt.pd}\\%$')]),
    (T('From the market: equity 40, equity volatility 50\\%, debt 70, risk-free rate 3\\%', 'Din piață: capitaluri proprii 40, volatilitatea acțiunilor 50\\%, datorie 70, rata fără risc 3\\%'),
     [T('solving $E = V\\Phi(d_1) - De^{-rT}\\Phi(d_2)$ and $\\sigma_E E = \\Phi(d_1)\\sigma_V V$: $V = @{me.V}$, $\\sigma_V = @{me.s}\\%$, DD $= @{me.dd}$, PD $= @{me.pd}\\%$',
        'rezolvînd $E = V\\Phi(d_1) - De^{-rT}\\Phi(d_2)$ și $\\sigma_E E = \\Phi(d_1)\\sigma_V V$: $V = @{me.V}$, $\\sigma_V = @{me.s}\\%$, DD $= @{me.dd}$, PD $= @{me.pd}\\%$'),
      T('$E$, $\\sigma_E$: value and volatility of equity; $r$: the risk-free rate; $d_1$, $d_2$: the Black--Scholes terms of Chapter 4, with $V$ in place of the share price', '$E$, $\\sigma_E$: valoarea și volatilitatea capitalurilor proprii; $r$: rata fără risc; $d_1$, $d_2$: termenii Black--Scholes din Capitolul 4, cu $V$ în locul prețului acțiunii')]),
    T('Interpretation: a market-based PD reacts every day to prices; an accounting score changes once a year', 'Interpretare: o PD din piață reacționează zilnic la prețuri; un scor contabil se schimbă o dată pe an')) + ql('SFM_ch12_altman_merton'), 'footnotesize')

chart(T('Merton: Asset Paths and PD', 'Merton: traiectoriile activelor și PD'), 'sfm_ch12_merton', 'SFM_ch12_altman_merton', [
    T('Left: 40 simulated asset paths ($V_0 = 100$, $\\mu = 5\\%$, $\\sigma_V = 25\\%$); red: the paths that end below the debt of 70', 'Stînga: 40 de traiectorii simulate ale activelor ($V_0 = 100$, $\\mu = 5\\%$, $\\sigma_V = 25\\%$); roșu: traiectoriile care se termină sub datoria de 70'),
    T('Right: one-year PD against leverage $D/V$ for three asset volatilities', 'Dreapta: PD pe un an în funcție de gradul de îndatorare $D/V$ pentru trei volatilități ale activelor'),
    T('Interpretation: PD is almost zero at low leverage and rises steeply between 70\\% and 90\\%; the higher the volatility, the earlier the rise', 'Interpretare: PD este aproape zero la îndatorare mică și crește abrupt între 70\\% și 90\\%; cu cît volatilitatea este mai mare, cu atît creșterea începe mai devreme')],
    h='0.46\\textheight')

D.frame(T('Case Study: How Good Is the Merton Model?', 'Studiu de caz: cît de bun este modelul Merton?'), items(
    (T('\\refBS: US firms; the Merton DD against a ``naive\'\' DD with the same functional form but without solving the model', '\\refBS: companii americane; DD Merton față de un DD „naiv”, cu aceeași formă funcțională, dar fără rezolvarea modelului'),
     [T('design: hazard models of default and out-of-sample forecasts; comparison with CDS (credit default swap) spreads and bond yields', 'designul: modele de hazard ale nerambursării și prognoze în afara eșantionului; comparație cu spread-urile CDS (credit default swap) și randamentele obligațiunilor')]),
    (T('Findings', 'Rezultatele'),
     [T('the naive DD performs slightly better than the Merton DD out of sample', 'DD naiv funcționează puțin mai bine decît DD Merton în afara eșantionului'),
      T('other variables (e.g.\\ past returns) add information: DD is not a sufficient statistic for PD', 'alte variabile (de exemplu randamentele trecute) aduc informație: DD nu este o statistică suficientă pentru PD'),
      T('its functional form (leverage scaled by volatility) is what makes it useful', 'forma sa funcțională (îndatorarea raportată la volatilitate) este ceea ce îl face util')]),
    T('Lesson: theory gives a good variable, statistics decides how to use it; the same holds for the Z-score', 'Lecția: teoria dă o variabilă bună, statistica decide cum o folosim; la fel pentru scorul Z')))

D.frame(T('Recap: Altman and Merton', 'Recapitulare: Altman și Merton'), items(
    T('Altman: LDA on five accounting ratios; zones 1.81 and 2.99', 'Altman: LDA pe cinci indicatori contabili; pragurile 1,81 și 2,99'),
    T('Merton: equity is a call on the assets; DD $= [\\ln(V/D) + (\\mu - \\sigma^2/2)T]/(\\sigma\\sqrt{T})$, PD $= \\Phi(-\\mathrm{DD})$', 'Merton: capitalurile proprii sînt un call pe active; DD $= [\\ln(V/D) + (\\mu - \\sigma^2/2)T]/(\\sigma\\sqrt{T})$, PD $= \\Phi(-\\mathrm{DD})$'),
    T('Both are best used as inputs of a statistical PD model', 'Ambele sînt folosite cel mai bine ca variabile ale unui model statistic de PD')))

# =============================================================================
# 10. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Do flexible machine-learning scores improve default prediction enough to justify their cost in transparency and fairness?}',
       '\\textbf{Îmbunătățesc scorurile flexibile de învățare automată prognoza nerambursării suficient pentru a justifica pierderea de transparență și de echitate?}'),
     [T('today: WoE logit and LDA reach a test AUC of about @{cv.wl.test}; benchmarks find significant gains for ensembles \\refLessmann', 'azi: logit-ul WoE și LDA ating un AUC de test de aproximativ @{cv.wl.test}; comparațiile găsesc cîștiguri semnificative pentru ansambluri \\refLessmann'),
      T('the gains may be unequal across groups \\refFuster; support vector machines for default: \\refCHM', 'cîștigurile pot fi inegale între grupuri \\refFuster; mașini cu vectori suport pentru nerambursare: \\refCHM')]),
    T('Why it is open: gains depend on the data set, the measure and the population; the outcomes of rejected applicants are never seen',
      'Întrebarea rămîne deschisă: cîștigurile depind de setul de date, de măsură și de populație; rezultatele solicitanților respinși nu se văd niciodată'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of benchmark studies of credit scoring and of studies on fairness in lending, with their data sets', '\\textbf{Literatura}: o listă a comparațiilor de metode de scoring și a studiilor despre echitatea creditării, cu seturile lor de date'),
    T('\\textbf{Code}: a first draft of a cross-validated comparison of the WoE logit with gradient boosting on the Taiwan data', '\\textbf{Cod}: o primă versiune a unei comparații prin validare încrucișată a logit-ului WoE cu gradient boosting pe datele din Taiwan'),
    T('\\textbf{Robustness}: other cut-offs, other measures (Brier, cost), group-wise AUC and calibration', '\\textbf{Robustețe}: alte praguri, alte măsuri (Brier, cost), AUC și calibrare pe grupuri'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that compares a WoE logistic regression with gradient boosting on the Taiwan credit card default data by repeated stratified 5-fold cross-validation, recomputing the WoE bins inside each fold, and reports AUC, Brier score and the approval rates of men and women at a 75 percent approval cut-off.}',
        '\\aiprompt{Write Python code that compares a WoE logistic regression with gradient boosting on the Taiwan credit card default data by repeated stratified 5-fold cross-validation, recomputing the WoE bins inside each fold, and reports AUC, Brier score and the approval rates of men and women at a 75 percent approval cut-off.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The coding of the outcome: default $= 1$; a reversed coding flips every sign and every curve', 'Codificarea rezultatului: nerambursare $= 1$; o codificare inversă schimbă fiecare semn și fiecare curbă'),
    T('The WoE sign convention: ln(goods/bads) here; some sources use ln(bads/goods)', 'Convenția de semn pentru WoE: aici ln(bun/rău); unele surse folosesc ln(rău/bun)'),
    T('No leakage: bins, WoE and variable selection inside each training fold', 'Fără scurgere de informație: intervalele, WoE și selecția variabilelor în fiecare parte de estimare'),
    T('AUC is not accuracy, and Gini $= 2\\,$AUC $- 1$, not AUC $- 0.5$', 'AUC nu este acuratețea, iar Gini $= 2\\,$AUC $- 1$, nu AUC $- 0{,}5$'),
    T('PDs from an oversampled data set need the prior correction before any EL or capital number', 'PD-urile estimate pe un set de date cu rău-platnici suprareprezentați cer corecția a priori înaintea oricărei cifre de EL sau capital'),
    T('References: every cited paper and data set must exist; check the DOI', 'Referințele: fiecare lucrare și set de date citat trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: on the Taiwan credit card data, how much AUC does gradient boosting add over the WoE logit, and what happens to the approval rates by sex and age?',
       '\\textbf{Întrebarea}: pe datele cardurilor din Taiwan, cît adaugă gradient boosting la AUC față de logit-ul WoE și ce se întîmplă cu ratele de aprobare pe sexe și vîrste?'),
     [T('data: \\refTW (CC BY 4.0), saved in data/credit', 'date: \\refTW (CC BY 4.0), salvate în data/credit')]),
    (T('Steps', 'Pași'),
     [T('WoE bins and IV for every variable; a WoE logit; gradient boosting with default settings', 'intervale WoE și IV pentru fiecare variabilă; un logit WoE; gradient boosting cu setările implicite'),
      T('repeated stratified cross-validation: AUC with DeLong intervals, Brier score, calibration', 'validare încrucișată stratificată repetată: AUC cu intervale DeLong, scorul Brier, calibrarea'),
      T('fairness at one cut-off: demographic parity and equal opportunity for sex and age groups', 'echitatea la un prag: paritatea demografică și egalitatea de șanse pentru sexe și grupe de vîrstă')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('EL $=$ PD $\\times$ LGD $\\times$ EAD is priced in; IRB capital covers the unexpected loss', 'EL $=$ PD $\\times$ LGD $\\times$ EAD intră în preț; capitalul IRB acoperă pierderea neașteptată'),
    T('WoE turns every variable into log-odds evidence; IV ranks the variables', 'WoE transformă fiecare variabilă într-o dovadă pe scala logaritmului șansei; IV ordonează variabilele'),
    T('Logit: $e^{\\beta}$ is an odds ratio; LDA (Fisher) gives the same linear log-odds under Normality and ranks equally well here', 'Logit: $e^{\\beta}$ este un raport al șanselor; LDA (Fisher) dă același logaritm liniar al șansei sub ipoteza de normalitate și ordonează la fel de bine aici'),
    T('A scorecard: Score $=$ Offset $+$ Factor $\\ln$(good:bad odds), PDO $= 20$ points per doubling', 'Un scorecard: Scorul $=$ Offset $+$ Factor $\\ln$(șansa bun:rău), PDO $= 20$ de puncte pentru o dublare'),
    T('Validate discrimination (AUC, Gini, KS) and calibration (Brier, reliability diagram) out of sample; choose the cut-off from costs', 'Validăm discriminarea (AUC, Gini, KS) și calibrarea (Brier, diagrama de calibrare) în afara eșantionului; alegem pragul din costuri'),
    T('Rejected applicants and protected groups need explicit checks; Altman and Merton add accounting and market information', 'Solicitanții respinși și grupurile protejate cer verificări explicite; Altman și Merton adaugă informație contabilă și de piață')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['EL & $\\mathrm{PD} \\times \\mathrm{LGD} \\times \\mathrm{EAD}$',
     'WoE, IV & $\\ln\\frac{g_i/G}{b_i/B}$, \\ $\\sum_i (g_i/G - b_i/B)\\,\\mathrm{WoE}_i$',
     'Logit & $\\ln\\frac{p}{1 - p} = \\beta_0 + x^\\top\\beta$, \\ $p = 1/(1 + e^{-\\eta})$',
     'LDA & $w = S_W^{-1}(m_1 - m_0)$',
     T('Prior correction', 'Corecția a priori') + ' & $+\\ln\\frac{\\pi/(1 - \\pi)}{\\pi_s/(1 - \\pi_s)}$ ' + T('to the constant', 'la termenul liber'),
     'Scorecard & $\\text{Offset} + \\frac{\\text{PDO}}{\\ln 2}\\ln\\frac{1 - p}{p}$',
     'AUC, Gini & $P(s_{\\text{bad}} > s_{\\text{good}})$, \\ $2\\,\\mathrm{AUC} - 1$',
     'KS, Brier & $\\max_s|F_1(s) - F_0(s)|$, \\ $\\frac1n\\sum(p_i - y_i)^2$',
     T('Bayes cut-off', 'Pragul Bayes') + ' & $p > C_{FP}/(C_{FP} + C_{FN})$',
     'Merton & $\\mathrm{DD} = \\frac{\\ln(V/D) + (\\mu - \\sigma^2/2)T}{\\sigma\\sqrt{T}}$, \\ $\\mathrm{PD} = \\Phi(-\\mathrm{DD})$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a model has an accuracy of 70\\% on the South German Credit test sample. Is it good?', '\\textbf{Întrebare}: un model are o acuratețe de 70\\% pe eșantionul de test South German Credit. Este bun?'),
     [T('\\textbf{Answer}: not necessarily: ``everybody is good\'\' already reaches 70\\%; look at AUC, TPR, FPR and the cost', '\\textbf{Răspuns}: nu neapărat: regula „toți sînt bun-platnici” atinge deja 70\\%; ne uităm la AUC, TPR, FPR și cost')]),
    (T('\\textbf{Question}: the coefficient of a dummy in a logit is 0.69. What does it mean?', '\\textbf{Întrebare}: coeficientul unei variabile indicator într-un logit este 0,69. Ce înseamnă?'),
     [T('\\textbf{Answer}: the odds of default are multiplied by $e^{0.69} \\approx 2$ for the group, relative to the reference group', '\\textbf{Răspuns}: șansa de nerambursare se înmulțește cu $e^{0{,}69} \\approx 2$ pentru grup, față de grupul de referință')]),
    (T('\\textbf{Question}: does correcting the PDs for oversampling change the AUC?', '\\textbf{Întrebare}: schimbă corecția pentru suprareprezentare valoarea AUC?'),
     [T('\\textbf{Answer}: no: it shifts all log-odds by the same constant, so the ranking is unchanged', '\\textbf{Răspuns}: nu: deplasează toți logaritmii șansei cu aceeași constantă, deci ordonarea nu se schimbă')]),
    T('Next: Chapter 13, machine learning', 'Urmează: Capitolul 13, învățarea automată')))

D.references(BIB, per=17)

if __name__ == '__main__':
    D.write(V)
