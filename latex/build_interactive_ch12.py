"""Build the interactive pages of Chapter 12: interactive/ch12.html (EN) and interactive/ch12_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch12.js, inputs in interactive/ch12_data.js (Quantlets/Ch_12/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch12.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 12
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_12'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('cpd', 'sfm_ch12_conditional_pd', 'def'), ('irb', 'sfm_ch12_irb', 'def'), ('status', 'sfm_ch12_default_by_status', 'data'),
          ('woe-dur', 'sfm_ch12_woe_duration', 'woe'), ('iv', 'sfm_ch12_iv', 'woe'), ('logit-curve', 'sfm_ch12_logistic_curve', 'logit'),
          ('odds', 'sfm_ch12_odds_ratios', 'logit'), ('lda', 'sfm_ch12_lda_2d', 'lda'), ('score', 'sfm_ch12_score_dist', 'card'),
          ('cost', 'sfm_ch12_cost_cutoff', 'val'), ('roc', 'sfm_ch12_roc', 'val'), ('cap', 'sfm_ch12_cap', 'val'), ('ks', 'sfm_ch12_ks', 'val'),
          ('calib', 'sfm_ch12_calibration', 'val'), ('cv', 'sfm_ch12_cv_auc', 'val'), ('fair', 'sfm_ch12_fairness', 'rej'), ('merton', 'sfm_ch12_merton', 'corp')]
# table and worked-example blocks: (anchor, frame number in the deck (RO and EN have the same frames), section key, chart block that carries it on the web page)
XBLOCKS = [('irb-we', 12, 'def', 'irb'), ('woe-tab', 20, 'woe', 'woe-dur'), ('odds-we', 24, 'logit', 'logit-curve'), ('small-tab', 28, 'logit', 'odds'),
           ('pd-we', 30, 'logit', 'odds'), ('woe-logit-tab', 31, 'logit', 'odds'), ('prior-we', 32, 'logit', 'odds'), ('fisher-we', 35, 'lda', 'lda'),
           ('lda-tab', 37, 'lda', 'lda'), ('scale-we', 39, 'card', 'score'), ('cm-tab', 42, 'val', 'cost'), ('auc-we', 44, 'val', 'roc'), ('ks-we', 46, 'val', 'cap'),
           ('brier-we', 48, 'val', 'calib'), ('reject-tab', 55, 'rej', 'fair'), ('z-we', 60, 'corp', 'merton'), ('dd-we', 62, 'corp', 'merton')]
ORDER = ['cpd', 'irb-we', 'irb', 'status', 'woe-tab', 'woe-dur', 'iv', 'odds-we', 'logit-curve', 'small-tab', 'odds', 'pd-we', 'woe-logit-tab', 'prior-we',
         'fisher-we', 'lda', 'lda-tab', 'scale-we', 'score', 'cm-tab', 'cost', 'auc-we', 'roc', 'ks-we', 'cap', 'ks', 'brier-we', 'calib', 'cv', 'reject-tab', 'fair',
         'z-we', 'dd-we', 'merton']
TITLES = {'irb-we': 'Exemplu rezolvat: capitalul IRB al unui credit de retail', 'woe-tab': 'Exemplu rezolvat: WoE și IV pentru contul curent',
          'odds-we': 'Șansa (odds) și logaritmul șansei', 'small-tab': 'Un model logit mic pe eșantionul complet', 'pd-we': 'Exemplu rezolvat: PD a unui solicitant',
          'woe-logit-tab': 'Scorecard-ul logit pe variabile WoE', 'prior-we': 'Corecția pentru suprareprezentarea rău-platnicilor',
          'fisher-we': 'Exemplu rezolvat: direcția Fisher cu două variabile', 'lda-tab': 'Compararea modelelor logit și LDA', 'scale-we': 'Exemplu rezolvat: scoruri și puncte',
          'cm-tab': 'Două praguri pe eșantionul de test', 'auc-we': 'Curba ROC și AUC', 'ks-we': 'Curba CAP, raportul de acuratețe și KS',
          'brier-we': 'Calibrarea și scorul Brier', 'reject-tab': 'Un experiment de reject inference pe datele din Taiwan', 'z-we': 'Exemplu rezolvat: un scor Z',
          'dd-we': 'Exemplu rezolvat: distanța pînă la nerambursare'}

PFX = {'cpd': 'cp', 'irb': 'ir', 'status': 'ds', 'woe-dur': 'wd', 'iv': 'iv', 'logit-curve': 'lc', 'odds': 'or', 'lda': 'ld', 'score': 'sd', 'cost': 'co', 'roc': 'rc',
       'cap': 'ca', 'ks': 'ks', 'calib': 'cb', 'cv': 'cv', 'fair': 'fa', 'merton': 'me', 'irb-we': 'iw', 'woe-tab': 'wt', 'odds-we': 'ow', 'small-tab': 'sm', 'pd-we': 'pw',
       'woe-logit-tab': 'wl', 'prior-we': 'pr', 'fisher-we': 'fw', 'lda-tab': 'lt', 'scale-we': 'sw', 'cm-tab': 'cm', 'auc-we': 'aw', 'ks-we': 'kw', 'brier-we': 'bw',
       'reject-tab': 'rj', 'z-we': 'zw', 'dd-we': 'dd'}

SECTIONS = {
 'ro': dict(def_='Scorul de credit: definiție', data='Datele de nerambursare', woe='Weight of Evidence și Information Value', logit='Regresia logistică',
            lda='Analiza discriminantă liniară', card='Construirea scorecard-ului', val='Validarea', rej='Reject inference și echitatea',
            corp='Nerambursarea companiilor: Altman și Merton'),
 'en': dict(def_='The Credit Score: Definition', data='Default Data', woe='Weight of Evidence and Information Value', logit='Logistic Regression',
            lda='Linear Discriminant Analysis', card='Building the Scorecard', val='Validation', rej='Reject Inference and Fairness',
            corp='Corporate Default: Altman and Merton'),
}
for _l in SECTIONS.values():
    _l['def'] = _l.pop('def_')

L = {
 'ro': dict(file='ch12_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 12 · Modele de scoring', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: PD (probability of default, probabilitatea de nerambursare), LGD (loss given default, pierderea '
        'în caz de nerambursare), EAD (exposure at default, expunerea la nerambursare), EL (expected loss, pierderea așteptată), IRB (internal ratings-based, abordarea bazată pe '
        'ratinguri interne), RWA (risk-weighted assets, active ponderate la risc), WoE (weight of evidence), IV (information value), LDA (linear discriminant analysis, analiza '
        'discriminantă liniară), ROC (receiver operating characteristic), AUC (area under the curve, aria de sub curba ROC), CAP (cumulative accuracy profile), AR (accuracy ratio, '
        'raportul de acuratețe), KS (statistica Kolmogorov–Smirnov), HL (testul Hosmer–Lemeshow), TP, FN, FP, TN (true positive, false negative, false positive, true negative), '
        'TPR (true positive rate), FPR (false positive rate), PDO (points to double the odds, punctele care dublează șansa), LR (likelihood ratio, raportul de verosimilitate), '
        'CI (confidence interval, interval de încredere), se (eroarea standard), DM (mărci germane), EBIT (profitul înainte de dobînzi și impozite), TA (activele totale), '
        'DD (distance to default, distanța pînă la nerambursare), GBM (geometric Brownian motion, mișcarea browniană geometrică).',
   src='South German Credit (Grömping, 2019; UCI, doi:10.24432/C5QG88): 1000 de credite de consum din 1973–1975, 300 neperformante; împărțirea stratificată 700/300 (sămînța 2026). '
       'Cardurile de credit din Taiwan (Yeh și Lien, 2009; UCI, doi:10.24432/C55S3H): 30 000 de clienți, eșantion de test de 9000. Ambele seturi sînt salvate în data/credit. '
       'Browserul calculează din înregistrările South German Credit tabelele WoE, IV, modelele logit și LDA, scorecard-ul, matricile de confuzie, ROC, AUC, DeLong, Gini, CAP, KS, '
       'Brier și Hosmer–Lemeshow; din PD, rezultatul, sexul și vîrsta celor 9000 de clienți de test din Taiwan, calibrarea și echitatea. AUC din cele 100 de părți ale validării '
       'încrucișate, traiectoriile Merton (sămînța 2026) și experimentul de reject inference vin din codul din curs. Alte variabile, alte eșantioane, praguri, costuri, niveluri de '
       'încredere, ponderi de aprobare și acceptare (50, 60, 80 și 90%) sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch12.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 12 · Scoring Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: PD (probability of default), LGD (loss given default), EAD (exposure at default), '
        'EL (expected loss), IRB (internal ratings-based approach), RWA (risk-weighted assets), WoE (weight of evidence), IV (information value), LDA (linear discriminant analysis), '
        'ROC (receiver operating characteristic), AUC (area under the ROC curve), CAP (cumulative accuracy profile), AR (accuracy ratio), KS (Kolmogorov–Smirnov statistic), '
        'HL (Hosmer–Lemeshow test), TP, FN, FP, TN (true positive, false negative, false positive, true negative), TPR (true positive rate), FPR (false positive rate), '
        'PDO (points to double the odds), LR (likelihood ratio), CI (confidence interval), se (standard error), DM (Deutsche Mark), EBIT (earnings before interest and taxes), '
        'TA (total assets), DD (distance to default), GBM (geometric Brownian motion).',
   src='South German Credit (Grömping, 2019; UCI, doi:10.24432/C5QG88): 1,000 consumer credits of 1973–1975, 300 bad; stratified 700/300 split (seed 2026). '
       'Taiwan credit card data (Yeh and Lien, 2009; UCI, doi:10.24432/C55S3H): 30,000 clients, test sample of 9,000. Both data sets are saved in data/credit. '
       'From the South German Credit records the browser computes the WoE tables, IV, the logit and LDA models, the scorecard, the confusion matrices, ROC, AUC, DeLong, Gini, CAP, KS, '
       'Brier and Hosmer–Lemeshow; from the PD, outcome, sex and age of the 9,000 Taiwan test clients, the calibration and the fairness figures. The AUC of the 100 cross-validation '
       'folds, the Merton paths (seed 2026) and the reject-inference experiment come from the code of the lecture. Other variables, samples, cut-offs, costs, confidence levels, '
       'approval and acceptance shares (50, 60, 80 and 90%) are extra runs of the same code.'),
}

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child,.xt td.l,.xt th.l{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt .hi{font-weight:700;color:#2E7D32}.xt .bad{color:#CD0000}
.xh{font-weight:600;font-size:.92rem;color:#1F2A44;margin:.9rem 0 .1rem}
.we{border:1px solid #C5D2E8;border-left:4px solid #1A3A6E;border-radius:6px;padding:.55rem .8rem;margin:.8rem 0 .4rem;background:#F6F8FC}
.we-h{font-weight:600;margin:0 0 .2rem;color:#1A3A6E}
.we-f{overflow-x:auto;overflow-y:hidden;font-size:.95rem}
.we ol{margin:.3rem 0 0 1.3rem;padding:0;font-size:.92rem}.we li{margin:.18rem 0;overflow-wrap:anywhere}
.tw.veiled-chart,.we.veiled-chart{filter:blur(10px);opacity:.35;user-select:none}
.xhold{display:none}
.wctl{display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:flex-end;margin:.3rem 0 .2rem;max-width:420px}
</style>"""


def tw(i):
    return f'<div class="tw" id="{i}"></div>'


def xh(t):
    return f'<p class="xh">{t}</p>'


def we(i, title, tex, ctl=''):
    c = f'<div class="wctl">{ctl}</div>' if ctl else ''
    return f'<div class="we" id="{i}-box"><p class="we-h">{title}</p><div class="we-f">\\[{tex}\\]</div>{c}<ol id="{i}"></ol></div>'


def widgets_all(lang):
    ro = lang == 'ro'
    lec = ' (curs)' if ro else ' (lecture)'
    c = (lambda s: s.replace('.', '{,}')) if ro else (lambda s: s)
    n = (lambda s: s.replace('.', ',')) if ro else (lambda s: s)
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    T = lambda r, e: r if ro else e
    VAR4 = lambda d: [(k, t + (lec if k == d else '')) for k, t in (('status', T('contul curent', 'checking account')), ('credit_history', T('istoricul de credit', 'credit history')),
                                                                     ('savings', T('economiile', 'savings')), ('employment_duration', T('vechimea la angajator', 'employment duration')))]
    SMP = lambda: [('full', T('complet (1000 de credite)', 'full (1,000 credits)') + lec), ('train', T('de estimare (700 de credite)', 'training (700 credits)'))]
    MOD = [('logit', T('logit WoE', 'WoE logit') + lec), ('lda', T('LDA Fisher', 'Fisher LDA')), ('single', T('doar contul curent', 'checking account only'))]
    GR = lambda i: select(i, T('Numărul de grupuri', 'Number of groups'), [('10', '10' + lec), ('5', '5'), ('20', '20')], '10')
    return {
     'cpd': (slider('cp-r', T('Ponderea factorului comun ρ', 'Weight of the common factor ρ'), 0.05, 0.5, 0.05, 0.2) + play('cp'), CSS + charts('cp-chart')),
     'irb': (select('ir-q', T('Nivelul de încredere', 'Confidence level'), [('0.999', n('99.9%') + lec), ('0.99', '99%'), ('0.995', n('99.5%')), ('0.9999', n('99.99%'))], '0.999')
             + slider('ir-l', 'LGD', 0.1, 1, 0.05, 0.45) + play('ir'), charts('ir-chart')),
     'status': (select('ds-v', T('Variabila', 'Variable'), VAR4('status'), 'status') + play('ds'), charts('ds-chart')),
     'woe-dur': (select('wd-v', T('Variabila', 'Variable'), [('duration', T('durata (luni)', 'duration (months)') + lec), ('amount', T('suma (DM)', 'amount (DM)')), ('age', T('vîrsta (ani)', 'age (years)'))], 'duration')
                 + select('wd-s', T('Eșantionul', 'Sample'), SMP(), 'full') + play('wd'), charts('wd-chart')),
     'iv': (select('iv-s', T('Eșantionul', 'Sample'), SMP() + [('test', T('de test (300 de credite)', 'test (300 credits)'))], 'full') + play('iv'), charts('iv-chart')),
     'logit-curve': (select('lc-v', T('Variabila explicativă', 'Explanatory variable'), [('duration', T('durata', 'duration') + lec), ('age', T('vîrsta', 'age'))], 'duration')
                     + select('lc-m', T('Numărul minim de credite pe punct', 'Minimum number of credits per dot'), [('5', '5' + lec), ('1', '1'), ('10', '10'), ('20', '20')], '5') + play('lc'), charts('lc-chart')),
     'odds': (select('or-l', T('Nivelul intervalului', 'Level of the interval'), [('0.95', '95%' + lec), ('0.9', '90%'), ('0.99', '99%')], '0.95') + play('or'), charts('or-chart')),
     'lda': (slider('ld-p', T('PD pe frontieră', 'PD on the boundary'), 0.1, 0.6, 0.05, 0.3) + play('ld'), charts('ld-chart')),
     'score': (select('sd-c', T('Pragul PD', 'PD cut-off'), [('0.1666666667', '1/6' + lec), ('0.1', n('0.1')), ('0.3', n('0.3')), ('0.5', n('0.5'))], '0.1666666667')
               + select('sd-b', T('Lățimea clasei (puncte)', 'Bin width (points)'), [('10', '10' + lec), ('5', '5'), ('20', '20')], '10') + play('sd'), charts('sd-chart')),
     'cost': (select('co-f', T('Costul acceptării unui rău-platnic, C_FN (C_FP = 1)', 'Cost of accepting a bad, C_FN (C_FP = 1)'), [('5', '5' + lec), ('2', '2'), ('3', '3'), ('10', '10')], '5') + play('co'), charts('co-chart')),
     'roc': (slider('rc-c', T('Pragul PD al punctului marcat', 'PD cut-off of the marked point'), 0.05, 0.95, 0.05, 0.5)
             + select('rc-s', T('Eșantionul', 'Sample'), [('test', T('de test (300 de credite)', 'test (300 credits)') + lec), ('train', T('de estimare (700 de credite)', 'training (700 credits)'))], 'test') + play('rc'), charts('rc-chart')),
     'cap': (slider('ca-x', T('Ponderea respinșilor', 'Share rejected'), 0.05, 0.95, 0.05, 0.3) + play('ca'), charts('ca-chart')),
     'ks': (slider('ks-s', T('Scorul s (puncte)', 'Score s (points)'), 420, 610, 2, 516) + play('ks'), charts('ks-chart')),
     'calib': (GR('cb-g') + play('cb'), charts('cb-chart') + xh(T('Tabelul de calibrare (cu roșu: termen HL peste 3,84)', 'Calibration table (in red: HL term above 3.84)')) + tw('cb-tab')),
     'cv': (select('cv-r', T('Repetări incluse', 'Repetitions included'), [('20', '20' + lec), ('1', '1'), ('5', '5'), ('10', '10')], '20') + play('cv'), charts('cv-chart')),
     'fair': (slider('fa-a', T('Ponderea aprobaților', 'Share approved'), 0.5, 0.95, 0.05, 0.75) + play('fa'), charts('fa-chart')),
     'merton': (slider('me-m', T('Rata de creștere μ', 'Drift μ'), 0, 0.1, 0.01, 0.05)
                + select('me-s', T('Volatilitatea activelor σ_V', 'Asset volatility σ_V'), [('0.25', '25%' + lec), ('0.15', '15%'), ('0.4', '40%')], '0.25') + play('me'), charts('me-chart')),
     'irb-we': (slider('iw-p', 'PD', 0.005, 0.2, 0.005, 0.02) + play('iw'),
                charts('iw-chart') + we('iw-we', f'{WE}: {T("capitalul IRB, alte expuneri de retail", "IRB capital, other retail exposures")}',
                                        c(r'K = \mathrm{LGD}\Big[\Phi\Big(\frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\,\Phi^{-1}(0.999)}{\sqrt{1 - R}}\Big) - \mathrm{PD}\Big], \quad R = 0.03\,w + 0.16\,(1 - w), \quad w = \frac{1 - e^{-35\,\mathrm{PD}}}{1 - e^{-35}}'),
                                        select('iw-e', T('Expunerea EAD (lei)', 'Exposure EAD (lei)'), [('100000', n('100 000') + lec if ro else '100,000' + lec), ('50000', '50 000' if ro else '50,000'), ('250000', '250 000' if ro else '250,000')], '100000'))),
     'woe-tab': (select('wt-v', T('Variabila', 'Variable'), VAR4('status'), 'status') + play('wt'),
                 charts('wt-chart') + xh(T('Tabelul WoE pe eșantionul complet', 'WoE table on the full sample')) + tw('wt-tab')
                 + we('wt-we', f'{WE}: {T("WoE și termenul IV al unui rînd", "WoE and IV term of one row")}', r'\mathrm{WoE}_i = \ln\frac{g_i/G}{b_i/B}, \qquad \mathrm{IV} = \sum_i \Big(\frac{g_i}{G} - \frac{b_i}{B}\Big)\mathrm{WoE}_i',
                      select('wt-r', T('Rîndul', 'Row'), [('last', T('ultimul', 'last') + lec), ('first', T('primul', 'first')), ('max', T('cel mai mare termen IV', 'largest IV term'))], 'last'))),
     'odds-we': (slider('ow-p', T('Probabilitatea de nerambursare p', 'Probability of default p'), 0.02, 0.98, 0.02, 0.2) + play('ow'),
                 charts('ow-chart') + we('ow-we', f'{WE}: {T("șansa și logaritmul șansei", "odds and log-odds")}', r'\text{odds} = \frac{p}{1 - p}, \qquad \mathrm{logit}(p) = \ln\frac{p}{1 - p}, \qquad \mathrm{logit}(1 - p) = -\mathrm{logit}(p)')),
     'small-tab': (select('sm-d', T('Variabila omisă', 'Variable left out'), [('none', T('niciuna', 'none') + lec), ('duration_years', T('durata', 'duration')), ('age_decades', T('vîrsta', 'age')),
                                                                          ('amount_1000', T('suma', 'amount')), ('acct', T('indicatorii de cont', 'the account indicators'))], 'none') + play('sm'),
                   charts('sm-chart') + xh(T('Coeficienții, eșantionul complet (1000 de credite)', 'Coefficients, full sample (1,000 credits)')) + tw('sm-tab')),
     'pd-we': (slider('pw-d', T('Durata (luni)', 'Duration (months)'), 6, 72, 6, 36) + play('pw'),
               charts('pw-chart') + we('pw-we', f'{WE}: {T("PD a unui solicitant (coeficienți rotunjiți)", "PD of an applicant (rounded coefficients)")}',
                                       c(r'\eta = -2.101 + 0.385\,\tfrac{\text{dur}}{12} - 0.159\,\tfrac{\text{age}}{10} + 0.029\,\tfrac{\text{amount}}{1000} + 1.859\,\mathbb{1}_{\text{no acc.}} + 1.338\,\mathbb{1}_{<0}, \qquad \mathrm{PD} = \frac{1}{1 + e^{-\eta}}'),
                                       select('pw-a', T('Contul curent', 'Checking account'), [('none', T('fără cont curent', 'no checking account') + lec), ('neg', T('sold < 0 DM', 'balance < 0 DM')), ('ok', T('sold nenegativ', 'non-negative balance'))], 'none')
                                       + select('pw-g', T('Vîrsta (ani)', 'Age (years)'), [('30', '30' + lec), ('25', '25'), ('40', '40'), ('60', '60')], '30'))),
     'woe-logit-tab': (select('wl-t', T('Pragul IV pentru selecție', 'IV threshold for selection'), [('0.1', n('0.10') + lec), ('0.02', n('0.02')), ('0.05', n('0.05')), ('0.2', n('0.20')), ('0.3', n('0.30'))], '0.1') + play('wl'),
                       charts('wl-chart') + xh(T('Scorecard-ul logit pe variabile WoE (eșantionul de estimare, 700 de credite)', 'The WoE logit scorecard (training sample, 700 credits)')) + tw('wl-tab')),
     'prior-we': (slider('pr-pi', T('Rata rău-platnicilor în populație π', 'Bad rate of the population π'), 0.01, 0.3, 0.01, 0.05) + play('pr'),
                  charts('pr-chart') + we('pr-we', f'{WE}: {T("corecția a priori", "the prior correction")}', r'\ln \text{odds}_{\text{pop}} = \ln \text{odds}_{\text{sample}} + \ln\frac{\pi/(1 - \pi)}{\pi_s/(1 - \pi_s)}',
                                          select('pr-p', T('PD din eșantion', 'Sample PD'), [('app', T('solicitantul din exemplu, 63,5%', 'the applicant of the example, 63.5%') + lec), ('0.3', '30%'), ('0.5', '50%')], 'app'))),
     'fisher-we': (select('fw-v', T('Variabilele', 'Variables'), [('duration,age', T('durata și vîrsta', 'duration and age') + lec), ('duration,amount', T('durata și suma (1000 DM)', 'duration and amount (1000 DM)')),
                                                                ('age,amount', T('vîrsta și suma (1000 DM)', 'age and amount (1000 DM)'))], 'duration,age') + play('fw'),
                   charts('fw-chart') + we('fw-we', f'{WE}: {T("direcția Fisher", "Fisher’s direction")}', r'J(w) = \frac{[w^\top(m_1 - m_0)]^2}{w^\top S_W w}, \qquad w = S_W^{-1}(m_1 - m_0)')),
     'lda-tab': (select('lt-s', T('Eșantionul', 'Sample'), [('test', T('de test (300 de credite)', 'test (300 credits)') + lec), ('train', T('de estimare (700 de credite)', 'training (700 credits)'))], 'test') + play('lt'),
                 charts('lt-chart') + xh(T('Logit și LDA pe aceleași șase variabile WoE', 'Logit and LDA on the same six WoE variables')) + tw('lt-tab')),
     'scale-we': (select('sw-o', T('PDO (puncte)', 'PDO (points)'), [('20', '20' + lec), ('10', '10'), ('30', '30'), ('40', '40')], '20') + play('sw'),
                  charts('sw-chart') + we('sw-we', f'{WE}: {T("scoruri și puncte", "scores and points")}', r'\text{Score} = \text{Offset} + \text{Factor}\,\ln\frac{1 - p}{p}, \quad \text{Factor} = \frac{\text{PDO}}{\ln 2}, \quad \text{Offset} = 600 - \text{Factor}\,\ln 50',
                                          select('sw-p', T('PD pentru pasul 2', 'PD for step 2'), [('0.05', '5%' + lec), ('0.3', '30%'), ('0.01', '1%'), ('0.1', '10%')], '0.05'))),
     'cm-tab': (slider('cm-c', T('Pragul PD c', 'PD cut-off c'), 0.05, 0.9, 0.01, 0.5) + play('cm'),
                charts('cm-chart') + xh(T('Matricile de confuzie pe eșantionul de test (ultimul rînd: pragul ales)', 'Confusion matrices on the test sample (last row: the chosen cut-off)')) + tw('cm-tbl')
                + we('cm-we', f'{WE}: {T("ratele la pragul ales", "the rates at the chosen cut-off")}', r'\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \quad \mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}, \quad \text{cost} = \frac{5\,\mathrm{FN} + \mathrm{FP}}{n}')),
     'auc-we': (select('aw-m', T('Scorul', 'Score'), MOD, 'logit') + play('aw'),
                charts('aw-chart') + we('aw-we', f'{WE}: {T("AUC ca probabilitate de ordonare", "AUC as a rank probability")}', r'\mathrm{AUC} = P(s_{\text{bad}} > s_{\text{good}}) + \tfrac12 P(s_{\text{bad}} = s_{\text{good}}), \qquad \text{Gini} = 2\,\mathrm{AUC} - 1')),
     'ks-we': (select('kw-m', T('Scorul', 'Score'), MOD, 'logit') + play('kw'),
               charts('kw-chart') + we('kw-we', f'{WE}: {T("KS, Gini și AR", "KS, Gini and AR")}', r'\mathrm{KS} = \max_s |F_{\text{bad}}(s) - F_{\text{good}}(s)| = \max_c\,(\mathrm{TPR} - \mathrm{FPR}), \qquad \mathrm{AR} \approx \text{Gini}')),
     'brier-we': (GR('bw-g') + play('bw'),
                  charts('bw-chart') + we('bw-we', f'{WE}: {T("scorul Brier și testul Hosmer–Lemeshow", "the Brier score and the Hosmer–Lemeshow test")}',
                                          r'\mathrm{BS} = \frac1n\sum_i (p_i - y_i)^2, \qquad \mathrm{HL} = \sum_g \frac{(O_g - n_g\bar p_g)^2}{n_g\bar p_g(1 - \bar p_g)} \approx \chi^2(G - 2)')
                  + xh(T('Tabelul de calibrare al scorecard-ului (eșantionul de test)', 'Calibration table of the scorecard (test sample)')) + tw('bw-tab')),
     'reject-tab': (select('rj-a', T('Ponderea acceptată de scorul vechi', 'Share accepted by the old score'), [('0.7', '70%' + lec), ('0.5', '50%'), ('0.6', '60%'), ('0.8', '80%'), ('0.9', '90%')], '0.7') + play('rj'),
                    charts('rj-chart') + xh(T('Experimentul de reject inference (eșantionul de test, 9000 de clienți)', 'The reject-inference experiment (test sample, 9,000 clients)')) + tw('rj-tab')),
     'z-we': (slider('zw-x', 'X₃ = EBIT / TA', 0, 0.4, 0.01, 0.08) + play('zw'),
              charts('zw-chart') + we('zw-we', f'{WE}: {T("scorul Z", "the Z-score")}', c(r'Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5'),
                                      select('zw-4', T('X₄ = capitaluri proprii / datorii', 'X₄ = equity / debt'), [('0.9', n('0.90') + lec), ('0.5', n('0.50')), ('1.5', n('1.50'))], '0.9'))),
     'dd-we': (slider('dd-s', T('Volatilitatea activelor σ_V', 'Asset volatility σ_V'), 0.1, 0.6, 0.01, 0.25) + play('dd'),
               charts('dd-chart') + we('dd-st', f'{WE}: {T("distanța pînă la nerambursare", "distance to default")}', r'\mathrm{DD} = \frac{\ln(V_0/D) + (\mu - \sigma_V^2/2)T}{\sigma_V\sqrt{T}}, \qquad \mathrm{PD} = \Phi(-\mathrm{DD})',
                                       select('dd-e', T('Capitalurile proprii E (pasul 3)', 'Equity value E (step 3)'), [('40', '40' + lec), ('30', '30'), ('60', '60')], '40'))),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tt = lambda t: latex_text(re.sub(r'\\sfmchlinkt?\{[^{}]*\}\{([^{}]*)\}', r'\1', t))
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    ro = frames_of(open(os.path.join(ROOT, TEX['ro']) + '.tex', encoding='utf-8').read())
    assert len(ro) == len(frames) == len(pages), (lang, len(ro), len(frames), len(pages))
    res = {}
    for a, fr, _, _ in XBLOCKS:
        assert tt(ro[fr - 1][0]) == TITLES[a], (a, tt(ro[fr - 1][0]))
        k = fr - 1
        assert not re.search(r'includegraphics[^{]*\{sfm_', frames[k][1]), (lang, a)
        assert frames[k][1].count('tabular') == ro[fr - 1][1].count('tabular'), (lang, a)
        res[a] = (int(pages[k][0]), tt(frames[k][0]))
    return res


def xblock_html(anchor, lang, web, pg=None, title=None):
    t = L[lang]
    ctl, ch = widgets_all(lang)[anchor]
    q, a = QA[lang][anchor]
    tail = (f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>'
            f'<details><summary>{t["answer"]}</summary><p>{a}</p></details>')
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    if web:
        return f'<section class="blk wlive xblk" id="{anchor}" data-frame="{pg}">{live}{tail}</section>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch12.js)."""
    W = widgets_all(lang)
    fp = frame_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''.join(xblock_html(x, lang, True, fp[x][0]) for x, _, _, host in XBLOCKS if host == a)
        res[a] = (ctl, ch + (f'<div class="xhold">{held}</div>' if held else ''))
    return res


MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'options: { skipHtmlTags: [\'script\', \'noscript\', \'style\', \'textarea\', \'pre\', \'code\', \'select\', \'option\', \'button\', \'label\'] }, '
           "startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute('data-mj', 'done')) } };</script>\n"
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def page(lang):
    """The interactive page: interactive_common.page for the chart blocks, then the table blocks inserted in slide order."""
    import types
    import interactive_common as ic
    sp = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA')})
    sp.widgets = widgets_all
    html = ic.page(sp, lang)
    t, fp = L[lang], frame_pages(lang)
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    blocks = {m.group(2): m.group(1) for m in re.finditer(r'(\n<section class="blk" id="([^"]+)">.*?</section>)', html, flags=re.S)}
    start, end = html.index('<p class="toc">'), html.index('<p class="src foot">')
    body, nav, sec = [], [], None
    secs = {a: s for a, _, s in BLOCKS} | {a: s for a, _, s, _ in XBLOCKS}
    for a in ORDER:
        if secs[a] != sec:
            sec = secs[a]
            body.append(f'<h2 class="sec" id="s-{sec}">{SECTIONS[lang][sec]}</h2>')
            nav.append(f'<a href="#s-{sec}">{SECTIONS[lang][sec]}</a>')
        if a in blocks:
            body.append(blocks[a])
            continue
        pg, title = fp[a]
        live, tail = xblock_html(a, lang, False)
        body.append(f'''
<section class="blk" id="{a}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch12_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    html = html[:start] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[end:]
    return html.replace('</head>', MATHJAX + '</head>', 1)


def build():
    import pymupdf
    import types
    import interactive_common as ic
    os.makedirs(os.path.join(ic.OUT, 'slides'), exist_ok=True)
    for lang in ('en', 'ro'):
        with open(os.path.join(ic.OUT, L[lang]['file']), 'w', encoding='utf-8') as f:
            f.write(page(lang))
        with open(os.path.join(ic.OUT, f'ch{N}_text_{lang}.js'), 'w', encoding='utf-8') as f:
            f.write(f'// Readouts of the Chapter {N} interactive page, written by latex/build_interactive_ch{N}.py.\n' + TEXT_JS[lang])
        doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
        sp = types.SimpleNamespace(TEX=TEX, BLOCKS=BLOCKS)
        for chart, (pg, _) in ic.slide_pages(sp, lang).items():
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'{chart}_{lang}.jpg'), jpg_quality=82)
        for a, (pg, _) in frame_pages(lang).items():
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch12_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s\u00a0%]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const X = ' (rulare suplimentară a codului din curs)';
const lst = a => (a.length ? a.join(', ') : 'niciuna');
const cap = s => s[0].toUpperCase() + s.slice(1);
const bl = l => l.replace('<=', '≤').replace('-', '–');
const VN = { status: 'contul curent', credit_history: 'istoricul de credit', purpose: 'destinația creditului', savings: 'economiile', duration: 'durata', employment_duration: 'vechimea la angajator',
  amount: 'suma', age: 'vîrsta', property: 'proprietatea', housing: 'locuința', installment_rate: 'rata ca pondere din venit', personal_status_sex: 'starea civilă și sexul', other_debtors: 'alți debitori',
  present_residence: 'vechimea la adresă', other_installment_plans: 'alte credite în rate', number_credits: 'numărul de credite', job: 'ocupația', people_liable: 'persoanele în întreținere',
  telephone: 'telefonul', foreign_worker: 'lucrătorul străin' };
const LAB = { status: { 1: 'fără cont curent', 2: 'sold < 0 DM', 3: '0–200 DM', 4: '≥ 200 DM sau cont de salariu' },
  credit_history: { 0: 'întîrzieri la plată în trecut', 1: 'cont critic sau alte credite în altă parte', 2: 'niciun credit sau toate rambursate la timp', 3: 'creditele existente rambursate la timp pînă acum', 4: 'toate creditele la această bancă rambursate la timp' },
  savings: { 1: 'fără cont de economii sau necunoscut', 2: '< 100 DM', 3: '100–500 DM', 4: '500–1000 DM', 5: '≥ 1000 DM' },
  employment_duration: { 1: 'fără loc de muncă', 2: '< 1 an', 3: '1–4 ani', 4: '4–7 ani', 5: '≥ 7 ani' } };
const SV = { const: 'termenul liber', duration_years: 'durata (ani)', age_decades: 'vîrsta (decenii)', amount_1000: 'suma (1000 DM)', no_account: 'fără cont curent', negative_balance: 'sold < 0 DM' };
const MOD = { 'WoE logit': 'logit WoE', 'Fisher LDA': 'LDA Fisher', 'checking account only': 'doar contul curent', LDA: 'LDA', 'small logit': 'logit mic', 'status only': 'contul curent' };
const FGL = { men: 'bărbați', women: 'femei', 'age <= 25': 'vîrsta ≤ 25', 'age 26-35': 'vîrsta 26–35', 'age 36-50': 'vîrsta 36–50', 'age > 50': 'vîrsta > 50' };
const CLS = { strong: 'puternică', medium: 'medie', weak: 'slabă', useless: 'inutilă' };
const ZONE = { distress: 'de dificultate', grey: 'gri', safe: 'sigură' };
const SMP = { full: 'complet (1000 de credite)', train: 'de estimare (700 de credite)', test: 'de test (300 de credite)' };
const DROP = { duration_years: 'durata', age_decades: 'vîrsta', amount_1000: 'suma', acct: 'indicatorii de cont' };
const AC = { none: 'fără cont curent', neg: 'sold < 0 DM', ok: 'sold nenegativ' };
const PK = { 'duration,age': 'durata în luni, vîrsta în ani', 'duration,amount': 'durata în luni, suma în 1000 DM', 'age,amount': 'vîrsta în ani, suma în 1000 DM' };
window.CH12_TEXT = {
 vn: v => VN[v], lab: (v, c) => (LAB[v] && LAB[v][c]) || String(c), sv: n => SV[n], cls: c => CLS[c], fg: g => FGL[g],
 h: { cal: ['grupul', 'n', 'PD medie', 'rata observată', 'O (nerambursări)', 'E = n × PD medie', 'termen HL'],
      wt: ['Grupul', 'bun-pl.', 'rău-pl.', 'g_i/G', 'b_i/B', 'WoE', 'termen IV'], total: 'total',
      sm: ['Variabila', 'β̂', 'se', 'z', 'p', 'raportul șanselor (CI 95%)'], wl: ['Variabila WoE', 'IV (estimare)', 'β̂', 'se'], const: 'termenul liber',
      lt: ['', 'Logit', 'LDA (Fisher)'], auc: 'AUC', brier: 'scorul Brier', corr: 'corelația logaritmilor șansei', delong: 'testul DeLong (AUC egale)',
      cm: ['prag', 'TP', 'FN', 'FP', 'TN', 'acuratețe', 'TPR', 'FPR', 'cost'],
      rj: ['', 'estimat pe acceptați', 'estimat pe toți', 'scorul vechi'], rjAuc: 'AUC, toți solicitanții de test', rjAuc2: 'AUC, acceptații de test', rjRej: 'PD medie a respinșilor de test',
      rjObs: 'rata observată a respinșilor de test' },
 cp: (rho, a, b, c, yI, pI, x) => `ρ = ${rho}${x ? X : ''}. Un credit cu PD = 2% intră în nerambursare cu probabilitatea ${a} într-un an prost (y = −3), ${b} într-un an obișnuit (y = 0) și ${c} într-un an bun (y = 3). În starea folosită de IRB, y = Φ⁻¹(0,001) = ${yI}: ${pI}.`,
 ir: (lgd, q, rows, x) => `LGD = ${lgd}, nivelul de încredere ${q}%${x ? X : ''}. ` + rows.map(([p, el, a, b, c]) => `PD = ${p}: EL = ${el}; K = ${a} (alte expuneri de retail), ${b} (ipotecare), ${c} (revolving)`).join('; ') + '.',
 ds: (v, hl, hr, hn, ll, lr, ln, ratio, x) => `Rata creditelor neperformante după ${VN[v]}, eșantionul complet${x ? X : ''}: cea mai mare la „${hl}” (${hr}, ${hn} ${de(hn)}credite), cea mai mică la „${ll}” (${lr}, ${ln} ${de(ln)}credite); raportul ${ratio}.`,
 wd: (v, s, iv, cls, rows, x) => `WoE pentru ${VN[v]}, eșantionul ${SMP[s]}${x ? X : ''}: ` + rows.map(([l, w, r]) => `${bl(l)}: WoE ${w}, ${r} neperformante`).join('; ') + `. IV = ${iv} (${cls}).`,
 iv: (s, v1, i1, v2, i2, sel, use, x) => `Eșantionul ${SMP[s]}${x ? X : ''}: cele mai informative sînt ${v1} (IV = ${i1}) și ${v2} (${i2}). IV ≥ 0,1: ${sel.length} ${sel.length === 1 ? 'variabilă' : 'variabile'} (${lst(sel)}); sub 0,02: ${lst(use)}.`,
 lc: (v, b0, b1, se, u, or, a, pa, b, pb, np, mn, x) => `Logit doar cu ${v === 'age' ? 'vîrsta' : 'durata'}${x ? X : ''}: β̂₀ = ${b0}, β̂₁ = ${b1} pe ${v === 'age' ? 'an' : 'lună'} (se ${se}); ${v === 'age' ? 'zece ani în plus înmulțesc' : 'un an în plus înmulțește'} șansa de nerambursare cu e^(${u} β̂₁) = ${or}. PD = ${pa} la ${a} ${de(a)}${v === 'age' ? 'ani' : 'luni'} și ${pb} la ${b} ${de(b)}${v === 'age' ? 'ani' : 'luni'}; ${np} ${de(np)}puncte cu cel puțin ${mn} ${mn === 1 ? 'credit' : 'credite'}.`,
 or: (lvl, z, rows, inc, x) => `Intervale de ${lvl}% (z = ${z})${x ? X : ''}: ` + rows.map(([n, o, l, h]) => `${n} ${o} (${l}–${h})`).join('; ') + `. Conțin valoarea 1: ${lst(inc)}.`,
 ld: (p, sL, sG, dL30, dG30, dL50, dG50, x) => `Frontierele la PD = ${p}${x ? X : ''}: pe frontieră, o lună de durată în plus cere ${sL} ani de vîrstă în plus (LDA), respectiv ${sG} (logit). La 30 de ani, frontiera trece prin ${dL30} luni (LDA) și ${dG30} luni (logit); la 50 de ani, prin ${dL50} și ${dG50} luni.`,
 sd: (cut, cs, ga, gr, ba, br, mn, mx, mg, mb, x) => `Pragul PD = ${cut} corespunde la ${cs} ${de(cs)}puncte${x ? X : ''}; acceptăm scorurile de cel puțin ${cs}. Acceptați: ${ga} ${de(ga)}bun-platnici și ${ba} ${de(ba)}rău-platnici; respinși: ${gr} ${de(gr)}bun-platnici și ${br} ${de(br)}rău-platnici. Scorurile: de la ${mn} la ${mx} puncte; media ${mg} (bun-platnici) și ${mb} (rău-platnici).`,
 co: (cfn, frac, bay, cb, rb, best, bc, acc, rej, x) => `C_FN = ${cfn}, C_FP = 1${x ? X : ''}: pragul Bayes este ${frac} = ${bay}; la el, costul pe solicitant este ${cb} și respingem ${rb} din solicitanți. Pe grila de praguri, minimul este ${bc}, la ${best}. Acceptăm pe toată lumea: ${acc}; respingem pe toată lumea: ${rej}.`,
 rc: (smp, aL, aQ, aS, lo, hi, g, c, tpr, fpr, np, x) => `Eșantionul ${smp === 'test' ? 'de test' : 'de estimare'}${x ? X : ''}, ${np} ${de(np)}perechi (rău, bun): AUC = ${aL} pentru logit WoE (CI 95% DeLong [${lo}; ${hi}], Gini ${g}), ${aQ} pentru LDA, ${aS} doar cu contul curent. Respingînd dacă PD > ${c}: TPR ${tpr}, FPR ${fpr}.`,
 ca: (xr, nr, hit, nb, perf, ar, g, x) => `Respingînd cei mai riscanți ${xr} (${nr} ${de(nr)}solicitanți)${x ? X : ''}, identificăm ${hit} din rău-platnici (${nb} din 90); modelul perfect ar identifica ${perf}. AR = ${ar}; Gini = 2 AUC − 1 = ${g}.`,
 ks: (ks, at, s, F1, F0, gap, x) => `KS = ${ks}, la ${at} ${de(at)}puncte. La s = ${s}${x ? X : ''}: ${F1} din rău-platnici și ${F0} din bun-platnici au scorul cel mult s; diferența ${gap}.`,
 cb: (k, hl, df, p, b, br, gain, pd1, r1, x) => `${k} ${de(k)}grupuri${x ? X : ''}: Hosmer–Lemeshow = ${hl} cu ${df} ${de(df)}grade de libertate, p ${p.startsWith('<') ? p : '= ' + p}. Scorul Brier ${b}, față de ${br} pentru o PD constantă (cîștig ${gain}). Grupul cel mai sigur: PD medie ${pd1}, rata observată ${r1}.`,
 cv: (r, nfo, rows, gap, x) => `${+r === 1 ? 'Prima repetare' : `Primele ${r} ${de(r)}repetări`} (${nfo} ${de(nfo)}părți)${x ? X : ''}; AUC medie de estimare / de test (abaterea standard la test): ` + rows.map(([m, a, b, s]) => `${MOD[m]} ${a} / ${b} (${s})`).join('; ') + `. Logit-ul WoE pierde ${gap} în afara eșantionului.`,
 fa: (a, cut, wA, mA, gap, wT, mT, minA, minG, x) => `Aprobăm cei ${a} cu PD cea mai mică (PD ≤ ${cut})${x ? X : ''}. Aprobate: femei ${wA}, bărbați ${mA} (diferența ${gap} puncte procentuale); bun-platnici aprobați: femei ${wT}, bărbați ${mT}. Cea mai mică rată de aprobare: ${minA} (${minG}).`,
 me: (mu, s, nr, dd, pd, x) => `μ = ${mu}, σ_V = ${s}${x ? X : ''}: ${nr} din cele 40 de traiectorii se termină sub datoria de 70. Pentru V₀ = 100 și D = 70: DD = ${dd}, PD = ${pd}.`,
 iw1: (pd, w, R) => `PD = ${pd}: w = (1 − e^(−35 PD))/(1 − e^(−35)) = ${w}; R = 0,03 w + 0,16 (1 − w) = ${R}.`,
 iw2: (a, z, sr, arg, pb) => `Φ⁻¹(PD) = ${a}, Φ⁻¹(0,999) = ${z}, √R = ${sr}; argumentul (${a} + ${sr} × ${z})/√(1 − R) = ${arg}; Φ(${arg}) = ${pb}: PD într-un an prost.`,
 iw3: (pb, pd, K, ead, Kl) => `K = 0,45 × (${pb} − ${pd}) = ${K} din EAD: ${Kl} ${de(Kl)}lei pentru un credit de ${ead} ${de(ead)}lei.`,
 iw4: (Kl, rwa, wgt) => `RWA = 12,5 × ${Kl} = ${rwa} ${de(rwa)}lei: o pondere de risc de ${wgt}.`,
 iw5: (pd, ead, el, Kl) => `EL = PD × LGD × EAD = ${pd} × 0,45 × ${ead} = ${el} ${de(el)}lei intră în preț; capitalul (${Kl} ${de(Kl)}lei) acoperă pierderea neașteptată.`,
 iw: (pd, R, pb, K, wgt, x) => `PD = ${pd}${x ? X : ''}: R = ${R}, PD într-un an prost ${pb}, K = ${K} din EAD, pondere de risc ${wgt}.`,
 wt1: (lab, g, b, dg, db) => `Rîndul „${lab}”: ${g} ${de(g)}bun-platnici din 700 și ${b} ${de(b)}rău-platnici din 300; g/G = ${g}/700 = ${dg}, b/B = ${b}/300 = ${db}.`,
 wt2: (dg, db, w, iv) => `WoE = ln(${dg}/${db}) = ${w}; termenul IV = (${dg} − ${db}) × (${w}) = ${iv}.`,
 wt3: (iv, cls) => `IV = suma termenilor = ${iv}: o variabilă ${cls} după regula lui Siddiqi.`,
 wt: (v, iv, cls, x) => `${cap(VN[v])}, eșantionul complet${x ? X : ''}: IV = ${iv} (${cls}).`,
 ow1: (p, q, o) => `p = ${p}: șansa de nerambursare p/(1 − p) = ${p}/${q} = ${o}.`,
 ow2: (o, l) => `Logaritmul șansei: ln ${o} = ${l}.`,
 ow3: (gb, nl) => `Șansa bun:rău (1 − p)/p = ${gb}:1; logit(1 − p) = ${nl} = −logit(p).`,
 ow: (p, o, l, x) => `p = ${p}${x ? X : ''}: șansa ${o}, logaritmul șansei ${l}.`,
 sm: (d, n, ll, ll0, lr, df, ra, x) => `${d === 'none' ? 'Modelul din curs' : 'Modelul fără ' + DROP[d]}${x ? X : ''}: n = ${n}, log-verosimilitatea ${ll} față de ${ll0} doar cu termenul liber, LR = ${lr} cu ${df} grade de libertate.` + (ra ? ` Coeficientul sumei: ${ra[0]} (p ${ra[1].startsWith('<') ? ra[1] : '= ' + ra[1]}).` : ''),
 pw1: (dm, ag, ac) => `Solicitantul: ${dm} ${de(dm)}luni, ${ag} de ani, 4000 DM, ${AC[ac]}; coeficienții modelului mic, rotunjiți.`,
 pw2: (b0, tD, tA, tM, tAcc, e) => `η = ${b0} + ${tD} + (${tA}) + ${tM} + ${tAcc} = ${e}.`,
 pw3: (e, p, o) => `PD = 1/(1 + e^(−η)) = ${p}, cu η = ${e}; șansa e^η = ${o}.`,
 pw4: (e0, p0) => `Același solicitant cu un sold nenegativ: η = ${e0}, PD = ${p0}.`,
 pw5: (d, dpp) => `Un an în plus de durată: Δp ≈ p(1 − p)β = ${d}, adică ${dpp} puncte procentuale.`,
 pw: (dm, p, e, x) => `${dm} ${de(dm)}luni${x ? X : ''}: η = ${e}, PD = ${p} în eșantion (30% rău-platnici).`,
 wl: (t, k, aT, aR, ll, x) => `Pragul IV ${t}${x ? X : ''}: ${k} ${k === 1 ? 'variabilă' : 'variabile'} alese pe eșantionul de estimare; AUC de test ${aT}, AUC de estimare ${aR}; log-verosimilitatea ${ll}.`,
 pr1: (pi, lpi, lps, sh) => `π = ${pi}, π_s = 30%: corecția = ln(π/(1 − π)) − ln(π_s/(1 − π_s)) = ${lpi} − (${lps}) = ${sh}.`,
 pr2: (p30, p50) => `O PD de 30% în eșantion devine ${p30}; o PD de 50% devine ${p50}.`,
 pr3: (w, e0, sh, e1, p1, p0) => `${w === 'app' ? 'Solicitantul din exemplul rezolvat' : 'O PD de ' + p0 + ' în eșantion'}: logaritmul șansei ${e0} + (${sh}) = ${e1}, PD = ${p1} în populație, în loc de ${p0}.`,
 pr4: (f, msh, pts) => `În puncte: logaritmul șansei bun:rău crește cu ${msh}, deci fiecare scor crește cu ${f} × ${msh} = ${pts} puncte.`,
 pr: (pi, sh, pa, x) => `π = ${pi}${x ? X : ''}: corecția termenului liber ${sh}; solicitantul din exemplu are PD = ${pa} în populație.`,
 fw1: (pk, n0, n1, m0, m1, dm) => `x = (${PK[pk]}), eșantionul complet: ${n0} ${de(n0)}bun-platnici, ${n1} ${de(n1)}rău-platnici; m₀ = ${m0}, m₁ = ${m1}, m₁ − m₀ = ${dm}.`,
 fw2: (s11, s12, s22, det) => `S_W are pe diagonală ${s11} și ${s22}, în afara ei ${s12}; det S_W = ${det}.`,
 fw3: (w, b) => `w = S_W⁻¹(m₁ − m₀) = ${w}; logit cu aceleași două variabile: pantele ${b}.`,
 fw4: (pk, rL, rG) => `Raportul dintre cele două ponderi: LDA ${rL}, logit ${rG}.`,
 fw: (pk, w0, w1, aL, aG, x) => `x = (${PK[pk]})${x ? X : ''}: w = (${w0}; ${w1}); AUC în eșantion ${aL} (LDA) și ${aG} (logit).`,
 lt: (te, n, a1, a2, r, z, p, x) => `Eșantionul ${te ? 'de test' : 'de estimare'} (${n} ${de(n)}credite)${x ? X : ''}: AUC ${a1} (logit) și ${a2} (LDA); corelația logaritmilor șansei ${r}; DeLong z = ${z}, p = ${p}.`,
 sw1: (pdo, f, o) => `PDO = ${pdo}: Factor = ${pdo}/ln 2 = ${f}; Offset = 600 − ${f} × ln 50 = ${o}.`,
 sw2: (pp, od, lo, s) => `PD = ${pp}: șansa bun:rău ${od}:1, ln ${od} = ${lo}; scorul = Offset + Factor × ${lo} = ${s}.`,
 sw3: (lo, od, pd) => `De la 560 de puncte la PD: ln șansa = (560 − Offset)/Factor = ${lo}, șansa ${od}:1, PD = ${pd}.`,
 sw4: (k, pts) => `Punctele contului curent (k = ${k} variabile, WoE din estimare): fără cont ${pts[0]}, sold < 0 ${pts[1]}, 0–200 DM ${pts[2]}, cel puțin 200 DM sau salariu ${pts[3]}.`,
 sw5: (f, sh, pts) => `Corecția a priori pentru o populație cu 5% rău-platnici: fiecare scor crește cu ${f} × ${sh} = ${pts} puncte.`,
 sw: (pdo, s5, s30, pd560, x) => `PDO = ${pdo}${x ? X : ''}: PD 5% → ${s5} puncte, PD 30% → ${s30} puncte; 560 de puncte → PD ${pd560}.`,
 cm1: (c, TP, FN, FP, TN) => `Respingem dacă PD > ${c}: TP = ${TP}, FN = ${FN}, FP = ${FP}, TN = ${TN} (90 de rău-platnici, 210 bun-platnici).`,
 cm2: (TP, FN, tpr, FP, TN, fpr) => `TPR = ${TP}/(${TP} + ${FN}) = ${tpr}; FPR = ${FP}/(${FP} + ${TN}) = ${fpr}.`,
 cm3: (TP, FP, pr, TN, acc) => `Precizia = ${TP}/(${TP} + ${FP}) = ${pr}; acuratețea = (${TP} + ${TN})/300 = ${acc}.`,
 cm4: (FN, FP, c) => `Costul pe solicitant = (5 × ${FN} + ${FP})/300 = ${c}.`,
 cm: (c, acc, tpr, fpr, cost, x) => `Pragul ${c}${x ? X : ''}: acuratețea ${acc}, TPR ${tpr}, FPR ${fpr}, costul pe solicitant ${cost}.`,
 aw1: (n1, n0, N) => `Eșantionul de test: ${n1} ${de(n1)}rău-platnici × ${n0} ${de(n0)}bun-platnici = ${N} ${de(N)}perechi (rău, bun).`,
 aw2: (m, gt, eq, lt) => `${cap(MOD[m])}: în ${gt} ${de(gt)}perechi rău-platnicul are scorul de risc mai mare, în ${eq} ${de(eq)}perechi scorurile sînt egale, în ${lt} ${de(lt)}perechi bun-platnicul are scorul mai mare.`,
 aw3: (gt, eq, N, A) => `AUC = (${gt} + ½ × ${eq})/${N} = ${A}: statistica Mann–Whitney împărțită la n₁n₀.`,
 aw4: (A, g) => `Gini = 2 × ${A} − 1 = ${g}.`,
 aw5: (se, lo, hi) => `Eroarea standard DeLong ${se}; intervalul de 95%: [${lo}; ${hi}].`,
 aw: (m, A, eq, g, x) => `${cap(MOD[m])}${x ? X : ''}: AUC = ${A}, perechi cu scoruri egale ${eq}, Gini = ${g}.`,
 kw1: (m, ks, tpr, fpr) => `${cap(MOD[m])}: KS = max (TPR − FPR) = ${ks}, atins la TPR = ${tpr} și FPR = ${fpr}.`,
 kw2: (A, g, ar) => `AUC = ${A}, deci Gini = 2 AUC − 1 = ${g}; AR din curba CAP = ${ar}.`,
 kw3: single => (single ? 'Cu doar patru valori ale scorului, CAP ordonează solicitanții cu același scor în ordinea din fișier, deci AR se îndepărtează de Gini; egalitatea AR = Gini cere un scor continuu.'
   : 'Pentru un scor continuu, AR și Gini coincid, cu excepția rotunjirii și a celor cîteva scoruri egale.'),
 kw: (m, ks, g, ar, x) => `${cap(MOD[m])}${x ? X : ''}: KS = ${ks}, Gini = ${g}, AR = ${ar}.`,
 bw1: (b, rate, br) => `BS = (1/300) Σ (p_i − y_i)² = ${b}; referința, o PD constantă de ${rate} (rata din eșantionul de estimare): ${br}.`,
 bw2: (b, br, g) => `Cîștigul = 1 − ${b}/${br} = ${g}.`,
 bw3: (k, hl, df, p) => `Hosmer–Lemeshow cu ${k} ${de(k)}grupuri: ${hl}, χ² cu ${df} ${de(df)}grade de libertate, p = ${p}.`,
 bw: (k, b, hl, df, p, x) => `${k} ${de(k)}grupuri${x ? X : ''}: Brier ${b}; HL = ${hl}, ${df} ${de(df)}grade de libertate, p = ${p}.`,
 rj: (a, nacc, ba, br, nd, aA, aF, aO, pdR, rR, under, x) => `Scorul vechi acceptă cei mai buni ${a} dintre clienții de estimare (${nacc})${x ? X : ''}; rata de nerambursare: ${ba} la acceptați, ${br} la respinși. ${nd ? nd + ' variabile nu variază printre acceptați și ies din model.' : 'Toate variabilele variază printre acceptați.'} AUC pe toți solicitanții de test: ${aA} (estimat pe acceptați), ${aF} (estimat pe toți), ${aO} (scorul vechi). Respinșii de test: rata observată ${rR}, PD medie a modelului estimat pe acceptați ${pdR} (${num(under) > 0 ? 'subestimare cu ' + under + '%' : 'supraestimare cu ' + under.replace('−', '') + '%'}).`,
 zw1: X_ => `X₁ = ${X_[0]}; X₂ = ${X_[1]}; X₃ = ${X_[2]}; X₄ = ${X_[3]}; X₅ = ${X_[4]}.`,
 zw2: (c, Z, zone) => `Z = ${c.join(' + ')} = ${Z}: zona ${ZONE[zone]}.`,
 zw3: (need, rest) => `Fără X₃, ceilalți patru termeni dau ${rest}; Z = 2,99 cere X₃ = (2,99 − ${rest})/3,3 = ${need}.`,
 zw: (x3, Z, zone, x) => `X₃ = ${x3}${x ? X : ''}: Z = ${Z}, zona ${ZONE[zone]}.`,
 dd1: (l, s, dr) => `ln(100/70) = ${l}; cu σ_V = ${s}: μ − σ_V²/2 = ${dr}.`,
 dd2: (n, s, dd, pd) => `DD = ${n}/${s} = ${dd}; PD = Φ(−${dd}) = ${pd}.`,
 dd3: (E, V, sV, dd, pd) => `Din piață (capitaluri proprii ${E}, volatilitatea acțiunilor 50%, datoria 70, rata fără risc 3%): V = ${V}, σ_V = ${sV}, DD = ${dd}, PD = ${pd}.`,
 dd: (s, dd, pd, x) => `σ_V = ${s}${x ? X : ''}: DD = ${dd}, PD = ${pd}.`,
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/[\s,%]/g, '').replace('−', '-');
const X = ' (extra run of the code of the lecture)';
const lst = a => (a.length ? a.join(', ') : 'none');
const cap = s => s[0].toUpperCase() + s.slice(1);
const bl = l => l.replace('<=', '≤').replace('-', '–');
const pl = (n, s, p) => `${n} ${+num(n) === 1 ? s : p || s + 's'}`;
const VN = { status: 'checking account', credit_history: 'credit history', purpose: 'purpose', savings: 'savings', duration: 'duration', employment_duration: 'employment duration',
  amount: 'amount', age: 'age', property: 'property', housing: 'housing', installment_rate: 'instalment rate', personal_status_sex: 'personal status and sex', other_debtors: 'other debtors',
  present_residence: 'present residence', other_installment_plans: 'other instalment plans', number_credits: 'number of credits', job: 'job', people_liable: 'people liable',
  telephone: 'telephone', foreign_worker: 'foreign worker' };
const LAB = { status: { 1: 'no checking account', 2: 'balance < 0 DM', 3: '0–200 DM', 4: '≥ 200 DM or salary account' },
  credit_history: { 0: 'delays in paying off in the past', 1: 'critical account or other credits elsewhere', 2: 'no credits taken or all paid back duly', 3: 'existing credits paid back duly till now', 4: 'all credits at this bank paid back duly' },
  savings: { 1: 'no savings account or unknown', 2: '< 100 DM', 3: '100–500 DM', 4: '500–1000 DM', 5: '≥ 1000 DM' },
  employment_duration: { 1: 'unemployed', 2: '< 1 year', 3: '1–4 years', 4: '4–7 years', 5: '≥ 7 years' } };
const SV = { const: 'constant', duration_years: 'duration (years)', age_decades: 'age (decades)', amount_1000: 'amount (1000 DM)', no_account: 'no checking account', negative_balance: 'balance < 0 DM' };
const MOD = { 'WoE logit': 'WoE logit', 'Fisher LDA': 'Fisher LDA', 'checking account only': 'checking account only', LDA: 'LDA', 'small logit': 'small logit', 'status only': 'checking account only' };
const FGL = { men: 'men', women: 'women', 'age <= 25': 'age ≤ 25', 'age 26-35': 'age 26–35', 'age 36-50': 'age 36–50', 'age > 50': 'age > 50' };
const CLS = { strong: 'strong', medium: 'medium', weak: 'weak', useless: 'useless' };
const ZONE = { distress: 'distress', grey: 'grey', safe: 'safe' };
const SMP = { full: 'full sample (1,000 credits)', train: 'training sample (700 credits)', test: 'test sample (300 credits)' };
const DROP = { duration_years: 'duration', age_decades: 'age', amount_1000: 'amount', acct: 'the account indicators' };
const AC = { none: 'no checking account', neg: 'balance < 0 DM', ok: 'non-negative balance' };
const PK = { 'duration,age': 'duration in months, age in years', 'duration,amount': 'duration in months, amount in 1000 DM', 'age,amount': 'age in years, amount in 1000 DM' };
window.CH12_TEXT = {
 vn: v => VN[v], lab: (v, c) => (LAB[v] && LAB[v][c]) || String(c), sv: n => SV[n], cls: c => CLS[c], fg: g => FGL[g],
 h: { cal: ['group', 'n', 'mean PD', 'observed rate', 'O (defaults)', 'E = n × mean PD', 'HL term'],
      wt: ['Group', 'goods', 'bads', 'g_i/G', 'b_i/B', 'WoE', 'IV term'], total: 'total',
      sm: ['Variable', 'β̂', 'se', 'z', 'p', 'odds ratio (95% CI)'], wl: ['WoE variable', 'IV (training)', 'β̂', 'se'], const: 'constant',
      lt: ['', 'Logit', 'LDA (Fisher)'], auc: 'AUC', brier: 'Brier score', corr: 'correlation of the log-odds', delong: 'DeLong test (equal AUC)',
      cm: ['cut-off', 'TP', 'FN', 'FP', 'TN', 'accuracy', 'TPR', 'FPR', 'cost'],
      rj: ['', 'trained on accepted', 'trained on all', 'old score'], rjAuc: 'AUC, all test applicants', rjAuc2: 'AUC, accepted test applicants', rjRej: 'mean PD of the rejected test applicants',
      rjObs: 'observed rate of the rejected test applicants' },
 cp: (rho, a, b, c, yI, pI, x) => `ρ = ${rho}${x ? X : ''}. A loan with PD = 2% defaults with probability ${a} in a bad year (y = −3), ${b} in a typical year (y = 0) and ${c} in a good year (y = 3). In the state used by IRB, y = Φ⁻¹(0.001) = ${yI}: ${pI}.`,
 ir: (lgd, q, rows, x) => `LGD = ${lgd}, confidence level ${q}%${x ? X : ''}. ` + rows.map(([p, el, a, b, c]) => `PD = ${p}: EL = ${el}; K = ${a} (other retail), ${b} (mortgage), ${c} (revolving)`).join('; ') + '.',
 ds: (v, hl, hr, hn, ll, lr, ln, ratio, x) => `Bad rate by ${VN[v]}, full sample${x ? X : ''}: highest for “${hl}” (${hr}, ${pl(hn, 'credit')}), lowest for “${ll}” (${lr}, ${pl(ln, 'credit')}); ratio ${ratio}.`,
 wd: (v, s, iv, cls, rows, x) => `WoE of the ${VN[v]}, ${SMP[s]}${x ? X : ''}: ` + rows.map(([l, w, r]) => `${bl(l)}: WoE ${w}, ${r} bad`).join('; ') + `. IV = ${iv} (${cls}).`,
 iv: (s, v1, i1, v2, i2, sel, use, x) => `${cap(SMP[s])}${x ? X : ''}: the most informative are ${v1} (IV = ${i1}) and ${v2} (${i2}). IV ≥ 0.1: ${pl(sel.length, 'variable')} (${lst(sel)}); below 0.02: ${lst(use)}.`,
 lc: (v, b0, b1, se, u, or, a, pa, b, pb, np, mn, x) => `Logit with ${v === 'age' ? 'age' : 'duration'} alone${x ? X : ''}: β̂₀ = ${b0}, β̂₁ = ${b1} per ${v === 'age' ? 'year' : 'month'} (se ${se}); ${v === 'age' ? 'ten more years multiply' : 'one more year multiplies'} the odds of default by e^(${u} β̂₁) = ${or}. PD = ${pa} at ${a} ${v === 'age' ? 'years' : 'months'} and ${pb} at ${b} ${v === 'age' ? 'years' : 'months'}; ${pl(np, 'dot')} with at least ${pl(mn, 'credit')}.`,
 or: (lvl, z, rows, inc, x) => `${lvl}% intervals (z = ${z})${x ? X : ''}: ` + rows.map(([n, o, l, h]) => `${n} ${o} (${l}–${h})`).join('; ') + `. Containing 1: ${lst(inc)}.`,
 ld: (p, sL, sG, dL30, dG30, dL50, dG50, x) => `Boundaries at PD = ${p}${x ? X : ''}: on the boundary, one more month of duration needs ${sL} more years of age (LDA) or ${sG} (logit). At age 30 the boundary passes through ${dL30} months (LDA) and ${dG30} months (logit); at age 50 through ${dL50} and ${dG50} months.`,
 sd: (cut, cs, ga, gr, ba, br, mn, mx, mg, mb, x) => `The cut-off PD = ${cut} corresponds to ${cs} points${x ? X : ''}; we accept scores of at least ${cs}. Accepted: ${ga} goods and ${ba} bads; rejected: ${gr} goods and ${br} bads. Scores from ${mn} to ${mx} points; mean ${mg} (goods) and ${mb} (bads).`,
 co: (cfn, frac, bay, cb, rb, best, bc, acc, rej, x) => `C_FN = ${cfn}, C_FP = 1${x ? X : ''}: the Bayes cut-off is ${frac} = ${bay}; there the cost per applicant is ${cb} and we reject ${rb} of the applicants. On the grid of cut-offs the minimum is ${bc}, at ${best}. Accept everybody: ${acc}; reject everybody: ${rej}.`,
 rc: (smp, aL, aQ, aS, lo, hi, g, c, tpr, fpr, np, x) => `${smp === 'test' ? 'Test' : 'Training'} sample${x ? X : ''}, ${np} (bad, good) pairs: AUC = ${aL} for the WoE logit (DeLong 95% CI [${lo}, ${hi}], Gini ${g}), ${aQ} for the LDA, ${aS} with the checking account only. Rejecting if PD > ${c}: TPR ${tpr}, FPR ${fpr}.`,
 ca: (xr, nr, hit, nb, perf, ar, g, x) => `Rejecting the riskiest ${xr} (${nr} applicants)${x ? X : ''}, we catch ${hit} of the bads (${nb} of 90); the perfect model would catch ${perf}. AR = ${ar}; Gini = 2 AUC − 1 = ${g}.`,
 ks: (ks, at, s, F1, F0, gap, x) => `KS = ${ks}, at ${at} points. At s = ${s}${x ? X : ''}: ${F1} of the bads and ${F0} of the goods score at most s; the gap is ${gap}.`,
 cb: (k, hl, df, p, b, br, gain, pd1, r1, x) => `${k} groups${x ? X : ''}: Hosmer–Lemeshow = ${hl} with ${df} degrees of freedom, p ${p.startsWith('<') ? p : '= ' + p}. Brier score ${b}, against ${br} for a constant PD (gain ${gain}). Safest group: mean PD ${pd1}, observed rate ${r1}.`,
 cv: (r, nfo, rows, gap, x) => `${+r === 1 ? 'First repetition' : `First ${r} repetitions`} (${nfo} folds)${x ? X : ''}; mean training / test AUC (sd of the test AUC): ` + rows.map(([m, a, b, s]) => `${MOD[m]} ${a} / ${b} (${s})`).join('; ') + `. The WoE logit loses ${gap} out of sample.`,
 fa: (a, cut, wA, mA, gap, wT, mT, minA, minG, x) => `We approve the ${a} with the lowest PD (PD ≤ ${cut})${x ? X : ''}. Approved: women ${wA}, men ${mA} (a gap of ${gap} percentage points); goods approved: women ${wT}, men ${mT}. Lowest approval rate: ${minA} (${minG}).`,
 me: (mu, s, nr, dd, pd, x) => `μ = ${mu}, σ_V = ${s}${x ? X : ''}: ${nr} of the 40 paths end below the debt of 70. For V₀ = 100 and D = 70: DD = ${dd}, PD = ${pd}.`,
 iw1: (pd, w, R) => `PD = ${pd}: w = (1 − e^(−35 PD))/(1 − e^(−35)) = ${w}; R = 0.03 w + 0.16 (1 − w) = ${R}.`,
 iw2: (a, z, sr, arg, pb) => `Φ⁻¹(PD) = ${a}, Φ⁻¹(0.999) = ${z}, √R = ${sr}; the argument (${a} + ${sr} × ${z})/√(1 − R) = ${arg}; Φ(${arg}) = ${pb}: the PD in a bad year.`,
 iw3: (pb, pd, K, ead, Kl) => `K = 0.45 × (${pb} − ${pd}) = ${K} of EAD: ${Kl} lei for a credit of ${ead} lei.`,
 iw4: (Kl, rwa, wgt) => `RWA = 12.5 × ${Kl} = ${rwa} lei: a risk weight of ${wgt}.`,
 iw5: (pd, ead, el, Kl) => `EL = PD × LGD × EAD = ${pd} × 0.45 × ${ead} = ${el} lei goes into the price; the capital (${Kl} lei) covers the unexpected loss.`,
 iw: (pd, R, pb, K, wgt, x) => `PD = ${pd}${x ? X : ''}: R = ${R}, PD in a bad year ${pb}, K = ${K} of EAD, risk weight ${wgt}.`,
 wt1: (lab, g, b, dg, db) => `Row “${lab}”: ${g} of the 700 goods and ${b} of the 300 bads; g/G = ${g}/700 = ${dg}, b/B = ${b}/300 = ${db}.`,
 wt2: (dg, db, w, iv) => `WoE = ln(${dg}/${db}) = ${w}; IV term = (${dg} − ${db}) × (${w}) = ${iv}.`,
 wt3: (iv, cls) => `IV = sum of the terms = ${iv}: a ${cls} variable by Siddiqi's rule.`,
 wt: (v, iv, cls, x) => `${cap(VN[v])}, full sample${x ? X : ''}: IV = ${iv} (${cls}).`,
 ow1: (p, q, o) => `p = ${p}: the odds of default p/(1 − p) = ${p}/${q} = ${o}.`,
 ow2: (o, l) => `The log-odds: ln ${o} = ${l}.`,
 ow3: (gb, nl) => `Good:bad odds (1 − p)/p = ${gb}:1; logit(1 − p) = ${nl} = −logit(p).`,
 ow: (p, o, l, x) => `p = ${p}${x ? X : ''}: odds ${o}, log-odds ${l}.`,
 sm: (d, n, ll, ll0, lr, df, ra, x) => `${d === 'none' ? 'The model of the lecture' : 'The model without ' + DROP[d]}${x ? X : ''}: n = ${n}, log-likelihood ${ll} against ${ll0} with the constant only, LR = ${lr} with ${df} degrees of freedom.` + (ra ? ` Coefficient of the amount: ${ra[0]} (p ${ra[1].startsWith('<') ? ra[1] : '= ' + ra[1]}).` : ''),
 pw1: (dm, ag, ac) => `The applicant: ${dm} months, ${ag} years old, 4000 DM, ${AC[ac]}; coefficients of the small logit, rounded.`,
 pw2: (b0, tD, tA, tM, tAcc, e) => `η = ${b0} + ${tD} + (${tA}) + ${tM} + ${tAcc} = ${e}.`,
 pw3: (e, p, o) => `PD = 1/(1 + e^(−η)) = ${p}, with η = ${e}; odds e^η = ${o}.`,
 pw4: (e0, p0) => `The same applicant with a non-negative balance: η = ${e0}, PD = ${p0}.`,
 pw5: (d, dpp) => `One more year of duration: Δp ≈ p(1 − p)β = ${d}, that is ${dpp} percentage points.`,
 pw: (dm, p, e, x) => `${dm} months${x ? X : ''}: η = ${e}, PD = ${p} in the sample (30% bads).`,
 wl: (t, k, aT, aR, ll, x) => `IV threshold ${t}${x ? X : ''}: ${pl(k, 'variable')} chosen on the training sample; test AUC ${aT}, training AUC ${aR}; log-likelihood ${ll}.`,
 pr1: (pi, lpi, lps, sh) => `π = ${pi}, π_s = 30%: correction = ln(π/(1 − π)) − ln(π_s/(1 − π_s)) = ${lpi} − (${lps}) = ${sh}.`,
 pr2: (p30, p50) => `A sample PD of 30% becomes ${p30}; a sample PD of 50% becomes ${p50}.`,
 pr3: (w, e0, sh, e1, p1, p0) => `${w === 'app' ? 'The applicant of the worked example' : 'A sample PD of ' + p0}: log-odds ${e0} + (${sh}) = ${e1}, PD = ${p1} in the population, instead of ${p0}.`,
 pr4: (f, msh, pts) => `In points: the good:bad log-odds rise by ${msh}, so every score rises by ${f} × ${msh} = ${pts} points.`,
 pr: (pi, sh, pa, x) => `π = ${pi}${x ? X : ''}: correction of the constant ${sh}; the applicant of the example has PD = ${pa} in the population.`,
 fw1: (pk, n0, n1, m0, m1, dm) => `x = (${PK[pk]}), full sample: ${n0} goods, ${n1} bads; m₀ = ${m0}, m₁ = ${m1}, m₁ − m₀ = ${dm}.`,
 fw2: (s11, s12, s22, det) => `S_W has ${s11} and ${s22} on the diagonal and ${s12} off it; det S_W = ${det}.`,
 fw3: (w, b) => `w = S_W⁻¹(m₁ − m₀) = ${w}; logit with the same two variables: slopes ${b}.`,
 fw4: (pk, rL, rG) => `Ratio of the two weights: LDA ${rL}, logit ${rG}.`,
 fw: (pk, w0, w1, aL, aG, x) => `x = (${PK[pk]})${x ? X : ''}: w = (${w0}, ${w1}); in-sample AUC ${aL} (LDA) and ${aG} (logit).`,
 lt: (te, n, a1, a2, r, z, p, x) => `${te ? 'Test' : 'Training'} sample (${n} credits)${x ? X : ''}: AUC ${a1} (logit) and ${a2} (LDA); correlation of the log-odds ${r}; DeLong z = ${z}, p = ${p}.`,
 sw1: (pdo, f, o) => `PDO = ${pdo}: Factor = ${pdo}/ln 2 = ${f}; Offset = 600 − ${f} × ln 50 = ${o}.`,
 sw2: (pp, od, lo, s) => `PD = ${pp}: good:bad odds ${od}:1, ln ${od} = ${lo}; score = Offset + Factor × ${lo} = ${s}.`,
 sw3: (lo, od, pd) => `From 560 points to the PD: ln odds = (560 − Offset)/Factor = ${lo}, odds ${od}:1, PD = ${pd}.`,
 sw4: (k, pts) => `Points of the checking account (k = ${k} variables, training WoE): no account ${pts[0]}, balance < 0 ${pts[1]}, 0–200 DM ${pts[2]}, at least 200 DM or salary ${pts[3]}.`,
 sw5: (f, sh, pts) => `Prior correction for a population with 5% bads: every score rises by ${f} × ${sh} = ${pts} points.`,
 sw: (pdo, s5, s30, pd560, x) => `PDO = ${pdo}${x ? X : ''}: PD 5% → ${s5} points, PD 30% → ${s30} points; 560 points → PD ${pd560}.`,
 cm1: (c, TP, FN, FP, TN) => `Reject if PD > ${c}: TP = ${TP}, FN = ${FN}, FP = ${FP}, TN = ${TN} (90 bads, 210 goods).`,
 cm2: (TP, FN, tpr, FP, TN, fpr) => `TPR = ${TP}/(${TP} + ${FN}) = ${tpr}; FPR = ${FP}/(${FP} + ${TN}) = ${fpr}.`,
 cm3: (TP, FP, pr, TN, acc) => `Precision = ${TP}/(${TP} + ${FP}) = ${pr}; accuracy = (${TP} + ${TN})/300 = ${acc}.`,
 cm4: (FN, FP, c) => `Cost per applicant = (5 × ${FN} + ${FP})/300 = ${c}.`,
 cm: (c, acc, tpr, fpr, cost, x) => `Cut-off ${c}${x ? X : ''}: accuracy ${acc}, TPR ${tpr}, FPR ${fpr}, cost per applicant ${cost}.`,
 aw1: (n1, n0, N) => `Test sample: ${n1} bads × ${n0} goods = ${N} (bad, good) pairs.`,
 aw2: (m, gt, eq, lt) => `${cap(MOD[m])}: in ${gt} pairs the bad has the higher risk score, in ${eq} pairs the scores are equal, in ${lt} pairs the good has the higher score.`,
 aw3: (gt, eq, N, A) => `AUC = (${gt} + ½ × ${eq})/${N} = ${A}: the Mann–Whitney statistic divided by n₁n₀.`,
 aw4: (A, g) => `Gini = 2 × ${A} − 1 = ${g}.`,
 aw5: (se, lo, hi) => `DeLong standard error ${se}; 95% interval [${lo}, ${hi}].`,
 aw: (m, A, eq, g, x) => `${cap(MOD[m])}${x ? X : ''}: AUC = ${A}, tied pairs ${eq}, Gini = ${g}.`,
 kw1: (m, ks, tpr, fpr) => `${cap(MOD[m])}: KS = max (TPR − FPR) = ${ks}, reached at TPR = ${tpr} and FPR = ${fpr}.`,
 kw2: (A, g, ar) => `AUC = ${A}, so Gini = 2 AUC − 1 = ${g}; AR from the CAP curve = ${ar}.`,
 kw3: single => (single ? 'With only four score values, the CAP orders applicants with the same score in file order, so AR departs from Gini; AR = Gini needs a continuous score.'
   : 'For a continuous score AR and Gini coincide, up to rounding and the few tied scores.'),
 kw: (m, ks, g, ar, x) => `${cap(MOD[m])}${x ? X : ''}: KS = ${ks}, Gini = ${g}, AR = ${ar}.`,
 bw1: (b, rate, br) => `BS = (1/300) Σ (p_i − y_i)² = ${b}; the reference, a constant PD of ${rate} (the rate of the training sample): ${br}.`,
 bw2: (b, br, g) => `Gain = 1 − ${b}/${br} = ${g}.`,
 bw3: (k, hl, df, p) => `Hosmer–Lemeshow with ${k} groups: ${hl}, χ² with ${df} degrees of freedom, p = ${p}.`,
 bw: (k, b, hl, df, p, x) => `${k} groups${x ? X : ''}: Brier ${b}; HL = ${hl}, ${df} degrees of freedom, p = ${p}.`,
 rj: (a, nacc, ba, br, nd, aA, aF, aO, pdR, rR, under, x) => `The old score accepts the best ${a} of the training clients (${nacc})${x ? X : ''}; default rate ${ba} among the accepted, ${br} among the rejected. ${nd ? nd + ' variables do not vary among the accepted and drop out of the model.' : 'All variables vary among the accepted.'} AUC on all test applicants: ${aA} (trained on the accepted), ${aF} (trained on all), ${aO} (old score). Rejected test applicants: observed rate ${rR}, mean PD of the model trained on the accepted ${pdR} (${num(under) > 0 ? 'underestimated by ' + under + '%' : 'overestimated by ' + under.replace('−', '') + '%'}).`,
 zw1: X_ => `X₁ = ${X_[0]}, X₂ = ${X_[1]}, X₃ = ${X_[2]}, X₄ = ${X_[3]}, X₅ = ${X_[4]}.`,
 zw2: (c, Z, zone) => `Z = ${c.join(' + ')} = ${Z}: ${ZONE[zone]} zone.`,
 zw3: (need, rest) => `Without X₃ the other four terms give ${rest}; Z = 2.99 needs X₃ = (2.99 − ${rest})/3.3 = ${need}.`,
 zw: (x3, Z, zone, x) => `X₃ = ${x3}${x ? X : ''}: Z = ${Z}, ${ZONE[zone]} zone.`,
 dd1: (l, s, dr) => `ln(100/70) = ${l}; with σ_V = ${s}: μ − σ_V²/2 = ${dr}.`,
 dd2: (n, s, dd, pd) => `DD = ${n}/${s} = ${dd}; PD = Φ(−${dd}) = ${pd}.`,
 dd3: (E, V, sV, dd, pd) => `From the market (equity ${E}, equity volatility 50%, debt 70, risk-free rate 3%): V = ${V}, σ_V = ${sV}, DD = ${dd}, PD = ${pd}.`,
 dd: (s, dd, pd, x) => `σ_V = ${s}${x ? X : ''}: DD = ${dd}, PD = ${pd}.`,
};
})();
""",
}

QA = {
 'ro': {
  'cpd': ('▶ crește ρ de la 0,2 la 0,5. Ce se întîmplă cu probabilitatea ca un credit cu PD = 2% să intre în nerambursare într-un an prost (y = −3)?',
          'Crește de la 21,3% la 53,8%; într-un an obișnuit (y = 0) scade de la 1,08% la 0,18%. O pondere mai mare a factorului comun concentrează nerambursările în anii proști, '
          'iar media peste toți anii rămîne 2%.'),
  'irb-we': ('▶ crește PD de la 2% la 4%. Se dublează capitalul K?',
             'Nu: crește de la 4,64% la 5,20% din EAD (+12%), în timp ce EL se dublează, de la 0,90% la 1,80%. Corelația R scade de la 0,095 la 0,062 cînd PD crește, iar '
             'nerambursările în plus sînt tot mai mult așteptate, nu neașteptate.'),
  'irb': ('Alte expuneri de retail, LGD 45%: PD crește de la 5% la 10%, deci EL se dublează (de la 2,25% la 4,50% din EAD). Ce se întîmplă cu capitalul K?',
          'Crește doar de la 5,31% la 6,04% din EAD (+14%). K acoperă pierderea neașteptată: cu cît PD este mai mare, cu atît o parte mai mare din pierdere este așteptată și inclusă '
          'în preț, iar R scade de la 0,053 la 0,034. Peste o PD de aproximativ 16%, K este chiar mai mic decît EL.'),
  'status': ('În etichetele vechi Statlog, cele 394 de credite cu un sold de cel puțin 200 DM se numeau „fără cont curent” și păreau grupul cel mai sigur (11,7%). În datele corectate, '
             'care este rata creditelor neperformante fără cont curent?',
             '49,3% (274 de credite), grupul cel mai riscant, față de 11,7% pentru un sold de cel puțin 200 DM sau un cont de salariu: un raport de 4,2. Cifrele erau corecte în ambele '
             'versiuni; doar etichetele erau inversate.'),
  'woe-tab': ('Grupul „≥ 200 DM sau cont de salariu”: 348 din cei 700 de bun-platnici și 46 din cei 300 de rău-platnici. Cît este WoE?',
              'ln(0,497/0,153) = 1,176: ponderea bun-platnicilor este de 3,2 ori ponderea rău-platnicilor. Termenul IV, (0,497 − 0,153) × 1,176 = 0,404, este cel mai mare dintre cele patru; '
              'IV total este 0,666, o variabilă puternică.'),
  'woe-dur': ('▶ trece la suma creditului (intervale pînă la 1300, 2000, 3000, 5000 DM și peste). Crește rata creditelor neperformante constant cu suma?',
              'Nu: este 30,9% la creditele cele mai mici, scade la 23,9% la 2001–3000 DM și urcă la 41,5% peste 5000 DM (WoE de la 0,31 la −0,50). Relația are forma unui U, pe care '
              'un singur termen liniar într-un logit nu o surprinde, iar intervalele da; IV = 0,082, o variabilă slabă.'),
  'iv': ('Pe eșantionul complet, proprietatea are IV = 0,113 ≥ 0,1. Trece de pragul 0,1 și pe eșantionul de estimare, unde se aleg variabilele scorecard-ului?',
         'Nu: pe eșantionul de estimare IV este 0,082, iar vechimea la angajator (0,086 pe eșantionul complet, 0,122 pe cel de estimare) îi ia locul. Șase variabile trec pragul în '
         'ambele cazuri. IV calculat pe cîteva sute de credite variază cu cîteva sutimi între eșantioane: pe cele 300 de credite de test, IV al contului curent este 0,913, față de 0,666 pe toate cele 1000.'),
  'odds-we': ('p = 0,2 dă logaritmul șansei −1,386. ▶ duce p la 0,8. Cît este logaritmul șansei acolo?',
              '+1,386: logit(1 − p) = −logit(p), logit-ul este simetric față de p = 0,5. Șansa este 4 (patru nerambursări la un credit rambursat), inversul lui 0,25.'),
  'logit-curve': ('▶ estimează logit-ul doar cu vîrsta. Cu cît înmulțesc zece ani în plus șansa de nerambursare?',
                  'Cu e^(10 × (−0,0185)) = 0,83, adică −17%: PD scade de la 34,0% la 25 de ani la 21,3% la 60 de ani. Durata singură dă 1,57 pe an: solicitanții mai în vîrstă sînt mai '
                  'siguri, creditele mai lungi mai riscante.'),
  'small-tab': ('▶ scoate durata din model. Ce se întîmplă cu coeficientul sumei (0,029, p = 0,368)?',
                'Crește la 0,112 pe 1000 DM, p < 0,001: fără durată, suma preia riscul creditelor lungi, cu care este corelată. Un coeficient este efectul cu celelalte variabile fixate; '
                'omiterea unei variabile corelate îl schimbă.'),
  'odds': ('Intervalul de 95% pentru vîrstă (+10 ani) este 0,74–0,98. Conține intervalul de 99% valoarea 1?',
           'Da: 0,71–1,02. Efectul vîrstei are p = 0,022: semnificativ la 5%, nu și la 1%. Intervalele duratei și ale celor doi indicatori de cont exclud 1 și la 99%; suma conține 1 la orice nivel.'),
  'pd-we': ('▶ crește durata de la 36 la 48 de luni pentru același solicitant. Cît este PD?',
            '71,8%: η crește cu 0,385, la 0,937. Aproximarea p(1 − p)β = 8,9 puncte de la 63,5% dă 72,4%, aproape, dar modificarea exactă, 8,3 puncte, este mai mică, pentru că '
            'curba logistică se aplatizează cînd p se îndepărtează de 0,5.'),
  'woe-logit-tab': ('▶ coboară pragul IV la 0,02, deci intră 15 variabile WoE în loc de 6. Crește AUC de test?',
                    'Nu: scade de la 0,801 la 0,787, în timp ce AUC de estimare crește de la 0,789 la 0,825. Cele nouă variabile slabe se adaptează la zgomotul celor 700 de credite de estimare.'),
  'prior-we': ('▶ coboară rata rău-platnicilor din populație π. Pentru solicitantul cu PD de 63,5% în eșantion, cît este PD la π = 1%?',
               '3,9%: corecția este ln(0,01/0,99) − ln(0,3/0,7) = −3,748, deci η = 0,552 − 3,748 = −3,196. Se mută doar termenul liber; ordonarea solicitanților, deci și AUC, rămîne aceeași.'),
  'fisher-we': ('Doar cu durata și vîrsta: ce AUC în eșantion atinge discriminantul Fisher pe cele 1000 de credite?',
                '0,640, exact cît logit-ul cu aceleași două variabile. Scorecard-ul cu șase variabile WoE atinge 0,801 pe eșantionul de test: două variabile sînt departe de a fi suficiente, '
                'cum arată și norii care se suprapun.'),
  'lda': ('▶ mută PD de pe frontieră de la 30% la 50%. Ce se întîmplă cu cele două drepte?',
          'Se deplasează paralel spre durate mai lungi: la 30 de ani, de la aproximativ 20 de luni (19,9 LDA, 19,4 logit) la aproximativ 41 de luni (40,9 și 42,0). Panta nu depinde de '
          'nivelul PD; se schimbă doar termenul liber.'),
  'lda-tab': ('Pe eșantionul de estimare (700 de credite), este AUC al logit-ului mai mare decît al LDA?',
              'Nu, este puțin mai mic: 0,789 față de 0,790, DeLong z = −0,48, p = 0,63. Pe eșantionul de test, logit-ul este puțin înainte (0,8013 față de 0,8009, z = 0,18). Cele două scoruri '
              'au corelația 0,999; niciuna dintre ordonări nu este separată de zgomot de acest test.'),
  'scale-we': ('▶ fixează PDO la 40 de puncte. Ce scor primește acum o PD de 5% (572,1 cu PDO = 20)?',
               '544,2: Factor 57,71, Offset 374,25. Punctul de bază rămîne 600 la șansa 50:1; o PD de 5% înseamnă șansa 19:1, sub bază, iar cu o scală mai abruptă scorul coboară și mai mult sub 600.'),
  'score': ('Pragul la PD = 1/6 este 534 de puncte. Cîți dintre cei 90 de rău-platnici de test au cel puțin 534 de puncte și ar fi acceptați?',
            '8 (FN): 82 din cei 90 de rău-platnici sînt respinși. Prețul: 112 din cei 210 bun-platnici sînt respinși. Cele două distribuții se suprapun, deci orice prag schimbă o eroare pe cealaltă.'),
  'cm-tab': ('▶ coboară pragul de la 0,5 la 0,3. Crește sau scade acuratețea și ce se întîmplă cu costul?',
             'Acuratețea scade de la 76,0% la 72,7%, costul scade de la 0,867 la 0,527: TPR crește de la 47,8% la 78,9%, FPR de la 11,9% la 30,0%. Acuratețea cîntărește la fel cele două erori; '
             'costul cîntărește de cinci ori un rău-platnic acceptat.'),
  'cost': ('▶ crește costul acceptării unui rău-platnic de la 5 la 10. Ce pondere din solicitanți respinge pragul Bayes?',
           '81%: pragul scade de la 1/6 la 1/11 = 0,091, iar ponderea respinșilor crește de la 64,7% la 81,0%; costul așteptat este 0,620, sub 0,700, costul respingerii tuturor.'),
  'auc-we': ('Contul curent singur are patru valori. Ce pondere din cele 18 900 de perechi (rău, bun) au scoruri egale?',
             '23,1% (4362 de perechi); fiecare contează 1/2, deci AUC = (11 579 + 2181)/18 900 = 0,728. Logit-ul WoE are doar 12 perechi egale (0,06%).'),
  'roc': ('Pe eșantionul de estimare (700 de credite), AUC al logit-ului WoE este peste sau sub 0,801, valoarea de pe eșantionul de test?',
          'Sub: 0,789. Performanța în eșantion este de obicei optimistă, dar această împărțire a fost norocoasă: în cele 100 de părți ale validării încrucișate, AUC medie de test este 0,764.'),
  'ks-we': ('Contul curent singur: este raportul de acuratețe AR egal cu indicele Gini, 0,456?',
            'Nu: AR = 0,240. Cu doar patru valori ale scorului, CAP ordonează solicitanții cu același scor în ordinea din fișier, care este arbitrară; AR = Gini este valabil pentru un scor '
            'continuu (logit WoE: 0,602 și 0,603). KS este 0,410, față de 0,511.'),
  'cap': ('Respingînd cei mai riscanți 30% dintre cei 300 de solicitanți de test, ce pondere din cei 90 de rău-platnici identificăm?',
          '60% (54 din 90). Modelul perfect i-ar identifica pe toți, deoarece 30% este exact rata rău-platnicilor; un scor aleator ar identifica 30%.'),
  'ks': ('La pragul Bayes, 534 de puncte, cît este diferența F_rău(s) − F_bun(s), față de KS = 0,511 la 516 puncte?',
         '0,373: 91,1% din rău-platnici și 53,8% din bun-platnici au cel mult 534 de puncte. Aceasta este TPR − FPR la acest prag; KS este maximul ei peste toate pragurile. Pragul care '
         'minimizează costul așteptat nu este cel care maximizează KS, pentru că KS cîntărește la fel cele două erori.'),
  'brier-we': ('Eșantionul de test al scorecard-ului: cu 5 grupuri de cîte 60 de credite în loc de 10, ce p-value dă testul Hosmer–Lemeshow?',
               '0,90 (HL = 0,589, 3 grade de libertate); 0,59 cu 10 grupuri, 0,82 cu 20. Nerespins în nicio grupare: cu 300 de credite testul are o putere mică, deci „nerespins” nu dovedește '
               'că PD sînt exacte.'),
  'calib': ('Cu 5 grupuri de cîte 1800 de clienți în loc de 10, mai respinge testul Hosmer–Lemeshow calibrarea la 5%?',
            'Da: HL = 22,6 cu 3 grade de libertate, p < 0,001 (cu 20 de grupuri, 67,5, p < 0,001). Cu 9000 de clienți, testul detectează abateri de unul sau două puncte procentuale; '
            'contează mărimea abaterilor, nu doar p-value-ul.'),
  'cv': ('▶ arată primele 1, 5, 10 și 20 de repetări. Cît se mișcă AUC medie de test a logit-ului WoE?',
         'Foarte puțin: 0,766; 0,767; 0,766; 0,764. O singură rulare cu 5 părți face deja media a cinci părți de test; împărțirea unică din curs (0,801) este o singură extragere dintr-o '
         'distribuție cu abaterea standard 0,032.'),
  'reject-tab': ('▶ lasă scorul vechi să accepte 90% în loc de 70%. Mai subestimează modelul estimat pe acceptați riscul respinșilor?',
                 'Nu, acum îl supraestimează: PD medie 80,0% față de 69,4% observat, iar AUC pe toți solicitanții (0,777) este cel al modelului complet. La 90%, printre acceptați sînt și '
                 'clienți care au întîrziat în septembrie, deci nicio variabilă nu iese din model; cei 10% respinși se află dincolo de datele de estimare, iar logit-ul extrapolează.'),
  'fair': ('▶ crește ponderea aprobaților de la 75% la 90%. Crește sau scade diferența dintre ratele de aprobare ale femeilor și bărbaților?',
           'Scade de la 4,1 la 2,2 puncte procentuale (90,9% față de 88,7%). Ordinea grupurilor rămîne aceeași, pentru că modelul este calibrat în fiecare grup, iar bărbații intră mai des '
           'în nerambursare (24,7% față de 20,5%).'),
  'z-we': ('▶ crește EBIT/TA de la 0. La ce valoare iese compania din zona gri (Z > 2,99)?',
           'Peste 0,27 (exact 0,2697): ceilalți patru termeni dau 2,100, iar 3,3 X₃ trebuie să adauge 0,890. EBIT/TA are cea mai mare pondere, 3,3, deci profitabilitatea mișcă cel mai mult scorul Z.'),
  'dd-we': ('▶ crește σ_V. Cît este PD pe un an la σ_V = 40% (6,66% la 25%)?',
            '20,7%: DD scade la (0,3567 + 0,05 − 0,08)/0,40 = 0,817. Volatilitatea intră de două ori: micșorează termenul de creștere și împarte distanța.'),
  'merton': ('Aceleași 40 de traiectorii, σ_V = 40% în loc de 25%: cîte se termină sub datoria de 70?',
             '8 din 40, față de 3 la 25% și niciuna la 15%; PD a modelului la D/V = 70% este 20,7%, 6,66% și 0,42%. Cu 40 de traiectorii, numărătorile sînt zgomotoase: 8/40 = 20%.'),
 },
 'en': {
  'cpd': ('▶ raises ρ from 0.2 to 0.5. What happens to the probability that a loan with PD = 2% defaults in a bad year (y = −3)?',
          'It rises from 21.3% to 53.8%; in a typical year (y = 0) it falls from 1.08% to 0.18%. A larger weight of the common factor concentrates the defaults in the bad years, '
          'while the average over all years stays at 2%.'),
  'irb-we': ('▶ raises PD from 2% to 4%. Does the capital K double?',
             'No: it rises from 4.64% to 5.20% of EAD (+12%), while EL doubles from 0.90% to 1.80%. The correlation R falls from 0.095 to 0.062 as PD rises, and the extra defaults are '
             'increasingly expected rather than unexpected.'),
  'irb': ('Other retail exposures, LGD 45%: PD rises from 5% to 10%, so EL doubles (from 2.25% to 4.50% of EAD). What happens to the capital K?',
          'It rises only from 5.31% to 6.04% of EAD (+14%). K covers the unexpected loss: the higher the PD, the larger the share of the loss that is expected and priced in, and R falls '
          'from 0.053 to 0.034. Above a PD of about 16%, K is even smaller than EL.'),
  'status': ('In the old Statlog labels, the 394 credits with a balance of at least 200 DM were called “no checking account” and looked like the safest group (11.7%). In the corrected '
             'data, what is the bad rate of the credits without a checking account?',
             '49.3% (274 credits), the riskiest group, against 11.7% for a balance of at least 200 DM or a salary account: a ratio of 4.2. The figures were right in both versions; only '
             'the labels were swapped.'),
  'woe-tab': ('Group “≥ 200 DM or salary account”: 348 of the 700 goods and 46 of the 300 bads. What is its WoE?',
              'ln(0.497/0.153) = 1.176: the share of goods is 3.2 times the share of bads. Its IV term, (0.497 − 0.153) × 1.176 = 0.404, is the largest of the four; the total IV is 0.666, '
              'a strong variable.'),
  'woe-dur': ('▶ moves to the credit amount (bins up to 1300, 2000, 3000, 5000 DM and above). Does the bad rate rise steadily with the amount?',
              'No: it is 30.9% for the smallest loans, falls to 23.9% at 2001–3000 DM and rises to 41.5% above 5000 DM (WoE from 0.31 to −0.50). The relation is U-shaped, which a single '
              'linear term in a logit misses and the bins capture; IV = 0.082, a weak variable.'),
  'iv': ('On the full sample, property has IV = 0.113 ≥ 0.1. Does it also pass the 0.1 threshold on the training sample, where the scorecard variables are chosen?',
         'No: on the training sample its IV is 0.082, and employment duration (0.086 on the full sample, 0.122 on the training sample) takes its place. Six variables pass in both cases. '
         'IV computed on a few hundred credits moves by several hundredths between samples: on the 300 test credits the IV of the checking account is 0.913, against 0.666 on all 1,000.'),
  'odds-we': ('p = 0.2 gives the log-odds −1.386. ▶ moves p to 0.8. What are the log-odds there?',
              '+1.386: logit(1 − p) = −logit(p), the logit is symmetric around p = 0.5. The odds are 4 (four defaults per repaid credit), the inverse of 0.25.'),
  'logit-curve': ('▶ fits the logit on age alone. By how much do ten more years multiply the odds of default?',
                  'By e^(10 × (−0.0185)) = 0.83, that is −17%: PD falls from 34.0% at age 25 to 21.3% at age 60. Duration alone gives 1.57 per year: older applicants are safer, longer '
                  'credits riskier.'),
  'small-tab': ('▶ drops duration from the model. What happens to the coefficient of the amount (0.029, p = 0.368)?',
                'It rises to 0.112 per 1000 DM, p < 0.001: without duration, the amount picks up the risk of long credits, with which it is correlated. A coefficient is the effect with '
                'the other variables held fixed; leaving out a correlated variable changes it.'),
  'odds': ('The 95% interval for age (+10 years) is 0.74–0.98. Does the 99% interval contain 1?',
           'Yes: 0.71–1.02. The age effect has p = 0.022: significant at 5%, not at 1%. The intervals of duration and of both account indicators exclude 1 also at 99%; the amount contains 1 at every level.'),
  'pd-we': ('▶ raises the duration from 36 to 48 months for the same applicant. What is the PD?',
            '71.8%: η rises by 0.385 to 0.937. The approximation p(1 − p)β = 8.9 points from 63.5% gives 72.4%, close, but the exact change, 8.3 points, is smaller because the logistic '
            'curve flattens as p moves away from 0.5.'),
  'woe-logit-tab': ('▶ lowers the IV threshold to 0.02, so 15 WoE variables enter instead of 6. Does the test AUC rise?',
                    'No: it falls from 0.801 to 0.787, while the training AUC rises from 0.789 to 0.825. The nine weak variables fit the noise of the 700 training credits.'),
  'prior-we': ('▶ lowers the population bad rate π. For the applicant with a sample PD of 63.5%, what is the PD at π = 1%?',
               '3.9%: the correction is ln(0.01/0.99) − ln(0.3/0.7) = −3.748, so η = 0.552 − 3.748 = −3.196. Only the constant moves; the ranking of the applicants, and with it the AUC, stays the same.'),
  'fisher-we': ('Duration and age alone: what in-sample AUC does Fisher’s discriminant reach on the 1,000 credits?',
                '0.640, exactly as much as the logit with the same two variables. The scorecard with six WoE variables reaches 0.801 on the test sample: two variables are far from enough, '
                'as the overlapping clouds also show.'),
  'lda': ('▶ moves the PD on the boundary from 30% to 50%. What happens to the two lines?',
          'They move in parallel to longer durations: at age 30 from about 20 months (19.9 LDA, 19.4 logit) to about 41 months (40.9 and 42.0). The slope does not depend on the PD level; '
          'only the constant changes.'),
  'lda-tab': ('On the training sample (700 credits), is the AUC of the logit higher than that of the LDA?',
              'No, slightly lower: 0.789 against 0.790, DeLong z = −0.48, p = 0.63. On the test sample the logit is slightly ahead (0.8013 against 0.8009, z = 0.18). The two scores '
              'correlate 0.999; neither ranking is separated from noise by this test.'),
  'scale-we': ('▶ sets PDO to 40 points. What score does a PD of 5% get now (572.1 with PDO = 20)?',
               '544.2: Factor 57.71, Offset 374.25. The base point stays at 600 for odds of 50:1; a PD of 5% means odds of 19:1, below the base, and with a steeper scale the score falls further below 600.'),
  'score': ('The cut-off at PD = 1/6 is 534 points. How many of the 90 bad test credits score at least 534 and would be accepted?',
            '8 (FN): 82 of the 90 bads are rejected. The price: 112 of the 210 goods are rejected. The two distributions overlap, so every cut-off trades one error for the other.'),
  'cm-tab': ('▶ lowers the cut-off from 0.5 to 0.3. Does accuracy rise or fall, and what happens to the cost?',
             'Accuracy falls from 76.0% to 72.7%, the cost falls from 0.867 to 0.527: TPR rises from 47.8% to 78.9%, FPR from 11.9% to 30.0%. Accuracy weighs both errors equally; '
             'the cost weighs an accepted bad five times.'),
  'cost': ('▶ raises the cost of accepting a bad from 5 to 10. Which share of the applicants does the Bayes cut-off reject?',
           '81%: the cut-off falls from 1/6 to 1/11 = 0.091 and the share rejected rises from 64.7% to 81.0%; the expected cost is 0.620, below the 0.700 of rejecting everybody.'),
  'auc-we': ('The checking account alone has four values. What share of the 18,900 (bad, good) pairs are tied?',
             '23.1% (4,362 pairs); each counts 1/2, so AUC = (11,579 + 2,181)/18,900 = 0.728. The WoE logit has only 12 tied pairs (0.06%).'),
  'roc': ('On the training sample (700 credits), is the AUC of the WoE logit above or below 0.801, the value on the test sample?',
          'Below: 0.789. In-sample performance is usually optimistic, but this split was lucky: over the 100 folds of the cross-validation the mean test AUC is 0.764.'),
  'ks-we': ('The checking account alone: is its accuracy ratio AR equal to its Gini, 0.456?',
            'No: AR = 0.240. With only four score values, the CAP orders applicants with the same score in file order, which is arbitrary; AR = Gini holds for a continuous score '
            '(WoE logit: 0.602 and 0.603). KS is 0.410, against 0.511.'),
  'cap': ('Rejecting the riskiest 30% of the 300 test applicants, what share of the 90 bads do we catch?',
          '60% (54 of 90). The perfect model would catch all of them, since 30% is exactly the bad rate; a random score would catch 30%.'),
  'ks': ('At the Bayes cut-off, 534 points, how large is the gap F_bad(s) − F_good(s), compared with KS = 0.511 at 516 points?',
         '0.373: 91.1% of the bads and 53.8% of the goods score at most 534. This is TPR − FPR at this cut-off; KS is its maximum over all cut-offs. The cut-off that minimises the expected '
         'cost is not the one that maximises KS, because KS weighs both errors equally.'),
  'brier-we': ('Scorecard test sample: with 5 groups of 60 credits instead of 10, what p-value does the Hosmer–Lemeshow test give?',
               '0.90 (HL = 0.589, 3 degrees of freedom); 0.59 with 10 groups, 0.82 with 20. Not rejected in any grouping: with 300 credits the test has little power, so “not rejected” '
               'does not prove that the PDs are exact.'),
  'calib': ('With 5 groups of 1,800 clients instead of 10, does the Hosmer–Lemeshow test still reject calibration at 5%?',
            'Yes: HL = 22.6 with 3 degrees of freedom, p < 0.001 (with 20 groups, 67.5, p < 0.001). With 9,000 clients the test detects deviations of one or two percentage points; '
            'what matters is the size of the deviations, not only the p-value.'),
  'cv': ('▶ shows the first 1, 5, 10 and 20 repetitions. How much does the mean test AUC of the WoE logit move?',
         'Very little: 0.766, 0.767, 0.766, 0.764. A single 5-fold run already averages five test folds; the single split of the lecture (0.801) is one draw from a distribution with standard deviation 0.032.'),
  'reject-tab': ('▶ lets the old score accept 90% instead of 70%. Does the model trained on the accepted still underestimate the risk of the rejected?',
                 'No, it now overestimates it: mean PD 80.0% against an observed 69.4%, and its AUC on all applicants (0.777) is that of the full model. At 90%, clients who paid late in '
                 'September are among the accepted, so no variable drops out; the rejected 10% lie beyond the training data, and the logit extrapolates.'),
  'fair': ('▶ raises the approval share from 75% to 90%. Does the gap between the approval rates of women and men grow or shrink?',
           'It shrinks from 4.1 to 2.2 percentage points (90.9% against 88.7%). The order of the groups stays the same, because the model is calibrated in each group and men default '
           'more often (24.7% against 20.5%).'),
  'z-we': ('▶ raises EBIT/TA from 0. At which value does the company leave the grey zone (Z > 2.99)?',
           'Above 0.27 (exactly 0.2697): the other four terms give 2.100, and 3.3 X₃ has to add 0.890. EBIT/TA has the largest weight, 3.3, so profitability moves the Z-score most.'),
  'dd-we': ('▶ raises σ_V. What is the one-year PD at σ_V = 40% (6.66% at 25%)?',
            '20.7%: DD falls to (0.3567 + 0.05 − 0.08)/0.40 = 0.817. Volatility enters twice: it lowers the drift term and divides the distance.'),
  'merton': ('The same 40 paths, σ_V = 40% instead of 25%: how many end below the debt of 70?',
             '8 of 40, against 3 at 25% and none at 15%; the model PD at D/V = 70% is 20.7%, 6.66% and 0.42%. With 40 paths the counts are noisy: 8/40 = 20%.'),
 },
}

PREDICT = {
 'cpd': (0, [('Crește la circa 54%.', 'It rises to about 54%.'), ('Rămîne 21,3%: PD necondiționată a creditului este tot 2%.', 'It stays at 21.3%: the unconditional PD of the loan is still 2%.'),
             ('Scade: un factor comun mai mare diversifică riscul.', 'It falls: a larger common factor diversifies the risk.')]),
 'irb-we': (0, [('Nu: crește cu circa 12%.', 'No: it rises by about 12%.'), ('Da: K este proporțional cu PD, ca EL.', 'Yes: K is proportional to PD, like EL.'),
                ('Scade: un credit mai riscant este mai mult pierdere așteptată.', 'It falls: a riskier loan is more expected loss.')]),
 'irb': (0, [('Crește doar cu circa 14%.', 'It rises by only about 14%.'), ('Se dublează, ca EL.', 'It doubles, like EL.'), ('Scade, pentru că R scade.', 'It falls, because R falls.')]),
 'status': (0, [('49,3%: grupul cel mai riscant.', '49.3%: the riskiest group.'), ('11,7%: schimbarea etichetelor nu schimbă grupul cel mai sigur.', '11.7%: relabelling does not change the safest group.'),
                ('Circa 30%, media eșantionului.', 'About 30%, the sample mean.')]),
 'woe-tab': (0, [('1,176 = ln(0,497/0,153).', '1.176 = ln(0.497/0.153).'), ('0,344 = 0,497 − 0,153: diferența ponderilor.', '0.344 = 0.497 − 0.153: the difference of the shares.'),
                 ('−1,176: ponderea rău-platnicilor împărțită la cea a bun-platnicilor.', '−1.176: the share of bads divided by the share of goods.')]),
 'woe-dur': (0, [('Nu: relația are forma unui U.', 'No: the relation is U-shaped.'), ('Da: creditele mai mari sînt mereu mai riscante.', 'Yes: larger loans are always riskier.'),
                 ('Nu există nicio relație: WoE este aproape 0 în toate intervalele.', 'There is no relation: WoE is near 0 in every bin.')]),
 'iv': (0, [('Nu: vechimea la angajator îi ia locul.', 'No: employment duration takes its place.'), ('Da: IV nu depinde de eșantion.', 'Yes: IV does not depend on the sample.'),
            ('Da, și toate cele 20 de variabile trec pragul pe un eșantion mai mic.', 'Yes, and all 20 variables pass the threshold on a smaller sample.')]),
 'odds-we': (0, [('+1,386.', '+1.386.'), ('−0,223 = ln 0,8: logaritmul lui p.', '−0.223 = ln 0.8: the log of p.'), ('4: șansa, nu logaritmul ei.', '4: the odds, not their log.')]),
 'logit-curve': (0, [('Cu circa 0,83 (−17%).', 'By about 0.83 (−17%).'), ('Cu 0,98: coeficientul pe un an, nu pe zece.', 'By 0.98: the coefficient per year, not per ten years.'),
                     ('Cu 1,57, ca durata.', 'By 1.57, like duration.')]),
 'small-tab': (0, [('Crește la circa 0,11 și devine semnificativ.', 'It rises to about 0.11 and becomes significant.'),
                   ('Rămîne 0,029: un coeficient nu depinde de celelalte variabile.', 'It stays at 0.029: a coefficient does not depend on the other variables.'),
                   ('Devine negativ: suma și durata se compensează.', 'It turns negative: amount and duration offset each other.')]),
 'odds': (0, [('Da: 0,71–1,02.', 'Yes: 0.71–1.02.'), ('Nu: estimarea 0,85 este sub 1 la orice nivel.', 'No: the estimate 0.85 is below 1 at any level.'),
              ('Nu: un interval mai larg exclude mai ușor valoarea 1.', 'No: a wider interval excludes 1 more easily.')]),
 'pd-we': (0, [('71,8%.', '71.8%.'), ('93%: 63,5% înmulțit cu raportul șanselor 1,47.', '93%: 63.5% times the odds ratio 1.47.'),
               ('100%: 63,5% plus β = 38,5 puncte procentuale.', '100%: 63.5% plus β = 38.5 percentage points.')]),
 'woe-logit-tab': (0, [('Nu: scade la 0,787 (supraajustare).', 'No: it falls to 0.787 (overfitting).'), ('Da: mai multe variabile aduc mai multă informație.', 'Yes: more variables bring more information.'),
                       ('Rămîne 0,801: variabilele slabe primesc coeficienți nuli.', 'It stays at 0.801: weak variables get zero coefficients.')]),
 'prior-we': (0, [('Circa 3,9%.', 'About 3.9%.'), ('2,1% = 63,5% × 1%/30%.', '2.1% = 63.5% × 1%/30%.'), ('63,5%: PD unei persoane nu depinde de eșantion.', '63.5%: the PD of a person does not depend on the sample.')]),
 'fisher-we': (0, [('0,64.', '0.64.'), ('0,80, ca scorecard-ul.', '0.80, like the scorecard.'), ('0,50: norii se suprapun, deci nu există discriminare.', '0.50: the clouds overlap, so there is no discrimination.')]),
 'lda': (0, [('Se deplasează paralel spre durate mai lungi.', 'They shift in parallel to longer durations.'), ('Se rotesc în jurul punctului mediu.', 'They rotate around the mean point.'),
             ('Dreapta LDA rămîne pe loc: probabilitatea a priori este fixă.', 'The LDA line stays put: the prior is fixed.')]),
 'lda-tab': (0, [('Nu: LDA este puțin înainte (0,790 față de 0,789).', 'No: the LDA is slightly ahead (0.790 against 0.789).'),
                 ('Da: logit-ul maximizează verosimilitatea pe aceste date, deci ordonează mai bine.', 'Yes: the logit maximises the likelihood on these data, so it ranks better.'),
                 ('Da, clar: DeLong p < 0,05.', 'Yes, clearly: DeLong p < 0.05.')]),
 'scale-we': (0, [('544,2.', '544.2.'), ('572,1: PDO nu mută scorurile.', '572.1: PDO does not move the scores.'), ('Circa 657: punctele de peste Offset se dublează.', 'About 657: the points above the Offset double.')]),
 'score': (0, [('8.', '8.'), ('45: jumătate, mediile diferă doar cu 37 de puncte.', '45: half, the means differ by only 37 points.'), ('Niciunul: scorul îi separă pe toți.', 'None: the score separates all of them.')]),
 'cm-tab': (0, [('Acuratețea scade, costul scade.', 'Accuracy falls, the cost falls.'), ('Amîndouă cresc.', 'Both rise.'), ('Acuratețea crește, costul crește.', 'Accuracy rises, the cost rises.')]),
 'cost': (0, [('Circa 81%.', 'About 81%.'), ('64,7%: pragul nu depinde de costuri.', '64.7%: the cut-off does not depend on the costs.'), ('30%: rata rău-platnicilor.', '30%: the bad rate.')]),
 'auc-we': (0, [('Circa 23%.', 'About 23%.'), ('0%: două credite nu au niciodată exact același scor.', '0%: two credits never have exactly the same score.'),
                ('Circa 61%: perechile ordonate corect.', 'About 61%: the correctly ordered pairs.')]),
 'roc': (0, [('Sub: 0,789.', 'Below: 0.789.'), ('Peste: performanța în eșantion este întotdeauna optimistă.', 'Above: in-sample performance is always optimistic.'),
             ('Exact 0,801: AUC nu depinde de eșantion.', 'Exactly 0.801: the AUC does not depend on the sample.')]),
 'ks-we': (0, [('Nu: AR = 0,240.', 'No: AR = 0.240.'), ('Da: AR = Gini întotdeauna.', 'Yes: AR = Gini always.'), ('Nu: AR = AUC = 0,728.', 'No: AR = AUC = 0.728.')]),
 'cap': (0, [('60%.', '60%.'), ('30%: cîți respingem.', '30%: as many as we reject.'), ('100%, ca modelul perfect.', '100%, like the perfect model.')]),
 'ks': (0, [('0,373.', '0.373.'), ('0,511: KS este diferența la orice scor.', '0.511: KS is the gap at every score.'), ('0,911: ponderea rău-platnicilor.', '0.911: the share of bads.')]),
 'brier-we': (0, [('Circa 0,9: nerespins.', 'About 0.9: not rejected.'), ('Sub 0,05, ca pe datele din Taiwan.', 'Below 0.05, as on the Taiwan data.'),
                  ('Tot 0,589: HL nu depinde de grupare.', 'Still 0.589: HL does not depend on the grouping.')]),
 'calib': (0, [('Da: HL = 22,6, p < 0,001.', 'Yes: HL = 22.6, p < 0.001.'), ('Nu: mai puține grupuri înseamnă mai puține abateri.', 'No: fewer groups mean fewer deviations.'),
               ('Nu se poate calcula: testul cere 10 grupuri.', 'It cannot be computed: the test needs 10 groups.')]),
 'cv': (0, [('Foarte puțin: între 0,764 și 0,767.', 'Very little: between 0.764 and 0.767.'), ('Urcă spre 0,801 cu mai multe repetări.', 'It climbs towards 0.801 with more repetitions.'),
            ('Scade spre 0,70, AUC a contului curent.', 'It falls towards 0.70, the AUC of the checking account.')]),
 'reject-tab': (0, [('Nu: acum îl supraestimează.', 'No: it now overestimates it.'), ('Da, cu circa 36%, ca la 70%.', 'Yes, by about 36%, as at 70%.'), ('Nu: îl estimează exact.', 'No: it estimates it exactly.')]),
 'fair': (0, [('Scade.', 'It shrinks.'), ('Crește.', 'It grows.'), ('Dispare: la 90% toată lumea este aprobată.', 'It disappears: at 90% everybody is approved.')]),
 'z-we': (0, [('0,27.', '0.27.'), ('0,19: creșterea necesară, citită ca nivel.', '0.19: the increase needed, read as the level.'), ('0,89: diferența, fără împărțirea la 3,3.', '0.89: the gap, without dividing by 3.3.')]),
 'dd-we': (0, [('20,7%.', '20.7%.'), ('10,7%: PD proporțională cu σ_V (6,66 × 40/25).', '10.7%: PD proportional to σ_V (6.66 × 40/25).'), ('6,66%: îndatorarea, nu volatilitatea, fixează PD.', '6.66%: leverage, not volatility, sets the PD.')]),
 'merton': (0, [('8.', '8.'), ('3: aceleași extrageri, aceleași nerambursări.', '3: the same draws, the same defaults.'), ('5 = 3 × 40/25: proporțional cu volatilitatea.', '5 = 3 × 40/25: in proportion to the volatility.')]),
}

# blocks whose initial chart, table or worked example shows the answer: veil the readout and those elements
VEIL = {a: {'out': True} for a in PREDICT}
VEIL.update({'irb': {'out': True, 'charts': ['ir-chart']}, 'status': {'out': True, 'charts': ['ds-chart']}, 'cap': {'out': True, 'charts': ['ca-chart']},
             'woe-tab': {'out': True, 'charts': ['wt-chart', 'wt-tab', 'wt-we-box']}})

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Un model are o acuratețe de 70% pe eșantionul de test South German Credit. Este bun?', 'A model has an accuracy of 70% on the South German Credit test sample. Is it good?'),
      key=0, opts=[
  (('Nu neapărat: regula „toți sînt bun-platnici” atinge deja 70%.', 'Not necessarily: the rule “everyone is good” already reaches 70%.'),
   ('210 din cele 300 de credite de test sînt bune; ne uităm la AUC, TPR, FPR și cost.', '210 of the 300 test credits are good; look at AUC, TPR, FPR and the cost.')),
  (('Da: 70% este mult peste 50%.', 'Yes: 70% is well above 50%.'),
   ('50% este o referință doar pentru clase echilibrate; aici 70% din credite sînt bune.', '50% is a reference only for balanced classes; here 70% of the credits are good.')),
  (('Nu: un model bun are cel puțin 90% acuratețe.', 'No: a good model has at least 90% accuracy.'),
   ('Nu există un prag universal: acuratețea se compară cu rata de bază și cu costurile erorilor.', 'There is no universal threshold: accuracy is compared with the base rate and the costs of the errors.'))]),
 dict(q=('Coeficientul unei variabile indicator într-un logit este 0,69. Ce înseamnă?', 'The coefficient of an indicator variable in a logit is 0.69. What does it mean?'),
      key=0, opts=[
  (('Șansa de nerambursare se înmulțește cu e^0,69 ≈ 2 pentru grup, față de grupul de referință.', 'The odds of default are multiplied by e^0.69 ≈ 2 for the group, against the reference group.'),
   ('β este modificarea logaritmului șansei; e^β este raportul șanselor.', 'β is the change in the log-odds; e^β is the odds ratio.')),
  (('PD crește cu 0,69 puncte procentuale.', 'PD rises by 0.69 percentage points.'),
   ('β este pe scala logaritmului șansei, nu a probabilității.', 'β is on the log-odds scale, not on the probability scale.')),
  (('PD se dublează.', 'PD doubles.'),
   ('Se dublează șansa, nu probabilitatea; cele două coincid aproximativ doar pentru PD mici.', 'The odds double, not the probability; the two roughly coincide only for small PDs.'))]),
 dict(q=('Schimbă corecția pentru suprareprezentarea rău-platnicilor valoarea AUC?', 'Does the correction for oversampled bads change the AUC?'),
      key=0, opts=[
  (('Nu: deplasează toți logaritmii șansei cu aceeași constantă.', 'No: it shifts all log-odds by the same constant.'),
   ('Ordonarea solicitanților nu se schimbă, iar AUC depinde doar de ordonare.', 'The ranking of the applicants does not change, and the AUC depends only on the ranking.')),
  (('Da: AUC scade pentru că rău-platnicii sînt mai rari în populație.', 'Yes: the AUC falls because bads are rarer in the population.'),
   ('AUC compară perechi (rău, bun); nu depinde de rata rău-platnicilor.', 'The AUC compares (bad, good) pairs; it does not depend on the bad rate.')),
  (('Da: PD se schimbă, deci se schimbă și curba ROC.', 'Yes: the PDs change, so the ROC curve changes.'),
   ('Curba ROC depinde doar de ordinea scorurilor; o transformare monotonă nu o schimbă.', 'The ROC curve depends only on the order of the scores; a monotone transformation does not change it.'))]),
 dict(q=('Scorecard-ul are AUC = 0,801 pe eșantionul de test. Cît este indicele Gini?', 'The scorecard has AUC = 0.801 on the test sample. What is its Gini?'),
      key=0, opts=[
  (('0,603 = 2 × 0,801 − 1.', '0.603 = 2 × 0.801 − 1.'), ('Gini = 2 AUC − 1: 0 pentru un scor aleator, 1 pentru unul perfect.', 'Gini = 2 AUC − 1: 0 for a random score, 1 for a perfect one.')),
  (('0,301 = 0,801 − 0,5.', '0.301 = 0.801 − 0.5.'), ('Lipsește factorul 2: Gini este aria dintre curbă și diagonală raportată la aria maximă, 0,5.', 'The factor 2 is missing: Gini is the area between the curve and the diagonal divided by the maximum area, 0.5.')),
  (('0,801: Gini și AUC sînt același lucru.', '0.801: Gini and AUC are the same thing.'), ('AUC = 0,5 pentru un scor aleator, Gini = 0; cele două măsuri au scale diferite.', 'AUC = 0.5 for a random score, Gini = 0; the two measures have different scales.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch12-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH12_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH12_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH12_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', '\u00a0', t)
    for a, (q, ans) in QA['ro'].items():
        QA['ro'][a] = (fix(q), fix(ans))
    for a, (k, opts) in PREDICT.items():
        PREDICT[a] = (k, [(fix(r), e) for r, e in opts])
    for t in TICKET:
        t['q'] = (fix(t['q'][0]), t['q'][1])
        t['opts'] = [((fix(o[0]), o[1]), (fix(w[0]), w[1])) for o, w in t['opts']]


_ro_numbers()
_predict_js()

if __name__ == '__main__':
    build()
