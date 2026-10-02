r"""
build_seminar1.py -- Seminarul 1 (Date, randamente și indicatori), EN + RO dintr-o singură sursă
==============================================================================================
Seminarul are loc ÎNAINTEA cursului 1: secțiunea „Ce vă trebuie azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C întrebări
deschise și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_01/sem1_results.json (seminar1.py).
Ieșire:
  EN/Seminars/seminar1_data_returns_indicators.tex          (+ _solutions.tex)
  RO/Seminarii/seminar1_date_randamente_indicatori_ro.tex   (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_01/seminar1.py && python3 latex/build_seminar1.py && python3 latex/sfm_build.py compile 1
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, block, cols, items, enum, table, fig   # noqa: E402
from ch1_common import QL, REFS, T, put_date   # noqa: E402

with open(os.path.join(QL, 'sem1_results.json')) as f:
    S = json.load(f)
V = Values()
D = Deck(1, 'seminar', refs=REFS)
SEMQL = 'SFM_ch1_seminar'


def qlsem():
    return '\\quantlet{SFM\\_ch1\\_seminar}{\\qlurl{SFM_ch1_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for k in ['a1', 'a2']:
    for i, (R, r) in enumerate(zip(A[k]['R'], A[k]['r']), 1):
        V.put(f'{k}.R{i}', R, 2)
        V.put(f'{k}.r{i}', r, 2)
    for c in ['R_total', 'R_compound', 'R_sum', 'r_sum', 'r_total']:
        V.put(f'{k}.{c}', A[k][c], 2)
V.put('a3.div', A['a3']['events']['div'], 4)
V.put('a3.split', A['a3']['events']['split'], 4)
V.put('a3.both', A['a3']['events']['div'] * A['a3']['events']['split'], 4)
for i, x in enumerate(A['a3']['adj']):
    V.put(f'a3.adj{i}', x, 2)
V.put('a4.div', A['a4']['events']['div'], 4)
V.put('a4.both', A['a4']['events']['div'] * A['a4']['events']['cons'], 4)
for i, x in enumerate(A['a4']['adj']):
    V.put(f'a4.adj{i}', x, 2)
a5 = A['a5']
for c, d in [('mu_log', 1), ('vol', 1), ('cagr', 1), ('mu_arith', 1), ('drag', 2), ('se', 1), ('ci_lo', 1), ('ci_hi', 1)]:
    V.put(f'a5.{c}', a5[c], d)
V.put('a5.var', 0.015 ** 2 * 250, 4)
for k in ['a6', 'a7']:
    a = A[k]
    V.raw(f'{k}.Rs', '; '.join('$' + f'{x:+.2f}' + '$' for x in a['R']))
    for c, d in [('mean', 2), ('sd', 2), ('mean_ann', 1), ('vol_ann', 1), ('sharpe', 2), ('dsd', 2), ('sortino', 2),
                 ('mdd', 2), ('cagr', 1), ('calmar', 2)]:
        V.put(f'{k}.{c}', a[c], d)
B1 = S['B1']
V.int('b1.n', B1['n'])
V.raw('b1.nbig', str(B1['n_big']))
V.put('b1.share', 100 * B1['share_big'], 1)
V.put('b1.volb', B1['vol_bnr'], 1)
V.put('b1.vole', B1['vol_eodhd'], 1)
V.put('b1.volc', B1['vol_eodhd_clean'], 1)
V.raw('b1.wy', str(B1['worst_year']))
V.raw('b1.wyn', str(B1['worst_year_n']))
for i, t in enumerate(B1['top'][:3], 1):
    put_date(V, f'b1.t{i}.date', t['date'])
    V.put(f'b1.t{i}.e', t['eodhd'], 1)
    V.put(f'b1.t{i}.b', t['bnr'], 2)
for key, d in [('b2', S['B2'])] + [(f'b3.{k}', v) for k, v in S['B3'].items()]:
    V.int(f'{key}.n', d['n'])
    V.put(f'{key}.ppy', d['obs_per_year'], 0)
    V.put(f'{key}.mean', d['mean'], 3)
    V.put(f'{key}.sd', d['sd'], 2)
    V.put(f'{key}.skew', d['skew'], 2)
    V.put(f'{key}.exkurt', d['exkurt'], 1)
    V.put(f'{key}.min', d['min'], 1)
    put_date(V, f'{key}.mindate', d['min_date'])
    V.put(f'{key}.max', d['max'], 1)
    V.put(f'{key}.annmean', d['ann_mean'], 1)
    V.put(f'{key}.annvol', d['ann_vol'], 1)
    V.put(f'{key}.se', d['se_ann_mean'], 1)
    V.put(f'{key}.lo', d['ci_lo'], 1)
    V.put(f'{key}.hi', d['ci_hi'], 1)
    V.put(f'{key}.t', d['t'], 2)
    V.put(f'{key}.p', d['p'], 3)
    V.put(f'{key}.years', d['years'], 1)
B4 = S['B4']
V.int('b4.n', B4['n'])
for c, d in [('q', 1), ('mean_d', 4), ('sd_d', 3), ('sr_d', 4), ('sr', 2), ('se', 2), ('lo', 2), ('hi', 2), ('bs_lo', 2), ('bs_hi', 2), ('bs_sd', 2)]:
    V.put(f'b4.{c}', B4[c], d)
V.put('b4.sqrtq', B4['q'] ** 0.5, 2)
V.put('b4.se_d', ((1 + B4['sr_d'] ** 2 / 2) / B4['n']) ** 0.5, 4)
B5 = S['B5']
for k in ['bet', 'tlv']:
    V.put(f'b5.{k}.sr', B5[k]['sr'], 2)
    V.put(f'b5.{k}.se', B5[k]['se'], 2)
V.put('b5.diff', B5['diff'], 2)
V.put('b5.lo', B5['diff_lo'], 2)
V.put('b5.hi', B5['diff_hi'], 2)
V.int('b5.n', B5['n'])
V.put('b5.corr', B5['corr'], 2)
B6 = S['B6']
V.int('b6.peak_level', round(B6['peak_level']))
V.int('b6.trough_level', round(B6['trough_level']))
for c, d in [('mdd', 1), ('years_to_recover', 1), ('need', 0), ('cagr', 1), ('mdd_period', 1), ('calmar', 2)]:
    V.put(f'b6.{c}', B6[c], d)
V.put('b6.below', 100 * B6['share_below20'], 0)
V.put('b6.absmdd', -B6['mdd_period'], 1)
for c in ['peak', 'trough', 'recovery', 'period_peak', 'period_trough']:
    put_date(V, f'b6.{c}', B6[c])
for k, b in S['B7'].items():
    for c, d in [('mdd', 1), ('mean_arith', 1), ('mean_log', 1), ('drag', 2), ('half_var', 2), ('vol', 1), ('cagr', 1), ('calmar', 2)]:
        V.put(f'b7.{k}.{c}', b[c], d)
    for c in ['peak', 'trough', 'recovery']:
        put_date(V, f'b7.{k}.{c}', b[c])
C1 = S['C1']
V.put('c1.rho', C1['spearman'], 2)
V.put('c1.p', C1['p'], 2)
V.raw('c1.n', str(C1['n']))
C2 = S['C2']
for c, d in [('btc_sd', 2), ('btc_vol365', 1), ('btc_vol252', 1), ('q', 0), ('btc_sr', 2), ('btc_se', 2), ('bet_sr', 2), ('bet_se', 2),
             ('btc_worst_day', 1), ('btc_mdd', 1), ('tlv_close_max', 0), ('tlv_adj_max', 1), ('tlv_vol_close', 1), ('tlv_vol_adj', 1)]:
    V.put(f'c2.{c}', C2[c], d)
put_date(V, 'c2.tlvdate', C2['tlv_close_max_date'])
# cifre derivate folosite în interpretări
import math   # noqa: E402
V.put('a3.raw5', 100 * (10.1 / 29.8 - 1), 1)
V.put('a3.adj5r', 100 * (A['a3']['adj'][5] / A['a3']['adj'][4] - 1), 1)
V.put('a4.raw', 100 * math.log(19.90 / 1.98), 0)
V.put('b1.ratio', B1['vol_eodhd'] / B1['vol_bnr'], 1)
V.put('b3.ratio', S['B3']['btc']['ann_vol'] / S['B3']['sp500']['ann_vol'], 1)
V.put('b7.share', 100 * S['B7']['btc']['drag'] / S['B7']['btc']['mean_arith'], 0)
V.put('b7.maxgap', max(abs(b['drag'] - b['half_var']) for b in S['B7'].values()), 2)
V.put('c1.se', sum(r['se1'] + r['se2'] for r in C1['table'].values()) / (2 * len(C1['table'])), 2)


def num(x, d=2):
    """A literal number for a table cell, marked for the RO decimal comma."""
    return '⁅' + f'{x:.{d}f}' + '⁆'


c1rows = [f'{name} & ${num(r["sr1"])}$ & ${num(r["se1"])}$ & ${num(r["sr2"])}$ & ${num(r["se2"])}$' for name, r in C1['table'].items()]

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we measure, from real data, how much an investment grew and how risky it was?',
       '\\textbf{Întrebarea}: cum măsurăm, din date reale, cît a crescut o investiție și cît de riscantă a fost?'),
     [T('this seminar comes \\textbf{before} Lecture 1: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 1: secțiunea „Noțiuni necesare azi” conține toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: returns, adjusted prices, annualisation and indicators on paper', 'Partea A: randamente, prețuri ajustate, anualizare și indicatori pe hîrtie'),
      T('Part B: the same quantities on real data, each with a measure of precision and an interpretation question',
        'Partea B: aceleași mărimi pe date reale, fiecare cu o măsură a preciziei și o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns generat de AI, de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('Nothing is handed in: the seminar is for practice; the solutions of [Proposed] tasks are discussed in class',
      'Temele nu se notează: seminarul servește drept exercițiu; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.0cm}' + TB + 'p{7.6cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('do simple and log returns add up over time?', 'se adună în timp randamentele simple și cele logaritmice?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A1',
     'A3, A4 & ' + T('how do a dividend, a split and a consolidation change past prices?', 'cum schimbă un dividend, un split și o consolidare prețurile trecute?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A3',
     'A5 & ' + T('from daily moments to annual mean, volatility, CAGR and drag', 'de la momentele zilnice la media anuală, volatilitatea anuală, CAGR și volatility drag') + ' & ' + T('Solved', 'Rezolvat') + ' & --',
     'A6, A7 & ' + T('Sharpe, Sortino, MDD and Calmar from a short price path', 'Sharpe, Sortino, MDD și Calmar dintr-o traiectorie scurtă de prețuri') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & A6',
     'B1 & ' + T('can we trust a EUR/RON series?', 'ne putem baza pe o serie EUR/RON?') + ' & ' + T('Solved', 'Rezolvat') + ' & --',
     'B2, B3 & ' + T('is the mean return of the BET (and of other markets) significantly positive?', 'este randamentul mediu al BET (și al altor piețe) semnificativ pozitiv?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B2',
     'B4, B5 & ' + T('is the Sharpe ratio gap between TLV and the BET significant?', 'este semnificativă diferența dintre rapoartele Sharpe ale TLV și BET?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B4',
     'B6, B7 & ' + T('what do drawdowns and the volatility drag show on real data?', 'ce arată drawdown-urile și volatility drag pe date reale?') + ' & ' + T('Solved, Proposed', 'Rezolvat, Propus') + ' & B6',
     'C1 & ' + T('do Sharpe ratios persist?', 'persistă rapoartele Sharpe?') + ' & ' + T('Proposed', 'Propus') + ' & B4, B6',
     'C2 & ' + T('what is wrong in an AI answer?', 'ce greșeli conține un răspuns generat de AI?') + ' & ' + T('Proposed', 'Propus') + ' & --'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')))

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500, DAX & EODHD & close & 2015--2026',
     'BET (B6) & ' + T('EODHD, official BVB values', 'EODHD, valorile oficiale BVB') + ' & close & 1997--2026',
     'Bitcoin & EODHD & ' + T('close, 7 days a week', 'close, 7 zile pe săptămînă') + ' & 2015--2026',
     'TLV, SNP, BRD, TGN, SNG, SNN & EODHD & adjusted close & 2015--2026',
     T('EUR/RON reference rate', 'Cursul de referință EUR/RON') + ' & BNR & ' + T('daily fixing', 'fixingul zilnic') + ' & 2005--2026',
     T('EUR/RON series (B1 only)', 'Seria EUR/RON (doar B1)') + ' & EODHD & close & 2005--2026'],
    size='footnotesize') + items(
    T('Weekends and repeated holiday closes are dropped (except Bitcoin); each series on its own calendar, joint analyses on common days',
      'Weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin); fiecare serie se analizează pe propriul calendar, iar analizele comune, pe zilele comune'),
    T('In the notebook: \\texttt{load\\_close(\'bet\')}, \\texttt{log\\_returns(\'sp500\')}; no account or key is needed',
      'În notebook: \\texttt{load\\_close(\'bet\')}, \\texttt{log\\_returns(\'sp500\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# CE VĂ TREBUIE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/4): Prices and Adjusted Prices', 'Noțiuni necesare azi (1/4): prețuri și prețuri ajustate'), items(
    T('Daily bar: open, high, low, close and volume of a trading day; $P_t$ = close (indices, FX, crypto) or adjusted close (stocks)',
      'Bara zilnică: deschiderea, maximul, minimul, închiderea și volumul unei zile; $P_t$ = închiderea (indici, valută, cripto) sau închiderea ajustată (acțiuni)'),
    (T('A \\textbf{corporate action} changes the price but not the value of a holding', 'Un \\textbf{eveniment corporativ} schimbă prețul, dar nu și valoarea deținerii'),
     [T('cash dividend $D$ paid on day $t$: factor $A = 1 - D/P_{t-1}$', 'dividend $D$ plătit în ziua $t$: factorul $A = 1 - D/P_{t-1}$'),
      T('split: $m$ new shares for $n$ old ones, $A = n/m$ (2-for-1: $A = 1/2$); consolidation of $n$ old shares into 1: $A = n$',
        'split: $m$ acțiuni noi pentru $n$ vechi, $A = n/m$ (2-la-1: $A = 1/2$); consolidarea a $n$ acțiuni vechi într-una: $A = n$')]),
    T('\\textbf{Adjusted price}: each price before the event is multiplied by $A$; the prices after it stay unchanged',
      '\\textbf{Prețul ajustat}: fiecare preț dinaintea evenimentului se înmulțește cu $A$; prețurile de după rămîn neschimbate'),
    T('Rule: returns from adjusted prices have no artificial jumps and include dividends',
      'Regula: randamentele din prețuri ajustate nu au salturi artificiale și includ dividendele')))

D.frame(T('What You Need for Today (2/4): Simple and Log Returns', 'Noțiuni necesare azi (2/4): randamente simple și logaritmice'), items(
    T('\\textbf{Simple return}: $R_t = P_t/P_{t-1} - 1$; \\textbf{log return}: $r_t = \\ln(P_t/P_{t-1}) = \\ln(1 + R_t)$, so $R_t = e^{r_t} - 1$',
      '\\textbf{Randamentul simplu}: $R_t = P_t/P_{t-1} - 1$; \\textbf{randamentul logaritmic}: $r_t = \\ln(P_t/P_{t-1}) = \\ln(1 + R_t)$, deci $R_t = e^{r_t} - 1$'),
    T('Always $r_t < R_t$ (for $R_t \\neq 0$); $r_t \\approx R_t - R_t^2/2$, so the two are close for daily moves',
      'Întotdeauna $r_t < R_t$ (pentru $R_t \\neq 0$); $r_t \\approx R_t - R_t^2/2$, deci cele două sînt apropiate pentru variații zilnice'),
    (T('Over $k$ periods', 'Pe $k$ perioade'),
     [T('simple returns compound: $1 + R_t(k) = \\prod_{j=0}^{k-1}(1 + R_{t-j}) = P_t/P_{t-k}$', 'randamentele simple se compun: $1 + R_t(k) = \\prod_{j=0}^{k-1}(1 + R_{t-j}) = P_t/P_{t-k}$'),
      T('log returns add: $r_t(k) = \\sum_{j=0}^{k-1} r_{t-j} = \\ln(P_t/P_{t-k})$', 'randamentele logaritmice se adună: $r_t(k) = \\sum_{j=0}^{k-1} r_{t-j} = \\ln(P_t/P_{t-k})$')]),
    T('Portfolio with weights $w_i$ set at the start of the period: $R_p = \\sum_i w_i R_i$ exactly; $\\sum_i w_i r_i$ is only an approximation of $r_p$',
      'Portofoliu cu ponderile $w_i$ fixate la începutul perioadei: $R_p = \\sum_i w_i R_i$ exact; $\\sum_i w_i r_i$ este doar o aproximare a lui $r_p$')))

D.frame(T('What You Need for Today (3/4): Annualisation and Precision', 'Noțiuni necesare azi (3/4): anualizare și precizie'), items(
    T('$q$ = observations per year: about 250 for an exchange, 365 for Bitcoin, 12 for monthly data',
      '$q$ = observații pe an: aproximativ 250 pentru o bursă, 365 pentru Bitcoin, 12 pentru date lunare'),
    T('Annual mean $= q \\times$ mean; annual \\textbf{volatility} (standard deviation) $= \\sqrt{q} \\times$ standard deviation',
      'Media anuală $= q \\times$ media; \\textbf{volatilitatea} anuală (abaterea standard) $= \\sqrt{q} \\times$ abaterea standard'),
    T('\\textbf{CAGR} (compound annual growth rate) $= (P_T/P_0)^{1/Y} - 1$, with $Y$ years; equals $e^{\\text{annual mean log return}} - 1$',
      '\\textbf{CAGR} (compound annual growth rate, rata anuală compusă de creștere) $= (P_T/P_0)^{1/Y} - 1$, cu $Y$ ani; egală cu $e^{\\text{media anuală logaritmică}} - 1$'),
    T('\\textbf{Volatility drag}: annual mean of simple returns $\\approx$ annual mean of log returns $+\\, \\sigma^2/2$',
      '\\textbf{Volatility drag}: media anuală a randamentelor simple $\\approx$ media anuală a randamentelor logaritmice $+\\, \\sigma^2/2$'),
    (T('Precision of the annual mean: $\\text{SE} = \\sigma_{\\text{year}}/\\sqrt{Y}$; 95\\% CI (confidence interval): estimate $\\pm 1.96\\,\\text{SE}$',
       'Precizia mediei anuale: $\\text{SE} = \\sigma_{\\text{an}}/\\sqrt{Y}$; intervalul de încredere 95\\%: estimarea $\\pm 1.96\\,\\text{SE}$'),
     [T('test of $H_0$: mean $= 0$: $z = \\text{estimate}/\\text{SE}$, reject at 5\\% if $|z| > 1.96$', 'testul ipotezei $H_0$: media $= 0$: $z = \\text{estimarea}/\\text{SE}$, respingem la 5\\% dacă $|z| > 1.96$')])),
    'footnotesize')

D.frame(T('What You Need for Today (4/4): Performance Indicators', 'Noțiuni necesare azi (4/4): indicatori de performanță'), items(
    (T('\\textbf{Sharpe ratio} \\refSharpeB: $\\text{SR} = (\\mu - r_f)/\\sigma$; here $r_f = 0$; annual SR $= \\sqrt{q} \\times$ daily SR',
       '\\textbf{Raportul Sharpe} (Sharpe ratio) \\refSharpeB: $\\text{SR} = (\\mu - r_f)/\\sigma$; aici $r_f = 0$; SR anual $= \\sqrt{q} \\times$ SR zilnic'),
     [T('standard error for i.i.d. returns \\refLo: $\\text{SE} = \\sqrt{(1 + \\text{SR}^2/2)/n}$ per period, times $\\sqrt{q}$ for the annual SR',
        'eroarea standard pentru randamente i.i.d. \\refLo: $\\text{SE} = \\sqrt{(1 + \\text{SR}^2/2)/n}$ pe perioadă, înmulțită cu $\\sqrt{q}$ pentru SR anual'),
      T('\\textbf{bootstrap}: resample the days with replacement many times, recompute SR each time, take the 2.5\\% and 97.5\\% percentiles',
        '\\textbf{bootstrap}: reeșantionăm zilele cu întoarcere de multe ori, recalculăm SR de fiecare dată, luăm percentilele 2,5\\% și 97,5\\%')]),
    T('\\textbf{Sortino ratio} \\refSortino: $\\mu/\\sigma_D$, with downside deviation $\\sigma_D = \\sqrt{\\frac1n\\sum_t \\min(R_t, 0)^2}$',
      '\\textbf{Raportul Sortino} (Sortino ratio) \\refSortino: $\\mu/\\sigma_D$, cu abaterea negativă $\\sigma_D = \\sqrt{\\frac1n\\sum_t \\min(R_t, 0)^2}$'),
    T('\\textbf{Drawdown} $\\text{DD}_t = P_t/\\max_{s \\le t} P_s - 1$; \\textbf{MDD} (maximum drawdown) $= \\min_t \\text{DD}_t$; a loss $x$ needs a gain $x/(1-x)$',
      '\\textbf{Drawdown} $\\text{DD}_t = P_t/\\max_{s \\le t} P_s - 1$; \\textbf{MDD} (maximum drawdown, drawdown maxim) $= \\min_t \\text{DD}_t$; o pierdere $x$ cere un cîștig $x/(1-x)$'),
    T('\\textbf{Calmar ratio} $= \\text{CAGR}/|\\text{MDD}|$', '\\textbf{Raportul Calmar} (Calmar ratio) $= \\text{CAGR}/|\\text{MDD}|$')),
    'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Simple and Log Returns over Two Days', 'A1: randamente simple și logaritmice pe două zile'),
         items(T('A share closes at 50, 55 and 48 lei on three consecutive days.', 'Prețurile de închidere ale unei acțiuni în trei zile consecutive sînt 50, 55 și 48 de lei.'),
               T('1. Compute the two daily simple returns and the two daily log returns.', '1. Calculați cele două randamente simple zilnice și cele două randamente logaritmice zilnice.'),
               T('2. Compute the two-day simple return from the prices and by compounding.', '2. Calculați randamentul simplu pe două zile din prețuri și prin compunere.'),
               T('3. Check whether the sum of the simple returns and the sum of the log returns equal the two-day returns.',
                 '3. Verificați dacă suma randamentelor simple și suma randamentelor logaritmice sînt egale cu randamentele pe două zile.'),
               T('Report: four daily returns, two two-day returns, one sentence on which returns add up.',
                 'Raportați: patru randamente zilnice, două randamente pe două zile, o frază care precizează ce randamente se adună.')),
         items(T('1. $R_1 = 55/50 - 1 = @{a1.R1}\\%$, $R_2 = 48/55 - 1 = @{a1.R2}\\%$; $r_1 = \\ln(55/50) = @{a1.r1}\\%$, $r_2 = \\ln(48/55) = @{a1.r2}\\%$',
                 '1. $R_1 = 55/50 - 1 = @{a1.R1}\\%$, $R_2 = 48/55 - 1 = @{a1.R2}\\%$; $r_1 = \\ln(55/50) = @{a1.r1}\\%$, $r_2 = \\ln(48/55) = @{a1.r2}\\%$'),
               T('2. $48/50 - 1 = @{a1.R_total}\\%$; $(1 + R_1)(1 + R_2) - 1 = @{a1.R_compound}\\%$, the same', '2. $48/50 - 1 = @{a1.R_total}\\%$; $(1 + R_1)(1 + R_2) - 1 = @{a1.R_compound}\\%$, același rezultat'),
               T('3. $R_1 + R_2 = @{a1.R_sum}\\% \\neq @{a1.R_total}\\%$; $r_1 + r_2 = @{a1.r_sum}\\% = \\ln(48/50)$', '3. $R_1 + R_2 = @{a1.R_sum}\\% \\neq @{a1.R_total}\\%$; $r_1 + r_2 = @{a1.r_sum}\\% = \\ln(48/50)$'),
               T('Log returns add up over time; simple returns must be compounded.', 'Randamentele logaritmice se adună în timp; cele simple trebuie compuse.')),
         size='footnotesize')

D.proposed(T('A2: A Three-Day Path', 'A2: o traiectorie pe trei zile'),
           items(T('A share closes at 80, 100, 75 and 90 lei on four consecutive days. Model: A1.', 'Prețurile de închidere ale unei acțiuni în patru zile consecutive sînt 80, 100, 75 și 90 de lei. Model: A1.'),
                 T('1. Compute the three daily simple returns and the three daily log returns.', '1. Calculați cele trei randamente simple zilnice și cele trei randamente logaritmice zilnice.'),
                 T('2. Compute the three-day simple return in two ways: from the first and last price, and by compounding.',
                   '2. Calculați randamentul simplu pe trei zile în două moduri: din primul și ultimul preț și prin compunere.'),
                 T('3. Explain why the sum of the simple returns overstates the three-day return.', '3. Explicați de ce suma randamentelor simple supraestimează randamentul pe trei zile.'),
                 T('Report: six daily returns, the three-day simple and log returns, one sentence of explanation.',
                   'Raportați: șase randamente zilnice, randamentul simplu și randamentul logaritmic pe trei zile, o frază de explicație.')),
           items(T('1. $R$: $@{a2.R1}\\%$, $@{a2.R2}\\%$, $@{a2.R3}\\%$; $r$: $@{a2.r1}\\%$, $@{a2.r2}\\%$, $@{a2.r3}\\%$',
                   '1. $R$: $@{a2.R1}\\%$, $@{a2.R2}\\%$, $@{a2.R3}\\%$; $r$: $@{a2.r1}\\%$, $@{a2.r2}\\%$, $@{a2.r3}\\%$'),
                 T('2. $90/80 - 1 = @{a2.R_total}\\%$ and $\\prod(1 + R_t) - 1 = @{a2.R_compound}\\%$; log: $\\sum r_t = \\ln(90/80) = @{a2.r_total}\\%$',
                   '2. $90/80 - 1 = @{a2.R_total}\\%$ și $\\prod(1 + R_t) - 1 = @{a2.R_compound}\\%$; logaritmic: $\\sum r_t = \\ln(90/80) = @{a2.r_total}\\%$'),
                 T('3. $\\sum R_t = @{a2.R_sum}\\%$: a 25\\% loss after a 25\\% gain falls on a larger base, so adding the percentages ignores compounding',
                   '3. $\\sum R_t = @{a2.R_sum}\\%$: pierderea de 25\\% după un cîștig de 25\\% se aplică unei baze mai mari, deci adunarea procentelor ignoră compunerea')),
           size='footnotesize')

D.solved(T('A3: A Dividend and a Split', 'A3: un dividend și un split'),
         items(T('Closes on days 0--6: 30, 31, 30.5, 29.2, 29.8, 10.1, 10.3 lei. On day 3 the share pays a dividend of 1.5 lei; on day 5 it splits 3-for-1.',
                 'Închideri în zilele 0--6: 30; 31; 30,5; 29,2; 29,8; 10,1; 10,3 lei. În ziua 3 se plătește un dividend de 1,5 lei; în ziua 5 are loc un split 3-la-1.'),
               T('1. Compute the adjustment factor of each event.', '1. Calculați factorul de ajustare al fiecărui eveniment.'),
               T('2. Compute the adjusted prices of days 0--6.', '2. Calculați prețurile ajustate din zilele 0--6.'),
               T('3. Compare the raw and the adjusted return of day 5.', '3. Comparați randamentul neajustat și cel ajustat din ziua 5.'),
               T('Report: two factors, seven adjusted prices, two returns of day 5.', 'Raportați: doi factori, șapte prețuri ajustate, două randamente pentru ziua 5.')),
         items(T('1. Dividend: $A = 1 - 1.5/30.5 = @{a3.div}$; split: $A = 1/3 = @{a3.split}$', '1. Dividend: $A = 1 - 1.5/30.5 = @{a3.div}$; split: $A = 1/3 = @{a3.split}$'),
               T('2. Days 0--2 are multiplied by both factors, $@{a3.both}$: $@{a3.adj0}$, $@{a3.adj1}$, $@{a3.adj2}$; days 3--4 by $1/3$: $@{a3.adj3}$, $@{a3.adj4}$; days 5--6 unchanged: $@{a3.adj5}$, $@{a3.adj6}$',
                 '2. Zilele 0--2 se înmulțesc cu ambii factori, $@{a3.both}$: $@{a3.adj0}$, $@{a3.adj1}$, $@{a3.adj2}$; zilele 3--4 cu $1/3$: $@{a3.adj3}$, $@{a3.adj4}$; zilele 5--6 neschimbate: $@{a3.adj5}$, $@{a3.adj6}$'),
               T('3. Raw: $10.1/29.8 - 1 = @{a3.raw5}\\%$, a fall that never happened; adjusted: $@{a3.adj5}/@{a3.adj4} - 1 = @{a3.adj5r}\\%$',
                 '3. Neajustat: $10.1/29.8 - 1 = @{a3.raw5}\\%$, o scădere care nu a avut loc; ajustat: $@{a3.adj5}/@{a3.adj4} - 1 = @{a3.adj5r}\\%$')),
         size='footnotesize')

D.proposed(T('A4: A Dividend and a Consolidation', 'A4: un dividend și o consolidare'),
           items(T('Closes on days 0--5: 2.05, 2.10, 1.95, 1.98, 19.90, 20.40 lei. On day 2 the share pays a dividend of 0.18 lei; on day 4, ten old shares are consolidated into one. Model: A3.',
                   'Închideri în zilele 0--5: 2,05; 2,10; 1,95; 1,98; 19,90; 20,40 lei. În ziua 2 se plătește un dividend de 0,18 lei; în ziua 4, zece acțiuni vechi se consolidează într-una. Model: A3.'),
                 T('1. Compute the adjustment factor of each event.', '1. Calculați factorul de ajustare al fiecărui eveniment.'),
                 T('2. Compute the adjusted prices of days 0--5.', '2. Calculați prețurile ajustate din zilele 0--5.'),
                 T('3. Compute the raw log return of day 4 and say what a volatility estimate would look like if it were kept.',
                   '3. Calculați randamentul logaritmic neajustat din ziua 4 și explicați ce efect ar avea asupra estimării volatilității dacă ar fi păstrat.'),
                 T('Report: two factors, six adjusted prices, one return and one sentence.', 'Raportați: doi factori, șase prețuri ajustate, un randament și o frază.')),
           items(T('1. Dividend: $A = 1 - 0.18/2.10 = @{a4.div}$; consolidation: $A = 10$', '1. Dividend: $A = 1 - 0.18/2.10 = @{a4.div}$; consolidare: $A = 10$'),
                 T('2. Days 0--1 times $@{a4.both}$: $@{a4.adj0}$, $@{a4.adj1}$; days 2--3 times 10: $@{a4.adj2}$, $@{a4.adj3}$; days 4--5: $@{a4.adj4}$, $@{a4.adj5}$',
                   '2. Zilele 0--1 înmulțite cu $@{a4.both}$: $@{a4.adj0}$, $@{a4.adj1}$; zilele 2--3 cu 10: $@{a4.adj2}$, $@{a4.adj3}$; zilele 4--5: $@{a4.adj4}$, $@{a4.adj5}$'),
                 T('3. $\\ln(19.90/1.98) = +@{a4.raw}\\%$ in one day, close to $\\ln 10$: one such day dominates the sample variance; the same happens with TLV in August 2022 (Lecture 1)',
                   '3. $\\ln(19.90/1.98) = +@{a4.raw}\\%$ într-o zi, aproape de $\\ln 10$: o asemenea zi domină varianța de selecție; la fel se întîmplă cu TLV în august 2022 (Cursul 1)')),
           size='footnotesize')

D.solved(T('A5: From Daily Moments to Annual Indicators', 'A5: de la momentele zilnice la indicatori anuali'),
         items(T('A stock has daily log returns with mean $0.06\\%$ and standard deviation $1.5\\%$; it trades 250 days a year; you have 10 years of data.',
                 'O acțiune are randamente logaritmice zilnice cu media $0.06\\%$ și abaterea standard $1.5\\%$; se tranzacționează 250 de zile pe an; aveți 10 ani de date.'),
               T('1. Compute the annual mean log return and the annual volatility.', '1. Calculați media anuală a randamentelor logaritmice și volatilitatea anuală.'),
               T('2. Compute the CAGR and the annual mean of simple returns implied by the volatility drag.', '2. Calculați CAGR și media anuală a randamentelor simple implicată de volatility drag.'),
               T('3. Compute the standard error and the 95\\% CI of the annual mean log return.', '3. Calculați eroarea standard și intervalul de încredere 95\\% pentru media anuală a randamentelor logaritmice.'),
               T('Report: five numbers and one sentence on the width of the interval.', 'Raportați: cinci valori și o frază despre lățimea intervalului.')),
         items(T('1. $250 \\times 0.06\\% = @{a5.mu_log}\\%$; $\\sqrt{250} \\times 1.5\\% = @{a5.vol}\\%$', '1. $250 \\times 0.06\\% = @{a5.mu_log}\\%$; $\\sqrt{250} \\times 1.5\\% = @{a5.vol}\\%$'),
               T('2. CAGR $= e^{0.15} - 1 = @{a5.cagr}\\%$; $\\sigma^2/2 = @{a5.var}/2 = @{a5.drag}\\%$, so the arithmetic mean is $@{a5.mu_log}\\% + @{a5.drag}\\% \\approx @{a5.mu_arith}\\%$',
                 '2. CAGR $= e^{0.15} - 1 = @{a5.cagr}\\%$; $\\sigma^2/2 = @{a5.var}/2 = @{a5.drag}\\%$, deci media aritmetică este $@{a5.mu_log}\\% + @{a5.drag}\\% \\approx @{a5.mu_arith}\\%$'),
               T('3. $\\text{SE} = @{a5.vol}\\%/\\sqrt{10} = @{a5.se}\\%$; CI $[@{a5.ci_lo}\\%, @{a5.ci_hi}\\%]$', '3. $\\text{SE} = @{a5.vol}\\%/\\sqrt{10} = @{a5.se}\\%$; intervalul de încredere $[@{a5.ci_lo}\\%; @{a5.ci_hi}\\%]$'),
               T('Ten years of data barely exclude zero: mean returns are imprecise.', 'Zece ani de date abia exclud zero: randamentele medii sînt imprecise.')),
         size='footnotesize')

D.solved(T('A6: Indicators of a Short Price Path', 'A6: indicatorii unei traiectorii scurte de prețuri'),
         items(T('A fund\'s month-end values: 100, 104, 101, 107, 99, 103, 110 (six months, $q = 12$, $r_f = 0$).',
                 'Valorile unui fond la sfîrșit de lună: 100, 104, 101, 107, 99, 103, 110 (șase luni, $q = 12$, $r_f = 0$).'),
               T('1. Compute the six monthly simple returns, their mean and standard deviation.', '1. Calculați cele șase randamente simple lunare, media și abaterea lor standard.'),
               T('2. Compute the annualised Sharpe ratio and Sortino ratio.', '2. Calculați rapoartele Sharpe și Sortino anualizate.'),
               T('3. Compute the MDD, the CAGR and the Calmar ratio.', '3. Calculați MDD, CAGR și raportul Calmar.'),
               T('Report: a table of returns, five indicators, one sentence on why Sortino exceeds Sharpe.',
                 'Raportați: un tabel cu randamente, cinci indicatori, o frază care explică de ce raportul Sortino depășește raportul Sharpe.')),
         items(T('1. Returns (\\%): @{a6.Rs}; mean $@{a6.mean}\\%$, s.d. $@{a6.sd}\\%$', '1. Randamente (\\%): @{a6.Rs}; media $@{a6.mean}\\%$, abaterea std. $@{a6.sd}\\%$'),
               T('2. SR $= \\sqrt{12} \\times @{a6.mean}/@{a6.sd} = @{a6.sharpe}$; $\\sigma_D = @{a6.dsd}\\%$, Sortino $= \\sqrt{12} \\times @{a6.mean}/@{a6.dsd} = @{a6.sortino}$',
                 '2. SR $= \\sqrt{12} \\times @{a6.mean}/@{a6.sd} = @{a6.sharpe}$; $\\sigma_D = @{a6.dsd}\\%$, Sortino $= \\sqrt{12} \\times @{a6.mean}/@{a6.dsd} = @{a6.sortino}$'),
               T('3. MDD $= 99/107 - 1 = @{a6.mdd}\\%$; CAGR $= 1.10^{2} - 1 = @{a6.cagr}\\%$; Calmar $= @{a6.calmar}$',
                 '3. MDD $= 99/107 - 1 = @{a6.mdd}\\%$; CAGR $= 1.10^{2} - 1 = @{a6.cagr}\\%$; Calmar $= @{a6.calmar}$'),
               T('Only two of six months are losses, so $\\sigma_D < \\sigma$; with half a year of data all these numbers are very imprecise.',
                 'Doar două din șase luni sînt pierderi, deci $\\sigma_D < \\sigma$; cu o jumătate de an de date toate aceste cifre sînt foarte imprecise.')),
         size='footnotesize')

D.proposed(T('A7: Indicators of Another Fund', 'A7: indicatorii altui fond'),
           items(T('Month-end values: 200, 190, 205, 215, 180, 196, 210, 222 (seven months, $q = 12$, $r_f = 0$). Model: A6.',
                   'Valori la sfîrșit de lună: 200, 190, 205, 215, 180, 196, 210, 222 (șapte luni, $q = 12$, $r_f = 0$). Model: A6.'),
                 T('1. Compute the monthly simple returns, their mean and standard deviation.', '1. Calculați randamentele simple lunare, media și abaterea lor standard.'),
                 T('2. Compute the annualised Sharpe and Sortino ratios.', '2. Calculați rapoartele Sharpe și Sortino anualizate.'),
                 T('3. Compute the MDD, the CAGR and the Calmar ratio.', '3. Calculați MDD, CAGR și raportul Calmar.'),
                 T('4. Compare with the fund of A6: which indicators rank the two funds differently?', '4. Comparați cu fondul din A6: ce indicatori clasează diferit cele două fonduri?'),
                 T('Report: five indicators and one sentence comparing the two funds.', 'Raportați: cinci indicatori și o frază care compară cele două fonduri.')),
           items(T('1. Returns (\\%): @{a7.Rs}; mean $@{a7.mean}\\%$, s.d. $@{a7.sd}\\%$', '1. Randamente (\\%): @{a7.Rs}; media $@{a7.mean}\\%$, abaterea std. $@{a7.sd}\\%$'),
                 T('2. SR $= @{a7.sharpe}$; $\\sigma_D = @{a7.dsd}\\%$, Sortino $= @{a7.sortino}$', '2. SR $= @{a7.sharpe}$; $\\sigma_D = @{a7.dsd}\\%$, Sortino $= @{a7.sortino}$'),
                 T('3. MDD $= 180/215 - 1 = @{a7.mdd}\\%$; CAGR $= 1.11^{12/7} - 1 = @{a7.cagr}\\%$; Calmar $= @{a7.calmar}$',
                   '3. MDD $= 180/215 - 1 = @{a7.mdd}\\%$; CAGR $= 1.11^{12/7} - 1 = @{a7.cagr}\\%$; Calmar $= @{a7.calmar}$'),
                 T('4. A7 has the higher mean monthly return but a larger volatility and a deeper drawdown: A6 wins on Sharpe, Sortino, CAGR and Calmar',
                   '4. A7 are media lunară mai mare, dar volatilitate mai mare și un drawdown mai adînc: A6 este superior după Sharpe, Sortino, CAGR și Calmar')),
           size='footnotesize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Precision and Interpretation', 'Partea B: date reale, precizie și interpretare')

D.task(T('B1: Can We Trust a EUR/RON Series? [Solved]', 'B1: ne putem baza pe o serie EUR/RON? [Rezolvat]'),
       T('do the EUR/RON series from EODHD and the BNR reference rate describe the same exchange rate?',
         'descriu seria EUR/RON de la EODHD și cursul de referință BNR același curs de schimb?'),
       T('the two daily EUR/RON series on their common days, 2005--2026', 'cele două serii zilnice EUR/RON în zilele lor comune, 2005--2026'),
       [T('Compute the daily log returns of both series on the common days.', 'Calculați randamentele logaritmice zilnice ale ambelor serii în zilele comune.'),
        T('Count the days on which the two returns differ by more than one percentage point.', 'Numărați zilele în care cele două randamente diferă cu mai mult de un punct procentual.'),
        T('Compute the annualised volatility of each series, and of the EODHD series without those days.', 'Calculați volatilitatea anualizată a fiecărei serii și a seriei EODHD fără acele zile.'),
        T('Interpretation: which series is suitable for measuring the currency risk of a EUR loan? Justify the choice.',
          'Interpretare: ce serie este potrivită pentru a măsura riscul valutar al unui credit în euro? Justificați alegerea.')],
       T('the number of days, three volatilities and one sentence of interpretation', 'numărul de zile, trei volatilități și o frază de interpretare'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch1_sem_b1', h='0.48') + items(
    T('@{b1.n} common days; on @{b1.nbig} of them (@{b1.share}\\%) the gap exceeds one percentage point; most in @{b1.wy} (@{b1.wyn} days)',
      '@{b1.n} zile comune; în @{b1.nbig} dintre ele (@{b1.share}\\%) diferența depășește un punct procentual; cele mai multe în @{b1.wy} (@{b1.wyn} zile)'),
    T('Largest gaps: @{b1.t1.date} (EODHD $@{b1.t1.e}\\%$, BNR $@{b1.t1.b}\\%$), @{b1.t2.date} (EODHD $@{b1.t2.e}\\%$, BNR $@{b1.t2.b}\\%$)',
      'Cele mai mari diferențe: @{b1.t1.date} (EODHD $@{b1.t1.e}\\%$, BNR $@{b1.t1.b}\\%$), @{b1.t2.date} (EODHD $@{b1.t2.e}\\%$, BNR $@{b1.t2.b}\\%$)'),
    T('Volatility: BNR @{b1.volb}\\%, EODHD @{b1.vole}\\%, EODHD without the bad days @{b1.volc}\\%', 'Volatilitatea: BNR @{b1.volb}\\%, EODHD @{b1.vole}\\%, EODHD fără zilele eronate @{b1.volc}\\%'),
    T('Interpretation: the BNR rate, the primary source of the number; the EODHD series would multiply the measured currency risk by @{b1.ratio}',
      'Interpretare: cursul BNR, sursa primară a cifrei; seria EODHD ar multiplica de @{b1.ratio} ori riscul valutar măsurat')) + qlsem(), 'footnotesize')

D.task(T('B2: Is the Mean Return of the BET Positive? [Solved]', 'B2: este pozitiv randamentul mediu al BET? [Rezolvat]'),
       T('is the mean daily return of the BET significantly different from zero over 2015--2026?', 'este randamentul zilnic mediu al BET semnificativ diferit de zero în 2015--2026?'),
       T('BET closes, 2015--2026, daily log returns in \\%', 'închiderile BET, 2015--2026, randamente logaritmice zilnice în \\%'),
       [T('Compute the mean, standard deviation, skewness, excess kurtosis, minimum (with its date) and maximum of the daily log returns.',
          'Calculați media, abaterea standard, asimetria, excesul de aplatizare, minimul (cu data lui) și maximul randamentelor logaritmice zilnice.'),
        T('Annualise the mean and the standard deviation with the actual number of observations per year.', 'Anualizați media și abaterea standard cu numărul real de observații pe an.'),
        T('Compute the standard error of the annual mean, its 95\\% CI and the $z$ statistic of $H_0$: mean $= 0$.',
          'Calculați eroarea standard a mediei anuale, intervalul de încredere 95\\% și statistica $z$ pentru $H_0$: media $= 0$.'),
        T('Interpretation: does the histogram look like the Normal distribution?',
          'Interpretare: seamănă histograma cu distribuția Normală?'),
        T('Interpretation: what does the confidence interval tell an investor?',
          'Interpretare: ce informație îi oferă intervalul de încredere unui investitor?')],
       T('a table of statistics, the CI, $z$ and $p$, two sentences of interpretation', 'un tabel de statistici, intervalul de încredere, $z$ și $p$, două fraze de interpretare'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Solved]', 'B2: rezolvare [Rezolvat]'), cols(fig('ch1_sem_b2', h='0.55', w='1.0'), items(
    T('$n = @{b2.n}$ days, @{b2.ppy} a year; mean $@{b2.mean}\\%$, s.d. $@{b2.sd}\\%$', '$n = @{b2.n}$ zile, @{b2.ppy} pe an; media $@{b2.mean}\\%$, abaterea std. $@{b2.sd}\\%$'),
    T('Skewness $@{b2.skew}$, excess kurtosis $@{b2.exkurt}$; worst day $@{b2.min}\\%$ on @{b2.mindate}, best $@{b2.max}\\%$',
      'Asimetria $@{b2.skew}$, excesul de aplatizare $@{b2.exkurt}$; cea mai slabă zi $@{b2.min}\\%$ pe @{b2.mindate}, cea mai bună $@{b2.max}\\%$'),
    T('Annual: mean @{b2.annmean}\\%, volatility @{b2.annvol}\\%; SE $= @{b2.annvol}/\\sqrt{@{b2.years}} = @{b2.se}\\%$',
      'Anual: media @{b2.annmean}\\%, volatilitatea @{b2.annvol}\\%; SE $= @{b2.annvol}/\\sqrt{@{b2.years}} = @{b2.se}\\%$'),
    T('95\\% CI [@{b2.lo}\\%, @{b2.hi}\\%]; $z = @{b2.t}$, $p = @{b2.p}$: we reject a zero mean at 5\\%', 'Intervalul de încredere 95\\%: [@{b2.lo}\\%; @{b2.hi}\\%]; $z = @{b2.t}$, $p = @{b2.p}$: respingem ipoteza mediei zero la 5\\%'),
    T('Interpretation: a taller centre and heavier tails than the Normal distribution; the mean is positive, but anywhere between @{b2.lo}\\% and @{b2.hi}\\% a year',
      'Interpretare: centru mai înalt și cozi mai groase decît distribuția Normală; media este pozitivă, dar se poate afla oriunde între @{b2.lo}\\% și @{b2.hi}\\% pe an')),
    wl='0.50', wr='0.48') + qlsem(), 'footnotesize')

D.task(T('B3: Four More Markets [Proposed]', 'B3: încă patru piețe [Propus]'),
       T('which of the S\\&P 500, DAX, Bitcoin and Banca Transilvania have a mean return significantly above zero?',
         'care dintre S\\&P 500, DAX, Bitcoin și Banca Transilvania au un randament mediu semnificativ peste zero?'),
       T('daily log returns, 2015--2026, each on its own calendar; model: B2', 'randamente logaritmice zilnice, 2015--2026, fiecare serie pe propriul calendar; model: B2'),
       [T('Repeat steps 1--3 of B2 for each of the four series.', 'Repetați pașii 1--3 din B2 pentru fiecare dintre cele patru serii.'),
        T('Put the annual mean, the volatility, the CI and $p$ of all five series (with the BET) in one table.', 'Reuniți într-un tabel media anuală, volatilitatea, intervalul de încredere și valoarea $p$ pentru toate cele cinci serii (inclusiv BET).'),
        T('Interpretation: why is the interval of Bitcoin so much wider than that of the S\\&P 500, although both cover the same years?',
          'Interpretare: de ce intervalul Bitcoin este mult mai larg decît cel al S\\&P 500, deși ambele acoperă aceiași ani?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Proposed]', 'B3: rezolvare [Propus]'), table(
    'lrrrrr', T('Series', 'Seria') + ' & ' + T('Mean \\%', 'Media \\%') + ' & Vol. \\% & SE \\% & ' + T('95\\% CI', 'Interval 95\\%') + ' & $p$',
    [f'{lab} & $@{{{k}.annmean}}$ & $@{{{k}.annvol}}$ & $@{{{k}.se}}$ & ' + T('[$@{%s.lo}$, $@{%s.hi}$]' % (k, k), '[$@{%s.lo}$; $@{%s.hi}$]' % (k, k)) + f' & $@{{{k}.p}}$'
     for lab, k in [('BET', 'b2'), ('S\\&P 500', 'b3.sp500'), ('DAX', 'b3.dax'), ('Bitcoin', 'b3.btc'), ('Banca Transilvania', 'b3.tlv')]],
    size='footnotesize') + items(
    T('Significant at 5\\%: every series whose interval excludes zero; the DAX interval contains zero', 'Semnificative la 5\\%: toate seriile al căror interval exclude zero; intervalul DAX conține zero'),
    T('Interpretation: SE $= \\sigma_{\\text{year}}/\\sqrt{Y}$ with the same $Y$; Bitcoin\'s volatility is @{b3.ratio} times that of the S\\&P 500, and so is the width of its interval',
      'Interpretare: SE $= \\sigma_{\\text{an}}/\\sqrt{Y}$ cu același $Y$; volatilitatea Bitcoin este de @{b3.ratio} ori mai mare decît cea a S\\&P 500, iar intervalul este de tot atîtea ori mai larg')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B4: How Precise Is a Sharpe Ratio? [Solved]', 'B4: cît de precis este raportul Sharpe? [Rezolvat]'),
       T('how precisely is the Sharpe ratio of the S\\&P 500 estimated from 2015--2026 data?', 'cît de precis se estimează raportul Sharpe al S\\&P 500 din datele 2015--2026?'),
       T('S\\&P 500 closes, daily simple returns, $r_f = 0$', 'închiderile S\\&P 500, randamente simple zilnice, $r_f = 0$'),
       [T('Compute the daily mean, standard deviation and Sharpe ratio, and annualise the Sharpe ratio.', 'Calculați media zilnică, abaterea standard și raportul Sharpe zilnic, apoi anualizați raportul Sharpe.'),
        T('Compute the standard error of \\refLo{} and the 95\\% CI.', 'Calculați eroarea standard după \\refLo{} și intervalul de încredere 95\\%.'),
        T('Bootstrap the Sharpe ratio with 2000 resamples of the days and take the 2.5\\% and 97.5\\% percentiles.', 'Aplicați bootstrap raportului Sharpe, cu 2000 de reeșantionări ale zilelor, și calculați percentilele 2,5\\% și 97,5\\%.'),
        T('Interpretation: could the true Sharpe ratio of the S\\&P 500 be below 0.25?', 'Interpretare: ar putea raportul Sharpe adevărat al S\\&P 500 să fie sub 0,25?')],
       T('the Sharpe ratio, two intervals and one sentence', 'raportul Sharpe, două intervale și o frază'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Solved]', 'B4: rezolvare [Rezolvat]'), cols(fig('ch1_sem_b4', h='0.55', w='1.0'), items(
    T('$n = @{b4.n}$, $q = @{b4.q}$; daily mean $@{b4.mean_d}\\%$, s.d. $@{b4.sd_d}\\%$, daily SR $@{b4.sr_d}$',
      '$n = @{b4.n}$, $q = @{b4.q}$; media zilnică $@{b4.mean_d}\\%$, abaterea std. $@{b4.sd_d}\\%$, SR zilnic $@{b4.sr_d}$'),
    T('Annual SR $= @{b4.sr_d} \\times @{b4.sqrtq} = @{b4.sr}$', 'SR anual $= @{b4.sr_d} \\times @{b4.sqrtq} = @{b4.sr}$'),
    T('Lo: SE $= \\sqrt{(1 + @{b4.sr_d}^2/2)/@{b4.n}} \\times @{b4.sqrtq} = @{b4.se}$; CI [$@{b4.lo}$, $@{b4.hi}$]',
      'Lo: SE $= \\sqrt{(1 + @{b4.sr_d}^2/2)/@{b4.n}} \\times @{b4.sqrtq} = @{b4.se}$; intervalul de încredere [$@{b4.lo}$; $@{b4.hi}$]'),
    T('Bootstrap: [$@{b4.bs_lo}$, $@{b4.bs_hi}$], very close', 'Bootstrap: [$@{b4.bs_lo}$; $@{b4.bs_hi}$], foarte apropiat'),
    T('Interpretation: yes; every value from $@{b4.bs_lo}$ to $@{b4.bs_hi}$ is compatible with the data', 'Interpretare: da; orice valoare de la $@{b4.bs_lo}$ la $@{b4.bs_hi}$ este compatibilă cu datele')),
    wl='0.50', wr='0.48') + qlsem(), 'footnotesize')

D.task(T('B5: Is Banca Transilvania Better than the BET? [Proposed]', 'B5: este Banca Transilvania mai bună decît BET? [Propus]'),
       T('is the Sharpe ratio of TLV significantly higher than that of the BET index?', 'este raportul Sharpe al TLV semnificativ mai mare decît cel al indicelui BET?'),
       T('BET close and TLV adjusted close, 2015--2026, joined on common days first; model: B4', 'închiderea BET și închiderea ajustată TLV, 2015--2026, aliniate întîi pe zilele comune; model: B4'),
       [T('Join the two price series on their common days, then compute daily simple returns.', 'Aliniați cele două serii de prețuri pe zilele lor comune, apoi calculați randamentele simple zilnice.'),
        T('Compute both annualised Sharpe ratios with their Lo standard errors.', 'Calculați cele două rapoarte Sharpe anualizate, cu erorile standard Lo.'),
        T('Bootstrap the difference of the two Sharpe ratios, resampling the same days for both series.', 'Aplicați bootstrap diferenței dintre cele două rapoarte Sharpe, reeșantionînd aceleași zile pentru ambele serii.'),
        T('Interpretation: why must the days be resampled in pairs?', 'Interpretare: de ce trebuie reeșantionate zilele în perechi?')],
       T('two Sharpe ratios, the difference with its interval, one sentence', 'cele două rapoarte Sharpe, diferența cu intervalul ei, o frază'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Proposed]', 'B5: rezolvare [Propus]'), items(
    T('@{b5.n} common days; BET SR $@{b5.bet.sr}$ (SE $@{b5.bet.se}$), TLV SR $@{b5.tlv.sr}$ (SE $@{b5.tlv.se}$)',
      '@{b5.n} zile comune; SR BET $@{b5.bet.sr}$ (SE $@{b5.bet.se}$), SR TLV $@{b5.tlv.sr}$ (SE $@{b5.tlv.se}$)'),
    T('Difference TLV $-$ BET $= @{b5.diff}$; paired bootstrap 95\\% interval [$@{b5.lo}$, $@{b5.hi}$], contains zero',
      'Diferența TLV $-$ BET $= @{b5.diff}$; intervalul bootstrap 95\\% pe perechi, [$@{b5.lo}$; $@{b5.hi}$], conține zero'),
    T('No significant difference', 'Nicio diferență semnificativă'),
    T('Interpretation: the two return series are correlated ($\\rho = @{b5.corr}$); resampling days in pairs keeps this correlation, which makes the difference more precise than two separate intervals suggest',
      'Interpretare: cele două serii de randamente sînt corelate ($\\rho = @{b5.corr}$); reeșantionarea zilelor în perechi păstrează corelația, ceea ce face diferența mai precisă decît sugerează două intervale separate')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B6: The Deepest Fall of the BET [Solved]', 'B6: cea mai adîncă cădere a BET [Rezolvat]'),
       T('what were the depth and the duration of the worst fall of the BET since its launch?', 'care au fost adîncimea și durata celei mai mari căderi a BET de la lansare?'),
       T('official BET closes since 19 September 1997; for the Calmar ratio, 2015--2026', 'închiderile oficiale BET din 19 septembrie 1997; pentru raportul Calmar, 2015--2026'),
       [T('Compute the drawdown series and the maximum drawdown with the dates of the peak, the trough and the recovery.', 'Calculați seria drawdown-urilor și drawdown-ul maxim, cu datele vîrfului, minimului și revenirii.'),
        T('Compute the gain needed to recover from the trough and the share of days spent more than 20\\% below the peak.', 'Calculați cîștigul necesar pentru revenirea de la minim și ponderea zilelor petrecute cu peste 20\\% sub vîrf.'),
        T('Compute the CAGR, the MDD and the Calmar ratio over 2015--2026.', 'Calculați CAGR, MDD și raportul Calmar pentru 2015--2026.'),
        T('Interpretation: what does the drawdown add to what the volatility already tells us?', 'Interpretare: ce adaugă drawdown-ul la ceea ce ne spune deja volatilitatea?')],
       T('the MDD with three dates, two percentages, three indicators, one sentence', 'MDD cu trei date, două procente, trei indicatori, o frază'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Solved]', 'B6: rezolvare [Rezolvat]'), fig('ch1_sem_b6', h='0.48') + items(
    T('MDD $@{b6.mdd}\\%$: from @{b6.peak_level} points on @{b6.peak} to @{b6.trough_level} on @{b6.trough}; the old peak was regained on @{b6.recovery}, after @{b6.years_to_recover} years',
      'MDD $@{b6.mdd}\\%$: de la @{b6.peak_level} puncte pe @{b6.peak} la @{b6.trough_level} pe @{b6.trough}; vechiul vîrf a fost atins din nou pe @{b6.recovery}, după @{b6.years_to_recover} ani'),
    T('From the trough, a gain of $@{b6.need}\\%$ was needed; @{b6.below}\\% of all days were more than 20\\% below the peak',
      'De la minim era nevoie de un cîștig de $@{b6.need}\\%$; @{b6.below}\\% din toate zilele au fost cu peste 20\\% sub vîrf'),
    T('2015--2026: CAGR @{b6.cagr}\\%, MDD $@{b6.mdd_period}\\%$ (@{b6.period_peak} to @{b6.period_trough}), Calmar $= @{b6.cagr}/@{b6.absmdd} = @{b6.calmar}$',
      '2015--2026: CAGR @{b6.cagr}\\%, MDD $@{b6.mdd_period}\\%$ (@{b6.period_peak} -- @{b6.period_trough}), Calmar $= @{b6.cagr}/@{b6.absmdd} = @{b6.calmar}$'),
    T('Interpretation: volatility ignores the order of returns; the drawdown shows how long an investor stays under water',
      'Interpretare: volatilitatea ignoră ordinea randamentelor; drawdown-ul arată cît timp rămîne investitorul sub vîrful anterior (underwater)')) + qlsem(), 'footnotesize')

D.task(T('B7: Bitcoin, OMV Petrom and the Volatility Drag [Proposed]', 'B7: Bitcoin, OMV Petrom și volatility drag [Propus]'),
       T('what do the drawdowns and the volatility drag show for Bitcoin and SNP?', 'ce arată drawdown-urile și volatility drag pentru Bitcoin și SNP?'),
       T('Bitcoin close and SNP adjusted close, 2015--2026; model: B6', 'închiderea Bitcoin și închiderea ajustată SNP, 2015--2026; model: B6'),
       [T('Compute the MDD of each series with the dates of the peak, the trough and the recovery.', 'Calculați MDD pentru fiecare serie, cu datele vîrfului, minimului și revenirii.'),
        T('Compute the CAGR and the Calmar ratio of each series.', 'Calculați CAGR și raportul Calmar pentru fiecare serie.'),
        T('Compute the arithmetic annual mean of simple returns, the annual mean of log returns and their difference.', 'Calculați media anuală aritmetică a randamentelor simple, media anuală a randamentelor logaritmice și diferența lor.'),
        T('Compare the difference with $\\sigma^2/2$, using the annual volatility of simple returns.', 'Comparați diferența cu $\\sigma^2/2$, folosind volatilitatea anuală a randamentelor simple.'),
        T('Interpretation: for which asset does the drag matter for a long-term investor?', 'Interpretare: pentru care dintre cele două active este volatility drag important pentru un investitor pe termen lung?')],
       T('a table with MDD, dates, CAGR, Calmar, the two means, the drag and $\\sigma^2/2$, one sentence', 'un tabel cu MDD, date, CAGR, Calmar, cele două medii, volatility drag și $\\sigma^2/2$, o frază'), size='footnotesize', nb='B7')

D.frame(T('B7: Solution [Proposed]', 'B7: rezolvare [Propus]'), table(
    'lrrrrrrr', T('Series', 'Seria') + ' & MDD \\% & CAGR \\% & Calmar & ' + T('Arith. \\%', 'Aritm. \\%') + ' & Log \\% & Drag \\% & $\\sigma^2/2$ \\%',
    [f'{lab} & $@{{b7.{k}.mdd}}$ & $@{{b7.{k}.cagr}}$ & $@{{b7.{k}.calmar}}$ & $@{{b7.{k}.mean_arith}}$ & $@{{b7.{k}.mean_log}}$ & $@{{b7.{k}.drag}}$ & $@{{b7.{k}.half_var}}$'
     for lab, k in [('Bitcoin', 'btc'), ('OMV Petrom', 'snp')]], size='footnotesize') + items(
    T('Bitcoin: @{b7.btc.peak} to @{b7.btc.trough}, recovered on @{b7.btc.recovery}; SNP: @{b7.snp.peak} to @{b7.snp.trough}, recovered on @{b7.snp.recovery}',
      'Bitcoin: @{b7.btc.peak} -- @{b7.btc.trough}, cu revenire pe @{b7.btc.recovery}; SNP: @{b7.snp.peak} -- @{b7.snp.trough}, cu revenire pe @{b7.snp.recovery}'),
    T('The drag and $\\sigma^2/2$ differ by at most @{b7.maxgap} percentage points', 'Volatility drag și $\\sigma^2/2$ diferă cu cel mult @{b7.maxgap} puncte procentuale'),
    T('Interpretation: for Bitcoin the drag removes @{b7.share}\\% of the arithmetic mean; for SNP it is small',
      'Interpretare: pentru Bitcoin, volatility drag reprezintă @{b7.share}\\% din media aritmetică; pentru SNP este mic')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns generat de AI')

D.task(T('C1: Do Sharpe Ratios Persist? [Proposed]', 'C1: persistă rapoartele Sharpe? [Propus]'),
       T('does a BVB stock with a high Sharpe ratio in 2015--2020 also have a high one in 2021--2026?', 'are o acțiune BVB cu un raport Sharpe mare în 2015--2020 un raport Sharpe mare și în 2021--2026?'),
       T('adjusted closes of TLV, SNP, BRD, TGN, SNG, SNN, 2015--2026; models: B4, B6', 'închiderile ajustate TLV, SNP, BRD, TGN, SNG, SNN, 2015--2026; modele: B4, B6'),
       [T('Compute the annualised Sharpe ratio and its standard error for each stock in each half.', 'Calculați raportul Sharpe anualizat și eroarea lui standard pentru fiecare acțiune în fiecare jumătate.'),
        T('Compute the Spearman rank correlation between the Sharpe ratios of the two halves.', 'Calculați corelația rangurilor Spearman între rapoartele Sharpe din cele două jumătăți.'),
        T('Propose one way to make the answer more reliable (more stocks, other indicators, other windows).', 'Propuneți o modalitate de a face răspunsul mai robust (mai multe acțiuni, alți indicatori, alte ferestre).'),
        T('Interpretation: what can six stocks tell us about persistence?', 'Interpretare: ce ne pot spune șase acțiuni despre persistență?')],
       T('a table, the rank correlation with its $p$-value, a short plan for a project', 'un tabel, corelația rangurilor cu valoarea $p$, un scurt plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrr', T('Stock', 'Acțiunea') + ' & SR 2015--2020 & SE & SR 2021--2026 & SE', c1rows, size='footnotesize') + items(
    T('Spearman rank correlation $@{c1.rho}$ ($p = @{c1.p}$, $n = @{c1.n}$): no evidence of persistence; if anything, a reversal',
      'Corelația rangurilor Spearman $@{c1.rho}$ ($p = @{c1.p}$, $n = @{c1.n}$): nicio dovadă de persistență; mai degrabă o inversare'),
    T('Each half-sample Sharpe ratio has a standard error of about $@{c1.se}$: rankings are mostly noise', 'Fiecare raport Sharpe estimat pe o jumătate de eșantion are o eroare standard de circa $@{c1.se}$: clasamentele sînt în mare parte zgomot'),
    T('Project directions: all liquid BVB stocks, rolling three-year windows, Sortino and Calmar, comparison with \\refHLZ',
      'Direcții de proiect: toate acțiunile lichide BVB, ferestre mobile de trei ani, Sortino și Calmar, comparație cu \\refHLZ')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificarea unui răspuns generat de AI [Propus]'), items(
    T('A student asked an AI assistant: ``Compare the risk and the performance of Bitcoin and the BET in 2015--2026.\'\' The answer:',
      'Un student a întrebat un asistent AI: „Compară riscul și performanța Bitcoin și ale BET în 2015--2026.” Răspunsul:'),
    T('\\aiprompt{(a) Bitcoin\'s daily volatility is @{c2.btc_sd}\\%, so its annual volatility is @{c2.btc_sd}\\% x sqrt(252) = @{c2.btc_vol252}\\%.}',
      '\\aiprompt{(a) Volatilitatea zilnică a Bitcoin este @{c2.btc_sd}\\%, deci volatilitatea anuală este @{c2.btc_sd}\\% x sqrt(252) = @{c2.btc_vol252}\\%.}'),
    T('\\aiprompt{(b) Bitcoin\'s Sharpe ratio (@{c2.btc_sr}) is higher than the BET\'s (@{c2.bet_sr}), which proves that Bitcoin is the better investment.}',
      '\\aiprompt{(b) Raportul Sharpe al Bitcoin (@{c2.btc_sr}) este mai mare decît al BET (@{c2.bet_sr}), ceea ce dovedește că Bitcoin este investiția mai bună.}'),
    T('\\aiprompt{(c) Bitcoin\'s maximum drawdown is its worst daily return, @{c2.btc_worst_day}\\%.}',
      '\\aiprompt{(c) Drawdown-ul maxim al Bitcoin este cel mai slab randament zilnic, @{c2.btc_worst_day}\\%.}'),
    T('\\aiprompt{(d) For a 50/50 portfolio, the log return is exactly the average of the two log returns.}',
      '\\aiprompt{(d) Pentru un portofoliu 50/50, randamentul logaritmic este exact media celor două randamente logaritmice.}'),
    T('\\aiprompt{(e) For BVB stocks such as TLV I used the close column, which is the official price.}',
      '\\aiprompt{(e) Pentru acțiunile BVB, ca TLV, am folosit coloana close, care este prețul oficial.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, unde este posibil, dați cifra corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of five verdicts with one line of justification each.', '2. Raportați: o listă de cinci verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong factor: Bitcoin trades about @{c2.q} days a year; $@{c2.btc_sd}\\% \\times \\sqrt{365} = @{c2.btc_vol365}\\%$, not @{c2.btc_vol252}\\%',
      '(a) Factor greșit: Bitcoin se tranzacționează aproximativ @{c2.q} de zile pe an; $@{c2.btc_sd}\\% \\times \\sqrt{365} = @{c2.btc_vol365}\\%$, nu @{c2.btc_vol252}\\%'),
    T('(b) Not proved: SE $\\approx @{c2.btc_se}$ for each Sharpe ratio, so the gap of the two estimates is well within the noise; and past Sharpe ratios do not guarantee future ones',
      '(b) Nu este dovedit: SE $\\approx @{c2.btc_se}$ pentru fiecare raport Sharpe, deci diferența dintre cele două estimări se încadrează în zgomot; iar rapoartele Sharpe din trecut nu le garantează pe cele viitoare'),
    T('(c) Wrong definition: MDD is the worst peak-to-trough fall, $@{c2.btc_mdd}\\%$; the worst day was $@{c2.btc_worst_day}\\%$ (log return)',
      '(c) Definiție greșită: MDD este cea mai mare cădere de la vîrf la minim, $@{c2.btc_mdd}\\%$; cea mai slabă zi a fost $@{c2.btc_worst_day}\\%$ (randament logaritmic)'),
    T('(d) Wrong: $R_p = \\sum_i w_i R_i$ is exact; $r_p = \\ln(\\sum_i w_i e^{r_i})$ differs from $\\sum_i w_i r_i$',
      '(d) Greșit: $R_p = \\sum_i w_i R_i$ este exact; $r_p = \\ln(\\sum_i w_i e^{r_i})$ diferă de $\\sum_i w_i r_i$'),
    T('(e) Wrong column: TLV\'s close jumps by $+@{c2.tlv_close_max}\\%$ (log) on @{c2.tlvdate}, a share consolidation; volatility from the close @{c2.tlv_vol_close}\\% vs @{c2.tlv_vol_adj}\\% from the adjusted close',
      '(e) Coloană greșită: prețul de închidere TLV crește brusc cu $+@{c2.tlv_close_max}\\%$ (logaritmic) pe @{c2.tlvdate}, din cauza unei consolidări a acțiunilor; volatilitatea calculată din close este @{c2.tlv_vol_close}\\%, față de @{c2.tlv_vol_adj}\\% din prețul ajustat')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHIDERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Check the data before computing: source, adjusted prices, a second series', 'Verificați datele înainte de calcule: sursa, prețurile ajustate, o a doua serie'),
    T('Log returns add over time; simple returns add across assets', 'Randamentele logaritmice se adună în timp; cele simple se adună între active'),
    T('Mean returns and Sharpe ratios come with wide intervals: report them', 'Randamentele medii și rapoartele Sharpe au intervale de încredere largi: raportați-le'),
    T('The drawdown and the volatility drag show what the volatility alone hides', 'Drawdown-ul și volatility drag arată ceea ce volatilitatea singură nu arată'),
    T('An AI answer is a draft: check every factor, definition and column', 'Un răspuns generat de AI este o ciornă: verificați fiecare factor, definiție și coloană')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 1 develops each topic of today: data quality, returns, annualisation, descriptive statistics and performance indicators',
      'Cursul 1 dezvoltă fiecare temă de azi: calitatea datelor, randamente, anualizare, statistici descriptive și indicatori de performanță'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: more stocks, rolling windows, other indicators', 'C1 poate deveni un proiect de echipă: mai multe acțiuni, ferestre mobile, alți indicatori'),
    T('Reading: \\refFHH, Ch.~11; exercises with solutions: \\refFHHex', 'Lectură: \\refFHH, cap.~11; exerciții rezolvate: \\refFHHex')))

D.references([
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\textit{Statistics of Financial Markets: Exercises and Solutions}}, 2nd ed. Springer.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    r"Harvey, C.R., Liu, Y., Zhu, H. (2016). \href{https://doi.org/10.1093/rfs/hhv059}{\ldots and the cross-section of expected returns}. \textit{Review of Financial Studies}, 29(1), 5--68.",
    r"Lo, A.W. (2002). \href{https://doi.org/10.2469/faj.v58.n4.2453}{The statistics of Sharpe ratios}. \textit{Financial Analysts Journal}, 58(4), 36--52.",
    r"Sharpe, W.F. (1994). \href{https://doi.org/10.3905/jpm.1994.409501}{The Sharpe ratio}. \textit{Journal of Portfolio Management}, 21(1), 49--58.",
    r"Sortino, F.A., van der Meer, R. (1991). \href{https://doi.org/10.3905/jpm.1991.409343}{Downside risk}. \textit{Journal of Portfolio Management}, 17(4), 27--31.",
])

if __name__ == '__main__':
    D.write(V)
