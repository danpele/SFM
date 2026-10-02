r"""
build_seminar0.py -- Seminarul 0 (Introducere: prețuri, randamente, volatilitate, drawdown), EN + RO
=====================================================================================================
Seminarul are loc ÎNAINTEA cursului 0: este autonom (o scurtă introducere teoretică la început). Format A/B/C:
  A1-A3 [Rezolvat], A4 [Propus]       calcule pe hîrtie
  B1-B2 [Rezolvat], B3-B4 [Propus]    aceleași idei pe date reale
  C2 [Propus]                         critica unui răspuns generat de AI
Studenții nu predau nimic; temele sînt exercițiu. Rezolvările problemelor propuse apar doar în versiunea
profesorului (*_solutions.tex, \solutionstrue).
Cifrele vin din Quantlets/Ch_00/sem0_results.json (Quantlets/Ch_00/seminar0.py, fișier al profesorului).
Ieșire:
  EN/Seminars/seminar0_introduction.tex (+ _solutions.tex)
  RO/Seminarii/seminar0_introducere_ro.tex (+ _solutions.tex)
Rulare:  python3 Quantlets/Ch_00/seminar0.py && python3 latex/build_seminar0.py && python3 latex/sfm_build.py compile 0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import MARK, ROOT, Deck, Values, items, cols, table, enum   # noqa: E402

VAL = json.load(open(os.path.join(ROOT, 'Quantlets', 'Ch_00', 'sem0_results.json')))
MONTHS = {'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December'],
          'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
                 'octombrie', 'noiembrie', 'decembrie']}


def dt(s):
    y, m, d = s.split('-')
    return f'⟦{int(d)} {MONTHS["en"][int(m) - 1]} {y}||{int(d)} {MONTHS["ro"][int(m) - 1]} {y}⟧'


class LangValues(Values):
    """Values whose entries may hold ⟦EN||RO⟧ text (dates), resolved for the language being rendered."""
    lang = 'en'

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return MARK.sub(lambda m: m.group(1) if self.lang == 'en' else m.group(2), v) if isinstance(v, str) else v


class LangDeck(Deck):
    def head(self, lang):
        V.lang = lang
        return super().head(lang)


V = LangValues()
for k, x in VAL.items():
    if isinstance(x, str):
        V.raw(k, dt(x) if len(x) == 10 and x[4] == '-' else x)
    elif k.endswith(('n_prices', '_n', 'n_big', 'n_gap', 'n_m', 'n_b')):
        V.int(k, x)
    elif k.endswith(('_quote', '_prev', '_next', '_bnr')) and k.startswith('b2'):
        V.put(k, x, 4)
    elif k.endswith(('_P0', '_P1', '_P2', '_PT')):
        V.put(k, x, 2)
    elif k.endswith(('mean_d', 'sd_d', '_R1', '_r1', '_R2', '_r2')) or k.startswith(('a1_', 'a4_R', 'a4_r', 'a4_sum', 'a4_tot')):
        V.put(k, x, 2 if not k.endswith('mean_d') else 3)
    else:
        V.put(k, x, 1)

import datetime as _dt
_years = (_dt.date.fromisoformat(VAL['b1_last_date']) - _dt.date.fromisoformat(VAL['b1_first_date'])).days / 365.25
V.put('b1_years', _years, 1)
V.put('b1_val', 100 * VAL['b1_PT'] / VAL['b1_P0'], 0)
V.put('b4_mult', VAL['b4_PT'] / VAL['b4_P0'], 0)
V.put('b4_ratio', (365 / 252) ** 0.5, 2)
V.put('b4_mdd_abs', -VAL['b4_mdd'], 0)
V.put('b2_move_abs', VAL['b2_move_in'], 0)
V.put('c2_ratio_ai0', VAL['c2_ratio_ai'], 1)
V.put('c2_vol_ratio', VAL['c2_btc_ok_vol'] / VAL['c2_sp500_ok_vol'], 1)

SOLVED, PROP = '⟦[Solved]||[Rezolvat]⟧', '⟦[Proposed]||[Propus]⟧'
TASKW, SOLW = '⟦Task||Cerință⟧', '⟦Solution||Rezolvare⟧'
QL = '\\sfmquantlet{Ch_00}{SFM_ch0_seminar}'
NB_EN, NB_RO = '\\href{\\nb}{seminar notebook}', '\\href{\\nb}{notebook-ul seminarului}'


def fig(name, h='0.5\\textheight'):
    return f'\\begin{{center}}\n\\includegraphics[width=0.94\\textwidth,height={h},keepaspectratio]{{{name}.pdf}}\n\\end{{center}}\n\\vspace{{-2mm}}\n'


D = LangDeck(0, 'seminar')

# ===============================================================================================================
D.section('Route and Primer', 'Traseu și noțiuni de bază')
# ===============================================================================================================
D.frame('⟦Today\'s Questions and Route||Întrebările de azi și traseul⟧', items(
    '⟦First seminar of the course, before the Chapter 0 lecture: everything you need is on the next slides||Primul seminar al cursului, înaintea cursului din Capitolul 0: tot ce vă trebuie este pe slide-urile următoare⟧',
    ('⟦\\textbf{Questions of the seminar}||\\textbf{Întrebările seminarului}⟧',
     ['⟦how do we turn a price series into returns?||cum transformăm o serie de prețuri în randamente?⟧',
      '⟦how risky was an asset, per year?||cît de riscant a fost un activ, pe an?⟧',
      '⟦what was the worst loss from a peak?||care a fost cea mai mare pierdere de la un vîrf?⟧',
      '⟦how do we know that the prices are right?||de unde știm că prețurile sînt corecte?⟧']),
    ('⟦\\textbf{Part I}: setup and calculations on paper (A1--A3), then the same ideas on real data (B1, B2)||\\textbf{Partea I}: pregătirea și calcule pe hîrtie (A1--A3), apoi aceleași idei pe date reale (B1, B2)⟧', []),
    ('⟦\\textbf{Part II}: your turn (A4, B3, B4) and the critique of an AI answer (C2)||\\textbf{Partea a II-a}: rîndul vostru (A4, B3, B4) și critica unui răspuns generat de AI (C2)⟧', []),
    '⟦Nothing is handed in: the proposed exercises are practice for the exam and the project||Nu se predă nimic: exercițiile propuse sînt exercițiu pentru examen și proiect⟧',
    f'⟦Open the {NB_EN} in Google Colab||Deschideți {NB_RO} în Google Colab⟧').replace('\\begin{itemize}\n    \\end{itemize}', '').replace(
    '    \\begin{itemize}\n    \\end{itemize}\n', ''))

MAP = [
    'A1 & ⟦simple and log returns on paper||randamente simple și logaritmice pe hîrtie⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A2 & ⟦annualising a daily mean and volatility||anualizarea mediei și a volatilității zilnice⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A3 & ⟦drawdown of a short price path||drawdown-ul unei serii scurte de prețuri⟧ & ⟦Solved||Rezolvat⟧ & --',
    'B1 & ⟦S\\&P 500, 2015--2026: returns, risk, drawdown||S\\&P 500, 2015--2026: randamente, risc, drawdown⟧ & ⟦Solved||Rezolvat⟧ & --',
    'B2 & ⟦data check: one suspicious EUR/RON quote||verificarea datelor: o cotație EUR/RON suspectă⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A4 & ⟦the same calculations on new numbers||aceleași calcule pe alte cifre⟧ & ⟦Proposed||Propus⟧ & A1--A3',
    'B3 & ⟦BET-TR, 2015--2026||BET-TR, 2015--2026⟧ & ⟦Proposed||Propus⟧ & B1',
    'B4 & ⟦Bitcoin: 365 or 252 days a year?||Bitcoin: 365 sau 252 de zile pe an?⟧ & ⟦Proposed||Propus⟧ & A2, B1',
    'C2 & ⟦find the errors in an AI answer||găsiți greșelile dintr-un răspuns generat de AI⟧ & ⟦Proposed||Propus⟧ & A2, B1',
]
D.frame('⟦Exercise Map||Harta exercițiilor⟧', table(
    'l>{\\raggedright\\arraybackslash}p{6.4cm}ll', '\\textbf{⟦Exercise||Exercițiu⟧} & \\textbf{⟦Question||Întrebare⟧} & \\textbf{⟦Type||Tip⟧} & \\textbf{Model}',
    MAP, size='footnotesize') + items(
    '⟦\\textbf{[Solved]}: full solution in the slides and the notebook, a model to copy; \\textbf{[Proposed]}: you solve it, the solution is discussed in class||\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, rezolvarea se discută la seminar⟧',
    '⟦Part A: on paper; Part B: real data, each ending with an interpretation question; Part C: critique of an AI answer||Partea A: pe hîrtie; Partea B: date reale, fiecare încheiată cu o întrebare de interpretare; Partea C: critica unui răspuns generat de AI⟧'),
    size='footnotesize')

D.frame('⟦Primer (1/2): Prices and Returns||Noțiuni de bază (1/2): prețuri și randamente⟧', items(
    '⟦$P_t$: the closing price on day $t$; for ETFs and shares, the \\textbf{adjusted close} (corrected for dividends and splits)||$P_t$: prețul de închidere din ziua $t$; pentru ETF-uri și acțiuni, \\textbf{prețul ajustat} (corectat pentru dividende și splituri)⟧',
    ('⟦\\textbf{Simple return}: $R_t = P_t / P_{t-1} - 1$||\\textbf{Randament simplu}: $R_t = P_t / P_{t-1} - 1$⟧',
     ['⟦the percentage gain of the day; over several days, simple returns \\textbf{compound}: $1 + R_{0\\to T} = \\prod_t (1 + R_t)$||cîștigul procentual al zilei; pe mai multe zile, randamentele simple \\textbf{se compun}: $1 + R_{0\\to T} = \\prod_t (1 + R_t)$⟧']),
    ('⟦\\textbf{Log return}: $r_t = \\ln(P_t / P_{t-1}) = \\ln(1 + R_t)$||\\textbf{Randament logaritmic}: $r_t = \\ln(P_t / P_{t-1}) = \\ln(1 + R_t)$⟧',
     ['⟦over several days, log returns \\textbf{add}: $\\ln(P_T / P_0) = \\sum_t r_t$||pe mai multe zile, randamentele logaritmice \\textbf{se adună}: $\\ln(P_T / P_0) = \\sum_t r_t$⟧',
      '⟦close to $R_t$ for small moves: $R_t = 1\\%$ gives $r_t = 0.995\\%$||apropiat de $R_t$ pentru variații mici: $R_t = 1\\%$ dă $r_t = 0.995\\%$⟧']),
    ('⟦\\textbf{Cumulative return} from day 0 to day $T$: $P_T / P_0 - 1$||\\textbf{Randamentul cumulat} de la ziua 0 la ziua $T$: $P_T / P_0 - 1$⟧',
     ['⟦the value of 100 invested on day 0 is $100 \\, P_t / P_0$||valoarea a 100 de unități investite în ziua 0 este $100 \\, P_t / P_0$⟧'])))

D.frame('⟦Primer (2/2): Volatility and Drawdown||Noțiuni de bază (2/2): volatilitate și drawdown⟧', items(
    ('⟦From $n$ daily returns: \\textbf{mean} $\\bar r$ and \\textbf{volatility} $s$ (the sample standard deviation)||Din $n$ randamente zilnice: \\textbf{media} $\\bar r$ și \\textbf{volatilitatea} $s$ (abaterea standard de selecție)⟧',
     ['$s = \\sqrt{\\frac{1}{n-1} \\sum_{t=1}^n (r_t - \\bar r)^2}$']),
    ('⟦\\textbf{Annualisation} with $q$ trading days per year||\\textbf{Anualizarea} cu $q$ zile de tranzacționare pe an⟧',
     ['⟦mean $\\times q$, volatility $\\times \\sqrt{q}$: variances of independent days add up, standard deviations do not||media $\\times q$, volatilitatea $\\times \\sqrt{q}$: varianțele zilelor independente se adună, abaterile standard nu⟧',
      '⟦$q \\approx 252$ for exchanges (weekdays without holidays), $q = 365$ for Bitcoin (every day)||$q \\approx 252$ pentru burse (zile lucrătoare fără sărbători), $q = 365$ pentru Bitcoin (în fiecare zi)⟧']),
    ('⟦\\textbf{Drawdown}: $DD_t = P_t / M_t - 1$, with the running peak $M_t = \\max_{s \\le t} P_s$||\\textbf{Drawdown}: $DD_t = P_t / M_t - 1$, cu vîrful curent $M_t = \\max_{s \\le t} P_s$⟧',
     ['⟦\\textbf{maximum drawdown} (MDD): the most negative $DD_t$, the worst loss from a previous peak||\\textbf{drawdown-ul maxim} (MDD): cel mai negativ $DD_t$, cea mai mare pierdere de la un vîrf anterior⟧']),
    ('⟦\\textbf{Data check}: before computing, look for prices that cannot be true||\\textbf{Verificarea datelor}: înainte de calcule, căutați prețuri care nu pot fi adevărate⟧',
     ['⟦a \\textbf{bad tick}: an isolated wrong quote, a jump that reverses the next day||un \\textbf{bad tick}: o cotație greșită izolată, un salt care se inversează a doua zi⟧'])))

DATA = [
    'S\\&P 500 & ⟦index, close||indice, închidere⟧ & USD & ⟦US trading days||zilele de tranzacționare din SUA⟧ & B1, C2',
    'BET-TR & ⟦index with dividends, close||indice cu dividende, închidere⟧ & RON & ⟦Romanian trading days||zilele de tranzacționare din România⟧ & B3',
    'Bitcoin & BTC/USD, ⟦close||închidere⟧ & USD & ⟦every day, 24:00 UTC||în fiecare zi, 24:00 UTC⟧ & B4, C2',
    '⟦EUR/RON, official||EUR/RON, oficial⟧ & ⟦BNR reference rate||cursul de referință BNR⟧ & ⟦RON per EUR||lei pe euro⟧ & ⟦Romanian working days||zilele lucrătoare din România⟧ & B2',
    '⟦EUR/RON, market||EUR/RON, piață⟧ & ⟦series from EODHD, close||seria de la EODHD, închidere⟧ & ⟦RON per EUR||lei pe euro⟧ & ⟦weekdays||zilele lucrătoare⟧ & B2',
]
D.frame('⟦Data Used Today||Datele de azi⟧', table(
    '>{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{1.6cm}>{\\raggedright\\arraybackslash}p{3.4cm}l',
    '\\textbf{⟦Series||Serie⟧} & \\textbf{⟦Price used||Prețul folosit⟧} & \\textbf{⟦Unit||Unitate⟧} & \\textbf{⟦Calendar||Calendar⟧} & \\textbf{⟦Exercise||Exercițiu⟧}',
    DATA, size='footnotesize') + items(
    '⟦Daily data from EODHD (EOD Historical Data), saved in the course repository; sample from 2 January 2015 to 18 September 2026||Date zilnice de la EODHD (EOD Historical Data), salvate în repository-ul cursului; eșantion de la 2 ianuarie 2015 la 18 septembrie 2026⟧',
    '⟦BNR: the National Bank of Romania; its reference rate is fixed once a day, around 13:00 Bucharest time||BNR: Banca Națională a României; cursul de referință se stabilește o dată pe zi, în jurul orei 13:00⟧',
    '⟦BET-TR: the BET index of the BVB (Bucharest Stock Exchange) with dividends reinvested||BET-TR: indicele BET al BVB cu dividendele reinvestite⟧'), size='footnotesize')

D.task('⟦Setup (1/2): Open the Notebook and Save a Copy||Pregătire (1/2): deschideți notebook-ul și salvați o copie⟧',
       '⟦can you run Python on the course data in the browser?||puteți rula Python pe datele cursului, în browser?⟧',
       '⟦Google Colab: Google\'s free notebook service; Python runs on Google\'s servers, a Google account is enough||Google Colab: serviciul gratuit de notebook-uri al Google; Python rulează pe serverele Google, ajunge un cont Google⟧',
       [f'⟦Open the {NB_EN} in Colab.||Deschideți {NB_RO} în Colab.⟧',
        '⟦Choose \\emph{File $\\to$ Save a copy in Drive}, or run the Drive-save cell under the banner, so that your changes are kept.||Alegeți \\emph{File $\\to$ Save a copy in Drive} sau rulați celula de salvare în Drive de sub banner, ca să vă păstrați modificările.⟧',
        '⟦Run the cells of the section \\emph{Setup} in order with Shift + Enter; they import the libraries and define the data loader.||Rulați în ordine celulele secțiunii \\emph{Setup}, cu Shift + Enter; ele importă bibliotecile și definesc funcțiile de citire a datelor.⟧',
        '⟦If a cell fails, choose \\emph{Runtime $\\to$ Restart session} and run again from the top.||Dacă o celulă dă eroare, alegeți \\emph{Runtime $\\to$ Restart session} și rulați din nou de la început.⟧'],
       '⟦nothing; you are ready when the check on the next slide prints the expected numbers||nimic; sînteți gata cînd verificarea de pe slide-ul următor afișează cifrele așteptate⟧')

D.frame('⟦Setup (2/2): Load the Data||Pregătire (2/2): citiți datele⟧', cols(
    r"""\begin{lstlisting}
# load_close(name, start): one daily series
# from the course data, with the right
# price column and calendar
p = load_close('sp500', start='2015-01-02')
print(len(p), p.index[0].date(),
      p.index[-1].date())
print(p.head(3).round(2))
\end{lstlisting}""" + '\n' + items('⟦\\texttt{SERIES} in the notebook lists every available name||\\texttt{SERIES} din notebook listează toate numele disponibile⟧'),
    items(('⟦\\textbf{Expected output}||\\textbf{Rezultatul așteptat}⟧',
           ['⟦@{b1_n_prices} prices, from @{b1_first_date} to @{b1_last_date}||@{b1_n_prices} de prețuri, de la @{b1_first_date} la @{b1_last_date}⟧',
            '⟦first three closes: @{b1_P0}, @{b1_P1}, @{b1_P2}||primele trei închideri: @{b1_P0}, @{b1_P1}, @{b1_P2}⟧']),
          '⟦If your count or first value differs, check the name and the start date before going on||Dacă numărul sau prima valoare diferă, verificați numele și data de început înainte de a continua⟧'),
    '0.55', '0.42'), opts='fragile')

# ===============================================================================================================
D.section('Part A: On Paper', 'Partea A: pe hîrtie')
# ===============================================================================================================
D.task(f'A1 {SOLVED}: ⟦Simple and Log Returns --- Task||Randamente simple și logaritmice --- cerință⟧',
       '⟦can two daily returns simply be added to get the two-day return?||se pot aduna pur și simplu două randamente zilnice ca să obținem randamentul pe două zile?⟧',
       '⟦prices $P_0 = 100$, $P_1 = 110$, $P_2 = 99$||prețurile $P_0 = 100$, $P_1 = 110$, $P_2 = 99$⟧',
       ['⟦Compute the simple returns $R_1, R_2$ and the log returns $r_1, r_2$.||Calculați randamentele simple $R_1, R_2$ și randamentele logaritmice $r_1, r_2$.⟧',
        '⟦Add the two simple returns, and then add the two log returns.||Adunați cele două randamente simple, apoi cele două randamente logaritmice.⟧',
        '⟦Compute the two-day returns $P_2/P_0 - 1$ and $\\ln(P_2/P_0)$, and compare them with the sums of step 2.||Calculați randamentele pe două zile $P_2/P_0 - 1$ și $\\ln(P_2/P_0)$ și comparați-le cu sumele de la pasul 2.⟧'],
       '⟦a table with the columns day, price, simple return, log return, and one sentence on which returns add up||un tabel cu coloanele zi, preț, randament simplu, randament logaritmic și o propoziție despre ce randamente se adună⟧',
       nb='A1')

D.frame(f'A1 {SOLVED}: ⟦Solution||Rezolvare⟧', cols(
    table('lrrr', '$t$ & $P_t$ & $R_t$ & $r_t$',
          ['0 & 100 & -- & --', '1 & 110 & $+$@{a1_R1}\\% & $+$@{a1_r1}\\%', '2 & 99 & @{a1_R2}\\% & @{a1_r2}\\%',
           '\\midrule ⟦sum||sumă⟧ & & @{a1_sumR}\\% & @{a1_sumr}\\%', '$0 \\to 2$ & & @{a1_totR}\\% & @{a1_totr}\\%'],
          size='footnotesize'),
    items(('⟦\\textbf{Step by step}||\\textbf{Pas cu pas}⟧',
           ['$R_1 = 110/100 - 1$, $r_1 = \\ln 1.1$',
            '$R_2 = 99/110 - 1$, $r_2 = \\ln 0.9$']),
          ('⟦\\textbf{Why}||\\textbf{De ce}⟧',
           ['⟦log returns telescope: $(\\ln P_1 - \\ln P_0) + (\\ln P_2 - \\ln P_1) = \\ln(P_2/P_0)$||randamentele logaritmice se reduc telescopic: $(\\ln P_1 - \\ln P_0) + (\\ln P_2 - \\ln P_1) = \\ln(P_2/P_0)$⟧',
            '⟦simple returns compound: $1.1 \\times 0.9 - 1 = -1\\%$||randamentele simple se compun: $1.1 \\times 0.9 - 1 = -1\\%$⟧']),
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦adding simple returns reports 0\\% for a path that lost 1\\%: add log returns, compound simple returns||suma randamentelor simple arată 0\\% pentru o serie care a pierdut 1\\%: adunăm randamentele logaritmice, compunem randamentele simple⟧'])),
    '0.42', '0.55') + '\n' + QL)

D.task(f'A2 {SOLVED}: ⟦Annualisation --- Task||Anualizarea --- cerință⟧',
       '⟦what do a daily mean and a daily volatility mean per year?||ce înseamnă pe an o medie zilnică și o volatilitate zilnică?⟧',
       '⟦a stock index with daily mean log return $0.05\\%$ and daily volatility $1.2\\%$; Bitcoin with daily volatility $3\\%$||un indice bursier cu randamentul logaritmic mediu zilnic de $0.05\\%$ și volatilitatea zilnică de $1.2\\%$; Bitcoin cu volatilitatea zilnică de $3\\%$⟧',
       ['⟦Annualise the mean and the volatility of the index with $q = 252$.||Anualizați media și volatilitatea indicelui, cu $q = 252$.⟧',
        '⟦Annualise the volatility of Bitcoin with $q = 365$, and then with $q = 252$.||Anualizați volatilitatea Bitcoin cu $q = 365$, apoi cu $q = 252$.⟧',
        '⟦Explain which value of $q$ is right for Bitcoin.||Explicați ce valoare a lui $q$ este corectă pentru Bitcoin.⟧'],
       '⟦three annual numbers and one sentence on the choice of $q$||trei cifre anuale și o propoziție despre alegerea lui $q$⟧', nb='A2')

D.frame(f'A2 {SOLVED}: ⟦Solution||Rezolvare⟧', items(
    ('⟦\\textbf{Index}, $q = 252$||\\textbf{Indicele}, $q = 252$⟧',
     ['⟦mean: $252 \\times 0.05\\% = $ @{a2_mean}\\% a year||media: $252 \\times 0.05\\% = $ @{a2_mean}\\% pe an⟧',
      '⟦volatility: $\\sqrt{252} \\times 1.2\\% = $ @{a2_sqrt252} $\\times 1.2\\% = $ @{a2_vol}\\% a year||volatilitatea: $\\sqrt{252} \\times 1.2\\% = $ @{a2_sqrt252} $\\times 1.2\\% = $ @{a2_vol}\\% pe an⟧']),
    ('⟦\\textbf{Bitcoin}||\\textbf{Bitcoin}⟧',
     ['⟦$q = 365$: $\\sqrt{365} \\times 3\\% = $ @{a2_btc365}\\% a year||$q = 365$: $\\sqrt{365} \\times 3\\% = $ @{a2_btc365}\\% pe an⟧',
      '⟦$q = 252$: $\\sqrt{252} \\times 3\\% = $ @{a2_btc252}\\% a year, too low||$q = 252$: $\\sqrt{252} \\times 3\\% = $ @{a2_btc252}\\% pe an, prea puțin⟧']),
    ('⟦\\textbf{Why the square root}||\\textbf{De ce rădăcina pătrată}⟧',
     ['⟦for independent days, the variance of a year is $q$ times the daily variance, so the standard deviation is $\\sqrt{q}$ times the daily one||pentru zile independente, varianța pe un an este de $q$ ori varianța zilnică, deci abaterea standard este de $\\sqrt{q}$ ori cea zilnică⟧']),
    ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
     ['⟦$q$ must be the number of observations per year of the series: Bitcoin trades every day, so $q = 365$||$q$ trebuie să fie numărul de observații pe an al seriei: Bitcoin se tranzacționează în fiecare zi, deci $q = 365$⟧'])) + '\n' + QL)

D.task(f'A3 {SOLVED}: ⟦Drawdown on Paper --- Task||Drawdown pe hîrtie --- cerință⟧',
       '⟦what was the worst loss from a previous peak?||care a fost cea mai mare pierdere față de un vîrf anterior?⟧',
       '⟦prices on days 0--5: 100, 120, 90, 110, 130, 104||prețurile din zilele 0--5: 100, 120, 90, 110, 130, 104⟧',
       ['⟦Compute the running peak $M_t$ for each day.||Calculați vîrful curent $M_t$ pentru fiecare zi.⟧',
        '⟦Compute the drawdown $DD_t = P_t / M_t - 1$ for each day.||Calculați drawdown-ul $DD_t = P_t / M_t - 1$ pentru fiecare zi.⟧',
        '⟦Find the maximum drawdown and the days of its peak and of its trough.||Găsiți drawdown-ul maxim și zilele vîrfului și ale minimului corespunzător.⟧'],
       '⟦a table with $t$, $P_t$, $M_t$, $DD_t$, the maximum drawdown, and whether the price recovered by day 5||un tabel cu $t$, $P_t$, $M_t$, $DD_t$, drawdown-ul maxim și dacă prețul și-a revenit pînă în ziua 5⟧',
       nb='A3')

D.frame(f'A3 {SOLVED}: ⟦Solution||Rezolvare⟧', cols(
    table('rrrr', '$t$ & $P_t$ & $M_t$ & $DD_t$',
          ['0 & 100 & 100 & @{a3_dd0}\\%', '1 & 120 & 120 & @{a3_dd1}\\%', '2 & 90 & 120 & @{a3_dd2}\\%',
           '3 & 110 & 120 & @{a3_dd3}\\%', '4 & 130 & 130 & @{a3_dd4}\\%', '5 & 104 & 130 & @{a3_dd5}\\%'],
          size='footnotesize'),
    items(('⟦\\textbf{Step by step}||\\textbf{Pas cu pas}⟧',
           ['⟦day 2: $90/120 - 1 = -25\\%$; day 3: $110/120 - 1$||ziua 2: $90/120 - 1 = -25\\%$; ziua 3: $110/120 - 1$⟧',
            '⟦day 4: a new peak, so $M_4 = 130$ and $DD_4 = 0$||ziua 4: un vîrf nou, deci $M_4 = 130$ și $DD_4 = 0$⟧',
            '⟦day 5: $104/130 - 1 = -20\\%$, measured from the new peak||ziua 5: $104/130 - 1 = -20\\%$, măsurat de la noul vîrf⟧']),
          '⟦\\textbf{Maximum drawdown}: @{a3_mdd}\\%, from the peak of day 1 (120) to the trough of day 2 (90)||\\textbf{Drawdown-ul maxim}: @{a3_mdd}\\%, de la vîrful din ziua 1 (120) la minimul din ziua 2 (90)⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦the price recovered on day 4, but on day 5 the investor is again 20\\% below the peak||prețul și-a revenit în ziua 4, dar în ziua 5 investitorul este din nou cu 20\\% sub vîrf⟧'])),
    '0.38', '0.58') + '\n' + QL)

# ===============================================================================================================
D.section('Part B: Real Data', 'Partea B: date reale')
# ===============================================================================================================
D.task(f'B1 {SOLVED}: ⟦The S\\&P 500, 2015--2026 --- Task||S\\&P 500, 2015--2026 --- cerință⟧',
       '⟦how much did the S\\&P 500 return, and how risky was it, from 2015 to 2026?||cît a adus S\\&P 500 și cît de riscant a fost, din 2015 pînă în 2026?⟧',
       '⟦S\\&P 500 closing prices, 2 January 2015 -- 18 September 2026 (\\texttt{load\\_close(\'sp500\')})||prețurile de închidere S\\&P 500, 2 ianuarie 2015 -- 18 septembrie 2026 (\\texttt{load\\_close(\'sp500\')})⟧',
       ['⟦Compute the daily simple and log returns, and check the first two step by step from the first three prices.||Calculați randamentele zilnice simple și logaritmice și verificați primele două, pas cu pas, din primele trei prețuri.⟧',
        '⟦Compute the number of returns per year $q$, the annualised mean log return and the annualised volatility.||Calculați numărul de randamente pe an $q$, randamentul logaritmic mediu anualizat și volatilitatea anualizată.⟧',
        '⟦Compute the cumulative return $P_T/P_0 - 1$, and check it against $\\exp(\\sum_t r_t) - 1$.||Calculați randamentul cumulat $P_T/P_0 - 1$ și verificați-l cu $\\exp(\\sum_t r_t) - 1$.⟧',
        '⟦Plot the value of 100 invested and the drawdown, and find the maximum drawdown with its dates.||Reprezentați grafic valoarea a 100 de unități investite și drawdown-ul și găsiți drawdown-ul maxim, cu datele lui.⟧'],
       '⟦$q$, the annual mean and volatility, the cumulative return, the maximum drawdown with its peak and trough dates, and one sentence of interpretation||$q$, media și volatilitatea anuale, randamentul cumulat, drawdown-ul maxim cu datele vîrfului și ale minimului și o propoziție de interpretare⟧',
       size='footnotesize', nb='B1')

D.frame(f'B1 {SOLVED}: ⟦Solution (1/3) --- Step by Step||Rezolvare (1/3) --- pas cu pas⟧', items(
    ('⟦\\textbf{First returns}, from $P_0 = $ @{b1_P0}, $P_1 = $ @{b1_P1}, $P_2 = $ @{b1_P2}||\\textbf{Primele randamente}, din $P_0 = $ @{b1_P0}, $P_1 = $ @{b1_P1}, $P_2 = $ @{b1_P2}⟧',
     ['$R_1 = P_1/P_0 - 1 = $ @{b1_R1}\\%, $r_1 = $ @{b1_r1}\\%',
      '$R_2 = $ @{b1_R2}\\%, $r_2 = $ @{b1_r2}\\%']),
    ('⟦\\textbf{Annualisation}||\\textbf{Anualizarea}⟧',
     ['⟦@{b1_n} returns in @{b1_years} years: $q = $ @{b1_q} returns a year||@{b1_n} randamente în @{b1_years} ani: $q = $ @{b1_q} randamente pe an⟧',
      '⟦daily mean @{b1_mean_d}\\%, daily volatility @{b1_sd_d}\\%||media zilnică @{b1_mean_d}\\%, volatilitatea zilnică @{b1_sd_d}\\%⟧',
      '⟦annual mean @{b1_ann_mean}\\%, annual volatility @{b1_ann_vol}\\%||media anuală @{b1_ann_mean}\\%, volatilitatea anuală @{b1_ann_vol}\\%⟧']),
    ('⟦\\textbf{Cumulative return}||\\textbf{Randamentul cumulat}⟧',
     ['⟦$P_T/P_0 - 1 = $ @{b1_PT} / @{b1_P0} $- 1 = $ @{b1_cum}\\%; $\\exp(\\sum_t r_t) - 1$ gives the same @{b1_cum_from_r}\\%||$P_T/P_0 - 1 = $ @{b1_PT} / @{b1_P0} $- 1 = $ @{b1_cum}\\%; $\\exp(\\sum_t r_t) - 1$ dă același @{b1_cum_from_r}\\%⟧'])) + '\n' + QL)

D.frame(f'B1 {SOLVED}: ⟦Solution (2/3) --- Value of 100 Invested||Rezolvare (2/3) --- valoarea a 100 de unități investite⟧',
        fig('ch0_sem_b1_growth', '0.6\\textheight') + items(
            '⟦100 invested on @{b1_first_date} were worth @{b1_val} on @{b1_last_date} (price index, without dividends)||100 de unități investite pe @{b1_first_date} valorau @{b1_val} pe @{b1_last_date} (indice de preț, fără dividende)⟧') + '\n' + QL,
        size='footnotesize')

D.frame(f'B1 {SOLVED}: ⟦Solution (3/3) --- Drawdown and Interpretation||Rezolvare (3/3) --- drawdown și interpretare⟧', cols(
    fig('ch0_sem_b1_drawdown', '0.55\\textheight'),
    items('⟦\\textbf{Maximum drawdown}: @{b1_mdd}\\%, from the peak of @{b1_mdd_peak} to @{b1_mdd_date} (COVID-19)||\\textbf{Drawdown-ul maxim}: @{b1_mdd}\\%, de la vîrful din @{b1_mdd_peak} la @{b1_mdd_date} (COVID-19)⟧',
          '⟦Second deepest episode: 2022, when interest rates rose||Al doilea episod ca adîncime: 2022, cînd au crescut dobînzile⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦a good average year (@{b1_ann_mean}\\%) hides a fall of a third in one month||un an mediu bun (@{b1_ann_mean}\\%) ascunde o scădere de o treime într-o lună⟧',
            '⟦report the volatility and the drawdown next to the mean||raportați volatilitatea și drawdown-ul alături de medie⟧'])),
    '0.55', '0.42') + '\n' + QL, size='footnotesize')

D.task(f'B2 {SOLVED}: ⟦One Suspicious Quote --- Task||O cotație suspectă --- cerință⟧',
       '⟦is the largest EUR/RON move in the market series a real price?||este cea mai mare variație EUR/RON din seria de piață un preț real?⟧',
       '⟦the EUR/RON series from EODHD (\\texttt{eurron\\_eodhd}) and the BNR reference rate (\\texttt{eurron}), 2015--2026||seria EUR/RON de la EODHD (\\texttt{eurron\\_eodhd}) și cursul de referință BNR (\\texttt{eurron}), 2015--2026⟧',
       ['⟦For each day, compute the gap between the EODHD quote and the BNR rate of the same day, in percent.||Pentru fiecare zi, calculați diferența procentuală dintre cotația EODHD și cursul BNR din aceeași zi.⟧',
        '⟦Find the day with the largest absolute gap, and print the quotes of the day before and the day after.||Găsiți ziua cu cea mai mare diferență în valoare absolută și afișați cotațiile din ziua anterioară și din ziua următoare.⟧',
        '⟦Compute the log returns into and out of that day.||Calculați randamentele logaritmice de intrare în acea zi și de ieșire din ea.⟧',
        '⟦Plot both series in the month around that day.||Reprezentați grafic ambele serii în luna din jurul acelei zile.⟧'],
       '⟦the date, the quote, the BNR rate, the two log returns, and your decision: real price or bad tick||data, cotația, cursul BNR, cele două randamente logaritmice și decizia voastră: preț real sau bad tick⟧',
       nb='B2')

D.frame(f'B2 {SOLVED}: ⟦Solution (1/2) --- Step by Step||Rezolvare (1/2) --- pas cu pas⟧', items(
    ('⟦\\textbf{Largest gap}: @{b2_date}||\\textbf{Cea mai mare diferență}: @{b2_date}⟧',
     ['⟦EODHD quote @{b2_quote} against the BNR rate @{b2_bnr}: a gap of +@{b2_gap}\\%||cotația EODHD @{b2_quote} față de cursul BNR @{b2_bnr}: o diferență de +@{b2_gap}\\%⟧',
      '⟦day before (@{b2_prev_date}): @{b2_prev}; day after (@{b2_next_date}): @{b2_next}||ziua anterioară (@{b2_prev_date}): @{b2_prev}; ziua următoare (@{b2_next_date}): @{b2_next}⟧']),
    ('⟦\\textbf{Two log returns}||\\textbf{Două randamente logaritmice}⟧',
     ['⟦into the day: +@{b2_move_in}\\%; out of the day: @{b2_move_out}\\%||la intrare: +@{b2_move_in}\\%; la ieșire: @{b2_move_out}\\%⟧',
      '⟦for comparison, the largest daily move of the BNR rate since 2015 is @{b2_bnr_max_move}\\%||pentru comparație, cea mai mare variație zilnică a cursului BNR din 2015 este de @{b2_bnr_max_move}\\%⟧']),
    ('⟦\\textbf{Decision}: a bad tick||\\textbf{Decizia}: un bad tick⟧',
     ['⟦a jump of @{b2_move_abs}\\% that fully reverses the next day, absent from the official rate||un salt de @{b2_move_abs}\\% care se inversează complet a doua zi și lipsește din cursul oficial⟧',
      '⟦kept in the data, it would add two of the largest daily moves of the leu in a decade||păstrat în date, ar adăuga două dintre cele mai mari variații zilnice ale leului dintr-un deceniu⟧']),
    '⟦In the course we use the BNR rate for EUR/RON||În curs folosim cursul BNR pentru EUR/RON⟧') + '\n' + QL)

D.frame(f'B2 {SOLVED}: ⟦Solution (2/2) --- Chart and Interpretation||Rezolvare (2/2) --- grafic și interpretare⟧',
        fig('ch0_sem_b2_eurron', '0.58\\textheight') + items(
            '⟦One quote far from every other quote and from the official rate: always look at the data before computing statistics||O cotație departe de toate celelalte și de cursul oficial: priviți întotdeauna datele înainte de a calcula statistici⟧',
            '⟦Interpretation: a volatility computed with this tick would describe an error, not the market||Interpretare: o volatilitate calculată cu acest tick ar descrie o eroare, nu piața⟧') + '\n' + QL,
        size='footnotesize')

# ===============================================================================================================
D.section('Your Turn', 'Rîndul vostru')
# ===============================================================================================================
D.task(f'A4 {PROP}: ⟦New Numbers --- Task||Alte cifre --- cerință⟧',
       '⟦can you repeat A1--A3 without the solution in front of you?||puteți repeta A1--A3 fără rezolvarea în față?⟧',
       '⟦a share at $P_0 = 50$, $P_1 = 55$, $P_2 = 44$; a fund at 200, 180, 220, 165, 198, 230 on days 0--5; daily mean $0.03\\%$ and daily volatility $1.5\\%$||o acțiune la $P_0 = 50$, $P_1 = 55$, $P_2 = 44$; un fond la 200, 180, 220, 165, 198, 230 în zilele 0--5; media zilnică $0.03\\%$ și volatilitatea zilnică $1.5\\%$⟧',
       ['⟦Model: A1, A2 and A3 [Solved].||Model: A1, A2 și A3 [Rezolvat].⟧',
        '⟦Compute the two simple and the two log returns of the share, their sums and the two-day returns.||Calculați cele două randamente simple și cele două randamente logaritmice ale acțiunii, sumele lor și randamentele pe două zile.⟧',
        '⟦Compute the drawdown of the fund for each day and its maximum drawdown.||Calculați drawdown-ul fondului pentru fiecare zi și drawdown-ul lui maxim.⟧',
        '⟦Annualise the daily mean and volatility with $q = 252$.||Anualizați media și volatilitatea zilnice, cu $q = 252$.⟧'],
       '⟦the two tables, the maximum drawdown, the two annual numbers, and one sentence on which return sum matches the two-day return||cele două tabele, drawdown-ul maxim, cele două cifre anuale și o propoziție despre care sumă coincide cu randamentul pe două zile⟧',
       nb='A4')

D.frame('A4: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', items(
    ('⟦\\textbf{Share}||\\textbf{Acțiunea}⟧',
     ['$R_1 = +$@{a4_R1}\\%, $R_2 = $ @{a4_R2}\\%; $r_1 = +$@{a4_r1}\\%, $r_2 = $ @{a4_r2}\\%',
      '⟦sums: @{a4_sumR}\\% (simple), @{a4_sumr}\\% (log); two-day: @{a4_totR}\\% (simple), @{a4_totr}\\% (log)||sume: @{a4_sumR}\\% (simple), @{a4_sumr}\\% (logaritmice); pe două zile: @{a4_totR}\\% (simplu), @{a4_totr}\\% (logaritmic)⟧',
      '⟦only the log sum equals the two-day log return||doar suma logaritmică este egală cu randamentul logaritmic pe două zile⟧']),
    ('⟦\\textbf{Fund}||\\textbf{Fondul}⟧',
     ['$DD_t$: @{a4_dd0}, @{a4_dd1}, @{a4_dd2}, @{a4_dd3}, @{a4_dd4}, @{a4_dd5} (\\%)',
      '⟦maximum drawdown @{a4_mdd}\\%, from 220 (day 2) to 165 (day 3); a new peak on day 5||drawdown maxim @{a4_mdd}\\%, de la 220 (ziua 2) la 165 (ziua 3); vîrf nou în ziua 5⟧']),
    ('⟦\\textbf{Annualisation}||\\textbf{Anualizarea}⟧',
     ['⟦mean $252 \\times 0.03\\% = $ @{a4_mean}\\%; volatility $\\sqrt{252} \\times 1.5\\% = $ @{a4_vol}\\%||media $252 \\times 0.03\\% = $ @{a4_mean}\\%; volatilitatea $\\sqrt{252} \\times 1.5\\% = $ @{a4_vol}\\%⟧'])),
    instructor_only=True)

D.task(f'B3 {PROP}: ⟦The BET-TR, 2015--2026 --- Task||BET-TR, 2015--2026 --- cerință⟧',
       '⟦how did the Romanian market with dividends compare with the S\\&P 500?||cum s-a comportat piața din România, cu dividende, față de S\\&P 500?⟧',
       '⟦BET-TR closing prices from 2 January 2015 (\\texttt{load\\_close(\'bettr\')}) to 18 September 2026||prețurile de închidere BET-TR de la 2 ianuarie 2015 (\\texttt{load\\_close(\'bettr\')}) la 18 septembrie 2026⟧',
       ['⟦Model: B1 [Solved]; reuse its code and change only the series name.||Model: B1 [Rezolvat]; refolosiți codul și schimbați doar numele seriei.⟧',
        '⟦Compute $q$, the annualised mean log return and the annualised volatility.||Calculați $q$, randamentul logaritmic mediu anualizat și volatilitatea anualizată.⟧',
        '⟦Compute the cumulative return and the maximum drawdown with its dates.||Calculați randamentul cumulat și drawdown-ul maxim, cu datele lui.⟧',
        '⟦Compare each number with the S\\&P 500 result of B1.||Comparați fiecare cifră cu rezultatul S\\&P 500 din B1.⟧'],
       '⟦a two-column table (S\\&P 500, BET-TR) and one sentence: why is the comparison not fully fair?||un tabel cu două coloane (S\\&P 500, BET-TR) și o propoziție: de ce comparația nu este pe deplin corectă?⟧',
       nb='B3')

D.frame('B3: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', cols(
    table('lrr', '& S\\&P 500 & BET-TR',
          ['$q$ & @{b1_q} & @{b3_q}', '⟦mean, \\% a year||media, \\% pe an⟧ & @{b1_ann_mean} & @{b3_ann_mean}',
           '⟦volatility, \\% a year||volatilitatea, \\% pe an⟧ & @{b1_ann_vol} & @{b3_ann_vol}',
           '⟦cumulative, \\%||cumulat, \\%⟧ & @{b1_cum} & @{b3_cum}', 'MDD, \\% & @{b1_mdd} & @{b3_mdd}'],
          size='footnotesize'),
    items('⟦BET-TR: maximum drawdown from @{b3_mdd_peak} to @{b3_mdd_date}||BET-TR: drawdown maxim de la @{b3_mdd_peak} la @{b3_mdd_date}⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦BET-TR: higher mean, lower volatility in this window||BET-TR: medie mai mare, volatilitate mai mică în această fereastră⟧',
            '⟦not fully fair: BET-TR includes dividends, the S\\&P 500 price index does not; currencies differ (RON, USD)||nu este pe deplin corectă: BET-TR include dividendele, indicele de preț S\\&P 500 nu; monedele diferă (RON, USD)⟧'])),
    '0.45', '0.52'), instructor_only=True)

D.task(f'B4 {PROP}: ⟦Bitcoin: 365 or 252 Days? --- Task||Bitcoin: 365 sau 252 de zile? --- cerință⟧',
       '⟦how large is the error if Bitcoin is annualised like a stock index?||cît de mare este eroarea dacă Bitcoin este anualizat ca un indice bursier?⟧',
       '⟦Bitcoin closing prices from 2 January 2015 (\\texttt{load\\_close(\'btc\')}) to 18 September 2026||prețurile de închidere Bitcoin de la 2 ianuarie 2015 (\\texttt{load\\_close(\'btc\')}) la 18 septembrie 2026⟧',
       ['⟦Model: A2 and B1 [Solved].||Model: A2 și B1 [Rezolvat].⟧',
        '⟦Compute the number of returns per year $q$ from the data.||Calculați din date numărul de randamente pe an $q$.⟧',
        '⟦Compute the annualised volatility with $q = 365$, and then with $q = 252$.||Calculați volatilitatea anualizată cu $q = 365$, apoi cu $q = 252$.⟧',
        '⟦Compute the cumulative return and the maximum drawdown with its dates.||Calculați randamentul cumulat și drawdown-ul maxim, cu datele lui.⟧'],
       '⟦$q$, the two volatilities and their ratio, the cumulative return, the maximum drawdown, and one sentence on which volatility is right||$q$, cele două volatilități și raportul lor, randamentul cumulat, drawdown-ul maxim și o propoziție despre volatilitatea corectă⟧',
       nb='B4')

D.frame('B4: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', items(
    '⟦@{b4_n} daily returns: $q = $ @{b4_q}, so 365 is right||@{b4_n} randamente zilnice: $q = $ @{b4_q}, deci 365 este corect⟧',
    ('⟦\\textbf{Volatility}||\\textbf{Volatilitatea}⟧',
     ['⟦with 365: @{b4_ann_vol_365}\\% a year; with 252: @{b4_ann_vol_252}\\% a year||cu 365: @{b4_ann_vol_365}\\% pe an; cu 252: @{b4_ann_vol_252}\\% pe an⟧',
      '⟦ratio $\\sqrt{365/252} = $ @{b4_ratio}: using 252 understates the risk by about a sixth||raportul $\\sqrt{365/252} = $ @{b4_ratio}: folosirea lui 252 subestimează riscul cu aproximativ o șesime⟧']),
    ('⟦\\textbf{Return and drawdown}||\\textbf{Randament și drawdown}⟧',
     ['⟦annual mean @{b4_ann_mean}\\%; cumulative return @{b4_cum}\\% (price $\\times$@{b4_mult})||media anuală @{b4_ann_mean}\\%; randament cumulat @{b4_cum}\\% (prețul $\\times$@{b4_mult})⟧',
      '⟦maximum drawdown @{b4_mdd}\\%, from @{b4_mdd_peak} to @{b4_mdd_date}||drawdown maxim @{b4_mdd}\\%, de la @{b4_mdd_peak} la @{b4_mdd_date}⟧']),
    '⟦Interpretation: huge growth and a loss of @{b4_mdd_abs}\\% within one year are both part of the same series||Interpretare: creșterea uriașă și pierderea de @{b4_mdd_abs}\\% într-un an fac parte din aceeași serie⟧'),
    instructor_only=True)

# ===============================================================================================================
D.section('Part C: Critique an AI Answer', 'Partea C: critica unui răspuns generat de AI')
# ===============================================================================================================
D.frame(f'C2 {PROP}: ⟦Critique an AI Answer --- the Answer||Critica unui răspuns generat de AI --- răspunsul⟧', items(
    '⟦\\textbf{Prompt} sent to an AI assistant: ``From the course data, compute the annualised volatility and the cumulative return of the S\\&P 500 and of Bitcoin since 2 January 2015, from daily simple returns.\'\'||\\textbf{Cererea} trimisă unui asistent AI: „Din datele cursului, calculează volatilitatea anualizată și randamentul cumulat ale S\\&P 500 și Bitcoin din 2 ianuarie 2015, din randamentele simple zilnice.”⟧') + r"""
\begin{lstlisting}
# read_market(symbol): the daily table of the course data
for name, sym in [('S&P 500', 'GSPC.INDX'), ('Bitcoin', 'BTC-USD.CC')]:
    p = read_market(sym)['close'].loc['2015-01-02':]   # closing prices
    R = p.pct_change().dropna()          # daily simple returns
    vol = R.std() * 252                  # annualised volatility
    cum = R.sum()                        # cumulative return
    print(name, round(100 * vol, 1), round(100 * cum, 1))
\end{lstlisting}
""" + items('⟦\\textbf{Conclusion of the AI}: ``Volatility: S\\&P 500 @{c2_sp500_ai_vol}\\%, Bitcoin @{c2_btc_ai_vol}\\%. Cumulative return: S\\&P 500 @{c2_sp500_ai_cum}\\%, Bitcoin @{c2_btc_ai_cum}\\%. Bitcoin earned only about five times as much as the S\\&P 500.\'\'||\\textbf{Concluzia AI}: „Volatilitate: S\\&P 500 @{c2_sp500_ai_vol}\\%, Bitcoin @{c2_btc_ai_vol}\\%. Randament cumulat: S\\&P 500 @{c2_sp500_ai_cum}\\%, Bitcoin @{c2_btc_ai_cum}\\%. Bitcoin a cîștigat doar de aproximativ cinci ori mai mult decît S\\&P 500.”⟧'),
        opts='fragile', size='footnotesize')

D.task(f'C2 {PROP}: ⟦Critique an AI Answer --- Task||Critica unui răspuns generat de AI --- cerință⟧',
       '⟦can you trust code that runs without an error message?||puteți avea încredere într-un cod care rulează fără niciun mesaj de eroare?⟧',
       '⟦the code and the conclusion on the previous slide; the definitions of the primer||codul și concluzia de pe slide-ul anterior; definițiile din noțiunile de bază⟧',
       ['⟦Model: A1, A2 and B1 [Solved].||Model: A1, A2 și B1 [Rezolvat].⟧',
        '⟦Find the three errors planted in the code.||Găsiți cele trei greșeli plantate în cod.⟧',
        '⟦For each error, say how you would detect it: a check on the output, the primer, or a solved exercise.||Pentru fiecare greșeală, spuneți cum ați detecta-o: o verificare a rezultatului, noțiunile de bază sau un exercițiu rezolvat.⟧',
        '⟦Write the corrected code, and report the corrected volatility and cumulative return of both assets.||Scrieți codul corectat și raportați volatilitatea și randamentul cumulat corectate pentru ambele active.⟧',
        '⟦Say whether the conclusion ``Bitcoin earned only about five times as much\'\' survives.||Spuneți dacă rezistă concluzia „Bitcoin a cîștigat doar de aproximativ cinci ori mai mult”.⟧'],
       '⟦a table AI versus corrected, and one line per error in the format: request, answer, error, how found, correction||un tabel AI față de corectat și cîte o linie pentru fiecare greșeală, în formatul: cerere, răspuns, greșeală, cum a fost găsită, corectare⟧',
       nb='C2')

D.frame('C2: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', cols(
    items(('⟦\\textbf{Error 1}: \\texttt{R.std() * 252}; volatility scales with $\\sqrt{252}$||\\textbf{Greșeala 1}: \\texttt{R.std() * 252}; volatilitatea crește cu $\\sqrt{252}$⟧',
           ['⟦detect: a volatility of @{c2_sp500_ai_vol}\\% a year for a stock index is impossible (A2)||detectare: o volatilitate de @{c2_sp500_ai_vol}\\% pe an pentru un indice bursier este imposibilă (A2)⟧']),
          ('⟦\\textbf{Error 2}: 252 days for Bitcoin, which trades every day||\\textbf{Greșeala 2}: 252 de zile pentru Bitcoin, care se tranzacționează în fiecare zi⟧',
           ['⟦detect: count the observations per year (@{c2_btc_q})||detectare: numărați observațiile pe an (@{c2_btc_q})⟧']),
          ('⟦\\textbf{Error 3}: \\texttt{R.sum()} is not the cumulative return; simple returns compound||\\textbf{Greșeala 3}: \\texttt{R.sum()} nu este randamentul cumulat; randamentele simple se compun⟧',
           ['⟦detect: compare with $P_T/P_0 - 1$ (A1, B1)||detectare: comparați cu $P_T/P_0 - 1$ (A1, B1)⟧'])),
    table('lrr', '& ⟦Vol. \\%||Vol. \\%⟧ & ⟦Cum. \\%||Cumulat \\%⟧',
          ['S\\&P 500, AI & @{c2_sp500_ai_vol} & @{c2_sp500_ai_cum}', '⟦S\\&P 500, corrected||S\\&P 500, corectat⟧ & @{c2_sp500_ok_vol} & @{c2_sp500_ok_cum}',
           'Bitcoin, AI & @{c2_btc_ai_vol} & @{c2_btc_ai_cum}', '⟦Bitcoin, corrected||Bitcoin, corectat⟧ & @{c2_btc_ok_vol} & @{c2_btc_ok_cum}'],
          size='footnotesize') + items(
        '⟦Corrected: \\texttt{R.std() * np.sqrt(q)} with $q$ = 252 or 365, and \\texttt{p.iloc[-1] / p.iloc[0] - 1}||Corectat: \\texttt{R.std() * np.sqrt(q)} cu $q$ = 252 sau 365 și \\texttt{p.iloc[-1] / p.iloc[0] - 1}⟧',
        '⟦The conclusion fails: Bitcoin earned about @{c2_ratio_ok} times as much as the S\\&P 500, not @{c2_ratio_ai0} times, with @{c2_vol_ratio} times the volatility||Concluzia cade: Bitcoin a cîștigat de aproximativ @{c2_ratio_ok} ori mai mult decît S\\&P 500, nu de @{c2_ratio_ai0} ori, cu o volatilitate de @{c2_vol_ratio} ori mai mare⟧'),
    '0.50', '0.47'), instructor_only=True, size='footnotesize')

# ===============================================================================================================
D.section('Wrap-up', 'Încheiere')
# ===============================================================================================================
D.frame('⟦Practice at Home||Exercițiu acasă⟧', items(
    ('⟦\\textbf{Finish the proposed exercises}: A4, B3, B4 and C2||\\textbf{Terminați exercițiile propuse}: A4, B3, B4 și C2⟧',
     ['⟦each names its model, a solved exercise to copy||fiecare își numește modelul, un exercițiu rezolvat de urmat⟧',
      '⟦the solutions are discussed at the next seminar||rezolvările se discută la seminarul următor⟧']),
    ('⟦\\textbf{Try one more series}||\\textbf{Încercați încă o serie}⟧',
     ['⟦repeat B1 for the DAX (\\texttt{dax}) or for gold (\\texttt{gold}), and compare the annual volatilities||repetați B1 pentru DAX (\\texttt{dax}) sau pentru aur (\\texttt{gold}) și comparați volatilitățile anuale⟧']),
    '⟦\\textbf{Take the Chapter 0 quiz} on the course website: 20 questions, each answer explained||\\textbf{Rezolvați quiz-ul Capitolului 0} pe site-ul cursului: 20 de întrebări, fiecare răspuns explicat⟧',
    '⟦Nothing is handed in: this is practice for the exam and for the project||Nu se predă nimic: este exercițiu pentru examen și pentru proiect⟧'))

D.frame('⟦Recap, and Next: the Chapter 0 Lecture||Recapitulare, iar în continuare: cursul din Capitolul 0⟧', cols(
    items(('⟦\\textbf{Today}||\\textbf{Azi}⟧',
           ['⟦simple returns compound, log returns add up||randamentele simple se compun, cele logaritmice se adună⟧',
            '⟦annualise the mean with $q$ and the volatility with $\\sqrt{q}$; $q = 365$ for Bitcoin||anualizăm media cu $q$ și volatilitatea cu $\\sqrt{q}$; $q = 365$ pentru Bitcoin⟧',
            '⟦the maximum drawdown is the worst loss from a peak||drawdown-ul maxim este cea mai mare pierdere de la un vîrf⟧',
            '⟦look at the data first: one bad tick can dominate a statistic||priviți întîi datele: un singur bad tick poate domina o statistică⟧',
            '⟦AI code can run and still be wrong||codul generat de AI poate rula și totuși să fie greșit⟧'])),
    items(('⟦\\textbf{In the lecture}||\\textbf{La curs}⟧',
           ['⟦How is the course organised and graded?||Cum este organizat și evaluat cursul?⟧',
            '⟦What do the 2026 markets look like in the data?||Cum arată piețele din 2026 în date?⟧',
            '⟦How did exchanges, crashes and models develop?||Cum au evoluat bursele, crahurile și modelele?⟧',
            '⟦Why are daily returns not Normal?||De ce randamentele zilnice nu au distribuția Normală?⟧'])),
    '0.52', '0.45'))

if __name__ == '__main__':
    D.write(V)
