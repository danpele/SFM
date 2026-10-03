r"""
build_chapter1.py -- Capitolul 1 (Date, randamente și indicatori), EN + RO dintr-o singură sursă
==============================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_01/ch1_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter1_data_returns_indicators.tex
  RO/Cursuri/capitol1_date_randamente_indicatori.tex
Rulare:
  python3 Quantlets/Ch_01/generate_all_charts.py
  python3 latex/build_chapter1.py && python3 latex/sfm_build.py compile 1
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import date as date_

from sfm_build import ROOT, Deck, block, cols, items, table, photo, ql   # noqa: E402
from ch1_common import BIB, INDICES, NAMES, REFS, SHORTNAME, STOCKS, T, date, load, put_date, values   # noqa: E402

N = load()
V = values(N)
D = Deck(1, 'lecture', refs=REFS)


TBL = '>{\\raggedright\\arraybackslash}'


def side(title, fig, folder, bullets, w=0.56, size='footnotesize', h='0.80\\textheight'):
    """Chart on the left, bullets on the right, Quantlet link below."""
    left = f'\\begin{{center}}\n\\includegraphics[width=\\linewidth,height={h},keepaspectratio]{{{fig}.pdf}}\n\\end{{center}}'
    D.frame(title, cols(left, items(*bullets), wl=f'{w:.2f}', wr=f'{0.96 - w:.2f}') + '\n' + ql(folder), size)


def chart(title, fig, folder, bullets, h='0.58\\textheight', size='footnotesize'):
    D.chart(title, fig, folder, bullets, height=h, size=size)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
# +10% apoi -10%
p0, p1 = 100.0, 110.0
p2 = p1 * 0.9
V.put('ex.p2', p2, 0)
V.put('ex.r1', 100 * math.log(p1 / p0), 2)
V.put('ex.r2', 100 * math.log(p2 / p1), 2)
V.put('ex.rsum', 100 * math.log(p2 / p0), 2)
V.put('ex.Rtot', 100 * (p2 / p0 - 1), 1)
# +50% apoi -50%
V.put('ex.half', 100 * (1.5 * 0.5 - 1), 0)
V.put('ex.halflog', 100 * math.log(0.75), 1)
# aproximarea r ~ R - R^2/2 pentru R = 1% și R = 10%
for R, k in [(0.01, 'one'), (0.10, 'ten')]:
    V.put(f'ex.{k}.r', 100 * math.log1p(R), 3)
    V.put(f'ex.{k}.taylor', 100 * (R - R * R / 2), 3)
    V.put(f'ex.{k}.gap', 1e4 * (R - math.log1p(R)), 1)
# preț ajustat: dividend 2 la 100 în ziua 5, split 2-la-1 în ziua 10
Adiv = 1 - 2 / 100
Asplit = 0.5
V.put('adj.div', Adiv, 2)
V.put('adj.split', Asplit, 2)
V.put('adj.both', Adiv * Asplit, 2)
V.put('adj.p', 100 * Adiv * Asplit, 0)
# portofoliu cu două active pe o perioadă
w1, Ra, Rb = 0.6, 0.10, -0.05
Rp = w1 * Ra + (1 - w1) * Rb
V.put('pf.Rp', 100 * Rp, 1)
V.put('pf.rp', 100 * math.log1p(Rp), 2)
V.put('pf.rsum', 100 * (w1 * math.log1p(Ra) + (1 - w1) * math.log1p(Rb)), 2)
V.put('pf.ra', 100 * math.log1p(Ra), 2)
V.put('pf.rb', 100 * math.log1p(Rb), 2)
# volatility drag: două active cu aceeași medie aritmetică 10%, volatilitate 5% și 30%
for s, k in [(0.05, 'a'), (0.30, 'b')]:
    g = 0.10 - s * s / 2
    V.put(f'vd.{k}.drag', 100 * s * s / 2, 2)
    V.put(f'vd.{k}.g', 100 * g, 2)
    V.put(f'vd.{k}.w10', math.exp(10 * g), 2)
# recuperarea după un drawdown
for k in ['sp500', 'bet', 'btc']:
    V.put(f'dd.{k}.mdd', 100 * N['dd'][k]['mdd'], 1)
    V.put(f'dd.{k}.need', 100 * (1 / (1 + N['dd'][k]['mdd']) - 1), 0)
    put_date(V, f'dd.{k}.peak', N['dd'][k]['peak'], short=True)
    put_date(V, f'dd.{k}.trough', N['dd'][k]['trough'], short=True)
    put_date(V, f'dd.{k}.rec', N['dd'][k]['recovery'], short=True)
    V.put(f'dd.{k}.under', 100 * N['dd'][k]['under_share'], 0)
    V.raw(f'dd.{k}.first', N['dd'][k]['first'][:4])
for x in [0.10, 0.20, 0.50, 0.80]:
    V.put(f'need.{int(100 * x)}', 100 * (1 / (1 - x) - 1), 0)
# date: EUR/RON, TLV, BET vs BET-TR, ADF
E = N['eurron']
V.int('eur.n', E['n'])
V.raw('eur.days', str(E['days_diff_1pct']))
V.put('eur.volb', E['vol_bnr'], 1)
V.put('eur.vole', E['vol_eodhd'], 1)
V.put('eur.maxe', E['max_eodhd'], 1)
put_date(V, 'eur.maxedate', E['max_eodhd_date'])
V.put('eur.maxb', E['max_bnr'], 1)
put_date(V, 'eur.maxbdate', E['max_bnr_date'])
TL = N['tlv']
put_date(V, 'tlv.date', TL['jump_date'])
V.put('tlv.jc', TL['jump_close'], 0)
V.put('tlv.ja', TL['jump_adj'], 1)
V.put('tlv.before', TL['close_before'], 2)
V.put('tlv.after', TL['close_after'], 2)
V.put('tlv.ratio', TL['close_after'] / TL['close_before'], 0)
V.raw('tlv.nev', str(TL['n_events_2pct']))
BT = N['bettr']
put_date(V, 'bt.start', BT['start'])
V.int('bt.bet', round(BT['end_bet']))
V.int('bt.tr', round(BT['end_bettr']))
V.put('bt.gbet', BT['cagr_bet'], 1, pct=True)
V.put('bt.gtr', BT['cagr_bettr'], 1, pct=True)
V.put('bt.gap', BT['cagr_bettr'] - BT['cagr_bet'], 1, pct=True)
A = N['adf']
V.put('adf.p', A['adf_price'], 2)
V.put('adf.pp', A['p_price'], 2)
V.put('adf.r', A['adf_ret'], 1)
V.put('adf.c5ct', A['crit5_ct'], 2)
V.put('adf.c5c', A['crit5_c'], 2)
V.raw('adf.first', A['first'][:4])
SL = N['simplelog']
for k in ['btc_min', 'btc_max', 'sp500_min', 'sp500_max']:
    V.put(f'sl.{k}.R', SL[k]['R'], 1)
    V.put(f'sl.{k}.r', SL[k]['r'], 1)
    put_date(V, f'sl.{k}.date', SL[k]['date'])
G = N['growth']
for k in G:
    V.int(f'g.{k}.end', round(G[k]['end']))
    V.int(f'g.{k}.cs', round(100 * G[k]['cum_simple']))
    V.put(f'g.{k}.cl', G[k]['cum_log'], 2)
PF = N['port']
V.put('pf.err', PF['mean_abs_err_bp'], 2)
V.put('pf.maxerr', PF['max_err_bp'], 1)
put_date(V, 'pf.maxdate', PF['max_err_date'])
V.put('pf.sumx', PF['sum_exact'], 2)
V.put('pf.suma', PF['sum_approx'], 2)
V.put('pf.wx', math.exp(PF['sum_exact']), 1)
V.put('pf.wa', math.exp(PF['sum_approx']), 1)
V.put('pf.cagr', PF['cagr_port'], 1, pct=True)
V.put('pf.cagra', PF['cagr_a'], 1, pct=True)
V.put('pf.cagrb', PF['cagr_b'], 1, pct=True)
H = N['hist']
for k in ['sp500', 'btc']:
    V.put(f'h.{k}.in1', 100 * H[k]['within1sd'], 1)
    V.put(f'h.{k}.out3', 100 * H[k]['beyond3sd'], 2)
V.put('h.n.in1', 100 * H['sp500']['normal_within1sd'], 1)
V.put('h.n.out3', 100 * H['sp500']['normal_beyond3sd'], 2)
V.put('h.ratio', H['sp500']['beyond3sd'] / H['sp500']['normal_beyond3sd'], 1)
RV = N['rollvol']
for k in RV:
    V.put(f'rv.{k}.min', RV[k]['min'], 0)
    V.put(f'rv.{k}.max', RV[k]['max'], 0)
    V.put(f'rv.{k}.last', RV[k]['last'], 0)
OH = N['ohlc']
ohlc_rows = []
for d_, r in OH.items():
    ohlc_rows.append(f"{date(d_, short=True)} & $" + f"{r['open']:.2f}$ & ${r['high']:.2f}$ & ${r['low']:.2f}$ & ${r['close']:.2f}$ & "
                     + f"{int(r['volume']):,}".replace(',', '\\,'))
flat = [d_ for d_, r in OH.items() if r['open'] == r['high'] == r['low'] == r['close']]
put_date(V, 'ohlc.flat', flat[0] if flat else None)
V.int('ohlc.flatvol', OH[flat[0]]['volume'] if flat else 0)
# precizia mediei: aceeași durată, frecvență diferită
V.put('se.sp.ann', N['desc']['sp500']['ann_vol'] / math.sqrt(N['desc']['sp500']['n'] / N['desc']['sp500']['obs_per_year']), 1)
V.put('sp.daily.sd', N['desc']['sp500']['sd'], 2)
V.put('sp.ann.from', N['desc']['sp500']['sd'] * math.sqrt(252), 1)
V.put('btc.ann.from', N['desc']['btc']['sd'] * math.sqrt(365), 1)
V.put('btc.ann.wrong', N['desc']['btc']['sd'] * math.sqrt(252), 1)
V.put('sqrt252', math.sqrt(252), 2)
V.put('sqrt365', math.sqrt(365), 2)
# Sharpe: exemplu și ranguri
P = N['perf']
sr_sp = P['sp500']['sharpe']
V.put('sr.sp.d', sr_sp / math.sqrt(P['sp500']['obs_per_year']), 3)
best = {m: max(P, key=lambda k: P[k][m]) for m in ['cagr', 'sharpe', 'sortino', 'calmar']}
best['vol'] = min(P, key=lambda k: P[k]['vol'])
best['mdd'] = max(P, key=lambda k: P[k]['mdd'])
for m, k in best.items():
    V.raw(f'best.{m}', SHORTNAME[k])
V.put('btc.drag', 100 * P['btc']['drag'], 1)
V.put('btc.halfvar', 100 * P['btc']['half_var'], 1)
V.put('btc.ma', 100 * P['btc']['mean_arith'], 1)
V.put('btc.ml', 100 * P['btc']['mean_log'], 1)
V.put('btc.cagr', 100 * P['btc']['cagr'], 1)
V.put('bars', 6.5 * 60 / 5, 0)
V.put('sp.absmdd', -100 * P['sp500']['mdd'], 1)
V.put('sp.ml.exp', 100 * (math.exp(P['sp500']['mean_log']) - 1), 1)
V.put('dd.bet.years', (date_.fromisoformat(N['dd']['bet']['recovery']) - date_.fromisoformat(N['dd']['bet']['peak'])).days / 365.25, 1)
V.put('se.sr.full', 1 / math.sqrt(P['sp500']['years']), 1)
V.put('se.sr.half', 1 / math.sqrt(P['sp500']['years'] / 2), 1)
V.put('se.sr.years', P['sp500']['years'], 0)
with open(os.path.join(ROOT, 'data', 'manifest.csv')) as f_:
    V.raw('nseries', str(sum(1 for _ in f_) - 1))
# verificați-vă: cifre calculate din enunț
V.put('cy.mu', 252 * 0.05, 1)
V.put('cy.vol', math.sqrt(252) * 1.0, 1)
V.put('cy.need', 100 * 0.6 / 0.4, 0)


# =============================================================================
# TITLU, TRASEU
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we turn raw market data into numbers that let us compare investments?',
       '\\textbf{Întrebarea}: cum transformăm datele brute de piață în cifre care permit compararea investițiilor?'),
     [T('a price level says little; a return, a risk measure and a drawdown say a lot',
        'un nivel de preț spune puțin; un randament, o măsură a riscului și un drawdown sînt mult mai informative')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('data sources and data quality: where prices come from and what can go wrong',
        'surse de date și calitatea datelor: de unde provin prețurile și ce erori pot apărea'),
      T('simple and log returns, over several periods and for portfolios',
        'randamente simple și logaritmice, pe mai multe perioade și pentru portofolii'),
      T('annualisation and descriptive statistics of returns',
        'anualizarea și statisticile descriptive ale randamentelor'),
      T('performance indicators: CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, volatility drag',
        'indicatori de performanță: CAGR, volatilitate, Sharpe, Sortino, drawdown maxim, Calmar, volatility drag')]),
    T('Real data throughout: BET, S\\&P 500, DAX, Bitcoin and five stocks of the BVB',
      'Date reale în tot capitolul: BET, S\\&P 500, DAX, Bitcoin și cinci acțiuni de la BVB')))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Choose a reliable data source and check a price series before using it',
      'Alegerea unei surse de date de încredere și verificarea unei serii de prețuri înainte de utilizare'),
    T('Compute simple and log returns, and aggregate them over time and across assets',
      'Calculul randamentelor simple și logaritmice și agregarea lor în timp și între active'),
    T('Annualise means and volatilities with the actual frequency of the data',
      'Anualizarea mediilor și a volatilităților cu frecvența reală a datelor'),
    T('Describe a return series by its moments, quantiles and extremes',
      'Descrierea unei serii de randamente prin momente, cuantile și valori extreme'),
    T('Compute and interpret CAGR, Sharpe, Sortino, maximum drawdown, Calmar and the volatility drag',
      'Calculul și interpretarea indicatorilor CAGR, Sharpe, Sortino, drawdown maxim, Calmar și volatility drag'),
    T('Judge whether a difference between two indicators is larger than its estimation error',
      'Compararea diferenței dintre doi indicatori cu eroarea lor de estimare')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, \\textit{Statistics of Financial Markets}, 5th ed., Ch.~11 (returns, definitions)',
       'Manual: \\refFHH, \\textit{Statistics of Financial Markets}, ed. a 5-a, cap.~11 (randamente, definiții)'),
     [T('exercises with solutions: \\refFHHex', 'exerciții rezolvate: \\refFHHex')]),
    T('Complementary: \\refTsay, Ch.~1 (asset returns); \\refCLM, Ch.~1',
      'Lecturi complementare: \\refTsay, cap.~1 (randamentele activelor); \\refCLM, cap.~1'),
    (T('Python Quantlets of this chapter: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_01}{Quantlets/Ch\\_01}',
       'Quantlet-urile Python ale capitolului: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_01}{Quantlets/Ch\\_01}'),
     [T('ported from the SFE Quantlets of the textbook (SFEReturns, SFEcomplogreturns, SFEportlogreturns, SFEfdescStat)',
        'adaptate după Quantlet-urile SFE ale manualului (SFEReturns, SFEcomplogreturns, SFEportlogreturns, SFEfdescStat)'),
      T('each chart in these slides links to the Quantlet that draws it', 'fiecare grafic din aceste slide-uri are un link către Quantlet-ul care îl generează')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter1_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter1_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}',
      'Curs video: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}')))

# =============================================================================
# 1. DATE
# =============================================================================
D.section('Financial Data: Sources and Quality', 'Date financiare: surse și calitate')

D.frame(T('What Financial Data Are', 'Tipuri de date financiare'), items(
    (T('\\textbf{Market data}: what happens on a trading venue', '\\textbf{Date de piață}: informațiile produse de un loc de tranzacționare'),
     [T('trade prices, best bid and ask quotes, traded volume', 'prețuri de tranzacționare, cele mai bune cotații bid și ask, volum tranzacționat'),
      T('index levels, exchange rates, interest rates, crypto prices', 'niveluri de indici, cursuri de schimb, rate ale dobînzii, prețuri cripto')]),
    (T('\\textbf{Corporate actions}: events that change the price without changing the value of a holding',
       '\\textbf{Evenimente corporative}: evenimente care schimbă prețul fără a schimba valoarea deținerilor'),
     [T('dividends, splits, share consolidations, bonus shares', 'dividende, splituri, consolidări de acțiuni, acțiuni gratuite')]),
    (T('\\textbf{Fundamental and macro data}', '\\textbf{Date fundamentale și macroeconomice}'),
     [T('balance sheets and earnings; inflation, GDP, policy rates', 'bilanțuri și profituri; inflație, PIB, dobînzi de politică monetară')]),
    (T('\\textbf{Alternative data}', '\\textbf{Date alternative}'),
     [T('news, social media, satellite images, blockchain transactions', 'știri, rețele sociale, imagini din satelit, tranzacții pe blockchain')]),
    T('This course works mainly with \\textbf{daily market data}', 'Cursul folosește în principal \\textbf{date zilnice de piață}')))

D.frame(T('From the Ticker Tape to Data Feeds', 'De la banda telegrafică la fluxurile de date'), cols(items(
    (T('From 1867, \\textbf{stock tickers} printed trade prices on a paper tape, sent by telegraph',
       'Din 1867, \\textbf{telegrafele bursiere} (stock tickers) tipăreau pe o bandă de hîrtie prețurile tranzacțiilor, primite prin telegraf'),
     [T('the first real-time price feed: the price of the last trade, minutes after it happened',
        'primul flux de prețuri în timp real: prețul ultimei tranzacții, la cîteva minute după ce a avut loc')]),
    (T('Today: electronic feeds from the exchanges, resold by data vendors', 'Azi: fluxuri electronice de la burse, redistribuite de furnizorii de date'),
     [T('every quote and trade, time-stamped to the microsecond', 'fiecare cotație și tranzacție, cu marcaj de timp la microsecundă'),
      T('daily summaries (OHLCV) are what most studies use', 'majoritatea studiilor folosesc rezumatele zilnice (OHLCV)')]),
    T('The ticker survived: scrolling prices and news on building fronts; ticker symbols such as TLV, SNP, H2O on the BVB',
      'Banda de cotații a rămas: prețuri și știri care defilează pe fațadele clădirilor; simbolurile bursiere (tickere), ca TLV, SNP, H2O la BVB')),
    photo('ch1_reuters_ticker.jpg', T('News ticker on the Reuters building, New York', 'Bandă de știri pe clădirea Reuters, New York'),
          'https://commons.wikimedia.org/wiki/File:Reuters_News_Ticker.jpg',
          T('Photo: bgilliard (2005); CC BY-SA 2.0; Wikimedia Commons', 'Foto: bgilliard (2005); CC BY-SA 2.0; Wikimedia Commons'),
          h='0.50\\textheight'), wl='0.58', wr='0.38'))

D.frame(T('Data Frequency', 'Frecvența datelor'), items(
    (T('\\textbf{Tick data}: every trade and quote', '\\textbf{Date tick}: fiecare tranzacție și cotație'),
     [T('millions of records per day; noise from the bid--ask bounce', 'milioane de înregistrări pe zi; zgomot cauzat de alternanța tranzacțiilor între bid și ask (bid--ask bounce)')]),
    (T('\\textbf{Intraday bars}: 1, 5 or 60 minutes', '\\textbf{Bare intrazilnice}: 1, 5 sau 60 de minute'),
     [T('used for realised volatility (Chapter 8)', 'folosite pentru volatilitatea realizată (Capitolul 8)')]),
    (T('\\textbf{Daily data}: one row per trading day', '\\textbf{Date zilnice}: un rînd pe zi de tranzacționare'),
     [T('the standard of this course and of most academic studies', 'standardul acestui curs și al majorității studiilor academice'),
      T('exchanges trade about 250 days a year; crypto markets trade 365 days a year',
        'bursele funcționează aproximativ 250 de zile pe an; piețele cripto, 365 de zile pe an')]),
    (T('\\textbf{Weekly, monthly, yearly data}', '\\textbf{Date săptămînale, lunare, anuale}'),
     [T('fewer observations, closer to the horizon of long-term investors', 'mai puține observații, mai aproape de orizontul investitorilor pe termen lung')]),
    T('Rule: annualise with the \\textbf{actual} number of observations per year of each series',
      'Regulă: anualizați cu numărul \\textbf{real} de observații pe an al fiecărei serii')))

D.frame(T('Where the Data Come From', 'Surse de date'), table(
    TBL + 'p{3.4cm}' + TBL + 'p{5.6cm}' + TBL + 'p{3.4cm}',
    T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{What it provides}', '\\textbf{Date disponibile}') + ' & ' + T('\\textbf{Access}', '\\textbf{Acces}'),
    [T('Exchanges (BVB, Deutsche Börse, NYSE)', 'Burse (BVB, Deutsche Börse, NYSE)') + ' & ' + T('official prices, volumes, index levels', 'prețuri oficiale, volume, niveluri ale indicilor') + ' & ' + T('website, paid feeds', 'site, fluxuri contra cost'),
     T('Data vendors (Bloomberg, LSEG, EODHD)', 'Furnizori de date (Bloomberg, LSEG, EODHD)') + ' & ' + T('prices from many venues, adjusted prices, fundamentals', 'prețuri de pe multe piețe, prețuri ajustate, date fundamentale') + ' & ' + T('subscription', 'abonament'),
     T('Central banks (BNR, ECB, Federal Reserve)', 'Bănci centrale (BNR, BCE, Rezerva Federală)') + ' & ' + T('reference exchange rates, interest rates', 'cursuri de referință, rate ale dobînzii') + ' & ' + T('free', 'gratuit'),
     T('Statistical databases (FRED, Eurostat)', 'Baze de date statistice (FRED, Eurostat)') + ' & ' + T('macro series, yields', 'serii macroeconomice, randamente ale titlurilor de stat') + ' & ' + T('free', 'gratuit'),
     T('Academic databases (CRSP, Kenneth French library)', 'Baze de date academice (CRSP, biblioteca Kenneth French)') + ' & ' + T('cleaned returns, including delisted firms; factor returns', 'randamente curățate, inclusiv firme delistate; randamente ale factorilor') + ' & ' + T('university licence or free', 'licență universitară sau gratuit'),
     T('Crypto exchanges and aggregators', 'Burse și agregatoare cripto') + ' & ' + T('prices 24/7, on-chain data', 'prețuri 24/7, date on-chain') + ' & ' + T('free or subscription', 'gratuit sau abonament')],
    size='footnotesize') + items(T('A vendor collects and cleans; the \\textbf{primary source} of a number (exchange, central bank) is the reference when two sources disagree',
                                   'Furnizorul colectează și curăță datele; \\textbf{sursa primară} a cifrei (bursa, banca centrală) este referința atunci cînd două surse diferă')))

D.frame(T('The Data of This Course', 'Datele acestui curs'), items(
    (T('Daily market data from \\textbf{EODHD} (EOD Historical Data), saved once in the course repository',
       'Date zilnice de piață de la \\textbf{EODHD} (EOD Historical Data), salvate o singură dată în repository-ul cursului'),
     [T('@{nseries} series: indices, ETFs, stocks (BVB, US, Europe), crypto, exchange rates, bond yields',
        '@{nseries} de serii: indici, ETF-uri, acțiuni (BVB, SUA, Europa), cripto, cursuri de schimb, randamente ale obligațiunilor'),
      T('columns: date, open, high, low, close, adjusted close, volume', 'coloane: data, deschidere, maxim, minim, închidere, închidere ajustată, volum')]),
    (T('EUR/RON: the official \\textbf{BNR} reference rate', 'EUR/RON: cursul oficial de referință al \\textbf{BNR}'),
     [T('the EUR/RON series from EODHD has erroneous quotes (next slides)', 'seria EUR/RON de la EODHD are cotații eronate (slide-urile următoare)')]),
    (T('Conventions', 'Convenții'),
     [T('indices, exchange rates, crypto: \\textit{close}; stocks and ETFs: \\textit{adjusted close}',
        'indici, cursuri de schimb, cripto: \\textit{close}; acțiuni și ETF-uri: \\textit{adjusted close}'),
      T('weekend quotes and repeated holiday closes are dropped, except for crypto',
        'cotațiile de weekend și închiderile repetate în zilele libere se elimină, cu excepția activelor cripto'),
      T('each series is analysed on its own calendar', 'fiecare serie se analizează pe propriul calendar')]),
    T('Comparison period of the chapter: @{start} to @{end}', 'Perioada de comparație a capitolului: @{start} -- @{end}')))

D.frame(T('Daily Bars: OHLCV', 'Barele zilnice: OHLCV'), items(
    T('\\textbf{OHLCV}: open $O_t$, high $H_t$, low $L_t$, close $C_t$, volume $V_t$ of day $t$',
      '\\textbf{OHLCV}: prețul de deschidere $O_t$, maxim $H_t$, minim $L_t$, de închidere $C_t$ și volumul $V_t$ din ziua $t$'),
    T('By construction $L_t \\le O_t, C_t \\le H_t$; the range $H_t - L_t$ measures the intraday spread of prices',
      'Prin construcție $L_t \\le O_t, C_t \\le H_t$; amplitudinea $H_t - L_t$ măsoară împrăștierea prețurilor în timpul zilei'),
    T('Banca Transilvania (TLV), the last five trading days in our data:', 'Banca Transilvania (TLV), ultimele cinci zile de tranzacționare din datele noastre:')) +
    table('lrrrrr', T('Date', 'Data') + ' & Open & High & Low & Close & ' + T('Volume', 'Volum'), ohlc_rows, size='footnotesize') +
    items(T('Check: on @{ohlc.flat} open = high = low = close with @{ohlc.flatvol} shares traded; a flat bar with a large volume is suspicious',
            'Verificare: pe @{ohlc.flat} deschidere = maxim = minim = închidere, cu @{ohlc.flatvol} acțiuni tranzacționate; o bară plată cu volum mare este suspectă')),
    size='footnotesize')

D.frame(T('Corporate Actions and the Adjusted Close', 'Evenimente corporative și prețul ajustat'), items(
    T('The traded close $P_t$ jumps on the day of a corporate action; the value of a holding does not',
      'Prețul de închidere $P_t$ face un salt în ziua unui eveniment corporativ; valoarea deținerii nu se modifică'),
    (T('\\textbf{Adjustment factor} $A$ on the event day', '\\textbf{Factorul de ajustare} $A$ în ziua evenimentului'),
     [T('cash dividend $D$: $A = 1 - D/P_{t-1}$', 'dividend în numerar $D$: $A = 1 - D/P_{t-1}$'),
      T('split $m$-for-$n$ ($m$ new shares for $n$ old): $A = n/m$; a consolidation is a split with $m < n$',
        'split $m$-la-$n$ ($m$ acțiuni noi pentru $n$ vechi): $A = n/m$; o consolidare este un split cu $m < n$')]),
    T('\\textbf{Adjusted close}: every earlier price is multiplied by the factors of all later events, $P^{\\text{adj}}_t = P_t \\prod_{k:\\, t_k > t} A_{t_k}$',
      '\\textbf{Prețul ajustat}: fiecare preț anterior se înmulțește cu factorii tuturor evenimentelor ulterioare, $P^{\\text{adj}}_t = P_t \\prod_{k:\\, t_k > t} A_{t_k}$'),
    T('Returns computed from adjusted prices include the dividend and have no artificial jumps',
      'Randamentele calculate din prețuri ajustate includ dividendul și nu au salturi artificiale'),
    T('With dividends, the simple return is $R_t = (P_t + D_t)/P_{t-1} - 1$ \\refFHH',
      'Cu dividende, randamentul simplu este $R_t = (P_t + D_t)/P_{t-1} - 1$ \\refFHH')))

D.frame(T('Worked Example: Adjusted Prices', 'Exemplu rezolvat: prețuri ajustate'), items(
    T('A share trades at 100; on day 5 it pays a cash dividend of 2; on day 10 it splits 2-for-1',
      'O acțiune cotează la 100; în ziua 5 plătește un dividend de 2; în ziua 10 are loc un split 2-la-1'),
    (T('Adjustment factors', 'Factorii de ajustare'),
     [T('dividend: $A_5 = 1 - 2/100 = @{adj.div}$; split: $A_{10} = 1/2 = @{adj.split}$',
        'dividend: $A_5 = 1 - 2/100 = @{adj.div}$; split: $A_{10} = 1/2 = @{adj.split}$')]),
    (T('Adjusted prices', 'Prețurile ajustate'),
     [T('before day 5: $P^{\\text{adj}}_t = P_t \\times @{adj.both}$, so 100 becomes @{adj.p}',
        'înainte de ziua 5: $P^{\\text{adj}}_t = P_t \\times @{adj.both}$, deci 100 devine @{adj.p}'),
      T('between days 5 and 9: $P^{\\text{adj}}_t = P_t \\times @{adj.split}$; from day 10: $P^{\\text{adj}}_t = P_t$',
        'între zilele 5 și 9: $P^{\\text{adj}}_t = P_t \\times @{adj.split}$; din ziua 10: $P^{\\text{adj}}_t = P_t$')]),
    T('The adjusted series changes every time a new event happens: always keep the download date',
      'Seria ajustată se schimbă la fiecare eveniment nou: păstrați întotdeauna data descărcării'),
    T('Indices: a \\textbf{price index} ignores dividends; a \\textbf{total return index} reinvests them',
      'Indici: un \\textbf{indice de preț} ignoră dividendele; un \\textbf{indice de randament total} le reinvestește')))

chart(T('Real Data: Banca Transilvania, Close vs Adjusted Close', 'Date reale: Banca Transilvania, prețul de închidere și prețul ajustat'),
      'sfm_ch1_tlv_adjusted', 'SFM_ch1_data_quality', [
          T('On @{tlv.date} the close jumps from @{tlv.before} to @{tlv.after} RON: a share consolidation, @{tlv.ratio} old shares into one',
            'Pe @{tlv.date} prețul urcă brusc de la @{tlv.before} la @{tlv.after} lei: o consolidare a acțiunilor, @{tlv.ratio} acțiuni vechi într-una'),
          T('Log return from the close: $+@{tlv.jc}\\%$ in one day; from the adjusted close: $@{tlv.ja}\\%$',
            'Randamentul logaritmic din prețul de închidere: $+@{tlv.jc}\\%$ într-o zi; din prețul ajustat: $@{tlv.ja}\\%$'),
          T('Since 2015 the two series disagree by more than 2\\% on @{tlv.nev} days (dividends, bonus shares)',
            'Din 2015, cele două serii diferă cu peste 2\\% în @{tlv.nev} zile (dividende, acțiuni gratuite)')])

chart(T('Dividends Matter: BET vs BET-TR', 'Dividendele contează: BET și BET-TR'), 'sfm_ch1_bet_bettr', 'SFM_ch1_data_quality', [
    T('100 invested on @{bt.start}: @{bt.bet} in the BET (price index), @{bt.tr} in the BET-TR (dividends reinvested)',
      '100 de unități investite pe @{bt.start}: @{bt.bet} în BET (indice de preț), @{bt.tr} în BET-TR (dividende reinvestite)'),
    T('Growth rate: @{bt.gbet}\\% vs @{bt.gtr}\\% a year; dividends add @{bt.gap} percentage points a year',
      'Rata de creștere: @{bt.gbet}\\% pe an pentru BET, față de @{bt.gtr}\\% pentru BET-TR; dividendele adaugă @{bt.gap} puncte procentuale pe an'),
    T('Compare like with like: the DAX is a total return index, the S\\&P 500 and the BET are price indices',
      'Comparați lucruri comparabile: DAX este un indice de randament total, S\\&P 500 și BET sînt indici de preț')])

chart(T('Data Errors: Two EUR/RON Series', 'Erori în date: două serii EUR/RON'), 'sfm_ch1_eurron_check', 'SFM_ch1_data_quality', [
    T('Same exchange rate, @{eur.n} common days: the EODHD series moves by more than 1 percentage point more than the BNR rate on @{eur.days} days',
      'Același curs, @{eur.n} de zile comune: variația zilnică a seriei EODHD diferă cu peste 1 punct procentual de cea a cursului BNR în @{eur.days} zile'),
    T('Annualised volatility: @{eur.vole}\\% (EODHD) vs @{eur.volb}\\% (BNR); largest EODHD move $@{eur.maxe}\\%$ on @{eur.maxedate}',
      'Volatilitatea anualizată: @{eur.vole}\\% (EODHD), față de @{eur.volb}\\% (BNR); cea mai mare variație zilnică a seriei EODHD: $@{eur.maxe}\\%$, pe @{eur.maxedate}'),
    T('Fix: use the primary source of the number, the BNR reference rate', 'Soluția: folosiți sursa primară a cifrei, cursul de referință BNR')],
    h='0.54\\textheight')

D.frame(T('Three Biases That Make Results Look Too Good', 'Trei erori sistematice care fac rezultatele să pară prea bune'), items(
    (T('\\textbf{Survivorship bias}: only firms or funds that still exist are in the sample', '\\textbf{Survivorship bias}: în eșantion apar doar firmele sau fondurile care încă există'),
     [T('the losers have disappeared, so the average return is too high \\refBGIR', 'firmele cu rezultate slabe au dispărut, deci randamentul mediu este supraestimat \\refBGIR'),
      T('example: today\'s members of the BET, followed back to 2000', 'exemplu: componența actuală a BET, urmărită înapoi pînă în 2000')]),
    (T('\\textbf{Look-ahead bias}: using information not yet available at the decision date', '\\textbf{Look-ahead bias}: folosirea unei informații care nu era disponibilă la data deciziei'),
     [T('annual reports published in March used for decisions in January', 'rapoarte anuale publicate în martie folosite pentru decizii din ianuarie')]),
    (T('\\textbf{Data snooping}: trying many rules on the same data and reporting the best one', '\\textbf{Data snooping}: încercarea multor reguli pe aceleași date și raportarea celei mai bune'),
     [T('hundreds of published return predictors are likely false discoveries \\refHLZ', 'sute de predictori publicați ai randamentelor sînt probabil descoperiri false \\refHLZ')]),
    T('Each bias inflates returns; none shows up in the summary statistics', 'Fiecare dintre aceste erori supraestimează randamentele; niciuna nu se vede în statisticile descriptive')))

D.frame(T('A Data Checklist', 'Lista de verificare a datelor'), items(
    T('Source: the primary source of the number and the column used (close or adjusted close)',
      'Sursa: sursa primară a cifrei și coloana folosită (close sau adjusted close)'),
    T('Calendar: weekends, holidays, missing days; same dates for all assets in a joint analysis',
      'Calendarul: weekenduri, zile libere, zile lipsă; aceleași date pentru toate activele într-o analiză comună'),
    T('Jumps: list the 10 largest absolute returns and check each against the news',
      'Salturi: listați cele mai mari 10 randamente în valoare absolută și verificați-le pe fiecare în știri'),
    T('Flat bars, zero volume, repeated prices: holidays filled with the last price or stale quotes',
      'Bare plate, volum zero, prețuri repetate: zile libere completate cu ultimul preț sau cotații vechi'),
    T('Units and currency: RON, EUR, USD; index points; prices in cents', 'Unități și monedă: lei, euro, dolari; puncte de indice; prețuri în cenți'),
    T('Keep the raw file and the download date; never edit data manually', 'Păstrați fișierul brut și data descărcării; nu modificați niciodată datele manual')))

D.recap(('Data', 'date'), [
    T('Prices come from exchanges, vendors and central banks; the primary source of a number is the reference',
      'Prețurile provin de la burse, furnizori și bănci centrale; sursa primară a cifrei este referința'),
    T('Use adjusted prices for stocks, total return indices when dividends matter',
      'Folosiți prețuri ajustate pentru acțiuni și indici de randament total cînd dividendele contează'),
    T('Real data contain errors: check jumps, flat bars and a second source', 'Datele reale conțin erori: verificați salturile, barele plate și o a doua sursă'),
    T('Survivorship, look-ahead and data snooping inflate results silently', 'Survivorship bias, look-ahead bias și data snooping supraestimează rezultatele fără să lase urme vizibile')])

# =============================================================================
# 2. DE LA PREȚURI LA RANDAMENTE
# =============================================================================
D.section('From Prices to Returns', 'De la prețuri la randamente')

D.frame(T('Why Returns, Not Prices?', 'De ce randamente, nu prețuri?'), items(
    (T('A price has a unit and a scale; a return does not', 'Un preț are unitate de măsură și scară; un randament nu are'),
     [T('a 1-leu move is large for a 2-leu share and small for a 500-leu share', 'o mișcare de 1 leu este mare pentru o acțiune de 2 lei și mică pentru una de 500 de lei')]),
    T('Returns of different assets, currencies and periods can be compared directly',
      'Randamentele unor active, monede și perioade diferite se pot compara direct'),
    (T('Prices trend and wander; returns fluctuate around a stable level', 'Prețurile au trend și nu revin la un nivel fix; randamentele oscilează în jurul unui nivel stabil'),
     [T('\\textbf{weakly stationary} series: constant mean, constant variance, and a covariance that depends only on the lag',
        'serie \\textbf{slab staționară}: medie constantă, varianță constantă și o covarianță care depinde doar de decalaj'),
      T('most statistical methods of this course assume (at least) weak stationarity',
        'majoritatea metodelor statistice ale cursului presupun (cel puțin) staționaritate slabă')]),
    T('The investor cares about the relative change of wealth, which is exactly a return',
      'Investitorul este interesat de variația relativă a averii, adică exact de un randament')))

chart(T('S\\&P 500: Price Level vs Daily Returns', 'S\\&P 500: nivelul prețului și randamentele zilnice'), 'sfm_ch1_price_returns',
      'SFM_ch1_prices_returns', [
          T('Top: the price trends upward with long swings; its mean and spread change over time',
            'Sus: prețul are trend crescător, cu oscilații lungi; media și împrăștierea lui se schimbă în timp'),
          T('Bottom: the returns fluctuate around zero, in calm and turbulent periods (2008, 2020)',
            'Jos: randamentele oscilează în jurul lui zero, în perioade calme și agitate (2008, 2020)')], h='0.62\\textheight')

D.frame(T('A Formal Check: the ADF Test', 'O verificare formală: testul ADF'), items(
    (T('\\textbf{ADF test} \\refDF: is there a unit root (a random walk) in the series?', '\\textbf{Testul ADF} \\refDF: are seria o rădăcină unitară (un mers aleator)?'),
     [T('$H_0$: unit root, non-stationary; $H_1$: stationary', '$H_0$: rădăcină unitară, nestaționară; $H_1$: staționară'),
      T('regression $\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} + \\sum_j \\delta_j \\Delta y_{t-j} + \\varepsilon_t$; reject $H_0$ when the $t$-statistic of $\\gamma$ is very negative',
        'regresia $\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} + \\sum_j \\delta_j \\Delta y_{t-j} + \\varepsilon_t$; respingem $H_0$ cînd statistica $t$ a lui $\\gamma$ este foarte negativă')]),
    (T('S\\&P 500, daily data since @{adf.first}', 'S\\&P 500, date zilnice din @{adf.first}'),
     [T('log price (with trend): statistic $@{adf.p}$, 5\\% critical value $@{adf.c5ct}$, $p = @{adf.pp}$: we do not reject the unit root',
        'logaritmul prețului (cu trend): statistica $@{adf.p}$, valoarea critică 5\\% $@{adf.c5ct}$, $p = @{adf.pp}$: nu respingem rădăcina unitară'),
      T('log returns: statistic $@{adf.r}$, 5\\% critical value $@{adf.c5c}$, $p < 0.001$: we reject it',
        'randamente logaritmice: statistica $@{adf.r}$, valoarea critică 5\\% $@{adf.c5c}$, $p < 0.001$: o respingem')]),
    T('Unit roots and random walks are the subject of Chapter 7', 'Rădăcinile unitare și mersul aleator sînt subiectul Capitolului 7')) + ql('SFM_ch1_prices_returns'))

# =============================================================================
# 3. RANDAMENTE SIMPLE ȘI LOGARITMICE
# =============================================================================
D.section('Simple and Log Returns', 'Randamente simple și logaritmice')

D.frame(T('The Simple Return', 'Randamentul simplu'), items(
    (T('\\textbf{Simple (net) return} from $t-1$ to $t$ \\refFHH, \\refTsay', '\\textbf{Randamentul simplu (net)} de la $t-1$ la $t$ \\refFHH, \\refTsay'),
     [T('$R_t = \\dfrac{P_t - P_{t-1}}{P_{t-1}} = \\dfrac{P_t}{P_{t-1}} - 1$', '$R_t = \\dfrac{P_t - P_{t-1}}{P_{t-1}} = \\dfrac{P_t}{P_{t-1}} - 1$'),
      T('\\textbf{gross return}: $1 + R_t = P_t / P_{t-1}$', '\\textbf{randamentul brut}: $1 + R_t = P_t / P_{t-1}$')]),
    T('Bounded below: $R_t \\ge -1$ (limited liability: you cannot lose more than you invested)',
      'Mărginit inferior: $R_t \\ge -1$ (răspundere limitată: nu puteți pierde mai mult decît ați investit)'),
    T('Example: $P_{t-1} = 100$, $P_t = 110$: $R_t = 10\\%$', 'Exemplu: $P_{t-1} = 100$, $P_t = 110$: $R_t = 10\\%$'),
    T('It is what an investor sees on the account statement: ``the fund gained 10\\%\'\'',
      'Este ceea ce vede investitorul în extrasul de cont: „fondul a cîștigat 10\\%”'),
    T('Notation: from here on $P_t$ is the close of an index or the adjusted close of a stock',
      'Notație: de aici încolo $P_t$ este prețul de închidere al unui indice sau prețul ajustat al unei acțiuni')))

D.frame(T('The Log Return', 'Randamentul logaritmic'), items(
    (T('\\textbf{Log return} (continuously compounded return)', '\\textbf{Randamentul logaritmic} (randamentul compus continuu)'),
     [T('$r_t = \\ln\\dfrac{P_t}{P_{t-1}} = \\ln P_t - \\ln P_{t-1} = \\ln(1 + R_t)$', '$r_t = \\ln\\dfrac{P_t}{P_{t-1}} = \\ln P_t - \\ln P_{t-1} = \\ln(1 + R_t)$'),
      T('back to the simple return: $R_t = e^{r_t} - 1$', 'înapoi la randamentul simplu: $R_t = e^{r_t} - 1$')]),
    T('Name: $e^{r}$ is the growth of 1 leu at rate $r$ compounded continuously over the period',
      'Originea denumirii: $e^{r}$ este valoarea la care ajunge un leu la rata $r$, compusă continuu pe durata perioadei'),
    T('Unbounded: $r_t \\in (-\\infty, +\\infty)$; a total loss ($P_t = 0$) is $r_t = -\\infty$',
      'Nemărginit: $r_t \\in (-\\infty, +\\infty)$; pierderea totală ($P_t = 0$) înseamnă $r_t = -\\infty$'),
    T('Example: $P_{t-1} = 100$, $P_t = 110$: $r_t = \\ln 1.1 = @{ex.r1}\\%$', 'Exemplu: $P_{t-1} = 100$, $P_t = 110$: $r_t = \\ln 1.1 = @{ex.r1}\\%$'),
    T('Log returns are the default in statistics of financial markets: they add up over time (next section)',
      'Randamentele logaritmice sînt alegerea standard în statistica piețelor financiare: se adună în timp (secțiunea următoare)')))

D.frame(T('How Far Apart Are $r$ and $R$?', 'Cît de mult diferă $r$ și $R$?'), items(
    (T('Taylor expansion of the logarithm around 0', 'Dezvoltarea în serie Taylor a logaritmului în jurul lui 0'),
     [T('$r = \\ln(1 + R) = R - \\dfrac{R^2}{2} + \\dfrac{R^3}{3} - \\dots$', '$r = \\ln(1 + R) = R - \\dfrac{R^2}{2} + \\dfrac{R^3}{3} - \\dots$'),
      T('$\\ln$ is concave, so $r < R$ for every $R \\neq 0$', '$\\ln$ este concavă, deci $r < R$ pentru orice $R \\neq 0$')]),
    (T('Daily move, $R = 1\\%$', 'Mișcare zilnică, $R = 1\\%$'),
     [T('$r = @{ex.one.r}\\%$, and $R - R^2/2 = @{ex.one.taylor}\\%$: the gap $R - r$ is @{ex.one.gap} basis points',
        '$r = @{ex.one.r}\\%$, iar $R - R^2/2 = @{ex.one.taylor}\\%$: diferența $R - r$ este de @{ex.one.gap} puncte de bază')]),
    (T('Large move, $R = 10\\%$', 'Mișcare mare, $R = 10\\%$'),
     [T('$r = @{ex.ten.r}\\%$, and $R - R^2/2 = @{ex.ten.taylor}\\%$: the gap $R - r$ is @{ex.ten.gap} basis points',
        '$r = @{ex.ten.r}\\%$, iar $R - R^2/2 = @{ex.ten.taylor}\\%$: diferența $R - r$ este de @{ex.ten.gap} puncte de bază')]),
    T('A \\textbf{basis point} (bp) is 0.01 percentage points', 'Un \\textbf{punct de bază} (bp) este 0,01 puncte procentuale'),
    T('Conclusion: for daily returns $r \\approx R$; for large moves and long horizons they differ',
      'Concluzie: pentru randamente zilnice $r \\approx R$; pentru mișcări mari și orizonturi lungi diferă')))

side(T('Simple vs Log Returns on Real Extremes', 'Randamente simple și logaritmice în zilele extreme'), 'sfm_ch1_simple_log', 'SFM_ch1_simple_log_returns', [
    T('Bitcoin, @{sl.btc_min.date}: $R = @{sl.btc_min.R}\\%$ but $r = @{sl.btc_min.r}\\%$', 'Bitcoin, @{sl.btc_min.date}: $R = @{sl.btc_min.R}\\%$, dar $r = @{sl.btc_min.r}\\%$'),
    T('Bitcoin, @{sl.btc_max.date}: $R = +@{sl.btc_max.R}\\%$, $r = +@{sl.btc_max.r}\\%$', 'Bitcoin, @{sl.btc_max.date}: $R = +@{sl.btc_max.R}\\%$, $r = +@{sl.btc_max.r}\\%$'),
    T('S\\&P 500, @{sl.sp500_min.date}: $R = @{sl.sp500_min.R}\\%$, $r = @{sl.sp500_min.r}\\%$', 'S\\&P 500, @{sl.sp500_min.date}: $R = @{sl.sp500_min.R}\\%$, $r = @{sl.sp500_min.r}\\%$'),
    T('The curve bends down: losses look larger and gains smaller in log returns', 'Curba este concavă: în randamente logaritmice, pierderile par mai mari, iar cîștigurile mai mici')])

D.frame(T('Worked Example: Up 10\\%, Then Down 10\\%', 'Exemplu rezolvat: +10\\%, apoi --10\\%'), cols(items(
    (T('Price path $100 \\to 110 \\to @{ex.p2}$', 'Traiectoria prețului $100 \\to 110 \\to @{ex.p2}$'),
     [T('simple returns: $+10\\%$ and $-10\\%$; their sum, 0, is misleading', 'randamente simple: $+10\\%$ și $-10\\%$; suma lor, 0, este înșelătoare'),
      T('compounded: $1.1 \\times 0.9 - 1 = @{ex.Rtot}\\%$', 'compuse: $1.1 \\times 0.9 - 1 = @{ex.Rtot}\\%$')]),
    (T('Log returns', 'Randamente logaritmice'),
     [T('$r_1 = \\ln 1.1 = @{ex.r1}\\%$, $r_2 = \\ln 0.9 = @{ex.r2}\\%$', '$r_1 = \\ln 1.1 = @{ex.r1}\\%$, $r_2 = \\ln 0.9 = @{ex.r2}\\%$'),
      T('sum: $@{ex.rsum}\\% = \\ln(@{ex.p2}/100)$, exact', 'suma: $@{ex.rsum}\\% = \\ln(@{ex.p2}/100)$, exact')])),
    block(T('Rule', 'Regula'), items(T('Simple returns compound (multiply); log returns add', 'Randamentele simple se compun (se înmulțesc); cele logaritmice se adună'))), wl='0.58', wr='0.38'))

D.frame(T('Question for the Room: Up 50\\%, Down 50\\%', 'Întrebare pentru sală: +50\\%, apoi --50\\%'), items(
    T('A share gains 50\\% one year and loses 50\\% the next', 'O acțiune crește cu 50\\% într-un an și scade cu 50\\% în anul următor'),
    T('\\textbf{What do you think?} Where is the price after two years, relative to the start?',
      '\\textbf{Ce credeți?} La ce nivel ajunge prețul după doi ani, față de nivelul inițial?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('$1.5 \\times 0.5 = 0.75$: the wealth changed by $@{ex.half}\\%$', '$1.5 \\times 0.5 = 0.75$: averea s-a modificat cu $@{ex.half}\\%$'),
      T('log returns: $\\ln 1.5 + \\ln 0.5 = \\ln 0.75 = @{ex.halflog}\\%$', 'randamente logaritmice: $\\ln 1.5 + \\ln 0.5 = \\ln 0.75 = @{ex.halflog}\\%$'),
      T('a loss of $x$ needs a gain of $x/(1-x)$ to recover: 50\\% needs 100\\%', 'o pierdere $x$ cere un cîștig de $x/(1-x)$ pentru revenire: 50\\% cere 100\\%')])))

D.frame(T('Which Return When?', 'Alegerea tipului de randament'), table(
    TBL + 'p{4.2cm}' + TBL + 'p{2.6cm}' + TBL + 'p{5.6cm}',
    T('\\textbf{Task}', '\\textbf{Situația}') + ' & ' + T('\\textbf{Use}', '\\textbf{Randament}') + ' & ' + T('\\textbf{Reason}', '\\textbf{Motivul}'),
    [T('Aggregating over time', 'Agregarea în timp') + ' & ' + T('log', 'logaritmic') + ' & ' + T('$r_{t}(k) = r_t + \\dots + r_{t-k+1}$, a sum', '$r_{t}(k) = r_t + \\dots + r_{t-k+1}$, o sumă'),
     T('Portfolio of several assets', 'Portofoliu din mai multe active') + ' & ' + T('simple', 'simplu') + ' & ' + T('$R_p = \\sum_i w_i R_i$, exact', '$R_p = \\sum_i w_i R_i$, exact'),
     T('Statistical modelling', 'Modelare statistică') + ' & ' + T('log', 'logaritmic') + ' & ' + T('unbounded, closer to a symmetric distribution', 'nemărginit, mai aproape de o distribuție simetrică'),
     T('Reporting to investors', 'Raportare către investitori') + ' & ' + T('simple', 'simplu') + ' & ' + T('``the fund gained 10\\%\'\' is a simple return', '„fondul a cîștigat 10\\%” este un randament simplu'),
     T('Daily data, small moves', 'Date zilnice, mișcări mici') + ' & ' + T('either', 'oricare') + ' & ' + T('the difference is about $R^2/2$', 'diferența este aproximativ $R^2/2$')],
    size='footnotesize') + items(T('Always say which one you use: a ``return of 10\\%\'\' is ambiguous', 'Precizați întotdeauna tipul de randament folosit: un „randament de 10\\%” este ambiguu')))

D.recap(('Simple and Log Returns', 'randamente simple și logaritmice'), [
    T('$R_t = P_t/P_{t-1} - 1$; $r_t = \\ln(P_t/P_{t-1}) = \\ln(1 + R_t)$', '$R_t = P_t/P_{t-1} - 1$; $r_t = \\ln(P_t/P_{t-1}) = \\ln(1 + R_t)$'),
    T('$r < R$ for every move; the gap is about $R^2/2$, negligible for daily data', '$r < R$ pentru orice mișcare; diferența este aproximativ $R^2/2$, neglijabilă pentru date zilnice'),
    T('Log returns add over time; simple returns add across assets', 'Randamentele logaritmice se adună în timp; cele simple se adună între active'),
    T('Gains and losses of the same size do not cancel', 'Cîștigurile și pierderile de aceeași mărime nu se anulează')])

# =============================================================================
# 4. MAI MULTE PERIOADE ȘI PORTOFOLII
# =============================================================================
D.section('Multi-Period and Portfolio Returns', 'Randamente pe mai multe perioade și ale portofoliilor')

D.frame(T('Returns over $k$ Periods', 'Randamente pe $k$ perioade'), items(
    (T('\\textbf{Simple}: gross returns multiply', '\\textbf{Simple}: randamentele brute se înmulțesc'),
     [T('$1 + R_t(k) = \\dfrac{P_t}{P_{t-k}} = (1 + R_t)(1 + R_{t-1}) \\cdots (1 + R_{t-k+1})$',
        '$1 + R_t(k) = \\dfrac{P_t}{P_{t-k}} = (1 + R_t)(1 + R_{t-1}) \\cdots (1 + R_{t-k+1})$')]),
    (T('\\textbf{Log}: returns add', '\\textbf{Logaritmice}: randamentele se adună'),
     [T('$r_t(k) = \\ln\\dfrac{P_t}{P_{t-k}} = r_t + r_{t-1} + \\dots + r_{t-k+1}$', '$r_t(k) = \\ln\\dfrac{P_t}{P_{t-k}} = r_t + r_{t-1} + \\dots + r_{t-k+1}$')]),
    T('Consequence: the mean and variance of a $k$-day log return follow from those of the daily returns',
      'Consecință: media și varianța randamentului logaritmic pe $k$ zile rezultă din cele ale randamentelor zilnice'),
    T('Weekly, monthly or yearly returns are computed from the prices at the end of each period, not by averaging',
      'Randamentele săptămînale, lunare sau anuale se calculează din prețurile de la sfîrșitul fiecărei perioade, nu prin mediere')))

D.frame(T('Cumulative Return and Wealth', 'Randamentul cumulat și averea'), items(
    T('Wealth after $T$ periods, starting from $W_0$: $W_T = W_0 \\prod_{t=1}^{T} (1 + R_t) = W_0 \\exp\\Big(\\sum_{t=1}^{T} r_t\\Big)$',
      'Averea după $T$ perioade, pornind de la $W_0$: $W_T = W_0 \\prod_{t=1}^{T} (1 + R_t) = W_0 \\exp\\Big(\\sum_{t=1}^{T} r_t\\Big)$'),
    (T('\\textbf{Cumulative simple return}: $\\text{CSR}_T = W_T/W_0 - 1$', '\\textbf{Randamentul simplu cumulat}: $\\text{CSR}_T = W_T/W_0 - 1$'),
     [T('what the investor gained, in percent', 'cîștigul investitorului, în procente')]),
    (T('\\textbf{Cumulative log return}: $\\text{CLR}_T = \\ln(W_T/W_0) = \\sum_t r_t$', '\\textbf{Randamentul logaritmic cumulat}: $\\text{CLR}_T = \\ln(W_T/W_0) = \\sum_t r_t$'),
     [T('link: $\\text{CSR}_T = e^{\\text{CLR}_T} - 1$', 'legătura: $\\text{CSR}_T = e^{\\text{CLR}_T} - 1$')]),
    T('Log scale on charts: equal vertical distances are equal percentage changes',
      'Scara logaritmică pe grafice: distanțe verticale egale corespund unor variații procentuale egale')))

chart(T('Growth of 100 Invested, @{y0}--@{y1}', 'Evoluția a 100 de unități investite, @{y0}--@{y1}'), 'sfm_ch1_growth', 'SFM_ch1_portfolio_returns', [
    T('Final values: S\\&P 500 @{g.sp500.end}, DAX @{g.dax.end}, BET @{g.bet.end}, Bitcoin @{g.btc.end}',
      'Valori finale: S\\&P 500 @{g.sp500.end}, DAX @{g.dax.end}, BET @{g.bet.end}, Bitcoin @{g.btc.end}'),
    T('Bitcoin: cumulative simple return @{g.btc.cs}\\%, cumulative log return @{g.btc.cl}; BET: @{g.bet.cs}\\% and @{g.bet.cl}',
      'Bitcoin: randament simplu cumulat @{g.btc.cs}\\%, randament logaritmic cumulat @{g.btc.cl}; BET: @{g.bet.cs}\\% și @{g.bet.cl}'),
    T('Without the log scale, every series except Bitcoin would look flat', 'Fără scara logaritmică, toate seriile, cu excepția Bitcoin, ar părea plate')])

D.frame(T('Portfolio Returns', 'Randamentul unui portofoliu'), items(
    (T('Portfolio with weights $w_1, \\dots, w_n$, $\\sum_i w_i = 1$, set at the start of the period', 'Portofoliu cu ponderile $w_1, \\dots, w_n$, $\\sum_i w_i = 1$, fixate la începutul perioadei'),
     [T('\\textbf{simple}: $R_p = \\sum_i w_i R_i$, exact', '\\textbf{simplu}: $R_p = \\sum_i w_i R_i$, exact'),
      T('\\textbf{log}: $r_p = \\ln\\big(\\sum_i w_i e^{r_i}\\big) \\neq \\sum_i w_i r_i$', '\\textbf{logaritmic}: $r_p = \\ln\\big(\\sum_i w_i e^{r_i}\\big) \\neq \\sum_i w_i r_i$')]),
    (T('Example: 60\\% in A ($R_A = 10\\%$), 40\\% in B ($R_B = -5\\%$)', 'Exemplu: 60\\% în A ($R_A = 10\\%$), 40\\% în B ($R_B = -5\\%$)'),
     [T('$R_p = 0.6 \\times 10\\% + 0.4 \\times (-5\\%) = @{pf.Rp}\\%$; exact $r_p = \\ln(1 + R_p) = @{pf.rp}\\%$',
        '$R_p = 0.6 \\times 10\\% + 0.4 \\times (-5\\%) = @{pf.Rp}\\%$; exact $r_p = \\ln(1 + R_p) = @{pf.rp}\\%$'),
      T('weighted log returns: $0.6 \\times @{pf.ra}\\% + 0.4 \\times (@{pf.rb}\\%) = @{pf.rsum}\\%$, too low',
        'randamente logaritmice ponderate: $0.6 \\times @{pf.ra}\\% + 0.4 \\times (@{pf.rb}\\%) = @{pf.rsum}\\%$, prea mic')]),
    T('Weights drift when prices move: a fixed-weight portfolio must be \\textbf{rebalanced}',
      'Ponderile se modifică odată cu prețurile: un portofoliu cu ponderi fixe trebuie \\textbf{reechilibrat}')) + ql('SFM_ch1_portfolio_returns'))

D.frame(T('Real Data: a TLV--SNP Portfolio', 'Date reale: un portofoliu TLV--SNP'), items(
    T('50\\% Banca Transilvania, 50\\% OMV Petrom, rebalanced daily, @{y0}--@{y1}', '50\\% Banca Transilvania, 50\\% OMV Petrom, reechilibrat zilnic, @{y0}--@{y1}'),
    (T('One day: exact log return vs weighted sum of log returns', 'O zi: randamentul logaritmic exact față de suma ponderată a randamentelor logaritmice'),
     [T('mean absolute gap @{pf.err} basis points; largest gap @{pf.maxerr} basis points, on @{pf.maxdate}',
        'diferența medie în valoare absolută @{pf.err} puncte de bază; cea mai mare @{pf.maxerr} puncte de bază, pe @{pf.maxdate}')]),
    (T('Over the whole period the small gaps accumulate', 'Pe întreaga perioadă diferențele mici se acumulează'),
     [T('exact cumulative log return @{pf.sumx}, so 1 leu grows to @{pf.wx} lei', 'randamentul logaritmic cumulat exact @{pf.sumx}, deci 1 leu devine @{pf.wx} lei'),
      T('weighted log returns give @{pf.suma}, so only @{pf.wa} lei', 'randamentele logaritmice ponderate conduc la @{pf.suma}, deci doar la @{pf.wa} lei')]),
    T('Growth rate: portfolio @{pf.cagr}\\% a year; TLV @{pf.cagra}\\%; SNP @{pf.cagrb}\\%', 'Rata de creștere: portofoliul @{pf.cagr}\\% pe an; TLV @{pf.cagra}\\%; SNP @{pf.cagrb}\\%'),
    T('Rule: build portfolios with simple returns, then take logs if needed', 'Regula: construiți portofoliile cu randamente simple, apoi treceți la logaritmi dacă este nevoie')) + ql('SFM_ch1_portfolio_returns'))

D.recap(('Multi-Period and Portfolio Returns', 'randamente pe mai multe perioade și ale portofoliilor'), [
    T('Over time: multiply gross returns or add log returns', 'În timp: înmulțiți randamentele brute sau adunați randamentele logaritmice'),
    T('Wealth $W_T = W_0 e^{\\text{CLR}_T}$; $\\text{CSR}_T = e^{\\text{CLR}_T} - 1$', 'Averea $W_T = W_0 e^{\\text{CLR}_T}$; $\\text{CSR}_T = e^{\\text{CLR}_T} - 1$'),
    T('Across assets: $R_p = \\sum_i w_i R_i$ exactly; the log version is only an approximation', 'Între active: $R_p = \\sum_i w_i R_i$ exact; varianta logaritmică este doar o aproximare'),
    T('Small daily approximation errors become visible over a decade', 'Erorile zilnice mici de aproximare devin vizibile pe un deceniu')])

# =============================================================================
# 5. ANUALIZARE
# =============================================================================
D.section('Annualisation', 'Anualizarea')

D.frame(T('Annualising the Mean and the Volatility', 'Anualizarea mediei și a volatilității'), items(
    T('$q$ = number of observations per year: about 252 for exchanges, 365 for crypto, 52 for weekly, 12 for monthly data',
      '$q$ = numărul de observații pe an: aproximativ 252 pentru burse, 365 pentru cripto, 52 pentru date săptămînale, 12 pentru date lunare'),
    (T('\\textbf{Mean}: log returns add, so $\\mu_{\\text{year}} = q\\,\\mu$', '\\textbf{Media}: randamentele logaritmice se adună, deci $\\mu_{\\text{an}} = q\\,\\mu$'),
     [T('the same rule is used for the mean of simple returns (arithmetic annualisation)', 'aceeași regulă se folosește pentru media randamentelor simple (anualizare aritmetică)')]),
    (T('\\textbf{Volatility} = standard deviation of returns: $\\sigma_{\\text{year}} = \\sqrt{q}\\,\\sigma$, the \\textbf{square-root-of-time rule}',
       '\\textbf{Volatilitatea} = abaterea standard a randamentelor: $\\sigma_{\\text{an}} = \\sqrt{q}\\,\\sigma$, \\textbf{regula rădăcinii pătrate a timpului}'),
     [T('valid when returns are uncorrelated with constant variance: then $\\Var(r_1 + \\dots + r_q) = q\\,\\sigma^2$',
        'valabilă cînd randamentele sînt necorelate și au varianță constantă: atunci $\\Var(r_1 + \\dots + r_q) = q\\,\\sigma^2$')]),
    (T('S\\&P 500: daily $\\sigma = @{sp.daily.sd}\\%$, so $\\sigma_{\\text{year}} = @{sp.daily.sd}\\% \\times @{sqrt252} = @{sp.ann.from}\\%$',
       'S\\&P 500: $\\sigma$ zilnic $= @{sp.daily.sd}\\%$, deci $\\sigma_{\\text{an}} = @{sp.daily.sd}\\% \\times @{sqrt252} = @{sp.ann.from}\\%$'),
     [T('Bitcoin: $\\sqrt{365} = @{sqrt365}$ gives @{btc.ann.from}\\%; the stock-market factor $\\sqrt{252}$ would give only @{btc.ann.wrong}\\%',
        'Bitcoin: $\\sqrt{365} = @{sqrt365}$ conduce la @{btc.ann.from}\\%; factorul bursier $\\sqrt{252}$ ar conduce doar la @{btc.ann.wrong}\\%')])))

D.frame(T('The Compound Annual Growth Rate', 'Rata anuală compusă de creștere'), items(
    (T('\\textbf{CAGR} (compound annual growth rate): the constant yearly rate that turns $P_0$ into $P_T$ in $Y$ years',
       '\\textbf{CAGR} (compound annual growth rate, rata anuală compusă de creștere): rata anuală constantă care transformă $P_0$ în $P_T$ în $Y$ ani'),
     [T('$\\text{CAGR} = \\big(P_T / P_0\\big)^{1/Y} - 1$, with $Y$ in calendar years', '$\\text{CAGR} = \\big(P_T / P_0\\big)^{1/Y} - 1$, cu $Y$ în ani calendaristici')]),
    T('Link with log returns: $\\text{CAGR} = \\exp(\\text{annual mean log return}) - 1$',
      'Legătura cu randamentele logaritmice: $\\text{CAGR} = \\exp(\\text{media anuală a randamentelor logaritmice}) - 1$'),
    T('S\\&P 500, @{y0}--@{y1}: mean log return @{p.sp500.mean_log}\\% a year, $e^{@{p.sp500.mean_log}\\%} - 1 = @{sp.ml.exp}\\%$, the CAGR is @{p.sp500.cagr}\\%',
      'S\\&P 500, @{y0}--@{y1}: media randamentelor logaritmice @{p.sp500.mean_log}\\% pe an, $e^{@{p.sp500.mean_log}\\%} - 1 = @{sp.ml.exp}\\%$, iar CAGR este @{p.sp500.cagr}\\%'),
    T('The CAGR uses only the first and the last price: it says nothing about the path in between',
      'CAGR folosește doar primul și ultimul preț: nu spune nimic despre traiectoria dintre ele'),
    T('Also called the geometric mean return', 'Se mai numește randamentul mediu geometric')))

D.frame(T('How Precisely Do We Know the Mean?', 'Cît de precis cunoaștem media?'), items(
    (T('Standard error of the annual mean: $\\text{SE}(\\hat\\mu_{\\text{year}}) = \\sigma_{\\text{year}} / \\sqrt{Y}$, with $Y$ years of data',
       'Eroarea standard a mediei anuale: $\\text{SE}(\\hat\\mu_{\\text{an}}) = \\sigma_{\\text{an}} / \\sqrt{Y}$, cu $Y$ ani de date'),
     [T('it depends on the number of \\textbf{years}, not on the number of observations', 'depinde de numărul de \\textbf{ani}, nu de numărul de observații')]),
    (T('S\\&P 500, @{d.sp500.years} years: annual mean log return @{d.sp500.annmean}\\%, SE @{d.sp500.se}\\%',
       'S\\&P 500, @{d.sp500.years} ani: media anuală a randamentelor logaritmice @{d.sp500.annmean}\\%, SE @{d.sp500.se}\\%'),
     [T('95\\% CI (confidence interval): [@{d.sp500.cilo}\\%, @{d.sp500.cihi}\\%]', 'intervalul de încredere 95\\%: [@{d.sp500.cilo}\\%; @{d.sp500.cihi}\\%]')]),
    (T('Bitcoin: @{d.btc.annmean}\\% $\\pm$ $1.96 \\times @{d.btc.se}\\%$', 'Bitcoin: @{d.btc.annmean}\\% $\\pm$ $1.96 \\times @{d.btc.se}\\%$'),
     [T('the interval is [@{d.btc.cilo}\\%, @{d.btc.cihi}\\%]: very wide', 'intervalul este [@{d.btc.cilo}\\%; @{d.btc.cihi}\\%]: foarte larg')]),
    T('Volatility, by contrast, is estimated precisely from daily data', 'Volatilitatea, în schimb, se estimează precis din date zilnice')) + ql('SFM_ch1_descriptive_stats'))

D.frame(T('Question for the Room: More Data, Better Mean?', 'Întrebare pentru sală: mai multe date, o medie mai bună?'), items(
    T('You replace 10 years of daily S\\&P 500 data by 10 years of 5-minute data: @{bars} bars in a 6.5-hour trading day, so @{bars} times more observations',
      'Înlocuiți 10 ani de date zilnice S\\&P 500 cu 10 ani de date la 5 minute: @{bars} de bare într-o zi de tranzacționare de 6,5 ore, deci de @{bars} de ori mai multe observații'),
    T('\\textbf{What do you think?} Does the standard error of the annual mean return become smaller?',
      '\\textbf{Ce credeți?} Devine mai mică eroarea standard a randamentului mediu anual?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('No: the mean over 10 years is fixed by the first and last price, $\\hat\\mu = \\ln(P_T/P_0)/Y$',
        'Nu: media pe 10 ani este determinată de primul și ultimul preț, $\\hat\\mu = \\ln(P_T/P_0)/Y$'),
      T('only more \\textbf{years} reduce its standard error, as $1/\\sqrt{Y}$', 'doar mai mulți \\textbf{ani} îi reduc eroarea standard, ca $1/\\sqrt{Y}$'),
      T('the volatility estimate, in contrast, does improve with higher frequency', 'estimarea volatilității, în schimb, se îmbunătățește cînd crește frecvența')])))

D.recap(('Annualisation', 'anualizarea'), [
    T('Mean $\\times\\, q$, volatility $\\times \\sqrt{q}$, with $q$ the actual observations per year', 'Media $\\times\\, q$, volatilitatea $\\times \\sqrt{q}$, cu $q$ numărul real de observații pe an'),
    T('$\\sqrt{q}$ assumes uncorrelated returns with constant variance', '$\\sqrt{q}$ presupune randamente necorelate, cu varianță constantă'),
    T('CAGR $= (P_T/P_0)^{1/Y} - 1$, the exponential of the annual mean log return minus 1', 'CAGR $= (P_T/P_0)^{1/Y} - 1$, exponențiala mediei anuale a randamentelor logaritmice minus 1'),
    T('Mean returns are imprecise: SE $= \\sigma_{\\text{year}}/\\sqrt{Y}$', 'Mediile randamentelor sînt imprecise: SE $= \\sigma_{\\text{an}}/\\sqrt{Y}$')])

# =============================================================================
# 6. STATISTICI DESCRIPTIVE
# =============================================================================
D.section('Descriptive Statistics of Returns', 'Statisticile descriptive ale randamentelor')

D.frame(T('Location and Spread', 'Poziție și împrăștiere'), items(
    T('Sample $r_1, \\dots, r_n$ of daily log returns (in \\%)', 'Eșantion $r_1, \\dots, r_n$ de randamente logaritmice zilnice (în \\%)'),
    T('\\textbf{Mean}: $\\bar r = \\frac1n \\sum_{t=1}^{n} r_t$, the average daily return', '\\textbf{Media}: $\\bar r = \\frac1n \\sum_{t=1}^{n} r_t$, randamentul zilnic mediu'),
    (T('\\textbf{Variance}: $\\hat\\sigma^2 = \\frac{1}{n-1} \\sum_{t=1}^{n} (r_t - \\bar r)^2$; \\textbf{standard deviation} $\\hat\\sigma = \\sqrt{\\hat\\sigma^2}$',
       '\\textbf{Varianța}: $\\hat\\sigma^2 = \\frac{1}{n-1} \\sum_{t=1}^{n} (r_t - \\bar r)^2$; \\textbf{abaterea standard} $\\hat\\sigma = \\sqrt{\\hat\\sigma^2}$'),
     [T('in finance the standard deviation of returns is called \\textbf{volatility}', 'în finanțe abaterea standard a randamentelor se numește \\textbf{volatilitate}')]),
    (T('\\textbf{Quantile} $q_\\alpha$: the value below which a fraction $\\alpha$ of returns lie', '\\textbf{Cuantila} $q_\\alpha$: valoarea sub care se află o fracție $\\alpha$ din randamente'),
     [T('median $q_{0.5}$; the 1\\% quantile $q_{0.01}$ is the basis of VaR 1\\% (Chapter 10)', 'mediana $q_{0.5}$; cuantila de 1\\% $q_{0.01}$ stă la baza VaR 1\\% (Capitolul 10)')]),
    T('\\textbf{Minimum} and \\textbf{maximum}: the worst and best day, always with their dates', '\\textbf{Minimul} și \\textbf{maximul}: cea mai slabă și cea mai bună zi, întotdeauna cu datele lor')))

D.frame(T('Shape: Skewness and Kurtosis', 'Forma: asimetria și boltirea'), items(
    (T('\\textbf{Skewness}: $\\hat S = \\frac1n \\sum_t (r_t - \\bar r)^3 / \\hat\\sigma^3$', '\\textbf{Asimetria} (skewness): $\\hat S = \\frac1n \\sum_t (r_t - \\bar r)^3 / \\hat\\sigma^3$'),
     [T('$S = 0$: symmetric; $S < 0$: a longer left tail, large losses more frequent than large gains',
        '$S = 0$: simetrică; $S < 0$: coadă stîngă mai lungă, pierderile mari mai frecvente decît cîștigurile mari')]),
    (T('\\textbf{Kurtosis}: $\\hat K = \\frac1n \\sum_t (r_t - \\bar r)^4 / \\hat\\sigma^4$; \\textbf{excess kurtosis} $\\hat K - 3$',
       '\\textbf{Boltirea} (kurtosis): $\\hat K = \\frac1n \\sum_t (r_t - \\bar r)^4 / \\hat\\sigma^4$; \\textbf{excesul de boltire} (excess kurtosis) $\\hat K - 3$'),
     [T('the Normal distribution has $K = 3$, so excess kurtosis 0', 'distribuția Normală are $K = 3$, deci exces de boltire 0'),
      T('excess kurtosis $> 0$: \\textbf{heavy tails}, extreme days more frequent than under the Normal distribution',
        'exces de boltire $> 0$: \\textbf{cozi groase}, zile extreme mai frecvente decît sub distribuția Normală')]),
    T('Both are sensitive to single extreme days: report them with the extremes', 'Ambele sînt sensibile la zile extreme izolate: raportați-le împreună cu extremele'),
    T('Heavy tails and asymmetry are \\textbf{stylised facts} of returns \\refCont; formal tests in Chapter 2',
      'Cozile groase și asimetria sînt \\textbf{fapte stilizate} ale randamentelor \\refCont; testele formale în Capitolul 2')))


def drow(k):
    return T(*[f'{NAMES[k]} & $@{{d.{k}.mean}}$ & $@{{d.{k}.sd}}$ & $@{{d.{k}.skew}}$ & $@{{d.{k}.exkurt}}$ & '
            f'$@{{d.{k}.min}}$ (@{{d.{k}.mindate}}) & $@{{d.{k}.max}}$'] * 2)


DHEAD = (T('Series', 'Seria') + ' & ' + T('Mean', 'Media') + ' & ' + T('Std. dev.', 'Abaterea std.') + ' & ' + T('Skew.', 'Asim.') + ' & '
         + T('Exc. kurt.', 'Exces aplat.') + ' & ' + T('Minimum (date)', 'Minimul (data)') + ' & ' + T('Maximum', 'Maximul'))

D.frame(T('Daily Log Returns: Indices and Bitcoin, @{y0}--@{y1}', 'Randamente logaritmice zilnice: indici și Bitcoin, @{y0}--@{y1}'),
        table('lrrrrlr', DHEAD, [drow(k) for k in INDICES], size='footnotesize') + items(
            T('All values in \\% except skewness and excess kurtosis; each series on its own calendar', 'Toate valorile în \\%, cu excepția asimetriei și a excesului de boltire; fiecare serie pe calendarul ei'),
            T('Bitcoin: daily standard deviation about three times that of the stock indices', 'Bitcoin: abaterea standard zilnică de aproximativ trei ori mai mare decît a indicilor bursieri'),
            T('All series: negative skewness and large excess kurtosis', 'Toate seriile: asimetrie negativă și exces de boltire mare'),
            T('The worst days cluster in March 2020 (COVID-19) and on 19 December 2018 for the BET',
              'Cele mai slabe zile se concentrează în martie 2020 (COVID-19) și, pentru BET, pe 19 decembrie 2018')) + ql('SFM_ch1_descriptive_stats'))

D.frame(T('Daily Log Returns: Five BVB Stocks, @{y0}--@{y1}', 'Randamente logaritmice zilnice: cinci acțiuni BVB, @{y0}--@{y1}'),
        table('lrrrrlr', DHEAD, [drow(k) for k in STOCKS], size='footnotesize') + items(
            T('Single stocks are more volatile than the BET index: diversification at work', 'Acțiunile individuale sînt mai volatile decît indicele BET: efectul diversificării'),
            T('Banca Transilvania and BRD lost most on 19 December 2018, after the announcement of a tax on bank assets, adopted as OUG 114/2018',
              'Banca Transilvania și BRD au pierdut cel mai mult pe 19 decembrie 2018, după anunțarea taxei pe activele bancare, adoptată prin OUG 114/2018'),
            T('Excess kurtosis differs widely between stocks: compare TLV with SNN', 'Excesul de boltire diferă mult între acțiuni: comparați TLV cu SNN'),
            T('Crashes on the BVB have their own statistics \\refPele', 'Crahurile de la BVB au o statistică proprie \\refPele')) + ql('SFM_ch1_descriptive_stats'))

chart(T('Histograms vs the Normal Distribution', 'Histogramele față de distribuția Normală'), 'sfm_ch1_hist_normal', 'SFM_ch1_descriptive_stats', [
    T('Black line: the Normal density with the same mean and standard deviation', 'Linia neagră: densitatea Normală cu aceeași medie și abatere standard'),
    T('S\\&P 500: @{h.sp500.in1}\\% of days lie within one standard deviation of the mean (Normal: @{h.n.in1}\\%)',
      'S\\&P 500: @{h.sp500.in1}\\% din zile sînt la cel mult o abatere standard de medie (distribuția Normală: @{h.n.in1}\\%)'),
    T('Beyond three standard deviations: @{h.sp500.out3}\\% of days vs @{h.n.out3}\\% under the Normal distribution, @{h.ratio} times more',
      'Dincolo de trei abateri standard: @{h.sp500.out3}\\% din zile, față de @{h.n.out3}\\% sub distribuția Normală, adică de @{h.ratio} ori mai multe')],
      h='0.55\\textheight')

chart(T('Volatility Changes over Time', 'Volatilitatea se schimbă în timp'), 'sfm_ch1_rolling_vol_1y', 'SFM_ch1_descriptive_stats', [
    T('Rolling one-year volatility, annualised: S\\&P 500 between @{rv.sp500.min}\\% and @{rv.sp500.max}\\%; Bitcoin between @{rv.btc.min}\\% and @{rv.btc.max}\\%',
      'Volatilitatea pe ferestre mobile de un an, anualizată: S\\&P 500 între @{rv.sp500.min}\\% și @{rv.sp500.max}\\%; Bitcoin între @{rv.btc.min}\\% și @{rv.btc.max}\\%'),
    T('One number for the whole sample hides calm and turbulent periods', 'O singură cifră pentru tot eșantionul ascunde perioadele calme și pe cele agitate'),
    T('Volatility estimators and GARCH models: Chapters 8 and 9', 'Estimatori de volatilitate și modele GARCH: Capitolele 8 și 9')], h='0.55\\textheight')

D.recap(('Descriptive Statistics', 'statistici descriptive'), [
    T('Report mean, standard deviation, skewness, excess kurtosis, quantiles and extremes with dates',
      'Raportați media, abaterea standard, asimetria, excesul de boltire, cuantilele și extremele cu datele lor'),
    T('Daily returns: mean near 0, volatility 1--3\\% a day, heavy tails, often negative skewness',
      'Randamente zilnice: medie apropiată de 0, volatilitate 1--3\\% pe zi, cozi groase, adesea asimetrie negativă'),
    T('Extreme days are several times more frequent than under the Normal distribution', 'Zilele extreme sînt de cîteva ori mai frecvente decît sub distribuția Normală'),
    T('Volatility is not constant over time', 'Volatilitatea nu este constantă în timp')])

# =============================================================================
# 7. INDICATORI DE PERFORMANȚĂ
# =============================================================================
D.section('Performance Indicators', 'Indicatori de performanță')

D.frame(T('Why One Number Is Not Enough', 'Limitele unui singur indicator'), items(
    T('Return alone rewards risk taking: a leveraged bet can show a high return for years',
      'Randamentul singur răsplătește asumarea de risc: un pariu cu efect de levier poate arăta un randament mare ani la rînd'),
    (T('Three questions an investor asks', 'Trei întrebări pe care le pune un investitor'),
     [T('how much did it grow? (CAGR)', 'cît a crescut? (CAGR)'),
      T('how bumpy was the ride? (volatility, downside deviation)', 'cît de mari au fost fluctuațiile? (volatilitate, abaterea negativă)'),
      T('how deep was the worst fall? (maximum drawdown)', 'cît de adîncă a fost cea mai mare cădere? (drawdown maxim)')]),
    T('\\textbf{Risk-adjusted} indicators divide a return by a risk measure: Sharpe, Sortino, Calmar',
      'Indicatorii \\textbf{ajustați la risc} împart un randament la o măsură a riscului: Sharpe, Sortino, Calmar'),
    T('All are estimates from one sample: each has an estimation error', 'Toate sînt estimări dintr-un singur eșantion: fiecare are o eroare de estimare')))

D.frame(T('The Sharpe Ratio', 'Raportul Sharpe'), cols(items(
    (T('\\textbf{Sharpe ratio} \\refSharpeA, \\refSharpeB: excess return per unit of total risk',
       '\\textbf{Raportul Sharpe} (Sharpe ratio) \\refSharpeA, \\refSharpeB: randamentul în exces pe unitatea de risc total'),
     [T('$\\text{SR} = \\dfrac{\\mu - r_f}{\\sigma}$; $\\mu$ mean return, $r_f$ risk-free rate, $\\sigma$ volatility',
        '$\\text{SR} = \\dfrac{\\mu - r_f}{\\sigma}$; $\\mu$ randamentul mediu, $r_f$ rata fără risc, $\\sigma$ volatilitatea'),
      T('from daily to annual: $\\text{SR}_{\\text{year}} = \\sqrt{q}\\,\\text{SR}_{\\text{day}}$', 'de la zilnic la anual: $\\text{SR}_{\\text{an}} = \\sqrt{q}\\,\\text{SR}_{\\text{zi}}$')]),
    (T('Here: mean of daily simple returns, $r_f = 0$', 'Aici: media randamentelor simple zilnice, $r_f = 0$'),
     [T('the assets are in RON, EUR and USD, each with its own risk-free rate', 'activele sînt în lei, euro și dolari, fiecare cu propria rată fără risc'),
      T('S\\&P 500: daily SR $@{sr.sp.d}$, annual SR $@{p.sp500.sharpe}$', 'S\\&P 500: SR zilnic $@{sr.sp.d}$, SR anual $@{p.sp500.sharpe}$')]),
    T('Same Sharpe ratio, same return per unit of risk, whatever the leverage', 'Același raport Sharpe înseamnă același randament pe unitatea de risc, oricare ar fi efectul de levier')),
    photo('ch1_sharpe_2007.jpg', T('William F. Sharpe, Nobel Prize in Economics 1990', 'William F. Sharpe, Premiul Nobel pentru economie 1990'),
          'https://commons.wikimedia.org/wiki/File:William_sharpe_2007.jpg',
          T('Photo: Larry D. Moore (2007); CC BY 4.0; Wikimedia Commons', 'Foto: Larry D. Moore (2007); CC BY 4.0; Wikimedia Commons'),
          h='0.55\\textheight'), wl='0.64', wr='0.32'))

D.frame(T('The Sharpe Ratio Is an Estimate', 'Raportul Sharpe este o estimare'), items(
    (T('Standard error for i.i.d. (independent, identically distributed) returns \\refLo', 'Eroarea standard pentru randamente i.i.d. (independente și identic distribuite) \\refLo'),
     [T('$\\text{SE}(\\widehat{\\text{SR}}) \\approx \\sqrt{(1 + \\text{SR}^2/2)/n}$ per period; annual: multiply by $\\sqrt{q}$',
        '$\\text{SE}(\\widehat{\\text{SR}}) \\approx \\sqrt{(1 + \\text{SR}^2/2)/n}$ pe perioadă; anual: înmulțiți cu $\\sqrt{q}$'),
      T('roughly $1/\\sqrt{Y}$ for the annual Sharpe ratio: about $@{se.sr.full}$ with @{se.sr.years} years of data', 'aproximativ $1/\\sqrt{Y}$ pentru raportul Sharpe anual: circa $@{se.sr.full}$ cu @{se.sr.years} ani de date')]),
    (T('S\\&P 500, @{y0}--@{y1}: $\\text{SR} = @{p.sp500.sharpe}$, SE $@{p.sp500.sharpe_se}$', 'S\\&P 500, @{y0}--@{y1}: $\\text{SR} = @{p.sp500.sharpe}$, SE $@{p.sp500.sharpe_se}$'),
     [T('95\\% CI [$@{p.sp500.sharpe_lo}$, $@{p.sp500.sharpe_hi}$]', 'intervalul de încredere 95\\%: [$@{p.sp500.sharpe_lo}$; $@{p.sp500.sharpe_hi}$]')]),
    T('Heavy tails and volatility clustering make the true error larger than this formula', 'Cozile groase și volatility clustering fac eroarea reală mai mare decît cea dată de această formulă'),
    T('Choosing the best of many strategies inflates the winner\'s Sharpe ratio \\refDSR', 'Alegerea celei mai bune dintre multe strategii supraestimează raportul Sharpe al strategiei alese \\refDSR')) + ql('SFM_ch1_performance'))

side(T('Sharpe Ratios with 95\\% Intervals, @{y0}--@{y1}', 'Raportul Sharpe cu intervale de încredere 95\\%, @{y0}--@{y1}'), 'sfm_ch1_sharpe_ci', 'SFM_ch1_performance', [
    T('Highest point estimate: @{best.sharpe}', 'Cea mai mare estimare punctuală: @{best.sharpe}'),
    T('All intervals overlap: the ranking is not statistically significant', 'Toate intervalele se suprapun: clasamentul nu este semnificativ statistic'),
    T('The DAX interval contains 0', 'Intervalul DAX conține 0'),
    T('Interval widths are almost equal: they depend mainly on the length of the sample', 'Lățimile intervalelor sînt aproape egale: depind mai ales de lungimea eșantionului')], w=0.58)

D.frame(T('The Sortino Ratio', 'Raportul Sortino'), items(
    T('Investors dislike losses, not gains: the volatility treats both the same', 'Investitorii se tem de pierderi, nu de cîștiguri; volatilitatea le tratează însă la fel'),
    (T('\\textbf{Downside deviation} below a target $\\tau$ (here $\\tau = 0$)', '\\textbf{Abaterea negativă} (downside deviation) sub un prag $\\tau$ (aici $\\tau = 0$)'),
     [T('$\\sigma_D = \\sqrt{\\frac1n \\sum_t \\min(R_t - \\tau, 0)^2}$, only days below the target count',
        '$\\sigma_D = \\sqrt{\\frac1n \\sum_t \\min(R_t - \\tau, 0)^2}$, contează doar zilele sub prag')]),
    T('\\textbf{Sortino ratio} \\refSortino: $\\text{Sortino} = \\dfrac{\\mu - \\tau}{\\sigma_D}$, annualised like the Sharpe ratio',
      '\\textbf{Raportul Sortino} (Sortino ratio) \\refSortino: $\\text{Sortino} = \\dfrac{\\mu - \\tau}{\\sigma_D}$, anualizat ca raportul Sharpe'),
    T('For a symmetric distribution $\\sigma_D \\approx \\sigma/\\sqrt2$, so Sortino $\\approx 1.4 \\times$ Sharpe',
      'Pentru o distribuție simetrică $\\sigma_D \\approx \\sigma/\\sqrt2$, deci Sortino $\\approx 1.4 \\times$ Sharpe'),
    T('S\\&P 500, @{y0}--@{y1}: Sharpe $@{p.sp500.sharpe}$, Sortino $@{p.sp500.sortino}$', 'S\\&P 500, @{y0}--@{y1}: Sharpe $@{p.sp500.sharpe}$, Sortino $@{p.sp500.sortino}$')))

D.frame(T('Drawdown and Maximum Drawdown', 'Drawdown și drawdown maxim'), items(
    (T('\\textbf{Drawdown}: the fall from the highest price reached so far', '\\textbf{Drawdown}: căderea de la cel mai mare preț atins pînă acum'),
     [T('$\\text{DD}_t = \\dfrac{P_t}{\\max_{s \\le t} P_s} - 1 \\le 0$; zero at a new maximum', '$\\text{DD}_t = \\dfrac{P_t}{\\max_{s \\le t} P_s} - 1 \\le 0$; zero la un nou maxim')]),
    (T('\\textbf{MDD} (maximum drawdown): $\\text{MDD} = \\min_t \\text{DD}_t$, the worst peak-to-trough fall',
       '\\textbf{MDD} (maximum drawdown, drawdown maxim): $\\text{MDD} = \\min_t \\text{DD}_t$, cea mai mare cădere de la vîrf la minim'),
     [T('also report the dates: peak, trough and \\textbf{recovery} (back to the old peak)', 'raportați și datele: vîrful, minimul și \\textbf{revenirea} la vechiul vîrf')]),
    (T('A loss $x$ needs a gain $x/(1 - x)$ to recover', 'O pierdere $x$ cere un cîștig $x/(1 - x)$ pentru revenire'),
     [T('10\\% needs @{need.10}\\%; 20\\% needs @{need.20}\\%; 50\\% needs @{need.50}\\%; 80\\% needs @{need.80}\\%',
        '10\\% cere @{need.10}\\%; 20\\% cere @{need.20}\\%; 50\\% cere @{need.50}\\%; 80\\% cere @{need.80}\\%')]),
    T('The MDD depends on the path and grows with the length of the sample \\refMagdon', 'MDD depinde de traiectorie și crește cu lungimea eșantionului \\refMagdon')))

chart(T('Drawdowns over the Full History', 'Drawdown-uri pe întreaga istorie'), 'sfm_ch1_drawdowns', 'SFM_ch1_performance', [
    T('S\\&P 500: MDD $@{dd.sp500.mdd}\\%$, @{dd.sp500.peak} to @{dd.sp500.trough}, recovered on @{dd.sp500.rec}',
      'S\\&P 500: MDD $@{dd.sp500.mdd}\\%$, de la @{dd.sp500.peak} la @{dd.sp500.trough}, cu revenire pe @{dd.sp500.rec}'),
    T('BET (since @{dd.bet.first}): MDD $@{dd.bet.mdd}\\%$, @{dd.bet.peak} to @{dd.bet.trough}, recovered only on @{dd.bet.rec}',
      'BET (din @{dd.bet.first}): MDD $@{dd.bet.mdd}\\%$, de la @{dd.bet.peak} la @{dd.bet.trough}, cu revenire abia pe @{dd.bet.rec}'),
    T('Bitcoin: MDD $@{dd.btc.mdd}\\%$, @{dd.btc.peak} to @{dd.btc.trough}', 'Bitcoin: MDD $@{dd.btc.mdd}\\%$, de la @{dd.btc.peak} la @{dd.btc.trough}')],
      h='0.55\\textheight')

D.frame(T('Reading the Drawdown Chart', 'Interpretarea graficului drawdown-urilor'), items(
    (T('Depth', 'Adîncimea'),
     [T('after a drawdown of $@{dd.bet.mdd}\\%$, the BET needed $+@{dd.bet.need}\\%$ to recover; Bitcoin, $+@{dd.btc.need}\\%$',
        'după un drawdown de $@{dd.bet.mdd}\\%$, BET a avut nevoie de $+@{dd.bet.need}\\%$ pentru revenire; Bitcoin, de $+@{dd.btc.need}\\%$')]),
    (T('Duration', 'Durata'),
     [T('the BET needed @{dd.bet.years} years to regain its 2007 peak', 'BET a avut nevoie de @{dd.bet.years} ani ca să revină la vîrful din 2007'),
      T('share of days more than 20\\% below the peak: S\\&P 500 @{dd.sp500.under}\\%, BET @{dd.bet.under}\\%, Bitcoin @{dd.btc.under}\\%',
        'ponderea zilelor cu peste 20\\% sub vîrf: S\\&P 500 @{dd.sp500.under}\\%, BET @{dd.bet.under}\\%, Bitcoin @{dd.btc.under}\\%')]),
    T('Volatility and Sharpe ratio ignore the order of returns; the drawdown does not',
      'Volatilitatea și raportul Sharpe ignoră ordinea randamentelor; drawdown-ul ține cont de ea'),
    T('In the comparison period @{y0}--@{y1}, the MDD of each stock index happened in February--March 2020; Bitcoin and several BVB stocks had theirs at other dates',
      'În perioada de comparație @{y0}--@{y1}, MDD-ul fiecărui indice bursier s-a produs în februarie--martie 2020; pentru Bitcoin și pentru mai multe acțiuni BVB, MDD-ul s-a produs la alte date')))

D.frame(T('The Calmar Ratio', 'Raportul Calmar'), items(
    T('\\textbf{Calmar ratio}: growth per unit of the worst fall, $\\text{Calmar} = \\dfrac{\\text{CAGR}}{|\\text{MDD}|}$',
      '\\textbf{Raportul Calmar} (Calmar ratio): creșterea pe unitatea celei mai mari căderi, $\\text{Calmar} = \\dfrac{\\text{CAGR}}{|\\text{MDD}|}$'),
    T('Popular with fund managers: the MDD is the loss an investor actually lives through', 'Popular printre administratorii de fonduri: MDD este pierderea pe care investitorul o suportă efectiv'),
    (T('S\\&P 500, @{y0}--@{y1}: CAGR @{p.sp500.cagr}\\%, MDD $@{p.sp500.mdd}\\%$', 'S\\&P 500, @{y0}--@{y1}: CAGR @{p.sp500.cagr}\\%, MDD $@{p.sp500.mdd}\\%$'),
     [T('Calmar $= @{p.sp500.cagr}/@{sp.absmdd} = @{p.sp500.calmar}$', 'Calmar $= @{p.sp500.cagr}/@{sp.absmdd} = @{p.sp500.calmar}$')]),
    T('A single event decides the denominator: the Calmar ratio is even noisier than the Sharpe ratio',
      'Un singur eveniment determină numitorul: raportul Calmar este și mai zgomotos decît raportul Sharpe'),
    T('Compare Calmar ratios only over the same period', 'Comparați rapoartele Calmar doar pe aceeași perioadă')))

D.frame(T('Volatility Drag', 'Volatility drag'), items(
    (T('For a log return with mean $m$ and variance $\\sigma^2$, the simple return has mean $\\approx m + \\sigma^2/2$', 'Pentru un randament logaritmic cu media $m$ și varianța $\\sigma^2$, randamentul simplu are media $\\approx m + \\sigma^2/2$'),
     [T('so the growth rate is $\\mu_{\\text{log}} \\approx \\mu_{\\text{arith}} - \\sigma^2/2$', 'deci rata de creștere este $\\mu_{\\text{log}} \\approx \\mu_{\\text{arith}} - \\sigma^2/2$'),
      T('the gap $\\sigma^2/2$ is the \\textbf{volatility drag} \\refBoothFama; it comes from the concavity of $\\ln$ (Jensen\'s inequality)',
        'diferența $\\sigma^2/2$ este \\textbf{volatility drag} \\refBoothFama; provine din concavitatea lui $\\ln$ (inegalitatea lui Jensen)')]),
    (T('Two assets with the same arithmetic mean, 10\\% a year', 'Două active cu aceeași medie aritmetică, 10\\% pe an'),
     [T('$\\sigma = 5\\%$: drag @{vd.a.drag}\\%, growth @{vd.a.g}\\% a year; after 10 years 1 leu becomes @{vd.a.w10}',
        '$\\sigma = 5\\%$: drag @{vd.a.drag}\\%, creștere @{vd.a.g}\\% pe an; după 10 ani 1 leu devine @{vd.a.w10}'),
      T('$\\sigma = 30\\%$: drag @{vd.b.drag}\\%, growth @{vd.b.g}\\% a year; after 10 years 1 leu becomes @{vd.b.w10}',
        '$\\sigma = 30\\%$: drag @{vd.b.drag}\\%, creștere @{vd.b.g}\\% pe an; după 10 ani 1 leu devine @{vd.b.w10}')]),
    T('Doubling the volatility quadruples the drag', 'Dublarea volatilității mărește de patru ori volatility drag')))

side(T('Volatility Drag on Real Data', 'Volatility drag pe date reale'), 'sfm_ch1_vol_drag', 'SFM_ch1_volatility_drag', [
    T('Each point: one asset, @{y0}--@{y1}', 'Fiecare punct: un activ, @{y0}--@{y1}'),
    T('Vertical: arithmetic minus log annual mean; horizontal: $\\sigma^2/2$', 'Vertical: media anuală aritmetică minus cea logaritmică; orizontal: $\\sigma^2/2$'),
    T('All points lie on the line: the formula works on real data', 'Toate punctele se află pe dreaptă: formula se verifică pe date reale'),
    T('Bitcoin: arithmetic mean @{btc.ma}\\%, log mean @{btc.ml}\\%, drag @{btc.drag} percentage points; $\\sigma^2/2 = @{btc.halfvar}\\%$',
      'Bitcoin: media aritmetică @{btc.ma}\\%, media logaritmică @{btc.ml}\\%, drag @{btc.drag} puncte procentuale; $\\sigma^2/2 = @{btc.halfvar}\\%$')], w=0.56)

D.frame(T('Question for the Room: Which Bitcoin Return?', 'Întrebare pentru sală: ce randament pentru Bitcoin?'), items(
    T('Bitcoin, @{y0}--@{y1}: arithmetic mean of simple returns @{btc.ma}\\% a year; CAGR @{btc.cagr}\\%',
      'Bitcoin, @{y0}--@{y1}: media aritmetică a randamentelor simple @{btc.ma}\\% pe an; CAGR @{btc.cagr}\\%'),
    T('\\textbf{What do you think?} Which number describes the growth of an investor\'s wealth?',
      '\\textbf{Ce credeți?} Care dintre cele două cifre descrie creșterea averii unui investitor?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('the CAGR: it is the rate that actually turned the first price into the last', 'CAGR: este rata care a transformat efectiv primul preț în ultimul'),
      T('the arithmetic mean is the expected return of a single year, not the growth over many years',
        'media aritmetică este randamentul așteptat al unui singur an, nu creșterea pe mulți ani'),
      T('the gap is the volatility drag; for low-volatility assets it is small', 'diferența este volatility drag, neglijabil pentru activele cu volatilitate mică')])))


def prow(k):
    return (f'{NAMES[k]} & $@{{p.{k}.cagr}}$ & $@{{p.{k}.vol}}$ & $@{{p.{k}.sharpe}}$ & $@{{p.{k}.sortino}}$ & '
            f'$@{{p.{k}.mdd}}$ & $@{{p.{k}.calmar}}$ & $@{{p.{k}.drag}}$')


PHEAD = ('& CAGR \\% & ' + T('Vol.', 'Vol.') + ' \\% & Sharpe & Sortino & MDD \\% & Calmar & Drag \\%')

D.frame(T('Performance Indicators: Indices and Bitcoin, @{y0}--@{y1}', 'Indicatori de performanță: indici și Bitcoin, @{y0}--@{y1}'),
        table('lrrrrrrr', PHEAD, [prow(k) for k in INDICES], size='footnotesize') + items(
            T('Annualised with the actual frequency of each series; Sharpe and Sortino with $r_f = 0$', 'Anualizate cu frecvența reală a fiecărei serii; Sharpe și Sortino cu $r_f = 0$'),
            T('Bitcoin: by far the highest CAGR, the highest volatility and the deepest drawdown', 'Bitcoin: de departe cel mai mare CAGR, cea mai mare volatilitate și cel mai adînc drawdown'),
            T('BET-TR vs BET: same risk, dividends lift every return-based indicator', 'BET-TR față de BET: același risc, dar dividendele cresc toți indicatorii calculați pe baza randamentului')) + ql('SFM_ch1_performance'))

D.frame(T('Performance Indicators: BVB Stocks, @{y0}--@{y1}', 'Indicatori de performanță: acțiuni BVB, @{y0}--@{y1}'),
        table('lrrrrrrr', PHEAD, [prow(k) for k in STOCKS], size='footnotesize') + items(
            T('Adjusted prices: dividends included', 'Prețuri ajustate: dividendele sînt incluse'),
            T('Single stocks: higher volatility and deeper drawdowns than the BET index', 'Acțiunile individuale: volatilitate mai mare și drawdown-uri mai adînci decît indicele BET'),
            T('Remember the standard error, about $@{p.tlv.sharpe_se}$ for each Sharpe ratio: small differences are not significant',
              'Nu uitați eroarea standard, de circa $@{p.tlv.sharpe_se}$ pentru fiecare raport Sharpe: diferențele mici nu sînt semnificative')) + ql('SFM_ch1_performance'))

side(T('Risk and Return, @{y0}--@{y1}', 'Risc și randament, @{y0}--@{y1}'), 'sfm_ch1_risk_return', 'SFM_ch1_performance', [
    T('Horizontal: annualised volatility; vertical: CAGR', 'Orizontal: volatilitatea anualizată; vertical: CAGR'),
    T('Best by CAGR: @{best.cagr}; by Sharpe: @{best.sharpe}; by Sortino: @{best.sortino}; by Calmar: @{best.calmar}',
      'Cel mai bun după CAGR: @{best.cagr}; după Sharpe: @{best.sharpe}; după Sortino: @{best.sortino}; după Calmar: @{best.calmar}'),
    T('Lowest volatility: @{best.vol}; shallowest MDD: @{best.mdd}', 'Cea mai mică volatilitate: @{best.vol}; cel mai mic MDD în valoare absolută: @{best.mdd}'),
    T('The ranking depends on the indicator: always say which one you use', 'Clasamentul depinde de indicator: precizați întotdeauna indicatorul folosit')], w=0.56)

D.recap(('Performance Indicators', 'indicatori de performanță'), [
    T('CAGR (growth), volatility (risk), Sharpe and Sortino (return per unit of risk)', 'CAGR (creștere), volatilitate (risc), Sharpe și Sortino (randament pe unitatea de risc)'),
    T('MDD and Calmar: the worst fall, which the investor actually lives through', 'MDD și Calmar: cea mai mare cădere, pe care investitorul o suportă efectiv'),
    T('Volatility drag: $\\mu_{\\text{log}} \\approx \\mu_{\\text{arith}} - \\sigma^2/2$, growth is below the arithmetic mean', 'Volatility drag: $\\mu_{\\text{log}} \\approx \\mu_{\\text{arith}} - \\sigma^2/2$, creșterea este sub media aritmetică'),
    T('Every indicator is an estimate: compare differences with their standard errors', 'Fiecare indicator este o estimare: comparați diferențele cu erorile lor standard')])

# =============================================================================
# 8. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Do Sharpe ratios persist?}', '\\textbf{Persistă în timp rapoartele Sharpe?}'),
     [T('if a BVB stock had a high Sharpe ratio in 2015--2020, did it also have one in 2021--2026?',
        'o acțiune BVB cu un raport Sharpe mare în 2015--2020 a avut un raport Sharpe mare și în 2021--2026?'),
      T('if not, ranking funds or stocks by past Sharpe ratios is of little use to an investor',
        'dacă nu, clasamentul fondurilor sau al acțiunilor după raportul Sharpe din trecut îi este de puțin folos unui investitor')]),
    T('Why it is open: short samples, few stocks, and a standard error of about $@{se.sr.half}$ for a Sharpe ratio estimated on half of our sample',
      'De ce este deschisă: eșantioane scurte, puține acțiuni și o eroare standard de circa $@{se.sr.half}$ pentru un raport Sharpe estimat pe jumătate din eșantionul nostru'),
    T('Related evidence: many published return patterns do not survive new data \\refHLZ', 'Rezultate conexe: multe regularități ale randamentelor, publicate în literatură, nu se confirmă pe date noi \\refHLZ'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: list papers on the persistence of performance and summarise their methods',
      '\\textbf{Literatura}: lista lucrărilor despre persistența performanței și rezumarea metodelor lor'),
    T('\\textbf{Code}: draft a Python function for rolling Sharpe ratios on the course data',
      '\\textbf{Cod}: o primă versiune a unei funcții Python pentru raportul Sharpe pe ferestre mobile, pe datele cursului'),
    T('\\textbf{Robustness}: propose other windows, other indicators (Sortino, Calmar), other markets',
      '\\textbf{Robustețe}: propunerea altor ferestre, altor indicatori (Sortino, Calmar), altor piețe'),
    T('\\textbf{Explanation}: turn a table of results into a first draft of the interpretation',
      '\\textbf{Explicație}: transformarea unui tabel de rezultate într-o primă versiune a interpretării'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that computes the annualised Sharpe ratio of daily simple returns on non-overlapping 3-year windows, with the Lo (2002) standard error.}',
        '\\aiprompt{Write a Python function that computes the annualised Sharpe ratio of daily simple returns on non-overlapping 3-year windows, with the Lo (2002) standard error.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Data: adjusted close for stocks, the same calendar for all assets, the right period',
      'Datele: prețul ajustat pentru acțiuni, același calendar pentru toate activele, perioada corectă'),
    T('Formulas: the annualisation factor ($\\sqrt{252}$ vs $\\sqrt{365}$), simple vs log returns, $n$ vs $n - 1$',
      'Formulele: factorul de anualizare ($\\sqrt{252}$ sau $\\sqrt{365}$), randamente simple sau logaritmice, $n$ sau $n - 1$'),
    T('Numbers: recompute two or three values on small examples or with a second method',
      'Cifrele: recalculați două sau trei valori pe exemple mici sau cu o a doua metodă'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul'),
    T('Selection: if you tried ten windows, say so; a backtest tried many times overfits \\refBBLZ',
      'Selecția: dacă ați încercat zece ferestre, precizați acest lucru; un backtest repetat de multe ori duce la supraajustare (overfitting) \\refBBLZ')))

D.frame(T('Project Seed', 'Idee de proiect'), items(
    (T('\\textbf{Question}: do Sharpe ratios of BVB stocks persist from one period to the next?', '\\textbf{Întrebarea}: persistă rapoartele Sharpe ale acțiunilor BVB de la o perioadă la alta?'),
     [T('data: adjusted closes of 10--15 BVB stocks, 2010--2026, from the course data', 'date: prețurile ajustate a 10--15 acțiuni BVB, 2010--2026, din datele cursului')]),
    (T('Steps', 'Pași'),
     [T('split the sample into two halves; compute Sharpe ratios and their standard errors in each',
        'împărțiți eșantionul în două jumătăți; calculați rapoartele Sharpe și erorile lor standard în fiecare jumătate'),
      T('measure persistence with the rank correlation between the two halves', 'măsurați persistența cu corelația rangurilor dintre cele două jumătăți'),
      T('repeat with Sortino and Calmar ratios', 'repetați cu rapoartele Sortino și Calmar')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show',
      'Livrabile: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a AI și enumerați erorile AI pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei principale'), items(
    T('Check the data first: source, adjusted prices, calendar, jumps', 'Verificați întîi datele: sursa, prețurile ajustate, calendarul, salturile'),
    T('Log returns add over time; simple returns add across assets', 'Randamentele logaritmice se adună în timp; cele simple se adună între active'),
    T('Annualise with the actual frequency: mean $\\times\\,q$, volatility $\\times\\sqrt{q}$', 'Anualizați cu frecvența reală: media $\\times\\,q$, volatilitatea $\\times\\sqrt{q}$'),
    T('Returns have heavy tails, negative skewness and changing volatility', 'Randamentele au cozi groase, asimetrie negativă și volatilitate variabilă'),
    T('Judge performance with several indicators: CAGR, volatility, Sharpe, Sortino, MDD, Calmar', 'Judecați performanța cu mai mulți indicatori: CAGR, volatilitate, Sharpe, Sortino, MDD, Calmar'),
    T('Volatility costs growth: $\\sigma^2/2$ a year', 'Volatilitatea reduce creșterea cu $\\sigma^2/2$ pe an'),
    T('Mean returns and Sharpe ratios are imprecise: report standard errors', 'Randamentele medii și rapoartele Sharpe sînt imprecise: raportați erorile standard')))

D.frame(T('Key Formulas', 'Formule de reținut'), table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Simple return', 'Randament simplu') + ' & $R_t = P_t/P_{t-1} - 1$',
     T('Log return', 'Randament logaritmic') + ' & $r_t = \\ln(P_t/P_{t-1}) = \\ln(1 + R_t)$',
     T('$k$-period returns', 'Randamente pe $k$ perioade') + ' & $1 + R_t(k) = \\prod_{j=0}^{k-1}(1 + R_{t-j})$, \\quad $r_t(k) = \\sum_{j=0}^{k-1} r_{t-j}$',
     T('Portfolio', 'Portofoliu') + ' & $R_p = \\sum_i w_i R_i$',
     T('Annualisation & $\\mu_{\\text{year}} = q\\,\\mu$, \\quad $\\sigma_{\\text{year}} = \\sqrt{q}\\,\\sigma$', 'Anualizare & $\\mu_{\\text{an}} = q\\,\\mu$, \\quad $\\sigma_{\\text{an}} = \\sqrt{q}\\,\\sigma$'),
     'CAGR & $(P_T/P_0)^{1/Y} - 1$',
     'Sharpe, Sortino & $(\\mu - r_f)/\\sigma$, \\quad $(\\mu - \\tau)/\\sigma_D$',
     'MDD, Calmar & $\\min_t \\big(P_t / \\max_{s \\le t} P_s - 1\\big)$, \\quad $\\text{CAGR}/|\\text{MDD}|$',
     'Volatility drag & $\\mu_{\\text{log}} \\approx \\mu_{\\text{arith}} - \\sigma^2/2$',
     T('SE of mean, of Sharpe & $\\sigma_{\\text{year}}/\\sqrt{Y}$, \\quad $\\sqrt{(1 + \\text{SR}^2/2)/n}$', 'SE a mediei și a raportului Sharpe & $\\sigma_{\\text{an}}/\\sqrt{Y}$, \\quad $\\sqrt{(1 + \\text{SR}^2/2)/n}$')],
    size='footnotesize'))

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a fund reports daily log returns with mean 0.05\\% and standard deviation 1\\%; what are its annual mean and volatility?',
       '\\textbf{Întrebare}: un fond raportează randamente logaritmice zilnice cu media 0,05\\% și abaterea standard 1\\%; care sînt valorile anuale ale mediei și volatilității?'),
     [T('\\textbf{Answer}: $252 \\times 0.05\\% = @{cy.mu}\\%$ and $\\sqrt{252} \\times 1\\% = @{cy.vol}\\%$', '\\textbf{Răspuns}: $252 \\times 0.05\\% = @{cy.mu}\\%$ și $\\sqrt{252} \\times 1\\% = @{cy.vol}\\%$')]),
    (T('\\textbf{Question}: why is the arithmetic mean return of Bitcoin far above its CAGR?', '\\textbf{Întrebare}: de ce este media aritmetică a randamentelor Bitcoin mult peste CAGR?'),
     [T('\\textbf{Answer}: the volatility drag $\\sigma^2/2$ is large when $\\sigma$ is large', '\\textbf{Răspuns}: volatility drag $\\sigma^2/2$ este mare cînd $\\sigma$ este mare')]),
    (T('\\textbf{Question}: an index falls by 60\\%; what gain does it need to recover?', '\\textbf{Întrebare}: un indice scade cu 60\\%; ce cîștig este necesar pentru revenirea la nivelul anterior?'),
     [T('\\textbf{Answer}: $0.6/0.4 = @{cy.need}\\%$', '\\textbf{Răspuns}: $0.6/0.4 = @{cy.need}\\%$')]),
    T('Next: Chapter 2, classical distributions and stylised facts of returns', 'Urmează: Capitolul 2, distribuții clasice și fapte stilizate ale randamentelor')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
