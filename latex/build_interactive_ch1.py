"""Build the interactive pages of Chapter 1: interactive/ch1.html (EN) and interactive/ch1_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or real-data example (XBLOCKS), in the
order of the slides: the slide on the left, its interactive version on the right, with live tables and worked examples.
Logic in interactive/ch1.js, inputs in interactive/ch1_data.js (Quantlets/Ch_01/export_interactive.py), slide images in
interactive/slides/. On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and
ch1.js moves each one to its own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 1
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_01'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('tlv-adjusted', 'sfm_ch1_tlv_adjusted', 'data'), ('bet-bettr', 'sfm_ch1_bet_bettr', 'data'), ('eurron', 'sfm_ch1_eurron_check', 'data'),
          ('price-returns', 'sfm_ch1_price_returns', 'ret'), ('simple-log', 'sfm_ch1_simple_log', 'sl'), ('growth', 'sfm_ch1_growth', 'mp'),
          ('hist-normal', 'sfm_ch1_hist_normal', 'desc'), ('rolling-vol', 'sfm_ch1_rolling_vol_1y', 'desc'),
          ('sharpe-ci', 'sfm_ch1_sharpe_ci', 'perf'), ('drawdowns', 'sfm_ch1_drawdowns', 'perf'), ('vol-drag', 'sfm_ch1_vol_drag', 'perf'),
          ('risk-return', 'sfm_ch1_risk_return', 'perf')]

# table blocks: (anchor, frame number in the deck (RO = EN), section key, chart block that carries it on the web page, title check)
XBLOCKS = [('portfolio', 36, 'mp', 'growth', 'TLV'), ('annualise', 38, 'ann', 'growth', 'nuali'), ('mean-se', 40, 'ann', 'growth', 'Preci'),
           ('desc-indices', 45, 'desc', 'hist-normal', 'Bitcoin'), ('desc-bvb', 46, 'desc', 'hist-normal', 'BVB'),
           ('perf-indices', 62, 'perf', 'risk-return', 'Bitcoin'), ('perf-bvb', 63, 'perf', 'risk-return', 'BVB')]
ORDER = ['tlv-adjusted', 'bet-bettr', 'eurron', 'price-returns', 'simple-log', 'growth', 'portfolio', 'annualise', 'mean-se', 'desc-indices', 'desc-bvb',
         'hist-normal', 'rolling-vol', 'sharpe-ci', 'drawdowns', 'vol-drag', 'perf-indices', 'perf-bvb', 'risk-return']

PFX = {'tlv-adjusted': 'tl', 'bet-bettr': 'bb', 'eurron': 'eu', 'price-returns': 'pr', 'simple-log': 'sl', 'growth': 'gr', 'hist-normal': 'hi',
       'rolling-vol': 'rv', 'sharpe-ci': 'sc', 'drawdowns': 'dd', 'vol-drag': 'vd', 'risk-return': 'rr',
       'portfolio': 'pf', 'annualise': 'an', 'mean-se': 'ms', 'desc-indices': 'di', 'desc-bvb': 'db', 'perf-indices': 'pi', 'perf-bvb': 'pb'}

SECTIONS = {
 'ro': dict(data='Date financiare: surse și calitate', ret='De la prețuri la randamente', sl='Randamente simple și logaritmice',
            mp='Randamente pe mai multe perioade și ale portofoliilor', ann='Anualizarea', desc='Statisticile descriptive ale randamentelor', perf='Indicatori de performanță'),
 'en': dict(data='Financial Data: Sources and Quality', ret='From Prices to Returns', sl='Simple and Log Returns',
            mp='Multi-Period and Portfolio Returns', ann='Annualisation', desc='Descriptive Statistics of Returns', perf='Performance Indicators'),
}

L = {
 'ro': dict(file='ch1_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 1 · Date, randamente și indicatori', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; tabelele și exemplele lucrate se recalculează odată cu el.',
   src='Prețuri zilnice EODHD pînă la 18 septembrie 2026: indicii și Bitcoin la închidere, acțiunile BVB ajustate pentru dividende și operațiuni corporative; '
       'cursul de referință EUR/RON al BNR. Perioada de comparație: din 2015; S&P 500 din 2000, BET din 1997 și Bitcoin din 2014 pe istoria completă. '
       'Randamentele săptămînale și lunare (ultimul preț din fiecare săptămînă sau lună), o rată fără risc diferită de 0 și varianta TLV fără zilele de 30 și 31 mai 2016 '
       'sînt calcule suplimentare pe aceleași prețuri.'),
 'en': dict(file='ch1.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 1 · Data, Returns and Indicators', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; the tables and worked examples are recomputed with it.',
   src='EODHD daily prices up to 18 September 2026: indices and Bitcoin at the close, BVB stocks adjusted for dividends and corporate actions; '
       'the BNR EUR/RON reference rate. Comparison period: from 2015; S&P 500 from 2000, BET from 1997 and Bitcoin from 2014 over the full history. '
       'Weekly and monthly returns (the last price of each week or month), a risk-free rate other than 0 and the TLV variant without the days of 30 and 31 May 2016 '
       'are extra computations on the same prices.'),
}

YEARS = [(str(y), str(y)) for y in range(2015, 2025)]


CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt .bad{color:#CD0000}
.xt tr.sel td{background:#EEF3FB}.xt .pv{font-size:.76rem;color:#1F2A44}
.xh{font-weight:600;font-size:.92rem;color:#1F2A44;margin:.9rem 0 .1rem}
.we{border:1px solid #C5D2E8;border-left:4px solid #1A3A6E;border-radius:6px;padding:.55rem .8rem;margin:.8rem 0 .4rem;background:#F6F8FC}
.we-h{font-weight:600;margin:0 0 .2rem;color:#1A3A6E}
.we-f{overflow-x:auto;overflow-y:hidden;font-size:.95rem}
.we ol{margin:.3rem 0 0 1.3rem;padding:0;font-size:.92rem}.we li{margin:.18rem 0}
.tw.veiled-chart,.we.veiled-chart{filter:blur(10px);opacity:.35;user-select:none}
.xhold{display:none}
.wctl{display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:flex-end;margin:.3rem 0 .2rem;max-width:420px}
</style>"""


def tw(i):
    return f'<div class="tw" id="{i}"></div>'


def xh(t):
    return f'<p class="xh">{t}</p>'


def we(i, title, tex):
    return f'<div class="we" id="{i}-box"><p class="we-h">{title}</p><div class="we-f">\\[{tex}\\]</div><ol id="{i}"></ol></div>'


WE_TEX = {
 'sl': r'r_t = \ln\frac{P_t}{P_{t-1}} = \ln(1+R_t) = R_t - \frac{R_t^2}{2} + \frac{R_t^3}{3} - \dots',
 'gr': r'W_T = W_0\prod_{t=1}^{T}(1+R_t) = W_0\exp\Big(\sum_{t=1}^{T} r_t\Big), \qquad \text{CAGR} = \Big(\frac{P_T}{P_0}\Big)^{1/Y} - 1',
 'pf': r'R_{p,t} = w\,R_{1,t} + (1-w)\,R_{2,t}, \qquad r_{p,t} = \ln(1+R_{p,t}) \neq w\,r_{1,t} + (1-w)\,r_{2,t}',
 'an': r'\mu_{\text{year}} = A\,\mu, \qquad \sigma_{\text{year}} = \sqrt{A}\,\sigma, \qquad A = n/Y',
 'ms': r'\text{SE}(\hat\mu_{\text{year}}) = \frac{\sigma_{\text{year}}}{\sqrt{Y}}, \qquad \hat\mu_{\text{year}} \pm z\,\text{SE}',
 'dd': r'\text{DD}_t = \frac{P_t}{\max_{s \le t} P_s} - 1, \qquad \text{MDD} = \min_t \text{DD}_t, \qquad g = \frac{x}{1-x},\ x = |\text{MDD}|',
 'vd': r'\mu_{\text{log}} \approx \mu_{\text{arith}} - \frac{\sigma^2}{2}, \qquad \text{CAGR} = e^{\mu_{\text{log}}} - 1',
 'pi': r'\text{SR} = \frac{\mu - r_f}{\sigma}\sqrt{A}, \qquad \text{SE}(\widehat{\text{SR}}) \approx \sqrt{\frac{1 + \text{SR}_d^2/2}{n}}\,\sqrt{A}',
 'pb': r'\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}, \qquad \text{CAGR} = \Big(\frac{P_T}{P_0}\Big)^{1/Y} - 1',
}


def widgets_all(lang):
    ro = lang == 'ro'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    wt = lambda k, r, e: we(k + '-we', f'{WE}: {r if ro else e}', WE_TEX[k].replace(r'\text{year}', r'\text{an}') if ro else WE_TEX[k])
    A10 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('bettr', 'BET-TR'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'),
           ('snp', 'OMV Petrom (SNP)'), ('brd', 'BRD'), ('snn', 'Nuclearelectrica (SNN)'), ('tgn', 'Transgaz (TGN)')]
    I5, B5 = A10[:5], A10[5:]
    FQ = [('d', 'zilnică (slide)' if ro else 'daily (slide)'), ('w', 'săptămînală' if ro else 'weekly'), ('m', 'lunară' if ro else 'monthly')]
    ser = 'Seria exemplului lucrat' if ro else 'Series of the worked example'
    yr = lambda i: slider(i, 'Eșantionul începe în' if ro else 'Sample starts in', 2015, 2024, 1, 2015)
    fq = lambda i: select(i, 'Frecvența randamentelor' if ro else 'Return frequency', FQ, 'd')
    rf = lambda i: slider(i, 'Rata fără risc r<sub>f</sub> (% pe an; slide: 0)' if ro else 'Risk-free rate r<sub>f</sub> (% a year; slide: 0)', 0, 5, 0.25, 0)
    days = [('btc:max', 'Bitcoin, 7 decembrie 2017' if ro else 'Bitcoin, 7 December 2017'), ('btc:min', 'Bitcoin, 12 martie 2020' if ro else 'Bitcoin, 12 March 2020'),
            ('sp500:max', 'S&P 500, 13 octombrie 2008' if ro else 'S&P 500, 13 October 2008'), ('sp500:min', 'S&P 500, 16 martie 2020' if ro else 'S&P 500, 16 March 2020')]
    W = widgets_old(lang)
    W['tlv-adjusted'] = (W['tlv-adjusted'][0], CSS + W['tlv-adjusted'][1])
    W['simple-log'] = (W['simple-log'][0] + select('sl-d', 'Ziua exemplului lucrat' if ro else 'Day of the worked example', days, 'btc:max'),
                       W['simple-log'][1] + wt('sl', 'randamentul simplu și randamentul logaritmic al unei perechi de prețuri reale', 'simple and log return of a real price pair'))
    W['growth'] = (W['growth'][0] + select('gr-a', ser, [I5[0], I5[1], I5[2], I5[4]], 'sp500'),
                   W['growth'][1] + wt('gr', 'compunerea pe mai multe perioade', 'compounding over many periods'))
    W['drawdowns'] = (W['drawdowns'][0] + select('dd-a', ser, [I5[0], I5[2], I5[4]], 'sp500'),
                      W['drawdowns'][1] + wt('dd', 'drawdown-ul maxim al unei traiectorii reale de prețuri', 'the maximum drawdown of a real price path'))
    W['vol-drag'] = (W['vol-drag'][0] + select('vd-a', ser, A10, 'btc'),
                     W['vol-drag'][1] + wt('vd', 'media aritmetică minus media logaritmică, 2015–2026', 'arithmetic minus log mean, 2015–2026'))
    W.update({
     'portfolio': (slider('pf-w', 'Ponderea Banca Transilvania (TLV); restul OMV Petrom (SNP)' if ro else 'Weight of Banca Transilvania (TLV); the rest OMV Petrom (SNP)', 10, 90, 5, 50) + play('pf'),
                   charts('pf-chart') + xh('Randamente cumulate și creșterea anuală, 2015–2026' if ro else 'Cumulative returns and annual growth, 2015–2026') + tw('pf-tab')
                   + wt('pf', 'randamentul portofoliului în ziua cu cea mai mare diferență', 'the portfolio return on the day with the largest gap')),
     'annualise': (yr('an-y') + select('an-f', 'Frecvența exemplului lucrat (pe grafic, alături de cea zilnică)' if ro else 'Frequency of the worked example (on the chart, next to the daily one)', FQ, 'd')
                   + select('an-a', ser, A10, 'sp500') + play('an'),
                   charts('an-chart') + xh('Volatilitatea anualizată (%) după frecvența randamentelor logaritmice' if ro else 'Annualised volatility (%) by frequency of the log returns') + tw('an-tab')
                   + wt('an', 'anualizarea mediei și a volatilității', 'annualising the mean and the volatility')),
     'mean-se': (yr('ms-y') + select('ms-cl', 'Nivelul de încredere' if ro else 'Confidence level', [('90', '90%'), ('95', '95%'), ('99', '99%')], '95')
                 + select('ms-a', ser, A10, 'sp500') + play('ms'),
                 charts('ms-chart') + xh('Media anuală a randamentelor logaritmice zilnice și precizia ei' if ro else 'Annual mean of the daily log returns and its precision') + tw('ms-tab')
                 + wt('ms', 'eroarea standard a mediei anuale', 'the standard error of the annual mean')),
     'desc-indices': (yr('di-y') + fq('di-f') + play('di'),
                      xh('Randamente logaritmice (%), fiecare serie pe propriul calendar' if ro else 'Log returns (%), each series on its own calendar') + tw('di-tab')),
     'desc-bvb': (yr('db-y') + fq('db-f') + select('db-t', 'Banca Transilvania (TLV), randamente zilnice' if ro else 'Banca Transilvania (TLV), daily returns',
                  [('raw', 'cu 30 și 31 mai 2016 (slide)' if ro else 'with 30 and 31 May 2016 (slide)'), ('clean', 'fără 30 și 31 mai 2016' if ro else 'without 30 and 31 May 2016')], 'raw') + play('db'),
                  xh('Randamente logaritmice (%), prețuri ajustate' if ro else 'Log returns (%), adjusted prices') + tw('db-tab')),
     'perf-indices': (yr('pi-y') + rf('pi-rf') + select('pi-a', ser, I5, 'sp500') + play('pi'),
                      xh('Indicatori de performanță; Sharpe cu eroarea standard în paranteză' if ro else 'Performance indicators; Sharpe with its standard error in brackets') + tw('pi-tab')
                      + wt('pi', 'raportul Sharpe și eroarea lui standard', 'the Sharpe ratio and its standard error')),
     'perf-bvb': (yr('pb-y') + select('pb-a', ser, B5, 'tlv') + play('pb'),
                  xh('Indicatori de performanță; Sharpe cu eroarea standard în paranteză' if ro else 'Performance indicators; Sharpe with its standard error in brackets')
                  + f'<div class="wctl">{rf("pb-rf")}</div>' + tw('pb-tab')
                  + xh('Durata episodului de drawdown maxim, toate cele zece serii' if ro else 'Duration of the maximum-drawdown episode, all ten series') + charts('pb-chart')
                  + wt('pb', 'drawdown-ul maxim și raportul Calmar', 'the maximum drawdown and the Calmar ratio')),
    })
    return W


def widgets_old(lang):
    ro = lang == 'ro'
    cl = [('90', '90%'), ('95', '95%'), ('99', '99%')]
    met = [('cagr', 'CAGR'), ('sharpe', 'Sharpe'), ('sortino', 'Sortino'), ('calmar', 'Calmar'), ('mdd', 'MDD')]
    sc = [('log', 'logaritmică' if ro else 'logarithmic'), ('lin', 'liniară' if ro else 'linear')]
    return {
     'tlv-adjusted': (slider('tl-th', 'Pragul diferenței dintre cele două randamente logaritmice (puncte procentuale)' if ro else 'Threshold on the gap between the two log returns (percentage points)', 0.5, 50, 0.5, 2) + play('tl'), charts('tl-chart')),
     'bet-bettr': (slider('bb-y', 'Anul investiției (2014 = 23 septembrie 2014)' if ro else 'Year of the investment (2014 = 23 September 2014)', 2014, 2025, 1, 2014) + play('bb'), charts('bb-chart')),
     'eurron': (slider('eu-th', 'Pragul diferenței dintre variațiile zilnice (puncte procentuale)' if ro else 'Threshold on the gap between the daily changes (percentage points)', 0.25, 5, 0.25, 1) + play('eu'), charts('eu-chart')),
     'price-returns': (slider('pr-k', 'Numărul de subperioade egale (0 = fără)' if ro else 'Number of equal subperiods (0 = off)', 0, 6, 1, 0) + play('pr'), charts('pr-chart')),
     'simple-log': (slider('sl-R', 'Randamentul simplu R (%)' if ro else 'Simple return R (%)', -60, 60, 0.2, 10) + play('sl'), charts('sl-chart')),
     'growth': (slider('gr-y', 'Anul investiției' if ro else 'Year of the investment', 2015, 2025, 1, 2015)
                + select('gr-sc', 'Scara verticală' if ro else 'Vertical scale', sc, 'log') + play('gr'), charts('gr-chart')),
     'hist-normal': (slider('hi-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 1, 6, 0.5, 3) + play('hi'), charts('hi-chart')),
     'rolling-vol': (slider('rv-w', 'Fereastra (zile de tranzacționare; Bitcoin: aceeași durată calendaristică)' if ro else 'Window (trading days; Bitcoin: the same calendar length)', 21, 500, 1, 250) + play('rv'), charts('rv-chart')),
     'sharpe-ci': (slider('sc-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2015, 2024, 1, 2015)
                   + select('sc-cl', 'Nivelul de încredere' if ro else 'Confidence level', cl, '95') + play('sc'), charts('sc-chart')),
     'drawdowns': (slider('dd-x', 'Pragul x: zile cu peste x% sub vîrf' if ro else 'Threshold x: days more than x% below the peak', 5, 80, 5, 20) + play('dd'), charts('dd-chart')),
     'vol-drag': (slider('vd-s', 'Volatilitatea anuală σ a unui activ cu media aritmetică 10% pe an (%)' if ro else 'Annual volatility σ of an asset with arithmetic mean 10% a year (%)', 0, 80, 1, 30) + play('vd'), charts('vd-chart')),
     'risk-return': (select('rr-m', 'Axa verticală' if ro else 'Vertical axis', met, 'cagr')
                     + slider('rr-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2015, 2024, 1, 2015) + play('rr'), charts('rr-chart')),
    }


QA = {
 'ro': {
  'tlv-adjusted': ('Pe slide, randamentele zilnice ale celor două serii diferă cu peste 2 puncte procentuale în 24 de zile din 2015 încoace. Butonul ▶ ridică pragul. Cîte zile rămîn la un prag de 10 puncte procentuale?',
                   '11 zile. Pe lîngă consolidarea din 16 august 2022 (+230,3% din prețul de închidere, 0,0% din prețul ajustat), alte 10 zile au o diferență de peste 10 puncte procentuale între cele două randamente logaritmice, toate cu prețul de închidere în scădere; cea mai mare, 19,5 puncte procentuale, pe 3 august 2017. Peste 20 de puncte procentuale rămîne doar consolidarea. Un randament calculat din prețul de închidere ar trata aceste zile drept pierderi ale acționarilor, deși valoarea deținerii nu s-a modificat.'),
  'bet-bettr': ('Butonul ▶ mută data investiției din 2014 pînă în 2025. Cu investiția făcută la începutul lui 2022, cu cîte puncte procentuale pe an crește BET-TR mai repede decît BET?',
                'Cu 8,3 puncte procentuale: 29,9% pe an față de 21,6%. Valorile finale sînt 342 și 251, o diferență de 91 de unități, mult mai mică decît 1 005 față de 455 pentru investiția din 23 septembrie 2014. Diferența dintre ratele de creștere rămîne însă între 7,5 și 8,4 puncte procentuale pentru orice an de început: dividendele adaugă în fiecare an aproximativ același randament, iar efectul lor asupra averii se acumulează în timp.'),
  'eurron': ('Volatilitatea anualizată a seriei EODHD este 13,4%, a cursului BNR 4,5%. Cît devine volatilitatea seriei EODHD dacă eliminăm cele 412 zile în care variațiile zilnice diferă cu peste 1 punct procentual, adică 8% din zile?',
             'Circa 4,7%, aproape de 4,3%, volatilitatea cursului BNR în aceleași zile. Volatilitatea depinde de pătratul abaterilor, deci cîteva sute de cotații greșite o triplează. Dintre cele 412 zile, 307 sînt în 2010–2013. Peste 2 puncte procentuale diferă 283 de zile, peste 5 puncte procentuale 10 zile.'),
  'price-returns': ('Butonul ▶ împarte eșantionul 2000–2026 în pînă la 6 subperioade egale. Cu 5 subperioade, ce se întîmplă cu media și abaterea standard a randamentelor zilnice de la o subperioadă la alta?',
                    'Media randamentelor rămîne aproape de zero, între −0,016% și 0,057% pe zi, în timp ce media prețului crește de la 1 145 la 5 235 de puncte. Abaterea standard a randamentelor variază însă între 0,97% și 1,53% pe zi: randamentele oscilează în jurul unui nivel stabil, dar volatilitatea lor se schimbă în timp. Testul ADF respinge rădăcina unitară pentru randamente (statistica −14,3), nu și pentru logaritmul prețului (statistica −2,08, p-value 0,56).'),
  'simple-log': ('Bitcoin, 7 decembrie 2017: R = +25,2%, r = +22,5%, deci R − r = 2,7 puncte procentuale. Cît este diferența R − r pentru o scădere de aceeași mărime, R = −25,2%?',
                 '3,8 puncte procentuale: r = ln(0,748) = −29,0%. Funcția ln este concavă, iar diferența este mai mare pentru scăderi decît pentru creșteri de aceeași mărime. Aproximarea R²/2 = 3,2 puncte procentuale este aceeași pentru ambele semne; termenul R³/3 al seriei Taylor o corectează în sus la scăderi și în jos la creșteri. La R = −37,2% (Bitcoin, 12 martie 2020), diferența ajunge la 9,3 puncte procentuale.'),
  'growth': ('Butonul ▶ mută data investiției din 2015 pînă în 2025. Cu investiția făcută la începutul lui 2022, care dintre cele patru serii are cea mai mare valoare finală?',
             'BET: 251, față de Bitcoin 170, S&P 500 161 și DAX 159. Pentru investiția din 2015, Bitcoin ajunge la 29 475, iar pentru cea din 2018 la 592, tot pe primul loc. Clasamentul după randamentul cumulat depinde de data de început.'),
  'hist-normal': ('Dincolo de 3 abateri standard, S&P 500 are de 5,2 ori mai multe zile decît distribuția Normală. Butonul ▶ ridică pragul. De cîte ori mai multe zile are S&P 500 dincolo de 4 abateri standard?',
                  'De circa 113 ori: 21 de zile din 2 943 (0,71%), față de 0,0063% pentru distribuția Normală. Raportul crește cu pragul, deoarece coada distribuției Normale scade foarte repede; la Bitcoin raportul este 103 (28 de zile). Frecvența zilelor extreme nu rezultă din abaterea standard prin distribuția Normală.'),
  'rolling-vol': ('Cu o fereastră de 250 de zile, volatilitatea anualizată a S&P 500 ajunge cel mult la 35,0%. Butonul ▶ schimbă fereastra. Cît este maximul cu o fereastră de 21 de zile (o lună)?',
                  '97,5%, în fereastra care se termină pe 6 aprilie 2020. O fereastră scurtă urmărește rapid o perioadă agitată, cum a fost martie 2020; o fereastră de un an o diluează cu lunile calme dinainte și de după. Minimul scade și el, de la 6,7% la 3,5%. O fereastră scurtă reacționează repede, dar estimarea ei are o eroare mai mare.'),
  'sharpe-ci': ('Pe 2015–2026 (11,7 ani), intervalul de încredere 95% este raportul Sharpe ± 0,57. Butonul ▶ mută începutul eșantionului spre 2024. Cu începutul în 2023 (3,7 ani), cît devine jumătatea intervalului?',
                'Circa ± 1,02, aproximativ 1,96/√3,7. Eroarea standard a raportului Sharpe anual scade ca 1/√Y, cu Y numărul de ani, oricîte observații zilnice ar conține eșantionul. Cu începutul în 2023, intervalul Nuclearelectrica conține 0, iar intervalul DAX nu mai conține 0.'),
  'drawdowns': ('Peste 20% sub vîrf se află 27% din zilele S&P 500, 56% din zilele BET și 66% din zilele Bitcoin. Butonul ▶ ridică pragul. Ce pondere din zile a petrecut BET, din 1997, la peste 50% sub vîrful anterior?',
                '18%, aproape o zi din cinci; S&P 500 doar 0,3%, Bitcoin 31,6%. O scădere de 50% cere un cîștig de 100% pentru revenire. Adîncimea drawdown-ului maxim (−82,5% la BET) nu spune cît durează scăderea; ponderea zilelor sub prag arată durata.'),
  'vol-drag': ('Media aritmetică 10% pe an, σ = 30%: volatility drag este 4,5%, iar 1 leu devine 1,73 lei după 10 ani. Butonul ▶ crește σ. Cu σ = 60%, crește averea după 10 ani?',
               'Nu. Drag-ul este σ²/2 = 18%, de patru ori mai mare, iar rata de creștere este 10% − 18% = −8% pe an: după 10 ani 1 leu devine 0,45 lei. O medie aritmetică pozitivă nu garantează creșterea averii. Pe date reale, drag-ul Bitcoin este 22,1 puncte procentuale, aproape de σ²/2 = 21,9%.'),
  'risk-return': ('Pe 2015–2026, Bitcoin are cel mai mare CAGR. Butonul ▶ mută începutul eșantionului. Cu începutul în 2018, care activ are cel mai mare CAGR?',
                  'Nuclearelectrica (SNN), cu 40,7% pe an; tot SNN are în acest eșantion cel mai mare raport Sharpe, Sortino și Calmar. CAGR-ul Bitcoin scade de la 60,6% la 22,7%, deoarece eșantionul începe la două săptămîni după vîrful din 16 decembrie 2017. Clasamentul depinde de indicator și de perioadă.'),
  'portfolio': ('Cu ponderi egale, suma ponderată a randamentelor logaritmice dă un randament logaritmic cumulat de 2,47 în loc de 2,56. Butonul ▶ mută ponderea TLV de la 10% la 90%. La ce pondere este cea mai mare diferența dintre cele două randamente cumulate?',
                'La 50%: 0,093. La 25% și 75% diferența este circa 0,07, la 10% și 90% doar 0,033. Diferența zilnică este aproximativ w(1 − w)(R<sub>TLV</sub> − R<sub>SNP</sub>)²/2, cea mai mare la ponderi egale și zero pentru o singură acțiune. Cea mai mare diferență zilnică, 13,9 puncte de bază pe 31 mai 2016, acoperă trei zile de tranzacționare: SNP nu are prețuri pe 27 și 30 mai 2016, așa că ambele randamente se calculează de la 26 mai; TLV crește cu 6,6%, SNP scade cu 4,0%.'),
  'annualise': ('Din randamente zilnice, volatilitatea anualizată a S&P 500 pe 2015–2026 este 17,7%. Butonul ▶ trece la randamente săptămînale și lunare. Cît dă √12 × σ calculat din randamentele lunare?',
                '15,0%; din randamente săptămînale, 16,7%. Regula rădăcinii pătrate presupune randamente necorelate, cu varianța constantă, iar S&P 500 nu o respectă exact pe 2015–2026. Seria lunară are doar 140 de randamente, deci și estimarea ei are o eroare mai mare. La Bitcoin cele trei valori sînt apropiate: 66,7%, 66,7% și 66,3%.'),
  'mean-se': ('Pe 2015–2026 (11,7 ani), intervalul de încredere 95% al mediei anuale S&P 500 este [1,1%; 21,4%]. Butonul ▶ mută începutul eșantionului. Cu începutul în 2020 (6,7 ani), conține intervalul valoarea 0?',
              'Da: 12,7% ± 15,3%, adică [−2,6%; 28,1%]. Media estimată este mai mare decît pe 2015–2026, dar eroarea standard crește de la 5,2% la 7,8%, aproximativ ca 1/√Y. Cele 1 686 de randamente zilnice nu ajută: precizia mediei depinde de numărul de ani. Un interval care conține 0 nu arată că media este zero, ci doar că 6,7 ani de date nu o separă de zgomot.'),
  'desc-indices': ('La randamente zilnice, excesul de boltire al S&P 500 este 15,8. Butonul ▶ trece la randamente săptămînale și lunare. Ce se întîmplă cu excesul de boltire?',
                   'Scade: 7,2 la randamente săptămînale și 0,7 la randamente lunare. Un randament lunar este suma a circa 21 de randamente zilnice, iar suma este mai aproape de distribuția Normală decît fiecare termen. La Bitcoin excesul de boltire scade de la 12,2 la 2,1 și 0,2. Asimetria S&P 500 rămîne negativă la toate cele trei frecvențe (−0,64, −0,89, −0,50).'),
  'desc-bvb': ('Banca Transilvania are excesul de boltire 25,2, cel mai mare din tabel. Butonul ▶ elimină randamentele din 30 și 31 mai 2016. Cît devine excesul de boltire al TLV?',
               '18,1. Prețul ajustat din 30 mai 2016 nu se potrivește cu cele din zilele vecine, așa că randamentele zilnice sînt −15,0% și +20,5%, deși pe cele două zile împreună prețul ajustat crește cu 5,5%. Cele două zile ridică excesul de boltire de la 18,1 la 25,2 și dau maximul din tabel; fără ele, cel mai mare randament zilnic al TLV este 11,2%, iar asimetria devine −1,08. TLV rămîne acțiunea cu cea mai mare boltire. Două zile din 2 677 schimbă momentul de ordinul patru, care depinde de puterea a patra a abaterilor.'),
  'perf-indices': ('Cu r<sub>f</sub> = 0, raportul Sharpe al S&P 500 este 0,72. Butonul ▶ ridică rata fără risc pînă la 5% pe an. Cît devine raportul Sharpe al S&P 500 la r<sub>f</sub> = 5%?',
                   '0,44: scade cu r<sub>f</sub>/σ = 5%/17,6% = 0,28. La Bitcoin scade doar de la 1,05 la 0,97, pentru că σ = 66,2%, iar DAX coboară la 0,26. Eroarea standard rămîne circa 0,29, iar raportul Sortino al S&P 500 scade de la 1,02 la 0,62. Clasamentul după raportul Sharpe depinde de rata fără risc folosită.'),
  'perf-bvb': ('Pe 2015–2026, Nuclearelectrica are cel mai mare raport Calmar dintre acțiunile BVB (0,83). Butonul ▶ mută începutul eșantionului. Cu începutul în 2022, care acțiune BVB are cel mai mare raport Calmar?',
               'OMV Petrom (SNP), 1,96: CAGR 38,0% și MDD de doar −19,3%. Transgaz are cel mai mare CAGR (47,6%), dar MDD −34,5% îi dă un raport Calmar de 1,38. Nuclearelectrica coboară la 0,40. Numitorul este un singur episod, deci raportul Calmar se schimbă mult cu perioada.')
 },
 'en': {
  'tlv-adjusted': ('On the slide, the daily returns of the two series differ by more than 2 percentage points on 24 days since 2015. The ▶ button raises the threshold. How many days remain at a threshold of 10 percentage points?',
                   '11 days. Besides the consolidation of 16 August 2022 (+230.3% from the close, 0.0% from the adjusted close), 10 other days show a gap of more than 10 percentage points between the two log returns, all with the close falling; the largest, 19.5 percentage points, on 3 August 2017. Above 20 percentage points only the consolidation remains. A return computed from the close would treat these days as losses of the shareholders, although the value of the holding did not change.'),
  'bet-bettr': ('The ▶ button moves the date of the investment from 2014 to 2025. With the investment made at the start of 2022, by how many percentage points a year does BET-TR grow faster than BET?',
                'By 8.3 percentage points: 29.9% a year against 21.6%. The final values are 342 and 251, a gap of 91 units, much smaller than 1,005 against 455 for the investment of 23 September 2014. The gap between the growth rates, however, stays between 7.5 and 8.4 percentage points for every start year: dividends add roughly the same return every year, and their effect on wealth accumulates over time.'),
  'eurron': ('The annualised volatility of the EODHD series is 13.4%, that of the BNR rate 4.5%. What does the volatility of the EODHD series become if we drop the 412 days on which the daily changes differ by more than 1 percentage point, that is 8% of the days?',
             'About 4.7%, close to 4.3%, the volatility of the BNR rate on the same days. Volatility depends on squared deviations, so a few hundred wrong quotes triple it. Of the 412 days, 307 fall in 2010–2013. 283 days differ by more than 2 percentage points, 10 days by more than 5 percentage points.'),
  'price-returns': ('The ▶ button splits the 2000–2026 sample into up to 6 equal subperiods. With 5 subperiods, what happens to the mean and the standard deviation of daily returns from one subperiod to the next?',
                    'The mean of returns stays close to zero, between −0.016% and 0.057% a day, while the mean of the price rises from 1,145 to 5,235 points. The standard deviation of returns, however, varies between 0.97% and 1.53% a day: returns fluctuate around a stable level, but their volatility changes over time. The ADF test rejects the unit root for returns (statistic −14.3), not for the log price (statistic −2.08, p-value 0.56).'),
  'simple-log': ('Bitcoin, 7 December 2017: R = +25.2%, r = +22.5%, so R − r = 2.7 percentage points. What is the gap R − r for a fall of the same size, R = −25.2%?',
                 '3.8 percentage points: r = ln(0.748) = −29.0%. The function ln is concave, and the gap is larger for falls than for rises of the same size. The approximation R²/2 = 3.2 percentage points is the same for both signs; the R³/3 term of the Taylor series corrects it upwards for falls and downwards for rises. At R = −37.2% (Bitcoin, 12 March 2020), the gap reaches 9.3 percentage points.'),
  'growth': ('The ▶ button moves the date of the investment from 2015 to 2025. With the investment made at the start of 2022, which of the four series has the largest final value?',
             'BET: 251, against Bitcoin 170, S&P 500 161 and DAX 159. For the investment of 2015, Bitcoin reaches 29,475, and for that of 2018 592, still first. The ranking by cumulative return depends on the start date.'),
  'hist-normal': ('Beyond 3 standard deviations, the S&P 500 has 5.2 times as many days as the Normal distribution. The ▶ button raises the threshold. How many times as many days does the S&P 500 have beyond 4 standard deviations?',
                  'About 113 times: 21 days out of 2,943 (0.71%), against 0.0063% for the Normal distribution. The ratio grows with the threshold, because the tail of the Normal distribution falls very fast; for Bitcoin the ratio is 103 (28 days). The frequency of extreme days does not follow from the standard deviation through the Normal distribution.'),
  'rolling-vol': ('With a 250-day window, the annualised volatility of the S&P 500 reaches at most 35.0%. The ▶ button changes the window. What is the maximum with a 21-day window (one month)?',
                  '97.5%, in the window ending on 6 April 2020. A short window follows a turbulent period such as March 2020 quickly; a one-year window dilutes it with the calm months before and after. The minimum falls too, from 6.7% to 3.5%. A short window reacts fast, but its estimate has a larger error.'),
  'sharpe-ci': ('Over 2015–2026 (11.7 years), the 95% confidence interval is the Sharpe ratio ± 0.57. The ▶ button moves the start of the sample towards 2024. With the start in 2023 (3.7 years), what does the half-width of the interval become?',
                'About ± 1.02, roughly 1.96/√3.7. The standard error of the annual Sharpe ratio falls as 1/√Y, with Y the number of years, however many daily observations the sample holds. With the start in 2023, the Nuclearelectrica interval contains 0, and the DAX interval no longer does.'),
  'drawdowns': ('More than 20% below the peak lie 27% of the S&P 500 days, 56% of the BET days and 66% of the Bitcoin days. The ▶ button raises the threshold. What share of its days did BET, since 1997, spend more than 50% below the previous peak?',
                '18%, almost one day in five; the S&P 500 only 0.3%, Bitcoin 31.6%. A fall of 50% needs a gain of 100% to recover. The depth of the maximum drawdown (−82.5% for BET) does not say how long the fall lasts; the share of days below the threshold shows the duration.'),
  'vol-drag': ('Arithmetic mean 10% a year, σ = 30%: the volatility drag is 4.5%, and 1 RON becomes 1.73 RON after 10 years. The ▶ button raises σ. With σ = 60%, does wealth grow over 10 years?',
               'No. The drag is σ²/2 = 18%, four times larger, and the growth rate is 10% − 18% = −8% a year: after 10 years 1 RON becomes 0.45 RON. A positive arithmetic mean does not guarantee that wealth grows. On real data, the drag of Bitcoin is 22.1 percentage points, close to σ²/2 = 21.9%.'),
  'risk-return': ('Over 2015–2026, Bitcoin has the highest CAGR. The ▶ button moves the start of the sample. With the start in 2018, which asset has the highest CAGR?',
                  'Nuclearelectrica (SNN), with 40.7% a year; SNN also has the highest Sharpe, Sortino and Calmar ratios in this sample. The CAGR of Bitcoin falls from 60.6% to 22.7%, because the sample starts two weeks after the peak of 16 December 2017. The ranking depends on the indicator and on the period.'),
  'portfolio': ('With equal weights, the weighted sum of log returns gives a cumulative log return of 2.47 instead of 2.56. The ▶ button moves the TLV weight from 10% to 90%. At which weight is the gap between the two cumulative returns largest?',
                'At 50%: 0.093. At 25% and 75% the gap is about 0.07, at 10% and 90% only 0.033. The daily gap is about w(1 − w)(R<sub>TLV</sub> − R<sub>SNP</sub>)²/2, largest for equal weights and zero for a single stock. The largest daily gap, 13.9 basis points on 31 May 2016, covers three trading days: SNP has no prices on 27 and 30 May 2016, so both returns run from 26 May; TLV rises by 6.6%, SNP falls by 4.0%.'),
  'annualise': ('From daily returns, the annualised volatility of the S&P 500 over 2015–2026 is 17.7%. The ▶ button switches to weekly and monthly returns. What does √12 × σ computed from the monthly returns give?',
                '15.0%; from weekly returns, 16.7%. The square-root-of-time rule assumes uncorrelated returns with constant variance, and the S&P 500 does not follow it exactly over 2015–2026. The monthly series has only 140 returns, so its estimate also has a larger error. For Bitcoin the three values are close: 66.7%, 66.7% and 66.3%.'),
  'mean-se': ('Over 2015–2026 (11.7 years), the 95% confidence interval of the annual mean of the S&P 500 is [1.1%; 21.4%]. The ▶ button moves the start of the sample. With the start in 2020 (6.7 years), does the interval contain 0?',
              'Yes: 12.7% ± 15.3%, that is [−2.6%; 28.1%]. The estimated mean is higher than over 2015–2026, but the standard error rises from 5.2% to 7.8%, roughly as 1/√Y. The 1,686 daily returns do not help: the precision of the mean depends on the number of years. An interval that contains 0 does not show that the mean is zero, only that 6.7 years of data do not separate it from noise.'),
  'desc-indices': ('For daily returns, the excess kurtosis of the S&P 500 is 15.8. The ▶ button switches to weekly and monthly returns. What happens to the excess kurtosis?',
                   'It falls: 7.2 for weekly and 0.7 for monthly returns. A monthly return is the sum of about 21 daily returns, and the sum is closer to the Normal distribution than each term. For Bitcoin the excess kurtosis falls from 12.2 to 2.1 and 0.2. The skewness of the S&P 500 stays negative at all three frequencies (−0.64, −0.89, −0.50).'),
  'desc-bvb': ('Banca Transilvania has an excess kurtosis of 25.2, the largest in the table. The ▶ button removes the returns of 30 and 31 May 2016. What does the excess kurtosis of TLV become?',
               '18.1. The adjusted close of 30 May 2016 does not fit those of the neighbouring days, so the daily returns are −15.0% and +20.5%, although over the two days together the adjusted close rises by 5.5%. The two days raise the excess kurtosis from 18.1 to 25.2 and give the maximum of the table; without them, the largest daily return of TLV is 11.2%, and the skewness becomes −1.08. TLV remains the stock with the largest kurtosis. Two days out of 2,677 change the fourth moment, which depends on the fourth power of the deviations.'),
  'perf-indices': ('With r<sub>f</sub> = 0, the Sharpe ratio of the S&P 500 is 0.72. The ▶ button raises the risk-free rate up to 5% a year. What does the Sharpe ratio of the S&P 500 become at r<sub>f</sub> = 5%?',
                   '0.44: it falls by r<sub>f</sub>/σ = 5%/17.6% = 0.28. For Bitcoin it falls only from 1.05 to 0.97, because σ = 66.2%, and the DAX drops to 0.26. The standard error stays about 0.29, and the Sortino ratio of the S&P 500 falls from 1.02 to 0.62. The ranking by Sharpe ratio depends on the risk-free rate used.'),
  'perf-bvb': ('Over 2015–2026, Nuclearelectrica has the highest Calmar ratio among the BVB stocks (0.83). The ▶ button moves the start of the sample. With the start in 2022, which BVB stock has the highest Calmar ratio?',
               'OMV Petrom (SNP), 1.96: CAGR 38.0% and an MDD of only −19.3%. Transgaz has the highest CAGR (47.6%), but its MDD of −34.5% gives a Calmar ratio of 1.38. Nuclearelectrica drops to 0.40. The denominator is a single episode, so the Calmar ratio changes a lot with the period.')
 },
}

def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    res = {}
    for a, fr, _, _, key in XBLOCKS:
        title = latex_text(frames[fr - 1][0])
        assert key in title and 'includegraphics' not in frames[fr - 1][1], (lang, a, title)
        res[a] = (int(pages[fr - 1][0]), title)
    return res


def xblock_html(anchor, lang, web, pg=None):
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved to their frames by ch1.js)."""
    W = widgets_all(lang)
    fp = frame_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''.join(xblock_html(x, lang, True, fp[x][0]) for x, _, _, host, _ in XBLOCKS if host == a)
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
    secs = {a: s for a, _, s in BLOCKS} | {a: s for a, _, s, _, _ in XBLOCKS}
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch1_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    html = html[:start] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[end:]
    return html.replace('</head>', MATHJAX + '</head>', 1)


def build():
    import types
    import pymupdf
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch1_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH1_TEXT = {
 tl: (th, n, up, dn) => `Zile în care randamentele logaritmice din cele două serii diferă cu peste ${th} puncte procentuale: ${n} din 2015 încoace; prețul de închidere crește în ${up} dintre ele și scade în ${dn}.`,
 tlBig: (d, rc, ra) => `Cea mai mare diferență: ${d}, randamentul logaritmic ${rc}% din prețul de închidere și ${ra}% din prețul ajustat.`,
 bb: (d, b, t) => `100 de unități investite pe ${d} devin ${b} în BET și ${t} în BET-TR pe 18 septembrie 2026.`,
 bb2: (gb, gt, g) => `Rata anuală compusă de creștere: BET ${gb}%, BET-TR ${gt}%; dividendele adaugă ${g} puncte procentuale pe an.`,
 eu: (n, th, k) => `${n} zile comune; variația zilnică a seriei EODHD diferă cu peste ${th} puncte procentuale de cea a cursului BNR în ${k} zile (marcate pe grafic).`,
 eu2: (ve, vb, ve2, vb2) => `Volatilitatea anualizată în toate zilele: ${ve}% (EODHD) și ${vb}% (BNR); fără zilele marcate: ${ve2}% (EODHD) și ${vb2}% (BNR).`,
 pr: (a, c, p, a2, c2) => `Testul ADF: logaritmul prețului (cu trend), statistica ${a}, valoarea critică 5% ${c}, p-value ${p}; randamentele logaritmice, statistica ${a2}, valoarea critică 5% ${c2}, p-value < 0,001.`,
 prSeg: t => `Media ± abaterea standard pe subperioade, pentru preț (puncte) și pentru randamentul zilnic: ${t}.`,
 prItem: (y0, y1, mp, sp, mr, sr) => `${y0}–${y1}: ${mp} ± ${sp} și ${mr}% ± ${sr}%`,
 sl: (R, r, t, bp, pp) => `R = ${R}%: r = ln(1 + R) = ${r}%, iar R − R²/2 = ${t}%; diferența R − r este ${bp} puncte de bază (${pp} puncte procentuale).`,
 slExt: t => `Zilele extreme: ${t}.`,
 slItem: (a, d, R, r) => `${a}, ${d}: R = ${R}%, r = ${r}%`,
 gr: (d, t) => `100 de unități investite pe ${d}; valori finale: ${t}.`,
 gr2: t => `Randamentul simplu cumulat și randamentul logaritmic cumulat: ${t}.`,
 grItem: (a, c, l) => `${a} ${c}% și ${l}`,
 hi: (k, pn) => `Dincolo de ${k} abateri standard de medie, distribuția Normală dă ${pn} din zile.`,
 hiItem: (a, n, m, p, ra, w) => `${a}: ${n} zile din ${m} (${p}), de ${ra} ori mai multe; ${w} din zile se află la cel mult o abatere standard de medie (distribuția Normală: 68,3%)`,
 rv: w => `Fereastra de ${w} zile de tranzacționare; volatilitatea anualizată, minimul, maximul și data de sfîrșit a ferestrei cu maximul:`,
 rvItem: (a, w, lo, hi, d) => `${a} (fereastra de ${w} zile): între ${lo}% și ${hi}%, maximul pe ${d}`,
 sc: (d, y, cl, hw) => `Eșantion din ${d}, ${y} ani; intervalul de încredere ${cl}% este, în medie, raportul Sharpe ± ${hw}.`,
 sc2: (a, s, z, ov) => `Cea mai mare estimare punctuală: ${a} (${s}). Intervale care conțin 0: ${z}. ${ov ? 'Toate intervalele se suprapun.' : 'Cel puțin două intervale nu se suprapun.'}`,
 dd: (x, g) => `O scădere de ${x}% cere un cîștig de ${g}% pentru revenire. Ponderea zilelor aflate cu peste ${x}% sub vîrful anterior și drawdown-ul maxim:`,
 ddItem: (a, y, s, m) => `${a} (din ${y}): ${s} din zile; MDD ${m}%`,
 vd: (s, d, g, w) => `Media aritmetică 10% pe an, σ = ${s}%: volatility drag σ²/2 = ${d}%, creșterea ${g}% pe an; după 10 ani 1 leu devine ${w} lei.`,
 vd2: (a, l, d, h) => `Bitcoin, 2015–2026: media aritmetică ${a}%, media logaritmică ${l}%, drag ${d} puncte procentuale; σ²/2 = ${h}%.`,
 rr: (d, c, s, so, ca) => `Eșantion din ${d}. Cel mai bun după CAGR: ${c}; după Sharpe: ${s}; după Sortino: ${so}; după Calmar: ${ca}.`,
 rr2: (v, m) => `Cea mai mică volatilitate: ${v}; cel mai mic MDD în valoare absolută: ${m}.`,
 h: { clr: 'Randamentul logaritmic cumulat', wealth: '1 leu devine', cagr: 'CAGR (%)', pexact: 'Portofoliul, exact (din randamente simple)', papprox: 'Portofoliul, suma ponderată a randamentelor logaritmice',
      A: 'A (observații pe an)', sdD: 'σ zilnic (%)', volD: 'σ<sub>an</sub> = √A σ (%)', vol252: '√252 σ (%)', volW: 'săptămînal: √A σ (%)', volM: 'lunar: √A σ (%)',
      Y: 'Y (ani)', mu: 'μ<sub>an</sub> (%)', ci: cl => `IC ${cl}% (%)`, mean: 'Media', sd: 'Abaterea std.', skew: 'Asimetria', exk: 'Excesul de boltire', min: 'Minimul (data)', max: 'Maximul (data)', vol: 'Vol. %' },
 fq: { d: 'zilnice', w: 'săptămînale', m: 'lunare' },
 slw1: (a, d0, p0, d1, p1) => `${a}: prețul de închidere este ${p0} pe ${d0} și ${p1} pe ${d1}.`,
 slw2: (p1, p0, R, r) => `R = ${p1}/${p0} − 1 = ${R}%; r = ln(${p1}/${p0}) = ${r}%.`,
 slw3: (g, h, t) => `R − r = ${g} puncte procentuale; termenul de ordinul doi, R²/2, dă ${h}, iar cu termenul de ordinul trei, R²/2 − R³/3, ${t}.`,
 grw1: (a, p0, d0, pT, dT, Y) => `${a}: P<sub>0</sub> = ${p0} (${d0}), P<sub>T</sub> = ${pT} (${dT}); Y = ${Y} ani calendaristici.`,
 grw2: (n, g, c) => `Produsul celor ${n} randamente brute zilnice 1 + R<sub>t</sub> se reduce la P<sub>T</sub>/P<sub>0</sub> = ${g}: randamentul simplu cumulat CSR = ${c}%.`,
 grw3: (n, s, l, c) => `Suma celor ${n} randamente logaritmice zilnice: Σ r<sub>t</sub> = ${s} = ln(P<sub>T</sub>/P<sub>0</sub>) = ${l}; e<sup>${s}</sup> − 1 = ${c}%, același CSR.`,
 grw4: (g, Y, c1, s, c2) => `CAGR = ${g}<sup>1/${Y}</sup> − 1 = ${c1}%, la fel ca e<sup>${s}/${Y}</sup> − 1 = ${c2}%.`,
 grw5: (d, c, d0) => `Pe calendarul propriu al seriei, de la ${d}, CAGR este ${c}% (tabelul de performanță); graficul începe pe ${d0}, prima zi cu prețuri pentru toate cele patru serii.`,
 ddw1: (a, y, dp, pp, dt, pt) => `${a}, din ${y}: cel mai mare preț înainte de minim este ${pp}, pe ${dp}; drawdown-ul cel mai adînc este atins pe ${dt}, la ${pt}.`,
 ddw2: (pt, pp, m) => `MDD = ${pt}/${pp} − 1 = ${m}%.`,
 ddw3: (x, g) => `Cîștigul necesar pentru revenire: x/(1 − x), cu x = ${x}: ${g}%.`,
 ddw4: (dr, a, b, y) => `Prețul revine la vîrful anterior pe ${dr}: ${a} zile de la vîrf la minim, ${b} zile de la minim la revenire, ${y} ani sub vîrful anterior.`,
 ddw4n: (a, e) => `${a} zile de la vîrf la minim; pînă pe ${e}, prețul nu a revenit la vîrful anterior.`,
 vdw1: (a, q, m, mu) => `${a}, 2015–2026: A = ${q} observații pe an; media randamentelor simple zilnice ${m}%, deci μ<sub>arith</sub> = A × ${m}% = ${mu}%.`,
 vdw2: (m, mu) => `Media randamentelor logaritmice zilnice ${m}%, deci μ<sub>log</sub> = ${mu}%.`,
 vdw3: (a, l, d, v, h) => `Drag = ${a}% − ${l}% = ${d} puncte procentuale; cu σ = ${v}% (randamente simple), σ²/2 = ${h}%.`,
 vdw4: (l, e, c) => `Creșterea averii: e<sup>${l}</sup> − 1 = ${e}% pe an, practic CAGR-ul din tabelul de performanță (${c}%). Media aritmetică nu este rata de creștere a averii.`,
 pfw1: (d, a0, a1, Ra, b0, b1, Rb) => `${d}, prețuri ajustate (lei): TLV de la ${a0} la ${a1} (R = ${Ra}%), SNP de la ${b0} la ${b1} (R = ${Rb}%).`,
 pfw2: (w, Ra, w2, Rb, Rp, rp) => `R<sub>p</sub> = ${w} × (${Ra}%) + ${w2} × (${Rb}%) = ${Rp}%; randamentul logaritmic exact ln(1 + R<sub>p</sub>) = ${rp}%.`,
 pfw3: (w, ra, w2, rb, ap) => `Suma ponderată a randamentelor logaritmice: ${w} × (${ra}%) + ${w2} × (${rb}%) = ${ap}%.`,
 pfw4: (g, n, m) => `Diferența: ${g} puncte de bază în această zi, cea mai mare dintre cele ${n} zile; în medie ${m} puncte de bază pe zi. Adunate pe toată perioada, diferențele dau diferența dintre randamentele logaritmice cumulate din tabel.`,
 pf: (n, m, g, d) => `${n} zile comune; randamentul logaritmic zilnic exact minus suma ponderată a randamentelor logaritmice: în medie ${m} puncte de bază, cel mult ${g} puncte de bază (${d}).`,
 pf2: (dl, c, ca, cb) => `Diferența randamentelor logaritmice cumulate: ${dl}. CAGR: portofoliul ${c}%, TLV ${ca}%, SNP ${cb}%.`,
 anw1: (a, f, d0, n, Y, q) => `${a}, randamente logaritmice ${f} din ${d0}: ${n} randamente în Y = ${Y} ani, deci A = n/Y = ${q}.`,
 anw2: (q, m, mu) => `μ<sub>an</sub> = A μ = ${q} × ${m}% = ${mu}%.`,
 anw3: (q, sq, s, v) => `σ<sub>an</sub> = √A σ = √${q} × ${s}% = ${sq} × ${s}% = ${v}%.`,
 anw4: (v252, v365) => `Cu √252 = 15,87 rezultatul ar fi ${v252}%, cu √365 = 19,10 ar fi ${v365}%: factorul trebuie să corespundă frecvenței seriei.`,
 anw4f: (v, vd) => `Din randamente zilnice, aceeași serie dă ${vd}%, iar la această frecvență √A σ = ${v}%. Regula rădăcinii pătrate este exactă numai pentru randamente necorelate, cu varianța constantă.`,
 an: (d, a, b, c) => `S&P 500 din ${d}, volatilitatea anualizată din randamente logaritmice zilnice, săptămînale și lunare: ${a}%, ${b}%, ${c}%.`,
 an2: (v, v252, q) => `Bitcoin are ${q} observații pe an: √A σ = ${v}%; factorul bursier √252 ar da ${v252}%.`,
 msw1: (a, d0, mu, v, n, q, Y) => `${a} din ${d0}: μ̂<sub>an</sub> = ${mu}%, σ<sub>an</sub> = ${v}%; ${n} randamente zilnice, A = ${q}, Y = n/A = ${Y} ani.`,
 msw2: (v, Y, sq, se) => `SE = σ<sub>an</sub>/√Y = ${v}%/√${Y} = ${v}%/${sq} = ${se}%.`,
 msw3: (cl, z, mu, se, lo, hi) => `Intervalul ${cl}%: ${mu}% ± ${z} × ${se}% = [${lo}%; ${hi}%].`,
 msw4: (nm, sem, n, se) => `Din cele ${nm} randamente lunare ale acelorași ani: SE = ${sem}%, față de ${se}% din ${n} randamente zilnice. Precizia mediei depinde de numărul de ani.`,
 ms: (d, cl, z) => `Eșantion din ${d}; intervale ${cl}% care conțin 0: ${z}.`,
 ms2: (se, Y) => `Eroarea standard medie a mediei anuale: ${se} puncte procentuale, cu ${Y} ani de date.`,
 di: (f, d, ra, neg, hk, k) => `Randamente logaritmice ${f} din ${d}. Abaterea standard a Bitcoin este de ${ra} ori media celor ale S&P 500, DAX și BET. Asimetrie negativă: ${neg} din 5 serii; cel mai mare exces de boltire: ${hk} (${k}).`,
 di2: (lv, sv) => `S&P 500: volatilitatea anualizată din aceste randamente logaritmice este ${lv}%; tabelul de performanță anualizează randamentele simple zilnice: ${sv}%.`,
 db: (f, d, lo, hi, b) => `Randamente logaritmice ${f} din ${d}: abaterile standard ale celor cinci acțiuni sînt între ${lo}% și ${hi}%, a indicelui BET ${b}%.`,
 db2: (hk, k1, lk, k2) => `Cel mai mare exces de boltire: ${hk} (${k1}); cel mai mic: ${lk} (${k2}).`,
 db3: (k0, k1, mx) => `Banca Transilvania: prețul ajustat din 30 mai 2016 (3,10 lei) nu se potrivește cu cele din 27 mai (3,60) și 31 mai (3,81), deci randamentele zilnice sînt −15,0% și +20,5%. Cu aceste două zile, excesul de boltire zilnic este ${k0}; fără ele, ${k1}, iar cel mai mare randament zilnic devine ${mx}%.`,
 piw1: (a, d0, n, q, m, s) => `${a} din ${d0}: n = ${n} randamente simple zilnice, A = ${q}; media ${m}%, abaterea standard ${s}%.`,
 piw2: (m, rf, q, s, srd, sq, sr) => `Raportul Sharpe zilnic SR<sub>d</sub> = (${m}% − ${rf}%/${q})/${s}% = ${srd}; anual: ${srd} × ${sq} = ${sr}.`,
 piw3: (srd, n, sed, sq, se) => `SE = √((1 + ${srd}²/2)/${n}) = ${sed} pe zi; × ${sq} = ${se} pe an.`,
 piw4: (sr, se, lo, hi, pos) => `Intervalul 95%: ${sr} ± 1,96 × ${se} = [${lo}; ${hi}]; ${pos ? 'nu conține 0' : 'conține 0'}. Formula presupune randamente i.i.d.; cozile groase și gruparea volatilității fac eroarea reală mai mare.`,
 pi: (d, rf, b, s, se) => `Eșantion din ${d}, r<sub>f</sub> = ${rf}% pe an. Cel mai mare raport Sharpe: ${b} (${s}, SE ${se}).`,
 pi2: (vs, vl, mb, mt) => `Volatilitatea S&P 500: ${vs}% din randamente simple zilnice (acest tabel), ${vl}% din randamente logaritmice (tabelul statisticilor descriptive). MDD: BET ${mb}%, BET-TR ${mt}%.`,
 pbw1: (a, p0, d0, pT, dT, Y, c) => `${a}: ${p0} lei pe ${d0}, ${pT} lei pe ${dT}; Y = ${Y} ani, CAGR = (P<sub>T</sub>/P<sub>0</sub>)<sup>1/Y</sup> − 1 = ${c}%.`,
 pbw2: (dp, pp, dt, pt, m) => `Vîrful dinaintea celei mai mari scăderi: ${pp} lei pe ${dp}; minimul: ${pt} lei pe ${dt}; MDD = ${m}%.`,
 pbw3: (c, m, ca) => `Calmar = ${c}/${m} = ${ca}.`,
 pbw4: (dr, n) => `Prețul revine la vîrful anterior pe ${dr}, la ${n} zile după vîrf.`,
 pbw4n: e => `Pînă pe ${e}, prețul nu a revenit la vîrful anterior.`,
 pb: (d, c, m) => `Eșantion din ${d}. Cel mai mare raport Calmar dintre acțiunile BVB: ${c}; cel mai adînc MDD: ${m}.`,
 pb2: (a, n, done, nr) => `Cel mai lung episod de drawdown maxim: ${a}, ${n} zile${done ? ' de la vîrf la revenire' : ', fără revenire'}. Fără revenire pînă la sfîrșitul eșantionului: ${nr}.`
};
""",
 'en': r"""window.CH1_TEXT = {
 tl: (th, n, up, dn) => `Days on which the log returns of the two series differ by more than ${th} percentage points: ${n} since 2015; the close rises on ${up} of them and falls on ${dn}.`,
 tlBig: (d, rc, ra) => `Largest gap: ${d}, log return ${rc}% from the close and ${ra}% from the adjusted close.`,
 bb: (d, b, t) => `100 units invested on ${d} become ${b} in BET and ${t} in BET-TR on 18 September 2026.`,
 bb2: (gb, gt, g) => `Compound annual growth rate: BET ${gb}%, BET-TR ${gt}%; dividends add ${g} percentage points a year.`,
 eu: (n, th, k) => `${n} common days; the daily change of the EODHD series differs from that of the BNR rate by more than ${th} percentage points on ${k} days (marked on the chart).`,
 eu2: (ve, vb, ve2, vb2) => `Annualised volatility on all days: ${ve}% (EODHD) and ${vb}% (BNR); without the marked days: ${ve2}% (EODHD) and ${vb2}% (BNR).`,
 pr: (a, c, p, a2, c2) => `ADF test: log price (with trend), statistic ${a}, 5% critical value ${c}, p-value ${p}; log returns, statistic ${a2}, 5% critical value ${c2}, p-value < 0.001.`,
 prSeg: t => `Mean ± standard deviation by subperiod, for the price (points) and for the daily return: ${t}.`,
 prItem: (y0, y1, mp, sp, mr, sr) => `${y0}–${y1}: ${mp} ± ${sp} and ${mr}% ± ${sr}%`,
 sl: (R, r, t, bp, pp) => `R = ${R}%: r = ln(1 + R) = ${r}%, and R − R²/2 = ${t}%; the gap R − r is ${bp} basis points (${pp} percentage points).`,
 slExt: t => `Extreme days: ${t}.`,
 slItem: (a, d, R, r) => `${a}, ${d}: R = ${R}%, r = ${r}%`,
 gr: (d, t) => `100 units invested on ${d}; final values: ${t}.`,
 gr2: t => `Cumulative simple return and cumulative log return: ${t}.`,
 grItem: (a, c, l) => `${a} ${c}% and ${l}`,
 hi: (k, pn) => `Beyond ${k} standard deviations from the mean, the Normal distribution gives ${pn} of the days.`,
 hiItem: (a, n, m, p, ra, w) => `${a}: ${n} of ${m} days (${p}), ${ra} times as many; ${w} of the days lie within one standard deviation of the mean (Normal distribution: 68.3%)`,
 rv: w => `Window of ${w} trading days; annualised volatility, minimum, maximum and end date of the window with the maximum:`,
 rvItem: (a, w, lo, hi, d) => `${a} (${w}-day window): between ${lo}% and ${hi}%, maximum on ${d}`,
 sc: (d, y, cl, hw) => `Sample from ${d}, ${y} years; the ${cl}% confidence interval is, on average, the Sharpe ratio ± ${hw}.`,
 sc2: (a, s, z, ov) => `Largest point estimate: ${a} (${s}). Intervals containing 0: ${z}. ${ov ? 'All intervals overlap.' : 'At least two intervals do not overlap.'}`,
 dd: (x, g) => `A fall of ${x}% needs a gain of ${g}% to recover. Share of days more than ${x}% below the previous peak, and maximum drawdown:`,
 ddItem: (a, y, s, m) => `${a} (from ${y}): ${s} of the days; MDD ${m}%`,
 vd: (s, d, g, w) => `Arithmetic mean 10% a year, σ = ${s}%: volatility drag σ²/2 = ${d}%, growth ${g}% a year; after 10 years 1 RON becomes ${w} RON.`,
 vd2: (a, l, d, h) => `Bitcoin, 2015–2026: arithmetic mean ${a}%, log mean ${l}%, drag ${d} percentage points; σ²/2 = ${h}%.`,
 rr: (d, c, s, so, ca) => `Sample from ${d}. Best by CAGR: ${c}; by Sharpe: ${s}; by Sortino: ${so}; by Calmar: ${ca}.`,
 rr2: (v, m) => `Lowest volatility: ${v}; smallest MDD in absolute value: ${m}.`,
 h: { clr: 'Cumulative log return', wealth: '1 RON becomes', cagr: 'CAGR (%)', pexact: 'Portfolio, exact (from simple returns)', papprox: 'Portfolio, weighted sum of log returns',
      A: 'A (observations a year)', sdD: 'daily σ (%)', volD: 'σ<sub>year</sub> = √A σ (%)', vol252: '√252 σ (%)', volW: 'weekly: √A σ (%)', volM: 'monthly: √A σ (%)',
      Y: 'Y (years)', mu: 'μ<sub>year</sub> (%)', ci: cl => `${cl}% CI (%)`, mean: 'Mean', sd: 'Std. dev.', skew: 'Skew.', exk: 'Exc. kurt.', min: 'Minimum (date)', max: 'Maximum (date)', vol: 'Vol. %' },
 fq: { d: 'daily', w: 'weekly', m: 'monthly' },
 slw1: (a, d0, p0, d1, p1) => `${a}: the close is ${p0} on ${d0} and ${p1} on ${d1}.`,
 slw2: (p1, p0, R, r) => `R = ${p1}/${p0} − 1 = ${R}%; r = ln(${p1}/${p0}) = ${r}%.`,
 slw3: (g, h, t) => `R − r = ${g} percentage points; the second-order term, R²/2, gives ${h}, and with the third-order term, R²/2 − R³/3, ${t}.`,
 grw1: (a, p0, d0, pT, dT, Y) => `${a}: P<sub>0</sub> = ${p0} (${d0}), P<sub>T</sub> = ${pT} (${dT}); Y = ${Y} calendar years.`,
 grw2: (n, g, c) => `The product of the ${n} daily gross returns 1 + R<sub>t</sub> reduces to P<sub>T</sub>/P<sub>0</sub> = ${g}: cumulative simple return CSR = ${c}%.`,
 grw3: (n, s, l, c) => `The sum of the ${n} daily log returns: Σ r<sub>t</sub> = ${s} = ln(P<sub>T</sub>/P<sub>0</sub>) = ${l}; e<sup>${s}</sup> − 1 = ${c}%, the same CSR.`,
 grw4: (g, Y, c1, s, c2) => `CAGR = ${g}<sup>1/${Y}</sup> − 1 = ${c1}%, the same as e<sup>${s}/${Y}</sup> − 1 = ${c2}%.`,
 grw5: (d, c, d0) => `On the series' own calendar, from ${d}, the CAGR is ${c}% (performance table); the chart starts on ${d0}, the first day with prices for all four series.`,
 ddw1: (a, y, dp, pp, dt, pt) => `${a}, since ${y}: the highest price before the trough is ${pp}, on ${dp}; the deepest drawdown is reached on ${dt}, at ${pt}.`,
 ddw2: (pt, pp, m) => `MDD = ${pt}/${pp} − 1 = ${m}%.`,
 ddw3: (x, g) => `Gain needed to recover: x/(1 − x), with x = ${x}: ${g}%.`,
 ddw4: (dr, a, b, y) => `The price is back at the old peak on ${dr}: ${a} days from peak to trough, ${b} days from trough to recovery, ${y} years below the old peak.`,
 ddw4n: (a, e) => `${a} days from peak to trough; by ${e}, the price has not returned to the old peak.`,
 vdw1: (a, q, m, mu) => `${a}, 2015–2026: A = ${q} observations a year; mean daily simple return ${m}%, so μ<sub>arith</sub> = A × ${m}% = ${mu}%.`,
 vdw2: (m, mu) => `Mean daily log return ${m}%, so μ<sub>log</sub> = ${mu}%.`,
 vdw3: (a, l, d, v, h) => `Drag = ${a}% − ${l}% = ${d} percentage points; with σ = ${v}% (simple returns), σ²/2 = ${h}%.`,
 vdw4: (l, e, c) => `Growth of wealth: e<sup>${l}</sup> − 1 = ${e}% a year, practically the CAGR of the performance table (${c}%). The arithmetic mean is not the growth rate of wealth.`,
 pfw1: (d, a0, a1, Ra, b0, b1, Rb) => `${d}, adjusted prices (RON): TLV from ${a0} to ${a1} (R = ${Ra}%), SNP from ${b0} to ${b1} (R = ${Rb}%).`,
 pfw2: (w, Ra, w2, Rb, Rp, rp) => `R<sub>p</sub> = ${w} × (${Ra}%) + ${w2} × (${Rb}%) = ${Rp}%; exact log return ln(1 + R<sub>p</sub>) = ${rp}%.`,
 pfw3: (w, ra, w2, rb, ap) => `Weighted sum of the log returns: ${w} × (${ra}%) + ${w2} × (${rb}%) = ${ap}%.`,
 pfw4: (g, n, m) => `Gap: ${g} basis points on this day, the largest of the ${n} days; on average ${m} basis points a day. Summed over the whole period, the gaps give the difference between the cumulative log returns in the table.`,
 pf: (n, m, g, d) => `${n} common days; exact daily log return minus the weighted sum of log returns: on average ${m} basis points, at most ${g} basis points (${d}).`,
 pf2: (dl, c, ca, cb) => `Gap between the cumulative log returns: ${dl}. CAGR: portfolio ${c}%, TLV ${ca}%, SNP ${cb}%.`,
 anw1: (a, f, d0, n, Y, q) => `${a}, ${f} log returns from ${d0}: ${n} returns in Y = ${Y} years, so A = n/Y = ${q}.`,
 anw2: (q, m, mu) => `μ<sub>year</sub> = A μ = ${q} × ${m}% = ${mu}%.`,
 anw3: (q, sq, s, v) => `σ<sub>year</sub> = √A σ = √${q} × ${s}% = ${sq} × ${s}% = ${v}%.`,
 anw4: (v252, v365) => `With √252 = 15.87 the result would be ${v252}%, with √365 = 19.10 it would be ${v365}%: the factor must match the frequency of the series.`,
 anw4f: (v, vd) => `From daily returns the same series gives ${vd}%, and at this frequency √A σ = ${v}%. The square-root-of-time rule is exact only for uncorrelated returns with constant variance.`,
 an: (d, a, b, c) => `S&P 500 from ${d}, annualised volatility from daily, weekly and monthly log returns: ${a}%, ${b}%, ${c}%.`,
 an2: (v, v252, q) => `Bitcoin has ${q} observations a year: √A σ = ${v}%; the stock-market factor √252 would give ${v252}%.`,
 msw1: (a, d0, mu, v, n, q, Y) => `${a} from ${d0}: μ̂<sub>year</sub> = ${mu}%, σ<sub>year</sub> = ${v}%; ${n} daily returns, A = ${q}, Y = n/A = ${Y} years.`,
 msw2: (v, Y, sq, se) => `SE = σ<sub>year</sub>/√Y = ${v}%/√${Y} = ${v}%/${sq} = ${se}%.`,
 msw3: (cl, z, mu, se, lo, hi) => `${cl}% interval: ${mu}% ± ${z} × ${se}% = [${lo}%; ${hi}%].`,
 msw4: (nm, sem, n, se) => `From the ${nm} monthly returns of the same years: SE = ${sem}%, against ${se}% from ${n} daily returns. The precision of the mean depends on the number of years.`,
 ms: (d, cl, z) => `Sample from ${d}; ${cl}% intervals containing 0: ${z}.`,
 ms2: (se, Y) => `Average standard error of the annual mean: ${se} percentage points, with ${Y} years of data.`,
 di: (f, d, ra, neg, hk, k) => `${f.charAt(0).toUpperCase() + f.slice(1)} log returns from ${d}. The standard deviation of Bitcoin is ${ra} times the average of those of the S&P 500, DAX and BET. Negative skewness: ${neg} of 5 series; largest excess kurtosis: ${hk} (${k}).`,
 di2: (lv, sv) => `S&P 500: the annualised volatility from these log returns is ${lv}%; the performance table annualises the daily simple returns: ${sv}%.`,
 db: (f, d, lo, hi, b) => `${f.charAt(0).toUpperCase() + f.slice(1)} log returns from ${d}: the standard deviations of the five stocks are between ${lo}% and ${hi}%, that of the BET index ${b}%.`,
 db2: (hk, k1, lk, k2) => `Largest excess kurtosis: ${hk} (${k1}); smallest: ${lk} (${k2}).`,
 db3: (k0, k1, mx) => `Banca Transilvania: the adjusted close of 30 May 2016 (3.10 RON) does not fit those of 27 May (3.60) and 31 May (3.81), so the daily returns are −15.0% and +20.5%. With these two days the daily excess kurtosis is ${k0}; without them ${k1}, and the largest daily return becomes ${mx}%.`,
 piw1: (a, d0, n, q, m, s) => `${a} from ${d0}: n = ${n} daily simple returns, A = ${q}; mean ${m}%, standard deviation ${s}%.`,
 piw2: (m, rf, q, s, srd, sq, sr) => `Daily Sharpe ratio SR<sub>d</sub> = (${m}% − ${rf}%/${q})/${s}% = ${srd}; annual: ${srd} × ${sq} = ${sr}.`,
 piw3: (srd, n, sed, sq, se) => `SE = √((1 + ${srd}²/2)/${n}) = ${sed} per day; × ${sq} = ${se} a year.`,
 piw4: (sr, se, lo, hi, pos) => `95% interval: ${sr} ± 1.96 × ${se} = [${lo}; ${hi}]; ${pos ? 'it does not contain 0' : 'it contains 0'}. The formula assumes i.i.d. returns; heavy tails and volatility clustering make the true error larger.`,
 pi: (d, rf, b, s, se) => `Sample from ${d}, r<sub>f</sub> = ${rf}% a year. Highest Sharpe ratio: ${b} (${s}, SE ${se}).`,
 pi2: (vs, vl, mb, mt) => `S&P 500 volatility: ${vs}% from daily simple returns (this table), ${vl}% from log returns (descriptive table). MDD: BET ${mb}%, BET-TR ${mt}%.`,
 pbw1: (a, p0, d0, pT, dT, Y, c) => `${a}: ${p0} RON on ${d0}, ${pT} RON on ${dT}; Y = ${Y} years, CAGR = (P<sub>T</sub>/P<sub>0</sub>)<sup>1/Y</sup> − 1 = ${c}%.`,
 pbw2: (dp, pp, dt, pt, m) => `Peak before the largest fall: ${pp} RON on ${dp}; trough: ${pt} RON on ${dt}; MDD = ${m}%.`,
 pbw3: (c, m, ca) => `Calmar = ${c}/${m} = ${ca}.`,
 pbw4: (dr, n) => `The price is back at the old peak on ${dr}, ${n} days after the peak.`,
 pbw4n: e => `By ${e}, the price has not returned to the old peak.`,
 pb: (d, c, m) => `Sample from ${d}. Highest Calmar ratio among the BVB stocks: ${c}; deepest MDD: ${m}.`,
 pb2: (a, n, done, nr) => `Longest maximum-drawdown episode: ${a}, ${n} days${done ? ' from peak to recovery' : ', not recovered'}. Not recovered by the end of the sample: ${nr}.`
};
""",
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (checked with a node script on ch1_data.js + common.js + ch1.js).
PREDICT = {
 'tlv-adjusted': (0, [('11 zile.', '11 days.'),
                      ('1 zi, consolidarea: dividendele și acțiunile gratuite mută prețul doar cu cîteva procente.', '1 day, the consolidation: dividends and bonus shares move the price by a few per cent only.'),
                      ('Tot 24: orice eveniment corporativ produce un salt de peste 10%.', 'Still 24: every corporate action produces a jump of more than 10%.')]),
 'bet-bettr': (0, [('Circa 8 puncte procentuale, ca pentru investiția din 2014.', 'About 8 percentage points, as for the 2014 investment.'),
                   ('Mult mai puțin: diferența valorilor finale scade de la 550 la 91 de unități.', 'Much less: the gap between the final values falls from 550 to 91 units.'),
                   ('Zero: dividendele contează numai pe orizonturi de peste zece ani.', 'Zero: dividends matter only over horizons of more than ten years.')]),
 'eurron': (2, [('Circa 13,4%: zilele greșite au variații pozitive și negative care se compensează.', 'About 13.4%: the wrong days have positive and negative changes that cancel out.'),
                ('Circa 12,3%: fără 8% din zile, volatilitatea scade cu 8%.', 'About 12.3%: without 8% of the days, volatility falls by 8%.'),
                ('Circa 4,7%, aproape de volatilitatea cursului BNR.', 'About 4.7%, close to the volatility of the BNR rate.')]),
 'price-returns': (1, [('Media și abaterea standard sînt practic aceleași în toate subperioadele.', 'The mean and the standard deviation are practically the same in every subperiod.'),
                       ('Media rămîne aproape de zero; abaterea standard variază între circa 1% și 1,5% pe zi.', 'The mean stays close to zero; the standard deviation varies between about 1% and 1.5% a day.'),
                       ('Media randamentelor crește de la o subperioadă la alta, ca media prețului.', 'The mean of returns rises from one subperiod to the next, like the mean of the price.')]),
 'simple-log': (1, [('2,7 puncte procentuale: diferența este simetrică.', '2.7 percentage points: the gap is symmetric.'),
                    ('3,8 puncte procentuale.', '3.8 percentage points.'),
                    ('3,2 puncte procentuale: diferența este R²/2 pentru orice R.', '3.2 percentage points: the gap is R²/2 for every R.')]),
 'growth': (1, [('Bitcoin: are cel mai mare randament cumulat pe 2015–2026.', 'Bitcoin: it has the largest cumulative return over 2015–2026.'),
                ('BET.', 'BET.'),
                ('S&P 500: are cea mai mică volatilitate dintre cele patru serii.', 'S&P 500: it has the lowest volatility of the four series.')]),
 'hist-normal': (0, [('De circa 113 ori.', 'About 113 times.'),
                     ('Tot de circa 5 ori: raportul nu depinde de prag.', 'Still about 5 times: the ratio does not depend on the threshold.'),
                     ('Mai puține decît distribuția Normală: dincolo de 4 abateri standard nu există zile.', 'Fewer than the Normal distribution: there are no days beyond 4 standard deviations.')]),
 'rolling-vol': (2, [('Circa 35%: lungimea ferestrei nu schimbă maximul.', 'About 35%: the window length does not change the maximum.'),
                     ('Circa 20%: o fereastră scurtă netezește volatilitatea.', 'About 20%: a short window smooths volatility.'),
                     ('Circa 97%.', 'About 97%.')]),
 'sharpe-ci': (0, [('Circa ± 1,02.', 'About ± 1.02.'),
                   ('Tot ± 0,57: eroarea standard depinde doar de volatilitate.', 'Still ± 0.57: the standard error depends only on volatility.'),
                   ('Circa ± 1,81: eroarea standard crește proporțional cu 1/Y.', 'About ± 1.81: the standard error grows in proportion to 1/Y.')]),
 'drawdowns': (2, [('0,3%, ca la S&P 500: scăderile adînci durează puțin.', '0.3%, as for the S&P 500: deep falls do not last long.'),
                   ('56%, ca la pragul de 20%: pragul nu schimbă ponderea.', '56%, as at the 20% threshold: the threshold does not change the share.'),
                   ('18%.', '18%.')]),
 'vol-drag': (0, [('Nu: rata de creștere este −8% pe an.', 'No: the growth rate is −8% a year.'),
                  ('Da: drag-ul se dublează la 9%, iar creșterea este 1% pe an.', 'Yes: the drag doubles to 9%, and growth is 1% a year.'),
                  ('Da: cu media aritmetică 10%, averea crește oricare ar fi σ.', 'Yes: with an arithmetic mean of 10%, wealth grows whatever σ is.')]),
 'risk-return': (1, [('Bitcoin: clasamentul pe 2015–2026 se păstrează.', 'Bitcoin: the 2015–2026 ranking holds.'),
                     ('Nuclearelectrica (SNN).', 'Nuclearelectrica (SNN).'),
                     ('BET-TR: are cel mai mare raport Sharpe pe 2015–2026.', 'BET-TR: it has the highest Sharpe ratio over 2015–2026.')]),
 'portfolio': (0, [('La 50%.', 'At 50%.'),
                   ('La 90%: TLV are volatilitatea mai mare, deci și diferența cea mai mare.', 'At 90%: TLV has the higher volatility, so also the largest gap.'),
                   ('Nicăieri: diferența este 0,093 pentru orice pondere.', 'Nowhere: the gap is 0.093 for every weight.')]),
 'annualise': (1, [('17,7%: regula √A dă aceeași volatilitate la orice frecvență.', '17.7%: the √A rule gives the same volatility at every frequency.'),
                   ('Circa 15%.', 'About 15%.'),
                   ('Circa 52%: la date lunare abaterea standard se înmulțește cu A = 12.', 'About 52%: with monthly data the standard deviation is multiplied by A = 12.')]),
 'mean-se': (0, [('Da.', 'Yes.'),
                 ('Nu: media estimată din 2020 este mai mare, 12,7%, deci intervalul se mută spre dreapta.', 'No: the mean estimated from 2020 is higher, 12.7%, so the interval moves to the right.'),
                 ('Nu: cu 1 686 de randamente zilnice, eroarea standard rămîne mică.', 'No: with 1,686 daily returns, the standard error stays small.')]),
 'desc-indices': (1, [('Rămîne circa 15,8: boltirea nu depinde de frecvență.', 'It stays about 15.8: kurtosis does not depend on the frequency.'),
                      ('Scade spre 0.', 'It falls towards 0.'),
                      ('Crește: fiecare lună conține mai multe zile extreme decît o zi.', 'It rises: each month contains more extreme days than a single day.')]),
 'desc-bvb': (1, [('Tot circa 25: două zile din 2 677 nu schimbă boltirea.', 'Still about 25: two days out of 2,677 do not change the kurtosis.'),
                  ('Circa 18.', 'About 18.'),
                  ('Circa 5, ca la SNN: toată boltirea TLV vine din cele două zile.', 'About 5, as for SNN: all the kurtosis of TLV comes from the two days.')]),
 'perf-indices': (0, [('Circa 0,44.', 'About 0.44.'),
                      ('Circa 0,67: se scad 0,05 din raportul Sharpe.', 'About 0.67: 0.05 is subtracted from the Sharpe ratio.'),
                      ('Practic tot 0,72: rata fără risc zilnică, 5%/252, este neglijabilă.', 'Practically still 0.72: the daily risk-free rate, 5%/252, is negligible.')]),
 'perf-bvb': (1, [('Nuclearelectrica: are cel mai mare CAGR pe 2015–2026.', 'Nuclearelectrica: it has the highest CAGR over 2015–2026.'),
                  ('OMV Petrom (SNP).', 'OMV Petrom (SNP).'),
                  ('Transgaz: are cel mai mare CAGR din 2022.', 'Transgaz: it has the highest CAGR from 2022.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout and those elements
VEIL = {'annualise': {'out': True, 'charts': ['an-tab', 'an-chart']}}

# Exit ticket „Verifică-te” / "Check Yourself": questions of the slides „Autoevaluare” and „Formule de reținut”.
# Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Un fond raportează randamente logaritmice zilnice cu media 0,05% și abaterea standard 1%. Care sînt valorile anuale ale mediei și volatilității (252 de zile pe an)?',
         'A fund reports daily log returns with mean 0.05% and standard deviation 1%. What are the annual mean and volatility (252 days a year)?'),
      key=1, opts=[
  (('0,79% și 15,9%.', '0.79% and 15.9%.'), ('Media se înmulțește cu 252, nu cu √252: 252 × 0,05% = 12,6%.', 'The mean is multiplied by 252, not by √252: 252 × 0.05% = 12.6%.')),
  (('12,6% și 15,9%.', '12.6% and 15.9%.'), ('Media se înmulțește cu A = 252, volatilitatea cu √252 = 15,87: 12,6% și 15,9%.', 'The mean is multiplied by A = 252, the volatility by √252 = 15.87: 12.6% and 15.9%.')),
  (('12,6% și 252%.', '12.6% and 252%.'), ('Varianța crește cu A, deci volatilitatea crește cu √A: √252 × 1% = 15,9%.', 'The variance grows with A, so the volatility grows with √A: √252 × 1% = 15.9%.'))]),
 dict(q=('Bitcoin, 2015–2026: media aritmetică a randamentelor simple este 69,5% pe an, CAGR 60,6%. De ce este media aritmetică mult peste CAGR?',
         'Bitcoin, 2015–2026: the arithmetic mean of simple returns is 69.5% a year, the CAGR 60.6%. Why is the arithmetic mean far above the CAGR?'),
      key=0, opts=[
  (('Volatility drag σ²/2 este mare cînd σ este mare.', 'The volatility drag σ²/2 is large when σ is large.'),
   ('Cu volatilitatea 66,2%, σ²/2 = 21,9%, aproape cît diferența dintre media aritmetică (69,5%) și media logaritmică (47,4%).', 'With a volatility of 66.2%, σ²/2 = 21.9%, close to the gap between the arithmetic mean (69.5%) and the log mean (47.4%).')),
  (('Media aritmetică este anualizată cu 365 de zile, CAGR nu.', 'The arithmetic mean is annualised with 365 days, the CAGR is not.'),
   ('Ambele măsoară creșterea pe un an calendaristic; numărul de zile pe an nu explică diferența.', 'Both measure growth per calendar year; the number of days a year does not explain the gap.')),
  (('Media aritmetică include cîteva erori de date, pe care CAGR le ignoră.', 'The arithmetic mean includes a few data errors that the CAGR ignores.'),
   ('Ambele sînt calculate din aceleași prețuri; diferența apare și pe date corecte, din concavitatea lui ln.', 'Both are computed from the same prices; the gap appears on correct data too, from the concavity of ln.'))]),
 dict(q=('Un indice scade cu 60%. Ce cîștig este necesar pentru revenirea la nivelul anterior?',
         'An index falls by 60%. What gain is needed to return to the previous level?'),
      key=2, opts=[
  (('60%.', '60%.'), ('Cîștigul se aplică unui nivel mai mic: 0,4 × 1,6 = 0,64, nu 1.', 'The gain applies to a lower level: 0.4 × 1.6 = 0.64, not 1.')),
  (('250%.', '250%.'), ('2,5 = 1/0,4 este randamentul brut; cîștigul net este 2,5 − 1 = 150%.', '2.5 = 1/0.4 is the gross return; the net gain is 2.5 − 1 = 150%.')),
  (('150%.', '150%.'), ('O pierdere x cere un cîștig x/(1 − x) = 0,6/0,4 = 150%.', 'A loss x needs a gain x/(1 − x) = 0.6/0.4 = 150%.'))]),
 dict(q=('Un preț crește de la 100 la 400 în 10 ani. Cît este CAGR?',
         'A price rises from 100 to 400 in 10 years. What is the CAGR?'),
      key=0, opts=[
  (('14,9% pe an.', '14.9% a year.'), ('CAGR = (P_T/P_0)^(1/Y) − 1 = 4^0,1 − 1 = 14,9%.', 'CAGR = (P_T/P_0)^(1/Y) − 1 = 4^0.1 − 1 = 14.9%.')),
  (('30% pe an.', '30% a year.'), ('Randamentul cumulat de 300% împărțit la 10 ani ignoră compunerea.', 'The cumulative return of 300% divided by 10 years ignores compounding.')),
  (('13,9% pe an.', '13.9% a year.'), ('ln 4/10 = 13,9% este media anuală a randamentelor logaritmice; CAGR = e^0,139 − 1 = 14,9%.', 'ln 4/10 = 13.9% is the annual mean of log returns; CAGR = e^0.139 − 1 = 14.9%.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch1-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH1_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH1_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH1_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    import re
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', ' ', t)
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
