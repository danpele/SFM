"""Build the interactive pages of Chapter 8: interactive/ch8.html (EN) and interactive/ch8_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch8.js, inputs in interactive/ch8_data.js (Quantlets/Ch_08/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch8.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 8
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_08'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('returns-bursts', 'sfm_ch8_returns_bursts', 'def'), ('rolling-windows', 'sfm_ch8_rolling_windows', 'hist'),
          ('ewma-weights', 'sfm_ch8_ewma_weights', 'ewma'), ('ghost', 'sfm_ch8_ghost', 'ewma'),
          ('ohlc-path', 'sfm_ch8_ohlc_path', 'range'), ('efficiency', 'sfm_ch8_efficiency', 'range'), ('range-rolling', 'sfm_ch8_range_rolling', 'range'),
          ('realised', 'sfm_ch8_realised', 'rv'), ('signature', 'sfm_ch8_signature', 'rv'),
          ('acf-clustering', 'sfm_ch8_acf_clustering', 'cl'), ('shuffle', 'sfm_ch8_shuffle', 'cl'), ('kurtosis', 'sfm_ch8_kurtosis', 'cl'),
          ('long-term', 'sfm_ch8_long_term', 'lt'), ('vix', 'sfm_ch8_vix', 'lt'), ('leverage', 'sfm_ch8_leverage', 'lt'), ('news-impact', 'sfm_ch8_news_impact', 'lt'),
          ('forecast-paths', 'sfm_ch8_forecast_paths', 'fc'), ('lambda', 'sfm_ch8_lambda', 'fc')]
# table and worked-example blocks: (anchor, frame number in the deck (RO and EN alike), section key, chart block that carries it on the web page)
XBLOCKS = [('vol-table', 10, 'def', 'returns-bursts'), ('annual', 12, 'def', 'returns-bursts'), ('precision', 16, 'hist', 'rolling-windows'),
           ('ewma-update', 20, 'ewma', 'ewma-weights'), ('parkinson', 26, 'range', 'ohlc-path'), ('gk-rs', 27, 'range', 'ohlc-path'), ('yz', 28, 'range', 'ohlc-path'),
           ('march16', 29, 'range', 'ohlc-path'), ('eff-table', 31, 'range', 'efficiency'), ('est-table', 32, 'range', 'efficiency'), ('rv-def', 35, 'rv', 'range-rolling'),
           ('acf-table', 40, 'cl', 'acf-clustering'), ('lb-sq', 42, 'cl', 'shuffle'), ('arch-lm', 43, 'cl', 'shuffle'), ('cl-table', 44, 'cl', 'shuffle'),
           ('vix-table', 50, 'lt', 'vix'), ('loss', 55, 'fc', 'forecast-paths'), ('fc-table', 56, 'fc', 'forecast-paths')]
ORDER = ['returns-bursts', 'vol-table', 'annual', 'rolling-windows', 'precision', 'ewma-weights', 'ewma-update', 'ghost', 'ohlc-path', 'parkinson', 'gk-rs', 'yz',
         'march16', 'efficiency', 'eff-table', 'est-table', 'range-rolling', 'rv-def', 'realised', 'signature', 'acf-clustering', 'acf-table', 'shuffle', 'lb-sq',
         'arch-lm', 'cl-table', 'kurtosis', 'long-term', 'vix', 'vix-table', 'leverage', 'news-impact', 'loss', 'fc-table', 'forecast-paths', 'lambda']
TITLES = {'vol-table': 'Interpretarea volatilității pe patru piețe', 'annual': 'Exemplu rezolvat: anualizarea volatilității',
          'precision': 'Precizia unei volatilități de selecție', 'ewma-update': 'Exemplu rezolvat: o actualizare EWMA', 'parkinson': 'Estimatorul Parkinson (1980)',
          'gk-rs': 'Garman–Klass (1980) și Rogers–Satchell (1991)', 'yz': 'Yang–Zhang (2000): includerea randamentului overnight', 'march16': 'Exemplu rezolvat: S&P 500 pe 16 martie 2020',
          'eff-table': 'Interpretarea simulării: precizie și deplasare', 'est-table': 'Estimatorii volatilității pentru cinci active',
          'rv-def': 'Varianța realizată din randamentele intrazilnice', 'acf-table': 'Interpretarea volatility clustering pe trei piețe',
          'lb-sq': 'Testarea volatility clustering: Ljung–Box pe randamentele la pătrat', 'arch-lm': 'Testul ARCH-LM (Engle, 1982)',
          'cl-table': 'Testele de volatility clustering pe șase serii', 'vix-table': 'Interpretarea relației dintre volatilitatea implicită și cea realizată',
          'loss': 'Funcții de pierdere: MSE și QLIKE', 'fc-table': 'S&P 500: compararea prognozelor'}

PFX = {'returns-bursts': 'rb', 'vol-table': 'vt', 'annual': 'an', 'rolling-windows': 'rw', 'precision': 'pr', 'ewma-weights': 'ew', 'ewma-update': 'eu', 'ghost': 'gh',
       'ohlc-path': 'op', 'parkinson': 'pk', 'gk-rs': 'gr', 'yz': 'yz', 'march16': 'ms', 'efficiency': 'ef', 'eff-table': 'et', 'est-table': 'es', 'range-rolling': 'rr',
       'rv-def': 'rd', 'realised': 're', 'signature': 'sg', 'acf-clustering': 'ac', 'acf-table': 'at', 'shuffle': 'sh', 'lb-sq': 'lq', 'arch-lm': 'al', 'cl-table': 'ct',
       'kurtosis': 'ku', 'long-term': 'lo', 'vix': 'vx', 'vix-table': 'vi', 'leverage': 'lv', 'news-impact': 'ni', 'loss': 'ls', 'fc-table': 'ft', 'forecast-paths': 'fp',
       'lambda': 'la'}

SECTIONS = {
 'ro': dict(**{'def': 'Volatilitatea: definiție și caracter latent'}, hist='Volatilitatea istorică și alegerea ferestrei', ewma='EWMA și RiskMetrics',
            range='Estimatori de amplitudine (range-based)', rv='Volatilitatea realizată', cl='Volatility clustering',
            lt='Comportamentul pe termen lung, VIX și efectul de levier', fc='Evaluarea prognozelor de volatilitate'),
 'en': dict(**{'def': 'What Is Volatility?'}, hist='Historical Volatility and the Choice of the Window', ewma='EWMA and RiskMetrics', range='Range-Based Estimators',
            rv='Realised Volatility', cl='Volatility Clustering', lt='Long-Run Behaviour, the VIX and the Leverage Effect', fc='Evaluating Volatility Forecasts'),
}

L = {
 'ro': dict(file='ch8_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 8 · Estimatori ai volatilității', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: CC (close-to-close, închidere–închidere), P (Parkinson), GK (Garman–Klass), RS (Rogers–Satchell), '
        'YZ (Yang–Zhang), OHLC (open, high, low, close: deschidere, maxim, minim, închidere), EWMA (exponentially weighted moving average, media mobilă ponderată exponențial), '
        'RV (realised variance, varianța realizată), ACF (autocorrelation function, funcția de autocorelație), ARCH-LM (testul multiplicatorului Lagrange pentru efecte ARCH), '
        'GARCH (generalised autoregressive conditional heteroskedasticity), VIX (indicele de volatilitate al Cboe), MSE (mean squared error, eroarea pătratică medie), '
        'QLIKE (quasi-likelihood, cvasi-verosimilitate), MZ (Mincer–Zarnowitz), DM (Diebold–Mariano).',
   src='Randamente logaritmice zilnice, în %: S&P 500 și DAX din ianuarie 1990, BET din septembrie 1997, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV) '
       'și OMV Petrom (SNP) din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. Prețuri OHLC: S&P 500 din 2008, '
       'DAX din 2006, Bitcoin din 2014, TLV și SNP din 2010 (acțiunile ajustate cu factorul prețului de închidere). VIX: închiderile zilnice, pe calendarul S&P 500. Prețuri EODHD. '
       'Ziua simulată OHLC, simularea eficienței (1 000 de ferestre), prețurile simulate la un minut și amestecarea randamentelor sînt cele din curs (același cod, aceeași sămînță '
       'a generatorului); volatilitățile, estimatorii de amplitudine, autocorelațiile, testele Ljung–Box și ARCH-LM, volatilitățile anuale și lunare, comparația cu VIX, '
       'corelațiile de levier, curbele neparametrice și pierderile prognozelor sînt recalculate în browser din aceleași date. Alte semințe, alte ferestre ale simulării, alte '
       'niveluri ale zgomotului, amestecarea pe blocuri, alte începuturi ale eșantionului, alte orizonturi, laguri și lățimi de bandă sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch8.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 8 · Volatility Estimators', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: CC (close-to-close), P (Parkinson), GK (Garman–Klass), RS (Rogers–Satchell), YZ (Yang–Zhang), '
        'OHLC (open, high, low, close), EWMA (exponentially weighted moving average), RV (realised variance), ACF (autocorrelation function), ARCH-LM (Lagrange multiplier test '
        'for ARCH effects), GARCH (generalised autoregressive conditional heteroskedasticity), VIX (Cboe volatility index), MSE (mean squared error), QLIKE (quasi-likelihood), '
        'MZ (Mincer–Zarnowitz), DM (Diebold–Mariano).',
   src='Daily log returns, in %: S&P 500 and DAX from January 1990, BET from September 1997, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV) and OMV Petrom '
       '(SNP) from 2010, from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. OHLC prices: S&P 500 from 2008, DAX from 2006, Bitcoin '
       'from 2014, TLV and SNP from 2010 (stocks adjusted with the factor of the close). VIX: daily closes on the S&P 500 calendar. EODHD prices. The simulated OHLC day, the '
       'efficiency simulation (1,000 windows), the simulated one-minute prices and the shuffle of the returns are those of the lecture (the same code and the same generator '
       'seed); the volatilities, the range-based estimators, the autocorrelations, the Ljung–Box and ARCH-LM tests, the annual and monthly volatilities, the VIX comparison, the '
       'leverage correlations, the nonparametric curves and the forecast losses are recomputed in the browser from the same data. Other seeds, other windows of the '
       'simulation, other noise levels, the block shuffle, other sample starts, horizons, lags and bandwidths are extra runs of the same code.'),
}

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child,.xt td.l,.xt th.l{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt thead tr:first-child th[colspan]{text-align:center;border-bottom:1px solid #C5D2E8}
.xt .hi{font-weight:700;color:#2E7D32}.xt .bad{color:#CD0000}
.xt tr.sel td{background:#EEF3FB}
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
    M4 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
    S6 = M4 + [('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)')]
    O5 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)')]
    ser = 'Seria' if ro else 'Series'
    wser = 'Seria exemplului lucrat' if ro else 'Series of the worked example'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    y0 = 'Începutul eșantionului (anul)' if ro else 'Start of the sample (year)'
    lec = ' (curs)' if ro else ' (lecture)'
    c = (lambda s: s.replace('.', '{,}')) if ro else (lambda s: s)
    NW = [('5', '5'), ('10', '10'), ('21', '21' + lec), ('63', '63')]
    MM = [(str(m), str(m) + (lec if m == 5 else '')) for m in (1, 2, 3, 5, 10, 15, 30, 65, 130, 390)]
    H5 = [(str(h), str(h) + (lec if h == 21 else '')) for h in (5, 10, 21, 42, 63)]
    year = 'Anul' if ro else 'Year'
    return {
     'returns-bursts': (slider('rb-k', 'Pragul: |r| peste k abateri standard' if ro else 'Threshold: |r| above k standard deviations', 3, 10, 1, 5) + play('rb'), CSS + charts('rb-chart')),
     'vol-table': (slider('vt-y', y0, 1990, 2022, 2, 1990) + play('vt'),
                   charts('vt-chart') + xh('Volatilitatea pe patru piețe' if ro else 'Volatility in four markets') + tw('vt-tab')),
     'annual': (select('an-c', 'Convenția de anualizare' if ro else 'Annualisation convention',
                       [('act', 'frecvența reală, ca în curs' if ro else 'actual frequency, as in the lecture'), ('252', 'A = 252'), ('250', 'A = 250'), ('260', 'A = 260'), ('365', 'A = 365')], 'act') + play('an'),
                charts('an-chart') + we('an-we', f'{WE}: {"anualizarea" if ro else "annualisation"}',
                                        c(r'\sigma_{\text{an}} = \sigma_{\text{zi}}\sqrt{A}, \qquad A = \frac{n}{D/365.25}') if ro else r'\sigma_{\text{year}} = \sigma_{\text{day}}\sqrt{A}, \qquad A = \frac{n}{D/365.25}',
                                        select('an-s', wser, M4, 'btc'))),
     'rolling-windows': (slider('rw-n', 'Fereastra scurtă n (zile; 63 și 252 rămîn fixe)' if ro else 'Short window n (days; 63 and 252 stay fixed)', 5, 126, 1, 21) + play('rw'), charts('rw-chart')),
     'precision': (slider('pr-n', 'Fereastra n (zile)' if ro else 'Window n (days)', 5, 252, 1, 21) + select('pr-s', ser, M4, 'sp500') + play('pr'),
                   charts('pr-chart') + we('pr-we', f'{WE}: {"intervalul pentru ultima volatilitate" if ro else "the interval for the latest volatility"}',
                                           r'\frac{\mathrm{SE}(\hat\sigma)}{\sigma} \approx \frac{1}{\sqrt{2n}}, \qquad \frac{\mathrm{SE}(\hat\sigma)}{\sigma} \approx \sqrt{\frac{\kappa - 1}{4n}}, \qquad \hat\sigma\,\big(1 \pm 1.96\,\mathrm{SE}/\sigma\big)'.replace('1.96', '1{,}96' if ro else '1.96'))),
     'ewma-weights': (slider('ew-l', 'Factorul de atenuare λ (linia roșie)' if ro else 'Decay factor λ (red line)', 0.80, 0.99, 0.01, 0.94)
                      + slider('ew-n', 'Fereastra mobilă n (zile)' if ro else 'Rolling window n (days)', 10, 252, 1, 63) + play('ew'), charts('ew-chart')),
     'ewma-update': (slider('eu-l', 'Factorul de atenuare λ' if ro else 'Decay factor λ', 0.80, 0.99, 0.01, 0.94)
                     + slider('eu-r', 'Randamentul de ieri r(t − 1) (%)' if ro else "Yesterday's return r(t − 1) (%)", -8, 8, 0.5, -3) + play('eu'),
                     charts('eu-chart') + we('eu-we', f'{WE}: {"o actualizare EWMA" if ro else "one EWMA update"}', r'\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2',
                                             select('eu-x', 'Exemplul' if ro else 'Example', [('deck', 'σ²(t − 1) = 1,00 și r(t − 1) ales (curs)' if ro else 'σ²(t − 1) = 1.00 and the chosen r(t − 1) (lecture)'),
                                                                                              ('sp', 'S&P 500, 16 martie 2020' if ro else 'S&P 500, 16 March 2020')], 'deck'))),
     'ghost': (slider('gh-n', 'Fereastra mobilă n (zile)' if ro else 'Rolling window n (days)', 21, 126, 21, 63)
               + select('gh-l', 'λ al EWMA' if ro else 'EWMA λ', [('0.9', '0,90' if ro else '0.90'), ('0.94', '0,94' + lec if ro else '0.94' + lec), ('0.97', '0,97' if ro else '0.97')], '0.94') + play('gh'),
               charts('gh-chart')),
     'ohlc-path': (select('op-s', 'Sămînța generatorului' if ro else 'Generator seed', [(str(s), str(s) + (lec if s == 3 else '')) for s in range(1, 11)], '3')
                   + slider('op-o', 'Saltul overnight o (%)' if ro else 'Overnight gap o (%)', -1, 1, 0.05, 0.35) + play('op'), charts('op-chart')),
     'parkinson': (select('pk-s', ser, O5, 'sp500') + slider('pk-y', year, 2006, 2026, 1, 2020) + play('pk'),
                   charts('pk-chart') + we('pk-we', f'{WE}: {"Parkinson din H și L, ziua cu cea mai mare amplitudine a anului" if ro else "Parkinson from H and L, the day with the widest range of the year"}',
                                           r'\hat\sigma^2_{P,t} = \frac{(u_t - d_t)^2}{4\ln 2}, \qquad u_t - d_t = \ln(H_t/L_t)')),
     'gk-rs': (select('gr-s', ser, O5, 'btc') + slider('gr-y', year, 2006, 2026, 1, 2021) + play('gr'),
               charts('gr-chart') + we('gr-we', f'{WE}: Garman–Klass {"și" if ro else "and"} Rogers–Satchell',
                                       c(r'\hat\sigma^2_{GK,t} = 0.5\,(u_t - d_t)^2 - (2\ln 2 - 1)\,c_t^2, \qquad \hat\sigma^2_{RS,t} = u_t(u_t - c_t) + d_t(d_t - c_t)'))),
     'yz': (slider('yz-n', 'Fereastra n (zile, pînă la 31 martie 2020)' if ro else 'Window n (days, to 31 March 2020)', 2, 126, 1, 21) + play('yz'),
            charts('yz-chart') + we('yz-we', f'{WE}: {"ponderea k și estimatorul Yang–Zhang" if ro else "the weight k and the Yang–Zhang estimator"}',
                                    c(r'\hat\sigma^2_{YZ} = \hat\sigma_O^2 + k\,\hat\sigma_C^2 + (1-k)\,\hat\sigma_{RS}^2, \qquad k = \frac{0.34}{1.34 + (n+1)/(n-1)}'), select('yz-s', wser, O5, 'sp500'))),
     'march16': (select('ms-s', ser, O5, 'sp500') + select('ms-d', 'Ziua (cele 10 zile cu cel mai mare |r|)' if ro else 'Day (the 10 days with the largest |r|)', [('2020-03-16', '2020-03-16')], '2020-03-16') + play('ms'),
                 charts('ms-chart') + we('ms-we', f'{WE}: {"o zi, șase estimări" if ro else "one day, six estimates"}',
                                         r'o_t = \ln\frac{O_t}{C_{t-1}}, \quad u_t = \ln\frac{H_t}{O_t}, \quad d_t = \ln\frac{L_t}{O_t}, \quad c_t = \ln\frac{C_t}{O_t}, \quad r_t = o_t + c_t')),
     'efficiency': (select('ef-n', 'Fereastra n (zile)' if ro else 'Window n (days)', NW, '21')
                    + select('ef-s', 'Panoul din dreapta' if ro else 'Right panel', [('overnight', 'salt overnight' + lec if ro else 'overnight jump' + lec), ('discrete', '26 de prețuri pe zi' if ro else '26 prices a day')], 'overnight')
                    + play('ef'), charts('ef-chart')),
     'eff-table': (select('et-n', 'Fereastra n (zile)' if ro else 'Window n (days)', NW, '21') + play('et'),
                   charts('et-chart') + xh('Eficiență, deplasare și raportul MSE (1 000 de ferestre)' if ro else 'Efficiency, bias and MSE ratio (1,000 windows)') + tw('et-tab')),
     'est-table': (slider('es-y', 'Începutul eșantionului OHLC (anul; 2006 = începutul fiecărei serii)' if ro else 'Start of the OHLC sample (year; 2006 = start of each series)', 2006, 2024, 2, 2006) + play('es'),
                   charts('es-chart') + xh('Volatilitatea anualizată (%) pe eșantionul OHLC' if ro else 'Annualised volatility (%) on the OHLC sample') + tw('es-tab')),
     'range-rolling': (slider('rr-n', 'Fereastra n (zile)' if ro else 'Window n (days)', 5, 63, 1, 21) + play('rr'), charts('rr-chart')),
     'rv-def': (slider('rd-d', 'Ziua simulată' if ro else 'Simulated day', 1, 60, 1, 1) + select('rd-m', 'Intervalul de eșantionare (minute)' if ro else 'Sampling interval (minutes)', MM, '5') + play('rd'),
                charts('rd-chart') + we('rd-we', f'{WE}: {"varianța realizată a unei zile" if ro else "the realised variance of one day"}',
                                        r'\mathrm{RV}_t = \sum_{i=1}^{M} r_{t,i}^2, \qquad M = 390/m')),
     'realised': (select('re-m', 'Intervalul randamentelor RV (minute)' if ro else 'Interval of the RV returns (minutes)', MM, '5') + play('re'), charts('re-chart')),
     'signature': (select('sg-z', 'Abaterea standard a zgomotului (% din preț)' if ro else 'Standard deviation of the noise (% of the price)',
                          [('0.0', '0'), ('0.005', '0,005' if ro else '0.005'), ('0.01', '0,01' if ro else '0.01'), ('0.025', ('0,025' if ro else '0.025') + lec), ('0.05', '0,05' if ro else '0.05'), ('0.1', '0,1' if ro else '0.1')], '0.025')
                   + play('sg'), charts('sg-chart')),
     'acf-clustering': (slider('ac-y', y0, 1990, 2020, 2, 1990) + play('ac'), charts('ac-chart')),
     'acf-table': (slider('at-k', 'Lagul k' if ro else 'Lag k', 1, 100, 1, 1) + play('at'),
                   charts('at-chart') + xh('Autocorelațiile lui |r| și r² (în afara benzii: din 100 de laguri)' if ro else 'Autocorrelations of |r| and r² (outside the band: of 100 lags)') + tw('at-tab')),
     'shuffle': (select('sh-b', 'Amestecăm blocuri de b zile' if ro else 'We shuffle blocks of b days',
                        [('1', 'b = 1, zile individuale' + lec if ro else 'b = 1, single days' + lec), ('5', '5'), ('21', '21'), ('63', '63'), ('252', '252')], '1') + play('sh'), charts('sh-chart')),
     'lb-sq': (select('lq-s', ser, S6, 'sp500') + slider('lq-m', 'Laguri testate m' if ro else 'Lags tested m', 1, 20, 1, 3) + play('lq'),
               charts('lq-chart') + we('lq-we', f'{WE}: {"testul McLeod–Li" if ro else "the McLeod–Li test"}',
                                       r'Q(m) = T(T+2)\sum_{k=1}^{m}\frac{\hat\rho_k(r^2)^2}{T-k} \approx \chi^2(m)')),
     'arch-lm': (select('al-s', ser, S6, 'sp500') + slider('al-q', 'Laguri q' if ro else 'Lags q', 1, 20, 1, 5) + play('al'),
                 charts('al-chart') + we('al-we', f'{WE}: {"testul ARCH-LM" if ro else "the ARCH-LM test"}',
                                         r'e_t^2 = c_0 + c_1 e_{t-1}^2 + \dots + c_q e_{t-q}^2 + u_t, \qquad \mathrm{LM} = nR^2 \approx \chi^2(q)')),
     'cl-table': (slider('ct-m', 'Laguri Ljung–Box m' if ro else 'Ljung–Box lags m', 1, 30, 1, 10) + select('ct-q', 'Laguri ARCH-LM q' if ro else 'ARCH-LM lags q', [(str(i), str(i)) for i in range(1, 11)], '5') + play('ct'),
                  charts('ct-chart') + xh('Teste pe randamentele zilnice (cu roșu: p-value sub 0,05)' if ro else 'Tests on the daily returns (in red: p-value below 0.05)') + tw('ct-tab')),
     'kurtosis': (slider('ku-s', 'Raportul abaterilor standard, zile agitate / zile liniștite' if ro else 'Ratio of standard deviations, turbulent / calm days', 1.5, 5, 0.5, 2)
                  + slider('ku-b', 'β al GARCH(1,1)' if ro else 'β of the GARCH(1,1)', 0.70, 0.95, 0.01, 0.85) + '<input type="hidden" id="ku-t" value="1">' + play('ku'), charts('ku-chart')),
     'long-term': (slider('lo-y', y0, 1990, 2020, 2, 1990) + select('lo-s', 'Seria din panoul din dreapta' if ro else 'Series of the right panel', M4, 'sp500') + play('lo'), charts('lo-chart')),
     'vix': (select('vx-h', 'Orizontul h al volatilității realizate (zile de tranzacționare)' if ro else 'Horizon h of the realised volatility (trading days)', H5, '21') + play('vx'), charts('vx-chart')),
     'vix-table': (select('vi-h', 'Orizontul h (zile de tranzacționare)' if ro else 'Horizon h (trading days)', H5, '21') + slider('vi-y', y0, 1990, 2020, 2, 1990) + play('vi'),
                   charts('vi-chart') + xh('VIX și volatilitatea realizată în următoarele h zile' if ro else 'VIX and the realised volatility of the next h days') + tw('vi-tab')),
     'leverage': (slider('lv-y', y0, 1990, 2020, 2, 1990) + select('lv-m', 'Volatilitatea viitoare măsurată prin' if ro else 'Future volatility measured by', [('abs', '|r(t + j)|' + lec), ('sq', 'r(t + j)²')], 'abs') + play('lv'),
                  charts('lv-chart')),
     'news-impact': (slider('ni-h', 'Lățimea de bandă h (randamente zecimale)' if ro else 'Bandwidth h (decimal returns)', 0.01, 0.1, 0.01, 0.04) + play('ni'), charts('ni-chart')),
     'loss': (slider('ls-p', 'Variabila proxy σ̂²' if ro else 'Proxy σ̂²', 0.25, 4, 0.25, 1) + play('ls'),
              charts('ls-chart') + we('ls-we', f'{WE}: MSE {"și" if ro else "and"} QLIKE', r'L_{MSE} = (\hat\sigma_t^2 - h_t)^2, \qquad L_{QLIKE} = \frac{\hat\sigma_t^2}{h_t} + \ln h_t')),
     'fc-table': (select('ft-p', 'Variabila proxy din grafic' if ro else 'Proxy in the chart', [('r2', 'r²' + lec), ('range', 'o² + Garman–Klass')], 'r2')
                  + slider('ft-y', 'Începutul evaluării (anul)' if ro else 'Start of the evaluation (year)', 2010, 2022, 2, 2010) + play('ft'),
                  charts('ft-chart') + xh('Prognoze ale varianței pentru ziua următoare (cu verde: cea mai mică pierdere)' if ro else 'Next-day variance forecasts (in green: the smallest loss)') + tw('ft-tab')),
     'forecast-paths': (slider('fp-y', year, 2008, 2025, 1, 2020) + select('fp-l', 'λ al EWMA' if ro else 'EWMA λ', [('0.94', ('0,94' if ro else '0.94') + lec), ('0.97', '0,97' if ro else '0.97')], '0.94') + play('fp'),
                        charts('fp-chart')),
     'lambda': (slider('la-y', 'Începutul evaluării (anul)' if ro else 'Start of the evaluation (year)', 2010, 2022, 2, 2010) + play('la'), charts('la-chart')),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    ro = frames_of(open(os.path.join(ROOT, TEX['ro']) + '.tex', encoding='utf-8').read())
    res = {}
    for a, fr, _, _ in XBLOCKS:
        assert latex_text(ro[fr - 1][0]) == TITLES[a], (a, latex_text(ro[fr - 1][0]))
        k = fr - 1
        assert not re.search(r'includegraphics[^{]*\{sfm_', frames[k][1]), (lang, a)
        res[a] = (int(pages[k][0]), latex_text(frames[k][0]))
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch8.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch8_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    html = html[:start] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[end:]
    return html.replace('</head>', MATHJAX + '</head>', 1)


def build():
    import pymupdf
    import interactive_common as ic
    os.makedirs(os.path.join(ic.OUT, 'slides'), exist_ok=True)
    for lang in ('en', 'ro'):
        with open(os.path.join(ic.OUT, L[lang]['file']), 'w', encoding='utf-8') as f:
            f.write(page(lang))
        with open(os.path.join(ic.OUT, f'ch{N}_text_{lang}.js'), 'w', encoding='utf-8') as f:
            f.write(f'// Readouts of the Chapter {N} interactive page, written by latex/build_interactive_ch{N}.py.\n' + TEXT_JS[lang])
        doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
        import types
        sp = types.SimpleNamespace(TEX=TEX, BLOCKS=BLOCKS)
        for chart, (pg, _) in ic.slide_pages(sp, lang).items():
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'{chart}_{lang}.jpg'), jpg_quality=82)
        for a, (pg, _) in frame_pages(lang).items():
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch8_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => `${s} ${de(s)}zile`;
const minute = m => (+m === 1 ? 'un minut' : `${m} ${de(m)}minute`);
const ML = { hist21: 'fereastra de 21 de zile', hist63: 'fereastra de 63 de zile', hist252: 'fereastra de 252 de zile', ewma94: 'EWMA 0,94', ewma97: 'EWMA 0,97', vix: 'VIX' };
window.CH8_TEXT = {
 h: { from: 'din', annvol: 'volatilitate anuală (%)', maxmove: 'cea mai mare variație (%)', date: 'data', insd: 'în abateri standard', exk: 'exces de boltire', night: 'ponderea nopții',
      eff_c: 'Eficiența, tranzacționare continuă', bias_c: 'Deplasarea (%), tranzacționare continuă', bias_o: 'Deplasarea (%), salt overnight', mse_o: 'Raportul MSE, salt overnight',
      bias_d: 'Deplasarea (%), 26 de prețuri pe zi', outabs: '|r| în afara benzii', bestd: 'd cu ACF maxim', days: 'zile', mvix: 'media VIX', mrv: 'media realizată', above: 'VIX peste realizată',
      gap: 'diferența medie', r2past: 'R², realizată trecută', mz_r2: 'MZ R², proxy r²', mz_rg: 'MZ R², proxy de amplitudine', mse_rg: 'MSE, proxy de amplitudine', ql_rg: 'QLIKE, proxy de amplitudine',
      ml: { hist21: '21 de zile', hist63: '63 de zile', hist252: '252 de zile', ewma94: 'EWMA 0,94', ewma97: 'EWMA 0,97', vix: 'VIX' } },
 rb: (k, items) => `Zile cu |r| peste ${k} abateri standard ale propriei serii (în paranteză: numărul așteptat dacă randamentele ar urma distribuția Normală): ` + items.map(([s, n, e, m]) => `${s} ${n} (${e}), cea mai mare variație: ${m} abateri standard`).join('; ') + '.',
 vt: (y, a, b, items) => `Eșantioane de la ${y} (sau de la începutul seriei): ` + items.map(([s, v, e]) => `${s} ${v}% pe an, exces de boltire ${e}`).join('; ') + `. Cea mai mare variație zilnică a fiecărei serii reprezintă între ${a} și ${b} abateri standard.`,
 an: (cv, items) => (cv === null ? 'Fiecare serie anualizată cu propria frecvență: ' + items.map(([s, v]) => `${s} ${v}%`).join('; ') + '.'
   : `Cu A = ${cv} pentru toate seriile: ` + items.map(([s, v, w]) => `${s} ${w}% în loc de ${v}%`).join('; ') + '. Se schimbă vizibil doar seriile al căror calendar este departe de A.'),
 an1: (s, n, y, A, sd) => `${s}: ${n} ${de(n)}randamente în ${y} ani, deci A = ${A} observații pe an; σ̂ = ${sd}% pe zi.`,
 an2: (sd, A, r, v) => `Cu frecvența reală: σ̂√A = ${sd} × √${A} = ${sd} × ${r} = ${v}%.`,
 an3: (A, sd, r, v, p) => `Cu A = ${A}: ${sd} × √${A} = ${sd} × ${r} = ${v}%, o abatere de ${p}% față de valoarea cu frecvența reală.`,
 rw: (items, w) => items.map(([m, mx, d, sc, l]) => `Fereastra de ${zile(m)}: maximul ${mx}% (${d}), variația zilnică tipică a lui ln σ̂ ${sc}%, ultima valoare ${l}%`).join('; ') + `. Volatilitatea pe întregul eșantion: ${w}%.`,
 pr1: (s, k, n, i, p) => `${s}: coeficientul de boltire κ = ${k}; cu randamente Normale, SE(σ̂)/σ ≈ 1/√(2 × ${n}) = ${i}, adică ${p}%.`,
 pr2: (k, n, p) => `Cu boltirea observată: √((${k} − 1)/(4 × ${n})) = ${p}%.`,
 pr3: (n, v, a, b, c, d) => `Ultima volatilitate pe ${zile(n)}: ${v}%; intervalul aproximativ de 95%: [${a}%; ${b}%] cu randamente Normale, [${c}%; ${d}%] cu boltirea observată.`,
 pr: (s, n, a, b, r) => `${s}, n = ${n}: eroarea standard relativă ${a}% (randamente Normale) și ${b}% (boltirea observată), de ${r} ori mai mare. Benzile din grafic sînt σ̂(1 ± 1,96 SE/σ) în fiecare zi.`,
 ew: (l, h, lag, d, w, n, wn, d97) => `λ = ${l}: ponderea zilei de ieri ${w}, timpul de înjumătățire ${h} zile, lagul mediu 1/(1 − λ) = ${lag} zile, 99% din pondere în ultimele ${d} zile (λ = 0,97: ${d97} zile). Fereastra de ${zile(n)} dă fiecărei zile ponderea 1/${n} = ${wn} și 0 zilelor mai vechi.`,
 eu1: (lab, s0, sg, r) => (lab === null ? `Varianța EWMA de ieri σ²(t − 1) = ${s0} (%²), adică ${sg}% pe zi; randamentul de ieri r(t − 1) = ${r}%.`
   : `S&P 500, ${lab}: varianța EWMA cunoscută la închiderea zilei precedente σ²(t − 1) = ${s0} (%²), adică ${sg}% pe zi; randamentul zilei r = ${r}%.`),
 eu2: (l, s0, a) => `Pasul 1: λσ²(t − 1) = ${l} × ${s0} = ${a}.`,
 eu3: (w, r2, a) => `Pasul 2: (1 − λ)r² = ${w} × ${r2} = ${a}.`,
 eu4: (a, b, s, sg, an) => `Pasul 3: σ²(t) = ${a} + ${b} = ${s}, deci σ(t) = ${sg}% pe zi, ${sg} × √252 = ${an}% pe an.`,
 eu5: (l, s, ls, r2, w) => `Dacă ziua următoare are r = 0: σ²(t + 1) = ${l} × ${s} = ${ls}. O fereastră de 63 de zile ar adăuga ${r2}/63 = ${w} la varianță și ar elimina această contribuție dintr-odată, după 63 de zile.`,
 eu: (l, r, e1, w1, eW, W) => `λ = ${l}, șocul r = ${r}% după o varianță egală cu 1: a doua zi volatilitatea EWMA este ${e1}%, iar cea pe fereastra de ${W} de zile ${w1}%; după ${W} de zile EWMA a coborît la ${eW}%, iar fereastra revine brusc la 1%.`,
 gh: (n, dd, b, a, ld, lr, l, ewd, ep, epd, rp, rpd) => `Fereastra de ${zile(n)}: cea mai mare scădere într-o zi între aprilie și septembrie 2020 are loc pe ${dd}, de la ${b}% la ${a}%, cînd randamentul din ${ld} (${lr}%) iese din fereastră; în aceeași zi EWMA (λ = ${l}) se modifică cu ${ewd} puncte procentuale. Vîrfurile din 2020: EWMA ${ep}% (${epd}), fereastra ${rp}% (${rpd}).`,
 op: (seed, o, u, d, c, r, r2, rg, P, GK, RS) => `Sămînța ${seed}: o = ${o}%, u = ${u}%, d = ${d}%, c = ${c}%; randamentul zilei r = o + c = ${r}%, r² = ${r2}; amplitudinea u − d = ${rg}%, Parkinson ${P}, Garman–Klass ${GK}, Rogers–Satchell ${RS} (%²). Saltul overnight schimbă r², dar nu și amplitudinea.`,
 pk1: (s, dt, H, L, rg) => `${s}, ${dt} (ziua cu cea mai mare amplitudine a anului): H = ${H}, L = ${L}; u − d = 100 ln(H/L) = ${rg}%.`,
 pk2: (rg, r2, k, P, sp) => `σ̂²<sub>P</sub> = ${rg}²/(4 ln 2) = ${r2}/${k} = ${P} (%²); σ̂<sub>P</sub> = ${sp}% pe zi.`,
 pk3: (cc, c2, P) => `Pentru comparație, randamentul închidere–închidere al zilei: r = ${cc}%, r² = ${c2}, față de estimarea Parkinson ${P}.`,
 pk: (s, y, n, a, b, d) => `${s}, ${y}, ${zile(n)}: √(media estimărilor Parkinson) = ${a}% pe zi, abaterea standard închidere–închidere ${b}% pe zi; abaterea standard a diferenței zilnice dintre √Parkinson și |r|: ${d} puncte procentuale.`,
 gr1: (s, dt, u, d, c) => `${s}, ${dt} (ziua cu cea mai mare amplitudine a anului): u = ${u}%, d = ${d}%, c = ${c}%.`,
 gr2: (rg, half, k, c2, gk) => `Garman–Klass: 0,5 × ${rg}² = ${half}; (2 ln 2 − 1)c² = ${k} × ${c2}; σ̂²<sub>GK</sub> = ${gk} (%²).`,
 gr3: (u, uc, d, dc, a, b, rs) => `Rogers–Satchell: u(u − c) + d(d − c) = ${u} × ${uc} + (${d}) × (${dc}) = ${a} + ${b} = ${rs} (%²).`,
 gr4: (P, GK, RS) => `Aceeași zi: Parkinson ${P}, Garman–Klass ${GK}, Rogers–Satchell ${RS}.`,
 gr: (s, y, dr, a, b, c) => `${s}, ${y}: media lui c<sub>t</sub> (tendința din timpul zilei) ${dr}% pe zi; pe întregul an, volatilitatea anualizată Parkinson ${a}%, Garman–Klass ${b}%, Rogers–Satchell ${c}%. Liniile din grafic folosesc ferestre de 21 de zile.`,
 yz1: (n, a, b, r, den, k) => `n = ${n}: (n + 1)/(n − 1) = ${a}/${b} = ${r}; k = 0,34/(1,34 + ${r}) = 0,34/${den} = ${k}.`,
 yz2: (s, f, t, so, sc, rs) => `${s}, de la ${f} la ${t}: σ̂²<sub>O</sub> = ${so}, σ̂²<sub>C</sub> = ${sc}, media termenului Rogers–Satchell ${rs} (%² pe zi).`,
 yz3: (so, k, sc, k1, rs, yz) => `σ̂²<sub>YZ</sub> = ${so} + ${k} × ${sc} + ${k1} × ${rs} = ${yz}.`,
 yz4: (vyz, vrs, vcc, A) => `Anualizat (A = ${A}): Yang–Zhang ${vyz}%, doar ședința (Rogers–Satchell) ${vrs}%, închidere–închidere ${vcc}%.`,
 yz: (n, k, items) => `n = ${n}, k = ${k}; ferestrele se încheie pe 31 martie 2020. Ponderea nopții în σ̂²<sub>YZ</sub> și raportul √(YZ/CC): ` + items.map(([s, a, b]) => `${s} ${a}, ${b}`).join('; ') + '.',
 ms1: (dt, cp, o, h, l, c) => `${dt}: închiderea precedentă ${cp}; deschiderea ${o}, maximul ${h}, minimul ${l}, închiderea ${c}.`,
 ms2: (o, u, d, c, r) => `o = ${o}, u = ${u}, d = ${d}, c = ${c} (în %); închidere–închidere r = o + c = ${r}.`,
 ms3: (r2, rg, k, P, GK, RS) => `Estimări pentru o zi (%²): r² = ${r2}; Parkinson (${rg})²/${k} = ${P}; Garman–Klass ${GK}; Rogers–Satchell ${RS}.`,
 ms4: (o2, s, sh) => `Partea overnight o² = ${o2} (${sh} din r²); overnight plus Rogers–Satchell: ${s}.`,
 ms: (s, dt, r, o, r2, P, s2) => `${s}, ${dt}: r = ${r}%, din care overnight o = ${o}%; r² = ${r2}, Parkinson ${P}, o² + Rogers–Satchell ${s2}. Estimatorii care ignoră o<sub>t</sub> nu văd saltul de la deschidere.`,
 ef: (n, items, sc, x) => `Ferestre de ${zile(n)}${x ? ' (rulare suplimentară a simulării din curs)' : ''}: eficiența față de CC cu tranzacționare continuă: ` + items.map(([e, a]) => `${e} ${a}`).join(', ')
   + `; deplasarea varianței (%) ${sc === 'overnight' ? 'cu salt overnight' : 'cu 26 de prețuri pe zi'}: ` + items.map(([e, , b]) => `${e} ${b}`).join(', ') + '.',
 et: (n, a, b, p, y, m, x) => `Ferestre de ${zile(n)}${x ? ' (rulare suplimentară a simulării din curs)' : ''}: cu tranzacționare continuă, estimatorii de amplitudine sînt de ${a}–${b} ori mai eficienți decît CC. Cu salt overnight, Parkinson are deplasarea ${p}%, Yang–Zhang ${y}%, iar raportul MSE al Yang–Zhang este ${m}.`,
 es: (y, items) => `Eșantioane de la ${y} (sau de la începutul datelor OHLC); YZ/CC, ponderea nopții și corr(o, c): ` + items.map(([s, f, r, n, c]) => `${s} (din ${f}) ${r}, ${n}, ${c}`).join('; ') + '. Var(r) = Var(o) + Var(c) + 2Cov(o, c), iar estimatorii de amplitudine nu văd termenul de covarianță.',
 rr: (n, items) => `Fereastra de ${zile(n)}: media volatilității anualizate și variația zilnică tipică a logaritmului estimării, ambele din 2019: ` + items.map(([e, m, r]) => `${e} ${m}%, ${r}%`).join('; ') + '.',
 rd1: (day, M, m, f) => `Ziua ${day}: M = 390/${m} = ${M} ${de(M)}randamente pe intervale de ${minute(m)}; primele trei: ${f} (%).`,
 rd2: (sq, M, rv, sr) => `RV = ${sq} + … (${M} ${de(M)}termeni) = ${rv} (%²); √RV = ${sr}%.`,
 rd3: (sg, e, ar, er) => `Volatilitatea reală a zilei: ${sg}%; √RV se abate cu ${e}%, iar |r| al zilei (${ar}%) cu ${er}%.`,
 rd: (day, m, sr, sg, mae) => `Ziua ${day}, intervale de ${minute(m)}: √RV = ${sr}% față de volatilitatea reală ${sg}%; eroarea relativă absolută medie pe cele 60 de zile: ${mae}.`,
 re: (m, a, b, c, d, e, f) => `RV din randamente pe ${minute(m)}: eroarea relativă absolută medie ${a}, față de ${b} pentru Parkinson și ${c} pentru |r|; corelația cu volatilitatea reală ${d}, ${e} și ${f}. Simularea nu are zgomot de microstructură.`,
 sg: (z, c1, c5, t1, t5, x) => `Zgomot cu abaterea standard ${z}%${x ? ' (rulare suplimentară a codului din curs)' : ''}: RV/varianța reală = ${c1} la un minut și ${c5} la 5 minute; formula 1 + 2M × varianța zgomotului/varianța reală dă ${t1} și ${t5}.`,
 ac: (y, items) => `Eșantioane de la ${y}; din 100 de laguri, în afara benzii i.i.d.: ` + items.map(([s, f, T, a, q, t, r]) => `${s} (din ${f}, T = ${T}): |r| ${a}, r² ${q}; |r| mai autocorelat decît r² la ${t} dintre laguri; ρ̂₁₀₀(|r|) = ${r}`).join('; ') + '.',
 at: (lag, items) => `Lagul ${lag}: puterea d cu cea mai mare autocorelație a lui |r|<sup>d</sup> (în paranteză: valoarea ei, apoi ρ̂ pentru |r| și r²): ` + items.map(([s, d, v, a, q]) => `${s} d = ${d} (${v}; ${a}, ${q})`).join('; ') + '.',
 sh: (b, a0, a1, q0, q1, ex) => `${b === 1 ? 'Zile amestecate una cîte una' : `Blocuri de ${zile(b)} amestecate (rulare suplimentară)`}: ρ̂₁(|r|) = ${a1} față de ${a0} în ordinea reală; Ljung–Box Q(10) pentru r²: ${q1} față de ${q0}. Excesul de boltire rămîne ${ex}: aceleași randamente, în altă ordine.`,
 lq1: (s, T, m, r) => `${s}, T = ${T}, m = ${m}; ρ̂<sub>k</sub>(r²): ${r}.`,
 lq2: (T, T2, f, t) => `Factorul T(T + 2)/(T − k) pentru k = 1: ${T} × ${T2}/(T − 1) = ${f}; termenii T(T + 2)ρ̂<sub>k</sub>²/(T − k): ${t}.`,
 lq3: (m, q, c, p, rej) => `Q(${m}) = ${q} față de χ²<sub>0,95</sub>(${m}) = ${c} (p ${p}): ${rej ? 'respingem H<sub>0</sub>, randamentele la pătrat sînt autocorelate' : 'nu respingem H<sub>0</sub>'}.`,
 lq4: (m, q, c, p, rej) => `Același test pe randamente: Q(${m}) = ${q} (p ${p}), ${rej ? 'respinge și el, dar este mult mai mic' : 'nu respinge'}.`,
 lq: (s, m, q2, q1, c, rej) => `${s}, m = ${m}: Q pentru r² = ${q2}, pentru r = ${q1}; valoarea critică de 5% este ${c}. ${rej ? 'Testul pe r respinge și el, dar' : 'Testul pe r nu respinge;'} statistica pentru r² este mult mai mare.`,
 al1: (s, q, n, b) => `${s}, q = ${q}: regresia lui e<sub>t</sub>² pe o constantă și ${q === 1 ? 'un lag' : q + ' laguri'}, ${n} ${de(n)}observații; ĉ<sub>0</sub>, ĉ<sub>1</sub>, …: ${b}.`,
 al2: r2 => `R² = ${r2}: proporția din varianța lui e<sub>t</sub>² explicată de ultimele zile.`,
 al3: (n, r2, lm, q, c, p) => `LM = nR² = ${n} × ${r2} = ${lm}, față de χ²<sub>0,95</sub>(${q}) = ${c} (p ${p}).`,
 al: (s, q, lm, r2, n) => `${s}, q = ${q}: LM = ${lm}, R² = ${r2}, n = ${n}. LM crește proporțional cu n: o serie mai scurtă dă un LM mai mic la același R².`,
 ct: (m, c, list, pm, a, b, q, cq, nn) => `m = ${m}, valoarea critică ${c}: Q(${m}) pentru r respinge la 5% pentru ${list.length === 6 ? 'toate cele șase serii' : list.length ? list.join(', ') : 'nicio serie'} (cel mai mare p: ${pm}); Q(${m}) pentru r² este de ${a}–${b} ori mai mare decît pentru r. ARCH-LM(${q}), valoarea critică ${cq}: ${nn === 0 ? 'toate testele pe r², |r| și ARCH-LM resping' : nn + ' serii nu resping'}.`,
 ku: (s, e2, e4, k, pm, km, b, g, am) => `Amestec cu p = 0,2 și raportul ${s}: E[σ²] = ${e2}, E[σ⁴] = ${e4}, boltirea 3E[σ⁴]/E[σ²]² = ${k}; maximul pe p, ${km}, apare la p = ${pm}. GARCH(1,1) cu α = 0,10 și β = ${b}: ${g === null ? 'boltirea nu este finită' : 'boltirea ' + g}; cu acest β boltirea este finită pentru α &lt; ${am}.`,
 lo: (y, items, sR, n, phi, se, hl, lr, sk, skl, mn, md) => `De la ${y}: ` + items.map(([s, a, ay, b, by]) => `${s} de la ${a}% (${ay}) la ${b}% (${by})`).join('; ')
   + `. ${sR}, ${n} ${de(n)}luni: AR(1) al logaritmului volatilității lunare, φ̂ = ${phi} (SE ${se}), timpul de înjumătățire ${hl} luni, nivelul pe termen lung ${lr}%; coeficientul de asimetrie ${sk} pentru volatilitate și ${skl} pentru logaritmul ei; media ${mn}% față de mediana ${md}%.`,
 vx: (h, n, sh, mv, mr, g, mx, md, x) => `h = ${h}${x ? ' (rulare suplimentară)' : ''}: VIX depășește volatilitatea realizată din următoarele ${zile(h)} în ${sh} din cele ${n} ${de(n)}zile; media VIX ${mv}% față de ${mr}%, o diferență de ${g} puncte. Maximul VIX: ${mx}, pe ${md}.`,
 vi: (h, y, sh, a, b, r2, r2p, c, x) => `Din ${y}, h = ${h}${x ? ' (rulare suplimentară)' : ''}: VIX peste volatilitatea realizată în ${sh} din zile; regresia realizată = ${a} + ${b} VIX, R² = ${r2}; cu volatilitatea realizată din ultimele h zile, R² = ${r2p}. Corelația dintre variațiile zilnice ale ln VIX și randamentele S&P 500: ${c}.`,
 lv: (y, f, items) => `Din ${y}: corr(r<sub>t</sub>, ${f === 'sq' ? 'r<sub>t+1</sub>²' : '|r<sub>t+1</sub>|'}), corelația în sens invers corr(${f === 'sq' ? 'r<sub>t−1</sub>²' : '|r<sub>t−1</sub>|'}, r<sub>t</sub>) și raportul dintre media lui r<sub>t+1</sub>² după o zi de scădere și după o zi de creștere: `
   + items.map(([s, fr, a, b, c]) => `${s} (din ${fr}) ${a}, ${b}, ${c}`).join('; ') + '.',
 ni: (h, items, x) => `h = ${h}${x ? ' (rulare suplimentară; în curs 0,04)' : ''}: ` + items.map(([s, a, b, z, m, r]) => `${s} σ̂(−2%) = ${a}%, σ̂(+2%) = ${b}%, σ̂(0) = ${z}%, minimul la x ≈ ${m}%, raportul ${r}`).join('; ') + '.',
 fp: (y, l, items, a, b, c) => `${y}: vîrfurile prognozelor anualizate: ` + items.map(([m, v, d]) => `${m === 'hist252' ? 'fereastra de 252 de zile' : m === 'ew' ? `EWMA (λ = ${l})` : 'VIX'} ${v}% (${d})`).join('; ') + `. QLIKE mediu în ${y}: ${a}, ${b} și ${c}.`,
 ls1: s => `Varianța reală și variabila proxy egale cu ${s}; două prognoze, h = 0,5 și h = 2.`,
 ls2: (s, a, b) => `MSE: (${s} − 0,5)² = ${a}; (${s} − 2)² = ${b}.`,
 ls3: (s, a, l5, q5, b, l2, q2) => `QLIKE: ${s}/0,5 + ln 0,5 = ${a} + (${l5}) = ${q5}; ${s}/2 + ln 2 = ${b} + ${l2} = ${q2}.`,
 ls4: (pr, e, qe, h, qh) => `O zi reală, S&P 500 pe 16 martie 2020: proxy r² = ${pr}; prognoza EWMA (λ = 0,94) h = ${e}, QLIKE = ${qe}; prognoza pe 252 de zile h = ${h}, QLIKE = ${qh}.`,
 ls: (s, m05, q05) => `Proxy ${s}: MSE preferă h = ${m05 ? '0,5' : '2'}, QLIKE preferă h = ${q05 ? '0,5' : '2'}. QLIKE are minimul la h = proxy și crește mai repede cînd h coboară sub proxy.`,
 ft: (y, n, f, pr, best, t1, p1, t2, p2, x) => `Evaluare din ${f}${x ? ' (rulare suplimentară; în curs din 2010)' : ''}, ${n} ${de(n)}zile: cel mai mic QLIKE cu proxy ${pr === 'r2' ? 'r²' : 'de amplitudine'}: ${ML[best]}. Testul DM al QLIKE: EWMA 0,94 față de fereastra de 21 de zile t = ${t1} (p ${p1}); VIX față de EWMA 0,94 t = ${t2} (p ${p2}).`,
 la: (y, items, x) => `Evaluare din ${y}${x ? ' (rulare suplimentară)' : ''}; cel mai bun λ și creșterea QLIKE la λ = 0,94: ` + items.map(([s, f, b, q]) => `${s} (din ${f}) ${b}, ${q}`).join('; ') + '.'
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/,/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
const minute = m => (+m === 1 ? 'one minute' : `${m} minutes`);
const ML = { hist21: 'the 21-day window', hist63: 'the 63-day window', hist252: 'the 252-day window', ewma94: 'EWMA 0.94', ewma97: 'EWMA 0.97', vix: 'the VIX' };
window.CH8_TEXT = {
 h: { from: 'from', annvol: 'annual volatility (%)', maxmove: 'largest move (%)', date: 'date', insd: 'in standard deviations', exk: 'excess kurtosis', night: 'night share',
      eff_c: 'Efficiency, continuous trading', bias_c: 'Bias (%), continuous trading', bias_o: 'Bias (%), overnight jump', mse_o: 'MSE ratio, overnight jump',
      bias_d: 'Bias (%), 26 prices a day', outabs: '|r| outside the band', bestd: 'd with the largest ACF', days: 'days', mvix: 'mean VIX', mrv: 'mean realised', above: 'VIX above realised',
      gap: 'mean gap', r2past: 'R², past realised', mz_r2: 'MZ R², proxy r²', mz_rg: 'MZ R², range proxy', mse_rg: 'MSE, range proxy', ql_rg: 'QLIKE, range proxy',
      ml: { hist21: '21 days', hist63: '63 days', hist252: '252 days', ewma94: 'EWMA 0.94', ewma97: 'EWMA 0.97', vix: 'VIX' } },
 rb: (k, items) => `Days with |r| above ${k} standard deviations of the series (in brackets: the number expected if the returns followed the Normal distribution): ` + items.map(([s, n, e, m]) => `${s} ${n} (${e}), largest move: ${m} standard deviations`).join('; ') + '.',
 vt: (y, a, b, items) => `Samples from ${y} (or from the start of the series): ` + items.map(([s, v, e]) => `${s} ${v}% a year, excess kurtosis ${e}`).join('; ') + `. The largest daily move of each series is between ${a} and ${b} standard deviations.`,
 an: (cv, items) => (cv === null ? 'Each series annualised with its own frequency: ' + items.map(([s, v]) => `${s} ${v}%`).join('; ') + '.'
   : `With A = ${cv} for every series: ` + items.map(([s, v, w]) => `${s} ${w}% instead of ${v}%`).join('; ') + '. Only the series whose calendar is far from A change visibly.'),
 an1: (s, n, y, A, sd) => `${s}: ${n} returns in ${y} years, so A = ${A} observations a year; σ̂ = ${sd}% a day.`,
 an2: (sd, A, r, v) => `With the actual frequency: σ̂√A = ${sd} × √${A} = ${sd} × ${r} = ${v}%.`,
 an3: (A, sd, r, v, p) => `With A = ${A}: ${sd} × √${A} = ${sd} × ${r} = ${v}%, a deviation of ${p}% from the value with the actual frequency.`,
 rw: (items, w) => items.map(([m, mx, d, sc, l]) => `${m}-day window: maximum ${mx}% (${d}), typical daily change of ln σ̂ ${sc}%, last value ${l}%`).join('; ') + `. Whole-sample volatility: ${w}%.`,
 pr1: (s, k, n, i, p) => `${s}: kurtosis κ = ${k}; with Normal returns, SE(σ̂)/σ ≈ 1/√(2 × ${n}) = ${i}, that is ${p}%.`,
 pr2: (k, n, p) => `With the observed kurtosis: √((${k} − 1)/(4 × ${n})) = ${p}%.`,
 pr3: (n, v, a, b, c, d) => `Latest ${n}-day volatility: ${v}%; approximate 95% interval: [${a}%; ${b}%] with Normal returns, [${c}%; ${d}%] with the observed kurtosis.`,
 pr: (s, n, a, b, r) => `${s}, n = ${n}: relative standard error ${a}% (Normal returns) and ${b}% (observed kurtosis), ${r} times larger. The bands of the chart are σ̂(1 ± 1.96 SE/σ) on each day.`,
 ew: (l, h, lag, d, w, n, wn, d97) => `λ = ${l}: weight of yesterday ${w}, half-life ${h} days, mean lag 1/(1 − λ) = ${lag} days, 99% of the weight in the last ${d} days (λ = 0.97: ${d97} days). The ${n}-day window gives each day the weight 1/${n} = ${wn} and older days 0.`,
 eu1: (lab, s0, sg, r) => (lab === null ? `Yesterday's EWMA variance σ²(t − 1) = ${s0} (%²), a daily volatility of ${sg}%; yesterday's return r(t − 1) = ${r}%.`
   : `S&P 500, ${lab}: EWMA variance known at the previous close σ²(t − 1) = ${s0} (%²), a daily volatility of ${sg}%; the return of the day r = ${r}%.`),
 eu2: (l, s0, a) => `Step 1: λσ²(t − 1) = ${l} × ${s0} = ${a}.`,
 eu3: (w, r2, a) => `Step 2: (1 − λ)r² = ${w} × ${r2} = ${a}.`,
 eu4: (a, b, s, sg, an) => `Step 3: σ²(t) = ${a} + ${b} = ${s}, so σ(t) = ${sg}% a day, ${sg} × √252 = ${an}% a year.`,
 eu5: (l, s, ls, r2, w) => `If the next day has r = 0: σ²(t + 1) = ${l} × ${s} = ${ls}. A 63-day window would add ${r2}/63 = ${w} to the variance and drop this contribution all at once, 63 days later.`,
 eu: (l, r, e1, w1, eW, W) => `λ = ${l}, shock r = ${r}% after a variance of 1: the next day the EWMA volatility is ${e1}% and the ${W}-day window volatility ${w1}%; after ${W} days the EWMA is down to ${eW}%, and the window drops back to 1% at once.`,
 gh: (n, dd, b, a, ld, lr, l, ewd, ep, epd, rp, rpd) => `${n}-day window: the largest one-day fall between April and September 2020 is on ${dd}, from ${b}% to ${a}%, when the return of ${ld} (${lr}%) leaves the window; on that day the EWMA (λ = ${l}) moves by ${ewd} percentage points. Peaks of 2020: EWMA ${ep}% (${epd}), window ${rp}% (${rpd}).`,
 op: (seed, o, u, d, c, r, r2, rg, P, GK, RS) => `Seed ${seed}: o = ${o}%, u = ${u}%, d = ${d}%, c = ${c}%; return of the day r = o + c = ${r}%, r² = ${r2}; range u − d = ${rg}%, Parkinson ${P}, Garman–Klass ${GK}, Rogers–Satchell ${RS} (%²). The overnight gap changes r², not the range.`,
 pk1: (s, dt, H, L, rg) => `${s}, ${dt} (the day with the widest range of the year): H = ${H}, L = ${L}; u − d = 100 ln(H/L) = ${rg}%.`,
 pk2: (rg, r2, k, P, sp) => `σ̂²<sub>P</sub> = ${rg}²/(4 ln 2) = ${r2}/${k} = ${P} (%²); σ̂<sub>P</sub> = ${sp}% a day.`,
 pk3: (cc, c2, P) => `For comparison, the close-to-close return of the day: r = ${cc}%, r² = ${c2}, against the Parkinson estimate ${P}.`,
 pk: (s, y, n, a, b, d) => `${s}, ${y}, ${n} days: √(mean of the Parkinson estimates) = ${a}% a day, close-to-close standard deviation ${b}% a day; standard deviation of the daily difference between √Parkinson and |r|: ${d} percentage points.`,
 gr1: (s, dt, u, d, c) => `${s}, ${dt} (the day with the widest range of the year): u = ${u}%, d = ${d}%, c = ${c}%.`,
 gr2: (rg, half, k, c2, gk) => `Garman–Klass: 0.5 × ${rg}² = ${half}; (2 ln 2 − 1)c² = ${k} × ${c2}; σ̂²<sub>GK</sub> = ${gk} (%²).`,
 gr3: (u, uc, d, dc, a, b, rs) => `Rogers–Satchell: u(u − c) + d(d − c) = ${u} × ${uc} + (${d}) × (${dc}) = ${a} + ${b} = ${rs} (%²).`,
 gr4: (P, GK, RS) => `Same day: Parkinson ${P}, Garman–Klass ${GK}, Rogers–Satchell ${RS}.`,
 gr: (s, y, dr, a, b, c) => `${s}, ${y}: mean of c<sub>t</sub> (the drift within the day) ${dr}% a day; over the whole year, annualised volatility Parkinson ${a}%, Garman–Klass ${b}%, Rogers–Satchell ${c}%. The lines of the chart use 21-day windows.`,
 yz1: (n, a, b, r, den, k) => `n = ${n}: (n + 1)/(n − 1) = ${a}/${b} = ${r}; k = 0.34/(1.34 + ${r}) = 0.34/${den} = ${k}.`,
 yz2: (s, f, t, so, sc, rs) => `${s}, from ${f} to ${t}: σ̂²<sub>O</sub> = ${so}, σ̂²<sub>C</sub> = ${sc}, mean Rogers–Satchell term ${rs} (%² a day).`,
 yz3: (so, k, sc, k1, rs, yz) => `σ̂²<sub>YZ</sub> = ${so} + ${k} × ${sc} + ${k1} × ${rs} = ${yz}.`,
 yz4: (vyz, vrs, vcc, A) => `Annualised (A = ${A}): Yang–Zhang ${vyz}%, the session only (Rogers–Satchell) ${vrs}%, close-to-close ${vcc}%.`,
 yz: (n, k, items) => `n = ${n}, k = ${k}; the windows end on 31 March 2020. Night share of σ̂²<sub>YZ</sub> and the ratio √(YZ/CC): ` + items.map(([s, a, b]) => `${s} ${a}, ${b}`).join('; ') + '.',
 ms1: (dt, cp, o, h, l, c) => `${dt}: previous close ${cp}; open ${o}, high ${h}, low ${l}, close ${c}.`,
 ms2: (o, u, d, c, r) => `o = ${o}, u = ${u}, d = ${d}, c = ${c} (in %); close-to-close r = o + c = ${r}.`,
 ms3: (r2, rg, k, P, GK, RS) => `One-day estimates (%²): r² = ${r2}; Parkinson (${rg})²/${k} = ${P}; Garman–Klass ${GK}; Rogers–Satchell ${RS}.`,
 ms4: (o2, s, sh) => `Overnight part o² = ${o2} (${sh} of r²); overnight plus Rogers–Satchell: ${s}.`,
 ms: (s, dt, r, o, r2, P, s2) => `${s}, ${dt}: r = ${r}%, of which overnight o = ${o}%; r² = ${r2}, Parkinson ${P}, o² + Rogers–Satchell ${s2}. The estimators that ignore o<sub>t</sub> miss the jump at the open.`,
 ef: (n, items, sc, x) => `${n}-day windows${x ? ' (extra run of the lecture simulation)' : ''}: efficiency relative to CC with continuous trading: ` + items.map(([e, a]) => `${e} ${a}`).join(', ')
   + `; bias of the variance (%) ${sc === 'overnight' ? 'with an overnight jump' : 'with 26 prices a day'}: ` + items.map(([e, , b]) => `${e} ${b}`).join(', ') + '.',
 et: (n, a, b, p, y, m, x) => `${n}-day windows${x ? ' (extra run of the lecture simulation)' : ''}: with continuous trading the range-based estimators are ${a}–${b} times as efficient as CC. With an overnight jump Parkinson has a bias of ${p}%, Yang–Zhang ${y}%, and the MSE ratio of Yang–Zhang is ${m}.`,
 es: (y, items) => `Samples from ${y} (or from the start of the OHLC data); YZ/CC, night share and corr(o, c): ` + items.map(([s, f, r, n, c]) => `${s} (from ${f}) ${r}, ${n}, ${c}`).join('; ') + '. Var(r) = Var(o) + Var(c) + 2Cov(o, c), and the range-based estimators do not see the covariance term.',
 rr: (n, items) => `${n}-day window: mean annualised volatility and typical daily change of the log estimate, both since 2019: ` + items.map(([e, m, r]) => `${e} ${m}%, ${r}%`).join('; ') + '.',
 rd1: (day, M, m, f) => `Day ${day}: M = 390/${m} = ${M} returns over intervals of ${minute(m)}; the first three: ${f} (%).`,
 rd2: (sq, M, rv, sr) => `RV = ${sq} + … (${M} terms) = ${rv} (%²); √RV = ${sr}%.`,
 rd3: (sg, e, ar, er) => `True volatility of the day: ${sg}%; √RV deviates by ${e}%, the |r| of the day (${ar}%) by ${er}%.`,
 rd: (day, m, sr, sg, mae) => `Day ${day}, intervals of ${minute(m)}: √RV = ${sr}% against the true volatility ${sg}%; mean absolute relative error over the 60 days: ${mae}.`,
 re: (m, a, b, c, d, e, f) => `RV from returns over ${minute(m)}: mean absolute relative error ${a}, against ${b} for Parkinson and ${c} for |r|; correlation with the true volatility ${d}, ${e} and ${f}. The simulation has no microstructure noise.`,
 sg: (z, c1, c5, t1, t5, x) => `Noise with standard deviation ${z}%${x ? ' (extra run of the lecture code)' : ''}: RV/true variance = ${c1} at one minute and ${c5} at 5 minutes; the formula 1 + 2M × noise variance/true variance gives ${t1} and ${t5}.`,
 ac: (y, items) => `Samples from ${y}; of 100 lags, outside the i.i.d. band: ` + items.map(([s, f, T, a, q, t, r]) => `${s} (from ${f}, T = ${T}): |r| ${a}, r² ${q}; |r| more autocorrelated than r² at ${t} of the lags; ρ̂₁₀₀(|r|) = ${r}`).join('; ') + '.',
 at: (lag, items) => `Lag ${lag}: the power d with the largest autocorrelation of |r|<sup>d</sup> (in brackets: its value, then ρ̂ for |r| and r²): ` + items.map(([s, d, v, a, q]) => `${s} d = ${d} (${v}; ${a}, ${q})`).join('; ') + '.',
 sh: (b, a0, a1, q0, q1, ex) => `${b === 1 ? 'Days shuffled one by one' : `Blocks of ${b} days shuffled (extra run)`}: ρ̂₁(|r|) = ${a1} against ${a0} in the actual order; Ljung–Box Q(10) of r²: ${q1} against ${q0}. The excess kurtosis stays ${ex}: the same returns in another order.`,
 lq1: (s, T, m, r) => `${s}, T = ${T}, m = ${m}; ρ̂<sub>k</sub>(r²): ${r}.`,
 lq2: (T, T2, f, t) => `The factor T(T + 2)/(T − k) for k = 1: ${T} × ${T2}/(T − 1) = ${f}; the terms T(T + 2)ρ̂<sub>k</sub>²/(T − k): ${t}.`,
 lq3: (m, q, c, p, rej) => `Q(${m}) = ${q} against χ²<sub>0.95</sub>(${m}) = ${c} (p ${p}): ${rej ? 'we reject H<sub>0</sub>, the squared returns are autocorrelated' : 'we do not reject H<sub>0</sub>'}.`,
 lq4: (m, q, c, p, rej) => `The same test on returns: Q(${m}) = ${q} (p ${p}), ${rej ? 'it rejects too, but it is much smaller' : 'it does not reject'}.`,
 lq: (s, m, q2, q1, c, rej) => `${s}, m = ${m}: Q of r² = ${q2}, of r = ${q1}; the 5% critical value is ${c}. ${rej ? 'The test on r rejects too, but' : 'The test on r does not reject;'} the statistic of r² is much larger.`,
 al1: (s, q, n, b) => `${s}, q = ${q}: regression of e<sub>t</sub>² on a constant and ${q === 1 ? 'one lag' : q + ' lags'}, ${n} observations; ĉ<sub>0</sub>, ĉ<sub>1</sub>, …: ${b}.`,
 al2: r2 => `R² = ${r2}: the share of the variance of e<sub>t</sub>² explained by the last days.`,
 al3: (n, r2, lm, q, c, p) => `LM = nR² = ${n} × ${r2} = ${lm}, against χ²<sub>0.95</sub>(${q}) = ${c} (p ${p}).`,
 al: (s, q, lm, r2, n) => `${s}, q = ${q}: LM = ${lm}, R² = ${r2}, n = ${n}. LM grows in proportion to n: a shorter series gives a smaller LM at the same R².`,
 ct: (m, c, list, pm, a, b, q, cq, nn) => `m = ${m}, critical value ${c}: Q(${m}) of r rejects at 5% for ${list.length === 6 ? 'all six series' : list.length ? list.join(', ') : 'no series'} (largest p: ${pm}); Q(${m}) of r² is ${a}–${b} times that of r. ARCH-LM(${q}), critical value ${cq}: ${nn === 0 ? 'all tests on r², |r| and ARCH-LM reject' : nn + ' series do not reject'}.`,
 ku: (s, e2, e4, k, pm, km, b, g, am) => `Mixture with p = 0.2 and ratio ${s}: E[σ²] = ${e2}, E[σ⁴] = ${e4}, kurtosis 3E[σ⁴]/E[σ²]² = ${k}; its maximum over p, ${km}, is at p = ${pm}. GARCH(1,1) with α = 0.10 and β = ${b}: ${g === null ? 'the kurtosis is not finite' : 'kurtosis ' + g}; with this β the kurtosis is finite for α &lt; ${am}.`,
 lo: (y, items, sR, n, phi, se, hl, lr, sk, skl, mn, md) => `From ${y}: ` + items.map(([s, a, ay, b, by]) => `${s} from ${a}% (${ay}) to ${b}% (${by})`).join('; ')
   + `. ${sR}, ${n} months: AR(1) of the log monthly volatility, φ̂ = ${phi} (SE ${se}), half-life ${hl} months, long-run level ${lr}%; skewness ${sk} for the volatility and ${skl} for its logarithm; mean ${mn}% against median ${md}%.`,
 vx: (h, n, sh, mv, mr, g, mx, md, x) => `h = ${h}${x ? ' (extra run)' : ''}: the VIX exceeds the realised volatility of the next ${h} days on ${sh} of the ${n} days; mean VIX ${mv}% against ${mr}%, a gap of ${g} points. Maximum VIX: ${mx}, on ${md}.`,
 vi: (h, y, sh, a, b, r2, r2p, c, x) => `From ${y}, h = ${h}${x ? ' (extra run)' : ''}: VIX above the realised volatility on ${sh} of the days; regression realised = ${a} + ${b} VIX, R² = ${r2}; with the realised volatility of the last h days, R² = ${r2p}. Correlation of the daily changes of ln VIX with the S&P 500 returns: ${c}.`,
 lv: (y, f, items) => `From ${y}: corr(r<sub>t</sub>, ${f === 'sq' ? 'r<sub>t+1</sub>²' : '|r<sub>t+1</sub>|'}), the reverse correlation corr(${f === 'sq' ? 'r<sub>t−1</sub>²' : '|r<sub>t−1</sub>|'}, r<sub>t</sub>) and the ratio of the mean of r<sub>t+1</sub>² after a down day to that after an up day: `
   + items.map(([s, fr, a, b, c]) => `${s} (from ${fr}) ${a}, ${b}, ${c}`).join('; ') + '.',
 ni: (h, items, x) => `h = ${h}${x ? ' (extra run; 0.04 in the lecture)' : ''}: ` + items.map(([s, a, b, z, m, r]) => `${s} σ̂(−2%) = ${a}%, σ̂(+2%) = ${b}%, σ̂(0) = ${z}%, minimum at x ≈ ${m}%, ratio ${r}`).join('; ') + '.',
 fp: (y, l, items, a, b, c) => `${y}: peaks of the annualised forecasts: ` + items.map(([m, v, d]) => `${m === 'hist252' ? '252-day window' : m === 'ew' ? `EWMA (λ = ${l})` : 'VIX'} ${v}% (${d})`).join('; ') + `. Mean QLIKE in ${y}: ${a}, ${b} and ${c}.`,
 ls1: s => `True variance and proxy equal to ${s}; two forecasts, h = 0.5 and h = 2.`,
 ls2: (s, a, b) => `MSE: (${s} − 0.5)² = ${a}; (${s} − 2)² = ${b}.`,
 ls3: (s, a, l5, q5, b, l2, q2) => `QLIKE: ${s}/0.5 + ln 0.5 = ${a} + (${l5}) = ${q5}; ${s}/2 + ln 2 = ${b} + ${l2} = ${q2}.`,
 ls4: (pr, e, qe, h, qh) => `A real day, S&P 500 on 16 March 2020: proxy r² = ${pr}; EWMA forecast (λ = 0.94) h = ${e}, QLIKE = ${qe}; 252-day forecast h = ${h}, QLIKE = ${qh}.`,
 ls: (s, m05, q05) => `Proxy ${s}: MSE prefers h = ${m05 ? '0.5' : '2'}, QLIKE prefers h = ${q05 ? '0.5' : '2'}. QLIKE has its minimum at h = proxy and rises faster when h falls below the proxy.`,
 ft: (y, n, f, pr, best, t1, p1, t2, p2, x) => `Evaluation from ${f}${x ? ' (extra run; from 2010 in the lecture)' : ''}, ${n} days: the smallest QLIKE with the ${pr === 'r2' ? 'r²' : 'range'} proxy: ${ML[best]}. DM test of QLIKE: EWMA 0.94 against the 21-day window t = ${t1} (p ${p1}); VIX against EWMA 0.94 t = ${t2} (p ${p2}).`,
 la: (y, items, x) => `Evaluation from ${y}${x ? ' (extra run)' : ''}; best λ and the increase of QLIKE at λ = 0.94: ` + items.map(([s, f, b, q]) => `${s} (from ${f}) ${b}, ${q}`).join('; ') + '.'
};
})();
""",
}

# Question for the room and answer of every block (RO, EN)
QA = {
 'ro': {
  'returns-bursts': ('Abaterea standard zilnică a S&P 500 din 1990 este 1,13%. Cîte dintre cele 9 240 de zile au |r| peste 5 abateri standard, adică peste 5,67%?',
                     '30 de zile, față de 0,0053 așteptate dacă randamentele ar urma distribuția Normală. DAX are 20 de astfel de zile, BET 31, Bitcoin 8. Cea mai mare variație a S&P 500 (−12,8% pe 16 martie 2020) reprezintă 11,3 abateri standard; pentru cele patru piețe valorile sînt între 8,9 și 13,3.'),
  'vol-table': ('Cea mai mare variație zilnică a Bitcoin este −46,5%, a S&P 500 −12,8%. Măsurată în abaterile standard ale propriei serii, care este mai extremă?',
                'Bitcoin: 46,5/3,49 = 13,3 abateri standard, față de 12,8/1,13 = 11,3 la S&P 500, 9,5 la DAX și 8,9 la BET. Pentru distribuția Normală, probabilitatea unei variații de peste 9 abateri standard este de ordinul 10⁻¹⁹: excesul de boltire (10,9; 6,0; 9,9; 12,0) arată cozi groase pe toate cele patru piețe.'),
  'annual': ('Dacă folosim A = 252 pentru toate cele patru serii, cum se schimbă volatilitățile anualizate?',
             'Bitcoin scade de la 66,6% la 55,3% (−16,9%, factorul √(252/365,3) = 0,831); S&P 500 rămîne 18,0%, DAX 21,8%, iar BET trece de la 23,4% la 23,5%, pentru că frecvența lor reală (251,7; 253,0; 249,9 observații pe an) este aproape de 252. Cu A = 252, Bitcoin ar părea mai puțin riscant cu o șesime.'),
  'rolling-windows': ('În ce fereastră ați avea încredere în ziua de după un crah?',
                      'O fereastră scurtă sau EWMA: media pe o fereastră lungă include încă zilele liniștite dinaintea crahului. Fereastra de 21 de zile a urcat la 97,5% pe 6 aprilie 2020; cea de 252 de zile a atins maximul, 45,6%, abia pe 15 iulie 2009, la cîteva luni după vîrful crizei din 2008. Prețul reacției rapide este zgomotul: variația zilnică tipică a lui ln σ̂ este 6,7% la 21 de zile și 0,8% la 252 de zile.'),
  'precision': ('Coeficientul de boltire al S&P 500 este κ = 13,9. Care este eroarea standard relativă a unei volatilități calculate pe 21 de zile?',
                '39,1%, de 2,54 ori mai mult decît 15,4% din formula pentru randamente Normale (raportul este √((κ − 1)/2)). Pentru ultima volatilitate pe 21 de zile, 9,6%, intervalul aproximativ de 95% este [6,7%; 12,5%] cu randamente Normale și [2,2%; 17,0%] cu boltirea observată. Chiar și cu n = 252, eroarea relativă rămîne 11,3%; 4,5% este valoarea pentru n = 252 cu randamente Normale.'),
  'ewma-weights': ('Pentru λ = 0,94, 99% din pondere se află în ultimele 74,4 zile. Cîte zile sînt necesare pentru λ = 0,99?',
                   '458,2 zile, aproape doi ani de tranzacționare: ln 0,01/ln 0,99. Timpul de înjumătățire este 69,0 zile, iar lagul mediu 100 de zile. Pentru orice λ, numărul de zile care poartă 99% din pondere este de 6,64 ori timpul de înjumătățire (ln 0,01/ln 0,5).'),
  'ewma-update': ('Cu σ²(t − 1) = 1,00 și r(t − 1) = −3%, λ = 0,94 dă σ²(t) = 1,48. Cît este σ²(t) cu λ = 0,80?',
                  '2,60 = 0,80 × 1 + 0,20 × 9, deci σ(t) = 1,61% pe zi. Un λ mai mic dă o pondere mai mare (1 − λ = 0,20) șocului de ieri: reacția este mai puternică, iar memoria mai scurtă (timpul de înjumătățire scade de la 11,2 la 3,1 zile). Varianta 1,40 uită să ridice randamentul la pătrat.'),
  'ghost': ('De ce scade brusc linia albastră (fereastra de 63 de zile) la mijlocul lui iunie 2020, într-o zi liniștită?',
            'Pe 15 iunie 2020, randamentul din 16 martie 2020 (−12,8%) a ieșit din fereastra de 63 de zile: volatilitatea pe fereastră a scăzut de la 50,4% la 43,0% într-o singură zi, fără nicio știre. Cu o fereastră de 21 de zile, aceeași zi de crah iese pe 15 aprilie 2020 (de la 78,8% la 64,0%), cu 126 de zile pe 14 septembrie 2020. EWMA s-a modificat în ziua respectivă cu doar −0,9 puncte procentuale.'),
  'ohlc-path': ('Dacă saltul overnight crește de la 0,35% la 1%, ce se schimbă pentru această zi simulată?',
                'Randamentul zilei r = o + c crește de la 0,985% la 1,635%, iar r² de la 0,970 la 2,672. Maximul, minimul și închiderea sînt măsurate față de deschiderea zilei, deci Parkinson (0,627), Garman–Klass (0,714) și Rogers–Satchell (0,681) rămîn aceleași: estimatorii de amplitudine nu văd varianța nopții.'),
  'parkinson': ('S&P 500, 13 martie 2020: H = 2 711,33 și L = 2 492,37, deci ln(H/L) = 8,421%. Cît este volatilitatea zilnică Parkinson?',
                'σ̂²_P = 8,421²/2,7726 = 25,57, deci σ̂_P = 5,06% pe zi. Amplitudinea depășește σ în medie de √(4 ln 2) = 1,67 ori, iar împărțirea la 4 ln 2 corectează exact acest lucru. Randamentul închidere–închidere al zilei a fost 8,88%, cu r² = 78,87, de trei ori estimarea Parkinson. Varianta 8,42% este amplitudinea nescalată, iar 3,04% împarte amplitudinea la 4 ln 2 fără rădăcina pătrată.'),
  'gk-rs': ('Bitcoin, 19 mai 2021: u = 1,390%, d = −33,626%, c = −14,894%. Cît este termenul Rogers–Satchell u(u − c) + d(d − c)?',
            '652,5 (%²) = 1,390 × 16,284 + (−33,626) × (−18,732) = 22,6 + 629,9, deci σ̂_RS = 25,5% pe zi. Garman–Klass dă 527,4, Parkinson 442,2. Ziua a avut o tendință puternică (c = −14,9%), iar Rogers–Satchell, nedeplasat pentru orice tendință, dă cea mai mare estimare. Varianta 1 133 = u² + d² omite termenii cu c.'),
  'yz': ('Pentru n = 21, k = 0,34/(1,34 + 22/20) = 0,139. Ce se întîmplă cu k cînd fereastra crește la 126 de zile?',
         'k(126) = 0,34/(1,34 + 127/125) = 0,144, iar limita pentru n mare este 0,34/2,34 = 0,145; pentru n = 2, k = 0,078. Ponderea variază puțin cu n: Yang–Zhang este dominat de varianța nopții σ̂²_O și de termenul Rogers–Satchell, cu ponderea 1 − k ≈ 0,86.'),
  'march16': ('S&P 500 pe 16 martie 2020: r = −12,765%, din care overnight o = −7,760%. Ce parte din r² = 163,0 reprezintă o²?',
              'o² = 60,2, adică 37% din r². Restul nu se împarte simplu, pentru că r² = o² + c² + 2oc: aici c = −5,005%, c² = 25,0 și 2oc = 77,7. Estimatorii care ignoră noaptea (Parkinson 19,6, Garman–Klass 17,5, Rogers–Satchell 16,5) văd mai puțin de o optime din r²; overnight plus Rogers–Satchell dă 76,7. Varianta 61% compară randamentele, nu pătratele lor (7,760/12,765), iar 12% este raportul Parkinson/r².'),
  'efficiency': ('Cu ferestre de 21 de zile, Garman–Klass este de 9,3 ori mai eficient decît CC. Ce se întîmplă cu eficiența lui cînd fereastra are doar 5 zile?',
                 'Rămîne aproape aceeași: 10,3 la n = 5 și 8,0 la n = 63 (rulări suplimentare ale simulării). Ambele varianțe scad aproximativ ca 1/n, deci raportul lor se schimbă puțin: cîștigul de eficiență vine din informația dintr-o zi, nu din lungimea ferestrei. Cu salt overnight, deplasarea Parkinson, GK și RS rămîne în jur de −22% la orice n.'),
  'eff-table': ('Cu salt overnight, varianța estimării Parkinson este de 8,8 ori mai mică decît a CC, dar raportul MSE este doar 1,8. De ce?',
                'MSE = varianță + deplasare². Cu salt overnight, Parkinson, GK și RS măsoară doar ședința și subestimează varianța cu circa 22%, iar eroarea lor pătratică medie este dominată de această deplasare, deși varianța lor este de 8,8–13,6 ori mai mică decît a CC. Yang–Zhang adaugă σ̂²_O: deplasarea −2% și raportul MSE 9,0.'),
  'est-table': ('Pentru S&P 500, YZ (16,3%) rămîne sub CC (19,9%), deși include varianța nopții. Ce explică diferența?',
                'Var(r) = Var(o) + Var(c) + 2Cov(o, c), iar la S&P 500 corr(o, c) = 0,22; o explicație probabilă sînt prețurile stale (neactualizate) din deschiderea indicelui. La DAX și Bitcoin corelația este 0,02, iar raportul YZ/CC este 0,97 și 0,95. Ponderea nopții (13%) este deja inclusă în YZ, iar S&P 500 nu are zile cu H = L.'),
  'range-rolling': ('Cu fereastra de 21 de zile, variația zilnică tipică a logaritmului estimării este 7,0% pentru CC și 4,4% pentru Parkinson. Cum se schimbă aceste valori pentru o fereastră de 5 zile?',
                    'Ambele cresc mult: 30,3% pentru CC, 13,9% pentru Parkinson și 14,8% pentru YZ; la 63 de zile scad la 2,8% și 2,0%. Avantajul estimatorilor de amplitudine este mai mare pe ferestrele scurte: raportul CC/Parkinson este 2,2 la n = 5, 1,6 la n = 21 și 1,4 la n = 63.'),
  'rv-def': ('Ziua 1: √RV din randamente de 5 minute este 1,151%, iar volatilitatea reală 1,000%. Ce dă √RV din randamente de un minut?',
             '1,051% (390 de randamente, eroarea +5,1%). Simularea nu are zgomot de microstructură, deci RV converge către varianța integrată cînd M crește: pe cele 60 de zile, eroarea relativă absolută medie scade de la 5,9% (5 minute) la 2,9% (un minut). Cu zgomot, signature plot arată efectul opus la intervalele scurte.'),
  'realised': ('Eroarea relativă absolută medie este 6% pentru RV la 5 minute, 23% pentru Parkinson și 58% pentru |r|. Cît devine eroarea RV dacă folosim randamente pe 390 de minute, adică un singur randament pe zi?',
               '58%, ca |r|: cu M = 1, RV este chiar r², iar corelația cu volatilitatea reală scade la 0,59. La un minut eroarea este 3%, cu corelația 0,997. Parkinson (23%) folosește doar maximul și minimul și se află între cele două extreme.'),
  'signature': ('Cu zgomot de 0,025%, RV la un minut supraestimează varianța de 1,79 ori. De cîte ori o supraestimează dacă zgomotul se dublează, la 0,05%?',
                'De 4,16 ori în simulare și de 4,15 ori după formula 1 + 2M × varianța zgomotului/varianța reală, cu M = 390: excesul de 0,79 crește cu pătratul zgomotului, la circa 4 × 0,79 = 3,16. La 5 minute (M = 78) raportul este 1,62. Fiecare randament observat la pătrat conține, în medie, de două ori varianța zgomotului.'),
  'acf-clustering': ('Pe eșantionul din curs, toate cele 100 de autocorelații ale lui |r| ale S&P 500 sînt în afara benzii. Cîte rămîn în afara benzii dacă eșantionul începe în 2010?',
                     '72 pentru |r| și 40 pentru r². Banda ±1,96/√T se lărgește de la ±0,020 la ±0,030 cînd T scade de la 9 240 la 4 202, iar ρ̂₁₀₀(|r|) scade de la 0,09 la 0,02. La lagurile mici volatility clustering rămîne evident; pentru BET din 2010, 53 de laguri ale lui |r| sînt în afara benzii.'),
  'acf-table': ('La lagul 1, pentru ce putere d este cea mai mare autocorelația lui |r|<sup>d</sup> a BET?',
                'd = 1,00 (0,414): |r| este mai autocorelat decît r² (0,333), efectul Taylor. Optimul depinde însă de serie și de lag: la lagul 1, d = 1,75 pentru S&P 500 și 0,75 pentru Bitcoin; la lagul 10, 1,25, 0,75 și 0,50. Puterile mari dau o pondere mare cîtorva zile extreme.'),
  'shuffle': ('Amestecarea zilelor una cîte una duce ρ̂₁(|r|) de la 0,27 la −0,02. Ce se întîmplă dacă amestecăm blocuri de cîte 5 zile consecutive?',
              'ρ̂₁(|r|) rămîne 0,23, pentru că patru din cinci perechi de zile consecutive rămîn în același bloc, dar Q(10) pentru r² scade de la 8 015 la 1 413: lagurile mai lungi decît blocul își pierd structura. Cu blocuri de 63 de zile, ρ̂₁(|r|) = 0,27 și Q(10) = 7 038. Excesul de boltire rămîne 10,9 în toate cazurile.'),
  'lb-sq': ('S&P 500, T = 9 240: ρ̂₁, ρ̂₂, ρ̂₃ ale lui r² sînt 0,311; 0,408; 0,250. Cît este Q(3)?',
            'Termenii T(T + 2)ρ̂²/(T − k) sînt 892; 1 540; 579, deci Q(3) = 3 012, mult peste χ²<sub>0,95</sub>(3) = 7,81. Q(3) pentru randamente este 57,7: respinge și el, dar este de 52 de ori mai mic. Varianta 8 958 uită să ridice autocorelațiile la pătrat, iar 0,33 uită factorul T(T + 2)/(T − k).'),
  'arch-lm': ('S&P 500, q = 5: R² = 0,2412 și n = 9 235. Cît este statistica LM?',
              'LM = 9 235 × 0,2412 = 2 227, față de χ²<sub>0,95</sub>(5) = 11,07 (p < 0,001). R² spune că circa un sfert din variația lui e²ₜ de la o zi la alta este anticipată de ultimele cinci zile. Varianta 4 535 folosește R în loc de R², iar 1,21 înmulțește R² cu q în loc de n.'),
  'cl-table': ('Bitcoin are excesul de boltire 12,0, dar cea mai mică statistică ARCH-LM(5) din tabel: 115, față de 2 227 la S&P 500. De ce?',
               'LM = nR²: Bitcoin are mai puțin de jumătate din observațiile S&P 500 (4 379 față de 9 235 în regresie), iar R² este 0,026 față de 0,241, pentru că cîteva zile extreme domină regresia pătratelor. Testul respinge totuși puternic (p < 0,001), iar Q(10) pentru |r|, mai puțin sensibil la zilele extreme, este 1 086. Q(10) pentru r respinge și el în toate cele șase serii (p ≤ 0,028), dar este de 10,5–171,7 ori mai mic decît pentru r².'),
  'kurtosis': ('Zilele agitate au abaterea standard de 3 ori mai mare decît cele liniștite. Pentru ce pondere p a zilelor agitate este boltirea amestecului cea mai mare?',
               'La p = 0,1, cu boltirea 8,33; la p = 0,5 boltirea este 4,92, iar la p = 0,02, 5,80. Cînd p tinde spre 0 sau spre 1, amestecul devine o singură distribuție Normală, cu boltirea 3. Pentru raportul 2 și p = 0,2, ca pe slide, boltirea este 4,69.'),
  'long-term': ('Din 1990, AR(1) al logaritmului volatilității lunare a S&P 500 are φ̂ = 0,67 și timpul de înjumătățire 1,7 luni. Ce dă același AR(1) dacă eșantionul începe în 2010?',
                'φ̂ = 0,53 (SE 0,06) pe 201 luni, timpul de înjumătățire 1,1 luni și nivelul pe termen lung 12,9%; din 2000, φ̂ = 0,66. Eșantionul din 2010 nu conține criza lungă din 2008–2009, iar episodul din 2020 a fost scurt, deci revenirea la medie pare mai rapidă.'),
  'vix': ('Linia roșie (VIX) este de obicei deasupra sau dedesubtul celei albastre (volatilitatea realizată din următoarele 21 de zile)?',
          'Deasupra, în 86% din cele 9 199 de zile; media VIX este 19,4%, media volatilității realizate 15,3%, o diferență de 4,1 puncte. Este prima de risc a varianței: investitorii plătesc pentru a fi asigurați împotriva salturilor de volatilitate. Cu h = 63 de zile, VIX este deasupra în 82% din zile.'),
  'vix-table': ('Din 1990, VIX explică R² = 0,52 din volatilitatea realizată a următoarelor 21 de zile, față de 0,44 pentru volatilitatea realizată trecută. Ce se întîmplă dacă eșantionul începe în 2010?',
                'Ambele R² scad, la 0,37 și 0,26, iar VIX rămîne mai informativ; regresia devine realizată = −0,16 + 0,81 VIX. După 2010 sînt mai puține episoade lungi de volatilitate ridicată, deci mai puțină variație de explicat. Panta sub 1 arată că VIX este informativ, dar deplasat ca prognoză.'),
  'leverage': ('Pe întregul eșantion, corr(r_t, |r_t+1|) este −0,122 la S&P 500 și −0,073 la BET. Ce se întîmplă cu BET dacă eșantionul începe în 2010?',
               'Efectul devine mai puternic: corelația este −0,141, iar raportul dintre media lui r²ₜ₊₁ după o zi de scădere și după o zi de creștere urcă de la 1,41 la 1,86. La S&P 500 corelația rămîne −0,122. Bitcoin are cea mai slabă asimetrie (raportul 1,06).'),
  'news-impact': ('De ce minimul volatilității condiționate a S&P 500 este la x ≈ 0,65%, nu la 0?',
                  'Din cauza asimetriei: după o zi de creștere moderată urmează, în medie, ziua cea mai liniștită, iar după o scădere de aceeași mărime una mai agitată; σ̂(−2%) = 1,61% față de σ̂(+2%) = 1,16% (raportul 1,39), la DAX 1,79% față de 1,43%. Media randamentelor, 0,03% pe zi, este prea mică pentru a explica deplasarea. Cu h = 0,01 minimul se mută la 0,17%, cu h = 0,10 la 1,61%, dar σ̂(−2%) > σ̂(+2%) rămîne.'),
  'loss': ('Varianța reală și variabila proxy sînt egale cu 1, iar prognozele sînt h = 0,5 și h = 2. Ce prognoză preferă fiecare funcție de pierdere?',
           'MSE preferă h = 0,5 (0,25 față de 1), QLIKE preferă h = 2 (1,193 față de 1,307): MSE penalizează mai mult supraestimarea, QLIKE subestimarea, eroarea costisitoare în managementul riscului. QLIKE are minimul la h = proxy. MSE devine indiferent între cele două prognoze la proxy = 1,25, iar QLIKE la proxy = 2 ln 2/1,5 = 0,924.'),
  'fc-table': ('Cu proxy r², din 2010, ce prognoză a varianței pentru ziua următoare are cel mai mic QLIKE?',
               'VIX (0,799), urmat de EWMA 0,94 (0,808), dar diferența nu este semnificativă: testul DM dă t = −0,32 (p = 0,751). EWMA 0,94 este semnificativ mai bună decît fereastra de 21 de zile (t = −4,50, p < 0,001), iar fereastra de 252 de zile este cea mai slabă (1,099). Cu proxy de amplitudine, VIX coboară pe locul cinci (0,635): acest proxy este în medie mai mic decît r² (0,82 față de 1,18), pentru că nu vede Cov(o, c), iar VIX, care depășește deja volatilitatea realizată, pierde cel mai mult.'),
  'forecast-paths': ('În 2020, VIX și EWMA au atins vîrful în martie, iar fereastra de 252 de zile pe 29 decembrie. Cum arată vîrfurile în 2008?',
                     'EWMA 79,0% pe 29 octombrie, VIX 80,9% pe 21 noiembrie, fereastra de 252 de zile 41,1% pe 31 decembrie. Fereastra lungă cumulează șocurile și scade doar cînd acestea ies din ea, deci vîrful ei vine tîrziu și este mai mic. QLIKE mediu în 2008: 3,132 pentru fereastra de 252 de zile, 2,420 pentru EWMA, 2,350 pentru VIX.'),
  'lambda': ('Cu evaluarea din 2010, cel mai bun λ pentru S&P 500 este 0,925. Ce λ alege QLIKE dacă evaluarea începe în 2022?',
             '0,940, valoarea RiskMetrics (diferența de QLIKE la 0,94 este sub 0,001); din 2020, 0,920; din 2016, 0,930. Pentru BET cel mai bun λ este între 0,885 și 0,900, pentru Bitcoin între 0,970 și 0,995. Un singur λ pentru toate activele și perioadele este un compromis comod, nu un optim.'),
 },
 'en': {
  'returns-bursts': ('The daily standard deviation of the S&P 500 since 1990 is 1.13%. How many of the 9,240 days have |r| above 5 standard deviations, that is above 5.67%?',
                     '30 days, against 0.0053 expected if the returns followed the Normal distribution. The DAX has 20 such days, the BET 31, Bitcoin 8. The largest S&P 500 move (−12.8% on 16 March 2020) is 11.3 standard deviations; for the four markets the values are between 8.9 and 13.3.'),
  'vol-table': ("Bitcoin's largest daily move is −46.5%, the S&P 500's −12.8%. Measured in each series' own standard deviations, which is more extreme?",
                'Bitcoin: 46.5/3.49 = 13.3 standard deviations, against 12.8/1.13 = 11.3 for the S&P 500, 9.5 for the DAX and 8.9 for the BET. Under the Normal distribution, the probability of a move beyond 9 standard deviations is of the order of 10⁻¹⁹: the excess kurtosis (10.9; 6.0; 9.9; 12.0) shows heavy tails in all four markets.'),
  'annual': ('If we use A = 252 for all four series, how do the annualised volatilities change?',
             'Bitcoin falls from 66.6% to 55.3% (−16.9%, the factor √(252/365.3) = 0.831); the S&P 500 stays at 18.0%, the DAX at 21.8%, and the BET moves from 23.4% to 23.5%, because their actual frequencies (251.7; 253.0; 249.9 observations a year) are close to 252. With A = 252, Bitcoin would look one sixth less risky.'),
  'rolling-windows': ('Which window would you trust on the day after a crash?',
                      'A short window or EWMA: the mean over a long window still contains the calm days before the crash. The 21-day window rose to 97.5% on 6 April 2020; the 252-day window reached its maximum, 45.6%, only on 15 July 2009, months after the peak of the 2008 crisis. The price of the fast reaction is noise: the typical daily change of ln σ̂ is 6.7% with 21 days and 0.8% with 252 days.'),
  'precision': ('The kurtosis of the S&P 500 is κ = 13.9. What is the relative standard error of a volatility computed from 21 days?',
                '39.1%, 2.54 times the 15.4% of the formula for Normal returns (the ratio is √((κ − 1)/2)). For the latest 21-day volatility, 9.6%, the approximate 95% interval is [6.7%; 12.5%] with Normal returns and [2.2%; 17.0%] with the observed kurtosis. Even with n = 252 the relative error is still 11.3%; 4.5% is the value for n = 252 with Normal returns.'),
  'ewma-weights': ('For λ = 0.94, 99% of the weight lies in the last 74.4 days. How many days are needed for λ = 0.99?',
                   '458.2 days, almost two years of trading: ln 0.01/ln 0.99. The half-life is 69.0 days and the mean lag 100 days. For any λ, the number of days that carry 99% of the weight is 6.64 times the half-life (ln 0.01/ln 0.5).'),
  'ewma-update': ('With σ²(t − 1) = 1.00 and r(t − 1) = −3%, λ = 0.94 gives σ²(t) = 1.48. What is σ²(t) with λ = 0.80?',
                  '2.60 = 0.80 × 1 + 0.20 × 9, so σ(t) = 1.61% a day. A smaller λ gives a larger weight (1 − λ = 0.20) to yesterday\'s shock: the reaction is stronger and the memory shorter (the half-life falls from 11.2 to 3.1 days). The option 1.40 forgets to square the return.'),
  'ghost': ('Why does the blue line (63-day window) drop abruptly in mid-June 2020, on a calm day?',
            'On 15 June 2020 the return of 16 March 2020 (−12.8%) left the 63-day window: the window volatility fell from 50.4% to 43.0% in a single day, without any news. With a 21-day window the same crash day leaves on 15 April 2020 (from 78.8% to 64.0%), with 126 days on 14 September 2020. The EWMA moved by only −0.9 percentage points that day.'),
  'ohlc-path': ('If the overnight gap rises from 0.35% to 1%, what changes for this simulated day?',
                'The return of the day r = o + c rises from 0.985% to 1.635% and r² from 0.970 to 2.672. The high, the low and the close are measured from the open of the day, so Parkinson (0.627), Garman–Klass (0.714) and Rogers–Satchell (0.681) stay the same: the range-based estimators do not see the variance of the night.'),
  'parkinson': ('S&P 500, 13 March 2020: H = 2,711.33 and L = 2,492.37, so ln(H/L) = 8.421%. What is the daily Parkinson volatility?',
                'σ̂²_P = 8.421²/2.7726 = 25.57, so σ̂_P = 5.06% a day. The range exceeds σ on average by a factor √(4 ln 2) = 1.67, and dividing by 4 ln 2 corrects exactly for this. The close-to-close return of the day was 8.88%, with r² = 78.87, three times the Parkinson estimate. The option 8.42% is the unscaled range, and 3.04% divides the range by 4 ln 2 without the square root.'),
  'gk-rs': ('Bitcoin, 19 May 2021: u = 1.390%, d = −33.626%, c = −14.894%. What is the Rogers–Satchell term u(u − c) + d(d − c)?',
            '652.5 (%²) = 1.390 × 16.284 + (−33.626) × (−18.732) = 22.6 + 629.9, so σ̂_RS = 25.5% a day. Garman–Klass gives 527.4, Parkinson 442.2. The day had a strong drift (c = −14.9%), and Rogers–Satchell, unbiased for any drift, gives the largest estimate. The option 1,133 = u² + d² leaves out the terms in c.'),
  'yz': ('For n = 21, k = 0.34/(1.34 + 22/20) = 0.139. What happens to k when the window grows to 126 days?',
         'k(126) = 0.34/(1.34 + 127/125) = 0.144, and the limit for large n is 0.34/2.34 = 0.145; for n = 2, k = 0.078. The weight varies little with n: Yang–Zhang is dominated by the night variance σ̂²_O and by the Rogers–Satchell term, with the weight 1 − k ≈ 0.86.'),
  'march16': ('S&P 500 on 16 March 2020: r = −12.765%, of which overnight o = −7.760%. What share of r² = 163.0 is o²?',
              'o² = 60.2, that is 37% of r². The rest does not split simply, because r² = o² + c² + 2oc: here c = −5.005%, c² = 25.0 and 2oc = 77.7. The estimators that ignore the night (Parkinson 19.6, Garman–Klass 17.5, Rogers–Satchell 16.5) see less than one eighth of r²; overnight plus Rogers–Satchell gives 76.7. The option 61% compares the returns, not their squares (7.760/12.765), and 12% is the ratio Parkinson/r².'),
  'efficiency': ('With 21-day windows, Garman–Klass is 9.3 times as efficient as CC. What happens to its efficiency when the window has only 5 days?',
                 'It stays about the same: 10.3 at n = 5 and 8.0 at n = 63 (extra runs of the simulation). Both variances fall roughly as 1/n, so their ratio changes little: the efficiency gain comes from the information within a day, not from the length of the window. With an overnight jump, the bias of Parkinson, GK and RS stays around −22% for every n.'),
  'eff-table': ('With an overnight jump, the variance of the Parkinson estimate is 8.8 times smaller than that of CC, but its MSE ratio is only 1.8. Why?',
                'MSE = variance + bias². With an overnight jump, Parkinson, GK and RS measure only the session and underestimate the variance by about 22%; their mean squared error is dominated by this bias, although their variance is 8.8–13.6 times smaller than that of CC. Yang–Zhang adds σ̂²_O: bias −2% and MSE ratio 9.0.'),
  'est-table': ('For the S&P 500, YZ (16.3%) stays below CC (19.9%), although it includes the night variance. What explains the gap?',
                'Var(r) = Var(o) + Var(c) + 2Cov(o, c), and for the S&P 500 corr(o, c) = 0.22; a likely explanation is the stale prices in the opening value of the index. For the DAX and Bitcoin the correlation is 0.02, and the ratio YZ/CC is 0.97 and 0.95. The night share (13%) is already in YZ, and the S&P 500 has no days with H = L.'),
  'range-rolling': ('With the 21-day window, the typical daily change of the log estimate is 7.0% for CC and 4.4% for Parkinson. How do these values change for a 5-day window?',
                    'Both rise sharply: 30.3% for CC, 13.9% for Parkinson and 14.8% for YZ; with 63 days they fall to 2.8% and 2.0%. The advantage of the range-based estimators is larger for short windows: the ratio CC/Parkinson is 2.2 at n = 5, 1.6 at n = 21 and 1.4 at n = 63.'),
  'rv-def': ('Day 1: √RV from 5-minute returns is 1.151%, and the true volatility 1.000%. What does √RV from one-minute returns give?',
             '1.051% (390 returns, an error of +5.1%). The simulation has no microstructure noise, so RV converges to the integrated variance as M grows: over the 60 days the mean absolute relative error falls from 5.9% (5 minutes) to 2.9% (one minute). With noise, the signature plot shows the opposite effect at short intervals.'),
  'realised': ('The mean absolute relative error is 6% for 5-minute RV, 23% for Parkinson and 58% for |r|. What is the error of RV if we use 390-minute returns, a single return a day?',
               '58%, as for |r|: with M = 1, RV is just r², and the correlation with the true volatility falls to 0.59. At one minute the error is 3%, with a correlation of 0.997. Parkinson (23%) uses only the high and the low and lies between the two extremes.'),
  'signature': ('With noise of 0.025%, one-minute RV overstates the variance 1.79 times. How many times does it overstate it if the noise doubles, to 0.05%?',
                '4.16 times in the simulation and 4.15 times by the formula 1 + 2M × noise variance/true variance, with M = 390: the excess of 0.79 grows with the square of the noise, to about 4 × 0.79 = 3.16. At 5 minutes (M = 78) the ratio is 1.62. Each squared observed return contains, on average, twice the noise variance.'),
  'acf-clustering': ('In the lecture sample, all 100 autocorrelations of |r| of the S&P 500 are outside the band. How many stay outside if the sample starts in 2010?',
                     '72 for |r| and 40 for r². The band ±1.96/√T widens from ±0.020 to ±0.030 as T falls from 9,240 to 4,202, and ρ̂₁₀₀(|r|) falls from 0.09 to 0.02. At short lags volatility clustering remains evident; for the BET from 2010, 53 lags of |r| are outside the band.'),
  'acf-table': ('At lag 1, for which power d is the autocorrelation of |r|<sup>d</sup> of the BET largest?',
                'd = 1.00 (0.414): |r| is more autocorrelated than r² (0.333), the Taylor effect. The optimum depends on the series and the lag, though: at lag 1, d = 1.75 for the S&P 500 and 0.75 for Bitcoin; at lag 10, 1.25, 0.75 and 0.50. Large powers give a large weight to a few extreme days.'),
  'shuffle': ('Shuffling the days one by one takes ρ̂₁(|r|) from 0.27 to −0.02. What happens if we shuffle blocks of 5 consecutive days?',
              'ρ̂₁(|r|) stays at 0.23, because four of five pairs of consecutive days remain in the same block, but Q(10) of r² falls from 8,015 to 1,413: the lags longer than the block lose their structure. With 63-day blocks, ρ̂₁(|r|) = 0.27 and Q(10) = 7,038. The excess kurtosis stays 10.9 in all cases.'),
  'lb-sq': ('S&P 500, T = 9,240: ρ̂₁, ρ̂₂, ρ̂₃ of r² are 0.311; 0.408; 0.250. What is Q(3)?',
            'The terms T(T + 2)ρ̂²/(T − k) are 892; 1,540; 579, so Q(3) = 3,012, far above χ²<sub>0.95</sub>(3) = 7.81. Q(3) of the returns is 57.7: it rejects too, but it is 52 times smaller. The option 8,958 forgets to square the autocorrelations, and 0.33 forgets the factor T(T + 2)/(T − k).'),
  'arch-lm': ('S&P 500, q = 5: R² = 0.2412 and n = 9,235. What is the LM statistic?',
              'LM = 9,235 × 0.2412 = 2,227, against χ²<sub>0.95</sub>(5) = 11.07 (p < 0.001). R² says that about a quarter of the day-to-day variation of e²ₜ is anticipated by the last five days. The option 4,535 uses R instead of R², and 1.21 multiplies R² by q instead of n.'),
  'cl-table': ('Bitcoin has an excess kurtosis of 12.0, but the smallest ARCH-LM(5) statistic of the table: 115, against 2,227 for the S&P 500. Why?',
               'LM = nR²: Bitcoin has less than half the observations of the S&P 500 (4,379 against 9,235 in the regression), and its R² is 0.026 against 0.241, because a few extreme days dominate the regression of squares. The test still rejects strongly (p < 0.001), and Q(10) of |r|, less sensitive to extreme days, is 1,086. Q(10) of r also rejects in all six series (p ≤ 0.028), but it is 10.5–171.7 times smaller than that of r².'),
  'kurtosis': ('Turbulent days have a standard deviation 3 times that of calm days. For which share p of turbulent days is the kurtosis of the mixture largest?',
               'At p = 0.1, with kurtosis 8.33; at p = 0.5 the kurtosis is 4.92, at p = 0.02, 5.80. As p tends to 0 or 1 the mixture becomes a single Normal distribution, with kurtosis 3. For the ratio 2 and p = 0.2, as on the slide, the kurtosis is 4.69.'),
  'long-term': ('Since 1990, the AR(1) of the log monthly volatility of the S&P 500 has φ̂ = 0.67 and a half-life of 1.7 months. What does the same AR(1) give if the sample starts in 2010?',
                'φ̂ = 0.53 (SE 0.06) on 201 months, a half-life of 1.1 months and a long-run level of 12.9%; from 2000, φ̂ = 0.66. The sample from 2010 does not contain the long 2008–2009 crisis, and the 2020 episode was short, so mean reversion looks faster.'),
  'vix': ('Is the red line (VIX) usually above or below the blue one (the realised volatility of the next 21 days)?',
          'Above, on 86% of the 9,199 days; the mean VIX is 19.4%, the mean realised volatility 15.3%, a gap of 4.1 points. This is the variance risk premium: investors pay to be insured against volatility jumps. With h = 63 days, the VIX is above on 82% of the days.'),
  'vix-table': ('Since 1990 the VIX explains R² = 0.52 of the realised volatility of the next 21 days, against 0.44 for the past realised volatility. What happens if the sample starts in 2010?',
                'Both R² fall, to 0.37 and 0.26, and the VIX stays more informative; the regression becomes realised = −0.16 + 0.81 VIX. After 2010 there are fewer long episodes of high volatility, so less variation to explain. The slope below 1 shows that the VIX is informative but biased as a forecast.'),
  'leverage': ('Over the whole sample, corr(r_t, |r_t+1|) is −0.122 for the S&P 500 and −0.073 for the BET. What happens for the BET if the sample starts in 2010?',
               'The effect becomes stronger: the correlation is −0.141, and the ratio of the mean of r²ₜ₊₁ after a down day to that after an up day rises from 1.41 to 1.86. For the S&P 500 the correlation stays at −0.122. Bitcoin has the weakest asymmetry (ratio 1.06).'),
  'news-impact': ('Why is the minimum of the conditional volatility of the S&P 500 at x ≈ 0.65%, not at 0?',
                  'Because of the asymmetry: a moderate up day is followed, on average, by the calmest day, and a fall of the same size by a more turbulent one; σ̂(−2%) = 1.61% against σ̂(+2%) = 1.16% (ratio 1.39), for the DAX 1.79% against 1.43%. The mean return, 0.03% a day, is too small to explain the shift. With h = 0.01 the minimum moves to 0.17%, with h = 0.10 to 1.61%, but σ̂(−2%) > σ̂(+2%) remains.'),
  'loss': ('The true variance and the proxy are both 1, and the forecasts are h = 0.5 and h = 2. Which forecast does each loss prefer?',
           'MSE prefers h = 0.5 (0.25 against 1), QLIKE prefers h = 2 (1.193 against 1.307): MSE penalises overestimation more, QLIKE underestimation, the costly error in risk management. QLIKE has its minimum at h = proxy. MSE becomes indifferent between the two forecasts at proxy = 1.25, QLIKE at proxy = 2 ln 2/1.5 = 0.924.'),
  'fc-table': ('With the r² proxy, from 2010, which next-day variance forecast has the smallest QLIKE?',
               'The VIX (0.799), followed by EWMA 0.94 (0.808), but the difference is not significant: the DM test gives t = −0.32 (p = 0.751). EWMA 0.94 is significantly better than the 21-day window (t = −4.50, p < 0.001), and the 252-day window is the weakest (1.099). With the range proxy the VIX drops to fifth place (0.635): this proxy is on average smaller than r² (0.82 against 1.18), because it does not see Cov(o, c), and the VIX, which already exceeds the realised volatility, loses most.'),
  'forecast-paths': ('In 2020 the VIX and the EWMA peaked in March, the 252-day window on 29 December. What do the peaks look like in 2008?',
                     'EWMA 79.0% on 29 October, VIX 80.9% on 21 November, the 252-day window 41.1% on 31 December. The long window accumulates the shocks and falls only when they leave it, so its peak comes late and is lower. Mean QLIKE in 2008: 3.132 for the 252-day window, 2.420 for the EWMA, 2.350 for the VIX.'),
  'lambda': ('With the evaluation from 2010, the best λ for the S&P 500 is 0.925. Which λ does QLIKE choose if the evaluation starts in 2022?',
             '0.940, the RiskMetrics value (the QLIKE difference at 0.94 is below 0.001); from 2020, 0.920; from 2016, 0.930. For the BET the best λ is between 0.885 and 0.900, for Bitcoin between 0.970 and 0.995. A single λ for all assets and periods is a convenient compromise, not an optimum.'),
 },
}

# Predict, then check: (key, [(RO, EN), ...]); the keys agree with what the page computes (node script on ch8_data.js + common.js + ch8.js).
PREDICT = {
 'returns-bursts': (1, [('Niciuna: pentru distribuția Normală, o astfel de zi apare o dată la sute de ani de date.', 'None: under the Normal distribution such a day occurs once in hundreds of years of data.'),
                        ('30 de zile, de circa 5 700 de ori mai multe decît cele 0,005 așteptate pentru distribuția Normală.', '30 days, about 5,700 times the 0.005 expected under the Normal distribution.'),
                        ('Circa 460: 5% dintre zile, ca la un prag de 5%.', 'About 460: 5% of the days, as for a 5% threshold.')]),
 'vol-table': (0, [('Bitcoin: 13,3 abateri standard, față de 11,3 la S&P 500.', 'Bitcoin: 13.3 standard deviations, against 11.3 for the S&P 500.'),
                   ('S&P 500: o variație de 12,8% este mult mai rară pentru un indice de acțiuni.', 'The S&P 500: a 12.8% move is much rarer for a stock index.'),
                   ('Sînt la fel de extreme: amîndouă sînt zile de crah, din martie 2020.', 'Equally extreme: both are crash days, in March 2020.')]),
 'annual': (1, [('Toate scad cu aceeași proporție.', 'They all fall in the same proportion.'),
                ('Practic doar Bitcoin se schimbă: 66,6% devine 55,3%, de √(252/365,3) = 0,831 ori.', 'Practically only Bitcoin changes: 66.6% becomes 55.3%, √(252/365.3) = 0.831 times.'),
                ('Bitcoin crește cu 45%, raportul 365,3/252.', 'Bitcoin rises by 45%, the ratio 365.3/252.')]),
 'rolling-windows': (1, [('Cea de 252 de zile: este cea mai netedă, deci cea mai precisă.', 'The 252-day window: it is the smoothest, hence the most precise.'),
                         ('Cea de 21 de zile (sau EWMA): o fereastră lungă include încă zilele liniștite dinaintea crahului.', 'The 21-day window (or EWMA): a long window still contains the calm days before the crash.'),
                         ('Volatilitatea pe întregul eșantion, 18,0%: folosește cele mai multe date.', 'The whole-sample volatility, 18.0%: it uses the most data.')]),
 'precision': (1, [('Circa 15%.', 'About 15%.'), ('Circa 39%.', 'About 39%.'), ('Circa 4,5%.', 'About 4.5%.')]),
 'ewma-weights': (2, [('Circa 69.', 'About 69.'), ('Circa 100.', 'About 100.'), ('Circa 458.', 'About 458.')]),
 'ewma-update': (1, [('1,48: λ schimbă doar viteza de atenuare de după șoc.', '1.48: λ only changes how fast the shock fades afterwards.'),
                     ('2,60: 0,80 × 1 + 0,20 × 9.', '2.60: 0.80 × 1 + 0.20 × 9.'), ('1,40: 0,80 × 1 + 0,20 × 3.', '1.40: 0.80 × 1 + 0.20 × 3.')]),
 'ghost': (0, [('Pe 15 iunie 2020, randamentul din 16 martie 2020 (−12,8%) a ieșit din fereastra de 63 de zile.', 'On 15 June 2020 the return of 16 March 2020 (−12.8%) left the 63-day window.'),
               ('O știre bună din ziua respectivă a calmat piața.', 'Good news on that day calmed the market.'),
               ('Scăderea EWMA a tras în jos și volatilitatea pe fereastră.', 'The fall of the EWMA pulled the window volatility down too.')]),
 'ohlc-path': (1, [('Și r², și estimarea Parkinson cresc.', 'Both r² and the Parkinson estimate rise.'),
                   ('Doar r² crește; amplitudinea u − d se măsoară față de deschidere și nu se schimbă.', 'Only r² rises; the range u − d is measured from the open and does not change.'),
                   ('Doar estimarea Parkinson crește, pentru că maximul este mai sus.', 'Only the Parkinson estimate rises, because the high is higher.')]),
 'parkinson': (1, [('8,42%.', '8.42%.'), ('5,06%.', '5.06%.'), ('3,04%.', '3.04%.')]),
 'gk-rs': (0, [('Circa 653 (%²).', 'About 653 (%²).'), ('Circa 442 (%²).', 'About 442 (%²).'), ('Circa 1 133 (%²).', 'About 1,133 (%²).')]),
 'yz': (1, [('Tinde spre 0: cu multe zile, varianța deschidere–închidere nu mai contează.', 'It tends to 0: with many days the open-to-close variance no longer matters.'),
            ('Crește ușor, la 0,144; limita pentru n mare este 0,34/2,34 = 0,145.', 'It rises slightly, to 0.144; the limit for large n is 0.34/2.34 = 0.145.'),
            ('Crește spre 0,5: noaptea și ședința primesc aceeași pondere.', 'It rises towards 0.5: the night and the session get the same weight.')]),
 'march16': (1, [('Circa 61%.', 'About 61%.'), ('Circa 37%.', 'About 37%.'), ('Circa 12%.', 'About 12%.')]),
 'efficiency': (0, [('Rămîne în jur de 9–10: raportul varianțelor depinde puțin de n.', 'It stays around 9–10: the ratio of the variances depends little on n.'),
                    ('Scade spre 1: pe ferestre scurte toți estimatorii sînt la fel de zgomotoși.', 'It falls towards 1: over short windows all estimators are equally noisy.'),
                    ('Crește peste 50: amplitudinile contează mai mult cînd sînt puține zile.', 'It rises above 50: ranges matter more when there are few days.')]),
 'eff-table': (1, [('Varianța estimării Parkinson crește mult cu saltul.', 'The variance of the Parkinson estimate rises a lot with the jump.'),
                   ('Deplasarea de circa −22%: MSE = varianță + deplasare², iar deplasarea domină.', 'The bias of about −22%: MSE = variance + bias², and the bias dominates.'),
                   ('Simularea are prea puține ferestre pentru a compara MSE.', 'The simulation has too few windows to compare MSE.')]),
 'est-table': (1, [('Ponderea mică a nopții (13%).', 'The small night share (13%).'),
                   ('Corelația pozitivă dintre o și c (0,22): niciun estimator de amplitudine nu vede termenul 2Cov(o, c).', 'The positive correlation of o and c (0.22): no range-based estimator sees the term 2Cov(o, c).'),
                   ('Zilele fără tranzacții, cu H = L.', 'Days without trades, with H = L.')]),
 'range-rolling': (0, [('Ambele cresc mult, la circa 30% pentru CC și 14% pentru Parkinson.', 'Both rise sharply, to about 30% for CC and 14% for Parkinson.'),
                       ('Ambele scad: o fereastră scurtă reacționează mai repede.', 'Both fall: a short window reacts faster.'),
                       ('Devin egale: pe ferestre scurte amplitudinea nu mai aduce informație.', 'They become equal: over short windows the range adds no information.')]),
 'rv-def': (0, [('1,051%: mai multe randamente, o eroare mai mică.', '1.051%: more returns, a smaller error.'),
                ('Peste 1,15%: zgomotul de microstructură crește RV la frecvențe mari.', 'Above 1.15%: microstructure noise raises RV at high frequencies.'),
                ('Exact 1,000%: cu un minut RV este varianța reală.', 'Exactly 1.000%: at one minute RV is the true variance.')]),
 'realised': (1, [('Tot circa 6%: RV nu depinde de interval.', 'Still about 6%: RV does not depend on the interval.'),
                  ('58%, ca |r|: cu M = 1, RV este chiar r².', '58%, as |r|: with M = 1, RV is just r².'), ('23%, ca Parkinson.', '23%, as Parkinson.')]),
 'signature': (1, [('De 3,58 ori: excesul se dublează odată cu zgomotul.', '3.58 times: the excess doubles with the noise.'),
                   ('De circa 4,2 ori: excesul de 0,79 crește cu pătratul zgomotului.', 'About 4.2 times: the excess of 0.79 grows with the square of the noise.'),
                   ('Tot de 1,79 ori: la un minut zgomotul se anulează.', 'Still 1.79 times: at one minute the noise cancels out.')]),
 'acf-clustering': (1, [('Tot 100: volatility clustering este la fel de persistent.', 'Still 100: volatility clustering is just as persistent.'),
                        ('72: banda este mai largă (T = 4 202), iar autocorelațiile la lagurile mari sînt mai mici.', '72: the band is wider (T = 4,202), and the autocorrelations at long lags are smaller.'),
                        ('Niciuna: după 2010 piața nu mai are volatility clustering.', 'None: after 2010 the market has no volatility clustering.')]),
 'acf-table': (1, [('d = 2: varianța se măsoară prin pătrate.', 'd = 2: variance is measured by squares.'), ('d = 1, adică |r|: efectul Taylor.', 'd = 1, that is |r|: the Taylor effect.'),
                   ('d = 0,25: puterile mici reduc influența zilelor extreme.', 'd = 0.25: small powers reduce the influence of extreme days.')]),
 'shuffle': (1, [('Tot circa 0: orice amestecare distruge volatility clustering.', 'Still about 0: any shuffle destroys volatility clustering.'),
                 ('Rămîne circa 0,23: patru din cinci perechi de zile consecutive rămîn în același bloc.', 'It stays around 0.23: four of five pairs of consecutive days remain in the same block.'),
                 ('Crește peste 0,27: blocurile concentrează zilele agitate.', 'It rises above 0.27: the blocks concentrate the turbulent days.')]),
 'lb-sq': (0, [('Circa 3 012.', 'About 3,012.'), ('Circa 8 958.', 'About 8,958.'), ('Circa 0,33.', 'About 0.33.')]),
 'arch-lm': (0, [('Circa 2 227.', 'About 2,227.'), ('Circa 1,21.', 'About 1.21.'), ('Circa 4 535.', 'About 4,535.')]),
 'cl-table': (1, [('Bitcoin nu are volatility clustering.', 'Bitcoin has no volatility clustering.'),
                  ('Seria este mai scurtă, iar cîteva zile extreme reduc R² la 0,026.', 'The series is shorter, and a few extreme days reduce R² to 0.026.'),
                  ('ARCH-LM cere distribuția Normală, iar Bitcoin are cozi groase.', 'ARCH-LM requires the Normal distribution, and Bitcoin has heavy tails.')]),
 'kurtosis': (1, [('p = 0,5: amestecul cel mai echilibrat.', 'p = 0.5: the most balanced mixture.'), ('p = 0,1, cu boltirea 8,33.', 'p = 0.1, with kurtosis 8.33.'),
                  ('p foarte aproape de 0: cu cît zilele agitate sînt mai rare, cu atît cozile sînt mai groase.', 'p very close to 0: the rarer the turbulent days, the heavier the tails.')]),
 'long-term': (0, [('φ̂ = 0,53 și timpul de înjumătățire 1,1 luni: șocurile se atenuează mai repede.', 'φ̂ = 0.53 and a half-life of 1.1 months: shocks fade faster.'),
                   ('φ̂ = 0,95: volatilitatea a devenit aproape un mers aleator.', 'φ̂ = 0.95: volatility has become almost a random walk.'),
                   ('φ̂ = 0,67, la fel: persistența nu depinde de eșantion.', 'φ̂ = 0.67, the same: persistence does not depend on the sample.')]),
 'vix': (0, [('Deasupra: investitorii plătesc o primă pentru asigurarea împotriva salturilor de volatilitate.', 'Above: investors pay a premium for insurance against volatility jumps.'),
             ('Dedesubt: VIX subestimează crizele.', 'Below: the VIX underestimates crises.'),
             ('La fel de des deasupra și dedesubt: VIX este o prognoză nedeplasată.', 'As often above as below: the VIX is an unbiased forecast.')]),
 'vix-table': (0, [('Ambele R² scad, la 0,37 și 0,26, iar VIX rămîne mai informativ.', 'Both R² fall, to 0.37 and 0.26, and the VIX stays more informative.'),
                   ('VIX devine mai slab decît volatilitatea trecută.', 'The VIX becomes weaker than the past volatility.'),
                   ('Ambele R² cresc: piețele recente sînt mai previzibile.', 'Both R² rise: recent markets are more predictable.')]),
 'leverage': (1, [('Efectul dispare: după 2010 BET este o piață matură.', 'The effect disappears: after 2010 the BET is a mature market.'),
                  ('Devine mai puternic: −0,141, iar raportul după scădere/după creștere urcă la 1,86.', 'It becomes stronger: −0.141, and the ratio after down/after up days rises to 1.86.'),
                  ('Își schimbă semnul: creșterile sînt urmate de o volatilitate mai mare.', 'It changes sign: rises are followed by higher volatility.')]),
 'news-impact': (1, [('Din cauza mediei pozitive a randamentelor (0,03% pe zi).', 'Because of the positive mean return (0.03% a day).'),
                     ('Din cauza asimetriei: după o creștere moderată urmează ziua cea mai liniștită, după o scădere de aceeași mărime una mai agitată.', 'Because of the asymmetry: a moderate rise is followed by the calmest day, a fall of the same size by a more turbulent one.'),
                     ('Este un artefact al nucleului cvartic.', 'It is an artefact of the quartic kernel.')]),
 'loss': (1, [('Amîndouă preferă h = 2.', 'Both prefer h = 2.'), ('MSE preferă h = 0,5 (0,25 față de 1), QLIKE preferă h = 2 (1,193 față de 1,307).', 'MSE prefers h = 0.5 (0.25 against 1), QLIKE prefers h = 2 (1.193 against 1.307).'),
              ('Amîndouă preferă h = 0,5: este mai aproape de 1.', 'Both prefer h = 0.5: it is closer to 1.')]),
 'fc-table': (1, [('Fereastra de 252 de zile: folosește cele mai multe date.', 'The 252-day window: it uses the most data.'),
                  ('VIX, dar diferența față de EWMA 0,94 nu este semnificativă (DM: t = −0,32).', 'The VIX, but its difference from EWMA 0.94 is not significant (DM: t = −0.32).'),
                  ('EWMA 0,94, semnificativ mai bună decît toate celelalte.', 'EWMA 0.94, significantly better than all the others.')]),
 'forecast-paths': (0, [('Fereastra de 252 de zile atinge din nou vîrful la sfîrșitul anului, după EWMA și VIX.', 'The 252-day window again peaks at the end of the year, after the EWMA and the VIX.'),
                        ('Toate trei ating vîrful în aceeași zi.', 'All three peak on the same day.'),
                        ('Fereastra de 252 de zile atinge prima vîrful: are cea mai lungă memorie.', 'The 252-day window peaks first: it has the longest memory.')]),
 'lambda': (0, [('0,940: valoarea RiskMetrics devine optimă.', '0.940: the RiskMetrics value becomes optimal.'), ('0,800: piețele recente cer o memorie mai scurtă.', '0.800: recent markets need a shorter memory.'),
                ('0,925, la fel: λ optim nu depinde de perioadă.', '0.925, the same: the optimal λ does not depend on the period.')]),
}

# blocks whose initial chart, table or worked example shows the answer: veil the readout and those elements
VEIL = {'vol-table': {'out': True, 'charts': ['vt-tab']}, 'annual': {'out': True, 'charts': ['an-we-box']}, 'precision': {'out': True, 'charts': ['pr-chart', 'pr-we-box']},
        'parkinson': {'out': True, 'charts': ['pk-we-box']}, 'gk-rs': {'out': True, 'charts': ['gr-we-box']}, 'march16': {'out': True, 'charts': ['ms-chart', 'ms-we-box']},
        'eff-table': {'out': True, 'charts': ['et-tab']}, 'est-table': {'out': True, 'charts': ['es-tab']}, 'acf-table': {'out': True, 'charts': ['at-chart', 'at-tab']},
        'lb-sq': {'out': True, 'charts': ['lq-chart', 'lq-we-box']}, 'arch-lm': {'out': True, 'charts': ['al-we-box']}, 'cl-table': {'out': True, 'charts': ['ct-tab']},
        'kurtosis': {'out': True, 'charts': ['ku-chart']}, 'vix': {'out': True, 'charts': ['vx-chart']}, 'loss': {'out': True, 'charts': ['ls-chart', 'ls-we-box']},
        'fc-table': {'out': True, 'charts': ['ft-chart', 'ft-tab']}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('O acțiune are într-o zi H = 102 și L = 98. Care este volatilitatea zilnică Parkinson?', 'A stock has H = 102 and L = 98 on one day. What is the daily Parkinson volatility?'),
      key=1, opts=[
  (('4,00%: amplitudinea ln(102/98).', '4.00%: the range ln(102/98).'), ('Amplitudinea depășește σ în medie de √(4 ln 2) = 1,67 ori; trebuie rescalată.', 'The range exceeds σ on average by a factor √(4 ln 2) = 1.67; it must be rescaled.')),
  (('2,40%: √(4,00²/2,77).', '2.40%: √(4.00²/2.77).'), ('ln(102/98) = 4,00%; 4,00²/2,77 = 5,77; √5,77 = 2,40%.', 'ln(102/98) = 4.00%; 4.00²/2.77 = 5.77; √5.77 = 2.40%.')),
  (('1,44%: 4,00/2,77.', '1.44%: 4.00/2.77.'), ('Împarte la 4 ln 2 amplitudinea, nu pătratul ei, și uită rădăcina.', 'It divides the range, not its square, by 4 ln 2 and forgets the square root.'))]),
 dict(q=('Testul Ljung–Box pe randamente nu respinge, iar pe randamentele la pătrat respinge puternic. Ce concluzie trageți?', 'The Ljung–Box test does not reject on the returns, and rejects strongly on the squared returns. What do you conclude?'),
      key=0, opts=[
  (('Media este imprevizibilă, varianța este previzibilă: volatility clustering fără autocorelația randamentelor.', 'The mean is unpredictable, the variance is predictable: volatility clustering without autocorrelation of the returns.'),
   ('Randamentele pot fi necorelate și totuși dependente: mărimea lor de azi anticipă mărimea de mîine.', 'Returns can be uncorrelated and still dependent: today\'s size anticipates tomorrow\'s.')),
  (('Randamentele sînt independente.', 'The returns are independent.'), ('Independența ar însemna și r² necorelat; testul pe r² respinge.', 'Independence would also mean uncorrelated r²; the test on r² rejects.')),
  (('Piața este ineficientă: randamentele pot fi prognozate.', 'The market is inefficient: returns can be forecast.'), ('O varianță previzibilă nu face previzibil semnul sau media randamentului.', 'A predictable variance does not make the sign or the mean of the return predictable.'))]),
 dict(q=('De ce nu poate EWMA cu λ = 0,94 să anticipeze revenirea volatilității la nivelul normal?', 'Why can EWMA with λ = 0.94 not anticipate the return of volatility to its normal level?'),
      key=0, opts=[
  (('α + β = 1 și ω = 0: prognozele lui sînt constante la orice orizont.', 'α + β = 1 and ω = 0: its forecasts are constant at every horizon.'),
   ('EWMA este IGARCH: E_t[σ²(t + h)] = σ²(t + 1) pentru orice h, deci nu există un nivel pe termen lung spre care să revină.', 'EWMA is IGARCH: E_t[σ²(t + h)] = σ²(t + 1) for every h, so there is no long-run level to revert to.')),
  (('λ = 0,94 este prea mic.', 'λ = 0.94 is too small.'), ('Orice λ din (0, 1) dă aceleași prognoze constante; λ schimbă doar viteza de reacție.', 'Any λ in (0, 1) gives the same constant forecasts; λ only changes the speed of reaction.')),
  (('EWMA folosește doar ultimele 75 de zile.', 'EWMA uses only the last 75 days.'), ('EWMA folosește toate zilele, cu ponderi care scad geometric; ultimele 75 de zile poartă doar 99% din pondere.', 'EWMA uses all days, with geometrically decaying weights; the last 75 days carry only 99% of the weight.'))]),
 dict(q=('Un activ tranzacționat în toate cele 365 de zile ale anului are volatilitatea zilnică 2%. Cît este volatilitatea anualizată?', 'An asset traded on all 365 days of the year has a daily volatility of 2%. What is its annualised volatility?'),
      key=0, opts=[
  (('38,2%: 2 × √365.', '38.2%: 2 × √365.'), ('A este frecvența reală a observațiilor: √365 = 19,10, iar 2 × 19,10 = 38,2%.', 'A is the actual frequency of the observations: √365 = 19.10, and 2 × 19.10 = 38.2%.')),
  (('31,7%: 2 × √252.', '31.7%: 2 × √252.'), ('Convenția de la acțiuni subestimează volatilitatea unui activ tranzacționat în fiecare zi, ca la Bitcoin (55,3% în loc de 66,6%).', 'The equity convention understates the volatility of an asset traded every day, as for Bitcoin (55.3% instead of 66.6%).')),
  (('730%: 2 × 365.', '730%: 2 × 365.'), ('Varianțele se adună în timp, nu abaterile standard: volatilitatea crește cu √A.', 'Variances add up over time, not standard deviations: volatility grows with √A.'))]),
 dict(q=('Varianța EWMA de ieri este 4 (%²), randamentul de ieri 1%, λ = 0,94. Cît este varianța EWMA de azi?', "Yesterday's EWMA variance is 4 (%²), yesterday's return 1%, λ = 0.94. What is today's EWMA variance?"),
      key=0, opts=[
  (('3,82: 0,94 × 4 + 0,06 × 1.', '3.82: 0.94 × 4 + 0.06 × 1.'), ('λ multiplică varianța de ieri, 1 − λ randamentul de ieri la pătrat; o zi liniștită coboară varianța.', 'λ multiplies yesterday\'s variance, 1 − λ yesterday\'s squared return; a calm day lowers the variance.')),
  (('1,18: 0,06 × 4 + 0,94 × 1.', '1.18: 0.06 × 4 + 0.94 × 1.'), ('Ponderile sînt inversate: cu λ = 0,94, trecutul primește ponderea mare.', 'The weights are swapped: with λ = 0.94 the past gets the large weight.')),
  (('5,00: 4 + 1.', '5.00: 4 + 1.'), ('EWMA este o medie ponderată, nu o sumă: ponderile 0,94 și 0,06 au suma 1.', 'EWMA is a weighted average, not a sum: the weights 0.94 and 0.06 add up to 1.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch8-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH8_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH8_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH8_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', ' ', t)
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
