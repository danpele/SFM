"""Build the interactive pages of Chapter 11: interactive/ch11.html (EN) and interactive/ch11_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch11.js, inputs in interactive/ch11_data.js (Quantlets/Ch_11/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch11.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 11
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_11'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('scaling', 'sfm_ch11_scaling', 'self'), ('rs-example', 'sfm_ch11_rs_example', 'rs'), ('fbm', 'sfm_ch11_fbm_paths', 'lm'), ('fgn', 'sfm_ch11_fgn_acf', 'lm'),
          ('arfima', 'sfm_ch11_arfima', 'lm'), ('rs-dfa', 'sfm_ch11_rs_dfa', 'est'), ('gph', 'sfm_ch11_gph', 'est'), ('lo', 'sfm_ch11_lo_test', 'est'), ('mc', 'sfm_ch11_mc_bias', 'est'),
          ('nile', 'sfm_ch11_nile', 'spur'), ('spurious', 'sfm_ch11_spurious', 'spur'), ('vol-acf', 'sfm_ch11_vol_acf', 'vol'), ('markets', 'sfm_ch11_markets', 'vol'),
          ('rolling', 'sfm_ch11_rolling', 'roll')]
# table and worked-example blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page);
# the EN deck has one acronym frame less, so its frame number is one lower (checked in frame_pages)
XBLOCKS = [('fractal-dim', 14, 'self', 'scaling'), ('rs-hurst', 21, 'rs', 'rs-example'), ('memory-we', 25, 'lm', 'fbm'), ('arfima-we', 33, 'lm', 'arfima'),
           ('dfa-we', 35, 'est', 'rs-dfa'), ('gph-we', 37, 'est', 'gph'), ('lo-we', 39, 'est', 'lo'), ('sp-table', 43, 'est', 'mc'), ('mk-table', 52, 'vol', 'markets'),
           ('risk-we', 61, 'risk', 'rolling')]
ORDER = ['fractal-dim', 'scaling', 'rs-example', 'rs-hurst', 'memory-we', 'fbm', 'fgn', 'arfima', 'arfima-we', 'dfa-we', 'rs-dfa', 'gph-we', 'gph', 'lo-we', 'lo', 'mc',
         'sp-table', 'nile', 'spurious', 'vol-acf', 'mk-table', 'markets', 'rolling', 'risk-we']
TITLES = {'fractal-dim': 'Dimensiunea fractală', 'rs-hurst': 'De la R/S la exponentul Hurst', 'memory-we': 'Exemplu rezolvat: cît de repede se stinge memoria?',
          'arfima-we': 'Exemplu rezolvat: ARFIMA(0; 0,3; 0)', 'dfa-we': 'Analiza fluctuațiilor fără tendință (DFA)', 'gph-we': 'Regresia GPH pe log-periodogramă',
          'lo-we': 'Testul R/S modificat al lui Lo', 'sp-table': 'S&P 500: toți estimatorii', 'mk-table': 'Șase piețe: randamente și randamente absolute',
          'risk-we': 'Scalarea riscului cu h^H'}

PFX = {'scaling': 'sc', 'rs-example': 'rx', 'fbm': 'fb', 'fgn': 'fg', 'arfima': 'af', 'rs-dfa': 'rd', 'gph': 'gp', 'lo': 'lo', 'mc': 'mc', 'nile': 'nl', 'spurious': 'su',
       'vol-acf': 'va', 'markets': 'mk', 'rolling': 'ro', 'fractal-dim': 'fd', 'rs-hurst': 'rh', 'memory-we': 'mw', 'arfima-we': 'aw', 'dfa-we': 'dw', 'gph-we': 'gw',
       'lo-we': 'lw', 'sp-table': 'st', 'mk-table': 'mt', 'risk-we': 'rk'}

SECTIONS = {
 'ro': dict(self='Autosimilaritate și fractali', rs='Hurst, Nilul și statistica R/S', lm='Memoria lungă: definiții și modele', est='Estimatori ai memoriei lungi',
            spur='Memorie lungă aparentă', vol='Memoria lungă a volatilității', roll='Exponenți Hurst pe ferestre mobile', risk='Memoria lungă și riscul'),
 'en': dict(self='Self-Similarity and Fractals', rs='Hurst, the Nile and the R/S Statistic', lm='Long Memory: Definitions and Models', est='Estimators of Long Memory',
            spur='Spurious Long Memory', vol='Long Memory in Volatility', roll='Rolling Hurst Exponents', risk='Long Memory and Risk'),
}

L = {
 'ro': dict(file='ch11_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 11 · Piețe fractale și memorie lungă', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: FMH (fractal market hypothesis, ipoteza pieței fractale), R/S (rescaled range, amplitudinea rescalată), '
        'DFA (detrended fluctuation analysis, analiza fluctuațiilor fără tendință), GPH (estimatorul Geweke–Porter-Hudak), fBm (fractional Brownian motion, mișcarea browniană fracționară), '
        'fGn (fractional Gaussian noise, zgomotul gaussian fracționar), ARFIMA (autoregressive fractionally integrated moving average), ACF (autocorrelation function, funcția de autocorelație), '
        'AR(1) (model autoregresiv de ordinul 1), GARCH (generalised autoregressive conditional heteroskedasticity), OLS (ordinary least squares, metoda celor mai mici pătrate), '
        'SE (standard error, eroarea standard), i.i.d. (independente și identic distribuite), VaR (value at risk, valoarea expusă la risc), TLV (Banca Transilvania), SNP (OMV Petrom).',
   src='Randamente logaritmice zilnice, în %: S&P 500, DAX și BET din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV) și OMV Petrom (SNP) '
       'din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. Prețuri EODHD. Debitul anual al Nilului la Aswan, 1871–1970 '
       '(Cobb, 1978; setul de date din statsmodels). Browserul calculează din aceste date scalarea, R/S, DFA, regresia GPH, testul lui Lo, ACF și exponenții pe ferestre mobile. '
       'Seriile simulate (fBm, fGn, ARFIMA) și toate estimările Monte Carlo (500 de serii i.i.d. pentru fiecare N, 500 de serii pentru testul lui Lo, 300 pentru memoria aparentă) vin din '
       'codul din curs, cu aceleași semințe; browserul calculează din ele mediile, benzile și ratele de respingere. Alte valori ale lui H și d, alte lungimi N, alte lățimi de bandă, '
       'alte ferestre, alte praguri ale rupturii Nilului și curbele i.i.d. pentru celelalte cinci serii sînt rulări suplimentare ale aceluiași cod; numărarea cutiilor pe traiectoriile '
       'fBm este un calcul suplimentar, care nu este în curs.'),
 'en': dict(file='ch11.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 11 · Fractal Markets and Long Memory', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: FMH (fractal market hypothesis), R/S (rescaled range), DFA (detrended fluctuation analysis), '
        'GPH (the Geweke–Porter-Hudak estimator), fBm (fractional Brownian motion), fGn (fractional Gaussian noise), ARFIMA (autoregressive fractionally integrated moving average), '
        'ACF (autocorrelation function), AR(1) (first-order autoregressive model), GARCH (generalised autoregressive conditional heteroskedasticity), OLS (ordinary least squares), '
        'SE (standard error), i.i.d. (independent and identically distributed), VaR (value at risk), TLV (Banca Transilvania), SNP (OMV Petrom).',
   src='Daily log returns, in %: S&P 500, DAX and BET from January 2000, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV) and OMV Petrom (SNP) from 2010, '
       'from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. EODHD prices. The annual flow of the Nile at Aswan, 1871–1970 (Cobb, 1978; '
       'the statsmodels data set). From these data the browser computes the scaling, R/S, DFA, the GPH regression, Lo\'s test, the ACF and the rolling exponents. The simulated series '
       '(fBm, fGn, ARFIMA) and every Monte Carlo estimate (500 i.i.d. series for each N, 500 series for Lo\'s test, 300 for spurious memory) come from the code of the lecture, with the '
       'same seeds; the browser computes the means, bands and rejection rates from them. Other values of H and d, other lengths N, bandwidths, windows and Nile break years, and the '
       'i.i.d. curves for the other five series are extra runs of the same code; box counting on the fBm paths is an extra computation that is not in the lecture.'),
}

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child,.xt td.l,.xt th.l{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt thead tr:first-child th[colspan]{text-align:center;border-bottom:1px solid #C5D2E8}
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
    M6 = lambda d: [(k, t + (lec if k == d else '')) for k, t in (('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)'))]
    ser = 'Seria' if ro else 'Series'
    YR = [('r', ('randamentele r' if ro else 'returns r') + lec), ('abs', 'randamentele absolute |r|' if ro else 'absolute returns |r|')]
    yl = 'Variabila' if ro else 'Variable'
    POW = lambda d: [(p, n(f'T^{p}') + (lec if p == d else '')) for p in ('0.5', '0.4', '0.6', '0.7')]
    bw = 'Lățimea de bandă m' if ro else 'Bandwidth m'
    return {
     'scaling': (select('sc-h', 'Orizontul maxim h (zile)' if ro else 'Largest horizon h (days)', [('250', '250' + lec), ('100', '100'), ('50', '50'), ('20', '20'), ('10', '10')], '250') + play('sc'),
                 CSS + charts('sc-chart')),
     'fractal-dim': (select('fd-h', 'H al traiectoriei fBm (sămînța 2026)' if ro else 'H of the fBm path (seed 2026)', [(h, n(h)) for h in ('0.5', '0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9')], '0.5') + play('fd'),
                     charts('fd-chart') + we('fd-we', f'{WE}: {"dimensiunea prin numărarea cutiilor (calcul suplimentar)" if ro else "box-counting dimension (extra computation)"}',
                                             r'D = \frac{\ln N}{\ln s}, \qquad N(\varepsilon) \propto \varepsilon^{-D}, \qquad D_{\text{fBm}} = 2 - H',
                                             select('fd-k', 'Cea mai mică latură ε' if ro else 'Smallest side ε', [('7', '1/128'), ('5', '1/32'), ('6', '1/64'), ('8', '1/256')], '7'))),
     'rs-example': (select('rx-w', 'Cele opt randamente' if ro else 'The eight returns', [('lec', ('exemplul din curs' if ro else 'example of the lecture') + lec), ('2000-01-04', 'S&P 500, 4–13 ian. 2000' if ro else 'S&P 500, 4–13 Jan 2000'),
                                                                                       ('2008-10-06', 'S&P 500, 6–15 oct. 2008' if ro else 'S&P 500, 6–15 Oct 2008'), ('2020-03-09', 'S&P 500, 9–18 mar. 2020' if ro else 'S&P 500, 9–18 Mar 2020'),
                                                                                       ('last', 'S&P 500, 9–18 sep. 2026' if ro else 'S&P 500, 9–18 Sep 2026')], 'lec') + play('rx'),
                    charts('rx-chart') + we('rx-we', f'{WE}: {"R/S pentru un bloc" if ro else "R/S of one block"}',
                                            r'Y_k = \sum_{j=1}^{k}(x_j - \bar x), \quad R_n = \max_k Y_k - \min_k Y_k, \quad S_n = \sqrt{\tfrac1n \sum_j (x_j - \bar x)^2}, \quad (R/S)_n = R_n/S_n')),
     'rs-hurst': (select('rh-s', ser, M6('sp500'), 'sp500') + select('rh-y', yl, YR, 'r') + play('rh'),
                  charts('rh-chart') + we('rh-we', f'{WE}: {"panta graficului log-log" if ro else "the slope of the log-log plot"}',
                                          r'\log_{10}(R/S)_n = \log_{10} c + H \log_{10} n + e_n, \qquad \hat H = \frac{\sum (u - \bar u)(v - \bar v)}{\sum (u - \bar u)^2}',
                                          slider('rh-n', 'Blocul n afișat (al cîtelea)' if ro else 'Block size n shown (index)', 1, 20, 1, 1))),
     'memory-we': (slider('mw-h', 'H al fGn' if ro else 'H of the fGn', 0.55, 0.95, 0.05, 0.8) + play('mw'),
                   charts('mw-chart') + we('mw-we', f'{WE}: {"fGn față de AR(1)" if ro else "fGn against AR(1)"}',
                                           r'\rho_{\text{fGn}}(k) = \tfrac12\left(|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}\right), \qquad \rho_{\text{AR}}(k) = \phi^k, \ \phi = \rho(1)',
                                           select('mw-k', 'Lagul k' if ro else 'Lag k', [('100', '100' + lec), ('10', '10'), ('50', '50'), ('250', '250')], '100'))),
     'fbm': (select('fb-d', 'Exponenții H' if ro else 'Exponents H', [('0.2', n('0.3; 0.5; 0.7') + lec if ro else '0.3, 0.5, 0.7' + lec), ('0.1', n('0.4; 0.5; 0.6') if ro else '0.4, 0.5, 0.6'),
                                                                       ('0.3', n('0.2; 0.5; 0.8') if ro else '0.2, 0.5, 0.8'), ('0.4', n('0.1; 0.5; 0.9') if ro else '0.1, 0.5, 0.9')], '0.2') + play('fb'), charts('fb-chart')),
     'fgn': (select('fg-h', 'H (și 1 − H)' if ro else 'H (and 1 − H)', [('0.8', n('0.8 și 0.2') + lec if ro else '0.8 and 0.2' + lec), ('0.6', n('0.6 și 0.4') if ro else '0.6 and 0.4'),
                                                                       ('0.7', n('0.7 și 0.3') if ro else '0.7 and 0.3'), ('0.9', n('0.9 și 0.1') if ro else '0.9 and 0.1')], '0.8') + play('fg'), charts('fg-chart')),
     'arfima': (select('af-d', 'd (și −d)' if ro else 'd (and −d)', [('0.4', n('0.4') + lec), ('0.1', n('0.1')), ('0.2', n('0.2')), ('0.3', n('0.3')), ('0.45', n('0.45'))], '0.4') + play('af'), charts('af-chart')),
     'arfima-we': (slider('aw-d', 'd', 0.05, 0.45, 0.05, 0.3) + play('aw'),
                   charts('aw-chart') + we('aw-we', f'{WE}: {"ponderile și autocorelațiile ARFIMA(0,d,0)" if ro else "weights and autocorrelations of ARFIMA(0,d,0)"}',
                                           r'\pi_j = \pi_{j-1}\frac{j - 1 - d}{j}, \qquad \psi_j = \psi_{j-1}\frac{j - 1 + d}{j}, \qquad \rho(k) = \rho(k - 1)\frac{k - 1 + d}{k - d}',
                                           select('aw-t', 'Pragul autocorelației' if ro else 'Autocorrelation threshold', [('0.05', n('0.05') + lec), ('0.1', n('0.1')), ('0.01', n('0.01'))], '0.05'))),
     'dfa-we': (select('dw-s', ser, M6('sp500'), 'sp500') + select('dw-y', yl, YR, 'r') + slider('dw-n', 'Mărimea blocului n (a cîta)' if ro else 'Block size n (index)', 1, 20, 1, 1) + play('dw'),
                charts('dw-chart') + we('dw-we', f'{WE}: {"fluctuația F(n)" if ro else "the fluctuation F(n)"}',
                                        r'Y_k = \sum_{j \le k}(x_j - \bar x), \qquad F(n) = \sqrt{\frac{1}{mn}\sum_{b=1}^{m}\sum_{i=1}^{n}\big(Y_{(b-1)n+i} - \hat a_b - \hat b_b\, i\big)^2} \propto n^{\alpha}')),
     'rs-dfa': (select('rd-s', ser, M6('sp500'), 'sp500') + play('rd'), charts('rd-chart')),
     'gph-we': (select('gw-s', ser, M6('sp500'), 'sp500') + select('gw-p', bw, POW('0.5'), '0.5') + play('gw'),
                charts('gw-chart') + we('gw-we', f'{WE}: {"regresia GPH" if ro else "the GPH regression"}',
                                        c(r'\ln I(\lambda_j) = c - d\,\ln\!\big(4\sin^2(\lambda_j/2)\big) + e_j, \quad j = 1, \dots, m, \qquad \mathrm{SE} = \pi/\sqrt{24m}'),
                                        select('gw-y', yl, YR, 'r'))),
     'gph': (select('gp-s', ser, M6('sp500'), 'sp500') + select('gp-p', bw, POW('0.5'), '0.5') + play('gp'), charts('gp-chart')),
     'lo-we': (select('lw-s', ser, M6('sp500'), 'sp500') + play('lw'),
               charts('lw-chart') + we('lw-we', f'{WE}: {"statistica V_N(q)" if ro else "the statistic V_N(q)"}',
                                       r'V_N(q) = \frac{R_N}{\sqrt{N}\,\hat\sigma_N(q)}, \qquad \hat\sigma^2_N(q) = \hat\gamma_0 + 2\sum_{j=1}^{q}\Big(1 - \frac{j}{q + 1}\Big)\hat\gamma_j',
                                       select('lw-y', yl, YR, 'r'))
               + xh('Testul lui Lo pe cele șase serii (la 5%: memoria scurtă este respinsă în afara [0,809; 1,862])' if ro else "Lo's test on the six series (at 5%: short memory is rejected outside [0.809, 1.862])") + tw('lw-tab')),
     'lo': (select('lo-n', 'Lungimea seriilor N' if ro else 'Length of the series N', [('1000', '1000' + lec), ('250', '250'), ('500', '500'), ('2000', '2000')], '1000') + play('lo'), charts('lo-chart')),
     'mc': (select('mc-l', 'Nivelul benzii' if ro else 'Level of the band', [('0.95', '95%' + lec), ('0.9', '90%'), ('0.99', '99%')], '0.95') + play('mc'), charts('mc-chart')),
     'sp-table': (select('st-s', ser, M6('sp500'), 'sp500') + play('st'),
                  charts('st-chart') + xh('Toți estimatorii; benzile de 95% vin din 500 de serii i.i.d. de aceeași lungime (cu roșu: în afara benzii)' if ro
                                          else 'All estimators; the 95% bands come from 500 i.i.d. series of the same length (in red: outside the band)') + tw('st-tab')),
     'nile': (slider('nl-b', 'Ultimul an înainte de ruptură' if ro else 'Last year before the break', 1880, 1960, 1, 1898) + play('nl'), charts('nl-chart')),
     'spurious': (select('su-n', 'Lungimea seriilor N' if ro else 'Length of the series N', [('2000', '2000' + lec), ('500', '500'), ('1000', '1000'), ('4000', '4000')], '2000') + play('su'), charts('su-chart')),
     'vol-acf': (select('va-s', ser, M6('sp500'), 'sp500') + select('va-l', 'Numărul de laguri' if ro else 'Number of lags', [('250', '250' + lec), ('100', '100'), ('500', '500')], '250') + play('va'), charts('va-chart')),
     'mk-table': (select('mt-p', 'Lățimea de bandă GPH m' if ro else 'GPH bandwidth m', POW('0.5'), '0.5') + play('mt'),
                  charts('mt-chart') + xh('Șase piețe (cu roșu: DFA peste bandă, testul lui Lo respinge memoria scurtă)' if ro else "Six markets (in red: DFA above the band, Lo's test rejects short memory)") + tw('mt-tab')),
     'markets': (select('mk-e', 'Estimatorul' if ro else 'Estimator', [('dfa', 'DFA' + lec), ('rs', 'R/S'), ('gph', 'GPH, d + 0,5' if ro else 'GPH, d + 0.5')], 'dfa') + play('mk'), charts('mk-chart')),
     'rolling': (select('ro-w', 'Fereastra (observații)' if ro else 'Window (observations)', [('1000', '1000' + lec), ('250', '250'), ('500', '500'), ('2000', '2000')], '1000') + play('ro'), charts('ro-chart')),
     'risk-we': (slider('rk-h', 'H', 0.4, 0.6, 0.01, 0.55) + play('rk'),
                 charts('rk-chart') + we('rk-we', f'{WE}: {"VaR 1% pe h zile" if ro else "VaR 1% over h days"}',
                                         c(r'\mathrm{VaR}_{0.01}(h) = -z_{0.01}\,\sigma\,h^{H}, \qquad z_{0.01} = -2.326, \quad \sigma = 1.2\%'),
                                         select('rk-n', 'Orizontul h (zile)' if ro else 'Horizon h (days)', [('10', '10' + lec), ('20', '20'), ('50', '50'), ('100', '100'), ('250', '250')], '10'))),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tt = lambda t: latex_text(re.sub(r'\\sfmchlinkt?\{[^{}]*\}\{([^{}]*)\}', r'\1', t))
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    ro = frames_of(open(os.path.join(ROOT, TEX['ro']) + '.tex', encoding='utf-8').read())
    off = len(ro) - len(frames)
    res = {}
    for a, fr, _, _ in XBLOCKS:
        assert tt(ro[fr - 1][0]) == TITLES[a], (a, tt(ro[fr - 1][0]))
        k = fr - 1 - off
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch11.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch11_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch11_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s  ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => (num(s) === 1 ? '1 zi' : `${s} ${de(s)}zile`);
const X = ' (rulare suplimentară a codului din curs)';
const lst = a => (a.length ? a.join(', ') : 'niciuna');
const yl = y => (y === 'abs' ? 'randamentele absolute |r|' : 'randamentele r');
const sub = i => String(i).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
const neg = s => String(s).replace('−', '');
window.CH11_TEXT = {
 h: { r: 'randamente rₜ', abs: 'randamente absolute |rₜ|', band: 'banda de 95%, i.i.d.',
      st: ['', 'R/S', 'DFA', 'GPH d̂ (SE)', 'd̂ ± 1,96 SE', 'Lo V_N(q)', 'q'],
      lw: [['', { v: 'randamente r', span: 4 }, { v: 'randamente absolute |r|', span: 4 }], ['', 'V(0)', 'V(q)', 'q', 'memoria scurtă', 'V(0)', 'V(q)', 'q', 'memoria scurtă']],
      mt: [['', '', { v: 'randamente rₜ', span: 4 }, { v: 'randamente absolute |rₜ|', span: 3 }, ''], ['', 'N', 'R/S', 'DFA', 'GPH d̂', 'd̂ ± 1,96 SE', 'DFA', 'GPH d̂', 'Lo V', 'banda DFA']] },
 sc: (hm, s10, shm, items, x) => `Abaterea standard a randamentelor pe h zile, fără suprapunere, împărțită la cea zilnică, h de la 1 la ${hm}${x ? X : ''}. `
   + items.map(([n, H, r10, rh]) => `${n}: panta ${H}, raportul la 10 zile ${r10}` + (+hm > 10 ? `, la ${zile(hm)} ${rh}` : '')).join('; ')
   + `. Regula rădăcinii pătrate a timpului: √10 = ${s10}` + (+hm > 10 ? `, √${hm} = ${shm}.` : '.'),
 rxlec: 'cele opt randamente din curs',
 rxsrc: (d0, d1) => `randamentele S&P 500 din ${d0} – ${d1} (8 zile de tranzacționare)`,
 rx1: (src, xs, s, n, m) => `${src[0].toUpperCase() + src.slice(1)}, în %: ${xs}; suma ${s}, media x̄ = ${s}/${n} = ${m}.`,
 rx2: devs => `Abaterile de la medie, x_k − x̄: ${devs}.`,
 rx3: (Y, yn) => `Abaterile cumulate Y_k: ${Y}; Y₈ = ${yn}, zero prin construcție.`,
 rx4: (mx, kx, mn, kn, R, ss, n, S) => `R = max Y_k − min Y_k = ${mx} (k = ${kx}) − (${mn}) (k = ${kn}) = ${R}; S = √(Σ(x_k − x̄)²/n) = √(${ss}/${n}) = ${S}.`,
 rx5: (R, S, rs) => `(R/S)₈ = ${R}/${S} = ${rs}. Un bloc dă un singur punct al graficului log-log; H are nevoie de multe mărimi de bloc.`,
 rx: (src, R, S, rs, x) => `${src[0].toUpperCase() + src.slice(1)}${x ? X : ''}: R = ${R}, S = ${S}, (R/S)₈ = ${rs}.`,
 fb: (items, x) => `Aceleași numere aleatoare (sămînța 2026) pentru toate traiectoriile${x ? X : ''}. ` + items.map(([H, a, t, r]) => `H = ${H}: autocorelația de ordinul 1 a creșterilor `
   + `${a} (teoretic 2^(2H − 1) − 1 = ${t}), amplitudinea traiectoriei ${r}`).join('; ') + '.',
 fg: (H, r1, Hb, r1b, r10, r50, r100, a10, a100, s1, gap, x) => `H = ${H}${x ? X : ''}: ρ(1) = 2^(2H − 1) − 1 = ${r1}; pentru H = ${Hb}, ρ(1) = ${r1b}. fGn cu H = ${H}: ρ(10) = ${r10}, `
   + `ρ(50) = ${r50}, ρ(100) = ${r100}; AR(1) cu φ = ${r1}: ρ(10) = ${a10}, ρ(100) = ${a100}. ACF de selecție (n = 5000) la lagul 1: ${s1}; în medie, pe lagurile 1–50, ea este `
   + (num(gap) >= 0 ? `cu ${gap} sub cea teoretică.` : `cu ${neg(gap)} peste cea teoretică.`),
 af: (items, x) => `Traiectorii de 1000 de valori, ACF de selecție din 20000 de valori${x ? X : ''}. ` + items.map(([d, r1, r10, r50, s1, sd, st]) => `d = ${d}: ρ(1) = d/(1 − d) = ${r1}, `
   + `ρ(10) = ${r10}, ρ(50) = ${r50} (ACF de selecție la lagul 1: ${s1}); abaterea standard a traiectoriei ${sd}, teoretic ${st}`).join('; ') + '.',
 rd: (s, n, K, n1, nK, rs, rsi, dfa, dfai, rsa, dfaa, x) => `${s}, ${n} ${de(n)}randamente, ${K} ${de(K)}mărimi de bloc de la ${n1} la ${nK} ${de(nK)}zile${x ? X : ''}. Randamente: panta R/S ${rs} `
   + `(serii i.i.d. de aceeași lungime: ${rsi}), DFA ${dfa} (i.i.d.: ${dfai}). Randamente absolute: R/S ${rsa}, DFA ${dfaa}.`,
 gp: (s, p, m, d, se, lo, hi, da, Ha, x) => `${s}, m = ⌊T^${p}⌋ = ${m} frecvențe${x ? X : ''}: randamente d̂ = ${d} (SE ${se}; d̂ ± 1,96 SE = [${lo}; ${hi}], `
   + (num(lo) < 0 && num(hi) > 0 ? 'conține 0' : 'nu conține 0') + `); randamente absolute d̂ = ${da}, Ĥ = d̂ + 0,5 = ${Ha}.`,
 lo: (n, c5, m5, q5, c7, m7, c0, m0, m8, x) => `500 de serii de ${n} ${de(n)}valori${x ? X : ''}. AR(1) cu φ = 0,5 (memorie scurtă, H₀ adevărată): testul clasic respinge în ${c5} din cazuri, `
   + `testul modificat în ${m5} (q mediu ${q5}); cu φ = 0: ${c0} și ${m0}. fGn cu H = 0,7: ${c7} și ${m7}; cu H = 0,8, testul modificat respinge în ${m8}.`,
 mc: (lv, r250, r1000, r5000, d1000, g1000, wr, wd, wg, wd250, lo, hi, x) => `500 de serii i.i.d. Normale pentru fiecare N, banda de ${lv}%${x ? X : ''}. R/S: media ${r250} la N = 250, `
   + `${r1000} la N = 1000, ${r5000} la N = 5000; DFA ${d1000} și GPH ${g1000} la N = 1000. Lățimea benzii la N = 1000: R/S ${wr} (de la ${lo} la ${hi}), DFA ${wd}, GPH ${wg}; `
   + `la N = 250, DFA ${wd250}.`,
 nl: (b, mp, mq, dr, H1, d1, H2, d2, lo, hi, nm, x) => `Ruptura după ${b}${x ? X : ''}: debitul mediu ${mp} înainte, ${mq} după (în 10⁸ m³), `
   + (num(dr) >= 0 ? `o scădere de ${dr}%` : `o creștere de ${neg(dr)}%`) + `. Seria brută: Ĥ_R/S = ${H1}, DFA ${d1}; după eliminarea celor două medii de regim: ${H2} și ${d2}. `
   + `Banda i.i.d. pentru R/S cu N = 100: [${lo}; ${hi}], centrată în ${nm}.`,
 su: (n, g5, d5, r5, r0, a95, a99, rlo, rhi, x) => `300 de serii de ${n} ${de(n)}valori${x ? X : ''}. Schimbare a mediei de 0,5 abateri standard la mijloc: GPH Ĥ = ${g5}, DFA ${d5}, `
   + `R/S ${r5} (fără schimbare, R/S dă deja ${r0}: deplasarea de eșantion mic). GARCH(1,1): DFA pentru |r| ${a95} la α + β = 0,95 și ${a99} la 0,99; pentru randamente, DFA ` + (rlo === rhi ? `${rlo} la toate persistențele.` : `între ${rlo} și ${rhi}.`),
 va: (s, L, band, a1, a50, a100, np, s1, s50, r1, mx, no, ex, top, x) => `${s}, primele ${L} ${de(L)}laguri${x ? X : ''}; banda de 95%: ±${band}. |r|: ρ(1) = ${a1}, ρ(50) = ${a50}, `
   + `ρ(100) = ${a100}; ${np} dintre cele ${L} ${de(L)}laguri sînt peste bandă. r²: ρ(1) = ${s1}, ρ(50) = ${s50}. r: ρ(1) = ${r1}, apoi |ρ(k)| ≤ ${mx}; ${no} dintre lagurile 2–${L} `
   + `sînt în afara benzii (circa ${ex} așteptate din întîmplare), cele mai mari la lagurile ${top}.`,
 mk: (e, amin, amax, smin, smax, ins, above, x) => `Estimatorul ${e}${x ? X : ''}: pentru |r|, între ${amin} și ${amax}; după o permutare aleatoare, între ${smin} și ${smax}, `
   + `${ins} din 6 în interiorul benzilor: memoria se află în ordinea zilelor. Randamente peste bandă: ${lst(above)}.`,
 ro: (W, lo, hi, items, x) => `Ferestre de ${W} de observații, mutate cu 21${x ? X : ''}; banda i.i.d. de 95%: [${lo}; ${hi}]. ` + items.map(([s, n, d0, mn, mx, dmn, dmx, ab, be, t1, t3, amn, amx]) =>
   `${s}: ${n} ${de(n)}ferestre, prima încheiată la ${d0}; Ĥ al randamentelor între ${mn} (${dmn}) și ${mx} (${dmx}); ${ab}% dintre ferestre peste bandă, ${be}% sub; media primei treimi `
   + `${t1}, a ultimei ${t3}; |r| între ${amn} și ${amx}`).join('. ') + '.',
 fd1: 'Dimensiunea de similaritate D = ln N/ln s: segmentul ln 2/ln 2 = 1, pătratul ln 4/ln 2 = 2, curba Koch ln 4/ln 3 = 1,262, triunghiul Sierpinski ln 3/ln 2 = 1,585.',
 fd2: (H, n) => `Calcul suplimentar, care nu este în curs: traiectoria fBm cu H = ${H} (${n} de creșteri, sămînța 2026) este rescalată în pătratul unitate; pentru ε = 1/2^k împărțim `
   + 'pătratul în coloane de lățime ε și numărăm în fiecare coloană cutiile de latură ε atinse de traiectorie.',
 fd3: list => 'Numărul de cutii N(ε): ' + list.map(([k, m, n]) => `ε = 1/${m}: ${n}`).join('; ') + '.',
 fd4: (km, D, th, df) => `Panta OLS a lui log₂ N(ε) în funcție de log₂(1/ε), pentru ε de la 1/2 la 1/${2 ** km}: D = ${D}; pentru fBm, D = 2 − H = ${th} (diferența ${df}).`,
 fd: (H, km, D, th) => `Calcul suplimentar (numărarea cutiilor pe traiectoriile fBm din curs): H = ${H}, ε pînă la 1/${2 ** km}: D = ${D}, față de 2 − H = ${th}.`,
 rh1: (s, y, N, K, n1, nK) => `${s}, ${yl(y)}: N = ${N}; ${K} ${de(K)}mărimi de bloc între ${n1} și ${nK} ≈ N/4, la distanțe egale pe scară logaritmică.`,
 rh2: (n, m, rest, rs, lx, ly) => `n = ${n}: ⌊N/n⌋ = ${m} ${de(m)}blocuri fără suprapunere (` + (rest === 0 ? 'toate randamentele intră în blocuri' : rest === 1 ? 'ultimul randament rămîne în afara lor'
   : `ultimele ${rest} ${de(rest)}randamente rămîn în afara lor`) + `); media R/S pe blocuri: (R/S)_n = ${rs}; u = log₁₀ n = ${lx}, v = log₁₀ (R/S)_n = ${ly}.`,
 rh3: (mu, mv, sxx, sxy, b, a) => `Pe toate punctele graficului: ū = ${mu}, v̄ = ${mv}, Σ(u − ū)² = ${sxx}, Σ(u − ū)(v − v̄) = ${sxy}; Ĥ = ${sxy}/${sxx} = ${b}; log₁₀ c = v̄ − Ĥū = ${a}.`,
 rh4: (N, lo, hi, st, y) => `Banda de 95% a lui Ĥ_R/S pentru serii i.i.d. Normale cu N = ${N}: [${lo}; ${hi}]; ` + (st > 0 ? 'Ĥ este peste bandă' + (y === 'abs' ? ': memorie lungă a volatilității.' : '.')
   : st < 0 ? 'Ĥ este sub bandă.' : 'Ĥ se află în bandă: nu se deosebește de seriile fără memorie.'),
 rh: (s, y, H, lo, hi, x) => `${s}, ${yl(y)}${x ? X : ''}: Ĥ_R/S = ${H}; banda i.i.d. de 95%: [${lo}; ${hi}].`,
 mw1: (H, e, r1) => `H = ${H}: ρ(1) = 2^(2H − 1) − 1 = 2^${e} − 1 = ${r1}.`,
 mw2: (r10, r50, r100, k, rk, ap) => `fGn: ρ(10) = ${r10}, ρ(50) = ${r50}, ρ(100) = ${r100}; la lagul ${k}: ρ = ${rk}, iar aproximarea H(2H − 1)k^(2H − 2) dă ${ap}.`,
 mw3: (phi, a10, k, ak) => `AR(1) cu același ρ(1), φ = ${phi}: ρ(10) = φ¹⁰ = ${a10}; ρ(${k}) = φ^${k} = ${ak}.`,
 mw4: (k, rk, ak) => `După ${k} ${de(k)}laguri, un șoc mai contează în fGn cu autocorelația ${rk}, iar în AR(1) practic deloc (${ak}): același comportament pe termen scurt, un termen lung foarte diferit.`,
 mw: (H, r1, r100, a100, x) => `H = ${H}${x ? X : ''}: ρ(1) = ${r1}; ρ(100) = ${r100} pentru fGn și ${a100} pentru AR(1) cu același ρ(1).`,
 aw1: (d, p1, p2) => `Ponderile lui (1 − L)^d, d = ${d}: π₁ = −d = ${p1}, π₂ = π₁(1 − d)/2 = ${p2}.`,
 aw2: (s1, s2) => `Ponderile MA: ψ₁ = d = ${s1}, ψ₂ = ψ₁(1 + d)/2 = ${s2}.`,
 aw3: (d, d1, r1, dp, dm, r2, r10, r100) => `ρ(1) = d/(1 − d) = ${d}/${d1} = ${r1}; ρ(2) = ρ(1) × ${dp}/${dm} = ${r2}; ρ(10) = ${r10}, ρ(100) = ${r100}.`,
 aw4: (r1, a10) => `AR(1) cu φ = ${r1}: ρ(10) = φ¹⁰ = ${a10}.`,
 aw5: (t, kA, kF, H) => `Autocorelația scade sub ${t} după ${kA} ${de(kA)}laguri pentru AR(1) și după ${kF} ${de(kF)}laguri pentru ARFIMA; H = d + 0,5 = ${H}.`,
 aw: (d, r1, r10, r100, t, kA, kF, x) => `d = ${d}${x ? X : ''}: ρ(1) = ${r1}, ρ(10) = ${r10}, ρ(100) = ${r100}; sub ${t} după ${kA} ${de(kA)}laguri (AR(1)) și după ${kF} ${de(kF)}laguri (ARFIMA).`,
 dw1: (s, y, N, m, Y1, Y2, YN) => `${s}, ${yl(y)}: N = ${N}, media x̄ = ${m}; profilul Y_k = Σ_{j≤k}(x_j − x̄): Y₁ = ${Y1}, Y₂ = ${Y2}, …, Y_N = ${YN}.`,
 dw2: (n, m, rest, a, b, r) => `n = ${n}: ${m} ${de(m)}blocuri (` + (rest === 0 ? 'tot profilul intră în blocuri' : rest === 1 ? 'ultima valoare a profilului rămîne în afara lor'
   : `ultimele ${rest} ${de(rest)}valori ale profilului rămîn în afara lor`) + `). Primul bloc: dreapta OLS are la k = 1 valoarea ${a} și panta ${b}; rădăcina mediei pătratelor abaterilor de la ea: ${r}.`,
 dw3: (m, n, F) => `F(${n}) = rădăcina mediei pătratelor abaterilor pe toate cele ${m} ${de(m)}blocuri de cîte ${n} ${de(n)}valori: ${F}.`,
 dw4: (K, a, lo, hi, y) => `Panta lui log F(n) în funcție de log n pe cele ${K} ${de(K)}mărimi de bloc: α̂ = ${a}; banda i.i.d. de 95% a DFA pentru același N: [${lo}; ${hi}]`
   + (y === 'abs' ? '; pentru |r|, α̂ mult peste bandă indică memorie lungă a volatilității.' : '.'),
 dw: (s, y, n, F, a, x) => `${s}, ${yl(y)}${x ? X : ''}: F(${n}) = ${F}; exponentul DFA α̂ = ${a}.`,
 gw1: (s, y, T, p, m, l1, lm) => `${s}, ${yl(y)}: T = ${T}, m = ⌊T^${p}⌋ = ${m}; frecvențele λ_j = 2πj/T, de la λ₁ = ${l1} la λ_m = ${lm}.`,
 gw2: (I1, X1, y1) => `j = 1: I(λ₁) = ${I1}; regresorul −ln(4 sin²(λ₁/2)) = ${X1}; ln I(λ₁) = ${y1}.`,
 gw3: (mx, my, sxx, sxy, d) => `Pe cele m puncte: media regresorului ${mx}, media lui ln I ${my}, Σ(X − X̄)² = ${sxx}, Σ(X − X̄)(y − ȳ) = ${sxy}; d̂ = ${sxy}/${sxx} = ${d}.`,
 gw4: (m, se, lo, hi, H) => `SE = π/√(24m) = π/√(24 × ${m}) = ${se}; intervalul de 95%: d̂ ± 1,96 SE = [${lo}; ${hi}]` + (num(lo) < 0 && num(hi) > 0 ? ', conține 0' : ', nu conține 0') + `; Ĥ = d̂ + 0,5 = ${H}.`,
 gw: (s, p, m, dr, da, se, x) => `${s}, m = ⌊T^${p}⌋ = ${m}${x ? X : ''}: d̂ = ${dr} pentru randamente și ${da} pentru randamentele absolute, SE ${se}. Un m mai mare micșorează SE, dar lasă să intre dinamica de termen scurt.`,
 lw1: (s, y, N, R, g0, V0) => `${s}, ${yl(y)}: N = ${N}; amplitudinea abaterilor cumulate R_N = ${R}; γ̂₀ = ${g0}; R/S clasic: V_N(0) = R_N/√(Nγ̂₀) = ${V0}.`,
 lw2: (rho, A, B, AB, q) => `ρ̂₁ = ${rho}; (3N/2)^(1/3) = ${A}; (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = ${B}; produsul ${AB}, deci q = ${q}.`,
 lw3: (q, g0, gs, s2) => (q === 0 ? `Cu q = 0, σ̂²(0) = γ̂₀ = ${g0}.` : `σ̂²(${q}) = γ̂₀ + 2 Σ (1 − j/${q + 1}) γ̂_j, j = 1, …, ${q}, cu ` + gs.map((v, i) => `γ̂${sub(i + 1)} = ${v}`).join(', ')
   + (q > gs.length ? ', …' : '') + `: σ̂²(${q}) = ${s2}.`),
 lw4: (R, N, s2, V, rej) => `V_N(q) = ${R}/√(${N} × ${s2}) = ${V}: ` + (rej ? 'în afara intervalului [0,809; 1,862], deci memoria scurtă este respinsă la 5%.' : 'în intervalul [0,809; 1,862], deci memoria scurtă nu este respinsă la 5%.'),
 lw: (s, Vr, qr, rr, Va, qa, ra, list, x) => `${s}${x ? X : ''}: randamente V = ${Vr} (q = ${qr}), memoria scurtă ${rr ? 'este respinsă' : 'nu este respinsă'}; randamente absolute V = ${Va} (q = ${qa}), `
   + `${ra ? 'respinsă' : 'nerespinsă'}. Dintre cele șase serii, memoria scurtă a randamentelor este respinsă pentru: ${lst(list)}.`,
 st: (s, n, d0, outR, lr, nA, la, x) => `${s}: N = ${n} randamente zilnice, din ${d0} pînă la 18 septembrie 2026${x ? X : ''}. Randamente: ` + (outR.length ? `în afara benzii: ${lst(outR)}` : 'fiecare estimare se află în banda ei')
   + `; testul lui Lo ${lr ? 'respinge' : 'nu respinge'} memoria scurtă. Randamente absolute: ${nA} din 3 estimări în afara benzii; testul lui Lo ${la ? 'respinge' : 'nu respinge'} memoria scurtă.`,
 mt: (p, m0, m1, ex, dmn, dmx, nr, x) => `Lățimea de bandă GPH m = ⌊T^${p}⌋, între ${m0} și ${m1} frecvențe${x ? X : ''}. Randamente: d̂ ± 1,96 SE exclude 0 pentru: ${lst(ex)}. `
   + `Randamente absolute: d̂ între ${dmn} și ${dmx}; testul lui Lo respinge memoria scurtă pentru ${nr} din 6 serii.`,
 rk1: (z, s, v) => `VaR 1% pe o zi, cazul Normal cu media 0: −z₀,₀₁ σ = ${z} × ${s} = ${v}%.`,
 rk2: (h, H, f, v) => `h = ${h}, H = ${H}: factorul h^H = ${h}^${H} = ${f}; VaR 1% pe ${zile(h)} = ${v}%.`,
 rk3: (h, sq, vs, p) => `Regula √h: √${h} = ${sq}, VaR = ${vs}%; cu exponentul H, riscul este cu ${neg(p)}% ${num(p) >= 0 ? 'mai mare' : 'mai mic'}.`,
 rk4: (sp, bt, s10) => `Datele: raportul abaterilor standard pe 10 zile și pe o zi este ${sp} pentru S&P 500 și ${bt} pentru BET, față de √10 = ${s10}.`,
 rk: (H, h, f, v, vs, p, p250, x) => `H = ${H}, h = ${h}${x ? X : ''}: factorul ${f}, VaR 1% = ${v}%, față de ${vs}% cu regula √h; diferența: ${p}% la ${zile(h)}, ${p250}% la 250 de zile.`
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/[\s,  ]/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
const X = ' (extra run of the code of the lecture)';
const lst = a => (a.length ? a.join(', ') : 'none');
const yl = y => (y === 'abs' ? 'absolute returns |r|' : 'returns r');
const sub = i => String(i).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
const neg = s => String(s).replace('−', '');
window.CH11_TEXT = {
 h: { r: 'returns rₜ', abs: 'absolute returns |rₜ|', band: '95% band, i.i.d.',
      st: ['', 'R/S', 'DFA', 'GPH d̂ (SE)', 'd̂ ± 1.96 SE', 'Lo V_N(q)', 'q'],
      lw: [['', { v: 'returns r', span: 4 }, { v: 'absolute returns |r|', span: 4 }], ['', 'V(0)', 'V(q)', 'q', 'short memory', 'V(0)', 'V(q)', 'q', 'short memory']],
      mt: [['', '', { v: 'returns rₜ', span: 4 }, { v: 'absolute returns |rₜ|', span: 3 }, ''], ['', 'N', 'R/S', 'DFA', 'GPH d̂', 'd̂ ± 1.96 SE', 'DFA', 'GPH d̂', 'Lo V', 'DFA band']] },
 sc: (hm, s10, shm, items, x) => `Standard deviation of non-overlapping h-day returns divided by the daily one, h from 1 to ${hm}${x ? X : ''}. `
   + items.map(([n, H, r10, rh]) => `${n}: slope ${H}, ratio at 10 days ${r10}` + (+hm > 10 ? `, at ${days(hm)} ${rh}` : '')).join('; ')
   + `. The square-root-of-time rule: √10 = ${s10}` + (+hm > 10 ? `, √${hm} = ${shm}.` : '.'),
 rxlec: 'the eight returns of the lecture',
 rxsrc: (d0, d1) => `the S&P 500 returns from ${d0} to ${d1} (8 trading days)`,
 rx1: (src, xs, s, n, m) => `${src[0].toUpperCase() + src.slice(1)}, in %: ${xs}; sum ${s}, mean x̄ = ${s}/${n} = ${m}.`,
 rx2: devs => `Deviations from the mean, x_k − x̄: ${devs}.`,
 rx3: (Y, yn) => `Cumulative deviations Y_k: ${Y}; Y₈ = ${yn}, zero by construction.`,
 rx4: (mx, kx, mn, kn, R, ss, n, S) => `R = max Y_k − min Y_k = ${mx} (k = ${kx}) − (${mn}) (k = ${kn}) = ${R}; S = √(Σ(x_k − x̄)²/n) = √(${ss}/${n}) = ${S}.`,
 rx5: (R, S, rs) => `(R/S)₈ = ${R}/${S} = ${rs}. One block gives a single point of the log-log plot; H needs many block sizes.`,
 rx: (src, R, S, rs, x) => `${src[0].toUpperCase() + src.slice(1)}${x ? X : ''}: R = ${R}, S = ${S}, (R/S)₈ = ${rs}.`,
 fb: (items, x) => `The same random numbers (seed 2026) for all paths${x ? X : ''}. ` + items.map(([H, a, t, r]) => `H = ${H}: first-order autocorrelation of the increments `
   + `${a} (theoretical 2^(2H − 1) − 1 = ${t}), range of the path ${r}`).join('; ') + '.',
 fg: (H, r1, Hb, r1b, r10, r50, r100, a10, a100, s1, gap, x) => `H = ${H}${x ? X : ''}: ρ(1) = 2^(2H − 1) − 1 = ${r1}; for H = ${Hb}, ρ(1) = ${r1b}. fGn with H = ${H}: ρ(10) = ${r10}, `
   + `ρ(50) = ${r50}, ρ(100) = ${r100}; AR(1) with φ = ${r1}: ρ(10) = ${a10}, ρ(100) = ${a100}. Sample ACF (n = 5000) at lag 1: ${s1}; on average over lags 1–50 it is `
   + (num(gap) >= 0 ? `${gap} below the theoretical one.` : `${neg(gap)} above the theoretical one.`),
 af: (items, x) => `Paths of 1000 values, sample ACF from 20000 values${x ? X : ''}. ` + items.map(([d, r1, r10, r50, s1, sd, st]) => `d = ${d}: ρ(1) = d/(1 − d) = ${r1}, `
   + `ρ(10) = ${r10}, ρ(50) = ${r50} (sample ACF at lag 1: ${s1}); standard deviation of the path ${sd}, theoretical ${st}`).join('; ') + '.',
 rd: (s, n, K, n1, nK, rs, rsi, dfa, dfai, rsa, dfaa, x) => `${s}, ${n} returns, ${K} block sizes from ${n1} to ${nK} days${x ? X : ''}. Returns: R/S slope ${rs} `
   + `(i.i.d. series of the same length: ${rsi}), DFA ${dfa} (i.i.d.: ${dfai}). Absolute returns: R/S ${rsa}, DFA ${dfaa}.`,
 gp: (s, p, m, d, se, lo, hi, da, Ha, x) => `${s}, m = ⌊T^${p}⌋ = ${m} frequencies${x ? X : ''}: returns d̂ = ${d} (SE ${se}; d̂ ± 1.96 SE = [${lo}, ${hi}], `
   + (num(lo) < 0 && num(hi) > 0 ? 'contains 0' : 'does not contain 0') + `); absolute returns d̂ = ${da}, Ĥ = d̂ + 0.5 = ${Ha}.`,
 lo: (n, c5, m5, q5, c7, m7, c0, m0, m8, x) => `500 series of ${n} values${x ? X : ''}. AR(1) with φ = 0.5 (short memory, H₀ true): the classical test rejects in ${c5} of the cases, `
   + `the modified test in ${m5} (mean q ${q5}); with φ = 0: ${c0} and ${m0}. fGn with H = 0.7: ${c7} and ${m7}; with H = 0.8 the modified test rejects in ${m8}.`,
 mc: (lv, r250, r1000, r5000, d1000, g1000, wr, wd, wg, wd250, lo, hi, x) => `500 i.i.d. Normal series for each N, ${lv}% band${x ? X : ''}. R/S: mean ${r250} at N = 250, `
   + `${r1000} at N = 1000, ${r5000} at N = 5000; DFA ${d1000} and GPH ${g1000} at N = 1000. Width of the band at N = 1000: R/S ${wr} (from ${lo} to ${hi}), DFA ${wd}, GPH ${wg}; `
   + `at N = 250, DFA ${wd250}.`,
 nl: (b, mp, mq, dr, H1, d1, H2, d2, lo, hi, nm, x) => `Break after ${b}${x ? X : ''}: mean flow ${mp} before, ${mq} after (in 10⁸ m³), `
   + (num(dr) >= 0 ? `a fall of ${dr}%` : `a rise of ${neg(dr)}%`) + `. Raw series: Ĥ_R/S = ${H1}, DFA ${d1}; after removing the two regime means: ${H2} and ${d2}. `
   + `The i.i.d. band for R/S with N = 100: [${lo}, ${hi}], centred at ${nm}.`,
 su: (n, g5, d5, r5, r0, a95, a99, rlo, rhi, x) => `300 series of ${n} values${x ? X : ''}. A mean shift of 0.5 standard deviations in the middle: GPH Ĥ = ${g5}, DFA ${d5}, `
   + `R/S ${r5} (without a shift R/S already gives ${r0}: the small-sample bias). GARCH(1,1): DFA of |r| ${a95} at α + β = 0.95 and ${a99} at 0.99; for the returns, DFA ` + (rlo === rhi ? `${rlo} at every persistence.` : `between ${rlo} and ${rhi}.`),
 va: (s, L, band, a1, a50, a100, np, s1, s50, r1, mx, no, ex, top, x) => `${s}, the first ${L} lags${x ? X : ''}; 95% band: ±${band}. |r|: ρ(1) = ${a1}, ρ(50) = ${a50}, `
   + `ρ(100) = ${a100}; ${np} of the ${L} lags are above the band. r²: ρ(1) = ${s1}, ρ(50) = ${s50}. r: ρ(1) = ${r1}, then |ρ(k)| ≤ ${mx}; ${no} of the lags 2–${L} `
   + `are outside the band (about ${ex} expected by chance), the largest at lags ${top}.`,
 mk: (e, amin, amax, smin, smax, ins, above, x) => `The ${e} estimator${x ? X : ''}: for |r|, between ${amin} and ${amax}; after a random shuffle, between ${smin} and ${smax}, `
   + `${ins} of 6 inside the bands: the memory lies in the order of the days. Returns above the band: ${lst(above)}.`,
 ro: (W, lo, hi, items, x) => `Windows of ${W} observations moved by 21${x ? X : ''}; 95% i.i.d. band: [${lo}, ${hi}]. ` + items.map(([s, n, d0, mn, mx, dmn, dmx, ab, be, t1, t3, amn, amx]) =>
   `${s}: ${n} windows, the first ending on ${d0}; Ĥ of the returns between ${mn} (${dmn}) and ${mx} (${dmx}); ${ab}% of the windows above the band, ${be}% below; mean of the first third `
   + `${t1}, of the last ${t3}; |r| between ${amn} and ${amx}`).join('. ') + '.',
 fd1: 'Similarity dimension D = ln N/ln s: the segment ln 2/ln 2 = 1, the square ln 4/ln 2 = 2, the Koch curve ln 4/ln 3 = 1.262, the Sierpinski triangle ln 3/ln 2 = 1.585.',
 fd2: (H, n) => `Extra computation, not in the lecture: the fBm path with H = ${H} (${n} increments, seed 2026) is rescaled to the unit square; for ε = 1/2^k we split the square `
   + 'into columns of width ε and count in each column the boxes of side ε that the path touches.',
 fd3: list => 'Number of boxes N(ε): ' + list.map(([k, m, n]) => `ε = 1/${m}: ${n}`).join('; ') + '.',
 fd4: (km, D, th, df) => `OLS slope of log₂ N(ε) on log₂(1/ε), for ε from 1/2 to 1/${2 ** km}: D = ${D}; for fBm, D = 2 − H = ${th} (difference ${df}).`,
 fd: (H, km, D, th) => `Extra computation (box counting on the fBm paths of the lecture): H = ${H}, ε down to 1/${2 ** km}: D = ${D}, against 2 − H = ${th}.`,
 rh1: (s, y, N, K, n1, nK) => `${s}, ${yl(y)}: N = ${N}; ${K} block sizes between ${n1} and ${nK} ≈ N/4, evenly spaced on a log scale.`,
 rh2: (n, m, rest, rs, lx, ly) => `n = ${n}: ⌊N/n⌋ = ${m} non-overlapping blocks (` + (rest === 0 ? 'all returns fall into blocks' : rest === 1 ? 'the last return is left out'
   : `the last ${rest} returns are left out`) + `); average R/S over the blocks: (R/S)_n = ${rs}; u = log₁₀ n = ${lx}, v = log₁₀ (R/S)_n = ${ly}.`,
 rh3: (mu, mv, sxx, sxy, b, a) => `Over all points of the plot: ū = ${mu}, v̄ = ${mv}, Σ(u − ū)² = ${sxx}, Σ(u − ū)(v − v̄) = ${sxy}; Ĥ = ${sxy}/${sxx} = ${b}; log₁₀ c = v̄ − Ĥū = ${a}.`,
 rh4: (N, lo, hi, st, y) => `The 95% band of Ĥ_R/S for i.i.d. Normal series with N = ${N}: [${lo}, ${hi}]; ` + (st > 0 ? 'Ĥ is above the band' + (y === 'abs' ? ': long memory in volatility.' : '.')
   : st < 0 ? 'Ĥ is below the band.' : 'Ĥ lies inside the band: it is not separated from series without memory.'),
 rh: (s, y, H, lo, hi, x) => `${s}, ${yl(y)}${x ? X : ''}: Ĥ_R/S = ${H}; 95% i.i.d. band: [${lo}, ${hi}].`,
 mw1: (H, e, r1) => `H = ${H}: ρ(1) = 2^(2H − 1) − 1 = 2^${e} − 1 = ${r1}.`,
 mw2: (r10, r50, r100, k, rk, ap) => `fGn: ρ(10) = ${r10}, ρ(50) = ${r50}, ρ(100) = ${r100}; at lag ${k}: ρ = ${rk}, and the approximation H(2H − 1)k^(2H − 2) gives ${ap}.`,
 mw3: (phi, a10, k, ak) => `AR(1) with the same ρ(1), φ = ${phi}: ρ(10) = φ¹⁰ = ${a10}; ρ(${k}) = φ^${k} = ${ak}.`,
 mw4: (k, rk, ak) => `After ${k} lags a shock still matters in the fGn with autocorrelation ${rk}, and practically not at all in the AR(1) (${ak}): the same short-run behaviour, a very different long run.`,
 mw: (H, r1, r100, a100, x) => `H = ${H}${x ? X : ''}: ρ(1) = ${r1}; ρ(100) = ${r100} for the fGn and ${a100} for the AR(1) with the same ρ(1).`,
 aw1: (d, p1, p2) => `Weights of (1 − L)^d, d = ${d}: π₁ = −d = ${p1}, π₂ = π₁(1 − d)/2 = ${p2}.`,
 aw2: (s1, s2) => `MA weights: ψ₁ = d = ${s1}, ψ₂ = ψ₁(1 + d)/2 = ${s2}.`,
 aw3: (d, d1, r1, dp, dm, r2, r10, r100) => `ρ(1) = d/(1 − d) = ${d}/${d1} = ${r1}; ρ(2) = ρ(1) × ${dp}/${dm} = ${r2}; ρ(10) = ${r10}, ρ(100) = ${r100}.`,
 aw4: (r1, a10) => `AR(1) with φ = ${r1}: ρ(10) = φ¹⁰ = ${a10}.`,
 aw5: (t, kA, kF, H) => `The autocorrelation falls below ${t} after ${kA} lags for the AR(1) and after ${kF} lags for the ARFIMA; H = d + 0.5 = ${H}.`,
 aw: (d, r1, r10, r100, t, kA, kF, x) => `d = ${d}${x ? X : ''}: ρ(1) = ${r1}, ρ(10) = ${r10}, ρ(100) = ${r100}; below ${t} after ${kA} lags (AR(1)) and after ${kF} lags (ARFIMA).`,
 dw1: (s, y, N, m, Y1, Y2, YN) => `${s}, ${yl(y)}: N = ${N}, mean x̄ = ${m}; the profile Y_k = Σ_{j≤k}(x_j − x̄): Y₁ = ${Y1}, Y₂ = ${Y2}, …, Y_N = ${YN}.`,
 dw2: (n, m, rest, a, b, r) => `n = ${n}: ${m} blocks (` + (rest === 0 ? 'the whole profile falls into blocks' : rest === 1 ? 'the last value of the profile is left out'
   : `the last ${rest} values of the profile are left out`) + `). First block: the OLS line has the value ${a} at k = 1 and the slope ${b}; root mean square of the deviations from it: ${r}.`,
 dw3: (m, n, F) => `F(${n}) = root mean square of the deviations over all ${m} blocks of ${n} values: ${F}.`,
 dw4: (K, a, lo, hi, y) => `Slope of log F(n) on log n over the ${K} block sizes: α̂ = ${a}; the 95% i.i.d. band of DFA for the same N: [${lo}, ${hi}]`
   + (y === 'abs' ? '; for |r|, an α̂ far above the band points to long memory in volatility.' : '.'),
 dw: (s, y, n, F, a, x) => `${s}, ${yl(y)}${x ? X : ''}: F(${n}) = ${F}; DFA exponent α̂ = ${a}.`,
 gw1: (s, y, T, p, m, l1, lm) => `${s}, ${yl(y)}: T = ${T}, m = ⌊T^${p}⌋ = ${m}; the frequencies λ_j = 2πj/T go from λ₁ = ${l1} to λ_m = ${lm}.`,
 gw2: (I1, X1, y1) => `j = 1: I(λ₁) = ${I1}; the regressor −ln(4 sin²(λ₁/2)) = ${X1}; ln I(λ₁) = ${y1}.`,
 gw3: (mx, my, sxx, sxy, d) => `Over the m points: mean of the regressor ${mx}, mean of ln I ${my}, Σ(X − X̄)² = ${sxx}, Σ(X − X̄)(y − ȳ) = ${sxy}; d̂ = ${sxy}/${sxx} = ${d}.`,
 gw4: (m, se, lo, hi, H) => `SE = π/√(24m) = π/√(24 × ${m}) = ${se}; the 95% interval: d̂ ± 1.96 SE = [${lo}, ${hi}]` + (num(lo) < 0 && num(hi) > 0 ? ', which contains 0' : ', which does not contain 0') + `; Ĥ = d̂ + 0.5 = ${H}.`,
 gw: (s, p, m, dr, da, se, x) => `${s}, m = ⌊T^${p}⌋ = ${m}${x ? X : ''}: d̂ = ${dr} for the returns and ${da} for the absolute returns, SE ${se}. A larger m lowers the SE but lets the short-run dynamics in.`,
 lw1: (s, y, N, R, g0, V0) => `${s}, ${yl(y)}: N = ${N}; range of the cumulative deviations R_N = ${R}; γ̂₀ = ${g0}; classical R/S: V_N(0) = R_N/√(Nγ̂₀) = ${V0}.`,
 lw2: (rho, A, B, AB, q) => `ρ̂₁ = ${rho}; (3N/2)^(1/3) = ${A}; (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = ${B}; the product is ${AB}, so q = ${q}.`,
 lw3: (q, g0, gs, s2) => (q === 0 ? `With q = 0, σ̂²(0) = γ̂₀ = ${g0}.` : `σ̂²(${q}) = γ̂₀ + 2 Σ (1 − j/${q + 1}) γ̂_j, j = 1, …, ${q}, with ` + gs.map((v, i) => `γ̂${sub(i + 1)} = ${v}`).join(', ')
   + (q > gs.length ? ', …' : '') + `: σ̂²(${q}) = ${s2}.`),
 lw4: (R, N, s2, V, rej) => `V_N(q) = ${R}/√(${N} × ${s2}) = ${V}: ` + (rej ? 'outside [0.809, 1.862], so short memory is rejected at 5%.' : 'inside [0.809, 1.862], so short memory is not rejected at 5%.'),
 lw: (s, Vr, qr, rr, Va, qa, ra, list, x) => `${s}${x ? X : ''}: returns V = ${Vr} (q = ${qr}), short memory ${rr ? 'rejected' : 'not rejected'}; absolute returns V = ${Va} (q = ${qa}), `
   + `${ra ? 'rejected' : 'not rejected'}. Among the six series, short memory of the returns is rejected for: ${lst(list)}.`,
 st: (s, n, d0, outR, lr, nA, la, x) => `${s}: N = ${n} daily returns, from ${d0} to 18 September 2026${x ? X : ''}. Returns: ` + (outR.length ? `outside the band: ${lst(outR)}` : 'every estimate lies inside its band')
   + `; Lo's test ${lr ? 'rejects' : 'does not reject'} short memory. Absolute returns: ${nA} of 3 estimates outside the band; Lo's test ${la ? 'rejects' : 'does not reject'} short memory.`,
 mt: (p, m0, m1, ex, dmn, dmx, nr, x) => `GPH bandwidth m = ⌊T^${p}⌋, between ${m0} and ${m1} frequencies${x ? X : ''}. Returns: d̂ ± 1.96 SE excludes 0 for: ${lst(ex)}. `
   + `Absolute returns: d̂ between ${dmn} and ${dmx}; Lo's test rejects short memory for ${nr} of 6 series.`,
 rk1: (z, s, v) => `One-day VaR 1%, Normal case with mean 0: −z₀.₀₁ σ = ${z} × ${s} = ${v}%.`,
 rk2: (h, H, f, v) => `h = ${h}, H = ${H}: the factor h^H = ${h}^${H} = ${f}; VaR 1% over ${days(h)} = ${v}%.`,
 rk3: (h, sq, vs, p) => `The √h rule: √${h} = ${sq}, VaR = ${vs}%; with the exponent H the risk is ${neg(p)}% ${num(p) >= 0 ? 'higher' : 'lower'}.`,
 rk4: (sp, bt, s10) => `The data: the ratio of the 10-day to the one-day standard deviation is ${sp} for the S&P 500 and ${bt} for the BET, against √10 = ${s10}.`,
 rk: (H, h, f, v, vs, p, p250, x) => `H = ${H}, h = ${h}${x ? X : ''}: factor ${f}, VaR 1% = ${v}%, against ${vs}% with the √h rule; difference: ${p}% at ${days(h)}, ${p250}% at 250 days.`
};
})();
""",
}

# question for the room and answer of every block (RO, EN)
QA = {
 'ro': {
  'scaling': ('S&P 500: raportul dintre abaterea standard a randamentelor pe 10 zile și cea zilnică este peste sau sub √10 = 3,16?',
              'Sub: 2,80. Randamentele zilnice ale S&P 500 au autocorelația de ordinul 1 negativă, −0,099, deci zilele succesive se compensează parțial (ușoară revenire la medie). '
              'BET are 3,50, peste √10 (autocorelație pozitivă), iar Bitcoin 3,20, aproape de √10.'),
  'fractal-dim': ('Numărăm cutiile pe traiectoria fBm cu H = 0,5 (calcul suplimentar, care nu este în curs). Ce dimensiune iese?',
                  'Circa 1,5: panta este 1,47 cu ε de la 1/2 la 1/128, aproape de D = 2 − H = 1,5. Traiectoria este o curbă, dar atît de neregulată încît numărul de cutii crește ca ε^(−1,5), '
                  'nu ca ε^(−1). Pentru H = 0,1 numărarea dă doar 1,74 în loc de 1,9: cu 1000 de puncte, neregularitatea de sub pasul de timp nu se vede, deci D este subestimat la traiectoriile foarte neregulate.'),
  'rs-example': ('▶ trece la ferestre de 8 zile ale S&P 500. În 9–18 martie 2020 abaterea standard este 7,56%, de circa 15 ori mai mare decît în exemplul din curs (0,489). Cum se compară (R/S)₈ cu 1,788?',
                 'Este de același ordin: 1,530. R și S cresc împreună cu volatilitatea, deci raportul nu are unitate de măsură; pentru 8 valori R/S rămîne între circa 1,5 și 2,8 '
                 '(2,317 în ianuarie 2000, 2,774 în octombrie 2008, 2,155 în septembrie 2026). Împărțirea la S face statistica comparabilă între active.'),
  'rs-hurst': ('Panta OLS a lui log₁₀ (R/S)_n pentru randamentele S&P 500 este 0,530. Ce devine ea pentru randamentele absolute |r|?',
               'Circa 0,86 (0,861), mult peste banda i.i.d. [0,51; 0,57]: memorie lungă a volatilității. Pentru randamente, 0,530 se află în bandă, deoarece R/S este deplasat în sus în eșantioane mici.'),
  'memory-we': ('▶ crește H al fGn de la 0,55 la 0,95. Cum se schimbă ρ(100)?',
                'Crește de la circa 0,001 (H = 0,55) la circa 0,54 (H = 0,95); la H = 0,8 este 0,076. Pentru H mare, ρ(k) scade ca k^(2H − 2), adică foarte încet. AR(1) cu același ρ(1) '
                'are ρ(100) practic 0 pentru orice H: scăderea lui este geometrică.'),
  'fbm': ('Ce autocorelație de ordinul 1 au creșterile traiectoriei cu H = 0,7?',
          'Teoretic 2^(2H − 1) − 1 = 2^0,4 − 1 = 0,32; pe cele 1000 de creșteri simulate, 0,32 (0,316). Pentru H = 0,3: −0,23 pe date, −0,24 teoretic. H nu este o autocorelație și nici o probabilitate.'),
  'fgn': ('Pe slide, ACF de selecție a fGn cu H = 0,8 este sub cea teoretică. De ce?',
          'Estimarea mediei elimină o parte din componenta lentă a seriei: cu memorie lungă, media de selecție urmează valurile lungi, iar abaterile de la ea sînt mai puțin corelate. '
          'Pe lagurile 1–50, ACF de selecție este în medie cu 0,040 sub cea teoretică. Simularea prin scufundare circulantă este exactă, iar formula ACF a fGn este exactă la orice lag.'),
  'arfima': ('ARFIMA(0,d,0) cu d = 0,4: ρ(1) = 0,667. Cît este ρ(50)?',
             '0,307: scăderea hiperbolică păstrează mult din ρ(1), de aceea seria face valuri lente în jurul mediei. Un AR(1) cu φ = 0,667 ar avea ρ(50) = 0,667⁵⁰, practic 0. Cu d = 0,1, ρ(50) este doar 0,005.'),
  'arfima-we': ('ARFIMA(0; 0,3; 0): după cîte laguri scade autocorelația sub 0,05?',
                'După 222 de laguri; un AR(1) cu același ρ(1) = 0,429 scade sub 0,05 după 4 laguri. ρ(k) tinde totuși la 0 (staționaritate, d &lt; 0,5), doar foarte încet: ρ(100) = 0,069.'),
  'dfa-we': ('Cum se schimbă F(n) al randamentelor S&P 500 cînd n crește de la 10 la 1678?',
             'Crește de circa 13 ori (de la 0,935 la 12,37), aproape cît √(1678/10) = 13,0, deoarece α̂ = 0,49 ≈ 0,5. Pentru |r| crește de circa 100 de ori: α̂ = 0,95. '
             'F(n) crește proporțional cu n doar pentru un profil cu tendință (fBm are α = H + 1).'),
  'rs-dfa': ('Randamentele S&P 500 au panta R/S 0,53. Este o dovadă de H > 0,5?',
             'Nu: seriile i.i.d. Normale de aceeași lungime dau în medie panta 0,55, pentru că R/S este deplasat în sus în eșantioane mici. DFA dă 0,49, față de 0,50 pentru seriile i.i.d. '
             'Randamentele absolute au R/S 0,86 și DFA 0,95: memoria lungă este în volatilitate.'),
  'gph-we': ('Randamentele S&P 500: d̂ = 0,036, SE = π/√(24 × 81) = 0,071. Ce lățime are intervalul de 95% și conține el 0?',
             'Circa 0,28, de la −0,103 la 0,176, și conține 0: estimarea nu se separă de 0 prin acest test. Asta nu dovedește că d = 0, doar că datele sînt compatibile cu d = 0. '
             'Pentru |r|, intervalul este [0,33; 0,61].'),
  'gph': ('Lățimea de bandă trece de la m = 81 (T^0,5) la m = 477 (T^0,7). Ce se întîmplă cu d̂ pentru |r| și cu SE?',
          'SE scade de la 0,071 la 0,029, iar d̂ crește de la 0,470 la 0,554: frecvențele mai mari aduc mai multă informație, dar și dinamica de termen scurt a volatilității. '
          'La m = 33 (T^0,4), d̂ = 0,364 cu SE 0,112.'),
  'lo-we': ('Randamentele S&P 500: ρ̂₁ = −0,099 și N = 6 714. Ce q dă regula lui Andrews?',
            '7: (3N/2)^(1/3) = 21,60 și (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = 0,341, produsul 7,36. Corecția depinde de mărimea lui ρ̂₁, nu de semnul lui. Cu q = 7, V = 1,56 se află în [0,809; 1,862]; '
            'pentru |r|, q = 15 și V = 2,66: memoria scurtă este respinsă.'),
  'lo': ('Pentru AR(1) cu φ = 0,5 (memorie scurtă, H₀ adevărată), în ce proporție respinge testul R/S clasic memoria scurtă la 5%?',
         'În 60% din cazuri; testul modificat al lui Lo, în 5%, cît este nivelul nominal. R/S clasic confundă memoria scurtă cu memoria lungă. Corecția costă putere: pentru fGn cu H = 0,7, '
         'ratele de respingere sînt 91% (clasic) și 36% (modificat).'),
  'mc': ('La N = 1000, care estimator are cea mai largă bandă de 95% sub H = 0,5?',
         'GPH: lățimea 0,53, față de 0,16 pentru DFA și 0,12 pentru R/S. GPH este practic nedeplasat (media 0,498), dar folosește doar m = 31 de frecvențe; R/S are banda cea mai îngustă, '
         'dar centrată la 0,562, nu la 0,5.'),
  'sp-table': ('Pentru randamentele S&P 500, doar un estimator ar părea să indice H > 0,5. Care și de ce nu este o dovadă?',
               'R/S, cu 0,53: banda lui i.i.d. pentru N = 6 714 este [0,51; 0,57], deci 0,53 se află în ea. DFA (0,49), GPH (Ĥ = 0,54, d̂ ± 1,96 SE = [−0,10; 0,18]) și testul lui Lo '
               '(V = 1,56) sînt toate compatibile cu absența memoriei lungi.'),
  'nile': ('După eliminarea celor două medii de regim (ruptura din 1898), ce devine Ĥ_R/S al Nilului (0,78 pe seria brută)?',
           '0,63, în banda i.i.d. [0,45; 0,75] pentru N = 100, centrată în 0,60; DFA scade de la 0,84 la 0,50. Cea mai mare parte a memoriei aparente era ruptura, nu persistența debitelor.'),
  'spurious': ('O serie i.i.d. are o schimbare a mediei de 0,5 abateri standard la mijloc. Ce Ĥ dă GPH, în medie?',
               'Circa 0,79 (DFA 0,56), deși seria nu are nicio memorie. Abaterea comună de la media generală face ca observațiile din același regim să pară corelate la toate lagurile. '
               'La N = 4000, GPH dă tot 0,79: memoria aparentă nu dispare cu mai multe date.'),
  'vol-acf': ('Cîte dintre lagurile 2–250 ale ACF a randamentelor S&P 500 sînt în afara benzii de 95%?',
              '31, față de circa 12 așteptate din întîmplare (5% din 249); cele mai mari la lagurile 15, 16 și 34, toate sub 0,061 în valoare absolută. Banda 1,96/√n este o referință '
              'asimptotică pentru o serie i.i.d.; cu volatility clustering, autocorelațiile de selecție variază mai mult decît arată ea, deci depășirile nu dovedesc, singure, predictibilitatea.'),
  'mk-table': ('Pentru randamentele căror serii exclude intervalul d̂ ± 1,96 SE valoarea 0?',
               'Doar pentru BET: d̂ = 0,17, intervalul [0,03; 0,31]. Tot la BET, DFA (0,57) este peste bandă și testul lui Lo (V = 1,96) respinge memoria scurtă. Pentru |r|, d̂ este între 0,21 și 0,48, '
               'iar testul lui Lo respinge pe toate cele șase serii.'),
  'markets': ('După o permutare aleatoare a lui |r|, ce devine exponentul DFA (între 0,73 și 0,95 înainte)?',
              'Scade între 0,46 și 0,55, în interiorul benzilor i.i.d.: permutarea păstrează distribuția (deci și cozile groase) și distruge doar ordinea zilelor. Memoria lungă a volatilității '
              'se află în ordinea zilelor.'),
  'rolling': ('S&P 500, ferestre de 1000 de zile: ce pondere din ferestre are exponentul DFA al randamentelor sub banda i.i.d.?',
              '24%, niciuna peste. Minimul, 0,32, este atins la 31 decembrie 2008: inversări zilnice (antipersistență) în panică. Ferestrele consecutive au în comun 979 din 1000 de observații, '
              'deci 24% nu înseamnă 65 de rezultate independente.'),
  'risk-we': ('Cu H = 0,55 în loc de 0,5, cu cît crește VaR 1% pe 10 zile, la o volatilitate zilnică de 1,2%?',
              'Cu circa 12%: 9,91% față de 8,83%, pentru că 10^0,55/10^0,5 = 10^0,05 = 1,12. La 250 de zile diferența este 32%. Regula √h supraestimează riscul S&P 500 (raportul pe 10 zile 2,80) '
              'și îl subestimează pe cel al BET (3,50).'),
 },
 'en': {
  'scaling': ('S&P 500: is the ratio of the 10-day to the daily standard deviation of returns above or below √10 = 3.16?',
              'Below: 2.80. The daily S&P 500 returns have a negative first-order autocorrelation, −0.099, so consecutive days partly offset each other (mild mean reversion). '
              'The BET has 3.50, above √10 (positive autocorrelation), and Bitcoin 3.20, close to √10.'),
  'fractal-dim': ('We count boxes on the fBm path with H = 0.5 (extra computation, not in the lecture). Which dimension comes out?',
                  'About 1.5: the slope is 1.47 with ε from 1/2 to 1/128, close to D = 2 − H = 1.5. The path is a curve, but so rough that the number of boxes grows like ε^(−1.5), '
                  'not like ε^(−1). For H = 0.1 the count gives only 1.74 instead of 1.9: with 1000 points the roughness below the time step is invisible, so D is underestimated for very rough paths.'),
  'rs-example': ('▶ moves to 8-day windows of the S&P 500. On 9–18 March 2020 the standard deviation is 7.56%, about 15 times that of the lecture example (0.489). How does (R/S)₈ compare with 1.788?',
                 'It is of the same order: 1.530. R and S grow together with volatility, so the ratio has no unit; for 8 values R/S stays between about 1.5 and 2.8 '
                 '(2.317 in January 2000, 2.774 in October 2008, 2.155 in September 2026). Dividing by S makes the statistic comparable across assets.'),
  'rs-hurst': ('The OLS slope of log₁₀ (R/S)_n for the S&P 500 returns is 0.530. What does it become for the absolute returns |r|?',
               'About 0.86 (0.861), far above the i.i.d. band [0.51, 0.57]: long memory in volatility. For the returns, 0.530 lies inside the band, because R/S is biased upwards in small samples.'),
  'memory-we': ('▶ raises H of the fGn from 0.55 to 0.95. How does ρ(100) change?',
                'It rises from about 0.001 (H = 0.55) to about 0.54 (H = 0.95); at H = 0.8 it is 0.076. For large H, ρ(k) decays like k^(2H − 2), that is very slowly. An AR(1) with the same ρ(1) '
                'has ρ(100) practically 0 for every H: its decay is geometric.'),
  'fbm': ('What is the first-order autocorrelation of the increments of the path with H = 0.7?',
          'In theory 2^(2H − 1) − 1 = 2^0.4 − 1 = 0.32; on the 1000 simulated increments, 0.32 (0.316). For H = 0.3: −0.23 in the data, −0.24 in theory. H is neither an autocorrelation nor a probability.'),
  'fgn': ('On the slide, the sample ACF of the fGn with H = 0.8 lies below the theoretical one. Why?',
          'Estimating the mean removes part of the slow component of the series: with long memory the sample mean follows the long waves, and the deviations from it are less correlated. '
          'Over lags 1–50 the sample ACF is on average 0.040 below the theoretical one. The circulant-embedding simulation is exact, and the fGn ACF formula is exact at every lag.'),
  'arfima': ('ARFIMA(0,d,0) with d = 0.4: ρ(1) = 0.667. What is ρ(50)?',
             '0.307: the hyperbolic decay keeps much of ρ(1), which is why the series makes slow waves around its mean. An AR(1) with φ = 0.667 would have ρ(50) = 0.667⁵⁰, practically 0. With d = 0.1, ρ(50) is only 0.005.'),
  'arfima-we': ('ARFIMA(0, 0.3, 0): after how many lags does the autocorrelation fall below 0.05?',
                'After 222 lags; an AR(1) with the same ρ(1) = 0.429 falls below 0.05 after 4 lags. ρ(k) still goes to 0 (stationarity, d &lt; 0.5), only very slowly: ρ(100) = 0.069.'),
  'dfa-we': ('How does F(n) of the S&P 500 returns change as n grows from 10 to 1678?',
             'It grows about 13 times (from 0.935 to 12.37), close to √(1678/10) = 13.0, because α̂ = 0.49 ≈ 0.5. For |r| it grows about 100 times: α̂ = 0.95. '
             'F(n) grows in proportion to n only for a profile with a trend (fBm has α = H + 1).'),
  'rs-dfa': ('The S&P 500 returns have an R/S slope of 0.53. Is this evidence of H > 0.5?',
             'No: i.i.d. Normal series of the same length give a mean slope of 0.55, because R/S is biased upwards in small samples. DFA gives 0.49, against 0.50 for the i.i.d. series. '
             'The absolute returns have R/S 0.86 and DFA 0.95: the long memory is in volatility.'),
  'gph-we': ('S&P 500 returns: d̂ = 0.036, SE = π/√(24 × 81) = 0.071. How wide is the 95% interval, and does it contain 0?',
             'About 0.28, from −0.103 to 0.176, and it contains 0: the estimate is not separated from 0 by this test. This does not prove d = 0, only that the data are compatible with d = 0. '
             'For |r| the interval is [0.33, 0.61].'),
  'gph': ('The bandwidth moves from m = 81 (T^0.5) to m = 477 (T^0.7). What happens to d̂ for |r| and to the SE?',
          'The SE falls from 0.071 to 0.029 and d̂ rises from 0.470 to 0.554: the higher frequencies bring more information, but also the short-run dynamics of volatility. '
          'At m = 33 (T^0.4), d̂ = 0.364 with SE 0.112.'),
  'lo-we': ('S&P 500 returns: ρ̂₁ = −0.099 and N = 6,714. Which q does the Andrews rule give?',
            '7: (3N/2)^(1/3) = 21.60 and (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = 0.341, product 7.36. The correction depends on the size of ρ̂₁, not on its sign. With q = 7, V = 1.56 lies in [0.809, 1.862]; '
            'for |r|, q = 15 and V = 2.66: short memory is rejected.'),
  'lo': ('For an AR(1) with φ = 0.5 (short memory, H₀ true), in what share of the cases does the classical R/S test reject short memory at 5%?',
         'In 60% of the cases; Lo\'s modified test in 5%, the nominal level. The classical R/S confuses short memory with long memory. The correction costs power: for fGn with H = 0.7, '
         'the rejection rates are 91% (classical) and 36% (modified).'),
  'mc': ('At N = 1000, which estimator has the widest 95% band under H = 0.5?',
         'GPH: width 0.53, against 0.16 for DFA and 0.12 for R/S. GPH is practically unbiased (mean 0.498) but uses only m = 31 frequencies; R/S has the narrowest band, '
         'but centred at 0.562, not at 0.5.'),
  'sp-table': ('For the S&P 500 returns, only one estimator would seem to point to H > 0.5. Which one, and why is it not evidence?',
               'R/S, with 0.53: its i.i.d. band for N = 6,714 is [0.51, 0.57], so 0.53 lies inside it. DFA (0.49), GPH (Ĥ = 0.54, d̂ ± 1.96 SE = [−0.10, 0.18]) and Lo\'s test '
               '(V = 1.56) are all compatible with no long memory.'),
  'nile': ('After removing the two regime means (the 1898 break), what becomes of the Nile\'s Ĥ_R/S (0.78 on the raw series)?',
           '0.63, inside the i.i.d. band [0.45, 0.75] for N = 100, centred at 0.60; DFA falls from 0.84 to 0.50. Most of the apparent memory was the break, not persistence of the flows.'),
  'spurious': ('An i.i.d. series has a mean shift of 0.5 standard deviations in the middle. Which Ĥ does GPH give, on average?',
               'About 0.79 (DFA 0.56), although the series has no memory at all. The common deviation from the overall mean makes observations of the same regime look correlated at all lags. '
               'At N = 4000, GPH still gives 0.79: spurious memory does not go away with more data.'),
  'vol-acf': ('How many of the lags 2–250 of the ACF of the S&P 500 returns lie outside the 95% band?',
              '31, against about 12 expected by chance (5% of 249); the largest at lags 15, 16 and 34, all below 0.061 in absolute value. The 1.96/√n band is an asymptotic reference '
              'for an i.i.d. series; with volatility clustering the sample autocorrelations vary more than it shows, so the exceedances do not, by themselves, prove predictability.'),
  'mk-table': ('For the returns of which series does the interval d̂ ± 1.96 SE exclude 0?',
               'Only for the BET: d̂ = 0.17, interval [0.03, 0.31]. Also for the BET, DFA (0.57) is above its band and Lo\'s test (V = 1.96) rejects short memory. For |r|, d̂ lies between 0.21 and 0.48, '
               'and Lo\'s test rejects for all six series.'),
  'markets': ('After a random shuffle of |r|, what becomes of the DFA exponent (between 0.73 and 0.95 before)?',
              'It falls to between 0.46 and 0.55, inside the i.i.d. bands: the shuffle keeps the distribution (heavy tails included) and destroys only the order of the days. The long memory in '
              'volatility lies in the order of the days.'),
  'rolling': ('S&P 500, windows of 1000 days: what share of the windows has the DFA exponent of the returns below the i.i.d. band?',
              '24%, none above. The minimum, 0.32, is reached on 31 December 2008: daily reversals (anti-persistence) in the panic. Consecutive windows share 979 of their 1000 observations, '
              'so 24% does not mean 65 independent results.'),
  'risk-we': ('With H = 0.55 instead of 0.5, by how much does the 10-day VaR 1% rise, at a daily volatility of 1.2%?',
              'By about 12%: 9.91% against 8.83%, because 10^0.55/10^0.5 = 10^0.05 = 1.12. At 250 days the difference is 32%. The √h rule overstates the S&P 500 risk (10-day ratio 2.80) '
              'and understates the BET risk (3.50).'),
 },
}

PREDICT = {
 'scaling': (0, [('Sub: circa 2,80.', 'Below: about 2.80.'), ('Exact 3,16: regula rădăcinii pătrate este exactă pentru orice serie staționară.', 'Exactly 3.16: the square-root rule is exact for any stationary series.'),
                 ('Peste: cozile groase măresc riscul pe mai multe zile.', 'Above: heavy tails raise the multi-day risk.')]),
 'fractal-dim': (0, [('Circa 1,5 = 2 − H.', 'About 1.5 = 2 − H.'), ('1: traiectoria este o curbă, deci are dimensiunea unui segment.', '1: the path is a curve, so it has the dimension of a segment.'),
                     ('2: traiectoria umple pătratul.', '2: the path fills the square.')]),
 'rs-example': (0, [('Este de același ordin, circa 1,5.', 'It is of the same order, about 1.5.'), ('Este de circa 15 ori mai mare: R crește cu volatilitatea.', 'It is about 15 times larger: R grows with volatility.'),
                    ('Este aproape 0: în criză abaterile se compensează.', 'It is almost 0: in a crisis the deviations offset each other.')]),
 'rs-hurst': (0, [('Circa 0,86.', 'About 0.86.'), ('Tot 0,53: |r| are aceeași ordine a zilelor ca r.', 'Still 0.53: |r| has the same order of days as r.'),
                  ('Circa 0,5: valoarea absolută elimină semnul, deci și memoria.', 'About 0.5: the absolute value removes the sign, and with it the memory.')]),
 'memory-we': (0, [('Crește de la circa 0,001 la circa 0,54.', 'It rises from about 0.001 to about 0.54.'), ('Rămîne aproape 0: după 100 de laguri orice autocorelație dispare.', 'It stays near 0: after 100 lags every autocorrelation is gone.'),
                   ('Crește liniar cu H, de la 0,55 la 0,95.', 'It rises linearly with H, from 0.55 to 0.95.')]),
 'fbm': (0, [('Circa 0,32.', 'About 0.32.'), ('0,7: autocorelația creșterilor este egală cu H.', '0.7: the autocorrelation of the increments equals H.'),
             ('0: creșterile fBm sînt independente, ca la mișcarea browniană.', '0: fBm increments are independent, as for Brownian motion.')]),
 'fgn': (0, [('Media estimată absoarbe o parte din componenta lentă a seriei.', 'The estimated mean absorbs part of the slow component of the series.'),
             ('Simularea prin scufundare circulantă nu este exactă.', 'The circulant-embedding simulation is not exact.'), ('Formula ACF a fGn este valabilă doar pentru k mare.', 'The fGn ACF formula holds only for large k.')]),
 'arfima': (0, [('Circa 0,31.', 'About 0.31.'), ('0,667⁵⁰, practic 0: scădere geometrică, ca la AR(1).', '0.667⁵⁰, practically 0: geometric decay, as for an AR(1).'), ('0,4: ρ(k) = d la orice lag.', '0.4: ρ(k) = d at every lag.')]),
 'arfima-we': (0, [('După 222 de laguri.', 'After 222 lags.'), ('După 4 laguri, ca AR(1) cu același ρ(1).', 'After 4 lags, like the AR(1) with the same ρ(1).'),
                   ('Niciodată: memoria lungă înseamnă că ρ(k) nu tinde la 0.', 'Never: long memory means ρ(k) does not go to 0.')]),
 'dfa-we': (0, [('Crește de circa 13 ori, aproape ca √n.', 'It grows about 13 times, close to √n.'), ('Crește de circa 168 de ori, proporțional cu n.', 'It grows about 168 times, in proportion to n.'),
                ('Scade: în blocurile mari dreapta elimină mai mult.', 'It falls: in large blocks the line removes more.')]),
 'rs-dfa': (0, [('Nu: seriile i.i.d. de aceeași lungime dau 0,55.', 'No: i.i.d. series of the same length give 0.55.'), ('Da: orice pantă peste 0,5 înseamnă persistență.', 'Yes: any slope above 0.5 means persistence.'),
                ('Da, pentru că DFA confirmă cu 0,49.', 'Yes, because DFA confirms it with 0.49.')]),
 'gph-we': (0, [('Circa 0,28 și conține 0.', 'About 0.28, and it contains 0.'), ('Circa 0,14 (± 1 SE) și nu conține 0.', 'About 0.14 (± 1 SE), and it does not contain 0.'),
                ('Circa 0,28 și nu conține 0, pentru că d̂ = 0,036 > 0.', 'About 0.28, and it does not contain 0, because d̂ = 0.036 > 0.')]),
 'gph': (0, [('SE scade la circa 0,03, iar d̂ crește la circa 0,55.', 'The SE falls to about 0.03 and d̂ rises to about 0.55.'), ('SE rămîne 0,071: depinde doar de T.', 'The SE stays at 0.071: it depends only on T.'),
             ('d̂ rămîne 0,47: estimatorul nu depinde de m.', 'd̂ stays at 0.47: the estimator does not depend on m.')]),
 'lo-we': (0, [('7.', '7.'), ('0: ρ̂₁ este negativ, deci nu există memorie scurtă de corectat.', '0: ρ̂₁ is negative, so there is no short memory to correct.'),
               ('81, ca lățimea de bandă GPH ⌊√N⌋.', '81, like the GPH bandwidth ⌊√N⌋.')]),
 'lo': (0, [('Circa 60%.', 'About 60%.'), ('Circa 5%: nivelul nominal al testului.', 'About 5%: the nominal level of the test.'), ('Circa 50%: φ = 0,5.', 'About 50%: φ = 0.5.')]),
 'mc': (0, [('GPH: lățimea circa 0,53.', 'GPH: width about 0.53.'), ('R/S: este deplasat, deci și cel mai imprecis.', 'R/S: it is biased, so also the least precise.'),
            ('DFA: eliminarea tendințelor locale pierde informație.', 'DFA: removing local trends loses information.')]),
 'sp-table': (0, [('R/S (0,53): banda lui pentru N = 6 714 este [0,51; 0,57].', 'R/S (0.53): its band for N = 6,714 is [0.51, 0.57].'),
                  ('GPH (Ĥ = 0,54): d̂ = 0,036 este pozitiv.', 'GPH (Ĥ = 0.54): d̂ = 0.036 is positive.'), ('Testul lui Lo (V = 1,56): este peste 1.', "Lo's test (V = 1.56): it is above 1.")]),
 'nile': (0, [('Circa 0,63, în banda i.i.d. pentru N = 100.', 'About 0.63, inside the i.i.d. band for N = 100.'), ('Rămîne 0,78: memoria lungă nu depinde de medie.', 'It stays at 0.78: long memory does not depend on the mean.'),
              ('Exact 0,5: fără ruptură, seria este zgomot alb.', 'Exactly 0.5: without the break the series is white noise.')]),
 'spurious': (0, [('Circa 0,79.', 'About 0.79.'), ('0,5: seria nu are memorie, deci GPH este nedeplasat.', '0.5: the series has no memory, so GPH is unbiased.'),
                  ('Circa 0,55, cît deplasarea R/S de eșantion mic.', 'About 0.55, the size of the small-sample R/S bias.')]),
 'vol-acf': (0, [('31, față de circa 12 așteptate din întîmplare.', '31, against about 12 expected by chance.'), ('0: randamentele sînt necorelate, deci nicio autocorelație nu iese din bandă.', '0: returns are uncorrelated, so no autocorrelation leaves the band.'),
                 ('227, ca pentru |r|.', '227, as for |r|.')]),
 'mk-table': (0, [('Doar pentru BET.', 'Only for the BET.'), ('Pentru toate șase: toate au DFA peste 0,7 pentru |r|.', 'For all six: they all have DFA above 0.7 for |r|.'),
                  ('Pentru niciuna: randamentele nu au memorie lungă pe nicio piață.', 'For none: returns have no long memory in any market.')]),
 'markets': (0, [('Scade în banda i.i.d., între 0,46 și 0,55.', 'It falls into the i.i.d. band, between 0.46 and 0.55.'), ('Rămîne la fel: cozile groase ale lui |r| nu se schimbă prin permutare.', 'It stays the same: the heavy tails of |r| do not change under a shuffle.'),
                 ('Crește: permutarea adună zilele extreme.', 'It rises: the shuffle brings the extreme days together.')]),
 'rolling': (0, [('Circa 24%, niciuna peste.', 'About 24%, none above.'), ('Circa 2,5%: atît dă banda din întîmplare.', 'About 2.5%: that is what the band gives by chance.'),
                 ('Circa 24%, dar peste bandă: crizele creează persistență.', 'About 24%, but above the band: crises create persistence.')]),
 'risk-we': (0, [('Cu circa 12%.', 'By about 12%.'), ('Cu 10%: H crește cu 10%.', 'By 10%: H rises by 10%.'), ('Cu 0,05 puncte procentuale: diferența dintre exponenți.', 'By 0.05 percentage points: the difference of the exponents.')]),
}

# blocks whose initial chart, table or worked example shows the answer: veil the readout and those elements
VEIL = {'scaling': {'out': True, 'charts': ['sc-chart']}, 'fractal-dim': {'out': True, 'charts': ['fd-chart', 'fd-we-box']}, 'rs-example': {'out': True},
        'rs-hurst': {'out': True}, 'memory-we': {'out': True}, 'fbm': {'out': True}, 'fgn': {'out': True}, 'arfima': {'out': True, 'charts': ['af-chart']},
        'arfima-we': {'out': True, 'charts': ['aw-chart', 'aw-we-box']}, 'dfa-we': {'out': True, 'charts': ['dw-chart']}, 'rs-dfa': {'out': True, 'charts': ['rd-chart']},
        'gph-we': {'out': True, 'charts': ['gw-we-box']}, 'gph': {'out': True}, 'lo-we': {'out': True, 'charts': ['lw-chart', 'lw-we-box', 'lw-tab']},
        'lo': {'out': True, 'charts': ['lo-chart']}, 'mc': {'out': True, 'charts': ['mc-chart']}, 'sp-table': {'out': True, 'charts': ['st-chart', 'st-tab']},
        'nile': {'out': True, 'charts': ['nl-chart']}, 'spurious': {'out': True, 'charts': ['su-chart']}, 'vol-acf': {'out': True, 'charts': ['va-chart']},
        'mk-table': {'out': True, 'charts': ['mt-chart', 'mt-tab']}, 'markets': {'out': True, 'charts': ['mk-chart']}, 'rolling': {'out': True, 'charts': ['ro-chart']},
        'risk-we': {'out': True, 'charts': ['rk-we-box']}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('O estimare R/S pe 1000 de randamente i.i.d. este Ĥ = 0,56. Este o dovadă de memorie lungă?', 'An R/S estimate on 1000 i.i.d. returns is Ĥ = 0.56. Is it evidence of long memory?'),
      key=0, opts=[
  (('Nu: banda i.i.d. pentru N = 1000 este [0,50; 0,63], centrată în jur de 0,562.', 'No: the i.i.d. band for N = 1000 is [0.50, 0.63], centred around 0.562.'),
   ('Comparăm Ĥ cu distribuția lui sub ipoteza nulă, pentru același N; R/S este deplasat în sus în eșantioane mici.', 'We compare Ĥ with its distribution under the null, for the same N; R/S is biased upwards in small samples.')),
  (('Da: 0,56 > 0,5.', 'Yes: 0.56 > 0.5.'),
   ('Pragul 0,5 ignoră deplasarea: seriile i.i.d. cu N = 1000 dau în medie 0,562.', 'The threshold 0.5 ignores the bias: i.i.d. series with N = 1000 give 0.562 on average.')),
  (('Nu se poate spune: R/S nu are o bandă de încredere.', 'It cannot be said: R/S has no confidence band.'),
   ('Banda Monte Carlo se obține pentru orice estimator, aplicîndu-l pe serii simulate de aceeași lungime.', 'A Monte Carlo band exists for any estimator: apply it to simulated series of the same length.'))]),
 dict(q=('GPH dă d̂ = 0,35 pentru |r|. Cît este Ĥ?', 'GPH gives d̂ = 0.35 for |r|. What is Ĥ?'),
      key=0, opts=[
  (('0,85 = d̂ + 0,5: memorie lungă a volatilității.', '0.85 = d̂ + 0.5: long memory in volatility.'), ('d = H − 0,5, deci H = d + 0,5; estimarea privește |r|, deci volatilitatea, nu direcția randamentelor.', 'd = H − 0.5, so H = d + 0.5; the estimate concerns |r|, so volatility, not the direction of returns.')),
  (('0,35: GPH estimează direct H.', '0.35: GPH estimates H directly.'), ('GPH estimează parametrul de memorie d din panta log-periodogramei.', 'GPH estimates the memory parameter d from the slope of the log periodogram.')),
  (('0,15 = 0,5 − d̂: antipersistență.', '0.15 = 0.5 − d̂: anti-persistence.'), ('Semnul este inversat: d > 0 înseamnă H > 0,5.', 'The sign is reversed: d > 0 means H > 0.5.'))]),
 dict(q=('După o permutare aleatoare a lui |r|, DFA scade de la 0,9 la 0,5. Ce arată acest lucru?', 'After a random shuffle of |r|, DFA falls from 0.9 to 0.5. What does this show?'),
      key=0, opts=[
  (('Memoria provine din ordinea zilelor, nu din distribuție (cozi groase).', 'The memory comes from the order of the days, not from the distribution (heavy tails).'),
   ('Permutarea păstrează distribuția și distruge doar ordinea; exponentul a scăzut, deci ordinea îl producea.', 'The shuffle keeps the distribution and destroys only the order; the exponent fell, so the order produced it.')),
  (('Cozile groase ale lui |r| produc exponentul 0,9.', 'The heavy tails of |r| produce the exponent 0.9.'),
   ('Cozile sînt aceleași după permutare, iar exponentul a scăzut: nu ele îl produceau.', 'The tails are the same after the shuffle and the exponent fell: they did not produce it.')),
  (('Permutarea a introdus o ruptură structurală.', 'The shuffle introduced a structural break.'),
   ('O permutare aleatoare distruge rupturile și tendințele, nu le creează.', 'A random shuffle destroys breaks and trends; it does not create them.'))]),
 dict(q=('VaR 1% pe o zi este 2,79%. Cu H = 0,45, cît este VaR 1% pe 10 zile?', 'The one-day VaR 1% is 2.79%. With H = 0.45, what is the 10-day VaR 1%?'),
      key=0, opts=[
  (('7,87% = 2,79 × 10^0,45.', '7.87% = 2.79 × 10^0.45.'), ('Riscul se scalează cu h^H; 10^0,45 = 2,82.', 'Risk scales with h^H; 10^0.45 = 2.82.')),
  (('8,83%: regula √10.', '8.83%: the √10 rule.'), ('Regula √h presupune H = 0,5; cu H = 0,45 factorul este 2,82, nu 3,16.', 'The √h rule assumes H = 0.5; with H = 0.45 the factor is 2.82, not 3.16.')),
  (('12,6% = 2,79 × 10 × 0,45.', '12.6% = 2.79 × 10 × 0.45.'), ('Factorul este h la puterea H, nu h înmulțit cu H.', 'The factor is h to the power H, not h times H.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch11-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH11_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH11_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH11_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
