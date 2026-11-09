"""Build the interactive pages of Chapter 10: interactive/ch10.html (EN) and interactive/ch10_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch10.js, inputs in interactive/ch10_data.js (Quantlets/Ch_10/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch10.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 10
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_10'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('def', 'sfm_ch10_var_es_def', 'def'), ('curves', 'sfm_ch10_var_curves', 'est'), ('rolling', 'sfm_ch10_rolling_var', 'cond'), ('corr', 'sfm_ch10_rolling_corr', 'port'),
          ('copula', 'sfm_ch10_copula', 'port'), ('binom', 'sfm_ch10_binomial', 'bt'), ('hits', 'sfm_ch10_hits', 'bt'), ('traffic', 'sfm_ch10_traffic_time', 'bt')]
# table and worked-example blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page);
# the EN deck has one acronym frame less, so its frame number is one lower (checked in frame_pages)
XBLOCKS = [('normal-we', 16, 'def', 'def'), ('horizon', 18, 'def', 'def'), ('hs-we', 24, 'est', 'def'), ('precision', 25, 'est', 'def'), ('cf-we', 26, 'est', 'def'),
           ('evt-we', 27, 'est', 'def'), ('var-table', 29, 'est', 'curves'), ('es-table', 30, 'est', 'curves'), ('cond-we', 32, 'cond', 'curves'),
           ('cond-table', 33, 'cond', 'curves'), ('stress-table', 36, 'cond', 'rolling'), ('vc-we', 39, 'port', 'rolling'), ('tail-dep', 41, 'port', 'corr'),
           ('copula-we', 43, 'port', 'corr'), ('port-table', 46, 'port', 'copula'), ('kupiec-we', 49, 'bt', 'binom'), ('kupiec-sp', 50, 'bt', 'binom'),
           ('chris-we', 51, 'bt', 'binom'), ('basel', 52, 'bt', 'binom'), ('bt-sd', 54, 'bt', 'hits'), ('bt-bb', 55, 'bt', 'hits'), ('score', 56, 'bt', 'hits'),
           ('es-bt', 58, 'bt', 'traffic')]
ORDER = ['def', 'normal-we', 'horizon', 'hs-we', 'precision', 'cf-we', 'evt-we', 'curves', 'var-table', 'es-table', 'cond-we', 'cond-table', 'rolling', 'stress-table',
         'vc-we', 'corr', 'tail-dep', 'copula-we', 'copula', 'port-table', 'binom', 'kupiec-we', 'kupiec-sp', 'chris-we', 'basel', 'hits', 'bt-sd', 'bt-bb', 'score',
         'traffic', 'es-bt']
TITLES = {'normal-we': 'Exemplu rezolvat: VaR și ES Normale pentru S&P 500', 'horizon': 'De la o zi la zece zile', 'hs-we': 'Simularea istorică',
          'precision': 'Precizia unui VaR istoric', 'cf-we': 'Dezvoltarea Cornish–Fisher', 'evt-we': 'EVT: peaks over threshold (Capitolul 5)',
          'var-table': 'VaR 1% pe metode, pentru șase serii', 'es-table': 'ES 2,5% pe metode, pentru șase serii', 'cond-we': 'Exemplu rezolvat: S&P 500 la 18 septembrie 2026',
          'cond-table': 'VaR și ES pentru ziua următoare, pe șase serii', 'stress-table': 'Perioade de stres: depășiri în 2008 și 2020', 'vc-we': 'Metoda varianță–covarianță',
          'tail-dep': 'Corelația și dependența în cozi', 'copula-we': 'Copula Gaussiană și copula t', 'port-table': 'VaR 1% și ES 2,5% ale portofoliului, pe metode',
          'kupiec-we': 'Testul POF al lui Kupiec', 'kupiec-sp': 'Testul Kupiec pe 21 de ani de prognoze pentru S&P 500', 'chris-we': 'Testul de independență al lui Christoffersen',
          'basel': 'Semaforul Basel', 'bt-sd': 'Rezultatele backtesting-ului: S&P 500 și DAX', 'bt-bb': 'Rezultatele backtesting-ului: BET și Bitcoin',
          'score': 'Ordonarea prognozelor și zona de azi', 'es-bt': 'Backtesting pentru ES: Acerbi și Székely'}

PFX = {'def': 'df', 'normal-we': 'nw', 'horizon': 'hz', 'hs-we': 'hs', 'precision': 'pr', 'cf-we': 'cf', 'evt-we': 'ev', 'curves': 'cv', 'var-table': 'vt', 'es-table': 'et',
       'cond-we': 'cw', 'cond-table': 'ct', 'rolling': 'rv', 'stress-table': 'sx', 'vc-we': 'vc', 'corr': 'rc', 'tail-dep': 'td', 'copula-we': 'cq', 'copula': 'cp',
       'port-table': 'pt', 'binom': 'bn', 'kupiec-we': 'kp', 'kupiec-sp': 'ks', 'chris-we': 'cr', 'basel': 'bs', 'hits': 'ht', 'bt-sd': 'b1', 'bt-bb': 'b2', 'score': 'sc',
       'traffic': 'tt', 'es-bt': 'eb'}

SECTIONS = {
 'ro': dict(**{'def': 'VaR și ES: definiții'}, est='Estimarea VaR și ES', cond='VaR condiționat: GARCH și simularea istorică filtrată', port='VaR de portofoliu și dependența',
            bt='Backtesting'),
 'en': dict(**{'def': 'VaR and ES: Definitions'}, est='Estimating VaR and ES', cond='Conditional VaR: GARCH and Filtered Historical Simulation', port='Portfolio VaR and Dependence',
            bt='Backtesting'),
}

BCBS = {'1996a': 'https://www.bis.org/publ/bcbs24.htm', '1996b': 'https://www.bis.org/publ/bcbs22.htm', '2019': 'https://www.bis.org/bcbs/publ/d457.htm',
        '2023a': 'https://www.bis.org/basel_framework/chapter/MAR/32.htm', '2023b': 'https://www.bis.org/basel_framework/chapter/MAR/33.htm'}


def bc(k):
    return f'<a href="{BCBS[k]}" target="_blank" rel="noopener">BCBS ({k})</a>'


L = {
 'ro': dict(file='ch10_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 10 · VaR, ES și backtesting', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: VaR (value at risk, valoarea expusă la risc), ES (expected shortfall, pierderea așteptată în coadă), '
        'HS (historical simulation, simularea istorică), EVT (extreme value theory, teoria valorilor extreme), POT (peaks over threshold, metoda vîrfurilor peste prag), '
        'GPD (generalised Pareto distribution, distribuția Pareto generalizată), GARCH (generalised autoregressive conditional heteroskedasticity, heteroscedasticitate condiționată '
        'autoregresivă generalizată), FHS (filtered historical simulation, simularea istorică filtrată), POF (proportion of failures, proporția depășirilor), LR (likelihood ratio, '
        'raportul de verosimilitate), BCBS (Basel Committee on Banking Supervision, Comitetul de la Basel pentru supraveghere bancară), FRTB (Fundamental Review of the Trading Book, '
        'revizuirea fundamentală a portofoliului de tranzacționare), MAR32 și MAR33 (capitolele 32 și 33 ale standardului Basel pentru riscul de piață), BVB (Bursa de Valori București), '
        'CLT (central limit theorem, teorema limită centrală).',
   src='Randamente logaritmice zilnice, în %: S&P 500, DAX și BET din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV) și OMV Petrom (SNP) '
       'din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. Portofoliul: randamente simple ale BET și S&P 500 pe cele '
       '6 466 de zile comune. Prețuri EODHD. Browserul calculează din randamente VaR și ES istorice, Normale, Cornish–Fisher și EVT, prognozele HS și Normale pe 500 de zile, corelațiile, '
       'pseudo-observațiile și dependența în cozi; parametrii Student-t, GPD și GARCH(1,1)-t sînt cei estimați de codul din curs. Prognozele GARCH-t și FHS pe fereastră mobilă '
       '(reestimate la fiecare 250 de zile) sînt reconstruite din volatilitatea condiționată și din parametrii fiecărei reestimări; depășirile, testele Kupiec și Christoffersen, '
       'semaforul Basel, pierderea cuantilă și Z₂ sînt recalculate în browser. Valorile p ale lui Z₂ (1000 de simulări), bootstrap-ul (2000 de reeșantionări) și simulările copulelor '
       '(200 000 de extrageri) vin din codul din curs. Alte praguri EVT, alte ferestre, alte niveluri, alte ponderi ale portofoliului, alte semințe, alte începuturi ale evaluării '
       'și cerința de capital Basel calculată din prognoze sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch10.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 10 · VaR, ES and Backtesting', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: VaR (value at risk), ES (expected shortfall), HS (historical simulation), EVT (extreme value '
        'theory), POT (peaks over threshold), GPD (generalised Pareto distribution), GARCH (generalised autoregressive conditional heteroskedasticity), FHS (filtered historical '
        'simulation), POF (proportion of failures), LR (likelihood ratio), BCBS (Basel Committee on Banking Supervision), FRTB (Fundamental Review of the Trading Book), MAR32 and '
        'MAR33 (chapters 32 and 33 of the Basel market risk standard), BVB (Bucharest Stock Exchange), CLT (central limit theorem).',
   src='Daily log returns, in %: S&P 500, DAX and BET from January 2000, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV) and OMV Petrom (SNP) from 2010, '
       'from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. The portfolio: simple returns of the BET and the S&P 500 on the 6,466 '
       'common days. EODHD prices. From the returns the browser computes the historical, Normal, Cornish–Fisher and EVT VaR and ES, the 500-day HS and Normal forecasts, the '
       'correlations, the pseudo-observations and the tail dependence; the Student-t, GPD and GARCH(1,1)-t parameters are those estimated by the code of the lecture. The rolling '
       'GARCH-t and FHS forecasts (re-estimated every 250 days) are rebuilt from the conditional volatility and the parameters of each re-estimation; the exceedances, the Kupiec '
       'and Christoffersen tests, the Basel traffic light, the quantile loss and Z₂ are recomputed in the browser. The p-values of Z₂ (1000 simulations), the bootstrap (2000 '
       'resamples) and the copula simulations (200,000 draws) come from the code of the lecture. Other EVT thresholds, windows, levels, portfolio weights, seeds and evaluation '
       'starts, and the Basel capital requirement computed from the forecasts, are extra runs of the same code.'),
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
    M6 = [('sp500', 'S&P 500' + lec), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)')]
    M4 = M6[:4]
    M3 = M6[:3]
    ser = 'Seria' if ro else 'Series'
    LEVV = [('0.01', '1%' + lec), ('0.005', n('0.5%')), ('0.025', n('2.5%')), ('0.05', '5%')]
    LEVE = [('0.025', n('2.5%') + lec), ('0.01', '1%'), ('0.05', '5%')]
    LEV2 = [('0.01', '1%' + lec), ('0.025', n('2.5%'))]
    lv = 'Nivelul VaR' if ro else 'VaR level'
    le = 'Nivelul ES' if ro else 'ES level'
    MET = lambda d: [(m, (('Normală' if ro else 'Normal') if m == 'Normal' else m) + (lec if m == d else '')) for m in ('HS', 'Normal', 'GARCH-t', 'FHS')]
    meth = 'Metoda' if ro else 'Method'
    POTQ = [('0.9', '90%' + lec), ('0.85', '85%'), ('0.95', '95%'), ('0.975', n('97.5%'))]
    thr = 'Pragul EVT: cuantila pierderilor' if ro else 'EVT threshold: quantile of the losses'
    days = 'zile' if ro else 'days'
    return {
     'def': (select('df-s', ser, M6, 'sp500') + select('df-v', lv, LEVV, '0.01') + select('df-e', le, LEVE, '0.025') + play('df'), CSS + charts('df-chart')),
     'normal-we': (select('nw-s', ser, M6, 'sp500') + play('nw'),
                   charts('nw-chart') + we('nw-we', f'{WE}: {"VaR și ES Normale" if ro else "Normal VaR and ES"}',
                                           r'\mathrm{VaR}_\alpha = -(\mu + \sigma z_\alpha), \qquad \mathrm{ES}_\alpha = -\mu + \sigma\,\frac{\varphi(z_\alpha)}{\alpha}',
                                           select('nw-a', lv, LEVV, '0.01') + select('nw-e', le, LEVE, '0.025'))),
     'horizon': (select('hz-s', ser, M6, 'sp500') + slider('hz-h', 'Orizontul h (zile)' if ro else 'Horizon h (days)', 1, 20, 1, 10) + play('hz'), charts('hz-chart')),
     'hs-we': (select('hs-s', ser, M6, 'sp500') + select('hs-n', 'Fereastra' if ro else 'Window', [('0', ('toate zilele' if ro else 'all days') + lec), ('250', ('ultimele 250 de zile' if ro else 'last 250 days')),
                                                                                                    ('500', ('ultimele 500 de zile' if ro else 'last 500 days')), ('1000', ('ultimele 1000 de zile' if ro else 'last 1000 days')),
                                                                                                    ('2500', ('ultimele 2500 de zile' if ro else 'last 2500 days'))], '0') + play('hs'),
               charts('hs-chart') + we('hs-wx', f'{WE}: {"VaR și ES istorice" if ro else "historical VaR and ES"}',
                                       r'\widehat{\mathrm{VaR}}_\alpha = -r_{(k)},\ k = \lceil n\alpha \rceil, \qquad \widehat{\mathrm{ES}}_\alpha = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}')),
     'precision': (select('pr-s', ser, M6, 'sp500') + select('pr-n', 'Ultimele n zile (panoul din dreapta)' if ro else 'Last n days (right panel)',
                                                             [('500', '500' + lec), ('250', '250'), ('1000', '1000'), ('2500', '2500')], '500') + play('pr'), charts('pr-chart')),
     'cf-we': (select('cf-s', ser, M6, 'sp500') + play('cf'),
               charts('cf-chart') + we('cf-wx', f'{WE}: {"cuantila Cornish–Fisher" if ro else "the Cornish–Fisher quantile"}',
                                       r'z_{CF} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}, \qquad \mathrm{VaR}_\alpha = -(\mu + \sigma z_{CF})',
                                       select('cf-a', lv, LEVV, '0.01'))),
     'evt-we': (select('ev-s', ser, M6, 'sp500') + select('ev-u', thr, POTQ, '0.9') + play('ev'),
                charts('ev-chart') + we('ev-we', f'{WE}: {"VaR și ES din GPD" if ro else "VaR and ES from the GPD"}',
                                        r'\mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}\Big[\Big(\frac{n\alpha}{N_u}\Big)^{-\xi} - 1\Big], \qquad \mathrm{ES}_\alpha = \frac{\mathrm{VaR}_\alpha + \beta - \xi u}{1 - \xi}',
                                        select('ev-a', lv, [('0.01', '1%' + lec), ('0.005', n('0.5%')), ('0.001', n('0.1%')), ('0.025', n('2.5%'))], '0.01'))),
     'curves': (select('cv-s', 'Seria din panoul din dreapta' if ro else 'Series of the right panel', [('bet', 'BET' + lec), ('dax', 'DAX'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'),
                                                                                                    ('snp', 'OMV Petrom (SNP)')], 'bet') + select('cv-u', thr, POTQ, '0.9') + play('cv'), charts('cv-chart')),
     'var-table': (select('vt-a', lv, LEVV, '0.01') + play('vt'),
                   charts('vt-chart') + xh('VaR pe metode, eșantioanele complete, în % (K: excesul de boltire)' if ro else 'VaR by method, full samples, in % (K: excess kurtosis)') + tw('vt-tab')),
     'es-table': (select('et-a', le, LEVE, '0.025') + play('et'),
                  charts('et-chart') + xh('ES pe metode, eșantioanele complete, în %' if ro else 'ES by method, full samples, in %') + tw('et-tab')),
     'cond-we': (select('cw-s', ser, M6, 'sp500') + play('cw'),
                 charts('cw-chart') + we('cw-we', f'{WE}: {"VaR și ES condiționate" if ro else "conditional VaR and ES"}',
                                         r'\mathrm{VaR}_{t+1} = -\big(\mu + \sigma_{t+1}\,q_\alpha(z)\big), \qquad \mathrm{ES}_{t+1} = -\mu + \sigma_{t+1}\,\mathrm{ES}_\alpha(z)',
                                         select('cw-a', lv, LEVV, '0.01'))),
     'cond-table': (select('ct-l', 'Nivelurile' if ro else 'Levels', [('0.01|0.025', n('VaR 1%, ES 2.5%') + lec), ('0.005|0.01', n('VaR 0.5%, ES 1%')), ('0.025|0.05', n('VaR 2.5%, ES 5%'))], '0.01|0.025') + play('ct'),
                    charts('ct-chart') + xh('Prognoze pentru ziua de după 18 septembrie 2026, în %' if ro else 'Forecasts for the day after 18 September 2026, in %') + tw('ct-tab')),
     'rolling': (select('rv-s', ser, M3, 'sp500') + select('rv-a', lv, LEV2, '0.01') + play('rv'), charts('rv-chart')),
     'stress-table': (select('sx-a', lv, LEV2, '0.01') + play('sx'),
                      charts('sx-chart') + xh('Depășiri ale VaR: zile cu r(t) &lt; −VaR(t)' if ro else 'VaR exceptions: days with r(t) &lt; −VaR(t)') + tw('sx-tab')),
     'vc-we': (slider('vc-w', 'Ponderea BET' if ro else 'Weight of the BET', 0, 1, 0.05, 0.5)
               + select('vc-r', 'Corelația folosită' if ro else 'Correlation used', [('s', n('corelația de selecție 0.21') + lec if ro else 'sample correlation 0.21' + lec), ('calm', n('2017 (an liniștit): 0.20') if ro else '2017 (calm year): 0.20'),
                                                                                    ('2008', n('sep. 2008 – mar. 2009: 0.37') if ro else 'Sep 2008 – Mar 2009: 0.37'), ('2020', n('feb. – mai 2020: 0.71') if ro else 'Feb – May 2020: 0.71'), ('one', '1')], 's') + play('vc'),
               charts('vc-chart') + we('vc-wx', f'{WE}: {"VaR-ul portofoliului 50/50" if ro else "VaR of the 50/50 portfolio"}',
                                       r'\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\sigma_1\sigma_2, \qquad \mathrm{VaR}_p = -(\mu_p + z_\alpha\sigma_p)')),
     'corr': (select('rc-w', 'Fereastra corelației (zile)' if ro else 'Correlation window (days)', [('250', '250' + lec), ('60', '60'), ('125', '125'), ('500', '500')], '250') + play('rc'), charts('rc-chart')),
     'tail-dep': (slider('td-q', 'q', 0.005, 0.1, 0.005, 0.05) + select('td-p', 'Perioada' if ro else 'Period', [('all', ('ianuarie 2000 – septembrie 2026' if ro else 'January 2000 – September 2026') + lec),
                                                                                                             ('p1', '2000–2009'), ('p2', '2010–2019'), ('p3', '2020–2026')], 'all') + play('td'), charts('td-chart')),
     'copula-we': (slider('cq-n', 'ν', 2, 40, 0.25, 4.5) + select('cq-r', 'ρ', [('tau', n('0.157 = sin(πτ̂/2)') + lec), ('s', n('0.21 (Pearson)')), ('2008', n('0.37 (2008)')), ('2020', n('0.71 (2020)'))], 'tau') + play('cq'),
                   charts('cq-chart') + we('cq-we', f'{WE}: {"dependența în coada stîngă a copulei t" if ro else "lower tail dependence of the t copula"}',
                                           r'\rho = \sin(\pi\hat\tau/2), \qquad \lambda_L = 2\,t_{\nu+1}\Big(-\sqrt{(\nu + 1)(1 - \rho)/(1 + \rho)}\Big)')),
     'copula': (slider('cp-q', 'Pragul pătratului q' if ro else 'Threshold of the square q', 0.01, 0.1, 0.01, 0.05)
                + select('cp-s', 'Semințele eșantioanelor copulelor' if ro else 'Seeds of the copula samples', [('2027', ('2027 și 2028' if ro else '2027 and 2028') + lec), ('2037', '2037 și 2038' if ro else '2037 and 2038'),
                                                                                                                 ('2047', '2047 și 2048' if ro else '2047 and 2048')], '2027') + play('cp'), charts('cp-chart')),
     'port-table': (select('pt-w', 'Ponderea BET' if ro else 'Weight of the BET', [('0.5', n('0.5') + lec), ('0.1', n('0.1')), ('0.25', n('0.25')), ('0.75', n('0.75')), ('0.9', n('0.9'))], '0.5') + play('pt'),
                    charts('pt-chart') + xh('VaR 1% și ES 2,5% ale portofoliului, în %' if ro else 'Portfolio VaR 1% and ES 2.5%, in %') + tw('pt-tab')),
     'binom': (slider('bn-p', 'Rata reală a depășirilor' if ro else 'True exceedance rate', 0.005, 0.03, 0.0025, 0.01)
               + select('bn-n', 'Zile' if ro else 'Days', [('250', '250' + lec), ('500', '500'), ('1000', '1000')], '250') + play('bn'), charts('bn-chart')),
     'kupiec-we': (slider('kp-x', 'Depășiri x' if ro else 'Exceptions x', 0, 15, 1, 7) + select('kp-n', 'Zile n' if ro else 'Days n', [('250', '250' + lec), ('500', '500'), ('1000', '1000')], '250') + play('kp'),
                   charts('kp-chart') + we('kp-we', f'{WE}: {"testul POF" if ro else "the POF test"}',
                                           c(r'LR_{uc} = -2\big[(n-x)\ln(1-\alpha) + x\ln\alpha - (n-x)\ln(1-\hat\pi) - x\ln\hat\pi\big], \qquad \hat\pi = x/n, \quad \alpha = 0.01'))),
     'kupiec-sp': (select('ks-s', ser, M4, 'sp500') + select('ks-m', meth, MET('GARCH-t'), 'GARCH-t') + play('ks'),
                   charts('ks-chart') + we('ks-we', f'{WE}: {"testul Kupiec pe prognozele din curs" if ro else "Kupiec on the forecasts of the lecture"}',
                                           r'\begin{gathered}\ln L_0 = (n - x)\ln(1-\alpha) + x\ln\alpha, \qquad \ln L_1 = (n - x)\ln(1-\hat\pi) + x\ln\hat\pi \\ LR_{uc} = -2(\ln L_0 - \ln L_1)\end{gathered}')),
     'chris-we': (select('cr-s', ser, M4, 'sp500') + select('cr-m', meth, MET('HS'), 'HS') + play('cr'),
                  charts('cr-chart') + we('cr-we', f'{WE}: {"independența și acoperirea condiționată" if ro else "independence and conditional coverage"}',
                                          r'LR_{ind} = -2\ln\frac{(1 - \hat\pi)^{n_{00} + n_{10}}\hat\pi^{n_{01} + n_{11}}}{(1 - \hat\pi_{01})^{n_{00}}\hat\pi_{01}^{n_{01}}(1 - \hat\pi_{11})^{n_{10}}\hat\pi_{11}^{n_{11}}}, \qquad LR_{cc} = LR_{uc} + LR_{ind}')),
     'basel': (select('bs-s', ser, M4, 'sp500') + select('bs-m', meth, MET(''), 'HS') + play('bs'),
               charts('bs-chart') + xh(f'Semaforul Basel: Binomial(250; 0,01) ({bc("1996b")})' if ro else f'The Basel traffic light: Binomial(250, 0.01) ({bc("1996b")})') + tw('bs-tab')
               + we('bs-we1', f'{WE}: {"zona din numărul depășirilor" if ro else "the zone from the number of exceptions"}', r'x \le 4: \text{' + ('verde' if ro else 'green') + r'}, \quad 5 \le x \le 9: \text{'
                    + ('galbenă' if ro else 'yellow') + r'}, \quad x \ge 10: \text{' + ('roșie' if ro else 'red') + '}',
                    slider('bs-x', 'Depășiri în 250 de zile' if ro else 'Exceptions in 250 days', 0, 12, 1, 6))
               + we('bs-we2', f'{WE}: {"cerința de capital" if ro else "the capital requirement"} ({bc("1996a")})',
                    r'\text{' + ('Capital' if ro else 'Capital') + r'} = \max\{\mathrm{VaR}_{t-1},\ (3 + \text{' + ('adaos' if ro else 'plus') + r'}) \times \overline{\mathrm{VaR}}_{60}\}',
                    select('bs-d', 'Ziua' if ro else 'Date', [('2026-09-18', '18 septembrie 2026' if ro else '18 September 2026'), ('2008-10-15', '15 octombrie 2008' if ro else '15 October 2008'),
                                                             ('2009-03-31', '31 martie 2009' if ro else '31 March 2009'), ('2020-03-31', '31 martie 2020' if ro else '31 March 2020'),
                                                             ('2022-10-14', '14 octombrie 2022' if ro else '14 October 2022')], '2026-09-18'))),
     'hits': (select('ht-s', ser, M4, 'sp500') + select('ht-a', lv, LEV2, '0.01') + play('ht'), charts('ht-chart')),
     'bt-sd': (select('b1-y', 'Începutul evaluării' if ro else 'Start of the evaluation', [('0', '2005' + lec), ('1', '2010'), ('2', '2015'), ('3', '2020')], '0') + play('b1'),
               charts('b1-chart') + xh('Backtesting-ul VaR 1% (p ind.: independența Christoffersen; p ac.c.: acoperirea condiționată)' if ro else 'Backtests of VaR 1% (p ind.: Christoffersen independence; p c.c.: conditional coverage)') + tw('b1-tab')),
     'bt-bb': (select('b2-y', 'Începutul evaluării' if ro else 'Start of the evaluation', [('0', '2005 (Bitcoin: 2017)' + lec), ('1', '2010 (Bitcoin: 2019)'), ('2', '2015 (Bitcoin: 2021)'), ('3', '2020 (Bitcoin: 2023)')], '0') + play('b2'),
               charts('b2-chart') + xh('Backtesting-ul VaR 1%, coloanele ca în tabelul precedent' if ro else 'Backtests of VaR 1%, columns as in the previous table') + tw('b2-tab')),
     'score': (select('sc-d', 'Ultima zi a evaluării' if ro else 'Last day of the evaluation', [('2026-09-18', ('18 septembrie 2026' if ro else '18 September 2026') + lec), ('2009-03-31', '31 martie 2009' if ro else '31 March 2009'),
                                                                                                  ('2020-05-29', '29 mai 2020' if ro else '29 May 2020'), ('2022-12-30', '30 decembrie 2022' if ro else '30 December 2022')], '2026-09-18') + play('sc'),
               charts('sc-chart') + xh('Pierderea cuantilă medie × 100 (cu verde: cea mai mică) și depășirile din ultimele 250 de zile' if ro else 'Mean quantile loss × 100 (in green: the smallest) and the exceptions of the last 250 days') + tw('sc-tab')),
     'traffic': (select('tt-s', ser, M4, 'sp500') + play('tt'), charts('tt-chart')),
     'es-bt': (select('eb-s', ser, M4, 'sp500') + play('eb'),
               charts('eb-chart') + we('eb-we', f'{WE}: {"statistica Z₂" if ro else "the Z₂ statistic"}',
                                       c(r'Z_2 = \frac{1}{T\alpha}\sum_{t=1}^{T}\frac{r_t I_t}{\mathrm{ES}_t} + 1, \qquad I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t\}, \quad \alpha = 0.025'),
                                       select('eb-m', meth, MET('FHS'), 'FHS'))),
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch10.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch10_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch10_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s  ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => (num(s) === 1 ? '1 zi' : `${s} ${de(s)}zile`);
const dep = s => (num(s) === 1 ? '1 depășire' : `${s} ${de(s)}depășiri`);
const pe = p => (String(p).startsWith('&lt;') ? 'p ' + p : 'p = ' + p);
const f2 = v => v.toFixed(2).replace('.', ',');
const pc = v => Math.round(Math.abs(100 * (1 - v)));
const X = ' (rulare suplimentară a codului din curs)';
const lst = a => (a.length ? a.join(', ') : 'niciuna');
window.CH10_TEXT = {
 h: { normal: 'Normală', hsall: 'HS (tot)', p2008: 'sep. 2008 – mar. 2009', p2020: 'feb. – mai 2020', exp: 'aștept.', method: 'Metoda', vc: 'varianță–covarianță (Normală)',
      gauss: 'copula Gaussiană, marginale empirice', tcop: 'copula t, marginale empirice', hsdata: 'simulare istorică (datele)', joint: 'P(prăbușire simultană la 1%)',
      zone: 'Zona', cum: 'Cumul. (%)', plus: 'Adaos', factor: 'Factor', zg: 'verde', zy: 'galbenă', zr: 'roșie', rate: 'rata (%)', pind: 'p ind.', pcc: 'p ac.c.', max250: 'max. 250 z.',
      qloss: 'pierderea cuantilă medie × 100', last250: 'depășiri, ultimele 250 de zile', per: { all: 'Ianuarie 2000 – septembrie 2026', p1: '2000–2009', p2: '2010–2019', p3: '2020–2026' } },
 df: (s, n, lv, v, k1, le, q, e, k25, nt, mn, dmn, l1, l2, x) => `${s}, ${zile(n)}${x ? X : ''}: VaR ${lv} = ${v}% (a ${k1}-a cea mai proastă zi); VaR ${le} = ${q}%; ES ${le} = ${e}%, `
   + `media celor mai proaste ${zile(k25)}. În ${zile(nt)} pierderea a fost mai mare decît VaR ${lv}; cea mai proastă zi: ${mn}% (${dmn}). Pentru o poziție de 1 milion de lei: `
   + `VaR ${lv} ≈ ${l1} ${de(l1)}lei, ES ${le} ≈ ${l2} ${de(l2)}lei.`,
 nw1: (s, n, mu, sd) => `Momentele de selecție ale randamentelor zilnice ale ${s} (${zile(n)}): μ̂ = ${mu}, σ̂ = ${sd} (în %).`,
 nw2: (lv, mu, z, sd, zs, v) => `VaR ${lv}: z = −${z}; VaR = −(μ̂ + zσ̂) = −${mu} + ${z} × ${sd} = −${mu} + ${zs} = ${v}%.`,
 nw3: (le, z, ph, f, mu, sd, fs, e, r) => `ES ${le}: z = −${z}, φ(z) = ${ph}, φ(z)/α = ${f}; ES = −${mu} + ${f} × ${sd} = −${mu} + ${fs} = ${e}%. `
   + `Cu media 0, ES ${le} / VaR 1% al distribuției Normale este φ(z)/α împărțit la 2,326, adică ${r}.`,
 nw4: (lv, vh, le, eh, d, d2) => `Datele (simulare istorică): VaR ${lv} = ${vh}%, ES ${le} = ${eh}%; VaR-ul Normal este ${num(d) >= 0 ? `cu ${d}% sub` : `cu ${String(d).replace('−', '')}% peste`} cel istoric, `
   + `ES-ul Normal ${num(d2) >= 0 ? `cu ${d2}% sub` : `cu ${String(d2).replace('−', '')}% peste`} cel istoric.`,
 nw: (s, lv, vn, vh, le, en, eh, x) => `${s}${x ? X : ''}: VaR ${lv} Normal ${vn}%, istoric ${vh}%; ES ${le} Normal ${en}%, istoric ${eh}%.`,
 hz: (s, h, v1, sh, vs, vh, r, vn, ac, above, x) => `${s}${x ? X : ''}: VaR 1% pe o zi ${v1}%; √${h} × ${v1} = ${vs}%; VaR 1% al randamentelor pe ${zile(h)} suprapuse: ${vh}%, `
   + `de ${r} ori VaR-ul pe o zi (√${h} = ${sh}); metoda Normală: ${vn}%. Autocorelația de ordinul 1 a randamentelor zilnice: ${ac}`
   + (above ? '; raportul este peste √h: zilele succesive tind să se miște în același sens, iar riscul pe mai multe zile crește mai repede decît √h.' : '.'),
 hs1: (s, n, from, na, k, v) => `${s}, ${zile(n)} din ${from}: k = ⌈n × 0,01⌉ = ⌈${na}⌉ = ${k}; VaR 1% = −r(${k}), a ${k}-a cea mai proastă zi: ${v}%.`,
 hs2: (na, k, e) => `ES 2,5%: k = ⌈n × 0,025⌉ = ⌈${na}⌉ = ${k}; media celor mai proaste ${zile(k)}, cu semn schimbat: ${e}%.`,
 hs3: (k, v, r) => `VaR 2,5% = −r(${k}) = ${v}%, marginea cozii pe care o mediază ES 2,5%; ES 2,5% / VaR 1% = ${r}.`,
 hs: (s, n, k1, v, k25, e, x) => `${s}, ${zile(n)}${x ? X : ''}: VaR 1% = ${v}%, din ${k1} ${de(k1)}${k1 === 1 ? 'observație' : 'observații'} în coadă; ES 2,5% = ${e}%, media a ${zile(k25)}.`,
 pr: (s, A, B, wA, wB, k, x) => `${s}${x ? X : ''}, intervale bootstrap de 95% (2000 de reeșantionări): toate cele ${zile(A[0])}: VaR 1% = ${A[1]}% [${A[2]}; ${A[3]}], ES 2,5% = ${A[4]}% [${A[5]}; ${A[6]}]; `
   + `ultimele ${zile(B[0])}: VaR 1% = ${B[1]}% [${B[2]}; ${B[3]}], ES 2,5% = ${B[4]}% [${B[5]}; ${B[6]}]. Lățimea intervalului pentru VaR 1%: ${wA} puncte procentuale pe tot eșantionul, `
   + `${wB} pe ultimele ${zile(B[0])}, unde VaR 1% se bazează pe ${k} ${de(k)}observații.`,
 cf1: (s, S, K, lv, z) => `${s}: asimetria S = ${S}, excesul de boltire K = ${K}; cuantila Normală pentru α = ${lv}: z = ${z}.`,
 cf2: (a, b, c, z) => `Corecțiile: (z² − 1)S/6 = ${a}; (z³ − 3z)K/24 = ${b}; −(2z³ − 5z)S²/36 = ${c}; deci z_CF = ${z}.`,
 cf3: (lv, mu, sd, z, v, vh, r) => `VaR ${lv} = −(${mu} + ${sd} × (${z})) = ${v}%, față de ${vh}% din simularea istorică: de ${r} ori.`,
 cf: (s, lv, z, v, vh, r, x) => `${s}${x ? X : ''}: z_CF = ${z}; VaR ${lv} Cornish–Fisher ${v}%, istoric ${vh}% (raportul ${r}).`,
 ev1: (s, q, u, Nu, sh, xi, b) => `${s}: pragul u = cuantila de ${q}% a pierderilor = ${u}%; N_u = ${Nu} ${de(Nu)}excese (${sh}% dintre zile); GPD estimată: ξ̂ = ${xi}, β̂ = ${b}.`,
 ev2: (lv, r, rx, bx, u, v) => `VaR ${lv}: nα/N_u = ${r}; ${r}^(−ξ̂) = ${rx}; β̂/ξ̂ = ${bx}; VaR = ${u} + ${bx} × (${rx} − 1) = ${v}%.`,
 ev3: (v25, b, xi, u, e) => `ES 2,5% = (VaR 2,5% + β̂ − ξ̂u)/(1 − ξ̂) = (${v25} + ${b} − ${xi} × ${u})/(1 − ${xi}) = ${e}%.`,
 ev4: (lv, vh, w, v0) => `Pentru comparație: VaR ${lv} istoric ${vh}%; cea mai mare pierdere observată ${w}%; VaR 0,1% EVT ${v0}%, extrapolat din coada estimată.`,
 ev: (s, q, lv, v, e, vh, x) => `${s}, pragul la cuantila de ${q}%${x ? X : ''}: VaR ${lv} EVT ${v}% (istoric ${vh}%); ES 2,5% EVT ${e}%.`,
 cv: (q, items, x) => `Pragul EVT la cuantila de ${q}% a pierderilor${x ? X : ''}. ` + items.map(([s, h, n, t, c, e, h5, lo, hi]) => `${s}, α = 0,1%: istoric ${h}%, Normală ${n}%, Student-t ${t}%, `
   + `Cornish–Fisher ${c}%, EVT ${e}%; α = 5%: toate metodele între ${lo}% și ${hi}% (istoric ${h5}%)`).join('. ') + '.',
 vt: (lv, d, lo, hi, clo, chi, x) => `VaR ${lv}${x ? X : ''}: HS, Student-t și EVT diferă cu cel mult ${d} puncte procentuale; `
   + (hi < 1 ? `VaR-ul Normal este peste tot sub cel istoric, cu ${pc(hi)}–${pc(lo)}%` : lo > 1 ? `VaR-ul Normal este peste tot peste cel istoric, cu ${pc(lo)}–${pc(hi)}%`
     : `raportul VaR Normal / VaR istoric este între ${f2(lo)} și ${f2(hi)}`) + `; Cornish–Fisher dă de ${clo} pînă la ${chi} ori VaR-ul istoric.`,
 et: (le, nr, lo, hi, tmax, bt, bh, x) => `ES ${le}${x ? X : ''}: pentru distribuția Normală cu media 0, ES ${le} / VaR 1% = ${nr}; pe date, ES ${le} istoric / VaR 1% istoric − 1 este între ${lo}% și ${hi}% `
   + `(din valorile nerotunjite). Student-t dă cel mai mare ES dintre HS, Normală, Student-t și EVT pentru: ${lst(tmax)}; Bitcoin: ${bt}%, față de ${bh}% istoric.`,
 cw1: (s, d, mu, a, b, nu, sg) => `${s}, GARCH(1,1)-t pe toate datele pînă la ${d}: μ̂ = ${mu}, α̂ = ${a}, β̂ = ${b}, ν̂ = ${nu}; prognoza pentru ziua următoare σ(t + 1) = ${sg}%.`,
 cw2: (lv, q, e, mu, sg, v, es) => `GARCH-t: q(z) = t⁻¹(α)√((ν − 2)/ν) = ${q}, ES 2,5%(z) = ${e}; VaR ${lv} = −(${mu} + ${sg} × (${q})) = ${v}%; ES 2,5% = −${mu} + ${sg} × ${e} = ${es}%.`,
 cw3: (lv, n, k, q, e, v, es) => `FHS: a ${k}-a cea mai mică dintre cele ${n} ${de(n)}reziduuri standardizate este ${q}; ES 2,5% al reziduurilor este ${e}; VaR ${lv} = −(μ̂ + σ(t + 1) × (${q})) = ${v}%; ES 2,5% = ${es}%.`,
 cw4: (lv, vh, r) => `VaR ${lv} istoric necondiționat: ${vh}%; VaR-ul GARCH-t de mîine este de ${r} ori această valoare.`,
 cw: (s, lv, vg, vf, vh, x) => `${s}${x ? X : ''}, VaR ${lv} pentru ziua următoare: GARCH-t ${vg}%, FHS ${vf}%; istoric necondiționat ${vh}%.`,
 ct: (lv, le, below, above, x) => `VaR ${lv} și ES ${le} pentru ziua de după 18 septembrie 2026${x ? X : ''}. VaR GARCH-t sub cel istoric: ${lst(below)}; peste cel istoric: ${lst(above)}.`,
 rv: (s, lv, items, x) => `${s}, VaR ${lv}${x ? X : ''}. ` + items.map(([p, a, b, c, d, vh, vg, dt]) => `${p}: depășiri HS ${a}, Normal ${b}, GARCH-t ${c}, FHS ${d}; la ${dt}, VaR HS ${vh}%, VaR GARCH-t ${vg}%`).join('. ') + '.',
 sx: (lv, n, xn, xh, xg, xf, n2, xn2, xg2, x) => `S&P 500, VaR ${lv}${x ? X : ''}: din septembrie 2008 pînă în martie 2009, VaR Normal este depășit în ${xn} din ${zile(n)}, HS în ${xh}, GARCH-t în ${xg}, FHS în ${xf}; `
   + `din februarie pînă în mai 2020 (${zile(n2)}): Normal ${xn2}, GARCH-t ${xg2}.`,
 vc1: (w1, w2, s1, s2, r, a, b, cv, v, sp) => `Ponderi: BET ${w1}, S&P 500 ${w2}; σ_BET = ${s1}, σ_S&P = ${s2}, ρ = ${r}, covarianța ρσ₁σ₂ = ${cv}. σp² = ${a}² + ${b}² + 2 × ${w1} × ${w2} × ${cv} = ${v}, σp = ${sp}.`,
 vc2: (mp, sp, vp, w1, v1, w2, v2, vs, ben) => `VaR 1% = −(${mp} − 2,326 × ${sp}) = ${vp}%; suma ponderată a celor două VaR Normale: ${w1} × ${v1} + ${w2} × ${v2} = ${vs}%; beneficiul diversificării: ${ben}%.`,
 vc3: (c1, c2, p1, p2) => `Contribuțiile Euler w_i(Σw)_i/σp: BET ${c1} (${p1}% din σp), S&P 500 ${c2} (${p2}%); suma lor este σp.`,
 vc: (w, r, vp, vs, ben, x) => `w(BET) = ${w}, ρ = ${r}${x ? X : ''}: VaR 1% al portofoliului ${vp}%, suma ponderată a celor două VaR ${vs}%, beneficiul diversificării ${ben}%.`,
 rc: (w, mx, dmx, mn, dmn, l, a, per, x) => `Corelația pe ${zile(w)}${x ? X : ''}: maximul ${mx} la ${dmx}, minimul ${mn} la ${dmn}, ultima valoare ${l}; pe tot eșantionul ${a}. `
   + `Corelația în fiecare perioadă: 2017 ${per[0]}; sep. 2008 – mar. 2009 ${per[1]}; feb. – mai 2020 ${per[2]}.`,
 td: (p, n, q, c, c5, c1, j1, x) => `${p} (${zile(n)} comune)${x ? X : ''}: la q = ${q}%, P(U_BET ≤ q | U_S&P ≤ q) = ${c}, iar la independență ar fi q; q = 5%: ${c5}; q = 1%: ${c1}. `
   + `Prăbușire simultană (ambii indici sub propria cuantilă de 1%): ${j1}% dintre zile, față de 0,01% la independență.`,
 cq1: (tau, r) => `τ̂ al lui Kendall = ${tau}; ρ = sin(πτ̂/2) = ${r}, același pentru ambele copule.`,
 cq2: (g, t, nu, lr) => `Log-verosimilitatea: copula Gaussiană ${g}, copula t ${t} (ν̂ = ${nu}); LR = 2 × (${t} − ${g}) = ${lr}, mult peste χ²₀,₉₅(1) = 3,84.`,
 cq3: (nu, r, nu1, s, tc, lam, ll) => `ν = ${nu}, ρ = ${r}: √((ν + 1)(1 − ρ)/(1 + ρ)) = ${s}; λ_L = 2 t_${nu1}(−${s}) = 2 × ${tc} = ${lam}. Log-verosimilitatea copulei t în ν = ${nu} (cu ρ = 0,157): ${ll}.`,
 cq: (nu, r, lam, ll, x) => `ν = ${nu}, ρ = ${r}${x ? X : ''}: λ_L = ${lam}; copula Gaussiană are λ_L = 0 pentru orice ρ &lt; 1.`,
 cp: (q, a, b, c, ind, seed, x) => `Zile în care ambii indici sînt printre cele mai proaste ${q}% ale lor${x ? X : ''}: datele ${a}, copula Gaussiană ${b}, copula t ${c}; la independență, circa ${ind}. `
   + `Eșantioanele copulelor (6 466 de perechi) folosesc semințele ${seed} și ${+seed + 1}.`,
 pt: (w, vn, vg, vt, vh, vs, x) => `Ponderea BET ${w}${x ? X : ''}: VaR 1% varianță–covarianță ${vn}%, copula Gaussiană ${vg}%, copula t ${vt}%, simulare istorică ${vh}%; suma ponderată a celor două VaR istorice: ${vs}%.`,
 bn: (n, p, np, p0, cg, pn, ylo, yhi, py, pr, x) => `${zile(n)}, rata reală a depășirilor ${p}%${x ? X : ''}: în medie ${np} depășiri; 0 depășiri cu probabilitatea ${p0}%. Zonele vin din VaR 1% corect: `
   + `verde 0–${ylo - 1} (probabilitatea cumulată ${cg}%), galbenă ${ylo}–${yhi}, roșie de la ${yhi + 1}. Cu rata ${p}%, modelul ajunge în zona galbenă sau roșie cu probabilitatea ${pn}% `
   + `(galbenă ${py}%, roșie ${pr}%).`,
 kp1: (n, x, r) => `n = ${n}, x = ${x}: π̂ = x/n = ${r}.`,
 kp2: (nx, x, l0) => `ln L₀ = ${nx} ln 0,99 + ${x} ln 0,01 = ${l0}.`,
 kp3: (nx, om, x, r, l1) => (x === 0 ? `ln L₁ = ${nx} ln 1 = 0 (cu x = 0, termenul x ln π̂ este 0).` : `ln L₁ = ${nx} ln ${om} + ${x} ln ${r} = ${l1}.`),
 kp4: (l0, l1, lr, p, rej) => `LR_uc = −2(${l0} − (${l1})) = ${lr}, ${pe(p)}: ${rej ? 'respins la 5% (LR_uc > 3,84)' : 'nu este respins la 5% (LR_uc ≤ 3,84)'}.`,
 kp5: (n, lo, hi, pw) => `În ${zile(n)} testul nu respinge la 5% niciun număr de depășiri între ${lo} și ${hi}; un model cu rata reală de 2% cade în acest interval cu probabilitatea ${pw}%.`,
 kp: (n, x, lr, p, rej, xx) => `${dep(x)} în ${zile(n)}${xx ? X : ''}: LR_uc = ${lr}, ${pe(p)}; ${rej ? 'respins' : 'nu este respins'} la 5%.`,
 ks1: (s, m, d0, d1, n, x, r, e, lo, hi) => `${s}, VaR 1% ${m}, ${d0} – ${d1}: n = ${n}, x = ${x}, π̂ = ${r}; așteptat nα = ${e}; intervalul binomial de 95% pentru rată, sub H₀: ${lo}–${hi}%.`,
 ks2: (nx, x, l0) => `ln L₀ = ${nx} ln 0,99 + ${x} ln 0,01 = ${l0}.`,
 ks3: (nx, r, x, l1) => `ln L₁ = ${nx} ln(1 − ${r}) + ${x} ln ${r} = ${l1}.`,
 ks4: (lr, p, rej) => `LR_uc = −2(ln L₀ − ln L₁) = ${lr}, ${pe(p)}: ${rej ? 'rata este respinsă la 5%' : 'rata nu este respinsă la 5%'}.`,
 ks: (s, m, x, e, r, lr, p, rej, xx) => `${s}, VaR 1% ${m}${xx ? X : ''}: ${dep(x)}, față de ${e} așteptate (rata ${r}%); LR_uc = ${lr}, ${pe(p)}, ${rej ? 'respins' : 'nu este respins'} la 5%.`,
 cr1: (s, m, n, a, b, c, d) => `${s}, VaR 1% ${m}, ${zile(n)}: n₀₀ = ${a}, n₀₁ = ${b}, n₁₀ = ${c}, n₁₁ = ${d}.`,
 cr2: (n01, n0, p01, n11, n1, p11, r) => `π̂₀₁ = ${n01}/${n0} = ${p01}%; π̂₁₁ = ${n11}/${n1} = ${p11}%; π̂₁₁/π̂₀₁ = ${r}.`,
 cr3: (l, p, rej) => `LR_ind = ${l}, ${pe(p)}: ${rej ? 'independența este respinsă la 5%, depășirile apar grupat' : 'independența nu este respinsă la 5%'}. `
   + 'Testul privește doar trecerile de la o zi la alta: depășiri adunate în aceeași lună, dar nu în zile consecutive, nu îl influențează.',
 cr4: (uc, i, c, p, rej) => `Acoperirea condiționată: LR_cc = LR_uc + LR_ind = ${uc} + ${i} = ${c}, ${pe(p)}: ${rej ? 'respins' : 'nu este respins'} la 5% (valoarea critică χ²₀,₉₅(2) = 5,99).`,
 cr: (s, m, p01, p11, l, p, lc, pc_, x) => `${s}, VaR 1% ${m}${x ? X : ''}: π̂₀₁ = ${p01}%, π̂₁₁ = ${p11}%; LR_ind = ${l} (${pe(p)}), LR_cc = ${lc} (${pe(pc_)}).`,
 bs1: (x, c, c1) => `${dep(x)} în 250 de zile: P(X ≤ ${x}) = ${c}% în distribuția Binomial(250; 0,01)` + (x ? `; P(X ≤ ${x - 1}) = ${c1}%.` : '.')
   + ' Zona galbenă începe unde probabilitatea cumulată trece de 95%, zona roșie unde atinge 99,99%.',
 bs2: (x, z, pl, f) => `Zona ${z}: adaosul ${pl}, factorul de multiplicare 3 + ${pl} = ${f}.`,
 bs3: (s, m, d, x, z, f) => `${s}, VaR ${m}, ${d}${X}: ${dep(x)} în cele 250 de zile anterioare, zona ${z}, factorul ${f}.`,
 bs4: (v, a, f, fa, cap, first) => `VaR 1% pe 10 zile de ieri: √10 × VaR-ul pe o zi = ${v}%; media ultimelor 60 de valori: ${a}%; ${f} × ${a} = ${fa}%; capitalul = max{${v}; ${fa}} = ${cap}% din valoarea poziției`
   + `${first ? ', dat de primul termen' : ''}. Primul termen contează doar cînd VaR-ul de ieri depășește de ${f} ori media pe 60 de zile, adică după un salt brusc al volatilității.`,
 bs: (s, m, sh, mx, dmx, nb, nd) => `${s}, VaR ${m}${X}: factorul este peste 3 în ${sh}% din cele ${zile(nd)}; capitalul maxim: ${mx}% din poziție (${dmx}); VaR-ul de ieri depășește termenul cu factor în ${zile(nb)}.`,
 ht: (s, lv, d0, n, e, items, x) => `${s}, VaR ${lv}, din ${d0} (${zile(n)})${x ? X : ''}; așteptate circa ${e} depășiri pentru fiecare metodă: `
   + items.map(([m, c, d]) => `${m} ${c} (${d} ${d === 1 ? 'zi' : `${de(d)}zile`} imediat după o altă depășire)`).join('; ') + '.',
 bt: (items, pass, x) => items.map(([s, d, n]) => `${s} din ${d} (${zile(n)})`).join(', ') + `${x ? X : ''}. Trec toate cele trei teste (POF, independență, acoperire condiționată) la 5%: ${lst(pass)}. `
   + 'Valorile p ale lui Z₂ vin din 1000 de simulări din distribuția prognozată.',
 sc: (d, best, nz, na, x) => `Pînă la ${d}${x ? X : ''}; cea mai mică pierdere cuantilă: ` + best.map(([s, m]) => `${s} ${m}`).join(', ')
   + `. În afara zonei verzi în ultimele 250 de zile: ${nz.length ? nz.map(([s, m, c, z]) => `${s} ${m} (${c}, zona ${z})`).join(', ') : 'niciun model'}${na.length ? `; fără prognoze pînă atunci: ${na.join(', ')}` : ''}.`,
 tt: (s, d0, items, x) => `${s}, de la ${d0}${x ? X : ''}: ponderea zilelor în zona roșie și cel mai mare număr de depășiri în 250 de zile: ` + items.map(([m, sh, mx]) => `${m} ${sh}% (${mx})`).join(', ') + '.',
 eb1: (s, m, T, nh, e) => `${s}, ${m}: T = ${zile(T)} de test, α = 2,5%; ${dep(nh)} ale VaR 2,5%, față de Tα = ${e}.`,
 eb2: (s, T, Ta, r, z) => `Σ r(t)I(t)/ES(t) = ${s}; împărțit la Tα = ${T} × 0,025 = ${Ta}: ${r}; Z₂ = ${r} + 1 = ${z}.`,
 eb3: (z, p, rej) => `p-value-ul prin simulare (1000 de extrageri din distribuția prognozată a fiecărei zile): ${p}. `
   + (rej ? `Z₂ = ${z} &lt; 0 și p &lt; 0,05: pierderile de dincolo de VaR sînt mai mari sau mai dese decît prognozează ES.` : `Z₂ = ${z}: diferența față de 0 nu este separată de zgomot de acest test, la 5%.`),
 eb: (s, items, x) => `${s}${x ? X : ''}, Z₂ pe toată perioada de test (p): ` + items.map(([m, z, p]) => `${m} ${z} (${p})`).join(', ') + '.'
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/,/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
const exc = s => (num(s) === 1 ? '1 exception' : `${s} exceptions`);
const pe = p => (String(p).startsWith('&lt;') ? 'p ' + p : 'p = ' + p);
const f2 = v => v.toFixed(2);
const pc = v => Math.round(Math.abs(100 * (1 - v)));
const X = ' (an extra run of the code of the lecture)';
const lst = a => (a.length ? a.join(', ') : 'none');
window.CH10_TEXT = {
 h: { normal: 'Normal', hsall: 'HS (all)', p2008: 'Sep 2008 – Mar 2009', p2020: 'Feb – May 2020', exp: 'expect.', method: 'Method', vc: 'variance–covariance (Normal)',
      gauss: 'Gaussian copula, empirical margins', tcop: 't copula, empirical margins', hsdata: 'historical simulation (the data)', joint: 'P(joint crash at 1%)',
      zone: 'Zone', cum: 'Cum. (%)', plus: 'Plus', factor: 'Factor', zg: 'green', zy: 'yellow', zr: 'red', rate: 'rate (%)', pind: 'p ind.', pcc: 'p c.c.', max250: 'max. 250 d.',
      qloss: 'mean quantile loss × 100', last250: 'exceptions, last 250 days', per: { all: 'January 2000 – September 2026', p1: '2000–2009', p2: '2010–2019', p3: '2020–2026' } },
 df: (s, n, lv, v, k1, le, q, e, k25, nt, mn, dmn, l1, l2, x) => `${s}, ${days(n)}${x ? X : ''}: VaR ${lv} = ${v}% (the ${k1}th worst day); VaR ${le} = ${q}%; ES ${le} = ${e}%, `
   + `the average of the worst ${days(k25)}. On ${days(nt)} the loss exceeded VaR ${lv}; the worst day: ${mn}% (${dmn}). For a position of 1 million lei: `
   + `VaR ${lv} ≈ ${l1} lei, ES ${le} ≈ ${l2} lei.`,
 nw1: (s, n, mu, sd) => `Sample moments of the daily returns of the ${s} (${days(n)}): μ̂ = ${mu}, σ̂ = ${sd} (in %).`,
 nw2: (lv, mu, z, sd, zs, v) => `VaR ${lv}: z = −${z}; VaR = −(μ̂ + zσ̂) = −${mu} + ${z} × ${sd} = −${mu} + ${zs} = ${v}%.`,
 nw3: (le, z, ph, f, mu, sd, fs, e, r) => `ES ${le}: z = −${z}, φ(z) = ${ph}, φ(z)/α = ${f}; ES = −${mu} + ${f} × ${sd} = −${mu} + ${fs} = ${e}%. `
   + `With mean 0, ES ${le} / VaR 1% of the Normal distribution is φ(z)/α divided by 2.326, that is ${r}.`,
 nw4: (lv, vh, le, eh, d, d2) => `The data (historical simulation): VaR ${lv} = ${vh}%, ES ${le} = ${eh}%; the Normal VaR is ${num(d) >= 0 ? `${d}% below` : `${String(d).replace('−', '')}% above`} the historical one, `
   + `the Normal ES ${num(d2) >= 0 ? `${d2}% below` : `${String(d2).replace('−', '')}% above`} the historical one.`,
 nw: (s, lv, vn, vh, le, en, eh, x) => `${s}${x ? X : ''}: VaR ${lv} Normal ${vn}%, historical ${vh}%; ES ${le} Normal ${en}%, historical ${eh}%.`,
 hz: (s, h, v1, sh, vs, vh, r, vn, ac, above, x) => `${s}${x ? X : ''}: one-day VaR 1% ${v1}%; √${h} × ${v1} = ${vs}%; VaR 1% of the overlapping ${h}-day returns: ${vh}%, `
   + `${r} times the one-day VaR (√${h} = ${sh}); Normal method: ${vn}%. First-order autocorrelation of the daily returns: ${ac}`
   + (above ? '; the ratio is above √h: consecutive days tend to move in the same direction, and the multi-day risk grows faster than √h.' : '.'),
 hs1: (s, n, from, na, k, v) => `${s}, ${days(n)} from ${from}: k = ⌈n × 0.01⌉ = ⌈${na}⌉ = ${k}; VaR 1% = −r(${k}), the ${k}th worst day: ${v}%.`,
 hs2: (na, k, e) => `ES 2.5%: k = ⌈n × 0.025⌉ = ⌈${na}⌉ = ${k}; the average of the worst ${days(k)}, with the sign changed: ${e}%.`,
 hs3: (k, v, r) => `VaR 2.5% = −r(${k}) = ${v}%, the edge of the tail that ES 2.5% averages; ES 2.5% / VaR 1% = ${r}.`,
 hs: (s, n, k1, v, k25, e, x) => `${s}, ${days(n)}${x ? X : ''}: VaR 1% = ${v}%, from ${k1} ${k1 === 1 ? 'observation' : 'observations'} in the tail; ES 2.5% = ${e}%, the average of ${days(k25)}.`,
 pr: (s, A, B, wA, wB, k, x) => `${s}${x ? X : ''}, 95% bootstrap intervals (2000 resamples): all ${days(A[0])}: VaR 1% = ${A[1]}% [${A[2]}, ${A[3]}], ES 2.5% = ${A[4]}% [${A[5]}, ${A[6]}]; `
   + `last ${days(B[0])}: VaR 1% = ${B[1]}% [${B[2]}, ${B[3]}], ES 2.5% = ${B[4]}% [${B[5]}, ${B[6]}]. Width of the interval for VaR 1%: ${wA} percentage points on the whole sample, `
   + `${wB} on the last ${days(B[0])}, where VaR 1% rests on ${k} observations.`,
 cf1: (s, S, K, lv, z) => `${s}: skewness S = ${S}, excess kurtosis K = ${K}; Normal quantile for α = ${lv}: z = ${z}.`,
 cf2: (a, b, c, z) => `The corrections: (z² − 1)S/6 = ${a}; (z³ − 3z)K/24 = ${b}; −(2z³ − 5z)S²/36 = ${c}; so z_CF = ${z}.`,
 cf3: (lv, mu, sd, z, v, vh, r) => `VaR ${lv} = −(${mu} + ${sd} × (${z})) = ${v}%, against ${vh}% from historical simulation: ${r} times.`,
 cf: (s, lv, z, v, vh, r, x) => `${s}${x ? X : ''}: z_CF = ${z}; Cornish–Fisher VaR ${lv} ${v}%, historical ${vh}% (ratio ${r}).`,
 ev1: (s, q, u, Nu, sh, xi, b) => `${s}: threshold u = the ${q}% quantile of the losses = ${u}%; N_u = ${Nu} exceedances (${sh}% of the days); fitted GPD: ξ̂ = ${xi}, β̂ = ${b}.`,
 ev2: (lv, r, rx, bx, u, v) => `VaR ${lv}: nα/N_u = ${r}; ${r}^(−ξ̂) = ${rx}; β̂/ξ̂ = ${bx}; VaR = ${u} + ${bx} × (${rx} − 1) = ${v}%.`,
 ev3: (v25, b, xi, u, e) => `ES 2.5% = (VaR 2.5% + β̂ − ξ̂u)/(1 − ξ̂) = (${v25} + ${b} − ${xi} × ${u})/(1 − ${xi}) = ${e}%.`,
 ev4: (lv, vh, w, v0) => `For comparison: historical VaR ${lv} ${vh}%; largest observed loss ${w}%; EVT VaR 0.1% ${v0}%, extrapolated from the fitted tail.`,
 ev: (s, q, lv, v, e, vh, x) => `${s}, threshold at the ${q}% quantile${x ? X : ''}: EVT VaR ${lv} ${v}% (historical ${vh}%); EVT ES 2.5% ${e}%.`,
 cv: (q, items, x) => `EVT threshold at the ${q}% quantile of the losses${x ? X : ''}. ` + items.map(([s, h, n, t, c, e, h5, lo, hi]) => `${s}, α = 0.1%: historical ${h}%, Normal ${n}%, Student-t ${t}%, `
   + `Cornish–Fisher ${c}%, EVT ${e}%; α = 5%: all methods between ${lo}% and ${hi}% (historical ${h5}%)`).join('. ') + '.',
 vt: (lv, d, lo, hi, clo, chi, x) => `VaR ${lv}${x ? X : ''}: HS, Student-t and EVT differ by at most ${d} percentage points; `
   + (hi < 1 ? `the Normal VaR is below the historical one everywhere, by ${pc(hi)}–${pc(lo)}%` : lo > 1 ? `the Normal VaR is above the historical one everywhere, by ${pc(lo)}–${pc(hi)}%`
     : `the ratio Normal VaR / historical VaR is between ${f2(lo)} and ${f2(hi)}`) + `; Cornish–Fisher gives ${clo} to ${chi} times the historical VaR.`,
 et: (le, nr, lo, hi, tmax, bt, bh, x) => `ES ${le}${x ? X : ''}: for the Normal distribution with mean 0, ES ${le} / VaR 1% = ${nr}; in the data, historical ES ${le} / historical VaR 1% − 1 is between ${lo}% and ${hi}% `
   + `(from the unrounded values). Student-t gives the largest ES among HS, Normal, Student-t and EVT for: ${lst(tmax)}; Bitcoin: ${bt}%, against ${bh}% historical.`,
 cw1: (s, d, mu, a, b, nu, sg) => `${s}, GARCH(1,1)-t on all data up to ${d}: μ̂ = ${mu}, α̂ = ${a}, β̂ = ${b}, ν̂ = ${nu}; next-day forecast σ(t + 1) = ${sg}%.`,
 cw2: (lv, q, e, mu, sg, v, es) => `GARCH-t: q(z) = t⁻¹(α)√((ν − 2)/ν) = ${q}, ES 2.5%(z) = ${e}; VaR ${lv} = −(${mu} + ${sg} × (${q})) = ${v}%; ES 2.5% = −${mu} + ${sg} × ${e} = ${es}%.`,
 cw3: (lv, n, k, q, e, v, es) => `FHS: the ${k}th smallest of the ${n} standardised residuals is ${q}; the ES 2.5% of the residuals is ${e}; VaR ${lv} = −(μ̂ + σ(t + 1) × (${q})) = ${v}%; ES 2.5% = ${es}%.`,
 cw4: (lv, vh, r) => `Unconditional historical VaR ${lv}: ${vh}%; tomorrow's GARCH-t VaR is ${r} times this value.`,
 cw: (s, lv, vg, vf, vh, x) => `${s}${x ? X : ''}, next-day VaR ${lv}: GARCH-t ${vg}%, FHS ${vf}%; unconditional historical ${vh}%.`,
 ct: (lv, le, below, above, x) => `VaR ${lv} and ES ${le} for the day after 18 September 2026${x ? X : ''}. GARCH-t VaR below the historical one: ${lst(below)}; above the historical one: ${lst(above)}.`,
 rv: (s, lv, items, x) => `${s}, VaR ${lv}${x ? X : ''}. ` + items.map(([p, a, b, c, d, vh, vg, dt]) => `${p}: exceptions HS ${a}, Normal ${b}, GARCH-t ${c}, FHS ${d}; on ${dt}, HS VaR ${vh}%, GARCH-t VaR ${vg}%`).join('. ') + '.',
 sx: (lv, n, xn, xh, xg, xf, n2, xn2, xg2, x) => `S&P 500, VaR ${lv}${x ? X : ''}: from September 2008 to March 2009 the Normal VaR is exceeded on ${xn} of ${days(n)}, HS on ${xh}, GARCH-t on ${xg}, FHS on ${xf}; `
   + `from February to May 2020 (${days(n2)}): Normal ${xn2}, GARCH-t ${xg2}.`,
 vc1: (w1, w2, s1, s2, r, a, b, cv, v, sp) => `Weights: BET ${w1}, S&P 500 ${w2}; σ_BET = ${s1}, σ_S&P = ${s2}, ρ = ${r}, covariance ρσ₁σ₂ = ${cv}. σp² = ${a}² + ${b}² + 2 × ${w1} × ${w2} × ${cv} = ${v}, σp = ${sp}.`,
 vc2: (mp, sp, vp, w1, v1, w2, v2, vs, ben) => `VaR 1% = −(${mp} − 2.326 × ${sp}) = ${vp}%; weighted sum of the two Normal VaRs: ${w1} × ${v1} + ${w2} × ${v2} = ${vs}%; diversification benefit: ${ben}%.`,
 vc3: (c1, c2, p1, p2) => `Euler contributions w_i(Σw)_i/σp: BET ${c1} (${p1}% of σp), S&P 500 ${c2} (${p2}%); they add up to σp.`,
 vc: (w, r, vp, vs, ben, x) => `w(BET) = ${w}, ρ = ${r}${x ? X : ''}: portfolio VaR 1% ${vp}%, weighted sum of the two VaRs ${vs}%, diversification benefit ${ben}%.`,
 rc: (w, mx, dmx, mn, dmn, l, a, per, x) => `${w}-day correlation${x ? X : ''}: maximum ${mx} on ${dmx}, minimum ${mn} on ${dmn}, last value ${l}; whole sample ${a}. `
   + `Correlation within each period: 2017 ${per[0]}; Sep 2008 – Mar 2009 ${per[1]}; Feb – May 2020 ${per[2]}.`,
 td: (p, n, q, c, c5, c1, j1, x) => `${p} (${days(n)} in common)${x ? X : ''}: at q = ${q}%, P(U_BET ≤ q | U_S&P ≤ q) = ${c}, while under independence it would be q; q = 5%: ${c5}; q = 1%: ${c1}. `
   + `Joint crash (both indices below their own 1% quantile): ${j1}% of the days, against 0.01% under independence.`,
 cq1: (tau, r) => `Kendall's τ̂ = ${tau}; ρ = sin(πτ̂/2) = ${r}, the same for both copulas.`,
 cq2: (g, t, nu, lr) => `Log-likelihood: Gaussian copula ${g}, t copula ${t} (ν̂ = ${nu}); LR = 2 × (${t} − ${g}) = ${lr}, far above χ²₀.₉₅(1) = 3.84.`,
 cq3: (nu, r, nu1, s, tc, lam, ll) => `ν = ${nu}, ρ = ${r}: √((ν + 1)(1 − ρ)/(1 + ρ)) = ${s}; λ_L = 2 t_${nu1}(−${s}) = 2 × ${tc} = ${lam}. Log-likelihood of the t copula at ν = ${nu} (with ρ = 0.157): ${ll}.`,
 cq: (nu, r, lam, ll, x) => `ν = ${nu}, ρ = ${r}${x ? X : ''}: λ_L = ${lam}; the Gaussian copula has λ_L = 0 for every ρ &lt; 1.`,
 cp: (q, a, b, c, ind, seed, x) => `Days on which both indices are among their worst ${q}%${x ? X : ''}: the data ${a}, Gaussian copula ${b}, t copula ${c}; under independence, about ${ind}. `
   + `The copula samples (6,466 pairs) use the seeds ${seed} and ${+seed + 1}.`,
 pt: (w, vn, vg, vt, vh, vs, x) => `Weight of the BET ${w}${x ? X : ''}: VaR 1% variance–covariance ${vn}%, Gaussian copula ${vg}%, t copula ${vt}%, historical simulation ${vh}%; weighted sum of the two historical VaRs: ${vs}%.`,
 bn: (n, p, np, p0, cg, pn, ylo, yhi, py, pr, x) => `${days(n)}, true exceedance rate ${p}%${x ? X : ''}: on average ${np} exceptions; 0 exceptions with probability ${p0}%. The zones come from a correct VaR 1%: `
   + `green 0–${ylo - 1} (cumulative probability ${cg}%), yellow ${ylo}–${yhi}, red from ${yhi + 1}. With the rate ${p}%, the model lands in the yellow or red zone with probability ${pn}% `
   + `(yellow ${py}%, red ${pr}%).`,
 kp1: (n, x, r) => `n = ${n}, x = ${x}: π̂ = x/n = ${r}.`,
 kp2: (nx, x, l0) => `ln L₀ = ${nx} ln 0.99 + ${x} ln 0.01 = ${l0}.`,
 kp3: (nx, om, x, r, l1) => (x === 0 ? `ln L₁ = ${nx} ln 1 = 0 (with x = 0, the term x ln π̂ is 0).` : `ln L₁ = ${nx} ln ${om} + ${x} ln ${r} = ${l1}.`),
 kp4: (l0, l1, lr, p, rej) => `LR_uc = −2(${l0} − (${l1})) = ${lr}, ${pe(p)}: ${rej ? 'rejected at 5% (LR_uc > 3.84)' : 'not rejected at 5% (LR_uc ≤ 3.84)'}.`,
 kp5: (n, lo, hi, pw) => `In ${days(n)} the test does not reject at 5% any number of exceptions between ${lo} and ${hi}; a model with a true rate of 2% falls in this range with probability ${pw}%.`,
 kp: (n, x, lr, p, rej, xx) => `${exc(x)} in ${days(n)}${xx ? X : ''}: LR_uc = ${lr}, ${pe(p)}; ${rej ? 'rejected' : 'not rejected'} at 5%.`,
 ks1: (s, m, d0, d1, n, x, r, e, lo, hi) => `${s}, ${m} VaR 1%, ${d0} – ${d1}: n = ${n}, x = ${x}, π̂ = ${r}; expected nα = ${e}; 95% binomial interval for the rate under H₀: ${lo}–${hi}%.`,
 ks2: (nx, x, l0) => `ln L₀ = ${nx} ln 0.99 + ${x} ln 0.01 = ${l0}.`,
 ks3: (nx, r, x, l1) => `ln L₁ = ${nx} ln(1 − ${r}) + ${x} ln ${r} = ${l1}.`,
 ks4: (lr, p, rej) => `LR_uc = −2(ln L₀ − ln L₁) = ${lr}, ${pe(p)}: ${rej ? 'the rate is rejected at 5%' : 'the rate is not rejected at 5%'}.`,
 ks: (s, m, x, e, r, lr, p, rej, xx) => `${s}, ${m} VaR 1%${xx ? X : ''}: ${exc(x)}, against ${e} expected (rate ${r}%); LR_uc = ${lr}, ${pe(p)}, ${rej ? 'rejected' : 'not rejected'} at 5%.`,
 cr1: (s, m, n, a, b, c, d) => `${s}, ${m} VaR 1%, ${days(n)}: n₀₀ = ${a}, n₀₁ = ${b}, n₁₀ = ${c}, n₁₁ = ${d}.`,
 cr2: (n01, n0, p01, n11, n1, p11, r) => `π̂₀₁ = ${n01}/${n0} = ${p01}%; π̂₁₁ = ${n11}/${n1} = ${p11}%; π̂₁₁/π̂₀₁ = ${r}.`,
 cr3: (l, p, rej) => `LR_ind = ${l}, ${pe(p)}: ${rej ? 'independence is rejected at 5%, the exceptions come in clusters' : 'independence is not rejected at 5%'}. `
   + 'The test looks only at day-to-day transitions: exceptions bunched in the same month, but not on consecutive days, do not affect it.',
 cr4: (uc, i, c, p, rej) => `Conditional coverage: LR_cc = LR_uc + LR_ind = ${uc} + ${i} = ${c}, ${pe(p)}: ${rej ? 'rejected' : 'not rejected'} at 5% (critical value χ²₀.₉₅(2) = 5.99).`,
 cr: (s, m, p01, p11, l, p, lc, pc_, x) => `${s}, ${m} VaR 1%${x ? X : ''}: π̂₀₁ = ${p01}%, π̂₁₁ = ${p11}%; LR_ind = ${l} (${pe(p)}), LR_cc = ${lc} (${pe(pc_)}).`,
 bs1: (x, c, c1) => `${exc(x)} in 250 days: P(X ≤ ${x}) = ${c}% under Binomial(250, 0.01)` + (x ? `; P(X ≤ ${x - 1}) = ${c1}%.` : '.')
   + ' The yellow zone starts where the cumulative probability passes 95%, the red zone where it reaches 99.99%.',
 bs2: (x, z, pl, f) => `${z[0].toUpperCase() + z.slice(1)} zone: plus factor ${pl}, multiplier 3 + ${pl} = ${f}.`,
 bs3: (s, m, d, x, z, f) => `${s}, ${m} VaR, ${d}${X}: ${exc(x)} in the previous 250 days, ${z} zone, multiplier ${f}.`,
 bs4: (v, a, f, fa, cap, first) => `Yesterday's 10-day VaR 1%: √10 × one-day VaR = ${v}%; average of the last 60 values: ${a}%; ${f} × ${a} = ${fa}%; capital = max{${v}, ${fa}} = ${cap}% of the position`
   + `${first ? ', set by the first term' : ''}. The first term matters only when yesterday's VaR exceeds ${f} times the 60-day average, that is after a sudden jump in volatility.`,
 bs: (s, m, sh, mx, dmx, nb, nd) => `${s}, ${m} VaR${X}: the multiplier is above 3 on ${sh}% of the ${days(nd)}; maximum capital: ${mx}% of the position (${dmx}); yesterday's VaR exceeds the multiplier term on ${days(nb)}.`,
 ht: (s, lv, d0, n, e, items, x) => `${s}, VaR ${lv}, from ${d0} (${days(n)})${x ? X : ''}; about ${e} exceptions expected for each method: `
   + items.map(([m, c, d]) => `${m} ${c} (${days(d)} right after another exception)`).join('; ') + '.',
 bt: (items, pass, x) => items.map(([s, d, n]) => `${s} from ${d} (${days(n)})`).join(', ') + `${x ? X : ''}. Passing all three tests (POF, independence, conditional coverage) at 5%: ${lst(pass)}. `
   + 'The p-values of Z₂ come from 1000 simulations from the forecast distribution.',
 sc: (d, best, nz, na, x) => `Up to ${d}${x ? X : ''}; smallest quantile loss: ` + best.map(([s, m]) => `${s} ${m}`).join(', ')
   + `. Outside the green zone in the last 250 days: ${nz.length ? nz.map(([s, m, c, z]) => `${s} ${m} (${c}, ${z} zone)`).join(', ') : 'no model'}${na.length ? `; no forecasts by then: ${na.join(', ')}` : ''}.`,
 tt: (s, d0, items, x) => `${s}, from ${d0}${x ? X : ''}: share of the days in the red zone and the largest number of exceptions in 250 days: ` + items.map(([m, sh, mx]) => `${m} ${sh}% (${mx})`).join(', ') + '.',
 eb1: (s, m, T, nh, e) => `${s}, ${m}: T = ${days(T)} of testing, α = 2.5%; ${exc(nh)} of VaR 2.5%, against Tα = ${e}.`,
 eb2: (s, T, Ta, r, z) => `Σ r(t)I(t)/ES(t) = ${s}; divided by Tα = ${T} × 0.025 = ${Ta}: ${r}; Z₂ = ${r} + 1 = ${z}.`,
 eb3: (z, p, rej) => `p-value by simulation (1000 draws from each day's forecast distribution): ${p}. `
   + (rej ? `Z₂ = ${z} &lt; 0 and p &lt; 0.05: the losses beyond VaR are larger or more frequent than the ES forecast says.` : `Z₂ = ${z}: the difference from 0 is not separated from noise by this test at 5%.`),
 eb: (s, items, x) => `${s}${x ? X : ''}, Z₂ over the whole test period (p): ` + items.map(([m, z, p]) => `${m} ${z} (${p})`).join(', ') + '.'
};
})();
""",
}

# question for the room and answer of every block (RO, EN)
QA = {
 'ro': {
  'def': ('Pe S&P 500 din 2000, VaR 1% istoric este 3,46%. Cît este ES 2,5% istoric?',
          '3,76%: media celor mai proaste 168 de zile, cu semn schimbat. VaR 2,5% = 2,51% este doar marginea acestei cozi, iar ES 2,5% face media întregii cozi, inclusiv a zilelor din 2008 și 2020. '
          f'ES 2,5% și VaR 1% sînt apropiate (3,76% față de 3,46%), de aceea FRTB a putut înlocui VaR 1% cu ES 2,5% fără să schimbe mult nivelul capitalului ({bc("2019")}; {bc("2023b")}).'),
  'normal-we': ('Ce se întîmplă cu VaR 1% Normal față de cel istoric cînd trecem la celelalte cinci serii?',
                'Rămîne sub cel istoric peste tot, cu 14–24%: 22% la DAX, 23% la BET, 24% la Bitcoin, 22% la TLV, 14% la SNP (S&P 500: 19%). Doi parametri, media și abaterea standard, nu pot descrie '
                'o coadă groasă: distribuția Normală pune prea puțină probabilitate dincolo de 2,3 abateri standard.'),
  'horizon': ('VaR 1% al randamentelor S&P 500 pe 10 zile: este peste sau sub √10 × VaR-ul pe o zi (10,93%)?',
              'Sub: 10,37%, adică de 3,00 ori VaR-ul pe o zi, față de √10 = 3,16. Suma a 10 randamente cu cozi groase este mai aproape de distribuția Normală (CLT), iar autocorelația '
              'de ordinul 1 a randamentelor S&P 500 este −0,10, deci zilele succesive se compensează parțial. La BET raportul este 3,86, peste √10: autocorelația lui este +0,11.'),
  'hs-we': ('Cu ultimele 250 de zile ale S&P 500, pe a cîta cea mai proastă zi se sprijină VaR 1% istoric?',
            'Pe a treia: k = ⌈250 × 0,01⌉ = ⌈2,5⌉ = 3. Cu 250 de zile, VaR 1% depinde de 3 observații, iar ES 2,5% de 7 (⌈6,25⌉); pe tot eșantionul de 6 714 zile, VaR 1% este a 68-a cea mai proastă zi.'),
  'precision': ('Pe ultimele 500 de zile ale S&P 500, intervalul bootstrap de 95% pentru VaR 1% este de cîte ori mai lat decît pe toate cele 6 714 zile?',
                'De circa 6 ori: [1,99; 4,96], adică 2,97 puncte procentuale, față de [3,18; 3,66], adică 0,48. Regula √n ar da √(6 714/500) = 3,7; intervalul este mai lat, pentru că VaR 1% '
                'se sprijină pe a 5-a cea mai proastă zi, iar între cele cinci zile proaste ale ferestrei sînt salturi mari. Pe ultimele 250 de zile intervalul are doar 1,12 puncte: o fereastră '
                'fără zile extreme pare precisă, deși are mai puține observații.'),
  'cf-we': ('S&P 500: S = −0,35, K = 10,7. Cum se compară VaR 1% Cornish–Fisher cu cel istoric (3,46%)?',
            'Este aproape dublu: 6,08%. Termenul de boltire (z³ − 3z)K/24 = −2,49 adaugă singur mai mult decît distanța dintre cuantila Normală și cea empirică; dezvoltarea este o corecție '
            'mică în jurul distribuției Normale și nu mai este valabilă cînd K trece de 10.'),
  'evt-we': ('Ridicăm pragul EVT al S&P 500 de la cuantila de 90% la cea de 97,5% a pierderilor. Ce face VaR 1% EVT?',
             'Se schimbă puțin: 3,56% la 90%, 3,54% la 95%, 3,50% la 97,5% (3,59% la 85%). ξ̂ crește de la 0,16 la 0,23, dar la 1% suntem încă aproape de datele folosite; diferența apare '
             'la 0,1%, unde VaR EVT trece de la 6,91% la 7,13%.'),
  'curves': ('La α = 0,1%, care metodă dă cel mai mare VaR pentru S&P 500?',
             'Cornish–Fisher: 15,06%, aproape dublu față de valoarea istorică de 7,90%. Student-t dă 8,58%, EVT 6,91%, distribuția Normală doar 3,72%. La α = 5% toate metodele sînt între 1,67% și 1,97%.'),
  'var-table': ('La VaR 5%, mai este VaR-ul Normal sub cel istoric?',
                'Nu: la 5% VaR-ul Normal este peste cel istoric pentru toate cele șase serii, cu 3–16% (S&P 500: 1,97% față de 1,85%). O distribuție cu cozi groase și aceeași varianță are mai multă '
                'probabilitate în centru și în coada extremă și mai puțină pe „umeri”: cuantila ei de 5% este mai aproape de 0 decît cea Normală.'),
  'es-table': ('Pentru distribuția Normală, ES 2,5% / VaR 1% = 1,005. Pe date, cu cît depășește ES 2,5% istoric VaR 1% istoric?',
               'Cu 1,5–10,5%, după serie (DAX 1,5%, BET 10,5%, din valorile nerotunjite). ES 2,5% istoric face media unei cozi mai lungi decît prevede distribuția Normală, deci raportul trece de 1,005.'),
  'cond-we': ('S&P 500, 18 septembrie 2026: cum se compară VaR 1% GARCH-t pentru ziua următoare cu VaR 1% istoric (3,46%)?',
              'Este cam jumătate: 1,75%. σ(t + 1) = 0,714% este mult sub abaterea standard de selecție de 1,21%: piața este liniștită. FHS dă 1,91%, pentru că reziduurile standardizate au '
              'cuantila de 1% −2,787, mai jos decît −2,557 pentru Student-t estimată.'),
  'cond-table': ('Pentru care dintre cele șase serii este VaR 1% GARCH-t de mîine peste VaR 1% istoric?',
                 'BET (4,51% față de 4,14%), TLV (7,18% față de 5,11%) și SNP (5,06% față de 4,29%): pe acestea volatilitatea curentă este mare. Pe S&P 500, DAX și Bitcoin perioada este liniștită.'),
  'rolling': ('Din iunie 2007 pînă în decembrie 2009, pe S&P 500: care metodă are cele mai multe depășiri ale VaR 1%?',
              'Metoda Normală: 44, față de 27 pentru HS și 15 pentru GARCH-t și FHS. Fereastra de 500 de zile se schimbă încet, iar distribuția Normală are coada prea subțire; '
              'după criză, HS rămîne ridicat doi ani (ghost effect).'),
  'stress-table': ('Din septembrie 2008 pînă în martie 2009, de cîte ori a fost depășit VaR 1% Normal al S&P 500, din 146 de zile?',
                   'De 21 de ori, față de 1,46 așteptate; HS de 15 ori, GARCH-t și FHS de cîte 5 ori. Prognoza volatilității contează mai mult decît forma cozii.'),
  'vc-we': ('Cu ρ = 1 în locul corelației de selecție, ce devine VaR 1% al portofoliului 50/50?',
            '3,03%: suma ponderată a celor două VaR Normale, 0,5 × 3,23 + 0,5 × 2,82; beneficiul diversificării, 22% cu ρ = 0,21, dispare. Cu ρ = 0,71 (primăvara lui 2020) VaR-ul este '
            '2,80% și beneficiul doar 8%.'),
  'corr': ('Cînd a atins corelația pe 250 de zile dintre BET și S&P 500 maximul?',
           'Pe 25 martie 2020, 0,67. Corelația din fiecare perioadă: 0,20 în 2017, 0,37 din septembrie 2008 pînă în martie 2009, 0,71 din februarie pînă în mai 2020. BVB tranzacționează '
           'pînă la 17:45, ora României, iar New York se deschide la 16:30, deci o parte din mișcarea de la New York ajunge la București abia a doua zi.'),
  'tail-dep': ('La q = 1%: cît este P(BET printre cele mai proaste 1% ale lui | S&P 500 printre cele mai proaste 1% ale lui)?',
               '0,19, de 19 ori mai mult decît la independență (0,01). La q = 5% probabilitatea este 0,22. Corelația Pearson de 0,21 este dominată de zilele obișnuite și nu spune nimic despre '
               'cît de des cad cei doi indici împreună.'),
  'copula-we': ('Ce se întîmplă cu λ_L al copulei t cînd ν crește de la 4,5 la 40, cu ρ = 0,157?',
                'Scade spre 0: 0,096 la ν = 4,5, 0,016 la ν = 10 și practic 0 la ν = 40. Cînd ν → ∞, copula t devine copula Gaussiană, care are λ_L = 0.'),
  'copula': ('În 6 466 de zile, în cîte zile sînt ambii indici printre cele mai proaste 5% ale lor?',
             'În 70 de zile, față de circa 16 (6 466 × 0,05²) la independență. Copula Gaussiană estimată dă 37, copula t 55: doar copula t reproduce o parte din prăbușirile simultane.'),
  'port-table': ('Portofoliul 50/50 BET–S&P 500: care metodă dă cel mai mare VaR 1%?',
                 'Simularea istorică: 3,08%. Copula t dă 2,81%, copula Gaussiană 2,75%, metoda varianță–covarianță 2,35%. Copula t rămîne sub datele istorice: legătura dintre cei doi indici '
                 'este mai puternică în crize decît poate descrie o copulă constantă.'),
  'binom': ('Un model are rata reală a depășirilor 2%, dublu față de VaR 1%. Cu ce probabilitate rămîne în zona verde într-un an (250 de zile)?',
            f'Circa 44% (P(X ≤ 4) = 43,9% pentru Binomial(250; 0,02)). Un an de date separă doar aproximativ un model bun de unul prost; un model corect ajunge în zona galbenă cu probabilitatea 10,8% ({bc("1996b")}).'),
  'kupiec-we': ('250 de zile și 0 depășiri: respinge testul Kupiec modelul la 5%?',
                'Da: ln L₀ = 250 ln 0,99 = −2,51, ln L₁ = 0, LR_uc = 5,03 > 3,84. Prea puține depășiri indică un VaR prea prudent, care irosește capital. Cu 250 de zile, testul nu respinge '
                'doar numerele de depășiri de la 1 la 6.'),
  'kupiec-sp': ('În afară de GARCH-t, trece vreo metodă testul Kupiec la 5% pe S&P 500, din 2005 pînă în 2026?',
                'Niciuna: HS are rata 1,41% (p = 0,004), FHS 1,58%, metoda Normală 2,71% (ambele p &lt; 0,001). Cu 5 460 de zile, testul are o putere mare și respinge chiar rate de 1,4%.'),
  'chris-we': ('Pentru care metode respinge testul de independență Christoffersen la 5% pe S&P 500?',
               'HS (p &lt; 0,001; π̂₁₁ = 10,4% față de π̂₀₁ = 1,28%) și metoda Normală (p &lt; 0,001). GARCH-t (p = 0,114) și FHS (p = 0,060) nu sînt respinse: prognoza volatilității '
               'desface grupurile de depășiri. Testul privește doar zilele consecutive.'),
  'basel': ('Un model are 6 depășiri ale VaR 1% în 250 de zile. Ce factor de multiplicare se aplică?',
            f'3,50: P(X ≤ 6) = 98,63% este între 95% și 99,99%, deci zona galbenă, cu adaosul 0,50 ({bc("1996b")}). Capitalul este max{{VaR(t − 1), 3,50 × media pe 60 de zile}} ({bc("1996a")}); '
            f'FRTB calculează capitalul din ES 2,5% ({bc("2023b")}), dar backtesting-ul numără tot depășirile VaR 1% ({bc("2023a")}).'),
  'hits': ('S&P 500, 2005–2026: care metodă are cele mai puține depășiri ale VaR 1%?',
           'HS: 77, față de 86 pentru FHS, 95 pentru GARCH-t și 148 pentru metoda Normală; așteptate circa 55. HS are cele mai puține depășiri, dar ele apar grupat în crize, '
           'în timp ce depășirile GARCH-t și FHS sînt răspîndite și mai multe în anii liniștiți.'),
  'bt-sd': ('Evaluăm din 2015 în loc de 2005. Trece metoda Normală testul POF pe S&P 500 sau pe DAX?',
            'Nu: din 2015 rata ei este 2,72% pe S&P 500 și 2,52% pe DAX, ambele cu p &lt; 0,001. Fără 2008 rata scade puțin, dar coada Normală rămîne prea subțire.'),
  'bt-bb': ('BET, din 2005: care metode trec toate cele trei teste (POF, independență, acoperire condiționată) la 5%?',
            'GARCH-t (p 0,158; 0,807; 0,358) și FHS (p 0,158; 0,240; 0,185). HS trece POF (p = 0,123), dar depășirile ei apar grupat (p ind. 0,001).'),
  'score': ('Bitcoin: care metodă are cea mai mică pierdere cuantilă medie?',
            'GARCH-t, la limită: 13,53 față de 13,57 pentru HS, 13,75 pentru FHS și 14,52 pentru metoda Normală. Pe indicii de acțiuni diferențele sînt mari (S&P 500: 3,68 pentru FHS, '
            '5,80 pentru metoda Normală); pe Bitcoin cele patru metode sînt apropiate.'),
  'traffic': ('S&P 500, de la 28 decembrie 2005 (sfîrșitul primei ferestre de 250 de zile): ce pondere din zile a petrecut VaR Normal în zona roșie?',
              f'O treime (33,3%); HS 11,8%, GARCH-t 0,5%. Cel mai mare număr de depășiri în 250 de zile: 29 pentru metoda Normală, 18 pentru HS, 10 pentru GARCH-t. O bancă cu VaR Normal '
              f'ar fi plătit penalizarea zonei roșii ({bc("1996b")}) în cea mai mare parte din 2008–2009.'),
  'es-bt': ('Ce semn are Z₂ pentru prognozele ES 2,5% ale S&P 500 și ce înseamnă?',
            'Negativ pentru toate cele patru metode: HS −0,39, Normal −1,15, GARCH-t −0,54, FHS −0,20 (p 0,013 pentru FHS, sub 0,001 pentru celelalte). Pierderile de dincolo de VaR sînt mai '
            'mari sau mai dese decît prognozează ES; FHS este cel mai aproape de 0, pentru că preia coada lui z din date.'),
 },
 'en': {
  'def': ('On the S&P 500 since 2000, the historical VaR 1% is 3.46%. What is the historical ES 2.5%?',
          '3.76%: the average of the worst 168 days, with the sign changed. VaR 2.5% = 2.51% is only the edge of this tail, while ES 2.5% averages the whole tail, including the days of 2008 '
          f'and 2020. ES 2.5% and VaR 1% are close (3.76% against 3.46%), which is why FRTB could replace VaR 1% by ES 2.5% without changing the capital level much ({bc("2019")}; {bc("2023b")}).'),
  'normal-we': ('What happens to the Normal VaR 1% relative to the historical one when we move to the other five series?',
                'It stays below the historical one everywhere, by 14–24%: 22% for the DAX, 23% for the BET, 24% for Bitcoin, 22% for TLV, 14% for SNP (S&P 500: 19%). Two parameters, the mean and '
                'the standard deviation, cannot describe a heavy tail: the Normal distribution puts too little probability beyond 2.3 standard deviations.'),
  'horizon': ('The VaR 1% of the 10-day S&P 500 returns: is it above or below √10 × the one-day VaR (10.93%)?',
              'Below: 10.37%, that is 3.00 times the one-day VaR, against √10 = 3.16. The sum of 10 heavy-tailed returns is closer to the Normal distribution (CLT), and the first-order '
              'autocorrelation of the S&P 500 returns is −0.10, so consecutive days partly offset each other. For the BET the ratio is 3.86, above √10: its autocorrelation is +0.11.'),
  'hs-we': ('With the last 250 days of the S&P 500, which worst day does the historical VaR 1% rest on?',
            'The third: k = ⌈250 × 0.01⌉ = ⌈2.5⌉ = 3. With 250 days, VaR 1% depends on 3 observations and ES 2.5% on 7 (⌈6.25⌉); on the whole sample of 6,714 days, VaR 1% is the 68th worst day.'),
  'precision': ('On the last 500 days of the S&P 500, how many times wider is the 95% bootstrap interval for VaR 1% than on all 6,714 days?',
                'About 6 times: [1.99, 4.96], that is 2.97 percentage points, against [3.18, 3.66], that is 0.48. The √n rule would give √(6,714/500) = 3.7; the interval is wider because VaR 1% '
                'rests on the 5th worst day, and the five worst days of the window are far apart. On the last 250 days the interval is only 1.12 points wide: a window without extreme days '
                'looks precise although it has fewer observations.'),
  'cf-we': ('S&P 500: S = −0.35, K = 10.7. How does the Cornish–Fisher VaR 1% compare with the historical one (3.46%)?',
            'It is almost double: 6.08%. The kurtosis term (z³ − 3z)K/24 = −2.49 alone adds more than the distance between the Normal and the empirical quantile; the expansion is a small '
            'correction around the Normal distribution and no longer holds when K exceeds 10.'),
  'evt-we': ('We raise the EVT threshold of the S&P 500 from the 90% to the 97.5% quantile of the losses. What does the EVT VaR 1% do?',
             'It changes little: 3.56% at 90%, 3.54% at 95%, 3.50% at 97.5% (3.59% at 85%). ξ̂ rises from 0.16 to 0.23, but at 1% we are still close to the data used; the difference shows '
             'at 0.1%, where the EVT VaR moves from 6.91% to 7.13%.'),
  'curves': ('At α = 0.1%, which method gives the largest VaR for the S&P 500?',
             'Cornish–Fisher: 15.06%, almost double the historical 7.90%. Student-t gives 8.58%, EVT 6.91%, the Normal distribution only 3.72%. At α = 5% all methods lie between 1.67% and 1.97%.'),
  'var-table': ('At VaR 5%, is the Normal VaR still below the historical one?',
                'No: at 5% the Normal VaR is above the historical one for all six series, by 3–16% (S&P 500: 1.97% against 1.85%). A heavy-tailed distribution with the same variance has more '
                'probability in the centre and in the far tail and less on the "shoulders": its 5% quantile is closer to 0 than the Normal one.'),
  'es-table': ('For the Normal distribution, ES 2.5% / VaR 1% = 1.005. In the data, by how much does the historical ES 2.5% exceed the historical VaR 1%?',
               'By 1.5–10.5%, depending on the series (DAX 1.5%, BET 10.5%, from the unrounded values). The historical ES 2.5% averages a longer tail than the Normal distribution predicts, so the ratio exceeds 1.005.'),
  'cond-we': ('S&P 500, 18 September 2026: how does the next-day GARCH-t VaR 1% compare with the historical VaR 1% (3.46%)?',
              'It is about half: 1.75%. σ(t + 1) = 0.714% is far below the sample standard deviation of 1.21%: the market is calm. FHS gives 1.91%, because the standardised residuals have '
              'a 1% quantile of −2.787, lower than −2.557 for the fitted Student-t.'),
  'cond-table': ("For which of the six series is tomorrow's GARCH-t VaR 1% above the historical VaR 1%?",
                 'BET (4.51% against 4.14%), TLV (7.18% against 5.11%) and SNP (5.06% against 4.29%): current volatility is high there. For the S&P 500, the DAX and Bitcoin the period is calm.'),
  'rolling': ('From June 2007 to December 2009, on the S&P 500: which method has the most exceptions of VaR 1%?',
              'The Normal method: 44, against 27 for HS and 15 for GARCH-t and FHS. The 500-day window changes slowly, and the Normal distribution has too thin a tail; after the crisis, '
              'HS stays high for two years (ghost effect).'),
  'stress-table': ('From September 2008 to March 2009, how many times was the Normal VaR 1% of the S&P 500 exceeded, out of 146 days?',
                   '21 times, against 1.46 expected; HS 15 times, GARCH-t and FHS 5 times each. The volatility forecast matters more than the shape of the tail.'),
  'vc-we': ('With ρ = 1 instead of the sample correlation, what becomes of the VaR 1% of the 50/50 portfolio?',
            '3.03%: the weighted sum of the two Normal VaRs, 0.5 × 3.23 + 0.5 × 2.82; the diversification benefit, 22% with ρ = 0.21, disappears. With ρ = 0.71 (spring 2020) the VaR is '
            '2.80% and the benefit only 8%.'),
  'corr': ('When did the 250-day correlation of the BET and the S&P 500 reach its maximum?',
           'On 25 March 2020, at 0.67. The correlation within each period: 0.20 in 2017, 0.37 from September 2008 to March 2009, 0.71 from February to May 2020. The BVB trades until 17:45 '
           'Romanian time and New York opens at 16:30, so part of the New York move reaches Bucharest only the next day.'),
  'tail-dep': ('At q = 1%: what is P(BET among its worst 1% | S&P 500 among its worst 1%)?',
               '0.19, 19 times more than under independence (0.01). At q = 5% the probability is 0.22. The Pearson correlation of 0.21 is dominated by ordinary days and says nothing about '
               'how often the two indices fall together.'),
  'copula-we': ('What happens to the λ_L of the t copula when ν rises from 4.5 to 40, with ρ = 0.157?',
                'It falls towards 0: 0.096 at ν = 4.5, 0.016 at ν = 10 and practically 0 at ν = 40. As ν → ∞, the t copula becomes the Gaussian copula, which has λ_L = 0.'),
  'copula': ('In 6,466 days, on how many days are both indices among their worst 5%?',
             'On 70 days, against about 16 (6,466 × 0.05²) under independence. The fitted Gaussian copula gives 37, the t copula 55: only the t copula reproduces part of the joint crashes.'),
  'port-table': ('The 50/50 BET–S&P 500 portfolio: which method gives the largest VaR 1%?',
                 'Historical simulation: 3.08%. The t copula gives 2.81%, the Gaussian copula 2.75%, the variance–covariance method 2.35%. The t copula stays below the historical value: the link '
                 'between the two indices is stronger in crises than a constant copula can describe.'),
  'binom': ('A model has a true exceedance rate of 2%, double that of VaR 1%. With what probability does it stay in the green zone in one year (250 days)?',
            f'About 44% (P(X ≤ 4) = 43.9% under Binomial(250, 0.02)). One year of data separates a good model from a bad one only roughly; a correct model lands in the yellow zone with probability 10.8% ({bc("1996b")}).'),
  'kupiec-we': ('250 days and 0 exceptions: does the Kupiec test reject the model at 5%?',
                'Yes: ln L₀ = 250 ln 0.99 = −2.51, ln L₁ = 0, LR_uc = 5.03 > 3.84. Too few exceptions point to an overly prudent VaR that wastes capital. With 250 days, the test does not reject '
                'only the counts from 1 to 6.'),
  'kupiec-sp': ('Apart from GARCH-t, does any method pass the Kupiec test at 5% on the S&P 500, from 2005 to 2026?',
                'None: HS has the rate 1.41% (p = 0.004), FHS 1.58%, the Normal method 2.71% (both p &lt; 0.001). With 5,460 days the test has high power and rejects even rates of 1.4%.'),
  'chris-we': ("For which methods does Christoffersen's independence test reject at 5% on the S&P 500?",
               'HS (p &lt; 0.001; π̂₁₁ = 10.4% against π̂₀₁ = 1.28%) and the Normal method (p &lt; 0.001). GARCH-t (p = 0.114) and FHS (p = 0.060) are not rejected: the volatility forecast '
               'breaks up the clusters of exceptions. The test looks only at consecutive days.'),
  'basel': ('A model has 6 exceptions of VaR 1% in 250 days. Which multiplier applies?',
            f'3.50: P(X ≤ 6) = 98.63% is between 95% and 99.99%, so the yellow zone, with plus factor 0.50 ({bc("1996b")}). The capital is max{{VaR(t − 1), 3.50 × 60-day average}} ({bc("1996a")}); '
            f'FRTB computes the capital from ES 2.5% ({bc("2023b")}), but the backtest still counts the exceptions of VaR 1% ({bc("2023a")}).'),
  'hits': ('S&P 500, 2005–2026: which method has the fewest exceptions of VaR 1%?',
           'HS: 77, against 86 for FHS, 95 for GARCH-t and 148 for the Normal method; about 55 expected. HS has the fewest exceptions, but they come in clusters during crises, '
           'while the GARCH-t and FHS exceptions are spread out and more frequent in calm years.'),
  'bt-sd': ('We evaluate from 2015 instead of 2005. Does the Normal method pass the POF test on the S&P 500 or on the DAX?',
            'No: from 2015 its rate is 2.72% on the S&P 500 and 2.52% on the DAX, both with p &lt; 0.001. Without 2008 the rate falls a little, but the Normal tail is still too thin.'),
  'bt-bb': ('BET, from 2005: which methods pass all three tests (POF, independence, conditional coverage) at 5%?',
            'GARCH-t (p 0.158, 0.807, 0.358) and FHS (p 0.158, 0.240, 0.185). HS passes POF (p = 0.123), but its exceptions come in clusters (p ind. 0.001).'),
  'score': ('Bitcoin: which method has the smallest mean quantile loss?',
            'GARCH-t, by a hair: 13.53 against 13.57 for HS, 13.75 for FHS and 14.52 for the Normal method. On the stock indices the differences are large (S&P 500: 3.68 for FHS, 5.80 for '
            'the Normal method); on Bitcoin the four methods are close.'),
  'traffic': ('S&P 500, from 28 December 2005 (the end of the first 250-day window): what share of the days did the Normal VaR spend in the red zone?',
              f'One third (33.3%); HS 11.8%, GARCH-t 0.5%. The largest number of exceptions in 250 days: 29 for the Normal method, 18 for HS, 10 for GARCH-t. A bank with a Normal VaR '
              f'would have paid the red-zone penalty ({bc("1996b")}) for most of 2008–2009.'),
  'es-bt': ('What is the sign of Z₂ for the ES 2.5% forecasts of the S&P 500, and what does it mean?',
            'Negative for all four methods: HS −0.39, Normal −1.15, GARCH-t −0.54, FHS −0.20 (p 0.013 for FHS, below 0.001 for the others). The losses beyond VaR are larger or more '
            'frequent than ES forecasts; FHS is closest to 0 because it takes the tail of z from the data.'),
 },
}

PREDICT = {
 'def': (0, [('3,76%: puțin peste VaR 1%.', '3.76%: a little above VaR 1%.'), ('2,51%: cuantila de 2,5% a randamentelor.', '2.51%: the 2.5% quantile of the returns.'),
             ('Circa 8,6%: VaR 1% × 2,5, coada fiind de 2,5 ori mai lungă.', 'About 8.6%: VaR 1% × 2.5, since the tail is 2.5 times longer.')]),
 'normal-we': (0, [('Rămîne sub cel istoric pentru toate seriile, cu 14–24%.', 'It stays below the historical one for all series, by 14–24%.'),
                   ('Ajunge peste cel istoric pentru Bitcoin: volatilitatea mare compensează cozile.', 'It ends above the historical one for Bitcoin: the high volatility makes up for the tails.'),
                   ('Este egal cu cel istoric pe seriile lungi: cu multe date, media și abaterea standard ajung.', 'It equals the historical one on the long series: with many data, mean and standard deviation suffice.')]),
 'horizon': (0, [('Sub: circa 10,4%.', 'Below: about 10.4%.'), ('Exact 10,93%: regula √h este exactă pentru orice distribuție.', 'Exactly 10.93%: the √h rule is exact for any distribution.'),
                 ('Peste: cozile groase se adună, deci raportul trece de √10.', 'Above: heavy tails add up, so the ratio exceeds √10.')]),
 'hs-we': (0, [('Pe a treia: ⌈2,5⌉ = 3.', 'The third: ⌈2.5⌉ = 3.'), ('Pe a doua: 250 × 0,01 = 2,5, rotunjit în jos.', 'The second: 250 × 0.01 = 2.5, rounded down.'),
               ('Pe a 68-a, ca pe tot eșantionul.', 'The 68th, as on the whole sample.')]),
 'precision': (0, [('De circa 6 ori.', 'About 6 times.'), ('De circa 3,7 ori: regula √n.', 'About 3.7 times: the √n rule.'),
                   ('La fel de lat: bootstrap-ul face tot 2000 de reeșantionări.', 'Just as wide: the bootstrap still draws 2000 resamples.')]),
 'cf-we': (0, [('Aproape dublu: circa 6,1%.', 'Almost double: about 6.1%.'), ('Aproape egal: corecția pentru boltire aduce cuantila la valoarea datelor.', 'About equal: the kurtosis correction brings the quantile to the data.'),
               ('Mai mic: asimetria negativă micșorează pierderea.', 'Smaller: the negative skewness reduces the loss.')]),
 'evt-we': (0, [('Se schimbă puțin, între 3,50% și 3,59%.', 'It changes little, between 3.50% and 3.59%.'), ('Se dublează: cu mai puține excese, coada estimată explodează.', 'It doubles: with fewer exceedances the fitted tail explodes.'),
                ('Coboară spre VaR Normal (2,80%): pragul înalt lasă doar cîteva puncte.', 'It falls towards the Normal VaR (2.80%): a high threshold leaves only a few points.')]),
 'curves': (0, [('Cornish–Fisher.', 'Cornish–Fisher.'), ('Student-t: are cozile cele mai groase (ν̂ = 2,67).', 'Student-t: it has the heaviest tails (ν̂ = 2.67).'),
                ('EVT: este construită pentru coada extremă.', 'EVT: it is built for the far tail.')]),
 'var-table': (0, [('Nu: la 5% este peste cel istoric.', 'No: at 5% it is above the historical one.'), ('Da, tot cu 14–24%: cozile Normale sînt subțiri la orice nivel.', 'Yes, still by 14–24%: Normal tails are thin at every level.'),
                   ('Da, și mai mult: diferența crește cu α.', 'Yes, and by more: the gap grows with α.')]),
 'es-table': (0, [('Cu 1,5–10,5%, după serie.', 'By 1.5–10.5%, depending on the series.'), ('Cu 0,5%, ca la distribuția Normală.', 'By 0.5%, as for the Normal distribution.'),
                  ('De 2,5 ori: ES face media unei cozi de 2,5 ori mai lungi.', '2.5 times: ES averages a tail 2.5 times longer.')]),
 'cond-we': (0, [('Cam jumătate: circa 1,75%.', 'About half: about 1.75%.'), ('La fel, 3,46%: GARCH folosește aceleași date.', 'The same, 3.46%: GARCH uses the same data.'),
                 ('Aproape dublu: inovațiile Student-t au cozi groase.', 'Almost double: Student-t innovations have heavy tails.')]),
 'cond-table': (0, [('BET, TLV și SNP.', 'BET, TLV and SNP.'), ('S&P 500, DAX și Bitcoin: au cele mai lungi istorii.', 'S&P 500, DAX and Bitcoin: they have the longest histories.'),
                    ('Niciuna: GARCH-t micșorează întotdeauna VaR-ul.', 'None: GARCH-t always lowers the VaR.')]),
 'rolling': (0, [('Metoda Normală.', 'The Normal method.'), ('GARCH-t: reacționează prea brusc la șocuri.', 'GARCH-t: it reacts too abruptly to shocks.'),
                 ('HS: nu presupune nicio distribuție.', 'HS: it assumes no distribution.')]),
 'stress-table': (0, [('21.', '21.'), ('1 sau 2: sînt așteptate 1,46.', '1 or 2: 1.46 are expected.'), ('5, ca GARCH-t: toate metodele văd aceeași criză.', '5, like GARCH-t: all methods see the same crisis.')]),
 'vc-we': (0, [('3,03%: suma ponderată a celor două VaR.', '3.03%: the weighted sum of the two VaRs.'), ('2,35%: ponderile nu se schimbă, deci nici VaR-ul.', '2.35%: the weights do not change, so neither does the VaR.'),
               ('6,06%: suma celor două VaR.', '6.06%: the sum of the two VaRs.')]),
 'corr': (0, [('În martie 2020.', 'In March 2020.'), ('În octombrie 2008, în criza financiară globală.', 'In October 2008, in the global financial crisis.'),
              ('În 2000: piețele erau mai integrate la începutul eșantionului.', 'In 2000: markets were more integrated at the start of the sample.')]),
 'tail-dep': (0, [('Circa 0,19.', 'About 0.19.'), ('0,01 = q: corelația de 0,21 este mică.', '0.01 = q: the correlation of 0.21 is small.'), ('0,21: egală cu corelația Pearson.', '0.21: equal to the Pearson correlation.')]),
 'copula-we': (0, [('Scade spre 0.', 'It falls towards 0.'), ('Crește: mai multe grade de libertate, mai multă dependență.', 'It rises: more degrees of freedom, more dependence.'),
                   ('Rămîne 0,096: λ_L depinde doar de ρ.', 'It stays at 0.096: λ_L depends only on ρ.')]),
 'copula': (0, [('În 70 de zile.', 'On 70 days.'), ('În circa 16 zile: 6 466 × 0,05².', 'On about 16 days: 6,466 × 0.05².'), ('În 323 de zile: 5% din zile.', 'On 323 days: 5% of the days.')]),
 'port-table': (0, [('Simularea istorică.', 'Historical simulation.'), ('Copula t: are dependență în coadă.', 'The t copula: it has tail dependence.'),
                    ('Varianță–covarianță: folosește toată matricea de covarianță.', 'Variance–covariance: it uses the whole covariance matrix.')]),
 'binom': (0, [('Circa 44%.', 'About 44%.'), ('Circa 0%: media, 5, este deja în zona galbenă.', 'About 0%: the mean, 5, is already in the yellow zone.'),
               ('Circa 89%, ca un model corect.', 'About 89%, like a correct model.')]),
 'kupiec-we': (0, [('Da: LR_uc = 5,03 > 3,84.', 'Yes: LR_uc = 5.03 > 3.84.'), ('Nu: zero depășiri este rezultatul cel mai bun.', 'No: zero exceptions is the best possible result.'),
                   ('Testul nu se poate calcula: ln 0 nu există.', 'The test cannot be computed: ln 0 does not exist.')]),
 'kupiec-sp': (0, [('Niciuna.', 'None.'), ('HS: rata 1,41% este aproape de 1%.', 'HS: the rate 1.41% is close to 1%.'), ('FHS: preia coada din date.', 'FHS: it takes the tail from the data.')]),
 'chris-we': (0, [('HS și metoda Normală.', 'HS and the Normal method.'), ('Toate patru: toate au mai multe depășiri decît 1%.', 'All four: they all have more exceptions than 1%.'),
                  ('Doar GARCH-t: are mai multe depășiri în anii liniștiți.', 'Only GARCH-t: it has more exceptions in calm years.')]),
 'basel': (0, [('3,50.', '3.50.'), ('4,00: peste 4 depășiri se aplică factorul maxim.', '4.00: above 4 exceptions the maximum factor applies.'),
               ('3,00: 6 depășiri trec testul Kupiec la 5%.', '3.00: 6 exceptions pass the Kupiec test at 5%.')]),
 'hits': (0, [('HS.', 'HS.'), ('GARCH-t: urmărește volatilitatea.', 'GARCH-t: it tracks the volatility.'), ('FHS: combină GARCH cu coada din date.', 'FHS: it combines GARCH with the tail of the data.')]),
 'bt-sd': (0, [('Nu, pe niciuna.', 'No, on neither.'), ('Da: fără 2008, metoda Normală devine corectă.', 'Yes: without 2008 the Normal method becomes correct.'),
               ('Doar pe DAX: are cozi mai subțiri decît S&P 500.', 'Only on the DAX: it has thinner tails than the S&P 500.')]),
 'bt-bb': (0, [('GARCH-t și FHS.', 'GARCH-t and FHS.'), ('Doar HS: rata ei, 1,22%, nu este respinsă.', 'Only HS: its rate, 1.22%, is not rejected.'),
               ('Toate în afară de metoda Normală.', 'All except the Normal method.')]),
 'score': (0, [('GARCH-t, la limită.', 'GARCH-t, by a hair.'), ('FHS: are cele mai puține depășiri.', 'FHS: it has the fewest exceptions.'), ('HS: Bitcoin are puține date.', 'HS: Bitcoin has few data.')]),
 'traffic': (0, [('O treime.', 'One third.'), ('Circa 1%: doar din întîmplare.', 'About 1%: only by chance.'), ('Circa 12%, ca HS.', 'About 12%, like HS.')]),
 'es-bt': (0, [('Negativ: pierderile de dincolo de VaR sînt mai mari sau mai dese decît prognozează ES.', 'Negative: the losses beyond VaR are larger or more frequent than ES forecasts.'),
               ('Pozitiv: ES este prea prudent.', 'Positive: ES is too prudent.'), ('Zero: E[Z₂] = 0 pentru orice model.', 'Zero: E[Z₂] = 0 for every model.')]),
}

# blocks whose initial chart, table or worked example shows the answer: veil the readout and those elements
VEIL = {'def': {'out': True, 'charts': ['df-chart']}, 'normal-we': {'out': True}, 'horizon': {'out': True, 'charts': ['hz-chart']}, 'hs-we': {'out': True},
        'precision': {'out': True, 'charts': ['pr-chart']}, 'cf-we': {'out': True, 'charts': ['cf-wx-box']}, 'evt-we': {'out': True}, 'curves': {'out': True, 'charts': ['cv-chart']},
        'var-table': {'out': True}, 'es-table': {'out': True, 'charts': ['et-tab']}, 'cond-we': {'out': True, 'charts': ['cw-we-box']},
        'cond-table': {'out': True, 'charts': ['ct-chart', 'ct-tab']}, 'rolling': {'out': True, 'charts': ['rv-chart']}, 'stress-table': {'out': True, 'charts': ['sx-chart', 'sx-tab']},
        'vc-we': {'out': True}, 'corr': {'out': True, 'charts': ['rc-chart']}, 'tail-dep': {'out': True, 'charts': ['td-chart']}, 'copula-we': {'out': True},
        'copula': {'out': True, 'charts': ['cp-chart']}, 'port-table': {'out': True, 'charts': ['pt-chart', 'pt-tab']}, 'binom': {'out': True}, 'kupiec-we': {'out': True},
        'kupiec-sp': {'out': True}, 'chris-we': {'out': True, 'charts': ['cr-chart', 'cr-we-box']}, 'basel': {'out': True, 'charts': ['bs-tab', 'bs-we1-box']},
        'hits': {'out': True, 'charts': ['ht-chart']}, 'bt-sd': {'out': True}, 'bt-bb': {'out': True, 'charts': ['b2-tab']}, 'score': {'out': True, 'charts': ['sc-chart', 'sc-tab']},
        'traffic': {'out': True, 'charts': ['tt-chart']}, 'es-bt': {'out': True, 'charts': ['eb-chart', 'eb-we-box']}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('O divizie raportează „VaR 1% = −2,4%”. Ce este greșit?', 'A desk reports "VaR 1% = −2.4%". What is wrong?'),
      key=0, opts=[
  (('Semnul: −2,4% este cuantila q₀,₀₁; VaR 1% = −q₀,₀₁ = 2,4%.', 'The sign: −2.4% is the quantile q₀.₀₁; VaR 1% = −q₀.₀₁ = 2.4%.'),
   ('VaR este o pierdere pozitivă: semnul minus din definiție transformă cuantila negativă într-o pierdere.', 'VaR is a positive loss: the minus sign in the definition turns the negative quantile into a loss.')),
  (('Nivelul: trebuia raportat VaR 99%.', 'The level: it should report VaR 99%.'),
   ('Numărul din denumire este probabilitatea cozii; „VaR 99%” amestecă convențiile.', 'The number in the name is the tail probability; "VaR 99%" mixes the conventions.')),
  (('Nimic: un VaR negativ înseamnă că divizia cîștigă sigur.', 'Nothing: a negative VaR means the desk surely gains.'),
   ('Un VaR negativ ar însemna că și în cea mai proastă zi din 100 poziția cîștigă; aici −2,4% este o pierdere scrisă cu semnul randamentului.', 'A negative VaR would mean the position gains even on the worst day in 100; here −2.4% is a loss written with the sign of a return.'))]),
 dict(q=('Un model are 3 depășiri în 250 de zile, toate în aceeași săptămînă. Ce test detectează problema?', 'A model has 3 exceptions in 250 days, all in the same week. Which test detects the problem?'),
      key=0, opts=[
  (('Testul de independență Christoffersen, dacă depășirile cad în zile consecutive; cu 3 depășiri puterea lui este mică.', "Christoffersen's independence test, if the exceptions fall on consecutive days; with 3 exceptions its power is low."),
   ('Testul privește doar trecerile de la o zi la alta; depășiri în aceeași săptămînă, dar cu zile fără depășire între ele, nu schimbă π̂₁₁.', 'The test looks only at day-to-day transitions; exceptions in the same week but with days without exception in between do not change π̂₁₁.')),
  (('Testul Kupiec: 3 depășiri într-o săptămînă sînt prea multe.', 'The Kupiec test: 3 exceptions in one week are too many.'),
   ('Kupiec numără depășirile pe toată perioada: 3 este aproape de 2,5 așteptate și nu este respins.', 'Kupiec counts the exceptions over the whole period: 3 is close to the 2.5 expected and is not rejected.')),
  (('Semaforul Basel: 3 depășiri duc modelul în zona galbenă.', 'The Basel traffic light: 3 exceptions put the model in the yellow zone.'),
   ('Zona galbenă începe la 5 depășiri; 3 este în zona verde.', 'The yellow zone starts at 5 exceptions; 3 is in the green zone.'))]),
 dict(q=('Două portofolii au același VaR 1%. Pot avea ES 2,5% diferit?', 'Two portfolios have the same VaR 1%. Can they have a different ES 2.5%?'),
      key=0, opts=[
  (('Da: ES depinde de întreaga coadă, VaR doar de un punct.', 'Yes: ES depends on the whole tail, VaR only on one point.'),
   ('O pierdere de 4% și una de 40% dincolo de VaR dau același VaR, dar ES diferit.', 'A loss of 4% and one of 40% beyond VaR give the same VaR but a different ES.')),
  (('Nu: ES se obține din VaR printr-un factor fix.', 'No: ES follows from VaR by a fixed factor.'),
   ('Factorul fix (1,005 pentru ES 2,5% / VaR 1%) este valabil doar pentru distribuția Normală cu media 0.', 'The fixed factor (1.005 for ES 2.5% / VaR 1%) holds only for the Normal distribution with mean 0.')),
  (('Doar dacă au volatilități diferite.', 'Only if they have different volatilities.'),
   ('Două distribuții cu aceeași volatilitate și același VaR pot avea cozi diferite dincolo de cuantilă.', 'Two distributions with the same volatility and the same VaR can have different tails beyond the quantile.'))]),
 dict(q=('n = 250, x = 7, LR_uc = 5,50. Ce concluzie dă testul Kupiec la 5%?', 'n = 250, x = 7, LR_uc = 5.50. What does the Kupiec test conclude at 5%?'),
      key=0, opts=[
  (('Respingem: 5,50 > 3,84, valoarea critică a lui χ²(1).', 'Reject: 5.50 > 3.84, the critical value of χ²(1).'), ('LR_uc are o distribuție χ²(1) sub H₀; p = 0,019.', 'LR_uc is χ²(1) under H₀; p = 0.019.')),
  (('Nu respingem: 5,50 < 5,99.', 'Do not reject: 5.50 < 5.99.'), ('5,99 este valoarea critică a lui χ²(2), pentru LR_cc, nu pentru LR_uc.', '5.99 is the critical value of χ²(2), for LR_cc, not for LR_uc.')),
  (('Nu respingem: 7 depășiri sînt încă în zona galbenă, nu în cea roșie.', 'Do not reject: 7 exceptions are still in the yellow zone, not in the red one.'),
   ('Zonele Basel sînt o regulă de capital, nu testul Kupiec; testul respinge la 5% orice x în afara intervalului 1–6.', 'The Basel zones are a capital rule, not the Kupiec test; the test rejects at 5% any x outside 1–6.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch10-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH10_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH10_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH10_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
