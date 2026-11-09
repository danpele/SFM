"""Build the interactive pages of Chapter 9: interactive/ch9.html (EN) and interactive/ch9_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch9.js, inputs in interactive/ch9_data.js (Quantlets/Ch_09/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch9.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 9
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_09'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('returns', 'sfm_ch9_returns', 'ep'), ('simulated', 'sfm_ch9_simulated', 'arch'), ('lik-arch1', 'sfm_ch9_lik_arch1', 'ml'), ('lik-garch', 'sfm_ch9_lik_garch', 'ml'),
          ('qq', 'sfm_ch9_qq', 'tails'), ('vol-sd', 'sfm_ch9_vol_sp500_dax', 'mk'), ('vol-bb', 'sfm_ch9_vol_bet_btc', 'mk'), ('persistence', 'sfm_ch9_persistence', 'mk'),
          ('nic', 'sfm_ch9_nic', 'asym'), ('acf-diag', 'sfm_ch9_acf_diag', 'diag'), ('term', 'sfm_ch9_term_structure', 'fc'), ('fc-eval', 'sfm_ch9_forecast_eval', 'fc'),
          ('var', 'sfm_ch9_var', 'var')]
# table and worked-example blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page);
# the EN deck has one acronym frame less, so its frame number is one lower (checked in frame_pages)
XBLOCKS = [('mix', 14, 'cm', 'returns'), ('arch1', 17, 'arch', 'returns'), ('archq', 19, 'arch', 'simulated'), ('kurt', 22, 'garch', 'simulated'),
           ('half-life', 23, 'garch', 'simulated'), ('step', 24, 'garch', 'simulated'), ('igarch', 25, 'garch', 'simulated'), ('step-table', 29, 'ml', 'lik-garch'),
           ('dist-table', 32, 'tails', 'lik-garch'), ('mk-table', 34, 'mk', 'qq'), ('longrun', 35, 'mk', 'qq'), ('gjr', 40, 'asym', 'persistence'),
           ('asym-table', 42, 'asym', 'nic'), ('diag-table', 45, 'diag', 'acf-diag'), ('ms-table', 46, 'sel', 'acf-diag'), ('fc-we', 48, 'fc', 'acf-diag'),
           ('var-we', 52, 'var', 'fc-eval'), ('exc', 54, 'var', 'var')]
ORDER = ['returns', 'mix', 'arch1', 'simulated', 'archq', 'kurt', 'half-life', 'step', 'igarch', 'lik-arch1', 'lik-garch', 'step-table', 'dist-table', 'qq', 'mk-table',
         'longrun', 'vol-sd', 'vol-bb', 'persistence', 'gjr', 'nic', 'asym-table', 'acf-diag', 'diag-table', 'ms-table', 'fc-we', 'term', 'fc-eval', 'var-we', 'var', 'exc']
TITLES = {'mix': 'Exemplu rezolvat: varianța condiționată și varianța necondiționată', 'arch1': 'ARCH(1): proprietăți', 'archq': 'ARCH(q) și limitele lui',
          'kurt': 'Staționaritate, varianța pe termen lung și boltire', 'half-life': 'Persistența și timpul de înjumătățire', 'step': 'Exemplu rezolvat: un pas GARCH(1,1)',
          'igarch': 'IGARCH și legătura cu EWMA', 'step-table': 'Estimarea pas cu pas și cu un pachet', 'dist-table': 'S&P 500: trei distribuții ale inovațiilor',
          'mk-table': 'Estimările GARCH(1,1)-t', 'longrun': 'Interpretarea rezultatelor: Bitcoin și volatilitatea pe termen lung', 'gjr': 'GJR-GARCH și EGARCH',
          'asym-table': 'Asimetria pe șase piețe', 'diag-table': 'Diagnostic pe date reale', 'ms-table': 'Nouă modele pentru S&P 500',
          'fc-we': 'Exemplu rezolvat: S&P 500 la 18 septembrie 2026', 'var-we': 'VaR condiționat', 'exc': 'Depășiri 2015–2026: o primă privire asupra backtesting-ului'}

PFX = {'returns': 'rt', 'mix': 'mx', 'arch1': 'a1', 'simulated': 'sm', 'archq': 'aq', 'kurt': 'ku', 'half-life': 'hl', 'step': 'st', 'igarch': 'ig', 'lik-arch1': 'l1',
       'lik-garch': 'lg', 'step-table': 'et', 'dist-table': 'dt', 'qq': 'qq', 'mk-table': 'mt', 'longrun': 'lr', 'vol-sd': 'v1', 'vol-bb': 'v2', 'persistence': 'ps',
       'gjr': 'gj', 'nic': 'ni', 'asym-table': 'at', 'acf-diag': 'ad', 'diag-table': 'dg', 'ms-table': 'ms', 'fc-we': 'fw', 'term': 'ts', 'fc-eval': 'fe', 'var-we': 'vw',
       'var': 'vr', 'exc': 'ex'}

SECTIONS = {
 'ro': dict(ep='Volatility clustering', cm='Media condiționată și varianța condiționată', arch='Modelul ARCH', garch='Modelul GARCH(1,1)',
            ml='Estimarea prin verosimilitate maximă', tails='Inovații cu cozi groase', mk='GARCH pe șase piețe', asym='Asimetria și efectul de levier',
            diag='Diagnostic', sel='Alegerea modelului', fc='Prognoza volatilității', var='VaR 1% dintr-un model GARCH'),
 'en': dict(ep='Volatility Comes in Episodes', cm='Conditional Mean and Conditional Variance', arch='The ARCH Model', garch='The GARCH(1,1) Model',
            ml='Estimation by Maximum Likelihood', tails='Heavy-Tailed Innovations', mk='GARCH in Six Markets', asym='Asymmetry and the Leverage Effect',
            diag='Diagnostics', sel='Model Selection', fc='Forecasting Volatility', var='VaR 1% from a GARCH Model'),
}

L = {
 'ro': dict(file='ch9_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 9 · Modele ARCH și GARCH', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: ARCH (autoregressive conditional heteroskedasticity, heteroscedasticitate condiționată autoregresivă), '
        'GARCH (generalised ARCH, ARCH generalizat), GJR (Glosten–Jagannathan–Runkle), EGARCH (exponential GARCH, GARCH exponențial), IGARCH (integrated GARCH, GARCH integrat), '
        'EWMA (exponentially weighted moving average, media mobilă ponderată exponențial), SE (standard error, eroarea standard), LR (likelihood ratio, raportul de verosimilitate), '
        'AIC și BIC (criteriile informaționale Akaike și bayesian), ACF (autocorrelation function, funcția de autocorelație), ARCH-LM (testul multiplicatorului Lagrange pentru efecte ARCH), '
        'QLIKE (quasi-likelihood, cvasi-verosimilitate), DM (Diebold–Mariano), VaR (value at risk, valoarea expusă la risc).',
   src='Randamente logaritmice zilnice, în %: S&P 500, DAX și BET din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV) și OMV Petrom (SNP) '
       'din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. Prețuri EODHD. Modelele sînt estimate cu pachetul arch, '
       'cu codul din curs (verosimilitate maximă, SE robuste); din parametrii estimați, browserul reconstruiește varianța condiționată zi cu zi și calculează din ea reziduurile '
       'standardizate, graficele QQ, curbele de impact al știrilor, testele de diagnostic, criteriile informaționale, prognozele și VaR. Prognozele în afara eșantionului ale GARCH-t și GJR-t '
       '(reestimate la fiecare 250 de zile) sînt cele din curs; EWMA, pierderile QLIKE, testele DM și depășirile sînt recalculate în browser. Simulările folosesc extragerile Normale ale '
       'generatorului din curs (aceeași sămînță). Alte distribuții ale inovațiilor, alte începuturi ale eșantionului, alte piețe pentru tabelele S&P 500, ARCH(q) cu alte ordine, alte '
       'semințe și alți parametri ai simulărilor, alte niveluri ale VaR și alte perioade de evaluare sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch9.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 9 · ARCH and GARCH Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: ARCH (autoregressive conditional heteroskedasticity), GARCH (generalised ARCH), '
        'GJR (Glosten–Jagannathan–Runkle), EGARCH (exponential GARCH), IGARCH (integrated GARCH), EWMA (exponentially weighted moving average), SE (standard error), '
        'LR (likelihood ratio), AIC and BIC (Akaike and Bayesian information criteria), ACF (autocorrelation function), ARCH-LM (Lagrange multiplier test for ARCH effects), '
        'QLIKE (quasi-likelihood), DM (Diebold–Mariano), VaR (value at risk).',
   src='Daily log returns, in %: S&P 500, DAX and BET from January 2000, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV) and OMV Petrom (SNP) from 2010, '
       'from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. EODHD prices. The models are estimated with the arch package, with the '
       'code of the lecture (maximum likelihood, robust SE); from the fitted parameters the browser rebuilds the conditional variance day by day and computes from it the '
       'standardised residuals, the QQ plots, the news impact curves, the diagnostic tests, the information criteria, the forecasts and the VaR. The out-of-sample forecasts of '
       'GARCH-t and GJR-t (re-estimated every 250 days) are those of the lecture; the EWMA, the QLIKE losses, the DM tests and the exceedances are recomputed in the browser. The '
       'simulations use the Normal draws of the lecture\'s generator (the same seed). Other innovation distributions, other sample starts, other markets for the S&P 500 tables, '
       'ARCH(q) of other orders, other seeds and parameters of the simulations, other VaR levels and other evaluation periods are extra runs of the same code.'),
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
    M6 = [('sp500', 'S&P 500' + lec), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)')]
    M4 = [('sp500', 'S&P 500' + lec), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
    ser = 'Seria' if ro else 'Series'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    DI = [('t', 'Student-t' + lec), ('n', 'Normale' if ro else 'Normal'), ('st', 't asimetrică' if ro else 'skewed t')]
    dist = 'Distribuția inovațiilor' if ro else 'Distribution of the innovations'
    LEV = [('0.01', '1%' + lec), ('0.025', n('2.5%')), ('0.05', '5%')]
    lev = 'Nivelul VaR' if ro else 'VaR level'
    Y5 = [(str(y), str(y) + (lec if y == 2015 else '')) for y in (2015, 2017, 2019, 2021, 2023)]
    START = 'Începutul eșantionului' if ro else 'Start of the sample'
    ST4 = [('0', ('2000 (Bitcoin: 2014)' if ro else '2000 (Bitcoin: 2014)') + lec), ('1', '2005 (Bitcoin: 2016)'), ('2', '2010 (Bitcoin: 2018)'), ('3', '2015 (Bitcoin: 2020)')]
    MOD4 = [('garch:t', 'GARCH(1,1)-t' + lec), ('garch:n', 'GARCH(1,1)-N'), ('gjr:t', 'GJR-GARCH(1,1)-t'), ('egarch:t', 'EGARCH(1,1)-t')]
    return {
     'returns': (select('rt-s', ser, M6, 'sp500') + slider('rt-y', 'Anul liniștit de comparație' if ro else 'Calm year for the comparison', 2000, 2026, 1, 2017) + play('rt'), CSS + charts('rt-chart')),
     'mix': (slider('mx-s', 'Varianța zilelor agitate σ²' if ro else 'Variance of the stormy days σ²', 1, 10, 0.5, 4.5)
             + slider('mx-p', 'Ponderea zilelor agitate' if ro else 'Share of the stormy days', 0.05, 0.95, 0.05, 0.5) + play('mx'),
             charts('mx-chart') + we('mx-we', f'{WE}: {"legea varianței totale" if ro else "the law of total variance"}',
                                     r'\mathrm{Var}(r_t) = E[\mathrm{Var}(r_t \mid \mathcal{F}_{t-1})] + \mathrm{Var}(E[r_t \mid \mathcal{F}_{t-1}]), \qquad K = \frac{3\,E[\sigma_t^4]}{(E[\sigma_t^2])^2}',
                                     select('mx-c', 'Varianța zilelor liniștite σ²' if ro else 'Variance of the calm days σ²', [('0.5', n('0.5') + lec), ('0.25', n('0.25')), ('1', '1')], '0.5'))),
     'arch1': (slider('a1-a', 'α al ARCH(1)' if ro else 'α of the ARCH(1)', 0, 0.95, 0.05, 0.5) + slider('a1-e', 'Șocul de ieri ε(t − 1)' if ro else "Yesterday's shock ε(t − 1)", -4, 4, 0.5, 2) + play('a1'),
               charts('a1-chart') + we('a1-we', f'{WE}: {"varianța necondiționată, boltirea și un pas ARCH(1)" if ro else "unconditional variance, kurtosis and one ARCH(1) step"}',
                                       c(r'\bar\sigma^2 = \frac{\omega}{1-\alpha}, \qquad K = 3\,\frac{1-\alpha^2}{1-3\alpha^2}, \qquad \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2'),
                                       select('a1-w', 'ω', [('0.5', n('0.5') + lec), ('0.25', n('0.25')), ('1', '1')], '0.5'))),
     'simulated': (select('sm-s', 'Sămînța generatorului' if ro else 'Generator seed', [('2026', '2026' + lec), ('2036', '2036'), ('2046', '2046'), ('2056', '2056'), ('2066', '2066'), ('2076', '2076')], '2026')
                   + slider('sm-a', 'α al ARCH(1), cu ω = 1 − α' if ro else 'α of the ARCH(1), with ω = 1 − α', 0.05, 0.95, 0.05, 0.5)
                   + slider('sm-b', 'β al GARCH(1,1), cu α = 0,10 și ω = 0,9 − β' if ro else 'β of the GARCH(1,1), with α = 0.10 and ω = 0.9 − β', 0.5, 0.89, 0.01, 0.88) + play('sm'), charts('sm-chart')),
     'archq': (select('aq-s', ser, M6, 'sp500') + slider('aq-q', 'Ordinul q din rîndul al doilea' if ro else 'Order q of the second row', 2, 9, 1, 5) + play('aq'),
               charts('aq-chart') + xh('Modele cu inovații Normale (cu verde: cel mai mic AIC și BIC)' if ro else 'Models with Normal innovations (in green: the smallest AIC and BIC)') + tw('aq-tab')),
     'kurt': (slider('ku-a', 'α', 0.02, 0.3, 0.01, 0.1) + slider('ku-b', 'β', 0.5, 0.97, 0.01, 0.88) + play('ku'),
              charts('ku-chart') + we('ku-we', f'{WE}: {"varianța pe termen lung și boltirea" if ro else "long-run variance and kurtosis"}',
                                      r'\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}, \qquad K = 3\,\frac{1-(\alpha+\beta)^2}{1-(\alpha+\beta)^2-2\alpha^2}',
                                      select('ku-w', 'ω', [('0.02', n('0.02') + lec), ('0.01', n('0.01')), ('0.05', n('0.05'))], '0.02'))),
     'half-life': (slider('hl-p', 'Persistența α + β' if ro else 'Persistence α + β', 0.8, 0.999, 0.0001, 0.98) + play('hl'),
                   charts('hl-chart') + we('hl-we', f'{WE}: {"timpul de înjumătățire" if ro else "the half-life"}',
                                           c(r'(\alpha+\beta)^{h} = \tfrac12 \;\Rightarrow\; h_{1/2} = \frac{\ln 0.5}{\ln(\alpha+\beta)} \approx \frac{\ln 2}{1-\alpha-\beta}'))),
     'step': (slider('st-r', 'Randamentul de azi r(t) (%)' if ro else "Today's return r(t) (%)", -6, 6, 0.5, -3) + slider('st-s', 'Varianța de azi σ²(t)' if ro else "Today's variance σ²(t)", 0.5, 4, 0.1, 1.2) + play('st'),
              charts('st-chart') + we('st-we', f'{WE}: {"un pas GARCH(1,1)" if ro else "one GARCH(1,1) step"}', r'\sigma_{t+1}^2 = \omega + \alpha\,\varepsilon_t^2 + \beta\,\sigma_t^2, \qquad \varepsilon_t = r_t - \mu',
                                      select('st-p', 'Parametrii' if ro else 'Parameters', [('deck', ('ω = 0,02, α = 0,10, β = 0,88, μ = 0' if ro else 'ω = 0.02, α = 0.10, β = 0.88, μ = 0') + lec),
                                                                                            ('sp500', 'S&P 500, GARCH(1,1)-t')], 'deck'))),
     'igarch': (slider('ig-l', 'λ al EWMA' if ro else 'EWMA λ', 0.9, 0.99, 0.01, 0.94)
                + select('ig-w', 'ω al IGARCH' if ro else 'ω of the IGARCH', [('0', '0 (EWMA)' + lec), ('0.0157', n('0.0157') + (' (ω al S&P 500)' if ro else ' (ω of the S&P 500)')),
                                                                             ('0.194', n('0.194') + (' (ω al Bitcoin)' if ro else ' (ω of Bitcoin)'))], '0') + play('ig'),
                charts('ig-chart') + we('ig-we', f'{WE}: {"prognozele IGARCH și GARCH după un crah" if ro else "IGARCH and GARCH forecasts after a crash"}',
                                        r'\alpha+\beta = 1: \quad E_t[\sigma_{t+h}^2] = \sigma_{t+1}^2 + (h-1)\,\omega; \qquad \alpha+\beta < 1: \quad E_t[\sigma_{t+h}^2] = \bar\sigma^2 + (\alpha+\beta)^{h-1}(\sigma_{t+1}^2 - \bar\sigma^2)')),
     'lik-arch1': (slider('l1-a', 'α adevărat al simulării (ω = 1 − α)' if ro else 'True α of the simulation (ω = 1 − α)', 0.1, 0.9, 0.05, 0.5) + play('l1'), charts('l1-chart')),
     'lik-garch': (select('lg-s', 'Sămînța generatorului' if ro else 'Generator seed', [('2033', '2033' + lec), ('2043', '2043'), ('2053', '2053'), ('2063', '2063'), ('2073', '2073'), ('2083', '2083')], '2033')
                   + select('lg-n', 'n în panoul din stînga' if ro else 'n of the left panel', [('250', '250'), ('500', '500' + lec), ('1000', '1000')], '500') + play('lg'), charts('lg-chart')),
     'step-table': (select('et-s', ser, M6, 'sp500') + play('et'),
                    charts('et-chart') + xh('GARCH(1,1) cu inovații Normale: pas cu pas (scipy) și cu pachetul arch' if ro else 'GARCH(1,1) with Normal innovations: step by step (scipy) and with the arch package') + tw('et-tab')),
     'dist-table': (select('dt-s', ser, M6, 'sp500') + play('dt'),
                    charts('dt-chart') + xh('GARCH(1,1) cu trei distribuții ale inovațiilor' if ro else 'GARCH(1,1) with three innovation distributions') + tw('dt-tab')),
     'qq': (select('qq-s', ser, M6, 'sp500') + play('qq'), charts('qq-chart')),
     'mk-table': (select('mt-d', dist, DI, 't') + play('mt'),
                  charts('mt-chart') + xh('GARCH(1,1) pe șase piețe (SE robuste; volatilitățile anualizate, în %)' if ro else 'GARCH(1,1) in six markets (robust SE; annualised volatilities, in %)') + tw('mt-tab')),
     'longrun': (select('lr-s', ser, M6, 'sp500') + play('lr'),
                 charts('lr-chart') + we('lr-we', f'{WE}: {"volatilitatea pe termen lung" if ro else "the long-run volatility"}', r'\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}, \qquad \bar\sigma_{\text{an}} = \sqrt{A\,\bar\sigma^2}' if ro else r'\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}, \qquad \bar\sigma_{\text{year}} = \sqrt{A\,\bar\sigma^2}')),
     'vol-sd': (select('v1-d', dist, DI, 't') + select('v1-y', START, ST4, '0') + play('v1'), charts('v1-chart')),
     'vol-bb': (select('v2-d', dist, DI, 't') + select('v2-y', START, ST4, '0') + play('v2'), charts('v2-chart')),
     'persistence': (select('ps-m', 'Modelul' if ro else 'Model', MOD4, 'garch:t') + slider('ps-h', 'Orizontul (zile)' if ro else 'Horizon (days)', 50, 500, 10, 250) + play('ps'), charts('ps-chart')),
     'gjr': (slider('gj-e', 'Mărimea șocului |ε(t − 1)| (%)' if ro else 'Size of the shock |ε(t − 1)| (%)', 0.5, 5, 0.5, 2) + select('gj-s', ser, M6, 'sp500') + play('gj'),
             charts('gj-chart') + we('gj-we', f'{WE}: {"impactul unei scăderi și al unei creșteri" if ro else "the impact of a fall and of a rise"}',
                                     r'\sigma_t^2 = \omega + (\alpha + \gamma\,I_{t-1})\,\varepsilon_{t-1}^2 + \beta\,\bar\sigma^2, \qquad I_{t-1} = 1 \text{ dacă } \varepsilon_{t-1} < 0' if ro else
                                     r'\sigma_t^2 = \omega + (\alpha + \gamma\,I_{t-1})\,\varepsilon_{t-1}^2 + \beta\,\bar\sigma^2, \qquad I_{t-1} = 1 \text{ if } \varepsilon_{t-1} < 0')),
     'nic': (select('ni-s', 'Seria din panoul din dreapta' if ro else 'Series of the right panel', [('btc', 'Bitcoin' + lec), ('dax', 'DAX'), ('bet', 'BET'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)')], 'btc')
             + slider('ni-v', 'σ²(t − 1), în multipli ai varianței de selecție' if ro else 'σ²(t − 1), in multiples of the sample variance', 0.25, 4, 0.25, 1) + play('ni'), charts('ni-chart')),
     'asym-table': (select('at-d', dist, DI, 't') + play('at'),
                    charts('at-chart') + xh('GJR-GARCH(1,1) și EGARCH(1,1): parametrul de asimetrie γ (statistici t cu SE robuste)' if ro else 'GJR-GARCH(1,1) and EGARCH(1,1): the asymmetry parameter γ (t statistics with robust SE)') + tw('at-tab')),
     'acf-diag': (select('ad-s', ser, M6, 'sp500') + select('ad-m', 'Modelul reziduurilor' if ro else 'Model of the residuals', MOD4, 'garch:t')
                  + slider('ad-k', 'Laguri' if ro else 'Lags', 10, 100, 10, 50) + play('ad'), charts('ad-chart')),
     'diag-table': (select('dg-s', ser, M6, 'sp500') + slider('dg-m', 'Laguri Ljung–Box m' if ro else 'Ljung–Box lags m', 5, 30, 1, 10) + play('dg'),
                    charts('dg-chart') + xh('Teste pe randamente și pe reziduurile standardizate' if ro else 'Tests on the returns and on the standardised residuals') + tw('dg-tab')),
     'ms-table': (select('ms-s', ser, M6, 'sp500') + select('ms-c', 'Criteriul' if ro else 'Criterion', [('bic', 'BIC' + lec), ('aic', 'AIC')], 'bic') + play('ms'),
                  charts('ms-chart') + xh('Log-verosimilitatea și diferența față de cel mai bun model (cu verde)' if ro else 'Log-likelihood and the difference from the best model (in green)') + tw('ms-tab')),
     'fc-we': (slider('fw-h', 'Orizontul H (zile)' if ro else 'Horizon H (days)', 1, 250, 1, 10)
               + select('fw-d', 'Ziua prognozei' if ro else 'Forecast date', [('2026-09-18', ('18 septembrie 2026' if ro else '18 September 2026') + lec), ('2020-03-16', '16 martie 2020' if ro else '16 March 2020'),
                                                                              ('2017-11-03', '3 noiembrie 2017' if ro else '3 November 2017')], '2026-09-18') + play('fw'),
               charts('fw-chart') + we('fw-we', f'{WE}: {"prognoza pe H zile" if ro else "the H-day forecast"}',
                                       r'E_t[\sigma_{t+h}^2] = \bar\sigma^2 + (\alpha+\beta)^{h-1}(\sigma_{t+1}^2 - \bar\sigma^2), \qquad \mathrm{Var}_t\Big(\sum_{h=1}^{H} r_{t+h}\Big) = \sum_{h=1}^{H} E_t[\sigma_{t+h}^2]')),
     'term': (select('ts-m', 'Modelul' if ro else 'Model', [('t', 'GARCH(1,1)-t' + lec), ('n', 'GARCH(1,1)-N')], 't')
              + slider('ts-d', 'Ziua celei de-a treia prognoze' if ro else 'Date of the third forecast', 250, 6713, 1, 6713) + play('ts'), charts('ts-chart')),
     'fc-eval': (select('fe-y', 'Începutul evaluării' if ro else 'Start of the evaluation', Y5, '2015') + slider('fe-l', 'λ al EWMA' if ro else 'EWMA λ', 0.9, 0.99, 0.01, 0.94) + play('fe'),
                 charts('fe-chart') + xh('Prognoze ale varianței pentru ziua următoare (cu verde: cel mai mic QLIKE; DM: t și p)' if ro else 'Next-day variance forecasts (in green: the smallest QLIKE; DM: t and p)') + tw('fe-tab')),
     'var-we': (select('vw-a', lev, LEV, '0.01') + slider('vw-h', 'Orizontul H (zile)' if ro else 'Horizon H (days)', 1, 20, 1, 10) + play('vw'),
                charts('vw-chart') + we('vw-we', f'{WE}: {"VaR pe o zi și pe H zile" if ro else "one-day and H-day VaR"}',
                                        r'\mathrm{VaR}_{t+1} = -\big(\mu + \sigma_{t+1}\,q_\alpha(z)\big), \qquad q_\alpha(z) = t_\nu^{-1}(\alpha)\sqrt{\tfrac{\nu-2}{\nu}}')),
     'var': (select('vr-s', ser, M4, 'sp500') + select('vr-w', 'Fereastra de doi ani începe în iulie' if ro else 'The two-year window starts in July', [(str(y), str(y) + (lec if y == 2019 else '')) for y in range(2015, 2025)], '2019')
             + select('vr-a', lev, LEV, '0.01') + play('vr'), charts('vr-chart')),
     'exc': (select('ex-a', lev, LEV, '0.01') + select('ex-y', 'Începutul evaluării' if ro else 'Start of the evaluation', Y5, '2015') + play('ex'),
             charts('ex-chart') + xh('Depășiri ale VaR pe o zi, în afara eșantionului' if ro else 'Exceedances of the one-day VaR, out of sample') + tw('ex-tab')),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    ro = frames_of(open(os.path.join(ROOT, TEX['ro']) + '.tex', encoding='utf-8').read())
    off = len(ro) - len(frames)
    res = {}
    for a, fr, _, _ in XBLOCKS:
        assert latex_text(ro[fr - 1][0]) == TITLES[a], (a, latex_text(ro[fr - 1][0]))
        k = fr - 1 - off
        assert not re.search(r'includegraphics[^{]*\{sfm_', frames[k][1]), (lang, a)
        assert frames[k][1].count('tabular') == ro[fr - 1][1].count('tabular'), (lang, a)
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch9.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch9_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch9_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => `${s} ${de(s)}zile`;
const X = ' (rulare suplimentară a codului din curs)';
const lst = a => (a.length ? a.join(', ') : 'niciuna');
window.CH9_TEXT = {
 h: { days: 'zile', dm1: 'DM t, GARCH-t față de EWMA (p)', dm2: 'DM t, GJR-t față de GARCH-t (p)', normalz: 'z(t) Normale', params: 'parametri', ll: 'log-verosimilitate',
      sumalpha: 'Σα (+β)', step: 'pas cu pas (SE)', secl: 'SE clasică', serob: 'SE robustă', ratio: 'raportul robustă/clasică', dn: 'Normale', dst: 't asimetrică',
      from: 'din', hl: 'înjumătățire (zile)', lrvol: 'vol. termen lung', svol: 'vol. selecție', bicbest: 'BIC alege', sq: 'al pătratelor', sb: 'sign bias', ret: 'randamente r(t)',
      zn: 'ẑ(t), GARCH Normal', zt: 'ẑ(t), GARCH-t', expected: 'așteptat', dist: { n: 'Normale', t: 'Student-t', st: 't asimetrică' } },
 rt: (s, y, n, sy, s08, s20, sa, r08, r20, mn, dmn, mx, dmx) => `${s}, ${n} ${de(n)}zile: abaterea standard zilnică ${sy === null ? `nu se poate calcula în ${y} (seria începe mai tîrziu)` : `${sy}% în ${y}`}; `
   + (s08 === null ? '' : `${s08}% din septembrie 2008 pînă în martie 2009${r08 === null ? '' : ` (de ${r08} ori mai mare)`}; `) + `${s20}% din februarie pînă în mai 2020${r20 === null ? '' : ` (de ${r20} ori mai mare)`}; ${sa}% pe întregul eșantion. `
   + `Cea mai mare scădere: ${mn}% (${dmn}); cea mai mare creștere: ${mx}% (${dmx}).`,
 sm: (seed, xs, k0, k1, k2, t0, t1, t2, m1, m2, xp) => `Sămînța ${seed}${xs || xp ? X : ''}: coeficientul de boltire de selecție i.i.d. ${k0}, ARCH(1) ${k1}, GARCH(1,1) ${k2}; teoretic ${t0}, ${t1} și ${t2}. `
   + `Cea mai mare valoare absolută: ARCH ${m1}, GARCH ${m2}. Toate cele trei serii au varianța necondiționată 1.`,
 l1: (a0, items, x, r) => `α adevărat ${a0}${x ? X : ''}: ` + items.map(([n, a, s]) => `n = ${n}: α̂ = ${a} (eroarea standard ${s})`).join('; ') + `. Eroarea standard scade de ${r} ori cînd n crește de 10 ori (regula √n: 3,16).`,
 lg: (seed, x, items) => `Sămînța ${seed}${x ? X : ''}: ` + items.map(([n, a, b, d]) => `n = ${n}: maximul pe grilă în α = ${a}, β = ${b}; în valorile adevărate log-verosimilitatea este cu ${d} sub maxim`).join('; ') + '.',
 qq: (s, x, qn, thn, nu, qt, tht, raw) => `${s}${x ? X : ''}: cuantila de 0,1% a reziduurilor GARCH-N este ${qn}, față de ${thn} pentru distribuția Normală; reziduurile GARCH-t (ν̂ = ${nu}) au ${qt}, `
   + `față de ${tht} pentru t standardizată (= ${raw} × √((ν − 2)/ν)).`,
 vol: (d, x, items, r) => `Inovații ${d}${x ? X : ''}: ` + items.map(([s, f, p8, d8, p0, d0, md, l, sv]) => `${s} (din ${f}): ${p8 === null ? '' : `vîrf ${p8}% (${d8}), `}vîrf ${p0}% (${d0}), mediana ${md}%, ultima valoare ${l}%, volatilitatea de selecție ${sv}%`).join('; ')
   + `. Mediana din panoul de jos este de ${r} ori cea din panoul de sus.`,
 ps: (m, x, mm, items) => `${m}${x ? X : ''}; persistența ${mm === 'egarch' ? 'β (pentru ln σ²)' : mm === 'gjr' ? 'α + β + γ/2' : 'α + β'} și timpul de înjumătățire: `
   + items.map(([s, p, h]) => `${s} ${p}, ${h === null ? 'fără timp de înjumătățire (IGARCH)' : zile(h)}`).join('; ') + '.',
 ni: (m, x, items) => `σ²(t − 1) = ${m} × varianța de selecție${x ? X : ''}; raportul dintre varianța de după o scădere de 2% și cea de după o creștere de 2%: ` + items.map(([s, a, b]) => `${s} GJR ${a}, EGARCH ${b}`).join('; ') + '.',
 ad: (s, m, x, K, a11, a1K, a21, mx, b, o) => `${s}, ${m}${x ? X : ''}: ACF a lui r² este ${a11} la lagul 1 și ${a1K} la lagul ${K}; ACF a lui ẑ² este ${a21} la lagul 1 și cel mult ${mx} în valoare absolută, cu banda ±${b}; `
   + `${o} din ${K} ${de(K)}laguri ies din bandă.`,
 ts: (m, x, items, lr) => `${m}${x ? X : ''}, volatilitatea anualizată mîine, peste 22 de zile și media pe 250 de zile: ` + items.map(([d, a, b, c]) => `${d}: ${a}%, ${b}%, ${c}%`).join('; ') + `. Nivelul pe termen lung: ${lr}%.`,
 fe: (y, l, x, items) => `Evaluare din ${y}, EWMA cu λ = ${l}${x ? X : ''}; testul DM al QLIKE, GARCH-t față de EWMA: ` + items.map(([s, t, p, g]) => `${s} t = ${t} (p ${p}${g ? ', GARCH-t mai bun' : ', fără diferență detectabilă'})`).join('; ') + '.',
 vr: (s, a, b, n, lev, eg, ee, ex, x) => `${s}, ${a} – ${b} (${n} ${de(n)}zile), VaR ${lev}${x ? X : ''}: GARCH-t este depășit de ${eg} ori, EWMA-Normal de ${ee} ori; așteptat circa ${ex}.`,
 mx1: (c, s, q, p) => `Zile liniștite: σ² = ${c}, cu probabilitatea ${q}; zile agitate: σ² = ${s}, cu probabilitatea ${p}; media condiționată este 0.`,
 mx2: (q, c, p, s, v, sd) => `Var(r) = ${q} × ${c} + ${p} × ${s} + 0 = ${v}, deci abaterea standard necondiționată este √${v} = ${sd}%.`,
 mx3: (a, b, m, sd) => `Volatilitatea condiționată: ${a}% într-o zi liniștită, ${b}% într-o zi agitată; media lor ponderată, ${m}%, nu este ${sd}%: se mediază varianțele, nu abaterile standard.`,
 mx4: (e4, v2, K) => `Boltirea amestecului: 3 × ${e4}/${v2} = ${K}, peste 3, deși fiecare zi este Normală.`,
 mx: (v, sd, K, x) => `Varianța necondiționată ${v}, abaterea standard ${sd}%, boltirea ${K}${x ? ' (alți parametri decît în curs)' : ''}. Linia întreruptă: distribuția Normală cu aceeași varianță; amestecul are vîrful mai înalt și cozile mai groase.`,
 a1a: (w, a, q, s) => (s === null ? `α = ${a} ≥ 1: varianța necondiționată nu este finită.` : `Varianța necondiționată: σ̄² = ω/(1 − α) = ${w}/${q} = ${s}.`),
 a1b: (a2, n, d, K) => (K === null ? `3α² = ${d} ≥ 1: momentul de ordinul patru este infinit, boltirea nu există.` : `Boltirea: K = 3(1 − ${a2})/(1 − 3 × ${a2}) = 3 × ${n}/${d} = ${K}.`),
 a1c: (w, a, e, e2, s, sd) => `După șocul ε(t − 1) = ${e}: σ²(t) = ${w} + ${a} × ${e2} = ${s}, σ(t) = ${sd}.`,
 a1: (a, K, s, x) => `α = ${a}${x ? ' (alți parametri decît în curs)' : ''}: ${K === null ? 'boltirea este infinită (α ≥ 0,577)' : `boltirea ${K}`}; varianța de azi după șocul ales: ${s}.`,
 aq: (s, x, sa, pr, d10, qb, db) => `${s}${x ? X : ''}: ARCH(10) are Σα = ${sa}, GARCH(1,1) are α + β = ${pr}; BIC al ARCH(10) este cu ${d10} peste cel al GARCH(1,1). `
   + (qb === 10 ? 'BIC al ARCH(q) scade cu fiecare lag pînă la q = 10.' : `Cel mai bun ARCH(q) după BIC, q = ${qb}, rămîne cu ${db} peste GARCH(1,1).`),
 ku1: (a, b, p, st) => `Persistența α + β = ${a} + ${b} = ${p}${st ? ' &lt; 1: modelul este staționar în covarianță.' : ' ≥ 1: modelul nu este staționar în covarianță.'}`,
 ku2: (w, q, s, v) => (s === null ? 'Varianța necondiționată nu este finită.' : `σ̄² = ω/(1 − α − β) = ${w}/${q} = ${s}; volatilitatea anualizată √(252 × ${s}) = ${v}%.`),
 ku3: (p, a, c, K) => `(α + β)² + 2α² = ${p}² + 2 × ${a}² = ${c}` + (K === null ? ' ≥ 1: momentul de ordinul patru este infinit.' : `; K = 3(1 − ${p}²)/(1 − ${c}) = ${K}.`),
 ku: (a, b, K, v, x) => `α = ${a}, β = ${b}${x ? ' (alți parametri decît în curs)' : ''}: ${K === null ? 'boltirea este infinită' : `boltirea ${K}`}; ${v === null ? 'varianța pe termen lung nu este finită' : `volatilitatea pe termen lung ${v}%`}.`,
 hl1: (p, lp, l5, h) => `h½ = ln 0,5/ln(α + β) = ${l5}/${lp} = ${h} zile (α + β = ${p}).`,
 hl2: (q, h) => `Aproximarea pentru α + β aproape de 1: ln 2/(1 − α − β) = 0,6931/${q} = ${h} zile.`,
 hl3: (a, b, c, p, h) => `Pentru comparație: 0,90 dă ${a} zile, 0,98 dă ${b}, 0,99 dă ${c}; S&P 500, GARCH(1,1)-t: α + β = ${p}, ${h} zile.`,
 hl: (p, h, s, x) => `α + β = ${p}${x ? ' (altă valoare decît în exemplul din curs)' : ''}: timpul de înjumătățire ${h} zile; după 21 de zile rămîne ${s} din șoc.`,
 st1: (sp, w, a, b, m) => (sp ? `Parametrii GARCH(1,1)-t estimați pentru S&P 500: ω = ${w}, α = ${a}, β = ${b}, μ = ${m}.` : `Parametrii din curs: ω = ${w}, α = ${a}, β = ${b}, μ = ${m}.`),
 st2: (p, w, q, s, v, h) => `Persistența α + β = ${p}; σ̄² = ${w}/${q} = ${s}; volatilitatea anualizată √(252 × ${s}) = ${v}%; timpul de înjumătățire ${h} zile.`,
 st3: (s2, r, e, w, a, e2, b, s1, sd) => `Azi σ²(t) = ${s2} și r(t) = ${r}%, deci ε(t) = ${e}; mîine: σ²(t + 1) = ${w} + ${a} × ${e2} + ${b} × ${s2} = ${s1}, σ(t + 1) = ${sd}%.`,
 st4: (s1, sb, p, d10) => `Apoi prognoza se apropie de ${sb} cu factorul ${p} pe zi: în ziua a zecea ${sb} + ${p}⁹ × (${s1} − ${sb}) = ${d10}.`,
 st: (r, s2, s1, sd, s0, x) => `r(t) = ${r}%, σ²(t) = ${s2}${x ? ' (alte valori decît în curs)' : ''}: σ²(t + 1) = ${s1} (σ = ${sd}%); fără șoc ar fi ${s0}. Linia roșie: prognozele care urmează, spre varianța pe termen lung.`,
 ig1: (l, a) => `EWMA cu λ = ${l} este un IGARCH cu ω = 0, α = ${a}, β = ${l}: α + β = 1.`,
 ig2: (l, s, v) => `S&P 500, la închiderea din 16 martie 2020: varianța EWMA (λ = ${l}) pentru ziua următoare σ²(t + 1) = ${s}, adică ${v}% pe an.`,
 ig3: (w, s, s250, v) => `IGARCH cu ω = ${w}: E[σ²(t + 250)] = ${s} + 249 × ${w} = ${s250}, adică ${v}% pe an. Cu ω = 0 prognoza este aceeași pentru orice orizont.`,
 ig4: (s, sb, p, f, v) => `GARCH(1,1)-t: σ²(t + 1) = ${s}, σ̄² = ${sb}, α + β = ${p}; peste 250 de zile ${f}, adică ${v}% pe an.`,
 ig: (l, e, g, lr, x) => `λ = ${l}${x ? ' (altă valoare decît în curs)' : ''}: EWMA prognozează ${e}% pe an pentru orice orizont; GARCH(1,1)-t coboară la ${g}% după 250 de zile, spre ${lr}%.`,
 et: (s, x, l1, l2, ra, rb, rw, rm) => `${s}${x ? X : ''}: log-verosimilitatea ${l1} pas cu pas (σ²(1) = varianța de selecție) și ${l2} cu arch; SE robuste / SE clasice: α ${ra}, β ${rb}, ω ${rw}, μ ${rm}.`,
 dt: (s, x, l1, l2, nu, lm, an, at, bn, bt) => `${s}${x ? X : ''}: LR t față de Normal ${l1}, t asimetrică față de t ${l2} (valoarea critică 3,84); ν̂ = ${nu}, λ̂ = ${lm}; α ${an} → ${at}, β ${bn} → ${bt}.`,
 mt: (d, x, p0, p1, h0, h1, s0, s1, ba, a, ig) => `Inovații ${d}${x ? X : ''}: indicii au α + β între ${p0} și ${p1} și timpi de înjumătățire de ${h0}–${zile(h1)}; SE robuste ale lui α̂ și β̂ între ${s0} și ${s1}; `
   + `cel mai mare α: ${ba} (${a})${ig.length ? `; IGARCH (α + β ≥ 0,9995): ${ig.join(', ')}` : ''}.`,
 lr0: (s, p, w, b) => `${s}: α̂ + β̂ = ${p}, pe frontieră (IGARCH): 1 − α − β este practic 0, deci ω/(1 − α − β) nu este finit; nu există volatilitate pe termen lung. Cu ω = ${w} > 0, prognoza varianței crește cu ω pe zi; modelul se comportă ca un EWMA cu λ ≈ ${b}.`,
 lr1: (s, w, p, q, sb) => `${s}: ω = ${w}, α + β = ${p}; 1 − α − β = ${q}; σ̄² = ${w}/${q} = ${sb}.`,
 lr2: (A, sb, v, sv) => `Anualizat cu A = ${A} observații pe an: √(${A} × ${sb}) = ${v}%, față de volatilitatea de selecție ${sv}%.`,
 lr3: (p, q, v) => `Cu α + β = ${p} (cu 0,002 mai mic): 1 − α − β = ${q}, volatilitatea pe termen lung ar fi ${v}%.`,
 lr4: (v, p) => (v === null ? `GARCH cu inovații Normale: α + β = ${p}, tot pe frontieră.` : `GARCH cu inovații Normale (α + β = ${p}) dă ${v}%.`),
 lr: (s, ig, v, sv, x) => `${s}${x ? X : ''}: ${ig ? 'IGARCH, fără volatilitate pe termen lung' : `volatilitatea pe termen lung ${v}%`}, volatilitatea de selecție ${sv}%. Curba arată cît de repede crește nivelul pe termen lung cînd α + β se apropie de 1.`,
 gj1: (s, w, a, g, b, sb) => `${s}, GJR-GARCH(1,1)-t: ω = ${w}, α = ${a}, γ = ${g}, β = ${b}; σ²(t − 1) fixat la varianța de selecție ${sb}.`,
 gj2: (e, w, ag, e2, b, sb, v) => `Scădere de ${e}%: σ² = ${w} + ${ag} × ${e2} + ${b} × ${sb} = ${v}.`,
 gj3: (e, w, a, e2, b, sb, v) => `Creștere de ${e}%: σ² = ${w} + ${a} × ${e2} + ${b} × ${sb} = ${v}.`,
 gj4: (r, re) => `Raportul: ${r}; cu EGARCH(1,1)-t raportul este ${re}.`,
 gj: (s, e, r, re, x) => `${s}, șocuri de ±${e}%${x ? X : ''}: după scădere varianța de mîine este de ${r} ori cea de după creștere (EGARCH: ${re}).`,
 at: (d, x, sig, gb, lr, lt) => `Inovații ${d}${x ? X : ''}: γ al GJR pozitiv și semnificativ la 5% (testul LR): ${lst(sig)}; BIC preferă GARCH simetric: ${lst(gb)}. BET: LR = ${lr}, sub ln T = ${lt}, penalizarea BIC pentru un parametru în plus.`,
 dg: (s, m, x, q2, z2, pz2, st, psb, pz) => `${s}, m = ${m}${x ? X : ''}: Q(${m}) al pătratelor scade de la ${q2} (randamente) la ${z2} (ẑ, GARCH-t; p ${pz2}); Q(${m}) pe ẑ are p ${pz}; `
   + `testul de asimetrie: t al semnului ${st}, p comun ${psb}.`,
 ms: (s, c, x, bm, bd, da, dt, db) => `${s}, ${c}${x ? X : ''}: cel mai bun model este ${bm} cu inovații ${bd}. Față de GARCH cu inovații Normale, doar asimetria (EGARCH-Normal) reduce ${c} cu ${da}, `
   + `doar cozile groase (GARCH cu t asimetrică) cu ${dt}, cel mai bun model cu ${db}.`,
 fw1: (d, p, sb, s1) => `${d}, GARCH(1,1)-t: α + β = ${p}, σ̄² = ${sb}; prognoza pentru ziua următoare σ²(t + 1) = ${s1}.`,
 fw2: (H, sb, p, h1, ph, s1, f) => `Ziua ${H}: ${sb} + ${p}<sup>${h1}</sup> × (${s1} − ${sb}) = ${sb} + ${ph} × (${s1} − ${sb}) = ${f}.`,
 fw3: (H, s, v) => `Varianța randamentului pe ${zile(H)}: suma celor ${H} prognoze zilnice = ${s}, deci volatilitatea este √${s} = ${v}%.`,
 fw4: (H, s1, v, below) => `Regula rădăcinii pătrate a timpului: √(${H} × ${s1}) = ${v}%: ${below ? 'prea mică, pentru că volatilitatea este așteptată să crească spre nivelul pe termen lung' : 'prea mare, pentru că volatilitatea este așteptată să scadă spre nivelul pe termen lung'}.`,
 fw: (d, H, v, r, below, x) => `${d}, H = ${H}${x ? ' (alt orizont sau altă zi decît în curs)' : ''}: volatilitatea pe H zile din suma prognozelor ${v}%, regula rădăcinii pătrate ${r}% (${below ? 'subestimează' : 'supraestimează'}).`,
 vw1: (lev, m, s2, s, nu) => `S&P 500 la 18 septembrie 2026, GARCH(1,1)-t: μ̂ = ${m}, σ²(t + 1) = ${s2}, σ(t + 1) = ${s}, ν̂ = ${nu}.`,
 vw2: (lev, tq, sc, q) => `Cuantila de ${lev} a lui z: ${tq} × ${sc} = ${q} (cuantila Student-t rescalată la varianța 1).`,
 vw3: (lev, m, s, q, v, zq, vn) => `VaR ${lev} pe o zi: −(${m} + ${s} × (${q})) = ${v}%; cu z(t) Normal (cuantila ${zq}): ${vn}%.`,
 vw4: (H, q, s, v, r) => `Pe ${zile(H)} (suma varianțelor, fără μ): ${q} × √${s} = ${v}%; regula √H aplicată VaR-ului pe o zi dă ${r}%. Este o aproximare: randamentul pe ${zile(H)} nu are o distribuție Student-t cu același ν, iar cuantila lui exactă cere simulare.`,
 vw: (lev, v1, vn, H, vH, x) => `VaR ${lev}${x ? ' (alt nivel sau alt orizont decît în curs)' : ''}: ${v1}% pe o zi cu cuantila t, ${vn}% cu cuantila Normală; pe ${zile(H)}, aproximativ ${vH}%.`,
 ex: (lev, y, x, g0, g1, e0, e1, a) => `VaR ${lev}, din ${y}${x ? X : ''}: GARCH-t este depășit în ${g0}–${g1}% dintre zile, EWMA-Normal în ${e0}–${e1}%; un VaR corect este depășit în ${a}% dintre zile.`
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/,/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
const X = ' (an extra run of the code of the lecture)';
const lst = a => (a.length ? a.join(', ') : 'none');
window.CH9_TEXT = {
 h: { days: 'days', dm1: 'DM t, GARCH-t vs EWMA (p)', dm2: 'DM t, GJR-t vs GARCH-t (p)', normalz: 'Normal z(t)', params: 'parameters', ll: 'log-likelihood',
      sumalpha: 'Σα (+β)', step: 'step by step (SE)', secl: 'classic SE', serob: 'robust SE', ratio: 'robust/classic ratio', dn: 'Normal', dst: 'skewed t',
      from: 'from', hl: 'half-life (days)', lrvol: 'long-run vol.', svol: 'sample vol.', bicbest: 'BIC picks', sq: 'of the squares', sb: 'sign bias', ret: 'returns r(t)',
      zn: 'ẑ(t), Normal GARCH', zt: 'ẑ(t), GARCH-t', expected: 'expected', dist: { n: 'Normal', t: 'Student-t', st: 'skewed t' } },
 rt: (s, y, n, sy, s08, s20, sa, r08, r20, mn, dmn, mx, dmx) => `${s}, ${n} days: daily standard deviation ${sy === null ? `not available in ${y} (the series starts later)` : `${sy}% in ${y}`}; `
   + (s08 === null ? '' : `${s08}% from September 2008 to March 2009${r08 === null ? '' : ` (${r08} times larger)`}; `) + `${s20}% from February to May 2020${r20 === null ? '' : ` (${r20} times larger)`}; ${sa}% over the whole sample. `
   + `Largest fall: ${mn}% (${dmn}); largest rise: ${mx}% (${dmx}).`,
 sm: (seed, xs, k0, k1, k2, t0, t1, t2, m1, m2, xp) => `Seed ${seed}${xs || xp ? X : ''}: sample kurtosis i.i.d. ${k0}, ARCH(1) ${k1}, GARCH(1,1) ${k2}; theory ${t0}, ${t1} and ${t2}. `
   + `Largest absolute value: ARCH ${m1}, GARCH ${m2}. All three series have unconditional variance 1.`,
 l1: (a0, items, x, r) => `True α ${a0}${x ? X : ''}: ` + items.map(([n, a, s]) => `n = ${n}: α̂ = ${a} (standard error ${s})`).join('; ') + `. The standard error falls ${r} times when n grows 10 times (the √n rule: 3.16).`,
 lg: (seed, x, items) => `Seed ${seed}${x ? X : ''}: ` + items.map(([n, a, b, d]) => `n = ${n}: maximum on the grid at α = ${a}, β = ${b}; at the true values the log-likelihood is ${d} below the maximum`).join('; ') + '.',
 qq: (s, x, qn, thn, nu, qt, tht, raw) => `${s}${x ? X : ''}: the 0.1% quantile of the GARCH-N residuals is ${qn}, against ${thn} for the Normal distribution; the GARCH-t residuals (ν̂ = ${nu}) have ${qt}, `
   + `against ${tht} for the standardised t (= ${raw} × √((ν − 2)/ν)).`,
 vol: (d, x, items, r) => `${d} innovations${x ? X : ''}: ` + items.map(([s, f, p8, d8, p0, d0, md, l, sv]) => `${s} (from ${f}): ${p8 === null ? '' : `peak ${p8}% (${d8}), `}peak ${p0}% (${d0}), median ${md}%, last value ${l}%, sample volatility ${sv}%`).join('; ')
   + `. The median of the lower panel is ${r} times that of the upper panel.`,
 ps: (m, x, mm, items) => `${m}${x ? X : ''}; persistence ${mm === 'egarch' ? 'β (of ln σ²)' : mm === 'gjr' ? 'α + β + γ/2' : 'α + β'} and half-life: `
   + items.map(([s, p, h]) => `${s} ${p}, ${h === null ? 'no half-life (IGARCH)' : days(h)}`).join('; ') + '.',
 ni: (m, x, items) => `σ²(t − 1) = ${m} × the sample variance${x ? X : ''}; ratio of the variance after a 2% fall to the variance after a 2% rise: ` + items.map(([s, a, b]) => `${s} GJR ${a}, EGARCH ${b}`).join('; ') + '.',
 ad: (s, m, x, K, a11, a1K, a21, mx, b, o) => `${s}, ${m}${x ? X : ''}: the ACF of r² is ${a11} at lag 1 and ${a1K} at lag ${K}; the ACF of ẑ² is ${a21} at lag 1 and at most ${mx} in absolute value, with the band ±${b}; `
   + `${o} of ${K} lags leave the band.`,
 ts: (m, x, items, lr) => `${m}${x ? X : ''}, annualised volatility tomorrow, in 22 days and the average over 250 days: ` + items.map(([d, a, b, c]) => `${d}: ${a}%, ${b}%, ${c}%`).join('; ') + `. Long-run level: ${lr}%.`,
 fe: (y, l, x, items) => `Evaluation from ${y}, EWMA with λ = ${l}${x ? X : ''}; DM test of the QLIKE, GARCH-t against EWMA: ` + items.map(([s, t, p, g]) => `${s} t = ${t} (p ${p}${g ? ', GARCH-t better' : ', no detectable difference'})`).join('; ') + '.',
 vr: (s, a, b, n, lev, eg, ee, ex, x) => `${s}, ${a} – ${b} (${n} days), VaR ${lev}${x ? X : ''}: GARCH-t is exceeded ${eg} times, EWMA-Normal ${ee} times; about ${ex} expected.`,
 mx1: (c, s, q, p) => `Calm days: σ² = ${c}, with probability ${q}; stormy days: σ² = ${s}, with probability ${p}; the conditional mean is 0.`,
 mx2: (q, c, p, s, v, sd) => `Var(r) = ${q} × ${c} + ${p} × ${s} + 0 = ${v}, so the unconditional standard deviation is √${v} = ${sd}%.`,
 mx3: (a, b, m, sd) => `Conditional volatility: ${a}% on a calm day, ${b}% on a stormy day; their weighted average, ${m}%, is not ${sd}%: the variances are averaged, not the standard deviations.`,
 mx4: (e4, v2, K) => `Kurtosis of the mixture: 3 × ${e4}/${v2} = ${K}, above 3, although every day is Normal.`,
 mx: (v, sd, K, x) => `Unconditional variance ${v}, standard deviation ${sd}%, kurtosis ${K}${x ? ' (other parameters than in the lecture)' : ''}. Dashed: the Normal distribution with the same variance; the mixture has a higher peak and heavier tails.`,
 a1a: (w, a, q, s) => (s === null ? `α = ${a} ≥ 1: the unconditional variance is not finite.` : `Unconditional variance: σ̄² = ω/(1 − α) = ${w}/${q} = ${s}.`),
 a1b: (a2, n, d, K) => (K === null ? `3α² = ${d} ≥ 1: the fourth moment is infinite, the kurtosis does not exist.` : `Kurtosis: K = 3(1 − ${a2})/(1 − 3 × ${a2}) = 3 × ${n}/${d} = ${K}.`),
 a1c: (w, a, e, e2, s, sd) => `After the shock ε(t − 1) = ${e}: σ²(t) = ${w} + ${a} × ${e2} = ${s}, σ(t) = ${sd}.`,
 a1: (a, K, s, x) => `α = ${a}${x ? ' (other parameters than in the lecture)' : ''}: ${K === null ? 'the kurtosis is infinite (α ≥ 0.577)' : `kurtosis ${K}`}; today's variance after the chosen shock: ${s}.`,
 aq: (s, x, sa, pr, d10, qb, db) => `${s}${x ? X : ''}: ARCH(10) has Σα = ${sa}, GARCH(1,1) has α + β = ${pr}; the BIC of ARCH(10) is ${d10} above that of GARCH(1,1). `
   + (qb === 10 ? 'The BIC of ARCH(q) falls with every lag up to q = 10.' : `The best ARCH(q) by BIC, q = ${qb}, stays ${db} above GARCH(1,1).`),
 ku1: (a, b, p, st) => `Persistence α + β = ${a} + ${b} = ${p}${st ? ' &lt; 1: the model is covariance stationary.' : ' ≥ 1: the model is not covariance stationary.'}`,
 ku2: (w, q, s, v) => (s === null ? 'The unconditional variance is not finite.' : `σ̄² = ω/(1 − α − β) = ${w}/${q} = ${s}; annualised volatility √(252 × ${s}) = ${v}%.`),
 ku3: (p, a, c, K) => `(α + β)² + 2α² = ${p}² + 2 × ${a}² = ${c}` + (K === null ? ' ≥ 1: the fourth moment is infinite.' : `; K = 3(1 − ${p}²)/(1 − ${c}) = ${K}.`),
 ku: (a, b, K, v, x) => `α = ${a}, β = ${b}${x ? ' (other parameters than in the lecture)' : ''}: ${K === null ? 'the kurtosis is infinite' : `kurtosis ${K}`}; ${v === null ? 'the long-run variance is not finite' : `long-run volatility ${v}%`}.`,
 hl1: (p, lp, l5, h) => `h½ = ln 0.5/ln(α + β) = ${l5}/${lp} = ${h} days (α + β = ${p}).`,
 hl2: (q, h) => `Approximation for α + β close to 1: ln 2/(1 − α − β) = 0.6931/${q} = ${h} days.`,
 hl3: (a, b, c, p, h) => `For comparison: 0.90 gives ${a} days, 0.98 gives ${b}, 0.99 gives ${c}; S&P 500, GARCH(1,1)-t: α + β = ${p}, ${h} days.`,
 hl: (p, h, s, x) => `α + β = ${p}${x ? ' (another value than in the lecture example)' : ''}: half-life ${h} days; after 21 days ${s} of the shock is left.`,
 st1: (sp, w, a, b, m) => (sp ? `GARCH(1,1)-t parameters estimated for the S&P 500: ω = ${w}, α = ${a}, β = ${b}, μ = ${m}.` : `Parameters of the lecture: ω = ${w}, α = ${a}, β = ${b}, μ = ${m}.`),
 st2: (p, w, q, s, v, h) => `Persistence α + β = ${p}; σ̄² = ${w}/${q} = ${s}; annualised volatility √(252 × ${s}) = ${v}%; half-life ${h} days.`,
 st3: (s2, r, e, w, a, e2, b, s1, sd) => `Today σ²(t) = ${s2} and r(t) = ${r}%, so ε(t) = ${e}; tomorrow: σ²(t + 1) = ${w} + ${a} × ${e2} + ${b} × ${s2} = ${s1}, σ(t + 1) = ${sd}%.`,
 st4: (s1, sb, p, d10) => `Then the forecast approaches ${sb} by the factor ${p} a day: on day ten ${sb} + ${p}⁹ × (${s1} − ${sb}) = ${d10}.`,
 st: (r, s2, s1, sd, s0, x) => `r(t) = ${r}%, σ²(t) = ${s2}${x ? ' (other values than in the lecture)' : ''}: σ²(t + 1) = ${s1} (σ = ${sd}%); without a shock it would be ${s0}. Red line: the forecasts that follow, towards the long-run variance.`,
 ig1: (l, a) => `EWMA with λ = ${l} is an IGARCH with ω = 0, α = ${a}, β = ${l}: α + β = 1.`,
 ig2: (l, s, v) => `S&P 500, at the close of 16 March 2020: EWMA variance (λ = ${l}) for the next day σ²(t + 1) = ${s}, that is ${v}% a year.`,
 ig3: (w, s, s250, v) => `IGARCH with ω = ${w}: E[σ²(t + 250)] = ${s} + 249 × ${w} = ${s250}, that is ${v}% a year. With ω = 0 the forecast is the same for every horizon.`,
 ig4: (s, sb, p, f, v) => `GARCH(1,1)-t: σ²(t + 1) = ${s}, σ̄² = ${sb}, α + β = ${p}; 250 days ahead ${f}, that is ${v}% a year.`,
 ig: (l, e, g, lr, x) => `λ = ${l}${x ? ' (another value than in the lecture)' : ''}: EWMA forecasts ${e}% a year for every horizon; GARCH(1,1)-t falls to ${g}% after 250 days, towards ${lr}%.`,
 et: (s, x, l1, l2, ra, rb, rw, rm) => `${s}${x ? X : ''}: log-likelihood ${l1} step by step (σ²(1) = the sample variance) and ${l2} with arch; robust SE / classic SE: α ${ra}, β ${rb}, ω ${rw}, μ ${rm}.`,
 dt: (s, x, l1, l2, nu, lm, an, at, bn, bt) => `${s}${x ? X : ''}: LR t against Normal ${l1}, skewed t against t ${l2} (critical value 3.84); ν̂ = ${nu}, λ̂ = ${lm}; α ${an} → ${at}, β ${bn} → ${bt}.`,
 mt: (d, x, p0, p1, h0, h1, s0, s1, ba, a, ig) => `${d} innovations${x ? X : ''}: the indices have α + β between ${p0} and ${p1} and half-lives of ${h0}–${days(h1)}; robust SE of α̂ and β̂ between ${s0} and ${s1}; `
   + `the largest α: ${ba} (${a})${ig.length ? `; IGARCH (α + β ≥ 0.9995): ${ig.join(', ')}` : ''}.`,
 lr0: (s, p, w, b) => `${s}: α̂ + β̂ = ${p}, on the boundary (IGARCH): 1 − α − β is practically 0, so ω/(1 − α − β) is not finite; there is no long-run volatility. With ω = ${w} > 0 the variance forecast grows by ω a day; the model behaves like an EWMA with λ ≈ ${b}.`,
 lr1: (s, w, p, q, sb) => `${s}: ω = ${w}, α + β = ${p}; 1 − α − β = ${q}; σ̄² = ${w}/${q} = ${sb}.`,
 lr2: (A, sb, v, sv) => `Annualised with A = ${A} observations a year: √(${A} × ${sb}) = ${v}%, against the sample volatility ${sv}%.`,
 lr3: (p, q, v) => `With α + β = ${p} (0.002 lower): 1 − α − β = ${q}, the long-run volatility would be ${v}%.`,
 lr4: (v, p) => (v === null ? `GARCH with Normal innovations: α + β = ${p}, also on the boundary.` : `GARCH with Normal innovations (α + β = ${p}) gives ${v}%.`),
 lr: (s, ig, v, sv, x) => `${s}${x ? X : ''}: ${ig ? 'IGARCH, no long-run volatility' : `long-run volatility ${v}%`}, sample volatility ${sv}%. The curve shows how fast the long-run level grows as α + β approaches 1.`,
 gj1: (s, w, a, g, b, sb) => `${s}, GJR-GARCH(1,1)-t: ω = ${w}, α = ${a}, γ = ${g}, β = ${b}; σ²(t − 1) fixed at the sample variance ${sb}.`,
 gj2: (e, w, ag, e2, b, sb, v) => `Fall of ${e}%: σ² = ${w} + ${ag} × ${e2} + ${b} × ${sb} = ${v}.`,
 gj3: (e, w, a, e2, b, sb, v) => `Rise of ${e}%: σ² = ${w} + ${a} × ${e2} + ${b} × ${sb} = ${v}.`,
 gj4: (r, re) => `Ratio: ${r}; with EGARCH(1,1)-t the ratio is ${re}.`,
 gj: (s, e, r, re, x) => `${s}, shocks of ±${e}%${x ? X : ''}: after the fall tomorrow's variance is ${r} times that after the rise (EGARCH: ${re}).`,
 at: (d, x, sig, gb, lr, lt) => `${d} innovations${x ? X : ''}: GJR γ positive and significant at 5% (LR test): ${lst(sig)}; BIC prefers the symmetric GARCH: ${lst(gb)}. BET: LR = ${lr}, below ln T = ${lt}, the BIC penalty for one more parameter.`,
 dg: (s, m, x, q2, z2, pz2, st, psb, pz) => `${s}, m = ${m}${x ? X : ''}: Q(${m}) of the squares falls from ${q2} (returns) to ${z2} (ẑ, GARCH-t; p ${pz2}); Q(${m}) on ẑ has p ${pz}; `
   + `sign-bias test: sign t ${st}, joint p ${psb}.`,
 ms: (s, c, x, bm, bd, da, dt, db) => `${s}, ${c}${x ? X : ''}: the best model is ${bm} with ${bd} innovations. Against GARCH with Normal innovations, asymmetry alone (EGARCH-Normal) lowers ${c} by ${da}, `
   + `heavy tails alone (GARCH with skewed t) by ${dt}, the best model by ${db}.`,
 fw1: (d, p, sb, s1) => `${d}, GARCH(1,1)-t: α + β = ${p}, σ̄² = ${sb}; forecast for the next day σ²(t + 1) = ${s1}.`,
 fw2: (H, sb, p, h1, ph, s1, f) => `Day ${H}: ${sb} + ${p}<sup>${h1}</sup> × (${s1} − ${sb}) = ${sb} + ${ph} × (${s1} − ${sb}) = ${f}.`,
 fw3: (H, s, v) => `Variance of the ${H}-day return: the sum of the ${H} daily forecasts = ${s}, so the volatility is √${s} = ${v}%.`,
 fw4: (H, s1, v, below) => `Square-root-of-time rule: √(${H} × ${s1}) = ${v}%: ${below ? 'too small, because volatility is expected to rise towards its long-run level' : 'too large, because volatility is expected to fall towards its long-run level'}.`,
 fw: (d, H, v, r, below, x) => `${d}, H = ${H}${x ? ' (another horizon or date than in the lecture)' : ''}: H-day volatility from the sum of the forecasts ${v}%, square-root-of-time rule ${r}% (${below ? 'too small' : 'too large'}).`,
 vw1: (lev, m, s2, s, nu) => `S&P 500 on 18 September 2026, GARCH(1,1)-t: μ̂ = ${m}, σ²(t + 1) = ${s2}, σ(t + 1) = ${s}, ν̂ = ${nu}.`,
 vw2: (lev, tq, sc, q) => `The ${lev} quantile of z: ${tq} × ${sc} = ${q} (the Student-t quantile rescaled to variance 1).`,
 vw3: (lev, m, s, q, v, zq, vn) => `One-day VaR ${lev}: −(${m} + ${s} × (${q})) = ${v}%; with Normal z(t) (quantile ${zq}): ${vn}%.`,
 vw4: (H, q, s, v, r) => `Over ${days(H)} (sum of the variances, without μ): ${q} × √${s} = ${v}%; the √H rule applied to the one-day VaR gives ${r}%. This is an approximation: the ${H}-day return is not Student-t with the same ν, and its exact quantile needs simulation.`,
 vw: (lev, v1, vn, H, vH, x) => `VaR ${lev}${x ? ' (another level or horizon than in the lecture)' : ''}: ${v1}% for one day with the t quantile, ${vn}% with the Normal quantile; over ${days(H)}, about ${vH}%.`,
 ex: (lev, y, x, g0, g1, e0, e1, a) => `VaR ${lev}, from ${y}${x ? X : ''}: GARCH-t is exceeded on ${g0}–${g1}% of the days, EWMA-Normal on ${e0}–${e1}%; a correct VaR is exceeded on ${a}% of the days.`
};
})();
""",
}

# Question for the room and answer of every block (RO, EN)
QA = {
 'ro': {
  'returns': ('Abaterea standard zilnică a S&P 500 a fost 0,42% în 2017. De cîte ori a fost mai mare între septembrie 2008 și martie 2009?',
              'De 8,3 ori: 3,52/0,422 = 8,34; între februarie și mai 2020, 3,70/0,422 = 8,76. Abaterea standard a întregului eșantion, 1,21%, este o medie a zilelor liniștite și a celor agitate '
              'și nu descrie niciun regim: episodul din 2008 este de 2,9 ori peste ea.'),
  'mix': ('Zile liniștite cu σ² = 0,5 și zile agitate cu σ² = 4,5, fiecare cu probabilitatea 1/2, media 0. Cît este abaterea standard necondiționată?',
          '1,58%: varianța necondiționată este media varianțelor, 0,5 × 0,5 + 0,5 × 4,5 = 2,50, iar √2,50 = 1,58. Media volatilităților, (0,71 + 2,12)/2 = 1,41%, este mai mică, pentru că rădăcina '
          'pătrată este concavă. Amestecul are boltirea 4,92, deși fiecare regim este Normal.'),
  'arch1': ('ARCH(1) cu ω = 0,5: pentru α = 0,5 boltirea este 9,0. Ce se întîmplă cu boltirea cînd ▶ duce α la 0,6?',
            'Devine infinită: 3α² = 1,08 ≥ 1, adică α depășește 1/√3 = 0,577, iar momentul de ordinul patru nu mai există. Varianța necondiționată rămîne finită, ω/(1 − α) = 0,5/0,4 = 1,25.'),
  'simulated': ('Pentru ARCH(1) cu ω = α = 0,5 boltirea teoretică este 9,0. Ce boltire are seria simulată de 1000 de valori?',
                '4,91, mult sub 9,0: momentul de ordinul patru converge lent, pentru că depinde de cîteva vîrfuri rare. Seria i.i.d. are 3,12 (teoretic 3), GARCH(1,1) are 3,74 (teoretic 6,06). '
                'Cu alte semințe ale generatorului valorile de selecție se schimbă mult.'),
  'archq': ('S&P 500: ARCH(10) are 12 parametri și log-verosimilitatea −9 227,6; GARCH(1,1) are 4 parametri și −9 218,0. Ce model preferă BIC?',
            'GARCH(1,1): BIC 18 471,2 față de 18 561,0 pentru ARCH(10), cu 89,8 mai mic. Are o verosimilitate mai mare cu de trei ori mai puțini parametri: termenul βσ²(t − 1) rezumă toate '
            'șocurile trecute, cu ponderi geometrice. BIC al ARCH(q) scade cu fiecare lag adăugat pînă la q = 10, dar rămîne peste cel al GARCH(1,1).'),
  'kurt': ('α = 0,10 și β = 0,88 dau boltirea 6,06. Cît devine boltirea cînd ▶ duce β la 0,89?',
           'Infinită: (α + β)² + 2α² = 0,9801 + 0,02 = 1,0001 ≥ 1. Varianța necondiționată rămîne finită (α + β = 0,99 < 1; cu ω = 0,02, σ̄² = 2,00), dar momentul de ordinul patru nu mai '
           'există: condiția pentru boltire este mai strictă decît cea pentru staționaritate.'),
  'half-life': ('Pentru α + β = 0,98 timpul de înjumătățire este 34,3 zile. Cît este pentru α + β = 0,99?',
                '69,0 zile, aproape dublu: h½ ≈ ln 2/(1 − α − β), iar 1 − α − β se înjumătățește, de la 0,02 la 0,01. Pentru 0,90 sînt 6,6 zile; pentru S&P 500 (α + β = 0,9947), 131 de zile.'),
  'step': ('Cu σ²(t) = 1,2 și r(t) = −3%, σ²(t + 1) = 1,976. Cît este σ²(t + 1) după r(t) = +3%?',
           'Tot 1,976: GARCH(1,1) folosește ε², deci semnul șocului nu contează (aici μ = 0). Varianta 1,376 = 0,02 + 0,10 × 3 + 0,88 × 1,2 uită ridicarea la pătrat. Asimetria cere GJR-GARCH '
           'sau EGARCH (secțiunea 8).'),
  'igarch': ('Un analist prognozează volatilitatea peste un an cu EWMA imediat după un crah. Unde greșește?',
             'EWMA păstrează nivelul de criză pentru tot anul. S&P 500, la închiderea din 16 martie 2020: EWMA (λ = 0,94) prognozează 84,0% pe an pentru orice orizont; GARCH(1,1)-t pornește de la '
             '111,0% și coboară la 62,0% după 250 de zile, spre nivelul pe termen lung de 27,3%. EWMA este un IGARCH cu ω = 0: fără revenire la medie.'),
  'lik-arch1': ('Cu n = 100, α̂ = 0,432 și eroarea standard 0,127. Cum se schimbă eroarea standard cu n = 1000?',
                'Scade la 0,035, de 3,6 ori; regula √n dă √10 = 3,16. Curba log-verosimilității devine mai ascuțită, iar α̂ = 0,491 este mai aproape de valoarea adevărată 0,5.'),
  'lik-garch': ('Cu n = 500, maximul pe grilă este în (0,045; 0,63), departe de valorile adevărate (0,1; 0,8). Unde este maximul cu n = 2000?',
                'În (0,100; 0,78), aproape de valorile adevărate: creasta lungă de-a lungul căreia α și β se compensează se îngustează cînd n crește. Cu alte semințe ale generatorului (rulări '
                'suplimentare), maximul pentru n = 500 variază între (0,025; 0,95) și (0,205; 0,67).'),
  'step-table': ('S&P 500, GARCH(1,1) cu inovații Normale: SE clasică a lui α̂ este 0,0086. Cît este SE robustă?',
                 '0,0118, de 1,37 ori mai mare (β: 1,37; ω: 1,63; μ: 1,02). Inovațiile au cozi groase, iar SE clasice, calculate ca și cum z(t) ar fi Normal, supraestimează precizia parametrilor '
                 'varianței. Pentru BET raportul trece de 2,6.'),
  'dist-table': ('Trecerea de la inovații Normale la Student-t ridică log-verosimilitatea S&P 500 de la −9 218,0 la −9 062,5. Se schimbă mult α și β?',
                 'Puțin: α de la 0,121 la 0,123, β de la 0,860 la 0,872; persistența urcă de la 0,980 la 0,995. Statistica LR, 311,0, depășește de departe 3,84: distribuția inovațiilor contează '
                 'pentru cuantilele din coadă (VaR) mai mult decît pentru σ(t).'),
  'qq': ('Cuantila de 0,1% a reziduurilor GARCH-N este −4,81, față de −3,09 pentru distribuția Normală. Care este cuantila de 0,1% a t standardizate cu ν̂ = 6,26?',
         '−4,19 = t⁻¹(0,001) × √((ν − 2)/ν) = −5,08 × 0,825. Reziduurile GARCH-t au −4,79: coada t este mult mai aproape, dar scăderile cele mai mari rămîn mai adînci.'),
  'mk-table': ('BET are cel mai mare α̂ dintre cele șase serii (0,192). Ce înseamnă?',
               'O reacție mai puternică la știrile de ieri: un șoc de 2% adaugă 0,192 × 4 = 0,77 la varianța de mîine, față de 0,123 × 4 = 0,49 la S&P 500. Persistența este însă apropiată '
               '(0,9907 față de 0,9947), pentru că β̂ al BET este mai mic (0,799).'),
  'longrun': ('S&P 500: volatilitatea pe termen lung 27,3%, volatilitatea de selecție 19,2%. Este greșit modelul?',
              'Nu neapărat: σ̄² = ω/(1 − α − β) se împarte la 1 − α − β = 0,0053, un număr mic și imprecis. Cu α + β mai mic cu 0,002 (0,9927), volatilitatea pe termen lung ar fi 23,3%; '
              'GARCH cu inovații Normale dă 18,1%. Nivelul pe termen lung este cel mai puțin sigur rezultat al unui GARCH persistent.'),
  'vol-sd': ('Vîrful din 2020 al volatilității GARCH-t a S&P 500 este 111% pe an (17 martie 2020). Cît a fost vîrful din 2008?',
             '93%, pe 16 octombrie 2008: mai jos decît în 2020, deși criza din 2008 a durat mai mult; vîrful reflectă cele mai mari șocuri dintr-un interval scurt, nu durata crizei. '
             'DAX: 80% în 2008 și 82% în 2020. Mediana S&P 500 este 14%, iar valoarea din 18 septembrie 2026 este 12%.'),
  'vol-bb': ('Volatilitatea GARCH-t mediană a Bitcoin este 59% pe an, a S&P 500 14%. De cîte ori este mai volatil Bitcoin?',
             'De 4,1 ori (58,7/14,3); raportul varianțelor ar fi 4,1² ≈ 17. Vîrfurile: Bitcoin 301% pe 13 martie 2020, BET 117% pe 15 octombrie 2008 și 88% în martie 2020.'),
  'persistence': ('Care este timpul de înjumătățire al unui șoc de varianță pentru Bitcoin?',
                  'Nu există: α̂ + β̂ = 1,000 (IGARCH), deci un șoc nu se stinge. S&P 500 are nevoie de 131 de zile, DAX de 88, BET de 74, TLV de 18, SNP de 12. Cu GJR-t, persistența S&P 500 '
                  '(α + β + γ/2) este 0,983, cu un timp de înjumătățire de 41,5 zile.'),
  'gjr': ('S&P 500, GJR-t: α̂ = 0 și γ̂ = 0,205. Cu cît crește varianța de mîine după o creștere de 2%, față de o zi fără șoc?',
          'Cu 0: știrile bune au panta α = 0. După o scădere de 2%, termenul (α + γ)ε² adaugă 0,205 × 4 = 0,821; cu σ²(t − 1) fixat la varianța de selecție 1,471, varianța de mîine este 2,136 '
          'după scădere și 1,315 după creștere: raportul 1,62.'),
  'nic': ('S&P 500, GJR: varianța de după o scădere de 2% este de 1,62 ori cea de după o creștere de 2%, cu σ²(t − 1) egal cu varianța de selecție. Ce se întîmplă cu raportul cînd ▶ crește σ²(t − 1) la de 4 ori varianța de selecție?',
          'Scade la 1,16: termenul comun βσ²(t − 1) crește, iar diferența γε² rămîne aceeași, deci contează relativ mai puțin. Cu σ²(t − 1) de 4 ori mai mic decît varianța de selecție raportul '
          'urcă la 3,39: asimetria se vede cel mai bine în perioadele liniștite.'),
  'asym-table': ('BET: γ̂ = 0,043, t = 2,1, testul LR are p = 0,026, dar BIC preferă GARCH simetric. Cum se explică?',
                 'BIC cere ca LR să depășească ln T = ln 6 670 = 8,8 pentru un parametru în plus; LR = 4,9 trece de 3,84 (pragul de 5%), dar nu de 8,8: efectul de levier al BET este slab. '
                 'Pentru Bitcoin BIC alege EGARCH deși γ nu este semnificativ: forma logaritmică descrie mai bine dinamica varianței, nu asimetria.'),
  'acf-diag': ('ACF a lui r² la lagul 1 este 0,31. Cît este ACF a lui ẑ² (GARCH-t) la lagul 1?',
               '−0,005, în banda ±0,024; cea mai mare valoare absolută pe 50 de laguri este 0,028. Trei parametri ai varianței absorb aproape tot volatility clustering; la lagul 50, ACF a lui r² '
               'este încă 0,053.'),
  'diag-table': ('S&P 500, GARCH-t: Q(10) pe ẑ² are p = 0,192, iar testul de asimetrie are p < 0,001. Este modelul corect?',
                 'Nu complet: nu au rămas efecte ARCH (Q(10) al pătratelor scade de la 6 133 la 13,6), dar semnul șocurilor trecute încă anticipează varianța (t al semnului 3,56): lipsește '
                 'asimetria. Un test care nu respinge nu dovedește că modelul este corect.'),
  'ms-table': ('S&P 500, pornind de la GARCH cu inovații Normale: ce reduce mai mult BIC, asimetria (EGARCH cu inovații Normale) sau cozile groase (GARCH cu t asimetrică)?',
               'Cozile groase: 341, față de 276 pentru asimetrie; ambele împreună (EGARCH cu t asimetrică) reduc BIC cu 649, cel mai bun dintre cele nouă modele. AIC dă aceeași ordine.'),
  'fc-we': ('La 18 septembrie 2026, σ²(t + 1) = 0,509, sub σ̄² = 2,972. Regula rădăcinii pătrate a timpului dă volatilitatea pe 10 zile √(10 × 0,509) = 2,26%. Ce dă suma prognozelor GARCH?',
            '2,38%: prognozele zilnice cresc spre nivelul pe termen lung (ziua 10: 0,624), deci suma lor, 5,67, depășește 10 × 0,509 = 5,09. Într-o criză relația se inversează: regula '
            'rădăcinii pătrate dă valori prea mari.'),
  'term': ('Pe 16 martie 2020 prognoza pentru ziua următoare este 111,0% pe an. Cît este volatilitatea medie prognozată pentru următoarele 250 de zile?',
           '84,7%: cu α + β = 0,9947, abaterea față de nivelul pe termen lung se micșorează cu doar 0,53% pe zi, iar după 250 de zile prognoza zilnică este încă 62,0%. Într-o zi liniștită '
           '(3 noiembrie 2017) curba urcă: 7,3% mîine, 19,0% în medie pe un an.'),
  'fc-eval': ('Bitcoin, 2015–2026: QLIKE GARCH-t 3,383, EWMA 3,390, testul DM t = −0,20 (p = 0,845). Ce concluzie trageți?',
              'Nicio diferență detectabilă: GARCH-ul Bitcoin este un IGARCH, adică aproape un EWMA. Un p mare nu arată că prognozele sînt egale, doar că testul nu le separă. Pentru S&P 500 '
              '(t = −3,69), DAX (−3,19) și BET (−2,22), GARCH-t este semnificativ mai bun.'),
  'var-we': ('S&P 500 la 18 septembrie 2026: VaR 1% pe o zi este 1,75%. Regula √10 dă pe 10 zile 1,75 × √10 = 5,52%. Ce dă suma prognozelor varianței (cu cuantila t și fără μ)?',
             '6,09% = 2,557 × √5,67: volatilitatea este așteptată să crească spre nivelul pe termen lung, deci regula √10 dă prea puțin. Este o aproximare: randamentul pe 10 zile nu are o '
             'distribuție Student-t cu același ν, iar cuantila lui exactă cere simulare.'),
  'var': ('S&P 500, iulie 2019 – iunie 2021, 505 zile: de cîte ori este depășit VaR 1% din GARCH-t?',
          'De 13 ori, față de circa 5 așteptate (1% din 505); EWMA-Normal este depășit de 17 ori. Ambele modele reacționează, dar abia după primele scăderi mari din martie 2020: o prognoză '
          'pe o zi nu poate anticipa primul șoc.'),
  'exc': ('EWMA cu inovații Normale depășește VaR 1% în 2,2–2,5% dintre zile pe toate cele patru piețe. De ce?',
          'Cuantila Normală, −2,326, este prea puțin adîncă pentru cozile groase. GARCH-t, cu cuantila Student-t, are 1,2–1,7%: mult mai aproape de 1%, dar tot peste 1% pe toate cele patru '
          'piețe. Dacă diferențele sînt semnificative se testează în Capitolul 10 (testul Kupiec).'),
 },
 'en': {
  'returns': ('The daily standard deviation of the S&P 500 was 0.42% in 2017. How many times larger was it from September 2008 to March 2009?',
              '8.3 times: 3.52/0.422 = 8.34; from February to May 2020, 3.70/0.422 = 8.76. The whole-sample standard deviation, 1.21%, averages calm and stormy days and describes neither '
              'regime: the 2008 episode is 2.9 times above it.'),
  'mix': ('Calm days with σ² = 0.5 and stormy days with σ² = 4.5, each with probability 1/2, mean 0. What is the unconditional standard deviation?',
          '1.58%: the unconditional variance is the average of the variances, 0.5 × 0.5 + 0.5 × 4.5 = 2.50, and √2.50 = 1.58. The average of the volatilities, (0.71 + 2.12)/2 = 1.41%, is '
          'smaller, because the square root is concave. The mixture has kurtosis 4.92, although each regime is Normal.'),
  'arch1': ('ARCH(1) with ω = 0.5: for α = 0.5 the kurtosis is 9.0. What happens to the kurtosis when ▶ takes α to 0.6?',
            'It becomes infinite: 3α² = 1.08 ≥ 1, that is α exceeds 1/√3 = 0.577 and the fourth moment no longer exists. The unconditional variance stays finite, ω/(1 − α) = 0.5/0.4 = 1.25.'),
  'simulated': ('For ARCH(1) with ω = α = 0.5 the theoretical kurtosis is 9.0. What kurtosis does the simulated series of 1000 values have?',
                '4.91, far below 9.0: the fourth moment converges slowly, because it depends on a few rare spikes. The i.i.d. series has 3.12 (theory 3), GARCH(1,1) has 3.74 (theory 6.06). '
                'With other seeds of the generator the sample values change a lot.'),
  'archq': ('S&P 500: ARCH(10) has 12 parameters and log-likelihood −9,227.6; GARCH(1,1) has 4 parameters and −9,218.0. Which model does BIC prefer?',
            'GARCH(1,1): BIC 18,471.2 against 18,561.0 for ARCH(10), 89.8 lower. It has a higher likelihood with a third of the parameters: the term βσ²(t − 1) summarises all past shocks '
            'with geometric weights. The BIC of ARCH(q) falls with every added lag up to q = 10, but stays above that of GARCH(1,1).'),
  'kurt': ('α = 0.10 and β = 0.88 give kurtosis 6.06. What does the kurtosis become when ▶ takes β to 0.89?',
           'Infinite: (α + β)² + 2α² = 0.9801 + 0.02 = 1.0001 ≥ 1. The unconditional variance stays finite (α + β = 0.99 < 1; with ω = 0.02, σ̄² = 2.00), but the fourth moment no longer '
           'exists: the condition for the kurtosis is stricter than that for stationarity.'),
  'half-life': ('For α + β = 0.98 the half-life is 34.3 days. What is it for α + β = 0.99?',
                '69.0 days, almost double: h½ ≈ ln 2/(1 − α − β), and 1 − α − β halves, from 0.02 to 0.01. For 0.90 it is 6.6 days; for the S&P 500 (α + β = 0.9947), 131 days.'),
  'step': ('With σ²(t) = 1.2 and r(t) = −3%, σ²(t + 1) = 1.976. What is σ²(t + 1) after r(t) = +3%?',
           'Also 1.976: GARCH(1,1) uses ε², so the sign of the shock does not matter (here μ = 0). The value 1.376 = 0.02 + 0.10 × 3 + 0.88 × 1.2 forgets the square. Asymmetry needs '
           'GJR-GARCH or EGARCH (Section 8).'),
  'igarch': ('An analyst forecasts the volatility one year ahead with EWMA right after a crash. What goes wrong?',
             'EWMA keeps the crisis level for the whole year. S&P 500, at the close of 16 March 2020: EWMA (λ = 0.94) forecasts 84.0% a year for every horizon; GARCH(1,1)-t starts at 111.0% and '
             'falls to 62.0% after 250 days, towards the long-run level of 27.3%. EWMA is an IGARCH with ω = 0: no mean reversion.'),
  'lik-arch1': ('With n = 100, α̂ = 0.432 and the standard error is 0.127. How does the standard error change with n = 1000?',
                'It falls to 0.035, 3.6 times smaller; the √n rule gives √10 = 3.16. The log-likelihood curve gets sharper, and α̂ = 0.491 is closer to the true value 0.5.'),
  'lik-garch': ('With n = 500 the maximum on the grid is at (0.045, 0.63), far from the true values (0.1, 0.8). Where is the maximum with n = 2000?',
                'At (0.100, 0.78), close to the true values: the long ridge along which α and β trade off narrows as n grows. With other seeds of the generator (extra runs) the maximum for '
                'n = 500 ranges from (0.025, 0.95) to (0.205, 0.67).'),
  'step-table': ('S&P 500, GARCH(1,1) with Normal innovations: the classic SE of α̂ is 0.0086. What is the robust SE?',
                 '0.0118, 1.37 times larger (β: 1.37; ω: 1.63; μ: 1.02). The innovations have heavy tails, and the classic SE, computed as if z(t) were Normal, overstate the precision of the '
                 'variance parameters. For the BET the ratio exceeds 2.6.'),
  'dist-table': ('Moving from Normal to Student-t innovations raises the S&P 500 log-likelihood from −9,218.0 to −9,062.5. Do α and β change much?',
                 'Little: α from 0.121 to 0.123, β from 0.860 to 0.872; the persistence rises from 0.980 to 0.995. The LR statistic, 311.0, is far above 3.84: the innovation distribution matters '
                 'for the tail quantiles (VaR) more than for σ(t).'),
  'qq': ('The 0.1% quantile of the GARCH-N residuals is −4.81, against −3.09 for the Normal distribution. What is the 0.1% quantile of the standardised t with ν̂ = 6.26?',
         '−4.19 = t⁻¹(0.001) × √((ν − 2)/ν) = −5.08 × 0.825. The GARCH-t residuals have −4.79: the t tail is much closer, but the largest falls are still deeper.'),
  'mk-table': ('The BET has the largest α̂ of the six series (0.192). What does it mean?',
               'A stronger reaction to yesterday\'s news: a 2% shock adds 0.192 × 4 = 0.77 to tomorrow\'s variance, against 0.123 × 4 = 0.49 for the S&P 500. The persistence is close, though '
               '(0.9907 against 0.9947), because the β̂ of the BET is smaller (0.799).'),
  'longrun': ('S&P 500: long-run volatility 27.3%, sample volatility 19.2%. Is the model wrong?',
              'Not necessarily: σ̄² = ω/(1 − α − β) divides by 1 − α − β = 0.0053, a small and imprecise number. With α + β lower by 0.002 (0.9927) the long-run volatility would be 23.3%; '
              'the Normal GARCH gives 18.1%. The long-run level is the least reliable output of a persistent GARCH.'),
  'vol-sd': ('The 2020 peak of the S&P 500 GARCH-t volatility is 111% a year (17 March 2020). What was the 2008 peak?',
             '93%, on 16 October 2008: lower than in 2020, although the 2008 crisis lasted longer; the peak reflects the largest shocks of a short interval, not the length of the crisis. '
             'DAX: 80% in 2008 and 82% in 2020. The S&P 500 median is 14%, and the value on 18 September 2026 is 12%.'),
  'vol-bb': ('The median GARCH-t volatility of Bitcoin is 59% a year, that of the S&P 500 14%. How many times more volatile is Bitcoin?',
             '4.1 times (58.7/14.3); the ratio of the variances would be 4.1² ≈ 17. Peaks: Bitcoin 301% on 13 March 2020, BET 117% on 15 October 2008 and 88% in March 2020.'),
  'persistence': ('What is the half-life of a variance shock for Bitcoin?',
                  'There is none: α̂ + β̂ = 1.000 (IGARCH), so a shock does not die out. The S&P 500 needs 131 days, the DAX 88, the BET 74, TLV 18, SNP 12. With GJR-t the S&P 500 persistence '
                  '(α + β + γ/2) is 0.983, a half-life of 41.5 days.'),
  'gjr': ('S&P 500, GJR-t: α̂ = 0 and γ̂ = 0.205. By how much does a 2% rise raise tomorrow\'s variance, compared with a day without a shock?',
          'By 0: good news has the slope α = 0. After a 2% fall the term (α + γ)ε² adds 0.205 × 4 = 0.821; with σ²(t − 1) fixed at the sample variance 1.471, tomorrow\'s variance is 2.136 '
          'after the fall and 1.315 after the rise: the ratio 1.62.'),
  'nic': ('S&P 500, GJR: the variance after a 2% fall is 1.62 times that after a 2% rise, with σ²(t − 1) equal to the sample variance. What happens to the ratio when ▶ raises σ²(t − 1) to 4 times the sample variance?',
          'It falls to 1.16: the common term βσ²(t − 1) grows while the difference γε² stays the same, so it matters relatively less. With σ²(t − 1) at a quarter of the sample variance the '
          'ratio rises to 3.39: asymmetry shows most in calm periods.'),
  'asym-table': ('BET: γ̂ = 0.043, t = 2.1, the LR test has p = 0.026, but BIC prefers the symmetric GARCH. Why?',
                 'BIC needs LR to exceed ln T = ln 6,670 = 8.8 for one more parameter; LR = 4.9 passes 3.84 (the 5% threshold) but not 8.8: the BET leverage effect is weak. For Bitcoin BIC '
                 'picks EGARCH although γ is not significant: the logarithmic form fits the variance dynamics better, not the asymmetry.'),
  'acf-diag': ('The ACF of r² at lag 1 is 0.31. What is the ACF of ẑ² (GARCH-t) at lag 1?',
               '−0.005, inside the band ±0.024; the largest absolute value over 50 lags is 0.028. Three variance parameters absorb almost all the volatility clustering; at lag 50 the ACF of r² '
               'is still 0.053.'),
  'diag-table': ('S&P 500, GARCH-t: Q(10) on ẑ² has p = 0.192, and the sign-bias test has p < 0.001. Is the model correct?',
                 'Not entirely: no ARCH effects are left (Q(10) of the squares falls from 6,133 to 13.6), but the sign of past shocks still anticipates the variance (sign t = 3.56): the asymmetry '
                 'is missing. A test that does not reject does not prove the model correct.'),
  'ms-table': ('S&P 500, starting from GARCH with Normal innovations: which lowers BIC more, asymmetry (EGARCH with Normal innovations) or heavy tails (GARCH with skewed t)?',
               'Heavy tails: 341, against 276 for asymmetry; both together (EGARCH with skewed t) lower BIC by 649, the best of the nine models. AIC gives the same order.'),
  'fc-we': ('On 18 September 2026, σ²(t + 1) = 0.509, below σ̄² = 2.972. The square-root-of-time rule gives the 10-day volatility √(10 × 0.509) = 2.26%. What does the sum of the GARCH forecasts give?',
            '2.38%: the daily forecasts rise towards the long-run level (day 10: 0.624), so their sum, 5.67, exceeds 10 × 0.509 = 5.09. In a crisis the relation reverses: the square-root rule '
            'gives values that are too large.'),
  'term': ('On 16 March 2020 the forecast for the next day is 111.0% a year. What is the forecast average volatility over the next 250 days?',
           '84.7%: with α + β = 0.9947 the gap to the long-run level shrinks by only 0.53% a day, and after 250 days the daily forecast is still 62.0%. On a calm day (3 November 2017) the '
           'curve rises: 7.3% tomorrow, 19.0% on average over a year.'),
  'fc-eval': ('Bitcoin, 2015–2026: QLIKE GARCH-t 3.383, EWMA 3.390, DM test t = −0.20 (p = 0.845). What do you conclude?',
              'No detectable difference: the Bitcoin GARCH is an IGARCH, that is almost an EWMA. A large p does not show that the forecasts are equal, only that the test does not separate them. '
              'For the S&P 500 (t = −3.69), the DAX (−3.19) and the BET (−2.22) GARCH-t is significantly better.'),
  'var-we': ('S&P 500 on 18 September 2026: the one-day VaR 1% is 1.75%. The √10 rule gives over 10 days 1.75 × √10 = 5.52%. What does the sum of the variance forecasts give (with the t quantile and without μ)?',
             '6.09% = 2.557 × √5.67: volatility is expected to rise towards its long-run level, so the √10 rule gives too little. This is an approximation: the 10-day return is not Student-t '
             'with the same ν, and its exact quantile needs simulation.'),
  'var': ('S&P 500, July 2019 – June 2021, 505 days: how many times is the GARCH-t VaR 1% exceeded?',
          '13 times, against about 5 expected (1% of 505); EWMA-Normal is exceeded 17 times. Both models react, but only after the first large falls of March 2020: a one-day forecast '
          'cannot anticipate the first shock.'),
  'exc': ('EWMA with Normal innovations exceeds the VaR 1% on 2.2–2.5% of the days in all four markets. Why?',
          'The Normal quantile, −2.326, is not deep enough for heavy tails. GARCH-t, with the Student-t quantile, has 1.2–1.7%: much closer to 1%, but still above 1% in all four markets. '
          'Whether the differences are significant is tested in Chapter 10 (the Kupiec test).'),
 },
}

# Predict, then check: (key, [(RO, EN), ...]); the keys agree with what the page computes (node script on ch9_data.js + common.js + ch9.js).
PREDICT = {
 'returns': (0, [('De circa 8,3 ori: 3,52% față de 0,42%.', 'About 8.3 times: 3.52% against 0.42%.'),
                 ('De circa 2,9 ori: 3,52% față de 1,21%, abaterea standard a întregului eșantion.', 'About 2.9 times: 3.52% against 1.21%, the whole-sample standard deviation.'),
                 ('De circa 69 de ori: raportul varianțelor.', 'About 69 times: the ratio of the variances.')]),
 'mix': (0, [('1,58%: √2,50.', '1.58%: √2.50.'), ('1,41%: media volatilităților 0,71% și 2,12%.', '1.41%: the average of the volatilities 0.71% and 2.12%.'),
             ('2,50%: media varianțelor.', '2.50%: the average of the variances.')]),
 'arch1': (0, [('Infinită: 3α² = 1,08 depășește 1.', 'Infinite: 3α² = 1.08 exceeds 1.'),
               ('Circa 10,8: crește proporțional cu α (9,0 × 0,6/0,5).', 'About 10.8: it grows in proportion to α (9.0 × 0.6/0.5).'),
               ('Rămîne 9,0: boltirea depinde doar de distribuția inovațiilor.', 'Still 9.0: the kurtosis depends only on the innovation distribution.')]),
 'simulated': (0, [('Circa 4,9: mult sub valoarea teoretică.', 'About 4.9: far below the theoretical value.'),
                   ('Circa 9,0: 1000 de valori ajung pentru valoarea teoretică.', 'About 9.0: 1000 values are enough for the theoretical value.'),
                   ('Circa 3,0: inovațiile sînt Normale, deci și seria.', 'About 3.0: the innovations are Normal, so the series is too.')]),
 'archq': (0, [('GARCH(1,1): o verosimilitate mai mare cu mai puțini parametri.', 'GARCH(1,1): a higher likelihood with fewer parameters.'),
               ('ARCH(10): are mai mulți parametri, deci descrie mai bine datele.', 'ARCH(10): it has more parameters, so it describes the data better.'),
               ('Niciunul: diferența de 9,6 în log-verosimilitate nu contează.', 'Neither: the difference of 9.6 in log-likelihood does not matter.')]),
 'kurt': (0, [('Infinită: (α + β)² + 2α² = 1,0001.', 'Infinite: (α + β)² + 2α² = 1.0001.'), ('Circa 6,1: β crește cu doar 0,01.', 'About 6.1: β rises by only 0.01.'),
              ('Nici varianța necondiționată nu mai există: modelul devine nestaționar.', 'The unconditional variance no longer exists either: the model becomes non-stationary.')]),
 'half-life': (0, [('Circa 69 de zile: aproape dublu.', 'About 69 days: almost double.'), ('Circa 35 de zile: persistența crește cu doar 1%.', 'About 35 days: the persistence rises by only 1%.'),
                   ('Infinit: la 0,99 șocurile nu se mai sting.', 'Infinite: at 0.99 shocks no longer die out.')]),
 'step': (0, [('1,976: contează doar ε².', '1.976: only ε² matters.'), ('1,376: 0,02 + 0,10 × 3 + 0,88 × 1,2.', '1.376: 0.02 + 0.10 × 3 + 0.88 × 1.2.'),
              ('1,076: o creștere nu adaugă nimic la varianță.', '1.076: a rise adds nothing to the variance.')]),
 'igarch': (0, [('EWMA păstrează nivelul de criză pentru tot anul; GARCH îl lasă să scadă spre nivelul pe termen lung.', 'EWMA keeps the crisis level for the whole year; GARCH lets it fall towards the long-run level.'),
                ('EWMA coboară prea repede spre nivelul pe termen lung.', 'EWMA falls too fast towards the long-run level.'),
                ('EWMA crește cu ω în fiecare zi, deci supraestimează tot mai mult.', 'EWMA grows by ω every day, so it overstates more and more.')]),
 'lik-arch1': (0, [('Scade la circa 0,035, aproximativ 0,127/√10.', 'It falls to about 0.035, roughly 0.127/√10.'), ('Scade la circa 0,013: de zece ori mai mică.', 'It falls to about 0.013: ten times smaller.'),
                   ('Rămîne circa 0,127: eroarea standard depinde doar de α.', 'It stays about 0.127: the standard error depends only on α.')]),
 'lik-garch': (0, [('Aproape de valorile adevărate (0,1; 0,8).', 'Close to the true values (0.1, 0.8).'), ('Tot în jur de (0,045; 0,63): creasta nu depinde de n.', 'Still around (0.045, 0.63): the ridge does not depend on n.'),
                   ('Pe frontiera α + β = 1: eșantioanele lungi duc spre IGARCH.', 'On the boundary α + β = 1: long samples lead to IGARCH.')]),
 'step-table': (0, [('Circa 0,012: de circa 1,4 ori mai mare.', 'About 0.012: about 1.4 times larger.'), ('Tot 0,0086: SE robustă diferă doar cînd estimările diferă.', 'Still 0.0086: the robust SE differs only when the estimates differ.'),
                    ('Circa 0,006: SE robuste sînt mai mici.', 'About 0.006: robust SE are smaller.')]),
 'dist-table': (0, [('Puțin: α 0,121 → 0,123, β 0,860 → 0,872.', 'Little: α 0.121 → 0.123, β 0.860 → 0.872.'),
                    ('α se dublează: cozile groase cer o reacție mai puternică la șocuri.', 'α doubles: heavy tails need a stronger reaction to shocks.'),
                    ('β scade sub 0,8: inovațiile t explică singure volatility clustering.', 'β falls below 0.8: t innovations alone explain volatility clustering.')]),
 'qq': (0, [('Circa −4,19.', 'About −4.19.'), ('Circa −5,08: cuantila t fără rescalarea la varianța 1.', 'About −5.08: the t quantile without the rescaling to variance 1.'),
            ('Circa −3,09: la 0,1% distribuțiile t și Normală coincid.', 'About −3.09: at 0.1% the t and the Normal distributions coincide.')]),
 'mk-table': (0, [('O reacție mai puternică la șocul de ieri.', "A stronger reaction to yesterday's shock."), ('O memorie mai lungă a varianței.', 'A longer memory of the variance.'),
                  ('O volatilitate pe termen lung mai mică.', 'A lower long-run volatility.')]),
 'longrun': (0, [('Nu neapărat: împărțim la 1 − α − β = 0,0053, un număr mic și imprecis.', 'Not necessarily: we divide by 1 − α − β = 0.0053, a small and imprecise number.'),
                 ('Da: un model corect reproduce volatilitatea de selecție.', 'Yes: a correct model reproduces the sample volatility.'),
                 ('Da: inovațiile Student-t umflă volatilitatea pe termen lung.', 'Yes: Student-t innovations inflate the long-run volatility.')]),
 'vol-sd': (0, [('Circa 93%: mai jos decît în 2020.', 'About 93%: lower than in 2020.'), ('Peste 111%: criza din 2008 a fost mai lungă, deci și vîrful este mai mare.', 'Above 111%: the 2008 crisis was longer, so the peak is higher too.'),
                ('Circa 19%: volatilitatea de selecție a întregului eșantion.', 'About 19%: the sample volatility of the whole sample.')]),
 'vol-bb': (0, [('De circa 4,1 ori.', 'About 4.1 times.'), ('De circa 17 ori: raportul varianțelor.', 'About 17 times: the ratio of the variances.'),
                ('De circa 1,2 ori: diferența vine doar din calendarul de 365 de zile.', 'About 1.2 times: the difference comes only from the 365-day calendar.')]),
 'persistence': (0, [('Nu există: α + β = 1, șocurile nu se sting.', 'There is none: α + β = 1, shocks do not die out.'), ('Circa 7 zile: piața cripto absoarbe repede șocurile.', 'About 7 days: the crypto market absorbs shocks fast.'),
                     ('131 de zile, ca la S&P 500: toate piețele au aceeași memorie.', '131 days, as for the S&P 500: all markets have the same memory.')]),
 'gjr': (0, [('Cu 0: doar șocurile negative cresc varianța.', 'By 0: only negative shocks raise the variance.'), ('Cu 0,82 = 0,205 × 4: γ se aplică oricărui șoc.', 'By 0.82 = 0.205 × 4: γ applies to every shock.'),
             ('Cu 0,41 = 0,205 × 2: γ înmulțește șocul, nu pătratul lui.', 'By 0.41 = 0.205 × 2: γ multiplies the shock, not its square.')]),
 'nic': (0, [('Scade spre 1 (circa 1,16).', 'It falls towards 1 (about 1.16).'), ('Rămîne 1,62: raportul depinde doar de γ.', 'It stays 1.62: the ratio depends only on γ.'),
             ('Crește peste 3: asimetria se amplifică în perioadele agitate.', 'It rises above 3: asymmetry is amplified in turbulent periods.')]),
 'asym-table': (0, [('BIC penalizează parametrul în plus cu ln T = 8,8, mai mult decît LR = 4,9.', 'BIC penalises the extra parameter by ln T = 8.8, more than LR = 4.9.'),
                    ('Testul LR nu este valid cînd inovațiile sînt Student-t.', 'The LR test is not valid when the innovations are Student-t.'),
                    ('γ̂ are semnul greșit pentru un efect de levier.', 'γ̂ has the wrong sign for a leverage effect.')]),
 'acf-diag': (0, [('Aproape 0 (−0,005).', 'Almost 0 (−0.005).'), ('Tot circa 0,31: GARCH nu schimbă datele.', 'Still about 0.31: GARCH does not change the data.'),
                  ('Circa 0,15: jumătate din clustering rămîne.', 'About 0.15: half of the clustering is left.')]),
 'diag-table': (0, [('Nu complet: lipsește asimetria.', 'Not entirely: the asymmetry is missing.'), ('Da: Q(10) nu respinge, deci modelul este corect.', 'Yes: Q(10) does not reject, so the model is correct.'),
                    ('Nu: p = 0,192 înseamnă că modelul este greșit cu probabilitatea 81%.', 'No: p = 0.192 means the model is wrong with probability 81%.')]),
 'ms-table': (0, [('Cozile groase: 341, față de 276.', 'Heavy tails: 341, against 276.'), ('Asimetria: efectul de levier al S&P 500 este foarte puternic.', 'Asymmetry: the S&P 500 leverage effect is very strong.'),
                  ('La fel: fiecare reduce BIC cu circa 325, jumătate din 649.', 'The same: each lowers BIC by about 325, half of 649.')]),
 'fc-we': (0, [('Circa 2,38%: mai mult decît regula.', 'About 2.38%: more than the rule.'), ('Tot 2,26%: varianțele zilnice sînt egale.', 'Still 2.26%: the daily variances are equal.'),
               ('Circa 5,45% = √(10 × 2,972): nivelul pe termen lung, imediat.', 'About 5.45% = √(10 × 2.972): the long-run level, at once.')]),
 'term': (0, [('Circa 85%.', 'About 85%.'), ('Circa 27%: nivelul pe termen lung.', 'About 27%: the long-run level.'), ('Tot 111%: prognoza rămîne constantă.', 'Still 111%: the forecast stays constant.')]),
 'fc-eval': (0, [('Nicio diferență detectabilă: GARCH-ul Bitcoin este aproape un EWMA.', 'No detectable difference: the Bitcoin GARCH is almost an EWMA.'),
                 ('GARCH-t este mai bun: QLIKE-ul lui este mai mic.', 'GARCH-t is better: its QLIKE is smaller.'), ('Prognozele sînt egale: p = 0,845 o arată.', 'The forecasts are equal: p = 0.845 shows it.')]),
 'var-we': (0, [('Circa 6,09%.', 'About 6.09%.'), ('Tot 5,52%: regula √10 este exactă pentru GARCH.', 'Still 5.52%: the √10 rule is exact for GARCH.'),
                ('Circa 17,5% = 10 × 1,75: VaR-urile zilnice se adună.', 'About 17.5% = 10 × 1.75: the daily VaRs add up.')]),
 'var': (0, [('De 13 ori.', '13 times.'), ('De circa 5 ori: un VaR 1% este depășit în 1% dintre zile.', 'About 5 times: a VaR 1% is exceeded on 1% of the days.'),
             ('O singură dată: VaR 1% înseamnă o depășire.', 'Once: VaR 1% means one exceedance.')]),
 'exc': (0, [('Cuantila Normală este prea puțin adîncă pentru cozile groase.', 'The Normal quantile is not deep enough for heavy tails.'), ('EWMA supraestimează volatilitatea.', 'EWMA overstates the volatility.'),
             ('Un VaR 1% corect trebuie depășit în 2,5% dintre zile.', 'A correct VaR 1% must be exceeded on 2.5% of the days.')]),
}

# blocks whose initial chart, table or worked example shows the answer: veil the readout and those elements
VEIL = {'returns': {'out': True, 'charts': ['rt-chart']}, 'mix': {'out': True, 'charts': ['mx-we-box']}, 'lik-garch': {'out': True, 'charts': ['lg-chart']},
        'step-table': {'out': True, 'charts': ['et-chart', 'et-tab']}, 'dist-table': {'out': True, 'charts': ['dt-chart', 'dt-tab']}, 'qq': {'out': True, 'charts': ['qq-chart']},
        'vol-sd': {'out': True, 'charts': ['v1-chart']}, 'vol-bb': {'out': True, 'charts': ['v2-chart']}, 'persistence': {'out': True, 'charts': ['ps-chart']},
        'gjr': {'out': True, 'charts': ['gj-we-box']}, 'acf-diag': {'out': True, 'charts': ['ad-chart']}, 'ms-table': {'out': True, 'charts': ['ms-chart', 'ms-tab']},
        'fc-we': {'out': True, 'charts': ['fw-chart', 'fw-we-box']}, 'term': {'out': True, 'charts': ['ts-chart']}, 'var-we': {'out': True, 'charts': ['vw-chart', 'vw-we-box']},
        'var': {'out': True, 'charts': ['vr-chart']}, 'simulated': {'out': True}, 'lik-arch1': {'out': True}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('ω = 0,05, α = 0,05, β = 0,90. Care sînt varianța pe termen lung și timpul de înjumătățire?', 'ω = 0.05, α = 0.05, β = 0.90. What are the long-run variance and the half-life?'),
      key=0, opts=[
  (('1,00 și 13,5 zile.', '1.00 and 13.5 days.'), ('σ̄² = 0,05/(1 − 0,95) = 1,00; h½ = ln 0,5/ln 0,95 = 13,5 zile.', 'σ̄² = 0.05/(1 − 0.95) = 1.00; h½ = ln 0.5/ln 0.95 = 13.5 days.')),
  (('0,50 și 6,6 zile.', '0.50 and 6.6 days.'), ('Ignoră α: ω/(1 − β) = 0,5 și ln 0,5/ln 0,90 = 6,6; persistența este α + β = 0,95.', 'It ignores α: ω/(1 − β) = 0.5 and ln 0.5/ln 0.90 = 6.6; the persistence is α + β = 0.95.')),
  (('1,00 și 10,0 zile.', '1.00 and 10.0 days.'), ('0,5/0,05 = 10 presupune o scădere liniară a șocului; scăderea este geometrică, cu factorul 0,95 pe zi.', '0.5/0.05 = 10 assumes a linear decay of the shock; the decay is geometric, by the factor 0.95 a day.'))]),
 dict(q=('GJR-GARCH dă α̂ = 0 și γ̂ = 0,2. Ce se întîmplă cu varianța de mîine după o creștere de 3%?', "GJR-GARCH gives α̂ = 0 and γ̂ = 0.2. What happens to tomorrow's variance after a 3% rise?"),
      key=0, opts=[
  (('Șocul nu adaugă nimic: σ²(t + 1) = ω + βσ²(t).', 'The shock adds nothing: σ²(t + 1) = ω + βσ²(t).'),
   ('Pentru ε(t) > 0 indicatorul I este 0 și panta este α = 0: ziua contează ca una cu ε(t) = 0; doar șocurile negative cresc varianța.', 'For ε(t) > 0 the indicator I is 0 and the slope is α = 0: the day counts as one with ε(t) = 0; only negative shocks raise the variance.')),
  (('Crește cu 0,2 × 9 = 1,8.', 'It rises by 0.2 × 9 = 1.8.'), ('γ se aplică doar șocurilor negative (I = 1 dacă ε < 0).', 'γ applies only to negative shocks (I = 1 if ε < 0).')),
  (('Scade cu 0,2 × 9 = 1,8: creșterile reduc volatilitatea.', 'It falls by 0.2 × 9 = 1.8: rises lower volatility.'), ('Toți parametrii sînt nenegativi: niciun șoc nu scade varianța sub ω + βσ²(t).', 'All parameters are non-negative: no shock lowers the variance below ω + βσ²(t).'))]),
 dict(q=('Q(10) al lui ẑ² are p = 0,40. Este modelul corect?', 'Q(10) of ẑ² has p = 0.40. Is the model correct?'),
      key=0, opts=[
  (('Știm doar că nu au rămas efecte ARCH; pot rămîne asimetria sau o distribuție greșită.', 'We only know that no ARCH effects are left; asymmetry or a wrong distribution may remain.'),
   ('Un test care nu respinge nu dovedește corectitudinea; testul de asimetrie și graficul QQ verifică alte aspecte (la S&P 500 asimetria lipsea, deși Q(10) avea p = 0,192).', 'A test that does not reject does not prove correctness; the sign-bias test and the QQ plot check other aspects (for the S&P 500 the asymmetry was missing although Q(10) had p = 0.192).')),
  (('Da, cu probabilitatea 40%.', 'Yes, with probability 40%.'), ('p nu este probabilitatea ca modelul să fie corect, ci probabilitatea unei statistici cel puțin la fel de mari dacă H₀ este adevărată.', 'p is not the probability that the model is correct, but the probability of a statistic at least as large if H₀ is true.')),
  (('Nu: p sub 0,5 arată că modelul este respins.', 'No: p below 0.5 shows that the model is rejected.'), ('La nivelul de 5% respingem doar dacă p < 0,05.', 'At the 5% level we reject only if p < 0.05.'))]),
 dict(q=('μ = 0,08, σ(t + 1) = 0,714 și q₀,₀₁(z) = −2,557. Cît este VaR 1% pe o zi?', 'μ = 0.08, σ(t + 1) = 0.714 and q₀.₀₁(z) = −2.557. What is the one-day VaR 1%?'),
      key=0, opts=[
  (('1,75%: −(0,08 + 0,714 × (−2,557)).', '1.75%: −(0.08 + 0.714 × (−2.557)).'), ('0,714 × 2,557 = 1,826; 1,826 − 0,08 = 1,75: media pozitivă reduce pierderea.', '0.714 × 2.557 = 1.826; 1.826 − 0.08 = 1.75: the positive mean reduces the loss.')),
  (('1,91%: 0,08 + 0,714 × 2,557.', '1.91%: 0.08 + 0.714 × 2.557.'), ('Adună media în loc să o scadă: VaR = −(μ + σq), cu q negativ.', 'It adds the mean instead of subtracting it: VaR = −(μ + σq), with q negative.')),
  (('1,66%: 0,714 × 2,326.', '1.66%: 0.714 × 2.326.'), ('Folosește cuantila Normală și ignoră μ; cu z Student-t cuantila este −2,557.', 'It uses the Normal quantile and ignores μ; with Student-t z the quantile is −2.557.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch9-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH9_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH9_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH9_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', ' ', t)
    for lg in QA:
        for a, (q, ans) in QA[lg].items():
            QA[lg][a] = (q.replace('<', '&lt;'), ans.replace('<', '&lt;'))
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
