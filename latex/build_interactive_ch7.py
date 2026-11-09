"""Build the interactive pages of Chapter 7: interactive/ch7.html (EN) and interactive/ch7_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch7.js, inputs in interactive/ch7_data.js (Quantlets/Ch_07/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch7.js moves each one to its
own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 7
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_07'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('rw-paths', 'sfm_ch7_rw_paths', 'prev'), ('scatter-lag', 'sfm_ch7_scatter_lag', 'prev'),
          ('white-noise', 'sfm_ch7_white_noise', 'wn'), ('acf-processes', 'sfm_ch7_acf_processes', 'wn'), ('acf-returns', 'sfm_ch7_acf_returns', 'wn'),
          ('acf-abs', 'sfm_ch7_acf_abs', 'wn'),
          ('spurious', 'sfm_ch7_spurious', 'ur'), ('unit-root-bet', 'sfm_ch7_unit_root_bet', 'ur'),
          ('vr-size', 'sfm_ch7_vr_size', 'vr'), ('vr-profile', 'sfm_ch7_vr_profile', 'vr'),
          ('vr-markets', 'sfm_ch7_vr_markets', 'mk'), ('rolling-vr', 'sfm_ch7_rolling_vr', 'mk'),
          ('calendar', 'sfm_ch7_calendar', 'an')]
# table and worked-example blocks: (anchor, frame number in the RO deck (EN = RO - 1), section key, chart block that carries it on the web page)
XBLOCKS = [('sqrt-time', 23, 'rw', 'scatter-lag'), ('lb-example', 32, 'act', 'acf-abs'), ('ac-tests', 34, 'act', 'acf-abs'), ('runs', 35, 'act', 'acf-abs'),
           ('adf-example', 39, 'ur', 'spurious'), ('ur-table', 42, 'ur', 'unit-root-bet'), ('vr-acf', 45, 'vr', 'vr-size'), ('lm-tests', 46, 'vr', 'vr-size'),
           ('vr-table', 49, 'vr', 'vr-profile'), ('cd-table', 52, 'vr', 'vr-profile'), ('sub-table', 58, 'mk', 'rolling-vr')]
ORDER = ['rw-paths', 'scatter-lag', 'sqrt-time', 'white-noise', 'acf-processes', 'acf-returns', 'acf-abs', 'lb-example', 'ac-tests', 'runs',
         'spurious', 'adf-example', 'unit-root-bet', 'ur-table', 'vr-acf', 'lm-tests', 'vr-size', 'vr-profile', 'vr-table', 'cd-table',
         'vr-markets', 'rolling-vr', 'sub-table', 'calendar']
TITLES = {'sqrt-time': 'Exemplu lucrat: riscul în funcție de orizont', 'lb-example': 'Exemplu lucrat: Ljung–Box pas cu pas',
          'ac-tests': 'Teste de autocorelație pe date reale', 'runs': 'Testul runs', 'adf-example': 'Testul augmentat și un exemplu lucrat',
          'ur-table': 'Teste de rădăcină unitară pe prețuri și randamente', 'vr-acf': 'Exemplu lucrat: VR din ACF-ul S&P 500',
          'lm-tests': 'Lo și MacKinlay (1988): estimator și teste', 'vr-table': 'Rapoarte ale varianțelor și teste robuste',
          'cd-table': 'Teste comune și automate pe date reale', 'sub-table': 'Trei subperioade pentru fiecare piață'}

PFX = {'rw-paths': 'rw', 'scatter-lag': 'sc', 'sqrt-time': 'st', 'white-noise': 'wn', 'acf-processes': 'ap', 'acf-returns': 'ar', 'acf-abs': 'aa', 'lb-example': 'lb',
       'ac-tests': 'at', 'runs': 'ru', 'spurious': 'sp', 'adf-example': 'df', 'unit-root-bet': 'ub', 'ur-table': 'ut', 'vr-acf': 'va', 'lm-tests': 'lm', 'vr-size': 'vs',
       'vr-profile': 'vp', 'vr-table': 'vt', 'cd-table': 'cd', 'vr-markets': 'vm', 'rolling-vr': 'rv', 'sub-table': 'sb', 'calendar': 'ca'}

SECTIONS = {
 'ro': dict(prev='Previzibilitatea prețurilor', rw='Ipotezele mersului aleator', wn='Zgomotul alb și funcția de autocorelație', act='Testarea autocorelației',
            ur='Rădăcini unitare: prețuri față de randamente', vr='Testul variance ratio', mk='Eficiența pe piețe și în timp',
            an='Forma semi-tare, forma tare și anomalii'),
 'en': dict(prev='Predictability of Prices', rw='Random Walk Hypotheses', wn='White Noise and the Autocorrelation Function', act='Testing for Autocorrelation',
            ur='Unit Roots: Prices against Returns', vr='The Variance-Ratio Test', mk='Efficiency across Markets and over Time',
            an='Semi-Strong Form, Strong Form and Anomalies'),
}

L = {
 'ro': dict(file='ch7_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 7 · Ipoteza piețelor eficiente, mersul aleator și testele VR', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: ACF (autocorrelation function, funcția de autocorelație), VR (variance ratio, raportul varianțelor), '
        'ADF (augmented Dickey–Fuller, Dickey–Fuller augmentat), PP (Phillips–Perron), KPSS (Kwiatkowski–Phillips–Schmidt–Shin), LB (Ljung–Box), CD (Chow–Denning), '
        'OLS (ordinary least squares, metoda celor mai mici pătrate), GARCH (generalised autoregressive conditional heteroskedasticity).',
   src='Randamente logaritmice zilnice, în %: S&P 500 și DAX din ianuarie 1990, BET din septembrie 1997, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV), '
       'OMV Petrom (SNP), BRD și Transgaz (TGN) din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. Comparația '
       'între piețe: 16 indici și criptomonede, 19 septembrie 2016 – 18 septembrie 2026. Prețuri de închidere EODHD. Mersurile aleatoare simulate, zgomotul alb, traiectoriile ARMA, '
       'regresiile false, simularea mărimii testelor Z și Z*, ordinele de lag ADF alese prin AIC și p-value-urile wild bootstrap sînt cele din curs (același cod, aceeași sămînță a '
       'generatorului); autocorelațiile, testele portmanteau și runs, regresiile Dickey–Fuller, PP, KPSS, rapoartele varianțelor, testul Chow–Denning, testul VR automat, '
       'ferestrele mobile, subperioadele și regresiile calendaristice sînt recalculate în browser din aceleași date. Alte lungimi T ale regresiei false, alte valori α ale '
       'simulării GARCH, alte lățimi de bandă, laguri, orizonturi și începuturi ale eșantionului sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch7.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 7 · Efficient Markets, Random Walk and Variance-Ratio Tests', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: ACF (autocorrelation function), VR (variance ratio), ADF (augmented Dickey–Fuller), '
        'PP (Phillips–Perron), KPSS (Kwiatkowski–Phillips–Schmidt–Shin), LB (Ljung–Box), CD (Chow–Denning), OLS (ordinary least squares), '
        'GARCH (generalised autoregressive conditional heteroskedasticity).',
   src='Daily log returns, in %: S&P 500 and DAX from January 1990, BET from September 1997, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV), OMV Petrom (SNP), '
       'BRD and Transgaz (TGN) from 2010, from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. Cross-market comparison: 16 indices and '
       'cryptocurrencies, 19 September 2016 to 18 September 2026. EODHD closing prices. The simulated random walks, the white noise, the ARMA paths, the spurious regressions, '
       'the size simulation of Z and Z*, the ADF lag orders chosen by AIC and the wild-bootstrap p-values are those of the lecture (the same code and the same generator seed); '
       'the autocorrelations, the portmanteau and runs tests, the Dickey–Fuller regressions, PP, KPSS, the variance ratios, the Chow–Denning test, the automatic VR test, the rolling '
       'windows, the sub-periods and the calendar regressions are recomputed in the browser from the same data. Other lengths T of the spurious regression, other values α of the '
       'GARCH simulation, and other bandwidths, lags, horizons and sample starts are extra runs of the same code.'),
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
    S8 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)'), ('brd', 'BRD'),
          ('tgn', 'Transgaz (TGN)')]
    ser = 'Seria' if ro else 'Series'
    wser = 'Seria exemplului lucrat' if ro else 'Series of the worked example'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    y0 = 'Începutul eșantionului (anul)' if ro else 'Start of the sample (year)'
    c = (lambda s: s.replace('.', '{,}')) if ro else (lambda s: s)
    Q4 = [('2', '2'), ('5', '5'), ('10', '10'), ('20', '20')]
    return {
     'rw-paths': (slider('rw-y', 'Traiectoriile pînă la sfîrșitul anului' if ro else 'Paths up to the end of', 1991, 2026, 1, 2026) + play('rw'), CSS + charts('rw-chart')),
     'scatter-lag': (slider('sc-k', 'Lagul k: r(t) față de r(t − k)' if ro else 'Lag k: r(t) against r(t − k)', 1, 10, 1, 1) + play('sc'), charts('sc-chart')),
     'sqrt-time': (select('st-s', ser, S8, 'sp500') + play('st'),
                   charts('st-chart') + xh('Abaterea standard a randamentelor pe q zile: regula rădăcinii pătrate și datele' if ro else 'Standard deviation of q-day returns: the square-root rule and the data') + tw('st-tab')
                   + we('st-we', f'{WE}: {"regula rădăcinii pătrate a timpului" if ro else "the square-root-of-time rule"}',
                        r'\mathrm{sd}\big(r_t(q)\big) = \sigma\sqrt{q}, \qquad \frac{\mathrm{sd}\big(r_t(q)\big)}{\sigma\sqrt{q}} \approx \sqrt{\mathrm{VR}(q)}',
                        slider('st-q', 'Orizontul q al exemplului lucrat (zile)' if ro else 'Horizon q of the worked example (days)', 1, 250, 1, 60))),
     'white-noise': (slider('wn-n', 'Numărul de extrageri n' if ro else 'Number of draws n', 100, 1000, 100, 1000) + play('wn'), charts('wn-chart')),
     'acf-processes': (slider('ap-n', 'Lungimea traiectoriei n' if ro else 'Length of the path n', 50, 1000, 50, 1000) + play('ap'), charts('ap-chart')),
     'acf-returns': (slider('ar-y', y0, 1990, 2022, 2, 1990) + play('ar'),
                     charts('ar-chart') + we('ar-we', f'{WE}: {"banda robustă a ACF-ului" if ro else "the robust band of the ACF"}',
                                             c(r'\hat\delta(k) = \frac{T\sum_t e_t^2 e_{t-k}^2}{\big(\sum_t e_t^2\big)^2}, \qquad \pm 1.96\sqrt{\hat\delta(k)/T}, \qquad e_t = r_t - \bar r'),
                                             select('ar-s', wser, S8[:4], 'sp500') + slider('ar-k', 'Lagul k' if ro else 'Lag k', 1, 20, 1, 1))),
     'acf-abs': (slider('aa-d', 'Puterea d din |r(t)|^d' if ro else 'Power d in |r(t)|^d', 0.25, 2, 0.25, 1) + play('aa'), charts('aa-chart')),
     'lb-example': (slider('lb-T', 'Numărul de randamente T' if ro else 'Number of returns T', 500, 5000, 500, 1000)
                    + select('lb-m', 'Laguri testate m' if ro else 'Lags tested m', [('1', '1'), ('2', '2'), ('3', '3')], '3') + play('lb'),
                    charts('lb-chart') + we('lb-we', f'{WE}: Box–Pierce {"și" if ro else "and"} Ljung–Box',
                                            r'Q_{BP}(m) = T\sum_{k=1}^{m}\hat\rho(k)^2, \qquad Q_{LB}(m) = T(T+2)\sum_{k=1}^{m}\frac{\hat\rho(k)^2}{T-k} \approx \chi^2(m)')),
     'ac-tests': (slider('at-m', 'Numărul de laguri m' if ro else 'Number of lags m', 1, 30, 1, 10) + play('at'),
                  charts('at-chart') + xh('Teste pe randamentele zilnice (p-value sub 0,05 cu roșu)' if ro else 'Tests on the daily returns (p-value below 0.05 in red)') + tw('at-tab')
                  + we('at-we', f'{WE}: {"Q<sub>LB</sub> și Q̃ pe date reale" if ro else "Q<sub>LB</sub> and Q̃ on real data"}',
                       r'\tilde Q(m) = \sum_{k=1}^{m}\frac{T\hat\rho(k)^2}{\hat\delta(k)} \approx \chi^2(m)', select('at-s', wser, S8, 'sp500'))),
     'runs': (slider('ru-y', y0, 1990, 2022, 2, 1990) + play('ru'),
              charts('ru-chart') + tw('ru-tab') + we('ru-we', f'{WE}: {"testul runs" if ro else "the runs test"}',
                                                     r'E[R] = \frac{2n_1n_2}{n} + 1, \quad \mathrm{Var}(R) = \frac{2n_1n_2(2n_1n_2 - n)}{n^2(n-1)}, \quad z = \frac{R - E[R]}{\sqrt{\mathrm{Var}(R)}}',
                                                     select('ru-s', wser, S8, 'bet'))),
     'spurious': (slider('sp-T', 'Lungimea seriilor T (100, 250, 500, 1 000, 2 000)' if ro else 'Length of the series T (100, 250, 500, 1,000, 2,000)', 0, 4, 1, 2) + play('sp'),
                  charts('sp-chart')),
     'adf-example': (select('df-s', ser, S8, 'bet') + select('df-c', 'Termenii deterministici' if ro else 'Deterministic terms', [('c', 'constantă' if ro else 'constant'), ('ct', 'constantă și tendință' if ro else 'constant and trend')], 'c') + play('df'),
                     charts('df-chart') + we('df-we', f'{WE}: {"regresia Dickey–Fuller" if ro else "the Dickey–Fuller regression"}',
                                             r'\Delta p_t = c + bt + \gamma\, p_{t-1} + \sum_{j=1}^{L}\delta_j\Delta p_{t-j} + \varepsilon_t, \qquad \tau = \frac{\hat\gamma}{\mathrm{SE}(\hat\gamma)}',
                                             slider('df-L', 'Laguri L ale diferențelor' if ro else 'Lags L of the differences', 0, 12, 1, 0))),
     'unit-root-bet': (select('ub-s', ser, S8, 'bet') + play('ub'), charts('ub-chart')),
     'ur-table': (select('ut-l', 'Lățimea de bandă l pentru PP și KPSS' if ro else 'Bandwidth l for PP and KPSS', [('rule', '12(T/100)^(1/4), ca în curs' if ro else '12(T/100)^(1/4), as in the lecture'),
                                                                                                                     ('0', '0'), ('5', '5'), ('10', '10'), ('20', '20'), ('40', '40'), ('80', '80')], 'rule') + play('ut'),
                  charts('ut-chart') + '<div class="wctl">' + select('ut-L', 'Laguri ADF în tabel' if ro else 'ADF lags in the table', [('aic', 'alese prin AIC, ca în curs' if ro else 'chosen by AIC, as in the lecture'),
                                                                                                                                    ('0', '0'), ('1', '1'), ('5', '5'), ('10', '10'), ('20', '20')], 'aic') + '</div>'
                  + xh('Statistici de test (cu roșu: respingere la 5%)' if ro else 'Test statistics (in red: rejection at 5%)') + tw('ut-tab')
                  + we('ut-we', f'{WE}: {"statistica KPSS pe prețul logaritmic" if ro else "the KPSS statistic on the log price"}',
                       r'\mathrm{KPSS} = \frac{\sum_{t=1}^{T} S_t^2}{T^2\hat\lambda^2}, \quad S_t = \sum_{s \le t} e_s, \quad \hat\lambda^2 = \hat\gamma(0) + 2\sum_{j=1}^{l}\Big(1 - \frac{j}{l+1}\Big)\hat\gamma(j)',
                       select('ut-s', wser, S8, 'sp500'))),
     'vr-acf': (slider('va-q', 'Orizontul q' if ro else 'Horizon q', 2, 20, 1, 5) + select('va-s', ser, S8, 'sp500') + play('va'),
                charts('va-chart') + we('va-we', f'{WE}: {"VR din ACF" if ro else "VR from the ACF"}', r'\mathrm{VR}(q) = 1 + 2\sum_{k=1}^{q-1}\Big(1 - \frac{k}{q}\Big)\hat\rho(k)')),
     'lm-tests': (slider('lm-q', 'Orizontul q' if ro else 'Horizon q', 2, 40, 1, 2) + play('lm'),
                  charts('lm-chart') + we('lm-we', f'{WE}: Z(q) {"și" if ro else "and"} Z*(q)',
                                          r'Z(q) = \frac{\widehat{\mathrm{VR}}(q) - 1}{\sqrt{2(2q-1)(q-1)/(3qT)}}, \qquad Z^*(q) = \frac{\widehat{\mathrm{VR}}(q) - 1}{\sqrt{\hat\theta(q)/T}}, \qquad \hat\theta(q) = \sum_{j=1}^{q-1}\Big[\frac{2(q-j)}{q}\Big]^2\hat\delta(j)',
                                          select('lm-s', wser, S8, 'sp500'))),
     'vr-size': (slider('vs-a', 'Reacția GARCH α (β = 0,86)' if ro else 'GARCH reaction α (β = 0.86)', 0, 4, 1, 4) + play('vs'), charts('vs-chart')),
     'vr-profile': (slider('vp-y', y0, 1990, 2022, 2, 1990) + play('vp'), charts('vp-chart')),
     'vr-table': (slider('vt-y', y0, 1990, 2022, 2, 1990) + select('vt-z', 'Statistica' if ro else 'Statistic', [('zs', 'Z* robust, ca în curs' if ro else 'robust Z*, as in the lecture'), ('z', 'Z homoscedastic' if ro else 'homoskedastic Z')], 'zs') + play('vt'),
                  charts('vt-chart') + tw('vt-tab')),
     'cd-table': (select('cd-h', 'Orizonturile testului Chow–Denning' if ro else 'Horizons of the Chow–Denning test', [('lec', '2, 5, 10, 20 (curs)' if ro else '2, 5, 10, 20 (lecture)'), ('pow2', '2, 4, 8, 16'), ('short', '2, 3, 4, 5'),
                                                                                                                      ('long', '5, 10, 20, 40'), ('all', '2, 3, …, 20')], 'lec')
                  + select('cd-a', 'Nivelul de semnificație' if ro else 'Significance level', [('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%')], '0.05') + play('cd'),
                  charts('cd-chart') + tw('cd-tab')
                  + we('cd-we', f'{WE}: {"testul Chow–Denning" if ro else "the Chow–Denning test"}', r'\mathrm{CD} = \max_i |Z^*(q_i)|, \qquad P(\mathrm{CD} \le c) = \big(2\Phi(c) - 1\big)^m',
                       select('cd-s', wser, S8, 'tgn'))
                  + we('av-we', f'{WE}: {"testul VR automat" if ro else "the automatic VR test"}',
                       c(r'\hat k = 1.3221\,(\hat a_2 T)^{1/5}, \quad \hat a_2 = \frac{4\hat\rho^2}{(1-\hat\rho)^4}, \qquad \frac{\sqrt{T/\hat k}\,\big(\mathrm{VR}(\hat k) - 1\big)}{\sqrt 2} \approx N(0, 1)'),
                       select('cd-s2', wser, S8, 'sp500'))),
     'vr-markets': (slider('vm-q', 'Orizontul q' if ro else 'Horizon q', 2, 20, 1, 5) + select('vm-c', 'Nivelul intervalului' if ro else 'Interval level', [('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')], '0.95') + play('vm'),
                    charts('vm-chart')),
     'rolling-vr': (select('rv-w', 'Lungimea ferestrei (zile)' if ro else 'Window length (days)', [('250', '250'), ('500', '500'), ('750', '750'), ('1000', '1 000' if ro else '1,000'), ('1500', '1 500' if ro else '1,500')], '500')
                    + select('rv-q', 'Orizontul q' if ro else 'Horizon q', Q4, '5') + play('rv'), charts('rv-chart')),
     'sub-table': (select('sb-q', 'Orizontul q' if ro else 'Horizon q', Q4, '5') + play('sb'), charts('sb-chart') + tw('sb-tab')),
     'calendar': (select('ca-v', 'Gruparea' if ro else 'Grouping', [('day', 'zile ale săptămînii' if ro else 'days of the week'), ('month', 'luni' if ro else 'months')], 'day')
                  + select('ca-l', 'Laguri Newey–West' if ro else 'Newey–West lags', [('0', '0'), ('5', '5 (curs)' if ro else '5 (lecture)'), ('10', '10'), ('20', '20')], '5')
                  + slider('ca-y', y0, 1990, 2022, 2, 1990) + play('ca'),
                  charts('ca-chart') + we('ca-we', f'{WE}: {"efectul ianuarie" if ro else "the January effect"}',
                                          r'r_t = a + b\,\mathbf{1}\{\text{' + ('ianuarie' if ro else 'January') + r'}\}_t + u_t, \qquad t = \frac{\hat b}{\mathrm{SE}_{NW}(\hat b)}',
                                          select('ca-s', wser, [('sp500', 'S&P 500'), ('bet', 'BET')], 'bet'))),
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
        k = fr - 1 if lang == 'ro' else fr - 2
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch7.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch7_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch7_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => `${s} ${de(s)}zile`;
window.CH7_TEXT = {
 h: { real: 'sd pe q zile (%)', ratio: 'raport', from: 'din', price: 'preț logaritmic (constantă, tendință)', rets: 'randamente (constantă)', period: 'perioada', avr: 'VR automat', pboot: 'p (bootstrap)' },
 rw: (d, sp, lo, hi, n) => `La ${d}: S&P 500 este la ${sp} (logaritmul prețului minus valoarea de start); cele patru mersuri aleatoare sînt între ${lo} și ${hi}, iar ${n === 1 ? 'unul dintre ele este' : n + ' dintre ele sînt'} deasupra S&P 500. Pașii simulați au media 0,033% și abaterea standard 1,13% pe zi, ca randamentele S&P 500.`,
 sc: (k, rs, bs, rb, bb) => `Lagul ${k}: S&P 500 ρ̂(${k}) = ${rs}, panta dreptei ${bs}; BET ρ̂(${k}) = ${rb}, panta ${bb}. Banda i.i.d. de 95%, ±1,96/√T, este ±0,020 pentru S&P 500 și ±0,023 pentru BET.`,
 st1: (s, sd, n) => `${s}: abaterea standard zilnică σ̂ = ${sd}% (${zile(n)}, numitorul n − 1).`,
 st2: (sd, r5, v5, r20, v20, r250, v250) => `Regula rădăcinii pătrate: q = 5: ${sd} × ${r5} = ${v5}%; q = 20: ${sd} × ${r20} = ${v20}%; q = 250: ${sd} × ${r250} = ${v250}%.`,
 st3: (q, sd, rq, vq, real, ratio) => `Orizontul ales, q = ${q}: σ̂√q = ${sd} × ${rq} = ${vq}%; abaterea standard a randamentelor pe ${zile(q)}, din sumele suprapuse, este ${real}%, adică ${ratio} din valoarea regulii.`,
 st: (s, v, r) => `${s}: regula dă ${v}% pe 250 de zile, sumele suprapuse dau ${r}%. Raportul lor este aproximativ √VR(250): sub 1 cînd randamentele sînt negativ autocorelate, peste 1 cînd sînt pozitiv autocorelate.`,
 wn: (n, b, o, e) => `n = ${n}: banda ±1,96/√n = ±${b}; ${o} din 30 de autocorelații de selecție ies din bandă (în medie ${e} doar din întîmplare).`,
 ap: (n, b, rows) => `n = ${n} (banda zgomotului alb: ±${b}). Cea mai mare abatere |ACF de selecție − ACF teoretic| pe lagurile 1–20: ` + rows.map(([l, d]) => `${l}: ${d}`).join('; ') + '.',
 ar: (y, items) => `Eșantioane de la ${y} (sau de la începutul seriei); din 20 de laguri: ` + items.map(([s, r, a, b]) => `${s} ρ̂(1) = ${r}, ${a} ${de(a)}în afara benzii i.i.d., ${b} în afara benzii robuste`).join('; ') + '.',
 ar1: (s, T, k, s4, s2, d) => `${s}, T = ${T}, lagul k = ${k}: Σ e<sub>t</sub>²e<sub>t−k</sub>² = ${s4}, Σ e<sub>t</sub>² = ${s2}; δ̂(${k}) = T × ${s4}/${s2}² = ${d}.`,
 ar2: (bi, d, T, br) => `Banda i.i.d.: ±1,96/√T = ±${bi}; banda robustă: ±1,96√(δ̂/T) = ±1,96√(${d}/${T}) = ±${br}.`,
 ar3: (k, r, oi, or) => `ρ̂(${k}) = ${r}: ${oi ? 'în afara' : 'în interiorul'} benzii i.i.d., ${or ? 'în afara' : 'în interiorul'} benzii robuste. δ̂ > 1 înseamnă că volatility clustering mărește varianța lui ρ̂(k).`,
 aa: (d, items) => `ACF-ul lui |r<sub>t</sub>|<sup>${d}</sup> la lagurile 1, 20 și 100: ` + items.map(([n, a, b, c]) => `${n} ${a}, ${b}, ${c}`).join('; ') + '. Randamentele însele (linii punctate) rămîn aproape de 0.',
 lb1: (T, r, m) => `T = ${T}; autocorelațiile exemplului: ${r}; testăm m = ${m} ${m === 1 ? 'lag' : 'laguri'}.`,
 lb2: (t, bp) => `Termenii Tρ̂(k)²: ${t}; Q<sub>BP</sub>(m) = ${bp}.`,
 lb3: (T, T2, terms, lb) => `Q<sub>LB</sub>(m) = ${T} × ${T2} × (${terms}) = ${lb}.`,
 lb4: (m, c, lb, rej, p) => `χ²<sub>0,95</sub>(${m}) = ${c}; Q<sub>LB</sub> = ${lb} ${rej ? '&gt;' : '&lt;'} ${c}: ${rej ? 'respingem' : 'nu respingem'} H<sub>0</sub> la 5% (p ${p}).`,
 lb: (T, bp, lb, rej, b) => `T = ${T}: Q<sub>BP</sub> = ${bp}, Q<sub>LB</sub> = ${lb}, H<sub>0</sub> ${rej ? 'respinsă' : 'nerespinsă'} la 5%; banda unei singure autocorelații este ±${b}.`,
 at1: (s, T, m, r) => `${s}, T = ${T}: ρ̂(1), …, ρ̂(${m}) = ${r}.`,
 at2: (T, ss, bp) => `Σ ρ̂(k)² = ${ss}; Q<sub>BP</sub> = T Σ ρ̂(k)² = ${T} × ${ss} = ${bp}.`,
 at3: (lb, r) => `Q<sub>LB</sub> = T(T + 2) Σ ρ̂(k)²/(T − k) = ${lb}; cu T mare, aproape de Q<sub>BP</sub> (raportul ${r}).`,
 at4: (d, rob) => `Statistica robustă împarte fiecare Tρ̂(k)² la δ̂(k); primele valori δ̂(k): ${d}; Q̃ = Σ Tρ̂(k)²/δ̂(k) = ${rob}.`,
 at5: (m, c, lb, plb, rob, prob) => `χ²<sub>0,95</sub>(${m}) = ${c}: Q<sub>LB</sub> = ${lb} (p ${plb}), Q̃ = ${rob} (p ${prob}).`,
 at: (m, c, n, list) => `m = ${m}, valoarea critică ${c}: Q<sub>LB</sub> respinge la 5% pentru ${n} din 8 serii, Q̃ robust pentru ${list.length ? list.join(', ') : 'niciuna'}.`,
 ru1: (s, n1, n2, n) => `${s}: n<sub>1</sub> = ${n1} randamente pozitive, n<sub>2</sub> = ${n2} negative, n = ${n} (zilele cu randament 0 sînt omise).`,
 ru2: (n1, n2, n, m) => `E[R] = 2 × ${n1} × ${n2}/${n} + 1 = ${m}.`,
 ru3: (v, sd) => `Var(R) = 2n<sub>1</sub>n<sub>2</sub>(2n<sub>1</sub>n<sub>2</sub> − n)/(n²(n − 1)) = ${v}; sd(R) = ${sd}.`,
 ru4: (R, m, sd, z, p, zv) => `R = ${R} ${de(R)}secvențe observate; z = (${R} − ${m})/${sd} = ${z}, p ${p}: ${zv < 0 ? 'prea puține secvențe, semnele persistă' : 'prea multe secvențe, semnele alternează'}.`,
 ru: (y, list) => `Din ${y}: testul runs respinge la 5% pentru ${list.length ? list.join(', ') : 'nicio serie'}.`,
 sp: (T, rej, r2, b40, rd, x) => `T = ${T}${x ? ' (rulare suplimentară a codului din curs)' : ''}: în niveluri, |t| > 1,96 în ${rej} din regresii, R² median ${r2}, |t| > 40 în ${b40}; pe diferențe, |t| > 1,96 în ${rd}.`,
 df1: (s, reg, L, n) => `${s}, prețul logaritmic: Δp<sub>t</sub> = c ${reg === 'ct' ? '+ bt ' : ''}+ γp<sub>t−1</sub>${L ? ` + Σ<sub>j=1</sub><sup>${L}</sup> δ<sub>j</sub>Δp<sub>t−j</sub>` : ''} + ε<sub>t</sub>, estimată prin OLS pe ${n} ${de(n)}observații.`,
 df2: (g, se) => `γ̂ = ${g}, SE(γ̂) = ${se}.`,
 df3: (g, se, t) => `τ = γ̂/SE(γ̂) = ${g}/${se} = ${t}.`,
 df4: (t, c, rej, rejN) => `Valoarea critică de 5% (MacKinnon) este ${c}: τ = ${t} ${rej ? '&lt;' : '&gt;'} ${c}, deci ${rej ? 'respingem' : 'nu respingem'} rădăcina unitară.` + (rej !== rejN ? ` Cu valoarea critică a distribuției Normale, −1,645, concluzia ar fi, greșit, ${rejN ? 'respingerea' : 'păstrarea'} rădăcinii unitare.` : ''),
 df: (s, reg, mn, mx, nr, ng, La) => `${s}, ${reg === 'ct' ? 'constantă și tendință' : 'constantă'}: τ între ${mn} și ${mx} pentru L = 0–40; sub valoarea critică la ${nr} din ${ng} valori ale lui L. Cu constantă și tendință, AIC alege L = ${La}.`,
 ub: (s, ap, La, p, kp, ar, Lr, kr, P, R) => `${s}: prețul logaritmic, ADF = ${ap} (L = ${La}, ales prin AIC; p = ${p}), KPSS = ${kp}; randamentele, ADF = ${ar} (L = ${Lr}), KPSS = ${kr}. `
   + (P[0] && P[1] ? 'Prețul: ADF și KPSS resping amîndouă, un caz neconcludent. ' : !P[0] && P[1] ? 'Prețul: ADF nu respinge, KPSS respinge, deci I(1). ' : P[0] ? 'Prețul: ADF respinge, KPSS nu, deci I(0). ' : 'Prețul: niciun test nu respinge, neconcludent. ')
   + (R[0] && !R[1] ? 'Randamentele: ADF respinge, KPSS nu, deci I(0).' : 'Randamentele: neconcludent.'),
 ut: (l, nP, nR, Lv) => `${l === null ? 'Lățimea de bandă din curs, l = ⌈12(T/100)<sup>1/4</sup>⌉ (între 30 și 38, după serie)' : 'Lățimea de bandă l = ' + l}: KPSS respinge staționaritatea prețurilor pentru ${nP} din 8 serii și a randamentelor pentru ${nR}. ADF în tabel: ${Lv === 'aic' ? 'L ales prin AIC, ca în curs' : 'L = ' + Lv}.`,
 ut1: (s, n, l) => `${s}, prețul logaritmic, T = ${n}: e<sub>t</sub> sînt reziduurile regresiei pe constantă și tendință, S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>; l = ${l}.`,
 ut2: (eta, g0, lam) => `Σ S<sub>t</sub>²/T² = ${eta}; γ̂(0) = ${g0}; cu autocovarianțele ponderate Bartlett, λ̂² = ${lam}.`,
 ut3: (eta, lam, s, rej) => `KPSS = ${eta}/${lam} = ${s} ${rej ? '&gt;' : '&lt;'} 0,146: ${rej ? 'respingem' : 'nu respingem'} staționaritatea în jurul tendinței la 5%.`,
 va1: (s, q, r) => `${s}: ρ̂(1), …, ρ̂(${q - 1}) = ${r}.`,
 va2: (q, w) => `Ponderile 2(1 − k/${q}), k = 1, …, ${q - 1}: ${w}.`,
 va3: (q, a, b, d) => `VR(${q}) din ACF = 1 + Σ 2(1 − k/q)ρ̂(k) = ${a}; estimatorul Lo–MacKinlay pe aceleași randamente: ${b} (diferența ${d}).`,
 va: (s, q, a, b) => `${s}, q = ${q}: VR din ACF ${a}, Lo–MacKinlay ${b}. Barele albastre (negative) trag VR sub 1, cele roșii (pozitive) peste 1.`,
 lm1: (s, T, mu, va) => `${s}: T = ${T}, μ̂ = ${mu}%, σ̂<sub>a</sub>² = Σ(r<sub>t</sub> − μ̂)²/(T − 1) = ${va}.`,
 lm2: (q, m, vc, vr) => `q = ${q}: m = q(T − q + 1)(1 − q/T) = ${m}; σ̂<sub>c</sub>²(q) = ${vc}; VR(${q}) = σ̂<sub>c</sub>²/σ̂<sub>a</sub>² = ${vr}.`,
 lm3: (q, T, se, z) => `Abaterea standard homoscedastică √(2(2q − 1)(q − 1)/(3qT)) = ${se}; Z(${q}) = (VR − 1)/${se} = ${z}.`,
 lm4: (d, th, se, zs) => `δ̂(j) = ${d}; θ̂ = Σ[2(q − j)/q]²δ̂(j) = ${th}; √(θ̂/T) = ${se}; Z*(q) = (VR − 1)/${se} = ${zs}.`,
 lm5: (rs, rz) => `Abaterea standard robustă este de ${rs} ori cea homoscedastică, deci Z este de ${rz} ori Z*.`,
 lm: (q, nZ, list) => `q = ${q}: Z respinge la 5% pentru ${nZ} din 8 serii, Z* pentru ${list.length ? list.join(', ') : 'niciuna'}.`,
 vs: (a, ab, gz, gzs, iz, x) => `α = ${a}, β = 0,86 (α + β = ${ab})${x ? ', rulare suplimentară a simulării din curs' : ''}: cu randamente GARCH, Z(2) respinge ipoteza nulă adevărată în ${gz} din cele 1 000 de eșantioane, Z*(2) în ${gzs}; cu randamente i.i.d., Z(2) în ${iz}.`,
 vp: (y, items) => `Din ${y}: ` + items.map(([s, v, a, b, n]) => `${s} VR(40) = ${v} (între ${a} și ${b}), Z* respinge la ${n} din 39 de orizonturi`).join('; ') + '.',
 vt: (y, zs, n, N, e) => `Din ${y}: ${zs ? 'Z*' : 'Z'} depășește 1,96 în modul la ${n} din ${N} de teste; la 5%, dacă toate ipotezele nule ar fi adevărate și testele independente, ne-am aștepta în medie la ${e} respingeri false.`,
 cd1: (s, set, zs, cd) => `${s}, orizonturile ${set}: Z*(q) = ${zs}; CD = max |Z*| = ${cd}.`,
 cd2: (m, a, r, c) => `m = ${m}, α = ${a}: (1 − α)<sup>1/m</sup> = ${r}; c = Φ<sup>−1</sup>(1 − (1 − ${r})/2) = ${c}.`,
 cd3: (cd, ph, m, p, rej) => `p = 1 − (2Φ(${cd}) − 1)<sup>${m}</sup> = 1 − ${ph}<sup>${m}</sup>, p ${p}: ${rej ? 'respingem' : 'nu respingem'} mersul aleator pentru toate orizonturile împreună.`,
 av1: (s, rho, a2, T, k) => `${s}: ρ̂ = ${rho}; â<sub>2</sub> = 4ρ̂²/(1 − ρ̂)⁴ = ${a2}; k̂ = 1,3221 × (${a2} × ${T})<sup>1/5</sup> = ${k}.`,
 av2: vr => `VR(k̂) = 1 + 2 Σ<sub>i</sub> w(i/k̂)ρ̂(i), cu nucleul spectral pătratic w, pe toate cele T − 1 laguri: ${vr}.`,
 av3: (T, k, vr, st, p, pb) => `√(${T}/${k}) × (${vr} − 1)/√2 = ${st}; p asimptotic ${p}; p wild bootstrap din curs (500 de reeșantionări): ${pb}.`,
 cd: (c, a, m, list) => `Valoarea critică la ${a}%, m = ${m}: ${c}; CD respinge pentru ${list.length ? list.join(', ') : 'nicio serie'}. Testul automat și p-value-ul lui bootstrap nu depind de mulțimea de orizonturi.`,
 vm: (q, l, list, n, e) => `q = ${q}, interval robust de ${l}%: ${list.length ? 'intervalele care exclud valoarea 1: ' + list.join(', ') : 'niciun interval nu exclude valoarea 1'}. Din ${n} piețe, în medie ${e} ar exclude 1 doar din întîmplare.`,
 rv: (w, q, ov, items) => `Ferestre de ${zile(w)}, VR(${q}); ferestrele consecutive au în comun ${zile(ov)}. ` + items.map(([s, n, sh, p, ng, a, b]) => `${s}: ${n} ${de(n)}ferestre, ${sh} resping la 5% (${p} cu VR &gt; 1, ${ng} cu VR &lt; 1), VR între ${a} și ${b}`).join('; ') + '.',
 sb: (q, list) => `q = ${q}: Z*(${q}) respinge la 5% pentru ${list.length ? list.join('; ') : 'nicio subperioadă'}.`,
 ca: (mo, y, L, p1, p2, df) => `${mo ? 'Pe luni' : 'Pe zile ale săptămînii'}, din ${y}, erori standard Newey–West cu ${L} ${de(L)}laguri: testul Wald al egalității celor ${df + 1} medii (χ²(${df})) dă p = ${p1} pentru S&P 500 și p = ${p2} pentru BET.`,
 ca1: (s, y, nj, jm, no, om) => `${s}, din ${y}: ${zile(nj)} în ianuarie, media ${jm}%; ${zile(no)} în celelalte luni, media ${om}%.`,
 ca2: (b, L, se) => `Regresia r<sub>t</sub> = a + b·1{ianuarie}<sub>t</sub> + u<sub>t</sub>: b̂ = ${b} (diferența mediilor); eroarea standard Newey–West cu ${L} ${de(L)}laguri: ${se}.`,
 ca3: (b, se, t, p) => `t = ${b}/${se} = ${t}, p ${p}.`
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/,/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
window.CH7_TEXT = {
 h: { real: 'sd over q days (%)', ratio: 'ratio', from: 'from', price: 'log price (constant, trend)', rets: 'returns (constant)', period: 'period', avr: 'automatic VR', pboot: 'p (bootstrap)' },
 rw: (d, sp, lo, hi, n) => `On ${d}: the S&P 500 is at ${sp} (log price minus its first value); the four random walks are between ${lo} and ${hi}, and ${n === 1 ? 'one of them is' : n + ' of them are'} above the S&P 500. The simulated steps have mean 0.033% and standard deviation 1.13% a day, as the S&P 500 returns.`,
 sc: (k, rs, bs, rb, bb) => `Lag ${k}: S&P 500 ρ̂(${k}) = ${rs}, slope of the line ${bs}; BET ρ̂(${k}) = ${rb}, slope ${bb}. The 95% i.i.d. band, ±1.96/√T, is ±0.020 for the S&P 500 and ±0.023 for the BET.`,
 st1: (s, sd, n) => `${s}: daily standard deviation σ̂ = ${sd}% (${days(n)}, denominator n − 1).`,
 st2: (sd, r5, v5, r20, v20, r250, v250) => `Square-root rule: q = 5: ${sd} × ${r5} = ${v5}%; q = 20: ${sd} × ${r20} = ${v20}%; q = 250: ${sd} × ${r250} = ${v250}%.`,
 st3: (q, sd, rq, vq, real, ratio) => `Chosen horizon, q = ${q}: σ̂√q = ${sd} × ${rq} = ${vq}%; the standard deviation of the ${q}-day returns, from the overlapping sums, is ${real}%, that is ${ratio} of the rule's value.`,
 st: (s, v, r) => `${s}: the rule gives ${v}% over 250 days, the overlapping sums give ${r}%. Their ratio is about √VR(250): below 1 when returns are negatively autocorrelated, above 1 when they are positively autocorrelated.`,
 wn: (n, b, o, e) => `n = ${n}: band ±1.96/√n = ±${b}; ${o} of 30 sample autocorrelations lie outside the band (on average ${e} by chance alone).`,
 ap: (n, b, rows) => `n = ${n} (white-noise band: ±${b}). Largest deviation |sample ACF − theoretical ACF| over lags 1–20: ` + rows.map(([l, d]) => `${l}: ${d}`).join('; ') + '.',
 ar: (y, items) => `Samples from ${y} (or from the start of the series); of 20 lags: ` + items.map(([s, r, a, b]) => `${s} ρ̂(1) = ${r}, ${a} outside the i.i.d. band, ${b} outside the robust band`).join('; ') + '.',
 ar1: (s, T, k, s4, s2, d) => `${s}, T = ${T}, lag k = ${k}: Σ e<sub>t</sub>²e<sub>t−k</sub>² = ${s4}, Σ e<sub>t</sub>² = ${s2}; δ̂(${k}) = T × ${s4}/${s2}² = ${d}.`,
 ar2: (bi, d, T, br) => `i.i.d. band: ±1.96/√T = ±${bi}; robust band: ±1.96√(δ̂/T) = ±1.96√(${d}/${T}) = ±${br}.`,
 ar3: (k, r, oi, or) => `ρ̂(${k}) = ${r}: ${oi ? 'outside' : 'inside'} the i.i.d. band, ${or ? 'outside' : 'inside'} the robust band. δ̂ > 1 means that volatility clustering raises the variance of ρ̂(k).`,
 aa: (d, items) => `ACF of |r<sub>t</sub>|<sup>${d}</sup> at lags 1, 20 and 100: ` + items.map(([n, a, b, c]) => `${n} ${a}, ${b}, ${c}`).join('; ') + '. The returns themselves (dotted lines) stay close to 0.',
 lb1: (T, r, m) => `T = ${T}; the autocorrelations of the example: ${r}; we test m = ${m} ${m === 1 ? 'lag' : 'lags'}.`,
 lb2: (t, bp) => `The terms Tρ̂(k)²: ${t}; Q<sub>BP</sub>(m) = ${bp}.`,
 lb3: (T, T2, terms, lb) => `Q<sub>LB</sub>(m) = ${T} × ${T2} × (${terms}) = ${lb}.`,
 lb4: (m, c, lb, rej, p) => `χ²<sub>0.95</sub>(${m}) = ${c}; Q<sub>LB</sub> = ${lb} ${rej ? '&gt;' : '&lt;'} ${c}: we ${rej ? 'reject' : 'do not reject'} H<sub>0</sub> at 5% (p ${p}).`,
 lb: (T, bp, lb, rej, b) => `T = ${T}: Q<sub>BP</sub> = ${bp}, Q<sub>LB</sub> = ${lb}, H<sub>0</sub> ${rej ? 'rejected' : 'not rejected'} at 5%; the band of a single autocorrelation is ±${b}.`,
 at1: (s, T, m, r) => `${s}, T = ${T}: ρ̂(1), …, ρ̂(${m}) = ${r}.`,
 at2: (T, ss, bp) => `Σ ρ̂(k)² = ${ss}; Q<sub>BP</sub> = T Σ ρ̂(k)² = ${T} × ${ss} = ${bp}.`,
 at3: (lb, r) => `Q<sub>LB</sub> = T(T + 2) Σ ρ̂(k)²/(T − k) = ${lb}; with a large T, close to Q<sub>BP</sub> (ratio ${r}).`,
 at4: (d, rob) => `The robust statistic divides each Tρ̂(k)² by δ̂(k); the first values δ̂(k): ${d}; Q̃ = Σ Tρ̂(k)²/δ̂(k) = ${rob}.`,
 at5: (m, c, lb, plb, rob, prob) => `χ²<sub>0.95</sub>(${m}) = ${c}: Q<sub>LB</sub> = ${lb} (p ${plb}), Q̃ = ${rob} (p ${prob}).`,
 at: (m, c, n, list) => `m = ${m}, critical value ${c}: Q<sub>LB</sub> rejects at 5% for ${n} of 8 series, the robust Q̃ for ${list.length ? list.join(', ') : 'none'}.`,
 ru1: (s, n1, n2, n) => `${s}: n<sub>1</sub> = ${n1} positive returns, n<sub>2</sub> = ${n2} negative, n = ${n} (days with a zero return are dropped).`,
 ru2: (n1, n2, n, m) => `E[R] = 2 × ${n1} × ${n2}/${n} + 1 = ${m}.`,
 ru3: (v, sd) => `Var(R) = 2n<sub>1</sub>n<sub>2</sub>(2n<sub>1</sub>n<sub>2</sub> − n)/(n²(n − 1)) = ${v}; sd(R) = ${sd}.`,
 ru4: (R, m, sd, z, p, zv) => `R = ${R} observed runs; z = (${R} − ${m})/${sd} = ${z}, p ${p}: ${zv < 0 ? 'too few runs, the signs persist' : 'too many runs, the signs alternate'}.`,
 ru: (y, list) => `From ${y}: the runs test rejects at 5% for ${list.length ? list.join(', ') : 'no series'}.`,
 sp: (T, rej, r2, b40, rd, x) => `T = ${T}${x ? ' (extra run of the lecture code)' : ''}: in levels, |t| > 1.96 in ${rej} of the regressions, median R² ${r2}, |t| > 40 in ${b40}; in increments, |t| > 1.96 in ${rd}.`,
 df1: (s, reg, L, n) => `${s}, log price: Δp<sub>t</sub> = c ${reg === 'ct' ? '+ bt ' : ''}+ γp<sub>t−1</sub>${L ? ` + Σ<sub>j=1</sub><sup>${L}</sup> δ<sub>j</sub>Δp<sub>t−j</sub>` : ''} + ε<sub>t</sub>, estimated by OLS on ${n} observations.`,
 df2: (g, se) => `γ̂ = ${g}, SE(γ̂) = ${se}.`,
 df3: (g, se, t) => `τ = γ̂/SE(γ̂) = ${g}/${se} = ${t}.`,
 df4: (t, c, rej, rejN) => `The 5% critical value (MacKinnon) is ${c}: τ = ${t} ${rej ? '&lt;' : '&gt;'} ${c}, so we ${rej ? 'reject' : 'do not reject'} the unit root.` + (rej !== rejN ? ` With the critical value of the Normal distribution, −1.645, the conclusion would wrongly be to ${rejN ? 'reject' : 'keep'} the unit root.` : ''),
 df: (s, reg, mn, mx, nr, ng, La) => `${s}, ${reg === 'ct' ? 'constant and trend' : 'constant'}: τ between ${mn} and ${mx} for L = 0–40; below the critical value for ${nr} of ${ng} values of L. With constant and trend, AIC chooses L = ${La}.`,
 ub: (s, ap, La, p, kp, ar, Lr, kr, P, R) => `${s}: log price, ADF = ${ap} (L = ${La}, chosen by AIC; p = ${p}), KPSS = ${kp}; returns, ADF = ${ar} (L = ${Lr}), KPSS = ${kr}. `
   + (P[0] && P[1] ? 'Price: ADF and KPSS both reject, an inconclusive case. ' : !P[0] && P[1] ? 'Price: ADF does not reject, KPSS rejects, so I(1). ' : P[0] ? 'Price: ADF rejects, KPSS does not, so I(0). ' : 'Price: neither test rejects, inconclusive. ')
   + (R[0] && !R[1] ? 'Returns: ADF rejects, KPSS does not, so I(0).' : 'Returns: inconclusive.'),
 ut: (l, nP, nR, Lv) => `${l === null ? 'The bandwidth of the lecture, l = ⌈12(T/100)<sup>1/4</sup>⌉ (between 30 and 38, by series)' : 'Bandwidth l = ' + l}: KPSS rejects stationarity of the prices for ${nP} of 8 series and of the returns for ${nR}. ADF in the table: ${Lv === 'aic' ? 'L chosen by AIC, as in the lecture' : 'L = ' + Lv}.`,
 ut1: (s, n, l) => `${s}, log price, T = ${n}: e<sub>t</sub> are the residuals of the regression on a constant and a trend, S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>; l = ${l}.`,
 ut2: (eta, g0, lam) => `Σ S<sub>t</sub>²/T² = ${eta}; γ̂(0) = ${g0}; with the Bartlett-weighted autocovariances, λ̂² = ${lam}.`,
 ut3: (eta, lam, s, rej) => `KPSS = ${eta}/${lam} = ${s} ${rej ? '&gt;' : '&lt;'} 0.146: we ${rej ? 'reject' : 'do not reject'} trend stationarity at 5%.`,
 va1: (s, q, r) => `${s}: ρ̂(1), …, ρ̂(${q - 1}) = ${r}.`,
 va2: (q, w) => `The weights 2(1 − k/${q}), k = 1, …, ${q - 1}: ${w}.`,
 va3: (q, a, b, d) => `VR(${q}) from the ACF = 1 + Σ 2(1 − k/q)ρ̂(k) = ${a}; the Lo–MacKinlay estimator on the same returns: ${b} (difference ${d}).`,
 va: (s, q, a, b) => `${s}, q = ${q}: VR from the ACF ${a}, Lo–MacKinlay ${b}. The blue (negative) bars pull VR below 1, the red (positive) ones above 1.`,
 lm1: (s, T, mu, va) => `${s}: T = ${T}, μ̂ = ${mu}%, σ̂<sub>a</sub>² = Σ(r<sub>t</sub> − μ̂)²/(T − 1) = ${va}.`,
 lm2: (q, m, vc, vr) => `q = ${q}: m = q(T − q + 1)(1 − q/T) = ${m}; σ̂<sub>c</sub>²(q) = ${vc}; VR(${q}) = σ̂<sub>c</sub>²/σ̂<sub>a</sub>² = ${vr}.`,
 lm3: (q, T, se, z) => `Homoskedastic standard error √(2(2q − 1)(q − 1)/(3qT)) = ${se}; Z(${q}) = (VR − 1)/${se} = ${z}.`,
 lm4: (d, th, se, zs) => `δ̂(j) = ${d}; θ̂ = Σ[2(q − j)/q]²δ̂(j) = ${th}; √(θ̂/T) = ${se}; Z*(q) = (VR − 1)/${se} = ${zs}.`,
 lm5: (rs, rz) => `The robust standard error is ${rs} times the homoskedastic one, so Z is ${rz} times Z*.`,
 lm: (q, nZ, list) => `q = ${q}: Z rejects at 5% for ${nZ} of 8 series, Z* for ${list.length ? list.join(', ') : 'none'}.`,
 vs: (a, ab, gz, gzs, iz, x) => `α = ${a}, β = 0.86 (α + β = ${ab})${x ? ', extra run of the lecture simulation' : ''}: with GARCH returns, Z(2) rejects the true null in ${gz} of the 1,000 samples, Z*(2) in ${gzs}; with i.i.d. returns, Z(2) in ${iz}.`,
 vp: (y, items) => `From ${y}: ` + items.map(([s, v, a, b, n]) => `${s} VR(40) = ${v} (between ${a} and ${b}), Z* rejects at ${n} of 39 horizons`).join('; ') + '.',
 vt: (y, zs, n, N, e) => `From ${y}: ${zs ? 'Z*' : 'Z'} exceeds 1.96 in absolute value in ${n} of ${N} tests; at 5%, if all null hypotheses were true and the tests independent, we would expect ${e} false rejections on average.`,
 cd1: (s, set, zs, cd) => `${s}, horizons ${set}: Z*(q) = ${zs}; CD = max |Z*| = ${cd}.`,
 cd2: (m, a, r, c) => `m = ${m}, α = ${a}: (1 − α)<sup>1/m</sup> = ${r}; c = Φ<sup>−1</sup>(1 − (1 − ${r})/2) = ${c}.`,
 cd3: (cd, ph, m, p, rej) => `p = 1 − (2Φ(${cd}) − 1)<sup>${m}</sup> = 1 − ${ph}<sup>${m}</sup>, p ${p}: we ${rej ? 'reject' : 'do not reject'} the random walk for all horizons jointly.`,
 av1: (s, rho, a2, T, k) => `${s}: ρ̂ = ${rho}; â<sub>2</sub> = 4ρ̂²/(1 − ρ̂)⁴ = ${a2}; k̂ = 1.3221 × (${a2} × ${T})<sup>1/5</sup> = ${k}.`,
 av2: vr => `VR(k̂) = 1 + 2 Σ<sub>i</sub> w(i/k̂)ρ̂(i), with the quadratic spectral kernel w, over all T − 1 lags: ${vr}.`,
 av3: (T, k, vr, st, p, pb) => `√(${T}/${k}) × (${vr} − 1)/√2 = ${st}; asymptotic p ${p}; wild-bootstrap p of the lecture (500 resamples): ${pb}.`,
 cd: (c, a, m, list) => `Critical value at ${a}%, m = ${m}: ${c}; CD rejects for ${list.length ? list.join(', ') : 'no series'}. The automatic test and its bootstrap p-value do not depend on the set of horizons.`,
 vm: (q, l, list, n, e) => `q = ${q}, robust ${l}% interval: ${list.length ? 'the value 1 is excluded for ' + list.join(', ') : 'no interval excludes the value 1'}. Of ${n} markets, on average ${e} would exclude 1 by chance alone.`,
 rv: (w, q, ov, items) => `Windows of ${days(w)}, VR(${q}); consecutive windows share ${days(ov)}. ` + items.map(([s, n, sh, p, ng, a, b]) => `${s}: ${n} windows, ${sh} reject at 5% (${p} with VR &gt; 1, ${ng} with VR &lt; 1), VR between ${a} and ${b}`).join('; ') + '.',
 sb: (q, list) => `q = ${q}: Z*(${q}) rejects at 5% for ${list.length ? list.join('; ') : 'no sub-period'}.`,
 ca: (mo, y, L, p1, p2, df) => `${mo ? 'By month' : 'By day of the week'}, from ${y}, Newey–West standard errors with ${L} lags: the Wald test of equal means (${df + 1} means, χ²(${df})) gives p = ${p1} for the S&P 500 and p = ${p2} for the BET.`,
 ca1: (s, y, nj, jm, no, om) => `${s}, from ${y}: ${days(nj)} in January, mean ${jm}%; ${days(no)} in the other months, mean ${om}%.`,
 ca2: (b, L, se) => `The regression r<sub>t</sub> = a + b·1{January}<sub>t</sub> + u<sub>t</sub>: b̂ = ${b} (the difference of the means); Newey–West standard error with ${L} lags: ${se}.`,
 ca3: (b, se, t, p) => `t = ${b}/${se} = ${t}, p ${p}.`
};
})();
""",
}

QA = {
 'ro': {
  'rw-paths': ('Patru mersuri aleatoare cu pași Normali i.i.d. (media și abaterea standard ale S&P 500) și S&P 500 însuși. Butonul ▶ desenează traiectoriile an cu an, din 1991 pînă în 2026. La sfîrșitul lui 2008, cîte dintre cele patru mersuri aleatoare sînt deasupra S&P 500?',
               'Trei. La sfîrșitul lui 2008 S&P 500 este la 0,91 (logaritmul prețului minus valoarea de start), iar mersurile simulate la 1,12, 1,24, 2,69 și 0,65. În septembrie 2026 doar unul (4,50) este deasupra S&P 500 (3,06). Scăderile din 2000–2002 și 2008 au mărimea episoadelor pe care le produce și zgomotul pur: o singură traiectorie nu separă S&P 500 de un mers aleator.'),
  'scatter-lag': ('Corelația dintre randamentul de azi și cel de ieri este −0,079 pentru S&P 500 și 0,156 pentru BET. Butonul ▶ mută lagul de la 1 la 10. Cît devine corelația BET la lagul 2, între randamentul de azi și cel de acum două zile?',
                  '0,033, de aproape cinci ori mai mică; la lagul 3 este −0,008, la lagul 5 0,031, la lagul 10 0,004. Pentru S&P 500 corelația la lagul 2 este −0,004. Dependența randamentelor BET stă aproape toată la lagul 1, cum produce tranzacționarea redusă: o știre intră în indice pe parcursul a două zile.'),
  'sqrt-time': ('Regula rădăcinii pătrate a timpului dă pentru S&P 500 o abatere standard pe 250 de zile de 1,1337% × √250 = 17,92%. Abaterea standard a randamentelor pe 250 de zile, calculată direct din sumele suprapuse, este mai mare sau mai mică?',
                'Mai mică: 15,66%, adică 0,874 din valoarea regulii. Randamentele zilnice ale S&P 500 sînt ușor negativ autocorelate (ρ̂(1) = −0,079, VR(250) = 0,785), deci mișcările zilnice se anulează parțial în timp. BET, cu ρ̂(1) = 0,156, arată efectul invers: 35,91% față de 23,38% (de 1,54 ori). Regula presupune randamente necorelate; scalarea VaR sau a volatilității cu √q moștenește această ipoteză.'),
  'white-noise': ('Pentru n = 1 000 de extrageri, banda de 95% este ±0,062 și 0 din 30 de autocorelații de selecție ies din ea. Butonul ▶ folosește doar primele n extrageri, de la 100 la 1 000. Pentru n = 100, cu banda ±0,196, cîte autocorelații ies din bandă?',
                  'Una: cea mai mare în modul este 0,223. Cu 30 de laguri ne așteptăm la 1,5 depășiri doar din întîmplare, la orice n: banda se lărgește ca 1/√n, iar autocorelațiile de selecție ale zgomotului alb sînt mai mari în același ritm. La n = 400 iese tot una, la celelalte valori ale lui n niciuna.'),
  'acf-processes': ('Punctele roșii sînt ACF-ul de selecție al unei traiectorii simulate de 1 000 de valori. Butonul ▶ folosește doar primele n valori, de la 50 la 1 000. Pentru AR(1) cu α = 0,9, cît de departe ajunge ACF-ul de selecție de cel teoretic la n = 50?',
                    'Pînă la 0,40 (cea mai mare abatere pe lagurile 1–20), față de 0,08 la n = 1 000. Banda zgomotului alb pentru n = 50 este ±0,28 (1/√50 = 0,14), dar abaterea este mai mare, pentru că varianța lui ρ̂(k) a unui proces persistent depășește 1/T. Pentru MA(2), abaterea maximă este 0,19 la n = 50 și 0,06 la n = 1 000.'),
  'acf-returns': ('Din 1990, S&P 500 are 9 autocorelații (din 20) în afara benzii i.i.d. și 3 în afara benzii robuste. Butonul ▶ mută începutul eșantionului spre 2022. Ce se întîmplă cu ρ̂(1) al S&P 500 pe eșantionul care începe în 2016?',
                  'Devine mai negativ: −0,142, față de −0,079 din 1990, cu 13 laguri în afara benzii i.i.d. și 7 în afara benzii robuste. Pe eșantionul care începe în 2022, ρ̂(1) este doar −0,015. Pe ferestre mobile (secțiunea „Eficiența pe piețe și în timp”) revenirile S&P 500 sînt cele mai puternice în crize: eșantionul din 2016 include 2020, cel din 2022 nu.'),
  'acf-abs': ('ACF-ul lui |r<sub>t</sub>| pentru S&P 500 este 0,27 la lagul 1, 0,22 la lagul 20 și 0,09 la lagul 100. Butonul ▶ mărește puterea d din |r<sub>t</sub>|<sup>d</sup> pînă la 2 (randamentele la pătrat). Ce se întîmplă cu autocorelația la lagul 100 cînd d trece de la 1 la 2?',
              'Scade: 0,039 pentru d = 2, față de 0,092 pentru d = 1; la lagul 1 crește puțin (0,311 față de 0,275). Puterile mari dau pondere cîtorva zile extreme, iar autocorelația lor se stinge mai repede. Pentru orice d mărimea randamentelor rămîne corelată, deci RW1 și RW2 sînt respinse.'),
  'lb-example': ('Cu T = 1 000 și autocorelațiile 0,06, −0,04 și 0,03, Q<sub>LB</sub>(3) = 6,12 < 7,81: nu respingem. Butonul ▶ crește T pînă la 5 000, cu aceleași autocorelații. De la ce T respinge testul la 5%?',
                 'De la T = 1 500 (în pașii de 500): Q<sub>LB</sub>(3) = 9,17 > 7,81, p = 0,027. Statistica crește aproape proporțional cu T (30,52 la T = 5 000); pragul exact este în jur de T = 1 280. Autocorelațiile sînt aceleași; cu mai multe date ele sînt estimate mai precis și se separă de zero. În eșantioanele lungi, autocorelații mici devin semnificative, fără să fie mai mari.'),
  'ac-tests': ('Cu m = 10 laguri, Q<sub>LB</sub> clasic respinge pentru toate cele opt serii, Q̃ robust doar pentru S&P 500 și BET. Butonul ▶ mărește m pînă la 30. Cu m = 30, pentru ce serii respinge Q̃ robust?',
               'Doar pentru BET (Q̃(30) = 67,9 > 43,77). S&P 500 nu mai respinge: Q̃(30) = 39,7, pentru că autocorelația lui de la lagul 1 se diluează printre 29 de laguri aproape nule; slide-ul spune același lucru: un m mai mare diluează un efect de scurtă durată. Q<sub>LB</sub> clasic respinge la m = 30 pentru cinci serii, cu o distribuție χ² care nu ține seama de volatility clustering.'),
  'runs': ('Pe tot eșantionul, BET are 3 278 de secvențe față de 3 606 așteptate (z = −7,75), iar S&P 500 4 738 față de 4 596 (z = 2,97). Butonul ▶ mută începutul eșantionului spre 2022. Pe eșantionul care începe în 2022, rămîne vreuna dintre aceste două respingeri?',
           'Doar cea a BET, mult mai slabă: z = −2,03, față de 0,06 pentru S&P 500. Semnele BET persistă și în anii recenți, dar mult mai puțin decît pe tot eșantionul; la S&P 500 alternanța semnelor dispare. Din 2022, BRD are z = −3,79.'),
  'spurious': ('Cu T = 500, regresia unui mers aleator pe altul, independent, dă |t| > 1,96 în 90% din cele 2 000 de repetări. Butonul ▶ mărește T la 1 000 și 2 000 (T ≠ 500: rulări suplimentare ale codului din curs). Se apropie proporția respingerilor de 5% cînd T crește?',
               'Nu, crește: 92,7% la T = 1 000 și 94,5% la T = 2 000 (77,3% la T = 100); |t| > 40 apare în 21,6% din regresii la T = 2 000. În regresia în niveluri, statistica t crește odată cu T, deci mai multe date fac relația falsă mai convingătoare. R² median rămîne între 0,16 și 0,18. Pe diferențe, proporția rămîne între 5,1% și 6,0% pentru orice T.'),
  'adf-example': ('Pentru prețul logaritmic BET, regresia Dickey–Fuller cu constantă și fără laguri dă τ = −0,09. Butonul ▶ trece prin cele opt serii; curba arată τ pentru L = 0–40 laguri ale diferențelor. Respinge testul cu constantă rădăcina unitară pentru vreo serie, la vreun L?',
                  'Nu. Cel mai mic τ cu constantă este −1,41 (Bitcoin, L = 20), departe de −2,86; pentru TLV, SNP, BRD și TGN τ este chiar pozitiv. Prețurile cresc în timp, deci specificația potrivită are constantă și tendință: cu ea, TLV coboară sub −3,41 la 12 din cele 16 valori ale lui L (τ = −3,46 la L = 20, ordinul ales de AIC).'),
  'unit-root-bet': ('Pentru BET, ADF pe prețul logaritmic este −1,73 (p = 0,736) și KPSS 2,30: prețul este I(1). Butonul ▶ trece prin cele opt serii. Pentru care serie testele pe prețul logaritmic nu dau concluzia I(1)?',
                    'TLV: ADF = −3,46 < −3,41 (p = 0,044) respinge rădăcina unitară, iar KPSS = 0,57 > 0,146 respinge staționaritatea în jurul tendinței. Testele se contrazic: un caz neconcludent, nu o dovadă de revenire la tendință. Pentru celelalte șapte serii ADF nu respinge și KPSS respinge. Randamentele sînt I(0) pentru toate cele opt.'),
  'ur-table': ('Cu lățimea de bandă din curs (l = 38 pentru S&P 500), KPSS respinge staționaritatea prețurilor pentru toate cele opt serii; KPSS pe prețul S&P 500 este 2,23. Butonul ▶ schimbă lățimea de bandă l a varianței pe termen lung, începînd cu l = 0. Cît este KPSS pentru prețul S&P 500 la l = 0?',
               '86,0. Un preț integrat are autocovarianțe mari și pozitive; fără ele, λ̂² este doar varianța reziduurilor și subestimează de zeci de ori varianța pe termen lung. Statistica scade cînd l crește, dar la l = 80 este încă 1,09 > 0,146. Pe randamente, KPSS rămîne sub 0,463 pentru orice l: concluzia I(1) pentru prețuri și I(0) pentru randamente nu depinde de lățimea de bandă.'),
  'vr-acf': ('Din ACF-ul S&P 500, VR(5) = 1 + 1,6ρ̂(1) + 1,2ρ̂(2) + 0,8ρ̂(3) + 0,4ρ̂(4) = 0,855, iar estimatorul Lo–MacKinlay dă 0,856. Butonul ▶ mărește q pînă la 20. Cît de mult diferă cele două valori la q = 20?',
             'Foarte puțin: 0,768 din ACF și 0,767 cu estimatorul Lo–MacKinlay, o diferență de 0,0005. Pentru BET, la q = 20: 1,514 și 1,517. Cele două forme diferă doar prin corecțiile de eșantion mic, care contează puțin cu T = 9 240 și q = 20.'),
  'lm-tests': ('Pentru S&P 500 la q = 2, Z(2) = −7,54 și Z*(2) = −3,37: statistica homoscedastică este de 2,24 ori mai mare în modul. Butonul ▶ mărește q pînă la 40. La q = 40, mai respinge Z* mersul aleator pentru S&P 500?',
               'Nu: Z*(40) = −1,81, deși Z(40) = −3,70 respinge încă. La q = 40, Z* respinge doar pentru BET. Raportul Z/Z* rămîne între 2,0 și 2,3 la toate orizonturile: volatility clustering dublează abaterea standard a lui VR(q) estimat, iar Z(q) nu ține seama de ea.'),
  'vr-size': ('Sub un martingal GARCH(1,1) cu α = 0,12 și β = 0,86, Z(2) respinge ipoteza nulă adevărată în 17,1% din cazuri, Z*(2) în 4,7%. Butonul ▶ schimbă reacția α de la 0 la 0,12, cu β = 0,86 fix (α ≠ 0,12: rulări suplimentare ale simulării din curs). La α = 0,09, cît de des respinge Z(2)?',
              'În 9,0% din cazuri; la α = 0,06 în 6,2%, la α = 0 în 4,8%. Distorsiunea crește repede cînd α + β se apropie de 1 (0,95 la α = 0,09, 0,98 la α = 0,12), pentru că volatilitatea devine foarte persistentă. Z*(2) rămîne între 4,3% și 4,9% pentru toate valorile lui α.'),
  'vr-profile': ('Din 1990, VR(40) este 0,724 pentru S&P 500 și 1,743 pentru BET, iar Z* respinge pentru BET la toate cele 39 de orizonturi. Butonul ▶ mută începutul eșantionului spre 2022. Pe eșantionul care începe în 2010, mai respinge Z* momentumul BET la vreun orizont q = 2, …, 40?',
                 'Nu: VR(40) al BET scade la 1,193 și niciun Z*(q) nu depășește 1,96. Momentumul BET vine din primii ani (1997–2006, ρ̂(1) = 0,270). S&P 500 respinge din 2010 la un singur orizont din 39, cu VR(40) = 0,663, sub valoarea din 1990: revenirea la medie rămîne, dar banda robustă se lărgește odată cu eșantionul mai scurt.'),
  'vr-table': ('Pe tot eșantionul, 10 din cele 32 de statistici Z* din tabel depășesc 1,96 în modul. Butonul ▶ mută începutul eșantionului spre 2022. Pe eșantionul care începe în 2010, cîte rămîn?',
               'Trei: S&P 500 la q = 2 și Transgaz la q = 2 și 5 (pentru Transgaz eșantionul începe oricum în 2010). BET nu mai respinge la niciun orizont. Cu statistica homoscedastică Z ar fi 13, pentru că Z ignoră volatility clustering. La 32 de teste independente la 5% ne-am aștepta în medie la 1,6 respingeri false.'),
  'cd-table': ('Cu orizonturile 2, 5, 10, 20 (m = 4), valoarea critică Chow–Denning la 5% este 2,49 și testul respinge pentru S&P 500, BET și Transgaz. Butonul ▶ schimbă mulțimea de orizonturi. Cu toate orizonturile 2, …, 20 (m = 19), care serii mai resping?',
               'S&P 500 și BET. Valoarea critică crește la 3,00, iar Transgaz, cu CD = 2,77, nu o mai depășește. Mai multe orizonturi înseamnă mai multe ocazii pentru o respingere falsă, deci un prag mai mare; CD crește doar dacă orizonturile adăugate au un |Z*| mai mare.'),
  'vr-markets': ('La q = 5, doar intervalele robuste de 95% ale PX (Z* = 2,08) și BET (Z* = 2,03) exclud valoarea 1. Butonul ▶ mărește orizontul q pînă la 20. La q = 20, ce piață exclude valoarea 1?',
                 'Doar Ethereum (VR(20) = 1,29, Z* = 2,45); PX și BET revin în interval. La q = 2 este S&P 500 (VR(2) = 0,86, Z* = −2,33). Fiecare orizont arată altă piață: cu 16 piețe și multe orizonturi, unele intervale exclud 1 din întîmplare, iar testul Chow–Denning pe patru orizonturi nu respinge pentru nicio piață.'),
  'rolling-vr': ('Pe ferestre de 500 de zile, 29% dintre ferestrele BET și 4% dintre ferestrele S&P 500 resping la 5%. Butonul ▶ mărește fereastra la 750, 1 000 și 1 500 de zile. Ce se întîmplă cu ponderea ferestrelor S&P 500 care resping, la 1 500 de zile?',
                 'Crește la 20%, toate cu VR < 1. O fereastră lungă estimează mai precis aceeași autocorelație negativă mică, deci o separă mai ușor de zero; la 250 de zile ponderea este 1,6%. Ferestrele consecutive au în comun 1 479 din 1 500 de zile, deci ponderea descrie datele, nu este un test.'),
  'sub-table': ('La q = 5, S&P 500 respinge în 2002–2013 (Z*(5) = −2,53). Butonul ▶ mărește q la 10 și 20. La q = 20, mai respinge vreo subperioadă a S&P 500?',
                'Niciuna: Z*(20) = −1,33, −1,54 și −0,91, deși VR(20) = 0,72 în 2002–2013, mai departe de 1 decît VR(5) = 0,80. Varianța lui VR(q) crește cu q, iar cu circa 3 000 de zile pe subperioadă intervalul robust devine larg. Doar BET 1997–2006 respinge la q = 20 (Z*(20) = 5,14).'),
  'calendar': ('Testul Wald al egalității mediilor pe zile ale săptămînii dă p = 0,69 pentru S&P 500 și 0,24 pentru BET (erori standard Newey–West cu 5 laguri). Butonul ▶ mută începutul eșantionului spre 2022. Coboară p-value-ul BET sub 0,05 pentru vreun început?',
               'Nu: cel mai mic este 0,088, pentru eșantionul care începe în 2016; S&P 500 rămîne peste 0,62. Încercarea a 17 începuturi diferite este ea însăși o căutare repetată în date: chiar un p sub 0,05 la unul dintre ele nu ar fi o dovadă a unui efect de zi a săptămînii.'),
 },
 'en': {
  'rw-paths': ('Four random walks with i.i.d. Normal steps (the mean and standard deviation of the S&P 500) and the S&P 500 itself. The ▶ button draws the paths year by year, from 1991 to 2026. At the end of 2008, how many of the four random walks are above the S&P 500?',
               'Three. At the end of 2008 the S&P 500 is at 0.91 (log price minus its first value), the simulated walks at 1.12, 1.24, 2.69 and 0.65. In September 2026 only one (4.50) is above the S&P 500 (3.06). The falls of 2000–2002 and 2008 have the size of episodes that pure noise also produces: a single path does not separate the S&P 500 from a random walk.'),
  'scatter-lag': ('The correlation between today\'s and yesterday\'s return is −0.079 for the S&P 500 and 0.156 for the BET. The ▶ button moves the lag from 1 to 10. What is the BET correlation at lag 2, between today\'s return and that of two days ago?',
                  '0.033, almost five times smaller; at lag 3 it is −0.008, at lag 5 0.031, at lag 10 0.004. For the S&P 500 the lag-2 correlation is −0.004. Almost all the dependence of the BET returns sits at lag 1, as thin trading produces: news enters the index over two days.'),
  'sqrt-time': ('The square-root-of-time rule gives the S&P 500 a 250-day standard deviation of 1.1337% × √250 = 17.92%. Is the standard deviation of 250-day returns, computed directly from the overlapping sums, larger or smaller?',
                'Smaller: 15.66%, that is 0.874 of the rule\'s value. The daily S&P 500 returns are slightly negatively autocorrelated (ρ̂(1) = −0.079, VR(250) = 0.785), so daily moves partly cancel over time. The BET, with ρ̂(1) = 0.156, shows the opposite: 35.91% against 23.38% (1.54 times). The rule assumes uncorrelated returns; scaling VaR or volatility by √q inherits this assumption.'),
  'white-noise': ('For n = 1,000 draws the 95% band is ±0.062 and 0 of 30 sample autocorrelations lie outside it. The ▶ button uses only the first n draws, from 100 to 1,000. For n = 100, with the band ±0.196, how many autocorrelations lie outside the band?',
                  'One: the largest in absolute value is 0.223. With 30 lags we expect 1.5 exceedances by chance alone, at any n: the band widens as 1/√n, and the sample autocorrelations of white noise grow at the same rate. At n = 400 one lies outside as well, at the other values of n none.'),
  'acf-processes': ('The red points are the sample ACF of one simulated path of 1,000 values. The ▶ button uses only the first n values, from 50 to 1,000. For the AR(1) with α = 0.9, how far does the sample ACF get from the theoretical one at n = 50?',
                    'Up to 0.40 (the largest deviation over lags 1–20), against 0.08 at n = 1,000. The white-noise band for n = 50 is ±0.28 (1/√50 = 0.14), but the deviation is larger, because the variance of ρ̂(k) of a persistent process exceeds 1/T. For the MA(2), the largest deviation is 0.19 at n = 50 and 0.06 at n = 1,000.'),
  'acf-returns': ('From 1990, the S&P 500 has 9 autocorrelations (of 20) outside the i.i.d. band and 3 outside the robust band. The ▶ button moves the start of the sample towards 2022. What happens to the S&P 500 ρ̂(1) on the sample that starts in 2016?',
                  'It becomes more negative: −0.142, against −0.079 from 1990, with 13 lags outside the i.i.d. band and 7 outside the robust band. On the sample that starts in 2022, ρ̂(1) is only −0.015. On rolling windows (section "Efficiency across Markets and over Time") the S&P 500 reversals are strongest in crises: the sample from 2016 includes 2020, the one from 2022 does not.'),
  'acf-abs': ('The ACF of |r<sub>t</sub>| for the S&P 500 is 0.27 at lag 1, 0.22 at lag 20 and 0.09 at lag 100. The ▶ button raises the power d in |r<sub>t</sub>|<sup>d</sup> up to 2 (squared returns). What happens to the lag-100 autocorrelation when d goes from 1 to 2?',
              'It falls: 0.039 for d = 2, against 0.092 for d = 1; at lag 1 it rises slightly (0.311 against 0.275). High powers give weight to a few extreme days, and their autocorrelation dies out faster. For every d the size of the returns stays correlated, so RW1 and RW2 are rejected.'),
  'lb-example': ('With T = 1,000 and the autocorrelations 0.06, −0.04 and 0.03, Q<sub>LB</sub>(3) = 6.12 < 7.81: we do not reject. The ▶ button raises T up to 5,000, with the same autocorrelations. From which T does the test reject at 5%?',
                 'From T = 1,500 (in steps of 500): Q<sub>LB</sub>(3) = 9.17 > 7.81, p = 0.027. The statistic grows almost in proportion to T (30.52 at T = 5,000); the exact threshold is around T = 1,280. The autocorrelations are the same; with more data they are estimated more precisely and separate from zero. In long samples, small autocorrelations become significant without being larger.'),
  'ac-tests': ('With m = 10 lags, the classical Q<sub>LB</sub> rejects for all eight series, the robust Q̃ only for the S&P 500 and the BET. The ▶ button raises m up to 30. With m = 30, for which series does the robust Q̃ reject?',
               'Only for the BET (Q̃(30) = 67.9 > 43.77). The S&P 500 no longer rejects: Q̃(30) = 39.7, because its lag-1 autocorrelation is diluted among 29 nearly zero lags; the slide says the same: a larger m dilutes a short-lived effect. The classical Q<sub>LB</sub> rejects at m = 30 for five series, with a χ² distribution that ignores volatility clustering.'),
  'runs': ('Over the whole sample, the BET has 3,278 runs against 3,606 expected (z = −7.75), the S&P 500 4,738 against 4,596 (z = 2.97). The ▶ button moves the start of the sample towards 2022. On the sample that starts in 2022, does either of these two rejections remain?',
           'Only the BET one, much weaker: z = −2.03, against 0.06 for the S&P 500. The BET signs persist in recent years too, but much less than over the whole sample; for the S&P 500 the alternation of signs disappears. From 2022, BRD has z = −3.79.'),
  'spurious': ('With T = 500, regressing one random walk on another, independent one gives |t| > 1.96 in 90% of the 2,000 replications. The ▶ button raises T to 1,000 and 2,000 (T ≠ 500: extra runs of the lecture code). Does the share of rejections approach 5% as T grows?',
               'No, it rises: 92.7% at T = 1,000 and 94.5% at T = 2,000 (77.3% at T = 100); |t| > 40 occurs in 21.6% of the regressions at T = 2,000. In the regression in levels the t-statistic grows with T, so more data make the spurious relation more convincing. The median R² stays between 0.16 and 0.18. In increments, the share stays between 5.1% and 6.0% for every T.'),
  'adf-example': ('For the BET log price, the Dickey–Fuller regression with a constant and no lags gives τ = −0.09. The ▶ button runs through the eight series; the curve shows τ for L = 0–40 lags of the differences. Does the test with a constant reject the unit root for any series, at any L?',
                  'No. The lowest τ with a constant is −1.41 (Bitcoin, L = 20), far from −2.86; for TLV, SNP, BRD and TGN τ is even positive. Prices grow over time, so the right specification has a constant and a trend: with it, TLV falls below −3.41 at 12 of the 16 values of L (τ = −3.46 at L = 20, the order chosen by AIC).'),
  'unit-root-bet': ('For the BET, ADF on the log price is −1.73 (p = 0.736) and KPSS 2.30: the price is I(1). The ▶ button runs through the eight series. For which series do the tests on the log price not give the conclusion I(1)?',
                    'TLV: ADF = −3.46 < −3.41 (p = 0.044) rejects the unit root, and KPSS = 0.57 > 0.146 rejects trend stationarity. The tests contradict each other: an inconclusive case, not evidence of reversion to a trend. For the other seven series ADF does not reject and KPSS rejects. The returns are I(0) for all eight.'),
  'ur-table': ('With the bandwidth of the lecture (l = 38 for the S&P 500), KPSS rejects stationarity of the prices for all eight series; KPSS on the S&P 500 price is 2.23. The ▶ button changes the bandwidth l of the long-run variance, starting with l = 0. What is KPSS for the S&P 500 price at l = 0?',
               '86.0. An integrated price has large positive autocovariances; without them, λ̂² is just the variance of the residuals and understates the long-run variance by a factor of several dozen. The statistic falls as l grows, but at l = 80 it is still 1.09 > 0.146. On returns, KPSS stays below 0.463 for every l: the conclusion I(1) for prices and I(0) for returns does not depend on the bandwidth.'),
  'vr-acf': ('From the ACF of the S&P 500, VR(5) = 1 + 1.6ρ̂(1) + 1.2ρ̂(2) + 0.8ρ̂(3) + 0.4ρ̂(4) = 0.855, and the Lo–MacKinlay estimator gives 0.856. The ▶ button raises q up to 20. How much do the two values differ at q = 20?',
             'Very little: 0.768 from the ACF and 0.767 with the Lo–MacKinlay estimator, a difference of 0.0005. For the BET at q = 20: 1.514 and 1.517. The two forms differ only by small-sample corrections, which matter little with T = 9,240 and q = 20.'),
  'lm-tests': ('For the S&P 500 at q = 2, Z(2) = −7.54 and Z*(2) = −3.37: the homoskedastic statistic is 2.24 times larger in absolute value. The ▶ button raises q up to 40. At q = 40, does Z* still reject the random walk for the S&P 500?',
               'No: Z*(40) = −1.81, although Z(40) = −3.70 still rejects. At q = 40, Z* rejects only for the BET. The ratio Z/Z* stays between 2.0 and 2.3 at all horizons: volatility clustering doubles the standard error of the estimated VR(q), and Z(q) ignores it.'),
  'vr-size': ('Under a GARCH(1,1) martingale with α = 0.12 and β = 0.86, Z(2) rejects the true null in 17.1% of the cases, Z*(2) in 4.7%. The ▶ button moves the reaction α from 0 to 0.12, with β = 0.86 fixed (α ≠ 0.12: extra runs of the lecture simulation). At α = 0.09, how often does Z(2) reject?',
              'In 9.0% of the cases; at α = 0.06 in 6.2%, at α = 0 in 4.8%. The distortion grows fast as α + β approaches 1 (0.95 at α = 0.09, 0.98 at α = 0.12), because volatility becomes very persistent. Z*(2) stays between 4.3% and 4.9% for every α.'),
  'vr-profile': ('From 1990, VR(40) is 0.724 for the S&P 500 and 1.743 for the BET, and Z* rejects for the BET at all 39 horizons. The ▶ button moves the start of the sample towards 2022. On the sample that starts in 2010, does Z* still reject the BET momentum at any horizon q = 2, …, 40?',
                 'No: the BET VR(40) falls to 1.193 and no Z*(q) exceeds 1.96. The BET momentum comes from the early years (1997–2006, ρ̂(1) = 0.270). From 2010 the S&P 500 rejects at a single horizon of 39, with VR(40) = 0.663, below the value from 1990: the mean reversion remains, but the robust band widens with the shorter sample.'),
  'vr-table': ('Over the whole sample, 10 of the 32 Z* statistics in the table exceed 1.96 in absolute value. The ▶ button moves the start of the sample towards 2022. On the sample that starts in 2010, how many remain?',
               'Three: the S&P 500 at q = 2 and Transgaz at q = 2 and 5 (for Transgaz the sample starts in 2010 anyway). The BET no longer rejects at any horizon. With the homoskedastic Z there would be 13, because Z ignores volatility clustering. With 32 independent tests at 5% we would expect 1.6 false rejections on average.'),
  'cd-table': ('With the horizons 2, 5, 10, 20 (m = 4), the 5% Chow–Denning critical value is 2.49 and the test rejects for the S&P 500, the BET and Transgaz. The ▶ button changes the set of horizons. With all horizons 2, …, 20 (m = 19), which series still reject?',
               'The S&P 500 and the BET. The critical value rises to 3.00, and Transgaz, with CD = 2.77, no longer exceeds it. More horizons mean more chances of a false rejection, hence a higher threshold; CD rises only if the added horizons have a larger |Z*|.'),
  'vr-markets': ('At q = 5, only the robust 95% intervals of PX (Z* = 2.08) and the BET (Z* = 2.03) exclude the value 1. The ▶ button raises the horizon q up to 20. At q = 20, which market excludes the value 1?',
                 'Only Ethereum (VR(20) = 1.29, Z* = 2.45); PX and the BET return inside the interval. At q = 2 it is the S&P 500 (VR(2) = 0.86, Z* = −2.33). Each horizon shows a different market: with 16 markets and many horizons, some intervals exclude 1 by chance, and the Chow–Denning test on four horizons rejects for no market.'),
  'rolling-vr': ('On 500-day windows, 29% of the BET windows and 4% of the S&P 500 windows reject at 5%. The ▶ button lengthens the window to 750, 1,000 and 1,500 days. What happens to the share of rejecting S&P 500 windows at 1,500 days?',
                 'It rises to 20%, all with VR < 1. A long window estimates the same small negative autocorrelation more precisely, so it separates it from zero more easily; at 250 days the share is 1.6%. Consecutive windows share 1,479 of 1,500 days, so the share describes the data and is not a test.'),
  'sub-table': ('At q = 5, the S&P 500 rejects in 2002–2013 (Z*(5) = −2.53). The ▶ button raises q to 10 and 20. At q = 20, does any S&P 500 sub-period still reject?',
                'None: Z*(20) = −1.33, −1.54 and −0.91, although VR(20) = 0.72 in 2002–2013, further from 1 than VR(5) = 0.80. The variance of VR(q) grows with q, and with about 3,000 days per sub-period the robust interval becomes wide. Only the BET 1997–2006 rejects at q = 20 (Z*(20) = 5.14).'),
  'calendar': ('The Wald test of equal means across the days of the week gives p = 0.69 for the S&P 500 and 0.24 for the BET (Newey–West standard errors with 5 lags). The ▶ button moves the start of the sample towards 2022. Does the BET p-value fall below 0.05 for any start?',
               'No: the lowest is 0.088, for the sample that starts in 2016; the S&P 500 stays above 0.62. Trying 17 different starts is itself a repeated search of the data: even a p below 0.05 at one of them would not be evidence of a day-of-week effect.'),
 },
}

# Predict, then check: (key, [(RO, EN), ...]); the keys agree with what the page computes (node script on ch7_data.js + common.js + ch7.js).
PREDICT = {
 'rw-paths': (1, [('Niciunul: căderea din 2008 nu are echivalent într-un mers aleator.', 'None: the 2008 fall has no counterpart in a random walk.'),
                  ('Trei: un mers aleator poate rămîne ani la rînd deasupra.', 'Three: a random walk can stay above for years.'),
                  ('Toate patru: mersurile simulate cresc mai repede decît S&P 500.', 'All four: the simulated walks grow faster than the S&P 500.')]),
 'scatter-lag': (1, [('Tot circa 0,156: momentumul persistă la toate lagurile.', 'Still about 0.156: the momentum persists at all lags.'),
                     ('Circa 0,03: corelația scade de circa cinci ori.', 'About 0.03: the correlation falls about fivefold.'),
                     ('Circa −0,156: după o continuare urmează o revenire.', 'About −0.156: a continuation is followed by a reversal.')]),
 'sqrt-time': (2, [('Mai mare: cozile groase măresc riscul pe orizonturi lungi.', 'Larger: heavy tails raise the risk over long horizons.'),
                   ('Egală, 17,92%: regula este exactă pentru orice serie.', 'Equal, 17.92%: the rule is exact for any series.'),
                   ('Mai mică, circa 15,7%: autocorelația negativă anulează o parte din mișcări.', 'Smaller, about 15.7%: the negative autocorrelation cancels part of the moves.')]),
 'white-noise': (1, [('Circa 10: un eșantion mic produce autocorelații mari.', 'About 10: a small sample produces large autocorrelations.'),
                     ('Una sau două: banda se lărgește odată cu autocorelațiile.', 'One or two: the band widens together with the autocorrelations.'),
                     ('Niciuna: un eșantion mic nu poate respinge nimic.', 'None: a small sample cannot reject anything.')]),
 'acf-processes': (1, [('Sub 0,05: 50 de valori ajung pentru un ACF.', 'Below 0.05: 50 values are enough for an ACF.'),
                       ('Circa 0,4: o traiectorie persistentă scurtă dă un ACF imprecis.', 'About 0.4: a short persistent path gives an imprecise ACF.'),
                       ('Circa 0,14 = 1/√50, mărimea erorii standard a zgomotului alb.', 'About 0.14 = 1/√50, the standard error of white noise.')]),
 'acf-returns': (1, [('Se apropie de 0: piața devine mai eficientă în timp.', 'It approaches 0: the market becomes more efficient over time.'),
                     ('Devine mai negativ, circa −0,14.', 'It becomes more negative, about −0.14.'),
                     ('Devine pozitiv, ca la BET.', 'It turns positive, as for the BET.')]),
 'acf-abs': (1, [('Crește: pătratele accentuează volatility clustering.', 'It rises: squares accentuate volatility clustering.'),
                 ('Scade, de la circa 0,09 la circa 0,04.', 'It falls, from about 0.09 to about 0.04.'),
                 ('Rămîne 0,09: puterea nu schimbă autocorelațiile.', 'It stays at 0.09: the power does not change the autocorrelations.')]),
 'lb-example': (1, [('Niciodată: autocorelațiile sînt aceleași, deci și decizia.', 'Never: the autocorrelations are the same, so is the decision.'),
                    ('De la T = 1 500: Q crește proporțional cu T.', 'From T = 1,500: Q grows in proportion to T.'),
                    ('Doar de la T = 5 000, cînd Q este de cinci ori mai mare.', 'Only from T = 5,000, when Q is five times larger.')]),
 'ac-tests': (1, [('S&P 500 și BET, ca la m = 10.', 'The S&P 500 and the BET, as at m = 10.'),
                  ('Doar BET: autocorelația S&P 500 se diluează printre multe laguri.', 'Only the BET: the S&P 500 autocorrelation is diluted among many lags.'),
                  ('Toate opt: mai multe laguri dau mai multă putere.', 'All eight: more lags give more power.')]),
 'runs': (1, [('Ambele: un tipar al semnelor nu se schimbă în timp.', 'Both: a pattern of signs does not change over time.'),
              ('Doar cea a BET, mult mai slabă (z circa −2).', 'Only the BET one, much weaker (z about −2).'),
              ('Niciuna: după 2022 ambele piețe sînt eficiente.', 'Neither: after 2022 both markets are efficient.')]),
 'spurious': (1, [('Da: cu mai multe date estimarea devine mai precisă.', 'Yes: with more data the estimate becomes more precise.'),
                  ('Nu: crește, spre 94% la T = 2 000.', 'No: it rises, towards 94% at T = 2,000.'),
                  ('Rămîne 90% pentru orice T.', 'It stays at 90% for every T.')]),
 'adf-example': (1, [('Da, pentru TLV, ca în tabelul testelor.', 'Yes, for TLV, as in the table of tests.'),
                     ('Nu: cu constantă, τ rămîne peste −1,5 pentru toate seriile.', 'No: with a constant, τ stays above −1.5 for every series.'),
                     ('Da, pentru toate: cu destule laguri orice serie respinge.', 'Yes, for all: with enough lags every series rejects.')]),
 'unit-root-bet': (1, [('Bitcoin: prețul a crescut prea repede pentru un mers aleator.', 'Bitcoin: the price rose too fast for a random walk.'),
                       ('TLV: ADF respinge, iar KPSS respinge și el.', 'TLV: ADF rejects, and KPSS rejects too.'),
                       ('Niciuna: toate prețurile sînt I(1).', 'None: all prices are I(1).')]),
 'ur-table': (1, [('Tot circa 2,2: lățimea de bandă schimbă doar valoarea critică.', 'Still about 2.2: the bandwidth changes only the critical value.'),
                  ('Mult mai mare, circa 86.', 'Much larger, about 86.'),
                  ('Sub 0,146: fără corecție, testul acceptă staționaritatea.', 'Below 0.146: without the correction the test accepts stationarity.')]),
 'vr-acf': (0, [('Foarte puțin, sub 0,001.', 'Very little, below 0.001.'),
                ('Circa 0,1: ponderile 2(1 − k/q) nu mai aproximează estimatorul la q mare.', 'About 0.1: the weights 2(1 − k/q) no longer approximate the estimator at large q.'),
                ('Formula din ACF nu se mai poate aplica la q = 20.', 'The ACF formula can no longer be applied at q = 20.')]),
 'lm-tests': (1, [('Da: revenirea la medie de la q = 2 se vede la orice orizont.', 'Yes: the mean reversion at q = 2 shows at every horizon.'),
                  ('Nu: Z*(40) este circa −1,8, deși Z(40) respinge încă.', 'No: Z*(40) is about −1.8, although Z(40) still rejects.'),
                  ('Nu, și nici Z(40) nu mai respinge.', 'No, and Z(40) does not reject either.')]),
 'vr-size': (0, [('Circa 9%: distorsiunea crește repede cînd α + β se apropie de 1.', 'About 9%: the distortion grows fast as α + β approaches 1.'),
                 ('Circa 13%, proporțional cu α: 17,1% × 0,09/0,12.', 'About 13%, in proportion to α: 17.1% × 0.09/0.12.'),
                 ('5%, oricare ar fi α: randamentele GARCH sînt necorelate.', '5%, whatever α: GARCH returns are uncorrelated.')]),
 'vr-profile': (1, [('Da, la toate orizonturile, ca din 1990.', 'Yes, at all horizons, as from 1990.'),
                    ('Nu: VR rămîne peste 1, dar în interiorul benzii robuste.', 'No: VR stays above 1, but inside the robust band.'),
                    ('Da, dar cu VR sub 1 (mean reversion).', 'Yes, but with VR below 1 (mean reversion).')]),
 'vr-table': (1, [('Tot circa 10: eșantionul mai scurt nu schimbă concluzia.', 'Still about 10: the shorter sample does not change the conclusion.'),
                  ('Trei, niciuna pentru BET.', 'Three, none for the BET.'),
                  ('Niciuna: după 2010 toate piețele sînt eficiente.', 'None: after 2010 all markets are efficient.')]),
 'cd-table': (1, [('Aceleași trei: CD nu poate scădea cînd adăugăm orizonturi.', 'The same three: CD cannot fall when we add horizons.'),
                  ('S&P 500 și BET: pragul crește mai mult decît CD.', 'The S&P 500 and the BET: the threshold rises more than CD.'),
                  ('Mai multe serii: cu 19 orizonturi testul are mai multă putere.', 'More series: with 19 horizons the test has more power.')]),
 'vr-markets': (1, [('Tot PX și BET: piețele mici rămîn ineficiente la orice orizont.', 'Still PX and the BET: small markets stay inefficient at every horizon.'),
                    ('Doar Ethereum, cu VR(20) peste 1.', 'Only Ethereum, with VR(20) above 1.'),
                    ('Niciuna: la orizonturi lungi toate intervalele includ 1.', 'None: at long horizons all intervals include 1.')]),
 'rolling-vr': (1, [('Scade: ferestrele lungi amestecă perioade diferite și anulează efectul.', 'It falls: long windows mix different periods and cancel the effect.'),
                    ('Crește, la circa 20%: o fereastră lungă estimează mai precis.', 'It rises, to about 20%: a long window estimates more precisely.'),
                    ('Rămîne circa 4%: ponderea nu depinde de lungimea ferestrei.', 'It stays at about 4%: the share does not depend on the window length.')]),
 'sub-table': (1, [('Da, 2002–2013, cu o revenire mai puternică (VR = 0,72).', 'Yes, 2002–2013, with a stronger reversal (VR = 0.72).'),
                   ('Niciuna: intervalele robuste se lărgesc la orizonturi lungi.', 'None: the robust intervals widen at long horizons.'),
                   ('Toate trei: revenirea la medie crește cu orizontul.', 'All three: the mean reversion grows with the horizon.')]),
 'calendar': (1, [('Da, pentru eșantioanele recente.', 'Yes, for the recent samples.'),
                  ('Nu: cel mai mic p-value este circa 0,09.', 'No: the lowest p-value is about 0.09.'),
                  ('Da, pentru orice început după 2000: lunea BET are media negativă.', 'Yes, for every start after 2000: the BET Monday mean is negative.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout and those elements
VEIL = {'sqrt-time': {'out': True, 'charts': ['st-chart', 'st-tab', 'st-we-box']}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('ρ̂(1) = 0,10 și ρ̂(k) = 0 pentru k ≥ 2. Cît este VR(2)?', 'ρ̂(1) = 0.10 and ρ̂(k) = 0 for k ≥ 2. What is VR(2)?'),
      key=0, opts=[
  (('1,10: momentum.', '1.10: momentum.'), ('VR(2) = 1 + 2(1 − 1/2)ρ̂(1) = 1 + ρ̂(1) = 1,10: varianța pe două zile este cu 10% mai mare decît la un mers aleator.', 'VR(2) = 1 + 2(1 − 1/2)ρ̂(1) = 1 + ρ̂(1) = 1.10: the two-day variance is 10% larger than under a random walk.')),
  (('1,20: momentum.', '1.20: momentum.'), ('1 + 2ρ̂(1) uită ponderea 1 − k/q = 1/2 de la lagul 1.', '1 + 2ρ̂(1) forgets the weight 1 − k/q = 1/2 at lag 1.')),
  (('0,90: mean reversion.', '0.90: mean reversion.'), ('O autocorelație pozitivă mărește varianța pe două zile, nu o micșorează.', 'A positive autocorrelation raises the two-day variance, it does not lower it.'))]),
 dict(q=('ADF nu respinge pe prețurile logaritmice, iar KPSS respinge. Este piața eficientă?', 'ADF does not reject on the log prices, and KPSS rejects. Is the market efficient?'),
      key=1, opts=[
  (('Da: rădăcina unitară arată că prețul este un mers aleator.', 'Yes: the unit root shows that the price is a random walk.'), ('Rădăcina unitară este necesară, dar nu suficientă: r(t) = 0,3 r(t−1) + ε(t) dă un preț I(1) cu creșteri previzibile.', 'A unit root is necessary but not sufficient: r(t) = 0.3 r(t−1) + ε(t) gives an I(1) price with predictable increments.')),
  (('Știm doar că prețurile sînt I(1); eficiența cere teste pe randamente (ACF, VR).', 'We only know that prices are I(1); efficiency needs tests on returns (ACF, VR).'), ('Testele de rădăcină unitară privesc nivelul prețului; previzibilitatea privește creșterile.', 'Unit-root tests concern the price level; predictability concerns the increments.')),
  (('Nu: KPSS respinge, deci prețurile sînt previzibile.', 'No: KPSS rejects, so prices are predictable.'), ('KPSS respinge staționaritatea, ceea ce este de așteptat pentru un mers aleator; nu spune nimic despre creșteri.', 'KPSS rejects stationarity, which is expected for a random walk; it says nothing about the increments.'))]),
 dict(q=('Pentru S&P 500, Z(q) este de circa două ori mai mare decît Z*(q). Pe care o raportăm?', 'For the S&P 500, Z(q) is about twice Z*(q). Which one do we report?'),
      key=0, opts=[
  (('Z*(q), robust la volatility clustering.', 'Z*(q), robust to volatility clustering.'), ('Volatility clustering mărește varianța lui VR(q) estimat; Z*(q) o include, Z(q) nu, deci Z(q) respinge prea des un martingal adevărat (17,1% în loc de 5% în simularea GARCH).', 'Volatility clustering raises the variance of the estimated VR(q); Z*(q) includes it, Z(q) does not, so Z(q) rejects a true martingale too often (17.1% instead of 5% in the GARCH simulation).')),
  (('Z(q), pentru că are mai multă putere.', 'Z(q), because it has more power.'), ('Puterea în plus vine dintr-o mărime greșită: Z respinge prea des și cînd ipoteza nulă este adevărată.', 'The extra power comes from a wrong size: Z rejects too often also when the null is true.')),
  (('Media lor, (Z + Z*)/2.', 'Their average, (Z + Z*)/2.'), ('Media a două statistici, dintre care una are o distribuție nulă greșită, nu are distribuția N(0, 1).', 'The average of two statistics, one with a wrong null distribution, is not N(0, 1).'))]),
 dict(q=('Testăm separat VR(q) pentru q = 2, 5, 10, 20, fiecare la 5%. Dacă testele ar fi independente, care este probabilitatea a cel puțin unei respingeri false?', 'We test VR(q) separately for q = 2, 5, 10, 20, each at 5%. If the tests were independent, what is the probability of at least one false rejection?'),
      key=1, opts=[
  (('5%: fiecare test are nivelul 5%.', '5%: each test has level 5%.'), ('Nivelul unui singur test nu este nivelul familiei de teste.', 'The level of one test is not the level of the family of tests.')),
  (('18,5%: 1 − 0,95⁴.', '18.5%: 1 − 0.95⁴.'), ('1 − 0,95⁴ = 0,185; de aceea testul Chow–Denning folosește valoarea critică 2,49 în loc de 1,96.', '1 − 0.95⁴ = 0.185; this is why the Chow–Denning test uses the critical value 2.49 instead of 1.96.')),
  (('20%: 4 × 5%.', '20%: 4 × 5%.'), ('4 × 5% este doar o margine superioară (Bonferroni); valoarea exactă pentru teste independente este 18,5%.', '4 × 5% is only an upper bound (Bonferroni); the exact value for independent tests is 18.5%.'))]),
 dict(q=('Abaterea standard zilnică a S&P 500 este 1,1337%. Cît dă regula rădăcinii pătrate pentru 20 de zile?', 'The daily standard deviation of the S&P 500 is 1.1337%. What does the square-root rule give for 20 days?'),
      key=2, opts=[
  (('22,67%: 1,1337% × 20.', '22.67%: 1.1337% × 20.'), ('Varianțele se adună, nu abaterile standard: abaterea standard crește cu √q.', 'Variances add up, not standard deviations: the standard deviation grows with √q.')),
  (('1,13%: volatilitatea nu depinde de orizont.', '1.13%: volatility does not depend on the horizon.'), ('Într-un mers aleator Var(r(t, q)) = qσ², deci riscul crește cu orizontul.', 'In a random walk Var(r(t, q)) = qσ², so risk grows with the horizon.')),
  (('5,07%: 1,1337% × √20.', '5.07%: 1.1337% × √20.'), ('√20 = 4,472; 1,1337 × 4,472 = 5,07%. Pe datele S&P 500, sumele suprapuse pe 20 de zile au 4,44%, sub regulă, pentru că randamentele sînt ușor negativ autocorelate.', '√20 = 4.472; 1.1337 × 4.472 = 5.07%. On the S&P 500 data the overlapping 20-day sums have 4.44%, below the rule, because the returns are slightly negatively autocorrelated.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch7-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH7_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH7_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH7_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', ' ', t)
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
