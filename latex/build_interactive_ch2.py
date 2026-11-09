"""Build the interactive pages of Chapter 2: interactive/ch2.html (EN) and interactive/ch2_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the
order of the slides: the slide on the left, its interactive version on the right, with live tables and worked examples.
Logic in interactive/ch2.js, inputs in interactive/ch2_data.js (Quantlets/Ch_02/export_interactive.py), slide images in
interactive/slides/. On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and
ch2.js moves each one to its own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 2
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_02'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('normal-pdf', 'sfm_ch2_normal_pdf', 'norm'), ('lognormal', 'sfm_ch2_lognormal', 'logn'),
          ('clt', 'sfm_ch2_clt', 'clt'), ('normal-approx', 'sfm_ch2_normal_approx', 'clt'),
          ('shapes', 'sfm_ch2_shapes', 'mom'), ('dax-density', 'sfm_ch2_dax_density', 'mom'),
          ('qq-normal', 'sfm_ch2_qq_normal', 'jb'),
          ('t-densities', 'sfm_ch2_t_densities', 't'), ('t-fit', 'sfm_ch2_t_fit_sp500', 't'), ('qq-t', 'sfm_ch2_qq_t', 't'),
          ('sigma-days', 'sfm_ch2_sigma_days', 'sf'), ('aggregation', 'sfm_ch2_aggregation', 'sf'),
          ('qq-horizons', 'sfm_ch2_qq_horizons', 'sf'), ('acf', 'sfm_ch2_acf', 'sf'), ('clustering', 'sfm_ch2_clustering', 'sf'),
          ('leverage', 'sfm_ch2_leverage', 'sf'), ('gain-loss', 'sfm_ch2_gain_loss', 'sf')]

PFX = {'normal-pdf': 'np', 'lognormal': 'ln', 'clt': 'cl', 'normal-approx': 'na', 'shapes': 'sh', 'dax-density': 'dx', 'qq-normal': 'qn',
       't-densities': 'td', 't-fit': 'tf', 'qq-t': 'qt', 'sigma-days': 'sg', 'aggregation': 'ag', 'qq-horizons': 'qh', 'acf': 'ac',
       'clustering': 'vc', 'leverage': 'lv', 'gain-loss': 'gl'}

SECTIONS = {
 'ro': dict(norm='Distribuția Normală', logn='Distribuția lognormală', clt='Teorema limită centrală', mom='Momentele și forma unei distribuții',
            jb='Testarea normalității: Jarque–Bera și QQ plots', t='Distribuția Student-t', sf='Faptele stilizate ale randamentelor'),
 'en': dict(norm='The Normal Distribution', logn='The Lognormal Distribution', clt='The Central Limit Theorem', mom='Moments and the Shape of a Distribution',
            jb='Testing Normality: Jarque–Bera and QQ Plots', t='The Student-t Distribution', sf='Stylised Facts of Returns'),
}

ASSETS = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin'), ('snp', 'SNP'), ('brd', 'BRD'), ('snn', 'SNN'), ('sng', 'SNG')]
INDICES = ASSETS[:4]

L = {
 'ro': dict(file='ch2_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 2 · Distribuții clasice și fapte stilizate', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Randamente logaritmice zilnice, în %, din 2010 (Bitcoin din septembrie 2014, SNN și SNG din noiembrie 2013) pînă la 18 septembrie 2026. '
       'Prețuri EODHD: indicii și Bitcoin la închidere, acțiunile BVB ajustate pentru dividende și operațiuni corporative.'),
 'en': dict(file='ch2.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 2 · Classical Distributions and Stylised Facts', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Daily log returns, in %, from 2010 (Bitcoin from September 2014, SNN and SNG from November 2013) to 18 September 2026. '
       'EODHD prices: indices and Bitcoin at the close, BVB stocks adjusted for dividends and corporate actions.'),
}


def widgets_old(lang):
    ro = lang == 'ro'
    smp = [('all', 'Tot eșantionul' if ro else 'Full sample'), ('ex', 'Fără 20 februarie–30 aprilie 2020' if ro else 'Without 20 February–30 April 2020')]
    return {
     'normal-pdf': (slider('np-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 0.5, 6, 0.1, 3) + play('np'), charts('np-chart')),
     'lognormal': (slider('ln-s', 'Parametrul s = σ√T al curbei alese' if ro else 'Parameter s = σ√T of the chosen curve', 0.05, 1.5, 0.05, 1) + play('ln'), charts('ln-chart')),
     'clt': (slider('cl-p', 'Probabilitatea p a extragerii Bernoulli' if ro else 'Probability p of the Bernoulli draw', 0.02, 0.5, 0.01, 0.2) + play('cl'), charts('cl-chart')),
     'normal-approx': (slider('na-n', 'Numărul de încercări n' if ro else 'Number of trials n', 2, 200, 1, 20)
                       + slider('na-p', 'Probabilitatea de succes p' if ro else 'Success probability p', 0.01, 0.99, 0.01, 0.1) + play('na'), charts('na-pmf', 'na-err')),
     'shapes': (slider('sh-nu', 'Gradele de libertate ν ale distribuției Student-t' if ro else 'Degrees of freedom ν of the Student-t', 4.2, 40, 0.1, 5)
                + slider('sh-a', 'Parametrul de formă α al densității skew-Normal' if ro else 'Shape parameter α of the skew-Normal density', 0, 20, 0.5, 5) + play('sh'), charts('sh-chart')),
     'dax-density': (select('dx-a', 'Seria' if ro else 'Series', ASSETS, 'dax')
                     + slider('dx-bw', 'Lățimea de bandă (multiplu al regulii Scott)' if ro else 'Bandwidth (multiple of Scott’s rule)', 0.25, 3, 0.05, 1) + play('dx'), charts('dx-chart')),
     'qq-normal': (slider('qn-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2010, 2024, 1, 2010) + play('qn'), charts('qn-chart')),
     't-densities': (slider('td-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 1, 6, 0.1, 4) + play('td'), charts('td-chart')),
     't-fit': (select('tf-a', 'Seria' if ro else 'Series', ASSETS, 'sp500')
               + slider('tf-nu', 'ν fixat (locația și scala reestimate)' if ro else 'Fixed ν (location and scale re-estimated)', 2.1, 30, 0.01, 2.78) + play('tf'), charts('tf-chart')),
     'qq-t': (slider('qt-nu', 'Gradele de libertate ν' if ro else 'Degrees of freedom ν', 0, 9, 1, 0) + play('qt'), charts('qt-chart')),
     'sigma-days': (slider('sg-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 2, 6, 0.5, 4) + play('sg'), charts('sg-chart')),
     'aggregation': (select('ag-s', 'Eșantionul' if ro else 'Sample', smp, 'all')
                     + slider('ag-h', 'Orizontul h (zile)' if ro else 'Horizon h (days)', 1, 63, 1, 21) + play('ag'), charts('ag-chart')),
     'qq-horizons': (select('qh-a', 'Seria' if ro else 'Series', INDICES, 'sp500')
                     + slider('qh-h', 'Orizontul h (zile)' if ro else 'Horizon h (days)', 1, 63, 1, 21) + play('qh'), charts('qh-chart')),
     'acf': (select('ac-s', 'Eșantionul' if ro else 'Sample', smp, 'all')
             + slider('ac-d', 'Puterea δ din |r<sub>t</sub>|<sup>δ</sup>' if ro else 'Power δ in |r<sub>t</sub>|<sup>δ</sup>', 0.25, 3, 0.25, 1) + play('ac'), charts('ac-chart')),
     'clustering': (slider('vc-w', 'Fereastra abaterii standard mobile' if ro else 'Window of the rolling standard deviation', 0, 252, 1, 0) + play('vc'), charts('vc-chart')),
     'leverage': (select('lv-s', 'Eșantionul' if ro else 'Sample', smp, 'all')
                  + slider('lv-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2010, 2022, 1, 2010) + play('lv'), charts('lv-chart')),
     'gain-loss': (slider('gl-a', 'Probabilitatea din coadă α' if ro else 'Tail probability α', 0, 4, 1, 2)
                   + slider('gl-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 1.5, 6, 0.1, 4) + play('gl'), charts('gl-chart')),
    }


# table and worked-example blocks: (anchor, frame number in the EN deck (RO: +1, one more acronym frame), section key,
# chart block that carries it on the web lecture, title check EN, title check RO)
XBLOCKS = [('rare-moves', 13, 'norm', 'normal-pdf', 'How Rare', 'Frecvența variațiilor'),
           ('large-losses', 14, 'norm', 'normal-pdf', 'Large Losses', 'pierderile mari'),
           ('bet-100', 19, 'logn', 'lognormal', '100 Lei', '100 de lei'),
           ('moments', 30, 'mom', 'dax-density', 'Moments, 2010', 'momente, 2010'),
           ('jb', 35, 'jb', 'qq-normal', 'Jarque', 'Jarque'),
           ('t-fit-table', 45, 't', 't-fit', 'Degrees of Freedom', 'gradele de libertate'),
           ('sigma-table', 52, 'sf', 'sigma-days', 'Heavy Tails, Counted', 'cozile groase în cifre'),
           ('agg-table', 54, 'sf', 'aggregation', 'Aggregational Gaussianity', 'gaussianitatea agregată'),
           ('ljung-box', 59, 'sf', 'acf', 'Weak Autocorrelation', 'autocorelația slabă'),
           ('facts-table', 65, 'sf', 'gain-loss', 'Six Stylised Facts in Eight', 'șase fapte stilizate în opt')]
ORDER = ['normal-pdf', 'rare-moves', 'large-losses', 'lognormal', 'bet-100', 'clt', 'normal-approx', 'shapes', 'moments', 'dax-density', 'jb', 'qq-normal',
         't-densities', 't-fit-table', 't-fit', 'qq-t', 'sigma-table', 'sigma-days', 'agg-table', 'aggregation', 'qq-horizons', 'acf', 'ljung-box',
         'clustering', 'leverage', 'gain-loss', 'facts-table']
PFX.update({'rare-moves': 'rm', 'large-losses': 'll', 'bet-100': 'b1', 'moments': 'mo', 'jb': 'jb', 't-fit-table': 'tt', 'sigma-table': 'st',
            'agg-table': 'ah', 'ljung-box': 'lb', 'facts-table': 'ft'})

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt .bad{color:#CD0000;font-weight:600}.xt .ok{color:#2E7D32;font-weight:600}
.xt tr.sel td{background:#EEF3FB}.xt tr.nrm td{font-style:italic;color:#1F2A44}.xt .pv{font-size:.76rem;color:#1F2A44}
.tnote{font-size:.78rem;color:#1F2A44;margin:.25rem 0 0;max-width:46rem}
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
 'll': r'z = \frac{-x - \bar r}{s}, \qquad P(r_t < -x) = \Phi(z), \qquad E = n\,\Phi(z)',
 'b1': r'\begin{gathered}\frac{P_T}{P_0} \sim LN\big(T\mu,\ T\sigma^2\big):\quad \text{MED} = 100\,e^{T\mu}, \quad \text{MEAN} = 100\,e^{T\mu + T\sigma^2/2},\\'
       r' P(P_T < P_0) = \Phi\Big(-\frac{\mu\sqrt{T}}{\sigma}\Big), \qquad q_\alpha = 100\,e^{T\mu + z_\alpha\,\sigma\sqrt{T}}\end{gathered}',
 'mo': r'm_k = \frac1n\sum_{t=1}^{n}(r_t-\bar r)^k, \qquad S = \frac{m_3}{m_2^{3/2}}, \qquad K = \frac{m_4}{m_2^2} - 3, \qquad \text{SE}(S) = \sqrt{6/n},\ \ \text{SE}(K) = \sqrt{24/n}',
 'jb': r'\text{JB} = \frac{n}{6}\Big(S^2 + \frac{K^2}{4}\Big) \;\overset{H_0}{\sim}\; \chi^2(2), \qquad p = P\big(\chi^2(2) > \text{JB}\big) = e^{-\text{JB}/2}',
 'tt': r'\text{Var}(X) = s^2\,\frac{\nu}{\nu-2}\ \ (\nu > 2), \qquad P\big(|X - \mu| > c\big) = 2\,F_{t(\nu)}\Big(-\frac{c}{s}\Big)',
 'st': r'p_k = P(|Z| > k) = 2\big(1 - \Phi(k)\big), \qquad \lambda = n\,p_k, \qquad P(N \ge N_{\text{obs}}) = 1 - \sum_{j=0}^{N_{\text{obs}}-1} e^{-\lambda}\frac{\lambda^j}{j!}',
 'cl': r'\begin{gathered}Z_n = \frac{\bar X_n - p}{\sqrt{p(1-p)/n}} = \frac{S_n - np}{\sqrt{np(1-p)}}, \qquad S_n \sim \text{Bin}(n, p),\\'
       r' P(Z_n > 1.96) = \sum_{j > np + 1.96\sqrt{np(1-p)}} \binom{n}{j} p^j (1-p)^{n-j}\end{gathered}',
}


def wtex(k, ro):
    t = WE_TEX[k]
    if ro:
        t = t.replace(r'\text{MED}', r'\text{mediana}').replace(r'\text{MEAN}', r'\text{media}').replace('1.96', '1{,}96')
    else:
        t = t.replace(r'\text{MED}', r'\text{median}').replace(r'\text{MEAN}', r'\text{mean}')
    return t


def widgets_all(lang):
    ro = lang == 'ro'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    wt = lambda k, r, e: we(k + '-we', f'{WE}: {r if ro else e}', wtex(k, ro))
    smp = [('all', 'Tot eșantionul' if ro else 'Full sample'), ('ex', 'Fără 20 februarie–30 aprilie 2020' if ro else 'Without 20 February–30 April 2020')]
    yr = lambda i: slider(i, 'Eșantionul începe în' if ro else 'Sample starts in', 2010, 2024, 1, 2010)
    ser = lambda i, lab_ro, lab_en, d: select(i, lab_ro if ro else lab_en, ASSETS, d)
    W = widgets_old(lang)
    W['normal-pdf'] = (W['normal-pdf'][0], CSS + W['normal-pdf'][1])
    W['clt'] = (W['clt'][0] + select('cl-n', 'Numărul de extrageri n al exemplului lucrat' if ro else 'Number of draws n of the worked example',
                                     [('5', '5'), ('30', '30'), ('300', '300')], '30'),
                W['clt'][1] + wt('cl', 'distribuția binomială exactă a mediei', 'the exact binomial distribution of the mean'))
    W.update({
     'rare-moves': (ser('rm-a', 'Seria (coloanele observate, rîndul evidențiat)', 'Series (observed columns, highlighted row)', 'sp500') + yr('rm-y') + play('rm'),
                    xh('Distribuția Normală: P(|Z| > k) și cît de des apare o astfel de zi; zilele observate ale seriei alese' if ro
                       else 'Normal distribution: P(|Z| > k) and how often such a day occurs; the observed days of the chosen series') + tw('rm-tab')
                    + xh('Ponderea zilelor aflate la cel mult k abateri standard de medie (regula 68–95–99,7)' if ro
                         else 'Share of days within k standard deviations of the mean (the 68–95–99.7 rule)') + tw('rm-tab2')),
     'large-losses': (ser('ll-a', 'Seria', 'Series', 'sp500') + slider('ll-x', 'Pragul pierderii x (%)' if ro else 'Loss threshold x (%)', 1, 10, 0.5, 3)
                      + yr('ll-y') + play('ll'),
                      charts('ll-chart') + xh('Zile cu randamentul logaritmic sub −x%: observate și așteptate după distribuția Normală' if ro
                                              else 'Days with a log return below −x%: observed and expected under the Normal distribution') + tw('ll-tab')
                      + wt('ll', 'pierderi mai mari decît x%', 'losses larger than x%')),
     'bet-100': (ser('b1-a', 'Seria', 'Series', 'bet') + slider('b1-T', 'Orizontul T (ani)' if ro else 'Horizon T (years)', 1, 20, 1, 1)
                 + select('b1-q', 'Cuantila α' if ro else 'Quantile α', [('1', '1%'), ('5', '5%'), ('10', '10%')], '5') + yr('b1-y') + play('b1'),
                 charts('b1-chart') + xh('Valoarea a 100 de lei investiți, P<sub>T</sub>/P<sub>0</sub> ~ LN(Tμ, Tσ²)' if ro
                                         else 'Value of 100 lei invested, P<sub>T</sub>/P<sub>0</sub> ~ LN(Tμ, Tσ²)') + tw('b1-tab')
                 + wt('b1', 'valoarea a 100 de lei după T ani', 'the value of 100 lei after T years')),
     'moments': (yr('mo-y') + select('mo-s', 'Eșantionul' if ro else 'Sample', smp, 'all') + ser('mo-a', 'Seria exemplului lucrat', 'Series of the worked example', 'sp500') + play('mo'),
                 xh('Randamente logaritmice zilnice (%): momente' if ro else 'Daily log returns (%): moments') + tw('mo-tab') + charts('mo-chart')
                 + wt('mo', 'asimetria și excesul de boltire din randamentele reale', 'skewness and excess kurtosis from real returns')),
     'jb': (ser('jb-a', 'Seria exemplului lucrat', 'Series of the worked example', 'sp500') + yr('jb-y') + play('jb'),
            charts('jb-chart') + xh('Testul Jarque–Bera, randamente logaritmice zilnice' if ro else 'Jarque–Bera test, daily log returns') + tw('jb-tab')
            + wt('jb', 'statistica Jarque–Bera pas cu pas', 'the Jarque–Bera statistic step by step')),
     't-fit-table': (yr('tt-y') + ser('tt-a', 'Seria exemplului lucrat', 'Series of the worked example', 'sp500') + play('tt'),
                     xh('Distribuția Student-t estimată prin verosimilitate maximă' if ro else 'Student-t fitted by maximum likelihood') + tw('tt-tab') + charts('tt-chart')
                     + wt('tt', 'varianța ν/(ν − 2) și probabilitatea din coadă a distribuției t estimate', 'the variance ν/(ν − 2) and the tail probability of the fitted t').replace(
                         '<ol id="tt-we">', '<div class="wctl">' + slider('tt-k', 'Pragul k (abateri standard)' if ro else 'Threshold k (standard deviations)', 2, 6, 0.5, 4) + '</div><ol id="tt-we">')),
     'sigma-table': (yr('st-y') + select('st-k', 'Pragul k al coloanei p-value și al exemplului lucrat' if ro else 'Threshold k of the p-value column and the worked example',
                                         [('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], '4')
                     + ser('st-a', 'Seria graficului și a exemplului lucrat', 'Series of the chart and the worked example', 'sp500') + play('st'),
                     xh('Zile cu |r<sub>t</sub> − r̄| > k s' if ro else 'Days with |r<sub>t</sub> − r̄| > k s') + tw('st-tab') + charts('st-chart')
                     + wt('st', 'P(|Z| > k), numărul așteptat de zile și p-value Poisson', 'P(|Z| > k), the expected number of days and the Poisson p-value')),
     'agg-table': (select('ah-s', 'Eșantionul' if ro else 'Sample', smp, 'all')
                   + select('ah-m', 'Statistica' if ro else 'Statistic', [('exkurt', 'excesul de boltire' if ro else 'excess kurtosis'), ('skew', 'asimetria' if ro else 'skewness'),
                                                                        ('jbp', 'p-value Jarque–Bera' if ro else 'Jarque–Bera p-value'), ('n', 'numărul de observații' if ro else 'number of observations')], 'exkurt') + play('ah'),
                   xh('Randamente logaritmice pe h zile, blocuri disjuncte, din 2010' if ro else 'h-day log returns, non-overlapping blocks, from 2010') + tw('ah-tab') + charts('ah-chart')),
     'ljung-box': (select('lb-s', 'Eșantionul' if ro else 'Sample', smp, 'all') + slider('lb-m', 'Numărul de laguri m' if ro else 'Number of lags m', 1, 50, 1, 10) + yr('lb-y') + play('lb'),
                   xh('Testul Ljung–Box pentru randamente și pentru randamentele absolute' if ro else 'Ljung–Box test for returns and for absolute returns') + tw('lb-tab') + charts('lb-chart')),
     'facts-table': (yr('ft-y') + select('ft-s', 'Eșantionul' if ro else 'Sample', smp, 'all')
                     + slider('ft-h', 'Orizontul h al coloanei a doua (zile)' if ro else 'Horizon h of the second column (days)', 2, 63, 1, 21) + play('ft'),
                     xh('Statistica fiecărui fapt stilizat' if ro else 'The statistic of each stylised fact') + tw('ft-tab')),
    })
    return W

QA = {
 'ro': {
  'normal-pdf': ('După distribuția Normală, o zi dincolo de 3 abateri standard apare o dată la 370 de zile. Butonul ▶ mută pragul pînă la 6. De cîte ori mai rară este o zi dincolo de 4 abateri standard?',
                 'De circa 43 de ori. P(|Z| > 4) = 0,0063%, adică o zi din 15 787, aproximativ 63 de ani de tranzacționare cu 252 de zile pe an; P(|Z| > 3) = 0,27%, o zi din 370. Raportul densităților în 3 și 4, φ(3)/φ(4) = 33, nu este raportul probabilităților din coadă. La 6 abateri standard, distribuția Normală dă o zi din 507 milioane.'),
  'lognormal': ('Butonul ▶ crește s de la 0,1 la 1,5. Ce se întîmplă cu modul, mediana și media distribuției lognormale LN(0, s²)?',
                'Mediana rămîne 1, deoarece e<sup>m</sup> nu depinde de s. Media e<sup>s²/2</sup> crește pînă la 3,08, iar modul e<sup>−s²</sup> scade pînă la 0,105. La s = 1, media este 1,65, modul 0,37 și asimetria 6,18. Distanța dintre medie și mediană crește cu volatilitatea și cu orizontul, prin s = σ√T.'),
  'clt': ('Pe slide p = 0,2, iar asimetria mediei standardizate este 0,27 pentru n = 30. Butonul ▶ scade p de la 0,5 la 0,02. Cu p = 0,02, este asimetria pentru n = 300 mai mică decît 0,27?',
          'Nu. Asimetria mediei este (1 − 2p)/√(np(1 − p)) = 0,96/√5,88 = 0,40 pentru n = 300 și p = 0,02, mai mare decît 0,27 pentru n = 30 și p = 0,2. Asimetria scade ca 1/√n, dar pornește de la un nivel care depinde de p: cu cît p este mai departe de 0,5, cu atît trebuie mai multe extrageri. Pagina calculează distribuția exactă a mediei; ponderea peste 1,96 pentru p = 0,2 este 5,8%, 2,6% și 2,8% (n = 5, 30, 300), iar în cele 20 000 de simulări ale slide-ului 5,8%, 2,7% și 2,8%.'),
  'normal-approx': ('Cu p = 0,1, butonul ▶ crește n de la 5 la 200. Regula practică cere np(1 − p) de cel puțin circa 9, adică n = 100. Ajunge eroarea maximă a CDF sub 0,01 pînă la n = 200?',
                    'Nu. Cu corecția de continuitate, eroarea maximă a CDF este 0,037 la n = 20, 0,017 la n = 100 și 0,012 la n = 200. Regula np(1 − p) ≥ 9 indică de unde aproximarea devine utilizabilă, nu o precizie garantată; eroarea scade lent, aproximativ ca 1/√n, cu mici oscilații de la un n la altul. Fără corecția de continuitate, eroarea la n = 20 este 0,177.'),
  'shapes': ('Pe slide, parametrul de formă α = 5 dă o densitate skew-Normal cu asimetria 0,85. Butonul ▶ crește α de la 0 la 20 (și coboară ν de la 40 la 4,5 în panoul din stînga). Ce asimetrie are densitatea la α = 20?',
             '0,99. Asimetria familiei skew-Normal crește cu α, dar are o limită de circa 0,995: la α = 10 este 0,96, la α = 20 este 0,99. O asimetrie mai mare nu se poate obține cu această familie. Excesul de boltire al distribuției Student-t cu varianța 1 este 6/(ν − 4): 6 pentru ν = 5, 1 pentru ν = 10.'),
  'dax-density': ('La −6%, densitatea estimată a randamentelor DAX este de 439 de ori mai mare decît densitatea Normală, cu lățimea de bandă dată de regula Scott. Ce se întîmplă cu acest raport dacă lățimea de bandă se înjumătățește?',
                  'Crește la circa 580; la un sfert din lățimea Scott ajunge la 1 100, iar la dublu este 522. În coadă sînt puține observații, iar estimarea densității acolo depinde de lățimea de bandă. Pentru orice lățime încercată, raportul rămîne de ordinul sutelor: concluzia că DAX are cozi mult mai groase decît distribuția Normală nu depinde de această alegere. Vîrful densității estimate scade de la 0,47 la 0,34 cînd lățimea crește de la 0,5 la 3 ori regula Scott.'),
  'qq-normal': ('Butonul ▶ mută începutul eșantionului de la 2010 la 2024. Dacă eșantionul începe în 2021, după martie 2020, mai depășește cel mai mic randament standardizat al S&P 500 cuantila Normală corespunzătoare?',
                'Da. De la începutul lui 2021, cel mai mic randament al S&P 500 este −6,2%, pe 4 aprilie 2025: −6,0 abateri standard, față de cuantila Normală −3,4. Pe tot eșantionul, ziua de 16 martie 2020 dă −11,8 față de −3,7. Martie 2020 mărește abaterea, dar nu o produce: și BET (−6,1) și Bitcoin (−5,8) rămîn departe de dreaptă după 2021.'),
  't-densities': ('Cu ν = 5 și varianța 1, o zi dincolo de 4 abateri standard este de 56 de ori mai probabilă decît în cazul distribuției Normale. Butonul ▶ crește pragul pînă la 6. Ce se întîmplă cu acest raport?',
                  'Crește: 2 316 la 5 abateri standard și circa 290 000 la 6. Coada distribuției Student-t scade ca o putere, |x|<sup>−ν</sup>, iar coada Normală ca e<sup>−x²/2</sup>, deci raportul crește fără limită. La 6 abateri standard, P(|X| > 6) este 0,057% pentru t(5) și 0,0000002% pentru distribuția Normală.'),
  't-fit': ('Butonul ▶ fixează ν la valori de la 2,2 la 30 și reestimează locația și scala pentru S&P 500. Care valoare dă o log-verosimilitate mai mare: ν = 2,2 sau ν = 4?',
            'ν = 2,2: log-verosimilitatea este −5 704,8, față de −5 714,6 la ν = 4. Maximul, −5 693,8, este la estimația ν̂ = 2,78, iar profilul nu este simetric în jurul ei: scade încet spre ν mic și mai repede spre ν mare. La ν = 30 log-verosimilitatea este −6 054,5, spre −6 301,2, valoarea distribuției Normale. Diferența de AIC dintre distribuția Normală și distribuția t este 12 606,4 − 11 393,6 = 1 212,8 în favoarea distribuției t.'),
  'qq-t': ('Cu ν estimat (2,78), distribuția t prevede pentru S&P 500 un minim de −15,8%, sub minimul observat de −12,8%. Butonul ▶ fixează ν la valori mai mari. Pentru ce ν se apropie minimul prevăzut de cel observat?',
           'Între ν = 3 și ν = 4. Cu locația și scala reestimate, minimul prevăzut este −13,6% la ν = 3 și −8,7% la ν = 4, față de −12,8% observat. Un ν mai mare înseamnă cozi mai subțiri. La ν = 4, centrul se potrivește la fel de bine, dar extremele observate ies din nou sub dreaptă; nicio valoare unică a lui ν nu potrivește simultan centrul și extremele.'),
  'sigma-days': ('La 4 abateri standard, S&P 500 are 28 de zile, față de 0,27 așteptate după distribuția Normală. Butonul ▶ ridică pragul pînă la 6. Cîte zile dincolo de 6 abateri standard are S&P 500?',
                 '8 zile, față de 0,0000083 așteptate (o zi din 507 milioane). BET are 11, DAX 4, Bitcoin 4. Raportul dintre zilele observate și cele așteptate crește cu pragul: 5,5 la 3 abateri standard, 105 la 4, circa un milion la 6.'),
  'aggregation': ('Excesul de boltire al randamentelor zilnice ale S&P 500 este 13,5. Cît devine dacă eliminăm perioada 20 februarie–30 aprilie 2020, adică 50 dintre cele 4 201 zile?',
                  'Circa 5,4. Excesul de boltire lunar (h = 21) scade de la 3,7 la 0,8. Momentul de ordinul 4 depinde mult de cîteva valori extreme: 50 de zile, 1,2% din eșantion, explică mai mult de jumătate din excesul de boltire zilnic. Nici fără ele randamentele zilnice nu sînt Normale: 5,4 este mult peste 0.'),
  'qq-horizons': ('La h = 63 de zile (un trimestru, n = 66), mai respinge testul Jarque–Bera normalitatea randamentelor S&P 500 la 5%?',
                  'Da: p-value 0,00004, cu asimetria −1,13 și excesul de boltire 1,5. Pentru BET (p-value 0,07) și Bitcoin (0,22), testul nu mai respinge la acest orizont. Nerespingerea nu arată că randamentele trimestriale sînt Normale: cu 66 de observații, testul are putere mică.'),
  'acf': ('Pe tot eșantionul, ρ̂(1) al randamentelor S&P 500 este −0,11, iar banda de 95% este ±0,03. Cît devine ρ̂(1) fără perioada 20 februarie–30 aprilie 2020?',
          '−0,04, aproape de marginea benzii ±0,03. Cea mai mare parte a valorii −0,11 vine din martie 2020, cînd scăderile mari și revenirile au alternat. Autocorelațiile randamentelor absolute rămîn mari și fără această perioadă. Puterea δ arată același lucru pentru |r<sub>t</sub>|<sup>δ</sup>: la S&P 500, autocorelația la lagul 10 este 0,20 pentru δ = 0,5, 0,26 pentru δ = 1 și 0,22 pentru δ = 2.'),
  'clustering': ('Cu o fereastră de 63 de zile (un trimestru), de cîte ori este mai mare cea mai mare abatere standard mobilă a S&P 500 decît cea mai mică?',
                 'De circa 12 ori: 3,94% pe zi în fereastra care se termină pe 20 mai 2020, față de 0,32% în fereastra care se termină pe 13 decembrie 2017. Raportul varianțelor este circa 154. La BET raportul abaterilor standard este 8,2, la Bitcoin 8,3. Volatilitatea unei zile depinde de perioada în care se află.'),
  'leverage': ('Butonul ▶ mută începutul eșantionului de la 2010 la 2022. Dacă eșantionul începe în 2021, rămîne media L(1..5) a S&P 500 negativă și în afara benzii de 95%?',
               'Da. De la începutul lui 2021, media L(1..5) a S&P 500 este −0,10 (−0,12 pe tot eșantionul), iar banda este ±0,05. Efectul de levier nu vine dintr-o singură criză: fără 20 februarie–30 aprilie 2020, media pe tot eșantionul este −0,11. La Bitcoin, media rămîne aproape de zero în toate variantele.'),
  'gain-loss': ('La 1%, pierderea depășită într-o zi din 100 este mai mare decît cîștigul la S&P 500, DAX, BET și Bitcoin. Butonul ▶ crește probabilitatea α de la 0,1% la 5% și coboară pragul k din panoul din dreapta la cuantila Normală corespunzătoare. La α = 5%, este pierderea mai mare decît cîștigul la toți cei patru indici?',
                'Nu. La 5%, S&P 500 (1,65% față de 1,54%) și DAX (1,94% față de 1,82%) au încă pierderea mai mare, dar la BET (1,48% față de 1,52%) și Bitcoin (5,33% față de 5,35%) cîștigul este puțin mai mare. Asimetria cîștig/pierdere se vede în coada îndepărtată; la α = 0,1%, pierderea este mai mare la toți patru (S&P 500 6,14% față de 6,01%).'),
 },
 'en': {
  'normal-pdf': ('Under the Normal distribution, a day beyond 3 standard deviations occurs once in 370 days. The ▶ button moves the threshold up to 6. How many times rarer is a day beyond 4 standard deviations?',
                 'About 43 times. P(|Z| > 4) = 0.0063%, one day in 15,787, about 63 years of trading at 252 days a year; P(|Z| > 3) = 0.27%, one day in 370. The ratio of the densities at 3 and 4, φ(3)/φ(4) = 33, is not the ratio of the tail probabilities. At 6 standard deviations, the Normal distribution gives one day in 507 million.'),
  'lognormal': ('The ▶ button raises s from 0.1 to 1.5. What happens to the mode, the median and the mean of the lognormal distribution LN(0, s²)?',
                'The median stays at 1, because e<sup>m</sup> does not depend on s. The mean e<sup>s²/2</sup> rises to 3.08 and the mode e<sup>−s²</sup> falls to 0.105. At s = 1, the mean is 1.65, the mode 0.37 and the skewness 6.18. The gap between mean and median grows with volatility and with the horizon, through s = σ√T.'),
  'clt': ('On the slide p = 0.2, and the skewness of the standardised mean is 0.27 for n = 30. The ▶ button lowers p from 0.5 to 0.02. With p = 0.02, is the skewness for n = 300 below 0.27?',
          'No. The skewness of the mean is (1 − 2p)/√(np(1 − p)) = 0.96/√5.88 = 0.40 for n = 300 and p = 0.02, above the 0.27 for n = 30 and p = 0.2. The skewness falls as 1/√n, but from a level that depends on p: the further p is from 0.5, the more draws are needed. The page computes the exact distribution of the mean; the share above 1.96 for p = 0.2 is 5.8%, 2.6% and 2.8% (n = 5, 30, 300), and in the 20,000 simulations of the slide 5.8%, 2.7% and 2.8%.'),
  'normal-approx': ('With p = 0.1, the ▶ button raises n from 5 to 200. The rule of thumb asks for np(1 − p) of at least about 9, that is n = 100. Does the largest CDF error fall below 0.01 by n = 200?',
                    'No. With the continuity correction, the largest CDF error is 0.037 at n = 20, 0.017 at n = 100 and 0.012 at n = 200. The rule np(1 − p) ≥ 9 says where the approximation becomes usable, not a guaranteed accuracy; the error falls slowly, roughly as 1/√n, with small oscillations from one n to the next. Without the continuity correction, the error at n = 20 is 0.177.'),
  'shapes': ('On the slide, the shape parameter α = 5 gives a skew-Normal density with skewness 0.85. The ▶ button raises α from 0 to 20 (and lowers ν from 40 to 4.5 in the left panel). What is the skewness of the density at α = 20?',
             '0.99. The skewness of the skew-Normal family grows with α but has a limit of about 0.995: at α = 10 it is 0.96, at α = 20 it is 0.99. A larger skewness cannot be obtained with this family. The excess kurtosis of the unit-variance Student-t distribution is 6/(ν − 4): 6 for ν = 5, 1 for ν = 10.'),
  'dax-density': ('At −6%, the estimated density of DAX returns is 439 times the Normal density, with the bandwidth given by Scott’s rule. What happens to this ratio if the bandwidth is halved?',
                  'It rises to about 580; at a quarter of Scott’s bandwidth it reaches 1,100, and at twice the bandwidth it is 522. There are few observations in the tail, and the density estimate there depends on the bandwidth. For every bandwidth tried, the ratio stays in the hundreds: the conclusion that the DAX has much heavier tails than the Normal distribution does not depend on this choice. The peak of the estimated density falls from 0.47 to 0.34 as the bandwidth rises from 0.5 to 3 times Scott’s rule.'),
  'qq-normal': ('The ▶ button moves the start of the sample from 2010 to 2024. If the sample starts in 2021, after March 2020, does the smallest standardised S&P 500 return still lie beyond the corresponding Normal quantile?',
                'Yes. Since the start of 2021, the smallest S&P 500 return is −6.2%, on 4 April 2025: −6.0 standard deviations, against the Normal quantile −3.4. Over the whole sample, 16 March 2020 gives −11.8 against −3.7. March 2020 makes the deviation larger but does not create it: BET (−6.1) and Bitcoin (−5.8) also stay far from the line after 2021.'),
  't-densities': ('With ν = 5 and unit variance, a day beyond 4 standard deviations is 56 times more likely than under the Normal distribution. The ▶ button raises the threshold up to 6. What happens to this ratio?',
                  'It rises: 2,316 at 5 standard deviations and about 290,000 at 6. The Student-t tail decays as a power, |x|<sup>−ν</sup>, and the Normal tail as e<sup>−x²/2</sup>, so the ratio grows without bound. At 6 standard deviations, P(|X| > 6) is 0.057% for t(5) and 0.0000002% for the Normal distribution.'),
  't-fit': ('The ▶ button fixes ν at values from 2.2 to 30 and re-estimates the location and scale for the S&P 500. Which value gives the higher log-likelihood: ν = 2.2 or ν = 4?',
            'ν = 2.2: the log-likelihood is −5,704.8, against −5,714.6 at ν = 4. The maximum, −5,693.8, is at the estimate ν̂ = 2.78, and the profile is not symmetric around it: it falls slowly towards small ν and faster towards large ν. At ν = 30 the log-likelihood is −6,054.5, towards −6,301.2, the value of the Normal distribution. The AIC difference between the Normal and the t distribution is 12,606.4 − 11,393.6 = 1,212.8 in favour of the t distribution.'),
  'qq-t': ('With the estimated ν (2.78), the t distribution predicts a minimum of −15.8% for the S&P 500, below the observed minimum of −12.8%. The ▶ button fixes ν at larger values. For which ν does the predicted minimum come close to the observed one?',
           'Between ν = 3 and ν = 4. With the location and scale re-estimated, the predicted minimum is −13.6% at ν = 3 and −8.7% at ν = 4, against −12.8% observed. A larger ν means thinner tails. At ν = 4 the centre fits as well, but the observed extremes again fall below the line; no single value of ν fits the centre and the extremes at the same time.'),
  'sigma-days': ('At 4 standard deviations, the S&P 500 has 28 days, against 0.27 expected under the Normal distribution. The ▶ button raises the threshold up to 6. How many days beyond 6 standard deviations does the S&P 500 have?',
                 '8 days, against 0.0000083 expected (one day in 507 million). BET has 11, DAX 4, Bitcoin 4. The ratio of observed to expected days grows with the threshold: 5.5 at 3 standard deviations, 105 at 4, about one million at 6.'),
  'aggregation': ('The excess kurtosis of daily S&P 500 returns is 13.5. What does it become if we remove the period 20 February–30 April 2020, that is 50 of the 4,201 days?',
                  'About 5.4. The monthly excess kurtosis (h = 21) falls from 3.7 to 0.8. The fourth moment depends heavily on a few extreme values: 50 days, 1.2% of the sample, account for more than half of the daily excess kurtosis. Even without them, daily returns are not Normal: 5.4 is far above 0.'),
  'qq-horizons': ('At h = 63 days (a quarter, n = 66), does the Jarque–Bera test still reject normality of S&P 500 returns at 5%?',
                  'Yes: p-value 0.00004, with skewness −1.13 and excess kurtosis 1.5. For BET (p-value 0.07) and Bitcoin (0.22), the test no longer rejects at this horizon. Not rejecting does not show that quarterly returns are Normal: with 66 observations, the test has little power.'),
  'acf': ('Over the full sample, ρ̂(1) of S&P 500 returns is −0.11, and the 95% band is ±0.03. What is ρ̂(1) without the period 20 February–30 April 2020?',
          '−0.04, close to the edge of the ±0.03 band. Most of the −0.11 comes from March 2020, when large falls and rebounds alternated. The autocorrelations of absolute returns stay large without this period too. The power δ shows the same for |r<sub>t</sub>|<sup>δ</sup>: for the S&P 500, the autocorrelation at lag 10 is 0.20 for δ = 0.5, 0.26 for δ = 1 and 0.22 for δ = 2.'),
  'clustering': ('With a 63-day window (a quarter), how many times larger is the largest rolling standard deviation of the S&P 500 than the smallest?',
                 'About 12 times: 3.94% a day in the window ending on 20 May 2020, against 0.32% in the window ending on 13 December 2017. The ratio of the variances is about 154. For BET the ratio of standard deviations is 8.2, for Bitcoin 8.3. The volatility of a day depends on the period it falls in.'),
  'leverage': ('The ▶ button moves the start of the sample from 2010 to 2022. If the sample starts in 2021, does the mean L(1..5) of the S&P 500 stay negative and outside the 95% band?',
               'Yes. Since the start of 2021, the mean L(1..5) of the S&P 500 is −0.10 (−0.12 over the full sample), and the band is ±0.05. The leverage effect does not come from a single crisis: without 20 February–30 April 2020, the full-sample mean is −0.11. For Bitcoin, the mean stays close to zero in every variant.'),
  'gain-loss': ('At 1%, the loss exceeded on one day in 100 is larger than the gain for the S&P 500, DAX, BET and Bitcoin. The ▶ button raises the probability α from 0.1% to 5% and lowers the threshold k of the right panel to the matching Normal quantile. At α = 5%, is the loss larger than the gain for all four indices?',
                'No. At 5%, the S&P 500 (1.65% against 1.54%) and the DAX (1.94% against 1.82%) still have the larger loss, but for BET (1.48% against 1.52%) and Bitcoin (5.33% against 5.35%) the gain is slightly larger. The gain/loss asymmetry shows in the far tail; at α = 0.1%, the loss is larger for all four (S&P 500 6.14% against 6.01%).'),
 },
}

# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH2_TEXT = {
 np: (k, a, b, d, y) => `La cel mult ${k} abateri standard de medie: ${a} din probabilitate. Dincolo de ${k}: ${b}, o zi din ${d} (${y} ani de 252 de zile).`,
 ln: (s, mo, mn, sk) => `s = ${s}: modul ${mo}, mediana 1, media ${mn}, asimetria ${sk}.`,
 cl: (p, t, s) => `p = ${p}. Ponderea mediilor standardizate peste 1,96 (distribuția exactă): ${t}; distribuția Normală: 2,5%. Asimetria mediei pentru n = 5, 30, 300: ${s}.`,
 clSim: t => `Pe slide, din 20 000 de simulări: ${t}.`,
 na: (n, p, v, e, e0) => `n = ${n}, p = ${p}: np(1 − p) = ${v}. Cea mai mare eroare a CDF: ${e} cu corecția de continuitate, ${e0} fără ea.`,
 sh: (nu, k, a, s) => `Student-t(${nu}) cu varianța 1: excesul de boltire 6/(ν − 4) = ${k}. Skew-Normal cu α = ${a}: asimetria ±${s}.`,
 dx: (a, b, p, pn, w, wn, r) => `${a}, lățimea de bandă ${b} × Scott: vîrful densității estimate ${p}, față de ${pn} pentru distribuția Normală; ${w} din zile se află la cel mult o abatere standard de medie (${wn} pentru distribuția Normală); la −6%, densitatea estimată este de ${r} ori densitatea Normală.`,
 qn: (y, t) => `Eșantion din ${y}. Cel mai mic randament standardizat față de cuantila Normală: ${t}.`,
 qnItem: (a, e, q) => `${a} ${e} față de ${q}`,
 qtFit: 'ν estimat pentru fiecare serie',
 qt: t => `Cel mai mic randament observat față de cuantila distribuției t (locația și scala estimate): ${t}.`,
 qtItem: (a, nu, e, q) => `${a} (ν = ${nu}) ${e}% față de ${q}%`,
 td: (k, l, n, r) => `P(|X| > ${k}) cu varianța 1: ${l}; distribuția Normală ${n}. Cu ν = 5, o zi dincolo de ${k} abateri standard este de ${r} ori mai probabilă decît în cazul distribuției Normale.`,
 tf: (a, nu, ll, nh, llh, ln) => `${a}, ν = ${nu}: log-verosimilitatea ${ll}. Estimația de verosimilitate maximă: ν̂ = ${nh}, log-verosimilitatea ${llh}; distribuția Normală: ${ln}.`,
 sg: (k, p, e, a, o, x, eo) => `Dincolo de ${k} abateri standard, distribuția Normală dă probabilitatea ${p}, o zi din ${e}. ${a}: ${ch2De(o)} zile observate, ${x} așteptate (o zi din ${eo} observate).`,
 sgAll: (o, x) => `Toate cele opt serii: ${ch2De(o)} zile observate, ${x} așteptate.`,
 agAll: 'tot eșantionul', agEx: 'fără 20 februarie–30 aprilie 2020',
 ag: (h, s, t) => `h = ${h} zile, ${s}. Excesul de boltire: ${t}.`,
 qh: (a, h, n, s, k, p, k1) => `${a}, randamente pe ${h} zile: n = ${n}, asimetria ${s}, excesul de boltire ${k} (zilnic: ${k1}), Jarque–Bera p-value ${p}.`,
 ac: (s, r1, d, a1, a10, a50, b) => `${s[0].toUpperCase() + s.slice(1)}. ρ̂(1) al randamentelor: ${r1}; banda de 95%: ±${b}. S&P 500, |r|^${d}: ${a1} la lagul 1, ${a10} la lagul 10, ${a50} la lagul 50.`,
 vc0: 'Randamentele zilnice; o fereastră mai mare decît zero adaugă banda ± 2 abateri standard mobile.',
 vcItem0: (a, s, m, d) => `${a}: abaterea standard pe tot eșantionul ${s}%; cea mai slabă zi ${m}%, pe ${d}`,
 vc: w => `Abaterea standard mobilă pe ${w} zile, cea mai mare și cea mai mică (data de sfîrșit a ferestrei):`,
 vcItem: (a, h, dh, l, dl, r) => `${a}: ${h}% (${dh}) și ${l}% (${dl}), raportul ${r}`,
 lv: (y, s, t, b) => `Eșantion din ${y}, ${s}. Media L(1..5): ${t}; banda de 95%: ±${b}.`,
 gl: (a, b, n, t, o) => `Pierderea −(cuantila ${a}) depășește cîștigul (cuantila ${b}) la ${n} din 8 serii: ${t}. Cîștigul este cel puțin egal cu pierderea la: ${o}.`,
 gl2: (k, n, o) => `Dincolo de ${k} abateri standard: mai multe zile de scădere decît de creștere la ${n} din 8 serii; nu și la: ${o}.`
};
""",
 'en': r"""window.CH2_TEXT = {
 np: (k, a, b, d, y) => `Within ${k} standard deviations of the mean: ${a} of the probability. Beyond ${k}: ${b}, one day in ${d} (${y} years of 252 days).`,
 ln: (s, mo, mn, sk) => `s = ${s}: mode ${mo}, median 1, mean ${mn}, skewness ${sk}.`,
 cl: (p, t, s) => `p = ${p}. Share of standardised means above 1.96 (exact distribution): ${t}; Normal distribution: 2.5%. Skewness of the mean for n = 5, 30, 300: ${s}.`,
 clSim: t => `On the slide, from 20,000 simulations: ${t}.`,
 na: (n, p, v, e, e0) => `n = ${n}, p = ${p}: np(1 − p) = ${v}. Largest CDF error: ${e} with the continuity correction, ${e0} without it.`,
 sh: (nu, k, a, s) => `Unit-variance Student-t(${nu}): excess kurtosis 6/(ν − 4) = ${k}. Skew-Normal with α = ${a}: skewness ±${s}.`,
 dx: (a, b, p, pn, w, wn, r) => `${a}, bandwidth ${b} × Scott: peak of the estimated density ${p}, against ${pn} for the Normal distribution; ${w} of the days lie within one standard deviation of the mean (${wn} for the Normal distribution); at −6%, the estimated density is ${r} times the Normal density.`,
 qn: (y, t) => `Sample from ${y}. Smallest standardised return against the Normal quantile: ${t}.`,
 qnItem: (a, e, q) => `${a} ${e} against ${q}`,
 qtFit: 'ν estimated for each series',
 qt: t => `Smallest observed return against the quantile of the t distribution (estimated location and scale): ${t}.`,
 qtItem: (a, nu, e, q) => `${a} (ν = ${nu}) ${e}% against ${q}%`,
 td: (k, l, n, r) => `P(|X| > ${k}) at unit variance: ${l}; Normal distribution ${n}. With ν = 5, a day beyond ${k} standard deviations is ${r} times more likely than under the Normal distribution.`,
 tf: (a, nu, ll, nh, llh, ln) => `${a}, ν = ${nu}: log-likelihood ${ll}. Maximum likelihood estimate: ν̂ = ${nh}, log-likelihood ${llh}; Normal distribution: ${ln}.`,
 sg: (k, p, e, a, o, x, eo) => `Beyond ${k} standard deviations, the Normal distribution gives probability ${p}, one day in ${e}. ${a}: ${o} days observed, ${x} expected (one day in ${eo} observed).`,
 sgAll: (o, x) => `All eight series: ${o} days observed, ${x} expected.`,
 agAll: 'full sample', agEx: 'without 20 February–30 April 2020',
 ag: (h, s, t) => `h = ${h} days, ${s}. Excess kurtosis: ${t}.`,
 qh: (a, h, n, s, k, p, k1) => `${a}, ${h}-day returns: n = ${n}, skewness ${s}, excess kurtosis ${k} (daily: ${k1}), Jarque–Bera p-value ${p}.`,
 ac: (s, r1, d, a1, a10, a50, b) => `${s[0].toUpperCase() + s.slice(1)}. ρ̂(1) of returns: ${r1}; 95% band: ±${b}. S&P 500, |r|^${d}: ${a1} at lag 1, ${a10} at lag 10, ${a50} at lag 50.`,
 vc0: 'Daily returns; a window above zero adds the band ± 2 rolling standard deviations.',
 vcItem0: (a, s, m, d) => `${a}: full-sample standard deviation ${s}%; worst day ${m}%, on ${d}`,
 vc: w => `Rolling ${w}-day standard deviation, largest and smallest (end date of the window):`,
 vcItem: (a, h, dh, l, dl, r) => `${a}: ${h}% (${dh}) and ${l}% (${dl}), ratio ${r}`,
 lv: (y, s, t, b) => `Sample from ${y}, ${s}. Mean L(1..5): ${t}; 95% band: ±${b}.`,
 gl: (a, b, n, t, o) => `The loss −(${a} quantile) exceeds the gain (${b} quantile) for ${n} of 8 series: ${t}. The gain is at least as large as the loss for: ${o}.`,
 gl2: (k, n, o) => `Beyond ${k} standard deviations: more down days than up days for ${n} of 8 series; not for: ${o}.`
};
""",
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (checked with a node script on ch2_data.js + common.js + ch2.js).
PREDICT = {
 'normal-pdf': (0, [('De circa 43 de ori: o zi din 15 787, față de una din 370.', 'About 43 times: one day in 15,787, against one in 370.'),
                    ('De circa 33 de ori: raportul densităților Normale în 3 și în 4.', 'About 33 times: the ratio of the Normal densities at 3 and at 4.'),
                    ('De circa 1,3 ori: probabilitatea scade proporțional cu pragul, 4/3.', 'About 1.3 times: the probability falls in proportion to the threshold, 4/3.')]),
 'lognormal': (1, [('Toate trei cresc: curba se deplasează spre dreapta.', 'All three rise: the curve shifts to the right.'),
                   ('Mediana rămîne 1, media crește, modul scade.', 'The median stays at 1, the mean rises, the mode falls.'),
                   ('Media și mediana cresc împreună, ca la distribuția Normală; modul scade.', 'The mean and the median rise together, as for the Normal distribution; the mode falls.')]),
 'clt': (2, [('Da: cu n de 10 ori mai mare, asimetria scade de √10 ori, la 0,09, oricare ar fi p.', 'Yes: with 10 times more draws, the skewness falls by √10, to 0.09, whatever p is.'),
             ('Da: pentru n peste 30 asimetria mediei este practic zero.', 'Yes: for n above 30 the skewness of the mean is practically zero.'),
             ('Nu: asimetria este 0,40, deoarece depinde și de p.', 'No: the skewness is 0.40, because it also depends on p.')]),
 'normal-approx': (0, [('Nu: la n = 200 eroarea este încă 0,012.', 'No: at n = 200 the error is still 0.012.'),
                       ('Da: de la np(1 − p) ≥ 9 aproximarea este exactă la două zecimale.', 'Yes: from np(1 − p) ≥ 9 the approximation is exact to two decimals.'),
                       ('Da: eroarea scade ca 1/n, deci la n = 200 este circa 0,004.', 'Yes: the error falls as 1/n, so at n = 200 it is about 0.004.')]),
 'shapes': (0, [('0,99: asimetria are o limită, circa 0,995.', '0.99: the skewness has a limit, about 0.995.'),
                ('3,4: asimetria crește proporțional cu α, 4 × 0,85.', '3.4: the skewness grows in proportion to α, 4 × 0.85.'),
                ('0,85: după standardizare, α nu mai schimbă forma.', '0.85: after standardisation, α no longer changes the shape.')]),
 'dax-density': (1, [('Rămîne 439: lățimea de bandă schimbă doar centrul densității.', 'It stays at 439: the bandwidth changes only the centre of the density.'),
                     ('Crește, la circa 580.', 'It rises, to about 580.'),
                     ('Scade la jumătate, circa 220: densitatea estimată este proporțională cu lățimea de bandă.', 'It halves, to about 220: the estimated density is proportional to the bandwidth.')]),
 'qq-normal': (0, [('Da: −6,0 abateri standard, față de cuantila Normală −3,4.', 'Yes: −6.0 standard deviations, against the Normal quantile −3.4.'),
                   ('Nu: cozile groase ale S&P 500 vin numai din martie 2020.', 'No: the heavy tails of the S&P 500 come only from March 2020.'),
                   ('Nu: fără martie 2020, abaterea standard scade, iar punctele ajung pe dreaptă.', 'No: without March 2020 the standard deviation falls and the points move onto the line.')]),
 't-densities': (0, [('Crește, pînă la circa 290 000 la k = 6.', 'It rises, to about 290,000 at k = 6.'),
                     ('Rămîne 56: cele două cozi scad în același ritm.', 'It stays at 56: both tails decay at the same rate.'),
                     ('Scade spre 1: la praguri mari ambele probabilități sînt practic zero.', 'It falls towards 1: at large thresholds both probabilities are practically zero.')]),
 't-fit': (0, [('ν = 2,2.', 'ν = 2.2.'),
               ('ν = 4: este mai aproape de distribuția Normală, care descrie bine centrul.', 'ν = 4: it is closer to the Normal distribution, which describes the centre well.'),
               ('La fel: log-verosimilitatea este simetrică în jurul lui ν̂ = 2,78.', 'The same: the log-likelihood is symmetric around ν̂ = 2.78.')]),
 'qq-t': (2, [('La ν = 30: un ν mai mare dă cozi mai groase.', 'At ν = 30: a larger ν gives heavier tails.'),
              ('Pentru niciun ν: minimul prevăzut depinde numai de scară.', 'For no ν: the predicted minimum depends only on the scale.'),
              ('Între ν = 3 și ν = 4.', 'Between ν = 3 and ν = 4.')]),
 'sigma-days': (1, [('Niciuna: la 6 abateri standard nici datele nu mai au zile.', 'None: at 6 standard deviations the data have no days either.'),
                    ('8 zile.', '8 days.'),
                    ('Practic zero, 0,0009 zile: de 105 ori numărul așteptat, ca la 4 abateri standard.', 'Practically zero, 0.0009 days: 105 times the expected number, as at 4 standard deviations.')]),
 'aggregation': (2, [('Circa 13,3: 50 de zile din 4 201 nu schimbă mult un moment estimat.', 'About 13.3: 50 days out of 4,201 do not change an estimated moment much.'),
                     ('Circa 0: fără criză randamentele zilnice sînt Normale.', 'About 0: without the crisis daily returns are Normal.'),
                     ('Circa 5,4.', 'About 5.4.')]),
 'qq-horizons': (0, [('Da: p-value 0,00004.', 'Yes: p-value 0.00004.'),
                     ('Nu: la orizontul trimestrial randamentele sînt Normale, ca la BET și Bitcoin.', 'No: at the quarterly horizon returns are Normal, as for BET and Bitcoin.'),
                     ('Nu: cu n = 66 testul nu poate respinge.', 'No: with n = 66 the test cannot reject.')]),
 'acf': (1, [('−0,11: 50 de zile nu schimbă o autocorelație estimată pe 4 201 de zile.', '−0.11: 50 days do not change an autocorrelation estimated on 4,201 days.'),
             ('−0,04, aproape de marginea benzii.', '−0.04, close to the edge of the band.'),
             ('Zero: fără criză randamentele sînt independente.', 'Zero: without the crisis returns are independent.')]),
 'clustering': (0, [('De circa 12 ori.', 'About 12 times.'),
                    ('De circa 154 de ori.', 'About 154 times.'),
                    ('De circa 1,5 ori: volatilitatea zilnică a unui indice este aproape constantă.', 'About 1.5 times: the daily volatility of an index is almost constant.')]),
 'leverage': (0, [('Da: −0,10, în afara benzii ±0,05.', 'Yes: −0.10, outside the ±0.05 band.'),
                  ('Nu: efectul de levier vine din criza din martie 2020.', 'No: the leverage effect comes from the March 2020 crisis.'),
                  ('Nu: cu mai puține zile, banda se lărgește și include media.', 'No: with fewer days the band widens and contains the mean.')]),
 'gain-loss': (1, [('Da: asimetria negativă face pierderea mai mare la orice α.', 'Yes: negative skewness makes the loss larger at every α.'),
                   ('Nu: la BET și Bitcoin cîștigul este puțin mai mare.', 'No: for BET and Bitcoin the gain is slightly larger.'),
                   ('Nu: la 5%, pierderea și cîștigul sînt egale la toți patru.', 'No: at 5%, loss and gain are equal for all four.')]),
}

# Exit ticket „Verifică-te” / "Check Yourself": questions of the slides „Autoevaluare” and „Formule de reținut”.
# Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Randamentele zilnice sînt N(0,05%, 1%²). La cîte abateri standard se află o zi de −3%?',
         'Daily returns are N(0.05%, 1%²). How many standard deviations away is a day of −3%?'),
      key=0, opts=[
  (('−3,05.', '−3.05.'), ('z = (−3 − 0,05)/1 = −3,05.', 'z = (−3 − 0.05)/1 = −3.05.')),
  (('−3,00.', '−3.00.'), ('Media de 0,05% nu se poate ignora: z = (x − μ)/σ.', 'The mean of 0.05% cannot be ignored: z = (x − μ)/σ.')),
  (('−2,95.', '−2.95.'), ('Media se scade, nu se adună: (−3 − 0,05)/1 = −3,05.', 'The mean is subtracted, not added: (−3 − 0.05)/1 = −3.05.'))]),
 dict(q=('Un eșantion are n = 2 500, asimetria S = −0,5 și excesul de boltire K = 6. Cît este statistica Jarque–Bera?',
         'A sample has n = 2,500, skewness S = −0.5 and excess kurtosis K = 6. What is the Jarque–Bera statistic?'),
      key=1, opts=[
  (('729.', '729.'), ('Rezultatul folosește K/4 în loc de K²/4. Corect: (2 500/6)(0,25 + 9) = 3 854.', 'This uses K/4 instead of K²/4. Correct: (2,500/6)(0.25 + 9) = 3,854.')),
  (('3 854; normalitatea este respinsă.', '3,854; normality is rejected.'),
   ('JB = (n/6)(S² + K²/4) = (2 500/6)(0,25 + 9) = 3 854, mult peste valoarea critică 5,99 a distribuției χ²(2) la 5%.',
    'JB = (n/6)(S² + K²/4) = (2,500/6)(0.25 + 9) = 3,854, far above the 5% critical value 5.99 of the χ²(2) distribution.')),
  (('8 542.', '8,542.'), ('Rezultatul folosește boltirea 9 în loc de excesul de boltire 6.', 'This uses the kurtosis 9 instead of the excess kurtosis 6.'))]),
 dict(q=('Un randament logaritmic anual este N(5%, 20%²). Care sînt mediana și media valorii unei investiții de 100 după un an?',
         'An annual log return is N(5%, 20%²). What are the median and the mean of the value of an investment of 100 after one year?'),
      key=2, opts=[
  (('Mediana și media sînt 105,1.', 'The median and the mean are both 105.1.'),
   ('e^x nu este liniară: media valorii este 100e^(0,05 + 0,2²/2) = 107,3, peste mediană.', 'e^x is not linear: the mean value is 100e^(0.05 + 0.2²/2) = 107.3, above the median.')),
  (('Mediana 105,1, media 109,4.', 'Median 105.1, mean 109.4.'),
   ('Termenul este s²/2 = 0,02, nu s² = 0,04: media este 100e^0,07 = 107,3.', 'The term is s²/2 = 0.02, not s² = 0.04: the mean is 100e^0.07 = 107.3.')),
  (('Mediana 105,1, media 107,3.', 'Median 105.1, mean 107.3.'),
   ('Mediana este 100e^0,05 = 105,1, iar media 100e^(0,05 + 0,02) = 107,3.', 'The median is 100e^0.05 = 105.1 and the mean 100e^(0.05 + 0.02) = 107.3.'))]),
 dict(q=('O distribuție Student-t are ν = 5 grade de libertate. Cît este excesul ei de boltire?',
         'A Student-t distribution has ν = 5 degrees of freedom. What is its excess kurtosis?'),
      key=0, opts=[
  (('6.', '6.'), ('Excesul de boltire este 6/(ν − 4) = 6/1 = 6.', 'The excess kurtosis is 6/(ν − 4) = 6/1 = 6.')),
  (('1,67.', '1.67.'), ('1,67 este varianța ν/(ν − 2) = 5/3, nu excesul de boltire.', '1.67 is the variance ν/(ν − 2) = 5/3, not the excess kurtosis.')),
  (('Infinit.', 'Infinite.'), ('Momentul de ordinul 4 există pentru ν > 4, deci și pentru ν = 5; excesul este infinit doar pentru ν ≤ 4.',
                              'The fourth moment exists for ν > 4, hence for ν = 5; the excess is infinite only for ν ≤ 4.'))]),
]


QA['ro'].update({
 'rare-moves': ('După distribuția Normală, 68,27% din zile se află la cel mult o abatere standard de medie. Ce pondere din zilele S&P 500, 2010–2026, se află la cel mult o abatere standard de medie?',
                '79,1%, mai mult decît după distribuția Normală. Cîteva zile extreme măresc abaterea standard, iar majoritatea zilelor sînt mici în raport cu ea: centrul distribuției este mai înalt. La cel mult două abateri standard ponderea este 95,3%, aproape de 95,45%; dincolo de trei abateri standard sînt 62 de zile în loc de 11,3. Pentru cele opt serii, ponderea la cel mult o abatere standard este între 75,7% (BRD) și 81,4% (BET).'),
 'large-losses': ('S&P 500 are 49 de zile sub −3%, față de 10,5 așteptate după distribuția Normală cu aceeași medie și aceeași abatere standard. La Bitcoin, cu același prag de −3%, sînt mai multe sau mai puține astfel de zile decît așteaptă distribuția Normală a seriei?',
                  'Mai puține: 512 zile, față de 813 așteptate. Abaterea standard a Bitcoin este 3,49%, deci −3% se află la numai 0,89 abateri standard sub medie, între centru și coadă, unde randamentele cu cozi groase au mai puține zile decît distribuția Normală. Cozile groase apar la praguri de mai multe abateri standard: sub −10% (2,90 abateri standard), Bitcoin are 56 de zile, față de 8,1 așteptate. Un prag exprimat în % compară seriile numai dacă volatilitățile lor sînt apropiate.'),
 'bet-100': ('Pentru 100 de lei investiți în BET, probabilitatea unei pierderi este 24,8% după un an și 1,5% după zece ani. Butonul ▶ crește orizontul pînă la 20 de ani. Care este probabilitatea unei pierderi după 20 de ani?',
             '0,11%: Φ(−μ√T/σ) = Φ(−0,682 × √20) = Φ(−3,05). Media randamentului logaritmic crește cu T, iar abaterea lui standard cu √T, deci raportul μ√T/σ crește cu √T. Mediana devine 1 009,6, iar media 1 345,6: distanța dintre ele crește cu orizontul. Calculul presupune că media randamentului logaritmic din 2010–2026, 11,6% pe an, se menține și că randamentele logaritmice sînt Normale și independente; pentru S&P 500, același calcul dă 0,15%, pentru Bitcoin 0,19%.'),
 'moments': ('Fără perioada 20 februarie–30 aprilie 2020, excesul de boltire al S&P 500 scade de la 13,5 la 5,4. Ce se întîmplă cu excesul de boltire al BET?',
             'Rămîne 18,0 (17,9 pe tot eșantionul). Cea mai slabă zi a BET, −11,9% pe 19 decembrie 2018 (OUG 114, taxa pe activele bancare), se află în afara acestei perioade. Boltirea de selecție depinde de cele cîteva zile extreme ale fiecărei serii, iar acestea diferă de la o piață la alta. Fără această perioadă, Bitcoin scade de la 12,0 la 5,4, iar DAX de la 7,4 la 3,1.'),
 'jb': ('Butonul ▶ mută începutul eșantionului pînă în 2024, cînd fiecare serie are doar între 600 și 1 000 de zile. Mai respinge testul Jarque–Bera normalitatea la 5% pentru toate cele opt serii?',
        'Da. Din 2024, cea mai mică valoare JB este 33, la SNP (n = 632, S = −0,03, K = 1,1), peste valoarea critică 5,99; S&P 500 are încă 6 318, cu K = 14,9, din zilele din aprilie 2025. Cu cîteva sute de observații, un exces de boltire de circa 1 dă deja o valoare JB mult peste 5,99. Pe 2010–2026, JB este între 3 174 (SNN) și 56 762 (BET).'),
 't-fit-table': ('Cu eșantionul din 2010, ν̂ este sub 4 pentru 7 din cele 8 serii. Butonul ▶ mută începutul eșantionului. Dacă eșantionul începe în 2021, pentru cîte serii este ν̂ sub 4?',
                 'Pentru 3: BET (3,76), Bitcoin (2,69) și SNN (3,03). S&P 500 trece de la 2,78 la 4,05, BRD de la 4,67 la 6,26. Fără 2020 cozile sînt mai subțiri, dar toate valorile ν̂ rămîn între 2,7 și 6,3, departe de distribuția Normală (ν → ∞), iar ΔAIC rămîne în favoarea distribuției t pentru fiecare serie (cel puțin 66, la SNP). ν̂ este o estimație care depinde de eșantion, mai ales de cele cîteva zile extreme pe care le conține.'),
 'sigma-table': ('La 4σ, S&P 500 are 28 de zile, față de 0,27 așteptate: p-value Poisson 2 × 10⁻⁴⁶. La 3σ are 62 de zile, față de 11,3 așteptate. Este p-value la 3σ mai mare sau mai mic decît la 4σ?',
                 'Mai mare: 1 × 10⁻²⁵. La 3σ se observă mai multe zile, dar raportul față de numărul așteptat este 5,5 în loc de 105, iar p-value depinde de cît de departe este numărul observat de λ, în raport cu dispersia distribuției Poisson. Ambele valori sînt mult sub orice prag uzual. La 5σ și 6σ, p-value pentru S&P 500 este 8 × 10⁻⁴¹ și 6 × 10⁻⁴⁶.'),
 'agg-table': ('La h = 63 de zile (trimestre, între 46 și 69 de randamente pe serie), pentru cîte dintre cele opt serii respinge testul Jarque–Bera normalitatea la 5%?',
               'Pentru 2: S&P 500 (p-value 0,00004) și DAX (0,002). Pentru celelalte șase, p-value este între 0,07 (BET) și 0,71 (SNG). La h = 21, testul respinge încă pentru 7 serii, toate în afară de BRD (0,16). Nerespingerea nu arată că randamentele trimestriale sînt Normale: cu mai puțin de 70 de observații, testul are putere mică.'),
 'ljung-box': ('Cu m = 10, testul Ljung–Box respinge absența autocorelației randamentelor la 5% pentru toate cele opt serii (Bitcoin: Q = 20,2, p-value 0,028). Fără perioada 20 februarie–30 aprilie 2020, pentru cîte serii mai respinge?',
               'Pentru 6: Bitcoin (p-value 0,053) și BRD (0,059) nu mai resping. La S&P 500, Q(10) scade de la 199 la 32. Distribuția χ² a statisticii Q presupune randamente i.i.d.; cu volatility clustering testul respinge prea des, deci „semnificativ” este aici o concluzie slabă. Pentru randamentele absolute, Q(10) rămîne între 325 și 1 712 la toate seriile.'),
 'facts-table': ('Faptul 5 se măsoară prin media L(1..5) a corelațiilor de levier; criteriul este o medie sub −1,96/√n. Pe tot eșantionul, pentru ce serii nu este îndeplinit criteriul?',
                 'Pentru SNP (−0,031, față de −0,032) și SNN (−0,025, față de −0,036). Bitcoin (−0,031, față de −0,030) și SNG (−0,039, față de −0,036) îl îndeplinesc la limită: cele patru valori, între −0,025 și −0,039, sînt toate aproape de prag, deci faptul 5 este la fel de slab pentru Bitcoin, SNP, SNN și SNG, în timp ce S&P 500, DAX și BET au între −0,10 și −0,12, iar BRD −0,05. Faptul 6 nu este îndeplinit doar la SNN, care are asimetrie pozitivă (0,27) și |q<sub>0,01</sub>|/q<sub>0,99</sub> = 0,88; SNP (1,04) și SNG (1,02) sînt aproape de 1.'),
})
QA['en'].update({
 'rare-moves': ('Under the Normal distribution, 68.27% of the days lie within one standard deviation of the mean. What share of S&P 500 days, 2010–2026, lies within one standard deviation of the mean?',
                '79.1%, more than under the Normal distribution. A few extreme days inflate the standard deviation, and most days are small relative to it: the centre of the distribution is taller. Within two standard deviations the share is 95.3%, close to 95.45%; beyond three standard deviations there are 62 days instead of 11.3. For the eight series, the share within one standard deviation is between 75.7% (BRD) and 81.4% (BET).'),
 'large-losses': ('The S&P 500 has 49 days below −3%, against 10.5 expected under the Normal distribution with the same mean and standard deviation. For Bitcoin, with the same threshold of −3%, are there more or fewer such days than the series’ Normal distribution expects?',
                  'Fewer: 512 days, against 813 expected. Bitcoin’s standard deviation is 3.49%, so −3% is only 0.89 standard deviations below the mean, between the centre and the tail, where heavy-tailed returns have fewer days than the Normal distribution. Heavy tails show at thresholds of several standard deviations: below −10% (2.90 standard deviations), Bitcoin has 56 days, against 8.1 expected. A threshold in % compares series only when their volatilities are similar.'),
 'bet-100': ('For 100 lei invested in the BET, the probability of a loss is 24.8% after one year and 1.5% after ten years. The ▶ button raises the horizon up to 20 years. What is the probability of a loss after 20 years?',
             '0.11%: Φ(−μ√T/σ) = Φ(−0.682 × √20) = Φ(−3.05). The mean of the log return grows with T and its standard deviation with √T, so the ratio μ√T/σ grows with √T. The median becomes 1,009.6 and the mean 1,345.6: the gap between them widens with the horizon. The calculation assumes that the mean log return of 2010–2026, 11.6% a year, persists and that log returns are Normal and independent; for the S&P 500 the same calculation gives 0.15%, for Bitcoin 0.19%.'),
 'moments': ('Without the period 20 February–30 April 2020, the excess kurtosis of the S&P 500 falls from 13.5 to 5.4. What happens to the excess kurtosis of the BET?',
             'It stays at 18.0 (17.9 on the full sample). The worst BET day, −11.9% on 19 December 2018 (OUG 114, a tax on bank assets), lies outside this period. The sample kurtosis depends on the few extreme days of each series, and these differ from market to market. Without this period, Bitcoin falls from 12.0 to 5.4 and the DAX from 7.4 to 3.1.'),
 'jb': ('The ▶ button moves the start of the sample up to 2024, when each series has only 600 to 1,000 days. Does the Jarque–Bera test still reject normality at 5% for all eight series?',
        'Yes. From 2024, the smallest JB is 33, for SNP (n = 632, S = −0.03, K = 1.1), above the critical value 5.99; the S&P 500 still has 6,318, with K = 14.9, from the days of April 2025. With several hundred observations, an excess kurtosis of about 1 already gives a JB far above 5.99. Over 2010–2026, JB is between 3,174 (SNN) and 56,762 (BET).'),
 't-fit-table': ('With the sample from 2010, ν̂ is below 4 for 7 of the 8 series. The ▶ button moves the start of the sample. If the sample starts in 2021, for how many series is ν̂ below 4?',
                 'For 3: BET (3.76), Bitcoin (2.69) and SNN (3.03). The S&P 500 moves from 2.78 to 4.05, BRD from 4.67 to 6.26. Without 2020 the tails are thinner, but every ν̂ stays between 2.7 and 6.3, far from the Normal distribution (ν → ∞), and ΔAIC still favours the t distribution for every series (at least 66, for SNP). ν̂ is an estimate that depends on the sample, above all on the few extreme days it contains.'),
 'sigma-table': ('At 4σ, the S&P 500 has 28 days, against 0.27 expected: Poisson p-value 2 × 10⁻⁴⁶. At 3σ it has 62 days, against 11.3 expected. Is the p-value at 3σ larger or smaller than at 4σ?',
                 'Larger: 1 × 10⁻²⁵. More days are observed at 3σ, but the ratio to the expected number is 5.5 instead of 105, and the p-value depends on how far the observed count lies from λ relative to the spread of the Poisson distribution. Both values are far below any usual level. At 5σ and 6σ, the S&P 500 p-values are 8 × 10⁻⁴¹ and 6 × 10⁻⁴⁶.'),
 'agg-table': ('At h = 63 days (quarters, 46 to 69 returns per series), for how many of the eight series does the Jarque–Bera test reject normality at 5%?',
               'For 2: S&P 500 (p-value 0.00004) and DAX (0.002). For the other six, the p-value is between 0.07 (BET) and 0.71 (SNG). At h = 21 the test still rejects for 7 series, all except BRD (0.16). Not rejecting does not show that quarterly returns are Normal: with fewer than 70 observations, the test has little power.'),
 'ljung-box': ('With m = 10, the Ljung–Box test rejects the absence of autocorrelation of returns at 5% for all eight series (Bitcoin: Q = 20.2, p-value 0.028). Without the period 20 February–30 April 2020, for how many series does it still reject?',
               'For 6: Bitcoin (p-value 0.053) and BRD (0.059) no longer reject. For the S&P 500, Q(10) falls from 199 to 32. The χ² distribution of Q assumes i.i.d. returns; with volatility clustering the test rejects too often, so “significant” is a weak conclusion here. For absolute returns, Q(10) stays between 325 and 1,712 for every series.'),
 'facts-table': ('Fact 5 is measured by the mean L(1..5) of the leverage correlations; the criterion is a mean below −1.96/√n. On the full sample, for which series is the criterion not met?',
                 'For SNP (−0.031, against −0.032) and SNN (−0.025, against −0.036). Bitcoin (−0.031, against −0.030) and SNG (−0.039, against −0.036) meet it by a small margin: the four values, between −0.025 and −0.039, are all close to the limit, so fact 5 is equally weak for Bitcoin, SNP, SNN and SNG, while the S&P 500, DAX and BET have −0.10 to −0.12 and BRD −0.05. Fact 6 is not met only for SNN, which has positive skewness (0.27) and |q<sub>0.01</sub>|/q<sub>0.99</sub> = 0.88; SNP (1.04) and SNG (1.02) are close to 1.'),
})

PREDICT.update({
 'rare-moves': (0, [('Circa 79%: mai mult decît 68%.', 'About 79%: more than 68%.'),
                    ('Circa 60%: cozile groase iau probabilitate din centrul distribuției.', 'About 60%: heavy tails take probability away from the centre.'),
                    ('68%: regula 68–95–99,7 este valabilă pentru orice distribuție cu aceeași abatere standard.', '68%: the 68–95–99.7 rule holds for any distribution with the same standard deviation.')]),
 'large-losses': (1, [('Mai multe: cozile groase înseamnă mai multe pierderi mari la orice prag.', 'More: heavy tails mean more large losses at any threshold.'),
                      ('Mai puține: 512 față de 813.', 'Fewer: 512 against 813.'),
                      ('Aproximativ la fel: abaterea standard mai mare a Bitcoin compensează exact cozile groase.', 'About the same: Bitcoin’s larger standard deviation exactly offsets the heavy tails.')]),
 'bet-100': (0, [('0,11%.', '0.11%.'),
                 ('Circa 0,75%: probabilitatea se înjumătățește cînd orizontul se dublează.', 'About 0.75%: the probability halves when the horizon doubles.'),
                 ('24,8%: probabilitatea unei pierderi nu depinde de orizont.', '24.8%: the probability of a loss does not depend on the horizon.')]),
 'moments': (1, [('Scade la circa 5, ca la S&P 500: martie 2020 explică cozile groase pe toate piețele.', 'It falls to about 5, as for the S&P 500: March 2020 explains the heavy tails in every market.'),
                 ('Rămîne circa 18.', 'It stays at about 18.'),
                 ('Scade spre 0: fără criză, randamentele BET sînt Normale.', 'It falls towards 0: without the crisis, BET returns are Normal.')]),
 'jb': (0, [('Da: cea mai mică valoare este 33, la SNP.', 'Yes: the smallest value is 33, for SNP.'),
            ('Nu: cu mai puțin de 1 000 de observații testul nu poate respinge.', 'No: with fewer than 1,000 observations the test cannot reject.'),
            ('Nu: cozile groase din 2010–2026 vin din 2020, care nu mai este în eșantion.', 'No: the heavy tails of 2010–2026 come from 2020, which is no longer in the sample.')]),
 't-fit-table': (0, [('Pentru 3: BET, Bitcoin și SNN.', 'For 3: BET, Bitcoin and SNN.'),
                     ('Tot pentru 7: ν este o proprietate a pieței, nu a eșantionului.', 'Still for 7: ν is a property of the market, not of the sample.'),
                     ('Pentru niciuna: fără 2020, toate randamentele devin Normale.', 'For none: without 2020, all returns become Normal.')]),
 'sigma-table': (0, [('Mai mare: 1 × 10⁻²⁵.', 'Larger: 1 × 10⁻²⁵.'),
                     ('Mai mic: mai multe zile observate înseamnă o dovadă mai puternică.', 'Smaller: more observed days mean stronger evidence.'),
                     ('Același: ambele sînt 0 la orice precizie de calcul.', 'The same: both are 0 at any computing precision.')]),
 'agg-table': (1, [('Pentru toate 8: cozile groase zilnice rămîn la orice orizont.', 'For all 8: daily heavy tails remain at every horizon.'),
                   ('Pentru 2: S&P 500 și DAX.', 'For 2: S&P 500 and DAX.'),
                   ('Pentru niciuna: după CLT, sumele pe un trimestru sînt Normale.', 'For none: by the CLT, quarterly sums are Normal.')]),
 'ljung-box': (0, [('Pentru 6.', 'For 6.'),
                   ('Tot pentru 8: 50 de zile din peste 2 900 nu schimbă statistica Q.', 'Still for 8: 50 days out of more than 2,900 do not change the Q statistic.'),
                   ('Pentru niciuna: fără criză, randamentele sînt independente.', 'For none: without the crisis, returns are independent.')]),
 'facts-table': (1, [('Pentru toate cele patru acțiuni BVB: efectul de levier apare doar pe piețele mari.', 'For all four BVB stocks: the leverage effect appears only in large markets.'),
                     ('Pentru SNP și SNN.', 'For SNP and SNN.'),
                     ('Pentru niciuna: toate mediile L(1..5) sînt negative.', 'For none: every mean L(1..5) is negative.')]),
})

# blocks whose initial table shows the answer: veil the readout and those elements
VEIL = {'rare-moves': {'out': True, 'charts': ['rm-tab2']}, 'large-losses': {'out': True, 'charts': ['ll-tab']},
        'facts-table': {'out': True, 'charts': ['ft-tab']}}

TEXT_JS['ro'] += r"""function ch2De(s) { s = String(s); if (/,/.test(s)) return s; const n = +s.replace(/\D/g, ''); return n >= 20 && (n % 100 === 0 || n % 100 >= 20) ? s + ' de' : s; }
Object.assign(window.CH2_TEXT, {
 h: { series: 'Seria', from: 'din', mean: 'Media %', sd: 'Abaterea std. %', skew: 'Asimetria', exk: 'Excesul de boltire', min: 'Min % (data)',
      everyD: 'o dată la … zile de tranzacționare', everyY: 'o dată la … ani', obsDays: 'zile dincolo de k abateri standard', within: j => `la cel mult ${j} ab. std.`,
      normal: 'Distribuția Normală', normalN: n => `Distribuția Normală, n = ${n}`, expD: 'zile așteptate', obsD: 'zile observate', ratio: 'observate / așteptate',
      T: 'T (ani)', median: 'mediana', meanV: 'media', ploss: 'P(pierdere)', quant: 'cuantila', loc: 'locația %', scale: 'scala %', llt: 'log L (t)', lln: 'log L (Normală)',
      sdt: 'abaterea std. a distribuției t %', sds: 'abaterea std. de selecție %' },
 rm: (a, d, n, w1, w2, c3, e3, c4, e4) => `${a} din ${d}, n = ${n}: ${w1} din zile la cel mult o abatere standard de medie (distribuția Normală: 68,3%), ${w2} la cel mult două (95,4%); ${ch2De(c3)} zile dincolo de 3 abateri standard (${e3} așteptate), ${c4} dincolo de 4 (${e4} așteptate).`,
 ll: (x, more, less) => `Zile sub −${x}%: mai multe decît așteaptă distribuția Normală la ${more}; mai puține sau tot atîtea la ${less}.`,
 llw1: (a, d, n, m, s) => `${a} din ${d}: n = ${ch2De(n)} randamente logaritmice zilnice, media r̄ = ${m}%, abaterea standard s = ${s}%.`,
 llw2: (x, mt, s, z) => `O pierdere mai mare decît ${x}%: z = (−${x} ${mt})/${s} = ${z}.`,
 llw3: (z, P, n, E) => `Distribuția Normală: Φ(${z}) = ${P}; zile așteptate: n × Φ(z) = ${n} × Φ(${z}) = ${E}.`,
 llw4: (o, x, cmp) => `Observate: ${ch2De(o)} zile sub −${x}%, ${cmp}.`,
 llMore: r => `de ${r} ori numărul așteptat`,
 llLess: (r, z) => `numai ${r} din numărul așteptat: pragul se află la ${z} abateri standard sub medie, între centru și coadă, unde randamentele cu cozi groase au mai puține zile decît distribuția Normală`,
 b1w1: (a, d, m, s, A, mu, sg) => `${a} din ${d}: media zilnică ${m}%, abaterea standard zilnică ${s}%, A = ${A} randamente pe an: μ = A × ${m}% = ${mu}%, σ = √A × ${s}% = ${sg}%.`,
 b1w2: (T, mT, vT, med, mn) => `T = ${T}: Tμ = ${mT}, Tσ² = ${vT}; mediana 100e<sup>Tμ</sup> = ${med}; media 100e<sup>Tμ + Tσ²/2</sup> = ${mn}.`,
 b1w3: (z, p) => `Probabilitatea unei pierderi: Φ(−μ√T/σ) = Φ(${z}) = ${p}.`,
 b1w4: (qa, za, mT, sT, q) => `Cuantila ${qa}: z<sub>α</sub> = ${za}; 100e<sup>${mT} + (${za}) × ${sT}</sup> = ${q}.`,
 b1: (a, T, med, mn, p, qa, q) => `${a}, 100 investiți, după ${T === 1 ? '1 an' : T >= 20 ? T + ' de ani' : T + ' ani'}: mediana ${med}, media ${mn}, probabilitatea unei pierderi ${p}, cuantila ${qa} ${q}.`,
 mow1: (a, d, n, m) => `${a} din ${d}: n = ${ch2De(n)} randamente logaritmice zilnice, media r̄ = ${m}%.`,
 mow2: (m2, m3, m4) => `Momentele centrale: m₂ = ${m2} (%²), m₃ = ${m3} (%³), m₄ = ${m4} (%⁴).`,
 mow3: (m3, m2, S, m4, K) => `S = ${m3}/${m2}<sup>3/2</sup> = ${S}; K = ${m4}/${m2}² − 3 = ${K}.`,
 mow4: (n, se1, se2, zS, zK) => `Dacă datele ar fi Normale, SE(S) = √(6/${n}) = ${se1} și SE(K) = √(24/${n}) = ${se2}: S se află la ${zS}, iar K la ${zK} erori standard de 0. Aceste erori standard sînt valabile numai pentru date Normale.`,
 mo: (y, s, lk, lv, hk, hv, nneg, pos) => `Eșantioane din ${y}, ${s}. Excesul de boltire între ${lv} (${lk}) și ${hv} (${hk}); asimetrie negativă la ${nneg} din 8 serii; zero sau pozitivă la: ${pos}.`,
 jbw1: (a, d, n, S, K) => `${a} din ${d}: n = ${n}, S = ${S}, K = ${K}.`,
 jbw2: (n, a6, S, S2, K, K4) => `n/6 = ${n}/6 = ${a6}; S² = (${S})² = ${S2}; K²/4 = ${K}²/4 = ${K4}.`,
 jbw3: (a6, S2, K4, jb, sh) => `JB = ${a6} × (${S2} + ${K4}) = ${jb}; termenul boltirii dă ${sh} din valoare.`,
 jbw4: (jb, p) => `Valoarea critică a distribuției χ²(2) la 5%: 5,99. JB = ${jb} > 5,99: normalitatea este respinsă; p-value e<sup>−JB/2</sup> = ${p}.`,
 jbw4n: (jb, p) => `Valoarea critică a distribuției χ²(2) la 5%: 5,99. JB = ${jb} ≤ 5,99: testul nu respinge normalitatea la 5%; p-value e<sup>−JB/2</sup> = ${p}.`,
 jb: (y, lo, jbv, nrej) => `Eșantioane din ${y}: testul JB respinge normalitatea la 5% pentru ${nrej} din 8 serii; cea mai mică valoare JB: ${jbv} (${lo}).`,
 ttNote: 'ν̂ cu roșu: sub 4, distribuția t estimată are boltire infinită. ΔAIC = AIC(Normală) − AIC(t). Abaterea standard a distribuției t: s√(ν/(ν − 2)).',
 ttw1: (a, y, nu, loc, s) => `${a}, eșantion din ${y}: ν̂ = ${nu}, locația ${loc}%, scala s = ${s}%.`,
 ttw2: (nu, nu2, r, s, v, sd, ss) => `ν/(ν − 2) = ${nu}/${nu2} = ${r}; varianța distribuției t estimate: ${s}² × ${r} = ${v} (%²), abaterea standard ${sd}%, față de abaterea standard de selecție ${ss}%.`,
 ttw2inf: nu => `ν̂ = ${nu} ≤ 2: distribuția t estimată are varianță infinită.`,
 ttw3: (k, ss, c, s, cs, nu, p, n, e) => `O variație de peste k = ${k} abateri standard de selecție: c = ${k} × ${ss} = ${c}%, c/s = ${cs}; P(|X − μ| > c) = 2F<sub>t(${nu})</sub>(−${cs}) = ${p}, adică ${e} din cele ${ch2De(n)} zile.`,
 ttw4: (k, p, e, o) => `Distribuția Normală: P(|Z| > ${k}) = ${p}, adică ${ch2De(e)} zile. Observate: ${ch2De(o)} zile.`,
 tt: (y, n4, lo, lov, hi, hiv, daic) => `Eșantioane din ${y}: ν̂ sub 4 pentru ${n4} din 8 serii; ν̂ între ${lov} (${lo}) și ${hiv} (${hi}); cel mai mic ΔAIC: ${daic}.`,
 stw1: (k, p, ev) => `p<sub>${k}</sub> = 2(1 − Φ(${k})) = ${p}: o zi din ${ev}.`,
 stw2: (a, n, p, lam) => `${a}, n = ${n}: λ = n × p = ${n} × ${p} = ${ch2De(lam)} zile așteptate.`,
 stw3: (o, k, d, u, r) => `Observate: ${ch2De(o)} zile cu |r − r̄| > ${k}s (${d} sub medie, ${u} peste), de ${r} ori λ.`,
 stw4: (o, lam, first, pv) => `P(N ≥ ${o}) pentru N ~ Poisson(${lam}): primul termen, e<sup>−λ</sup>λ<sup>${o}</sup>/${o}! = ${first}, dă aproape toată valoarea; p-value ${pv}.`,
 stw4b: (o, lam, pv) => `P(N ≥ ${o}) pentru N ~ Poisson(${lam}) = ${pv}.`,
 st: (y, k, list, n, e) => `Eșantioane din ${y}. Zile dincolo de ${k} abateri standard: ${list}; distribuția Normală așteaptă ${e} pentru n = ${n}.`,
 ahNote: 'Cu verde: p-value de cel puțin 0,05, testul Jarque–Bera nu respinge normalitatea la 5%.',
 ah: (s, list, n1, n21, n63) => `${s[0].toUpperCase() + s.slice(1)}. Numărul de serii pentru care testul JB respinge normalitatea la 5%, după orizont: ${list}. S&P 500: n = ${n1} la h = 1, ${n21} la h = 21 (lunar) și ${n63} la h = 63 (trimestrial).`,
 lbNote: (m, cv) => `Cu verde: p-value de cel puțin 0,05. Valoarea critică a distribuției χ²(${m}) la 5%: ${cv}.`,
 lb: (m, y, s, nrej, list, cv, nabs) => `m = ${m}, eșantioane din ${y}, ${s}. Q(${m}) al randamentelor depășește ${cv} pentru ${nrej} din 8 serii; nu și pentru: ${list}. Q(${m}) al randamentelor absolute respinge pentru ${nabs} din 8.`,
 ftNote: 'Cu roșu: criteriul faptului nu este îndeplinit. 1: exces de boltire peste 1 și p-value JB sub 5%; 2: boltirea la h zile sub jumătate din cea zilnică; 3: media |ρ̂<sub>r</sub>(1..10)| sub 0,05; 4: ρ̂<sub>|r|</sub>(10) peste 3 × 1,96/√n; 5: L̄(1..5) sub −1,96/√n; 6: asimetrie negativă și |q<sub>0,01</sub>| > q<sub>0,99</sub>.',
 ft: (y, s, h) => `Eșantioane din ${y}, ${s}; faptul 2 la h = ${h} zile.`,
 ft2: (f1, f2, f3, f4, f5, f6) => `Criteriul nu este îndeplinit — faptul 1: ${f1}; faptul 2: ${f2}; faptul 3: ${f3}; faptul 4: ${f4}; faptul 5: ${f5}; faptul 6: ${f6}.`,
 clw1: (n, p, mu, sd) => `S<sub>n</sub>, numărul de succese din n = ${n} extrageri Bernoulli(${p}): media np = ${mu}, abaterea standard √(np(1 − p)) = ${sd}.`,
 clw2: (mu, sd, thr, k0) => `Z<sub>n</sub> > 1,96 înseamnă S<sub>n</sub> > ${mu} + 1,96 × ${sd} = ${thr}, adică S<sub>n</sub> ≥ ${k0}.`,
 clw3: (k0, n, p, ex, nt) => `P(S<sub>n</sub> ≥ ${k0}) = Σ<sub>j = ${k0}..${n}</sub> C(${n}, j) · ${p}<sup>j</sup> · (1 − ${p})<sup>${n} − j</sup> = ${ex} (valoarea exactă), față de ${nt} pentru N(0, 1).`,
 clw4: (n, sim, ex, se) => `Cele 20 000 de simulări ale slide-ului dau ${sim} pentru n = ${n}, valoarea exactă este ${ex}; eroarea standard a simulării este circa √(p̂(1 − p̂)/20 000) = ${se} puncte procentuale.`
});
"""
TEXT_JS['en'] += r"""Object.assign(window.CH2_TEXT, {
 h: { series: 'Series', from: 'from', mean: 'Mean %', sd: 'S.d. %', skew: 'Skew.', exk: 'Exc. kurt.', min: 'Min % (date)',
      everyD: 'once every … trading days', everyY: 'once every … years', obsDays: 'days beyond k sd', within: j => `within ${j} sd`,
      normal: 'Normal distribution', normalN: n => `Normal distribution, n = ${n}`, expD: 'expected days', obsD: 'observed days', ratio: 'observed / expected',
      T: 'T (years)', median: 'median', meanV: 'mean', ploss: 'P(loss)', quant: 'quantile', loc: 'location %', scale: 'scale %', llt: 'log L (t)', lln: 'log L (Normal)',
      sdt: 'sd of the t %', sds: 'sample sd %' },
 rm: (a, d, n, w1, w2, c3, e3, c4, e4) => `${a} from ${d}, n = ${n}: ${w1} of the days within one standard deviation of the mean (Normal distribution: 68.3%), ${w2} within two (95.4%); ${c3} days beyond 3 standard deviations (${e3} expected), ${c4} beyond 4 (${e4} expected).`,
 ll: (x, more, less) => `Days below −${x}%: more than the Normal distribution expects for ${more}; fewer or as many for ${less}.`,
 llw1: (a, d, n, m, s) => `${a} from ${d}: n = ${n} daily log returns, mean r̄ = ${m}%, standard deviation s = ${s}%.`,
 llw2: (x, mt, s, z) => `A loss larger than ${x}%: z = (−${x} ${mt})/${s} = ${z}.`,
 llw3: (z, P, n, E) => `Normal distribution: Φ(${z}) = ${P}; expected days: n × Φ(z) = ${n} × Φ(${z}) = ${E}.`,
 llw4: (o, x, cmp) => `Observed: ${o} days below −${x}%, ${cmp}.`,
 llMore: r => `${r} times the expected number`,
 llLess: (r, z) => `only ${r} of the expected number: the threshold lies ${z} standard deviations below the mean, between the centre and the tail, where heavy-tailed returns have fewer days than the Normal distribution`,
 b1w1: (a, d, m, s, A, mu, sg) => `${a} from ${d}: daily mean ${m}%, daily standard deviation ${s}%, A = ${A} returns a year: μ = A × ${m}% = ${mu}%, σ = √A × ${s}% = ${sg}%.`,
 b1w2: (T, mT, vT, med, mn) => `T = ${T}: Tμ = ${mT}, Tσ² = ${vT}; median 100e<sup>Tμ</sup> = ${med}; mean 100e<sup>Tμ + Tσ²/2</sup> = ${mn}.`,
 b1w3: (z, p) => `Probability of a loss: Φ(−μ√T/σ) = Φ(${z}) = ${p}.`,
 b1w4: (qa, za, mT, sT, q) => `${qa} quantile: z<sub>α</sub> = ${za}; 100e<sup>${mT} + (${za}) × ${sT}</sup> = ${q}.`,
 b1: (a, T, med, mn, p, qa, q) => `${a}, 100 invested, after ${T} year${T === 1 ? '' : 's'}: median ${med}, mean ${mn}, probability of a loss ${p}, ${qa} quantile ${q}.`,
 mow1: (a, d, n, m) => `${a} from ${d}: n = ${n} daily log returns, mean r̄ = ${m}%.`,
 mow2: (m2, m3, m4) => `Central moments: m₂ = ${m2} (%²), m₃ = ${m3} (%³), m₄ = ${m4} (%⁴).`,
 mow3: (m3, m2, S, m4, K) => `S = ${m3}/${m2}<sup>3/2</sup> = ${S}; K = ${m4}/${m2}² − 3 = ${K}.`,
 mow4: (n, se1, se2, zS, zK) => `If the data were Normal, SE(S) = √(6/${n}) = ${se1} and SE(K) = √(24/${n}) = ${se2}: S lies ${zS} and K ${zK} standard errors from 0. These standard errors hold only for Normal data.`,
 mo: (y, s, lk, lv, hk, hv, nneg, pos) => `Samples from ${y}, ${s}. Excess kurtosis between ${lv} (${lk}) and ${hv} (${hk}); negative skewness for ${nneg} of 8 series; zero or positive for: ${pos}.`,
 jbw1: (a, d, n, S, K) => `${a} from ${d}: n = ${n}, S = ${S}, K = ${K}.`,
 jbw2: (n, a6, S, S2, K, K4) => `n/6 = ${n}/6 = ${a6}; S² = (${S})² = ${S2}; K²/4 = ${K}²/4 = ${K4}.`,
 jbw3: (a6, S2, K4, jb, sh) => `JB = ${a6} × (${S2} + ${K4}) = ${jb}; the kurtosis term gives ${sh} of it.`,
 jbw4: (jb, p) => `5% critical value of the χ²(2) distribution: 5.99. JB = ${jb} > 5.99: normality is rejected; p-value e<sup>−JB/2</sup> = ${p}.`,
 jbw4n: (jb, p) => `5% critical value of the χ²(2) distribution: 5.99. JB = ${jb} ≤ 5.99: the test does not reject normality at 5%; p-value e<sup>−JB/2</sup> = ${p}.`,
 jb: (y, lo, jbv, nrej) => `Samples from ${y}: the JB test rejects normality at 5% for ${nrej} of 8 series; smallest JB: ${jbv} (${lo}).`,
 ttNote: 'ν̂ in red: below 4, the fitted t has infinite kurtosis. ΔAIC = AIC(Normal) − AIC(t). Standard deviation of the t: s√(ν/(ν − 2)).',
 ttw1: (a, y, nu, loc, s) => `${a}, sample from ${y}: ν̂ = ${nu}, location ${loc}%, scale s = ${s}%.`,
 ttw2: (nu, nu2, r, s, v, sd, ss) => `ν/(ν − 2) = ${nu}/${nu2} = ${r}; variance of the fitted t: ${s}² × ${r} = ${v} (%²), standard deviation ${sd}%, against the sample standard deviation ${ss}%.`,
 ttw2inf: nu => `ν̂ = ${nu} ≤ 2: the fitted t has infinite variance.`,
 ttw3: (k, ss, c, s, cs, nu, p, n, e) => `A move of more than k = ${k} sample standard deviations: c = ${k} × ${ss} = ${c}%, c/s = ${cs}; P(|X − μ| > c) = 2F<sub>t(${nu})</sub>(−${cs}) = ${p}, that is ${e} of the ${n} days.`,
 ttw4: (k, p, e, o) => `Normal distribution: P(|Z| > ${k}) = ${p}, that is ${e} days. Observed: ${o} days.`,
 tt: (y, n4, lo, lov, hi, hiv, daic) => `Samples from ${y}: ν̂ below 4 for ${n4} of 8 series; ν̂ between ${lov} (${lo}) and ${hiv} (${hi}); smallest ΔAIC: ${daic}.`,
 stw1: (k, p, ev) => `p<sub>${k}</sub> = 2(1 − Φ(${k})) = ${p}: one day in ${ev}.`,
 stw2: (a, n, p, lam) => `${a}, n = ${n}: λ = n × p = ${n} × ${p} = ${lam} days expected.`,
 stw3: (o, k, d, u, r) => `Observed: ${o} days with |r − r̄| > ${k}s (${d} below the mean, ${u} above), ${r} times λ.`,
 stw4: (o, lam, first, pv) => `P(N ≥ ${o}) for N ~ Poisson(${lam}): the first term, e<sup>−λ</sup>λ<sup>${o}</sup>/${o}! = ${first}, gives almost all of it; p-value ${pv}.`,
 stw4b: (o, lam, pv) => `P(N ≥ ${o}) for N ~ Poisson(${lam}) = ${pv}.`,
 st: (y, k, list, n, e) => `Samples from ${y}. Days beyond ${k} standard deviations: ${list}; the Normal distribution expects ${e} for n = ${n}.`,
 ahNote: 'In green: p-value of 0.05 or more, the Jarque–Bera test does not reject normality at 5%.',
 ah: (s, list, n1, n21, n63) => `${s[0].toUpperCase() + s.slice(1)}. Number of series for which the JB test rejects normality at 5%, by horizon: ${list}. S&P 500: ${n1} daily, ${n21} monthly (h = 21) and ${n63} quarterly (h = 63) returns.`,
 lbNote: (m, cv) => `In green: p-value of 0.05 or more. 5% critical value of the χ²(${m}) distribution: ${cv}.`,
 lb: (m, y, s, nrej, list, cv, nabs) => `m = ${m}, samples from ${y}, ${s}. Q(${m}) of returns exceeds ${cv} for ${nrej} of 8 series; not for: ${list}. Q(${m}) of absolute returns rejects for ${nabs} of 8.`,
 ftNote: 'In red: the criterion of the fact is not met. 1: excess kurtosis above 1 and JB p-value below 5%; 2: kurtosis at h days below half the daily one; 3: mean |ρ̂<sub>r</sub>(1..10)| below 0.05; 4: ρ̂<sub>|r|</sub>(10) above 3 × 1.96/√n; 5: L̄(1..5) below −1.96/√n; 6: negative skewness and |q<sub>0.01</sub>| > q<sub>0.99</sub>.',
 ft: (y, s, h) => `Samples from ${y}, ${s}; fact 2 at h = ${h} days.`,
 ft2: (f1, f2, f3, f4, f5, f6) => `Criterion not met — fact 1: ${f1}; fact 2: ${f2}; fact 3: ${f3}; fact 4: ${f4}; fact 5: ${f5}; fact 6: ${f6}.`,
 clw1: (n, p, mu, sd) => `S<sub>n</sub>, the number of successes in n = ${n} Bernoulli(${p}) draws: mean np = ${mu}, standard deviation √(np(1 − p)) = ${sd}.`,
 clw2: (mu, sd, thr, k0) => `Z<sub>n</sub> > 1.96 means S<sub>n</sub> > ${mu} + 1.96 × ${sd} = ${thr}, that is S<sub>n</sub> ≥ ${k0}.`,
 clw3: (k0, n, p, ex, nt) => `P(S<sub>n</sub> ≥ ${k0}) = Σ<sub>j = ${k0}..${n}</sub> C(${n}, j) · ${p}<sup>j</sup> · (1 − ${p})<sup>${n} − j</sup> = ${ex} (exact value), against ${nt} for N(0, 1).`,
 clw4: (n, sim, ex, se) => `The 20,000 simulations of the slide give ${sim} for n = ${n}, the exact value is ${ex}; the simulation standard error is about √(p̂(1 − p̂)/20,000) = ${se} percentage points.`
});
"""

def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch2-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH2_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH2_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH2_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    import re
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', '\u202f', t)
    for a, (q, ans) in QA['ro'].items():
        QA['ro'][a] = (fix(q), fix(ans))
    for a, (k, opts) in PREDICT.items():
        PREDICT[a] = (k, [(fix(r), e) for r, e in opts])
    for t in TICKET:
        t['q'] = (fix(t['q'][0]), t['q'][1])
        t['opts'] = [((fix(o[0]), o[1]), (fix(w[0]), w[1])) for o, w in t['opts']]



def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    res = {}
    for a, fr, _, _, ken, kro in XBLOCKS:
        i = fr - 1 + (lang == 'ro')
        title = latex_text(frames[i][0])
        assert (kro if lang == 'ro' else ken) in title and 'includegraphics' not in frames[i][1], (lang, a, title)
        res[a] = (int(pages[i][0]), title)
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved to their frames by ch2.js)."""
    W = widgets_all(lang)
    fp = frame_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''.join(xblock_html(x, lang, True, fp[x][0]) for x, _, _, host, _, _ in XBLOCKS if host == a)
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
    secs = {a: s for a, _, s in BLOCKS} | {a: s for a, _, s, _, _, _ in XBLOCKS}
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch2_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    assert set(blocks) <= set(ORDER), set(blocks) - set(ORDER)
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch2_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


_ro_numbers()
_predict_js()


if __name__ == '__main__':
    build()
