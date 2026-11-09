"""Build the interactive pages of Chapter 16: interactive/ch16.html (EN) and interactive/ch16_ro.html (RO).

One block per chart slide of the lecture body, one block per results table, extra chart and review problem (XBLOCKS), in the
order of the slides: the slide on the left, its interactive version on the right, with live tables and worked review problems.
Logic in interactive/ch16.js, inputs in interactive/ch16_data.js (Quantlets/Ch_16/export_interactive.py), slide images in
interactive/slides/. On the web lecture (build_web_lecture.py) the table and problem blocks travel inside a chart block
(widgets) and ch16.js moves each one to its own frame. The EN deck has one acronym frame less than the RO deck; XBLOCKS give
the RO frame number, checked against TITLES and TITLES_EN. The exam logistics frames have no block.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 16
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_16'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('three', 'sfm_ch16_three_series', 'map'), ('stylised', 'sfm_ch16_stylised', 'map'), ('backtest', 'sfm_ch16_backtest', 'c811')]
# table, extra-chart and review-problem blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page)
XBLOCKS = [('sum-tab', 11, 'map', 'three'), ('interp', 12, 'map', 'three'), ('perf', 17, 'c03', 'stylised'), ('agg', 21, 'c03', 'stylised'),
           ('p1', 56, 'exam', 'backtest'), ('p2', 58, 'exam', 'backtest'), ('p3', 60, 'exam', 'backtest'), ('p4', 62, 'exam', 'backtest'),
           ('p5', 64, 'exam', 'backtest'), ('p6', 66, 'exam', 'backtest'), ('p7', 68, 'exam', 'backtest'), ('p8', 70, 'exam', 'backtest')]
ORDER = ['three', 'sum-tab', 'interp', 'stylised', 'perf', 'agg', 'backtest', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
TITLES = {'sum-tab': 'Tabelul de sinteză al cursului', 'interp': 'Interpretarea tabelului', 'perf': 'Capitolul 1: date, randamente și indicatori (2/2)',
          'agg': 'Capitolul 3: distribuții α-stabile (2/2)', 'p1': 'Problema 1: rezolvare', 'p2': 'Problema 2: rezolvare', 'p3': 'Problema 3: rezolvare',
          'p4': 'Problema 4: rezolvare', 'p5': 'Problema 5: rezolvare', 'p6': 'Problema 6: rezolvare', 'p7': 'Problema 7: rezolvare', 'p8': 'Problema 8: rezolvare'}
TITLES_EN = {'sum-tab': 'Summary Table of the Course', 'interp': 'Interpreting the Table', 'perf': 'Chapter 1: Data, Returns and Indicators (2/2)',
             'agg': 'Chapter 3: α-Stable Distributions (2/2)', 'p1': 'Problem 1: Solution', 'p2': 'Problem 2: Solution', 'p3': 'Problem 3: Solution',
             'p4': 'Problem 4: Solution', 'p5': 'Problem 5: Solution', 'p6': 'Problem 6: Solution', 'p7': 'Problem 7: Solution', 'p8': 'Problem 8: Solution'}

PFX = {'three': 'ts', 'sum-tab': 'sm', 'interp': 'ip', 'stylised': 'sf', 'perf': 'pf', 'agg': 'ag', 'backtest': 'bt',
       'p1': 'q1', 'p2': 'q2', 'p3': 'q3', 'p4': 'q4', 'p5': 'q5', 'p6': 'q6', 'p7': 'q7', 'p8': 'q8'}

SECTIONS = {
 'ro': dict(map='Harta cursului', c03='Capitolele 0–3: date, randamente și distribuții', c811='Capitolele 8–11: volatilitate, risc și memorie', exam='Examenul'),
 'en': dict(map='The Course Map', c03='Chapters 0–3: Data, Returns and Distributions', c811='Chapters 8–11: Volatility, Risk and Memory', exam='The Exam'),
}

L = {
 'ro': dict(file='ch16_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 16 · Recapitulare', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și problemele de recapitulare se recalculează odată cu el. Abrevieri: VaR (value at risk), ES (expected shortfall), HS (simulare istorică), EWMA '
        '(exponentially weighted moving average, media mobilă ponderată exponențial), GARCH (generalised autoregressive conditional heteroskedasticity), IGARCH (GARCH '
        'integrat), ML (verosimilitate maximă), ACF (funcția de autocorelație), VR (raportul varianțelor), JB (Jarque–Bera), R/S (amplitudinea rescalată), MDD (drawdown '
        'maxim), CAGR (rata anuală compusă de creștere), SE (eroarea standard), CI (intervalul de încredere), LR (raportul de verosimilitate), GBM (mișcarea browniană '
        'geometrică), TPR (rata adevărat pozitivelor), FPR (rata fals pozitivelor), AUC (aria de sub curba ROC), EL (pierderea așteptată), PD (probabilitatea de '
        'nerambursare), LGD (pierderea în caz de nerambursare), EAD (expunerea la nerambursare), CoVaR (VaR-ul sistemului condiționat de o instituție), ΔCoVaR (contribuția '
        'instituției la CoVaR), BET (indicele Bursei de Valori București), i.i.d. (independente și identic distribuite).',
   src='Datele cursului din data/market (pînă la 18 septembrie 2026): închiderile zilnice ale S&P 500 și BET din ianuarie 2000 și ale Bitcoin din septembrie 2014. '
       'Browserul taie fereastra și recalculează randamentele, indicatorii de performanță, momentele, indicele Hill, ACF, testele Ljung–Box și VR, VaR și ES, exponentul '
       'Hurst, varianța de selecție cumulativă, prognozele VaR 1% și testul Kupiec. Estimațiile GARCH(1,1)-t vin din codul capitolului (Quantlets/Ch_16), estimările '
       'α ale legii stabile pe trei orizonturi din Capitolul 3 (Quantlets/Ch_03). Alte ferestre, niveluri α, ferestre ale simulării istorice și valori λ sînt rulări '
       'suplimentare ale aceluiași cod. Problemele de recapitulare pornesc de la cifrele de pe slide-uri.'),
 'en': dict(file='ch16.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 16 · Review', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and review problems are recomputed with it. Abbreviations: VaR (value at risk), ES (expected shortfall), HS (historical simulation), EWMA '
        '(exponentially weighted moving average), GARCH (generalised autoregressive conditional heteroskedasticity), IGARCH (integrated GARCH), ML (maximum likelihood), '
        'ACF (autocorrelation function), VR (variance ratio), JB (Jarque–Bera), R/S (rescaled range), MDD (maximum drawdown), CAGR (compound annual growth rate), SE '
        '(standard error), CI (confidence interval), LR (likelihood ratio), GBM (geometric Brownian motion), TPR (true positive rate), FPR (false positive rate), AUC '
        '(area under the ROC curve), EL (expected loss), PD (probability of default), LGD (loss given default), EAD (exposure at default), CoVaR (the VaR of the system '
        'conditional on an institution), ΔCoVaR (the institution’s contribution to CoVaR), BET (the index of the Bucharest Stock Exchange), i.i.d. (independent and '
        'identically distributed).',
   src='Course data in data/market (up to 18 September 2026): daily closes of the S&P 500 and the BET since January 2000 and of Bitcoin since September 2014. The browser '
       'cuts the window and recomputes the returns, the performance measures, the moments, the Hill index, the ACF, the Ljung–Box and VR tests, VaR and ES, the Hurst '
       'exponent, the running sample variance, the VaR 1% forecasts and the Kupiec test. The GARCH(1,1)-t estimates come from the code of the chapter (Quantlets/Ch_16), '
       'the ML estimates of the stable α at three horizons from Chapter 3 (Quantlets/Ch_03). Other windows, levels α, historical-simulation windows and values of λ are '
       'extra runs of the same code. The review problems start from the numbers of the slides.'),
}

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child,.xt td.l,.xt th.l{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt .hi{font-weight:700;color:#2E7D32}.xt .bad{color:#CD0000}
.xh{font-weight:600;font-size:.92rem;color:#1F2A44;margin:.9rem 0 .1rem}
.we{border:1px solid #C5D2E8;border-left:4px solid #1A3A6E;border-radius:6px;padding:.55rem .8rem;margin:.8rem 0 .4rem;background:#F6F8FC}
.we-h{font-weight:600;margin:0 0 .2rem;color:#1A3A6E}
.we-f{overflow-x:auto;overflow-y:hidden;font-size:.95rem}
.we ol{margin:.3rem 0 0 1.3rem;padding:0;font-size:.92rem}.we li{margin:.18rem 0;overflow-wrap:anywhere}
.tw.veiled-chart,.we.veiled-chart{filter:blur(10px);opacity:.35;user-select:none}
.xhold{display:none}
.wctl{display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:flex-end;margin:.3rem 0 .2rem;max-width:520px}
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
    sl = ' (slide)'
    T = lambda r, e: r if ro else e
    RP = 'Problemă de recapitulare' if ro else 'Review problem'
    WIN = lambda i: select(i, T('Începutul ferestrei', 'Start of the window'), [('2015-01-01', T('ianuarie 2015', 'January 2015') + lec), ('2017-01-01', T('ianuarie 2017', 'January 2017')),
                                                                              ('2019-01-01', T('ianuarie 2019', 'January 2019')), ('2021-01-01', T('ianuarie 2021', 'January 2021'))], '2015-01-01')
    n = (lambda s: s.replace('.', ',')) if ro else (lambda s: s)
    return {
     'three': (WIN('ts-w') + play('ts'), CSS + charts('ts-chart')),
     'sum-tab': (WIN('sm-w') + play('sm'),
                 charts('sm-chart') + xh(T('Randamente logaritmice zilnice pe fereastra aleasă (tabelul de pe slide pentru ianuarie 2015)', 'Daily log returns on the chosen window (the table of the slide for January 2015)')) + tw('sm-tab')),
     'interp': (select('ip-a', T('Nivelul α al VaR', 'VaR level α'), [('0.01', '1%' + sl), ('0.005', n('0.5%')), ('0.025', n('2.5%')), ('0.05', '5%')], '0.01') + play('ip'), charts('ip-chart')),
     'stylised': (WIN('sf-w') + slider('sf-l', T('Numărul de laguri', 'Number of lags'), 10, 100, 10, 50) + play('sf'), charts('sf-chart')),
     'perf': (WIN('pf-w') + play('pf'), charts('pf-chart')),
     'agg': (select('ag-k', T('Seria', 'Series'), [('sp500', 'S&P 500' + sl), ('bet', 'BET'), ('btc', 'Bitcoin')], 'sp500') + play('ag'), charts('ag-chart')),
     'backtest': (select('bt-k', T('Seria', 'Series'), [('bet', 'BET' + sl), ('sp500', 'S&P 500'), ('btc', 'Bitcoin')], 'bet')
                  + slider('bt-w', T('Fereastra simulării istorice (zile)', 'Historical-simulation window (days)'), 250, 1000, 50, 500)
                  + slider('bt-l', T('Factorul EWMA λ', 'EWMA factor λ'), 0.90, 0.99, 0.01, 0.94) + play('bt'), charts('bt-chart')),
     'p1': (slider('q1-p', T('Valoarea la sfîrșitul anului 2', 'Value at the end of year 2'), 20, 150, 4, 64) + play('q1'),
            charts('q1-chart') + we('q1-we', f'{RP}: {T("randamente și drawdown", "returns and drawdown")}',
                                    r'\begin{gathered} R_t = \frac{P_t}{P_{t-1}} - 1, \quad r_t = \ln(1 + R_t), \\ \text{CAGR} = \Big(\frac{P_3}{P_0}\Big)^{1/3} - 1, \quad \text{MDD} = \min_t\Big(\frac{P_t}{\max_{s \le t}P_s} - 1\Big) \end{gathered}',
                                    slider('q1-a', T('Valoarea la sfîrșitul anului 1', 'Value at the end of year 1'), 100, 250, 10, 160)
                                    + slider('q1-c', T('Valoarea la sfîrșitul anului 3', 'Value at the end of year 3'), 40, 200, 4, 96))),
     'p2': (slider('q2-k', T('Excesul de boltire K', 'Excess kurtosis K'), 0.5, 12, 0.5, 6) + play('q2'),
            charts('q2-chart') + we('q2-we', f'{RP}: {T("distribuție și cozi", "distribution and tails")}',
                                    r'\text{JB} = \frac{n}{6}\Big(S^2 + \frac{K^2}{4}\Big), \quad K = \frac{6}{\nu - 4}, \quad \hat\alpha = \Big[\frac{1}{k}\sum_{i=1}^{k}\ln\frac{L_{(i)}}{L_{(k+1)}}\Big]^{-1}',
                                    slider('q2-s', T('Asimetria S', 'Skewness S'), -2, 1, 0.1, -0.5)
                                    + select('q2-n', T('Numărul de randamente n', 'Number of returns n'), [('250', '250'), ('500', '500'), ('1000', '1000' + sl), ('2500', '2500')], '1000')
                                    + select('q2-h', T('Pierderi din coadă k', 'Tail losses k'), [('2', '2'), ('3', '3'), ('4', '4' + sl)], '4'))),
     'p3': (slider('q3-r', T('Corelația ρ', 'Correlation ρ'), -1, 1, 0.1, 0.3) + play('q3'),
            charts('q3-chart') + we('q3-we', f'{RP}: {T("portofoliu, Monte Carlo și GBM", "portfolio, Monte Carlo and GBM")}',
                                    r'\begin{gathered} \sigma_p^2 = w^2\sigma_1^2 + (1 - w)^2\sigma_2^2 + 2w(1 - w)\rho\,\sigma_1\sigma_2, \\ \text{SE} = \sqrt{\frac{p(1 - p)}{N}}, \quad P(S_1 < S_0) = \Phi\Big(-\frac{\mu - \sigma^2/2}{\sigma}\Big) \end{gathered}',
                                    slider('q3-w', T('Ponderea primului activ', 'Weight of the first asset'), 0, 1, 0.1, 0.6)
                                    + slider('q3-s', T('Volatilitatea GBM σ (% pe an)', 'GBM volatility σ (% a year)'), 5, 60, 5, 25)
                                    + select('q3-N', T('Extrageri Monte Carlo N', 'Monte Carlo draws N'), [('1000', '1000'), ('10000', '10000' + sl), ('100000', '100000')], '10000'))),
     'p4': (slider('q4-r', T('Autocorelația ρ̂(1)', 'Autocorrelation ρ̂(1)'), -0.1, 0.15, 0.01, 0.06) + play('q4'),
            charts('q4-chart') + we('q4-we', f'{RP}: {T("autocorelație și raportul varianțelor", "autocorrelation and the variance ratio")}',
                                    r'\begin{gathered} Q(m) = T(T + 2)\sum_{k=1}^{m}\frac{\hat\rho_k^2}{T - k}, \quad \mathrm{VR}(q) = 1 + 2\sum_{k=1}^{q-1}\Big(1 - \frac{k}{q}\Big)\hat\rho(k), \\ Z(q) = \frac{\widehat{\mathrm{VR}} - 1}{\sqrt{2(2q - 1)(q - 1)/(3qT)}} \end{gathered}',
                                    select('q4-T', T('Numărul de randamente T', 'Number of returns T'), [('500', '500'), ('1000', '1000'), ('2500', '2500' + sl), ('5000', '5000')], '2500')
                                    + slider('q4-e', T('SE robustă a lui VR(5)', 'Robust SE of VR(5)'), 0.03, 0.12, 0.005, 0.065))),
     'p5': (slider('q5-e', T('Șocul de azi εₜ', 'Today’s shock εₜ'), -5, 5, 0.25, -2.5) + play('q5'),
            charts('q5-chart') + we('q5-we', f'{RP}: {T("prognoze de volatilitate", "volatility forecasts")}',
                                    r'\begin{gathered} \sigma_{t+1}^2 = \omega + \alpha\varepsilon_t^2 + \beta\sigma_t^2, \quad \bar\sigma^2 = \frac{\omega}{1 - \alpha - \beta}, \\ E_t\sigma_{t+h}^2 = \bar\sigma^2 + (\alpha + \beta)^{h-1}(\sigma_{t+1}^2 - \bar\sigma^2) \end{gathered}',
                                    slider('q5-b', T('Coeficientul β', 'Coefficient β'), 0.8, 0.905, 0.005, 0.89)
                                    + select('q5-l', T('Factorul EWMA λ', 'EWMA factor λ'), [('0.94', n('0.94') + sl), ('0.9', n('0.90')), ('0.97', n('0.97'))], '0.94'))),
     'p6': (slider('q6-x', T('Depășiri în 250 de zile', 'Exceptions in 250 days'), 0, 15, 1, 6) + play('q6'),
            charts('q6-chart') + we('q6-we', f'{RP}: {T("VaR, ES și backtesting", "VaR, ES and backtesting")}',
                                    r'\begin{gathered} \mathrm{VaR}_\alpha = -(\mu + \sigma z_\alpha), \quad \mathrm{ES}_\alpha = -\mu + \sigma\frac{\varphi(z_\alpha)}{\alpha}, \\ LR_{uc} = -2\ln\frac{(1 - \alpha)^{n - x}\alpha^x}{(1 - \hat\pi)^{n - x}\hat\pi^x} \end{gathered}',
                                    slider('q6-s', T('Abaterea standard zilnică σ (%)', 'Daily standard deviation σ (%)'), 0.8, 3, 0.1, 1.6))),
     'p7': (slider('q7-b', T('R/S medie pentru n = 2000', 'Average R/S for n = 2000'), 20, 120, 2, 60) + play('q7'),
            charts('q7-chart') + we('q7-we', f'{RP}: {T("memorie lungă și scalarea riscului", "long memory and risk scaling")}',
                                    r'H = \frac{\log_{10}\big[(R/S)_{2000}/(R/S)_{20}\big]}{\log_{10}(2000/20)}, \quad d = H - ' + T('0{,}5', '0.5') + r', \quad \mathrm{VaR}^{(h)} = h^H\,\mathrm{VaR}^{(1)}',
                                    slider('q7-a', T('R/S medie pentru n = 20', 'Average R/S for n = 20'), 2, 8, 0.1, 4.1)
                                    + slider('q7-v', T('VaR 1% pe o zi (%)', 'One-day VaR 1% (%)'), 1, 5, 0.5, 2.5)
                                    + select('q7-h', T('Orizontul h (zile)', 'Horizon h (days)'), [('5', '5'), ('10', '10' + sl), ('20', '20'), ('60', '60')], '10'))),
     'p8': (slider('q8-f', T('Credite performante marcate ca neperformante', 'Good loans flagged as bad'), 0, 210, 6, 42) + play('q8'),
            charts('q8-chart') + we('q8-we', f'{RP}: {T("scoring de credit și risc sistemic", "credit scoring and systemic risk")}',
                                    r'\begin{gathered} \text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}, \\ \text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}, \quad \Delta\mathrm{CoVaR} = \hat b\,(q_{50\%} - q_{1\%}) \end{gathered}',
                                    slider('q8-t', T('Credite neperformante găsite', 'Bad loans found'), 0, 90, 3, 60)
                                    + slider('q8-b', T('Panta b̂ a regresiei cuantilice', 'Slope b̂ of the quantile regression'), 0, 1.2, 0.1, 0.6))),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang` (the EN deck has one front-matter frame less)."""
    from interactive_common import frames_of, latex_text, ROOT
    tt = lambda t: latex_text(re.sub(r'\\sfmchlinkt?\{[^{}]*\}\{([^{}]*)\}', r'\1', t))
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    ro = frames_of(open(os.path.join(ROOT, TEX['ro']) + '.tex', encoding='utf-8').read())
    assert len(frames) == len(pages), (lang, len(frames), len(pages))
    off = len(ro) - len(frames)
    res = {}
    for a, fr, _, _ in XBLOCKS:
        assert tt(ro[fr - 1][0]) == TITLES[a], (a, tt(ro[fr - 1][0]))
        k = fr - 1 - (off if lang == 'en' else 0)
        if lang == 'en':
            assert tt(frames[k][0]) == TITLES_EN[a], (a, tt(frames[k][0]))
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch16.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch16_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    html = html[:start] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[end:]
    return html.replace('</head>', MATHJAX + '</head>', 1)


def build():
    import pymupdf
    import types
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch16_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])




# readouts, table headers and worked-problem steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s %]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const X = ' (rulare suplimentară a codului din curs)';
const XS = ' (alte valori decît pe slide)';
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const list = a => a.length > 1 ? a.slice(0, -1).join(', ') + ' și ' + a[a.length - 1] : a[0];
const ex = a => a.map((s, i) => i === 0 ? s : (String(s)[0] === '−' ? ' − ' + String(s).slice(1) : ' + ' + s)).join('');
const P = p => p[0] === '<' ? `p ${p}` : `p = ${p}`;
const neg = s => s[0] === '−' ? s.slice(1) : '−' + s;
const NM = ['S&P 500', 'BET', 'Bitcoin'];
const ZN = { g: 'verde (0–4 depășiri)', y: 'galbenă (5–9 depășiri)', r: 'roșie (10 sau mai multe depășiri)' };
window.CH16_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 mom: ['media', 'varianța', 'asimetria', 'boltirea'],
 h: { smc: 'Randamente logaritmice zilnice',
      sm: ['Observații (pe an)', 'Randament logaritmic mediu, % pe an', 'Volatilitate, % pe an', 'Raportul Sharpe (r<sub>f</sub> = 0) ± SE', 'Asimetrie; excesul de boltire',
           'Tail index-ul Hill al pierderilor (k = 2,5% n)', 'Ljung–Box Q(10): rₜ; rₜ²', 'VR(5); Z*(5) robust', 'GARCH(1,1)-t: α + β', 'VaR 1%: istoric; Normal (%)',
           'ES 2,5%, istoric (%)', 'Hurst (R/S): rₜ; |rₜ|'] },
 ts: (y, rows, x) => `100 investiți la începutul ferestrei din ${y}${x ? X : ''}: ` + rows.map(r => `${r[0]} ajunge la ${r[1]}, cu drawdown-ul maxim ${r[2]}% (de la vîrful din ${r[3]} la minimul din ${r[4]})`).join('; ') + '.',
 sm: (y, d0, rows, crit, x) => `Fereastra din ${y}${x ? X : ''}: Ljung–Box Q(10) pe rₜ și pe rₜ²: ` + rows.map((r, i) => `${r[1]} și ${r[2]} la ${NM[i]} (de ${r[3]} ${de(r[3])}ori mai mare pe rₜ²)`).join(', ') + `; valoarea critică este ${crit}. Exponentul Hurst al lui |rₜ|: ` + rows.map((r, i) => `${NM[i]} ${r[4]}`).join(', ') + '.',
 ip: (a, rows, x) => `VaR ${a}%${x ? X : ''}, istoric și Normal: ` + rows.map((r, i) => `${NM[i]} ${r[0]}% și ${r[1]}% (VaR Normal ${num(r[2]) >= 0 ? 'prea mic' : 'prea mare'} cu ${r[2].replace('−', '')}%)`).join('; ') + '.',
 sf: (y, L, res, x) => `Fereastra din ${y}, ${L} ${de(L)}laguri${x ? X : ''}: ` + res.map(r => `${r[0]}: ρ̂₁(|r|) = ${r[1]}, ρ̂₁(r) = ${r[2]}, ${r[3]} din ${L} autocorelații ale lui r în afara benzii ±${r[5]}, ${r[4]} din ${L} autocorelații ale lui |r| pozitive`).join('; ') + '.',
 pf: (y, rows, x) => `Fereastra din ${y}${x ? X : ''}: ` + rows.map(r => `${r[0]}: CAGR ${r[1]}%, volatilitate ${r[2]}%, Sharpe ${r[3]} (CI 95% [${r[4]}; ${r[5]}]), media aritmetică anualizată ${r[6]}% (${r[7].replace(/^(?=\d)/, '+')} puncte față de CAGR)`).join('; ') + '.',
 ag: (nm, rows, d0, v10, vl, x) => `${nm}, estimări ML din Capitolul 3: ` + ['zilnic', 'săptămînal', 'lunar'].map((h, i) => rows[i] ? `${h} α̂ = ${rows[i][0]} (CI 95% [${rows[i][1]}; ${rows[i][2]}], n = ${rows[i][3]})` : `${h} la frontiera α = 2 (distribuția Normală)`).join(', ') + `. Varianța de selecție cumulativă a randamentelor zilnice${x ? X : ''}, de la ${d0}: ${v10 === null ? '' : `${v10} la 1 ianuarie 2010, `}${vl} la 18 septembrie 2026.`,
 bt: (nm, d0, n, e, hs, ew, w, lam, x, late) => `${nm}, din ${d0}, ${n} ${de(n)}zile, ${e} depășiri așteptate${x ? X : ''}: HS pe ${w} de zile ${hs[0]} ${de(hs[0])}depășiri (${hs[1]}%, Kupiec LR = ${hs[4]}, ${P(hs[2])}), cel mult ${hs[3]} în 250 de zile; EWMA-Normal cu λ = ${lam}: ${ew[0]} ${de(ew[0])}depășiri (${ew[1]}%, LR = ${ew[4]}, ${P(ew[2])}), cel mult ${ew[3]} în 250 de zile.` + (late ? ` Prima prognoză HS cere ${w} de randamente anterioare, de aceea seria începe mai tîrziu.` : ''),
 q1a: (Pt, R, q, r) => `R = ${R.join('%; ')}%; r = ln ${q[0]} = ${r[0]}, ln ${q[1]} = ${r[1]}, ln ${q[2]} = ${r[2]}.`,
 q1b: (s, P3, t) => `Σ rₜ = ${s} = ln(${P3}/100), deci randamentul total este e^(${s}) − 1 = ${t}%, la fel ca ${P3}/100 − 1.`,
 q1c: (R, m, P3, c, gt) => `Media aritmetică (${ex(R)})/3 = ${m}%; CAGR = (${P3}/100)^(1/3) − 1 = ${c}%. ` + (gt ? 'Media depășește CAGR: volatilitatea reduce creșterea compusă (volatility drag).' : 'Aici media nu depășește CAGR.'),
 q1d: (tr, pk, d, g) => `MDD = ${tr}/${pk} − 1 = ${d}%; pentru revenirea de la ${tr} la ${pk}, indicele are nevoie de ${pk}/${tr} − 1 = ${g}%.`,
 q1d0: () => 'Indicele nu coboară niciodată sub vîrful anterior: drawdown-ul maxim este 0.',
 q1: (m, c, d, g, x) => `Media aritmetică ${m}%, CAGR ${c}%, drawdown maxim ${d}%, cîștig necesar pentru revenire ${g}%${x ? XS : ''}.`,
 q2a: (n, S, S2, js, K, K2, jk, jb, rej) => `Componenta asimetriei: ${n}/6 × (${S})² = ${n}/6 × ${S2} = ${js}; componenta boltirii: ${n}/24 × ${K}² = ${n}/24 × ${K2} = ${jk}; JB = ${jb} ${rej ? '>' : '≤'} 5,99: normalitatea ${rej ? 'se respinge' : 'nu se respinge'} la 5%.`,
 q2b: (K, nu) => `6/(ν − 4) = ${K} ⇒ ν = 4 + 6/${K} = ${nu}.`,
 q2c: (k, it, m, al) => it.map(t => `ln(${t[0]}/${t[1]}) = ${t[2]}`).join(', ') + `; media ${m}, deci α̂ = 1/${m} = ${al} (k = ${k}).`,
 q2d: (al, mm, fin, se) => `Există momentele de ordin p < ${al}: ${list(mm)}; ${num(al) > 4 ? 'există și boltirea' : 'boltirea nu există'}. ` + (fin ? 'O lege stabilă cu α < 2 (varianță infinită) are cozi mai groase decît aceste date.' : 'Cu α̂ < 2, o lege stabilă cu α < 2 este compatibilă cu aceste date.') + ` SE ≈ α̂/√k = ${se}: estimarea este foarte zgomotoasă.`,
 q2: (jb, rej, nu, al, x) => `JB = ${jb} (${rej ? 'normalitatea se respinge' : 'normalitatea nu se respinge'}), ν = ${nu}, α̂ Hill = ${al}${x ? XS : ''}.`,
 q3a: (w1, w2, rho, a, b, c, v, sp, wa, less) => `Cu ponderile ${w1}% și ${w2}% și ρ = ${rho}: σp² = ${ex([a, b, c])} = ${v}, σp = ${sp}%; media ponderată a volatilităților este ${wa}%` + (less ? ', deci portofoliul este mai puțin riscant decît media componentelor.' : ': cu ρ = 1 nu există diversificare.'),
 q3b: (N, se, lo, hi) => `SE = √(0,01 × 0,99/${N}) = ${se}%; intervalul de 95%: [${lo}%; ${hi}%].`,
 q3c: (m1, half, md, dr, sg, z, pl) => `Media 100e^0,08 = ${m1}; mediana 100e^(0,08 − ${half}) = ${md}; P(S₁ < S₀) = Φ(${neg(dr)}/${sg}) = Φ(${z}) = ${pl}%.`,
 q3: (sp, wa, se, pl, x) => `σp = ${sp}% față de media ponderată ${wa}%; SE Monte Carlo ${se}%; P(S₁ < S₀) = ${pl}%${x ? XS : ''}.`,
 q4a: (T, Q, Ts, s2, rej) => `Cu T = ${T}: Q(4) = T(T + 2) Σ ρ̂ₖ²/(T − k) = ${Q}, apropiat de T Σ ρ̂ₖ² = ${Ts}, cu Σ ρ̂ₖ² = ${s2}; ${Q} ${rej ? '>' : '≤'} 9,49: absența autocorelației ${rej ? 'se respinge' : 'nu se respinge'}.`,
 q4b: (it, s, VR) => `VR(5) = 1 + 2(${ex(it.map(t => (t[2] ? '−' : '') + `${t[0]} × ${t[1]}`))}) = 1 + 2 × ${s} = ${VR}.`,
 q4c: (a, b, se, Z, rz, VR, seR, Zs, rzs) => `SE = √(${a}/${b}) = ${se}, Z(5) = ${Z}, |Z(5)| ${rz ? '>' : '≤'} 1,96; Z*(5) = (${VR} − 1)/${seR} = ${Zs}, |Z*(5)| ${rzs ? '>' : '≤'} 1,96 (robust): mersul aleator ${rzs ? 'se respinge' : 'nu se respinge'} după testul robust.`,
 q4: (Q, rq, VR, Z, Zs, rz, rzs, x) => `Q(4) = ${Q} (${rq ? 'respins' : 'nerespins'}), VR(5) = ${VR}, Z(5) = ${Z} (${rz ? 'respins' : 'nerespins'}), Z*(5) = ${Zs} (${rzs ? 'respins' : 'nerespins'})${x ? XS : ''}.`,
 q5a: (aE, bS, s1, sd, v) => `σ²ₜ₊₁ = 0,03 + ${aE} + ${bS} = ${s1}, σₜ₊₁ = ${sd}%; VaR 1% = 2,326 × ${sd} = ${v}%.`,
 q5b: (p, hl, om, lr, an) => `α + β = ${p}; h½ = ln 0,5/ln ${p} = ${hl} zile; σ̄² = 0,03/${om} = ${lr}; √(252 × ${lr}) = ${an}% pe an.`,
 q5c: (lr, p, p9, s1, f, lam, ol, e2, ew) => `Eₜσ²ₜ₊₁₀ = ${lr} + ${p}⁹ × (${s1} − ${lr}) = ${lr} + ${p9} × (${s1} − ${lr}) = ${f}; EWMA: ${lam} × 1,8 + ${ol} × ${e2} = ${ew}, aceeași pentru orice orizont.`,
 q5d: (sd, a, b) => `Bitcoin se tranzacționează în fiecare zi: ${sd} × √365 = ${a}%; cu 252, rezultatul ar fi greșit: ${b}%.`,
 q5: (s1, v, hl, f, ew, x) => `σ²ₜ₊₁ = ${s1}, VaR 1% = ${v}%, timp de înjumătățire ${hl} zile, Eₜσ²ₜ₊₁₀ = ${f}, EWMA ${ew}${x ? XS : ''}.`,
 q6a: (z, sg, V, Vl, ph, r, E, El, V10) => `VaR 1% = ${z} × ${sg} = ${V}% (${Vl} ${de(Vl)}lei); ES 2,5% = ${sg} × φ(z₂,₅%)/0,025 = ${sg} × ${ph}/0,025 = ${sg} × ${r} = ${E}% (${El} ${de(El)}lei); pe 10 zile: √10 × ${V} = ${V10}%.`,
 q6b: (na, kv, hv, ne, ke, s, he) => `k = ⌈${na}⌉ = ${kv}: VaR 1% = ${hv}%; k = ⌈${ne}⌉ = ${ke}: ES 2,5% = ${s}/${ke} = ${he}%.`,
 q6c: (x, nx, l0, ph, l1, lr, p, rej, z, q) => (x === 0 ? `ℓ₀ = 250 ln 0,99 = ${l0}; ℓ₁ = 250 ln 1 = 0` : `ℓ₀ = ${nx} ln 0,99 + ${x} ln 0,01 = ${l0}; ℓ₁ = ${nx} ln ${q} + ${x} ln ${ph} = ${l1}`) + `; LR_uc = ${lr} ${rej ? '>' : '<'} 3,84 (p = ${p}): ${rej ? 'se respinge' : 'nu se respinge'}; zona ${ZN[z]}.`,
 q6: (V, E, hv, he, x, lr, p, rej, z, xs) => `Normal: VaR 1% ${V}%, ES 2,5% ${E}%; istoric: VaR 1% ${hv}%, ES 2,5% ${he}%; ${x} ${de(x)}depășiri: LR_uc = ${lr} (p = ${p}, ${rej ? 'respins' : 'nerespins'}), zona ${ZN[z].split(' ')[0]}${xs ? XS : ''}.`,
 q7a: (b, a, r, lg, H, d) => `H = log₁₀(${b}/${a})/log₁₀(2000/20) = log₁₀(${r})/2 = ${lg}/2 = ${H}; d = H − 0,5 = ${d}.`,
 q7b: (h, H, fh, v1, Vh, fs, Vs, gt) => `${h}^H = ${h}^${H} = ${fh}: VaR = ${v1} × ${fh} = ${Vh}%; √${h} = ${fs}: VaR = ${Vs}%. ` + (gt ? 'Cu H > 0,5, regula rădăcinii pătrate a timpului subestimează riscul pe mai multe zile.' : 'Cu H < 0,5, regula rădăcinii pătrate a timpului supraestimează riscul pe mai multe zile.'),
 q7c: () => 'Înainte de concluzie: o bandă Monte Carlo pentru Ĥ pe date i.i.d. de aceeași lungime (R/S este deplasat în sus), testul Lo, subeșantioane și date permutate (rupturi, GARCH).',
 q7: (H, d, h, Vh, Vs, x) => `H = ${H}, d = ${d}; VaR pe ${h} ${de(h)}zile: ${Vh}% cu h^H, ${Vs}% cu √h${x ? XS : ''}.`,
 q8a: (TP, TN, acc, tpr, FP, fpr, fl, pr, ag, gt) => `Acuratețea (${TP} + ${TN})/300 = ${acc}%; TPR = ${TP}/90 = ${tpr}%; FPR = ${FP}/210 = ${fpr}%; precizia ${pr === null ? 'nu este definită (niciun credit marcat)' : `${TP}/${fl} = ${pr}%`}; regula „mereu bun”: ${ag}%` + (gt ? '.' : ': modelul este mai puțin precis decît această regulă.'),
 q8b: () => 'Gini = 2 × 0,80 − 1 = 0,60; EL = 0,04 × 0,45 × 200 000 = 3 600 de lei.',
 q8c: (b, cv, cm, dc, df) => `CoVaR = −(−2,1 + ${b} × (−5,0)) = ${cv}%; la mediană: −(−2,1 + ${b} × 0,1) = ${cm}%; ΔCoVaR = ${b} × ${df} = ${dc} pp.`,
 q8: (acc, tpr, fpr, pr, dc, x) => `Acuratețe ${acc}%, TPR ${tpr}%, FPR ${fpr}%, precizie ${pr === null ? 'nedefinită' : pr + '%'}; ΔCoVaR ${dc} pp${x ? XS : ''}.`,
};
})();
""",
 'en': r"""(function () {
const X = ' (extra run of the chapter’s code)';
const XS = ' (values other than on the slide)';
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const list = a => a.length > 1 ? a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1] : a[0];
const ex = a => a.map((s, i) => i === 0 ? s : (String(s)[0] === '−' ? ' − ' + String(s).slice(1) : ' + ' + s)).join('');
const P = p => p[0] === '<' ? `p ${p}` : `p = ${p}`;
const neg = s => s[0] === '−' ? s.slice(1) : '−' + s;
const num = s => +String(s).replace(/[\s,%]/g, '').replace('−', '-');
const NM = ['S&P 500', 'BET', 'Bitcoin'];
const ZN = { g: 'green (0–4 exceptions)', y: 'yellow (5–9 exceptions)', r: 'red (10 or more exceptions)' };
window.CH16_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 mom: ['the mean', 'the variance', 'the skewness', 'the kurtosis'],
 h: { smc: 'Daily log returns',
      sm: ['Observations (per year)', 'Mean log return, % a year', 'Volatility, % a year', 'Sharpe ratio (r<sub>f</sub> = 0) ± SE', 'Skewness; excess kurtosis', 'Hill tail index of losses (k = 2.5% n)',
           'Ljung–Box Q(10): rₜ; rₜ²', 'VR(5); robust Z*(5)', 'GARCH(1,1)-t: α + β', 'VaR 1%: historical; Normal (%)', 'ES 2.5%, historical (%)', 'Hurst (R/S): rₜ; |rₜ|'] },
 ts: (y, rows, x) => `100 invested at the start of the window from ${y}${x ? X : ''}: ` + rows.map(r => `${r[0]} reaches ${r[1]}, with a maximum drawdown of ${r[2]}% (from the peak of ${r[3]} to the trough of ${r[4]})`).join('; ') + '.',
 sm: (y, d0, rows, crit, x) => `Window from ${y}${x ? X : ''}: Ljung–Box Q(10) on rₜ and on rₜ²: ` + rows.map((r, i) => `${r[1]} and ${r[2]} for ${['the S&P 500', 'the BET', 'Bitcoin'][i]} (${r[3]} times larger on rₜ²)`).join(', ') + `; the critical value is ${crit}. Hurst exponent of |rₜ|: ` + rows.map((r, i) => `${NM[i]} ${r[4]}`).join(', ') + '.',
 ip: (a, rows, x) => `VaR ${a}%${x ? X : ''}, historical and Normal: ` + rows.map((r, i) => `${NM[i]} ${r[0]}% and ${r[1]}% (the Normal VaR is too ${num(r[2]) >= 0 ? 'low' : 'high'} by ${r[2].replace('−', '')}%)`).join('; ') + '.',
 sf: (y, L, res, x) => `Window from ${y}, ${L} lags${x ? X : ''}: ` + res.map(r => `${r[0]}: ρ̂₁(|r|) = ${r[1]}, ρ̂₁(r) = ${r[2]}, ${r[3]} of ${L} autocorrelations of r outside the band ±${r[5]}, ${r[4]} of ${L} autocorrelations of |r| positive`).join('; ') + '.',
 pf: (y, rows, x) => `Window from ${y}${x ? X : ''}: ` + rows.map(r => `${r[0]}: CAGR ${r[1]}%, volatility ${r[2]}%, Sharpe ${r[3]} (95% CI [${r[4]}, ${r[5]}]), annualised arithmetic mean ${r[6]}% (${r[7].replace(/^(?=\d)/, '+')} points against the CAGR)`).join('; ') + '.',
 ag: (nm, rows, d0, v10, vl, x) => `${nm}, ML estimates of Chapter 3: ` + ['daily', 'weekly', 'monthly'].map((h, i) => rows[i] ? `${h} α̂ = ${rows[i][0]} (95% CI [${rows[i][1]}, ${rows[i][2]}], n = ${rows[i][3]})` : `${h} at the bound α = 2 (the Normal distribution)`).join(', ') + `. Running sample variance of the daily returns${x ? X : ''}, from ${d0}: ${v10 === null ? '' : `${v10} on 1 January 2010, `}${vl} on 18 September 2026.`,
 bt: (nm, d0, n, e, hs, ew, w, lam, x, late) => `${nm}, from ${d0}, ${n} days, ${e} exceptions expected${x ? X : ''}: HS on ${w} days ${hs[0]} exceptions (${hs[1]}%, Kupiec LR = ${hs[4]}, ${P(hs[2])}), at most ${hs[3]} in 250 days; EWMA-Normal with λ = ${lam}: ${ew[0]} exceptions (${ew[1]}%, LR = ${ew[4]}, ${P(ew[2])}), at most ${ew[3]} in 250 days.` + (late ? ` The first HS forecast needs ${w} earlier returns, so the series starts later.` : ''),
 q1a: (Pt, R, q, r) => `R = ${R.join('%, ')}%; r = ln ${q[0]} = ${r[0]}, ln ${q[1]} = ${r[1]}, ln ${q[2]} = ${r[2]}.`,
 q1b: (s, P3, t) => `Σ rₜ = ${s} = ln(${P3}/100), so the total return is e^(${s}) − 1 = ${t}%, the same as ${P3}/100 − 1.`,
 q1c: (R, m, P3, c, gt) => `Arithmetic mean (${ex(R)})/3 = ${m}%; CAGR = (${P3}/100)^(1/3) − 1 = ${c}%. ` + (gt ? 'The mean exceeds the CAGR: volatility lowers the compound growth (volatility drag).' : 'Here the mean does not exceed the CAGR.'),
 q1d: (tr, pk, d, g) => `MDD = ${tr}/${pk} − 1 = ${d}%; to return from ${tr} to ${pk} the index needs ${pk}/${tr} − 1 = ${g}%.`,
 q1d0: () => 'The index never falls below its previous peak: the maximum drawdown is 0.',
 q1: (m, c, d, g, x) => `Arithmetic mean ${m}%, CAGR ${c}%, maximum drawdown ${d}%, gain needed to recover ${g}%${x ? XS : ''}.`,
 q2a: (n, S, S2, js, K, K2, jk, jb, rej) => `Skewness part: ${n}/6 × (${S})² = ${n}/6 × ${S2} = ${js}; kurtosis part: ${n}/24 × ${K}² = ${n}/24 × ${K2} = ${jk}; JB = ${jb} ${rej ? '>' : '≤'} 5.99: normality is ${rej ? 'rejected' : 'not rejected'} at 5%.`,
 q2b: (K, nu) => `6/(ν − 4) = ${K} ⇒ ν = 4 + 6/${K} = ${nu}.`,
 q2c: (k, it, m, al) => it.map(t => `ln(${t[0]}/${t[1]}) = ${t[2]}`).join(', ') + `; mean ${m}, so α̂ = 1/${m} = ${al} (k = ${k}).`,
 q2d: (al, mm, fin, se) => `Moments of order p < ${al} exist: ${list(mm)}; ${num(al) > 4 ? 'the kurtosis exists as well' : 'the kurtosis does not'}. ` + (fin ? 'A stable law with α < 2 (infinite variance) has heavier tails than these data.' : 'With α̂ < 2, a stable law with α < 2 is consistent with these data.') + ` SE ≈ α̂/√k = ${se}: the estimate is very noisy.`,
 q2: (jb, rej, nu, al, x) => `JB = ${jb} (normality ${rej ? 'rejected' : 'not rejected'}), ν = ${nu}, Hill α̂ = ${al}${x ? XS : ''}.`,
 q3a: (w1, w2, rho, a, b, c, v, sp, wa, less) => `With weights ${w1}% and ${w2}% and ρ = ${rho}: σp² = ${ex([a, b, c])} = ${v}, σp = ${sp}%; the weighted average of the volatilities is ${wa}%` + (less ? ', so the portfolio is less risky than the average of its parts.' : ': with ρ = 1 there is no diversification.'),
 q3b: (N, se, lo, hi) => `SE = √(0.01 × 0.99/${N}) = ${se}%; 95% interval: [${lo}%, ${hi}%].`,
 q3c: (m1, half, md, dr, sg, z, pl) => `Mean 100e^0.08 = ${m1}; median 100e^(0.08 − ${half}) = ${md}; P(S₁ < S₀) = Φ(${neg(dr)}/${sg}) = Φ(${z}) = ${pl}%.`,
 q3: (sp, wa, se, pl, x) => `σp = ${sp}% against the weighted average ${wa}%; Monte Carlo SE ${se}%; P(S₁ < S₀) = ${pl}%${x ? XS : ''}.`,
 q4a: (T, Q, Ts, s2, rej) => `With T = ${T}: Q(4) = T(T + 2) Σ ρ̂ₖ²/(T − k) = ${Q}, close to T Σ ρ̂ₖ² = ${Ts} with Σ ρ̂ₖ² = ${s2}; ${Q} ${rej ? '>' : '≤'} 9.49: the absence of autocorrelation is ${rej ? 'rejected' : 'not rejected'}.`,
 q4b: (it, s, VR) => `VR(5) = 1 + 2(${ex(it.map(t => (t[2] ? '−' : '') + `${t[0]} × ${t[1]}`))}) = 1 + 2 × ${s} = ${VR}.`,
 q4c: (a, b, se, Z, rz, VR, seR, Zs, rzs) => `SE = √(${a}/${b}) = ${se}, Z(5) = ${Z}, |Z(5)| ${rz ? '>' : '≤'} 1.96; robust Z*(5) = (${VR} − 1)/${seR} = ${Zs}, |Z*(5)| ${rzs ? '>' : '≤'} 1.96: the robust test ${rzs ? 'rejects' : 'does not reject'} the random walk.`,
 q4: (Q, rq, VR, Z, Zs, rz, rzs, x) => `Q(4) = ${Q} (${rq ? 'rejected' : 'not rejected'}), VR(5) = ${VR}, Z(5) = ${Z} (${rz ? 'rejected' : 'not rejected'}), Z*(5) = ${Zs} (${rzs ? 'rejected' : 'not rejected'})${x ? XS : ''}.`,
 q5a: (aE, bS, s1, sd, v) => `σ²ₜ₊₁ = 0.03 + ${aE} + ${bS} = ${s1}, σₜ₊₁ = ${sd}%; VaR 1% = 2.326 × ${sd} = ${v}%.`,
 q5b: (p, hl, om, lr, an) => `α + β = ${p}; h½ = ln 0.5/ln ${p} = ${hl} days; σ̄² = 0.03/${om} = ${lr}; √(252 × ${lr}) = ${an}% a year.`,
 q5c: (lr, p, p9, s1, f, lam, ol, e2, ew) => `Eₜσ²ₜ₊₁₀ = ${lr} + ${p}⁹ × (${s1} − ${lr}) = ${lr} + ${p9} × (${s1} − ${lr}) = ${f}; EWMA: ${lam} × 1.8 + ${ol} × ${e2} = ${ew}, the same for every horizon.`,
 q5d: (sd, a, b) => `Bitcoin trades every day: ${sd} × √365 = ${a}%; with 252 the result, ${b}%, would be wrong.`,
 q5: (s1, v, hl, f, ew, x) => `σ²ₜ₊₁ = ${s1}, VaR 1% = ${v}%, half-life ${hl} days, Eₜσ²ₜ₊₁₀ = ${f}, EWMA ${ew}${x ? XS : ''}.`,
 q6a: (z, sg, V, Vl, ph, r, E, El, V10) => `VaR 1% = ${z} × ${sg} = ${V}% (${Vl} RON); ES 2.5% = ${sg} × φ(z₂.₅%)/0.025 = ${sg} × ${ph}/0.025 = ${sg} × ${r} = ${E}% (${El} RON); 10 days: √10 × ${V} = ${V10}%.`,
 q6b: (na, kv, hv, ne, ke, s, he) => `k = ⌈${na}⌉ = ${kv}: VaR 1% = ${hv}%; k = ⌈${ne}⌉ = ${ke}: ES 2.5% = ${s}/${ke} = ${he}%.`,
 q6c: (x, nx, l0, ph, l1, lr, p, rej, z, q) => (x === 0 ? `ℓ₀ = 250 ln 0.99 = ${l0}; ℓ₁ = 250 ln 1 = 0` : `ℓ₀ = ${nx} ln 0.99 + ${x} ln 0.01 = ${l0}; ℓ₁ = ${nx} ln ${q} + ${x} ln ${ph} = ${l1}`) + `; LR_uc = ${lr} ${rej ? '>' : '<'} 3.84 (p = ${p}): ${rej ? 'rejected' : 'not rejected'}; ${ZN[z]} zone.`,
 q6: (V, E, hv, he, x, lr, p, rej, z, xs) => `Normal: VaR 1% ${V}%, ES 2.5% ${E}%; historical: VaR 1% ${hv}%, ES 2.5% ${he}%; ${x} exceptions: LR_uc = ${lr} (p = ${p}, ${rej ? 'rejected' : 'not rejected'}), ${ZN[z].split(' ')[0]} zone${xs ? XS : ''}.`,
 q7a: (b, a, r, lg, H, d) => `H = log₁₀(${b}/${a})/log₁₀(2000/20) = log₁₀(${r})/2 = ${lg}/2 = ${H}; d = H − 0.5 = ${d}.`,
 q7b: (h, H, fh, v1, Vh, fs, Vs, gt) => `${h}^H = ${h}^${H} = ${fh}: VaR = ${v1} × ${fh} = ${Vh}%; √${h} = ${fs}: VaR = ${Vs}%. ` + (gt ? 'With H > 0.5 the square-root-of-time rule understates multi-day risk.' : 'With H < 0.5 the square-root-of-time rule overstates multi-day risk.'),
 q7c: () => 'Before concluding: a Monte Carlo band of Ĥ for i.i.d. data of the same length (R/S is biased upwards), the Lo test, split samples and shuffled data (breaks, GARCH).',
 q7: (H, d, h, Vh, Vs, x) => `H = ${H}, d = ${d}; ${h}-day VaR: ${Vh}% with h^H, ${Vs}% with √h${x ? XS : ''}.`,
 q8a: (TP, TN, acc, tpr, FP, fpr, fl, pr, ag, gt) => `Accuracy (${TP} + ${TN})/300 = ${acc}%; TPR = ${TP}/90 = ${tpr}%; FPR = ${FP}/210 = ${fpr}%; precision ${pr === null ? 'not defined (no loan flagged)' : `${TP}/${fl} = ${pr}%`}; the rule “always good”: ${ag}%` + (gt ? '.' : ': the model is less accurate than this rule.'),
 q8b: () => 'Gini = 2 × 0.80 − 1 = 0.60; EL = 0.04 × 0.45 × 200,000 = 3,600 RON.',
 q8c: (b, cv, cm, dc, df) => `CoVaR = −(−2.1 + ${b} × (−5.0)) = ${cv}%; at the median: −(−2.1 + ${b} × 0.1) = ${cm}%; ΔCoVaR = ${b} × ${df} = ${dc} pp.`,
 q8: (acc, tpr, fpr, pr, dc, x) => `Accuracy ${acc}%, TPR ${tpr}%, FPR ${fpr}%, precision ${pr === null ? 'undefined' : pr + '%'}; ΔCoVaR ${dc} pp${x ? XS : ''}.`,
};
})();
""",
}

QA = {
 'ro': {
  'three': ('▶ mută începutul ferestrei în ianuarie 2019 (rulare suplimentară). Care dintre cele trei drawdown-uri maxime se schimbă?',
            'Doar cel al Bitcoin: −76,6% (de la 8 noiembrie 2021 la 21 noiembrie 2022) în loc de −83,4% (decembrie 2017 – decembrie 2018). S&P 500 (−33,9%) și BET (−31,1%) '
            'își păstrează minimul din 23 martie 2020, care rămîne în fereastră. Drawdown-ul maxim depinde de eșantion: el măsoară cea mai mare cădere observată, nu o '
            'proprietate fixă a activului.'),
  'sum-tab': ('Cu fereastra din ianuarie 2021 (▶, rulare suplimentară), mai respinge testul Ljung–Box Q(10) absența autocorelației randamentelor S&P 500?',
              'Nu: Q(10) = 9,1 < 18,31 (p = 0,52), față de 236 pe fereastra din 2015, în timp ce pe rₜ² rămîne 291. Statistica mare din tabel vine în bună parte din '
              'zilele din martie 2020, cînd randamente mari de semne opuse au venit una după alta (ρ̂₁ = −0,13); volatility clustering se vede în orice fereastră.'),
  'interp': ('▶ trece la nivelul α = 5%. Mai este VaR Normal prea mic față de VaR istoric?',
             'Nu: la 5%, VaR Normal este prea mare, cu 7,8% (S&P 500: 1,79% față de 1,66%), 13,1% (BET) și 6,4% (Bitcoin). Distribuția randamentelor are mai multă masă '
             'în centru și în cozile extreme decît distribuția Normală cu aceeași varianță, așa că cele două cuantile se încrucișează între 1% și 5% (la 2,5%, pentru BET, '
             'diferența este −0,5%). VaR Normal subestimează riscul tocmai la nivelurile mici, cele folosite în reglementare.'),
  'stylised': ('Cu fereastra din ianuarie 2021 (▶, rulare suplimentară), ce devine ρ̂₁(|r|) pentru S&P 500?',
               'Scade la 0,16, de la 0,36, pentru că zilele din martie 2020 au ieșit din eșantion; toate cele 50 de autocorelații ale lui |r| rămîn totuși pozitive. '
               'Mărimea autocorelației depinde de crizele din fereastră, semnul ei pozitiv persistent nu.'),
  'perf': ('Cu fereastra din ianuarie 2021 (▶, rulare suplimentară), care rapoarte Sharpe au un interval de 95% care conține 0?',
           'S&P 500 (0,85, CI [−0,10; 1,81]) și Bitcoin (0,60, CI [−0,29; 1,49]); BET (1,53, CI [0,32; 2,74]) nu. Cu circa 5,7 ani de date, eroarea standard '
           '√((1 + SR²/2)/Y) este de aproape 0,5: chiar un Sharpe de 0,85 nu se separă de zero.'),
  'agg': ('▶ trece la BET. Crește α̂ de la randamentele zilnice la cele lunare, ca la S&P 500 (1,54 → 1,82)?',
          'Nu clar: 1,49 zilnic, 1,56 săptămînal, 1,50 lunar, iar intervalul lunar [1,33; 1,67] este larg; BET este excepția de pe slide. Varianța de selecție '
          'cumulativă a BET scade totuși de la 3,32 (2010) la 1,96 (2026) și se stabilizează, cum se întîmplă cu o varianță finită. Pentru Bitcoin, α̂ lunar atinge '
          'frontiera α = 2.'),
  'backtest': ('Cu o fereastră de 250 de zile pentru simularea istorică (▶, rulare suplimentară), mai acceptă testul Kupiec VaR 1% al BET?',
               'Nu: 38 de depășiri (1,57%, așteptat 24,2), LR = 6,76, p = 0,009. Cu 250 de zile, VaR 1% este a treia cea mai mică valoare din fereastră și coada este '
               'estimată din foarte puține observații; cu 500 de zile sînt 28 de depășiri (p = 0,450), cu 750 de zile 26.'),
  'p1': ('▶ schimbă valoarea din anul 2. Dacă indicele coboară doar la 120 (rulare cu alte valori), ce devine CAGR?',
         'Rămîne −1,35%: CAGR depinde doar de prima și de ultima valoare, (96/100)^(1/3) − 1. Se schimbă media aritmetică, (60 − 25 − 20)/3 = 5,00%, și drawdown-ul '
         'maxim, 96/160 − 1 = −40%.'),
  'p2': ('▶ schimbă excesul de boltire. Dacă K scade de la 6 la 3, ce se întîmplă cu componenta boltirii din JB?',
         'Scade la un sfert: 1000/24 × 9 = 375, față de 1500, pentru că K intră la pătrat; JB = 41,67 + 375 = 416,7, tot mult peste 5,99. Gradele de libertate ale '
         'distribuției Student-t cu același K cresc la ν = 4 + 6/3 = 6.'),
  'p3': ('▶ trece prin corelații. Cît este volatilitatea zilnică a portofoliului la ρ = −1?',
         '0,10%: σp² = 0,81 + 1,00 − 1,80 = 0,01. Cu ρ = −1 riscul nu dispare complet, pentru că cele două poziții nu se compensează exact: 0,6 × 1,5 = 0,9 față de '
         '0,4 × 2,5 = 1,0. La ρ = 1, σp = 1,90%, egală cu media ponderată.'),
  'p4': ('▶ schimbă ρ̂(1). La ρ̂(1) = 0,09, respinge testul robust Z*(5) mersul aleator?',
         'Da: VR(5) = 1 + 2 × 0,082 = 1,164 și Z*(5) = 0,164/0,065 = 2,52 > 1,96 (Z(5) = 3,74, Q(4) = 21,78). Cu ρ̂(1) = 0,06 diferența dintre cele două teste '
         'decidea rezultatul; aici autocorelația este destul de mare ca să fie respinsă și cu eroarea standard robustă.'),
  'p5': ('▶ schimbă semnul și mărimea șocului. Cu εₜ = +2,5 în loc de −2,5, cît devine σ²ₜ₊₁?',
         'Tot 2,1945: GARCH(1,1) folosește ε², deci un cîștig și o pierdere de aceeași mărime cresc varianța la fel. Asimetria (leverage effect) cere modele ca GJR '
         'sau EGARCH.'),
  'p6': ('▶ crește numărul de depășiri. Ce apare primul: zona galbenă Basel sau respingerea Kupiec la 5%?',
         'Zona galbenă, de la 5 depășiri; pe măsură ce depășirile cresc, testul Kupiec respinge abia de la 7 (LR = 5,50, p = 0,019; cu 6, LR = 3,56 și p = 0,059). '
         'În 250 de zile testul are putere mică, de aceea semaforul Basel penalizează mai devreme. Testul respinge și la 0 depășiri (LR = 5,03): prea puține depășiri '
         'arată un VaR prea prudent.'),
  'p7': ('▶ schimbă R/S pentru n = 2000. Dacă este 40 în loc de 60, mai dă scalarea cu h^H un VaR pe 10 zile mai mare decît √h?',
         'Nu: H = log₁₀(40/4,1)/2 = 0,495 < 0,5, deci 10^H = 3,12 și VaR = 7,81%, puțin sub 7,91% cu √10. Panta, nu creșterea lui R/S cu n, decide: R/S crește și '
         'pentru date fără memorie, cu exponentul 0,5.'),
  'p8': ('▶ crește numărul creditelor performante marcate ca neperformante. Cu 120 (rulare cu alte valori), care rată nu se schimbă?',
         'TPR, 60/90 = 66,7%: ea privește doar creditele neperformante. Acuratețea scade la (60 + 90)/300 = 50%, sub regula „mereu bun” (70%), FPR crește la 57,1%, '
         'iar precizia scade la 60/180 = 33,3%.'),
 },
 'en': {
  'three': ('▶ moves the start of the window to January 2019 (extra run). Which of the three maximum drawdowns changes?',
            'Only Bitcoin’s: −76.6% (from 8 November 2021 to 21 November 2022) instead of −83.4% (December 2017 – December 2018). The S&P 500 (−33.9%) and the BET '
            '(−31.1%) keep their trough of 23 March 2020, which stays in the window. The maximum drawdown depends on the sample: it measures the largest fall observed, not '
            'a fixed property of the asset.'),
  'sum-tab': ('With the window from January 2021 (▶, extra run), does the Ljung–Box Q(10) test still reject no autocorrelation in the S&P 500 returns?',
              'No: Q(10) = 9.1 < 18.31 (p = 0.52), against 236 on the window from 2015, while on rₜ² it stays at 291. The large statistic of the table comes largely from '
              'the days of March 2020, when large returns of opposite signs followed each other (ρ̂₁ = −0.13); volatility clustering shows in every window.'),
  'interp': ('▶ moves to the level α = 5%. Is the Normal VaR still too low against the historical VaR?',
             'No: at 5% the Normal VaR is too high, by 7.8% (S&P 500: 1.79% against 1.66%), 13.1% (BET) and 6.4% (Bitcoin). The return distribution has more mass in the '
             'centre and in the far tails than the Normal distribution with the same variance, so the two quantiles cross between 1% and 5% (at 2.5% the BET difference is '
             '−0.5%). The Normal VaR understates risk exactly at the small levels used in regulation.'),
  'stylised': ('With the window from January 2021 (▶, extra run), what does ρ̂₁(|r|) of the S&P 500 become?',
               'It falls to 0.16 from 0.36, because the days of March 2020 left the sample; all 50 autocorrelations of |r| stay positive. The size of the autocorrelation '
               'depends on the crises in the window, its persistent positive sign does not.'),
  'perf': ('With the window from January 2021 (▶, extra run), which Sharpe ratios have a 95% interval that contains 0?',
           'The S&P 500 (0.85, CI [−0.10, 1.81]) and Bitcoin (0.60, CI [−0.29, 1.49]); the BET (1.53, CI [0.32, 2.74]) does not. With about 5.7 years of data the '
           'standard error √((1 + SR²/2)/Y) is close to 0.5: even a Sharpe ratio of 0.85 is not separated from zero.'),
  'agg': ('▶ moves to the BET. Does α̂ rise from daily to monthly returns, as for the S&P 500 (1.54 → 1.82)?',
          'Not clearly: 1.49 daily, 1.56 weekly, 1.50 monthly, and the monthly interval [1.33, 1.67] is wide; the BET is the exception of the slide. The running sample '
          'variance of the BET still falls from 3.32 (2010) to 1.96 (2026) and settles down, as a finite variance does. For Bitcoin the monthly α̂ reaches the bound α = 2.'),
  'backtest': ('With a 250-day historical-simulation window (▶, extra run), does the Kupiec test still accept the VaR 1% of the BET?',
               'No: 38 exceptions (1.57%, 24.2 expected), LR = 6.76, p = 0.009. With 250 days the VaR 1% is the third smallest value of the window, so the tail is '
               'estimated from very few observations; with 500 days there are 28 exceptions (p = 0.450), with 750 days 26.'),
  'p1': ('▶ changes the value of year 2. If the index falls only to 120 (values other than on the slide), what does the CAGR become?',
         'It stays −1.35%: the CAGR depends only on the first and the last value, (96/100)^(1/3) − 1. The arithmetic mean changes, (60 − 25 − 20)/3 = 5.00%, and so does '
         'the maximum drawdown, 96/160 − 1 = −40%.'),
  'p2': ('▶ changes the excess kurtosis. If K falls from 6 to 3, what happens to the kurtosis part of JB?',
         'It falls to a quarter: 1000/24 × 9 = 375 against 1500, because K enters squared; JB = 41.67 + 375 = 416.7, still far above 5.99. The degrees of freedom of '
         'the Student-t distribution with the same K rise to ν = 4 + 6/3 = 6.'),
  'p3': ('▶ moves through the correlations. What is the daily volatility of the portfolio at ρ = −1?',
         '0.10%: σp² = 0.81 + 1.00 − 1.80 = 0.01. With ρ = −1 the risk does not vanish completely, because the two positions do not offset exactly: 0.6 × 1.5 = 0.9 '
         'against 0.4 × 2.5 = 1.0. At ρ = 1, σp = 1.90%, equal to the weighted average.'),
  'p4': ('▶ changes ρ̂(1). At ρ̂(1) = 0.09, does the robust test Z*(5) reject the random walk?',
         'Yes: VR(5) = 1 + 2 × 0.082 = 1.164 and Z*(5) = 0.164/0.065 = 2.52 > 1.96 (Z(5) = 3.74, Q(4) = 21.78). With ρ̂(1) = 0.06 the choice of test decided the '
         'outcome; here the autocorrelation is large enough to be rejected with the robust standard error too.'),
  'p5': ('▶ changes the sign and the size of the shock. With εₜ = +2.5 instead of −2.5, what is σ²ₜ₊₁?',
         'Still 2.1945: GARCH(1,1) uses ε², so a gain and a loss of the same size raise the variance alike. Asymmetry (the leverage effect) needs models such as GJR '
         'or EGARCH.'),
  'p6': ('▶ raises the number of exceptions. What comes first: the Basel yellow zone or the Kupiec rejection at 5%?',
         'The yellow zone, from 5 exceptions; as the exceptions grow, the Kupiec test rejects only from 7 (LR = 5.50, p = 0.019; with 6, LR = 3.56 and p = 0.059). '
         'In 250 days the test has little power, which is why the Basel traffic light penalises earlier. The test also rejects at 0 exceptions (LR = 5.03): too few '
         'exceptions point to a VaR that is too cautious.'),
  'p7': ('▶ changes R/S for n = 2000. If it is 40 instead of 60, does the h^H scaling still give a larger 10-day VaR than √h?',
         'No: H = log₁₀(40/4.1)/2 = 0.495 < 0.5, so 10^H = 3.12 and VaR = 7.81%, slightly below 7.91% with √10. The slope decides, not the growth of R/S with n: '
         'R/S grows for data without memory as well, with exponent 0.5.'),
  'p8': ('▶ raises the number of good loans flagged as bad. With 120 (values other than on the slide), which rate does not change?',
         'The TPR, 60/90 = 66.7%: it involves only the bad loans. The accuracy falls to (60 + 90)/300 = 50%, below the rule “always good” (70%), the FPR rises to 57.1% '
         'and the precision falls to 60/180 = 33.3%.'),
 },
}

PREDICT = {
 'three': (0, [('Doar al Bitcoin.', 'Only Bitcoin’s.'), ('Toate trei scad: o fereastră mai scurtă dă drawdown-uri mai mici.', 'All three fall: a shorter window gives smaller drawdowns.'),
               ('Niciunul: drawdown-ul maxim este o proprietate a activului.', 'None: the maximum drawdown is a property of the asset.')]),
 'sum-tab': (0, [('Nu, Q(10) scade sub 18,31.', 'No, Q(10) falls below 18.31.'), ('Da, Q(10) rămîne aproape de 236: este aceeași serie.', 'Yes, Q(10) stays near 236: it is the same series.'),
                 ('Da, Q(10) crește: un eșantion mai scurt dă statistici mai mari.', 'Yes, Q(10) rises: a shorter sample gives larger statistics.')]),
 'interp': (0, [('Nu, devine prea mare.', 'No, it becomes too high.'), ('Da, tot cu circa 20%: cozile groase contează la orice nivel.', 'Yes, still by about 20%: heavy tails matter at every level.'),
                ('Da, și cu mai mult: diferența crește cu α.', 'Yes, and by more: the gap grows with α.')]),
 'stylised': (0, [('Scade la circa 0,16, dar rămîne pozitiv la toate cele 50 de laguri.', 'It falls to about 0.16 but stays positive at all 50 lags.'),
                  ('Rămîne 0,36: un fapt stilizat nu depinde de fereastră.', 'It stays 0.36: a stylised fact does not depend on the window.'),
                  ('Devine negativ: fără crize nu mai există volatility clustering.', 'It turns negative: without crises there is no volatility clustering.')]),
 'perf': (0, [('S&P 500 și Bitcoin.', 'The S&P 500 and Bitcoin.'), ('Niciunul: toate trei sînt pozitive.', 'None: all three are positive.'),
              ('Doar Bitcoin, cel mai volatil.', 'Only Bitcoin, the most volatile.')]),
 'agg': (0, [('Nu clar: 1,49, 1,56, 1,50.', 'Not clearly: 1.49, 1.56, 1.50.'), ('Da, spre 2: la orice serie agregarea duce la distribuția Normală.', 'Yes, towards 2: for every series aggregation leads to the Normal distribution.'),
             ('Nu, rămîne exact 1,49: o lege stabilă își păstrează α.', 'No, it stays exactly 1.49: a stable law keeps its α.')]),
 'backtest': (0, [('Nu, 38 de depășiri, p ≈ 0,009.', 'No, 38 exceptions, p ≈ 0.009.'), ('Da, tot 28: fereastra nu schimbă VaR 1%.', 'Yes, still 28: the window does not change VaR 1%.'),
                  ('Da, cu mai puține depășiri: o fereastră scurtă reacționează mai repede.', 'Yes, with fewer exceptions: a short window reacts faster.')]),
 'p1': (0, [('Rămîne −1,35%.', 'It stays −1.35%.'), ('Crește: scăderea din anul 2 este mai mică.', 'It rises: the fall of year 2 is smaller.'),
            ('Devine 5%, media aritmetică a randamentelor.', 'It becomes 5%, the arithmetic mean of the returns.')]),
 'p2': (0, [('Scade la un sfert, 375.', 'It falls to a quarter, 375.'), ('Scade la jumătate, 750.', 'It falls by half, to 750.'), ('Nu se schimbă: n este același.', 'It does not change: n is the same.')]),
 'p3': (0, [('0,10%.', '0.10%.'), ('Exact 0: corelația −1 elimină orice risc.', 'Exactly 0: a correlation of −1 removes all risk.'), ('1,35%, ca fără termenul de covarianță.', '1.35%, as without the covariance term.')]),
 'p4': (0, [('Da, Z*(5) ≈ 2,52.', 'Yes, Z*(5) ≈ 2.52.'), ('Nu: testul robust nu respinge cînd există volatility clustering.', 'No: the robust test does not reject when there is volatility clustering.'),
            ('Nu: VR(5) = 1,164 este aproape de 1.', 'No: VR(5) = 1.164 is close to 1.')]),
 'p5': (0, [('Tot 2,1945.', 'Still 2.1945.'), ('Mai mică: un cîștig scade volatilitatea (leverage effect).', 'Smaller: a gain lowers volatility (the leverage effect).'),
            ('Mai mare: șocul pozitiv se adună la varianță.', 'Larger: the positive shock adds to the variance.')]),
 'p6': (0, [('Zona galbenă, de la 5 depășiri.', 'The yellow zone, from 5 exceptions.'), ('Respingerea Kupiec, de la 3 depășiri.', 'The Kupiec rejection, from 3 exceptions.'),
            ('Amîndouă la 5 depășiri: folosesc același prag.', 'Both at 5 exceptions: they use the same threshold.')]),
 'p7': (0, [('Nu, H ≈ 0,495 < 0,5.', 'No, H ≈ 0.495 < 0.5.'), ('Da: R/S crește cu n, deci H > 0,5.', 'Yes: R/S grows with n, so H > 0.5.'),
            ('La fel: pentru randamente H este mereu 0,5.', 'The same: for returns H is always 0.5.')]),
 'p8': (0, [('TPR.', 'The TPR.'), ('Precizia: depinde doar de creditele neperformante.', 'The precision: it depends only on the bad loans.'),
            ('Acuratețea: modelul găsește tot 60 de credite neperformante.', 'The accuracy: the model still finds 60 bad loans.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout (all questions ask what ▶ will do)
VEIL = {a: {'out': True} for a in PREDICT}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Idei de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Ljung–Box nu respinge pe rₜ, dar respinge puternic pe rₜ². Ce înseamnă acest lucru?', 'Ljung–Box does not reject on rₜ but rejects strongly on rₜ². What does this mean?'),
      key=0, opts=[
  (('Randamentele sînt necorelate, dar nu independente: volatility clustering; modelăm varianța.', 'Returns are uncorrelated but not independent: volatility clustering; we model the variance.'),
   ('Mărimea randamentelor este previzibilă, semnul nu; în curs, Q(10) pe rₜ² este de 13 ori mai mare decît pe rₜ la S&P 500 și de 10 ori la Bitcoin.', 'The size of returns is predictable, the sign is not; in the lecture Q(10) on rₜ² is 13 times that on rₜ for the S&P 500 and 10 times for Bitcoin.')),
  (('Randamentele sînt independente; testul pe rₜ² este o alarmă falsă cauzată de cozile groase.', 'Returns are independent; the test on rₜ² is a false alarm caused by heavy tails.'),
   ('Necorelat nu înseamnă independent: ACF-ul lui |r| este pozitiv timp de zeci de laguri, ceea ce exclude independența.', 'Uncorrelated is not independent: the ACF of |r| is positive for dozens of lags, which rules out independence.')),
  (('Randamentele sînt previzibile ca semn: există momentum.', 'Returns are predictable in sign: there is momentum.'),
   ('Testul pe rₜ nu respinge; predictibilitatea privește pătratele, adică mărimea, nu direcția.', 'The test on rₜ does not reject; the predictability concerns the squares, the size, not the direction.'))]),
 dict(q=('O lege stabilă estimată dă α̂ = 1,5, iar estimarea Hill este α̂ = 3. Care are dreptate în privința varianței?', 'A fitted stable law gives α̂ = 1.5 and the Hill estimate is α̂ = 3. Which is right about the variance?'),
      key=0, opts=[
  (('Estimarea Hill: varianța este finită; legea stabilă descrie corpul distribuției.', 'The Hill estimate: the variance is finite; the stable law describes the body of the distribution.'),
   ('Hill privește doar coada, unde se decide existența momentelor; în plus, α̂ stabil crește cu orizontul (S&P 500: 1,82 lunar), ceea ce nu se întîmplă pentru o lege stabilă.', 'Hill looks only at the tail, where the existence of moments is decided; besides, the stable α̂ rises with the horizon (S&P 500: 1.82 monthly), which a stable law would not do.')),
  (('Legea stabilă: folosește toate datele, deci este mai sigură.', 'The stable law: it uses all the data, so it is more reliable.'),
   ('Tocmai pentru că folosește toate datele, estimarea stabilă este dominată de corpul distribuției; varianța de selecție a datelor se stabilizează, ca la o varianță finită.', 'Precisely because it uses all the data, the stable estimate is driven by the body of the distribution; the sample variance of the data settles down, as for a finite variance.')),
  (('Niciuna: varianța există doar dacă α > 4.', 'Neither: the variance exists only if α > 4.'),
   ('Confuzie între varianță (α > 2) și boltire (α > 4); cu α = 3 există media, varianța și asimetria.', 'A confusion between the variance (α > 2) and the kurtosis (α > 4); with α = 3 the mean, the variance and the skewness exist.'))]),
 dict(q=('Un model VaR 1% are 1% depășiri, toate în martie 2020. Este un model bun?', 'A VaR 1% model has 1% exceptions, all in March 2020. Is it a good model?'),
      key=0, opts=[
  (('Nu: rata este corectă, dar depășirile sînt grupate; testul Christoffersen îl respinge.', 'No: the rate is right but the exceptions cluster; the Christoffersen test rejects it.'),
   ('Depășirile unui model bun sînt independente în timp; gruparea arată că VaR nu urmărește volatilitatea.', 'The exceptions of a good model are independent over time; the clustering shows that the VaR does not follow volatility.')),
  (('Da: testul Kupiec acceptă o rată de 1%.', 'Yes: the Kupiec test accepts a rate of 1%.'),
   ('Kupiec verifică doar numărul depășirilor, nu și momentul lor; în curs, HS pentru BET are rata corectă, dar pînă la 10 depășiri în 250 de zile.', 'Kupiec checks only the number of exceptions, not their timing; in the lecture HS for the BET has the right rate but up to 10 exceptions in 250 days.')),
  (('Da: martie 2020 a fost un eveniment excepțional, care nu se ia în calcul.', 'Yes: March 2020 was an exceptional event that does not count.'),
   ('Tocmai zilele de criză sînt cele pentru care se calculează VaR; un model care le ratează pe toate odată nu protejează capitalul.', 'Crisis days are exactly what VaR is computed for; a model that misses them all at once does not protect capital.'))]),
 dict(q=('Cuantila de 1% a randamentelor zilnice ale S&P 500 este −3,29%. Cît este VaR 1%?', 'The 1% quantile of the daily S&P 500 returns is −3.29%. What is VaR 1%?'),
      key=0, opts=[
  (('3,29%.', '3.29%.'), ('VaR₁% = −q₁%: o pierdere, cu semn pozitiv.', 'VaR₁% = −q₁%: a loss, with a positive sign.')),
  (('−3,29%.', '−3.29%.'), ('Semnul greșit: în curs VaR este pierderea, deci pozitiv.', 'The wrong sign: in the course VaR is the loss, so positive.')),
  (('Cuantila de 99% a randamentelor.', 'The 99% quantile of the returns.'), ('Cuantila de 99% descrie coada cîștigurilor; VaR 1% vine din coada pierderilor, nivelul fiind probabilitatea cozii.', 'The 99% quantile describes the tail of gains; VaR 1% comes from the tail of losses, the level being the tail probability.'))]),
 dict(q=('Tail index-ul pierderilor este între 2 și 4. Ce momente ale randamentelor există?', 'The tail index of the losses is between 2 and 4. Which moments of the returns exist?'),
      key=0, opts=[
  (('Media și varianța; boltirea nu este garantată.', 'The mean and the variance; the kurtosis is not guaranteed.'),
   ('Momentul de ordin p există pentru p < α; cu α între 2 și 4, varianța este finită, boltirea (p = 4) nu.', 'The moment of order p exists for p < α; with α between 2 and 4 the variance is finite, the kurtosis (p = 4) is not.')),
  (('Doar media: cu cozi groase varianța este infinită.', 'Only the mean: with heavy tails the variance is infinite.'),
   ('Varianța este infinită doar pentru α < 2; cozile groase cu α între 2 și 4 au varianță finită.', 'The variance is infinite only for α < 2; heavy tails with α between 2 and 4 have a finite variance.')),
  (('Toate: randamentele zilnice sînt mărginite.', 'All of them: daily returns are bounded.'),
   ('Un tail index finit înseamnă o coadă de tip putere; momentele de ordin p ≥ α nu există.', 'A finite tail index means a power-law tail; the moments of order p ≥ α do not exist.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch16-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH16_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH16_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH16_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
