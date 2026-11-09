"""Build the interactive pages of Chapter 14: interactive/ch14.html (EN) and interactive/ch14_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch14.js, inputs in interactive/ch14_data.js (Quantlets/Ch_14/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch14.js moves each one to its
own frame. The RO and EN decks have the same frames: XBLOCKS give the frame number, checked against TITLES and TITLES_EN.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 14
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_14'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('halving', 'sfm_ch14_halving', 'btc'), ('rvol', 'sfm_ch14_rolling_vol', 'asset'), ('weekday', 'sfm_ch14_weekday', 'asset'),
          ('acf', 'sfm_ch14_acf', 'tails'), ('garch', 'sfm_ch14_garch_vol', 'tails'), ('rvr', 'sfm_ch14_rolling_vr', 'eff'),
          ('corr', 'sfm_ch14_rolling_corr', 'corr'), ('dd', 'sfm_ch14_drawdowns', 'bub'), ('etf', 'sfm_ch14_etf', 'etf'),
          ('supply', 'sfm_ch14_stablecoin_supply', 'stable'), ('peg', 'sfm_ch14_peg', 'stable'), ('svb', 'sfm_ch14_svb', 'stable'),
          ('ust', 'sfm_ch14_ust', 'stable'), ('btcvar', 'sfm_ch14_btc_var', 'risk')]
# table and worked-example blocks: (anchor, frame number in the deck, section key, chart block that carries it on the web page)
XBLOCKS = [('ann-we', 16, 'asset', 'rvol'), ('desc-tab', 17, 'asset', 'rvol'), ('hill-tab', 23, 'tails', 'acf'), ('lb-we', 24, 'tails', 'acf'),
           ('garch-tab', 27, 'tails', 'garch'), ('vr-tab', 31, 'eff', 'rvr'), ('rc-we', 36, 'corr', 'corr'), ('corr-tab', 38, 'corr', 'corr'),
           ('crisis-tab', 39, 'corr', 'corr'), ('haven-we', 40, 'corr', 'corr'), ('mdd-we', 41, 'bub', 'dd'), ('dd-tab', 43, 'bub', 'dd'),
           ('etf-tab', 47, 'etf', 'etf'), ('ibit-we', 48, 'etf', 'etf'), ('bp-we', 52, 'stable', 'peg'), ('usd-we', 61, 'risk', 'btcvar'),
           ('var-tab', 62, 'risk', 'btcvar'), ('sqrt-we', 63, 'risk', 'btcvar'), ('kup-we', 64, 'risk', 'btcvar')]
ORDER = ['halving', 'ann-we', 'desc-tab', 'rvol', 'weekday', 'hill-tab', 'lb-we', 'acf', 'garch-tab', 'garch', 'vr-tab', 'rvr', 'rc-we', 'corr', 'corr-tab',
         'crisis-tab', 'haven-we', 'mdd-we', 'dd', 'dd-tab', 'etf', 'etf-tab', 'ibit-we', 'supply', 'bp-we', 'peg', 'svb', 'ust', 'usd-we', 'var-tab',
         'sqrt-we', 'kup-we', 'btcvar']
TITLES = {'ann-we': 'Anualizarea cu frecvența reală (2/2)', 'desc-tab': 'Compararea a cinci active', 'hill-tab': 'Cozi groase: estimatorul Hill (2/2)',
          'lb-we': 'Interpretarea ACF și a testului Ljung–Box', 'garch-tab': 'GARCH(1,1)-t pentru active cripto și tradiționale (2/2)',
          'vr-tab': 'Testele raportului varianțelor pe întregul eșantion', 'rc-we': 'Măsurarea corelației între calendare diferite', 'corr-tab': 'Corelația an cu an',
          'crisis-tab': 'Trei episoade de criză', 'haven-we': 'Bitcoin ca instrument de acoperire sau activ de refugiu', 'mdd-we': 'Drawdown-ul', 'dd-tab': 'Marile drawdown-uri ale Bitcoin',
          'etf-tab': 'Bitcoin înainte și după lansarea ETF-urilor spot', 'ibit-we': 'Eroarea de replicare a unui ETF spot pe Bitcoin', 'bp-we': 'Abaterea de la paritate în puncte de bază',
          'usd-we': 'VaR 1% și ES 2,5% în dolari', 'var-tab': 'VaR 1% și ES 2,5% prin patru metode', 'sqrt-we': 'Interpretarea cifrelor de risc', 'kup-we': 'Backtesting: testul Kupiec'}
TITLES_EN = {'ann-we': 'Annualisation with the Actual Frequency (2/2)', 'desc-tab': 'Comparison of Five Assets', 'hill-tab': 'Heavy Tails: the Hill Estimator (2/2)',
             'lb-we': 'Reading the ACF and the Ljung–Box Test', 'garch-tab': 'GARCH(1,1)-t for Crypto and Traditional Assets (2/2)',
             'vr-tab': 'Variance-Ratio Tests on the Whole Sample', 'rc-we': 'Measuring Correlation Across Calendars', 'corr-tab': 'Correlation Year by Year',
             'crisis-tab': 'Three Crisis Episodes', 'haven-we': 'Bitcoin as a Hedge or a Safe Haven', 'mdd-we': 'Drawdown', 'dd-tab': "Bitcoin's Large Drawdowns",
             'etf-tab': 'Bitcoin before and after the Spot ETFs', 'ibit-we': 'Tracking Error of a Spot Bitcoin ETF', 'bp-we': 'Peg Deviation in Basis Points',
             'usd-we': 'VaR 1% and ES 2.5% in Dollars', 'var-tab': 'VaR 1% and ES 2.5% under Four Methods', 'sqrt-we': 'Interpretation of the Risk Numbers', 'kup-we': 'Backtesting: the Kupiec Test'}

PFX = {'halving': 'hv', 'ann-we': 'an', 'desc-tab': 'ds', 'rvol': 'rv', 'weekday': 'wd', 'hill-tab': 'hl', 'lb-we': 'lb', 'acf': 'ac', 'garch-tab': 'gt', 'garch': 'gv',
       'vr-tab': 'vt', 'rvr': 'vr', 'rc-we': 'rc', 'corr': 'co', 'corr-tab': 'cy', 'crisis-tab': 'cr', 'haven-we': 'sh', 'mdd-we': 'md', 'dd': 'dd', 'dd-tab': 'de',
       'etf': 'ef', 'etf-tab': 'et', 'ibit-we': 'ib', 'supply': 'su', 'bp-we': 'bp', 'peg': 'pg', 'svb': 'sv', 'ust': 'us', 'usd-we': 'ud', 'var-tab': 'vq',
       'sqrt-we': 'sq', 'kup-we': 'kp', 'btcvar': 'bt'}

SECTIONS = {
 'ro': dict(btc='Bitcoin și Ethereum pe scurt', asset='Activele cripto ca clasă de active', tails='Cozi și volatility clustering', eff='Eficiența pieței cripto',
            corr='Corelația cu acțiunile și aurul', bub='Bule și crahuri', etf='ETF-urile spot pe Bitcoin', stable='Stablecoins', risk='Riscul unei poziții cripto'),
 'en': dict(btc='Bitcoin and Ethereum in Brief', asset='Crypto as an Asset Class', tails='Tails and Volatility Clustering', eff='Efficiency of the Crypto Market',
            corr='Correlation with Equities and Gold', bub='Bubbles and Crashes', etf='Spot Bitcoin ETFs', stable='Stablecoins', risk='The Risk of a Crypto Position'),
}

L = {
 'ro': dict(file='ch14_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 14 · Active cripto', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: BTC (Bitcoin), ETF (exchange-traded fund, fond tranzacționat la bursă), IBIT (iShares Bitcoin '
        'Trust), USDT (Tether), USDC (USD Coin), DAI (Dai), UST (TerraUSD), SVB (Silicon Valley Bank), FDIC (Federal Deposit Insurance Corporation), ACF (funcția de '
        'autocorelație), VR (raportul varianțelor), HAC (erori standard robuste la heteroscedasticitate și autocorelație), GARCH (generalised autoregressive conditional '
        'heteroskedasticity), GJR (modelul GARCH asimetric Glosten–Jagannathan–Runkle), IGARCH (GARCH integrat), AR(1) (proces autoregresiv de ordinul 1), CI (intervalul de '
        'încredere), HS (simulare istorică), VaR (value at risk), ES (expected shortfall), TE (tracking error), bp (puncte de bază), i.i.d. (independente și identic distribuite).',
   src='Datele cursului din data/market (pînă la 18 septembrie 2026): închiderile zilnice ale Bitcoin, Ethereum, Solana, S&P 500, aurului (XAU/USD) și ale ETF-ului IBIT; '
       'închiderile și minimele zilnice ale USDT, USDC și DAI din 2021. Oferta de stablecoins și seriile TerraUSD vin de la DefiLlama, descărcate cu funcția capitolului '
       'pe 4 noiembrie 2026 și tăiate la 18 septembrie 2026; unde descărcarea diferă de valorile de pe slide-uri, pagina le arată pe amîndouă. Estimațiile GARCH(1,1)-t și GJR, '
       'volatilitățile condiționate, ajustările Student-t și prognozele GARCH-t ale backtesting-ului vin din codul capitolului (Quantlets/Ch_14). Browserul recalculează din '
       'prețuri randamentele, statisticile descriptive, indicii Hill, ACF și testul Ljung–Box, rapoartele varianțelor, corelațiile, drawdown-urile, testele de weekend și de '
       'refugiu, abaterile de la paritate, VaR, ES și testul Kupiec. Alte ferestre, praguri, orizonturi, perioade și niveluri α sînt rulări suplimentare ale aceluiași cod; '
       'persistența GARCH pe ferestre ETF de 1, 1,5 și 2,5 ani este o reestimare suplimentară cu codul capitolului, iar prognozele GARCH pe mai multe zile și partea rămasă '
       'dintr-un șoc sînt calcule suplimentare din estimațiile de pe slide.'),
 'en': dict(file='ch14.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 14 · Crypto Assets', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: BTC (Bitcoin), ETF (exchange-traded fund), IBIT (iShares Bitcoin Trust), USDT (Tether), '
        'USDC (USD Coin), DAI (Dai), UST (TerraUSD), SVB (Silicon Valley Bank), FDIC (Federal Deposit Insurance Corporation), ACF (autocorrelation function), VR (variance '
        'ratio), HAC (heteroskedasticity- and autocorrelation-consistent standard errors), GARCH (generalised autoregressive conditional heteroskedasticity), GJR (the '
        'asymmetric GARCH of Glosten, Jagannathan and Runkle), IGARCH (integrated GARCH), AR(1) (autoregressive process of order 1), CI (confidence interval), HS (historical '
        'simulation), VaR (value at risk), ES (expected shortfall), TE (tracking error), bp (basis points), i.i.d. (independent and identically distributed).',
   src='Course data in data/market (up to 18 September 2026): daily closes of Bitcoin, Ethereum, Solana, the S&P 500, gold (XAU/USD) and the IBIT ETF; daily closes and lows '
       'of USDT, USDC and DAI since 2021. The stablecoin supply and the TerraUSD series come from DefiLlama, downloaded with the chapter’s function on 4 November 2026 and cut '
       'at 18 September 2026; where the download differs from the values on the slides, the page shows both. The GARCH(1,1)-t and GJR estimates, the conditional '
       'volatilities, the Student-t fits and the GARCH-t forecasts of the backtest come from the code of the chapter (Quantlets/Ch_14). From the prices the browser recomputes '
       'the returns, the descriptive statistics, the Hill indices, the ACF and the Ljung–Box test, the variance ratios, the correlations, the drawdowns, the weekend and '
       'safe-haven tests, the peg deviations, VaR, ES and the Kupiec test. Other windows, thresholds, horizons, periods and levels α are extra runs of the same code; the GARCH '
       'persistence on ETF windows of 1, 1.5 and 2.5 years is an extra estimation with the chapter’s code, and the multi-day GARCH forecasts and the share of a shock left '
       'are extra computations from the estimates on the slide.'),
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
    n = (lambda s: s.replace('.', ',')) if ro else (lambda s: s)
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    T = lambda r, e: r if ro else e
    A5 = lambda i, d='btc', mark=lec: select(i, T('Seria', 'Series'), [(k, nm + (mark if k == d else '')) for k, nm in
                                                                       [('btc', 'Bitcoin'), ('eth', 'Ethereum'), ('sol', 'Solana'), ('sp500', 'S&P 500'), ('gold', T('aur', 'gold'))]], d)
    A3 = lambda i, d='btc', mark=sl: select(i, T('Activul', 'Asset'), [(k, nm + (mark if k == d else '')) for k, nm in [('btc', 'Bitcoin'), ('eth', 'Ethereum'), ('sp500', 'S&P 500')]], d)
    YRS = lambda a, b: [(str(y), str(y)) for y in range(a, b + 1)]
    return {
     'halving': (slider('hv-h', T('Orizontul după halving (zile)', 'Horizon after the halving (days)'), 30, 730, 5, 365) + play('hv'), CSS + charts('hv-chart')),
     'ann-we': (slider('an-A', T('Numărul de randamente pe an A', 'Number of returns per year A'), 200, 400, 1, 365) + play('an'),
                charts('an-chart') + we('an-we', f'{WE}: {T("anualizarea cu frecvența reală", "annualisation with the actual frequency")}',
                                        T(r'\text{volatilitatea anuală} = \sqrt{A}\,s, \qquad \text{media anuală} = A\,\bar r', r'\text{annual volatility} = \sqrt{A}\,s, \qquad \text{annual mean} = A\,\bar r'),
                                        A5('an-k', mark=sl))),
     'desc-tab': (select('ds-s', T('Începutul eșantionului', 'Start of the sample'), [('full', T('începutul fiecărei serii', 'the start of each series') + lec),
                                                                                     ('2020-04-11', T('11 aprilie 2020 (începutul seriei Solana)', '11 April 2020 (start of the Solana series)')),
                                                                                     ('2022-01-01', T('1 ianuarie 2022', '1 January 2022')), ('2024-01-01', T('1 ianuarie 2024', '1 January 2024'))], 'full') + play('ds'),
                  charts('ds-chart') + xh(T('Statisticile randamentelor logaritmice zilnice (tabelul de pe slide)', 'Statistics of the daily log returns (the table of the slide)')) + tw('ds-tab')),
     'rvol': (slider('rv-w', T('Fereastra pentru cripto (zile)', 'Window for crypto (days)'), 7, 180, 1, 30)
              + select('rv-a', T('Anualizarea pentru cripto', 'Annualisation for crypto'), [('365', '√365' + lec), ('252', T('√252 (greșit pentru cripto)', '√252 (wrong for crypto)'))], '365') + play('rv'),
              charts('rv-chart')),
     'weekday': (select('wd-y', T('Anul adăugat în grafic', 'Year added to the chart'), [('none', T('niciunul', 'none') + lec)] + YRS(2015, 2026), 'none')
                 + select('wd-m', T('Statistica', 'Statistic'), [('sd', T('abaterea standard', 'standard deviation') + lec), ('mean', T('media', 'mean'))], 'sd') + play('wd'),
                 charts('wd-chart')),
     'hill-tab': (select('hl-k', T('k ca pondere din n', 'k as a share of n'), [('0.025', n('2.5%') + lec), ('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%')], '0.025') + play('hl'),
                  charts('hl-chart') + xh(T('Indicii cozii estimați prin Hill (tabelul de pe slide)', 'Hill tail indices (the table of the slide)')) + tw('hl-tab')),
     'lb-we': (slider('lb-m', T('Numărul de laguri m', 'Number of lags m'), 1, 30, 1, 10) + play('lb'),
               charts('lb-chart') + xh(T('Statistica Ljung–Box Q(m) pentru rₜ și rₜ², cu p-value', 'The Ljung–Box Q(m) of rₜ and rₜ², with p-values')) + tw('lb-tab')
               + we('lb-st', f'{WE}: {T("statistica Ljung–Box", "the Ljung–Box statistic")}', r'Q(m) = n(n + 2)\sum_{k=1}^{m}\frac{\hat\rho(k)^2}{n - k} \sim \chi^2(m)',
                    A5('lb-k', mark=sl) + select('lb-x', T('Seria testată', 'Series tested'), [('r', T('randamentele rₜ', 'returns rₜ') + sl), ('r2', T('pătratele rₜ²', 'squares rₜ²') + sl)], 'r'))),
     'acf': (A5('ac-k') + slider('ac-L', T('Lagurile afișate', 'Lags shown'), 10, 60, 1, 30) + play('ac'), charts('ac-chart')),
     'garch-tab': (slider('gt-h', T('Zile după șoc h', 'Days after the shock h'), 0, 400, 1, 100) + play('gt'),
                   charts('gt-chart') + xh(T('Estimațiile GARCH(1,1)-t și GJR (tabelul de pe slide) și partea rămasă dintr-un șoc', 'GARCH(1,1)-t and GJR estimates (the table of the slide) and the share of a shock left')) + tw('gt-tab')
                   + we('gt-we', f'{WE}: {T("timpul de înjumătățire al unui șoc", "the half-life of a shock")}',
                        T(r'h_{1/2} = \frac{\ln 0{,}5}{\ln(\alpha + \beta)}, \qquad \text{partea rămasă după } h \text{ zile} = (\alpha + \beta)^h',
                          r'h_{1/2} = \frac{\ln 0.5}{\ln(\alpha + \beta)}, \qquad \text{share left after } h \text{ days} = (\alpha + \beta)^h'),
                        select('gt-k', T('Activul', 'Asset'), [('sp500', 'S&P 500' + sl), ('gold', T('aur', 'gold') + sl), ('btc', 'Bitcoin'), ('eth', 'Ethereum')], 'sp500'))),
     'garch': (slider('gv-h', T('Orizontul prognozei (zile)', 'Forecast horizon (days)'), 0, 365, 1, 0)
               + select('gv-k', T('Seriile', 'Series'), [('both', T('Bitcoin și S&P 500', 'Bitcoin and S&P 500') + lec), ('btc', 'Bitcoin'), ('sp500', 'S&P 500')], 'both') + play('gv'),
               charts('gv-chart')),
     'vr-tab': (select('vt-q', T('Orizontul q al lui Z*', 'Horizon q of Z*'), [('5', '5' + lec), ('2', '2'), ('10', '10'), ('20', '20')], '5') + play('vt'),
                charts('vt-chart') + xh(T('Întregul eșantion (primul tabel de pe slide)', 'Whole sample (the first table of the slide)')) + tw('vt-tab')
                + xh(T('Subperioade (al doilea tabel de pe slide)', 'Sub-periods (the second table of the slide)')) + tw('vt-tab2')),
     'rvr': (select('vr-q', T('Orizontul q', 'Horizon q'), [('5', '5' + lec), ('2', '2'), ('10', '10')], '5') + slider('vr-w', T('Fereastra (observații)', 'Window (observations)'), 250, 1000, 50, 500)
             + select('vr-z', T('Statistica', 'Statistic'), [('zs', T('robustă Z*', 'robust Z*') + lec), ('z', 'i.i.d. Z')], 'zs') + play('vr'), charts('vr-chart')),
     'rc-we': (slider('rc-w', T('Fereastra (zile comune)', 'Window (common days)'), 60, 500, 10, 250) + play('rc'),
               charts('rc-chart') + we('rc-st', f'{WE}: {T("corelația pe o fereastră mobilă", "the correlation on one rolling window")}',
                                       r'\hat\rho = \frac{\widehat{\mathrm{Cov}}(r^A, r^B)}{s_A\,s_B}, \qquad \mathrm{SE}(\hat\rho) \approx \frac{1 - \hat\rho^2}{\sqrt{w}}',
                                       select('rc-p', T('Perechea', 'Pair'), [('btc_sp500', 'Bitcoin – S&P 500' + sl), ('btc_gold', T('Bitcoin – aur', 'Bitcoin – gold') + sl), ('eth_sp500', 'Ethereum – S&P 500')], 'btc_sp500')
                                       + select('rc-e', T('Sfîrșitul ferestrei', 'End of the window'), [('max', T('ziua maximului', 'the day of the maximum') + sl), ('min', T('ziua minimului', 'the day of the minimum')),
                                                                                                        ('last', T('ultima zi, 18 septembrie 2026', 'the last day, 18 September 2026'))], 'max'))),
     'corr': (slider('co-w', T('Fereastra (zile comune)', 'Window (common days)'), 60, 500, 10, 250) + play('co'), charts('co-chart')),
     'corr-tab': (select('cy-y', T('Anul evidențiat', 'Year highlighted'), YRS(2014, 2026), '2022')
                  + select('cy-f', T('Randamentele', 'Returns'), [('d', T('zilnice', 'daily') + lec), ('w', T('săptămînale (vineri–vineri)', 'weekly (Friday to Friday)'))], 'd') + play('cy'),
                  charts('cy-chart') + xh(T('Corelația an cu an (tabelul de pe slide arată anii 2019–2025)', 'Correlation year by year (the slide shows 2019–2025)')) + tw('cy-tab')),
     'crisis-tab': (select('cr-e', T('Episodul', 'Episode'), [('COVID-19', 'COVID-19'), ('Terra/Luna', 'Terra/Luna'), ('FTX', 'FTX')], 'COVID-19')
                    + slider('cr-x', T('Zile adăugate înainte și după', 'Days added before and after'), 0, 30, 1, 0) + play('cr'),
                    charts('cr-chart') + xh(T('Randamente logaritmice cumulate (%), tabelul de pe slide', 'Cumulative log returns (%), the table of the slide')) + tw('cr-tab')),
     'haven-we': (slider('sh-q', T('Ponderea celor mai proaste zile ale S&P 500', 'Share of the worst S&P 500 days'), 1, 20, 0.5, 5) + play('sh'),
                  charts('sh-chart') + we('sh-we', f'{WE}: {T("testul de refugiu", "the safe-haven test")}',
                                          r't = \frac{\bar r_{\mathrm{bad}}}{s_{\mathrm{bad}}/\sqrt{n_{\mathrm{bad}}}} \sim t(n_{\mathrm{bad}} - 1)',
                                          select('sh-k', T('Activul', 'Asset'), [('btc', 'Bitcoin' + sl), ('gold', T('aur', 'gold') + sl)], 'btc'))),
     'mdd-we': (slider('md-d', T('Pierderea d', 'Loss d'), 1, 95, 1, 75) + play('md'),
                charts('md-chart') + we('md-we', f'{WE}: {T("drawdown-ul maxim și cîștigul necesar", "the maximum drawdown and the gain needed")}',
                                        r'D_t = \frac{P_t}{\max_{s \le t} P_s} - 1, \qquad (1 - d)(1 + g) = 1 \;\Rightarrow\; g = \frac{d}{1 - d}', A3('md-k'))),
     'dd': (slider('dd-y', T('Anul de început', 'Start year'), 2016, 2024, 1, 2016) + slider('dd-t', T('Pragul sub record', 'Threshold below the record'), 5, 75, 5, 10) + play('dd'),
            charts('dd-chart')),
     'dd-tab': (select('de-t', T('Adîncimea minimă a unui episod', 'Minimum depth of an episode'), [('50', '50%' + lec), ('30', '30%'), ('40', '40%'), ('60', '60%'), ('70', '70%')], '50') + play('de'),
                charts('de-chart') + xh(T('Episoadele Bitcoin din septembrie 2014 (tabelul de pe slide)', 'Bitcoin episodes since September 2014 (the table of the slide)')) + tw('de-tab')),
     'etf': (slider('ef-w', T('Fereastra volatilității (zile)', 'Volatility window (days)'), 30, 365, 5, 90)
             + select('ef-y', T('Anul evidențiat', 'Year highlighted'), [('none', T('niciunul', 'none') + lec)] + YRS(2015, 2026), 'none') + play('ef'), charts('ef-chart')),
     'etf-tab': (select('et-L', T('Lungimea fiecărei ferestre', 'Length of each window'), [('2', T('2 ani', '2 years') + lec), ('1', T('1 an', '1 year')), ('1.5', T('1,5 ani', '1.5 years')), ('2.5', T('2,5 ani', '2.5 years'))], '2') + play('et'),
                 charts('et-chart') + xh(T('Bitcoin înainte și după 11 ianuarie 2024 (tabelul de pe slide)', 'Bitcoin before and after 11 January 2024 (the table of the slide)')) + tw('et-tab')),
     'ibit-we': (select('ib-k', T('Orizontul randamentelor (zile de tranzacționare ale ETF-ului)', 'Return horizon (trading days of the ETF)'), [('1', '1' + lec), ('2', '2'), ('5', T('5 (o săptămînă)', '5 (a week)')), ('10', '10')], '1') + play('ib'),
                 charts('ib-chart') + we('ib-we', f'{WE}: tracking error', r'\mathrm{TE} = \sqrt{252/k}\;\mathrm{sd}\big(r^{\mathrm{IBIT}}_t - r^{\mathrm{BTC}}_t\big)',
                                         select('ib-s', T('Eșantionul', 'Sample'), [('all', T('toate cele 671 de zile', 'all 671 days') + sl), ('2024', '2024'), ('2025', '2025'), ('2026', '2026')], 'all'))),
     'supply': (select('su-y', T('Data marcată', 'Date marked'), [('end', T('18 septembrie 2026', '18 September 2026') + lec), ('2020-01-01', T('1 ianuarie 2020', '1 January 2020')),
                                                                 ('2022-04-02', T('2 aprilie 2022 (maximul din 2022)', '2 April 2022 (the 2022 maximum)')), ('2023-03-10', T('10 martie 2023', '10 March 2023')),
                                                                 ('2023-08-19', T('19 august 2023 (minimul)', '19 August 2023 (the trough)')), ('2023-12-31', T('31 decembrie 2023', '31 December 2023'))], 'end')
                + select('su-c', T('Seriile', 'Series'), [('all', T('toate', 'all') + lec), ('usdt', T('totalul și USDT', 'total and USDT')), ('usdc', T('totalul și USDC', 'total and USDC')), ('dai', T('totalul și DAI', 'total and DAI'))], 'all')
                + play('su'), charts('su-chart')),
     'bp-we': (slider('bp-d', T('Ziua din martie 2023', 'Day of March 2023'), 8, 20, 1, 11) + play('bp'),
               charts('bp-chart') + we('bp-st', f'{WE}: {T("abaterea de la paritate și pierderea", "the peg deviation and the loss")}',
                                       T(r'd_t = 10\,000\,(P_t - 1)\ \text{bp}, \qquad \text{pierderea} = W\,(1 - P_t)', r'd_t = 10\,000\,(P_t - 1)\ \text{bp}, \qquad \text{loss} = W\,(1 - P_t)'),
                                       select('bp-k', T('Moneda', 'Coin'), [('usdc', 'USDC' + sl), ('dai', 'DAI'), ('usdt', 'USDT')], 'usdc')
                                       + select('bp-p', T('Prețul', 'Price'), [('close', T('închiderea', 'close') + sl), ('low', T('minimul zilei', 'daily low') + sl)], 'close')
                                       + select('bp-W', T('Poziția W (unități)', 'Position W (coins)'), [('1000000', T('1 000 000', '1,000,000') + sl), ('100000', T('100 000', '100,000')), ('10000000', T('10 000 000', '10,000,000'))], '1000000'))),
     'peg': (select('pg-t', T('Pragul (bp)', 'Threshold (bp)'), [('10', '10' + lec), ('5', '5'), ('25', '25'), ('50', '50'), ('100', '100')], '10')
             + select('pg-p', T('Prețul', 'Price'), [('close', T('închiderea', 'close') + lec), ('low', T('minimul zilei', 'daily low'))], 'close') + play('pg'),
             charts('pg-chart') + xh(T('Statisticile parității din 2021 (cifrele de pe slide)', 'Peg statistics since 2021 (the numbers of the slide)')) + tw('pg-tab')),
     'svb': (slider('sv-d', T('Ultima zi afișată (martie 2023)', 'Last day shown (March 2023)'), 8, 20, 1, 20)
             + select('sv-c', T('Monedele', 'Coins'), [('all', T('toate trei', 'all three') + lec), ('usdc', 'USDC'), ('dai', 'DAI'), ('usdt', 'USDT')], 'all') + play('sv'), charts('sv-chart')),
     'ust': (slider('us-d', T('Ultima zi afișată', 'Last day shown'), 0, 75, 1, 75) + play('us'), charts('us-chart')),
     'usd-we': (slider('ud-W', T('Poziția W (mii USD)', 'Position W (thousand USD)'), 10, 1000, 10, 100) + play('ud'),
                charts('ud-chart') + we('ud-we', f'{WE}: {T("de la randamentul logaritmic la dolari", "from the log return to dollars")}',
                                        T(r'\text{pierderea} = W\,\big(1 - e^{-v/100}\big)', r'\text{loss} = W\,\big(1 - e^{-v/100}\big)'),
                                        A3('ud-k') + select('ud-m', T('Măsura', 'Measure'), [('hs_v', 'VaR 1% HS' + sl), ('n_v', 'VaR 1% Normal'), ('t_v', 'VaR 1% Student-t'), ('g_v', 'VaR 1% GARCH-t'),
                                                                                              ('hs_e', n('ES 2.5% HS')), ('n_e', n('ES 2.5% Normal')), ('t_e', n('ES 2.5% Student-t')), ('g_e', n('ES 2.5% GARCH-t')),
                                                                                              ('hs10', T('VaR 1% HS pe 10 zile', '10-day VaR 1% HS'))], 'hs_v'))),
     'var-tab': (select('vq-a', T('Nivelul VaR α', 'VaR level α'), [('0.01', '1%' + lec), ('0.025', n('2.5%')), ('0.05', '5%')], '0.01') + play('vq'),
                 charts('vq-chart') + xh(T('VaR și ES în procente (primul tabel de pe slide)', 'VaR and ES in percent (the first table of the slide)')) + tw('vq-tab')
                 + xh(T('În USD, pentru o poziție de 100 000 USD (al doilea tabel de pe slide)', 'In USD, for a position of 100,000 USD (the second table of the slide)')) + tw('vq-tab2')),
     'sqrt-we': (slider('sq-h', T('Orizontul h (zile)', 'Horizon h (days)'), 1, 30, 1, 10) + play('sq'),
                 charts('sq-chart') + we('sq-we', f'{WE}: {T("regula rădăcinii pătrate a timpului", "the square-root-of-time rule")}',
                                         r'\mathrm{VaR}(h\ \text{' + T('zile', 'days') + r'}) \overset{?}{=} \sqrt{h}\,\mathrm{VaR}(1\ \text{' + T('zi', 'day') + '})', A3('sq-k'))),
     'kup-we': (slider('kp-x', T('Numărul de excepții x', 'Number of exceptions x'), 0, 80, 1, 29) + play('kp'),
                charts('kp-chart') + we('kp-we', f'{WE}: {T("testul Kupiec", "the Kupiec test")}',
                                        r'LR_{uc} = -2\ln\frac{(1 - \alpha)^{n - x}\,\alpha^{x}}{(1 - \hat\pi)^{n - x}\,\hat\pi^{x}}, \qquad \hat\pi = \frac{x}{n}',
                                        select('kp-a', T('Nivelul α', 'Level α'), [('0.01', n('0.01') + sl), ('0.005', n('0.005')), ('0.02', n('0.02'))], '0.01'))),
     'btcvar': (slider('bt-w', T('Fereastra HS (zile)', 'HS window (days)'), 100, 1000, 5, 365)
                + select('bt-a', T('Nivelul VaR α', 'VaR level α'), [('0.01', '1%' + lec), ('0.025', n('2.5%')), ('0.05', '5%')], '0.01') + play('bt'), charts('bt-chart')),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang` (same frame numbers in RO and EN)."""
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch14.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch14_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch14_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
FED = '<a href="https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm" target="_blank" rel="noopener">'
FDIC = '<a href="https://www.fdic.gov/news/press-releases/2023/pr23016.html" target="_blank" rel="noopener">'
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s %]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const X = ' (rulare suplimentară a codului din curs)';
const XS = ' (alte valori decît pe slide)';
const XC = ' (calcul suplimentar din estimațiile de pe slide)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const abs = s => String(s).replace(/^[−-]/, '');
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const PAIR = { btc: 'Bitcoin', eth: 'Ethereum', sp500: 'S&P 500', gold: 'aurul' };
window.CH14_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`, an_: { gold: 'aurul', usdt: 'USDT', usdc: 'USDC', dai: 'DAI' }, notyet: 'încă nu',
 h: { ds: ['', 'din', 'n', 'media (%)', 'ab. std. (%)', 'vol. anuală (%)', 'asimetria', 'exces de boltire', 'min. (%)', 'data minimului', '|r| > 5% (% din zile)'],
      hl: ['', 'n', 'k', 'coada stîngă α̂', 'CI 95%', 'coada dreaptă α̂', 'CI 95%'],
      lb: m => ['', 'n', `Q(${m}) pentru rₜ`, 'p', `Q(${m}) pentru rₜ²`, 'p'],
      gt: h => ['', 'ω̂', 'α̂', 'β̂', 'α̂ + β̂', 'înjumătățire (zile)', 'ν̂', 'GJR γ̂ (t)', `rămas după ${h} ${de(h)}zile`],
      vt: q => ['', 'ρ̂(1)', 'Q(10) (p)', 'VR(2)', 'VR(5)', 'VR(10)', `Z*(${q}) (p)`], vt2: q => ['', 'perioada', 'n', 'ρ̂(1)', `VR(${q})`, `Z*(${q}) (p)`],
      cy: f => [`Randamente ${f === 'w' ? 'săptămînale' : 'zilnice'}`, ...Array.from({ length: 13 }, (_, i) => String(2014 + i)), 'toată perioada'],
      cyRows: ['Bitcoin cu S&P 500', 'Bitcoin cu aurul', 'Ethereum cu S&P 500'],
      cr: ['Episodul', 'perioada', 'Bitcoin', 'Ethereum', 'S&P 500', 'aur', 'cea mai proastă zi a Bitcoin'],
      de: ['Vîrful', 'USD', 'minimul', 'USD', 'adîncimea (%)', 'nou record', 'zile sub apă'],
      et: { head: ['Bitcoin', 'înainte', 'după'], rows: ['volatilitatea anualizată (%)', 'varianța weekend/zile lucrătoare', 'corelația cu S&P 500', 'GARCH(1,1)-t α̂ + β̂', 'zile'] },
      pg: t => ['', 'zile', 'mediana |d| (bp)', `|d| > ${t} bp (% din zile)`, '|d| > 50 bp (%)', '|d| > 100 bp (%)', 'cel mai mic d (bp, data)', 'cel mai mic minim (bp, data)', 'AR(1) φ̂', 'înjumătățire (zile)'],
      vq: a => [['', { v: `VaR ${a}% (%)`, span: 4 }, { v: 'ES 2,5% (%)', span: 4 }], ['', 'HS', 'Normal', 't', 'GARCH-t', 'HS', 'Normal', 't', 'GARCH-t']],
      vq2: a => ['Poziție de 100 000 USD', `VaR ${a}% HS`, 'ES 2,5% HS', `VaR ${a}% Normal`, `VaR ${a}% HS pe 10 zile`] },
 meth: { hs_v: 'VaR 1% HS', n_v: 'VaR 1% Normal', t_v: 'VaR 1% Student-t', g_v: 'VaR 1% GARCH-t', hs_e: 'ES 2,5% HS', n_e: 'ES 2,5% Normal', t_e: 'ES 2,5% Student-t', g_e: 'ES 2,5% GARCH-t', hs10: 'VaR 1% HS pe 10 zile' },
 hv: (h, rows, p0, pe, s24, s30, x) => `La ${h} ${de(h)}zile după halving-uri${x ? X : ''}: ` + rows.map(r => (r[2] === null ? `${r[5]}: încă neobservat` : `${r[5]}: de la ${r[1]} la ${r[2]} USD, ${r[3]}%`)).join('; ')
   + `. Oferta din regula protocolului: ${s24} milioane BTC la începutul lui 2024 și ${s30} milioane în 2030. Închiderea a fost ${p0} USD pe 17 septembrie 2014 și ${pe} USD pe 18 septembrie 2026.`,
 an1: (nm, n, m, s, A) => `${cap(nm)}: ${n} ${de(n)}randamente logaritmice zilnice, r̄ = ${m}% și s = ${s}% pe zi; frecvența reală este A = ${A} randamente pe an.`,
 an2: (A, sq, s, v, m, ma) => `Volatilitatea anuală: √A × s = ${sq} × ${s} = ${v}%; media anuală: A × r̄ = ${A} × ${m} = ${ma}%.`,
 an3: (v252, m252, uv, um, above, eq) => (eq ? `Aceasta este regula celor 252 de zile: corectă pentru acțiuni, greșită pentru o piață care tranzacționează 365 de zile pe an.`
   : `Cu A = 252: ${v252}% și ${m252}%, adică o volatilitate cu ${abs(uv)}% ${above ? 'mai mică' : 'mai mare'} și o medie cu ${abs(um)}% ${above ? 'mai mică' : 'mai mare'}: volatilitatea crește cu √A, media cu A.`),
 an4: (sr, mix) => `Raportul media/volatilitate (fără rata fără risc) este ${sr}; amestecul mediei × 252 cu volatilitatea × √365 dă ${mix}.`,
 an: (A, nm, v, m, xs, other) => `A = ${A}${xs ? XS : ''}: ${nm} are volatilitatea anuală ${v}% și media anuală ${m}%.`,
 ds: (st, r, vb, vs, s5, mn, mnd, nneg, nk, x) => `${st === 'full' ? 'Fiecare serie de la începutul ei pînă pe 18 septembrie 2026' : 'Toate seriile de la ' + window.CH14_TEXT.date(st)}${x ? X : ''}: Bitcoin este de ${r[0]} ori mai volatil decît S&P 500 (${vb}% față de ${vs}%), Ethereum de ${r[1]} ori, Solana de ${r[2]} ori. În ${s5}% din zile, Bitcoin se mișcă cu peste 5%; cea mai proastă zi: ${mn}% pe ${mnd}. ${nneg} din 5 serii au asimetrie negativă, ${nk} din 5 au exces de boltire pozitiv.`,
 rv: (w, ws, A, b, e, s, ratio, share, bl, sl, x) => `Ferestre de ${w} ${de(w)}randamente pentru cripto (× √${A}) și de ${ws} ${de(ws)}randamente pentru S&P 500 (× √252)${x ? X : ''}. Medianele: Bitcoin ${b[0]}%, Ethereum ${e[0]}%, S&P 500 ${s[0]}%; raportul median Bitcoin/S&P 500 este ${ratio}. Maximele: ${b[1]}% (${b[2]}), ${e[1]}% (${e[2]}), ${s[1]}% (${s[2]}). Volatilitatea Bitcoin a fost sub cea a S&P 500 în ${share}% din zilele comune.`,
 wd: (rows, m) => 'Coeficientul variabilei weekend b (p-value) și varianța weekend/zile lucrătoare (Brown–Forsythe p): ' + rows.map(r => `${r[0]}: ${r[1]} (${r[2]}), ${r[3]} (p ${r[4]})`).join('; ') + '. Niciun efect de weekend în medie; în weekend varianța este mai mică în toate trei perioadele.',
 wdY: (y, n, b, p, r, pb) => `${y}${X}: ${n} ${de(n)}randamente, b = ${b} (p ${p}), varianța weekend/zile lucrătoare ${r} (Brown–Forsythe p ${pb}).`,
 hl: (sh, k, al, ar, n4, nlr, x) => `k = ${sh}% din n (la Bitcoin k = ${k})${x ? X : ''}: Bitcoin are α̂ = ${al} în coada stîngă și ${ar} în coada dreaptă; ${n4} din 5 indici ai cozii stîngi sînt sub 4; coada stîngă este mai groasă la ${nlr} din 5 active.`,
 lb1: (nm, xs, n, rho, m) => `${cap(nm)}, ${xs === 'r' ? 'randamentele rₜ' : 'pătratele rₜ²'}: n = ${n}; ${rho.map((v, i) => `ρ̂(${i + 1}) = ${v}`).join(', ')}${m > 3 ? ', …' : '.'}`,
 lb2: (m, s6, n, q) => `Σ ρ̂(k)²/(n − k) pentru k = 1, …, ${m} este ${s6} × 10⁻⁶; Q(${m}) = n(n + 2) × această sumă = ${q}.`,
 lb3: (m, c, p, rej) => `Fără autocorelație, Q(${m}) urmează χ²(${m}); valoarea critică la 5% este ${c}, p-value ${p}: ${rej ? 'autocorelațiile sînt semnificative împreună la 5%' : 'autocorelațiile nu sînt semnificative împreună la 5%'}.`,
 lb: (m, nm, q1, p1, q2, p2, c, x) => `${cap(nm)}, m = ${m}${x ? X : ''}: Q(${m}) = ${q1} pentru rₜ (p ${p1}) și ${q2} pentru rₜ² (p ${p2}); valoarea critică la 5% este ${c}.`,
 ac: (nm, n, band, r1, o1, L, r2, o2, x) => `${cap(nm)}, n = ${n}${x ? X : ''}: banda ±1,96/√n = ±${band}. Randamentele: ρ̂(1) = ${r1}, ${o1} din ${L} ${de(L)}laguri în afara benzii; pătratele: ρ̂(1) = ${r2}, ${o2} din ${L} în afara benzii.`,
 ac2: (q1, p1, q2, p2, ratio, btc) => `Statistica Ljung–Box Q(10) este ${q1} pentru rₜ (p ${p1}) și ${q2} pentru rₜ² (p ${p2}), de ${ratio} ori mai mare${btc ? ': direcția este greu de prognozat, mărimea mișcării nu' : ''}.`,
 gt1: (nm, a, b, p) => `${cap(nm)}: α̂ = ${a}, β̂ = ${b}, persistența α̂ + β̂ = ${p}.`,
 gt2: (l, hl) => `Timpul de înjumătățire: ln 0,5/ln(α̂ + β̂) = −0,6931/(${l}) = ${hl} zile.`,
 gt2b: (l, my) => `ln(α̂ + β̂) = ${l} × 10⁻¹²: formula ar da circa ${my} de milioane de ani, de aceea capitolul nu raportează un timp de înjumătățire finit (α̂ + β̂ = 1 la precizia estimației, un GARCH integrat).`,
 gt3: (h, s) => `După h = ${h} ${de(h)}zile, (α̂ + β̂)ʰ = ${s}% din șocul varianței este încă prezent.`,
 gt4: (A, om, one, u, nm) => `Volatilitatea de termen lung √(A ω̂/(1 − α̂ − β̂)) = √(${A} × ${om}/${one}) = ${u}% pe an pentru ${nm}; cînd α̂ + β̂ este aproape de 1, o eroare mică a persistenței schimbă mult această valoare.`,
 gt4b: () => 'Cu α̂ + β̂ = 1 varianța de termen lung ω/(1 − α − β) nu există: un șoc al volatilității nu este uitat.',
 gt: (h, rows, x) => `După ${h} ${de(h)}zile rămîne din șocul varianței${XC}: ` + rows.map(r => `${r[0]} ${r[1]}%`).join(', ') + '.',
 gv: (pb, db, ps, ds, de_, lb, ls) => `Ambele ating maximul în martie 2020: Bitcoin ${pb}% (${db}), S&P 500 ${ps}% (${ds}); pe ${de_} modelul dă ${lb}% pentru Bitcoin și ${ls}% pentru S&P 500.`,
 gvF: (h, b1, bh, s1, sh, u, sv) => `Prognoza de la 18 septembrie 2026${XC}: Bitcoin ${b1}% pentru ziua următoare și ${bh}% peste ${h} ${de(h)}zile: cu α̂ + β̂ = 1 varianța prognozată crește în fiecare zi cu ω̂ și nu revine la un nivel de termen lung. S&P 500: de la ${s1}% la ${sh}%, spre nivelul de termen lung de ${u}%, care este peste volatilitatea din eșantion (${sv}%) pentru că α̂ + β̂ = 0,993 este aproape de 1.`,
 vt: (q, rej, rs, ns, x) => `Z*(${q})${x ? X : ''}: ${rej.length ? 'respinge mersul aleator la 5% pentru ' + rej.join(', ') : 'nicio serie nu respinge mersul aleator la 5%'}; ${rs} din ${ns} subperioade resping. Testul Ljung–Box respinge slab la cripto (p 0,028 și 0,012) pentru că ignoră volatility clustering.`,
 vr: (q, w, n, rb, re, rs, ib, is_, mn, mx, zk, x) => `q = ${q}, ferestre de ${w} ${de(w)}observații, cîte una la fiecare 21${x ? X : ''}: ${n} ${de(n)}ferestre pentru Bitcoin. Statistica robustă Z* respinge în ${rb}% din ferestrele Bitcoin, ${re}% la Ethereum și ${rs}% la S&P 500; cu statistica i.i.d. Z proporțiile ar fi ${ib}% (Bitcoin) și ${is_}% (S&P 500). VR(${q}) al Bitcoin între ${mn} și ${mx}.`,
 rc1: (a, b, w, d0, d1) => `${cap(a)} și ${b}: cele ${w} ${de(w)}zile comune de la ${d0} la ${d1} (întîi se unesc prețurile, apoi se calculează randamentele).`,
 rc2: (sa, sb, cv, r) => `Abaterile standard ${sa}% și ${sb}%, covarianța ${cv}; ρ̂ = ${cv}/(${sa} × ${sb}) = ${r}.`,
 rc3: (w, r, se, lo, hi, s0) => `Eroarea standard ≈ (1 − ${r}²)/√${w} = ${se}; o bandă aproximativă de 95%: de la ${lo} la ${hi}. Lîngă ρ = 0 eroarea standard este 1/√${w} = ${s0}: variațiile corelației mobile mai mici decît circa două erori standard sînt zgomot.`,
 rc4: (d, wk, nd, nw) => `Toată perioada: ${d} cu ${nd} ${de(nd)}randamente zilnice și ${wk} cu ${nw} ${de(nw)}randamente săptămînale (vineri–vineri): închiderile asincrone nu explică această corelație.`,
 rc: (w, a, b, e, r, d, se, x) => `${cap(a)} și ${b}, fereastră de ${w} ${de(w)}zile comune${x ? X : ''}: ${e === 'max' ? 'cea mai mare' : e === 'min' ? 'cea mai mică' : 'ultima'} corelație mobilă este ${r}, pe ${d} (eroarea standard ${se}).`,
 co: (w, f0, f1, f2, g25, g26, x) => `Fereastră de ${w} ${de(w)}zile comune${x ? X : ''}. Bitcoin și S&P 500: ${f0[0]} înainte de 2020, ${f0[1]} după; maximul corelației mobile ${f0[2]} pe ${f0[3]}; ultima valoare ${f0[4]}. Bitcoin și aurul: ${f1[0]} înainte de 2020, ${f1[1]} după; ${g25} în 2020–2025 și ${g26} în 2026; maximul corelației mobile ${f1[2]} pe ${f1[3]}. Ethereum și S&P 500: ${f2[0]} și ${f2[1]}; maximul ${f2[2]} pe ${f2[3]}.`,
 cy: (y, v, f, d, w, x) => `${y}${x ? X : ''}: Bitcoin cu S&P 500 ${v[0]}, cu aurul ${v[1]}; Ethereum cu S&P 500 ${v[2]} (randamente ${f === 'w' ? 'săptămînale' : 'zilnice'}). Toată perioada, Bitcoin și S&P 500: zilnic ${d}, săptămînal ${w}.`,
 cr: (ep, a, b, bt, et, sp, g, w, wd, x) => `${ep}, ${a} – ${b}${x ? X : ''}: Bitcoin ${bt}%, Ethereum ${et}%, S&P 500 ${sp}%, aurul ${g}% (randamente logaritmice cumulate); cea mai proastă zi a Bitcoin: ${w}% pe ${wd}.`,
 sh1: (n, nb, q, thr, msp) => `${n} ${de(n)}zile comune; cele mai proaste ${q}% sînt cele ${nb} ${de(nb)}zile în care S&P 500 a avut un randament de cel mult ${thr}% (în medie ${msp}%).`,
 sh2: (nm, m, s, nb, t, p) => `${cap(nm)} în acele zile: media ${m}%, abaterea standard ${s}%; t = media/(s/√${nb}) = ${t}, p bilateral ${p} (Student-t cu ${nb} − 1 grade de libertate).`,
 sh3: (neg, cb, ma) => `Negativ în ${neg}% din acele zile; corelația cu S&P 500 în acele zile ${cb}; media în toate zilele comune ${ma}%.`,
 sh: (q, nb, bm, bt, bp, gm, gt, gp, x) => `Cele mai proaste ${q}% din zilele S&P 500 (${nb})${x ? X : ''}: Bitcoin ${bm}% (t = ${bt}, p ${bp}), aurul ${gm}% (t = ${gt}, p ${gp}).`,
 md1: (nm, pd, pp, td, tp, d, pts) => `${cap(nm)}: recordul ${pp} ${pts ? 'puncte' : 'USD'} pe ${pd}, cea mai mică închidere ulterioară ${tp} pe ${td}: drawdown-ul maxim D = ${d}%.`,
 md2: (d, one, g) => `Pierderea d = ${d}: prețul este ${one} din record, deci recuperarea cere un cîștig d/(1 − d) = ${g}%.`,
 md3: (rd, days) => `Nou record pe ${rd}, la ${days} ${de(days)}zile după vîrf (timpul sub apă).`,
 md3b: () => 'Niciun nou record pînă pe 18 septembrie 2026.',
 md: (d, g, x) => `O pierdere de ${d}% cere un cîștig de ${g}% pentru recuperare${x ? XS : ''}.`,
 dd: (y, rows, t, last, de_, x) => `Închideri zilnice din ${y}${x ? X : ''}: drawdown-urile maxime ` + rows.map(r => `${r[0]} ${r[1]}% (${r[2]})`).join(', ') + `; zile la peste ${t}% sub record: ` + rows.map(r => `${r[0]} ${r[3]}%`).join(', ') + `. Pe ${de_} Bitcoin era cu ${last}% sub record.`,
 de: (th, n, nr, lg, yr, x) => `Episoade mai adînci de ${th}%${x ? X : ''}: ${n}, dintre care ${nr} recuperate; cel mai lung timp sub apă: ${lg} ${de(lg)}zile (vîrful din ${yr}).`,
 ef: (w, v0, v1, vl, de_, mn, mx, x) => `Volatilitatea pe ${w} ${de(w)}zile${x ? X : ''}: ${v1}% pe 10 ianuarie 2023, ${v0}% pe 10 ianuarie 2024, ziua dinaintea ETF-urilor, și ${vl}% pe ${de_}. Ponderea weekendului în suma pătratelor randamentelor a fost între ${mn}% și ${mx}% în fiecare an din 2020, sub 2/7 = 28,6%.`,
 efY: (y, mv, sh) => `${y}: ${mv === null ? 'volatilitatea mobilă este afișată din 2020' : 'volatilitatea medie ' + mv + '%'}; ponderea weekendului ${sh}%.`,
 et: (L, b0, b1, a0, a1, vb, va, bf, p, sig, x) => `Ferestrele ${b0} – ${b1} și ${a0} – ${a1}${x ? X : ''}: volatilitatea ${vb}% înainte și ${va}% după; statistica Brown–Forsythe ${bf}, p = ${p}: ${sig ? 'schimbarea volatilității este semnificativă la 5%' : 'schimbarea volatilității nu este semnificativă la 5%'}.`,
 ib1: (k, s, nd, n) => `IBIT și Bitcoin, prețuri unite în cele ${nd} ${de(nd)}zile de tranzacționare ale ETF-ului${s === 'all' ? '' : ' din ' + s}; ${+k === 1 ? 'randamente zilnice' : n + ' ' + de(n) + 'randamente pe ' + k + ' zile, fără suprapunere'}.`,
 ib2: (c, b) => `Corelația ${c}; panta lui IBIT față de Bitcoin ${b}.`,
 ib3: (sd, k, ann, te) => `Abaterea standard a diferenței randamentelor ${sd}%; × √(252/${k}) = ${ann}: tracking error ${te}% pe an.`,
 ib: (k, c, b, te, t1, t5, c5, x) => `Randamente pe ${k} ${+k === 1 ? 'zi' : 'zile'}${x ? X : ''}: corelația ${c}, panta ${b}, tracking error ${te}%. Zilnic: ${t1}%; săptămînal (5 zile): ${t5}%, cu corelația ${c5}: diferența se compensează într-o săptămînă.`,
 su: (d, t, u, c, dai, us, cs, dl) => `${d} (DefiLlama, descărcarea din ${dl}): toate stablecoins în dolari, ${t} miliarde USD; USDT ${u}, USDC ${c} și DAI ${dai} miliarde; USDT deține ${us}%, USDC ${cs}%.`,
 su2: (ts, tn, gs, gn, us, un, cs, cn, mx, mxd, tr, trd, c0, c1, dl) => `Slide-urile față de descărcarea din ${dl}: totalul pe 18 septembrie 2026, ${ts} față de ${tn} ${de(tn)}miliarde USD; factorul de creștere față de începutul lui 2020, ${gs} față de ${gn}; ponderea USDT, ${us}% față de ${un}%, și a USDC, ${cs}% față de ${cn}%; DefiLlama își revizuiește istoricul. Neschimbate: maximul din 2022, ${mx} ${de(mx)}miliarde (${mxd}), minimul, ${tr} ${de(tr)}miliarde (${trd}), și scăderea USDC de la ${c0} la ${c1} ${de(c1)}miliarde în 2023.`,
 bp1: (nm, d, f, P) => `${nm}, ${f === 'close' ? 'închiderea' : 'minimul zilei'} din ${d}: P = ${P} USD.`,
 bp2: (P, dv, pc) => `d = 10 000 × (${P} − 1) = ${dv} bp, adică ${pc}%.`,
 bp3: (W, P, loss, below) => (below ? `Un deținător a ${W} de monede care a vîndut la acest preț a pierdut ${W} × (1 − ${P}) = ${loss} USD.` : `Un deținător care a vîndut ${W} de monede la acest preț a primit cu ${loss} USD mai mult decît la paritate.`),
 bp4: (c, l, ll) => `Închiderea ${c} și minimul ${l}: la minim, diferența față de paritate ar fi fost ${ll} USD.`,
 bp: (d, rows, x) => `${d}${x ? XS : ''}: ` + rows.map(r => `${r[0]} ${r[1]} bp la închidere, ${r[2]} bp la minim`).join('; ') + '.',
 pg: (f, t, rows, x) => `${f === 'close' ? 'Închideri' : 'Minime'} zilnice din 2021${x ? X : ''}: abaterea absolută mediană ` + rows.map(r => `${r[0]} ${r[1]} bp`).join(', ') + `; zile cu |d| > ${t} bp: ` + rows.map(r => `${r[0]} ${r[2]}%`).join(', ') + '; timpi de înjumătățire ' + rows.map(r => `${r[0]} ${r[3]}`).join(', ') + ' zile.',
 sv: (d, rows) => `${d}: ` + rows.map(r => `${r[0]} închidere ${r[1]} (${r[2]} bp), minim ${r[3]} (${r[4]} bp)`).join('; ') + '.',
 sv2: day => (day < 10 ? 'Înainte de 10 martie toate trei se tranzacționează la cîteva puncte de bază de 1 USD.'
   : day <= 12 ? `10 martie: SVB este închisă (${'""" + FDIC + r"""'}FDIC, 2023</a>); urmează weekendul, cu băncile închise și piețele cripto deschise.`
   : `În seara de 12 martie, Trezoreria SUA, Rezerva Federală și FDIC garantează toate depozitele SVB (${'""" + FED + r"""'}U.S. Treasury, Federal Reserve și FDIC, 2023</a>); din 13 martie USDC și DAI revin aproape de paritate.`),
 us: (d, pd, pl, sd, sl, pk, pkd, drop) => `Pînă pe ${d}: ultimul preț ${pl} USD (${pd}); oferta ${sl} miliarde UST (${sd}); maximul ofertei ${pk} miliarde pe ${pkd}${drop === null ? '' : `; scăderea față de maxim ${drop}%`}.`,
 ud1: (nm, me, v) => `${cap(nm)}, ${me}: un randament logaritmic de −${v}%.`,
 ud2: (v, e) => `e^(−${v}/100) = ${e}: poziția păstrează ${e} din valoare.`,
 ud3: (W, one, loss) => `Pierderea = ${W} × ${one} = ${loss} USD.`,
 ud4: (lin, diff) => `Citit ca randament simplu (W × v/100), același număr ar da ${lin} USD, cu ${diff} USD prea mult.`,
 ud: (W, rows, x) => `Poziție de ${W} USD${x ? XS : ''}: VaR 1% HS ` + rows.map(r => `${r[0]} ${r[1]} USD`).join(', ') + '; ES 2,5% HS ' + rows.map(r => `${r[0]} ${r[2]} USD`).join(', ') + '.',
 vq: (a, bh, sh, ratio, under, gv, nu, gs, x) => `VaR ${a}%${x ? X : ''}: Bitcoin ${bh}% față de ${sh}% pentru S&P 500 (HS), de ${ratio} ori mai mult; VaR-ul Normal al Bitcoin este ${num(under) >= 0 ? 'cu ' + under + '% sub' : 'cu ' + abs(under) + '% peste'} cel istoric; GARCH-t dă ${gv}% pentru ziua următoare (σ̂ = ${gs}%); Student-t ajustat are ν̂ = ${nu}.`,
 sq1: (nm, v) => `${cap(nm)}: VaR 1% HS pe o zi = ${v}%.`,
 sq2: (h, s, v, r) => `Regula rădăcinii pătrate: √${h} × ${v} = ${s} × ${v} = ${r}%.`,
 sq3: (h, v, u) => `VaR 1% istoric al randamentelor pe ${h} ${+h === 1 ? 'zi' : 'zile'}, cu suprapunere: ${v}% (${u} USD pentru 100 000 USD).`,
 sq4: (p, over, h) => (+h === 1 ? 'Pentru h = 1 cele două coincid.' : over ? `Regula supraestimează cu ${p}%: cu un tail index peste 2, o cuantilă extremă a unei sume crește mai încet decît √h.` : `Aici regula subestimează cu ${p}%.`),
 sq: (h, nm, hs, r, x) => `${cap(nm)}, h = ${h}${x ? X : ''}: ${hs}% istoric față de ${r}% din regula rădăcinii pătrate.`,
 kp1: (x, n, r, e) => `x = ${x} ${de(x)}excepții în n = ${n} ${de(n)}zile: π̂ = ${r}%; numărul așteptat αn = ${e}.`,
 kp2: (l0, l1) => `Log-verosimilitatea cu rata α: ${l0}; cu rata π̂: ${l1}.`,
 kp3: (lr, p, rej) => `LR_uc = −2 × (diferența) = ${lr}, p ${p}: ${rej ? 'respins la 5%' : 'nerespins la 5%'}.`,
 kp4: (lo, hi) => `Pentru aceste n și α, testul nu respinge pentru x între ${lo} și ${hi}.`,
 kp: (x, lr, p, lo, hi, xs) => `x = ${x}${xs ? XS : ''}: LR_uc = ${lr}, p ${p}; nerespins pentru ${lo} ≤ x ≤ ${hi}.`,
 bt: (w, a, n, H, G, g18, h18, x) => `${n} ${de(n)}prognoze pe o zi din 1 ianuarie 2018, VaR ${a}%, HS pe ${w} ${de(w)}zile${x ? X : ''}: HS ${H[0]} ${de(H[0])}excepții (${H[1]}%), GARCH-t ${G[0]} (${G[1]}%), față de ${H[2]} așteptate; Kupiec LR_uc = ${H[3]} (p = ${H[4]}) și ${G[3]} (p = ${G[4]}); VaR mediu ${H[5]}% (HS) și ${G[5]}% (GARCH-t); excepții în 2018: ${h18} (HS) și ${g18} (GARCH-t).`
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/[\s ,%]/g, '').replace('−', '-');
const X = ' (extra run of the chapter’s code)';
const XS = ' (values other than on the slide)';
const XC = ' (extra computation from the slide’s estimates)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const abs = s => String(s).replace(/^[−-]/, '');
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
window.CH14_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`, an_: { gold: 'gold', usdt: 'USDT', usdc: 'USDC', dai: 'DAI' }, notyet: 'not yet',
 h: { ds: ['', 'since', 'n', 'mean (%)', 'sd (%)', 'ann. vol. (%)', 'skew.', 'exc. kurt.', 'min (%)', 'date of the min', '|r| > 5% (% of days)'],
      hl: ['', 'n', 'k', 'left tail α̂', '95% CI', 'right tail α̂', '95% CI'],
      lb: m => ['', 'n', `Q(${m}) of rₜ`, 'p', `Q(${m}) of rₜ²`, 'p'],
      gt: h => ['', 'ω̂', 'α̂', 'β̂', 'α̂ + β̂', 'half-life (days)', 'ν̂', 'GJR γ̂ (t)', `left after ${h} days`],
      vt: q => ['', 'ρ̂(1)', 'Q(10) (p)', 'VR(2)', 'VR(5)', 'VR(10)', `Z*(${q}) (p)`], vt2: q => ['', 'period', 'n', 'ρ̂(1)', `VR(${q})`, `Z*(${q}) (p)`],
      cy: f => [`${f === 'w' ? 'Weekly' : 'Daily'} returns`, ...Array.from({ length: 13 }, (_, i) => String(2014 + i)), 'whole period'],
      cyRows: ['Bitcoin with S&P 500', 'Bitcoin with gold', 'Ethereum with S&P 500'],
      cr: ['Episode', 'period', 'Bitcoin', 'Ethereum', 'S&P 500', 'gold', 'worst Bitcoin day'],
      de: ['Peak', 'USD', 'trough', 'USD', 'depth (%)', 'new record', 'days under water'],
      et: { head: ['Bitcoin', 'before', 'after'], rows: ['annualised volatility (%)', 'weekend/weekday variance', 'correlation with the S&P 500', 'GARCH(1,1)-t α̂ + β̂', 'days'] },
      pg: t => ['', 'days', 'median |d| (bp)', `|d| > ${t} bp (% of days)`, '|d| > 50 bp (%)', '|d| > 100 bp (%)', 'lowest d (bp, date)', 'lowest low (bp, date)', 'AR(1) φ̂', 'half-life (days)'],
      vq: a => [['', { v: `VaR ${a}% (%)`, span: 4 }, { v: 'ES 2.5% (%)', span: 4 }], ['', 'HS', 'Normal', 't', 'GARCH-t', 'HS', 'Normal', 't', 'GARCH-t']],
      vq2: a => ['Position of 100,000 USD', `VaR ${a}% HS`, 'ES 2.5% HS', `VaR ${a}% Normal`, `10-day VaR ${a}% HS`] },
 meth: { hs_v: 'VaR 1% HS', n_v: 'VaR 1% Normal', t_v: 'VaR 1% Student-t', g_v: 'VaR 1% GARCH-t', hs_e: 'ES 2.5% HS', n_e: 'ES 2.5% Normal', t_e: 'ES 2.5% Student-t', g_e: 'ES 2.5% GARCH-t', hs10: '10-day VaR 1% HS' },
 hv: (h, rows, p0, pe, s24, s30, x) => `${h} days after the halvings${x ? X : ''}: ` + rows.map(r => (r[2] === null ? `${r[5]}: not yet observed` : `${r[5]}: from ${r[1]} to ${r[2]} USD, ${r[3]}%`)).join('; ')
   + `. Supply from the protocol rule: ${s24} million BTC at the start of 2024 and ${s30} million in 2030. The close was ${p0} USD on 17 September 2014 and ${pe} USD on 18 September 2026.`,
 an1: (nm, n, m, s, A) => `${cap(nm)}: ${n} daily log returns, r̄ = ${m}% and s = ${s}% per day; the actual frequency is A = ${A} returns a year.`,
 an2: (A, sq, s, v, m, ma) => `Annual volatility: √A × s = ${sq} × ${s} = ${v}%; annual mean: A × r̄ = ${A} × ${m} = ${ma}%.`,
 an3: (v252, m252, uv, um, above, eq) => (eq ? 'This is the 252 rule: right for equities, wrong for a market that trades 365 days a year.'
   : `With A = 252: ${v252}% and ${m252}%, a volatility ${abs(uv)}% ${above ? 'lower' : 'higher'} and a mean ${abs(um)}% ${above ? 'lower' : 'higher'}: the volatility scales with √A, the mean with A.`),
 an4: (sr, mix) => `Mean/volatility ratio (without the risk-free rate): ${sr}; mixing the mean × 252 with the volatility × √365 gives ${mix}.`,
 an: (A, nm, v, m, xs, other) => `A = ${A}${xs ? XS : ''}: ${nm} has an annual volatility of ${v}% and an annual mean of ${m}%.`,
 ds: (st, r, vb, vs, s5, mn, mnd, nneg, nk, x) => `${st === 'full' ? 'Each series from its start until 18 September 2026' : 'All series from ' + window.CH14_TEXT.date(st)}${x ? X : ''}: Bitcoin is ${r[0]} times as volatile as the S&P 500 (${vb}% against ${vs}%), Ethereum ${r[1]} times, Solana ${r[2]} times. On ${s5}% of the days Bitcoin moves by more than 5%; its worst day: ${mn}% on ${mnd}. ${nneg} of 5 series have negative skewness, ${nk} of 5 positive excess kurtosis.`,
 rv: (w, ws, A, b, e, s, ratio, share, bl, sl, x) => `Windows of ${w} returns for crypto (× √${A}) and ${ws} returns for the S&P 500 (× √252)${x ? X : ''}. Medians: Bitcoin ${b[0]}%, Ethereum ${e[0]}%, S&P 500 ${s[0]}%; the median ratio Bitcoin/S&P 500 is ${ratio}. Maxima: ${b[1]}% (${b[2]}), ${e[1]}% (${e[2]}), ${s[1]}% (${s[2]}). Bitcoin's volatility was below that of the S&P 500 on ${share}% of the common days.`,
 wd: (rows, m) => 'Weekend dummy b (p-value) and weekend/weekday variance (Brown–Forsythe p): ' + rows.map(r => `${r[0]}: ${r[1]} (${r[2]}), ${r[3]} (p ${r[4]})`).join('; ') + '. No weekend effect in the mean; the weekend variance is lower in all three periods.',
 wdY: (y, n, b, p, r, pb) => `${y}${X}: ${n} returns, b = ${b} (p ${p}), weekend/weekday variance ${r} (Brown–Forsythe p ${pb}).`,
 hl: (sh, k, al, ar, n4, nlr, x) => `k = ${sh}% of n (Bitcoin: k = ${k})${x ? X : ''}: Bitcoin has α̂ = ${al} in the left tail and ${ar} in the right tail; ${n4} of 5 left-tail indices are below 4; the left tail is the heavier one for ${nlr} of 5 assets.`,
 lb1: (nm, xs, n, rho, m) => `${cap(nm)}, ${xs === 'r' ? 'returns rₜ' : 'squared returns rₜ²'}: n = ${n}; ${rho.map((v, i) => `ρ̂(${i + 1}) = ${v}`).join(', ')}${m > 3 ? ', …' : '.'}`,
 lb2: (m, s6, n, q) => `Σ ρ̂(k)²/(n − k) over k = 1, …, ${m} is ${s6} × 10⁻⁶; Q(${m}) = n(n + 2) × this sum = ${q}.`,
 lb3: (m, c, p, rej) => `Without autocorrelation Q(${m}) follows χ²(${m}); the 5% critical value is ${c}, the p-value ${p}: ${rej ? 'the autocorrelations are jointly significant at 5%' : 'the autocorrelations are not jointly significant at 5%'}.`,
 lb: (m, nm, q1, p1, q2, p2, c, x) => `${cap(nm)}, m = ${m}${x ? X : ''}: Q(${m}) = ${q1} for rₜ (p ${p1}) and ${q2} for rₜ² (p ${p2}); the 5% critical value is ${c}.`,
 ac: (nm, n, band, r1, o1, L, r2, o2, x) => `${cap(nm)}, n = ${n}${x ? X : ''}: band ±1.96/√n = ±${band}. Returns: ρ̂(1) = ${r1}, ${o1} of ${L} lags outside the band; squared returns: ρ̂(1) = ${r2}, ${o2} of ${L} outside.`,
 ac2: (q1, p1, q2, p2, ratio, btc) => `Ljung–Box Q(10) is ${q1} for rₜ (p ${p1}) and ${q2} for rₜ² (p ${p2}), ${ratio} times as large${btc ? ': the direction is hard to predict, the size of the move is not' : ''}.`,
 gt1: (nm, a, b, p) => `${cap(nm)}: α̂ = ${a}, β̂ = ${b}, persistence α̂ + β̂ = ${p}.`,
 gt2: (l, hl) => `Half-life: ln 0.5/ln(α̂ + β̂) = −0.6931/(${l}) = ${hl} days.`,
 gt2b: (l, my) => `ln(α̂ + β̂) = ${l} × 10⁻¹²: the formula would give about ${my} million years, so the chapter reports no finite half-life (α̂ + β̂ = 1 to the precision of the estimate, an integrated GARCH).`,
 gt3: (h, s) => `After h = ${h} days, (α̂ + β̂)ʰ = ${s}% of the variance shock is still there.`,
 gt4: (A, om, one, u, nm) => `Long-run volatility √(A ω̂/(1 − α̂ − β̂)) = √(${A} × ${om}/${one}) = ${u}% a year for ${nm}; with α̂ + β̂ close to 1, a small error in the persistence moves this number a lot.`,
 gt4b: () => 'With α̂ + β̂ = 1 the long-run variance ω/(1 − α − β) does not exist: a shock to volatility is not forgotten.',
 gt: (h, rows, x) => `Share of a variance shock left after ${h} days${XC}: ` + rows.map(r => `${r[0]} ${r[1]}%`).join(', ') + '.',
 gv: (pb, db, ps, ds, de, lb, ls) => `Both peak in March 2020: Bitcoin ${pb}% (${db}), S&P 500 ${ps}% (${ds}); on ${de} the model gives ${lb}% for Bitcoin and ${ls}% for the S&P 500.`,
 gvF: (h, b1, bh, s1, sh, u, sv) => `Forecast from 18 September 2026${XC}: Bitcoin ${b1}% for the next day and ${bh}% in ${h} days: with α̂ + β̂ = 1 the forecast variance grows by ω̂ every day and never returns to a long-run level. S&P 500: from ${s1}% to ${sh}%, towards its long-run level of ${u}%, which lies above the sample volatility (${sv}%) because α̂ + β̂ = 0.993 is close to 1.`,
 vt: (q, rej, rs, ns, x) => `Z*(${q})${x ? X : ''}: ${rej.length ? 'rejects the random walk at 5% for ' + rej.join(', ') : 'no series rejects the random walk at 5%'}; ${rs} of ${ns} sub-periods reject. The Ljung–Box test rejects weakly for crypto (p 0.028 and 0.012) because it ignores volatility clustering.`,
 vr: (q, w, n, rb, re, rs, ib, is_, mn, mx, zk, x) => `q = ${q}, windows of ${w} observations, one every 21${x ? X : ''}: ${n} windows for Bitcoin. The robust Z* rejects on ${rb}% of the Bitcoin windows, ${re}% for Ethereum and ${rs}% for the S&P 500; with the i.i.d. Z the shares would be ${ib}% (Bitcoin) and ${is_}% (S&P 500). VR(${q}) of Bitcoin between ${mn} and ${mx}.`,
 rc1: (a, b, w, d0, d1) => `${cap(a)} and ${b}: the ${w} common days from ${d0} to ${d1} (prices joined first, then returns).`,
 rc2: (sa, sb, cv, r) => `Standard deviations ${sa}% and ${sb}%, covariance ${cv}; ρ̂ = ${cv}/(${sa} × ${sb}) = ${r}.`,
 rc3: (w, r, se, lo, hi, s0) => `Standard error ≈ (1 − ${r}²)/√${w} = ${se}; an approximate 95% band from ${lo} to ${hi}. Near ρ = 0 the standard error is 1/√${w} = ${s0}: moves of the rolling correlation smaller than about two standard errors are noise.`,
 rc4: (d, wk, nd, nw) => `Whole period: ${d} with ${nd} daily returns and ${wk} with ${nw} weekly returns (Friday to Friday): the asynchronous closes do not explain this correlation.`,
 rc: (w, a, b, e, r, d, se, x) => `${cap(a)} and ${b}, window of ${w} common days${x ? X : ''}: the ${e === 'max' ? 'highest' : e === 'min' ? 'lowest' : 'last'} rolling correlation is ${r}, on ${d} (standard error ${se}).`,
 co: (w, f0, f1, f2, g25, g26, x) => `Window of ${w} common days${x ? X : ''}. Bitcoin and S&P 500: ${f0[0]} before 2020, ${f0[1]} after; rolling maximum ${f0[2]} on ${f0[3]}; last value ${f0[4]}. Bitcoin and gold: ${f1[0]} before 2020, ${f1[1]} after; ${g25} in 2020–2025 and ${g26} in 2026; rolling maximum ${f1[2]} on ${f1[3]}. Ethereum and S&P 500: ${f2[0]} and ${f2[1]}; maximum ${f2[2]} on ${f2[3]}.`,
 cy: (y, v, f, d, w, x) => `${y}${x ? X : ''}: Bitcoin with the S&P 500 ${v[0]}, with gold ${v[1]}; Ethereum with the S&P 500 ${v[2]} (${f === 'w' ? 'weekly' : 'daily'} returns). Whole period, Bitcoin and S&P 500: daily ${d}, weekly ${w}.`,
 cr: (ep, a, b, bt, et, sp, g, w, wd, x) => `${ep}, ${a} – ${b}${x ? X : ''}: Bitcoin ${bt}%, Ethereum ${et}%, S&P 500 ${sp}%, gold ${g}% (cumulative log returns); Bitcoin's worst day: ${w}% on ${wd}.`,
 sh1: (n, nb, q, thr, msp) => `${n} common days; the worst ${q}% are the ${nb} days with an S&P 500 return of at most ${thr}% (on average ${msp}%).`,
 sh2: (nm, m, s, nb, t, p) => `${cap(nm)} on those days: mean ${m}%, standard deviation ${s}%; t = mean/(s/√${nb}) = ${t}, two-sided p ${p} (Student-t with ${nb} − 1 degrees of freedom).`,
 sh3: (neg, cb, ma) => `Negative on ${neg}% of those days; correlation with the S&P 500 on those days ${cb}; mean on all common days ${ma}%.`,
 sh: (q, nb, bm, bt, bp, gm, gt, gp, x) => `Worst ${q}% of S&P 500 days (${nb})${x ? X : ''}: Bitcoin ${bm}% (t = ${bt}, p ${bp}), gold ${gm}% (t = ${gt}, p ${gp}).`,
 md1: (nm, pd, pp, td, tp, d, pts) => `${cap(nm)}: record ${pp} ${pts ? 'points' : 'USD'} on ${pd}, lowest later close ${tp} on ${td}: maximum drawdown D = ${d}%.`,
 md2: (d, one, g) => `Loss d = ${d}: the price is ${one} of the record, so recovery needs a gain of d/(1 − d) = ${g}%.`,
 md3: (rd, days) => `New record on ${rd}, ${days} days after the peak (time under water).`,
 md3b: () => 'No new record by 18 September 2026.',
 md: (d, g, x) => `A loss of ${d}% needs a gain of ${g}% to recover${x ? XS : ''}.`,
 dd: (y, rows, t, last, de, x) => `Daily closes since ${y}${x ? X : ''}: maximum drawdowns ` + rows.map(r => `${r[0]} ${r[1]}% (${r[2]})`).join(', ') + `; days more than ${t}% below the record: ` + rows.map(r => `${r[0]} ${r[3]}%`).join(', ') + `. On ${de} Bitcoin was ${last}% below its record.`,
 de: (th, n, nr, lg, yr, x) => `Episodes deeper than ${th}%${x ? X : ''}: ${n}, of which ${nr} recovered; the longest time under water: ${lg} days (peak of ${yr}).`,
 ef: (w, v0, v1, vl, de, mn, mx, x) => `${w}-day volatility${x ? X : ''}: ${v1}% on 10 January 2023, ${v0}% on 10 January 2024, the day before the ETFs, and ${vl}% on ${de}. The weekend share of the squared returns has been between ${mn}% and ${mx}% each year since 2020, below 2/7 = 28.6%.`,
 efY: (y, mv, sh) => `${y}: ${mv === null ? 'the rolling volatility is shown from 2020' : 'mean volatility ' + mv + '%'}; weekend share ${sh}%.`,
 et: (L, b0, b1, a0, a1, vb, va, bf, p, sig, x) => `Windows ${b0} – ${b1} and ${a0} – ${a1}${x ? X : ''}: volatility ${vb}% before and ${va}% after; Brown–Forsythe statistic ${bf}, p = ${p}: ${sig ? 'the change in volatility is significant at 5%' : 'the change in volatility is not significant at 5%'}.`,
 ib1: (k, s, nd, n) => `IBIT and Bitcoin, prices joined on the ${nd} trading days of the ETF${s === 'all' ? '' : ' in ' + s}; ${+k === 1 ? 'daily returns' : n + ' non-overlapping ' + k + '-day returns'}.`,
 ib2: (c, b) => `Correlation ${c}; slope of IBIT on Bitcoin ${b}.`,
 ib3: (sd, k, ann, te) => `Standard deviation of the return difference ${sd}%; × √(252/${k}) = ${ann}: tracking error ${te}% a year.`,
 ib: (k, c, b, te, t1, t5, c5, x) => `${k}-day returns${x ? X : ''}: correlation ${c}, slope ${b}, tracking error ${te}%. Daily: ${t1}%; weekly (5 days): ${t5}%, with a correlation of ${c5}: the difference averages out over a week.`,
 su: (d, t, u, c, dai, us, cs, dl) => `${d} (DefiLlama, download of ${dl}): all USD stablecoins ${t} billion USD; USDT ${u}, USDC ${c}, DAI ${dai} billion; USDT holds ${us}%, USDC ${cs}%.`,
 su2: (ts, tn, gs, gn, us, un, cs, cn, mx, mxd, tr, trd, c0, c1, dl) => `Slides against the download of ${dl}: total on 18 September 2026, ${ts} against ${tn} billion USD; growth factor since the start of 2020, ${gs} against ${gn}; share of USDT, ${us}% against ${un}%, and of USDC, ${cs}% against ${cn}%; DefiLlama revises its history. Unchanged: the 2022 maximum, ${mx} billion (${mxd}), the trough, ${tr} billion (${trd}), and the fall of USDC from ${c0} to ${c1} billion in 2023.`,
 bp1: (nm, d, f, P) => `${nm}, ${f === 'close' ? 'close' : 'daily low'} on ${d}: P = ${P} USD.`,
 bp2: (P, dv, pc) => `d = 10,000 × (${P} − 1) = ${dv} bp, that is ${pc}%.`,
 bp3: (W, P, loss, below) => (below ? `A holder of ${W} coins who sold at this price lost ${W} × (1 − ${P}) = ${loss} USD.` : `A holder who sold ${W} coins at this price received ${loss} USD more than at the peg.`),
 bp4: (c, l, ll) => `Close ${c} and low ${l}: at the low, the difference from the peg would have been ${ll} USD.`,
 bp: (d, rows, x) => `${d}${x ? XS : ''}: ` + rows.map(r => `${r[0]} ${r[1]} bp at the close, ${r[2]} bp at the low`).join('; ') + '.',
 pg: (f, t, rows, x) => `Daily ${f === 'close' ? 'closes' : 'lows'} since 2021${x ? X : ''}: median absolute deviation ` + rows.map(r => `${r[0]} ${r[1]} bp`).join(', ') + `; days with |d| > ${t} bp: ` + rows.map(r => `${r[0]} ${r[2]}%`).join(', ') + '; half-lives ' + rows.map(r => `${r[0]} ${r[3]}`).join(', ') + ' days.',
 sv: (d, rows) => `${d}: ` + rows.map(r => `${r[0]} close ${r[1]} (${r[2]} bp), low ${r[3]} (${r[4]} bp)`).join('; ') + '.',
 sv2: day => (day < 10 ? 'Before 10 March all three trade within a few basis points of 1 USD.'
   : day <= 12 ? `10 March: SVB is closed (${'""" + FDIC + r"""'}FDIC, 2023</a>); the weekend follows, with banks closed and crypto markets open.`
   : `On the evening of 12 March the U.S. Treasury, the Federal Reserve and the FDIC guarantee all SVB deposits (${'""" + FED + r"""'}U.S. Treasury, Federal Reserve & FDIC, 2023</a>); from 13 March USDC and DAI are back near the peg.`),
 us: (d, pd, pl, sd, sl, pk, pkd, drop) => `Up to ${d}: last price ${pl} USD (${pd}); supply ${sl} billion UST (${sd}); peak supply ${pk} billion on ${pkd}${drop === null ? '' : `; fall from the peak ${drop}%`}.`,
 ud1: (nm, me, v) => `${cap(nm)}, ${me}: a log return of −${v}%.`,
 ud2: (v, e) => `e^(−${v}/100) = ${e}: the position keeps ${e} of its value.`,
 ud3: (W, one, loss) => `Loss = ${W} × ${one} = ${loss} USD.`,
 ud4: (lin, diff) => `Read as a simple return (W × v/100), the same number would give ${lin} USD, ${diff} USD too much.`,
 ud: (W, rows, x) => `Position of ${W} USD${x ? XS : ''}: VaR 1% HS ` + rows.map(r => `${r[0]} ${r[1]} USD`).join(', ') + '; ES 2.5% HS ' + rows.map(r => `${r[0]} ${r[2]} USD`).join(', ') + '.',
 vq: (a, bh, sh, ratio, under, gv, nu, gs, x) => `VaR ${a}%${x ? X : ''}: Bitcoin ${bh}% against ${sh}% for the S&P 500 (HS), ${ratio} times as much; Bitcoin's Normal VaR is ${num(under) >= 0 ? under + '% below' : abs(under) + '% above'} the historical one; GARCH-t gives ${gv}% for the next day (σ̂ = ${gs}%); the fitted Student-t has ν̂ = ${nu}.`,
 sq1: (nm, v) => `${cap(nm)}: one-day HS VaR 1% = ${v}%.`,
 sq2: (h, s, v, r) => `Square-root-of-time rule: √${h} × ${v} = ${s} × ${v} = ${r}%.`,
 sq3: (h, v, u) => `Historical VaR 1% of the overlapping ${h}-day returns: ${v}% (${u} USD on 100,000 USD).`,
 sq4: (p, over, h) => (+h === 1 ? 'For h = 1 the two coincide.' : over ? `The rule overstates by ${p}%: with a tail index above 2, an extreme quantile of a sum grows more slowly than √h.` : `Here the rule understates by ${p}%.`),
 sq: (h, nm, hs, r, x) => `${cap(nm)}, h = ${h}${x ? X : ''}: ${hs}% historical against ${r}% from the square-root-of-time rule.`,
 kp1: (x, n, r, e) => `x = ${x} exceptions in n = ${n} days: π̂ = ${r}%; the expected number αn = ${e}.`,
 kp2: (l0, l1) => `Log-likelihood under the rate α: ${l0}; under π̂: ${l1}.`,
 kp3: (lr, p, rej) => `LR_uc = −2 × (the difference) = ${lr}, p ${p}: ${rej ? 'rejected at 5%' : 'not rejected at 5%'}.`,
 kp4: (lo, hi) => `For this n and α, the test does not reject for x between ${lo} and ${hi}.`,
 kp: (x, lr, p, lo, hi, xs) => `x = ${x}${xs ? XS : ''}: LR_uc = ${lr}, p ${p}; not rejected for ${lo} ≤ x ≤ ${hi}.`,
 bt: (w, a, n, H, G, g18, h18, x) => `${n} one-day forecasts since 1 January 2018, VaR ${a}%, HS on ${w} days${x ? X : ''}: HS ${H[0]} exceptions (${H[1]}%), GARCH-t ${G[0]} (${G[1]}%), against ${H[2]} expected; Kupiec LR_uc = ${H[3]} (p = ${H[4]}) and ${G[3]} (p = ${G[4]}); mean VaR ${H[5]}% (HS) and ${G[5]}% (GARCH-t); exceptions in 2018: ${h18} (HS) and ${g18} (GARCH-t).`
};
})();
""",
}

QA = {
 'ro': {
  'halving': ('▶ scurtează orizontul la 180 de zile (rulare suplimentară). Cu cît era mai mare prețul la 180 de zile după halving-ul din aprilie 2024?',
              'Cu 3,7%, față de 31% la un an. Pentru halving-urile din 2016 și 2020 creșterile la 180 de zile sînt 56% și 72%. Variația după o dată cunoscută din timp depinde '
              'mult de orizontul ales, iar trei observații nu permit o concluzie: pe o piață eficientă, reducerea anunțată a ofertei este deja în preț.'),
  'ann-we': ('Cu cît subestimează regula celor 252 de zile volatilitatea anuală a Bitcoin, față de A = 365?',
             'Cu 16,9% = 1 − √(252/365): 55,3% în loc de 66,6%. Media este subestimată mai mult, cu 31,0% = 1 − 252/365 (29,8% în loc de 43,1%), pentru că media crește cu A, '
             'iar volatilitatea doar cu √A.'),
  'desc-tab': ('▶ începe toate seriile pe 1 ianuarie 2024 (rulare suplimentară). De cîte ori este Bitcoin mai volatil decît S&P 500?',
               'De 3,1 ori (47,5% față de 15,2%), față de 3,8 ori pe întregul eșantion; Ethereum de 4,5 ori, Solana de 5,2 ori. Volatilitatea Bitcoin a scăzut, dar raportul '
               'rămîne mare.'),
  'rvol': ('Cu o fereastră de 7 zile (▶, rulare suplimentară), în ce proporție din zile este volatilitatea Bitcoin sub cea a S&P 500?',
           'În 3,8% din zilele comune, față de 0,11% cu fereastra de 30 de zile. Pe 7 randamente estimările sînt mult mai zgomotoase, deci cele două curbe se intersectează mai des, '
           'deși raportul median rămîne 3,9.'),
  'weekday': ('▶ adaugă cîte un an. În ce an este cel mai mic raportul dintre varianța din weekend și cea din zilele lucrătoare la Bitcoin?',
              'În 2022: 0,30 (Brown–Forsythe p < 0,001), apoi 2023 (0,31) și 2024 și 2026 (0,32). Înainte de 2020 raportul era mai aproape de 1: 0,66 în 2015 și 0,89 în 2016. '
              'Media din weekend nu diferă în niciun an semnificativ de cea din zilele lucrătoare.'),
  'hill-tab': ('Cu k = 10% din n (▶, rulare suplimentară), ce se întîmplă cu α̂ al cozii stîngi a Bitcoin?',
               'Scade la 1,83 (k = 438), de la 2,71 cu k = 2,5%; cu k = 1% urcă la 3,83. Un k mare include observații din centrul distribuției, unde legea putere nu se aplică, '
               'iar estimația scade; de aceea k = 2,5% din n este o alegere, nu un rezultat.'),
  'lb-we': ('Pentru randamentele Bitcoin rₜ, de la ce m este Q(m) pentru prima dată semnificativ la 5% (▶, m = 1 … 30)?',
            'De la m = 10: Q(10) = 20,2 față de valoarea critică 18,3. La m = 12 nu mai este (20,6 < 21,0), iar la m = 30 nici (43,5 < 43,8). Autocorelațiile randamentelor sînt '
            'mici, deci verdictul depinde de numărul de laguri; pentru rₜ² Q(m) depășește valoarea critică la orice m.'),
  'acf': ('▶ trece prin cele cinci serii. Care are cea mai mare autocorelație de ordinul 1 a pătratelor randamentelor?',
          'S&P 500: ρ̂(1) = 0,451, apoi Solana 0,247, Ethereum 0,175, Bitcoin 0,137 și aurul 0,090. Volatility clustering nu ține de nivelul volatilității: acțiunile au '
          'volatilitate mai mică, dar mai persistentă de la o zi la alta.'),
  'garch-tab': ('Ce parte dintr-un șoc al volatilității mai rămîne în varianța S&P 500 după 200 de zile?',
                'Circa 25%: 0,9931²⁰⁰ = 0,251, adică doi timpi de înjumătățire de 100 de zile. Pentru aur rămîne 29%, pentru Bitcoin practic 100%, pentru că α̂ + β̂ = 1.'),
  'garch': ('▶ prelungește prognoza la 365 de zile. Ce volatilitate dă modelul pentru Bitcoin peste un an?',
            'Circa 169%, de la 54% pentru ziua următoare. Cu α̂ + β̂ = 1, varianța prognozată crește în fiecare zi cu ω̂ = 0,194, fără o revenire la un nivel de termen lung. '
            'Pentru S&P 500 prognoza urcă de la 11,5% la 28,4%, spre nivelul de termen lung de 29,4%.'),
  'vr-tab': ('Cu q = 2 (▶, rulare suplimentară), respinge vreo statistică Z* pe întregul eșantion mersul aleator la 5%?',
             'Da, pentru S&P 500: Z*(2) = −2,30 (p 0,022), pentru că VR(2) = 0,875 ≈ 1 + ρ̂(1) reflectă revenirea zilnică a indicelui spre medie. Bitcoin (−0,88) și Ethereum '
             '(−0,43) nu resping. Respingerea vine din piața considerată cea mai eficientă, nu din cripto.'),
  'rvr': ('Cu ferestre de 1 000 de observații (▶, rulare suplimentară), ce proporție din ferestrele S&P 500 resping cu statistica i.i.d. Z(5)?',
          'Circa 50% (50,5%), față de 21,7% cu ferestre de 500, iar cu Z* robust tot 0%. O fereastră mai lungă dă mai multă putere unui test greșit: ignorarea volatility '
          'clustering produce respingeri false.'),
  'rc-we': ('Cu o fereastră de 60 de zile (▶, rulare suplimentară), maximul corelației mobile Bitcoin–S&P 500 este peste sau sub 0,57?',
            'Peste: 0,67, pe 25 octombrie 2024. Eroarea standard lîngă ρ = 0 este 1/√60 = 0,13, față de 0,063 cu 250 de zile: ferestrele scurte dau valori extreme mai mari.'),
  'corr': ('Cu o fereastră de 500 de zile (▶, rulare suplimentară), cît este maximul corelației mobile Bitcoin–aur?',
           'Circa 0,25, pe 18 septembrie 2026, față de 0,37 pe 11 septembrie 2026 cu 250 de zile. Fereastra lungă amestecă anul 2026 cu anii 2024–2025, cînd corelația era 0,12, '
           'deci creșterea din 2026 apare diluată.'),
  'corr-tab': ('Cu randamente săptămînale (rulare suplimentară), corelația Bitcoin–S&P 500 din 2022 este mai mare sau mai mică decît cu randamente zilnice?',
               'Mai mică: 0,43 față de 0,56; pe toată perioada 0,19 față de 0,24. Închiderile asincrone ar fi trebuit să micșoreze corelația zilnică, nu pe cea săptămînală; '
               'pe un an rămîn doar circa 50 de randamente săptămînale, iar estimația este mai zgomotoasă.'),
  'crisis-tab': ('În episodul FTX (5–21 noiembrie 2022), ce a făcut S&P 500?',
                 'A crescut cu 4,6%, iar aurul cu 3,4%, în timp ce Bitcoin a pierdut 29,9% și Ethereum 38,4%. Un șoc specific pieței cripto nu a ajuns la acțiuni.'),
  'haven-we': ('În cele mai proaste 1% din zilele S&P 500 (▶, rulare suplimentară), cît este randamentul mediu al aurului?',
               '−0,39% (t = −1,27, p 0,215), deci nu diferă semnificativ de zero în cele 31 de zile, în timp ce Bitcoin are −6,30% (t = −3,23, p 0,003). Aurul își păstrează valoarea, '
               'fără să crească în cele mai grele zile.'),
  'mdd-we': ('Drawdown-ul maxim al Ethereum este −94,0%. Ce cîștig este necesar pentru recuperare?',
             '1 556% = 0,940/0,060: din 6,0% din valoarea de record trebuie să urce de circa 16,6 ori. Ethereum a revenit la record pe 2 februarie 2021.'),
  'dd': ('▶ începe eșantionul în 2022 (rulare suplimentară). Cît este drawdown-ul maxim al Bitcoin?',
         '−66,9%, față de −83,4% din 2016. Drawdown-ul se măsoară față de cel mai mare preț din eșantion, iar vîrful din noiembrie 2021 nu mai este inclus. Ethereum −74,1%, '
         'S&P 500 −25,4%.'),
  'dd-tab': ('Cu pragul de 30% (▶, rulare suplimentară), cîte episoade are Bitcoin din septembrie 2014?',
             'Opt: la cele cinci de pe slide se adaugă trei scăderi din 2017 (−32,6%, −34,8% și −35,5%) care au fost recuperate în cîteva luni.'),
  'etf': ('În ce an din 2020 a fost cea mai mare ponderea weekendului în suma pătratelor randamentelor Bitcoin?',
          'În 2025: 19,0%, față de 10,9–14,0% în ceilalți ani; tot sub 2/7 = 28,6%, valoarea pentru o varianță egală în fiecare zi.'),
  'etf-tab': ('Cu ferestre de un an (▶, rulare suplimentară), găsește testul Brown–Forsythe o schimbare semnificativă a volatilității?',
              'Da (p 0,001), dar volatilitatea a crescut, de la 44,4% la 52,9%. Cu 2,5 ani este tot semnificativă (p 0,008), dar în sens invers (de la 57,2% la 48,0%), iar cu doi ani '
              'nu este (p 0,27). Rezultatul depinde de fereastră, deci comparația înainte–după nu măsoară un efect cauzal al ETF-urilor.'),
  'ibit-we': ('Cu randamente pe 5 zile (▶, rulare suplimentară), cît este tracking error-ul IBIT?',
              '8,9% pe an, cu corelația 0,98, față de 20,8% zilnic. Diferența dintre închiderea de la New York și cea de la miezul nopții UTC se compensează în cîteva zile.'),
  'supply': ('Cu cît a scăzut oferta totală de stablecoins între 2 aprilie 2022 (187 de miliarde USD) și 19 august 2023 (123 de miliarde)?',
             'Cu 34,5% = 1 − 122,7/187,4. Pe slide-uri totalul de pe 18 septembrie 2026 este 308 de miliarde USD; descărcarea din 4 noiembrie 2026 dă 303 de miliarde pentru aceeași dată, '
             'pentru că DefiLlama își revizuiește istoricul.'),
  'bp-we': ('Minimul DAI din 11 martie 2023 a fost 0,8970 USD. Cît este abaterea de la paritate?',
            '−1 030 bp = 10 000 × (0,8970 − 1), adică −10,3%. Minimul USDC din aceeași zi a fost −1 226 bp.'),
  'peg': ('Cu pragul de 50 bp (▶), în ce proporție din zile se abate închiderea DAI mai mult?',
          'În 0,57% din zile (12 din 2 087), față de 0,14% la USDT și la USDC; la pragul de 10 bp proporțiile sînt 10,7%, 11,4% și 1,5%.'),
  'svb': ('Unde a închis USDC pe 13 martie 2023?',
          'La 0,9989 USD, adică −11 bp, înapoi aproape de paritate după garantarea depozitelor SVB din seara de 12 martie (' + FED + 'U.S. Treasury, Federal Reserve și FDIC, 2023</a>). '
          'Pe 11 martie închiderea fusese 0,9715 (−285 bp). USDT a închis pe 13 martie la +39 bp.'),
  'ust': ('Cu cît scăzuse oferta UST între maximul din 7 mai 2022 și ultima zi publicată, 11 mai?',
          'Cu 27%, de la 18,8 la 13,7 miliarde UST, în patru zile. Prețul a scăzut mai mult: 0,75 USD pe 9 mai, 0,37 pe 13 mai și 0,02 pe 31 mai.'),
  'usd-we': ('VaR 1% HS pentru S&P 500 este un randament logaritmic de −3,256%. Cît înseamnă pentru o poziție de 1 000 000 USD?',
             '32 037 USD = 1 000 000 × (1 − e^(−0,03256)). Citit ca randament simplu, același număr ar da 32 562 USD.'),
  'var-tab': ('La α = 5% (▶, rulare suplimentară), este VaR-ul Normal al Bitcoin tot sub cel istoric?',
              'Nu: 5,62% față de 5,33% istoric, cu 5% peste. Distribuția Bitcoin are cozi groase și un centru mai ascuțit decît distribuția Normală: la 1% distribuția Normală subestimează '
              'pierderea (7,99% față de 10,54%), la 5% o supraestimează.'),
  'sqrt-we': ('Pentru Bitcoin și h = 30 de zile (▶), regula rădăcinii pătrate dă mai mult sau mai puțin decît VaR 1% istoric pe 30 de zile?',
              'Mai mult: 57,7% față de 48,2%, cu 20% peste. Cu un tail index între 2 și 4, cuantilele extreme ale sumelor cresc mai încet decît √h.'),
  'kup-we': ('Cu n = 3 183 de zile și α = 1%, ce numere de excepții acceptă testul Kupiec la 5%?',
             'De la 22 la 43. HS (29) și GARCH-t (41) sînt în interval; 41 este aproape de margine (LR_uc = 2,45, p 0,12). Testul respinge și prea puține excepții, nu doar prea multe.'),
  'btcvar': ('Cu o fereastră HS de 1 000 de zile (▶, rulare suplimentară), ce spune testul Kupiec?',
             'Respinge, pentru prea puține excepții: 21 (0,66%), p 0,04. Fereastra lungă păstrează pierderile din 2018–2022 și dă un VaR mediu de 10,1%, prea mare pentru anii mai calmi.'),
 },
 'en': {
  'halving': ('▶ shortens the horizon to 180 days (extra run). How much higher was the price 180 days after the April 2024 halving?',
              'By 3.7%, against 31% after one year. For the 2016 and 2020 halvings the 180-day changes are 56% and 72%. The change after a date known in advance depends strongly on '
              'the horizon chosen, and three observations allow no conclusion: in an efficient market, an announced cut in supply is already in the price.'),
  'ann-we': ('By how much does the 252 rule understate Bitcoin’s annual volatility, compared with A = 365?',
             'By 16.9% = 1 − √(252/365): 55.3% instead of 66.6%. The mean is understated more, by 31.0% = 1 − 252/365 (29.8% instead of 43.1%), because the mean scales with A '
             'and the volatility only with √A.'),
  'desc-tab': ('▶ starts all series on 1 January 2024 (extra run). How many times as volatile as the S&P 500 is Bitcoin?',
               '3.1 times (47.5% against 15.2%), against 3.8 times on the whole sample; Ethereum 4.5 times, Solana 5.2 times. Bitcoin’s volatility has fallen, but the ratio stays large.'),
  'rvol': ('With a 7-day window (▶, extra run), on what share of the days is Bitcoin’s volatility below that of the S&P 500?',
           'On 3.8% of the common days, against 0.11% with the 30-day window. On 7 returns the estimates are much noisier, so the two curves cross more often, although the median '
           'ratio stays 3.9.'),
  'weekday': ('▶ adds one year at a time. In which year is Bitcoin’s ratio of the weekend to the weekday variance lowest?',
              'In 2022: 0.30 (Brown–Forsythe p < 0.001), then 2023 (0.31) and 2024 and 2026 (0.32). Before 2020 the ratio was closer to 1: 0.66 in 2015 and 0.89 in 2016. The weekend '
              'mean is not significantly different from the weekday mean in any year.'),
  'hill-tab': ('With k = 10% of n (▶, extra run), what happens to Bitcoin’s left-tail α̂?',
               'It falls to 1.83 (k = 438), from 2.71 with k = 2.5%; with k = 1% it rises to 3.83. A large k takes in observations from the centre of the distribution, where the power '
               'law does not hold, and the estimate falls; k = 2.5% of n is a choice, not a result.'),
  'lb-we': ('For Bitcoin’s returns rₜ, from which m is Q(m) significant at 5% for the first time (▶, m = 1 … 30)?',
            'From m = 10: Q(10) = 20.2 against the critical value 18.3. At m = 12 it is not (20.6 < 21.0), nor at m = 30 (43.5 < 43.8). The autocorrelations of the returns are small, '
            'so the verdict depends on the number of lags; for rₜ² Q(m) exceeds the critical value at every m.'),
  'acf': ('▶ goes through the five series. Which has the largest first-order autocorrelation of the squared returns?',
          'The S&P 500: ρ̂(1) = 0.451, then Solana 0.247, Ethereum 0.175, Bitcoin 0.137 and gold 0.090. Volatility clustering is not about the level of volatility: equities have a '
          'lower volatility, but one that is more persistent from one day to the next.'),
  'garch-tab': ('What share of a volatility shock is still in the S&P 500 variance after 200 days?',
                'About 25%: 0.9931²⁰⁰ = 0.251, two half-lives of 100 days. For gold 29% is left, for Bitcoin practically 100%, because α̂ + β̂ = 1.'),
  'garch': ('▶ extends the forecast to 365 days. What volatility does the model give for Bitcoin one year ahead?',
            'About 169%, from 54% for the next day. With α̂ + β̂ = 1 the forecast variance grows by ω̂ = 0.194 every day, with no return to a long-run level. For the S&P 500 the '
            'forecast rises from 11.5% to 28.4%, towards the long-run level of 29.4%.'),
  'vr-tab': ('With q = 2 (▶, extra run), does any whole-sample Z* reject the random walk at 5%?',
             'Yes, for the S&P 500: Z*(2) = −2.30 (p 0.022), because VR(2) = 0.875 ≈ 1 + ρ̂(1) reflects the daily mean reversion of the index. Bitcoin (−0.88) and Ethereum (−0.43) '
             'do not reject. The rejection comes from the market seen as the most efficient, not from crypto.'),
  'rvr': ('With windows of 1,000 observations (▶, extra run), what share of the S&P 500 windows reject with the i.i.d. Z(5)?',
          'About 50% (50.5%), against 21.7% with windows of 500, and still 0% with the robust Z*. A longer window gives more power to a wrong test: ignoring volatility clustering '
          'creates false rejections.'),
  'rc-we': ('With a 60-day window (▶, extra run), is the highest Bitcoin–S&P 500 rolling correlation above or below 0.57?',
            'Above: 0.67, on 25 October 2024. The standard error near ρ = 0 is 1/√60 = 0.13, against 0.063 with 250 days: short windows give more extreme values.'),
  'corr': ('With a 500-day window (▶, extra run), what is the maximum of the Bitcoin–gold rolling correlation?',
           'About 0.25, on 18 September 2026, against 0.37 on 11 September 2026 with 250 days. The long window mixes 2026 with 2024–2025, when the correlation was 0.12, so the rise '
           'of 2026 is diluted.'),
  'corr-tab': ('With weekly returns (extra run), is the 2022 correlation of Bitcoin and the S&P 500 higher or lower than with daily returns?',
               'Lower: 0.43 against 0.56; over the whole period 0.19 against 0.24. Asynchronous closes would lower the daily correlation, not the weekly one; one year has only about 50 '
               'weekly returns, so the estimate is noisier.'),
  'crisis-tab': ('During the FTX episode (5–21 November 2022), what did the S&P 500 do?',
                 'It rose by 4.6%, and gold by 3.4%, while Bitcoin lost 29.9% and Ethereum 38.4%. A shock specific to crypto did not reach equities.'),
  'haven-we': ('On the worst 1% of S&P 500 days (▶, extra run), what is gold’s mean return?',
               '−0.39% (t = −1.27, p 0.215), not significantly different from zero on those 31 days, while Bitcoin has −6.30% (t = −3.23, p 0.003). Gold holds its value, without rising '
               'on the hardest days.'),
  'mdd-we': ('Ethereum’s maximum drawdown is −94.0%. What gain does it need to recover?',
             '1,556% = 0.940/0.060: from 6.0% of its record value it has to rise about 16.6 times. Ethereum was back at its record on 2 February 2021.'),
  'dd': ('▶ starts the sample in 2022 (extra run). What is Bitcoin’s maximum drawdown?',
         '−66.9%, against −83.4% since 2016. The drawdown is measured from the highest price in the sample, and the peak of November 2021 is no longer in it. Ethereum −74.1%, '
         'S&P 500 −25.4%.'),
  'dd-tab': ('With a threshold of 30% (▶, extra run), how many Bitcoin episodes are there since September 2014?',
             'Eight: three falls of 2017 (−32.6%, −34.8% and −35.5%), recovered within months, join the five of the slide.'),
  'etf': ('In which year since 2020 was the weekend share of Bitcoin’s squared returns highest?',
          'In 2025: 19.0%, against 10.9–14.0% in the other years; still below 2/7 = 28.6%, the value for an equal variance on every day.'),
  'etf-tab': ('With windows of one year (▶, extra run), does the Brown–Forsythe test find a significant change in volatility?',
              'Yes (p 0.001), but volatility rose, from 44.4% to 52.9%. With 2.5 years it is also significant (p 0.008), in the opposite direction (57.2% to 48.0%), and with two '
              'years it is not (p 0.27). The result depends on the window, so the before–after comparison does not measure a causal effect of the ETFs.'),
  'ibit-we': ('With 5-day returns (▶, extra run), what is IBIT’s tracking error?',
              '8.9% a year, with a correlation of 0.98, against 20.8% for daily returns. The gap between the New York close and the midnight UTC close averages out within days.'),
  'supply': ('By how much did the total stablecoin supply fall between 2 April 2022 (187 billion USD) and 19 August 2023 (123 billion)?',
             'By 34.5% = 1 − 122.7/187.4. On the slides the total on 18 September 2026 is 308 billion USD; the download of 4 November 2026 gives 303 billion for the same date, '
             'because DefiLlama revises its history.'),
  'bp-we': ('DAI’s low on 11 March 2023 was 0.8970 USD. What is its peg deviation?',
            '−1,030 bp = 10,000 × (0.8970 − 1), that is −10.3%. The USDC low on the same day was −1,226 bp.'),
  'peg': ('With a threshold of 50 bp (▶), on what share of the days does DAI’s close deviate by more?',
          'On 0.57% of the days (12 of 2,087), against 0.14% for USDT and for USDC; at the 10 bp threshold the shares are 10.7%, 11.4% and 1.5%.'),
  'svb': ('Where did USDC close on 13 March 2023?',
          'At 0.9989 USD, that is −11 bp, back near the peg after the guarantee of the SVB deposits on the evening of 12 March (' + FED + 'U.S. Treasury, Federal Reserve & FDIC, 2023</a>). '
          'On 11 March the close had been 0.9715 (−285 bp). USDT closed at +39 bp on 13 March.'),
  'ust': ('By how much had the UST supply fallen between its peak on 7 May 2022 and the last published day, 11 May?',
          'By 27%, from 18.8 to 13.7 billion UST, in four days. The price fell further: 0.75 USD on 9 May, 0.37 on 13 May and 0.02 on 31 May.'),
  'usd-we': ('The S&P 500 HS VaR 1% is a log return of −3.256%. What is it for a position of 1,000,000 USD?',
             '32,037 USD = 1,000,000 × (1 − e^(−0.03256)). Read as a simple return, the same number would give 32,562 USD.'),
  'var-tab': ('At α = 5% (▶, extra run), is Bitcoin’s Normal VaR still below the historical one?',
              'No: 5.62% against 5.33% historical, 5% above. Bitcoin’s distribution has heavy tails and a more peaked centre than the Normal distribution: at 1% the Normal distribution '
              'understates the loss (7.99% against 10.54%), at 5% it overstates it.'),
  'sqrt-we': ('For Bitcoin and h = 30 days (▶), does the square-root-of-time rule give more or less than the historical 30-day VaR 1%?',
              'More: 57.7% against 48.2%, 20% above. With a tail index between 2 and 4, the extreme quantiles of sums grow more slowly than √h.'),
  'kup-we': ('With n = 3,183 days and α = 1%, which numbers of exceptions does the Kupiec test accept at 5%?',
             'From 22 to 43. HS (29) and GARCH-t (41) are inside; 41 is near the edge (LR_uc = 2.45, p 0.12). The test also rejects too few exceptions, not only too many.'),
  'btcvar': ('With an HS window of 1,000 days (▶, extra run), what does the Kupiec test say?',
             'It rejects, for too few exceptions: 21 (0.66%), p 0.04. The long window keeps the losses of 2018–2022 and gives a mean VaR of 10.1%, too large for the calmer years.'),
 },
}

PREDICT = {
 'halving': (0, [('Circa 3,7%.', 'About 3.7%.'), ('Circa 15,5%: jumătate din creșterea de 31% pe un an.', 'About 15.5%: half of the one-year rise of 31%.'),
                 ('Tot circa 31%: efectul halving-ului apare imediat.', 'Still about 31%: the effect of the halving is immediate.')]),
 'ann-we': (0, [('Cu 16,9%.', 'By 16.9%.'), ('Cu 31,0% = 1 − 252/365.', 'By 31.0% = 1 − 252/365.'), ('Cu 20,4% = √(365/252) − 1.', 'By 20.4% = √(365/252) − 1.')]),
 'desc-tab': (0, [('De circa 3,1 ori.', 'About 3.1 times.'), ('Tot de 3,8 ori: raportul nu depinde de perioadă.', 'Still 3.8 times: the ratio does not depend on the period.'),
                  ('De circa o dată: după ETF-uri Bitcoin se mișcă la fel ca acțiunile.', 'About once: after the ETFs Bitcoin moves like equities.')]),
 'rvol': (0, [('Circa 3,8%.', 'About 3.8%.'), ('Tot 0,11%: fereastra nu schimbă comparația.', 'Still 0.11%: the window does not change the comparison.'),
              ('0%: Bitcoin este mereu mai volatil.', '0%: Bitcoin is always more volatile.')]),
 'weekday': (0, [('2022.', '2022.'), ('2016: piața timpurie, cu puțini investitori instituționali.', '2016: the early market, with few institutional traders.'),
                 ('2024: ETF-urile au mutat tranzacționarea în orele bursei.', '2024: the ETFs moved trading to stock-market hours.')]),
 'hill-tab': (0, [('Scade, la circa 1,8.', 'It falls, to about 1.8.'), ('Rămîne circa 2,7: α̂ nu depinde de k.', 'It stays about 2.7: α̂ does not depend on k.'),
                  ('Crește peste 4: mai multe date înseamnă cozi mai subțiri.', 'It rises above 4: more data mean thinner tails.')]),
 'lb-we': (0, [('m = 10.', 'm = 10.'), ('m = 1: ρ̂(1) este cea mai mare autocorelație.', 'm = 1: ρ̂(1) is the largest autocorrelation.'),
               ('La orice m: randamentele Bitcoin sînt autocorelate.', 'At every m: Bitcoin’s returns are autocorrelated.')]),
 'acf': (0, [('S&P 500.', 'The S&P 500.'), ('Solana: cea mai volatilă serie are cel mai mult volatility clustering.', 'Solana: the most volatile series has the most volatility clustering.'),
             ('Bitcoin.', 'Bitcoin.')]),
 'garch-tab': (0, [('Circa 25%.', 'About 25%.'), ('0%: după doi timpi de înjumătățire șocul a dispărut.', '0%: after two half-lives the shock is gone.'),
                   ('50%: timpul de înjumătățire este momentul în care scăderea se oprește.', '50%: the half-life is the time when the decay stops.')]),
 'garch': (0, [('Circa 170%: crește în continuare.', 'About 170%: it keeps rising.'), ('Revine la volatilitatea din eșantion, 66,6%.', 'It returns to the sample volatility, 66.6%.'),
               ('Rămîne la 54%, prognoza pentru ziua următoare.', 'It stays at 54%, the forecast for the next day.')]),
 'vr-tab': (0, [('Da, S&P 500.', 'Yes, the S&P 500.'), ('Da, Bitcoin: piața cea mai puțin eficientă.', 'Yes, Bitcoin: the least efficient market.'),
                ('Nu, nicio serie.', 'No, no series.')]),
 'rvr': (0, [('Circa 50%.', 'About 50%.'), ('Tot circa 22%: lungimea ferestrei nu schimbă proporția.', 'Still about 22%: the window length does not change the share.'),
             ('0%: ferestrele lungi elimină zgomotul.', '0%: long windows average out the noise.')]),
 'rc-we': (0, [('Peste, circa 0,67.', 'Above, about 0.67.'), ('Tot 0,57: maximul nu depinde de fereastră.', 'Still 0.57: the maximum does not depend on the window.'),
               ('Sub: mai puține zile dau o corelație mai mică.', 'Below: fewer days give a smaller correlation.')]),
 'corr': (0, [('Circa 0,25.', 'About 0.25.'), ('Tot 0,37.', 'Still 0.37.'), ('Circa 0,5: o fereastră mai lungă arată o legătură mai puternică.', 'About 0.5: a longer window shows a stronger link.')]),
 'corr-tab': (0, [('Mai mică.', 'Lower.'), ('Mai mare: închiderile asincrone ascund legătura în datele zilnice.', 'Higher: the asynchronous closes hide the link in daily data.'),
                  ('La fel: frecvența nu schimbă corelația.', 'The same: the frequency does not change the correlation.')]),
 'crisis-tab': (0, [('A crescut cu circa 4,6%.', 'It rose by about 4.6%.'), ('A scăzut cu circa 30%, ca Bitcoin.', 'It fell by about 30%, like Bitcoin.'),
                    ('A scăzut cu circa 5%, ca în episodul Terra/Luna.', 'It fell by about 5%, as in the Terra/Luna episode.')]),
 'haven-we': (0, [('Circa −0,4%, nesemnificativ.', 'About −0.4%, not significant.'), ('Pozitiv: aurul crește în crize.', 'Positive: gold rises in crashes.'),
                  ('Circa −6%, ca Bitcoin.', 'About −6%, like Bitcoin.')]),
 'mdd-we': (0, [('Circa 1 556%.', 'About 1,556%.'), ('94%: cîștigul egal cu pierderea.', '94%: a gain equal to the loss.'), ('188% = 2 × 94%.', '188% = 2 × 94%.')]),
 'dd': (0, [('Circa −66,9%.', 'About −66.9%.'), ('Tot −83,4%: drawdown-ul maxim nu depinde de începutul eșantionului.', 'Still −83.4%: the maximum drawdown does not depend on the start of the sample.'),
            ('−76,6%, ca în episodul care începe în noiembrie 2021.', '−76.6%, as in the episode that starts in November 2021.')]),
 'dd-tab': (0, [('8.', '8.'), ('Tot 5: pragul nu schimbă numărul de episoade.', 'Still 5: the threshold does not change the number of episodes.'), ('2.', '2.')]),
 'etf': (0, [('2025.', '2025.'), ('2020: anul crahului COVID-19.', '2020: the year of the COVID-19 crash.'), ('2024: anul lansării ETF-urilor.', '2024: the year the ETFs started.')]),
 'etf-tab': (0, [('Da, dar volatilitatea a crescut.', 'Yes, but volatility rose.'), ('Da, volatilitatea a scăzut, ca în comparația pe doi ani.', 'Yes, volatility fell, as in the two-year comparison.'),
                 ('Nu: p rămîne circa 0,27.', 'No: p stays about 0.27.')]),
 'ibit-we': (0, [('Circa 8,9%.', 'About 8.9%.'), ('Tot 20,8%: tracking error-ul nu depinde de orizont.', 'Still 20.8%: the tracking error does not depend on the horizon.'),
                 ('Circa 46% = 20,8% × √5.', 'About 46% = 20.8% × √5.')]),
 'supply': (0, [('Cu circa 35%.', 'By about 35%.'), ('Cu circa 53% = 187/123 − 1.', 'By about 53% = 187/123 − 1.'), ('Cu circa 65%: diferența de 65 de miliarde citită ca procent.', 'By about 65%: the gap of 65 billion read as a percentage.')]),
 'bp-we': (0, [('−1 030 bp.', '−1,030 bp.'), ('−10,3 bp: procentul citit ca puncte de bază.', '−10.3 bp: the percentage read as basis points.'), ('−103 bp: înmulțit cu 1 000.', '−103 bp: multiplied by 1,000.')]),
 'peg': (0, [('Circa 0,6%.', 'About 0.6%.'), ('Tot circa 10,7%, ca la 10 bp.', 'Still about 10.7%, as at 10 bp.'), ('0%: DAI nu s-a abătut niciodată cu peste 50 bp.', '0%: DAI never deviated by more than 50 bp.')]),
 'svb': (0, [('Aproape de paritate, circa −11 bp.', 'Near the peg, about −11 bp.'), ('Tot circa −285 bp, ca pe 11 martie.', 'Still about −285 bp, as on 11 March.'),
             ('Peste paritate, ca USDT.', 'Above the peg, like USDT.')]),
 'ust': (0, [('Cu circa 27%.', 'By about 27%.'), ('Cu circa 98%, cît prețul.', 'By about 98%, as much as the price.'), ('Deloc: oferta unui stablecoin algoritmic este fixă.', 'Not at all: the supply of an algorithmic stablecoin is fixed.')]),
 'usd-we': (0, [('32 037 USD.', '32,037 USD.'), ('32 562 USD: randamentul logaritmic citit ca randament simplu.', '32,562 USD: the log return read as a simple return.'),
                ('3 204 USD: cifra pentru 100 000 USD.', '3,204 USD: the figure for 100,000 USD.')]),
 'var-tab': (0, [('Nu, este peste cel istoric.', 'No, it is above the historical one.'), ('Da, tot cu circa 24% sub.', 'Yes, still about 24% below.'),
                 ('Sînt egale: la 5% cozile nu mai contează.', 'They are equal: at 5% the tails no longer matter.')]),
 'sqrt-we': (0, [('Mai mult.', 'More.'), ('Mai puțin: cozile groase fac riscul să crească mai repede decît √h.', 'Less: heavy tails make risk grow faster than √h.'),
                 ('La fel: regula este exactă.', 'The same: the rule is exact.')]),
 'kup-we': (0, [('De la 22 la 43.', 'From 22 to 43.'), ('Doar 31 sau 32, adică αn.', 'Only 31 or 32, that is αn.'), ('De la 0 la 31: mai puține excepții decît αn sînt întotdeauna acceptate.', 'From 0 to 31: fewer exceptions than αn are always accepted.')]),
 'btcvar': (0, [('Respinge: prea puține excepții.', 'It rejects: too few exceptions.'), ('Trece, ca pentru 365 de zile.', 'It passes, as for 365 days.'),
                ('Respinge: prea multe excepții, fereastra lungă reacționează încet.', 'It rejects: too many exceptions, the long window reacts slowly.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout and those elements
VEIL = {a: {'out': True} for a in PREDICT}
VEIL.update({'ann-we': {'out': True, 'charts': ['an-we-box']}, 'crisis-tab': {'out': True, 'charts': ['cr-tab']}, 'etf': {'out': True, 'charts': ['ef-chart']},
             'peg': {'out': True, 'charts': ['pg-tab']}, 'svb': {'out': True, 'charts': ['sv-chart']}, 'ust': {'out': True, 'charts': ['us-chart']},
             'kup-we': {'out': True, 'charts': ['kp-chart', 'kp-we-box']}, 'supply': {'out': True, 'charts': ['su-chart']}, 'hill-tab': {'out': True, 'charts': ['hl-chart']},
             'lb-we': {'out': True, 'charts': ['lb-chart']}, 'garch-tab': {'out': True, 'charts': ['gt-chart']}, 'sqrt-we': {'out': True, 'charts': ['sq-chart']},
             'mdd-we': {'out': True, 'charts': ['md-chart']}})

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Un fond raportează pentru Bitcoin o volatilitate de 55% pe an, calculată dintr-o abatere standard zilnică de 3,5%. Este corect?',
         'A fund reports a Bitcoin volatility of 55% a year from a daily standard deviation of 3.5%. Is it right?'),
      key=0, opts=[
  (('Nu: s-au folosit 252 de zile; cu 365 volatilitatea este circa 66,9%.', 'No: 252 days were used; with 365 the volatility is about 66.9%.'),
   ('3,5 × √252 ≈ 55,6%, dar Bitcoin se tranzacționează 365 de zile pe an: 3,5 × √365 ≈ 66,9%.', '3.5 × √252 ≈ 55.6%, but Bitcoin trades 365 days a year: 3.5 × √365 ≈ 66.9%.')),
  (('Da: 252 de zile de tranzacționare este convenția pieței.', 'Yes: 252 trading days is the market convention.'),
   ('Convenția vine de la acțiuni; o serie cripto are A = 365 de randamente pe an.', 'The convention comes from equities; a crypto series has A = 365 returns a year.')),
  (('Nu: trebuie 3,5 × 365 ≈ 1 278%.', 'No: it should be 3.5 × 365 ≈ 1,278%.'),
   ('Volatilitatea crește cu √A, nu cu A; doar media se înmulțește cu A.', 'The volatility scales with √A, not with A; only the mean is multiplied by A.'))]),
 dict(q=('USDT închide la 1,0040 USD. Cît este abaterea de la paritate?', 'USDT closes at 1.0040 USD. What is the peg deviation?'),
      key=0, opts=[
  (('+40 bp.', '+40 bp.'), ('10 000 × (1,0040 − 1) = 40 bp peste paritate: cererea de USDT depășește ce a adus arbitrajul.', '10,000 × (1.0040 − 1) = 40 bp above the peg: demand for USDT exceeds what arbitrage has supplied.')),
  (('+0,40 bp.', '+0.40 bp.'), ('0,40 este abaterea în procente; un punct de bază este 0,01%.', '0.40 is the deviation in percent; a basis point is 0.01%.')),
  (('−40 bp.', '−40 bp.'), ('Prețul este peste 1 USD, deci abaterea este pozitivă.', 'The price is above 1 USD, so the deviation is positive.'))]),
 dict(q=('De ce s-a pierdut UST, iar USDC a revenit la paritate?', 'Why was UST lost while USDC returned to the peg?'),
      key=0, opts=[
  (('USDC avea rezerve în dolari, iar depozitele lor bancare au fost garantate; UST era susținut doar de LUNA.', 'USDC had dollar reserves whose bank deposits were guaranteed; UST was backed only by LUNA.'),
   ('Valoarea LUNA a scăzut odată cu UST, deci garanția a dispărut exact cînd era nevoie de ea; USDC a revenit după garantarea depozitelor SVB.', 'LUNA lost value together with UST, so the backing vanished exactly when it was needed; USDC returned after the SVB deposits were guaranteed.')),
  (('Codul USDC era mai sigur decît codul UST.', 'The code of USDC was safer than the code of UST.'),
   ('Panica asupra USDC a privit rezervele, nu codul; nici UST nu a căzut din cauza unei erori de program.', 'The run on USDC was on the reserves, not on the code; UST did not fall because of a software bug either.')),
  (('UST era prea mic pentru a avea arbitrajori.', 'UST was too small to attract arbitrageurs.'),
   ('Oferta UST a atins 18,8 miliarde pe 7 mai 2022; problema era lipsa rezervelor, nu dimensiunea.', 'The UST supply peaked at 18.8 billion on 7 May 2022; the problem was the lack of reserves, not the size.'))]),
 dict(q=('Un model GARCH(1,1) are α̂ + β̂ = 0,9931. Cît este timpul de înjumătățire al unui șoc al volatilității?', 'A GARCH(1,1) model has α̂ + β̂ = 0.9931. What is the half-life of a volatility shock?'),
      key=0, opts=[
  (('Circa 100 de zile.', 'About 100 days.'), ('ln 0,5/ln 0,9931 = −0,6931/−0,00692 ≈ 100 de zile, valoarea S&P 500 de pe slide.', 'ln 0.5/ln 0.9931 = −0.6931/−0.00692 ≈ 100 days, the S&P 500 value of the slide.')),
  (('Circa 145 de zile = 1/(1 − 0,9931).', 'About 145 days = 1/(1 − 0.9931).'), ('1/(1 − α − β) nu este timpul de înjumătățire; formula este ln 0,5/ln(α + β).', '1/(1 − α − β) is not the half-life; the formula is ln 0.5/ln(α + β).')),
  (('Circa 72 de zile = 0,5/0,0069.', 'About 72 days = 0.5/0.0069.'), ('Numărătorul este ln 0,5 = −0,693, nu 0,5.', 'The numerator is ln 0.5 = −0.693, not 0.5.'))]),
 dict(q=('Un activ a pierdut 75% de la ultimul record. Ce cîștig îi trebuie ca să revină la record?', 'An asset has lost 75% from its last record. What gain does it need to get back to the record?'),
      key=0, opts=[
  (('300%.', '300%.'), ('d/(1 − d) = 0,75/0,25 = 3, adică 300%.', 'd/(1 − d) = 0.75/0.25 = 3, that is 300%.')),
  (('75%.', '75%.'), ('Un cîștig de 75% aduce prețul de la 0,25 la 0,4375 din record, nu la record.', 'A gain of 75% takes the price from 0.25 to 0.4375 of the record, not to the record.')),
  (('33% = 0,25/0,75.', '33% = 0.25/0.75.'), ('Raportul este inversat: cîștigul necesar este d/(1 − d), nu (1 − d)/d.', 'The ratio is upside down: the gain needed is d/(1 − d), not (1 − d)/d.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch14-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH14_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH14_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH14_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
