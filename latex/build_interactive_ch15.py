"""Build the interactive pages of Chapter 15: interactive/ch15.html (EN) and interactive/ch15_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch15.js, inputs in interactive/ch15_data.js (Quantlets/Ch_15/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch15.js moves each one to its
own frame. The EN deck has one acronym frame less than the RO deck; XBLOCKS give the RO frame number, checked against TITLES and
TITLES_EN.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 15
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_15'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('indices', 'sfm_ch15_bank_indices', 'crises'), ('episodes', 'sfm_ch15_episodes', 'crises'), ('rcorr', 'sfm_ch15_rolling_corr', 'dep'),
          ('tscatter', 'sfm_ch15_tail_scatter', 'dep'), ('covarqr', 'sfm_ch15_covar_qr', 'covar'), ('dcovar', 'sfm_ch15_dcovar', 'covar'),
          ('vardcovar', 'sfm_ch15_var_vs_dcovar', 'covar'), ('dyn', 'sfm_ch15_dcovar_dynamic', 'covar'), ('mes', 'sfm_ch15_mes', 'mes'),
          ('mescrisis', 'sfm_ch15_mes_crisis', 'mes'), ('srisk', 'sfm_ch15_srisk', 'mes'), ('dgc', 'sfm_ch15_dgc', 'net'), ('gnet', 'sfm_ch15_granger_net', 'net'),
          ('dytab', 'sfm_ch15_dy_table', 'net'), ('dyroll', 'sfm_ch15_dy_rolling', 'net')]
# table and worked-example blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page)
XBLOCKS = [('corr-tab', 25, 'dep', 'rcorr'), ('fr-we', 27, 'dep', 'rcorr'), ('tail-tab', 30, 'dep', 'tscatter'), ('covar-we', 35, 'covar', 'covarqr'),
           ('dcovar-tab', 38, 'covar', 'dcovar'), ('tlv-we', 39, 'covar', 'dcovar'), ('mes-we', 45, 'mes', 'mes'), ('mes-tab', 47, 'mes', 'mes'),
           ('lrmes-we', 49, 'mes', 'srisk'), ('srisk-we', 50, 'mes', 'srisk'), ('dy-we', 57, 'net', 'dytab'), ('dy-tab', 59, 'net', 'dytab'),
           ('basel-we', 63, 'macro', 'dyroll')]
ORDER = ['indices', 'episodes', 'rcorr', 'corr-tab', 'fr-we', 'tail-tab', 'tscatter', 'covar-we', 'covarqr', 'dcovar', 'dcovar-tab', 'tlv-we', 'vardcovar', 'dyn',
         'mes-we', 'mes', 'mes-tab', 'mescrisis', 'lrmes-we', 'srisk-we', 'srisk', 'dgc', 'gnet', 'dy-we', 'dytab', 'dy-tab', 'dyroll', 'basel-we']
TITLES = {'corr-tab': 'Interpretarea corelațiilor: cînd diversificarea nu mai funcționează', 'fr-we': 'Contagiune sau interdependență? Forbes și Rigobon (2/2)',
          'tail-tab': 'Dependența în cozi dintre bănci și piețe', 'covar-we': 'Exemplu rezolvat: JPMorgan Chase și S&P 500', 'dcovar-tab': 'Interpretarea ΔCoVaR',
          'tlv-we': 'Exemplu rezolvat: Banca Transilvania și BET', 'mes-we': 'Exemplu rezolvat: MES din zece zile', 'mes-tab': 'Interpretarea MES și a ΔCoVaR',
          'lrmes-we': 'De la MES la LRMES', 'srisk-we': 'SRISK: deficitul de capital într-o criză', 'dy-we': 'Conectivitatea Diebold–Yilmaz',
          'dy-tab': 'Interpretarea tabelului de conectivitate', 'basel-we': 'Amortizoarele de capital din Basel III'}
TITLES_EN = {'corr-tab': 'Interpretation: When Diversification Fails', 'fr-we': 'Contagion or Interdependence? Forbes and Rigobon (2/2)',
             'tail-tab': 'Tail Dependence of Banks and Markets', 'covar-we': 'Worked Example: JPMorgan Chase and the S&P 500', 'dcovar-tab': 'Interpretation of ΔCoVaR',
             'tlv-we': 'Worked Example: Banca Transilvania and the BET', 'mes-we': 'Worked Example: MES from Ten Days', 'mes-tab': 'Interpretation: MES and ΔCoVaR',
             'lrmes-we': 'From MES to LRMES', 'srisk-we': 'SRISK: the Capital Shortfall in a Crisis', 'dy-we': 'Diebold–Yilmaz Connectedness',
             'dy-tab': 'Interpretation of the Connectedness Table', 'basel-we': 'Basel III Capital Buffers'}

PFX = {'indices': 'bi', 'episodes': 'ep', 'rcorr': 'rc', 'corr-tab': 'ct', 'fr-we': 'fr', 'tail-tab': 'tt', 'tscatter': 'ts', 'covar-we': 'cw', 'covarqr': 'cq',
       'dcovar': 'dc', 'dcovar-tab': 'dx', 'tlv-we': 'tb', 'vardcovar': 'vd', 'dyn': 'dn', 'mes-we': 'mw', 'mes': 'ms', 'mes-tab': 'mt', 'mescrisis': 'mc',
       'lrmes-we': 'lr', 'srisk-we': 'sw', 'srisk': 'sr', 'dgc': 'dg', 'gnet': 'gn', 'dy-we': 'dw', 'dytab': 'dt', 'dy-tab': 'dz', 'dyroll': 'dr', 'basel-we': 'bs'}

SECTIONS = {
 'ro': dict(crises='Patru crize în date', dep='Corelația și dependența în cozi în crize', covar='CoVaR și Delta-CoVaR', mes='MES și SRISK', net='Rețele de bănci',
            macro='Politica macroprudențială'),
 'en': dict(crises='Four Crises in the Data', dep='Correlation and Tail Dependence in Crises', covar='CoVaR and Delta-CoVaR', mes='MES and SRISK', net='Networks of Banks',
            macro='Macroprudential Policy'),
}

BCBS11 = '<a href="https://www.bis.org/publ/bcbs189.htm" target="_blank" rel="noopener">BCBS (2011)</a>'
BCBS18 = '<a href="https://www.bis.org/bcbs/publ/d445.htm" target="_blank" rel="noopener">BCBS (2018)</a>'

L = {
 'ro': dict(file='ch15_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 15 · Riscul sistemic', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: VaR (value at risk), CoVaR (VaR-ul sistemului condiționat de o instituție), ΔCoVaR '
        '(contribuția instituției la CoVaR), ES (expected shortfall), MES (marginal expected shortfall), LRMES (long-run MES, MES pe termen lung), SRISK (deficitul '
        'de capital într-o criză), OLS (metoda celor mai mici pătrate), DGC (degree of Granger causality, gradul de cauzalitate Granger), VAR (model vector '
        'autoregresiv), FROM, TO, NET (proporțiile primite de la celelalte bănci, transmise celorlalte și diferența lor), CET1 (common equity tier 1, capital comun de '
        'nivel 1), G-SIB (global systemically important bank, bancă de importanță sistemică globală), BCBS (Basel Committee on Banking Supervision, Comitetul de la '
        'Basel pentru Supraveghere Bancară), BET (indicele Bursei de Valori București), VIX (indicele volatilității implicite a S&P 500), JPM, BAC, C, GS, MS, WFC '
        '(JPMorgan Chase, Bank of America, Citigroup, Goldman Sachs, Morgan Stanley, Wells Fargo), DBK, BNP, SAN, INGA, HSBA (Deutsche Bank, BNP Paribas, Santander, '
        'ING, HSBC), TLV (Banca Transilvania), BRD (BRD – Groupe Société Générale).',
   src='Datele cursului din data/market (EODHD, ianuarie 2005 – 18 septembrie 2026): prețurile de închidere ajustate zilnice ale celor 13 bănci și închiderile '
       'S&P 500, Euro Stoxx 50 și BET; băncile românești din 2010. Coeficienții regresiilor cuantilice, intervalele bootstrap pe blocuri ale ΔCoVaR 1% și ale MES 5%, '
       'parametrii copulei t, ΔCoVaR variabil în timp și conectivitatea totală pe ferestre mobile vin din codul capitolului (Quantlets/Ch_15). Browserul recalculează '
       'din prețuri randamentele, portofoliile de bănci, episoadele, corelațiile, testul Forbes–Rigobon, dependența empirică în cozi, cuantilele, CoVaR și ΔCoVaR, MES, '
       'LRMES, SRISK, rețelele Granger și tabelul Diebold–Yilmaz. Alte niveluri α, ferestre, praguri, perioade și orizonturi sînt rulări suplimentare ale aceluiași '
       'cod. Exemplele cu MES din zece zile, LRMES, SRISK și amortizoarele Basel III folosesc cifrele de pe slide-uri; cerințele de capital urmează ' + BCBS11 + ' și ' + BCBS18 + '.'),
 'en': dict(file='ch15.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 15 · Systemic Risk', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: VaR (value at risk), CoVaR (the VaR of the system conditional on an institution), '
        'ΔCoVaR (the institution’s contribution to CoVaR), ES (expected shortfall), MES (marginal expected shortfall), LRMES (long-run MES), SRISK (the capital '
        'shortfall in a crisis), OLS (ordinary least squares), DGC (degree of Granger causality), VAR (vector autoregression), FROM, TO, NET (the shares received from '
        'the other banks, passed to them, and their difference), CET1 (common equity tier 1), G-SIB (global systemically important bank), BCBS (Basel Committee on '
        'Banking Supervision), BET (the index of the Bucharest Stock Exchange), VIX (the implied volatility index of the S&P 500), JPM, BAC, C, GS, MS, WFC (JPMorgan '
        'Chase, Bank of America, Citigroup, Goldman Sachs, Morgan Stanley, Wells Fargo), DBK, BNP, SAN, INGA, HSBA (Deutsche Bank, BNP Paribas, Santander, ING, HSBC), '
        'TLV (Banca Transilvania), BRD (BRD – Groupe Société Générale).',
   src='Course data in data/market (EODHD, January 2005 – 18 September 2026): daily adjusted closes of the 13 banks and the closes of the S&P 500, Euro Stoxx 50 and BET; '
       'the Romanian banks since 2010. The quantile-regression coefficients, the block-bootstrap intervals of ΔCoVaR 1% and MES 5%, the t-copula parameters, the '
       'time-varying ΔCoVaR and the rolling total connectedness come from the code of the chapter (Quantlets/Ch_15). From the prices the browser recomputes the returns, '
       'the bank portfolios, the episodes, the correlations, the Forbes–Rigobon test, the empirical tail dependence, the quantiles, CoVaR and ΔCoVaR, MES, LRMES, SRISK, '
       'the Granger networks and the Diebold–Yilmaz table. Other levels α, windows, thresholds, periods and horizons are extra runs of the same code. The examples of MES '
       'from ten days, LRMES, SRISK and the Basel III buffers use the numbers of the slides; the capital requirements follow ' + BCBS11 + ' and ' + BCBS18 + '.'),
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


BANK_NAMES = [('JPM', 'JPMorgan Chase'), ('BAC', 'Bank of America'), ('C', 'Citigroup'), ('GS', 'Goldman Sachs'), ('MS', 'Morgan Stanley'), ('WFC', 'Wells Fargo'),
              ('DBK', 'Deutsche Bank'), ('BNP', 'BNP Paribas'), ('SAN', 'Santander'), ('INGA', 'ING'), ('HSBA', 'HSBC'), ('TLV', 'Banca Transilvania'), ('BRD', 'BRD')]


def widgets_all(lang):
    ro = lang == 'ro'
    lec = ' (curs)' if ro else ' (lecture)'
    sl = ' (slide)'
    n = (lambda s: s.replace('.', ',')) if ro else (lambda s: s)
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    T = lambda r, e: r if ro else e
    AL = lambda i, mark=lec: select(i, T('Nivelul α', 'Level α'), [('0.01', '1%' + mark), ('0.025', n('2.5%')), ('0.05', '5%'), ('0.1', '10%')], '0.01')
    BANKS = lambda i, d, mark=sl, keys=None: select(i, T('Banca', 'Bank'), [(k, nm + (mark if k == d else '')) for k, nm in BANK_NAMES if keys is None or k in keys], d)
    MESA = lambda i: select(i, T('Nivelul α al MES', 'MES level α'), [('0.05', '5%' + lec), ('0.01', '1%'), ('0.025', n('2.5%')), ('0.1', '10%')], '0.05')
    YRS = lambda a, b: [(str(y), str(y)) for y in range(a, b + 1)]
    INTL = [(k, nm) for k, nm in BANK_NAMES[:11]]
    return {
     'indices': (select('bi-y', T('Anul investiției celor 100 de unități', 'Year the 100 units are invested'), [('2005', '2005' + lec)] + [(y, y) for y in ('2008', '2010', '2013', '2016', '2020')], '2005')
                 + play('bi'), CSS + charts('bi-chart')),
     'episodes': (slider('ep-d', T('Începutul ferestrelor mutat cu (zile)', 'Start of the windows moved by (days)'), -30, 30, 5, 0) + play('ep'), charts('ep-chart')),
     'rcorr': (slider('rc-w', T('Fereastra (zile)', 'Window (days)'), 40, 250, 10, 120) + play('rc'), charts('rc-chart')),
     'corr-tab': (select('ct-y', T('Anul din tabel', 'Year in the table'), YRS(2005, 2026), '2017')
                  + select('ct-w', T('Fereastra corelației', 'Correlation window'), [('120', T('120 de zile', '120 days') + lec), ('60', T('60 de zile', '60 days')), ('250', T('250 de zile', '250 days'))], '120') + play('ct'),
                  charts('ct-chart') + xh(T('Media corelației medii dintre perechile de bănci, pe perioade (cifrele de pe slide-ul precedent)', 'Mean of the average pairwise correlation by period (the numbers of the previous slide)')) + tw('ct-tab')),
     'fr-we': (select('fr-e', T('Perioada de criză', 'Crisis period'), [('lehman', T('15 septembrie – 31 decembrie 2008', '15 September – 31 December 2008') + sl), ('euro', T('zona euro, iulie 2011 – septembrie 2012', 'euro area, July 2011 – September 2012')),
                                                                         ('covid', T('martie 2020 (14 februarie – 30 aprilie)', 'March 2020 (14 February – 30 April)')), ('svb', T('martie 2023 (28 februarie – 28 aprilie)', 'March 2023 (28 February – 28 April)'))], 'lehman') + play('fr'),
               charts('fr-chart') + we('fr-st', f'{WE}: {T("corecția Forbes–Rigobon", "the Forbes–Rigobon correction")}',
                                       T(r'\rho^* = \frac{\rho}{\sqrt{1 + \delta(1 - \rho^2)}}, \qquad \delta = \frac{\sigma^2_{\text{criză}}}{\sigma^2_{\text{calm}}} - 1',
                                         r'\rho^* = \frac{\rho}{\sqrt{1 + \delta(1 - \rho^2)}}, \qquad \delta = \frac{\sigma^2_{\text{crisis}}}{\sigma^2_{\text{calm}}} - 1'),
                                       select('fr-s', T('Piața-sursă', 'Source market'), [('US', T('băncile americane', 'US banks') + sl), ('EU', T('băncile europene', 'European banks'))], 'US'))),
     'tail-tab': (slider('tt-q', T('Nivelul q al coloanei suplimentare (%)', 'Level q of the extra column (%)'), 0.5, 20, 0.5, 5) + play('tt'),
                  charts('tt-chart') + xh(T('Dependența în cozi (tabelul de pe slide și λ(q) la nivelul ales)', 'Tail dependence (the table of the slide and λ(q) at the chosen level)')) + tw('tt-tab')),
     'tscatter': (slider('ts-q', T('Pragul q (%)', 'Threshold q (%)'), 1, 15, 0.5, 5) + play('ts'), charts('ts-chart')),
     'covar-we': (AL('cw-a', sl) + play('cw'),
                  charts('cw-chart') + we('cw-we', f'{WE}: {T("CoVaR din regresia cuantilică", "CoVaR from the quantile regression")}',
                                          r'\mathrm{CoVaR}_\alpha = -\big(\hat a + \hat b\,\hat q_\alpha(X^i)\big), \qquad \Delta\mathrm{CoVaR}_\alpha = \hat b\,\big(\hat q_{50\%}(X^i) - \hat q_\alpha(X^i)\big)',
                                          BANKS('cw-k', 'JPM'))),
     'covarqr': (AL('cq-a') + BANKS('cq-k', 'JPM', lec) + play('cq'), charts('cq-chart')),
     'dcovar': (AL('dc-a') + play('dc'), charts('dc-chart')),
     'dcovar-tab': (select('dx-a', T('Al doilea nivel α₂', 'Second level α₂'), [('0.05', '5%' + sl), ('0.025', n('2.5%')), ('0.1', '10%')], '0.05') + play('dx'),
                    charts('dx-chart') + xh(T('CoVaR și ΔCoVaR 1% pentru cele 13 bănci (cifrele din graficul precedent) și ΔCoVaR la α₂', 'CoVaR and ΔCoVaR 1% of the 13 banks (the numbers of the previous chart) and ΔCoVaR at α₂')) + tw('dx-tab')),
     'tlv-we': (AL('tb-a', sl) + play('tb'),
                charts('tb-chart') + we('tb-we', f'{WE}: {T("ΔCoVaR între mediana și starea de criză", "ΔCoVaR between the median and the stress state")}',
                                        r'\Delta\mathrm{CoVaR}_\alpha = \mathrm{CoVaR}^{sys|X^i = q_\alpha}_\alpha - \mathrm{CoVaR}^{sys|X^i = q_{50\%}}_\alpha = \hat b\,\big(\hat q_{50\%} - \hat q_\alpha\big)',
                                        BANKS('tb-k', 'TLV'))),
     'vardcovar': (AL('vd-a') + play('vd'), charts('vd-chart')),
     'dyn': (select('dn-a', T('Nivelul α', 'Level α'), [('0.01', '1%' + lec), ('0.025', n('2.5%')), ('0.05', '5%')], '0.01')
             + select('dn-k', T('Băncile', 'Banks'), [('all', T('toate trei', 'all three') + lec), ('JPM', 'JPMorgan Chase'), ('DBK', 'Deutsche Bank'), ('TLV', 'Banca Transilvania')], 'all') + play('dn'),
             charts('dn-chart')),
     'mes-we': (slider('mw-a', T('Nivelul α (%)', 'Level α (%)'), 10, 50, 10, 20) + play('mw'),
                charts('mw-chart') + we('mw-we', f'{WE}: {T("MES ca medie a randamentelor băncii în zilele din coada pieței", "MES as the average return of the bank on the market’s tail days")}',
                                        T(r'\mathrm{MES}_\alpha = -\frac{1}{k}\sum_{t \in \text{cele mai proaste } k \text{ zile ale pieței}} X^i_t, \qquad k = \lceil n\alpha \rceil',
                                          r'\mathrm{MES}_\alpha = -\frac{1}{k}\sum_{t \in \text{worst } k \text{ market days}} X^i_t, \qquad k = \lceil n\alpha \rceil'))),
     'mes': (MESA('ms-a') + play('ms'), charts('ms-chart')),
     'mes-tab': (MESA('mt-a') + play('mt'),
                 charts('mt-chart') + xh(T('MES, beta OLS, ES al pieței, LRMES și efectul de levier de echilibru (cifrele de pe slide-uri)', 'MES, OLS beta, ES of the market, LRMES and the break-even leverage (the numbers of the slides)')) + tw('mt-tab')),
     'mescrisis': (MESA('mc-a') + play('mc'), charts('mc-chart')),
     'lrmes-we': (slider('lr-m', T('MES (% pe zi)', 'MES (% a day)'), 0.5, 6, 0.25, 3) + play('lr'),
                  charts('lr-chart') + we('lr-st', f'{WE}: {T("de la MES la LRMES", "from MES to LRMES")}', r'\mathrm{LRMES} \approx 1 - \exp(-18 \times \mathrm{MES})',
                                          select('lr-c', T('Pragul pieței pentru MES-ul băncilor', 'Market threshold for the banks’ MES'), [('-2', '−2%' + sl), ('-1', '−1%'), ('-3', '−3%')], '-2'))),
     'srisk-we': (slider('sw-W', T('Valoarea de piață a capitalului propriu W (miliarde)', 'Market value of equity W (billion)'), 20, 200, 10, 100) + play('sw'),
                  charts('sw-chart') + we('sw-st', f'{WE}: SRISK', r'\mathrm{SRISK} = kD - (1 - k)(1 - \mathrm{LRMES})\,W',
                                          select('sw-k', T('Fracția de capital k', 'Capital fraction k'), [('0.08', '8%' + sl), ('0.055', n('5.5%'))], '0.08')
                                          + select('sw-L', 'LRMES', [('ex', T('41,7% (exemplul de pe slide)', '41.7% (the example of the slide)'))] + [(k, f'{nm}') for k, nm in BANK_NAMES], 'ex'))),
     'srisk': (slider('sr-k', T('Fracția de capital k (%)', 'Capital fraction k (%)'), 3, 12, 0.5, 8) + play('sr'), charts('sr-chart')),
     'dgc': (slider('dg-w', T('Fereastra (luni)', 'Window (months)'), 24, 60, 6, 36)
             + select('dg-l', T('Nivelul testului', 'Level of the test'), [('0.05', '5%' + lec), ('0.01', '1%'), ('0.1', '10%')], '0.05') + play('dg'), charts('dg-chart')),
     'gnet': (slider('gn-s', T('Ambele ferestre mutate cu (luni)', 'Both windows moved by (months)'), -12, 24, 6, 0) + play('gn'), charts('gn-chart')),
     'dy-we': (slider('dw-h', T('Orizontul H (zile)', 'Horizon H (days)'), 1, 30, 1, 10) + play('dw'),
               charts('dw-chart') + we('dw-we', f'{WE}: {T("o proporție din descompunerea varianței și conectivitatea totală", "a share of the variance decomposition and the total connectedness")}',
                                       r'\mathrm{FROM}_i = \sum_{j \ne i} d_{ij} = 100 - d_{ii}, \quad \mathrm{TO}_i = \sum_{j \ne i} d_{ji}, \quad \text{total} = \frac{1}{N}\sum_{i \ne j} d_{ij}',
                                       select('dw-k', T('Banca', 'Bank'), [(k, nm + (sl if k == 'JPM' else '')) for k, nm in INTL], 'JPM'))),
     'dytab': (slider('dt-h', T('Orizontul H (zile)', 'Horizon H (days)'), 1, 30, 1, 10) + play('dt'), charts('dt-chart')),
     'dy-tab': (select('dz-p', T('Perioada', 'Period'), [('full', T('2005–2026', '2005–2026') + sl), ('p1', '2005–2009'), ('p2', '2010–2019'), ('p3', '2020–2026')], 'full') + play('dz'),
                charts('dz-chart') + xh(T('Partea proprie, FROM, TO și NET (în %, orizontul de 10 zile)', 'Own share, FROM, TO and NET (in %, horizon of 10 days)')) + tw('dz-tab')),
     'dyroll': (select('dr-w', T('Fereastra (zile)', 'Window (days)'), [('200', '200' + lec), ('100', '100'), ('150', '150'), ('250', '250'), ('300', '300')], '200') + play('dr'), charts('dr-chart')),
     'basel-we': (slider('bs-c', T('Amortizorul anticiclic (%)', 'Countercyclical buffer (%)'), 0, 2.5, 0.25, 0) + play('bs'),
                  charts('bs-chart') + we('bs-st', f'{WE}: {T("cerința de capital CET1 a unei bănci", "the CET1 requirement of a bank")}',
                                          T(r'\text{CET1 cerut} = 4{,}5\% + 2{,}5\% + \text{anticiclic} + \text{supliment G-SIB}', r'\text{CET1 required} = 4.5\% + 2.5\% + \text{countercyclical} + \text{G-SIB surcharge}'),
                                          select('bs-g', T('Suplimentul G-SIB', 'G-SIB surcharge'), [('1', T('1% (minimul)', '1% (the minimum)') + sl), ('3.5', T('3,5% (maximul)', '3.5% (the maximum)')),
                                                                                                  ('0', T('0 (bancă fără supliment)', '0 (bank without a surcharge)'))], '1'))),
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch15.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch15_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch15_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s %]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const X = ' (rulare suplimentară a codului din curs)';
const XS = ' (alte valori decît pe slide)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const REG = { US: 'băncile americane', EU: 'băncile europene', RO: 'băncile românești' };
const month = s => `${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`;
const list = a => a.length > 1 ? a.slice(0, -1).join(', ') + ' și ' + a[a.length - 1] : a[0];
window.CH15_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 reg: { US: 'Bănci americane', EU: 'Bănci europene', RO: 'Bănci românești' }, src: { US: 'băncile americane', EU: 'băncile europene' },
 ep_: { lehman: '15 septembrie – 31 decembrie 2008', euro: 'zona euro, 2011–2012', covid: 'martie 2020', svb: 'martie 2023' },
 pair: { US: 'bănci americane, S&P 500', EU: 'bănci europene, Euro Stoxx 50', RO: 'bănci românești, BET', USEU: 'bănci europene, bănci americane' },
 h: { ct: y => ['Grupul', 'iul. 2005 – iun. 2007', 'oct. 2008 – mar. 2009', 'apr. – iun. 2020', String(y), 'maximul (data)', 'ultima valoare'],
      tt: q => ['Perechea', 'n', 'corel.', 'λ(5%)', 'λ(1%)', 'ν̂', 'λ_L (t)', `λ(${q}%)`],
      dx: a => ['Banca', 'n', 'â', 'b̂', 'q̂ 1%', 'q̂ 50%', 'CoVaR 1%', 'CoVaR la mediană', 'ΔCoVaR 1%', 'interval 95%', `ΔCoVaR ${a}%`],
      mt: a => ['Banca', 'n', `MES ${a}%`, 'interval 95% (MES 5%)', 'beta OLS', `ES ${a}% al pieței`, 'beta × ES', 'MES la −2%', 'zile', 'LRMES (%)', 'L*'],
      dz: ['Banca', 'partea proprie', 'FROM', 'TO', 'NET'] },
 bi: (y, r, x) => `100 de unități investite la începutul lui ${y}${x ? X : ''}: ` + ['US', 'EU', 'RO'].map((k, i) => `${REG[k]} ajung la ${r[i][0]} (drawdown maxim ${r[i][1]}%, pe ${r[i][2]}${i === 2 && y < '2010' ? '; date din ' + r[i][3] : ''})`).join('; ') + '.',
 ep: (sh, rows) => `${sh !== '0' ? `Începutul fiecărei ferestre mutat cu ${sh} ${de(sh)}zile${X}. ` : ''}Minimul randamentului cumulat: ` + rows.map(r => `${r[0].replace('Euro area 2011-2012', 'zona euro 2011–2012').replace('March', 'martie')} (de la ${r[1]}): americane ${r[2]}%, europene ${r[3]}%${r[4] === null ? '' : `, românești ${r[4]}%`}, S&P 500 ${r[5]}% (la final ${r[6]}%)`).join('; ') + '.',
 rc: (w, uc, um, ud, ec, em, ed, r17, r20, rm, rd, x) => `Fereastră de ${w} ${de(w)}zile${x ? X : ''}. Băncile americane: ${uc} în iulie 2005 – iunie 2007, maximul ${um} pe ${ud}. Băncile europene: ${ec}, maximul ${em} pe ${ed}. TLV și BRD: ${r17} în 2017, ${r20} în aprilie–iunie 2020, maximul ${rm} pe ${rd}.`,
 ct: (w, y, v, x) => `${y}, fereastră de ${w} ${de(w)}zile${x ? X : ''}: media corelației ${v[0]} la băncile americane, ${v[1]} la cele europene, ${v[2]} la TLV și BRD.`,
 fr1: (s, o, d0, d1, n0, e0, e1, n1) => `Sursa: portofoliul ${s.replace('băncile', 'băncilor')}; cealaltă piață: ${o}. Perioada calmă ${d0} – ${d1} (n = ${n0}), criza ${e0} – ${e1} (n = ${n1}).`,
 fr2: (rc, rk, v0, v1, d) => `ρ_calm = ${rc}, ρ_criză = ${rk}; varianța sursei crește de la ${v0} la ${v1}, deci δ = ${v1}/${v0} − 1 = ${d}.`,
 fr3: (rk, d, om, ins, adj) => `ρ* = ${rk}/√(1 + ${d} × ${om}) = ${rk}/√${ins} = ${adj}.`,
 fr4: (adj, rc, c) => (c ? `ρ* = ${adj} > ρ_calm = ${rc}: corelația crește și după corecție, ceea ce indică o contagiune.` : `ρ* = ${adj} ≤ ρ_calm = ${rc}: după corecție corelația nu mai crește; datele sînt compatibile cu interdependența.`),
 fr: (e, s, rc, rk, d, adj, c, x) => `${cap(e)}, sursa ${s}${x ? X : ''}: ρ crește de la ${rc} la ${rk}, δ = ${d}, ρ* = ${adj}: ${c ? 'contagiune' : 'interdependență'}.`,
 tt: (q, v, qd, x) => `λ(${q}%)${x ? X : ''}: ${v[0]} pentru băncile americane și S&P 500, ${v[1]} pentru cele europene și Euro Stoxx 50, ${v[2]} pentru cele românești și BET, ${v[3]} între băncile europene și cele americane; la independență, λ(q) = ${qd}.`,
 ts: (q, ub, ue, ur, rb, re, rr, nu, nr, x) => `Ambele serii în cele mai proaste ${q}% dintre zile${x ? X : ''}: SUA ${ub} ${de(ub)}zile din ${nu}, față de ${ue} la independență (q²n), de ${ur} ori mai multe; România ${rb} din ${nr}, față de ${re}, de ${rr} ori.`,
 cw1: (nm, ix, n, a, ah, bh) => `${nm} și ${ix}, randamente logaritmice zilnice (n = ${n}); regresia cuantilică la ${a}%: â = ${ah}, b̂ = ${bh}.`,
 cw2: (nm, a, q, v, qm) => `Cuantilele băncii: q̂ ${a}% = ${q}% (deci VaR ${a}% al băncii = ${v}%), mediana q̂ 50% = ${qm}%.`,
 cw3: (a, A, B, Q, mA, mp, s, cv, cm) => `CoVaR ${a}% = −(${A} + ${B} × (${Q})) = ${mA} + ${mp} = ${s}%; din estimațiile nerotunjite: ${cv}%. CoVaR la mediana băncii: ${cm}%.`,
 cw4: (a, B, M, Q, d, p, dc, vs, ix) => `ΔCoVaR ${a}% = ${B} × (${M} − (${Q})) = ${B} × ${d} = ${p}%; din estimațiile nerotunjite: ${dc}%. VaR ${a}% necondiționat al ${ix}: ${vs}%.`,
 cw: (nm, a, cv, cm, dc, vs, ix, x) => `${nm}, nivelul ${a}%${x ? X : ''}: CoVaR ${cv}% cînd banca este la cuantila ei, ${cm}% la mediana ei; ΔCoVaR ${dc}%; VaR ${a}% necondiționat al ${ix}: ${vs}%.`,
 cq: (nm, a, ah, bh, a5, b5, cv, cm, dc, x) => `${nm}, ${a}%${x ? X : ''}: dreapta cuantilică ${ah} + ${bh} x, mediana ${a5} + ${b5} x. Minus CoVaR = −${cv}% la cuantila băncii, −${cm}% la mediana ei; ΔCoVaR = distanța pe verticală = ${dc}%.`,
 dc: (a, us, eu, tlv, brd, mx, ci, x) => `ΔCoVaR ${a}%${x ? X : ''}: băncile americane ${us[0]}–${us[1]}%, cele europene ${eu[0]}–${eu[1]}%; TLV ${tlv}%, BRD ${brd}%; cel mai mare: ${mx}.${ci ? ' Intervalele de 95% vin din bootstrap pe blocuri mobile.' : ' Intervalele bootstrap sînt calculate doar la 1%.'}`,
 dx: (a, r, x) => `Corelația Spearman dintre ordinea băncilor după ΔCoVaR 1% și după ΔCoVaR ${a}%${x ? X : ''}: ${r}.`,
 tb1: (nm, ix, n, a, b) => `${nm} și ${ix}, randamente logaritmice zilnice (n = ${n}); regresia cuantilică la ${a}%: b̂ = ${b}.`,
 tb2: (a, q, v, qm) => `Starea de criză: q̂ ${a}% = ${q}% (VaR ${a}% al băncii = ${v}%); starea obișnuită: mediana q̂ 50% = ${qm}%.`,
 tb3: (a, b, qm, mq, g, dc, cv) => `ΔCoVaR ${a}% = ${b} × (${qm} + ${mq}) = ${b} × ${g} = ${dc}%; CoVaR ${a}% al indicelui = ${cv}%.`,
 tb4: (lo, hi) => `Intervalul bootstrap pe blocuri de 95% pentru ΔCoVaR 1%: [${lo}; ${hi}].`,
 tb4x: () => 'Intervalul bootstrap este calculat în curs doar la nivelul 1%.',
 tb5: (dc, ix) => `Interpretare: o zi de criză pentru bancă, față de o zi obișnuită, coboară cuantila indicelui ${ix} cu aproximativ ${dc} puncte procentuale.`,
 tb: (nm, a, dc, cv, cm, x) => `${nm}, ${a}%${x ? X : ''}: ΔCoVaR ${dc}% = CoVaR în criză ${cv}% − CoVaR la mediană ${cm}%.`,
 vd: (a, r, v3, d3, x) => `Nivelul ${a}%${x ? X : ''}: corelația Spearman dintre VaR și ΔCoVaR este ${r}. Primele trei după VaR: ${list(v3)}; ultimele trei după ΔCoVaR: ${list(d3)}.`,
 dn: (a, rows, x) => `ΔCoVaR ${a}% în timp${x ? X : ''}: ` + rows.map(r => `${r[0]}: media ${r[1]}%, maximul ${r[2]}% (${r[3]}), media din 2017 ${r[4]}%, panta b̂ = ${r[5]}`).join('; ') + '.',
 mw1: (a, na, k, L) => `n = 10 zile, α = ${a}%: k = ⌈10 × ${na}⌉ = ${k} ${de(k)}zile, cele mai proaste ale pieței: ` + L.map(r => `ziua ${r[0]} (${r[1]})`).join(', ') + '.',
 mw2: (a, b, k, m) => `MES ${a}% = −(${b.join(' + ').replace(/\+ −/g, '− ').replace(/^−/, '−')})/${k} = ${m}%.`,
 mw3: (a, mk, k, e) => `ES ${a}% al pieței = −(${mk.join(' + ').replace(/\+ −/g, '− ')})/${k} = ${e}%.`,
 mw4: r => `MES/ES = ${r}: în zilele proaste ale pieței, această bancă pierde de ${r} ori mai mult decît piața. Zilele se aleg după randamentul pieței, nu al băncii.`,
 mw: (a, k, m, e, x) => `α = ${a}%${x ? XS : ''}: ${k} ${de(k)}zile, MES ${m}%, ES al pieței ${e}%.`,
 ms: (a, n1, v1, n2, v2, n3, v3, tlv, brd, ci, x) => `MES ${a}%${x ? X : ''}: cele mai mari: ${n1} (${v1}%) și ${n2} (${v2}%); cel mai mic: ${n3} (${v3}%); TLV ${tlv}%, BRD ${brd}%.${ci ? ' Intervale bootstrap pe blocuri de 95%.' : ' Intervalele bootstrap sînt calculate doar la 5%.'}`,
 mt: (a, cb, cm, hb, hm, r, x) => `MES ${a}%${x ? X : ''}: Citigroup beta ${cb}, MES ${cm}%; HSBC beta ${hb}, MES ${hm}%. Corelația Spearman dintre MES ${a}% și ΔCoVaR 1%: ${r}.`,
 mc: (a, r, p, n, wn, wr, np, x) => `MES ${a}% din iunie 2006 – iunie 2007 (${np} ${de(np)}zile) față de randamentul din iulie 2007 – decembrie 2008${x ? X : ''}: Spearman ${r} (p = ${p}, n = ${n}); cel mai prost randament: ${wn} (${wr}%).`,
 lr1: (m, f) => `MES = ${m}% = ${f} (pierderea zilnică medie a băncii în zilele în care piața scade cu peste 2%).`,
 lr2: (f, x, e, L) => `18 × ${f} = ${x}; exp(−${x}) = ${e}; LRMES = 1 − ${e} = ${L}%.`,
 lr3: L => `Banca ar pierde aproximativ ${L}% din valoarea de piață dacă piața scade cu 40% în șase luni.`,
 lr4: (c, m0, m1, l0, l1, x) => `Cele 13 bănci, MES în zilele în care piața scade cu peste ${c.replace('−', '')}%${x ? X : ''}: între ${m0}% și ${m1}%, deci LRMES între ${l0}% și ${l1}%.`,
 lr: (m, L, x) => `MES ${m}%${x ? XS : ''}: LRMES ${L}%.`,
 sw1: (D, W, lr, k, src) => `D = ${D}, W = ${W} (miliarde), LRMES = ${lr}%${src === 'ex' ? '' : ' (LRMES-ul băncii din curs)'}, k = ${k}%.`,
 sw2: (k, D, need) => `Capitalul cerut pentru datorii: k D = ${k} × ${D} = ${need}.`,
 sw3: (k1, l1, W, left) => `Capitalul rămas după criză, net de cerința lui: (1 − k)(1 − LRMES) W = ${k1} × ${l1} × ${W} = ${left}.`,
 sw4: (need, left, s) => `SRISK = ${need} − ${left} = ${s} miliarde.`,
 sw5: (L, Ls, pos) => `Efectul de levier L = (D + W)/W = ${L}; pragul L* = 1 + (1 − k)(1 − LRMES)/k = ${Ls}: ${pos ? 'L > L*, banca are nevoie de capital într-o criză' : 'L ≤ L*, banca are un surplus de capital (SRISK negativ)'}.`,
 sw: (W, s, L, Ls, x) => `W = ${W} ${de(W)}miliarde${x ? XS : ''}: SRISK ${s} miliarde; L = ${L}, L* = ${Ls}.`,
 sr: (k, rows, mn, mx, x) => `k = ${k}%${x ? XS : ''}: SRISK > 0 peste un efect de levier de ` + rows.map(r => `${r[1]} (${r[0]})`).join(', ') + `; la cele 13 bănci, L* între ${mn} și ${mx}.`,
 dg: (w, lv, f, mx, dmx, mn, mi, dmi, last, nm, x) => `Ferestre de ${w} ${de(w)}luni, nivelul ${lv}%, ${nm} ${de(nm)}luni de date${x ? X : ''}: prima fereastră se încheie pe ${f}; DGC maxim ${mx}% (${dmx}), media ${mn}%, minimul ${mi}% (${dmi}), ultima valoare ${last}%.`,
 gn: (sh, rows, x) => `${sh !== '0' ? `Ferestre mutate cu ${sh} ${de(sh)}luni${X}. ` : ''}` + rows.map((r, i) => `${i ? month(r[0]) : cap(month(r[0]))} – ${month(r[1])}: ${r[2]} ${de(r[2])}legături, DGC ${r[3]}%`).join('; ') + '.',
 dw1: (nm, H, n, own, top) => `${nm}, H = ${H} ${de(H)}zile, ${n} ${de(n)}zile comune: partea proprie dᵢᵢ = ${own}%; cele mai mari proporții de la alte bănci: ` + top.map(t => `${t[0]} ${t[1]}%`).join(', ') + '.',
 dw2: (own, f) => `FROM = suma rîndului fără diagonală = 100 − ${own} = ${f}%.`,
 dw3: (to, f, net) => `TO = suma coloanei fără diagonală = ${to}; NET = TO − FROM = ${to} − ${f} = ${net}.`,
 dw4: (s, tot) => `Conectivitatea totală = (suma celor 11 valori FROM)/11 = ${s}/11 = ${tot}%.`,
 dw: (nm, H, f, to, net, tot, x) => `${nm}, H = ${H}${x ? X : ''}: FROM ${f}%, TO ${to}, NET ${net}; conectivitatea totală ${tot}%.`,
 dt: (H, n, tot, tn, tv, ln, lv, f0, f1, x) => `${n} ${de(n)}zile comune, VAR(1), H = ${H}${x ? X : ''}: conectivitatea totală ${tot}%; cel mai mare TO: ${tn} (${tv}); cel mai mic NET: ${ln} (${lv}); FROM între ${f0} și ${f1}.`,
 dz: (d0, d1, n, tot, f0, f1, us, eu, x) => `${d0} – ${d1}, ${n} ${de(n)}zile${x ? X : ''}: conectivitatea totală ${tot}%, FROM între ${f0} și ${f1}; NET mediu ${us} la băncile americane și ${eu} la cele europene.`,
 dr: (w, mn, dmn, mx, dmx, y8, y16, m16, d16, last, x) => `Ferestre de ${w} de zile${x ? X : ''}: minimul ${mn}% (${dmn}), maximul ${mx}% (${dmx}); octombrie 2008 – martie 2009: ${y8}%; 2016: media ${y16}%, maximul ${m16}% (${d16}); ultima valoare ${last}%.`,
 bs1: () => 'Capitalul CET1 minim: 4,5% din activele ponderate la risc; amortizorul de conservare: 2,5%; împreună 7%.',
 bs2: c => `Amortizorul anticiclic, între 0 și 2,5%: ${c}%.`,
 bs3: g => `Suplimentul G-SIB, între 1% și 3,5% (${g}% aici), după scorul mărimii, interconectării, substituibilității, complexității și activității transfrontaliere.`,
 bs4: (c, g, t) => `Cerința CET1 = 4,5 + 2,5 + ${c} + ${g} = ${t}% din activele ponderate la risc.`,
 bs5: t => `Cu un CET1 între 4,5% și ${t}%, banca respectă minimul, dar dividendele și bonusurile ei sînt restricționate.`,
 bs: (c, g, t) => `Anticiclic ${c}%, supliment G-SIB ${g}%: cerința CET1 este ${t}%.`,
};
})();
""",
 'en': r"""(function () {
const X = ' (extra run of the chapter’s code)';
const XS = ' (values other than on the slide)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const REG = { US: 'the US banks', EU: 'the European banks', RO: 'the Romanian banks' };
const month = s => `${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`;
const list = a => a.length > 1 ? a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1] : a[0];
window.CH15_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 reg: { US: 'US banks', EU: 'European banks', RO: 'Romanian banks' }, src: { US: 'the US banks', EU: 'the European banks' },
 ep_: { lehman: '15 September – 31 December 2008', euro: 'euro area, 2011–2012', covid: 'March 2020', svb: 'March 2023' },
 pair: { US: 'US banks, S&P 500', EU: 'European banks, Euro Stoxx 50', RO: 'Romanian banks, BET', USEU: 'European banks, US banks' },
 h: { ct: y => ['Group', 'Jul 2005 – Jun 2007', 'Oct 2008 – Mar 2009', 'Apr – Jun 2020', String(y), 'maximum (date)', 'last value'],
      tt: q => ['Pair', 'n', 'corr.', 'λ(5%)', 'λ(1%)', 'ν̂', 'λ_L (t)', `λ(${q}%)`],
      dx: a => ['Bank', 'n', 'â', 'b̂', 'q̂ 1%', 'q̂ 50%', 'CoVaR 1%', 'CoVaR at the median', 'ΔCoVaR 1%', '95% interval', `ΔCoVaR ${a}%`],
      mt: a => ['Bank', 'n', `MES ${a}%`, '95% interval (MES 5%)', 'OLS beta', `ES ${a}% of the market`, 'beta × ES', 'MES at −2%', 'days', 'LRMES (%)', 'L*'],
      dz: ['Bank', 'own share', 'FROM', 'TO', 'NET'] },
 bi: (y, r, x) => `100 units invested at the start of ${y}${x ? X : ''}: ` + ['US', 'EU', 'RO'].map((k, i) => `${REG[k]} reach ${r[i][0]} (maximum drawdown ${r[i][1]}%, on ${r[i][2]}${i === 2 && y < '2010' ? '; data from ' + r[i][3] : ''})`).join('; ') + '.',
 ep: (sh, rows) => `${sh !== '0' ? `Start of each window moved by ${sh} days${X}. ` : ''}Minimum of the cumulative return: ` + rows.map(r => `${r[0].replace('Euro area 2011-2012', 'euro area 2011–2012')} (from ${r[1]}): US ${r[2]}%, European ${r[3]}%${r[4] === null ? '' : `, Romanian ${r[4]}%`}, S&P 500 ${r[5]}% (at the end ${r[6]}%)`).join('; ') + '.',
 rc: (w, uc, um, ud, ec, em, ed, r17, r20, rm, rd, x) => `Window of ${w} days${x ? X : ''}. US banks: ${uc} in July 2005 – June 2007, maximum ${um} on ${ud}. European banks: ${ec}, maximum ${em} on ${ed}. TLV and BRD: ${r17} in 2017, ${r20} in April–June 2020, maximum ${rm} on ${rd}.`,
 ct: (w, y, v, x) => `${y}, window of ${w} days${x ? X : ''}: mean correlation ${v[0]} for the US banks, ${v[1]} for the European banks, ${v[2]} for TLV and BRD.`,
 fr1: (s, o, d0, d1, n0, e0, e1, n1) => `Source: the portfolio of ${s.replace('the ', '')}; the other market: ${o}. Calm period ${d0} – ${d1} (n = ${n0}), crisis ${e0} – ${e1} (n = ${n1}).`,
 fr2: (rc, rk, v0, v1, d) => `ρ_calm = ${rc}, ρ_crisis = ${rk}; the variance of the source rises from ${v0} to ${v1}, so δ = ${v1}/${v0} − 1 = ${d}.`,
 fr3: (rk, d, om, ins, adj) => `ρ* = ${rk}/√(1 + ${d} × ${om}) = ${rk}/√${ins} = ${adj}.`,
 fr4: (adj, rc, c) => (c ? `ρ* = ${adj} > ρ_calm = ${rc}: the correlation rises even after the correction, which points to contagion.` : `ρ* = ${adj} ≤ ρ_calm = ${rc}: after the correction the correlation no longer rises; the data are consistent with interdependence.`),
 fr: (e, s, rc, rk, d, adj, c, x) => `${cap(e)}, source ${s}${x ? X : ''}: ρ rises from ${rc} to ${rk}, δ = ${d}, ρ* = ${adj}: ${c ? 'contagion' : 'interdependence'}.`,
 tt: (q, v, qd, x) => `λ(${q}%)${x ? X : ''}: ${v[0]} for the US banks and the S&P 500, ${v[1]} for the European banks and the Euro Stoxx 50, ${v[2]} for the Romanian banks and the BET, ${v[3]} between the European and the US banks; under independence λ(q) = ${qd}.`,
 ts: (q, ub, ue, ur, rb, re, rr, nu, nr, x) => `Both series in their worst ${q}% of days${x ? X : ''}: US ${ub} days of ${nu}, against ${ue} under independence (q²n), ${ur} times as many; Romania ${rb} of ${nr}, against ${re}, ${rr} times.`,
 cw1: (nm, ix, n, a, ah, bh) => `${nm} and the ${ix}, daily log returns (n = ${n}); quantile regression at ${a}%: â = ${ah}, b̂ = ${bh}.`,
 cw2: (nm, a, q, v, qm) => `Quantiles of the bank: q̂ ${a}% = ${q}% (so the VaR ${a}% of the bank = ${v}%), median q̂ 50% = ${qm}%.`,
 cw3: (a, A, B, Q, mA, mp, s, cv, cm) => `CoVaR ${a}% = −(${A} + ${B} × (${Q})) = ${mA} + ${mp} = ${s}%; from the unrounded estimates: ${cv}%. CoVaR at the bank’s median: ${cm}%.`,
 cw4: (a, B, M, Q, d, p, dc, vs, ix) => `ΔCoVaR ${a}% = ${B} × (${M} − (${Q})) = ${B} × ${d} = ${p}%; from the unrounded estimates: ${dc}%. Unconditional VaR ${a}% of the ${ix}: ${vs}%.`,
 cw: (nm, a, cv, cm, dc, vs, ix, x) => `${nm}, level ${a}%${x ? X : ''}: CoVaR ${cv}% when the bank is at its quantile, ${cm}% at its median; ΔCoVaR ${dc}%; unconditional VaR ${a}% of the ${ix}: ${vs}%.`,
 cq: (nm, a, ah, bh, a5, b5, cv, cm, dc, x) => `${nm}, ${a}%${x ? X : ''}: quantile line ${ah} + ${bh} x, median line ${a5} + ${b5} x. Minus CoVaR = −${cv}% at the bank’s quantile, −${cm}% at its median; ΔCoVaR = the vertical distance = ${dc}%.`,
 dc: (a, us, eu, tlv, brd, mx, ci, x) => `ΔCoVaR ${a}%${x ? X : ''}: US banks ${us[0]}–${us[1]}%, European banks ${eu[0]}–${eu[1]}%; TLV ${tlv}%, BRD ${brd}%; the largest: ${mx}.${ci ? ' The 95% intervals come from a moving-block bootstrap.' : ' The bootstrap intervals are computed at 1% only.'}`,
 dx: (a, r, x) => `Spearman correlation between the ranking of the banks by ΔCoVaR 1% and by ΔCoVaR ${a}%${x ? X : ''}: ${r}.`,
 tb1: (nm, ix, n, a, b) => `${nm} and the ${ix}, daily log returns (n = ${n}); quantile regression at ${a}%: b̂ = ${b}.`,
 tb2: (a, q, v, qm) => `Stress state: q̂ ${a}% = ${q}% (the VaR ${a}% of the bank = ${v}%); normal state: the median q̂ 50% = ${qm}%.`,
 tb3: (a, b, qm, mq, g, dc, cv) => `ΔCoVaR ${a}% = ${b} × (${qm} + ${mq}) = ${b} × ${g} = ${dc}%; CoVaR ${a}% of the index = ${cv}%.`,
 tb4: (lo, hi) => `95% moving-block bootstrap interval of ΔCoVaR 1%: [${lo}; ${hi}].`,
 tb4x: () => 'The lecture computes the bootstrap interval at the 1% level only.',
 tb5: (dc, ix) => `Interpretation: a stress day for the bank, against a normal day, lowers the quantile of the ${ix} by about ${dc} percentage points.`,
 tb: (nm, a, dc, cv, cm, x) => `${nm}, ${a}%${x ? X : ''}: ΔCoVaR ${dc}% = CoVaR in stress ${cv}% − CoVaR at the median ${cm}%.`,
 vd: (a, r, v3, d3, x) => `Level ${a}%${x ? X : ''}: the Spearman correlation between VaR and ΔCoVaR is ${r}. Top three by VaR: ${list(v3)}; bottom three by ΔCoVaR: ${list(d3)}.`,
 dn: (a, rows, x) => `ΔCoVaR ${a}% through time${x ? X : ''}: ` + rows.map(r => `${r[0]}: mean ${r[1]}%, maximum ${r[2]}% (${r[3]}), mean in 2017 ${r[4]}%, slope b̂ = ${r[5]}`).join('; ') + '.',
 mw1: (a, na, k, L) => `n = 10 days, α = ${a}%: k = ⌈10 × ${na}⌉ = ${k} days, the market’s worst: ` + L.map(r => `day ${r[0]} (${r[1]})`).join(', ') + '.',
 mw2: (a, b, k, m) => `MES ${a}% = −(${b.join(' + ').replace(/\+ −/g, '− ')})/${k} = ${m}%.`,
 mw3: (a, mk, k, e) => `ES ${a}% of the market = −(${mk.join(' + ').replace(/\+ −/g, '− ')})/${k} = ${e}%.`,
 mw4: r => `MES/ES = ${r}: on the market’s bad days this bank loses ${r} times as much as the market. The days are chosen by the market’s return, not the bank’s.`,
 mw: (a, k, m, e, x) => `α = ${a}%${x ? XS : ''}: ${k} days, MES ${m}%, ES of the market ${e}%.`,
 ms: (a, n1, v1, n2, v2, n3, v3, tlv, brd, ci, x) => `MES ${a}%${x ? X : ''}: the largest ${n1} (${v1}%) and ${n2} (${v2}%); the smallest ${n3} (${v3}%); TLV ${tlv}%, BRD ${brd}%.${ci ? ' 95% moving-block bootstrap intervals.' : ' The bootstrap intervals are computed at 5% only.'}`,
 mt: (a, cb, cm, hb, hm, r, x) => `MES ${a}%${x ? X : ''}: Citigroup beta ${cb}, MES ${cm}%; HSBC beta ${hb}, MES ${hm}%. Spearman correlation between MES ${a}% and ΔCoVaR 1%: ${r}.`,
 mc: (a, r, p, n, wn, wr, np, x) => `MES ${a}% from June 2006 to June 2007 (${np} days) against the return from July 2007 to December 2008${x ? X : ''}: Spearman ${r} (p = ${p}, n = ${n}); the worst return: ${wn} (${wr}%).`,
 lr1: (m, f) => `MES = ${m}% = ${f} (the bank’s average daily loss on the days the market falls by more than 2%).`,
 lr2: (f, x, e, L) => `18 × ${f} = ${x}; exp(−${x}) = ${e}; LRMES = 1 − ${e} = ${L}%.`,
 lr3: L => `The bank would lose about ${L}% of its market value if the market fell by 40% in six months.`,
 lr4: (c, m0, m1, l0, l1, x) => `The 13 banks, MES on the days the market falls by more than ${c.replace('−', '')}%${x ? X : ''}: between ${m0}% and ${m1}%, so LRMES between ${l0}% and ${l1}%.`,
 lr: (m, L, x) => `MES ${m}%${x ? XS : ''}: LRMES ${L}%.`,
 sw1: (D, W, lr, k, src) => `D = ${D}, W = ${W} (billion), LRMES = ${lr}%${src === 'ex' ? '' : ' (the bank’s LRMES from the lecture)'}, k = ${k}%.`,
 sw2: (k, D, need) => `Capital required for the debt: k D = ${k} × ${D} = ${need}.`,
 sw3: (k1, l1, W, left) => `Equity left after the crisis, net of its own requirement: (1 − k)(1 − LRMES) W = ${k1} × ${l1} × ${W} = ${left}.`,
 sw4: (need, left, s) => `SRISK = ${need} − ${left} = ${s} billion.`,
 sw5: (L, Ls, pos) => `Leverage L = (D + W)/W = ${L}; threshold L* = 1 + (1 − k)(1 − LRMES)/k = ${Ls}: ${pos ? 'L > L*, the bank needs capital in a crisis' : 'L ≤ L*, the bank has a capital surplus (negative SRISK)'}.`,
 sw: (W, s, L, Ls, x) => `W = ${W} billion${x ? XS : ''}: SRISK ${s} billion; L = ${L}, L* = ${Ls}.`,
 sr: (k, rows, mn, mx, x) => `k = ${k}%${x ? XS : ''}: SRISK > 0 above a leverage of ` + rows.map(r => `${r[1]} (${r[0]})`).join(', ') + `; for the 13 banks, L* between ${mn} and ${mx}.`,
 dg: (w, lv, f, mx, dmx, mn, mi, dmi, last, nm, x) => `Windows of ${w} months, level ${lv}%, ${nm} months of data${x ? X : ''}: the first window ends on ${f}; DGC maximum ${mx}% (${dmx}), mean ${mn}%, minimum ${mi}% (${dmi}), last value ${last}%.`,
 gn: (sh, rows, x) => `${sh !== '0' ? `Windows moved by ${sh} months${X}. ` : ''}` + rows.map(r => `${month(r[0])} – ${month(r[1])}: ${r[2]} links, DGC ${r[3]}%`).join('; ') + '.',
 dw1: (nm, H, n, own, top) => `${nm}, H = ${H} days, ${n} common days: own share dᵢᵢ = ${own}%; the largest shares from other banks: ` + top.map(t => `${t[0]} ${t[1]}%`).join(', ') + '.',
 dw2: (own, f) => `FROM = the row sum without the diagonal = 100 − ${own} = ${f}%.`,
 dw3: (to, f, net) => `TO = the column sum without the diagonal = ${to}; NET = TO − FROM = ${to} − ${f} = ${net}.`,
 dw4: (s, tot) => `Total connectedness = (sum of the 11 FROM values)/11 = ${s}/11 = ${tot}%.`,
 dw: (nm, H, f, to, net, tot, x) => `${nm}, H = ${H}${x ? X : ''}: FROM ${f}%, TO ${to}, NET ${net}; total connectedness ${tot}%.`,
 dt: (H, n, tot, tn, tv, ln, lv, f0, f1, x) => `${n} common days, VAR(1), H = ${H}${x ? X : ''}: total connectedness ${tot}%; largest TO: ${tn} (${tv}); lowest NET: ${ln} (${lv}); FROM between ${f0} and ${f1}.`,
 dz: (d0, d1, n, tot, f0, f1, us, eu, x) => `${d0} – ${d1}, ${n} days${x ? X : ''}: total connectedness ${tot}%, FROM between ${f0} and ${f1}; mean NET ${us} for the US banks and ${eu} for the European banks.`,
 dr: (w, mn, dmn, mx, dmx, y8, y16, m16, d16, last, x) => `Windows of ${w} days${x ? X : ''}: minimum ${mn}% (${dmn}), maximum ${mx}% (${dmx}); October 2008 – March 2009: ${y8}%; 2016: mean ${y16}%, maximum ${m16}% (${d16}); last value ${last}%.`,
 bs1: () => 'Minimum CET1 capital: 4.5% of risk-weighted assets; conservation buffer: 2.5%; together 7%.',
 bs2: c => `Countercyclical buffer, between 0 and 2.5%: ${c}%.`,
 bs3: g => `G-SIB surcharge, between 1% and 3.5% (${g}% here), set by a score of size, interconnectedness, substitutability, complexity and cross-border activity.`,
 bs4: (c, g, t) => `CET1 requirement = 4.5 + 2.5 + ${c} + ${g} = ${t}% of risk-weighted assets.`,
 bs5: t => `With CET1 between 4.5% and ${t}%, the bank meets the minimum, but its dividends and bonuses are restricted.`,
 bs: (c, g, t) => `Countercyclical ${c}%, G-SIB surcharge ${g}%: the CET1 requirement is ${t}%.`,
};
})();
""",
}

QA = {
 'ro': {
  'indices': ('▶ mută investiția celor 100 de unități la începutul lui 2010 (rulare suplimentară). Cît este drawdown-ul maxim al portofoliului băncilor americane?',
              '−51%, pe 23 noiembrie 2011, față de −82% cu investiția din 2005: minimul din martie 2009 nu mai este în eșantion. Drawdown-ul maxim depinde de începutul '
              'eșantionului; pentru băncile europene, cel mai adînc punct după 2010 este 21 aprilie 2020 (−61%).'),
  'episodes': ('▶ mută începutul fiecărei ferestre cu 30 de zile mai tîrziu (rulare suplimentară). Ce devine minimul băncilor americane în episodul din martie 2020?',
               '−10,8% în loc de −47,6%: fereastra începe pe 15 martie 2020, după cea mai mare parte a scăderii, iar la final portofoliul are +16,9%. Randamentul cumulat se '
               'măsoară de la începutul ferestrei, deci comparația dintre episoade depinde de această dată.'),
  'rcorr': ('Cu o fereastră de 60 de zile (▶, rulare suplimentară), unde ajunge maximul corelației dintre TLV și BRD?',
            'La 0,96, pe 6 februarie 2019, tot în afara unei crize (cu 120 de zile: 0,93 pe 9 mai 2019). Pentru o singură pereche de acțiuni, o fereastră scurtă dă valori '
            'extreme mai mari; maximul băncilor americane rămîne pe 16 martie 2020 (0,95).'),
  'corr-tab': ('▶ trece prin anii 2005–2026. În ce an este cea mai mare media anuală a corelației dintre TLV și BRD?',
               'În 2019: 0,80, peste 2020 (0,76) și peste 2011 (0,56). Corelația pe 120 de zile a două acțiuni este zgomotoasă și poate atinge maximul într-un an liniștit, '
               'cum arată și maximul zilnic din 9 mai 2019.'),
  'fr-we': ('▶ alege criza din martie 2020 (rulare suplimentară). Depășește corelația corectată ρ* corelația din perioada calmă, 0,40?',
            'Nu: ρ_criză = 0,74, dar varianța portofoliului american crește de 47 de ori (δ = 46,0), deci ρ* = 0,74/√(1 + 46,0 × 0,451) = 0,16. Dintre cele patru ferestre, '
            'doar martie 2023 dă ρ* = 0,42, puțin peste 0,40, pentru că volatilitatea a crescut puțin (δ = 3,7).'),
  'tail-tab': ('▶ coboară q la 0,5% (rulare suplimentară). Ce se întîmplă cu λ(q) pentru băncile americane și S&P 500?',
               'Crește la 0,73, față de 0,64 la 5%; la independență ar fi 0,005. Estimarea se sprijină pe cele mai proaste 26 de zile ale S&P 500, deci este imprecisă: pentru '
               'băncile europene λ(0,5%) scade la 0,58.'),
  'tscatter': ('La q = 1% (▶, rulare suplimentară), de cîte ori sînt mai multe zilele proaste comune ale băncilor americane și S&P 500 decît la independență?',
               'De circa 63 de ori: 33 de zile față de q²n = 0,0001 × 5 229 = 0,52, față de 12,7 ori la q = 5%. Raportul crește spre coadă: zilele foarte proaste vin împreună.'),
  'covar-we': ('▶ trece la α = 5% (rulare suplimentară). CoVaR 5% al S&P 500 condiționat de JPM este mai mare sau mai mic decît 4,79%?',
               'Mai mic: 2,45%. Cuantila JPM (−3,07%) și termenul liber (−1,26) sînt mai puțin extreme, iar panta rămîne 0,39; ΔCoVaR 5% = 1,22%, aproape jumătate din 2,47%.'),
  'covarqr': ('▶ trece prin α = 2,5%, 5% și 10% (rulare suplimentară). Cum se schimbă panta b̂ a dreptei roșii pentru JPM?',
              'Aproape deloc: 0,385, 0,390 și 0,401, față de 0,390 la 1% și 0,398 la mediană. Se mută termenul liber, de la −2,35 la −0,86: dreptele sînt aproape paralele, '
              'deci ΔCoVaR scade pentru că distanța dintre cuantilele JPM scade.'),
  'dcovar': ('La nivelul 5% (▶, rulare suplimentară), ce grup de bănci are cel mai mare ΔCoVaR?',
             'Tot băncile europene: 1,43–1,68% (Santander 1,68%), față de 1,12–1,33% la cele americane; TLV 1,51%, BRD 1,45%. La 1%, cea mai mare valoare era a TLV (3,31%).'),
  'dcovar-tab': ('Cu α₂ = 10% (▶, rulare suplimentară), cît este corelația Spearman dintre ordinea băncilor la 1% și la 10%?',
                 '0,84, față de 0,87 la 5% și la 2,5%: ordinea băncilor rămîne apropiată; se schimbă nivelul ΔCoVaR (0,74–1,20% la 10%), nu ordinea.'),
  'tlv-we': ('La α = 5% (▶, rulare suplimentară), cît este ΔCoVaR al TLV?',
             '1,51%, mai puțin de jumătate din 3,31%: panta scade de la 0,611 la 0,542, iar distanța dintre mediana TLV și cuantila ei scade de la 5,41 la 2,79 puncte procentuale; '
             '0,542 × 2,79 = 1,51.'),
  'vardcovar': ('La nivelul 10% (▶, rulare suplimentară), rămîne negativă corelația Spearman dintre VaR și ΔCoVaR?',
                'Nu: devine +0,20 (la 2,5%: −0,13; la 5%: +0,03). Cu 13 bănci, niciuna dintre aceste corelații, nici −0,31 de la 1% (p = 0,31), nu este semnificativă la 5%; '
                'ce rămîne este că VaR-ul propriu nu ordonează băncile după ΔCoVaR.'),
  'dyn': ('La α = 2,5% (▶, rulare suplimentară), cînd atinge maximul ΔCoVaR al JPMorgan Chase?',
          'Pe 28 octombrie 2008 (7,58%), nu în martie 2020 ca la 1% și 5%; în martie 2020 ajunge la 7,52%. Cele două vîrfuri sînt aproape egale, deci data maximului depinde de α.'),
  'mes-we': ('La α = 30% (▶), cît este MES?',
             '6,20% = −(−9,0 − 6,5 − 3,1)/3: se adaugă ziua 9 (piața −2,2%, banca −3,1%). ES 30% al pieței = 4,07%, iar MES scade față de 7,75%, pentru că a treia zi este mai puțin '
             'extremă.'),
  'mes': ('La nivelul 1% (▶, rulare suplimentară), care bancă are cel mai mare MES?',
          'Citigroup, 10,64%, aproape la egalitate cu Bank of America (10,62%); HSBC rămîne cea mai mică (4,86%). La 1%, MES se sprijină pe 53 de zile ale S&P 500.'),
  'mes-tab': ('La nivelul 1% (▶, rulare suplimentară), cum se schimbă corelația Spearman dintre MES și ΔCoVaR 1%?',
              'Devine mai negativă: −0,58, față de −0,29 la 5%. Cele două măsuri răspund la întrebări diferite, iar la același nivel de 1% ordinea lor diferă și mai mult.'),
  'mescrisis': ('La nivelul 1% (▶, rulare suplimentară), devine semnificativă corelația Spearman dintre MES și pierderea din 2007–2008?',
                'Nu: −0,37 (p = 0,26); la 10%: −0,41 (p = 0,21). Semnul este cel din lucrare, dar cu 11 bănci nu este separat de zgomot.'),
  'lrmes-we': ('La MES = 6% (▶), cît este LRMES?',
               '66,0% = 1 − e^(−1,08), nu 83,4% = 2 × 41,7%: funcția 1 − e^(−x) crește din ce în ce mai încet și rămîne sub 100%.'),
  'srisk-we': ('Cu W = 50 de miliarde (▶), cît este SRISK?',
               '45,2 miliarde = 72 − 0,92 × 0,583 × 50 = 72 − 26,8. Efectul de levier L = 950/50 = 19 este mult peste L* = 7,7: o scădere a valorii de piață a capitalului mărește '
               'deficitul.'),
  'srisk': ('Cu k = 5,5% (▶, regula europeană de pe slide-ul următor), peste ce efect de levier are Citigroup nevoie de capital într-o criză?',
            '6,9, față de 5,0 cu k = 8%: un k mai mic cere mai puțin capital, deci pragul L* = 1 + 0,945 × 0,345/0,055 crește. Pentru HSBC pragul urcă la 11,3.'),
  'dgc': ('Cu ferestre de 60 de luni (▶, rulare suplimentară), cît este maximul DGC?',
          '35,5% (fereastra care se încheie în ianuarie 2010), față de 39,1% cu 36 de luni; media crește de la 9,6% la 12,0%, pentru că anii 2007–2009 rămîn în mai multe ferestre.'),
  'gnet': ('▶ mută ambele ferestre cu 12 luni mai devreme (2006–2008 și 2016–2018). Cîte legături are fereastra de criză?',
           '43 (DGC 39%), față de 26 în 2007–2009; fereastra liniștită are 7. Fereastra care se încheie în decembrie 2008 este maximul din graficul precedent.'),
  'dy-we': ('Cu H = 1 zi (▶), cît este FROM pentru JPMorgan Chase?',
            '80,6%, practic ca la H = 10 (80,65%); conectivitatea totală este 78,0% față de 78,3%. Randamentele zilnice sînt aproape imprevizibile, deci descompunerea vine aproape '
            'în întregime din corelația șocurilor din aceeași zi, nu din dinamica VAR.'),
  'dytab': ('Cu H = 30 de zile (▶), cum se schimbă conectivitatea totală?',
            'Rămîne 78,3%: de la H = 5 valoarea nu se mai schimbă la prima zecimală. Coeficienții VAR(1) ai randamentelor zilnice sînt mici, deci efectele unui șoc dispar în cîteva zile.'),
  'dy-tab': ('În subperioada 2020–2026 (▶, rulare suplimentară), care bancă are cel mai negativ NET?',
             'HSBC, −20,1, ca pe întreaga perioadă; băncile americane rămîn transmițători neți (JPMorgan Chase +8,8, Bank of America +11,3), iar conectivitatea totală este 80,2%.'),
  'dyroll': ('Cu ferestre de 100 de zile (▶, rulare suplimentară), rămîne maximul conectivității totale în martie 2020?',
             'Da: 87,9% pe 17 martie 2020. Maximul din 2016 urcă la 86,0%, iar minimul coboară la 54,5% (20 mai 2024): ferestrele scurte dau extreme mai mari în ambele direcții.'),
  'basel-we': ('La sfîrșitul lui ▶ amortizorul anticiclic ajunge la 2,5%. Cît este cerința CET1 a unei bănci G-SIB cu suplimentul minim de 1%?',
               '10,5% = 4,5 + 2,5 + 2,5 + 1 din activele ponderate la risc (' + BCBS11 + '; ' + BCBS18 + '). Cu suplimentul maxim de 3,5% ar fi 13%.'),
 },
 'en': {
  'indices': ('▶ moves the investment of the 100 units to the start of 2010 (extra run). What is the maximum drawdown of the US bank portfolio?',
              '−51%, on 23 November 2011, against −82% with the 2005 investment: the March 2009 trough is no longer in the sample. The maximum drawdown depends on the start of the '
              'sample; for the European banks the deepest point after 2010 is 21 April 2020 (−61%).'),
  'episodes': ('▶ moves the start of each window 30 days later (extra run). What becomes of the minimum of the US banks in the March 2020 episode?',
               '−10.8% instead of −47.6%: the window starts on 15 March 2020, after most of the fall, and the portfolio ends at +16.9%. The cumulative return is measured from the '
               'start of the window, so the comparison of episodes depends on that date.'),
  'rcorr': ('With a 60-day window (▶, extra run), where does the maximum correlation of TLV and BRD go?',
            'To 0.96, on 6 February 2019, still outside any crisis (with 120 days: 0.93 on 9 May 2019). For a single pair of stocks a short window gives larger extremes; the '
            'maximum of the US banks stays on 16 March 2020 (0.95).'),
  'corr-tab': ('▶ goes through the years 2005–2026. In which year is the yearly mean of the TLV–BRD correlation highest?',
               'In 2019: 0.80, above 2020 (0.76) and 2011 (0.56). The 120-day correlation of two stocks is noisy and can peak in a calm year, as the daily maximum of 9 May 2019 '
               'also shows.'),
  'fr-we': ('▶ picks the March 2020 crisis (extra run). Does the corrected correlation ρ* exceed the calm-period correlation of 0.40?',
            'No: ρ_crisis = 0.74, but the variance of the US portfolio rises 47-fold (δ = 46.0), so ρ* = 0.74/√(1 + 46.0 × 0.451) = 0.16. Of the four windows only March 2023 '
            'gives ρ* = 0.42, just above 0.40, because volatility rose little (δ = 3.7).'),
  'tail-tab': ('▶ lowers q to 0.5% (extra run). What happens to λ(q) for the US banks and the S&P 500?',
               'It rises to 0.73, from 0.64 at 5%; under independence it would be 0.005. The estimate rests on the 26 worst days of the S&P 500, so it is imprecise: for the '
               'European banks λ(0.5%) falls to 0.58.'),
  'tscatter': ('At q = 1% (▶, extra run), how many times as many joint bad days do the US banks and the S&P 500 have as under independence?',
               'About 63 times: 33 days against q²n = 0.0001 × 5,229 = 0.52, against 12.7 times at q = 5%. The ratio grows towards the tail: very bad days come together.'),
  'covar-we': ('▶ moves to α = 5% (extra run). Is the CoVaR 5% of the S&P 500 conditional on JPM larger or smaller than 4.79%?',
               'Smaller: 2.45%. The JPM quantile (−3.07%) and the intercept (−1.26) are less extreme, and the slope stays 0.39; ΔCoVaR 5% = 1.22%, about half of 2.47%.'),
  'covarqr': ('▶ goes through α = 2.5%, 5% and 10% (extra run). How does the slope b̂ of the red line change for JPM?',
              'Hardly at all: 0.385, 0.390 and 0.401, against 0.390 at 1% and 0.398 at the median. The intercept moves, from −2.35 to −0.86: the lines are almost parallel, so '
              'ΔCoVaR falls because the distance between the JPM quantiles falls.'),
  'dcovar': ('At the 5% level (▶, extra run), which group of banks has the largest ΔCoVaR?',
             'Still the European banks: 1.43–1.68% (Santander 1.68%), against 1.12–1.33% for the US banks; TLV 1.51%, BRD 1.45%. At 1% the largest value was TLV’s (3.31%).'),
  'dcovar-tab': ('With α₂ = 10% (▶, extra run), what is the Spearman correlation between the ranking of the banks at 1% and at 10%?',
                 '0.84, against 0.87 at 5% and at 2.5%: the ranking stays close; the level of ΔCoVaR changes (0.74–1.20% at 10%), not the ranking.'),
  'tlv-we': ('At α = 5% (▶, extra run), what is the ΔCoVaR of TLV?',
             '1.51%, less than half of 3.31%: the slope falls from 0.611 to 0.542 and the distance between TLV’s median and its quantile falls from 5.41 to 2.79 percentage points; '
             '0.542 × 2.79 = 1.51.'),
  'vardcovar': ('At the 10% level (▶, extra run), does the Spearman correlation between VaR and ΔCoVaR stay negative?',
                'No: it becomes +0.20 (at 2.5%: −0.13; at 5%: +0.03). With 13 banks none of these correlations, not even −0.31 at 1% (p = 0.31), is significant at 5%; what '
                'remains is that a bank’s own VaR does not rank the banks by ΔCoVaR.'),
  'dyn': ('At α = 2.5% (▶, extra run), when does the ΔCoVaR of JPMorgan Chase peak?',
          'On 28 October 2008 (7.58%), not in March 2020 as at 1% and 5%; in March 2020 it reaches 7.52%. The two peaks are almost equal, so the date of the maximum depends on α.'),
  'mes-we': ('At α = 30% (▶), what is MES?',
             '6.20% = −(−9.0 − 6.5 − 3.1)/3: day 9 is added (market −2.2%, bank −3.1%). The ES 30% of the market = 4.07%, and MES falls from 7.75% because the third day is less '
             'extreme.'),
  'mes': ('At the 1% level (▶, extra run), which bank has the largest MES?',
          'Citigroup, 10.64%, almost tied with Bank of America (10.62%); HSBC stays the smallest (4.86%). At 1% MES rests on 53 days of the S&P 500.'),
  'mes-tab': ('At the 1% level (▶, extra run), how does the Spearman correlation between MES and ΔCoVaR 1% change?',
              'It becomes more negative: −0.58, against −0.29 at 5%. The two measures answer different questions, and at the same 1% level their rankings differ even more.'),
  'mescrisis': ('At the 1% level (▶, extra run), does the Spearman correlation between MES and the 2007–2008 loss become significant?',
                'No: −0.37 (p = 0.26); at 10%: −0.41 (p = 0.21). The sign is the paper’s, but with 11 banks it is not separated from noise.'),
  'lrmes-we': ('At MES = 6% (▶), what is LRMES?',
               '66.0% = 1 − e^(−1.08), not 83.4% = 2 × 41.7%: the function 1 − e^(−x) rises more and more slowly and stays below 100%.'),
  'srisk-we': ('With W = 50 billion (▶), what is SRISK?',
               '45.2 billion = 72 − 0.92 × 0.583 × 50 = 72 − 26.8. The leverage L = 950/50 = 19 is far above L* = 7.7: a fall in the market value of equity increases the shortfall.'),
  'srisk': ('With k = 5.5% (▶, the European rule of the next slide), above what leverage does Citigroup need capital in a crisis?',
            '6.9, against 5.0 with k = 8%: a smaller k requires less capital, so the threshold L* = 1 + 0.945 × 0.345/0.055 rises. For HSBC the threshold rises to 11.3.'),
  'dgc': ('With 60-month windows (▶, extra run), what is the maximum DGC?',
          '35.5% (the window that ends in January 2010), against 39.1% with 36 months; the mean rises from 9.6% to 12.0%, because 2007–2009 stays in more windows.'),
  'gnet': ('▶ moves both windows 12 months earlier (2006–2008 and 2016–2018). How many links does the crisis window have?',
           '43 (DGC 39%), against 26 in 2007–2009; the calm window has 7. The window that ends in December 2008 is the maximum of the previous chart.'),
  'dy-we': ('With H = 1 day (▶), what is FROM for JPMorgan Chase?',
            '80.6%, practically as at H = 10 (80.65%); the total connectedness is 78.0% against 78.3%. Daily returns are almost unpredictable, so the decomposition comes almost '
            'entirely from the correlation of same-day shocks, not from the VAR dynamics.'),
  'dytab': ('With H = 30 days (▶), how does the total connectedness change?',
            'It stays 78.3%: from H = 5 on, the value no longer changes at the first decimal. The VAR(1) coefficients of daily returns are small, so the effects of a shock die out '
            'within a few days.'),
  'dy-tab': ('In the 2020–2026 sub-period (▶, extra run), which bank has the most negative NET?',
             'HSBC, −20.1, as over the whole period; the US banks stay net transmitters (JPMorgan Chase +8.8, Bank of America +11.3), and the total connectedness is 80.2%.'),
  'dyroll': ('With 100-day windows (▶, extra run), does the maximum of the total connectedness stay in March 2020?',
             'Yes: 87.9% on 17 March 2020. The 2016 maximum rises to 86.0% and the minimum falls to 54.5% (20 May 2024): short windows give larger extremes in both directions.'),
  'basel-we': ('At the end of ▶ the countercyclical buffer reaches 2.5%. What is the CET1 requirement of a G-SIB with the minimum surcharge of 1%?',
               '10.5% = 4.5 + 2.5 + 2.5 + 1 of risk-weighted assets (' + BCBS11 + '; ' + BCBS18 + '). With the maximum surcharge of 3.5% it would be 13%.'),
 },
}

PREDICT = {
 'indices': (0, [('Circa −51%.', 'About −51%.'), ('Tot circa −82%: drawdown-ul maxim nu depinde de începutul eșantionului.', 'Still about −82%: the maximum drawdown does not depend on the start of the sample.'),
                 ('Circa 0%: portofoliul a crescut de la 100 la 817.', 'About 0%: the portfolio rose from 100 to 817.')]),
 'episodes': (0, [('Circa −11%.', 'About −11%.'), ('Tot circa −48%: minimul este o dată din calendar.', 'Still about −48%: the minimum is a calendar date.'),
                  ('Sub −48%: o fereastră mai scurtă dă pierderi mai mari.', 'Below −48%: a shorter window gives larger losses.')]),
 'rcorr': (0, [('Circa 0,96, tot în 2019, în afara unei crize.', 'About 0.96, still in 2019, outside a crisis.'), ('În martie 2020: maximul unei corelații apare într-o criză.', 'In March 2020: the maximum of a correlation comes in a crisis.'),
               ('Sub 0,93: o fereastră mai scurtă dă corelații mai mici.', 'Below 0.93: a shorter window gives smaller correlations.')]),
 'corr-tab': (0, [('2019.', '2019.'), ('2020: corelația este cea mai mare în criză.', '2020: correlation is highest in a crisis.'),
                  ('2011: criza din zona euro, cea mai mare scădere a băncilor românești.', '2011: the euro-area crisis, the largest fall of the Romanian banks.')]),
 'fr-we': (0, [('Nu, ρ* este circa 0,16.', 'No, ρ* is about 0.16.'), ('Da: 0,74 > 0,40, deci contagiune.', 'Yes: 0.74 > 0.40, so contagion.'),
               ('Da: ρ* este pozitiv, deci legătura s-a întărit.', 'Yes: ρ* is positive, so the link got stronger.')]),
 'tail-tab': (0, [('Crește, la circa 0,73.', 'It rises, to about 0.73.'), ('Scade spre 0,005: la q mic dependența dispare, ca la copula Gaussiană.', 'It falls towards 0.005: at small q dependence vanishes, as in the Gaussian copula.'),
                  ('Rămîne exact 0,64: λ(q) nu depinde de q.', 'It stays exactly 0.64: λ(q) does not depend on q.')]),
 'tscatter': (0, [('De circa 63 de ori.', 'About 63 times.'), ('Tot de circa 13 ori, ca la q = 5%.', 'Still about 13 times, as at q = 5%.'),
                  ('De circa 0,6 ori: 33 față de q × n = 52.', 'About 0.6 times: 33 against q × n = 52.')]),
 'covar-we': (0, [('Mai mic, circa 2,45%.', 'Smaller, about 2.45%.'), ('Tot 4,79%: CoVaR nu depinde de α.', 'Still 4.79%: CoVaR does not depend on α.'),
                  ('Mai mare: 5% este o probabilitate mai mare decît 1%.', 'Larger: 5% is a larger probability than 1%.')]),
 'covarqr': (0, [('Rămîne aproape 0,39; se mută termenul liber.', 'It stays about 0.39; the intercept moves.'), ('Crește spre 1: la α mare indicele urmează banca unu la unu.', 'It rises towards 1: at a large α the index follows the bank one for one.'),
                 ('Scade spre 0: la 10% banca nu mai contează pentru indice.', 'It falls towards 0: at 10% the bank no longer matters for the index.')]),
 'dcovar': (0, [('Băncile europene.', 'The European banks.'), ('Băncile românești, ca la 1%.', 'The Romanian banks, as at 1%.'),
                ('Băncile americane: cele mai mari bănci au cel mai mare ΔCoVaR.', 'The US banks: the largest banks have the largest ΔCoVaR.')]),
 'dcovar-tab': (0, [('Circa 0,84.', 'About 0.84.'), ('Circa 0: la 10% cozile nu mai contează.', 'About 0: at 10% the tails no longer matter.'), ('Exact 1: ordinea nu depinde de α.', 'Exactly 1: the ranking does not depend on α.')]),
 'tlv-we': (0, [('Circa 1,5%.', 'About 1.5%.'), ('Tot 3,31%: ΔCoVaR nu depinde de α.', 'Still 3.31%: ΔCoVaR does not depend on α.'), ('Circa 16,5% = 5 × 3,31%: probabilitatea este de 5 ori mai mare.', 'About 16.5% = 5 × 3.31%: the probability is 5 times as large.')]),
 'vardcovar': (0, [('Nu, devine pozitivă, circa +0,20.', 'No, it turns positive, about +0.20.'), ('Da, tot circa −0,31.', 'Yes, still about −0.31.'),
                   ('Da, aproape −1: băncile riscante contribuie puțin la riscul sistemic.', 'Yes, close to −1: risky banks contribute little to systemic risk.')]),
 'dyn': (0, [('În octombrie 2008.', 'In October 2008.'), ('Tot în martie 2020.', 'Still in March 2020.'), ('În martie 2023, la falimentul SVB.', 'In March 2023, at the SVB failure.')]),
 'mes-we': (0, [('6,20%.', '6.20%.'), ('Tot 7,75%: tot zilele 5 și 1.', 'Still 7.75%: still days 5 and 1.'), ('18,6%: suma pierderilor din cele trei zile.', '18.6%: the sum of the losses of the three days.')]),
 'mes': (0, [('Citigroup.', 'Citigroup.'), ('ING, a doua la 5%.', 'ING, second at 5%.'), ('HSBC: cea mai mare bancă are cel mai mare MES.', 'HSBC: the largest bank has the largest MES.')]),
 'mes-tab': (0, [('Devine mai negativă, circa −0,58.', 'It becomes more negative, about −0.58.'), ('Devine +1: la același nivel cele două măsuri coincid.', 'It becomes +1: at the same level the two measures coincide.'),
                 ('Devine aproape zero.', 'It becomes about zero.')]),
 'mescrisis': (0, [('Nu: p rămîne peste 0,2.', 'No: p stays above 0.2.'), ('Da: MES la 1% măsoară coada mai bine.', 'Yes: MES at 1% measures the tail better.'),
                   ('Da: semnul negativ arată deja că MES prezice pierderile.', 'Yes: the negative sign already shows that MES predicts the losses.')]),
 'lrmes-we': (0, [('Circa 66%.', 'About 66%.'), ('Circa 83%: dublul lui 41,7%.', 'About 83%: twice 41.7%.'), ('108% = 18 × 6%.', '108% = 18 × 6%.')]),
 'srisk-we': (0, [('Circa 45 de miliarde.', 'About 45 billion.'), ('9,2 miliarde: jumătate din 18,4.', '9.2 billion: half of 18.4.'), ('18,4 miliarde: SRISK nu depinde de W.', '18.4 billion: SRISK does not depend on W.')]),
 'srisk': (0, [('Circa 6,9.', 'About 6.9.'), ('Tot 5,0: L* nu depinde de k.', 'Still 5.0: L* does not depend on k.'), ('Sub 5,0: un k mai mic face banca mai riscantă.', 'Below 5.0: a smaller k makes the bank riskier.')]),
 'dgc': (0, [('Circa 36%.', 'About 36%.'), ('Aproape 100%: mai multe date fac toate legăturile semnificative.', 'Close to 100%: more data make every link significant.'),
             ('Circa 5%: ferestrele lungi elimină legăturile din criză.', 'About 5%: long windows remove the crisis links.')]),
 'gnet': (0, [('Circa 43.', 'About 43.'), ('Tot 26.', 'Still 26.'), ('Circa 5, ca în anii liniștiți.', 'About 5, as in the calm years.')]),
 'dy-we': (0, [('Tot circa 81%.', 'Still about 81%.'), ('Circa 0%: într-o zi șocurile celorlalte bănci nu au timp să ajungă.', 'About 0%: in one day the other banks’ shocks have no time to arrive.'),
               ('Circa 8%: o zecime din valoarea pentru H = 10.', 'About 8%: a tenth of the value for H = 10.')]),
 'dytab': (0, [('Rămîne circa 78%.', 'It stays about 78%.'), ('Crește spre 100%: pe termen lung totul este conectat.', 'It rises towards 100%: in the long run everything is connected.'),
               ('Scade: la orizonturi lungi rămîne doar partea proprie.', 'It falls: at long horizons only the own share remains.')]),
 'dy-tab': (0, [('HSBC.', 'HSBC.'), ('JPMorgan Chase: cea mai mare bancă primește cel mai mult.', 'JPMorgan Chase: the largest bank receives the most.'), ('Deutsche Bank.', 'Deutsche Bank.')]),
 'dyroll': (0, [('Da, cu circa 88%.', 'Yes, at about 88%.'), ('Nu, trece în 2016.', 'No, it moves to 2016.'), ('Nu, trece în octombrie 2008.', 'No, it moves to October 2008.')]),
 'basel-we': (0, [('10,5%.', '10.5%.'), ('8%: amortizorul anticiclic nu intră în cerința CET1.', '8%: the countercyclical buffer is not part of the CET1 requirement.'),
                  ('13%: o bancă G-SIB are întotdeauna suplimentul maxim.', '13%: a G-SIB always has the maximum surcharge.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout and those elements
VEIL = {a: {'out': True} for a in PREDICT}
VEIL.update({'corr-tab': {'out': True, 'charts': ['ct-tab']}})

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Banca A are un VaR 1% mai mare decît banca B. Este A mai importantă sistemic?', 'Bank A has a larger VaR 1% than bank B. Is A more systemically important?'),
      key=0, opts=[
  (('Nu neapărat: importanța sistemică depinde de legătura cu sistemul (ΔCoVaR, MES).', 'Not necessarily: systemic importance depends on the link with the system (ΔCoVaR, MES).'),
   ('VaR descrie coada proprie a băncii; în curs, Citigroup și Bank of America sînt printre primele trei după VaR 1% și printre ultimele trei după ΔCoVaR.', 'VaR describes the bank’s own tail; in the lecture Citigroup and Bank of America are among the top three by VaR 1% and the bottom three by ΔCoVaR.')),
  (('Da: o pierdere proprie mai mare înseamnă o contribuție mai mare la riscul sistemului.', 'Yes: a larger own loss means a larger contribution to the risk of the system.'),
   ('Corelația Spearman dintre VaR și ΔCoVaR pentru cele 13 bănci este −0,31: VaR-ul propriu nu ordonează băncile după contribuția la sistem.', 'The Spearman correlation between VaR and ΔCoVaR of the 13 banks is −0.31: the own VaR does not rank banks by their contribution to the system.')),
  (('Nu: o bancă cu VaR mare are întotdeauna un ΔCoVaR mic.', 'No: a bank with a large VaR always has a small ΔCoVaR.'),
   ('Corelația negativă din curs este slabă și nesemnificativă cu 13 bănci; nu există o regulă inversă.', 'The negative correlation of the lecture is weak and not significant with 13 banks; there is no reverse rule.'))]),
 dict(q=('O regresie cuantilică la 1% dă b̂ = 0,5, q̂₁% = −6% și q̂₅₀% = 0. Cît este ΔCoVaR 1%?', 'A 1% quantile regression gives b̂ = 0.5, q̂₁% = −6% and q̂₅₀% = 0. What is ΔCoVaR 1%?'),
      key=0, opts=[
  (('3%.', '3%.'), ('b̂ × (q̂₅₀% − q̂₁%) = 0,5 × (0 − (−6)) = 3%.', 'b̂ × (q̂₅₀% − q̂₁%) = 0.5 × (0 − (−6)) = 3%.')),
  (('−3%.', '−3%.'), ('Diferența este luată invers (q̂₁% − q̂₅₀%); ΔCoVaR este o creștere a pierderii, deci pozitiv.', 'The difference is taken the wrong way (q̂₁% − q̂₅₀%); ΔCoVaR is an increase in the loss, so positive.')),
  (('6%.', '6%.'), ('6 este distanța dintre cele două cuantile ale băncii; ea trebuie înmulțită cu panta b̂ = 0,5.', '6 is the distance between the two quantiles of the bank; it must be multiplied by the slope b̂ = 0.5.'))]),
 dict(q=('De ce poate fi SRISK negativ?', 'Why can SRISK be negative?'),
      key=0, opts=[
  (('O bancă cu efect de levier mic rămîne cu suficient capital chiar după o scădere a pieței de 40%.', 'A bank with low leverage keeps enough capital even after a 40% market fall.'),
   ('SRISK = kD − (1 − k)(1 − LRMES)W < 0 cînd L = (D + W)/W < L*: banca are un surplus de capital.', 'SRISK = kD − (1 − k)(1 − LRMES)W < 0 when L = (D + W)/W < L*: the bank has a capital surplus.')),
  (('Cînd LRMES depășește 100%.', 'When LRMES exceeds 100%.'), ('LRMES = 1 − exp(−18 MES) este mereu sub 100%; în plus, un LRMES mai mare crește SRISK.', 'LRMES = 1 − exp(−18 MES) is always below 100%; besides, a larger LRMES raises SRISK.')),
  (('Cînd MES este negativ, adică banca cîștigă în zilele proaste ale pieței.', 'When MES is negative, that is, the bank gains on the market’s bad days.'),
   ('Toate cele 13 bănci au MES pozitiv; SRISK este negativ din cauza efectului de levier mic, nu a unui MES negativ.', 'All 13 banks have a positive MES; SRISK is negative because of low leverage, not of a negative MES.'))]),
 dict(q=('Corelația dintre băncile americane și cele europene crește de la 0,40 la 0,60 în criza din 2008, iar varianța sursei crește de 85 de ori (δ = 83,7). Ce arată testul Forbes–Rigobon?',
         'The correlation of the US and European banks rises from 0.40 to 0.60 in the 2008 crisis, and the variance of the source rises 85-fold (δ = 83.7). What does the Forbes–Rigobon test show?'),
      key=0, opts=[
  (('Interdependență: ρ* = 0,08 < 0,40.', 'Interdependence: ρ* = 0.08 < 0.40.'), ('0,60/√(1 + 83,7 × 0,638) = 0,60/√54,4 = 0,08: corectată pentru volatilitate, corelația nu crește.', '0.60/√(1 + 83.7 × 0.638) = 0.60/√54.4 = 0.08: corrected for volatility, the correlation does not rise.')),
  (('Contagiune: 0,60 > 0,40.', 'Contagion: 0.60 > 0.40.'), ('Comparația ignoră că o varianță mai mare a sursei crește automat corelația măsurată.', 'The comparison ignores that a larger variance of the source raises the measured correlation automatically.')),
  (('Contagiune: ρ* este pozitiv.', 'Contagion: ρ* is positive.'), ('ρ* se compară cu ρ_calm = 0,40, nu cu zero.', 'ρ* is compared with ρ_calm = 0.40, not with zero.'))]),
 dict(q=('Care este direcția condiționării în CoVaR și în MES?', 'What is the direction of conditioning in CoVaR and in MES?'),
      key=0, opts=[
  (('CoVaR: sistemul condiționat de bancă; MES: banca condiționată de piață.', 'CoVaR: the system conditional on the bank; MES: the bank conditional on the market.'),
   ('CoVaR = −qα(Xˢʸˢ | Xⁱ = qα(Xⁱ)); MES = −E[Xⁱ | Xᵐ ≤ qα(Xᵐ)].', 'CoVaR = −qα(Xˢʸˢ | Xⁱ = qα(Xⁱ)); MES = −E[Xⁱ | Xᵐ ≤ qα(Xᵐ)].')),
  (('Ambele condiționează sistemul de bancă.', 'Both condition the system on the bank.'), ('MES alege zilele după randamentul pieței și face media randamentelor băncii.', 'MES chooses the days by the market’s return and averages the bank’s returns.')),
  (('CoVaR: banca condiționată de sistem; MES: piața condiționată de bancă.', 'CoVaR: the bank conditional on the system; MES: the market conditional on the bank.'),
   ('Direcțiile sînt inversate; în CoVaR, banca este variabila explicativă a regresiei cuantilice.', 'The directions are swapped; in CoVaR the bank is the explanatory variable of the quantile regression.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch15-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH15_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH15_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH15_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
