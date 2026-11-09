"""Build the interactive pages of Chapter 6: interactive/ch6.html (EN) and interactive/ch6_ro.html (RO).

One block per chart slide of the lecture body and one block per results table (XBLOCKS), in the order of the slides: the slide
on the left, its interactive version on the right, with live tables and worked examples. Logic in interactive/ch6.js, inputs
in interactive/ch6_data.js (Quantlets/Ch_06/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside the chart block of the same section (widgets) and
ch6.js moves each one to its own frame.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 6
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_06'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('candidates', 'sfm_ch6_candidates', 'cand'), ('dax-monthly', 'sfm_ch6_dax_monthly', 'cand'),
          ('profile-nu', 'sfm_ch6_profile_nu', 'ml'),
          ('delta-aic', 'sfm_ch6_delta_aic', 'ic'),
          ('ks-tail', 'sfm_ch6_ks_tail', 'gof'), ('gof-power', 'sfm_ch6_gof_power', 'gof'),
          ('pp', 'sfm_ch6_pp', 'ppqq'), ('qq', 'sfm_ch6_qq', 'ppqq'), ('qq-assets', 'sfm_ch6_qq_assets', 'ppqq'),
          ('var-models', 'sfm_ch6_var_models', 'var'), ('left-tail', 'sfm_ch6_left_tail', 'var'),
          ('oos', 'sfm_ch6_oos', 'oos'), ('overfit', 'sfm_ch6_overfit', 'oos'), ('rolling-var', 'sfm_ch6_rolling_var', 'oos'),
          ('var-qqplot', 'sfm_ch6_var_qqplot', 'oos'),
          ('model-uncertainty', 'sfm_ch6_model_uncertainty', 'unc')]
# table blocks: (anchor, frame number in the RO deck (EN = RO - 1), section key, chart block that carries it on the web page)
XBLOCKS = [('desc-stats', 12, 'cand', 'dax-monthly'), ('ml-table', 16, 'ml', 'profile-nu'), ('lr-test', 22, 'lr', 'profile-nu'),
           ('vuong', 24, 'lr', 'profile-nu'), ('aic-table', 28, 'ic', 'delta-aic'), ('ks-five', 33, 'gof', 'ks-tail'),
           ('edf-table', 38, 'gof', 'gof-power'), ('var-es', 45, 'var', 'var-models'), ('kupiec', 48, 'var', 'left-tail'),
           ('model-avg', 59, 'unc', 'model-uncertainty')]
ORDER = ['candidates', 'dax-monthly', 'desc-stats', 'ml-table', 'profile-nu', 'lr-test', 'vuong', 'aic-table', 'delta-aic', 'ks-five', 'ks-tail',
         'gof-power', 'edf-table', 'pp', 'qq', 'qq-assets', 'var-es', 'var-models', 'left-tail', 'kupiec', 'oos', 'overfit', 'rolling-var', 'var-qqplot',
         'model-uncertainty', 'model-avg']

PFX = {'candidates': 'ca', 'dax-monthly': 'dm', 'desc-stats': 'ds', 'ml-table': 'ml', 'profile-nu': 'pn', 'lr-test': 'lr', 'vuong': 'vu', 'aic-table': 'ai',
       'delta-aic': 'da', 'ks-five': 'k5', 'ks-tail': 'kt', 'gof-power': 'gp', 'edf-table': 'ed', 'pp': 'pp', 'qq': 'qq', 'qq-assets': 'qa', 'var-es': 've',
       'var-models': 'vm', 'left-tail': 'lt', 'kupiec': 'ku', 'oos': 'oo', 'overfit': 'of', 'rolling-var': 'rv', 'var-qqplot': 'vq', 'model-uncertainty': 'mu',
       'model-avg': 'av'}

SECTIONS = {
 'ro': dict(cand='Candidații', ml='Recapitulare: metoda verosimilității maxime', lr='Modele imbricate: testul raportului de verosimilitate',
            ic='Criterii informaționale: AIC și BIC', gof='Teste de concordanță', ppqq='Grafice PP și QQ',
            var='Compararea modelelor după VaR 1%', oos='În afara eșantionului: validare și supraajustare', unc='Incertitudinea de model'),
 'en': dict(cand='The Candidates', ml='Review of Maximum Likelihood Estimation', lr='Nested Models: the Likelihood-Ratio Test', ic='Information Criteria: AIC and BIC',
            gof='Goodness-of-Fit Tests', ppqq='PP and QQ Plots', var='Comparing the Models on VaR 1%',
            oos='Out of Sample: Validation and Overfitting', unc='Model Uncertainty'),
}

L = {
 'ro': dict(file='ch6_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 6 · Selecția modelului și managementul riscului', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; tabelele și exemplele lucrate se recalculează odată cu el.',
   src='Randamente logaritmice zilnice, în %: BET, S&P 500 și DAX din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), Banca Transilvania (TLV), '
       'OMV Petrom (SNP) și BRD din 2010, din prețul ajustat, pînă la 18 septembrie 2026; cele două zile greșite ale TLV din mai 2016 sînt eliminate. DAX lunar: '
       'prima zi de tranzacționare a fiecărei luni (SFESumm). Prețuri de închidere EODHD. Parametrii estimați, log-verosimilitățile, cuantilele modelelor, '
       'VaR pe fereastră mobilă și reeșantionările bootstrap sînt cele din curs (același cod, aceeași sămînță a generatorului); statisticile descriptive, '
       'VaR și ES empirice, depășirile, testele Kupiec, scorurile pinball și graficul SFEVaRqqplot sînt recalculate în browser din aceleași randamente. '
       'Nivelurile VaR diferite de 1%, celelalte serii ale profilului lui ν, testul Christoffersen și ponderile BIC sînt rulări suplimentare ale aceluiași cod.'),
 'en': dict(file='ch6.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 6 · Model Selection and Risk Management', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; the tables and worked examples are recomputed with it.',
   src='Daily log returns, in %: BET, S&P 500 and DAX from January 2000, Bitcoin from September 2014 (7 days a week), Banca Transilvania (TLV), OMV Petrom (SNP) '
       'and BRD from 2010, from the adjusted price, to 18 September 2026; the two erroneous TLV days of May 2016 are removed. Monthly DAX: first trading day of '
       'each month (SFESumm). EODHD closing prices. The fitted parameters, log-likelihoods, model quantiles, rolling VaR and bootstrap resamples are those of the '
       'lecture (the same code and the same generator seed); the summary statistics, empirical VaR and ES, exceedances, Kupiec tests, pinball losses and the '
       'SFEVaRqqplot chart are recomputed in the browser from the same returns. VaR levels other than 1%, the profile of ν for the other series, the '
       'Christoffersen test and the BIC weights are extra runs of the same code.'),
}

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child,.xt td.l,.xt th.l{text-align:left}
.xt td.l{white-space:normal;min-width:16rem}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt .hi{font-weight:700;color:#2E7D32}.xt .bad{color:#CD0000}.xt .x{font-style:italic}
.xt tr.sel td{background:#EEF3FB}.xt .pv{font-size:.76rem;color:#1F2A44}
.xh{font-weight:600;font-size:.92rem;color:#1F2A44;margin:.9rem 0 .1rem}
.we{border:1px solid #C5D2E8;border-left:4px solid #1A3A6E;border-radius:6px;padding:.55rem .8rem;margin:.8rem 0 .4rem;background:#F6F8FC}
.we-h{font-weight:600;margin:0 0 .2rem;color:#1A3A6E}
.we-f{overflow-x:auto;overflow-y:hidden;font-size:.95rem}
.we ol{margin:.3rem 0 0 1.3rem;padding:0;font-size:.92rem}.we li{margin:.18rem 0}
.tw.veiled-chart,.we.veiled-chart{filter:blur(10px);opacity:.35;user-select:none}
.xhold{display:none}
.ctl input[type=range]:disabled{opacity:.4}
</style>"""


def tw(i):
    return f'<div class="tw" id="{i}"></div>'


def xh(t):
    return f'<p class="xh">{t}</p>'


def we(i, title, tex):
    return f'<div class="we" id="{i}-box"><p class="we-h">{title}</p><div class="we-f">\\[{tex}\\]</div><ol id="{i}"></ol></div>'


def widgets_all(lang):
    ro = lang == 'ro'
    S7 = [('bet', 'BET'), ('sp500', 'S&P 500'), ('dax', 'DAX'), ('btc', 'Bitcoin'), ('tlv', 'Banca Transilvania (TLV)'), ('snp', 'OMV Petrom (SNP)'), ('brd', 'BRD')]
    S4 = S7[:4]
    MN = {'Normal': 'Normală' if ro else 'Normal', 'Student-t': 'Student-t', 'Skewed-t': 'skewed-t', 'GED': 'GED', 'Mixture': 'amestec Normal' if ro else 'Normal mixture',
          'NIG': 'NIG', 'Stable': 'stabilă' if ro else 'stable'}
    M7 = [(m, MN[m]) for m in MN]
    lv = [('0', '0,1%' if ro else '0.1%'), ('1', '0,25%' if ro else '0.25%'), ('2', '0,5%' if ro else '0.5%'), ('3', '1%'), ('4', '2,5%' if ro else '2.5%'), ('5', '5%')]
    ser = 'Seria' if ro else 'Series'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    return {
     'candidates': (slider('ca-x', 'Pragul x: probabilitatea P(X < −x)' if ro else 'Threshold x: the probability P(X < −x)', 0, 8, 1, 4) + play('ca'),
                    CSS + charts('ca-chart') + xh('P(X < −x) pentru fiecare candidat (media 0, varianța 1)' if ro else 'P(X < −x) for each candidate (mean 0, variance 1)') + tw('ca-tab')),
     'dax-monthly': (slider('dm-n', 'Sfîrșitul eșantionului (luna)' if ro else 'End of the sample (month)', 119, 266, 1, 266) + play('dm'),
                     charts('dm-chart') + xh('Statistici descriptive, randamente logaritmice lunare DAX (%)' if ro else 'Summary statistics, DAX monthly log returns (%)') + tw('dm-tab')),
     'desc-stats': (select('ds-b', 'Banca Transilvania (TLV)', [('clean', 'fără cele două zile greșite' if ro else 'without the two erroneous days'), ('raw', 'cu cele două zile greșite' if ro else 'with the two erroneous days')], 'clean')
                    + select('ds-s', 'Seria exemplului lucrat' if ro else 'Series of the worked example', S7, 'sp500') + play('ds'),
                    xh('Randamente logaritmice zilnice (%)' if ro else 'Daily log returns (%)') + tw('ds-tab')
                    + we('ds-we', f'{WE}: {"testul Jarque–Bera" if ro else "the Jarque–Bera test"}', r'JB = \frac{n}{6}\Big[S^2 + \frac{(K-3)^2}{4}\Big] \approx \chi^2(2)')),
     'ml-table': (select('ml-s', ser, S7, 'sp500') + play('ml'),
                  xh('Estimări ML și log-verosimilitatea maximizată' if ro else 'ML estimates and the maximised log-likelihood') + tw('ml-tab')
                  + we('ml-we', f'{WE}: {"modelul Normal, pas cu pas" if ro else "the Normal model, step by step"}', r'\ell(\hat\mu, \hat\sigma) = -\frac{n}{2}\big[\ln(2\pi\hat\sigma^2) + 1\big], \quad \hat\sigma^2 = \frac{1}{n}\sum_t (r_t - \bar r)^2')),
     'profile-nu': (select('pn-s', ser, S7, 'sp500') + select('pn-c', 'Nivelul de încredere' if ro else 'Confidence level', [('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')], '0.95')
                    + select('pn-y', 'Axa verticală' if ro else 'Vertical axis', [('full', 'completă, ca pe slide' if ro else 'full, as on the slide'), ('cut', 'tăiată la −40' if ro else 'cut at −40')], 'full')
                    + slider('pn-nu', 'ν testat (punctul portocaliu)' if ro else 'Tested ν (orange point)', 2.2, 6, 0.1, 3) + play('pn'),
                    charts('pn-chart') + we('pn-we', f'{WE}: {"este ν testat în interval?" if ro else "is the tested ν inside the interval?"}', r'LR(\nu) = 2\big[\ell(\hat\nu) - \ell(\nu)\big] \le \chi^2_{1-a}(1)')),
     'lr-test': (select('lr-p', 'Perechea de modele imbricate' if ro else 'Pair of nested models', [('t_vs_skt', 'Student-t în skewed-t (λ = 0)' if ro else 'Student-t inside skewed-t (λ = 0)'),
                    ('n_vs_ged', 'Normală în GED (β = 2)' if ro else 'Normal inside GED (β = 2)'), ('n_vs_t', 'Normală în Student-t (1/ν = 0, frontieră)' if ro else 'Normal inside Student-t (1/ν = 0, boundary)')], 't_vs_skt')
                 + select('lr-s', ser, S7, 'sp500') + play('lr'),
                 we('lr-we', f'{WE}: {"testul raportului de verosimilitate" if ro else "the likelihood-ratio test"}', r'LR = 2\,[\ell_1(\hat\theta_1) - \ell_0(\hat\theta_0)] \approx \chi^2(k_1 - k_0)')
                 + xh('Același test pe cele șapte serii' if ro else 'The same test on the seven series') + tw('lr-tab')
                 + xh('p-value ca funcție de LR' if ro else 'The p-value as a function of LR') + charts('lr-chart')),
     'vuong': (select('vu-p', 'Perechea de modele neimbricate' if ro else 'Pair of non-nested models', [('vu_t_ged', 'Student-t față de GED' if ro else 'Student-t against GED'),
                  ('vu_t_stable', 'Student-t față de stabilă' if ro else 'Student-t against stable'), ('vu_skt_nig', 'skewed-t față de NIG' if ro else 'skewed-t against NIG')], 'vu_t_ged')
               + select('vu-s', ser, S7, 'sp500') + play('vu'),
               we('vu-we', f'{WE}: {"statistica Vuong" if ro else "the Vuong statistic"}', r'm_t = \ln f_1(r_t) - \ln f_2(r_t), \qquad V = \frac{\sqrt{n}\,\bar m}{s_m} \approx N(0, 1)')
               + xh('Statistica V și p-value pe cele șapte serii' if ro else 'The statistic V and its p-value on the seven series') + tw('vu-tab')
               + xh('Suma cumulată a diferențelor zilnice m<sub>t</sub>' if ro else 'Cumulative sum of the daily differences m<sub>t</sub>') + charts('vu-chart')),
     'aic-table': (select('ai-s', ser, S7, 'sp500') + select('ai-m', 'Modelul exemplului lucrat' if ro else 'Model of the worked example', M7, 'NIG') + play('ai'),
                   tw('ai-tab') + we('ai-we', f'{WE}: AIC, BIC {"și ponderea Akaike" if ro else "and the Akaike weight"}',
                                     r'\mathrm{AIC} = -2\ell(\hat\theta) + 2k, \quad \mathrm{BIC} = -2\ell(\hat\theta) + k\ln n, \quad w_i = \frac{e^{-\Delta_i/2}}{\sum_j e^{-\Delta_j/2}}')),
     'delta-aic': (select('da-c', 'Criteriul' if ro else 'Criterion', [('aic', 'ΔAIC'), ('bic', 'ΔBIC')], 'aic') + play('da'),
                   charts('da-chart') + xh('Cel mai bun model după fiecare criteriu' if ro else 'The best model under each criterion') + tw('da-tab')),
     'ks-five': (select('k5-h', 'Ipoteza nulă' if ro else 'Null hypothesis', [('known', 'N(0, σ²), complet specificată' if ro else 'N(0, σ²), fully specified'),
                    ('est', 'Normală cu μ și σ estimate (Lilliefors)' if ro else 'Normal with estimated μ and σ (Lilliefors)')], 'known')
                 + slider('k5-s', 'σ în ipoteza nulă' if ro else 'σ under the null', 0.5, 2, 0.1, 1) + play('k5'),
                 tw('k5-tab') + we('k5-we', f'{WE}: {"statistica KS" if ro else "the KS statistic"}', r'D_n = \max_i \max\Big\{\frac{i}{n} - u_{(i)},\ u_{(i)} - \frac{i-1}{n}\Big\}, \quad u_{(i)} = F(r_{(i)})')
                 + charts('k5-chart')),
     'ks-tail': (select('kt-s', ser, S7, 'sp500') + play('kt'),
                 charts('kt-chart') + we('kt-we', f'{WE}: {"distanța KS în punctul de maxim" if ro else "the KS distance at its maximum"}', r'D_n = \max_i \max\Big\{\frac{i}{n} - u_{(i)},\ u_{(i)} - \frac{i-1}{n}\Big\}, \qquad c_{0{,}05} = \frac{1{,}358}{\sqrt n}' if ro else r'D_n = \max_i \max\Big\{\frac{i}{n} - u_{(i)},\ u_{(i)} - \frac{i-1}{n}\Big\}, \qquad c_{0.05} = \frac{1.358}{\sqrt n}')),
     'gof-power': (slider('gp-n', 'Mărimea eșantionului n (linia punctată)' if ro else 'Sample size n (dotted line)', 0, 4, 1, 2) + play('gp'),
                   charts('gp-chart') + xh('Puterea la 5%: date Student-t (ν = 5) și date Normale cu 2% din zile de 4 ori mai volatile' if ro else 'Power at 5%: Student-t data (ν = 5) and Normal data with 2% of days 4 times as volatile') + tw('gp-tab')
                   + xh('Valori critice KS la 5%: parametri estimați (simulare, 4 000 de eșantioane) și F cunoscută' if ro else 'KS critical values at 5%: estimated parameters (simulation, 4,000 samples) and known F') + tw('gp-tab2')),
     'edf-table': (select('ed-s', ser, S7, 'sp500') + play('ed'),
                   xh('Statisticile EDF ale modelelor estimate; clasamentul după AD și după AIC' if ro else 'EDF statistics of the fitted models; ranks by AD and by AIC') + tw('ed-tab')),
     'pp': (select('pp-s', ser, S7, 'sp500') + select('pp-4', 'Al patrulea panou' if ro else 'Fourth panel', [('Stable', MN['Stable']), ('NIG', 'NIG'), ('GED', 'GED'), ('Mixture', MN['Mixture'])], 'Stable') + play('pp'),
            charts('pp-chart')),
     'qq': (select('qq-s', ser, S7, 'sp500') + select('qq-4', 'Al patrulea panou' if ro else 'Fourth panel', [('Stable', MN['Stable']), ('NIG', 'NIG'), ('GED', 'GED'), ('Mixture', MN['Mixture'])], 'Stable') + play('qq'),
            charts('qq-chart') + xh('Cuantila fiecărui model la probabilitatea celui mai mic randament, 0,5/n (%)' if ro else 'Quantile of each model at the probability of the smallest return, 0.5/n (%)') + tw('qq-tab')),
     'qq-assets': (slider('qa-k', 'Randamentul marcat: al k-lea cel mai mic' if ro else 'Marked return: the k-th smallest', 1, 10, 1, 1) + play('qa'), charts('qa-chart')),
     'var-es': (select('ve-a', 'Nivelul VaR' if ro else 'VaR level', lv, '3') + select('ve-e', 'Nivelul ES' if ro else 'ES level', lv, '4') + select('ve-k', 'Seria exemplului lucrat' if ro else 'Series of the worked example', S7, 'sp500') + play('ve'),
                f'<p class="xh" id="ve-h1"></p>' + tw('ve-tab') + f'<p class="xh" id="ve-h2"></p>' + tw('ve-tab2')
                + we('ve-we', f'{WE}: VaR {"și ES ale modelului Normal" if ro else "and ES of the Normal model"}', r'\mathrm{VaR}_\alpha = -(\hat\mu + \hat\sigma z_\alpha), \qquad \mathrm{ES}_\alpha = -\hat\mu + \hat\sigma\,\frac{\varphi(z_\alpha)}{\alpha}')),
     'var-models': (select('vm-a', 'Nivelul VaR' if ro else 'VaR level', lv, '3') + select('vm-x', 'Modelele' if ro else 'Models', [('six', 'cele șase din slide' if ro else 'the six of the slide'), ('mix', 'și amestecul Normal' if ro else 'and the Normal mixture')], 'six') + play('vm'),
                    charts('vm-chart')),
     'left-tail': (select('lt-s', ser, S7, 'sp500') + slider('lt-x', 'Pierderea x (linia punctată verticală)' if ro else 'Loss x (dashed line)', 0, 10, 1, 4) + play('lt'), charts('lt-chart')),
     'kupiec': (select('ku-a', 'Nivelul VaR' if ro else 'VaR level', lv, '3') + select('ku-k', ser, S7, 'sp500') + select('ku-m', 'Modelul' if ro else 'Model', [(m, MN[m]) for m in ('Normal', 'Student-t', 'Skewed-t', 'GED', 'NIG', 'Stable')], 'NIG') + play('ku'),
                xh('Depășiri în eșantion (p-value Kupiec în paranteză)' if ro else 'In-sample exceedances (Kupiec p-value in brackets)') + tw('ku-tab')
                + we('ku-we', f'{WE}: {"testul Kupiec" if ro else "the Kupiec test"}', r'LR_{uc} = -2\ln\frac{(1-\alpha)^{n-x}\alpha^x}{(1 - x/n)^{n-x}(x/n)^x} \approx \chi^2(1)')
                + charts('ku-chart')),
     'oos': (select('oo-r', 'Panoul din dreapta' if ro else 'Right panel', [('rate', 'rata depășirilor' if ro else 'exceedance rate'), ('pin', 'pierderea pinball' if ro else 'pinball loss'), ('kup', 'statistica Kupiec' if ro else 'Kupiec statistic')], 'rate')
             + select('oo-s', 'Seria exemplului lucrat' if ro else 'Series of the worked example', S4, 'sp500')
             + select('oo-m', 'Metoda exemplului lucrat' if ro else 'Method of the worked example', [(m, MN[m]) for m in ('Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG', 'Stable')] + [('Historical', 'simulare istorică' if ro else 'historical simulation')], 'Student-t') + play('oo'),
             charts('oo-chart') + xh('Depășiri ale VaR 1% în zilele de test, 2020–2026 (p-value Kupiec în paranteză)' if ro else 'VaR 1% exceedances on the test days, 2020–2026 (Kupiec p-value in brackets)') + tw('oo-tab')
             + we('oo-we', f'{WE}: {"pierderea pinball a cuantilei de 1%" if ro else "the pinball loss of the 1% quantile"}', r'\frac{1}{n}\sum_t \big(\alpha - \mathbf{1}\{r_t < q\}\big)(r_t - q), \qquad \alpha = 0{,}01' if ro else r'\frac{1}{n}\sum_t \big(\alpha - \mathbf{1}\{r_t < q\}\big)(r_t - q), \qquad \alpha = 0.01')),
     'overfit': (slider('of-p', 'Penalizarea pe parametru (2 = AIC, ln 501 = 6,22 = BIC)' if ro else 'Penalty per parameter (2 = AIC, ln 501 = 6.22 = BIC)', 0, 10, 0.1, 2)
                 + select('of-c', 'Numărul de componente al exemplului lucrat' if ro else 'Components in the worked example', [(str(c), str(c)) for c in range(1, 7)], '3') + play('of'),
                 charts('of-chart') + tw('of-tab') + we('of-we', f'{WE}: AIC {"și" if ro else "and"} BIC', r'\mathrm{AIC} = -2\ell + 2k, \quad \mathrm{BIC} = -2\ell + k\ln n, \quad k = 3c - 1')),
     'rolling-var': (select('rv-s', ser, [('sp500', 'S&P 500'), ('bet', 'BET')], 'sp500')
                     + select('rv-f', 'Perioada afișată' if ro else 'Period shown', [('2003-01-01', '2003–2026'), ('2008-01-01', '2008–2026'), ('2018-01-01', '2018–2026 (slide)'), ('2020-01-01', '2020–2026')], '2018-01-01')
                     + select('rv-m', 'Depășirile marcate' if ro else 'Marked exceedances', [('Normal', MN['Normal']), ('Student-t', 'Student-t'), ('Skewed-t', 'skewed-t'), ('Historical', 'simulare istorică' if ro else 'historical simulation')], 'Student-t') + play('rv'),
                     charts('rv-chart') + xh('Depășiri pe an' if ro else 'Exceedances per year') + charts('rv-bar') + xh('Backtesting pe toată perioada de test' if ro else 'Backtest over the whole test period') + tw('rv-tab')
                     + we('rv-we', f'{WE}: {"testul de independență Christoffersen (rulare suplimentară)" if ro else "the Christoffersen independence test (extra run)"}',
                          r'LR_{ind} = -2\ln\frac{(1-\hat\pi)^{n_{00}+n_{10}}\,\hat\pi^{\,n_{01}+n_{11}}}{(1-\hat\pi_{01})^{n_{00}}\hat\pi_{01}^{\,n_{01}}(1-\hat\pi_{11})^{n_{10}}\hat\pi_{11}^{\,n_{11}}} \approx \chi^2(1)')),
     'var-qqplot': (slider('vq-l', 'λ al EMA' if ro else 'λ of the EMA', 0.9, 0.99, 0.01, 0.96) + select('vq-h', 'Fereastra h (zile)' if ro else 'Window h (days)', [('125', '125'), ('250', '250'), ('500', '500')], '250') + play('vq'),
                    charts('vq-chart') + tw('vq-tab')),
     'model-uncertainty': (slider('mu-b', 'Numărul de reeșantionări B' if ro else 'Number of resamples B', 20, 200, 20, 200) + play('mu'),
                           charts('mu-chart') + xh('Reeșantionările bootstrap pe blocuri mobile, S&P 500' if ro else 'Moving-block bootstrap resamples, S&P 500') + tw('mu-tab')),
     'model-avg': (select('av-a', 'Nivelul VaR' if ro else 'VaR level', lv, '3') + select('av-w', 'Ponderile' if ro else 'Weights', [('aic', 'Akaike (AIC)'), ('bic', 'BIC')], 'aic')
                   + select('av-x', 'Intervalul cozilor groase' if ro else 'Heavy-tailed range', [('mix', 'șase modele nenormale (slide)' if ro else 'the six non-Normal models (slide)'), ('nomix', 'fără amestecul Normal' if ro else 'without the Normal mixture')], 'mix')
                   + select('av-s', 'Seria exemplului lucrat' if ro else 'Series of the worked example', S7, 'tlv') + play('av'),
                   xh('VaR (%) al datelor, al modelelor și mediat' if ro else 'VaR (%) of the data, of the models and averaged') + tw('av-tab')
                   + we('av-we', f'{WE}: {"VaR mediat" if ro else "the averaged VaR"}', r'\mathrm{VaR}^{avg}_\alpha = \sum_i w_i\,\mathrm{VaR}_{\alpha,i}, \qquad \text{interval} = \frac{\max_i \mathrm{VaR}_i}{\min_i \mathrm{VaR}_i} - 1' if ro else r'\mathrm{VaR}^{avg}_\alpha = \sum_i w_i\,\mathrm{VaR}_{\alpha,i}, \qquad \text{range} = \frac{\max_i \mathrm{VaR}_i}{\min_i \mathrm{VaR}_i} - 1')),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    res = {}
    for a, fr, _, _ in XBLOCKS:
        k = fr - 1 if lang == 'ro' else fr - 2
        assert 'includegraphics' not in frames[k][1] and ('tabular' in frames[k][1] or 'Exemplu lucrat' in frames[k][0] or 'Worked Example' in frames[k][0]), (lang, a)
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its section's table blocks (moved by ch6.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch6_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch6_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s\u00a0]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => num(s) === 1 ? 'o zi' : `${s} ${de(s)}zile`;
window.CH6_TEXT = {
 mn: { Normal: 'Normală', 'Student-t': 'Student-t', 'Skewed-t': 'skewed-t', GED: 'GED', Mixture: 'amestec Normal', NIG: 'NIG', Stable: 'stabilă' },
 h: { model: 'Model', ratioN: 'față de Normală', oneIn: 'o zi din', sd: 'ab. std.', annvol: 'vol. anuală', skew: 'asim.', kurt: 'boltire', from: 'din', est: 'Estimări',
   loc: 'locație', scale: 'scală', dec: 'decizia la 5%', rej: 'respins', norej: 'nerespins', pref: 'preferat la 5%', none: 'niciunul', bestA: 'cel mai bun AIC', bestB: 'cel mai bun BIC',
   second: 'al doilea (Δ)', kst: 'KS (cozi)', cvmt: 'CvM (cozi)', adt: 'AD (cozi)', lil: 'μ, σ estimate (Lilliefors)', ksk: 'F cunoscută', rAD: 'loc după AD', rAIC: 'loc după AIC',
   qmin: 'cuantila (%)', ratioObs: 'față de minimul observat', data: 'date', expct: 'așteptat', hist: 'istorică', bestAIC: 'cel mai bun AIC', bestLS: 'cel mai bun scor', bestPin: 'cel mai bun pinball',
   pcrit: '−2ℓ + penalizare·k', inS: 'în eșantion', outS: 'în afara eșantionului', rate: 'rata', maxY: 'maximul anual', pind: 'p (independență)', exc: 'depășiri', maxR: 'max L/VaR',
   wins: 'cîștigă AIC', med: 'VaR 1% median', i90: 'interval de 90%', width: 'lățime', rangeH: 'interval (cozi groase)', avg: 'mediat', rangeP: 'interval (%)' },
 ca: (x, r, pn) => `La x = ${x}, distribuția Student-t (ν = 4) dă o probabilitate de ${r} ${de(r)}ori mai mare decît distribuția Normală (${pn}). Toate curbele au media 0 și varianța 1 (distribuția stabilă: același interval intercuartilic ca distribuția Normală).`,
 dm: (e, n, k, mn, w) => `Eșantionul iulie 2004 – ${e}: ${n} ${de(n)}luni, boltirea ${k}, cea mai mare scădere ${mn}% (${w}).`,
 ds1: (s, n, S, K) => `${s}: n = ${n}, asimetria S = ${S}, boltirea K = ${K}.`,
 ds2: (s2, k2) => `S<sup>2</sup> = ${s2}; (K − 3)<sup>2</sup>/4 = ${k2}.`,
 ds3: (n, jb) => `JB = ${n}/6 × (suma de mai sus) = ${jb}.`,
 ds4: (jb, p) => `Sub ipoteza de normalitate JB ~ χ²(2), cu valoarea critică 5,99 la 5%: p-value = e<sup>−JB/2</sup> ${p}; normalitatea este respinsă dacă JB > 5,99.`,
 dsBad: (k1, k0, n) => `Cu cele două zile greșite (30 și 31 mai 2016), TLV are ${n} ${de(n)}zile și boltirea ${k1}, față de ${k0} fără ele.`,
 dsClean: (k0, n) => `TLV fără cele două zile greșite: ${n} ${de(n)}zile, boltirea ${k0}; selectorul de mai sus le adaugă înapoi.`,
 ml1: (s, n, m, sd) => `${s}: n = ${n}, μ̂ = r̄ = ${m}%, σ̂ = ${sd}% (numitorul n).`,
 ml2: (sd, v, l) => `σ̂<sup>2</sup> = ${sd}<sup>2</sup> = ${v}; ln(2π σ̂<sup>2</sup>) = ${l}.`,
 ml3: (n, l, ll) => `ℓ(θ̂) = −${n}/2 × (${l} + 1) = ${ll}.`,
 ml4: (lt, d, m) => `Tabelul dă ${lt} pentru modelul Normal; cel mai bun model după log-verosimilitate, ${m}, are cu ${d} mai mult.`,
 ml: (s, nu, sc, sd, sdd) => `${s}, Student-t: ν̂ = ${nu}; scala ${sc} nu este abaterea standard: sd = scală·√(ν/(ν − 2)) = ${sd}%, față de abaterea standard de selecție ${sdd}%.`,
 pn1: (s, nu, l, lm, nh) => `${s}: ℓ(ν = ${nu}) = ${l}, maximizată în raport cu locația și scala; ℓ(ν̂) = ${lm} la ν̂ = ${nh}.`,
 pn2: (d, lr) => `ℓ(ν) − ℓ(ν̂) = ${d}, deci LR = 2 × [ℓ(ν̂) − ℓ(ν)] = ${lr}.`,
 pn3: (lr, c, lev, ins) => `Valoarea critică χ²(1) la ${lev}% este ${c}: LR = ${lr} ${ins ? '≤' : '>'} ${c}, deci ν ${ins ? 'este în interval' : 'este în afara intervalului'}.`,
 pn: (s, nu, lev, a, b, c, d) => `${s}: ν̂ = ${nu}. Intervalul de ${lev}% citit pe grila cu pasul 0,1 a graficului: [${a}; ${b}]; pe o grilă cu pasul 0,01 (rulare suplimentară): [${c}; ${d}].`,
 pn4: (l3, l5, full) => `La ν = 3 curba este cu ${l3.replace('−', '')} sub maxim, la ν = 5 cu ${l5.replace('−', '')}` + (full ? '.' : ', sub marginea de jos a axei tăiate la −40.'),
 lr1: (s, m0, l0, m1, l1) => `${s}: ℓ<sub>0</sub> (${m0}) = ${l0}; ℓ<sub>1</sub> (${m1}) = ${l1}.`,
 lr2: (l1, l0, lr) => `LR = 2 × (${l1} − (${l0})) = ${lr}.`,
 lr3: (b, c, lr, p, pn) => b ? `Frontieră (1/ν = 0): statistica urmează amestecul 50:50 de χ²(0) și χ²(1); valoarea critică la 5% este ${c}, p-value = ½ P(χ²(1) > ${lr}) ${p} (p-value-ul naiv ${pn.replace('= ', '')}).` : `Un parametru fixat, d = 1: valoarea critică la 5% este ${c}; p-value = P(χ²(1) > ${lr}) ${p}.`,
 lr4: (rej, m0) => rej ? `LR depășește valoarea critică: modelul restrîns (${m0}) este respins la 5%.` : `LR nu depășește valoarea critică: datele nu resping modelul restrîns (${m0}) la 5%; testul nu arată că parametrul suplimentar este zero.`,
 lr: (m0, m1, b, c) => `${m0} în ${m1}${b ? ' (frontieră: valoarea critică 2,71)' : ''}: modelul restrîns este respins la 5% pentru ${c} din 7 serii.`,
 vu1: (s, a, b, n, m, sm) => `${s}, ${a} față de ${b}: n = ${n} zile, media diferențelor zilnice m̄ = ${m}, abaterea lor standard s<sub>m</sub> = ${sm}.`,
 vu2: (n, m, sm, v) => `V = √${n} × ${m} / ${sm} = ${v}.`,
 vu3: (v, p, w) => `p-value bilateral = 2[1 − Φ(|V|)] ${p}; ` + (w ? `|V| > 1,96: ${w} este mai aproape de distribuția datelor.` : '|V| ≤ 1,96: testul nu separă cele două modele.'),
 vu4: (n, m, s, d) => `Suma diferențelor, n × m̄ = ${n} × ${m} = ${s}, este diferența log-verosimilităților maximizate (${d}): V o raportează la variabilitatea zilnică a lui m<sub>t</sub>.`,
 vu: (s, top) => `${s}, zilele cu cea mai mare |m<sub>t</sub>|: ` + top.map(([d, r, m]) => `${d} (randament ${r}%, m<sub>t</sub> = ${m})`).join('; ') + '.',
 ai1: (m, s, l, k, a) => `${m}, ${s}: ℓ(θ̂) = ${l}, k = ${k}; AIC = −2 × (${l}) + 2 × ${k} = ${a}.`,
 ai2: (l, k, n, ln, b, m2) => `BIC = −2 × (${l}) + ${k} × ln ${n} = ${m2} + ${k} × ${ln} = ${b}.`,
 ai3: (a, mn, d, e) => `Δ = ${a} − ${mn} = ${d}; e<sup>−Δ/2</sup> = ${e}.`,
 ai4: (list, s, e, w) => `Termenii e<sup>−Δ<sub>j</sub>/2</sup>: ${list}; suma ${s}; w = ${e} / ${s} = ${w}.`,
 ai: (s, b, s2, d, w, ln) => `${s}: cel mai bun AIC are ${b}; următorul, ${s2}, are ΔAIC = ${d} și ponderea ${w}. Penalizarea BIC pe parametru: ln n = ${ln}.`,
 da: (c, a, b, diff) => `Modelul Normal are Δ${c} între ${a} și ${b}. ` + (diff.length ? `AIC și BIC aleg modele diferite pentru: ${diff.join(', ')}.` : 'AIC și BIC aleg același model pe toate seriile.'),
 k51: (est, m, s) => est ? `μ și σ estimate pe cele cinci randamente: μ̂ = ${m}, σ̂ = ${s} (numitorul n); u<sub>(i)</sub> = Φ((r<sub>(i)</sub> − μ̂)/σ̂).` : `H<sub>0</sub>: N(0, ${s}²), complet specificată; u<sub>(i)</sub> = Φ(r<sub>(i)</sub>/${s}).`,
 k52: u => `u<sub>(i)</sub>: ${u}.`,
 k53: (i, side, d) => `Cea mai mare distanță apare la i = ${i}: ` + (side ? `u<sub>(${i})</sub> − ${i - 1}/5` : `${i}/5 − u<sub>(${i})</sub>`) + ` = ${d}.`,
 k54: (d, c, est, rej) => `D<sub>5</sub> = ${d} ${rej ? '>' : '<'} ${c}, ` + (est ? 'valoarea critică Lilliefors la 5% pentru n = 5 (simulare din curs, 4 000 de eșantioane)' : 'valoarea critică exactă la 5% pentru n = 5') + (rej ? ': respingem.' : ': nu respingem.'),
 k5: (d, c, est, rej) => `D<sub>5</sub> = ${d}; valoarea critică ${est ? 'Lilliefors' : 'exactă'} la 5%: ${c}; ${rej ? 'H<sub>0</sub> este respinsă' : 'H<sub>0</sub> nu este respinsă'}.`,
 kt1: (s, n, m, sd) => `${s}: n = ${n}; distribuția Normală estimată are μ̂ = ${m}%, σ̂ = ${sd}%.`,
 kt2: (i, x, u, a, b) => `Maximul distanței: al ${i}-lea cel mai mic randament, r<sub>(i)</sub> = ${x}%; u<sub>(i)</sub> = F(r<sub>(i)</sub>) = ${u}; i/n = ${a}; (i − 1)/n = ${b}.`,
 kt3: (p, m, d) => `i/n − u<sub>(i)</sub> = ${p}; u<sub>(i)</sub> − (i − 1)/n = ${m}; D<sub>n</sub> = ${d}.`,
 kt4: (n, c, d) => `Valoarea critică pentru F cunoscută: 1,358/√${n} = ${c}; D<sub>n</sub> = ${d} o depășește. Cu parametri estimați, valoarea critică corectă este și mai mică (problema Lilliefors).`,
 kt: (s, x, f, d, xw, zw) => `${s}: distanța KS are maximul la x = ${x}% (F = ${f}%), D<sub>n</sub> = ${d}; distanța ponderată Anderson–Darling are maximul la x = ${xw}% (valoarea ${zw}).`,
 gp: (n, a, b, c, d, e, f) => `n = ${n}: date Student-t, puterea KS ${a}%, CvM ${b}%, AD ${c}%; abateri doar în cozi: KS ${d}%, CvM ${e}%, AD ${f}%. 1 000 de eșantioane simulate, valori critice simulate pentru parametri estimați.`,
 ed: (s, n, c, diff) => `${s}, n = ${n}: valoarea critică KS pentru F cunoscută ar fi 1,358/√n = ${c}; cu parametri estimați este mai mică. ` + (diff.length ? `Clasamentul după AD diferă de cel după AIC pentru: ${diff.join(', ')}.` : 'Clasamentul după AD coincide cu cel după AIC.'),
 ed2: (rows, nv, B) => `Bootstrap parametric, B = ${B}: ` + rows.map(([m, k, a, p]) => `${m}: valorile critice KS ${k}, AD ${a}, p bootstrap ${p}`).join('; ') + `. Student-t, p-value-ul KS naiv (ca și cum θ ar fi cunoscut): ${nv}.`,
 pp: (s, rows) => `${s}, cea mai mare distanță pe verticală în graficul PP (pe toate zilele) și statistica KS: ` + rows.map(([m, d, k]) => `${m} ${d} (KS ${k})`).join('; ') + '.',
 qq: (s, mn, d, n, p) => `${s}: cel mai mic randament ${mn}% (${d}), probabilitatea în grafic 0,5/n = ${p} (n = ${n}).`,
 qa: k => `Al ${k}-lea cel mai mic randament și cuantilele modelelor la aceeași probabilitate (k − 0,5)/n:`,
 qaItem: (a, d, r, n, t, s) => `${a}: ${d}, ${r}%; Normală ${n}%, Student-t ${t}%, skewed-t ${s}%`,
 veH: (l, v) => v ? `VaR ${l} (%) al datelor și al fiecărui model estimat; amestecul Normal în italic` : `ES ${l} (%) al datelor și al fiecărui model estimat`,
 ve1: (s, m, sd, a, z) => `${s}, modelul Normal estimat: μ̂ = ${m}%, σ̂ = ${sd}%; z<sub>α</sub> pentru α = ${a}: ${z}.`,
 ve2: (m, z, sd, v, e) => `VaR = −(${m} + (${z}) × ${sd}) = ${v}%; VaR empiric (cuantila de selecție) ${e}%.`,
 ve3: (a, z, f, m, sd, es, e) => `ES ${a}: z = ${z}, φ(z) = ${f}; ES = −${m} + ${sd} × ${f}/α = ${es}%; ES empiric (media randamentelor sub cuantilă, cu semn schimbat) ${e}%.`,
 ve: (a, e) => `VaR ${a} și ES ${e}: valorile modelelor provin din funcțiile de cuantilă ale modelelor estimate în curs; 1% și 2,5% sînt cele din curs, celelalte niveluri sînt rulări suplimentare.`,
 vm: a => `VaR ${a} al modelului împărțit la VaR ${a} empiric, cel mai mic și cel mai mare raport pe cele șapte serii (abaterea în %):`,
 vmItem: (m, lo, hi) => `${m}: de la ${lo}% la ${hi}%`,
 lt: (s, x, c, n, p) => `${s}: ${zile(String(c))} din ${n} cu o pierdere peste ${x}% (proporția ${p}). Aceeași probabilitate în modele, și numărul de zile așteptat în eșantion:`,
 ltItem: rows => rows.map(([m, p, e]) => `${m} ${p} (${e} ${de(e)}zile)`).join('; '),
 ku1: (m, s, a, v, n, x, r) => `${m}, ${s}: VaR ${a} = ${v}%; din n = ${n} zile, x = ${x} au r<sub>t</sub> < −VaR (rata ${r}%).`,
 ku2: (nx, a, x, l0) => `ln L<sub>0</sub> = ${nx} × ln(1 − ${a}) + ${x} × ln ${a} = ${l0}.`,
 ku3: (r, l1) => `ln L<sub>1</sub>, cu rata observată x/n = ${r}: ${l1}.`,
 ku4: (l0, l1, lr, p, rej) => `LR<sub>uc</sub> = −2 × (${l0} − (${l1})) = ${lr}; p = P(χ²(1) > LR) ${p}: ${rej ? 'numărul depășirilor este respins la 5%' : 'numărul depășirilor nu este respins la 5%'}.`,
 ku: (m, s, x, e, y, c) => `${m}, ${s}: ${x} ${de(x)}depășiri față de ${e} așteptate; cele mai multe într-un an: ${c} în ${y}.`,
 oo1: (s, m, ne, v, nt, d) => `${s}, ${m}: VaR 1% estimat pe ${ne} ${de(ne)}zile pînă la 31.12.2019, VaR = ${v}%, deci q = −${v}%; ${nt} ${de(nt)}zile de test, din ${d}.`,
 oo2: (x, s1) => `Cele ${x} ${de(x)}zile cu r<sub>t</sub> < q contribuie (α − 1)(r<sub>t</sub> − q), în total ${s1}.`,
 oo3: (x, s0) => `Celelalte ${x} ${de(x)}zile contribuie α(r<sub>t</sub> − q), în total ${s0}.`,
 oo4: (a, b, n, p) => `Pierderea pinball = (${a} + ${b}) / ${n} = ${p}.`,
 oo: list => `Cîștigătorul AIC din perioada de estimare are și cel mai bun scor logaritmic pentru: ${list.join(', ') || 'nicio serie'}.`,
 of1: (c, k, l, n) => `c = ${c} componente: k = 3c − 1 = ${k} parametri; ℓ = ${l} pe n = ${n} zile (2017–2018).`,
 of2: (l, k, a) => `AIC = −2 × (${l}) + 2 × ${k} = ${a}.`,
 of3: (l, k, ln, b) => `BIC = −2 × (${l}) + ${k} × ${ln} = ${b}.`,
 of4: (i, o, d) => `Log-verosimilitatea medie pe zi: ${i} în eșantion, ${o} în afara lui (2019–2026); diferența ${d}.`,
 of: (p, c, ln) => `Cu penalizarea ${p} pe parametru, criteriul −2ℓ + penalizare·k alege c = ${c}. AIC folosește 2, BIC ln 501 = ${ln}; cel mai bun în afara eșantionului este c = 3.`,
 rv1: (s, m, n, a, b, c, d) => `${s}, ${m}: ${n} ${de(n)}zile de test; perechi de zile consecutive: n<sub>00</sub> = ${a}, n<sub>01</sub> = ${b}, n<sub>10</sub> = ${c}, n<sub>11</sub> = ${d} (1 = depășire).`,
 rv2: (a, b, p01, c, d, p11, p) => `π̂<sub>01</sub> = ${a}/${b} = ${p01} (o depășire după o zi fără); π̂<sub>11</sub> = ${c}/${d} = ${p11} (o depășire după o depășire); π̂ = ${p}.`,
 rv3: (l0, l1, lr, p, rej) => `ln L(π̂) = ${l0}, ln L(π̂<sub>01</sub>, π̂<sub>11</sub>) = ${l1}; LR<sub>ind</sub> = ${lr}, p ${p}: ${rej ? 'depășirile apar grupat, independența este respinsă la 5%' : 'independența nu este respinsă la 5%'}.`,
 rv: (s, d, n, f) => `${s}: fiecare model este reestimat la fiecare 20 de zile pe cele 1 000 de zile anterioare; backtesting din ${d}, ${n} ${de(n)}zile. Graficul arată perioada de la ${f.slice(0, 4)}; tabelul acoperă toată perioada.`,
 vq: (n, d, h, l, hl) => `DAX, ${n} ${de(n)}zile din ${d}; volatilitatea din ultimele ${h} ${de(h)}zile (RMA) sau cu ponderi λ<sup>j</sup>, λ = ${l} (EMA, timpul de înjumătățire al ponderii ${hl} zile).`,
 mu: (b, w, full) => (full ? 'Toate cele 200 de reeșantionări din curs' : `Primele ${b} ${b >= 20 ? 'de ' : ''}reeșantionări (cursul folosește 200)`) + `, pe blocuri de 20 de zile: NIG are cel mai mic AIC în ${w}% dintre ele.`,
 av1: (s, c, rows) => `${s}, ponderile ${c}: ` + rows.map(([m, w, v]) => `${m} w = ${w}, VaR ${v}%`).join('; ') + '.',
 av2: (t, a, l) => `VaR ${l} mediat = ${t} = ${a}% (termenii cu ponderi sub 0,0005 sînt omiși).`,
 av3: (h, mh, l, ml, r) => `Intervalul cozilor groase: de la ${l}% (${ml}) la ${h}% (${mh}); ${h}/${l} − 1 = ${r}%.`,
 av: (a, c, mix) => `VaR ${a}; ponderi ${c}; intervalul ${mix ? 'include amestecul Normal, ca în curs' : 'fără amestecul Normal'}.`
};
})();
""",
 'en': r"""(function () {
const num = s => +String(s).replace(/,/g, '').replace('−', '-');
const days = s => num(s) === 1 ? '1 day' : `${s} days`;
window.CH6_TEXT = {
 mn: { Normal: 'Normal', 'Student-t': 'Student-t', 'Skewed-t': 'skewed-t', GED: 'GED', Mixture: 'Normal mixture', NIG: 'NIG', Stable: 'stable' },
 h: { model: 'Model', ratioN: 'relative to Normal', oneIn: 'one day in', sd: 'st. dev.', annvol: 'annual vol.', skew: 'skew.', kurt: 'kurtosis', from: 'from', est: 'Estimates',
   loc: 'location', scale: 'scale', dec: 'decision at 5%', rej: 'rejected', norej: 'not rejected', pref: 'preferred at 5%', none: 'neither', bestA: 'best AIC', bestB: 'best BIC',
   second: 'second (Δ)', kst: 'KS (tails)', cvmt: 'CvM (tails)', adt: 'AD (tails)', lil: 'μ, σ estimated (Lilliefors)', ksk: 'known F', rAD: 'rank by AD', rAIC: 'rank by AIC',
   qmin: 'quantile (%)', ratioObs: 'relative to the observed minimum', data: 'data', expct: 'expected', hist: 'historical', bestAIC: 'best AIC', bestLS: 'best score', bestPin: 'best pinball',
   pcrit: '−2ℓ + penalty·k', inS: 'in sample', outS: 'out of sample', rate: 'rate', maxY: 'annual maximum', pind: 'p (independence)', exc: 'exceedances', maxR: 'max L/VaR',
   wins: 'wins AIC', med: 'median VaR 1%', i90: '90% interval', width: 'width', rangeH: 'range (heavy tails)', avg: 'averaged', rangeP: 'range (%)' },
 ca: (x, r, pn) => `At x = ${x}, the Student-t distribution (ν = 4) gives a probability ${r} times that of the Normal distribution (${pn}). All curves have mean 0 and variance 1 (the stable distribution: the same interquartile range as the Normal distribution).`,
 dm: (e, n, k, mn, w) => `Sample July 2004 – ${e}: ${n} months, kurtosis ${k}, largest fall ${mn}% (${w}).`,
 ds1: (s, n, S, K) => `${s}: n = ${n}, skewness S = ${S}, kurtosis K = ${K}.`,
 ds2: (s2, k2) => `S<sup>2</sup> = ${s2}; (K − 3)<sup>2</sup>/4 = ${k2}.`,
 ds3: (n, jb) => `JB = ${n}/6 × (the sum above) = ${jb}.`,
 ds4: (jb, p) => `Under Normality JB ~ χ²(2), with critical value 5.99 at 5%: p-value = e<sup>−JB/2</sup> ${p}; Normality is rejected if JB > 5.99.`,
 dsBad: (k1, k0, n) => `With the two erroneous days (30 and 31 May 2016), TLV has ${n} days and kurtosis ${k1}, against ${k0} without them.`,
 dsClean: (k0, n) => `TLV without the two erroneous days: ${n} days, kurtosis ${k0}; the selector above adds them back.`,
 ml1: (s, n, m, sd) => `${s}: n = ${n}, μ̂ = r̄ = ${m}%, σ̂ = ${sd}% (denominator n).`,
 ml2: (sd, v, l) => `σ̂<sup>2</sup> = ${sd}<sup>2</sup> = ${v}; ln(2π σ̂<sup>2</sup>) = ${l}.`,
 ml3: (n, l, ll) => `ℓ(θ̂) = −${n}/2 × (${l} + 1) = ${ll}.`,
 ml4: (lt, d, m) => `The table gives ${lt} for the Normal model; the best model by log-likelihood, ${m}, has ${d} more.`,
 ml: (s, nu, sc, sd, sdd) => `${s}, Student-t: ν̂ = ${nu}; the scale ${sc} is not the standard deviation: sd = scale·√(ν/(ν − 2)) = ${sd}%, against the sample standard deviation ${sdd}%.`,
 pn1: (s, nu, l, lm, nh) => `${s}: ℓ(ν = ${nu}) = ${l}, maximised over location and scale; ℓ(ν̂) = ${lm} at ν̂ = ${nh}.`,
 pn2: (d, lr) => `ℓ(ν) − ℓ(ν̂) = ${d}, so LR = 2 × [ℓ(ν̂) − ℓ(ν)] = ${lr}.`,
 pn3: (lr, c, lev, ins) => `The χ²(1) critical value at ${lev}% is ${c}: LR = ${lr} ${ins ? '≤' : '>'} ${c}, so ν is ${ins ? 'inside' : 'outside'} the interval.`,
 pn: (s, nu, lev, a, b, c, d) => `${s}: ν̂ = ${nu}. The ${lev}% interval read on the chart's grid of step 0.1: [${a}, ${b}]; on a grid of step 0.01 (extra run): [${c}, ${d}].`,
 pn4: (l3, l5, full) => `At ν = 3 the curve is ${l3.replace('−', '')} below its maximum, at ν = 5 ${l5.replace('−', '')} below` + (full ? '.' : ', under the bottom of the axis cut at −40.'),
 lr1: (s, m0, l0, m1, l1) => `${s}: ℓ<sub>0</sub> (${m0}) = ${l0}; ℓ<sub>1</sub> (${m1}) = ${l1}.`,
 lr2: (l1, l0, lr) => `LR = 2 × (${l1} − (${l0})) = ${lr}.`,
 lr3: (b, c, lr, p, pn) => b ? `Boundary (1/ν = 0): the statistic follows the 50:50 mixture of χ²(0) and χ²(1); the 5% critical value is ${c}, p-value = ½ P(χ²(1) > ${lr}) ${p} (naive p-value ${pn.replace('= ', '')}).` : `One fixed parameter, d = 1: the 5% critical value is ${c}; p-value = P(χ²(1) > ${lr}) ${p}.`,
 lr4: (rej, m0) => rej ? `LR exceeds the critical value: the restricted model (${m0}) is rejected at 5%.` : `LR does not exceed the critical value: the data do not reject the restricted model (${m0}) at 5%; the test does not show that the extra parameter is zero.`,
 lr: (m0, m1, b, c) => `${m0} inside ${m1}${b ? ' (boundary: critical value 2.71)' : ''}: the restricted model is rejected at 5% for ${c} of 7 series.`,
 vu1: (s, a, b, n, m, sm) => `${s}, ${a} against ${b}: n = ${n} days, mean of the daily differences m̄ = ${m}, their standard deviation s<sub>m</sub> = ${sm}.`,
 vu2: (n, m, sm, v) => `V = √${n} × ${m} / ${sm} = ${v}.`,
 vu3: (v, p, w) => `Two-sided p-value = 2[1 − Φ(|V|)] ${p}; ` + (w ? `|V| > 1.96: ${w} is closer to the distribution of the data.` : '|V| ≤ 1.96: the test does not separate the two models.'),
 vu4: (n, m, s, d) => `The sum of the differences, n × m̄ = ${n} × ${m} = ${s}, is the difference of the maximised log-likelihoods (${d}): V scales it by the day-to-day variability of m<sub>t</sub>.`,
 vu: (s, top) => `${s}, the days with the largest |m<sub>t</sub>|: ` + top.map(([d, r, m]) => `${d} (return ${r}%, m<sub>t</sub> = ${m})`).join('; ') + '.',
 ai1: (m, s, l, k, a) => `${m}, ${s}: ℓ(θ̂) = ${l}, k = ${k}; AIC = −2 × (${l}) + 2 × ${k} = ${a}.`,
 ai2: (l, k, n, ln, b, m2) => `BIC = −2 × (${l}) + ${k} × ln ${n} = ${m2} + ${k} × ${ln} = ${b}.`,
 ai3: (a, mn, d, e) => `Δ = ${a} − ${mn} = ${d}; e<sup>−Δ/2</sup> = ${e}.`,
 ai4: (list, s, e, w) => `The terms e<sup>−Δ<sub>j</sub>/2</sup>: ${list}; sum ${s}; w = ${e} / ${s} = ${w}.`,
 ai: (s, b, s2, d, w, ln) => `${s}: the best AIC belongs to ${b}; the next, ${s2}, has ΔAIC = ${d} and weight ${w}. BIC penalty per parameter: ln n = ${ln}.`,
 da: (c, a, b, diff) => `The Normal model has Δ${c} between ${a} and ${b}. ` + (diff.length ? `AIC and BIC choose different models for: ${diff.join(', ')}.` : 'AIC and BIC choose the same model on every series.'),
 k51: (est, m, s) => est ? `μ and σ estimated from the five returns: μ̂ = ${m}, σ̂ = ${s} (denominator n); u<sub>(i)</sub> = Φ((r<sub>(i)</sub> − μ̂)/σ̂).` : `H<sub>0</sub>: N(0, ${s}²), fully specified; u<sub>(i)</sub> = Φ(r<sub>(i)</sub>/${s}).`,
 k52: u => `u<sub>(i)</sub>: ${u}.`,
 k53: (i, side, d) => `The largest distance is at i = ${i}: ` + (side ? `u<sub>(${i})</sub> − ${i - 1}/5` : `${i}/5 − u<sub>(${i})</sub>`) + ` = ${d}.`,
 k54: (d, c, est, rej) => `D<sub>5</sub> = ${d} ${rej ? '>' : '<'} ${c}, ` + (est ? 'the Lilliefors 5% critical value for n = 5 (simulation of the lecture, 4,000 samples)' : 'the exact 5% critical value for n = 5') + (rej ? ': reject.' : ': do not reject.'),
 k5: (d, c, est, rej) => `D<sub>5</sub> = ${d}; ${est ? 'Lilliefors' : 'exact'} 5% critical value: ${c}; ${rej ? 'H<sub>0</sub> is rejected' : 'H<sub>0</sub> is not rejected'}.`,
 kt1: (s, n, m, sd) => `${s}: n = ${n}; the fitted Normal distribution has μ̂ = ${m}%, σ̂ = ${sd}%.`,
 kt2: (i, x, u, a, b) => `The distance is largest at the ${i}-th smallest return, r<sub>(i)</sub> = ${x}%; u<sub>(i)</sub> = F(r<sub>(i)</sub>) = ${u}; i/n = ${a}; (i − 1)/n = ${b}.`,
 kt3: (p, m, d) => `i/n − u<sub>(i)</sub> = ${p}; u<sub>(i)</sub> − (i − 1)/n = ${m}; D<sub>n</sub> = ${d}.`,
 kt4: (n, c, d) => `Critical value for a known F: 1.358/√${n} = ${c}; D<sub>n</sub> = ${d} exceeds it. With estimated parameters the correct critical value is smaller still (the Lilliefors problem).`,
 kt: (s, x, f, d, xw, zw) => `${s}: the KS distance is largest at x = ${x}% (F = ${f}%), D<sub>n</sub> = ${d}; the Anderson–Darling weighted distance is largest at x = ${xw}% (value ${zw}).`,
 gp: (n, a, b, c, d, e, f) => `n = ${n}: Student-t data, power KS ${a}%, CvM ${b}%, AD ${c}%; differences only in the tails: KS ${d}%, CvM ${e}%, AD ${f}%. 1,000 simulated samples, simulated critical values for estimated parameters.`,
 ed: (s, n, c, diff) => `${s}, n = ${n}: the KS critical value for a known F would be 1.358/√n = ${c}; with estimated parameters it is smaller. ` + (diff.length ? `The ranking by AD differs from the ranking by AIC for: ${diff.join(', ')}.` : 'The ranking by AD equals the ranking by AIC.'),
 ed2: (rows, nv, B) => `Parametric bootstrap, B = ${B}: ` + rows.map(([m, k, a, p]) => `${m}: critical values KS ${k}, AD ${a}, bootstrap p ${p}`).join('; ') + `. Student-t, naive KS p-value (as if θ were known): ${nv}.`,
 pp: (s, rows) => `${s}, the largest vertical distance in the PP plot (over all days) and the KS statistic: ` + rows.map(([m, d, k]) => `${m} ${d} (KS ${k})`).join('; ') + '.',
 qq: (s, mn, d, n, p) => `${s}: smallest return ${mn}% (${d}), plotted at probability 0.5/n = ${p} (n = ${n}).`,
 qa: k => `The ${k}-th smallest return and the model quantiles at the same probability (k − 0.5)/n:`,
 qaItem: (a, d, r, n, t, s) => `${a}: ${d}, ${r}%; Normal ${n}%, Student-t ${t}%, skewed-t ${s}%`,
 veH: (l, v) => v ? `VaR ${l} (%) of the data and of each fitted model; the Normal mixture in italics` : `ES ${l} (%) of the data and of each fitted model`,
 ve1: (s, m, sd, a, z) => `${s}, the fitted Normal model: μ̂ = ${m}%, σ̂ = ${sd}%; z<sub>α</sub> for α = ${a}: ${z}.`,
 ve2: (m, z, sd, v, e) => `VaR = −(${m} + (${z}) × ${sd}) = ${v}%; empirical VaR (sample quantile) ${e}%.`,
 ve3: (a, z, f, m, sd, es, e) => `ES ${a}: z = ${z}, φ(z) = ${f}; ES = −${m} + ${sd} × ${f}/α = ${es}%; empirical ES (mean of the returns below the quantile, sign changed) ${e}%.`,
 ve: (a, e) => `VaR ${a} and ES ${e}: the model values come from the quantile functions of the models fitted in the lecture; 1% and 2.5% are the lecture's, the other levels are extra runs.`,
 vm: a => `Model VaR ${a} divided by the empirical VaR ${a}, the smallest and the largest ratio over the seven series (deviation in %):`,
 vmItem: (m, lo, hi) => `${m}: from ${lo}% to ${hi}%`,
 lt: (s, x, c, n, p) => `${s}: ${days(String(c))} of ${n} with a loss above ${x}% (share ${p}). The same probability under the models, and the number of days expected in the sample:`,
 ltItem: rows => rows.map(([m, p, e]) => `${m} ${p} (${e} days)`).join('; '),
 ku1: (m, s, a, v, n, x, r) => `${m}, ${s}: VaR ${a} = ${v}%; of n = ${n} days, x = ${x} have r<sub>t</sub> < −VaR (rate ${r}%).`,
 ku2: (nx, a, x, l0) => `ln L<sub>0</sub> = ${nx} × ln(1 − ${a}) + ${x} × ln ${a} = ${l0}.`,
 ku3: (r, l1) => `ln L<sub>1</sub>, with the observed rate x/n = ${r}: ${l1}.`,
 ku4: (l0, l1, lr, p, rej) => `LR<sub>uc</sub> = −2 × (${l0} − (${l1})) = ${lr}; p = P(χ²(1) > LR) ${p}: ${rej ? 'the number of exceedances is rejected at 5%' : 'the number of exceedances is not rejected at 5%'}.`,
 ku: (m, s, x, e, y, c) => `${m}, ${s}: ${x} exceedances against ${e} expected; the most in one year: ${c} in ${y}.`,
 oo1: (s, m, ne, v, nt, d) => `${s}, ${m}: VaR 1% estimated on ${ne} days to 31 Dec 2019, VaR = ${v}%, so q = −${v}%; ${nt} test days, from ${d}.`,
 oo2: (x, s1) => `The ${x} days with r<sub>t</sub> < q contribute (α − 1)(r<sub>t</sub> − q), in total ${s1}.`,
 oo3: (x, s0) => `The other ${x} days contribute α(r<sub>t</sub> − q), in total ${s0}.`,
 oo4: (a, b, n, p) => `Pinball loss = (${a} + ${b}) / ${n} = ${p}.`,
 oo: list => `The AIC winner of the estimation period also has the best log score for: ${list.join(', ') || 'no series'}.`,
 of1: (c, k, l, n) => `c = ${c} components: k = 3c − 1 = ${k} parameters; ℓ = ${l} on n = ${n} days (2017–2018).`,
 of2: (l, k, a) => `AIC = −2 × (${l}) + 2 × ${k} = ${a}.`,
 of3: (l, k, ln, b) => `BIC = −2 × (${l}) + ${k} × ${ln} = ${b}.`,
 of4: (i, o, d) => `Mean log-likelihood per day: ${i} in sample, ${o} out of sample (2019–2026); difference ${d}.`,
 of: (p, c, ln) => `With a penalty of ${p} per parameter, the criterion −2ℓ + penalty·k selects c = ${c}. AIC uses 2, BIC ln 501 = ${ln}; the best out of sample is c = 3.`,
 rv1: (s, m, n, a, b, c, d) => `${s}, ${m}: ${n} test days; pairs of consecutive days: n<sub>00</sub> = ${a}, n<sub>01</sub> = ${b}, n<sub>10</sub> = ${c}, n<sub>11</sub> = ${d} (1 = exceedance).`,
 rv2: (a, b, p01, c, d, p11, p) => `π̂<sub>01</sub> = ${a}/${b} = ${p01} (an exceedance after a day without); π̂<sub>11</sub> = ${c}/${d} = ${p11} (an exceedance after an exceedance); π̂ = ${p}.`,
 rv3: (l0, l1, lr, p, rej) => `ln L(π̂) = ${l0}, ln L(π̂<sub>01</sub>, π̂<sub>11</sub>) = ${l1}; LR<sub>ind</sub> = ${lr}, p ${p}: ${rej ? 'the exceedances come in clusters, independence is rejected at 5%' : 'independence is not rejected at 5%'}.`,
 rv: (s, d, n, f) => `${s}: each model is refitted every 20 days on the previous 1,000 days; backtest from ${d}, ${n} days. The chart shows the period from ${f.slice(0, 4)}; the table covers the whole period.`,
 vq: (n, d, h, l, hl) => `DAX, ${n} days from ${d}; volatility from the last ${h} days (RMA) or with weights λ<sup>j</sup>, λ = ${l} (EMA, half-life of the weight ${hl} days).`,
 mu: (b, w, full) => (full ? 'All 200 resamples of the lecture' : `The first ${b} resamples (the lecture uses 200)`) + `, moving blocks of 20 days: NIG has the lowest AIC in ${w}% of them.`,
 av1: (s, c, rows) => `${s}, ${c} weights: ` + rows.map(([m, w, v]) => `${m} w = ${w}, VaR ${v}%`).join('; ') + '.',
 av2: (t, a, l) => `Averaged VaR ${l} = ${t} = ${a}% (terms with weights below 0.0005 omitted).`,
 av3: (h, mh, l, ml, r) => `Heavy-tailed range: from ${l}% (${ml}) to ${h}% (${mh}); ${h}/${l} − 1 = ${r}%.`,
 av: (a, c, mix) => `VaR ${a}; ${c} weights; the range ${mix ? 'includes the Normal mixture, as in the lecture' : 'without the Normal mixture'}.`
};
})();
""",
}

QA = {
 'ro': {
  'candidates': ('La x = 4, distribuția Student-t (ν = 4) dă o probabilitate P(X < −4) de 76 de ori mai mare decît distribuția Normală. Butonul ▶ mută pragul de la 2 la 6. De cîte ori mai mare este probabilitatea Student-t la x = 6?',
                 'De circa 536 000 de ori: 5,3 × 10<sup>−4</sup> față de 9,9 × 10<sup>−10</sup>. La x = 2 cele două probabilități sînt aproape egale (raportul 1,0), la x = 3 raportul este 4,9, la x = 5 este 3 682. Coada Normală scade ca e<sup>−x²/2</sup>, coada Student-t ca o putere a lui x, deci raportul crește tot mai repede cu pragul. Media și varianța sînt aceleași; diferența stă numai în coadă, acolo unde se află VaR 1%.'),
  'dax-monthly': ('Boltirea randamentelor lunare DAX este 4,60 pe eșantionul SFESumm (iulie 2004 – mai 2014) și 5,06 pe 2004–2026. Butonul ▶ extinde eșantionul cu cîte un an. Crește boltirea treptat, pe măsură ce se adaugă ani?',
                  'Nu. Între 2014 și 2019 scade, pînă la 4,30 în mai 2016, pentru că lunile adăugate sînt liniștite. Urcă brusc la 4,93 cînd intră randamentul dintre începutul lui martie și începutul lui aprilie 2020, −21,70%, noul minim al eșantionului (cel vechi era −19,35%). Boltirea depinde de cîteva luni extreme, nu de numărul de luni. Pe 2004–2026 asimetria rămîne negativă (−0,72), iar testul Jarque–Bera respinge normalitatea cu JB = 70,2.'),
  'desc-stats': ('Fără cele două zile greșite din mai 2016, boltirea randamentelor zilnice TLV este 16,5. Cît devine dacă cele două zile, −15,0% și +20,5%, rămîn în date?',
                 '20,5. Boltirea folosește puterea a patra a abaterilor de la medie, deci două zile extreme o mută mult, chiar dacă suma lor, +5,5%, este mică. Asimetria trece de la −0,87 la −0,58. În curs, ν̂ al distribuției Student-t scade de la 3,44 la 3,33, iar clasamentul AIC nu se schimbă (cîștigă Student-t în ambele cazuri). JB al TLV crește și el, pentru că (K − 3)² crește.'),
  'ml-table': ('Pentru S&P 500, ν̂ al distribuției Student-t este 2,67. Bitcoin are o boltire de selecție mai mare (15,0 față de 13,7). Este ν̂ al Bitcoin mai mic sau mai mare?',
               'Mai mic: 2,23, cel mai mic dintre cele șapte serii, deci coada cea mai groasă. BET are 2,43, DAX 3,09, BRD 4,67. Un ν̂ apropiat de 2 înseamnă o varianță abia finită. Abaterea standard mare a Bitcoin (3,5% pe zi) intră în scală, nu în ν: scala Student-t a Bitcoin este 1,85, iar abaterea standard implicată, scală·√(ν/(ν − 2)), este 5,75%, peste abaterea standard de selecție (3,49%), pentru că ν̂ este aproape de 2.'),
  'profile-nu': ('Slide-ul dă intervalul de 95% pentru ν al S&P 500, [2,47; 2,89], unde profilul intersectează pragul. Ce puncte ale grilei cu pasul 0,1 folosite în grafic se află în interval?',
                 'De la 2,5 la 2,8. Capetele adevărate, 2,47 și 2,89, se află între punctele grilei (2,4–2,5 și 2,8–2,9); o grilă de 0,1 citită direct ar da [2,5; 2,8], un interval prea îngust. Intervalul nu este simetric în jurul lui ν̂ = 2,67: 0,20 la stînga, 0,22 la dreapta, pentru că profilul nu este o parabolă. La ν = 5, log-verosimilitatea profil este cu 92,3 sub maxim.'),
  'lr-test': ('Pentru BRD, testul Student-t în skewed-t (λ = 0) dă LR = 3,01. Este parametrul de asimetrie necesar la 5%?',
              'Nu: 3,01 < 3,84, p = 0,083. Valoarea fixată λ = 0 este în interiorul intervalului (−1, 1), deci teorema lui Wilks se aplică și valoarea critică este 3,84; 2,71 este valoarea pentru frontieră (Normală în Student-t). AIC preferă totuși skewed-t (ΔAIC al Student-t: 1,0), pentru că AIC nu este un test: penalizează un parametru cu 2, iar LR > 2. Asimetria este semnificativă doar pentru S&P 500 (LR = 19,63) și DAX (20,92).'),
  'vuong': ('Pe slide, Student-t este preferat distribuției stabile pe toate cele patru piețe. Pentru BRD, V = 1,89. Este preferința semnificativă la 5%?',
            'Nu: |V| = 1,89 < 1,96, p = 0,059. Student-t explică în medie mai bine zilele BRD, dar diferența nu este separată de zgomot. Ziua cu cea mai mare |m<sub>t</sub>| este 19 decembrie 2018 (m<sub>t</sub> = −2,48: distribuția stabilă explică mai bine această cădere). Pentru TLV și SNP, V = 3,77 și 4,16. Suma diferențelor zilnice este exact diferența log-verosimilităților, ℓ(Student-t) − ℓ(stabilă).'),
  'aic-table': ('Pentru BRD, AIC preferă skewed-t, iar Student-t are ΔAIC = 1,0. Ce ponderi Akaike primesc cele două modele?',
                '0,62 pentru skewed-t și 0,38 pentru Student-t (celelalte cinci modele au ponderi practic nule): w = e<sup>−Δ/2</sup>/Σ e<sup>−Δ<sub>j</sub>/2</sup>, cu e<sup>−0,5</sup> = 0,61. O diferență de 1 lasă ambele modele plauzibile; AIC nu este un test. Cu ponderile BIC, ordinea se inversează: 0,93 pentru Student-t.'),
  'delta-aic': ('Pentru BRD, AIC preferă skewed-t (ΔAIC al Student-t: 1,0). Butonul ▶ trece la BIC. Ce model preferă BIC?',
                'Student-t, cu ΔBIC = 5,2 pentru skewed-t. BIC penalizează parametrul suplimentar cu ln 3 771 = 8,24, mai mult decît cîștigul de ajustare (LR = 3,01). Pentru TLV și SNP, ambele criterii aleg Student-t; pentru BET, S&P 500 și DAX, NIG; pentru Bitcoin, GED. Modelul Normal are ΔBIC între 472 și 2 412.'),
  'ks-five': ('Cu H<sub>0</sub>: N(0, 1), D<sub>5</sub> = 0,218 < 0,563. Butonul ▶ mută σ din ipoteza nulă de la 0,5 la 2. Pentru ce σ din acest interval respinge testul la 5%?',
              'Pentru niciunul: D<sub>5</sub> este între 0,188 (σ = 1,4) și 0,364 (σ = 0,5), mereu sub 0,563. Cu cinci observații, testul nu separă o abatere standard de 0,5% de una de 2%: puterea lui este foarte mică. Cu μ și σ estimate din cele cinci randamente (Lilliefors), D<sub>5</sub> = 0,175, iar valoarea critică scade la 0,351 (simularea din curs, ca pe slide).'),
  'ks-tail': ('Pentru S&P 500, distanța KS dintre distribuția empirică și distribuția Normală estimată are maximul la x = −0,41%, aproape de centru. Unde este maximul pentru Bitcoin?',
              'Tot în corpul distribuției: la x = −1,69% (F = 30%), cu D<sub>n</sub> = 0,099. Pentru toate cele șapte serii, maximul KS este între x = −1,7% și +1,0%, unde funcțiile de repartiție au pante mari. Distanța ponderată Anderson–Darling are maximul în coadă: −6,9% pentru S&P 500, −20,1% pentru Bitcoin. KS măsoară forma corpului, AD forma cozilor.'),
  'gof-power': ('La n = 250, cînd abaterile de la distribuția Normală sînt doar în cozi, puterea este 38% (KS), 50% (CvM) și 57% (AD). Butonul ▶ crește n pînă la 1 000. Cît devine puterea testului KS?',
                '82,5%, sub cea a testului AD (95,6%) și CvM (92,0%). Cu date Student-t (ν = 5), la n = 1 000 toate trei resping aproape mereu (99,9%–100%). Testul AD pune o pondere mai mare pe cozi, acolo unde se află abaterea. La n = 100, KS respinge datele Student-t doar în 30% din eșantioane: un test KS pe 100 de randamente care nu respinge distribuția Normală nu spune mare lucru despre VaR 1%.'),
  'edf-table': ('Ordonează statistica AD modelele S&P 500 în aceeași ordine ca AIC?',
                'Nu exact. După AD: NIG (0,88), skewed-t (2,25), GED (3,94), stabilă (4,61), Student-t (4,69), amestec Normal (5,80), Normală (127,35). După AIC: NIG, skewed-t, Student-t, GED, stabilă, amestec Normal, Normală. Primele două și ultimele două locuri coincid; Student-t coboară de pe locul 3 pe locul 5. AD și AIC măsoară lucruri diferite: AD distanța ponderată dintre funcțiile de repartiție, AIC densitatea în punctele observate.'),
  'pp': ('În graficul PP al S&P 500, cea mai mare distanță pe verticală a modelului Normal este 0,093. Cît este pentru Student-t?',
         '0,018, practic statistica KS (0,018). Skewed-t are 0,017, distribuția stabilă 0,022: în graficul PP cele trei nu se pot deosebi, deși cozile lor diferă cu ordine de mărime. Graficul PP comprimă cozile în colțuri; graficul QQ de pe slide-ul următor le dilată.'),
  'qq': ('Pentru S&P 500, cuantila distribuției stabile la probabilitatea celui mai mic randament (−12,8%) este −116,1%. Pentru Bitcoin, cu cel mai mic randament −46,5% (12 martie 2020), cît dă distribuția stabilă?',
         '−385,6%, de peste 8 ori mai mult decît s-a observat. Student-t dă −85,0%, NIG −31,3%, GED −24,2%, modelul Normal −12,7%. La probabilitatea 0,5/n = 1,1 × 10<sup>−4</sup>, modelele diferă cel mai mult, iar datele conțin un singur punct: acolo incertitudinea este cea mai mare.'),
  'qq-assets': ('Pentru Bitcoin, cea mai mare scădere zilnică (12 martie 2020, −46,5%) are în modelul Student-t cuantila −85,0%. Butonul ▶ trece la al doilea, al treilea… cel mai mic randament. Ce cuantilă Student-t are al doilea cel mai mic randament, −23,8%?',
                '−51,9%, de peste două ori randamentul observat; la al zecelea (−15,5%), −22,5%. Coada Student-t estimată pe Bitcoin (ν̂ = 2,23) este mai lungă decît cea a datelor. Pentru TLV situația este inversă: cea mai mare scădere, −22,2%, are cuantila Student-t −17,8%. Modelul Normal rămîne mult sub date pentru toate cele trei serii (Bitcoin: −12,7%).'),
  'var-es': ('La VaR 1%, modelul Normal dă pentru S&P 500 2,80%, iar datele 3,45%. Butonul ▶ trece prin niveluri. La VaR 0,1%, cît dă modelul Normal și cît dau datele?',
             'Modelul Normal 3,72%, datele 7,22%: la 0,1% modelul Normal acoperă doar jumătate din pierdere. Student-t dă 8,58%, NIG 6,98%, distribuția stabilă 19,70%. La VaR 5%, ordinea se inversează: modelul Normal dă 1,97%, peste datele (1,85%). ES 2,5% al modelului Normal din formula φ(z)/α (2,81%) coincide cu valoarea obținută prin integrarea cuantilei.'),
  'var-models': ('La VaR 1%, raportul dintre VaR al modelului Normal și VaR empiric este între 0,76 și 0,86 pe cele șapte serii. Butonul ▶ trece prin niveluri. Ce devine raportul la VaR 5%?',
                 'Peste 1 pe toate seriile, între 1,03 (DAX) și 1,16 (BET): la 5% modelul Normal supraestimează pierderea. Aceeași abatere standard σ̂ este prea mare pentru corpul distribuției și prea mică pentru coada ei; cele două erori se compensează pe undeva între 2,5% și 5%. La 0,1% raportul scade la 0,45–0,65.'),
  'left-tail': ('S&P 500 are o singură zi cu o pierdere de peste 10% (16 martie 2020) în 6 714 zile. Cîte astfel de zile așteaptă în eșantion distribuția stabilă estimată?',
                'Circa 19. Modelul Normal așteaptă practic zero (probabilitatea 7 × 10<sup>−17</sup>), Student-t 4,5, NIG 1,0, GED 0,08. Pentru pierderi de peste 5%, datele au 23 de zile, stabila așteaptă 57, Student-t 27, NIG 26. Coada stabilă (α̂ = 1,54) este mult prea groasă pentru S&P 500.'),
  'kupiec': ('La VaR 1%, NIG (cîștigătorul AIC) are pentru S&P 500 52 de depășiri față de 67 așteptate (p = 0,053). Butonul ▶ trece prin niveluri. La VaR 2,5%, este numărul depășirilor NIG respins la 5%?',
             'Nu: 151 de depășiri față de 167,9 așteptate, p = 0,18. Student-t, perfect la 1% (67 față de 67), este respins la 2,5%: 211 depășiri, p = 0,001. Un model bun pentru o cuantilă nu este neapărat bun pentru alta. La VaR 0,1%, NIG are 7 depășiri față de 6,7 așteptate, iar modelul Normal 52.'),
  'oos': ('Pe perioada de test 2020–2026, care metodă are cea mai mică pierdere pinball a cuantilei de 1% pentru BET?',
          'Modelul Normal (0,0475), urmat de GED (0,0494); Student-t, cel mai bun după scorul logaritmic, are 0,0517, iar NIG, cîștigătorul AIC pe 2000–2019, 0,0513. Perioada de test a fost mai calmă pentru BET, iar VaR Normal, mai mic, a avut 17 depășiri față de 16,7 așteptate. Numărul potrivit vine dintr-un motiv greșit: un model static nu urmărește volatilitatea.'),
  'overfit': ('Cu penalizarea 2 pe parametru (AIC), criteriul alege c = 5 componente; cu ln 501 = 6,22 (BIC), c = 3. Butonul ▶ crește penalizarea de la 0 la 10. De la ce penalizare alegerea coboară la c = 3?',
              'De la 5,71 (pe pașii de 0,1 ai cursorului, de la 5,8): trecerea de la 3 la 4 componente aduce 2(ℓ<sub>4</sub> − ℓ<sub>3</sub>) = 17,14 pentru 3 parametri în plus, adică 5,71 pe parametru. Sub 0,8 se alege c = 6, între 0,8 și 3 c = 5, între 3 și 5,71 c = 4. În afara eșantionului cel mai bun este c = 3 (−1,608 pe zi), iar c = 6 dă −1,698.'),
  'rolling-var': ('S&P 500, VaR 1% Student-t pe fereastră mobilă: 89 de depășiri în 5 714 zile (1,56%), Kupiec p < 0,001. Sînt depășirile independente în timp? (testul Christoffersen, rulare suplimentară)',
                  'Nu: LR<sub>ind</sub> = 19,81, p < 0,001. După o zi cu depășire urmează o altă depășire în 10,1% din cazuri (9 din 89), față de 1,4% după o zi fără depășire. Depășirile se adună în august 2007, septembrie–noiembrie 2008 (6, 9 și 6) și martie 2020 (9). Pentru BET, simularea istorică trece testul Kupiec (p = 0,066), dar nu și testul de independență (p = 0,014).'),
  'var-qqplot': ('Cu λ = 0,96, VaR Normal cu volatilitatea EMA este depășit în 2,07% din zilele DAX. Butonul ▶ mută λ de la 0,90 la 0,99. Se apropie rata de 1% pentru vreun λ?',
                 'Nu: rata rămîne între 1,96% (λ = 0,97) și 2,40% (λ = 0,99). Volatilitatea EMA se adaptează, dar cuantila Normală z<sub>0,01</sub> = −2,326 rămîne prea mică pentru randamentele standardizate, care au cozi mai groase decît distribuția Normală: capătul drept al graficului QQ se curbează în sus pentru orice λ. RMA cu h = 250 are 2,13%; cu h = 125, 2,21%.'),
  'model-uncertainty': ('Cu toate cele 200 de reeșantionări, NIG are cel mai mic AIC în 93,5% dintre ele. Butonul ▶ crește B de la 20 la 200. Ce proporție dau primele 20 de reeșantionări?',
                        '100%: NIG cîștigă în toate primele 20. Cu B = 40, 95%; cu B = 100, 95%; cu B = 200, 93,5% (GED 3,5%, skewed-t 3%). Un număr mic de reeșantionări supraestimează siguranța cîștigătorului. Intervalele de 90% ale VaR 1% (Student-t [3,14; 3,87]%, NIG [3,34; 4,18]%) se suprapun: incertitudinea dintre modele este comparabilă cu cea din interiorul unui model.'),
  'model-avg': ('Pentru TLV, intervalul VaR 1% al modelelor cu cozi groase este [4,49; 5,41]. Distribuția stabilă, cu cozile cele mai groase, dă 4,92%. Ce model dă capătul de sus, 5,41%?',
                'Amestecul Normal (coloana „amestec” din tabelul VaR al cursului), unul dintre cele șase modele nenormale. Fără el, intervalul TLV devine [4,49; 4,92], adică 10% în loc de 21%. La fel pentru SNP (4,70%) și BRD (4,74%). Pentru indici și Bitcoin, capătul de sus vine de la distribuția stabilă.'),
 },
 'en': {
  'candidates': ('At x = 4, the Student-t distribution (ν = 4) gives a probability P(X < −4) 76 times that of the Normal distribution. The ▶ button moves the threshold from 2 to 6. How many times larger is the Student-t probability at x = 6?',
                 'About 536,000 times: 5.3 × 10<sup>−4</sup> against 9.9 × 10<sup>−10</sup>. At x = 2 the two probabilities are almost equal (ratio 1.0), at x = 3 the ratio is 4.9, at x = 5 it is 3,682. The Normal tail falls as e<sup>−x²/2</sup>, the Student-t tail as a power of x, so the ratio grows ever faster with the threshold. The mean and the variance are the same; the difference lies only in the tail, where VaR 1% sits.'),
  'dax-monthly': ('The kurtosis of the DAX monthly returns is 4.60 on the SFESumm sample (July 2004 – May 2014) and 5.06 on 2004–2026. The ▶ button extends the sample one year at a time. Does the kurtosis rise gradually as years are added?',
                  'No. Between 2014 and 2019 it falls, to 4.30 in May 2016, because the months added are calm. It jumps to 4.93 when the return from early March to early April 2020 enters, −21.70%, the new minimum of the sample (the old one was −19.35%). The kurtosis depends on a few extreme months, not on the number of months. On 2004–2026 the skewness stays negative (−0.72), and the Jarque–Bera test rejects Normality with JB = 70.2.'),
  'desc-stats': ('Without the two erroneous days of May 2016, the kurtosis of the TLV daily returns is 16.5. What does it become if the two days, −15.0% and +20.5%, stay in the data?',
                 '20.5. Kurtosis uses the fourth power of the deviations from the mean, so two extreme days move it a lot, even though their sum, +5.5%, is small. The skewness moves from −0.87 to −0.58. In the lecture, ν̂ of the Student-t falls from 3.44 to 3.33, and the AIC ranking does not change (Student-t wins in both cases). The JB of TLV rises too, because (K − 3)² rises.'),
  'ml-table': ('For the S&P 500, ν̂ of the Student-t distribution is 2.67. Bitcoin has a larger sample kurtosis (15.0 against 13.7). Is the ν̂ of Bitcoin smaller or larger?',
               'Smaller: 2.23, the smallest of the seven series, hence the heaviest tail. BET has 2.43, DAX 3.09, BRD 4.67. A ν̂ close to 2 means a variance that is barely finite. The large standard deviation of Bitcoin (3.5% per day) goes into the scale, not into ν: the Student-t scale of Bitcoin is 1.85, and the implied standard deviation, scale·√(ν/(ν − 2)), is 5.75%, above the sample standard deviation (3.49%), because ν̂ is close to 2.'),
  'profile-nu': ('The slide gives the 95% interval for ν of the S&P 500, [2.47, 2.89], where the profile crosses the cut-off. Which points of the grid of step 0.1 used in the chart lie inside the interval?',
                 'From 2.5 to 2.8. The true end points, 2.47 and 2.89, lie between grid points (2.4–2.5 and 2.8–2.9); a 0.1 grid read directly would give [2.5, 2.8], an interval that is too narrow. The interval is not symmetric around ν̂ = 2.67: 0.20 to the left, 0.22 to the right, because the profile is not a parabola. At ν = 5, the profile log-likelihood is 92.3 below its maximum.'),
  'lr-test': ('For BRD, the test of Student-t inside skewed-t (λ = 0) gives LR = 3.01. Is the skewness parameter needed at 5%?',
              'No: 3.01 < 3.84, p = 0.083. The fixed value λ = 0 lies inside the interval (−1, 1), so Wilks\' theorem applies and the critical value is 3.84; 2.71 is the boundary value (Normal inside Student-t). AIC still prefers skewed-t (ΔAIC of Student-t: 1.0), because AIC is not a test: it charges 2 for a parameter, and LR > 2. The skewness is significant only for the S&P 500 (LR = 19.63) and the DAX (20.92).'),
  'vuong': ('On the slide, Student-t is preferred to the stable distribution on all four markets. For BRD, V = 1.89. Is the preference significant at 5%?',
            'No: |V| = 1.89 < 1.96, p = 0.059. Student-t explains the BRD days better on average, but the difference is not separated from noise. The day with the largest |m<sub>t</sub>| is 19 December 2018 (m<sub>t</sub> = −2.48: the stable distribution explains this fall better). For TLV and SNP, V = 3.77 and 4.16. The sum of the daily differences is exactly the difference of the log-likelihoods, ℓ(Student-t) − ℓ(stable).'),
  'aic-table': ('For BRD, AIC prefers skewed-t, and Student-t has ΔAIC = 1.0. What Akaike weights do the two models get?',
                '0.62 for skewed-t and 0.38 for Student-t (the other five models have weights of practically zero): w = e<sup>−Δ/2</sup>/Σ e<sup>−Δ<sub>j</sub>/2</sup>, with e<sup>−0.5</sup> = 0.61. A difference of 1 leaves both models plausible; AIC is not a test. With BIC weights the order reverses: 0.93 for Student-t.'),
  'delta-aic': ('For BRD, AIC prefers skewed-t (ΔAIC of Student-t: 1.0). The ▶ button switches to BIC. Which model does BIC prefer?',
                'Student-t, with ΔBIC = 5.2 for skewed-t. BIC charges the extra parameter ln 3,771 = 8.24, more than the gain in fit (LR = 3.01). For TLV and SNP, both criteria choose Student-t; for BET, S&P 500 and DAX, NIG; for Bitcoin, GED. The Normal model has ΔBIC between 472 and 2,412.'),
  'ks-five': ('With H<sub>0</sub>: N(0, 1), D<sub>5</sub> = 0.218 < 0.563. The ▶ button moves σ of the null from 0.5 to 2. For which σ in this range does the test reject at 5%?',
              'None: D<sub>5</sub> lies between 0.188 (σ = 1.4) and 0.364 (σ = 0.5), always below 0.563. With five observations, the test cannot separate a standard deviation of 0.5% from one of 2%: its power is very low. With μ and σ estimated from the five returns (Lilliefors), D<sub>5</sub> = 0.175, and the critical value falls to 0.351 (the lecture\'s simulation, as on the slide).'),
  'ks-tail': ('For the S&P 500, the KS distance between the empirical distribution and the fitted Normal distribution is largest at x = −0.41%, close to the centre. Where is the maximum for Bitcoin?',
              'Also in the body of the distribution: at x = −1.69% (F = 30%), with D<sub>n</sub> = 0.099. For all seven series, the KS maximum lies between x = −1.7% and +1.0%, where the distribution functions are steep. The Anderson–Darling weighted distance is largest in the tail: −6.9% for the S&P 500, −20.1% for Bitcoin. KS measures the shape of the body, AD the shape of the tails.'),
  'gof-power': ('At n = 250, when the departures from the Normal distribution are only in the tails, the power is 38% (KS), 50% (CvM) and 57% (AD). The ▶ button raises n to 1,000. What does the power of the KS test become?',
                '82.5%, below that of AD (95.6%) and CvM (92.0%). With Student-t data (ν = 5), at n = 1,000 all three reject almost always (99.9%–100%). The AD test puts more weight on the tails, where the departure is. At n = 100, KS rejects Student-t data in only 30% of the samples: a KS test on 100 returns that does not reject the Normal distribution says little about VaR 1%.'),
  'edf-table': ('Does the AD statistic rank the S&P 500 models in the same order as AIC?',
                'Not exactly. By AD: NIG (0.88), skewed-t (2.25), GED (3.94), stable (4.61), Student-t (4.69), Normal mixture (5.80), Normal (127.35). By AIC: NIG, skewed-t, Student-t, GED, stable, Normal mixture, Normal. The first two and the last two places agree; Student-t drops from 3rd to 5th. AD and AIC measure different things: AD the weighted distance between distribution functions, AIC the density at the observed points.'),
  'pp': ('In the PP plot of the S&P 500, the largest vertical distance of the Normal model is 0.093. What is it for the Student-t?',
         '0.018, practically the KS statistic (0.018). Skewed-t has 0.017, the stable distribution 0.022: in the PP plot the three cannot be told apart, although their tails differ by orders of magnitude. The PP plot squeezes the tails into the corners; the QQ plot on the next slide stretches them.'),
  'qq': ('For the S&P 500, the quantile of the stable distribution at the probability of the smallest return (−12.8%) is −116.1%. For Bitcoin, with the smallest return −46.5% (12 March 2020), what does the stable distribution give?',
         '−385.6%, more than 8 times the observed value. Student-t gives −85.0%, NIG −31.3%, GED −24.2%, the Normal model −12.7%. At the probability 0.5/n = 1.1 × 10<sup>−4</sup>, the models differ most and the data hold a single point: the uncertainty is largest there.'),
  'qq-assets': ('For Bitcoin, the largest daily fall (12 March 2020, −46.5%) has the Student-t quantile −85.0%. The ▶ button moves to the second, third… smallest return. What Student-t quantile does the second smallest return, −23.8%, get?',
                '−51.9%, more than twice the observed return; at the tenth (−15.5%), −22.5%. The Student-t tail fitted to Bitcoin (ν̂ = 2.23) is longer than that of the data. For TLV it is the other way round: the largest fall, −22.2%, has the Student-t quantile −17.8%. The Normal model stays far inside the data for all three series (Bitcoin: −12.7%).'),
  'var-es': ('At VaR 1%, the Normal model gives 2.80% for the S&P 500 and the data 3.45%. The ▶ button goes through the levels. At VaR 0.1%, what do the Normal model and the data give?',
             'The Normal model 3.72%, the data 7.22%: at 0.1% the Normal model covers only half of the loss. Student-t gives 8.58%, NIG 6.98%, the stable distribution 19.70%. At VaR 5% the order reverses: the Normal model gives 1.97%, above the data (1.85%). The Normal ES 2.5% from the formula φ(z)/α (2.81%) equals the value obtained by integrating the quantile.'),
  'var-models': ('At VaR 1%, the ratio of the Normal VaR to the empirical VaR lies between 0.76 and 0.86 on the seven series. The ▶ button goes through the levels. What does the ratio become at VaR 5%?',
                 'Above 1 on every series, between 1.03 (DAX) and 1.16 (BET): at 5% the Normal model overstates the loss. The same standard deviation σ̂ is too large for the body of the distribution and too small for its tail; the two errors cancel somewhere between 2.5% and 5%. At 0.1% the ratio falls to 0.45–0.65.'),
  'left-tail': ('The S&P 500 has a single day with a loss above 10% (16 March 2020) in 6,714 days. How many such days does the fitted stable distribution expect in the sample?',
                'About 19. The Normal model expects practically zero (probability 7 × 10<sup>−17</sup>), Student-t 4.5, NIG 1.0, GED 0.08. For losses above 5%, the data have 23 days, the stable expects 57, Student-t 27, NIG 26. The stable tail (α̂ = 1.54) is far too heavy for the S&P 500.'),
  'kupiec': ('At VaR 1%, NIG (the AIC winner) has 52 exceedances for the S&P 500 against 67 expected (p = 0.053). The ▶ button goes through the levels. At VaR 2.5%, is the number of NIG exceedances rejected at 5%?',
             'No: 151 exceedances against 167.9 expected, p = 0.18. Student-t, perfect at 1% (67 against 67), is rejected at 2.5%: 211 exceedances, p = 0.001. A model that is good for one quantile is not necessarily good for another. At VaR 0.1%, NIG has 7 exceedances against 6.7 expected, the Normal model 52.'),
  'oos': ('On the test period 2020–2026, which method has the lowest pinball loss of the 1% quantile for BET?',
          'The Normal model (0.0475), followed by GED (0.0494); Student-t, the best by log score, has 0.0517, and NIG, the AIC winner on 2000–2019, 0.0513. The test period was calmer for BET, and the smaller Normal VaR had 17 exceedances against 16.7 expected. The right number for the wrong reason: a static model does not follow the volatility.'),
  'overfit': ('With a penalty of 2 per parameter (AIC), the criterion selects c = 5 components; with ln 501 = 6.22 (BIC), c = 3. The ▶ button raises the penalty from 0 to 10. From which penalty does the choice fall to c = 3?',
              'From 5.71 (in the slider\'s steps of 0.1, from 5.8): going from 3 to 4 components gains 2(ℓ<sub>4</sub> − ℓ<sub>3</sub>) = 17.14 for 3 extra parameters, i.e. 5.71 per parameter. Below 0.8, c = 6 is selected, between 0.8 and 3 c = 5, between 3 and 5.71 c = 4. Out of sample the best is c = 3 (−1.608 per day), while c = 6 gives −1.698.'),
  'rolling-var': ('S&P 500, rolling Student-t VaR 1%: 89 exceedances in 5,714 days (1.56%), Kupiec p < 0.001. Are the exceedances independent over time? (the Christoffersen test, extra run)',
                  'No: LR<sub>ind</sub> = 19.81, p < 0.001. After a day with an exceedance, another follows in 10.1% of the cases (9 of 89), against 1.4% after a day without. The exceedances pile up in August 2007, September–November 2008 (6, 9 and 6) and March 2020 (9). For BET, historical simulation passes the Kupiec test (p = 0.066) but not the independence test (p = 0.014).'),
  'var-qqplot': ('With λ = 0.96, the Normal VaR with EMA volatility is exceeded on 2.07% of the DAX days. The ▶ button moves λ from 0.90 to 0.99. Does the rate approach 1% for any λ?',
                 'No: the rate stays between 1.96% (λ = 0.97) and 2.40% (λ = 0.99). The EMA volatility adapts, but the Normal quantile z<sub>0.01</sub> = −2.326 remains too small for the standardised returns, whose tails are heavier than those of the Normal distribution: the right end of the QQ plot bends upwards for every λ. RMA with h = 250 has 2.13%; with h = 125, 2.21%.'),
  'model-uncertainty': ('With all 200 resamples, NIG has the lowest AIC in 93.5% of them. The ▶ button raises B from 20 to 200. What share do the first 20 resamples give?',
                        '100%: NIG wins all of the first 20. With B = 40, 95%; with B = 100, 95%; with B = 200, 93.5% (GED 3.5%, skewed-t 3%). A small number of resamples overstates the certainty about the winner. The 90% intervals of VaR 1% (Student-t [3.14, 3.87]%, NIG [3.34, 4.18]%) overlap: the uncertainty between models is comparable to the uncertainty within a model.'),
  'model-avg': ('For TLV, the VaR 1% range of the heavy-tailed models is [4.49, 5.41]. The stable distribution, with the heaviest tails, gives 4.92%. Which model gives the upper end, 5.41%?',
                'The Normal mixture (the mixture column of the lecture\'s VaR table), one of the six non-Normal models. Without it, the TLV range becomes [4.49, 4.92], 10% instead of 21%. The same holds for SNP (4.70%) and BRD (4.74%). For the indices and Bitcoin, the upper end comes from the stable distribution.'),
 },
}

# Predict, then check: (key, [(RO, EN), ...]); the keys agree with what the page computes (node script on ch6_data.js + common.js + ch6.js).
PREDICT = {
 'candidates': (0, [('Circa 5 × 10⁵: coada Normală scade ca e^(−x²/2), raportul crește tot mai repede.', 'About 5 × 10⁵: the Normal tail falls as e^(−x²/2), the ratio grows ever faster.'),
                    ('Circa 110: raportul crește proporțional cu pragul.', 'About 110: the ratio grows in proportion to the threshold.'),
                    ('Tot circa 76: varianța este aceeași, deci raportul nu se schimbă.', 'Still about 76: the variance is the same, so the ratio does not change.')]),
 'dax-monthly': (1, [('Da: fiecare an adaugă luni de criză, deci boltirea crește treptat.', 'Yes: every year adds crisis months, so the kurtosis rises gradually.'),
                     ('Nu: scade spre 4,3 pînă în 2019 și sare la 4,9 cînd intră luna crahului din 2020.', 'No: it falls towards 4.3 until 2019 and jumps to 4.9 when the 2020 crash month enters.'),
                     ('Nu: scade spre 3, pentru că mai multe date apropie boltirea de cea a distribuției Normale.', 'No: it falls towards 3, because more data bring the kurtosis closer to that of the Normal distribution.')]),
 'desc-stats': (0, [('Crește la 20,5: puterea a patra amplifică cele două zile extreme.', 'It rises to 20.5: the fourth power amplifies the two extreme days.'),
                    ('Rămîne circa 16,5: cele două zile se compensează (suma lor este +5,5%).', 'It stays about 16.5: the two days offset each other (their sum is +5.5%).'),
                    ('Scade: două abateri de semne opuse fac distribuția mai simetrică.', 'It falls: two deviations of opposite signs make the distribution more symmetric.')]),
 'ml-table': (0, [('Mai mic, circa 2,2: coada Bitcoin este mai groasă.', 'Smaller, about 2.2: the Bitcoin tail is heavier.'),
                  ('Mai mare, circa 4: abaterea standard a Bitcoin este de trei ori mai mare.', 'Larger, about 4: the standard deviation of Bitcoin is three times larger.'),
                  ('Tot 2,67: ν este o proprietate a modelului, nu a seriei.', 'Still 2.67: ν is a property of the model, not of the series.')]),
 'profile-nu': (1, [('De la 2,4 la 2,9: capetele se rotunjesc la grila cea mai apropiată.', 'From 2.4 to 2.9: the end points round to the nearest grid point.'),
                    ('De la 2,5 la 2,8: capetele adevărate sînt între punctele grilei.', 'From 2.5 to 2.8: the true end points lie between the grid points.'),
                    ('Doar 2,6 și 2,7, cele mai apropiate de ν̂ = 2,67.', 'Only 2.6 and 2.7, the closest to ν̂ = 2.67.')]),
 'lr-test': (2, [('Da: LR = 3,01 > 2,71.', 'Yes: LR = 3.01 > 2.71.'),
                 ('Da: AIC preferă skewed-t.', 'Yes: AIC prefers skewed-t.'),
                 ('Nu: LR = 3,01 < 3,84, p = 0,083.', 'No: LR = 3.01 < 3.84, p = 0.083.')]),
 'vuong': (1, [('Da: V > 0 înseamnă că Student-t este preferat.', 'Yes: V > 0 means that Student-t is preferred.'),
               ('Nu: |V| = 1,89 < 1,96, p = 0,059.', 'No: |V| = 1.89 < 1.96, p = 0.059.'),
               ('Da: Student-t cîștigă pe toate celelalte piețe.', 'Yes: Student-t wins on all the other markets.')]),
 'aic-table': (2, [('1 și 0: cîștigătorul AIC ia toată ponderea.', '1 and 0: the AIC winner takes all the weight.'),
                   ('0,5 și 0,5: o diferență sub 2 înseamnă modele echivalente.', '0.5 and 0.5: a difference below 2 means equivalent models.'),
                   ('Circa 0,62 și 0,38: e^(−1/2) = 0,61 față de 1.', 'About 0.62 and 0.38: e^(−1/2) = 0.61 against 1.')]),
 'delta-aic': (1, [('Tot skewed-t: ambele criterii aleg același model.', 'Still skewed-t: both criteria choose the same model.'),
                   ('Student-t: penalizarea ln n ≈ 8,2 depășește cîștigul de ajustare.', 'Student-t: the penalty ln n ≈ 8.2 exceeds the gain in fit.'),
                   ('NIG, cîștigătorul indicilor.', 'NIG, the winner of the indices.')]),
 'ks-five': (2, [('Pentru σ = 0,5 și σ = 2: o abatere standard greșită de două ori este detectată.', 'For σ = 0.5 and σ = 2: a standard deviation wrong by a factor of two is detected.'),
                 ('Pentru orice σ ≠ 1: testul detectează orice abatere de la ipoteza nulă.', 'For every σ ≠ 1: the test detects any departure from the null.'),
                 ('Pentru niciunul: D₅ rămîne sub 0,563 (cel mult 0,36).', 'For none: D₅ stays below 0.563 (at most 0.36).')]),
 'ks-tail': (0, [('În corpul distribuției, la circa −1,7%.', 'In the body of the distribution, at about −1.7%.'),
                 ('În coada stîngă, la cea mai mare scădere (−46,5%).', 'In the left tail, at the largest fall (−46.5%).'),
                 ('Exact în 0, unde se află mediana.', 'Exactly at 0, where the median is.')]),
 'gof-power': (1, [('Circa 100%, ca pentru datele Student-t.', 'About 100%, as for the Student-t data.'),
                   ('Circa 83%, sub cea a testului AD (96%).', 'About 83%, below that of AD (96%).'),
                   ('Tot circa 38%: puterea nu depinde de n.', 'Still about 38%: the power does not depend on n.')]),
 'edf-table': (1, [('Da, exact aceeași ordine.', 'Yes, exactly the same order.'),
                   ('Primul și ultimul coincid (NIG, Normală), mijlocul diferă.', 'The first and the last agree (NIG, Normal), the middle differs.'),
                   ('Nu: AD ordonează modelele invers față de AIC.', 'No: AD ranks the models in the reverse order of AIC.')]),
 'pp': (0, [('Circa 0,018, aproape cît statistica KS.', 'About 0.018, close to the KS statistic.'),
            ('Tot circa 0,09: toate modelele sînt respinse.', 'Still about 0.09: all models are rejected.'),
            ('0: punctele stau pe diagonală.', '0: the points lie on the diagonal.')]),
 'qq': (2, [('Circa −47%: coada cea mai groasă reproduce cele mai mari scăderi.', 'About −47%: the heaviest tail reproduces the largest falls.'),
            ('Circa −13%, ca modelul Normal.', 'About −13%, as the Normal model.'),
            ('Circa −386%, de peste 8 ori valoarea observată.', 'About −386%, more than 8 times the observed value.')]),
 'qq-assets': (1, [('Circa −24%: celelalte puncte stau pe bisectoare.', 'About −24%: the other points lie on the diagonal.'),
                   ('Circa −52%, tot de peste două ori randamentul observat.', 'About −52%, still more than twice the observed return.'),
                   ('Tot −85%: cuantila nu depinde de rang.', 'Still −85%: the quantile does not depend on the rank.')]),
 'var-es': (0, [('Normal circa 3,7%, datele circa 7,2%.', 'Normal about 3.7%, the data about 7.2%.'),
                ('Ambele circa 5%: în coada îndepărtată modelele converg.', 'Both about 5%: far in the tail the models converge.'),
                ('Normal circa 28%: VaR crește de 10 ori cînd α scade de 10 ori.', 'Normal about 28%: VaR grows 10 times when α falls 10 times.')]),
 'var-models': (2, [('Tot între 0,76 și 0,86: modelul Normal subestimează la orice nivel.', 'Still between 0.76 and 0.86: the Normal model understates at every level.'),
                    ('Mai mic, circa 0,6: eroarea crește cu nivelul.', 'Smaller, about 0.6: the error grows with the level.'),
                    ('Peste 1, între 1,03 și 1,16: la 5% modelul Normal supraestimează.', 'Above 1, between 1.03 and 1.16: at 5% the Normal model overstates.')]),
 'left-tail': (0, [('Circa 19.', 'About 19.'), ('Circa 1, cît s-a observat.', 'About 1, as observed.'), ('Practic zero, ca modelul Normal.', 'Practically zero, as the Normal model.')]),
 'kupiec': (1, [('Da: NIG are prea puține depășiri la orice nivel.', 'Yes: NIG has too few exceedances at every level.'),
                ('Nu: 151 față de 168 așteptate, p = 0,18.', 'No: 151 against 168 expected, p = 0.18.'),
                ('Nu: cîștigătorul AIC are exact numărul așteptat.', 'No: the AIC winner has exactly the expected number.')]),
 'oos': (0, [('Modelul Normal.', 'The Normal model.'), ('Student-t, cel mai bun după scorul logaritmic.', 'Student-t, the best by log score.'), ('NIG, cîștigătorul AIC pe 2000–2019.', 'NIG, the AIC winner on 2000–2019.')]),
 'overfit': (2, [('De la orice penalizare peste 2.', 'From any penalty above 2.'), ('Doar de la ln 501 = 6,22.', 'Only from ln 501 = 6.22.'), ('De la circa 5,7.', 'From about 5.7.')]),
 'rolling-var': (0, [('Nu: după o depășire urmează alta în 10% din cazuri, față de 1,4%.', 'No: after an exceedance another follows in 10% of the cases, against 1.4%.'),
                     ('Da: rata de 1,56% este aproape de 1%.', 'Yes: the rate of 1.56% is close to 1%.'),
                     ('Nu se poate spune: testul Kupiec respinge deja modelul.', 'Cannot tell: the Kupiec test already rejects the model.')]),
 'var-qqplot': (1, [('Da, la λ = 0,99: o volatilitate mai netedă este mai stabilă.', 'Yes, at λ = 0.99: a smoother volatility is more stable.'),
                    ('Nu: rămîne între circa 2% și 2,4%.', 'No: it stays between about 2% and 2.4%.'),
                    ('Da, la λ = 0,90: o volatilitate care reacționează repede prinde crizele.', 'Yes, at λ = 0.90: a volatility that reacts fast catches the crises.')]),
 'model-uncertainty': (2, [('Tot circa 94%.', 'Still about 94%.'), ('Circa 50%: cu puține reeșantionări, cîștigătorul este aleator.', 'About 50%: with few resamples, the winner is random.'),
                           ('100%: NIG cîștigă în toate primele 20.', '100%: NIG wins all of the first 20.')]),
 'model-avg': (1, [('Distribuția stabilă, care are cozile cele mai groase.', 'The stable distribution, which has the heaviest tails.'),
                   ('Amestecul Normal.', 'The Normal mixture.'),
                   ('Student-t, cîștigătorul AIC al TLV.', 'Student-t, the AIC winner of TLV.')]),
}

# blocks whose initial chart or table shows the answer: veil the readout and those elements
VEIL = {'lr-test': {'out': True, 'charts': ['lr-tab', 'lr-chart']}, 'vuong': {'out': True, 'charts': ['vu-tab']}, 'delta-aic': {'out': True, 'charts': ['da-tab']},
        'gof-power': {'out': True, 'charts': ['gp-chart', 'gp-tab']}, 'edf-table': {'out': True, 'charts': ['ed-tab']}, 'pp': {'out': True, 'charts': ['pp-chart']},
        'qq': {'out': True, 'charts': ['qq-tab']}, 'var-es': {'out': True, 'charts': ['ve-tab']}, 'kupiec': {'out': True, 'charts': ['ku-tab']},
        'oos': {'out': True, 'charts': ['oo-tab']}, 'rolling-var': {'out': True, 'charts': ['rv-tab', 'rv-we-box']}, 'model-avg': {'out': True, 'charts': ['av-tab', 'av-we-box']},
        'ml-table': {'out': True}, 'profile-nu': {'out': True}}

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Modelul A are ℓ = −1 000 cu k = 3, modelul B are ℓ = −998 cu k = 5. Care are AIC mai mic?', 'Model A has ℓ = −1,000 with k = 3, model B has ℓ = −998 with k = 5. Which has the lower AIC?'),
      key=2, opts=[
  (('A.', 'A.'), ('A are 2 006, dar și B are 2 006: −2 × (−998) + 2 × 5 = 2 006.', 'A has 2,006, but B also has 2,006: −2 × (−998) + 2 × 5 = 2,006.')),
  (('B, pentru că are log-verosimilitatea mai mare.', 'B, because it has the higher log-likelihood.'), ('Log-verosimilitatea mai mare este compensată exact de penalizarea celor doi parametri în plus.', 'The higher log-likelihood is offset exactly by the penalty of the two extra parameters.')),
  (('Niciunul: AIC este 2 006 pentru ambele.', 'Neither: AIC is 2,006 for both.'), ('A: 2 000 + 6; B: 1 996 + 10. Δ = 0: cîștigul de ajustare este egal cu penalizarea.', 'A: 2,000 + 6; B: 1,996 + 10. Δ = 0: the gain in fit equals the penalty.'))]),
 dict(q=('A este imbricat în B (aceleași cifre). Este B semnificativ mai bun la 5%?', 'A is nested in B (the same numbers). Is B significantly better at 5%?'),
      key=1, opts=[
  (('Da: LR = 4 > 3,84.', 'Yes: LR = 4 > 3.84.'), ('3,84 este valoarea critică pentru o restricție; aici sînt d = k₁ − k₀ = 2.', '3.84 is the critical value for one restriction; here there are d = k₁ − k₀ = 2.')),
  (('Nu: LR = 4 < χ²₀,₉₅(2) = 5,99.', 'No: LR = 4 < χ²₀.₉₅(2) = 5.99.'), ('LR = 2(ℓ₁ − ℓ₀) = 2 × 2 = 4, comparat cu χ²(2): nu respingem modelul A.', 'LR = 2(ℓ₁ − ℓ₀) = 2 × 2 = 4, compared with χ²(2): we do not reject model A.')),
  (('Nu se poate testa: AIC este egal.', 'It cannot be tested: AIC is equal.'), ('Testul LR se aplică modelelor imbricate indiferent de AIC; AIC nu este un test.', 'The LR test applies to nested models whatever the AIC; AIC is not a test.'))]),
 dict(q=('De ce un test KS cu parametri estimați nu poate folosi tabelul KS clasic?', 'Why can a KS test with estimated parameters not use the classical KS table?'),
      key=0, opts=[
  (('Estimarea apropie F de date, deci D_n este prea mic.', 'Estimation pulls F towards the data, so D_n is too small.'), ('Valorile critice clasice sînt prea mari; se folosesc Lilliefors sau un bootstrap parametric (n = 1 000: 0,029 în loc de 0,043).', 'The classical critical values are too large; use Lilliefors or a parametric bootstrap (n = 1,000: 0.029 instead of 0.043).')),
  (('Estimarea face D_n prea mare.', 'Estimation makes D_n too large.'), ('Este invers: parametrii estimați pe aceleași date micșorează distanța.', 'The reverse: parameters estimated on the same data shrink the distance.')),
  (('Tabelul clasic este valabil doar pentru n mic.', 'The classical table holds only for small n.'), ('Problema nu este n, ci faptul că F nu mai este complet specificată.', 'The problem is not n but that F is no longer fully specified.'))]),
 dict(q=('Un model VaR 1% are 20 de depășiri în 1 000 de zile. Ce spune testul Kupiec la 5%?', 'A VaR 1% model has 20 exceedances in 1,000 days. What does the Kupiec test say at 5%?'),
      key=1, opts=[
  (('Nu respinge: 20 este aproape de 10.', 'No rejection: 20 is close to 10.'), ('LR_uc = 7,83 > 3,84 (p = 0,005): dublul ratei țintă este respins.', 'LR_uc = 7.83 > 3.84 (p = 0.005): twice the target rate is rejected.')),
  (('Respinge: LR_uc = 7,83 > 3,84.', 'Rejection: LR_uc = 7.83 > 3.84.'), ('ln L₀ = 980 ln 0,99 + 20 ln 0,01 = −101,95; ln L₁ = 980 ln 0,98 + 20 ln 0,02 = −98,04; LR = 2 × 3,91 = 7,83.', 'ln L₀ = 980 ln 0.99 + 20 ln 0.01 = −101.95; ln L₁ = 980 ln 0.98 + 20 ln 0.02 = −98.04; LR = 2 × 3.91 = 7.83.')),
  (('Respinge: LR_uc = 10, diferența dintre 20 și 10.', 'Rejection: LR_uc = 10, the difference between 20 and 10.'), ('LR_uc nu este diferența numărului de depășiri, ci −2 ln al raportului verosimilităților: 7,83.', 'LR_uc is not the difference in counts but −2 ln of the likelihood ratio: 7.83.'))]),
 dict(q=('Două modele au ΔAIC = 0 și ΔAIC = 2. Ce ponderi Akaike au?', 'Two models have ΔAIC = 0 and ΔAIC = 2. What are their Akaike weights?'),
      key=2, opts=[
  (('1 și 0.', '1 and 0.'), ('Ponderile nu desemnează un singur cîștigător: e^(−1) = 0,37 nu este zero.', 'The weights do not name a single winner: e^(−1) = 0.37 is not zero.')),
  (('0,5 și 0,5.', '0.5 and 0.5.'), ('Δ = 2 nu înseamnă egalitate; ponderile scad exponențial cu Δ/2.', 'Δ = 2 does not mean a tie; the weights fall exponentially with Δ/2.')),
  (('Circa 0,73 și 0,27.', 'About 0.73 and 0.27.'), ('w = 1/(1 + e^(−1)) = 0,73 și e^(−1)/(1 + e^(−1)) = 0,27.', 'w = 1/(1 + e^(−1)) = 0.73 and e^(−1)/(1 + e^(−1)) = 0.27.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch6-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH6_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH6_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH6_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
