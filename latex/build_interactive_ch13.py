"""Build the interactive pages of Chapter 13: interactive/ch13.html (EN) and interactive/ch13_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of the
slides: the slide on the left, its interactive version on the right, with live tables and worked examples. Logic in
interactive/ch13.js, inputs in interactive/ch13_data.js (Quantlets/Ch_13/export_interactive.py), slide images in interactive/slides/.
On the web lecture (build_web_lecture.py) the table blocks travel inside a chart block (widgets) and ch13.js moves each one to its
own frame. The RO deck has one acronym frame more than the EN deck: XBLOCKS give the RO frame number, the EN frame is found by
the offset and checked against TITLES_EN.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 13
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_13'

# (anchor, chart file, section key); order of the slides. Only the chart blocks: build_web_lecture.py replaces these charts.
BLOCKS = [('bv', 'sfm_ch13_bias_variance', 'bv'), ('cvs', 'sfm_ch13_cv_schemes', 'val'), ('leak', 'sfm_ch13_leakage', 'val'), ('shrink', 'sfm_ch13_shrinkage', 'pen'),
          ('tree', 'sfm_ch13_tree', 'trees'), ('ens', 'sfm_ch13_ensembles', 'trees'), ('mlp', 'sfm_ch13_mlp', 'nn'), ('nnarch', 'sfm_ch13_nnarch', 'nn'),
          ('nnjpy', 'sfm_ch13_nnjpy', 'nn'), ('rvf', 'sfm_ch13_rv_forecasts', 'rv'), ('imp', 'sfm_ch13_importance', 'rv'), ('sign', 'sfm_ch13_sign', 'sign'),
          ('qvar', 'sfm_ch13_qvar', 'var'), ('snoop', 'sfm_ch13_snooping', 'snoop'), ('gkx', 'sfm_ch13_gkx', 'gkx')]
# table and worked-example blocks: (anchor, frame number in the RO deck, section key, chart block that carries it on the web page)
XBLOCKS = [('wf-we', 22, 'val', 'cvs'), ('ridge-we', 27, 'pen', 'shrink'), ('lasso-we', 28, 'pen', 'shrink'), ('split-we', 32, 'trees', 'tree'),
           ('rf-we', 34, 'trees', 'ens'), ('gb-we', 35, 'trees', 'ens'), ('mlp-we', 39, 'nn', 'mlp'), ('har-we', 51, 'rv', 'rvf'), ('r2-we', 53, 'rv', 'rvf'),
           ('qlike-we', 54, 'rv', 'rvf'), ('dm-we', 55, 'rv', 'rvf'), ('rv-tab', 56, 'rv', 'rvf'), ('sign-we', 61, 'sign', 'sign'), ('pin-we', 64, 'var', 'qvar'),
           ('var-tab', 66, 'var', 'qvar'), ('maxsr-we', 70, 'snoop', 'snoop'), ('dsr-we', 73, 'snoop', 'snoop')]
ORDER = ['bv', 'cvs', 'wf-we', 'leak', 'ridge-we', 'lasso-we', 'shrink', 'split-we', 'tree', 'rf-we', 'gb-we', 'ens', 'mlp-we', 'mlp', 'nnarch', 'nnjpy',
         'har-we', 'r2-we', 'qlike-we', 'dm-we', 'rv-tab', 'rvf', 'imp', 'sign-we', 'sign', 'pin-we', 'var-tab', 'qvar', 'maxsr-we', 'snoop', 'dsr-we', 'gkx']
TITLES = {'wf-we': 'Validarea walk-forward, pas cu pas', 'ridge-we': 'Regresia ridge (2/2)', 'lasso-we': 'Lasso', 'split-we': 'Exemplu rezolvat: o împărțire, pas cu pas',
          'rf-we': 'Random forest', 'gb-we': 'Gradient boosting', 'mlp-we': 'De la perceptron la perceptronul multistrat (2/2)', 'har-we': 'Reperul HAR (1/2)',
          'r2-we': 'Evaluarea (1/3): R² în afara eșantionului', 'qlike-we': 'Evaluarea (2/3): QLIKE', 'dm-we': 'Evaluarea (3/3): testul Diebold–Mariano',
          'rv-tab': 'Rezultatele: S&P 500 și Bitcoin', 'sign-we': 'Reperul clasei majoritare', 'pin-we': 'Regresia cuantilică (1/2)',
          'var-tab': 'Backtesting pentru VaR 1%: metode cuantilice, HS și GARCH-t (1/2)', 'maxsr-we': 'Cea mai bună dintre N strategii (2/2)',
          'dsr-we': 'Raportul Sharpe deflatat (2/2)'}
TITLES_EN = {'wf-we': 'Walk-Forward Validation, Step by Step', 'ridge-we': 'Ridge Regression (2/2)', 'lasso-we': 'The Lasso', 'split-we': 'Worked Example: One Split, Step by Step',
             'rf-we': 'Random Forest', 'gb-we': 'Gradient Boosting', 'mlp-we': 'From the Perceptron to the Multilayer Perceptron (2/2)', 'har-we': 'The HAR Benchmark (1/2)',
             'r2-we': 'Evaluation (1/3): Out-of-Sample R²', 'qlike-we': 'Evaluation (2/3): QLIKE', 'dm-we': 'Evaluation (3/3): the Diebold–Mariano Test',
             'rv-tab': 'Results: S&P 500 and Bitcoin', 'sign-we': 'The Majority-Class Baseline', 'pin-we': 'Quantile Regression (1/2)',
             'var-tab': 'Backtest of VaR 1%: Quantile Methods, HS and GARCH-t (1/2)', 'maxsr-we': 'The Best of N Strategies (2/2)', 'dsr-we': 'The Deflated Sharpe Ratio (2/2)'}

PFX = {'bv': 'bv', 'cvs': 'cs', 'wf-we': 'wf', 'leak': 'lk', 'ridge-we': 'rw', 'lasso-we': 'lw', 'shrink': 'sh', 'split-we': 'sp', 'tree': 'tr', 'rf-we': 'bg', 'gb-we': 'gb',
       'ens': 'en', 'mlp-we': 'mw', 'mlp': 'ml', 'nnarch': 'na', 'nnjpy': 'nj', 'har-we': 'hw', 'r2-we': 'r2', 'qlike-we': 'ql', 'dm-we': 'dm', 'rv-tab': 'rt', 'rvf': 'rf',
       'imp': 'im', 'sign-we': 'sz', 'sign': 'sg', 'pin-we': 'pl', 'var-tab': 'vt', 'qvar': 'qv', 'maxsr-we': 'mx', 'snoop': 'sn', 'dsr-we': 'ds', 'gkx': 'gk'}

SECTIONS = {
 'ro': dict(bv='Overfitting, deplasare și varianță', val='Validarea pentru serii de timp', pen='Regresia penalizată: ridge și lasso',
            trees='Arbori de decizie, random forest și gradient boosting', nn='Rețele neuronale', rv='Aplicație: prognoza volatilității realizate',
            sign='Aplicație: semnul randamentului de mîine', var='Aplicație: VaR 1% prin metode cuantilice', snoop='Data snooping și raportul Sharpe deflatat',
            gkx='Studiu de caz: Gu, Kelly și Xiu (2020)'),
 'en': dict(bv='Overfitting, Bias and Variance', val='Validation for Time Series', pen='Penalised Regression: Ridge and Lasso',
            trees='Trees, Random Forest and Gradient Boosting', nn='Neural Networks', rv='Application: Forecasting Realised Volatility',
            sign='Application: the Sign of Tomorrow’s Return', var='Application: VaR 1% by Quantile Methods', snoop='Data Snooping and the Deflated Sharpe Ratio',
            gkx='Case Study: Gu, Kelly and Xiu (2020)'),
}

L = {
 'ro': dict(file='ch13_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 13 · Învățarea automată în finanțe', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'tabelele și exemplele lucrate se recalculează odată cu el. Abrevieri: ML (machine learning, învățare automată), OLS (ordinary least squares, metoda celor mai mici '
        'pătrate), CV (cross-validation, validare încrucișată), SSE (suma pătratelor erorilor), MSE (mean squared error, eroarea pătratică medie), RMSE (rădăcina erorii pătratice '
        'medii), RV (varianța realizată), HAR (heterogeneous autoregressive, modelul autoregresiv heterogen), RF (random forest), GB (gradient boosting), MLP (multilayer perceptron, '
        'perceptron multistrat), ReLU (rectified linear unit), RBF (radial basis function, funcție de bază radială), GARCH (generalised autoregressive conditional heteroskedasticity), '
        'QLIKE (pierderea quasi-likelihood), DM (testul Diebold–Mariano), LRV (long-run variance, varianța pe termen lung), OOS (out-of-sample, în afara eșantionului), AUC (area under '
        'the curve, aria de sub curba ROC), VaR (value at risk), QR (regresie cuantilică), QGB (quantile gradient boosting), HS (simulare istorică), SR (raportul Sharpe), PSR '
        '(probabilistic Sharpe ratio), DSR (deflated Sharpe ratio, raportul Sharpe deflatat), MA (medie mobilă), H-L (decila de sus minus decila de jos), BET (indicele Bursei de '
        'Valori București), pp (puncte procentuale).',
   src='Datele cursului din data/market (pînă la 18 septembrie 2026): S&P 500 și Bitcoin (prețuri de deschidere, maxim, minim și închidere), BET, GBP/USD, randamentul '
       'obligațiunilor germane pe 10 ani, aurul, USD/JPY. Simularea deplasare–varianță, schema de validare, traiectoriile ridge și lasso, arborii, curbele random forest și '
       'boosting, rețelele RBF și MLP, prognozele walk-forward ale celor șase modele de volatilitate, ale clasificatorilor de semn și ale celor patru metode VaR vin din codul '
       'din curs (Quantlets/Ch_13, semințele 2026 și 3); modelele grele nu sînt reestimate în browser. Browserul recalculează din aceste prognoze HAR, R² în afara eșantionului, '
       'QLIKE, testul Diebold–Mariano, acuratețea, testul z, AUC, testele Kupiec și Christoffersen, pierderea cuantilică, maximul așteptat al raportului Sharpe și DSR. '
       'Gu, Kelly și Xiu (2020): valorile publicate în tabelele 1 și 7 ale lucrării. Alte niveluri de zgomot, orizonturi, adîncimi, numere de arbori și de unități, ponderi de '
       'antrenare, perioade și ani de separare sînt rulări suplimentare ale aceluiași cod; exemplul de boosting pe cele șase săptămîni este un calcul suplimentar.'),
 'en': dict(file='ch13.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 13 · Machine Learning in Finance', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by '
        'step; the tables and worked examples are recomputed with it. Abbreviations: ML (machine learning), OLS (ordinary least squares), CV (cross-validation), SSE (sum of '
        'squared errors), MSE (mean squared error), RMSE (root mean squared error), RV (realised variance), HAR (heterogeneous autoregressive model), RF (random forest), '
        'GB (gradient boosting), MLP (multilayer perceptron), ReLU (rectified linear unit), RBF (radial basis function), GARCH (generalised autoregressive conditional '
        'heteroskedasticity), QLIKE (quasi-likelihood loss), DM (Diebold–Mariano test), LRV (long-run variance), OOS (out-of-sample), AUC (area under the ROC curve), '
        'VaR (value at risk), QR (quantile regression), QGB (quantile gradient boosting), HS (historical simulation), SR (Sharpe ratio), PSR (probabilistic Sharpe ratio), '
        'DSR (deflated Sharpe ratio), MA (moving average), H-L (top decile minus bottom decile), BET (index of the Bucharest Stock Exchange), pp (percentage points).',
   src='Course data in data/market (up to 18 September 2026): S&P 500 and Bitcoin (open, high, low and close prices), BET, GBP/USD, the German 10-year yield, gold, '
       'USD/JPY. The bias–variance simulation, the validation diagram, the ridge and lasso paths, the trees, the random-forest and boosting curves, the RBF and MLP networks, '
       'and the walk-forward forecasts of the six volatility models, of the sign classifiers and of the four VaR methods come from the code of the lecture (Quantlets/Ch_13, '
       'seeds 2026 and 3); the heavy models are not refitted in the browser. From these forecasts the browser recomputes HAR, the out-of-sample R², QLIKE, the Diebold–Mariano '
       'test, accuracy, the z test, AUC, the Kupiec and Christoffersen tests, the quantile loss, the expected maximum Sharpe ratio and the DSR. Gu, Kelly and Xiu (2020): the '
       'values published in Tables 1 and 7 of the paper. Other noise levels, horizons, depths, numbers of trees and units, training shares, periods and split years are extra '
       'runs of the same code; the boosting example on the six weeks is an extra computation.'),
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
    sl = ' (slide)'
    c = (lambda s: s.replace('.', '{,}')) if ro else (lambda s: s)
    n = (lambda s: s.replace('.', ',')) if ro else (lambda s: s)
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    T = lambda r, e: r if ro else e
    ASSET = lambda i, extra=(): select(i, T('Seria', 'Series'), [('sp500', 'S&P 500' + lec), ('btc', 'Bitcoin')] + list(extra), 'sp500')
    PERIOD = lambda i: select(i, T('Perioada de test', 'Test period'), [('all', T('toată perioada', 'the whole period') + lec), ('pre', T('pînă în 2019', 'up to 2019')),
                                                                      ('post', T('din 2020', 'from 2020'))], 'all')
    MODELS = [('MLP', 'MLP' + lec), ('Lasso', 'lasso'), ('RF', 'random forest'), ('GB', 'gradient boosting')]
    FEATS = [('d', T('log RV zilnic', 'log RV day')), ('w', T('log RV săptămînal', 'log RV week') + lec), ('m', T('log RV lunar', 'log RV month')),
             ('d1', T('log RV, lagul 1', 'log RV lag 1')), ('d2', T('log RV, lagul 2', 'log RV lag 2')), ('d3', T('log RV, lagul 3', 'log RV lag 3')),
             ('d4', T('log RV, lagul 4', 'log RV lag 4')), ('q', T('log RV trimestrial', 'log RV quarter')), ('r', T('randamentul', 'return')),
             ('r_neg', T('randamentul negativ', 'negative return')), ('abs_r', T('randamentul absolut', 'absolute return'))]
    YRS = [(str(y), str(y)) for y in range(2010, 2027)]
    return {
     'bv': (slider('bv-d', T('Adîncimea arborelui evidențiat', 'Highlighted tree depth'), 1, 10, 1, 3)
            + select('bv-s', T('Varianța zgomotului σ²', 'Noise variance σ²'), [('0.3', n('0.09') + lec), ('0.2', n('0.04')), ('0.5', n('0.25'))], '0.3') + play('bv'),
            CSS + charts('bv-chart')),
     'cvs': (slider('cs-k', T('Numărul de partiții K', 'Number of folds K'), 3, 8, 1, 5) + slider('cs-h', T('Rînduri eliminate prin purging', 'Rows purged'), 0, 6, 1, 3)
             + slider('cs-e', T('Rînduri de embargo', 'Embargo rows'), 0, 5, 1, 2) + play('cs'), charts('cs-chart')),
     'wf-we': (slider('wf-y', T('Anul de test', 'Test year'), 2013, 2026, 1, 2013) + play('wf'),
               charts('wf-chart') + xh(T('Cele 14 partiții walk-forward (S&P 500)', 'The 14 walk-forward folds (S&P 500)')) + tw('wf-tab')
               + we('wf-st', f'{WE}: {T("o partiție walk-forward cu purging", "one walk-forward fold with purging")}',
                    T(r'\text{antrenare}_Y = \{t < 1\,\text{ian.}\,Y\} \setminus \{\text{ultimele } h \text{ rînduri}\}', r'\text{train}_Y = \{t < 1\,\text{Jan}\,Y\} \setminus \{\text{last } h \text{ rows}\}')
                    + r', \qquad \text{test}_Y = \{t \in Y\}, \qquad y_t = \ln\Big(\tfrac1h\sum_{j=1}^{h} v_{t+j}\Big)',
                    select('wf-h', T('Orizontul țintei și al purging-ului h (zile)', 'Horizon of the target and of the purging h (days)'),
                           [('5', '5' + lec), ('1', '1'), ('10', '10'), ('22', '22')], '5'))),
     'leak': (select('lk-h', T('Orizontul țintei h (zile)', 'Target horizon h (days)'), [('21', '21' + lec), ('5', '5'), ('63', '63')], '21') + play('lk'), charts('lk-chart')),
     'ridge-we': (slider('rw-l', T('Penalizarea λ', 'Penalty λ'), 0, 200, 5, 50) + play('rw'),
                  charts('rw-chart') + we('rw-we', f'{WE}: {T("ridge cu o singură variabilă", "ridge with one variable")}',
                                          r'\hat\beta^{\,ridge} = \frac{S_{xy}}{S_{xx} + \lambda} = \frac{S_{xx}}{S_{xx} + \lambda}\,\hat\beta^{\,OLS}, \qquad \hat\beta^{\,OLS} = \frac{S_{xy}}{S_{xx}}',
                                          select('rw-s', 'S_xx', [('100', '100' + sl), ('50', '50'), ('200', '200')], '100'))),
     'lasso-we': (slider('lw-l', T('Penalizarea λ', 'Penalty λ'), 0, 100, 2, 20) + play('lw'),
                  charts('lw-chart') + we('lw-we', f'{WE}: {T("soft thresholding", "soft thresholding")}',
                                          r'\hat\beta^{\,lasso} = \mathrm{sign}(S_{xy})\,\frac{\max(|S_{xy}| - \lambda/2,\, 0)}{S_{xx}}, \qquad S_{xx} = 100',
                                          select('lw-s', 'S_xy', [('30', '30' + sl), ('-30', '−30'), ('10', '10')], '30'))),
     'shrink': (slider('sh-i', T('Penalizarea (poziția pe grila de 60 de valori)', 'Penalty (position on the grid of 60 values)'), 0, 59, 1, 28)
                + select('sh-f', T('Variabila evidențiată', 'Highlighted feature'), FEATS, 'w') + play('sh'), charts('sh-chart')),
     'split-we': (select('sp-s', T('Pragul candidat s', 'Candidate threshold s'), [('1.25', n('1.25') + sl), ('0.7', n('0.70')), ('0.9', n('0.90')), ('1.75', n('1.75')), ('2.2', n('2.20'))], '1.25') + play('sp'),
                  charts('sp-chart') + xh(T('Împărțirile candidate (tabelul de pe slide)', 'The candidate splits (the table of the slide)')) + tw('sp-tab')
                  + we('sp-we', f'{WE}: {T("alegerea pragului", "choosing the threshold")}', r'\mathrm{SSE}(s) = \sum_{x_t \le s}(y_t - \bar y_L)^2 + \sum_{x_t > s}(y_t - \bar y_R)^2',
                       select('sp-m', T('Numărul minim de observații într-o frunză', 'Minimum number of observations in a leaf'), [('1', '1' + sl), ('2', '2')], '1'))),
     'tree': (select('tr-d', T('Adîncimea maximă', 'Maximum depth'), [('2', '2' + lec), ('1', '1'), ('3', '3')], '2')
              + select('tr-l', T('Numărul minim de zile într-o frunză', 'Minimum number of days in a leaf'), [('50', '50' + lec), ('20', '20'), ('100', '100')], '50') + play('tr'),
              charts('tr-chart')),
     'rf-we': (slider('bg-B', T('Numărul de arbori B', 'Number of trees B'), 1, 300, 1, 10) + play('bg'),
               charts('bg-chart') + we('bg-we', f'{WE}: {T("varianța unei medii de arbori", "the variance of an average of trees")}',
                                       r'\mathrm{Var}\Big(\frac1B\sum_{b=1}^{B}\hat f_b\Big) = \rho\sigma^2 + \frac{(1 - \rho)\,\sigma^2}{B}',
                                       select('bg-r', T('Corelația dintre arbori ρ', 'Correlation between trees ρ'), [('0.3', n('0.3') + sl), ('0', '0'), ('0.1', n('0.1')), ('0.6', n('0.6'))], '0.3'))),
     'gb-we': (slider('gb-m', T('Numărul de pași m', 'Number of steps m'), 1, 50, 1, 1) + play('gb'),
               charts('gb-chart') + we('gb-st', f'{WE}: {T("boosting cu arbori de adîncime 1 pe cele șase săptămîni (calcul suplimentar)", "boosting with depth-1 trees on the six weeks (extra computation)")}',
                                       r'F_0 = \bar y, \qquad h_m = \arg\min_{h}\sum_t \big(y_t - F_{m-1}(x_t) - h(x_t)\big)^2, \qquad F_m = F_{m-1} + \nu\,h_m',
                                       select('gb-v', T('Rata de învățare ν', 'Learning rate ν'), [('0.1', n('0.1')), ('0.3', n('0.3')), ('0.03', n('0.03')), ('1', '1')], '0.1'))),
     'ens': (slider('en-n', T('Numărul de arbori (random forest: cel mult 200)', 'Number of trees (random forest: at most 200)'), 1, 500, 1, 200)
             + slider('en-d', T('Adîncimea marcată (random forest)', 'Marked depth (random forest)'), 1, 14, 1, 8) + play('en'), charts('en-chart')),
     'mlp-we': (slider('mw-k', T('Neuroni ascunși K', 'Hidden neurons K'), 1, 64, 1, 16) + play('mw'),
                charts('mw-chart') + we('mw-we', f'{WE}: {T("numărul de parametri", "the number of parameters")}',
                                        r'\hat y = c + \sum_{k=1}^{K} v_k\, g(b_k + w_k^\top x), \qquad \#\theta = \underbrace{pK + K}_{\text{' + T('strat ascuns', 'hidden') + r'}} + \underbrace{K + 1}_{\text{' + T('ieșire', 'output') + '}}',
                                        select('mw-p', T('Numărul de intrări p', 'Number of inputs p'), [('11', '11' + sl), ('3', '3'), ('5', '5')], '11'))),
     'mlp': (select('ml-g', T('Funcția de activare g', 'Activation function g'), [('relu', 'ReLU'), ('logistic', T('logistică', 'logistic')), ('tanh', 'tanh')], 'relu')
             + slider('ml-z', T('Argumentul z', 'Argument z'), -4, 4, 0.25, 1) + slider('ml-k', T('Neuroni ascunși K', 'Hidden neurons K'), 1, 8, 1, 5) + play('ml'), charts('ml-chart')),
     'nnarch': (select('na-p', T('Perioada', 'Period'), [('all', '2010–2026' + lec)] + YRS, 'all')
                + select('na-u', T('Unitățile rețelei RBF', 'Units of the RBF network'), [('3', '3' + lec), ('2', '2'), ('5', '5')], '3') + play('na'), charts('na-chart')),
     'nnjpy': (select('nj-s', T('Ponderea zilelor de antrenare', 'Share of training days'), [('0.8', '80%' + lec), ('0.6', '60%'), ('0.9', '90%')], '0.8') + play('nj'), charts('nj-chart')),
     'har-we': (slider('hw-y', T('Partiția (anul de test)', 'Fold (test year)'), 2013, 2026, 1, 2013) + play('hw'),
                charts('hw-chart') + we('hw-we', f'{WE}: {T("o prognoză HAR", "one HAR forecast")}',
                                        r'\hat y_t = \hat\beta_0 + \hat\beta_d\ln v_t + \hat\beta_w\ln v_t^{(w)} + \hat\beta_m\ln v_t^{(m)}, \qquad \hat h_t = \exp(\hat y_t + \hat s^2/2)',
                                        select('hw-d', T('Ziua prognozei', 'Day of the forecast'), [('first', T('prima zi a anului', 'the first day of the year')),
                                                                                                    ('max', T('ziua cu prognoza maximă', 'the day with the highest forecast')), ('last', T('ultima zi', 'the last day'))], 'first'))),
     'r2-we': (select('r2-m', T('Modelul', 'Model'), MODELS + [('Mean', T('media de antrenare', 'training mean'))], 'MLP') + play('r2'),
               charts('r2-chart') + we('r2-st', f'{WE}: {T("R² în afara eșantionului", "the out-of-sample R²")}', r'R^2_{OOS} = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \tilde y_t)^2}',
                                       select('r2-b', T('Reperul', 'Benchmark'), [('HAR', 'HAR' + lec), ('Mean', T('media de antrenare', 'training mean'))], 'HAR') + ASSET('r2-a'))),
     'qlike-we': (slider('ql-h', T('Prognoza varianței ĥ', 'Variance forecast ĥ'), 0.1, 3.5, 0.05, 0.5) + play('ql'),
                  charts('ql-chart') + we('ql-we', f'{WE}: {T("pierderea QLIKE", "the QLIKE loss")}', r'L_t = \frac{\mathrm{RV}_t}{\hat h_t} + \ln\hat h_t',
                                          select('ql-r', T('Varianța realizată RV', 'Realised variance RV'), [('1', '1' + sl), ('0.5', n('0.5')), ('2', '2')], '1'))),
     'dm-we': (slider('dm-l', T('Laguri Newey–West', 'Newey–West lags'), 0, 22, 1, 5) + play('dm'),
               charts('dm-chart') + we('dm-st', f'{WE}: {T("statistica Diebold–Mariano", "the Diebold–Mariano statistic")}',
                                       r'd_t = L_t^{A} - L_t^{HAR}, \qquad \widehat{\mathrm{LRV}} = \hat\gamma_0 + 2\sum_{j=1}^{q}\Big(1 - \frac{j}{q + 1}\Big)\hat\gamma_j, \qquad \mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}/n}}',
                                       select('dm-m', T('Modelul A', 'Model A'), MODELS, 'MLP') + ASSET('dm-a'))),
     'rv-tab': (PERIOD('rt-p') + play('rt'),
                charts('rt-chart') + xh(T('Rezultatele în afara eșantionului: R²_OOS pe log RV (%), QLIKE, statistica DM față de HAR (p-value)', 'Out-of-sample results: R²_OOS on log RV (%), QLIKE, DM statistic against HAR (p-value)'))
                + '<div class="wctl">' + select('rt-l', T('Laguri Newey–West ale testului DM', 'Newey–West lags of the DM test'), [('5', '5' + lec), ('0', '0'), ('10', '10'), ('22', '22')], '5') + '</div>'
                + tw('rt-tab')),
     'rvf': (select('rf-m', T('Modelele comparate cu HAR', 'Models compared with HAR'), [('RF,MLP', T('random forest și MLP', 'random forest and MLP') + lec), ('Lasso,GB', T('lasso și gradient boosting', 'lasso and gradient boosting')),
                                                                                  ('Lasso,MLP', T('lasso și MLP', 'lasso and MLP')), ('RF,GB', T('random forest și gradient boosting', 'random forest and gradient boosting'))], 'RF,MLP')
             + ASSET('rf-a') + select('rf-y', T('Anul din stînga', 'Year on the left'), [('2020', '2020' + lec), ('2018', '2018'), ('2022', '2022')], '2020') + play('rf'), charts('rf-chart')),
     'imp': (select('im-f', T('Valorile Shapley afișate', 'Shapley values shown'), [('all', T('toate trei', 'all three') + lec), ('d', T('ziua', 'day')), ('w', T('săptămîna', 'week')), ('m', T('luna', 'month'))], 'all') + play('im'),
             charts('im-chart') + xh(T('Importanța prin permutare (10 amestecări) și media |φ| (random forest pe cele 3 variabile HAR)', 'Permutation importance (10 shuffles) and mean |φ| (random forest on the 3 HAR features)')) + tw('im-tab')),
     'sign-we': (select('sz-m', T('Modelul', 'Model'), [('RF', 'random forest' + lec), ('GB', 'gradient boosting'), ('Logit', 'logit')], 'RF') + play('sz'),
                 charts('sz-chart') + we('sz-we', f'{WE}: {T("testul față de reper", "the test against the baseline")}',
                                         r'z = \frac{\widehat{\mathrm{acc}} - \mathrm{acc}_0}{\sqrt{\mathrm{acc}_0(1 - \mathrm{acc}_0)/n}} \approx N(0, 1)',
                                         select('sz-a', T('Seria', 'Series'), [('sp500', 'S&P 500' + lec), ('bet', 'BET'), ('btc', 'Bitcoin')], 'sp500'))),
     'sign': (slider('sg-e', T('Ultimul an de test', 'Last test year'), 2010, 2026, 1, 2026)
              + select('sg-c', T('Pragul probabilității pentru „creștere”', 'Probability cut-off for “up”'), [('0.5', n('0.5') + lec), ('0.52', n('0.52')), ('0.55', n('0.55'))], '0.5') + play('sg'),
              charts('sg-chart')),
     'pin-we': (slider('pl-q', T('Prognoza constantă q (%)', 'Constant forecast q (%)'), -8, 1, 0.05, -2.7) + play('pl'),
                charts('pl-chart') + we('pl-we', f'{WE}: {T("pierderea cuantilică pe randamentele S&P 500", "the quantile loss on the S&P 500 returns")}',
                                        r'\rho_\alpha(u) = u\,(\alpha - \mathbf{1}\{u < 0\}), \qquad u = y - q, \qquad \hat q = \arg\min_q \frac1n\sum_t \rho_\alpha(y_t - q)',
                                        select('pl-a', T('Nivelul α', 'Level α'), [('0.01', n('0.01') + lec), ('0.05', n('0.05')), ('0.1', n('0.10'))], '0.01'))),
     'var-tab': (PERIOD('vt-p') + play('vt'),
                 charts('vt-chart') + xh(T('Backtesting pentru VaR 1% (tabelul de pe slide)', 'Backtest of VaR 1% (the table of the slide)')) + tw('vt-tab')),
     'qvar': (select('qv-m', T('Depășirile marcate', 'Exceedances marked'), [('QR', 'QR' + lec), ('QGB', 'QGB'), ('HS', 'HS'), ('GARCH-t', 'GARCH-t')], 'QR') + ASSET('qv-a')
              + select('qv-y', T('Anul din stînga', 'Year on the left'), [('2020', '2020' + lec), ('2018', '2018'), ('2022', '2022')], '2020') + play('qv'), charts('qv-chart')),
     'maxsr-we': (slider('mx-n', T('Numărul de strategii N (poziția pe grilă)', 'Number of strategies N (position on the grid)'), 0, 9, 1, 6) + play('mx'),
                  charts('mx-chart') + we('mx-we', f'{WE}: {T("maximul așteptat al celor N rapoarte Sharpe", "the expected maximum of N Sharpe ratios")}',
                                          r'\mathrm{E}[\max] \approx \sigma\Big[(1 - \gamma)\,\Phi^{-1}\big(1 - \tfrac1N\big) + \gamma\,\Phi^{-1}\big(1 - \tfrac{1}{Ne}\big)\Big], \qquad \sigma = 1/\sqrt{Y}',
                                          select('mx-y', T('Anii de date Y', 'Years of data Y'), [('10', '10' + sl), ('5', '5'), ('20', '20')], '10'))),
     'snoop': (slider('sn-n', T('Regulile încercate N', 'Rules tried N'), 1, 30, 1, 30)
               + select('sn-s', T('Sfîrșitul eșantionului de estimare', 'End of the estimation sample'), [('2014', '2014' + lec), ('2009', '2009'), ('2019', '2019')], '2014')
               + select('sn-y', T('Anii de date în simulare Y', 'Years of data in the simulation Y'), [('10', '10' + lec), ('5', '5'), ('20', '20')], '10') + play('sn'), charts('sn-chart')),
     'dsr-we': (slider('ds-n', T('Numărul de reguli încercate N', 'Number of rules tried N'), 1, 200, 1, 30) + play('ds'),
                charts('ds-chart') + we('ds-we', f'{WE}: {T("raportul Sharpe deflatat al celei mai bune reguli MA", "the deflated Sharpe ratio of the best MA rule")}',
                                        r'\mathrm{DSR} = \Phi\Big(\frac{(\widehat{SR} - SR_0)\sqrt{T - 1}}{\sqrt{1 - \hat\gamma_3\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\Big)',
                                        select('ds-k', T('Asimetria și boltirea', 'Skewness and kurtosis'), [('sample', T('din eșantion, γ₃ = −0,44, γ₄ = 12,0', 'sample, γ₃ = −0.44, γ₄ = 12.0') + lec),
                                                                                                        ('normal', T('distribuția Normală, γ₃ = 0, γ₄ = 3', 'Normal distribution, γ₃ = 0, γ₄ = 3'))], 'sample'))),
     'gkx': (select('gk-m', T('Modelul evidențiat', 'Highlighted model'), [(m, m) for m in ('NN3', 'OLS-3+H', 'PLS', 'PCR', 'ENet+H', 'GLM+H', 'RF', 'GBRT+H', 'NN1', 'NN2', 'NN4', 'NN5')], 'NN3')
             + play('gk'), charts('gk-chart')),
    }


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang` (RO frame numbers; EN = RO minus the extra acronym frame)."""
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved by ch13.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch13_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch13_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


# readouts, table headers and worked-example steps: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const num = s => +String(s).replace(/[\s\u00a0%]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const X = ' (rulare suplimentară a codului din curs)';
const XS = ' (alte valori decît pe slide)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const pv = p => (p.startsWith('<') ? p : '= ' + p);
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const FLR = { d: 'log RV zilnic', w: 'log RV săptămînal', m: 'log RV lunar', d1: 'log RV, lagul 1', d2: 'log RV, lagul 2', d3: 'log RV, lagul 3', d4: 'log RV, lagul 4',
  q: 'log RV trimestrial', r: 'randamentul', r_neg: 'randamentul negativ', abs_r: 'randamentul absolut' };
const MN = { 'training mean': 'media de antrenare', HAR: 'HAR', lasso: 'lasso', 'random forest': 'random forest', 'gradient boosting': 'gradient boosting', MLP: 'MLP', logit: 'logit' };
const mn = m => MN[m] || m;
const PERN = { all: 'Toată perioada de test', pre: 'Zilele de test pînă în 2019', post: 'Zilele de test din 2020' };
const GN = { relu: 'ReLU', logistic: 'logistică', tanh: 'tanh' };
window.CH13_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`, fl: g => FLR[g],
 h: { wf: ['Anul de test', 'Zile de antrenare', 'Zile de test', 'Ultima zi de antrenare', 'Primul rînd eliminat'], sp: ['s', 'ȳ_L', 'ȳ_R', 'SSE'], na: 'nepermis',
      rt: ['Seria (zile)', 'Modelul', 'R² față de medie', 'R² față de HAR', 'QLIKE', 'DM (p)'],
      im: ['Variabila', 'Creșterea MSE', 'Abaterea standard', 'Media |φ|'], vt: ['Seria (zile)', 'Metoda', 'x', 'Rata (%)', 'p Kupiec', 'p ind.', 'Pierdere × 100', 'VaR mediu (%)'] },
 bv: (s2, d, b2, v, te, tr, best, bte, x) => `Simulare cu σ² = ${s2}, n = 80 și 300 de eșantioane de antrenare${x ? X : ''}. Adîncimea ${d}: deplasare² ${b2}, varianță ${v}; eroarea de test așteptată, deplasare² + varianță + σ², este ${te}, iar eroarea de antrenare ${tr}. Eroarea de test este minimă la adîncimea ${best} (${bte}).`,
 cs: (k, nb, nt, size, last, h, e, ntr, x) => `K = ${k}, ${h} ${h === 1 ? 'rînd eliminat' : 'rînduri eliminate'} prin purging, embargo de ${e} ${e === 1 ? 'rînd' : 'rînduri'}${x ? X : ''}. K-fold aleator, partiția 1: ${nb} din cele ${nt} observații de test au o vecină (t − 1 sau t + 1) în antrenare. Walk-forward: antrenarea crește de la ${size} la ${last} ${de(last)}observații, cu blocuri de test de cîte ${size}. K-fold cu purging, partiția din mijloc: ${ntr} ${de(ntr)}rînduri de antrenare din 60.`,
 wf1: (yr, a, b, n) => `Partiția ${yr}: modelul se estimează pe zilele de la ${a} la ${b}, adică ${n} ${de(n)}zile (fereastră extinsă).`,
 wf2: (h, fp, yr) => (+h === 1 ? `Purging: ultimul rînd de antrenare, din ${fp}, este eliminat: ținta lui, variabila proxy a zilei următoare, cade în ${yr}.` : `Purging: ultimele ${h} rînduri de antrenare, de la ${fp}, sînt eliminate: ținta lor, media variabilei proxy pe următoarele ${h} zile, cade parțial în ${yr}.`),
 wf3: (n, yr) => `Test: cele ${n} ${de(n)}zile de tranzacționare din ${yr}; partiția următoare le adaugă la antrenare. Embargoul nu este necesar, pentru că testul vine întotdeauna după antrenare.`,
 wf: (yr, ntr, nte, h, nf, x) => `Orizontul h = ${h}${x ? X : ''}: ${nf} partiții; partiția ${yr} antrenează pe ${ntr} ${de(ntr)}zile și testează pe ${nte} ${de(nte)}zile.`,
 lk: (h, sk, sw, sp, pk, pw, pp, nS, nP, x) => `Ținta: randamentul cumulat din următoarele ${h} ${de(h)}zile${x ? X : ''}. Mers aleator simulat (${nS} ${de(nS)}zile): K-fold aleator ${sk}, walk-forward ${sw}, walk-forward cu purging ${sp}. S&P 500 (${nP} ${de(nP)}zile): ${pk}, ${pw} și ${pp}.`,
 sh: (lamL, nz, fl, cl, co, lamR, cr, ratio, cv, nzcv) => (cv
   ? `Lasso la λ = ${lamL}, ales prin validare încrucișată walk-forward: ${nzcv} din 11 coeficienți nenuli; ${fl}: ${cl} (OLS ${co}). `
   : `Lasso la λ = ${lamL}: ${nz} din 11 coeficienți nenuli; ${fl}: ${cl} (OLS ${co}). `) + `Ridge la λ = ${lamR}: ${fl} ${cr}; norma vectorului de coeficienți este ${ratio} din norma OLS. Coeficienții OLS ai lui r, min(r, 0) și |r| nu sînt unici, pentru că |r| = r − 2 min(r, 0) (rangul matricei este 10): se afișează soluția de normă minimă.`,
 rw1: (sxx, sxy, b) => `OLS: β̂ = S_xy/S_xx = ${sxy}/${sxx} = ${b}.`,
 rw2: (lam, sxx, sxy, b, f) => `Ridge cu λ = ${lam}: β̂ = ${sxy}/(${sxx} + ${lam}) = ${b}; factorul de contracție S_xx/(S_xx + λ) = ${f}.`,
 rw3: (bl, lam) => `Lasso cu același λ = ${lam}: ${bl}. Ridge contractă coeficientul spre zero, dar nu îl anulează pentru niciun λ finit.`,
 rw: (lam, b, f, x) => `λ = ${lam}${x ? XS : ''}: β̂ ridge = ${b}, adică ${f} din coeficientul OLS.`,
 lw1: (sxy, lam, half) => `Pragul de contracție: λ/2 = ${half}; |S_xy| = ${Math.abs(sxy)}.`,
 lw2: (sxy, lam, nm, b) => `β̂ lasso = sign(S_xy) · max(|S_xy| − λ/2, 0)/S_xx = ${sxy < 0 ? '−' : '+'}${nm}/100 = ${b}.`,
 lw3: lim => `Coeficientul devine exact 0 pentru λ ≥ 2|S_xy| = ${lim}.`,
 lw: (lam, sxy, b, x) => `λ = ${lam}, S_xy = ${sxy}${x ? XS : ''}: β̂ lasso = ${b}.`,
 sp1: (yb, s0) => `Fără împărțire: ȳ = ${yb}, SSE = ${s0}.`,
 sp2: (s, nL, nR, yL, yR, e, ok, ml) => (ok ? `Pragul s = ${s}: ${nL} ${nL === 1 ? 'săptămînă' : 'săptămîni'} în stînga (ȳ_L = ${yL}), ${nR} în dreapta (ȳ_R = ${yR}); SSE = ${e}.` : `Pragul s = ${s} lasă mai puțin de ${ml} observații într-o frunză: nu este permis.`),
 sp3: (bs, be, s0, sh) => `Cea mai bună împărțire permisă: s = ${bs}, SSE = ${be}; ea explică 1 − ${be}/${s0} = ${sh} din variație.`,
 sp: (s, e, sh, bs, x) => (e === null ? `Pragul s = ${s} nu este permis${x ? XS : ''}; cea mai bună împărțire: s = ${bs}.` : `Pragul s = ${s}${x ? XS : ''}: SSE = ${e}, adică ${sh} din variație explicată; cea mai bună împărțire: s = ${bs}.`),
 tr: (dep, ml, f, thr, vol, lv, r2, n, x) => `Adîncimea ${dep}, cel puțin ${ml} ${de(ml)}zile într-o frunză${x ? X : ''}; ${n} ${de(n)}zile (2008–2012). Prima împărțire: ${f} ≤ ${thr}, adică o volatilitate anuală de circa ${vol}%. Frunzele (media țintei, zile): ${lv.map(([v, k]) => `${v} (${k})`).join('; ')}. Arborele explică ${r2} din variația țintei în eșantion.`,
 bg1: (rho, B) => `B = ${B} ${B === 1 ? 'arbore' : de(B) + 'arbori'}, fiecare cu varianța σ² = 1 și corelația ρ = ${rho} între oricare doi.`,
 bg2: (rho, B, a, b, v) => `Var(media) = ρσ² + (1 − ρ)σ²/B = ${a} + ${b} = ${v}.`,
 bg3: (rho, v1, v10, v300) => `Pentru ρ = ${rho}: B = 1: ${v1}; B = 10: ${v10}; B = 300: ${v300}. Partea ρσ², comună tuturor arborilor, nu scade cu B.`,
 bg: (B, rho, v, x) => `B = ${B}, ρ = ${rho}${x ? XS : ''}: varianța mediei este ${v}.`,
 gb1: (f0, s0) => `F₀ = ȳ = ${f0}; SSE de antrenare ${s0}.`,
 gb2: (s, a, b) => `Pasul 1: reziduurile y − F₀; cel mai bun arbore de adîncime 1 pe reziduuri împarte la s = ${s}, cu valorile ${a} (stînga) și ${b} (dreapta).`,
 gb3: (nu, a, b, s1) => `F₁ = F₀ + ν h₁, cu ν = ${nu}: ${a} în stînga și ${b} în dreapta; SSE = ${s1}.`,
 gb4: (M, e, a, b) => `După ${M} ${M === 1 ? 'pas' : de(M) + 'pași'}: SSE = ${e}; prognoza merge de la ${a} (prima săptămînă) la ${b} (ultima).`,
 gb: (nu, M, e) => `Calcul suplimentar pe cele șase săptămîni ale exemplului rezolvat, cu arbori de adîncime 1: ν = ${nu}, ${M} ${M === 1 ? 'pas' : de(M) + 'pași'}, SSE de antrenare ${e}.`,
 en: (n, B, d, rf, bd, bv, td, gb, har, ntr, nva, x) => `Antrenare 2008–2018 (${ntr} de zile), validare 2019–2022 (${nva} de zile)${x ? X : ''}. Random forest cu ${B} ${B === 1 ? 'arbore' : de(B) + 'arbori'}: MSE de validare ${rf} la adîncimea ${d} (un singur arbore: ${td}); minimul ${bv} la adîncimea ${bd}. Boosting după ${n} ${n === 1 ? 'arbore' : de(n) + 'arbori'}: ` + gb.map(([l, v, b, k]) => `rata ${l.replace('.', ',')}: ${v} (minimul ${b} după ${k} ${k === 1 ? 'arbore' : de(k) + 'arbori'})`).join('; ') + `. HAR: ${har}.`,
 mw1: (p, K, pk, k, hid) => `Stratul ascuns: ${p} × ${K} ponderi + ${K} constante = ${hid}.`,
 mw2: (K, o) => `Ieșirea: ${K} ponderi + 1 constantă = ${o}.`,
 mw3: t => `În total: ${t} ${de(t)}parametri.`,
 mw: (p, K, t, x) => `${p} intrări, ${K} neuroni ascunși${x ? XS : ''}: ${t} ${de(t)}parametri.`,
 ml: (g, z, gz, dg, K, np) => `g = ${GN[g]}: g(${z}) = ${gz}, derivata g′(${z}) = ${dg}` + (num(dg) < 0.05 && g !== 'relu' ? ' (zona de saturare: gradientul aproape dispare)' : g === 'relu' && num(z) <= 0 ? ' (ReLU este 0 pentru z ≤ 0, cu gradientul 0)' : '') + `. Rețeaua din stînga, cu 3 intrări și ${K} ${K === 1 ? 'neuron ascuns' : 'neuroni ascunși'}, are ${np} ${de(np)}parametri.`,
 na: (per, u, mR, mG, xR, dR, xG, dG, cR, cM, n, x) => `${per === 'all' ? '2010–2026' : per} (${n} ${de(n)}zile), rețeaua RBF cu ${u} unități${x ? X : ''}: volatilitatea anualizată medie ${mR}% (GARCH ${mG}%); maximul rețelei ${xR}% (${dR}), al GARCH ${xG}% (${dG}). Corelația cu volatilitatea GARCH: ${cR} (rețeaua RBF), ${cM} (MLP).`,
 nj: (sh, split, rtr, wtr, rte, wte, ratio, mtr, mte, r2, x) => `Antrenare pe primele ${sh}% din zile, test de la ${split}${x ? X : ''}. RMSE în yeni: antrenare ${rtr} (mersul aleator ${wtr}), test ${rte} (mersul aleator ${wte}): în afara eșantionului rețeaua este de ${ratio} ${de(ratio)}ori mai slabă. Maximul din antrenare ${mtr}, din test ${mte}. Aceeași rețea pe randamente: R² în afara eșantionului ${r2}.`,
 hw1: (yr, a, b, n, h) => `Partiția ${yr}: OLS pe zilele de la ${a} la ${b} (${n} ${de(n)}zile; ultimele ${h} rînduri eliminate prin purging).`,
 hw2: (b0, bd, bw, bm, ps) => `β̂₀ = ${b0}, β̂_d = ${bd}, β̂_w = ${bw}, β̂_m = ${bm}; suma ponderilor β̂_d + β̂_w + β̂_m = ${ps}.`,
 hw3: (day, d, w, m, yh) => `La închiderea zilei de ${day}: ln v_t = ${d}, ln v_t(w) = ${w}, ln v_t(m) = ${m}; ŷ_t = ${yh}.`,
 hw4: (s2, h, vol) => `ŝ² = ${s2} (varianța reziduală din antrenare); ĥ = exp(ŷ + ŝ²/2) = ${h} %² pe zi, adică √(252 ĥ) = ${vol}% pe an.`,
 hw5: (rv, rvol, y) => `Realizat în următoarele 5 zile: RV = ${rv} (${rvol}% pe an); ținta y = ln RV = ${y}.`,
 hw: (yr, n, rmse, ps) => `Partiția ${yr}: ${n} ${de(n)}prognoze HAR, reestimate aici; rădăcina erorii pătratice medii pe log RV ${rmse}; suma ponderilor ${ps}.`,
 r21: (a, n, m, mm, b, mb) => `${a}, ${n} ${de(n)}zile de test: MSE pe log RV ${mm} (${mn(m)}) și ${mb} (${mn(b)}).`,
 r22: (mm, mb, ratio, r2) => `R²_OOS = 1 − ${mm}/${mb} = 1 − ${ratio} = ${r2}.`,
 r23: pos => (pos ? 'Pozitiv: modelul are o eroare pătratică mai mică decît reperul; curba cumulată urcă în zilele în care modelul cîștigă.' : 'Negativ: modelul este mai slab decît reperul pe aceste zile; curba cumulată coboară.'),
 r2: (a, m, b, r2) => `${a}: ${mn(m)} față de ${mn(b)}, R²_OOS = ${r2}.`,
 ql1: (rv, h, a, lh, L) => `RV = ${rv}, ĥ = ${h}: L = ${rv}/${h} + ln ${h} = ${a} + (${lh}) = ${L}.`,
 ql2: (lo, Llo, hi, Lhi) => `Asimetria: ĥ = ${lo} (jumătate din RV) costă ${Llo}, ĥ = ${hi} (dublul) costă ${Lhi}.`,
 ql3: (Lmin, mlo, mhi) => `Minimul, la ĥ = RV, este 1 + ln RV = ${Lmin}. Eroarea pătratică ordonează invers aceleași două prognoze: ${mlo} pentru jumătate, ${mhi} pentru dublu.`,
 ql: (rv, h, L, ex, x) => `RV = ${rv}, ĥ = ${h}${x ? XS : ''}: QLIKE = ${L}, cu ${ex} peste minim.`,
 dm1: (a, m, n, db) => `${a}, ${mn(m)} față de HAR: d_t = QLIKE(${mn(m)}) − QLIKE(HAR), n = ${n}; d̄ = ${db}.`,
 dm2: (g0, lag, lrv, ratio) => `Varianța lui d_t: ${g0}; varianța pe termen lung Newey–West cu ${lag} ${lag === 1 ? 'lag' : 'laguri'}: ${lrv} (${ratio} × varianța), pentru că țintele suprapuse fac d_t autocorelat.`,
 dm3: (t, p) => `DM = d̄/√(LRV/n) = ${t}; p ${pv(p)}.`,
 dm4: (t0, t) => `Fără corecția Newey–West (0 laguri) statistica ar fi ${t0}, față de ${t}: ignorarea autocorelației exagerează semnificația.`,
 dm: (a, m, lag, t, p, x) => `${a}, ${mn(m)} față de HAR, ${lag} ${lag === 1 ? 'lag' : 'laguri'} Newey–West${x ? X : ''}: DM = ${t}, p ${pv(p)}.`,
 rt: (per, lag, a, at, b, bt, g, gp, x) => `${PERN[per]}, ${lag} laguri Newey–West${x ? X : ''}. S&P 500: MLP ${a}% față de HAR (DM ${at}), lasso ${b}% (DM ${bt}). Bitcoin: gradient boosting are DM ${g} (p ${pv(gp)}).`,
 rf: (a, info, x) => `${a}${x ? X : ''}: ` + info.map(([y, r, d, f]) => `în ${y}, volatilitatea realizată atinge ${r}% (${d}); maximele prognozelor: ${f.map(([m, v]) => `${m} ${v}%`).join(', ')}`).join('; ') + '.',
 im: (f, s, n, base) => `Test 2020–2026 (${n} ${de(n)}zile). ` + s.map(([g, p, phi, lo, hi], i) => `${i ? g : cap(g)}: amestecarea crește MSE cu ${p}; media |φ| ${phi}, φ între ${lo} și ${hi}`).join('; ') + `. φ₀, prognoza medie a pădurii pe 3 variabile: ${base}.`,
 sz1: (a, n, up, acc) => `${a}: ${n} ${de(n)}zile de test, dintre care ${up} în creștere; regula „mereu creștere” are acuratețea acc₀ = ${acc}.`,
 sz2: (se, band) => `Eroarea standard √(acc₀(1 − acc₀)/n) = ${se} puncte procentuale; banda de 95%: ±${band} puncte.`,
 sz3: (m, acc, z, p) => `${cap(mn(m))}: acuratețea ${acc}, z = ${z}, p ${pv(p)}.`,
 sz4: (auc, sh) => `AUC = ${auc}; modelul prognozează „creștere” în ${sh} din zile.`,
 sz: (a, m, d, z, p) => `${a}, ${mn(m)}: ${d} puncte procentuale față de reper, z = ${z} (p ${pv(p)}).`,
 sg: (end, c, rows, x) => `Test pînă în ${end}, „creștere” dacă p ≥ ${c}${x ? X : ''}; diferența de acuratețe față de reper, în puncte procentuale (z): ` + rows.map(([a, n, l, r, g]) => (n === null ? `${a}: niciun an de test` : `${a} (${n} ${de(n)}zile): logit ${l[0]} (${l[1]}), random forest ${r[0]} (${r[1]}), boosting ${g[0]} (${g[1]})`)).join('; ') + '.',
 pl1: (a, b) => `ρ_α(u) = u(α − 1{u < 0}) cu α = ${a}: o eroare sub cuantilă (u < 0) costă ${b}|u|, una deasupra ei ${a}|u|.`,
 pl2: (n, q, sh, L) => `Cele ${n} ${de(n)}randamente de test ale S&P 500, prognoza constantă q = ${q}%: ${sh} din zile sub q; pierderea medie ${L}.`,
 pl3: (qb, sb, lb, a) => `Minimul pierderii medii: q = ${qb}%, cu ${sb} din zile sub q (ținta: ${a}) și pierderea ${lb}.`,
 pl: (a, q, sh, qb, x) => `α = ${a}, q = ${q}%${x ? XS : ''}: ${sh} din zile sub q; minimul la q = ${qb}%.`,
 vt: (per, best, bl, qr, qri, gr, grp, hs, qg, qgp, x) => `${PERN[per]}${x ? X : ''}. S&P 500: cea mai mică pierdere cuantilică are ${best} (${bl}); QR este depășit în ${qr}% din zile (p ind. ${qri}), GARCH-t în ${gr}% (p Kupiec ${grp}). Bitcoin: HS ${hs}%, QGB ${qg}% (p Kupiec ${qgp}).`,
 qv: (a, m, cnt, x) => `${a}${x ? X : ''}: ` + cnt.map(([y, c, d]) => `${y} (${d} ${de(d)}zile), depășiri: ${c.map(([mm, k]) => `${mm} ${k}`).join(', ')}`).join('; ') + `. Punctele roșii marchează depășirile ${m}.`,
 mx1: (Y, sd) => `Y = ${Y} ani: σ = 1/√${Y} = ${sd}.`,
 mx2: (N, a, b) => `N = ${N}: Φ⁻¹(1 − 1/N) = ${a}, Φ⁻¹(1 − 1/(Ne)) = ${b}.`,
 mx2b: () => 'N = 1: o singură strategie; maximul așteptat este media, 0.',
 mx3: (N, em, sim, q5, q95) => `E[max] ≈ σ[(1 − γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(Ne))] = ${em}; în simulare (2000 de repetări), cel mai bun raport Sharpe are media ${sim} și se află între ${q5} și ${q95} în 90% din cazuri.`,
 mx: (N, Y, em, sim, x) => `N = ${N}, Y = ${Y} ani${x ? X : ''}: maximul așteptat ${em}, simularea ${sim}.`,
 sn: (N, s, l, is, oos, rk, tot, cr, bi, bo, em, Y, sp, x) => `${N === 1 ? 'Prima regulă încercată' : `Primele ${N} ${de(N)}reguli încercate`}${x ? X : ''}: cea mai bună în eșantion este MA ${s}/${l} (Sharpe ${is}); după ${sp} obține ${oos}, locul ${rk} din ${tot}. Buy and hold: ${bi} și ${bo}. Cel mai bun raport Sharpe dintre ${N} ${N === 1 ? 'strategie' : de(N) + 'strategii'} fără abilitate pe ${Y} ani: în medie ${em}. Corelația dintre rapoartele Sharpe din eșantion și din afara lui (30 de reguli): ${cr}.`,
 ds1: (N, T, sr, sa, sk, ku) => `N = ${N} ${N === 1 ? 'regulă' : de(N) + 'reguli'}, T = ${T} ${de(T)}zile: SR̂ = ${sr} pe zi (× √252 = ${sa}); γ̂₃ = ${sk}, γ̂₄ = ${ku}.`,
 ds2: (sd, a, b, sr0, ann) => `SR₀ = ${sd}·[(1 − γ)·${a} + γ·${b}] = ${sr0} (anual ${ann}).`,
 ds2b: () => 'N = 1: SR₀ = 0, iar DSR coincide cu PSR.',
 ds3: (den, z, d) => `Numitorul √(1 − γ̂₃SR̂ + (γ̂₄ − 1)SR̂²/4) = ${den}; z = (SR̂ − SR₀)√(T − 1)/numitor = ${z}; DSR = Φ(z) = ${d}.`,
 ds4: (psr, sig) => `Fără corecția pentru N (PSR, SR₀ = 0): ${psr}. ${sig ? 'DSR peste 0,95: raportul Sharpe rămîne semnificativ după corecție.' : 'DSR sub 0,95: după corecția pentru numărul de încercări, raportul Sharpe nu mai este semnificativ.'}`,
 ds: (N, d, sig, mo, x) => `N = ${N}${mo === 'normal' ? ', momentele distribuției Normale' : ''}${x ? XS : ''}: DSR = ${d}, ${sig ? 'peste' : 'sub'} pragul 0,95.`,
 gk: (m, r2, top, sr, rr, rs) => `${m} (Gu, Kelly și Xiu, 2020): R²_OOS lunar ${r2}% pe toate acțiunile și ${top}% pe cele mai mari 1000; raportul Sharpe anual al diferenței dintre decile ${sr}. Locul ${rr} din 12 după R², locul ${rs} din 12 după raportul Sharpe.`,
};
})();
""",
 'en': r"""(function () {
const X = ' (extra run of the lecture code)';
const XS = ' (other values than on the slide)';
const cap = s => s[0].toUpperCase() + s.slice(1);
const pv = p => (p.startsWith('<') ? p : '= ' + p);
const pl = (n, s, p) => `${n} ${+String(n).replace(/,/g, '') === 1 ? s : p}`;
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const FLE = { d: 'log RV day', w: 'log RV week', m: 'log RV month', d1: 'log RV lag 1', d2: 'log RV lag 2', d3: 'log RV lag 3', d4: 'log RV lag 4',
  q: 'log RV quarter', r: 'return', r_neg: 'negative return', abs_r: 'absolute return' };
const PERN = { all: 'The whole test period', pre: 'Test days up to 2019', post: 'Test days from 2020' };
const GN = { relu: 'ReLU', logistic: 'logistic', tanh: 'tanh' };
window.CH13_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`, fl: g => FLE[g],
 h: { wf: ['Test year', 'Training days', 'Test days', 'Last training day', 'First purged row'], sp: ['s', 'ȳ_L', 'ȳ_R', 'SSE'], na: 'not allowed',
      rt: ['Series (days)', 'Model', 'R² against the mean', 'R² against HAR', 'QLIKE', 'DM (p)'],
      im: ['Feature', 'MSE increase', 'Standard deviation', 'Mean |φ|'], vt: ['Series (days)', 'Method', 'x', 'Rate (%)', 'p Kupiec', 'p ind.', 'Loss × 100', 'Mean VaR (%)'] },
 bv: (s2, d, b2, v, te, tr, best, bte, x) => `Simulation with σ² = ${s2}, n = 80 and 300 training samples${x ? X : ''}. Depth ${d}: bias² ${b2}, variance ${v}; the expected test error, bias² + variance + σ², is ${te}, and the training error ${tr}. The test error is lowest at depth ${best} (${bte}).`,
 cs: (k, nb, nt, size, last, h, e, ntr, x) => `K = ${k}, ${pl(h, 'row', 'rows')} purged, an embargo of ${pl(e, 'row', 'rows')}${x ? X : ''}. Random K-fold, fold 1: ${nb} of the ${nt} test observations have a neighbour (t − 1 or t + 1) in training. Walk-forward: training grows from ${size} to ${last} observations, with test blocks of ${size}. Purged K-fold, middle fold: ${ntr} training rows out of 60.`,
 wf1: (yr, a, b, n) => `Fold ${yr}: the model is estimated on the days from ${a} to ${b}, that is ${n} days (expanding window).`,
 wf2: (h, fp, yr) => (+h === 1 ? `Purging: the last training row, ${fp}, is removed: its target, the variance proxy of the next day, falls in ${yr}.` : `Purging: the last ${h} training rows, from ${fp}, are removed: their target, the mean variance proxy over the next ${h} days, reaches into ${yr}.`),
 wf3: (n, yr) => `Test: the ${n} trading days of ${yr}; the next fold adds them to training. No embargo is needed, because the test block always comes after training.`,
 wf: (yr, ntr, nte, h, nf, x) => `Horizon h = ${h}${x ? X : ''}: ${nf} folds; fold ${yr} trains on ${ntr} days and tests on ${nte} days.`,
 lk: (h, sk, sw, sp, pk, pw, pp, nS, nP, x) => `Target: the cumulative return over the next ${h} days${x ? X : ''}. Simulated random walk (${nS} days): random K-fold ${sk}, walk-forward ${sw}, purged walk-forward ${sp}. S&P 500 (${nP} days): ${pk}, ${pw} and ${pp}.`,
 sh: (lamL, nz, fl, cl, co, lamR, cr, ratio, cv, nzcv) => (cv
   ? `Lasso at λ = ${lamL}, chosen by walk-forward cross-validation: ${nzcv} of 11 coefficients non-zero; ${fl}: ${cl} (OLS ${co}). `
   : `Lasso at λ = ${lamL}: ${nz} of 11 coefficients non-zero; ${fl}: ${cl} (OLS ${co}). `) + `Ridge at λ = ${lamR}: ${fl} ${cr}; the norm of the coefficient vector is ${ratio} of the OLS norm. The OLS coefficients of r, min(r, 0) and |r| are not unique, because |r| = r − 2 min(r, 0) (the matrix has rank 10): the minimum-norm solution is shown.`,
 rw1: (sxx, sxy, b) => `OLS: β̂ = S_xy/S_xx = ${sxy}/${sxx} = ${b}.`,
 rw2: (lam, sxx, sxy, b, f) => `Ridge with λ = ${lam}: β̂ = ${sxy}/(${sxx} + ${lam}) = ${b}; shrinkage factor S_xx/(S_xx + λ) = ${f}.`,
 rw3: (bl, lam) => `Lasso with the same λ = ${lam}: ${bl}. Ridge shrinks the coefficient towards zero but does not set it to zero for any finite λ.`,
 rw: (lam, b, f, x) => `λ = ${lam}${x ? XS : ''}: ridge β̂ = ${b}, that is ${f} of the OLS coefficient.`,
 lw1: (sxy, lam, half) => `The threshold: λ/2 = ${half}; |S_xy| = ${Math.abs(sxy)}.`,
 lw2: (sxy, lam, nm, b) => `Lasso β̂ = sign(S_xy) · max(|S_xy| − λ/2, 0)/S_xx = ${sxy < 0 ? '−' : '+'}${nm}/100 = ${b}.`,
 lw3: lim => `The coefficient is exactly 0 for λ ≥ 2|S_xy| = ${lim}.`,
 lw: (lam, sxy, b, x) => `λ = ${lam}, S_xy = ${sxy}${x ? XS : ''}: lasso β̂ = ${b}.`,
 sp1: (yb, s0) => `No split: ȳ = ${yb}, SSE = ${s0}.`,
 sp2: (s, nL, nR, yL, yR, e, ok, ml) => (ok ? `Threshold s = ${s}: ${pl(nL, 'week', 'weeks')} on the left (ȳ_L = ${yL}), ${nR} on the right (ȳ_R = ${yR}); SSE = ${e}.` : `Threshold s = ${s} leaves fewer than ${ml} observations in a leaf: not allowed.`),
 sp3: (bs, be, s0, sh) => `The best allowed split: s = ${bs}, SSE = ${be}; it explains 1 − ${be}/${s0} = ${sh} of the variation.`,
 sp: (s, e, sh, bs, x) => (e === null ? `Threshold s = ${s} is not allowed${x ? XS : ''}; the best split: s = ${bs}.` : `Threshold s = ${s}${x ? XS : ''}: SSE = ${e}, that is ${sh} of the variation explained; the best split: s = ${bs}.`),
 tr: (dep, ml, f, thr, vol, lv, r2, n, x) => `Depth ${dep}, at least ${ml} days in a leaf${x ? X : ''}; ${n} days (2008–2012). First split: ${f} ≤ ${thr}, an annual volatility of about ${vol}%. Leaves (mean target, days): ${lv.map(([v, k]) => `${v} (${k})`).join('; ')}. The tree explains ${r2} of the variation of the target in the sample.`,
 bg1: (rho, B) => `B = ${pl(B, 'tree', 'trees')}, each with variance σ² = 1 and correlation ρ = ${rho} between any two.`,
 bg2: (rho, B, a, b, v) => `Var(mean) = ρσ² + (1 − ρ)σ²/B = ${a} + ${b} = ${v}.`,
 bg3: (rho, v1, v10, v300) => `For ρ = ${rho}: B = 1: ${v1}; B = 10: ${v10}; B = 300: ${v300}. The part ρσ², shared by all trees, does not fall with B.`,
 bg: (B, rho, v, x) => `B = ${B}, ρ = ${rho}${x ? XS : ''}: the variance of the average is ${v}.`,
 gb1: (f0, s0) => `F₀ = ȳ = ${f0}; training SSE ${s0}.`,
 gb2: (s, a, b) => `Step 1: the residuals y − F₀; the best depth-1 tree on the residuals splits at s = ${s}, with values ${a} (left) and ${b} (right).`,
 gb3: (nu, a, b, s1) => `F₁ = F₀ + ν h₁, with ν = ${nu}: ${a} on the left and ${b} on the right; SSE = ${s1}.`,
 gb4: (M, e, a, b) => `After ${pl(M, 'step', 'steps')}: SSE = ${e}; the fit runs from ${a} (first week) to ${b} (last week).`,
 gb: (nu, M, e) => `Extra computation on the six weeks of the worked example, with depth-1 trees: ν = ${nu}, ${pl(M, 'step', 'steps')}, training SSE ${e}.`,
 en: (n, B, d, rf, bd, bv, td, gb, har, ntr, nva, x) => `Training 2008–2018 (${ntr} days), validation 2019–2022 (${nva} days)${x ? X : ''}. Random forest with ${pl(B, 'tree', 'trees')}: validation MSE ${rf} at depth ${d} (single tree: ${td}); lowest ${bv} at depth ${bd}. Boosting after ${pl(n, 'tree', 'trees')}: ` + gb.map(([l, v, b, k]) => `learning rate ${l}: ${v} (lowest ${b} after ${pl(k, 'tree', 'trees')})`).join('; ') + `. HAR: ${har}.`,
 mw1: (p, K, pk, k, hid) => `Hidden layer: ${p} × ${K} weights + ${K} constants = ${hid}.`,
 mw2: (K, o) => `Output: ${K} weights + 1 constant = ${o}.`,
 mw3: t => `In total: ${t} parameters.`,
 mw: (p, K, t, x) => `${p} inputs, ${K} hidden neurons${x ? XS : ''}: ${t} parameters.`,
 ml: (g, z, gz, dg, K, np) => `g = ${GN[g]}: g(${z}) = ${gz}, derivative g′(${z}) = ${dg}` + (+dg < 0.05 && g !== 'relu' ? ' (saturation: the gradient almost vanishes)' : g === 'relu' && +z <= 0 ? ' (ReLU is 0 for z ≤ 0, with gradient 0)' : '') + `. The network on the left, with 3 inputs and ${pl(K, 'hidden neuron', 'hidden neurons')}, has ${np} parameters.`,
 na: (per, u, mR, mG, xR, dR, xG, dG, cR, cM, n, x) => `${per === 'all' ? '2010–2026' : per} (${n} days), RBF network with ${u} units${x ? X : ''}: mean annualised volatility ${mR}% (GARCH ${mG}%); maximum of the network ${xR}% (${dR}), of GARCH ${xG}% (${dG}). Correlation with the GARCH volatility: ${cR} (RBF network), ${cM} (MLP).`,
 nj: (sh, split, rtr, wtr, rte, wte, ratio, mtr, mte, r2, x) => `Training on the first ${sh}% of the days, test from ${split}${x ? X : ''}. RMSE in yen: training ${rtr} (random walk ${wtr}), test ${rte} (random walk ${wte}): out of sample the network is ${ratio} times worse. Maximum in training ${mtr}, in the test ${mte}. The same network on returns: out-of-sample R² ${r2}.`,
 hw1: (yr, a, b, n, h) => `Fold ${yr}: OLS on the days from ${a} to ${b} (${n} days; the last ${h} rows purged).`,
 hw2: (b0, bd, bw, bm, ps) => `β̂₀ = ${b0}, β̂_d = ${bd}, β̂_w = ${bw}, β̂_m = ${bm}; sum of the weights β̂_d + β̂_w + β̂_m = ${ps}.`,
 hw3: (day, d, w, m, yh) => `At the close of ${day}: ln v_t = ${d}, ln v_t(w) = ${w}, ln v_t(m) = ${m}; ŷ_t = ${yh}.`,
 hw4: (s2, h, vol) => `ŝ² = ${s2} (residual variance in training); ĥ = exp(ŷ + ŝ²/2) = ${h} %² per day, that is √(252 ĥ) = ${vol}% per year.`,
 hw5: (rv, rvol, y) => `Realised over the next 5 days: RV = ${rv} (${rvol}% per year); the target y = ln RV = ${y}.`,
 hw: (yr, n, rmse, ps) => `Fold ${yr}: ${n} HAR forecasts, refitted here; root mean squared error on log RV ${rmse}; sum of the weights ${ps}.`,
 r21: (a, n, m, mm, b, mb) => `${a}, ${n} test days: MSE on log RV ${mm} (${m}) and ${mb} (${b}).`,
 r22: (mm, mb, ratio, r2) => `R²_OOS = 1 − ${mm}/${mb} = 1 − ${ratio} = ${r2}.`,
 r23: pos => (pos ? 'Positive: the model has a smaller squared error than the benchmark; the cumulative curve rises on the days the model wins.' : 'Negative: the model is worse than the benchmark on these days; the cumulative curve falls.'),
 r2: (a, m, b, r2) => `${a}: ${m} against ${b}, R²_OOS = ${r2}.`,
 ql1: (rv, h, a, lh, L) => `RV = ${rv}, ĥ = ${h}: L = ${rv}/${h} + ln ${h} = ${a} + (${lh}) = ${L}.`,
 ql2: (lo, Llo, hi, Lhi) => `The asymmetry: ĥ = ${lo} (half of RV) costs ${Llo}, ĥ = ${hi} (double) costs ${Lhi}.`,
 ql3: (Lmin, mlo, mhi) => `The minimum, at ĥ = RV, is 1 + ln RV = ${Lmin}. The squared error ranks the same two forecasts the other way: ${mlo} for half, ${mhi} for double.`,
 ql: (rv, h, L, ex, x) => `RV = ${rv}, ĥ = ${h}${x ? XS : ''}: QLIKE = ${L}, ${ex} above the minimum.`,
 dm1: (a, m, n, db) => `${a}, ${m} against HAR: d_t = QLIKE(${m}) − QLIKE(HAR), n = ${n}; d̄ = ${db}.`,
 dm2: (g0, lag, lrv, ratio) => `Variance of d_t: ${g0}; Newey–West long-run variance with ${pl(lag, 'lag', 'lags')}: ${lrv} (${ratio} × the variance), because the overlapping targets make d_t autocorrelated.`,
 dm3: (t, p) => `DM = d̄/√(LRV/n) = ${t}; p ${pv(p)}.`,
 dm4: (t0, t) => `Without the Newey–West correction (0 lags) the statistic would be ${t0}, against ${t}: ignoring the autocorrelation overstates the significance.`,
 dm: (a, m, lag, t, p, x) => `${a}, ${m} against HAR, ${pl(lag, 'Newey–West lag', 'Newey–West lags')}${x ? X : ''}: DM = ${t}, p ${pv(p)}.`,
 rt: (per, lag, a, at, b, bt, g, gp, x) => `${PERN[per]}, ${lag} Newey–West lags${x ? X : ''}. S&P 500: MLP ${a}% against HAR (DM ${at}), lasso ${b}% (DM ${bt}). Bitcoin: gradient boosting has DM ${g} (p ${pv(gp)}).`,
 rf: (a, info, x) => `${a}${x ? X : ''}: ` + info.map(([y, r, d, f]) => `in ${y} the realised volatility peaks at ${r}% (${d}); peaks of the forecasts: ${f.map(([m, v]) => `${m} ${v}%`).join(', ')}`).join('; ') + '.',
 im: (f, s, n, base) => `Test 2020–2026 (${n} days). ` + s.map(([g, p, phi, lo, hi], i) => `${i ? g : cap(g)}: shuffling raises the MSE by ${p}; mean |φ| ${phi}, φ between ${lo} and ${hi}`).join('; ') + `. φ₀, the mean forecast of the forest on 3 features: ${base}.`,
 sz1: (a, n, up, acc) => `${a}: ${n} test days, ${up} of them up; the rule “always up” has accuracy acc₀ = ${acc}.`,
 sz2: (se, band) => `Standard error √(acc₀(1 − acc₀)/n) = ${se} percentage points; 95% band: ±${band} points.`,
 sz3: (m, acc, z, p) => `${cap(m)}: accuracy ${acc}, z = ${z}, p ${pv(p)}.`,
 sz4: (auc, sh) => `AUC = ${auc}; the model forecasts “up” on ${sh} of the days.`,
 sz: (a, m, d, z, p) => `${a}, ${m}: ${d} percentage points against the baseline, z = ${z} (p ${pv(p)}).`,
 sg: (end, c, rows, x) => `Test up to ${end}, “up” if p ≥ ${c}${x ? X : ''}; accuracy minus the baseline, in percentage points (z): ` + rows.map(([a, n, l, r, g]) => (n === null ? `${a}: no test year` : `${a} (${n} days): logit ${l[0]} (${l[1]}), random forest ${r[0]} (${r[1]}), boosting ${g[0]} (${g[1]})`)).join('; ') + '.',
 pl1: (a, b) => `ρ_α(u) = u(α − 1{u < 0}) with α = ${a}: an error below the quantile (u < 0) costs ${b}|u|, one above it ${a}|u|.`,
 pl2: (n, q, sh, L) => `The ${n} S&P 500 test returns, constant forecast q = ${q}%: ${sh} of the days below q; mean loss ${L}.`,
 pl3: (qb, sb, lb, a) => `Minimum of the mean loss: q = ${qb}%, with ${sb} of the days below q (target: ${a}) and loss ${lb}.`,
 pl: (a, q, sh, qb, x) => `α = ${a}, q = ${q}%${x ? XS : ''}: ${sh} of the days below q; the minimum at q = ${qb}%.`,
 vt: (per, best, bl, qr, qri, gr, grp, hs, qg, qgp, x) => `${PERN[per]}${x ? X : ''}. S&P 500: the lowest quantile loss is that of ${best} (${bl}); QR is exceeded on ${qr}% of the days (p ind. ${qri}), GARCH-t on ${gr}% (p Kupiec ${grp}). Bitcoin: HS ${hs}%, QGB ${qg}% (p Kupiec ${qgp}).`,
 qv: (a, m, cnt, x) => `${a}${x ? X : ''}: ` + cnt.map(([y, c, d]) => `${y} (${d} days), exceedances: ${c.map(([mm, k]) => `${mm} ${k}`).join(', ')}`).join('; ') + `. The red dots mark the exceedances of ${m}.`,
 mx1: (Y, sd) => `Y = ${Y} years: σ = 1/√${Y} = ${sd}.`,
 mx2: (N, a, b) => `N = ${N}: Φ⁻¹(1 − 1/N) = ${a}, Φ⁻¹(1 − 1/(Ne)) = ${b}.`,
 mx2b: () => 'N = 1: a single strategy; the expected maximum is the mean, 0.',
 mx3: (N, em, sim, q5, q95) => `E[max] ≈ σ[(1 − γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(Ne))] = ${em}; in the simulation (2,000 repetitions) the best Sharpe ratio has mean ${sim} and lies between ${q5} and ${q95} in 90% of the cases.`,
 mx: (N, Y, em, sim, x) => `N = ${N}, Y = ${Y} years${x ? X : ''}: expected maximum ${em}, simulation ${sim}.`,
 sn: (N, s, l, is, oos, rk, tot, cr, bi, bo, em, Y, sp, x) => `${N === 1 ? 'The first rule tried' : `The first ${N} rules tried`}${x ? X : ''}: the best in sample is MA ${s}/${l} (Sharpe ${is}); after ${sp} it earns ${oos}, rank ${rk} of ${tot}. Buy and hold: ${bi} and ${bo}. Best Sharpe ratio of ${pl(N, 'strategy', 'strategies')} without skill over ${Y} years: ${em} on average. Correlation of the in-sample and out-of-sample Sharpe ratios (30 rules): ${cr}.`,
 ds1: (N, T, sr, sa, sk, ku) => `N = ${pl(N, 'rule', 'rules')}, T = ${T} days: SR̂ = ${sr} per day (× √252 = ${sa}); γ̂₃ = ${sk}, γ̂₄ = ${ku}.`,
 ds2: (sd, a, b, sr0, ann) => `SR₀ = ${sd}·[(1 − γ)·${a} + γ·${b}] = ${sr0} (annual ${ann}).`,
 ds2b: () => 'N = 1: SR₀ = 0, and the DSR equals the PSR.',
 ds3: (den, z, d) => `Denominator √(1 − γ̂₃SR̂ + (γ̂₄ − 1)SR̂²/4) = ${den}; z = (SR̂ − SR₀)√(T − 1)/denominator = ${z}; DSR = Φ(z) = ${d}.`,
 ds4: (psr, sig) => `Without the correction for N (PSR, SR₀ = 0): ${psr}. ${sig ? 'DSR above 0.95: the Sharpe ratio stays significant after the correction.' : 'DSR below 0.95: after the correction for the number of trials the Sharpe ratio is no longer significant.'}`,
 ds: (N, d, sig, mo, x) => `N = ${N}${mo === 'normal' ? ', moments of the Normal distribution' : ''}${x ? XS : ''}: DSR = ${d}, ${sig ? 'above' : 'below'} the threshold 0.95.`,
 gk: (m, r2, top, sr, rr, rs) => `${m} (Gu, Kelly and Xiu, 2020): monthly R²_OOS ${r2}% on all stocks and ${top}% on the largest 1,000; annualised Sharpe ratio of the decile spread ${sr}. Rank ${rr} of 12 by R², rank ${rs} of 12 by Sharpe ratio.`,
};
})();
""",
}

QA = {
 'ro': {
  'bv': ('Cu varianța zgomotului σ² = 0,04 în loc de 0,09 (rulare suplimentară), la ce adîncime este minimă eroarea de test așteptată?',
         'La adîncimea 4 (0,063), față de adîncimea 3 (0,132) cu σ² = 0,09. Cu mai puțin zgomot, varianța arborilor adînci costă mai puțin, deci optimul se mută spre un model mai '
         'flexibil; cu σ² = 0,25 minimul rămîne la adîncimea 3 (0,329).'),
  'cvs': ('K-fold aleator, partiția 1 (K = 5): din cele 12 observații de test, cîte au cel puțin o vecină (ziua dinainte sau ziua de după) în antrenare?',
          '11 din 12. Partiția aleatoare împrăștie testul printre zilele de antrenare, iar cu ținte suprapuse vecina are aproape aceeași țintă: modelul a văzut practic răspunsul. '
          'Cu K = 8 situația este aceeași, 7 din 8.'),
  'wf-we': ('Cu ținta pe 22 de zile (h = 22, rulare suplimentară), cîte zile de antrenare are partiția 2013?',
            '1171: purging-ul elimină 22 de rînduri în loc de 5, deci cu 17 mai puține decît cele 1188 de zile cu h = 5; ultima zi de antrenare devine 28 noiembrie 2012. '
            'Testul rămîne cele 252 de zile din 2013.'),
  'leak': ('Cu ținta pe 5 zile în loc de 21 (rulare suplimentară), ce R² raportează K-fold aleator pentru mersul aleator simulat?',
           '17,8%, față de 34,5% cu 21 de zile: țintele a două zile vecine au acum 4 din 5 randamente comune, nu 20 din 21, deci leakage-ul este mai mic, dar tot mare. '
           'Walk-forward rămîne negativ (−17,5%); cu 63 de zile, K-fold aleator urcă la 35,8%.'),
  'ridge-we': ('▶ crește λ de la 50 la 200. Cît devine β̂ ridge?',
               '0,10 = 30/(100 + 200); factorul de contracție scade de la 0,67 la 0,33. Coeficientul se apropie de zero, dar nu devine zero pentru niciun λ finit; lasso cu '
               'λ = 200 este deja 0.'),
  'lasso-we': ('▶ crește λ de la 20 la 60. Cît devine β̂ lasso?',
               'Exact 0: λ/2 = 30 = |S_xy|, deci max(30 − 30, 0) = 0, și rămîne 0 pentru orice λ ≥ 60. Ridge cu același λ ar da 30/160 = 0,19.'),
  'shrink': ('▶ mărește penalizarea lasso pînă la λ = 1. Care variabilă rămîne ultima cu coeficient nenul?',
             'Log RV săptămînal: rămîne nenul pînă la λ = 0,86 pe grilă, după log RV lunar (0,46), log RV zilnic (0,21) și randamentul negativ (0,18). Luată singură, varianța săptămînală este cea mai informativă variabilă; ridge, în stînga, nu anulează niciun coeficient.'),
  'split-we': ('Ce pondere din variație explică pragul s = 1,75, al doilea cel mai bun din tabel?',
               '0,78 = 1 − 0,505/2,268, față de 0,90 pentru s = 1,25. Pragul 1,75 trece săptămîna cu x = 1,5 (y = 1,6) în frunza din stînga, unde media urcă la 1,00 și abaterile cresc.'),
  'tree': ('▶ trece la adîncimea 3 (rulare suplimentară). Folosesc toate împărțirile tot log RV săptămînal?',
           'Nu: 5 din cele 6 împărțiri folosesc log RV săptămînal, una folosește log RV lunar (≤ 0,069), în ramura cu volatilitate săptămînală moderată. Prima împărțire rămîne '
           'log RV săptămînal ≤ 0,44, pentru că un arbore greedy nu își revizuiește alegerile de sus.'),
  'rf-we': ('▶ crește B de la 10 la 300, cu ρ = 0,3 și σ² = 1. Cît devine varianța mediei?',
            '0,302 = 0,3 + 0,7/300, față de 0,370 la B = 10. Partea comună ρσ² = 0,3 nu dispare oricîți arbori am adăuga; pragul coboară doar dacă arborii devin mai puțin corelați.'),
  'gb-we': ('Calcul suplimentar, ν = 0,1: după cîți pași scade SSE de antrenare sub jumătate din 2,268?',
            'După 4 pași: SSE scade la 1,880, 1,566, 1,312 și 1,101. Fiecare pas aplică doar ν = 10% din ajustarea arborelui pe reziduurile curente, deci eroarea scade geometric, nu liniar. După 50 de pași SSE = 0,007: pe datele de antrenare boosting ajunge aproape la zero, de aceea numărul de pași se alege pe validare.'),
  'ens': ('▶ reduce pădurea la un singur arbore (B = 1). Cît este MSE de validare la adîncimea 8?',
          '0,670, mai mult decît arborele simplu de aceeași adîncime (0,631) și mult peste 0,458 cu 200 de arbori. Fiecare arbore al pădurii vede un eșantion bootstrap și doar '
          'jumătate din variabile la o împărțire, deci singur este mai slab decît un arbore obișnuit; media îi elimină varianța, iar cu 10 arbori MSE este deja 0,478.'),
  'mlp-we': ('Cu 11 intrări și K = 32 de neuroni ascunși, cîți parametri are rețeaua?',
             '417 = 11 × 32 + 32 + 32 + 1. Nu se dublează față de 209: fiecare neuron în plus adaugă p + 2 = 13 parametri, iar constanta ieșirii nu depinde de K.'),
  'mlp': ('Pentru z = 4, care funcție de activare are cel mai mare gradient g′(z)?',
          'ReLU: g′(4) = 1. Logistica are g′(4) = 0,018, iar tanh 0,0013: ambele se saturează pentru |z| mare, deci gradientul care trece prin neuron aproape dispare. '
          'De aceea rețelele cu ReLU se antrenează mai repede.'),
  'nnarch': ('Cu 5 unități RBF în loc de 3 (rulare suplimentară), crește sau scade corelația volatilității rețelei cu GARCH (0,45 cu 3 unități)?',
             'Crește la 0,61, iar maximul rețelei urcă de la 22,0% la 27,6%; cu 2 unități corelația scade la 0,26. Mai multe unități dau o formă mai flexibilă, dar rețeaua vede tot '
             'doar ultimele 3 randamente: maximul GARCH de după votul pentru Brexit, 37,3%, rămîne departe.'),
  'nnjpy': ('Cu 90% din zile la antrenare (rulare suplimentară), de cîte ori este RMSE de test al rețelei mai mare decît al mersului aleator?',
            'De 13 ori (12,12 față de 0,91 yeni), față de 31 de ori cu 80%. Maximul din antrenare urcă la 151,6, dar testul ajunge la 163,9: rețeaua tot nu poate extrapola '
            'dincolo de nivelurile văzute.'),
  'har-we': ('Partiția 2020, ziua cu prognoza HAR maximă (18 martie 2020): este prognoza peste sau sub volatilitatea realizată în următoarele 5 zile?',
             'Puțin peste: 69,3% față de 66,1% pe an. Maximul volatilității realizate în 2020, 102,4%, a venit însă înaintea vîrfului prognozei: HAR reacționează după șoc, nu îl anticipează.'),
  'r2-we': ('▶ trece la random forest (S&P 500, față de HAR). Cît este R²_OOS?',
            '+1,0%: MSE 0,429 față de 0,433. Față de media de antrenare, același model are 61,9%: R²_OOS depinde de reper, iar HAR este un reper mult mai exigent. Testul DM nu '
            'separă acest cîștig de zgomot (p = 0,747).'),
  'qlike-we': ('RV = 1: care prognoză costă mai mult în QLIKE, ĥ = 1/3 sau ĥ = 3?',
               'ĥ = 1/3: 3 + ln(1/3) = 1,901, față de 1/3 + ln 3 = 1,432. Subestimarea de trei ori costă mai mult decît supraestimarea de trei ori, deși eroarea pătratică ordonează '
               'invers: (1 − 1/3)² = 0,444 și (1 − 3)² = 4.'),
  'dm-we': ('▶ coboară numărul de laguri Newey–West de la 5 la 0. Ce se întîmplă cu statistica DM a MLP față de HAR (S&P 500, −3,96)?',
            'Crește în valoare absolută, la −4,61: varianța pe termen lung cu 5 laguri este de 1,35 ori varianța simplă, pentru că țintele pe 5 zile se suprapun și fac d_t autocorelat. '
            'Fără corecție testul pare mai sigur decît este; aici concluzia (MLP mai bun) rămîne aceeași.'),
  'rv-tab': ('▶ trece la zilele de test din 2020. Mai este MLP semnificativ mai bun decît HAR pe S&P 500 la 5%?',
             'Da: R²_OOS +7,9%, DM −2,28 (p = 0,023); lasso +7,4%, DM −3,42. Random forest și boosting nu sînt semnificativ mai bune decît HAR în nicio subperioadă: pînă în 2019 au '
             '−1,5% și −11,0%, din 2020 +3,5% și +5,8%, cu p peste 0,6.'),
  'rvf': ('În 2020, este maximul prognozei random forest peste sau sub maximul volatilității realizate (102% pe an)?',
          'Mult sub: 53%, față de 69% pentru HAR și 159% pentru MLP. Un random forest prezice medii ale frunzelor din antrenare, deci rămîne sub valorile extreme; MLP și lasso '
          '(172%) extrapolează și pot depăși ținta.'),
  'imp': ('▶ afișează doar log RV zilnic. Este intervalul valorilor Shapley mai larg sau mai îngust decît la log RV săptămînal?',
          'Mai îngust: φ zilnic între −0,32 și 0,35, față de −0,66 și 2,25 pentru săptămînă; media |φ| este 0,13, față de 0,52. Pădurea se bazează pe varianța săptămînală, care '
          'conține deja o parte din informația zilei.'),
  'sign-we': ('BET în loc de S&P 500, random forest: cît este statistica z față de reper?',
              '−0,21 (53,6% față de 53,8%, p = 0,84). Pe S&P 500 random forest are z = −1,79, pe Bitcoin −0,99: pe nicio piață un model nu este semnificativ mai bun decît „mereu creștere”.'),
  'sign': ('▶ testează doar pînă în 2012 (5 ani). Este vreun model semnificativ mai slab decît reperul pe S&P 500?',
           'Da: gradient boosting, −3,9 puncte, z = −2,77; logit și random forest au z = −1,36 și −1,30. Banda de 95% este mai largă cu mai puține zile, totuși boosting rămîne în '
           'afara ei: modelul învață tipare din zgomotul anilor de antrenare.'),
  'pin-we': ('Cu α = 0,05, ce prognoză constantă q minimizează pierderea medie pe randamentele de test ale S&P 500?',
             'q = −1,59%, cuantila empirică de 5%: 4,99% din zile se află sub ea. Pentru α = 0,01 minimul este la −3,07% (0,99% din zile sub el). Media randamentelor, 0,05%, '
             'minimizează eroarea pătratică, nu pierderea cuantilică.'),
  'var-tab': ('▶ trece la zilele de test din 2020. Respinge testul Kupiec la 5% rata depășirilor GARCH-t pe S&P 500?',
              'Da: 1,60% (27 din 1686 de zile), p = 0,022. QR are 0,71% (p = 0,21), iar depășirile HS apar grupat (p ind. 0,005). Pentru Bitcoin din 2020, QR este depășit prea rar: '
              '0,45%, p Kupiec 0,002.'),
  'qvar': ('În 2025 (pînă în septembrie), care metodă are cele mai multe depășiri pe S&P 500?',
           'HS: 5 depășiri, față de 4 pentru GARCH-t și cîte 1 pentru QR și QGB. După 500 de zile liniștite, simularea istorică intră în aprilie 2025 cu un VaR prea mic și reacționează lent.'),
  'maxsr-we': ('Cu 20 de ani de date în loc de 10, cît este maximul așteptat pentru N = 100?',
               '0,566 = 0,800/√2, pentru că σ = 1/√20 = 0,224; simularea dă 0,559. Mai mulți ani reduc distorsiunea de selecție doar cu √Y: cu N = 1000 maximul este tot 0,728.'),
  'snoop': ('▶ încearcă doar primele 10 reguli. Care este cea mai bună în eșantion și ce loc ocupă după 2014?',
            'MA 5/200 (Sharpe 0,57 în eșantion), cu 0,74 după 2014, locul 3 din 30. Cu toate cele 30 de reguli, cîștigătoarea, MA 50/200 (0,66), ajunge pe locul 22: alegerea '
            'dintre mai multe reguli ridică raportul Sharpe din eșantion, nu pe cel viitor.'),
  'dsr-we': ('Pînă la ce număr de reguli încercate rămîne cea mai bună regulă MA semnificativă (DSR > 0,95)?',
             'Pînă la N = 6: DSR = 0,961 la N = 5, 0,955 la N = 6 și 0,949 la N = 7; la N = 30, 0,884. Fără corecție (PSR) regula ar părea semnificativă, cu 0,994: SR₀ crește lent cu N, dar SR̂ = 0,0416 pe zi are o margine mică.'),
  'gkx': ('Este modelul cu cel mai mare R²_OOS pe toate acțiunile și modelul cu cel mai mare raport Sharpe același?',
          'Nu: NN3 are cel mai mare R² (0,40%), NN4 cel mai mare raport Sharpe (1,35, față de 1,20 pentru NN3). Diferențele de R² dintre rețele sînt de cîteva sutimi de punct '
          'procentual, iar clasamentul după valoarea economică poate diferi.'),
 },
 'en': {
  'bv': ('With noise variance σ² = 0.04 instead of 0.09 (extra run), at which depth is the expected test error lowest?',
         'At depth 4 (0.063), against depth 3 (0.132) with σ² = 0.09. With less noise the variance of deep trees costs less, so the optimum moves to a more flexible model; '
         'with σ² = 0.25 the minimum stays at depth 3 (0.329).'),
  'cvs': ('Random K-fold, fold 1 (K = 5): of its 12 test observations, how many have at least one neighbour (the day before or the day after) in training?',
          '11 of 12. A random fold scatters the test days among the training days, and with overlapping targets the neighbour has almost the same target: the model has practically '
          'seen the answer. With K = 8 it is the same, 7 of 8.'),
  'wf-we': ('With a 22-day target (h = 22, extra run), how many training days does the 2013 fold have?',
            '1,171: the purging removes 22 rows instead of 5, so 17 fewer than the 1,188 days with h = 5; the last training day becomes 28 November 2012. The test remains the 252 days of 2013.'),
  'leak': ('With a 5-day target instead of 21 (extra run), what R² does random K-fold report for the simulated random walk?',
           '17.8%, against 34.5% with 21 days: the targets of two neighbouring days now share 4 of 5 returns, not 20 of 21, so the leakage is smaller but still large. '
           'Walk-forward stays negative (−17.5%); with 63 days random K-fold rises to 35.8%.'),
  'ridge-we': ('▶ raises λ from 50 to 200. What is the ridge β̂ now?',
               '0.10 = 30/(100 + 200); the shrinkage factor falls from 0.67 to 0.33. The coefficient approaches zero but is not zero for any finite λ; the lasso with λ = 200 is already 0.'),
  'lasso-we': ('▶ raises λ from 20 to 60. What is the lasso β̂ now?',
               'Exactly 0: λ/2 = 30 = |S_xy|, so max(30 − 30, 0) = 0, and it stays 0 for any λ ≥ 60. Ridge with the same λ would give 30/160 = 0.19.'),
  'shrink': ('▶ raises the lasso penalty up to λ = 1. Which feature is the last with a non-zero coefficient?',
             'Log RV week: it stays non-zero up to λ = 0.86 on the grid, after log RV month (0.46), log RV day (0.21) and the negative return (0.18). On its own, the weekly variance is the most informative feature; ridge, on the left, sets no coefficient to zero.'),
  'split-we': ('What share of the variation does the threshold s = 1.75, the second best in the table, explain?',
               '0.78 = 1 − 0.505/2.268, against 0.90 for s = 1.25. The threshold 1.75 moves the week with x = 1.5 (y = 1.6) into the left leaf, whose mean rises to 1.00 and whose deviations grow.'),
  'tree': ('▶ moves to depth 3 (extra run). Do all splits still use log RV week?',
           'No: 5 of the 6 splits use log RV week, one uses log RV month (≤ 0.069), in the branch with moderate weekly volatility. The first split stays log RV week ≤ 0.44, '
           'because a greedy tree does not revise its choices at the top.'),
  'rf-we': ('▶ raises B from 10 to 300, with ρ = 0.3 and σ² = 1. What is the variance of the average now?',
            '0.302 = 0.3 + 0.7/300, against 0.370 at B = 10. The shared part ρσ² = 0.3 does not vanish however many trees are added; the floor falls only if the trees become less correlated.'),
  'gb-we': ('Extra computation, ν = 0.1: after how many steps does the training SSE fall below half of 2.268?',
            'After 4 steps: the SSE falls to 1.880, 1.566, 1.312 and 1.101. Each step applies only ν = 10% of the tree fitted to the current residuals, so the error falls geometrically, not linearly. After 50 steps the SSE is 0.007: on the training data boosting gets close to zero, which is why the number of steps is chosen on validation data.'),
  'ens': ('▶ reduces the forest to a single tree (B = 1). What is the validation MSE at depth 8?',
          '0.670, more than the plain tree of the same depth (0.631) and far above 0.458 with 200 trees. Each tree of the forest sees a bootstrap sample and only half of the '
          'features at a split, so alone it is weaker than an ordinary tree; the average removes its variance, and with 10 trees the MSE is already 0.478.'),
  'mlp-we': ('With 11 inputs and K = 32 hidden neurons, how many parameters does the network have?',
             '417 = 11 × 32 + 32 + 32 + 1. It does not double from 209: each extra neuron adds p + 2 = 13 parameters, and the output constant does not depend on K.'),
  'mlp': ('At z = 4, which activation function has the largest gradient g′(z)?',
          'ReLU: g′(4) = 1. The logistic has g′(4) = 0.018 and tanh 0.0013: both saturate for large |z|, so the gradient passing through the neuron almost vanishes. '
          'This is why ReLU networks train faster.'),
  'nnarch': ('With 5 RBF units instead of 3 (extra run), does the correlation of the network volatility with GARCH (0.45 with 3 units) rise or fall?',
             'It rises to 0.61, and the maximum of the network goes from 22.0% to 27.6%; with 2 units the correlation falls to 0.26. More units give a more flexible shape, but the '
             'network still sees only the last 3 returns: the GARCH peak after the Brexit vote, 37.3%, stays far away.'),
  'nnjpy': ('With 90% of the days for training (extra run), how many times larger is the test RMSE of the network than that of the random walk?',
            '13 times (12.12 against 0.91 yen), against 31 times with 80%. The training maximum rises to 151.6, but the test reaches 163.9: the network still cannot extrapolate '
            'beyond the levels it has seen.'),
  'har-we': ('Fold 2020, the day with the highest HAR forecast (18 March 2020): is the forecast above or below the volatility realised over the next 5 days?',
             'Slightly above: 69.3% against 66.1% per year. The peak of the realised volatility in 2020, 102.4%, came before the peak of the forecast: HAR reacts after the shock, it does not anticipate it.'),
  'r2-we': ('▶ moves to the random forest (S&P 500, against HAR). What is R²_OOS?',
            '+1.0%: MSE 0.429 against 0.433. Against the training mean the same model has 61.9%: R²_OOS depends on the benchmark, and HAR is a much tougher one. The DM test does '
            'not separate this gain from noise (p = 0.747).'),
  'qlike-we': ('RV = 1: which forecast costs more in QLIKE, ĥ = 1/3 or ĥ = 3?',
               'ĥ = 1/3: 3 + ln(1/3) = 1.901, against 1/3 + ln 3 = 1.432. Underestimating three times costs more than overestimating three times, although the squared error ranks them '
               'the other way: (1 − 1/3)² = 0.444 and (1 − 3)² = 4.'),
  'dm-we': ('▶ lowers the number of Newey–West lags from 5 to 0. What happens to the DM statistic of the MLP against HAR (S&P 500, −3.96)?',
            'It grows in absolute value, to −4.61: the long-run variance with 5 lags is 1.35 times the plain variance, because the 5-day targets overlap and make d_t autocorrelated. '
            'Without the correction the test looks surer than it is; here the conclusion (MLP better) stays the same.'),
  'rv-tab': ('▶ moves to the test days from 2020. Is the MLP still significantly better than HAR on the S&P 500 at 5%?',
             'Yes: R²_OOS +7.9%, DM −2.28 (p = 0.023); lasso +7.4%, DM −3.42. The random forest and boosting are not significantly better than HAR in either subperiod: up to 2019 '
             'they have −1.5% and −11.0%, from 2020 +3.5% and +5.8%, with p above 0.6.'),
  'rvf': ('In 2020, is the peak of the random-forest forecast above or below the peak of the realised volatility (102% per year)?',
          'Far below: 53%, against 69% for HAR and 159% for the MLP. A random forest predicts averages of training leaves, so it stays below extreme values; the MLP and the lasso '
          '(172%) extrapolate and can overshoot the target.'),
  'imp': ('▶ shows log RV day only. Is the range of its Shapley values wider or narrower than for log RV week?',
          'Narrower: daily φ between −0.32 and 0.35, against −0.66 and 2.25 for the week; mean |φ| 0.13 against 0.52. The forest relies on the weekly variance, which already '
          'contains part of the information of the day.'),
  'sign-we': ('BET instead of the S&P 500, random forest: what is the z statistic against the baseline?',
              '−0.21 (53.6% against 53.8%, p = 0.84). On the S&P 500 the random forest has z = −1.79, on Bitcoin −0.99: on no market is a model significantly better than “always up”.'),
  'sign': ('▶ tests only up to 2012 (5 years). Is any model significantly worse than the baseline on the S&P 500?',
           'Yes: gradient boosting, −3.9 points, z = −2.77; logit and the random forest have z = −1.36 and −1.30. The 95% band is wider with fewer days, yet boosting stays outside it: '
           'the model learns patterns from the noise of the training years.'),
  'pin-we': ('With α = 0.05, which constant forecast q minimises the mean loss on the S&P 500 test returns?',
             'q = −1.59%, the empirical 5% quantile: 4.99% of the days lie below it. For α = 0.01 the minimum is at −3.07% (0.99% of the days below). The mean return, 0.05%, '
             'minimises the squared error, not the quantile loss.'),
  'var-tab': ('▶ moves to the test days from 2020. Does the Kupiec test reject the exceedance rate of GARCH-t on the S&P 500 at 5%?',
              'Yes: 1.60% (27 of 1,686 days), p = 0.022. QR has 0.71% (p = 0.21), and the HS exceedances cluster (p ind. 0.005). For Bitcoin from 2020, QR is exceeded too rarely: '
              '0.45%, p Kupiec 0.002.'),
  'qvar': ('In 2025 (up to September), which method has the most exceedances on the S&P 500?',
           'HS: 5 exceedances, against 4 for GARCH-t and 1 each for QR and QGB. After 500 calm days, historical simulation enters April 2025 with a VaR that is too small and reacts slowly.'),
  'maxsr-we': ('With 20 years of data instead of 10, what is the expected maximum for N = 100?',
               '0.566 = 0.800/√2, because σ = 1/√20 = 0.224; the simulation gives 0.559. More years reduce the selection bias only by √Y: with N = 1,000 the maximum is still 0.728.'),
  'snoop': ('▶ tries only the first 10 rules. Which is the best in sample, and what is its rank after 2014?',
            'MA 5/200 (Sharpe 0.57 in sample), with 0.74 after 2014, rank 3 of 30. With all 30 rules the winner, MA 50/200 (0.66), ends at rank 22: choosing among more rules '
            'raises the in-sample Sharpe ratio, not the future one.'),
  'dsr-we': ('Up to which number of rules tried does the best MA rule stay significant (DSR > 0.95)?',
             'Up to N = 6: DSR = 0.961 at N = 5, 0.955 at N = 6 and 0.949 at N = 7; at N = 30, 0.884. Without the correction (PSR) the rule would look significant, with 0.994: SR₀ grows slowly with N, but SR̂ = 0.0416 per day has a thin margin.'),
  'gkx': ('Is the model with the highest R²_OOS on all stocks also the one with the highest Sharpe ratio?',
          'No: NN3 has the highest R² (0.40%), NN4 the highest Sharpe ratio (1.35, against 1.20 for NN3). The R² differences between the networks are a few hundredths of a '
          'percentage point, and the ranking by economic value can differ.'),
 },
}

PREDICT = {
 'bv': (0, [('La adîncimea 4.', 'At depth 4.'), ('Tot la 3: adîncimea optimă nu depinde de zgomot.', 'Still at 3: the best depth does not depend on the noise.'),
            ('La adîncimea 2: mai puțin zgomot cere un model mai simplu.', 'At depth 2: less noise calls for a simpler model.')]),
 'cvs': (0, [('11.', '11.'), ('0: partițiile sînt disjuncte, deci testul nu are vecini în antrenare.', '0: the folds are disjoint, so the test has no neighbours in training.'),
             ('2: doar zilele de la marginile blocului de test.', '2: only the days at the edges of the test block.')]),
 'wf-we': (0, [('1171.', '1,171.'), ('1188: purging-ul elimină mereu 5 rînduri.', '1,188: purging always removes 5 rows.'),
               ('1166 = 1188 − 22: se scad 22 de rînduri din cifra pentru h = 5.', '1,166 = 1,188 − 22: subtract 22 rows from the figure for h = 5.')]),
 'leak': (0, [('Circa 18%: mai puțină suprapunere, mai puțin leakage.', 'About 18%: less overlap, less leakage.'), ('Circa 0%: cu 5 zile nu mai există suprapunere.', 'About 0%: with 5 days there is no overlap left.'),
              ('Tot circa 34,5%: R² nu depinde de orizont.', 'Still about 34.5%: R² does not depend on the horizon.')]),
 'ridge-we': (0, [('0,10.', '0.10.'), ('0: la un λ suficient de mare ridge anulează coeficientul.', '0: at a large enough λ ridge sets the coefficient to zero.'),
                  ('0,05 = 0,20 × 50/200: coeficientul scade proporțional cu 1/λ.', '0.05 = 0.20 × 50/200: the coefficient falls in proportion to 1/λ.')]),
 'lasso-we': (0, [('0.', '0.'), ('−0,30 = (30 − 60)/100: fără max(·, 0) și fără λ/2.', '−0.30 = (30 − 60)/100: without max(·, 0) and without λ/2.'),
                  ('0,19 = 30/(100 + 60), ca la ridge.', '0.19 = 30/(100 + 60), as for ridge.')]),
 'shrink': (0, [('log RV săptămînal.', 'log RV week.'), ('log RV zilnic: informația cea mai recentă.', 'log RV day: the most recent information.'),
                ('Randamentul negativ: efectul de levier.', 'The negative return: the leverage effect.')]),
 'split-we': (0, [('0,78.', '0.78.'), ('0,22 = 0,505/2,268: raportul SSE, fără 1 −.', '0.22 = 0.505/2.268: the SSE ratio, without 1 −.'),
                  ('0,505: SSE-ul citit ca pondere explicată.', '0.505: the SSE read as the share explained.')]),
 'tree': (0, [('Nu: una folosește log RV lunar.', 'No: one uses log RV month.'), ('Da: varianța săptămînală este cea mai informativă, deci o folosesc toate.', 'Yes: the weekly variance is the most informative, so all use it.'),
              ('Nu: prima împărțire trece la log RV zilnic.', 'No: the first split switches to log RV day.')]),
 'rf-we': (0, [('0,302.', '0.302.'), ('0,003 = 1/300: media a 300 de arbori are varianța σ²/B.', '0.003 = 1/300: the average of 300 trees has variance σ²/B.'),
               ('0: cu destui arbori varianța dispare.', '0: with enough trees the variance vanishes.')]),
 'gb-we': (0, [('4.', '4.'), ('1: primul pas face deja împărțirea cea mai bună.', '1: the first step already makes the best split.'),
               ('5: fiecare pas elimină 10% din SSE-ul inițial.', '5: each step removes 10% of the initial SSE.')]),
 'ens': (0, [('Circa 0,67.', 'About 0.67.'), ('0,458: numărul de arbori nu schimbă eroarea.', '0.458: the number of trees does not change the error.'),
             ('Sub 0,458: mai puțini arbori înseamnă mai puțin overfitting.', 'Below 0.458: fewer trees mean less overfitting.')]),
 'mlp-we': (0, [('417.', '417.'), ('418 = 2 × 209: parametrii sînt proporționali cu K.', '418 = 2 × 209: the parameters are proportional to K.'),
                ('384 = 11 × 32 + 32: doar stratul ascuns.', '384 = 11 × 32 + 32: the hidden layer only.')]),
 'mlp': (0, [('ReLU.', 'ReLU.'), ('tanh: are cea mai mare pantă în 0.', 'tanh: it has the steepest slope at 0.'), ('Logistica: g′(z) = g(1 − g) nu se anulează niciodată.', 'The logistic: g′(z) = g(1 − g) never reaches zero.')]),
 'nnarch': (0, [('Crește, la circa 0,6.', 'It rises, to about 0.6.'), ('Scade: mai multe unități înseamnă overfitting.', 'It falls: more units mean overfitting.'),
                ('Rămîne 0,45: numărul de unități nu schimbă memoria rețelei.', 'It stays 0.45: the number of units does not change the memory of the network.')]),
 'nnjpy': (0, [('Circa 13.', 'About 13.'), ('Circa 1: cu mai multe date rețeaua ajunge la mersul aleator.', 'About 1: with more data the network catches up with the random walk.'),
               ('Circa 31, ca la 80%: raportul nu depinde de împărțire.', 'About 31, as with 80%: the ratio does not depend on the split.')]),
 'har-we': (0, [('Puțin peste: 69% față de 66%.', 'Slightly above: 69% against 66%.'), ('Mult sub: HAR nu depășește niciodată volatilitatea realizată.', 'Far below: HAR never exceeds the realised volatility.'),
                ('Sub, la jumătate: prognoza urmează media lunară, nu ultima săptămînă.', 'Below, at half: the forecast follows the monthly mean, not the last week.')]),
 'r2-we': (0, [('+1,0%.', '+1.0%.'), ('+61,9%: R² față de medie.', '+61.9%: the R² against the mean.'), ('+8,3%, ca MLP: modelele flexibile au același cîștig.', '+8.3%, as the MLP: flexible models have the same gain.')]),
 'qlike-we': (0, [('ĥ = 1/3.', 'ĥ = 1/3.'), ('ĥ = 3: abaterea absolută este mai mare (2 față de 0,67).', 'ĥ = 3: the absolute deviation is larger (2 against 0.67).'),
                  ('Costă la fel: ambele sînt greșite de trei ori.', 'They cost the same: both are three times off.')]),
 'dm-we': (0, [('Crește în valoare absolută, la circa −4,6.', 'It grows in absolute value, to about −4.6.'), ('Rămîne −3,96: lagurile schimbă doar p-value-ul.', 'It stays −3.96: the lags change only the p-value.'),
               ('Scade spre 0: fără laguri testul are mai puțină informație.', 'It falls towards 0: without lags the test has less information.')]),
 'rv-tab': (0, [('Da: DM circa −2,3.', 'Yes: DM about −2.3.'), ('Nu: după 2020 HAR este mai bun decît toate modelele.', 'No: after 2020 HAR beats all models.'),
                ('Nu se poate spune: testul DM nu se aplică pe subperioade.', 'Cannot tell: the DM test does not apply to subperiods.')]),
 'rvf': (0, [('Mult sub: circa 53%.', 'Far below: about 53%.'), ('Peste: modelele flexibile reacționează cel mai puternic la șocuri.', 'Above: flexible models react most strongly to shocks.'),
             ('Aproape egal: random forest urmărește bine ținta în crize.', 'About equal: the random forest tracks the target well in crises.')]),
 'imp': (0, [('Mai îngust.', 'Narrower.'), ('Mai larg: ziua cea mai recentă contează cel mai mult.', 'Wider: the most recent day matters most.'),
             ('La fel: valorile Shapley se împart egal între variabile.', 'The same: Shapley values are split equally between the features.')]),
 'sign-we': (0, [('Circa −0,2.', 'About −0.2.'), ('Peste 1,96: piețele mai mici sînt mai ușor de prognozat.', 'Above 1.96: smaller markets are easier to forecast.'),
                 ('−1,79, ca pe S&P 500: z nu depinde de piață.', '−1.79, as on the S&P 500: z does not depend on the market.')]),
 'sign': (0, [('Da: gradient boosting, z ≈ −2,8.', 'Yes: gradient boosting, z ≈ −2.8.'), ('Nu: pe 5 ani niciun rezultat nu poate fi semnificativ.', 'No: over 5 years no result can be significant.'),
              ('Da: logit, cel mai simplu model.', 'Yes: the logit, the simplest model.')]),
 'pin-we': (0, [('Circa −1,6%.', 'About −1.6%.'), ('Media randamentelor, circa 0,05%.', 'The mean return, about 0.05%.'), ('−3,07%, ca la α = 0,01: minimul nu depinde de α.', '−3.07%, as for α = 0.01: the minimum does not depend on α.')]),
 'var-tab': (0, [('Da: p ≈ 0,02.', 'Yes: p ≈ 0.02.'), ('Nu: după 2020 toate metodele trec testul.', 'No: after 2020 all methods pass the test.'),
                 ('Nu se aplică: testul Kupiec cere toată perioada.', 'It does not apply: the Kupiec test needs the whole period.')]),
 'qvar': (0, [('HS.', 'HS.'), ('QR: o regresie liniară reacționează cel mai încet.', 'QR: a linear regression reacts most slowly.'), ('GARCH-t: cozile t subestimează pierderile mari.', 'GARCH-t: the t tails underestimate large losses.')]),
 'maxsr-we': (0, [('0,57.', '0.57.'), ('0,40 = 0,80/2: distorsiunea scade cu 1/Y.', '0.40 = 0.80/2: the bias falls with 1/Y.'), ('0,80: maximul depinde doar de N.', '0.80: the maximum depends only on N.')]),
 'snoop': (0, [('MA 5/200, locul 3.', 'MA 5/200, rank 3.'), ('MA 50/200, locul 22: cîștigătoarea nu depinde de N.', 'MA 50/200, rank 22: the winner does not depend on N.'),
               ('MA 5/50, locul 1: prima regulă încercată rămîne cea mai bună.', 'MA 5/50, rank 1: the first rule tried stays the best.')]),
 'dsr-we': (0, [('6.', '6.'), ('30: regula a fost aleasă dintre 30, deci corecția o lasă semnificativă.', '30: the rule was chosen among 30, so the correction leaves it significant.'),
                ('Niciunul: chiar și PSR (0,994) este sub 0,95.', 'None: even the PSR (0.994) is below 0.95.')]),
 'gkx': (0, [('Nu: NN3 după R², NN4 după raportul Sharpe.', 'No: NN3 by R², NN4 by Sharpe ratio.'), ('Da: NN3 la ambele.', 'Yes: NN3 on both.'),
             ('Nu: random forest după R², NN4 după raportul Sharpe.', 'No: the random forest by R², NN4 by Sharpe ratio.')]),
}

# blocks whose initial chart shows the answer: veil the readout and those charts
VEIL = {a: {'out': True} for a in PREDICT}
VEIL.update({'cvs': {'out': True, 'charts': ['cs-chart']}, 'shrink': {'out': True, 'charts': ['sh-chart']}, 'rvf': {'out': True, 'charts': ['rf-chart']},
             'imp': {'out': True, 'charts': ['im-chart', 'im-tab']}, 'qvar': {'out': True, 'charts': ['qv-chart']}, 'gkx': {'out': True, 'charts': ['gk-chart']},
             'split-we': {'out': True, 'charts': ['sp-we-box']}})

# Exit ticket „Verifică-te” / "Check Yourself": the slides „Autoevaluare” and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Un model are un R² de antrenare de 80% și un R² în afara eșantionului de −5%. Ce s-a întîmplat?',
         'A model has a training R² of 80% and an out-of-sample R² of −5%. What happened?'),
      key=0, opts=[
  (('Overfitting: modelul a învățat zgomotul eșantionului de antrenare.', 'Overfitting: the model learned the noise of the training sample.'),
   ('Eroare mică pe antrenare și mare pe date noi; pe date noi modelul pierde în fața reperului.', 'Small error in training, large on new data; on new data the model loses against the benchmark.')),
  (('Datele de test sînt greșite: un R² nu poate fi negativ.', 'The test data are wrong: an R² cannot be negative.'),
   ('R² în afara eșantionului poate fi negativ: înseamnă o eroare pătratică mai mare decît a reperului.', 'The out-of-sample R² can be negative: it means a larger squared error than the benchmark.')),
  (('Underfitting: modelul este prea rigid.', 'Underfitting: the model is too rigid.'),
   ('Un model prea rigid ar avea și pe antrenare o eroare mare, nu un R² de 80%.', 'A model that is too rigid would also have a large training error, not an R² of 80%.'))]),
 dict(q=('De ce este greșit K-fold aleator pentru o țintă de tip randament pe 21 de zile?', 'Why is random K-fold wrong for a 21-day return target?'),
      key=0, opts=[
  (('Țintele vecine se suprapun, deci răspunsurile de test ajung în antrenare.', 'Neighbouring targets overlap, so the test answers leak into training.'),
   ('Ziua t + 1 are 20 din 21 de randamente comune cu ziua t; folosiți walk-forward cu purging.', 'Day t + 1 shares 20 of 21 returns with day t; use walk-forward with purging.')),
  (('Cu 5 partiții eșantionul de test este prea mic.', 'With 5 folds the test sample is too small.'),
   ('Mărimea partițiilor nu este problema: pe un mers aleator, K-fold aleator raportează R² = 34,5% din leakage.', 'The size of the folds is not the problem: on a random walk, random K-fold reports R² = 34.5% from leakage.')),
  (('Nu este greșit: amestecarea face observațiile independente.', 'It is not wrong: shuffling makes the observations independent.'),
   ('Amestecarea nu schimbă dependența dintre ținte, doar o ascunde.', 'Shuffling does not change the dependence between the targets, it only hides it.'))]),
 dict(q=('Un clasificator are dreptate în 53% din zile, iar piața a crescut în 54% dintre ele. Este util?',
         'A classifier is right on 53% of the days, and the market rose on 54% of them. Is it useful?'),
      key=0, opts=[
  (('Nu: „mereu creștere” are dreptate în 54% din zile.', 'No: “always up” is right on 54% of the days.'),
   ('Reperul corect este clasa majoritară, nu 50%; pe S&P 500 random forest are 53,1% față de 54,4%.', 'The right baseline is the majority class, not 50%; on the S&P 500 the random forest has 53.1% against 54.4%.')),
  (('Da: 53% este peste 50%.', 'Yes: 53% is above 50%.'),
   ('50% este reperul unei monede, nu al unei piețe care crește în 54% din zile.', '50% is the baseline of a coin, not of a market that rises on 54% of the days.')),
  (('Da, dacă este semnificativ față de 50%.', 'Yes, if it is significant against 50%.'),
   ('Testul trebuie făcut față de reperul 54%, unde diferența este negativă.', 'The test must be against the 54% baseline, where the difference is negative.'))]),
 dict(q=('Ridge cu o singură variabilă: S_xx = 100, S_xy = 30, λ = 50. Cît este β̂?', 'Ridge with one variable: S_xx = 100, S_xy = 30, λ = 50. What is β̂?'),
      key=0, opts=[
  (('0,20 = 30/(100 + 50).', '0.20 = 30/(100 + 50).'), ('Ridge: S_xy/(S_xx + λ); factorul de contracție este 100/150 = 0,67.', 'Ridge: S_xy/(S_xx + λ); the shrinkage factor is 100/150 = 0.67.')),
  (('0,30 = 30/100.', '0.30 = 30/100.'), ('Acesta este coeficientul OLS: penalizarea λ lipsește.', 'This is the OLS coefficient: the penalty λ is missing.')),
  (('0,05 = (30 − 25)/100.', '0.05 = (30 − 25)/100.'), ('Aceasta este formula lasso (soft thresholding cu λ/2 = 25), nu ridge.', 'This is the lasso formula (soft thresholding with λ/2 = 25), not ridge.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch13-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH13_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH13_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH13_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', '\u00a0', t)
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
