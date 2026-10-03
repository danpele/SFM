r"""
build_seminar14.py -- Seminarul 14 (Active cripto: criptomonede și stablecoins), EN + RO dintr-o singură sursă
=============================================================================================================
Seminarul are loc ÎNAINTEA cursului 14: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu inferență și o întrebare de interpretare, C o întrebare
deschisă și critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în
versiunea profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_14/sem14_results.json (seminar14.py).
Ieșire:
  EN/Seminars/seminar14_crypto_assets.tex          (+ _solutions.tex)
  RO/Seminarii/seminar14_active_cripto_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_14/seminar14.py && python3 latex/build_seminar14.py && python3 latex/sfm_build.py compile 14
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, items, table, fig   # noqa: E402
from ch14_common import REFS, T, bib, load_sem, pv, money, d   # noqa: E402

S = load_sem()
V = Values()
D = Deck(14, 'seminar', refs=REFS)


def qlsem():
    return '\\sfmquantlet{Ch_14}{SFM_ch14_seminar}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for key in ('a1', 'a2', 'a1sp'):
    a = A[key]
    for c in ['vol_true', 'vol_wrong', 'mean_true', 'mean_wrong', 'under_vol', 'under_mean']:
        V.put(f'{key}.{c}', a[c], 1)
    for c in ['sharpe', 'sharpe_mixed', 'ratio', 'sqrt_true', 'sqrt_wrong']:
        V.put(f'{key}.{c}', a[c], 3 if c == 'ratio' else 2)
for key in ('a3', 'a4'):
    a = A[key]
    V.raw(f'{key}.dd', '; '.join(f'$⁅{x:.1f}⁆$' for x in a['dd']))
    V.raw(f'{key}.max', '; '.join(f'⁅{x:g}⁆' for x in a['max']))
    V.put(f'{key}.mdd', a['mdd'], 1)
    V.put(f'{key}.need', a['gain_needed'], 2)
    V.raw(f'{key}.rec', str(a['recovery_index'] + 1))
for key in ('a5', 'a6'):
    a = A[key]
    for day, v in a['close'].items():
        V.put(f'{key}.c{day[-2:]}', v, 4)
        V.put(f'{key}.b{day[-2:]}', a['bp'][day], 0)
    V.put(f'{key}.low', a['low'], 4)
    V.put(f'{key}.lowbp', a['low_bp'], 0)
    V.raw(f'{key}.lossc', money(a['loss_close']))
    V.raw(f'{key}.lossl', money(a['loss_low']))
for key in ('a7n', 'a8n'):
    a = A[key]
    for c in ['var1', 'es25', 'var_h', 'zs', 'fs']:
        V.put(f'{key}.{c}', a[c], 2)
    for c in ['var1_usd', 'es25_usd', 'var_h_usd']:
        V.raw(f'{key}.{c}', money(a[c]))
for key in ('a7', 'a8'):
    a = A[key]
    d(V, f'{key}.first', a['first'])
    d(V, f'{key}.last', a['last'])
    V.raw(f'{key}.list', '; '.join(f'$⁅{x:.2f}⁆$' for x in a['worst']))
    V.put(f'{key}.var1', a['var1'], 2)
    V.put(f'{key}.es25', a['es25'], 2)
    V.put(f'{key}.sum', a['sum'], 2)
    V.put(f'{key}.ev', a['e_var1'], 4)
    V.put(f'{key}.ee', a['e_es25'], 4)
    V.raw(f'{key}.var1u', money(a['var1_usd']))
    V.raw(f'{key}.es25u', money(a['es25_usd']))
    V.raw(f'{key}.k1', str(a['k1']))
    V.raw(f'{key}.k25', str(a['k25']))
V.put('z1', -stats.norm.ppf(0.01), 3)
V.put('phi25', stats.norm.pdf(stats.norm.ppf(0.025)), 4)


def put_desc(key, s):
    V.int(f'{key}.n', s['n'])
    V.raw(f'{key}.y0', s['first'][:4])
    V.put(f'{key}.mean', s['mean'], 3)
    V.put(f'{key}.sd', s['sd'], 2)
    V.put(f'{key}.skew', s['skew'], 2)
    V.put(f'{key}.k', s['exkurt'], 1)
    V.put(f'{key}.ppy', s['ppy'], 1)
    for c in ['ann_vol', 'ann_vol_252', 'ann_mean', 'ann_mean_252', 'share5']:
        V.put(f'{key}.{c}', s[c], 1)
    for c in ['hill_left', 'hill_left_lo', 'hill_left_hi', 'hill_right', 'hill_right_lo', 'hill_right_hi']:
        V.put(f'{key}.{c}', s[c], 2)
    V.raw(f'{key}.kh', str(s['k']))


for k, s in S['B1'].items():
    put_desc(f'b1.{k}', s)
for k, s in S['B2'].items():
    put_desc(f'b2.{k}', s)


def put_corr(key, c):
    V.int(f'{key}.n', c['n'])
    V.int(f'{key}.npre', c['n_pre'])
    V.int(f'{key}.npost', c['n_post'])
    for f in ['pre', 'post', 'ci_lo', 'ci_hi', 'weekly_all', 'daily_all', 'bad_mean', 'bad_mean_b']:
        V.put(f'{key}.{f}', c[f], 2)
    V.put(f'{key}.z', c['z'], 2)
    V.raw(f'{key}.p', pv(c['p']))
    V.raw(f'{key}.badn', str(c['bad_n']))
    for y in ('2020', '2022', '2023', '2025'):
        V.put(f'{key}.y{y}', c['year'][y], 2)
        V.put(f'{key}.w{y}', c['week_year'][y], 2)
    for lab, v in c['crisis'].items():
        lk = lab.split('/')[0].split('-')[0].lower()
        for a, x in v.items():
            V.put(f'{key}.cr.{lk}.{a}', x, 1)


put_corr('b3', S['B3'])
put_corr('b4g', S['B4']['btc_gold'])
put_corr('b4e', S['B4']['eth_sp500'])


def put_peg(key, p):
    V.int(f'{key}.n', p['n'])
    for c in ['sd', 'mad', 'gt10', 'sd_2023', 'sd_after', 'mad_after']:
        V.put(f'{key}.{c}', p[c], 1)
    V.put(f'{key}.gt50', p['gt50'], 2)
    for c in ['close_min', 'close_max', 'low_min']:
        V.put(f'{key}.{c}', p[c], 0)
    d(V, f'{key}.cmind', p['close_min_date'])
    d(V, f'{key}.cmaxd', p['close_max_date'])
    V.put(f'{key}.ar', p['ar'], 3)
    V.put(f'{key}.arse', p['ar_se'], 3)
    V.put(f'{key}.hl', p['ar_hl'], 2)
    for i, (lab, v) in enumerate(p['bins'].items()):
        V.put(f'{key}.bin{i}', v, 1)


put_peg('b5', S['B5'])
put_peg('b6t', S['B6']['usdt'])
put_peg('b6d', S['B6']['dai'])


def put_var(key, b):
    v = b['var']
    V.int(f'{key}.n', v['n'])
    V.raw(f'{key}.y0', v['first'][:4])
    for m, mk in (('HS', 'hs'), ('Normal', 'n'), ('Student-t', 't'), ('GARCH-t', 'g')):
        V.put(f'{key}.{mk}.v', v[m]['var'], 2)
        V.put(f'{key}.{mk}.e', v[m]['es'], 2)
        V.raw(f'{key}.{mk}.vu', money(v[m]['var_usd']))
        V.raw(f'{key}.{mk}.eu', money(v[m]['es_usd']))
    V.put(f'{key}.nu', v['nu'], 2)
    V.put(f'{key}.gnu', v['g_nu'], 2)
    V.put(f'{key}.gs', v['g_sigma'], 2)
    bt = b['bt']
    V.int(f'{key}.bt.n', bt['n'])
    d(V, f'{key}.bt.start', bt['start'])
    for m, mk in (('HS', 'hs'), ('GARCH-t', 'g')):
        r = bt[m]
        V.raw(f'{key}.bt.{mk}.x', str(r['x']))
        V.put(f'{key}.bt.{mk}.rate', r['rate'], 2)
        V.put(f'{key}.bt.{mk}.exp', r['exp'], 1)
        V.put(f'{key}.bt.{mk}.lr', r['lr'], 2)
        V.put(f'{key}.bt.{mk}.p', r['p'], 2)
        V.put(f'{key}.bt.{mk}.mv', r['mean_var'], 1)
        V.raw(f'{key}.bt.{mk}.ymax', str(max(r['year'].values())))
        V.raw(f'{key}.bt.{mk}.yarg', max(r['year'], key=r['year'].get))
    V.put(f'{key}.gap', 100 * (1 - v['Normal']['var'] / v['HS']['var']), 0)


put_var('b7', S['B7'])
put_var('b8', S['B8'])
C1 = S['C1']
for c in ['vol_before', 'vol_after', 'we_share_before', 'we_share_after']:
    V.put(f'c1.{c}', C1[c], 1)
for c in ['we_ratio_before', 'we_ratio_after', 'corr_before', 'corr_after', 'p_bf', 'pers_after']:
    V.put(f'c1.{c}', C1[c], 2)
V.put('c1.pers_before', C1['pers_before'], 3)
C2 = S['C2']
V.put('c2.v252', C2['vol252'], 1)
V.put('c2.v365', C2['vol365'], 1)
V.put('c2.var', C2['hs_var'], 2)
V.raw('c2.varu', money(C2['hs_var_usd']))
V.raw('c2.varwrong', money(1000 * C2['hs_var']))
V.put('c2.bp', C2['usdc_bp'], 0)
V.put('c2.bpwrong', -C2['usdc_bp'] / 100, 2)
V.put('c2.pre', C2['pre2020'], 2)
V.put('c2.post', C2['post2020'], 2)
V.put('c2.worst', C2['worst']['mean'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how risky is a crypto position, and how stable is a ``stable\'\' coin?', '\\textbf{Întrebarea}: cît de riscantă este o poziție cripto și cît de stabil este un stablecoin?'),
     [T('this seminar comes \\textbf{before} Lecture 14: the section ``What You Need for Today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 14: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: annualisation with 365 days, drawdowns, peg deviations in basis points, VaR of a crypto position, on paper',
        'Partea A: anualizarea cu 365 de zile, drawdown-uri, abateri de la paritate în puncte de bază, VaR al unei poziții cripto, pe hîrtie'),
      T('Part B: tails, correlation with equities, the peg of USDC and a backtest of Bitcoin VaR on real data, each with an interpretation question',
        'Partea B: cozile, corelația cu acțiunile, paritatea USDC și backtesting-ul VaR pentru Bitcoin, pe date reale, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    T('The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class',
      'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise Map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('annualising a crypto series with 365 and with 252 days', 'anualizarea unei serii cripto cu 365 și cu 252 de zile') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('the maximum drawdown of a price path and the gain needed to recover', 'drawdown-ul maxim al unei traiectorii de preț și cîștigul necesar pentru recuperare') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('peg deviations of USDC and DAI in March 2023, in basis points', 'abaterile de la paritate ale USDC și DAI în martie 2023, în puncte de bază') + ' & ' + SP + ' & A5',
     'A7, A8 & ' + T('VaR 1\\% and ES 2.5\\% of a Bitcoin and of an Ethereum position', 'VaR 1\\% și ES 2,5\\% ale unei poziții în Bitcoin și în Ethereum') + ' & ' + SP + ' & A7',
     'B1, B2 & ' + T('moments, annualisation and tail index: Bitcoin and S\\&P 500; Ethereum and Solana', 'momente, anualizare și indicele de coadă: Bitcoin și S\\&P 500; Ethereum și Solana') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('correlation in calm and crisis periods: Bitcoin and S\\&P 500; Bitcoin and gold, Ethereum and S\\&P 500', 'corelația în perioade calme și de criză: Bitcoin și S\\&P 500; Bitcoin și aur, Ethereum și S\\&P 500') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('the peg of USDC; USDT and DAI', 'paritatea USDC; USDT și DAI') + ' & ' + SP + ' & B5',
     'B7, B8 & ' + T('VaR, ES and a backtest: Bitcoin; Ethereum', 'VaR, ES și backtesting: Bitcoin; Ethereum') + ' & ' + SP + ' & B7',
     'C1, C2 & ' + T('did the spot ETFs change Bitcoin? what is wrong in an AI answer?', 'au schimbat ETF-urile spot comportamentul Bitcoin? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='scriptsize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați voi, după model')), 'footnotesize')

D.frame(T('Data Used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Price used}', '\\textbf{Prețul folosit}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['Bitcoin, Ethereum, Solana & EODHD & ' + T('close, 7 days a week', 'închidere, 7 zile pe săptămînă') + ' & 2014/2016/2020--2026',
     'USDT, USDC, DAI & EODHD & ' + T('close and daily low', 'închidere și minimul zilei') + ' & 2021--2026',
     T('S\\&P 500, gold (XAU/USD)', 'S\\&P 500, aur (XAU/USD)') + ' & EODHD & ' + T('close, weekdays', 'închidere, zile lucrătoare') + ' & 2014--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, each series on its own calendar; the last day is 18 September 2026',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, fiecare serie pe propriul calendar; ultima zi este 18 septembrie 2026'),
    T('Two series together: join the prices on common days first, then compute returns', 'Două serii împreună: întîi unim prețurile în zilele comune, apoi calculăm randamentele'),
    T('In the notebook: \\texttt{returns(\'btc\')}, \\texttt{joint\\_returns(\'btc\', \'sp500\')}, \\texttt{describe(k)}, \\texttt{peg\\_stats(\'usdc\')}; no account or key is needed',
      'În notebook: \\texttt{returns(\'btc\')}, \\texttt{joint\\_returns(\'btc\', \'sp500\')}, \\texttt{describe(k)}, \\texttt{peg\\_stats(\'usdc\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What You Need for Today', 'Noțiuni necesare azi')

D.frame(T('What You Need for Today (1/4): Crypto Assets and Annualisation', 'Noțiuni necesare azi (1/4): activele cripto și anualizarea'), items(
    (T('\\textbf{Crypto asset}: a digital asset whose ownership is recorded on a \\textbf{blockchain}, a public chain of blocks of transactions', '\\textbf{Activ cripto}: un activ digital a cărui proprietate este înregistrată într-un \\textbf{blockchain}, un lanț public de blocuri de tranzacții'),
     [T('Bitcoin (2009): fixed supply of 21 million coins; Ethereum (2015): runs programs (smart contracts)', 'Bitcoin (2009): ofertă fixă de 21 de milioane de monede; Ethereum (2015): execută programe (contracte inteligente)'),
      T('traded on \\textbf{exchanges} (trading platforms) 24 hours a day, 7 days a week', 'tranzacționate pe \\textbf{exchange-uri} (platforme de tranzacționare) 24 de ore din 24, 7 zile pe săptămînă')]),
    (T('\\textbf{Annualisation} with $m$ observations a year (i.i.d.\\ returns): mean $m\\,\\bar r$, volatility $\\sqrt{m}\\,s$', '\\textbf{Anualizarea} cu $m$ observații pe an (randamente i.i.d.): media $m\\,\\bar r$, volatilitatea $\\sqrt{m}\\,s$'),
     [T('crypto: $m = 365$, $\\sqrt{365} = 19.105$; equities: $m = 252$, $\\sqrt{252} = 15.875$', 'cripto: $m = 365$, $\\sqrt{365} = 19{,}105$; acțiuni: $m = 252$, $\\sqrt{252} = 15{,}875$'),
      T('\\textbf{Sharpe ratio} (risk-free rate 0): annual mean / annual volatility $= \\sqrt{m}\\,\\bar r/s$', '\\textbf{Raportul Sharpe} (rata fără risc 0): media anuală / volatilitatea anuală $= \\sqrt{m}\\,\\bar r/s$')]),
    T('Hill tail index (Chapter 5): $\\hat\\alpha = [\\frac1k\\sum_{i \\le k}\\ln(X_{(i)}/X_{(k+1)})]^{-1}$ on the $k$ largest losses; 95\\% CI $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$; smaller $\\alpha$, heavier tail',
      'Indicele de coadă Hill (Capitolul 5): $\\hat\\alpha = [\\frac1k\\sum_{i \\le k}\\ln(X_{(i)}/X_{(k+1)})]^{-1}$ pe cele mai mari $k$ pierderi; CI 95\\% $\\hat\\alpha(1 \\pm 1{,}96/\\sqrt{k})$; $\\alpha$ mai mic, coadă mai groasă')))

D.frame(T('What You Need for Today (2/4): Drawdown and Correlation', 'Noțiuni necesare azi (2/4): drawdown și corelație'), items(
    (T('\\textbf{Drawdown}: $D_t = P_t/M_t - 1$, $M_t = \\max_{s \\le t} P_s$ the running maximum (the last record high)', '\\textbf{Drawdown}: $D_t = P_t/M_t - 1$, $M_t = \\max_{s \\le t} P_s$ maximul curent (ultimul maxim istoric)'),
     [T('\\textbf{maximum drawdown} $= \\min_t D_t$; to recover from a loss $d$, the price must rise by $d/(1 - d)$', '\\textbf{drawdown maxim} $= \\min_t D_t$; pentru a recupera o pierdere $d$, prețul trebuie să crească cu $d/(1 - d)$')]),
    (T('\\textbf{Correlation} of two series with different calendars: join the prices on common days, then compute log returns', '\\textbf{Corelația} a două serii cu calendare diferite: unim prețurile în zilele comune, apoi calculăm randamentele logaritmice'),
     [T('Fisher $z$: $z = \\tanh^{-1}(\\hat\\rho)$ is approximately $N(\\tanh^{-1}\\rho, 1/(n - 3))$; 95\\% CI: $\\tanh(z \\pm 1.96/\\sqrt{n - 3})$', 'transformarea Fisher: $z = \\tanh^{-1}(\\hat\\rho)$ este aproximativ $N(\\tanh^{-1}\\rho, 1/(n - 3))$; CI 95\\%: $\\tanh(z \\pm 1{,}96/\\sqrt{n - 3})$'),
      T('two independent periods: $(z_2 - z_1)/\\sqrt{1/(n_1 - 3) + 1/(n_2 - 3)} \\sim N(0, 1)$ if $\\rho_1 = \\rho_2$', 'două perioade independente: $(z_2 - z_1)/\\sqrt{1/(n_1 - 3) + 1/(n_2 - 3)} \\sim N(0, 1)$ dacă $\\rho_1 = \\rho_2$')]),
    T('\\textbf{Safe haven} \\refBL: an asset that does not fall (or rises) when the stock market crashes', '\\textbf{Activ de refugiu} (safe haven) \\refBL: un activ care nu scade (sau crește) cînd bursa se prăbușește')))

D.frame(T('What You Need for Today (3/4): Stablecoins and Basis Points', 'Noțiuni necesare azi (3/4): stablecoins și puncte de bază'), items(
    (T('\\textbf{Stablecoin}: a crypto asset designed to trade at a fixed price, usually 1 USD (the \\textbf{peg})', '\\textbf{Stablecoin}: un activ cripto conceput să se tranzacționeze la un preț fix, de obicei 1 USD (\\textbf{paritatea}, peg)'),
     [T('fiat-backed (USDT, USDC: dollar reserves), crypto-backed (DAI: crypto collateral), algorithmic (UST: no reserves, collapsed in May 2022)', 'garantat fiat (USDT, USDC: rezerve în dolari), garantat cripto (DAI: garanții cripto), algoritmic (UST: fără rezerve, prăbușit în mai 2022)'),
      T('\\textbf{depeg}: the price moves away from the peg; March 2023: Silicon Valley Bank (SVB), which held part of the USDC reserves, failed', '\\textbf{depeg}: prețul se îndepărtează de paritate; martie 2023: a intrat în faliment Silicon Valley Bank (SVB), care deținea o parte din rezervele USDC')]),
    (T('\\textbf{Basis point} (bp) $= 0.01\\% = 0.0001$; \\textbf{peg deviation} $d_t = 10\\,000\\,(P_t - 1)$ bp', '\\textbf{Punct de bază} (bp) $= 0{,}01\\% = 0{,}0001$; \\textbf{abaterea de la paritate} $d_t = 10\\,000\\,(P_t - 1)$ bp'),
     [T('a holder who sells $Q$ coins at price $P < 1$ loses $Q(1 - P)$ dollars', 'un deținător care vinde $Q$ monede la prețul $P < 1$ pierde $Q(1 - P)$ dolari')]),
    (T('\\textbf{AR(1)} for the deviation: $d_t = c + \\phi\\,d_{t-1} + e_t$, $|\\phi| < 1$', '\\textbf{AR(1)} pentru abatere: $d_t = c + \\phi\\,d_{t-1} + e_t$, $|\\phi| < 1$'),
     [T('\\textbf{half-life} $\\ln 0.5/\\ln\\phi$: the number of days after which half of a deviation is gone', '\\textbf{timpul de înjumătățire} $\\ln 0{,}5/\\ln\\phi$: numărul de zile după care dispare jumătate din abatere')])))

D.frame(T('What You Need for Today (4/4): VaR, ES and Backtesting', 'Noțiuni necesare azi (4/4): VaR, ES și backtesting'), items(
    (T('$\\mathrm{VaR}_\\alpha = -q_\\alpha$: the loss exceeded with probability $\\alpha$ (\\textbf{VaR 1\\%}); $\\mathrm{ES}_\\alpha = -E[r \\mid r \\le q_\\alpha]$ (\\textbf{ES 2.5\\%})', '$\\mathrm{VaR}_\\alpha = -q_\\alpha$: pierderea depășită cu probabilitatea $\\alpha$ (\\textbf{VaR 1\\%}); $\\mathrm{ES}_\\alpha = -E[r \\mid r \\le q_\\alpha]$ (\\textbf{ES 2,5\\%})'),
     [T('Normal: $\\mathrm{VaR}_{1\\%} = -\\mu + 2.326\\,\\sigma$, $\\mathrm{ES}_{2.5\\%} = -\\mu + \\sigma\\varphi(1.960)/0.025$, $\\varphi(1.960) = 0.0584$', 'distribuția Normală: $\\mathrm{VaR}_{1\\%} = -\\mu + 2{,}326\\,\\sigma$, $\\mathrm{ES}_{2,5\\%} = -\\mu + \\sigma\\varphi(1{,}960)/0{,}025$, $\\varphi(1{,}960) = 0{,}0584$'),
      T('historical simulation: $k = \\lceil n\\alpha\\rceil$, VaR $= -r_{(k)}$, ES $= -$ the mean of the $k$ smallest returns', 'simularea istorică: $k = \\lceil n\\alpha\\rceil$, VaR $= -r_{(k)}$, ES $= -$ media celor mai mici $k$ randamente')]),
    (T('In money: for simple returns $W \\times \\mathrm{VaR}/100$; for log returns $W(1 - e^{-\\mathrm{VaR}/100})$', 'În bani: pentru randamente simple $W \\times \\mathrm{VaR}/100$; pentru randamente logaritmice $W(1 - e^{-\\mathrm{VaR}/100})$'),
     [T('$h$ days (i.i.d., mean 0): $\\sqrt{h}\\,\\mathrm{VaR}$; crypto: $h$ calendar days', '$h$ zile (i.i.d., media 0): $\\sqrt{h}\\,\\mathrm{VaR}$; cripto: $h$ zile calendaristice')]),
    (T('\\textbf{Backtest} \\refKupiec: $x$ exceptions ($r_t < -\\mathrm{VaR}_t$) in $n$ days; $LR = -2[\\ln L(0.01) - \\ln L(x/n)] \\sim \\chi^2(1)$, critical value 3.84', '\\textbf{Backtesting} \\refKupiec: $x$ depășiri ($r_t < -\\mathrm{VaR}_t$) în $n$ zile; $LR = -2[\\ln L(0{,}01) - \\ln L(x/n)] \\sim \\chi^2(1)$, valoarea critică 3,84'),
     [T('GARCH(1,1)-t (Chapter 9): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$; VaR for tomorrow $= -(\\mu + \\sigma_{t+1}q_{1\\%})$', 'GARCH(1,1)-t (Capitolul 9): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$; VaR pentru mîine $= -(\\mu + \\sigma_{t+1}q_{1\\%})$')])), 'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: Computations on Paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: Annualising Bitcoin', 'A1: anualizarea Bitcoin'),
         items(T('Bitcoin daily log returns: mean $\\bar r = 0.12\\%$, standard deviation $s = 3.5\\%$. An analyst uses the equity convention of 252 days.',
                 'Randamentele logaritmice zilnice ale Bitcoin: media $\\bar r = 0{,}12\\%$, abaterea standard $s = 3{,}5\\%$. Un analist folosește convenția de la acțiuni, cu 252 de zile.'),
               T('1. Compute the annual mean and volatility with 365 days.', '1. Calculați media și volatilitatea anuale cu 365 de zile.'),
               T('2. Compute them with 252 days and say by how much they are understated.', '2. Calculați-le cu 252 de zile și precizați cu cît sînt subestimate.'),
               T('3. Compute the Sharpe ratio (risk-free rate 0) correctly and with the mean $\\times 252$ and the volatility $\\times\\sqrt{365}$.', '3. Calculați raportul Sharpe (rata fără risc 0) corect și cu media $\\times 252$ și volatilitatea $\\times\\sqrt{365}$.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. $365 \\times 0.12 = @{a1.mean_true}\\%$; $\\sqrt{365} \\times 3.5 = @{a1.sqrt_true} \\times 3.5 = @{a1.vol_true}\\%$', '1. $365 \\times 0{,}12 = @{a1.mean_true}\\%$; $\\sqrt{365} \\times 3{,}5 = @{a1.sqrt_true} \\times 3{,}5 = @{a1.vol_true}\\%$'),
               T('2. $252 \\times 0.12 = @{a1.mean_wrong}\\%$ (@{a1.under_mean}\\% too low); $\\sqrt{252} \\times 3.5 = @{a1.vol_wrong}\\%$ (@{a1.under_vol}\\% too low)', '2. $252 \\times 0{,}12 = @{a1.mean_wrong}\\%$ (o subestimare de @{a1.under_mean}\\%); $\\sqrt{252} \\times 3{,}5 = @{a1.vol_wrong}\\%$ (o subestimare de @{a1.under_vol}\\%)'),
               T('3. Correct: $@{a1.mean_true}/@{a1.vol_true} = @{a1.sharpe}$; mixed: $@{a1.mean_wrong}/@{a1.vol_true} = @{a1.sharpe_mixed}$', '3. Corect: $@{a1.mean_true}/@{a1.vol_true} = @{a1.sharpe}$; amestecat: $@{a1.mean_wrong}/@{a1.vol_true} = @{a1.sharpe_mixed}$'),
               T('The 252-day convention makes Bitcoin look calmer; mixing conventions makes it look worse.', 'Convenția cu 252 de zile face ca Bitcoin să pară mai liniștit; amestecul convențiilor îl face să pară mai prost.')),
         size='scriptsize')

D.proposed(T('A2: Annualising Ethereum', 'A2: anualizarea Ethereum'),
           items(T('Ethereum daily log returns: $\\bar r = 0.20\\%$, $s = 5.0\\%$. Model: A1.', 'Randamentele logaritmice zilnice ale Ethereum: $\\bar r = 0{,}20\\%$, $s = 5{,}0\\%$. Model: A1.'),
                 T('1. Compute the annual mean and volatility with 365 and with 252 days.', '1. Calculați media și volatilitatea anuale cu 365 și cu 252 de zile.'),
                 T('2. Compute the ratio of the two volatilities.', '2. Calculați raportul celor două volatilități.'),
                 T('3. Compute the correct and the mixed Sharpe ratio.', '3. Calculați raportul Sharpe corect și pe cel amestecat.'),
                 T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
           items(T('1. Mean: @{a2.mean_true}\\% (365) and @{a2.mean_wrong}\\% (252); volatility: @{a2.vol_true}\\% and @{a2.vol_wrong}\\%', '1. Media: @{a2.mean_true}\\% (365) și @{a2.mean_wrong}\\% (252); volatilitatea: @{a2.vol_true}\\% și @{a2.vol_wrong}\\%'),
                 T('2. $\\sqrt{365/252} = @{a2.ratio}$ for any $s$', '2. $\\sqrt{365/252} = @{a2.ratio}$ pentru orice $s$'),
                 T('3. Correct $@{a2.sharpe}$; mixed $@{a2.sharpe_mixed}$', '3. Corect $@{a2.sharpe}$; amestecat $@{a2.sharpe_mixed}$'),
                 T('The understatement (@{a2.under_vol}\\% for the volatility) does not depend on the asset, only on the convention.', 'Subestimarea (@{a2.under_vol}\\% pentru volatilitate) nu depinde de activ, ci doar de convenție.')),
           size='scriptsize')

D.solved(T('A3: The Maximum Drawdown of a Price Path', 'A3: drawdown-ul maxim al unei traiectorii de preț'),
         items(T('Monthly closes of a crypto asset, in thousand USD: 20, 35, 69, 50, 30, 16, 25, 45, 73.', 'Închideri lunare ale unui activ cripto, în mii USD: 20, 35, 69, 50, 30, 16, 25, 45, 73.'),
               T('1. Write the running maximum $M_t$ and the drawdown $D_t$ of each month.', '1. Scrieți maximul curent $M_t$ și drawdown-ul $D_t$ pentru fiecare lună.'),
               T('2. Give the maximum drawdown and the gain needed to recover from the trough.', '2. Precizați drawdown-ul maxim și cîștigul necesar pentru recuperarea de la minim.'),
               T('3. Say in which month the asset recovers.', '3. Precizați în ce lună își revine activul.'),
               T('Report: two numbers, a list and one sentence.', 'Raportați: două valori, o listă și o frază.')),
         items(T('1. $M_t$: @{a3.max}; $D_t$ (\\%): @{a3.dd}', '1. $M_t$: @{a3.max}; $D_t$ (\\%): @{a3.dd}'),
               T('2. Maximum drawdown $16/69 - 1 = @{a3.mdd}\\%$; gain needed $69/16 - 1 = @{a3.need}\\%$', '2. Drawdown-ul maxim $16/69 - 1 = @{a3.mdd}\\%$; cîștigul necesar $69/16 - 1 = @{a3.need}\\%$'),
               T('3. Month @{a3.rec} (73 $>$ 69): a new record high.', '3. Luna @{a3.rec} (73 $>$ 69): un nou maxim istoric.'),
               T('A fall of about three quarters needs more than a quadrupling to recover.', 'O scădere de aproximativ trei sferturi cere mai mult decît o cvadruplare pentru recuperare.')),
         size='scriptsize')

D.proposed(T('A4: A Second Price Path', 'A4: o a doua traiectorie de preț'),
           items(T('Quarterly closes of Ethereum, in thousand USD: 1.4, 4.8, 3.2, 0.9, 1.2, 2.5, 4.9. Model: A3.', 'Închideri trimestriale ale Ethereum, în mii USD: 1,4; 4,8; 3,2; 0,9; 1,2; 2,5; 4,9. Model: A3.'),
                 T('1. Write the running maximum and the drawdowns.', '1. Scrieți maximul curent și drawdown-urile.'),
                 T('2. Give the maximum drawdown and the gain needed to recover.', '2. Precizați drawdown-ul maxim și cîștigul necesar pentru recuperare.'),
                 T('3. Say in which quarter the asset recovers.', '3. Precizați în ce trimestru își revine activul.'),
                 T('Report: two numbers, a list and one sentence.', 'Raportați: două valori, o listă și o frază.')),
           items(T('1. $M_t$: @{a4.max}; $D_t$ (\\%): @{a4.dd}', '1. $M_t$: @{a4.max}; $D_t$ (\\%): @{a4.dd}'),
                 T('2. $0.9/4.8 - 1 = @{a4.mdd}\\%$; gain needed $4.8/0.9 - 1 = @{a4.need}\\%$', '2. $0{,}9/4{,}8 - 1 = @{a4.mdd}\\%$; cîștigul necesar $4{,}8/0{,}9 - 1 = @{a4.need}\\%$'),
                 T('3. Quarter @{a4.rec}.', '3. Trimestrul @{a4.rec}.'),
                 T('The recovery gain grows fast with the depth: $d/(1 - d)$ explodes as $d \\to 1$.', 'Cîștigul de recuperare crește repede cu adîncimea: $d/(1 - d)$ explodează cînd $d \\to 1$.')),
           size='scriptsize')

D.solved(T('A5: USDC in March 2023', 'A5: USDC în martie 2023'),
         items(T('USDC daily closes: 10 March 2023 @{a5.c10}, 11 March @{a5.c11}, 12 March @{a5.c12}, 13 March @{a5.c13}; lowest daily low @{a5.low} (11 March).',
                 'Închiderile zilnice USDC: 10 martie 2023 @{a5.c10}, 11 martie @{a5.c11}, 12 martie @{a5.c12}, 13 martie @{a5.c13}; cel mai mic minim zilnic @{a5.low} (11 martie).'),
               T('1. Compute the peg deviation of each close in basis points.', '1. Calculați abaterea de la paritate a fiecărei închideri în puncte de bază.'),
               T('2. Compute the deviation of the lowest low.', '2. Calculați abaterea celui mai mic minim.'),
               T('3. Compute the loss of a holder of 1\\,000\\,000 USDC who sold at the worst close, and at the lowest low.', '3. Calculați pierderea unui deținător de 1\\,000\\,000 USDC care a vîndut la cea mai proastă închidere și la cel mai mic minim.'),
               T('Report: five deviations, two losses and one sentence.', 'Raportați: cinci abateri, două pierderi și o frază.')),
         items(T('1. $10\\,000(P - 1)$: $@{a5.b10}$, $@{a5.b11}$, $@{a5.b12}$, $@{a5.b13}$ bp', '1. $10\\,000(P - 1)$: $@{a5.b10}$; $@{a5.b11}$; $@{a5.b12}$; $@{a5.b13}$ bp'),
               T('2. $10\\,000 \\times (@{a5.low} - 1) = @{a5.lowbp}$ bp', '2. $10\\,000 \\times (@{a5.low} - 1) = @{a5.lowbp}$ bp'),
               T('3. $1\\,000\\,000 \\times (1 - @{a5.c11}) = @{a5.lossc}$ USD; at the low: @{a5.lossl} USD', '3. $1\\,000\\,000 \\times (1 - @{a5.c11}) = @{a5.lossc}$ USD; la minim: @{a5.lossl} USD'),
               T('Who sold in panic on Saturday lost; who held until Monday, after the deposits were guaranteed, lost almost nothing.', 'Cine a vîndut în panică sîmbătă a pierdut; cine a păstrat moneda pînă luni, după garantarea depozitelor, nu a pierdut aproape nimic.')),
         size='scriptsize')

D.proposed(T('A6: DAI in March 2023', 'A6: DAI în martie 2023'),
           items(T('DAI daily closes: 10 March 2023 @{a6.c10}, 11 March @{a6.c11}, 12 March @{a6.c12}, 13 March @{a6.c13}; lowest daily low @{a6.low}. Model: A5.',
                   'Închiderile zilnice DAI: 10 martie 2023 @{a6.c10}, 11 martie @{a6.c11}, 12 martie @{a6.c12}, 13 martie @{a6.c13}; cel mai mic minim zilnic @{a6.low}. Model: A5.'),
                 T('1. Compute the four closing deviations and the deviation of the low in bp.', '1. Calculați cele patru abateri ale închiderilor și abaterea minimului în bp.'),
                 T('2. Compute the loss on 1\\,000\\,000 DAI sold at the worst close and at the low.', '2. Calculați pierderea pentru 1\\,000\\,000 DAI vînduți la cea mai proastă închidere și la minim.'),
                 T('3. Explain why DAI, a crypto-backed coin, followed USDC.', '3. Explicați de ce DAI, o monedă garantată cripto, a urmat USDC.'),
                 T('Report: five deviations, two losses and one sentence.', 'Raportați: cinci abateri, două pierderi și o frază.')),
           items(T('1. $@{a6.b10}$, $@{a6.b11}$, $@{a6.b12}$, $@{a6.b13}$ bp; low $@{a6.lowbp}$ bp', '1. $@{a6.b10}$; $@{a6.b11}$; $@{a6.b12}$; $@{a6.b13}$ bp; minimul $@{a6.lowbp}$ bp'),
                 T('2. @{a6.lossc} USD at the close; @{a6.lossl} USD at the low', '2. @{a6.lossc} USD la închidere; @{a6.lossl} USD la minim'),
                 T('3. A large part of the collateral of DAI was USDC: a crypto-backed coin inherits the risk of its collateral.', '3. O mare parte din garanția DAI era în USDC: o monedă garantată cripto moștenește riscul garanției ei.')),
           size='scriptsize')

D.solved(T('A7: VaR of a Bitcoin Position', 'A7: VaR al unei poziții în Bitcoin'),
         items(T('A position of 100\\,000 USD in Bitcoin.', 'O poziție de 100\\,000 USD în Bitcoin.'),
               T('1. With Normal daily simple returns ($\\mu = 0.10\\%$, $\\sigma = 3.5\\%$), compute VaR 1\\%, ES 2.5\\% and the 10-day VaR 1\\% in \\% and in USD.', '1. Cu randamente zilnice simple Normale ($\\mu = 0{,}10\\%$, $\\sigma = 3{,}5\\%$), calculați VaR 1\\%, ES 2,5\\% și VaR 1\\% pe 10 zile, în \\% și în USD.'),
               T('2. The 10 smallest of the last 365 log returns (@{a7.first} -- @{a7.last}), in \\%: @{a7.list}. Compute the historical VaR 1\\% and ES 2.5\\% in \\% and in USD.', '2. Cele mai mici 10 din ultimele 365 de randamente logaritmice (@{a7.first} -- @{a7.last}), în \\%: @{a7.list}. Calculați VaR 1\\% și ES 2,5\\% istorice, în \\% și în USD.'),
               T('Report: ten numbers and one sentence.', 'Raportați: zece valori și o frază.')),
         items(T('1. VaR $= -0.10 + 2.326 \\times 3.5 = @{a7n.var1}\\%$ (@{a7n.var1_usd} USD); ES $= -0.10 + 3.5 \\times 0.0584/0.025 = @{a7n.es25}\\%$ (@{a7n.es25_usd} USD); 10 days: $\\sqrt{10} \\times 2.326 \\times 3.5 = @{a7n.var_h}\\%$ (@{a7n.var_h_usd} USD)',
                 '1. VaR $= -0{,}10 + 2{,}326 \\times 3{,}5 = @{a7n.var1}\\%$ (@{a7n.var1_usd} USD); ES $= -0{,}10 + 3{,}5 \\times 0{,}0584/0{,}025 = @{a7n.es25}\\%$ (@{a7n.es25_usd} USD); 10 zile: $\\sqrt{10} \\times 2{,}326 \\times 3{,}5 = @{a7n.var_h}\\%$ (@{a7n.var_h_usd} USD)'),
               T('2. $k = \\lceil 3.65 \\rceil = @{a7.k1}$: VaR 1\\% $= @{a7.var1}\\%$, $100\\,000(1 - @{a7.ev}) = @{a7.var1u}$ USD; $k = \\lceil 9.125 \\rceil = @{a7.k25}$: ES 2.5\\% $= @{a7.sum}/10 = @{a7.es25}\\%$ (@{a7.es25u} USD)',
                 '2. $k = \\lceil 3{,}65 \\rceil = @{a7.k1}$: VaR 1\\% $= @{a7.var1}\\%$, $100\\,000(1 - @{a7.ev}) = @{a7.var1u}$ USD; $k = \\lceil 9{,}125 \\rceil = @{a7.k25}$: ES 2,5\\% $= @{a7.sum}/10 = @{a7.es25}\\%$ (@{a7.es25u} USD)'),
               T('ES 2.5\\% can be below VaR 1\\%: ES $\\ge$ VaR holds only at the same level.', 'ES 2,5\\% poate fi sub VaR 1\\%: ES $\\ge$ VaR este valabilă doar la același nivel.')),
         size='scriptsize')

D.proposed(T('A8: VaR of an Ethereum Position', 'A8: VaR al unei poziții în Ethereum'),
           items(T('A position of 100\\,000 USD in Ethereum. Model: A7.', 'O poziție de 100\\,000 USD în Ethereum. Model: A7.'),
                 T('1. With Normal simple returns ($\\mu = 0.20\\%$, $\\sigma = 5.0\\%$), compute VaR 1\\%, ES 2.5\\% and the 10-day VaR 1\\%, in \\% and in USD.', '1. Cu randamente simple Normale ($\\mu = 0{,}20\\%$, $\\sigma = 5{,}0\\%$), calculați VaR 1\\%, ES 2,5\\% și VaR 1\\% pe 10 zile, în \\% și în USD.'),
                 T('2. The 10 smallest of the last 365 log returns, in \\%: @{a8.list}. Compute the historical VaR 1\\% and ES 2.5\\% in \\% and in USD.', '2. Cele mai mici 10 din ultimele 365 de randamente logaritmice, în \\%: @{a8.list}. Calculați VaR 1\\% și ES 2,5\\% istorice, în \\% și în USD.'),
                 T('Report: ten numbers and one sentence.', 'Raportați: zece valori și o frază.')),
           items(T('1. VaR @{a8n.var1}\\% (@{a8n.var1_usd} USD); ES @{a8n.es25}\\% (@{a8n.es25_usd} USD); 10 days @{a8n.var_h}\\% (@{a8n.var_h_usd} USD)', '1. VaR @{a8n.var1}\\% (@{a8n.var1_usd} USD); ES @{a8n.es25}\\% (@{a8n.es25_usd} USD); 10 zile @{a8n.var_h}\\% (@{a8n.var_h_usd} USD)'),
                 T('2. VaR 1\\% $= @{a8.var1}\\%$ (@{a8.var1u} USD); ES 2.5\\% $= @{a8.sum}/10 = @{a8.es25}\\%$ (@{a8.es25u} USD)', '2. VaR 1\\% $= @{a8.var1}\\%$ (@{a8.var1u} USD); ES 2,5\\% $= @{a8.sum}/10 = @{a8.es25}\\%$ (@{a8.es25u} USD)'),
                 T('The last year was calmer than the Normal model with $\\sigma = 5\\%$ assumes.', 'Ultimul an a fost mai liniștit decît presupune modelul Normal cu $\\sigma = 5\\%$.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: Real Data, Inference and Interpretation', 'Partea B: date reale, inferență și interpretare')

D.task(T('B1: Bitcoin Against the S\\&P 500 [Solved]', 'B1: Bitcoin comparat cu S\\&P 500 [Rezolvat]'),
       T('is Bitcoin\'s distribution different from that of the S\\&P 500 only in scale, or also in the shape of its tails?', 'diferă distribuția Bitcoin de cea a S\\&P 500 doar prin scală sau și prin forma cozilor?'),
       T('Bitcoin and S\\&P 500 closes since September 2014, daily log returns in \\%, each on its own calendar', 'închiderile Bitcoin și S\\&P 500 din septembrie 2014, randamente logaritmice zilnice în \\%, fiecare pe propriul calendar'),
       [T('Compute the mean, the standard deviation, the skewness and the excess kurtosis of each series.', 'Calculați media, abaterea standard, asimetria și excesul de boltire pentru fiecare serie.'),
        T('Annualise the mean and the volatility with the actual frequency and with 252 days.', 'Anualizați media și volatilitatea cu frecvența reală și cu 252 de zile.'),
        T('Compute the Hill index of the left and of the right tail with $k = 2.5\\%$ of $n$ and its 95\\% interval.', 'Calculați indicele Hill pentru coada stîngă și pentru coada dreaptă cu $k = 2{,}5\\%$ din $n$ și intervalul de încredere de 95\\%.'),
        T('Draw the Hill plot of the losses for both series.', 'Desenați graficul Hill al pierderilor pentru ambele serii.'),
        T('Interpretation: are Bitcoin\'s tails heavier than those of the S\\&P 500?', 'Interpretare: are Bitcoin cozi mai groase decît S\\&P 500?')],
       T('a table of 12 numbers, the chart and two sentences', 'un tabel cu 12 valori, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: Solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch14_sem_b1', h='0.30') + table(
    'lrrrrrrrr', T('& $n$ & sd & skew. & exc.\\ kurt. & vol.\\ (actual $m$) & vol.\\ (252) & $\\hat\\alpha$ left [95\\% CI] & $\\hat\\alpha$ right',
                   '& $n$ & ab.\\ std. & asim. & exces boltire & vol.\\ ($m$ real) & vol.\\ (252) & $\\hat\\alpha$ stînga [CI 95\\%] & $\\hat\\alpha$ dreapta'),
    [f'{nm} & @{{b1.{k}.n}} & $@{{b1.{k}.sd}}$ & $@{{b1.{k}.skew}}$ & $@{{b1.{k}.k}}$ & $@{{b1.{k}.ann_vol}}$ & $@{{b1.{k}.ann_vol_252}}$ & $@{{b1.{k}.hill_left}}$ $[@{{b1.{k}.hill_left_lo}}, @{{b1.{k}.hill_left_hi}}]$ & $@{{b1.{k}.hill_right}}$'
     for k, nm in (('btc', 'Bitcoin'), ('sp500', 'S\\&P 500'))], size='scriptsize') + items(
    T('Means: @{b1.btc.mean}\\% and @{b1.sp500.mean}\\% a day; annual means @{b1.btc.ann_mean}\\% ($m = @{b1.btc.ppy}$) and @{b1.sp500.ann_mean}\\% ($m = @{b1.sp500.ppy}$); with 252 days Bitcoin\'s volatility would be @{b1.btc.ann_vol_252}\\%',
      'Mediile: @{b1.btc.mean}\\% și @{b1.sp500.mean}\\% pe zi; mediile anuale @{b1.btc.ann_mean}\\% ($m = @{b1.btc.ppy}$) și @{b1.sp500.ann_mean}\\% ($m = @{b1.sp500.ppy}$); cu 252 de zile, volatilitatea Bitcoin ar fi @{b1.btc.ann_vol_252}\\%'),
    T('Interpretation: no: the left-tail indices are almost equal and their intervals overlap; Bitcoin is about four times more volatile, but its tail has the same shape: the difference is in scale',
      'Interpretare: nu: indicii cozii stîngi sînt aproape egali, iar intervalele lor se suprapun; Bitcoin este de circa patru ori mai volatil, dar coada lui are aceeași formă: diferența este de scală')) + qlsem(), 'scriptsize')

D.task(T('B2: Ethereum and Solana [Proposed]', 'B2: Ethereum și Solana [Propus]'),
       T('do the conclusions of B1 hold for two other large crypto assets? Model: B1.', 'se păstrează concluziile din B1 pentru alte două active cripto mari? Model: B1.'),
       T('Ethereum since 2016, Solana since April 2020; daily log returns in \\%', 'Ethereum din 2016, Solana din aprilie 2020; randamente logaritmice zilnice în \\%'),
       [T('Compute the four moments of each series.', 'Calculați cele patru momente pentru fiecare serie.'),
        T('Annualise the volatility with 365 and with 252 days.', 'Anualizați volatilitatea cu 365 și cu 252 de zile.'),
        T('Compute both Hill indices with their 95\\% intervals.', 'Calculați ambii indici Hill cu intervalele lor de 95\\%.'),
        T('Interpretation: which of the five assets of B1 and B2 has the heaviest left tail?', 'Interpretare: care dintre activele din B1 și B2 are cea mai groasă coadă stîngă?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: Solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& $n$ & sd & skew. & exc.\\ kurt. & vol.\\ 365 & vol.\\ 252 & $\\hat\\alpha$ left [95\\% CI]', '& $n$ & ab.\\ std. & asim. & exces boltire & vol.\\ 365 & vol.\\ 252 & $\\hat\\alpha$ stînga [CI 95\\%]'),
    [f'{nm} & @{{b2.{k}.n}} & $@{{b2.{k}.sd}}$ & $@{{b2.{k}.skew}}$ & $@{{b2.{k}.k}}$ & $@{{b2.{k}.ann_vol}}$ & $@{{b2.{k}.ann_vol_252}}$ & $@{{b2.{k}.hill_left}}$ $[@{{b2.{k}.hill_left_lo}}, @{{b2.{k}.hill_left_hi}}]$'
     for k, nm in (('eth', 'Ethereum'), ('sol', 'Solana'))], size='scriptsize') + items(
    T('Right tails: $\\hat\\alpha = @{b2.eth.hill_right}$ (Ethereum), $@{b2.sol.hill_right}$ (Solana): the left tail is heavier, as for Bitcoin', 'Cozile drepte: $\\hat\\alpha = @{b2.eth.hill_right}$ (Ethereum), $@{b2.sol.hill_right}$ (Solana): coada stîngă este mai groasă, ca la Bitcoin'),
    T('Interpretation: Ethereum has the smallest point estimate ($@{b2.eth.hill_left}$), but all intervals overlap: with @{b2.sol.kh} to @{b2.eth.kh} tail observations we cannot rank the five tails; the volatilities differ clearly',
      'Interpretare: Ethereum are cea mai mică estimare punctuală ($@{b2.eth.hill_left}$), dar toate intervalele se suprapun: cu @{b2.sol.kh}--@{b2.eth.kh} de observații în coadă nu putem ordona cele cinci cozi; volatilitățile diferă clar')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: Bitcoin and the S\\&P 500 in Calm and Crisis [Solved]', 'B3: Bitcoin și S\\&P 500 în perioade calme și de criză [Rezolvat]'),
       T('did the correlation between Bitcoin and equities change after 2020, and what happened in the 2020 and 2022 crises?', 's-a schimbat corelația dintre Bitcoin și acțiuni după 2020 și ce s-a întîmplat în crizele din 2020 și 2022?'),
       T('Bitcoin and S\\&P 500 closes since September 2014, joined on common days; daily and weekly (Friday to Friday) log returns', 'închiderile Bitcoin și S\\&P 500 din septembrie 2014, unite în zilele comune; randamente logaritmice zilnice și săptămînale (vineri--vineri)'),
       [T('Compute the correlation before 2020 and from 2020 on, and test their equality with the Fisher $z$ test.', 'Calculați corelația înainte de 2020 și din 2020 și testați egalitatea lor cu testul $z$ al lui Fisher.'),
        T('Compute the correlation of daily and of weekly returns year by year and draw both.', 'Calculați corelația randamentelor zilnice și săptămînale an cu an și reprezentați-le grafic.'),
        T('Compute the cumulative returns of both series in the COVID-19, Terra/Luna and FTX episodes.', 'Calculați randamentele cumulate ale ambelor serii în episoadele COVID-19, Terra/Luna și FTX.'),
        T('Compute Bitcoin\'s mean return on the worst 5\\% of S\\&P 500 days.', 'Calculați randamentul mediu al Bitcoin în cele mai proaste 5\\% dintre zilele S\\&P 500.'),
        T('Interpretation: does Bitcoin diversify an equity portfolio when it is most needed?', 'Interpretare: diversifică Bitcoin un portofoliu de acțiuni atunci cînd este cea mai mare nevoie?')],
       T('six numbers, the chart and two sentences', 'șase valori, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: Solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch14_sem_b3', h='0.32') + items(
    T('Before 2020: $\\hat\\rho = @{b3.pre}$ ($n = @{b3.npre}$); from 2020: $\\hat\\rho = @{b3.post}$, 95\\% CI $[@{b3.ci_lo}, @{b3.ci_hi}]$ ($n = @{b3.npost}$); Fisher $z = @{b3.z}$, p @{b3.p}',
      'Înainte de 2020: $\\hat\\rho = @{b3.pre}$ ($n = @{b3.npre}$); din 2020: $\\hat\\rho = @{b3.post}$, CI 95\\% $[@{b3.ci_lo}, @{b3.ci_hi}]$ ($n = @{b3.npost}$); Fisher $z = @{b3.z}$, p @{b3.p}'),
    T('Cumulative log returns (Bitcoin; S\\&P 500): COVID-19 $@{b3.cr.covid.btc}\\%$; $@{b3.cr.covid.sp500}\\%$; Terra/Luna $@{b3.cr.terra.btc}\\%$; $@{b3.cr.terra.sp500}\\%$; FTX $@{b3.cr.ftx.btc}\\%$; $@{b3.cr.ftx.sp500}\\%$',
      'Randamente logaritmice cumulate (Bitcoin; S\\&P 500): COVID-19 $@{b3.cr.covid.btc}\\%$; $@{b3.cr.covid.sp500}\\%$; Terra/Luna $@{b3.cr.terra.btc}\\%$; $@{b3.cr.terra.sp500}\\%$; FTX $@{b3.cr.ftx.btc}\\%$; $@{b3.cr.ftx.sp500}\\%$'),
    T('On the @{b3.badn} worst S\\&P 500 days (mean $@{b3.bad_mean_b}\\%$), Bitcoin returned on average $@{b3.bad_mean}\\%$; weekly correlation over the whole period @{b3.weekly_all}, daily @{b3.daily_all}',
      'În cele @{b3.badn} de zile cele mai proaste ale S\\&P 500 (media $@{b3.bad_mean_b}\\%$), Bitcoin a avut în medie $@{b3.bad_mean}\\%$; corelația săptămînală pe întreaga perioadă @{b3.weekly_all}, cea zilnică @{b3.daily_all}'),
    T('Interpretation: no: since 2020 the correlation is significantly positive, and on the worst equity days Bitcoin falls as much as the S\\&P 500; crypto-specific crises (Terra, FTX) did not spread to equities',
      'Interpretare: nu: din 2020 corelația este semnificativ pozitivă, iar în cele mai proaste zile ale acțiunilor Bitcoin scade cît S\\&P 500; crizele specifice cripto (Terra, FTX) nu s-au extins la acțiuni')) + qlsem(), 'scriptsize')

D.task(T('B4: Bitcoin and Gold, Ethereum and the S\\&P 500 [Proposed]', 'B4: Bitcoin și aurul, Ethereum și S\\&P 500 [Propus]'),
       T('is Bitcoin linked with gold, and does Ethereum behave like Bitcoin towards equities? Model: B3.', 'este Bitcoin legat de aur și se comportă Ethereum ca Bitcoin față de acțiuni? Model: B3.'),
       T('Bitcoin and gold (XAU/USD) since 2014; Ethereum and S\\&P 500 since 2016; prices joined on common days', 'Bitcoin și aurul (XAU/USD) din 2014; Ethereum și S\\&P 500 din 2016; prețuri unite în zilele comune'),
       [T('Compute both correlations before and after 2020 and test their equality.', 'Calculați ambele corelații înainte și după 2020 și testați egalitatea lor.'),
        T('Compute the correlations of 2022 and 2025, daily and weekly.', 'Calculați corelațiile din 2022 și 2025, zilnice și săptămînale.'),
        T('Compute the mean return of Bitcoin on the worst 5\\% of gold days, and of Ethereum on the worst 5\\% of S\\&P 500 days.', 'Calculați randamentul mediu al Bitcoin în cele mai proaste 5\\% dintre zilele aurului și al Ethereum în cele mai proaste 5\\% dintre zilele S\\&P 500.'),
        T('Interpretation: is Bitcoin ``digital gold\'\'?', 'Interpretare: este Bitcoin „aur digital”?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: Solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& before 2020 & from 2020 & Fisher $z$ (p) & 2022 d./w. & 2025 d./w. & mean on bad days', '& înainte de 2020 & din 2020 & Fisher $z$ (p) & 2022 z./săpt. & 2025 z./săpt. & media în zilele proaste'),
    [T('Bitcoin, gold', 'Bitcoin, aur') + ' & $@{b4g.pre}$ & $@{b4g.post}$ & $@{b4g.z}$ (@{b4g.p}) & $@{b4g.y2022}$/$@{b4g.w2022}$ & $@{b4g.y2025}$/$@{b4g.w2025}$ & $@{b4g.bad_mean}$',
     T('Ethereum, S\\&P 500', 'Ethereum, S\\&P 500') + ' & $@{b4e.pre}$ & $@{b4e.post}$ & $@{b4e.z}$ (@{b4e.p}) & $@{b4e.y2022}$/$@{b4e.w2022}$ & $@{b4e.y2025}$/$@{b4e.w2025}$ & $@{b4e.bad_mean}$'],
    size='scriptsize') + items(
    T('Bad days: the worst 5\\% of days of the second series (gold; S\\&P 500); mean of the first series in \\%', 'Zilele proaste: cele mai proaste 5\\% dintre zilele celei de-a doua serii (aurul; S\\&P 500); media primei serii în \\%'),
    T('Interpretation: no: the correlation with gold rose only to @{b4g.post} and Bitcoin does not hold value when gold falls or when stocks fall; Ethereum moves with equities even more than Bitcoin',
      'Interpretare: nu: corelația cu aurul a crescut doar la @{b4g.post}, iar Bitcoin nu își păstrează valoarea nici cînd scade aurul, nici cînd scad acțiunile; Ethereum se mișcă împreună cu acțiunile chiar mai mult decît Bitcoin')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: How Stable Is USDC? [Solved]', 'B5: cît de stabil este USDC? [Rezolvat]'),
       T('how far and for how long does USDC move away from 1 USD?', 'cît de mult și pentru cît timp se îndepărtează USDC de 1 USD?'),
       T('USDC daily closes and lows since 1 January 2021', 'închiderile și minimele zilnice USDC din 1 ianuarie 2021'),
       [T('Compute the peg deviations in bp and their standard deviation and median absolute value.', 'Calculați abaterile de la paritate în bp, abaterea lor standard și valoarea absolută mediană.'),
        T('Compute the share of days in the classes below 1, 1--5, 5--10, 10--50 and above 50 bp, and draw them.', 'Calculați proporția zilelor din clasele sub 1, 1--5, 5--10, 10--50 și peste 50 bp și reprezentați-le grafic.'),
        T('Fit an AR(1) to the deviation and compute the half-life of a deviation.', 'Estimați un AR(1) pentru abatere și calculați timpul de înjumătățire al unei abateri.'),
        T('Compare the standard deviation before 2023 and after April 2023.', 'Comparați abaterea standard dinainte de 2023 cu cea de după aprilie 2023.'),
        T('Interpretation: can a treasurer treat USDC as cash?', 'Interpretare: poate un trezorier să trateze USDC ca numerar?')],
       T('six numbers, the chart and two sentences', 'șase valori, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: Solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch14_sem_b5', h='0.30') + items(
    T('$n = @{b5.n}$ days; sd @{b5.sd} bp, median $|d_t|$ @{b5.mad} bp; @{b5.gt10}\\% of days beyond 10 bp, @{b5.gt50}\\% beyond 50 bp; worst close $@{b5.close_min}$ bp (@{b5.cmind}), worst low $@{b5.low_min}$ bp',
      '$n = @{b5.n}$ zile; abaterea standard @{b5.sd} bp, $|d_t|$ median @{b5.mad} bp; @{b5.gt10}\\% din zile peste 10 bp, @{b5.gt50}\\% peste 50 bp; cea mai proastă închidere $@{b5.close_min}$ bp (@{b5.cmind}), cel mai prost minim $@{b5.low_min}$ bp'),
    T('AR(1): $\\hat\\phi = @{b5.ar}$ (se $@{b5.arse}$), half-life @{b5.hl} days; sd @{b5.sd_2023} bp in 2021--2022, @{b5.sd_after} bp after April 2023',
      'AR(1): $\\hat\\phi = @{b5.ar}$ (se $@{b5.arse}$), timpul de înjumătățire @{b5.hl} zile; abaterea standard @{b5.sd_2023} bp în 2021--2022, @{b5.sd_after} bp după aprilie 2023'),
    T('Interpretation: almost always: on a typical day USDC is within a few bp and deviations close within a day; but the tail is real: on one weekend of 2023 a holder could lose 3\\% at the close and 12\\% at the low',
      'Interpretare: aproape întotdeauna: într-o zi obișnuită USDC este la cîțiva bp de paritate, iar abaterile se închid într-o zi; dar coada este reală: într-un weekend din 2023 un deținător putea pierde 3\\% la închidere și 12\\% la minim')) + qlsem(), 'scriptsize')

D.task(T('B6: USDT and DAI [Proposed]', 'B6: USDT și DAI [Propus]'),
       T('are USDT and DAI as stable as USDC? Model: B5.', 'sînt USDT și DAI la fel de stabile ca USDC? Model: B5.'),
       T('USDT and DAI daily closes and lows since 1 January 2021', 'închiderile și minimele zilnice USDT și DAI din 1 ianuarie 2021'),
       [T('Compute the standard deviation, the median absolute deviation and the shares beyond 10 and 50 bp.', 'Calculați abaterea standard, abaterea absolută mediană și proporțiile peste 10 și 50 bp.'),
        T('Find the worst close and the worst low of each coin, with their dates.', 'Găsiți cea mai proastă închidere și cel mai prost minim pentru fiecare monedă, cu datele lor.'),
        T('Fit the AR(1) and compute the half-lives.', 'Estimați AR(1) și calculați timpii de înjumătățire.'),
        T('Interpretation: which of the three coins keeps the peg best in normal times, and which in a crisis?', 'Interpretare: care dintre cele trei monede păstrează cel mai bine paritatea în perioade normale și care în criză?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')

D.frame(T('B6: Solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& sd & median $|d|$ & $>$10 bp (\\%) & $>$50 bp (\\%) & worst close & worst low & half-life', '& ab.\\ std. & $|d|$ median & $>$10 bp (\\%) & $>$50 bp (\\%) & cea mai proastă închidere & cel mai prost minim & înjumătățire'),
    ['USDT & $@{b6t.sd}$ & $@{b6t.mad}$ & $@{b6t.gt10}$ & $@{b6t.gt50}$ & $@{b6t.close_min}$ & $@{b6t.low_min}$ & $@{b6t.hl}$',
     'DAI & $@{b6d.sd}$ & $@{b6d.mad}$ & $@{b6d.gt10}$ & $@{b6d.gt50}$ & $@{b6d.close_min}$ & $@{b6d.low_min}$ & $@{b6d.hl}$',
     'USDC & $@{b5.sd}$ & $@{b5.mad}$ & $@{b5.gt10}$ & $@{b5.gt50}$ & $@{b5.close_min}$ & $@{b5.low_min}$ & $@{b5.hl}$'],
    size='scriptsize') + items(
    T('Worst closes: USDT on @{b6t.cmind}, DAI on @{b6d.cmind}; deviations in bp, half-lives in days', 'Cele mai proaste închideri: USDT pe @{b6t.cmind}, DAI pe @{b6d.cmind}; abaterile în bp, timpii de înjumătățire în zile'),
    T('Interpretation: in normal times USDC is the tightest (median @{b5.mad} bp) and USDT the loosest, with the slowest return to the peg; in March 2023 USDT stayed at the peg while USDC and DAI broke it: stability in calm times says little about crises',
      'Interpretare: în perioade normale USDC este cel mai strîns legat de paritate (mediana @{b5.mad} bp), iar USDT cel mai puțin, cu cea mai lentă revenire; în martie 2023 USDT a rămas la paritate, în timp ce USDC și DAI au rupt-o: stabilitatea din perioadele calme spune puțin despre crize')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B7: Risk of a Bitcoin Position [Solved]', 'B7: riscul unei poziții în Bitcoin [Rezolvat]'),
       T('how much can a position of 100\\,000 USD in Bitcoin lose in one day, and which model would have passed a backtest?', 'cît poate pierde într-o zi o poziție de 100\\,000 USD în Bitcoin și ce model ar fi trecut un backtesting?'),
       T('Bitcoin closes since September 2014, daily log returns in \\%', 'închiderile Bitcoin din septembrie 2014, randamente logaritmice zilnice în \\%'),
       [T('Compute VaR 1\\% and ES 2.5\\% by historical simulation, Normal, Student-t (maximum likelihood) and GARCH(1,1)-t for the next day.', 'Calculați VaR 1\\% și ES 2,5\\% prin simulare istorică, cu distribuția Normală, cu Student-t (verosimilitate maximă) și cu GARCH(1,1)-t pentru ziua următoare.'),
        T('Convert them into USD with $W(1 - e^{-v/100})$.', 'Convertiți-le în USD cu $W(1 - e^{-v/100})$.'),
        T('Backtest the one-day VaR 1\\% since 2018: historical simulation on the last 365 days and GARCH(1,1)-t re-estimated every 365 days; count the exceptions and run the Kupiec test.', 'Verificați VaR 1\\% pe o zi din 2018: simulare istorică pe ultimele 365 de zile și GARCH(1,1)-t reestimat la fiecare 365 de zile; numărați depășirile și aplicați testul Kupiec.'),
        T('Interpretation: which number would you report to a risk committee?', 'Interpretare: ce valoare ați raporta unui comitet de risc?')],
       T('a table of eight numbers, the chart, two test results and two sentences', 'un tabel cu opt valori, graficul, două rezultate de test și două fraze'), size='footnotesize', nb='B7')

D.frame(T('B7: Solution [Solved]', 'B7: rezolvare [Rezolvat]'), fig('ch14_sem_b7', h='0.28') + table(
    'lrrrr', T('& HS & Normal & Student-t & GARCH-t', '& HS & Normală & Student-t & GARCH-t'),
    ['VaR 1\\% (\\%) & $@{b7.hs.v}$ & $@{b7.n.v}$ & $@{b7.t.v}$ & $@{b7.g.v}$',
     T('VaR 1\\% (USD)', 'VaR 1\\% (USD)') + ' & @{b7.hs.vu} & @{b7.n.vu} & @{b7.t.vu} & @{b7.g.vu}',
     T('ES 2.5\\% (\\%)', 'ES 2,5\\% (\\%)') + ' & $@{b7.hs.e}$ & $@{b7.n.e}$ & $@{b7.t.e}$ & $@{b7.g.e}$'], size='scriptsize') + items(
    T('$n = @{b7.n}$; Student-t $\\hat\\nu = @{b7.nu}$; GARCH-t $\\hat\\sigma_{t+1} = @{b7.gs}\\%$, $\\hat\\nu = @{b7.gnu}$; backtest since @{b7.bt.start} ($n = @{b7.bt.n}$, expected @{b7.bt.hs.exp}): HS @{b7.bt.hs.x} exceptions, LR $= @{b7.bt.hs.lr}$ (p $= @{b7.bt.hs.p}$); GARCH-t @{b7.bt.g.x}, LR $= @{b7.bt.g.lr}$ (p $= @{b7.bt.g.p}$)',
      '$n = @{b7.n}$; Student-t $\\hat\\nu = @{b7.nu}$; GARCH-t $\\hat\\sigma_{t+1} = @{b7.gs}\\%$, $\\hat\\nu = @{b7.gnu}$; backtesting din @{b7.bt.start} ($n = @{b7.bt.n}$, așteptat @{b7.bt.hs.exp}): HS @{b7.bt.hs.x} depășiri, LR $= @{b7.bt.hs.lr}$ (p $= @{b7.bt.hs.p}$); GARCH-t @{b7.bt.g.x}, LR $= @{b7.bt.g.lr}$ (p $= @{b7.bt.g.p}$)'),
    T('Interpretation: report the historical VaR 1\\% (@{b7.hs.vu} USD) as the long-run number and the GARCH-t VaR (@{b7.g.vu} USD) as today\'s number; the Normal VaR is @{b7.gap}\\% too low; both backtested models pass',
      'Interpretare: raportăm VaR 1\\% istoric (@{b7.hs.vu} USD) ca valoare de termen lung și VaR GARCH-t (@{b7.g.vu} USD) ca valoare de azi; VaR Normal este cu @{b7.gap}\\% prea mic; ambele modele verificate trec testul')) + qlsem(), 'scriptsize')

D.task(T('B8: Risk of an Ethereum Position [Proposed]', 'B8: riscul unei poziții în Ethereum [Propus]'),
       T('does the conclusion of B7 hold for Ethereum? Model: B7.', 'se păstrează concluzia din B7 pentru Ethereum? Model: B7.'),
       T('Ethereum closes since 2016; backtest since 1 January 2019', 'închiderile Ethereum din 2016; backtesting din 1 ianuarie 2019'),
       [T('Compute VaR 1\\% and ES 2.5\\% by the four methods of B7, in \\% and in USD.', 'Calculați VaR 1\\% și ES 2,5\\% prin cele patru metode din B7, în \\% și în USD.'),
        T('Backtest the two one-day VaR 1\\% forecasts and run the Kupiec test.', 'Verificați cele două prognoze VaR 1\\% pe o zi și aplicați testul Kupiec.'),
        T('Find the largest number of exceptions in one calendar year for each model.', 'Găsiți cel mai mare număr de depășiri într-un an calendaristic pentru fiecare model.'),
        T('Interpretation: which model would you keep for Ethereum?', 'Interpretare: ce model ați păstra pentru Ethereum?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B8')

D.frame(T('B8: Solution [Proposed]', 'B8: rezolvare [Propus]'), table(
    'lrrrr', T('& HS & Normal & Student-t & GARCH-t', '& HS & Normală & Student-t & GARCH-t'),
    ['VaR 1\\% (\\%) & $@{b8.hs.v}$ & $@{b8.n.v}$ & $@{b8.t.v}$ & $@{b8.g.v}$',
     'VaR 1\\% (USD) & @{b8.hs.vu} & @{b8.n.vu} & @{b8.t.vu} & @{b8.g.vu}',
     T('ES 2.5\\% (\\%)', 'ES 2,5\\% (\\%)') + ' & $@{b8.hs.e}$ & $@{b8.n.e}$ & $@{b8.t.e}$ & $@{b8.g.e}$'], size='scriptsize') + items(
    T('Backtest since @{b8.bt.start} ($n = @{b8.bt.n}$, expected @{b8.bt.hs.exp}): HS @{b8.bt.hs.x} exceptions (p $= @{b8.bt.hs.p}$), at most @{b8.bt.hs.ymax} in one year; GARCH-t @{b8.bt.g.x} (p $= @{b8.bt.g.p}$), at most @{b8.bt.g.ymax} in one year',
      'Backtesting din @{b8.bt.start} ($n = @{b8.bt.n}$, așteptat @{b8.bt.hs.exp}): HS @{b8.bt.hs.x} depășiri (p $= @{b8.bt.hs.p}$), cel mult @{b8.bt.hs.ymax} într-un an; GARCH-t @{b8.bt.g.x} (p $= @{b8.bt.g.p}$), cel mult @{b8.bt.g.ymax} într-un an'),
    T('Interpretation: both pass; HS is closer to the target rate, GARCH-t is slightly conservative but adapts faster; the Normal VaR is @{b8.gap}\\% too low; keep HS for reporting and GARCH-t for daily limits',
      'Interpretare: ambele trec testul; HS este mai aproape de rata țintă, GARCH-t este puțin conservator, dar se adaptează mai repede; VaR Normal este cu @{b8.gap}\\% prea mic; păstrăm HS pentru raportare și GARCH-t pentru limitele zilnice')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: Open Questions and AI Critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: Did the Spot ETFs Change Bitcoin? [Proposed]', 'C1: au schimbat ETF-urile spot comportamentul Bitcoin? [Propus]'),
       T('since 11 January 2024 US investors can hold Bitcoin through spot ETFs traded on weekdays; did Bitcoin\'s volatility, its weekend pattern or its link with equities change?',
         'din 11 ianuarie 2024, investitorii americani pot deține Bitcoin prin ETF-uri spot tranzacționate în zilele lucrătoare; s-au schimbat volatilitatea Bitcoin, tiparul ei de weekend sau legătura cu acțiunile?'),
       T('Bitcoin and S\\&P 500; two years before and two years after the launch; models: B1, B3', 'Bitcoin și S\\&P 500; doi ani înainte și doi ani după lansare; modele: B1, B3'),
       [T('Compute the annualised volatility in both windows and test the equality of variances (Brown--Forsythe).', 'Calculați volatilitatea anuală în ambele ferestre și testați egalitatea varianțelor (Brown--Forsythe).'),
        T('Compute the ratio of weekend to weekday variance in both windows.', 'Calculați raportul dintre varianța din weekend și cea din zilele lucrătoare în ambele ferestre.'),
        T('Compute the correlation with the S\\&P 500 and the GARCH(1,1)-t persistence in both windows.', 'Calculați corelația cu S\\&P 500 și persistența GARCH(1,1)-t în ambele ferestre.'),
        T('Interpretation: can a before--after comparison prove an effect of the ETFs?', 'Interpretare: poate o comparație înainte--după să demonstreze un efect al ETF-urilor?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: Reference Analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrr', T('Bitcoin & 11 Jan 2022 -- 10 Jan 2024 & 11 Jan 2024 -- 10 Jan 2026', 'Bitcoin & 11 ian. 2022 -- 10 ian. 2024 & 11 ian. 2024 -- 10 ian. 2026'),
    [T('annualised volatility (\\%)', 'volatilitatea anuală (\\%)') + ' & $@{c1.vol_before}$ & $@{c1.vol_after}$',
     T('weekend/weekday variance', 'varianța weekend/zile lucrătoare') + ' & $@{c1.we_ratio_before}$ & $@{c1.we_ratio_after}$',
     T('weekend share of squared returns (\\%)', 'ponderea weekendului în pătratele randamentelor (\\%)') + ' & $@{c1.we_share_before}$ & $@{c1.we_share_after}$',
     T('correlation with the S\\&P 500', 'corelația cu S\\&P 500') + ' & $@{c1.corr_before}$ & $@{c1.corr_after}$',
     T('GARCH(1,1)-t $\\hat\\alpha + \\hat\\beta$', 'GARCH(1,1)-t $\\hat\\alpha + \\hat\\beta$') + ' & $@{c1.pers_before}$ & $@{c1.pers_after}$'], size='scriptsize') + items(
    T('Brown--Forsythe p $= @{c1.p_bf}$: the fall in volatility is not significant; the weekend share rose, the correlation fell', 'Brown--Forsythe p $= @{c1.p_bf}$: scăderea volatilității nu este semnificativă; ponderea weekendului a crescut, corelația a scăzut'),
    T('No: rates, the April 2024 halving and the US election changed in the same window; a credible design uses a control group (coins without an ETF), intraday data around 9:30 New York time, or ETF flows as a continuous treatment',
      'Nu: dobînzile, halving-ul din aprilie 2024 și alegerile din SUA s-au schimbat în aceeași fereastră; un design credibil folosește un grup de control (monede fără ETF), date intrazilnice în jurul orei 9:30 la New York sau fluxurile ETF ca tratament continuu')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: Audit an AI Answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to summarise the risks of Bitcoin and stablecoins. The answer:', 'Un student a cerut unui asistent AI să rezume riscurile Bitcoin și ale stablecoin-urilor. Răspunsul:'),
    T('\\aiprompt{(a) Bitcoin\'s annual volatility is @{c2.v252}\\%: the daily standard deviation times sqrt(252).}', '\\aiprompt{(a) Volatilitatea anuală a Bitcoin este @{c2.v252}\\%: abaterea standard zilnică înmulțită cu sqrt(252).}'),
    T('\\aiprompt{(b) The VaR 99\\% of 100,000 USD in Bitcoin is -@{c2.varwrong} USD.}', '\\aiprompt{(b) VaR 99\\% pentru 100.000 USD în Bitcoin este -@{c2.varwrong} USD.}'),
    T('\\aiprompt{(c) On 11 March 2023 USDC lost @{c2.bpwrong} basis points.}', '\\aiprompt{(c) Pe 11 martie 2023, USDC a pierdut @{c2.bpwrong} puncte de bază.}'),
    T('\\aiprompt{(d) TerraUSD was a fiat-backed stablecoin whose reserves were US Treasury bills.}', '\\aiprompt{(d) TerraUSD era un stablecoin garantat fiat, cu rezerve în titluri de stat americane.}'),
    T('\\aiprompt{(e) Bitcoin is a safe haven: its correlation with the S\\&P 500 is @{c2.pre}.}', '\\aiprompt{(e) Bitcoin este un activ de refugiu: corelația lui cu S\\&P 500 este @{c2.pre}.}'),
    T('\\aiprompt{(f) Bitcoin returns are less volatile at weekends than on weekdays.}', '\\aiprompt{(f) Randamentele Bitcoin sînt mai puțin volatile în weekend decît în zilele lucrătoare.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: Solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: crypto trades 365 days a year: $\\sqrt{365}\\,s = @{c2.v365}\\%$', '(a) Greșit: activele cripto se tranzacționează 365 de zile pe an: $\\sqrt{365}\\,s = @{c2.v365}\\%$'),
    T('(b) Wrong three times: the level is VaR 1\\%, VaR is a positive loss, and a log-return VaR of @{c2.var}\\% is $100\\,000(1 - e^{-@{c2.var}/100}) = @{c2.varu}$ USD', '(b) Greșit de trei ori: nivelul este VaR 1\\%, VaR este o pierdere pozitivă, iar un VaR de @{c2.var}\\% pe randamente logaritmice înseamnă $100\\,000(1 - e^{-@{c2.var}/100}) = @{c2.varu}$ USD'),
    T('(c) Wrong unit: the close of 0.9715 is $@{c2.bp}$ bp, i.e.\\ 2.85\\%, not 2.85 bp', '(c) Unitate greșită: închiderea de 0,9715 înseamnă $@{c2.bp}$ bp, adică 2,85\\%, nu 2,85 bp'),
    T('(d) Wrong: UST was algorithmic, backed only by the LUNA token; that is why it could not be rescued', '(d) Greșit: UST era algoritmic, garantat doar de token-ul LUNA; de aceea nu a putut fi salvat'),
    T('(e) Wrong: @{c2.pre} is the correlation before 2020; since 2020 it is @{c2.post}, and on the worst 5\\% of S\\&P 500 days Bitcoin lost on average $@{c2.worst}\\%$', '(e) Greșit: @{c2.pre} este corelația dinainte de 2020; din 2020 este @{c2.post}, iar în cele mai proaste 5\\% dintre zilele S\\&P 500 Bitcoin a pierdut în medie $@{c2.worst}\\%$'),
    T('(f) Correct: the weekend variance is about a third to two thirds of the weekday variance (Lecture 14)', '(f) Corect: varianța din weekend este între o treime și două treimi din cea din zilele lucrătoare (Cursul 14)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-Up', 'Încheiere')

D.frame(T('What You Should Take from Today', 'Idei de reținut'), items(
    T('Annualise crypto with 365 days; never mix 252 and 365 in one ratio', 'Anualizăm cripto cu 365 de zile; nu amestecăm niciodată 252 și 365 în același raport'),
    T('A fall of $d$ needs a gain of $d/(1 - d)$; crypto drawdowns of 75--85\\% need gains of 300--570\\%', 'O scădere $d$ cere un cîștig $d/(1 - d)$; drawdown-urile cripto de 75--85\\% cer cîștiguri de 300--570\\%'),
    T('Peg deviations are measured in basis points; they are tiny in normal times and large in a run', 'Abaterile de la paritate se măsoară în puncte de bază; sînt foarte mici în perioade normale și mari în timpul unei retrageri masive'),
    T('Since 2020 Bitcoin moves with equities; it is not a safe haven in this sample', 'Din 2020, Bitcoin se mișcă împreună cu acțiunile; nu este un activ de refugiu în acest eșantion'),
    T('An AI answer is a draft: check the annualisation, the VaR level and sign, and the units', 'Un răspuns AI este o ciornă: verificați anualizarea, nivelul și semnul VaR, precum și unitățile de măsură')))

D.frame(T('After the Seminar', 'După seminar'), items(
    T('Lecture 14 develops each topic of today: blockchain basics, crypto as an asset class, efficiency, correlation, bubbles, ETFs, stablecoins and regulation',
      'Cursul 14 dezvoltă fiecare temă de azi: noțiunile de bază despre blockchain, cripto ca o clasă de active, eficiența, corelația, bulele, ETF-urile, stablecoins și reglementarea'),
    T('Try the [Proposed] tasks in the notebook; the solutions are discussed in class', 'Încercați cerințele [Propus] în notebook; rezolvările se discută la seminar'),
    T('C1 can grow into a team project: a control group, intraday data, ETF flows', 'C1 poate deveni un proiect de echipă: un grup de control, date intrazilnice, fluxurile ETF'),
    T('Reading: \\refFHH, Ch.~23; \\refLT; \\refLVN', 'Lectură: \\refFHH, cap.~23; \\refLT; \\refLVN')))

D.references(bib(['BL', 'FHH', 'Hill', 'Kupiec', 'LT', 'LVN', 'LMS', 'GS', 'Fed', 'SEC']), per=16)

if __name__ == '__main__':
    D.write(V)
