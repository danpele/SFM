"""Build the interactive pages of Chapter 3: interactive/ch3.html (EN) and interactive/ch3_ro.html (RO).

One block per chart slide of the lecture body and one block per results table or worked example (XBLOCKS), in the order of
the slides: the slide on the left, its interactive version on the right, with live tables and worked examples. On the web
lecture the table blocks travel inside a chart block (widgets) and ch3.js moves each one to its own frame. Logic in interactive/ch3.js, inputs in interactive/ch3_data.js
(Quantlets/Ch_03/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 3
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_03'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('stability-qq', 'sfm_ch3_stability_qq', 'stab'), ('gclt', 'sfm_ch3_gclt', 'gclt'),
          ('density-alpha', 'sfm_ch3_density_alpha', 'par'), ('density-beta', 'sfm_ch3_density_beta', 'par'),
          ('s0-s1', 'sfm_ch3_s0_s1', 'par'), ('special-cases', 'sfm_ch3_special_cases', 'par'),
          ('tails-loglog', 'sfm_ch3_tails_loglog', 'tail'), ('running-var-sim', 'sfm_ch3_running_var_sim', 'tail'),
          ('cms', 'sfm_ch3_cms', 'sim'),
          ('mcculloch-map', 'sfm_ch3_mcculloch_map', 'est'), ('estimators-mc', 'sfm_ch3_estimators_mc', 'est'),
          ('fit-density-1', 'sfm_ch3_fit_density_1', 'real'), ('fit-density-2', 'sfm_ch3_fit_density_2', 'real'),
          ('qq-real-1', 'sfm_ch3_qq_real_1', 'real'), ('qq-real-2', 'sfm_ch3_qq_real_2', 'real'),
          ('tails-real-1', 'sfm_ch3_tails_real_1', 'real'), ('tails-real-2', 'sfm_ch3_tails_real_2', 'real'),
          ('aggregation', 'sfm_ch3_aggregation', 'crit'), ('running-var-real', 'sfm_ch3_running_var_real', 'crit')]

PFX = {'stability-qq': 'sq', 'gclt': 'gc', 'density-alpha': 'da', 'density-beta': 'db', 's0-s1': 'ss', 'special-cases': 'sc',
       'tails-loglog': 'tl', 'running-var-sim': 'rv', 'cms': 'cm', 'mcculloch-map': 'mm', 'estimators-mc': 'mc',
       'fit-density-1': 'f1', 'fit-density-2': 'f2', 'qq-real-1': 'q1', 'qq-real-2': 'q2', 'tails-real-1': 't1',
       'tails-real-2': 't2', 'aggregation': 'ag', 'running-var-real': 'rr'}

SECTIONS = {
 'ro': dict(stab='Stabilitatea la adunare', gclt='Teorema limită centrală generalizată', par='Parametrii și funcția caracteristică',
            tail='Cozi și momente', sim='Simulare', est='Estimare', real='Estimarea pe randamente reale',
            crit='Critica modelului stabil: varianță finită sau infinită'),
 'en': dict(stab='Stability under Summation', gclt='The Generalised Central Limit Theorem', par='Parameters and the Characteristic Function',
            tail='Tails and Moments', sim='Simulation', est='Estimation', real='Fitting Real Returns',
            crit='Critique of the Stable Model: Finite or Infinite Variance'),
}

ASSETS = [('bet', 'BET'), ('sp500', 'S&P 500'), ('dax', 'DAX'), ('btc', 'Bitcoin')]

L = {
 'ro': dict(file='ch3_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 3 · Distribuții α-stabile', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Randamente logaritmice zilnice, în %: BET, S&P 500 și DAX din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), '
       'pînă la 18 septembrie 2026. Prețuri de închidere EODHD. Legile estimate sînt cele din curs: distribuția Normală, Student-t și '
       'legea stabilă estimată prin verosimilitate maximă în parametrizarea S0.'),
 'en': dict(file='ch3.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 3 · α-Stable Distributions', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Daily log returns, in %: BET, S&P 500 and DAX from January 2000, Bitcoin from September 2014 (7 days a week), '
       'to 18 September 2026. EODHD closing prices. The fitted laws are those of the lecture: the Normal distribution, the Student-t and '
       'the stable law fitted by maximum likelihood in the S0 parameterisation.'),
}


def widgets_old(lang):
    ro = lang == 'ro'
    ns = [(str(n), str(n)) for n in (2, 5, 10, 20, 50)]
    return {
     'stability-qq': (select('sq-n', 'Numărul de extrageri adunate n' if ro else 'Number of draws summed n', ns, '10')
                      + slider('sq-a', 'Exponentul α′ al împărțitorului n<sup>1/α′</sup> (albastru)' if ro else 'Exponent α′ of the divisor n<sup>1/α′</sup> (blue)', 1, 2, 0.05, 1.7)
                      + play('sq'), charts('sq-chart')),
     'gclt': (slider('gc-k', 'Pragul k pentru P(|·| > k)' if ro else 'Threshold k for P(|·| > k)', 2, 15, 0.5, 10) + play('gc'), charts('gc-chart')),
     'density-alpha': (slider('da-a', 'α al curbei alese (punctat)' if ro else 'α of the chosen curve (dotted)', 0.5, 2, 0.05, 1.7) + play('da'), charts('da-chart')),
     'density-beta': (slider('db-b', 'β al curbei alese, α = 1,3 (stînga)' if ro else 'β of the chosen curve, α = 1.3 (left)', -1, 1, 0.1, 0.5)
                      + slider('db-g', 'γ al curbei alese, α = 1,7 (dreapta; ▶ îl mută odată cu β)' if ro else 'γ of the chosen curve, α = 1.7 (right; ▶ moves it with β)', 0.25, 3, 0.25, 1.5)
                      + play('db'), charts('db-chart')),
     's0-s1': (slider('ss-a', 'α al curbei alese (punctat), β = 0,8' if ro else 'α of the chosen curve (dotted), β = 0.8', 0.7, 1.3, 0.01, 0.9) + play('ss'), charts('ss-chart')),
     'special-cases': (slider('sc-x', 'Pragul x pentru P(X > x)' if ro else 'Threshold x for P(X > x)', 1, 20, 0.5, 3) + play('sc'), charts('sc-chart')),
     'tails-loglog': (slider('tl-a', 'α al curbei alese' if ro else 'α of the chosen curve', 1.1, 1.95, 0.05, 1.6) + play('tl'), charts('tl-chart')),
     'running-var-sim': (slider('rv-a', 'α al celor trei eșantioane stabile' if ro else 'α of the three stable samples', 0, 7, 1, 3) + play('rv'), charts('rv-chart')),
     'cms': (select('cm-a', 'α', [(a, a.replace('.', ',') if ro else a) for a in ('1.2', '1.5', '1.8')], '1.5')
             + slider('cm-b', 'β', -1, 1, 0.5, 0.5) + play('cm'), charts('cm-chart')),
     'mcculloch-map': (slider('mm-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2000, 2020, 1, 2000) + play('mm'), charts('mm-chart')),
     'estimators-mc': (slider('mc-r', 'Numărul de eșantioane Monte Carlo folosite' if ro else 'Number of Monte Carlo samples used', 10, 200, 10, 200) + play('mc'), charts('mc-chart')),
     'fit-density-1': (slider('f1-k', 'Randamentul x, în abateri standard' if ro else 'Return x, in standard deviations', -10, 10, 0.5, -5) + play('f1'), charts('f1-chart')),
     'fit-density-2': (slider('f2-k', 'Randamentul x, în abateri standard' if ro else 'Return x, in standard deviations', -10, 10, 0.5, -5) + play('f2'), charts('f2-chart')),
     'qq-real-1': (slider('q1-p', 'Nivelul p al cuantilei (cercurile)' if ro else 'Level p of the quantile (circles)', 0, 5, 1, 2) + play('q1'), charts('q1-chart')),
     'qq-real-2': (slider('q2-p', 'Nivelul p al cuantilei (cercurile)' if ro else 'Level p of the quantile (circles)', 0, 5, 1, 2) + play('q2'), charts('q2-chart')),
     'tails-real-1': (slider('t1-x', 'Pierderea x' if ro else 'Loss x', 2, 25, 0.5, 10) + play('t1'), charts('t1-chart')),
     'tails-real-2': (slider('t2-x', 'Pierderea x' if ro else 'Loss x', 2, 25, 0.5, 10) + play('t2'), charts('t2-chart')),
     'aggregation': (select('ag-e', 'Estimatorul' if ro else 'Estimator', [('ml', 'ML' if not ro else 'ML (verosimilitate maximă)'), ('mcc', 'McCulloch')], 'ml')
                     + slider('ag-c', 'Nivelul de încredere al intervalelor' if ro else 'Confidence level of the intervals', 0, 4, 1, 2) + play('ag'), charts('ag-chart')),
     'running-var-real': (select('rr-a', 'Seria' if ro else 'Series', ASSETS, 'sp500') + play('rr'), charts('rr-chart')),
    }


# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH3_TEXT = {
 sq: (n, a, fa, qa, fs, qs, q1) => `Suma a ${n}${n >= 20 ? ' de' : ''} extrageri împărțită la ${n}<sup>1/${a}</sup> = ${fa}: cuantila de 99% este ${qa}; împărțită la √${n} = ${fs}: ${qs}. O singură extragere: ${q1}.`,
 gc: (k, t, lim, nr) => `P(|S<sub>n</sub>/n<sup>1/1,5</sup>| > ${k}) în simulare: ${t}. Limita stabilă: ${lim}; distribuția Normală cu același interval intercuartilic: ${nr}.`,
 da: (a, f0, f5, n0, n5, r, p5) => `α = ${a}: f(0) = ${f0}, f(5) = ${f5}; distribuția Normală (α = 2): f(0) = ${n0}, f(5) = ${n5}. În x = 5, densitatea este de ${r} ori cea Normală; P(|X| > 5) = ${p5}.`,
 db: (b, r3, l3, r10, lim, mode) => `α = 1,3, β = ${b}: P(X > 3) = ${r3}, P(X &lt; −3) = ${l3}; raportul P(X > 10)/P(X &lt; −10) = ${r10}, limita (1 + β)/(1 − β) = ${lim}. Modul: ${mode}.`,
 dg: (g, w, w1) => `α = 1,7, γ = ${g}: distanța dintre cuantilele de 5% și 95% este ${w} = ${g} × ${w1}.`,
 ss: (a, sh, m1, m0, o) => `α = ${a}, β = 0,8: deplasarea β tan(πα/2) = ${sh}. Modul în S1: ${m1}${o ? ' (în afara graficului)' : ''}; în S0: ${m0}.`,
 sc: (x, n, c, l) => `P(X > ${x}): distribuția Normală ${n}, Cauchy ${c}, Lévy ${l}.`,
 tl: (a, s10, p10, r, r2) => `α = ${a}: P(X > 10) = ${s10}; legea de tip putere c<sub>α</sub>10<sup>−α</sup> = ${p10}. P(X > 20)/P(X > 10) = ${r}; 2<sup>−α</sup> = ${r2}.`,
 rv: (a, v1, ve, ne, mx) => `α = ${a}. Varianța de selecție a celor trei traiectorii stabile după 1 000 de extrageri: ${v1}; după 20 000: ${ve}. Distribuția Normală după 20 000: ${ne}. Cea mai mare extragere în valoare absolută: ${mx}.`,
 cm: (a, b, q01, q99, e01, e99, med, o) => `S(${a}; ${b}; 1; 0 | 1), 100 000 de extrageri CMS: cuantilele de 1% și 99% ${q01} și ${q99}, exact ${e01} și ${e99}; mediana ${med}; în afara intervalului (−8, 8): ${o} din extrageri.`,
 cmKs: (s, p) => `Testul KS cu două eșantioane față de levy_stable.rvs: statistica ${s}, p-value ${p}.`,
 mm: y => `Randamente logaritmice zilnice din ${y} (Bitcoin cel mai devreme din septembrie 2014) pînă la 18 septembrie 2026, estimatorul McCulloch:`,
 mmItem: (a, n, nua, nub, al) => `${a}: n = ${n}, ν<sub>α</sub> = ${nua}, ν<sub>β</sub> = ${nub}, α̂ = ${al}`,
 mc: R => `Primele ${R} din cele 200 de eșantioane din S(1,7; 0; 1; 0), pentru fiecare mărime n:`,
 mcItem: (n, m1, s1, m2, s2, r) => `n = ${n}: McCulloch, media ${m1} și abaterea standard ${s1}; ML, media ${m2} și abaterea standard ${s2}; raportul abaterilor standard ${r}`,
 fd: k => `Densitățile în x = ${k} abateri standard (abaterea standard a distribuției Normale estimate):`,
 fdItem: (a, x, h, n, t, s) => `${a}, x = ${x}%: histograma ${h}; Normală ${n}, Student-t ${t}, stabilă ${s}`,
 qq: p => `VaR ${p} = −q, în % pe zi:`,
 qqItem: (a, e, n, t, s) => `${a}: date ${e}; Normală ${n}, Student-t ${t}, stabilă ${s}`,
 tr: x => `Zile cu o pierdere peste ${x}%: observate și așteptate (n × P(r &lt; −${x}%)) conform fiecărui model estimat:`,
 trItem: (a, o, n, t, s) => `${a}: observate ${o}; Normală ${n}, Student-t ${t}, stabilă ${s}`,
 ag: (e, lv, z) => e === 'ml' ? `Estimări ML cu intervale de încredere de ${lv} (±${z} erori standard). α̂ zilnic → săptămînal → lunar:` : 'Estimatorul McCulloch, fără erori standard. α̂ zilnic → săptămînal → lunar:',
 agItem: (a, t, b) => `${a}: ${t}${b ? ' (lunar: la limita α = 2, fără interval)' : ''}`,
 agOv: (no, yes) => `Intervalele orizontului zilnic și lunar nu se suprapun: ${no}; se suprapun: ${yes}.`,
 rr: (a, al, v, v10, sims) => `${a} (α̂ = ${al}): varianța cumulată la sfîrșitul eșantionului ${v}${v10 ? `, la 1 ianuarie 2010 ${v10}` : ''}. Varianțele finale ale traiectoriilor simulate din legea stabilă estimată: ${sims}.`
};
""",
 'en': r"""window.CH3_TEXT = {
 sq: (n, a, fa, qa, fs, qs, q1) => `Sum of ${n} draws divided by ${n}<sup>1/${a}</sup> = ${fa}: 99% quantile ${qa}; divided by √${n} = ${fs}: ${qs}. One draw: ${q1}.`,
 gc: (k, t, lim, nr) => `P(|S<sub>n</sub>/n<sup>1/1.5</sup>| > ${k}) in the simulation: ${t}. Stable limit: ${lim}; Normal distribution with the same interquartile range: ${nr}.`,
 da: (a, f0, f5, n0, n5, r, p5) => `α = ${a}: f(0) = ${f0}, f(5) = ${f5}; Normal distribution (α = 2): f(0) = ${n0}, f(5) = ${n5}. At x = 5, the density is ${r} times the Normal one; P(|X| > 5) = ${p5}.`,
 db: (b, r3, l3, r10, lim, mode) => `α = 1.3, β = ${b}: P(X > 3) = ${r3}, P(X &lt; −3) = ${l3}; ratio P(X > 10)/P(X &lt; −10) = ${r10}, limit (1 + β)/(1 − β) = ${lim}. Mode: ${mode}.`,
 dg: (g, w, w1) => `α = 1.7, γ = ${g}: the distance between the 5% and 95% quantiles is ${w} = ${g} × ${w1}.`,
 ss: (a, sh, m1, m0, o) => `α = ${a}, β = 0.8: shift β tan(πα/2) = ${sh}. Mode in S1: ${m1}${o ? ' (outside the chart)' : ''}; in S0: ${m0}.`,
 sc: (x, n, c, l) => `P(X > ${x}): Normal distribution ${n}, Cauchy ${c}, Lévy ${l}.`,
 tl: (a, s10, p10, r, r2) => `α = ${a}: P(X > 10) = ${s10}; power law c<sub>α</sub>10<sup>−α</sup> = ${p10}. P(X > 20)/P(X > 10) = ${r}; 2<sup>−α</sup> = ${r2}.`,
 rv: (a, v1, ve, ne, mx) => `α = ${a}. Sample variance of the three stable paths after 1,000 draws: ${v1}; after 20,000: ${ve}. Normal distribution after 20,000: ${ne}. Largest draw in absolute value: ${mx}.`,
 cm: (a, b, q01, q99, e01, e99, med, o) => `S(${a}, ${b}, 1, 0 | 1), 100,000 CMS draws: 1% and 99% quantiles ${q01} and ${q99}, exact ${e01} and ${e99}; median ${med}; outside (−8, 8): ${o} of the draws.`,
 cmKs: (s, p) => `Two-sample KS test against levy_stable.rvs: statistic ${s}, p-value ${p}.`,
 mm: y => `Daily log returns from ${y} (Bitcoin from September 2014 at the earliest) to 18 September 2026, McCulloch estimator:`,
 mmItem: (a, n, nua, nub, al) => `${a}: n = ${n}, ν<sub>α</sub> = ${nua}, ν<sub>β</sub> = ${nub}, α̂ = ${al}`,
 mc: R => `The first ${R} of the 200 samples from S(1.7, 0, 1, 0), for each sample size n:`,
 mcItem: (n, m1, s1, m2, s2, r) => `n = ${n}: McCulloch, mean ${m1} and standard deviation ${s1}; ML, mean ${m2} and standard deviation ${s2}; ratio of the standard deviations ${r}`,
 fd: k => `Densities at x = ${k} standard deviations (standard deviation of the fitted Normal distribution):`,
 fdItem: (a, x, h, n, t, s) => `${a}, x = ${x}%: histogram ${h}; Normal ${n}, Student-t ${t}, stable ${s}`,
 qq: p => `VaR ${p} = −q, in % per day:`,
 qqItem: (a, e, n, t, s) => `${a}: data ${e}; Normal ${n}, Student-t ${t}, stable ${s}`,
 tr: x => `Days with a loss above ${x}%: observed and expected (n × P(r &lt; −${x}%)) under each fitted model:`,
 trItem: (a, o, n, t, s) => `${a}: observed ${o}; Normal ${n}, Student-t ${t}, stable ${s}`,
 ag: (e, lv, z) => e === 'ml' ? `ML estimates with ${lv} confidence intervals (±${z} standard errors). α̂ daily → weekly → monthly:` : 'McCulloch estimator, without standard errors. α̂ daily → weekly → monthly:',
 agItem: (a, t, b) => `${a}: ${t}${b ? ' (monthly: at the bound α = 2, no interval)' : ''}`,
 agOv: (no, yes) => `The daily and monthly intervals do not overlap: ${no}; they overlap: ${yes}.`,
 rr: (a, al, v, v10, sims) => `${a} (α̂ = ${al}): running variance at the end of the sample ${v}${v10 ? `, on 1 January 2010 ${v10}` : ''}. Final variances of the paths simulated from the fitted stable law: ${sims}.`
};
""",
}

QA = {
 'ro': {
  'stability-qq': ('Pe slide, suma a 10 extrageri din S(1,7; 0; 1; 0) împărțită la √10 are cuantila de 99% egală cu 6,31, față de 5,04 pentru o singură extragere. Cît devine această cuantilă dacă adunăm 50 de extrageri și împărțim tot la √n?',
                   'Circa 7,27. Împărțită la 50<sup>1/1,7</sup> = 9,98, aceeași sumă are cuantila de 99% egală cu 5,15, adică legea unei singure extrageri. Regula √n subestimează scala sumei cu factorul n<sup>1/α − 1/2</sup>, care crește cu n: 10<sup>0,088</sup> = 1,23 și 50<sup>0,088</sup> = 1,41. Cu varianță infinită, mai mulți termeni nu aduc suma mai aproape de regula distribuției Normale.'),
  'gclt': ('La pragul 10, distribuția Normală cu același interval intercuartilic ca limita stabilă dă P(|·| > 10) = 0,0005%, față de 2,64% la limită. Ce probabilitate dă aceeași distribuție Normală la pragul 3?',
           '17,2%, aproape de 19,2% pentru sumele cu n = 100 și de 21,7% pentru limita stabilă. Același interval intercuartilic face corpurile celor două legi asemănătoare; diferența apare departe în cozi: la pragul 10, 0,0005% față de 2,73% în simulare. Butonul ▶ arată cum raportul crește odată cu pragul.'),
  'density-alpha': ('De la α = 2 la α = 1,5, ce se schimbă mai mult: densitatea în x = 0 sau densitatea în x = 5?',
                    'Densitatea în x = 5. f(0) crește de la 0,2821 la 0,2874, cu 1,9%, iar f(5) crește de la 0,000545 la 0,00711, de 13 ori. Pe scara liniară curbele par apropiate; pe scara logaritmică se vede că α schimbă mai ales cozile. P(|X| > 5) este 0,041% pentru α = 2 și 4,14% pentru α = 1,5.'),
  'density-beta': ('Pentru α = 1,3 și β = 0,5, de cîte ori este mai probabilă o valoare peste 10 decît o valoare sub −10?',
                   'De circa 3 ori: raportul este 3,29 la pragul 10 și tinde la (1 + β)/(1 − β) = 1,5/0,5 = 3, din formula cozilor P(X > x) ∼ c<sub>α</sub>(1 + β)x<sup>−α</sup> și P(X < −x) ∼ c<sub>α</sub>(1 − β)x<sup>−α</sup>. La pragul 3 probabilitățile sînt 10,8% și 3,37%. Cu β = 1, coada stîngă scade mult mai repede decît x<sup>−α</sup>.'),
  's0-s1': ('Butonul ▶ mută α de la 0,80 la 1,20, cu β = 0,8. Unde se află modul densității în parametrizarea S1 pentru α = 0,99?',
            'La circa 50,5, departe în afara graficului. Deplasarea dintre cele două parametrizări este β tan(πα/2) = 0,8 × 63,66 = 50,93, iar modul în S0 este −0,38. La α = 1,01 tangenta își schimbă semnul și modul în S1 ajunge la circa −51. În S0 modul rămîne între −0,48 și −0,26 pe tot intervalul.'),
  'special-cases': ('Pentru pragul x = 20, care dintre cele trei legi are cea mai mare probabilitate P(X > x)?',
                    'Legea Lévy: 17,7%, față de 1,59% pentru Cauchy și circa 3 × 10<sup>−87</sup>% pentru distribuția Normală. Coada Lévy scade ca x<sup>−1/2</sup> (α = 1/2), coada Cauchy ca x<sup>−1</sup>. Pentru x = 3 probabilitățile sînt 43,6% (Lévy), 10,2% (Cauchy) și 0,13% (distribuția Normală).'),
  'tails-loglog': ('Pentru α = 1,9, P(X > 10) = 0,064%. Cît este P(X > 20)?',
                   '0,0165%: dublarea pragului împarte probabilitatea la circa 2<sup>1,9</sup> = 3,7 (raportul exact este 0,256, legea de tip putere dă 2<sup>−1,9</sup> = 0,268). Cu α aproape de 2 coada rămîne de tip putere, departe de distribuția Normală, pentru care P(X > 10) este de ordinul 10<sup>−12</sup>.'),
  'running-var-sim': ('Butonul ▶ trece α de la 1,1 la 2. Pentru α = 1,9, ajung cele trei varianțe de selecție lîngă 2 după 20 000 de extrageri, ca în cazul distribuției Normale?',
                      'Nu: 3,16, 2,62 și 4,12, față de 1,96 pentru distribuția Normală. Pentru orice α < 2 varianța este infinită și fiecare extragere mare produce un salt. Cu α = 1,95 valorile finale sînt 2,46, 2,22 și 2,83; numai α = 2 dă 2,00, 1,97 și 1,99. Cu α = 1,7 (slide-ul): 10,4, 6,6 și 20,2.'),
  'cms': ('Pentru α = 1,5 și β = 0,5, cuantilele exacte de 1% și 99% sînt −5,39 și 9,79 (grila FFT a graficului dă −5,40 și 9,76). Ce devin ele pentru β = −0,5?',
          'În oglindă: −9,79 și 5,39 exact, −9,81 și 5,38 pe grila FFT a graficului (în eșantionul simulat −9,99 și 5,21). În S1 cu δ = 0, legea cu −β este legea cu β reflectată față de 0, deci β schimbă coada mai lungă, nu doar poziția. Histograma urmează densitatea exactă pentru fiecare combinație de α și β.'),
  'mcculloch-map': ('Butonul ▶ mută începutul eșantionului de la 2000 la 2020. Eșantionul S&P 500 din 2020 are 1 687 de zile, iar criza din martie 2020 cîntărește mai mult în el. Scade α̂ McCulloch față de 1,438?',
                    'Nu, crește la 1,543: ν<sub>α</sub> scade de la 3,338 la 3,044. Estimatorul McCulloch folosește doar cuantilele de 5%, 25%, 50%, 75% și 95%, deci zilele extreme nu intră direct în calcul; contează cît de lată este banda 5–95% față de intervalul intercuartilic. DAX trece de la 1,455 la 1,590, BET de la 1,414 la 1,480, Bitcoin de la 1,322 la 1,400.'),
  'estimators-mc': ('Butonul ▶ crește numărul de eșantioane Monte Carlo de la 10 la 200. Se schimbă abaterea standard a lui α̂ ML pentru n = 500 între 100 și 200 de eșantioane?',
                    'Da: de la 0,050 la 0,061. Precizia lui α̂ depinde de mărimea eșantionului n; numărul de eșantioane Monte Carlo determină doar cît de bine este măsurată această precizie. Cu 100 de eșantioane, abaterea standard estimată are încă o eroare relativă de cel puțin 1/√(2 × 100) = 7%. Pentru McCulloch, 0,110 la 100 și 0,107 la 200.'),
  'fit-density-1': ('La x = −6 abateri standard (−7,28% pe zi pentru S&P 500), de cîte ori este densitatea distribuției Normale estimate mai mică decît densitatea Student-t?',
                    'De circa 120 000 de ori: 4,4 × 10<sup>−9</sup> față de 5,5 × 10<sup>−4</sup>. Densitatea stabilă este 1,0 × 10<sup>−3</sup>, aproape de două ori densitatea Student-t. Coada Normală scade ca e<sup>−x²/2</sup>, iar cozile celorlalte două ca o putere a lui x, deci raportul crește foarte repede cu |x|.'),
  'fit-density-2': ('La x = −8 abateri standard pentru DAX (−11,26% pe zi), de cîte ori este densitatea stabilă mai mare decît densitatea Student-t?',
                    'De circa 2,9 ori: 3,7 × 10<sup>−4</sup> față de 1,3 × 10<sup>−4</sup>. Coada stabilă scade ca |x|<sup>−α̂−1</sup> cu α̂ = 1,61, coada Student-t ca |x|<sup>−ν̂−1</sup> cu ν̂ = 3,09; un exponent mai mic înseamnă o coadă mai groasă. Raportul este 1,6 la −5 abateri standard și 3,9 la −10.'),
  'qq-real-1': ('Butonul ▶ coboară nivelul p de la 5% la 0,1%. La p = 0,1%, ce VaR dă legea stabilă estimată pentru S&P 500, față de 7,22% în date?',
                '19,70%, de 2,7 ori cuantila empirică. Student-t dă 8,58%, distribuția Normală 3,72%. La p = 1%, valorile sînt 3,45% (date), 4,54% (stabilă), 3,46% (Student-t) și 2,80% (Normală): legea stabilă exagerează deja, dar diferența crește cu cît p este mai mic.'),
  'qq-real-2': ('La p = 1%, VaR Bitcoin din date este 10,51%. Care model estimat dă valoarea cea mai apropiată?',
                'Student-t: 11,06%. Legea stabilă dă 14,26%, distribuția Normală 7,99%. La p = 0,1%, datele dau 18,05%, Student-t 31,95% și legea stabilă 74,16%: și Student-t exagerează coada îndepărtată a Bitcoin, dar mult mai puțin decît legea stabilă.'),
  'tails-real-1': ('S&P 500 are 23 de zile cu pierderi peste 5% în 6 714 zile. Cîte astfel de zile așteaptă legea stabilă estimată?',
                   '57,3, de 2,5 ori mai multe. Student-t așteaptă 27,0, distribuția Normală 0,1. Pentru pierderi peste 10%: o zi observată, 4,5 conform Student-t și 19,1 conform legii stabile. Raportul dintre zilele așteptate de legea stabilă și cele observate crește odată cu pragul.'),
  'tails-real-2': ('Bitcoin are 4 zile cu pierderi peste 20% în 4 384 de zile. Cîte zile așteaptă Student-t și cîte legea stabilă?',
                   '12,3 conform Student-t și 26,9 conform legii stabile; distribuția Normală dă practic zero. Ambele modele cu cozi groase exagerează coada îndepărtată a Bitcoin, legea stabilă de peste 6 ori. Pentru pierderi peste 10%: 56 observate, 54,2 (Student-t), 73,9 (stabilă).'),
  'aggregation': ('Pe slide, intervalele de 95% ale lui α̂ ML pentru S&P 500 la orizontul zilnic (1,54) și lunar (1,82) nu se suprapun. Se suprapun la nivelul de încredere 99%?',
                  'Nu. La 99% (±2,576 erori standard), intervalul zilnic este [1,486; 1,594], iar cel lunar [1,655; 1,984]. Nici la 99,9% (±3,29 erori standard) nu se suprapun, deși distanța dintre ele ajunge aproape la zero. Pentru DAX și BET intervalele se suprapun deja la 95%. Intervalele sînt asimptotice, iar cel lunar se bazează pe 321 de luni.'),
  'running-var-real': ('Pentru Bitcoin (α̂ = 1,410), varianța cumulată a datelor la sfîrșitul eșantionului este 12,14. Unde ajung varianțele finale ale celor trei traiectorii simulate din legea stabilă estimată?',
                       'Mult mai sus: 252,1, 111,4 și 52,6, de 4 pînă la 21 de ori varianța datelor. La fel pentru S&P 500 (slide-ul): datele 1,47, traiectoriile 5,9, 5,4 și 15,6. O lege stabilă cu α̂ < 2 produce extreme pe care seriile reale nu le au, iar varianța lor de selecție nu se stabilizează.'),
 },
 'en': {
  'stability-qq': ('On the slide, the sum of 10 draws from S(1.7, 0, 1, 0) divided by √10 has a 99% quantile of 6.31, against 5.04 for a single draw. What does this quantile become if we add 50 draws and still divide by √n?',
                   'About 7.27. Divided by 50<sup>1/1.7</sup> = 9.98, the same sum has a 99% quantile of 5.15, the law of a single draw. The √n rule understates the scale of the sum by the factor n<sup>1/α − 1/2</sup>, which grows with n: 10<sup>0.088</sup> = 1.23 and 50<sup>0.088</sup> = 1.41. With infinite variance, more terms do not bring the sum closer to the rule of the Normal distribution.'),
  'gclt': ('At the threshold 10, the Normal distribution with the same interquartile range as the stable limit gives P(|·| > 10) = 0.0005%, against 2.64% for the limit. What probability does the same Normal distribution give at the threshold 3?',
           '17.2%, close to 19.2% for the sums with n = 100 and to 21.7% for the stable limit. The same interquartile range makes the bodies of the two laws similar; the difference is far in the tails: at the threshold 10, 0.0005% against 2.73% in the simulation. The ▶ button shows how the ratio grows with the threshold.'),
  'density-alpha': ('From α = 2 to α = 1.5, what changes more: the density at x = 0 or the density at x = 5?',
                    'The density at x = 5. f(0) rises from 0.2821 to 0.2874, by 1.9%, and f(5) rises from 0.000545 to 0.00711, 13 times. On the linear scale the curves look close; the log scale shows that α changes mainly the tails. P(|X| > 5) is 0.041% for α = 2 and 4.14% for α = 1.5.'),
  'density-beta': ('For α = 1.3 and β = 0.5, how many times more likely is a value above 10 than a value below −10?',
                   'About 3 times: the ratio is 3.29 at the threshold 10 and tends to (1 + β)/(1 − β) = 1.5/0.5 = 3, from the tail formula P(X > x) ∼ c<sub>α</sub>(1 + β)x<sup>−α</sup> and P(X < −x) ∼ c<sub>α</sub>(1 − β)x<sup>−α</sup>. At the threshold 3 the probabilities are 10.8% and 3.37%. With β = 1, the left tail falls much faster than x<sup>−α</sup>.'),
  's0-s1': ('The ▶ button moves α from 0.80 to 1.20, with β = 0.8. Where is the mode of the density in the S1 parameterisation for α = 0.99?',
            'At about 50.5, far outside the chart. The shift between the two parameterisations is β tan(πα/2) = 0.8 × 63.66 = 50.93, and the mode in S0 is −0.38. At α = 1.01 the tangent changes sign and the mode in S1 moves to about −51. In S0 the mode stays between −0.48 and −0.26 over the whole range.'),
  'special-cases': ('For the threshold x = 20, which of the three laws has the largest probability P(X > x)?',
                    'The Lévy law: 17.7%, against 1.59% for the Cauchy and about 3 × 10<sup>−87</sup>% for the Normal distribution. The Lévy tail falls as x<sup>−1/2</sup> (α = 1/2), the Cauchy tail as x<sup>−1</sup>. For x = 3 the probabilities are 43.6% (Lévy), 10.2% (Cauchy) and 0.13% (Normal distribution).'),
  'tails-loglog': ('For α = 1.9, P(X > 10) = 0.064%. What is P(X > 20)?',
                   '0.0165%: doubling the threshold divides the probability by about 2<sup>1.9</sup> = 3.7 (the exact ratio is 0.256, the power law gives 2<sup>−1.9</sup> = 0.268). With α close to 2 the tail is still a power law, far from the Normal distribution, for which P(X > 10) is of the order of 10<sup>−12</sup>.'),
  'running-var-sim': ('The ▶ button takes α from 1.1 to 2. For α = 1.9, do the three sample variances end near 2 after 20,000 draws, as for the Normal distribution?',
                      'No: 3.16, 2.62 and 4.12, against 1.96 for the Normal distribution. For any α < 2 the variance is infinite and each large draw makes a jump. With α = 1.95 the final values are 2.46, 2.22 and 2.83; only α = 2 gives 2.00, 1.97 and 1.99. With α = 1.7 (the slide): 10.4, 6.6 and 20.2.'),
  'cms': ('For α = 1.5 and β = 0.5, the exact 1% and 99% quantiles are −5.39 and 9.79 (the FFT grid of the chart gives −5.40 and 9.76). What do they become for β = −0.5?',
          'Mirrored: −9.79 and 5.39 exactly, −9.81 and 5.38 on the FFT grid of the chart (in the simulated sample −9.99 and 5.21). In S1 with δ = 0, the law with −β is the law with β reflected about 0, so β changes which tail is longer, not only the location. The histogram follows the exact density for every combination of α and β.'),
  'mcculloch-map': ('The ▶ button moves the start of the sample from 2000 to 2020. The S&P 500 sample from 2020 has 1,687 days, and the crisis of March 2020 weighs more in it. Does the McCulloch α̂ fall below 1.438?',
                    'No, it rises to 1.543: ν<sub>α</sub> falls from 3.338 to 3.044. The McCulloch estimator uses only the 5%, 25%, 50%, 75% and 95% quantiles, so the extreme days do not enter the calculation directly; what matters is how wide the 5–95% band is relative to the interquartile range. The DAX goes from 1.455 to 1.590, BET from 1.414 to 1.480, Bitcoin from 1.322 to 1.400.'),
  'estimators-mc': ('The ▶ button raises the number of Monte Carlo samples from 10 to 200. Does the standard deviation of the ML α̂ for n = 500 change between 100 and 200 samples?',
                    'Yes: from 0.050 to 0.061. The precision of α̂ depends on the sample size n; the number of Monte Carlo samples only determines how well this precision is measured. With 100 samples, the estimated standard deviation still has a relative error of at least 1/√(2 × 100) = 7%. For McCulloch, 0.110 at 100 and 0.107 at 200.'),
  'fit-density-1': ('At x = −6 standard deviations (−7.28% per day for the S&P 500), how many times smaller is the density of the fitted Normal distribution than the Student-t density?',
                    'About 120,000 times: 4.4 × 10<sup>−9</sup> against 5.5 × 10<sup>−4</sup>. The stable density is 1.0 × 10<sup>−3</sup>, almost twice the Student-t density. The Normal tail falls as e<sup>−x²/2</sup>, the other two as a power of x, so the ratio grows very fast with |x|.'),
  'fit-density-2': ('At x = −8 standard deviations for the DAX (−11.26% per day), how many times larger is the stable density than the Student-t density?',
                    'About 2.9 times: 3.7 × 10<sup>−4</sup> against 1.3 × 10<sup>−4</sup>. The stable tail falls as |x|<sup>−α̂−1</sup> with α̂ = 1.61, the Student-t tail as |x|<sup>−ν̂−1</sup> with ν̂ = 3.09; a smaller exponent means a heavier tail. The ratio is 1.6 at −5 standard deviations and 3.9 at −10.'),
  'qq-real-1': ('The ▶ button lowers the level p from 5% to 0.1%. At p = 0.1%, what VaR does the fitted stable law give for the S&P 500, against 7.22% in the data?',
                '19.70%, 2.7 times the empirical quantile. The Student-t gives 8.58%, the Normal distribution 3.72%. At p = 1%, the values are 3.45% (data), 4.54% (stable), 3.46% (Student-t) and 2.80% (Normal): the stable law already overstates, but the gap grows as p gets smaller.'),
  'qq-real-2': ('At p = 1%, the VaR of Bitcoin in the data is 10.51%. Which fitted model gives the closest value?',
                'The Student-t: 11.06%. The stable law gives 14.26%, the Normal distribution 7.99%. At p = 0.1%, the data give 18.05%, the Student-t 31.95% and the stable law 74.16%: the Student-t also overstates the far tail of Bitcoin, but much less than the stable law.'),
  'tails-real-1': ('The S&P 500 has 23 days with losses above 5% in 6,714 days. How many such days does the fitted stable law expect?',
                   '57.3, 2.5 times as many. The Student-t expects 27.0, the Normal distribution 0.1. For losses above 10%: one day observed, 4.5 under the Student-t and 19.1 under the stable law. The ratio between the days expected by the stable law and those observed grows with the threshold.'),
  'tails-real-2': ('Bitcoin has 4 days with losses above 20% in 4,384 days. How many days do the Student-t and the stable law expect?',
                   '12.3 under the Student-t and 26.9 under the stable law; the Normal distribution gives practically zero. Both heavy-tailed models overstate the far tail of Bitcoin, the stable law more than 6 times. For losses above 10%: 56 observed, 54.2 (Student-t), 73.9 (stable).'),
  'aggregation': ('On the slide, the 95% intervals of the ML α̂ for the S&P 500 at the daily (1.54) and monthly (1.82) horizon do not overlap. Do they overlap at the 99% confidence level?',
                  'No. At 99% (±2.576 standard errors), the daily interval is [1.486, 1.594] and the monthly one [1.655, 1.984]. Not even at 99.9% (±3.29 standard errors) do they overlap, although the gap between them almost vanishes. For the DAX and BET the intervals already overlap at 95%. The intervals are asymptotic, and the monthly one rests on 321 months.'),
  'running-var-real': ('For Bitcoin (α̂ = 1.410), the running variance of the data at the end of the sample is 12.14. Where do the final variances of the three paths simulated from the fitted stable law end?',
                       'Much higher: 252.1, 111.4 and 52.6, 4 to 21 times the variance of the data. The same holds for the S&P 500 (the slide): data 1.47, paths 5.9, 5.4 and 15.6. A stable law with α̂ < 2 produces extremes that the real series do not have, and their sample variance does not settle.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (checked with a node script on ch3_data.js + common.js + ch3.js).
PREDICT = {
 'stability-qq': (0, [('Circa 7,27: abaterea față de o singură extragere crește cu n.', 'About 7.27: the gap to a single draw grows with n.'),
                      ('Circa 5,04: cu mai mulți termeni, CLT face ca √n să fie împărțitorul potrivit.', 'About 5.04: with more terms, the CLT makes √n the right divisor.'),
                      ('Tot 6,31: raportul dintre cele două împărțiri nu depinde de n.', 'Still 6.31: the ratio between the two divisions does not depend on n.')]),
 'gclt': (0, [('17,2%, aproape de limita stabilă (21,7%).', '17.2%, close to the stable limit (21.7%).'),
              ('Tot circa 0,0005%: distribuția Normală ratează cozile la orice prag.', 'Still about 0.0005%: the Normal distribution misses the tails at every threshold.'),
              ('Exact 21,7%: același interval intercuartilic fixează întreaga lege.', 'Exactly 21.7%: the same interquartile range fixes the whole law.')]),
 'density-alpha': (1, [('Densitatea în 0: α controlează mai ales înălțimea vîrfului.', 'The density at 0: α mainly controls the height of the peak.'),
                       ('Densitatea în 5: f(0) crește cu 2%, f(5) de 13 ori.', 'The density at 5: f(0) rises by 2%, f(5) 13 times.'),
                       ('Ambele cresc în aceeași proporție: curba se rescalează.', 'Both rise in the same proportion: the curve is rescaled.')]),
 'density-beta': (2, [('O dată: β mută doar centrul densității.', 'Once: β only moves the centre of the density.'),
                      ('De 1,5 ori: raportul este 1 + β.', '1.5 times: the ratio is 1 + β.'),
                      ('De circa 3 ori: raportul tinde la (1 + β)/(1 − β).', 'About 3 times: the ratio tends to (1 + β)/(1 − β).')]),
 's0-s1': (0, [('La circa 50,5, departe în dreapta.', 'At about 50.5, far to the right.'),
               ('Aproape de 0, ca în S0: parametrizarea nu schimbă densitatea.', 'Close to 0, as in S0: the parameterisation does not change the density.'),
               ('La circa −50,5, departe în stînga.', 'At about −50.5, far to the left.')]),
 'special-cases': (1, [('Cauchy: are cele mai groase cozi dintre legile simetrice.', 'The Cauchy: it has the heaviest tails among the symmetric laws.'),
                       ('Lévy: 17,7%, față de 1,6% pentru Cauchy.', 'The Lévy law: 17.7%, against 1.6% for the Cauchy.'),
                       ('Niciuna: la x = 20 toate trei probabilitățile sînt practic zero.', 'None: at x = 20 all three probabilities are practically zero.')]),
 'tails-loglog': (2, [('Practic zero: cu α = 1,9 coada este aproape Normală.', 'Practically zero: with α = 1.9 the tail is almost Normal.'),
                      ('Circa 0,032%: probabilitatea se înjumătățește.', 'About 0.032%: the probability halves.'),
                      ('Circa 0,0165%: dublarea pragului împarte la circa 2^(1,9).', 'About 0.0165%: doubling the threshold divides by about 2^(1.9).')]),
 'running-var-sim': (0, [('Nu: 3,16, 2,62 și 4,12.', 'No: 3.16, 2.62 and 4.12.'),
                         ('Da: la α = 1,9 legea este practic Normală.', 'Yes: at α = 1.9 the law is practically Normal.'),
                         ('Nu: toate trei depășesc 20, ca o traiectorie cu α = 1,7.', 'No: all three exceed 20, like one path with α = 1.7.')]),
 'cms': (1, [('Rămîn −5,39 și 9,79: β mută doar poziția.', 'They stay at −5.39 and 9.79: β only moves the location.'),
             ('În oglindă: circa −9,8 și 5,4.', 'Mirrored: about −9.8 and 5.4.'),
             ('Se deplasează amîndouă spre stînga cu aceeași cantitate.', 'Both move to the left by the same amount.')]),
 'mcculloch-map': (2, [('Da: criza cîntărește mai mult, cozile sînt mai groase, deci α̂ este mai mic.', 'Yes: the crisis weighs more, the tails are heavier, so α̂ is smaller.'),
                       ('Rămîne 1,438: α̂ este o proprietate a seriei, nu a eșantionului.', 'It stays at 1.438: α̂ is a property of the series, not of the sample.'),
                       ('Nu, crește la 1,543.', 'No, it rises to 1.543.')]),
 'estimators-mc': (0, [('Da: de la 0,050 la 0,061.', 'Yes: from 0.050 to 0.061.'),
                       ('Nu: după 100 de eșantioane valoarea este stabilă la a treia zecimală.', 'No: after 100 samples the value is stable to the third decimal.'),
                       ('Da, scade spre 0: mai multe eșantioane fac estimatorul mai precis.', 'Yes, it falls towards 0: more samples make the estimator more precise.')]),
 'fit-density-1': (2, [('De circa 6 ori: densitatea scade proporțional cu distanța față de medie.', 'About 6 times: the density falls in proportion to the distance from the mean.'),
                       ('De circa 2 ori: la −6 abateri standard toate densitățile sînt practic zero.', 'About 2 times: at −6 standard deviations all densities are practically zero.'),
                       ('De circa 120 000 de ori.', 'About 120,000 times.')]),
 'fit-density-2': (0, [('De circa 2,9 ori.', 'About 2.9 times.'),
                       ('O dată: departe în cozi, cele două legi cu cozi groase coincid.', 'Once: far in the tails, the two heavy-tailed laws coincide.'),
                       ('De circa 0,35 ori: Student-t are coada mai groasă.', 'About 0.35 times: the Student-t has the heavier tail.')]),
 'qq-real-1': (1, [('8,58%: legea stabilă, cu α̂ < 2, urmează cel mai bine cozile.', '8.58%: the stable law, with α̂ < 2, follows the tails best.'),
                   ('19,70%, de 2,7 ori valoarea din date.', '19.70%, 2.7 times the value in the data.'),
                   ('3,72%: la p mic toate modelele dau aceeași cuantilă.', '3.72%: at a small p all models give the same quantile.')]),
 'qq-real-2': (0, [('Student-t: 11,06%.', 'The Student-t: 11.06%.'),
                   ('Legea stabilă: α̂ = 1,41 este cel mai mic, deci cozile Bitcoin sînt prinse cel mai bine.', 'The stable law: α̂ = 1.41 is the smallest, so the Bitcoin tails are caught best.'),
                   ('Distribuția Normală: la 1% sîntem încă în corpul distribuției.', 'The Normal distribution: at 1% we are still in the body of the distribution.')]),
 'tails-real-1': (0, [('57,3, de 2,5 ori mai multe.', '57.3, 2.5 times as many.'),
                      ('Circa 23: legea stabilă descrie bine corpul, deci și pierderile de 5%.', 'About 23: the stable law fits the body well, hence also losses of 5%.'),
                      ('Circa 0,1: o pierdere de 5% este de peste 4 abateri standard.', 'About 0.1: a loss of 5% is more than 4 standard deviations.')]),
 'tails-real-2': (2, [('Circa 4 pentru ambele: modelele cu cozi groase prind extremele.', 'About 4 for both: the heavy-tailed models catch the extremes.'),
                      ('Student-t 26,9, legea stabilă 12,3: Student-t are cozile mai groase.', 'Student-t 26.9, stable law 12.3: the Student-t has the heavier tails.'),
                      ('Student-t 12,3, legea stabilă 26,9.', 'Student-t 12.3, stable law 26.9.')]),
 'aggregation': (0, [('Nu: nici la 99,9% nu se suprapun.', 'No: not even at 99.9% do they overlap.'),
                     ('Da: la 99% intervalele sînt de 1,3 ori mai largi și se ating.', 'Yes: at 99% the intervals are 1.3 times wider and touch.'),
                     ('Da: peste 95%, intervalul lunar, estimat pe 321 de luni, devine destul de larg.', 'Yes: above 95%, the monthly interval, estimated on 321 months, becomes wide enough.')]),
 'running-var-real': (1, [('Aproape de 12,14: legea a fost estimată pe aceleași date.', 'Close to 12.14: the law was fitted to the same data.'),
                          ('Mult mai sus: între 52,6 și 252,1.', 'Much higher: between 52.6 and 252.1.'),
                          ('Sub 12,14: cu α̂ < 2 varianța simulată este mai mică decît a datelor.', 'Below 12.14: with α̂ < 2 the simulated variance is smaller than that of the data.')]),
}

# Exit ticket „Verifică-te” / "Check Yourself": questions of the slides „Autoevaluare”, „Regula generală și un exemplu lucrat”
# and „Formule de reținut”. Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('Randamentele zilnice sînt i.i.d. S(1,5; 0; γ; 0). De cîte ori crește scala de la 1 zi la 20 de zile?',
         'Daily returns are i.i.d. S(1.5, 0, γ, 0). By what factor does the scale grow from 1 day to 20 days?'),
      key=1, opts=[
  (('4,47.', '4.47.'), ('4,47 = √20 este regula rădăcinii pătrate a timpului, valabilă doar pentru α = 2.', '4.47 = √20 is the square-root-of-time rule, valid only for α = 2.')),
  (('7,37.', '7.37.'), ('Suma a n copii are scala n^(1/α)γ: 20^(1/1,5) = 20^(2/3) = 7,37, de 1,65 ori factorul √20.', 'The sum of n copies has scale n^(1/α)γ: 20^(1/1.5) = 20^(2/3) = 7.37, 1.65 times the factor √20.')),
  (('89,4.', '89.4.'), ('89,4 = 20^(1,5) folosește exponentul α în loc de 1/α.', '89.4 = 20^(1.5) uses the exponent α instead of 1/α.'))]),
 dict(q=('Pentru α = 1,5, care este factorul de scală al sumei a două copii independente?',
         'For α = 1.5, what is the scale factor of the sum of two independent copies?'),
      key=0, opts=[
  (('1,59.', '1.59.'), ('c = (1^α + 1^α)^(1/α) = 2^(1/1,5) = 2^(2/3) = 1,59.', 'c = (1^α + 1^α)^(1/α) = 2^(1/1.5) = 2^(2/3) = 1.59.')),
  (('1,41.', '1.41.'), ('1,41 = √2 este factorul distribuției Normale (α = 2); cu α = 1,5 scala crește mai repede.', '1.41 = √2 is the factor of the Normal distribution (α = 2); with α = 1.5 the scale grows faster.')),
  (('2,00.', '2.00.'), ('Factorul 2 corespunde lui α = 1 (Cauchy), cînd scalele se adună.', 'The factor 2 belongs to α = 1 (Cauchy), when the scales add up.'))]),
 dict(q=('O lege stabilă estimată are α = 1,2. Există media și varianța?',
         'A fitted stable law has α = 1.2. Do the mean and the variance exist?'),
      key=2, opts=[
  (('Ambele există, deoarece au fost estimate pe date.', 'Both exist, since they were estimated from data.'),
   ('Media și varianța de selecție se pot calcula pe orice eșantion, dar E|X|^p este finit doar pentru p < α = 1,2.', 'The sample mean and variance can be computed on any sample, but E|X|^p is finite only for p < α = 1.2.')),
  (('Niciuna: α < 2 înseamnă că nu există niciun moment.', 'Neither: α < 2 means that no moment exists.'),
   ('Momentele de ordin p < α există; pentru α = 1,2 există media (p = 1).', 'The moments of order p < α exist; for α = 1.2 the mean (p = 1) exists.')),
  (('Media există, varianța nu.', 'The mean exists, the variance does not.'),
   ('E|X|^p < ∞ dacă și numai dacă p < α: 1 < 1,2, deci media există; 2 > 1,2, deci varianța este infinită.', 'E|X|^p < ∞ if and only if p < α: 1 < 1.2, so the mean exists; 2 > 1.2, so the variance is infinite.'))]),
 dict(q=('scipy raportează δ = 0,05 cu α = 1,7, β = −0,2, γ = 0,6. Cît este δ₀?',
         'scipy reports δ = 0.05 with α = 1.7, β = −0.2, γ = 0.6. What is δ₀?'),
      key=0, opts=[
  (('0,111.', '0.111.'), ('scipy folosește S1: δ₀ = δ₁ + βγ tan(πα/2) = 0,05 + (−0,2)(0,6) tan(0,85π) = 0,05 + 0,061 = 0,111.', 'scipy uses S1: δ₀ = δ₁ + βγ tan(πα/2) = 0.05 + (−0.2)(0.6) tan(0.85π) = 0.05 + 0.061 = 0.111.')),
  (('0,050.', '0.050.'), ('Cele două parametrizări coincid doar pentru β = 0; aici β = −0,2.', 'The two parameterisations coincide only for β = 0; here β = −0.2.')),
  (('−0,011.', '−0.011.'), ('Rezultatul scade termenul în loc să îl adune; tan(0,85π) = −0,51 este negativ, deci βγ tan(πα/2) = +0,061.', 'This subtracts the term instead of adding it; tan(0.85π) = −0.51 is negative, so βγ tan(πα/2) = +0.061.'))]),
 dict(q=('Ce varianță are legea S(2; 0; 1; 0)?', 'What is the variance of the law S(2, 0, 1, 0)?'),
      key=1, opts=[
  (('1.', '1.'), ('Capcana scalei: S(2; 0; γ; δ) este N(δ, 2γ²), nu N(δ, γ²).', 'The scale trap: S(2, 0, γ, δ) is N(δ, 2γ²), not N(δ, γ²).')),
  (('2.', '2.'), ('S(2; β; γ; δ) = N(δ, 2γ²); cu γ = 1 varianța este 2. De aceea γ = σ/√2 pentru distribuția Normală.', 'S(2, β, γ, δ) = N(δ, 2γ²); with γ = 1 the variance is 2. Hence γ = σ/√2 for the Normal distribution.')),
  (('Infinită.', 'Infinite.'), ('Varianța este infinită doar pentru α < 2; α = 2 este distribuția Normală.', 'The variance is infinite only for α < 2; α = 2 is the Normal distribution.'))]),
]



# ================================================================= table and worked-example blocks
# (anchor, frame number in the deck (same in RO and EN), section key, chart block that carries it on the web lecture,
#  title check EN, title check RO)
XBLOCKS = [('horizons', 13, 'stab', 'stability-qq', 'Horizons and Diversification', 'orizonturi și diversificare'),
           ('cf-s0s1', 29, 'par', 'special-cases', 'S1 vs S0 at', 'S1 și S0 în'),
           ('tail-table', 35, 'tail', 'running-var-sim', 'Tail Probabilities', 'Probabilitățile din cozi'),
           ('cms-draw', 40, 'sim', 'cms', 'One Draw Step', 'o extragere pas cu pas'),
           ('mcc-bet', 46, 'est', 'mcculloch-map', 'McCulloch on the BET', 'McCulloch pe BET'),
           ('fits-table', 50, 'real', 'estimators-mc', 'Stable Fits of Daily', 'Legi stabile estimate'),
           ('loss-count', 58, 'real', 'tails-real-2', 'Counting Large Losses', 'Numărarea pierderilor'),
           ('var-table', 59, 'real', 'tails-real-2', 'VaR 1% and VaR 0.1%', 'VaR 1% și VaR 0,1%'),
           ('aic-table', 60, 'real', 'tails-real-2', 'Comparison of the Three Models', 'Compararea celor trei')]
ORDER = ['stability-qq', 'horizons', 'gclt', 'density-alpha', 'density-beta', 's0-s1', 'special-cases', 'cf-s0s1', 'tails-loglog', 'running-var-sim',
         'tail-table', 'cms-draw', 'cms', 'mcculloch-map', 'mcc-bet', 'estimators-mc', 'fits-table', 'fit-density-1', 'fit-density-2', 'qq-real-1',
         'qq-real-2', 'tails-real-1', 'tails-real-2', 'loss-count', 'var-table', 'aic-table', 'aggregation', 'running-var-real']
PFX.update({'horizons': 'hz', 'cf-s0s1': 'cf', 'tail-table': 'tt', 'cms-draw': 'cd', 'mcc-bet': 'mb', 'fits-table': 'fs', 'loss-count': 'lc',
            'var-table': 'vt', 'aic-table': 'ai'})

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt thead th[colspan]{text-align:center;border-bottom:1px solid #1F2A44}
.xt tr.sel td{background:#EEF3FB}
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
 'hz': r'X_1 + \dots + X_n \overset{d}{=} n^{1/\alpha}X + d_n, \qquad \gamma_n = n^{1/\alpha}\gamma, \qquad \frac{\gamma_n}{\sqrt{n}\,\gamma} = n^{1/\alpha - 1/2}, \qquad \gamma_{N} = N^{1/\alpha - 1}\gamma',
 'cf': r'\begin{gathered}\text{S1: } \ln\varphi(t) = -|t|^\alpha\Big[1 - i\beta\,\mathrm{sign}(t)\tan\frac{\pi\alpha}{2}\Big],\\'
       r' \text{S0: } \ln\varphi(t) = -|t|^\alpha\Big[1 + i\beta\,\mathrm{sign}(t)\tan\frac{\pi\alpha}{2}\big(|t|^{1-\alpha} - 1\big)\Big],\\'
       r' \gamma = 1, \qquad \delta_0 = \delta_1 + \beta\gamma\tan\frac{\pi\alpha}{2}\end{gathered}',
 'cd': r'\begin{gathered}V = \pi\big(U - \tfrac12\big), \qquad B = \frac{\arctan(\beta\tan\frac{\pi\alpha}{2})}{\alpha}, \qquad S = \Big(1 + \beta^2\tan^2\frac{\pi\alpha}{2}\Big)^{1/(2\alpha)},\\'
       r' X = S\,\frac{\sin(\alpha(V + B))}{(\cos V)^{1/\alpha}}\left(\frac{\cos(V - \alpha(V + B))}{W}\right)^{(1-\alpha)/\alpha} \sim S(\alpha, \beta, 1, 0; 1)\end{gathered}',
 'mb': r'\begin{gathered}\nu_\alpha = \frac{\hat q_{0.95} - \hat q_{0.05}}{\hat q_{0.75} - \hat q_{0.25}}, \qquad \nu_\beta = \frac{\hat q_{0.95} + \hat q_{0.05} - 2\hat q_{0.5}}{\hat q_{0.95} - \hat q_{0.05}},\\'
       r' \hat\alpha \approx \alpha_j + \frac{\nu_\alpha - \nu_j}{\nu_{j+1} - \nu_j}\,(\alpha_{j+1} - \alpha_j), \qquad \hat\gamma = \frac{\hat q_{0.75} - \hat q_{0.25}}{\phi_3(\hat\alpha, \hat\beta)}\end{gathered}',
 'fs': r'\delta_1 = \delta_0 - \beta\gamma\tan\frac{\pi\alpha}{2} \qquad (\alpha \ne 1)',
 'lc': r'\begin{gathered}P(r_t < -x) = P\Big(Z < -\frac{x + \hat\delta_0}{\hat\gamma}\Big) \approx c_\alpha (1 - \hat\beta)\Big(\frac{x + \hat\delta_0}{\hat\gamma}\Big)^{-\hat\alpha},'
       r' \qquad c_\alpha = \frac{\Gamma(\alpha)\sin(\pi\alpha/2)}{\pi},\\ Z \sim S(\hat\alpha, \hat\beta, 1, 0; 0), \qquad E = n\,P(r_t < -x)\end{gathered}',
 'vt': r'\begin{gathered}\mathrm{VaR}_p = -q_p: \qquad \text{NRM: } -(\hat\mu + \hat\sigma z_p), \quad z_p = \Phi^{-1}(p);\\'
       r' \text{STB: } -(\hat\delta_0 + \hat\gamma\, z^{S}_p), \quad z^{S}_p = F^{-1}_{S(\hat\alpha, \hat\beta, 1, 0; 0)}(p)\end{gathered}',
}


def wtex(k, ro):
    t = WE_TEX[k]
    if ro:
        for a in ('0.95', '0.05', '0.75', '0.25', '0.5'):
            t = t.replace('_{' + a + '}', '_{' + a.replace('.', '{,}') + '}')
        t = t.replace(r'\text{NRM: }', r'\text{distribuția Normală: }').replace(r'\text{STB: }', r'\text{legea stabilă: }')
    else:
        t = t.replace(r'\text{NRM: }', r'\text{Normal: }').replace(r'\text{STB: }', r'\text{stable: }')
    return t


LEVELS = [('0.05', '5%'), ('0.025', '2.5%'), ('0.01', '1%'), ('0.005', '0.5%'), ('0.001', '0.1%'), ('0.0005', '0.05%')]


def widgets_all(lang):
    ro = lang == 'ro'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    wt = lambda k, r, e: we(k + '-we', f'{WE}: {r if ro else e}', wtex(k, ro))
    ser = lambda i, lab_ro, lab_en, d: select(i, lab_ro if ro else lab_en, ASSETS, d)
    lev = [(v, t.replace('.', ',') if ro else t) for v, t in LEVELS]
    W = widgets_old(lang)
    W['stability-qq'] = (W['stability-qq'][0], CSS + W['stability-qq'][1])
    W['estimators-mc'] = (W['estimators-mc'][0], W['estimators-mc'][1]
                          + xh('Media și abaterea standard a lui α̂, α = 1,7' if ro else 'Mean and standard deviation of α̂, α = 1.7') + tw('mc-tab'))
    W.update({
     'horizons': (slider('hz-n', 'Orizontul n (zile)' if ro else 'Horizon n (days)', 1, 250, 1, 20)
                  + ser('hz-a', 'Seria (rîndul α̂ și exemplul lucrat)', 'Series (the α̂ row and the worked example)', 'sp500') + play('hz'),
                  xh('Factorul de scală al sumei pe n zile și al portofoliului cu N active (γ = 1)' if ro
                     else 'Scale factor of the n-day sum and of the portfolio of N assets (γ = 1)')
                  + '<div class="wctl">' + slider('hz-N', 'Numărul de active N (tabel și exemplul lucrat)' if ro else 'Number of assets N (table and worked example)', 2, 100, 1, 10) + '</div>' + tw('hz-tab') + charts('hz-chart')
                  + wt('hz', 'scala riscului pe n zile, cu α̂ și γ̂ estimate', 'the scale of n-day risk, with the fitted α̂ and γ̂')),
     'cf-s0s1': (slider('cf-a', 'α', 1.1, 1.9, 0.05, 1.5) + slider('cf-b', 'β', -1, 1, 0.1, 0.3)
                 + slider('cf-t', 't', 0.25, 3, 0.25, 1) + play('cf'),
                 xh('Diferența de locație dintre S0 și S1' if ro else 'The location gap between S0 and S1') + tw('cf-tab') + charts('cf-chart')
                 + wt('cf', 'φ(t) în S1 și în S0', 'φ(t) in S1 and in S0')),
     'tail-table': (slider('tt-a', 'α al legii stabile' if ro else 'α of the stable law', 1.2, 1.95, 0.05, 1.7)
                    + slider('tt-k', 'Pragul k (rîndul evidențiat)' if ro else 'Threshold k (highlighted row)', 1, 20, 0.5, 10) + play('tt'),
                    xh('P(|X| > k): distribuția Normală N(0, 2) și legea stabilă S(α; 0; 1; 0)' if ro
                       else 'P(|X| > k): the Normal distribution N(0, 2) and the stable law S(α, 0, 1, 0)') + tw('tt-tab') + charts('tt-chart')),
     'cms-draw': (slider('cd-u', 'Extragerea uniformă U' if ro else 'Uniform draw U', 0.01, 0.99, 0.01, 0.8)
                  + slider('cd-w', 'Extragerea exponențială W' if ro else 'Exponential draw W', 0.05, 5, 0.05, 1.2)
                  + slider('cd-a', 'α', 1.2, 2, 0.1, 1.5) + slider('cd-b', 'β', -1, 1, 0.1, 0) + play('cd'),
                  charts('cd-chart') + wt('cd', 'o extragere CMS din (U, W)', 'one CMS draw from (U, W)')),
     'mcc-bet': (ser('mb-a', 'Seria exemplului lucrat', 'Series of the worked example', 'bet')
                 + slider('mb-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2000, 2020, 1, 2000) + play('mb'),
                 xh('Estimatorul McCulloch pe randamentele logaritmice zilnice (%)' if ro else 'The McCulloch estimator on daily log returns (%)') + tw('mb-tab')
                 + wt('mb', 'de la cuantile la α̂, β̂, γ̂ și δ̂<sub>0</sub>', 'from the quantiles to α̂, β̂, γ̂ and δ̂<sub>0</sub>')),
     'fits-table': (select('fs-m', 'Estimatorul' if ro else 'Estimator', [('ml', 'ML (verosimilitate maximă)' if ro else 'ML (maximum likelihood)'), ('mcc', 'McCulloch')], 'ml')
                    + slider('fs-c', 'Nivelul de încredere al intervalelor' if ro else 'Confidence level of the intervals', 0, 4, 1, 2)
                    + ser('fs-a', 'Seria exemplului lucrat', 'Series of the worked example', 'sp500') + play('fs'),
                    xh('Legi stabile estimate (S0) pe randamentele logaritmice zilnice' if ro else 'Stable laws (S0) fitted to daily log returns') + tw('fs-tab') + charts('fs-chart')
                    + wt('fs', 'locația din S0 în S1', 'the location from S0 to S1')),
     'loss-count': (ser('lc-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                    + slider('lc-x', 'Pragul pierderii x' if ro else 'Loss threshold x', 2, 25, 0.5, 10) + play('lc'),
                    xh('Zile cu pierderi peste x%: observate și așteptate (n × P(r &lt; −x)) conform fiecărui model estimat' if ro
                       else 'Days with losses above x%: observed and expected (n × P(r &lt; −x)) under each fitted model') + tw('lc-tab')
                    + wt('lc', 'coada de tip putere a legii stabile estimate', 'the power-law tail of the fitted stable law')),
     'var-table': (select('vt-p1', 'Nivelul p al primului grup de coloane' if ro else 'Level p of the first group of columns', lev, '0.01')
                   + select('vt-p2', 'Nivelul p al celui de-al doilea grup' if ro else 'Level p of the second group', lev, '0.001')
                   + ser('vt-a', 'Seria graficului și a exemplului lucrat', 'Series of the chart and the worked example', 'sp500') + play('vt'),
                   xh('VaR<sub>p</sub> = −q<sub>p</sub> din date și din cele trei modele estimate (% pe zi)' if ro
                      else 'VaR<sub>p</sub> = −q<sub>p</sub> from the data and from the three fitted models (% a day)') + tw('vt-tab') + charts('vt-chart')
                   + wt('vt', 'VaR din legea stabilă estimată și din distribuția Normală', 'VaR from the fitted stable law and from the Normal distribution')),
     'aic-table': (select('ai-c', 'Criteriul' if ro else 'Criterion', [('aic', 'AIC'), ('bic', 'BIC')], 'aic') + play('ai'),
                   xh('Criterii informaționale ale celor trei modele estimate' if ro else 'Information criteria of the three fitted models') + tw('ai-tab') + charts('ai-chart')),
    })
    return W


QA['ro'].update({
 'horizons': ('Pe slide, cu α = 1,5 scala sumei pe 20 de zile este 7,37γ, de 1,65 ori factorul √20 = 4,47. Butonul ▶ mută orizontul n pînă la 250 de zile. De cîte ori este scala pentru α = 1,5 mai mare decît cea dată de regula rădăcinii pătrate a timpului la n = 250?',
              'De 2,51 ori: 250<sup>1/1,5</sup> = 39,69, față de √250 = 15,81. Raportul este n<sup>1/α − 1/2</sup> = 250<sup>1/6</sup>, deci crește cu orizontul. Cu α̂ = 1,540 estimat pentru S&P 500, raportul este 1,56 la 20 de zile și 2,28 la 250 de zile. Pentru un portofoliu cu ponderi egale din N = 10 active, scala scade la 0,46γ pentru α = 1,5, față de 0,32γ pentru distribuția Normală.'),
 'cf-s0s1': ('Pe slide, α = 1,5 și β = 0,3 dau δ<sub>0</sub> − δ<sub>1</sub> = −0,3. Butonul ▶ crește α de la 1,1 la 1,9, cu β = 0,3. Pentru care valoare este diferența dintre cele două locații mai mare în valoare absolută: α = 1,1 sau α = 1,9?',
             'Pentru α = 1,1: β tan(πα/2) = 0,3 × (−6,314) = −1,894, față de 0,3 × (−0,158) = −0,048 pentru α = 1,9. Tangenta tinde la infinit cînd α se apropie de 1, de aceea parametrizarea S1 deplasează densitatea foarte mult în jurul lui α = 1, iar S0 nu. Modulul |φ(t)| = e<sup>−|t|<sup>α</sup></sup> este același în ambele parametrizări; diferă doar argumentul, printr-un termen liniar în t.'),
 'tail-table': ('Pe slide, pentru α = 1,7 și k = 10, P(|X| > k) este 0,56%. Cît dă legea de tip putere 2c<sub>α</sub>k<sup>−α</sup>, cu c<sub>1,7</sub> = 0,1313?',
                '0,52%, adică 94% din valoarea exactă. La k = 20 aproximarea dă 0,161% față de 0,166% exact (97%), iar la k = 2 doar 8,1% față de 18,6%: formula cozii este asimptotică și devine precisă departe în coadă. Raportul față de distribuția Normală N(0, 2) este 1,2 la k = 2 și 3,6 × 10<sup>9</sup> la k = 10.'),
 'cms-draw': ('Pe slide, U = 0,8 și W = 1,2 dau X = 1,554 pentru α = 1,5 și β = 0. Ce devine X dacă W crește la 2, cu același U?',
              '1,843. Pentru β = 0, W intră doar prin factorul (cos((1 − α)V)/W)<sup>(1 − α)/α</sup>; exponentul (1 − α)/α = −1/3 este negativ, deci X crește proporțional cu W<sup>1/3</sup>: 1,554 × (2/1,2)<sup>1/3</sup> = 1,843. Un W mai mic micșorează extragerea: W = 0,5 dă 1,161. Pentru α = 2 formula devine 2 sin V √W, o extragere din N(0, 2): 1,772 pentru U = 0,8 și W = 1,2.'),
 'mcc-bet': ('Pe slide, ν<sub>α</sub> = 3,425 al BET dă α̂ = 1,414. Butonul ▶ mută începutul eșantionului; din 2010, ν<sub>α</sub> al BET scade la 3,082. Ce α̂ dă Tabelul III?',
             'Circa 1,53: între ν<sub>α</sub> = 3,0 ↦ 1,563 și ν<sub>α</sub> = 3,2 ↦ 1,484, interpolarea liniară dă 1,531, iar cu |ν<sub>β</sub>| = 0,036 rezultă α̂ = 1,529. Un ν<sub>α</sub> mai mic înseamnă o bandă 5–95% mai îngustă față de intervalul intercuartilic, deci cozi mai subțiri și α̂ mai mare. Distribuția Normală are ν<sub>α</sub> = 2,439 și α = 2.'),
 'fits-table': ('La 95%, intervalul ML pentru α al S&P 500 este [1,499; 1,581], iar α̂ McCulloch este 1,438, sub interval. Pentru cîte dintre cele patru serii este α̂ McCulloch sub intervalul ML de 99%?',
                'Pentru toate patru. La 99% (±2,576 erori standard) intervalele sînt BET [1,435; 1,537], S&P 500 [1,486; 1,594], DAX [1,551; 1,661] și Bitcoin [1,343; 1,476], iar α̂ McCulloch este 1,414, 1,438, 1,455 și 1,322. Nici la 99,9% nu se schimbă concluzia (Bitcoin: limita inferioară 1,324). Cele două metode ponderează diferit cozile. Intervalele sînt asimptotice și nu includ incertitudinea estimației McCulloch.'),
 'loss-count': ('Legea stabilă estimată pe S&P 500 așteaptă 19,1 zile cu pierderi peste 10% în 6 714 zile. Cîte zile dă aproximarea cozii de tip putere c<sub>α</sub>(1 − β̂)z<sup>−α̂</sup>, cu z = (10 + δ̂<sub>0</sub>)/γ̂ = 17,33?',
                '18,3 zile, 96% din valoarea exactă: c<sub>1,540</sub> = 0,1868, P ≈ 0,1868 × 1,181 × 17,33<sup>−1,540</sup> = 0,00273. Factorul 1 − β̂ = 1,181 vine din β̂ = −0,181: coada stîngă a legii estimate este mai groasă decît cea dreaptă. Observate: o singură zi. Pentru Bitcoin la 10%, aproximarea dă 68,2 zile, față de 73,9 exact și 56 observate.'),
 'var-table': ('La VaR 1% pentru S&P 500, legea stabilă dă 4,54%, de 1,62 ori VaR-ul distribuției Normale (2,80%). Butonul ▶ coboară nivelul celui de-al doilea grup de coloane. Cît devine raportul stabil/Normal la VaR 0,05%?',
               'Circa 7,9: 31,21% față de 3,97%. Raportul crește pe măsură ce p scade, deoarece cuantila legii stabile crește ca p<sup>−1/α</sup>, iar cea Normală doar ca √(2 ln(1/p)). Cuantila empirică la 0,05% este 9,31%, deci legea stabilă depășește datele de 3,4 ori. La 5% raportul stabil/Normal este 0,91: în corpul distribuției, legea stabilă dă un VaR mai mic.'),
 'aic-table': ('Pe slide, AIC preferă distribuția Student-t legii stabile cu 92 de puncte la S&P 500. Cît devine diferența dacă folosim BIC = k ln n − 2ℓ?',
               '98,4 puncte. Legea stabilă are un parametru în plus (4 față de 3), iar BIC îl penalizează cu ln 6 714 = 8,81 în loc de 2: 91,6 + 6,8 = 98,4. Concluzia nu se schimbă. Față de distribuția Normală, diferențele AIC ale distribuției Student-t sînt între 1 306 (DAX) și 2 411 (BET) puncte, cele ale legii stabile între 1 204 și 2 319, exact ca pe slide.'),
})
QA['en'].update({
 'horizons': ('On the slide, with α = 1.5 the scale of the 20-day sum is 7.37γ, 1.65 times the factor √20 = 4.47. The ▶ button moves the horizon n up to 250 days. How many times larger is the scale for α = 1.5 than the one given by the square-root-of-time rule at n = 250?',
              '2.51 times: 250<sup>1/1.5</sup> = 39.69, against √250 = 15.81. The ratio is n<sup>1/α − 1/2</sup> = 250<sup>1/6</sup>, so it grows with the horizon. With α̂ = 1.540 fitted to the S&P 500, the ratio is 1.56 at 20 days and 2.28 at 250 days. For an equally weighted portfolio of N = 10 assets, the scale falls to 0.46γ for α = 1.5, against 0.32γ for the Normal distribution.'),
 'cf-s0s1': ('On the slide, α = 1.5 and β = 0.3 give δ<sub>0</sub> − δ<sub>1</sub> = −0.3. The ▶ button raises α from 1.1 to 1.9, with β = 0.3. For which value is the gap between the two locations larger in absolute value: α = 1.1 or α = 1.9?',
             'For α = 1.1: β tan(πα/2) = 0.3 × (−6.314) = −1.894, against 0.3 × (−0.158) = −0.048 for α = 1.9. The tangent goes to infinity as α approaches 1, which is why the S1 parameterisation moves the density a lot around α = 1 and S0 does not. The modulus |φ(t)| = e<sup>−|t|<sup>α</sup></sup> is the same in both parameterisations; only the argument differs, by a term linear in t.'),
 'tail-table': ('On the slide, for α = 1.7 and k = 10, P(|X| > k) is 0.56%. What does the power law 2c<sub>α</sub>k<sup>−α</sup>, with c<sub>1.7</sub> = 0.1313, give?',
                '0.52%, that is 94% of the exact value. At k = 20 the approximation gives 0.161% against 0.166% exact (97%), and at k = 2 only 8.1% against 18.6%: the tail formula is asymptotic and becomes accurate far in the tail. The ratio to the Normal distribution N(0, 2) is 1.2 at k = 2 and 3.6 × 10<sup>9</sup> at k = 10.'),
 'cms-draw': ('On the slide, U = 0.8 and W = 1.2 give X = 1.554 for α = 1.5 and β = 0. What does X become if W rises to 2, with the same U?',
              '1.843. For β = 0, W enters only through the factor (cos((1 − α)V)/W)<sup>(1 − α)/α</sup>; the exponent (1 − α)/α = −1/3 is negative, so X grows in proportion to W<sup>1/3</sup>: 1.554 × (2/1.2)<sup>1/3</sup> = 1.843. A smaller W shrinks the draw: W = 0.5 gives 1.161. For α = 2 the formula becomes 2 sin V √W, a draw from N(0, 2): 1.772 for U = 0.8 and W = 1.2.'),
 'mcc-bet': ('On the slide, ν<sub>α</sub> = 3.425 of the BET gives α̂ = 1.414. The ▶ button moves the start of the sample; from 2010, ν<sub>α</sub> of the BET falls to 3.082. What α̂ does Table III give?',
             'About 1.53: between ν<sub>α</sub> = 3.0 ↦ 1.563 and ν<sub>α</sub> = 3.2 ↦ 1.484, linear interpolation gives 1.531, and with |ν<sub>β</sub>| = 0.036 the result is α̂ = 1.529. A smaller ν<sub>α</sub> means a narrower 5–95% band relative to the interquartile range, hence thinner tails and a larger α̂. The Normal distribution has ν<sub>α</sub> = 2.439 and α = 2.'),
 'fits-table': ('At 95%, the ML interval for α of the S&P 500 is [1.499, 1.581], and the McCulloch α̂ is 1.438, below the interval. For how many of the four series is the McCulloch α̂ below the 99% ML interval?',
                'For all four. At 99% (±2.576 standard errors) the intervals are BET [1.435, 1.537], S&P 500 [1.486, 1.594], DAX [1.551, 1.661] and Bitcoin [1.343, 1.476], and the McCulloch α̂ is 1.414, 1.438, 1.455 and 1.322. Not even at 99.9% does the conclusion change (Bitcoin: lower bound 1.324). The two methods weigh the tails differently. The intervals are asymptotic and do not include the uncertainty of the McCulloch estimate.'),
 'loss-count': ('The stable law fitted to the S&P 500 expects 19.1 days with losses above 10% in 6,714 days. How many days does the power-law tail approximation c<sub>α</sub>(1 − β̂)z<sup>−α̂</sup> give, with z = (10 + δ̂<sub>0</sub>)/γ̂ = 17.33?',
                '18.3 days, 96% of the exact value: c<sub>1.540</sub> = 0.1868, P ≈ 0.1868 × 1.181 × 17.33<sup>−1.540</sup> = 0.00273. The factor 1 − β̂ = 1.181 comes from β̂ = −0.181: the left tail of the fitted law is heavier than the right one. Observed: a single day. For Bitcoin at 10%, the approximation gives 68.2 days, against 73.9 exact and 56 observed.'),
 'var-table': ('At VaR 1% for the S&P 500, the stable law gives 4.54%, 1.62 times the VaR of the Normal distribution (2.80%). The ▶ button lowers the level of the second group of columns. What does the stable/Normal ratio become at VaR 0.05%?',
               'About 7.9: 31.21% against 3.97%. The ratio grows as p falls, because the quantile of the stable law grows like p<sup>−1/α</sup> and the Normal one only like √(2 ln(1/p)). The empirical quantile at 0.05% is 9.31%, so the stable law exceeds the data 3.4 times. At 5% the stable/Normal ratio is 0.91: in the body of the distribution, the stable law gives a smaller VaR.'),
 'aic-table': ('On the slide, AIC prefers the Student-t to the stable law by 92 points for the S&P 500. What does the gap become with BIC = k ln n − 2ℓ?',
               '98.4 points. The stable law has one more parameter (4 against 3), and BIC penalises it by ln 6,714 = 8.81 instead of 2: 91.6 + 6.8 = 98.4. The conclusion does not change. Against the Normal distribution, the AIC gains of the Student-t lie between 1,306 (DAX) and 2,411 (BET) points, those of the stable law between 1,204 and 2,319, as on the slide.'),
})

PREDICT.update({
 'horizons': (1, [('Tot 1,65: raportul nu depinde de orizont.', 'Still 1.65: the ratio does not depend on the horizon.'),
                  ('2,51: raportul n^(1/α − 1/2) crește cu n.', '2.51: the ratio n^(1/α − 1/2) grows with n.'),
                  ('15,8: atît cît √250.', '15.8: as much as √250.')]),
 'cf-s0s1': (0, [('α = 1,1: −1,89, față de −0,05 la α = 1,9.', 'α = 1.1: −1.89, against −0.05 at α = 1.9.'),
                 ('α = 1,9: mai aproape de distribuția Normală, deplasarea este mai mare.', 'α = 1.9: closer to the Normal distribution, the shift is larger.'),
                 ('Aceeași: diferența depinde doar de β.', 'The same: the gap depends only on β.')]),
 'tail-table': (0, [('Circa 0,52%, aproape de valoarea exactă.', 'About 0.52%, close to the exact value.'),
                    ('Circa 0,26%: cα·k^(−α), cu o singură coadă.', 'About 0.26%: cα·k^(−α), with one tail only.'),
                    ('Circa 2,0%: 10^(−1,7), fără constanta cα.', 'About 2.0%: 10^(−1.7), without the constant cα.')]),
 'cms-draw': (2, [('Rămîne 1,554: pentru β = 0, W nu influențează extragerea.', 'It stays at 1.554: for β = 0, W does not affect the draw.'),
                  ('Scade la circa 1,31: un W mai mare micșorează X.', 'It falls to about 1.31: a larger W shrinks X.'),
                  ('Crește la circa 1,84: X este proporțional cu W^(1/3).', 'It rises to about 1.84: X is proportional to W^(1/3).')]),
 'mcc-bet': (0, [('Circa 1,53: να mai mic, α̂ mai mare.', 'About 1.53: smaller να, larger α̂.'),
                 ('Circa 1,30: un να mai mic înseamnă cozi mai groase.', 'About 1.30: a smaller να means heavier tails.'),
                 ('Tot 1,414: α̂ nu depinde de eșantion.', 'Still 1.414: α̂ does not depend on the sample.')]),
 'fits-table': (0, [('Pentru toate patru.', 'For all four.'),
                    ('Pentru niciuna: două estimații ale aceluiași parametru cad în intervalul de 99%.', 'For none: two estimates of the same parameter fall within the 99% interval.'),
                    ('Doar pentru Bitcoin, care are cel mai mic α̂.', 'Only for Bitcoin, which has the smallest α̂.')]),
 'loss-count': (0, [('Circa 18,3, aproape de 19,1.', 'About 18.3, close to 19.1.'),
                    ('Circa 12,7: coada stîngă cu factorul 1 + β̂.', 'About 12.7: the left tail with the factor 1 + β̂.'),
                    ('Circa 43: pragul x = 10 folosit direct, fără δ̂₀ și γ̂.', 'About 43: the threshold x = 10 used directly, without δ̂₀ and γ̂.')]),
 'var-table': (2, [('Tot 1,62: legile diferă doar prin scală.', 'Still 1.62: the laws differ only in scale.'),
                   ('Circa 5,3, ca la 0,1%: raportul se stabilizează în coadă.', 'About 5.3, as at 0.1%: the ratio settles in the tail.'),
                   ('Circa 7,9: raportul crește cînd p scade.', 'About 7.9: the ratio grows as p falls.')]),
 'aic-table': (1, [('Tot 91,6: BIC și AIC penalizează la fel.', 'Still 91.6: BIC and AIC penalise alike.'),
                   ('98,4: BIC penalizează parametrul în plus cu ln n.', '98.4: BIC penalises the extra parameter by ln n.'),
                   ('84,8: BIC penalizează mai puțin parametrul în plus.', '84.8: BIC penalises the extra parameter less.')]),
})

VEIL = {'horizons': {'out': True, 'charts': ['hz-chart']}, 'cf-s0s1': {'out': True, 'charts': ['cf-tab']},
        'tail-table': {'out': True, 'charts': ['tt-tab']}, 'cms-draw': {'out': True, 'charts': ['cd-chart']},
        'fits-table': {'out': True, 'charts': ['fs-tab', 'fs-chart']}, 'loss-count': {'out': True, 'charts': ['lc-we-box']},
        'var-table': {'out': True, 'charts': ['vt-chart']}}

TEXT_JS['ro'] += r"""window.ch3De = n => (n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? ' de' : '');
Object.assign(window.CH3_TEXT, {
 h: { nrm: 'Normală', normal: 'distribuția Normală', ratio: 'raportul', ttN: 'P(|X| > k), Normală (%)', ttS: a => `P(|X| > k), α = ${a} (%)`, ttPL: '2c<sub>α</sub>k<sup>−α</sup> (%)',
   series: 'Seria', aMcc: 'α̂ McC.', aML: 'α̂ ML (SE)', bML: 'β̂ ML (SE)', ci: l => `intervalul de ${l} pentru α`, aMLs: 'α̂ ML', aMccs: 'α̂ McC.', bMcc: 'β̂ McC.',
   dMcc: 'α̂ McC. − α̂ ML', lossAbove: 'pierdere peste', observed: 'observate', stable: 'stabilă', stObs: 'stabilă / observate', data: 'date',
   estimator: 'estimatorul', mean: 'media', bias: 'deplasarea (bias)', sd: 'abaterea standard', sdRatio: 'raportul abaterilor standard' },
 hz: (n, f15, fs, N, d15, dn) => `n = ${n}${window.ch3De(n)} zile: scala sumei este de ${f15} ori scala zilnică pentru α = 1,5, față de √n = ${fs} pentru distribuția Normală. Portofoliu din N = ${N} active: ${d15} din scală pentru α = 1,5, ${dn} pentru α = 2.`,
 hzw1: (a, al, g) => `${a}, legea stabilă estimată prin ML (S0): α̂ = ${al}, γ̂ = ${g}% pe zi.`,
 hzw2: (n, al, inv, f, g, gn) => `Suma a n = ${n} randamente zilnice i.i.d.: n<sup>1/α̂</sup> = ${n}<sup>1/${al}</sup> = ${n}<sup>${inv}</sup> = ${f}; scala γ<sub>n</sub> = ${f} × ${g} = ${gn}% pe ${n}${window.ch3De(n)} zile.`,
 hzw3: (n, g, sq, r, e) => `Regula rădăcinii pătrate a timpului dă √${n} × ${g} = ${sq}%; scala stabilă este de n<sup>1/α̂ − 1/2</sup> = ${n}<sup>${e}</sup> = ${r} ori mai mare.`,
 hzw4: (N, e, f, g, gN, gn) => `Portofoliu cu ponderi egale din N = ${N} copii i.i.d.: N<sup>1/α̂ − 1</sup> = ${N}<sup>${e}</sup> = ${f}, scala ${f} × ${g} = ${gN}%, față de γ̂/√N = ${gn}% dacă α ar fi 2.`,
 cf: (a, b, s) => `α = ${a}, β = ${b}, γ = 1: δ<sub>0</sub> − δ<sub>1</sub> = β tan(πα/2) = ${s}. Legea S1 cu δ<sub>1</sub> = 0 este legea S0 cu δ<sub>0</sub> = ${s}.`,
 cfw1: (a, b, t, h, ta, A) => `α = ${a}, β = ${b}, γ = 1, δ = 0, t = ${t}: tan(πα/2) = tan(${h}π) = ${ta}; |t|<sup>α</sup> = ${A}.`,
 cfw2: (z, re, im, phi, m) => `S1: ln φ(t) = ${z}; φ(t) = e<sup>${re}</sup>(cos(${im}) + i sin(${im})) = ${phi}, |φ(t)| = ${m}.`,
 cfw3: (t, f, z, phi) => `S0: factorul |t|<sup>1−α</sup> − 1 este ${f} în t = ${t}; ln φ(t) = ${z}, φ(t) = ${phi}: același modul.`,
 cfw4: (d, s) => `Argumentele diferă prin ${d} = −β tan(πα/2) t, adică printr-o deplasare a locației. δ<sub>0</sub> = δ<sub>1</sub> + βγ tan(πα/2): legea S1 cu δ<sub>1</sub> = 0 este legea S0 cu δ<sub>0</sub> = ${s}.`,
 tt: (a, k, pn, ps, r, c, pl, q) => `α = ${a}, k = ${k}: P(|X| > k) este ${pn} pentru distribuția Normală N(0, 2) și ${ps} pentru legea stabilă, de ${r} ori mai mare. Legea de tip putere 2c<sub>α</sub>k<sup>−α</sup>, cu c<sub>α</sub> = ${c}, dă ${pl}, adică ${q} din valoarea exactă.`,
 cd: (x, a, b, F) => `X = ${x}; conform S1(${a}; ${b}; 1; 0), ${F}% din distribuție se află sub această valoare.`,
 cdw1: (u, v) => `V = π(U − 1/2) = π(${u} − 0,5) = ${v}: un unghi uniform pe (−π/2, π/2).`,
 cdwB: (a, b, ta, B, S) => `β ≠ 0: tan(πα/2) = ${ta}; B = arctan(β tan(πα/2))/α = ${B}; S = (1 + β² tan²(πα/2))<sup>1/(2α)</sup> = ${S}.`,
 cdw2: (g, s1, c1) => g ? `sin(α(V + B)) = ${s1}; (cos V)<sup>1/α</sup> = ${c1}.` : `sin(αV) = ${s1}; (cos V)<sup>1/α</sup> = ${c1}.`,
 cdw3: (g, w, c2) => g ? `(cos(V − α(V + B))/W)<sup>(1 − α)/α</sup> = ${c2}, cu W = ${w}.` : `(cos((1 − α)V)/W)<sup>(1 − α)/α</sup> = ${c2}, cu W = ${w}.`,
 cdw4: (S, s1, c1, c2, x) => `X = ${S}${s1}/${c1} × ${c2} = ${x}.`,
 mb: (y, list) => `α̂ McCulloch, eșantioane din ${y}: ${list}.`,
 mbNote: 'Randamente logaritmice zilnice (%) din anul ales (Bitcoin cel mai devreme din septembrie 2014) pînă la 18 septembrie 2026; γ̂ și δ̂<sub>0</sub> în % pe zi.',
 mbw1: (a, y, n, q05, q25, q50, q75, q95) => `${a}, randamente logaritmice zilnice (%) din ${y} (n = ${n}): q̂<sub>0,05</sub> = ${q05}, q̂<sub>0,25</sub> = ${q25}, q̂<sub>0,5</sub> = ${q50}, q̂<sub>0,75</sub> = ${q75}, q̂<sub>0,95</sub> = ${q95}.`,
 mbw2: (q95, q05, w, q75, q25, iqr, nua, num, nub) => `ν<sub>α</sub> = (${q95} − ${q05})/(${q75} − ${q25}) = ${w}/${iqr} = ${nua}; ν<sub>β</sub> = (q̂<sub>0,95</sub> + q̂<sub>0,05</sub> − 2q̂<sub>0,5</sub>)/${w} = ${num}/${w} = ${nub}.`,
 mbw3: (n0, a0, n1, a1, nua, ai, anb, al) => `Tabelul III, coloana ν<sub>β</sub> = 0: ν<sub>α</sub> = ${n0} ↦ ${a0}, ν<sub>α</sub> = ${n1} ↦ ${a1}; interpolarea liniară în ν<sub>α</sub> = ${nua}: α̂ ≈ ${ai}. Cu interpolare și în |ν<sub>β</sub>| = ${anb}: α̂ = ${al}.`,
 mbw4: (iqr, phi, g, b, d) => `γ̂ = (q̂<sub>0,75</sub> − q̂<sub>0,25</sub>)/φ<sub>3</sub>(α̂, β̂) = ${iqr}/${phi} = ${g}% pe zi; din Tabelele IV și VII: β̂ = ${b}, δ̂<sub>0</sub> = ${d}.`,
 fs: (lv, list, below) => `ML, intervale de ${lv} pentru α: ${list}. α̂ McCulloch sub intervalul ML: ${below}.`,
 fsNote: 'Randamente logaritmice zilnice în %; ML în S0, cu erori standard (SE); γ̂, δ̂<sub>0</sub>, δ̂<sub>1</sub> în % pe zi; McC.: McCulloch. Intervalele α̂ ± z·SE sînt asimptotice.',
 fsNoteM: 'Estimări McCulloch (fără erori standard); γ̂, δ̂<sub>0</sub>, δ̂<sub>1</sub> în % pe zi; δ̂<sub>1</sub> = δ̂<sub>0</sub> − β̂γ̂ tan(πα̂/2).',
 fsw1: (a, m, al, b, g, d) => `${a}, estimări ${m} în S0: α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d} (% pe zi).`,
 fsw2: (h, ta, b, g, tb, s) => `tan(πα̂/2) = tan(${h}π) = ${ta}; β̂γ̂ tan(πα̂/2) = ${b} × ${g} × ${tb} = ${s}.`,
 fsw3: (d0, s, d1, d1t) => `δ̂<sub>1</sub> = δ̂<sub>0</sub> − β̂γ̂ tan(πα̂/2) = ${d0} − ${s} = ${d1} (în tabel: ${d1t}).`,
 fsw4: s => `Cele două locații descriu aceeași lege estimată și diferă cu ${s} puncte procentuale pe zi. levy_stable din scipy folosește implicit S1, deci raportează δ̂<sub>1</sub>.`,
 lc: (a, x, o, n, t, s) => `${a}, pierderi peste ${x}%: zile observate ${o}; așteptate: distribuția Normală ${n}, Student-t ${t}, legea stabilă ${s}.`,
 lcw1: (a, n, al, b, g, d) => `${a}, n = ${n} zile; legea stabilă estimată prin ML (S0): α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d}.`,
 lcw2: (x, d, g, z) => `r &lt; −${x}% înseamnă Z = (r − δ̂<sub>0</sub>)/γ̂ &lt; −(${x} + ${d})/${g} = −${z}.`,
 lcw3: (al, c, b, z, p, n, e) => `c<sub>α</sub> = Γ(${al}) sin(π × ${al}/2)/π = ${c}; P ≈ ${c} × (1 − ${b}) × ${z}<sup>−${al}</sup> = ${p}; n × P = ${n} × ${p} = ${e} zile.`,
 lcw4: (pe, e, q, o) => `Valoarea exactă a legii estimate: P(Z &lt; −z) = ${pe}, adică ${e} zile; legea de tip putere dă ${q} din ea. Zile observate: ${o}.`,
 vt: (p, list) => `VaR ${p}: VaR-ul legii stabile împărțit la VaR-ul empiric: ${list}.`,
 vtw1: (a, mu, s, al, b, g, d) => `${a}: distribuția Normală μ̂ = ${mu}, σ̂ = ${s}; legea stabilă (ML, S0) α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d} (% pe zi).`,
 vtw2: (p, z, mu, s, zp, vn, zs, d, g, zsp, vs, r, ve) => `VaR ${p}: z<sub>p</sub> = Φ<sup>−1</sup>(p) = ${z}, VaR-ul distribuției Normale = −(${mu} + ${s} × ${zp}) = ${vn}%; cuantila de ordin p a legii S(α̂; β̂; 1; 0 | 0) este ${zs}, VaR-ul legii stabile = −(${d} + ${g} × ${zsp}) = ${vs}%, de ${r} ori VaR-ul distribuției Normale. Datele: ${ve}%.`,
 ai: (nm, l1, l2) => `${nm} al distribuției Normale minus ${nm} al distribuției Student-t: ${l1}. ${nm} al legii stabile minus ${nm} al distribuției Student-t: ${l2}.`,
 aiNote: nm => nm === 'AIC' ? 'AIC = 2k − 2ℓ(θ̂), cu k = 2 (Normală), 3 (Student-t), 4 (stabilă); se preferă valoarea mai mică.' : 'BIC (Bayesian information criterion, criteriul informațional bayesian) = k ln n − 2ℓ(θ̂), cu k = 2 (Normală), 3 (Student-t), 4 (stabilă); se preferă valoarea mai mică.',
 mcNote: R => `Primele ${R} din cele 200 de eșantioane pentru fiecare n; α = 1,7. RMSE (root mean squared error): rădăcina erorii pătratice medii.`
});
"""
TEXT_JS['en'] += r"""Object.assign(window.CH3_TEXT, {
 h: { nrm: 'Normal', normal: 'Normal distribution', ratio: 'ratio', ttN: 'P(|X| > k), Normal (%)', ttS: a => `P(|X| > k), α = ${a} (%)`, ttPL: '2c<sub>α</sub>k<sup>−α</sup> (%)',
   series: 'Series', aMcc: 'α̂ McC.', aML: 'α̂ ML (SE)', bML: 'β̂ ML (SE)', ci: l => `${l} interval for α`, aMLs: 'α̂ ML', aMccs: 'α̂ McC.', bMcc: 'β̂ McC.',
   dMcc: 'α̂ McC. − α̂ ML', lossAbove: 'loss above', observed: 'observed', stable: 'stable', stObs: 'stable / observed', data: 'data',
   estimator: 'estimator', mean: 'mean', bias: 'bias', sd: 'standard deviation', sdRatio: 'ratio of the standard deviations' },
 hz: (n, f15, fs, N, d15, dn) => `n = ${n} days: the scale of the sum is ${f15} times the daily scale for α = 1.5, against √n = ${fs} for the Normal distribution. Portfolio of N = ${N} assets: ${d15} of the scale for α = 1.5, ${dn} for α = 2.`,
 hzw1: (a, al, g) => `${a}, stable law fitted by ML (S0): α̂ = ${al}, γ̂ = ${g}% a day.`,
 hzw2: (n, al, inv, f, g, gn) => `Sum of n = ${n} i.i.d. daily returns: n<sup>1/α̂</sup> = ${n}<sup>1/${al}</sup> = ${n}<sup>${inv}</sup> = ${f}; scale γ<sub>n</sub> = ${f} × ${g} = ${gn}% over ${n} days.`,
 hzw3: (n, g, sq, r, e) => `The square-root-of-time rule gives √${n} × ${g} = ${sq}%; the stable scale is n<sup>1/α̂ − 1/2</sup> = ${n}<sup>${e}</sup> = ${r} times larger.`,
 hzw4: (N, e, f, g, gN, gn) => `Equally weighted portfolio of N = ${N} i.i.d. copies: N<sup>1/α̂ − 1</sup> = ${N}<sup>${e}</sup> = ${f}, scale ${f} × ${g} = ${gN}%, against γ̂/√N = ${gn}% if α were 2.`,
 cf: (a, b, s) => `α = ${a}, β = ${b}, γ = 1: δ<sub>0</sub> − δ<sub>1</sub> = β tan(πα/2) = ${s}. The S1 law with δ<sub>1</sub> = 0 is the S0 law with δ<sub>0</sub> = ${s}.`,
 cfw1: (a, b, t, h, ta, A) => `α = ${a}, β = ${b}, γ = 1, δ = 0, t = ${t}: tan(πα/2) = tan(${h}π) = ${ta}; |t|<sup>α</sup> = ${A}.`,
 cfw2: (z, re, im, phi, m) => `S1: ln φ(t) = ${z}; φ(t) = e<sup>${re}</sup>(cos(${im}) + i sin(${im})) = ${phi}, |φ(t)| = ${m}.`,
 cfw3: (t, f, z, phi) => `S0: the factor |t|<sup>1−α</sup> − 1 is ${f} at t = ${t}; ln φ(t) = ${z}, φ(t) = ${phi}: the same modulus.`,
 cfw4: (d, s) => `The arguments differ by ${d} = −β tan(πα/2) t, a shift of location. δ<sub>0</sub> = δ<sub>1</sub> + βγ tan(πα/2): the S1 law with δ<sub>1</sub> = 0 is the S0 law with δ<sub>0</sub> = ${s}.`,
 tt: (a, k, pn, ps, r, c, pl, q) => `α = ${a}, k = ${k}: P(|X| > k) is ${pn} for the Normal distribution N(0, 2) and ${ps} for the stable law, ${r} times more. The power law 2c<sub>α</sub>k<sup>−α</sup>, with c<sub>α</sub> = ${c}, gives ${pl}, that is ${q} of the exact value.`,
 cd: (x, a, b, F) => `X = ${x}; under S1(${a}, ${b}, 1, 0), ${F}% of the distribution lies below this value.`,
 cdw1: (u, v) => `V = π(U − 1/2) = π(${u} − 0.5) = ${v}: a uniform angle on (−π/2, π/2).`,
 cdwB: (a, b, ta, B, S) => `β ≠ 0: tan(πα/2) = ${ta}; B = arctan(β tan(πα/2))/α = ${B}; S = (1 + β² tan²(πα/2))<sup>1/(2α)</sup> = ${S}.`,
 cdw2: (g, s1, c1) => g ? `sin(α(V + B)) = ${s1}; (cos V)<sup>1/α</sup> = ${c1}.` : `sin(αV) = ${s1}; (cos V)<sup>1/α</sup> = ${c1}.`,
 cdw3: (g, w, c2) => g ? `(cos(V − α(V + B))/W)<sup>(1 − α)/α</sup> = ${c2}, with W = ${w}.` : `(cos((1 − α)V)/W)<sup>(1 − α)/α</sup> = ${c2}, with W = ${w}.`,
 cdw4: (S, s1, c1, c2, x) => `X = ${S}${s1}/${c1} × ${c2} = ${x}.`,
 mb: (y, list) => `McCulloch α̂, samples from ${y}: ${list}.`,
 mbNote: 'Daily log returns (%) from the chosen year (Bitcoin from September 2014 at the earliest) to 18 September 2026; γ̂ and δ̂<sub>0</sub> in % a day.',
 mbw1: (a, y, n, q05, q25, q50, q75, q95) => `${a}, daily log returns (%) from ${y} (n = ${n}): q̂<sub>0.05</sub> = ${q05}, q̂<sub>0.25</sub> = ${q25}, q̂<sub>0.5</sub> = ${q50}, q̂<sub>0.75</sub> = ${q75}, q̂<sub>0.95</sub> = ${q95}.`,
 mbw2: (q95, q05, w, q75, q25, iqr, nua, num, nub) => `ν<sub>α</sub> = (${q95} − ${q05})/(${q75} − ${q25}) = ${w}/${iqr} = ${nua}; ν<sub>β</sub> = (q̂<sub>0.95</sub> + q̂<sub>0.05</sub> − 2q̂<sub>0.5</sub>)/${w} = ${num}/${w} = ${nub}.`,
 mbw3: (n0, a0, n1, a1, nua, ai, anb, al) => `Table III, column ν<sub>β</sub> = 0: ν<sub>α</sub> = ${n0} ↦ ${a0}, ν<sub>α</sub> = ${n1} ↦ ${a1}; linear interpolation at ν<sub>α</sub> = ${nua}: α̂ ≈ ${ai}. Interpolating also in |ν<sub>β</sub>| = ${anb}: α̂ = ${al}.`,
 mbw4: (iqr, phi, g, b, d) => `γ̂ = (q̂<sub>0.75</sub> − q̂<sub>0.25</sub>)/φ<sub>3</sub>(α̂, β̂) = ${iqr}/${phi} = ${g}% a day; from Tables IV and VII: β̂ = ${b}, δ̂<sub>0</sub> = ${d}.`,
 fs: (lv, list, below) => `ML, ${lv} intervals for α: ${list}. McCulloch α̂ below the ML interval: ${below}.`,
 fsNote: 'Daily log returns in %; ML in S0, with standard errors (SE); γ̂, δ̂<sub>0</sub>, δ̂<sub>1</sub> in % a day; McC.: McCulloch. The intervals α̂ ± z·SE are asymptotic.',
 fsNoteM: 'McCulloch estimates (no standard errors); γ̂, δ̂<sub>0</sub>, δ̂<sub>1</sub> in % a day; δ̂<sub>1</sub> = δ̂<sub>0</sub> − β̂γ̂ tan(πα̂/2).',
 fsw1: (a, m, al, b, g, d) => `${a}, ${m} estimates in S0: α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d} (% a day).`,
 fsw2: (h, ta, b, g, tb, s) => `tan(πα̂/2) = tan(${h}π) = ${ta}; β̂γ̂ tan(πα̂/2) = ${b} × ${g} × ${tb} = ${s}.`,
 fsw3: (d0, s, d1, d1t) => `δ̂<sub>1</sub> = δ̂<sub>0</sub> − β̂γ̂ tan(πα̂/2) = ${d0} − ${s} = ${d1} (table: ${d1t}).`,
 fsw4: s => `The two locations describe the same fitted law and differ by ${s} percentage points a day. scipy's levy_stable uses S1 by default, so it reports δ̂<sub>1</sub>.`,
 lc: (a, x, o, n, t, s) => `${a}, losses above ${x}%: observed days ${o}; expected: Normal distribution ${n}, Student-t ${t}, stable law ${s}.`,
 lcw1: (a, n, al, b, g, d) => `${a}, n = ${n} days; stable law fitted by ML (S0): α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d}.`,
 lcw2: (x, d, g, z) => `r &lt; −${x}% means Z = (r − δ̂<sub>0</sub>)/γ̂ &lt; −(${x} + ${d})/${g} = −${z}.`,
 lcw3: (al, c, b, z, p, n, e) => `c<sub>α</sub> = Γ(${al}) sin(π × ${al}/2)/π = ${c}; P ≈ ${c} × (1 − ${b}) × ${z}<sup>−${al}</sup> = ${p}; n × P = ${n} × ${p} = ${e} days.`,
 lcw4: (pe, e, q, o) => `The exact value of the fitted law: P(Z &lt; −z) = ${pe}, that is ${e} days; the power law gives ${q} of it. Observed days: ${o}.`,
 vt: (p, list) => `VaR ${p}: VaR of the stable law divided by the empirical VaR: ${list}.`,
 vtw1: (a, mu, s, al, b, g, d) => `${a}: Normal distribution μ̂ = ${mu}, σ̂ = ${s}; stable law (ML, S0) α̂ = ${al}, β̂ = ${b}, γ̂ = ${g}, δ̂<sub>0</sub> = ${d} (% a day).`,
 vtw2: (p, z, mu, s, zp, vn, zs, d, g, zsp, vs, r, ve) => `VaR ${p}: z<sub>p</sub> = Φ<sup>−1</sup>(p) = ${z}, Normal VaR = −(${mu} + ${s} × ${zp}) = ${vn}%; the p-quantile of S(α̂, β̂, 1, 0; 0) is ${zs}, stable VaR = −(${d} + ${g} × ${zsp}) = ${vs}%, ${r} times the Normal VaR. Data: ${ve}%.`,
 ai: (nm, l1, l2) => `${nm} of the Normal distribution minus ${nm} of the Student-t: ${l1}. ${nm} of the stable law minus ${nm} of the Student-t: ${l2}.`,
 aiNote: nm => nm === 'AIC' ? 'AIC = 2k − 2ℓ(θ̂), with k = 2 (Normal), 3 (Student-t), 4 (stable); lower is better.' : 'BIC (Bayesian information criterion) = k ln n − 2ℓ(θ̂), with k = 2 (Normal), 3 (Student-t), 4 (stable); lower is better.',
 mcNote: R => `The first ${R} of the 200 samples for each n; α = 1.7. RMSE: root mean squared error.`
});
"""


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Autoevaluare' if lang == 'ro' else 'Self-Assessment', id='check', store=f'sfm-ch3-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH3_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH3_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH3_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    import re
    fix = lambda t: re.sub(r'(?<=\d) (?=\d{3}(?!\d))', ' ', t)
    for a, (q, ans) in QA['ro'].items():
        QA['ro'][a] = (fix(q), fix(ans))
    for a, (k, opts) in PREDICT.items():
        PREDICT[a] = (k, [(fix(r), e) for r, e in opts])
    for t in TICKET:
        t['q'] = (fix(t['q'][0]), t['q'][1])
        t['opts'] = [((fix(o[0]), o[1]), (fix(w[0]), w[1])) for o, w in t['opts']]


_ro_numbers()
_predict_js()


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    from interactive_common import frames_of, latex_text, ROOT
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    res = {}
    for a, fr, _, _, ken, kro in XBLOCKS:
        i = fr - 1
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved to their frames by ch3.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch3_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    assert set(blocks) <= set(ORDER) and len(blocks) == len(BLOCKS), set(blocks) ^ {a for a, _, _ in BLOCKS}
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch3_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


if __name__ == '__main__':
    build()
