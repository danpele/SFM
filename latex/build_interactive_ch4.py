"""Build the interactive pages of Chapter 4: interactive/ch4.html (EN) and interactive/ch4_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch4.js, inputs in interactive/ch4_data.js
(Quantlets/Ch_04/export_interactive.py), slide images in interactive/slides/.
"""
import json

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 4
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_04'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('random-variables', 'sfm_ch4_random_variables', 'rv'), ('cdf-quantile', 'sfm_ch4_cdf_quantile', 'rv'),
          ('joint', 'sfm_ch4_joint', 'joint'), ('uncorrelated-dependent', 'sfm_ch4_uncorrelated_dependent', 'joint'),
          ('conditional', 'sfm_ch4_conditional', 'cond'), ('lln', 'sfm_ch4_lln', 'lln'),
          ('randu', 'sfm_ch4_randu', 'mc'), ('inverse-transform', 'sfm_ch4_inverse_transform', 'mc'), ('monte-carlo', 'sfm_ch4_monte_carlo', 'mc'),
          ('binomial', 'sfm_ch4_binomial', 'bin'), ('processes', 'sfm_ch4_processes', 'proc'), ('spot-the-real', 'sfm_ch4_spot_the_real', 'proc'),
          ('wiener', 'sfm_ch4_wiener', 'wien'), ('gbm-paths', 'sfm_ch4_gbm_paths', 'wien'), ('gbm-fan', 'sfm_ch4_gbm_fan', 'wien'),
          ('gbm-check', 'sfm_ch4_gbm_check', 'wien')]

PFX = {'random-variables': 'rv', 'cdf-quantile': 'cq', 'joint': 'jt', 'uncorrelated-dependent': 'ud', 'conditional': 'cd', 'lln': 'ln',
       'randu': 'ru', 'inverse-transform': 'it', 'monte-carlo': 'mc', 'binomial': 'bn', 'processes': 'pr', 'spot-the-real': 'sr',
       'wiener': 'wn', 'gbm-paths': 'gp', 'gbm-fan': 'gf', 'gbm-check': 'gc'}

SECTIONS = {
 'ro': dict(rv='Variabile aleatoare și distribuții', joint='Distribuții comune și dependență', cond='Media condiționată și varianța condiționată',
            lln='Legea numerelor mari și CLT', mc='Numere aleatoare și Monte Carlo', bin='Modelul binomial',
            proc='Procese stochastice în timp discret', wien='Procesul Wiener și mișcarea browniană geometrică'),
 'en': dict(rv='Random Variables and Distributions', joint='Joint Distributions and Dependence', cond='Conditional Expectation and Conditional Variance',
            lln='The Law of Large Numbers and the CLT', mc='Random Numbers and Monte Carlo', bin='The Binomial Model',
            proc='Discrete-Time Stochastic Processes', wien='The Wiener Process and Geometric Brownian Motion'),
}

L = {
 'ro': dict(file='ch4_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 4 · Probabilitate', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Randamente logaritmice zilnice, în %: S&P 500, DAX și BET din ianuarie 2000, Bitcoin din septembrie 2014 (7 zile pe săptămînă), '
       'pînă la 18 septembrie 2026. Prețuri de închidere EODHD. Simulările sînt cele din curs: aceleași modele, aceleași numere de traiectorii '
       'și aceleași semințe ale generatorului.'),
 'en': dict(file='ch4.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 4 · Probability', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Daily log returns, in %: S&P 500, DAX and BET from January 2000, Bitcoin from September 2014 (7 days a week), '
       'to 18 September 2026. EODHD closing prices. The simulations are those of the lecture: the same models, the same numbers of paths '
       'and the same generator seeds.'),
}


def widgets_old(lang):
    ro = lang == 'ro'
    A4 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
    bands = [(b, f'{p}%') for b, p in (('0.5', 50), ('0.8', 80), ('0.9', 90), ('0.95', 95), ('0.99', 99))]
    return {
     'random-variables': (select('rv-a', 'Seria' if ro else 'Series', A4[:3], 'sp500') + play('rv'), charts('rv-chart')),
     'cdf-quantile': (select('cq-a', 'Seria' if ro else 'Series', A4, 'sp500')
                      + slider('cq-p', 'Nivelul α al cuantilei (VaR α = −q<sub>α</sub>)' if ro else 'Level α of the quantile (VaR α = −q<sub>α</sub>)', 0, 5, 1, 3)
                      + play('cq'), charts('cq-chart')),
     'joint': (slider('jt-y', 'Zilele evidențiate' if ro else 'Highlighted days', 0, 27, 1, 0) + play('jt'), charts('jt-chart')),
     'uncorrelated-dependent': (slider('ud-h', 'Lagul h (zile)' if ro else 'Lag h (days)', 1, 20, 1, 1) + play('ud'), charts('ud-chart')),
     'conditional': (slider('cd-q', 'Numărul de grupuri după |r<sub>t−1</sub>|' if ro else 'Number of groups by |r<sub>t−1</sub>|', 2, 10, 1, 5) + play('cd'), charts('cd-chart')),
     'lln': (slider('ln-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2000, 2020, 1, 2000) + play('ln'), charts('ln-chart')),
     'randu': (slider('ru-s', 'Sămînța x<sub>0</sub> a RANDU (impară)' if ro else 'RANDU seed x<sub>0</sub> (odd)', 1, 99, 2, 1) + play('ru'), charts('ru-chart')),
     'inverse-transform': (slider('it-l', 'λ al distribuției exponențiale (stînga)' if ro else 'λ of the exponential distribution (left)', 0.4, 4, 0.2, 2)
                           + slider('it-x', 'Pragul x pentru extrageri sub −x% (dreapta; ▶ mișcă și λ = 0,4x)' if ro else 'Threshold x for draws below −x% (right; ▶ also moves λ = 0.4x)', 1, 12, 0.5, 5)
                           + play('it'), charts('it-chart')),
     'monte-carlo': (slider('mc-L', 'Pierderea lunară L, în %' if ro else 'Monthly loss L, in %', 5, 20, 1, 10) + play('mc'), charts('mc-chart')),
     'binomial': (slider('bn-n', 'Numărul de pași n pe un an' if ro else 'Number of steps n in one year', 0, 6, 1, 6) + play('bn'), charts('bn-chart')),
     'processes': (slider('pr-phi', 'φ al procesului AR(1) din panoul al treilea (linia portocalie din celelalte panouri)' if ro else 'φ of the AR(1) process in the third panel (amber line in the other panels)', 0, 1, 0.01, 0.95) + play('pr'), charts('pr-chart')),
     'spot-the-real': (slider('sr-k', 'Pragul k, în abateri standard (linii punctate; zilele dincolo de prag sînt marcate)' if ro else 'Threshold k, in standard deviations (dashed lines; the days beyond it are marked)', 2, 8, 0.5, 4) + play('sr'), charts('sr-chart')),
     'wiener': (slider('wn-k', 'Banda ±k√t (linii punctate, în ambele panouri)' if ro else 'Band ±k√t (dotted lines, in both panels)', 0.5, 3, 0.1, 2) + play('wn'), charts('wn-chart')),
     'gbm-paths': (slider('gp-s', 'Volatilitatea σ (μ = 8,08% fixat)' if ro else 'Volatility σ (μ = 8.08% fixed)', 5, 60, 0.01, 19.26) + play('gp'), charts('gp-chart')),
     'gbm-fan': (select('gf-b', 'Banda exterioară a evantaiului' if ro else 'Outer band of the fan', bands, '0.9') + play('gf'), charts('gf-chart')),
     'gbm-check': (slider('gc-k', 'Pragul k (zilele de k sigma; cercurile goale: randamente limitate la ±k abateri standard)' if ro else 'Threshold k (k-sigma days; open circles: returns clipped at ±k standard deviations)', 3, 8, 0.5, 4) + play('gc'), charts('gc-chart')),
    }


# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH4_TEXT = {
 rv: (a, w, p, m, m5, v, v5, r) => `${a}: K în ${w} de săptămîni complete, ponderea zilelor de creștere p = ${p}. Media lui K ${m}, în distribuția binomială 5p = ${m5}; varianța ${v}, în distribuția binomială 5p(1 − p) = ${v5} (raportul ${r}).`,
 cq: (a, n, al, ve, vn, r) => `${a}, n = ${n} zile. VaR ${al}: empiric ${ve}%, în distribuția Normală cu aceeași medie și abatere standard ${vn}% (raportul ${r}).`,
 cq2: (p3e, p3n, p5e, p5n, r) => `P(r &lt; −3%): ${p3e} în date, ${p3n} în distribuția Normală; P(r &lt; −5%): ${p5e} față de ${p5n} (raportul ${r}).`,
 jt: y => y ? `Corelația pe întreaga perioadă și în zilele din ${y} (punctele violet):` : 'Corelația pe zilele comune ale întregii perioade:',
 jtItem: (a, b, n, c, y, ny, cy) => `${a} și ${b}: ${n} zile, ρ = ${c}` + (y ? `; în ${y}: ρ = ${cy} (${ny} zile)` : ''),
 ud: (h, cxy) => `Stînga: Corr(X, X²) = ${cxy} în cele 3 000 de extrageri simulate. Dreapta, lagul h = ${h}:`,
 udL: (h, lo, mn, hi) => `Stînga, linia roșie: S&P 500, z = (r − media)/abaterea standard. Media lui z<sub>t</sub>² este ${lo} în decila cea mai joasă a lui z<sub>t−${h}</sub>, ${hi} în cea mai înaltă și ${mn} în decila cu media cea mai mică (pe toate zilele, circa 1).`,
 udItem: (a, c1, c2, b) => `${a}: Corr(r<sub>t</sub>, r<sub>t−h</sub>) = ${c1}, Corr(r<sub>t</sub>², r<sub>t−h</sub>²) = ${c2}; banda ±${b}`,
 cd: q => `${q} grupuri de zile după |r<sub>t−1</sub>|, de la cel mai calm (grupul 1) la cel mai agitat (grupul ${q}):`,
 cdItem: (a, m0, m1, s1, q, sq, r, tot, w, b, sh) => `${a}: media condiționată între ${m0}% și ${m1}%; abaterea standard condiționată ${s1}% (grupul 1) și ${sq}% (grupul ${q}), raportul ${r}. Varianța totală ${tot} = ${w} (în interiorul grupurilor) + ${b} (între grupuri, ${sh}% din total)`,
 ln: (y, n, yr, m, se, t, ma, lo, hi, y3) => `S&P 500 din ${y}: n = ${n} zile (${yr} ani), media ${m}% pe zi, eroarea standard ${se}%, t = ${t}. Media anuală ${ma}%, intervalul de încredere de 95%: [${lo}%; ${hi}%].` + (y3 ? ` Pentru t = 3, la această medie și volatilitate: circa ${y3} de ani de date.` : ''),
 ru: (s, nv, lo, hi, c, nd, nt) => `RANDU, x<sub>0</sub> = ${s}: 9u<sub>k</sub> − 6u<sub>k+1</sub> + u<sub>k+2</sub> ia ${nv} valori întregi, de la ${lo} la ${hi}; corelația perechilor (u<sub>k</sub>, u<sub>k+1</sub>) ${c}. PCG64 cu aceeași sămînță: ${nd} de valori diferite în ${nt} de triplete.`,
 it: (l, m, t) => `Stînga: 200 000 de extrageri −ln(1 − U)/${l}, media de selecție ${m} (teoretic 1/λ = ${t}).`,
 it2: (x, cn, pn, pe, en, ch, ph, pd) => `Extrageri sub −${x}%: ${cn} din cele Normale (${pn}); distribuția Normală exactă dă ${pe}, adică ${en} extrageri așteptate din 200 000. Extrageri istorice: ${ch} (${ph}); în date: ${pd}.`,
 it3: (mn, mh) => `Cea mai mică extragere: ${mn}% (Normală) față de ${mh}% (istorică).`,
 mc: (L, z, p) => `Pierdere pe 21 de zile peste ${L}%: z = ${z}, valoarea exactă p = Φ(z) = ${p}.`,
 mc2: (e3, s3, e5, s5, e6, s6) => `Estimarea Monte Carlo: N = 1 000: ${e3} (eroarea standard ${s3}); N = 100 000: ${e5} (${s5}); N = 10<sup>6</sup>: ${e6} (${s6}).`,
 mc3: n => `Pentru o eroare relativă de 10% la nivelul de încredere de 95%: N ≈ 1,96²(1 − p)/(0,1² p) = ${n}.`,
 bn: (n, u, d, p, es, et) => `n = ${n} pași: u = ${u}, d = ${d}, p = ${p}. E[S<sub>T</sub>] = ${es}, 100e<sup>μ</sup> = ${et}.`,
 bn2: (le, lt, tie, ln) => `P(S<sub>T</sub> ≤ 100) = ${le}, P(S<sub>T</sub> &lt; 100) = ${lt}` + (tie ? ` (nodul S<sub>T</sub> = 100 are probabilitatea ${tie})` : ' (n impar: niciun nod în 100)') + `; în limita lognormală P(S<sub>T</sub> &lt; 100) = ${ln}.`,
 bn3: (qb, ql) => `Cuantila de 5% a lui S<sub>T</sub>: ${qb} în modelul binomial, ${ql} în limita lognormală.`,
 pr: (phi, v, r5, hl, vs) => `φ = ${phi}: varianța staționară 1/(1 − φ²) = ${v === null ? '∞ (nestaționar)' : v}, ρ(5) = φ⁵ = ${r5}, timpul de înjumătățire ${hl} perioade. Varianța de selecție a celor 500 de puncte din panoul al treilea: ${vs}.`,
 prSim: (phi, v, vt, r5) => `Simularea lungă din curs (σ = 1, φ = ${phi}): varianța ${v} (teoretic ${vt}), ρ(5) = ${r5}.`,
 prRw: (v10, v50, v100) => `Mersul aleator din curs (5 000 de traiectorii): Var(S<sub>t</sub>) = ${v10}, ${v50} și ${v100} la t = 10, 50 și 100.`,
 sr: k => `Zile cu |r − media| > ${k} abateri standard, excesul de boltire, Corr(r<sub>t</sub>², r<sub>t−1</sub>²) și cea mai mare mișcare zilnică în valoare absolută:`,
 srItem: (c, f, k, a, m) => `coloana ${c}: ${f === 1 ? '1 zi' : f % 100 >= 20 || (f >= 100 && f % 100 === 0) ? f + ' de zile' : f + ' zile'}; exces de boltire ${k}; Corr(r<sub>t</sub>², r<sub>t−1</sub>²) = ${a}; cea mai mare mișcare ${m}%`,
 wn: (k, s, th, n20) => `W(1) în ±${k}: ${s} din 10 000 de traiectorii simulate, teoretic 2Φ(${k}) − 1 = ${th}; ${n20} din cele 20 de traiectorii ale graficului.`,
 wn2: (v, qm, qs) => `Var(W(1)) simulată ${v}; variația pătratică pe 1 000 de pași: media ${qm}, abaterea standard ${qs}.`,
 gp: (s, mu, dr, sl) => `σ = ${s}%, μ = ${mu}%: tendința logaritmică μ − σ²/2 = ${dr}% pe an.` + (sl ? ' Calibrarea din curs, S&P 500 2000–2026.' : ' Rulare suplimentară a formulelor capitolului, cu aceleași șocuri; cursul folosește σ = 19,26%.'),
 gp2: (m, md, sb, ss, b40, pl) => `După T = 10 ani: media ${m}, mediana ${md}. Ponderea traiectoriilor sub medie: Φ(σ√T/2) = ${sb}` + (ss ? ` (${ss} în 200 000 de traiectorii simulate)` : '') + `; ${b40} din cele 40 de traiectorii ale graficului. Probabilitatea unei pierderi după 10 ani: ${pl}.`,
 gf: (b, nom) => `Ponderea zilelor în care traiectoria reală este în afara benzii GBM de ${b} (nominal ${nom}):`,
 gfItem: (a, o, s) => `${a} (σ = ${s}% pe an): ${o}`,
 gc: k => `Zile cu |r − media| > ${k} abateri standard: date față de 300 de simulări GBM de aceeași lungime (mediana, cuantila de 95%, maximul):`,
 gcItem: (a, n, f, m, q, mx) => `${a} (${n} zile): ${f} în date; GBM ${m}, ${q}, ${mx}`,
 gcClip: (k, items, nk) => `Randamentele reale limitate la media ± ${k} abateri standard (cercurile goale): exces de boltire și Corr(r<sub>t</sub>², r<sub>t−1</sub>²), față de seria nelimitată: ${items}. Pentru randamente i.i.d. Normale, aceeași limitare dă excesul de boltire ${nk}, deci benzile GBM s-ar deplasa cu aproximativ atît.`,
 gcClipItem: (a, kc, kr, ac, ar) => `${a} ${kc} (${kr}), ${ac} (${ar})`
};
""",
 'en': r"""window.CH4_TEXT = {
 rv: (a, w, p, m, m5, v, v5, r) => `${a}: K in ${w} full weeks, share of up days p = ${p}. Mean of K ${m}, binomial 5p = ${m5}; variance ${v}, binomial 5p(1 − p) = ${v5} (ratio ${r}).`,
 cq: (a, n, al, ve, vn, r) => `${a}, n = ${n} days. VaR ${al}: empirical ${ve}%, Normal distribution with the same mean and standard deviation ${vn}% (ratio ${r}).`,
 cq2: (p3e, p3n, p5e, p5n, r) => `P(r &lt; −3%): ${p3e} in the data, ${p3n} under the Normal distribution; P(r &lt; −5%): ${p5e} against ${p5n} (ratio ${r}).`,
 jt: y => y ? `Correlation over the whole period and on the days of ${y} (purple points):` : 'Correlation on the common days of the whole period:',
 jtItem: (a, b, n, c, y, ny, cy) => `${a} and ${b}: ${n} days, ρ = ${c}` + (y ? `; in ${y}: ρ = ${cy} (${ny} days)` : ''),
 ud: (h, cxy) => `Left: Corr(X, X²) = ${cxy} in the 3,000 simulated draws. Right, lag h = ${h}:`,
 udL: (h, lo, mn, hi) => `Left, red line: S&P 500, z = (r − mean)/standard deviation. The mean of z<sub>t</sub>² is ${lo} in the lowest decile of z<sub>t−${h}</sub>, ${hi} in the highest and ${mn} in the decile with the smallest mean (over all days, about 1).`,
 udItem: (a, c1, c2, b) => `${a}: Corr(r<sub>t</sub>, r<sub>t−h</sub>) = ${c1}, Corr(r<sub>t</sub>², r<sub>t−h</sub>²) = ${c2}; band ±${b}`,
 cd: q => `${q} groups of days by |r<sub>t−1</sub>|, from the calmest (group 1) to the most turbulent (group ${q}):`,
 cdItem: (a, m0, m1, s1, q, sq, r, tot, w, b, sh) => `${a}: conditional mean between ${m0}% and ${m1}%; conditional standard deviation ${s1}% (group 1) and ${sq}% (group ${q}), ratio ${r}. Total variance ${tot} = ${w} (within groups) + ${b} (between groups, ${sh}% of the total)`,
 ln: (y, n, yr, m, se, t, ma, lo, hi, y3) => `S&P 500 from ${y}: n = ${n} days (${yr} years), mean ${m}% per day, standard error ${se}%, t = ${t}. Annual mean ${ma}%, 95% confidence interval: [${lo}%, ${hi}%].` + (y3 ? ` For t = 3, at this mean and volatility: about ${y3} years of data.` : ''),
 ru: (s, nv, lo, hi, c, nd, nt) => `RANDU, x<sub>0</sub> = ${s}: 9u<sub>k</sub> − 6u<sub>k+1</sub> + u<sub>k+2</sub> takes ${nv} integer values, from ${lo} to ${hi}; correlation of the pairs (u<sub>k</sub>, u<sub>k+1</sub>) ${c}. PCG64 with the same seed: ${nd} different values in ${nt} triples.`,
 it: (l, m, t) => `Left: 200,000 draws −ln(1 − U)/${l}, sample mean ${m} (theoretical 1/λ = ${t}).`,
 it2: (x, cn, pn, pe, en, ch, ph, pd) => `Draws below −${x}%: ${cn} of the Normal ones (${pn}); the exact Normal distribution gives ${pe}, i.e. ${en} expected draws out of 200,000. Historical draws: ${ch} (${ph}); in the data: ${pd}.`,
 it3: (mn, mh) => `Smallest draw: ${mn}% (Normal) against ${mh}% (historical).`,
 mc: (L, z, p) => `21-day loss above ${L}%: z = ${z}, exact value p = Φ(z) = ${p}.`,
 mc2: (e3, s3, e5, s5, e6, s6) => `Monte Carlo estimate: N = 1,000: ${e3} (standard error ${s3}); N = 100,000: ${e5} (${s5}); N = 10<sup>6</sup>: ${e6} (${s6}).`,
 mc3: n => `For a relative error of 10% at the 95% confidence level: N ≈ 1.96²(1 − p)/(0.1² p) = ${n}.`,
 bn: (n, u, d, p, es, et) => `n = ${n} steps: u = ${u}, d = ${d}, p = ${p}. E[S<sub>T</sub>] = ${es}, 100e<sup>μ</sup> = ${et}.`,
 bn2: (le, lt, tie, ln) => `P(S<sub>T</sub> ≤ 100) = ${le}, P(S<sub>T</sub> &lt; 100) = ${lt}` + (tie ? ` (the node S<sub>T</sub> = 100 has probability ${tie})` : ' (odd n: no node at 100)') + `; lognormal limit P(S<sub>T</sub> &lt; 100) = ${ln}.`,
 bn3: (qb, ql) => `5% quantile of S<sub>T</sub>: ${qb} in the binomial model, ${ql} in the lognormal limit.`,
 pr: (phi, v, r5, hl, vs) => `φ = ${phi}: stationary variance 1/(1 − φ²) = ${v === null ? '∞ (non-stationary)' : v}, ρ(5) = φ⁵ = ${r5}, half-life ${hl} periods. Sample variance of the 500 points of the third panel: ${vs}.`,
 prSim: (phi, v, vt, r5) => `Long simulation of the lecture (σ = 1, φ = ${phi}): variance ${v} (theoretical ${vt}), ρ(5) = ${r5}.`,
 prRw: (v10, v50, v100) => `Random walk of the lecture (5,000 paths): Var(S<sub>t</sub>) = ${v10}, ${v50} and ${v100} at t = 10, 50 and 100.`,
 sr: k => `Days with |r − mean| > ${k} standard deviations, excess kurtosis, Corr(r<sub>t</sub>², r<sub>t−1</sub>²) and the largest daily move in absolute value:`,
 srItem: (c, f, k, a, m) => `column ${c}: ${f} ${f === 1 ? 'day' : 'days'}; excess kurtosis ${k}; Corr(r<sub>t</sub>², r<sub>t−1</sub>²) = ${a}; largest move ${m}%`,
 wn: (k, s, th, n20) => `W(1) within ±${k}: ${s} of 10,000 simulated paths, theoretical 2Φ(${k}) − 1 = ${th}; ${n20} of the 20 paths of the chart.`,
 wn2: (v, qm, qs) => `Simulated Var(W(1)) ${v}; quadratic variation over 1,000 steps: mean ${qm}, standard deviation ${qs}.`,
 gp: (s, mu, dr, sl) => `σ = ${s}%, μ = ${mu}%: log drift μ − σ²/2 = ${dr}% per year.` + (sl ? ' The calibration of the lecture, S&P 500 2000–2026.' : ' Extra run of the chapter\'s formulas, with the same shocks; the lecture uses σ = 19.26%.'),
 gp2: (m, md, sb, ss, b40, pl) => `After T = 10 years: mean ${m}, median ${md}. Share of paths below the mean: Φ(σ√T/2) = ${sb}` + (ss ? ` (${ss} in 200,000 simulated paths)` : '') + `; ${b40} of the 40 paths of the chart. Probability of a loss after 10 years: ${pl}.`,
 gf: (b, nom) => `Share of the days on which the real path is outside the ${b} GBM band (nominal ${nom}):`,
 gfItem: (a, o, s) => `${a} (σ = ${s}% per year): ${o}`,
 gc: k => `Days with |r − mean| > ${k} standard deviations: data against 300 GBM simulations of the same length (median, 95% quantile, maximum):`,
 gcItem: (a, n, f, m, q, mx) => `${a} (${n} days): ${f} in the data; GBM ${m}, ${q}, ${mx}`,
 gcClip: (k, items, nk) => `Real returns clipped at the mean ± ${k} standard deviations (open circles): excess kurtosis and Corr(r<sub>t</sub>², r<sub>t−1</sub>²), against the unclipped series: ${items}. For i.i.d. Normal returns, the same clipping gives an excess kurtosis of ${nk}, so the GBM bands would shift by about that much.`,
 gcClipItem: (a, kc, kr, ac, ar) => `${a} ${kc} (${kr}), ${ac} (${ar})`
};
""",
}

QA = {
 'ro': {
  'random-variables': ('Pentru S&P 500, varianța lui K (1,16) este puțin sub varianța binomială 5p(1 − p) = 1,24. Butonul ▶ trece la DAX și BET. Ce se întîmplă pentru BET, al cărui randament zilnic are autocorelația de ordinul 1 egală cu 0,109?',
                       'Varianța lui K depășește valoarea binomială: 1,47 față de 1,24, raportul 1,18. Distribuția binomială presupune zile independente; cu autocorelația pozitivă a randamentelor BET, zilele de creștere vin mai des una după alta, deci săptămînile cu multe sau cu puține zile de creștere sînt mai frecvente. Media rămîne aproape de 5p: 2,67 față de 2,70. Pentru S&P 500, cu autocorelația −0,099, raportul este 0,93; pentru DAX tot 0,93.'),
  'cdf-quantile': ('La nivelul 1%, VaR empiric al S&P 500 (3,45%) este de 1,24 ori VaR din distribuția Normală cu aceeași medie și abatere standard (2,80%). Butonul ▶ coboară nivelul α de la 5% la 0,1%. Cît devine raportul la α = 0,1%?',
                   'Circa 1,94: VaR 0,1% este 7,22% în date și 3,72% în distribuția Normală. La α = 5% raportul este 0,94 (1,85% față de 1,97%). Dacă diferența ar fi doar de scală, raportul ar fi același la orice nivel; el crește spre coadă, deci problema este forma distribuției: un vîrf mai înalt și cozi mai groase. P(r < −5%) este 0,34% în date și 1,7 × 10<sup>−3</sup>% în distribuția Normală, de 199 de ori mai puțin.'),
  'joint': ('Corelația zilnică S&P 500–BET pe întreaga perioadă este 0,21. Cît este în zilele din 2020?',
            '0,63, în 242 de zile comune, de trei ori valoarea pe întreaga perioadă. În 2008 corelația este 0,28, în 2017 0,20. Pentru S&P 500 și DAX: 0,69 în 2020, 0,62 în 2008 și 0,57 în 2017, față de 0,60 pe toată perioada. Corelația crește în anii de criză, adică exact cînd diversificarea ar fi necesară.'),
  'uncorrelated-dependent': ('Butonul ▶ mărește lagul h de la 1 la 20. La h = 20, mai iese Corr(r<sub>t</sub>², r<sub>t−20</sub>²) a S&P 500 din banda ±1,96/√n = ±0,024?',
                             'Da: 0,164 la lagul 20, față de 0,313 la lagul 1, în timp ce corelația randamentelor la lagul 20 este 0,014, în bandă. La lagul 1, corelația pătratelor este 0,31 (S&P 500), 0,15 (DAX), 0,33 (BET) și 0,14 (Bitcoin). La lagul 20 rămîne peste bandă pentru DAX (0,116) și BET (0,090); pentru Bitcoin (0,026) intră în banda ±0,030. Volatility clustering durează săptămîni. Banda este o valoare de referință asimptotică pentru o serie i.i.d., nu un test exact pentru fiecare lag.'),
  'conditional': ('Cu 5 grupuri (chintile), raportul abaterilor standard dintre grupul cel mai agitat și cel mai calm este 1,88 pentru S&P 500. Butonul ▶ trece de la 2 la 10 grupuri. Cît devine raportul la 10 grupuri?',
                  '2,22: 2,03% în grupul cel mai agitat față de 0,92% în cel mai calm. Cu 2 grupuri raportul este 1,43: grupurile extreme sînt mai extreme cînd sînt mai înguste. Componenta dintre grupuri a varianței totale rămîne neglijabilă: 0,054% din total la 10 grupuri, 0,009% la 5. Mișcarea de ieri spune mult despre riscul de azi și aproape nimic despre media de azi.'),
  'lln': ('Pe 2000–2026, media zilnică a S&P 500 are t = 1,67. Butonul ▶ mută începutul eșantionului de la 2000 la 2020. Ce valoare are t pentru eșantionul care începe în 2009?',
          '2,83: media zilnică este 0,0480% (12,1% pe an), eroarea standard 0,0169%. Eșantionul din 2009 începe după criza din 2008. Din 2003, t = 2,38; din 2020, t = 1,64 cu doar 1 687 de zile. Intervalul de 95% al mediei anuale este [3,7%; 20,4%] din 2009 și [−1,1%; 13,5%] din 2000. Media estimată depinde mult de începutul eșantionului; un început ales după ce s-au văzut datele nu mai dă un test valid.'),
  'randu': ('Butonul ▶ schimbă sămînța x<sub>0</sub> a RANDU (numere impare de la 1 la 99). Cîte valori distincte ia combinația 9u<sub>k</sub> − 6u<sub>k+1</sub> + u<sub>k+2</sub> pentru x<sub>0</sub> = 99?',
            '15, de la −5 la 9, ca pentru orice sămînță impară. Relația vine din multiplicator: 65539 = 2<sup>16</sup> + 3, deci a² = 6a − 9 (mod 2<sup>31</sup>) și x<sub>k+2</sub> = 6x<sub>k+1</sub> − 9x<sub>k</sub> (mod 2<sup>31</sup>), oricare ar fi x<sub>0</sub>. Corelația perechilor rămîne mică (0,0078 pentru x<sub>0</sub> = 99), deci un test pe perechi nu vede defectul. Cu PCG64, combinația ia 29 998 de valori diferite în 29 998 de triplete.'),
  'inverse-transform': ('Dintre cele 200 000 de extrageri din distribuția Normală, 7 (0,0035%) sînt sub −5%. Cît dă distribuția Normală exactă, cu aceeași medie și abatere standard, pentru P(r < −5%)?',
                        '1,7 × 10<sup>−3</sup>%, adică 0,0017%, sau 3,4 extrageri așteptate din 200 000. Cu atît de puține evenimente, abaterea standard a numărului de extrageri este circa √3,4 ≈ 1,84, deci 7 extrageri sînt zgomot de simulare, nu un defect al metodei. Extragerile istorice dau 0,34% (685 din 200 000), ca datele: transformarea inversă cu funcția cuantilă empirică reproduce coada eșantionului. La −8%: nicio extragere Normală, 148 istorice.'),
  'monte-carlo': ('Pentru o pierdere lunară peste 10%, o eroare relativă de 10% (la 95%) cere N = 12 762 de simulări. Cîte cere aceeași precizie pentru o pierdere lunară peste 15%?',
                  'Circa 146 000 (146 201): p scade de la 2,922% la 0,262%, iar N ≈ 1,96²(1 − p)/(0,1² p) crește aproape ca 1/p. Pentru 20%, p = 0,0112% și N = 3,44 milioane; cu un milion de simulări estimarea este 0,0117%, cu eroarea standard 0,0011%, circa 10% din p. Eroarea absolută √(p(1 − p)/N) scade odată cu p, dar eroarea relativă crește.'),
  'binomial': ('Cu 252 de pași, P(S<sub>T</sub> ≤ 100) = 39,7%, față de 37,3% în limita lognormală. Cît este P(S<sub>T</sub> ≤ 100) cu doar 2 pași?',
               '61,9%, din care 47,3% este nodul S<sub>T</sub> = 100 (o creștere și o scădere, ud = 1); P(S<sub>T</sub> < 100) este doar 14,7%. Cu n par, arborele are un nod exact în S<sub>0</sub>, cu o probabilitate care scade ca 1/√n: 6,7% la 126 de pași, 4,8% la 252. Cu 5 pași nu există acest nod și P(S<sub>T</sub> < 100) = 36,5%. Valoarea de 39,7% de pe slide este P(S<sub>T</sub> ≤ 100); strict sub 100 sînt 34,9%. E[S<sub>T</sub>] = 108,42 la orice n, prin construcția lui p.'),
  'processes': ('Pentru φ = 0,95, timpul de înjumătățire al unui șoc este 13,5 perioade. Cît este pentru φ = 0,99?',
                'ln 0,5/ln 0,99 = 69,0 perioade, de cinci ori mai mult. Varianța staționară 1/(1 − φ²) crește de la 10,26 la 50,25, iar ρ(5) = 0,99⁵ = 0,951. Procesul este încă staționar și revine la 0, dar pe 500 de puncte se deosebește greu de un mers aleator; numai φ = 1 face șocurile permanente.'),
  'spot-the-real': ('Care coloană este S&P 500, după randamente?',
                    'Coloana C. Prețurile nu o deosebesc: și mersurile aleatoare cu tendință au avînturi și căderi lungi. Randamentele o deosebesc: exces de boltire 10,67 față de cel mult 0,02 în simulări, Corr(r<sub>t</sub>², r<sub>t−1</sub>²) = 0,313 față de cel mult 0,006, cea mai mare mișcare zilnică 12,8% față de 6,2%. Peste 4 abateri standard: 46 de zile în C, cel mult una în coloanele simulate; peste 6 abateri standard: 12 zile în C, niciuna în simulări.'),
  'wiener': ('68,2% din cele 10 000 de valori simulate W(1) sînt în ±1. Ce pondere este în ±2?',
             '95,37% în simulare, teoretic 2Φ(2) − 1 = 95,45%, pentru că W(1) ~ N(0, 1). Inegalitatea lui Cebîșev garantează doar cel puțin 1 − 1/2² = 75%, pentru orice distribuție cu varianță finită. Banda ±k√t descrie distribuția lui W(t), nu o margine a traiectoriilor: dintre cele 20 de traiectorii ale graficului, toate se termină în ±2, dar doar 14 în ±1. Pentru k = 3: 99,67% față de 99,73%.'),
  'gbm-paths': ('Butonul ▶ mărește σ de la 5% la 60%, cu aceeași medie μ = 8,08% pe an. Pentru σ = 40%, unde ajunge mediana lui S<sub>10</sub>?',
                'La 100,8 = 100e<sup>(0,0808 − 0,08) × 10</sup>: tendința logaritmică μ − σ²/2 scade la 0,08% pe an. Media rămîne 224,4, pentru că depinde doar de μ. Probabilitatea unei pierderi după 10 ani crește de la 15,3% (σ = 19,26%) la 49,7%, iar ponderea traiectoriilor sub medie, Φ(σ√T/2), de la 62,0% la 73,6%. σ = 40% este o rulare suplimentară a formulelor capitolului, nu un rezultat din curs.'),
  'gbm-fan': ('Cu banda de 90%, BET este în afara evantaiului GBM în 24% din zile. Ce pondere dă banda de 99%, unde nominal ar fi 1%?',
              '12,7%: mai puțin, dar de 13 ori valoarea nominală; avîntul din 2000–2007 și crahul din 2008 duc traiectoria departe de GBM. S&P 500: 7,0% în afara benzii de 90% și 0,0% în afara celei de 99%; Bitcoin: 4,5% și 0,1%. Zilele unei singure traiectorii nu sînt observații independente: traiectoria rămîne mult timp de aceeași parte a medianei, așa că și S&P 500 este în afara benzii de 50% în 64,2% din zile.'),
  'gbm-check': ('Butonul ▶ ridică pragul k de la 3 la 8 abateri standard. La k = 6, cîte zile are S&P 500 și cîte are cea mai extremă dintre cele 300 de simulări GBM?',
                '12 zile în date și 0 în fiecare dintre cele 300 de simulări. BET are 17 astfel de zile, Bitcoin 4. La k = 3, S&P 500 are 105 zile, față de mediana GBM 18; la k = 8, încă 4 zile. În distribuția Normală, numărul așteptat de zile peste 6 abateri standard în 6 714 zile este de ordinul 10<sup>−5</sup>.'),
 },
 'en': {
  'random-variables': ('For the S&P 500, the variance of K (1.16) is slightly below the binomial variance 5p(1 − p) = 1.24. The ▶ button moves to the DAX and BET. What happens for BET, whose daily return has a first-order autocorrelation of 0.109?',
                       'The variance of K exceeds the binomial value: 1.47 against 1.24, ratio 1.18. The binomial distribution assumes independent days; with the positive autocorrelation of BET returns, up days come one after another more often, so weeks with many or with few up days are more frequent. The mean stays close to 5p: 2.67 against 2.70. For the S&P 500, with autocorrelation −0.099, the ratio is 0.93; for the DAX also 0.93.'),
  'cdf-quantile': ('At the 1% level, the empirical VaR of the S&P 500 (3.45%) is 1.24 times the VaR of the Normal distribution with the same mean and standard deviation (2.80%). The ▶ button lowers the level α from 5% to 0.1%. What does the ratio become at α = 0.1%?',
                   'About 1.94: VaR 0.1% is 7.22% in the data and 3.72% under the Normal distribution. At α = 5% the ratio is 0.94 (1.85% against 1.97%). If the difference were only one of scale, the ratio would be the same at every level; it grows towards the tail, so the problem is the shape of the distribution: a higher peak and heavier tails. P(r < −5%) is 0.34% in the data and 1.7 × 10<sup>−3</sup>% under the Normal distribution, 199 times less.'),
  'joint': ('The daily correlation of the S&P 500 and BET over the whole period is 0.21. What is it on the days of 2020?',
            '0.63, on 242 common days, three times the value over the whole period. In 2008 the correlation is 0.28, in 2017 0.20. For the S&P 500 and the DAX: 0.69 in 2020, 0.62 in 2008 and 0.57 in 2017, against 0.60 over the whole period. Correlation rises in crisis years, exactly when diversification would be needed.'),
  'uncorrelated-dependent': ('The ▶ button raises the lag h from 1 to 20. At h = 20, is Corr(r<sub>t</sub>², r<sub>t−20</sub>²) of the S&P 500 still outside the band ±1.96/√n = ±0.024?',
                             'Yes: 0.164 at lag 20, against 0.313 at lag 1, while the correlation of the returns at lag 20 is 0.014, inside the band. At lag 1, the correlation of the squares is 0.31 (S&P 500), 0.15 (DAX), 0.33 (BET) and 0.14 (Bitcoin). At lag 20 it stays above the band for the DAX (0.116) and BET (0.090); for Bitcoin (0.026) it falls inside the band ±0.030. Volatility clustering lasts for weeks. The band is an asymptotic reference value for an i.i.d. series, not an exact test at every lag.'),
  'conditional': ('With 5 groups (quintiles), the ratio of the standard deviations of the most turbulent and the calmest group is 1.88 for the S&P 500. The ▶ button goes from 2 to 10 groups. What does the ratio become with 10 groups?',
                  '2.22: 2.03% in the most turbulent group against 0.92% in the calmest. With 2 groups the ratio is 1.43: the extreme groups are more extreme when they are narrower. The between-group component of the total variance stays negligible: 0.054% of the total with 10 groups, 0.009% with 5. Yesterday\'s move says a lot about today\'s risk and almost nothing about today\'s mean.'),
  'lln': ('Over 2000–2026, the daily mean of the S&P 500 has t = 1.67. The ▶ button moves the start of the sample from 2000 to 2020. What is t for the sample that starts in 2009?',
          '2.83: the daily mean is 0.0480% (12.1% per year), the standard error 0.0169%. The sample from 2009 starts after the 2008 crisis. From 2003, t = 2.38; from 2020, t = 1.64 with only 1,687 days. The 95% interval of the annual mean is [3.7%, 20.4%] from 2009 and [−1.1%, 13.5%] from 2000. The estimated mean depends strongly on the start of the sample; a start chosen after looking at the data no longer gives a valid test.'),
  'randu': ('The ▶ button changes the RANDU seed x<sub>0</sub> (odd numbers from 1 to 99). How many distinct values does the combination 9u<sub>k</sub> − 6u<sub>k+1</sub> + u<sub>k+2</sub> take for x<sub>0</sub> = 99?',
            '15, from −5 to 9, as for every odd seed. The relation comes from the multiplier: 65539 = 2<sup>16</sup> + 3, so a² = 6a − 9 (mod 2<sup>31</sup>) and x<sub>k+2</sub> = 6x<sub>k+1</sub> − 9x<sub>k</sub> (mod 2<sup>31</sup>), whatever x<sub>0</sub>. The correlation of the pairs stays small (0.0078 for x<sub>0</sub> = 99), so a test on pairs does not see the flaw. With PCG64, the combination takes 29,998 different values in 29,998 triples.'),
  'inverse-transform': ('Of the 200,000 draws from the Normal distribution, 7 (0.0035%) are below −5%. What does the exact Normal distribution, with the same mean and standard deviation, give for P(r < −5%)?',
                        '1.7 × 10<sup>−3</sup>%, i.e. 0.0017%, or 3.4 expected draws out of 200,000. With so few events, the standard deviation of the number of draws is about √3.4 ≈ 1.84, so 7 draws are simulation noise, not a flaw of the method. The historical draws give 0.34% (685 of 200,000), as the data: the inverse transform with the empirical quantile function reproduces the tail of the sample. At −8%: no Normal draw, 148 historical ones.'),
  'monte-carlo': ('For a monthly loss above 10%, a relative error of 10% (at 95%) needs N = 12,762 simulations. How many does the same precision need for a monthly loss above 15%?',
                  'About 146,000 (146,201): p falls from 2.922% to 0.262%, and N ≈ 1.96²(1 − p)/(0.1² p) grows almost as 1/p. For 20%, p = 0.0112% and N = 3.44 million; with one million simulations the estimate is 0.0117%, with standard error 0.0011%, about 10% of p. The absolute error √(p(1 − p)/N) falls with p, but the relative error grows.'),
  'binomial': ('With 252 steps, P(S<sub>T</sub> ≤ 100) = 39.7%, against 37.3% in the lognormal limit. What is P(S<sub>T</sub> ≤ 100) with only 2 steps?',
               '61.9%, of which 47.3% is the node S<sub>T</sub> = 100 (one up and one down move, ud = 1); P(S<sub>T</sub> < 100) is only 14.7%. With an even n, the tree has a node exactly at S<sub>0</sub>, with a probability that falls as 1/√n: 6.7% at 126 steps, 4.8% at 252. With 5 steps there is no such node and P(S<sub>T</sub> < 100) = 36.5%. The 39.7% of the slide is P(S<sub>T</sub> ≤ 100); strictly below 100 is 34.9%. E[S<sub>T</sub>] = 108.42 for every n, by the construction of p.'),
  'processes': ('For φ = 0.95, the half-life of a shock is 13.5 periods. What is it for φ = 0.99?',
                'ln 0.5/ln 0.99 = 69.0 periods, five times as long. The stationary variance 1/(1 − φ²) rises from 10.26 to 50.25, and ρ(5) = 0.99⁵ = 0.951. The process is still stationary and returns to 0, but over 500 points it is hard to tell from a random walk; only φ = 1 makes the shocks permanent.'),
  'spot-the-real': ('Which column is the S&P 500, judging by the returns?',
                    'Column C. The prices do not tell it apart: random walks with drift also have long rallies and falls. The returns do: excess kurtosis 10.67 against at most 0.02 in the simulations, Corr(r<sub>t</sub>², r<sub>t−1</sub>²) = 0.313 against at most 0.006, the largest daily move 12.8% against 6.2%. Beyond 4 standard deviations: 46 days in C, at most one in the simulated columns; beyond 6 standard deviations: 12 days in C, none in the simulations.'),
  'wiener': ('68.2% of the 10,000 simulated values W(1) are within ±1. What share is within ±2?',
             '95.37% in the simulation, theoretically 2Φ(2) − 1 = 95.45%, because W(1) ~ N(0, 1). Chebyshev\'s inequality only guarantees at least 1 − 1/2² = 75%, for any distribution with finite variance. The band ±k√t describes the distribution of W(t), not a bound on the paths: of the 20 paths of the chart, all end within ±2, but only 14 within ±1. For k = 3: 99.67% against 99.73%.'),
  'gbm-paths': ('The ▶ button raises σ from 5% to 60%, with the same mean μ = 8.08% per year. For σ = 40%, where does the median of S<sub>10</sub> end?',
                'At 100.8 = 100e<sup>(0.0808 − 0.08) × 10</sup>: the log drift μ − σ²/2 falls to 0.08% per year. The mean stays at 224.4, because it depends only on μ. The probability of a loss after 10 years rises from 15.3% (σ = 19.26%) to 49.7%, and the share of paths below the mean, Φ(σ√T/2), from 62.0% to 73.6%. σ = 40% is an extra run of the chapter\'s formulas, not a result of the lecture.'),
  'gbm-fan': ('With the 90% band, BET is outside the GBM fan on 24% of the days. What share does the 99% band give, where the nominal value would be 1%?',
              '12.7%: less, but 13 times the nominal value; the boom of 2000–2007 and the crash of 2008 take the path far from GBM. S&P 500: 7.0% outside the 90% band and 0.0% outside the 99% band; Bitcoin: 4.5% and 0.1%. The days of a single path are not independent observations: the path stays on the same side of the median for a long time, so even the S&P 500 is outside the 50% band on 64.2% of the days.'),
  'gbm-check': ('The ▶ button raises the threshold k from 3 to 8 standard deviations. At k = 6, how many days does the S&P 500 have, and how many does the most extreme of the 300 GBM simulations have?',
                '12 days in the data and 0 in each of the 300 simulations. BET has 17 such days, Bitcoin 4. At k = 3, the S&P 500 has 105 days, against a GBM median of 18; at k = 8, still 4 days. Under the Normal distribution, the expected number of days beyond 6 standard deviations in 6,714 days is of the order of 10<sup>−5</sup>.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (checked with a node script on ch4_data.js + common.js + ch4.js).
PREDICT = {
 'random-variables': (0, [('Peste valoarea binomială: circa 1,47 față de 1,24.', 'Above the binomial value: about 1.47 against 1.24.'),
                          ('Sub valoarea binomială, ca la S&P 500: varianța depinde doar de p.', 'Below the binomial value, as for the S&P 500: the variance depends only on p.'),
                          ('Egală cu 1,24: media lui K este 5p, deci și varianța este 5p(1 − p).', 'Equal to 1.24: the mean of K is 5p, so the variance is 5p(1 − p).')]),
 'cdf-quantile': (2, [('Tot circa 1,24: distribuția Normală doar subestimează scala.', 'Still about 1.24: the Normal distribution only understates the scale.'),
                      ('Sub 1,24: departe în coadă, distribuția Normală se apropie de date.', 'Below 1.24: far in the tail, the Normal distribution gets closer to the data.'),
                      ('Circa 1,94: 7,22% față de 3,72%.', 'About 1.94: 7.22% against 3.72%.')]),
 'joint': (1, [('Circa 0,21: corelația este o proprietate stabilă a perechii.', 'About 0.21: the correlation is a stable property of the pair.'),
               ('Circa 0,63: corelația crește în criză.', 'About 0.63: the correlation rises in a crisis.'),
               ('Sub 0,21: în criză fiecare piață reacționează la propriile știri.', 'Below 0.21: in a crisis each market reacts to its own news.')]),
 'uncorrelated-dependent': (0, [('Da: 0,164, de aproape 7 ori marginea benzii.', 'Yes: 0.164, almost 7 times the edge of the band.'),
                                ('Nu: după o lună de tranzacționare dependența dispare, ca la randamente.', 'No: after a month of trading the dependence is gone, as for the returns.'),
                                ('Nu: banda se lărgește cu lagul și acoperă valoarea.', 'No: the band widens with the lag and covers the value.')]),
 'conditional': (1, [('Tot 1,88: raportul descrie seria, nu împărțirea în grupuri.', 'Still 1.88: the ratio describes the series, not the grouping.'),
                     ('Circa 2,22: grupurile extreme sînt mai înguste și mai extreme.', 'About 2.22: the extreme groups are narrower and more extreme.'),
                     ('Aproape 1: cu 10 grupuri fiecare are prea puține zile pentru o diferență.', 'Close to 1: with 10 groups each has too few days for a difference.')]),
 'lln': (2, [('Sub 1,67: un eșantion mai scurt are o eroare standard mai mare.', 'Below 1.67: a shorter sample has a larger standard error.'),
             ('Tot circa 1,67: media nu depinde de eșantion.', 'Still about 1.67: the mean does not depend on the sample.'),
             ('Circa 2,83: media zilnică aproape se dublează.', 'About 2.83: the daily mean almost doubles.')]),
 'randu': (0, [('15, de la −5 la 9, ca pentru x₀ = 1.', '15, from −5 to 9, as for x₀ = 1.'),
               ('Circa 30 000: cu altă sămînță tripletele nu mai au structura.', 'About 30,000: with another seed the triples lose the structure.'),
               ('Mai puține: o sămînță mai mare scurtează perioada.', 'Fewer: a larger seed shortens the period.')]),
 'inverse-transform': (1, [('0,0035%: frecvența dintr-un eșantion de 200 000 este exactă.', '0.0035%: the frequency in a sample of 200,000 is exact.'),
                           ('0,0017%: 3,4 extrageri așteptate, iar 7 sînt zgomot de simulare.', '0.0017%: 3.4 expected draws, and 7 are simulation noise.'),
                           ('0,34%: același U dă aceeași coadă prin orice funcție cuantilă.', '0.34%: the same U gives the same tail through any quantile function.')]),
 'monte-carlo': (0, [('Circa 146 000: N crește aproape ca 1/p.', 'About 146,000: N grows almost as 1/p.'),
                     ('Tot 12 762: eroarea Monte Carlo depinde doar de N.', 'Still 12,762: the Monte Carlo error depends only on N.'),
                     ('Circa 19 000: N crește cu raportul pragurilor, 15/10.', 'About 19,000: N grows with the ratio of the thresholds, 15/10.')]),
 'binomial': (2, [('37,3%: calibrarea Cox–Ross–Rubinstein dă limita lognormală la orice n.', '37.3%: the Cox–Ross–Rubinstein calibration gives the lognormal limit for any n.'),
                  ('Circa 39,7%: E[S_T] = 108,42 nu depinde de n, deci nici P(S_T ≤ 100).', 'About 39.7%: E[S_T] = 108.42 does not depend on n, so neither does P(S_T ≤ 100).'),
                  ('61,9%: nodul S_T = 100 are probabilitatea 47,3%.', '61.9%: the node S_T = 100 has probability 47.3%.')]),
 'processes': (0, [('Circa 69 de perioade.', 'About 69 periods.'),
                   ('Circa 14 perioade: φ = 0,99 și φ = 0,95 sînt practic la fel de persistente.', 'About 14 periods: φ = 0.99 and φ = 0.95 are practically equally persistent.'),
                   ('Infinit: cu φ atît de aproape de 1, procesul este un mers aleator.', 'Infinite: with φ so close to 1, the process is a random walk.')]),
 'spot-the-real': (2, [('Oricare: cu aceeași medie și volatilitate, randamentele au aceeași distribuție.', 'Any of them: with the same mean and volatility, the returns have the same distribution.'),
                       ('Nu se poate spune: randamentele zilnice ale oricărei serii par zgomot.', 'Impossible to tell: the daily returns of any series look like noise.'),
                       ('C: zile peste 4 abateri standard și perioade agitate grupate.', 'C: days beyond 4 standard deviations and clustered turbulent periods.')]),
 'wiener': (1, [('Cel puțin 75%, din inegalitatea lui Cebîșev.', 'At least 75%, from Chebyshev\'s inequality.'),
                ('Circa 95,4%: W(1) ~ N(0, 1).', 'About 95.4%: W(1) ~ N(0, 1).'),
                ('100%: ±2√t este anvelopa traiectoriilor.', '100%: ±2√t is the envelope of the paths.')]),
 'gbm-paths': (0, [('Circa 100,8: μ − σ²/2 scade la 0,08% pe an.', 'About 100.8: μ − σ²/2 falls to 0.08% per year.'),
                   ('Tot 186,4: σ lărgește evantaiul, dar nu mută mijlocul.', 'Still 186.4: σ widens the fan but does not move its middle.'),
                   ('Peste 224,4: o volatilitate mai mare aduce o creștere mai mare.', 'Above 224.4: a higher volatility brings a higher growth.')]),
 'gbm-fan': (1, [('Circa 1%: o bandă de 99% prinde aproape orice traiectorie.', 'About 1%: a 99% band catches almost any path.'),
                 ('Circa 12,7%: de 13 ori valoarea nominală.', 'About 12.7%: 13 times the nominal value.'),
                 ('Tot 24%: lățimea benzii nu schimbă ponderea.', 'Still 24%: the width of the band does not change the share.')]),
 'gbm-check': (2, [('Practic 0 în ambele: la 6 abateri standard nu mai rămîne nimic.', 'Practically 0 in both: at 6 standard deviations nothing is left.'),
                   ('Circa 31 în date (46 × 4/6) și 0 în simulări.', 'About 31 in the data (46 × 4/6) and 0 in the simulations.'),
                   ('12 în date și 0 în fiecare simulare.', '12 in the data and 0 in every simulation.')]),
}

# Exit ticket „Verifică-te” / "Check Yourself": questions of the slides „Autoevaluare” and „Formule de reținut”.
# Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('σ₁ = 2%, σ₂ = 3%, ρ = 0,5. Cît este volatilitatea portofoliului 50/50?',
         'σ₁ = 2%, σ₂ = 3%, ρ = 0.5. What is the volatility of the 50/50 portfolio?'),
      key=1, opts=[
  (('2,50%.', '2.50%.'), ('2,50% este media volatilităților, adică rezultatul pentru ρ = 1; cu ρ = 0,5 diversificarea reduce riscul.', '2.50% is the average of the volatilities, the result for ρ = 1; with ρ = 0.5 diversification reduces the risk.')),
  (('2,18%.', '2.18%.'), ('σ_p² = 0,25 × 4 + 0,25 × 9 + 2 × 0,25 × 0,5 × 2 × 3 = 4,75, deci σ_p = 2,18%.', 'σ_p² = 0.25 × 4 + 0.25 × 9 + 2 × 0.25 × 0.5 × 2 × 3 = 4.75, so σ_p = 2.18%.')),
  (('1,80%.', '1.80%.'), ('1,80% = √3,25 omite termenul de covarianță 2ab Cov(X, Y) = 1,5; ar fi rezultatul pentru ρ = 0.', '1.80% = √3.25 leaves out the covariance term 2ab Cov(X, Y) = 1.5; it would be the result for ρ = 0.'))]),
 dict(q=('GBM cu μ = 8%, σ = 20%, S₀ = 100. Cît sînt media și mediana lui S₁?',
         'GBM with μ = 8%, σ = 20%, S₀ = 100. What are the mean and the median of S₁?'),
      key=0, opts=[
  (('Media 108,33, mediana 106,18.', 'Mean 108.33, median 106.18.'), ('E[S₁] = 100e^0,08 = 108,33; mediana 100e^(0,08 − 0,02) = 106,18, mai mică din cauza lui σ²/2.', 'E[S₁] = 100e^0.08 = 108.33; median 100e^(0.08 − 0.02) = 106.18, lower because of σ²/2.')),
  (('Media și mediana 108,33.', 'Mean and median 108.33.'), ('Distribuția lognormală este asimetrică la dreapta: mediana este sub medie.', 'The lognormal distribution is skewed to the right: the median is below the mean.')),
  (('Media 110,52, mediana 108,33.', 'Mean 110.52, median 108.33.'), ('Aici μ = 8% este luat drept tendința logaritmică; în GBM, μ este tendința lui dS/S, iar tendința logaritmică este μ − σ²/2 = 6%.', 'This takes μ = 8% as the log drift; in GBM, μ is the drift of dS/S, and the log drift is μ − σ²/2 = 6%.'))]),
 dict(q=('AR(1) cu φ = 0,8 și σ = 1. Cît este varianța lui?', 'AR(1) with φ = 0.8 and σ = 1. What is its variance?'),
      key=2, opts=[
  (('5,00.', '5.00.'), ('5 = 1/(1 − 0,8) omite pătratul: varianța este σ²/(1 − φ²).', '5 = 1/(1 − 0.8) leaves out the square: the variance is σ²/(1 − φ²).')),
  (('1,00.', '1.00.'), ('1 este varianța șocului ε_t; persistența φ adaugă varianță.', '1 is the variance of the shock ε_t; the persistence φ adds variance.')),
  (('2,78.', '2.78.'), ('σ²/(1 − φ²) = 1/(1 − 0,64) = 2,78.', 'σ²/(1 − φ²) = 1/(1 − 0.64) = 2.78.'))]),
 dict(q=('Eroarea Monte Carlo este √(p(1 − p)/N). De cîte ori trebuie mărit N pentru o eroare standard de două ori mai mică?',
         'The Monte Carlo error is √(p(1 − p)/N). By what factor must N grow for a standard error half as large?'),
      key=0, opts=[
  (('De 4 ori.', '4 times.'), ('Eroarea scade ca 1/√N: N × 4 împarte eroarea la √4 = 2. Pe slide, o precizie de 10 ori mai mare cere de 100 de ori mai multe simulări.', 'The error falls as 1/√N: N × 4 divides the error by √4 = 2. On the slide, 10 times more precision needs 100 times more simulations.')),
  (('De 2 ori.', '2 times.'), ('Dublarea lui N împarte eroarea doar la √2 = 1,41.', 'Doubling N divides the error only by √2 = 1.41.')),
  (('De √2 ori.', '√2 times.'), ('Aici formula este inversată: √2 este factorul cu care scade eroarea cînd N se dublează.', 'This inverts the formula: √2 is the factor by which the error falls when N doubles.'))]),
 dict(q=('Mers aleator S_t = S_{t−1} + ε_t cu σ = 1. Cît este Var(S₁₀₀)?', 'Random walk S_t = S_{t−1} + ε_t with σ = 1. What is Var(S₁₀₀)?'),
      key=1, opts=[
  (('10.', '10.'), ('10 = √100 este abaterea standard, nu varianța.', '10 = √100 is the standard deviation, not the variance.')),
  (('100.', '100.'), ('Var(S_t) = tσ² = 100; simularea din curs, cu 5 000 de traiectorii, dă 100,9.', 'Var(S_t) = tσ² = 100; the simulation of the lecture, with 5,000 paths, gives 100.9.')),
  (('1.', '1.'), ('Varianța constantă este proprietatea unui proces staționar; mersul aleator acumulează șocurile.', 'A constant variance belongs to a stationary process; the random walk accumulates the shocks.'))]),
]


# ================================================================= table and worked-example blocks
# (anchor, frame number in the deck (same in RO and EN), section key, chart block that carries it on the web lecture,
#  title check EN, title check RO)
XBLOCKS = [('down-days', 9, 'ax', 'random-variables', 'Down Days in a Row', 'zile de scădere consecutive'),
           ('annual-moments', 19, 'ev', 'cdf-quantile', 'Annual Mean and Volatility', 'media și volatilitatea anuale'),
           ('chebyshev', 20, 'ev', 'cdf-quantile', 'Chebyshev', 'Cebîșev'),
           ('portfolio', 25, 'joint', 'joint', 'a 50/50 Portfolio', 'un portofoliu 50/50'),
           ('two-regimes', 31, 'cond', 'conditional', 'Two Regimes', 'două regimuri'),
           ('lln-table', 35, 'lln', 'lln', 'Two Limit Theorems', 'Două teoreme limită'),
           ('lcg', 39, 'mc', 'randu', 'an LCG with', 'un LCG cu'),
           ('inverse-exp', 41, 'mc', 'inverse-transform', 'The Inverse Transform Method', 'Metoda transformării inverse'),
           ('mc-error', 44, 'mc', 'monte-carlo', 'by Monte Carlo (2/2)', 'prin Monte Carlo (2/2)'),
           ('two-step-tree', 48, 'bin', 'binomial', 'a Two-Step Tree', 'un arbore cu doi pași'),
           ('crr-tree', 49, 'bin', 'binomial', 'Calibrating the Tree', 'Calibrarea arborelui'),
           ('ar1-sim', 56, 'proc', 'processes', 'The AR(1) Process: Simulation', 'Procesul AR(1): simulare'),
           ('gbm-median', 66, 'wien', 'gbm-paths', 'GBM: Mean, Median', 'GBM: media, mediana')]
ORDER = ['down-days', 'random-variables', 'cdf-quantile', 'annual-moments', 'chebyshev', 'joint', 'portfolio', 'uncorrelated-dependent',
         'two-regimes', 'conditional', 'lln-table', 'lln', 'lcg', 'randu', 'inverse-exp', 'inverse-transform', 'mc-error', 'monte-carlo',
         'two-step-tree', 'crr-tree', 'binomial', 'ar1-sim', 'processes', 'spot-the-real', 'wiener', 'gbm-median', 'gbm-paths', 'gbm-fan', 'gbm-check']
PFX.update({'down-days': 'dw', 'annual-moments': 'am', 'chebyshev': 'cb', 'portfolio': 'pf', 'two-regimes': 'rg', 'lln-table': 'lt', 'lcg': 'lc',
            'inverse-exp': 'ie', 'mc-error': 'me', 'two-step-tree': 'tt', 'crr-tree': 'cr', 'ar1-sim': 'ar', 'gbm-median': 'gm'})
SECTIONS['ro'].update(ax='De la jocuri de noroc la axiome', ev='Media și varianța')
SECTIONS['en'].update(ax='From Games of Chance to Axioms', ev='Expectation and Variance')

CSS = """<style>
.tw{overflow-x:auto;max-width:100%;margin:.5rem 0 .3rem;-webkit-overflow-scrolling:touch}
.xt{border-collapse:collapse;font-size:.84rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.xt th,.xt td{padding:.22rem .5rem;border-bottom:1px solid #E3E9F4;text-align:right;vertical-align:top}
.xt th:first-child,.xt td:first-child{text-align:left}
.xt thead th{border-bottom:2px solid #1F2A44;color:#1F2A44;font-weight:600}
.xt thead th[colspan]{text-align:center;border-bottom:1px solid #1F2A44}
.xt tr.sel td{background:#EEF3FB}
.tnote{font-size:.78rem;color:#1F2A44;margin:.25rem 0 0;max-width:46rem;white-space:normal}
.xh{font-weight:600;font-size:.92rem;color:#1F2A44;margin:.9rem 0 .1rem}
.we{border:1px solid #C5D2E8;border-left:4px solid #1A3A6E;border-radius:6px;padding:.55rem .8rem;margin:.8rem 0 .4rem;background:#F6F8FC}
.we-h{font-weight:600;margin:0 0 .2rem;color:#1A3A6E}
.we-f{overflow-x:auto;overflow-y:hidden;font-size:.95rem}
.we ol{margin:.3rem 0 0 1.3rem;padding:0;font-size:.92rem}.we li{margin:.18rem 0;overflow-wrap:anywhere}
.tw.veiled-chart,.we.veiled-chart{filter:blur(10px);opacity:.35;user-select:none}
.xhold{display:none}
</style>"""


def tw(i):
    return f'<div class="tw" id="{i}"></div>'


def xh(t):
    return f'<p class="xh">{t}</p>'


def we(i, title, tex):
    return f'<div class="we" id="{i}-box"><p class="we-h">{title}</p><div class="we-f">\\[{tex}\\]</div><ol id="{i}"></ol></div>'


WE_TEX = {
 'am': r'E\Big[\sum_{t=1}^{h} r_t\Big] = h\,\bar r, \qquad \mathrm{Var}\Big(\sum_{t=1}^{h} r_t\Big) = h\,\sigma^2 \;\Rightarrow\; \sigma_h = \sigma\sqrt{h}',
 'cb': r'P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}, \qquad P(|X - \mu| > k\sigma) = 2\Phi(-k) \;\text{ for } X \sim N(\mu, \sigma^2)',
 'pf': r'\begin{gathered}\sigma_p^2 = a^2\sigma_1^2 + b^2\sigma_2^2 + 2ab\,\mathrm{Cov}(X, Y), \qquad b = 1 - a,\\'
       r' \rho = \frac{\mathrm{Cov}(X, Y)}{\sigma_1\sigma_2}, \qquad a^* = \frac{\sigma_2^2 - \mathrm{Cov}(X, Y)}{\sigma_1^2 + \sigma_2^2 - 2\,\mathrm{Cov}(X, Y)}\end{gathered}',
 'rg': r'E[r] = \sum_j \pi_j \mu_j, \qquad \mathrm{Var}(r) = \underbrace{\sum_j \pi_j \sigma_j^2}_{E[\mathrm{Var}(r \mid X)]} + \underbrace{\sum_j \pi_j (\mu_j - E[r])^2}_{\mathrm{Var}(E[r \mid X])}',
 'lt': r'\bar r_n = \frac1n\sum_{t=1}^n r_t, \qquad \mathrm{SE} = \frac{s}{\sqrt n}, \qquad t = \frac{\bar r_n}{s/\sqrt n}, \qquad n_{t = 2} = \Big(\frac{2s}{\bar r}\Big)^2',
 'lc': r'x_{k+1} = (a\,x_k + c) \bmod M, \qquad u_k = \frac{x_k}{M}, \qquad M = 16',
 'ie': r'\begin{gathered}X = F^{-1}(U), \qquad U \sim U(0, 1):\\ x = -\frac{\ln(1 - u)}{\lambda} \;\text{(EXP)}, \qquad x = \mu + \sigma\,\Phi^{-1}(u) \;\text{(NRM)}, \qquad x = \hat F^{-1}(u) \;\text{(HIST)}\end{gathered}',
 'me': r'\begin{gathered}z = \frac{-L - h\,m}{s\sqrt h}, \qquad p = \Phi(z), \qquad \hat p_N = \frac{1}{N}\sum_{i=1}^N \mathbf 1\{R_i < -L\},\\'
       r' \mathrm{SE} = \sqrt{\frac{p(1 - p)}{N}}, \qquad N_{10\%} = \frac{1{.}96^2 (1 - p)}{0{.}1^2\,p}\end{gathered}',
 'tt': r'\begin{gathered}S_n = S_0\,u^K d^{\,n-K}, \qquad P(K = k) = \binom{n}{k} p^k (1 - p)^{n-k},\\ E[S_n] = S_0\big(pu + (1 - p)d\big)^n, \qquad p^* = \frac{1 - d}{u - d}\end{gathered}',
 'cr': r'\begin{gathered}u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac1u, \quad p = \frac{e^{\mu\Delta t} - d}{u - d}, \qquad S_T = S_0\,u^{2K - n},\\'
       r' P(S_T \le S_0) = \sum_{k \le n/2} \binom{n}{k} p^k (1 - p)^{n-k}, \qquad P(S_T < S_0) = \sum_{k < n/2} \binom{n}{k} p^k (1 - p)^{n-k}\end{gathered}',
 'ar': r'X_t = \phi X_{t-1} + \varepsilon_t: \qquad \mathrm{Var}(X_t) = \frac{\sigma^2}{1 - \phi^2}, \qquad \rho(h) = \phi^h, \qquad h_{1/2} = \frac{\ln 0{.}5}{\ln \phi}',
 'gm': r'\begin{gathered}E[S_T] = S_0 e^{\mu T}, \qquad \mathrm{Me}(S_T) = S_0 e^{(\mu - \sigma^2/2)T}, \qquad q_\alpha(S_T) = S_0 e^{(\mu - \sigma^2/2)T + \sigma\sqrt T z_\alpha},\\'
       r' P(S_T < S_0) = \Phi\Big(-\frac{(\mu - \sigma^2/2)\sqrt T}{\sigma}\Big)\end{gathered}',
}


def wtex(k, ro):
    t = WE_TEX[k].replace('{.}', '{,}' if ro else '.')
    lab = {'EXP': ('exponențială', 'exponential'), 'NRM': ('Normală', 'Normal'), 'HIST': ('istorică', 'historical')}
    for a, (r, e) in lab.items():
        t = t.replace(r'\text{(' + a + ')}', r'\text{(' + (r if ro else e) + ')}')
    return t.replace(r'\text{ for }', r'\text{ pentru }' if ro else r'\text{ for }')


ASSETS = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
PAIRS = [('sp500_dax', 'S&P 500 / DAX'), ('sp500_bet', 'S&P 500 / BET'), ('sp500_btc', 'S&P 500 / Bitcoin'),
         ('dax_bet', 'DAX / BET'), ('dax_btc', 'DAX / Bitcoin'), ('bet_btc', 'BET / Bitcoin')]


def widgets_all(lang):
    ro = lang == 'ro'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    wt = lambda k, r, e: we(k + '-we', f'{WE}: {r if ro else e}', wtex(k, ro))
    ser = lambda i, lab_ro, lab_en, d, opts=ASSETS: select(i, lab_ro if ro else lab_en, opts, d)
    W = widgets_old(lang)
    W['random-variables'] = (W['random-variables'][0], CSS + W['random-variables'][1])
    W['gbm-check'] = (W['gbm-check'][0], W['gbm-check'][1]
                      + xh('Zile cu |r − media| > k abateri standard: date, simulări GBM și distribuția Normală' if ro
                           else 'Days with |r − mean| > k standard deviations: data, GBM simulations and the Normal distribution')
                      + tw('gc-tab') + charts('gc-tail'))
    W.update({
     'down-days': (ser('dw-a', 'Seria (rîndul evidențiat)', 'Series (highlighted row)', 'sp500')
                   + slider('dw-x', 'Evenimentul D: randament sub −x%, pragul x' if ro else 'Event D: return below −x%, threshold x', 0, 3, 0.25, 0) + play('dw'),
                   xh('Probabilități condiționate pentru evenimentul D, randamente logaritmice zilnice' if ro
                      else 'Conditional probabilities of the event D, daily log returns') + tw('dw-tab')),
     'annual-moments': (ser('am-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                        + slider('am-h', 'Orizontul h (zile)' if ro else 'Horizon h (days)', 1, 365, 1, 252) + play('am'),
                        xh('Momentele randamentelor logaritmice zilnice (%)' if ro else 'Moments of the daily log returns (%)') + tw('am-tab')
                        + wt('am', 'media și volatilitatea pe h zile', 'the mean and volatility over h days')),
     'chebyshev': (ser('cb-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                   + slider('cb-k', 'Distanța k, în abateri standard' if ro else 'Distance k, in standard deviations', 1.5, 8, 0.5, 4) + play('cb'),
                   xh('Zile cu |r − media| > kσ: date, distribuția Normală și marginea lui Cebîșev' if ro
                      else 'Days with |r − mean| > kσ: data, the Normal distribution and Chebyshev\'s bound') + tw('cb-tab') + charts('cb-chart')
                   + wt('cb', 'marginea lui Cebîșev față de date', 'Chebyshev\'s bound against the data')),
     'portfolio': (ser('pf-p', 'Perechea (rîndul evidențiat, graficul și exemplul lucrat)', 'Pair (highlighted row, chart and worked example)', 'sp500_dax', PAIRS)
                   + slider('pf-w', 'Ponderea a a primei serii' if ro else 'Weight a of the first series', 0, 1, 0.05, 0.5) + play('pf'),
                   xh('Varianțe, covarianțe și corelații ale randamentelor zilnice (%)' if ro else 'Variances, covariances and correlations of daily returns (%)')
                   + tw('pf-mat') + xh('Perechi de serii pe zilele comune' if ro else 'Pairs of series on their common days') + tw('pf-tab') + charts('pf-chart')
                   + wt('pf', 'volatilitatea portofoliului a/b', 'the volatility of the a/b portfolio')),
     'two-regimes': (slider('rg-p', 'Probabilitatea regimului calm π' if ro else 'Probability of the calm regime π', 0.5, 0.95, 0.05, 0.7)
                     + slider('rg-s', 'Abaterea standard în regimul agitat (%)' if ro else 'Standard deviation in the turbulent regime (%)', 1, 4, 0.25, 2)
                     + ser('rg-a', 'Seria tabelului de date', 'Series of the data table', 'sp500') + play('rg'),
                     xh('Două regimuri: componentele varianței' if ro else 'Two regimes: the components of the variance') + tw('rg-tab')
                     + wt('rg', 'media și varianța totală', 'the mean and the total variance')
                     + xh('Datele: r<sub>t</sub> după chintila lui |r<sub>t−1</sub>|' if ro else 'The data: r<sub>t</sub> by the quintile of |r<sub>t−1</sub>|') + tw('rg-dat')),
     'lln-table': (slider('lt-y', 'Eșantionul începe în' if ro else 'Sample starts in', 2000, 2020, 1, 2000) + play('lt'),
                   xh('S&P 500: media cumulată după n zile' if ro else 'S&P 500: the running mean after n days') + tw('lt-tab')
                   + wt('lt', 'eroarea standard a mediei', 'the standard error of the mean')),
     'lcg': (slider('lc-a', 'Multiplicatorul a' if ro else 'Multiplier a', 0, 15, 1, 5) + slider('lc-c', 'Incrementul c' if ro else 'Increment c', 0, 15, 1, 1)
             + slider('lc-x', 'Sămînța x<sub>0</sub>' if ro else 'Seed x<sub>0</sub>', 0, 15, 1, 7) + play('lc'),
             xh('M = 16: perioada pentru fiecare multiplicator a, cu c și x<sub>0</sub> alese' if ro else 'M = 16: the period for every multiplier a, with the chosen c and x<sub>0</sub>')
             + tw('lc-tab') + wt('lc', 'pașii unui LCG și perioada lui', 'the steps of an LCG and its period')),
     'inverse-exp': (slider('ie-l', 'λ al distribuției exponențiale' if ro else 'λ of the exponential distribution', 0.2, 4, 0.1, 2)
                     + slider('ie-u', 'Extragerea uniformă u' if ro else 'Uniform draw u', 0.01, 0.99, 0.01, 0.9) + play('ie'),
                     wt('ie', 'același u prin trei funcții cuantilă', 'the same u through three quantile functions')
                     + xh('Coada stîngă a celor 200 000 de extrageri din curs (S&P 500)' if ro else 'The left tail of the 200,000 draws of the lecture (S&P 500)')
                     + tw('ie-tab') + charts('ie-chart')),
     'mc-error': (slider('me-L', 'Pierderea lunară L, în %' if ro else 'Monthly loss L, in %', 5, 20, 1, 10)
                  + select('me-n', 'Numărul de simulări N (rîndul evidențiat și exemplul lucrat)' if ro else 'Number of simulations N (highlighted row and worked example)',
                           [(str(n), f'{n:,}'.replace(',', ' ' if ro else ',')) for n in (100, 1000, 10000, 100000, 1000000)], '1000') + play('me'),
                  xh('Estimarea Monte Carlo și eroarea ei standard' if ro else 'The Monte Carlo estimate and its standard error') + tw('me-tab') + charts('me-chart')
                  + wt('me', 'probabilitatea exactă, estimarea și eroarea standard', 'the exact probability, the estimate and the standard error')),
     'two-step-tree': (slider('tt-u', 'Factorul de creștere u' if ro else 'Up factor u', 1.05, 1.5, 0.05, 1.2) + slider('tt-d', 'Factorul de scădere d' if ro else 'Down factor d', 0.5, 0.95, 0.05, 0.8)
                       + slider('tt-p', 'Probabilitatea p a unei creșteri' if ro else 'Probability p of an up move', 0.05, 0.95, 0.05, 0.6)
                       + slider('tt-n', 'Numărul de pași n' if ro else 'Number of steps n', 1, 6, 1, 2) + play('tt'),
                       xh('Nodurile finale ale arborelui, S<sub>0</sub> = 100 (rîndurile evidențiate: S<sub>n</sub> &lt; 100)' if ro
                          else 'The terminal nodes of the tree, S<sub>0</sub> = 100 (highlighted rows: S<sub>n</sub> &lt; 100)') + tw('tt-tab')
                       + wt('tt', 'media, probabilitatea unei pierderi și p*', 'the mean, the probability of a loss and p*')),
     'crr-tree': (select('cr-n', 'Numărul de pași n pe un an (rîndul evidențiat, graficul și exemplul lucrat)' if ro else 'Number of steps n in one year (highlighted row, chart and worked example)',
                         [(str(n), str(n)) for n in (1, 2, 3, 5, 10, 21, 63, 125, 126, 251, 252, 253)], '252') + play('cr'),
                  xh('Arborele CRR calibrat pe S&P 500 (σ = 19,26%, μ = 8,08%), S<sub>0</sub> = 100, T = 1 an' if ro
                     else 'The CRR tree calibrated to the S&P 500 (σ = 19.26%, μ = 8.08%), S<sub>0</sub> = 100, T = 1 year') + tw('cr-tab') + charts('cr-chart')
                  + wt('cr', 'P(S<sub>T</sub> ≤ 100) și P(S<sub>T</sub> &lt; 100)', 'P(S<sub>T</sub> ≤ 100) and P(S<sub>T</sub> &lt; 100)')),
     'ar1-sim': (slider('ar-phi', 'Persistența φ (rîndul evidențiat și exemplul lucrat)' if ro else 'Persistence φ (highlighted row and worked example)', 0, 0.99, 0.01, 0.5) + play('ar'),
                 xh('AR(1) cu σ = 1: valori teoretice, cele 500 de șocuri ale graficului și simularea lungă din curs' if ro
                    else 'AR(1) with σ = 1: theoretical values, the 500 shocks of the chart and the long simulation of the lecture') + tw('ar-tab')
                 + wt('ar', 'varianța, autocorelația și timpul de înjumătățire', 'the variance, the autocorrelation and the half-life')),
     'gbm-median': (ser('gm-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                    + slider('gm-T', 'Orizontul T (ani)' if ro else 'Horizon T (years)', 0.5, 30, 0.5, 10)
                    + select('gm-q', 'Nivelul α al cuantilei' if ro else 'Level α of the quantile', [('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%'), ('0.25', '25%')], '0.05') + play('gm'),
                    xh('GBM calibrat pe fiecare serie, S<sub>0</sub> = 100' if ro else 'GBM calibrated to each series, S<sub>0</sub> = 100') + tw('gm-tab')
                    + wt('gm', 'media, mediana și cuantila lui S<sub>T</sub>', 'the mean, the median and the quantile of S<sub>T</sub>')),
    })
    return W


QA['ro'].update({
 'down-days': ('Pentru S&P 500, P(scădere azi | scădere ieri) = 43,4% este chiar sub P(scădere) = 46,3%. Butonul ▶ ridică pragul x: evenimentul D devine o pierdere de peste x%. La x = 2%, cît este P(D | D ieri) față de P(D) = 4,27%?',
               '11,85%, de 2,8 ori P(D). Două pierderi de peste 2% la rînd au frecvența 0,51%, față de 0,18% sub independență. Semnul randamentului aproape nu persistă la S&P 500, dar pierderile mari vin grupate: volatility clustering. La BET persistă și semnul, la x = 0: 50,0% după o zi de scădere față de 42,6% după o zi fără scădere, efect al tranzacționării rare (<i>thin trading</i>).'),
 'annual-moments': ('Bitcoin se tranzacționează în fiecare zi, deci capitolul anualizează cu A = 365. Abaterea standard zilnică este 3,485%. Cît este volatilitatea anuală?',
                    '66,6% = 3,485 × √365. Cu A = 252, ca pentru acțiuni, ar fi 55,3%. Media anuală este 43,1% cu A = 365 (29,8% cu A = 252). Volatilitatea crește cu √A, media cu A, iar ambele presupun randamente i.i.d. Excesul de boltire 11,96 arată că distribuția zilnică nu este Normală, chiar dacă regula rădăcinii pătrate a timpului folosește doar varianța.'),
 'chebyshev': ('Pentru k = 2, inegalitatea lui Cebîșev permite pînă la 25% din zile. Ce pondere din zilele S&P 500 este la peste 2 abateri standard de medie?',
               '4,90% (329 de zile), aproape de 4,55% în distribuția Normală. La 2 abateri standard datele seamănă cu distribuția Normală; diferența apare în coadă: la k = 4 sînt 46 de zile (0,69%), de 108 ori mai mult decît 0,0063% în distribuția Normală, iar la k = 6 sînt 12 zile, față de 1,3 × 10<sup>−5</sup> zile așteptate. Marginea lui Cebîșev, 1/k², este valabilă pentru orice distribuție cu varianță finită, de aceea este mult peste date.'),
 'portfolio': ('Butonul ▶ mută ponderea a a S&P 500 în portofoliul S&P 500/DAX de la 0 la 1. La ce pondere este volatilitatea portofoliului cea mai mică?',
               'La a* = (σ<sub>2</sub>² − Cov)/(σ<sub>1</sub>² + σ<sub>2</sub>² − 2Cov) = 0,69, cu σ<sub>p</sub> = 1,163%, față de 1,186% la a = 0,5. Ponderea S&P 500 este mai mare pentru că volatilitatea lui este mai mică (1,221% față de 1,429%), dar nu 1: cu ρ = 0,60, o parte de DAX reduce încă riscul. Pentru S&P 500/BET, cu ρ = 0,21: a* = 0,60 și σ<sub>p</sub> = 1,025%.'),
 'two-regimes': ('În exemplul cu două regimuri, abaterea standard a randamentului este 1,286%. Cît devine dacă regimul agitat are abaterea standard 3% în loc de 2%?',
                 '1,776%: varianța în interiorul regimurilor crește la 0,7 × 0,8² + 0,3 × 3² = 3,148, iar componenta dintre regimuri rămîne 0,0047, deci Var(r) = 3,153. Media ponderată a abaterilor standard, 0,7 × 0,8 + 0,3 × 3 = 1,46%, subestimează riscul: se adună varianțe, nu abateri standard. În date, pentru S&P 500, componenta dintre chintile este 0,009% din varianța totală.'),
 'lln-table': ('Pe 2000–2026, media zilnică a S&P 500 are t = 1,67. Cu aceeași medie și volatilitate, cîți ani de date ar trebui pentru t = 2?',
               'Circa 38 de ani: n = (2s/r̄)² = (2 × 1,213/0,0247)² = 9 633 de zile. t crește cu √n, deci pentru un t de 2/1,67 = 1,20 ori mai mare trebuie de 1,43 ori mai multe zile. Pentru o eroare standard de două ori mai mică trebuie de patru ori mai multe zile (26 856). Media cumulată din 2000 este încă negativă după 100, 1 000 și 2 500 de zile.'),
 'lcg': ('Cu a = 4, c = 1 și x<sub>0</sub> = 7, după cîți pași rămîne șirul blocat și la ce valoare?',
         'După doi pași: x<sub>1</sub> = 29 mod 16 = 13, x<sub>2</sub> = 53 mod 16 = 5, apoi 4 × 5 + 1 = 21 ≡ 5, deci șirul rămîne 5 (perioada 1). Cu M = 16 și c impar, perioada maximă 16 apare exact cînd a − 1 este multiplu de 4 (a = 1, 5, 9, 13, condiția Hull–Dobell); a = 1 dă însă șirul 7, 8, 9, …, cu perioadă maximă și fără nimic aleator.'),
 'inverse-exp': ('Cu λ = 2, media distribuției exponențiale este 1/λ = 0,5. Pentru ce valori ale lui u depășește extragerea x = −ln(1 − u)/2 media?',
                 'Pentru u > 1 − e<sup>−1</sup> = 0,632: P(X > 0,5) = e<sup>−λ × 0,5</sup> = e<sup>−1</sup> = 36,8%. Mediana este ln 2/λ = 0,347, sub medie: distribuția exponențială este asimetrică la dreapta. În tabel, cele 7 extrageri Normale sub −5%, față de 3,4 așteptate, diferă cu 1,9 abateri standard Poisson (√3,44 = 1,85): zgomot de simulare, nu un defect al metodei.'),
 'mc-error': ('Pentru o pierdere lunară peste 20%, p = 0,0112%. Cît este eroarea standard a estimării Monte Carlo cu N = 1 000 de simulări?',
              '√(p(1 − p)/N) = 0,0334%, de trei ori p. Cu 1 000 de simulări estimarea este 0 (nicio lună simulată sub −20%); cu 100 000 este 0,007%, cu un milion 0,0117% (eroarea standard 0,0011%). Pentru L = 10%, eroarea standard la N = 1 000 este 0,533%, adică 18% din p = 2,922%.'),
 'two-step-tree': ('Cu S<sub>0</sub> = 100, u = 1,2, d = 0,8 și p = 0,6, P(S<sub>2</sub> < 100) = 0,64. Cît este P(S<sub>3</sub> < 100) cu trei pași?',
                   '0,352: nodurile sînt 172,8, 115,2, 76,8 și 51,2, iar sub 100 sînt cele cu cel mult o creștere: 0,4³ + 3 × 0,6 × 0,4² = 0,064 + 0,288. Cu doi pași, nodul ud = 96 era sub 100 pentru că ud = 0,96 < 1; cu trei pași, nodul cu două creșteri este 115,2. Media E[S<sub>3</sub>] = 100 × 1,04³ = 112,49 crește oricum cu n.'),
 'crr-tree': ('Cu 252 de pași, P(S<sub>T</sub> ≤ 100) = 39,7%. Cît este probabilitatea ca S<sub>T</sub> să fie strict sub 100?',
              '34,9%. Valoarea de 39,7% este P(S<sub>T</sub> ≤ 100): include nodul S<sub>T</sub> = 100 (k = 126 de creșteri), cu probabilitatea C(252, 126)p<sup>126</sup>(1 − p)<sup>126</sup> = 4,8%. Limita lognormală, 37,3%, este practic media celor două valori. Cu un număr impar de pași nu există nod în 100: 37,3% la 251 și la 253 de pași.'),
 'ar1-sim': ('Pentru φ = 0,5 timpul de înjumătățire al unui șoc este 1,0 perioade, pentru φ = 0,95 este 13,5. Cît este pentru φ = 0,9?',
             'ln 0,5/ln 0,9 = 6,6 perioade. Varianța staționară este 1/(1 − 0,81) = 5,26, iar ρ(5) = 0,9⁵ = 0,590. Pe cele 500 de șocuri ale graficului, varianța de selecție a procesului cu φ = 0,9 este 6,44: cu o persistență mare, 500 de puncte conțin puține perioade aproape independente, iar varianța de selecție se depărtează de valoarea teoretică.'),
 'gbm-median': ('GBM calibrat pe Bitcoin are σ = 66,6% pe an. De cîte ori este media lui S<sub>10</sub> mai mare decît mediana?',
                'De e<sup>σ²T/2</sup> = e<sup>0,666² × 5</sup> = 9,2 ori: media 68 252, mediana 7 437. Raportul dintre medie și mediană depinde doar de σ²T. Pentru S&P 500 (σ = 19,26%) raportul este 1,20 (224,4 față de 186,4). Media unui GBM este dominată de cîteva traiectorii foarte mari; o traiectorie tipică urmează mediana.'),
})
QA['en'].update({
 'down-days': ('For the S&P 500, P(down today | down yesterday) = 43.4% is even below P(down) = 46.3%. The ▶ button raises the threshold x: the event D becomes a loss of more than x%. At x = 2%, what is P(D | D yesterday) against P(D) = 4.27%?',
               '11.85%, 2.8 times P(D). Two losses of more than 2% in a row have a frequency of 0.51%, against 0.18% under independence. The sign of the return barely persists for the S&P 500, but large losses come in clusters: volatility clustering. For BET the sign persists too, at x = 0: 50.0% after a down day against 42.6% after a day that is not down, an effect of thin trading.'),
 'annual-moments': ('Bitcoin trades every day, so the chapter annualises with A = 365. The daily standard deviation is 3.485%. What is the annual volatility?',
                    '66.6% = 3.485 × √365. With A = 252, as for stocks, it would be 55.3%. The annual mean is 43.1% with A = 365 (29.8% with A = 252). The volatility grows with √A, the mean with A, and both assume i.i.d. returns. The excess kurtosis of 11.96 shows that the daily distribution is not Normal, even though the square-root-of-time rule uses only the variance.'),
 'chebyshev': ('For k = 2, Chebyshev\'s inequality allows up to 25% of the days. What share of the S&P 500 days lies more than 2 standard deviations from the mean?',
               '4.90% (329 days), close to 4.55% under the Normal distribution. At 2 standard deviations the data look like the Normal distribution; the difference is in the tail: at k = 4 there are 46 days (0.69%), 108 times the 0.0063% of the Normal distribution, and at k = 6 there are 12 days, against 1.3 × 10<sup>−5</sup> expected days. Chebyshev\'s bound, 1/k², holds for any distribution with finite variance, which is why it lies far above the data.'),
 'portfolio': ('The ▶ button moves the weight a of the S&P 500 in the S&P 500/DAX portfolio from 0 to 1. At which weight is the volatility of the portfolio lowest?',
               'At a* = (σ<sub>2</sub>² − Cov)/(σ<sub>1</sub>² + σ<sub>2</sub>² − 2Cov) = 0.69, with σ<sub>p</sub> = 1.163%, against 1.186% at a = 0.5. The S&P 500 gets the larger weight because its volatility is lower (1.221% against 1.429%), but not 1: with ρ = 0.60, some DAX still reduces the risk. For S&P 500/BET, with ρ = 0.21: a* = 0.60 and σ<sub>p</sub> = 1.025%.'),
 'two-regimes': ('In the two-regime example, the standard deviation of the return is 1.286%. What does it become if the turbulent regime has a standard deviation of 3% instead of 2%?',
                 '1.776%: the within-regime variance rises to 0.7 × 0.8² + 0.3 × 3² = 3.148, and the between-regime component stays at 0.0047, so Var(r) = 3.153. The weighted average of the standard deviations, 0.7 × 0.8 + 0.3 × 3 = 1.46%, understates the risk: variances add, not standard deviations. In the data, for the S&P 500, the between-quintile component is 0.009% of the total variance.'),
 'lln-table': ('Over 2000–2026, the daily mean of the S&P 500 has t = 1.67. With the same mean and volatility, how many years of data would give t = 2?',
               'About 38 years: n = (2s/r̄)² = (2 × 1.213/0.0247)² = 9,633 days. t grows with √n, so a t that is 2/1.67 = 1.20 times larger needs 1.43 times as many days. A standard error half as large needs four times as many days (26,856). The running mean from 2000 is still negative after 100, 1,000 and 2,500 days.'),
 'lcg': ('With a = 4, c = 1 and x<sub>0</sub> = 7, after how many steps does the sequence get stuck, and at which value?',
         'After two steps: x<sub>1</sub> = 29 mod 16 = 13, x<sub>2</sub> = 53 mod 16 = 5, then 4 × 5 + 1 = 21 ≡ 5, so the sequence stays at 5 (period 1). With M = 16 and an odd c, the maximal period 16 occurs exactly when a − 1 is a multiple of 4 (a = 1, 5, 9, 13, the Hull–Dobell condition); a = 1, however, gives the sequence 7, 8, 9, …, with a maximal period and nothing random about it.'),
 'inverse-exp': ('With λ = 2, the mean of the exponential distribution is 1/λ = 0.5. For which values of u does the draw x = −ln(1 − u)/2 exceed the mean?',
                 'For u > 1 − e<sup>−1</sup> = 0.632: P(X > 0.5) = e<sup>−λ × 0.5</sup> = e<sup>−1</sup> = 36.8%. The median is ln 2/λ = 0.347, below the mean: the exponential distribution is skewed to the right. In the table, the 7 Normal draws below −5%, against 3.4 expected, differ by 1.9 Poisson standard deviations (√3.44 = 1.85): simulation noise, not a flaw of the method.'),
 'mc-error': ('For a monthly loss above 20%, p = 0.0112%. What is the standard error of the Monte Carlo estimate with N = 1,000 simulations?',
              '√(p(1 − p)/N) = 0.0334%, three times p. With 1,000 simulations the estimate is 0 (no simulated month below −20%); with 100,000 it is 0.007%, with one million 0.0117% (standard error 0.0011%). For L = 10%, the standard error at N = 1,000 is 0.533%, i.e. 18% of p = 2.922%.'),
 'two-step-tree': ('With S<sub>0</sub> = 100, u = 1.2, d = 0.8 and p = 0.6, P(S<sub>2</sub> < 100) = 0.64. What is P(S<sub>3</sub> < 100) with three steps?',
                   '0.352: the nodes are 172.8, 115.2, 76.8 and 51.2, and below 100 are those with at most one up move: 0.4³ + 3 × 0.6 × 0.4² = 0.064 + 0.288. With two steps, the node ud = 96 was below 100 because ud = 0.96 < 1; with three steps, the node with two up moves is 115.2. The mean E[S<sub>3</sub>] = 100 × 1.04³ = 112.49 grows with n in any case.'),
 'crr-tree': ('With 252 steps, P(S<sub>T</sub> ≤ 100) = 39.7%. What is the probability that S<sub>T</sub> is strictly below 100?',
              '34.9%. The 39.7% is P(S<sub>T</sub> ≤ 100): it includes the node S<sub>T</sub> = 100 (k = 126 up moves), with probability C(252, 126)p<sup>126</sup>(1 − p)<sup>126</sup> = 4.8%. The lognormal limit, 37.3%, is practically the average of the two values. With an odd number of steps there is no node at 100: 37.3% with 251 and with 253 steps.'),
 'ar1-sim': ('For φ = 0.5 the half-life of a shock is 1.0 periods, for φ = 0.95 it is 13.5. What is it for φ = 0.9?',
             'ln 0.5/ln 0.9 = 6.6 periods. The stationary variance is 1/(1 − 0.81) = 5.26, and ρ(5) = 0.9⁵ = 0.590. On the 500 shocks of the chart, the sample variance of the process with φ = 0.9 is 6.44: with high persistence, 500 points contain few nearly independent stretches, and the sample variance moves away from the theoretical value.'),
 'gbm-median': ('GBM calibrated to Bitcoin has σ = 66.6% per year. How many times larger is the mean of S<sub>10</sub> than the median?',
                'e<sup>σ²T/2</sup> = e<sup>0.666² × 5</sup> = 9.2 times: mean 68,252, median 7,437. The ratio of the mean to the median depends only on σ²T. For the S&P 500 (σ = 19.26%) the ratio is 1.20 (224.4 against 186.4). The mean of a GBM is dominated by a few very large paths; a typical path follows the median.'),
})

PREDICT.update({
 'down-days': (0, [('Circa 11,9%, de 2,8 ori P(D): pierderile mari vin grupate.', 'About 11.9%, 2.8 times P(D): large losses come in clusters.'),
                   ('Circa 4,3%, ca P(D): ziua de ieri nu contează, ca la x = 0.', 'About 4.3%, as P(D): yesterday does not matter, as at x = 0.'),
                   ('Sub 4,27%: după o pierdere mare urmează de obicei o revenire.', 'Below 4.27%: a large loss is usually followed by a rebound.')]),
 'annual-moments': (1, [('Circa 55,3%: un an are 252 de zile de tranzacționare.', 'About 55.3%: a year has 252 trading days.'),
                        ('Circa 66,6% = 3,485 × √365.', 'About 66.6% = 3.485 × √365.'),
                        ('Circa 1 272% = 3,485 × 365.', 'About 1,272% = 3.485 × 365.')]),
 'chebyshev': (2, [('Circa 25%: cozile groase ating marginea lui Cebîșev.', 'About 25%: heavy tails reach Chebyshev\'s bound.'),
                   ('Circa 0,69%, ca la 4 abateri standard.', 'About 0.69%, as at 4 standard deviations.'),
                   ('Circa 4,9%, aproape de 4,55% în distribuția Normală.', 'About 4.9%, close to the 4.55% of the Normal distribution.')]),
 'portfolio': (1, [('La a = 0,5: ponderile egale dau mereu riscul minim.', 'At a = 0.5: equal weights always give the lowest risk.'),
                   ('La a ≈ 0,69: mai mult din seria mai puțin volatilă, dar nu totul.', 'At a ≈ 0.69: more of the less volatile series, but not all of it.'),
                   ('La a = 1: totul în seria cu volatilitatea mai mică.', 'At a = 1: everything in the series with the lower volatility.')]),
 'two-regimes': (0, [('Circa 1,78%: se adună varianțele ponderate.', 'About 1.78%: the weighted variances add up.'),
                     ('Circa 1,46%: media ponderată a abaterilor standard.', 'About 1.46%: the weighted average of the standard deviations.'),
                     ('Tot 1,29%: mediile regimurilor nu s-au schimbat.', 'Still 1.29%: the means of the regimes did not change.')]),
 'lln-table': (2, [('Circa 32 de ani: n crește proporțional cu t.', 'About 32 years: n grows in proportion to t.'),
                   ('Circa 27 de ani: datele de pînă acum ajung, t crește doar prin medie.', 'About 27 years: the data so far suffice, t grows only through the mean.'),
                   ('Circa 38 de ani: n crește cu pătratul lui t.', 'About 38 years: n grows with the square of t.')]),
 'lcg': (0, [('După doi pași, la 5: 7, 13, 5, 5, …', 'After two steps, at 5: 7, 13, 5, 5, …'),
             ('După un pas, la 13: un multiplicator par blochează imediat șirul.', 'After one step, at 13: an even multiplier sticks at once.'),
             ('Niciodată: cu c = 1 perioada este mereu 16.', 'Never: with c = 1 the period is always 16.')]),
 'inverse-exp': (1, [('Pentru u > 0,5: media și mediana coincid.', 'For u > 0.5: the mean and the median coincide.'),
                     ('Pentru u > 0,632 = 1 − e⁻¹.', 'For u > 0.632 = 1 − e⁻¹.'),
                     ('Pentru u > 0,393 = 1 − e^(−0,5): λ omis din exponent.', 'For u > 0.393 = 1 − e^(−0.5): λ left out of the exponent.')]),
 'mc-error': (0, [('Circa 0,033%, de trei ori p.', 'About 0.033%, three times p.'),
                  ('Circa 0,533%, ca pentru L = 10%: eroarea depinde doar de N.', 'About 0.533%, as for L = 10%: the error depends only on N.'),
                  ('Circa 0,0011%: eroarea este mică pentru că p este mic.', 'About 0.0011%: the error is small because p is small.')]),
 'two-step-tree': (2, [('Tot 0,64: probabilitatea unei pierderi nu depinde de n.', 'Still 0.64: the probability of a loss does not depend on n.'),
                       ('0,064: doar drumul ddd pierde.', '0.064: only the path ddd loses.'),
                       ('0,352: nodurile cu cel mult o creștere.', '0.352: the nodes with at most one up move.')]),
 'crr-tree': (1, [('39,7%: nodul S_T = 100 nu contează.', '39.7%: the node S_T = 100 does not matter.'),
                  ('34,9%: 39,7% include nodul S_T = 100, de 4,8%.', '34.9%: the 39.7% includes the node S_T = 100, of 4.8%.'),
                  ('37,3%: cu 252 de pași arborele dă limita lognormală.', '37.3%: with 252 steps the tree gives the lognormal limit.')]),
 'ar1-sim': (0, [('Circa 6,6 perioade.', 'About 6.6 periods.'),
                 ('10 perioade: 1/(1 − φ).', '10 periods: 1/(1 − φ).'),
                 ('Circa 5,3 perioade: 1/(1 − φ²), varianța.', 'About 5.3 periods: 1/(1 − φ²), the variance.')]),
 'gbm-median': (2, [('1: media și mediana coincid pentru randamente Normale.', '1: the mean and the median coincide for Normal returns.'),
                    ('Circa 1,25 = e^(σ²/2): raportul pe un an.', 'About 1.25 = e^(σ²/2): the ratio over one year.'),
                    ('Circa 9,2 = e^(σ²T/2).', 'About 9.2 = e^(σ²T/2).')]),
})

VEIL = {'down-days': {'out': True, 'charts': ['dw-tab']}, 'annual-moments': {'out': True, 'charts': ['am-tab']},
        'chebyshev': {'out': True, 'charts': ['cb-tab', 'cb-chart']}, 'portfolio': {'out': True, 'charts': ['pf-chart', 'pf-we-box']},
        'lln-table': {'out': True, 'charts': ['lt-we-box']}, 'inverse-exp': {'out': True, 'charts': ['ie-tab', 'ie-chart']}, 'mc-error': {'out': True, 'charts': ['me-tab']},
        'crr-tree': {'out': True, 'charts': ['cr-tab', 'cr-chart', 'cr-we-box']}, 'ar1-sim': {'out': True, 'charts': ['ar-tab']},
        'gbm-median': {'out': True, 'charts': ['gm-tab']}, 'gbm-check': {'out': True, 'charts': ['gc-tab', 'gc-tail']}}

TEXT_JS['ro'] += r"""window.ch4De = n => (n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? ' de' : '');
window.ch4D = s => window.ch4De(+String(s).replace(/[^0-9]/g, ''));
Object.assign(window.CH4_TEXT, {
 h: { series: 'Seria', notD: 'P(D | fără D<sub>t−1</sub>)', both: 'P(D<sub>t</sub> și D<sub>t−1</sub>)', meanD: 'media (% pe zi)', sdD: 'abaterea standard (% pe zi)',
   skew: 'asimetria', kurt: 'excesul de boltire', meanA: 'A × media (% pe an)', sdA: 'σ√A (% pe an)', days: k => `zile peste ${k}σ`, share: 'ponderea',
   normShare: '2Φ(−k)', normDays: 'zile așteptate (Normală)', cheb: '1/k²', ratioN: 'date / Normală', pair: 'Perechea', common: 'zile comune',
   regime: 'Regimul', prob: 'probabilitatea', meanP: 'media (%)', sdP: 'abaterea standard (%)', varP: 'varianța', within: 'π × varianța', between: 'π × (media − E[r])²',
   calm: 'calm', turb: 'agitat', total: 'total', group: 'Grupul', ndays: 'zile', nDays: 'n (zile)', until: 'pînă la', runMean: 'media (% pe zi)', ciA: 'IÎ 95% al mediei anuale',
   firstVals: 'primele valori x<sub>0</sub>, x<sub>1</sub>, …', tail: 'pași pînă la ciclu', period: 'perioada', hd: 'a − 1 multiplu de 4, c impar', yes: 'da', no: 'nu',
   thr: 'pragul', drawsN: 'extrageri Normale', expN: 'așteptate E', zN: '(extrageri − E)/√E', drawsH: 'extrageri istorice', shareH: 'ponderea istorică', shareD: 'ponderea în date',
   hits: 'luni sub −L', ci95: 'intervalul de 95%', covers: 'conține p', path: 'drumul', paths: 'drumuri C(n, K)', probK: 'probabilitatea', cum: 'P(S<sub>n</sub> ≤ nodul)',
   dt: 'Δt', pl: 'P(S<sub>T</sub> &lt; 100)', pe: 'P(S<sub>T</sub> = 100)', ple: 'P(S<sub>T</sub> ≤ 100)', q05: 'cuantila de 5%', lognormal: 'limita lognormală',
   hl: 'timpul de înjumătățire', var500: 'varianța (500 de șocuri)', rho500: 'ρ̂(5) (500 de șocuri)', varSim: 'varianța (simularea din curs)', rhoSim: 'ρ̂(5) (simularea din curs)',
   logDrift: 'μ − σ²/2 (% pe an)', sigma: 'σ (% pe an)', mu: 'μ (% pe an)', mean: 'E[S<sub>T</sub>]', median: 'mediana', qa: a => `cuantila de ${a}`, ploss: 'P(S<sub>T</sub> &lt; 100)',
   kk: 'k', data: 'date', gbmMed: 'GBM mediana', gbmMax: 'GBM maximul', normal: 'Normală n·2Φ(−k)' },
 dwNote: x => (+x.replace(',', '.') === 0 ? 'D: zi de scădere, r<sub>t</sub> &lt; 0.' : `D: pierdere de peste ${x}%, r<sub>t</sub> &lt; −${x}%.`) + ' Sub independență, P(D | D<sub>t−1</sub>) = P(D) și P(D<sub>t</sub> și D<sub>t−1</sub>) = P(D)².',
 dw: (a, z, x, p, pc, pcu, r, pb, pi) => `${a}, ${z ? 'zile de scădere' : `pierderi de peste ${x}%`}: P(D) = ${p}; după o astfel de zi ${pc} (de ${r} ori P(D)), altfel ${pcu}. Două la rînd: ${pb}, față de P(D)² = ${pi} sub independență.`,
 amNote: 'A: zile pe an folosite la anualizare (252 pentru acțiuni, 365 pentru Bitcoin). Asimetria și excesul de boltire sînt momentele de selecție (fără corecție).',
 amw1: (a, n, m, s) => `${a}, ${n}${window.ch4D(n)} zile: media zilnică r̄ = ${m}%, abaterea standard σ = ${s}%.`,
 amw2: (h, m, E) => `Media sumei pe h = ${h}${window.ch4De(h)} zile (liniaritatea mediei): h × r̄ = ${h} × ${m} = ${E}%.`,
 amw3: (s, h, sq, sh) => `Covarianțele sînt zero (i.i.d.), deci varianța este hσ² și σ<sub>h</sub> = σ√h = ${s} × ${sq} = ${sh}%.`,
 amw4: (h, E, sh, r) => `Raportul medie/abatere standard pe ${h}${window.ch4De(h)} zile: ${E}/${sh} = √h × r̄/σ = ${r}; crește doar cu √h.`,
 am: (a, h, E, sh, r) => `${a}, h = ${h}${window.ch4De(h)} zile: media sumei ${E}%, abaterea standard ${sh}%, raportul ${r}.`,
 cbNote: 'σ: abaterea standard de selecție a fiecărei serii; distribuția Normală are aceeași medie și abatere standard.',
 cbw1: (k, b, d, n, a) => `Cebîșev: P(|r − media| ≥ ${k}σ) ≤ 1/${k}² = ${b}, adică cel mult ${d} din cele ${n}${window.ch4D(n)} zile ale seriei ${a}.`,
 cbw2: (k, p, e) => `Distribuția Normală: 2Φ(−${k}) = ${p}, adică ${e} zile așteptate.`,
 cbw3: (c, s, rn, rc) => `Datele: ${c}${window.ch4D(c)} ${c === '1' ? 'zi' : 'zile'} (${s}), ${rn === null ? 'în timp ce distribuția Normală nu dă practic nicio zi' : `de ${rn} ori valoarea Normală`}, și ${rc} din marginea lui Cebîșev.`,
 cb: (a, k, c, s, p) => `${a}, k = ${k}: ${c}${window.ch4D(c)} ${c === '1' ? 'zi' : 'zile'} (${s}) peste ${k} abateri standard; distribuția Normală ${p}.`,
 pfMatNote: 'Diagonala: varianța pe întregul eșantion al seriei. Deasupra diagonalei: covarianța, sub diagonală: corelația, ambele pe zilele comune ale perechii.',
 pfNote: 'σ în % pe zi; aσ<sub>1</sub> + bσ<sub>2</sub> este volatilitatea portofoliului pentru ρ = 1.',
 pfw1: (u, v, n, s1, s2, c, r) => `${u} / ${v}, ${n}${window.ch4D(n)} zile comune: σ<sub>1</sub> = ${s1}%, σ<sub>2</sub> = ${s2}%, Cov = ${c}; ρ = ${c}/(${s1} × ${s2}) = ${r}.`,
 pfw2: (a, b, a2, v1, b2, v2, vs, ab, c, ct) => `a = ${a}, b = 1 − a = ${b}: a²σ<sub>1</sub>² + b²σ<sub>2</sub>² = ${a2} × ${v1} + ${b2} × ${v2} = ${vs}; 2ab Cov = ${ab} × ${c} = ${ct}.`,
 pfw3: (vs, ct, vp, sp, avg) => `σ<sub>p</sub>² = ${vs} + ${ct} = ${vp}, σ<sub>p</sub> = ${sp}%, față de aσ<sub>1</sub> + bσ<sub>2</sub> = ${avg}% (valoarea pentru ρ = 1).`,
 pfw4: (as, sm) => `Ponderea de varianță minimă: a* = ${as}, cu σ<sub>p</sub> = ${sm}%.`,
 pf: (u, v, a, sp, avg, r) => `${u} / ${v}, a = ${a}: σ<sub>p</sub> = ${sp}% față de ${avg}% pentru ρ = 1; reducerea riscului ${r}.`,
 rgw1: (p1, m1, p2, m2, E) => `Media (proprietatea turnului): E[r] = ${p1} × ${m1} + ${p2} × ${m2} = ${E}%.`,
 rgw2: (p1, s1, p2, s2, W) => `În interiorul regimurilor: E[Var(r | X)] = ${p1} × ${s1}² + ${p2} × ${s2}² = ${W}.`,
 rgw3: (p1, d1, p2, d2, B) => `Între regimuri: Var(E[r | X]) = ${p1} × (${d1})² + ${p2} × (${d2})² = ${B}.`,
 rgw4: (W, B, V, s, sh) => `Var(r) = ${W} + ${B} = ${V}, abaterea standard ${s}%; componenta dintre regimuri: ${sh} din varianță.`,
 rgDat: (a, t, w, b, s) => `${a}: varianța totală ${t} = ${w} (în interiorul grupurilor) + ${b} (între grupuri, ${s}% din total). Grupurile sînt chintilele lui |r<sub>t−1</sub>|.`,
 rg: (s, sh) => `Două regimuri: abaterea standard a randamentului ${s}%, componenta dintre regimuri ${sh} din varianță.`,
 ltNote: y => `Randamente logaritmice zilnice ale S&P 500 din ${y}; SE = s<sub>n</sub>/√n; intervalul anual = 252 × (r̄<sub>n</sub> ± 1,96 SE).`,
 ltw1: (y, n, m, s) => `S&P 500 din ${y}: n = ${n}${window.ch4D(n)} zile, r̄ = ${m}%, s = ${s}%.`,
 ltw2: (s, n, se, m, t) => `SE = s/√n = ${s}/√${n} = ${se}%; t = r̄/SE = ${m}/${se} = ${t}.`,
 ltw3: (m, se, lo, hi) => `Media anuală: intervalul de 95% 252 × (r̄ ± 1,96 SE) = [${lo}%; ${hi}%].`,
 ltw4: (s, m, n, y) => n === null ? 'Media este negativă sau zero pe acest eșantion: nicio lungime nu dă t = 2 la această medie.' : `t = 2 ar cere n = (2s/r̄)² = (2 × ${s}/${m})² = ${n}${window.ch4D(n)} zile, adică ${y} ani.`,
 lt: (y, n, m, t) => `S&P 500 din ${y}: după ${n}${window.ch4D(n)} zile media este ${m}% pe zi, t = ${t}.`,
 lcw1: (a, x, c, v, q, r) => `x<sub>1</sub> = (${a} × ${x} + ${c}) mod 16 = ${v} mod 16 = ${r} (${v} = ${q} × 16 + ${r}).`,
 lcw2: (a, x, c, v, r) => `x<sub>2</sub> = (${a} × ${x} + ${c}) mod 16 = ${v} mod 16 = ${r}.`,
 lcw3: (seq, t, p) => `Șirul: ${seq}. ` + (t === 0 ? `Revine la x<sub>0</sub> după ${p} pași: perioada ${p}${p === 16 ? ', maximul posibil M' : ''}.` : `După ${t} ${t === 1 ? 'pas' : 'pași'} intră într-un ciclu de lungime ${p}.`),
 lcw4: us => `Numerele uniforme u<sub>k</sub> = x<sub>k</sub>/16: ${us}, …`,
 lc: (n, l) => `Cu c și x<sub>0</sub> alese, ${n} din cele 16 valori ale lui a dau perioada maximă 16: ${l}.`,
 lcNone: 'Cu c și x<sub>0</sub> alese, niciun multiplicator a nu dă perioada maximă 16.',
 ieNote: 'Extragerile din curs: 200 000 de valori U, transformate prin funcția cuantilă Normală (aceeași medie și abatere standard ca S&P 500) și prin cea empirică. E = 200 000 × Φ((−x − media)/σ); √E: abaterea standard Poisson a numărului de extrageri.',
 iew1: (u, l, a, x) => `Exponențială: x = −ln(1 − ${u})/${l} = ${a}/${l} = ${x}.`,
 iew2: (l, x, u) => `Verificare: F(x) = 1 − e<sup>−${l} × ${x}</sup> = ${u}; media 1/λ, mediana ln 2/λ.`,
 iew3: (m, s, z, x) => `Normală (S&P 500): x = ${m} + ${s} × Φ<sup>−1</sup>(u) = ${m} + ${s} × ${z} = ${x}%.`,
 iew4: (x, d) => `Istorică: x = F̂<sup>−1</sup>(u) = ${x}%, cuantila empirică a celor ${d}${window.ch4D(d)} randamente zilnice ale S&P 500.`,
 ie: (c, e, s) => `Sub −5%: ${c} extrageri Normale față de ${e} așteptate (abaterea standard Poisson ${s}).`,
 meNote: 'Lunile simulate din curs: un milion de extrageri N(21m, 21s²) cu m și s ale S&P 500; p̂<sub>N</sub> folosește primele N. SE = √(p(1 − p)/N) cu p exact.',
 mew1: (m, mu, s, sd, L, z, p) => `21 de zile: media 21 × ${m} = ${mu}, abaterea standard ${s} × √21 = ${sd}; z = (−${L} − ${mu})/${sd} = ${z}, p = Φ(z) = ${p}.`,
 mew2: (N, k, ph) => `N = ${N}: ${k}${window.ch4D(k)} ${k === '1' ? 'lună simulată' : 'luni simulate'} sub −L%, p̂<sub>N</sub> = ${k}/${N} = ${ph}.`,
 mew3: (p, N, se, d) => `SE = √(${p}(1 − ${p})/${N}) = ${se}; |p̂<sub>N</sub> − p| = ${d} erori standard.`,
 mew4: (n) => `Pentru o eroare relativă de 10% la 95%: N = 1,96²(1 − p)/(0,1² p) = ${n}.`,
 me: (L, N, ph, p, se) => `L = ${L}%, N = ${N}: p̂ = ${ph}, p = ${p}, SE = ${se}.`,
 ttw1: (list) => `Nodurile finale: ${list}.`,
 ttw2: (p, u, q, d, g, n, e) => `E[S<sub>n</sub>] = 100 × (${p} × ${u} + ${q} × ${d})<sup>${n}</sup> = 100 × ${g}<sup>${n}</sup> = ${e}.`,
 ttw3: (lt, le, at) => `P(S<sub>n</sub> &lt; 100) = ${lt}; P(S<sub>n</sub> ≤ 100) = ${le}` + (at ? ` (nodul S<sub>n</sub> = 100 are ${at}).` : ' (niciun nod în 100).'),
 ttw4: (d, u, ps) => `p* = (1 − ${d})/(${u} − ${d}) = ${ps}: cu p = p* prețul este un joc echitabil, E[S<sub>t+1</sub> | S<sub>t</sub>] = S<sub>t</sub>.`,
 tt: (n, e, lt) => `n = ${n}: E[S<sub>n</sub>] = ${e}, P(S<sub>n</sub> &lt; 100) = ${lt}.`,
 crNote: 'Ultimul rînd: limita lognormală, ln S<sub>T</sub> ~ N(ln 100 + μ − σ²/2, σ²).',
 crw1: (n, s, u, d, p) => `n = ${n}, Δt = 1/${n}: u = e<sup>${s}√(1/${n})</sup> = ${u}, d = 1/u = ${d}, p = (e<sup>μΔt</sup> − d)/(u − d) = ${p}.`,
 crw2: (n, h) => `S<sub>T</sub> = 100u<sup>2K − ${n}</sup>, deci S<sub>T</sub> ≤ 100 exact cînd K ≤ ${n}/2 = ${h}.`,
 crw3e: (n, h, le, eq, lt) => `P(S<sub>T</sub> ≤ 100) = Σ<sub>k=0</sub><sup>${h}</sup> C(${n}, k)p<sup>k</sup>(1 − p)<sup>${n}−k</sup> = ${le}; nodul k = ${h}: C(${n}, ${h})p<sup>${h}</sup>(1 − p)<sup>${h}</sup> = ${eq}; P(S<sub>T</sub> &lt; 100) = ${le} − ${eq} = ${lt}.`,
 crw3o: (n, h, lt) => `n impar, niciun nod în 100: P(S<sub>T</sub> &lt; 100) = P(S<sub>T</sub> ≤ 100) = Σ<sub>k=0</sub><sup>${h}</sup> C(${n}, k)p<sup>k</sup>(1 − p)<sup>${n}−k</sup> = ${lt}.`,
 crw4: (ml, s, z, ln, avg) => `Limita lognormală: Φ(−(μ − σ²/2)/σ) = Φ(−${ml}/${s}) = Φ(${z}) = ${ln}` + (avg ? `; media dintre P(&lt;) și P(≤) este ${avg}.` : '.'),
 cr: (n, lt, le, ln) => `n = ${n}: P(S<sub>T</sub> &lt; 100) = ${lt}, P(S<sub>T</sub> ≤ 100) = ${le}; limita lognormală ${ln}.`,
 arNote: 'Coloanele „500 de șocuri” folosesc șocurile graficului de mai sus (aceeași sămînță); simularea lungă din curs este dată pentru φ = 0,5 și φ = 0,95.',
 arw1: (f, v) => `Varianța staționară: σ²/(1 − φ²) = 1/(1 − ${f}²) = ${v}.`,
 arw2: (f, r1, r5) => `Autocorelațiile: ρ(1) = ${f}, ρ(5) = ${f}<sup>5</sup> = ${r5}.`,
 arw3: (f, h) => h === null ? 'φ = 0: un șoc dispare imediat (zgomot alb).' : `Timpul de înjumătățire: ln 0,5/ln ${f} = ${h} perioade.`,
 arw4: (v, r) => `Pe cele 500 de șocuri ale graficului: varianța de selecție ${v}, ρ̂(5) = ${r}.`,
 ar: (f, v, h) => `φ = ${f}: varianța ${v}, timpul de înjumătățire ${h} perioade.`,
 gmNote: 'Calibrarea capitolului: μ − σ²/2 = A × media zilnică, σ = abaterea standard zilnică × √A, cu A = 252 (365 pentru Bitcoin).',
 gmw1: (a, ml, s, mu, mp) => `${a}: μ − σ²/2 = ${ml}, σ = ${s}, deci μ = ${ml} + ${s}²/2 = ${mu} pe an (${mp}).`,
 gmw2: (mu, T, e, ml, md) => `E[S<sub>T</sub>] = 100e<sup>${mu} × ${T}</sup> = ${e}; mediana = 100e<sup>${ml} × ${T}</sup> = ${md}.`,
 gmw3: (a, z, ml, T, s, q) => `Cuantila de ${a}: z<sub>α</sub> = ${z}; 100 exp(${ml} × ${T} ${z.startsWith('−') ? '−' : '+'} ${s} × √${T} × ${z.replace('−', '')}) = ${q}.`,
 gmw4: (ml, T, s, z, p) => `P(S<sub>T</sub> &lt; 100) = Φ(−${ml} × √${T}/${s}) = Φ(${z}) = ${p}.`,
 gm: (a, T, e, md, r) => `${a}, T = ${T} ani: media ${e}, mediana ${md}, raportul ${r} = e<sup>σ²T/2</sup>.`,
 gcNote: n => `Mediana și maximul din cele 300 de simulări GBM ale fiecărei serii; ultima coloană: numărul așteptat în distribuția Normală pentru cele ${n}${window.ch4D(n)} zile ale S&P 500.`
});
"""
TEXT_JS['en'] += r"""Object.assign(window.CH4_TEXT, {
 h: { series: 'Series', notD: 'P(D | not D<sub>t−1</sub>)', both: 'P(D<sub>t</sub> and D<sub>t−1</sub>)', meanD: 'mean (% a day)', sdD: 'standard deviation (% a day)',
   skew: 'skewness', kurt: 'excess kurtosis', meanA: 'A × mean (% a year)', sdA: 'σ√A (% a year)', days: k => `days beyond ${k}σ`, share: 'share',
   normShare: '2Φ(−k)', normDays: 'expected days (Normal)', cheb: '1/k²', ratioN: 'data / Normal', pair: 'Pair', common: 'common days',
   regime: 'Regime', prob: 'probability', meanP: 'mean (%)', sdP: 'standard deviation (%)', varP: 'variance', within: 'π × variance', between: 'π × (mean − E[r])²',
   calm: 'calm', turb: 'turbulent', total: 'total', group: 'Group', ndays: 'days', nDays: 'n (days)', until: 'until', runMean: 'mean (% a day)', ciA: '95% CI of the annual mean',
   firstVals: 'first values x<sub>0</sub>, x<sub>1</sub>, …', tail: 'steps to the cycle', period: 'period', hd: 'a − 1 multiple of 4, c odd', yes: 'yes', no: 'no',
   thr: 'threshold', drawsN: 'Normal draws', expN: 'expected E', zN: '(draws − E)/√E', drawsH: 'historical draws', shareH: 'historical share', shareD: 'share in the data',
   hits: 'months below −L', ci95: '95% interval', covers: 'contains p', path: 'path', paths: 'paths C(n, K)', probK: 'probability', cum: 'P(S<sub>n</sub> ≤ node)',
   dt: 'Δt', pl: 'P(S<sub>T</sub> &lt; 100)', pe: 'P(S<sub>T</sub> = 100)', ple: 'P(S<sub>T</sub> ≤ 100)', q05: '5% quantile', lognormal: 'lognormal limit',
   hl: 'half-life', var500: 'variance (500 shocks)', rho500: 'ρ̂(5) (500 shocks)', varSim: 'variance (lecture simulation)', rhoSim: 'ρ̂(5) (lecture simulation)',
   logDrift: 'μ − σ²/2 (% a year)', sigma: 'σ (% a year)', mu: 'μ (% a year)', mean: 'E[S<sub>T</sub>]', median: 'median', qa: a => `${a} quantile`, ploss: 'P(S<sub>T</sub> &lt; 100)',
   kk: 'k', data: 'data', gbmMed: 'GBM median', gbmMax: 'GBM maximum', normal: 'Normal n·2Φ(−k)' },
 dwNote: x => (+x === 0 ? 'D: a down day, r<sub>t</sub> &lt; 0.' : `D: a loss of more than ${x}%, r<sub>t</sub> &lt; −${x}%.`) + ' Under independence, P(D | D<sub>t−1</sub>) = P(D) and P(D<sub>t</sub> and D<sub>t−1</sub>) = P(D)².',
 dw: (a, z, x, p, pc, pcu, r, pb, pi) => `${a}, ${z ? 'down days' : `losses of more than ${x}%`}: P(D) = ${p}; after such a day ${pc} (${r} times P(D)), otherwise ${pcu}. Two in a row: ${pb}, against P(D)² = ${pi} under independence.`,
 amNote: 'A: days per year used to annualise (252 for stocks, 365 for Bitcoin). Skewness and excess kurtosis are the sample moments (no bias correction).',
 amw1: (a, n, m, s) => `${a}, ${n} days: daily mean r̄ = ${m}%, standard deviation σ = ${s}%.`,
 amw2: (h, m, E) => `Mean of the sum over h = ${h} days (linearity of expectation): h × r̄ = ${h} × ${m} = ${E}%.`,
 amw3: (s, h, sq, sh) => `The covariances are zero (i.i.d.), so the variance is hσ² and σ<sub>h</sub> = σ√h = ${s} × ${sq} = ${sh}%.`,
 amw4: (h, E, sh, r) => `Ratio of mean to standard deviation over ${h} days: ${E}/${sh} = √h × r̄/σ = ${r}; it grows only with √h.`,
 am: (a, h, E, sh, r) => `${a}, h = ${h} days: mean of the sum ${E}%, standard deviation ${sh}%, ratio ${r}.`,
 cbNote: 'σ: the sample standard deviation of each series; the Normal distribution has the same mean and standard deviation.',
 cbw1: (k, b, d, n, a) => `Chebyshev: P(|r − mean| ≥ ${k}σ) ≤ 1/${k}² = ${b}, i.e. at most ${d} of the ${n} days of ${a}.`,
 cbw2: (k, p, e) => `Normal distribution: 2Φ(−${k}) = ${p}, i.e. ${e} expected days.`,
 cbw3: (c, s, rn, rc) => `Data: ${c} days (${s}), ${rn === null ? 'while the Normal distribution gives practically no day' : `${rn} times the Normal value`}, and ${rc} of Chebyshev's bound.`,
 cb: (a, k, c, s, p) => `${a}, k = ${k}: ${c} days (${s}) beyond ${k} standard deviations; Normal distribution ${p}.`,
 pfMatNote: 'Diagonal: the variance of the series over its whole sample. Above the diagonal: the covariance, below: the correlation, both on the common days of the pair.',
 pfNote: 'σ in % a day; aσ<sub>1</sub> + bσ<sub>2</sub> is the volatility of the portfolio for ρ = 1.',
 pfw1: (u, v, n, s1, s2, c, r) => `${u} / ${v}, ${n} common days: σ<sub>1</sub> = ${s1}%, σ<sub>2</sub> = ${s2}%, Cov = ${c}; ρ = ${c}/(${s1} × ${s2}) = ${r}.`,
 pfw2: (a, b, a2, v1, b2, v2, vs, ab, c, ct) => `a = ${a}, b = 1 − a = ${b}: a²σ<sub>1</sub>² + b²σ<sub>2</sub>² = ${a2} × ${v1} + ${b2} × ${v2} = ${vs}; 2ab Cov = ${ab} × ${c} = ${ct}.`,
 pfw3: (vs, ct, vp, sp, avg) => `σ<sub>p</sub>² = ${vs} + ${ct} = ${vp}, σ<sub>p</sub> = ${sp}%, against aσ<sub>1</sub> + bσ<sub>2</sub> = ${avg}% (the value for ρ = 1).`,
 pfw4: (as, sm) => `Minimum-variance weight: a* = ${as}, with σ<sub>p</sub> = ${sm}%.`,
 pf: (u, v, a, sp, avg, r) => `${u} / ${v}, a = ${a}: σ<sub>p</sub> = ${sp}% against ${avg}% for ρ = 1; risk reduction ${r}.`,
 rgw1: (p1, m1, p2, m2, E) => `Mean (tower property): E[r] = ${p1} × ${m1} + ${p2} × ${m2} = ${E}%.`,
 rgw2: (p1, s1, p2, s2, W) => `Within the regimes: E[Var(r | X)] = ${p1} × ${s1}² + ${p2} × ${s2}² = ${W}.`,
 rgw3: (p1, d1, p2, d2, B) => `Between the regimes: Var(E[r | X]) = ${p1} × (${d1})² + ${p2} × (${d2})² = ${B}.`,
 rgw4: (W, B, V, s, sh) => `Var(r) = ${W} + ${B} = ${V}, standard deviation ${s}%; the between-regime component: ${sh} of the variance.`,
 rgDat: (a, t, w, b, s) => `${a}: total variance ${t} = ${w} (within groups) + ${b} (between groups, ${s}% of the total). The groups are the quintiles of |r<sub>t−1</sub>|.`,
 rg: (s, sh) => `Two regimes: standard deviation of the return ${s}%, between-regime component ${sh} of the variance.`,
 ltNote: y => `Daily log returns of the S&P 500 from ${y}; SE = s<sub>n</sub>/√n; annual interval = 252 × (r̄<sub>n</sub> ± 1.96 SE).`,
 ltw1: (y, n, m, s) => `S&P 500 from ${y}: n = ${n} days, r̄ = ${m}%, s = ${s}%.`,
 ltw2: (s, n, se, m, t) => `SE = s/√n = ${s}/√${n} = ${se}%; t = r̄/SE = ${m}/${se} = ${t}.`,
 ltw3: (m, se, lo, hi) => `Annual mean: 95% interval 252 × (r̄ ± 1.96 SE) = [${lo}%, ${hi}%].`,
 ltw4: (s, m, n, y) => n === null ? 'The mean is negative or zero on this sample: no length gives t = 2 at this mean.' : `t = 2 would need n = (2s/r̄)² = (2 × ${s}/${m})² = ${n} days, i.e. ${y} years.`,
 lt: (y, n, m, t) => `S&P 500 from ${y}: after ${n} days the mean is ${m}% a day, t = ${t}.`,
 lcw1: (a, x, c, v, q, r) => `x<sub>1</sub> = (${a} × ${x} + ${c}) mod 16 = ${v} mod 16 = ${r} (${v} = ${q} × 16 + ${r}).`,
 lcw2: (a, x, c, v, r) => `x<sub>2</sub> = (${a} × ${x} + ${c}) mod 16 = ${v} mod 16 = ${r}.`,
 lcw3: (seq, t, p) => `The sequence: ${seq}. ` + (t === 0 ? `It returns to x<sub>0</sub> after ${p} steps: period ${p}${p === 16 ? ', the maximum possible, M' : ''}.` : `After ${t} ${t === 1 ? 'step' : 'steps'} it enters a cycle of length ${p}.`),
 lcw4: us => `The uniform numbers u<sub>k</sub> = x<sub>k</sub>/16: ${us}, …`,
 lc: (n, l) => `With the chosen c and x<sub>0</sub>, ${n} of the 16 values of a give the maximal period 16: ${l}.`,
 lcNone: 'With the chosen c and x<sub>0</sub>, no multiplier a gives the maximal period 16.',
 ieNote: 'The draws of the lecture: 200,000 values U, transformed by the Normal quantile function (the same mean and standard deviation as the S&P 500) and by the empirical one. E = 200,000 × Φ((−x − mean)/σ); √E: the Poisson standard deviation of the number of draws.',
 iew1: (u, l, a, x) => `Exponential: x = −ln(1 − ${u})/${l} = ${a}/${l} = ${x}.`,
 iew2: (l, x, u) => `Check: F(x) = 1 − e<sup>−${l} × ${x}</sup> = ${u}; mean 1/λ, median ln 2/λ.`,
 iew3: (m, s, z, x) => `Normal (S&P 500): x = ${m} + ${s} × Φ<sup>−1</sup>(u) = ${m} + ${s} × ${z} = ${x}%.`,
 iew4: (x, d) => `Historical: x = F̂<sup>−1</sup>(u) = ${x}%, the empirical quantile of the ${d} daily returns of the S&P 500.`,
 ie: (c, e, s) => `Below −5%: ${c} Normal draws against ${e} expected (Poisson standard deviation ${s}).`,
 meNote: 'The simulated months of the lecture: one million draws N(21m, 21s²) with the m and s of the S&P 500; p̂<sub>N</sub> uses the first N. SE = √(p(1 − p)/N) with the exact p.',
 mew1: (m, mu, s, sd, L, z, p) => `21 days: mean 21 × ${m} = ${mu}, standard deviation ${s} × √21 = ${sd}; z = (−${L} − ${mu})/${sd} = ${z}, p = Φ(z) = ${p}.`,
 mew2: (N, k, ph) => `N = ${N}: ${k} simulated months below −L%, p̂<sub>N</sub> = ${k}/${N} = ${ph}.`,
 mew3: (p, N, se, d) => `SE = √(${p}(1 − ${p})/${N}) = ${se}; |p̂<sub>N</sub> − p| = ${d} standard errors.`,
 mew4: (n) => `For a relative error of 10% at 95%: N = 1.96²(1 − p)/(0.1² p) = ${n}.`,
 me: (L, N, ph, p, se) => `L = ${L}%, N = ${N}: p̂ = ${ph}, p = ${p}, SE = ${se}.`,
 ttw1: (list) => `The terminal nodes: ${list}.`,
 ttw2: (p, u, q, d, g, n, e) => `E[S<sub>n</sub>] = 100 × (${p} × ${u} + ${q} × ${d})<sup>${n}</sup> = 100 × ${g}<sup>${n}</sup> = ${e}.`,
 ttw3: (lt, le, at) => `P(S<sub>n</sub> &lt; 100) = ${lt}; P(S<sub>n</sub> ≤ 100) = ${le}` + (at ? ` (the node S<sub>n</sub> = 100 has ${at}).` : ' (no node at 100).'),
 ttw4: (d, u, ps) => `p* = (1 − ${d})/(${u} − ${d}) = ${ps}: with p = p* the price is a fair game, E[S<sub>t+1</sub> | S<sub>t</sub>] = S<sub>t</sub>.`,
 tt: (n, e, lt) => `n = ${n}: E[S<sub>n</sub>] = ${e}, P(S<sub>n</sub> &lt; 100) = ${lt}.`,
 crNote: 'Last row: the lognormal limit, ln S<sub>T</sub> ~ N(ln 100 + μ − σ²/2, σ²).',
 crw1: (n, s, u, d, p) => `n = ${n}, Δt = 1/${n}: u = e<sup>${s}√(1/${n})</sup> = ${u}, d = 1/u = ${d}, p = (e<sup>μΔt</sup> − d)/(u − d) = ${p}.`,
 crw2: (n, h) => `S<sub>T</sub> = 100u<sup>2K − ${n}</sup>, so S<sub>T</sub> ≤ 100 exactly when K ≤ ${n}/2 = ${h}.`,
 crw3e: (n, h, le, eq, lt) => `P(S<sub>T</sub> ≤ 100) = Σ<sub>k=0</sub><sup>${h}</sup> C(${n}, k)p<sup>k</sup>(1 − p)<sup>${n}−k</sup> = ${le}; the node k = ${h}: C(${n}, ${h})p<sup>${h}</sup>(1 − p)<sup>${h}</sup> = ${eq}; P(S<sub>T</sub> &lt; 100) = ${le} − ${eq} = ${lt}.`,
 crw3o: (n, h, lt) => `Odd n, no node at 100: P(S<sub>T</sub> &lt; 100) = P(S<sub>T</sub> ≤ 100) = Σ<sub>k=0</sub><sup>${h}</sup> C(${n}, k)p<sup>k</sup>(1 − p)<sup>${n}−k</sup> = ${lt}.`,
 crw4: (ml, s, z, ln, avg) => `Lognormal limit: Φ(−(μ − σ²/2)/σ) = Φ(−${ml}/${s}) = Φ(${z}) = ${ln}` + (avg ? `; the average of P(&lt;) and P(≤) is ${avg}.` : '.'),
 cr: (n, lt, le, ln) => `n = ${n}: P(S<sub>T</sub> &lt; 100) = ${lt}, P(S<sub>T</sub> ≤ 100) = ${le}; lognormal limit ${ln}.`,
 arNote: 'The "500 shocks" columns use the shocks of the chart above (the same seed); the long simulation of the lecture is given for φ = 0.5 and φ = 0.95.',
 arw1: (f, v) => `Stationary variance: σ²/(1 − φ²) = 1/(1 − ${f}²) = ${v}.`,
 arw2: (f, r1, r5) => `Autocorrelations: ρ(1) = ${f}, ρ(5) = ${f}<sup>5</sup> = ${r5}.`,
 arw3: (f, h) => h === null ? 'φ = 0: a shock is gone at once (white noise).' : `Half-life: ln 0.5/ln ${f} = ${h} periods.`,
 arw4: (v, r) => `On the 500 shocks of the chart: sample variance ${v}, ρ̂(5) = ${r}.`,
 ar: (f, v, h) => `φ = ${f}: variance ${v}, half-life ${h} periods.`,
 gmNote: 'The calibration of the chapter: μ − σ²/2 = A × daily mean, σ = daily standard deviation × √A, with A = 252 (365 for Bitcoin).',
 gmw1: (a, ml, s, mu, mp) => `${a}: μ − σ²/2 = ${ml}, σ = ${s}, so μ = ${ml} + ${s}²/2 = ${mu} a year (${mp}).`,
 gmw2: (mu, T, e, ml, md) => `E[S<sub>T</sub>] = 100e<sup>${mu} × ${T}</sup> = ${e}; median = 100e<sup>${ml} × ${T}</sup> = ${md}.`,
 gmw3: (a, z, ml, T, s, q) => `${a} quantile: z<sub>α</sub> = ${z}; 100 exp(${ml} × ${T} ${z.startsWith('−') ? '−' : '+'} ${s} × √${T} × ${z.replace('−', '')}) = ${q}.`,
 gmw4: (ml, T, s, z, p) => `P(S<sub>T</sub> &lt; 100) = Φ(−${ml} × √${T}/${s}) = Φ(${z}) = ${p}.`,
 gm: (a, T, e, md, r) => `${a}, T = ${T} years: mean ${e}, median ${md}, ratio ${r} = e<sup>σ²T/2</sup>.`,
 gcNote: n => `Median and maximum of the 300 GBM simulations of each series; last column: the expected number under the Normal distribution for the ${n} days of the S&P 500.`
});
"""


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch4-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH4_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH4_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH4_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    import re
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


def frame_pages(lang):
    """(page, title) of each table block's frame in the deck of `lang`."""
    import os
    import re
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved to their frames by ch4.js)."""
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
    import re
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch4_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>''')
    assert set(blocks) <= set(ORDER) and len(blocks) == len(BLOCKS), set(blocks) ^ {a for a, _, _ in BLOCKS}
    html = html[:start] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[end:]
    return html.replace('</head>', MATHJAX + '</head>', 1)


def build():
    import os
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch4_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


if __name__ == '__main__':
    build()
