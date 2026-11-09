"""Build the interactive pages of Chapter 5: interactive/ch5.html (EN) and interactive/ch5_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch5.js, inputs in interactive/ch5_data.js
(Quantlets/Ch_05/export_interactive.py), slide images in interactive/slides/.
"""
import json

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts

N = 5
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_05'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('crash-days', 'sfm_ch5_crash_days', 'why'),
          ('light-heavy', 'sfm_ch5_light_heavy', 'heavy'), ('loglog-real', 'sfm_ch5_loglog_real', 'heavy'),
          ('hill-plot', 'sfm_ch5_hill_plot', 'hill'), ('hill-bvb', 'sfm_ch5_hill_bvb', 'hill'),
          ('mean-excess', 'sfm_ch5_mean_excess', 'me'),
          ('gev-types', 'sfm_ch5_gev_types', 'gev'), ('maxima-sim', 'sfm_ch5_maxima_sim', 'gev'), ('gev-ppqq', 'sfm_ch5_gev_ppqq', 'gev'),
          ('return-levels', 'sfm_ch5_return_levels', 'gev'),
          ('gpd-densities', 'sfm_ch5_gpd_densities', 'pot'), ('threshold-stability', 'sfm_ch5_threshold_stability', 'pot'),
          ('gpd-ppqq', 'sfm_ch5_gpd_ppqq', 'pot'), ('tail-fit', 'sfm_ch5_tail_fit', 'pot'),
          ('oos', 'sfm_ch5_oos', 'var')]

PFX = {'crash-days': 'cd', 'light-heavy': 'lh', 'loglog-real': 'll', 'hill-plot': 'hp', 'hill-bvb': 'hb', 'mean-excess': 'me', 'gev-types': 'gt',
       'maxima-sim': 'ms', 'gev-ppqq': 'gq', 'return-levels': 'rl', 'gpd-densities': 'gd', 'threshold-stability': 'ts', 'gpd-ppqq': 'gp',
       'tail-fit': 'tf', 'oos': 'os'}

SECTIONS = {
 'ro': dict(why='Importanța cozilor', heavy='Cozi groase și variația regulată', hill='Estimatorul Hill', me='Funcția mean excess',
            gev='Block maxima și distribuția GEV', pot='Peaks over threshold și distribuția GPD', var='VaR și ES prin teoria valorilor extreme'),
 'en': dict(why='The Importance of Tails', heavy='Heavy Tails and Regular Variation', hill='The Hill Estimator', me='The Mean Excess Function',
            gev='Block Maxima and the GEV Distribution', pot='Peaks over Threshold and the GPD', var='VaR and ES by Extreme Value Theory'),
}

L = {
 'ro': dict(file='ch5_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 5 · Cozi groase și teoria valorilor extreme', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Pierderi zilnice L = −r, în %, din randamentele logaritmice: S&P 500 și DAX din ianuarie 1990, BET din septembrie 1997, Bitcoin din '
       'septembrie 2014 (7 zile pe săptămînă), pînă la 18 septembrie 2026; OMV Petrom, BRD, Banca Transilvania și Transgaz din 2010, din prețul '
       'ajustat. Prețuri de închidere EODHD. Ajustările GEV, intervalele bootstrap și maximele simulate sînt cele din curs (aceleași reguli și '
       'aceeași sămînță a generatorului); estimările Hill, GPD, VaR și ES sînt recalculate în browser din aceleași pierderi.'),
 'en': dict(file='ch5.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 5 · Heavy Tails and Extreme Value Theory', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Daily losses L = −r, in %, from log returns: S&P 500 and DAX from January 1990, BET from September 1997, Bitcoin from September 2014 '
       '(7 days a week), to 18 September 2026; OMV Petrom, BRD, Banca Transilvania and Transgaz from 2010, from the adjusted price. EODHD '
       'closing prices. The GEV fits, the bootstrap intervals and the simulated maxima are those of the lecture (the same rules and the same '
       'generator seed); Hill, GPD, VaR and ES are recomputed in the browser from the same losses.'),
}


def widgets_old(lang):
    ro = lang == 'ro'
    A4 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
    S4 = [('snp', 'OMV Petrom (SNP)'), ('brd', 'BRD'), ('tlv', 'Banca Transilvania (TLV)'), ('tgn', 'Transgaz (TGN)')]
    return {
     'crash-days': (slider('cd-t', 'Nivelul distribuției Normale: pierderea depășită o dată la T ani' if ro else 'Normal level: the loss exceeded once in T years', 0, 4, 1, 2)
                    + play('cd'), charts('cd-chart')),
     'light-heavy': (slider('lh-x', 'Pragul x (linia punctată)' if ro else 'Threshold x (dotted line)', 0, 5, 1, 3) + play('lh'), charts('lh-chart')),
     'loglog-real': (slider('ll-k', 'Dreapta celor mai mici pătrate prin cele mai mari k/n dintre pierderi' if ro else 'Least-squares line through the largest k/n of the losses', 0.5, 10, 0.5, 2.5)
                     + play('ll'), charts('ll-chart')),
     'hill-plot': (slider('hp-k', 'k/n marcat (linia punctată)' if ro else 'Marked k/n (dotted line)', 0.5, 10, 0.25, 2.5)
                   + select('hp-b', 'Banda de 95%' if ro else '95% band', A4, 'sp500') + play('hp'), charts('hp-chart')),
     'hill-bvb': (slider('hb-k', 'k/n marcat (linia punctată)' if ro else 'Marked k/n (dotted line)', 0.5, 10, 0.25, 2.5)
                  + select('hb-b', 'Banda de 95%' if ro else '95% band', S4, 'tlv') + play('hb'), charts('hb-chart')),
     'mean-excess': (slider('me-q', 'Pragul u: cuantila pierderilor' if ro else 'Threshold u: quantile of the losses', 80, 98, 1, 90) + play('me'), charts('me-chart')),
     'gev-types': (slider('gt-s', '|ξ| pentru Fréchet (ξ > 0) și Weibull (ξ < 0)' if ro else '|ξ| of Fréchet (ξ > 0) and Weibull (ξ < 0)', 0.1, 1, 0.05, 0.5) + play('gt'), charts('gt-chart')),
     'maxima-sim': (slider('ms-n', 'Mărimea n a fiecărui eșantion' if ro else 'Size n of each sample', 0, 6, 1, 6) + play('ms'), charts('ms-chart')),
     'gev-ppqq': (select('gq-a', 'Seria' if ro else 'Series', A4, 'sp500')
                  + select('gq-m', 'Modelul' if ro else 'Model', [('gev', 'GEV (ξ estimat)' if ro else 'GEV (ξ estimated)'), ('gumbel', 'Gumbel (ξ = 0)')], 'gev')
                  + play('gq'), charts('gq-chart')),
     'return-levels': (slider('rl-t', 'Perioada de revenire T (linia punctată)' if ro else 'Return period T (dotted line)', 0, 6, 1, 3) + play('rl'), charts('rl-chart')),
     'gpd-densities': (slider('gd-x', 'ξ al curbei portocalii' if ro else 'ξ of the orange curve', -0.5, 1, 0.05, 0.3) + play('gd'), charts('gd-chart')),
     'threshold-stability': (slider('ts-q', 'Pragul u marcat: cuantila pierderilor' if ro else 'Marked threshold u: quantile of the losses', 80, 99, 1, 90) + play('ts'), charts('ts-chart')),
     'gpd-ppqq': (select('gp-a', 'Seria' if ro else 'Series', A4, 'sp500')
                  + slider('gp-q', 'Pragul u: cuantila pierderilor' if ro else 'Threshold u: quantile of the losses', 80, 98, 1, 90) + play('gp'), charts('gp-chart')),
     'tail-fit': (slider('tf-p', 'Nivelul p al VaR și ES' if ro else 'Level p of VaR and ES', 0, 5, 1, 1) + play('tf'), charts('tf-chart')),
     'oos': (select('os-a', 'Seria' if ro else 'Series', A4, 'sp500')
             + select('os-p', 'Nivelul' if ro else 'Level', [('var1', 'VaR 1%'), ('var01', 'VaR 0,1%' if ro else 'VaR 0.1%')], 'var1') + play('os'), charts('os-chart')),
    }


# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""(function () {
const zile = n => n === 0 ? 'nicio zi' : n === 1 ? 'o zi' : (n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? `${n} de zile` : `${n} zile`);
const luni = n => n === 1 ? 'o lună' : (n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? `${n} de luni` : `${n} luni`);
const ori = n => n === 0 ? 'niciodată' : n === 1 ? 'o dată' : (n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? `de ${n} de ori` : `de ${n} ori`);
const ani = t => t === '1' ? '1 an' : (+t.replace(/\s/g, '') % 100 >= 20 || +t.replace(/\s/g, '') >= 100 ? `${t} de ani` : `${t} ani`);
const M = { EVT: 'EVT', Historical: 'Istoric', Normal: 'Normal' };
window.CH5_TEXT = {
 cd: T => `Nivelul pierderii zilnice depășite o dată la ${ani(T)} în distribuția Normală ajustată fiecărei serii (252 de zile pe an):`,
 cdItem: (a, y, lv, nb, ex, n) => `${a} (din ${y}, ${n} de zile): nivelul ${lv}%; observat ${zile(nb)} sub el, distribuția Normală așteaptă ${ex}`,
 lh: (x, n, e, t, p) => `P(X &gt; ${x}): distribuția Normală ${n}, distribuția exponențială ${e}, Student-t cu 3 grade de libertate ${t}, Pareto ${p}.`,
 lh2: (x, n, e, t, p) => `Raportul P(X &gt; ${2 * x})/P(X &gt; ${x}): distribuția Normală ${n}, distribuția exponențială ${e}, Student-t ${t}, Pareto ${p} = 2<sup>−3</sup>.`,
 ll: f => `Pantele dreptelor celor mai mici pătrate prin cele mai mari ${f} dintre pierderi (tail index-ul α = −panta) și estimatorul Hill cu același k:`,
 llItem: (a, k, s, h) => `${a}: k = ${k}, α din pantă ${s}, Hill ${h}`,
 hp: f => `Estimatorul Hill la k = ${f} din n, cu intervalul de 95% α̂(1 ± 1,96/√k) și SE i.i.d. α̂/√k:`,
 hpItem: (a, k, al, lo, hi, se, sb) => `${a}: k = ${k}, α̂ = ${al}, CI 95% [${lo}; ${hi}], SE i.i.d. ${se}` + (sb ? `, SE bootstrap pe blocuri de 20 de zile ${sb}` : ''),
 me: q => `Pragul u = cuantila de ${q}% a pierderilor; GPD ajustată prin verosimilitate maximă pe excesele L − u:`,
 meItem: (a, u, nu, e, xi, b) => `${a}: u = ${u}%, N<sub>u</sub> = ${nu}, e(u) = ${e}%; ξ̂ = ${xi}, β̂ = ${b}`,
 gt: (s, end, al) => `ξ = ±${s}: distribuția Weibull se oprește la capătul superior μ − σ/ξ = ${end}; coada Fréchet scade ca x<sup>−1/ξ</sup>, cu tail index-ul α = ${al}.`,
 gt2: (g, f, w) => `P(X &gt; 5): Gumbel ${g}, Fréchet ${f}, Weibull ${w}.`,
 ms: (n, ex) => `5 000 de eșantioane, fiecare cu n = ${n} ${n % 100 >= 20 ? 'de ' : ''}extrageri.` + (ex ? ' Rulare suplimentară a codului din curs, cu aceeași sămînță (2026); cursul folosește n = 250.' : ' Simularea din curs.'),
 ms2: (e, eg, p, pf, u, uw) => `Media maximelor normalizate față de media legii limită: exponențială ${e} (Gumbel ${eg}), Pareto(3) ${p} (Fréchet ${pf}), uniformă ${u} (Weibull ${uw}).`,
 gq: (a, n, m, xi, mu, s) => `${a}: ${n} de maxime lunare ale pierderilor zilnice. ` + (m === 'gev' ? `GEV din curs: ξ̂ = ${xi}, μ̂ = ${mu}%, σ̂ = ${s}%.` : `Gumbel (ξ = 0), ajustat prin verosimilitate maximă: μ̂ = ${mu}%, σ̂ = ${s}%. Rulare suplimentară, pentru comparație; cursul ajustează GEV.`),
 gq2: (y1, m1, q1, y2, m2, q2, ks) => `Cel mai mare maxim lunar: ${m1}% (${y1}), cuantila ajustată ${q1}%; al doilea: ${m2}% (${y2}), cuantila ${q2}%. Distanța maximă dintre funcția de repartiție ajustată și cea empirică: ${ks}.`,
 rl: T => `Return level pentru T = ${ani(String(T))}: pierderea zilnică pe care maximul lunar o depășește, în medie, o dată la T ani (GEV din curs):`,
 rlItem: (a, z, na, y, ex, ci) => `${a}: ${z}%` + (ci ? ` (bootstrap 95%: [${ci[0]}; ${ci[1]}])` : '') + (na ? `; depășit ${na === 1 ? 'într-o lună' : 'în ' + luni(na)} din ${y} ani de date` : `; nedepășit în cei ${y} ani de date`) + (ex ? '; T este mai lung decît eșantionul: extrapolare' : ''),
 gd: (s, al, mf, vf, end, p10) => `ξ = ${s}: ` + (al ? `coadă de tip Pareto cu α = 1/ξ = ${al}; ` : (end ? `coadă mărginită, capătul superior −β/ξ = ${end}; ` : 'coadă exponențială; ')) +
   (mf ? `media excesului β/(1 − ξ) = ${mf}β` : 'media excesului este infinită (ξ ≥ 1)') + `; varianța ${vf ? 'este finită (ξ &lt; 1/2)' : 'este infinită (ξ ≥ 1/2)'}. P(Y &gt; 10) = ${p10} (β = 1).`,
 gd2: (a, b, c) => `Celelalte curbe, P(Y &gt; 10): ξ = −0,3: ${a}; ξ = 0: ${b}; ξ = 0,6: ${c}.`,
 ts: q => `Pragul u = cuantila de ${q}% a pierderilor (estimările din curs):`,
 tsItem: (a, u, nu, xi, lo, hi, v1, v01, xmn, xmx, vmn, vmx) => `${a}: u = ${u}%, N<sub>u</sub> = ${nu}, ξ̂ = ${xi} [${lo}; ${hi}], VaR 1% ${v1}%, VaR 0,1% ${v01}%. Între cuantilele de 85% și 95%: ξ̂ între ${xmn} și ${xmx}, VaR 1% între ${vmn}% și ${vmx}%`,
 gp: (a, q, u, nu, xi, b) => `${a}, u = cuantila de ${q}% = ${u}%: N<sub>u</sub> = ${nu} ${+String(nu).replace(/\s/g, '') % 100 >= 20 || +String(nu).replace(/\s/g, '') % 100 === 0 ? 'de ' : ''}excese, ξ̂ = ${xi}, β̂ = ${b}.`,
 gp2: (y1, q1, y2, q2) => `Cel mai mare exces: ${y1}%, cuantila GPD ajustată ${q1}%; al doilea: ${y2}%, cuantila ${q2}%.`,
 tf: p => `VaR ${p} și ES ${p} ale pierderilor zilnice (GPD peste cuantila de 90%; Hill cu k = 2,5% din n):`,
 tfItem: (a, e, h, hs, nm, ee, eh, en, np, lo) => `${a}: VaR EVT ${e}%, Hill ${h}%, istoric ${hs}%, Normal ${nm}%; ES EVT ${ee}%, istoric ${eh}%, Normal ${en}%. Zile așteptate dincolo de VaR în eșantion: ${np}` + (lo ? ' (sub o zi: cuantila istorică este interpolată între cele mai mari două pierderi)' : ''),
 os: (a, p, ne, nt, ex) => `${a}: VaR ${p} estimat pe ${ne} de zile pînă la sfîrșitul lui 2019, testat pe ${nt} de zile din 2020–2026; sînt așteptate ${ex} depășiri.`,
 osItem: (m, v, e, s) => `${M[m]}: VaR ${v}%, ` + (e ? `depășit ${ori(e)} (${s}% din zilele de test)` : 'nedepășit în zilele de test')
};
})();
""",
 'en': r"""(function () {
const days = n => n === 1 ? '1 day' : `${n} days`;
const times = n => n === 0 ? 'never' : n === 1 ? 'once' : `${n} times`;
const months = n => n === 1 ? '1 month' : `${n} months`;
const yrs = t => t === '1' ? '1 year' : `${t} years`;
window.CH5_TEXT = {
 cd: T => `Level of the daily loss exceeded once in ${yrs(T)} under the Normal distribution fitted to each series (252 days per year):`,
 cdItem: (a, y, lv, nb, ex, n) => `${a} (from ${y}, ${n} days): level ${lv}%; observed ${days(nb)} below it, the Normal distribution expects ${ex}`,
 lh: (x, n, e, t, p) => `P(X &gt; ${x}): Normal distribution ${n}, exponential distribution ${e}, Student-t with 3 degrees of freedom ${t}, Pareto ${p}.`,
 lh2: (x, n, e, t, p) => `Ratio P(X &gt; ${2 * x})/P(X &gt; ${x}): Normal distribution ${n}, exponential distribution ${e}, Student-t ${t}, Pareto ${p} = 2<sup>−3</sup>.`,
 ll: f => `Slopes of the least-squares lines through the largest ${f} of the losses (tail index α = −slope) and the Hill estimator with the same k:`,
 llItem: (a, k, s, h) => `${a}: k = ${k}, α from the slope ${s}, Hill ${h}`,
 hp: f => `Hill estimator at k = ${f} of n, with the 95% interval α̂(1 ± 1.96/√k) and the i.i.d. SE α̂/√k:`,
 hpItem: (a, k, al, lo, hi, se, sb) => `${a}: k = ${k}, α̂ = ${al}, 95% CI [${lo}, ${hi}], i.i.d. SE ${se}` + (sb ? `, moving-block bootstrap SE (blocks of 20 days) ${sb}` : ''),
 me: q => `Threshold u = ${q}% quantile of the losses; GPD fitted by maximum likelihood to the excesses L − u:`,
 meItem: (a, u, nu, e, xi, b) => `${a}: u = ${u}%, N<sub>u</sub> = ${nu}, e(u) = ${e}%; ξ̂ = ${xi}, β̂ = ${b}`,
 gt: (s, end, al) => `ξ = ±${s}: the Weibull distribution stops at the upper endpoint μ − σ/ξ = ${end}; the Fréchet tail falls as x<sup>−1/ξ</sup>, with tail index α = ${al}.`,
 gt2: (g, f, w) => `P(X &gt; 5): Gumbel ${g}, Fréchet ${f}, Weibull ${w}.`,
 ms: (n, ex) => `5,000 samples of n = ${n} draws.` + (ex ? ' Extra run of the lecture\'s code, with the same seed (2026); the lecture uses n = 250.' : ' The simulation of the lecture.'),
 ms2: (e, eg, p, pf, u, uw) => `Mean of the normalised maxima against the mean of the limit law: exponential ${e} (Gumbel ${eg}), Pareto(3) ${p} (Fréchet ${pf}), uniform ${u} (Weibull ${uw}).`,
 gq: (a, n, m, xi, mu, s) => `${a}: ${n} monthly maxima of the daily losses. ` + (m === 'gev' ? `GEV of the lecture: ξ̂ = ${xi}, μ̂ = ${mu}%, σ̂ = ${s}%.` : `Gumbel (ξ = 0), fitted by maximum likelihood: μ̂ = ${mu}%, σ̂ = ${s}%. Extra run, for comparison; the lecture fits the GEV.`),
 gq2: (y1, m1, q1, y2, m2, q2, ks) => `Largest monthly maximum: ${m1}% (${y1}), fitted quantile ${q1}%; second: ${m2}% (${y2}), quantile ${q2}%. Largest distance between the fitted and the empirical distribution function: ${ks}.`,
 rl: T => `Return level for T = ${yrs(String(T))}: the daily loss that the monthly maximum exceeds on average once in T years (GEV of the lecture):`,
 rlItem: (a, z, na, y, ex, ci) => `${a}: ${z}%` + (ci ? ` (95% bootstrap: [${ci[0]}, ${ci[1]}])` : '') + (na ? `; exceeded in ${months(na)} of ${y} years of data` : `; never exceeded in ${y} years of data`) + (ex ? '; T is longer than the sample: extrapolation' : ''),
 gd: (s, al, mf, vf, end, p10) => `ξ = ${s}: ` + (al ? `Pareto-type tail with α = 1/ξ = ${al}; ` : (end ? `bounded tail, upper endpoint −β/ξ = ${end}; ` : 'exponential tail; ')) +
   (mf ? `mean excess β/(1 − ξ) = ${mf}β` : 'the mean excess is infinite (ξ ≥ 1)') + `; the variance ${vf ? 'is finite (ξ &lt; 1/2)' : 'is infinite (ξ ≥ 1/2)'}. P(Y &gt; 10) = ${p10} (β = 1).`,
 gd2: (a, b, c) => `The other curves, P(Y &gt; 10): ξ = −0.3: ${a}; ξ = 0: ${b}; ξ = 0.6: ${c}.`,
 ts: q => `Threshold u = ${q}% quantile of the losses (estimates of the lecture):`,
 tsItem: (a, u, nu, xi, lo, hi, v1, v01, xmn, xmx, vmn, vmx) => `${a}: u = ${u}%, N<sub>u</sub> = ${nu}, ξ̂ = ${xi} [${lo}, ${hi}], VaR 1% ${v1}%, VaR 0.1% ${v01}%. Between the 85% and 95% quantiles: ξ̂ between ${xmn} and ${xmx}, VaR 1% between ${vmn}% and ${vmx}%`,
 gp: (a, q, u, nu, xi, b) => `${a}, u = ${q}% quantile = ${u}%: N<sub>u</sub> = ${nu} excesses, ξ̂ = ${xi}, β̂ = ${b}.`,
 gp2: (y1, q1, y2, q2) => `Largest excess: ${y1}%, fitted GPD quantile ${q1}%; second: ${y2}%, quantile ${q2}%.`,
 tf: p => `VaR ${p} and ES ${p} of the daily losses (GPD above the 90% quantile; Hill with k = 2.5% of n):`,
 tfItem: (a, e, h, hs, nm, ee, eh, en, np, lo) => `${a}: VaR EVT ${e}%, Hill ${h}%, historical ${hs}%, Normal ${nm}%; ES EVT ${ee}%, historical ${eh}%, Normal ${en}%. Days expected beyond VaR in the sample: ${np}` + (lo ? ' (less than one day: the historical quantile is interpolated between the two largest losses)' : ''),
 os: (a, p, ne, nt, ex) => `${a}: VaR ${p} estimated on ${ne} days to the end of 2019, tested on ${nt} days of 2020–2026; ${ex} exceedances expected.`,
 osItem: (m, v, e, s) => `${m}: VaR ${v}%, ` + (e ? `exceeded ${times(e)} (${s}% of the test days)` : 'never exceeded on the test days')
};
})();
""",
}

QA = {
 'ro': {
  'crash-days': ('Sub nivelul pierderii depășite o dată la 100 de ani în distribuția Normală, S&P 500 are 35 de zile din 1990. Butonul ▶ mută perioada T de la 1 an la 10 000 de ani. Cîte zile ale S&P 500 sînt sub nivelul unei pierderi de „o dată la 10 000 de ani”?',
                 '17 zile, sub −5,56%. Distribuția Normală așteaptă 0,0037 astfel de zile în 9 240 (probabilitatea 1/(10 000 × 252) pe zi). Nivelul coboară doar de la −4,44% la −5,56%: coada distribuției Normale scade atît de repede încît o perioadă de 100 de ori mai lungă mută pragul cu puțin peste un punct procentual. BET are 22 de zile sub nivelul său de 10 000 de ani (−7,25%). La T = 1 an, S&P 500 are 120 de zile sub −2,98%, față de 36,7 așteptate.'),
  'light-heavy': ('La x = 10, P(X > 20)/P(X > 10) = 0,128 pentru distribuția Student-t cu 3 grade de libertate. Cît devine raportul P(X > 2x)/P(X > x) al distribuției Student-t la x = 50?',
                  '0,125 = 2<sup>−3</sup>: pentru o coadă cu variație regulată, P(X > tx)/P(X > x) → t<sup>−α</sup>, cu α = 3 pentru Student-t cu 3 grade de libertate. La x = 2 raportul este încă 0,201, la x = 20 este 0,126. Pentru distribuția exponențială raportul este e<sup>−x</sup>, 1,9 × 10<sup>−22</sup> la x = 50, iar pentru distribuția Normală circa 10<sup>−1629</sup>: cozile subțiri dispar la fiecare dublare a pragului, coada de tip putere pierde mereu același factor 8.'),
  'loglog-real': ('Prin cele mai mari 2,5% dintre pierderi, panta S&P 500 este −3,00. Butonul ▶ mărește proporția pînă la 10%. Ce pantă dă dreapta prin cele mai mari 10% dintre pierderi?',
                  '−2,50: dreapta intră în corpul distribuției, unde punctele se curbează și aplatizează dreapta. La 10%, estimatorul Hill scade și el, la 2,04. DAX: de la −3,57 la −2,65; BET: −2,17; Bitcoin: −2,31. La 0,5% (46 de pierderi pentru S&P 500), panta DAX este −4,39: puține puncte, variabilitate mare. Legea de putere descrie doar coada, iar estimarea depinde de cîte pierderi intră în ea.'),
  'hill-plot': ('La k = 2,5% din n, α̂ al S&P 500 este 2,97, cu intervalul de 95% [2,59; 3,35]. Butonul ▶ mută k/n de la 0,5% la 10%. Cît este α̂ al S&P 500 la k = 10% din n?',
                '2,04, cu intervalul i.i.d. [1,91; 2,18], care nu mai conține 2,97. Cu k mare intră corpul distribuției, care deplasează estimarea în jos; intervalul se îngustează (SE = α̂/√k), deci un k mare dă o estimare precisă a unei mărimi deplasate. BET coboară la 1,82, sub 2, deși în zona plată este 2,57. Intervalul i.i.d. subestimează și incertitudinea: la k = 2,5%, SE i.i.d. a S&P 500 este 0,20, iar SE bootstrap pe blocuri de 20 de zile 0,30, de 1,5 ori mai mare, pentru că pierderile mari vin grupat; pentru DAX cele două coincid (0,20).'),
  'hill-bvb': ('La k = 2,5% din n, TLV are α̂ = 2,49 și intervalul de 95% [1,98; 2,99], lat de 1,01. Cu n = 3 788, k = 1% din n înseamnă 37 de pierderi. Cît de lat este intervalul TLV la k = 1% din n?',
               '1,96, de la 2,07 la 4,03, aproape dublu. Lățimea intervalului este 2 × 1,96 α̂/√k: k scade de la 94 la 37 (√(94/37) = 1,59), iar α̂ crește de la 2,49 la 3,05. Eșantioanele BVB sînt de 2,5 ori mai scurte decît cele ale indicilor, deci același k/n dă mai puține pierderi și benzi mai largi; S&P 500 are la k = 1% intervalul [2,35; 3,56]. În zona plată (1%–5%), niciuna dintre cele patru acțiuni nu are α̂ sub 2.'),
  'mean-excess': ('La pragul u = cuantila de 90% (1,16%), excesul mediu al pierderilor S&P 500 este 0,92%. Butonul ▶ ridică pragul de la cuantila de 80% la cea de 98%. Cît este e(u) la cuantila de 98% (u = 2,52%)?',
                  '1,22%: excesul mediu crește cu pragul, cum cere o coadă groasă. Pentru o coadă Pareto, e(u) = u/(α − 1) crește liniar; pentru distribuția exponențială ar fi constant, pentru distribuția Normală ar scădea spre 0. BET: de la 1,30% la 1,79%. La cuantila de 98%, GPD se ajustează pe doar 185 de excese ale S&P 500 (față de 924 la 90%), iar ξ̂ urcă de la 0,158 la 0,255: estimări mai zgomotoase, cum arată și ultimele puncte ale graficului.'),
  'gev-types': ('Cu ξ = −0,5, distribuția Weibull se oprește la x = 2. Unde se oprește pentru ξ = −0,25?',
                'La x = 4: capătul superior este μ − σ/ξ = 0 − 1/(−0,25). Cînd |ξ| scade spre 0, capătul se depărtează și distribuția Weibull se apropie de Gumbel; distribuția Fréchet cu ξ = 0,25 are tail index-ul 1/ξ = 4, iar P(X > 5) scade de la 0,078 (ξ = 0,5) la 0,038, față de 0,0067 pentru Gumbel. Semnul lui ξ decide tipul cozii, mărimea lui decide cît de groasă sau cît de scurtă este.'),
  'maxima-sim': ('Cu n = 250, media maximelor exponențiale minus ln n este 0,568, aproape de media legii Gumbel, 0,577 (constanta Euler–Mascheroni). Butonul ▶ trece prin n = 2, 5, 10, 21, 50, 100, 250. Cît este media pentru n = 2?',
                 '0,837 în simulare; exact, E[M<sub>2</sub>] − ln 2 = 1 + 1/2 − 0,693 = 0,807. În general E[M<sub>n</sub>] = 1 + 1/2 + … + 1/n, iar diferența față de ln n tinde la 0,577: convergența către Gumbel este lentă, de ordinul 1/n. Pentru distribuția uniformă, media lui n(M − 1) este exact −n/(n + 1): −0,667 la n = 2, aproape −1 la n = 250. La n = 21 (o lună de tranzacționare) mediile sînt deja 0,593, 1,373 și −0,957. Valorile n ≠ 250 sînt rulări suplimentare ale codului din curs, cu aceeași sămînță.'),
  'gev-ppqq': ('În graficul QQ al GEV, luna cu cea mai mare pierdere a S&P 500 (martie 2020, 12,77%) are cuantila ajustată 12,43%. Butonul ▶ trece la ajustarea Gumbel (ξ = 0). Ce cuantilă ajustată primește aceeași lună?',
               '6,99%: o coadă exponențială pune cea mai extremă lună la jumătate din valoarea observată, iar punctele de sus ale graficului QQ urcă mult deasupra bisectoarei. Nivelul de 10 ani scade de la 7,4% (GEV) la 5,3% (Gumbel), un nivel depășit în 12 dintre cele 440 de luni, adică de peste trei ori mai des decît o dată la 10 ani. Ajustarea Gumbel este o rulare suplimentară, pentru comparație; intervalul de 95% al lui ξ̂ din curs, [0,13; 0,29], exclude 0. Pentru BET (ξ̂ = 0,36), GEV pune cea mai mare lună la 26,1%, față de 13,1% observat: coada ajustată este mai groasă decît datele.'),
  'return-levels': ('Nivelul de 10 ani al BET este 13,3%, iar cea mai mare pierdere observată în 28,5 ani este 13,1%. Butonul ▶ mută T de la 1 la 100 de ani. Cît este nivelul de 100 de ani al BET?',
                    '32,2%, de aproape 2,5 ori cea mai mare pierdere observată. Cu ξ̂ = 0,36, return level crește aproximativ ca T<sup>ξ</sup>: fiecare înmulțire a lui T cu 10 mărește nivelul cu un factor de circa 2,4. Este o extrapolare la de 3,5 ori lungimea eșantionului; deja la 10 ani, intervalul bootstrap este [10,3%; 17,5%]. S&P 500, cu ξ̂ = 0,21: 13,4% la 100 de ani, aproape de cea mai mare pierdere observată (12,8%).'),
  'gpd-densities': ('Cu ξ = 0,3, coada GPD are tail index-ul 1/ξ ≈ 3,3: media și varianța excesului sînt finite. De la ce valoare a lui ξ varianța excesului devine infinită?',
                    'De la ξ = 1/2: varianța există doar pentru α = 1/ξ > 2. Media excesului, β/(1 − ξ), există pînă la ξ = 1. Pentru ξ = 0,5, P(Y > 10) = 0,028 (β = 1), față de 0,0098 pentru ξ = 0,3 și 4,5 × 10<sup>−5</sup> pentru ξ = 0. Ajustările GPD din curs au ξ̂ între 0,12 și 0,22 pentru cele patru serii: cozi groase, cu varianță finită.'),
  'threshold-stability': ('La pragul de 90%, ξ̂ al S&P 500 este 0,158, iar VaR 1% EVT 3,30%. Butonul ▶ ridică pragul de la cuantila de 80% la cea de 99%. Cît devin ξ̂ și VaR 1% la pragul de 99% (93 de excese)?',
                          'ξ̂ = 0,203, cu un interval de 95% de la −0,07 la 0,47 (SE 0,138, de patru ori mai mare decît la 90%), dar VaR 1% rămîne 3,16%. VaR 1% este în interiorul datelor (92 de zile dincolo de el în eșantion), deci depinde puțin de forma cozii; ξ̂ depinde mult de prag. Între cuantilele de 85% și 95%: ξ̂ între 0,13 și 0,22, VaR 1% între 3,25% și 3,33%. BET la 99%: ξ̂ = −0,07, VaR 1% 4,35%.'),
  'gpd-ppqq': ('La pragul de 90%, cel mai mare exces al S&P 500 (11,60%, martie 2020) este peste cuantila GPD ajustată (11,11%). Butonul ▶ ridică pragul de la cuantila de 80% la cea de 98%. Ce se întîmplă cu acest punct în graficul QQ la pragul de 98%?',
               'Trece sub bisectoare: excesul este 10,24% (pragul a urcat la 2,52%), iar cuantila ajustată 12,65%, pentru că ξ̂ crește la 0,255 pe doar 185 de excese. Ultimul punct al graficului QQ depinde mult de ξ̂, deci de prag. Pentru BET, GPD de la cuantila de 90% pune cel mai mare exces la 18,56%, față de 11,74% observat: coada ajustată este mai groasă decît cele mai mari pierderi din eșantion.'),
  'tail-fit': ('Butonul ▶ coboară nivelul p de la 2,5% la 0,01%. Cu p = 0,01% (o zi din 10 000), cît este VaR al S&P 500 după EVT și după distribuția Normală?',
               'EVT 10,79%, Normal 4,18%, de 2,6 ori mai puțin. La p = 1%, raportul este 1,27 (3,30% față de 2,60%). Eșantionul de 9 240 de zile conține în medie 0,9 zile dincolo de VaR 0,01%, deci cuantila istorică (10,21%) este doar o interpolare între cele mai mari două pierderi. Cuantila Hill dă 15,02%: departe în coadă, estimatorii diferă mult. BET: EVT 18,10%, Hill 26,16%, istoric 12,24%.'),
  'oos': ('Pentru S&P 500, VaR 1% al distribuției Normale este depășit în 44 de zile din 2020–2026, față de 16,9 așteptate. Butonul ▶ trece prin cele patru serii. De cîte ori este depășit VaR 1% Normal al BET, unde sînt așteptate 16,8 depășiri?',
          'De 13 ori, sub valoarea așteptată; VaR EVT de 4 ori, cel istoric de 5 ori. Perioada de test a fost mai liniștită pentru BET decît perioada de estimare 1997–2019. Un VaR estimat o singură dată nu urmărește volatilitatea: depășirile sînt prea multe după 2020 la S&P 500 (EVT 25, istoric 26) și prea puține la BET. La VaR 0,1%, distribuția Normală rămîne depășită prea des pentru toate cele patru serii: S&P 500 23 de zile față de 1,7, BET 4 față de 1,7, Bitcoin 8 față de 2,5.'),
 },
 'en': {
  'crash-days': ('Below the level of the loss exceeded once in 100 years under the Normal distribution, the S&P 500 has 35 days since 1990. The ▶ button moves the period T from 1 year to 10,000 years. How many S&P 500 days are below the level of a "once in 10,000 years" loss?',
                 '17 days, below −5.56%. The Normal distribution expects 0.0037 such days in 9,240 (probability 1/(10,000 × 252) per day). The level only moves from −4.44% to −5.56%: the tail of the Normal distribution falls so fast that a period 100 times longer moves the threshold by little more than one percentage point. BET has 22 days below its 10,000-year level (−7.25%). At T = 1 year, the S&P 500 has 120 days below −2.98%, against 36.7 expected.'),
  'light-heavy': ('At x = 10, P(X > 20)/P(X > 10) = 0.128 for the Student-t distribution with 3 degrees of freedom. What does the ratio P(X > 2x)/P(X > x) of the Student-t become at x = 50?',
                  '0.125 = 2<sup>−3</sup>: for a regularly varying tail, P(X > tx)/P(X > x) → t<sup>−α</sup>, with α = 3 for the Student-t with 3 degrees of freedom. At x = 2 the ratio is still 0.201, at x = 20 it is 0.126. For the exponential distribution the ratio is e<sup>−x</sup>, 1.9 × 10<sup>−22</sup> at x = 50, and for the Normal distribution about 10<sup>−1629</sup>: light tails vanish at every doubling of the threshold, the power-law tail always loses the same factor 8.'),
  'loglog-real': ('Through the largest 2.5% of the losses, the slope of the S&P 500 is −3.00. The ▶ button raises the share to 10%. What slope does the line through the largest 10% of the losses give?',
                  '−2.50: the line reaches into the body of the distribution, where the points bend and flatten the line. At 10%, the Hill estimator also falls, to 2.04. DAX: from −3.57 to −2.65; BET: −2.17; Bitcoin: −2.31. At 0.5% (46 losses for the S&P 500), the slope of the DAX is −4.39: few points, high variability. The power law describes only the tail, and the estimate depends on how many losses enter it.'),
  'hill-plot': ('At k = 2.5% of n, α̂ of the S&P 500 is 2.97, with the 95% interval [2.59, 3.35]. The ▶ button moves k/n from 0.5% to 10%. What is α̂ of the S&P 500 at k = 10% of n?',
                '2.04, with the i.i.d. interval [1.91, 2.18], which no longer contains 2.97. With a large k the body of the distribution enters and biases the estimate downwards; the interval narrows (SE = α̂/√k), so a large k gives a precise estimate of a biased quantity. BET falls to 1.82, below 2, although it is 2.57 in the flat zone. The i.i.d. interval also understates the uncertainty: at k = 2.5%, the i.i.d. SE of the S&P 500 is 0.20 and the moving-block bootstrap SE (blocks of 20 days) 0.30, 1.5 times larger, because large losses come in clusters; for the DAX the two coincide (0.20).'),
  'hill-bvb': ('At k = 2.5% of n, TLV has α̂ = 2.49 and the 95% interval [1.98, 2.99], 1.01 wide. With n = 3,788, k = 1% of n means 37 losses. How wide is the TLV interval at k = 1% of n?',
               '1.96, from 2.07 to 4.03, almost twice as wide. The width of the interval is 2 × 1.96 α̂/√k: k falls from 94 to 37 (√(94/37) = 1.59), and α̂ rises from 2.49 to 3.05. The BVB samples are 2.5 times shorter than those of the indices, so the same k/n gives fewer losses and wider bands; the S&P 500 has the interval [2.35, 3.56] at k = 1%. In the flat zone (1%–5%), none of the four stocks has α̂ below 2.'),
  'mean-excess': ('At the threshold u = 90% quantile (1.16%), the mean excess of the S&P 500 losses is 0.92%. The ▶ button raises the threshold from the 80% to the 98% quantile. What is e(u) at the 98% quantile (u = 2.52%)?',
                  '1.22%: the mean excess grows with the threshold, as a heavy tail requires. For a Pareto tail, e(u) = u/(α − 1) grows linearly; for the exponential distribution it would be constant, for the Normal distribution it would fall towards 0. BET: from 1.30% to 1.79%. At the 98% quantile, the GPD is fitted to only 185 excesses of the S&P 500 (against 924 at 90%), and ξ̂ rises from 0.158 to 0.255: noisier estimates, as the last points of the chart also show.'),
  'gev-types': ('With ξ = −0.5, the Weibull distribution stops at x = 2. Where does it stop for ξ = −0.25?',
                'At x = 4: the upper endpoint is μ − σ/ξ = 0 − 1/(−0.25). As |ξ| falls towards 0, the endpoint moves away and the Weibull distribution approaches the Gumbel; the Fréchet distribution with ξ = 0.25 has tail index 1/ξ = 4, and P(X > 5) falls from 0.078 (ξ = 0.5) to 0.038, against 0.0067 for the Gumbel. The sign of ξ decides the type of tail, its size decides how heavy or how short it is.'),
  'maxima-sim': ('With n = 250, the mean of the exponential maxima minus ln n is 0.568, close to the mean of the Gumbel law, 0.577 (the Euler–Mascheroni constant). The ▶ button goes through n = 2, 5, 10, 21, 50, 100, 250. What is the mean for n = 2?',
                 '0.837 in the simulation; exactly, E[M<sub>2</sub>] − ln 2 = 1 + 1/2 − 0.693 = 0.807. In general E[M<sub>n</sub>] = 1 + 1/2 + … + 1/n, and its difference from ln n tends to 0.577: the convergence to the Gumbel is slow, of order 1/n. For the uniform distribution, the mean of n(M − 1) is exactly −n/(n + 1): −0.667 at n = 2, almost −1 at n = 250. At n = 21 (one trading month) the means are already 0.593, 1.373 and −0.957. The values n ≠ 250 are extra runs of the lecture\'s code, with the same seed.'),
  'gev-ppqq': ('In the QQ plot of the GEV, the month with the largest loss of the S&P 500 (March 2020, 12.77%) has the fitted quantile 12.43%. The ▶ button switches to the Gumbel fit (ξ = 0). What fitted quantile does the same month get?',
               '6.99%: an exponential tail puts the most extreme month at half its observed value, and the top points of the QQ plot rise far above the diagonal. The 10-year level falls from 7.4% (GEV) to 5.3% (Gumbel), a level exceeded in 12 of the 440 months, more than three times as often as once in 10 years. The Gumbel fit is an extra run, for comparison; the 95% interval of ξ̂ in the lecture, [0.13, 0.29], excludes 0. For BET (ξ̂ = 0.36), the GEV puts the largest month at 26.1%, against 13.1% observed: the fitted tail is heavier than the data.'),
  'return-levels': ('The 10-year level of BET is 13.3%, and the largest loss observed in 28.5 years is 13.1%. The ▶ button moves T from 1 to 100 years. What is the 100-year level of BET?',
                    '32.2%, almost 2.5 times the largest observed loss. With ξ̂ = 0.36, the return level grows roughly as T<sup>ξ</sup>: every multiplication of T by 10 raises the level by a factor of about 2.4. It is an extrapolation to 3.5 times the length of the sample; already at 10 years the bootstrap interval is [10.3%, 17.5%]. S&P 500, with ξ̂ = 0.21: 13.4% at 100 years, close to the largest observed loss (12.8%).'),
  'gpd-densities': ('With ξ = 0.3, the GPD tail has tail index 1/ξ ≈ 3.3: the mean and the variance of the excess are finite. From which value of ξ does the variance of the excess become infinite?',
                    'From ξ = 1/2: the variance exists only for α = 1/ξ > 2. The mean excess, β/(1 − ξ), exists up to ξ = 1. For ξ = 0.5, P(Y > 10) = 0.028 (β = 1), against 0.0098 for ξ = 0.3 and 4.5 × 10<sup>−5</sup> for ξ = 0. The GPD fits of the lecture have ξ̂ between 0.12 and 0.22 for the four series: heavy tails with finite variance.'),
  'threshold-stability': ('At the 90% threshold, ξ̂ of the S&P 500 is 0.158 and the EVT VaR 1% 3.30%. The ▶ button raises the threshold from the 80% to the 99% quantile. What do ξ̂ and VaR 1% become at the 99% threshold (93 excesses)?',
                          'ξ̂ = 0.203, with a 95% interval from −0.07 to 0.47 (SE 0.138, four times larger than at 90%), but VaR 1% stays at 3.16%. VaR 1% lies inside the data (92 days beyond it in the sample), so it depends little on the shape of the tail; ξ̂ depends strongly on the threshold. Between the 85% and 95% quantiles: ξ̂ between 0.13 and 0.22, VaR 1% between 3.25% and 3.33%. BET at 99%: ξ̂ = −0.07, VaR 1% 4.35%.'),
  'gpd-ppqq': ('At the 90% threshold, the largest excess of the S&P 500 (11.60%, March 2020) is above the fitted GPD quantile (11.11%). The ▶ button raises the threshold from the 80% to the 98% quantile. What happens to this point of the QQ plot at the 98% threshold?',
               'It falls below the diagonal: the excess is 10.24% (the threshold rose to 2.52%), and the fitted quantile 12.65%, because ξ̂ rises to 0.255 on only 185 excesses. The last point of the QQ plot depends strongly on ξ̂, hence on the threshold. For BET, the GPD above the 90% quantile puts the largest excess at 18.56%, against 11.74% observed: the fitted tail is heavier than the largest losses of the sample.'),
  'tail-fit': ('The ▶ button lowers the level p from 2.5% to 0.01%. With p = 0.01% (one day in 10,000), what is the VaR of the S&P 500 by EVT and by the Normal distribution?',
               'EVT 10.79%, Normal 4.18%, 2.6 times less. At p = 1%, the ratio is 1.27 (3.30% against 2.60%). The sample of 9,240 days holds on average 0.9 days beyond VaR 0.01%, so the historical quantile (10.21%) is only an interpolation between the two largest losses. The Hill quantile gives 15.02%: far in the tail, the estimators differ a lot. BET: EVT 18.10%, Hill 26.16%, historical 12.24%.'),
  'oos': ('For the S&P 500, the VaR 1% of the Normal distribution is exceeded on 44 days of 2020–2026, against 16.9 expected. The ▶ button goes through the four series. How many times is the Normal VaR 1% of BET exceeded, where 16.8 exceedances are expected?',
          '13 times, below the expected value; the EVT VaR 4 times, the historical one 5 times. The test period was calmer for BET than the estimation period 1997–2019. A VaR estimated once does not follow the volatility: there are too many exceedances after 2020 for the S&P 500 (EVT 25, historical 26) and too few for BET. At VaR 0.1%, the Normal distribution is still exceeded too often for all four series: S&P 500 23 days against 1.7, BET 4 against 1.7, Bitcoin 8 against 2.5.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (checked with a node script on ch5_data.js + common.js + ch5.js).
PREDICT = {
 'crash-days': (0, [('17 zile: nivelul coboară doar pînă la −5,56%.', '17 days: the level only moves down to −5.56%.'),
                    ('Niciuna: un eveniment de 10 000 de ani nu apare în 37 de ani.', 'None: a 10,000-year event does not occur in 37 years.'),
                    ('Circa 0,35: de 100 de ori mai puține decît cele 35 de zile.', 'About 0.35: 100 times fewer than the 35 days.')]),
 'light-heavy': (1, [('Aproape 0: la x mare, orice coadă dispare, ca la distribuția Normală.', 'Almost 0: at a large x every tail vanishes, as for the Normal distribution.'),
                     ('0,125 = 2⁻³: dublarea pragului împarte probabilitatea la 2^α.', '0.125 = 2⁻³: doubling the threshold divides the probability by 2^α.'),
                     ('0,5: dublarea pragului înjumătățește probabilitatea.', '0.5: doubling the threshold halves the probability.')]),
 'loglog-real': (1, [('Tot circa −3,00: o lege de putere are aceeași pantă oriunde.', 'Still about −3.00: a power law has the same slope everywhere.'),
                     ('Circa −2,50: punctele din corpul distribuției aplatizează dreapta.', 'About −2.50: the points of the body flatten the line.'),
                     ('Circa −3,5: mai multe puncte fac panta mai abruptă.', 'About −3.5: more points make the slope steeper.')]),
 'hill-plot': (1, [('Tot circa 2,97, cu un interval mai îngust: mai multe pierderi, estimare mai precisă.', 'Still about 2.97, with a narrower interval: more losses, a more precise estimate.'),
                   ('Circa 2,04: corpul distribuției deplasează estimarea în jos.', 'About 2.04: the body of the distribution biases the estimate downwards.'),
                   ('Peste 4: cu pierderile mici incluse, coada pare mai subțire.', 'Above 4: with the small losses included, the tail looks lighter.')]),
 'hill-bvb': (2, [('Tot circa 1,0: lățimea intervalului depinde doar de α̂.', 'Still about 1.0: the width of the interval depends only on α̂.'),
                  ('Circa 2,6: lățimea crește ca 1/k, de 2,5 ori.', 'About 2.6: the width grows as 1/k, 2.5 times.'),
                  ('Circa 2,0: lățimea crește ca 1/√k, iar α̂ este mai mare.', 'About 2.0: the width grows as 1/√k, and α̂ is larger.')]),
 'mean-excess': (0, [('Mai mare, circa 1,22%: excesul mediu crește cu pragul.', 'Larger, about 1.22%: the mean excess grows with the threshold.'),
                     ('Tot circa 0,92%: excesul mediu nu depinde de prag.', 'Still about 0.92%: the mean excess does not depend on the threshold.'),
                     ('Mai mic, circa 0,5%: dincolo de un prag înalt rămîne mai puțin loc.', 'Smaller, about 0.5%: beyond a high threshold there is less room left.')]),
 'gev-types': (0, [('La x = 4 = 1/|ξ|.', 'At x = 4 = 1/|ξ|.'),
                   ('La x = 1: capătul scade odată cu |ξ|.', 'At x = 1: the endpoint shrinks with |ξ|.'),
                   ('Nicăieri: cu ξ aproape de 0, distribuția nu are capăt.', 'Nowhere: with ξ close to 0, the distribution has no endpoint.')]),
 'maxima-sim': (1, [('Tot circa 0,58: normalizarea M − ln n funcționează la orice n.', 'Still about 0.58: the normalisation M − ln n works for any n.'),
                    ('Circa 0,8: cu 2 extrageri limita nu este atinsă.', 'About 0.8: with 2 draws the limit is not reached.'),
                    ('Circa 1,5: media maximului a două extrageri exponențiale.', 'About 1.5: the mean of the maximum of two exponential draws.')]),
 'gev-ppqq': (2, [('Tot circa 12,4%: ambele modele se ajustează pe aceleași maxime.', 'Still about 12.4%: both models are fitted to the same maxima.'),
                  ('Peste 15%: fără ξ, ajustarea compensează cu o scală mai mare.', 'Above 15%: without ξ, the fit compensates with a larger scale.'),
                  ('Circa 7%: o coadă exponențială subestimează lunile extreme.', 'About 7%: an exponential tail understates the extreme months.')]),
 'return-levels': (1, [('Circa 13,1%: nivelul nu poate depăși cea mai mare pierdere observată.', 'About 13.1%: the level cannot exceed the largest observed loss.'),
                       ('Circa 32%: cu ξ̂ = 0,36, nivelul crește repede cu T.', 'About 32%: with ξ̂ = 0.36, the level grows fast with T.'),
                       ('Circa 133%: nivelul crește proporțional cu T.', 'About 133%: the level grows in proportion to T.')]),
 'gpd-densities': (0, [('De la ξ = 1/2: varianța cere α = 1/ξ > 2.', 'From ξ = 1/2: the variance needs α = 1/ξ > 2.'),
                       ('De la ξ = 1: atunci nu mai există nici media.', 'From ξ = 1: then the mean does not exist either.'),
                       ('De la orice ξ > 0: o coadă de tip Pareto are varianță infinită.', 'From any ξ > 0: a Pareto-type tail has infinite variance.')]),
 'threshold-stability': (2, [('Ambele aproape neschimbate: un prag mai înalt dă o estimare mai bună.', 'Both almost unchanged: a higher threshold gives a better estimate.'),
                             ('VaR 1% crește mult, pentru că ξ̂ crește.', 'VaR 1% rises a lot, because ξ̂ rises.'),
                             ('ξ̂ = 0,20 cu un interval larg (−0,07; 0,47), VaR 1% aproape neschimbat (3,16%).', 'ξ̂ = 0.20 with a wide interval (−0.07, 0.47), VaR 1% almost unchanged (3.16%).')]),
 'gpd-ppqq': (1, [('Rămîne deasupra bisectoarei: martie 2020 este prea extrem pentru orice GPD.', 'It stays above the diagonal: March 2020 is too extreme for any GPD.'),
                  ('Trece sub bisectoare: ξ̂ crește la 0,255, cuantila ajustată la 12,65%.', 'It falls below the diagonal: ξ̂ rises to 0.255, the fitted quantile to 12.65%.'),
                  ('Ajunge pe bisectoare: cu un prag mai înalt, GPD este exactă.', 'It lands on the diagonal: with a higher threshold the GPD is exact.')]),
 'tail-fit': (0, [('EVT circa 10,8%, Normal circa 4,2%.', 'EVT about 10.8%, Normal about 4.2%.'),
                  ('Ambele circa 4,2%: la probabilități foarte mici modelele se apropie.', 'Both about 4.2%: at very small probabilities the models get closer.'),
                  ('EVT circa 6,4%, ca la 0,1%: dincolo de date modelul nu mai crește.', 'EVT about 6.4%, as at 0.1%: beyond the data the model stops growing.')]),
 'oos': (2, [('Circa 44, ca la S&P 500: distribuția Normală subestimează riscul în orice perioadă.', 'About 44, as for the S&P 500: the Normal distribution understates the risk in any period.'),
             ('Circa 17: în afara eșantionului, numărul depășirilor se potrivește cu cel așteptat.', 'About 17: out of sample, the number of exceedances matches the expected one.'),
             ('13, sub cele 16,8 așteptate: anii de test au fost liniștiți pentru BET.', '13, below the 16.8 expected: the test years were calm for BET.')]),
}

# Exit ticket „Verifică-te” / "Check Yourself": questions of the slides „Autoevaluare” and „Formule de reținut”.
# Each option: (text RO, text EN), (why RO, why EN).
TICKET = [
 dict(q=('O ajustare GPD dă ξ̂ = 0,25. Ce momente ale pierderilor există?', 'A GPD fit gives ξ̂ = 0.25. Which moments of the losses exist?'),
      key=1, opts=[
  (('Doar media și varianța.', 'Only the mean and the variance.'), ('ξ < 1/2 garantează varianța, dar α = 1/ξ = 4 lasă finit și momentul de ordinul 3.', 'ξ < 1/2 guarantees the variance, but α = 1/ξ = 4 also leaves the third moment finite.')),
  (('Media, varianța și asimetria; boltirea nu.', 'The mean, the variance and the skewness; not the kurtosis.'), ('α = 1/ξ = 4: există momentele de ordin m < 4; boltirea (m = 4) este infinită.', 'α = 1/ξ = 4: the moments of order m < 4 exist; the kurtosis (m = 4) is infinite.')),
  (('Toate momentele: ξ este mic.', 'All moments: ξ is small.'), ('Orice ξ > 0 dă o coadă de tip putere, cu momente infinite de la ordinul 1/ξ în sus.', 'Any ξ > 0 gives a power-law tail, with infinite moments from order 1/ξ upwards.'))]),
 dict(q=('De ce folosim pentru estimatorul Hill cele mai mari 2,5% dintre pierderi și nu toate pierderile?', 'Why does the Hill estimator use the largest 2.5% of the losses and not all of them?'),
      key=0, opts=[
  (('Legea de putere este valabilă doar în coadă.', 'The power law holds only in the tail.'), ('Dacă intră corpul distribuției, α̂ este deplasat: la 10% din n, S&P 500 dă 2,04 în loc de 2,97.', 'If the body of the distribution enters, α̂ is biased: at 10% of n, the S&P 500 gives 2.04 instead of 2.97.')),
  (('Pentru ca eroarea standard să fie cît mai mică.', 'To make the standard error as small as possible.'), ('SE = α̂/√k scade cînd k crește; mai multe pierderi ar da o SE mai mică, dar o estimare deplasată.', 'SE = α̂/√k falls as k grows; more losses would give a smaller SE but a biased estimate.')),
  (('Pentru că pierderile mici sînt zgomot de măsurare.', 'Because small losses are measurement noise.'), ('Pierderile mici sînt date corecte; ele urmează alt tip de distribuție decît coada.', 'Small losses are correct data; they follow another kind of distribution than the tail.'))]),
 dict(q=('Este VaR 0,1% prin simulare istorică fiabil cu 10 ani de date?', 'Is a historical-simulation VaR 0.1% reliable with 10 years of data?'),
      key=2, opts=[
  (('Da: 2 500 de zile ajung pentru orice cuantilă.', 'Yes: 2,500 days are enough for any quantile.'), ('Cuantila de 0,1% depinde de cele cîteva zile din coadă, nu de numărul total de zile.', 'The 0.1% quantile depends on the few days in the tail, not on the total number of days.')),
  (('Da, dacă perioada include o criză.', 'Yes, if the period contains a crisis.'), ('O criză adaugă cîteva pierderi mari, dar estimarea rămîne bazată pe 2–3 puncte.', 'A crisis adds a few large losses, but the estimate still rests on 2–3 points.')),
  (('Nu: 2 500 de zile conțin doar 2–3 pierderi dincolo de el.', 'No: 2,500 days hold only 2–3 losses beyond it.'), ('2 500 × 0,001 = 2,5; EVT folosește întreaga coadă peste prag pentru a o extrapola.', '2,500 × 0.001 = 2.5; EVT uses the whole tail above the threshold to extrapolate it.'))]),
 dict(q=('Pierderile au o coadă Pareto cu α = 3 și P(L > 3%) = 1%. Ce pierdere este depășită cu probabilitatea 0,1%?', 'Losses have a Pareto tail with α = 3 and P(L > 3%) = 1%. Which loss is exceeded with probability 0.1%?'),
      key=0, opts=[
  (('6,46%.', '6.46%.'), ('1% × (x/3)⁻³ = 0,1% dă x = 3 × 10^(1/3) = 6,46%: o probabilitate de 10 ori mai mică înmulțește pierderea cu 10^(1/α).', '1% × (x/3)⁻³ = 0.1% gives x = 3 × 10^(1/3) = 6.46%: a probability 10 times smaller multiplies the loss by 10^(1/α).')),
  (('30%.', '30%.'), ('Aici pierderea crește de 10 ori, ca probabilitatea; cu α = 3 factorul este doar 10^(1/3) = 2,15.', 'This multiplies the loss by 10, as the probability; with α = 3 the factor is only 10^(1/3) = 2.15.')),
  (('9%.', '9%.'), ('9% = 3 × 3 înmulțește pragul cu α; pierderea depășită cu 0,1% este 3 × 10^(1/3).', '9% = 3 × 3 multiplies the threshold by α; the loss exceeded with 0.1% is 3 × 10^(1/3).'))]),
 dict(q=('S&P 500: u = 1,163%, N_u/n = 10%, ξ̂ = 0,158, β̂ = 0,769, β̂/ξ̂ = 4,860 și 0,1^(−0,158) = 1,439. Cît este VaR 1% prin POT?',
         'S&P 500: u = 1.163%, N_u/n = 10%, ξ̂ = 0.158, β̂ = 0.769, β̂/ξ̂ = 4.860 and 0.1^(−0.158) = 1.439. What is the POT VaR 1%?'),
      key=2, opts=[
  (('1,50%.', '1.50%.'), ('1,163 + 0,769 × 0,439 = 1,50 înmulțește cu β̂ în loc de β̂/ξ̂.', '1.163 + 0.769 × 0.439 = 1.50 multiplies by β̂ instead of β̂/ξ̂.')),
  (('2,13%.', '2.13%.'), ('4,860 × 0,439 = 2,13 este doar cuantila excesului; VaR adaugă pragul u.', '4.860 × 0.439 = 2.13 is only the quantile of the excess; VaR adds the threshold u.')),
  (('3,30%.', '3.30%.'), ('VaR = u + (β̂/ξ̂)[(np/N_u)^(−ξ̂) − 1] = 1,163 + 4,860 × 0,439 = 3,30%.', 'VaR = u + (β̂/ξ̂)[(np/N_u)^(−ξ̂) − 1] = 1.163 + 4.860 × 0.439 = 3.30%.'))]),
]


# ================================================================= table and worked-example blocks
# (anchor, frame number in the deck (same in RO and EN), section key, chart block that carries it on the web lecture,
#  title check EN, title check RO)
XBLOCKS = [('crash-table', 10, 'why', 'crash-days', 'Worst Days under the Normal Law', 'celor mai proaste zile'),
           ('hill-six', 21, 'hill', 'loglog-real', 'Hill on Six Losses', 'Hill pe șase pierderi'),
           ('hill-table', 25, 'hill', 'hill-plot', 'Hill Estimates for Four Markets', 'Estimări Hill pentru patru piețe'),
           ('bvb-table', 27, 'hill', 'hill-plot', 'Four Stocks of the Bucharest', 'Patru acțiuni de la Bursa'),
           ('me-table', 31, 'me', 'mean-excess', 'Reading the Mean Excess Plot', 'mean excess'),
           ('gev-table', 38, 'gev', 'maxima-sim', 'GEV Fits to Monthly Maxima', 'Ajustări GEV la maximele lunare'),
           ('rl-example', 40, 'gev', 'gev-ppqq', 'Return Levels', 'Return levels'),
           ('rl-table', 42, 'gev', 'return-levels', '10-Year Loss', 'pierderii de 10 ani'),
           ('gpd-table', 48, 'pot', 'threshold-stability', 'GPD Fits above the 90% Quantile', 'Ajustări GPD peste cuantila de 90%'),
           ('crash-evt', 52, 'pot', 'tail-fit', 'Waiting Times of the Worst Days', 'Timpii de așteptare'),
           ('var-example', 55, 'var', 'tail-fit', 'Step by Step', 'pas cu pas'),
           ('var-es', 56, 'var', 'tail-fit', 'Three Methods', 'trei metode'),
           ('var-far', 57, 'var', 'tail-fit', 'VaR at the 0.1', 'VaR la nivelul de 0,1'),
           ('exceed', 59, 'var', 'oos', 'Counting the Exceedances', 'Numărarea depășirilor')]
ORDER = ['crash-days', 'crash-table', 'light-heavy', 'loglog-real', 'hill-six', 'hill-plot', 'hill-table', 'bvb-table', 'hill-bvb', 'mean-excess', 'me-table',
         'gev-types', 'maxima-sim', 'gev-table', 'gev-ppqq', 'rl-example', 'return-levels', 'rl-table', 'gpd-densities', 'threshold-stability', 'gpd-table',
         'gpd-ppqq', 'tail-fit', 'crash-evt', 'var-example', 'var-es', 'var-far', 'oos', 'exceed']
PFX.update({'crash-table': 'ct', 'hill-six': 'hs', 'hill-table': 'ht', 'bvb-table': 'bt', 'me-table': 'mt', 'gev-table': 'gx', 'rl-example': 'rx',
            'rl-table': 'rt', 'gpd-table': 'gb', 'crash-evt': 'ce', 'var-example': 've', 'var-es': 'vt', 'var-far': 'vf', 'exceed': 'ex'})

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
 'ct': r'z = \frac{r - \hat\mu}{\hat\sigma}, \qquad p = \Phi(z), \qquad \text{ONCE} \;\; \frac{1}{p \times d} \;\; \text{YEARS} \quad (d = 252;\; 365 \text{ BTC})',
 'hs': r'\hat\xi = \frac{1}{k}\sum_{i=1}^{k} \ln\frac{X_{(i)}}{X_{(k+1)}}, \qquad \hat\alpha = \frac{1}{\hat\xi}, \qquad \mathrm{SE}(\hat\alpha) \approx \frac{\hat\alpha}{\sqrt{k}}',
 'rx': r'z_T = \mu - \frac{\sigma}{\xi}\Big[1 - \big(-\ln(1 - 1/m)\big)^{-\xi}\Big], \qquad m = 12\,T',
 'gb': r'P(L > x) \approx c\,x^{-1/\xi} \;\Rightarrow\; \alpha = \frac{1}{\xi}, \qquad \hat\xi \pm 1{.}96\,\mathrm{SE} \;\Rightarrow\; \alpha \in \Big[\frac{1}{\hat\xi + 1{.}96\,\mathrm{SE}},\; \frac{1}{\hat\xi - 1{.}96\,\mathrm{SE}}\Big]',
 've': r'\mathrm{VaR}_p = u + \frac{\beta}{\xi}\Big[\Big(\frac{n}{N_u}\,p\Big)^{-\xi} - 1\Big], \qquad \mathrm{ES}_p = \frac{\mathrm{VaR}_p + \beta - \xi u}{1 - \xi}',
 'ex': r'\begin{gathered}\mathrm{LR} = -2\ln\big[(1 - p)^{n - x} p^{x}\big] + 2\ln\big[(1 - \hat\pi)^{n - x}\hat\pi^{x}\big], \qquad \hat\pi = \frac{x}{n},\\ \mathrm{LR} \sim \chi^2_1: \;\text{REJ}\; \mathrm{LR} > 3{.}84\end{gathered}',
}


def wtex(k, ro):
    t = WE_TEX[k].replace('{.}', '{,}' if ro else '.')
    rep = {r'\text{ONCE}': (r'\text{o dată la}', r'\text{once in}'), r'\text{YEARS}': (r'\text{ani}', r'\text{years}'),
           r'\text{ BTC}': (r'\text{ pentru Bitcoin}', r'\text{ for Bitcoin}'),
           r'\text{REJ}': (r'\text{respins la 5}\%\text{ dacă}', r'\text{rejected at 5}\%\text{ if}')}
    for a, (r, e) in rep.items():
        t = t.replace(a, r if ro else e)
    return t


A4 = [('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('btc', 'Bitcoin')]
S4 = [('snp', 'OMV Petrom (SNP)'), ('brd', 'BRD'), ('tlv', 'Banca Transilvania (TLV)'), ('tgn', 'Transgaz (TGN)')]
VP = ['0.05', '0.025', '0.01', '0.005', '0.001', '0.0005', '0.0001']


def widgets_all(lang):
    ro = lang == 'ro'
    WE = 'Exemplu lucrat' if ro else 'Worked example'
    wt = lambda k, r, e: we(k + '-we', f'{WE}: {r if ro else e}', wtex(k, ro))
    ser = lambda i, r, e, d, opts=A4: select(i, r if ro else e, opts, d)
    lv = [(v, f'{float(v) * 100:g}%'.replace('.', ',' if ro else '.')) for v in VP]
    row = ('Seria (rîndul evidențiat)', 'Series (highlighted row)')
    W = widgets_old(lang)
    W['crash-days'] = (W['crash-days'][0], CSS + W['crash-days'][1])
    W.update({
     'crash-table': (ser('ct-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                     + slider('ct-k', 'Pragul k pentru zilele sub −kσ' if ro else 'Threshold k for the days below −kσ', 3, 6, 0.5, 4) + play('ct'),
                     tw('ct-tab') + wt('ct', 'perioada de revenire în distribuția Normală', 'the Normal return period')),
     'hill-six': (ser('hs-a', 'Seria', 'Series', 'sp500') + slider('hs-k', 'Numărul k al pierderilor din coadă' if ro else 'Number k of tail losses', 0, 7, 1, 0) + play('hs'),
                  xh('Cele mai mari pierderi zilnice și pragul X<sub>(k+1)</sub>' if ro else 'The largest daily losses and the threshold X<sub>(k+1)</sub>')
                  + tw('hs-tab') + wt('hs', 'estimatorul Hill pe pierderile reale', 'the Hill estimator on the real losses')),
     'hill-table': (ser('ht-a', *row, 'sp500') + slider('ht-k', 'Proporția k/n a pierderilor din coadă' if ro else 'Share k/n of the tail losses', 0, 8, 1, 3) + play('ht'),
                    tw('ht-tab') + charts('ht-chart')),
     'bvb-table': (ser('bt-a', *row, 'tlv', S4) + slider('bt-k', 'Proporția k/n a pierderilor din coadă' if ro else 'Share k/n of the tail losses', 0, 8, 1, 3) + play('bt'),
                   tw('bt-tab') + charts('bt-chart')),
     'me-table': (ser('mt-a', *row, 'sp500') + slider('mt-q', 'Pragul u: cuantila pierderilor' if ro else 'Threshold u: quantile of the losses', 80, 98, 1, 90) + play('mt'),
                  tw('mt-tab')),
     'gev-table': (ser('gx-a', *row, 'sp500') + slider('gx-b', 'Mărimea blocului' if ro else 'Block size', 0, 3, 1, 1) + play('gx'), tw('gx-tab') + charts('gx-chart')),
     'rl-example': (ser('rx-a', 'Seria', 'Series', 'sp500') + slider('rx-t', 'Perioada de revenire T' if ro else 'Return period T', 0, 6, 1, 3) + play('rx'),
                    wt('rx', 'return level din GEV', 'a GEV return level')),
     'rl-table': (ser('rt-a', 'Seria (rîndul evidențiat și graficul)', 'Series (highlighted row and chart)', 'sp500')
                  + slider('rt-t', 'Perioada de revenire T (coloana a treia, linia punctată)' if ro else 'Return period T (third column, dotted line)', 0, 6, 1, 3) + play('rt'),
                  tw('rt-tab') + charts('rt-chart')),
     'gpd-table': (ser('gb-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'dax')
                   + slider('gb-q', 'Pragul u: cuantila pierderilor' if ro else 'Threshold u: quantile of the losses', 80, 98, 1, 90) + play('gb'),
                   tw('gb-tab') + wt('gb', 'tail index-ul α = 1/ξ', 'the tail index α = 1/ξ')),
     'crash-evt': (slider('ce-q', 'Pragul u al ajustării GPD: cuantila pierderilor' if ro else 'Threshold u of the GPD fit: quantile of the losses', 80, 98, 1, 90) + play('ce'),
                   tw('ce-tab')),
     'var-example': (ser('ve-a', 'Seria', 'Series', 'sp500') + select('ve-p', 'Nivelul p al VaR' if ro else 'Level p of VaR', lv, '0.01')
                     + select('ve-e', 'Nivelul p al ES' if ro else 'Level p of ES', lv, '0.025') + play('ve'),
                     wt('ve', 'VaR și ES din GPD', 'VaR and ES from the GPD')),
     'var-es': (ser('vt-a', *row, 'sp500') + select('vt-p', 'Nivelul VaR' if ro else 'VaR level', lv, '0.01') + select('vt-e', 'Nivelul ES' if ro else 'ES level', lv, '0.025') + play('vt'),
                tw('vt-tab')),
     'var-far': (ser('vf-a', 'Seria (rîndul evidențiat și graficul)', 'Series (highlighted row and chart)', 'sp500', A4 + S4)
                 + select('vf-p', 'Nivelul p al VaR' if ro else 'Level p of VaR', lv, '0.001') + play('vf'),
                 tw('vf-tab') + charts('vf-chart')),
     'exceed': (ser('ex-a', 'Seria (rîndul evidențiat și exemplul lucrat)', 'Series (highlighted row and worked example)', 'sp500')
                + select('ex-p', 'Nivelul' if ro else 'Level', [('var1', 'VaR 1%'), ('var01', 'VaR 0,1%' if ro else 'VaR 0.1%')], 'var1')
                + select('ex-m', 'Metoda din exemplul lucrat' if ro else 'Method of the worked example', [('EVT', 'EVT'), ('Historical', 'istoric' if ro else 'historical'), ('Normal', 'Normal')], 'EVT')
                + play('ex'),
                tw('ex-tab') + charts('ex-chart') + wt('ex', 'testul Kupiec', 'the Kupiec test')),
    })
    return W


TEXT_JS['ro'] += r"""(function () {
const LUNI = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const de = s => { const n = +String(s).replace(/[^0-9]/g, ''); return n % 100 >= 20 || (n >= 100 && n % 100 === 0) ? ' de' : ''; };
const zile = n => n === 0 ? 'nicio zi' : n === 1 ? 'o zi' : `${n}${de(n)} zile`;
const luni = n => n === 0 ? 'nicio lună' : n === 1 ? 'o lună' : `${n}${de(n)} luni`;
Object.assign(window.CH5_TEXT, {
 date: s => `${+s.slice(8, 10)} ${LUNI[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 h: { series: 'Seria', worst: 'cea mai proastă zi', lossPct: 'pierdere (%)', normOnce: 'distribuția Normală: o dată la (ani)', below: k => `zile sub −${k}σ`, expected: 'așteptat',
   day: 'ziua', ofN: 'din n', thrRow: 'pragul', ci95: 'IÎ 95%', seIid: 'SE i.i.d.', seBlock: 'SE bloc', gains: 'cîștiguri', meRatio: 'e(u)/u', slopeEmp: 'panta lui e(u) peste u',
   slopeGpd: 'panta GPD ξ̂/(1 − ξ̂)', alphaSlope: '1 + 1/pantă', blocks: 'blocuri', week: 'săptămînale', month: 'lunare', quarter: 'trimestriale', year: 'anuale',
   ciBoot: 'interval de 95%', largest: 'cea mai mare observată', yearsData: 'ani de date', hill25: 'Hill α̂ (2,5%)', onceIn: 'o dată la (ani)', normLaw: 'distribuția Normală',
   bm: 'S&P 500, Black Monday', hist: 'istoric', norm: 'Normal', largestL: 'cea mai mare pierdere', beyond: 'zile așteptate dincolo de VaR', exc: 'depășiri',
   kupiec: 'valoarea p Kupiec', testDays: 'zile de test', m: { EVT: 'EVT', Historical: 'ist.', Normal: 'Normal' } },
 ctNote: 'z = (r − μ̂)/σ̂, cu media și abaterea standard ale întregului eșantion; o dată la 1/(Φ(z) × 252) ani (365 pentru Bitcoin), rotunjit la o putere a lui 10. Așteptat: n Φ(−k), numărul de zile sub −kσ în distribuția Normală.',
 ct: (a, k, nb, ex, r) => `${a}: ${zile(nb)} sub −${k}σ, față de ${ex} așteptate în distribuția Normală (de ${r} ori mai multe).`,
 ctw1: (a, n, mu, sd) => `${a}: n = ${n} de randamente logaritmice zilnice, media μ̂ = ${mu}%, abaterea standard σ̂ = ${sd}%.`,
 ctw2: (r, mu, sd, z) => `Cea mai proastă zi: z = (${r} − ${mu})/${sd} = ${z}.`,
 ctw3: (z, m, e) => `p = Φ(${z}) = ${m} × 10<sup>−${e}</sup>: probabilitatea, în distribuția Normală, a unei zile cel puțin la fel de proaste.`,
 ctw4: (d, ly, r) => `O dată la 1/(p × ${d}) = 10<sup>${ly}</sup> ≈ 10<sup>${r}</sup> ani.`,
 ctw5: (k, p, d, y) => `Pentru comparație, o pierdere sub −${k}σ: p = Φ(−${k}) = ${p}, adică o dată la 1/(p × ${d}) = ${y} ani în distribuția Normală.`,
 hsNote: 'Cele mai mari pierderi zilnice, în ordine descrescătoare; rîndul k + 1 este pragul X<sub>(k+1)</sub>. Sînt afișate doar primele cinci rapoarte logaritmice. Slide-ul folosește șase pierderi ilustrative (9,5; 7,2; 6,1; 5,4; 4,8; 4,0); aici aceiași pași se aplică celor mai mari pierderi reale.',
 hs: (a, k, xi, al, se, a25) => `${a}, k = ${k}: ξ̂ = ${xi}, tail index-ul α̂ = 1/ξ̂ = ${al}, SE ${se}; la k = 2,5% din n: α̂ = ${a25}.`,
 hsw1: (a, n, k, th) => `${a}: n = ${n} de pierderi zilnice, k = ${k}; pragul este a (k + 1)-a cea mai mare pierdere, X<sub>(k+1)</sub> = ${th}%.`,
 hsw2: (t, s) => `Rapoartele logaritmice ln(X<sub>(i)</sub>/X<sub>(k+1)</sub>): ${t}; suma lor este ${s}.`,
 hsw3: (s, k, xi) => `ξ̂ = ${s}/${k} = ${xi}.`,
 hsw4: (xi, al) => `α̂ = 1/ξ̂ = 1/${xi} = ${al}.`,
 hsw5: (al, k, sk, se) => `SE(α̂) ≈ α̂/√k = ${al}/√${k} = ${al}/${sk} = ${se}.`,
 htNote: (f, four) => `k = ${f} din n; IÎ 95%: α̂(1 ± 1,96/√k); SE bloc: abaterea standard a lui α̂ în 500 de eșantioane moving-block bootstrap (blocuri de 20 de zile, sămînța 2026); cîștiguri: indicele Hill al randamentelor r<sub>t</sub> &gt; 0, cu același k.` + (four ? ' k = 1% / 5%: estimările la aceste proporții.' : ''),
 ht: (a, f, k, al, se, sb, r, g) => `${a}, k = ${f} din n (${k} pierderi): α̂ = ${al}; SE i.i.d. ${se}, SE bloc ${sb} (raportul ${r}); indicele Hill al cîștigurilor ${g}.`,
 mtNote: 'e(u): media lui L − u pe pierderile peste u. Panta lui e(u) peste u: dreapta celor mai mici pătrate prin e(u) empiric la cuantilele pierderilor de la u la 99,5% (punctele graficului mean excess). Pentru o coadă Pareto cu indicele α, e(u) = u/(α − 1): panta este 1/(α − 1), deci α = 1 + 1/pantă; pentru GPD ajustată peste u, panta este ξ̂/(1 − ξ̂).',
 mt: (a, q, u, e, b, xi, g) => `${a}, u = cuantila de ${q}% = ${u}%: e(u) = ${e}%; panta lui e(u) empiric peste u ${b}, panta GPD ξ̂/(1 − ξ̂) = ${g} (ξ̂ = ${xi}).`,
 gxNote: (b, mn) => `Verosimilitate maximă pe maximele ${b} ale pierderilor zilnice; IÎ 95%: ξ̂ ± 1,96 SE. Blocurile cu mai puțin de ${mn} zile sînt eliminate. Slide-ul folosește maximele lunare; celelalte mărimi ale blocului sînt rulări suplimentare ale aceleiași ajustări.`,
 gx: (a, b, n, xi, ci, xm, isM) => `${a}, maxime ${b} (${n} blocuri): ξ̂ = ${xi}, IÎ 95% ${ci}` + (isM ? '.' : `; cu maxime lunare (slide-ul): ξ̂ = ${xm}.`)
   + (b === 'săptămînale' ? ' Blocurile de 5 zile sînt departe de limita teoremei: multe maxime săptămînale sînt zile obișnuite, unele chiar cîștiguri (5% din săptămîni la S&P 500, 7% la BET), iar ξ̂ urmează corpul distribuției.' : ''),
 rxw1: (a, xi, mu, s) => `${a}, GEV ajustată maximelor lunare: ξ̂ = ${xi}, μ̂ = ${mu}%, σ̂ = ${s}%.`,
 rxw2: (T, m) => `T = ${T}: m = 12T = ${m} luni, deci nivelul este cuantila 1 − 1/${m} a maximului lunar.`,
 rxw3: (m, a, xi, b) => `−ln(1 − 1/${m}) = ${a}; ridicat la puterea −ξ̂ = −${xi}: ${b}.`,
 rxw4: (mu, s, xi, b, z, sx) => `z<sub>T</sub> = ${mu} − (${s}/${xi})(1 − ${b}) = ${mu} + ${sx} × (${b} − 1) = ${z}%.`,
 rxw5: (z, na, n, y, ex) => `${luni(na)[0].toUpperCase() + luni(na).slice(1)} din cele ${n} maxime lunare (${y} ani) ${na === 1 ? 'este' : 'sînt'} peste ${z}%; GEV așteaptă ${n}/m = ${ex}.`,
 rx: (a, T, z, na, ex) => `${a}: return level pentru T = ${T}: ${z}%; depășit în ${luni(na)}, ${ex} așteptate.`,
 rtNote: 'Pierderea zilnică (%) depășită de maximul lunar, în medie, o dată la T ani (GEV din slide); interval de 95%: percentilele de 2,5% și 97,5% ale nivelului în cele 200 de eșantioane bootstrap ale maximelor lunare din curs.',
 rt: (a, T, z, ci, r, rm) => `${a}, T = ${T}: ${z}%, intervalul bootstrap de 95% ${ci}; limita de sus este de ${r} ori cea de jos. Raportul dintre nivel și cea mai mare pierdere observată: ${rm}.`,
 gbNote: 'Verosimilitate maximă pe excesele L − u; SE din curbura log-verosimilității. 1/ξ̂ este calculat din ξ̂ nerotunjit. Hill α̂: k = 2,5% din n.',
 gb: (a, q, xi, al, h) => `${a}, u = cuantila de ${q}%: ξ̂ = ${xi}, 1/ξ̂ = ${al}; Hill α̂ = ${h}.`,
 gbw1: (a, q, u, nu, xi, se) => `${a}, u = cuantila de ${q}% = ${u}%: N<sub>u</sub> = ${nu} excese, ξ̂ = ${xi} (SE ${se}).`,
 gbw2: (xi, al, xr, alr) => al ? `Tail index-ul α = 1/ξ̂ = 1/${xi} = ${al}; din ξ̂ rotunjit la ${xr}: 1/${xr} = ${alr}.` : 'ξ̂ ≤ 0: coada nu este de tip putere, α nu este definit.',
 gbw3: (ci, lo, hi) => lo ? `IÎ 95% al lui ξ̂: ${ci}; inversat, α între ${lo} și ${hi}.` : `IÎ 95% al lui ξ̂: ${ci}; conține 0, deci intervalul lui α nu este mărginit superior.`,
 gbw4: (h, xh) => `Hill la k = 2,5% din n: α̂ = ${h}, adică 1/α̂ = ${xh} pe scala lui ξ.`,
 ceNote: q => `Perioada de așteptare EVT: 1/(F̂(x) × 252) ani (365 pentru Bitcoin), cu GPD ajustată peste cuantila de ${q}% a întregului eșantion. Distribuția Normală: 1/(Φ(z) × 252) ani, rotunjit la o putere a lui 10. Black Monday: randamentul logaritmic ln(1 − 0,20) = −22,3%, în afara eșantionului.`,
 ce: (q, sp, bet, y87, xi) => `GPD peste cuantila de ${q}%: 16 martie 2020 (S&P 500) o dată la ${sp} ani, 7 ianuarie 2009 (BET) o dată la ${bet} ani, 19 octombrie 1987 o dată la ${y87} ani (ξ̂ al S&P 500 ${xi}).`,
 vew1: (a, n, u, nu, sh, xi, b, bx) => `${a}: n = ${n}, u = ${u}%, N<sub>u</sub> = ${nu} (${sh}%), ξ̂ = ${xi}, β̂ = ${b}, β̂/ξ̂ = ${bx}.`,
 vew2: (p, r, f, u, bx, v) => `VaR ${p}: np/N<sub>u</sub> = ${r}; ${r}<sup>−ξ̂</sup> = ${f}; VaR = ${u} + ${bx} × (${f} − 1) = ${v}%.`,
 vew3: (p, r, f, u, bx, v) => `VaR ${p}, necesar pentru ES ${p}: np/N<sub>u</sub> = ${r}, ${r}<sup>−ξ̂</sup> = ${f}, VaR = ${u} + ${bx} × (${f} − 1) = ${v}%.`,
 vew4: (p, v, b, xi, u, es, r, lim) => `ES ${p} = (${v} + ${b} − ${xi} × ${u})/(1 − ${xi}) = ${es}%; ES/VaR = ${r} (limita departe în coadă: 1/(1 − ξ̂) = ${lim}).`,
 ve: (a, p, v, pe, es) => `${a}: VaR ${p} = ${v}%, ES ${pe} = ${es}%.`,
 vtNote: 'EVT: GPD peste cuantila de 90%; istoric: cuantila empirică și media pierderilor egale cu ea sau mai mari; Normal: media și abaterea standard ale pierderilor.',
 vt: (a, p, dv, upV, pe, de2, upE, dm) => `${a}: VaR ${p} EVT este cu ${dv}% ${upV ? 'peste' : 'sub'} cel Normal, ES ${pe} EVT cu ${de2}% ${upE ? 'peste' : 'sub'} cel Normal. Cea mai mare diferență între EVT și istoric, pe cele patru piețe: ${dm} puncte procentuale.`
   + (upV ? '' : ' La acest nivel VaR Normal este mai mare: pierderile cu cozi groase sînt și mai concentrate în jurul lui 0, astfel că cuantila empirică de 5% este sub cea a distribuției Normale cu aceeași abatere standard.'),
 vfNote: p => `VaR ${p} (%): pierderea depășită cu probabilitatea ${p} într-o zi. EVT: GPD peste cuantila de 90%; Hill: cuantila Weissman cu k = 2,5% din n; ultima coloană: n × p, numărul de zile din eșantion așteptate dincolo de VaR. Acțiunile BVB din 2010.`,
 vf: (a, p, e, nm, r, nb, dh) => `${a}, VaR ${p}: EVT ${e}%, Normal ${nm}% (raportul ${r}); ${nb} zile din eșantion așteptate dincolo de el; Hill minus EVT: ${dh} puncte procentuale.`,
 exNote: 'VaR estimat pe pierderile pînă la 31 decembrie 2019 și testat pe toate zilele de după (pentru DAX și BET, 1 711 și 1 675 de zile de test, de la prima ședință din 2020). Testul Kupiec (1995) de acoperire necondiționată (<i>unconditional coverage</i>): raportul de verosimilitate (LR, <i>likelihood ratio</i>) are distribuția χ² cu 1 grad de libertate; o valoare p sub 0,05 înseamnă că ponderea depășirilor diferă de p, în oricare direcție.',
 ex: (p, e, h, nm) => `VaR ${p}: testul Kupiec respinge acoperirea corectă la 5% pentru ${e} din cele 4 piețe cu EVT, ${h} cu simularea istorică și ${nm} cu distribuția Normală.`,
 exw1: (a, m, p, v, n, x, ex) => `${a}, VaR ${p} ${m} = ${v}% (estimat pînă în 2019): n = ${n} zile de test, x = ${x} depășiri, ${ex} așteptate.`,
 exw2: (n, x, ph, p) => `Ponderea observată π̂ = x/n = ${x}/${n} = ${ph}, față de p = ${p}.`,
 exw3: (nx, p, x, l1p, lp, lph, l1ph, lr) => `LR = −2[${nx} ln(1 − p) + ${x} ln p] + 2[${nx} ln(1 − π̂) + ${x} ln π̂], cu ln(1 − p) = ${l1p}, ln p = ${lp}` + (lph ? `, ln π̂ = ${lph}, ln(1 − π̂) = ${l1ph}` : ' și x ln π̂ = 0 pentru x = 0') + `: LR = ${lr}.`,
 exw4: (lr, pv, rej, lo, hi) => `Valoarea p = P(χ²<sub>1</sub> &gt; ${lr}) = ${pv}: ` + (rej ? 'acoperirea corectă este respinsă la 5% (LR &gt; 3,84).' : 'acoperirea corectă nu este respinsă la 5% (LR ≤ 3,84).') + ` Cu aceste zile de test, trec testul numerele de depășiri de la ${lo} la ${hi}.`
});
})();
"""

TEXT_JS['en'] += r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const days = n => n === 1 ? '1 day' : `${n} days`;
const months = n => n === 1 ? '1 month' : `${n} months`;
Object.assign(window.CH5_TEXT, {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 h: { series: 'Series', worst: 'worst day', lossPct: 'loss (%)', normOnce: 'Normal: once in (years)', below: k => `days below −${k}σ`, expected: 'expected',
   day: 'day', ofN: 'of n', thrRow: 'threshold', ci95: '95% CI', seIid: 'SE i.i.d.', seBlock: 'SE block', gains: 'gains', meRatio: 'e(u)/u', slopeEmp: 'slope of e(u) above u',
   slopeGpd: 'GPD slope ξ̂/(1 − ξ̂)', alphaSlope: '1 + 1/slope', blocks: 'blocks', week: 'weekly', month: 'monthly', quarter: 'quarterly', year: 'yearly',
   ciBoot: '95% interval', largest: 'largest observed', yearsData: 'years of data', hill25: 'Hill α̂ (2.5%)', onceIn: 'once in (years)', normLaw: 'Normal law',
   bm: 'S&P 500, Black Monday', hist: 'historical', norm: 'Normal', largestL: 'largest loss', beyond: 'expected days beyond VaR', exc: 'exceedances',
   kupiec: 'Kupiec p-value', testDays: 'test days', m: { EVT: 'EVT', Historical: 'hist.', Normal: 'Normal' } },
 ctNote: 'z = (r − μ̂)/σ̂, with the mean and the standard deviation of the whole sample; once in 1/(Φ(z) × 252) years (365 for Bitcoin), rounded to a power of 10. Expected: n Φ(−k), the number of days below −kσ under the Normal distribution.',
 ct: (a, k, nb, ex, r) => `${a}: ${days(nb)} below −${k}σ, against ${ex} expected under the Normal distribution (${r} times more).`,
 ctw1: (a, n, mu, sd) => `${a}: n = ${n} daily log returns, mean μ̂ = ${mu}%, standard deviation σ̂ = ${sd}%.`,
 ctw2: (r, mu, sd, z) => `Worst day: z = (${r} − ${mu})/${sd} = ${z}.`,
 ctw3: (z, m, e) => `p = Φ(${z}) = ${m} × 10<sup>−${e}</sup>: the Normal probability of a day at least this bad.`,
 ctw4: (d, ly, r) => `Once in 1/(p × ${d}) = 10<sup>${ly}</sup> ≈ 10<sup>${r}</sup> years.`,
 ctw5: (k, p, d, y) => `For comparison, a loss below −${k}σ: p = Φ(−${k}) = ${p}, i.e. once in 1/(p × ${d}) = ${y} years under the Normal distribution.`,
 hsNote: 'The largest daily losses in decreasing order; the row k + 1 is the threshold X<sub>(k+1)</sub>. Only the first five log ratios are listed. The slide uses six illustrative losses (9.5, 7.2, 6.1, 5.4, 4.8, 4.0); here the same steps are applied to the largest real losses.',
 hs: (a, k, xi, al, se, a25) => `${a}, k = ${k}: ξ̂ = ${xi}, tail index α̂ = 1/ξ̂ = ${al}, SE ${se}; at k = 2.5% of n: α̂ = ${a25}.`,
 hsw1: (a, n, k, th) => `${a}: n = ${n} daily losses, k = ${k}; the threshold is the (k + 1)-th largest loss, X<sub>(k+1)</sub> = ${th}%.`,
 hsw2: (t, s) => `Log ratios ln(X<sub>(i)</sub>/X<sub>(k+1)</sub>): ${t}; their sum is ${s}.`,
 hsw3: (s, k, xi) => `ξ̂ = ${s}/${k} = ${xi}.`,
 hsw4: (xi, al) => `α̂ = 1/ξ̂ = 1/${xi} = ${al}.`,
 hsw5: (al, k, sk, se) => `SE(α̂) ≈ α̂/√k = ${al}/√${k} = ${al}/${sk} = ${se}.`,
 htNote: (f, four) => `k = ${f} of n; 95% CI: α̂(1 ± 1.96/√k); SE block: standard deviation of α̂ in 500 moving-block bootstrap samples (blocks of 20 days, seed 2026); gains: Hill index of the returns r<sub>t</sub> &gt; 0, with the same k.` + (four ? ' k = 1% / 5%: the estimates at these shares.' : ''),
 ht: (a, f, k, al, se, sb, r, g) => `${a}, k = ${f} of n (${k} losses): α̂ = ${al}; SE i.i.d. ${se}, SE block ${sb} (ratio ${r}); Hill index of the gains ${g}.`,
 mtNote: 'e(u): the mean of L − u over the losses above u. Slope of e(u) above u: least-squares line through the empirical e(u) at the quantiles of the losses from u to 99.5% (the points of the mean excess chart). For a Pareto tail with index α, e(u) = u/(α − 1): the slope is 1/(α − 1), so α = 1 + 1/slope; for the GPD fitted above u, the slope is ξ̂/(1 − ξ̂).',
 mt: (a, q, u, e, b, xi, g) => `${a}, u = ${q}% quantile = ${u}%: e(u) = ${e}%; slope of the empirical e(u) above u ${b}, GPD slope ξ̂/(1 − ξ̂) = ${g} (ξ̂ = ${xi}).`,
 gxNote: (b, mn) => `Maximum likelihood on the ${b} maxima of the daily losses; 95% CI: ξ̂ ± 1.96 SE. Blocks with fewer than ${mn} days are dropped. The slide uses monthly maxima; the other block sizes are extra runs of the same fit.`,
 gx: (a, b, n, xi, ci, xm, isM) => `${a}, ${b} maxima (${n} blocks): ξ̂ = ${xi}, 95% CI ${ci}` + (isM ? '.' : `; with monthly maxima (the slide): ξ̂ = ${xm}.`)
   + (b === 'weekly' ? ' Blocks of 5 days are far from the limit of the theorem: many weekly maxima are ordinary days, some even gains (5% of the weeks for the S&P 500, 7% for BET), and ξ̂ follows the body of the distribution.' : ''),
 rxw1: (a, xi, mu, s) => `${a}, GEV fitted to the monthly maxima: ξ̂ = ${xi}, μ̂ = ${mu}%, σ̂ = ${s}%.`,
 rxw2: (T, m) => `T = ${T}: m = 12T = ${m} months, so the level is the 1 − 1/${m} quantile of the monthly maximum.`,
 rxw3: (m, a, xi, b) => `−ln(1 − 1/${m}) = ${a}; raised to the power −ξ̂ = −${xi}: ${b}.`,
 rxw4: (mu, s, xi, b, z, sx) => `z<sub>T</sub> = ${mu} − (${s}/${xi})(1 − ${b}) = ${mu} + ${sx} × (${b} − 1) = ${z}%.`,
 rxw5: (z, na, n, y, ex) => `${na} of the ${n} monthly maxima (${y} years) ${na === 1 ? 'is' : 'are'} above ${z}%; the GEV expects ${n}/m = ${ex}.`,
 rx: (a, T, z, na, ex) => `${a}: return level for T = ${T}: ${z}%; exceeded in ${months(na)}, ${ex} expected.`,
 rtNote: 'Daily loss (%) exceeded by the monthly maximum on average once in T years (GEV of the slide); 95% interval: the 2.5% and 97.5% percentiles of the level over the 200 bootstrap resamples of the monthly maxima of the lecture.',
 rt: (a, T, z, ci, r, rm) => `${a}, T = ${T}: ${z}%, 95% bootstrap interval ${ci}; the upper bound is ${r} times the lower one. Level relative to the largest observed loss: ${rm}.`,
 gbNote: 'Maximum likelihood on the excesses L − u; SE from the curvature of the log-likelihood. 1/ξ̂ is computed from the unrounded ξ̂. Hill α̂: k = 2.5% of n.',
 gb: (a, q, xi, al, h) => `${a}, u = ${q}% quantile: ξ̂ = ${xi}, 1/ξ̂ = ${al}; Hill α̂ = ${h}.`,
 gbw1: (a, q, u, nu, xi, se) => `${a}, u = ${q}% quantile = ${u}%: N<sub>u</sub> = ${nu} excesses, ξ̂ = ${xi} (SE ${se}).`,
 gbw2: (xi, al, xr, alr) => al ? `Tail index α = 1/ξ̂ = 1/${xi} = ${al}; from ξ̂ rounded to ${xr}: 1/${xr} = ${alr}.` : 'ξ̂ ≤ 0: the tail is not a power law, α is not defined.',
 gbw3: (ci, lo, hi) => lo ? `95% CI of ξ̂: ${ci}; inverted, α between ${lo} and ${hi}.` : `95% CI of ξ̂: ${ci}; it contains 0, so the interval of α has no upper bound.`,
 gbw4: (h, xh) => `Hill at k = 2.5% of n: α̂ = ${h}, i.e. 1/α̂ = ${xh} on the scale of ξ.`,
 ceNote: q => `EVT waiting time: 1/(F̂(x) × 252) years (365 for Bitcoin), with the GPD fitted above the ${q}% quantile of the whole sample. Normal law: 1/(Φ(z) × 252) years, rounded to a power of 10. Black Monday: log return ln(1 − 0.20) = −22.3%, outside the sample.`,
 ce: (q, sp, bet, y87, xi) => `GPD above the ${q}% quantile: 16 March 2020 (S&P 500) once in ${sp} years, 7 January 2009 (BET) once in ${bet} years, 19 October 1987 once in ${y87} years (ξ̂ of the S&P 500 ${xi}).`,
 vew1: (a, n, u, nu, sh, xi, b, bx) => `${a}: n = ${n}, u = ${u}%, N<sub>u</sub> = ${nu} (${sh}%), ξ̂ = ${xi}, β̂ = ${b}, β̂/ξ̂ = ${bx}.`,
 vew2: (p, r, f, u, bx, v) => `VaR ${p}: np/N<sub>u</sub> = ${r}; ${r}<sup>−ξ̂</sup> = ${f}; VaR = ${u} + ${bx} × (${f} − 1) = ${v}%.`,
 vew3: (p, r, f, u, bx, v) => `VaR ${p}, needed for ES ${p}: np/N<sub>u</sub> = ${r}, ${r}<sup>−ξ̂</sup> = ${f}, VaR = ${u} + ${bx} × (${f} − 1) = ${v}%.`,
 vew4: (p, v, b, xi, u, es, r, lim) => `ES ${p} = (${v} + ${b} − ${xi} × ${u})/(1 − ${xi}) = ${es}%; ES/VaR = ${r} (limit far in the tail: 1/(1 − ξ̂) = ${lim}).`,
 ve: (a, p, v, pe, es) => `${a}: VaR ${p} = ${v}%, ES ${pe} = ${es}%.`,
 vtNote: 'EVT: GPD above the 90% quantile; historical: the empirical quantile and the mean of the losses at or beyond it; Normal: the mean and the standard deviation of the losses.',
 vt: (a, p, dv, upV, pe, de2, upE, dm) => `${a}: the EVT VaR ${p} is ${dv}% ${upV ? 'above' : 'below'} the Normal one, the EVT ES ${pe} ${de2}% ${upE ? 'above' : 'below'} it. Largest gap between EVT and historical over the four markets: ${dm} percentage points.`
   + (upV ? '' : ' At this level the Normal VaR is higher: heavy-tailed losses are also more concentrated around 0, so the empirical 5% quantile lies below that of the Normal distribution with the same standard deviation.'),
 vfNote: p => `VaR ${p} (%): the loss exceeded with probability ${p} on a given day. EVT: GPD above the 90% quantile; Hill: Weissman quantile with k = 2.5% of n; last column: n × p, the number of days of the sample expected beyond VaR. BVB stocks since 2010.`,
 vf: (a, p, e, nm, r, nb, dh) => `${a}, VaR ${p}: EVT ${e}%, Normal ${nm}% (ratio ${r}); ${nb} days of the sample expected beyond it; Hill minus EVT: ${dh} percentage points.`,
 exNote: 'VaR estimated on the losses until 31 December 2019 and tested on all the days after it (for DAX and BET, 1,711 and 1,675 test days, from the first session of 2020). Kupiec (1995) test of unconditional coverage: the likelihood ratio (LR) is χ² with 1 degree of freedom; a p-value below 0.05 means that the share of exceedances differs from p, in either direction.',
 ex: (p, e, h, nm) => `VaR ${p}: the Kupiec test rejects correct coverage at 5% for ${e} of the 4 markets with EVT, ${h} with historical simulation and ${nm} with the Normal distribution.`,
 exw1: (a, m, p, v, n, x, ex) => `${a}, ${m} VaR ${p} = ${v}% (estimated until 2019): n = ${n} test days, x = ${x} exceedances, ${ex} expected.`,
 exw2: (n, x, ph, p) => `Observed share π̂ = x/n = ${x}/${n} = ${ph}, against p = ${p}.`,
 exw3: (nx, p, x, l1p, lp, lph, l1ph, lr) => `LR = −2[${nx} ln(1 − p) + ${x} ln p] + 2[${nx} ln(1 − π̂) + ${x} ln π̂], with ln(1 − p) = ${l1p}, ln p = ${lp}` + (lph ? `, ln π̂ = ${lph}, ln(1 − π̂) = ${l1ph}` : ' and x ln π̂ = 0 for x = 0') + `: LR = ${lr}.`,
 exw4: (lr, pv, rej, lo, hi) => `p-value P(χ²<sub>1</sub> &gt; ${lr}) = ${pv}: ` + (rej ? 'correct coverage is rejected at 5% (LR &gt; 3.84).' : 'correct coverage is not rejected at 5% (LR ≤ 3.84).') + ` With these test days, counts from ${lo} to ${hi} pass the test.`
});
})();
"""

QA['ro'].update({
 'crash-table': ('La pragul de 4σ, S&P 500 are 35 de zile sub −4σ, față de 0,29 așteptate în distribuția Normală. Butonul ▶ ridică pragul pînă la 6σ. Cîte zile ale S&P 500 sînt sub −6σ?',
                 '11 zile. Distribuția Normală așteaptă 9 240 × Φ(−6) = 9,1 × 10<sup>−6</sup> zile, adică o asemenea zi o dată la 4 milioane de ani. Cele 35 de zile sub −4σ nu se împart după forma distribuției Normale (35 × Φ(−6)/Φ(−4) = 0,001): o treime dintre ele sînt dincolo de 6σ. DAX are 5 zile sub −6σ, BET 13, Bitcoin 2.'),
 'hill-six': ('Pe slide, șase pierderi ilustrative dau α̂ = 2,12. Pentru cele mai mari șase pierderi reale ale S&P 500 (k = 5, pragul X<sub>(6)</sub> = 7,92%), cît este α̂?',
              '4,15: rapoartele logaritmice sînt 0,477, 0,232, 0,178, 0,166 și 0,152, cu suma 1,205, deci ξ̂ = 0,241 și α̂ = 1/0,241 = 4,15. Patru dintre cele cinci pierderi (9,99% … 9,22%) sînt aproape una de alta, iar cel mai mare raport vine doar din 16 martie 2020. SE = 4,15/√5 = 1,86: intervalul i.i.d. de 95%, [0,51; 7,78], conține și estimarea de la k = 2,5% din n (2,97). Cu cinci pierderi, α̂ nu este precis.'),
 'hill-table': ('Ultima coloană este indicele Hill al cîștigurilor, cu același k. Pentru S&P 500, pierderile au α̂ = 2,97 la k = 2,5% din n. Cît este α̂ al cîștigurilor?',
                '2,96: la S&P 500 cîștigurile au o coadă la fel de groasă ca pierderile. Cele mai mari cîștiguri sînt revenirile din aceleași crize: 10,96% (13 octombrie 2008), 10,25% (28 octombrie 2008), 9,09% (9 aprilie 2025), 8,97% (24 martie 2020). DAX (3,22), BET (2,82) și Bitcoin (3,37) au cîștiguri cu cozi mai subțiri decît pierderile. Tot la k = 2,5% din n, SE bloc a S&P 500 este 0,30, de 1,5 ori SE i.i.d. (0,20), pentru că pierderile mari vin grupate; pentru DAX cele două coincid (0,20).'),
 'bvb-table': ('La k = 2,5% din n, niciuna dintre cele patru acțiuni BVB nu are α̂ sub 2,4 (cea mai mică este TLV, 2,49). Butonul ▶ mută k/n de la 1% la 10%. Cîte dintre ele au α̂ sub 2,4 la k = 10% din n?',
               'Toate patru: SNP 2,27, BRD 2,35, TLV 2,08, TGN 1,87. Cu 10% din n intră în calcul corpul distribuției, care deplasează estimarea în jos, ca la indici (S&P 500 2,04). Intervalele se îngustează (SE i.i.d. între 0,10 și 0,12), dar sînt intervale în jurul unei mărimi deplasate. La 1% din n, BRD are 3,86, cu intervalul [2,62; 5,11]: puține pierderi, variabilitate mare.'),
 'me-table': ('La pragul de 90%, excesul mediu al S&P 500 este 0,92%, iar e(u)/u = 0,79. Butonul ▶ ridică pragul pînă la cuantila de 98% (u = 2,52%). Cît devine e(u)/u?',
              '0,48: e(u) crește, la 1,22%, dar mai încet decît pragul. Pentru o coadă Pareto, e(u)/u = 1/(α − 1): raportul 0,48 corespunde lui α = 1 + 1/0,48 = 3,1, aproape de estimarea Hill (2,97); raportul 0,79 de la 90% ar da α = 2,3, pentru că acolo contează încă pierderile din corpul distribuției. BET: de la 0,95 la 0,52.'),
 'gev-table': ('Cu maxime lunare, ξ̂ al BET este 0,36. Butonul ▶ trece prin maxime săptămînale, lunare, trimestriale și anuale. Cît este ξ̂ al BET cu maxime săptămînale?',
               '0,05, cu intervalul [0,03; 0,07]. Un bloc de 5 zile este departe de limita teoremei Fisher–Tippett–Gnedenko: 7% din maximele săptămînale ale BET sînt cîștiguri (nicio zi de pierdere în săptămînă), iar ajustarea urmează corpul distribuției. Cu blocuri trimestriale, ξ̂ = 0,42; cu blocuri anuale rămîn doar 28 de maxime, iar intervalul [−0,19; 0,65] conține și 0. Blocul trebuie să fie destul de lung pentru teoremă și destul de scurt pentru a lăsa multe maxime.'),
 'rl-example': ('Pentru S&P 500, nivelul de 10 ani (7,40%) a fost depășit în 4 luni din 440, față de 440/120 = 3,67 așteptate. Nivelul de 1 an este 3,66%. În cîte luni a fost depășit?',
                'În 31 de luni, față de 440/12 = 36,7 așteptate. „O dată pe an” înseamnă în medie o lună din 12, adică circa 37 de luni în 36,7 ani. La T = 1 an: −ln(1 − 1/12) = 0,0870, ridicat la −0,211 dă 1,675, iar z = 1,311 + 3,486 × 0,675 = 3,66%. Pentru DAX: 36 de luni peste 4,45%, cît se așteaptă.'),
 'rl-table': ('La T = 10 ani, intervalul bootstrap al BET este [10,3%; 17,5%]: limita de sus este de 1,70 ori cea de jos. Butonul ▶ mută T de la 1 la 100 de ani. Cît devine acest raport la T = 100 de ani?',
              '2,46: intervalul este [20,8%; 51,4%], în jurul unui nivel de 32,2%. Cu cît T este mai mare, cu atît nivelul depinde mai mult de ξ, iar incertitudinea lui ξ̂ (intervalul [0,26; 0,46]) se amplifică. Pentru Bitcoin, la 100 de ani: [31,1%; 94,4%], cu 12 ani de date. S&P 500 la 100 de ani: [9,9%; 18,3%].'),
 'gpd-table': ('La pragul de 90%, 1/ξ̂ al DAX este 8,5. Butonul ▶ ridică pragul de la cuantila de 80% la cea de 98%. Cît devine 1/ξ̂ al DAX la pragul de 98% (186 de excese)?',
               'Circa 19: ξ̂ scade la 0,053, cu SE 0,075, iar intervalul de 95% [−0,09; 0,20] conține 0. Inversul unei estimări apropiate de 0 este instabil: din același interval, α poate fi oriunde peste 5,0. La pragul de 80%, DAX are 1/ξ̂ = 15,6; S&P 500 coboară de la 6,3 (90%) la 3,9 (98%). Estimarea Hill a DAX (k = 2,5% din n) este 3,00.'),
 'crash-evt': ('Cu GPD peste cuantila de 90%, EVT dă pentru 19 octombrie 1987 o perioadă de așteptare de 1 602 ani. Butonul ▶ mută pragul de la cuantila de 80% la cea de 98%. Cît devine perioada cu pragul de 98%?',
               'Circa 306 ani: ξ̂ al S&P 500 crește de la 0,158 la 0,255, iar o coadă mai groasă face evenimentul de 22% mult mai puțin rar. La pragul de 80% perioada este 2 770 de ani. Cu cît pierderea este mai departe de date, cu atît perioada depinde mai mult de prag. DAX merge în sens invers: 85 de ani la 90%, 187 la 98%, pentru că ξ̂ scade la 0,053. Distribuția Normală rămîne la 10<sup>84</sup> ani pentru orice prag.'),
 'var-example': ('Pentru S&P 500, ES 2,5% / VaR 2,5% = 3,49/2,35 = 1,48. Butonul ▶ coboară nivelul ES de la 5% la 0,01%. Cît devine raportul ES/VaR la 0,01%?',
                 '1,25: ES 0,01% = 13,52%, VaR 0,01% = 10,79%. Departe în coadă, ES/VaR tinde la 1/(1 − ξ̂) = 1/(1 − 0,158) = 1,19: excesul peste VaR este tot GPD cu același ξ, iar media lui crește proporțional cu VaR. La 2,5%, raportul este mai mare pentru că termenul (β̂ − ξ̂u)/(1 − ξ̂) = 0,695 contează mult față de un VaR mic.'),
 'var-es': ('La VaR 1%, EVT dă pentru S&P 500 3,30%, iar distribuția Normală 2,60%. Butonul ▶ trece nivelul VaR prin 5%, 2,5%, 1%, 0,5%, 0,1%, 0,05% și 0,01%. La VaR 5%, care metodă dă cel mai mare VaR pentru S&P 500?',
            'Distribuția Normală: 1,83%, față de 1,73% EVT și 1,72% istoric; la fel pentru DAX, BET și Bitcoin. Pierderile cu cozi groase sînt și mai concentrate în jurul lui 0, așa că, aproape de corpul distribuției, cuantila empirică este sub cea Normală cu aceeași abatere standard. De la 2,5% (2,35% EVT, 2,19% Normal) ordinea se inversează, iar diferența crește spre coadă.'),
 'var-far': ('La VaR 0,1%, EVT dă pentru BET 9,56%, de 2,1 ori VaR-ul Normal (4,52%). Butonul ▶ trece nivelul de la 5% la 0,01%. De cîte ori este VaR 0,01% EVT al BET mai mare decît cel Normal?',
             'De 3,3 ori: 18,10% față de 5,45%. Cuantila Normală crește foarte încet (z trece de la 3,09 la 3,72), cea EVT ca o putere a lui 1/p. La 0,01%, eșantionul BET de 7 243 de zile conține în medie 0,7 zile dincolo de VaR, iar cea mai mare pierdere observată este 13,1%: valoarea EVT este o extrapolare. Hill dă 26,16%.'),
 'exceed': ('BET: VaR 1% EVT, estimat pînă în 2019, este depășit în 4 din cele 1 675 de zile de test, față de 16,8 așteptate. Ce spune testul Kupiec la 5%?',
            'Respinge acoperirea corectă: LR = 14,14, valoarea p 0,0002. Testul este bilateral: prea puține depășiri înseamnă un VaR prea mare pentru perioada testată, care a fost mai liniștită pentru BET decît anii 1997–2019. VaR Normal al BET, cu 13 depășiri, nu este respins (valoarea p 0,34), deși la 0,1% are 4 depășiri față de 1,7. La S&P 500, EVT (25 de depășiri) nu este respins (valoarea p 0,063), simularea istorică (26) este respinsă (0,038), iar distribuția Normală (44) are valoarea p 3 × 10<sup>−8</sup>.'),
})
QA['en'].update({
 'crash-table': ('At the 4σ threshold, the S&P 500 has 35 days below −4σ, against 0.29 expected under the Normal distribution. The ▶ button raises the threshold up to 6σ. How many S&P 500 days are below −6σ?',
                 '11 days. The Normal distribution expects 9,240 × Φ(−6) = 9.1 × 10<sup>−6</sup> days, i.e. one such day in 4 million years. The 35 days below −4σ do not thin out as the Normal shape would have them (35 × Φ(−6)/Φ(−4) = 0.001): a third of them lie beyond 6σ. DAX has 5 days below −6σ, BET 13, Bitcoin 2.'),
 'hill-six': ('On the slide, six illustrative losses give α̂ = 2.12. For the six largest real losses of the S&P 500 (k = 5, threshold X<sub>(6)</sub> = 7.92%), what is α̂?',
              '4.15: the log ratios are 0.477, 0.232, 0.178, 0.166 and 0.152, with sum 1.205, so ξ̂ = 0.241 and α̂ = 1/0.241 = 4.15. Four of the five losses (9.99% … 9.22%) are close to each other, and the largest ratio comes from 16 March 2020 alone. SE = 4.15/√5 = 1.86: the i.i.d. 95% interval, [0.51, 7.78], also contains the estimate at k = 2.5% of n (2.97). With five losses, α̂ is not precise.'),
 'hill-table': ('The last column is the Hill index of the gains, with the same k. For the S&P 500, the losses have α̂ = 2.97 at k = 2.5% of n. What is α̂ of the gains?',
                '2.96: for the S&P 500 the gains have a tail as heavy as the losses. The largest gains are the rebounds of the same crises: 10.96% (13 October 2008), 10.25% (28 October 2008), 9.09% (9 April 2025), 8.97% (24 March 2020). DAX (3.22), BET (2.82) and Bitcoin (3.37) have gains with lighter tails than their losses. Also at k = 2.5% of n, the block SE of the S&P 500 is 0.30, 1.5 times the i.i.d. SE (0.20), because large losses come in clusters; for the DAX the two coincide (0.20).'),
 'bvb-table': ('At k = 2.5% of n, none of the four BVB stocks has α̂ below 2.4 (the smallest is TLV, 2.49). The ▶ button moves k/n from 1% to 10%. How many of them have α̂ below 2.4 at k = 10% of n?',
               'All four: SNP 2.27, BRD 2.35, TLV 2.08, TGN 1.87. With 10% of n the body of the distribution enters the estimate and biases it downwards, as for the indices (S&P 500 2.04). The intervals narrow (i.i.d. SE between 0.10 and 0.12), but they are intervals around a biased quantity. At 1% of n, BRD has 3.86, with the interval [2.62, 5.11]: few losses, high variability.'),
 'me-table': ('At the 90% threshold, the mean excess of the S&P 500 is 0.92%, and e(u)/u = 0.79. The ▶ button raises the threshold up to the 98% quantile (u = 2.52%). What does e(u)/u become?',
              '0.48: e(u) grows, to 1.22%, but more slowly than the threshold. For a Pareto tail, e(u)/u = 1/(α − 1): the ratio 0.48 corresponds to α = 1 + 1/0.48 = 3.1, close to the Hill estimate (2.97); the ratio 0.79 at 90% would give α = 2.3, because there the losses of the body still matter. BET: from 0.95 to 0.52.'),
 'gev-table': ('With monthly maxima, ξ̂ of BET is 0.36. The ▶ button goes through weekly, monthly, quarterly and yearly maxima. What is ξ̂ of BET with weekly maxima?',
               '0.05, with the interval [0.03, 0.07]. A block of 5 days is far from the limit of the Fisher–Tippett–Gnedenko theorem: 7% of the weekly maxima of BET are gains (no loss day in the week), and the fit follows the body of the distribution. With quarterly blocks, ξ̂ = 0.42; with yearly blocks only 28 maxima remain, and the interval [−0.19, 0.65] also contains 0. The block must be long enough for the theorem and short enough to leave many maxima.'),
 'rl-example': ('For the S&P 500, the 10-year level (7.40%) was exceeded in 4 months of 440, against 440/120 = 3.67 expected. The 1-year level is 3.66%. In how many months was it exceeded?',
                'In 31 months, against 440/12 = 36.7 expected. "Once a year" means on average one month in 12, i.e. about 37 months in 36.7 years. At T = 1 year: −ln(1 − 1/12) = 0.0870, raised to −0.211 gives 1.675, and z = 1.311 + 3.486 × 0.675 = 3.66%. For the DAX: 36 months above 4.45%, as expected.'),
 'rl-table': ('At T = 10 years, the bootstrap interval of BET is [10.3%, 17.5%]: the upper bound is 1.70 times the lower one. The ▶ button moves T from 1 to 100 years. What does this ratio become at T = 100 years?',
              '2.46: the interval is [20.8%, 51.4%], around a level of 32.2%. The larger T, the more the level depends on ξ, and the uncertainty of ξ̂ (interval [0.26, 0.46]) is amplified. For Bitcoin, at 100 years: [31.1%, 94.4%], with 12 years of data. S&P 500 at 100 years: [9.9%, 18.3%].'),
 'gpd-table': ('At the 90% threshold, 1/ξ̂ of the DAX is 8.5. The ▶ button raises the threshold from the 80% to the 98% quantile. What does 1/ξ̂ of the DAX become at the 98% threshold (186 excesses)?',
               'About 19: ξ̂ falls to 0.053, with SE 0.075, and the 95% interval [−0.09, 0.20] contains 0. The inverse of an estimate close to 0 is unstable: from the same interval, α can be anywhere above 5.0. At the 80% threshold, DAX has 1/ξ̂ = 15.6; the S&P 500 falls from 6.3 (90%) to 3.9 (98%). The Hill estimate of the DAX (k = 2.5% of n) is 3.00.'),
 'crash-evt': ('With the GPD above the 90% quantile, EVT gives 19 October 1987 a waiting time of 1,602 years. The ▶ button moves the threshold from the 80% to the 98% quantile. What does the waiting time become with the 98% threshold?',
               'About 306 years: ξ̂ of the S&P 500 rises from 0.158 to 0.255, and a heavier tail makes the 22% event much less rare. At the 80% threshold the waiting time is 2,770 years. The farther the loss lies from the data, the more its waiting time depends on the threshold. The DAX moves the other way: 85 years at 90%, 187 at 98%, because its ξ̂ falls to 0.053. The Normal distribution stays at 10<sup>84</sup> years for any threshold.'),
 'var-example': ('For the S&P 500, ES 2.5% / VaR 2.5% = 3.49/2.35 = 1.48. The ▶ button lowers the ES level from 5% to 0.01%. What does the ratio ES/VaR become at 0.01%?',
                 '1.25: ES 0.01% = 13.52%, VaR 0.01% = 10.79%. Far in the tail, ES/VaR tends to 1/(1 − ξ̂) = 1/(1 − 0.158) = 1.19: the excess over VaR is again GPD with the same ξ, and its mean grows in proportion to VaR. At 2.5% the ratio is larger because the term (β̂ − ξ̂u)/(1 − ξ̂) = 0.695 weighs a lot against a small VaR.'),
 'var-es': ('At VaR 1%, EVT gives 3.30% for the S&P 500 and the Normal distribution 2.60%. The ▶ button moves the VaR level through 5%, 2.5%, 1%, 0.5%, 0.1%, 0.05% and 0.01%. At VaR 5%, which method gives the largest VaR for the S&P 500?',
            'The Normal distribution: 1.83%, against 1.73% EVT and 1.72% historical; the same holds for DAX, BET and Bitcoin. Heavy-tailed losses are also more concentrated around 0, so near the body of the distribution the empirical quantile lies below the Normal one with the same standard deviation. From 2.5% on (2.35% EVT, 2.19% Normal) the order reverses, and the gap widens towards the tail.'),
 'var-far': ('At VaR 0.1%, EVT gives 9.56% for BET, 2.1 times the Normal VaR (4.52%). The ▶ button moves the level from 5% to 0.01%. How many times larger than the Normal one is the EVT VaR 0.01% of BET?',
             '3.3 times: 18.10% against 5.45%. The Normal quantile grows very slowly (z moves from 3.09 to 3.72), the EVT one as a power of 1/p. At 0.01%, the BET sample of 7,243 days holds on average 0.7 days beyond VaR, and the largest observed loss is 13.1%: the EVT value is an extrapolation. Hill gives 26.16%.'),
 'exceed': ('BET: the EVT VaR 1%, estimated until 2019, is exceeded on 4 of the 1,675 test days, against 16.8 expected. What does the Kupiec test say at 5%?',
            'It rejects correct coverage: LR = 14.14, p-value 0.0002. The test is two-sided: too few exceedances mean a VaR too high for the test period, which was calmer for BET than the years 1997–2019. The Normal VaR of BET, with 13 exceedances, is not rejected (p-value 0.34), although at 0.1% it has 4 exceedances against 1.7. For the S&P 500, EVT (25 exceedances) is not rejected (p-value 0.063), historical simulation (26) is rejected (0.038), and the Normal distribution (44) has a p-value of 3 × 10<sup>−8</sup>.'),
})

PREDICT.update({
 'crash-table': (0, [('11 zile: o treime dintre cele 35 sînt dincolo de 6σ.', '11 days: a third of the 35 lie beyond 6σ.'),
                     ('Niciuna: o zi de 6σ apare o dată la milioane de ani.', 'None: a 6σ day occurs once in millions of years.'),
                     ('Circa 0,001: 35 înmulțit cu Φ(−6)/Φ(−4).', 'About 0.001: 35 times Φ(−6)/Φ(−4).')]),
 'hill-six': (1, [('Circa 2,97, ca la k = 231: α̂ nu depinde de k.', 'About 2.97, as at k = 231: α̂ does not depend on k.'),
                  ('Circa 4,1: cinci pierderi apropiate de prag dau rapoarte mici.', 'About 4.1: five losses close to the threshold give small ratios.'),
                  ('Circa 1,6: căderea de 12,77% face coada să pară foarte groasă.', 'About 1.6: the 12.77% fall makes the tail look very heavy.')]),
 'hill-table': (0, [('Circa 2,96: o coadă la fel de groasă ca a pierderilor.', 'About 2.96: a tail as heavy as that of the losses.'),
                    ('Circa 3,5: cîștigurile au întotdeauna o coadă mai subțire.', 'About 3.5: gains always have a lighter tail.'),
                    ('Circa 2,0: revenirile de după crize sînt mai mari decît căderile.', 'About 2.0: the rebounds after crises are larger than the falls.')]),
 'bvb-table': (2, [('Niciuna: α̂ este o proprietate a acțiunii, nu a lui k.', 'None: α̂ is a property of the stock, not of k.'),
                   ('Doar TLV, cea mai apropiată de 2,4.', 'Only TLV, the closest to 2.4.'),
                   ('Toate patru: corpul distribuției deplasează estimarea în jos.', 'All four: the body of the distribution biases the estimate downwards.')]),
 'me-table': (0, [('Circa 0,48: e(u) crește mai încet decît u.', 'About 0.48: e(u) grows more slowly than u.'),
                  ('Tot 0,79: e(u) este proporțional cu u.', 'Still 0.79: e(u) is proportional to u.'),
                  ('Peste 1: excesul mediu crește mai repede decît pragul.', 'Above 1: the mean excess grows faster than the threshold.')]),
 'gev-table': (1, [('Tot circa 0,36: ξ este o proprietate a cozii, nu a blocului.', 'Still about 0.36: ξ is a property of the tail, not of the block.'),
                   ('Circa 0,05: blocurile de 5 zile sînt prea scurte pentru teoremă.', 'About 0.05: blocks of 5 days are too short for the theorem.'),
                   ('Peste 0,5: mai multe maxime fac coada mai groasă.', 'Above 0.5: more maxima make the tail heavier.')]),
 'rl-example': (2, [('4, ca pentru nivelul de 10 ani.', '4, as for the 10-year level.'),
                    ('12: o dată pe an înseamnă 12 luni.', '12: once a year means 12 months.'),
                    ('31, aproape de 440/12 = 36,7.', '31, close to 440/12 = 36.7.')]),
 'rl-table': (1, [('Tot circa 1,7: intervalul crește proporțional cu nivelul.', 'Still about 1.7: the interval grows in proportion to the level.'),
                  ('Circa 2,5: incertitudinea lui ξ̂ se amplifică cu T.', 'About 2.5: the uncertainty of ξ̂ is amplified with T.'),
                  ('Sub 1,5: la T mare nivelul se stabilizează.', 'Below 1.5: at a large T the level settles.')]),
 'gpd-table': (1, [('Circa 3: un prag mai înalt apropie GPD de Hill.', 'About 3: a higher threshold brings the GPD close to Hill.'),
                   ('Circa 19: ξ̂ scade la 0,05, iar intervalul lui conține 0.', 'About 19: ξ̂ falls to 0.05, and its interval contains 0.'),
                   ('Tot 8,5: ξ nu depinde de prag.', 'Still 8.5: ξ does not depend on the threshold.')]),
 'crash-evt': (1, [('Tot circa 1 602: pragul doar alege datele.', 'Still about 1,602: the threshold only selects the data.'),
                   ('Circa 300: ξ̂ crește la 0,255, coada devine mai groasă.', 'About 300: ξ̂ rises to 0.255, the tail gets heavier.'),
                   ('Peste 10 000: mai puține excese, eveniment mai rar.', 'Above 10,000: fewer excesses, a rarer event.')]),
 'var-example': (0, [('Circa 1,25, spre limita 1/(1 − ξ̂) = 1,19.', 'About 1.25, towards the limit 1/(1 − ξ̂) = 1.19.'),
                     ('Tot 1,48: ES este un multiplu fix al VaR la orice nivel.', 'Still 1.48: ES is a fixed multiple of VaR at any level.'),
                     ('Peste 2: departe în coadă ES crește mai repede decît VaR.', 'Above 2: far in the tail ES grows faster than VaR.')]),
 'var-es': (1, [('EVT, ca la 1%: coada groasă ridică VaR la orice nivel.', 'EVT, as at 1%: the heavy tail raises VaR at any level.'),
                ('Distribuția Normală: aproape de corp, pierderile sînt concentrate în jurul lui 0.', 'The Normal distribution: near the body, losses are concentrated around 0.'),
                ('Simularea istorică: la 5% datele sînt cele mai multe.', 'Historical simulation: at 5% the data are the most numerous.')]),
 'var-far': (1, [('Tot circa 2,1: raportul depinde doar de serie.', 'Still about 2.1: the ratio depends only on the series.'),
                 ('Circa 3,3: cuantila Normală crește foarte încet.', 'About 3.3: the Normal quantile grows very slowly.'),
                 ('Circa 1: departe în coadă modelele converg.', 'About 1: far in the tail the models converge.')]),
 'exceed': (0, [('Respinge: prea puține depășiri înseamnă tot o acoperire greșită.', 'Rejected: too few exceedances also mean a wrong coverage.'),
                ('Nu respinge: mai puține depășiri înseamnă un VaR prudent.', 'Not rejected: fewer exceedances mean a prudent VaR.'),
                ('Respinge: prea multe depășiri.', 'Rejected: too many exceedances.')]),
})

VEIL = {'hill-six': {'out': True, 'charts': ['hs-tab', 'hs-we-box']}, 'hill-table': {'out': True, 'charts': ['ht-tab']},
        'gev-table': {'out': True, 'charts': ['gx-chart']}, 'rl-table': {'out': True, 'charts': ['rt-chart']},
        'exceed': {'out': True, 'charts': ['ex-tab', 'ex-chart', 'ex-we-box']}}


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch5-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH5_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH5_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH5_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


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
        assert (kro if lang == 'ro' else ken) in title, (lang, a, title)
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
    """Chart blocks for build_web_lecture.py: each carries the web version of its table blocks (moved to their frames by ch5.js)."""
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
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/ch5_{a}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
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
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', f'ch5_{a}_{lang}.jpg'), jpg_quality=82)
        print('wrote', L[lang]['file'])


if __name__ == '__main__':
    build()
