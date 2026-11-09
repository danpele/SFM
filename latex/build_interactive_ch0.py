"""Build the interactive pages of Chapter 0: interactive/ch0.html (EN) and interactive/ch0_ro.html (RO).

One block for each of six chart slides of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch0.js, inputs in interactive/ch0_data.js
(Quantlets/Ch_00/export_interactive.py), slide images in interactive/slides/.
Charts of the lecture without a block: sfm_ch0_bet_history (the BET since 1997; its drawdown is the BET line of the
drawdown block) and sfm_ch0_btc_vol (in the section on AI for scientific discovery).
Run:  cd latex && python3 build_interactive_ch0.py
"""
import json
import re
import sys

from build_web_lecture import site_decks
from interactive_common import slider, select, play, charts, build

N = 0
TEX = site_decks()[N]
QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_00'

# (anchor, chart file, section key); order of the slides
BLOCKS = [('markets', 'sfm_ch0_markets', 'mk'), ('risk-return', 'sfm_ch0_risk_return', 'mk'), ('drawdowns', 'sfm_ch0_drawdowns', 'mk'),
          ('vix', 'sfm_ch0_vix', 'mk'), ('returns', 'sfm_ch0_returns', 'ret'), ('histogram', 'sfm_ch0_histogram', 'ret')]

PFX = {'markets': 'mk', 'risk-return': 'rr', 'drawdowns': 'dd', 'vix': 'vx', 'returns': 'rt', 'histogram': 'hs'}

SECTIONS = {
 'ro': dict(mk='Piețele financiare în 2026: statistici descriptive', ret='De la prețuri la randamente: primii pași'),
 'en': dict(mk='Financial Markets in 2026: Descriptive Statistics', ret='From Prices to Returns: a First Look'),
}

L = {
 'ro': dict(file='ch0_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Statistica piețelor financiare',
   h1='Capitolul 0 · Introducere', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas. '
        'Abrevieri: DAX (Deutscher Aktienindex, indicele bursei din Frankfurt), BET (Bucharest Exchange Trading, indicele de referință al Bursei de Valori București), '
        'BET-TR (BET total return, cu dividendele reinvestite), WIG20 (indicele celor mai mari 20 de firme de la bursa din Varșovia), TLT (un ETF de titluri de stat '
        'americane cu scadența peste 20 de ani), ETF (exchange-traded fund, fond tranzacționat la bursă), VIX (Cboe Volatility Index, volatilitatea S&P 500 așteptată '
        'de piața de opțiuni), SE (eroarea standard), GARCH (generalised autoregressive conditional heteroskedasticity).',
   src='Datele cursului din data/market (sursa EODHD), închideri zilnice pînă la 18 septembrie 2026, fiecare serie pe propriul calendar: cele cinci piețe și cele '
       'unsprezece piețe ale graficului risc–randament din ianuarie 2015, S&P 500, BET și VIX din ianuarie 2000. Browserul recalculează creșterea a 100 de unități, '
       'randamentele și volatilitățile anualizate, drawdown-urile, statisticile VIX, volatilitatea mobilă, histograma și numărul de zile din cozi. Alte date de '
       'început ale ferestrei, alte praguri și alte ferestre ale volatilității mobile sînt rulări suplimentare ale aceluiași calcul.'),
 'en': dict(file='ch0.html', html_lang='en', site='../index.html#chapters', course='Statistics of Financial Markets',
   h1='Chapter 0 · Introduction', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step. '
        'Abbreviations: DAX (Deutscher Aktienindex, the index of the Frankfurt Stock Exchange), BET (Bucharest Exchange Trading, the benchmark index of the Bucharest '
        'Stock Exchange), BET-TR (BET total return, with dividends reinvested), WIG20 (the index of the 20 largest companies on the Warsaw Stock Exchange), TLT (an ETF '
        'of US Treasury bonds with maturities over 20 years), ETF (exchange-traded fund), VIX (Cboe Volatility Index, the S&P 500 volatility expected by the options '
        'market), SE (standard error), GARCH (generalised autoregressive conditional heteroskedasticity).',
   src='Course data in data/market (source: EODHD), daily closes up to 18 September 2026, each series on its own calendar: the five markets and the eleven markets '
       'of the risk–return chart from January 2015, the S&P 500, the BET and the VIX from January 2000. The browser recomputes the growth of 100, the annualised '
       'returns and volatilities, the drawdowns, the VIX statistics, the rolling volatility, the histogram and the number of days in the tails. Other start dates of '
       'the window, other thresholds and other rolling windows are extra runs of the same computation.'),
}


def widgets(lang):
    ro = lang == 'ro'
    T = lambda r, e: r if ro else e
    W = [(str(w), f'{w} {T("de zile", "days")} ({m})') for w, m in ((21, T('o lună', 'one month')), (63, T('un trimestru', 'one quarter')),
                                                                 (126, T('jumătate de an', 'half a year')), (252, T('un an', 'one year')))]
    return {
     'markets': (slider('mk-y', T('Fereastra începe în ianuarie (slide: 2015)', 'The window starts in January (slide: 2015)'), 2015, 2025, 1, 2015) + play('mk'),
                 charts('mk-chart')),
     'risk-return': (slider('rr-y', T('Fereastra începe în ianuarie (slide: 2015)', 'The window starts in January (slide: 2015)'), 2015, 2024, 1, 2015) + play('rr'),
                     charts('rr-chart')),
     'drawdowns': (slider('dd-y', T('Datele se opresc la sfîrșitul anului (slide: 18.09.2026)', 'The data stop at the end of year (slide: 18.09.2026)'), 2001, 2026, 1, 2026) + play('dd'),
                   charts('dd-chart')),
     'vix': (slider('vx-t', T('Prag VIX', 'VIX threshold'), 20, 80, 5, 30) + play('vx'), charts('vx-chart')),
     'returns': (select('rt-w', T('Fereastra volatilității mobile', 'Window of the rolling volatility'), W, '21') + play('rt'), charts('rt-chart')),
     'histogram': (slider('hs-k', T('Prag: abateri standard de la medie (slide: 4)', 'Threshold: standard deviations from the mean (slide: 4)'), 2, 6, 0.5, 4) + play('hs'),
                   charts('hs-chart')),
    }


# readouts: JavaScript, one object per language
TEXT_JS = {
 'ro': r"""(function () {
const X = ' (rulare suplimentară)';
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const NAME = { sp500: 'S&P 500', dax: 'DAX', bettr: 'BET-TR', btc: 'Bitcoin', gold: 'aur', bet: 'BET', ndx: 'Nasdaq 100', stoxx50: 'Euro Stoxx 50', nikkei: 'Nikkei 225',
               wig20: 'WIG20', tlt: 'TLT', eth: 'Ethereum' };
const num = s => +String(s).replace(/[\s  ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
window.CH0_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 name: k => NAME[k],
 endLabel: '18.09.2026',
 mk: (y, rows, x) => `100 de unități investite în ianuarie ${y}${x ? X : ''}: ` + rows.map(r => `${r[0]} ×${r[1]} (randament logaritmic mediu ${r[2]}% pe an, volatilitate ${r[3]}%)`).join('; ') + '.',
 mk2: (best, worst, dr, dv, ratio) => `Cel mai mare randament anual: ${best}; cel mai mic: ${worst}. BET-TR față de S&P 500: ${dr} puncte procentuale de randament, ${dv} de volatilitate. Volatilitatea Bitcoin este de ${ratio} ori mai mare decît a S&P 500.`,
 rr: (y, vlo, vhi, eLo, rLo, eHi, rHi, vb, ve, rt, x) => `Din ianuarie ${y}${x ? X : ''}: cei șapte indici bursieri au volatilități între ${vlo}% și ${vhi}% pe an și randamente între ${rLo}% (${eLo}) și ${rHi}% (${eHi}); volatilitatea Bitcoin este ${vb}%, a Ethereum ${ve}%; TLT: ${rt}% pe an.`,
 rr2: (Y, se, half, seb) => `Eroarea standard (SE) a unui randament mediu anual estimat pe Y = ${Y} ani este σ/√Y: ${se} puncte procentuale pentru S&P 500 (interval de 95%: ±${half} puncte), ${seb} puncte pentru Bitcoin.`,
 dd: (nm, mdd, tr, pk, rec, g, last, ld, lg) => `${nm}: drawdown maxim ${mdd}% pe ${tr}, de la vîrful din ${pk}; ` + (rec ? `revine la vîrf pe ${rec}` : `nu a revenit la vîrf pînă pe ${ld}`) + `; de la minim, revenirea cere un cîștig de ${g}%. Pe ${ld}: ` + (num(last) === 0 ? 'indicele este la un nou maxim.' : `drawdown ${last}%, iar revenirea la vîrf cere +${lg}%.`),
 vx: (m, med, d1, v1, d2, v2, last) => `Din 2000: media ${m}, mediana ${med}; cele mai mari închideri: ${v1} pe ${d1} și ${v2} pe ${d2}; pe 18 septembrie 2026, ${last}.`,
 vx2: (t, k, N, sh, runs, lg, d0, stay) => `Închideri peste ${t}: ${k} ${de(k)}zile din ${N} (${sh}%), în ${runs} ${de(runs)}episoade; cel mai lung durează ${lg} ${num(lg) === 1 ? 'zi' : de(lg) + 'zile'} de tranzacționare, începînd cu ${d0}. După o zi peste ${t}, și ziua următoare este peste ${t} în ${stay}% din cazuri, față de ${sh}% din toate zilele.`,
 rt: (N, d0, m, s, va, ratio, t) => `${N} ${de(N)}randamente logaritmice zilnice de la ${d0}: media ${m}% pe zi, abaterea standard ${s}% (${va}% pe an); abaterea standard este de ${ratio} de ori mai mare decît media. Statistica t a mediei, media/(s/√n) = ${t}` + (num(t) < 1.96 ? ', sub 1,96: pe 26 de ani, testul bilateral de 5% nu separă media de zero.' : '.'),
 rt2: (w, lo, dlo, hi, dhi, o, c, sh, nsh) => `Abaterea standard mobilă pe ${w} de zile anterioare, anualizată: de la ${lo}% (${dlo}) la ${hi}% (${dhi}). Randamente în afara benzii ±2 abateri standard mobile: ${o} din ${c} ${de(c)}zile (${sh}%; ${nsh}% la distribuția Normală).`,
 hs: (k, o, lo, hi, e, ratio, N) => `Zile la mai mult de ${k} abateri standard de medie: ${o} din ${N} (${lo} ${de(lo)}pierderi, ${hi} ${de(hi)}cîștiguri); număr așteptat la distribuția Normală cu aceeași medie și varianță: ${e}. Datele au de ${ratio} ori mai multe astfel de zile.`,
 hs2: (K, z, d) => `Coeficientul de boltire K = ${K} (3 la distribuția Normală); cea mai slabă zi, ${d}, se află la ${z} abateri standard de medie.`,
};
})();
""",
 'en': r"""(function () {
const X = ' (extra run)';
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const NAME = { sp500: 'the S&P 500', dax: 'the DAX', bettr: 'the BET-TR', btc: 'Bitcoin', gold: 'gold', bet: 'the BET', ndx: 'the Nasdaq 100', stoxx50: 'the Euro Stoxx 50',
               nikkei: 'the Nikkei 225', wig20: 'the WIG20', tlt: 'TLT', eth: 'Ethereum' };
const num = s => +String(s).replace(/[\s,%]/g, '').replace('−', '-');
const cap = s => s[0].toUpperCase() + s.slice(1);
window.CH0_TEXT = {
 date: s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`,
 name: k => NAME[k],
 endLabel: '18.09.2026',
 mk: (y, rows, x) => `100 invested in January ${y}${x ? X : ''}: ` + rows.map(r => `${r[0].replace(/^the /, '')} ×${r[1]} (mean log return ${r[2]}% a year, volatility ${r[3]}%)`).join('; ') + '.',
 mk2: (best, worst, dr, dv, ratio) => `Highest annual return: ${best.replace(/^the /, '')}; lowest: ${worst.replace(/^the /, '')}. BET-TR against the S&P 500: ${dr} percentage points of return, ${dv} of volatility. Bitcoin’s volatility is ${ratio} times that of the S&P 500.`,
 rr: (y, vlo, vhi, eLo, rLo, eHi, rHi, vb, ve, rt, x) => `From January ${y}${x ? X : ''}: the seven equity indices have volatilities between ${vlo}% and ${vhi}% a year and returns between ${rLo}% (${eLo.replace(/^the /, '')}) and ${rHi}% (${eHi.replace(/^the /, '')}); the volatility of Bitcoin is ${vb}%, that of Ethereum ${ve}%; TLT: ${rt}% a year.`,
 rr2: (Y, se, half, seb) => `The standard error (SE) of an annual mean return estimated over Y = ${Y} years is σ/√Y: ${se} percentage points for the S&P 500 (95% interval: ±${half} points), ${seb} points for Bitcoin.`,
 dd: (nm, mdd, tr, pk, rec, g, last, ld, lg) => `${cap(nm)}: maximum drawdown ${mdd}% on ${tr}, from the peak of ${pk}; ` + (rec ? `back at the peak on ${rec}` : `not back at the peak by ${ld}`) + `; from the trough, recovery needs a gain of ${g}%. On ${ld}: ` + (num(last) === 0 ? 'the index is at a new high.' : `drawdown ${last}%, and getting back to the peak needs +${lg}%.`),
 vx: (m, med, d1, v1, d2, v2, last) => `Since 2000: mean ${m}, median ${med}; highest closes: ${v1} on ${d1} and ${v2} on ${d2}; ${last} on 18 September 2026.`,
 vx2: (t, k, N, sh, runs, lg, d0, stay) => `Closes above ${t}: ${k} of ${N} days (${sh}%), in ${runs} episodes; the longest lasts ${lg} trading day${num(lg) === 1 ? '' : 's'}, from ${d0}. After a day above ${t}, the next day is also above ${t} in ${stay}% of cases, against ${sh}% of all days.`,
 rt: (N, d0, m, s, va, ratio, t) => `${N} daily log returns from ${d0}: mean ${m}% a day, standard deviation ${s}% (${va}% a year); the standard deviation is ${ratio} times the mean. The t statistic of the mean, mean/(s/√n) = ${t}` + (num(t) < 1.96 ? ', below 1.96: over 26 years the two-sided 5% test does not separate the mean from zero.' : '.'),
 rt2: (w, lo, dlo, hi, dhi, o, c, sh, nsh) => `Rolling standard deviation over the ${w} previous days, annualised: from ${lo}% (${dlo}) to ${hi}% (${dhi}). Returns outside the band ±2 rolling standard deviations: ${o} of ${c} days (${sh}%; ${nsh}% for the Normal distribution).`,
 hs: (k, o, lo, hi, e, ratio, N) => `Days more than ${k} standard deviations from the mean: ${o} of ${N} (${lo} losses, ${hi} gains); expected number under the Normal distribution with the same mean and variance: ${e}. The data have ${ratio} times as many such days.`,
 hs2: (K, z, d) => `Kurtosis K = ${K} (3 for the Normal distribution); the worst day, ${d}, lies ${z} standard deviations from the mean.`,
};
})();
""",
}

QA = {
 'ro': {
  'markets': ('▶ mută începutul ferestrei an cu an, pînă în ianuarie 2025 (rulări suplimentare). Care dintre cele cinci piețe are cel mai mic randament anual în fereastra din 2025?',
              'Bitcoin: −9,0% pe an (100 de unități devin 86), deși în fereastra din 2015 avea cel mai mare randament, 47,4%; în fereastra din 2025 conduce BET-TR, cu 44,0%. '
              'Pe ferestre scurte, clasamentul depinde aproape în întregime de data de început: un randament mediu estimat pe mai puțin de doi ani este foarte imprecis.'),
  'risk-return': ('▶ mută începutul ferestrei pînă în 2024 (rulări suplimentare). Unde ajunge Ethereum, activul cu cea mai mare volatilitate, în fereastra din ianuarie 2022?',
                  'Sub zero: −7,8% pe an, cu o volatilitate de 68,9%, sub S&P 500 (9,9%) și BET-TR (26,2%). Graficul nu demonstrează că riscul mai mare aduce un randament '
                  'mai mare: punctele sînt medii ale unui singur eșantion. Eroarea standard a randamentului mediu anual pe Y ani este σ/√Y; pentru S&P 500 din 2015, '
                  '17,7/√11,7 = 5,2 puncte procentuale, pentru Bitcoin 19,5 puncte.'),
  'drawdowns': ('▶ mută sfîrșitul datelor din 2001 pînă în 2026. În ce an revine BET la vîrful din 24 iulie 2007?',
                'În 2021, pe 16 martie, după mai mult de 13 ani; S&P 500 revenise la vîrful din 9 octombrie 2007 pe 28 martie 2013. De la minimul din 25 februarie 2009 '
                '(−82,5%), BET avea nevoie de un cîștig de 473%, adică de o creștere de 5,7 ori; S&P 500, după −56,8%, de un cîștig de 131%. După o pierdere x, revenirea '
                'cere un cîștig x/(1 − x).'),
  'vix': ('Pragul este 30. Într-o zi în care VIX închide peste 30, cît de probabil este ca și ziua următoare să închidă peste 30?',
          'Probabilitatea empirică este 88,2%, deși doar 9,3% dintre zilele din 2000 au închideri peste 30. Zilele cu volatilitate mare vin în episoade (74 de episoade '
          'peste 30, cel mai lung de 170 de zile de tranzacționare): este fenomenul de volatility clustering, modelat cu GARCH în Capitolul 9.'),
  'returns': ('▶ mărește fereastra volatilității mobile de la 21 la 252 de zile. Ce se întîmplă cu cea mai mare volatilitate mobilă anualizată?',
              'Scade la mai puțin de jumătate: de la 97,4% (7 aprilie 2020) la 45,6% (16 iulie 2009). O fereastră lungă amestecă zilele de criză cu zile calme și '
              'netezește vîrfurile. Pentru orice fereastră, între 6,0% și 6,7% dintre zile ies din banda ±2 abateri standard mobile, față de 4,6% la distribuția Normală.'),
  'histogram': ('▶ ridică pragul de la 4 la 6 abateri standard. Cum se schimbă raportul dintre numărul de zile observate dincolo de prag și numărul așteptat la distribuția Normală?',
                'Crește foarte mult: la 4 abateri standard, 46 de zile față de 0,43 așteptate (de 108 ori mai multe); la 6 abateri standard, 12 zile față de 0,000013 '
                '(de circa 906 000 de ori mai multe). Coada distribuției Normale scade mult mai repede decît coada randamentelor, de aceea riscul calculat cu distribuția '
                'Normală este subestimat în cozi (Capitolele 2, 5 și 10).'),
 },
 'en': {
  'markets': ('▶ moves the start of the window year by year, up to January 2025 (extra runs). Which of the five markets has the lowest annual return in the window from 2025?',
              'Bitcoin: −9.0% a year (100 becomes 86), although in the window from 2015 it had the highest return, 47.4%; in the window from 2025 the BET-TR leads with '
              '44.0%. Over short windows the ranking depends almost entirely on the start date: a mean return estimated over less than two years is very imprecise.'),
  'risk-return': ('▶ moves the start of the window up to 2024 (extra runs). Where does Ethereum, the most volatile asset, land in the window from January 2022?',
                  'Below zero: −7.8% a year with a volatility of 68.9%, below the S&P 500 (9.9%) and the BET-TR (26.2%). The chart does not prove that higher risk brings '
                  'higher return: the points are averages of one sample. The standard error of an annual mean return over Y years is σ/√Y; for the S&P 500 since 2015, '
                  '17.7/√11.7 = 5.2 percentage points, for Bitcoin 19.5 points.'),
  'drawdowns': ('▶ moves the end of the data from 2001 to 2026. In which year does the BET regain its peak of 24 July 2007?',
                'In 2021, on 16 March, more than 13 years later; the S&P 500 had regained its peak of 9 October 2007 on 28 March 2013. From the trough of 25 February 2009 '
                '(−82.5%) the BET needed a gain of 473%, a 5.7-fold rise; the S&P 500, after −56.8%, a gain of 131%. After a loss x, recovery needs a gain of x/(1 − x).'),
  'vix': ('The threshold is 30. On a day when the VIX closes above 30, how likely is it that the next day also closes above 30?',
          'The empirical probability is 88.2%, although only 9.3% of the days since 2000 close above 30. High-volatility days come in episodes (74 episodes above 30, '
          'the longest 170 trading days long): this is volatility clustering, modelled with GARCH in Chapter 9.'),
  'returns': ('▶ lengthens the window of the rolling volatility from 21 to 252 days. What happens to the highest annualised rolling volatility?',
              'It falls to less than half: from 97.4% (7 April 2020) to 45.6% (16 July 2009). A long window mixes crisis days with calm days and smooths the peaks. For '
              'every window, between 6.0% and 6.7% of the days fall outside the band ±2 rolling standard deviations, against 4.6% for the Normal distribution.'),
  'histogram': ('▶ raises the threshold from 4 to 6 standard deviations. How does the ratio of the observed number of days beyond the threshold to the number expected under the Normal distribution change?',
                'It grows enormously: at 4 standard deviations, 46 days against 0.43 expected (108 times as many); at 6 standard deviations, 12 days against 0.000013 '
                '(about 906,000 times as many). The tail of the Normal distribution falls much faster than the tail of the returns, so risk computed with the Normal '
                'distribution is underestimated in the tails (Chapters 2, 5 and 10).'),
 },
}

# question shown with the options; (key, [(RO, EN), ...])
PREDICT = {
 'markets': (0, [('Bitcoin, cu un randament negativ.', 'Bitcoin, with a negative return.'),
                 ('DAX, ca în fereastra din 2015.', 'The DAX, as in the window from 2015.'),
                 ('BET-TR, care avea cea mai mică volatilitate în fereastra din 2015.', 'The BET-TR, which had the lowest volatility in the window from 2015.')]),
 'risk-return': (0, [('Sub zero, cu o volatilitate tot de circa 70%.', 'Below zero, with a volatility still near 70%.'),
                     ('Tot cel mai sus: riscul cel mai mare aduce randamentul cel mai mare.', 'Still at the top: the highest risk brings the highest return.'),
                     ('Lîngă indicii bursieri: volatilitatea lui scade la circa 20%.', 'Next to the equity indices: its volatility falls to about 20%.')]),
 'drawdowns': (0, [('În 2021.', 'In 2021.'),
                   ('În 2013, ca S&P 500: piețele își revin împreună.', 'In 2013, like the S&P 500: markets recover together.'),
                   ('În 2010: după o scădere de 82,5% ajunge o creștere de 82,5%.', 'In 2010: after a fall of 82.5%, a rise of 82.5% is enough.')]),
 'vix': (0, [('Aproximativ 88%.', 'About 88%.'),
             ('Aproximativ 9%, cît reprezintă zilele peste 30 din total.', 'About 9%, the share of all days above 30.'),
             ('Aproximativ 50%: a doua zi VIX poate urca sau coborî.', 'About 50%: the next day the VIX can go up or down.')]),
 'returns': (0, [('Scade la mai puțin de jumătate.', 'It falls to less than half.'),
                 ('Rămîne aceeași: datele sînt aceleași.', 'It stays the same: the data are the same.'),
                 ('Crește: o fereastră lungă cuprinde mai multe zile extreme.', 'It rises: a long window contains more extreme days.')]),
 'histogram': (0, [('Crește foarte mult.', 'It grows enormously.'),
                   ('Rămîne în jur de 100: cozile groase înmulțesc toate probabilitățile cu același factor.', 'It stays near 100: heavy tails multiply all probabilities by the same factor.'),
                   ('Scade: dincolo de 6 abateri standard aproape că nu mai există zile.', 'It falls: beyond 6 standard deviations there are almost no days left.')]),
}

# all questions ask what ▶ will do or what the readout states: veil the readout
VEIL = {a: {'out': True} for a in PREDICT}

# Exit ticket „Verifică-te” / "Check Yourself": the slide „Autoevaluare” and the worked numbers of the lecture.
TICKET = [
 dict(q=('Un preț trece de la 100 la 110 și apoi la 99. De ce suma randamentelor simple nu dă randamentul total?',
         'A price goes from 100 to 110 and then to 99. Why does the sum of the simple returns not give the total return?'),
      key=0, opts=[
  (('Randamentele simple se compun prin înmulțire: +10% și −10% dau −1%, nu 0; randamentele logaritmice se adună.',
    'Simple returns compound by multiplication: +10% and −10% give −1%, not 0; log returns add up.'),
   ('(1,10)(0,90) − 1 = −1%; ln 1,10 + ln 0,90 = 9,53% − 10,54% = −1,01% = ln 0,99.', '(1.10)(0.90) − 1 = −1%; ln 1.10 + ln 0.90 = 9.53% − 10.54% = −1.01% = ln 0.99.')),
  (('Suma dă randamentul total; diferența de 1% este o eroare de rotunjire.', 'The sum does give the total return; the 1% gap is a rounding error.'),
   ('Diferența nu vine din rotunjire: 100 × 1,10 × 0,90 = 99 exact.', 'The gap is not rounding: 100 × 1.10 × 0.90 = 99 exactly.')),
  (('Pentru că a doua perioadă pornește de la 100, nu de la 110.', 'Because the second period starts from 100, not from 110.'),
   ('Este invers: a doua perioadă pornește de la 110, iar −10% din 110 este 11, mai mult decît cîștigul de 10.', 'It is the other way round: the second period starts from 110, and −10% of 110 is 11, more than the gain of 10.'))]),
 dict(q=('Bitcoin are o volatilitate zilnică de 3%. Cît este volatilitatea anuală?', 'Bitcoin has a daily volatility of 3%. What is its annual volatility?'),
      key=0, opts=[
  (('√365 × 3% = 57,3%.', '√365 × 3% = 57.3%.'), ('Bitcoin se tranzacționează în fiecare zi: A = 365, iar varianțele zilelor independente se adună.',
                                                  'Bitcoin trades every day: A = 365, and the variances of independent days add up.')),
  (('√252 × 3% = 47,6%.', '√252 × 3% = 47.6%.'), ('252 este numărul de zile de tranzacționare al burselor, nu al Bitcoin.', '252 is the number of trading days of exchanges, not of Bitcoin.')),
  (('365 × 3% = 1 095%.', '365 × 3% = 1,095%.'), ('Se înmulțește cu A media, nu volatilitatea; volatilitatea se înmulțește cu √A.', 'The mean is multiplied by A, not the volatility; the volatility is multiplied by √A.'))]),
 dict(q=('Ce înseamnă pentru un investitor un drawdown maxim de −80%?', 'What does a maximum drawdown of −80% mean for an investor?'),
      key=0, opts=[
  (('Pentru a reveni la vîrf, prețul trebuie să crească de cinci ori (+400%).', 'To regain the peak, the price must grow five times (+400%).'),
   ('0,8/(1 − 0,8) = 4; BET, după −82,5% în 2009, a avut nevoie de +473% și de peste 13 ani pînă în martie 2021.',
    '0.8/(1 − 0.8) = 4; the BET, after −82.5% in 2009, needed +473% and more than 13 years, until March 2021.')),
  (('Pentru a reveni la vîrf, prețul trebuie să crească cu 80%.', 'To regain the peak, the price must rise by 80%.'),
   ('O creștere de 80% de la 20 duce la 36, departe de 100.', 'A rise of 80% from 20 leads to 36, far from 100.')),
  (('A pierdut 80% din randamentul anual mediu.', 'It lost 80% of the mean annual return.'),
   ('Drawdown-ul compară prețul cu cel mai mare preț anterior, nu cu un randament mediu.', 'The drawdown compares the price with the highest earlier price, not with a mean return.'))]),
 dict(q=('Din 2000, 46 de zile ale S&P 500 se află la peste 4 abateri standard de medie. Ce arată acest număr?',
         'Since 2000, 46 days of the S&P 500 lie more than 4 standard deviations from the mean. What does this number show?'),
      key=0, opts=[
  (('Cozi mai groase decît la distribuția Normală, care ar da 0,43 zile.', 'Tails heavier than those of the Normal distribution, which would give 0.43 days.'),
   ('6 714 × 2 × P(Z > 4) = 0,43; coeficientul de boltire este 13,7, față de 3 la distribuția Normală.', '6,714 × 2 × P(Z > 4) = 0.43; the kurtosis is 13.7, against 3 for the Normal distribution.')),
  (('Erori în date: zilele de peste 4 abateri standard ar trebui eliminate.', 'Errors in the data: days beyond 4 standard deviations should be removed.'),
   ('Sînt zile reale (13 octombrie 2008, 16 martie 2020); tocmai ele contează pentru risc.', 'They are real days (13 October 2008, 16 March 2020); they are exactly the ones that matter for risk.')),
  (('Un eșantion prea mic: cu mai multe date, numărul s-ar apropia de cel al distribuției Normale.', 'A sample too small: with more data the count would approach that of the Normal distribution.'),
   ('Cu 6 714 zile, distribuția Normală ar da sub o zi; numărul observat este de peste 100 de ori mai mare.', 'With 6,714 days the Normal distribution would give less than one day; the observed count is more than 100 times as large.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'sfm-ch0-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH0_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH0_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n'
                          + 'window.CH0_TEXT.veil = ' + json.dumps(VEIL) + ';\n')


def _ro_numbers():
    """RO: a non-breaking space as the thousands separator (6 714, 906 000)."""
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
    build(sys.modules[__name__])
