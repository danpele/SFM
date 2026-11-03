// Chapter 1 interactive page: one block per chart slide, computed in the browser from ch1_data.js.
(function () {
'use strict';
const D = window.CH1_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH1_TEXT;
const COL = D.colors, SH = D.short, LAB = D.labels;
const RO = LANG === 'ro';

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
const dayOf = s => Date.parse(s) / 864e5;
const CACHE = {};
function raw(key) {                 // {days, dates, v: {column: values}} of one exported table
  if (CACHE[key]) return CACHE[key];
  const R = D.s[key], days = [R.d0];
  for (const dd of R.dd) days.push(days[days.length - 1] + dd);
  return (CACHE[key] = { days, dates: days.map(iso), v: R.v });
}
function slice(S, d0) {             // rows of S from day d0 on
  const i = S.days.findIndex(d => d >= d0), v = {};
  for (const c in S.v) v[c] = S.v[c].slice(i);
  return { days: S.days.slice(i), dates: S.dates.slice(i), v };
}
function asset(k, y0) {             // close of asset k from START (or from 1 January of year y0), {days, dates, p}
  const key = k + (y0 || 0);
  if (CACHE[key]) return CACHE[key];
  const S = D.s[k] ? raw(k) : slice(raw(k + '_all'), dayOf(D.start));
  const X = y0 && y0 > 2015 ? slice(S, dayOf(y0 + '-01-01')) : S;
  return (CACHE[key] = { days: X.days, dates: X.dates, p: X.v[k] });
}
function full(k) { const S = raw(k + '_all'); return { days: S.days, dates: S.dates, p: S.v[k] }; }

// ================================================================= statistics (same formulas as generate_all_charts.py)
const mean = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
const sd = a => { const m = mean(a); let s = 0; for (const x of a) s += (x - m) ** 2; return Math.sqrt(s / (a.length - 1)); };
const logRet = p => p.slice(1).map((x, i) => Math.log(x / p[i]));
const simpleRet = p => p.slice(1).map((x, i) => x / p[i] - 1);
const perYear = (n, d0, d1) => n / ((d1 - d0) / 365.25);           // periods_per_year: observations per calendar year
function quantile(a, q) { const s = Float64Array.from(a).sort(), h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
function drawdown(p) { let m = -Infinity; return p.map(x => { m = Math.max(m, x); return x / m - 1; }); }
function performance(A) {           // performance() of generate_all_charts.py, risk-free rate 0
  const { p, days } = A, R = simpleRet(p), r = logRet(p), n = R.length;
  const q = perYear(n, days[1], days[days.length - 1]), m = mean(R), s = sd(R);
  const muA = m * q, vol = s * Math.sqrt(q), srd = m / s, sr = srd * Math.sqrt(q), se = Math.sqrt((1 + 0.5 * srd * srd) / n) * Math.sqrt(q);
  let dn = 0; for (const x of R) dn += Math.min(x, 0) ** 2; dn = Math.sqrt(dn / n);
  const years = (days[days.length - 1] - days[0]) / 365.25, cagr = (p[p.length - 1] / p[0]) ** (1 / years) - 1;
  const mdd = Math.min(...drawdown(p)), muL = mean(r) * q;
  return { n, q, years, muA, muL, cagr, vol, drag: muA - muL, halfVar: 0.5 * vol * vol, sharpe: sr, se, sortino: muA / (dn * Math.sqrt(q)), mdd, calmar: cagr / Math.abs(mdd) };
}
function rollSd(r, w) {             // pandas rolling(w).std(), ddof 1
  const o = new Array(r.length).fill(null); let s = 0, s2 = 0;
  for (let i = 0; i < r.length; i++) {
    s += r[i]; s2 += r[i] * r[i];
    if (i >= w) { s -= r[i - w]; s2 -= r[i - w] * r[i - w]; }
    if (i >= w - 1) o[i] = Math.sqrt(Math.max(0, (s2 - s * s / w) / (w - 1)));
  }
  return o;
}
const SQ2PI = Math.sqrt(2 * Math.PI);
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const normSf = x => 0.5 * erfc(x / Math.SQRT2);

// ================================================================= formatting and panels
const sep = RO ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep);
const MON = RO ? ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
  : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const fdate = s => { const [y, m, d] = s.split('-'); return `${+d} ${MON[+m - 1]} ${y}`; };
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
// Romanian „de” between a number and its noun (20 de zile, 112 zile, 100 de zile); empty in English
const de = n => (RO && (n % 100 >= 20 || (n >= 100 && n % 100 === 0)) ? ' de' : '');
const nint = n => fint(n) + de(Math.round(n));
const sgn = (x, d) => (x > 0 ? '+' : '') + nf(x, d);
const XT = (title, extra) => AX(title, extra);
const label = id => $(id + '-v');
const H = h => h + (document.documentElement.clientWidth < 600 ? 40 : 0);   // a little taller on phones (legend rows)
// n panels in one chart: side by side when the chart is wide enough, stacked otherwise (as in ch2.js)
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.stack || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 250) * n + (o.extra || 70) : (o.h || 380)) + 'px';
  const g = stack ? (o.vgap || 0.1) : (o.gap || 0.09), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', dom = [i * (w + g), i * (w + g) + w];
    const dy = stack ? [1 - dom[1], 1 - dom[0]] : [0, 1];
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: dy, anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const hl = (y, color, dash, axes) => ({ type: 'line', xref: (axes ? axes.x : 'x') + ' domain', yref: axes ? axes.y : 'y', x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 1.2, dash: dash || 'solid' } });
const LOGT = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
const DATE = { type: 'date', gridcolor: C.grid, color: C.navy, tickfont: { size: 11, color: C.navy }, automargin: true };

// ================================================================= 1. Banca Transilvania: close vs adjusted close
function tlvAdjusted() {
  const th = +$('tl-th').value; label('tl-th').textContent = nf(th, 1) + '%';
  const S = raw('tlvpair'), c = S.v.close, a = S.v.adjusted_close, ev = [];
  for (let i = 1; i < c.length; i++) {
    const d = 100 * (Math.log(c[i] / c[i - 1]) - Math.log(a[i] / a[i - 1]));
    if (Math.abs(d) > th) ev.push({ i, d, rc: 100 * Math.log(c[i] / c[i - 1]), ra: 100 * Math.log(a[i] / a[i - 1]) });
  }
  const up = ev.filter(e => e.d > 0).length;
  $('tl-chart').style.height = H(380) + 'px';
  plot('tl-chart', [
    { x: S.dates, y: c, mode: 'lines', name: 'Close (as traded)', line: { color: COL.bet, width: 1.1 }, hovertemplate: '%{x}: close %{y:.4f} RON<extra></extra>' },
    { x: S.dates, y: a, mode: 'lines', name: 'Adjusted close', line: { color: COL.tlv, width: 1.1 }, hovertemplate: '%{x}: adjusted %{y:.4f} RON<extra></extra>' },
    { x: ev.map(e => S.dates[e.i]), y: ev.map(e => c[e.i]), mode: 'markers', name: `days where the two log returns differ by more than ${th.toFixed(1)}%`,
      marker: { color: C.navy, size: 7, symbol: 'diamond' }, text: ev.map(e => `close ${e.rc.toFixed(1)}%, adjusted ${e.ra.toFixed(1)}%`), hovertemplate: '%{x}: %{text}<extra></extra>' }
  ], layout({ xaxis: DATE, yaxis: XT('Price, RON (log scale)', { type: 'log', tickvals: LOGT }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  const big = ev.reduce((b, e) => (!b || Math.abs(e.d) > Math.abs(b.d) ? e : b), null);
  out('tl-out', [T.tl(nf(th, 1), ev.length, up, ev.length - up),
    big ? T.tlBig(fdate(S.dates[big.i]), sgn(big.rc, 1), sgn(big.ra, 1)) : '']);
}

// ================================================================= 2. BET vs BET-TR
function betBettr() {
  const y = +$('bb-y').value; label('bb-y').textContent = y;
  const S0 = raw('betpair'), S = y > 2014 ? slice(S0, dayOf(y + '-01-01')) : S0;
  const b = S.v.bet, t = S.v.bettr, ib = b.map(x => 100 * x / b[0]), it = t.map(x => 100 * x / t[0]);
  const yrs = (S.days[S.days.length - 1] - S.days[0]) / 365.25, gB = (b[b.length - 1] / b[0]) ** (1 / yrs) - 1, gT = (t[t.length - 1] / t[0]) ** (1 / yrs) - 1;
  $('bb-chart').style.height = H(380) + 'px';
  plot('bb-chart', [
    { x: S.dates, y: ib, mode: 'lines', name: 'BET (price index)', line: { color: COL.bet, width: 1.3 }, hovertemplate: '%{x}: BET %{y:.0f}<extra></extra>' },
    { x: S.dates, y: it, mode: 'lines', name: 'BET-TR (dividends reinvested)', line: { color: COL.bettr, width: 1.3 }, hovertemplate: '%{x}: BET-TR %{y:.0f}<extra></extra>' }
  ], layout({ xaxis: DATE, yaxis: XT('Value of 100 invested', { rangemode: 'tozero' }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  out('bb-out', [T.bb(fdate(S.dates[0]), fint(ib[ib.length - 1]), fint(it[it.length - 1])),
    T.bb2(nf(100 * gB, 1), nf(100 * gT, 1), nf(100 * (gT - gB), 1))]);
}

// ================================================================= 3. EUR/RON: two series
function eurron() {
  const th = +$('eu-th').value; label('eu-th').textContent = nf(th, 2);
  const S = raw('eurpair'), b = S.v.eurron, e = S.v.eurron_eodhd, rb = [], re = [], dates = [];
  for (let i = 1; i < b.length; i++) { rb.push(100 * Math.log(b[i] / b[i - 1])); re.push(100 * Math.log(e[i] / e[i - 1])); dates.push(S.dates[i]); }
  const q = perYear(rb.length, S.days[1], S.days[S.days.length - 1]), keep = [], bad = [];
  rb.forEach((x, i) => (Math.abs(re[i] - x) > th ? bad : keep).push(i));
  const vol = (r, ix) => sd(ix.map(i => r[i])) * Math.sqrt(q);
  $('eu-chart').style.height = H(380) + 'px';
  plot('eu-chart', [
    { x: dates, y: re, mode: 'lines', name: 'EUR/RON series from EODHD', line: { color: COL.eurron_eodhd, width: 0.7 }, hovertemplate: '%{x}: EODHD %{y:.2f}%<extra></extra>' },
    { x: dates, y: rb, mode: 'lines', name: 'EUR/RON, BNR reference rate', line: { color: COL.eurron, width: 0.7 }, hovertemplate: '%{x}: BNR %{y:.2f}%<extra></extra>' },
    { x: bad.map(i => dates[i]), y: bad.map(i => re[i]), mode: 'markers', name: `EODHD days differing by more than ${th.toFixed(2)} pp`, marker: { color: C.navy, size: 4 },
      hovertemplate: '%{x}: EODHD %{y:.2f}%<extra></extra>' }
  ], layout({ xaxis: DATE, yaxis: XT('Daily log return (%)'), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  const all = rb.map((_, i) => i);
  out('eu-out', [T.eu(nint(rb.length), nf(th, 2), nint(bad.length)),
    T.eu2(nf(vol(re, all), 1), nf(vol(rb, all), 1), nf(vol(re, keep), 1), nf(vol(rb, keep), 1))]);
}

// ================================================================= 4. S&P 500: price level and daily returns
function priceReturns() {
  const k = +$('pr-k').value; label('pr-k').textContent = k || (RO ? 'fără' : 'off');
  const A = full('sp500'), r = logRet(A.p).map(x => 100 * x), rd = A.dates.slice(1);
  const Pn = panels('pr-chart', 2, { stack: true, hs: 200, extra: 80, vgap: 0.08 }), [a, b] = Pn.ax;
  const tr = [
    { x: A.dates, y: A.p, mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'S&P 500 price', line: { color: COL.sp500, width: 1 }, hovertemplate: '%{x}: %{y:.0f}<extra></extra>' },
    { x: rd, y: r, mode: 'lines', xaxis: b.x, yaxis: b.y, name: 'S&P 500 daily log return (%)', line: { color: COL.bet, width: 0.5 }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }
  ];
  const rows = [];
  if (k) {
    const n = r.length, seg = [];
    for (let j = 0; j < k; j++) seg.push([Math.round(j * n / k), Math.round((j + 1) * n / k)]);
    seg.forEach(([i0, i1], j) => {
      const pp = A.p.slice(i0 + 1, i1 + 1), rr = r.slice(i0, i1), x = [rd[i0], rd[i1 - 1]], mp = mean(pp), mr = mean(rr), sp = sd(pp), sr = sd(rr);
      const st = { showlegend: j === 0, legendgroup: 'm' };
      tr.push({ x, y: [mp, mp], mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'subperiod mean', line: { color: C.navy, width: 2.5 }, hovertemplate: `mean ${mp.toFixed(0)}<extra></extra>`, ...st });
      tr.push({ x, y: [mr, mr], mode: 'lines', xaxis: b.x, yaxis: b.y, name: 'subperiod mean', line: { color: C.navy, width: 2.5 }, showlegend: false, hovertemplate: `mean ${mr.toFixed(3)}%<extra></extra>` });
      [1, -1].forEach(s => {
        tr.push({ x, y: [mp + s * sp, mp + s * sp], mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'mean ± 1 standard deviation', legendgroup: 's', showlegend: j === 0 && s > 0, line: { color: C.navy, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' });
        tr.push({ x, y: [mr + s * sr, mr + s * sr], mode: 'lines', xaxis: b.x, yaxis: b.y, showlegend: false, line: { color: C.navy, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' });
      });
      rows.push({ y0: rd[i0].slice(0, 4), y1: rd[i1 - 1].slice(0, 4), mp, sp, mr, sr });
    });
  }
  setAx(Pn, 0, Object.assign({}, DATE, { showticklabels: false }), XT('Index points'));
  setAx(Pn, 1, Object.assign({}, DATE, { matches: 'x' }), XT('Log return (%)'));
  plot('pr-chart', tr, layout(Object.assign({ margin: { l: 60, r: 12, t: 12, b: 40 }, legend: Object.assign(layout().legend, { y: -0.08 }) }, Pn.L)));
  const ad = D.adf, items = [T.pr(nf(ad.adf_price, 2), nf(ad.crit5_ct, 2), nf(ad.p_price, 2), nf(ad.adf_ret, 1), nf(ad.crit5_c, 2))];
  if (k) items.push(T.prSeg(rows.map(o => T.prItem(o.y0, o.y1, fint(o.mp), fint(o.sp), nf(o.mr, 3), nf(o.sr, 2))).join('; ')));
  out('pr-out', items);
}

// ================================================================= 5. simple vs log returns on real extremes
function simpleLogReturns() {
  const R = +$('sl-R').value / 100; label('sl-R').textContent = sgn(100 * R, 1) + '%';
  const xs = range(-60, 60, 0.3), lr = 100 * Math.log1p(R);
  const ext = [];
  [['btc', 'circle'], ['sp500', 'square']].forEach(([k, sym]) => {
    const A = full(k), Rk = simpleRet(A.p); let lo = 0, hi = 0;
    Rk.forEach((x, i) => { if (x < Rk[lo]) lo = i; if (x > Rk[hi]) hi = i; });
    ext.push({ k, sym, pts: [lo, hi].map(i => ({ d: A.dates[i + 1], R: 100 * Rk[i], r: 100 * Math.log1p(Rk[i]) })) });
  });
  const tr = [
    { x: xs, y: xs, mode: 'lines', name: 'r = R (45-degree line)', line: { color: C.navy, width: 0.9, dash: 'dash' }, hoverinfo: 'skip' },
    { x: xs, y: xs.map(v => 100 * Math.log1p(v / 100)), mode: 'lines', name: 'r = ln(1 + R)', line: { color: COL.sp500, width: 1.8 }, hovertemplate: 'R %{x:.1f}%: r %{y:.2f}%<extra></extra>' },
    { x: xs, y: xs.map(v => v - v * v / 200), mode: 'lines', name: 'R − R²/2 (Taylor, second order)', line: { color: COL.snp, width: 1.2, dash: 'dot' }, hovertemplate: 'R %{x:.1f}%: R − R²/2 = %{y:.2f}%<extra></extra>' },
    ...ext.map(e => ({ x: e.pts.map(p => p.R), y: e.pts.map(p => p.r), text: e.pts.map(p => p.d), mode: 'markers', name: `${LAB[e.k]}, largest daily moves`,
      marker: { color: COL[e.k], symbol: e.sym, size: 9 }, hovertemplate: '%{text}: R %{x:.1f}%, r %{y:.1f}%<extra></extra>' })),
    { x: [100 * R, 100 * R], y: [100 * R, lr], mode: 'lines+markers', name: `chosen R = ${(100 * R).toFixed(1)}%: gap R − r`, line: { color: COL.bet, width: 2.5 }, marker: { color: COL.bet, size: 7 }, hoverinfo: 'skip' }
  ];
  $('sl-chart').style.height = H(420) + 'px';
  plot('sl-chart', tr, layout({ xaxis: XT('Simple return R (%)', { range: [-62, 62] }), yaxis: XT('Log return r (%)', { range: [-95, 65] }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  const gap = 100 * R - lr, tay = 100 * R - 100 * R * R / 2;
  out('sl-out', [T.sl(sgn(100 * R, 1), sgn(lr, 3), sgn(tay, 3), nf(100 * gap, 1), nf(gap, 2)),
    T.slExt(ext.map(e => e.pts.map(p => T.slItem(LAB[e.k], fdate(p.d), sgn(p.R, 1), sgn(p.r, 1))).join('; ')).join('; '))]);
}

// ================================================================= 6. growth of 100 invested
const G4 = ['sp500', 'dax', 'bet', 'btc'];
function growth() {
  const y = +$('gr-y').value, lg = $('gr-sc').value === 'log'; label('gr-y').textContent = y;
  // union of dates, forward fill, rows from the first day on which every series has a price (pandas ffill().dropna())
  const A = G4.map(k => asset(k)), all = [...new Set(A.flatMap(a => a.days))].sort((u, v) => u - v);
  const vals = A.map(a => { const m = new Map(a.days.map((d, i) => [d, a.p[i]])); let last = null; return all.map(d => (m.has(d) ? (last = m.get(d)) : last)); });
  let i0 = all.findIndex((d, i) => vals.every(v => v[i] !== null) && d >= dayOf(y + '-01-01'));
  const dates = all.slice(i0).map(iso), idx = vals.map(v => v.slice(i0).map(x => 100 * x / v[i0]));
  $('gr-chart').style.height = H(400) + 'px';
  plot('gr-chart', G4.map((k, j) => ({ x: dates, y: idx[j], mode: 'lines', name: LAB[k], line: { color: COL[k], width: 1.3 }, hovertemplate: `${LAB[k]} %{x}: %{y:.0f}<extra></extra>` })),
    layout({ xaxis: DATE, yaxis: XT(lg ? 'Value of 100 invested (log scale)' : 'Value of 100 invested', lg ? { type: 'log', tickvals: LOGT } : { type: 'linear' }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  const end = idx.map(v => v[v.length - 1]);
  out('gr-out', [T.gr(fdate(dates[0]), G4.map((k, j) => `${LAB[k]} ${fint(end[j])}`).join(', ')),
    T.gr2(G4.map((k, j) => T.grItem(LAB[k], fint(end[j] - 100), nf(Math.log(end[j] / 100), 2))).join('; '))]);
}

// ================================================================= 7. histograms vs the Normal distribution
const H2 = ['sp500', 'btc'];
function histNormal() {
  const k = +$('hi-k').value; label('hi-k').textContent = nf(k, 1);
  const Pn = panels('hi-chart', 2, { h: 360 }), tr = [], shapes = [], items = [];
  H2.forEach((a, j) => {
    const r = logRet(asset(a).p).map(x => 100 * x), m = mean(r), s = sd(r), ax = Pn.ax[j];
    const lo = quantile(r, 0.001), hi = quantile(r, 0.999), c = r.map(x => Math.min(hi, Math.max(lo, x)));
    const nb = 90, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
    c.forEach(x => { cnt[Math.min(nb - 1, Math.floor((x - lo) / w))]++; });
    const mid = cnt.map((_, i) => lo + (i + 0.5) * w), xs = range(0, 299, 1).map(i => lo + i * (hi - lo) / 299);
    tr.push({ x: mid, y: cnt.map(v => v / (r.length * w)), type: 'bar', xaxis: ax.x, yaxis: ax.y, name: `${LAB[a]} (histogram)`, marker: { color: COL[a], opacity: 0.55 }, width: w, hovertemplate: '%{x:.2f}%: density %{y:.3f}<extra></extra>' });
    tr.push({ x: xs, y: xs.map(v => Math.exp(-0.5 * ((v - m) / s) ** 2) / (s * SQ2PI)), mode: 'lines', xaxis: ax.x, yaxis: ax.y, name: 'Normal density, same mean and s.d.', showlegend: j === 0, legendgroup: 'n', line: { color: C.navy, width: 1.3 }, hoverinfo: 'skip' });
    [m - k * s, m + k * s].forEach(v => shapes.push({ type: 'line', xref: ax.x, yref: ax.y + ' domain', x0: v, x1: v, y0: 0, y1: 1, line: { color: COL.bet, width: 1.2, dash: 'dash' } }));
    let nOut = 0; r.forEach(x => { if (Math.abs(x - m) > k * s) nOut++; });
    const pE = nOut / r.length, pN = 2 * normSf(k);
    const ra = pE / pN;
    items.push(T.hiItem(LAB[a], nint(nOut), fint(r.length), pc(pE, 2), ra >= 100 ? nint(ra) : nf(ra, 1), pc(mean(r.map(x => (Math.abs(x - m) <= s ? 1 : 0))), 1)));
    setAx(Pn, j, XT('Daily log return (%)', { range: [lo, hi] }), XT('', { showticklabels: false }));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', name: `mean ± ${k.toFixed(1)} standard deviations`, line: { color: COL.bet, width: 1.2, dash: 'dash' } });
  plot('hi-chart', tr, layout(Object.assign({ shapes, bargap: 0, margin: { l: 20, r: 12, t: 16, b: 40 } }, Pn.L)));
  const pN = 2 * normSf(k);
  const x2 = 100 * pN;
  out('hi-out', [T.hi(nf(k, 1), nf(x2, Math.max(0, 1 - Math.floor(Math.log10(x2)))) + '%'),
    ...items]);
}

// ================================================================= 8. rolling one-year volatility
const RV3 = ['sp500', 'bet', 'btc'];
function rollingVol() {
  const w = +$('rv-w').value; label('rv-w').textContent = w + (RO ? ' zile' : ' days');
  const tr = [], items = [];
  RV3.forEach(k => {
    const A = asset(k), r = logRet(A.p).map(x => 100 * x), q = perYear(r.length, A.days[1], A.days[A.days.length - 1]);
    const ww = k === 'btc' ? Math.round(w * q / 250) : w, v = rollSd(r, ww).map(x => (x === null ? null : x * Math.sqrt(q))), d = A.dates.slice(1);
    let hi = -1, lo = -1; v.forEach((x, i) => { if (x === null) return; if (hi < 0 || x > v[hi]) hi = i; if (lo < 0 || x < v[lo]) lo = i; });
    tr.push({ x: d, y: v, mode: 'lines', name: k === 'btc' ? `${LAB[k]} (${ww}-day window)` : LAB[k], line: { color: COL[k], width: 1.2 }, hovertemplate: `${LAB[k]} %{x}: %{y:.1f}%<extra></extra>` });
    items.push(T.rvItem(LAB[k], nint(ww), nf(v[lo], 1), nf(v[hi], 1), fdate(d[hi])));
  });
  $('rv-chart').style.height = H(380) + 'px';
  plot('rv-chart', tr, layout({ xaxis: DATE, yaxis: XT('Annualised volatility (%)', { rangemode: 'tozero' }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  out('rv-out', [T.rv(nint(w)), ...items]);
}

// ================================================================= 9. Sharpe ratios with confidence intervals
const PERF = {};
function perfAll(y) { return PERF[y] || (PERF[y] = Object.fromEntries(D.assets.map(k => [k, performance(asset(k, y))]))); }
const ZQ = { 90: 1.6448536, 95: 1.959964, 99: 2.5758293 };
function sharpeCI() {
  const y = +$('sc-y').value, cl = +$('sc-cl').value, z = cl === 95 ? 1.96 : ZQ[cl]; label('sc-y').textContent = y;
  const P = perfAll(y), ks = D.assets.slice().sort((a, b) => P[a].sharpe - P[b].sharpe), names = ks.map(k => LAB[k]);
  $('sc-chart').style.height = H(420) + 'px';
  plot('sc-chart', [
    { x: ks.map(k => P[k].sharpe), y: names, mode: 'markers', name: `Sharpe ratio with ${cl}% interval`, marker: { color: COL.sp500, size: 8 },
      error_x: { type: 'data', array: ks.map(k => z * P[k].se), color: COL.bet, thickness: 1.5, width: 4 },
      customdata: ks.map(k => [P[k].sharpe - z * P[k].se, P[k].sharpe + z * P[k].se]), hovertemplate: '%{y}: %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>' }
  ], layout({ shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: C.navy, width: 1 } }],
    xaxis: XT('Annualised Sharpe ratio (risk-free rate 0)'), yaxis: XT('', { automargin: true }), margin: { l: 130, r: 12, t: 16, b: 40 } }));
  const hw = mean(ks.map(k => z * P[k].se)), zero = ks.filter(k => P[k].sharpe - z * P[k].se <= 0).map(k => LAB[k]), top = ks[ks.length - 1];
  const yrs = mean(ks.map(k => P[k].years)), lo = ks.map(k => P[k].sharpe - z * P[k].se), hi = ks.map(k => P[k].sharpe + z * P[k].se);
  const overlap = Math.max(...lo) <= Math.min(...hi);
  out('sc-out', [T.sc(fdate(asset('sp500', y).dates[0]), nf(yrs, 1), cl, nf(hw, 2)),
    T.sc2(LAB[top], nf(P[top].sharpe, 2), zero.length ? zero.join(', ') : '—', overlap)]);
}

// ================================================================= 10. drawdowns over the full history
const DD3 = ['sp500', 'bet', 'btc'];
function drawdowns() {
  const x = +$('dd-x').value; label('dd-x').textContent = x + '%';
  const tr = [], items = [];
  DD3.forEach(k => {
    const A = full(k), dd = drawdown(A.p).map(v => 100 * v);
    let under = 0, im = 0; dd.forEach((v, i) => { if (v < -x) under++; if (v < dd[im]) im = i; });
    tr.push({ x: A.dates, y: dd, mode: 'lines', name: LAB[k], line: { color: COL[k], width: 1 }, hovertemplate: `${LAB[k]} %{x}: %{y:.1f}%<extra></extra>` });
    items.push(T.ddItem(LAB[k], A.dates[0].slice(0, 4), pc(under / dd.length, 1), nf(dd[im], 1)));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', name: `−${x}% threshold`, line: { color: C.navy, width: 1.2, dash: 'dash' } });
  $('dd-chart').style.height = H(380) + 'px';
  plot('dd-chart', tr, layout({ shapes: [hl(-x, C.navy, 'dash')], xaxis: DATE, yaxis: XT('Drawdown from the previous peak (%)', { range: [-100, 3] }), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  out('dd-out', [T.dd(x, nf(100 * x / (100 - x), 0)), ...items]);
}

// ================================================================= 11. volatility drag on real data
const SYM = { o: 'circle', '*': 'star', s: 'square' };
function volDrag() {
  const s = +$('vd-s').value / 100; label('vd-s').textContent = Math.round(100 * s) + '%';
  const P = perfAll(2015), Pn = panels('vd-chart', 2, { h: 600, hs: 320, extra: 300, vgap: 0.2 }), [a, b] = Pn.ax, tr = [];
  const xs = D.assets.map(k => 100 * P[k].halfVar), ys = D.assets.map(k => 100 * P[k].drag);
  const lo = Math.min(...xs, ...ys, 100 * s * s / 2 || 1) * 0.8, hi = Math.max(...xs, ...ys, 100 * s * s / 2) * 1.25;
  tr.push({ x: [lo, hi], y: [lo, hi], mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'drag = σ²/2', line: { color: C.navy, width: 0.9, dash: 'dash' }, hoverinfo: 'skip' });
  D.assets.forEach((k, i) => {
    const [c, m] = D.scatter[k];
    tr.push({ x: [xs[i]], y: [ys[i]], mode: 'markers', xaxis: a.x, yaxis: a.y, name: LAB[k], marker: { color: c, symbol: SYM[m], size: m === '*' ? 13 : 9 },
      hovertemplate: `${LAB[k]}: σ²/2 = %{x:.2f}%, drag %{y:.2f}%<extra></extra>` });
  });
  const dg = s * s / 2;
  if (s > 0) tr.push({ x: [100 * dg], y: [100 * dg], mode: 'markers', xaxis: a.x, yaxis: a.y, name: `asset with σ = ${(100 * s).toFixed(0)}%`, marker: { color: C.navy, symbol: 'x', size: 11 }, hovertemplate: 'σ²/2 = %{x:.2f}%<extra></extra>' });
  const yrs = range(0, 10, 0.25);
  [[0.05, COL.tlv, 'σ = 5%'], [s, COL.bet, `σ = ${(100 * s).toFixed(0)}% (chosen)`]].forEach(([v, c, nm]) => {
    tr.push({ x: yrs, y: yrs.map(t => Math.exp(t * (0.10 - v * v / 2))), mode: 'lines', xaxis: b.x, yaxis: b.y, name: `1 RON, arithmetic mean 10%, ${nm}`, line: { color: c, width: 1.8 }, hovertemplate: 'year %{x}: %{y:.2f} RON<extra></extra>' });
  });
  setAx(Pn, 0, XT('Half the annual variance, σ²/2 (% per year, log scale)', { type: 'log', tickvals: LOGT }), XT('Arithmetic minus log mean (% per year)', { type: 'log', tickvals: LOGT }));
  setAx(Pn, 1, XT('Years'), XT('Wealth (RON)', { rangemode: 'tozero' }));
  plot('vd-chart', tr, layout(Object.assign({ annotations: [ptitle(a, 'Real data, 2015–2026'), ptitle(b, 'Growth exp(t(10% − σ²/2))')], margin: { l: 60, r: 12, t: 28, b: 230 },
    legend: Object.assign(layout().legend, { y: Pn.stack ? -0.12 : -0.2 }) }, Pn.L)));
  const B = P.btc;
  out('vd-out', [T.vd(Math.round(100 * s), nf(100 * dg, 2), nf(100 * (0.10 - dg), 2), nf(Math.exp(10 * (0.10 - dg)), 2)),
    T.vd2(nf(100 * B.muA, 1), nf(100 * B.muL, 1), nf(100 * B.drag, 1), nf(100 * B.halfVar, 1))]);
}

// ================================================================= 12. risk and return
const MET = { cagr: ['CAGR (%)', 100], sharpe: ['Sharpe ratio', 1], sortino: ['Sortino ratio', 1], calmar: ['Calmar ratio', 1], mdd: ['Maximum drawdown (%)', 100] };
const RR_SHORT = { tlv: 'TLV', snp: 'SNP', snn: 'SNN', tgn: 'TGN' };
// label side as in fig_risk_return (RR_LABELS), so that nearby assets do not overlap
const RR_POS = { sp500: 'middle right', dax: 'bottom right', bet: 'middle left', bettr: 'middle left', btc: 'middle left', tlv: 'middle right', snp: 'bottom left', brd: 'bottom right', snn: 'middle right', tgn: 'top left' };
function riskReturn() {
  const y = +$('rr-y').value, m = $('rr-m').value; label('rr-y').textContent = y;
  const P = perfAll(y), [yt, f] = MET[m];
  const tr = D.assets.map(k => {
    const [c, mk] = D.scatter[k];
    return { x: [100 * P[k].vol], y: [f * P[k][m]], text: [RR_SHORT[k] || LAB[k]], mode: 'markers+text', textposition: RR_POS[k], textfont: { size: 11, color: C.navy },
      name: LAB[k], marker: { color: c, symbol: SYM[mk], size: mk === '*' ? 14 : 10 }, hovertemplate: `${LAB[k]}: volatility %{x:.1f}%, ${yt} %{y:.2f}<extra></extra>` };
  });
  $('rr-chart').style.height = H(440) + 'px';
  const vs = D.assets.map(k => 100 * P[k].vol);
  plot('rr-chart', tr, layout({ shapes: [hl(0, C.navy)], xaxis: XT('Annualised volatility (%)', { range: [Math.min(...vs) - 7, Math.max(...vs) + 4] }), yaxis: XT(yt), margin: { l: 60, r: 12, t: 16, b: 40 } }));
  const best = key => LAB[D.assets.reduce((b, k) => (P[k][key] > P[b][key] ? k : b))];
  const lowV = LAB[D.assets.reduce((b, k) => (P[k].vol < P[b].vol ? k : b))], lowM = LAB[D.assets.reduce((b, k) => (P[k].mdd > P[b].mdd ? k : b))];
  out('rr-out', [T.rr(fdate(asset('sp500', y).dates[0]), best('cagr'), best('sharpe'), best('sortino'), best('calmar')), T.rr2(lowV, lowM)]);
}

// ================================================================= wiring
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['tl-th'], tlvAdjusted); PL('tl-play', 'tl-th', [0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50], 800);
  on(['bb-y'], betBettr); PL('bb-play', 'bb-y', range(2014, 2025, 1), 800);
  on(['eu-th'], eurron); PL('eu-play', 'eu-th', range(0.25, 5, 0.25), 500);
  on(['pr-k'], priceReturns); PL('pr-play', 'pr-k', range(0, 6, 1), 1000);
  on(['sl-R'], simpleLogReturns); PL('sl-play', 'sl-R', range(-60, 60, 2.5), 200);
  on(['gr-y', 'gr-sc'], growth); PL('gr-play', 'gr-y', range(2015, 2025, 1), 800);
  on(['hi-k'], histNormal); PL('hi-play', 'hi-k', range(1, 6, 0.5), 700);
  on(['rv-w'], rollingVol); PL('rv-play', 'rv-w', [21, 42, 63, 125, 250, 500], 1000);
  on(['sc-y', 'sc-cl'], sharpeCI); PL('sc-play', 'sc-y', range(2015, 2024, 1), 800);
  on(['dd-x'], drawdowns); PL('dd-play', 'dd-x', range(5, 80, 5), 500);
  on(['vd-s'], volDrag); PL('vd-play', 'vd-s', range(0, 80, 5), 400);
  on(['rr-y', 'rr-m'], riskReturn); PL('rr-play', 'rr-y', range(2015, 2024, 1), 900);
  const PFX = { 'tlv-adjusted': 'tl', 'bet-bettr': 'bb', 'eurron': 'eu', 'price-returns': 'pr', 'simple-log': 'sl', 'growth': 'gr', 'hist-normal': 'hi',
    'rolling-vol': 'rv', 'sharpe-ci': 'sc', 'drawdowns': 'dd', 'vol-drag': 'vd', 'risk-return': 'rr' };
  const VEIL = {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [tlvAdjusted, betBettr, eurron, priceReturns, simpleLogReturns, growth, histNormal, rollingVol, sharpeCI, drawdowns, volDrag, riskReturn];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH1 = { raw, asset, full, slice, performance, perfAll, drawdown, rollSd, logRet, simpleRet, quantile, mean, sd, perYear, normSf, init };
if (window.CH1_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
