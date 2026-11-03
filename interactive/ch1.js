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
function mddDates(p, days) {        // max_drawdown() of generate_all_charts.py: indices of the trough, the previous peak and the recovery (-1: none)
  const dd = drawdown(p); let it = 0; dd.forEach((v, i) => { if (v < dd[it]) it = i; });
  let ip = 0; for (let i = 0; i <= it; i++) if (p[i] > p[ip]) ip = i;
  let ir = -1; for (let i = it; i < p.length; i++) if (p[i] >= p[ip]) { ir = i; break; }
  return { mdd: dd[it], ip, it, ir };
}
function performance(A, rf) {       // performance(p, rf) of generate_all_charts.py; rf: annual risk-free rate (decimal), 0 in the lecture
  rf = rf || 0;
  const { p, days } = A, R = simpleRet(p), r = logRet(p), n = R.length;
  const q = perYear(n, days[1], days[days.length - 1]), m = mean(R), s = sd(R);
  const muA = m * q, vol = s * Math.sqrt(q), srd = (m - rf / q) / s, sr = srd * Math.sqrt(q), se = Math.sqrt((1 + 0.5 * srd * srd) / n) * Math.sqrt(q);
  let dn = 0; for (const x of R) dn += Math.min(x, 0) ** 2; dn = Math.sqrt(dn / n);
  const years = (days[days.length - 1] - days[0]) / 365.25, cagr = (p[p.length - 1] / p[0]) ** (1 / years) - 1;
  const M = mddDates(p, days), mdd = M.mdd, muL = mean(r) * q;
  return { n, q, years, m, s, srd, muA, muL, cagr, vol, drag: muA - muL, halfVar: 0.5 * vol * vol, sharpe: sr, se, sed: se / Math.sqrt(q), dn,
    sortino: (muA - rf) / (dn * Math.sqrt(q)), mdd, calmar: cagr / Math.abs(mdd),
    peak: M.ip, trough: M.it, rec: M.ir, dates: A.dates, p };
}
function resample(A, f) {           // f = 'w': last price of each week (Monday to Sunday); 'm': of each calendar month; 'd': A
  if (f === 'd') return A;
  const key = d => (f === 'w' ? Math.floor((d + 3) / 7) : iso(d).slice(0, 7)), days = [], p = [];
  A.days.forEach((d, i) => {
    if (days.length && key(days[days.length - 1]) === key(d)) { days[days.length - 1] = d; p[p.length - 1] = A.p[i]; } else { days.push(d); p.push(A.p[i]); }
  });
  return { days, dates: days.map(iso), p };
}
function describe(A, drop) {        // describe() of generate_all_charts.py: log returns in %, sd with n - 1, skewness and kurtosis with n (scipy)
  let r = logRet(A.p).map(x => 100 * x), rd = A.dates.slice(1);   // drop: dates of daily returns left out (the two TLV days of May 2016)
  if (drop) { const k = rd.map(d => !drop.includes(d)); r = r.filter((_, i) => k[i]); rd = rd.filter((_, i) => k[i]); }
  const n = r.length, q = perYear(n, dayOf(rd[0]), dayOf(rd[n - 1])), m = mean(r);
  let m2 = 0, m3 = 0, m4 = 0, lo = 0, hi = 0;
  r.forEach((x, i) => { const e = x - m; m2 += e * e; m3 += e ** 3; m4 += e ** 4; if (x < r[lo]) lo = i; if (x > r[hi]) hi = i; });
  m2 /= n; m3 /= n; m4 /= n;
  const s = sd(r), annVol = s * Math.sqrt(q), Y = n / q;
  return { n, q, Y, mean: m, sd: s, skew: m3 / m2 ** 1.5, exkurt: m4 / (m2 * m2) - 3, min: r[lo], minDate: rd[lo], max: r[hi], maxDate: rd[hi],
    annMean: m * q, annVol, se: annVol / Math.sqrt(Y) };
}
const TLV_BAD = ['2016-05-30', '2016-05-31'];   // adjusted close of 30 May 2016 out of line (bonus shares): −15.0%, then +20.5%
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
  slWE();
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
  grWE(dates, vals, i0);
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
function perfAll(y, rf) { const key = y + ':' + (rf || 0); return PERF[key] || (PERF[key] = Object.fromEntries(D.assets.map(k => [k, performance(asset(k, y), rf)]))); }
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
  ddWE();
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
  vdWE();
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

// ================================================================= tables and worked examples
function table(id, head, rows) {    // rows: arrays of cells; a cell is a string or { v, c } (class); rows may carry .cls
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}>${o.v}</${tag}>`; };
  $(id).innerHTML = `<table class="xt"><thead><tr>${head.map(h => cell(h, 'th')).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const fnum = (x, d) => { const s = nf(x, d), i = s.search(/[.,]/), a = i < 0 ? s : s.slice(0, i), b = i < 0 ? '' : s.slice(i); return a.replace(/\B(?=(\d{3})+(?!\d))/g, sep) + b; };
const fprice = x => fnum(x, x >= 100 ? 2 : 4);
const dshort = s => fdate(s);
const ndays = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / 864e5);
const FREQ = { d: 'daily', w: 'weekly', m: 'monthly' };
const SEL = (k, sel) => (k === sel ? 'sel' : '');

// worked example of block 5: a real price pair
function slWE() {
  const [k, side] = $('sl-d').value.split(':'), A = full(k), Rk = simpleRet(A.p);
  let j = 0; Rk.forEach((x, i) => { if (side === 'max' ? x > Rk[j] : x < Rk[j]) j = i; });
  const p0 = A.p[j], p1 = A.p[j + 1], R = p1 / p0 - 1, r = Math.log(p1 / p0);
  steps('sl-we', [T.slw1(LAB[k], fdate(A.dates[j]), fprice(p0), fdate(A.dates[j + 1]), fprice(p1)),
    T.slw2(fprice(p1), fprice(p0), sgn(100 * R, 2), sgn(100 * r, 2)),
    T.slw3(nf(100 * (R - r), 2), nf(100 * R * R / 2, 2), nf(100 * (R * R / 2 - R ** 3 / 3), 2))]);
}
// worked example of block 6: compounding over many periods
function grWE(dates, vals, i0) {
  const k = $('gr-a').value, j = G4.indexOf(k), v = vals[j].slice(i0), A = asset(k);
  const p0 = v[0], pT = v[v.length - 1], g = pT / p0, Y = (Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 864e5 / 365.25;
  let i1 = 0; A.days.forEach((d, i) => { if (d <= dayOf(dates[0])) i1 = i; });   // own calendar from the last price on or before the first day
  const lr = logRet(A.p.slice(i1)), Ay = asset(k, +$('gr-y').value); let s = 0; for (const x of lr) s += x;
  steps('gr-we', [T.grw1(LAB[k], fprice(p0), fdate(dates[0]), fprice(pT), fdate(dates[dates.length - 1]), nf(Y, 2)),
    T.grw2(nint(lr.length), nf(g, 3), fnum(100 * (g - 1), 1)),
    T.grw3(nint(lr.length), nf(s, 4), nf(Math.log(g), 4), fnum(100 * (Math.exp(s) - 1), 1)),
    T.grw4(nf(g, 3), nf(Y, 2), nf(100 * (g ** (1 / Y) - 1), 2), nf(s, 4), nf(100 * (Math.exp(s / Y) - 1), 2)),
    T.grw5(fdate(Ay.dates[0]), nf(100 * performance(Ay).cagr, 2), fdate(dates[0]))]);
}
// worked example of block 10: the maximum drawdown of a real price path
function ddWE() {
  const k = $('dd-a').value, A = full(k), M = mddDates(A.p, A.days), x = -M.mdd, d = A.dates;
  steps('dd-we', [T.ddw1(LAB[k], d[0].slice(0, 4), fdate(d[M.ip]), fprice(A.p[M.ip]), fdate(d[M.it]), fprice(A.p[M.it])),
    T.ddw2(fprice(A.p[M.it]), fprice(A.p[M.ip]), nf(-100 * x, 1)),
    T.ddw3(nf(x, 3), nf(100 * x / (1 - x), 0)),
    M.ir >= 0 ? T.ddw4(fdate(d[M.ir]), nint(ndays(d[M.ip], d[M.it])), nint(ndays(d[M.it], d[M.ir])), nf(ndays(d[M.ip], d[M.ir]) / 365.25, 1))
      : T.ddw4n(nint(ndays(d[M.ip], d[M.it])), fdate(d[d.length - 1]))]);
}
// worked example of block 11: arithmetic minus log mean on real data
function vdWE() {
  const k = $('vd-a').value, P = perfAll(2015)[k], r = logRet(P.p);
  steps('vd-we', [T.vdw1(LAB[k], nf(P.q, 1), nf(100 * P.m, 4), nf(100 * P.muA, 1)),
    T.vdw2(nf(100 * mean(r), 4), nf(100 * P.muL, 1)),
    T.vdw3(nf(100 * P.muA, 1), nf(100 * P.muL, 1), nf(100 * P.drag, 2), nf(100 * P.vol, 1), nf(100 * P.halfVar, 2)),
    T.vdw4(nf(P.muL, 3), nf(100 * (Math.exp(P.muL) - 1), 1), nf(100 * P.cagr, 1))]);
}

// ================================================================= X1. a TLV–SNP portfolio (frame "Real Data: a TLV–SNP Portfolio")
function pairTS(a, b) {             // pandas concat(...).dropna(): the days on which both stocks have a price
  const A = asset(a), B = asset(b), m = new Map(B.days.map((d, i) => [d, B.p[i]])), days = [], pa = [], pb = [];
  A.days.forEach((d, i) => { if (m.has(d)) { days.push(d); pa.push(A.p[i]); pb.push(m.get(d)); } });
  return { days, dates: days.map(iso), pa, pb };
}
function portfolio(w) {             // portfolio_example() of generate_all_charts.py
  const S = pairTS('tlv', 'snp'), Ra = simpleRet(S.pa), Rb = simpleRet(S.pb), n = Ra.length;
  const ex = [], ap = [], err = []; let cE = 0, cA = 0, im = 0, sAbs = 0;
  const cumE = [0], cumA = [0];
  for (let i = 0; i < n; i++) {
    const Rp = w * Ra[i] + (1 - w) * Rb[i], e = Math.log1p(Rp), a = w * Math.log1p(Ra[i]) + (1 - w) * Math.log1p(Rb[i]);
    ex.push(e); ap.push(a); err.push(1e4 * (e - a)); cE += e; cA += a; cumE.push(cE); cumA.push(cA); sAbs += Math.abs(1e4 * (e - a));
    if (err[i] > err[im]) im = i;
  }
  const Y = (S.days[S.days.length - 1] - S.days[0]) / 365.25;
  return { S, Ra, Rb, n, ex, ap, err, cumE, cumA, sumE: cE, sumA: cA, im, mae: sAbs / n, Y, wealth: Math.exp(cE), cagr: Math.exp(cE / Y) - 1,
    cagrA: (S.pa[S.pa.length - 1] / S.pa[0]) ** (1 / Y) - 1, cagrB: (S.pb[S.pb.length - 1] / S.pb[0]) ** (1 / Y) - 1 };
}
function portfolioBlock() {
  const w = +$('pf-w').value / 100; label('pf-w').textContent = Math.round(100 * w) + '% TLV, ' + Math.round(100 * (1 - w)) + '% SNP';
  const P = portfolio(w), S = P.S, d1 = S.dates.slice(1);
  const Pn = panels('pf-chart', 2, { stack: true, hs: 190, extra: 110, vgap: 0.12 }), [a, b] = Pn.ax;
  setAx(Pn, 0, Object.assign({}, DATE, { showticklabels: false }), XT('Cumulative log return'));
  setAx(Pn, 1, Object.assign({}, DATE, { matches: 'x' }), XT('Basis points', { rangemode: 'tozero' }));
  plot('pf-chart', [
    { x: S.dates, y: P.cumE, mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'exact: sum of ln(1 + R_p)', line: { color: COL.tlv, width: 1.4 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
    { x: S.dates, y: P.cumA, mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'weighted sum of log returns', line: { color: COL.snp, width: 1.4, dash: 'dash' }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
    { x: d1, y: P.err, mode: 'lines', xaxis: b.x, yaxis: b.y, name: 'daily gap (basis points)', line: { color: COL.bet, width: 0.7 }, hovertemplate: '%{x}: %{y:.2f} bp<extra></extra>' }
  ], layout(Object.assign({ margin: { l: 60, r: 12, t: 24, b: 40 }, annotations: [ptitle(a, 'Cumulative log return of the portfolio'), ptitle(b, 'Exact minus weighted log return (bp)')],
    legend: Object.assign(layout().legend, { y: -0.1 }) }, Pn.L)));
  const cl = (x, nm, cls) => Object.assign([nm, nf(x, 3), nf(Math.exp(x), 2), nf(100 * (Math.exp(x / P.Y) - 1), 1)], { cls });
  table('pf-tab', ['', T.h.clr, T.h.wealth, T.h.cagr], [cl(P.sumE, T.h.pexact, 'sel'), cl(P.sumA, T.h.papprox),
    cl(Math.log(S.pa[S.pa.length - 1] / S.pa[0]), 'Banca Transilvania (TLV)'), cl(Math.log(S.pb[S.pb.length - 1] / S.pb[0]), 'OMV Petrom (SNP)')]);
  const i = P.im, wa = Math.round(100 * w) / 100;
  steps('pf-we', [T.pfw1(fdate(S.dates[i]) + ' → ' + fdate(S.dates[i + 1]), fprice(S.pa[i]), fprice(S.pa[i + 1]), sgn(100 * P.Ra[i], 3), fprice(S.pb[i]), fprice(S.pb[i + 1]), sgn(100 * P.Rb[i], 3)),
    T.pfw2(nf(wa, 2), sgn(100 * P.Ra[i], 3), nf(1 - wa, 2), sgn(100 * P.Rb[i], 3), sgn(100 * (wa * P.Ra[i] + (1 - wa) * P.Rb[i]), 3), sgn(100 * P.ex[i], 3)),
    T.pfw3(nf(wa, 2), sgn(100 * Math.log1p(P.Ra[i]), 3), nf(1 - wa, 2), sgn(100 * Math.log1p(P.Rb[i]), 3), sgn(100 * P.ap[i], 3)),
    T.pfw4(nf(P.err[i], 1), nint(P.n), nf(P.mae, 2))]);
  out('pf-out', [T.pf(nint(P.n), nf(P.mae, 2), nf(P.err[i], 1), fdate(S.dates[i + 1])),
    T.pf2(nf(P.sumE - P.sumA, 3), nf(100 * P.cagr, 1), nf(100 * P.cagrA, 1), nf(100 * P.cagrB, 1))]);
}

// ================================================================= X2. annualisation by frequency (frame "Annualising the Mean and the Volatility")
function annualiseBlock() {
  const y = +$('an-y').value, f = $('an-f').value, k = $('an-a').value; label('an-y').textContent = y;
  const rows = D.assets.map(a => {
    const X = ['d', 'w', 'm'].map(g => describe(resample(asset(a, y), g)));
    return { a, X };
  });
  const head = ['', T.h.A, T.h.sdD, T.h.volD, T.h.vol252, T.h.volW, T.h.volM];
  table('an-tab', head, rows.map(({ a, X }) => Object.assign([SH[a], nf(X[0].q, 1), nf(X[0].sd, 2), nf(X[0].annVol, 1), nf(X[0].sd * Math.sqrt(252), 1), nf(X[1].annVol, 1), nf(X[2].annVol, 1)], { cls: SEL(a, k) })));
  $('an-chart').style.height = H(400) + 'px';
  const FC = { d: COL.sp500, w: COL.dax, m: COL.bettr };
  plot('an-chart', ['d', 'w', 'm'].map((g, j) => ({ x: rows.map(o => SH[o.a]), y: rows.map(o => o.X[j].annVol), type: 'bar', name: `from ${FREQ[g]} log returns, √A × σ`,
    marker: { color: FC[g], opacity: g === f ? 1 : 0.45, line: { color: g === f ? C.navy : FC[g], width: g === f ? 1.5 : 0 } }, hovertemplate: `%{x}, ${FREQ[g]}: %{y:.1f}%<extra></extra>` })),
    layout({ barmode: 'group', xaxis: XT('', { tickangle: -30 }), yaxis: XT('Annualised volatility (%)', { rangemode: 'tozero' }), margin: { l: 60, r: 12, t: 16, b: 70 } }));
  const X = describe(resample(asset(k, y), f)), XD = describe(asset(k, y));
  steps('an-we', [T.anw1(LAB[k], T.fq[f], fdate(asset(k, y).dates[0]), nint(X.n), nf(X.Y, 2), nf(X.q, 2)),
    T.anw2(nf(X.q, 2), nf(X.mean, 4), nf(X.annMean, 2)),
    T.anw3(nf(X.q, 2), nf(Math.sqrt(X.q), 2), nf(X.sd, 3), nf(X.annVol, 2)),
    f === 'd' ? T.anw4(nf(X.sd * Math.sqrt(252), 1), nf(X.sd * Math.sqrt(365), 1)) : T.anw4f(nf(X.annVol, 1), nf(XD.annVol, 1))]);
  const sp = rows.find(o => o.a === 'sp500').X, bt = rows.find(o => o.a === 'btc').X;
  out('an-out', [T.an(fdate(asset('sp500', y).dates[0]), nf(sp[0].annVol, 1), nf(sp[1].annVol, 1), nf(sp[2].annVol, 1)),
    T.an2(nf(bt[0].annVol, 1), nf(bt[0].sd * Math.sqrt(252), 1), nf(bt[0].q, 1))]);
}

// ================================================================= X3. precision of the mean (frame "How Precisely Do We Know the Mean?")
function meanSeBlock() {
  const y = +$('ms-y').value, cl = +$('ms-cl').value, z = cl === 95 ? 1.96 : ZQ[cl], k = $('ms-a').value; label('ms-y').textContent = y;
  const X = Object.fromEntries(D.assets.map(a => [a, describe(asset(a, y))])), ks = D.assets.slice().sort((a, b) => X[a].annMean - X[b].annMean);
  $('ms-chart').style.height = H(420) + 'px';
  plot('ms-chart', [{ x: ks.map(a => X[a].annMean), y: ks.map(a => LAB[a]), mode: 'markers', name: `annual mean log return with ${cl}% interval`, marker: { color: COL.sp500, size: 8 },
    error_x: { type: 'data', array: ks.map(a => z * X[a].se), color: COL.bet, thickness: 1.5, width: 4 },
    customdata: ks.map(a => [X[a].annMean - z * X[a].se, X[a].annMean + z * X[a].se]), hovertemplate: '%{y}: %{x:.1f}% [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>' }],
    layout({ shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: C.navy, width: 1 } }],
      xaxis: XT('Annual mean log return (%)'), yaxis: XT('', { automargin: true }), margin: { l: 130, r: 12, t: 16, b: 40 } }));
  table('ms-tab', ['', T.h.Y, T.h.mu, T.h.volD, 'SE (%)', T.h.ci(cl)], D.assets.map(a => {
    const x = X[a], lo = x.annMean - z * x.se, hi = x.annMean + z * x.se;
    return Object.assign([SH[a], nf(x.Y, 1), nf(x.annMean, 1), nf(x.annVol, 1), nf(x.se, 1), { v: `[${nf(lo, 1)}; ${nf(hi, 1)}]`, c: lo <= 0 ? 'bad' : '' }], { cls: SEL(a, k) });
  }));
  const x = X[k], lo = x.annMean - z * x.se, hi = x.annMean + z * x.se;
  steps('ms-we', [T.msw1(LAB[k], fdate(asset(k, y).dates[0]), nf(x.annMean, 2), nf(x.annVol, 2), nint(x.n), nf(x.q, 1), nf(x.Y, 2)),
    T.msw2(nf(x.annVol, 2), nf(x.Y, 2), nf(Math.sqrt(x.Y), 3), nf(x.se, 2)),
    T.msw3(cl, nf(z, 3), nf(x.annMean, 2), nf(x.se, 2), nf(lo, 1), nf(hi, 1)),
    (XM => T.msw4(nint(XM.n), nf(XM.se, 2), nint(x.n), nf(x.se, 2)))(describe(resample(asset(k, y), 'm')))]);
  const zero = ks.filter(a => X[a].annMean - z * X[a].se <= 0).map(a => LAB[a]);
  out('ms-out', [T.ms(fdate(asset('sp500', y).dates[0]), cl, zero.length ? zero.join(', ') : '—'),
    T.ms2(nf(mean(D.assets.map(a => X[a].se)), 1), nf(mean(D.assets.map(a => X[a].Y)), 1))]);
}

// ================================================================= X4–X5. descriptive statistics of daily log returns (tables)
const IDX5 = ['sp500', 'dax', 'bet', 'bettr', 'btc'], BVB5 = ['tlv', 'snp', 'brd', 'snn', 'tgn'];
function descTable(pf, ks) {
  const y = +$(pf + '-y').value, f = $(pf + '-f').value, cl = $(pf + '-t') && $(pf + '-t').value === 'clean' && f === 'd'; label(pf + '-y').textContent = y;
  const X = Object.fromEntries(ks.map(a => [a, describe(resample(asset(a, y), f), a === 'tlv' && cl ? TLV_BAD : null)]));
  table(pf + '-tab', ['', 'n', T.h.mean, T.h.sd, T.h.skew, T.h.exk, T.h.min, T.h.max, T.h.volD], ks.map(a => {
    const x = X[a];
    return [LAB[a] + (pf === 'db' && SH[a] !== LAB[a] ? ` (${SH[a]})` : ''), fint(x.n), nf(x.mean, 3), nf(x.sd, 2), nf(x.skew, 2), nf(x.exkurt, 1), `${nf(x.min, 1)} <span class="pv">(${fdate(x.minDate)})</span>`,
      `${nf(x.max, 1)} <span class="pv">(${fdate(x.maxDate)})</span>`, nf(x.annVol, 1)];
  }));
  return { y, f, X, cl };
}
function descIdxBlock() {
  const { y, f, X } = descTable('di', IDX5);
  const hk = IDX5.reduce((b, a) => (X[a].exkurt > X[b].exkurt ? a : b)), neg = IDX5.filter(a => X[a].skew < 0).length;
  const P = perfAll(y);
  out('di-out', [T.di(T.fq[f], fdate(asset('sp500', y).dates[0]), nf(X.btc.sd / mean(['sp500', 'dax', 'bet'].map(a => X[a].sd)), 1), neg, LAB[hk], nf(X[hk].exkurt, 1)),
    T.di2(nf(X.sp500.annVol, 1), nf(100 * P.sp500.vol, 1))]);
}
function descBvbBlock() {
  const { y, f, X, cl } = descTable('db', BVB5);
  const hk = BVB5.reduce((b, a) => (X[a].exkurt > X[b].exkurt ? a : b)), lk = BVB5.reduce((b, a) => (X[a].exkurt < X[b].exkurt ? a : b));
  const B = describe(resample(asset('bet', y), f)), t0 = describe(asset('tlv', y)), t1 = describe(asset('tlv', y), TLV_BAD);
  out('db-out', [T.db(T.fq[f], fdate(asset('tlv', y).dates[0]), nf(Math.min(...BVB5.map(a => X[a].sd)), 2), nf(Math.max(...BVB5.map(a => X[a].sd)), 2), nf(B.sd, 2)),
    T.db2(LAB[hk], nf(X[hk].exkurt, 1), LAB[lk], nf(X[lk].exkurt, 1)),
    y <= 2016 ? T.db3(nf(t0.exkurt, 1), nf(t1.exkurt, 1), nf(t1.max, 1)) : ''].filter(Boolean));
}

// ================================================================= X6–X7. performance indicators (tables)
function perfTable(pf, ks) {
  const y = +$(pf + '-y').value, rf = +$(pf + '-rf').value / 100, k = $(pf + '-a').value; label(pf + '-y').textContent = y; label(pf + '-rf').textContent = nf(100 * rf, 2) + '%';
  const P = perfAll(y, rf);
  table(pf + '-tab', ['', 'CAGR %', T.h.vol, 'Sharpe (SE)', 'Sortino', 'MDD %', 'Calmar', 'Drag %'], ks.map(a => {
    const x = P[a];
    return Object.assign([LAB[a] + (pf === 'pb' && SH[a] !== LAB[a] ? ` (${SH[a]})` : ''), nf(100 * x.cagr, 1), nf(100 * x.vol, 1), `${nf(x.sharpe, 2)} <span class="pv">(${nf(x.se, 2)})</span>`, nf(x.sortino, 2),
      nf(100 * x.mdd, 1), nf(x.calmar, 2), nf(100 * x.drag, 2)], { cls: SEL(a, k) });
  }));
  return { y, rf, k, P };
}
function perfIdxBlock() {
  const { y, rf, k, P } = perfTable('pi', IDX5), x = P[k], lo = x.sharpe - 1.96 * x.se, hi = x.sharpe + 1.96 * x.se;
  steps('pi-we', [T.piw1(LAB[k], fdate(asset(k, y).dates[0]), fint(x.n), nf(x.q, 1), nf(100 * x.m, 4), nf(100 * x.s, 3)),
    T.piw2(nf(100 * x.m, 4), nf(100 * rf, 2), nf(x.q, 1), nf(100 * x.s, 3), nf(x.srd, 4), nf(Math.sqrt(x.q), 2), nf(x.sharpe, 3)),
    T.piw3(nf(x.srd, 4), fint(x.n), nf(x.sed, 4), nf(Math.sqrt(x.q), 2), nf(x.se, 3)),
    T.piw4(nf(x.sharpe, 2), nf(x.se, 3), nf(lo, 2), nf(hi, 2), lo > 0)]);
  const best = IDX5.reduce((b, a) => (P[a].sharpe > P[b].sharpe ? a : b));
  out('pi-out', [T.pi(fdate(asset('sp500', y).dates[0]), nf(100 * rf, 2), LAB[best], nf(P[best].sharpe, 2), nf(P[best].se, 2)),
    T.pi2(nf(100 * P.sp500.vol, 1), nf(describe(asset('sp500', y)).annVol, 1), nf(100 * P.bet.mdd, 3), nf(100 * P.bettr.mdd, 3))]);
}
function perfBvbBlock() {
  const { y, k, P } = perfTable('pb', BVB5);
  // drawdown duration: the maximum-drawdown episode of every asset, peak to trough and trough to recovery (calendar days)
  const ks = D.assets.slice().sort((a, b) => P[b].mdd - P[a].mdd), dur = ks.map(a => {
    const x = P[a], d = x.dates, end = d[d.length - 1];
    return { a, fall: ndays(d[x.peak], d[x.trough]), rec: ndays(d[x.trough], x.rec >= 0 ? d[x.rec] : end), done: x.rec >= 0 };
  });
  $('pb-chart').style.height = H(420) + 'px';
  plot('pb-chart', [
    { y: dur.map(o => LAB[o.a]), x: dur.map(o => o.fall), type: 'bar', orientation: 'h', name: 'peak to trough', marker: { color: COL.bet }, hovertemplate: '%{y}: %{x} days from peak to trough<extra></extra>' },
    { y: dur.map(o => LAB[o.a]), x: dur.map(o => o.rec), type: 'bar', orientation: 'h', name: `trough to recovery (hatched: not recovered by ${D.end})`,
      marker: { color: COL.tlv, pattern: { shape: dur.map(o => (o.done ? '' : '/')), fgcolor: C.navy } }, hovertemplate: '%{y}: %{x} days after the trough<extra></extra>' }
  ], layout({ barmode: 'stack', legend: Object.assign(layout().legend, { traceorder: 'normal' }), xaxis: XT('Calendar days of the maximum-drawdown episode'), yaxis: XT('', { automargin: true }), margin: { l: 130, r: 12, t: 16, b: 40 } }));
  const x = P[k], d = x.dates;
  steps('pb-we', [T.pbw1(LAB[k], fprice(x.p[0]), fdate(d[0]), fprice(x.p[x.p.length - 1]), fdate(d[d.length - 1]), nf(x.years, 2), nf(100 * x.cagr, 2)),
    T.pbw2(fdate(d[x.peak]), fprice(x.p[x.peak]), fdate(d[x.trough]), fprice(x.p[x.trough]), nf(100 * x.mdd, 2)),
    T.pbw3(nf(100 * x.cagr, 2), nf(-100 * x.mdd, 2), nf(x.calmar, 2)),
    x.rec >= 0 ? T.pbw4(fdate(d[x.rec]), nint(ndays(d[x.peak], d[x.rec]))) : T.pbw4n(fdate(d[d.length - 1]))]);
  const lg = dur.reduce((b, o) => (o.fall + o.rec > b.fall + b.rec ? o : b));
  out('pb-out', [T.pb(fdate(asset('tlv', y).dates[0]), LAB[BVB5.reduce((b, a) => (P[a].calmar > P[b].calmar ? a : b))], LAB[BVB5.reduce((b, a) => (P[a].mdd < P[b].mdd ? a : b))]),
    T.pb2(LAB[lg.a], nint(lg.fall + lg.rec), lg.done, dur.filter(o => !o.done).map(o => LAB[o.a]).join(', ') || '—')]);
}

// ================================================================= wiring
function placeExtras() {                // web lecture: the table blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['tl-th'], tlvAdjusted); PL('tl-play', 'tl-th', [0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50], 800);
  on(['bb-y'], betBettr); PL('bb-play', 'bb-y', range(2014, 2025, 1), 800);
  on(['eu-th'], eurron); PL('eu-play', 'eu-th', range(0.25, 5, 0.25), 500);
  on(['pr-k'], priceReturns); PL('pr-play', 'pr-k', range(0, 6, 1), 1000);
  on(['sl-R', 'sl-d'], simpleLogReturns); PL('sl-play', 'sl-R', range(-60, 60, 2.5), 200);
  on(['gr-y', 'gr-sc', 'gr-a'], growth); PL('gr-play', 'gr-y', range(2015, 2025, 1), 800);
  on(['hi-k'], histNormal); PL('hi-play', 'hi-k', range(1, 6, 0.5), 700);
  on(['rv-w'], rollingVol); PL('rv-play', 'rv-w', [21, 42, 63, 125, 250, 500], 1000);
  on(['sc-y', 'sc-cl'], sharpeCI); PL('sc-play', 'sc-y', range(2015, 2024, 1), 800);
  on(['dd-x', 'dd-a'], drawdowns); PL('dd-play', 'dd-x', range(5, 80, 5), 500);
  on(['vd-s', 'vd-a'], volDrag); PL('vd-play', 'vd-s', range(0, 80, 5), 400);
  on(['rr-y', 'rr-m'], riskReturn); PL('rr-play', 'rr-y', range(2015, 2024, 1), 900);
  on(['pf-w'], portfolioBlock); PL('pf-play', 'pf-w', range(10, 90, 5), 500);
  on(['an-y', 'an-f', 'an-a'], annualiseBlock); PL('an-play', 'an-f', ['d', 'w', 'm'], 1600);
  on(['ms-y', 'ms-cl', 'ms-a'], meanSeBlock); PL('ms-play', 'ms-y', range(2015, 2024, 1), 800);
  on(['di-y', 'di-f'], descIdxBlock); PL('di-play', 'di-f', ['d', 'w', 'm'], 1600);
  on(['db-y', 'db-f', 'db-t'], descBvbBlock); PL('db-play', 'db-t', ['raw', 'clean'], 1800);
  on(['pi-y', 'pi-rf', 'pi-a'], perfIdxBlock); PL('pi-play', 'pi-rf', range(0, 5, 0.25), 400);
  on(['pb-y', 'pb-rf', 'pb-a'], perfBvbBlock); PL('pb-play', 'pb-y', range(2015, 2024, 1), 900);
  const PFX = { 'tlv-adjusted': 'tl', 'bet-bettr': 'bb', 'eurron': 'eu', 'price-returns': 'pr', 'simple-log': 'sl', 'growth': 'gr', 'hist-normal': 'hi',
    'rolling-vol': 'rv', 'sharpe-ci': 'sc', 'drawdowns': 'dd', 'vol-drag': 'vd', 'risk-return': 'rr',
    portfolio: 'pf', annualise: 'an', 'mean-se': 'ms', 'desc-indices': 'di', 'desc-bvb': 'db', 'perf-indices': 'pi', 'perf-bvb': 'pb' };
  const VEIL = T.veil || {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [tlvAdjusted, betBettr, eurron, priceReturns, simpleLogReturns, growth, portfolioBlock, annualiseBlock, meanSeBlock, histNormal, rollingVol, descIdxBlock, descBvbBlock,
    sharpeCI, drawdowns, volDrag, perfIdxBlock, perfBvbBlock, riskReturn];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH1 = { raw, asset, full, slice, performance, perfAll, describe, resample, mddDates, portfolio, pairTS, drawdown, rollSd, logRet, simpleRet, quantile, mean, sd, perYear, normSf, init };
if (window.CH1_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
