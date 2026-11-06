// Chapter 0 interactive page: one block per chosen chart slide, computed in the browser from ch0_data.js.
(function () {
'use strict';
const D = window.CH0_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH0_TEXT;
const LAB = D.labels, COL = D.colors, PAL = D.palette;
const BLUE = PAL.blue, RED = PAL.red, DARK = PAL.dark;

// ================================================================= data: closes on their own calendars (load_close)
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
const dayOf = s => Date.parse(s + 'T00:00:00Z') / 864e5;
const CACHE = {};
function raw(key) {                 // {days, dates, p} of one exported series
  if (CACHE[key]) return CACHE[key];
  const R = D.s[key], days = [R.d0];
  for (const dd of R.dd) days.push(days[days.length - 1] + dd);
  return (CACHE[key] = { days, dates: days.map(iso), p: R.v });
}
function cut(S, d0, d1) {           // rows of S with d0 <= day <= d1
  let i = 0; while (i < S.days.length && S.days[i] < d0) i++;
  let j = S.days.length; if (d1 !== undefined) while (j > i && S.days[j - 1] > d1) j--;
  return { days: S.days.slice(i, j), dates: S.dates.slice(i, j), p: S.p.slice(i, j) };
}
function close(k, start) {          // load_close(k, start): the 2015 export, or the 2000 export cut at start
  const key = k + start;
  if (CACHE[key]) return CACHE[key];
  const S = D.s[k] ? raw(k) : raw(k + '_all');
  return (CACHE[key] = cut(S, dayOf(start)));
}

// ================================================================= statistics (the formulas of generate_all_charts.py)
const mean = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
const sd = a => { const m = mean(a); let s = 0; for (const x of a) s += (x - m) ** 2; return Math.sqrt(s / (a.length - 1)); };
const logRet = p => p.slice(1).map((x, i) => 100 * Math.log(x / p[i]));          // log_returns, in %
function erfc(x) {                    // series below 2.5, continued fraction above
  if (x < 0) return 2 - erfc(-x);
  if (x < 2.5) { let term = x, s = x; for (let n = 1; n < 200; n++) { term *= 2 * x * x / (2 * n + 1); s += term; if (term < 1e-17 * s) break; } return 1 - 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * s; }
  let f = x, c = x, d = 0;
  for (let n = 1; n < 300; n++) { const an = n / 2; d = x + an * d; d = 1 / d; c = x + an / c; const del = c * d; f *= del; if (Math.abs(del - 1) < 1e-16) break; }
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
}
const normSf = z => 0.5 * erfc(z / Math.SQRT2);
function marketRow(k, start) {        // market_table: annualised mean log return and volatility (%), multiple P_T / P_0
  const P = close(k, start), r = logRet(P.p), years = (P.days[P.days.length - 1] - P.days[1]) / 365.25, ppy = r.length / years;
  return { k, first: P.dates[1], n: r.length, ppy, ret: mean(r) * ppy, vol: sd(r) * Math.sqrt(ppy), mult: P.p[P.p.length - 1] / P.p[0], years };
}
function drawdown(p) {                // DD_t = P_t / max_{s<=t} P_s - 1 (in %)
  let mx = -Infinity; return p.map(v => { mx = Math.max(mx, v); return 100 * (v / mx - 1); });
}
function ddStats(P, dd) {             // maximum drawdown, its peak, its trough and the first day back at the peak (or null)
  let it = 0; dd.forEach((v, i) => { if (v < dd[it]) it = i; });
  let ip = 0; for (let i = 0; i <= it; i++) if (P.p[i] > P.p[ip]) ip = i;
  let ir = null; for (let i = it; i < P.p.length; i++) if (P.p[i] >= P.p[ip]) { ir = i; break; }
  return { mdd: dd[it], trough: P.dates[it], peak: P.dates[ip], rec: ir === null ? null : P.dates[ir], last: dd[dd.length - 1], lastDate: P.dates[P.dates.length - 1] };
}

// ================================================================= chart helpers
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 120 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const M0 = { l: 56, r: 12, t: 14, b: 40 };
const label = id => $(id + '-v');
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const sgn = (x, d) => (x > 0 ? '+' : '') + nf(x, d);
const DT = s => T.date(s);
const LOGT = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
const YEAR_END = y => y >= 2026 ? D.end : `${y}-12-31`;

// ================================================================= 1. five markets since 2015: growth of 100 invested
function mkBlock() {
  const y = +$('mk-y').value, start = `${y}-01-01`, n = single('mk-chart', 400);
  label('mk-y').textContent = y;
  // fig_markets: the five closes on the union of their dates, forward-filled, from the first date where all exist
  const S = D.markets.map(k => close(k, start)), all = [...new Set(S.flatMap(s => s.days))].sort((a, b) => a - b);
  const pos = S.map(() => 0), last = S.map(() => null), rows = [];
  all.forEach(d => {
    const v = S.map((s, i) => { while (pos[i] < s.days.length && s.days[pos[i]] <= d) last[i] = s.p[pos[i]++]; return last[i]; });
    if (v.every(x => x !== null)) rows.push([d, v]);
  });
  const x = rows.map(r => iso(r[0])), base = rows[0][1];
  const tr = D.markets.map((k, i) => ({ x, y: rows.map(r => 100 * r[1][i] / base[i]), mode: 'lines', line: { color: COL[k], width: 1.3 }, name: LAB[k],
    hovertemplate: `${LAB[k]} %{x}: %{y:.1f}<extra></extra>` }));
  plot('mk-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Value of 100 invested (log scale)', { type: 'log', tickvals: LOGT, ticktext: LOGT.map(String) }), legend: LEG(n, -0.1), margin: M0 }));
  const R = D.markets.map(k => marketRow(k, start)), best = R.reduce((a, b) => (b.ret > a.ret ? b : a)), worst = R.reduce((a, b) => (b.ret < a.ret ? b : a));
  const bt = R.find(r => r.k === 'bettr'), sp = R.find(r => r.k === 'sp500');
  out('mk-out', [T.mk(y, R.map(r => [T.name(r.k), nf(r.mult, 1), nf(r.ret, 1), nf(r.vol, 1)]), y !== 2015),
    T.mk2(T.name(best.k), T.name(worst.k), sgn(bt.ret - sp.ret, 1), sgn(bt.vol - sp.vol, 1), nf(R.find(r => r.k === 'btc').vol / sp.vol, 1))]);
}

// ================================================================= 2. risk and return, eleven markets
function rrBlock() {
  const y = +$('rr-y').value, start = `${y}-01-01`, n = single('rr-chart', 420);
  if (n) $('rr-chart').style.height = '700px';           // eleven legend rows below a readable scatter
  label('rr-y').textContent = y;
  const R = D.rr.map(k => marketRow(k, start));
  const tr = R.map(r => ({ x: [r.vol], y: [r.ret], mode: 'markers', type: 'scatter', name: LAB[r.k],
    marker: { color: D.colors_rr[r.k], size: 11, symbol: D.markers_rr[r.k] || 'circle' }, cliponaxis: false, hovertemplate: `${LAB[r.k]}: volatility %{x:.1f}%, return %{y:.1f}%<extra></extra>` }));
  const ys = R.map(r => r.ret), lo = Math.min(0, ...ys), hi = Math.max(...ys);
  tr.push({ x: [8, 160], y: [0, 0], mode: 'lines', line: { color: BLUE, width: 0.8, dash: 'dash' }, hoverinfo: 'skip', showlegend: false });
  plot('rr-chart', tr, layout({ xaxis: AX('Annualised volatility, % (log scale)', { type: 'log', tickvals: [10, 15, 20, 30, 50, 70, 100, 150], range: [Math.log10(9), Math.log10(160)] }),
    yaxis: AX('Annualised mean log return, %', { range: [lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)] }), legend: LEG(n, -0.16), margin: M0,
    annotations: R.map(r => { const o = D.offsets_rr[r.k] || [9, 0, 'left'];
      return { x: Math.log10(r.vol), y: r.ret, text: LAB[r.k].split(' (')[0], showarrow: false, xshift: o[0] * 4 / 3, yshift: o[1] * 4 / 3, xanchor: o[2] === 'center' ? 'center' : o[2], yanchor: 'middle', font: { size: 11, color: DARK } }; }) }));
  const EQ = ['sp500', 'ndx', 'dax', 'stoxx50', 'nikkei', 'bettr', 'wig20'], E = R.filter(r => EQ.includes(r.k)), g = k => R.find(r => r.k === k);
  const eLo = E.reduce((a, b) => (b.ret < a.ret ? b : a)), eHi = E.reduce((a, b) => (b.ret > a.ret ? b : a));
  const se = r => r.vol / Math.sqrt(r.years);
  out('rr-out', [T.rr(y, nf(Math.min(...E.map(r => r.vol)), 1), nf(Math.max(...E.map(r => r.vol)), 1), T.name(eLo.k), nf(eLo.ret, 1), T.name(eHi.k), nf(eHi.ret, 1),
    nf(g('btc').vol, 1), nf(g('eth').vol, 1), nf(g('tlt').ret, 1), y !== 2015),
    T.rr2(nf(g('sp500').years, 1), nf(se(g('sp500')), 1), nf(1.96 * se(g('sp500')), 1), nf(se(g('btc')), 1))]);
}

// ================================================================= 3. drawdowns of the S&P 500 and the BET since 2000
function ddBlock() {
  const y = +$('dd-y').value, n = single('dd-chart', 380), end = dayOf(YEAR_END(y));
  label('dd-y').textContent = y >= 2026 ? T.endLabel : y;
  const tr = [], rows = [];
  ['sp500', 'bet'].forEach(k => {
    const P = cut(close(k, D.start_long), -Infinity, end), dd = drawdown(P.p), s = ddStats(P, dd);
    tr.push({ x: P.dates, y: dd, mode: 'lines', line: { color: COL[k], width: 1.1 }, name: LAB[k], hovertemplate: `${LAB[k]} %{x}: %{y:.1f}%<extra></extra>` });
    rows.push([T.name(k), nf(s.mdd, 1), DT(s.trough), DT(s.peak), s.rec ? DT(s.rec) : null, nf(100 / (1 + s.mdd / 100) - 100, 0), nf(s.last, 1), DT(s.lastDate), nf(100 / (1 + s.last / 100) - 100, 1)]);
  });
  plot('dd-chart', tr, layout({ xaxis: AX('', { range: ['2000-01-01', D.end] }), yaxis: AX('Drawdown, %', { range: [-88, 3] }), legend: LEG(n, -0.1), margin: M0 }));
  out('dd-out', rows.map(r => T.dd(...r)));
}

// ================================================================= 4. VIX since 2000
function vxBlock() {
  const thr = +$('vx-t').value, n = single('vx-chart', 380), P = close('vix', D.start_long), v = P.p, N = v.length;
  label('vx-t').textContent = thr;
  const m = mean(v), above = v.map(x => x > thr), k = above.filter(Boolean).length;
  let runs = 0, longest = 0, cur = 0, stay = 0, from = 0, i0 = 0, best = 0;
  above.forEach((a, i) => {
    if (a) { cur++; if (cur === 1) { runs++; i0 = i; } if (cur > longest) { longest = cur; best = i0; } } else cur = 0;
    if (i < N - 1 && a) { from++; if (above[i + 1]) stay++; }
  });
  const sorted = [...v].sort((a, b) => a - b), med = (sorted[(N - 1) >> 1] + sorted[N >> 1]) / 2;
  const iy = s => P.dates.findIndex(d => d >= s);
  const t1 = (() => { let b = 0; for (let i = 0; i < iy('2013-01-01'); i++) if (v[i] > v[b]) b = i; return b; })();
  const t2 = (() => { let b = iy('2013-01-01'); for (let i = b; i < N; i++) if (v[i] > v[b]) b = i; return b; })();
  const tr = [{ x: P.dates, y: v, mode: 'lines', line: { color: COL.vix, width: 0.9 }, name: 'VIX (close)', hovertemplate: 'VIX %{x}: %{y:.2f}<extra></extra>' },
    { x: [P.dates[0], P.dates[N - 1]], y: [m, m], mode: 'lines', line: { color: BLUE, width: 0.8, dash: 'dash' }, name: `Mean since 2000: ${m.toFixed(1)}`, hoverinfo: 'skip' },
    { x: [P.dates[0], P.dates[N - 1]], y: [thr, thr], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dot' }, name: `Threshold: ${thr}`, hoverinfo: 'skip' },
    { x: P.dates.filter((_, i) => above[i]), y: v.filter((_, i) => above[i]), mode: 'markers', marker: { color: '#B5853F', size: 3 }, name: `Days above ${thr}`, hoverinfo: 'skip' }];
  const fmt = d => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' });
  const ann = [t1, t2].map(i => ({ x: P.dates[i], y: v[i], text: `${fmt(P.dates[i])}: ${v[i].toFixed(1)}`, showarrow: false, xanchor: 'left', xshift: 8, yshift: -4, font: { size: 11, color: '#000' } }));
  plot('vx-chart', tr, layout({ xaxis: AX(''), yaxis: AX('VIX, % a year', { range: [0, 92] }), legend: LEG(n, -0.1), margin: M0, annotations: ann }));
  out('vx-out', [T.vx(nf(m, 1), nf(med, 1), DT(P.dates[t1]), nf(v[t1], 1), DT(P.dates[t2]), nf(v[t2], 1), nf(v[N - 1], 1)),
    T.vx2(thr, fint(k), fint(N), nf(100 * k / N, k / N < 0.01 ? 2 : 1), runs, longest, longest ? DT(P.dates[best]) : '', from ? nf(100 * stay / from, 1) : null)]);
}

// ================================================================= 5. daily returns of the S&P 500 and their rolling volatility
function rtBlock() {
  const w = +$('rt-w').value, n = single('rt-chart', 380), P = close('sp500', D.start_long), r = logRet(P.p), x = P.dates.slice(1), N = r.length;
  const m = mean(r), s = sd(r), ppy = N / ((P.days[N] - P.days[1]) / 365.25);
  // rolling standard deviation over the w previous days (t - w ... t - 1): the band of day t uses only earlier returns
  const band = new Array(N).fill(null);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < N; i++) {
    if (i >= w) { const mm = s1 / w; band[i] = Math.sqrt(Math.max(0, (s2 - w * mm * mm) / (w - 1))); }
    s1 += r[i]; s2 += r[i] * r[i];
    if (i >= w) { s1 -= r[i - w]; s2 -= r[i - w] * r[i - w]; }
  }
  let lo = -1, hi = -1, outside = 0, cnt = 0;
  band.forEach((b, i) => { if (b === null) return; cnt++; if (Math.abs(r[i]) > 2 * b) outside++; if (lo < 0 || b < band[lo]) lo = i; if (hi < 0 || b > band[hi]) hi = i; });
  const tr = [{ x, y: r, mode: 'lines', line: { color: BLUE, width: 0.5 }, name: 'S&P 500, daily log return', hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x, y: band.map(b => b === null ? null : 2 * b), mode: 'lines', line: { color: RED, width: 1 }, name: `±2 × rolling standard deviation (${w} previous days)`, legendgroup: 'b', hoverinfo: 'skip' },
    { x, y: band.map(b => b === null ? null : -2 * b), mode: 'lines', line: { color: RED, width: 1 }, name: 'band', legendgroup: 'b', showlegend: false, hoverinfo: 'skip' }];
  plot('rt-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Daily log return, %', { range: [-14, 14] }), legend: LEG(n, -0.1), margin: M0 }));
  out('rt-out', [T.rt(fint(N), DT(P.dates[1]), nf(m, 3), nf(s, 2), nf(s * Math.sqrt(ppy), 1), nf(s / m, 0), nf(m / (s / Math.sqrt(N)), 2)),
    T.rt2(w, nf(band[lo] * Math.sqrt(ppy), 1), DT(x[lo]), nf(band[hi] * Math.sqrt(ppy), 1), DT(x[hi]), fint(outside), fint(cnt), nf(100 * outside / cnt, 1), nf(100 * 2 * normSf(2), 1))]);
}

// ================================================================= 6. histogram against the Normal density
function hsBlock() {
  const kk = +$('hs-k').value, n = single('hs-chart', 400), P = close('sp500', D.start_long), r = logRet(P.p), N = r.length;
  const kf = Number.isInteger(kk) ? String(kk) : nf(kk, 1);
  label('hs-k').textContent = kf;
  const m = mean(r), s = sd(r), h = 0.2, nb = 130, cnt = new Array(nb).fill(0);
  r.forEach(x => { const j = Math.floor((x + 13) / h + 1e-9); if (j >= 0 && j < nb) cnt[j]++; });
  const mids = cnt.map((_, j) => -13 + h * (j + 0.5)), xs = Array.from({ length: 600 }, (_, i) => -13 + 26 * i / 599);
  const pdf = xs.map(x => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI)));
  const z = r.map(x => (x - m) / s), lossK = z.filter(t => t < -kk).length, gainK = z.filter(t => t > kk).length, obs = lossK + gainK, exp = N * 2 * normSf(kk);
  let m4 = 0; r.forEach(x => { m4 += (x - m) ** 4; }); const K = m4 / N / (s * Math.sqrt((N - 1) / N)) ** 4, zmin = Math.min(...z);
  const lo = m - kk * s, hi = m + kk * s;
  const tr = [{ type: 'bar', x: mids, y: cnt.map(c => c ? c / (N * h) : null), width: h, marker: { color: BLUE, opacity: 0.75 }, name: 'S&P 500, daily log returns',
    hovertemplate: '%{x:.1f}%: density %{y:.4f}<extra></extra>' },
    { x: xs, y: pdf, mode: 'lines', line: { color: RED, width: 1.8 }, name: 'Normal distribution, same mean and variance', hoverinfo: 'skip' },
    { x: [lo, lo, null, hi, hi], y: [5e-5, 1, null, 5e-5, 1], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `Mean ± ${kk} standard deviations`, hoverinfo: 'skip' }];
  plot('hs-chart', tr, layout({ xaxis: AX('Daily log return, %', { range: [-13, 13] }), yaxis: AX('Density (log scale)', { type: 'log', range: [Math.log10(5e-5), 0], tickvals: [1e-4, 1e-3, 0.01, 0.1, 1], ticktext: ['0.0001', '0.001', '0.01', '0.1', '1'] }),
    legend: LEG(n, -0.2), margin: M0, bargap: 0 }));
  out('hs-out', [T.hs(kf, obs, lossK, gainK, nf(exp, exp >= 10 ? 0 : exp >= 0.01 ? 2 : Math.ceil(-Math.log10(exp)) + 1), obs / exp >= 10 ? fint(obs / exp) : nf(obs / exp, 1), fint(N)),
    T.hs2(nf(K, 1), nf(zmin, 1), DT(P.dates[1 + z.indexOf(zmin)]))]);
}

// ================================================================= wiring
const PFX = { markets: 'mk', 'risk-return': 'rr', drawdowns: 'dd', vix: 'vx', returns: 'rt', histogram: 'hs' };
const BLOCKS = { mkBlock, rrBlock, ddBlock, vxBlock, rtBlock, hsBlock };
const PLAYERS = {};
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const PLy = (btn, id, vals, ms) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms));
  const yrs = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
  on(['mk-y'], mkBlock); PLy('mk-play', 'mk-y', [...yrs(2015, 2025), 2015], 1100);
  on(['rr-y'], rrBlock); PLy('rr-play', 'rr-y', [...yrs(2015, 2024), 2015], 1100);
  on(['dd-y'], ddBlock); PLy('dd-play', 'dd-y', [2026, ...yrs(2001, 2026)], 600);
  on(['vx-t'], vxBlock); PLy('vx-play', 'vx-t', [30, 20, 25, 35, 40, 45, 50, 60, 70, 80, 30], 1000);
  on(['rt-w'], rtBlock); PLy('rt-play', 'rt-w', ['21', '63', '126', '252', '21'], 1500);
  on(['hs-k'], hsBlock); PLy('hs-play', 'hs-k', [4, 2, 2.5, 3, 3.5, 4.5, 5, 5.5, 6, 4], 1100);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH0 = { close, marketRow, drawdown, ddStats, logRet, mean, sd, normSf, init, PFX, blocks: BLOCKS };
if (window.CH0_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
