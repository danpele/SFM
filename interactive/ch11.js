// Chapter 11 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch11_data.js.
(function () {
'use strict';
const D = window.CH11_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH11_TEXT;
const NUM = D.N, COL = D.col, NAME = D.name, MK = D.names;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';
const SH = { sp500: 'S&P 500', dax: 'DAX', bet: 'BET', btc: 'Bitcoin', tlv: 'TLV', snp: 'SNP' };
const LO = D.lo_crit[0], HI = D.lo_crit[1];

// ================================================================= numerics
function gammln(z) {                // Lanczos
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
  for (const c of g) ser += c / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function normppf(p) {               // Acklam
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - 0.02425) return -normppf(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
const mean = x => { let s = 0; for (const v of x) s += v; return s / x.length; };
const sd0 = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / x.length); };
function npq(sorted, p) { const h = (sorted.length - 1) * p, i = Math.floor(h); return i + 1 < sorted.length ? sorted[i] + (h - i) * (sorted[i + 1] - sorted[i]) : sorted[i]; }
function summ(v) { const s = Float64Array.from(v).sort(); return { mean: mean(v), sd: sd0(v), q025: npq(s, 0.025), q975: npq(s, 0.975), q: p => npq(s, p) }; }
function ols(x, y) {                // OLS slope and intercept (np.polyfit degree 1) with the sums of the worked examples
  const n = x.length, mx = mean(x), my = mean(y); let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); }
  const b = sxy / sxx; return { b, a: my - b * mx, mx, my, sxx, sxy, n };
}
const L10 = a => a.map(v => Math.log10(v));

// ================================================================= estimators of the chapter (generate_fmh_charts.py)
function blockSizes(N, nmin, nmax, num) {
  if (!nmin && D.sizes[N]) return D.sizes[N];
  nmin = nmin || 10; nmax = nmax || Math.floor(N / 4); num = num || 20;
  const a = Math.log10(nmin), b = Math.log10(nmax), s = new Set();
  for (let i = 0; i < num; i++) s.add(Math.floor(10 ** (i === num - 1 ? b : a + (b - a) * i / (num - 1))));
  return [...s].sort((u, v) => u - v);
}
function rsBlock(x, i0, n) {        // R/S of one block: range of the cumulative deviations divided by the standard deviation (ddof 0)
  let m = 0; for (let j = 0; j < n; j++) m += x[i0 + j]; m /= n;
  let y = 0, mx = -Infinity, mn = Infinity, ss = 0;
  for (let j = 0; j < n; j++) { const d = x[i0 + j] - m; y += d; ss += d * d; if (y > mx) mx = y; if (y < mn) mn = y; }
  return { R: mx - mn, S: Math.sqrt(ss / n) };
}
function rsCurve(x, sizes) {
  return sizes.map(n => { const m = Math.floor(x.length / n); let s = 0, c = 0; for (let b = 0; b < m; b++) { const o = rsBlock(x, b * n, n); if (o.S > 0) { s += o.R / o.S; c++; } } return s / c; });
}
const hurstRS = (x, sizes) => ols(L10(sizes), L10(rsCurve(x, sizes))).b;
function profile(x) { const m = mean(x), Y = new Float64Array(x.length); let s = 0; for (let i = 0; i < x.length; i++) { s += x[i] - m; Y[i] = s; } return Y; }
function dfaBlock(Y, i0, n) {       // OLS line in one block of the profile: mean, slope on the centred time, sum of squared residuals
  const tc = (n - 1) / 2; let m = 0; for (let j = 0; j < n; j++) m += Y[i0 + j]; m /= n;
  let st = 0, tt = 0; for (let j = 0; j < n; j++) { const t = j - tc; st += (Y[i0 + j] - m) * t; tt += t * t; }
  const b = st / tt; let rr = 0; for (let j = 0; j < n; j++) { const e = Y[i0 + j] - m - b * (j - tc); rr += e * e; }
  return { m, b, rr, tc };
}
function dfaCurve(x, sizes, Y) {
  Y = Y || profile(x);
  return sizes.map(n => { const m = Math.floor(Y.length / n); let rr = 0; for (let b = 0; b < m; b++) rr += dfaBlock(Y, b * n, n).rr; return Math.sqrt(rr / (m * n)); });
}
const hurstDFA = (x, sizes) => ols(L10(sizes), L10(dfaCurve(x, sizes))).b;
const PG = new Map();
function periodogram(x, mmax, key) {  // I(lambda_j) = |sum_t (x_t - mean) exp(-i lambda_j t)|^2 / (2 pi T), j = 1..mmax
  if (key && PG.has(key) && PG.get(key).I.length >= mmax) return PG.get(key);
  const Tn = x.length, m0 = mean(x), I = new Float64Array(mmax), lam = new Float64Array(mmax);
  for (let j = 1; j <= mmax; j++) {
    const w = 2 * Math.PI * j / Tn, c = Math.cos(w), s = -Math.sin(w); let re = 0, im = 0, pr = 1, pi = 0;
    for (let t = 0; t < Tn; t++) {
      const v = x[t] - m0; re += v * pr; im += v * pi;
      const nr = pr * c - pi * s; pi = pr * s + pi * c; pr = nr;
      if ((t & 1023) === 1023) { const a = 2 * Math.PI * j * (t + 1) / Tn; pr = Math.cos(a); pi = -Math.sin(a); }
    }
    lam[j - 1] = w; I[j - 1] = (re * re + im * im) / (2 * Math.PI * Tn);
  }
  const P = { lam, I }; if (key) PG.set(key, P); return P;
}
function gphOf(P, Tn, m) {           // GPH: OLS slope of log I on -log(4 sin^2(lambda/2)), j = 1..m; SE pi / sqrt(24 m)
  const X = [], y = []; for (let j = 0; j < m; j++) { X.push(-Math.log(4 * Math.sin(P.lam[j] / 2) ** 2)); y.push(Math.log(P.I[j])); }
  const o = ols(X, y); return Object.assign(o, { d: o.b, se: Math.PI / Math.sqrt(24 * m), m, X, y, H: o.b + 0.5 });
}
const mOf = (Tn, p) => Math.floor(Tn ** p);
function gph(x, p, key) { const m = mOf(x.length, p); return gphOf(periodogram(x, m, key), x.length, m); }
function autocov(x, L) { const n = x.length, m = mean(x), xc = x.map(v => v - m), g = []; for (let j = 0; j <= L; j++) { let s = 0; for (let t = j; t < n; t++) s += xc[t] * xc[t - j]; g.push(s / n); } return { g, xc }; }
function loTest(x, qf) {             // Lo (1991): V(q) = R / sqrt(N sigma^2(q)), Bartlett weights; Andrews q from |rho_1|
  const N = x.length, { g, xc } = autocov(x, 40);
  let y = 0, mx = -Infinity, mn = Infinity; for (const v of xc) { y += v; if (y > mx) mx = y; if (y < mn) mn = y; }
  const R = mx - mn, rho = g[1] / g[0], A = (1.5 * N) ** (1 / 3), B = (2 * Math.abs(rho) / (1 - rho * rho)) ** (2 / 3), qa = Math.floor(A * B);
  const q = qf === undefined || qf === null ? qa : qf;
  const s2 = qq => { let s = g[0]; for (let j = 1; j <= qq; j++) s += 2 * (1 - j / (qq + 1)) * g[j]; return s; };
  const V = R / Math.sqrt(N * s2(q)), V0 = R / Math.sqrt(N * g[0]);
  return { N, R, g, rho, A, B, q, qa, s2: s2(q), V, V0, Vq: qq => R / Math.sqrt(N * s2(qq)), reject: !(LO <= V && V <= HI), reject0: !(LO <= V0 && V0 <= HI) };
}
function acf(x, L) { const m = mean(x), xc = x.map(v => v - m); let g0 = 0; for (const v of xc) g0 += v * v; const a = []; for (let k = 1; k <= L; k++) { let s = 0; for (let t = k; t < xc.length; t++) s += xc[t] * xc[t - k]; a.push(s / g0); } return a; }
function scaling(x, hs) { const sd = hs.map(h => { const m = Math.floor(x.length / h), s = []; for (let b = 0; b < m; b++) { let c = 0; for (let j = 0; j < h; j++) c += x[b * h + j]; s.push(c); } return sd0(s); }); return { sd, H: ols(L10(hs), L10(sd)).b }; }
const fgnAcvf = (k, H) => 0.5 * (Math.abs(k + 1) ** (2 * H) - 2 * Math.abs(k) ** (2 * H) + Math.abs(k - 1) ** (2 * H));
function arfAcf(n, d) { const r = [1]; for (let k = 1; k < n; k++) r.push(r[k - 1] * (k - 1 + d) / (k - d)); return r; }
const arfVar = d => Math.exp(gammln(1 - 2 * d) - 2 * gammln(1 - d));

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpack(R) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); return dd.map(iso); }
const SER = {};
function ser(k) {
  if (SER[k]) return SER[k];
  const x = D.ret[k].x;
  return (SER[k] = { x, abs: x.map(Math.abs), n: x.length, dates: unpack(D.ret[k]), sizes: blockSizes(x.length) });
}
const yOf = (S, y) => (y === 'abs' ? S.abs : y === 'sq' ? S.x.map(v => v * v) : S.x);
const EST = {};
function est(k, y) {                // all_estimates of the chapter, for r or |r|
  const key = k + y; if (EST[key]) return EST[key];
  const S = ser(k), x = yOf(S, y), g = gph(x, D.gph_power, key), lo = loTest(x);
  return (EST[key] = { n: S.n, rs: hurstRS(x, S.sizes), dfa: hurstDFA(x, S.sizes), gph_d: g.d, gph_se: g.se, gph_m: g.m, gph_H: g.H, lo });
}
const mcBand = (k, e) => NUM.markets[k].mc[e];
const RO = {};
function rolling(k, y, W) {          // rolling_hurst of the chapter: DFA on windows of W observations moved by 21, dated at the window's last day
  const key = k + y + W; if (RO[key]) return RO[key];
  const S = ser(k), x = yOf(S, y), sz = blockSizes(W), h = [], dts = [];
  for (let e = W; e <= x.length; e += D.step) { h.push(hurstDFA(x.slice(e - W, e), sz)); dts.push(S.dates[e - 1]); }
  return (RO[key] = { h, dts });
}
const bandOf = W => summ(D.mc[W].dfa);

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const MON = { ro: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
  en: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] };
const fdate = d => `${+d.slice(8, 10)} ${MON[LANG][+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const pc0 = v => nf(100 * v, 0) + '%';
const fz = (v, d) => nf(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);
const lst = a => a.join(', ');
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.span ? ` colspan="${o.span}"` : ''}>${o.v}</${tag}>`; };
  const hs = Array.isArray(head[0]) ? head : [head];
  $(id).innerHTML = `<table class="xt"><thead>${hs.map(h => `<tr>${h.map(x => cell(x, 'th')).join('')}</tr>`).join('')}</thead><tbody>` +
    rows.map(r => `<tr>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.stack || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 270) * n + (o.extra || 110) : (o.h || 380)) + 'px';
  const g = stack ? (o.vgap || 0.2) : (o.gap || 0.09), wr = o.w || new Array(n).fill(1), tot = wr.reduce((s, v) => s + v, 0), L = {}, ax = [];
  let x0 = 0;
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', w = (1 - g * (n - 1)) * wr[i] / tot, wv = (1 - g * (n - 1)) / n, dom = [x0, x0 + w], dv = [i * (wv + g), i * (wv + g) + wv];
    x0 += w + g;
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [1 - dv[1], 1 - dv[0]] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
function vstack(id, n, h, gap) {
  $(id).style.height = h + 'px';
  const g = gap || 0.06, w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '';
    L['xaxis' + s] = { anchor: 'y' + s, domain: [0, 1] }; if (i) L['xaxis' + s].matches = 'x';
    L['yaxis' + s] = { anchor: 'x' + s, domain: [1 - (i * (w + g) + w), 1 - i * (w + g)] };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack: true };
}
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const kSel = id => $(id).value;
const MG = { l: 56, r: 12, t: 26, b: 40 };
const LOGAX = (t, e) => AX(t, Object.assign({ type: 'log' }, e || {}));
const hkey = h => String(+h.toFixed(2));
const z0 = v => (Math.abs(v) < 1e-9 ? 0 : v);
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function sci(v) { if (v >= 1e-4) return nf(v, 4); const e = Math.floor(Math.log10(v)), m = v / 10 ** e; return `${nf(m, 1)} × 10${String(e).split('').map(c => SUP[c]).join('')}`; }

// ================================================================= 1. scaling of h-day returns
function scBlock() {
  const hm = +kSel('sc-h'), hs = D.h_list.filter(h => h <= hm), nw = single('sc-chart', 360), tr = [], items = [];
  ['sp500', 'bet', 'btc'].forEach(k => {
    const s = scaling(ser(k).x, hs), r = s.sd.map(v => v / s.sd[0]);
    tr.push({ x: hs, y: r, mode: 'lines+markers', line: { color: COL[k], width: 2 }, marker: { size: 7 }, name: `${NAME[k]}: slope H = ${s.H.toFixed(2)}`, hovertemplate: `${NAME[k]}<br>h = %{x}: %{y:.2f}<extra></extra>` });
    items.push([NAME[k], nf(s.H, 2), nf(r[hs.indexOf(10)], 2), nf(r[r.length - 1], 2)]);
  });
  tr.push({ x: hs, y: hs.map(h => Math.sqrt(h)), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'square-root-of-time rule, slope 0.5', hoverinfo: 'skip' });
  plot('sc-chart', tr, layout({ xaxis: LOGAX('horizon h in days (log scale)'), yaxis: LOGAX('sd of h-day returns / sd of 1-day returns'), legend: LEG(nw, -0.18), margin: { l: 60, r: 12, t: 14, b: 40 } }));
  out('sc-out', [T.sc(hm, nf(Math.sqrt(10), 2), nf(Math.sqrt(hm), 2), items, hm !== 250)]);
}

// ================================================================= 2. worked R/S example (eight returns)
function rxWindow(w) {
  if (w === 'lec') return { x: D.rs_example, d0: null };
  const S = ser('sp500'), i = w === 'last' ? S.n - 8 : S.dates.findIndex(d => d >= w);
  return { x: S.x.slice(i, i + 8), d0: S.dates[i], d1: S.dates[i + 7] };
}
function rxBlock() {
  const w = kSel('rx-w'), W = rxWindow(w), x = W.x, n = x.length, m = mean(x), dev = x.map(v => v - m), Y = []; let c = 0;
  dev.forEach(v => { c += v; Y.push(c); });
  const mx = Math.max(...Y), mn = Math.min(...Y), kx = Y.indexOf(mx) + 1, kn = Y.indexOf(mn) + 1, R = mx - mn, ss = dev.reduce((s, v) => s + v * v, 0), S = Math.sqrt(ss / n), k = Array.from({ length: n }, (_, i) => i + 1);
  const nw = single('rx-chart', 340);
  plot('rx-chart', [{ type: 'bar', x: k, y: dev, width: 0.6, marker: { color: hexA(TEAL, 0.6) }, name: 'deviation from the mean, x_k - mean', hovertemplate: 'k = %{x}: %{y:.3f}<extra></extra>' },
    { x: k, y: Y, mode: 'lines+markers', line: { color: BLUE, width: 2 }, marker: { size: 7 }, name: 'cumulative deviation Y_k', hovertemplate: 'Y_%{x} = %{y:.3f}<extra></extra>' },
    { x: [0.5, n + 0.5], y: [mx, mx], mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: `max Y_k = ${mx.toFixed(3)} (k = ${kx})`, hoverinfo: 'skip' },
    { x: [0.5, n + 0.5], y: [mn, mn], mode: 'lines', line: { color: RED, width: 1.4, dash: 'dash' }, name: `min Y_k = ${mn.toFixed(3)} (k = ${kn})`, hoverinfo: 'skip' },
    { x: [n + 0.75, n + 0.75], y: [mn, mx], mode: 'lines+markers', line: { color: DARK, width: 1.4 }, marker: { symbol: ['triangle-down', 'triangle-up'], size: 8, color: DARK }, name: `R = ${R.toFixed(3)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('k', { dtick: 1, range: [0.4, n + 1.2] }), yaxis: AX(W.d0 ? 'S&P 500 daily return (%)' : 'return (%)'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const f = v => nf(z0(v), 3), src = W.d0 ? T.rxsrc(fdate(W.d0), fdate(W.d1)) : T.rxlec;
  steps('rx-we', [T.rx1(src, x.map(f).join('; '), f(x.reduce((s, v) => s + v, 0)), n, f(m)), T.rx2(dev.map(v => (v >= 0 ? '+' : '') + f(v)).join('; ')), T.rx3(Y.map(f).join('; '), f(Y[n - 1])),
    T.rx4(f(mx), kx, f(mn), kn, f(R), f(ss), n, f(S)), T.rx5(f(R), f(S), f(R / S))]);
  out('rx-out', [T.rx(src, f(R), f(S), f(R / S), w !== 'lec')]);
}

// ================================================================= 3. fBm paths for three Hurst exponents
const FBM = {};
function fbmPath(H) { const k = hkey(H); if (FBM[k]) return FBM[k]; const x = D.fbm[k], b = [0]; for (const v of x) b.push(b[b.length - 1] + v); return (FBM[k] = { x, b }); }
function fbBlock() {
  const dl = +kSel('fb-d'), Hs = [0.5 - dl, 0.5, 0.5 + dl], cols = [RED, BLUE, GREEN], Pn = panels('fb-chart', 2, { h: 360, w: [1, 1] }), tr = [], items = [];
  Hs.forEach((H, i) => {
    const p = fbmPath(H), lab = Math.abs(H - 0.5) < 1e-9 ? 'H = 0.5: Brownian motion' : `H = ${+H.toFixed(1)}: ${H < 0.5 ? 'anti-persistent' : 'persistent'}`;
    tr.push(on2(Pn.ax[0], { x: p.b.map((_, t) => t), y: p.b, mode: 'lines', line: { color: cols[i], width: 1.1 }, name: lab, hovertemplate: `H = ${+H.toFixed(1)}, t = %{x}: %{y:.2f}<extra></extra>` }),
      on2(Pn.ax[1], { x: p.x.slice(0, 150).map((_, t) => t), y: p.x.slice(0, 150), mode: 'lines', line: { color: cols[i], width: 1 }, showlegend: false, hoverinfo: 'skip' }));
    items.push([nf(H, 1), nf(acf(p.x, 1)[0], 2), nf(2 ** (2 * H - 1) - 1, 2), nf(Math.max(...p.b) - Math.min(...p.b), 1)]);
  });
  setAx(Pn, 0, AX('time t'), AX('')); setAx(Pn, 1, AX('time t'), AX(''));
  plot('fb-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], 'fractional Brownian motion B_H(t)'), ptitle(Pn.ax[1], 'its increments (fGn), first 150')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  out('fb-out', [T.fb(items, dl !== 0.2)]);
}

// ================================================================= 4. ACF of fractional Gaussian noise
function fgBlock() {
  const H = +kSel('fg-h'), Hb = +(1 - H).toFixed(1), k = Array.from({ length: 50 }, (_, i) => i + 1), Pn = panels('fg-chart', 2, { h: 380 }), tr = [];
  [[H, BLUE], [Hb, RED]].forEach(([h, c]) => {
    tr.push(on2(Pn.ax[0], { x: k, y: D.fgn[hkey(h)], mode: 'markers', marker: { color: c, size: 5 }, name: `H = ${h}: sample ACF (n = 5000)`, hovertemplate: `H = ${h}, k = %{x}: %{y:.3f}<extra></extra>` }),
      on2(Pn.ax[0], { x: k, y: k.map(j => fgnAcvf(j, h)), mode: 'lines', line: { color: c, width: 1.6 }, name: `H = ${h}: theoretical ACF`, hoverinfo: 'skip' }));
  });
  const kk = Array.from({ length: 1000 }, (_, i) => i + 1), rf = kk.map(j => fgnAcvf(j, H)), phi = rf[0];
  tr.push(on2(Pn.ax[1], { x: kk, y: rf, mode: 'lines', line: { color: BLUE, width: 2 }, name: `fGn, H = ${H}: hyperbolic decay, k^(2H-2)`, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' }),
    on2(Pn.ax[1], { x: kk, y: kk.map(j => (phi ** j > 1e-5 ? phi ** j : null)), mode: 'lines', line: { color: ORANGE, width: 2 }, name: `AR(1), phi = ${phi.toFixed(3)}: exponential decay`, hovertemplate: 'k = %{x}: %{y:.2e}<extra></extra>' }));
  setAx(Pn, 0, AX('lag k'), AX('autocorrelation')); setAx(Pn, 1, LOGAX('lag k (log scale)'), LOGAX('autocorrelation (log scale)', { range: [-4, 0], dtick: 1, exponentformat: 'power' }));
  plot('fg-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  const s = D.fgn[hkey(H)], gap = mean(k.map((j, i) => fgnAcvf(j, H) - s[i]));
  out('fg-out', [T.fg(nf(H, 1), nf(phi, 3), nf(Hb, 1), nf(fgnAcvf(1, Hb), 3), nf(rf[9], 3), nf(rf[49], 3), nf(rf[99], 3), sci(phi ** 10), sci(phi ** 100), nf(s[0], 3), nf(gap, 3), H !== 0.8)]);
}

// ================================================================= 5. ARFIMA(0,d,0) with d and -d
function afBlock() {
  const d = +kSel('af-d'), keys = [d, -d], nw = narrow('af-chart'), L = {}, ax = [], tr = [], k = Array.from({ length: 50 }, (_, i) => i + 1), cols = [BLUE, RED];
  $('af-chart').style.height = (nw ? 900 : 480) + 'px';
  for (let i = 0; i < 4; i++) {
    const s = i ? String(i + 1) : '', row = i >> 1, col = i & 1;
    L['xaxis' + s] = { domain: nw ? [0, 1] : col ? [0.72, 1] : [0, 0.64], anchor: 'y' + s };
    L['yaxis' + s] = { domain: nw ? [1 - (i + 1) * 0.215 + 0.03, 1 - i * 0.215] : row ? [0.1, 0.46] : [0.6, 0.96], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  const items = [];
  keys.forEach((dd, r) => {
    const A = D.arfima[hkey(dd)], th = arfAcf(51, dd).slice(1);
    tr.push(on2(ax[2 * r], { x: A.path.map((_, t) => t), y: A.path, mode: 'lines', line: { color: cols[r], width: 0.8 }, name: `ARFIMA(0,d,0), d = ${dd}: path`, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' }),
      on2(ax[2 * r + 1], { type: 'bar', x: k, y: A.acf, marker: { color: hexA(cols[r], 0.55) }, name: `d = ${dd}: sample ACF (n = 20000)`, hovertemplate: 'k = %{x}: %{y:.3f}<extra></extra>' }),
      on2(ax[2 * r + 1], { x: k, y: th, mode: 'lines', line: { color: DARK, width: 1.4 }, name: 'theoretical ACF', showlegend: r === 0, hoverinfo: 'skip' }));
    const f3 = v => (Math.abs(v) < 0.01 ? nf(v, 4) : nf(v, 3));
    items.push([nf(dd, 2), f3(th[0]), f3(th[9]), f3(th[49]), f3(A.acf[0]), nf(sd0(A.path), 2), nf(Math.sqrt(arfVar(dd)), 2)]);
  });
  ax.forEach((a, i) => { Object.assign(L[a.xa], AX(i >> 1 ? (i & 1 ? 'lag k' : 'time t') : '')); Object.assign(L[a.ya], AX(i & 1 ? '' : 'x(t)')); });
  plot('af-chart', tr, layout(Object.assign(L, { bargap: 0.2, legend: Object.assign(layout().legend, { y: nw ? -0.06 : -0.12 }), margin: { l: 56, r: 12, t: 10, b: 30 } })));
  out('af-out', [T.af(items, d !== 0.4)]);
}

// ================================================================= 6. R/S and DFA: log-log plots
function rdBlock() {
  const k = kSel('rd-s'), S = ser(k), sz = S.sizes, Pn = panels('rd-chart', 2, { h: 380 }), tr = [], sl = {};
  const iid = D.iid[k], F0 = iid.F0.map(v => v / iid.F0[0]);
  [['rs', 0, iid.rs0], ['dfa', 1, F0]].forEach(([e, i, base]) => {
    [['r', S.x, COL[k], 'circle', 'returns r'], ['abs', S.abs, PURPLE, 'square', 'absolute returns |r|']].forEach(([y, x, c, sym, nm]) => {
      let v = e === 'rs' ? rsCurve(x, sz) : dfaCurve(x, sz); if (e === 'dfa') v = v.map(u => u / v[0]);
      const b = ols(L10(sz), L10(v)).b; sl[e + '_' + y] = b;
      tr.push(on2(Pn.ax[i], { x: sz, y: v, mode: 'markers', marker: { color: c, symbol: sym, size: 7 }, name: `${e === 'rs' ? 'R/S' : 'DFA'}, ${nm}: slope ${b.toFixed(2)}`, hovertemplate: 'n = %{x}: %{y:.3f}<extra></extra>' }));
    });
    sl[e + '_iid'] = ols(L10(sz), L10(base)).b;
    tr.push(on2(Pn.ax[i], { x: sz, y: base, mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'i.i.d. Normal (Monte Carlo mean)', showlegend: i === 0, hoverinfo: 'skip' }));
  });
  setAx(Pn, 0, LOGAX('block size n (log scale)'), LOGAX('(R/S)_n (log scale)')); setAx(Pn, 1, LOGAX('block size n (log scale)'), LOGAX('F(n) / F(10) (log scale)'));
  plot('rd-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], `R/S analysis, ${NAME[k]}`), ptitle(Pn.ax[1], `DFA, ${NAME[k]}`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  out('rd-out', [T.rd(NAME[k], fint(S.n), sz.length, sz[0], fint(sz[sz.length - 1]), nf(sl.rs_r, 2), nf(sl.rs_iid, 2), nf(sl.dfa_r, 2), nf(sl.dfa_iid, 2), nf(sl.rs_abs, 2), nf(sl.dfa_abs, 2), k !== 'sp500')]);
}

// ================================================================= 7. GPH regression
function gpBlock() {
  const k = kSel('gp-s'), p = +kSel('gp-p'), S = ser(k), Pn = panels('gp-chart', 2, { h: 380 }), tr = [], G = {};
  [['r', S.x, COL[k], 'returns r', 0], ['abs', S.abs, PURPLE, 'absolute returns |r|', 1]].forEach(([y, x, c, nm, i]) => {
    const g = gph(x, p, k + y); G[y] = g;
    const x0 = Math.min(...g.X), x1 = Math.max(...g.X);
    tr.push(on2(Pn.ax[i], { x: g.X, y: g.y, mode: 'markers', marker: { color: c, size: 5 }, name: `${nm}: log periodogram, m = ${g.m} frequencies`, hovertemplate: '%{x:.2f}: %{y:.2f}<extra></extra>' }),
      on2(Pn.ax[i], { x: [x0, x1], y: [g.a + g.b * x0, g.a + g.b * x1], mode: 'lines', line: { color: DARK, width: 1.4 }, name: 'OLS line: slope = d', showlegend: i === 0, hoverinfo: 'skip' }));
  });
  setAx(Pn, 0, AX('-log(4 sin^2(lambda_j / 2))'), AX('log I(lambda_j)')); setAx(Pn, 1, AX('-log(4 sin^2(lambda_j / 2))'), AX(''));
  plot('gp-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], `returns: slope d = ${G.r.d.toFixed(2)} (SE ${G.r.se.toFixed(2)})`), ptitle(Pn.ax[1], `absolute returns: slope d = ${G.abs.d.toFixed(2)} (SE ${G.abs.se.toFixed(2)})`)],
    legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  out('gp-out', [T.gp(NAME[k], nf(p, 1), G.r.m, nf(G.r.d, 3), nf(G.r.se, 3), nf(G.r.d - 1.96 * G.r.se, 3), nf(G.r.d + 1.96 * G.r.se, 3), nf(G.abs.d, 3), nf(G.abs.H, 2), k !== 'sp500' || p !== 0.5)]);
}

// ================================================================= 8. classical against modified R/S: size and power
const rej = v => v.filter(a => !(LO <= a && a <= HI)).length / v.length;
function loBlock() {
  const n = kSel('lo-n'), L = D.lo[n], Pn = panels('lo-chart', 2, { h: 380 }), tr = [];
  [['ar', D.phis, 0, 'AR(1) coefficient phi (short memory: H0 true)'], ['fgn', D.hs_power, 1, 'Hurst exponent H of fGn (long memory if H > 0.5)']].forEach(([g, xs, i]) => {
    const lab = xs.map(String);
    tr.push(on2(Pn.ax[i], { type: 'bar', x: lab, y: xs.map(v => rej(L[g][String(v)].V0)), marker: { color: RED }, name: 'classical R/S test (q = 0)', showlegend: i === 0, offsetgroup: 'a', hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
      on2(Pn.ax[i], { type: 'bar', x: lab, y: xs.map(v => rej(L[g][String(v)].V)), marker: { color: BLUE }, name: "Lo's modified R/S test (Andrews q)", showlegend: i === 0, offsetgroup: 'b', hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
      on2(Pn.ax[i], { x: [lab[0], lab[lab.length - 1]], y: [0.05, 0.05], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'nominal level 5%', showlegend: i === 0, hoverinfo: 'skip' }));
  });
  setAx(Pn, 0, AX('AR(1) coefficient phi (H0 true)', { type: 'category' }), AX('share of rejections of short memory', { range: [0, 1.02] }));
  setAx(Pn, 1, AX('Hurst exponent H of fGn', { type: 'category' }), AX('', { range: [0, 1.02] }));
  plot('lo-chart', tr, layout(Object.assign(Pn.L, { barmode: 'group', annotations: [ptitle(Pn.ax[0], `AR(1), N = ${n}`), ptitle(Pn.ax[1], `fractional Gaussian noise, N = ${n}`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  const g = (grp, v, e) => pc0(rej(L[grp][v][e]));
  out('lo-out', [T.lo(n, g('ar', '0.5', 'V0'), g('ar', '0.5', 'V'), nf(mean(L.ar['0.5'].q), 1), g('fgn', '0.7', 'V0'), g('fgn', '0.7', 'V'), g('ar', '0', 'V0'), g('ar', '0', 'V'), g('fgn', '0.8', 'V'), n !== '1000')]);
}

// ================================================================= 9. Monte Carlo: bias and spread
function mcBlock() {
  const lv = +kSel('mc-l'), a = (1 - lv) / 2, nw = single('mc-chart', 380), Ns = D.mc_n, tr = [], R = {};
  [['rs', RED, 'R/S', -0.06], ['dfa', BLUE, 'DFA', 0], ['gph', GREEN, 'GPH, H = d + 0.5', 0.06]].forEach(([e, c, lab, off]) => {
    const s = Ns.map(n => summ(D.mc[n][e])), m = s.map(o => o.mean), lo = s.map(o => o.q(a)), hi = s.map(o => o.q(1 - a)); R[e] = { m, lo, hi };
    const bx = [], by = []; Ns.forEach((n, i) => { const x = Math.log10(n) + off; bx.push(x, x, null); by.push(lo[i], hi[i], null); });
    tr.push({ x: bx, y: by, mode: 'lines+markers', line: { color: c, width: 1.8 }, marker: { symbol: 'line-ew-open', size: 9, color: c, line: { width: 1.8, color: c } }, name: `${lab}: ${Math.round(100 * lv)}% band`, legendgroup: e, showlegend: false, hoverinfo: 'skip' },
      { x: Ns.map(n => Math.log10(n) + off), y: m, mode: 'markers', marker: { color: c, size: 8 }, name: `${lab}: mean and ${Math.round(100 * lv)}% band`, legendgroup: e, customdata: Ns.map((n, i) => [n, lo[i], hi[i]]),
        hovertemplate: `${lab}, N = %{customdata[0]}: mean %{y:.3f}, band [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>` });
  });
  tr.push({ x: [Math.log10(200), Math.log10(6000)], y: [0.5, 0.5], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'true value H = 0.5', hoverinfo: 'skip' });
  plot('mc-chart', tr, layout({ xaxis: AX('sample size N (log scale)', { tickvals: Ns.map(n => Math.log10(n)), ticktext: Ns.map(String) }), yaxis: AX('estimated H'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const i1 = Ns.indexOf(1000), i0 = 0, w = (e, i) => nf(R[e].hi[i] - R[e].lo[i], 2);
  out('mc-out', [T.mc(Math.round(100 * lv), nf(R.rs.m[i0], 3), nf(R.rs.m[i1], 3), nf(R.rs.m[Ns.length - 1], 3), nf(R.dfa.m[i1], 3), nf(R.gph.m[i1], 3), w('rs', i1), w('dfa', i1), w('gph', i1), w('dfa', i0),
    nf(R.rs.lo[i1], 2), nf(R.rs.hi[i1], 2), lv !== 0.95)]);
}

// ================================================================= 10. the Nile: R/S before and after removing the break
function nlBlock() {
  const b = +$('nl-b').value, Nl = D.nile, yr = Nl.year, v = Nl.flow, sz = Nl.sizes, Pn = panels('nl-chart', 2, { h: 380, w: [1.6, 1] }); label('nl-b').textContent = String(b);
  const ip = yr.filter(y => y <= b).length, pre = v.slice(0, ip), post = v.slice(ip), mp = mean(pre), mq = mean(post), adj = pre.map(u => u - mp).concat(post.map(u => u - mq));
  const r1 = rsCurve(v, sz), r2 = rsCurve(adj, sz), H1 = ols(L10(sz), L10(r1)).b, H2 = ols(L10(sz), L10(r2)).b, d1 = hurstDFA(v, sz), d2 = hurstDFA(adj, sz), nu = summ(Nl.null);
  const tr = [on2(Pn.ax[0], { x: yr, y: v, mode: 'lines', line: { color: BLUE, width: 1.4 }, name: 'annual flow at Aswan (10^8 m^3)', hovertemplate: '%{x}: %{y}<extra></extra>' }),
    on2(Pn.ax[0], { x: [yr[0], b], y: [mp, mp], mode: 'lines', line: { color: RED, width: 3 }, name: `mean ${yr[0]}-${b}: ${mp.toFixed(0)}`, hoverinfo: 'skip' }),
    on2(Pn.ax[0], { x: [b + 1, yr[yr.length - 1]], y: [mq, mq], mode: 'lines', line: { color: GREEN, width: 3 }, name: `mean ${b + 1}-${yr[yr.length - 1]}: ${mq.toFixed(0)}`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: sz, y: r1, mode: 'lines+markers', marker: { symbol: 'circle', size: 6 }, line: { color: BLUE, width: 1.4 }, name: `raw series: H = ${H1.toFixed(2)}`, hovertemplate: 'n = %{x}: %{y:.2f}<extra></extra>' }),
    on2(Pn.ax[1], { x: sz, y: r2, mode: 'lines+markers', marker: { symbol: 'square', size: 6 }, line: { color: ORANGE, width: 1.4 }, name: `regime means removed: H = ${H2.toFixed(2)}`, hovertemplate: 'n = %{x}: %{y:.2f}<extra></extra>' }),
    on2(Pn.ax[1], { x: sz, y: sz.map(n => Math.sqrt(n) * r1[0] / Math.sqrt(sz[0])), mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'slope 0.5', hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('year'), AX('')); setAx(Pn, 1, LOGAX('block size n'), LOGAX('(R/S)_n'));
  plot('nl-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  out('nl-out', [T.nl(b, fint(mp), fint(mq), nf(100 * (1 - mq / mp), 0), nf(H1, 2), nf(d1, 2), nf(H2, 2), nf(d2, 2), nf(nu.q025, 2), nf(nu.q975, 2), nf(nu.mean, 2), b !== Nl.brk)]);
}

// ================================================================= 11. spurious long memory in simulations
function suBlock() {
  const n = kSel('su-n'), S = D.spurious[n], Pn = panels('su-chart', 2, { h: 380 }), tr = [], dl = D.deltas, ps = D.pers, M = (g, a, e) => mean(S[g][String(a)][e]);
  [['rs', RED, 'R/S'], ['dfa', BLUE, 'DFA'], ['gph', GREEN, 'GPH, d + 0.5']].forEach(([e, c, lab]) => tr.push(on2(Pn.ax[0], { x: dl, y: dl.map(a => M('break', a, e)), mode: 'lines+markers', line: { color: c, width: 1.8 }, marker: { size: 7 }, name: `${lab}: mean estimate`, hovertemplate: `${lab}, shift %{x}: %{y:.3f}<extra></extra>` })));
  [['abs_dfa', PURPLE, 'square', 'DFA of |r| (GARCH)'], ['abs_gph', ORANGE, 'diamond', 'GPH of |r| (GARCH), d + 0.5'], ['ret_dfa', TEAL, 'triangle-up', 'DFA of r (GARCH)']].forEach(([e, c, sym, lab]) =>
    tr.push(on2(Pn.ax[1], { x: ps, y: ps.map(a => M('garch', a, e)), mode: 'lines+markers', line: { color: c, width: 1.8 }, marker: { symbol: sym, size: 7 }, name: lab, hovertemplate: `${lab}, %{x}: %{y:.3f}<extra></extra>` })));
  tr.push(on2(Pn.ax[0], { x: [0, 1], y: [0.5, 0.5], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'H = 0.5 (no long memory)', hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: [0.8, 0.99], y: [0.5, 0.5], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, showlegend: false, hoverinfo: 'skip' }));
  setAx(Pn, 0, AX('mean shift at N/2 (in standard deviations)'), AX('estimated H')); setAx(Pn, 1, AX('GARCH persistence alpha + beta (short memory)'), AX(''));
  plot('su-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], `i.i.d. N(0,1) with a mean shift, N = ${n}`), ptitle(Pn.ax[1], `GARCH(1,1), alpha = 0.10, N = ${n}`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  const rr = ps.map(a => M('garch', a, 'ret_dfa'));
  out('su-out', [T.su(n, nf(M('break', 0.5, 'gph'), 2), nf(M('break', 0.5, 'dfa'), 2), nf(M('break', 0.5, 'rs'), 2), nf(M('break', 0, 'rs'), 2), nf(M('garch', 0.95, 'abs_dfa'), 2), nf(M('garch', 0.99, 'abs_dfa'), 2),
    nf(Math.min(...rr), 2), nf(Math.max(...rr), 2), n !== '2000')]);
}

// ================================================================= 12. ACF of r, |r| and r^2
function vaBlock() {
  const k = kSel('va-s'), Lg = +kSel('va-l'), S = ser(k), nw = single('va-chart', 380), lags = Array.from({ length: Lg }, (_, i) => i + 1), band = 1.96 / Math.sqrt(S.n), A = {};
  const tr = [{ x: [1, Lg, Lg, 1], y: [-band, -band, band, band], fill: 'toself', fillcolor: hexA(BLUE, 0.13), line: { width: 0 }, mode: 'lines', name: '95% band of an i.i.d. series', hoverinfo: 'skip' }];
  [['r', COL[k], 'returns r'], ['abs', PURPLE, 'absolute returns |r|'], ['sq', ORANGE, 'squared returns r^2']].forEach(([y, c, nm]) => {
    A[y] = acf(yOf(S, y), Lg);
    tr.push({ x: lags, y: A[y], mode: 'lines', line: { color: c, width: 1.3 }, name: nm, hovertemplate: `${nm}<br>k = %{x}: %{y:.3f}<extra></extra>` });
  });
  plot('va-chart', tr, layout({ xaxis: AX('lag k (days)'), yaxis: AX(`${NAME[k]}: autocorrelation`), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const r2 = A.r.slice(1), out2 = r2.filter(v => Math.abs(v) > band).length, top = r2.map((v, i) => [Math.abs(v), i + 2]).sort((u, v) => v[0] - u[0]).slice(0, 3).map(t => t[1]).sort((u, v) => u - v);
  const at = (y, L) => (L <= Lg ? nf(A[y][L - 1], 3) : '–');
  out('va-out', [T.va(NAME[k], Lg, nf(band, 3), at('abs', 1), at('abs', 50), at('abs', 100), A.abs.filter(v => v > band).length, at('sq', 1), at('sq', 50), at('r', 1), nf(Math.max(...r2.map(Math.abs)), 3), out2,
    nf(0.05 * (Lg - 1), 0), lst(top), k !== 'sp500' || Lg !== 250)]);
}

// ================================================================= 13. six markets: estimates and the shuffle test
const ENAME = { dfa: 'DFA', rs: 'R/S', gph: 'GPH (d + 0.5)' };
function mkBlock() {
  const e = kSel('mk-e'), nw = single('mk-chart', 380), names = MK.map(k => (nw ? SH[k] : NAME[k])), tr = [], shapes = [], vr = [], va = [], vs = [];
  MK.forEach((k, j) => {
    const b = mcBand(k, e), E = est(k, 'r'), A = est(k, 'abs'), pick = o => (e === 'gph' ? o.gph_H : o[e]);
    shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0: j - 0.3, x1: j + 0.3, y0: b.q025, y1: b.q975, fillcolor: hexA(BLUE, 0.15), line: { width: 0 }, layer: 'below' });
    vr.push(pick(E)); va.push(pick(A)); vs.push(D.shuffled[k][e]);
  });
  const xs = MK.map((_, j) => j);
  tr.push({ x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: hexA(BLUE, 0.3) }, name: '95% band of i.i.d. series of the same length', hoverinfo: 'skip' },
    { x: xs, y: vr, mode: 'markers', marker: { color: RED, size: 10 }, name: `${ENAME[e]} exponent of returns r`, customdata: names, hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' },
    { x: xs, y: va, mode: 'markers', marker: { color: PURPLE, symbol: 'square', size: 10 }, name: `${ENAME[e]} exponent of |r|`, customdata: names, hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' },
    { x: xs, y: vs, mode: 'markers', marker: { color: GREEN, symbol: 'x', size: 10 }, name: `${ENAME[e]} exponent of |r| after a random shuffle`, customdata: names, hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' },
    { x: [-0.5, MK.length - 0.5], y: [0.5, 0.5], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'H = 0.5', hoverinfo: 'skip' });
  plot('mk-chart', tr, layout({ xaxis: AX('', { tickvals: xs, ticktext: names, range: [-0.5, MK.length - 0.5] }), yaxis: AX('estimated H'), shapes, legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ins = MK.filter((k, j) => vs[j] >= mcBand(k, e).q025 && vs[j] <= mcBand(k, e).q975).length, above = MK.filter((k, j) => vr[j] > mcBand(k, e).q975).map(k => SH[k]);
  out('mk-out', [T.mk(ENAME[e], nf(Math.min(...va), 2), nf(Math.max(...va), 2), nf(Math.min(...vs), 2), nf(Math.max(...vs), 2), ins, above, e !== 'dfa')]);
}

// ================================================================= 14. rolling DFA exponents
function roBlock() {
  const W = +kSel('ro-w'), B = W === 1000 ? NUM.rolling.band : bandOf(W), ks = ['sp500', 'bet', 'btc'], Pn = vstack('ro-chart', 3, narrow('ro-chart') ? 720 : 600, 0.07), tr = [], shapes = [], items = [];
  ks.forEach((k, i) => {
    const a = Pn.ax[i], h = rolling(k, 'r', W), ha = rolling(k, 'abs', W), d0 = h.dts[0], d1 = h.dts[h.dts.length - 1], n = h.h.length;
    shapes.push({ type: 'rect', xref: a.x, yref: a.y, x0: d0, x1: d1, y0: B.q025, y1: B.q975, fillcolor: hexA(BLUE, 0.13), line: { width: 0 }, layer: 'below' },
      { type: 'line', xref: a.x, yref: a.y, x0: d0, x1: d1, y0: 0.5, y1: 0.5, line: { color: DARK, width: 1, dash: 'dash' } });
    tr.push(on2(a, { x: h.dts, y: h.h, mode: 'lines', line: { color: COL[k], width: 1.4 }, name: `${NAME[k]}: DFA exponent of r`, hovertemplate: `${NAME[k]}, r<br>%{x}: %{y:.3f}<extra></extra>` }),
      on2(a, { x: ha.dts, y: ha.h, mode: 'lines', line: { color: PURPLE, width: 1.1 }, name: 'DFA exponent of |r|', showlegend: i === 0, hovertemplate: `${NAME[k]}, |r|<br>%{x}: %{y:.3f}<extra></extra>` }));
    if (i === 0) tr.push(on2(a, { x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: hexA(BLUE, 0.3) }, name: `95% band of i.i.d. series (N = ${W})`, hoverinfo: 'skip' }));
    Pn.L[a.ya] = Object.assign(Pn.L[a.ya], AX('H'));
    const im = h.h.indexOf(Math.max(...h.h)), il = h.h.indexOf(Math.min(...h.h)), t3 = Math.floor(n / 3);
    items.push([NAME[k], fint(n), fdate(d0), nf(Math.min(...h.h), 2), nf(Math.max(...h.h), 2), fdate(h.dts[il]), fdate(h.dts[im]), nf(100 * h.h.filter(v => v > B.q975).length / n, 0), nf(100 * h.h.filter(v => v < B.q025).length / n, 0),
      nf(mean(h.h.slice(0, t3)), 2), nf(mean(h.h.slice(n + Math.floor(-n / 3))), 2), nf(Math.min(...ha.h), 2), nf(Math.max(...ha.h), 2)]);
  });
  Pn.L.xaxis = Object.assign(Pn.L.xaxis, { type: 'date' });
  plot('ro-chart', tr, layout(Object.assign(Pn.L, { shapes, legend: LEG(narrow('ro-chart'), -0.06), margin: { l: 56, r: 12, t: 10, b: 30 } })));
  out('ro-out', [T.ro(String(W), nf(B.q025, 2), nf(B.q975, 2), items, W !== 1000)]);
}

// ================================================================= x1. fractal dimension: box counting on the fBm paths (extra computation)
function boxCount(b, kmax) {
  const n = b.length - 1, mx = Math.max(...b), mn = Math.min(...b), y = b.map(v => (v - mn) / (mx - mn)), res = [];
  for (let k = 1; k <= kmax; k++) {
    const M = 2 ** k, eps = 1 / M; let N = 0; const boxes = [];
    for (let c = 0; c < M; c++) {
      const i0 = Math.floor(c * n / M), i1 = Math.ceil((c + 1) * n / M); let lo = Infinity, hi = -Infinity;
      for (let i = i0; i <= i1; i++) { if (y[i] < lo) lo = y[i]; if (y[i] > hi) hi = y[i]; }
      const b0 = Math.min(Math.floor(lo / eps), M - 1), b1 = Math.min(Math.floor(hi / eps), M - 1); N += b1 - b0 + 1; boxes.push([c, b0, b1]);
    }
    res.push({ k, eps, N, boxes });
  }
  const o = ols(res.map(r => r.k), res.map(r => Math.log2(r.N)));
  return { y, res, D: o.b, a: o.a };
}
function fdBlock() {
  const H = +kSel('fd-h'), km = +kSel('fd-k'), p = fbmPath(H), B = boxCount(p.b, km), show = B.res[Math.min(3, km - 1)], Pn = panels('fd-chart', 2, { h: 380 }), tr = [], n = p.b.length - 1;
  const bx = [], by = []; show.boxes.forEach(([c, b0, b1]) => { for (let j = b0; j <= b1; j++) { bx.push(c * show.eps, (c + 1) * show.eps, (c + 1) * show.eps, c * show.eps, c * show.eps, null); by.push(j * show.eps, j * show.eps, (j + 1) * show.eps, (j + 1) * show.eps, j * show.eps, null); } });
  tr.push(on2(Pn.ax[0], { x: bx, y: by, fill: 'toself', fillcolor: hexA(AMBER, 0.18), line: { color: hexA(AMBER, 0.7), width: 0.6 }, mode: 'lines', name: `boxes of side 1/${2 ** show.k} that cover the path: ${show.N}`, hoverinfo: 'skip' }),
    on2(Pn.ax[0], { x: B.y.map((_, i) => i / n), y: B.y, mode: 'lines', line: { color: BLUE, width: 1 }, name: `fBm path, H = ${H} (rescaled to the unit square)`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: B.res.map(r => r.k), y: B.res.map(r => Math.log2(r.N)), mode: 'markers', marker: { color: RED, size: 8 }, name: 'log2 N(epsilon)', hovertemplate: 'epsilon = 1/2^%{x}: log2 N = %{y:.2f}<extra></extra>' }),
    on2(Pn.ax[1], { x: [1, km], y: [B.a + B.D, B.a + B.D * km], mode: 'lines', line: { color: DARK, width: 1.4 }, name: `OLS slope: box-counting D = ${B.D.toFixed(2)}`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: [1, km], y: [B.a + B.D, B.a + B.D + (2 - H) * (km - 1)], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dash' }, name: `slope 2 - H = ${(2 - H).toFixed(1)}`, hoverinfo: 'skip' }));
  setAx(Pn, 0, AX('t (rescaled)', { range: [0, 1] }), AX('B_H(t) (rescaled)', { range: [0, 1] })); setAx(Pn, 1, AX('log2(1 / epsilon)', { dtick: 1 }), AX('log2 N(epsilon)'));
  plot('fd-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  steps('fd-we', [T.fd1, T.fd2(nf(H, 1), String(n)), T.fd3(B.res.map(r => [r.k, 2 ** r.k, fint(r.N)])), T.fd4(km, nf(B.D, 3), nf(2 - H, 1), nf(B.D - (2 - H), 3))]);
  out('fd-out', [T.fd(nf(H, 1), km, nf(B.D, 2), nf(2 - H, 1))]);
}

// ================================================================= x2. from R/S to the Hurst exponent (worked example)
function rhBlock() {
  const k = kSel('rh-s'), y = kSel('rh-y'), S = ser(k), x = yOf(S, y), sz = S.sizes, j = Math.min(+$('rh-n').value, sz.length) - 1, n = sz[j], rs = rsCurve(x, sz), lx = L10(sz), ly = L10(rs), o = ols(lx, ly), nw = single('rh-chart', 360);
  label('rh-n').textContent = String(n);
  plot('rh-chart', [{ x: lx, y: ly, mode: 'markers', marker: { color: y === 'abs' ? PURPLE : COL[k], size: 8 }, name: `${NAME[k]}, ${y === 'abs' ? '|r|' : 'r'}: log10 (R/S)_n`, hovertemplate: 'log10 n = %{x:.3f}: %{y:.3f}<extra></extra>' },
    { x: [lx[0], lx[lx.length - 1]], y: [o.a + o.b * lx[0], o.a + o.b * lx[lx.length - 1]], mode: 'lines', line: { color: DARK, width: 1.6 }, name: `OLS line: slope H = ${o.b.toFixed(3)}`, hoverinfo: 'skip' },
    { x: [lx[0], lx[lx.length - 1]], y: [ly[0], ly[0] + 0.5 * (lx[lx.length - 1] - lx[0])], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dash' }, name: 'slope 0.5 from the first point', hoverinfo: 'skip' },
    { x: [lx[j]], y: [ly[j]], mode: 'markers', marker: { color: AMBER, size: 15, symbol: 'circle-open', line: { width: 3 } }, name: `n = ${n}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('log10 n'), yaxis: AX('log10 (R/S)_n'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const m = Math.floor(S.n / n), b = mcBand(k, 'rs');
  steps('rh-we', [T.rh1(NAME[k], y, fint(S.n), sz.length, sz[0], fint(sz[sz.length - 1])), T.rh2(n, fint(m), S.n - m * n, nf(rs[j], 3), nf(lx[j], 3), nf(ly[j], 3)),
    T.rh3(nf(o.mx, 4), nf(o.my, 4), nf(o.sxx, 4), nf(o.sxy, 4), nf(o.b, 3), nf(o.a, 3)), T.rh4(fint(S.n), nf(b.q025, 2), nf(b.q975, 2), o.b > b.q975 ? 1 : o.b < b.q025 ? -1 : 0, y)]);
  out('rh-out', [T.rh(NAME[k], y, nf(o.b, 3), nf(b.q025, 2), nf(b.q975, 2), k !== 'sp500' || y !== 'r')]);
}

// ================================================================= x3. how fast does memory fade? (worked example)
function mwBlock() {
  const H = +$('mw-h').value, kl = +kSel('mw-k'), r1 = 2 ** (2 * H - 1) - 1, ks = Array.from({ length: 250 }, (_, i) => i + 1), nw = single('mw-chart', 360); label('mw-h').textContent = nf(H, 2);
  const f = ks.map(k => fgnAcvf(k, H)), ar = ks.map(k => (r1 ** k > 1e-6 ? r1 ** k : null));
  plot('mw-chart', [{ x: ks, y: f, mode: 'lines', line: { color: BLUE, width: 2 }, name: `fGn, H = ${H.toFixed(2)}: rho(k)`, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' },
    { x: ks, y: ks.map(k => H * (2 * H - 1) * k ** (2 * H - 2)), mode: 'lines', line: { color: TEAL, width: 1.2, dash: 'dot' }, name: 'large-k approximation H(2H - 1) k^(2H - 2)', hoverinfo: 'skip' },
    { x: ks, y: ar, mode: 'lines', line: { color: ORANGE, width: 2 }, name: `AR(1), phi = rho(1) = ${r1.toFixed(3)}`, hovertemplate: 'k = %{x}: %{y:.2e}<extra></extra>' },
    { x: [kl, kl], y: [fgnAcvf(kl, H), r1 ** kl > 1e-6 ? r1 ** kl : 1e-6], mode: 'markers', marker: { color: RED, size: 10 }, name: `lag ${kl}`, hoverinfo: 'skip' }],
  layout({ xaxis: LOGAX('lag k (log scale)'), yaxis: LOGAX('autocorrelation (log scale)', { range: [-6, 0], dtick: 1, exponentformat: 'power' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ex = sci;
  steps('mw-we', [T.mw1(nf(H, 2), nf(2 * H - 1, 2), nf(r1, 3)), T.mw2(nf(fgnAcvf(10, H), 3), nf(fgnAcvf(50, H), 3), nf(fgnAcvf(100, H), 3), kl, nf(fgnAcvf(kl, H), 4), nf(H * (2 * H - 1) * kl ** (2 * H - 2), 4)),
    T.mw3(nf(r1, 3), ex(r1 ** 10), kl, ex(r1 ** kl)), T.mw4(kl, nf(fgnAcvf(kl, H), 4), ex(r1 ** kl))]);
  out('mw-out', [T.mw(nf(H, 2), nf(r1, 3), nf(fgnAcvf(100, H), 3), ex(r1 ** 100), H !== 0.8)]);
}

// ================================================================= x4. ARFIMA(0, 0.3, 0) (worked example)
function crossLag(f0, step, thr, cap) { let r = f0, k = 1; while (r >= thr && k < cap) { k++; r = step(r, k); } return r < thr ? k : null; }
function awBlock() {
  const d = +$('aw-d').value, thr = +kSel('aw-t'), nw = single('aw-chart', 360); label('aw-d').textContent = nf(d, 2);
  const rho = arfAcf(1001, d), r1 = rho[1], ks = Array.from({ length: 1000 }, (_, i) => i + 1);
  const kA = crossLag(r1, (r, k) => r * r1, thr, 1e7), kF = crossLag(r1, (r, k) => r * (k - 1 + d) / (k - d), thr, 1e7);
  plot('aw-chart', [{ x: ks, y: ks.map(k => rho[k]), mode: 'lines', line: { color: BLUE, width: 2 }, name: `ARFIMA(0,d,0), d = ${d.toFixed(2)}: rho(k)`, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' },
    { x: ks, y: ks.map(k => (r1 ** k > 1e-6 ? r1 ** k : null)), mode: 'lines', line: { color: ORANGE, width: 2 }, name: `AR(1), phi = rho(1) = ${r1.toFixed(3)}`, hovertemplate: 'k = %{x}: %{y:.2e}<extra></extra>' },
    { x: [1, 1000], y: [thr, thr], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `threshold ${thr}`, hoverinfo: 'skip' },
    { x: [kA, kF].filter(v => v && v <= 1000), y: [kA, kF].filter(v => v && v <= 1000).map(() => thr), mode: 'markers', marker: { color: RED, size: 10 }, name: 'first lag below the threshold', hoverinfo: 'skip' }],
  layout({ xaxis: LOGAX('lag k (log scale)'), yaxis: LOGAX('autocorrelation (log scale)', { range: [-6, 0], dtick: 1, exponentformat: 'power' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const p1 = -d, p2 = p1 * (1 - d) / 2, s1 = d, s2 = s1 * (1 + d) / 2, fk = v => (v === null ? (LANG === 'ro' ? 'mai mult de ' + fint(1e7) + ' de' : 'more than ' + fint(1e7)) : fint(v));
  steps('aw-we', [T.aw1(nf(d, 2), nf(p1, 3), nf(p2, 3)), T.aw2(nf(s1, 3), nf(s2, 3)), T.aw3(nf(d, 2), nf(1 - d, 2), nf(r1, 3), nf(1 + d, 2), nf(2 - d, 2), nf(rho[2], 3), nf(rho[10], 3), nf(rho[100], 3)),
    T.aw4(nf(r1, 3), sci(r1 ** 10)), T.aw5(nf(thr, 2), kA, fk(kF), nf(d + 0.5, 2))]);
  out('aw-out', [T.aw(nf(d, 2), nf(r1, 3), nf(rho[10], 3), nf(rho[100], 3), nf(thr, 2), kA, fk(kF), d !== 0.3 || thr !== 0.05)]);
}

// ================================================================= x5. DFA: the fluctuation F(n) (worked example)
function dwBlock() {
  const k = kSel('dw-s'), y = kSel('dw-y'), S = ser(k), x = yOf(S, y), sz = S.sizes, j = Math.min(+$('dw-n').value, sz.length) - 1, n = sz[j], Y = profile(x), F = dfaCurve(x, sz, Y), o = ols(L10(sz), L10(F));
  label('dw-n').textContent = String(n);
  const Pn = panels('dw-chart', 2, { h: 380 }), nb = 4, tt = Array.from({ length: nb * n }, (_, i) => i + 1), tr = [];
  tr.push(on2(Pn.ax[0], { x: tt, y: tt.map(t => Y[t - 1]), mode: 'lines', line: { color: y === 'abs' ? PURPLE : COL[k], width: 1.3 }, name: 'profile Y_k (first four blocks)', hovertemplate: 'k = %{x}: %{y:.2f}<extra></extra>' }));
  const lx = [], ly = [];
  for (let b = 0; b < nb; b++) { const q = dfaBlock(Y, b * n, n); lx.push(b * n + 1, b * n + n, null); ly.push(q.m - q.b * q.tc, q.m + q.b * q.tc, null); }
  tr.push(on2(Pn.ax[0], { x: lx, y: ly, mode: 'lines', line: { color: RED, width: 2 }, name: `OLS line in each block of n = ${n}`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: sz, y: F, mode: 'markers', marker: { color: y === 'abs' ? PURPLE : COL[k], size: 7 }, name: 'F(n)', hovertemplate: 'n = %{x}: %{y:.3f}<extra></extra>' }),
    on2(Pn.ax[1], { x: [sz[0], sz[sz.length - 1]], y: [10 ** (o.a + o.b * Math.log10(sz[0])), 10 ** (o.a + o.b * Math.log10(sz[sz.length - 1]))], mode: 'lines', line: { color: DARK, width: 1.4 }, name: `OLS slope alpha = ${o.b.toFixed(3)}`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: [n], y: [F[j]], mode: 'markers', marker: { color: AMBER, size: 15, symbol: 'circle-open', line: { width: 3 } }, name: `n = ${n}`, hoverinfo: 'skip' }));
  setAx(Pn, 0, AX('k'), AX('cumulative deviation Y_k')); setAx(Pn, 1, LOGAX('block size n (log scale)'), LOGAX('F(n) (log scale)'));
  plot('dw-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  const m = Math.floor(S.n / n), q1 = dfaBlock(Y, 0, n), b = mcBand(k, 'dfa');
  steps('dw-we', [T.dw1(NAME[k], y, fint(S.n), nf(mean(x), 4), nf(Y[0], 3), nf(Y[1], 3), nf(z0(Y[S.n - 1]), 3)), T.dw2(n, fint(m), S.n - m * n, nf(q1.m - q1.b * q1.tc, 3), nf(q1.b, 4), nf(Math.sqrt(q1.rr / n), 4)),
    T.dw3(fint(m), n, nf(F[j], 4)), T.dw4(sz.length, nf(o.b, 3), nf(b.q025, 2), nf(b.q975, 2), y)]);
  out('dw-out', [T.dw(NAME[k], y, n, nf(F[j], 3), nf(o.b, 3), k !== 'sp500' || y !== 'r')]);
}

// ================================================================= x6. GPH regression (worked example) and the bandwidth
const GW = {};
function gphCurve(k, y) {
  const key = k + y; if (GW[key]) return GW[key];
  const S = ser(k), x = yOf(S, y), mmax = mOf(S.n, 0.75), P = periodogram(x, mmax, key), ms = [];
  for (let m = 10; m <= mmax; m += m < 100 ? 2 : 5) ms.push(m);
  return (GW[key] = { ms, d: ms.map(m => gphOf(P, S.n, m).d), P });
}
function gwBlock() {
  const k = kSel('gw-s'), p = +kSel('gw-p'), yy = kSel('gw-y'), S = ser(k), m = mOf(S.n, p), nw = single('gw-chart', 380), tr = [];
  [['r', COL[k], 'returns r'], ['abs', PURPLE, 'absolute returns |r|']].forEach(([y, c, nm]) => {
    const G = gphCurve(k, y), se = G.ms.map(mm => 1.96 * Math.PI / Math.sqrt(24 * mm));
    tr.push({ x: G.ms.concat([...G.ms].reverse()), y: G.d.map((v, i) => v + se[i]).concat(G.d.map((v, i) => v - se[i]).reverse()), fill: 'toself', fillcolor: hexA(c, 0.12), line: { width: 0 }, mode: 'lines', name: `${nm}: d +/- 1.96 SE`, hoverinfo: 'skip' },
      { x: G.ms, y: G.d, mode: 'lines', line: { color: c, width: 1.8 }, name: `${nm}: GPH d for bandwidth m`, hovertemplate: 'm = %{x}: d = %{y:.3f}<extra></extra>' },
      { x: [m], y: [gphOf(G.P, S.n, m).d], mode: 'markers', marker: { color: c, size: 11, line: { color: DARK, width: 1.5 } }, name: `m = T^${p} = ${m}`, showlegend: y === 'r', hoverinfo: 'skip' });
  });
  tr.push({ x: [10, mOf(S.n, 0.75)], y: [0, 0], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'd = 0', hoverinfo: 'skip' });
  plot('gw-chart', tr, layout({ xaxis: LOGAX('bandwidth m: number of low frequencies (log scale)'), yaxis: AX(`${NAME[k]}: GPH estimate of d`), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const G = gphCurve(k, yy), g = gphOf(G.P, S.n, m), lam1 = G.P.lam[0];
  steps('gw-we', [T.gw1(NAME[k], yy, fint(S.n), nf(p, 1), m, nf(lam1, 5), nf(G.P.lam[m - 1], 4)), T.gw2(nf(G.P.I[0], 4), nf(g.X[0], 3), nf(g.y[0], 3)),
    T.gw3(nf(g.mx, 3), nf(g.my, 3), nf(g.sxx, 2), nf(g.sxy, 2), nf(g.d, 3)), T.gw4(m, nf(g.se, 3), nf(g.d - 1.96 * g.se, 3), nf(g.d + 1.96 * g.se, 3), nf(g.H, 2))]);
  const gr = gphOf(gphCurve(k, 'r').P, S.n, m), ga = gphOf(gphCurve(k, 'abs').P, S.n, m);
  out('gw-out', [T.gw(NAME[k], nf(p, 1), m, nf(gr.d, 3), nf(ga.d, 3), nf(gr.se, 3), k !== 'sp500' || p !== 0.5)]);
}

// ================================================================= x7. Lo's modified R/S (worked example and six markets)
function lwBlock() {
  const k = kSel('lw-s'), yy = kSel('lw-y'), nw = single('lw-chart', 360), qs = Array.from({ length: 31 }, (_, i) => i), tr = [{ x: [0, 30, 30, 0], y: [LO, LO, HI, HI], fill: 'toself', fillcolor: hexA(BLUE, 0.12), line: { width: 0 }, mode: 'lines', name: '5% acceptance region of short memory [0.809, 1.862]', hoverinfo: 'skip' }];
  [['r', COL[k], 'returns r'], ['abs', PURPLE, 'absolute returns |r|']].forEach(([y, c, nm]) => {
    const t = est(k, y).lo;
    tr.push({ x: qs, y: qs.map(q => t.Vq(q)), mode: 'lines+markers', line: { color: c, width: 1.8 }, marker: { size: 4 }, name: `${nm}: V_N(q)`, hovertemplate: 'q = %{x}: %{y:.3f}<extra></extra>' },
      { x: [t.q], y: [t.V], mode: 'markers', marker: { color: c, size: 13, symbol: 'diamond', line: { color: DARK, width: 1.5 } }, name: `${nm}: Andrews q = ${t.q}`, hoverinfo: 'skip' });
  });
  plot('lw-chart', tr, layout({ xaxis: AX('lag truncation q (q = 0: classical R/S)'), yaxis: AX(`${NAME[k]}: V_N(q)`), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const t = est(k, yy).lo;
  steps('lw-we', [T.lw1(NAME[k], yy, fint(t.N), nf(t.R, 3), nf(t.g[0], 4), nf(t.V0, 3)), T.lw2(nf(t.rho, 4), nf(t.A, 3), nf(t.B, 4), nf(t.A * t.B, 3), t.q),
    T.lw3(t.q, nf(t.g[0], 4), t.g.slice(1, Math.min(t.q, 3) + 1).map(v => nf(v, 4)), nf(t.s2, 4)), T.lw4(nf(t.R, 3), fint(t.N), nf(t.s2, 4), nf(t.V, 3), t.reject)]);
  const yes = LANG === 'ro' ? 'respinsă' : 'rejected', no = LANG === 'ro' ? 'nerespinsă' : 'not rejected';
  table('lw-tab', T.h.lw, MK.map(m => { const a = est(m, 'r').lo, b = est(m, 'abs').lo; return [{ v: SH[m], c: m === k ? 'hi l' : 'l' }, nf(a.V0, 3), nf(a.V, 3), a.q, { v: a.reject ? yes : no, c: a.reject ? 'bad' : '' }, nf(b.V0, 3), nf(b.V, 3), b.q, { v: b.reject ? yes : no, c: b.reject ? 'bad' : '' }]; }));
  const a = est(k, 'r').lo, b = est(k, 'abs').lo;
  out('lw-out', [T.lw(NAME[k], nf(a.V, 3), a.q, a.reject, nf(b.V, 3), b.q, b.reject, MK.filter(m => est(m, 'r').lo.reject).map(m => SH[m]), k !== 'sp500')]);
}

// ================================================================= x8. S&P 500: all estimators (table)
function stBlock() {
  const k = kSel('st-s'), E = est(k, 'r'), A = est(k, 'abs'), nw = single('st-chart', 360), M = NUM.markets[k].mc, cats = ['R/S', 'DFA', 'GPH (d + 0.5)'], key = ['rs', 'dfa', 'gph'], xs = [0, 1, 2];
  const pick = (o, e) => (e === 'gph' ? o.gph_H : o[e]), shapes = key.map((e, i) => ({ type: 'rect', xref: 'x', yref: 'y', x0: i - 0.3, x1: i + 0.3, y0: M[e].q025, y1: M[e].q975, fillcolor: hexA(BLUE, 0.15), line: { width: 0 }, layer: 'below' }));
  plot('st-chart', [{ x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: hexA(BLUE, 0.3) }, name: `95% band of i.i.d. series, N = ${S_N(k)}`, hoverinfo: 'skip' },
    { x: xs, y: key.map(e => pick(E, e)), mode: 'markers', marker: { color: COL[k], size: 11 }, name: 'returns r', customdata: cats, hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' },
    { x: xs, y: key.map(e => pick(A, e)), mode: 'markers', marker: { color: PURPLE, symbol: 'square', size: 11 }, name: 'absolute returns |r|', customdata: cats, hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' },
    { x: [-0.5, 2.5], y: [0.5, 0.5], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'H = 0.5', hoverinfo: 'skip' }],
  layout({ xaxis: AX('', { tickvals: xs, ticktext: cats, range: [-0.5, 2.5] }), yaxis: AX(`${NAME[k]}: estimated H`), shapes, legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ci = o => `[${fz(o.gph_d - 1.96 * o.gph_se, 2)}; ${fz(o.gph_d + 1.96 * o.gph_se, 2)}]`, bd = e => `[${nf(M[e].q025, 2)}; ${nf(M[e].q975, 2)}]`, cls = (v, e) => (v > M[e].q975 || v < M[e].q025 ? 'bad' : '');
  const row = (lab, o) => [{ v: lab, c: 'l' }, { v: nf(o.rs, 2), c: cls(o.rs, 'rs') }, { v: nf(o.dfa, 2), c: cls(o.dfa, 'dfa') }, { v: `${fz(o.gph_d, 2)} (${nf(o.gph_se, 2)})`, c: cls(o.gph_H, 'gph') }, ci(o), { v: nf(o.lo.V, 2), c: o.lo.reject ? 'bad' : '' }, o.lo.q];
  table('st-tab', T.h.st, [row(T.h.r, E), row(T.h.abs, A), [{ v: T.h.band, c: 'l' }, bd('rs'), bd('dfa'), bd('gph'), '', `[${nf(LO, 3)}; ${nf(HI, 3)}]`, '']]);
  const inn = (o, e) => pick(o, e) >= M[e].q025 && pick(o, e) <= M[e].q975;
  out('st-out', [T.st(NAME[k], fint(E.n), fdate(NUM.markets[k].first), key.filter(e => !inn(E, e)).map((e, i) => cats[key.indexOf(e)]), E.lo.reject, key.filter(e => !inn(A, e)).length, A.lo.reject, k !== 'sp500')]);
}
const S_N = k => fint(ser(k).n);

// ================================================================= x9. six markets: returns and absolute returns (table)
function mtBlock() {
  const p = +kSel('mt-p'), nw = single('mt-chart', 380), names = MK.map(k => (nw ? SH[k] : NAME[k])), tr = [], G = {};
  MK.forEach(k => { const S = ser(k); G[k] = { r: gph(S.x, p, k + 'r'), abs: gph(S.abs, p, k + 'abs') }; });
  [['r', RED, 'circle', -0.12, 'returns r'], ['abs', PURPLE, 'square', 0.12, 'absolute returns |r|']].forEach(([y, c, sym, off, nm]) => {
    const d = MK.map(k => G[k][y].d), e = MK.map(k => 1.96 * G[k][y].se);
    tr.push({ x: MK.map((_, j) => j + off), y: d, mode: 'markers', marker: { color: c, symbol: sym, size: 9 }, error_y: { type: 'data', array: e, color: c, thickness: 1.6, width: 5 }, name: `${nm}: GPH d and 95% interval`, customdata: names, hovertemplate: '%{customdata}: d = %{y:.3f}<extra></extra>' });
  });
  tr.push({ x: [-0.5, MK.length - 0.5], y: [0, 0], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'd = 0 (no long memory)', hoverinfo: 'skip' });
  plot('mt-chart', tr, layout({ xaxis: AX('', { tickvals: MK.map((_, j) => j), ticktext: names, range: [-0.5, MK.length - 0.5] }), yaxis: AX(`GPH d, bandwidth m = T^${p}`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const cl = (v, b) => (v > b.q975 ? 'bad' : '');
  table('mt-tab', T.h.mt, MK.map(k => {
    const E = est(k, 'r'), A = est(k, 'abs'), b = mcBand(k, 'dfa'), g = G[k];
    return [{ v: SH[k], c: 'l' }, S_N(k), nf(E.rs, 2), { v: nf(E.dfa, 2), c: cl(E.dfa, b) }, fz(g.r.d, 2), `[${fz(g.r.d - 1.96 * g.r.se, 2)}; ${fz(g.r.d + 1.96 * g.r.se, 2)}]`, nf(A.dfa, 2), fz(g.abs.d, 2), { v: nf(A.lo.V, 2), c: A.lo.reject ? 'bad' : '' }, `[${nf(b.q025, 2)}; ${nf(b.q975, 2)}]`];
  }));
  const ex0 = MK.filter(k => G[k].r.d - 1.96 * G[k].r.se > 0).map(k => SH[k]), da = MK.map(k => G[k].abs.d), ms = MK.map(k => G[k].r.m);
  out('mt-out', [T.mt(nf(p, 1), Math.min(...ms), Math.max(...ms), ex0, nf(Math.min(...da), 2), nf(Math.max(...da), 2), MK.filter(k => est(k, 'abs').lo.reject).length, p !== 0.5)]);
}

// ================================================================= x10. scaling risk with h^H (worked example)
function rkBlock() {
  const H = +$('rk-h').value, h = +kSel('rk-n'), sg = 1.2, z = -normppf(0.01), v1 = z * sg, nw = single('rk-chart', 380), hs = []; label('rk-h').textContent = nf(H, 2);
  for (let j = 1; j <= 250; j++) hs.push(j);
  const tr = [{ x: hs, y: hs.map(j => v1 * j ** H), mode: 'lines', line: { color: RED, width: 2 }, name: `VaR 1% = 2.326 x 1.2 x h^H, H = ${H.toFixed(2)}`, hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    { x: hs, y: hs.map(j => v1 * Math.sqrt(j)), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'square-root-of-time rule, H = 0.5', hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' }];
  ['sp500', 'bet'].forEach(k => { const s = scaling(ser(k).x, D.h_list); tr.push({ x: D.h_list, y: s.sd.map(v => v1 * v / s.sd[0]), mode: 'markers', marker: { color: COL[k], size: 8 }, name: `2.79 x sd(h)/sd(1) of the ${NAME[k]} (data)`, hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' }); });
  tr.push({ x: [h], y: [v1 * h ** H], mode: 'markers', marker: { color: AMBER, size: 15, symbol: 'circle-open', line: { width: 3 } }, name: `h = ${h}`, hoverinfo: 'skip' });
  plot('rk-chart', tr, layout({ xaxis: LOGAX('horizon h in days (log scale)'), yaxis: LOGAX('VaR 1% on h days (%), daily volatility 1.2%'), legend: LEG(nw, -0.18), margin: { l: 60, r: 12, t: 14, b: 40 } }));
  const f = h ** H, sq = Math.sqrt(h), sp = scaling(ser('sp500').x, D.h_list), bt = scaling(ser('bet').x, D.h_list), i10 = D.h_list.indexOf(10);
  steps('rk-we', [T.rk1(nf(z, 3), nf(sg, 1), nf(v1, 2)), T.rk2(h, nf(H, 2), nf(f, 2), nf(v1 * f, 2)), T.rk3(h, nf(sq, 2), nf(v1 * sq, 2), nf(100 * (f / sq - 1), 0)),
    T.rk4(nf(sp.sd[i10] / sp.sd[0], 2), nf(bt.sd[i10] / bt.sd[0], 2), nf(Math.sqrt(10), 2))]);
  out('rk-out', [T.rk(nf(H, 2), h, nf(f, 2), nf(v1 * f, 2), nf(v1 * sq, 2), nf(100 * (f / sq - 1), 0), nf(100 * (250 ** H / Math.sqrt(250) - 1), 0), H !== 0.55 || h !== 10)]);
}

// ================================================================= wiring
const PFX = { scaling: 'sc', 'rs-example': 'rx', fbm: 'fb', fgn: 'fg', arfima: 'af', 'rs-dfa': 'rd', gph: 'gp', lo: 'lo', mc: 'mc', nile: 'nl', spurious: 'su', 'vol-acf': 'va', markets: 'mk', rolling: 'ro',
  'fractal-dim': 'fd', 'rs-hurst': 'rh', 'memory-we': 'mw', 'arfima-we': 'aw', 'dfa-we': 'dw', 'gph-we': 'gw', 'lo-we': 'lw', 'sp-table': 'st', 'mk-table': 'mt', 'risk-we': 'rk' };
const BLOCKS = { scBlock, rxBlock, fbBlock, fgBlock, afBlock, rdBlock, gpBlock, loBlock, mcBlock, nlBlock, suBlock, vaBlock, mkBlock, roBlock, fdBlock, rhBlock, mwBlock, awBlock, dwBlock, gwBlock, lwBlock, stBlock, mtBlock, rkBlock };
const PLAYERS = {};
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
  const PLy = (btn, id, vals, ms) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms));
  const MKR = (cur, L) => (L || MK).filter(k => k !== cur).concat([cur]);
  on(['sc-h'], scBlock); PLy('sc-play', 'sc-h', ['250', '10', '20', '50', '100', '250'], 1100);
  on(['rx-w'], rxBlock); PLy('rx-play', 'rx-w', ['2000-01-04', '2008-10-06', '2020-03-09', 'last', 'lec'], 1300);
  on(['fb-d'], fbBlock); PLy('fb-play', 'fb-d', ['0.2', '0.1', '0.3', '0.4', '0.2'], 1300);
  on(['fg-h'], fgBlock); PLy('fg-play', 'fg-h', ['0.8', '0.6', '0.7', '0.9', '0.8'], 1300);
  on(['af-d'], afBlock); PLy('af-play', 'af-d', ['0.4', '0.1', '0.2', '0.3', '0.45', '0.4'], 1300);
  on(['rd-s'], rdBlock); PLy('rd-play', 'rd-s', MKR('sp500'), 1300);
  on(['gp-s', 'gp-p'], gpBlock); PLy('gp-play', 'gp-p', ['0.5', '0.4', '0.6', '0.7', '0.5'], 1300);
  on(['lo-n'], loBlock); PLy('lo-play', 'lo-n', ['1000', '250', '500', '2000', '1000'], 1300);
  on(['mc-l'], mcBlock); PLy('mc-play', 'mc-l', ['0.95', '0.9', '0.99', '0.95'], 1300);
  on(['nl-b'], nlBlock); PLy('nl-play', 'nl-b', [1898, 1880, 1890, 1910, 1920, 1940, 1960, 1898], 1000);
  on(['su-n'], suBlock); PLy('su-play', 'su-n', ['2000', '500', '1000', '4000', '2000'], 1300);
  on(['va-s', 'va-l'], vaBlock); PLy('va-play', 'va-s', MKR('sp500'), 1300);
  on(['mk-e'], mkBlock); PLy('mk-play', 'mk-e', ['rs', 'gph', 'dfa'], 1400);
  on(['ro-w'], roBlock); PLy('ro-play', 'ro-w', ['1000', '250', '500', '2000', '1000'], 1600);
  on(['fd-h', 'fd-k'], fdBlock); PLy('fd-play', 'fd-h', ['0.5', '0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9', '0.5'], 1000);
  on(['rh-s', 'rh-y', 'rh-n'], rhBlock); PLy('rh-play', 'rh-s', MKR('sp500'), 1300);
  on(['mw-h', 'mw-k'], mwBlock); PLy('mw-play', 'mw-h', [0.8, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 0.8], 900);
  on(['aw-d', 'aw-t'], awBlock); PLy('aw-play', 'aw-d', [0.3, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.3], 900);
  on(['dw-s', 'dw-y', 'dw-n'], dwBlock); PLy('dw-play', 'dw-n', [1, 4, 7, 10, 13, 16, 19, 20, 1], 1000);
  on(['gw-s', 'gw-p', 'gw-y'], gwBlock); PLy('gw-play', 'gw-p', ['0.5', '0.4', '0.6', '0.7', '0.5'], 1300);
  on(['lw-s', 'lw-y'], lwBlock); PLy('lw-play', 'lw-s', MKR('sp500'), 1300);
  on(['st-s'], stBlock); PLy('st-play', 'st-s', MKR('sp500'), 1300);
  on(['mt-p'], mtBlock); PLy('mt-play', 'mt-p', ['0.5', '0.6', '0.4', '0.7', '0.5'], 1400);
  on(['rk-h', 'rk-n'], rkBlock); PLy('rk-play', 'rk-h', [0.55, 0.45, 0.48, 0.5, 0.52, 0.58, 0.6, 0.55], 900);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH11 = { ser, est, blockSizes, rsCurve, hurstRS, dfaCurve, hurstDFA, periodogram, gph, gphOf, loTest, acf, scaling, fgnAcvf, arfAcf, arfVar, rolling, bandOf, boxCount, crossLag, gphCurve, summ, rej, normppf,
  init, PFX, blocks: BLOCKS };
if (window.CH11_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
