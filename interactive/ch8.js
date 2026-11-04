// Chapter 8 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch8_data.js.
(function () {
'use strict';
const D = window.CH8_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, solve, LANG } = window.SFMI;
const T = window.CH8_TEXT;
const NUM = D.N, COL = D.col, NAME = D.name, ECOL = D.est_col, ELAB = D.est_label;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';
const SH = { sp500: 'S&P 500', dax: 'DAX', bet: 'BET', btc: 'Bitcoin', tlv: 'TLV', snp: 'SNP' };
const EST = ['cc', 'park', 'gk', 'rs', 'yz'], ESH = { cc: 'CC', park: 'P', gk: 'GK', rs: 'RS', yz: 'YZ' };
const MK4 = ['sp500', 'dax', 'bet', 'btc'], OHN = D.ohlc_names, RETN = D.ret_names;
const LN2 = Math.log(2);

// ================================================================= special functions
const SQ2PI = Math.sqrt(2 * Math.PI);
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
function gammln(z) {                // Lanczos
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
  for (const c of g) ser += c / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function gammq(a, x) {              // regularized upper incomplete gamma Q(a, x)
  if (x <= 0) return 1;
  if (x < a + 1) { let ap = a, s = 1 / a, del = s; for (let n = 0; n < 500; n++) { ap++; del *= x / ap; s += del; if (Math.abs(del) < Math.abs(s) * 1e-15) break; } return 1 - s * Math.exp(-x + a * Math.log(x) - gammln(a)); }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
  return Math.exp(-x + a * Math.log(x) - gammln(a)) * h;
}
const chi2sf = (x, df) => gammq(df / 2, x / 2);
function chi2ppf(p, df) { let lo = 0, hi = 10 * df + 100; for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (1 - chi2sf(m, df) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
const p2 = z => 2 * (1 - Phi(Math.abs(z)));

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpack(R) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); return dd.map(iso); }
const SER = {};
function ser(k) { return SER[k] || (SER[k] = { x: D.ret[k].x, dates: unpack(D.ret[k]), n: D.ret[k].x.length }); }
const firstFrom = (dates, d) => { const i = dates.findIndex(v => v >= d); return i < 0 ? dates.length : i; };
function fromYear(k, y) { const S = ser(k), i = firstFrom(S.dates, `${y}-01-01`); return { x: S.x.slice(i), dates: S.dates.slice(i), n: S.n - i }; }
const ppy = dates => dates.length / ((Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 864e5 / 365.25);
const yearOf = d => +d.slice(0, 4);

// ================================================================= statistics of the chapter (generate_all_charts.py)
const mean = x => { let s = 0; for (const v of x) s += v; return s / x.length; };
const sdev = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / (x.length - 1)); };
const vari = x => sdev(x) ** 2;
function moments(x) {               // population moments, as scipy.stats (bias = True)
  const m = mean(x), n = x.length; let m2 = 0, m3 = 0, m4 = 0;
  for (const v of x) { const e = v - m, e2 = e * e; m2 += e2; m3 += e2 * e; m4 += e2 * e2; }
  m2 /= n; m3 /= n; m4 /= n; return { m, kurt: m4 / (m2 * m2), exkurt: m4 / (m2 * m2) - 3, skew: m3 / m2 ** 1.5 };
}
function quantile(a, p) { const b = Float64Array.from(a).sort(), h = (b.length - 1) * p, i = Math.floor(h); return i + 1 < b.length ? b[i] + (h - i) * (b[i + 1] - b[i]) : b[i]; }
const median = a => quantile(a, 0.5);
function acf(x, nlags) {            // rho_hat(1..nlags) = sum e_t e_{t-k} / sum e_t^2
  const m = mean(x), n = x.length, e = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { e[i] = x[i] - m; s += e[i] * e[i]; }
  const r = []; for (let k = 1; k <= nlags; k++) { let c = 0; for (let t = k; t < n; t++) c += e[t] * e[t - k]; r.push(c / s); } return r;
}
function ljungBox(x, m) {           // Q(m) = T(T+2) sum rho_k^2/(T-k) ~ chi2(m)
  const T = x.length, rho = acf(x, m), terms = rho.map((r, i) => T * (T + 2) * r * r / (T - i - 1)), q = terms.reduce((s, v) => s + v, 0);
  return { T, m, rho, terms, q, p: chi2sf(q, m), crit: chi2ppf(0.95, m) };
}
function olsCols(y, cols) {         // OLS of y on the columns; coefficients, R^2, residual sum of squares
  const n = y.length, k = cols.length, A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let i = 0; i < k; i++) { const ci = cols[i]; let s = 0; for (let t = 0; t < n; t++) s += ci[t] * y[t]; b[i] = s; for (let j = 0; j <= i; j++) { const cj = cols[j]; let z = 0; for (let t = 0; t < n; t++) z += ci[t] * cj[t]; A[i][j] = A[j][i] = z; } }
  const beta = solve(A, b), my = mean(y); let ssr = 0, sst = 0;
  for (let t = 0; t < n; t++) { let f = 0; for (let i = 0; i < k; i++) f += beta[i] * cols[i][t]; ssr += (y[t] - f) ** 2; sst += (y[t] - my) ** 2; }
  return { beta, r2: 1 - ssr / sst, ssr, n, A };
}
const ols1 = (y, x) => { const o = olsCols(y, [new Float64Array(y.length).fill(1), x]); return { a: o.beta[0], b: o.beta[1], r2: o.r2 }; };
function archLM(x, q) {             // Engle (1982): e_t^2 on a constant and q own lags; LM = n R^2 ~ chi2(q)
  const m = mean(x), e2 = Float64Array.from(x, v => (v - m) ** 2), n = e2.length, y = e2.slice(q), cols = [new Float64Array(n - q).fill(1)];
  for (let j = 1; j <= q; j++) cols.push(e2.slice(q - j, n - j));
  const o = olsCols(y, cols), lm = y.length * o.r2;
  return { lm, r2: o.r2, n: y.length, b: o.beta, p: chi2sf(lm, q), crit: chi2ppf(0.95, q), q };
}
function rollingSd(x, n) {          // sample standard deviation of the last n values (ddof 1); NaN before
  const N = x.length, r = new Float64Array(N).fill(NaN);
  for (let t = n - 1; t < N; t++) { let s = 0, s2 = 0; for (let j = t - n + 1; j <= t; j++) { s += x[j]; s2 += x[j] * x[j]; } r[t] = Math.sqrt(Math.max((s2 - s * s / n) / (n - 1), 0)); }
  return r;
}
function ewmaVar(x, lam, n0 = 30) { // s2[t] = forecast for day t (t = 0..T): lam s2[t-1] + (1 - lam) r[t-1]^2, started with the mean of the first 30 r^2
  const n = x.length, s2 = new Float64Array(n + 1); let m = 0; for (let i = 0; i < n0; i++) m += x[i] * x[i]; s2[0] = m / n0;
  for (let t = 0; t < n; t++) s2[t + 1] = lam * s2[t] + (1 - lam) * x[t] * x[t];
  return s2;
}
const sdChange = v => { const d = []; for (let i = 1; i < v.length; i++) d.push(Math.log(v[i]) - Math.log(v[i - 1])); const m = mean(d); return Math.sqrt(mean(d.map(z => (z - m) ** 2))); };

// ---------------------------------------------------------------- OHLC and the range-based estimators
const OH = {};
function parts(k) {                 // range_parts: o, u, d, c, cc (log, in %) and the one-day estimators
  if (OH[k]) return OH[k];
  const R = D.ohlc[k], dates = unpack(R), n = dates.length - 1, L = a => a.map(v => 100 * Math.log(v));
  const lo = L(R.o), lh = L(R.h), ll = L(R.l), lc = L(R.c), P = { dates: dates.slice(1), n, O: R.o.slice(1), H: R.h.slice(1), Lw: R.l.slice(1), Cl: R.c.slice(1), Cp: R.c.slice(0, n) };
  ['o', 'u', 'd', 'c', 'cc', 'park', 'gk', 'rs'].forEach(f => { P[f] = new Float64Array(n); });
  for (let i = 0; i < n; i++) {
    const o = lo[i + 1] - lc[i], u = lh[i + 1] - lo[i + 1], d = ll[i + 1] - lo[i + 1], c = lc[i + 1] - lo[i + 1];
    P.o[i] = o; P.u[i] = u; P.d[i] = d; P.c[i] = c; P.cc[i] = lc[i + 1] - lc[i];
    P.park[i] = (u - d) ** 2 / (4 * LN2); P.gk[i] = 0.5 * (u - d) ** 2 - (2 * LN2 - 1) * c * c; P.rs[i] = u * (u - c) + d * (d - c);
  }
  P.idx = new Map(P.dates.map((d, i) => [d, i]));
  return (OH[k] = P);
}
const yzK = n => 0.34 / (1.34 + (n + 1) / (n - 1));
function windowEst(P, i0, i1) {     // window_estimators on days i0..i1-1
  const sl = f => P[f].subarray(i0, i1), n = i1 - i0, k = yzK(n), so = vari(sl('o')), sc = vari(sl('c')), rs = mean(sl('rs'));
  return { cc: vari(sl('cc')), park: mean(sl('park')), gk: mean(sl('gk')), rs, yz: so + k * sc + (1 - k) * rs, so, sc, k, n };
}
function rollingEst(P, n) {         // rolling n-day estimates (%^2 per day), from day n-1
  const out = { dates: P.dates.slice(n - 1) }; EST.forEach(e => { out[e] = new Float64Array(P.n - n + 1); });
  for (let t = n - 1; t < P.n; t++) { const w = windowEst(P, t - n + 1, t + 1); EST.forEach(e => { out[e][t - n + 1] = w[e]; }); }
  return out;
}

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const fnum = (x, d) => { const s = nf(x, d), i = s.search(/[.,]/), a = i < 0 ? s : s.slice(0, i), b = i < 0 ? '' : s.slice(i); return a.replace(/\B(?=(\d{3})+(?!\d))/g, sep) + b; };
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (Math.abs(x) >= 1e-3) return fnum(x, Math.max(0, d - 1 - e));
  let m = x / 10 ** e, ee = e; if (+Math.abs(m).toFixed(d - 1) >= 10) { m /= 10; ee++; }
  return nf(m, d - 1) + ' × 10<sup>' + (ee < 0 ? '−' : '') + Math.abs(ee) + '</sup>';
}
const fp = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const fpe = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : '= ' + nf(p, 3));
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
const MON = { ro: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
  en: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] };
const fdate = d => `${+d.slice(8, 10)} ${MON[LANG][+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.span ? ` colspan="${o.span}"` : ''}>${o.v}</${tag}>`; };
  const hs = Array.isArray(head[0]) ? head : [head];
  $(id).innerHTML = `<table class="xt"><thead>${hs.map(h => `<tr>${h.map(x => cell(x, 'th')).join('')}</tr>`).join('')}</thead><tbody>` +
    rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.stack || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 270) * n + (o.extra || 110) : (o.h || 380)) + 'px';
  const g = stack ? (o.vgap || 0.2) : (o.gap || 0.09), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', dom = [i * (w + g), i * (w + g) + w];
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [1 - dom[1], 1 - dom[0]] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const ZERO = (yref) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: yref || 'y', y0: 0, y1: 0, line: { color: '#000', width: 0.6 } });
const LOGT = { type: 'log', tickvals: [0.1, 1, 10, 100, 1000, 10000], ticktext: ['0.1', '1', '10', '100', '1,000', '10,000'] };

// ================================================================= 1. four markets, one pattern (returns in bursts)
function rbBlock() {
  const k = +$('rb-k').value, nw = narrow('rb-chart'); label('rb-k').textContent = String(k);
  $('rb-chart').style.height = (nw ? 620 : 560) + 'px';
  const L = {}, tr = [], items = [], g = 0.045, h = (1 - 3 * g) / 4;
  MK4.forEach((n, i) => {
    const S = ser(n), s = i ? String(i + 1) : '', sd = sdev(S.x), lim = Math.ceil(quantile(S.x.map(Math.abs), 0.999));
    L['xaxis' + s] = AX(i === 3 ? 'Date' : '', { anchor: 'y' + s, type: 'date', range: ['1989-07-01', '2027-01-01'], showticklabels: i === 3 });
    if (i) L['xaxis' + s].matches = 'x';
    L['yaxis' + s] = AX('%', { domain: [1 - (i * (h + g) + h), 1 - i * (h + g)], anchor: 'x' + s, range: [-lim, lim] });
    tr.push({ x: S.dates, y: S.x, mode: 'lines', line: { color: COL[n], width: 0.6 }, name: NAME[n], xaxis: 'x' + s, yaxis: 'y' + s, hovertemplate: `${NAME[n]}<br>%{x}<br>%{y:.2f}%<extra></extra>` });
    const ex = []; S.x.forEach((v, j) => { if (Math.abs(v) > k * sd) ex.push(j); });
    tr.push({ x: ex.map(j => S.dates[j]), y: ex.map(j => Math.max(-lim, Math.min(lim, S.x[j]))), mode: 'markers', marker: { color: DARK, size: 6, symbol: 'circle-open', line: { width: 1.3 } }, name: `|return| above ${k} standard deviations`,
      showlegend: i === 0, legendgroup: 'ex', xaxis: 'x' + s, yaxis: 'y' + s, customdata: ex.map(j => S.x[j] / sd), hovertemplate: `${NAME[n]}<br>%{x}<br>%{customdata:.1f} sd<extra></extra>` });
    items.push([SH[n], ex.length, sig(S.n * 2 * (1 - Phi(k)), 2), nf(Math.max(...S.x.map(Math.abs)) / sd, 1)]);
  });
  plot('rb-chart', tr, layout(Object.assign(L, { legend: LEG(nw, -0.09), margin: { l: 50, r: 10, t: 10, b: 30 } })));
  out('rb-out', [T.rb(k, items)]);
}

// ================================================================= 2. interpretation: volatility in four markets (table)
function burst(n, y) {
  const S = y ? fromYear(n, y) : ser(n), sd = sdev(S.x), A = ppy(S.dates), M = moments(S.x); let j = 0;
  S.x.forEach((v, i) => { if (Math.abs(v) > Math.abs(S.x[j])) j = i; });
  return { n: S.n, first: S.dates[0], A, sd, vol: sd * Math.sqrt(A), maxv: S.x[j], maxd: S.dates[j], ex: M.exkurt, sds: Math.abs(S.x[j]) / sd };
}
function vtBlock() {
  const y = +$('vt-y').value, P = panels('vt-chart', 2, { h: 330 }), [a, b] = P.ax; label('vt-y').textContent = String(y);
  const B = MK4.map(n => burst(n, y)), xs = MK4.map(n => NAME[n]);
  const tr = [on2(a, { type: 'bar', x: xs, y: B.map(o => o.vol), marker: { color: MK4.map(n => COL[n]) }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }),
    on2(b, { type: 'bar', x: xs, y: B.map(o => o.ex), marker: { color: MK4.map(n => hexA(COL[n], 0.6)) }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' })];
  setAx(P, 0, AX(''), AX('Annualised volatility (%)', { rangemode: 'tozero' })); setAx(P, 1, AX(''), AX('Excess kurtosis', { rangemode: 'tozero' }));
  plot('vt-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(a, `Volatility, samples from ${y}`), ptitle(b, 'Excess kurtosis (Normal: 0)')], margin: { l: 56, r: 12, t: 26, b: 30 } })));
  table('vt-tab', ['', T.h.from, 'T', 'A', 'σ̂ (%)', T.h.annvol, T.h.maxmove, T.h.date, T.h.insd, T.h.exk], B.map((o, i) => [SH[MK4[i]], o.first, fint(o.n), nf(o.A, 1), nf(o.sd, 4), nf(o.vol, 1), nf(o.maxv, 1), o.maxd, nf(o.sds, 1), nf(o.ex, 1)]));
  out('vt-out', [T.vt(y, nf(Math.min(...B.map(o => o.sds)), 1), nf(Math.max(...B.map(o => o.sds)), 1), B.map((o, i) => [SH[MK4[i]], nf(o.vol, 1), nf(o.ex, 1)]))]);
}

// ================================================================= 3. worked example: annualising volatility
const CONV = { act: null, 252: 252, 250: 250, 260: 260, 365: 365 };
function anBlock() {
  const cv = $('an-c').value, nw = single('an-chart', 340), B = MK4.map(n => burst(n)), Ac = o => (cv === 'act' ? o.A : +cv);
  const xs = MK4.map(n => NAME[n]);
  plot('an-chart', [{ type: 'bar', x: xs, y: B.map(o => o.vol), name: 'sigma x sqrt(A), A = actual observations per year', marker: { color: BLUE }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { type: 'bar', x: xs, y: B.map(o => o.sd * Math.sqrt(Ac(o))), name: cv === 'act' ? 'Chosen convention: actual frequency' : `Chosen convention: A = ${cv} for every series`, marker: { color: cv === 'act' ? hexA(BLUE, 0.45) : RED }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('Annualised volatility (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const kk = $('an-s').value, o = B[MK4.indexOf(kk)], Aw = cv === 'act' ? (kk === 'btc' ? 252 : 365) : +cv;
  steps('an-we', [T.an1(SH[kk], fint(o.n), nf((Date.parse(ser(kk).dates[ser(kk).n - 1]) - Date.parse(o.first)) / 864e5 / 365.25, 2), nf(o.A, 1), nf(o.sd, 4)),
    T.an2(nf(o.sd, 4), nf(o.A, 1), nf(Math.sqrt(o.A), 3), nf(o.vol, 1)),
    T.an3(Aw, nf(o.sd, 4), nf(Math.sqrt(Aw), 3), nf(o.sd * Math.sqrt(Aw), 1), nf(100 * (Math.sqrt(Aw / o.A) - 1), 1))]);
  out('an-out', [T.an(cv === 'act' ? null : cv, B.map((q, i) => [SH[MK4[i]], nf(q.vol, 1), nf(q.sd * Math.sqrt(Ac(q)), 1)]))]);
}

// ================================================================= 4. S&P 500: three windows, three answers
const RS = {};
const rsdC = (k, n) => { const key = k + n; return RS[key] || (RS[key] = rollingSd(ser(k).x, n)); };
function winStats(k, n) {
  const S = ser(k), a = ppy(S.dates), v = rsdC(k, n), d = [], y = [];
  for (let t = n - 1; t < S.n; t++) { d.push(S.dates[t]); y.push(v[t] * Math.sqrt(a)); }
  let im = 0; y.forEach((z, i) => { if (z > y[im]) im = i; });
  return { d, y, max: y[im], maxd: d[im], sdc: sdChange(y), last: y[y.length - 1], mean: mean(y) };
}
function rwBlock() {
  const n = +$('rw-n').value, nw = single('rw-chart', 380), S = ser('sp500'), a = ppy(S.dates), whole = sdev(S.x) * Math.sqrt(a); label('rw-n').textContent = String(n);
  const W = [[n, RED, 0.9], [63, BLUE, 1.4], [252, GREEN, 1.4]].map(([m, c, w]) => ({ m, c, w, s: winStats('sp500', m) }));
  const tr = W.map(o => ({ x: o.s.d, y: o.s.y, mode: 'lines', line: { color: o.c, width: o.w }, name: `${o.m}-day window`, hovertemplate: `${o.m} days<br>%{x}<br>%{y:.1f}%<extra></extra>` }));
  tr.push({ x: [S.dates[0], S.dates[S.n - 1]], y: [whole, whole], mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: 'Whole-sample volatility', hoverinfo: 'skip' });
  plot('rw-chart', tr, layout({ xaxis: AX('Date'), yaxis: AX('Annualised volatility (%)'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('rw-out', [T.rw(W.map(o => [o.m, nf(o.s.max, 1), fdate(o.s.maxd), nf(100 * o.s.sdc, 1), nf(o.s.last, 1)]), nf(whole, 1))]);
}

// ================================================================= 5. how precise is a sample volatility? (worked example)
function prBlock() {
  const n = +$('pr-n').value, k = $('pr-s').value, nw = single('pr-chart', 360), S = ser(k), a = ppy(S.dates), kap = moments(S.x).kurt; label('pr-n').textContent = String(n);
  const v = rsdC(k, n), i0 = firstFrom(S.dates, '2024-09-18'), d = S.dates.slice(i0), y = Array.from(v.subarray(i0), z => z * Math.sqrt(a));
  const sN = 1 / Math.sqrt(2 * n), sH = Math.sqrt((kap - 1) / (4 * n)), band = (s, sg) => y.map(z => z * (1 + sg * 1.96 * s));
  const tr = [{ x: d.concat([...d].reverse()), y: band(sH, 1).concat(band(sH, -1).reverse()), fill: 'toself', fillcolor: hexA(ORANGE, 0.18), line: { width: 0 }, name: 'Approx. 95% interval, observed kurtosis', hoverinfo: 'skip' },
    { x: d.concat([...d].reverse()), y: band(sN, 1).concat(band(sN, -1).reverse()), fill: 'toself', fillcolor: hexA(BLUE, 0.22), line: { width: 0 }, name: 'Approx. 95% interval, Normal returns', hoverinfo: 'skip' },
    { x: d, y, mode: 'lines', line: { color: COL[k], width: 1.6 }, name: `${NAME[k]}: ${n}-day volatility`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }];
  plot('pr-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Annualised volatility (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const last = y[y.length - 1];
  steps('pr-we', [T.pr1(SH[k], nf(kap, 1), n, nf(1 / Math.sqrt(2 * n), 4), nf(100 * sN, 1)), T.pr2(nf(kap, 1), n, nf(100 * sH, 1)),
    T.pr3(n, nf(last, 1), nf(last * (1 - 1.96 * sN), 1), nf(last * (1 + 1.96 * sN), 1), nf(Math.max(0, last * (1 - 1.96 * sH)), 1), nf(last * (1 + 1.96 * sH), 1))]);
  out('pr-out', [T.pr(SH[k], n, nf(100 * sN, 1), nf(100 * sH, 1), nf(sH / sN, 2))]);
}

// ================================================================= 6. EWMA weights against a rolling window
const hl = l => Math.log(0.5) / Math.log(l), d99 = l => Math.log(0.01) / Math.log(l);
function ewBlock() {
  const l = +$('ew-l').value, n = +$('ew-n').value, nw = single('ew-chart', 360), jm = Math.max(120, n + 20), j = range(1, jm, 1); label('ew-l').textContent = nf(l, 2); label('ew-n').textContent = String(n);
  const tr = [{ x: j, y: j.map(i => (1 - l) * l ** (i - 1)), mode: 'lines', line: { color: RED, width: 1.8 }, name: `EWMA, lambda = ${l.toFixed(2)}`, hovertemplate: 'j = %{x}<br>%{y:.4f}<extra></extra>' },
    { x: j, y: j.map(i => 0.03 * 0.97 ** (i - 1)), mode: 'lines', line: { color: BLUE, width: 1.8 }, name: 'EWMA, lambda = 0.97', hovertemplate: 'j = %{x}<br>%{y:.4f}<extra></extra>' },
    { x: j, y: j.map(i => (i <= n ? 1 / n : 0)), mode: 'lines', line: { color: GREEN, width: 1.6, shape: 'hvh' }, name: `Equal weights, ${n}-day window`, hovertemplate: 'j = %{x}<br>%{y:.4f}<extra></extra>' }];
  plot('ew-chart', tr, layout({ xaxis: AX('Days in the past (j)'), yaxis: AX('Weight of the squared return'), legend: LEG(nw, -0.16), margin: { l: 60, r: 12, t: 14, b: 40 } }));
  out('ew-out', [T.ew(nf(l, 2), nf(hl(l), 1), nf(1 / (1 - l), 1), nf(d99(l), 1), nf(1 - l, 2), n, nf(1 / n, 4), nf(d99(0.97), 1))]);
}

// ================================================================= 7. worked example: one EWMA update
function euBlock() {
  const l = +$('eu-l').value, r = +$('eu-r').value, nw = single('eu-chart', 340), h = range(0, 100, 1), W = 63; label('eu-l').textContent = nf(l, 2); label('eu-r').textContent = nf(r, 1);
  const ew = h.map(i => (i === 0 ? 1 : 1 + (r * r - 1) * (1 - l) * l ** (i - 1))), win = h.map(i => (i >= 1 && i <= W ? 1 + (r * r - 1) / W : 1));
  plot('eu-chart', [{ x: h, y: ew.map(Math.sqrt), mode: 'lines', line: { color: RED, width: 1.8 }, name: `EWMA, lambda = ${l.toFixed(2)}`, hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' },
    { x: h, y: win.map(Math.sqrt), mode: 'lines', line: { color: BLUE, width: 1.8, shape: 'hv' }, name: '63-day window', hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' }],
  layout({ xaxis: AX('Days after the shock (all other days: r^2 = 1)'), yaxis: AX('Daily volatility (%)'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ex = $('eu-x').value;
  let s0 = 1, rr = r, lab = null;
  if (ex === 'sp') { const S = ser('sp500'), i = S.dates.indexOf('2020-03-16'), s2 = ewmaVar(S.x, l); s0 = s2[i]; rr = S.x[i]; lab = fdate(S.dates[i]); }
  const a1 = l * s0, a2 = (1 - l) * rr * rr, s1 = a1 + a2;
  steps('eu-we', [T.eu1(lab, nf(s0, 2), nf(Math.sqrt(s0), 3), nf(rr, ex === 'sp' ? 3 : 1)), T.eu2(nf(l, 2), nf(s0, 2), nf(a1, 2)), T.eu3(nf(1 - l, 2), nf(rr * rr, ex === 'sp' ? 2 : 0), nf(a2, 2)),
    T.eu4(nf(a1, 2), nf(a2, 2), nf(s1, 2), nf(Math.sqrt(s1), 3), nf(Math.sqrt(s1) * Math.sqrt(252), 1)), T.eu5(nf(l, 2), nf(s1, 2), nf(l * s1, 2), nf(rr * rr, ex === 'sp' ? 2 : 0), nf(rr * rr / W, 2))]);
  out('eu-out', [T.eu(nf(l, 2), nf(r, 1), nf(Math.sqrt(ew[1]), 3), nf(Math.sqrt(win[1]), 3), nf(Math.sqrt(ew[W + 1]), 3), W)]);
}

// ================================================================= 8. the ghost effect
function ghost(n, l) {
  const S = ser('sp500'), a = ppy(S.dates), rv = rsdC('sp500', n), s2 = ewmaVar(S.x, l), ew = Array.from(s2.subarray(1), v => Math.sqrt(v * a)), roll = Array.from(rv, v => v * Math.sqrt(a));
  const i0 = firstFrom(S.dates, '2020-04-01'), i1 = firstFrom(S.dates, '2020-10-01'); let j = i0 + 1;
  for (let i = i0 + 1; i < i1; i++) if (roll[i] - roll[i - 1] < roll[j] - roll[j - 1]) j = i;
  const y0 = firstFrom(S.dates, '2020-01-01'), y1 = firstFrom(S.dates, '2021-01-01'); let pr = y0, pe = y0;
  for (let i = y0; i < y1; i++) { if (roll[i] > roll[pr]) pr = i; if (ew[i] > ew[pe]) pe = i; }
  return { S, a, roll, ew, drop: j, before: roll[j - 1], after: roll[j], left: j - n, ewd: ew[j] - ew[j - 1], pr, pe };
}
function ghBlock() {
  const n = +$('gh-n').value, l = +$('gh-l').value, nw = single('gh-chart', 380), G = ghost(n, l), S = G.S; label('gh-n').textContent = String(n);
  const i0 = firstFrom(S.dates, '2019-10-01'), i1 = firstFrom(S.dates, '2021-04-01'), d = S.dates.slice(i0, i1);
  plot('gh-chart', [{ type: 'bar', x: d, y: S.x.slice(i0, i1).map(v => Math.abs(v) * Math.sqrt(G.a)), marker: { color: hexA(AMBER, 0.55) }, name: '|return|, annualised', hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' },
    { x: d, y: G.roll.slice(i0, i1), mode: 'lines', line: { color: BLUE, width: 1.8 }, name: `${n}-day rolling volatility`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' },
    { x: d, y: G.ew.slice(i0, i1), mode: 'lines', line: { color: RED, width: 1.8 }, name: `EWMA volatility, lambda = ${l}`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }],
  layout({ bargap: 0, xaxis: AX('Date'), yaxis: AX('Annualised volatility (%)'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('gh-out', [T.gh(n, fdate(S.dates[G.drop]), nf(G.before, 1), nf(G.after, 1), fdate(S.dates[G.left]), nf(S.x[G.left], 1), nf(l, 2), nf(G.ewd, 1), nf(G.ew[G.pe], 1), fdate(S.dates[G.pe]), nf(G.roll[G.pr], 1), fdate(S.dates[G.pr]))]);
}

// ================================================================= 9. one trading day in OHLC data (simulated)
function opBlock() {
  const seed = $('op-s').value, o = +$('op-o').value, nw = single('op-chart', 360), x = D.path[seed].map(v => v - 0.35), p = x.map(v => o + v), m = range(0, 390, 1); label('op-o').textContent = nf(o, 2);
  let ih = 0, il = 0; p.forEach((v, i) => { if (v > p[ih]) ih = i; if (v < p[il]) il = i; });
  const u = p[ih] - o, d = p[il] - o, c = p[390] - o, lo = Math.min(Math.min(...p), 0) - 0.45;
  const pt = (i, col, nm) => ({ x: [i], y: [p[i]], mode: 'markers', marker: { color: col, size: 10 }, name: nm, hovertemplate: `${nm}: %{y:.3f}%<extra></extra>` });
  plot('op-chart', [{ x: m, y: p, mode: 'lines', line: { color: BLUE, width: 1.2 }, name: 'Log price during the day (%)', hovertemplate: 'minute %{x}<br>%{y:.3f}%<extra></extra>' },
    pt(0, GREEN, 'Open O'), pt(ih, RED, 'High H'), pt(il, ORANGE, 'Low L'), pt(390, PURPLE, 'Close C'),
    { x: [0, 390], y: [0, 0], mode: 'lines', line: { color: '#000', width: 0.7, dash: 'dot' }, name: 'Previous close', hoverinfo: 'skip' }],
  layout({ xaxis: AX('Minutes since the open'), yaxis: AX('ln P - ln C(t-1) (%)', { range: [lo, Math.max(...p) + 0.15] }), legend: LEG(nw, -0.16), margin: { l: 60, r: 12, t: 14, b: 40 } }));
  const r = o + c, P = (u - d) ** 2 / (4 * LN2), GK = 0.5 * (u - d) ** 2 - (2 * LN2 - 1) * c * c, RSv = u * (u - c) + d * (d - c);
  out('op-out', [T.op(seed, nf(o, 2), nf(u, 3), nf(d, 3), nf(c, 3), nf(r, 3), nf(r * r, 3), nf(u - d, 3), nf(P, 3), nf(GK, 3), nf(RSv, 3))]);
}

// ================================================================= 10. Parkinson from H and L (worked example)
function yearRange(P, y) { const a = firstFrom(P.dates, `${y}-01-01`), b = firstFrom(P.dates, `${y + 1}-01-01`); return [a, b]; }
const clampYear = (P, y) => Math.max(yearOf(P.dates[0]), Math.min(y, yearOf(P.dates[P.n - 1])));
function pkBlock() {
  const k = $('pk-s').value, P = parts(k), y = clampYear(P, +$('pk-y').value), nw = single('pk-chart', 360), [a, b] = yearRange(P, y); label('pk-y').textContent = String(y);
  const d = P.dates.slice(a, b), sp = Array.from(P.park.subarray(a, b), Math.sqrt), ab = Array.from(P.cc.subarray(a, b), Math.abs);
  let j = a; for (let i = a; i < b; i++) if (P.u[i] - P.d[i] > P.u[j] - P.d[j]) j = i;
  plot('pk-chart', [{ x: d, y: ab, mode: 'markers', marker: { color: hexA(BLUE, 0.7), size: 4 }, name: '|close-to-close return|', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: d, y: sp, mode: 'lines', line: { color: RED, width: 1.2 }, name: 'sqrt(Parkinson), one day', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }],
  layout({ xaxis: AX(''), yaxis: AX('Daily volatility (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const rg = P.u[j] - P.d[j];
  steps('pk-we', [T.pk1(SH[k], fdate(P.dates[j]), fnum(P.H[j], 2), fnum(P.Lw[j], 2), nf(rg, 3)), T.pk2(nf(rg, 3), nf(rg * rg, 2), nf(4 * LN2, 4), nf(P.park[j], 2), nf(Math.sqrt(P.park[j]), 2)),
    T.pk3(nf(P.cc[j], 2), nf(P.cc[j] ** 2, 2), nf(P.park[j], 2))]);
  const mP = mean(P.park.subarray(a, b)), mC = vari(P.cc.subarray(a, b));
  out('pk-out', [T.pk(SH[k], y, b - a, nf(Math.sqrt(mP), 2), nf(Math.sqrt(mC), 2), nf(sdev(sp.map((v, i) => v - ab[i])) , 2))]);
}

// ================================================================= 11. Garman-Klass and Rogers-Satchell (worked example)
function grBlock() {
  const k = $('gr-s').value, P = parts(k), y = clampYear(P, +$('gr-y').value), nw = single('gr-chart', 360), [a, b] = yearRange(P, y), A = ppy(P.dates); label('gr-y').textContent = String(y);
  const R = rollingEst(P, 21), r0 = Math.max(0, firstFrom(R.dates, `${y}-01-01`)), r1 = firstFrom(R.dates, `${y + 1}-01-01`), d = R.dates.slice(r0, r1);
  const tr = ['park', 'gk', 'rs'].map(e => ({ x: d, y: Array.from(R[e].subarray(r0, r1), v => Math.sqrt(Math.max(v, 0) * A)), mode: 'lines', line: { color: ECOL[e], width: e === 'rs' ? 1.8 : 1.2 }, name: `${ELAB[e]}, 21 days`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }));
  plot('gr-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Annualised volatility (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  let j = a; for (let i = a; i < b; i++) if (P.u[i] - P.d[i] > P.u[j] - P.d[j]) j = i;
  const u = P.u[j], dd = P.d[j], c = P.c[j], dr = mean(P.c.subarray(a, b));
  steps('gr-we', [T.gr1(SH[k], fdate(P.dates[j]), nf(u, 3), nf(dd, 3), nf(c, 3)), T.gr2(nf(u - dd, 3), nf(0.5 * (u - dd) ** 2, 2), nf(2 * LN2 - 1, 4), nf(c * c, 2), nf(P.gk[j], 2)),
    T.gr3(nf(u, 3), nf(u - c, 3), nf(dd, 3), nf(dd - c, 3), nf(u * (u - c), 2), nf(dd * (dd - c), 2), nf(P.rs[j], 2)), T.gr4(nf(P.park[j], 2), nf(P.gk[j], 2), nf(P.rs[j], 2))]);
  const W = windowEst(P, a, b);
  out('gr-out', [T.gr(SH[k], y, nf(dr, 3), nf(Math.sqrt(W.park * A), 1), nf(Math.sqrt(W.gk * A), 1), nf(Math.sqrt(W.rs * A), 1))]);
}

// ================================================================= 12. Yang-Zhang: adding the night (worked example)
const YZ_END = '2020-03-31';
function yzWin(k, n) { const P = parts(k), i1 = firstFrom(P.dates, '2020-04-01'); return { P, w: windowEst(P, i1 - n, i1), from: P.dates[i1 - n], to: P.dates[i1 - 1] }; }
function yzBlock() {
  const n = +$('yz-n').value, nw = single('yz-chart', 360); label('yz-n').textContent = String(n);
  const W = OHN.map(k => yzWin(k, n)), xs = OHN.map(k => NAME[k]), k0 = yzK(n);
  plot('yz-chart', [{ type: 'bar', x: xs, y: W.map(o => o.w.so), name: 'Overnight variance s2_O', marker: { color: BLUE }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { type: 'bar', x: xs, y: W.map(o => k0 * o.w.sc), name: `k x open-to-close variance s2_C (k = ${k0.toFixed(3)})`, marker: { color: ORANGE }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { type: 'bar', x: xs, y: W.map(o => (1 - k0) * o.w.rs), name: '(1 - k) x Rogers-Satchell', marker: { color: PURPLE }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: xs, y: W.map(o => o.w.cc), mode: 'markers', marker: { color: '#000', symbol: 'diamond', size: 9 }, name: 'Close-to-close variance', hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }],
  layout({ barmode: 'stack', xaxis: AX(''), yaxis: AX(`Daily variance (%^2), ${n} days to ${YZ_END}`, { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 60, r: 12, t: 14, b: 30 } }));
  const kk = $('yz-s').value, o = W[OHN.indexOf(kk)], w = o.w, A = ppy(o.P.dates);
  steps('yz-we', [T.yz1(n, n + 1, n - 1, nf((n + 1) / (n - 1), 3), nf(1.34 + (n + 1) / (n - 1), 3), nf(k0, 3)), T.yz2(SH[kk], o.from, o.to, nf(w.so, 3), nf(w.sc, 3), nf(w.rs, 3)),
    T.yz3(nf(w.so, 3), nf(k0, 3), nf(w.sc, 3), nf(1 - k0, 3), nf(w.rs, 3), nf(w.yz, 3)), T.yz4(nf(Math.sqrt(w.yz * A), 1), nf(Math.sqrt(w.rs * A), 1), nf(Math.sqrt(w.cc * A), 1), nf(A, 1))]);
  out('yz-out', [T.yz(n, nf(k0, 3), W.map((q, i) => [SH[OHN[i]], pc(q.w.so / q.w.yz, 0), nf(Math.sqrt(q.w.yz / q.w.cc), 2)]))]);
}

// ================================================================= 13. worked example: one day, six estimates
function bigDays(k) { const P = parts(k), idx = Array.from({ length: P.n }, (_, i) => i).sort((a, b) => Math.abs(P.cc[b]) - Math.abs(P.cc[a])).slice(0, 10); return idx.sort((a, b) => a - b).map(i => P.dates[i]); }
function msDays() {
  const k = $('ms-s').value, sel = $('ms-d'), cur = sel.value, ds = bigDays(k);
  sel.innerHTML = ds.map(d => `<option value="${d}">${d}</option>`).join('');
  sel.value = ds.includes(cur) ? cur : (ds.includes('2020-03-16') ? '2020-03-16' : ds[ds.length - 1]);
  if (PLAYERS['ms-play']) PLAYERS['ms-play'].values = ds;
}
function msBlock() {
  const k = $('ms-s').value, P = parts(k), day = $('ms-d').value, i = P.idx.get(day), nw = single('ms-chart', 340);
  const o = P.o[i], u = P.u[i], d = P.d[i], c = P.c[i], r = P.cc[i], E = [['r^2 (close to close)', r * r, BLUE], ['o^2 (overnight)', o * o, TEAL], ['Parkinson', P.park[i], ECOL.park], ['Garman-Klass', P.gk[i], ECOL.gk],
    ['Rogers-Satchell', P.rs[i], ECOL.rs], ['o^2 + Rogers-Satchell', o * o + P.rs[i], ECOL.yz]];
  plot('ms-chart', [{ type: 'bar', x: E.map(e => e[0]), y: E.map(e => e[1]), marker: { color: E.map(e => e[2]) }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' }],
    layout({ xaxis: AX(''), yaxis: AX(`One-day variance (%^2), ${NAME[k]}, ${day}`, { rangemode: 'tozero' }), margin: { l: 60, r: 12, t: 14, b: nw ? 90 : 60 } }));
  const dec = k === 'btc' ? 2 : (P.Cl[i] < 50 ? 4 : 2);
  steps('ms-we', [T.ms1(fdate(day), fnum(P.Cp[i], dec), fnum(P.O[i], dec), fnum(P.H[i], dec), fnum(P.Lw[i], dec), fnum(P.Cl[i], dec)),
    T.ms2(nf(o, 3), nf(u, 3), nf(d, 3), nf(c, 3), nf(r, 3)), T.ms3(nf(r * r, 1), nf(u - d, 3), nf(4 * LN2, 3), nf(P.park[i], 1), nf(P.gk[i], 1), nf(P.rs[i], 1)),
    T.ms4(nf(o * o, 1), nf(o * o + P.rs[i], 1), pc(o * o / (r * r), 0))]);
  out('ms-out', [T.ms(SH[k], fdate(day), nf(r, 2), nf(o, 2), nf(r * r, 1), nf(P.park[i], 1), nf(o * o + P.rs[i], 1))]);
}

// ================================================================= 14. efficiency by simulation
function effStats(n, sc) {
  const Dd = D.eff[String(n)][sc], cc = Dd.cc, vcc = popVar(cc), mcc = mean(cc.map(v => (v - 1) ** 2)), o = {};
  EST.forEach(e => { const v = Dd[e]; o[e] = { bias: mean(v) - 1, eff: vcc / popVar(v), mse: mcc / mean(v.map(z => (z - 1) ** 2)) }; });
  return o;
}
function popVar(v) { const m = mean(v); return mean(v.map(z => (z - m) ** 2)); }
function efBlock() {
  const n = $('ef-n').value, sc = $('ef-s').value, P = panels('ef-chart', 2, { h: 380, hs: 300 }), true_ = Math.sqrt(252), tr = [];
  const TT = { continuous: 'Continuous trading, no overnight jump', overnight: 'Overnight jump: 20% of the variance', discrete: 'Discrete trading: 26 prices a day' };
  ['continuous', sc].forEach((s, pi) => {
    const a = P.ax[pi];
    EST.forEach(e => tr.push(on2(a, { type: 'box', y: D.eff[n][s][e].map(v => Math.sqrt(Math.max(v, 0) * 252)), name: ELAB[e], marker: { color: ECOL[e], opacity: 0, size: 1 }, fillcolor: hexA(ECOL[e], 0.55), line: { color: ECOL[e] }, boxpoints: 'outliers', hoveron: 'boxes', legendgroup: e, showlegend: pi === 0, x0: ESH[e] })));
    tr.push(on2(a, { x: ['CC', 'YZ'], y: [true_, true_], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: `True volatility (${true_.toFixed(1)}%)`, showlegend: pi === 0, hoverinfo: 'skip' }));
  });
  setAx(P, 0, AX(''), AX('Annualised volatility estimate (%)')); setAx(P, 1, AX(''), AX(''));
  P.L.yaxis2.matches = 'y';
  plot('ef-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], TT.continuous), ptitle(P.ax[1], TT[sc])], legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  const A = effStats(n, 'continuous'), B = effStats(n, sc);
  out('ef-out', [T.ef(+n, EST.map(e => [ESH[e], nf(A[e].eff, 1), nf(Math.round(100 * B[e].bias) + 0, 0)]), sc, +n !== 21)]);
}

// ================================================================= 15. interpretation: precision is not accuracy (table)
function etBlock() {
  const n = $('et-n').value, P = panels('et-chart', 2, { h: 340 }), [a, b] = P.ax, S = { continuous: effStats(n, 'continuous'), overnight: effStats(n, 'overnight'), discrete: effStats(n, 'discrete') }, xs = EST.map(e => ESH[e]);
  const SC = [['continuous', BLUE, 'Continuous trading'], ['overnight', RED, 'Overnight jump'], ['discrete', ORANGE, '26 prices a day']];
  const tr = [];
  SC.forEach(([s, c, nm]) => {
    tr.push(on2(a, { type: 'bar', x: xs, y: EST.map(e => S[s][e].mse), marker: { color: c }, name: nm, legendgroup: s, hovertemplate: `${nm}<br>%{x}: %{y:.2f}<extra></extra>` }));
    tr.push(on2(b, { type: 'bar', x: xs, y: EST.map(e => 100 * S[s][e].bias), marker: { color: c }, name: nm, legendgroup: s, showlegend: false, hovertemplate: `${nm}<br>%{x}: %{y:.1f}%<extra></extra>` }));
  });
  setAx(P, 0, AX(''), AX('MSE(CC) / MSE')); setAx(P, 1, AX(''), AX('Bias of the variance (%)'));
  plot('et-chart', tr, layout(Object.assign(P.L, { barmode: 'group', annotations: [ptitle(a, `MSE ratio, ${n}-day windows`), ptitle(b, 'Bias')], legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  const f1 = x => nf(x, 1), f0 = x => nf(Math.round(100 * x) + 0, 0);
  table('et-tab', ['', ...xs], [[T.h.eff_c, ...EST.map(e => f1(S.continuous[e].eff))], [T.h.bias_c, ...EST.map(e => f0(S.continuous[e].bias))], [T.h.bias_o, ...EST.map(e => f0(S.overnight[e].bias))],
    [T.h.mse_o, ...EST.map(e => f1(S.overnight[e].mse))], [T.h.bias_d, ...EST.map(e => f0(S.discrete[e].bias))]]);
  out('et-out', [T.et(+n, nf(Math.min(...['park', 'gk', 'rs', 'yz'].map(e => S.continuous[e].eff)), 1), nf(Math.max(...['park', 'gk', 'rs', 'yz'].map(e => S.continuous[e].eff)), 1), f0(S.overnight.park.bias), f0(S.overnight.yz.bias), nf(S.overnight.yz.mse, 1), +n !== 21)]);
}

// ================================================================= 16. real data: five estimators, five assets (table)
function estTable(y) {
  return OHN.map(k => {
    const P = parts(k), i0 = firstFrom(P.dates, `${y}-01-01`), W = windowEst(P, i0, P.n), A = ppy(P.dates.slice(i0)), o = P.o.subarray(i0), c = P.c.subarray(i0);
    const mo = mean(o), mc = mean(c); let soc = 0, so = 0, sc = 0; for (let i = 0; i < o.length; i++) { soc += (o[i] - mo) * (c[i] - mc); so += (o[i] - mo) ** 2; sc += (c[i] - mc) ** 2; }
    const v = {}; EST.forEach(e => { v[e] = Math.sqrt(W[e] * A); });
    return { k, from: P.dates[i0], n: P.n - i0, v, night: W.so / (W.so + W.sc), corr: soc / Math.sqrt(so * sc) };
  });
}
function esBlock() {
  const y = +$('es-y').value, nw = single('es-chart', 360), R = estTable(y), xs = OHN.map(k => (k === 'tlv' ? 'TLV' : k === 'snp' ? 'SNP' : NAME[k])); label('es-y').textContent = String(y);
  plot('es-chart', EST.map(e => ({ type: 'bar', x: xs, y: R.map(o => o.v[e]), name: ELAB[e], marker: { color: ECOL[e] }, hovertemplate: `${ELAB[e]}<br>%{x}: %{y:.1f}%<extra></extra>` })),
    layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('Annualised volatility (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  table('es-tab', ['', T.h.from, 'T', ...EST.map(e => ESH[e]), T.h.night, 'corr(o, c)'], R.map(o => [SH[o.k], o.from.slice(0, 4), fint(o.n), ...EST.map(e => nf(o.v[e], 1)), pc(o.night, 0), nf(o.corr, 2)]));
  out('es-out', [T.es(y, R.map(o => [SH[o.k], o.from.slice(0, 4), nf(o.v.yz / o.v.cc, 2), pc(o.night, 0), nf(o.corr, 2)]))]);
}

// ================================================================= 17. S&P 500: rolling estimates since 2019
function rrBlock() {
  const n = +$('rr-n').value, nw = single('rr-chart', 380), P = parts('sp500'), A = ppy(P.dates), R = rollingEst(P, n), i0 = firstFrom(R.dates, '2019-01-01'), d = R.dates.slice(i0); label('rr-n').textContent = String(n);
  const V = {}; EST.forEach(e => { V[e] = Array.from(R[e].subarray(i0), v => Math.sqrt(Math.max(v, 0) * A)); });
  plot('rr-chart', EST.map(e => ({ x: d, y: V[e], mode: 'lines', line: { color: ECOL[e], width: e === 'cc' ? 1.6 : 1 }, name: ELAB[e], hovertemplate: `${ELAB[e]}<br>%{x}<br>%{y:.1f}%<extra></extra>` })),
    layout({ xaxis: AX('Date'), yaxis: AX('Annualised volatility (%)'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const rough = e => sdChange(Array.from(R[e].subarray(i0), v => Math.sqrt(Math.max(v, 1e-8))));
  out('rr-out', [T.rr(n, EST.map(e => [ESH[e], nf(mean(V[e]), 1), nf(100 * rough(e), 1)]))]);
}

// ================================================================= 18. realised variance (simulated one-minute prices)
const IN = D.intra;
function rvDays(m) { return IN.obs.map(p => { let s = 0; for (let i = m; i < p.length; i += m) s += (p[i] - p[i - m]) ** 2; return s; }); }
function rdBlock() {
  const day = +$('rd-d').value, m = +$('rd-m').value, nw = single('rd-chart', 340), p = IN.obs[day - 1], mm = range(0, 390, 1), sm = range(0, 390, m); label('rd-d').textContent = String(day);
  plot('rd-chart', [{ x: mm, y: p, mode: 'lines', line: { color: hexA(BLUE, 0.55), width: 1 }, name: 'One-minute log price (%)', hovertemplate: 'minute %{x}<br>%{y:.3f}<extra></extra>' },
    { x: sm, y: sm.map(i => p[i]), mode: 'lines+markers', line: { color: RED, width: 1.2 }, marker: { size: 5 }, name: `Sampled every ${m} minutes`, hovertemplate: 'minute %{x}<br>%{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('Minutes of the trading day'), yaxis: AX(`Log price (%), day ${day}`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const r = sm.slice(1).map(i => p[i] - p[i - m]), rv = r.reduce((s, v) => s + v * v, 0), sg = IN.sig[day - 1], dr = p[390] - p[0];
  steps('rd-we', [T.rd1(day, 390 / m, m, r.slice(0, 3).map(v => nf(v, 3)).join('; ')), T.rd2(r.slice(0, 3).map(v => nf(v * v, 4)).join(' + '), 390 / m, nf(rv, 3), nf(Math.sqrt(rv), 3)),
    T.rd3(nf(sg, 3), nf(100 * (Math.sqrt(rv) / sg - 1), 1), nf(Math.abs(dr), 3), nf(100 * (Math.abs(dr) / sg - 1), 0))]);
  const all = rvDays(m), err = mean(all.map((v, i) => Math.abs(Math.sqrt(v) - IN.sig[i]) / IN.sig[i]));
  out('rd-out', [T.rd(day, m, nf(Math.sqrt(rv), 3), nf(sg, 3), pc(err, 1))]);
}
function corr(a, b) { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return s / Math.sqrt(sa * sb); }
function reBlock() {
  const m = +$('re-m').value, nw = single('re-chart', 360), dd = range(1, 60, 1), sig = IN.sig, rv = rvDays(m);
  const r = IN.obs.map(p => p[390] - p[0]), park = IN.obs.map(p => (Math.max(...p) - Math.min(...p)) ** 2 / (4 * LN2));
  plot('re-chart', [{ x: dd, y: sig, mode: 'lines', line: { color: '#000', width: 2 }, name: 'True volatility', hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' },
    { x: dd, y: rv.map(Math.sqrt), mode: 'lines+markers', line: { color: RED, width: 1.4 }, marker: { size: 4 }, name: `sqrt(RV), ${m}-minute returns`, hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' },
    { x: dd, y: park.map(Math.sqrt), mode: 'lines', line: { color: GREEN, width: 1, dash: 'dash' }, name: 'sqrt(Parkinson)', hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' },
    { x: dd, y: r.map(Math.abs), mode: 'markers', marker: { color: BLUE, size: 6 }, name: '|daily return|', hovertemplate: 'day %{x}<br>%{y:.3f}%<extra></extra>' }],
  layout({ xaxis: AX('Day'), yaxis: AX('Daily volatility (%)'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const err = e => mean(e.map((v, i) => Math.abs(Math.sqrt(v) - sig[i]) / sig[i]));
  out('re-out', [T.re(m, pc(err(rv), 0), pc(err(park), 0), pc(err(r.map(v => v * v)), 0), nf(corr(rv.map(Math.sqrt), sig), 3), nf(corr(park.map(Math.sqrt), sig), 2), nf(corr(r.map(Math.abs), sig), 2))]);
}

// ================================================================= 19. microstructure noise and the signature plot
function sgBlock() {
  const z = $('sg-z').value, nz = +z, nw = single('sg-chart', 360), ev = IN.every, s5 = IN.sig500;
  const th = ev.map(m => mean(s5.map(s => 1 + 2 * (390 / m) * nz * nz / (s * s))));
  plot('sg-chart', [{ x: ev, y: IN.curves['0.0'], mode: 'lines+markers', line: { color: BLUE, width: 1.6 }, marker: { size: 7 }, name: 'Prices without noise', hovertemplate: '%{x} min<br>%{y:.3f}<extra></extra>' },
    { x: ev, y: IN.curves[z], mode: 'lines+markers', line: { color: RED, width: 1.6 }, marker: { size: 7, symbol: 'square' }, name: `Prices with microstructure noise (sd ${nz}%)`, hovertemplate: '%{x} min<br>%{y:.3f}<extra></extra>' },
    { x: ev, y: th, mode: 'lines', line: { color: ORANGE, width: 1.2, dash: 'dot' }, name: 'Mean of 1 + 2M x noise variance / true variance', hovertemplate: '%{x} min<br>%{y:.3f}<extra></extra>' },
    { x: [1, 390], y: [1, 1], mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: 'True variance', hoverinfo: 'skip' }],
  layout({ xaxis: AX('Sampling interval (minutes, log scale)', { type: 'log', tickvals: ev, ticktext: ev.map(String) }), yaxis: AX('Mean of RV / true variance'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const c = IN.curves[z];
  out('sg-out', [T.sg(nf(nz, 3), nf(c[0], 2), nf(c[3], 2), nf(th[0], 2), nf(th[3], 2), nz !== 0.025)]);
}

// ================================================================= 20. the sign is unpredictable, the size is not
const ACC = {};
function acfSet(k, y) {
  const key = k + y; if (ACC[key]) return ACC[key];
  const S = fromYear(k, y), x = S.x;
  return (ACC[key] = { T: S.n, from: S.dates[0], r: acf(x, 100), sq: acf(x.map(v => v * v), 100), ab: acf(x.map(Math.abs), 100) });
}
function acBlock() {
  const y = +$('ac-y').value, P = panels('ac-chart', 3, { h: 380, hs: 250, extra: 120, vgap: 0.1 }), k = range(1, 100, 1), tr = [], ann = [], items = []; label('ac-y').textContent = String(y);
  ['sp500', 'bet', 'btc'].forEach((n, i) => {
    const A = acfSet(n, y), a = P.ax[i], b = 1.96 / Math.sqrt(A.T), sl = i === 0;
    tr.push(on2(a, { x: [1, 100, 100, 1], y: [-b, -b, b, b], mode: 'lines', fill: 'toself', fillcolor: hexA(TEAL, 0.25), line: { width: 0 }, name: '95% band, i.i.d.', showlegend: sl, legendgroup: 'b', hoverinfo: 'skip' }));
    tr.push(on2(a, { x: k, y: A.ab, mode: 'lines', line: { color: RED, width: 1.5 }, name: '|returns|', showlegend: sl, legendgroup: 'a', hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }));
    tr.push(on2(a, { x: k, y: A.sq, mode: 'lines', line: { color: GREEN, width: 1.5 }, name: 'Squared returns', showlegend: sl, legendgroup: 's', hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }));
    tr.push(on2(a, { x: k, y: A.r, mode: 'lines', line: { color: BLUE, width: 1 }, name: 'Returns', showlegend: sl, legendgroup: 'r', hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }));
    setAx(P, i, AX('Lag (days)'), AX(i === 0 || P.stack ? 'Sample autocorrelation' : '', { range: [-0.12, 0.45] }));
    ann.push(ptitle(a, `${NAME[n]}, from ${A.from.slice(0, 4)}`));
    const out_ = v => v.filter(z => Math.abs(z) > b).length;
    items.push([SH[n], A.from, fint(A.T), out_(A.ab), out_(A.sq), pc(mean(A.ab.map((v, j) => (v > A.sq[j] ? 1 : 0))), 0), nf(A.ab[99], 2)]);
  });
  plot('ac-chart', tr, layout(Object.assign(P.L, { annotations: ann, legend: LEG(P.stack, P.stack ? -0.04 : -0.2), margin: { l: 56, r: 10, t: 24, b: 40 } })));
  out('ac-out', [T.ac(y, items)]);
}

// ================================================================= 21. interpretation: clustering in three markets (table)
const ACP = {};
function acfPow(k, d, lag) { const key = `${k}:${d}:${lag}`; if (ACP[key] !== undefined) return ACP[key]; const x = ser(k).x.map(v => Math.abs(v) ** d); return (ACP[key] = acf(x, lag)[lag - 1]); }
function atBlock() {
  const lag = +$('at-k').value, nw = single('at-chart', 360), ds = range(0.25, 3, 0.25); label('at-k').textContent = String(lag);
  const tr = ['sp500', 'bet', 'btc'].map(n => ({ x: ds, y: ds.map(d => acfPow(n, d, lag)), mode: 'lines+markers', line: { color: COL[n], width: 1.6 }, marker: { size: 6 }, name: NAME[n], hovertemplate: `${NAME[n]}<br>d = %{x}<br>%{y:.3f}<extra></extra>` }));
  plot('at-chart', tr, layout({ xaxis: AX('Power d in |r(t)|^d (d = 1: |r|, d = 2: r^2)'), yaxis: AX(`Autocorrelation at lag ${lag}`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 },
    shapes: [1, 2].map(x => ({ type: 'line', xref: 'x', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: { color: DARK, width: 0.8, dash: 'dot' } })) }));
  const rows = ['sp500', 'bet', 'btc'].map(n => { const A = acfSet(n, 1990), b = 1.96 / Math.sqrt(A.T), best = ds.reduce((m, d) => (acfPow(n, d, lag) > acfPow(n, m, lag) ? d : m), ds[0]); return { n, A, b, best }; });
  table('at-tab', ['', 'ρ̂₁(|r|)', 'ρ̂₁₀(|r|)', 'ρ̂₁₀₀(|r|)', 'ρ̂₁(r²)', 'ρ̂₁₀(r²)', T.h.outabs, `ρ̂<sub>${lag}</sub>(|r|)`, `ρ̂<sub>${lag}</sub>(r²)`, T.h.bestd],
    rows.map(o => [SH[o.n], nf(o.A.ab[0], 2), nf(o.A.ab[9], 2), nf(o.A.ab[99], 2), nf(o.A.sq[0], 2), nf(o.A.sq[9], 2), `${o.A.ab.filter(v => Math.abs(v) > o.b).length}/100`, nf(o.A.ab[lag - 1], 3), nf(o.A.sq[lag - 1], 3), nf(o.best, 2)]));
  out('at-out', [T.at(lag, rows.map(o => [SH[o.n], nf(o.best, 2), nf(acfPow(o.n, o.best, lag), 3), nf(o.A.ab[lag - 1], 3), nf(o.A.sq[lag - 1], 3)]))]);
}

// ================================================================= 22. is clustering real? shuffle the days
function shuffled(b) {
  const x = ser('sp500').x, p = D.perm[String(b)];
  if (b === 1) return p.map(i => x[i]);
  const outv = []; p.forEach(j => { for (let i = j * b; i < Math.min((j + 1) * b, x.length); i++) outv.push(x[i]); }); return outv;
}
function shBlock() {
  const b = +$('sh-b').value, S = ser('sp500'), s = shuffled(b), P = panels('sh-chart', 2, { h: 360, gap: 0.1 }), [a, c] = P.ax, j = range(1, 50, 1), band = 1.96 / Math.sqrt(S.n), lim = Math.ceil(Math.max(...S.x.map(Math.abs)));
  const A0 = acf(S.x.map(Math.abs), 50), A1 = acf(s.map(Math.abs), 50);
  const tr = [on2(a, { x: S.dates, y: s, mode: 'lines', line: { color: ORANGE, width: 0.4 }, name: b === 1 ? 'The same returns, random order' : `The same returns, blocks of ${b} days in random order`, hoverinfo: 'skip' }),
    on2(c, { x: [0.5, 50.5, 50.5, 0.5], y: [-band, -band, band, band], mode: 'lines', fill: 'toself', fillcolor: hexA(TEAL, 0.3), line: { width: 0 }, name: '95% band, i.i.d.', hoverinfo: 'skip' }),
    on2(c, { type: 'bar', x: j, y: A0, marker: { color: BLUE }, name: 'S&P 500 returns, actual order', hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }),
    on2(c, { type: 'bar', x: j, y: A1, marker: { color: ORANGE }, name: 'Shuffled', showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' })];
  setAx(P, 0, AX(''), AX('%', { range: [-lim, lim] })); setAx(P, 1, AX('Lag (days)'), AX('ACF of |returns|', { range: [-0.05, 0.42] }));
  plot('sh-chart', tr, layout(Object.assign(P.L, { barmode: 'group', annotations: [ptitle(a, b === 1 ? 'S&P 500 returns in a random order' : `Blocks of ${b} days in a random order`), ptitle(c, 'ACF of |returns|: actual (navy) and shuffled (orange)')], legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const q0 = ljungBox(S.x.map(v => v * v), 10).q, q1 = ljungBox(s.map(v => v * v), 10).q;
  out('sh-out', [T.sh(b, nf(A0[0], 2), nf(A1[0], 2), fnum(q0, 0), fnum(q1, 1), nf(moments(s).exkurt, 1), b !== 1)]);
}

// ================================================================= 23. testing for clustering: Ljung-Box on squared returns (worked example)
function lqBlock() {
  const k = $('lq-s').value, m = +$('lq-m').value, nw = single('lq-chart', 340), x = ser(k).x; label('lq-m').textContent = String(m);
  const L2 = ljungBox(x.map(v => v * v), m), L1 = ljungBox(x, m), ks = range(1, m, 1), cum = a => { let s = 0; return a.map(v => (s += v)); };
  plot('lq-chart', [{ x: ks, y: cum(L2.terms), mode: 'lines+markers', line: { color: GREEN, width: 1.6 }, name: `Q(k) of squared returns, ${NAME[k]}`, hovertemplate: 'k = %{x}<br>%{y:.1f}<extra></extra>' },
    { x: ks, y: cum(L1.terms), mode: 'lines+markers', line: { color: BLUE, width: 1.6 }, name: 'Q(k) of returns', hovertemplate: 'k = %{x}<br>%{y:.1f}<extra></extra>' },
    { x: ks, y: ks.map(q => chi2ppf(0.95, q)), mode: 'lines', line: { color: '#000', width: 0.9, dash: 'dash' }, name: '5% critical value', hovertemplate: 'k = %{x}<br>%{y:.2f}<extra></extra>' }],
  layout({ xaxis: AX('Number of lags tested', { dtick: m > 12 ? 2 : 1 }), yaxis: AX('Ljung-Box statistic (log scale)', LOGT), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const T0 = L2.T;
  steps('lq-we', [T.lq1(SH[k], fint(T0), m, L2.rho.slice(0, 6).map(v => nf(v, 3)).join('; ') + (m > 6 ? '; …' : '')),
    T.lq2(fint(T0), fint(T0 + 2), fnum(T0 * (T0 + 2) / (T0 - 1), 1), L2.terms.slice(0, 6).map(v => fnum(v, 0)).join('; ') + (m > 6 ? '; …' : '')),
    T.lq3(m, fnum(L2.q, 0), nf(L2.crit, 2), fpe(L2.p), L2.q > L2.crit), T.lq4(m, nf(L1.q, 1), nf(L1.crit, 2), fpe(L1.p), L1.q > L1.crit)]);
  out('lq-out', [T.lq(SH[k], m, fnum(L2.q, 0), nf(L1.q, 1), nf(L2.crit, 2), L1.q > L1.crit)]);
}

// ================================================================= 24. the ARCH-LM test (worked example)
const ALC = {};
const archC = (k, q) => { const key = k + q; return ALC[key] || (ALC[key] = archLM(ser(k).x, q)); };
function alBlock() {
  const k = $('al-s').value, q = +$('al-q').value, nw = single('al-chart', 340), A = archC(k, q), js = range(1, q, 1); label('al-q').textContent = String(q);
  plot('al-chart', [{ type: 'bar', x: js, y: A.b.slice(1), marker: { color: COL[k] }, name: `c_j of the auxiliary regression, ${NAME[k]}, q = ${q}`, hovertemplate: 'j = %{x}<br>%{y:.3f}<extra></extra>' }],
    layout({ xaxis: AX('Lag j of e(t-j)^2', { dtick: q > 12 ? 2 : 1 }), yaxis: AX('Coefficient c_j'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 }, showlegend: true }));
  steps('al-we', [T.al1(SH[k], q, fint(A.n), A.b.slice(0, Math.min(q + 1, 6)).map(v => nf(v, 3)).join('; ') + (q > 5 ? '; …' : '')), T.al2(nf(A.r2, 4)),
    T.al3(fint(A.n), nf(A.r2, 4), fnum(A.lm, 1), q, nf(A.crit, 2), fpe(A.p))]);
  out('al-out', [T.al(SH[k], q, fnum(A.lm, 0), nf(A.r2, 3), fint(A.n))]);
}

// ================================================================= 25. clustering tests in six series (table)
const LBC = {};
const lbC = (k, m, f) => { const key = `${k}${m}${f}`; if (LBC[key]) return LBC[key]; const x = ser(k).x; return (LBC[key] = ljungBox(f === 'r' ? x : f === 'sq' ? x.map(v => v * v) : x.map(Math.abs), m)); };
function ctBlock() {
  const m = +$('ct-m').value, q = +$('ct-q').value, nw = single('ct-chart', 360), xs = RETN.map(k => SH[k]); label('ct-m').textContent = String(m);
  const R = RETN.map(k => ({ k, r: lbC(k, m, 'r'), sq: lbC(k, m, 'sq'), ab: lbC(k, m, 'ab'), a: archC(k, q), x: ser(k).x }));
  plot('ct-chart', [{ type: 'bar', x: xs, y: R.map(o => o.r.q), name: `Q(${m}), returns`, marker: { color: BLUE }, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { type: 'bar', x: xs, y: R.map(o => o.sq.q), name: `Q(${m}), squared returns`, marker: { color: GREEN }, hovertemplate: '%{x}: %{y:.0f}<extra></extra>' },
    { type: 'bar', x: xs, y: R.map(o => o.ab.q), name: `Q(${m}), |returns|`, marker: { color: RED }, hovertemplate: '%{x}: %{y:.0f}<extra></extra>' },
    { type: 'bar', x: xs, y: R.map(o => o.a.lm), name: `ARCH-LM(${q})`, marker: { color: AMBER }, hovertemplate: '%{x}: %{y:.0f}<extra></extra>' },
    { x: [xs[0], xs[xs.length - 1]], y: [R[0].r.crit, R[0].r.crit], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: `5% critical value of Q(${m})`, hoverinfo: 'skip' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('Statistic (log scale)', LOGT), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const red = p => (p < 0.05 ? 'bad' : '');
  table('ct-tab', ['', 'T', 'ρ̂₁(r)', `Q(${m}), r`, 'p', 'ρ̂₁(r²)', `Q(${m}), r²`, `Q(${m}), |r|`, `ARCH-LM(${q})`, 'R²', T.h.exk],
    R.map(o => [SH[o.k], fint(o.r.T), nf(o.r.rho[0], 3), { v: nf(o.r.q, 1), c: red(o.r.p) }, { v: fp(o.r.p), c: red(o.r.p) }, nf(o.sq.rho[0], 3), { v: fnum(o.sq.q, 0), c: red(o.sq.p) }, { v: fnum(o.ab.q, 0), c: red(o.ab.p) },
      { v: fnum(o.a.lm, 0), c: red(o.a.p) }, nf(o.a.r2, 3), nf(moments(o.x).exkurt, 1)]));
  const rat = R.map(o => o.sq.q / o.r.q);
  out('ct-out', [T.ct(m, nf(R[0].r.crit, 2), R.filter(o => o.r.p < 0.05).map(o => SH[o.k]), fp(Math.max(...R.map(o => o.r.p))), nf(Math.min(...rat), 1), nf(Math.max(...rat), 1), q, nf(R[0].a.crit, 2), R.filter(o => o.sq.p >= 0.05 || o.a.p >= 0.05).length)]);
}

// ================================================================= 26. clustering creates heavy tails
const mixK = (p, s) => { const e2 = (1 - p) + p * s * s, e4 = (1 - p) + p * s ** 4; return 3 * e4 / (e2 * e2); };
const garchK = (a, b) => { const den = 1 - b * b - 2 * a * b - 3 * a * a; return den > 0 ? 3 + 6 * a * a / den : null; };
const KU_STEPS = [[1.5, 0.8], [2, 0.85], [2.5, 0.86], [3, 0.87], [3.5, 0.88], [4, 0.89], [2, 0.85]];
function kuBlock() {
  const s = +$('ku-s').value, be = +$('ku-b').value, P = panels('ku-chart', 2, { h: 380, gap: 0.1 }), [a, b] = P.ax, p = range(0, 1, 0.005), al = range(0, 0.2, 0.0005); label('ku-s').textContent = nf(s, 1); label('ku-b').textContent = nf(be, 2);
  const tr = [];
  [[2, BLUE], [3, RED], [4, GREEN]].forEach(([r, c]) => tr.push(on2(a, { x: p, y: p.map(z => mixK(z, r)), mode: 'lines', line: { color: c, width: 1.6 }, name: `Mixture, turbulent sd = ${r} x calm sd`, hovertemplate: 'p = %{x:.2f}<br>%{y:.2f}<extra></extra>' })));
  tr.push(on2(a, { x: p, y: p.map(z => mixK(z, s)), mode: 'lines', line: { color: DARK, width: 2.2, dash: 'dash' }, name: `Chosen mixture: sd ratio ${s}`, hovertemplate: 'p = %{x:.2f}<br>%{y:.2f}<extra></extra>' }));
  tr.push(on2(a, { x: [0, 1], y: [3, 3], mode: 'lines', line: { color: '#000', width: 0.7, dash: 'dot' }, name: 'Normal distribution (3)', hoverinfo: 'skip' }));
  [[0.8, ORANGE], [0.85, PURPLE], [0.9, TEAL]].forEach(([bb, c]) => tr.push(on2(b, { x: al, y: al.map(z => garchK(z, bb)), mode: 'lines', line: { color: c, width: 1.6 }, name: `GARCH(1,1), beta = ${bb}`, hovertemplate: 'alpha = %{x:.3f}<br>%{y:.2f}<extra></extra>' })));
  tr.push(on2(b, { x: al, y: al.map(z => garchK(z, be)), mode: 'lines', line: { color: DARK, width: 2.2, dash: 'dash' }, name: `Chosen GARCH(1,1): beta = ${be.toFixed(2)}`, hovertemplate: 'alpha = %{x:.3f}<br>%{y:.2f}<extra></extra>' }));
  tr.push(on2(b, { x: [0, 0.2], y: [3, 3], mode: 'lines', line: { color: '#000', width: 0.7, dash: 'dot' }, showlegend: false, hoverinfo: 'skip' }));
  setAx(P, 0, AX('Share of turbulent days p'), AX('Kurtosis')); setAx(P, 1, AX('alpha (weight of the last squared shock)'), AX(P.stack ? 'Kurtosis' : '', { range: [2.5, 15] }));
  plot('ku-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  const e2 = 0.8 + 0.2 * s * s, e4 = 0.8 + 0.2 * s ** 4; let pm = 0; p.forEach(z => { if (mixK(z, s) > mixK(pm, s)) pm = z; });
  let amax = 0; for (let x = 0; x < 1; x += 0.0001) { if (be * be + 2 * x * be + 3 * x * x < 1) amax = x; else break; }
  const g = garchK(0.1, be);
  out('ku-out', [T.ku(nf(s, 1), nf(e2, 2), nf(e4, 2), nf(3 * e4 / (e2 * e2), 2), nf(pm, 3), nf(mixK(pm, s), 2), nf(be, 2), g === null ? null : nf(g, 2), nf(amax, 3))]);
}

// ================================================================= 27. volatility over three decades
function annualVol(k, y) {
  const S = fromYear(k, Math.max(y, 1990)), A = ppy(S.dates), G = {};
  S.x.forEach((v, i) => { const yr = S.dates[i].slice(0, 4); (G[yr] = G[yr] || []).push(v); });
  return Object.keys(G).sort().filter(yr => G[yr].length >= 100).map(yr => ({ y: +yr, v: sdev(G[yr]) * Math.sqrt(A) }));
}
function monthlyRV(k, y) {
  const S = fromYear(k, Math.max(y, 1990)), A = ppy(S.dates), G = {}, keys = [];
  S.x.forEach((v, i) => { const m = S.dates[i].slice(0, 7); if (!G[m]) { G[m] = []; keys.push(m); } G[m].push(v * v); });
  const rv = keys.map(m => Math.sqrt(mean(G[m]) * A)), lm = rv.map(Math.log), X = lm.slice(0, -1), Y = lm.slice(1), o = ols1(Y, X);
  let ssr = 0; Y.forEach((v, i) => { ssr += (v - o.a - o.b * X[i]) ** 2; }); const mx = mean(X); let sxx = 0; X.forEach(v => { sxx += (v - mx) ** 2; });
  return { rv, lm, phi: o.b, c: o.a, se: Math.sqrt(ssr / (Y.length - 2) / sxx), hl: Math.log(0.5) / Math.log(o.b), lr: Math.exp(o.a / (1 - o.b)), n: rv.length, skew: moments(rv).skew, skewl: moments(lm).skew, mean: mean(rv), med: median(rv), from: keys[0] };
}
function loBlock() {
  const y = +$('lo-y').value, k = $('lo-s').value, P = panels('lo-chart', 2, { h: 380, gap: 0.1 }), [a, b] = P.ax, tr = [], items = []; label('lo-y').textContent = String(y);
  MK4.forEach(n => {
    const V = annualVol(n, y);
    tr.push(on2(a, { x: V.map(o => o.y), y: V.map(o => o.v), mode: 'lines+markers', line: { color: COL[n], width: 1.3 }, marker: { size: 5 }, name: NAME[n], hovertemplate: `${NAME[n]}<br>%{x}: %{y:.1f}%<extra></extra>` }));
    let lo = V[0], hi = V[0]; V.forEach(o => { if (o.v < lo.v) lo = o; if (o.v > hi.v) hi = o; });
    items.push([SH[n], nf(lo.v, 0), lo.y, nf(hi.v, 0), hi.y]);
  });
  const M = monthlyRV(k, y), g = [Math.min(...M.lm), Math.max(...M.lm)];
  tr.push(on2(b, { x: M.lm.slice(0, -1), y: M.lm.slice(1), mode: 'markers', marker: { color: hexA(COL[k], 0.6), size: 5 }, name: `${NAME[k]}, month m-1 against m`, hovertemplate: '%{x:.2f} -> %{y:.2f}<extra></extra>' }));
  tr.push(on2(b, { x: g, y: g.map(v => M.c + M.phi * v), mode: 'lines', line: { color: RED, width: 1.6 }, name: `AR(1) line, phi = ${M.phi.toFixed(2)}`, hoverinfo: 'skip' }));
  tr.push(on2(b, { x: g, y: g, mode: 'lines', line: { color: '#000', width: 0.7, dash: 'dash' }, name: '45-degree line', hoverinfo: 'skip' }));
  setAx(P, 0, AX('Year'), AX('Annual volatility (%)')); setAx(P, 1, AX('log RV, previous month'), AX('log RV, this month'));
  plot('lo-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } })));
  out('lo-out', [T.lo(y, items, SH[k], M.n, nf(M.phi, 2), nf(M.se, 2), nf(M.hl, 1), nf(M.lr, 1), nf(M.skew, 1), nf(M.skewl, 1), nf(M.mean, 1), nf(M.med, 1))]);
}

// ================================================================= 28. VIX against the volatility that followed
const VXC = {};
function vixData(h) {
  if (VXC[h]) return VXC[h];
  const S = ser('sp500'), a = ppy(S.dates), x = S.x, n = S.n, fwd = new Float64Array(n).fill(NaN), past = new Float64Array(n).fill(NaN);
  for (let t = 0; t + h < n; t++) { let s = 0; for (let j = 1; j <= h; j++) s += x[t + j] ** 2; fwd[t] = Math.sqrt(s * a / h); }
  for (let t = h - 1; t < n; t++) { let s = 0; for (let j = 0; j < h; j++) s += x[t - j] ** 2; past[t] = Math.sqrt(s / h * a); }
  return (VXC[h] = { dates: S.dates, vix: D.vix, fwd, past, x, a });
}
function vixStats(h, y) {
  const V = vixData(h), i0 = firstFrom(V.dates, `${y}-01-01`), idx = [];
  for (let t = i0; t < V.dates.length; t++) if (!isNaN(V.fwd[t]) && !isNaN(V.past[t]) && V.vix[t] !== null) idx.push(t);
  const vx = idx.map(t => V.vix[t]), fw = idx.map(t => V.fwd[t]), pa = idx.map(t => V.past[t]);
  const all = idx, vx2 = vx, fw2 = fw;
  const dv = [], rr = []; for (let t = Math.max(i0, 1); t < V.dates.length; t++) { dv.push(Math.log(V.vix[t]) - Math.log(V.vix[t - 1])); rr.push(V.x[t]); }
  return { n: all.length, mvix: mean(vx2), mrv: mean(fw2), share: mean(vx2.map((v, i) => (v > fw2[i] ? 1 : 0))), gap: mean(vx2.map((v, i) => v - fw2[i])), mz: ols1(fw2, vx2), mzp: ols1(fw, pa), corr: corr(dv, rr), vx: vx2, fw: fw2, all };
}
function vxBlock() {
  const h = +$('vx-h').value, nw = single('vx-chart', 380), V = vixData(h), S = vixStats(h, 1990);
  const d = V.dates, fw = Array.from(V.fwd).map(v => (isNaN(v) ? null : v));
  let im = 0; d.forEach((_, i) => { if (V.vix[i] > V.vix[im]) im = i; });
  plot('vx-chart', [{ x: d, y: V.vix, mode: 'lines', line: { color: RED, width: 0.9 }, name: 'VIX (implied volatility, %)', hovertemplate: '%{x}<br>VIX %{y:.1f}<extra></extra>' },
    { x: d, y: fw, mode: 'lines', line: { color: BLUE, width: 0.9 }, name: `Realised volatility of the next ${h} days (%)`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }],
  layout({ xaxis: AX('Date'), yaxis: AX('Annualised volatility (%)'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('vx-out', [T.vx(h, fint(S.n), pc(S.share, 0), nf(S.mvix, 1), nf(S.mrv, 1), nf(S.gap, 1), nf(V.vix[im], 1), fdate(d[im]), h !== 21)]);
}

// ================================================================= 29. interpretation: implied against realised volatility (table)
const VX_H = [5, 10, 21, 42, 63];
function viBlock() {
  const h = +$('vi-h').value, y = +$('vi-y').value, nw = single('vi-chart', 400), S = vixStats(h, y); label('vi-y').textContent = String(y);
  const g = [Math.min(...S.vx), Math.max(...S.vx)];
  plot('vi-chart', [{ x: S.vx, y: S.fw, mode: 'markers', marker: { color: hexA(BLUE, 0.25), size: 3 }, name: `Days from ${y}: VIX and the next ${h} days`, hoverinfo: 'skip' },
    { x: g, y: g.map(v => S.mz.a + S.mz.b * v), mode: 'lines', line: { color: RED, width: 1.8 }, name: `Regression: realised = ${S.mz.a.toFixed(2)} + ${S.mz.b.toFixed(2)} VIX`, hoverinfo: 'skip' },
    { x: g, y: g, mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: '45-degree line (VIX = realised)', hoverinfo: 'skip' }],
  layout({ xaxis: AX('VIX (%)'), yaxis: AX(`Realised volatility, next ${h} days (%)`), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const R = VX_H.map(hh => ({ hh, s: vixStats(hh, y) }));
  table('vi-tab', ['h', T.h.days, T.h.mvix, T.h.mrv, T.h.above, T.h.gap, 'a', 'b', 'R²', T.h.r2past],
    R.map(o => Object.assign([o.hh, fint(o.s.n), nf(o.s.mvix, 1), nf(o.s.mrv, 1), pc(o.s.share, 0), nf(o.s.gap, 1), nf(o.s.mz.a, 2), nf(o.s.mz.b, 2), nf(o.s.mz.r2, 2), nf(o.s.mzp.r2, 2)], { cls: o.hh === h ? 'sel' : '' })));
  out('vi-out', [T.vi(h, y, pc(S.share, 0), nf(S.mz.a, 2), nf(S.mz.b, 2), nf(S.mz.r2, 2), nf(S.mzp.r2, 2), nf(S.corr, 2), h !== 21 || y !== 1990)]);
}

// ================================================================= 30. bad news raises future volatility
function leverage(k, y, f) {
  const x = fromYear(k, y).x, n = x.length, g = f === 'sq' ? (v => v * v) : Math.abs, cc = {};
  for (let j = 1; j <= 10; j++) cc[j] = corr(x.slice(0, n - j), x.slice(j).map(g));
  cc[-1] = corr(x.slice(1), x.slice(0, n - 1).map(g));
  let sn = 0, nn = 0, sp = 0, np_ = 0; for (let t = 0; t < n - 1; t++) { const v = x[t + 1] ** 2; if (x[t] < 0) { sn += v; nn++; } else if (x[t] > 0) { sp += v; np_++; } }
  return { cc, ratio: (sn / nn) / (sp / np_), from: fromYear(k, y).dates[0] };
}
function lvBlock() {
  const y = +$('lv-y').value, f = $('lv-m').value, nw = single('lv-chart', 360), j = range(1, 10, 1), w = 0.2, tr = [], items = []; label('lv-y').textContent = String(y);
  MK4.forEach((k, i) => { const L = leverage(k, y, f); tr.push({ type: 'bar', x: j.map(v => v + (i - 1.5) * w), y: j.map(v => L.cc[v]), width: w, marker: { color: COL[k] }, name: NAME[k], hovertemplate: `${NAME[k]}<br>j = %{customdata}<br>%{y:.3f}<extra></extra>`, customdata: j });
    items.push([SH[k], L.from.slice(0, 4), nf(L.cc[1], 3), nf(L.cc[-1], 3), nf(L.ratio, 2)]); });
  plot('lv-chart', tr, layout({ xaxis: AX('j (days ahead)', { tickvals: j }), yaxis: AX(f === 'sq' ? 'corr( r(t), r(t+j)^2 )' : 'corr( r(t), |r(t+j)| )'), legend: LEG(nw, -0.16), margin: { l: 60, r: 12, t: 14, b: 40 }, shapes: [ZERO()] }));
  out('lv-out', [T.lv(y, f, items)]);
}

// ================================================================= 31. nonparametric volatility given yesterday's return
const NPC = {};
function npVol(k, h) {
  const key = k + h; if (NPC[key]) return NPC[key];
  const r = ser(k).x.map(v => v / 100), X = r.slice(0, -1), Y = r.slice(1), Y2 = Y.map(v => v * v), n = X.length;
  let lo = Infinity, hi = -Infinity; for (const v of X) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const grid = range(0, 99, 1).map(i => lo + (hi - lo) * i / 99), s = [];
  for (const g0 of grid) {
    let s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0, u0 = 0, u1 = 0;
    for (let i = 0; i < n; i++) { const z = (X[i] - g0) / h; if (Math.abs(z) >= 1) continue; const w = 15 / 16 * (1 - z * z) ** 2 / h, d = X[i] - g0; s0 += w; s1 += w * d; s2 += w * d * d; t0 += w * Y[i]; t1 += w * d * Y[i]; u0 += w * Y2[i]; u1 += w * d * Y2[i]; }
    const den = s0 * s2 - s1 * s1, m1 = (s2 * t0 - s1 * t1) / den, m2 = (s2 * u0 - s1 * u1) / den;
    s.push(100 * Math.sqrt(Math.max(m2 - m1 * m1, 0)));
  }
  return (NPC[key] = { x: grid.map(v => 100 * v), s });
}
function interp(x, xs, ys) { if (x <= xs[0]) return ys[0]; for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[ys.length - 1]; }
function niBlock() {
  const h = +$('ni-h').value, nw = single('ni-chart', 360), tr = [], items = []; label('ni-h').textContent = nf(h, 2);
  ['sp500', 'dax'].forEach(k => {
    const V = npVol(k, h), idx = V.x.map((v, i) => i).filter(i => V.x[i] > -5 && V.x[i] < 5), xs = idx.map(i => V.x[i]), ys = idx.map(i => V.s[i]);
    tr.push({ x: xs, y: ys, mode: 'lines', line: { color: COL[k], width: 1.8 }, name: NAME[k], hovertemplate: `${NAME[k]}<br>x = %{x:.2f}%<br>%{y:.2f}%<extra></extra>` });
    let im = 0; ys.forEach((v, i) => { if (v < ys[im]) im = i; });
    const m2 = interp(-2, V.x, V.s), p2v = interp(2, V.x, V.s);
    items.push([SH[k], nf(m2, 2), nf(p2v, 2), nf(interp(0, V.x, V.s), 2), nf(xs[im], 2), nf(m2 / p2v, 2)]);
  });
  plot('ni-chart', tr, layout({ xaxis: AX("Yesterday's return x (%)"), yaxis: AX("sd of today's return given x (%)"), legend: LEG(nw, -0.16), margin: { l: 60, r: 12, t: 14, b: 40 },
    shapes: [{ type: 'line', xref: 'x', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: '#000', width: 0.6, dash: 'dot' } }] }));
  out('ni-out', [T.ni(nf(h, 2), items, Math.abs(h - 0.04) > 1e-9)]);
}

// ================================================================= 32. forecasts: rolling windows, EWMA, VIX (and losses)
let FC = null;
function fcAll() {                  // forecasts() of the chapter on the whole sample: next-day variance forecasts and the two proxies
  if (FC) return FC;
  const S = ser('sp500'), x = S.x, n = S.n, a = ppy(S.dates), P = parts('sp500'), F = { dates: S.dates, r2: x.map(v => v * v) };
  [21, 63, 252].forEach(m => { const h = new Float64Array(n).fill(NaN); let s = 0; for (let t = 0; t < n; t++) { if (t >= 1) s += x[t - 1] ** 2; if (t >= m + 1) s -= x[t - m - 1] ** 2; if (t >= m) h[t] = s / m; } F['hist' + m] = h; });
  [0.94, 0.97].forEach(l => { F['ewma' + Math.round(100 * l)] = ewmaVar(x, l).subarray(0, n); });
  F.vix = new Float64Array(n).fill(NaN); for (let t = 1; t < n; t++) F.vix[t] = D.vix[t - 1] ** 2 / a;
  F.range = new Float64Array(n).fill(NaN); S.dates.forEach((d, t) => { const i = P.idx.get(d); if (i !== undefined) F.range[t] = P.o[i] ** 2 + P.gk[i]; });
  F.ewmaL = {};
  return (FC = F);
}
const METH = ['hist21', 'hist63', 'hist252', 'ewma94', 'ewma97', 'vix'];
const MLAB = { hist21: '21 days', hist63: '63 days', hist252: '252 days', ewma94: 'EWMA 0.94', ewma97: 'EWMA 0.97', vix: 'VIX' };
const MCOL = { hist21: RED, hist63: ORANGE, hist252: GREEN, ewma94: PURPLE, ewma97: TEAL, vix: BLUE };
function fcRows(start) { const F = fcAll(), i0 = firstFrom(F.dates, start), idx = []; for (let t = i0; t < F.dates.length; t++) { if ([...METH, 'range'].every(m => !isNaN(F[m][t]))) idx.push(t); } return idx; }
function fcEval(start) {
  const F = fcAll(), idx = fcRows(start), res = {};
  METH.forEach(m => {
    const h = idx.map(t => F[m][t]);
    res[m] = {};
    ['r2', 'range'].forEach(pr => { const p = idx.map(t => F[pr][t]); res[m][pr] = { mse: mean(p.map((v, i) => (v - h[i]) ** 2)), qlike: mean(p.map((v, i) => v / h[i] + Math.log(h[i]))), mz: ols1(p, h) }; });
  });
  return { n: idx.length, first: F.dates[idx[0]], res, idx };
}
function dmTest(start, m1, m2, lags = 5) {
  const F = fcAll(), idx = fcRows(start), L = m => idx.map(t => F.r2[t] / F[m][t] + Math.log(F[m][t])), a = L(m1), b = L(m2), d = a.map((v, i) => v - b[i]), db = mean(d), e = d.map(v => v - db), T0 = e.length;
  let v = e.reduce((s, z) => s + z * z, 0) / T0; for (let j = 1; j <= lags; j++) { let s = 0; for (let t = j; t < T0; t++) s += e[t] * e[t - j]; v += 2 * (1 - j / (lags + 1)) * s / T0; }
  const t = db / Math.sqrt(v / T0); return { d: db, t, p: p2(t) };
}
function fpBlock() {
  const y = +$('fp-y').value, l = $('fp-l').value, nw = single('fp-chart', 380), F = fcAll(), a = ppy(F.dates), idx = fcRows(`${y}-01-01`).filter(t => F.dates[t] < `${y + 1}-01-01`), d = idx.map(t => F.dates[t]); label('fp-y').textContent = String(y);
  const ew = 'ewma' + Math.round(100 * l), tr = [{ type: 'bar', x: d, y: idx.map(t => Math.sqrt(F.r2[t] * a)), marker: { color: hexA(AMBER, 0.55) }, name: '|return|, annualised', hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }];
  const pk = {};
  [['hist252', GREEN, '252-day window'], [ew, RED, `EWMA, lambda = ${l}`], ['vix', BLUE, 'VIX']].forEach(([m, c, nm]) => {
    const v = idx.map(t => Math.sqrt(F[m][t] * a)); let im = 0; v.forEach((z, i) => { if (z > v[im]) im = i; }); pk[m === ew ? 'ew' : m] = [v[im], d[im]];
    tr.push({ x: d, y: v, mode: 'lines', line: { color: c, width: 1.6 }, name: nm, hovertemplate: `${nm}<br>%{x}<br>%{y:.1f}%<extra></extra>` });
  });
  plot('fp-chart', tr, layout({ bargap: 0, xaxis: AX('Date'), yaxis: AX('Annualised volatility (%)'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const ql = m => mean(idx.map(t => F.r2[t] / F[m][t] + Math.log(F[m][t])));
  out('fp-out', [T.fp(y, nf(+l, 2), [['hist252', pk.hist252], ['ew', pk.ew], ['vix', pk.vix]].map(([m, [v, dd]]) => [m, nf(v, 1), fdate(dd)]), nf(ql('hist252'), 3), nf(ql(ew), 3), nf(ql('vix'), 3))]);
}

// ================================================================= 33. loss functions: MSE and QLIKE (worked example)
function lsBlock() {
  const s = +$('ls-p').value, P = panels('ls-chart', 2, { h: 340, gap: 0.1 }), [a, b] = P.ax, h = range(0.2, 4, 0.02); label('ls-p').textContent = nf(s, 2);
  const mse = h.map(v => (s - v) ** 2), ql = h.map(v => s / v + Math.log(v)), tr = [];
  tr.push(on2(a, { x: h, y: mse, mode: 'lines', line: { color: BLUE, width: 1.8 }, name: 'MSE loss (proxy - h)^2', hovertemplate: 'h = %{x:.2f}<br>%{y:.3f}<extra></extra>' }));
  tr.push(on2(b, { x: h, y: ql, mode: 'lines', line: { color: RED, width: 1.8 }, name: 'QLIKE loss proxy/h + ln h', hovertemplate: 'h = %{x:.2f}<br>%{y:.3f}<extra></extra>' }));
  [[0.5, GREEN], [2, ORANGE]].forEach(([v, c]) => {
    tr.push(on2(a, { x: [v], y: [(s - v) ** 2], mode: 'markers', marker: { color: c, size: 10 }, name: `Forecast h = ${v}`, legendgroup: 'h' + v, hovertemplate: `h = ${v}: %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { x: [v], y: [s / v + Math.log(v)], mode: 'markers', marker: { color: c, size: 10 }, name: `Forecast h = ${v}`, legendgroup: 'h' + v, showlegend: false, hovertemplate: `h = ${v}: %{y:.3f}<extra></extra>` }));
  });
  setAx(P, 0, AX('Variance forecast h'), AX('Loss')); setAx(P, 1, AX('Variance forecast h'), AX(''));
  plot('ls-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(a, `MSE, proxy = ${s}`), ptitle(b, `QLIKE, proxy = ${s} (minimum at h = proxy)`)], legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const F = fcAll(), t = F.dates.indexOf('2020-03-16'), e = F.ewma94[t], hh = F.hist252[t], pr = F.r2[t];
  steps('ls-we', [T.ls1(nf(s, 2)), T.ls2(nf(s, 2), nf((s - 0.5) ** 2, 3), nf((s - 2) ** 2, 3)), T.ls3(nf(s, 2), nf(s / 0.5, 3), nf(Math.log(0.5), 3), nf(s / 0.5 + Math.log(0.5), 3), nf(s / 2, 3), nf(Math.log(2), 3), nf(s / 2 + Math.log(2), 3)),
    T.ls4(nf(pr, 1), nf(e, 2), nf(pr / e + Math.log(e), 2), nf(hh, 2), nf(pr / hh + Math.log(hh), 2))]);
  out('ls-out', [T.ls(nf(s, 2), (s - 0.5) ** 2 < (s - 2) ** 2, s / 0.5 + Math.log(0.5) < s / 2 + Math.log(2))]);
}

// ================================================================= 34. S&P 500: which forecast is best? (table)
const FE = {};
const fcE = y => FE[y] || (FE[y] = fcEval(`${y}-01-01`));
function ftBlock() {
  const pr = $('ft-p').value, y = +$('ft-y').value, P = panels('ft-chart', 2, { h: 340 }), [a, b] = P.ax, E = fcE(y), xs = METH.map(m => MLAB[m]); label('ft-y').textContent = String(y);
  const col = METH.map(m => MCOL[m]);
  plot('ft-chart', [on2(a, { type: 'bar', x: xs, y: METH.map(m => E.res[m][pr].mse), marker: { color: col }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    on2(b, { type: 'bar', x: xs, y: METH.map(m => E.res[m][pr].qlike), marker: { color: col }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })],
  layout(Object.assign(P.L, { annotations: [ptitle(a, `MSE, proxy ${pr === 'r2' ? 'r^2' : 'o^2 + Garman-Klass'}, from ${y}`), ptitle(b, 'QLIKE (lower is better)')], margin: { l: 56, r: 12, t: 26, b: nw2('ft-chart') ? 60 : 50 } })));
  const best = (f, p) => METH.reduce((m, z) => (E.res[z][p][f] < E.res[m][p][f] ? z : m), METH[0]);
  const c = (f, p, m, d) => ({ v: nf(E.res[m][p][f], d), c: best(f, p) === m ? 'hi' : '' });
  table('ft-tab', ['', ...METH.map(m => T.h.ml[m])], [['MSE', ...METH.map(m => c('mse', 'r2', m, 2))], ['QLIKE', ...METH.map(m => c('qlike', 'r2', m, 3))], [T.h.mz_r2, ...METH.map(m => nf(E.res[m].r2.mz.r2, 2))],
    [T.h.mz_rg, ...METH.map(m => nf(E.res[m].range.mz.r2, 2))], [T.h.mse_rg, ...METH.map(m => c('mse', 'range', m, 2))], [T.h.ql_rg, ...METH.map(m => c('qlike', 'range', m, 3))]]);
  const d1 = dmTest(`${y}-01-01`, 'ewma94', 'hist21'), d2 = dmTest(`${y}-01-01`, 'vix', 'ewma94');
  out('ft-out', [T.ft(y, fint(E.n), E.first, pr, best('qlike', pr), nf(d1.t, 2), fpe(d1.p), nf(d2.t, 2), fpe(d2.p), y !== 2010)]);
}
const nw2 = id => narrow(id);

// ================================================================= 35. choosing lambda by QLIKE
const LAM = range(0.8, 0.995, 0.005).map(v => +v.toFixed(3)), LQC = {};
function lambdaQ(k, y) {
  const key = k + y; if (LQC[key]) return LQC[key];
  const S = ser(k), first = new Date(S.dates[0] + 'T00:00:00Z'); first.setUTCFullYear(first.getUTCFullYear() + 2);
  const s = [`${y}-01-01`, first.toISOString().slice(0, 10)].sort()[1], i0 = firstFrom(S.dates, s);
  const q = LAM.map(l => { const h = ewmaVar(S.x, l); let z = 0; for (let t = i0; t < S.n; t++) z += S.x[t] ** 2 / h[t] + Math.log(h[t]); return z / (S.n - i0); });
  return (LQC[key] = { q, from: S.dates[i0] });
}
function laBlock() {
  const y = +$('la-y').value, nw = single('la-chart', 360), tr = [], items = []; label('la-y').textContent = String(y);
  ['sp500', 'bet', 'btc'].forEach(k => {
    const Q = lambdaQ(k, y), mn = Math.min(...Q.q), ib = Q.q.indexOf(mn), best = LAM[ib];
    tr.push({ x: LAM, y: Q.q.map(v => v - mn), mode: 'lines', line: { color: COL[k], width: 1.8 }, name: `${NAME[k]} (best lambda = ${best.toFixed(3)})`, legendgroup: k, hovertemplate: `${NAME[k]}<br>lambda = %{x:.3f}<br>%{y:.4f}<extra></extra>` });
    tr.push({ x: [best], y: [0], mode: 'markers', marker: { color: COL[k], size: 9 }, showlegend: false, legendgroup: k, hoverinfo: 'skip' });
    items.push([SH[k], Q.from, nf(best, 3), nf(Q.q[LAM.indexOf(0.94)] - mn, 3)]);
  });
  plot('la-chart', tr, layout({ xaxis: AX('lambda'), yaxis: AX('QLIKE minus its minimum'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 },
    shapes: [{ type: 'line', xref: 'x', x0: 0.94, x1: 0.94, yref: 'paper', y0: 0, y1: 1, line: { color: '#000', width: 0.8, dash: 'dash' } }] }));
  out('la-out', [T.la(y, items, y !== 2010)]);
}

// ================================================================= wiring
const PFX = { 'returns-bursts': 'rb', 'vol-table': 'vt', annual: 'an', 'rolling-windows': 'rw', precision: 'pr', 'ewma-weights': 'ew', 'ewma-update': 'eu', ghost: 'gh', 'ohlc-path': 'op',
  parkinson: 'pk', 'gk-rs': 'gr', yz: 'yz', march16: 'ms', efficiency: 'ef', 'eff-table': 'et', 'est-table': 'es', 'range-rolling': 'rr', 'rv-def': 'rd', realised: 're', signature: 'sg',
  'acf-clustering': 'ac', 'acf-table': 'at', shuffle: 'sh', 'lb-sq': 'lq', 'arch-lm': 'al', 'cl-table': 'ct', kurtosis: 'ku', 'long-term': 'lo', vix: 'vx', 'vix-table': 'vi', leverage: 'lv',
  'news-impact': 'ni', loss: 'ls', 'fc-table': 'ft', 'forecast-paths': 'fp', lambda: 'la' };
const BLOCKS = { rbBlock, vtBlock, anBlock, rwBlock, prBlock, ewBlock, euBlock, ghBlock, opBlock, pkBlock, grBlock, yzBlock, msBlock, efBlock, etBlock, esBlock, rrBlock, rdBlock, reBlock, sgBlock,
  acBlock, atBlock, shBlock, lqBlock, alBlock, ctBlock, kuBlock, loBlock, vxBlock, viBlock, lvBlock, niBlock, lsBlock, ftBlock, fpBlock, laBlock };
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
  const PL = (btn, id, vals, ms) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms));
  const YRS = (a, b, s, back) => range(a, b, s).concat([back]);
  on(['rb-k'], rbBlock); PL('rb-play', 'rb-k', range(3, 10, 1), 900);
  on(['vt-y'], vtBlock); PL('vt-play', 'vt-y', YRS(1994, 2022, 4, 1990), 900);
  on(['an-c', 'an-s'], anBlock); PL('an-play', 'an-c', ['252', '250', '260', '365', 'act'], 1300);
  on(['rw-n'], rwBlock); PL('rw-play', 'rw-n', [5, 10, 42, 63, 126, 21], 1000);
  on(['pr-n', 'pr-s'], prBlock); PL('pr-play', 'pr-n', [10, 42, 63, 126, 252, 21], 1000);
  on(['ew-l', 'ew-n'], ewBlock); PL('ew-play', 'ew-l', range(0.8, 0.99, 0.01), 500);
  on(['eu-l', 'eu-r', 'eu-x'], euBlock); PL('eu-play', 'eu-l', range(0.8, 0.99, 0.01), 500);
  on(['gh-n', 'gh-l'], ghBlock); PL('gh-play', 'gh-n', [21, 42, 84, 105, 126, 63], 1200);
  on(['op-s', 'op-o'], opBlock); PL('op-play', 'op-s', D.seeds.map(String).filter(s => s !== '3').concat(['3']), 1000);
  on(['pk-s', 'pk-y'], pkBlock); PL('pk-play', 'pk-y', YRS(2008, 2026, 2, 2020), 900);
  on(['gr-s', 'gr-y'], grBlock); PL('gr-play', 'gr-y', YRS(2015, 2026, 1, 2021), 800);
  on(['yz-n', 'yz-s'], yzBlock); PL('yz-play', 'yz-n', [2, 5, 10, 42, 63, 126, 21], 1000);
  $('ms-s').addEventListener('change', () => { msDays(); msBlock(); }); on(['ms-d'], msBlock); msDays(); PL('ms-play', 'ms-d', bigDays($('ms-s').value), 1200);
  on(['ef-n', 'ef-s'], efBlock); PL('ef-play', 'ef-n', ['63', '5', '10', '21'], 1400);
  on(['et-n'], etBlock); PL('et-play', 'et-n', ['63', '5', '10', '21'], 1400);
  on(['es-y'], esBlock); PL('es-play', 'es-y', [2010, 2014, 2018, 2020, 2022, 2024, 2006], 1100);
  on(['rr-n'], rrBlock); PL('rr-play', 'rr-n', [5, 10, 42, 63, 21], 1100);
  on(['rd-d', 'rd-m'], rdBlock); PL('rd-play', 'rd-d', range(1, 60, 1), 450);
  on(['re-m'], reBlock); PL('re-play', 're-m', ['1', '2', '3', '10', '15', '30', '65', '130', '390', '5'], 900);
  on(['sg-z'], sgBlock); PL('sg-play', 'sg-z', ['0', '0.005', '0.01', '0.05', '0.1', '0.025'], 1000);
  on(['ac-y'], acBlock); PL('ac-play', 'ac-y', [1994, 1998, 2002, 2006, 2010, 2014, 2016, 2018, 2020, 1990], 1000);
  on(['at-k'], atBlock); PL('at-play', 'at-k', [2, 5, 10, 20, 50, 100, 1], 1000);
  on(['sh-b'], shBlock); PL('sh-play', 'sh-b', ['5', '21', '63', '252', '1'], 1400);
  on(['lq-s', 'lq-m'], lqBlock); PL('lq-play', 'lq-m', range(1, 20, 1), 600);
  on(['al-s', 'al-q'], alBlock); PL('al-play', 'al-q', range(1, 20, 1), 600);
  on(['ct-m', 'ct-q'], ctBlock); PL('ct-play', 'ct-m', [1, 2, 5, 20, 30, 10], 1000);
  on(['ku-s', 'ku-b'], kuBlock);
  $('ku-t').addEventListener('input', () => { const [s, b] = KU_STEPS[+$('ku-t').value]; $('ku-s').value = s; $('ku-b').value = b; kuBlock(); });
  PL('ku-play', 'ku-t', range(0, KU_STEPS.length - 1, 1), 1100);
  on(['lo-y', 'lo-s'], loBlock); PL('lo-play', 'lo-y', YRS(1994, 2018, 4, 1990), 1000);
  on(['vx-h'], vxBlock); PL('vx-play', 'vx-h', ['5', '10', '42', '63', '21'], 1300);
  on(['vi-h', 'vi-y'], viBlock); PL('vi-play', 'vi-y', YRS(1994, 2018, 4, 1990), 1100);
  on(['lv-y', 'lv-m'], lvBlock); PL('lv-play', 'lv-y', YRS(1994, 2018, 4, 1990), 1000);
  on(['ni-h'], niBlock); PL('ni-play', 'ni-h', [0.01, 0.02, 0.03, 0.05, 0.06, 0.08, 0.1, 0.04], 1000);
  on(['ls-p'], lsBlock); PL('ls-play', 'ls-p', [0.25, 0.5, 1.5, 2, 3, 4, 1], 900);
  on(['ft-p', 'ft-y'], ftBlock); PL('ft-play', 'ft-y', YRS(2012, 2022, 2, 2010), 1100);
  on(['fp-y', 'fp-l'], fpBlock); PL('fp-play', 'fp-y', YRS(2008, 2025, 1, 2020), 800);
  on(['la-y'], laBlock); PL('la-play', 'la-y', YRS(2012, 2022, 2, 2010), 1000);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH8 = { ser, fromYear, ppy, moments, acf, ljungBox, archLM, parts, windowEst, rollingEst, yzK, ewmaVar, rollingSd, ghost, burst, winStats, effStats, estTable, rvDays, acfSet, acfPow, shuffled,
  mixK, garchK, annualVol, monthlyRV, vixData, vixStats, leverage, npVol, interp, fcAll, fcEval, dmTest, lambdaQ, bigDays, corr, chi2ppf, chi2sf, Phi, init, PFX, blocks: BLOCKS };
if (window.CH8_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
