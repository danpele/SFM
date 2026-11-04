// Chapter 9 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch9_data.js.
(function () {
'use strict';
const D = window.CH9_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, solve, LANG } = window.SFMI;
const T = window.CH9_TEXT;
const NUM = D.N, COL = D.col, NAME = D.name, MK = D.names, MK4 = ['sp500', 'dax', 'bet', 'btc'];
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';
const SH = { sp500: 'S&P 500', dax: 'DAX', bet: 'BET', btc: 'Bitcoin', tlv: 'TLV', snp: 'SNP' };
const MLAB = { garch: 'GARCH(1,1)', gjr: 'GJR-GARCH(1,1)', egarch: 'EGARCH(1,1)' }, DLAB = { n: 'Normal', t: 't', st: 'skewed t' };
const BURN = 500, SQ2P = Math.sqrt(2 / Math.PI);

// ================================================================= special functions
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
function normppf(p) {               // Acklam, refined by one Halley step
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  let x;
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else { const q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  const e = Phi(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}
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
function betacf(a, b, x) {          // continued fraction of the incomplete beta function (Numerical Recipes)
  const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 3e-16) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
function tcdf(t, v) { const p = 0.5 * betai(v / 2, 0.5, v / (v + t * t)); return t > 0 ? 1 - p : p; }
const tpdfC = v => Math.exp(gammln((v + 1) / 2) - gammln(v / 2)) / Math.sqrt(v * Math.PI);
function tppf(p, v) {               // quantile of the Student-t with v degrees of freedom (Newton from the Normal quantile, with bisection safeguard)
  if (p > 0.5) return -tppf(1 - p, v);
  const k = tpdfC(v); let x = normppf(p), lo = -1e6, hi = 0;
  for (let i = 0; i < 60; i++) {
    const f = tcdf(x, v) - p; if (Math.abs(f) < 1e-14) break;
    if (f > 0) hi = x; else lo = x;
    let xn = x - f / (k * Math.pow(1 + x * x / v, -(v + 1) / 2));
    if (!(xn > lo && xn < hi)) xn = lo < -1e5 ? 2 * x - 1 : (lo + hi) / 2;
    x = xn;
  }
  return x;
}
const stdtq = (p, v) => tppf(p, v) * Math.sqrt((v - 2) / v);      // quantile of the standardised t (variance 1)

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpack(R) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); return dd.map(iso); }
const SER = {};
function ser(k) { return SER[k] || (SER[k] = { x: D.ret[k].x, dates: unpack(D.ret[k]), n: D.ret[k].x.length }); }
const firstFrom = (dates, d) => { const i = dates.findIndex(v => v >= d); return i < 0 ? dates.length : i; };
const ppy = dates => dates.length / ((Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 864e5 / 365.25);

// ================================================================= statistics
const mean = x => { let s = 0; for (const v of x) s += v; return s / x.length; };
const sdev = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / (x.length - 1)); };
const pvar = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return s / x.length; };
function kurt(x) { const m = mean(x); let m2 = 0, m4 = 0; for (const v of x) { const e2 = (v - m) ** 2; m2 += e2; m4 += e2 * e2; } m2 /= x.length; m4 /= x.length; return m4 / (m2 * m2); }
function quantile(a, p) { const b = Float64Array.from(a).sort(), h = (b.length - 1) * p, i = Math.floor(h); return i + 1 < b.length ? b[i] + (h - i) * (b[i + 1] - b[i]) : b[i]; }
const median = a => quantile(a, 0.5);
function acf(x, nlags) {
  const m = mean(x), n = x.length, e = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { e[i] = x[i] - m; s += e[i] * e[i]; }
  const r = []; for (let k = 1; k <= nlags; k++) { let c = 0; for (let t = k; t < n; t++) c += e[t] * e[t - k]; r.push(c / s); } return r;
}
function ljungBox(x, m) { const T = x.length, rho = acf(x, m); let q = 0; rho.forEach((r, i) => { q += T * (T + 2) * r * r / (T - i - 1); }); return { q, p: chi2sf(q, m), crit: chi2ppf(0.95, m) }; }
function olsCols(y, cols) {         // OLS: coefficients, R^2, t statistics (classic SE)
  const n = y.length, k = cols.length, A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let i = 0; i < k; i++) { const ci = cols[i]; let s = 0; for (let t = 0; t < n; t++) s += ci[t] * y[t]; b[i] = s; for (let j = 0; j <= i; j++) { const cj = cols[j]; let z = 0; for (let t = 0; t < n; t++) z += ci[t] * cj[t]; A[i][j] = A[j][i] = z; } }
  const beta = solve(A, b), my = mean(y); let ssr = 0, sst = 0;
  for (let t = 0; t < n; t++) { let f = 0; for (let i = 0; i < k; i++) f += beta[i] * cols[i][t]; ssr += (y[t] - f) ** 2; sst += (y[t] - my) ** 2; }
  const s2 = ssr / (n - k), tv = beta.map((bb, i) => { const ei = new Array(k).fill(0); ei[i] = 1; return bb / Math.sqrt(s2 * solve(A, ei)[i]); });
  return { beta, r2: 1 - ssr / sst, n, tv };
}
function archLM(x, q) {             // statsmodels het_arch: x^2 on a constant and q own lags (no demeaning); LM = n R^2 ~ chi2(q)
  const e2 = Float64Array.from(x, v => v * v), n = e2.length, y = e2.slice(q), cols = [new Float64Array(n - q).fill(1)];
  for (let j = 1; j <= q; j++) cols.push(e2.slice(q - j, n - j));
  const o = olsCols(y, cols), lm = y.length * o.r2;
  return { lm, p: chi2sf(lm, q) };
}
function signBias(z, e) {           // Engle-Ng (1993): z_t^2 on 1, S-, S- eps, S+ eps (lagged); joint T R^2 ~ chi2(3)
  const n = z.length - 1, y = new Float64Array(n), c0 = new Float64Array(n).fill(1), c1 = new Float64Array(n), c2 = new Float64Array(n), c3 = new Float64Array(n);
  for (let t = 0; t < n; t++) { y[t] = z[t + 1] ** 2; const el = e[t], s = el < 0 ? 1 : 0; c1[t] = s; c2[t] = s * el; c3[t] = (1 - s) * el; }
  const o = olsCols(y, [c0, c1, c2, c3]), joint = n * o.r2;
  return { sign_t: o.tv[1], neg_t: o.tv[2], pos_t: o.tv[3], joint, p: chi2sf(joint, 3) };
}
function dmTest(la, lb, L = 5) {    // Diebold-Mariano: HAC (Newey-West, Bartlett, L lags) t statistic of the mean of d = la - lb
  const n = la.length, d = new Float64Array(n); for (let i = 0; i < n; i++) d[i] = la[i] - lb[i];
  const m = mean(d), u = d.map(v => v - m); let S = 0; for (let t = 0; t < n; t++) S += u[t] * u[t];
  for (let j = 1; j <= L; j++) { let g = 0; for (let t = j; t < n; t++) g += u[t] * u[t - j]; S += 2 * (1 - j / (L + 1)) * g; }
  const t = m / Math.sqrt(S / n / n);
  return { mean: m, t, p: p2(t) };
}
function ewmaVar(x, lam, init = 250) { // ewma_variance of the chapter: forecast for day t made at t-1, started with the mean of the first 250 r^2
  const n = x.length, h = new Float64Array(n); let m = 0; for (let i = 0; i < init; i++) m += x[i] * x[i]; h[0] = m / init;
  for (let t = 1; t < n; t++) h[t] = lam * h[t - 1] + (1 - lam) * x[t - 1] * x[t - 1];
  return h;
}
const qlike = (px, h) => px / h + Math.log(h);

// ================================================================= fitted models: the recursion of each model (arch package conventions)
const PC = {};
const fkey = (k, m, d, s) => `${k}:${m}:${d}` + (s ? `:${s}` : '');
function path(key) {                // conditional variances, shocks and standardised residuals of a fitted model
  if (PC[key]) return PC[key];
  const F = D.fit[key], [k, m] = key.split(':'), S = ser(k), p = F.p, i0 = F.i0, n = F.n;
  const om = p.omega, a = p['alpha[1]'], b = p['beta[1]'], g = p['gamma[1]'] || 0, e = new Float64Array(n), s2 = new Float64Array(n), z = new Float64Array(n);
  for (let t = 0; t < n; t++) e[t] = S.x[i0 + t] - p.mu;
  s2[0] = F.s0;
  for (let t = 1; t < n; t++) {
    const el = e[t - 1];
    if (m === 'egarch') { const zl = el / Math.sqrt(s2[t - 1]); s2[t] = Math.exp(om + a * (Math.abs(zl) - SQ2P) + g * zl + b * Math.log(s2[t - 1])); }
    else s2[t] = om + (a + (el < 0 ? g : 0)) * el * el + b * s2[t - 1];
  }
  for (let t = 0; t < n; t++) z[t] = e[t] / Math.sqrt(s2[t]);
  const dates = S.dates.slice(i0, i0 + n), x = S.x.slice(i0, i0 + n);
  return (PC[key] = { k, m, n, i0, s2, e, z, x, dates, F, p, A: ppy(dates) });
}
function persOf(F, m) { const p = F.p; return m === 'egarch' ? p['beta[1]'] : p['alpha[1]'] + p['beta[1]'] + 0.5 * (p['gamma[1]'] || 0); }
const IG = 0.9995;                  // the chapter labels alpha + beta >= 0.9995 as IGARCH (no half-life)
const halfLife = pr => (pr > 0 && pr < 1 ? Math.log(0.5) / Math.log(pr) : Infinity);

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const fnum = (x, d) => { const s = nf(x, d), i = s.search(/[.,]/), a = i < 0 ? s : s.slice(0, i), b = i < 0 ? '' : s.slice(i); return a.replace(/\B(?=(\d{3})+(?!\d))/g, sep) + b; };
const fp = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const fpe = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : '= ' + nf(p, 3));
const MON = { ro: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
  en: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] };
const fdate = d => `${+d.slice(8, 10)} ${MON[LANG][+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const DASH = '–';
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
function vstack(id, n, h, gap, shareX) {   // n panels one above the other (time series)
  $(id).style.height = h + 'px';
  const g = gap || 0.06, w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '';
    L['xaxis' + s] = { anchor: 'y' + s, domain: [0, 1] }; if (i && shareX) L['xaxis' + s].matches = 'x';
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
const EPC = { 2008: PURPLE, 2020: ORANGE }, EPL = { 2008: 'Sep 2008 - Mar 2009 (global financial crisis)', 2020: 'Feb - May 2020 (COVID-19)' };
const epShapes = (a, first) => Object.entries(D.episodes).filter(([, [, e]]) => e >= first).map(([k, [s, e]]) =>
  ({ type: 'rect', xref: a.x, yref: a.y + ' domain', x0: s, x1: e, y0: 0, y1: 1, fillcolor: hexA(EPC[k], 0.15), line: { width: 0 }, layer: 'below' }));
const hl = (a, y, color, name, x0, x1, show) => on2(a, { x: [x0, x1], y: [y, y], mode: 'lines', line: { color, width: 1, dash: 'dash' }, name, showlegend: !!show, hoverinfo: 'skip' });
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const kSel = id => $(id).value;
const nameOf = k => SH[k];

// ================================================================= 1. S&P 500: calm years and storms
function sdBetween(S, a, b) { const v = []; S.dates.forEach((d, i) => { if (d >= a && d <= b) v.push(S.x[i]); }); return v.length > 2 ? sdev(v) : NaN; }
function rtBlock() {
  const k = kSel('rt-s'), y = +$('rt-y').value, S = ser(k); label('rt-y').textContent = String(y);
  const P = vstack('rt-chart', 2, narrow('rt-chart') ? 600 : 540, 0.16), [a, b] = P.ax;
  const iy = []; S.dates.forEach((d, i) => { if (+d.slice(0, 4) === y) iy.push(i); });
  const sy = iy.length > 2 ? sdev(iy.map(i => S.x[i])) : NaN, s08 = sdBetween(S, ...D.episodes['2008']), s20 = sdBetween(S, ...D.episodes['2020']), sa = sdev(S.x);
  const tr = [on2(a, { x: S.dates, y: S.x, mode: 'lines', line: { color: COL[k], width: 0.5 }, name: `${NAME[k]}: daily log return (%)`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(a, { x: iy.map(i => S.dates[i]), y: iy.map(i => S.x[i]), mode: 'lines', line: { color: GREEN, width: 0.8 }, name: `${y}: the chosen calm year`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' })];
  Object.keys(D.episodes).forEach(e => tr.push(on2(a, { x: [null], y: [null], mode: 'lines', line: { color: hexA(EPC[e], 0.5), width: 8 }, name: EPL[e], hoverinfo: 'skip' })));
  const lab = [String(y), 'Sep 2008 - Mar 2009', 'Feb - May 2020', 'whole sample'], vals = [sy, s08, s20, sa];
  tr.push(on2(b, { type: 'bar', x: lab, y: vals.map(v => (isFinite(v) ? v : null)), marker: { color: [GREEN, PURPLE, ORANGE, COL[k]] }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }));
  setAx(P, 0, AX('', { type: 'date' }), AX('%')); setAx(P, 1, AX(''), AX('Daily standard deviation (%)', { rangemode: 'tozero' }));
  plot('rt-chart', tr, layout(Object.assign(P.L, { shapes: epShapes(a, S.dates[0]), legend: LEG(narrow('rt-chart'), -0.07), margin: { l: 56, r: 12, t: 14, b: 30 } })));
  let im = 0, iM = 0; S.x.forEach((v, i) => { if (v < S.x[im]) im = i; if (v > S.x[iM]) iM = i; });
  const R = v => (isFinite(v) && isFinite(sy) ? nf(v / sy, 1) : null), F = v => (isFinite(v) ? nf(v, 2) : null);
  out('rt-out', [T.rt(NAME[k], y, fint(S.n), F(sy), F(s08), F(s20), F(sa), R(s08), R(s20), nf(S.x[im], 1), S.dates[im], nf(S.x[iM], 1), S.dates[iM])]);
}

// ================================================================= 2. simulated paths
function simPath(z, om, a, b, n) {  // simulate_garch of the chapter on the exported draws: s2[0] = om/(1 - a - b), burn-in of 500
  const m = z.length, s2 = new Float64Array(m), e = new Float64Array(m);
  s2[0] = om / (1 - a - b); e[0] = Math.sqrt(s2[0]) * z[0];
  for (let t = 1; t < m; t++) { s2[t] = om + a * e[t - 1] ** 2 + b * s2[t - 1]; e[t] = Math.sqrt(s2[t]) * z[t]; }
  return { e: Array.from(e.slice(BURN, BURN + (n || m - BURN))), s: Array.from(s2.slice(BURN, BURN + (n || m - BURN)), Math.sqrt) };
}
const kArch = a => (3 * a * a < 1 ? 3 * (1 - a * a) / (1 - 3 * a * a) : Infinity);
const kGarch = (a, b) => { const c = (a + b) ** 2 + 2 * a * a; return c < 1 ? 3 * (1 - (a + b) ** 2) / (1 - c) : Infinity; };
function smBlock() {
  const s = kSel('sm-s'), a = +$('sm-a').value, b = +$('sm-b').value, Z = D.sim.paths[s]; label('sm-a').textContent = nf(a, 2); label('sm-b').textContent = nf(b, 2);
  const X = [Z.iid, simPath(Z.arch, 1 - a, a, 0).e, simPath(Z.garch, 1 - 0.1 - b, 0.1, b).e], lim = Math.max(...X.map(x => Math.max(...x.map(Math.abs)))) * 1.05;
  const labs = ['i.i.d. Normal(0, 1)', `ARCH(1): omega = ${(1 - a).toFixed(2)}, alpha = ${a.toFixed(2)}`, `GARCH(1,1): omega = ${(0.9 - b).toFixed(2)}, alpha = 0.10, beta = ${b.toFixed(2)}`];
  const P = vstack('sm-chart', 3, narrow('sm-chart') ? 560 : 500, 0.05, true), t = Array.from({ length: 1000 }, (_, i) => i), cols = [BLUE, RED, GREEN];
  const tr = X.map((x, i) => on2(P.ax[i], { x: t, y: x, mode: 'lines', line: { color: cols[i], width: 0.7 }, name: labs[i], hovertemplate: 't = %{x}<br>%{y:.2f}<extra></extra>' }));
  P.ax.forEach((ax, i) => setAx(P, i, AX(i === 2 ? 'time t' : '', { showticklabels: i === 2 }), AX('value', { range: [-lim, lim] })));
  plot('sm-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('sm-chart'), -0.07), margin: { l: 56, r: 12, t: 10, b: 30 } })));
  const K = X.map(kurt), th = [3, kArch(a), kGarch(0.1, b)], f = v => (isFinite(v) ? nf(v, 2) : '∞');
  out('sm-out', [T.sm(s, +s !== D.seed, f(K[0]), f(K[1]), f(K[2]), f(th[0]), f(th[1]), f(th[2]), nf(Math.max(...X[1].map(Math.abs)), 2), nf(Math.max(...X[2].map(Math.abs)), 2), a !== 0.5 || b !== 0.88)]);
}

// ================================================================= 3. likelihood of ARCH(1)
function arch1ll(a, x) { let s = 0; for (let t = 1; t < x.length; t++) { const s2 = (1 - a) + a * x[t - 1] ** 2; s += -0.5 * Math.log(s2) - 0.5 * x[t] ** 2 / s2; } return s; }
function argmax(f, lo, hi) { const g = (Math.sqrt(5) - 1) / 2; let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (let i = 0; i < 100 && b - a > 1e-10; i++) { if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } } return (a + b) / 2; }
function lik1(n, a0) {
  const x = simPath(D.sim.lik1[n], 1 - a0, a0, 0, +n).e, ah = argmax(a => arch1ll(a, x), 0.001, 0.999), h = 1e-4;
  const d2 = (arch1ll(ah + h, x) - 2 * arch1ll(ah, x) + arch1ll(ah - h, x)) / (h * h);
  return { x, ah, se: 1 / Math.sqrt(-d2) };
}
function l1Block() {
  const a0 = +$('l1-a').value, nw = single('l1-chart', 380), G = linspace(0.05, 0.95, 181), tr = [], items = [], se = []; label('l1-a').textContent = nf(a0, 2);
  [['100', BLUE], ['1000', RED]].forEach(([n, c]) => {
    const R = lik1(n, a0), ll = G.map(a => arch1ll(a, R.x)), mx = Math.max(...ll);
    tr.push({ x: G, y: ll.map(v => v - mx), mode: 'lines', line: { color: c, width: 1.6 }, name: `n = ${n}: log-likelihood minus its maximum`, hovertemplate: 'alpha = %{x:.3f}<br>%{y:.2f}<extra></extra>' });
    tr.push({ x: [R.ah, R.ah], y: [-40, 2], mode: 'lines', line: { color: c, width: 1, dash: 'dash' }, name: `n = ${n}: maximum at alpha = ${R.ah.toFixed(3)}`, hoverinfo: 'skip' });
    items.push([n, nf(R.ah, 3), nf(R.se, 3)]); se.push(R.se);
  });
  tr.push({ x: [a0, a0], y: [-40, 2], mode: 'lines', line: { color: '#000', width: 1.2, dash: 'dot' }, name: `true alpha = ${a0.toFixed(2)}`, hoverinfo: 'skip' });
  plot('l1-chart', tr, layout({ xaxis: AX('alpha', { range: [0.05, 0.95] }), yaxis: AX('log-likelihood minus maximum', { range: [-40, 2] }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  out('l1-out', [T.l1(nf(a0, 2), items, Math.abs(a0 - 0.5) > 1e-9, nf(se[0] / se[1], 1))]);
}

// ================================================================= 4. likelihood of GARCH(1,1)
const GA = linspace(0.01, 0.30, 59), GB = linspace(0.50, 0.97, 48), LG = {};
function likGrid(seed, n) {
  const key = seed + ':' + n; if (LG[key]) return LG[key];
  const x = simPath(D.sim.likg[seed], 0.1, 0.1, 0.8, n).e, T0 = x.length, s2bar = pvar(x), x2 = x.map(v => v * v);
  let best = -Infinity, bi = 0, bj = 0; const Z = GB.map(() => new Array(GA.length).fill(null));
  GB.forEach((b, i) => GA.forEach((a, j) => {
    if (a + b >= 0.995) return;
    const om = (1 - a - b) * s2bar; let s2 = s2bar, L = 0;
    for (let t = 1; t < T0; t++) { s2 = om + a * x2[t - 1] + b * s2; L += -0.5 * Math.log(s2) - 0.5 * x2[t] / s2; }
    Z[i][j] = L; if (L > best) { best = L; bi = i; bj = j; }
  }));
  const tv = (() => { const om = 0.1 * s2bar; let s2 = s2bar, L = 0; for (let t = 1; t < T0; t++) { s2 = om + 0.1 * x2[t - 1] + 0.8 * s2; L += -0.5 * Math.log(s2) - 0.5 * x2[t] / s2; } return L; })();
  return (LG[key] = { Z: Z.map(r => r.map(v => (v === null ? null : v - best))), a: GA[bj], b: GB[bi], dTrue: tv - best });
}
const LGC = [PURPLE, BLUE, TEAL, GREEN, AMBER, ORANGE, RED], LGL = [-40, -20, -10, -5, -3, -2, -1];
function lgBlock() {
  const seed = kSel('lg-s'), nl = +kSel('lg-n'), P = panels('lg-chart', 2, { h: 420, hs: 330 }), tr = [], ann = [], items = [];
  [[nl, P.ax[0]], [2000, P.ax[1]]].forEach(([n, ax], pi) => {
    const R = likGrid(seed, n);
    LGL.forEach((v, li) => tr.push(on2(ax, { type: 'contour', x: GA, y: GB, z: R.Z, autocontour: false, contours: { coloring: 'lines', start: v, end: v, size: 1, showlabels: true, labelfont: { size: 9, color: LGC[li] } },
      line: { color: LGC[li], width: 1.1 }, colorscale: [[0, LGC[li]], [1, LGC[li]]], showscale: false, showlegend: false, hoverinfo: 'skip', connectgaps: false })));
    tr.push(on2(ax, { x: [R.a], y: [R.b], mode: 'markers', marker: { color: RED, size: 10 }, name: 'maximum on the grid', showlegend: pi === 0, legendgroup: 'm', hovertemplate: 'maximum: alpha = %{x:.3f}, beta = %{y:.2f}<extra></extra>' }));
    tr.push(on2(ax, { x: [0.1], y: [0.8], mode: 'markers', marker: { color: '#000', size: 10, symbol: 'x' }, name: 'true values: alpha = 0.1, beta = 0.8', showlegend: pi === 0, legendgroup: 't', hoverinfo: 'skip' }));
    tr.push(on2(ax, { x: GA, y: GA.map(a => 1 - a), mode: 'lines', line: { color: PURPLE, dash: 'dash', width: 1.2 }, name: 'alpha + beta = 1 (IGARCH boundary)', showlegend: pi === 0, legendgroup: 'i', hoverinfo: 'skip' }));
    setAx(P, pi, AX('alpha', { range: [0.01, 0.30] }), AX(pi ? '' : 'beta', { range: [0.50, 0.97] }));
    ann.push(ptitle(ax, `n = ${n}: maximum at alpha = ${R.a.toFixed(3)}, beta = ${R.b.toFixed(2)}`));
    items.push([fint(n), nf(R.a, 3), nf(R.b, 2), nf(-R.dTrue, 1)]);
  });
  plot('lg-chart', tr, layout(Object.assign(P.L, { annotations: ann, legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  out('lg-out', [T.lg(seed, +seed !== D.likg_seeds[0] || nl !== 500, items)]);
}

// ================================================================= 5. QQ plots
const QQC = {};
function qqData(k) {
  if (QQC[k]) return QQC[k];
  const zn = Float64Array.from(path(fkey(k, 'garch', 'n')).z).sort(), Pt = path(fkey(k, 'garch', 't')), zt = Float64Array.from(Pt.z).sort(), n = zn.length, nu = Pt.p.nu;
  const keep = []; for (let i = 0; i < n; i++) if (i < 300 || i >= n - 300 || i % 6 === 0) keep.push(i);
  const u = keep.map(i => (i + 0.5) / n);
  return (QQC[k] = { keep, zn, zt, nu, thn: u.map(normppf), tht: u.map(p => stdtq(p, nu)), qn: quantile(zn, 0.001), qt: quantile(zt, 0.001), thn1: normppf(0.001), tht1: stdtq(0.001, nu), raw1: tppf(0.001, nu) });
}
function qqBlock() {
  const k = kSel('qq-s'), Q = qqData(k), P = panels('qq-chart', 2, { h: 400, hs: 320 }), tr = [], ann = [];
  [[Q.thn, Q.zn, BLUE, 'GARCH(1,1)-N residuals against the Normal'], [Q.tht, Q.zt, RED, `GARCH(1,1)-t residuals against the standardised t(${Q.nu.toFixed(1)})`]].forEach(([th, z, c, lab], i) => {
    const ax = P.ax[i], zz = Q.keep.map(j => z[j]), lo = Math.min(th[0], zz[0]), hi = Math.max(th[th.length - 1], zz[zz.length - 1]);
    tr.push(on2(ax, { x: th, y: zz, mode: 'markers', marker: { color: hexA(c, 0.6), size: 4 }, name: lab, hovertemplate: 'theoretical %{x:.2f}<br>sample %{y:.2f}<extra></extra>' }));
    tr.push(on2(ax, { x: [lo, hi], y: [lo, hi], mode: 'lines', line: { color: '#000', width: 1, dash: 'dash' }, name: '45-degree line', showlegend: i === 0, hoverinfo: 'skip' }));
    setAx(P, i, AX('theoretical quantile'), AX(i ? '' : 'sample quantile'));
    ann.push(ptitle(ax, `${NAME[k]}, ${i ? 'GARCH-t' : 'GARCH-N'}`));
  });
  plot('qq-chart', tr, layout(Object.assign(P.L, { annotations: ann, legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  out('qq-out', [T.qq(NAME[k], k !== 'sp500', nf(Q.qn, 2), nf(Q.thn1, 2), nf(Q.nu, 2), nf(Q.qt, 2), nf(Q.tht1, 2), nf(Q.raw1, 2))]);
}

// ================================================================= 6-7. conditional volatility
function volFit(k, d, si) { return path(fkey(k, 'garch', d, si ? D.starts[k][si - 1] : '')); }
function volStats(P) {
  const ann = Math.sqrt(P.A), v = Array.from(P.s2, s => Math.sqrt(s) * ann), r = { v, sample: sdev(P.x) * ann, last: v[v.length - 1], median: median(v) };
  let im = 0; v.forEach((z, i) => { if (z > v[im]) im = i; }); r.max = v[im]; r.dmax = P.dates[im];
  Object.entries(D.episodes).forEach(([e, [a, b]]) => { let j = -1; P.dates.forEach((d, i) => { if (d >= a && d <= b && (j < 0 || v[i] > v[j])) j = i; }); r['p' + e] = j < 0 ? null : v[j]; r['d' + e] = j < 0 ? null : P.dates[j]; });
  return r;
}
function volBlock(pre, mks) {
  const d = kSel(pre + '-d'), si = +kSel(pre + '-y'), P = vstack(pre + '-chart', 2, narrow(pre + '-chart') ? 600 : 540, 0.12), tr = [], shapes = [], items = [], med = [];
  mks.forEach((k, i) => {
    const F = volFit(k, d, si), V = volStats(F), ax = P.ax[i];
    tr.push(on2(ax, { x: F.dates, y: V.v, mode: 'lines', line: { color: COL[k], width: 0.9 }, name: `${NAME[k]}: GARCH(1,1)-${d === 'n' ? 'N' : d === 't' ? 't' : 'skewed t'} volatility, annualised (%)`, hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>' }));
    tr.push(hl(ax, V.sample, '#000', 'sample volatility of each series, annualised (%)', F.dates[0], F.dates[F.n - 1], i === 0));
    shapes.push(...epShapes(ax, F.dates[0]));
    setAx(P, i, AX('', { type: 'date', range: [ser(mks[0]).dates[0] < ser(mks[1]).dates[0] ? ser(mks[0]).dates[0] : ser(mks[1]).dates[0], D.end] }), AX('% per year', { rangemode: 'tozero' }));
    const f0 = v => (v === null ? null : nf(v, 0));
    items.push([NAME[k], F.dates[0], f0(V.p2008), V.d2008, f0(V.p2020), V.d2020, nf(V.median, 0), nf(V.last, 0), nf(V.sample, 1)]); med.push(V.median);
  });
  Object.keys(D.episodes).forEach(e => tr.push({ x: [null], y: [null], mode: 'lines', line: { color: hexA(EPC[e], 0.5), width: 8 }, name: EPL[e], hoverinfo: 'skip' }));
  plot(pre + '-chart', tr, layout(Object.assign(P.L, { shapes, legend: LEG(narrow(pre + '-chart'), -0.07), margin: { l: 56, r: 12, t: 10, b: 30 } })));
  const ratio = med[1] / med[0];
  out(pre + '-out', [T.vol(T.h.dist[d], d !== 't' || si > 0, items, nf(ratio, 1))]);
}
const v1Block = () => volBlock('v1', ['sp500', 'dax']), v2Block = () => volBlock('v2', ['bet', 'btc']);

// ================================================================= 8. how long does a shock last?
const PSM = { 'garch:t': ['garch', 't'], 'garch:n': ['garch', 'n'], 'gjr:t': ['gjr', 't'], 'egarch:t': ['egarch', 't'] };
function psBlock() {
  const mk = kSel('ps-m'), [m, d] = PSM[mk], H = +$('ps-h').value, nw = single('ps-chart', 400), h = range(0, H, 1), tr = [], items = []; label('ps-h').textContent = String(H);
  MK.forEach(k => {
    const pr = Math.min(persOf(D.fit[fkey(k, m, d)], m), 1), ig = pr >= IG, hl0 = halfLife(pr);
    tr.push({ x: h, y: h.map(i => pr ** i), mode: 'lines', line: { color: COL[k], width: 1.5 }, name: ig ? `${NAME[k]}: persistence = 1 (IGARCH), no half-life` : `${NAME[k]}: persistence = ${pr.toFixed(3)}, half-life ${Math.round(hl0)} days`, hovertemplate: `${NAME[k]}<br>h = %{x}<br>%{y:.3f}<extra></extra>` });
    items.push([SH[k], nf(pr, 4), ig ? null : nf(hl0, 0)]);
  });
  tr.push({ x: [0, H], y: [0.5, 0.5], mode: 'lines', line: { color: '#000', width: 1, dash: 'dot' }, name: 'half of the shock', hoverinfo: 'skip' });
  plot('ps-chart', tr, layout({ xaxis: AX('days after the shock, h'), yaxis: AX('share of the shock left', { range: [0, 1.02] }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  out('ps-out', [T.ps(`${MLAB[m]}-${d === 'n' ? 'N' : 't'}`, mk !== 'garch:t', m, items)]);
}

// ================================================================= 9. news impact curves
function nicCurve(F, m, eps, v) {   // sigma_t^2 as a function of eps_{t-1}, with sigma_{t-1}^2 = v
  const p = F.p, a = p['alpha[1]'], b = p['beta[1]'], g = p['gamma[1]'] || 0;
  if (m === 'egarch') return eps.map(e => { const z = e / Math.sqrt(v); return Math.exp(p.omega + a * (Math.abs(z) - SQ2P) + g * z + b * Math.log(v)); });
  return eps.map(e => p.omega + (a + (e < 0 ? g : 0)) * e * e + b * v);
}
const KN = {};
function kernelNic(k) {
  if (KN[k]) return KN[k];
  const x = ser(k).x, n = x.length - 1, lo = quantile(x, 0.01), hi = quantile(x, 0.99), g = linspace(lo, hi, 201), xl = x.slice(0, n), h = 0.5 * Math.sqrt(pvar(xl));
  const y = g.map(v => { let sw = 0, sy = 0; for (let t = 0; t < n; t++) { const u = (v - xl[t]) / h, w = Math.exp(-0.5 * u * u); sw += w; sy += w * x[t + 1] ** 2; } return sy / sw; });
  return (KN[k] = { g, y, h, s2bar: sdev(x) ** 2 });
}
function niBlock() {
  const kr = kSel('ni-s'), mult = +$('ni-v').value, P = panels('ni-chart', 2, { h: 400, hs: 320 }), tr = [], ann = [], items = []; label('ni-v').textContent = nf(mult, 2);
  ['sp500', kr].forEach((k, i) => {
    const K = kernelNic(k), v = mult * K.s2bar, ax = P.ax[i];
    [['garch', 'GARCH(1,1)-t', BLUE], ['gjr', 'GJR-GARCH(1,1)-t', RED], ['egarch', 'EGARCH(1,1)-t', GREEN]].forEach(([m, lab, c]) =>
      tr.push(on2(ax, { x: K.g, y: nicCurve(D.fit[fkey(k, m, 't')], m, K.g, v), mode: 'lines', line: { color: c, width: 1.6 }, name: lab, showlegend: i === 0, legendgroup: m, hovertemplate: `${lab}<br>shock %{x:.2f}%<br>%{y:.2f}<extra></extra>` })));
    tr.push(on2(ax, { x: K.g, y: K.y, mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dash' }, name: 'kernel estimate of E[r(t)^2 | r(t-1)]', showlegend: i === 0, legendgroup: 'k', hovertemplate: 'kernel<br>%{x:.2f}%<br>%{y:.2f}<extra></extra>' }));
    setAx(P, i, AX('shock yesterday, eps(t-1) (%)'), AX(i ? '' : 'variance today (%^2)'));
    ann.push(ptitle(ax, NAME[k]));
    const r = m => { const c = nicCurve(D.fit[fkey(k, m, 't')], m, [-2, 2], v); return c[0] / c[1]; };
    items.push([NAME[k], nf(r('gjr'), 2), nf(r('egarch'), 2)]);
  });
  plot('ni-chart', tr, layout(Object.assign(P.L, { annotations: ann, legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  out('ni-out', [T.ni(nf(mult, 2), mult !== 1 || kr !== 'btc', items)]);
}

// ================================================================= 10. what GARCH removes (ACF)
function adBlock() {
  const k = kSel('ad-s'), mk = kSel('ad-m'), [m, d] = PSM[mk], K = +$('ad-k').value, nw = single('ad-chart', 380); label('ad-k').textContent = String(K);
  const S = ser(k), P = path(fkey(k, m, d)), a1 = acf(S.x.map(v => v * v), K), a2 = acf(Array.from(P.z, v => v * v), K), band = 1.96 / Math.sqrt(S.n), lags = range(1, K, 1);
  const lab = `${MLAB[m]}-${d === 'n' ? 'N' : 't'}`;
  plot('ad-chart', [{ type: 'bar', x: lags.map(l => l - 0.2), y: a1, width: 0.4, marker: { color: COL[k] }, name: `${NAME[k]}: squared returns`, hovertemplate: 'lag %{x:.0f}: %{y:.3f}<extra></extra>' },
    { type: 'bar', x: lags.map(l => l + 0.2), y: a2, width: 0.4, marker: { color: RED }, name: `${NAME[k]}: squared standardised residuals, ${lab}`, hovertemplate: 'lag %{x:.0f}: %{y:.3f}<extra></extra>' },
    { x: [0, K + 1], y: [band, band], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: '95% band, +/- 1.96/sqrt(T)', hoverinfo: 'skip' },
    { x: [0, K + 1], y: [-band, -band], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, showlegend: false, hoverinfo: 'skip' }],
  layout({ barmode: 'overlay', xaxis: AX('lag k (days)'), yaxis: AX('autocorrelation'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const outN = a2.filter(v => Math.abs(v) > band).length;
  out('ad-out', [T.ad(NAME[k], lab, k !== 'sp500' || mk !== 'garch:t', K, nf(a1[0], 3), nf(a1[K - 1], 3), nf(a2[0], 3), nf(Math.max(...a2.map(Math.abs)), 3), nf(band, 3), outN)]);
}

// ================================================================= 11. term structure
function termFc(P, i, H) {          // garch_path_forecast of the chapter, made at the end of day i
  const p = P.p, a = p['alpha[1]'], b = p['beta[1]'], pr = a + b, s2bar = p.omega / (1 - pr), e = P.x[i] - p.mu, s2n = p.omega + a * e * e + b * P.s2[i];
  const f = []; for (let h = 1; h <= H; h++) f.push(s2bar + pr ** (h - 1) * (s2n - s2bar));
  return { f, s2bar, s2n, pr };
}
function tsBlock() {
  const d = kSel('ts-m'), P = path(fkey('sp500', 'garch', d)), i3 = +$('ts-d').value, nw = single('ts-chart', 400), H = 250, h = range(1, H, 1), ann = P.A, tr = [], items = [];
  label('ts-d').textContent = fdate(P.dates[i3]);
  const idx = [...D.term_dates.map(t => P.dates.indexOf(t)), i3], cols = [GREEN, RED, BLUE];
  let s2bar = 0;
  idx.forEach((i, j) => {
    const R = termFc(P, i, H), dl = P.dates[i]; s2bar = R.s2bar; let cs = 0;
    const avg = R.f.map((v, q) => { cs += v; return Math.sqrt(cs / (q + 1) * ann); });
    tr.push({ x: h, y: R.f.map(v => Math.sqrt(v * ann)), mode: 'lines', line: { color: cols[j], width: 1.6 }, name: `forecast made on ${dl}: daily volatility on day t+h`, hovertemplate: 'h = %{x}<br>%{y:.1f}%<extra></extra>' });
    tr.push({ x: h, y: avg, mode: 'lines', line: { color: cols[j], width: 1.2, dash: 'dash' }, name: `forecast made on ${dl}: average volatility over the next h days`, hovertemplate: 'h = %{x}<br>%{y:.1f}%<extra></extra>' });
    items.push([dl, nf(Math.sqrt(R.f[0] * ann), 1), nf(Math.sqrt(R.f[21] * ann), 1), nf(avg[H - 1], 1)]);
  });
  const lr = Math.sqrt(s2bar * ann);
  tr.push({ x: [1, H], y: [lr, lr], mode: 'lines', line: { color: '#000', width: 1, dash: 'dot' }, name: 'long-run volatility', hoverinfo: 'skip' });
  plot('ts-chart', tr, layout({ xaxis: AX('horizon h (trading days)'), yaxis: AX('% per year', { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  out('ts-out', [T.ts(d === 't' ? 'GARCH(1,1)-t' : 'GARCH(1,1)-N', d !== 't' || i3 !== P.n - 1, items, nf(lr, 1))]);
}

// ================================================================= 12. GARCH against EWMA, out of sample (chart and table)
const OC = {};
function oosData(k, lam) {
  const key = k + lam; if (OC[key]) return OC[key];
  const O = D.oos[k], S = ser(k), n = O.garch.length, ew = ewmaVar(S.x, lam), r = S.x.slice(O.i0, O.i0 + n), dates = S.dates.slice(O.i0, O.i0 + n), ewma = ew.slice(O.i0, O.i0 + n);
  const mu = new Float64Array(n), nu = new Float64Array(n);
  O.blocks.forEach(([a, m, v], j) => { const b = j + 1 < O.blocks.length ? O.blocks[j + 1][0] : n; for (let i = a; i < b; i++) { mu[i] = m; nu[i] = v; } });
  return (OC[key] = { r, dates, garch: O.garch, gjr: O.gjr, ewma, mu, nu, n });
}
function feEval(k, lam, y) {
  const O = oosData(k, lam), j0 = firstFrom(O.dates, `${y}-01-01`), L = { garch: [], gjr: [], ewma: [] }, dates = O.dates.slice(j0);
  for (let i = j0; i < O.n; i++) { const px = O.r[i] ** 2; L.garch.push(qlike(px, O.garch[i])); L.gjr.push(qlike(px, O.gjr[i])); L.ewma.push(qlike(px, O.ewma[i])); }
  let c = 0; const cum = L.ewma.map((v, i) => (c += v - L.garch[i]));
  return { n: dates.length, first: dates[0], dates, cum, q: { garch: mean(L.garch), gjr: mean(L.gjr), ewma: mean(L.ewma) }, dm: dmTest(L.garch, L.ewma), dmj: dmTest(L.gjr, L.garch) };
}
function feBlock() {
  const y = +kSel('fe-y'), lam = +$('fe-l').value, nw = single('fe-chart', 360), tr = [], rows = []; label('fe-l').textContent = nf(lam, 2);
  const E = MK4.map(k => feEval(k, lam, y));
  MK4.forEach((k, i) => tr.push({ x: E[i].dates, y: E[i].cum, mode: 'lines', line: { color: COL[k], width: 1.4 }, name: NAME[k], hovertemplate: `${NAME[k]}<br>%{x}<br>%{y:.1f}<extra></extra>` }));
  plot('fe-chart', tr, layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('cumulative loss difference'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 26, b: 30 },
    title: { text: `Cumulative QLIKE(EWMA, lambda = ${lam.toFixed(2)}) - QLIKE(GARCH-t): rising = GARCH-t better`, x: 0, xanchor: 'left', font: { size: 12, color: C.navy } },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0, y1: 0, line: { color: '#000', width: 0.8, dash: 'dash' } }] }));
  MK4.forEach((k, i) => {
    const e = E[i], b = ['garch', 'gjr', 'ewma'].reduce((m, z) => (e.q[z] < e.q[m] ? z : m), 'garch'), c = z => ({ v: nf(e.q[z], 3), c: b === z ? 'hi' : '' });
    rows.push([SH[k], fint(e.n), c('garch'), c('gjr'), c('ewma'), `${nf(e.dm.t, 2)} (${fp(e.dm.p)})`, `${nf(e.dmj.t, 2)} (${fp(e.dmj.p)})`]);
  });
  table('fe-tab', ['', T.h.days, 'QLIKE GARCH-t', 'QLIKE GJR-t', `QLIKE EWMA (λ = ${nf(lam, 2)})`, T.h.dm1, T.h.dm2], rows);
  out('fe-out', [T.fe(y, nf(lam, 2), y !== 2015 || Math.abs(lam - 0.94) > 1e-9, MK4.map((k, i) => [SH[k], nf(E[i].dm.t, 2), fp(E[i].dm.p), E[i].dm.t < -1.96]))]);
}

// ================================================================= 13. VaR 1% through the COVID-19 crash
function varSeries(k, a) {          // one-day VaR of GARCH-t (the chapter's re-estimated mean and nu) and of EWMA-Normal (lambda = 0.94)
  const O = oosData(k, D.lam), zq = normppf(a), qc = {}, vg = new Float64Array(O.n), ve = new Float64Array(O.n);
  for (let i = 0; i < O.n; i++) { const v = O.nu[i]; if (!(v in qc)) qc[v] = stdtq(a, v); vg[i] = -(O.mu[i] + Math.sqrt(O.garch[i]) * qc[v]); ve[i] = -Math.sqrt(O.ewma[i]) * zq; }
  return { O, vg, ve };
}
function vrBlock() {
  const k = kSel('vr-s'), y = +kSel('vr-w'), a = +kSel('vr-a'), nw = single('vr-chart', 380), V = varSeries(k, a), O = V.O;
  const i0 = firstFrom(O.dates, `${y}-07-01`), i1 = firstFrom(O.dates, `${y + 2}-07-01`), ix = range(i0, i1 - 1, 1), d = ix.map(i => O.dates[i]);
  const e1 = ix.filter(i => O.r[i] < -V.vg[i]), e2 = ix.filter(i => O.r[i] < -V.ve[i]), lev = `${+(a * 100).toFixed(1)}%`;
  plot('vr-chart', [{ x: d, y: ix.map(i => O.r[i]), mode: 'lines', line: { color: COL[k], width: 0.7 }, name: `${NAME[k]}: daily log return (%)`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: d, y: ix.map(i => -V.vg[i]), mode: 'lines', line: { color: RED, width: 1.3 }, name: `minus VaR ${lev}, GARCH(1,1)-t`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: d, y: ix.map(i => -V.ve[i]), mode: 'lines', line: { color: GREEN, width: 1.1, dash: 'dash' }, name: `minus VaR ${lev}, EWMA-Normal`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: e1.map(i => O.dates[i]), y: e1.map(i => O.r[i]), mode: 'markers', marker: { color: RED, size: 8 }, name: 'exceedance of the GARCH-t VaR', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: e2.map(i => O.dates[i]), y: e2.map(i => O.r[i]), mode: 'markers', marker: { color: GREEN, size: 10, symbol: 'x' }, name: 'exceedance of the EWMA-Normal VaR', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }],
  layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('%'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('vr-out', [T.vr(NAME[k], fdate(d[0]), fdate(d[d.length - 1]), fint(ix.length), nf(a * 100, a === 0.025 ? 1 : 0) + '%', e1.length, e2.length, nf(ix.length * a, 0), k !== 'sp500' || y !== 2019 || a !== 0.01)]);
}

// ================================================================= x1. worked example: conditional and unconditional variance
const npdf = (x, v) => Math.exp(-x * x / (2 * v)) / Math.sqrt(2 * Math.PI * v);
function mxBlock() {
  const s2 = +$('mx-s').value, p = +$('mx-p').value, c2 = +kSel('mx-c'); label('mx-s').textContent = nf(s2, 1); label('mx-p').textContent = nf(p, 2);
  const v = (1 - p) * c2 + p * s2, K = 3 * ((1 - p) * c2 * c2 + p * s2 * s2) / (v * v), xs = linspace(-8, 8, 321);
  const fm = xs.map(x => (1 - p) * npdf(x, c2) + p * npdf(x, s2)), fn = xs.map(x => npdf(x, v)), P = panels('mx-chart', 2, { h: 340, hs: 260 }), [a, b] = P.ax;
  const tr = [on2(a, { x: xs, y: fm, mode: 'lines', line: { color: RED, width: 1.8 }, name: `mixture: calm sigma^2 = ${c2}, stormy sigma^2 = ${s2}, stormy share ${p.toFixed(2)}`, legendgroup: 'm', hovertemplate: '%{x:.1f}%: %{y:.4f}<extra></extra>' }),
    on2(a, { x: xs, y: fn, mode: 'lines', line: { color: BLUE, width: 1.5, dash: 'dash' }, name: `Normal with the same variance ${v.toFixed(2)}`, legendgroup: 'n', hovertemplate: '%{x:.1f}%: %{y:.4f}<extra></extra>' }),
    on2(b, { x: xs, y: fm, mode: 'lines', line: { color: RED, width: 1.8 }, showlegend: false, legendgroup: 'm', hovertemplate: '%{x:.1f}%: %{y:.2e}<extra></extra>' }),
    on2(b, { x: xs, y: fn, mode: 'lines', line: { color: BLUE, width: 1.5, dash: 'dash' }, showlegend: false, legendgroup: 'n', hovertemplate: '%{x:.1f}%: %{y:.2e}<extra></extra>' })];
  setAx(P, 0, AX('return (%)'), AX('density')); setAx(P, 1, AX('return (%)'), AX('', { type: 'log', exponentformat: 'power' }));
  plot('mx-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(a, 'Density'), ptitle(b, 'Density, log scale (the tails)')], legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const sc = Math.sqrt(c2), ss = Math.sqrt(s2);
  steps('mx-we', [T.mx1(nf(c2, 2), nf(s2, 1), nf(1 - p, 2), nf(p, 2)), T.mx2(nf(1 - p, 2), nf(c2, 2), nf(p, 2), nf(s2, 1), nf(v, 2), nf(Math.sqrt(v), 2)),
    T.mx3(nf(sc, 2), nf(ss, 2), nf((1 - p) * sc + p * ss, 2), nf(Math.sqrt(v), 2)), T.mx4(nf((1 - p) * c2 * c2 + p * s2 * s2, 2), nf(v * v, 2), nf(K, 2))]);
  out('mx-out', [T.mx(nf(v, 2), nf(Math.sqrt(v), 2), nf(K, 2), s2 !== 4.5 || p !== 0.5 || c2 !== 0.5)]);
}

// ================================================================= x2. ARCH(1): properties (worked example)
function a1Block() {
  const a = +$('a1-a').value, eps = +$('a1-e').value, om = +kSel('a1-w'), P = panels('a1-chart', 2, { h: 340, hs: 260 }), [pa, pb] = P.ax; label('a1-a').textContent = nf(a, 2); label('a1-e').textContent = nf(eps, 1);
  const ag = linspace(0, 0.57, 115), Kc = ag.map(kArch), K = kArch(a), es = linspace(-4, 4, 161), s2 = om + a * eps * eps;
  const tr = [on2(pa, { x: ag, y: Kc, mode: 'lines', line: { color: RED, width: 1.6 }, name: 'kurtosis 3(1 - alpha^2)/(1 - 3 alpha^2)', hovertemplate: 'alpha = %{x:.2f}<br>K = %{y:.2f}<extra></extra>' }),
    on2(pa, { x: [1 / Math.sqrt(3), 1 / Math.sqrt(3)], y: [3, 30], mode: 'lines', line: { color: PURPLE, dash: 'dash', width: 1.2 }, name: 'alpha = 1/sqrt(3) = 0.577: infinite fourth moment', hoverinfo: 'skip' }),
    on2(pa, { x: [Math.min(a, 0.6)], y: [isFinite(K) ? Math.min(K, 30) : 30], mode: 'markers', marker: { color: DARK, size: 10, symbol: isFinite(K) ? 'circle' : 'triangle-up' }, name: `chosen alpha = ${a.toFixed(2)}`, hovertemplate: 'alpha = %{x:.2f}<extra></extra>' }),
    on2(pb, { x: es, y: es.map(e => om + a * e * e), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: `sigma_t^2 = ${om} + ${a.toFixed(2)} eps(t-1)^2`, hovertemplate: 'eps = %{x:.1f}<br>%{y:.2f}<extra></extra>' }),
    on2(pb, { x: [eps], y: [s2], mode: 'markers', marker: { color: RED, size: 10 }, name: `shock eps(t-1) = ${eps.toFixed(1)}`, hovertemplate: 'eps = %{x:.1f}<br>%{y:.2f}<extra></extra>' })];
  setAx(P, 0, AX('alpha', { range: [0, 0.62] }), AX('kurtosis K', { range: [2, 30] })); setAx(P, 1, AX('shock yesterday, eps(t-1)'), AX('variance today, sigma_t^2', { rangemode: 'tozero' }));
  plot('a1-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(pa, 'Kurtosis with Normal innovations'), ptitle(pb, 'Variance after a shock')], legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  steps('a1-we', [T.a1a(nf(om, 2), nf(a, 2), nf(1 - a, 2), a < 1 ? nf(om / (1 - a), 2) : null),
    isFinite(K) ? T.a1b(nf(a * a, 4), nf(1 - a * a, 4), nf(1 - 3 * a * a, 4), nf(K, 2)) : T.a1b(nf(a * a, 4), null, nf(3 * a * a, 4), null),
    T.a1c(nf(om, 2), nf(a, 2), nf(eps, 1), nf(eps * eps, 2), nf(s2, 2), nf(Math.sqrt(s2), 2))]);
  out('a1-out', [T.a1(nf(a, 2), isFinite(K) ? nf(K, 2) : null, nf(s2, 2), a !== 0.5 || eps !== 2 || om !== 0.5)]);
}

// ================================================================= x3. ARCH(q) and its limits (table)
const crit = (o, c) => (c === 'aic' ? -2 * o.ll + 2 * o.k : -2 * o.ll + o.k * Math.log(o.n));
function aqBlock() {
  const k = kSel('aq-s'), q = +$('aq-q').value, nw = single('aq-chart', 340), A = D.archq[k], G = D.fit[fkey(k, 'garch', 'n')]; label('aq-q').textContent = String(q);
  const qs = range(1, 10, 1), aic = A.map(o => crit(o, 'aic')), bic = A.map(o => crit(o, 'bic')), ga = crit(G, 'aic'), gb = crit(G, 'bic');
  plot('aq-chart', [{ x: qs, y: aic, mode: 'lines+markers', line: { color: BLUE, width: 1.6 }, name: 'ARCH(q): AIC', hovertemplate: 'q = %{x}: %{y:.1f}<extra></extra>' },
    { x: qs, y: bic, mode: 'lines+markers', line: { color: RED, width: 1.6 }, name: 'ARCH(q): BIC', hovertemplate: 'q = %{x}: %{y:.1f}<extra></extra>' },
    { x: [1, 10], y: [ga, ga], mode: 'lines', line: { color: BLUE, dash: 'dash', width: 1.2 }, name: 'GARCH(1,1): AIC', hoverinfo: 'skip' },
    { x: [1, 10], y: [gb, gb], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.2 }, name: 'GARCH(1,1): BIC', hoverinfo: 'skip' },
    { x: [q, q], y: [aic[q - 1], bic[q - 1]], mode: 'markers', marker: { color: DARK, size: 11, symbol: 'circle-open', line: { width: 2 } }, name: `ARCH(${q}) in the table`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('number of ARCH lags q', { dtick: 1 }), yaxis: AX(`AIC and BIC, ${NAME[k]}`), legend: LEG(nw, -0.18), margin: { l: 64, r: 12, t: 14, b: 40 } }));
  const qq = [...new Set([1, q, 10])], raw = qq.map(i => A[i - 1]).concat([G]), lab = qq.map(i => `ARCH(${i})`).concat(['GARCH(1,1)']);
  const minA = Math.min(...raw.map(o => crit(o, 'aic'))), minB = Math.min(...raw.map(o => crit(o, 'bic')));
  table('aq-tab', [`${NAME[k]}, ${T.h.normalz}`, T.h.params, T.h.ll, 'AIC', 'BIC', T.h.sumalpha], raw.map((o, i) => [lab[i], o.k, fnum(o.ll, 1),
    { v: fnum(crit(o, 'aic'), 1), c: crit(o, 'aic') === minA ? 'hi' : '' }, { v: fnum(crit(o, 'bic'), 1), c: crit(o, 'bic') === minB ? 'hi' : '' }, nf(o === G ? G.p['alpha[1]'] + G.p['beta[1]'] : o.sa, 3)]));
  const qBest = bic.indexOf(Math.min(...bic)) + 1;
  out('aq-out', [T.aq(NAME[k], k !== 'sp500', nf(A[9].sa, 3), nf(G.p['alpha[1]'] + G.p['beta[1]'], 3), fnum(bic[9] - gb, 1), qBest, fnum(bic[qBest - 1] - gb, 1))]);
}

// ================================================================= x4. stationarity, long-run variance and kurtosis (worked example)
function kuBlock() {
  const a = +$('ku-a').value, b = +$('ku-b').value, om = +kSel('ku-w'), P = panels('ku-chart', 2, { h: 340, hs: 260 }), [pa, pb] = P.ax; label('ku-a').textContent = nf(a, 2); label('ku-b').textContent = nf(b, 2);
  const bg = linspace(0.5, 0.99, 99), Kc = bg.map(x => { const k = kGarch(a, x); return isFinite(k) && k < 20 ? k : null; }), K = kGarch(a, b), pr = a + b;
  const lv = bg.map(x => (a + x < 0.999 ? Math.sqrt(252 * om / (1 - a - x)) : null));
  const tr = [on2(pa, { x: bg, y: Kc, mode: 'lines', line: { color: RED, width: 1.6 }, name: `kurtosis K, alpha = ${a.toFixed(2)}`, hovertemplate: 'beta = %{x:.2f}<br>K = %{y:.2f}<extra></extra>' }),
    on2(pa, { x: [0.5, 0.99], y: [3, 3], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: 'Normal distribution: K = 3', hoverinfo: 'skip' }),
    on2(pa, { x: [b], y: [isFinite(K) && K < 20 ? K : 20], mode: 'markers', marker: { color: DARK, size: 10, symbol: isFinite(K) ? 'circle' : 'triangle-up' }, name: `chosen beta = ${b.toFixed(2)}`, hoverinfo: 'skip' }),
    on2(pb, { x: bg, y: lv, mode: 'lines', line: { color: BLUE, width: 1.6 }, name: `long-run volatility sqrt(252 omega/(1 - alpha - beta)), omega = ${om}`, hovertemplate: 'beta = %{x:.2f}<br>%{y:.1f}%<extra></extra>' }),
    on2(pb, { x: [b], y: [pr < 0.999 ? Math.sqrt(252 * om / (1 - pr)) : null], mode: 'markers', marker: { color: DARK, size: 10 }, showlegend: false, hoverinfo: 'skip' })];
  setAx(P, 0, AX('beta'), AX('kurtosis K', { range: [2, 20] })); setAx(P, 1, AX('beta'), AX('% per year', { rangemode: 'tozero' }));
  plot('ku-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(pa, 'Kurtosis with Normal innovations'), ptitle(pb, 'Long-run volatility, annualised')], legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const c = pr * pr + 2 * a * a, st = pr < 1;
  steps('ku-we', [T.ku1(nf(a, 2), nf(b, 2), nf(pr, 2), st),
    st ? T.ku2(nf(om, 2), nf(1 - pr, 2), nf(om / (1 - pr), 2), nf(Math.sqrt(252 * om / (1 - pr)), 1)) : T.ku2(nf(om, 2), nf(1 - pr, 2), null, null),
    T.ku3(nf(pr, 2), nf(a, 2), nf(c, 4), isFinite(K) ? nf(K, 2) : null)]);
  out('ku-out', [T.ku(nf(a, 2), nf(b, 2), isFinite(K) ? nf(K, 2) : null, st ? nf(Math.sqrt(252 * om / (1 - pr)), 1) : null, a !== 0.1 || b !== 0.88 || om !== 0.02)]);
}

// ================================================================= x5. persistence and half-life (worked example)
function hlBlock() {
  const p = +$('hl-p').value, P = panels('hl-chart', 2, { h: 340, hs: 260 }), [pa, pb] = P.ax, h = range(0, 250, 1), H = halfLife(p); label('hl-p').textContent = nf(p, 4);
  const pg = linspace(0.8, 0.999, 200), sp = D.fit['sp500:garch:t'].p;
  const tr = [on2(pa, { x: h, y: h.map(i => p ** i), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: `(alpha + beta)^h, alpha + beta = ${p.toFixed(4)}`, hovertemplate: 'h = %{x}<br>%{y:.3f}<extra></extra>' }),
    on2(pa, { x: [0, 250], y: [0.5, 0.5], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: 'half of the shock', hoverinfo: 'skip' }),
    on2(pa, { x: [Math.min(H, 250)], y: [H <= 250 ? 0.5 : p ** 250], mode: 'markers', marker: { color: RED, size: 10 }, name: `half-life ${H.toFixed(1)} days`, hoverinfo: 'skip' }),
    on2(pb, { x: pg, y: pg.map(halfLife), mode: 'lines', line: { color: PURPLE, width: 1.6 }, name: 'half-life ln 0.5 / ln(alpha + beta)', hovertemplate: 'alpha + beta = %{x:.3f}<br>%{y:.1f} days<extra></extra>' }),
    on2(pb, { x: [p], y: [H], mode: 'markers', marker: { color: RED, size: 10 }, showlegend: false, hoverinfo: 'skip' })];
  setAx(P, 0, AX('days after the shock, h'), AX('share of the shock left', { range: [0, 1.02] })); setAx(P, 1, AX('alpha + beta'), AX('half-life (days)', { type: 'log' }));
  plot('hl-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(pa, 'Decay of a variance shock'), ptitle(pb, 'Half-life against persistence')], legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  steps('hl-we', [T.hl1(nf(p, 4), nf(Math.log(p), 5), nf(Math.log(0.5), 5), nf(H, 1)), T.hl2(nf(1 - p, 4), nf(Math.log(2) / (1 - p), 1)),
    T.hl3(nf(halfLife(0.9), 1), nf(halfLife(0.98), 1), nf(halfLife(0.99), 1), nf(sp['alpha[1]'] + sp['beta[1]'], 4), nf(halfLife(sp['alpha[1]'] + sp['beta[1]']), 1))]);
  out('hl-out', [T.hl(nf(p, 4), nf(H, 1), nf(p ** 21, 3), Math.abs(p - 0.98) > 1e-9)]);
}

// ================================================================= x6. one GARCH(1,1) step (worked example)
const STP = () => (kSel('st-p') === 'deck' ? { om: 0.02, a: 0.10, b: 0.88, mu: 0 } : (p => ({ om: p.omega, a: p['alpha[1]'], b: p['beta[1]'], mu: p.mu }))(D.fit['sp500:garch:t'].p));
function stBlock() {
  const r = +$('st-r').value, s2t = +$('st-s').value, Q = STP(), nw = single('st-chart', 360); label('st-r').textContent = nf(r, 1); label('st-s').textContent = nf(s2t, 1);
  const pr = Q.a + Q.b, sb = Q.om / (1 - pr), e = r - Q.mu, s1 = Q.om + Q.a * e * e + Q.b * s2t, s0 = Q.om + Q.b * s2t, h = range(0, 120, 1);
  const pth = v1 => h.map(i => (i === 0 ? s2t : sb + pr ** (i - 1) * (v1 - sb)));
  plot('st-chart', [{ x: h, y: pth(s1), mode: 'lines', line: { color: RED, width: 1.8 }, name: `after r(t) = ${r.toFixed(1)}%: sigma^2(t+1) = ${s1.toFixed(3)}, then the forecasts`, hovertemplate: 'h = %{x}<br>%{y:.3f}<extra></extra>' },
    { x: h, y: pth(s0), mode: 'lines', line: { color: BLUE, width: 1.4, dash: 'dash' }, name: `after a zero shock: sigma^2(t+1) = ${s0.toFixed(3)}`, hovertemplate: 'h = %{x}<br>%{y:.3f}<extra></extra>' },
    { x: [0, 120], y: [sb, sb], mode: 'lines', line: { color: '#000', width: 1, dash: 'dot' }, name: `long-run variance ${sb.toFixed(3)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('days after today, h'), yaxis: AX('variance (%^2)', { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const sp = kSel('st-p') !== 'deck';
  steps('st-we', [T.st1(sp, nf(Q.om, 4), nf(Q.a, 4), nf(Q.b, 4), nf(Q.mu, 4)), T.st2(nf(pr, 4), nf(Q.om, 4), nf(1 - pr, 4), nf(sb, 3), nf(Math.sqrt(252 * sb), 1), nf(halfLife(pr), 1)),
    T.st3(nf(s2t, 1), nf(r, 1), nf(e, 3), nf(Q.om, 4), nf(Q.a, 4), nf(e * e, 3), nf(Q.b, 4), nf(s1, 3), nf(Math.sqrt(s1), 2)),
    T.st4(nf(s1, 3), nf(sb, 3), nf(pr, 4), nf(sb + pr ** 9 * (s1 - sb), 3))]);
  out('st-out', [T.st(nf(r, 1), nf(s2t, 1), nf(s1, 3), nf(Math.sqrt(s1), 2), nf(s0, 3), sp || r !== -3 || s2t !== 1.2)]);
}

// ================================================================= x7. IGARCH and the EWMA link (worked example)
function igBlock() {
  const lam = +$('ig-l').value, w = +kSel('ig-w'), nw = single('ig-chart', 360), S = ser('sp500'), i = S.dates.indexOf('2020-03-16'), H = 250, h = range(1, H, 1); label('ig-l').textContent = nf(lam, 2);
  const ew = ewmaVar(S.x, lam), s1 = ew[i + 1], P = path('sp500:garch:t'), G = termFc(P, i, H), A = P.A, ann = v => Math.sqrt(v * A);
  const fe = h.map(() => s1), fi = h.map(q => s1 + w * (q - 1));
  plot('ig-chart', [{ x: h, y: fe.map(ann), mode: 'lines', line: { color: GREEN, width: 1.8 }, name: `EWMA, lambda = ${lam.toFixed(2)} (IGARCH with omega = 0)`, hovertemplate: 'h = %{x}<br>%{y:.1f}%<extra></extra>' },
    { x: h, y: fi.map(ann), mode: 'lines', line: { color: ORANGE, width: 1.6, dash: 'dash' }, name: `IGARCH with omega = ${w}, alpha = ${(1 - lam).toFixed(2)}, beta = ${lam.toFixed(2)}`, hovertemplate: 'h = %{x}<br>%{y:.1f}%<extra></extra>' },
    { x: h, y: G.f.map(ann), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: `GARCH(1,1)-t, alpha + beta = ${G.pr.toFixed(4)}`, hovertemplate: 'h = %{x}<br>%{y:.1f}%<extra></extra>' },
    { x: [1, H], y: [ann(G.s2bar), ann(G.s2bar)], mode: 'lines', line: { color: '#000', width: 1, dash: 'dot' }, name: 'GARCH-t long-run volatility', hoverinfo: 'skip' }],
  layout({ xaxis: AX('horizon h (trading days)'), yaxis: AX('forecast volatility, % per year', { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 26, b: 40 },
    title: { text: 'S&P 500: forecasts made at the close of 16 March 2020', x: 0, xanchor: 'left', font: { size: 12, color: C.navy } } }));
  steps('ig-we', [T.ig1(nf(lam, 2), nf(1 - lam, 2)), T.ig2(nf(lam, 2), nf(s1, 2), nf(ann(s1), 1)),
    T.ig3(fnum(w, 4), nf(s1, 2), nf(s1 + w * (H - 1), 2), nf(ann(s1 + w * (H - 1)), 1)),
    T.ig4(nf(G.s2n, 2), nf(G.s2bar, 3), nf(G.pr, 4), nf(G.f[H - 1], 2), nf(ann(G.f[H - 1]), 1))]);
  out('ig-out', [T.ig(nf(lam, 2), nf(ann(s1), 1), nf(ann(G.f[H - 1]), 1), nf(ann(G.s2bar), 1), Math.abs(lam - 0.94) > 1e-9 || w !== 0)]);
}

// ================================================================= x8. estimation step by step and with a package (table)
const PN = ['mu', 'omega', 'alpha[1]', 'beta[1]'], PL = { mu: 'μ', omega: 'ω', 'alpha[1]': 'α', 'beta[1]': 'β' };
function etBlock() {
  const k = kSel('et-s'), S = D.step[k], F = D.fit[fkey(k, 'garch', 'n')], nw = single('et-chart', 320), rat = PN.map(p => F.se[p] / F.sec[p]);
  plot('et-chart', [{ type: 'bar', x: ['mu', 'omega', 'alpha', 'beta'], y: rat, marker: { color: [BLUE, PURPLE, RED, GREEN] }, name: 'robust SE / classic SE', hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: ['mu', 'beta'], y: [1, 1], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: 'ratio 1: the two SE agree', hoverinfo: 'skip' }],
  layout({ xaxis: AX(''), yaxis: AX(`robust SE / classic SE, ${NAME[k]}`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  table('et-tab', [`${NAME[k]}, GARCH(1,1)-N`, T.h.step, 'arch', T.h.secl, T.h.serob, T.h.ratio],
    PN.map((p, i) => [PL[p], `${nf(S.params[p], 4)} (${nf(S.se[p], 4)})`, nf(F.p[p], 4), nf(F.sec[p], 4), nf(F.se[p], 4), nf(rat[i], 2)]));
  out('et-out', [T.et(NAME[k], k !== 'sp500', fnum(S.loglik, 1), fnum(F.ll, 1), nf(rat[2], 2), nf(rat[3], 2), nf(rat[1], 2), nf(rat[0], 2))]);
}

// ================================================================= x9. three innovation distributions (table)
function dtBlock() {
  const k = kSel('dt-s'), P = panels('dt-chart', 2, { h: 320, hs: 250 }), [pa, pb] = P.ax, F = { n: D.fit[fkey(k, 'garch', 'n')], t: D.fit[fkey(k, 'garch', 't')], st: D.fit[fkey(k, 'garch', 'st')] };
  const lr1 = 2 * (F.t.ll - F.n.ll), lr2 = 2 * (F.st.ll - F.t.ll), c = chi2ppf(0.95, 1), pers = ['n', 't', 'st'].map(d => F[d].p['alpha[1]'] + F[d].p['beta[1]']);
  plot('dt-chart', [on2(pa, { type: 'bar', x: ['t vs Normal', 'skewed t vs t'], y: [lr1, lr2], marker: { color: [RED, PURPLE] }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' }),
    on2(pa, { x: ['t vs Normal', 'skewed t vs t'], y: [c, c], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: 'chi2(1) 5% critical value 3.84', hoverinfo: 'skip' }),
    on2(pb, { type: 'bar', x: ['Normal', 'Student-t', 'skewed t'], y: pers, marker: { color: [BLUE, RED, PURPLE] }, showlegend: false, hovertemplate: '%{x}: %{y:.4f}<extra></extra>' })],
  layout(Object.assign(P.L, { annotations: [ptitle(pa, `${NAME[k]}: likelihood-ratio statistics`), ptitle(pb, 'alpha + beta')], yaxis: Object.assign(P.L.yaxis, AX('LR', { type: 'log' })),
    yaxis2: Object.assign(P.L.yaxis2, AX('', { range: [Math.min(...pers) - 0.02, 1.0] })), legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  const row = (lab, f, nu, lam) => [lab, nf(f.p.omega, 4), nf(f.p['alpha[1]'], 3), nf(f.p['beta[1]'], 3), nu === null ? DASH : nf(nu, 2), lam === null ? DASH : nf(lam, 3), fnum(f.ll, 1), nf(f.p['alpha[1]'] + f.p['beta[1]'], 3)];
  table('dt-tab', ['GARCH(1,1)', 'ω', 'α', 'β', 'ν', 'λ', T.h.ll, 'α + β'], [row(T.h.dn, F.n, null, null), row('Student-t', F.t, F.t.p.nu, null), row(T.h.dst, F.st, F.st.p.eta, F.st.p.lambda)]);
  out('dt-out', [T.dt(NAME[k], k !== 'sp500', nf(lr1, 1), nf(lr2, 1), nf(F.t.p.nu, 2), nf(F.st.p.lambda, 3), nf(F.n.p['alpha[1]'], 3), nf(F.t.p['alpha[1]'], 3), nf(F.n.p['beta[1]'], 3), nf(F.t.p['beta[1]'], 3))]);
}

// ================================================================= x10. GARCH(1,1)-t estimates in six markets (table)
function mktRow(k, d) {
  const P = path(fkey(k, 'garch', d)), p = P.p, se = P.F.se, pr = p['alpha[1]'] + p['beta[1]'], ig = pr >= IG;
  return { k, first: P.dates[0], a: p['alpha[1]'], b: p['beta[1]'], sa: se['alpha[1]'], sb: se['beta[1]'], nu: d === 't' ? p.nu : d === 'st' ? p.eta : null, pr, ig, hl: ig ? null : halfLife(pr),
    lr: ig ? null : Math.sqrt(p.omega / (1 - pr) * P.A), sv: sdev(P.x) * Math.sqrt(P.A) };
}
function mtBlock() {
  const d = kSel('mt-d'), R = MK.map(k => mktRow(k, d)), nw = single('mt-chart', 360), ag = linspace(0.05, 0.25, 41);
  const tr = R.map(o => ({ x: [o.a], y: [o.b], mode: 'markers', marker: { color: COL[o.k], size: 10 }, name: NAME[o.k], error_x: { type: 'data', array: [1.96 * o.sa], color: COL[o.k], thickness: 1.2 }, error_y: { type: 'data', array: [1.96 * o.sb], color: COL[o.k], thickness: 1.2 },
    hovertemplate: `${NAME[o.k]}<br>alpha = %{x:.3f}<br>beta = %{y:.3f}<extra></extra>` }));
  tr.push({ x: ag, y: ag.map(a => 1 - a), mode: 'lines', line: { color: PURPLE, dash: 'dash', width: 1.2 }, name: 'alpha + beta = 1 (IGARCH)', hoverinfo: 'skip' });
  plot('mt-chart', tr, layout({ xaxis: AX('alpha (bars: +/- 1.96 robust SE)'), yaxis: AX('beta'), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  table('mt-tab', ['', T.h.from, 'α (SE)', 'β (SE)', 'ν', 'α + β', T.h.hl, T.h.lrvol, T.h.svol], R.map(o => [SH[o.k], o.first.slice(0, 4), `${nf(o.a, 3)} (${nf(o.sa, 3)})`, `${nf(o.b, 3)} (${nf(o.sb, 3)})`,
    o.nu === null ? DASH : nf(o.nu, 2), o.ig ? nf(1, 3) : nf(o.pr, 4), o.ig ? DASH : nf(o.hl, 0), o.ig ? DASH : nf(o.lr, 1), nf(o.sv, 1)]));
  const IX = R.filter(o => ['sp500', 'dax', 'bet'].includes(o.k) && !o.ig), se = R.flatMap(o => [o.sa, o.sb]), ba = R.reduce((m, o) => (o.a > m.a ? o : m), R[0]);
  out('mt-out', [T.mt(T.h.dist[d], d !== 't', nf(Math.min(...IX.map(o => o.pr)), 4), nf(Math.max(...IX.map(o => o.pr)), 4), nf(Math.min(...IX.map(o => o.hl)), 0), nf(Math.max(...IX.map(o => o.hl)), 0),
    nf(Math.min(...se), 3), nf(Math.max(...se), 3), SH[ba.k], nf(ba.a, 3), R.filter(o => o.ig).map(o => SH[o.k]))]);
}

// ================================================================= x11. Bitcoin and the long-run volatility (worked example)
function lrBlock() {
  const k = kSel('lr-s'), P = path(fkey(k, 'garch', 't')), p = P.p, pr = p['alpha[1]'] + p['beta[1]'], ig = pr >= IG, A = P.A, nw = single('lr-chart', 360), sv = sdev(P.x) * Math.sqrt(A);
  const pg = linspace(0.94, 0.999, 119), lv = pg.map(q => Math.sqrt(p.omega / (1 - q) * A)), N = D.fit[fkey(k, 'garch', 'n')].p, pn = N['alpha[1]'] + N['beta[1]'];
  const vn = pn < IG ? Math.sqrt(N.omega / (1 - pn) * A) : null, v0 = ig ? null : Math.sqrt(p.omega / (1 - pr) * A), v2 = ig ? null : Math.sqrt(p.omega / (1 - pr + 0.002) * A);
  plot('lr-chart', [{ x: pg, y: lv, mode: 'lines', line: { color: COL[k], width: 1.6 }, name: `sqrt(A omega/(1 - alpha - beta)) with the GARCH-t omega = ${p.omega.toFixed(4)}`, hovertemplate: 'alpha + beta = %{x:.4f}<br>%{y:.1f}%<extra></extra>' },
    { x: [Math.min(pr, 0.999)], y: [ig ? Math.max(...lv) : v0], mode: 'markers', marker: { color: RED, size: 11, symbol: ig ? 'triangle-up' : 'circle' }, name: ig ? 'estimate: alpha + beta = 1 (IGARCH), no long-run level' : `estimate: alpha + beta = ${pr.toFixed(4)}`, hoverinfo: 'skip' },
    { x: [0.94, 0.999], y: [sv, sv], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: `sample volatility ${sv.toFixed(1)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('alpha + beta'), yaxis: AX('long-run volatility, % per year', { type: 'log' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  steps('lr-we', ig ? [T.lr0(NAME[k], nf(pr, 4), fnum(p.omega, 4), nf(p['beta[1]'], 3))] : [T.lr1(NAME[k], nf(p.omega, 5), nf(pr, 4), nf(1 - pr, 5), nf(p.omega / (1 - pr), 2)),
    T.lr2(nf(A, 1), nf(p.omega / (1 - pr), 2), nf(v0, 1), nf(sv, 1)), T.lr3(nf(pr - 0.002, 4), nf(1 - pr + 0.002, 4), nf(v2, 1)), T.lr4(vn === null ? null : nf(vn, 1), nf(pn, 4))]);
  out('lr-out', [T.lr(NAME[k], ig, ig ? null : nf(v0, 1), nf(sv, 1), k !== 'sp500')]);
}

// ================================================================= x12. GJR-GARCH and EGARCH (worked example)
function gjBlock() {
  const eps = +$('gj-e').value, k = kSel('gj-s'), nw = single('gj-chart', 360), J = D.fit[fkey(k, 'gjr', 't')], G = D.fit[fkey(k, 'garch', 't')], E = D.fit[fkey(k, 'egarch', 't')], sb = sdev(ser(k).x) ** 2; label('gj-e').textContent = nf(eps, 1);
  const es = linspace(-6, 6, 241), cj = nicCurve(J, 'gjr', [-eps, eps], sb), ce = nicCurve(E, 'egarch', [-eps, eps], sb), p = J.p;
  plot('gj-chart', [{ x: es, y: nicCurve(J, 'gjr', es, sb), mode: 'lines', line: { color: RED, width: 1.7 }, name: 'GJR-GARCH(1,1)-t', hovertemplate: 'shock %{x:.1f}%<br>%{y:.2f}<extra></extra>' },
    { x: es, y: nicCurve(E, 'egarch', es, sb), mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: 'EGARCH(1,1)-t', hovertemplate: 'shock %{x:.1f}%<br>%{y:.2f}<extra></extra>' },
    { x: es, y: nicCurve(G, 'garch', es, sb), mode: 'lines', line: { color: BLUE, width: 1.2, dash: 'dot' }, name: 'GARCH(1,1)-t (symmetric)', hovertemplate: 'shock %{x:.1f}%<br>%{y:.2f}<extra></extra>' },
    { x: [-eps, eps], y: cj, mode: 'markers', marker: { color: DARK, size: 10 }, name: `GJR: shocks -${eps.toFixed(1)}% and +${eps.toFixed(1)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('shock yesterday, eps(t-1) (%)'), yaxis: AX(`${NAME[k]}: variance today (%^2)`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  steps('gj-we', [T.gj1(NAME[k], nf(p.omega, 4), nf(p['alpha[1]'], 3), nf(p['gamma[1]'], 3), nf(p['beta[1]'], 3), nf(sb, 3)),
    T.gj2(nf(eps, 1), nf(p.omega, 4), nf(p['alpha[1]'] + p['gamma[1]'], 3), nf(eps * eps, 2), nf(p['beta[1]'], 3), nf(sb, 3), nf(cj[0], 3)),
    T.gj3(nf(eps, 1), nf(p.omega, 4), nf(p['alpha[1]'], 3), nf(eps * eps, 2), nf(p['beta[1]'], 3), nf(sb, 3), nf(cj[1], 3)),
    T.gj4(nf(cj[0] / cj[1], 2), nf(ce[0] / ce[1], 2))]);
  out('gj-out', [T.gj(NAME[k], nf(eps, 1), nf(cj[0] / cj[1], 2), nf(ce[0] / ce[1], 2), k !== 'sp500' || eps !== 2)]);
}

// ================================================================= x13. asymmetry in six markets (table)
function atRow(k, d) {
  const G = D.fit[fkey(k, 'garch', d)], J = D.fit[fkey(k, 'gjr', d)], E = D.fit[fkey(k, 'egarch', d)], lr = 2 * (J.ll - G.ll), bic = [['GARCH', G], ['GJR', J], ['EGARCH', E]].map(([m, f]) => [m, crit(f, 'bic')]);
  return { k, ja: J.p['alpha[1]'], jg: J.p['gamma[1]'], jt: J.p['gamma[1]'] / J.se['gamma[1]'], jse: J.se['gamma[1]'], lr, p: chi2sf(lr, 1), eg: E.p['gamma[1]'], et: E.p['gamma[1]'] / E.se['gamma[1]'], ese: E.se['gamma[1]'],
    best: bic.reduce((m, z) => (z[1] < m[1] ? z : m))[0], lnT: Math.log(G.n) };
}
function atBlock() {
  const d = kSel('at-d'), R = MK.map(k => atRow(k, d)), P = panels('at-chart', 2, { h: 340, hs: 260 }), [pa, pb] = P.ax, xs = R.map(o => SH[o.k]);
  plot('at-chart', [on2(pa, { x: xs, y: R.map(o => o.jg), mode: 'markers', marker: { color: R.map(o => COL[o.k]), size: 10 }, error_y: { type: 'data', array: R.map(o => 1.96 * o.jse), color: DARK, thickness: 1.2 }, showlegend: false, hovertemplate: '%{x}: gamma = %{y:.3f}<extra></extra>' }),
    on2(pb, { x: xs, y: R.map(o => o.eg), mode: 'markers', marker: { color: R.map(o => COL[o.k]), size: 10, symbol: 'diamond' }, error_y: { type: 'data', array: R.map(o => 1.96 * o.ese), color: DARK, thickness: 1.2 }, showlegend: false, hovertemplate: '%{x}: gamma = %{y:.3f}<extra></extra>' }),
    on2(pa, { x: [xs[0], xs[5]], y: [0, 0], mode: 'lines', line: { color: '#000', width: 0.8 }, showlegend: false, hoverinfo: 'skip' }), on2(pb, { x: [xs[0], xs[5]], y: [0, 0], mode: 'lines', line: { color: '#000', width: 0.8 }, showlegend: false, hoverinfo: 'skip' })],
  layout(Object.assign(P.L, { annotations: [ptitle(pa, `GJR gamma (> 0: leverage), ${DLAB[d]} innovations`), ptitle(pb, `EGARCH gamma (< 0: leverage), ${DLAB[d]} innovations`)], showlegend: false, margin: { l: 56, r: 12, t: 26, b: 30 } })));
  table('at-tab', ['', 'GJR α', 'GJR γ', 't(γ)', 'LR', 'p', 'EGARCH γ', 't(γ)', T.h.bicbest], R.map(o => [SH[o.k], nf(o.ja, 3), nf(o.jg, 3), nf(o.jt, 1), nf(o.lr, 1), fp(o.p), nf(o.eg, 3), nf(o.et, 1), o.best]));
  const b = R.find(o => o.k === 'bet');
  out('at-out', [T.at(T.h.dist[d], d !== 't', R.filter(o => o.p < 0.05 && o.jg > 0).map(o => SH[o.k]), R.filter(o => o.best === 'GARCH').map(o => SH[o.k]), nf(b.lr, 1), nf(b.lnT, 1))]);
}

// ================================================================= x14. diagnostics on real data (table)
const DGC = {};
function dgCalc(k, m) {
  const key = k + m; if (DGC[key]) return DGC[key];
  const S = ser(k), N = path(fkey(k, 'garch', 'n')), Tt = path(fkey(k, 'garch', 't')), sq = x => Array.from(x, v => v * v);
  const R = { r: S.x, r2: sq(S.x), zn: N.z, zn2: sq(N.z), zt: Tt.z, zt2: sq(Tt.z) }, lb = {};
  Object.entries(R).forEach(([n, x]) => { lb[n] = ljungBox(x, m); });
  return (DGC[key] = { lb, lm: { r: archLM(S.x, 5), zn: archLM(N.z, 5), zt: archLM(Tt.z, 5) }, sb: signBias(Tt.z, Tt.e) });
}
function dgBlock() {
  const k = kSel('dg-s'), m = +$('dg-m').value, nw = single('dg-chart', 340), Q = dgCalc(k, m), keys = ['r', 'r2', 'zn', 'zn2', 'zt', 'zt2']; label('dg-m').textContent = String(m);
  const xl = ['r', 'r^2', 'z, GARCH-N', 'z^2, GARCH-N', 'z, GARCH-t', 'z^2, GARCH-t'], c = Q.lb.r.crit;
  plot('dg-chart', [{ type: 'bar', x: xl, y: keys.map(n => Q.lb[n].q), marker: { color: [BLUE, BLUE, PURPLE, PURPLE, RED, RED].map((cc, i) => (i % 2 ? cc : hexA(cc, 0.5))) }, name: `Ljung-Box Q(${m})`, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { x: [xl[0], xl[5]], y: [c, c], mode: 'lines', line: { color: '#000', dash: 'dash', width: 1 }, name: `5% critical value of chi2(${m}): ${c.toFixed(1)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX(''), yaxis: AX(`${NAME[k]}: Q(${m}), log scale`, { type: 'log' }), legend: LEG(nw, -0.2), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const qp = o => `${fnum(o.q, 1)} (${fp(o.p)})`, lmp = o => `${fnum(o.lm, 1)} (${fp(o.p)})`;
  table('dg-tab', [NAME[k], `Q(${m}) (p)`, `Q(${m}) ${T.h.sq} (p)`, 'ARCH-LM(5) (p)', `${T.h.sb} χ²(3) (p)`],
    [[T.h.ret, qp(Q.lb.r), qp(Q.lb.r2), lmp(Q.lm.r), DASH], [T.h.zn, qp(Q.lb.zn), qp(Q.lb.zn2), lmp(Q.lm.zn), DASH], [T.h.zt, qp(Q.lb.zt), qp(Q.lb.zt2), lmp(Q.lm.zt), `${fnum(Q.sb.joint, 1)} (${fp(Q.sb.p)})`]]);
  out('dg-out', [T.dg(NAME[k], m, k !== 'sp500' || m !== 10, fnum(Q.lb.r2.q, 1), fnum(Q.lb.zt2.q, 1), fp(Q.lb.zt2.p), nf(Q.sb.sign_t, 2), fp(Q.sb.p), fp(Q.lb.zt.p))]);
}

// ================================================================= x15. nine models (table)
const MSM = ['garch', 'gjr', 'egarch'], MSD = ['n', 't', 'st'];
function msBlock() {
  const k = kSel('ms-s'), c = kSel('ms-c'), nw = single('ms-chart', 340), V = MSM.map(m => MSD.map(d => crit(D.fit[fkey(k, m, d)], c))), mn = Math.min(...V.flat());
  plot('ms-chart', MSD.map((d, j) => ({ type: 'bar', x: MSM.map(m => MLAB[m]), y: MSM.map((m, i) => V[i][j] - mn), name: `${DLAB[d]} innovations`, marker: { color: [BLUE, RED, PURPLE][j] }, hovertemplate: `%{x}, ${DLAB[d]}: %{y:.1f}<extra></extra>` })),
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`${NAME[k]}: ${c.toUpperCase()} minus the smallest ${c.toUpperCase()}`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const D0 = (i, j) => ({ v: nf(V[i][j] - mn, 1), c: V[i][j] === mn ? 'hi' : '' });
  table('ms-tab', [['', { v: 'Normal', span: 2 }, { v: 'Student-t', span: 2 }, { v: T.h.dst, span: 2 }], ['', T.h.ll, 'Δ' + c.toUpperCase(), T.h.ll, 'Δ' + c.toUpperCase(), T.h.ll, 'Δ' + c.toUpperCase()]],
    MSM.map((m, i) => [MLAB[m], ...MSD.flatMap((d, j) => [fnum(D.fit[fkey(k, m, d)].ll, 1), D0(i, j)])]));
  let bi = 0, bj = 0; V.forEach((r, i) => r.forEach((v, j) => { if (v === mn) { bi = i; bj = j; } }));
  out('ms-out', [T.ms(NAME[k], c.toUpperCase(), k !== 'sp500' || c !== 'bic', MLAB[MSM[bi]], T.h.dist[MSD[bj]], nf(V[0][0] - V[2][0], 1), nf(V[0][0] - V[0][2], 1), nf(V[0][0] - mn, 1))]);
}

// ================================================================= x16. forecasting the S&P 500 (worked example)
function fwBlock() {
  const H = +$('fw-h').value, dt = kSel('fw-d'), P = path('sp500:garch:t'), i = P.dates.indexOf(dt), R = termFc(P, i, 250), Pn = panels('fw-chart', 2, { h: 340, hs: 260 }), [pa, pb] = Pn.ax; label('fw-h').textContent = String(H);
  const hs = range(1, 250, 1), cum = []; let s = 0; R.f.forEach(v => cum.push(s += v));
  const tr = [on2(pa, { x: range(1, H, 1), y: R.f.slice(0, H), mode: 'lines+markers', marker: { size: H > 40 ? 3 : 6 }, line: { color: RED, width: 1.5 }, name: 'E_t[sigma^2(t+h)], h = 1, ..., H', hovertemplate: 'h = %{x}<br>%{y:.3f}<extra></extra>' }),
    on2(pa, { x: [1, H], y: [R.s2bar, R.s2bar], mode: 'lines', line: { color: '#000', dash: 'dot', width: 1 }, name: `long-run variance ${R.s2bar.toFixed(3)}`, hoverinfo: 'skip' }),
    on2(pb, { x: hs, y: cum.map(Math.sqrt), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: 'H-day volatility: sqrt of the sum of the forecasts', hovertemplate: 'H = %{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(pb, { x: hs, y: hs.map(q => Math.sqrt(q * R.f[0])), mode: 'lines', line: { color: ORANGE, width: 1.4, dash: 'dash' }, name: 'square-root-of-time rule sqrt(H sigma^2(t+1))', hovertemplate: 'H = %{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(pb, { x: [H], y: [Math.sqrt(cum[H - 1])], mode: 'markers', marker: { color: DARK, size: 10 }, name: `H = ${H}`, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('h (days)'), AX('%^2', { rangemode: 'tozero' })); setAx(Pn, 1, AX('H (days)'), AX('%', { rangemode: 'tozero' }));
  plot('fw-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(pa, `Daily variance forecasts made on ${dt}`), ptitle(pb, 'Volatility of the H-day return (%)')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  steps('fw-we', [T.fw1(fdate(dt), nf(R.pr, 4), nf(R.s2bar, 3), nf(R.s2n, 3)), T.fw2(H, nf(R.s2bar, 3), nf(R.pr, 4), H - 1, nf(R.pr ** (H - 1), 3), nf(R.s2n, 3), nf(R.f[H - 1], 3)),
    T.fw3(H, nf(cum[H - 1], 2), nf(Math.sqrt(cum[H - 1]), 2)), T.fw4(H, nf(R.s2n, 3), nf(Math.sqrt(H * R.s2n), 2), R.s2n < R.s2bar)]);
  out('fw-out', [T.fw(fdate(dt), H, nf(Math.sqrt(cum[H - 1]), 2), nf(Math.sqrt(H * R.s2n), 2), R.s2n < R.s2bar, H !== 10 || dt !== D.end)]);
}

// ================================================================= x17. conditional VaR (worked example)
function vwBlock() {
  const a = +kSel('vw-a'), H = +$('vw-h').value, P = path('sp500:garch:t'), p = P.p, R = termFc(P, P.n - 1, 20), nw = single('vw-chart', 360), lev = `${+(a * 100).toFixed(1)}%`; label('vw-h').textContent = String(H);
  const tq = tppf(a, p.nu), sc = Math.sqrt((p.nu - 2) / p.nu), q = tq * sc, zq = normppf(a), s1 = Math.sqrt(R.s2n), hs = range(1, 20, 1), cum = []; let s = 0; R.f.forEach(v => cum.push(s += v));
  const v1 = -(p.mu + s1 * q), vn = -(p.mu + s1 * zq);
  plot('vw-chart', [{ x: hs, y: cum.map(c => -q * Math.sqrt(c)), mode: 'lines+markers', line: { color: RED, width: 1.6 }, name: 't quantile x sqrt(sum of the variance forecasts) (approximation)', hovertemplate: 'H = %{x}<br>%{y:.2f}%<extra></extra>' },
    { x: hs, y: cum.map(c => -zq * Math.sqrt(c)), mode: 'lines+markers', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: 'Normal quantile x sqrt(sum of the variance forecasts)', hovertemplate: 'H = %{x}<br>%{y:.2f}%<extra></extra>' },
    { x: hs, y: hs.map(h => v1 * Math.sqrt(h)), mode: 'lines', line: { color: ORANGE, width: 1.2, dash: 'dot' }, name: 'one-day VaR x sqrt(H)', hovertemplate: 'H = %{x}<br>%{y:.2f}%<extra></extra>' },
    { x: [H], y: [-q * Math.sqrt(cum[H - 1])], mode: 'markers', marker: { color: DARK, size: 11 }, name: `H = ${H}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('horizon H (days)', { dtick: 2 }), yaxis: AX(`S&P 500 VaR ${lev} (%)`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  steps('vw-we', [T.vw1(lev, nf(p.mu, 3), nf(R.s2n, 4), nf(s1, 3), nf(p.nu, 2)), T.vw2(lev, nf(tq, 3), nf(sc, 3), nf(q, 3)), T.vw3(lev, nf(p.mu, 3), nf(s1, 3), nf(q, 3), nf(v1, 2), nf(zq, 3), nf(vn, 2)),
    T.vw4(H, nf(-q, 3), nf(cum[H - 1], 2), nf(-q * Math.sqrt(cum[H - 1]), 2), nf(v1 * Math.sqrt(H), 2))]);
  out('vw-out', [T.vw(lev, nf(v1, 2), nf(vn, 2), H, nf(-q * Math.sqrt(cum[H - 1]), 2), a !== 0.01 || H !== 10)]);
}

// ================================================================= x18. exceedances 2015-2026 (table)
function exBlock() {
  const a = +kSel('ex-a'), y = +kSel('ex-y'), nw = single('ex-chart', 340), lev = `${+(a * 100).toFixed(1)}%`, rows = [], rg = [], re = [];
  MK4.forEach(k => {
    const V = varSeries(k, a), O = V.O, j0 = firstFrom(O.dates, `${y}-01-01`); let ng = 0, ne = 0;
    for (let i = j0; i < O.n; i++) { if (O.r[i] < -V.vg[i]) ng++; if (O.r[i] < -V.ve[i]) ne++; }
    const n = O.n - j0; rg.push(100 * ng / n); re.push(100 * ne / n);
    rows.push([SH[k], fint(n), fint(n * a), `${ng} (${nf(100 * ng / n, 2)}%)`, `${ne} (${nf(100 * ne / n, 2)}%)`]);
  });
  const xs = MK4.map(k => NAME[k]);
  plot('ex-chart', [{ type: 'bar', x: xs, y: rg, name: 'GARCH(1,1)-t', marker: { color: RED }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }, { type: 'bar', x: xs, y: re, name: 'EWMA-Normal', marker: { color: GREEN }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: [xs[0], xs[3]], y: [100 * a, 100 * a], mode: 'lines', line: { color: '#000', dash: 'dash', width: 1.2 }, name: `expected: ${lev} of the days`, hoverinfo: 'skip' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`days with r < -VaR ${lev} (%), from ${y}`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  table('ex-tab', [`VaR ${lev}`, T.h.days, T.h.expected, 'GARCH-t', 'EWMA-Normal'], rows);
  out('ex-out', [T.ex(lev, y, a !== 0.01 || y !== 2015, nf(Math.min(...rg), 2), nf(Math.max(...rg), 2), nf(Math.min(...re), 2), nf(Math.max(...re), 2), nf(100 * a, 1))]);
}

// ================================================================= wiring
const PFX = { returns: 'rt', mix: 'mx', arch1: 'a1', simulated: 'sm', archq: 'aq', kurt: 'ku', 'half-life': 'hl', step: 'st', igarch: 'ig', 'lik-arch1': 'l1', 'lik-garch': 'lg',
  'step-table': 'et', 'dist-table': 'dt', qq: 'qq', 'mk-table': 'mt', longrun: 'lr', 'vol-sd': 'v1', 'vol-bb': 'v2', persistence: 'ps', gjr: 'gj', nic: 'ni', 'asym-table': 'at',
  'acf-diag': 'ad', 'diag-table': 'dg', 'ms-table': 'ms', 'fc-we': 'fw', term: 'ts', 'fc-eval': 'fe', 'var-we': 'vw', var: 'vr', exc: 'ex' };
const BLOCKS = { rtBlock, mxBlock, a1Block, smBlock, aqBlock, kuBlock, hlBlock, stBlock, igBlock, l1Block, lgBlock, etBlock, dtBlock, qqBlock, mtBlock, lrBlock, v1Block, v2Block, psBlock,
  gjBlock, niBlock, atBlock, adBlock, dgBlock, msBlock, fwBlock, tsBlock, feBlock, vwBlock, vrBlock, exBlock };
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
  const MKR = cur => MK.filter(k => k !== cur).concat([cur]);
  on(['rt-s', 'rt-y'], rtBlock); PLy('rt-play', 'rt-y', [2017, 2003, 2005, 2010, 2013, 2019, 2021, 2024, 2017], 1000);
  on(['mx-s', 'mx-p', 'mx-c'], mxBlock); PLy('mx-play', 'mx-s', [4.5, 1, 2, 3, 6, 8, 10, 4.5], 900);
  on(['a1-a', 'a1-e', 'a1-w'], a1Block); PLy('a1-play', 'a1-a', [0.5, 0.1, 0.2, 0.3, 0.4, 0.55, 0.6, 0.7, 0.8, 0.5], 900);
  on(['sm-s', 'sm-a', 'sm-b'], smBlock); PLy('sm-play', 'sm-s', D.sim_seeds.slice(1).map(String).concat([String(D.seed)]), 1200);
  on(['aq-s', 'aq-q'], aqBlock); PLy('aq-play', 'aq-s', MKR('sp500'), 1100);
  on(['ku-a', 'ku-b', 'ku-w'], kuBlock); PLy('ku-play', 'ku-b', [0.88, 0.7, 0.75, 0.8, 0.85, 0.87, 0.89, 0.88], 900);
  on(['hl-p'], hlBlock); PLy('hl-play', 'hl-p', [0.98, 0.9, 0.95, 0.99, 0.9947, 0.999, 0.98], 1000);
  on(['st-r', 'st-s', 'st-p'], stBlock); PLy('st-play', 'st-r', [-3, -6, -5, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, -3], 700);
  on(['ig-l', 'ig-w'], igBlock); PLy('ig-play', 'ig-l', [0.94, 0.9, 0.92, 0.96, 0.97, 0.98, 0.99, 0.94], 900);
  on(['l1-a'], l1Block); PLy('l1-play', 'l1-a', [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5], 900);
  on(['lg-s', 'lg-n'], lgBlock); PLy('lg-play', 'lg-s', D.likg_seeds.slice(1).map(String).concat([String(D.likg_seeds[0])]), 1400);
  on(['et-s'], etBlock); PLy('et-play', 'et-s', MKR('sp500'), 1100);
  on(['dt-s'], dtBlock); PLy('dt-play', 'dt-s', MKR('sp500'), 1100);
  on(['qq-s'], qqBlock); PLy('qq-play', 'qq-s', MKR('sp500'), 1300);
  on(['mt-d'], mtBlock); PLy('mt-play', 'mt-d', ['n', 'st', 't'], 1300);
  on(['lr-s'], lrBlock); PLy('lr-play', 'lr-s', MKR('sp500'), 1100);
  on(['v1-d', 'v1-y'], v1Block); PLy('v1-play', 'v1-y', ['0', '1', '2', '3', '0'], 1300);
  on(['v2-d', 'v2-y'], v2Block); PLy('v2-play', 'v2-y', ['0', '1', '2', '3', '0'], 1300);
  on(['ps-m', 'ps-h'], psBlock); PLy('ps-play', 'ps-m', ['garch:n', 'gjr:t', 'egarch:t', 'garch:t'], 1300);
  on(['gj-e', 'gj-s'], gjBlock); PLy('gj-play', 'gj-e', [2, 0.5, 1, 1.5, 2.5, 3, 4, 5, 2], 800);
  on(['ni-s', 'ni-v'], niBlock); PLy('ni-play', 'ni-v', [1, 0.25, 0.5, 2, 3, 4, 1], 1100);
  on(['at-d'], atBlock); PLy('at-play', 'at-d', ['n', 'st', 't'], 1300);
  on(['ad-s', 'ad-m', 'ad-k'], adBlock); PLy('ad-play', 'ad-s', MKR('sp500'), 1200);
  on(['dg-s', 'dg-m'], dgBlock); PLy('dg-play', 'dg-s', MKR('sp500'), 1300);
  on(['ms-s', 'ms-c'], msBlock); PLy('ms-play', 'ms-s', MKR('sp500'), 1100);
  on(['fw-h', 'fw-d'], fwBlock); PLy('fw-play', 'fw-h', [10, 1, 5, 22, 66, 125, 250, 10], 1000);
  const Pts = path('sp500:garch:t'), tsd = ['2020-02-14', '2020-02-28', '2020-03-09', '2020-03-23', '2020-04-06', '2020-04-20', '2020-05-15', '2020-06-30'].map(d => firstFrom(Pts.dates, d));
  $('ts-d').max = String(Pts.n - 1); $('ts-d').value = String(Pts.n - 1);
  on(['ts-m', 'ts-d'], tsBlock); PLy('ts-play', 'ts-d', [Pts.n - 1, ...tsd, Pts.n - 1], 1000);
  on(['fe-y', 'fe-l'], feBlock); PLy('fe-play', 'fe-y', ['2015', '2017', '2019', '2021', '2023', '2015'], 1100);
  on(['vw-a', 'vw-h'], vwBlock); PLy('vw-play', 'vw-h', [10, 1, 2, 5, 15, 20, 10], 900);
  on(['vr-s', 'vr-w', 'vr-a'], vrBlock); PLy('vr-play', 'vr-w', ['2019', '2015', '2016', '2017', '2018', '2020', '2021', '2022', '2023', '2024', '2019'], 1100);
  on(['ex-a', 'ex-y'], exBlock); PLy('ex-play', 'ex-y', ['2015', '2017', '2019', '2021', '2023', '2015'], 1100);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH9 = { ser, path, persOf, halfLife, termFc, oosData, feEval, varSeries, dgCalc, qqData, likGrid, lik1, kernelNic, nicCurve, mktRow, atRow, crit, volFit, volStats, simPath, kurt,
  tppf, stdtq, normppf, chi2sf, acf, ljungBox, archLM, signBias, dmTest, ewmaVar, init, PFX, blocks: BLOCKS };
if (window.CH9_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
