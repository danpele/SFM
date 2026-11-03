// Chapter 2 interactive page: one block per chart slide, computed in the browser from ch2_data.js.
(function () {
'use strict';
const D = window.CH2_DATA;
const { C, nf, pct, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH2_TEXT;
const P = D.pal, COL = D.colors, SH = D.short;

// ================================================================= distributions
const SQ2PI = Math.sqrt(2 * Math.PI);
const phi = x => Math.exp(-0.5 * x * x) / SQ2PI;
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
const normSf = x => 0.5 * erfc(x / Math.SQRT2);
function normPpf(p) {               // Acklam's algorithm and one Halley step
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  let x;
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else { const q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  const e = Phi(x) - p, u = e * SQ2PI * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}
function lgamma(x) {                // Lanczos, g = 7
  const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = g[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += g[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(a, b, x) {
  let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < 1e-300) d = 1e-300;
  d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
  }
  return h;
}
function ibeta(a, b, x) {           // regularized incomplete beta I_x(a, b)
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const tLogC = nu => lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI);
const tpdf = (x, nu) => Math.exp(tLogC(nu) - (nu + 1) / 2 * Math.log(1 + x * x / nu));
const tSf = (x, nu) => { const p = 0.5 * ibeta(nu / 2, 0.5, nu / (nu + x * x)); return x >= 0 ? p : 1 - p; };    // P(T > x)
const tCdf = (x, nu) => 1 - tSf(x, nu);
function tPpf(u, nu) {              // quantile of the standard t(nu), bisection on the lower half
  if (u > 0.5) return -tPpf(1 - u, nu);
  if (u === 0.5) return 0;
  let lo = -1; while (tCdf(lo, nu) > u) lo *= 2;
  let hi = lo / 2 > -1 ? 0 : lo / 2;
  for (let i = 0; i < 80 && hi - lo > 1e-10 * Math.max(1, -lo); i++) { const m = (lo + hi) / 2; if (tCdf(m, nu) > u) hi = m; else lo = m; }
  return (lo + hi) / 2;
}
const tUnit = nu => Math.sqrt((nu - 2) / nu);                     // scale of a t(nu) with variance 1
const skewnormPdf = (x, a) => 2 * phi(x) * Phi(a * x);
function skewnormMoments(a) {
  const dl = a / Math.sqrt(1 + a * a), m = dl * Math.sqrt(2 / Math.PI), v = 1 - m * m;
  return { m, sd: Math.sqrt(v), skew: (4 - Math.PI) / 2 * m ** 3 / v ** 1.5 };
}
const binomPmf = (k, n, p) => Math.exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * Math.log(p) + (n - k) * Math.log(1 - p));

// ================================================================= statistics (same formulas as generate_all_charts.py)
const mean = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
function mom(a) {                   // mean, sd (ddof 1), skewness and excess kurtosis (population formulas, scipy.stats)
  const n = a.length, m = mean(a); let m2 = 0, m3 = 0, m4 = 0;
  for (const x of a) { const d = x - m, d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
  m2 /= n; m3 /= n; m4 /= n;
  const skew = m3 / m2 ** 1.5, exkurt = m4 / (m2 * m2) - 3, jb = n / 6 * (skew * skew + exkurt * exkurt / 4);
  return { n, mean: m, sd: Math.sqrt(m2 * n / (n - 1)), sd0: Math.sqrt(m2), skew, exkurt, jb, jb_p: Math.exp(-jb / 2) };
}
const sorted = a => Float64Array.from(a).sort();
function quantile(s, q) { const h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
function acf(x, nlags) {
  const m = mean(x), y = x.map(v => v - m); let d = 0; for (const v of y) d += v * v;
  const res = [];
  for (let h = 1; h <= nlags; h++) { let s = 0; for (let t = h; t < y.length; t++) s += y[t] * y[t - h]; res.push(s / d); }
  return res;
}
function corr(a, b) {
  const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
function leverage(r, lags) {        // L(k) = corr(r_t, |r_{t+k}|)
  const a = r.map(Math.abs), res = [];
  for (let k = 1; k <= lags; k++) res.push(corr(r.slice(0, r.length - k), a.slice(k)));
  return res;
}
function aggregate(r, h) {          // non-overlapping h-day sums
  const m = Math.floor(r.length / h), res = new Array(m);
  for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < h; j++) s += r[i * h + j]; res[i] = s; }
  return res;
}
function kde(r, x, bwMult) {        // Gaussian kernel, Scott's factor n^(-1/5) (scipy.stats.gaussian_kde)
  const n = r.length, h = mom(r).sd * Math.pow(n, -0.2) * (bwMult || 1);
  return x.map(v => { let s = 0; for (let i = 0; i < n; i++) { const z = (v - r[i]) / h; if (z > -40 && z < 40) s += Math.exp(-0.5 * z * z); } return s / (n * h * SQ2PI); });
}
function sigmaDays(r, k) {
  const { mean: m, sd } = mom(r); let up = 0, down = 0;
  for (const x of r) { const z = (x - m) / sd; if (z > k) up++; else if (z < -k) down++; }
  const p = 2 * normSf(k);
  return { obs: up + down, up, down, exp: r.length * p, p };
}
function tProfile(r, nu, init) {    // ML location and scale of a t(nu) with nu fixed (EM); log-likelihood
  let mu = init ? init.loc : quantile(sorted(r), 0.5), s = init ? init.scale : 1;
  const n = r.length;
  for (let it = 0; it < 500; it++) {
    let sw = 0, swx = 0;
    const w = r.map(x => { const z = (x - mu) / s; return (nu + 1) / (nu + z * z); });
    for (let i = 0; i < n; i++) { sw += w[i]; swx += w[i] * r[i]; }
    const mu1 = swx / sw; let ss = 0;
    for (let i = 0; i < n; i++) ss += w[i] * (r[i] - mu1) ** 2;
    const s1 = Math.sqrt(ss / n), done = Math.abs(mu1 - mu) < 1e-10 && Math.abs(s1 - s) < 1e-10;
    mu = mu1; s = s1; if (done) break;
  }
  return { loc: mu, scale: s, ll: tLL(r, nu, mu, s) };
}
function tLL(r, nu, loc, scale) { const c = tLogC(nu) - Math.log(scale); let s = 0; for (const x of r) { const z = (x - loc) / scale; s += c - (nu + 1) / 2 * Math.log(1 + z * z / nu); } return s; }
function normLL(r) { const { mean: m, sd0 } = mom(r); let s = 0; for (const x of r) s += -0.5 * ((x - m) / sd0) ** 2; return s - r.length * (Math.log(sd0) + 0.5 * Math.log(2 * Math.PI)); }

// ================================================================= data
const SER = {};
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function ser(k) {
  if (SER[k]) return SER[k];
  const R = D.ret[k], days = [R.d0];
  for (const dd of R.dd) days.push(days[days.length - 1] + dd);
  return (SER[k] = { r: R.r, days, dates: days.map(iso) });
}
const CR = D.crash.map(s => Date.parse(s) / 864e5);
function sample(k, opt) {           // returns of asset k from Jan 1 of year y0, optionally without the window D.crash
  const S = ser(k), y0 = opt && opt.y0 ? Date.parse(opt.y0 + '-01-01') / 864e5 : -Infinity, ex = opt && opt.ex;
  if (y0 === -Infinity && !ex) return S;
  const r = [], dates = [];
  S.r.forEach((x, i) => { const d = S.days[i]; if (d >= y0 && !(ex && d >= CR[0] && d <= CR[1])) { r.push(x); dates.push(S.dates[i]); } });
  return { r, dates };
}

// ================================================================= formatting and panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep);
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  if (Math.abs(x) >= 1e-4) { const e = Math.floor(Math.log10(Math.abs(x))); return nf(x, Math.max(0, d - 1 - e)); }
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e;
  return nf(m, d - 1) + ' × 10<sup>' + (e < 0 ? '−' : '') + Math.abs(e) + '</sup>';
}
const MON = LANG === 'ro' ? ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
  : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const fdate = s => { const [y, m, d] = s.split('-'); return `${+d} ${MON[+m - 1]} ${y}`; };
const spct = (x, d) => sig(100 * x, d || 3) + '%';
const list = items => items.join('; ');
const XT = (title, extra) => AX(title, extra);
// n panels in one chart: side by side when the chart is wide enough, stacked otherwise. Returns axis names and layout.
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
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.04, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const band = (a, b, axes) => ({ type: 'rect', xref: (axes ? axes.x : 'x') + ' domain', yref: axes ? axes.y : 'y', x0: 0, x1: 1, y0: -b, y1: b, fillcolor: P.green, opacity: 0.18, line: { width: 0 }, layer: 'below' });
const zline = axes => ({ type: 'line', xref: (axes ? axes.x : 'x') + ' domain', yref: axes ? axes.y : 'y', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.navy, width: 1 } });
const label = id => $(id + '-v');

// ================================================================= 1. the 68-95-99.7 rule
function normalPdf() {
  const k = +$('np-k').value; label('np-k').textContent = nf(k, 1);
  const xm = Math.max(4.2, k + 0.6), x = range(-xm, xm, xm / 300), y = x.map(phi);
  const seg = (a, b) => { const xs = range(a, b, (b - a) / 60); return { x: [a, ...xs, b], y: [0, ...xs.map(phi), 0] }; };
  const both = (a, b) => { const L = seg(-b, -a), R = seg(a, b); return { x: [...L.x, null, ...R.x], y: [...L.y, null, ...R.y] }; };
  const tr = [];
  [[3, P.amber, 0.35], [2, P.green, 0.35], [1, P.teal, 0.4]].forEach(([j, c, a]) => {
    const s = both(j - 1, j);
    tr.push({ x: s.x, y: s.y, fill: 'toself', mode: 'none', fillcolor: c, opacity: a, name: `within ${j} sd: ${(100 * (1 - 2 * normSf(j))).toFixed(2)}%`, hoverinfo: 'skip' });
  });
  const tl = both(k, xm);
  tr.push({ x: tl.x, y: tl.y, fill: 'toself', mode: 'none', fillcolor: P.red, opacity: 0.55, name: `beyond ${k.toFixed(1)} sd`, hoverinfo: 'skip' });
  tr.push({ x, y, mode: 'lines', name: 'Standard Normal density', line: { color: P.blue, width: 2 }, hovertemplate: 'z = %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  plot('np-chart', tr, layout({ shapes: [-k, k].map(v => ({ type: 'line', x0: v, x1: v, yref: 'paper', y0: 0, y1: 1, line: { color: P.red, width: 1.5, dash: 'dash' } })),
    xaxis: XT('z = (x − μ)/σ', { range: [-xm, xm] }), yaxis: XT('Density', { rangemode: 'tozero' }) }));
  const p = 2 * normSf(k);
  out('np-out', [T.np(nf(k, 1), spct(1 - p, 4), spct(p), fint(1 / p), sig(1 / p / 252, 3))]);
}

// ================================================================= 2. lognormal densities
const LN_S = [[0.25, P.blue], [0.5, P.red], [1.0, P.green]];
function lognormal() {
  const s = +$('ln-s').value; label('ln-s').textContent = nf(s, 2);
  const x = range(0.005, 4, 0.005), pdf = (v, q) => Math.exp(-0.5 * (Math.log(v) / q) ** 2) / (v * q * SQ2PI);
  const tr = LN_S.map(([q, c]) => ({ x, y: x.map(v => pdf(v, q)), mode: 'lines', name: `s = ${q}`, line: { color: c, width: 1.8 }, hovertemplate: `s = ${q}<br>x %{x:.2f}<br>density %{y:.3f}<extra></extra>` }));
  const mode = Math.exp(-s * s), mn = Math.exp(s * s / 2), sk = (Math.exp(s * s) + 2) * Math.sqrt(Math.exp(s * s) - 1);
  tr.push({ x, y: x.map(v => pdf(v, s)), mode: 'lines', name: `chosen s = ${s.toFixed(2)}`, line: { color: P.purple, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.3f}<extra></extra>' });
  tr.push({ x: [mode, 1, mn], y: [pdf(mode, s), pdf(1, s), pdf(mn, s)], mode: 'markers+text', text: ['mode', 'median', 'mean'], textposition: ['top center', 'top right', 'top right'],
    textfont: { color: P.purple, size: 11 }, marker: { color: P.purple, size: 9, line: { color: '#fff', width: 1.5 } }, showlegend: false, hovertemplate: '%{text}: %{x:.3f}<extra></extra>' });
  const shapes = LN_S.map(([q, c]) => ({ type: 'line', x0: Math.exp(q * q / 2), x1: Math.exp(q * q / 2), yref: 'paper', y0: 0, y1: 1, line: { color: c, width: 1, dash: 'dash' } }));
  shapes.push({ type: 'line', x0: 1, x1: 1, yref: 'paper', y0: 0, y1: 1, line: { color: C.navy, width: 1.2, dash: 'dot' } });
  tr.push({ x: [null], y: [null], mode: 'lines', name: 'median = 1 (all curves)', line: { color: C.navy, width: 1.2, dash: 'dot' } });
  tr.push({ x: [null], y: [null], mode: 'lines', name: 'mean e<sup>s²/2</sup> (colour of its curve)', line: { color: C.navy, width: 1, dash: 'dash' } });
  plot('ln-chart', tr, layout({ shapes, xaxis: XT('x = P<sub>T</sub>/P<sub>0</sub>', { range: [0, 4] }), yaxis: XT('Density', { range: [0, 1.75] }) }));
  out('ln-out', [T.ln(nf(s, 2), nf(mode, 3), nf(mn, 3), nf(sk, 2))]);
}

// ================================================================= 3. CLT
const CLT_N = [5, 30, 300];
function cltStats(n, p) {
  const sd = Math.sqrt(n * p * (1 - p)); let tail = 0; const z = [], h = [];
  for (let k = 0; k <= n; k++) {
    const pr = binomPmf(k, n, p), zz = (k - n * p) / sd;
    if (zz > 1.96) tail += pr;
    if (zz > -4.5 && zz < 5) { z.push(zz); h.push(pr * sd); }
  }
  return { z, h, w: 1 / sd, tail, skew: (1 - 2 * p) / sd };
}
function clt() {
  const p = +$('cl-p').value; label('cl-p').textContent = nf(p, 2);
  const Pn = panels('cl-chart', 3, { h: 360 }), tr = [], ann = [], xs = range(-4, 4, 0.02), res = [];
  CLT_N.forEach((n, i) => {
    const a = Pn.ax[i], s = cltStats(n, p); res.push(s);
    tr.push({ x: s.z, y: s.h, width: s.w * 0.9, type: 'bar', xaxis: a.x, yaxis: a.y, name: 'standardised mean of n draws', showlegend: i === 0, legendgroup: 'b',
      marker: { color: P.amber, opacity: 0.7 }, hovertemplate: 'z = %{x:.2f}<br>density %{y:.3f}<extra></extra>' });
    tr.push({ x: xs, y: xs.map(phi), mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'N(0, 1) density', showlegend: i === 0, legendgroup: 'n', line: { color: P.blue, width: 1.8 }, hoverinfo: 'skip' });
    ann.push(ptitle(a, `n = ${n}`));
    setAx(Pn, i, XT('z', { range: [-4, 5] }), XT(i === 0 || Pn.stack ? 'Density' : '', { rangemode: 'tozero' }));
  });
  plot('cl-chart', tr, layout(Object.assign({ annotations: ann, bargap: 0, margin: { l: 56, r: 12, t: 30, b: 40 } }, Pn.L)));
  out('cl-out', [T.cl(nf(p, 2), res.map((s, i) => `n = ${CLT_N[i]}: ${pct(s.tail, 1)}`).join(', '), res.map(s => nf(s.skew, 2)).join(', ')),
    ...(Math.abs(p - 0.2) < 1e-9 ? [T.clSim(CLT_N.map(n => pct(D.clt_sim[n], 1)).join(', '))] : [])]);
}

// ================================================================= 4. Normal approximation of the binomial
function approxErr(n, p, cc) {      // largest |F_binomial(k) - Phi((k + 0.5 - np)/sd)| over k = 0..n
  const mu = n * p, sd = Math.sqrt(n * p * (1 - p)); let F = 0, e = 0;
  for (let k = 0; k <= n; k++) { F += binomPmf(k, n, p); e = Math.max(e, Math.abs(F - Phi((k + (cc ? 0.5 : 0) - mu) / sd))); }
  return e;
}
const NA_CASES = [[5, 0.5], [20, 0.5], [20, 0.1], [100, 0.1]];
function normalApprox() {
  const n = +$('na-n').value, p = +$('na-p').value; label('na-n').textContent = n; label('na-p').textContent = nf(p, 2);
  const mu = n * p, sd = Math.sqrt(n * p * (1 - p)), lo = Math.max(0, Math.floor(mu - 4.5 * sd)), hi = Math.min(n, Math.ceil(mu + 4.5 * sd));
  const ks = range(lo, hi, 1), xs = range(lo - 0.5, hi + 0.5, (hi - lo + 1) / 300);
  plot('na-pmf', [
    { x: ks, y: ks.map(k => binomPmf(k, n, p)), type: 'bar', name: 'binomial probabilities', marker: { color: P.amber, opacity: 0.7 }, width: 0.85, hovertemplate: 'k = %{x}<br>P = %{y:.4f}<extra></extra>' },
    { x: xs, y: xs.map(v => phi((v - mu) / sd) / sd), mode: 'lines', name: 'Normal density N(np, np(1−p))', line: { color: P.blue, width: 1.8 }, hoverinfo: 'skip' }
  ], layout({ title: { text: `n = ${n}, p = ${p.toFixed(2)}`, x: 0, xanchor: 'left', font: { size: 12, color: C.navy } }, bargap: 0.1, xaxis: XT('k'), yaxis: XT('Probability', { rangemode: 'tozero' }) }));
  const ns = range(2, 200, 1), curve = q => ns.map(m => approxErr(m, q, true));
  const tr = [[0.5, P.blue], [0.1, P.red]].map(([q, c]) => ({ x: ns, y: curve(q), mode: 'lines', name: `p = ${q}`, line: { color: c, width: 1.5 }, hovertemplate: `p = ${q}, n = %{x}: %{y:.4f}<extra></extra>` }));
  if (Math.abs(p - 0.5) > 1e-9 && Math.abs(p - 0.1) > 1e-9) tr.push({ x: ns, y: curve(p), mode: 'lines', name: `p = ${p.toFixed(2)}`, line: { color: P.purple, width: 1.5 }, hovertemplate: `p = ${p.toFixed(2)}, n = %{x}: %{y:.4f}<extra></extra>` });
  tr.push({ x: NA_CASES.map(c => c[0]), y: NA_CASES.map(c => approxErr(c[0], c[1], true)), text: NA_CASES.map(c => `n = ${c[0]}, p = ${c[1]}`), mode: 'markers', name: 'panels of the slide',
    marker: { color: P.amber, size: 9, line: { color: C.navy, width: 1 } }, hovertemplate: '%{text}: %{y:.4f}<extra></extra>' });
  const e = approxErr(n, p, true);
  tr.push({ x: [n], y: [e], mode: 'markers', name: 'chosen n, p', marker: { color: P.purple, size: 12, symbol: 'diamond', line: { color: '#fff', width: 1.5 } }, hovertemplate: '%{y:.4f}<extra></extra>' });
  plot('na-err', tr, layout({ title: { text: 'Largest CDF error, with continuity correction', x: 0, xanchor: 'left', font: { size: 12, color: C.navy } },
    xaxis: XT('n (log scale)', { type: 'log' }), yaxis: XT('max |error| (log scale)', { type: 'log', exponentformat: 'power' }) }));
  out('na-out', [T.na(n, nf(p, 2), nf(n * p * (1 - p), 2), nf(e, 4), nf(approxErr(n, p, false), 4))]);
}

// ================================================================= 5. skewness and kurtosis
function shapes() {
  const nu = +$('sh-nu').value, al = +$('sh-a').value; label('sh-nu').textContent = nf(nu, 1); label('sh-a').textContent = nf(al, 1);
  const Pn = panels('sh-chart', 2), x = range(-5, 5, 0.0125), c = tUnit(nu), sm = skewnormMoments(al);
  const [a, b] = Pn.ax;
  const tr = [
    { x, y: x.map(phi), mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'Normal (excess kurtosis 0)', line: { color: P.blue, width: 1.8 }, hoverinfo: 'skip' },
    { x, y: x.map(v => tpdf(v / c, nu) / c), mode: 'lines', xaxis: a.x, yaxis: a.y, name: `Student-t(${+nu.toFixed(1)}), unit variance (excess kurtosis ${(6 / (nu - 4)).toFixed(1)})`, line: { color: P.red, width: 1.8 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    { x, y: x.map(v => skewnormPdf(-v * sm.sd + sm.m, al) * sm.sd), mode: 'lines', xaxis: b.x, yaxis: b.y, name: `negative skewness (${(-sm.skew).toFixed(2)})`, line: { color: P.green, width: 1.8 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    { x, y: x.map(v => skewnormPdf(v * sm.sd + sm.m, al) * sm.sd), mode: 'lines', xaxis: b.x, yaxis: b.y, name: `positive skewness (+${sm.skew.toFixed(2)})`, line: { color: P.purple, width: 1.8 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    { x, y: x.map(phi), mode: 'lines', xaxis: b.x, yaxis: b.y, showlegend: false, line: { color: P.blue, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' }
  ];
  setAx(Pn, 0, XT('Standardised value'), XT('Density', { range: [0, 0.75] })); setAx(Pn, 1, XT('Standardised value'), XT(Pn.stack ? 'Density' : '', { range: [0, 0.75] }));
  plot('sh-chart', tr, layout(Object.assign({ annotations: [ptitle(a, 'Kurtosis: taller centre, heavier tails'), ptitle(b, 'Skewness: a longer left or right tail')] }, Pn.L)));
  out('sh-out', [T.sh(nf(nu, 1), nf(6 / (nu - 4), 2), nf(al, 1), nf(sm.skew, 2))]);
}

// ================================================================= 6. DAX returns vs the Normal density
const kdeCache = {};
function daxDensity() {
  const k = $('dx-a').value, bw = +$('dx-bw').value; label('dx-bw').textContent = '× ' + nf(bw, 2);
  const r = ser(k).r, M = mom(r), key = k + ':' + bw;
  const span = Math.max(9, Math.ceil(7 * M.sd)), x = range(-span, span, 2 * span / 699);
  if (!kdeCache[key]) kdeCache[key] = { fx: kde(r, x, bw), peak: kde(r, [M.mean], bw)[0], at6: kde(r, [-6], bw)[0] };
  const K = kdeCache[key], fn = x.map(v => phi((v - M.mean) / M.sd) / M.sd);
  const Pn = panels('dx-chart', 2), [a, b] = Pn.ax, tr = [];
  [a, b].forEach((ax, i) => {
    tr.push({ x, y: K.fx, mode: 'lines', xaxis: ax.x, yaxis: ax.y, name: `${SH[k]}, kernel density estimate`, showlegend: i === 0, line: { color: COL[k], width: 1.8 }, hovertemplate: '%{x:.2f}%: %{y:.3g}<extra></extra>' });
    tr.push({ x, y: fn, mode: 'lines', xaxis: ax.x, yaxis: ax.y, name: 'Normal, same mean and sd', showlegend: i === 0, line: { color: P.red, width: 1.5, dash: 'dash' }, hovertemplate: '%{x:.2f}%: %{y:.3g}<extra></extra>' });
  });
  const lin = 5 * Math.max(1, M.sd / 1.3);
  setAx(Pn, 0, XT('Daily log return (%)', { range: [-lin, lin] }), XT('Density', { rangemode: 'tozero' }));
  setAx(Pn, 1, XT('Daily log return (%)', { range: [-span, span] }), XT(Pn.stack ? 'Density (log scale)' : '', { type: 'log', range: [-6, 0], exponentformat: 'power' }));
  plot('dx-chart', tr, layout(Object.assign({ annotations: [ptitle(a, 'Linear scale: the centre'), ptitle(b, 'Log scale: the tails')] }, Pn.L)));
  let w1 = 0; for (const v of r) if (Math.abs(v - M.mean) <= M.sd) w1++;
  const n6 = phi((-6 - M.mean) / M.sd) / M.sd;
  out('dx-out', [T.dx(SH[k], nf(bw, 2), nf(K.peak, 2), nf(phi(0) / M.sd, 2), pct(w1 / r.length, 1), pct(1 - 2 * normSf(1), 1), sig(K.at6 / n6, 3))]);
}

// ================================================================= QQ plots
function qqNormal(r) {              // standardised data against N(0,1), plotting positions (i - 0.5)/n
  const s = sorted(r), n = s.length, M = mom(r), tq = new Array(n), eq = new Array(n);
  for (let i = 0; i < n; i++) { tq[i] = normPpf((i + 0.5) / n); eq[i] = (s[i] - M.mean) / M.sd; }
  return { tq, eq };
}
const qqTCache = {};
function qqT(k, r, nu, loc, scale) { // raw data against the fitted t(nu, loc, scale)
  const key = k + ':' + nu.toFixed(4);
  if (!qqTCache[key]) {
    const n = r.length, z = new Array(n);
    for (let i = 0; i < Math.ceil(n / 2); i++) { const v = tPpf((i + 0.5) / n, nu); z[i] = v; z[n - 1 - i] = -v; }
    qqTCache[key] = z;
  }
  return { tq: qqTCache[key].map(v => loc + scale * v), eq: Array.from(sorted(r)) };
}
const QQ3 = ['sp500', 'bet', 'btc'];
function qqPanels(id, data, titles, xt, yt) {
  const Pn = panels(id, 3, { h: 380 }), tr = [], ann = [];
  data.forEach((q, i) => {
    const a = Pn.ax[i], k = QQ3[i], lo = Math.min(q.tq[0], q.eq[0]), hi = Math.max(q.tq[q.tq.length - 1], q.eq[q.eq.length - 1]);
    tr.push({ x: q.tq, y: q.eq, mode: 'markers', type: 'scatter', xaxis: a.x, yaxis: a.y, name: `${SH[k]} daily log returns`, marker: { color: COL[k], size: 4 }, hovertemplate: 'theoretical %{x:.2f}<br>empirical %{y:.2f}<extra></extra>' });
    tr.push({ x: [lo, hi], y: [lo, hi], mode: 'lines', xaxis: a.x, yaxis: a.y, name: '45-degree line', showlegend: i === 0, line: { color: C.navy, width: 1, dash: 'dash' }, hoverinfo: 'skip' });
    ann.push(ptitle(a, titles[i]));
    setAx(Pn, i, XT(xt), XT(i === 0 || Pn.stack ? yt : ''));
  });
  plot(id, tr, layout(Object.assign({ annotations: ann }, Pn.L)));
}
function qqNormalBlock() {
  const y = +$('qn-y').value; label('qn-y').textContent = y;
  const q = QQ3.map(k => qqNormal(sample(k, { y0: y > 2010 ? y : null }).r));
  qqPanels('qn-chart', q, QQ3.map(k => SH[k] + ' vs Normal'), 'Theoretical quantile (N(0,1))', 'Empirical quantile (standardised)');
  out('qn-out', [T.qn(y, QQ3.map((k, i) => T.qnItem(SH[k], nf(q[i].eq[0], 1), nf(q[i].tq[0], 1))).join('; '))]);
}
const QT_NU = [0, 2.5, 3, 4, 5, 6, 8, 10, 15, 30];       // 0: the ML estimate of each series
function qqTBlock() {
  const j = +$('qt-nu').value, nu0 = QT_NU[j];
  label('qt-nu').textContent = nu0 ? 'ν = ' + nu0 : T.qtFit;
  const q = [], nus = [];
  QQ3.forEach(k => {
    const r = ser(k).r, F = D.tfit[k], nu = nu0 || F.nu, fit = nu0 ? tProfile(r, nu, F) : F;
    nus.push(nu); q.push(qqT(k, r, nu, fit.loc, fit.scale));
  });
  qqPanels('qt-chart', q, QQ3.map((k, i) => `${SH[k]} vs t(${nus[i].toFixed(1)})`), 'Theoretical quantile (fitted t, %)', 'Empirical quantile (%)');
  out('qt-out', [T.qt(QQ3.map((k, i) => T.qtItem(SH[k], nf(nus[i], 2), nf(q[i].eq[0], 1), nf(q[i].tq[0], 1))).join('; '))]);
}

// ================================================================= 8. Student-t densities with unit variance
const TD_NU = [[3, P.red], [5, P.amber], [10, P.green]];
function tDensities() {
  const k = +$('td-k').value; label('td-k').textContent = nf(k, 1);
  const Pn = panels('td-chart', 2), x = range(-7, 7, 0.0175), tr = [];
  Pn.ax.forEach((a, i) => {
    tr.push({ x, y: x.map(phi), mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'Normal', showlegend: i === 0, line: { color: P.blue, width: 2 }, hovertemplate: 'Normal %{x:.2f}: %{y:.3g}<extra></extra>' });
    TD_NU.forEach(([nu, c]) => { const s = tUnit(nu); tr.push({ x, y: x.map(v => tpdf(v / s, nu) / s), mode: 'lines', xaxis: a.x, yaxis: a.y, name: `t(${nu}), unit variance`, showlegend: i === 0, line: { color: c, width: 1.5 }, hovertemplate: `t(${nu}) %{x:.2f}: %{y:.3g}<extra></extra>` }); });
  });
  const vl = a => [-k, k].map(v => ({ type: 'line', xref: a.x, yref: a.y + ' domain', x0: v, x1: v, y0: 0, y1: 1, line: { color: C.navy, width: 1.2, dash: 'dot' } }));
  setAx(Pn, 0, XT('Standardised value', { range: [-Math.max(4, k + 0.5), Math.max(4, k + 0.5)] }), XT('Density', { rangemode: 'tozero' }));
  setAx(Pn, 1, XT('Standardised value', { range: [-7, 7] }), XT(Pn.stack ? 'Density (log scale)' : '', { type: 'log', range: [-7, 0], exponentformat: 'power' }));
  plot('td-chart', tr, layout(Object.assign({ shapes: [...vl(Pn.ax[0]), ...vl(Pn.ax[1])], annotations: [ptitle(Pn.ax[0], 'Linear scale'), ptitle(Pn.ax[1], 'Log scale')] }, Pn.L)));
  const pn = 2 * normSf(k), pt = TD_NU.map(([nu]) => 2 * tSf(k / tUnit(nu), nu));
  out('td-out', [T.td(nf(k, 1), TD_NU.map(([nu], i) => `t(${nu}) ${spct(pt[i])}`).join(', '), spct(pn), sig(pt[1] / pn, 3))]);
}

// ================================================================= 9. S&P 500: Normal and Student-t fits
function hist(r, bins) {            // numpy.histogram(density=True) on [min, max]
  let lo = Infinity, hi = -Infinity; for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const w = (hi - lo) / bins, c = new Array(bins).fill(0);
  for (const v of r) c[Math.min(bins - 1, Math.floor((v - lo) / w))]++;
  return { mid: c.map((_, i) => lo + (i + 0.5) * w), dens: c.map(v => v / (r.length * w)), w, lo, hi };
}
const tfCache = {};
function tFit() {
  const k = $('tf-a').value, F = D.tfit[k];
  if ($('tf-nu').dataset.asset !== k) { $('tf-nu').dataset.asset = k; $('tf-nu').value = F.nu.toFixed(2); }
  const nuIn = +$('tf-nu').value, isFit = Math.abs(nuIn - +F.nu.toFixed(2)) < 1e-9, nu = isFit ? F.nu : nuIn;
  label('tf-nu').textContent = nf(nu, 2) + (isFit ? ' (ML)' : '');
  const r = ser(k).r, M = mom(r), key = k + ':' + nu;
  const fit = isFit ? { loc: F.loc, scale: F.scale, ll: F.ll_t } : (tfCache[key] = tfCache[key] || tProfile(r, nu, F));
  const h160 = hist(r, 160), h120 = hist(r, 120), x = range(h160.lo - 0.5, h160.hi + 0.5, (h160.hi - h160.lo + 1) / 799);
  const fn = x.map(v => phi((v - M.mean) / M.sd) / M.sd), ft = x.map(v => tpdf((v - fit.loc) / fit.scale, nu) / fit.scale);
  const Pn = panels('tf-chart', 2), [a, b] = Pn.ax, pos = h120.dens.map((d, i) => [h120.mid[i], d]).filter(p => p[1] > 0);
  const tr = [
    { x: h160.mid, y: h160.dens, type: 'bar', width: h160.w, xaxis: a.x, yaxis: a.y, name: `${SH[k]} (histogram)`, marker: { color: COL[k], opacity: 0.8 }, hovertemplate: '%{x:.2f}%: %{y:.3f}<extra></extra>' },
    { x: pos.map(p => p[0]), y: pos.map(p => p[1]), mode: 'markers', xaxis: b.x, yaxis: b.y, showlegend: false, marker: { color: COL[k], size: 5 }, hovertemplate: '%{x:.2f}%: %{y:.3g}<extra></extra>' }
  ];
  [a, b].forEach((ax, i) => {
    tr.push({ x, y: fn, mode: 'lines', xaxis: ax.x, yaxis: ax.y, name: 'Normal fit', showlegend: i === 0, line: { color: P.red, width: 1.5, dash: 'dash' }, hoverinfo: 'skip' });
    tr.push({ x, y: ft, mode: 'lines', xaxis: ax.x, yaxis: ax.y, name: `Student-t fit, nu = ${nu.toFixed(2)}`, showlegend: i === 0, line: { color: P.green, width: 1.8 }, hoverinfo: 'skip' });
  });
  const lin = 5 * Math.max(1, M.sd / 1.3);
  setAx(Pn, 0, XT('Daily log return (%)', { range: [-lin, lin] }), XT('Density', { rangemode: 'tozero' }));
  setAx(Pn, 1, XT('Daily log return (%)'), XT(Pn.stack ? 'Density (log scale)' : '', { type: 'log', range: [-5, Math.log10(2)], exponentformat: 'power' }));
  plot('tf-chart', tr, layout(Object.assign({ bargap: 0, annotations: [ptitle(a, 'Linear scale'), ptitle(b, 'Log scale (points: histogram heights)')] }, Pn.L)));
  out('tf-out', [T.tf(SH[k], nf(nu, 2), nf(fit.ll, 1), nf(F.nu, 2), nf(F.ll_t, 1), nf(normLL(r), 1))]);
}

// ================================================================= 11. 4-sigma days
function sigmaDaysBlock() {
  const k = +$('sg-k').value; label('sg-k').textContent = nf(k, 1);
  const A = D.assets, S = A.map(a => sigmaDays(ser(a).r, k)), obs = S.map(s => s.obs), exp = S.map(s => s.exp), names = A.map(a => SH[a]), cols = A.map(a => COL[a]);
  const fmtE = e => e >= 10 ? e.toFixed(0) : e >= 0.01 ? e.toFixed(2) : e.toPrecision(2);
  const ymax = Math.max(80, 1.8 * Math.max(...obs)), ymin = Math.min(0.05, 0.4 * Math.min(...exp));
  plot('sg-chart', [
    { x: names, y: obs.map(v => v || null), type: 'bar', name: `observed days beyond ${k.toFixed(1)} sd (solid)`, marker: { color: cols }, text: obs.map(String), textposition: 'outside', textfont: { color: C.navy, size: 11 }, cliponaxis: false, hovertemplate: '%{x}: %{y} days observed<extra></extra>' },
    { x: names, y: exp, type: 'bar', name: 'expected under the Normal distribution (hatched)', marker: { color: 'rgba(0,0,0,0)', line: { color: cols, width: 1.2 }, pattern: { shape: '/', fgcolor: cols, size: 7, solidity: 0.25 } },
      text: exp.map(fmtE), textposition: 'outside', textfont: { color: C.navy, size: 10 }, cliponaxis: false, hovertemplate: '%{x}: %{y:.3g} days expected<extra></extra>' }
  ], layout({ barmode: 'group', bargap: 0.25, bargroupgap: 0, xaxis: XT(''), yaxis: XT('Number of days (log scale)', { type: 'log', range: [Math.log10(ymin), Math.log10(ymax)], exponentformat: 'power' }) }));
  const i0 = 0, s0 = S[i0];
  out('sg-out', [T.sg(nf(k, 1), spct(s0.p), fint(1 / s0.p), SH[A[i0]], s0.obs, nf(s0.exp, s0.exp < 0.01 ? 6 : 2), s0.obs ? fint(ser(A[i0]).r.length / s0.obs) : '—'),
    T.sgAll(obs.reduce((s, v) => s + v, 0), nf(exp.reduce((s, v) => s + v, 0), 2))]);
}

// ================================================================= 12. kurtosis by horizon
function aggregation() {
  const h = +$('ag-h').value, ex = $('ag-s').value === 'ex'; label('ag-h').textContent = h;
  const I = D.indices, H = D.horizons, tr = [];
  const vals = I.map(k => { const r = sample(k, { ex }).r; return { line: H.map(q => mom(aggregate(r, q)).exkurt), at: mom(aggregate(r, h)) }; });
  I.forEach((k, i) => {
    tr.push({ x: H, y: vals[i].line, mode: 'lines+markers', name: SH[k], line: { color: COL[k], width: 1.6 }, marker: { size: 7 }, hovertemplate: SH[k] + ', h = %{x}: %{y:.1f}<extra></extra>' });
    tr.push({ x: [h], y: [vals[i].at.exkurt], mode: 'markers', showlegend: false, marker: { color: COL[k], size: 13, symbol: 'diamond', line: { color: '#fff', width: 1.5 } }, hovertemplate: SH[k] + ', h = ' + h + ': %{y:.2f}<extra></extra>' });
  });
  tr.push({ x: [1, 63], y: [0, 0], mode: 'lines', name: 'Normal distribution (0)', line: { color: C.navy, width: 1, dash: 'dash' }, hoverinfo: 'skip' });
  plot('ag-chart', tr, layout({ shapes: [{ type: 'line', x0: h, x1: h, yref: 'paper', y0: 0, y1: 1, line: { color: C.navy, width: 1, dash: 'dot' } }],
    xaxis: XT('Horizon h (days, log scale)', { type: 'log', tickvals: H, ticktext: H.map(String) }), yaxis: XT('Excess kurtosis of h-day returns') }));
  out('ag-out', [T.ag(h, ex ? T.agEx : T.agAll, I.map((k, i) => `${SH[k]} ${nf(vals[i].at.exkurt, 1)} (n = ${vals[i].at.n}, JB p ${vals[i].at.jb_p < 1e-4 ? '< 0' + (LANG === 'ro' ? ',' : '.') + '0001' : '= ' + nf(vals[i].at.jb_p, 4)})`).join('; '))]);
}

// ================================================================= 13. QQ plots, daily vs h-day
function qqHorizons() {
  const k = $('qh-a').value, h = +$('qh-h').value; label('qh-h').textContent = h;
  const r = ser(k).r, Pn = panels('qh-chart', 2), tr = [], ann = [], res = [];
  [1, h].forEach((q, i) => {
    const x = aggregate(r, q), Q = qqNormal(x), a = Pn.ax[i], lo = Math.min(Q.tq[0], Q.eq[0]), hi = Math.max(Q.tq[Q.tq.length - 1], Q.eq[Q.eq.length - 1]);
    const lab = q === 1 ? 'daily' : q === 21 ? 'monthly (21 days)' : `${q}-day`;
    res.push(mom(x));
    tr.push({ x: Q.tq, y: Q.eq, mode: 'markers', xaxis: a.x, yaxis: a.y, name: `${SH[k]}, ${lab}`, marker: { color: i ? P.purple : COL[k], size: i ? 6 : 4 }, hovertemplate: 'N(0,1) %{x:.2f}<br>empirical %{y:.2f}<extra></extra>' });
    tr.push({ x: [lo, hi], y: [lo, hi], mode: 'lines', xaxis: a.x, yaxis: a.y, name: '45-degree line', showlegend: i === 0, line: { color: C.navy, width: 1, dash: 'dash' }, hoverinfo: 'skip' });
    ann.push(ptitle(a, `${lab}, n = ${x.length}`));
    setAx(Pn, i, XT('N(0,1) quantile'), XT(i === 0 || Pn.stack ? 'Standardised empirical quantile' : ''));
  });
  plot('qh-chart', tr, layout(Object.assign({ annotations: ann }, Pn.L)));
  const m = res[1], jp = m.jb_p < 1e-4 ? '< 0' + (LANG === 'ro' ? ',' : '.') + '0001' : '= ' + nf(m.jb_p, 4);
  out('qh-out', [T.qh(SH[k], h, m.n, nf(m.skew, 2), nf(m.exkurt, 1), jp, nf(res[0].exkurt, 1))]);
}

// ================================================================= 14. ACF of returns and of |r|^delta
function acfBlock() {
  const dl = +$('ac-d').value, ex = $('ac-s').value === 'ex'; label('ac-d').textContent = nf(dl, 2);
  const Pn = panels('ac-chart', 2), [a, b] = Pn.ax, lags = range(1, 50, 1), tr = [], res = {};
  let nmin = 1e9;
  QQ3.forEach(k => {
    const r = sample(k, { ex }).r; nmin = Math.min(nmin, r.length);
    const ar = acf(r, 50), aa = acf(r.map(v => Math.pow(Math.abs(v), dl)), 50); res[k] = { ar, aa, n: r.length };
    tr.push({ x: lags, y: ar, mode: 'lines+markers', xaxis: a.x, yaxis: a.y, name: SH[k], line: { color: COL[k], width: 0.8 }, marker: { size: 4 }, hovertemplate: SH[k] + ', lag %{x}: %{y:.3f}<extra></extra>' });
    tr.push({ x: lags, y: aa, mode: 'lines+markers', xaxis: b.x, yaxis: b.y, showlegend: false, line: { color: COL[k], width: 0.8 }, marker: { size: 4 }, hovertemplate: SH[k] + ', lag %{x}: %{y:.3f}<extra></extra>' });
  });
  const bd = 1.96 / Math.sqrt(nmin);
  tr.push({ x: [null], y: [null], mode: 'lines', name: '95% band under independence', line: { color: P.green, width: 8 }, opacity: 0.35 });
  setAx(Pn, 0, XT('Lag (days)'), XT('Autocorrelation', { range: [-0.15, 0.5] })); setAx(Pn, 1, XT('Lag (days)'), XT(Pn.stack ? 'Autocorrelation' : '', { range: [-0.15, 0.5], matches: 'y' }));
  plot('ac-chart', tr, layout(Object.assign({ shapes: [band(0, bd, a), band(0, bd, b), zline(a), zline(b)],
    annotations: [ptitle(a, 'Returns r<sub>t</sub>'), ptitle(b, dl === 1 ? 'Absolute returns |r<sub>t</sub>|' : `|r<sub>t</sub>|<sup>${dl}</sup>`)] }, Pn.L)));
  const s = res.sp500;
  out('ac-out', [T.ac(ex ? T.agEx : T.agAll, QQ3.map(k => `${SH[k]} ${nf(res[k].ar[0], 2)}`).join(', '), nf(dl, 2), nf(s.aa[0], 2), nf(s.aa[9], 2), nf(s.aa[49], 2), nf(bd, 3))]);
}

// ================================================================= 15. calm and turbulent periods
function rollSd(r, w) {
  const out = new Array(r.length).fill(null); let s = 0, s2 = 0;
  for (let i = 0; i < r.length; i++) {
    s += r[i]; s2 += r[i] * r[i];
    if (i >= w) { s -= r[i - w]; s2 -= r[i - w] * r[i - w]; }
    if (i >= w - 1) out[i] = Math.sqrt(Math.max(0, (s2 - s * s / w) / (w - 1)));
  }
  return out;
}
function clustering() {
  const w = +$('vc-w').value; label('vc-w').textContent = w ? w + (LANG === 'ro' ? ' zile' : ' days') : (LANG === 'ro' ? 'fără' : 'off');
  const Pn = panels('vc-chart', 3, { stack: true, hs: 170, extra: 80, vgap: 0.06 }), tr = [], items = [];
  QQ3.forEach((k, i) => {
    const S = ser(k), a = Pn.ax[i];
    tr.push({ x: S.dates, y: S.r, mode: 'lines', xaxis: a.x, yaxis: a.y, name: `${SH[k]} daily log return (%)`, line: { color: COL[k], width: 0.6 }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' });
    if (w) {
      const sd = rollSd(S.r, w);
      tr.push({ x: S.dates, y: sd.map(v => v === null ? null : 2 * v), mode: 'lines', xaxis: a.x, yaxis: a.y, name: `± 2 × ${w}-day standard deviation`, showlegend: i === 0, legendgroup: 'sd', line: { color: C.navy, width: 1.2 }, hovertemplate: '%{x}: 2 sd = %{y:.2f}%<extra></extra>' });
      tr.push({ x: S.dates, y: sd.map(v => v === null ? null : -2 * v), mode: 'lines', xaxis: a.x, yaxis: a.y, showlegend: false, legendgroup: 'sd', line: { color: C.navy, width: 1.2 }, hoverinfo: 'skip' });
      let hi = -1, lo = -1;
      sd.forEach((v, j) => { if (v === null) return; if (hi < 0 || v > sd[hi]) hi = j; if (lo < 0 || v < sd[lo]) lo = j; });
      items.push(T.vcItem(SH[k], nf(sd[hi], 2), fdate(S.dates[hi]), nf(sd[lo], 2), fdate(S.dates[lo]), nf(sd[hi] / sd[lo], 1)));
    } else {
      const M = mom(S.r); let j = 0; S.r.forEach((v, t) => { if (v < S.r[j]) j = t; });
      items.push(T.vcItem0(SH[k], nf(M.sd, 2), nf(S.r[j], 1), fdate(S.dates[j])));
    }
    setAx(Pn, i, { type: 'date', matches: i ? 'x' : undefined, showticklabels: i === 2, gridcolor: C.grid, color: C.navy, tickfont: { size: 11, color: C.navy } }, XT('%'));
  });
  plot('vc-chart', tr, layout(Object.assign({ margin: { l: 56, r: 12, t: 16, b: 40 }, legend: Object.assign(layout().legend, { y: -0.07 }) }, Pn.L)));
  out('vc-out', [w ? T.vc(w) : T.vc0, ...items]);
}

// ================================================================= 16. leverage correlations
function leverageBlock() {
  const y = +$('lv-y').value, ex = $('lv-s').value === 'ex'; label('lv-y').textContent = y;
  const I = D.indices, tr = [], res = {}; let nmin = 1e9;
  I.forEach(k => {
    const r = sample(k, { y0: y > 2010 ? y : null, ex }).r; nmin = Math.min(nmin, r.length);
    const L = leverage(r, 20); res[k] = L;
    tr.push({ x: range(1, 20, 1), y: L, mode: 'lines+markers', name: SH[k], line: { color: COL[k], width: 1.2 }, marker: { size: 6 }, hovertemplate: SH[k] + ', k = %{x}: %{y:.3f}<extra></extra>' });
  });
  const bd = 1.96 / Math.sqrt(nmin);
  tr.push({ x: [null], y: [null], mode: 'lines', name: '95% band under independence', line: { color: P.green, width: 8 }, opacity: 0.35 });
  plot('lv-chart', tr, layout({ shapes: [band(0, bd), zline()], xaxis: XT('Lag k (days)', { dtick: 1 }), yaxis: XT('L(k) = corr(r<sub>t</sub>, |r<sub>t+k</sub>|)') }));
  const m5 = k => mean(res[k].slice(0, 5));
  out('lv-out', [T.lv(y, ex ? T.agEx : T.agAll, I.map(k => `${SH[k]} ${nf(m5(k), 2)}`).join(', '), nf(bd, 3))]);
}

// ================================================================= 17. gain/loss asymmetry
const GL_A = [0.001, 0.005, 0.01, 0.025, 0.05];
function gainLoss() {
  const al = GL_A[+$('gl-a').value], k = +$('gl-k').value; label('gl-a').textContent = pct(al, al < 0.01 ? 1 : al === 0.025 ? 1 : 0); label('gl-k').textContent = nf(k, 1);
  const A = D.assets, names = A.map(a => SH[a]);
  const Q = A.map(a => { const s = sorted(ser(a).r); return [-quantile(s, al), quantile(s, 1 - al)]; }), N = A.map(a => sigmaDays(ser(a).r, k));
  const Pn = panels('gl-chart', 2), [a, b] = Pn.ax, ap = pct(al, al < 0.01 || al === 0.025 ? 1 : 0), bp = pct(1 - al, al < 0.01 || al === 0.025 ? 1 : 0);
  const tr = [
    { x: names, y: Q.map(q => q[0]), type: 'bar', xaxis: a.x, yaxis: a.y, name: `loss: −(${ap} quantile)`, marker: { color: P.red }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: names, y: Q.map(q => q[1]), type: 'bar', xaxis: a.x, yaxis: a.y, name: `gain: ${bp} quantile`, marker: { color: P.green }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: names, y: N.map(s => s.down), type: 'bar', xaxis: b.x, yaxis: b.y, showlegend: false, marker: { color: P.red }, hovertemplate: '%{x}: %{y} days below<extra></extra>' },
    { x: names, y: N.map(s => s.up), type: 'bar', xaxis: b.x, yaxis: b.y, showlegend: false, marker: { color: P.green }, hovertemplate: '%{x}: %{y} days above<extra></extra>' }
  ];
  setAx(Pn, 0, XT('', { tickangle: -45 }), XT('% per day')); setAx(Pn, 1, XT('', { tickangle: -45 }), XT('Number of days'));
  plot('gl-chart', tr, layout(Object.assign({ barmode: 'group', bargap: 0.25, annotations: [ptitle(a, `Size of a 1-in-${Math.round(1 / al).toLocaleString('en-US')} day`), ptitle(b, `Days below −${k} sd and above +${k} sd`)] }, Pn.L)));
  const lossBig = A.filter((_, i) => Q[i][0] > Q[i][1]).map(x => SH[x]), downBig = A.filter((_, i) => N[i].down > N[i].up).map(x => SH[x]);
  out('gl-out', [T.gl(ap, bp, lossBig.length, lossBig.join(', ') || '—', A.filter((_, i) => Q[i][0] <= Q[i][1]).map(x => SH[x]).join(', ') || '—'),
    T.gl2(nf(k, 1), downBig.length, A.filter((_, i) => N[i].down <= N[i].up).map(x => SH[x]).join(', ') || '—')]);
}

// ================================================================= wiring
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['np-k'], normalPdf); PL('np-play', 'np-k', range(1, 6, 0.25), 450);
  on(['ln-s'], lognormal); PL('ln-play', 'ln-s', range(0.1, 1.5, 0.05), 250);
  on(['cl-p'], clt); PL('cl-play', 'cl-p', range(0.5, 0.02, -0.02), 350);
  on(['na-n', 'na-p'], normalApprox); PL('na-play', 'na-n', range(5, 200, 5), 300);
  on(['sh-nu', 'sh-a'], shapes); PL('sh-play', 'sh-a', range(0, 20, 0.5), 200);
  on(['dx-a', 'dx-bw'], daxDensity); PL('dx-play', 'dx-bw', range(0.25, 3, 0.25), 700);
  on(['qn-y'], qqNormalBlock); PL('qn-play', 'qn-y', range(2010, 2024, 1), 900);
  on(['td-k'], tDensities); PL('td-play', 'td-k', range(1, 6, 0.25), 400);
  on(['tf-a', 'tf-nu'], tFit); PL('tf-play', 'tf-nu', [2.2, 2.4, 2.6, 2.78, 3, 3.5, 4, 5, 6, 8, 10, 15, 20, 30], 700);
  on(['qt-nu'], qqTBlock); PL('qt-play', 'qt-nu', range(1, 9, 1), 1100);
  on(['sg-k'], sigmaDaysBlock); PL('sg-play', 'sg-k', range(2, 6, 0.5), 800);
  on(['ag-h', 'ag-s'], aggregation); PL('ag-play', 'ag-h', range(1, 63, 1), 160);
  on(['qh-a', 'qh-h'], qqHorizons); PL('qh-play', 'qh-h', [1, 2, 3, 5, 10, 15, 21, 30, 42, 63], 900);
  on(['ac-d', 'ac-s'], acfBlock); PL('ac-play', 'ac-d', range(0.25, 3, 0.25), 600);
  on(['vc-w'], clustering); PL('vc-play', 'vc-w', [5, 10, 21, 42, 63, 126, 252], 1000);
  on(['lv-y', 'lv-s'], leverageBlock); PL('lv-play', 'lv-y', range(2010, 2022, 1), 700);
  on(['gl-a', 'gl-k'], gainLoss); PL('gl-play', 'gl-a', range(0, 4, 1), 1000);
  const PFX = { 'normal-pdf': 'np', 'lognormal': 'ln', 'clt': 'cl', 'normal-approx': 'na', 'shapes': 'sh', 'dax-density': 'dx', 'qq-normal': 'qn', 't-densities': 'td',
    't-fit': 'tf', 'qq-t': 'qt', 'sigma-days': 'sg', 'aggregation': 'ag', 'qq-horizons': 'qh', 'acf': 'ac', 'clustering': 'vc', 'leverage': 'lv', 'gain-loss': 'gl' };
  const VEIL = { 'normal-approx': { out: true, charts: ['na-err'] } };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [normalPdf, lognormal, clt, normalApprox, shapes, daxDensity, qqNormalBlock, tDensities, tFit, qqTBlock, sigmaDaysBlock, aggregation, qqHorizons, acfBlock, clustering, leverageBlock, gainLoss];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH2 = { phi, Phi, normSf, normPpf, lgamma, ibeta, tpdf, tSf, tCdf, tPpf, tUnit, skewnormMoments, binomPmf, mom, sorted, quantile, acf, leverage, aggregate, kde,
  sigmaDays, tProfile, tLL, normLL, ser, sample, cltStats, approxErr, qqNormal, qqT, rollSd, hist, init };
if (window.CH2_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
