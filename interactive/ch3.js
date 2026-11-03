// Chapter 3 interactive page: one block per chart slide, computed in the browser from ch3_data.js.
(function () {
'use strict';
const D = window.CH3_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH3_TEXT;
const COL = D.colors, MC = D.model_col, SH = D.short;
const PURPLE = '#8E44AD', ORANGE = '#E67E22', TEAL = '#17A2B8', DBLUE = '#1A3A6E', RED = '#CD0000', GREEN = '#2E7D32', GOLD = '#B5853F';

// ================================================================= special functions
const SQ2PI = Math.sqrt(2 * Math.PI);
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
const normSf = x => 0.5 * erfc(x / Math.SQRT2);
const normPdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI);
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

// ================================================================= the stable law: FFT inversion of the characteristic function
// Same grid as StableGrid in generate_all_charts.py: N = 2^16 points, step h = 0.01, S0(alpha, beta, 1, 0).
const NF = 65536, HF = 0.01, X0 = -NF / 2 * HF, XL = (NF / 2 - 1) * HF;
const TW = (() => { const c = new Float64Array(NF / 2), s = new Float64Array(NF / 2); for (let k = 0; k < NF / 2; k++) { c[k] = Math.cos(2 * Math.PI * k / NF); s[k] = -Math.sin(2 * Math.PI * k / NF); } return { c, s }; })();
function fft(re, im) {              // in place, forward: X_j = sum_k x_k exp(-2 pi i jk/N)
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, st = n / len;
    for (let i = 0; i < n; i += len) for (let k = 0; k < half; k++) {
      const a = i + k, b = a + half, wr = TW.c[k * st], wi = TW.s[k * st];
      const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
const GRIDS = new Map();
function grid(alpha, beta) {
  const key = alpha.toFixed(6) + '|' + beta.toFixed(6);
  if (GRIDS.has(key)) { const g = GRIDS.get(key); GRIDS.delete(key); GRIDS.set(key, g); return g; }
  const re = new Float64Array(NF), im = new Float64Array(NF), dt = 2 * Math.PI / (NF * HF), tan = Math.tan(Math.PI * alpha / 2);
  for (let j = 0; j < NF; j++) {
    const k = j < NF / 2 ? j : j - NF, t = k * dt, a = Math.abs(t), sg = Math.sign(t);
    let pr, pi;
    if (alpha !== 1) { const aa = a ** alpha; pr = -aa; pi = -beta * tan * (t - sg * aa); }
    else { const lg = a > 0 ? Math.log(a) : 0; pr = -a; pi = -a * beta * (2 / Math.PI) * sg * lg; }
    const e = Math.exp(pr) * (j % 2 === 0 ? 1 : -1);
    re[j] = e * Math.cos(pi); im[j] = e * Math.sin(pi);
  }
  fft(re, im);
  const f = new Float64Array(NF), logf = new Float64Array(NF);
  for (let j = 0; j < NF; j++) { f[j] = Math.max(re[j] * dt / (2 * Math.PI), 1e-300); logf[j] = Math.log(f[j]); }
  const c = alpha >= 2 ? 0 : Math.exp(lgamma(alpha)) * Math.sin(Math.PI * alpha / 2) / Math.PI;
  const left = alpha < 2 ? c * (1 - beta) * Math.abs(X0) ** (-alpha) : 0, right = alpha < 2 ? c * (1 + beta) * XL ** (-alpha) : 0;
  const F = new Float64Array(NF), S = new Float64Array(NF);
  let s = 0; F[0] = Math.min(Math.max(left, 0), 1);
  for (let j = 1; j < NF; j++) { s += 0.5 * (f[j] + f[j - 1]) * HF; F[j] = Math.min(Math.max(left + s, 0), 1); }
  s = 0; S[NF - 1] = Math.min(Math.max(right, 0), 1);
  for (let j = NF - 2; j >= 0; j--) { s += 0.5 * (f[j] + f[j + 1]) * HF; S[j] = Math.min(Math.max(right + s, 0), 1); }
  const xj = j => (j - NF / 2) * HF;
  const interp = (z, A) => {
    if (z <= X0) return A[0]; if (z >= XL) return A[NF - 1];
    let i = Math.floor((z - X0) / HF); if (i >= NF - 1) i = NF - 2; if (xj(i) > z) i--; if (xj(i + 1) <= z) i++;
    const x0 = xj(i), x1 = xj(i + 1); return A[i] + (z - x0) * (A[i + 1] - A[i]) / (x1 - x0);
  };
  const G = {
    alpha, beta, c, f, logf, F, S, x: xj,
    pdf: z => (Math.abs(z) > XL && alpha < 2) ? alpha * c * (1 + Math.sign(z) * beta) * Math.abs(z) ** (-alpha - 1) : Math.exp(interp(z, logf)),
    cdf: z => z <= 0 ? interp(z, F) : 1 - interp(z, S),
    sf: z => z > 0 ? interp(z, S) : 1 - (z <= 0 ? interp(z, F) : 1 - interp(z, S)),
    ppf: p => {
      let i0 = 0;
      if (F[0] < 1e-12) { let lo = 0, hi = NF - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (F[m] < 1e-12) lo = m + 1; else hi = m; } i0 = lo; }
      if (p <= F[i0]) return xj(i0); if (p >= F[NF - 1]) return xj(NF - 1);
      let lo = i0, hi = NF - 1;                       // largest j with F[j] <= p
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F[m] <= p) lo = m; else hi = m; }
      const d = F[lo + 1] - F[lo]; return d > 0 ? xj(lo) + (p - F[lo]) * HF / d : xj(lo);
    },
    mode: (a, b) => { let jb = -1, best = -Infinity; for (let j = Math.ceil((a - X0) / HF); j <= Math.floor((b - X0) / HF); j++) if (logf[j] > best) { best = logf[j]; jb = j; } return xj(jb); }
  };
  GRIDS.set(key, G);
  if (GRIDS.size > 24) GRIDS.delete(GRIDS.keys().next().value);
  return G;
}
const s1Shift = (alpha, beta) => Math.abs(alpha - 1) < 1e-12 ? 0 : beta * Math.tan(Math.PI * alpha / 2);   // delta0 of the S1 law with delta1 = 0
// fitted models: 'Normal', 'Student-t', 'Stable' (S0)
const MODELS = ['Normal', 'Student-t', 'Stable'];
function model(f) {
  const n = f.normal, t = f.t, s = f.stable, G = grid(s.alpha, s.beta);
  return {
    Normal: { pdf: x => normPdf(x, n.mu, n.sigma), cdf: x => Phi((x - n.mu) / n.sigma), ppf: p => n.mu + n.sigma * normPpf(p) },
    'Student-t': { pdf: x => tpdf((x - t.loc) / t.scale, t.nu) / t.scale, cdf: x => tCdf((x - t.loc) / t.scale, t.nu), ppf: p => t.loc + t.scale * tPpf(p, t.nu) },
    Stable: { pdf: x => G.pdf((x - s.delta0) / s.gamma) / s.gamma, cdf: x => G.cdf((x - s.delta0) / s.gamma), ppf: p => s.delta0 + s.gamma * G.ppf(p) }
  };
}

// ================================================================= data and statistics
const SER = {};
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function ser(k) {
  if (SER[k]) return SER[k];
  const R = D.ret[k], days = [R.d0];
  for (const dd of R.dd) days.push(days[days.length - 1] + dd);
  return (SER[k] = { r: R.r, days, dates: days.map(iso), s: Float64Array.from(R.r).sort() });
}
function quantile(s, q) { const h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
const linspace = (a, b, n) => { const st = (b - a) / (n - 1), r = []; for (let i = 0; i < n; i++) r.push(a + i * st); r[n - 1] = b; return r; };
const logspace = (a, b, n) => linspace(a, b, n).map(v => 10 ** v);
function histDensity(r, edges) {    // numpy.histogram(density=True) on uniform bins
  const nb = edges.length - 1, lo = edges[0], hi = edges[nb], c = new Array(nb).fill(0);
  for (const x of r) {
    if (x < lo || x > hi) continue;
    let i = Math.floor((x - lo) / (hi - lo) * nb); if (i >= nb) i = nb - 1;
    if (x < edges[i]) i--; else if (i + 1 < nb && x >= edges[i + 1]) i++;
    c[i]++;
  }
  const tot = c.reduce((a, b) => a + b, 0);
  return c.map((v, i) => v / tot / (edges[i + 1] - edges[i]));
}
const mean = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
const sdev = a => { const m = mean(a); let s = 0; for (const x of a) s += (x - m) ** 2; return Math.sqrt(s / (a.length - 1)); };

// ================================================================= formatting and panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep);
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  if (Math.abs(x) >= 1e-4) { const e = Math.floor(Math.log10(Math.abs(x))); return nf(x, Math.max(0, d - 1 - e)); }
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e;
  return nf(m, d - 1) + ' × 10<sup>' + (e < 0 ? '−' : '') + Math.abs(e) + '</sup>';
}
const spct = (x, d) => sig(100 * x, d || 3) + '%';
const fnum = (x, d) => Math.abs(x) >= 1000 ? fint(x) : nf(x, d);
const XT = (title, extra) => AX(title, extra);
// n panels in one chart: side by side when the chart is wide enough, stacked otherwise. Returns axis names and layout.
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.stack || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 270) * n + (o.extra || 110) : (o.h || 380)) + 'px';
  const g = stack ? (o.vgap || 0.15) : (o.gap || 0.09), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
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
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
// single-panel chart: taller on a narrow screen, legend moved down below the axis title
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 360) + (n ? 130 : 0)) + 'px'; return n; }
// log axis with 1-2-5 ticks (decades only on a narrow screen), plain numbers
function logTicks(nw) { const v = []; for (let e = -1; e <= 5; e++) (nw ? [1] : [1, 2, 5]).forEach(m => v.push(+(m * 10 ** e).toPrecision(1))); return { tickvals: v, ticktext: v.map(String), dtick: undefined }; }
const LEG = (n, y) => Object.assign(layout().legend, { y: n ? y - 0.16 : y });
const LOGY = (title, extra) => XT(title, Object.assign({ type: 'log', exponentformat: 'power', dtick: 1 }, extra || {}));
const vl = (x, color, axes, dash) => ({ type: 'line', xref: axes ? axes.x : 'x', yref: (axes ? axes.y : 'y') + ' domain', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1.3, dash: dash || 'dash' } });
const diag = (lo, hi, axes, show) => ({ x: [lo, hi], y: [lo, hi], mode: 'lines', xaxis: axes ? axes.x : 'x', yaxis: axes ? axes.y : 'y', name: '45-degree line', showlegend: !!show, legendgroup: 'd',
  line: { color: C.navy, width: 1, dash: 'dash' }, hoverinfo: 'skip' });

// ================================================================= 1. stability in a simulation
function stabQQ() {
  const S = D.stab, n = +$('sq-n').value, a = +$('sq-a').value; label('sq-a').textContent = nf(a, 2);
  const q1 = S.one, qs = S.sums[n].q, fa = n ** (1 / a), fs = Math.sqrt(n);
  const lim = [Math.min(...q1), Math.max(...q1)], nw = single('sq-chart');
  const tr = [
    { x: q1, y: qs.map(v => v / fa), mode: 'markers', name: `sum of ${n} draws / ${n}^(1/${a.toFixed(2)}) = ${fa.toFixed(2)}`, marker: { color: DBLUE, size: 5 },
      hovertemplate: 'one draw %{x:.2f}<br>rescaled sum %{y:.2f}<extra></extra>' },
    { x: q1, y: qs.map(v => v / fs), mode: 'markers', name: `sum of ${n} draws / sqrt(${n}) = ${fs.toFixed(2)}`, marker: { color: RED, size: 5, symbol: 'square' },
      hovertemplate: 'one draw %{x:.2f}<br>rescaled sum %{y:.2f}<extra></extra>' },
    diag(lim[0], lim[1], null, true)];
  plot('sq-chart', tr, layout({ xaxis: XT('Quantiles of one draw of S(1.7, 0, 1, 0)'), yaxis: XT('Quantiles of the rescaled sum'), legend: LEG(nw, -0.18) }));
  const q99 = S.sums[n].q99;
  out('sq-out', [T.sq(n, nf(a, 2), nf(fa, 2), nf(q99 / fa, 2), nf(fs, 2), nf(q99 / fs, 2), nf(S.q99_one, 2))]);
}

// ================================================================= 2. GCLT in a simulation
const GC_COL = { 1: ORANGE, 10: GREEN, 100: PURPLE };
function gcltBlock() {
  const Gd = D.gclt, k = +$('gc-k').value; label('gc-k').textContent = nf(k, 1);
  const G = grid(1.5, 0), gam = Gd.gamma, x = linspace(-15, 15, 601), e = Gd.edges, mid = e.slice(1).map((v, i) => 0.5 * (v + e[i]));
  const sd = 2 * gam * G.ppf(0.75) / (2 * normPpf(0.75)), nw = single('gc-chart');
  const tr = Gd.ns.map(n => ({ x: mid, y: Gd.hist[n].map(h => h > 0 ? h : null), mode: 'lines', line: { shape: 'hvh', color: GC_COL[n], width: 1.3 }, name: `n = ${n}`,
    hovertemplate: `n = ${n}<br>x %{x:.2f}<br>density %{y:.2e}<extra></extra>` }));
  tr.push({ x, y: x.map(v => G.pdf(v / gam) / gam), mode: 'lines', line: { color: DBLUE, width: 2 }, name: `stable limit S(1.5, 0, ${gam.toFixed(2)}, 0)`, hovertemplate: 'x %{x:.2f}<br>density %{y:.2e}<extra></extra>' });
  tr.push({ x, y: x.map(v => normPdf(v, 0, sd)), mode: 'lines', line: { color: RED, width: 1.4, dash: 'dash' }, name: 'Normal, same interquartile range', hovertemplate: 'x %{x:.2f}<br>density %{y:.2e}<extra></extra>' });
  plot('gc-chart', tr, layout({ shapes: [vl(-k, C.navy), vl(k, C.navy)], xaxis: XT('Normalised sum S_n / n^(1/alpha)', { range: [-15, 15] }), yaxis: LOGY('Density (log scale)', { range: [-5, 0] }), legend: LEG(nw, -0.2) }));
  const i = Gd.ks.findIndex(v => Math.abs(v - k) < 1e-9);
  out('gc-out', [T.gc(nf(k, 1), Gd.ns.map(n => `n = ${n}: ${spct(Gd.tail[n][i])}`).join(', '), spct(2 * G.sf(k / gam)), spct(2 * normSf(k / sd)))]);
}

// ================================================================= 3. the effect of alpha
const DA = [[2.0, DBLUE], [1.7, TEAL], [1.5, GREEN], [1.0, ORANGE], [0.7, RED]];
function densityAlpha() {
  const a = +$('da-a').value; label('da-a').textContent = nf(a, 2);
  const x = linspace(-8, 8, 801), Pn = panels('da-chart', 2), [p, q] = Pn.ax, tr = [];
  const lab = v => `alpha = ${v}` + (v === 2 ? ' (Normal, variance 2)' : v === 1 ? ' (Cauchy)' : '');
  DA.forEach(([v, c]) => {
    const y = v === 0.7 ? D.da07 : x.map(grid(v, 0).pdf);      // alpha = 0.7: the slide's curve (scipy), as exported
    tr.push({ x, y, mode: 'lines', xaxis: p.x, yaxis: p.y, name: lab(v), legendgroup: String(v), line: { color: c, width: 1.6 }, hovertemplate: `alpha = ${v}<br>x %{x:.2f}<br>density %{y:.4f}<extra></extra>` });
    tr.push({ x, y, mode: 'lines', xaxis: q.x, yaxis: q.y, showlegend: false, legendgroup: String(v), line: { color: c, width: 1.6 }, hovertemplate: `alpha = ${v}<br>x %{x:.2f}<br>density %{y:.2e}<extra></extra>` });
  });
  const G = grid(a, 0), s07 = Math.abs(a - 0.7) < 1e-9, y = s07 ? D.da07 : x.map(G.pdf);
  tr.push({ x, y, mode: 'lines', xaxis: p.x, yaxis: p.y, name: `chosen alpha = ${a.toFixed(2)}`, legendgroup: 'c', line: { color: PURPLE, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  tr.push({ x, y, mode: 'lines', xaxis: q.x, yaxis: q.y, showlegend: false, legendgroup: 'c', line: { color: PURPLE, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.2e}<extra></extra>' });
  setAx(Pn, 0, XT(Pn.stack ? '' : 'x', { range: [-5, 5] }), XT('', { rangemode: 'tozero' })); setAx(Pn, 1, XT('x', { range: [-8, 8] }), LOGY('', { dtick: 1 }));
  plot('da-chart', tr, layout(Object.assign({ annotations: [ptitle(p, 'Density'), ptitle(q, 'Density, log scale')] }, Pn.L)));
  const N2 = grid(2, 0);
  const f0 = s07 ? D.da07[400] : G.pdf(0), f5 = s07 ? D.da07[650] : G.pdf(5);
  out('da-out', [T.da(nf(a, 2), nf(f0, 4), sig(f5, 3), nf(N2.pdf(0), 4), sig(N2.pdf(5), 3), fnum(f5 / N2.pdf(5), 1), spct(2 * G.sf(5)))]);
}

// ================================================================= 4. the effect of beta and gamma
const DBB = [[-1, RED], [-0.5, ORANGE], [0, DBLUE], [0.5, TEAL], [1, GREEN]], DBG = [[0.5, PURPLE], [1, DBLUE], [2, GOLD]];
function densityBeta() {
  const b = +$('db-b').value, g = +$('db-g').value; label('db-b').textContent = nf(b, 1); label('db-g').textContent = nf(g, 2);
  const x = linspace(-6, 6, 801), Pn = panels('db-chart', 2, { h: 430, hs: 270, extra: 110 }), [p, q] = Pn.ax, tr = [];
  DBB.forEach(([v, c]) => { const G = grid(1.3, v); tr.push({ x, y: x.map(G.pdf), mode: 'lines', xaxis: p.x, yaxis: p.y, name: `beta = ${v}`, line: { color: c, width: 1.6 }, hovertemplate: `beta = ${v}<br>x %{x:.2f}<br>density %{y:.4f}<extra></extra>` }); });
  const Gb = grid(1.3, b);
  tr.push({ x, y: x.map(Gb.pdf), mode: 'lines', xaxis: p.x, yaxis: p.y, name: `chosen beta = ${b.toFixed(1)}`, line: { color: C.navy, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  const G7 = grid(1.7, 0);
  DBG.forEach(([v, c]) => tr.push({ x, y: x.map(z => G7.pdf(z / v) / v), mode: 'lines', xaxis: q.x, yaxis: q.y, name: `gamma = ${v}`, line: { color: c, width: 1.6 }, hovertemplate: `gamma = ${v}<br>x %{x:.2f}<br>density %{y:.4f}<extra></extra>` }));
  tr.push({ x, y: x.map(z => G7.pdf(z / g) / g), mode: 'lines', xaxis: q.x, yaxis: q.y, name: `chosen gamma = ${g.toFixed(2)}`, line: { color: ORANGE, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  setAx(Pn, 0, XT(Pn.stack ? '' : 'x'), XT('', { rangemode: 'tozero' })); setAx(Pn, 1, XT('x'), XT('', { rangemode: 'tozero' }));
  plot('db-chart', tr, layout(Object.assign({ annotations: [ptitle(p, 'alpha = 1.3, gamma = 1: the effect of beta'), ptitle(q, 'alpha = 1.7, beta = 0: the effect of gamma')], legend: Object.assign(layout().legend, { y: Pn.stack ? -0.12 : -0.2 }) }, Pn.L)));
  const r3 = Gb.sf(3), l3 = Gb.cdf(-3), r10 = Gb.sf(10), l10 = Gb.cdf(-10), w = G7.ppf(0.95) - G7.ppf(0.05);
  out('db-out', [T.db(nf(b, 1), spct(r3), spct(l3), b === -1 ? '—' : nf(r10 / l10, 2), Math.abs(b) === 1 ? '—' : nf((1 + b) / (1 - b), 2), nf(Gb.mode(-3, 3), 2)),
    T.dg(nf(g, 2), nf(g * w, 2), nf(w, 2))]);
}

// ================================================================= 5. S1 and S0 near alpha = 1
const SS = [[0.8, RED], [0.95, ORANGE], [1.05, TEAL], [1.2, DBLUE]];
function s0s1() {
  const a = +$('ss-a').value; label('ss-a').textContent = nf(a, 2);
  const x = linspace(-6, 8, 701), Pn = panels('ss-chart', 2), [p, q] = Pn.ax, tr = [];
  SS.forEach(([v, c]) => {
    const G = grid(v, 0.8), sh = s1Shift(v, 0.8);
    tr.push({ x, y: x.map(z => G.pdf(z - sh)), mode: 'lines', xaxis: p.x, yaxis: p.y, name: `alpha = ${v}`, legendgroup: String(v), line: { color: c, width: 1.6 }, hovertemplate: `alpha = ${v}<br>x %{x:.2f}<br>density %{y:.4f}<extra></extra>` });
    tr.push({ x, y: x.map(G.pdf), mode: 'lines', xaxis: q.x, yaxis: q.y, showlegend: false, legendgroup: String(v), line: { color: c, width: 1.6 }, hovertemplate: `alpha = ${v}<br>x %{x:.2f}<br>density %{y:.4f}<extra></extra>` });
  });
  const G = grid(a, 0.8), sh = s1Shift(a, 0.8);
  tr.push({ x, y: x.map(z => G.pdf(z - sh)), mode: 'lines', xaxis: p.x, yaxis: p.y, name: `chosen alpha = ${a.toFixed(2)}`, legendgroup: 'c', line: { color: PURPLE, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  tr.push({ x, y: x.map(G.pdf), mode: 'lines', xaxis: q.x, yaxis: q.y, showlegend: false, legendgroup: 'c', line: { color: PURPLE, width: 3, dash: 'dot' }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' });
  setAx(Pn, 0, XT(Pn.stack ? '' : 'x', { range: [-6, 8] }), XT('', { rangemode: 'tozero' })); setAx(Pn, 1, XT('x', { range: [-6, 8] }), XT('', { rangemode: 'tozero' }));
  plot('ss-chart', tr, layout(Object.assign({ annotations: [ptitle(p, 'S1: beta = 0.8, gamma = 1, delta = 0'), ptitle(q, 'S0: beta = 0.8, gamma = 1, delta = 0')] }, Pn.L)));
  const m0 = G.mode(-3, 3), m1 = m0 + sh;
  out('ss-out', [T.ss(nf(a, 2), nf(sh, 2), nf(m1, 2), nf(m0, 2), m1 < -6 || m1 > 8)]);
}

// ================================================================= 6. Normal, Cauchy and Lévy
const levySf = x => x > 0 ? 1 - erfc(Math.sqrt(1 / (2 * x))) : 1;
function specialCases() {
  const k = +$('sc-x').value; label('sc-x').textContent = nf(k, 1);
  const x = linspace(-5, 8, 1301), xl = x.filter(v => v > 0), nw = single('sc-chart');
  const tr = [
    { x, y: x.map(v => normPdf(v, 0, 1)), mode: 'lines', name: 'Normal: alpha = 2, gamma = 1/sqrt(2) (variance 1)', line: { color: DBLUE, width: 1.8 }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' },
    { x, y: x.map(v => 1 / (Math.PI * (1 + v * v))), mode: 'lines', name: 'Cauchy: alpha = 1, beta = 0, gamma = 1', line: { color: ORANGE, width: 1.8 }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' },
    { x: xl, y: xl.map(v => Math.sqrt(1 / (2 * Math.PI)) * v ** -1.5 * Math.exp(-1 / (2 * v))), mode: 'lines', name: 'Lévy: alpha = 1/2, beta = 1, gamma = 1 (S1)', line: { color: GREEN, width: 1.8 }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' }];
  plot('sc-chart', tr, layout({ shapes: k <= 8 ? [vl(k, C.navy)] : [], xaxis: XT('x', { range: [-5, 8] }), yaxis: XT('Density', { rangemode: 'tozero' }), legend: LEG(nw, -0.16) }));
  out('sc-out', [T.sc(nf(k, 1), spct(normSf(k)), spct(0.5 - Math.atan(k) / Math.PI), spct(levySf(k)))]);
}

// ================================================================= 7. tails on log-log axes
const TL = [[2.0, DBLUE], [1.9, TEAL], [1.7, GREEN], [1.5, RED]];
function tailsLoglog() {
  const a = +$('tl-a').value; label('tl-a').textContent = nf(a, 2);
  const x = logspace(0, 2.2, 60), tr = [], nw = single('tl-chart');
  const curve = (v, c, w, name, show) => {
    const G = grid(v, 0), sf = v < 2 ? x.map(G.sf) : x.map(z => normSf(z / Math.SQRT2)), ok = sf.map(s => s > 1e-12);
    tr.push({ x: x.filter((_, i) => ok[i]), y: sf.filter((_, i) => ok[i]), mode: 'lines', name, line: { color: c, width: w }, hovertemplate: `alpha = ${v}<br>x %{x:.1f}<br>P(X > x) %{y:.2e}<extra></extra>` });
    if (v < 2) tr.push({ x, y: x.map(z => G.c * z ** -v), mode: 'lines', showlegend: false, line: { color: c, width: 1, dash: 'dot' }, hoverinfo: 'skip' });
  };
  TL.forEach(([v, c]) => curve(v, c, 1.8, `alpha = ${v}` + (v === 2 ? ' (Normal)' : '')));
  curve(a, PURPLE, 3, `chosen alpha = ${a.toFixed(2)}`);
  plot('tl-chart', tr, layout({ xaxis: XT('x (log scale)', { type: 'log', tickvals: [1, 2, 5, 10, 20, 50, 100], range: [0, 2.2] }), yaxis: LOGY('P(X > x) (log scale)', { range: [-8, 0], dtick: 2 }),
    annotations: [{ x: 2.18, y: Math.log10(2e-2), text: 'dotted: c x^(-alpha)', showarrow: false, xanchor: 'right', font: { size: 11, color: C.navy } }], legend: LEG(nw, -0.2) }));
  const G = grid(a, 0), s10 = G.sf(10), s20 = G.sf(20);
  out('tl-out', [T.tl(nf(a, 2), spct(s10), spct(G.c * 10 ** -a), nf(s20 / s10, 3), nf(2 ** -a, 3))]);
}

// ================================================================= 8. the sample variance does not settle
const RV_COL = [RED, ORANGE, GREEN];
function runVarSim() {
  const R = D.rvs, a = R.alphas[+$('rv-a').value]; label('rv-a').textContent = nf(a, 2);
  const nw = single('rv-chart'), P = R.paths[String(a)], tr = [{ x: R.k, y: R.normal, mode: 'lines', name: 'Normal = S(2, 0, 1, 0), true variance 2', line: { color: DBLUE, width: 1.6 }, hovertemplate: 'n = %{x}<br>variance %{y:.3f}<extra></extra>' }];
  P.forEach((p, i) => tr.push({ x: R.k, y: p.v, mode: 'lines', name: `stable alpha = ${a}, path ${i + 1}`, line: { color: RV_COL[i], width: 1.1 }, hovertemplate: 'n = %{x}<br>variance %{y:.3f}<extra></extra>' }));
  plot('rv-chart', tr, layout({ xaxis: XT('Sample size n (log scale)', { type: 'log', tickangle: 0, tickvals: nw ? [10, 100, 1000, 10000] : [10, 100, 1000, 10000, 20000], ticktext: nw ? ['10', '100', '1000', '10000'] : ['10', '100', '1000', '10000', '20000'] }),
    yaxis: LOGY('Sample variance (log scale)', logTicks(nw)), legend: LEG(nw, -0.18) }));
  out('rv-out', [T.rv(nf(a, 2), P.map(p => nf(p.v1000, 2)).join(', '), P.map(p => nf(p.vend, 2)).join(', '), nf(R.normal_end, 3), P.map(p => fnum(p.max, 1)).join(', '))]);
}

// ================================================================= 9. CMS draws against the exact density
function cmsBlock() {
  const M = D.cms, a = +$('cm-a').value, b = +$('cm-b').value; label('cm-b').textContent = nf(b, 1);
  const H = M.h[`${a}_${b}`], e = M.edges, mid = e.slice(1).map((v, i) => 0.5 * (v + e[i]));
  const G = grid(a, b), sh = s1Shift(a, b), x = linspace(-8, 8, 801), nw = single('cm-chart');
  const tr = [
    { x: mid, y: H.c.map(c => c / (M.n * 0.1)), type: 'bar', width: 0.1, name: 'CMS draws (histogram)', marker: { color: TEAL, opacity: 0.55 }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' },
    { x, y: x.map(z => G.pdf(z - sh)), mode: 'lines', name: `exact density S1(${a}, ${b}, 1, 0)`, line: { color: DBLUE, width: 1.8 }, hovertemplate: 'x %{x:.2f}<br>density %{y:.4f}<extra></extra>' }];
  plot('cm-chart', tr, layout({ bargap: 0, xaxis: XT('x', { range: [-8, 8] }), yaxis: XT('Density', { rangemode: 'tozero' }), legend: LEG(nw, -0.16) }));
  out('cm-out', [T.cm(nf(a, 1), nf(b, 1), nf(H.q01, 2), nf(H.q99, 2), nf(G.ppf(0.01) + sh, 2), nf(G.ppf(0.99) + sh, 2), nf(H.med, 2), nf(100 * H.out / M.n, 2) + '%'),
    ...(a === 1.5 && b === 0.5 ? [T.cmKs(nf(M.ks_stat, 4), nf(M.ks_p, 2))] : [])]);
}

// ================================================================= 10. McCulloch's map from nu_alpha to alpha
function mccMap() {
  const y = +$('mm-y').value; label('mm-y').textContent = String(y);
  const nw = single('mm-chart'), Lm = D.mcc_line, tr = [{ x: Lm.nu, y: Lm.alpha['0.0'], mode: 'lines', name: 'McCulloch (1986), Table III, nu_beta = 0', line: { color: DBLUE, width: 1.8 }, hovertemplate: 'nu_alpha %{x:.2f}<br>alpha %{y:.3f}<extra></extra>' }];
  D.assets.forEach(k => { const m = D.mcc[k][y]; tr.push({ x: [m.nu_a], y: [m.alpha], mode: 'markers', name: SH[k], marker: { symbol: 'star', size: 15, color: COL[k] }, hovertemplate: `${SH[k]}<br>nu_alpha %{x:.3f}<br>alpha %{y:.3f}<extra></extra>` }); });
  plot('mm-chart', tr, layout({ xaxis: XT('nu_alpha = (q95 − q05) / (q75 − q25)', { range: [2.3, 6.1] }), yaxis: XT('alpha', { range: [0.95, 2.05] }), legend: LEG(nw, -0.18) }));
  out('mm-out', [T.mm(String(y)), ...D.assets.map(k => { const m = D.mcc[k][y]; return T.mmItem(SH[k], fint(m.n), nf(m.nu_a, 3), nf(m.nu_b, 3), nf(m.alpha, 3)); })]);
}

// ================================================================= 11. McCulloch and ML in a Monte Carlo study
function mcBlock() {
  const R = +$('mc-r').value; label('mc-r').textContent = String(R);
  const tr = [], res = [], nw = single('mc-chart');
  ['500', '2500'].forEach(n => [['mcculloch', 'McCulloch', ORANGE], ['ml', 'ML', DBLUE]].forEach(([k, name, c]) => {
    const v = D.mc[n][k].slice(0, R); res.push({ n, name, m: mean(v), sd: sdev(v) });
    tr.push({ y: v, x: v.map(() => `${name}<br>n = ${n}`), type: 'box', name: `${name}, n = ${n}`, showlegend: false, quartilemethod: 'linear', boxpoints: 'outliers',
      fillcolor: c + '8C', line: { color: C.navy, width: 1 }, marker: { color: RED, size: 4 }, hovertemplate: '%{y:.3f}<extra></extra>' });
  }));
  tr.push({ x: [null], y: [null], mode: 'lines', name: 'true alpha = 1.7', line: { color: RED, dash: 'dash', width: 1 } });
  plot('mc-chart', tr, layout({ shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1.7, y1: 1.7, line: { color: RED, width: 1, dash: 'dash' } }],
    xaxis: XT(''), yaxis: XT('alpha-hat'), legend: LEG(nw, -0.2) }));
  const it = n => { const a = res.find(r => r.n === n && r.name === 'McCulloch'), b = res.find(r => r.n === n && r.name === 'ML'); return T.mcItem(n === '500' ? '500' : fint(2500), nf(a.m, 3), nf(a.sd, 3), nf(b.m, 3), nf(b.sd, 3), nf(a.sd / b.sd, 2)); };
  out('mc-out', [T.mc(R), it('500'), it('2500')]);
}

// ================================================================= 12-13. fitted densities on a log scale
function fitDensity(pf, names) {
  const k = +$(pf + '-k').value; label(pf + '-k').textContent = nf(k, 1);
  const Pn = panels(pf + '-chart', 2, { h: 400, hs: 290, extra: 160 }), tr = [], ann = [], items = [];
  names.forEach((nm, i) => {
    const a = Pn.ax[i], f = D.fits[nm], M = model(f), s = ser(nm).s, lo = quantile(s, 0.0005), hi = quantile(s, 0.9995);
    const e = linspace(lo, hi, 70), h = histDensity(ser(nm).r, e), mid = e.slice(1).map((v, j) => 0.5 * (v + e[j])), pos = h.map((v, j) => [mid[j], v]).filter(p => p[1] > 0);
    tr.push({ x: pos.map(p => p[0]), y: pos.map(p => p[1]), mode: 'markers', xaxis: a.x, yaxis: a.y, name: 'Data (histogram)', showlegend: i === 0, legendgroup: 'd', marker: { color: MC.Data, size: 5 },
      hovertemplate: 'return %{x:.2f}%<br>density %{y:.2e}<extra></extra>' });
    const x = linspace(lo, hi, 400);
    MODELS.forEach(m => tr.push({ x, y: x.map(M[m].pdf), mode: 'lines', xaxis: a.x, yaxis: a.y, name: m === 'Stable' ? 'Stable (ML)' : m, showlegend: i === 0, legendgroup: m, line: { color: MC[m], width: 1.6 },
      hovertemplate: `${m}<br>return %{x:.2f}%<br>density %{y:.2e}<extra></extra>` }));
    const hp = pos.map(p => p[1]), ymin = Math.max(Math.min(...hp) / 3, 1e-4), ymax = Math.max(...hp) * 3, xk = k * f.normal.sigma;
    setAx(Pn, i, XT(Pn.stack && i === 0 ? '' : 'Daily log return (%)'), LOGY('', { range: [Math.log10(ymin), Math.log10(ymax)] }));
    Pn.L.shapes = (Pn.L.shapes || []).concat(xk >= lo && xk <= hi ? [vl(xk, C.navy, a)] : []);
    ann.push(ptitle(a, `${SH[nm]}: alpha-hat = ${f.stable.alpha.toFixed(2)}`));
    const j = mid.findIndex((_, jj) => xk >= e[jj] && xk < e[jj + 1]), hd = j >= 0 ? h[j] : 0;
    items.push(T.fdItem(SH[nm], nf(xk, 2), hd > 0 ? sig(hd, 3) : '—', ...MODELS.map(m => sig(M[m].pdf(xk), 3))));
  });
  plot(pf + '-chart', tr, layout(Object.assign({ annotations: ann, legend: Object.assign(layout().legend, { y: Pn.stack ? -0.12 : -0.18 }) }, Pn.L)));
  out(pf + '-out', [T.fd(nf(k, 1)), ...items]);
}

// ================================================================= 14-15. QQ plots against the three models
const QQ_P = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001];
const QQ_GRID = (() => { const p = []; for (let i = 1; i < 400; i++) { const v = (i - 0.5) / 399; if (v > 0.003 && v < 0.997) p.push(v); } return [0.001, 0.002, ...p, 0.998, 0.999]; })();
const QQ_MK = { Normal: 'circle', 'Student-t': 'square', Stable: 'triangle-up' };
const QQC = {};
function qqData(nm) {
  if (QQC[nm]) return QQC[nm];
  const s = ser(nm).s, M = model(D.fits[nm]);
  return (QQC[nm] = { emp: QQ_GRID.map(p => quantile(s, p)), q: Object.fromEntries(MODELS.map(m => [m, QQ_GRID.map(M[m].ppf)])), M, s });
}
const ppct = p => (100 * p).toFixed(p < 0.01 ? (p < 0.0025 ? 1 : 2) : p === 0.025 ? 1 : 0).replace(/0+$/, '').replace(/\.$/, '');
function qqReal(pf, names) {
  const p = QQ_P[+$(pf + '-p').value], pl = LANG === 'ro' ? ppct(p).replace('.', ',') : ppct(p); label(pf + '-p').textContent = pl + '%';
  const Pn = panels(pf + '-chart', 2, { h: 400, hs: 290, extra: 160 }), tr = [], ann = [], items = [];
  names.forEach((nm, i) => {
    const a = Pn.ax[i], Q = qqData(nm);
    MODELS.forEach(m => tr.push({ x: Q.q[m], y: Q.emp, mode: 'markers', xaxis: a.x, yaxis: a.y, name: m, showlegend: i === 0, legendgroup: m, marker: { color: MC[m], size: 5, symbol: QQ_MK[m] },
      hovertemplate: `${m}<br>model %{x:.2f}%<br>empirical %{y:.2f}%<extra></extra>` }));
    const lo = Math.min(...Q.emp), hi = Math.max(...Q.emp);
    tr.push(diag(lo, hi, a, i === 0));
    const e = quantile(Q.s, p), mq = MODELS.map(m => Q.M[m].ppf(p));
    tr.push({ x: mq, y: mq.map(() => e), mode: 'markers', xaxis: a.x, yaxis: a.y, name: `quantile at p = ${ppct(p)}%`, showlegend: i === 0, legendgroup: 'p',
      marker: { color: 'rgba(0,0,0,0)', size: 14, line: { color: C.navy, width: 2 } }, hovertemplate: 'model %{x:.2f}%<br>empirical %{y:.2f}%<extra></extra>' });
    setAx(Pn, i, XT(Pn.stack && i === 0 ? '' : 'Model quantile (%)'), XT('Empirical quantile (%)'));
    ann.push(ptitle(a, SH[nm]));
    items.push(T.qqItem(SH[nm], nf(-e, 2), ...mq.map(v => nf(-v, 2))));
  });
  plot(pf + '-chart', tr, layout(Object.assign({ annotations: ann, legend: Object.assign(layout().legend, { y: Pn.stack ? -0.12 : -0.18 }) }, Pn.L)));
  out(pf + '-out', [T.qq(pl + '%'), ...items]);
}

// ================================================================= 16-17. the left tail on log-log axes
const TRC = {};
function tailData(nm) {
  if (TRC[nm]) return TRC[nm];
  const r = ser(nm).r, n = r.length, loss = r.filter(v => v < 0).map(v => -v).sort((a, b) => b - a);
  const emp = loss.map((_, i) => (i + 1) / n), keep = loss.map(v => v >= 0.5);
  const L = loss.filter((_, i) => keep[i]), E = emp.filter((_, i) => keep[i]), x = logspace(Math.log10(0.5), Math.log10(4 * L[0]), 80), M = model(D.fits[nm]);
  return (TRC[nm] = { n, r, L, E, x, M, pr: Object.fromEntries(MODELS.map(m => [m, x.map(v => M[m].cdf(-v))])) });
}
function tailsReal(pf, names) {
  const k = +$(pf + '-x').value; label(pf + '-x').textContent = nf(k, 1) + '%';
  const Pn = panels(pf + '-chart', 2, { h: 400, hs: 290, extra: 160 }), tr = [], ann = [], items = [], shapes = [];
  names.forEach((nm, i) => {
    const a = Pn.ax[i], Q = tailData(nm);
    tr.push({ x: Q.L, y: Q.E, mode: 'markers', xaxis: a.x, yaxis: a.y, name: 'Data: share of days with loss > x', showlegend: i === 0, legendgroup: 'd', marker: { color: MC.Data, size: 3 },
      hovertemplate: 'loss %{x:.2f}%<br>share %{y:.2e}<extra></extra>' });
    MODELS.forEach(m => { const ok = Q.pr[m].map(v => v > 1e-7);
      tr.push({ x: Q.x.filter((_, j) => ok[j]), y: Q.pr[m].filter((_, j) => ok[j]), mode: 'lines', xaxis: a.x, yaxis: a.y, name: m, showlegend: i === 0, legendgroup: m, line: { color: MC[m], width: 1.6 },
        hovertemplate: `${m}<br>loss %{x:.2f}%<br>P %{y:.2e}<extra></extra>` }); });
    tr.push({ x: [0.5, 4 * Q.L[0]], y: [1 / Q.n, 1 / Q.n], mode: 'lines', xaxis: a.x, yaxis: a.y, name: 'one day in the sample', showlegend: i === 0, legendgroup: 'o', line: { color: C.navy, width: 0.8, dash: 'dot' }, hoverinfo: 'skip' });
    if (k <= 4 * Q.L[0]) shapes.push(vl(k, C.navy, a));
    setAx(Pn, i, XT(Pn.stack && i === 0 ? '' : 'Loss x (%, log scale)', { type: 'log', range: [Math.log10(0.5), Math.log10(4 * Q.L[0])], tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100, 200] }),
      LOGY(i === 0 || Pn.stack ? 'P(loss > x)' : '', { range: [-6, 0], dtick: 1 }));
    ann.push(ptitle(a, SH[nm]));
    let obs = 0; for (const v of Q.r) if (v < -k) obs++;
    items.push(T.trItem(SH[nm], obs, ...MODELS.map(m => { const e = Q.n * Q.M[m].cdf(-k); return e < 0.05 ? '≈ 0' : nf(e, 1); })));
  });
  plot(pf + '-chart', tr, layout(Object.assign({ shapes, annotations: ann, legend: Object.assign(layout().legend, { y: Pn.stack ? -0.12 : -0.18 }) }, Pn.L)));
  out(pf + '-out', [T.tr(nf(k, 1)), ...items]);
}

// ================================================================= 18. aggregation: does alpha stay the same?
const AG_L = [0.8, 0.9, 0.95, 0.99, 0.999], AG_H = ['daily', 'weekly', 'monthly'];
function aggregationBlock() {
  const est = $('ag-e').value, lv = AG_L[+$('ag-c').value], z = normPpf(0.5 + lv / 2);
  label('ag-c').textContent = (LANG === 'ro' ? String(100 * lv).replace('.', ',') : String(+(100 * lv).toFixed(1))) + '%';
  const tr = [], items = [], over = [], nw = single('ag-chart');
  D.assets.forEach((k, j) => {
    const A = D.agg[k], xs = [0, 1, 2].map(v => v + (j - 1.5) * 0.08);
    const a = AG_H.map(h => est === 'ml' ? A[h].alpha : A[h].alpha_mcc), e = AG_H.map(h => est === 'ml' && A[h].se != null ? z * A[h].se : 0);
    tr.push({ x: xs, y: a, mode: 'lines+markers', name: SH[k], line: { color: COL[k], width: 1.6 }, marker: { color: COL[k], size: 7 },
      error_y: est === 'ml' ? { type: 'data', array: e, visible: true, color: COL[k], thickness: 1.3, width: 4 } : { visible: false }, hovertemplate: `${SH[k]}<br>alpha-hat %{y:.3f}<extra></extra>` });
    items.push(T.agItem(SH[k], a.map(v => nf(v, 2)).join(' → '), est === 'ml' && A.monthly.at_bound));
    if (est === 'ml' && !A.monthly.at_bound) {
      const ov = A.daily.alpha + z * A.daily.se >= A.monthly.alpha - z * A.monthly.se && A.monthly.alpha + z * A.monthly.se >= A.daily.alpha - z * A.daily.se;
      over.push([SH[k], ov]);
    }
  });
  tr.push({ x: [null], y: [null], mode: 'lines', name: 'alpha = 2 (Normal)', line: { color: C.navy, dash: 'dash', width: 1 } });
  plot('ag-chart', tr, layout({ shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 2, y1: 2, line: { color: C.navy, width: 1, dash: 'dash' } }],
    xaxis: XT('', { tickvals: [0, 1, 2], ticktext: ['Daily', 'Weekly', 'Monthly'], range: [-0.4, 2.4], zeroline: false }), yaxis: XT(est === 'ml' ? `alpha-hat (ML, ${(100 * lv).toFixed(lv === 0.999 ? 1 : 0)}% interval)` : 'alpha-hat (McCulloch)'),
    legend: LEG(nw, -0.14) }));
  out('ag-out', [T.ag(est, label('ag-c').textContent, nf(z, 3)), ...items,
    ...(est === 'ml' ? [T.agOv(over.filter(o => !o[1]).map(o => o[0]).join(', ') || '—', over.filter(o => o[1]).map(o => o[0]).join(', ') || '—')] : [])]);
}

// ================================================================= 19. does the sample variance settle?
function runVarReal() {
  const nw = single('rr-chart'), k = $('rr-a').value, S = ser(k), r = S.r, n = r.length, P = D.rvr[k];
  const v = new Array(n); let s1 = 0, s2 = 0;
  for (let i = 0; i < n; i++) { s1 += r[i]; s2 += r[i] * r[i]; v[i] = s2 / (i + 1) - (s1 / (i + 1)) ** 2; }
  const dx = P.idx.map(i => S.dates[i]);
  const tr = P.paths.map((p, j) => ({ x: dx, y: p, mode: 'lines', name: `simulated from the fitted stable law, path ${j + 1}`, line: { color: RV_COL[j], width: 1 }, hovertemplate: '%{x}<br>variance %{y:.3f}<extra></extra>' }));
  tr.push({ x: dx, y: P.idx.map(i => v[i]), mode: 'lines', name: `${SH[k]} daily log returns (%)`, line: { color: DBLUE, width: 2 }, hovertemplate: '%{x}<br>variance %{y:.3f}<extra></extra>' });
  plot('rr-chart', tr, layout({ xaxis: XT(''), yaxis: LOGY('Running sample variance (log scale)', logTicks(nw)), legend: LEG(nw, -0.12) }));
  const i10 = S.dates.findIndex(d => d >= '2010-01-01'), f = D.fits[k].stable;
  out('rr-out', [T.rr(SH[k], nf(f.alpha, 3), nf(v[n - 1], 2), S.dates[0] < '2010-01-01' ? nf(v[i10], 2) : null, P.paths.map(p => nf(p[p.length - 1], 1)).join(', '))]);
}

// ================================================================= wiring
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  const fd1 = () => fitDensity('f1', ['bet', 'sp500']), fd2 = () => fitDensity('f2', ['dax', 'btc']);
  const qq1 = () => qqReal('q1', ['bet', 'sp500']), qq2 = () => qqReal('q2', ['dax', 'btc']);
  const tr1 = () => tailsReal('t1', ['bet', 'sp500']), tr2 = () => tailsReal('t2', ['dax', 'btc']);
  on(['sq-n', 'sq-a'], stabQQ); PL('sq-play', 'sq-a', range(1, 2, 0.05), 350);
  on(['gc-k'], gcltBlock); PL('gc-play', 'gc-k', range(2, 15, 0.5), 400);
  on(['da-a'], densityAlpha); PL('da-play', 'da-a', range(2, 0.6, -0.05), 300);
  on(['db-b', 'db-g'], densityBeta); PL('db-play', 'db-b', range(-1, 1, 0.1), 350);
  on(['ss-a'], s0s1); PL('ss-play', 'ss-a', range(0.8, 1.2, 0.01), 250);
  on(['sc-x'], specialCases); PL('sc-play', 'sc-x', range(1, 20, 0.5), 250);
  on(['tl-a'], tailsLoglog); PL('tl-play', 'tl-a', range(1.95, 1.1, -0.05), 350);
  on(['rv-a'], runVarSim); PL('rv-play', 'rv-a', range(0, D.rvs.alphas.length - 1, 1), 1000);
  on(['cm-a', 'cm-b'], cmsBlock); PL('cm-play', 'cm-b', range(-1, 1, 0.5), 1000);
  on(['mm-y'], mccMap); PL('mm-play', 'mm-y', range(2000, 2020, 1), 600);
  on(['mc-r'], mcBlock); PL('mc-play', 'mc-r', range(10, 200, 10), 450);
  on(['f1-k'], fd1); PL('f1-play', 'f1-k', range(-1, -10, -0.5), 500);
  on(['f2-k'], fd2); PL('f2-play', 'f2-k', range(-1, -10, -0.5), 500);
  on(['q1-p'], qq1); PL('q1-play', 'q1-p', range(0, 5, 1), 1000);
  on(['q2-p'], qq2); PL('q2-play', 'q2-p', range(0, 5, 1), 1000);
  on(['t1-x'], tr1); PL('t1-play', 't1-x', range(2, 25, 1), 500);
  on(['t2-x'], tr2); PL('t2-play', 't2-x', range(2, 25, 1), 500);
  on(['ag-e', 'ag-c'], aggregationBlock); PL('ag-play', 'ag-c', range(0, 4, 1), 1000);
  on(['rr-a'], runVarReal); PL('rr-play', 'rr-a', ['bet', 'sp500', 'dax', 'btc'], 1500);
  const PFX = { 'stability-qq': 'sq', gclt: 'gc', 'density-alpha': 'da', 'density-beta': 'db', 's0-s1': 'ss', 'special-cases': 'sc', 'tails-loglog': 'tl', 'running-var-sim': 'rv',
    cms: 'cm', 'mcculloch-map': 'mm', 'estimators-mc': 'mc', 'fit-density-1': 'f1', 'fit-density-2': 'f2', 'qq-real-1': 'q1', 'qq-real-2': 'q2', 'tails-real-1': 't1',
    'tails-real-2': 't2', aggregation: 'ag', 'running-var-real': 'rr' };
  const VEIL = { 'density-alpha': { out: true, charts: true }, 'special-cases': { out: true, charts: true }, 'tails-loglog': { out: true, charts: true } };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [stabQQ, gcltBlock, densityAlpha, densityBeta, s0s1, specialCases, tailsLoglog, runVarSim, cmsBlock, mccMap, mcBlock, fd1, fd2, qq1, qq2, tr1, tr2, aggregationBlock, runVarReal];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH3 = { grid, s1Shift, model, ser, quantile, histDensity, normPpf, normSf, tPpf, tCdf, levySf, qqData, tailData, linspace, logspace, mean, sdev, init,
  blocks: { stabQQ, gcltBlock, densityAlpha, densityBeta, s0s1, specialCases, tailsLoglog, runVarSim, cmsBlock, mccMap, mcBlock, fitDensity, qqReal, tailsReal, aggregationBlock, runVarReal } };
if (window.CH3_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
