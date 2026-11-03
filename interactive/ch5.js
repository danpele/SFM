// Chapter 5 interactive page: one block per chart slide, computed in the browser from ch5_data.js.
(function () {
'use strict';
const D = window.CH5_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH5_TEXT;
const COL = D.colors, LAB = D.labels, MC = D.model_col, NUM = D.N;
const DARK = C.navy, GREEN = '#2E7D32', ORANGE = '#E67E22', PURPLE = '#8E44AD', RED = '#CD0000', BLUE = '#1A3A6E';

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
const phi = x => Math.exp(-0.5 * x * x) / SQ2PI;
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
// Student-t with 3 degrees of freedom: P(X > x) = (1/pi) [atan(1/s) - s/(1 + s^2)], s = x/sqrt(3), x > 0
const t3Sf = x => { const s = x / Math.sqrt(3); return (Math.atan(1 / s) - s / (1 + s * s)) / Math.PI; };
// GEV (xi = shape, scipy's c = -xi) and GPD
function gevT(x, xi, mu, s) { return 1 + xi * (x - mu) / s; }
function gevCdf(x, xi, mu, s) {
  if (Math.abs(xi) < 1e-12) return Math.exp(-Math.exp(-(x - mu) / s));
  const t = gevT(x, xi, mu, s); if (t <= 0) return xi > 0 ? 0 : 1;
  return Math.exp(-Math.pow(t, -1 / xi));
}
function gevPdf(x, xi, mu, s) {
  if (Math.abs(xi) < 1e-12) { const z = (x - mu) / s; return Math.exp(-z - Math.exp(-z)) / s; }
  const t = gevT(x, xi, mu, s); if (t <= 0) return 0;
  return Math.pow(t, -1 / xi - 1) * Math.exp(-Math.pow(t, -1 / xi)) / s;
}
const gevPpf = (p, xi, mu, s) => Math.abs(xi) < 1e-12 ? mu - s * Math.log(-Math.log(p)) : mu + s / xi * (Math.pow(-Math.log(p), -xi) - 1);
function gpdSf(y, xi, b) {
  if (y < 0) return 1;
  if (Math.abs(xi) < 1e-12) return Math.exp(-y / b);
  const t = 1 + xi * y / b; return t <= 0 ? 0 : Math.pow(t, -1 / xi);
}
function gpdPdf(y, xi, b) {
  if (y < 0) return 0;
  if (Math.abs(xi) < 1e-12) return Math.exp(-y / b) / b;
  const t = 1 + xi * y / b; return t <= 0 ? 0 : Math.pow(t, -1 / xi - 1) / b;
}
const gpdPpf = (p, xi, b) => Math.abs(xi) < 1e-12 ? -b * Math.log(1 - p) : b / xi * (Math.pow(1 - p, -xi) - 1);

// ================================================================= estimation
function nelderMead(f, x0, step, maxit) {
  const n = x0.length; let S = [x0.slice()];
  for (let i = 0; i < n; i++) { const v = x0.slice(); v[i] += step[i]; S.push(v); }
  let F = S.map(f);
  for (let it = 0; it < (maxit || 5000); it++) {
    const o = F.map((v, i) => i).sort((a, b) => F[a] - F[b]); S = o.map(i => S[i]); F = o.map(i => F[i]);
    let dx = 0; for (let i = 1; i <= n; i++) for (let j = 0; j < n; j++) dx = Math.max(dx, Math.abs(S[i][j] - S[0][j]));
    if (dx < 1e-10 && Math.abs(F[n] - F[0]) < 1e-11) break;
    const c = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += S[i][j] / n;
    const pt = a => c.map((v, j) => v + a * (S[n][j] - v));
    const xr = pt(-1), fr = f(xr);
    if (fr < F[0]) { const xe = pt(-2), fe = f(xe); if (fe < fr) { S[n] = xe; F[n] = fe; } else { S[n] = xr; F[n] = fr; } }
    else if (fr < F[n - 1]) { S[n] = xr; F[n] = fr; }
    else {
      const xc = fr < F[n] ? pt(-0.5) : pt(0.5), fc = f(xc);
      if (fc < Math.min(fr, F[n])) { S[n] = xc; F[n] = fc; }
      else for (let i = 1; i <= n; i++) { S[i] = S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j])); F[i] = f(S[i]); }
    }
  }
  return { x: S[0], f: F[0] };
}
function gpdNll(y, xi, b) {
  if (b <= 0) return Infinity;
  let s = 0;
  for (const v of y) { const t = 1 + xi * v / b; if (t <= 0) return Infinity; s += Math.log(t); }
  return y.length * Math.log(b) + (1 + 1 / xi) * s;
}
function fitGpd(y) {                // maximum likelihood (xi, beta) of the excesses y > 0
  const m = y.reduce((a, v) => a + v, 0) / y.length, v = y.reduce((a, w) => a + (w - m) ** 2, 0) / (y.length - 1);
  let x0 = [0.5 * (1 - m * m / v), 0.5 * m * (m * m / v + 1)];
  if (!(x0[1] > 0)) x0 = [0.1, m];
  let r = nelderMead(p => gpdNll(y, p[0], p[1]), x0, [0.05, 0.1 * x0[1]]);
  r = nelderMead(p => gpdNll(y, p[0], p[1]), r.x, [0.01, 0.02 * r.x[1]]);
  return { xi: r.x[0], beta: r.x[1], nll: r.f };
}
function quantile(s, q) { const h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
const meanA = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
const sdA = a => { const m = meanA(a); let s = 0; for (const x of a) s += (x - m) ** 2; return Math.sqrt(s / (a.length - 1)); };

const SER = {};
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function ser(k) {                   // losses in %: original order, ascending, descending with cumulative logs of the positive ones
  if (SER[k]) return SER[k];
  const R = D.loss[k], x = R.x, asc = Float64Array.from(x).sort(), n = x.length, desc = Float64Array.from(asc).reverse();
  const cl = [0]; for (let i = 0; i < n && desc[i] > 0; i++) cl.push(cl[i] + Math.log(desc[i]));
  let dates = null;
  if (R.d0 !== undefined) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); dates = dd.map(iso); }
  return (SER[k] = { x, asc, desc, cl, n, dates, mean: meanA(x), sd: sdA(x) });
}
const hillK = (S, k) => 1 / (S.cl[k] / k - Math.log(S.desc[k]));     // Hill (1975) with threshold X_(k+1)
const hillFr = (S, f) => hillK(S, Math.max(Math.trunc(f * S.n), 5));
function lsSlope(S, k) {            // least squares: slope of ln(i/n) on ln X_(i), i = 1..k, is -alpha
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < k; i++) { const a = Math.log(S.desc[i]), b = Math.log((i + 1) / S.n); sx += a; sy += b; sxx += a * a; sxy += a * b; }
  return -(k * sxy - sx * sy) / (k * sxx - sx * sx);
}
const hillQuantile = (S, k, p) => S.desc[k] * Math.pow(k / (S.n * p), 1 / hillK(S, k));
const POT = {};
function pot(k, q, x) {             // GPD above u = q-quantile of the losses (cached per series and q)
  const key = k + ':' + q;
  if (!x && POT[key]) return POT[key];
  const S = x ? { asc: Float64Array.from(x).sort(), n: x.length } : ser(k), u = quantile(S.asc, q), y = [];
  for (const v of S.asc) if (v > u) y.push(v - u);
  const f = Object.assign(fitGpd(y), { u, nu: y.length, n: S.n, q, y });
  if (!x) POT[key] = f;
  return f;
}
const potVar = (f, p) => f.u + f.beta / f.xi * (Math.pow(f.n * p / f.nu, -f.xi) - 1);
const potEs = (f, p) => potVar(f, p) / (1 - f.xi) + (f.beta - f.xi * f.u) / (1 - f.xi);
const potTail = (f, x) => f.nu / f.n * Math.pow(1 + f.xi * (x - f.u) / f.beta, -1 / f.xi);
const normVar = (S, p) => S.mean + S.sd * normPpf(1 - p);
const normEs = (S, p) => S.mean + S.sd * phi(normPpf(1 - p)) / p;
const histVar = (S, p) => quantile(S.asc, 1 - p);
function histEs(S, p) { const v = histVar(S, p); let s = 0, c = 0; for (let i = S.n - 1; i >= 0 && S.asc[i] >= v; i--) { s += S.asc[i]; c++; } return s / c; }
function meanExcess(S, u) { let s = 0, c = 0; for (let i = S.n - 1; i >= 0 && S.asc[i] > u; i--) { s += S.asc[i] - u; c++; } return c >= 5 ? s / c : null; }

// ================================================================= formatting and panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep);
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (Math.abs(x) >= 1e-3) return nf(x, Math.max(0, d - 1 - e));
  let m = x / 10 ** e, ee = e; if (+Math.abs(m).toFixed(d - 1) >= 10) { m /= 10; ee++; }
  return nf(m, d - 1) + ' × 10<sup>' + (ee < 0 ? '−' : '') + Math.abs(ee) + '</sup>';
}
// ln P(Z > x) for the standard Normal, also where P(Z > x) underflows (asymptotic series for x > 30)
const lnNormSf = x => x < 30 ? Math.log(normSf(x)) : -x * x / 2 - Math.log(x) - 0.5 * Math.log(2 * Math.PI) + Math.log(1 - 1 / (x * x) + 3 / x ** 4);
function fln(l, d) { const l10 = l / Math.LN10; if (l10 > -3) return sig(Math.exp(l), d); let e = Math.floor(l10), m = 10 ** (l10 - e); if (+m.toFixed(d - 1) >= 10) { m /= 10; e++; } return nf(m, d - 1) + ' × 10<sup>−' + Math.abs(e) + '</sup>'; }
const lvl = p => { const s = (100 * p).toFixed(2).replace(/\.?0+$/, ''); return (LANG === 'ro' ? s.replace('.', ',') : s) + '%'; };
const lvlEn = p => (100 * p).toFixed(2).replace(/\.?0+$/, '') + '%';
function panels(id, n, o) {         // n panels side by side when the chart is wide enough, stacked otherwise
  o = o || {};
  const el = $(id), stack = o.stack || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 270) * n + (o.extra || 110) : (o.h || 380)) + 'px';
  const g = stack ? (o.vgap || 0.15) : (o.gap || 0.09), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', dom = [i * (w + g), i * (w + g) + w];
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [1 - dom[1], 1 - dom[0]] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
function gridp(id, R, Cn, o) {      // R x Cn grid (row-major); one column on a narrow screen
  o = o || {};
  const el = $(id), nw = el.clientWidth < 560, rows = nw ? R * Cn : R, cols = nw ? 1 : Cn;
  el.style.height = (nw ? (o.hs || 250) * rows + (o.extra || 120) : (o.h || 260 * R + 100)) + 'px';
  const gx = cols > 1 ? (o.gx || 0.1) : 0, gy = nw ? (o.gyn || 0.075) : (o.gy || 0.2), w = (1 - gx * (cols - 1)) / cols, hh = (1 - gy * (rows - 1)) / rows;
  const L = {}, ax = [];
  for (let i = 0; i < rows * cols; i++) {
    const r = Math.floor(i / cols), c = i % cols, s = i ? String(i + 1) : '';
    L['xaxis' + s] = { domain: [c * (w + gx), c * (w + gx) + w], anchor: 'y' + s };
    L['yaxis' + s] = { domain: [1 - (r * (hh + gy) + hh), 1 - r * (hh + gy)], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, nw, rows, cols };
}
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: n ? y - 0.04 : y });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const vln = (x, color, axes, dash, w) => ({ type: 'line', xref: axes.x, yref: axes.y + ' domain', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: w || 1, dash: dash || 'dot' } });
const hln = (y, color, axes, dash, w) => ({ type: 'line', xref: axes.x + ' domain', yref: axes.y, x0: 0, x1: 1, y0: y, y1: y, line: { color, width: w || 1, dash: dash || 'solid' } });
const logspace = (a, b, n) => Array.from({ length: n }, (_, i) => 10 ** (a + (b - a) * i / (n - 1)));
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const LOGX = { type: 'log', tickvals: [0.5, 1, 2, 5, 10, 20, 50], ticktext: ['0.5', '1', '2', '5', '10', '20', '50'] };
const LOGP = r => ({ type: 'log', range: r, dtick: 1, exponentformat: 'power', showexponent: 'all' });
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const dEn = s => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`;

// ================================================================= 1. crash days against the Normal distribution
const CD_T = [1, 10, 100, 1000, 10000];
function crashBlock() {
  const Ty = CD_T[+$('cd-t').value]; label('cd-t').textContent = fint(Ty) + (LANG === 'ro' ? (Ty === 1 ? ' an' : Ty >= 20 ? ' de ani' : ' ani') : (Ty === 1 ? ' year' : ' years'));
  const Pn = panels('cd-chart', 2, { stack: true, hs: 230, extra: 110, vgap: 0.16 }), nwc = narrow('cd-chart'), tr = [], shapes = [], ann = [], items = [];
  ['sp500', 'bet'].forEach((k, j) => {
    const S = ser(k), ax = Pn.ax[j], r = S.x.map(v => -v), p = 1 / (Ty * 252), lv = -S.mean + S.sd * normPpf(p);   // returns: mean -mean(L), same sd
    tr.push(on2(ax, { x: S.dates, y: r, mode: 'lines', line: { color: COL[k], width: 0.6 }, name: `${LAB[k]} daily log return (%)`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }));
    tr.push(on2(ax, { x: [S.dates[0], S.dates[S.n - 1]], y: [lv, lv], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `Normal model: loss seen once in ${Ty >= 1000 ? Ty.toLocaleString('en-US') : Ty} ${Ty === 1 ? 'year' : 'years'}`, showlegend: j === 0, hoverinfo: 'skip' }));
    const idx = r.map((v, i) => i).sort((a, b) => r[a] - r[b]), seen = new Set(), w = [];
    for (const i of idx) { const m = S.dates[i].slice(0, 7); if (seen.has(m)) continue; seen.add(m); w.push(i); if (w.length === 3) break; }
    let last = null, dy = 0;
    (nwc ? w.slice(0, 1) : w.sort((a, b) => a - b)).forEach(i => {
      const dt = new Date(S.dates[i]); dy = last !== null && (dt - last) / 864e5 < 2200 ? dy + 13 : 0; last = dt;
      ann.push({ x: S.dates[i], y: r[i], xref: ax.x, yref: ax.y, text: dEn(S.dates[i]), showarrow: false, xanchor: nwc && i > 0.75 * S.n ? 'right' : 'left', yanchor: 'middle', xshift: nwc && i > 0.75 * S.n ? -6 : 6, yshift: dy, font: { size: nwc ? 9 : 10, color: DARK } });
    });
    ann.push(ptitle(ax, LAB[k]));
    setAx(Pn, j, AX('', { range: [S.dates[0], S.dates[S.n - 1]] }), AX('%'));
    const nb = r.filter(v => v < lv).length;
    items.push(T.cdItem(LAB[k], S.dates[0].slice(0, 4), nf(lv, 2), nb, nf(S.n * p, S.n * p < 0.1 ? 3 : 2), fint(S.n)));
  });
  plot('cd-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, shapes, legend: LEG(true, -0.07), margin: { l: 50, r: 12, t: 26, b: 30 } })));
  out('cd-out', [T.cd(fint(Ty)), ...items]);
}

// ================================================================= 2. power laws fall much more slowly
const LH_X = [2, 3, 5, 10, 20, 50];
function lightBlock() {
  const xv = LH_X[+$('lh-x').value]; label('lh-x').textContent = xv;
  const nw = single('lh-chart', 400), x = logspace(0, 2, 300);
  const law = [['Normal N(0, 1)', normSf, MC.Normal, 'solid'], ['Exponential (1)', v => Math.exp(-v), GREEN, 'solid'],
    ['Student-t, 3 degrees of freedom', t3Sf, BLUE, 'solid'], ['Pareto: P(X > x) = x^(-3)', v => v ** -3, PURPLE, 'dash']];
  const tr = law.map(([nm, f, c, dash]) => { const y = x.map(f); return { x: x.filter((_, i) => y[i] > 1e-13), y: y.filter(v => v > 1e-13), mode: 'lines', line: { color: c, width: 1.8, dash }, name: nm, hovertemplate: 'x = %{x:.1f}<br>P(X > x) = %{y:.2e}<extra></extra>' }; });
  tr.push({ x: law.map(() => xv), y: law.map(([, f]) => f(xv)), mode: 'markers', marker: { size: 8, color: law.map(l => l[2]) }, name: `P(X > ${xv})`, showlegend: false, hovertemplate: 'P(X > %{x}) = %{y:.2e}<extra></extra>' });
  plot('lh-chart', tr, layout({ shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: xv, x1: xv, y0: 0, y1: 1, line: { color: DARK, width: 1, dash: 'dot' } }],
    xaxis: AX('x (log scale)', { type: 'log', tickvals: [1, 2, 5, 10, 20, 50, 100], ticktext: ['1', '2', '5', '10', '20', '50', '100'] }),
    yaxis: AX('P(X > x) (log scale)', Object.assign(LOGP([-12, Math.log10(1.2)]), { dtick: 2 })), legend: LEG(nw, -0.18) }));
  const v = law.map(([, f]) => f(xv)), r2 = law.map(([, f]) => f(2 * xv) / f(xv));
  out('lh-out', [T.lh(xv, fln(lnNormSf(xv), 2), sig(v[1], 2), sig(v[2], 2), sig(v[3], 2)), T.lh2(xv, fln(lnNormSf(2 * xv) - lnNormSf(xv), 2), sig(r2[1], 2), nf(r2[2], 3), nf(r2[3], 3))]);
}

// ================================================================= 3. real losses on log-log axes
function loglogBlock() {
  const f = +$('ll-k').value / 100; label('ll-k').textContent = nf(100 * f, 1) + '%';
  const G = gridp('ll-chart', 2, 2, { h: 600, hs: 270, extra: 120, gy: 0.2, gx: 0.1, gyn: 0.1 }), tr = [], ann = [], items = [];
  D.assets.forEach((k, j) => {
    const S = ser(k), ax = G.ax[j], n = S.n, kk = Math.trunc(f * n), a = lsSlope(S, kk), xs = [], ys = [];
    for (let i = 0; i < n && S.desc[i] > 0.3; i++) { xs.push(S.desc[i]); ys.push((i + 1) / n); }
    tr.push(on2(ax, { x: xs, y: ys, mode: 'markers', marker: { size: 3, color: COL[k] }, name: 'Data: share of days with a loss above x', showlegend: j === 0, legendgroup: 'd', hovertemplate: 'loss %{x:.2f}%<br>share %{y:.2e}<extra></extra>' }));
    let lc = 0; for (let i = 0; i < kk; i++) lc += Math.log((i + 1) / n) + a * Math.log(S.desc[i]); const c = Math.exp(lc / kk);
    const xx = logspace(Math.log10(S.desc[kk]), Math.log10(1.5 * S.desc[0]), 50);
    tr.push(on2(ax, { x: xx, y: xx.map(v => c * v ** -a), mode: 'lines', line: { color: DARK, width: 1.4 }, name: `Least-squares line, ${lvlEn(f)} largest losses`, showlegend: j === 0, legendgroup: 'l', hoverinfo: 'skip' }));
    const xn = logspace(Math.log10(0.3), Math.log10(1.5 * S.desc[0]), 200), pn = xn.map(v => normSf((v - S.mean) / S.sd));
    tr.push(on2(ax, { x: xn.filter((_, i) => pn[i] > 1e-7), y: pn.filter(v => v > 1e-7), mode: 'lines', line: { color: MC.Normal, width: 1.3, dash: 'dash' }, name: 'Normal distribution', showlegend: j === 0, legendgroup: 'n', hoverinfo: 'skip' }));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], AX('Loss x (%, log scale)', Object.assign({ range: [Math.log10(0.3), Math.log10(1.5 * S.desc[0])] }, LOGX)));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], AX('P(L > x)', LOGP([-5, 0])));
    ann.push(ptitle(ax, `${LAB[k]}: slope -${a.toFixed(2)}`));
    items.push(T.llItem(LAB[k], fint(kk), nf(-a, 2), nf(hillK(S, kk), 2)));
  });
  plot('ll-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.12) })));
  out('ll-out', [T.ll(nf(100 * f, 1) + '%'), ...items]);
}

// ================================================================= 4./5. Hill plots
const FR = Array.from({ length: 39 }, (_, i) => +(0.005 + 0.0025 * i).toFixed(4));
function hillPlot(pfx, names, bandDefault) {
  const f = +$(pfx + '-k').value / 100, bk = $(pfx + '-b').value || bandDefault; label(pfx + '-k').textContent = nf(100 * f, 2).replace(/0$/, '') + '%';
  const nw = single(pfx + '-chart', 400), tr = [], items = [];
  names.forEach(k => {
    const S = ser(k), a = FR.map(fr => hillFr(S, fr));
    if (k === bk) {
      const kk = FR.map(fr => Math.max(Math.trunc(fr * S.n), 5)), lo = a.map((v, i) => v * (1 - 1.96 / Math.sqrt(kk[i]))), hi = a.map((v, i) => v * (1 + 1.96 / Math.sqrt(kk[i])));
      tr.push({ x: FR.map(v => 100 * v), y: lo, mode: 'lines', line: { width: 0, color: COL[k] }, showlegend: false, hoverinfo: 'skip' });
      tr.push({ x: FR.map(v => 100 * v), y: hi, mode: 'lines', line: { width: 0, color: COL[k] }, fill: 'tonexty', fillcolor: hexA(COL[k], 0.15), name: `${LAB[k]}: 95% interval`, hoverinfo: 'skip' });
    }
    tr.push({ x: FR.map(v => 100 * v), y: a, mode: 'lines', line: { color: COL[k], width: 1.8 }, name: LAB[k], hovertemplate: `${LAB[k]}<br>k/n = %{x:.2f}%<br>alpha = %{y:.2f}<extra></extra>` });
    const kk = Math.max(Math.trunc(f * S.n), 5), al = hillK(S, kk), H = NUM.hill[k];
    const se = Math.abs(f - D.hill_frac) < 1e-9 ? nf(H.se_block, 2) : null;
    items.push(T.hpItem(LAB[k], fint(kk), nf(al, 2), nf(al * (1 - 1.96 / Math.sqrt(kk)), 2), nf(al * (1 + 1.96 / Math.sqrt(kk)), 2), nf(al / Math.sqrt(kk), 2), se));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: `k = ${lvlEn(f)} of n` });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'alpha = 2: variance finite above' });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dashdot' }, name: 'alpha = 4: kurtosis finite above' });
  const ax = { x: 'x', y: 'y' };
  plot(pfx + '-chart', tr, layout({ shapes: [vln(100 * f, DARK, ax, 'dot'), hln(2, DARK, ax, 'dash'), hln(4, DARK, ax, 'dashdot')],
    xaxis: AX('Share of the largest losses used, k/n (%)', { range: [0, 10.3] }), yaxis: AX('Hill estimate of alpha', { range: [1, 6] }), legend: Object.assign(LEG(nw, -0.2), { traceorder: 'normal' }) }));
  out(pfx + '-out', [T.hp(nf(100 * f, 2).replace(/0$/, '') + '%'), ...items]);
}
function hexA(h, a) { const n = parseInt(h.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
const hillBlock = () => hillPlot('hp', D.assets, 'sp500');
const bvbBlock = () => hillPlot('hb', D.stocks, 'tlv');

// ================================================================= 6. empirical mean excess
function meBlock() {
  const q = +$('me-q').value / 100; label('me-q').textContent = nf(100 * q, 0) + '%';
  const G = gridp('me-chart', 2, 2, { h: 600, hs: 270, extra: 120, gy: 0.2, gx: 0.1, gyn: 0.1 }), tr = [], ann = [], shapes = [], items = [];
  D.assets.forEach((k, j) => {
    const S = ser(k), ax = G.ax[j], us = linspace(0.5, 0.995, 120).map(p => quantile(S.asc, p)), e = us.map(u => meanExcess(S, u)), F = pot(k, q);
    tr.push(on2(ax, { x: us, y: e, mode: 'markers', marker: { size: 4, color: COL[k] }, name: 'Empirical mean excess e(u)', showlegend: j === 0, legendgroup: 'e', hovertemplate: 'u = %{x:.2f}%<br>e(u) = %{y:.2f}%<extra></extra>' }));
    const v = linspace(F.u, us[us.length - 1], 50);
    tr.push(on2(ax, { x: v, y: v.map(w => (F.beta + F.xi * (w - F.u)) / (1 - F.xi)), mode: 'lines', line: { color: DARK, width: 1.5 }, name: 'GPD fitted above u', showlegend: j === 0, legendgroup: 'g', hoverinfo: 'skip' }));
    shapes.push(vln(F.u, DARK, ax, 'dot'));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], AX('Threshold u (daily loss, %)'));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], AX('e(u) (%)'));
    ann.push(ptitle(ax, LAB[k]));
    items.push(T.meItem(LAB[k], nf(F.u, 2), fint(F.nu), nf(meanExcess(S, F.u), 2), nf(F.xi, 3), nf(F.beta, 3)));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: `u = ${(100 * q).toFixed(0)}% quantile of the losses` });
  plot('me-chart', tr, layout(Object.assign(G.L, { annotations: ann, shapes, legend: LEG(G.nw, G.nw ? -0.05 : -0.12) })));
  out('me-out', [T.me(nf(100 * q, 0)), ...items]);
}

// ================================================================= 7. the three extreme value laws
function gevTypesBlock() {
  const s = +$('gt-s').value; label('gt-s').textContent = nf(s, 2);
  const Pn = panels('gt-chart', 2, { h: 360, hs: 260, extra: 120 }), [a, b] = Pn.ax, x = linspace(-4, 7, 600), tr = [];
  [[0, 'Gumbel (xi = 0)', BLUE], [s, `Frechet (xi = ${s.toFixed(2).replace(/0$/, '')})`, RED], [-s, `Weibull (xi = -${s.toFixed(2).replace(/0$/, '')})`, GREEN]].forEach(([xi, nm, c]) => {
    tr.push(on2(a, { x, y: x.map(v => gevPdf(v, xi, 0, 1)), mode: 'lines', line: { color: c, width: 1.8 }, name: nm, legendgroup: nm, hovertemplate: `${nm}<br>x = %{x:.2f}<br>h(x) = %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { x, y: x.map(v => gevCdf(v, xi, 0, 1)), mode: 'lines', line: { color: c, width: 1.8 }, name: nm, legendgroup: nm, showlegend: false, hovertemplate: `${nm}<br>x = %{x:.2f}<br>H(x) = %{y:.3f}<extra></extra>` }));
  });
  setAx(Pn, 0, AX('x'), AX('')); setAx(Pn, 1, AX('x'), AX('', { range: [-0.03, 1.05] }));
  plot('gt-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'Density h(x)'), ptitle(b, 'Distribution function H(x)')], legend: LEG(Pn.stack, Pn.stack ? -0.1 : -0.2) })));
  const sf = xi => 1 - gevCdf(5, xi, 0, 1);
  out('gt-out', [T.gt(nf(s, 2), nf(1 / s, 2), nf(1 / s, 2)), T.gt2(sig(sf(0), 3), sig(sf(s), 3), sf(-s) > 0 ? sig(sf(-s), 3) : '0')]);
}

// ================================================================= 8. the theorem in a simulation
const MS = D.msim;
function maxSimBlock() {
  const n = MS.ns[+$('ms-n').value], R = MS.runs[String(n)]; label('ms-n').textContent = n;
  const Pn = panels('ms-chart', 3, { h: 340, hs: 230, extra: 120, gap: 0.07 }), tr = [], ann = [];
  const cases = [['Exponential(1): M - ln n', v => gevPdf(v, 0, 0, 1), BLUE, 0.5772156649], ['Pareto(3): M / n^(1/3)', v => gevPdf(v, 1 / 3, 1, 1 / 3), RED, 1.3541179394],
    ['Uniform(0, 1): n (M - 1)', v => gevPdf(v, -1, -1, 1), GREEN, -1]];
  cases.forEach(([nm, f, c], j) => {
    const ax = Pn.ax[j], lim = MS.lims[j], bins = linspace(lim[0], lim[1], 50), w = bins[1] - bins[0], x = linspace(lim[0], lim[1], 400);
    tr.push(on2(ax, { x: bins.slice(0, -1).map(v => v + w / 2), y: R[j].dens, type: 'bar', width: w, marker: { color: hexA(c, 0.35), line: { width: 0 } }, name: `Simulated maxima (${fint(MS.reps).replace(' ', ',')} samples)`, showlegend: j === 0, legendgroup: 's', hovertemplate: '%{x:.2f}<br>density %{y:.3f}<extra></extra>' }));
    tr.push(on2(ax, { x, y: x.map(f), mode: 'lines', line: { color: DARK, width: 1.6 }, name: 'Limit law (GEV)', showlegend: j === 0, legendgroup: 'l', hoverinfo: 'skip' }));
    setAx(Pn, j, AX('', { range: lim }), AX(''));
    ann.push(ptitle(ax, nm));
  });
  plot('ms-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, bargap: 0, legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18) })));
  const extra = n !== 250;
  out('ms-out', [T.ms(n, extra), T.ms2(nf(R[0].mean, 3), nf(cases[0][3], 3), nf(R[1].mean, 3), nf(cases[1][3], 3), nf(R[2].mean, 3), nf(-1, 3))]);
}

// ================================================================= 9. GEV PP and QQ plots
function gevFit(k, model) {
  const G = NUM.gev[k], g = D.mmax[k].gumbel;
  return model === 'gumbel' ? { xi: 0, mu: g.mu, sigma: g.sigma } : { xi: G.xi, mu: G.mu, sigma: G.sigma };
}
function gevPPBlock() {
  const k = $('gq-a').value, model = $('gq-m').value, F = gevFit(k, model), M = D.mmax[k], m = Float64Array.from(M.m).sort(), n = m.length;
  const p = Array.from(m, (_, i) => (i + 0.5) / n), q = p.map(v => gevPpf(v, F.xi, F.mu, F.sigma));
  const ord = M.m.map((v, i) => i).sort((a, b) => M.m[a] - M.m[b]), ym = ord.map(i => M.ym[i]);
  const Pn = panels('gq-chart', 2, { h: 380, hs: 290, extra: 120 }), [a, b] = Pn.ax, lim = [Math.min(q[0], m[0]), Math.max(q[n - 1], m[n - 1])];
  const tr = [on2(a, { x: p, y: Array.from(m, v => gevCdf(v, F.xi, F.mu, F.sigma)), mode: 'markers', marker: { size: 4, color: COL[k] }, name: 'Monthly maxima', customdata: ym, hovertemplate: '%{customdata}<br>empirical %{x:.3f}<br>fitted %{y:.3f}<extra></extra>' }),
    on2(a, { x: [0, 1], y: [0, 1], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: '45-degree line', hoverinfo: 'skip' }),
    on2(b, { x: q, y: Array.from(m), mode: 'markers', marker: { size: 4, color: COL[k] }, showlegend: false, customdata: ym, hovertemplate: '%{customdata}<br>fitted quantile %{x:.2f}%<br>observed %{y:.2f}%<extra></extra>' }),
    on2(b, { x: lim, y: lim, mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, showlegend: false, hoverinfo: 'skip' })];
  const md = model === 'gumbel' ? 'Gumbel' : 'GEV';
  setAx(Pn, 0, AX('Empirical probability'), AX(`${md} probability`)); setAx(Pn, 1, AX(`${md} quantile (%)`), AX('Monthly maximum of daily losses (%)'));
  plot('gq-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'PP plot'), ptitle(b, 'QQ plot')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  let ks = 0; for (let i = 0; i < n; i++) { const h = gevCdf(m[i], F.xi, F.mu, F.sigma); ks = Math.max(ks, Math.abs(h - i / n), Math.abs(h - (i + 1) / n)); }
  const above = q.filter((v, i) => m[i] > v).length;
  out('gq-out', [T.gq(LAB[k], fint(n), model, nf(F.xi, 3), nf(F.mu, 2), nf(F.sigma, 2)),
    T.gq2(ym[n - 1], nf(m[n - 1], 2), nf(q[n - 1], 2), ym[n - 2], nf(m[n - 2], 2), nf(q[n - 2], 2), nf(ks, 3))]);
}

// ================================================================= 10. return level plot
const RL_T = [1, 2, 5, 10, 20, 50, 100];
function rlBlock() {
  const Ty = RL_T[+$('rl-t').value]; label('rl-t').textContent = Ty + (LANG === 'ro' ? (Ty === 1 ? ' an' : Ty >= 20 ? ' de ani' : ' ani') : (Ty === 1 ? ' year' : ' years'));
  const nw = single('rl-chart', 400), tr = [], items = [], TT = logspace(Math.log10(1 / 12 * 1.05), 2, 200);
  D.assets.forEach(k => {
    const G = NUM.gev[k], m = Float64Array.from(D.mmax[k].m).sort(), n = m.length;
    tr.push({ x: TT, y: TT.map(t => gevPpf(1 - 1 / (12 * t), G.xi, G.mu, G.sigma)), mode: 'lines', line: { color: COL[k], width: 1.8 }, name: `${LAB[k]} (GEV)`, legendgroup: k, hovertemplate: `${LAB[k]}<br>T = %{x:.1f} years<br>level %{y:.2f}%<extra></extra>` });
    tr.push({ x: Array.from(m, (_, i) => 1 / (12 * (1 - (i + 1) / (n + 1)))), y: Array.from(m), mode: 'markers', marker: { size: 4, color: COL[k], opacity: 0.6 }, showlegend: false, legendgroup: k, hovertemplate: `${LAB[k]}<br>empirical T = %{x:.1f} years<br>%{y:.2f}%<extra></extra>` });
    const z = gevPpf(1 - 1 / (12 * Ty), G.xi, G.mu, G.sigma), na = Array.from(m).filter(v => v > z).length, yrs = n / 12;
    items.push(T.rlItem(LAB[k], nf(z, 1), na, nf(yrs, 1), Ty > yrs, Ty === 10 ? [nf(G.rl10_lo, 1), nf(G.rl10_hi, 1)] : null));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: `${Ty}-year return period` });
  plot('rl-chart', tr, layout({ shapes: [vln(Ty, DARK, { x: 'x', y: 'y' }, 'dot')],
    xaxis: AX('Return period T (years, log scale)', { type: 'log', tickvals: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], ticktext: ['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'] }),
    yaxis: AX('Return level: daily loss (%)'), legend: LEG(nw, -0.2) }));
  out('rl-out', [T.rl(Ty), ...items]);
}

// ================================================================= 11. GPD densities and tails
function gpdDensBlock() {
  const s = +$('gd-x').value; label('gd-x').textContent = nf(s, 2);
  const Pn = panels('gd-chart', 2, { h: 360, hs: 260, extra: 130 }), [a, b] = Pn.ax, y = linspace(0, 6, 400), yy = logspace(-1, 2, 300), tr = [];
  [[-0.3, GREEN], [0, BLUE], [s, ORANGE], [0.6, RED]].forEach(([xi, c], j) => {
    const nm = `xi = ${+xi.toFixed(2)}`, sf = yy.map(v => gpdSf(v, xi, 1)), ok = sf.map(v => v > 1e-9);
    tr.push(on2(a, { x: y, y: y.map(v => gpdPdf(v, xi, 1)), mode: 'lines', line: { color: c, width: j === 2 ? 2.4 : 1.6 }, name: nm, legendgroup: 'g' + j, hovertemplate: `${nm}<br>y = %{x:.2f}<br>g(y) = %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { x: yy.filter((_, i) => ok[i]), y: sf.filter(v => v > 1e-9), mode: 'lines', line: { color: c, width: j === 2 ? 2.4 : 1.6 }, name: nm, legendgroup: 'g' + j, showlegend: false, hovertemplate: `${nm}<br>y = %{x:.2f}<br>P(Y > y) = %{y:.2e}<extra></extra>` }));
  });
  setAx(Pn, 0, AX('Excess y'), AX('', { range: [0, 1.05] }));
  setAx(Pn, 1, AX('Excess y (log scale)', { type: 'log', tickvals: [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], ticktext: ['0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'] }), AX('', LOGP([-6, Math.log10(1.2)])));
  plot('gd-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'Density g(y)'), ptitle(b, 'P(Y > y), log-log axes')], legend: LEG(Pn.stack, Pn.stack ? -0.1 : -0.2) })));
  out('gd-out', [T.gd(nf(s, 2), s > 0 ? nf(1 / s, 2) : null, s < 1 ? nf(1 / (1 - s), 2) : null, s < 0.5, s < 0 ? nf(-1 / s, 2) : null, sig(gpdSf(10, s, 1), 2)),
    T.gd2(sig(gpdSf(10, -0.3, 1), 2), sig(gpdSf(10, 0, 1), 2), sig(gpdSf(10, 0.6, 1), 2))]);
}

// ================================================================= 12. parameter stability
function stabBlock() {
  const q = +$('ts-q').value / 100; label('ts-q').textContent = nf(100 * q, 0) + '%';
  const Pn = panels('ts-chart', 2, { h: 370, hs: 270, extra: 110 }), [a, b] = Pn.ax, tr = [], items = [];
  ['sp500', 'bet'].forEach(k => {
    const R = NUM.stab[k], x = R.map(r => 100 * r.q);
    tr.push(on2(a, { x, y: R.map(r => r.xi - 1.96 * r.se), mode: 'lines', line: { width: 0, color: COL[k] }, showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(a, { x, y: R.map(r => r.xi + 1.96 * r.se), mode: 'lines', line: { width: 0, color: COL[k] }, fill: 'tonexty', fillcolor: hexA(COL[k], 0.15), showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(a, { x, y: R.map(r => r.xi), mode: 'lines+markers', marker: { size: 5, color: COL[k] }, line: { color: COL[k], width: 1.5 }, name: LAB[k], legendgroup: k, hovertemplate: `${LAB[k]}<br>u at %{x:.0f}%<br>xi = %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { x, y: R.map(r => r.var1), mode: 'lines+markers', marker: { size: 5, color: COL[k] }, line: { color: COL[k], width: 1.5 }, showlegend: false, legendgroup: k, hovertemplate: `${LAB[k]}<br>u at %{x:.0f}%<br>VaR 1%% = %{y:.2f}%<extra></extra>` }));
    const r = R.find(v => Math.abs(v.q - q) < 1e-9), i85 = R.filter(v => v.q >= 0.849 && v.q <= 0.951);
    items.push(T.tsItem(LAB[k], nf(r.u, 2), fint(r.n_u), nf(r.xi, 3), nf(r.xi - 1.96 * r.se, 2), nf(r.xi + 1.96 * r.se, 2), nf(r.var1, 2), nf(r.var01, 2),
      nf(Math.min(...i85.map(v => v.xi)), 2), nf(Math.max(...i85.map(v => v.xi)), 2), nf(Math.min(...i85.map(v => v.var1)), 2), nf(Math.max(...i85.map(v => v.var1)), 2)));
  });
  setAx(Pn, 0, AX('Threshold u: quantile of the losses (%)', { range: [79.5, 99.5] }), AX('xi-hat (95% interval)')); setAx(Pn, 1, AX('Threshold u: quantile of the losses (%)', { range: [79.5, 99.5] }), AX('EVT VaR 1% (%)'));
  plot('ts-chart', tr, layout(Object.assign(Pn.L, { shapes: [vln(100 * q, DARK, a), vln(100 * q, DARK, b), hln(0, DARK, a, 'dash')], legend: Object.assign(LEG(Pn.stack, Pn.stack ? -0.08 : -0.2), { traceorder: 'normal' }) })));
  out('ts-out', [T.ts(nf(100 * q, 0)), ...items]);
}

// ================================================================= 13. GPD PP and QQ plots
function gpdPPBlock() {
  const k = $('gp-a').value, q = +$('gp-q').value / 100; label('gp-q').textContent = nf(100 * q, 0) + '%';
  const F = pot(k, q), y = F.y, n = y.length, p = y.map((_, i) => (i + 0.5) / n), qq = p.map(v => gpdPpf(v, F.xi, F.beta));
  const Pn = panels('gp-chart', 2, { h: 380, hs: 290, extra: 120 }), [a, b] = Pn.ax, lim = [0, Math.max(qq[n - 1], y[n - 1])];
  const tr = [on2(a, { x: p, y: y.map(v => 1 - gpdSf(v, F.xi, F.beta)), mode: 'markers', marker: { size: 3.5, color: COL[k] }, name: 'Excesses over u', hovertemplate: 'empirical %{x:.3f}<br>GPD %{y:.3f}<extra></extra>' }),
    on2(a, { x: [0, 1], y: [0, 1], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: '45-degree line', hoverinfo: 'skip' }),
    on2(b, { x: qq, y, mode: 'markers', marker: { size: 3.5, color: COL[k] }, showlegend: false, hovertemplate: 'GPD quantile %{x:.2f}%<br>observed %{y:.2f}%<extra></extra>' }),
    on2(b, { x: lim, y: lim, mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, showlegend: false, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('Empirical probability'), AX('GPD probability')); setAx(Pn, 1, AX('GPD quantile of the excess (%)'), AX('Observed excess L - u (%)'));
  plot('gp-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'PP plot'), ptitle(b, 'QQ plot')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  out('gp-out', [T.gp(LAB[k], nf(100 * q, 0), nf(F.u, 2), fint(F.nu), nf(F.xi, 3), nf(F.beta, 3)), T.gp2(nf(y[n - 1], 2), nf(qq[n - 1], 2), nf(y[n - 2], 2), nf(qq[n - 2], 2))]);
}

// ================================================================= 14. the fitted tails
const TF_P = [0.025, 0.01, 0.005, 0.001, 0.0005, 0.0001];
function tailFitBlock() {
  const p = TF_P[+$('tf-p').value]; label('tf-p').textContent = lvl(p);
  const G = gridp('tf-chart', 2, 2, { h: 620, hs: 270, extra: 130, gy: 0.2, gx: 0.1, gyn: 0.1 }), tr = [], ann = [], shapes = [], items = [];
  D.assets.forEach((k, j) => {
    const S = ser(k), ax = G.ax[j], F = pot(k, D.pot_q), xs = [], ys = [];
    for (let i = 0; i < S.n && S.desc[i] > F.u; i++) { xs.push(S.desc[i]); ys.push((i + 1) / S.n); }
    const dat = on2(ax, { x: xs, y: ys, mode: 'markers', marker: { size: 4, color: MC.Data }, name: 'Data: share of days with a loss above x', showlegend: j === 0, legendgroup: 'd', legendrank: 1, hovertemplate: 'loss %{x:.2f}%<br>share %{y:.2e}<extra></extra>' });
    const xx = logspace(Math.log10(F.u), Math.log10(2 * S.desc[0]), 200);
    tr.push(on2(ax, { x: xx, y: xx.map(v => potTail(F, v)), mode: 'lines', line: { color: MC.EVT, width: 1.4 }, name: 'EVT: GPD above u', showlegend: j === 0, legendgroup: 'e', hovertemplate: 'x = %{x:.2f}%<br>EVT P(L > x) = %{y:.2e}<extra></extra>' }));
    const pn = xx.map(v => normSf((v - S.mean) / S.sd));
    tr.push(on2(ax, { x: xx.filter((_, i) => pn[i] > 1e-7), y: pn.filter(v => v > 1e-7), mode: 'lines', line: { color: MC.Normal, width: 1.3, dash: 'dash' }, name: 'Normal distribution', showlegend: j === 0, legendgroup: 'n', hoverinfo: 'skip' }), dat);
    shapes.push(hln(0.01, DARK, ax, 'dot', 0.8), hln(0.001, DARK, ax, 'dashdot', 0.8));
    if (p !== 0.01 && p !== 0.001) shapes.push(hln(p, GREEN, ax, 'dot', 1.4));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], AX('Loss x (%, log scale)', Object.assign({ range: [Math.log10(F.u), Math.log10(2 * S.desc[0])] }, LOGX)));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], AX('P(L > x)', LOGP([-5, Math.log10(0.15)])));
    ann.push(ptitle(ax, `${LAB[k]}: xi = ${F.xi.toFixed(2)}`));
    const kk = Math.trunc(D.hill_frac * S.n), nb = S.n * p;
    items.push(T.tfItem(LAB[k], nf(potVar(F, p), 2), nf(hillQuantile(S, kk, p), 2), nf(histVar(S, p), 2), nf(normVar(S, p), 2), nf(potEs(F, p), 2), nf(histEs(S, p), 2), nf(normEs(S, p), 2), nf(nb, 1), nb < 1));
  });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: 'p = 1%' });
  tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK, width: 1, dash: 'dashdot' }, name: 'p = 0.1%' });
  if (p !== 0.01 && p !== 0.001) tr.push({ x: [null], y: [null], mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dot' }, name: `p = ${lvlEn(p)}` });
  plot('tf-chart', tr, layout(Object.assign(G.L, { annotations: ann, shapes, legend: LEG(G.nw, G.nw ? -0.05 : -0.12) })));
  out('tf-out', [T.tf(lvl(p)), ...items]);
}

// ================================================================= 15. out of sample
function oosBlock() {
  const k = $('os-a').value, lv = $('os-p').value, p = lv === 'var1' ? 0.01 : 0.001, S = ser(k), O = NUM.oos[k];
  const i0 = S.dates.findIndex(d => d >= D.split) + 1, dates = S.dates.slice(i0), L = S.x.slice(i0), nw = single('os-chart', 400);
  const tr = [{ x: dates, y: L, mode: 'lines', line: { color: COL[k], width: 0.7 }, name: `${LAB[k]} daily loss (%)`, hovertemplate: '%{x}<br>loss %{y:.2f}%<extra></extra>' }];
  const lines = [['Normal', 'dash'], ['Historical', 'dot'], ['EVT', 'solid']], items = [];
  lines.forEach(([m, dash]) => {
    const v = O[lv][m].var, exc = L.filter(x => x > v).length;
    tr.push({ x: [dates[0], dates[dates.length - 1]], y: [v, v], mode: 'lines', line: { color: MC[m], width: 1.5, dash }, name: `VaR ${lvlEn(p)}, ${m}: ${v.toFixed(2)}%`, hoverinfo: 'skip' });
    items.push([m, v, exc]);
  });
  if (lv === 'var1') { const v = O.var01.EVT.var; tr.push({ x: [dates[0], dates[dates.length - 1]], y: [v, v], mode: 'lines', line: { color: MC.EVT, width: 1.1, dash: 'dashdot' }, name: `VaR 0.1%, EVT: ${v.toFixed(2)}%`, hoverinfo: 'skip' }); }
  plot('os-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Daily loss (%)'), legend: LEG(nw, -0.12), margin: { l: 56, r: 12, t: 20, b: 30 } }));
  const ex = items.map(([m, v, e]) => T.osItem(m, nf(v, 2), e, nf(100 * e / L.length, 2)));
  out('os-out', [T.os(LAB[k], lvl(p), fint(O.n_est), fint(L.length), nf(O[lv].expected, 1)), ...ex]);
}

// ================================================================= wiring
const PFX = { 'crash-days': 'cd', 'light-heavy': 'lh', 'loglog-real': 'll', 'hill-plot': 'hp', 'hill-bvb': 'hb', 'mean-excess': 'me', 'gev-types': 'gt',
  'maxima-sim': 'ms', 'gev-ppqq': 'gq', 'return-levels': 'rl', 'gpd-densities': 'gd', 'threshold-stability': 'ts', 'gpd-ppqq': 'gp', 'tail-fit': 'tf', oos: 'os' };
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['cd-t'], crashBlock); PL('cd-play', 'cd-t', range(0, 4, 1), 1300);
  on(['lh-x'], lightBlock); PL('lh-play', 'lh-x', range(0, 5, 1), 1000);
  on(['ll-k'], loglogBlock); PL('ll-play', 'll-k', range(0.5, 10, 0.5), 450);
  on(['hp-k', 'hp-b'], hillBlock); PL('hp-play', 'hp-k', range(0.5, 10, 0.25), 220);
  on(['hb-k', 'hb-b'], bvbBlock); PL('hb-play', 'hb-k', range(0.5, 10, 0.25), 220);
  on(['me-q'], meBlock); PL('me-play', 'me-q', range(80, 98, 1), 450);
  on(['gt-s'], gevTypesBlock); PL('gt-play', 'gt-s', range(0.1, 1, 0.05), 300);
  on(['ms-n'], maxSimBlock); PL('ms-play', 'ms-n', range(0, 6, 1), 1000);
  on(['gq-a', 'gq-m'], gevPPBlock); PL('gq-play', 'gq-m', ['gev', 'gumbel'], 1800);
  on(['rl-t'], rlBlock); PL('rl-play', 'rl-t', range(0, 6, 1), 900);
  on(['gd-x'], gpdDensBlock); PL('gd-play', 'gd-x', range(-0.5, 1, 0.05), 250);
  on(['ts-q'], stabBlock); PL('ts-play', 'ts-q', range(80, 99, 1), 400);
  on(['gp-a', 'gp-q'], gpdPPBlock); PL('gp-play', 'gp-q', range(80, 98, 1), 450);
  on(['tf-p'], tailFitBlock); PL('tf-play', 'tf-p', range(0, 5, 1), 1100);
  on(['os-a', 'os-p'], oosBlock); PL('os-play', 'os-a', D.assets, 1500);
  const VC = { out: true, charts: true }, VEIL = { 'light-heavy': VC, 'hill-plot': VC, 'hill-bvb': VC, 'mean-excess': VC, 'return-levels': VC, 'threshold-stability': VC, 'tail-fit': VC };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  [crashBlock, lightBlock, loglogBlock, hillBlock, bvbBlock, meBlock, gevTypesBlock, maxSimBlock, gevPPBlock, rlBlock, gpdDensBlock, stabBlock, gpdPPBlock, tailFitBlock, oosBlock]
    .forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH5 = { ser, hillK, hillFr, lsSlope, hillQuantile, pot, potVar, potEs, potTail, normVar, normEs, histVar, histEs, meanExcess, fitGpd, gevCdf, gevPpf, gpdSf, gpdPpf,
  t3Sf, normSf, normPpf, quantile, init, PFX,
  blocks: { crashBlock, lightBlock, loglogBlock, hillBlock, bvbBlock, meBlock, gevTypesBlock, maxSimBlock, gevPPBlock, rlBlock, gpdDensBlock, stabBlock, gpdPPBlock, tailFitBlock, oosBlock } };
if (window.CH5_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
