// Chapter 4 interactive page: one block per chart slide, computed in the browser from ch4_data.js.
(function () {
'use strict';
const D = window.CH4_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH4_TEXT;
const COL = D.colors, SH = D.short, PAL = D.pal, PALETTE = D.palette, N = D.N;
const DBLUE = PAL.blue, RED = PAL.red, GREEN = PAL.green, AMBER = PAL.amber, PURPLE = PAL.purple, TEAL = PAL.teal, CRIMSON = PAL.crimson;
const DARK = C.navy;

// ================================================================= special functions
const SQ2PI = Math.sqrt(2 * Math.PI);
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
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
function lchoose(n, k) { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }
const binomPmf = (k, n, p) => Math.exp(lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));

// ================================================================= data and statistics
const SER = {};
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpackDays(R) { const days = [R.d0]; for (const dd of R.dd) days.push(days[days.length - 1] + dd); return days; }
function ser(k) {
  if (SER[k]) return SER[k];
  const R = D.ret[k], days = unpackDays(R);
  return (SER[k] = { r: R.r, dates: days.map(iso), s: Float64Array.from(R.r).sort() });
}
function quantile(s, q) { const h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
const mean = a => { let s = 0; for (const x of a) s += x; return s / a.length; };
const sdev = (a, dd) => { const m = mean(a); let s = 0; for (const x of a) s += (x - m) ** 2; return Math.sqrt(s / (a.length - (dd === undefined ? 1 : dd))); };
function pearson(x, y) {
  const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxy / Math.sqrt(sxx * syy);
}
const lagCorr = (r, h, f) => { const a = [], b = []; for (let i = h; i < r.length; i++) { a.push(f ? f(r[i]) : r[i]); b.push(f ? f(r[i - h]) : r[i - h]); } return pearson(a, b); };
function acf1(x) { const m = mean(x); let num = 0, den = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - m; den += d * d; if (i) num += d * (x[i - 1] - m); } return num / den; }
function exKurt(x) { const m = mean(x); let m2 = 0, m4 = 0; for (const v of x) { const d = (v - m) ** 2; m2 += d; m4 += d * d; } m2 /= x.length; m4 /= x.length; return m4 / (m2 * m2) - 3; }
const linspace = (a, b, n) => { const st = (b - a) / (n - 1), r = []; for (let i = 0; i < n; i++) r.push(a + i * st); r[n - 1] = b; return r; };
function histCounts(r, lo, hi, nb) {     // numpy.histogram on uniform bins (last bin closed)
  const c = new Array(nb).fill(0), w = (hi - lo) / nb;
  for (const x of r) { if (x < lo || x > hi) continue; let i = Math.floor((x - lo) / w); if (i >= nb) i = nb - 1; c[i]++; }
  return c;
}

// ================================================================= formatting and panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep);
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  if (Math.abs(x) >= 1e-3) { const e = Math.floor(Math.log10(Math.abs(x))); return nf(x, Math.max(0, d - 1 - e)); }
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e;
  return nf(m, d - 1) + ' × 10<sup>' + (e < 0 ? '−' : '') + Math.abs(e) + '</sup>';
}
const spct = (x, d) => sig(100 * x, d || 3) + '%';
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
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
// R x Cn grid of panels (row-major); on a narrow screen the grid becomes nr x nc (default: one column)
function gridp(id, R, Cn, o) {
  o = o || {};
  const el = $(id), nw = el.clientWidth < 560, rows = nw ? (o.nr || R * Cn) : R, cols = nw ? (o.nc || 1) : Cn;
  el.style.height = (nw ? (o.hs || 230) * rows + (o.extra || 120) : (o.h || 200 * R + 100)) + 'px';
  const gx = cols > 1 ? (nw ? (o.gxn || o.gx || 0.1) : (o.gx || 0.1)) : 0, gy = nw ? (o.gyn || 0.09) : (o.gy || 0.2), w = (1 - gx * (cols - 1)) / cols, hh = (1 - gy * (rows - 1)) / rows;
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
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 360) + (n ? 130 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: n ? y - 0.06 : y });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const hseg = (y, x0, x1, color, axes, dash) => ({ type: 'line', xref: axes.x, yref: axes.y, x0, x1, y0: y, y1: y, line: { color, width: 1.2, dash: dash || 'dash' } });
const vln = (x, color, axes, dash) => ({ type: 'line', xref: axes.x, yref: axes.y + ' domain', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1.2, dash: dash || 'dot' } });
const hln = (y, color, axes, dash, w) => ({ type: 'line', xref: axes.x + ' domain', yref: axes.y, x0: 0, x1: 1, y0: y, y1: y, line: { color, width: w || 1, dash: dash || 'solid' } });
// log axis with plain tick labels: 1-2-5 per decade, or decades only (dec = true)
function logTicks(dec) {
  const v = []; for (let e = -1; e <= 7; e++) (dec ? [1] : [1, 2, 5]).forEach(m => v.push(+(m * 10 ** e).toPrecision(1)));
  return { tickvals: v, ticktext: v.map(x => x >= 1e6 ? x / 1e6 + 'M' : x >= 1e4 ? x / 1e3 + 'k' : String(x)) };
}

// ================================================================= 1. a discrete and a continuous random variable
function rvBlock() {
  const k = $('rv-a').value, U = D.up[k], S = ser(k), r = S.r, p = U.p, Pn = panels('rv-chart', 2, { h: 380, hs: 280, extra: 130, vgap: 0.2 }), [a, b] = Pn.ax;
  const K = [0, 1, 2, 3, 4, 5], bin = K.map(i => binomPmf(i, 5, p));
  const rf = r.filter(x => x > -6 && x < 6), lo = Math.min(...rf), hi = Math.max(...rf), c = histCounts(rf, lo, hi, 120), w = (hi - lo) / 120;
  const mid = c.map((_, i) => lo + (i + 0.5) * w), m = mean(r), s = sdev(r), x = linspace(-6, 6, 400);
  const tr = [
    on2(a, { x: K.map(i => i - 0.18), y: U.emp, type: 'bar', width: 0.36, name: 'weeks in the data', marker: { color: DBLUE }, hovertemplate: 'K = %{customdata}<br>share %{y:.3f}<extra></extra>', customdata: K }),
    on2(a, { x: K.map(i => i + 0.18), y: bin, type: 'bar', width: 0.36, name: `binomial B(5, ${p.toFixed(3)})`, marker: { color: AMBER }, hovertemplate: 'K = %{customdata}<br>B(5, p) %{y:.3f}<extra></extra>', customdata: K }),
    on2(b, { x: mid, y: c.map(v => v / (rf.length * w)), type: 'bar', width: w, name: 'daily log returns (histogram)', marker: { color: TEAL, opacity: 0.55 }, hovertemplate: 'return %{x:.2f}%<br>density %{y:.3f}<extra></extra>' }),
    on2(b, { x, y: x.map(v => normPdf(v, m, s)), mode: 'lines', name: 'Normal density, same mean and sd', line: { color: RED, width: 1.8 }, hoverinfo: 'skip' })];
  setAx(Pn, 0, XT('K = number of up days in a five-day week', { tickvals: K }), XT('Probability'));
  setAx(Pn, 1, XT('daily log return (%)', { range: [-6, 6] }), XT('Density'));
  plot('rv-chart', tr, layout(Object.assign(Pn.L, { barmode: 'overlay', annotations: [ptitle(a, SH[k]), ptitle(b, SH[k])], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  out('rv-out', [T.rv(SH[k], fint(U.n_weeks), nf(p, 3), nf(U.mean, 2), nf(5 * p, 2), nf(U.var, 2), nf(5 * p * (1 - p), 2), nf(U.var / (5 * p * (1 - p)), 2))]);
}

// ================================================================= 2. empirical CDF and the 1% quantile
const CQ_A = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05];
function cdfBlock() {
  const k = $('cq-a').value, al = CQ_A[+$('cq-p').value], S = ser(k), xs = S.s, n = xs.length, m = mean(S.r), s = sdev(S.r);
  const lvl = pc(al, 2).replace(/([.,]\d*?)0+%$/, '$1%').replace(/[.,]%$/, '%'); label('cq-p').textContent = lvl;
  const Pn = panels('cq-chart', 2, { h: 380, hs: 280, extra: 120, vgap: 0.2 }), [a, b] = Pn.ax, tr = [];
  const qe = quantile(xs, al), qn = m + s * normPpf(al), x = linspace(-12, 12, 800);
  [[a, -6, 6, true], [b, -12, -1, false]].forEach(([ax, lo, hi, leg]) => {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) if (xs[i] >= lo && xs[i] <= hi) { X.push(xs[i]); Y.push((i + 1) / n); }
    tr.push(on2(ax, { x: X, y: Y, mode: 'lines', line: { shape: 'hv', color: DBLUE, width: 1.5 }, name: 'empirical CDF', showlegend: leg, legendgroup: 'e', hovertemplate: 'x %{x:.2f}%<br>F(x) %{y:.4f}<extra></extra>' }));
    const xx = x.filter(v => v >= lo && v <= hi);
    tr.push(on2(ax, { x: xx, y: xx.map(v => Phi((v - m) / s)), mode: 'lines', line: { color: RED, width: 1.5, dash: 'dash' }, name: 'Normal CDF, same mean and sd', showlegend: leg, legendgroup: 'n', hovertemplate: 'x %{x:.2f}%<br>Normal F(x) %{y:.2e}<extra></extra>' }));
  });
  tr.push(on2(b, { x: [null], y: [null], mode: 'lines', line: { color: GREEN, dash: 'dot' }, name: `level ${(100 * al).toFixed(al < 0.01 ? 2 : 1).replace(/\.?0+$/, '')}%` }));
  setAx(Pn, 0, XT('daily log return x (%)'), XT('F(x) = P(r <= x)'));
  setAx(Pn, 1, XT('daily log return x (%)', { range: [-12, -1] }), XT('F(x) = P(r <= x)', { type: 'log', range: [-5, Math.log10(0.5)], exponentformat: 'power', dtick: 1 }));
  plot('cq-chart', tr, layout(Object.assign(Pn.L, { shapes: [hln(al, GREEN, b, 'dot'), vln(qe, DBLUE, b), vln(qn, RED, b)],
    annotations: [ptitle(a, 'whole distribution'), ptitle(b, 'left tail, log scale')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  const pe = x => S.r.filter(v => v < x).length / n;
  out('cq-out', [T.cq(SH[k], fint(n), lvl, nf(-qe, 2), nf(-qn, 2), nf(qe / qn, 2)),
    T.cq2(spct(pe(-3), 3), spct(Phi((-3 - m) / s), 2), spct(pe(-5), 2), spct(Phi((-5 - m) / s), 2), sig(pe(-5) / Phi((-5 - m) / s), 3))]);
}

// ================================================================= 3. joint distributions
const JT = {};
function jointSer(key) {
  if (JT[key]) return JT[key];
  const J = D.joint[key], dates = unpackDays(J).map(iso);
  return (JT[key] = { x: J.x, y: J.y, years: dates.map(d => +d.slice(0, 4)) });
}
function jointBlock() {
  const yi = +$('jt-y').value, yr = yi ? 1999 + yi : null;
  label('jt-y').textContent = yr || (LANG === 'ro' ? 'toți anii' : 'all years');
  const Pn = panels('jt-chart', 2, { h: 420, hs: 330, extra: 120 }), tr = [], items = [];
  [['sp500', 'dax'], ['sp500', 'bet']].forEach(([u, v], j) => {
    const J = jointSer(`${u}_${v}`), ax = Pn.ax[j], cl = z => Math.max(-12, Math.min(12, z)), all = pearson(J.x, J.y);
    tr.push(on2(ax, { x: J.x.map(cl), y: J.y.map(cl), mode: 'markers', marker: { size: 3, color: COL[v], opacity: yr ? 0.18 : 0.35 }, name: `${SH[u]} vs ${SH[v]}: correlation ${all.toFixed(2)}`,
      hovertemplate: `${SH[u]} %{x:.2f}%<br>${SH[v]} %{y:.2f}%<extra></extra>` }));
    let ry = null, ny = 0;
    if (yr) {
      const xi = [], yv = [];
      J.years.forEach((y, i) => { if (y === yr) { xi.push(J.x[i]); yv.push(J.y[i]); } });
      ny = xi.length; ry = ny > 2 ? pearson(xi, yv) : null;
      tr.push(on2(ax, { x: xi.map(cl), y: yv.map(cl), mode: 'markers', marker: { size: 5, color: PURPLE }, name: j ? '_' : `days of ${yr}`, showlegend: !j, legendgroup: 'y',
        hovertemplate: `${yr}<br>${SH[u]} %{x:.2f}%<br>${SH[v]} %{y:.2f}%<extra></extra>` }));
    }
    setAx(Pn, j, XT(`${SH[u]} daily log return (%)`, { range: [-12, 12] }), XT(`${SH[v]} daily log return (%)`, { range: [-12, 12] }));
    items.push(T.jtItem(SH[u], SH[v], fint(J.x.length), nf(all, 2), yr, ny, ry === null ? '—' : nf(ry, 2)));
  });
  plot('jt-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18) })));
  out('jt-out', [T.jt(yr), ...items]);
}

// ================================================================= 4. uncorrelated but dependent
function udBlock() {
  const h = +$('ud-h').value; label('ud-h').textContent = h;
  const Pn = panels('ud-chart', 2, { h: 400, hs: 300, extra: 140 }), [a, b] = Pn.ax, X = D.xsq, tr = [], shapes = [], items = [];
  tr.push(on2(a, { x: X, y: X.map(v => v * v), mode: 'markers', marker: { size: 4, color: PURPLE, opacity: 0.4 }, name: `simulated: Y = X^2, Corr(X, Y) = ${pearson(X, X.map(v => v * v)).toFixed(3)}`, hoverinfo: 'skip' }));
  const R1 = [], R2 = [], idx = D.assets.map((_, i) => i);
  D.assets.forEach((k, i) => {
    const r = ser(k).r, c1 = lagCorr(r, h), c2 = lagCorr(r, h, v => v * v), band = 1.96 / Math.sqrt(r.length);
    R1.push(c1); R2.push(c2);
    shapes.push(hseg(band, i - 0.42, i + 0.42, GREEN, b), hseg(-band, i - 0.42, i + 0.42, GREEN, b));
    items.push(T.udItem(SH[k], nf(c1, 3), nf(c2, 3), nf(band, 3)));
  });
  tr.push(on2(b, { x: idx.map(i => i - 0.18), y: R1, type: 'bar', width: 0.36, marker: { color: DBLUE }, name: `Corr(r_t, r_{t-${h}})`, hovertemplate: '%{y:.3f}<extra></extra>' }));
  tr.push(on2(b, { x: idx.map(i => i + 0.18), y: R2, type: 'bar', width: 0.36, marker: { color: RED }, name: `Corr(r_t^2, r_{t-${h}}^2)`, hovertemplate: '%{y:.3f}<extra></extra>' }));
  tr.push(on2(b, { x: [null], y: [null], mode: 'lines', line: { color: GREEN, dash: 'dash' }, name: 'i.i.d. band +/- 1.96/sqrt(n)' }));
  shapes.push(hln(0, DARK, b, 'solid', 0.6));
  setAx(Pn, 0, XT('X ~ N(0, 1)'), XT('Y = X^2'));
  setAx(Pn, 1, XT('', { tickvals: idx, ticktext: D.assets.map(k => SH[k]) }), XT(`lag-${h} correlation`, { range: [-0.15, 0.45] }));
  plot('ud-chart', tr, layout(Object.assign(Pn.L, { shapes, legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16) })));
  out('ud-out', [T.ud(h, nf(pearson(X, X.map(v => v * v)), 3)), ...items]);
}

// ================================================================= 5. conditional moments given yesterday's move
function condMoments(r, q) {
  const cur = r.slice(1), prev = r.slice(0, -1).map(Math.abs), n = cur.length, sp = Float64Array.from(prev).sort();
  const e = []; for (let i = 0; i <= q; i++) e.push(quantile(sp, i / q));
  const g = Array.from({ length: q }, () => []);
  for (let i = 0; i < n; i++) { let j = 0; while (j < q - 1 && prev[i] > e[j + 1]) j++; g[j].push(cur[i]); }
  const mu = mean(cur), M = g.map(mean), Sd = g.map(x => sdev(x, 0)), W = g.map(x => x.length / n);
  let within = 0, between = 0; for (let j = 0; j < q; j++) { within += W[j] * Sd[j] ** 2; between += W[j] * (M[j] - mu) ** 2; }
  return { mean: M, sd: Sd, cnt: g.map(x => x.length), within, between, total: sdev(cur, 0) ** 2, sdAll: sdev(cur, 0), lo: e.slice(0, -1), hi: e.slice(1) };
}
function condBlock() {
  const q = +$('cd-q').value; label('cd-q').textContent = q;
  const Pn = panels('cd-chart', 2, { h: 390, hs: 290, extra: 130 }), [a, b] = Pn.ax, tr = [], shapes = [hln(0, DARK, a, 'solid', 0.6)], items = [];
  const idx = Array.from({ length: q }, (_, i) => i + 1), wd = 0.72 / 2;
  ['sp500', 'btc'].forEach((k, j) => {
    const c = condMoments(ser(k).r, q), off = j ? wd / 2 : -wd / 2;
    tr.push(on2(a, { x: idx.map(i => i + off), y: c.mean, type: 'bar', width: wd, marker: { color: COL[k] }, name: SH[k], legendgroup: k, hovertemplate: `${SH[k]}<br>mean %{y:.3f}%<extra></extra>` }));
    tr.push(on2(b, { x: idx.map(i => i + off), y: c.sd, type: 'bar', width: wd, marker: { color: COL[k] }, name: SH[k], showlegend: false, legendgroup: k, hovertemplate: `${SH[k]}<br>sd %{y:.2f}%<extra></extra>` }));
    tr.push(on2(b, { x: [0.6, q + 0.4], y: [c.sdAll, c.sdAll], mode: 'lines', line: { color: COL[k], width: 1.1, dash: 'dash' }, name: `${SH[k]}: unconditional sd`, hoverinfo: 'skip' }));
    items.push(T.cdItem(SH[k], nf(Math.min(...c.mean), 3), nf(Math.max(...c.mean), 3), nf(c.sd[0], 2), q, nf(c.sd[q - 1], 2), nf(c.sd[q - 1] / c.sd[0], 2),
      nf(c.total, 3), nf(c.within, 3), sig(c.between, 2), sig(100 * c.between / c.total, 2)));
  });
  const tt = idx.map(i => i === 1 ? 'Q1<br>calm' : i === q ? `Q${q}<br>turbulent` : `Q${i}`);
  const xt = q === 5 ? 'quintile of |r_{t-1}|' : `group of |r_{t-1}| (${q} groups)`;
  setAx(Pn, 0, XT(xt, { tickvals: idx, ticktext: tt, tickangle: 0, tickfont: { size: q > 6 ? 9 : 11, color: C.navy } }), XT('E[r_t | group] (%)'));
  setAx(Pn, 1, XT(xt, { tickvals: idx, ticktext: tt, tickangle: 0, tickfont: { size: q > 6 ? 9 : 11, color: C.navy } }), XT('sd(r_t | group) (%)'));
  plot('cd-chart', tr, layout(Object.assign(Pn.L, { shapes, legend: LEG(Pn.stack, Pn.stack ? -0.1 : -0.24) })));
  out('cd-out', [T.cd(q), ...items]);
}

// ================================================================= 6. the LLN for the mean return
function llnBlock() {
  const y = +$('ln-y').value; label('ln-y').textContent = y;
  const S = ser('sp500'), i0 = S.dates.findIndex(d => d >= `${y}-01-01`), r = S.r.slice(i0), n = r.length, m = mean(r), s = sdev(r), nw = single('ln-chart', 380);
  const G = D.lln.grid.filter(g => g <= n); if (G[G.length - 1] !== n) G.push(n);
  const cm = []; let acc = 0, gi = 0; for (let i = 0; i < n; i++) { acc += r[i]; if (i + 1 === G[gi]) { cm.push(acc / (i + 1)); gi++; } }
  const ng = D.lln.grid.filter(g => g <= n).length;
  const tr = D.lln.e.map((e, i) => ({ x: D.lln.grid.slice(0, ng), y: e.slice(0, ng).map(v => m + s * v), mode: 'lines', line: { color: PALETTE[2 + i % 5], width: 0.9 }, opacity: 0.8,
    name: 'simulated i.i.d. Normal, same mean and sd', showlegend: i === 0, legendgroup: 's', hovertemplate: 'n = %{x}<br>mean %{y:.3f}%<extra></extra>' }));
  tr.push({ x: G, y: cm, mode: 'lines', line: { color: DBLUE, width: 1.8 }, name: 'S&P 500: running mean', hovertemplate: 'n = %{x}<br>running mean %{y:.3f}%<extra></extra>' });
  tr.push({ x: G, y: G.map(k => m + 1.96 * s / Math.sqrt(k)), mode: 'lines', line: { color: RED, width: 1.1, dash: 'dash' }, name: 'mean +/- 1.96 sd / sqrt(n)', legendgroup: 'b', hoverinfo: 'skip' });
  tr.push({ x: G, y: G.map(k => m - 1.96 * s / Math.sqrt(k)), mode: 'lines', line: { color: RED, width: 1.1, dash: 'dash' }, showlegend: false, legendgroup: 'b', hoverinfo: 'skip' });
  plot('ln-chart', tr, layout({ shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: DARK, width: 0.5 } }],
    xaxis: XT('number of days n (log scale)', { type: 'log', dtick: 1, exponentformat: 'power' }), yaxis: XT('running mean of r_t (%)', { range: [-0.6, 0.6] }), legend: LEG(nw, -0.2) }));
  const se = s / Math.sqrt(n), t = m / se, A = 252;
  out('ln-out', [T.ln(y, fint(n), nf(n / A, 1), nf(m, 4), nf(se, 4), nf(t, 2), nf(m * A, 1), nf((m - 1.96 * se) * A, 1), nf((m + 1.96 * se) * A, 1), m > 0 ? nf((3 * s / m) ** 2 / A, 0) : null)]);
}

// ================================================================= 7. RANDU
function randuBlock() {
  const sd = +$('ru-s').value; label('ru-s').textContent = sd;
  const n = 30000, M = 2147483648, x = new Float64Array(n); x[0] = sd;
  for (let k = 1; k < n; k++) x[k] = (65539 * x[k - 1]) % M;
  const u = Array.from(x, v => v / M), cnt = new Map();
  for (let k = 0; k < n - 2; k++) { const c = (9 * x[k] - 6 * x[k + 1] + x[k + 2]) / M; cnt.set(c, (cnt.get(c) || 0) + 1); }
  const vals = [...cnt.keys()].sort((p, q) => p - q), P = D.pcg[sd], w = (P.hi - P.lo) / P.c.length, tot = P.c.reduce((s, v) => s + v, 0);
  const Pn = panels('ru-chart', 3, { h: 360, hs: 260, extra: 130, gap: 0.08 }), [a, b, c] = Pn.ax;
  const tr = [
    on2(a, { x: u.slice(0, 4999), y: u.slice(1, 5000), mode: 'markers', marker: { size: 2, color: DBLUE, opacity: 0.6 }, name: 'RANDU pairs (u_k, u_{k+1})', hoverinfo: 'skip' }),
    on2(b, { x: vals, y: vals.map(v => cnt.get(v) / (n - 2)), type: 'bar', width: 0.5, marker: { color: RED }, name: 'RANDU: 9u_k - 6u_{k+1} + u_{k+2}', hovertemplate: 'value %{x}<br>share %{y:.3f}<extra></extra>' }),
    on2(c, { x: P.c.map((_, i) => P.lo + (i + 0.5) * w), y: P.c.map(v => v / (tot * w)), type: 'bar', width: w, marker: { color: GREEN, opacity: 0.75 }, name: 'PCG64: the same combination', hovertemplate: 'value %{x:.2f}<br>density %{y:.3f}<extra></extra>' })];
  setAx(Pn, 0, XT('u_k', { range: [0, 1] }), XT('u_{k+1}', { range: [0, 1] }));
  setAx(Pn, 1, XT('value of the combination', { range: [-7, 11] }), XT('share of triples'));
  setAx(Pn, 2, XT('value of the combination'), XT('density'));
  plot('ru-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(c, `seed ${sd}`)], legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.2) })));
  out('ru-out', [T.ru(sd, vals.length, nf(vals[0], 0), nf(vals[vals.length - 1], 0), nf(pearson(u.slice(0, -1), u.slice(1)), 4), fint(P.nd), fint(n - 2))]);
}

// ================================================================= 8. inverse transform
function invBlock() {
  const lam = +$('it-l').value, xq = +$('it-x').value, I = D.inv; label('it-l').textContent = nf(lam, 1); label('it-x').textContent = nf(xq, 1);
  const Pn = panels('it-chart', 2, { h: 390, hs: 290, extra: 130 }), [a, b] = Pn.ax;
  const nb = 120, bw = 0.025, cts = [];
  for (let j = 0; j < nb; j++) { const i0 = Math.round(5 * lam * j), i1 = Math.round(5 * lam * (j + 1)); let s = 0; for (let i = i0; i < i1; i++) s += I.fine[i]; cts.push(s); }
  const tot = cts.reduce((s, v) => s + v, 0), xg = linspace(0, 3, 300), e = I.edges, mids = e.slice(0, -1).map((v, i) => 0.5 * (v + e[i + 1]));
  const stepY = h => h.map(v => v > 0 ? v : null);
  const tr = [
    on2(a, { x: cts.map((_, j) => (j + 0.5) * bw), y: cts.map(v => v / (tot * bw)), type: 'bar', width: bw, marker: { color: AMBER, opacity: 0.6 }, name: `-ln(1 - U)/${lam.toFixed(1)}, histogram`, hovertemplate: 'x %{x:.3f}<br>density %{y:.3f}<extra></extra>' }),
    on2(a, { x: xg, y: xg.map(v => lam * Math.exp(-lam * v)), mode: 'lines', line: { color: DBLUE, width: 1.8 }, name: `exponential density ${lam.toFixed(1)} exp(-${lam.toFixed(1)}x)`, hoverinfo: 'skip' }),
    on2(b, { x: mids, y: stepY(I.hN), mode: 'lines', line: { shape: 'hvh', color: RED, width: 1.5 }, name: 'Normal quantile of U', hovertemplate: '%{x:.1f}%<br>density %{y:.2e}<extra></extra>' }),
    on2(b, { x: mids, y: stepY(I.hH), mode: 'lines', line: { shape: 'hvh', color: DBLUE, width: 1.5 }, name: 'empirical quantile of U (S&P 500)', hovertemplate: '%{x:.1f}%<br>density %{y:.2e}<extra></extra>' })];
  setAx(Pn, 0, XT('x', { range: [0, 3] }), XT('density'));
  setAx(Pn, 1, XT('simulated daily return (%)', { range: [-12, 12] }), XT('density (log scale)', { type: 'log', exponentformat: 'power', dtick: 1 }));
  plot('it-chart', tr, layout(Object.assign(Pn.L, { bargap: 0, shapes: [vln(-xq, GREEN, b, 'dash')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  const j = I.xs.indexOf(xq), r = ser('sp500').r, m = N.cdf.mean, s = N.cdf.sd, pex = Phi((-xq - m) / s), pd = r.filter(v => v < -xq).length / r.length;
  out('it-out', [T.it(nf(lam, 1), nf(I.meanV / lam, 3), nf(1 / lam, 3)),
    T.it2(nf(xq, 1), fint(I.cN[j]), spct(I.cN[j] / I.n, 2), spct(pex, 2), nf(pex * I.n, 1), fint(I.cH[j]), spct(I.cH[j] / I.n, 2), spct(pd, 2)),
    T.it3(nf(I.minN, 1), nf(I.minH, 1))]);
}

// ================================================================= 9. Monte Carlo
function mcBlock() {
  const L = +$('mc-L').value; label('mc-L').textContent = L;
  const M = D.mc, h = M.h, m = N.mc.m, s = N.mc.s, z = (-L - h * m) / (s * Math.sqrt(h)), p = Phi(z), est = M.est[L], Ns = M.Ns, nw = single('mc-chart', 370);
  const band = Ns.map(k => 1.96 * Math.sqrt(p * (1 - p) / k));
  const tr = [
    { x: Ns, y: Ns.map((_, i) => 100 * (p - band[i])), mode: 'lines', line: { width: 0, color: TEAL }, showlegend: false, hoverinfo: 'skip' },
    { x: Ns, y: Ns.map((_, i) => 100 * (p + band[i])), mode: 'lines', line: { width: 0, color: TEAL }, fill: 'tonexty', fillcolor: 'rgba(23,162,184,0.2)', name: '95% band p +/- 1.96 sqrt(p(1 - p)/N)', hoverinfo: 'skip' },
    { x: Ns, y: est.map(v => 100 * v), mode: 'lines', line: { color: DBLUE, width: 1.5 }, name: 'Monte Carlo estimate', hovertemplate: 'N = %{x}<br>estimate %{y:.3f}%<extra></extra>' },
    { x: [Ns[0], Ns[Ns.length - 1]], y: [100 * p, 100 * p], mode: 'lines', line: { color: RED, width: 1.2, dash: 'dash' }, name: `exact value ${(100 * p).toFixed(3)}%`, hoverinfo: 'skip' }];
  plot('mc-chart', tr, layout({ xaxis: XT('number of simulations N (log scale)', { type: 'log', dtick: 1, exponentformat: 'power' }),
    yaxis: XT(`P(21-day loss > ${L}%) (%)`, { range: [0, 300 * p] }), legend: LEG(nw, -0.2) }));
  const at = k => M.at[L][k] / k, se = k => Math.sqrt(p * (1 - p) / k);
  out('mc-out', [T.mc(L, nf(z, 3), spct(p, 4)),
    T.mc2(spct(at(1000), 4), spct(se(1000), 2), spct(at(100000), 4), spct(se(100000), 2), spct(at(1000000), 4), spct(se(1000000), 2)),
    T.mc3(fint(Math.ceil((1 - p) / p / 0.01 * 1.96 ** 2)))]);
}

// ================================================================= 10. binomial paths and the lognormal limit
const BN_N = [2, 5, 10, 21, 63, 126, 252];
function binBlock() {
  const n = BN_N[+$('bn-n').value]; label('bn-n').textContent = n;
  const B = D.bin, mu = B.mu, sg = B.sigma, S0 = 100, dt = 1 / n, u = Math.exp(sg * Math.sqrt(dt)), d = 1 / u, p = (Math.exp(mu * dt) - d) / (u - d);
  const K = Array.from({ length: n + 1 }, (_, k) => k), S = K.map(k => S0 * u ** k * d ** (n - k)), pr = K.map(k => binomPmf(k, n, p));
  const Pn = panels('bn-chart', 2, { h: 390, hs: 290, extra: 120, gap: 0.1 }), [a, b] = Pn.ax, tr = [];
  B.paths[n].forEach((P, i) => tr.push(on2(a, { x: P.map((_, j) => j * 252 / n), y: P, mode: 'lines', line: { color: PALETTE[i % PALETTE.length], width: 0.9 }, name: 'binomial paths', showlegend: i === 0, legendgroup: 'p', hovertemplate: 'day %{x:.0f}<br>S %{y:.2f}<extra></extra>' })));
  const wl = Math.log(u / d), keep = K.filter(k => pr[k] > 1e-6);
  tr.push(on2(b, { x: keep.map(k => S[k]), y: keep.map(k => pr[k] / (S[k] * wl)), type: 'bar', width: keep.map(k => S[k] * wl * 0.9), marker: { color: AMBER, opacity: 0.7 }, name: 'binomial distribution of S_T',
    customdata: keep.map(k => pr[k]), hovertemplate: 'S_T = %{x:.2f}<br>probability %{customdata:.4f}<extra></extra>' }));
  const ml = Math.log(S0) + (mu - sg * sg / 2), sl = sg, xs = linspace(Math.min(...keep.map(k => S[k])), Math.max(...keep.map(k => S[k])), 400);
  tr.push(on2(b, { x: xs, y: xs.map(v => normPdf(Math.log(v), ml, sl) / v), mode: 'lines', line: { color: DBLUE, width: 1.8 }, name: 'lognormal limit', hoverinfo: 'skip' }));
  setAx(Pn, 0, XT('trading day'), XT('S_t'));
  setAx(Pn, 1, XT('S_T'), XT('density'));
  plot('bn-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  let below = 0, tie = 0; K.forEach(k => { if (2 * k < n) below += pr[k]; else if (2 * k === n) tie += pr[k]; });
  let cum = 0, q05 = S[n]; for (const k of K) { cum += pr[k]; if (cum >= 0.05) { q05 = S[k]; break; } }
  const ES = K.reduce((s, k) => s + S[k] * pr[k], 0);
  out('bn-out', [T.bn(n, nf(u, 4), nf(d, 4), nf(p, 4), nf(ES, 2), nf(S0 * Math.exp(mu), 2)),
    T.bn2(pc(below + tie), pc(below), n % 2 ? null : pc(tie), pc(Phi((Math.log(S0) - ml) / sl))),
    T.bn3(nf(q05, 1), nf(Math.exp(ml + sl * normPpf(0.05)), 1))]);
}

// ================================================================= 11. white noise, AR(1) and random walk
function ar1(phi, eps) { const x = []; let p = 0; for (const e of eps) { p = phi * p + e; x.push(p); } return x; }
function procBlock() {
  const phi = +$('pr-phi').value; label('pr-phi').textContent = nf(phi, 2);
  const e = D.eps, t = e.map((_, i) => i), G = gridp('pr-chart', 2, 2, { h: 470, hs: 210, extra: 90, gy: 0.2 });
  const S = [['white noise eps_t', e, PURPLE], ['AR(1), phi = 0.5', ar1(0.5, e), GREEN], [`AR(1), phi = ${phi.toFixed(2)}`, ar1(phi, e), AMBER], ['random walk (phi = 1)', ar1(1, e), RED]];
  const tr = S.map(([nm, x, c], i) => on2(G.ax[i], { x: t, y: x, mode: 'lines', line: { color: c, width: 0.9 }, name: nm, showlegend: false, hovertemplate: 't = %{x}<br>%{y:.2f}<extra></extra>' }));
  S.forEach((_, i) => { G.L[G.ax[i].xa] = Object.assign(G.L[G.ax[i].xa], XT(i >= 2 || G.nw ? 't' : '', { matches: i ? 'x' : undefined })); G.L[G.ax[i].ya] = Object.assign(G.L[G.ax[i].ya], XT('')); });
  plot('pr-chart', tr, layout(Object.assign(G.L, { margin: { l: 46, r: 12, t: 30, b: 40 }, shapes: G.ax.map(a => hln(0, DARK, a, 'solid', 0.5)), annotations: S.map(([nm], i) => ptitle(G.ax[i], nm)) })));
  const x = S[2][1], hl = phi > 0 && phi < 1 ? Math.log(0.5) / Math.log(phi) : null;
  const sim = N.proc[phi === 0.5 ? '0.5' : phi === 0.95 ? '0.95' : ''];
  out('pr-out', [T.pr(nf(phi, 2), phi < 1 ? nf(1 / (1 - phi * phi), 2) : null, nf(phi ** 5, 3), hl === null ? (phi >= 1 ? '∞' : '—') : nf(hl, 1), nf(sdev(x) ** 2, 2)),
    sim ? T.prSim(nf(phi, 2), nf(sim.var, 2), nf(sim.var_th, 2), nf(sim.rho5, 3)) : T.prRw(nf(N.proc.rw_var['10'], 1), nf(N.proc.rw_var['50'], 1), nf(N.proc.rw_var['100'], 1))]);
}

// ================================================================= 12. which one is real?
function spotBlock() {
  const k = +$('sr-k').value; label('sr-k').textContent = nf(k, 1);
  const S = ser('sp500'), cols = [D.spot.sims[0], D.spot.sims[1], S.r, D.spot.sims[2]], G = gridp('sr-chart', 2, 4, { h: 470, nr: 4, nc: 2, hs: 190, extra: 60, gx: 0.06, gxn: 0.14, gy: 0.14, gyn: 0.07 });
  const tr = [], shapes = [], ann = [], items = [];
  cols.forEach((x, c) => {
    const ia = G.nw ? 2 * c : c, ib = G.nw ? 2 * c + 1 : 4 + c, A = G.ax[ia], B = G.ax[ib];
    const lp = [D.spot.lp0]; let acc = D.spot.lp0; for (const v of x) { acc += v / 100; lp.push(acc); }
    const m = mean(x), s = sdev(x), far = x.filter(v => Math.abs(v - m) > k * s).length;
    tr.push(on2(A, { x: lp.map((_, i) => i), y: lp.map(Math.exp), mode: 'lines', line: { color: DBLUE, width: 0.9 }, showlegend: false, hovertemplate: 'day %{x}<br>price %{y:.0f}<extra></extra>' }));
    tr.push(on2(B, { x: x.map((_, i) => i + 1), y: x, mode: 'lines', line: { color: RED, width: 0.5 }, showlegend: false, hovertemplate: 'day %{x}<br>return %{y:.2f}%<extra></extra>' }));
    shapes.push(hln(m + k * s, DARK, B, 'dash', 1), hln(m - k * s, DARK, B, 'dash', 1));
    G.L[A.xa] = Object.assign(G.L[A.xa], XT('', { showticklabels: false }));
    G.L[A.ya] = Object.assign(G.L[A.ya], XT(c === 0 && !G.nw ? 'price (log scale)' : '', Object.assign({ type: 'log', matches: c ? G.ax[0].y : undefined, showticklabels: c === 0 || G.nw }, logTicks(true))));
    G.L[B.xa] = Object.assign(G.L[B.xa], XT(G.nw && c < 3 ? '' : 'trading day'));
    G.L[B.ya] = Object.assign(G.L[B.ya], XT(c === 0 && !G.nw ? 'daily return (%)' : '', { range: [-13.5, 13.5], showticklabels: c === 0 || G.nw }));
    ann.push(ptitle(A, G.nw ? `${'ABCD'[c]}: price (log scale)` : 'ABCD'[c]));
    if (G.nw) ann.push(ptitle(B, `${'ABCD'[c]}: daily return (%)`));
    items.push(T.srItem('ABCD'[c], far, nf(exKurt(x), 2), nf(acf1(x.map(v => v * v)), 3), nf(Math.max(...x.map(Math.abs)), 1)));
  });
  plot('sr-chart', tr, layout(Object.assign(G.L, { margin: { l: 52, r: 10, t: 26, b: 40 }, shapes, annotations: ann })));
  out('sr-out', [T.sr(nf(k, 1)), ...items]);
}

// ================================================================= 13. Wiener process
function wienBlock() {
  const k = +$('wn-k').value, W = D.wien; label('wn-k').textContent = nf(k, 1);
  const Pn = panels('wn-chart', 2, { h: 400, hs: 300, extra: 140 }), [a, b] = Pn.ax, tr = [];
  [[10, AMBER, 1.1], [100, GREEN, 1.1], [10000, DBLUE, 0.8]].forEach(([n, c, lw]) => {
    const p = W.rw[n];
    tr.push(on2(a, { x: p.map((_, i) => i / n), y: p.map(v => v / Math.sqrt(n)), mode: 'lines', line: { shape: 'hv', color: c, width: lw }, name: `n = ${n} steps`, hovertemplate: 't %{x:.3f}<br>W_n(t) %{y:.3f}<extra></extra>' }));
  });
  W.W.forEach((p, i) => tr.push(on2(b, { x: p.map((_, j) => j * 0.002), y: p, mode: 'lines', line: { color: PALETTE[i % PALETTE.length], width: 0.7 }, opacity: 0.85, name: 'Wiener paths', showlegend: i === 0, legendgroup: 'w', hoverinfo: 'skip' })));
  const t = linspace(0, 1, 201);
  (Math.abs(k - 1) < 1e-9 ? [[1, 'dash']] : [[1, 'dash'], [k, 'dot']]).forEach(([kk, ds], j) => {
    tr.push(on2(b, { x: t, y: t.map(v => kk * Math.sqrt(v)), mode: 'lines', line: { color: DARK, width: 1.1, dash: ds }, name: `+/- ${kk.toFixed(kk % 1 ? 1 : 0)} sqrt(t)`, legendgroup: 'b' + j, hoverinfo: 'skip' }));
    tr.push(on2(b, { x: t, y: t.map(v => -kk * Math.sqrt(v)), mode: 'lines', line: { color: DARK, width: 1.1, dash: ds }, showlegend: false, legendgroup: 'b' + j, hoverinfo: 'skip' }));
  });
  setAx(Pn, 0, XT('t'), XT('W_n(t)'));
  setAx(Pn, 1, XT('t'), XT('W(t)'));
  plot('wn-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18) })));
  const j = W.ks.findIndex(v => Math.abs(v - k) < 1e-9), in20 = W.endW.filter(v => Math.abs(v) < k).length;
  out('wn-out', [T.wn(nf(k, 1), pc(W.share[j], 2), pc(2 * Phi(k) - 1, 2), in20), T.wn2(nf(N.wiener.var_W1, 3), nf(N.wiener.qv_mean, 3), nf(N.wiener.qv_sd, 3))]);
}

// ================================================================= 14. GBM calibrated to the S&P 500
function gbmBlock() {
  const v = +$('gp-s').value, G = D.gbm, sg = Math.abs(v - 100 * G.sigma) < 0.006 ? G.sigma : v / 100, mu = G.mu, T10 = G.T; label('gp-s').textContent = nf(100 * sg, 2) + '%';
  const nw = single('gp-chart', 380), t = G.idx.map(i => i * T10 / G.steps), dr = mu - sg * sg / 2;
  const tr = G.W.map((w, i) => ({ x: t, y: w.map((x, j) => 100 * Math.exp(dr * t[j] + sg * x)), mode: 'lines', line: { color: PALETTE[i % PALETTE.length], width: 0.6 }, opacity: 0.7, name: 'GBM paths', showlegend: i === 0, legendgroup: 'g', hoverinfo: 'skip' }));
  const tt = linspace(0, T10, 201);
  tr.push({ x: tt, y: tt.map(x => 100 * Math.exp(mu * x)), mode: 'lines', line: { color: DARK, width: 2.2 }, name: 'mean S_0 exp(mu t)', hovertemplate: 'year %{x:.1f}<br>mean %{y:.1f}<extra></extra>' });
  tr.push({ x: tt, y: tt.map(x => 100 * Math.exp(dr * x)), mode: 'lines', line: { color: CRIMSON, width: 2.2, dash: 'dash' }, name: 'median S_0 exp((mu - sigma^2/2) t)', hovertemplate: 'year %{x:.1f}<br>median %{y:.1f}<extra></extra>' });
  plot('gp-chart', tr, layout({ xaxis: XT('years'), yaxis: XT('S_t (log scale)', Object.assign({ type: 'log' }, logTicks(false))), legend: LEG(nw, -0.2) }));
  const slide = sg === G.sigma, below = G.W.filter(w => 100 * Math.exp(dr * T10 + sg * w[w.length - 1]) < 100 * Math.exp(mu * T10)).length;
  out('gp-out', [T.gp(nf(100 * sg, 2), nf(100 * mu, 2), nf(100 * dr, 2), slide),
    T.gp2(nf(100 * Math.exp(mu * T10), 1), nf(100 * Math.exp(dr * T10), 1), pc(Phi(sg * Math.sqrt(T10) / 2)), slide ? pc(G.share_below) : null, below, pc(Phi(-dr * Math.sqrt(T10) / sg)))]);
}

// ================================================================= 15. real paths in the GBM fan
const GF_IDX = { 0.5: [4, 6], 0.8: [3, 7], 0.9: [2, 8], 0.95: [1, 9], 0.99: [0, 10] };
function fanBlock() {
  const bl = $('gf-b').value, Pn = panels('gf-chart', 3, { h: 380, hs: 270, extra: 120, gap: 0.07 }), tr = [], ann = [], items = [];
  const [lo, hi] = GF_IDX[+bl];
  ['sp500', 'bet', 'btc'].forEach((k, j) => {
    const F = D.fan[k], S = ser(k), ax = Pn.ax[j], n = S.r.length, dates = [N.fan[k].first, ...S.dates];
    const lp = [F.lp0]; let acc = F.lp0; for (const v of S.r) { acc += v / 100; lp.push(acc); }
    const idx = []; for (let i = 0; i <= n; i += F.every) idx.push(i); if (idx[idx.length - 1] !== n) idx.push(n);
    const dx = idx.map(i => dates[i]), col = COL[k], rgba = a => `rgba(${parseInt(col.slice(1, 3), 16)},${parseInt(col.slice(3, 5), 16)},${parseInt(col.slice(5, 7), 16)},${a})`, Q = F.q;
    const band = (i0, i1, a, nm, sl) => { tr.push(on2(ax, { x: dx, y: Q[i0].map(Math.exp), mode: 'lines', line: { width: 0, color: col }, showlegend: false, hoverinfo: 'skip' }));
      tr.push(on2(ax, { x: dx, y: Q[i1].map(Math.exp), mode: 'lines', line: { width: 0, color: col }, fill: 'tonexty', fillcolor: rgba(a), name: nm, showlegend: sl, legendgroup: nm, hoverinfo: 'skip' })); };
    const lv = q => String(+(100 * D.fan.lev[q]).toFixed(1));
    band(lo, hi, 0.22, `GBM ${lv(lo)}%-${lv(hi)}% (series colour)`, j === 0);
    band(4, 6, 0.3, 'GBM 25%-75%', j === 0);
    tr.push(on2(ax, { x: dx, y: Q[5].map(Math.exp), mode: 'lines', line: { color: col, width: 1, dash: 'dash' }, name: 'GBM median', showlegend: j === 0, legendgroup: 'm', hoverinfo: 'skip' }));
    tr.push(on2(ax, { x: dates, y: lp.map(Math.exp), mode: 'lines', line: { color: DARK, width: 1 }, name: 'real path', showlegend: j === 0, legendgroup: 'r', hovertemplate: '%{x}<br>%{y:.1f}<extra></extra>' }));
    setAx(Pn, j, XT('', { tickangle: -30 }), XT(j === 0 || Pn.stack ? 'price (log scale)' : '', Object.assign({ type: 'log' }, logTicks(true))));
    ann.push(ptitle(ax, SH[k]));
    items.push(T.gfItem(SH[k], pc(F.outside[bl], 1), nf(100 * F.sigma, 1)));
  });
  plot('gf-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.22) })));
  out('gf-out', [T.gf(pc(+bl, 0), pc(1 - +bl, 0)), ...items]);
}

// ================================================================= 16. what GBM misses
function chkBlock() {
  const k = +$('gc-k').value, Ck = D.chk, j = Ck.ks.findIndex(v => Math.abs(v - k) < 1e-9); label('gc-k').textContent = nf(k, 1);
  const G = gridp('gc-chart', 2, 2, { h: 560, hs: 230, extra: 110, gy: 0.22, gx: 0.12 }), names = ['sp500', 'bet', 'btc'], idx = [0, 1, 2], tr = [], items = [];
  const panelsDef = [['kurt', 'excess kurtosis', 1], ['acf2', 'Corr(r_t^2, r_{t-1}^2)', 1], ['dd', 'maximum drawdown', 100], ['f', `days beyond ${k.toFixed(1)} sd`, 1]];
  panelsDef.forEach(([key, ttl, sc], p) => {
    const ax = G.ax[p];
    names.forEach((nm, i) => {
      const o = Ck[nm], q05 = key === 'f' ? o.f05[j] : o[key].q05, q50 = key === 'f' ? o.f50[j] : o[key].q50, q95 = key === 'f' ? o.f95[j] : o[key].q95, rv = key === 'f' ? o.fr[j] : o.real[key];
      tr.push(on2(ax, { x: [i, i], y: [sc * q05, sc * q95], mode: 'lines', line: { color: TEAL, width: 6 }, opacity: 0.6, name: 'GBM simulations, 5%-95%', showlegend: p === 0 && i === 0, legendgroup: 'r', hovertemplate: `${SH[nm]}<br>GBM 5%-95%<extra></extra>` }));
      tr.push(on2(ax, { x: [i], y: [sc * q50], mode: 'markers', marker: { symbol: 'line-ew-open', size: 16, color: TEAL, line: { width: 2, color: TEAL } }, name: 'GBM median', showlegend: p === 0 && i === 0, legendgroup: 'm', hovertemplate: `${SH[nm]}<br>GBM median %{y:.3f}<extra></extra>` }));
      tr.push(on2(ax, { x: [i], y: [sc * rv], mode: 'markers', marker: { size: 9, color: COL[nm] }, name: 'real series', showlegend: p === 0 && i === 0, legendgroup: 'x', hovertemplate: `${SH[nm]}<br>real %{y:.3f}<extra></extra>` }));
    });
    G.L[ax.xa] = Object.assign(G.L[ax.xa], XT('', { tickvals: idx, ticktext: names.map(n => SH[n]), range: [-0.5, 2.5] }));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], XT('', key === 'dd' ? { ticksuffix: '%' } : {}));
  });
  plot('gc-chart', tr, layout(Object.assign(G.L, { annotations: panelsDef.map(([, t], p) => ptitle(G.ax[p], t)), legend: LEG(G.nw, G.nw ? -0.05 : -0.1) })));
  names.forEach(nm => { const o = Ck[nm]; items.push(T.gcItem(SH[nm], fint(o.n), o.fr[j], nf(o.f50[j], 0), nf(o.f95[j], 0), o.fmax[j])); });
  out('gc-out', [T.gc(nf(k, 1)), ...items]);
}

// ================================================================= wiring
function init() {
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['rv-a'], rvBlock); PL('rv-play', 'rv-a', ['sp500', 'dax', 'bet'], 1500);
  on(['cq-a', 'cq-p'], cdfBlock); PL('cq-play', 'cq-p', [5, 4, 3, 2, 1, 0], 1000);
  on(['jt-y'], jointBlock); PL('jt-play', 'jt-y', range(1, 27, 1), 700);
  on(['ud-h'], udBlock); PL('ud-play', 'ud-h', range(1, 20, 1), 500);
  on(['cd-q'], condBlock); PL('cd-play', 'cd-q', range(2, 10, 1), 800);
  on(['ln-y'], llnBlock); PL('ln-play', 'ln-y', range(2000, 2020, 1), 600);
  on(['ru-s'], randuBlock); PL('ru-play', 'ru-s', range(1, 99, 2), 300);
  on(['it-l', 'it-x'], invBlock); PL('it-play', 'it-x', range(1, 12, 0.5), 400);
  on(['mc-L'], mcBlock); PL('mc-play', 'mc-L', range(5, 20, 1), 600);
  on(['bn-n'], binBlock); PL('bn-play', 'bn-n', range(0, 6, 1), 1000);
  on(['pr-phi'], procBlock); PL('pr-play', 'pr-phi', range(0, 1, 0.05), 400);
  on(['sr-k'], spotBlock); PL('sr-play', 'sr-k', range(3, 8, 0.5), 700);
  on(['wn-k'], wienBlock); PL('wn-play', 'wn-k', range(0.5, 3, 0.1), 300);
  on(['gp-s'], gbmBlock); PL('gp-play', 'gp-s', [5, 10, 15, 19.26, 25, 30, 35, 40, 50, 60], 800);
  on(['gf-b'], fanBlock); PL('gf-play', 'gf-b', ['0.5', '0.8', '0.9', '0.95', '0.99'], 1500);
  on(['gc-k'], chkBlock); PL('gc-play', 'gc-k', range(3, 8, 0.5), 700);
  const PFX = { 'random-variables': 'rv', 'cdf-quantile': 'cq', joint: 'jt', 'uncorrelated-dependent': 'ud', conditional: 'cd', lln: 'ln', randu: 'ru',
    'inverse-transform': 'it', 'monte-carlo': 'mc', binomial: 'bn', processes: 'pr', 'spot-the-real': 'sr', wiener: 'wn', 'gbm-paths': 'gp', 'gbm-fan': 'gf', 'gbm-check': 'gc' };
  const VEIL = {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [rvBlock, cdfBlock, jointBlock, udBlock, condBlock, llnBlock, randuBlock, invBlock, mcBlock, binBlock, procBlock, spotBlock, wienBlock, gbmBlock, fanBlock, chkBlock];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH4 = { ser, quantile, pearson, lagCorr, condMoments, binomPmf, Phi, normPpf, exKurt, acf1, ar1, init,
  blocks: { rvBlock, cdfBlock, jointBlock, udBlock, condBlock, llnBlock, randuBlock, invBlock, mcBlock, binBlock, procBlock, spotBlock, wienBlock, gbmBlock, fanBlock, chkBlock } };
if (window.CH4_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
