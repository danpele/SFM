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
  const lvN = `${(100 * al).toFixed(al < 0.01 ? 2 : 1).replace(/\.?0+$/, '')}%`, tl = [], tf = [];
  for (let i = 0; i < n && xs[i] <= qe; i++) if (xs[i] >= -6) { tl.push(xs[i]); tf.push((i + 1) / n); }
  if (tl.length) { tl.push(qe); tf.push(al); }
  tr.push(on2(a, { x: tl, y: tf, mode: 'lines', line: { shape: 'hv', color: GREEN, width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(40,167,69,0.25)', name: `days below the empirical ${lvN} quantile`, hoverinfo: 'skip' }));
  [a, b].forEach((ax, j) => {
    const y0 = j ? 1e-5 : 0;
    tr.push(on2(ax, { x: [ax === a ? -6 : -12, Math.max(qe, qn)], y: [al, al], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, name: `level ${lvN}`, showlegend: !j, legendgroup: 'lv', hoverinfo: 'skip' }));
    tr.push(on2(ax, { x: [qe, qe, null, qn, qn], y: [y0, al, null, y0, al], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, showlegend: false, legendgroup: 'lv', hoverinfo: 'skip' }));
    tr.push(on2(ax, { x: [qe, qn], y: [al, al], mode: 'markers', marker: { size: 9, color: [DBLUE, RED], line: { color: '#fff', width: 1 } }, name: `quantiles q_alpha: data ${qe.toFixed(2)}%, Normal ${qn.toFixed(2)}%`, showlegend: !j, legendgroup: 'q',
      customdata: ['empirical', 'Normal'], hovertemplate: '%{customdata} quantile<br>q = %{x:.2f}%<br>alpha = %{y:.4f}<extra></extra>' }));
  });
  setAx(Pn, 0, XT('daily log return x (%)'), XT('F(x) = P(r <= x)'));
  setAx(Pn, 1, XT('daily log return x (%)', { range: [-12, -1] }), XT('F(x) = P(r <= x)', { type: 'log', range: [-5, Math.log10(0.5)], exponentformat: 'power', dtick: 1 }));
  plot('cq-chart', tr, layout(Object.assign(Pn.L, {
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
  const cz = udCond(h);
  tr.push(on2(a, { x: cz.x, y: cz.y, mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 7, color: RED }, name: `S&P 500: mean of z_t^2 by decile of z_{t-${h}}`,
    hovertemplate: `mean z_{t-${h}} %{x:.2f}<br>mean z_t^2 %{y:.2f}<extra></extra>` }));
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
  setAx(Pn, 0, XT('X ~ N(0, 1); S&P 500: z_{t-h} = standardized r_{t-h}'), XT('Y = X^2; S&P 500: z_t^2'));
  setAx(Pn, 1, XT('', { tickvals: idx, ticktext: D.assets.map(k => SH[k]) }), XT(`lag-${h} correlation`, { range: [-0.15, 0.45] }));
  plot('ud-chart', tr, layout(Object.assign(Pn.L, { shapes, legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16) })));
  out('ud-out', [T.ud(h, nf(pearson(X, X.map(v => v * v)), 3)), ...items, T.udL(h, nf(cz.y[0], 2), nf(Math.min(...cz.y), 2), nf(cz.y[9], 2))]);
}
// S&P 500: deciles of z_{t-h} = (r_{t-h} - mean)/sd, and the mean of z_t^2 in each decile
function udCond(h) {
  const r = ser('sp500').r, m = mean(r), s = sdev(r), z = r.map(v => (v - m) / s), P = [];
  for (let i = h; i < z.length; i++) P.push([z[i - h], z[i] * z[i]]);
  P.sort((p, q) => p[0] - q[0]);
  const x = [], y = [], nP = P.length;
  for (let d = 0; d < 10; d++) { const g = P.slice(Math.floor(d * nP / 10), Math.floor((d + 1) * nP / 10)); x.push(mean(g.map(p => p[0]))); y.push(mean(g.map(p => p[1]))); }
  return { x, y, r: pearson(P.map(p => p[0]), P.map(p => p[1])) };
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
  const tail = h => mids.map((v, i) => (v < -xq && h[i] > 0 ? h[i] : null));
  [[I.hN, RED, 'Normal'], [I.hH, DBLUE, 'historical']].forEach(([h, c, nm]) => tr.push(on2(b, { x: mids, y: tail(h), mode: 'lines', line: { shape: 'hvh', color: c, width: 5 }, opacity: 0.45,
    name: `${nm} draws below -${xq.toFixed(1)}%`, hovertemplate: `${nm}<br>%{x:.1f}%<br>density %{y:.2e}<extra></extra>` })));
  const xl = 1 / lam;
  tr.push(on2(a, { x: [xl, xl], y: [0, lam], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `mean 1/lambda = ${xl.toFixed(3)}`, hoverinfo: 'skip' }));
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
  const e = D.eps, t = e.map((_, i) => i), G = gridp('pr-chart', 2, 2, { h: 510, hs: 210, extra: 130, gy: 0.2 });
  const S = [['white noise eps_t', e, PURPLE], ['AR(1), phi = 0.5', ar1(0.5, e), GREEN], [`AR(1), phi = ${phi.toFixed(2)}`, ar1(phi, e), AMBER], ['random walk (phi = 1)', ar1(1, e), RED]];
  const tr = [];
  [0, 1, 3].forEach((i, j) => tr.push(on2(G.ax[i], { x: t, y: S[2][1], mode: 'lines', line: { color: AMBER, width: 1.1 }, opacity: 0.55, name: `AR(1), phi = ${phi.toFixed(2)}, drawn in every panel`,
    showlegend: j === 0, legendgroup: 'cur', hovertemplate: `AR(1), phi = ${phi.toFixed(2)}<br>t = %{x}<br>%{y:.2f}<extra></extra>` })));
  S.forEach(([nm, x, c], i) => tr.push(on2(G.ax[i], { x: t, y: x, mode: 'lines', line: { color: c, width: 0.9 }, name: nm, showlegend: false, hovertemplate: 't = %{x}<br>%{y:.2f}<extra></extra>' })));
  S.forEach((_, i) => { G.L[G.ax[i].xa] = Object.assign(G.L[G.ax[i].xa], XT(i >= 2 || G.nw ? 't' : '', { matches: i ? 'x' : undefined })); G.L[G.ax[i].ya] = Object.assign(G.L[G.ax[i].ya], XT('')); });
  plot('pr-chart', tr, layout(Object.assign(G.L, { margin: { l: 46, r: 12, t: 30, b: 40 }, shapes: G.ax.map(a => hln(0, DARK, a, 'solid', 0.5)), annotations: S.map(([nm], i) => ptitle(G.ax[i], nm)),
    legend: LEG(G.nw, G.nw ? -0.04 : -0.12) })));
  const x = S[2][1], hl = phi > 0 && phi < 1 ? Math.log(0.5) / Math.log(phi) : null;
  const sim = N.proc[phi === 0.5 ? '0.5' : phi === 0.95 ? '0.95' : ''];
  out('pr-out', [T.pr(nf(phi, 2), phi < 1 ? nf(1 / (1 - phi * phi), 2) : null, nf(phi ** 5, 3), hl === null ? (phi >= 1 ? '∞' : '—') : nf(hl, 1), nf(sdev(x) ** 2, 2)),
    sim ? T.prSim(nf(phi, 2), nf(sim.var, 2), nf(sim.var_th, 2), nf(sim.rho5, 3)) : T.prRw(nf(N.proc.rw_var['10'], 1), nf(N.proc.rw_var['50'], 1), nf(N.proc.rw_var['100'], 1))]);
}

// ================================================================= 12. which one is real?
function spotBlock() {
  const k = +$('sr-k').value; label('sr-k').textContent = nf(k, 1);
  const S = ser('sp500'), cols = [D.spot.sims[0], D.spot.sims[1], S.r, D.spot.sims[2]], G = gridp('sr-chart', 2, 4, { h: 510, nr: 4, nc: 2, hs: 190, extra: 100, gx: 0.06, gxn: 0.14, gy: 0.14, gyn: 0.07 });
  const tr = [], shapes = [], ann = [], items = [];
  cols.forEach((x, c) => {
    const ia = G.nw ? 2 * c : c, ib = G.nw ? 2 * c + 1 : 4 + c, A = G.ax[ia], B = G.ax[ib];
    const lp = [D.spot.lp0]; let acc = D.spot.lp0; for (const v of x) { acc += v / 100; lp.push(acc); }
    const m = mean(x), s = sdev(x), far = x.filter(v => Math.abs(v - m) > k * s).length;
    tr.push(on2(A, { x: lp.map((_, i) => i), y: lp.map(Math.exp), mode: 'lines', line: { color: DBLUE, width: 0.9 }, showlegend: false, hovertemplate: 'day %{x}<br>price %{y:.0f}<extra></extra>' }));
    tr.push(on2(B, { x: x.map((_, i) => i + 1), y: x, mode: 'lines', line: { color: RED, width: 0.5 }, showlegend: false, hovertemplate: 'day %{x}<br>return %{y:.2f}%<extra></extra>' }));
    shapes.push(hln(m + k * s, DARK, B, 'dash', 1), hln(m - k * s, DARK, B, 'dash', 1));
    const fi = []; x.forEach((v, i) => { if (Math.abs(v - m) > k * s) fi.push(i); });
    tr.push(on2(A, { x: fi.map(i => i + 1), y: fi.map(i => Math.exp(lp[i + 1])), mode: 'markers', marker: { size: 5, color: GREEN }, name: `days with |r - mean| > ${k.toFixed(1)} sd`, showlegend: c === 0, legendgroup: 'far',
      hovertemplate: 'day %{x}<br>price %{y:.0f}<extra></extra>' }));
    tr.push(on2(B, { x: fi.map(i => i + 1), y: fi.map(i => x[i]), mode: 'markers', marker: { size: 5, color: GREEN }, name: 'far', showlegend: false, legendgroup: 'far', hovertemplate: 'day %{x}<br>return %{y:.2f}%<extra></extra>' }));
    G.L[A.xa] = Object.assign(G.L[A.xa], XT('', { showticklabels: false, range: [0, x.length + 1] }));
    G.L[A.ya] = Object.assign(G.L[A.ya], XT(c === 0 && !G.nw ? 'price (log scale)' : '', Object.assign({ type: 'log', matches: c ? G.ax[0].y : undefined, showticklabels: c === 0 || G.nw }, logTicks(true))));
    G.L[B.xa] = Object.assign(G.L[B.xa], XT(G.nw && c < 3 ? '' : 'trading day', { range: [0, x.length + 1] }));
    G.L[B.ya] = Object.assign(G.L[B.ya], XT(c === 0 && !G.nw ? 'daily return (%)' : '', { range: [-13.5, 13.5], showticklabels: c === 0 || G.nw }));
    ann.push(ptitle(A, G.nw ? `${'ABCD'[c]}: price (log scale)` : 'ABCD'[c]));
    if (G.nw) ann.push(ptitle(B, `${'ABCD'[c]}: daily return (%)`));
    items.push(T.srItem('ABCD'[c], far, nf(exKurt(x), 2), nf(acf1(x.map(v => v * v)), 3), nf(Math.max(...x.map(Math.abs)), 1)));
  });
  plot('sr-chart', tr, layout(Object.assign(G.L, { margin: { l: 52, r: 10, t: 26, b: 70 }, shapes, annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.16) })));
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
  tr.push(on2(a, { x: t.concat([null], t), y: t.map(v => k * Math.sqrt(v)).concat([null], t.map(v => -k * Math.sqrt(v))), mode: 'lines', line: { color: DARK, width: 1.1, dash: Math.abs(k - 1) < 1e-9 ? 'dash' : 'dot' },
    showlegend: false, legendgroup: 'b' + (Math.abs(k - 1) < 1e-9 ? 0 : 1), hoverinfo: 'skip' }));
  const e1 = W.endW, ins = e1.map(v => Math.abs(v) < k);
  tr.push(on2(b, { x: e1.map(() => 1), y: e1, mode: 'markers', marker: { size: 8, color: ins.map(z => (z ? GREEN : RED)), line: { color: '#fff', width: 1 } }, cliponaxis: false,
    name: `W(1) of the 20 paths: green inside +/- ${k.toFixed(1)}, red outside`, customdata: ins.map(z => (z ? 'inside' : 'outside')), hovertemplate: 'W(1) = %{y:.3f}<br>%{customdata}<extra></extra>' }));
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
  const clip = {}; names.forEach(nm => { clip[nm] = clipStats(nm, k); });
  const panelsDef = [['kurt', 'excess kurtosis', 1], ['acf2', 'Corr(r_t^2, r_{t-1}^2)', 1], ['dd', 'maximum drawdown', 100], ['f', `days beyond ${k.toFixed(1)} sd`, 1]];
  panelsDef.forEach(([key, ttl, sc], p) => {
    const ax = G.ax[p];
    names.forEach((nm, i) => {
      const o = Ck[nm], q05 = key === 'f' ? o.f05[j] : o[key].q05, q50 = key === 'f' ? o.f50[j] : o[key].q50, q95 = key === 'f' ? o.f95[j] : o[key].q95, rv = key === 'f' ? o.fr[j] : o.real[key];
      tr.push(on2(ax, { x: [i, i], y: [sc * q05, sc * q95], mode: 'lines', line: { color: TEAL, width: 6 }, opacity: 0.6, name: 'GBM simulations, 5%-95%', showlegend: p === 0 && i === 0, legendgroup: 'r', hovertemplate: `${SH[nm]}<br>GBM 5%-95%<extra></extra>` }));
      tr.push(on2(ax, { x: [i], y: [sc * q50], mode: 'markers', marker: { symbol: 'line-ew-open', size: 16, color: TEAL, line: { width: 2, color: TEAL } }, name: 'GBM median', showlegend: p === 0 && i === 0, legendgroup: 'm', hovertemplate: `${SH[nm]}<br>GBM median %{y:.3f}<extra></extra>` }));
      tr.push(on2(ax, { x: [i], y: [sc * rv], mode: 'markers', marker: { size: 9, color: COL[nm] }, name: 'real series', showlegend: p === 0 && i === 0, legendgroup: 'x', hovertemplate: `${SH[nm]}<br>real %{y:.3f}<extra></extra>` }));
    });
    if (key !== 'f') tr.push(on2(ax, { x: idx.map(i => i + 0.22), y: names.map(nm => sc * clip[nm][key]), mode: 'markers', marker: { size: 9, symbol: 'circle-open', color: names.map(nm => COL[nm]), line: { width: 2 } },
      name: `real series, returns clipped at +/- ${k.toFixed(1)} sd`, showlegend: p === 0, legendgroup: 'c', customdata: names.map(nm => SH[nm]), hovertemplate: `%{customdata}<br>clipped at ${k.toFixed(1)} sd: %{y:.3f}<extra></extra>` }));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], XT('', { tickvals: idx, ticktext: names.map(n => SH[n]), range: [-0.5, 2.5] }));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], XT('', key === 'dd' ? { ticksuffix: '%' } : {}));
  });
  plot('gc-chart', tr, layout(Object.assign(G.L, { annotations: panelsDef.map(([, t], p) => ptitle(G.ax[p], t)), legend: LEG(G.nw, G.nw ? -0.05 : -0.1) })));
  names.forEach(nm => { const o = Ck[nm]; items.push(T.gcItem(SH[nm], fint(o.n), o.fr[j], nf(o.f50[j], 0), nf(o.f95[j], 0), o.fmax[j])); });
  out('gc-out', [T.gc(nf(k, 1)), ...items, T.gcClip(nf(k, 1), names.map(nm => T.gcClipItem(SH[nm], nf(clip[nm].kurt, 2), nf(Ck[nm].real.kurt, 2), nf(clip[nm].acf2, 3), nf(Ck[nm].real.acf2, 3))).join('; '), nf(Math.abs(normClipKurt(k)) < 5e-4 ? 0 : normClipKurt(k), 3))]);
  gcExtra(k);
}
// excess kurtosis of a standard Normal variable clipped at +/- k
function normClipKurt(k) {
  const f = Math.exp(-k * k / 2) / SQ2PI, c = 2 * Phi(k) - 1, t = 2 * Phi(-k);
  const m2 = c - 2 * k * f + t * k * k, m4 = 3 * c - 2 * (k ** 3 + 3 * k) * f + t * k ** 4;
  return m4 / (m2 * m2) - 3;
}
// excess kurtosis, Corr(r_t^2, r_{t-1}^2) and maximum drawdown of the real returns clipped at mean +/- k sd
function clipStats(nm, k) {
  const r = ser(nm).r, m = mean(r), s = sdev(r), c = r.map(v => Math.max(m - k * s, Math.min(m + k * s, v)));
  let acc = 0, mx = 0, dd = 0; for (const v of c) { acc += v / 100; mx = Math.max(mx, acc); dd = Math.min(dd, Math.exp(acc - mx) - 1); }
  return { kurt: exKurt(c), acf2: acf1(c.map(v => v * v)), dd };
}

// ================================================================= tables and worked examples (blocks of the frames without a chart)
function table(id, head, rows, foot) {    // head: array of cells or an HTML string of header rows; a cell is a string or { v, c, s } (class, colspan)
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.s ? ` colspan="${o.s}"` : ''}>${o.v}</${tag}>`; };
  const H = typeof head === 'string' ? head : `<tr>${head.map(h => cell(h, 'th')).join('')}</tr>`;
  $(id).innerHTML = `<table class="xt"><thead>${H}</thead><tbody>` +
    rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>' + (foot ? `<p class="tnote">${foot}</p>` : '');
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const row = (cells, cls) => Object.assign(cells, { cls });
const sgn = (x, d) => (x < 0 ? '− ' + nf(-x, d) : '+ ' + nf(x, d));
const par = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));
const ppy = k => (k === 'btc' ? 365 : 252);
const big = x => (Math.abs(x) >= 1000 ? fint(x) : nf(x, 1));
const vl = (x, color) => ({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1.2, dash: 'dot' } });
const MOM = {};
function moments(k) {            // sample moments of the daily log returns (%): sd with n - 1, skewness and excess kurtosis without correction
  if (MOM[k]) return MOM[k];
  const r = ser(k).r, n = r.length, m = mean(r); let m2 = 0, m3 = 0;
  for (const v of r) { const d = v - m; m2 += d * d; m3 += d * d * d; }
  m2 /= n; m3 /= n;
  return (MOM[k] = { n, m, s: sdev(r), sk: m3 / m2 ** 1.5, ku: exKurt(r), A: ppy(k) });
}

// ----------------------------------------------------------------- X1. down days in a row (frame 9)
function downStats(k, x) {
  const r = ser(k).r, n = r.length; let nd = 0, nb = 0, nab = 0, nau = 0;
  for (let i = 0; i < n; i++) {
    const d = r[i] < -x; if (d) nd++;
    if (i) { if (r[i - 1] < -x) { nb++; if (d) nab++; } else if (d) nau++; }
  }
  return { n, p: nd / n, pc: nab / nb, pcu: nau / (n - 1 - nb), pb: nab / (n - 1), pi: (nd / n) ** 2 };
}
function downDays() {
  const k = $('dw-a').value, x = +$('dw-x').value; label('dw-x').textContent = nf(x, 2) + '%';
  const S = D.assets.map(a => [a, downStats(a, x)]);
  table('dw-tab', [T.h.series, 'n', 'P(D)', 'P(D | D<sub>t−1</sub>)', T.h.notD, T.h.both, 'P(D)²'],
    S.map(([a, o]) => row([SH[a], fint(o.n), pc(o.p, 2), pc(o.pc, 2), pc(o.pcu, 2), pc(o.pb, 2), pc(o.pi, 2)], a === k ? 'sel' : '')), T.dwNote(nf(x, 2)));
  const o = S.find(z => z[0] === k)[1];
  out('dw-out', [T.dw(SH[k], x === 0, nf(x, 2), pc(o.p, 2), pc(o.pc, 2), pc(o.pcu, 2), nf(o.pc / o.p, 2), pc(o.pb, 2), pc(o.pi, 2))]);
}

// ----------------------------------------------------------------- X2. annual mean and volatility (frame 19)
function annualBlock() {
  const k = $('am-a').value, h = +$('am-h').value; label('am-h').textContent = h;
  table('am-tab', [T.h.series, 'n', T.h.meanD, T.h.sdD, T.h.skew, T.h.kurt, 'A', T.h.meanA, T.h.sdA],
    D.assets.map(a => { const o = moments(a); return row([SH[a], fint(o.n), nf(o.m, 4), nf(o.s, 3), nf(o.sk, 2), nf(o.ku, 2), o.A, nf(o.A * o.m, 2), nf(o.s * Math.sqrt(o.A), 2)], a === k ? 'sel' : ''); }), T.amNote);
  const o = moments(k), E = h * o.m, sh = o.s * Math.sqrt(h), r = E / sh;
  steps('am-we', [T.amw1(SH[k], fint(o.n), nf(o.m, 4), nf(o.s, 3)), T.amw2(h, nf(o.m, 4), nf(E, 2)),
    T.amw3(nf(o.s, 3), h, nf(Math.sqrt(h), 2), nf(sh, 2)), T.amw4(h, nf(E, 2), nf(sh, 2), nf(r, 3))]);
  out('am-out', [T.am(SH[k], h, nf(E, 2), nf(sh, 2), nf(r, 3))]);
}

// ----------------------------------------------------------------- X3. Chebyshev's bound against the data (frame 20)
const ABSZ = {};
function absZ(k) {               // sorted |r - mean|/sd of a series
  if (ABSZ[k]) return ABSZ[k];
  const o = moments(k); return (ABSZ[k] = Float64Array.from(ser(k).r, v => Math.abs(v - o.m) / o.s).sort());
}
function beyond(k, kk) { const z = absZ(k); let lo = 0, hi = z.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (z[mid] > kk) hi = mid; else lo = mid + 1; } return z.length - lo; }
function chebBlock() {
  const kk = +$('cb-k').value, k = $('cb-a').value; label('cb-k').textContent = nf(kk, 1);
  const pn = 2 * Phi(-kk), cb = 1 / (kk * kk);
  table('cb-tab', [T.h.series, 'n', T.h.days(nf(kk, 1)), T.h.share, T.h.normShare, T.h.normDays, T.h.cheb, T.h.ratioN],
    D.assets.map(a => { const n = moments(a).n, c = beyond(a, kk); return row([SH[a], fint(n), fint(c), spct(c / n, 3), spct(pn, 3), sig(pn * n, 2), pc(cb, 2), c ? sig(c / n / pn, 3) : '0'], a === k ? 'sel' : ''); }), T.cbNote);
  const nw = single('cb-chart', 360), ks = range(1, 8, 0.1);
  const tr = D.assets.map(a => ({ x: ks, y: ks.map(v => { const c = beyond(a, v); return c ? 100 * c / moments(a).n : null; }), mode: 'lines', line: { color: COL[a], width: a === k ? 2.8 : 1.6, shape: 'hv' }, name: `${SH[a]}: data`, hovertemplate: `${SH[a]}<br>k = %{x:.1f}<br>%{y:.3g}%<extra></extra>` }));
  tr.push({ x: ks, y: ks.map(v => 200 * Phi(-v)), mode: 'lines', line: { color: RED, width: 1.6, dash: 'dash' }, name: 'Normal distribution 2 Phi(-k)', hovertemplate: 'k = %{x:.1f}<br>Normal %{y:.3g}%<extra></extra>' });
  tr.push({ x: ks, y: ks.map(v => 100 / (v * v)), mode: 'lines', line: { color: DARK, width: 1.6, dash: 'dot' }, name: "Chebyshev's bound 1/k^2", hovertemplate: 'k = %{x:.1f}<br>bound %{y:.3g}%<extra></extra>' });
  const mk = D.assets.map(a => { const c = beyond(a, kk); return c ? 100 * c / moments(a).n : null; });
  tr.push({ x: [kk, kk], y: [1e-4, 100], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, name: `k = ${kk.toFixed(1)}`, legendgroup: 'k', hoverinfo: 'skip' });
  tr.push({ x: D.assets.map(() => kk), y: mk, mode: 'markers', marker: { size: 10, color: D.assets.map(a => COL[a]), line: { color: '#fff', width: 1 } }, name: 'data at the chosen k', legendgroup: 'k',
    customdata: D.assets.map(a => SH[a]), hovertemplate: '%{customdata}<br>k = %{x:.1f}<br>%{y:.3g}%<extra></extra>' });
  tr.push({ x: [kk, kk], y: [200 * Phi(-kk), 100 / (kk * kk)], mode: 'markers', marker: { size: 10, symbol: 'diamond', color: [RED, DARK], line: { color: '#fff', width: 1 } }, name: 'Normal and bound at the chosen k', legendgroup: 'k',
    customdata: ['Normal', 'Chebyshev bound'], hovertemplate: '%{customdata}<br>k = %{x:.1f}<br>%{y:.3g}%<extra></extra>' });
  tr.push({ x: [kk], y: [mk[D.assets.indexOf(k)]], mode: 'markers', marker: { size: 20, color: 'rgba(0,0,0,0)', line: { color: C.navy, width: 2 } }, name: `worked example: ${SH[k]}`, legendgroup: 'k', hoverinfo: 'skip' });
  plot('cb-chart', tr, layout({ xaxis: XT('k (standard deviations)', { range: [1, 8] }),
    yaxis: XT('share of days beyond k sd (%, log scale)', { type: 'log', range: [-4, 2], exponentformat: 'power', dtick: 1 }), legend: LEG(nw, -0.2) }));
  const o = moments(k), c = beyond(k, kk), en = pn * o.n;
  steps('cb-we', [T.cbw1(nf(kk, 1), pc(cb, 2), fint(cb * o.n), fint(o.n), SH[k]), T.cbw2(nf(kk, 1), spct(pn, 3), sig(en, 2)),
    T.cbw3(fint(c), spct(c / o.n, 3), en < 1e-3 ? null : sig(c / o.n / pn, 3), pc(c / o.n / cb, 1))]);
  out('cb-out', [T.cb(SH[k], nf(kk, 1), fint(c), spct(c / o.n, 3), spct(pn, 3))]);
}

// ----------------------------------------------------------------- X4. a 50/50 portfolio: covariance and correlation (frame 25)
const pfSd = (P, a) => { const b = 1 - a; return Math.sqrt(a * a * P.s1 ** 2 + b * b * P.s2 ** 2 + 2 * a * b * P.cov); };
function pfBlock() {
  const key = $('pf-p').value, a = +$('pf-w').value, P = D.pairs[key], [u, v] = key.split('_'), A = D.assets, b = 1 - a; label('pf-w').textContent = nf(a, 2);
  const pk = (i, j) => D.pairs[i < j ? `${A[i]}_${A[j]}` : `${A[j]}_${A[i]}`];
  table('pf-mat', ['', ...A.map(x => SH[x])], A.map((x, i) => row([SH[x], ...A.map((_, j) => (i === j ? `<b>${nf(moments(x).s ** 2, 3)}</b>` : i < j ? nf(pk(i, j).cov, 3) : nf(pk(i, j).corr, 3)))])), T.pfMatNote);
  table('pf-tab', [T.h.pair, T.h.common, 'σ<sub>1</sub>', 'σ<sub>2</sub>', 'Cov', 'ρ', `σ<sub>p</sub> (a = ${nf(a, 2)})`, 'aσ<sub>1</sub> + bσ<sub>2</sub>'],
    Object.keys(D.pairs).map(q => { const Q = D.pairs[q], [x, y] = q.split('_'); return row([`${SH[x]} / ${SH[y]}`, fint(Q.n), nf(Q.s1, 3), nf(Q.s2, 3), nf(Q.cov, 3), nf(Q.corr, 3), nf(pfSd(Q, a), 3), nf(a * Q.s1 + (1 - a) * Q.s2, 3)], q === key ? 'sel' : ''); }), T.pfNote);
  const as = (P.s2 ** 2 - P.cov) / (P.s1 ** 2 + P.s2 ** 2 - 2 * P.cov), sm = pfSd(P, as), ws = linspace(0, 1, 101), nw = single('pf-chart', 340);
  plot('pf-chart', [
    { x: ws, y: ws.map(w => pfSd(P, w)), mode: 'lines', line: { color: DBLUE, width: 2 }, name: `portfolio sd, rho = ${P.corr.toFixed(2)}`, hovertemplate: 'a = %{x:.2f}<br>sd %{y:.3f}%<extra></extra>' },
    { x: ws, y: ws.map(w => w * P.s1 + (1 - w) * P.s2), mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'rho = 1: a sd_1 + b sd_2', hoverinfo: 'skip' },
    { x: ws, y: ws.map(w => Math.sqrt(w * w * P.s1 ** 2 + (1 - w) ** 2 * P.s2 ** 2)), mode: 'lines', line: { color: TEAL, width: 1.2, dash: 'dot' }, name: 'rho = 0', hoverinfo: 'skip' },
    { x: [as], y: [sm], mode: 'markers', marker: { size: 10, color: GREEN, symbol: 'diamond' }, name: `minimum variance, a* = ${as.toFixed(2)}`, hovertemplate: 'a* = %{x:.3f}<br>sd %{y:.3f}%<extra></extra>' },
    { x: [a], y: [pfSd(P, a)], mode: 'markers', marker: { size: 10, color: RED }, name: `chosen a = ${a.toFixed(2)}`, hovertemplate: 'a = %{x:.2f}<br>sd %{y:.3f}%<extra></extra>' }
  ], layout({ xaxis: XT(`weight a of ${SH[u]} (1 - a in ${SH[v]})`, { range: [0, 1] }), yaxis: XT('portfolio sd (% per day)'), legend: LEG(nw, -0.2) }));
  const vs = a * a * P.s1 ** 2 + b * b * P.s2 ** 2, ct = 2 * a * b * P.cov, sp = Math.sqrt(vs + ct), avg = a * P.s1 + b * P.s2;
  steps('pf-we', [T.pfw1(SH[u], SH[v], fint(P.n), nf(P.s1, 3), nf(P.s2, 3), nf(P.cov, 3), nf(P.corr, 3)),
    T.pfw2(nf(a, 2), nf(b, 2), nf(a * a, 4), nf(P.s1 ** 2, 3), nf(b * b, 4), nf(P.s2 ** 2, 3), nf(vs, 3), nf(2 * a * b, 3), nf(P.cov, 3), nf(ct, 3)),
    T.pfw3(nf(vs, 3), nf(ct, 3), nf(vs + ct, 3), nf(sp, 3), nf(avg, 3)), T.pfw4(nf(as, 3), nf(sm, 3))]);
  out('pf-out', [T.pf(SH[u], SH[v], nf(a, 2), nf(sp, 3), nf(avg, 3), avg > 0 ? pc(1 - sp / avg, 1) : '—')]);
}

// ----------------------------------------------------------------- X5. two regimes and the law of total variance (frame 31)
function regBlock() {
  const p1 = +$('rg-p').value, s2 = +$('rg-s').value, k = $('rg-a').value, p2 = 1 - p1; label('rg-p').textContent = nf(p1, 2); label('rg-s').textContent = nf(s2, 2) + '%';
  const R = [[T.h.calm, p1, 0.05, 0.8], [T.h.turb, p2, -0.10, s2]];
  const E = p1 * 0.05 + p2 * -0.10, W = p1 * 0.64 + p2 * s2 * s2, B = p1 * (0.05 - E) ** 2 + p2 * (-0.10 - E) ** 2, V = W + B;
  table('rg-tab', [T.h.regime, T.h.prob, T.h.meanP, T.h.sdP, T.h.varP, T.h.within, T.h.between],
    [...R.map(x => row([x[0], nf(x[1], 2), nf(x[2], 2), nf(x[3], 2), nf(x[3] ** 2, 3), nf(x[1] * x[3] ** 2, 3), nf(x[1] * (x[2] - E) ** 2, 4)])),
      row([T.h.total, '1', nf(E, 3), nf(Math.sqrt(V), 3), nf(V, 3), nf(W, 3), nf(B, 4)], 'sel')]);
  steps('rg-we', [T.rgw1(nf(p1, 2), nf(0.05, 2), nf(p2, 2), par(-0.10, 2), nf(E, 3)), T.rgw2(nf(p1, 2), nf(0.8, 1), nf(p2, 2), nf(s2, 2), nf(W, 3)),
    T.rgw3(nf(p1, 2), `${nf(0.05, 2)} ${sgn(-E, 3)}`, nf(p2, 2), `${nf(-0.10, 2)} ${sgn(-E, 3)}`, nf(B, 4)), T.rgw4(nf(W, 3), nf(B, 4), nf(V, 3), nf(Math.sqrt(V), 3), pc(B / V, 2))]);
  const c = condMoments(ser(k).r, 5);
  table('rg-dat', [T.h.group, '|r<sub>t−1</sub>| (%)', T.h.ndays, T.h.meanP, T.h.sdP, T.h.varP],
    c.mean.map((m, j) => row([`Q${j + 1}`, `${nf(c.lo[j], 2)} – ${nf(c.hi[j], 2)}`, fint(c.cnt[j]), nf(m, 3), nf(c.sd[j], 3), nf(c.sd[j] ** 2, 3)])),
    T.rgDat(SH[k], nf(c.total, 3), nf(c.within, 3), sig(c.between, 2), sig(100 * c.between / c.total, 2)));
  out('rg-out', [T.rg(nf(Math.sqrt(V), 3), pc(B / V, 2))]);
}

// ----------------------------------------------------------------- X6. the LLN and the CLT for the mean return (frame 35)
const LT_N = [10, 100, 250, 1000, 2500, 5000];
function lltBlock() {
  const y = +$('lt-y').value; label('lt-y').textContent = y;
  const S = ser('sp500'), i0 = S.dates.findIndex(d => d >= `${y}-01-01`), r = S.r.slice(i0), n = r.length, A = 252;
  const Ns = [...LT_N.filter(v => v < n), n], rows = [];
  let s1 = 0, s2 = 0, j = 0;
  for (let i = 0; i < n; i++) {
    s1 += r[i]; s2 += r[i] * r[i];
    if (i + 1 === Ns[j]) {
      const k = i + 1, m = s1 / k, sd = Math.sqrt((s2 - k * m * m) / (k - 1)), se = sd / Math.sqrt(k);
      rows.push(row([fint(k), S.dates[i0 + i], nf(m, 4), nf(sd, 3), nf(se, 4), nf(m / se, 2), `[${nf(A * (m - 1.96 * se), 1)}${LANG === 'ro' ? ';' : ','} ${nf(A * (m + 1.96 * se), 1)}]`], k === n ? 'sel' : ''));
      j++;
    }
  }
  table('lt-tab', [T.h.nDays, T.h.until, T.h.runMean, 's<sub>n</sub> (%)', 'SE (%)', 't', T.h.ciA], rows, T.ltNote(y));
  const m = mean(r), s = sdev(r), se = s / Math.sqrt(n), n2 = m > 0 ? (2 * s / m) ** 2 : null;
  steps('lt-we', [T.ltw1(y, fint(n), nf(m, 4), nf(s, 3)), T.ltw2(nf(s, 3), fint(n), nf(se, 4), nf(m, 4), nf(m / se, 2)),
    T.ltw3(nf(m, 4), nf(se, 4), nf(A * (m - 1.96 * se), 1), nf(A * (m + 1.96 * se), 1)), T.ltw4(nf(s, 3), nf(m, 4), n2 === null ? null : fint(n2), n2 === null ? null : nf(n2 / A, 1))]);
  out('lt-out', [T.lt(y, fint(n), nf(m, 4), nf(m / se, 2))]);
}

// ----------------------------------------------------------------- X7. an LCG with M = 16 (frame 39)
function lcgRun(a, c, x0, M) {
  const seen = new Map(), seq = []; let x = x0, k = 0;
  while (!seen.has(x)) { seen.set(x, k); seq.push(x); x = (a * x + c) % M; k++; }
  return { seq, tail: seen.get(x), period: k - seen.get(x), next: x };
}
function lcgBlock() {
  const a = +$('lc-a').value, c = +$('lc-c').value, x0 = +$('lc-x').value, M = 16;
  label('lc-a').textContent = a; label('lc-c').textContent = c; label('lc-x').textContent = x0;
  const R = Array.from({ length: 16 }, (_, aa) => lcgRun(aa, c, x0, M));
  table('lc-tab', ['a', T.h.firstVals, T.h.tail, T.h.period, T.h.hd],
    R.map((o, aa) => row([aa, o.seq.concat([o.next]).slice(0, 9).join(', ') + (o.seq.length > 8 ? ', …' : ''), o.tail, o.period, c % 2 === 1 && (aa - 1) % 4 === 0 ? T.h.yes : T.h.no], aa === a ? 'sel' : '')));
  const o = R[a], x1 = (a * x0 + c) % M, x2 = (a * x1 + c) % M;
  steps('lc-we', [T.lcw1(a, x0, c, a * x0 + c, Math.floor((a * x0 + c) / M), x1), T.lcw2(a, x1, c, a * x1 + c, x2),
    T.lcw3(o.seq.concat([o.next]).join(', '), o.tail, o.period), T.lcw4(o.seq.slice(0, 4).map(v => nf(v / M, 4)).join(', '))]);
  const full = R.map((z, aa) => [aa, z.period]).filter(z => z[1] === 16).map(z => z[0]);
  out('lc-out', [full.length ? T.lc(full.length, full.join(', ')) : T.lcNone]);
}

// ----------------------------------------------------------------- X8. the inverse transform method (frame 41)
const IE_X = [2, 3, 4, 5, 6, 8, 10];
function ieBlock() {
  const lam = +$('ie-l').value, u = +$('ie-u').value; label('ie-l').textContent = nf(lam, 1); label('ie-u').textContent = nf(u, 2);
  const I = D.inv, m = N.cdf.mean, s = N.cdf.sd, S = ser('sp500'), n = S.r.length;
  const dataShare = x => S.r.filter(v => v < -x).length / n;
  table('ie-tab', [T.h.thr, T.h.drawsN, T.h.expN, '√E', T.h.zN, T.h.drawsH, T.h.shareH, T.h.shareD],
    IE_X.map(x => { const j = I.xs.indexOf(x), E = I.n * Phi((-x - m) / s); return row([`−${x}%`, fint(I.cN[j]), E >= 1000 ? fint(E) : sig(E, 2), sig(Math.sqrt(E), 2), E > 1e-3 ? nf((I.cN[j] - E) / Math.sqrt(E), 1) : '—', fint(I.cH[j]), spct(I.cH[j] / I.n, 3), spct(dataShare(x), 3)], x === 5 ? 'sel' : ''); }), T.ieNote);
  const x = -Math.log(1 - u) / lam, z = normPpf(u), qh = quantile(S.s, u), qn = m + s * z;
  const Pn = panels('ie-chart', 2, { h: 380, hs: 280, extra: 130, gap: 0.1 }), [a, b] = Pn.ax, xm = Math.max(4 / lam, 1.15 * x), xg = linspace(0, xm, 300), xa = linspace(0, x, 120);
  const tr = [
    on2(a, { x: xa, y: xa.map(v => lam * Math.exp(-lam * v)), mode: 'lines', line: { width: 0, color: AMBER }, fill: 'tozeroy', fillcolor: 'rgba(255,193,7,0.45)', name: `area u = ${u.toFixed(2)} below x = -ln(1 - u)/lambda`, hoverinfo: 'skip' }),
    on2(a, { x: xg, y: xg.map(v => lam * Math.exp(-lam * v)), mode: 'lines', line: { color: DBLUE, width: 1.8 }, name: `exponential density, lambda = ${lam.toFixed(1)}`, hovertemplate: 'x %{x:.3f}<br>density %{y:.3f}<extra></extra>' }),
    on2(a, { x: [1 / lam, 1 / lam], y: [0, lam], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `mean 1/lambda = ${(1 / lam).toFixed(3)}`, hoverinfo: 'skip' }),
    on2(a, { x: [x], y: [0], mode: 'markers', marker: { size: 10, color: AMBER, line: { color: DARK, width: 1 } }, name: `exponential draw x = ${x.toFixed(3)}`, hovertemplate: 'x = %{x:.3f}<extra></extra>', cliponaxis: false })];
  const xr = linspace(-5, 5, 401), F = xr.map(v => { let lo = 0, hi = n; while (lo < hi) { const md = (lo + hi) >> 1; if (S.s[md] <= v) lo = md + 1; else hi = md; } return lo / n; });
  tr.push(on2(b, { x: xr, y: F, mode: 'lines', line: { shape: 'hv', color: DBLUE, width: 1.6 }, name: 'S&P 500 empirical CDF', hovertemplate: 'x %{x:.2f}%<br>F %{y:.3f}<extra></extra>' }));
  tr.push(on2(b, { x: xr, y: xr.map(v => Phi((v - m) / s)), mode: 'lines', line: { color: RED, width: 1.4, dash: 'dash' }, name: 'Normal CDF, same mean and sd', hoverinfo: 'skip' }));
  tr.push(on2(b, { x: [-5, Math.max(qh, qn), null, qh, qh, null, qn, qn], y: [u, u, null, u, 0, null, u, 0], mode: 'lines', line: { color: AMBER, width: 1.6 }, name: `u = ${u.toFixed(2)} -> quantile`, hoverinfo: 'skip' }));
  tr.push(on2(b, { x: [qh, qn], y: [0, 0], mode: 'markers', marker: { size: 10, color: [DBLUE, RED], line: { color: DARK, width: 1 } }, name: `draws: historical ${qh.toFixed(2)}%, Normal ${qn.toFixed(2)}%`, cliponaxis: false,
    customdata: ['historical', 'Normal'], hovertemplate: '%{customdata} draw %{x:.3f}%<extra></extra>' }));
  setAx(Pn, 0, XT('x', { range: [0, xm] }), XT('density', { rangemode: 'tozero' }));
  setAx(Pn, 1, XT('daily log return (%)', { range: [-5, 5] }), XT('F(x)', { range: [0, 1] }));
  plot('ie-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'exponential: x = F^{-1}(u)'), ptitle(b, 'S&P 500: the same u through two quantile functions')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.2) })));
  steps('ie-we', [T.iew1(nf(u, 2), nf(lam, 1), nf(-Math.log(1 - u), 3), nf(x, 3)), T.iew2(nf(lam, 1), nf(x, 3), nf(1 - Math.exp(-lam * x), 2)),
    T.iew3(nf(m, 4), nf(s, 3), nf(z, 4), nf(m + s * z, 3)), T.iew4(nf(quantile(S.s, u), 3), fint(n))]);
  const j5 = I.xs.indexOf(5), E5 = I.n * Phi((-5 - m) / s);
  out('ie-out', [T.ie(I.cN[j5], nf(E5, 1), nf(Math.sqrt(E5), 1))]);
}

// ----------------------------------------------------------------- X9. a monthly loss probability by Monte Carlo (frames 43-44)
const ME_N = [100, 1000, 10000, 100000, 1000000];
const ppc = x => (x === 0 ? '0' : x >= 1e-3 ? nf(100 * x, 3) + '%' : spct(x, 3));
function meBlock() {
  const L = +$('me-L').value, Nn = +$('me-n').value; label('me-L').textContent = L;
  const M = D.mc, h = M.h, m = N.mc.m, s = N.mc.s, mu = h * m, sd = s * Math.sqrt(h), z = (-L - mu) / sd, p = Phi(z), se = k => Math.sqrt(p * (1 - p) / k);
  table('me-tab', ['N', T.h.hits, 'p̂<sub>N</sub>', 'SE', '|p̂<sub>N</sub> − p|/SE', T.h.ci95, T.h.covers],
    ME_N.map(k => { const ph = M.at[L][k] / k, e = se(k), lo = ph - 1.96 * e, hi = ph + 1.96 * e;
      return row([fint(k), fint(M.at[L][k]), ppc(ph), ppc(e), nf(Math.abs(ph - p) / e, 2), `[${ppc(lo < 0 ? 0 : lo)}${LANG === 'ro' ? ';' : ','} ${ppc(hi)}]`, lo <= p && p <= hi ? T.h.yes : T.h.no], k === Nn ? 'sel' : ''); }), T.meNote);
  const Ns = M.Ns, est = M.est[L], nw = single('me-chart', 350);
  plot('me-chart', [
    { x: Ns, y: est.map(v => (Math.abs(v - p) > 0 ? 100 * Math.abs(v - p) : null)), mode: 'lines', line: { color: DBLUE, width: 1.2 }, name: '|estimate - p|', hovertemplate: 'N = %{x}<br>|error| %{y:.3g} pp<extra></extra>' },
    { x: Ns, y: Ns.map(k => 100 * se(k)), mode: 'lines', line: { color: RED, width: 1.8 }, name: 'SE = sqrt(p(1 - p)/N)', hovertemplate: 'N = %{x}<br>SE %{y:.3g} pp<extra></extra>' },
    { x: Ns, y: Ns.map(k => 196 * se(k)), mode: 'lines', line: { color: RED, width: 1.2, dash: 'dash' }, name: '1.96 SE', hoverinfo: 'skip' },
    { x: [Ns[0], Ns[Ns.length - 1]], y: [10 * p, 10 * p], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, name: '10% of p', hoverinfo: 'skip' }
  ], layout({ shapes: [vl(Nn, DARK)], xaxis: XT('number of simulations N (log scale)', { type: 'log', dtick: 1, exponentformat: 'power' }),
    yaxis: XT('error (percentage points, log scale)', { type: 'log', exponentformat: 'power', dtick: 1 }), legend: LEG(nw, -0.2) }));
  const k = M.at[L][Nn], ph = k / Nn;
  steps('me-we', [T.mew1(nf(m, 4), nf(mu, 3), nf(s, 3), nf(sd, 2), L, nf(z, 3), ppc(p)), T.mew2(fint(Nn), fint(k), ppc(ph)),
    T.mew3(nf(p, 5), fint(Nn), ppc(se(Nn)), nf(Math.abs(ph - p) / se(Nn), 2)), T.mew4(fint(Math.ceil(1.96 ** 2 * (1 - p) / (0.01 * p))))]);
  out('me-out', [T.me(L, fint(Nn), ppc(ph), ppc(p), ppc(se(Nn)))]);
}

// ----------------------------------------------------------------- X10. a two-step tree (frame 48)
function ttBlock() {
  const u = +$('tt-u').value, d = +$('tt-d').value, p = +$('tt-p').value, n = +$('tt-n').value;
  label('tt-u').textContent = nf(u, 2); label('tt-d').textContent = nf(d, 2); label('tt-p').textContent = nf(p, 2); label('tt-n').textContent = n;
  const K = Array.from({ length: n + 1 }, (_, i) => n - i), S = K.map(k => 100 * u ** k * d ** (n - k)), P = K.map(k => binomPmf(k, n, p));
  table('tt-tab', ['K', T.h.path, T.h.paths, 'S<sub>n</sub>', T.h.probK],
    K.map((k, i) => row([k, 'u'.repeat(k) + 'd'.repeat(n - k), fint(Math.exp(lchoose(n, k))), nf(S[i], 2), nf(P[i], 4)], S[i] < 100 - 1e-9 ? 'sel' : '')));
  let lt = 0, at = 0; K.forEach((_, i) => { if (S[i] < 100 - 1e-9) lt += P[i]; else if (Math.abs(S[i] - 100) < 1e-9) at += P[i]; });
  const g = p * u + (1 - p) * d, e = 100 * g ** n;
  steps('tt-we', [T.ttw1(K.map((k, i) => `${'u'.repeat(k) + 'd'.repeat(n - k)} ${nf(S[i], 2)} (${nf(P[i], 4)})`).join('; ')),
    T.ttw2(nf(p, 2), nf(u, 2), nf(1 - p, 2), nf(d, 2), nf(g, 4), n, nf(e, 2)), T.ttw3(nf(lt, 4), nf(lt + at, 4), at ? nf(at, 4) : null),
    T.ttw4(nf(d, 2), nf(u, 2), nf((1 - d) / (u - d), 4))]);
  out('tt-out', [T.tt(n, nf(e, 2), nf(lt, 4))]);
}

// ----------------------------------------------------------------- X11. the CRR tree calibrated to the S&P 500 (frame 49)
const CR_N = [1, 2, 3, 5, 10, 21, 63, 125, 126, 251, 252, 253];
const CRR = {};
function crrParts(n) {
  if (CRR[n]) return CRR[n];
  const B = D.bin, dt = 1 / n, u = Math.exp(B.sigma * Math.sqrt(dt)), d = 1 / u, p = (Math.exp(B.mu * dt) - d) / (u - d);
  let lt = 0, eq = 0, cum = 0, q05 = null;
  for (let k = 0; k <= n; k++) { const pr = binomPmf(k, n, p); if (2 * k < n) lt += pr; else if (2 * k === n) eq += pr; cum += pr; if (q05 === null && cum >= 0.05) q05 = 100 * u ** (2 * k - n); }
  return (CRR[n] = { u, d, p, lt, eq, le: lt + eq, q05 });
}
function crrBlock() {
  const n = +$('cr-n').value, B = D.bin, ml = B.mu - B.sigma ** 2 / 2, ln = Phi(-ml / B.sigma);
  table('cr-tab', ['n', T.h.dt, 'u', 'd', 'p', T.h.pl, T.h.pe, T.h.ple, T.h.q05],
    [...CR_N.map(k => { const o = crrParts(k); return row([k, `1/${k}`, nf(o.u, 4), nf(o.d, 4), nf(o.p, 4), pc(o.lt, 2), pc(o.eq, 2), pc(o.le, 2), nf(o.q05, 1)], k === n ? 'sel' : ''); }),
      row([T.h.lognormal, '→ 0', '', '', '', pc(ln, 2), pc(0, 2), pc(ln, 2), nf(100 * Math.exp(ml + B.sigma * normPpf(0.05)), 1)])], T.crNote);
  const o = crrParts(n), Pn = panels('cr-chart', 2, { h: 380, hs: 280, extra: 130, gap: 0.12 }), [a, b] = Pn.ax, tr = [];
  const ks = Array.from({ length: n + 1 }, (_, k) => k).filter(k => binomPmf(k, n, o.p) > 1e-5), Sv = k => 100 * o.u ** (2 * k - n), w = k => Sv(k) * (o.u / o.d - 1) * 0.85;
  [['S_T < 100', k => 2 * k < n, RED], ['S_T = 100', k => 2 * k === n, AMBER], ['S_T > 100', k => 2 * k > n, DBLUE]].forEach(([nm, f, c]) => {
    const kk = ks.filter(f); if (!kk.length) return;
    tr.push(on2(a, { x: kk.map(Sv), y: kk.map(k => binomPmf(k, n, o.p)), width: kk.map(w), type: 'bar', marker: { color: c }, name: nm, hovertemplate: 'S_T = %{x:.2f}<br>probability %{y:.4f}<extra></extra>' }));
  });
  const nn = Array.from({ length: 260 }, (_, i) => i + 1), P2 = nn.map(crrParts);
  tr.push(on2(b, { x: nn, y: P2.map(z => 100 * z.lt), mode: 'markers', marker: { color: RED, size: 3 }, name: 'P(S_T < 100)', hovertemplate: 'n = %{x}<br>P(<) %{y:.2f}%<extra></extra>' }));
  tr.push(on2(b, { x: nn, y: P2.map(z => 100 * z.le), mode: 'markers', marker: { color: DBLUE, size: 3 }, name: 'P(S_T <= 100)', hovertemplate: 'n = %{x}<br>P(<=) %{y:.2f}%<extra></extra>' }));
  tr.push(on2(b, { x: [1, 260], y: [100 * ln, 100 * ln], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `lognormal limit ${(100 * ln).toFixed(1)}%`, hoverinfo: 'skip' }));
  tr.push(on2(b, { x: [n], y: [100 * o.le], mode: 'markers', marker: { size: 9, color: GREEN }, name: `n = ${n}`, hovertemplate: 'n = %{x}<br>%{y:.2f}%<extra></extra>' }));
  setAx(Pn, 0, XT('S_T'), XT('probability'));
  setAx(Pn, 1, XT('number of steps n'), XT('probability (%)', { range: [10, 65] }));
  plot('cr-chart', tr, layout(Object.assign(Pn.L, { barmode: 'overlay', annotations: [ptitle(a, `distribution of S_T, n = ${n}`), ptitle(b, 'P(S_T < 100) and P(S_T <= 100) by n')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.22) })));
  const z = -ml / B.sigma;
  steps('cr-we', [T.crw1(n, nf(B.sigma, 4), nf(o.u, 4), nf(o.d, 4), nf(o.p, 4)), T.crw2(n, n % 2 ? nf(n / 2, 1) : n / 2),
    n % 2 ? T.crw3o(n, (n - 1) / 2, pc(o.lt, 2)) : T.crw3e(n, n / 2, pc(o.le, 2), pc(o.eq, 2), pc(o.lt, 2)),
    T.crw4(nf(ml, 4), nf(B.sigma, 4), nf(z, 4), pc(ln, 2), n % 2 ? null : pc((o.lt + o.le) / 2, 2))]);
  out('cr-out', [T.cr(n, pc(o.lt, 2), pc(o.le, 2), pc(ln, 2))]);
}

// ----------------------------------------------------------------- X12. the AR(1) process: simulation (frame 56)
const AR_PHI = [0, 0.5, 0.8, 0.9, 0.95, 0.99];
function arBlock() {
  const phi = +$('ar-phi').value, e = D.eps; label('ar-phi').textContent = nf(phi, 2);
  const list = AR_PHI.includes(phi) ? AR_PHI : [...AR_PHI, phi].sort((x, y) => x - y);
  const hl = f => (f > 0 ? Math.log(0.5) / Math.log(f) : null);
  table('ar-tab', ['φ', '1/(1 − φ²)', 'φ<sup>5</sup>', T.h.hl, T.h.var500, T.h.rho500, T.h.varSim, T.h.rhoSim],
    list.map(f => { const x = ar1(f, e), sim = N.proc[f === 0.5 ? '0.5' : f === 0.95 ? '0.95' : ''];
      return row([nf(f, 2), nf(1 / (1 - f * f), 2), nf(f ** 5, 3), hl(f) === null ? '0' : nf(hl(f), 1), nf(sdev(x) ** 2, 2), nf(lagCorr(x, 5), 3), sim ? nf(sim.var, 2) : '—', sim ? nf(sim.rho5, 3) : '—'], f === phi ? 'sel' : ''); }), T.arNote);
  const x = ar1(phi, e);
  steps('ar-we', [T.arw1(nf(phi, 2), nf(1 / (1 - phi * phi), 3)), T.arw2(nf(phi, 2), nf(phi, 2), nf(phi ** 5, 3)), T.arw3(nf(phi, 2), hl(phi) === null ? null : nf(hl(phi), 1)),
    T.arw4(nf(sdev(x) ** 2, 2), nf(lagCorr(x, 5), 3))]);
  out('ar-out', [T.ar(nf(phi, 2), nf(1 / (1 - phi * phi), 2), hl(phi) === null ? '0' : nf(hl(phi), 1))]);
}

// ----------------------------------------------------------------- X13. GBM: mean, median and quantiles (frame 66)
function gbmPar(k) { const o = moments(k), ml = o.A * o.m / 100, sg = o.s * Math.sqrt(o.A) / 100; return { ml, sg, mu: ml + sg * sg / 2, A: o.A }; }
function gmBlock() {
  const k = $('gm-a').value, Tn = +$('gm-T').value, q = +$('gm-q').value, zq = normPpf(q); label('gm-T').textContent = nf(Tn, 1);
  const lv = pc(q, 0), f = g => ({ e: 100 * Math.exp(g.mu * Tn), md: 100 * Math.exp(g.ml * Tn), q: 100 * Math.exp(g.ml * Tn + g.sg * Math.sqrt(Tn) * zq), pl: Phi(-g.ml * Math.sqrt(Tn) / g.sg) });
  table('gm-tab', [T.h.series, 'A', T.h.logDrift, T.h.sigma, T.h.mu, T.h.mean, T.h.median, T.h.qa(lv), T.h.ploss],
    D.assets.map(a => { const g = gbmPar(a), v = f(g); return row([SH[a], g.A, pc(g.ml, 2), pc(g.sg, 2), pc(g.mu, 2), big(v.e), big(v.md), big(v.q), pc(v.pl, 1)], a === k ? 'sel' : ''); }), T.gmNote);
  const g = gbmPar(k), v = f(g), zl = -g.ml * Math.sqrt(Tn) / g.sg;
  steps('gm-we', [T.gmw1(SH[k], nf(g.ml, 4), nf(g.sg, 4), nf(g.mu, 4), pc(g.mu, 2)), T.gmw2(nf(g.mu, 4), nf(Tn, 1), big(v.e), nf(g.ml, 4), big(v.md)),
    T.gmw3(lv, nf(zq, 3), nf(g.ml, 4), nf(Tn, 1), nf(g.sg, 4), big(v.q)), T.gmw4(nf(g.ml, 4), nf(Tn, 1), nf(g.sg, 4), nf(zl, 3), pc(v.pl, 1))]);
  out('gm-out', [T.gm(SH[k], nf(Tn, 1), big(v.e), big(v.md), nf(v.e / v.md, 2))]);
}

// ----------------------------------------------------------------- 16b. k-sigma days: data, GBM and the Normal distribution (table and chart under block 16)
function gcExtra(k) {
  const Ck = D.chk, names = ['sp500', 'bet', 'btc'], nS = Ck.sp500.n;
  const head = `<tr><th rowspan="2">k</th>${names.map(a => `<th colspan="3">${SH[a]}</th>`).join('')}<th rowspan="2">${T.h.normal}</th></tr>`
    + `<tr>${names.map(() => `<th>${T.h.data}</th><th>${T.h.gbmMed}</th><th>${T.h.gbmMax}</th>`).join('')}</tr>`;
  table('gc-tab', head, Ck.ks.map((kk, j) => row([nf(kk, 1), ...names.flatMap(a => [Ck[a].fr[j], nf(Ck[a].f50[j], 0), Ck[a].fmax[j]]), sig(2 * Phi(-kk) * nS, 2)], Math.abs(kk - k) < 1e-9 ? 'sel' : '')), T.gcNote(fint(nS)));
  const nw = single('gc-tail', 340), pos = v => (v > 0 ? v : null), tr = [];
  names.forEach(a => {
    tr.push({ x: Ck.ks, y: Ck[a].fr.map(pos), mode: 'lines+markers', line: { color: COL[a], width: 1.8 }, marker: { size: 6 }, name: `${SH[a]}: data`, hovertemplate: `${SH[a]}<br>k = %{x}<br>%{y} days<extra></extra>` });
    tr.push({ x: Ck.ks, y: Ck[a].f95.map(pos), mode: 'lines', line: { color: COL[a], width: 1.2, dash: 'dash' }, name: `${SH[a]}: GBM 95% quantile`, hovertemplate: `${SH[a]}<br>k = %{x}<br>GBM 95% quantile %{y}<extra></extra>` });
  });
  const j = Ck.ks.findIndex(v => Math.abs(v - k) < 1e-9);
  tr.push({ x: [k, k], y: [0.63, 400], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, name: `k = ${k.toFixed(1)}`, legendgroup: 'k', hoverinfo: 'skip' });
  tr.push({ x: names.map(() => k), y: names.map(a => pos(Ck[a].fr[j])), mode: 'markers', marker: { size: 11, color: names.map(a => COL[a]), line: { color: '#fff', width: 1 } }, name: 'data at the chosen k', legendgroup: 'k',
    customdata: names.map(a => SH[a]), hovertemplate: '%{customdata}<br>k = %{x}<br>%{y} days<extra></extra>' });
  plot('gc-tail', tr, layout({ xaxis: XT('k (standard deviations)'), yaxis: XT('days beyond k sd (log scale)', { type: 'log', range: [-0.2, 2.6], tickvals: [1, 2, 5, 10, 20, 50, 100, 200], ticktext: ['1', '2', '5', '10', '20', '50', '100', '200'] }), legend: LEG(nw, -0.2) }));
}

// ----------------------------------------------------------------- web lecture: table blocks travel inside a chart block
function placeExtras() {
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}

// ================================================================= wiring
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const players = {}, PL = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['rv-a'], rvBlock); PL('rv-play', 'rv-a', ['sp500', 'dax', 'bet'], 1500);
  on(['cq-a', 'cq-p'], cdfBlock); PL('cq-play', 'cq-p', [5, 4, 3, 2, 1, 0], 1000);
  on(['jt-y'], jointBlock); PL('jt-play', 'jt-y', range(1, 27, 1), 700);
  on(['ud-h'], udBlock); PL('ud-play', 'ud-h', range(1, 20, 1), 500);
  on(['cd-q'], condBlock); PL('cd-play', 'cd-q', range(2, 10, 1), 800);
  on(['ln-y'], llnBlock); PL('ln-play', 'ln-y', range(2000, 2020, 1), 600);
  on(['ru-s'], randuBlock); PL('ru-play', 'ru-s', range(1, 99, 2), 300);
  $('it-x').addEventListener('input', e => { if (!e.isTrusted) $('it-l').value = Math.min(4, +(0.4 * +$('it-x').value).toFixed(1)); });   // ▶ moves lambda with x
  on(['it-l', 'it-x'], invBlock); PL('it-play', 'it-x', range(1, 12, 0.5), 400);
  on(['mc-L'], mcBlock); PL('mc-play', 'mc-L', range(5, 20, 1), 600);
  on(['bn-n'], binBlock); PL('bn-play', 'bn-n', range(0, 6, 1), 1000);
  on(['pr-phi'], procBlock); PL('pr-play', 'pr-phi', range(0, 1, 0.05), 400);
  on(['sr-k'], spotBlock); PL('sr-play', 'sr-k', range(8, 2, -0.5), 700);
  on(['wn-k'], wienBlock); PL('wn-play', 'wn-k', range(0.5, 3, 0.1), 300);
  on(['gp-s'], gbmBlock); PL('gp-play', 'gp-s', [5, 10, 15, 19.26, 25, 30, 35, 40, 50, 60], 800);
  on(['gf-b'], fanBlock); PL('gf-play', 'gf-b', ['0.5', '0.8', '0.9', '0.95', '0.99'], 1500);
  on(['gc-k'], chkBlock); PL('gc-play', 'gc-k', range(3, 8, 0.5), 700);
  on(['dw-a', 'dw-x'], downDays); PL('dw-play', 'dw-x', range(0, 3, 0.25), 600);
  on(['am-a', 'am-h'], annualBlock); PL('am-play', 'am-h', [1, 5, 21, 63, 126, 252, 365], 900);
  on(['cb-a', 'cb-k'], chebBlock); PL('cb-play', 'cb-k', range(1.5, 8, 0.5), 600);
  on(['pf-p', 'pf-w'], pfBlock); PL('pf-play', 'pf-w', range(0, 1, 0.05), 400);
  on(['rg-p', 'rg-s', 'rg-a'], regBlock); PL('rg-play', 'rg-s', range(1, 4, 0.25), 500);
  on(['lt-y'], lltBlock); PL('lt-play', 'lt-y', range(2000, 2020, 1), 600);
  on(['lc-a', 'lc-c', 'lc-x'], lcgBlock); PL('lc-play', 'lc-a', range(0, 15, 1), 700);
  on(['ie-l', 'ie-u'], ieBlock); PL('ie-play', 'ie-u', range(0.05, 0.95, 0.05), 400);
  on(['me-L', 'me-n'], meBlock); PL('me-play', 'me-L', range(5, 20, 1), 600);
  on(['tt-u', 'tt-d', 'tt-p', 'tt-n'], ttBlock); PL('tt-play', 'tt-n', range(1, 6, 1), 900);
  on(['cr-n'], crrBlock); PL('cr-play', 'cr-n', ['1', '2', '3', '5', '10', '21', '63', '125', '126', '251', '252', '253'], 1000);
  on(['ar-phi'], arBlock); PL('ar-play', 'ar-phi', [0, 0.5, 0.8, 0.9, 0.95, 0.99], 1000);
  on(['gm-a', 'gm-T', 'gm-q'], gmBlock); PL('gm-play', 'gm-T', range(1, 30, 1), 400);
  const PFX = { 'random-variables': 'rv', 'cdf-quantile': 'cq', joint: 'jt', 'uncorrelated-dependent': 'ud', conditional: 'cd', lln: 'ln', randu: 'ru',
    'inverse-transform': 'it', 'monte-carlo': 'mc', binomial: 'bn', processes: 'pr', 'spot-the-real': 'sr', wiener: 'wn', 'gbm-paths': 'gp', 'gbm-fan': 'gf', 'gbm-check': 'gc',
    'down-days': 'dw', 'annual-moments': 'am', chebyshev: 'cb', portfolio: 'pf', 'two-regimes': 'rg', 'lln-table': 'lt', lcg: 'lc', 'inverse-exp': 'ie',
    'mc-error': 'me', 'two-step-tree': 'tt', 'crr-tree': 'cr', 'ar1-sim': 'ar', 'gbm-median': 'gm' };
  const VEIL = T.veil || {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  const all = [downDays, rvBlock, cdfBlock, annualBlock, chebBlock, jointBlock, pfBlock, udBlock, regBlock, condBlock, lltBlock, llnBlock, lcgBlock, randuBlock, ieBlock, invBlock,
    meBlock, mcBlock, ttBlock, crrBlock, binBlock, arBlock, procBlock, spotBlock, wienBlock, gmBlock, gbmBlock, fanBlock, chkBlock];
  all.forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH4 = { udCond, clipStats, normClipKurt, ser, quantile, pearson, lagCorr, condMoments, binomPmf, Phi, normPpf, exKurt, acf1, ar1, init, moments, downStats, beyond, pfSd, lcgRun, crrParts, gbmPar,
  blocks: { downDays, annualBlock, chebBlock, pfBlock, regBlock, lltBlock, lcgBlock, ieBlock, meBlock, ttBlock, crrBlock, arBlock, gmBlock, rvBlock, cdfBlock, jointBlock, udBlock, condBlock, llnBlock, randuBlock, invBlock, mcBlock, binBlock, procBlock, spotBlock, wienBlock, gbmBlock, fanBlock, chkBlock } };
if (window.CH4_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
