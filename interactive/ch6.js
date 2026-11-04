// Chapter 6 interactive page: one block per chart slide and per results table of the lecture, computed in the browser from ch6_data.js.
(function () {
'use strict';
const D = window.CH6_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH6_TEXT;
const MC = D.mcol, SC = D.scol, SH = D.short, LAB = D.labels, NUM = D.N, MOD = D.models, LV = D.levels;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22';
const MEN = m => (m === 'Mixture' ? 'Normal mixture' : m);            // model names in the charts (English)
const MN = m => T.mn[m];                                              // model names in the readouts and tables

// ================================================================= special functions
const SQ2PI = Math.sqrt(2 * Math.PI);
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
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
const chi2sf = (x, df) => (x <= 0 ? 1 : df === 1 ? erfc(Math.sqrt(x / 2)) : Math.exp(-x / 2));   // df = 1 or 2
const chi2ppf = (q, df) => (df === 1 ? normPpf(0.5 + q / 2) ** 2 : -2 * Math.log(1 - q));
function quantile(s, q) { const h = (s.length - 1) * q, i = Math.floor(h); return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
const SER = {};
function ser(k, bad) {              // daily log returns in %: original order, sorted, dates; bad = TLV with the two erroneous days
  const key = k + (bad ? ':bad' : '');
  if (SER[key]) return SER[key];
  const R = D.ret[k], dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v);
  let x = R.x.slice(), dates = dd.map(iso);
  if (bad) D.tlv_bad.forEach((b, j) => { x.splice(b.pos + j, 0, b.x); dates.splice(b.pos + j, 0, b.date); });
  const asc = Float64Array.from(x).sort();
  return (SER[key] = { x, dates, asc, n: x.length });
}
function mom(x) {                   // SFESumm statistics: sd with n - 1, skewness and kurtosis with n (scipy), Jarque-Bera
  const n = x.length; let m = 0, mn = Infinity, mx = -Infinity;
  for (const v of x) { m += v; if (v < mn) mn = v; if (v > mx) mx = v; }
  m /= n; let m2 = 0, m3 = 0, m4 = 0;
  for (const v of x) { const e = v - m, e2 = e * e; m2 += e2; m3 += e2 * e; m4 += e2 * e2; }
  m2 /= n; m3 /= n; m4 /= n;
  const skew = m3 / m2 ** 1.5, kurt = m4 / (m2 * m2), jb = n / 6 * (skew * skew + (kurt - 3) ** 2 / 4);
  return { n, mean: m, sd: Math.sqrt(m2 * n / (n - 1)), sd0: Math.sqrt(m2), skew, kurt, jb, jbp: chi2sf(jb, 2), min: mn, max: mx };
}
const ppy = S => S.n / ((Date.parse(S.dates[S.n - 1]) - Date.parse(S.dates[0])) / 864e5 / 365.25);
const empVar = (S, a) => -quantile(S.asc, a);
function empEs(S, a) { const q = quantile(S.asc, a); let s = 0, c = 0; for (const v of S.asc) { if (v > q) break; s += v; c++; } return -s / c; }
const FIT = (k, m) => D.fits[k].models[m];
const ell = (k, m) => FIT(k, m).ll;
const iL = a => LV.indexOf(a);
const mVar = (k, m, a) => D.tails[k][m].var[iL(a)];
const mEs = (k, m, a) => D.tails[k][m].es[iL(a)];
function kupiec(x, n, p) {          // Kupiec (1995): LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)]
  const ph = x / n, l0 = (n - x) * Math.log(1 - p) + x * Math.log(p), l1 = (n - x) * Math.log(1 - ph) + (x > 0 ? x * Math.log(ph) : 0);
  const lr = -2 * (l0 - l1);
  return { x, n, rate: ph, l0, l1, lr, p: chi2sf(lr, 1) };
}
function christoffersen(I) {        // Christoffersen (1998) independence test on a 0/1 exceedance sequence
  let n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (let t = 1; t < I.length; t++) { if (I[t - 1]) { if (I[t]) n11++; else n10++; } else if (I[t]) n01++; else n00++; }
  const p01 = n01 / (n00 + n01), p11 = n11 / (n10 + n11), p = (n01 + n11) / (n00 + n01 + n10 + n11);
  const lg = (k, q) => (k > 0 ? k * Math.log(q) : 0);
  const l0 = lg(n00 + n10, 1 - p) + lg(n01 + n11, p), l1 = lg(n00, 1 - p01) + lg(n01, p01) + lg(n10, 1 - p11) + lg(n11, p11), lr = -2 * (l0 - l1);
  return { n00, n01, n10, n11, p01, p11, pi: p, l0, l1, lr, p: chi2sf(lr, 1) };
}
const countBelow = (x, q) => { let c = 0; for (const v of x) if (v < q) c++; return c; };

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? '\u00a0' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const fnum = (x, d) => { const s = nf(x, d), i = s.search(/[.,]/), a = i < 0 ? s : s.slice(0, i), b = i < 0 ? '' : s.slice(i); return a.replace(/\B(?=(\d{3})+(?!\d))/g, sep) + b; };
function sig(x, d) {                // d significant digits; small numbers as m × 10^e
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (Math.abs(x) >= 1e-3) return nf(x, Math.max(0, d - 1 - e));
  let m = x / 10 ** e, ee = e; if (+Math.abs(m).toFixed(d - 1) >= 10) { m /= 10; ee++; }
  return nf(m, d - 1) + ' × 10<sup>' + (ee < 0 ? '−' : '') + Math.abs(ee) + '</sup>';
}
const fp = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const fpe = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : '= ' + nf(p, 3));
const lvl = p => { const s = (100 * p).toFixed(2).replace(/\.?0+$/, ''); return (LANG === 'ro' ? s.replace('.', ',') : s) + '%'; };
const lvlEn = p => (100 * p).toFixed(2).replace(/\.?0+$/, '') + '%';
const yr = s => s.slice(0, 4);
function table(id, head, rows) {    // rows: arrays of cells; a cell is a string or { v, c } (class); rows may carry .cls
  const cell = (x, j, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}>${o.v}</${tag}>`; };
  $(id).innerHTML = `<table class="xt"><thead><tr>${head.map((h, j) => cell(h, j, 'th')).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.map((x, j) => cell(x, j, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {         // n panels side by side when the chart is wide enough, stacked otherwise
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
  return { L, ax, nw, rows, cols, stack: nw };
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
const A0 = { x: 'x', y: 'y' };
const LOGP = r => ({ type: 'log', range: r, dtick: 1, exponentformat: 'power', showexponent: 'all' });
const pctTicks = vals => ({ type: 'log', tickvals: vals, ticktext: vals.map(String) });

// ================================================================= 1. the candidates
function candBlock() {
  const i = +$('ca-x').value, x = D.cand.thr[i], Cd = D.cand;
  label('ca-x').textContent = nf(x, 1);
  const Pn = panels('ca-chart', 2, { h: 380, hs: 270, extra: 150 }), [a, b] = Pn.ax, tr = [];
  const lab = { Normal: 'Normal', 'Student-t': 'Student-t (nu = 4)', 'Skewed-t': 'Skewed-t (nu = 4, lambda = -0.25)', GED: 'GED (beta = 1.2)',
    Mixture: 'Normal mixture (85% / 15%)', NIG: 'NIG', Stable: 'Stable (alpha = 1.7)' };
  const xs = Cd.x, inLin = xs.map(v => v >= -4 && v <= 4);
  MOD.forEach(m => {
    tr.push(on2(a, { x: xs.filter((_, j) => inLin[j]), y: Cd.f[m].filter((_, j) => inLin[j]), mode: 'lines', line: { color: MC[m], width: 1.5 }, name: lab[m], legendgroup: m, hovertemplate: `${MEN(m)}<br>x = %{x:.2f}<br>f(x) = %{y:.4f}<extra></extra>` }));
    tr.push(on2(b, { x: xs, y: Cd.f[m].map(v => Math.max(v, 1e-12)), mode: 'lines', line: { color: MC[m], width: 1.5 }, showlegend: false, legendgroup: m, hovertemplate: `${MEN(m)}<br>x = %{x:.2f}<br>f(x) = %{y:.2e}<extra></extra>` }));
  });
  // the tail beyond -x: shaded area P(X < -x) for the Normal and the Student-t (linear panel), thick tail segments of every model (log panel)
  const tl = xs.map(v => v <= -x + 1e-9);
  [['Normal', 0.35], ['Student-t', 0.3]].forEach(([m, op]) => tr.push(on2(a, { x: xs.filter((_, j) => tl[j] && inLin[j]).concat([-x]), y: Cd.f[m].filter((_, j) => tl[j] && inLin[j]).concat([0]), mode: 'lines', fill: 'tozeroy',
    fillcolor: hexA(MC[m], op), line: { color: MC[m], width: 2.6 }, name: `P(X < -x), ${MEN(m)}`, legendgroup: 'tail' + m, hoverinfo: 'skip' })));
  MOD.forEach(m => tr.push(on2(b, { x: xs.filter((_, j) => tl[j]), y: Cd.f[m].filter((_, j) => tl[j]).map(v => Math.max(v, 1e-12)), mode: 'lines', line: { color: MC[m], width: 4 }, opacity: 0.55, showlegend: false, legendgroup: m, hoverinfo: 'skip' })));
  setAx(Pn, 0, AX('x (standardised return)', { range: [-4, 4] }), AX(''));
  setAx(Pn, 1, AX('x (standardised return)', { range: [-7, 7] }), AX('', LOGP([-7, 0])));
  const shapes = [vln(-x, DARK, b, 'dash', 1.2)]; if (x <= 4) shapes.push(vln(-x, DARK, a, 'dash', 1.2));
  plot('ca-chart', tr, layout(Object.assign(Pn.L, { shapes, annotations: [ptitle(a, 'Density'), ptitle(b, `Density, log scale; dashed line: x = -${x.toFixed(1)}`)], legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.2) })));
  const pN = Cd.below.Normal[i];
  table('ca-tab', [T.h.model, `P(X &lt; −${nf(x, 1)})`, T.h.ratioN, T.h.oneIn], MOD.map(m => {
    const p = Cd.below[m][i];
    return [MN(m), sig(p, 2), m === 'Normal' ? '1' : fnum(p / pN, p / pN >= 100 ? 0 : 1), fint(1 / p)];
  }));
  out('ca-out', [T.ca(nf(x, 1), fnum(Cd.below['Student-t'][i] / pN, 0), sig(pN, 2))]);
}

// ================================================================= 2. DAX monthly (SFESumm)
function daxMonthlyBlock() {
  const M = D.dax_m, e = +$('dm-n').value, b0 = M.book[0], b1 = M.book[1], nw = single('dm-chart', 380);
  label('dm-n').textContent = M.ym[e].slice(0, 7);
  const wk = M.x.slice(b0, e + 1).reduce((j, v, i, arr) => (v < arr[j] ? i : j), 0) + b0;
  const tr = [{ x: M.ym, y: M.x, mode: 'lines', line: { color: '#B0B0B0', width: 1 }, name: 'DAX monthly log return, 2004-2026', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: M.ym.slice(b0, e + 1), y: M.x.slice(b0, e + 1), mode: 'lines', line: { color: TEAL, width: 1.4 }, name: `selected sample, ${M.ym[b0].slice(0, 7)} - ${M.ym[e].slice(0, 7)}`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: M.ym.slice(b0, b1 + 1), y: M.x.slice(b0, b1 + 1), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: 'SFESumm period, July 2004 - May 2014', hoverinfo: 'skip' },
    { x: [M.ym[wk]], y: [M.x[wk]], mode: 'markers+text', text: [`min ${M.x[wk].toFixed(2)}%`], textposition: 'middle right', textfont: { size: 11, color: RED }, marker: { size: 10, color: RED, symbol: 'diamond' },
      name: 'lowest month of the selected sample', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' },
    { x: [M.ym[0], M.ym[M.ym.length - 1]], y: [0, 0], mode: 'lines', line: { color: GREEN, width: 1, dash: 'dash' }, name: 'zero', hoverinfo: 'skip' }];
  const shapes = [{ type: 'rect', xref: 'x', yref: 'paper', x0: M.ym[b0], x1: M.ym[e], y0: 0, y1: 1, fillcolor: 'rgba(23,162,184,0.08)', line: { width: 0 } }, vln(M.ym[e], DARK, A0, 'dot', 1.4)];
  plot('dm-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Log return (%)'), shapes, legend: LEG(nw, -0.12), margin: { l: 56, r: 12, t: 16, b: 30 } }));
  const sB = mom(M.x.slice(b0, b1 + 1)), sW = mom(M.x.slice(b0, e + 1));
  const row = (lb, s) => [lb, fint(s.n), nf(s.min, 2), nf(s.max, 2), nf(s.sd, 2), nf(s.sd * Math.sqrt(12), 2), nf(s.skew, 2), nf(s.kurt, 2), nf(s.jb, 1), fp(s.jbp)];
  table('dm-tab', ['', 'n', 'min', 'max', T.h.sd, T.h.annvol, T.h.skew, T.h.kurt, 'JB', 'p'], [row('DAX 2004–2014', sB), Object.assign(row(`DAX ${M.ym[b0].slice(0, 7)} – ${M.ym[e].slice(0, 7)}`, sW), { cls: 'sel' })]);
  const worst = M.x.slice(b0, e + 1).reduce((j, v, i, arr) => (v < arr[j] ? i : j), 0);
  out('dm-out', [T.dm(M.ym[e].slice(0, 7), fint(sW.n), nf(sW.kurt, 2), nf(sW.min, 2), M.ym[b0 + worst].slice(0, 7))]);
}

// ================================================================= 3. descriptive statistics of the daily returns (table) + Jarque-Bera
function descBlock() {
  const bad = $('ds-b').value === 'raw', k = $('ds-s').value;
  const rows = D.names.map(n => {
    const S = ser(n, bad && n === 'tlv'), s = mom(S.x);
    return Object.assign([SH[n], S.dates[0], fint(s.n), nf(s.sd, 2), nf(s.sd * Math.sqrt(ppy(S)), 1), nf(s.skew, 2), nf(s.kurt, 1), fnum(s.jb, 0)], { cls: n === k ? 'sel' : '' });
  });
  table('ds-tab', ['', T.h.from, 'n', T.h.sd, T.h.annvol, T.h.skew, T.h.kurt, 'JB'], rows);
  const S = ser(k, bad && k === 'tlv'), s = mom(S.x);
  steps('ds-we', [T.ds1(SH[k], fint(s.n), nf(s.skew, 3), nf(s.kurt, 3)),
    T.ds2(nf(s.skew * s.skew, 4), nf((s.kurt - 3) ** 2 / 4, 3)),
    T.ds3(fint(s.n), fnum(s.jb, 1)), T.ds4(fnum(s.jb, 1), s.jbp > 0 ? '= ' + sig(s.jbp, 2) : `≈ 10<sup>−${fint(s.jb / 2 / Math.LN10)}</sup>`)]);
  const t0 = mom(ser('tlv').x), t1 = mom(ser('tlv', true).x);
  out('ds-out', [bad ? T.dsBad(nf(t1.kurt, 1), nf(t0.kurt, 1), fint(t1.n)) : T.dsClean(nf(t0.kurt, 1), fint(t0.n))]);
}

// ================================================================= 4. ML estimates (table) + the Normal model step by step
function estStr(m, p) {
  const f = (v, d) => nf(v, d);
  switch (m) {
    case 'Normal': return `μ̂ = ${f(p.mu, 3)}, σ̂ = ${f(p.sigma, 3)}`;
    case 'Student-t': return `ν̂ = ${f(p.nu, 2)}, ${T.h.loc} ${f(p.loc, 3)}, ${T.h.scale} ${f(p.scale, 3)}`;
    case 'Skewed-t': return `ν̂ = ${f(p.nu, 2)}, λ̂ = ${f(p.lam, 3)}, μ̂ = ${f(p.mu, 3)}, σ̂ = ${f(p.sigma, 2)}`;
    case 'GED': return `β̂ = ${f(p.beta, 2)}, ${T.h.loc} ${f(p.loc, 3)}, ${T.h.scale} ${f(p.scale, 3)}`;
    case 'Mixture': return `ŵ = ${f(100 * p.w[0], 0)}%: N(${f(p.mu[0], 3)}; ${f(p.sd[0], 2)}²); ${f(100 * p.w[1], 0)}%: N(${f(p.mu[1], 3)}; ${f(p.sd[1], 2)}²)`;
    case 'NIG': return `â = ${f(p.a, 3)}, b̂ = ${f(p.b, 3)}, ${T.h.loc} ${f(p.loc, 3)}, ${T.h.scale} ${f(p.scale, 3)}`;
    default: return `α̂ = ${f(p.alpha, 3)}, β̂ = ${f(p.beta, 3)}, γ̂ = ${f(p.gamma, 3)}, δ̂<sub>0</sub> = ${f(p.delta0, 3)}`;
  }
}
function mlBlock() {
  const k = $('ml-s').value, best = MOD.reduce((a, m) => (ell(k, m) > ell(k, a) ? m : a), MOD[0]);
  table('ml-tab', [T.h.model, 'k', { v: T.h.est, c: 'l' }, 'ℓ(θ̂)'], MOD.map(m => [MN(m), String(FIT(k, m).k), { v: estStr(m, FIT(k, m).params), c: 'l' }, { v: fnum(ell(k, m), 1), c: m === best ? 'hi' : '' }]));
  const S = ser(k), s = mom(S.x), l2 = Math.log(2 * Math.PI * s.sd0 * s.sd0), l = -S.n / 2 * (l2 + 1);
  steps('ml-we', [T.ml1(SH[k], fint(S.n), nf(s.mean, 4), nf(s.sd0, 4)), T.ml2(nf(s.sd0, 4), nf(s.sd0 * s.sd0, 4), nf(l2, 4)),
    T.ml3(fint(S.n), nf(l2, 4), fnum(l, 1)), T.ml4(fnum(ell(k, 'Normal'), 1), fnum(ell(k, best) - l, 1), MN(best))]);
  const t = FIT(k, 'Student-t').params;
  out('ml-out', [T.ml(SH[k], nf(t.nu, 2), nf(t.scale, 3), nf(t.scale * Math.sqrt(t.nu / (t.nu - 2)), 3), nf(s.sd, 3))]);
}

// ================================================================= 5. profile log-likelihood of nu
function profileBlock() {
  const k = $('pn-s').value, lev = $('pn-c').value, full = $('pn-y').value === 'full', nu = +$('pn-nu').value, Pr = D.prof[k];
  label('pn-nu').textContent = nf(nu, 1);
  const c = chi2ppf(+lev, 1), cut = -c / 2, nw = single('pn-chart', 380);
  const gi = Pr.grid.findIndex(v => Math.abs(v - nu) < 1e-6), lnu = Pr.ll[gi], ci = Pr.ci[lev], gc = Pr.grid_ci[lev];
  const ymin = full ? Math.min(...Pr.ll) - 5 : Math.max(Math.min(...Pr.ll), -40);
  const tr = [{ x: Pr.grid, y: Pr.ll, mode: 'lines', line: { color: BLUE, width: 1.8 }, name: 'profile log-likelihood l(nu) - l(nu_hat)', hovertemplate: 'nu = %{x:.1f}<br>%{y:.2f}<extra></extra>' },
    { x: [Pr.grid[0], Pr.grid[Pr.grid.length - 1]], y: [cut, cut], mode: 'lines', line: { color: RED, width: 1.2, dash: 'dash' }, name: `cut-off -${c.toFixed(2)} / 2 (${Math.round(100 * lev)}% interval)`, hoverinfo: 'skip' },
    { x: [Pr.nu, Pr.nu], y: [ymin, 2], mode: 'lines', line: { color: GREEN, width: 1.3, dash: 'dot' }, name: `ML estimate nu_hat = ${Pr.nu.toFixed(2)}`, hoverinfo: 'skip' },
    { x: [nu], y: [lnu], mode: 'markers', marker: { size: 10, color: ORANGE, line: { color: DARK, width: 1 } }, name: `selected nu = ${nu.toFixed(1)}`, hovertemplate: 'nu = %{x:.1f}<br>%{y:.2f}<extra></extra>' }];
  if (ci[0] !== null && ci[1] !== null) tr.push({ x: ci, y: [cut, cut], mode: 'markers', marker: { size: 8, symbol: 'line-ns-open', color: RED, line: { width: 2, color: RED } }, name: 'interval end points (step 0.01)', hoverinfo: 'skip' });
  plot('pn-chart', tr, layout({ xaxis: AX('Degrees of freedom nu', { range: [2.15, 6.05] }), yaxis: AX('Log-likelihood difference', { range: [ymin, 2] }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const lr = -2 * lnu, inside = lr <= c;
  steps('pn-we', [T.pn1(SH[k], nf(nu, 1), fnum(Pr.lmax + lnu, 2), fnum(Pr.lmax, 2), nf(Pr.nu, 2)), T.pn2(nf(lnu, 2), nf(lr, 2)),
    T.pn3(nf(lr, 2), nf(c, 2), Math.round(100 * lev), inside)]);
  const fci = v => (v === null ? '—' : nf(v, 2));
  out('pn-out', [T.pn(SH[k], nf(Pr.nu, 2), Math.round(100 * lev), nf(gc[0], 1), nf(gc[1], 1), fci(ci[0]), fci(ci[1])),
    T.pn4(nf(Pr.ll[Pr.grid.indexOf(3)], 2), nf(Pr.ll[Pr.grid.indexOf(5)], 1), full)]);
}

// ================================================================= 6. likelihood-ratio tests (worked example, table, chart)
const LRP = { t_vs_skt: ['Student-t', 'Skewed-t', false], n_vs_ged: ['Normal', 'GED', false], n_vs_t: ['Normal', 'Student-t', true] };
function lrCalc(k, pair) {
  const [m0, m1, bnd] = LRP[pair], l0 = ell(k, m0), l1 = ell(k, m1), lr = Math.max(2 * (l1 - l0), 0), p1 = chi2sf(lr, 1);
  return { m0, m1, bnd, l0, l1, lr, pn: p1, p: bnd ? 0.5 * p1 : p1, crit: bnd ? chi2ppf(0.9, 1) : chi2ppf(0.95, 1) };
}
function lrBlock() {
  const pair = $('lr-p').value, k = $('lr-s').value, R = lrCalc(k, pair), all = D.names.map(n => [n, lrCalc(n, pair)]);
  steps('lr-we', [T.lr1(SH[k], MN(R.m0), fnum(R.l0, 2), MN(R.m1), fnum(R.l1, 2)), T.lr2(fnum(R.l1, 2), fnum(R.l0, 2), fnum(R.lr, 2)),
    T.lr3(R.bnd, nf(R.crit, 2), fnum(R.lr, 2), fpe(R.p), fpe(R.pn)), T.lr4(R.lr > R.crit, MN(R.m0))]);
  table('lr-tab', ['', `ℓ(${MN(R.m0)})`, `ℓ(${MN(R.m1)})`, 'LR', 'p', T.h.dec], all.map(([n, r]) => Object.assign([SH[n], fnum(r.l0, 1), fnum(r.l1, 1), fnum(r.lr, 2), fp(r.p),
    { v: r.lr > r.crit ? T.h.rej : T.h.norej, c: r.lr > r.crit ? 'bad' : '' }], { cls: n === k ? 'sel' : '' })));
  const nw = single('lr-chart', 360), xs = logspace(-2, 4, 300), floor = 1e-12;
  const tr = [{ x: xs, y: xs.map(v => Math.max(chi2sf(v, 1), floor)), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: 'p-value under chi2(1)', hoverinfo: 'skip' },
    { x: xs, y: xs.map(v => Math.max(0.5 * chi2sf(v, 1), floor)), mode: 'lines', line: { color: PURPLE, width: 1.4, dash: 'dash' }, name: 'p-value under 50:50 chi2(0)/chi2(1) (boundary)', hoverinfo: 'skip' },
    { x: [xs[0], xs[xs.length - 1]], y: [0.05, 0.05], mode: 'lines', line: { color: GREEN, width: 1, dash: 'dot' }, name: '5% level', hoverinfo: 'skip' },
    { x: all.map(([, r]) => Math.max(r.lr, 0.011)), y: all.map(([, r]) => Math.max(r.p, floor)), text: all.map(([n, r]) => (r.p < floor ? '' : SH[n])), mode: 'markers+text', textposition: 'top center',
      textfont: { size: 10, color: DARK }, marker: { size: 9, color: all.map(([n]) => SC[n]), symbol: all.map(([, r]) => (r.p < floor ? 'triangle-down' : 'circle')), line: { color: DARK, width: 1 } },
      name: 'the seven series (p below 1e-12 drawn at 1e-12)', customdata: all.map(([n, r]) => [r.lr, r.p, SH[n]]), hovertemplate: '%{customdata[2]}<br>LR = %{customdata[0]:.2f}<br>p = %{customdata[1]:.3g}<extra></extra>' }];
  const sx = Math.max(R.lr, 0.011), sy = Math.max(R.p, floor);      // the series of the worked example: its LR and p read off the axes
  tr.push({ x: [0.01, sx, sx], y: [sy, sy, floor], mode: 'lines', line: { color: SC[k], width: 1.6, dash: 'dot' }, name: `${SH[k]}: LR = ${R.lr.toFixed(2)}, p = ${R.p < 1e-3 ? R.p.toExponential(1) : R.p.toFixed(3)}`, hoverinfo: 'skip' },
    { x: [sx], y: [sy], mode: 'markers', marker: { size: 18, color: 'rgba(0,0,0,0)', line: { color: SC[k], width: 2.5 } }, showlegend: false, hoverinfo: 'skip' });
  plot('lr-chart', tr, layout({ xaxis: AX('LR statistic (log scale)', { type: 'log', tickvals: [0.01, 0.1, 1, 3.84, 10, 100, 1000, 10000], ticktext: ['0.01', '0.1', '1', '3.84', '10', '100', '1000', '10000'] }),
    yaxis: AX('p-value (log scale)', Object.assign(LOGP([-12.3, 0.1]), { dtick: 2 })), shapes: [vln(R.crit, RED, A0, 'dash', 1.1)], legend: LEG(nw, -0.18), margin: { l: 64, r: 12, t: 14, b: 40 } }));
  out('lr-out', [T.lr(MN(R.m0), MN(R.m1), R.bnd, all.filter(([, r]) => r.lr > r.crit).length)]);
}

// ================================================================= 7. the Vuong test (worked example, table, cumulative chart)
const VUP = { vu_t_ged: ['Student-t', 'GED'], vu_t_stable: ['Student-t', 'Stable'], vu_skt_nig: ['Skewed-t', 'NIG'] };
function vuBlock() {
  const pair = $('vu-p').value, k = $('vu-s').value, [m1, m2] = VUP[pair], V = D.vu[k][pair], v = Math.sqrt(V.n) * V.mbar / V.sm, p = 2 * Phi(-Math.abs(v));
  steps('vu-we', [T.vu1(SH[k], MN(m1), MN(m2), fint(V.n), sig(V.mbar, 3), nf(V.sm, 4)), T.vu2(fint(V.n), V.mbar < 0 ? `(${sig(V.mbar, 3)})` : sig(V.mbar, 3), nf(V.sm, 4), nf(v, 2)),
    T.vu3(nf(v, 2), fpe(p), v > 1.96 ? MN(m1) : v < -1.96 ? MN(m2) : null), T.vu4(fint(V.n), V.mbar < 0 ? `(${sig(V.mbar, 3)})` : sig(V.mbar, 3), fnum(V.n * V.mbar, 1), fnum(ell(k, m1) - ell(k, m2), 1))]);
  table('vu-tab', ['', 'V', 'p', T.h.pref], D.names.map(n => { const t = NUM.tests[n][pair]; return Object.assign([SH[n], nf(t.v, 2), fp(t.p), t.v > 1.96 ? MN(m1) : t.v < -1.96 ? MN(m2) : T.h.none], { cls: n === k ? 'sel' : '' }); }));
  const S = ser(k), nw = single('vu-chart', 360);
  const tr = [{ x: V.idx.map(i => S.dates[i]), y: V.cum, mode: 'lines', line: { color: SC[k], width: 1.5 }, name: `cumulative sum of m_t = ln f(${MEN(m1)}) - ln f(${MEN(m2)})`, hovertemplate: '%{x}<br>%{y:.1f}<extra></extra>' },
    { x: V.top.map(([i]) => S.dates[i]), y: V.top.map(([i]) => { let j = V.idx.findIndex(t => t >= i); return V.cum[Math.max(j, 0)]; }), mode: 'markers', marker: { size: 9, color: RED, symbol: 'diamond' },
      name: 'three days with the largest |m_t|', customdata: V.top.map(([i, m]) => [S.x[i], m]), hovertemplate: '%{x}<br>return %{customdata[0]:.2f}%<br>m_t = %{customdata[1]:.2f}<extra></extra>' }];
  plot('vu-chart', tr, layout({ xaxis: AX(''), yaxis: AX('Cumulative log-likelihood difference'), shapes: [hln(0, DARK, A0, 'dash', 1)], legend: LEG(nw, -0.12), margin: { l: 60, r: 12, t: 14, b: 30 } }));
  out('vu-out', [T.vu(SH[k], V.top.map(([i, m]) => [S.dates[i], nf(S.x[i], 2), nf(m, 2)]))]);
}

// ================================================================= 8. AIC, BIC and Akaike weights (table + worked example)
function icRows(k) {
  const n = D.fits[k].n, R = MOD.map(m => { const f = FIT(k, m); return { m, k: f.k, l: f.ll, aic: -2 * f.ll + 2 * f.k, bic: -2 * f.ll + f.k * Math.log(n), aicc: -2 * f.ll + 2 * f.k + 2 * f.k * (f.k + 1) / (n - f.k - 1) }; });
  const amin = Math.min(...R.map(r => r.aic)), bmin = Math.min(...R.map(r => r.bic)), ws = R.reduce((s, r) => s + Math.exp(-(r.aic - amin) / 2), 0);
  const wb = R.reduce((s, r) => s + Math.exp(-(r.bic - bmin) / 2), 0);
  R.forEach(r => { r.da = r.aic - amin; r.db = r.bic - bmin; r.w = Math.exp(-r.da / 2) / ws; r.wb = Math.exp(-r.db / 2) / wb; });
  return { n, R, amin, bmin, ws };
}
function aicBlock() {
  const k = $('ai-s').value, m = $('ai-m').value, I = icRows(k), r = I.R.find(x => x.m === m);
  table('ai-tab', ['', 'k', 'ℓ(θ̂)', 'AIC', 'BIC', 'AICc', 'ΔAIC', 'ΔBIC', 'w'], I.R.map(x => Object.assign([MN(x.m), String(x.k), fnum(x.l, 1), fnum(x.aic, 1), fnum(x.bic, 1), fnum(x.aicc, 1),
    { v: nf(x.da, 1), c: x.da < 1e-9 ? 'hi' : '' }, { v: nf(x.db, 1), c: x.db < 1e-9 ? 'hi' : '' }, nf(x.w, 3)], { cls: x.m === m ? 'sel' : '' })));
  const ln = Math.log(I.n), terms = I.R.map(x => Math.exp(-x.da / 2));
  steps('ai-we', [T.ai1(MN(m), SH[k], fnum(r.l, 2), r.k, fnum(r.aic, 2)), T.ai2(fnum(r.l, 2), r.k, fint(I.n), nf(ln, 3), fnum(r.bic, 2), fnum(-2 * r.l, 2)),
    T.ai3(fnum(r.aic, 2), fnum(I.amin, 2), nf(r.da, 2), sig(Math.exp(-r.da / 2), 3)),
    T.ai4(I.R.map(x => `${MN(x.m)} ${sig(Math.exp(-x.da / 2), 2)}`).join('; '), sig(terms.reduce((a, b) => a + b, 0), 4), sig(Math.exp(-r.da / 2), 3), sig(r.w, 3))]);
  const best = I.R.find(x => x.da < 1e-9), sec = I.R.filter(x => x !== best).reduce((a, b) => (b.da < a.da ? b : a));
  out('ai-out', [T.ai(SH[k], MN(best.m), MN(sec.m), nf(sec.da, 1), sig(sec.w, 2), nf(Math.log(I.n), 2))]);
}

// ================================================================= 9. Delta AIC for seven series (heat map)
function deltaBlock() {
  const crit = $('da-c').value, key = crit === 'aic' ? 'da' : 'db', rows = D.names.map(n => icRows(n).R), nw = narrow('da-chart');
  $('da-chart').style.height = (nw ? 520 : 420) + 'px';
  const Z = rows.map(R => R.map(r => r[key])), zmax = Math.log10(Math.max(...Z.flat(), 10) + 1);
  const xl = MOD.map(m => (nw ? SH_M[m] : MEN(m))), yl = D.names.map(n => LAB[n]);
  const ann = [];
  Z.forEach((row, i) => row.forEach((v, j) => ann.push({ x: xl[j], y: yl[i], text: v < 0.5 ? '0' : v.toFixed(0), showarrow: false,
    font: { size: nw ? 9 : 11, color: v > 300 ? '#fff' : '#000', family: 'Inter, sans-serif' }, ...(v < 0.5 ? { font: { size: nw ? 10 : 12, color: '#000' } } : {}) })));
  plot('da-chart', [{ type: 'heatmap', x: xl, y: yl, z: Z.map(r => r.map(v => Math.log10(v + 1))), zmin: 0, zmax, colorscale: 'Blues', reversescale: true, customdata: Z,
    colorbar: { title: { text: `Delta ${crit.toUpperCase()} + 1 (log scale)`, side: 'right', font: { size: 11 } }, tickvals: [0, 1, 2, 3], ticktext: ['1', '10', '100', '1000'], thickness: 12, len: 0.9 },
    hovertemplate: '%{y}, %{x}<br>Delta = %{customdata:.1f}<extra></extra>' }],
  layout({ xaxis: AX('', { side: 'bottom', tickangle: nw ? -45 : 0 }), yaxis: AX('', { autorange: 'reversed' }), annotations: ann, margin: { l: 120, r: 20, t: 30, b: nw ? 90 : 50 },
    title: { text: `Delta ${crit.toUpperCase()} = ${crit.toUpperCase()} - min ${crit.toUpperCase()} (0 = best model of the series)`, x: 0, xanchor: 'left', font: { size: 12, color: C.navy } } }));
  table('da-tab', ['', T.h.bestA, T.h.bestB, T.h.second], D.names.map((n, i) => {
    const R = rows[i], ba = R.find(r => r.da < 1e-9), bb = R.find(r => r.db < 1e-9), s2 = R.filter(r => r[key] > 1e-9).reduce((a, b) => (b[key] < a[key] ? b : a));
    return [SH[n], MN(ba.m), { v: MN(bb.m), c: bb.m !== ba.m ? 'hi' : '' }, `${MN(s2.m)} (${nf(s2[key], 1)})`];
  }));
  const nR = rows.map(R => R.find(r => r.m === 'Normal')[key]);
  out('da-out', [T.da(crit.toUpperCase(), fint(Math.min(...nR)), fint(Math.max(...nR)), D.names.filter((n, i) => rows[i].find(r => r.da < 1e-9).m !== rows[i].find(r => r.db < 1e-9).m).map(n => SH[n]))]);
}
const SH_M = { Normal: 'Normal', 'Student-t': 't', 'Skewed-t': 'skew-t', GED: 'GED', Mixture: 'mixture', NIG: 'NIG', Stable: 'stable' };

// ================================================================= 10. KS on five returns (worked example, table, chart)
const K5 = [-2.1, -0.4, 0.3, 0.9, 1.6];
function ksFive(sg, est) {
  const n = 5, m = est ? K5.reduce((a, b) => a + b, 0) / n : 0, s = est ? Math.sqrt(K5.reduce((a, b) => a + (b - m) ** 2, 0) / n) : sg;
  const u = K5.map(r => Phi((r - m) / s)), dp = u.map((v, i) => (i + 1) / n - v), dm = u.map((v, i) => v - i / n);
  let D5 = -1, at = 0, side = 0; u.forEach((_, i) => { if (dp[i] > D5) { D5 = dp[i]; at = i; side = 0; } if (dm[i] > D5) { D5 = dm[i]; at = i; side = 1; } });
  return { m, s, u, dp, dm, D5, at, side, crit: est ? NUM.lillie['5'].ks : 0.563 };
}
function ksFiveBlock() {
  const est = $('k5-h').value === 'est', sg = +$('k5-s').value, R = ksFive(sg, est);
  label('k5-s').textContent = nf(sg, 1); $('k5-s').disabled = est;
  table('k5-tab', ['i', 'r<sub>(i)</sub>', 'u<sub>(i)</sub>', 'i/n', '(i − 1)/n', 'i/n − u<sub>(i)</sub>', 'u<sub>(i)</sub> − (i − 1)/n'], K5.map((r, i) => [String(i + 1), nf(r, 1), nf(R.u[i], 3), nf((i + 1) / 5, 1), nf(i / 5, 1),
    { v: nf(R.dp[i], 3), c: i === R.at && R.side === 0 ? 'hi' : '' }, { v: nf(R.dm[i], 3), c: i === R.at && R.side === 1 ? 'hi' : '' }]));
  steps('k5-we', [T.k51(est, nf(R.m, 2), nf(R.s, est ? 3 : 1)), T.k52(R.u.map(v => nf(v, 3)).join('; ')),
    T.k53(R.at + 1, R.side, nf(R.D5, 3)), T.k54(nf(R.D5, 3), nf(R.crit, 3), est, R.D5 > R.crit)]);
  const nw = single('k5-chart', 340), xs = Array.from({ length: 301 }, (_, i) => -4 + 8 * i / 300), sx = [-4], sy = [0];
  K5.forEach((r, i) => { sx.push(r, r); sy.push(i / 5, (i + 1) / 5); }); sx.push(4); sy.push(1);
  const r = K5[R.at], y0 = R.u[R.at], y1 = R.side === 0 ? (R.at + 1) / 5 : R.at / 5;
  plot('k5-chart', [{ x: sx, y: sy, mode: 'lines', line: { color: BLUE, width: 1.8, shape: 'linear' }, name: 'empirical distribution function F_n', hoverinfo: 'skip' },
    { x: xs, y: xs.map(v => Phi((v - R.m) / R.s)), mode: 'lines', line: { color: RED, width: 1.5 }, name: est ? `Normal(${R.m.toFixed(2)}, ${R.s.toFixed(2)}^2), estimated` : `Normal(0, ${R.s.toFixed(1)}^2)`, hoverinfo: 'skip' },
    { x: [r, r], y: [y0, y1], mode: 'lines+markers', line: { color: ORANGE, width: 3 }, marker: { size: 7 }, name: `largest gap D_5 = ${R.D5.toFixed(3)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('Return r (%)', { range: [-4, 4] }), yaxis: AX('F(r)', { range: [-0.02, 1.02] }), legend: LEG(nw, -0.18), margin: { l: 50, r: 12, t: 14, b: 40 } }));
  out('k5-out', [T.k5(nf(R.D5, 3), nf(R.crit, 3), est, R.D5 > R.crit)]);
}

// ================================================================= 11. KS and AD gaps in the tails
const symlog = v => (Math.abs(v) <= 10 ? v / 10 : Math.sign(v) * (1 + Math.log10(Math.abs(v) / 10)));
function ksTailBlock() {
  const k = $('kt-s').value, S = ser(k), x = S.asc, n = S.n, s = mom(S.x), mu = s.mean, sd = s.sd0;
  const F = Array.from(x, v => Phi((v - mu) / sd)), d = F.map((f, i) => (i + 1) / n - f), Fc = F.map(f => Math.min(Math.max(f, 1e-8), 1 - 1e-8));
  const z = d.map((v, i) => Math.sqrt(n) * v / Math.sqrt(Fc[i] * (1 - Fc[i])));
  let i = 0, j = 0; d.forEach((v, t) => { if (Math.abs(v) > Math.abs(d[i])) i = t; }); z.forEach((v, t) => { if (Math.abs(v) > Math.abs(z[j])) j = t; });
  const Pn = panels('kt-chart', 2, { h: 380, hs: 280, extra: 170, vgap: 0.2 }), [a, b] = Pn.ax, X = Array.from(x);
  const tv = [-1000, -100, -10, 0, 10, 100, 1000];
  const tr = [on2(a, { x: X, y: d, mode: 'lines', line: { color: BLUE, width: 1.2 }, name: 'F_n(x) - F(x)', hovertemplate: 'x = %{x:.2f}%<br>gap %{y:.4f}<extra></extra>' }),
    on2(a, { x: [x[i]], y: [d[i]], mode: 'markers', marker: { size: 9, color: RED }, name: `KS maximum at x = ${x[i].toFixed(2)}%`, hoverinfo: 'skip' }),
    on2(b, { x: X, y: z.map(symlog), customdata: z, mode: 'lines', line: { color: PURPLE, width: 1 }, name: 'sqrt(n) (F_n - F) / sqrt(F (1 - F))', hovertemplate: 'x = %{x:.2f}%<br>weighted gap %{customdata:.1f}<extra></extra>' }),
    on2(b, { x: [x[j]], y: [symlog(z[j])], mode: 'markers', marker: { size: 9, color: RED, symbol: 'diamond' }, name: `largest weighted gap at x = ${x[j].toFixed(2)}%`, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('Daily log return x (%)', { range: [-6, 6] }), AX('Gap'));
  setAx(Pn, 1, AX('Daily log return x (%)', { range: [x[0] - 0.5, x[n - 1] + 0.5] }), AX('Weighted gap (symmetric log scale)', { tickvals: tv.map(symlog), ticktext: tv.map(String) }));
  plot('kt-chart', tr, layout(Object.assign(Pn.L, { shapes: [hln(0, GREEN, a, 'dash', 0.9), hln(0, GREEN, b, 'dash', 0.9)],
    annotations: [ptitle(a, 'Unweighted gap (Kolmogorov-Smirnov)'), ptitle(b, 'Tail-weighted gap (Anderson-Darling weight)')], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  const dpl = (i + 1) / n - F[i], dmi = F[i] - i / n;
  steps('kt-we', [T.kt1(SH[k], fint(n), nf(mu, 3), nf(sd, 3)), T.kt2(fint(i + 1), nf(x[i], 2), nf(F[i], 4), nf((i + 1) / n, 4), nf(i / n, 4)),
    T.kt3(nf(dpl, 4), nf(dmi, 4), nf(Math.max(dpl, dmi), 4)), T.kt4(fint(n), nf(1.358 / Math.sqrt(n), 4), nf(Math.max(dpl, dmi), 4))]);
  out('kt-out', [T.kt(SH[k], nf(x[i], 2), nf(100 * F[i], 0), nf(Math.abs(d[i]), 3), nf(x[j], 2), fnum(Math.abs(z[j]), 0))]);
}

// ================================================================= 12. power of the three tests
const GPN = [50, 100, 250, 500, 1000];
function powerBlock() {
  const n = GPN[+$('gp-n').value], Pw = NUM.power; label('gp-n').textContent = fint(n);
  const Pn = panels('gp-chart', 2, { h: 380, hs: 270, extra: 140 }), [a, b] = Pn.ax, tr = [];
  const lab = { ks: 'Kolmogorov-Smirnov', cvm: 'Cramer-von Mises', ad: 'Anderson-Darling' }, col = { ks: RED, cvm: ORANGE, ad: BLUE };
  [['t5', a], ['contam', b]].forEach(([alt, ax], j) => {
    const upto = GPN.filter(v => v <= n);      // the full curves stay faint; the solid part grows up to the selected n
    ['ks', 'cvm', 'ad'].forEach(s => {
      tr.push(on2(ax, { x: GPN, y: GPN.map(v => Pw[alt][v][s]), mode: 'lines', line: { color: col[s], width: 1.2, dash: 'dot' }, opacity: 0.35, showlegend: false, legendgroup: s, hoverinfo: 'skip' }));
      tr.push(on2(ax, { x: upto, y: upto.map(v => Pw[alt][v][s]), mode: 'lines+markers', line: { color: col[s], width: 1.8 }, marker: { size: 7 }, name: lab[s], showlegend: j === 0, legendgroup: s, hovertemplate: `${lab[s]}<br>n = %{x}<br>%{y:.3f}<extra></extra>` }));
    });
    tr.push(on2(ax, { x: [n, n, n], y: ['ks', 'cvm', 'ad'].map(s => Pw[alt][n][s]), mode: 'markers', marker: { size: 12, color: ['ks', 'cvm', 'ad'].map(s => col[s]), line: { color: DARK, width: 1 } }, showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(ax, { x: [40, 1250], y: [0.05, 0.05], mode: 'lines', line: { color: GREEN, width: 1, dash: 'dash' }, name: '5% level', showlegend: j === 0, hoverinfo: 'skip' }));
  });
  const xa = AX('Sample size n (log scale)', { type: 'log', tickvals: GPN, ticktext: GPN.map(String), range: [Math.log10(40), Math.log10(1250)] });
  setAx(Pn, 0, xa, AX('Rejection rate of Normality', { range: [0, 1.05] })); setAx(Pn, 1, Object.assign({}, xa), AX('', { range: [0, 1.05] }));
  plot('gp-chart', tr, layout(Object.assign(Pn.L, { shapes: [vln(n, DARK, a, 'dot', 1.2), vln(n, DARK, b, 'dot', 1.2)],
    annotations: [ptitle(a, 'Student-t data, nu = 5'), ptitle(b, 'Normal data, 2% of days with 4x volatility')], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  table('gp-tab', ['n', 'KS (t)', 'CvM (t)', 'AD (t)', T.h.kst, T.h.cvmt, T.h.adt], GPN.map(v => Object.assign([fint(v), ...['ks', 'cvm', 'ad'].map(s => nf(100 * Pw.t5[v][s], 1) + '%'), ...['ks', 'cvm', 'ad'].map(s => nf(100 * Pw.contam[v][s], 1) + '%')], { cls: v === n ? 'sel' : '' })));
  const Li = NUM.lillie;
  table('gp-tab2', ['n', T.h.lil, T.h.ksk], ['5', '100', '1000'].map(v => [fint(+v), nf(Li[v].ks, 3), v === '5' ? nf(0.563, 3) : nf(1.358 / Math.sqrt(+v), 3)]));
  out('gp-out', [T.gp(fint(n), nf(100 * Pw.t5[n].ks, 0), nf(100 * Pw.t5[n].cvm, 0), nf(100 * Pw.t5[n].ad, 0), nf(100 * Pw.contam[n].ks, 0), nf(100 * Pw.contam[n].cvm, 0), nf(100 * Pw.contam[n].ad, 0))]);
}

// ================================================================= 13. EDF tests (table)
function edfBlock() {
  const k = $('ed-s').value, G = NUM.gof[k], I = icRows(k);
  const rk = key => { const o = [...MOD].sort((p, q) => key(p) - key(q)); return m => o.indexOf(m) + 1; };
  const rAD = rk(m => G[m].ad), rA = rk(m => I.R.find(r => r.m === m).aic);
  const bestKs = MOD.reduce((a, m) => (G[m].ks < G[a].ks ? m : a)), bestAd = MOD.reduce((a, m) => (G[m].ad < G[a].ad ? m : a));
  table('ed-tab', ['', 'KS', 'CvM', 'AD', T.h.rAD, T.h.rAIC], MOD.map(m => [MN(m), { v: nf(G[m].ks, 3), c: m === bestKs ? 'hi' : '' }, nf(G[m].cvm, 2), { v: nf(G[m].ad, 2), c: m === bestAd ? 'hi' : '' },
    String(rAD(m)), { v: String(rA(m)), c: rA(m) !== rAD(m) ? 'bad' : '' }]));
  const S = ser(k), crit = 1.358 / Math.sqrt(S.n), B = NUM.gof_boot;
  const items = [T.ed(SH[k], fint(S.n), nf(crit, 3), MOD.filter(m => rA(m) !== rAD(m)).map(MN))];
  if (k === 'sp500') items.push(T.ed2(['Normal', 'Student-t', 'Skewed-t'].map(m => [MN(m), nf(B[m].crit_boot.ks, 3), nf(B[m].crit_boot.ad, 2), nf(B[m].p_boot.ad, 3)]), nf(B['Student-t'].p_naive_ks, 3), B.Normal.B));
  out('ed-out', items);
}

// ================================================================= 14-15. PP and QQ plots of the S&P 500 (and the other series)
function ppqqData(k) {
  const S = ser(k), Q = D.pq[k], n = S.n, x = Q.sel.map(i => S.asc[i]), pe = Q.sel.map(i => (i + 0.5) / n);
  return { S, Q, n, x, pe };
}
function ppBlock() {
  const k = $('pp-s').value, m4 = $('pp-4').value, R = ppqqData(k), mods = ['Normal', 'Student-t', 'Skewed-t', m4];
  const G = gridp('pp-chart', 1, 4, { h: 330, hs: 260, extra: 120, gx: 0.05, gyn: 0.07 }), tr = [], ann = [];
  const idx = R.Q.s1;
  mods.forEach((m, j) => {
    const ax = G.ax[j], u = R.Q.m[m].u;
    tr.push(on2(ax, { x: idx.map(i => R.pe[i]), y: idx.map(i => u[i]), mode: 'markers', marker: { size: 3.5, color: MC[m] }, name: MEN(m), hovertemplate: `${MEN(m)}<br>empirical %{x:.3f}<br>model %{y:.3f}<extra></extra>` }));
    tr.push(on2(ax, { x: [0, 1], y: [0, 1], mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: '45-degree line', showlegend: j === 3, hoverinfo: 'skip' }));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], AX('Empirical probability', { range: [0, 1] }));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], AX(j === 0 || G.nw ? 'Model probability F(x)' : '', { range: [0, 1] }));
    ann.push(ptitle(ax, MEN(m)));
  });
  plot('pp-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.22) })));
  out('pp-out', [T.pp(SH[k], mods.map(m => [MN(m), nf(R.Q.m[m].maxdev, 3), nf(NUM.gof[k][m].ks, 3)]))]);
}
function qqBlock() {
  const k = $('qq-s').value, m4 = $('qq-4').value, R = ppqqData(k), mods = ['Normal', 'Student-t', 'Skewed-t', m4];
  const G = gridp('qq-chart', 1, 4, { h: 340, hs: 260, extra: 120, gx: 0.05, gyn: 0.07 }), tr = [], ann = [], idx = R.Q.s1;
  const lo = R.x[0], hi = R.x[R.x.length - 1];
  mods.forEach((m, j) => {
    const ax = G.ax[j], q = R.Q.m[m].q, qs = idx.map(i => q[i]);
    tr.push(on2(ax, { x: qs, y: idx.map(i => R.x[i]), mode: 'markers', marker: { size: 3.5, color: MC[m] }, name: MEN(m), hovertemplate: `${MEN(m)}<br>model %{x:.2f}%<br>observed %{y:.2f}%<extra></extra>` }));
    tr.push(on2(ax, { x: [lo, hi], y: [lo, hi], mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: '45-degree line', showlegend: j === 3, hoverinfo: 'skip' }));
    G.L[ax.xa] = Object.assign(G.L[ax.xa], AX('Model quantile (%)', { range: [Math.min(Math.min(...qs), lo) * 1.05, Math.max(Math.max(...qs), hi) * 1.05] }));
    G.L[ax.ya] = Object.assign(G.L[ax.ya], AX(j === 0 || G.nw ? 'Empirical quantile (%)' : '', { range: [lo * 1.08, hi * 1.08] }));
    ann.push(ptitle(ax, MEN(m)));
  });
  plot('qq-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.22) })));
  const Q = D.pq[k];
  table('qq-tab', [T.h.model, T.h.qmin, T.h.ratioObs], MOD.map(m => Object.assign([MN(m), nf(Q.m[m].qmin, 1), nf(Q.m[m].qmin / R.S.asc[0], 2)], { cls: mods.includes(m) ? 'sel' : '' })));
  out('qq-out', [T.qq(SH[k], nf(R.S.asc[0], 1), R.S.dates[R.S.x.indexOf(R.S.asc[0])], fint(R.n), sig(0.5 / R.n, 2))]);
}

// ================================================================= 16. QQ plots of BET, Bitcoin and Banca Transilvania
function qqAssetsBlock() {
  const kk = +$('qa-k').value, names = ['bet', 'btc', 'tlv'], mods = ['Normal', 'Student-t', 'Skewed-t']; label('qa-k').textContent = String(kk);
  const Pn = panels('qa-chart', 3, { h: 360, hs: 260, extra: 120, gap: 0.06 }), tr = [], ann = [], items = [];
  names.forEach((k, j) => {
    const R = ppqqData(k), ax = Pn.ax[j], idx = R.Q.s2, lo = R.x[0], hi = R.x[R.x.length - 1];
    mods.forEach(m => tr.push(on2(ax, { x: idx.map(i => R.Q.m[m].q[i]), y: idx.map(i => R.x[i]), mode: 'markers', marker: { size: 3.3, color: MC[m] }, name: MEN(m), showlegend: j === 0, legendgroup: m,
      hovertemplate: `${MEN(m)}<br>model %{x:.2f}%<br>observed %{y:.2f}%<extra></extra>` })));
    tr.push(on2(ax, { x: [lo, hi], y: [lo, hi], mode: 'lines', line: { color: '#000', width: 0.8, dash: 'dash' }, name: '45-degree line', showlegend: j === 0, hoverinfo: 'skip' }));
    const pos = R.Q.sel.indexOf(kk - 1), qk = mods.map(m => R.Q.m[m].q[pos]);
    tr.push(on2(ax, { x: qk, y: qk.map(() => R.x[pos]), mode: 'markers', marker: { size: 11, color: 'rgba(0,0,0,0)', line: { color: DARK, width: 2 } }, name: `k-th smallest return (k = ${kk})`, showlegend: j === 0, legendgroup: 'k', hoverinfo: 'skip' }));
    setAx(Pn, j, AX('Model quantile (%)'), AX(j === 0 || Pn.stack ? 'Empirical quantile (%)' : ''));
    ann.push(ptitle(ax, LAB[k]));
    const S = R.S, v = S.asc[kk - 1];
    items.push(T.qaItem(LAB[k], S.dates[S.x.indexOf(v)], nf(v, 2), nf(qk[0], 2), nf(qk[1], 2), nf(qk[2], 2)));
  });
  plot('qa-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.2) })));
  out('qa-out', [T.qa(kk), ...items]);
}

// ================================================================= 17. VaR and ES of every model (tables) + the Normal model step by step
const VM6 = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'NIG', 'Stable'];
function varEsBlock() {
  const a = LV[+$('ve-a').value], e = LV[+$('ve-e').value], k = $('ve-k').value, cols = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'NIG', 'Stable', 'Mixture'];
  const head = c => ['', T.h.data, ...c.map(MN)];
  table('ve-tab', head(cols), D.names.map(n => { const S = ser(n), d = empVar(S, a); return Object.assign([SH[n], nf(d, 2), ...cols.map(m => ({ v: nf(mVar(n, m, a), 2), c: m === 'Mixture' ? 'x' : '' }))], { cls: n === k ? 'sel' : '' }); }));
  table('ve-tab2', head(cols), D.names.map(n => { const S = ser(n); return Object.assign([SH[n], nf(empEs(S, e), 2), ...cols.map(m => ({ v: nf(mEs(n, m, e), 2), c: m === 'Mixture' ? 'x' : '' }))], { cls: n === k ? 'sel' : '' }); }));
  $('ve-h1').innerHTML = T.veH(lvl(a), true); $('ve-h2').innerHTML = T.veH(lvl(e), false);
  const p = FIT(k, 'Normal').params, za = normPpf(a), ze = normPpf(e), v = -(p.mu + p.sigma * za), es = -p.mu + p.sigma * phi(ze) / e, S = ser(k);
  steps('ve-we', [T.ve1(SH[k], nf(p.mu, 4), nf(p.sigma, 4), lvl(a), nf(za, 3)), T.ve2(nf(p.mu, 4), nf(za, 3), nf(p.sigma, 4), nf(v, 2), nf(empVar(S, a), 2)),
    T.ve3(lvl(e), nf(ze, 3), nf(phi(ze), 4), nf(p.mu, 4), nf(p.sigma, 4), nf(es, 2), nf(empEs(S, e), 2))]);
  out('ve-out', [T.ve(lvl(a), lvl(e))]);
}

// ================================================================= 18. model VaR relative to the data
function varModelsBlock() {
  const a = LV[+$('vm-a').value], mix = $('vm-x').value === 'mix', mods = mix ? [...VM6, 'Mixture'] : VM6, nw = single('vm-chart', 380);
  const xs = D.names.map(n => SH[n]), tr = mods.map(m => ({ type: 'bar', x: xs, y: D.names.map(n => mVar(n, m, a) / empVar(ser(n), a)), name: MEN(m), marker: { color: MC[m] },
    hovertemplate: `${MEN(m)}<br>%{x}: %{y:.3f}<extra></extra>` }));
  tr.push({ x: [xs[0], xs[xs.length - 1]], y: [1, 1], mode: 'lines', line: { color: '#000', width: 1, dash: 'dash' }, name: `empirical VaR ${lvlEn(a)}`, hoverinfo: 'skip', xaxis: 'x2' });
  plot('vm-chart', tr, layout({ barmode: 'group', bargap: 0.2, xaxis: AX(''), xaxis2: { overlaying: 'x', range: [0, 1], visible: false }, yaxis: AX(`Model VaR ${lvlEn(a)} / empirical VaR ${lvlEn(a)}`, { range: [0.6, null] }),
    shapes: [hln(1, '#000', A0, 'dash', 1)], legend: LEG(nw, -0.12), margin: { l: 60, r: 12, t: 14, b: 30 } }));
  tr.pop();
  const rg = m => { const r = D.names.map(n => mVar(n, m, a) / empVar(ser(n), a)); return [Math.min(...r), Math.max(...r)]; };
  out('vm-out', [T.vm(lvl(a)), ...mods.map(m => { const [lo, hi] = rg(m); return T.vmItem(MN(m), nf(100 * (lo - 1), 0), nf(100 * (hi - 1), 0)); })]);
}

// ================================================================= 19. the left tail on log-log axes
const LT_X = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 30];
function leftTailBlock() {
  const k = $('lt-s').value, xv = LT_X[+$('lt-x').value], S = ser(k), n = S.n, Lt = D.lt[k], mods = ['Normal', 'Student-t', 'Skewed-t', 'NIG', 'Stable'], nw = single('lt-chart', 400);
  label('lt-x').textContent = xv + '%';
  const L = Array.from(S.asc, v => -v), px = [], py = [];
  for (let i = 0; i < n && L[i] > 0.5; i++) { px.push(L[i]); py.push((i + 1) / n); }
  const tr = [{ x: px, y: py, mode: 'markers', marker: { size: 4, color: '#000' }, name: 'data', hovertemplate: 'loss %{x:.2f}%<br>share %{y:.2e}<extra></extra>' }];
  mods.forEach(m => tr.push({ x: Lt.grid, y: Lt.m[m], mode: 'lines', line: { color: MC[m], width: 1.6 }, name: MEN(m), hovertemplate: `${MEN(m)}<br>x = %{x:.2f}%<br>P = %{y:.2e}<extra></extra>` }));
  const xr = [Math.log10(0.5), Math.log10(Lt.grid[Lt.grid.length - 1])];
  const P = m => { const g = Lt.grid; let j = g.findIndex(v => v >= xv); if (j <= 0) return Lt.m[m][Math.max(j, 0)];
    const t = (Math.log(xv) - Math.log(g[j - 1])) / (Math.log(g[j]) - Math.log(g[j - 1])); return Math.exp(Math.log(Lt.m[m][j - 1]) + t * (Math.log(Lt.m[m][j]) - Math.log(Lt.m[m][j - 1]))); };
  const ix = px.map((v, i) => i).filter(i => px[i] > xv);      // the observed days with a loss above x, and each model's P(loss > x)
  tr.push({ x: ix.map(i => px[i]), y: ix.map(i => py[i]), mode: 'markers', marker: { size: 7, color: RED, line: { color: '#000', width: 0.6 } }, name: `data, loss > ${xv}% (${ix.length} day${ix.length === 1 ? '' : 's'})`, hovertemplate: 'loss %{x:.2f}%<br>share %{y:.2e}<extra></extra>' },
    { x: mods.map(() => xv), y: mods.map(P), mode: 'markers', marker: { size: 10, symbol: 'diamond', color: mods.map(m => MC[m]), line: { color: DARK, width: 1 } }, name: `models at x = ${xv}%`, customdata: mods.map(MEN), hovertemplate: '%{customdata}<br>P(loss > x) = %{y:.2e}<extra></extra>' });
  plot('lt-chart', tr, layout({ xaxis: AX('Daily loss x (%, log scale)', Object.assign(pctTicks([0.5, 1, 2, 5, 10, 20, 50, 100]), { range: xr })), yaxis: AX('P(loss > x) (log scale)', LOGP([Math.log10(0.5 / n), Math.log10(0.6)])),
    shapes: [hln(0.01, GREEN, A0, 'dot', 1), vln(xv, DARK, A0, 'dot', 1.4)], annotations: [{ x: Math.log10(0.6), y: Math.log10(0.0125), xref: 'x', yref: 'y', text: '1% (VaR 1%)', showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { size: 11, color: GREEN } }],
    legend: LEG(nw, -0.16), margin: { l: 64, r: 12, t: 14, b: 40 } }));
  const cnt = L.filter(v => v > xv).length;
  const fp0 = FIT(k, 'Normal').params, PP = m => (m === 'Normal' ? Phi((-xv - fp0.mu) / fp0.sigma) : P(m));     // the chart clips model probabilities at 1e-9
  out('lt-out', [T.lt(SH[k], String(xv), cnt, fint(n), sig(cnt / n, 2)), T.ltItem(mods.map(m => { const q = PP(m), low = m !== 'Normal' && q <= 1.0001e-9;
    return [MN(m), low ? '&lt; 10<sup>−9</sup>' : sig(q, 2), low ? '0,0'.replace(',', LANG === 'ro' ? ',' : '.') : fnum(q * n, q * n < 10 ? 1 : 0)]; }))]);
}

// ================================================================= 20. Kupiec test (table, worked example, exceedance chart)
function kupiecBlock() {
  const a = LV[+$('ku-a').value], k = $('ku-k').value, m = $('ku-m').value, mods = VM6;
  table('ku-tab', ['', 'n', T.h.expct, ...mods.map(MN)], D.names.map(n => { const S = ser(n); return Object.assign([SH[n], fint(S.n), nf(a * S.n, 1), ...mods.map(md => {
    const K = kupiec(countBelow(S.x, -mVar(n, md, a)), S.n, a); return { v: `${fint(K.x)}<span class="pv"> (${fp(K.p)})</span>`, c: K.p < 0.05 ? 'bad' : '' }; })], { cls: n === k ? 'sel' : '' }); }));
  const S = ser(k), v = mVar(k, m, a), K = kupiec(countBelow(S.x, -v), S.n, a);
  steps('ku-we', [T.ku1(MN(m), SH[k], lvl(a), nf(v, 3), fint(S.n), fint(K.x), nf(100 * K.rate, 3)),
    T.ku2(fint(S.n - K.x), nf(a, 4), fint(K.x), fnum(K.l0, 2)), T.ku3(nf(K.rate, 5), fnum(K.l1, 2)),
    T.ku4(fnum(K.l0, 2), fnum(K.l1, 2), nf(K.lr, 3), fpe(K.p), K.p < 0.05)]);
  const Pn = panels('ku-chart', 2, { stack: true, hs: 230, extra: 120, vgap: 0.18 }), [p, q] = Pn.ax, ex = [], cx = [], cy = [];
  let c = 0; S.x.forEach((r, i) => { if (r < -v) { ex.push(i); c++; cx.push(S.dates[i]); cy.push(c); } });
  const tr = [on2(p, { x: S.dates, y: S.x, mode: 'lines', line: { color: SC[k], width: 0.6 }, name: `${LAB[k]} daily log return (%)`, hoverinfo: 'skip' }),
    on2(p, { x: [S.dates[0], S.dates[S.n - 1]], y: [-v, -v], mode: 'lines', line: { color: MC[m], width: 1.5 }, name: `-VaR ${lvlEn(a)}, ${MEN(m)} (in sample)`, hoverinfo: 'skip' }),
    on2(p, { x: ex.map(i => S.dates[i]), y: ex.map(i => S.x[i]), mode: 'markers', marker: { size: 5, symbol: 'triangle-down', color: '#000' }, name: 'exceedance', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(q, { x: [S.dates[0], ...cx, S.dates[S.n - 1]], y: [0, ...cy, c], mode: 'lines', line: { color: MC[m], width: 1.6, shape: 'hv' }, name: 'cumulative exceedances', hovertemplate: '%{x}<br>%{y}<extra></extra>' }),
    on2(q, { x: [S.dates[0], S.dates[S.n - 1]], y: [0, a * S.n], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: `expected: ${lvlEn(a)} of the days`, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX(''), AX('Return (%)')); setAx(Pn, 1, AX(''), AX('Exceedances'));
  plot('ku-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(p, 'Daily returns and the in-sample VaR line'), ptitle(q, 'Cumulative number of exceedances')], legend: LEG(narrow('ku-chart'), -0.08) })));
  const byY = {}; ex.forEach(i => { const y = yr(S.dates[i]); byY[y] = (byY[y] || 0) + 1; }); const yb = Object.keys(byY).reduce((u, w) => (byY[w] > byY[u] ? w : u));
  out('ku-out', [T.ku(MN(m), SH[k], fint(K.x), nf(a * S.n, 1), yb, byY[yb])]);
}

// ================================================================= 21. out of sample, 2020-2026
const OOS_M = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG', 'Stable'];
function oosTest(k) {                // the chapter's rule: test = every day after the split (31 Dec 2019)
  const S = ser(k), i0 = S.dates.findIndex(d => d > D.split); return { S, i0, test: S.x.slice(i0), nEst: S.dates.filter(d => d <= D.split).length };
}
function pinball(r, q, a) { let s = 0; for (const v of r) s += (a - (v < q ? 1 : 0)) * (v - q); return s / r.length; }
function oosBlock() {
  const met = $('oo-r').value, ks = D.assets, O = NUM.oos, Pn = panels('oo-chart', 2, { h: 380, hs: 270, extra: 150 }), [a, b] = Pn.ax, tr = [];
  const xs = ks.map(k => LAB[k]);
  OOS_M.forEach(m => {
    tr.push(on2(a, { type: 'bar', x: xs, y: ks.map(k => O[k][m].logscore - O[k].Normal.logscore), marker: { color: MC[m] }, name: MEN(m), legendgroup: m, offsetgroup: m, hovertemplate: `${MEN(m)}<br>%{x}: %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { type: 'bar', x: xs, y: ks.map(k => (met === 'rate' ? 100 * O[k][m].kupiec.rate : met === 'pin' ? O[k][m].pinball : O[k][m].kupiec.lr)), marker: { color: MC[m] }, showlegend: false, legendgroup: m, offsetgroup: m,
      hovertemplate: `${MEN(m)}<br>%{x}: %{y:.3f}<extra></extra>` }));
  });
  if (met === 'pin') tr.push(on2(b, { type: 'bar', x: xs, y: ks.map(k => O[k].Historical.pinball), marker: { color: MC.Historical, pattern: { shape: '/' } }, name: 'historical simulation', offsetgroup: 'Historical', hovertemplate: 'historical<br>%{x}: %{y:.4f}<extra></extra>' }));
  // the series and the method of the worked example: a marker on that bar in both panels
  const k0 = $('oo-s').value, m0 = $('oo-m').value, H = m0 === 'Historical', rv = mm => (met === 'rate' ? 100 * O[k0][mm].kupiec.rate : met === 'pin' ? O[k0][mm].pinball : O[k0][mm].kupiec.lr);
  const mk = (ax, y, own) => on2(ax, Object.assign({ x: [LAB[k0]], y: [y], mode: 'markers+text', text: [H ? 'historical' : MEN(m0)], textposition: 'top center', textfont: { size: 11, color: DARK },
    marker: { size: 13, symbol: 'triangle-down', color: DARK }, showlegend: false, cliponaxis: false, hoverinfo: 'skip' }, own ? { offsetgroup: m0 } : {}));
  tr.push(mk(a, H ? Math.max(...OOS_M.map(mm => O[k0][mm].logscore - O[k0].Normal.logscore)) : O[k0][m0].logscore - O[k0].Normal.logscore, !H), mk(b, rv(m0), !H || met === 'pin'));
  const yb = met === 'rate' ? 'VaR 1% exceedances on the test days (%)' : met === 'pin' ? 'Pinball loss of the 1% quantile (lower = better)' : 'Kupiec LR statistic';
  setAx(Pn, 0, AX(''), AX('')); setAx(Pn, 1, AX(''), AX(''));
  const shp = met === 'rate' ? [hln(1, '#000', b, 'dash', 1)] : met === 'kup' ? [hln(3.84, '#000', b, 'dash', 1)] : [];
  plot('oo-chart', tr, layout(Object.assign(Pn.L, { barmode: 'group', scattermode: 'group', bargap: 0.18, shapes: shp, annotations: [ptitle(a, 'Out-of-sample log score minus Normal'), ptitle(b, yb)], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  const cm = ['Normal', 'Student-t', 'Skewed-t', 'NIG', 'Stable', 'Historical'];
  table('oo-tab', ['', T.h.expct, ...cm.map(m => (m === 'Historical' ? T.h.hist : MN(m))), T.h.bestAIC, T.h.bestLS, T.h.bestPin], ks.map(k => {
    const t = oosTest(k);
    return [SH[k], nf(0.01 * t.test.length, 1), ...cm.map(m => { const K = O[k][m].kupiec; return { v: `${K.x}<span class="pv"> (${fp(K.p)})</span>`, c: K.p < 0.05 ? 'bad' : '' }; }), MN(O[k].best_aic_est), MN(O[k].best_logscore),
      O[k].best_pinball === 'Historical' ? T.h.hist : MN(O[k].best_pinball)];
  }));
  const k = $('oo-s').value, m = $('oo-m').value, t = oosTest(k), q = -(m === 'Historical' ? O[k].Historical.var1 : O[k][m].var1), x = countBelow(t.test, q);
  let s1 = 0, s0 = 0; for (const v of t.test) { if (v < q) s1 += (0.01 - 1) * (v - q); else s0 += 0.01 * (v - q); }
  const pin = pinball(t.test, q, 0.01);
  steps('oo-we', [T.oo1(SH[k], m === 'Historical' ? T.h.hist : MN(m), fint(t.nEst), nf(-q, 3), fint(t.test.length), t.S.dates[t.i0]), T.oo2(fint(x), nf(s1, 2)), T.oo3(fint(t.test.length - x), nf(s0, 2)),
    T.oo4(nf(s1, 2), nf(s0, 2), fint(t.test.length), nf(pin, 4))]);
  out('oo-out', [T.oo(ks.filter(k2 => O[k2].best_aic_est === O[k2].best_logscore).map(k2 => SH[k2]))]);
}

// ================================================================= 22. overfitting with Normal mixtures
function overfitBlock() {
  const pen = +$('of-p').value, c0 = +$('of-c').value, R = NUM.overfit.res, cs = [1, 2, 3, 4, 5, 6], n = NUM.overfit.n_in, ln = Math.log(n);
  label('of-p').textContent = nf(pen, 1);
  const crit = c => -2 * R[c].loglik + pen * R[c].k, chosen = cs.reduce((a, c) => (crit(c) < crit(a) ? c : a), 1);
  const Pn = panels('of-chart', 2, { h: 380, hs: 270, extra: 150 }), [a, b] = Pn.ax;
  const dl = f => { const v = cs.map(f), m = Math.min(...v); return v.map(x => x - m); };
  const tr = [on2(a, { x: cs, y: cs.map(c => R[c].in), mode: 'lines+markers', marker: { symbol: 'circle', size: 7 }, line: { color: BLUE }, name: 'in sample (estimation years)', hovertemplate: 'c = %{x}<br>%{y:.3f}<extra></extra>' }),
    on2(a, { x: cs, y: cs.map(c => R[c].out), mode: 'lines+markers', marker: { symbol: 'square', size: 7 }, line: { color: RED }, name: 'out of sample (later years)', hovertemplate: 'c = %{x}<br>%{y:.3f}<extra></extra>' }),
    on2(a, { x: [chosen, chosen], y: [R[chosen].in, R[chosen].out], mode: 'lines+markers+text', text: [`in ${R[chosen].in.toFixed(3)}`, `out ${R[chosen].out.toFixed(3)}`], textposition: 'middle right',
      textfont: { size: 11, color: ORANGE }, line: { color: ORANGE, width: 2.5 }, marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: ORANGE, width: 2 } }, name: `selected c = ${chosen}: gap in - out = ${(R[chosen].in - R[chosen].out).toFixed(3)}`, hoverinfo: 'skip' }),
    on2(b, { x: cs, y: dl(c => R[c].aic), mode: 'lines+markers', marker: { symbol: 'circle', size: 7 }, line: { color: GREEN }, name: 'Delta AIC', hovertemplate: 'c = %{x}<br>%{y:.1f}<extra></extra>' }),
    on2(b, { x: cs, y: dl(c => R[c].bic), mode: 'lines+markers', marker: { symbol: 'square', size: 7 }, line: { color: PURPLE }, name: 'Delta BIC', hovertemplate: 'c = %{x}<br>%{y:.1f}<extra></extra>' }),
    on2(b, { x: cs, y: dl(crit), mode: 'lines+markers', marker: { symbol: 'diamond', size: 8 }, line: { color: ORANGE, dash: 'dot' }, name: `penalty ${pen.toFixed(1)} per parameter`, hovertemplate: 'c = %{x}<br>%{y:.1f}<extra></extra>' }),
    on2(b, { x: [chosen], y: [0], mode: 'markers', marker: { size: 14, color: 'rgba(0,0,0,0)', line: { color: ORANGE, width: 2 } }, name: `selected with this penalty: c = ${chosen}`, hoverinfo: 'skip' }),
    on2(a, { x: [c0, c0], y: [R[c0].in, R[c0].out], mode: 'markers', marker: { symbol: 'star', size: 13, color: C.navy, line: { color: '#fff', width: 1 } }, name: `worked example: c = ${c0}`, legendgroup: 'we', hoverinfo: 'skip' }),
    on2(b, { x: [c0, c0], y: [dl(c => R[c].aic)[c0 - 1], dl(c => R[c].bic)[c0 - 1]], mode: 'markers', marker: { symbol: 'star', size: 13, color: C.navy, line: { color: '#fff', width: 1 } }, name: `worked example: c = ${c0}`, legendgroup: 'we', showlegend: false, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('Number of Normal components', { dtick: 1 }), AX('')); setAx(Pn, 1, AX('Number of Normal components', { dtick: 1 }), AX(''));
  plot('of-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'Mean log-likelihood per day'), ptitle(b, 'Information criteria (0 = selected)')], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  table('of-tab', ['c', 'k', 'ℓ(θ̂)', 'AIC', 'BIC', T.h.pcrit, T.h.inS, T.h.outS], cs.map(c => Object.assign([String(c), String(R[c].k), fnum(R[c].loglik, 1), fnum(R[c].aic, 1), fnum(R[c].bic, 1),
    { v: fnum(crit(c), 1), c: c === chosen ? 'hi' : '' }, nf(R[c].in, 3), { v: nf(R[c].out, 3), c: c === 3 ? 'hi' : '' }], { cls: c === c0 ? 'sel' : '' })));
  const r = R[c0];
  steps('of-we', [T.of1(c0, r.k, fnum(r.loglik, 2), fint(n)), T.of2(fnum(r.loglik, 2), r.k, fnum(r.aic, 2)), T.of3(fnum(r.loglik, 2), r.k, nf(ln, 3), fnum(r.bic, 2)),
    T.of4(nf(r.in, 3), nf(r.out, 3), nf(r.in - r.out, 3))]);
  out('of-out', [T.of(nf(pen, 1), chosen, nf(ln, 2))]);
}

// ================================================================= 23. rolling VaR and its backtest
const RV_M = ['Normal', 'Student-t', 'Skewed-t', 'Historical'];
function rolling(k) {
  const S = ser(k), Rl = D.roll[k], n = S.n - Rl.i0, V = {}, I = {};
  RV_M.forEach(m => { V[m] = new Float64Array(n); I[m] = new Uint8Array(n); for (let j = 0; j < n; j++) { V[m][j] = Rl.m[m][Math.floor(j / Rl.step)]; I[m][j] = S.x[Rl.i0 + j] < V[m][j] ? 1 : 0; } });
  return { S, i0: Rl.i0, n, V, I, dates: S.dates.slice(Rl.i0), r: S.x.slice(Rl.i0) };
}
function rollingBlock() {
  const k = $('rv-s').value, from = $('rv-f').value, m0 = $('rv-m').value, R = rolling(k), j0 = Math.max(R.dates.findIndex(d => d >= from), 0), nw = narrow('rv-chart');
  const ds = R.dates.slice(j0), rs = R.r.slice(j0);
  $('rv-chart').style.height = (nw ? 540 : 400) + 'px';
  const tr = [{ x: ds, y: rs, mode: 'lines', line: { color: TEAL, width: 0.6 }, name: `${LAB[k]} daily log return (%)`, hoverinfo: 'skip' }];
  [['Normal', MC.Normal], ['Student-t', MC['Student-t']], ['Historical', PURPLE], ['Skewed-t', MC['Skewed-t']]].forEach(([m, c]) => {
    if (m === 'Skewed-t' && m0 !== 'Skewed-t') return;
    tr.push({ x: ds, y: Array.from(R.V[m].slice(j0)), mode: 'lines', line: { color: c, width: 1.3, shape: 'hv' }, name: m === 'Historical' ? 'VaR 1% line, historical simulation' : `VaR 1% line, ${m}`, hovertemplate: `${m}<br>%{x}<br>%{y:.2f}%<extra></extra>` });
  });
  const ex = []; for (let j = j0; j < R.n; j++) if (R.I[m0][j]) ex.push(j);
  tr.push({ x: ex.map(j => R.dates[j]), y: ex.map(j => R.r[j]), mode: 'markers', marker: { size: 5, symbol: 'triangle-down', color: '#000' }, name: `exceedance of the ${m0 === 'Historical' ? 'historical' : m0} VaR 1%`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' });
  plot('rv-chart', tr, layout({ xaxis: AX(''), yaxis: AX('%'), legend: LEG(nw, -0.1), margin: { l: 50, r: 12, t: 14, b: 30 } }));
  const years = [...new Set(R.dates.map(yr))], cnt = {}, nd = {};
  R.dates.forEach((d, j) => { const y = yr(d); nd[y] = (nd[y] || 0) + 1; cnt[y] = (cnt[y] || 0) + R.I[m0][j]; });
  const nwb = single('rv-bar', 280);
  plot('rv-bar', [{ type: 'bar', x: years, y: years.map(y => cnt[y]), marker: { color: m0 === 'Historical' ? PURPLE : MC[m0] }, name: `exceedances per year, ${m0 === 'Historical' ? 'historical simulation' : m0}`, hovertemplate: '%{x}: %{y}<extra></extra>' },
    { x: years, y: years.map(y => 0.01 * nd[y]), mode: 'lines+markers', line: { color: DARK, dash: 'dash', width: 1.2 }, marker: { size: 4 }, name: 'expected: 1% of the trading days', hovertemplate: '%{x}: %{y:.1f}<extra></extra>' }],
  layout({ xaxis: AX('', { type: 'category', tickangle: nwb ? -60 : -45 }), yaxis: AX('Exceedances'), legend: LEG(nwb, -0.3), margin: { l: 50, r: 12, t: 10, b: 50 } }));
  table('rv-tab', ['', T.h.rate, 'Kupiec LR', 'Kupiec p', T.h.maxY, 'LR<sub>ind</sub>', T.h.pind], RV_M.map(m => {
    const x = R.I[m].reduce((s, v) => s + v, 0), K = kupiec(x, R.n, 0.01), Ch = christoffersen(R.I[m]), by = {};
    R.dates.forEach((d, j) => { if (R.I[m][j]) by[yr(d)] = (by[yr(d)] || 0) + 1; }); const ym = Object.keys(by).reduce((u, w) => (by[w] > by[u] ? w : u));
    return Object.assign([m === 'Historical' ? T.h.hist : MN(m), nf(100 * K.rate, 2) + '%', nf(K.lr, 2), { v: fp(K.p), c: K.p < 0.05 ? 'bad' : '' }, `${by[ym]} (${ym})`, nf(Ch.lr, 2), { v: fp(Ch.p), c: Ch.p < 0.05 ? 'bad' : '' }], { cls: m === m0 ? 'sel' : '' });
  }));
  const Ch = christoffersen(R.I[m0]);
  steps('rv-we', [T.rv1(SH[k], m0 === 'Historical' ? T.h.hist : MN(m0), fint(R.n), fint(Ch.n00), fint(Ch.n01), fint(Ch.n10), fint(Ch.n11)),
    T.rv2(fint(Ch.n01), fint(Ch.n00 + Ch.n01), nf(Ch.p01, 4), fint(Ch.n11), fint(Ch.n10 + Ch.n11), nf(Ch.p11, 3), nf(Ch.pi, 4)),
    T.rv3(fnum(Ch.l0, 2), fnum(Ch.l1, 2), nf(Ch.lr, 2), fpe(Ch.p), Ch.p < 0.05)]);
  out('rv-out', [T.rv(SH[k], R.dates[0], fint(R.n), from)]);
}

// ================================================================= 24. the VaR reliability plot (SFEVaRqqplot)
function varRmaEma(y, h, lam) {
  const n = y.length, c = new Float64Array(n + 1); for (let i = 0; i < n; i++) c[i + 1] = c[i] + y[i] * y[i];
  const z = normPpf(0.01), w = Array.from({ length: h }, (_, t) => lam ** (h - 1 - t)), rma = [], ema = [];
  for (let j = h; j < n; j++) { rma.push(z * Math.sqrt((c[j] - c[j - h]) / h)); let s = 0; for (let t = 0; t < h; t++) s += w[t] * y[j - h + t] * y[j - h + t]; ema.push(z * Math.sqrt((1 - lam) * s)); }
  return { rma, ema };
}
function varQQBlock() {
  const lam = +$('vq-l').value, h = +$('vq-h').value; label('vq-l').textContent = nf(lam, 2);
  const S = ser('dax'), y = S.x.map(v => v / 100), V = varRmaEma(y, h, lam), p = y.slice(h), N = p.length;
  const Pn = panels('vq-chart', 2, { h: 400, hs: 290, extra: 150 }), tr = [], ann = [], rows = [];
  const sel = []; for (let i = 0; i < N; i += Math.max(1, Math.floor(N / 700))) sel.push(i); for (let i = Math.max(0, N - 40); i < N; i++) sel.push(i); for (let i = 0; i < 40; i++) sel.push(i);
  const S2 = [...new Set(sel)].sort((s, t) => s - t), qn = S2.map(i => normPpf((i + 0.5) / N));
  const RAT = { RMA: p.map((x, i) => x / V.rma[i]).sort((s, t) => s - t), EMA: p.map((x, i) => x / V.ema[i]).sort((s, t) => s - t) };
  [['RMA', V.rma, BLUE, `RMA, h = ${h}`], ['EMA', V.ema, ORANGE, `EMA, lambda = ${lam.toFixed(2)}`]].forEach(([nm, v, col, tt], j) => {
    const ratio = RAT[nm], ax = Pn.ax[j], oth = nm === 'RMA' ? ['EMA', `EMA, lambda = ${lam.toFixed(2)}`] : ['RMA', `RMA, h = ${h}`];
    // the other method in grey, for comparison: the RMA panel thus also follows lambda
    tr.push(on2(ax, { x: qn, y: S2.map(i => RAT[oth[0]][i]), mode: 'markers', marker: { size: 3, color: '#B0B0B0' }, name: `grey: the other panel (${oth[1]})`, showlegend: false, hovertemplate: `${oth[1]}<br>Normal quantile %{x:.2f}<br>L/VaR %{y:.2f}<extra></extra>` }));
    const b0 = quantile(ratio, 0.25), b1 = quantile(ratio, 0.75), qb = normPpf(0.75), sl = (b1 - b0) / (2 * qb);
    tr.push(on2(ax, { x: qn, y: S2.map(i => ratio[i]), mode: 'markers', marker: { size: 3, color: col }, name: `L/VaR, ${tt}`, hovertemplate: 'Normal quantile %{x:.2f}<br>L/VaR %{y:.2f}<extra></extra>' }));
    tr.push(on2(ax, { x: [qn[0], qn[qn.length - 1]], y: [b0 + sl * (qn[0] + qb), b0 + sl * (qn[qn.length - 1] + qb)], mode: 'lines', line: { color: '#000', width: 0.9, dash: 'dash' }, name: 'line through the quartiles', showlegend: j === 0, hoverinfo: 'skip' }));
    setAx(Pn, j, AX('Normal quantiles'), AX(j === 0 || Pn.stack ? 'L/VaR quantiles' : ''));
    ann.push(ptitle(ax, `${tt} (grey: ${oth[1]})`));
    const x = p.reduce((s, r, i) => s + (r < v[i] ? 1 : 0), 0), K = kupiec(x, N, 0.01);
    rows.push([nm, fint(x), nf(100 * K.rate, 2) + '%', nf(K.lr, 1), { v: fp(K.p), c: K.p < 0.05 ? 'bad' : '' }, nf(ratio[N - 1], 2)]);
  });
  plot('vq-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  table('vq-tab', ['', T.h.exc, T.h.rate, 'Kupiec LR', 'Kupiec p', T.h.maxR], rows);
  out('vq-out', [T.vq(fint(N), S.dates[h], h, nf(lam, 2), nf(Math.log(0.5) / Math.log(lam), 1))]);
}

// ================================================================= 25. stability of the winner (moving-block bootstrap)
const MU_M = ['Normal', 'Student-t', 'Skewed-t', 'GED', 'Mixture', 'NIG'];
function uncBlock() {
  const B = +$('mu-b').value, U = D.unc; label('mu-b').textContent = String(B);
  const win = {}; MU_M.forEach(m => (win[m] = 0));
  U.winner.slice(0, B).forEach(i => win[U.models[i]]++);
  const Pn = panels('mu-chart', 2, { h: 400, hs: 290, extra: 120 }), [a, b] = Pn.ax, emp = NUM.uncertainty.emp_var1;
  const tr = [on2(a, { type: 'bar', x: MU_M.map(MEN), y: MU_M.map(m => 100 * win[m] / B), marker: { color: MU_M.map(m => MC[m]) }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })];
  MU_M.forEach(m => tr.push(on2(b, { type: 'box', y: U.var[m].slice(0, B), name: MEN(m), marker: { color: RED, size: 4, outliercolor: RED }, line: { color: '#000', width: 1 }, fillcolor: hexA(MC[m], 0.6), boxpoints: 'outliers', showlegend: false, hovertemplate: '%{y:.2f}%<extra></extra>' })));
  tr.push(on2(b, { x: [MEN(MU_M[0]), MEN(MU_M[MU_M.length - 1])], y: [emp, emp], mode: 'lines', line: { color: RED, width: 1, dash: 'dash' }, name: `empirical VaR 1% = ${emp.toFixed(2)}%`, hoverinfo: 'skip' }));
  setAx(Pn, 0, AX('', { tickangle: -20 }), AX('% of resamples with the lowest AIC', { range: [0, 100] })); setAx(Pn, 1, AX('', { tickangle: -20 }), AX('VaR 1% (%)'));
  plot('mu-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, 'Which model wins?'), ptitle(b, 'Bootstrap spread of VaR 1%')], legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.25) })));
  const q = (arr, p) => quantile(Float64Array.from(arr).sort(), p);
  table('mu-tab', [T.h.model, T.h.wins, T.h.med, T.h.i90, T.h.width], MU_M.map(m => { const v = U.var[m].slice(0, B), lo = q(v, 0.05), hi = q(v, 0.95);
    return [MN(m), nf(100 * win[m] / B, 1) + '%', nf(q(v, 0.5), 2), `[${nf(lo, 2)}; ${nf(hi, 2)}]`, nf(hi - lo, 2)]; }));
  out('mu-out', [T.mu(B, nf(100 * win.NIG / B, 1), B === 200)]);
}
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;

// ================================================================= 26. model averaging and model risk (table + worked example)
function avgCalc(k, a, wkey, mix) {
  const I = icRows(k), w = I.R.map(r => (wkey === 'aic' ? r.w : r.wb)), vs = MOD.map(m => mVar(k, m, a));
  const avg = vs.reduce((s, v, i) => s + w[i] * v, 0), heavy = MOD.filter(m => m !== 'Normal' && (mix || m !== 'Mixture'));
  const hv = heavy.map(m => mVar(k, m, a)), lo = Math.min(...hv), hi = Math.max(...hv);
  return { w, vs, avg, lo, hi, mlo: heavy[hv.indexOf(lo)], mhi: heavy[hv.indexOf(hi)] };
}
function avgBlock() {
  const a = LV[+$('av-a').value], wkey = $('av-w').value, mix = $('av-x').value === 'mix', k = $('av-s').value;
  table('av-tab', ['', T.h.data, MN('Normal'), T.h.rangeH, T.h.avg, T.h.rangeP], D.names.map(n => { const R = avgCalc(n, a, wkey, mix);
    return Object.assign([SH[n], nf(empVar(ser(n), a), 2), nf(mVar(n, 'Normal', a), 2), `[${nf(R.lo, 2)}; ${nf(R.hi, 2)}]<span class="pv"> (${MN(R.mlo)}, ${MN(R.mhi)})</span>`, nf(R.avg, 2), nf(100 * (R.hi / R.lo - 1), 0)], { cls: n === k ? 'sel' : '' }); }));
  const R = avgCalc(k, a, wkey, mix);
  steps('av-we', [T.av1(SH[k], wkey.toUpperCase(), MOD.map((m, i) => [MN(m), sig(R.w[i], 3), nf(R.vs[i], 2)]).filter(x => true)),
    T.av2(MOD.map((m, i) => (R.w[i] > 5e-4 ? `${sig(R.w[i], 3)} × ${nf(R.vs[i], 2)}` : null)).filter(Boolean).join(' + '), nf(R.avg, 2), lvl(a)),
    T.av3(nf(R.hi, 2), MN(R.mhi), nf(R.lo, 2), MN(R.mlo), nf(100 * (R.hi / R.lo - 1), 0))]);
  out('av-out', [T.av(lvl(a), wkey.toUpperCase(), mix)]);
}

// ================================================================= wiring
const PFX = { candidates: 'ca', 'dax-monthly': 'dm', 'desc-stats': 'ds', 'ml-table': 'ml', 'profile-nu': 'pn', 'lr-test': 'lr', vuong: 'vu', 'aic-table': 'ai', 'delta-aic': 'da',
  'ks-five': 'k5', 'ks-tail': 'kt', 'gof-power': 'gp', 'edf-table': 'ed', pp: 'pp', qq: 'qq', 'qq-assets': 'qa', 'var-es': 've', 'var-models': 'vm', 'left-tail': 'lt', kupiec: 'ku',
  oos: 'oo', overfit: 'of', 'rolling-var': 'rv', 'var-qqplot': 'vq', 'model-uncertainty': 'mu', 'model-avg': 'av' };
const BLOCKS = { candBlock, daxMonthlyBlock, descBlock, mlBlock, profileBlock, lrBlock, vuBlock, aicBlock, deltaBlock, ksFiveBlock, ksTailBlock, powerBlock, edfBlock, ppBlock, qqBlock,
  qqAssetsBlock, varEsBlock, varModelsBlock, leftTailBlock, kupiecBlock, oosBlock, overfitBlock, rollingBlock, varQQBlock, uncBlock, avgBlock };
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
  const M = D.dax_m, dmv = []; for (let i = M.book[1]; i < M.ym.length; i += 12) dmv.push(i); if (dmv[dmv.length - 1] !== M.ym.length - 1) dmv.push(M.ym.length - 1);
  on(['ca-x'], candBlock); PL('ca-play', 'ca-x', range(0, 8, 1), 900);
  on(['dm-n'], daxMonthlyBlock); PL('dm-play', 'dm-n', dmv, 700);
  on(['ds-b', 'ds-s'], descBlock); PL('ds-play', 'ds-b', ['clean', 'raw'], 1800);
  on(['ml-s'], mlBlock); PL('ml-play', 'ml-s', D.names, 1600);
  on(['pn-s', 'pn-c', 'pn-y', 'pn-nu'], profileBlock); PL('pn-play', 'pn-nu', range(2.2, 6, 0.1), 220);
  on(['lr-p', 'lr-s'], lrBlock); PL('lr-play', 'lr-s', D.names, 1400);
  on(['vu-p', 'vu-s'], vuBlock); PL('vu-play', 'vu-s', D.names, 1500);
  on(['ai-s', 'ai-m'], aicBlock); PL('ai-play', 'ai-s', D.names, 1500);
  on(['da-c'], deltaBlock); PL('da-play', 'da-c', ['aic', 'bic'], 1800);
  on(['k5-s', 'k5-h'], ksFiveBlock); PL('k5-play', 'k5-s', range(0.5, 2, 0.1), 350);
  on(['kt-s'], ksTailBlock); PL('kt-play', 'kt-s', D.names, 1600);
  on(['gp-n'], powerBlock); PL('gp-play', 'gp-n', range(0, 4, 1), 1000);
  on(['ed-s'], edfBlock); PL('ed-play', 'ed-s', D.names, 1500);
  on(['pp-s', 'pp-4'], ppBlock); PL('pp-play', 'pp-s', D.names, 1600);
  on(['qq-s', 'qq-4'], qqBlock); PL('qq-play', 'qq-s', D.names, 1600);
  on(['qa-k'], qqAssetsBlock); PL('qa-play', 'qa-k', range(1, 10, 1), 800);
  on(['ve-a', 've-e', 've-k'], varEsBlock); PL('ve-play', 've-a', range(0, 5, 1), 1300);
  on(['vm-a', 'vm-x'], varModelsBlock); PL('vm-play', 'vm-a', range(0, 5, 1), 1300);
  on(['lt-s', 'lt-x'], leftTailBlock); PL('lt-play', 'lt-x', range(0, 10, 1), 800);
  on(['ku-a', 'ku-k', 'ku-m'], kupiecBlock); PL('ku-play', 'ku-a', range(0, 5, 1), 1400);
  on(['oo-r', 'oo-s', 'oo-m'], oosBlock); PL('oo-play', 'oo-s', D.assets, 1500);
  on(['of-p', 'of-c'], overfitBlock); PL('of-play', 'of-p', range(0, 10, 0.2), 180);
  on(['rv-s', 'rv-f', 'rv-m'], rollingBlock); PL('rv-play', 'rv-m', RV_M, 1600);
  on(['vq-l', 'vq-h'], varQQBlock); PL('vq-play', 'vq-l', range(0.9, 0.99, 0.01), 700);
  on(['mu-b'], uncBlock); PL('mu-play', 'mu-b', range(20, 200, 20), 600);
  on(['av-a', 'av-w', 'av-x', 'av-s'], avgBlock); PL('av-play', 'av-a', range(0, 5, 1), 1300);
  const VEIL = T.veil || {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH6 = { ser, mom, empVar, empEs, kupiec, christoffersen, icRows, lrCalc, ksFive, rolling, varRmaEma, avgCalc, oosTest, pinball, normPpf, Phi, chi2sf, chi2ppf, quantile, init, PFX, blocks: BLOCKS };
if (window.CH6_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
