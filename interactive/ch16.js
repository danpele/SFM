// Chapter 16 interactive page: one block per chart slide, results table, extra chart and review problem of the lecture, computed in the browser from ch16_data.js.
(function () {
'use strict';
const D = window.CH16_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH16_TEXT;
const NUM = D.N, COL = D.colors, NAME = D.name, AS = D.assets, K = D.consts;
const DARK = C.navy, RED = '#CD0000', BLUE = '#1A3A6E', AMBER = '#B5853F', GREEN = '#2E7D32', ORANGE = '#E67E22', PURPLE = '#6A1B9A', TEAL = '#17A2B8';

// ================================================================= numerics
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = LG[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function erfc(x) {                    // series without cancellation below 2.5, continued fraction above (relative error ~1e-14)
  if (x < 0) return 2 - erfc(-x);
  if (x < 2.5) { let term = x, s = x; for (let n = 1; n < 200; n++) { term *= 2 * x * x / (2 * n + 1); s += term; if (term < 1e-17 * s) break; } return 1 - 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * s; }
  let f = x, c = x, d = 0;
  for (let n = 1; n < 300; n++) { const an = n / 2; d = x + an * d; d = 1 / d; c = x + an / c; const del = c * d; f *= del; if (Math.abs(del - 1) < 1e-16) break; }
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
const phi = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function qnorm(p) {                   // Acklam, then two Newton steps
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  let x;
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else if (p > 1 - 0.02425) return -qnorm(1 - p);
  else { const q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  for (let i = 0; i < 2; i++) x -= (Phi(x) - p) / phi(x);
  return x;
}
function gammq(a, x) {                // regularised upper incomplete gamma Q(a, x)
  if (x <= 0) return 1;
  const gl = lgamma(a);
  if (x < a + 1) { let ap = a, s = 1 / a, del = s; for (let n = 0; n < 1000; n++) { ap++; del *= x / ap; s += del; if (Math.abs(del) < Math.abs(s) * 1e-16) break; } return 1 - s * Math.exp(-x + a * Math.log(x) - gl); }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-16) break; }
  return Math.exp(-x + a * Math.log(x) - gl) * h;
}
const chi2sf = (x, df) => gammq(df / 2, x / 2);
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
const sd1 = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / (a.length - 1)); };
const roundEven = v => { const f = Math.floor(v), r = v - f; return r > 0.5 ? f + 1 : r < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };

// ================================================================= data: closes on their own calendars (load_close)
const DAY = 864e5;
const isoOf = x => new Date(x).toISOString().slice(0, 10);
const msOf = s => Date.parse(s + 'T00:00:00Z');
const PX = {};
AS.forEach(k => { const s = D.px[k], b = msOf(s.t0); PX[k] = { d: s.t.map(t => isoOf(b + t * DAY)), p: s.v }; });
const WC = {};
function win(k, w) {                  // closes from w (load_close(k, w, END)) and their log returns in % (log_returns)
  const key = k + w; if (WC[key]) return WC[key];
  const P = PX[k], i0 = w ? P.d.findIndex(d => d >= w) : 0, d = P.d.slice(i0), p = P.p.slice(i0);
  const r = p.slice(1).map((v, i) => 100 * (Math.log(v) - Math.log(p[i])));
  return (WC[key] = { d, p, r, rd: d.slice(1) });
}
const yearsOf = (a, b) => (msOf(b) - msOf(a)) / DAY / 365.25;

// ================================================================= the chapter's measures (generate_all_charts.py)
function performance(k, w) {
  const W = win(k, w), p = W.p, r = W.r, R = p.slice(1).map((v, i) => 100 * (v / p[i] - 1));
  const q = r.length / yearsOf(W.rd[0], W.rd[r.length - 1]), years = yearsOf(W.d[0], W.d[W.d.length - 1]);
  const meanAr = q * mean(R), vol = Math.sqrt(q) * sd1(r);
  let mx = -Infinity, dd = 0, id = 0, pk = 0, ipk = 0;
  p.forEach((v, i) => { if (v > mx) { mx = v; ipk = i; } const z = v / mx - 1; if (z < dd) { dd = z; id = i; pk = ipk; } });
  const sh = meanAr / vol;
  return { n: r.length, q, first: W.rd[0], last: W.rd[r.length - 1], mean_log: q * mean(r), mean_arith: meanAr, vol, cagr: 100 * ((p[p.length - 1] / p[0]) ** (1 / years) - 1),
    sharpe: sh, sharpe_se: Math.sqrt((1 + sh * sh / 2) / years), mdd: 100 * dd, mdd_date: W.d[id], peak_date: W.d[pk], growth: p[p.length - 1] / p[0] };
}
function moments(x) {
  const m = mean(x); let m2 = 0, m3 = 0, m4 = 0;
  for (const v of x) { const e = v - m, e2 = e * e; m2 += e2; m3 += e2 * e; m4 += e2 * e2; }
  m2 /= x.length; m3 /= x.length; m4 /= x.length;
  const S = m3 / m2 ** 1.5, Kx = m4 / m2 ** 2 - 3, jb = x.length / 6 * (S * S + Kx * Kx / 4);
  return { skew: S, exkurt: Kx, jb };
}
function hill(x, k) { const s = Float64Array.from(x).sort().reverse(); let a = 0; for (let i = 0; i < k; i++) a += Math.log(s[i] / s[k]); return k / a; }
function tailIndex(r) { const L = r.map(v => -v), k = roundEven(K.hill * L.length), a = hill(L.filter(v => v > 0), k); return { k, alpha: a, se: a / Math.sqrt(k) }; }
function acf(x, nl) {
  const m = mean(x), e = x.map(v => v - m); let s = 0; for (const v of e) s += v * v;
  const o = []; for (let k = 1; k <= nl; k++) { let c = 0; for (let t = k; t < e.length; t++) c += e[t] * e[t - k]; o.push(c / s); }
  return o;
}
function ljungBox(x, m) { const T_ = x.length, rho = acf(x, m); let q = 0; rho.forEach((r, i) => { q += r * r / (T_ - i - 1); }); q *= T_ * (T_ + 2); return { q, p: chi2sf(q, m), rho1: rho[0] }; }
function varianceRatio(x, q) {
  const T_ = x.length, mu = mean(x), e = x.map(v => v - mu); let s2 = 0; for (const v of e) s2 += v * v;
  const varA = s2 / (T_ - 1); let acc = 0, run = 0;
  for (let t = 0; t < T_; t++) { run += x[t]; if (t >= q) run -= x[t - q]; if (t >= q - 1) { const a = run - q * mu; acc += a * a; } }
  const varC = acc / (q * (T_ - q + 1) * (1 - q / T_)), vr = varC / varA, z = (vr - 1) / Math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T_));
  let theta = 0; const den = s2 * s2;
  for (let j = 1; j < q; j++) { let c = 0; for (let t = j; t < T_; t++) c += e[t] * e[t] * e[t - j] * e[t - j]; theta += (2 * (q - j) / q) ** 2 * T_ * c / den; }
  const zs = (vr - 1) / Math.sqrt(theta / T_);
  return { vr, z, zs, p_zs: erfc(Math.abs(zs) / Math.SQRT2) };
}
const sortA = x => Float64Array.from(x).sort();
const hsVar = (x, a) => { const s = sortA(x); return -s[Math.ceil(s.length * a - 1e-9) - 1]; };
const hsEs = (x, a) => { const s = sortA(x), k = Math.ceil(s.length * a - 1e-9); let t = 0; for (let i = 0; i < k; i++) t += s[i]; return -t / k; };
const normalVar = (x, a) => -(mean(x) + sd1(x) * qnorm(a));
const normalEs = (x, a) => -mean(x) + sd1(x) * phi(qnorm(a)) / a;
function hurstRS(x, sizes) {
  const lx = [], ly = [];
  sizes.forEach(n => {
    const m = Math.floor(x.length / n); let acc = 0, cnt = 0;
    for (let b = 0; b < m; b++) {
      let mu = 0; for (let i = 0; i < n; i++) mu += x[b * n + i]; mu /= n;
      let y = 0, hi = -Infinity, lo = Infinity, ss = 0;
      for (let i = 0; i < n; i++) { const e = x[b * n + i] - mu; y += e; if (y > hi) hi = y; if (y < lo) lo = y; ss += e * e; }
      const S = Math.sqrt(ss / n); if (S > 0) { acc += (hi - lo) / S; cnt++; }
    }
    lx.push(Math.log10(n)); ly.push(Math.log10(acc / cnt));
  });
  const mx = mean(lx), my = mean(ly); let a = 0, b = 0; lx.forEach((v, i) => { a += (v - mx) * (ly[i] - my); b += (v - mx) ** 2; });
  return a / b;
}
const SC = {};
function summary(k, w) {
  const key = k + w; if (SC[key]) return SC[key];
  const r = win(k, w).r, ab = r.map(Math.abs), sz = D.sizes[k][w], o = { perf: performance(k, w), mom: moments(r), hill: tailIndex(r), lb_r: ljungBox(r, K.lb), lb_r2: ljungBox(r.map(v => v * v), K.lb),
    acf1_abs: acf(ab, 1)[0], vr: varianceRatio(r, K.vr), garch: D.garch[k][w], var_hs: hsVar(r, K.a_var), var_n: normalVar(r, K.a_var), es_hs: hsEs(r, K.a_es), es_n: normalEs(r, K.a_es),
    H_r: hurstRS(r, sz), H_abs: hurstRS(ab, sz) };
  o.gap_n = 100 * (1 - o.var_n / o.var_hs);
  return (SC[key] = o);
}

// one-day VaR 1% forecasts (var_backtest): historical simulation on the last w returns (rolling quantile, 'lower'), EWMA-Normal
const BC = {};
function backtest(k, w, lam, a) {
  a = a || K.a_var; const key = [k, w, lam, a].join('|'); if (BC[key]) return BC[key];
  const W = win(k, null), r = W.r, d = W.rd, n = r.length, s2 = new Float64Array(n);
  let m30 = 0; for (let i = 0; i < 30; i++) m30 += r[i]; m30 /= 30; let v30 = 0; for (let i = 0; i < 30; i++) v30 += (r[i] - m30) ** 2; s2[0] = v30 / 30;
  for (let t = 1; t < n; t++) s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2;
  const z = qnorm(a), t0 = Math.max(d.findIndex(x => x >= K.oos), w), qi = Math.floor((w - 1) * a);
  const srt = Array.from(r.slice(t0 - w, t0)).sort((x, y) => x - y);
  const ins = v => { let lo = 0, hi = srt.length; while (lo < hi) { const m = (lo + hi) >> 1; if (srt[m] < v) lo = m + 1; else hi = m; } srt.splice(lo, 0, v); };
  const del = v => { let lo = 0, hi = srt.length; while (lo < hi) { const m = (lo + hi) >> 1; if (srt[m] < v) lo = m + 1; else hi = m; } srt.splice(lo, 1); };
  const o = { d: [], r: [], HS: [], EWMA: [] };
  for (let t = t0; t < n; t++) {
    if (t > t0) { del(r[t - w - 1]); ins(r[t - 1]); }
    o.d.push(d[t]); o.r.push(r[t]); o.HS.push(-srt[qi]); o.EWMA.push(-Math.sqrt(s2[t]) * z);
  }
  o.n = o.d.length; o.first = o.d[0];
  ['HS', 'EWMA'].forEach(m => {
    const hit = o.r.map((x, i) => x < -o[m][i] ? 1 : 0), x = sum(hit);
    let mx = 0, run = 0; for (let i = 0; i < hit.length; i++) { run += hit[i]; if (i >= 250) run -= hit[i - 250]; if (i >= 249 && run > mx) mx = run; }
    o[m + '_hit'] = hit; o[m + '_st'] = Object.assign(kupiec(x, o.n, a), { max250: mx, mean_var: mean(o[m]) });
  });
  return (BC[key] = o);
}
function kupiec(x, n, a) {
  const ph = x / n, l0 = (n - x) * Math.log(1 - a) + x * Math.log(a), l1 = (ph < 1 ? (n - x) * Math.log(1 - ph) : 0) + (x > 0 ? x * Math.log(ph) : 0), lr = -2 * (l0 - l1);
  return { x, n, rate: 100 * ph, lr, p: chi2sf(lr, 1), l0, l1 };
}
function runVar(k) { const r = win(k, null).r, d = win(k, null).rd, v = []; let s = 0, s2 = 0; r.forEach((x, i) => { s += x; s2 += x * x; v.push(s2 / (i + 1) - (s / (i + 1)) ** 2); }); return { d, v }; }

// ================================================================= formatting, tables, panels
const SEP = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, SEP).replace('-', '−');
const DT = s => T.date(s);
const fp = p => p < 0.001 ? (LANG === 'ro' ? '< 0,001' : '< 0.001') : nf(p, 3);
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}>${o.v}</${tag}>`; };
  $(id).innerHTML = `<table class="xt"><thead><tr>${head.map(x => cell(x, 'th')).join('')}</tr></thead><tbody>` + rows.map(r => `<tr>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.rows || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 260) * n + 120 : (o.h || 380)) + 'px';
  const g = stack ? (o.gs || 0.1) : (o.gap || 0.1), L = {}, ax = [];
  const ws = o.widths && !stack ? o.widths : Array.from({ length: n }, () => 1 / n);
  let x0 = 0;
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', wi = ws[i] * (1 - g * (n - 1)), dom = [x0, x0 + wi]; x0 += wi + g;
    const hgt = (1 - g * (n - 1)) / n, top = 1 - i * (hgt + g);
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [Math.max(0, top - hgt), top] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s, left: stack ? 0 : dom[0], top: stack ? top : 1 });
  }
  return { L, ax, stack };
}
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const kSel = id => $(id).value;
const M0 = { l: 56, r: 12, t: 14, b: 40 };
const M1 = { l: 56, r: 12, t: 26, b: 40 };
const htr = (x0, x1, y, color, name, dash, extra) => Object.assign({ x: [x0, x1], y: [y, y], mode: 'lines', line: { color, width: 1.2, dash: dash || 'dash' }, name, hoverinfo: 'skip' }, extra || {});
const BAR = (x, y, color, name, extra) => Object.assign({ type: 'bar', x, y, marker: { color }, name }, extra || {});
const ptitle = (P, i, text) => ({ text, xref: 'paper', yref: 'paper', x: P.ax[i].left, y: P.ax[i].top, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: DARK } });
const NMS = AS.map(k => NAME[k]);
const IX = a => a.map((_, i) => i);                                   // bars on a numeric axis, labelled by CX: reference lines can span the panel
const CX = (a, extra) => Object.assign({ tickvals: IX(a), ticktext: a, range: [-0.5, a.length - 0.5] }, extra || {});
const WY = w => w.slice(0, 4);
const isLec = w => w === D.start;
const sgn = (x, d) => (x > 0 ? '+' : '') + nf(x, d);
const tz = s => s.replace(/([.,]\d*?)0+$/, '$1').replace(/[.,]$/, '');      // 60,00 -> 60, 1,600 -> 1,6
const nz = (x, d) => tz(nf(x, d));

// ================================================================= 1. three series, one window
function tsBlock() {
  const w = kSel('ts-w'), P = panels('ts-chart', 2, { rows: true, h: 520, hs: 230, gs: 0.08 }), tr = [], rows = [];
  $('ts-chart').style.height = (narrow('ts-chart') ? 640 : 560) + 'px';
  P.L.yaxis.domain = [0.42, 1]; P.L.yaxis2.domain = [0, 0.34];
  AS.forEach(k => {
    const W = win(k, w), pf = performance(k, w); let mx = -Infinity;
    const dd = W.p.map(v => { mx = Math.max(mx, v); return 100 * (v / mx - 1); });
    tr.push(on2(P.ax[0], { x: W.d, y: W.p.map(v => 100 * v / W.p[0]), mode: 'lines', line: { color: COL[k], width: 1.2 }, name: NAME[k], legendgroup: k, hovertemplate: `${NAME[k]} %{x}: %{y:.1f}<extra></extra>` }));
    tr.push(on2(P.ax[1], { x: W.d, y: dd, mode: 'lines', line: { color: COL[k], width: 1 }, name: NAME[k], legendgroup: k, showlegend: false, hovertemplate: `${NAME[k]} %{x}: %{y:.1f}%<extra></extra>` }));
    rows.push([NAME[k], fint(100 * pf.growth), nf(pf.mdd, 1), DT(pf.peak_date), DT(pf.mdd_date)]);
  });
  setAx(P, 0, AX('', { matches: 'x2', showticklabels: false }), AX(`value of 100 invested on ${DT0(win('sp500', w).d[0])} (log scale)`, { type: 'log' }));
  setAx(P, 1, AX(''), AX('drawdown (%)'));
  plot('ts-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('ts-chart'), -0.06), margin: M0 })));
  out('ts-out', [T.ts(WY(w), rows, !isLec(w))]);
}
const DT0 = s => { const M = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']; return `${+s.slice(8, 10)} ${M[+s.slice(5, 7) - 1]} ${s.slice(0, 4)}`; };

// ================================================================= x1. the course in one table (table)
function smBlock() {
  const w = kSel('sm-w'), S = AS.map(k => summary(k, w)), P = panels('sm-chart', 2, { h: 380 }), tr = [], crit = NUM.summary.sp500.lb_r.crit;
  [['returns r', 'lb_r', BLUE], ['squared returns r²', 'lb_r2', RED]].forEach(([n, f, c]) => tr.push(on2(P.ax[0], BAR(IX(NMS), S.map(s => s[f].q), c, `Ljung-Box Q(10), ${n}`, { customdata: NMS, text: S.map(s => s[f].q.toFixed(0)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: `%{customdata}: Q(10) = %{y:.1f}<extra>${n}</extra>` }))));
  tr.push(on2(P.ax[0], htr(-0.5, 2.5, crit, DARK, `5% critical value χ²(10) = ${crit.toFixed(2)}`)));
  [['returns r', 'H_r', BLUE], ['absolute returns |r|', 'H_abs', RED]].forEach(([n, f, c]) => tr.push(on2(P.ax[1], BAR(IX(NMS), S.map(s => s[f]), c, `Hurst exponent (R/S), ${n}`, { customdata: NMS, text: S.map(s => s[f].toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: `%{customdata}: H = %{y:.3f}<extra>${n}</extra>` }))));
  tr.push(on2(P.ax[1], htr(-0.5, 2.5, 0.5, AMBER, 'H = 0.5 (no memory)')));
  setAx(P, 0, AX('', CX(NMS)), AX('Q(10) (log scale)', { type: 'log', range: [0.5, 4.3] })); setAx(P, 1, AX('', CX(NMS)), AX('Hurst exponent H', { range: [0, 1.05] }));
  plot('sm-chart', tr, layout(Object.assign(P.L, { barmode: 'group', legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: M0 })));
  const f = (x, d) => nf(x, d), R = T.h.sm;
  table('sm-tab', [T.h.smc, ...NMS], [
    [R[0], ...S.map(s => `${fint(s.perf.n)} (${s.perf.q.toFixed(0)})`)], [R[1], ...S.map(s => f(s.perf.mean_log, 1))], [R[2], ...S.map(s => f(s.perf.vol, 1))],
    [R[3], ...S.map(s => `${f(s.perf.sharpe, 2)} ± ${f(s.perf.sharpe_se, 2)}`)], [R[4], ...S.map(s => `${f(s.mom.skew, 2)}; ${f(s.mom.exkurt, 1)}`)], [R[5], ...S.map(s => f(s.hill.alpha, 2))],
    [R[6], ...S.map(s => `${s.lb_r.q.toFixed(0)}; ${s.lb_r2.q.toFixed(0)}`)], [R[7], ...S.map(s => `${f(s.vr.vr, 2)}; ${f(s.vr.zs, 2)}`)], [R[8], ...S.map(s => f(s.garch.pers, 3))],
    [R[9], ...S.map(s => `${f(s.var_hs, 2)}; ${f(s.var_n, 2)}`)], [R[10], ...S.map(s => f(s.es_hs, 2))], [R[11], ...S.map(s => `${f(s.H_r, 2)}; ${f(s.H_abs, 2)}`)]]);
  out('sm-out', [T.sm(WY(w), DT(S[0].perf.first), S.map(s => [fint(s.perf.n), s.lb_r.q.toFixed(0), s.lb_r2.q.toFixed(0), nf(s.lb_r2.q / s.lb_r.q, 0), nf(s.H_abs, 2)]), nf(crit, 2), !isLec(w))]);
}

// ================================================================= x2. interpreting the table (extra chart)
function ipBlock() {
  const a = +kSel('ip-a'), w = D.start, P = panels('ip-chart', 2, { h: 380 }), tr = [], rows = [];
  const V = AS.map(k => { const r = win(k, w).r, h = hsVar(r, a), n = normalVar(r, a); return { h, n, gap: 100 * (1 - n / h) }; });
  tr.push(on2(P.ax[0], BAR(IX(NMS), V.map(v => v.h), BLUE, `VaR ${LVE(a)}%, historical simulation`, { customdata: NMS, text: V.map(v => v.h.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: '%{customdata}: %{y:.2f}%<extra>historical</extra>' })));
  tr.push(on2(P.ax[0], BAR(IX(NMS), V.map(v => v.n), RED, `VaR ${LVE(a)}%, Normal distribution`, { customdata: NMS, text: V.map(v => v.n.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: '%{customdata}: %{y:.2f}%<extra>Normal</extra>' })));
  tr.push(on2(P.ax[1], BAR(IX(NMS), V.map(v => v.gap), AS.map(k => COL[k]), 'Normal VaR below the historical VaR (%)', { customdata: NMS,  text: V.map(v => v.gap.toFixed(1) + '%'), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{customdata}: %{y:.1f}% too low<extra></extra>' })));
  const gmax = Math.max(5, ...V.map(v => Math.abs(v.gap))) * 1.3, gmin = Math.min(0, ...V.map(v => v.gap)) * 1.3;
  setAx(P, 0, AX('', CX(NMS)), AX(`VaR ${LVE(a)}% of a daily return (%)`, { range: [0, Math.max(...V.map(v => Math.max(v.h, v.n))) * 1.18] }));
  setAx(P, 1, AX('', CX(NMS)), AX('1 − Normal VaR / historical VaR (%)', { range: [gmin, gmax] }));
  plot('ip-chart', tr, layout(Object.assign(P.L, { barmode: 'group', legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: M0 })));
  out('ip-out', [T.ip(LV(a), V.map(v => [nf(v.h, 2), nf(v.n, 2), nf(v.gap, 1)]), a !== K.a_var)]);
}
const LV = a => nf(100 * a, Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);
const LVE = a => (100 * a).toFixed(Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);

// ================================================================= 2. stylised facts in one chart
function sfBlock() {
  const w = kSel('sf-w'), L = +$('sf-l').value, P = panels('sf-chart', 3, { h: 340, hs: 220 }), tr = [], an = [], res = []; label('sf-l').textContent = L;
  const lg = Array.from({ length: L }, (_, i) => i + 1);
  AS.forEach((k, i) => {
    const r = win(k, w).r, band = 1.96 / Math.sqrt(r.length), a1 = acf(r, L), a2 = acf(r.map(Math.abs), L);
    tr.push(on2(P.ax[i], BAR(lg.map(x => x - 0.2), a1, BLUE, 'returns r', { width: 0.4, legendgroup: 'r', showlegend: i === 0, hovertemplate: 'lag %{x:.0f}: %{y:.3f}<extra>r</extra>' })));
    tr.push(on2(P.ax[i], BAR(lg.map(x => x + 0.2), a2, RED, 'absolute returns |r|', { width: 0.4, legendgroup: 'a', showlegend: i === 0, hovertemplate: 'lag %{x:.0f}: %{y:.3f}<extra>|r|</extra>' })));
    tr.push(on2(P.ax[i], { x: [0, L + 1, L + 1, 0], y: [-band, -band, band, band], fill: 'toself', fillcolor: 'rgba(181,133,63,0.18)', line: { width: 0 }, mode: 'lines', name: '95% band ±1.96/√T', legendgroup: 'b', showlegend: i === 0, hoverinfo: 'skip' }));
    setAx(P, i, AX('lag (days)', { range: [0, L + 1] }), AX(i === 0 || P.stack ? 'autocorrelation' : '', { range: [-0.15, 0.5], matches: i ? 'y' : undefined, showticklabels: i === 0 || P.stack }));
    an.push(ptitle(P, i, NAME[k]));
    res.push([NAME[k], nf(a2[0], 2), nf(a1[0], 2), a1.filter(v => Math.abs(v) > band).length, a2.filter(v => v > 0).length, nf(band, 3)]);
  });
  plot('sf-chart', tr, layout(Object.assign(P.L, { barmode: 'overlay', legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: M1, annotations: an })));
  out('sf-out', [T.sf(WY(w), L, res, !isLec(w) || L !== 50)]);
}

// ================================================================= x3. Chapter 1 (2/2): Sharpe ratio with its SE, mean and CAGR (extra chart)
function pfBlock() {
  const w = kSel('pf-w'), S = AS.map(k => performance(k, w)), P = panels('pf-chart', 2, { h: 380 }), tr = [];
  tr.push(on2(P.ax[0], { x: IX(NMS), customdata: NMS, y: S.map(s => s.sharpe), type: 'scatter', mode: 'markers', marker: { color: AS.map(k => COL[k]), size: 11 }, error_y: { type: 'data', array: S.map(s => 1.96 * s.sharpe_se), color: DARK, thickness: 1.3, width: 6 }, name: 'Sharpe ratio (rf = 0) ± 1.96 SE', hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }));
  tr.push(on2(P.ax[0], htr(-0.5, 2.5, 0, AMBER, 'Sharpe ratio 0')));
  tr.push(on2(P.ax[1], BAR(IX(NMS), S.map(s => s.mean_arith), BLUE, 'arithmetic mean of simple returns (% a year)', { customdata: NMS,  text: S.map(s => s.mean_arith.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: '%{customdata}: %{y:.1f}%<extra>mean</extra>' })));
  tr.push(on2(P.ax[1], BAR(IX(NMS), S.map(s => s.cagr), RED, 'CAGR (% a year)', { customdata: NMS,  text: S.map(s => s.cagr.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: '%{customdata}: %{y:.1f}%<extra>CAGR</extra>' })));
  setAx(P, 0, AX('', CX(NMS)), AX('Sharpe ratio, 95% interval')); setAx(P, 1, AX('', CX(NMS)), AX('% a year'));
  plot('pf-chart', tr, layout(Object.assign(P.L, { barmode: 'group', legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: M0 })));
  out('pf-out', [T.pf(WY(w), S.map((s, i) => [NMS[i], nf(s.cagr, 1), nf(s.vol, 1), nf(s.sharpe, 2), nf(s.sharpe - 1.96 * s.sharpe_se, 2), nf(s.sharpe + 1.96 * s.sharpe_se, 2), nf(s.mean_arith, 1), nf(s.mean_arith - s.cagr, 1)]), !isLec(w))]);
}

// ================================================================= x4. Chapter 3 (2/2): stable alpha by horizon (Chapter 3 estimates) and the running sample variance (extra chart)
const HZ = ['daily', 'weekly', 'monthly'];
function agBlock() {
  const k = kSel('ag-k'), A = D.agg[k], P = panels('ag-chart', 2, { h: 380, widths: [0.42, 0.58] }), tr = [], an = [];
  const ok = HZ.filter(h => !A[h].at_bound), bd = HZ.filter(h => A[h].at_bound);
  tr.push(on2(P.ax[0], { x: ok, y: ok.map(h => A[h].alpha), mode: 'lines+markers', line: { color: COL[k], width: 1.5 }, marker: { size: 9, color: COL[k] }, error_y: { type: 'data', array: ok.map(h => 1.96 * A[h].se), color: COL[k], width: 5 }, name: `${NAME[k]}: ML alpha, 95% interval`, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }));
  if (bd.length) tr.push(on2(P.ax[0], { x: bd, y: bd.map(() => 2), mode: 'markers', marker: { size: 10, symbol: 'x', color: COL[k] }, name: 'estimate at the bound alpha = 2', hovertemplate: '%{x}: at the bound 2<extra></extra>' }));
  else tr.push(on2(P.ax[0], { x: [null], y: [null], mode: 'markers', marker: { size: 10, symbol: 'x', color: COL[k] }, showlegend: false, hoverinfo: 'skip' }));
  tr.push(on2(P.ax[0], { x: HZ, y: HZ.map(() => 2), mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'alpha = 2 (Normal)', hoverinfo: 'skip' }));
  const RV = runVar(k), i0 = 19;
  tr.push(on2(P.ax[1], { x: RV.d.slice(i0), y: RV.v.slice(i0), mode: 'lines', line: { color: COL[k], width: 1.6 }, name: `${NAME[k]}: running sample variance of daily returns`, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }));
  setAx(P, 0, AX('return horizon', { type: 'category' }), AX('alpha-hat (ML)', { range: [1.2, 2.1] })); setAx(P, 1, AX(''), AX('running sample variance (log scale)', { type: 'log' }));
  plot('ag-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const i10 = RV.d.findIndex(d => d >= '2010-01-01'), v10 = i10 > 0 ? nf(RV.v[i10], 2) : null, last = RV.v[RV.v.length - 1];
  out('ag-out', [T.ag(NAME[k], HZ.map(h => A[h].at_bound ? null : [nf(A[h].alpha, 2), nf(A[h].alpha - 1.96 * A[h].se, 2), nf(A[h].alpha + 1.96 * A[h].se, 2), fint(A[h].n)]), DT(RV.d[0]), v10, nf(last, 2), k !== 'sp500')]);
}

// ================================================================= 3. VaR 1% forecasts for the BET
function btBlock() {
  const k = kSel('bt-k'), w = +$('bt-w').value, lam = +$('bt-l').value, B = backtest(k, w, lam), nw = single('bt-chart', 400); label('bt-w').textContent = w; label('bt-l').textContent = nf(lam, 2);
  const tr = [{ x: B.d, y: B.r, mode: 'lines', line: { color: BLUE, width: 0.5 }, name: `${NAME[k]} daily return (%)`, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: B.d, y: B.HS.map(v => -v), mode: 'lines', line: { color: AMBER, width: 1.2 }, name: `−VaR 1%, historical simulation (${w} days)`, hovertemplate: '%{x}: %{y:.2f}%<extra>HS</extra>' },
    { x: B.d, y: B.EWMA.map(v => -v), mode: 'lines', line: { color: RED, width: 1 }, name: `−VaR 1%, EWMA-Normal (λ = ${lam.toFixed(2)})`, hovertemplate: '%{x}: %{y:.2f}%<extra>EWMA</extra>' }];
  [['HS', AMBER, 'circle'], ['EWMA', RED, 'x']].forEach(([m, c, s]) => { const ix = B[m + '_hit'].map((h, i) => h ? i : -1).filter(i => i >= 0); tr.push({ x: ix.map(i => B.d[i]), y: ix.map(i => B.r[i]), mode: 'markers', marker: { color: c, symbol: s, size: 6 }, name: `exceptions ${m}: ${ix.length} of ${B.n}`, hovertemplate: '%{x}: %{y:.2f}%<extra>exception</extra>' }); });
  plot('bt-chart', tr, layout({ xaxis: AX(''), yaxis: AX('%'), legend: LEG(nw, -0.1), margin: M0 }));
  const st = m => { const s = B[m + '_st']; return [s.x, nf(s.rate, 2), fp(s.p), s.max250, nf(s.lr, 2)]; };
  out('bt-out', [T.bt(NAME[k], DT(B.first), fint(B.n), nf(B.n * K.a_var, 1), st('HS'), st('EWMA'), w, nf(lam, 2), k !== 'bet' || w !== K.window || Math.abs(lam - K.lam) > 1e-9, B.first > K.oos.slice(0, 4) + '-01-10')]);
}

// ================================================================= review problems (worked examples; the numbers of the slides as defaults)
const XS = (...a) => a.some(Boolean);
// Problem 1
const Q1C = ['arithmetic mean of R', 'CAGR'], Q2C = ['skewness part', 'kurtosis part', 'JB'], Q4C = ['Q(4) / 9.49', '|Z(5)| / 1.96', '|Z*(5)| / 1.96'], Q8C = ['accuracy', 'TPR', 'FPR', 'precision'];
function q1Block() {
  const P1 = +$('q1-a').value, P2 = +$('q1-p').value, P3 = +$('q1-c').value, Pt = [100, P1, P2, P3]; label('q1-a').textContent = P1; label('q1-p').textContent = P2; label('q1-c').textContent = P3;
  const R = [1, 2, 3].map(t => Pt[t] / Pt[t - 1] - 1), r = [1, 2, 3].map(t => Math.log(Pt[t] / Pt[t - 1])), sr = sum(r), mR = mean(R), cagr = (P3 / 100) ** (1 / 3) - 1;
  let mx = -Infinity, ipk = 0, dd = 0, a = 0, b = 0; const pk = [];
  Pt.forEach((v, i) => { if (v > mx) { mx = v; ipk = i; } pk.push(mx); if (v / mx - 1 < dd) { dd = v / mx - 1; a = ipk; b = i; } });
  const gain = dd < 0 ? Pt[a] / Pt[b] - 1 : 0;
  const P = panels('q1-chart', 2, { h: 340, widths: [0.58, 0.42] }), yr = [0, 1, 2, 3];
  const tr = [on2(P.ax[0], { x: yr, y: Pt, mode: 'lines+markers+text', line: { color: BLUE, width: 2 }, marker: { size: 8 }, text: Pt.map(String), textposition: 'top center', textfont: { size: 11, color: DARK }, cliponaxis: false, name: 'index value', hovertemplate: 'year %{x}: %{y}<extra></extra>' }),
    on2(P.ax[0], { x: yr, y: pk, mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dash' }, name: 'running peak', hoverinfo: 'skip' }),
    on2(P.ax[0], { x: dd < 0 ? [a, b] : [null], y: dd < 0 ? [Pt[a], Pt[b]] : [null], mode: 'markers', marker: { size: 12, color: RED, symbol: 'diamond' }, name: 'peak and trough of the maximum drawdown', hoverinfo: 'skip' }),
    on2(P.ax[1], BAR(IX(Q1C), [100 * mR, 100 * cagr], [BLUE, RED], 'per year (%)', { customdata: Q1C,  text: [mR, cagr].map(v => (100 * v).toFixed(2) + '%'), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' }))];
  const ym = Math.max(...[mR, cagr].map(v => Math.abs(100 * v)), 5) * 1.35;
  setAx(P, 0, AX('end of year', { tickvals: yr, range: [-0.3, 3.3] }), AX('index value', { range: [0, Math.max(...Pt) * 1.2] })); setAx(P, 1, AX('', CX(Q1C)), AX('% a year', { range: [-ym, ym] }));
  plot('q1-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const pc = v => nz(100 * v, 2), x = XS(P1 !== 160, P2 !== 64, P3 !== 96);
  steps('q1-we', [T.q1a(Pt, R.map(v => nz(100 * v, 2)), Pt.slice(1).map((v, i) => nz(v / Pt[i], 3)), r.map(v => nf(v, 3))), T.q1b(nf(sr, 3), P3, pc(Math.exp(sr) - 1)),
    T.q1c(R.map(v => nz(100 * v, 2)), pc(mR), P3, pc(cagr), mR > cagr), dd < 0 ? T.q1d(Pt[b], Pt[a], pc(dd), pc(gain)) : T.q1d0()]);
  out('q1-out', [T.q1(pc(mR), pc(cagr), pc(dd), pc(gain), x)]);
}
// Problem 2
const LOSS = [8.2, 6.5, 5.9, 5.1, 4.6];
function q2Block() {
  const Kx = +$('q2-k').value, S = +$('q2-s').value, n = +kSel('q2-n'), k = +kSel('q2-h'); label('q2-k').textContent = nf(Kx, 1); label('q2-s').textContent = nf(S, 1);
  const js = n / 6 * S * S, jk = n / 24 * Kx * Kx, jb = js + jk, nu = 4 + 6 / Kx, lr = LOSS.slice(0, k).map(v => Math.log(v / LOSS[k])), al = 1 / mean(lr), se = al / Math.sqrt(k);
  const P = panels('q2-chart', 2, { h: 340 }), crit = 5.99;
  const tr = [on2(P.ax[0], BAR(IX(Q2C), [js, jk, jb], [AMBER, RED, BLUE], 'Jarque-Bera', { customdata: Q2C,  text: [js, jk, jb].map(v => v.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' })),
    on2(P.ax[0], htr(-0.5, 2.5, crit, DARK, 'χ²(2) 5% critical value 5.99')),
    on2(P.ax[1], { x: [0, 1], customdata: ['Student-t with the same K: ν', `Hill alpha, k = ${k}`], y: [nu, al], type: 'bar', marker: { color: [GREEN, PURPLE] }, error_y: { type: 'data', array: [0, se], color: DARK, width: 6 }, text: [nu.toFixed(2), al.toFixed(2)], textposition: 'inside', insidetextanchor: 'start', textfont: { size: 11, color: '#FFFFFF' }, name: 'tail index (Hill: ± 1 SE)', showlegend: false, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], htr(-0.5, 1.5, 2, AMBER, 'alpha = 2: variance exists above', 'dot')), on2(P.ax[1], htr(-0.5, 1.5, 4, RED, 'alpha = 4: kurtosis exists above', 'dot'))];
  setAx(P, 0, AX('', CX(Q2C)), AX('statistic (log scale)', { type: 'log' })); setAx(P, 1, AX('', CX(['Student-t with the same K: ν', `Hill alpha, k = ${k}`])), AX('tail index', { range: [0, Math.max(nu, al + se, 4.5) * 1.12] }));
  plot('q2-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.14), margin: M0 })));
  const mm = T.mom.filter((_, i) => i + 1 < al), x = XS(Kx !== 6, S !== -0.5, n !== 1000, k !== 4);
  steps('q2-we', [T.q2a(n, nf(S, 1), nz(S * S, 2), nf(js, 2), nz(Kx, 1), nz(Kx * Kx, 2), nf(jk, 1), nf(jb, 1), jb > crit), T.q2b(nz(Kx, 1), nz(nu, 2)),
    T.q2c(k, LOSS.slice(0, k).map((v, i) => [nf(v, 1), nf(LOSS[k], 1), nf(lr[i], 3)]), nf(mean(lr), 4), nf(al, 2)), T.q2d(nf(al, 2), mm, al > 2, nf(se, 2))]);
  out('q2-out', [T.q2(nf(jb, 1), jb > crit, nz(nu, 2), nf(al, 2), x)]);
}
// Problem 3
function q3Block() {
  const rho = +$('q3-r').value, w = +$('q3-w').value, sg = +$('q3-s').value / 100, N = +kSel('q3-N'), s1 = 1.5, s2 = 2.5, mu = 0.08, p = 0.01, S0 = 100;
  label('q3-r').textContent = nf(rho, 1); label('q3-w').textContent = nf(100 * w, 0) + '%'; label('q3-s').textContent = nf(100 * sg, 0) + '%';
  const vp = r => w * w * s1 * s1 + (1 - w) ** 2 * s2 * s2 + 2 * w * (1 - w) * r * s1 * s2, a = w * w * s1 * s1, b = (1 - w) ** 2 * s2 * s2, c = 2 * w * (1 - w) * rho * s1 * s2, sp = Math.sqrt(vp(rho)), wa = w * s1 + (1 - w) * s2;
  const rs = Array.from({ length: 41 }, (_, i) => -1 + i * 0.05), P = panels('q3-chart', 1, { h: 340 });
  const tr = [{ x: rs, y: rs.map(r => Math.sqrt(Math.max(0, vp(r)))), mode: 'lines', line: { color: BLUE, width: 2 }, name: `portfolio volatility, weights ${(100 * w).toFixed(0)}% / ${(100 - 100 * w).toFixed(0)}%`, hovertemplate: 'ρ = %{x:.2f}: %{y:.3f}%<extra></extra>' },
    htr(-1, 1, wa, RED, `weighted average of the volatilities: ${wa.toFixed(2)}%`),
    { x: [rho], y: [sp], mode: 'markers+text', marker: { size: 11, color: ORANGE, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`${sp.toFixed(2)}%`], textposition: 'top left', textfont: { size: 11, color: DARK }, cliponaxis: false, name: `ρ = ${rho.toFixed(1)}`, hoverinfo: 'skip' }];
  plot('q3-chart', tr, layout({ xaxis: AX('correlation ρ', { range: [-1.05, 1.05] }), yaxis: AX('daily volatility of the portfolio (%)', { range: [0, 2.8] }), legend: LEG(narrow('q3-chart'), -0.2), margin: M0 }));
  const se = Math.sqrt(p * (1 - p) / N), m1 = S0 * Math.exp(mu), md = S0 * Math.exp(mu - sg * sg / 2), z = -(mu - sg * sg / 2) / sg, pl = Phi(z), x = XS(Math.abs(rho - 0.3) > 1e-9, Math.abs(w - 0.6) > 1e-9, Math.abs(sg - 0.25) > 1e-9, N !== 10000);
  steps('q3-we', [T.q3a(nf(100 * w, 0), nf(100 - 100 * w, 0), nf(rho, 1), nf(a, 2), nf(b, 2), nf(c, 2), nf(vp(rho), 2), nf(sp, 2), nf(wa, 2), sp < wa - 1e-12),
    T.q3b(fint(N), nf(100 * se, 3), nf(100 * (p - 1.96 * se), 2), nf(100 * (p + 1.96 * se), 2)),
    T.q3c(nf(m1, 2), nf(sg * sg / 2, 5), nf(md, 2), nf(mu - sg * sg / 2, 4), nf(sg, 2), nf(z, 3), nf(100 * pl, 1))]);
  out('q3-out', [T.q3(nf(sp, 2), nf(wa, 2), nf(100 * se, 3), nf(100 * pl, 1), x)]);
}
// Problem 4
function q4Block() {
  const r1 = +$('q4-r').value, Tn = +kSel('q4-T'), seR = +$('q4-e').value, rho = [r1, 0.02, -0.01, 0.01], q = 5; label('q4-r').textContent = nf(r1, 2); label('q4-e').textContent = nf(seR, 3);
  let Q = 0; rho.forEach((r, i) => { Q += r * r / (Tn - i - 1); }); Q *= Tn * (Tn + 2);
  const s2 = sum(rho.map(r => r * r)), wsum = sum(rho.map((r, i) => (1 - (i + 1) / q) * r)), VR = 1 + 2 * wsum, se = Math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * Tn)), Z = (VR - 1) / se, Zs = (VR - 1) / seR, band = 1.96 / Math.sqrt(Tn);
  const P = panels('q4-chart', 2, { h: 340 });
  const tr = [on2(P.ax[0], { x: [0.4, 4.6, 4.6, 0.4], y: [-band, -band, band, band], fill: 'toself', fillcolor: 'rgba(181,133,63,0.18)', line: { width: 0 }, mode: 'lines', name: `±1.96/√T = ±${band.toFixed(3)}`, hoverinfo: 'skip' }),
    on2(P.ax[0], BAR([1, 2, 3, 4], rho, BLUE, 'sample autocorrelation', { showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' })),
    on2(P.ax[1], BAR(IX(Q4C), [Q / 9.49, Math.abs(Z) / 1.96, Math.abs(Zs) / 1.96], [BLUE, RED, GREEN], 'statistic / 5% critical value', { customdata: Q4C,  text: [Q, Z, Zs].map(v => v.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' })),
    on2(P.ax[1], htr(-0.5, 2.5, 1, DARK, 'rejection above 1'))];
  setAx(P, 0, AX('lag k', { tickvals: [1, 2, 3, 4], range: [0.4, 4.6] }), AX('ρ̂(k)', { range: [-0.12, 0.17] })); setAx(P, 1, AX('', CX(Q4C)), AX('statistic / critical value', { range: [0, Math.max(2, Q / 9.49, Math.abs(Z) / 1.96, Math.abs(Zs) / 1.96) * 1.18] }));
  plot('q4-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const x = XS(Math.abs(r1 - 0.06) > 1e-9, Tn !== 2500, Math.abs(seR - 0.065) > 1e-9);
  steps('q4-we', [T.q4a(fint(Tn), nf(Q, 2), nf(Tn * s2, 2), nf(s2, 4), Q > 9.49), T.q4b(rho.map((r, i) => [nf(1 - (i + 1) / q, 1), nf(Math.abs(r), 2), r < 0]), nf(wsum, 3), nf(VR, 3)),
    T.q4c(fint(2 * (2 * q - 1) * (q - 1)), fint(3 * q * Tn), nf(se, 4), nf(Z, 2), Math.abs(Z) > 1.96, nf(VR, 3), nf(seR, 3), nf(Zs, 2), Math.abs(Zs) > 1.96)]);
  out('q4-out', [T.q4(nf(Q, 2), Q > 9.49, nf(VR, 3), nf(Z, 2), nf(Zs, 2), Math.abs(Z) > 1.96, Math.abs(Zs) > 1.96, x)]);
}
// Problem 5
function q5Block() {
  const eps = +$('q5-e').value, be = +$('q5-b').value, lam = +kSel('q5-l'), om = 0.03, al = 0.09, s2 = 1.8, z = 2.326, sdB = 3.5;
  label('q5-e').textContent = nf(eps, 2); label('q5-b').textContent = nf(be, 3);
  const s1 = om + al * eps * eps + be * s2, pers = al + be, lr = om / (1 - pers), hl = Math.log(0.5) / Math.log(pers), f = h => lr + pers ** (h - 1) * (s1 - lr), ew = lam * s2 + (1 - lam) * eps * eps;
  const hs = Array.from({ length: 60 }, (_, i) => i + 1);
  const tr = [{ x: hs, y: hs.map(f), mode: 'lines', line: { color: BLUE, width: 2 }, name: `GARCH(1,1) forecast, α + β = ${pers.toFixed(3)}`, hovertemplate: 'h = %{x}: %{y:.3f}<extra>GARCH</extra>' },
    { x: hs, y: hs.map(() => ew), mode: 'lines', line: { color: RED, width: 1.6, dash: 'dash' }, name: `EWMA forecast (λ = ${lam.toFixed(2)})`, hovertemplate: 'h = %{x}: %{y:.3f}<extra>EWMA</extra>' },
    htr(1, 60, lr, AMBER, `long-run variance ω/(1 − α − β) = ${lr.toFixed(2)}`, 'dot'),
    { x: [10], y: [f(10)], mode: 'markers', marker: { size: 10, color: ORANGE, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `h = 10: ${f(10).toFixed(3)}`, hoverinfo: 'skip' }];
  const nw = single('q5-chart', 340);
  plot('q5-chart', tr, layout({ xaxis: AX('horizon h (days)'), yaxis: AX('forecast of the daily variance (%²)', { rangemode: 'tozero' }), legend: LEG(nw, -0.2), margin: M0 }));
  const x = XS(eps !== -2.5, Math.abs(be - 0.89) > 1e-9, lam !== 0.94);
  steps('q5-we', [T.q5a(nz(al * eps * eps, 4), nf(be * s2, 3), nf(s1, 4), nf(Math.sqrt(s1), 3), nf(z * Math.sqrt(s1), 2)), T.q5b(nz(pers, 3), nf(hl, 1), nz(1 - pers, 3), nf(lr, 2), nf(Math.sqrt(252 * lr), 2)),
    T.q5c(nf(lr, 2), nz(pers, 3), nf(pers ** 9, 4), nf(s1, 4), nf(f(10), 3), nf(lam, 2), nf(1 - lam, 2), nz(eps * eps, 4), nf(ew, 3)), T.q5d(nf(sdB, 1), nf(sdB * Math.sqrt(365), 1), nf(sdB * Math.sqrt(252), 1))]);
  out('q5-out', [T.q5(nf(s1, 4), nf(z * Math.sqrt(s1), 2), nf(hl, 1), nf(f(10), 3), nf(ew, 3), x)]);
}
// Problem 6
const HIST = [-5.1, -4.4, -3.9, -3.6, -3.2, -3.0, -2.9];
const zone = x => x <= 4 ? 'g' : x <= 9 ? 'y' : 'r';
function q6Block() {
  const xe = +$('q6-x').value, sg = +$('q6-s').value, n = 250, pos = 1e6, a = 0.01; label('q6-x').textContent = xe; label('q6-s').textContent = nf(sg, 1);
  const z1 = -qnorm(0.01), z25 = -qnorm(0.025), V = z1 * sg, E = sg * phi(z25) / 0.025, kv = Math.ceil(n * 0.01 - 1e-9), ke = Math.ceil(n * 0.025 - 1e-9), hv = -HIST[kv - 1], he = -mean(HIST.slice(0, ke));
  const K6 = kupiec(xe, n, a), xs = Array.from({ length: 16 }, (_, i) => i), lrs = xs.map(v => kupiec(v, n, a).lr);
  const tr = [{ x: xs, y: lrs, mode: 'lines+markers', line: { color: BLUE, width: 1.6 }, marker: { size: 6, color: xs.map(v => ({ g: GREEN, y: AMBER, r: RED })[zone(v)]) }, name: 'Kupiec LR_uc (250 days, VaR 1%)', hovertemplate: '%{x} exceptions: LR = %{y:.2f}<extra></extra>' },
    htr(0, 15, 3.84, DARK, 'χ²(1) 5% critical value 3.84'),
    { x: [xe], y: [K6.lr], mode: 'markers+text', marker: { size: 13, color: ORANGE, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`LR = ${K6.lr.toFixed(2)}`], textposition: 'top left', textfont: { size: 11, color: DARK }, cliponaxis: false, name: `${xe} exceptions`, hoverinfo: 'skip' }];
  const sh = [[-0.5, 4.5, 'rgba(46,125,50,0.10)'], [4.5, 9.5, 'rgba(181,133,63,0.16)'], [9.5, 15.5, 'rgba(205,0,0,0.10)']].map(([x0, x1, c]) => ({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: c, line: { width: 0 }, layer: 'below' }));
  const an = [['green zone', 2], ['yellow zone', 7], ['red zone', 12.5]].map(([t, x]) => ({ x, y: 1, xref: 'x', yref: 'paper', text: t, showarrow: false, yanchor: 'top', font: { size: 11, color: DARK } }));
  const nw = single('q6-chart', 340);
  plot('q6-chart', tr, layout({ xaxis: AX('exceptions of VaR 1% in 250 days', { range: [-0.5, 15.5], dtick: 1 }), yaxis: AX('LR_uc', { rangemode: 'tozero' }), legend: LEG(nw, -0.2), margin: M0, shapes: sh, annotations: an }));
  const x = XS(xe !== 6, Math.abs(sg - 1.6) > 1e-9);
  steps('q6-we', [T.q6a(nf(z1, 3), nf(sg, 1), nf(V, 2), fint(V / 100 * pos), nf(phi(z25), 5), nf(phi(z25) / 0.025, 3), nf(E, 2), fint(E / 100 * pos), nf(Math.sqrt(10) * V, 2)),
    T.q6b(nf(n * 0.01, 1), kv, nf(hv, 1), nf(n * 0.025, 2), ke, nf(-sum(HIST.slice(0, ke)), 1), nf(he, 2)),
    T.q6c(xe, n - xe, nf(K6.l0, 2), nf(xe / n, 3), nf(K6.l1, 2), nf(K6.lr, 2), nf(K6.p, 3), K6.lr > 3.84, zone(xe), nf(1 - xe / n, 3))]);
  out('q6-out', [T.q6(nf(V, 2), nf(E, 2), nf(hv, 1), nf(he, 2), xe, nf(K6.lr, 2), nf(K6.p, 3), K6.lr > 3.84, zone(xe), x)]);
}
// Problem 7
function q7Block() {
  const b = +$('q7-b').value, a = +$('q7-a').value, v1 = +$('q7-v').value, h = +kSel('q7-h'); label('q7-b').textContent = nf(b, 1); label('q7-a').textContent = nf(a, 1); label('q7-v').textContent = nf(v1, 1);
  const ratio = b / a, H = Math.log10(ratio) / Math.log10(2000 / 20), d = H - 0.5, fh = h ** H, fs = Math.sqrt(h);
  const P = panels('q7-chart', 2, { h: 340 }), ln = [Math.log10(20), Math.log10(2000)], hh = Array.from({ length: 60 }, (_, i) => i + 1);
  const tr = [on2(P.ax[0], { x: ln, y: [Math.log10(a), Math.log10(b)], mode: 'lines+markers', line: { color: BLUE, width: 2 }, marker: { size: 8 }, name: `average R/S, slope H = ${H.toFixed(3)}`, hovertemplate: 'log10 n = %{x:.2f}: log10 R/S = %{y:.3f}<extra></extra>' }),
    on2(P.ax[0], { x: ln, y: [Math.log10(a), Math.log10(a) + 0.5 * 2], mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dash' }, name: 'slope 0.5 (no memory)', hoverinfo: 'skip' }),
    on2(P.ax[1], { x: hh, y: hh.map(t => v1 * t ** H), mode: 'lines', line: { color: RED, width: 2 }, name: `VaR × h^H`, hovertemplate: 'h = %{x}: %{y:.2f}%<extra>h^H</extra>' }),
    on2(P.ax[1], { x: hh, y: hh.map(t => v1 * Math.sqrt(t)), mode: 'lines', line: { color: GREEN, width: 1.6, dash: 'dash' }, name: 'VaR × √h', hovertemplate: 'h = %{x}: %{y:.2f}%<extra>√h</extra>' }),
    on2(P.ax[1], { x: [h, h], y: [v1 * fh, v1 * fs], mode: 'markers', marker: { size: 10, color: [RED, GREEN], symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `h = ${h} days`, hoverinfo: 'skip' })];
  setAx(P, 0, AX('log10 n (block length)'), AX('log10 R/S')); setAx(P, 1, AX('horizon h (days)'), AX('VaR 1% (%)', { rangemode: 'tozero' }));
  plot('q7-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const x = XS(b !== 60, Math.abs(a - 4.1) > 1e-9, v1 !== 2.5, h !== 10);
  steps('q7-we', [T.q7a(nf(b, 1), nf(a, 1), nf(ratio, 2), nf(Math.log10(ratio), 3), nf(H, 3), nf(d, 3), H > 0.5), T.q7b(h, nf(H, 3), nf(fh, 2), nf(v1, 1), nf(v1 * fh, 2), nf(fs, 2), nf(v1 * fs, 2), H > 0.5), T.q7c()]);
  out('q7-out', [T.q7(nf(H, 3), nf(d, 3), h, nf(v1 * fh, 2), nf(v1 * fs, 2), x)]);
}
// Problem 8
function q8Block() {
  const FP = +$('q8-f').value, TP = +$('q8-t').value, bh = +$('q8-b').value, ah = -2.1, q = -5.0, md = 0.1, bad = 90, good = 210, N = 300;
  label('q8-f').textContent = FP; label('q8-t').textContent = TP; label('q8-b').textContent = nf(bh, 1);
  const TN = good - FP, acc = (TP + TN) / N, tpr = TP / bad, fpr = FP / good, prec = TP + FP > 0 ? TP / (TP + FP) : null, ag = good / N;
  const tr = [BAR(IX(Q8C), [acc, tpr, fpr, prec === null ? 0 : prec].map(v => 100 * v), [BLUE, GREEN, RED, PURPLE], 'rates (%)', { customdata: Q8C,  text: [acc, tpr, fpr, prec].map(v => v === null ? '–' : (100 * v).toFixed(1) + '%'), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{customdata}: %{y:.1f}%<extra></extra>' }),
    htr(-0.5, 3.5, 100 * ag, AMBER, `accuracy of „always good”: ${(100 * ag).toFixed(1)}%`)];
  const nw = single('q8-chart', 330);
  plot('q8-chart', tr, layout({ xaxis: AX('', CX(Q8C)), yaxis: AX('%', { range: [0, 115] }), legend: LEG(nw, -0.14), margin: M0 }));
  const cv = -(ah + bh * q), cm = -(ah + bh * md), dc = bh * (md - q), x = XS(FP !== 42, TP !== 60, Math.abs(bh - 0.6) > 1e-9);
  steps('q8-we', [T.q8a(TP, TN, nf(100 * acc, 1), nf(100 * tpr, 1), FP, nf(100 * fpr, 1), TP + FP, prec === null ? null : nf(100 * prec, 1), nf(100 * ag, 1), acc > ag),
    T.q8b(), T.q8c(nf(bh, 1), nf(cv, 2), nf(cm, 2), nf(dc, 2), nf(md - q, 1))]);
  out('q8-out', [T.q8(nf(100 * acc, 1), nf(100 * tpr, 1), nf(100 * fpr, 1), prec === null ? null : nf(100 * prec, 1), nf(dc, 2), x)]);
}

// ================================================================= wiring
const PFX = { three: 'ts', 'sum-tab': 'sm', interp: 'ip', stylised: 'sf', perf: 'pf', agg: 'ag', backtest: 'bt', p1: 'q1', p2: 'q2', p3: 'q3', p4: 'q4', p5: 'q5', p6: 'q6', p7: 'q7', p8: 'q8' };
const BLOCKS = { tsBlock, smBlock, ipBlock, sfBlock, pfBlock, agBlock, btBlock, q1Block, q2Block, q3Block, q4Block, q5Block, q6Block, q7Block, q8Block };
const PLAYERS = {};
function placeExtras() {                // web lecture: the table and worked-example blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const PLy = (btn, id, vals, ms_) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms_));
  const WS = D.windows;
  on(['ts-w'], tsBlock); PLy('ts-play', 'ts-w', [...WS.slice(1), WS[0]], 1500);
  on(['sm-w'], smBlock); PLy('sm-play', 'sm-w', [...WS.slice(1), WS[0]], 1700);
  on(['ip-a'], ipBlock); PLy('ip-play', 'ip-a', ['0.005', '0.025', '0.05', '0.01'], 1500);
  on(['sf-w', 'sf-l'], sfBlock); PLy('sf-play', 'sf-w', [...WS.slice(1), WS[0]], 1500);
  on(['pf-w'], pfBlock); PLy('pf-play', 'pf-w', [...WS.slice(1), WS[0]], 1500);
  on(['ag-k'], agBlock); PLy('ag-play', 'ag-k', ['bet', 'btc', 'sp500'], 1700);
  on(['bt-k', 'bt-w', 'bt-l'], btBlock); PLy('bt-play', 'bt-w', [250, 750, 1000, 500], 1700);
  on(['q1-p', 'q1-a', 'q1-c'], q1Block); PLy('q1-play', 'q1-p', [80, 100, 120, 140, 40, 64], 1100);
  on(['q2-k', 'q2-s', 'q2-n', 'q2-h'], q2Block); PLy('q2-play', 'q2-k', [1, 2, 3, 4, 8, 12, 6], 1000);
  on(['q3-r', 'q3-w', 'q3-s', 'q3-N'], q3Block); PLy('q3-play', 'q3-r', [-1, -0.5, 0, 0.5, 1, 0.3], 1100);
  on(['q4-r', 'q4-T', 'q4-e'], q4Block); PLy('q4-play', 'q4-r', [-0.06, -0.03, 0, 0.03, 0.09, 0.12, 0.06], 1000);
  on(['q5-e', 'q5-b', 'q5-l'], q5Block); PLy('q5-play', 'q5-e', [-5, -4, -1, 0, 1, 2.5, 5, -2.5], 1000);
  on(['q6-x', 'q6-s'], q6Block); PLy('q6-play', 'q6-x', [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 6], 800);
  on(['q7-b', 'q7-a', 'q7-v', 'q7-h'], q7Block); PLy('q7-play', 'q7-b', [20, 30, 40, 50, 80, 100, 120, 60], 1000);
  on(['q8-f', 'q8-t', 'q8-b'], q8Block); PLy('q8-play', 'q8-f', [0, 12, 24, 60, 90, 120, 42], 1000);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH16 = { win, performance, moments, tailIndex, acf, ljungBox, varianceRatio, hsVar, hsEs, normalVar, normalEs, hurstRS, summary, backtest, kupiec, runVar, qnorm, Phi, chi2sf, init, PFX, blocks: BLOCKS };
if (window.CH16_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
