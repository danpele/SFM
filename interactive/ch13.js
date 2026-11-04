// Chapter 13 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch13_data.js.
(function () {
'use strict';
const D = window.CH13_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG, solve } = window.SFMI;
const T = window.CH13_TEXT;
const NUM = D.N, COL = D.colors, MC = D.mcol, FC = D.feat_col, FL = D.feat_label;
const DARK = C.navy, GREEN = COL.Forest, RED = COL.IDAred, BLUE = COL.MainBlue, TEAL = COL.Teal, PURPLE = COL.Purple, ORANGE = COL.Orange, AMBER = COL.Amber;

// ================================================================= numerics
function Phi(x) {                    // Hart (1968) / West (2005), double precision
  const z = Math.abs(x); let c;
  if (z > 37) c = 0;
  else {
    const e = Math.exp(-z * z / 2);
    if (z < 7.07106781186547) {
      const n = ((((((0.0352624965998911 * z + 0.700383064443688) * z + 6.37396220353165) * z + 33.912866078383) * z + 112.079291497871) * z + 221.213596169931) * z + 220.206867912376);
      const d = (((((((0.0883883476483184 * z + 1.75566716318264) * z + 16.064177579207) * z + 86.7807322029461) * z + 296.564248779674) * z + 637.333633378831) * z + 793.826512519948) * z + 440.413735824752);
      c = e * n / d;
    } else { let b = z + 0.65; b = z + 4 / b; b = z + 3 / b; b = z + 2 / b; b = z + 1 / b; c = e / b / 2.506628274631; }
  }
  return x > 0 ? 1 - c : c;
}
function Pinv(p) {                   // Acklam, then one Halley step
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
const chi1sf = x => (x <= 0 ? 1 : 2 * Phi(-Math.sqrt(x)));      // chi-square(1) upper tail
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
const vr0 = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return s / a.length; };   // np.var (ddof 0)
const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; saa += u * u; sbb += v * v; } return sab / Math.sqrt(saa * sbb); };
function ranks(s) {
  const idx = s.map((_, i) => i).sort((a, b) => s[a] - s[b]), r = new Array(s.length); let i = 0;
  while (i < idx.length) { let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++; const v = (i + j) / 2 + 1; for (let q = i; q <= j; q++) r[idx[q]] = v; i = j + 1; }
  return r;
}
function auc(s, y) { const r = ranks(s); let n1 = 0, rs = 0; y.forEach((v, i) => { if (v === 1) { n1++; rs += r[i]; } }); const n0 = y.length - n1; return (rs - n1 * (n1 + 1) / 2) / (n1 * n0); }
function ols(X, y) {                 // OLS with a constant (first coefficient); X: rows of regressors
  const k = X[0].length + 1, A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let i = 0; i < y.length; i++) { const r = [1, ...X[i]]; for (let j = 0; j < k; j++) { b[j] += r[j] * y[i]; for (let l = 0; l < k; l++) A[j][l] += r[j] * r[l]; } }
  return solve(A, b);
}
const DAY = 864e5;
const DT = {};
function dates(o) { const key = o.t0 + ':' + o.t.length; if (!DT[key]) { const b = Date.parse(o.t0 + 'T00:00:00Z'); DT[key] = o.t.map(d => new Date(b + d * DAY).toISOString().slice(0, 10)); } return DT[key]; }
const yearOf = s => +s.slice(0, 4);

// ================================================================= the chapter's statistics (generate_all_charts.py)
const oosR2 = (y, f, b) => { let a = 0, c = 0; for (let i = 0; i < y.length; i++) { a += (y[i] - f[i]) ** 2; c += (y[i] - b[i]) ** 2; } return 1 - a / c; };
const qlike = (rv, h) => rv / h + Math.log(h);
function dmTest(d, lag) {            // Diebold-Mariano with a Newey-West (Bartlett) long-run variance of `lag` lags
  const n = d.length, m = mean(d), u = d.map(v => v - m);
  let lrv = sum(u.map(v => v * v)) / n; const g = [lrv];
  for (let j = 1; j <= lag; j++) { let s = 0; for (let i = j; i < n; i++) s += u[i] * u[i - j]; g.push(s / n); lrv += 2 * (1 - j / (lag + 1)) * s / n; }
  const t = m / Math.sqrt(lrv / n);
  return { t, p: 2 * Phi(-Math.abs(t)), mean: m, lrv, g0: g[0], n };
}
function acf(d, L) { const n = d.length, m = mean(d), u = d.map(v => v - m), g0 = sum(u.map(v => v * v)) / n; return Array.from({ length: L + 1 }, (_, j) => { let s = 0; for (let i = j; i < n; i++) s += u[i] * u[i - j]; return s / n / g0; }); }
function kupiec(x, n, a) {
  const ph = x / n, l0 = (n - x) * Math.log(1 - a) + x * Math.log(a), l1 = (n - x) * Math.log(1 - ph) + (x > 0 ? x * Math.log(ph) : 0);
  const lr = -2 * (l0 - l1); return { lr, p: chi1sf(lr) };
}
function christoffersen(h) {
  let n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (let i = 1; i < h.length; i++) { const a = h[i - 1], b = h[i]; if (!a && !b) n00++; else if (!a && b) n01++; else if (a && !b) n10++; else n11++; }
  const p01 = n01 / Math.max(n00 + n01, 1), p11 = n11 / Math.max(n10 + n11, 1), p = (n01 + n11) / Math.max(n00 + n01 + n10 + n11, 1);
  const ll = (q, k0, k1) => (k0 ? k0 * Math.log(1 - q) : 0) + (k1 ? k1 * Math.log(q) : 0);
  const lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11));
  return { lr, p: chi1sf(lr), n11 };
}
const EULER = D.snoop.euler;
const expMax = (N, sd) => (N === 1 ? 0 : sd * ((1 - EULER) * Pinv(1 - 1 / N) + EULER * Pinv(1 - 1 / (N * Math.E))));
function deflated(sr, sr0, Tn, sk, ku) {
  const den = Math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr * sr), z = (sr - sr0) * Math.sqrt(Tn - 1) / den, z0 = sr * Math.sqrt(Tn - 1) / den;
  return { z, dsr: Phi(z), psr: Phi(z0), den };
}

// realised variance: walk-forward forecasts of the six models (exported), metrics recomputed here
const RV = {};
function rvSet(k) {
  if (RV[k]) return RV[k];
  const R = D.rv[k];
  if (!R.y) { const F = R.frame, n = R.t.length; R.y = F.y.slice(R.off, R.off + n); R.rv = F.rv.slice(R.off, R.off + n); }   // S&P 500: the test rows of the HAR frame
  const ds = dates(R), yi = ds.map(s => R.years.indexOf(yearOf(s)));
  const v = {}; R.models.forEach(m => { v[m] = R.pred[m].map((p, i) => Math.exp(p + R.s2[m][yi[i]] / 2)); });
  return (RV[k] = { R, ds, yi, var: v });
}
function rvMetrics(k, bench, lags, keep, loss) {
  const S = rvSet(k), R = S.R, idx = R.y.map((_, i) => i).filter(i => !keep || keep(S.ds[i]));
  const y = idx.map(i => R.y[i]), rv = idx.map(i => R.rv[i]), P = m => idx.map(i => R.pred[m][i]), V = m => idx.map(i => S.var[m][i]);
  const L = m => (loss === 'mse' ? P(m).map((p, j) => (y[j] - p) ** 2) : V(m).map((h, j) => qlike(rv[j], h)));
  const Lb = L(bench), out = { n: idx.length };
  R.models.forEach(m => {
    const Lm = L(m), dm = m === bench ? null : dmTest(Lm.map((v, j) => v - Lb[j]), lags);
    out[m] = { mse: mean(P(m).map((p, j) => (y[j] - p) ** 2)), r2_mean: oosR2(y, P(m), P('Mean')), r2_b: oosR2(y, P(m), P(bench)),
      qlike: mean(V(m).map((h, j) => qlike(rv[j], h))), dm_t: dm ? dm.t : NaN, dm_p: dm ? dm.p : NaN, dm };
  });
  out.idx = idx;
  return out;
}
// HAR, refitted here on every walk-forward fold of the S&P 500 (the chapter's walk_forward with LinearRegression)
const HARF = {};
function harFold(yr) {
  if (HARF[yr]) return HARF[yr];
  const F = D.rv.sp500.frame, ds = dates(F), tr = [], te = [];
  ds.forEach((s, i) => { if (s < yr + '-01-01') tr.push(i); else if (yearOf(s) === yr) te.push(i); });
  const trp = tr.slice(0, tr.length - D.H), X = trp.map(i => [F.d[i], F.w[i], F.m[i]]), y = trp.map(i => F.y[i]);
  const b = ols(X, y), fit = r => b[0] + b[1] * r[0] + b[2] * r[1] + b[3] * r[2];
  const s2 = vr0(trp.map((i, j) => y[j] - fit(X[j])));
  const pred = te.map(i => fit([F.d[i], F.w[i], F.m[i]]));
  return (HARF[yr] = { b, s2, tr: trp, te, pred, ds, F, purged: tr.slice(tr.length - D.H) });
}
// sign classifiers: metrics of the exported walk-forward probabilities
function signMetrics(k, m, end, thr) {
  const S = D.sign[k], ds = dates(S), idx = S.y.map((_, i) => i).filter(i => yearOf(ds[i]) <= end);
  if (!idx.length) return null;
  const y = idx.map(i => S.y[i]), n = y.length, accB = mean(idx.map(i => +(S.base[i] === S.y[i]))), p = idx.map(i => S.prob[m][i]);
  const acc = mean(p.map((v, j) => +((v >= thr ? 1 : 0) === y[j]))), se = Math.sqrt(accB * (1 - accB) / n), z = (acc - accB) / se;
  return { n, up: mean(y), accB, acc, diff: acc - accB, z, p: 2 * Phi(-Math.abs(z)), se, auc: auc(p, y), shareUp: mean(p.map(v => +(v >= thr))) };
}
// VaR 1% backtest of the exported forecasts
function backtest(k, m, keep) {
  const Q = D.qvar[k], ds = dates(Q), idx = Q.y.map((_, i) => i).filter(i => !keep || keep(ds[i])), a = 0.01;
  const y = idx.map(i => Q.y[i]), V = idx.map(i => Q.var[m][i]), hit = y.map((v, j) => +(v < -V[j])), x = sum(hit), n = y.length;
  const ku = kupiec(x, n, a), ch = christoffersen(hit);
  return { x, n, rate: x / n, p_uc: ku.p, lr_uc: ku.lr, p_ind: ch.p, n11: ch.n11, qloss: mean(y.map((v, j) => (a - hit[j]) * (v + V[j]))), mean_var: mean(V), hit, idx };
}
function kupiecBand(n) { let lo = null, hi = null; for (let x = 0; x <= n; x++) { if (kupiec(x, n, 0.01).p > 0.05) { if (lo === null) lo = x; hi = x; } else if (lo !== null) break; } return [lo / n, hi / n]; }

// ================================================================= formatting, tables, panels
const SEP = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, SEP).replace('-', '−');
const pc = (v, d) => nf(100 * v, d === undefined ? 1 : d) + '%';
const sg = (x, d) => (x > 0 ? '+' : '') + nf(x, d);
const SIG = (p, k) => (p < 0.001 ? (LANG === 'ro' ? '< 0,001' : '< 0.001') : nf(p, k || 3));
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.span ? ` colspan="${o.span}"` : ''}>${o.v}</${tag}>`; };
  const hs = Array.isArray(head[0]) ? head : [head];
  $(id).innerHTML = `<table class="xt"><thead>${hs.map(h => `<tr>${h.map(x => cell(x, 'th')).join('')}</tr>`).join('')}</thead><tbody>` +
    rows.map(r => `<tr>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = o.rows || (n > 1 && el.clientWidth < (o.minW || 560));
  el.style.height = (stack ? (o.hs || 270) * n + 110 : (o.h || 380)) + 'px';
  const g = stack ? (o.rows ? 0.16 : 0.2) : (o.gap || 0.1), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', dom = [i * (w + g), i * (w + g) + w];
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [1 - dom[1], 1 - dom[0]] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const kSel = id => $(id).value;
const MG = { l: 56, r: 12, t: 26, b: 40 };
const M0 = { l: 56, r: 12, t: 14, b: 40 };
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const ANN = Math.sqrt(252);
const MNAME = { Mean: 'training mean', HAR: 'HAR', Lasso: 'lasso', RF: 'random forest', GB: 'gradient boosting', MLP: 'MLP', Logit: 'logit', QR: 'QR', QGB: 'QGB', HS: 'HS', 'GARCH-t': 'GARCH-t' };
const ANAME = { sp500: 'S&P 500', btc: 'Bitcoin', bet: 'BET' };
const PER = { all: null, pre: s => yearOf(s) <= 2019, post: s => yearOf(s) >= 2020 };

// ================================================================= 1. bias and variance of regression trees (simulation)
function bvBlock() {
  const d = +$('bv-d').value, sig = kSel('bv-s'), R = D.bv[sig + '|80'], P = panels('bv-chart', 2, { h: 390 }), grid = lin(0, 1, 400); label('bv-d').textContent = d;
  const tr = [on2(P.ax[0], { x: R.x, y: R.y, mode: 'markers', marker: { color: BLUE, size: 5, opacity: 0.6 }, name: 'training data', hovertemplate: 'x %{x:.2f}, y %{y:.2f}<extra></extra>' }),
    on2(P.ax[0], { x: grid, y: grid.map(x => Math.sin(2 * Math.PI * x)), mode: 'lines', line: { color: DARK, width: 1.6 }, name: 'true function', hoverinfo: 'skip' })];
  [[1, AMBER], [d, GREEN], [10, RED]].forEach(([k, c], j) => { if (j === 1 && (k === 1 || k === 10)) return;
    tr.push(on2(P.ax[0], { x: grid, y: R.fits[k - 1], mode: 'lines', line: { color: c, width: j === 1 ? 2.4 : 1.6, shape: 'hv' }, name: `tree of depth ${k}`, hovertemplate: `depth ${k}: %{y:.3f}<extra></extra>` })); });
  const dd = lin(1, 10, 10), B = dd.map(k => R.bv[k]), s2 = +sig * +sig;
  [['bias2', AMBER, 'squared bias', 'solid'], ['var', PURPLE, 'variance', 'solid'], ['test', RED, 'expected test error', 'solid'], ['train', GREEN, 'training error', 'dash']].forEach(([k, c, nm, ds]) =>
    tr.push(on2(P.ax[1], { x: dd, y: B.map(b => b[k]), mode: 'lines+markers', line: { color: c, width: k === 'test' ? 2.4 : 1.6, dash: ds }, marker: { size: 6 }, name: nm, hovertemplate: `depth %{x}: ${nm} %{y:.4f}<extra></extra>` })));
  tr.push(on2(P.ax[1], { x: [1, 10], y: [s2, s2], mode: 'lines', line: { color: BLUE, width: 1.2, dash: 'dot' }, name: 'noise variance', hoverinfo: 'skip' }));
  const b = R.bv[d], best = dd.reduce((a, k) => (R.bv[k].test < R.bv[a].test ? k : a), 1);
  tr.push(on2(P.ax[1], { x: [d, d], y: [b.test, b.train], mode: 'markers+text', marker: { color: [RED, GREEN], size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [b.test.toFixed(3), b.train.toFixed(3)], textposition: 'top right', textfont: { size: 11, color: DARK }, name: `depth ${d}`, hoverinfo: 'skip', showlegend: false }));
  setAx(P, 0, AX('x'), AX('y')); setAx(P, 1, AX('tree depth (complexity)', { dtick: 1 }), AX('mean squared error'));
  plot('bv-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: M0 })));
  out('bv-out', [T.bv(nf(s2, 2), d, nf(b.bias2, 3), nf(b.var, 3), nf(b.test, 3), nf(b.train, 3), best, nf(R.bv[best].test, 3), sig !== '0.3')]);
}

// ================================================================= 2. three validation schemes (diagram)
const CVC = ['training', 'test', 'purged (target overlaps the test block)', 'embargo', 'not used yet'], CVK = [BLUE, RED, AMBER, ORANGE, '#FFFFFF'];
function schemes(k, h, e) {
  const n = 60, perm = D.cv_perm, rows = [], base = Math.floor(n / k), extra = n % k, folds = [];
  let p0 = 0; for (let i = 0; i < k; i++) { const sz = base + (i < extra ? 1 : 0); folds.push(perm.slice(p0, p0 + sz)); p0 += sz; }      // np.array_split
  for (let i = 0; i < k; i++) { const lab = new Array(n).fill(0); folds[i].forEach(j => (lab[j] = 1)); rows.push([`random K-fold, fold ${i + 1}`, lab, 0]); }
  const size = Math.floor(n / (k + 1));
  for (let i = 0; i < k; i++) { const lab = new Array(n).fill(4); for (let t = 0; t < size * (i + 1); t++) lab[t] = 0; for (let t = size * (i + 1); t < Math.min(n, size * (i + 2)); t++) lab[t] = 1; rows.push([`walk-forward, step ${i + 1}`, lab, 1]); }
  const block = Math.floor(n / k);
  for (let i = 0; i < k; i++) { const lab = new Array(n).fill(0), a = i * block, b = (i + 1) * block; for (let t = a; t < b; t++) lab[t] = 1; for (let t = Math.max(0, a - h); t < a; t++) lab[t] = 2; for (let t = b; t < Math.min(n, b + e); t++) lab[t] = 3; rows.push([`purged K-fold, fold ${i + 1}`, lab, 2]); }
  return { rows, folds, size, block };
}
function csBlock() {
  const k = +$('cs-k').value, h = +$('cs-h').value, e = +$('cs-e').value, S = schemes(k, h, e), nw = narrow('cs-chart'); label('cs-k').textContent = k; label('cs-h').textContent = h; label('cs-e').textContent = e;
  const z = [], yl = [];
  S.rows.forEach((r, j) => { if (j && S.rows[j - 1][2] !== r[2]) { z.push(new Array(60).fill(null)); yl.push(' '); } z.push(r[1]); yl.push(r[0]); });
  z.reverse(); yl.reverse();
  const cs = []; CVK.forEach((c, i) => { cs.push([i / 5, c], [(i + 1) / 5, c]); });
  $('cs-chart').style.height = (60 + 19 * yl.length + (nw ? 150 : 90)) + 'px';
  const tr = [{ type: 'heatmap', z, x: lin(1, 60, 60), y: yl.map((_, i) => i), zmin: -0.5, zmax: 4.5, colorscale: cs, showscale: false, xgap: 1, ygap: 2, hovertemplate: 'observation %{x}<extra></extra>', name: 'schemes' }];
  CVC.forEach((nm, i) => tr.push({ x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: CVK[i], line: { color: BLUE, width: i === 4 ? 1 : 0 } }, name: nm, hoverinfo: 'skip' }));
  plot('cs-chart', tr, layout({ xaxis: AX('time (observation)', { range: [0.5, 60.5], showgrid: false }), yaxis: AX('', { tickvals: yl.map((_, i) => i), ticktext: yl, showgrid: false, ticks: '', tickfont: { size: 10, color: DARK } }),
    legend: Object.assign(layout().legend, { y: nw ? -0.2 : -0.13 }), margin: { l: 150, r: 12, t: 10, b: 40 } }));
  const f1 = S.rows[0][1], nb = f1.filter((v, t) => v === 1 && ((t > 0 && f1[t - 1] === 0) || (t < 59 && f1[t + 1] === 0))).length, mid = S.rows[2 * k + Math.floor(k / 2)][1];
  out('cs-out', [T.cs(k, nb, sum(f1), S.size, k * S.size, h, e, mid.filter(v => v === 0).length, k !== 5 || h !== 3 || e !== 2)]);
}

// ================================================================= x1. walk-forward validation, step by step (table)
function wfBlock() {
  const yr = +$('wf-y').value, h = kSel('wf-h'), W = D.design[h], R = W.rows, r = R.find(q => q.year === yr), nw = single('wf-chart', 330); label('wf-y').textContent = yr;
  const x = R.map(q => q.year);
  plot('wf-chart', [{ type: 'bar', x, y: R.map(q => q.n_train), marker: { color: BLUE }, name: 'training days (purged rows removed)', hovertemplate: '%{x}: %{y} training days<extra></extra>' },
    { type: 'bar', x, y: R.map(q => +h), marker: { color: AMBER }, name: `purged rows (${h})`, hovertemplate: '%{x}: %{y} purged<extra></extra>' },
    { type: 'bar', x, y: R.map(q => q.n_test), marker: { color: RED }, name: 'test days (the calendar year)', hovertemplate: '%{x}: %{y} test days<extra></extra>' },
    { x: [yr], y: [r.n_train + r.n_test + +h], mode: 'markers+text', marker: { color: GREEN, size: 12, symbol: 'triangle-down' }, text: [`${r.n_train} + ${h} + ${r.n_test}`], textposition: 'top center', textfont: { size: 11, color: DARK }, cliponaxis: false, name: `fold ${yr}`, hoverinfo: 'skip' }],
  layout({ barmode: 'stack', xaxis: AX('test year', { dtick: 1, tickangle: -45 }), yaxis: AX('days', { range: [0, 5200] }), legend: LEG(nw, -0.3), margin: M0 }));
  table('wf-tab', T.h.wf, R.map(q => [{ v: q.year, c: q.year === yr ? 'hi' : '' }, { v: fint(q.n_train), c: q.year === yr ? 'hi' : '' }, q.n_test, T.date(q.last_train), T.date(q.first_purged)]));
  steps('wf-st', [T.wf1(yr, T.date(W.first), T.date(r.last_train), fint(r.n_train)), T.wf2(h, T.date(r.first_purged), yr), T.wf3(r.n_test, yr)]);
  out('wf-out', [T.wf(yr, fint(r.n_train), r.n_test, h, W.n_folds, h !== '5')]);
}

// ================================================================= 3. leakage: random folds on overlapping targets
function lkBlock() {
  const h = kSel('lk-h'), L = D.leak[h], nw = single('lk-chart', 360), xs = ['simulated random walk<br>(nothing to predict)', 'S&P 500, 2000-2026'];
  const tr = [['kfold', 'random 5-fold (shuffled)', RED], ['wf', 'walk-forward', BLUE], ['purged', `walk-forward, purged (gap of ${h} days)`, GREEN]].map(([k, nm, c]) => {
    const v = ['sim', 'sp500'].map(s => 100 * L[s][k]);
    return { type: 'bar', x: xs, y: v, marker: { color: c }, name: nm, text: v.map(q => q.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, hovertemplate: `${nm}: %{y:.1f}%<extra></extra>` };
  });
  const lo = Math.min(...['sim', 'sp500'].flatMap(s => ['kfold', 'wf', 'purged'].map(k => 100 * L[s][k]))), hi = Math.max(...['sim', 'sp500'].map(s => 100 * L[s].kfold));
  plot('lk-chart', tr, layout({ barmode: 'group', xaxis: AX('', { type: 'category' }), yaxis: AX('out-of-sample R² (%)', { zeroline: true, zerolinecolor: DARK, range: [lo - 14, hi + 10] }), legend: LEG(nw, -0.18), margin: M0 }));
  const p = v => pc(v, 1);
  out('lk-out', [T.lk(h, p(L.sim.kfold), p(L.sim.wf), p(L.sim.purged), p(L.sp500.kfold), p(L.sp500.wf), p(L.sp500.purged), fint(L.sim.n), fint(L.sp500.n), h !== '21')]);
}

// ================================================================= 4. ridge and lasso paths
const SH = D.shrink;
const ridgeAt = lam => solve(SH.G.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0))), SH.c);
let RPATH = null;
const ridgePath = () => RPATH || (RPATH = SH.lam_r.map(ridgeAt));
const OLSB = () => D.ext.map(g => SH.ols[g]);   // r, min(r, 0) and |r| are collinear: OLS is the minimum-norm solution of the chapter (sklearn)
function shBlock() {
  const i = +$('sh-i').value, f = kSel('sh-f'), j = D.ext.indexOf(f), P = panels('sh-chart', 2, { h: 400 }), RP = ridgePath(), lr = SH.lam_r[i], ll = SH.lam_l[i];
  label('sh-i').textContent = `${nf(ll, 4)} (lasso), ${nf(lr, 1)} (ridge)`;
  const tr = [];
  D.ext.forEach((g, q) => {
    tr.push(on2(P.ax[0], { x: SH.lam_r, y: RP.map(r => r[q]), mode: 'lines', line: { color: FC[g], width: g === f ? 2.8 : 1.4 }, name: FL[g], legendgroup: g, hovertemplate: `${FL[g]}: %{y:.3f}<extra></extra>` }));
    tr.push(on2(P.ax[1], { x: SH.lam_l, y: SH.lasso.map(r => r[q]), mode: 'lines', line: { color: FC[g], width: g === f ? 2.8 : 1.4 }, name: FL[g], legendgroup: g, showlegend: false, hovertemplate: `${FL[g]}: %{y:.3f}<extra></extra>` }));
  });
  const rb = ridgeAt(lr), lb = SH.lasso[i];
  [[0, lr, rb], [1, ll, lb]].forEach(([a, lam, b]) => {
    tr.push(on2(P.ax[a], { x: D.ext.map(() => lam), y: b, mode: 'markers', marker: { color: D.ext.map(g => FC[g]), size: 7, line: { color: DARK, width: 0.6 } }, name: 'coefficients at the chosen penalty', showlegend: a === 0, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[a], { x: [lam], y: [b[j]], mode: 'markers+text', marker: { color: FC[f], size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [b[j].toFixed(3)], textposition: 'middle right', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' }));
  });
  const yr = [-0.15, 0.4];
  tr.push(on2(P.ax[1], { x: [SH.cv_alpha, SH.cv_alpha], y: yr, mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'lasso penalty chosen by walk-forward CV', hoverinfo: 'skip' }));
  setAx(P, 0, AX('ridge penalty λ (log scale)', { type: 'log' }), AX('coefficient (standardised features)', { zeroline: true, zerolinecolor: DARK }));
  setAx(P, 1, AX('lasso penalty λ (log scale)', { type: 'log' }), AX('', { zeroline: true, zerolinecolor: DARK }));
  plot('sh-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const ob = OLSB(), nz = lb.filter(v => Math.abs(v) > 1e-10).length, nrm = v => Math.sqrt(sum(v.map(q => q * q)));
  const cv = i === SH.cv_i, cvc = D.ext.map(g => SH.cv_coef[g]);
  out('sh-out', [T.sh(nf(ll, 4), nz, T.fl(f), nf(cv ? cvc[j] : lb[j], 3), nf(ob[j], 3), nf(lr, 1), nf(rb[j], 3), nf(nrm(rb) / nrm(ob), 2), cv, cv ? cvc.filter(v => Math.abs(v) > 1e-10).length : nz)]);
}

// ================================================================= x2. ridge with one variable (worked example)
function rwBlock() {
  const lam = +$('rw-l').value, sxx = +kSel('rw-s'), sxy = 30, nw = single('rw-chart', 320), ls = lin(0, 200, 201); label('rw-l').textContent = lam;
  const ridge = l => sxy / (sxx + l), lasso = l => Math.sign(sxy) * Math.max(Math.abs(sxy) - l / 2, 0) / sxx;
  plot('rw-chart', [{ x: ls, y: ls.map(ridge), mode: 'lines', line: { color: BLUE, width: 2.2 }, name: 'ridge: S_xy / (S_xx + λ)', hovertemplate: 'λ %{x}: %{y:.3f}<extra></extra>' },
    { x: ls, y: ls.map(lasso), mode: 'lines', line: { color: RED, width: 2, dash: 'dash' }, name: 'lasso: soft thresholding', hovertemplate: 'λ %{x}: %{y:.3f}<extra></extra>' },
    { x: [0, 200], y: [sxy / sxx, sxy / sxx], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: 'OLS: S_xy / S_xx', hoverinfo: 'skip' },
    { x: [lam], y: [ridge(lam)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [ridge(lam).toFixed(3)], textposition: 'top right', textfont: { size: 11, color: DARK }, name: `λ = ${lam}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('penalty λ'), yaxis: AX('coefficient', { range: [0, 0.32 * 100 / Math.min(sxx, 100) + 0.02] }), legend: LEG(nw, -0.2), margin: M0 }));
  const b = ridge(lam), fct = sxx / (sxx + lam);
  steps('rw-we', [T.rw1(sxx, sxy, nf(sxy / sxx, 2)), T.rw2(lam, sxx, sxy, nf(b, 3), nf(fct, 3)), T.rw3(nf(lasso(lam), 3), lam)]);
  out('rw-out', [T.rw(lam, nf(b, 3), nf(fct, 2), lam !== 50 || sxx !== 100)]);
}

// ================================================================= x3. lasso with one variable: soft thresholding (worked example)
function lwBlock() {
  const lam = +$('lw-l').value, sxy = +kSel('lw-s'), sxx = 100, nw = single('lw-chart', 320), xs = lin(-60, 60, 241); label('lw-l').textContent = lam;
  const lasso = s => Math.sign(s) * Math.max(Math.abs(s) - lam / 2, 0) / sxx, b = lasso(sxy);
  plot('lw-chart', [{ x: xs, y: xs.map(s => s / sxx), mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: 'OLS: S_xy / S_xx', hoverinfo: 'skip' },
    { x: xs, y: xs.map(s => s / (sxx + lam)), mode: 'lines', line: { color: BLUE, width: 1.8 }, name: `ridge, λ = ${lam}`, hovertemplate: 'S_xy %{x}: %{y:.3f}<extra></extra>' },
    { x: xs, y: xs.map(lasso), mode: 'lines', line: { color: RED, width: 2.4 }, name: `lasso, λ = ${lam}: zero for |S_xy| ≤ ${lam / 2}`, hovertemplate: 'S_xy %{x}: %{y:.3f}<extra></extra>' },
    { x: [sxy], y: [b], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [b.toFixed(3)], textposition: sxy > 0 ? 'top left' : 'bottom right', textfont: { size: 11, color: DARK }, name: `S_xy = ${sxy}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('S_xy (S_xx = 100)'), yaxis: AX('coefficient', { range: [-0.65, 0.65], zeroline: true, zerolinecolor: DARK }), legend: LEG(nw, -0.2), margin: M0 }));
  steps('lw-we', [T.lw1(sxy, lam, nf(lam / 2, 1)), T.lw2(sxy, lam, nf(Math.max(Math.abs(sxy) - lam / 2, 0), 1), nf(b, 3)), T.lw3(2 * Math.abs(sxy))]);
  out('lw-out', [T.lw(lam, sxy, nf(b, 3), lam !== 20 || sxy !== 30)]);
}

// ================================================================= x4. one split, step by step (worked example of the slide)
const SX = [0.6, 0.8, 1.0, 1.5, 2.0, 2.4], SY = [0.7, 0.9, 0.8, 1.6, 2.1, 2.2];   // the six weeks of the slide
function sse(v) { const m = mean(v); return sum(v.map(q => (q - m) ** 2)); }
function splits(x, y, minLeaf) {
  const out = []; for (let k = 1; k < x.length; k++) { const L = y.slice(0, k), R = y.slice(k), ok = L.length >= minLeaf && R.length >= minLeaf;
    out.push({ s: (x[k - 1] + x[k]) / 2, nL: L.length, nR: R.length, yL: mean(L), yR: mean(R), sse: sse(L) + sse(R), ok }); }
  return out;
}
function spBlock() {
  const s = +$('sp-s').value, ml = +kSel('sp-m'), C0 = splits(SX, SY, ml), c = C0.find(q => Math.abs(q.s - s) < 1e-9), P = panels('sp-chart', 2, { h: 340 });
  const ok = C0.filter(q => q.ok), best = ok.reduce((a, q) => (q.sse < a.sse ? q : a)), sse0 = sse(SY);
  const tr = [on2(P.ax[0], { x: SX, y: SY, mode: 'markers', marker: { color: BLUE, size: 10 }, name: 'the six weeks', hovertemplate: 'x %{x}: y %{y}<extra></extra>' }),
    on2(P.ax[0], { x: [0.5, s, s, 2.5], y: [c.yL, c.yL, c.yR, c.yR], mode: 'lines', line: { color: c.ok ? GREEN : RED, width: 2.4, shape: 'hv' }, name: `leaf means for s = ${s.toFixed(2)}`, hoverinfo: 'skip' }),
    on2(P.ax[0], { x: [0.5, 2.5], y: [mean(SY), mean(SY)], mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dot' }, name: 'no split: mean 1.38', hoverinfo: 'skip' }),
    on2(P.ax[1], { type: 'bar', x: C0.map(q => q.s.toFixed(2)), y: C0.map(q => q.sse), marker: { color: C0.map(q => (!q.ok ? '#FFFFFF' : Math.abs(q.s - s) < 1e-9 ? GREEN : BLUE)), line: { color: BLUE, width: 1 } },
      text: C0.map(q => (q.ok ? q.sse.toFixed(3) : 'not allowed')), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, name: 'SSE of the split', hovertemplate: 's = %{x}: SSE %{y:.3f}<extra></extra>' }),
    on2(P.ax[1], { x: ['0.70', '2.20'], y: [sse0, sse0], mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dot' }, name: 'SSE without a split (2.268)', hoverinfo: 'skip', showlegend: false }),
    on2(P.ax[1], { x: [c.s.toFixed(2)], y: [c.sse + 0.32], mode: 'markers', marker: { color: c.ok ? GREEN : RED, size: 12, symbol: 'triangle-down' }, name: `chosen s = ${s.toFixed(2)}`, hoverinfo: 'skip', showlegend: false })];
  setAx(P, 0, AX('x: volatility this week (%)'), AX('y: volatility next week (%)')); setAx(P, 1, AX('candidate threshold s', { type: 'category' }), AX('SSE', { range: [0, 2.6] }));
  plot('sp-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.1 : -0.22), margin: M0 })));
  table('sp-tab', T.h.sp, C0.map(q => { const cl = Math.abs(q.s - s) < 1e-9 ? 'hi' : q.ok ? '' : 'bad';
    return [{ v: nf(q.s, 2), c: cl }, { v: nf(q.yL, 2), c: cl }, { v: nf(q.yR, 2), c: cl }, { v: q.ok ? nf(q.sse, 3) : T.h.na, c: cl }]; }));
  steps('sp-we', [T.sp1(nf(mean(SY), 2), nf(sse0, 3)), T.sp2(nf(s, 2), c.nL, c.nR, nf(c.yL, 2), nf(c.yR, 2), nf(c.sse, 3), c.ok, ml), T.sp3(nf(best.s, 2), nf(best.sse, 3), nf(sse0, 3), nf(1 - best.sse / sse0, 2))]);
  out('sp-out', [T.sp(nf(s, 2), c.ok ? nf(c.sse, 3) : null, nf(1 - c.sse / sse0, 2), nf(best.s, 2), ml !== 1)]);
}

// ================================================================= 5. a depth-2 tree for next week's volatility
const TD = D.tree;
function leafOf(nd, i) { let k = 0; while (nd.l[k] >= 0) k = TD.X[nd.f[k]][i] <= nd.s[k] ? nd.l[k] : nd.r[k]; return k; }
function trBlock() {
  const dep = kSel('tr-d'), ml = kSel('tr-l'), nd = TD.trees[dep + '|' + ml], P = panels('tr-chart', 2, { h: 420, gap: 0.08 }), nn = nd.f.length;
  const depth = new Array(nn).fill(0), xpos = new Array(nn).fill(0); let leafX = 0;
  (function walk(k, d) { depth[k] = d; if (nd.l[k] < 0) { xpos[k] = leafX++; return; } walk(nd.l[k], d + 1); walk(nd.r[k], d + 1); xpos[k] = (xpos[nd.l[k]] + xpos[nd.r[k]]) / 2; })(0, 0);
  const maxd = Math.max(...depth), ex = [], ey = [];
  for (let k = 0; k < nn; k++) if (nd.l[k] >= 0) [nd.l[k], nd.r[k]].forEach(c => { ex.push(xpos[k], xpos[c], null); ey.push(-depth[k], -depth[c], null); });
  const SHORT = { d: 'day', w: 'week', m: 'month' }, cmp = maxd > 2 || P.stack;
  const ntext = k => (cmp ? (nd.l[k] >= 0 ? `${SHORT[D.har[nd.f[k]]]} <= ${nd.s[k].toFixed(2)}<br>` : '') + `n ${nd.n[k]}<br>${nd.v[k].toFixed(2)}`
    : (nd.l[k] >= 0 ? `${FL[D.har[nd.f[k]]]} <= ${nd.s[k].toFixed(2)}<br>` : '') + `samples = ${nd.n[k]}<br>value = ${nd.v[k].toFixed(2)}`);
  const tr = [on2(P.ax[0], { x: ex, y: ey, mode: 'lines', line: { color: DARK, width: 1 }, hoverinfo: 'skip', showlegend: false }),
    on2(P.ax[0], { x: xpos, y: depth.map(d => -d), mode: 'markers', marker: { color: depth.map((d, k) => (nd.l[k] < 0 ? GREEN : BLUE)), size: 8 }, text: nd.f.map((_, k) => ntext(k)), hovertemplate: '%{text}<extra></extra>', name: 'node (blue) and leaf (green)' })];
  const ann = nd.f.map((_, k) => ({ x: xpos[k], y: -depth[k], xref: P.ax[0].x, yref: P.ax[0].y, text: ntext(k), showarrow: false, font: { size: cmp ? 9 : 10, color: DARK }, bgcolor: '#FFFFFF', bordercolor: nd.l[k] < 0 ? GREEN : BLUE, borderpad: 3, borderwidth: 1 }));
  const leaves = []; for (let k = 0; k < nn; k++) if (nd.l[k] < 0) leaves.push(k);
  const LC = [BLUE, RED, GREEN, AMBER, PURPLE, TEAL, ORANGE, DARK], li = TD.y.map((_, i) => leafOf(nd, i)), w = TD.X[1];
  leaves.forEach((k, q) => { const idx = li.map((v, i) => (v === k ? i : -1)).filter(i => i >= 0);
    tr.push(on2(P.ax[1], { x: idx.map(i => w[i]), y: idx.map(i => TD.y[i]), mode: 'markers', marker: { color: hexA(LC[q % 8], 0.45), size: 4 }, name: `leaf ${q + 1}: mean ${nd.v[k].toFixed(2)} (n = ${nd.n[k]})`, hovertemplate: 'log RV week %{x:.2f}: target %{y:.2f}<extra></extra>' }));
    const xs = idx.map(i => w[i]); tr.push(on2(P.ax[1], { x: [Math.min(...xs), Math.max(...xs)], y: [nd.v[k], nd.v[k]], mode: 'lines', line: { color: LC[q % 8], width: 3 }, showlegend: false, hoverinfo: 'skip' })); });
  setAx(P, 0, AX('', { showgrid: false, zeroline: false, showticklabels: false, range: [-0.7, leafX - 0.3] }), AX('', { showgrid: false, zeroline: false, showticklabels: false, range: [-maxd - 0.45, 0.45] }));
  setAx(P, 1, AX('log RV week (feature at day t)'), AX('target: log RV of the next 5 days'));
  plot('tr-chart', tr, layout(Object.assign(P.L, { annotations: ann, legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const ym = mean(TD.y), sst = sum(TD.y.map(v => (v - ym) ** 2)), sres = sum(TD.y.map((v, i) => (v - nd.v[li[i]]) ** 2));
  out('tr-out', [T.tr(dep, ml, T.fl(D.har[nd.f[0]]), nf(nd.s[0], 2), nf(Math.sqrt(252 * Math.exp(nd.s[0])), 1), leaves.map(k => [nf(nd.v[k], 2), nd.n[k]]), nf(1 - sres / sst, 2), fint(TD.y.length), dep !== '2' || ml !== '50')]);
}

// ================================================================= x5. random forest: the variance of an average (worked example)
function bgBlock() {
  const B = +$('bg-B').value, rho = +kSel('bg-r'), s2 = 1, nw = single('bg-chart', 320), Bs = Array.from({ length: 300 }, (_, i) => i + 1); label('bg-B').textContent = B;
  const v = b => rho * s2 + (1 - rho) * s2 / b;
  plot('bg-chart', [{ x: Bs, y: Bs.map(v), mode: 'lines', line: { color: GREEN, width: 2.2 }, name: `Var(mean of B trees), ρ = ${rho}`, hovertemplate: 'B = %{x}: %{y:.3f}<extra></extra>' },
    { x: [1, 300], y: [rho * s2, rho * s2], mode: 'lines', line: { color: RED, width: 1.4, dash: 'dash' }, name: `floor ρσ² = ${rho}`, hoverinfo: 'skip' },
    { x: [B], y: [v(B)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [v(B).toFixed(3)], textposition: 'top right', textfont: { size: 11, color: DARK }, name: `B = ${B}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('number of trees B (log scale)', { type: 'log' }), yaxis: AX('variance of the average (σ² = 1)', { range: [0, 1.05] }), legend: LEG(nw, -0.2), margin: M0 }));
  steps('bg-we', [T.bg1(nf(rho, 1), B), T.bg2(nf(rho, 1), B, nf(rho * s2, 3), nf((1 - rho) / B, 3), nf(v(B), 3)), T.bg3(nf(rho, 1), nf(v(1), 3), nf(v(10), 3), nf(v(300), 3))]);
  out('bg-out', [T.bg(B, nf(rho, 1), nf(v(B), 3), rho !== 0.3)]);
}

// ================================================================= x6. gradient boosting on the six weeks (extra computation)
function stump(x, r) { let best = null; for (let k = 1; k < x.length; k++) { const L = r.slice(0, k), R = r.slice(k), e = sse(L) + sse(R); if (!best || e < best.sse - 1e-12) best = { s: (x[k - 1] + x[k]) / 2, mL: mean(L), mR: mean(R), sse: e }; } return best; }
function boost(nu, M) {
  const F = [SY.map(() => mean(SY))], H = []; let cur = F[0].slice();
  for (let m = 1; m <= M; m++) { const r = SY.map((y, i) => y - cur[i]), h = stump(SX, r); H.push(h); cur = cur.map((f, i) => f + nu * (SX[i] <= h.s ? h.mL : h.mR)); F.push(cur.slice()); }
  return { F, H, sse: F.map(f => sum(SY.map((y, i) => (y - f[i]) ** 2))) };
}
function gbBlock() {
  const M = +$('gb-m').value, nu = +kSel('gb-v'), B = boost(nu, 50), P = panels('gb-chart', 2, { h: 340 }); label('gb-m').textContent = M;
  const fM = B.F[M], xs = [0.5], ys = [fM[0]]; SX.forEach((x, i) => { if (i) { const s = (SX[i - 1] + x) / 2; xs.push(s); ys.push(fM[i]); } }); xs.push(2.5); ys.push(fM[5]);
  const tr = [on2(P.ax[0], { x: SX, y: SY, mode: 'markers', marker: { color: BLUE, size: 10 }, name: 'the six weeks', hoverinfo: 'skip' }),
    on2(P.ax[0], { x: [0.5, 2.5], y: [mean(SY), mean(SY)], mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dot' }, name: 'F₀ = mean', hoverinfo: 'skip' }),
    on2(P.ax[0], { x: xs, y: ys, mode: 'lines', line: { color: GREEN, width: 2.4, shape: 'hv' }, name: `F_${M} after ${M} steps`, hovertemplate: 'F: %{y:.3f}<extra></extra>' }),
    on2(P.ax[1], { x: B.sse.map((_, m) => m), y: B.sse, mode: 'lines', line: { color: PURPLE, width: 2 }, name: `training SSE, ν = ${nu}`, hovertemplate: 'step %{x}: SSE %{y:.3f}<extra></extra>' }),
    on2(P.ax[1], { x: [M], y: [B.sse[M]], mode: 'markers+text', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [B.sse[M].toFixed(3)], textposition: 'top right', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' })];
  setAx(P, 0, AX('x: volatility this week (%)'), AX('y and the boosted fit')); setAx(P, 1, AX('boosting step m'), AX('training SSE', { range: [0, 2.4] }));
  plot('gb-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.1 : -0.22), margin: M0 })));
  const h1 = B.H[0], F1 = B.F[1];
  steps('gb-st', [T.gb1(nf(mean(SY), 3), nf(B.sse[0], 3)), T.gb2(nf(h1.s, 2), nf(h1.mL, 3), nf(h1.mR, 3)), T.gb3(nf(nu, 2), nf(F1[0], 3), nf(F1[5], 3), nf(B.sse[1], 3)), T.gb4(M, nf(B.sse[M], 3), nf(fM[0], 3), nf(fM[5], 3))]);
  out('gb-out', [T.gb(nf(nu, 2), M, nf(B.sse[M], 3))]);
}

// ================================================================= 6. one tree, a forest and boosting on volatility data
const EN = D.ens;
function enBlock() {
  const n = +$('en-n').value, d = +$('en-d').value, Bt = Math.min(n, 200), P = panels('en-chart', 2, { h: 390 }); label('en-n').textContent = n; label('en-d').textContent = d;
  const rf = EN.rfB.map(r => r[Bt - 1]), bd = rf.indexOf(Math.min(...rf)) + 1, LRC = { '0.3': RED, '0.1': PURPLE, '0.03': ORANGE };
  const tr = [on2(P.ax[0], { x: EN.depth, y: EN.tree, mode: 'lines+markers', line: { color: AMBER, width: 2 }, marker: { size: 6 }, name: 'single tree', hovertemplate: 'depth %{x}: %{y:.3f}<extra></extra>' }),
    on2(P.ax[0], { x: EN.depth, y: rf, mode: 'lines+markers', line: { color: GREEN, width: 2 }, marker: { size: 6 }, name: `random forest (${Bt} trees)`, hovertemplate: 'depth %{x}: %{y:.3f}<extra></extra>' }),
    on2(P.ax[0], { x: [d], y: [rf[d - 1]], mode: 'markers+text', marker: { color: GREEN, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [rf[d - 1].toFixed(3)], textposition: 'top center', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' }),
    on2(P.ax[0], { x: [1, 14], y: [EN.har, EN.har], mode: 'lines', line: { color: BLUE, width: 1.4, dash: 'dot' }, name: 'HAR regression', hoverinfo: 'skip' })];
  const nn = Array.from({ length: 500 }, (_, i) => i + 1);
  Object.entries(EN.gb).forEach(([lr, g]) => {
    tr.push(on2(P.ax[1], { x: nn, y: g.val, mode: 'lines', line: { color: LRC[lr], width: 1.6 }, name: `boosting, learning rate ${lr}`, hovertemplate: `ν ${lr}, %{x} trees: %{y:.3f}<extra></extra>` }));
    tr.push(on2(P.ax[1], { x: nn, y: g.train, mode: 'lines', line: { color: LRC[lr], width: 1, dash: 'dash' }, showlegend: false, hoverinfo: 'skip' }));
  });
  tr.push(on2(P.ax[1], { x: [1, 500], y: [EN.har, EN.har], mode: 'lines', line: { color: BLUE, width: 1.4, dash: 'dot' }, showlegend: false, hoverinfo: 'skip' }));
  tr.push(on2(P.ax[1], { x: Object.keys(EN.gb).map(() => n), y: Object.values(EN.gb).map(g => g.val[n - 1]), mode: 'markers', marker: { color: Object.keys(EN.gb).map(l => LRC[l]), size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: Object.values(EN.gb).map(g => g.val[n - 1].toFixed(3)), name: `${n} trees`, hovertemplate: '%{text}<extra></extra>', showlegend: false }));
  const allTrain = Object.values(EN.gb).map(g => Math.min(...g.train));
  setAx(P, 0, AX('maximum depth', { dtick: 1 }), AX('validation MSE, log RV')); setAx(P, 1, AX('number of trees'), AX('', { range: [Math.min(...allTrain) * 0.9, Math.max(...EN.tree) * 1.05] }));
  plot('en-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: M0 })));
  const G = k => EN.gb[k].val, gbest = k => { const v = G(k); const m = Math.min(...v); return [nf(m, 3), v.indexOf(m) + 1]; };
  out('en-out', [T.en(n, Bt, d, nf(rf[d - 1], 3), bd, nf(rf[bd - 1], 3), nf(EN.tree[d - 1], 3), ['0.3', '0.1', '0.03'].map(k => [k, nf(G(k)[n - 1], 3), ...gbest(k)]), nf(EN.har, 3), fint(EN.n_train), fint(EN.n_val), Bt !== 200)]);
}

// ================================================================= x7. number of parameters of an MLP (worked example)
function mwBlock() {
  const K = +$('mw-k').value, p = +kSel('mw-p'), nw = single('mw-chart', 320), Ks = Array.from({ length: 64 }, (_, i) => i + 1), cnt = (pp, k) => pp * k + k + k + 1; label('mw-k').textContent = K;
  const tr = [...new Set([3, 11, p])].map((pp, q) => ({ x: Ks, y: Ks.map(k => cnt(pp, k)), mode: 'lines', line: { color: [BLUE, RED, GREEN][q], width: pp === p ? 2.6 : 1.4 }, name: `${pp} inputs`, hovertemplate: `${pp} inputs, K = %{x}: %{y} parameters<extra></extra>` }));
  tr.push({ x: [K], y: [cnt(p, K)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [String(cnt(p, K))], textposition: 'top left', textfont: { size: 11, color: DARK }, name: `${p} inputs, K = ${K}`, hoverinfo: 'skip' });
  plot('mw-chart', tr, layout({ xaxis: AX('hidden neurons K'), yaxis: AX('number of parameters'), legend: LEG(nw, -0.2), margin: M0 }));
  steps('mw-we', [T.mw1(p, K, p * K, K, p * K + K), T.mw2(K, K + 1), T.mw3(cnt(p, K))]);
  out('mw-out', [T.mw(p, K, cnt(p, K), p !== 11 || K !== 16)]);
}

// ================================================================= 7. a small network and its activation functions
const ACT = { logistic: z => 1 / (1 + Math.exp(-z)), tanh: z => Math.tanh(z), relu: z => Math.max(0, z) };
const DACT = { logistic: z => ACT.logistic(z) * (1 - ACT.logistic(z)), tanh: z => 1 - Math.tanh(z) ** 2, relu: z => +(z > 0) };
const ACTN = { logistic: 'logistic 1/(1+e^-z)', tanh: 'tanh z', relu: 'ReLU max(0, z)' }, ACTC = { logistic: BLUE, tanh: RED, relu: GREEN };
function mlBlock() {
  const g = kSel('ml-g'), z = +$('ml-z').value, K = +$('ml-k').value, P = panels('ml-chart', 2, { h: 380, gap: 0.08 }); label('ml-z').textContent = nf(z, 2); label('ml-k').textContent = K;
  const lay = [['log RV day', 'log RV week', 'log RV month'], new Array(K).fill(''), ['forecast']], xs = [0.1, 0.5, 0.9];
  const pos = lay.map((l, i) => (l.length > 1 ? lin(0.85, 0.15, l.length) : [0.5]).map(y => [xs[i], y]));
  const ex = [], ey = []; for (let a = 0; a < 2; a++) pos[a].forEach(p => pos[a + 1].forEach(q => { ex.push(p[0], q[0], null); ey.push(p[1], q[1], null); }));
  const tr = [on2(P.ax[0], { x: ex, y: ey, mode: 'lines', line: { color: TEAL, width: 0.8 }, hoverinfo: 'skip', showlegend: false })];
  [['input', BLUE], ['hidden neuron', GREEN], ['output', RED]].forEach(([nm, c], i) => tr.push(on2(P.ax[0], { x: pos[i].map(p => p[0]), y: pos[i].map(p => p[1]), mode: 'markers+text', marker: { color: c, size: 18, line: { color: DARK, width: 1 } },
    text: lay[i], textposition: i === 0 ? 'middle left' : 'middle right', textfont: { size: 10, color: DARK }, name: nm, hoverinfo: 'skip', cliponaxis: false })));
  tr.push(on2(P.ax[0], { x: [0.5, 0.9], y: [0.98, 0.33], mode: 'text', text: [`hidden layer: h_j = g(b_j + w_jᵀx), g = ${g === 'relu' ? 'ReLU' : g}`, 'output: ŷ = c + vᵀh'], textfont: { size: 10.5, color: DARK }, hoverinfo: 'skip', showlegend: false, cliponaxis: false }));
  const zs = lin(-4, 4, 401);
  Object.keys(ACT).forEach(k => tr.push(on2(P.ax[1], { x: zs, y: zs.map(ACT[k]), mode: 'lines', line: { color: ACTC[k], width: k === g ? 2.6 : 1.6 }, name: ACTN[k], hovertemplate: `${k}: %{y:.3f}<extra></extra>` })));
  tr.push(on2(P.ax[1], { x: [z], y: [ACT[g](z)], mode: 'markers+text', marker: { color: ACTC[g], size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`g(${z}) = ${ACT[g](z).toFixed(3)}`], textposition: 'top left', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' }));
  setAx(P, 0, AX('', { showgrid: false, zeroline: false, showticklabels: false, range: [-0.3, 1.25] }), AX('', { showgrid: false, zeroline: false, showticklabels: false, range: [0, 1.04] }));
  setAx(P, 1, AX('z'), AX('g(z)', { range: [-1.2, 2.5], zeroline: true, zerolinecolor: DARK }));
  plot('ml-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.18), margin: M0 })));
  out('ml-out', [T.ml(g, nf(z, 2), nf(ACT[g](z), 3), nf(DACT[g](z), 3), K, 3 * K + K + K + 1)]);
}

// ================================================================= 8. neural-network ARCH for GBP/USD
function naBlock() {
  const per = kSel('na-p'), u = kSel('na-u'), A = D.nnarch, ds = dates(A), idx = A.y.map((_, i) => i).filter(i => per === 'all' || yearOf(ds[i]) === +per), P = panels('na-chart', 2, { rows: true, hs: 220 });
  const X = idx.map(i => ds[i]), vR = idx.map(i => ANN * A.units[u].vol[i]), vG = idx.map(i => ANN * A.garch[i]), vM = idx.map(i => ANN * A.mlp[i]);
  const tr = [on2(P.ax[0], { x: X, y: idx.map(i => A.y[i]), mode: 'lines', line: { color: BLUE, width: 0.6 }, name: 'GBP/USD daily log return (%)', hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }),
    on2(P.ax[1], { x: X, y: vG, mode: 'lines', line: { color: RED, width: 1.1 }, name: 'GARCH(1,1)', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }),
    on2(P.ax[1], { x: X, y: vR, mode: 'lines', line: { color: GREEN, width: 1.1 }, name: `RBF network (SFEnnarch, ${u} units)`, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })];
  setAx(P, 0, AX('', { type: 'date' }), AX('return (%)')); setAx(P, 1, AX('', { type: 'date' }), AX('volatility (% per year)'));
  plot('na-chart', tr, layout(Object.assign(P.L, { legend: LEG(true, -0.06), margin: M0 })));
  const mx = v => { let k = 0; v.forEach((q, i) => { if (q > v[k]) k = i; }); return k; }, kR = mx(vR), kG = mx(vG);
  out('na-out', [T.na(per, u, nf(mean(vR), 1), nf(mean(vG), 1), nf(vR[kR], 1), T.date(X[kR]), nf(vG[kG], 1), T.date(X[kG]), nf(corr(vR, vG), 2), nf(corr(vM, vG), 2), fint(idx.length), u !== '3')]);
}

// ================================================================= 9. the RBF network against the random walk (USD/JPY)
function njBlock() {
  const sh = kSel('nj-s'), J = D.nnjpy, R = J.runs[sh], ds = dates(J), p = J.p, n = p.length, P = panels('nj-chart', 2, { rows: true, hs: 220 });
  const tX = ds.slice(3, R.n_tr), ytr = p.slice(3, R.n_tr), eX = ds.slice(R.n_tr), yte = p.slice(R.n_tr), rw = p.slice(R.n_tr - 1, n - 1);
  const tr = [on2(P.ax[0], { x: tX, y: ytr, mode: 'lines', line: { color: RED, width: 0.9 }, name: 'USD/JPY (yen per dollar)', hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    on2(P.ax[0], { x: tX, y: R.ftr, mode: 'lines', line: { color: BLUE, width: 0.9 }, name: 'RBF network, one day ahead', hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], { x: eX, y: yte, mode: 'lines', line: { color: RED, width: 0.9 }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], { x: eX, y: R.fte, mode: 'lines', line: { color: BLUE, width: 0.9 }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], { x: eX, y: rw, mode: 'lines', line: { color: GREEN, width: 0.9, dash: 'dash' }, name: 'random walk: tomorrow = today', hovertemplate: '%{x}: %{y:.2f}<extra></extra>' })];
  setAx(P, 0, AX('', { type: 'date' }), AX('yen per dollar')); setAx(P, 1, AX('', { type: 'date' }), AX('yen per dollar'));
  plot('nj-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], `training sample (first ${Math.round(100 * sh)}% of the days)`), ptitle(P.ax[1], `test sample (last ${Math.round(100 * (1 - sh))}% of the days)`)], legend: LEG(true, -0.06), margin: MG })));
  const rmse = (a, b) => Math.sqrt(mean(a.map((v, i) => (v - b[i]) ** 2))), rTr = rmse(R.ftr, ytr), rTe = rmse(R.fte, yte), wTr = rmse(p.slice(2, R.n_tr - 1), ytr), wTe = rmse(rw, yte);
  out('nj-out', [T.nj(Math.round(100 * sh), T.date(ds[R.n_tr]), nf(rTr, 2), nf(wTr, 2), nf(rTe, 2), nf(wTe, 2), nf(rTe / wTe, 0), nf(Math.max(...ytr), 1), nf(Math.max(...yte), 1), pc(R.r2_ret_te, 2), sh !== '0.8')]);
}

// ================================================================= x8. the HAR benchmark: one walk-forward fold (worked example)
function hwBlock() {
  const yr = +$('hw-y').value, day = kSel('hw-d'), H = harFold(yr), F = H.F, ds = H.ds, nw = single('hw-chart', 330); label('hw-y').textContent = yr;
  const te = H.te, hv = H.pred.map(p => Math.exp(p + H.s2 / 2)), j = day === 'first' ? 0 : day === 'last' ? te.length - 1 : hv.indexOf(Math.max(...hv)), i = te[j];
  plot('hw-chart', [{ x: te.map(q => ds[q]), y: te.map(q => Math.sqrt(252 * F.rv[q])), mode: 'lines', line: { color: DARK, width: 1.3 }, name: 'realised, next 5 days', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: te.map(q => ds[q]), y: hv.map(v => Math.sqrt(252 * v)), mode: 'lines', line: { color: BLUE, width: 1.6 }, name: `HAR forecast (fold ${yr})`, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: [ds[i]], y: [Math.sqrt(252 * hv[j])], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`${Math.sqrt(252 * hv[j]).toFixed(1)}%`], textposition: 'top right', textfont: { size: 11, color: DARK }, name: ds[i], hoverinfo: 'skip' }],
  layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('volatility (% per year)'), legend: LEG(nw, -0.2), margin: M0 }));
  const b = H.b, yh = H.pred[j];
  steps('hw-we', [T.hw1(yr, T.date(ds[H.tr[0]]), T.date(ds[H.tr[H.tr.length - 1]]), fint(H.tr.length), D.H),
    T.hw2(nf(b[0], 3), nf(b[1], 3), nf(b[2], 3), nf(b[3], 3), nf(b[1] + b[2] + b[3], 3)),
    T.hw3(T.date(ds[i]), nf(F.d[i], 3), nf(F.w[i], 3), nf(F.m[i], 3), nf(yh, 3)),
    T.hw4(nf(H.s2, 3), nf(hv[j], 3), nf(Math.sqrt(252 * hv[j]), 1)), T.hw5(nf(F.rv[i], 3), nf(Math.sqrt(252 * F.rv[i]), 1), nf(F.y[i], 3))]);
  out('hw-out', [T.hw(yr, fint(te.length), nf(Math.sqrt(mean(H.pred.map((p, q) => (F.y[te[q]] - p) ** 2))), 3), nf(b[1] + b[2] + b[3], 3))]);
}

// ================================================================= x9. out-of-sample R² (worked example)
function r2Block() {
  const m = kSel('r2-m'), bm = kSel('r2-b'), k = kSel('r2-a'), S = rvSet(k), R = S.R, nw = single('r2-chart', 330);
  const e2 = q => R.y.map((y, i) => (y - R.pred[q][i]) ** 2), em = e2(m), eb = e2(bm); let c = 0; const cum = em.map((v, i) => (c += eb[i] - v));
  plot('r2-chart', [{ x: S.ds, y: cum, mode: 'lines', line: { color: MC[m] || BLUE, width: 1.6 }, name: `cumulative (benchmark − model) squared error, ${MNAME[m]} against ${MNAME[bm]}`, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { x: [S.ds[0], S.ds[S.ds.length - 1]], y: [0, 0], mode: 'lines', line: { color: DARK, width: 1 }, hoverinfo: 'skip', showlegend: false },
    { x: [S.ds[S.ds.length - 1]], y: [cum[cum.length - 1]], mode: 'markers+text', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [cum[cum.length - 1].toFixed(1)], textposition: 'top left', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' }],
  layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('cumulative difference (log RV²)'), legend: LEG(nw, -0.2), margin: M0 }));
  const mm = mean(em), mb = mean(eb), r2 = 1 - mm / mb;
  steps('r2-st', [T.r21(ANAME[k], fint(R.y.length), MNAME[m], nf(mm, 3), MNAME[bm], nf(mb, 3)), T.r22(nf(mm, 3), nf(mb, 3), nf(mm / mb, 3), pc(r2, 1)), T.r23(r2 > 0)]);
  out('r2-out', [T.r2(ANAME[k], MNAME[m], MNAME[bm], pc(r2, 1))]);
}

// ================================================================= x10. QLIKE (worked example)
function qlBlock() {
  const h = +$('ql-h').value, rv = +kSel('ql-r'), hs = lin(0.1, 3.5, 341), nw = single('ql-chart', 330); label('ql-h').textContent = nf(h, 2);
  plot('ql-chart', [{ x: hs, y: hs.map(v => qlike(rv, v)), mode: 'lines', line: { color: BLUE, width: 2.2 }, name: `QLIKE: RV/h + ln h, RV = ${rv}`, hovertemplate: 'h %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: hs, y: hs.map(v => (rv - v) ** 2 + 1 + Math.log(rv)), mode: 'lines', line: { color: RED, width: 1.6, dash: 'dash' }, name: 'squared error (RV − h)², shifted to the same minimum', hovertemplate: 'h %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [h], y: [qlike(rv, h)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [qlike(rv, h).toFixed(3)], textposition: 'top right', textfont: { size: 11, color: DARK }, name: `forecast h = ${h}`, hoverinfo: 'skip' },
    { x: [rv], y: [1 + Math.log(rv)], mode: 'markers', marker: { color: GREEN, size: 10 }, name: 'minimum at h = RV', hoverinfo: 'skip' }],
  layout({ xaxis: AX('variance forecast h'), yaxis: AX('loss', { range: [Math.min(0, 1 + Math.log(rv)) - 0.2, 1 + Math.log(rv) + 2.2] }), legend: LEG(nw, -0.2), margin: M0 }));
  const lo = rv / 2, hi = 2 * rv;
  steps('ql-we', [T.ql1(rv, nf(h, 2), nf(rv / h, 3), nf(Math.log(h), 3), nf(qlike(rv, h), 3)), T.ql2(nf(lo, 2), nf(qlike(rv, lo), 3), nf(hi, 2), nf(qlike(rv, hi), 3)), T.ql3(nf(1 + Math.log(rv), 3), nf((rv - lo) ** 2, 3), nf((rv - hi) ** 2, 3))]);
  out('ql-out', [T.ql(rv, nf(h, 2), nf(qlike(rv, h), 3), nf(qlike(rv, h) - 1 - Math.log(rv), 3), h !== 0.5 || rv !== 1)]);
}

// ================================================================= x11. the Diebold-Mariano test (worked example)
function dmBlock() {
  const lag = +$('dm-l').value, m = kSel('dm-m'), k = kSel('dm-a'), M = rvMetrics(k, 'HAR', lag), dm = M[m].dm, S = rvSet(k), R = S.R, nw = single('dm-chart', 320); label('dm-l').textContent = lag;
  const d = R.y.map((_, i) => qlike(R.rv[i], S.var[m][i]) - qlike(R.rv[i], S.var.HAR[i])), r = acf(d, 22), js = r.map((_, j) => j);
  plot('dm-chart', [{ type: 'bar', x: js, y: r, marker: { color: js.map(j => (j <= lag ? BLUE : '#FFFFFF')), line: { color: BLUE, width: 1 } }, name: 'autocorrelation of d_t (filled: lags in the long-run variance)', hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' },
    { x: js.filter(j => j <= lag), y: js.filter(j => j <= lag).map(j => (j === 0 ? 1 : 1 - j / (lag + 1))), mode: 'lines+markers', line: { color: RED, width: 1.6 }, marker: { size: 6 }, name: `Bartlett weight 1 − j/(${lag} + 1)`, hovertemplate: 'lag %{x}: weight %{y:.2f}<extra></extra>' }],
  layout({ xaxis: AX('lag j (days)', { dtick: 2 }), yaxis: AX('autocorrelation / weight', { range: [-0.2, 1.05] }), legend: LEG(nw, -0.2), margin: M0 }));
  const nwOnly = dmTest(d, 0);
  steps('dm-st', [T.dm1(ANAME[k], MNAME[m], fint(dm.n), nf(dm.mean, 5)), T.dm2(nf(dm.g0, 4), lag, nf(dm.lrv, 4), nf(dm.lrv / dm.g0, 2)), T.dm3(nf(dm.t, 2), SIG(dm.p)), T.dm4(nf(nwOnly.t, 2), nf(dm.t, 2))]);
  out('dm-out', [T.dm(ANAME[k], MNAME[m], lag, nf(dm.t, 2), SIG(dm.p), lag !== 5)]);
}

// ================================================================= x12. results: S&P 500 and Bitcoin (table)
function rtBlock() {
  const per = kSel('rt-p'), lag = +kSel('rt-l'), P = panels('rt-chart', 2, { h: 340 }), mods = ['Lasso', 'RF', 'GB', 'MLP'], rows = [], tr = [];
  ['sp500', 'btc'].forEach((k, a) => {
    const M = rvMetrics(k, 'HAR', lag, PER[per]);
    tr.push(on2(P.ax[a], { type: 'bar', x: mods.map(q => MNAME[q]), y: mods.map(q => 100 * M[q].r2_b), marker: { color: mods.map(q => MC[q]) }, text: mods.map(q => `DM ${M[q].dm_t >= 0 ? '+' : ''}${M[q].dm_t.toFixed(2)}`), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, name: ANAME[k], showlegend: false, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }));
    tr.push(on2(P.ax[a], { x: [MNAME.Lasso, MNAME.MLP], y: [0, 0], mode: 'lines', line: { color: DARK, width: 1 }, hoverinfo: 'skip', showlegend: false }));
    ['HAR', ...mods].forEach((q, j) => { const r = M[q], sig = q !== 'HAR' && r.dm_p < 0.05;
      rows.push([j ? '' : { v: ANAME[k] + ` (${fint(M.n)})`, c: 'l' }, { v: MNAME[q], c: 'l' }, nf(100 * r.r2_mean, 1), q === 'HAR' ? '–' : { v: sg(100 * r.r2_b, 1), c: sig ? (r.r2_b > 0 ? 'hi' : 'bad') : '' },
        nf(r.qlike, 3), q === 'HAR' ? '–' : { v: `${sg(r.dm_t, 2)} (${SIG(r.dm_p)})`, c: sig ? (r.dm_t < 0 ? 'hi' : 'bad') : '' }]); });
    setAx(P, a, AX('', { type: 'category' }), AX(a ? '' : 'R² against HAR (%)', { range: [-12, 14] }));
  });
  plot('rt-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], 'S&P 500'), ptitle(P.ax[1], 'Bitcoin')], margin: MG })));
  table('rt-tab', T.h.rt, rows);
  const A = rvMetrics('sp500', 'HAR', lag, PER[per]), B = rvMetrics('btc', 'HAR', lag, PER[per]);
  out('rt-out', [T.rt(per, lag, sg(100 * A.MLP.r2_b, 1), sg(A.MLP.dm_t, 2), sg(100 * A.Lasso.r2_b, 1), sg(A.Lasso.dm_t, 2), sg(B.GB.dm_t, 2), SIG(B.GB.dm_p), per !== 'all' || lag !== 5)]);
}

// ================================================================= 10. forecasts in two storms: 2020 and 2025
function rfBlock() {
  const pair = kSel('rf-m').split(','), k = kSel('rf-a'), y1 = kSel('rf-y'), S = rvSet(k), R = S.R, P = panels('rf-chart', 2, { h: 380 }), tr = [], info = [];
  [[`${y1}-01-01`, `${y1}-12-31`], ['2025-01-01', '2025-09-18']].forEach(([a, b], q) => {
    const idx = S.ds.map((s, i) => (s >= a && s <= b ? i : -1)).filter(i => i >= 0), X = idx.map(i => S.ds[i]), real = idx.map(i => Math.sqrt(252 * R.rv[i]));
    tr.push(on2(P.ax[q], { x: X, y: real, mode: 'lines', line: { color: DARK, width: 1.4 }, name: 'realised, next 5 days', showlegend: q === 0, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }));
    const fc = {};
    [['HAR', 'solid'], [pair[0], 'dash'], [pair[1], 'dot']].forEach(([m, ls]) => { fc[m] = idx.map(i => Math.sqrt(252 * S.var[m][i]));
      tr.push(on2(P.ax[q], { x: X, y: fc[m], mode: 'lines', line: { color: MC[m], width: 1.4, dash: ls }, name: `${m} forecast`, showlegend: q === 0, hovertemplate: `${m}: %{y:.1f}%<extra></extra>` })); });
    const kr = real.indexOf(Math.max(...real));
    info.push([a.slice(0, 4), nf(real[kr], 0), T.date(X[kr]), ['HAR', ...pair].map(m => [m, nf(Math.max(...fc[m]), 0)])]);
    setAx(P, q, AX('', { type: 'date', tickangle: -30 }), AX(q ? '' : 'volatility (% per year)'));
  });
  plot('rf-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], `${ANAME[k]}, ${y1}`), ptitle(P.ax[1], `${ANAME[k]}, 2025`)], legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: MG })));
  out('rf-out', [T.rf(ANAME[k], info, k !== 'sp500' || y1 !== '2020')]);
}

// ================================================================= 11. the features used by the random forest
function imBlock() {
  const f = kSel('im-f'), I = D.imp, P = panels('im-chart', 2, { h: 400, gap: 0.12 }), order = D.ext.slice().sort((a, b) => I.perm[a] - I.perm[b]), sel = f === 'all' ? D.har : [f];
  const tr = [on2(P.ax[0], { type: 'bar', orientation: 'h', y: order.map(g => FL[g]), x: order.map(g => I.perm[g]), error_x: { type: 'data', array: order.map(g => I.perm_sd[g]), color: DARK, thickness: 1 }, marker: { color: order.map(g => FC[g]) },
    text: order.map(g => (sel.includes(g) ? I.perm[g].toFixed(3) : '')), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, name: 'permutation importance', showlegend: false, hovertemplate: '%{y}: %{x:.4f}<extra></extra>' })];
  sel.forEach(g => tr.push(on2(P.ax[1], { x: I.x[g], y: I.phi[g], mode: 'markers', marker: { color: hexA(FC[g], 0.7), size: 5 }, name: FL[g], hovertemplate: `${FL[g]} %{x:.2f}: %{y:.3f}<extra></extra>` })));
  tr.push(on2(P.ax[1], { x: [-3, 4], y: [0, 0], mode: 'lines', line: { color: DARK, width: 0.8 }, hoverinfo: 'skip', showlegend: false }));
  setAx(P, 0, AX('increase in test MSE when the feature is shuffled', { range: [-0.02, 0.26] }), AX('', { type: 'category', tickfont: { size: 10.5, color: DARK } }));
  setAx(P, 1, AX('feature value (log RV)'), AX('Shapley value (contribution to log RV)'));
  plot('im-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: { l: 110, r: 12, t: 14, b: 40 } })));
  table('im-tab', T.h.im, D.ext.slice().sort((a, b) => I.perm[b] - I.perm[a]).map(g => [{ v: T.fl(g), c: sel.includes(g) ? 'l hi' : 'l' }, nf(I.perm[g], 4), nf(I.perm_sd[g], 4), D.har.includes(g) ? nf(I.mean_abs_phi[g], 3) : '–']));
  const s = sel.map(g => [T.fl(g), nf(I.perm[g], 3), nf(I.mean_abs_phi[g], 2), nf(Math.min(...I.phi[g]), 2), nf(Math.max(...I.phi[g]), 2)]);
  out('im-out', [T.im(f, s, fint(I.n_test), nf(I.base, 3))]);
}

// ================================================================= x13. the right baseline (worked example)
function szBlock() {
  const m = kSel('sz-m'), k = kSel('sz-a'), M = signMetrics(k, m, 2026, 0.5), nw = single('sz-chart', 330), xs = lin(M.accB - 5 * M.se, M.accB + 5 * M.se, 301);
  const dens = xs.map(x => Math.exp(-0.5 * ((x - M.accB) / M.se) ** 2) / (M.se * Math.sqrt(2 * Math.PI)) / 100), top = Math.max(...dens);
  plot('sz-chart', [{ x: xs.map(x => 100 * x), y: dens, mode: 'lines', line: { color: BLUE, width: 2 }, fill: 'tozeroy', fillcolor: hexA(TEAL, 0.12), name: 'accuracy of a model without skill: N(acc₀, acc₀(1 − acc₀)/n)', hoverinfo: 'skip' },
    { x: [100 * (M.accB - 1.96 * M.se), 100 * (M.accB - 1.96 * M.se), null, 100 * (M.accB + 1.96 * M.se), 100 * (M.accB + 1.96 * M.se)], y: [0, top, null, 0, top], mode: 'lines', line: { color: TEAL, width: 1.4, dash: 'dash' }, name: '95% band', hoverinfo: 'skip' },
    { x: [100 * M.acc, 100 * M.acc], y: [0, top * 1.05], mode: 'lines+text', line: { color: MC[m], width: 2.6 }, text: ['', `${MNAME[m]}: ${(100 * M.acc).toFixed(1)}%, z = ${M.z.toFixed(2)}`], textposition: 'top center', textfont: { size: 11, color: DARK }, name: `${MNAME[m]}`, hoverinfo: 'skip', cliponaxis: false }],
  layout({ xaxis: AX('accuracy (%)'), yaxis: AX('density', { showticklabels: false, range: [0, top * 1.25] }), legend: LEG(nw, -0.2), margin: M0 }));
  steps('sz-we', [T.sz1(ANAME[k], fint(M.n), pc(M.up, 1), pc(M.accB, 1)), T.sz2(nf(100 * M.se, 2), nf(196 * M.se, 1)), T.sz3(MNAME[m], pc(M.acc, 1), nf(M.z, 2), SIG(M.p)), T.sz4(nf(M.auc, 3), pc(M.shareUp, 0))]);
  out('sz-out', [T.sz(ANAME[k], MNAME[m], sg(100 * M.diff, 1), nf(M.z, 2), SIG(M.p))]);
}

// ================================================================= 12. accuracy against the baseline
function sgBlock() {
  const end = +$('sg-e').value, c = +kSel('sg-c'), nw = single('sg-chart', 360), ks = ['sp500', 'bet', 'btc'], ms = ['Logit', 'RF', 'GB']; label('sg-e').textContent = end;
  const M = Object.fromEntries(ks.map(k => [k, Object.fromEntries(ms.map(m => [m, signMetrics(k, m, end, c)]))]));
  const xs = ks.map(k => { const r = M[k].Logit; return `${ANAME[k]}<br>(baseline ${r ? (100 * r.accB).toFixed(1) + '%' : 'no test year'})`; });
  const tr = ms.map((m, i) => ({ type: 'bar', x: xs, y: ks.map(k => (M[k][m] ? 100 * M[k][m].diff : null)), marker: { color: MC[m] }, name: { Logit: 'logit', RF: 'random forest', GB: 'gradient boosting' }[m], hovertemplate: `${m}: %{y:.2f} pp<extra></extra>` }));
  const shapes = [];
  ks.forEach((k, j) => { const r = M[k].Logit; if (!r) return; const b = 196 * r.se; shapes.push({ type: 'rect', layer: 'below', xref: 'x', yref: 'y', x0: j - 0.42, x1: j + 0.42, y0: -b, y1: b, fillcolor: hexA(TEAL, 0.2), line: { width: 0 } }); });
  tr.push({ x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: hexA(TEAL, 0.35) }, name: '95% band under no skill', hoverinfo: 'skip' });
  plot('sg-chart', tr, layout({ barmode: 'group', shapes, xaxis: AX('', { type: 'category', range: [-0.5, 2.5] }), yaxis: AX('accuracy minus baseline (pp)', { zeroline: true, zerolinecolor: DARK, range: [-4.5, 3] }), legend: LEG(nw, -0.3), margin: M0 }));
  out('sg-out', [T.sg(end, nf(c, 2), ks.map(k => [ANAME[k], M[k].Logit ? fint(M[k].Logit.n) : null, ...ms.map(m => (M[k][m] ? [sg(100 * M[k][m].diff, Math.abs(100 * M[k][m].diff) < 0.05 ? 2 : 1), nf(M[k][m].z, 2)] : null))]), end !== 2026 || c !== 0.5)]);
}

// ================================================================= x14. the quantile (pinball) loss (worked example)
function plBlock() {
  const q = +$('pl-q').value, a = +kSel('pl-a'), y = D.qvar.sp500.y, n = y.length, nw = single('pl-chart', 330), qs = lin(-8, 1, 181); label('pl-q').textContent = nf(q, 2) + '%';
  const L = c => mean(y.map(v => { const u = v - c; return u * (a - (u < 0 ? 1 : 0)); })), Ls = qs.map(L), sy = Float64Array.from(y).sort();
  let qb = qs[0], lb = Infinity; [...qs, ...sy.slice(0, Math.ceil(0.2 * n))].forEach(c => { const v = L(c); if (v < lb) { lb = v; qb = c; } });
  const below = c => mean(y.map(v => +(v < c)));
  plot('pl-chart', [{ x: qs, y: Ls, mode: 'lines', line: { color: BLUE, width: 2.2 }, name: `average pinball loss of a constant q, α = ${a}`, hovertemplate: 'q %{x:.2f}%: %{y:.4f}<extra></extra>' },
    { x: [q], y: [L(q)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`${(100 * below(q)).toFixed(2)}% below`], textposition: 'top right', textfont: { size: 11, color: DARK }, name: `q = ${q}%`, hoverinfo: 'skip' },
    { x: [qb], y: [lb], mode: 'markers', marker: { color: GREEN, size: 11 }, name: `minimum: q = ${qb.toFixed(2)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('constant forecast q of the next-day return (%)'), yaxis: AX('average loss'), legend: LEG(nw, -0.2), margin: M0 }));
  steps('pl-we', [T.pl1(nf(a, 2), nf(1 - a, 2)), T.pl2(fint(n), nf(q, 2), pc(below(q), 2), nf(L(q), 4)), T.pl3(nf(qb, 2), pc(below(qb), 2), nf(lb, 4), pc(a, 0))]);
  out('pl-out', [T.pl(nf(a, 2), nf(q, 2), pc(below(q), 2), nf(qb, 2), a !== 0.01 || q !== -2.7)]);
}

// ================================================================= x15. backtest of VaR 1% (table)
function vtBlock() {
  const per = kSel('vt-p'), P = panels('vt-chart', 2, { h: 340 }), tr = [], rows = [], ms = D.qvar.sp500.methods, mem = {}, shapes = [];
  ['sp500', 'btc'].forEach((k, a) => {
    const B = ms.map(m => backtest(k, m, PER[per])); mem[k] = B; const n = B[0].n, band = kupiecBand(n);
    shapes.push({ type: 'rect', layer: 'below', xref: P.ax[a].x + ' domain', yref: P.ax[a].y, x0: 0, x1: 1, y0: 100 * band[0], y1: 100 * band[1], fillcolor: hexA(TEAL, 0.2), line: { width: 0 } });
    if (a === 0) tr.push(on2(P.ax[a], { x: [null], y: [null], mode: 'markers', marker: { symbol: 'square', size: 12, color: hexA(TEAL, 0.35) }, name: 'rates not rejected by Kupiec at 5%', hoverinfo: 'skip' }));
    tr.push(on2(P.ax[a], { type: 'bar', x: [0, 1, 2, 3], y: B.map(b => 100 * b.rate), marker: { color: ms.map(m => MC[m]) }, text: B.map(b => `${b.x}`), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, name: ANAME[k], showlegend: false, hovertemplate: '%{y:.2f}%<extra></extra>' }));
    B.forEach((b, j) => rows.push([j ? '' : { v: `${ANAME[k]} (${fint(n)})`, c: 'l' }, { v: ms[j], c: 'l' }, b.x, nf(100 * b.rate, 2), { v: nf(b.p_uc, 3), c: b.p_uc < 0.05 ? 'bad' : '' }, { v: SIG(b.p_ind), c: b.p_ind < 0.05 ? 'bad' : '' }, nf(100 * b.qloss, 2), nf(b.mean_var, 2)]));
    setAx(P, a, AX('', { tickvals: [0, 1, 2, 3], ticktext: ms, range: [-0.5, 3.5] }), AX(a ? '' : 'exceedance rate (%)', { range: [0, 2.4] }));
  });
  plot('vt-chart', tr, layout(Object.assign(P.L, { shapes, annotations: [ptitle(P.ax[0], 'S&P 500'), ptitle(P.ax[1], 'Bitcoin')], legend: LEG(P.stack, P.stack ? -0.1 : -0.22), margin: MG })));
  table('vt-tab', T.h.vt, rows);
  const S = mem.sp500, Bt = mem.btc, best = A => A.reduce((p, b, j) => (b.qloss < A[p].qloss ? j : p), 0);
  out('vt-out', [T.vt(per, ms[best(S)], nf(100 * S[best(S)].qloss, 2), nf(100 * S[0].rate, 2), nf(S[0].p_ind, 3), nf(100 * S[3].rate, 2), nf(S[3].p_uc, 3), nf(100 * Bt[2].rate, 2), nf(100 * Bt[1].rate, 2), nf(Bt[1].p_uc, 3), per !== 'all')]);
}

// ================================================================= 13. VaR 1% forecasts in 2020 and 2025
function qvBlock() {
  const m = kSel('qv-m'), k = kSel('qv-a'), y1 = kSel('qv-y'), Q = D.qvar[k], ds = dates(Q), P = panels('qv-chart', 2, { h: 380 }), tr = [], cnt = [];
  [[`${y1}-01-01`, `${y1}-12-31`], ['2025-01-01', '2025-09-18']].forEach(([a, b], q) => {
    const idx = ds.map((s, i) => (s >= a && s <= b ? i : -1)).filter(i => i >= 0), X = idx.map(i => ds[i]);
    tr.push(on2(P.ax[q], { type: 'bar', x: X, y: idx.map(i => Q.y[i]), marker: { color: TEAL }, name: 'next-day log return (%)', showlegend: q === 0, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }));
    [['QR', 'solid'], ['QGB', 'solid'], ['HS', 'dash'], ['GARCH-t', 'dot']].forEach(([mm, ls]) => tr.push(on2(P.ax[q], { x: X, y: idx.map(i => -Q.var[mm][i]), mode: 'lines', line: { color: MC[mm], width: 1.3, dash: ls }, name: `minus VaR 1%, ${mm}`, showlegend: q === 0, hovertemplate: `${mm}: %{y:.2f}%<extra></extra>` })));
    const hit = idx.filter(i => Q.y[i] < -Q.var[m][i]);
    tr.push(on2(P.ax[q], { x: hit.map(i => ds[i]), y: hit.map(i => Q.y[i]), mode: 'markers', marker: { color: RED, size: 7 }, name: `exceedance of ${m} VaR`, showlegend: q === 0, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }));
    cnt.push([a.slice(0, 4), Q.methods.map(mm => [mm, idx.filter(i => Q.y[i] < -Q.var[mm][i]).length]), fint(idx.length)]);
    setAx(P, q, AX('', { type: 'date', tickangle: -30 }), AX(q ? '' : 'return (%)'));
  });
  plot('qv-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], `${ANAME[k]}, ${y1}`), ptitle(P.ax[1], `${ANAME[k]}, 2025`)], legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: MG, bargap: 0 })));
  out('qv-out', [T.qv(ANAME[k], m, cnt, k !== 'sp500' || y1 !== '2020')]);
}

// ================================================================= x16. the best of N strategies (worked example)
const NS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
function mxBlock() {
  const N = NS[+$('mx-n').value], Y = kSel('mx-y'), M = D.snoop.maxsr[Y], sd = 1 / Math.sqrt(+Y), nw = single('mx-chart', 330), Nn = Array.from({ length: 1000 }, (_, i) => i + 1); label('mx-n').textContent = N;
  plot('mx-chart', [{ x: NS.concat(NS.slice().reverse()), y: NS.map(n => M[n].q95).concat(NS.slice().reverse().map(n => M[n].q05)), mode: 'lines', fill: 'toself', fillcolor: hexA(TEAL, 0.25), line: { width: 0 }, name: '5%-95% of the best Sharpe ratio', hoverinfo: 'skip' },
    { x: NS, y: NS.map(n => M[n].mean), mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 6 }, name: 'average best Sharpe ratio (simulation)', hovertemplate: 'N = %{x}: %{y:.3f}<extra></extra>' },
    { x: Nn, y: Nn.map(n => expMax(n, sd)), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'expected maximum (formula)', hovertemplate: 'N = %{x}: %{y:.3f}<extra></extra>' },
    { x: [N], y: [expMax(N, sd)], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [expMax(N, sd).toFixed(2)], textposition: 'top left', textfont: { size: 11, color: DARK }, name: `N = ${N}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('number of strategies tried, N (log scale)', { type: 'log' }), yaxis: AX('annualised Sharpe ratio', { range: [-0.6, 1.8] }), legend: LEG(nw, -0.2), margin: M0 }));
  const a = N > 1 ? Pinv(1 - 1 / N) : null, b = N > 1 ? Pinv(1 - 1 / (N * Math.E)) : null;
  steps('mx-we', [T.mx1(Y, nf(sd, 3)), N > 1 ? T.mx2(N, nf(a, 3), nf(b, 3)) : T.mx2b(), T.mx3(N, nf(expMax(N, sd), 2), nf(M[N].mean, 2), nf(M[N].q05, 2), nf(M[N].q95, 2))]);
  out('mx-out', [T.mx(N, Y, nf(expMax(N, sd), 2), nf(M[N].mean, 2), Y !== '10')]);
}

// ================================================================= 14. selection bias in practice
function snBlock() {
  const N = +$('sn-n').value, sp = kSel('sn-s'), Y = kSel('sn-y'), S = D.snoop.runs[sp], R = S.D, M = D.snoop.maxsr[Y], P = panels('sn-chart', 2, { h: 380 }); label('sn-n').textContent = N;
  let b = 0; for (let i = 1; i < N; i++) if (R.is[i] > R.is[b]) b = i;
  const sd = 1 / Math.sqrt(+Y), Nn = Array.from({ length: 1000 }, (_, i) => i + 1), first = R.is.map((_, i) => i < N), rank = R.oos.filter(v => v > R.oos[b]).length + 1;
  const tr = [on2(P.ax[0], { x: NS.concat(NS.slice().reverse()), y: NS.map(n => M[n].q95).concat(NS.slice().reverse().map(n => M[n].q05)), mode: 'lines', fill: 'toself', fillcolor: hexA(TEAL, 0.25), line: { width: 0 }, name: '5%-95% of the best Sharpe ratio', hoverinfo: 'skip' }),
    on2(P.ax[0], { x: NS, y: NS.map(n => M[n].mean), mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 6 }, name: 'average best Sharpe ratio (simulation)', hovertemplate: 'N = %{x}: %{y:.3f}<extra></extra>' }),
    on2(P.ax[0], { x: Nn, y: Nn.map(n => expMax(n, sd)), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'expected maximum (formula)', hoverinfo: 'skip' }),
    on2(P.ax[0], { x: [N], y: [expMax(N, sd)], mode: 'markers+text', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`N = ${N}: ${expMax(N, sd).toFixed(2)}`], textposition: 'top left', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip' }),
    on2(P.ax[1], { x: R.is.filter((_, i) => first[i]), y: R.oos.filter((_, i) => first[i]), mode: 'markers', marker: { color: BLUE, size: 8 }, name: `moving-average rule (first ${N} tried)`, text: R.s.map((s, i) => `MA ${s}/${R.l[i]}`).filter((_, i) => first[i]), hovertemplate: '%{text}: %{x:.2f} / %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], { x: R.is.filter((_, i) => !first[i]), y: R.oos.filter((_, i) => !first[i]), mode: 'markers', marker: { color: '#FFFFFF', size: 7, line: { color: BLUE, width: 1 } }, name: 'not tried yet', text: R.s.map((s, i) => `MA ${s}/${R.l[i]}`).filter((_, i) => !first[i]), hovertemplate: '%{text}: %{x:.2f} / %{y:.2f}<extra></extra>' }),
    on2(P.ax[1], { x: [R.is[b]], y: [R.oos[b]], mode: 'markers', marker: { color: RED, size: 16, symbol: 'star' }, name: 'best rule in sample', hoverinfo: 'skip' }),
    on2(P.ax[1], { x: [S.bh_is], y: [S.bh_oos], mode: 'markers', marker: { color: GREEN, size: 11, symbol: 'square' }, name: 'buy and hold', hoverinfo: 'skip' })];
  setAx(P, 0, AX('number of strategies tried, N', { type: 'log' }), AX('annualised Sharpe ratio')); setAx(P, 1, AX(`Sharpe ratio, 2000-${sp} (in sample)`), AX(`Sharpe ratio, ${+sp + 1}-2026`));
  plot('sn-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  out('sn-out', [T.sn(N, R.s[b], R.l[b], nf(R.is[b], 2), nf(R.oos[b], 2), rank, R.is.length, nf(S.corr, 2), nf(S.bh_is, 2), nf(S.bh_oos, 2), nf(expMax(N, sd), 2), Y, sp, sp !== '2014' || Y !== '10')]);
}

// ================================================================= x17. the deflated Sharpe ratio (worked example)
function dsBlock() {
  const N = +$('ds-n').value, mo = kSel('ds-k'), S = D.snoop.runs['2014'], b = S.best, sk = mo === 'normal' ? 0 : b.skew, ku = mo === 'normal' ? 3 : b.kurt, nw = single('ds-chart', 330), Ns = Array.from({ length: 200 }, (_, i) => i + 1); label('ds-n').textContent = N;
  const f = n => deflated(b.is_pp, expMax(n, S.sd_pp), b.T, sk, ku), r = f(N);
  plot('ds-chart', [{ x: Ns, y: Ns.map(n => f(n).dsr), mode: 'lines', line: { color: BLUE, width: 2.2 }, name: 'deflated Sharpe ratio DSR', hovertemplate: 'N = %{x}: %{y:.3f}<extra></extra>' },
    { x: [1, 200], y: [r.psr, r.psr], mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: `PSR (no correction for N): ${r.psr.toFixed(3)}`, hoverinfo: 'skip' },
    { x: [1, 200], y: [0.95, 0.95], mode: 'lines', line: { color: RED, width: 1.2, dash: 'dot' }, name: 'significance threshold 0.95', hoverinfo: 'skip' },
    { x: [N], y: [r.dsr], mode: 'markers+text', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [r.dsr.toFixed(3)], textposition: 'bottom left', textfont: { size: 11, color: DARK }, name: `N = ${N}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('number of rules tried, N (log scale)', { type: 'log' }), yaxis: AX('probability', { range: [0, 1.05] }), legend: LEG(nw, -0.2), margin: M0 }));
  const sr0 = expMax(N, S.sd_pp);
  steps('ds-we', [T.ds1(N, fint(b.T), nf(b.is_pp, 4), nf(b.is, 2), nf(sk, 2), nf(ku, 1)), N > 1 ? T.ds2(nf(S.sd_pp, 4), nf(Pinv(1 - 1 / N), 3), nf(Pinv(1 - 1 / (N * Math.E)), 3), nf(sr0, 4), nf(sr0 * ANN, 2)) : T.ds2b(),
    T.ds3(nf(r.den, 3), nf(r.z, 2), nf(r.dsr, 3)), T.ds4(nf(r.psr, 3), r.dsr > 0.95)]);
  out('ds-out', [T.ds(N, nf(r.dsr, 3), r.dsr > 0.95, mo, N !== 30 || mo !== 'sample')]);
}

// ================================================================= 15. results of Gu, Kelly and Xiu (2020)
function gkBlock() {
  const m = kSel('gk-m'), G = D.gkx, P = panels('gk-chart', 2, { h: 380 }), j = G.models.indexOf(m), col = G.models.map((_, i) => (i < 5 ? BLUE : i < 7 ? GREEN : RED));
  const lab = (v, i) => (i === j ? v.toFixed(2) : '');
  const tr = [on2(P.ax[0], { type: 'bar', x: G.models, y: G.r2, marker: { color: col }, offsetgroup: 'a', name: 'all stocks', text: G.r2.map(lab), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }),
    on2(P.ax[0], { type: 'bar', x: G.models, y: G.top, marker: { color: '#FFFFFF', line: { color: col, width: 1.2 }, pattern: { shape: '/', fgcolor: col } }, offsetgroup: 'b', name: 'top 1,000 stocks', text: G.top.map(lab), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }),
    on2(P.ax[1], { type: 'bar', x: G.models, y: G.sr, marker: { color: col }, name: 'Sharpe ratio, value-weighted H-L decile', showlegend: false, text: G.sr.map(lab), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' })];
  setAx(P, 0, AX('', { type: 'category', tickangle: -60, tickfont: { size: 10, color: DARK } }), AX('monthly out-of-sample R² (%)', { zeroline: true, zerolinecolor: DARK }));
  setAx(P, 1, AX('', { type: 'category', tickangle: -60, tickfont: { size: 10, color: DARK } }), AX('annualised Sharpe ratio, H-L decile'));
  plot('gk-chart', tr, layout(Object.assign(P.L, { barmode: 'group', legend: LEG(P.stack, P.stack ? -0.12 : -0.32), margin: M0 })));
  const rk = a => a.filter(v => v > a[j]).length + 1;
  out('gk-out', [T.gk(m, nf(G.r2[j], 2), nf(G.top[j], 2), nf(G.sr[j], 2), rk(G.r2), rk(G.sr))]);
}

// ================================================================= wiring
const PFX = { bv: 'bv', cvs: 'cs', 'wf-we': 'wf', leak: 'lk', 'ridge-we': 'rw', 'lasso-we': 'lw', shrink: 'sh', 'split-we': 'sp', tree: 'tr', 'rf-we': 'bg', 'gb-we': 'gb', ens: 'en',
  'mlp-we': 'mw', mlp: 'ml', nnarch: 'na', nnjpy: 'nj', 'har-we': 'hw', 'r2-we': 'r2', 'qlike-we': 'ql', 'dm-we': 'dm', 'rv-tab': 'rt', rvf: 'rf', imp: 'im', 'sign-we': 'sz', sign: 'sg',
  'pin-we': 'pl', 'var-tab': 'vt', qvar: 'qv', 'maxsr-we': 'mx', snoop: 'sn', 'dsr-we': 'ds', gkx: 'gk' };
const BLOCKS = { bvBlock, csBlock, wfBlock, lkBlock, rwBlock, lwBlock, shBlock, spBlock, trBlock, bgBlock, gbBlock, enBlock, mwBlock, mlBlock, naBlock, njBlock, hwBlock, r2Block, qlBlock,
  dmBlock, rtBlock, rfBlock, imBlock, szBlock, sgBlock, plBlock, vtBlock, qvBlock, mxBlock, snBlock, dsBlock, gkBlock };
const PLAYERS = {};
function placeExtras() {                // web lecture: the table blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
const YEARS = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => String(a + i));
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const PLy = (btn, id, vals, ms) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms));
  on(['bv-d', 'bv-s'], bvBlock); PLy('bv-play', 'bv-d', [3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 3], 800);
  on(['cs-k', 'cs-h', 'cs-e'], csBlock); PLy('cs-play', 'cs-k', [5, 3, 4, 6, 7, 8, 5], 1100);
  on(['wf-y', 'wf-h'], wfBlock); PLy('wf-play', 'wf-y', [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2013], 700);
  on(['lk-h'], lkBlock); PLy('lk-play', 'lk-h', ['21', '5', '63', '21'], 1400);
  on(['rw-l', 'rw-s'], rwBlock); PLy('rw-play', 'rw-l', [50, 0, 10, 25, 75, 100, 150, 200, 50], 800);
  on(['lw-l', 'lw-s'], lwBlock); PLy('lw-play', 'lw-l', [20, 0, 10, 30, 40, 50, 60, 80, 100, 20], 800);
  on(['sh-i', 'sh-f'], shBlock); PLy('sh-play', 'sh-i', [28, 59, 52, 45, 38, 33, 24, 18, 12, 6, 0, 28], 800);
  on(['sp-s', 'sp-m'], spBlock); PLy('sp-play', 'sp-s', ['1.25', '0.7', '0.9', '1.75', '2.2', '1.25'], 1100);
  on(['tr-d', 'tr-l'], trBlock); PLy('tr-play', 'tr-d', ['2', '1', '3', '2'], 1600);
  on(['bg-B', 'bg-r'], bgBlock); PLy('bg-play', 'bg-B', [10, 1, 2, 5, 20, 50, 100, 200, 300, 10], 800);
  on(['gb-m', 'gb-v'], gbBlock); PLy('gb-play', 'gb-m', [1, 2, 3, 5, 8, 12, 20, 30, 50, 1], 800);
  on(['en-n', 'en-d'], enBlock); PLy('en-play', 'en-n', [200, 1, 2, 5, 10, 20, 50, 100, 150, 300, 400, 500, 200], 800);
  on(['mw-k', 'mw-p'], mwBlock); PLy('mw-play', 'mw-k', [16, 1, 2, 4, 8, 32, 64, 16], 900);
  on(['ml-g', 'ml-z', 'ml-k'], mlBlock); PLy('ml-play', 'ml-g', ['relu', 'logistic', 'tanh', 'relu'], 1400);
  on(['na-p', 'na-u'], naBlock); PLy('na-play', 'na-p', ['all', ...YEARS(2010, 2026), 'all'], 900);
  on(['nj-s'], njBlock); PLy('nj-play', 'nj-s', ['0.8', '0.6', '0.9', '0.8'], 1500);
  on(['hw-y', 'hw-d'], hwBlock); PLy('hw-play', 'hw-y', [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2013], 800);
  on(['r2-m', 'r2-b', 'r2-a'], r2Block); PLy('r2-play', 'r2-m', ['MLP', 'Lasso', 'RF', 'GB', 'MLP'], 1400);
  on(['ql-h', 'ql-r'], qlBlock); PLy('ql-play', 'ql-h', [0.5, 0.3, 0.7, 1, 1.5, 2, 2.5, 3, 0.5], 800);
  on(['dm-l', 'dm-m', 'dm-a'], dmBlock); PLy('dm-play', 'dm-l', [5, 0, 1, 2, 3, 10, 15, 22, 5], 900);
  on(['rt-p', 'rt-l'], rtBlock); PLy('rt-play', 'rt-p', ['all', 'pre', 'post', 'all'], 1600);
  on(['rf-m', 'rf-a', 'rf-y'], rfBlock); PLy('rf-play', 'rf-m', ['RF,MLP', 'Lasso,GB', 'Lasso,MLP', 'RF,GB', 'RF,MLP'], 1500);
  on(['im-f'], imBlock); PLy('im-play', 'im-f', ['all', 'd', 'w', 'm', 'all'], 1400);
  on(['sz-m', 'sz-a'], szBlock); PLy('sz-play', 'sz-m', ['RF', 'GB', 'Logit', 'RF'], 1400);
  on(['sg-e', 'sg-c'], sgBlock); PLy('sg-play', 'sg-e', [2026, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026], 900);
  on(['pl-q', 'pl-a'], plBlock); PLy('pl-play', 'pl-q', [-2.7, -6, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -2.7], 800);
  on(['vt-p'], vtBlock); PLy('vt-play', 'vt-p', ['all', 'pre', 'post', 'all'], 1600);
  on(['qv-m', 'qv-a', 'qv-y'], qvBlock); PLy('qv-play', 'qv-m', ['QR', 'QGB', 'HS', 'GARCH-t', 'QR'], 1400);
  on(['mx-n', 'mx-y'], mxBlock); PLy('mx-play', 'mx-n', [6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 6], 800);
  on(['sn-n', 'sn-s', 'sn-y'], snBlock); PLy('sn-play', 'sn-n', [30, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30], 800);
  on(['ds-n', 'ds-k'], dsBlock); PLy('ds-play', 'ds-n', [30, 1, 2, 5, 10, 20, 50, 100, 200, 30], 800);
  on(['gk-m'], gkBlock); PLy('gk-play', 'gk-m', ['NN3', 'OLS-3+H', 'PLS', 'PCR', 'ENet+H', 'GLM+H', 'RF', 'GBRT+H', 'NN1', 'NN2', 'NN4', 'NN5', 'NN3'], 900);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH13 = { Phi, Pinv, chi1sf, auc, ols, dmTest, kupiec, christoffersen, expMax, deflated, rvSet, rvMetrics, harFold, signMetrics, backtest, kupiecBand, ridgeAt, ridgePath, OLSB,
  schemes, splits, boost, stump, leafOf, dates, init, PFX, blocks: BLOCKS };
if (window.CH13_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
