// Chapter 12 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch12_data.js.
(function () {
'use strict';
const D = window.CH12_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG, solve } = window.SFMI;
const T = window.CH12_TEXT;
const NUM = D.N, COL = D.colors;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';

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
function gammln(z) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
  for (const v of g) ser += v / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function chi2sf(x, k) {              // upper regularized gamma Q(k/2, x/2)
  const a = k / 2, z = x / 2; if (z <= 0) return 1;
  if (z < a + 1) { let s = 1 / a, t = s; for (let n = 1; n < 500; n++) { t *= z / (a + n); s += t; if (Math.abs(t) < Math.abs(s) * 1e-15) break; } return 1 - s * Math.exp(-z + a * Math.log(z) - gammln(a)); }
  let b = z + 1 - a, cc = 1e300, dd = 1 / b, h = dd;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; dd = an * dd + b; if (Math.abs(dd) < 1e-300) dd = 1e-300; cc = b + an / cc; if (Math.abs(cc) < 1e-300) cc = 1e-300; dd = 1 / dd; const del = dd * cc; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
  return Math.exp(-z + a * Math.log(z) - gammln(a)) * h;
}
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
function npq(sorted, p) { const h = (sorted.length - 1) * p, i = Math.floor(h); return i + 1 < sorted.length ? sorted[i] + (h - i) * (sorted[i + 1] - sorted[i]) : sorted[i]; }
const quantile = (v, p) => npq(Float64Array.from(v).sort(), p);
function inv(A) { const n = A.length; return Array.from({ length: n }, (_, j) => solve(A, A.map((_, i) => (i === j ? 1 : 0)))); }   // columns of A^-1 (A symmetric)
const logistic = e => 1 / (1 + Math.exp(-e));

// ================================================================= the chapter's models (generate_all_charts.py)
function logitFit(X, y) {            // Newton-Raphson from 0; constant first; se from the inverse Hessian
  const n = y.length, k = (X[0] ? X[0].length : 0) + 1, R = X.map(r => [1, ...r]);
  let b = new Array(k).fill(0);
  const hess = () => { const H = Array.from({ length: k }, () => new Array(k).fill(0)), g = new Array(k).fill(0), p = new Array(n);
    for (let i = 0; i < n; i++) { const r = R[i]; let e = 0; for (let j = 0; j < k; j++) e += r[j] * b[j]; const q = 1 / (1 + Math.exp(-e)), w = q * (1 - q); p[i] = q;
      for (let j = 0; j < k; j++) { g[j] += (y[i] - q) * r[j]; const wr = w * r[j]; for (let l = 0; l <= j; l++) H[j][l] += wr * r[l]; } }
    for (let j = 0; j < k; j++) for (let l = 0; l < j; l++) H[l][j] = H[j][l]; return { H, g, p }; };
  for (let it = 0; it < 100; it++) { const { H, g } = hess(), st = solve(H, g); b = b.map((v, j) => v + st[j]); if (Math.max(...st.map(Math.abs)) < 1e-10) break; }
  const { H, p } = hess(), Hi = inv(H), se = Hi.map((c, j) => Math.sqrt(c[j]));
  let ll = 0; for (let i = 0; i < n; i++) ll += y[i] * Math.log(p[i]) + (1 - y[i]) * Math.log(1 - p[i]);
  return { b, se, ll, n, p, predict: Xn => Xn.map(r => logistic(b[0] + r.reduce((s, v, j) => s + v * b[j + 1], 0))) };
}
function ldaFit(X, y) {
  const k = X[0].length, m = [0, 1].map(c => { const r = X.filter((_, i) => y[i] === c); return { n: r.length, m: Array.from({ length: k }, (_, j) => mean(r.map(v => v[j]))) }; });
  const S = Array.from({ length: k }, () => new Array(k).fill(0));
  X.forEach((r, i) => { const mu = m[y[i]].m; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) S[a][b] += (r[a] - mu[a]) * (r[b] - mu[b]); });
  const nn = m[0].n + m[1].n; S.forEach(r => r.forEach((_, j) => (r[j] /= nn - 2)));
  const dm = m[1].m.map((v, j) => v - m[0].m[j]), w = solve(S, dm), pi1 = m[1].n / nn;
  const c = -0.5 * m[1].m.reduce((s, v, j) => s + (v + m[0].m[j]) * w[j], 0) + Math.log(pi1 / (1 - pi1));
  return { w, c, m0: m[0].m, m1: m[1].m, S, n0: m[0].n, n1: m[1].n, dm, eta: Xn => Xn.map(r => c + r.reduce((s, v, j) => s + v * w[j], 0)) };
}
function ranks(s) {
  const idx = s.map((_, i) => i).sort((a, b) => s[a] - s[b]), r = new Array(s.length); let i = 0;
  while (i < idx.length) { let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++; const v = (i + j) / 2 + 1; for (let q = i; q <= j; q++) r[idx[q]] = v; i = j + 1; }
  return r;
}
function auc(s, y) { const r = ranks(s); let n1 = 0, rs = 0; y.forEach((v, i) => { if (v === 1) { n1++; rs += r[i]; } }); const n0 = y.length - n1; return (rs - n1 * (n1 + 1) / 2) / (n1 * n0); }
function pairs(s, y) {               // correct, tied and wrong (bad, good) pairs
  const P = s.filter((_, i) => y[i] === 1), Q = s.filter((_, i) => y[i] === 0); let gt = 0, eq = 0;
  for (const p of P) for (const q of Q) { if (p > q) gt++; else if (p === q) eq++; }
  return { gt, eq, lt: P.length * Q.length - gt - eq, n1: P.length, n0: Q.length };
}
function delong(s, y, s2) {
  const comps = v => { const P = v.filter((_, i) => y[i] === 1), Q = v.filter((_, i) => y[i] === 0);
    return { a10: P.map(p => mean(Q.map(q => (p > q) + 0.5 * (p === q)))), a01: Q.map(q => mean(P.map(p => (p > q) + 0.5 * (p === q)))) }; };
  const cov = (u, v) => { const mu = mean(u), mv = mean(v); let c = 0; for (let i = 0; i < u.length; i++) c += (u[i] - mu) * (v[i] - mv); return c / (u.length - 1); };
  const A = comps(s), a = mean(A.a10);
  if (!s2) { const se = Math.sqrt(cov(A.a10, A.a10) / A.a10.length + cov(A.a01, A.a01) / A.a01.length); return { auc: a, se, lo: a - 1.96 * se, hi: a + 1.96 * se }; }
  const B = comps(s2), v = (cov(A.a10, A.a10) + cov(B.a10, B.a10) - 2 * cov(A.a10, B.a10)) / A.a10.length + (cov(A.a01, A.a01) + cov(B.a01, B.a01) - 2 * cov(A.a01, B.a01)) / A.a01.length;
  const z = (a - mean(B.a10)) / Math.sqrt(v); return { auc1: a, auc2: mean(B.a10), z, p: 2 * (1 - Phi(Math.abs(z))) };
}
function rocPts(s, y) {
  const thr = [Infinity, ...[...new Set(s)].sort((a, b) => b - a)], n1 = sum(y), n0 = y.length - n1;
  const tpr = [], fpr = []; thr.forEach(t => { let a = 0, b = 0; s.forEach((v, i) => { if (v >= t) { if (y[i]) a++; else b++; } }); tpr.push(a / n1); fpr.push(b / n0); });
  return { tpr, fpr, thr };
}
function capPts(s, y) {
  const o = s.map((_, i) => i).sort((a, b) => s[b] - s[a] || a - b), n = y.length, B = sum(y), x = [0], h = [0]; let c = 0;
  o.forEach((i, k) => { c += y[i]; x.push((k + 1) / n); h.push(c / B); });
  let ar = 0; for (let k = 1; k < x.length; k++) ar += (x[k] - x[k - 1]) * (h[k] + h[k - 1]) / 2;
  const pb = B / n; return { x, h, ar: (ar - 0.5) / (1 - pb / 2 - 0.5), pb };
}
function ksStat(s, y) {
  const u = [...new Set(s)].sort((a, b) => a - b), s1 = s.filter((_, i) => y[i] === 1).sort((a, b) => a - b), s0 = s.filter((_, i) => y[i] === 0).sort((a, b) => a - b);
  const F = (arr, v) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; } return lo / arr.length; };
  let best = -1, at = null; u.forEach(v => { const d = Math.abs(F(s0, v) - F(s1, v)); if (d > best) { best = d; at = v; } });
  return { ks: best, at, F1: v => F(s1, v), F0: v => F(s0, v), u };
}
const brier = (p, y) => mean(p.map((v, i) => (v - y[i]) ** 2));
function calTable(p, y, order, k) {   // calibration_table: groups of equal size in the order of pandas sort_values
  const n = order.length, G = Array.from({ length: k }, () => ({ n: 0, ps: 0, bad: 0 }));
  order.forEach((i, j) => { const g = G[Math.floor(j * k / n)]; g.n++; g.ps += p[i]; g.bad += y[i]; });
  let hl = 0; G.forEach(g => { g.pd = g.ps / g.n; g.rate = g.bad / g.n; g.exp = g.n * g.pd; hl += (g.bad - g.exp) ** 2 / (g.exp * (1 - g.pd)); });
  return { G, hl, p: chi2sf(hl, k - 2), df: k - 2 };
}
function confusion(p, y, cut, cfn) {
  cfn = cfn === undefined ? D.cost_fn : cfn; let TP = 0, FN = 0, FP = 0, TN = 0;
  p.forEach((v, i) => { const r = v > cut; if (y[i]) { if (r) TP++; else FN++; } else if (r) FP++; else TN++; });
  const n = p.length; return { cut, TP, FN, FP, TN, acc: (TP + TN) / n, tpr: TP / (TP + FN), tnr: TN / (TN + FP), fpr: FP / (TN + FP), precision: TP / Math.max(TP + FP, 1), cost: (cfn * FN + D.cost_fp * FP) / n, reject: (TP + FP) / n };
}
const FACTOR = D.pdo / Math.log(2), OFFSET = D.base_score - FACTOR * Math.log(D.base_odds);
const scale = pdo => { const f = pdo / Math.log(2); return { f, o: D.base_score - f * Math.log(D.base_odds) }; };
const scoreOf = (p, sc) => (sc || { f: FACTOR, o: OFFSET }).o + (sc || { f: FACTOR }).f * Math.log((1 - p) / p);
function irbK(pd, lgd, kind, q) {
  let R;
  if (kind === 'mortgage') R = 0.15; else if (kind === 'revolving') R = 0.04;
  else { const w = (1 - Math.exp(-35 * pd)) / (1 - Math.exp(-35)); R = 0.03 * w + 0.16 * (1 - w); }
  const arg = (Pinv(pd) + Math.sqrt(R) * Pinv(q || 0.999)) / Math.sqrt(1 - R), pb = Phi(arg);
  return { K: lgd * (pb - pd), R, arg, pb, w: (1 - Math.exp(-35 * pd)) / (1 - Math.exp(-35)) };
}
const condPD = (pd, y, rho) => Phi((Pinv(pd) - Math.sqrt(rho) * y) / Math.sqrt(1 - rho));
function mertonDD(V, Dd, s, mu, Tm) { Tm = Tm || 1; const dd = (Math.log(V / Dd) + (mu - 0.5 * s * s) * Tm) / (s * Math.sqrt(Tm)); return { dd, pd: Phi(-dd) }; }
function mertonEq(E, sE, Dd, r, Tm) {  // E = V N(d1) - D e^{-rT} N(d2), sigma_E E = N(d1) sigma_V V, Newton in logs
  Tm = Tm || 1;
  const F = z => { const V = Math.exp(z[0]), s = Math.exp(z[1]), d1 = (Math.log(V / Dd) + (r + 0.5 * s * s) * Tm) / (s * Math.sqrt(Tm)), d2 = d1 - s * Math.sqrt(Tm);
    return [V * Phi(d1) - Dd * Math.exp(-r * Tm) * Phi(d2) - E, Phi(d1) * s * V - sE * E]; };
  let z = [Math.log(E + Dd), Math.log(sE * E / (E + Dd))];
  for (let it = 0; it < 100; it++) {
    const f = F(z), h = 1e-7, J = [0, 1].map(j => { const zz = [...z]; zz[j] += h; const g = F(zz); return [(g[0] - f[0]) / h, (g[1] - f[1]) / h]; });
    const A = [[J[0][0], J[1][0]], [J[0][1], J[1][1]]], st = solve(A, f.map(v => -v)); z = [z[0] + st[0], z[1] + st[1]]; if (Math.max(Math.abs(st[0]), Math.abs(st[1])) < 1e-13) break;
  }
  const V = Math.exp(z[0]), s = Math.exp(z[1]), dd = (Math.log(V / Dd) + (r - 0.5 * s * s) * Tm) / (s * Math.sqrt(Tm));
  return { V, s, dd, pd: Phi(-dd) };
}

// ================================================================= South German Credit: data, WoE, scorecard
const SG = D.sgc, NR = SG.default.length, Y = SG.default, ALL = [...Array(NR).keys()];
const TE = D.te, TES = new Set(TE), TR = ALL.filter(i => !TES.has(i));
function binOf(v, x) { const e = D.bins[v]; if (!e) return x; for (let k = 0; k < e.length; k++) if (e[k] === null || x <= e[k]) return k; return e.length - 1; }
function woeTable(v, idx) {
  const m = new Map();
  for (const i of idx) { const b = binOf(v, SG[v][i]); const o = m.get(b) || { bin: b, n: 0, bad: 0 }; o.n++; o.bad += Y[i]; m.set(b, o); }
  const rows = [...m.values()].sort((a, b) => a.bin - b.bin);
  rows.forEach(r => { r.good = r.n - r.bad; const adj = r.good === 0 || r.bad === 0 ? 0.5 : 0; r.g = r.good + adj; r.b = r.bad + adj; });
  const G = sum(rows.map(r => r.g)), B = sum(rows.map(r => r.b));
  rows.forEach(r => { r.rate = r.bad / r.n; r.dg = r.g / G; r.db = r.b / B; r.woe = Math.log(r.dg / r.db); r.iv = (r.dg - r.db) * r.woe; });
  return { rows, iv: sum(rows.map(r => r.iv)), woe: new Map(rows.map(r => [r.bin, r.woe])) };
}
const strength = iv => (iv < 0.02 ? 'useless' : iv < 0.1 ? 'weak' : iv < 0.3 ? 'medium' : 'strong');
const WT = {};
const tables = (key, idx) => WT[key] || (WT[key] = Object.fromEntries(D.sgc_vars.map(v => [v, woeTable(v, idx)])));
const TAB_TR = tables('tr', TR);
const woeX = (idx, vars, tab) => idx.map(i => vars.map(v => { const w = tab[v].woe.get(binOf(v, SG[v][i])); return w === undefined ? 0 : w; }));
const ivSorted = tab => D.sgc_vars.map(v => [v, tab[v].iv]).sort((a, b) => b[1] - a[1]);
const SC = {};
function scorecard(thr) {            // sgc_scorecard with the IV threshold thr (0.10 in the lecture)
  const key = String(thr); if (SC[key]) return SC[key];
  const sel = ivSorted(TAB_TR).filter(([, iv]) => iv >= thr).map(([v]) => v), Xtr = woeX(TR, sel, TAB_TR), Xte = woeX(TE, sel, TAB_TR), ytr = TR.map(i => Y[i]), yte = TE.map(i => Y[i]);
  const lg = logitFit(Xtr, ytr), ld = ldaFit(Xtr, ytr);
  const o = { sel, lg, ld, ytr, yte, p_tr: lg.predict(Xtr), p_te: lg.predict(Xte), q_tr: ld.eta(Xtr).map(logistic), q_te: ld.eta(Xte).map(logistic),
    single_te: woeX(TE, ['status'], TAB_TR).map(r => -r[0]), single_tr: woeX(TR, ['status'], TAB_TR).map(r => -r[0]) };
  o.auc_te = auc(o.p_te, yte); o.auc_tr = auc(o.p_tr, ytr);
  return (SC[key] = o);
}
const S0 = () => scorecard(D.iv_min);
let SMALL = null;
const SMALLV = ['duration_years', 'age_decades', 'amount_1000', 'no_account', 'negative_balance'];
const smallX = i => [SG.duration[i] / 12, SG.age[i] / 10, SG.amount[i] / 1000, +(SG.status[i] === 1), +(SG.status[i] === 2)];
function smallLogit(drop) {
  const keep = SMALLV.map((v, j) => j).filter(j => !(drop || []).includes(SMALLV[j])), X = ALL.map(i => { const r = smallX(i); return keep.map(j => r[j]); });
  const f = logitFit(X, Y), f0 = logitFit(ALL.map(() => []), Y);
  return { f, names: ['const', ...keep.map(j => SMALLV[j])], ll0: f0.ll, lr: 2 * (f.ll - f0.ll), df: keep.length };
}
const small = () => SMALL || (SMALL = smallLogit([]));

// ================================================================= Taiwan
const TW = D.tw;

// ================================================================= formatting, tables, panels
const SEP = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, SEP).replace('-', '−');
const pc = (v, d) => nf(100 * v, d === undefined ? 1 : d) + '%';
const pcE = (v, d) => (100 * v).toFixed(d === undefined ? 1 : d) + '%';
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.span ? ` colspan="${o.span}"` : ''}>${o.v}</${tag}>`; };
  const hs = Array.isArray(head[0]) ? head : [head];
  $(id).innerHTML = `<table class="xt"><thead>${hs.map(h => `<tr>${h.map(x => cell(x, 'th')).join('')}</tr>`).join('')}</thead><tbody>` +
    rows.map(r => `<tr>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
  o = o || {};
  const el = $(id), stack = n > 1 && el.clientWidth < (o.minW || 560);
  el.style.height = (stack ? (o.hs || 270) * n + 110 : (o.h || 380)) + 'px';
  const g = stack ? 0.2 : (o.gap || 0.1), w = (1 - g * (n - 1)) / n, L = {}, ax = [];
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
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const kSel = id => $(id).value;
const MG = { l: 56, r: 12, t: 26, b: 40 };
const M0 = { l: 56, r: 12, t: 14, b: 40 };
const wrap = (s, w) => { const out = []; let line = ''; String(s).split(' ').forEach(t => { if ((line + ' ' + t).trim().length > w && line) { out.push(line); line = t; } else line = (line + ' ' + t).trim(); }); out.push(line); return out.join('<br>'); };
const VLAB = { status: 'checking account', credit_history: 'credit history', savings: 'savings', employment_duration: 'employment duration', property: 'property', housing: 'housing',
  duration: 'duration', amount: 'amount', age: 'age', purpose: 'purpose' };
const BINLAB = { duration: ['≤ 12', '13–18', '19–24', '25–36', '> 36'], amount: ['≤ 1300', '1301–2000', '2001–3000', '3001–5000', '> 5000'], age: ['≤ 25', '26–30', '31–35', '36–45', '> 45'] };
const CHLAB = { credit_history: ['delays in paying off in the past', 'critical account or other credits elsewhere', 'no credits taken or all paid back duly', 'existing credits paid back duly till now', 'all credits at this bank paid back duly'],
  savings: [null, 'no savings account or unknown', '< 100 DM', '100–500 DM', '500–1000 DM', '≥ 1000 DM'], employment_duration: [null, 'unemployed', '< 1 year', '1–4 years', '4–7 years', '≥ 7 years'] };
const codeLab = (v, c) => (v === 'status' ? D.status_lab[c] : BINLAB[v] ? BINLAB[v][c] : CHLAB[v] ? CHLAB[v][c] : (D.labels[v] || {})[c] || String(c));
const vline = (x, y0, y1, name, color, dash, ax) => Object.assign({ x: [x, x], y: [y0, y1], mode: 'lines', line: { color, width: 2, dash: dash || 'solid' }, name, hoverinfo: 'skip' }, ax ? { xaxis: ax.x, yaxis: ax.y } : {});
const SIG = (p, k) => (p < 0.001 ? (LANG === 'ro' ? '< 0,001' : '< 0.001') : nf(p, k || 3));

// ================================================================= 1. PD in good and bad years: the one-factor model
function cpBlock() {
  const rho = +$('cp-r').value, xs = lin(0.001, 0.999, 400), nw = single('cp-chart', 360); label('cp-r').textContent = nf(rho, 2);
  const tr = [[-3, RED, 'solid', 'bad state of the economy, y = -3'], [0, BLUE, 'dash', 'typical state, y = 0'], [3, GREEN, 'dashdot', 'good state, y = 3']].map(([y, c, d, nm]) =>
    ({ x: xs, y: xs.map(p => condPD(p, y, rho)), mode: 'lines', line: { color: c, width: 2.2, dash: d }, name: nm, hovertemplate: 'PD %{x:.3f}: %{y:.3f}<extra></extra>' }));
  const yI = Pinv(0.001);
  tr.push({ x: [0.02, 0.02, 0.02], y: [-3, 0, 3].map(y => condPD(0.02, y, rho)), mode: 'markers', marker: { color: AMBER, size: 10, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `PD = 2%, rho = ${rho.toFixed(2)}`, hovertemplate: 'PD 2%: %{y:.4f}<extra></extra>' });
  plot('cp-chart', tr, layout({ xaxis: AX('unconditional PD', { range: [0, 1] }), yaxis: AX('PD given the state y', { range: [0, 1] }), legend: LEG(nw, -0.18), margin: M0 }));
  out('cp-out', [T.cp(nf(rho, 2), pc(condPD(0.02, -3, rho)), pc(condPD(0.02, 0, rho), 2), (condPD(0.02, 3, rho) < 5e-7 ? (LANG === 'ro' ? '< 0,0001%' : '< 0.0001%') : nf(100 * condPD(0.02, 3, rho), 4) + '%'), nf(yI, 2), pc(condPD(0.02, yI, rho), 2), rho !== 0.2)]);
}

// ================================================================= 2. expected loss and IRB capital as functions of PD
const KINDS = [['other retail', RED, 'solid'], ['mortgage', BLUE, 'dash'], ['revolving', GREEN, 'dashdot']];
function irBlock() {
  const lgd = +$('ir-l').value, q = +kSel('ir-q'), pds = lin(0.0005, 0.25, 400), nw = single('ir-chart', 360); label('ir-l').textContent = pc(lgd, 0);
  const tr = [{ x: pds.map(p => 100 * p), y: pds.map(p => 100 * lgd * p), mode: 'lines', line: { color: AMBER, width: 2 }, name: 'expected loss EL = PD x LGD', hovertemplate: 'PD %{x:.2f}%: %{y:.2f}%<extra></extra>' }];
  KINDS.forEach(([k, c, d]) => tr.push({ x: pds.map(p => 100 * p), y: pds.map(p => 100 * irbK(p, lgd, k, q).K), mode: 'lines', line: { color: c, width: 2.2, dash: d }, name: `capital K, ${k}`, hovertemplate: 'PD %{x:.2f}%: %{y:.2f}%<extra></extra>' }));
  plot('ir-chart', tr, layout({ xaxis: AX('PD (%)'), yaxis: AX('% of EAD'), legend: LEG(nw, -0.18), margin: M0, annotations: [{ text: `LGD = ${(100 * lgd).toFixed(0)}%, confidence level ${(100 * q).toPrecision(4).replace(/0+$/, '').replace(/\.$/, '')}%`, xref: 'paper', yref: 'paper', x: 0.01, y: 0.98, showarrow: false, xanchor: 'left', font: { size: 11, color: DARK } }] }));
  const rows = [0.01, 0.05, 0.1].map(p => [pc(p, 0), pc(lgd * p, 2), ...KINDS.map(([k]) => pc(irbK(p, lgd, k, q).K, 2))]);
  out('ir-out', [T.ir(pc(lgd, 0), nf(100 * q, q === 0.9999 ? 2 : 1), rows, lgd !== 0.45 || q !== 0.999)]);
}

// ================================================================= 3. bad rate by checking account (and other categorical variables)
const BARC = [RED, ORANGE, AMBER, BLUE, GREEN, PURPLE, TEAL, DARK, RED, ORANGE, AMBER];
function dsBlock() {
  const v = kSel('ds-v'), t = woeTable(v, ALL).rows, nw = single('ds-chart', 380), labs = t.map(r => wrap(codeLab(v, r.bin), nw ? 12 : 18));
  plot('ds-chart', [{ type: 'bar', x: labs, y: t.map(r => 100 * r.rate), marker: { color: t.map((_, i) => BARC[i]) }, text: t.map(r => `${(100 * r.rate).toFixed(1)}% (n = ${r.n})`), textposition: 'outside', textfont: { size: 11, color: DARK }, cliponaxis: false, name: 'bad rate in the group', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: [labs[0], labs[labs.length - 1]], y: [30, 30], mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: 'bad rate of the sample (30%)', hoverinfo: 'skip' }],
  layout({ xaxis: AX(VLAB[v], { type: 'category' }), yaxis: AX('bad credits (%)', { range: [0, Math.max(60, 12 + Math.max(...t.map(r => 100 * r.rate)))] }), legend: LEG(nw, -0.3), margin: M0, bargap: 0.35 }));
  const hi = t.reduce((a, r) => (r.rate > a.rate ? r : a)), lo = t.reduce((a, r) => (r.rate < a.rate ? r : a));
  out('ds-out', [T.ds(v, T.lab(v, hi.bin), pc(hi.rate), hi.n, T.lab(v, lo.bin), pc(lo.rate), lo.n, nf(hi.rate / lo.rate, 1), v !== 'status')]);
}

// ================================================================= 4. WoE of the credit duration
function wdBlock() {
  const v = kSel('wd-v'), s = kSel('wd-s'), t = woeTable(v, s === 'train' ? TR : ALL), nw = single('wd-chart', 380), labs = t.rows.map(r => `${BINLAB[v][r.bin]}<br>(n = ${r.n})`);
  plot('wd-chart', [{ type: 'bar', x: labs, y: t.rows.map(r => r.woe), marker: { color: t.rows.map(r => (r.woe > 0 ? BLUE : RED)) }, name: 'WoE = ln(share of goods / share of bads)', hovertemplate: '%{x}: WoE %{y:.3f}<extra></extra>' },
    { x: labs, y: t.rows.map(r => 100 * r.rate), yaxis: 'y2', mode: 'lines+markers', line: { color: AMBER, width: 2 }, marker: { size: 8 }, name: 'bad rate (%, right axis)', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
  layout({ xaxis: AX(`${VLAB[v]} of the credit (${v === 'amount' ? 'DM' : v === 'age' ? 'years' : 'months'})`, { type: 'category' }), yaxis: AX('weight of evidence', { zeroline: true, zerolinecolor: DARK }),
    yaxis2: AX('bad rate (%)', { overlaying: 'y', side: 'right', range: [0, 60], showgrid: false }), legend: LEG(nw, -0.3), margin: { l: 56, r: 50, t: 14, b: 40 } }));
  out('wd-out', [T.wd(v, s, nf(t.iv, 3), T.cls(strength(t.iv)), t.rows.map(r => [BINLAB[v][r.bin], nf(r.woe, 2), pc(r.rate, 0)]), v !== 'duration' || s !== 'full')]);
}

// ================================================================= 5. information value of the 20 variables
const SCOL = { strong: RED, medium: ORANGE, weak: AMBER, useless: BLUE };
function ivBlock() {
  const s = kSel('iv-s'), tab = s === 'full' ? tables('all', ALL) : s === 'train' ? TAB_TR : tables('te', TE), iv = ivSorted(tab).reverse(), nw = single('iv-chart', 470), tr = [];
  ['strong', 'medium', 'weak', 'useless'].forEach(cl => { const r = iv.filter(([, x]) => strength(x) === cl);
    tr.push({ type: 'bar', orientation: 'h', y: r.map(([v]) => v.replace(/_/g, ' ')), x: r.map(([, x]) => x), marker: { color: SCOL[cl] }, name: cl, hovertemplate: '%{y}: IV %{x:.4f}<extra></extra>' }); });
  const shapes = [0.02, 0.1, 0.3].map(x => ({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: DARK, width: 1, dash: 'dot' } }));
  plot('iv-chart', tr, layout({ xaxis: AX('information value (IV)'), yaxis: AX('', { type: 'category', categoryorder: 'array', categoryarray: iv.map(([v]) => v.replace(/_/g, ' ')), tickfont: { size: 10.5, color: DARK } }), shapes, legend: LEG(nw, -0.12), margin: { l: 150, r: 12, t: 10, b: 40 }, barmode: 'overlay' }));
  const d = iv.slice().reverse(), sel = d.filter(([, x]) => x >= D.iv_min).map(([v]) => T.vn(v));
  out('iv-out', [T.iv(s, T.vn(d[0][0]), nf(d[0][1], 3), T.vn(d[1][0]), nf(d[1][1], 3), sel, d.filter(([, x]) => x < 0.02).map(([v]) => T.vn(v)), s !== 'full')]);
}

// ================================================================= 6. a logistic curve: PD and duration
const LC = {};
function lcFit(v) { return LC[v] || (LC[v] = logitFit(ALL.map(i => [SG[v][i]]), Y)); }
function lcBlock() {
  const v = kSel('lc-v'), mn = +kSel('lc-m'), f = lcFit(v), nw = single('lc-chart', 360), g = new Map();
  ALL.forEach(i => { const x = SG[v][i], o = g.get(x) || { n: 0, b: 0 }; o.n++; o.b += Y[i]; g.set(x, o); });
  const pts = [...g.entries()].filter(([, o]) => o.n >= mn).sort((a, b) => a[0] - b[0]), xs = v === 'age' ? lin(18, 76, 300) : lin(0, 75, 300);
  plot('lc-chart', [{ x: pts.map(p => p[0]), y: pts.map(p => p[1].b / p[1].n), mode: 'markers', marker: { color: hexA(BLUE, 0.7), size: pts.map(p => Math.sqrt(3 * p[1].n) * 1.3) }, name: `observed bad rate (${v === 'age' ? 'ages' : 'durations'} with at least ${mn} credit${mn > 1 ? 's' : ''})`, customdata: pts.map(p => p[1].n), hovertemplate: '%{x}: %{y:.3f} (n = %{customdata})<extra></extra>' },
    { x: xs, y: xs.map(x => logistic(f.b[0] + f.b[1] * x)), mode: 'lines', line: { color: RED, width: 2.2 }, name: `logit: PD = 1 / (1 + exp(-(${f.b[0].toFixed(2)} ${f.b[1] < 0 ? '-' : '+'} ${Math.abs(f.b[1]).toFixed(3)} ${v})))`, hoverinfo: 'skip' }],
  layout({ xaxis: AX(v === 'age' ? 'age (years)' : 'duration (months)'), yaxis: AX('probability of default', { range: [0, 1] }), legend: LEG(nw, -0.2), margin: M0 }));
  const u = v === 'age' ? 10 : 12, [a, b] = v === 'age' ? [25, 60] : [12, 48];
  out('lc-out', [T.lc(v, nf(f.b[0], 3), nf(f.b[1], 4), nf(f.se[1], 4), u, nf(Math.exp(u * f.b[1]), 2), a, pc(logistic(f.b[0] + f.b[1] * a)), b, pc(logistic(f.b[0] + f.b[1] * b)), pts.length, mn, v !== 'duration' || mn !== 5)]);
}

// ================================================================= 7. odds ratios with confidence intervals
const ORL = { duration_years: 'duration (+1 year)', age_decades: 'age (+10 years)', amount_1000: 'amount (+1000 DM)', no_account: 'no checking account', negative_balance: 'balance < 0 DM' };
function orBlock() {
  const lvl = +kSel('or-l'), z = Pinv(1 - (1 - lvl) / 2), S = small(), nw = single('or-chart', 330), names = S.names.slice(1), ys = names.map((_, i) => names.length - 1 - i);
  const b = names.map((_, i) => S.f.b[i + 1]), se = names.map((_, i) => S.f.se[i + 1]), o = b.map(Math.exp), lo = b.map((v, i) => Math.exp(v - z * se[i])), hi = b.map((v, i) => Math.exp(v + z * se[i]));
  plot('or-chart', [{ x: names.flatMap((_, i) => [lo[i], hi[i], null]), y: ys.flatMap(v => [v, v, null]), mode: 'lines+markers', line: { color: BLUE, width: 2.5 }, marker: { symbol: 'line-ns', size: 10, line: { color: BLUE, width: 2 } }, name: `${(100 * lvl).toFixed(0)}% confidence interval`, hoverinfo: 'skip' },
    { x: o, y: ys, mode: 'markers', marker: { color: RED, size: 10 }, name: 'odds ratio exp(b)', customdata: names.map(n => ORL[n]), hovertemplate: '%{customdata}: %{x:.3f}<extra></extra>' },
    { x: [1, 1], y: [-0.5, names.length - 0.5], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'no effect (odds ratio 1)', hoverinfo: 'skip' }],
  layout({ xaxis: AX('odds ratio (log scale)', { type: 'log', tickvals: [0.5, 1, 2, 4, 8], ticktext: ['0.5', '1', '2', '4', '8'] }), yaxis: AX('', { tickvals: ys, ticktext: names.map(n => ORL[n]) }), legend: LEG(nw, -0.25), margin: { l: 150, r: 12, t: 10, b: 40 } }));
  const inc = names.filter((_, i) => lo[i] <= 1 && hi[i] >= 1).map(n => T.sv(n));
  out('or-out', [T.or(nf(100 * lvl, 0), nf(z, 3), names.map((n, i) => [T.sv(n), nf(o[i], 2), nf(lo[i], 2), nf(hi[i], 2)]), inc, lvl !== 0.95)]);
}

// ================================================================= 8. Fisher's boundary and the logit boundary
let LD2 = null;
function ld2() { if (LD2) return LD2; const X = ALL.map(i => [SG.duration[i], SG.age[i]]); return (LD2 = { ld: ldaFit(X, Y), lg: logitFit(X, Y) }); }
function ldBlock() {
  const p = +$('ld-p').value, t = Math.log(p / (1 - p)), { ld, lg } = ld2(), nw = single('ld-chart', 400), J = D.jitter, xs = lin(4, 72, 50); label('ld-p').textContent = pc(p, 0);
  const g0 = ALL.filter(i => !Y[i]), g1 = ALL.filter(i => Y[i]);
  const bL = x => (t - ld.c - ld.w[0] * x) / ld.w[1], bG = x => (t - lg.b[0] - lg.b[1] * x) / lg.b[2];
  plot('ld-chart', [{ x: g0.map(i => J[0][i]), y: g0.map(i => J[1][i]), mode: 'markers', marker: { color: hexA(BLUE, 0.45), size: 5 }, name: 'good credit', hoverinfo: 'skip' },
    { x: g1.map(i => J[0][i]), y: g1.map(i => J[1][i]), mode: 'markers', marker: { color: hexA(RED, 0.65), size: 6, symbol: 'x' }, name: 'bad credit', hoverinfo: 'skip' },
    { x: xs, y: xs.map(bL), mode: 'lines', line: { color: GREEN, width: 2.4 }, name: `Fisher LDA: PD = ${(100 * p).toFixed(0)}%`, hovertemplate: 'duration %{x:.0f}: age %{y:.1f}<extra></extra>' },
    { x: xs, y: xs.map(bG), mode: 'lines', line: { color: ORANGE, width: 2.4, dash: 'dash' }, name: `logit: PD = ${(100 * p).toFixed(0)}%`, hovertemplate: 'duration %{x:.0f}: age %{y:.1f}<extra></extra>' }],
  layout({ xaxis: AX('duration (months)', { range: [2, 74] }), yaxis: AX('age (years)', { range: [17, 77] }), legend: LEG(nw, -0.17), margin: M0 }));
  const dL = a => (t - ld.c - ld.w[1] * a) / ld.w[0], dG = a => (t - lg.b[0] - lg.b[2] * a) / lg.b[1];
  out('ld-out', [T.ld(pc(p, 0), nf(-ld.w[0] / ld.w[1], 2), nf(-lg.b[1] / lg.b[2], 2), nf(dL(30), 1), nf(dG(30), 1), nf(dL(50), 1), nf(dG(50), 1), p !== 0.3)]);
}

// ================================================================= 9. scores of the test sample
function histo(v, e) { const c = new Array(e.length - 1).fill(0); v.forEach(x => { let k = Math.floor((x - e[0]) / (e[1] - e[0])); if (k >= c.length) k = c.length - 1; if (k >= 0) c[k]++; }); return c; }
function sdBlock() {
  const S = S0(), cut = +kSel('sd-c'), bw = +kSel('sd-b'), sc = S.p_te.map(p => scoreOf(p)), y = S.yte, cs = scoreOf(cut), nw = single('sd-chart', 360);
  const e = []; for (let x = Math.floor(Math.min(...sc) / 10) * 10; x <= Math.ceil(Math.max(...sc) / 10) * 10 + 10 + 1e-9; x += bw) e.push(x);
  const mid = e.slice(0, -1).map(x => x + bw / 2), hg = histo(sc.filter((_, i) => !y[i]), e), hb = histo(sc.filter((_, i) => y[i]), e), top = Math.max(...hg, ...hb);
  plot('sd-chart', [{ type: 'bar', x: mid, y: hg, width: bw, marker: { color: hexA(BLUE, 0.6) }, name: 'good credits (test sample)', hovertemplate: '%{x}: %{y}<extra></extra>' },
    { type: 'bar', x: mid, y: hb, width: bw, marker: { color: hexA(RED, 0.6) }, name: 'bad credits (test sample)', hovertemplate: '%{x}: %{y}<extra></extra>' },
    vline(cs, 0, top * 1.05, `cut-off at PD = ${Math.abs(cut - 1 / 6) < 1e-6 ? '1/6' : cut}: ${cs.toFixed(0)} points`, GREEN, 'dash')],
  layout({ barmode: 'overlay', xaxis: AX(nw ? 'score (points; 600 = odds 50:1)' : 'score (points; 600 = good:bad odds of 50:1, +20 points = double odds)'), yaxis: AX('number of credits'), legend: LEG(nw, -0.2), margin: M0 }));
  const acc = sc.map((s, i) => S.p_te[i] <= cut), ga = acc.filter((a, i) => a && !y[i]).length, ba = acc.filter((a, i) => a && y[i]).length;
  out('sd-out', [T.sd(Math.abs(cut - 1 / 6) < 1e-6 ? '1/6' : nf(cut, 1), fint(cs), ga, 210 - ga, ba, 90 - ba, fint(Math.min(...sc)), fint(Math.max(...sc)), fint(mean(sc.filter((_, i) => !y[i]))), fint(mean(sc.filter((_, i) => y[i]))), Math.abs(cut - 1 / 6) > 1e-6 || bw !== 10)]);
}

// ================================================================= 10. expected cost and the choice of the cut-off
function coBlock() {
  const S = S0(), cfn = +kSel('co-f'), p = S.p_te, y = S.yte, cuts = lin(0.02, 0.9, 89), cm = cuts.map(c => confusion(p, y, c, cfn)), nw = single('co-chart', 370), bay = D.cost_fp / (D.cost_fp + cfn), cb = confusion(p, y, bay, cfn), pb = mean(y);
  const i = cm.reduce((a, c, k) => (c.cost < cm[a].cost ? k : a), 0), top = Math.max(cfn * pb, 1) * 1.05;
  plot('co-chart', [{ x: cuts, y: cm.map(c => c.cost), mode: 'lines', line: { color: RED, width: 2.2 }, name: 'expected cost per applicant', hovertemplate: 'cut-off %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [0, 0.92], y: [cfn * pb, cfn * pb], mode: 'lines', line: { color: AMBER, width: 1.5, dash: 'dash' }, name: 'accept everybody', hoverinfo: 'skip' },
    { x: [0, 0.92], y: [1 - pb, 1 - pb], mode: 'lines', line: { color: PURPLE, width: 1.5, dash: 'dash' }, name: 'reject everybody', hoverinfo: 'skip' },
    vline(bay, 0, top, `Bayes cut-off PD = 1/${(1 + cfn).toFixed(0)}`, GREEN, 'dot'),
    { x: cuts, y: cm.map(c => 100 * c.reject), yaxis: 'y2', mode: 'lines', line: { color: BLUE, width: 1.8 }, name: 'rejected applicants (%, right axis)', hovertemplate: 'cut-off %{x:.2f}: %{y:.1f}%<extra></extra>' },
    { x: [bay], y: [100 * cb.reject], yaxis: 'y2', mode: 'markers', marker: { color: GREEN, size: 10, symbol: 'diamond' }, name: 'rejected at the Bayes cut-off', hovertemplate: '%{y:.1f}%<extra></extra>' }],
  layout({ xaxis: AX('PD cut-off (reject if PD > cut-off)', { range: [0, 0.92] }), yaxis: AX('cost per applicant', { range: [0, top] }), yaxis2: AX('rejected (%)', { overlaying: 'y', side: 'right', range: [0, 100], showgrid: false }), legend: LEG(nw, -0.2), margin: { l: 56, r: 50, t: 14, b: 40 } }));
  out('co-out', [T.co(cfn, `1/${1 + cfn}`, nf(bay, 3), nf(cb.cost, 3), pc(cb.reject), nf(cuts[i], 2), nf(cm[i].cost, 3), nf(cfn * pb, 2), nf(1 - pb, 2), cfn !== 5)]);
}

// ================================================================= 11. ROC curves on the test sample
function rcBlock() {
  const S = S0(), c = +$('rc-c').value, smp = kSel('rc-s'), te = smp === 'test', y = te ? S.yte : S.ytr, sets = [['WoE logit', te ? S.p_te : S.p_tr, COL.logit, 'solid'], ['Fisher LDA', te ? S.q_te : S.q_tr, COL.lda, 'dash'], ['checking account only', te ? S.single_te : S.single_tr, COL.single, 'solid']];
  const nw = single('rc-chart', 420), A = sets.map(s => auc(s[1], y)), cm = confusion(sets[0][1], y, c); label('rc-c').textContent = nf(c, 2);
  const tr = sets.map(([nm, s, col, d], k) => { const r = rocPts(s, y); return { x: r.fpr, y: r.tpr, mode: 'lines', line: { color: col, width: 2.2, dash: d }, name: `${nm} (AUC = ${A[k].toFixed(3)})`, hoverinfo: 'skip' }; });
  tr.push({ x: [0, 1], y: [0, 1], mode: 'lines', line: { color: COL.random, width: 1.5, dash: 'dot' }, name: 'random score (AUC = 0.5)', hoverinfo: 'skip' },
    { x: [cm.fpr], y: [cm.tpr], mode: 'markers', marker: { color: AMBER, size: 13, symbol: 'circle-open', line: { width: 3 } }, name: `WoE logit, reject if PD > ${c.toFixed(2)}`, hovertemplate: 'FPR %{x:.3f}, TPR %{y:.3f}<extra></extra>' });
  plot('rc-chart', tr, layout({ xaxis: AX('false positive rate: goods rejected (1 - specificity)', { range: [0, 1] }), yaxis: AX('true positive rate: bads rejected', { range: [0, 1] }), legend: LEG(nw, -0.16), margin: M0 }));
  const dl = delong(sets[0][1], y);
  out('rc-out', [T.rc(smp, nf(A[0], 3), nf(A[1], 3), nf(A[2], 3), nf(dl.lo, 3), nf(dl.hi, 3), nf(2 * A[0] - 1, 3), nf(c, 2), pc(cm.tpr), pc(cm.fpr), fint(sum(y) * (y.length - sum(y))), !te || c !== 0.5)]);
}

// ================================================================= 12. CAP on the test sample
function caBlock() {
  const S = S0(), xr = +$('ca-x').value, C0 = capPts(S.p_te, S.yte), k = Math.round(xr * (C0.x.length - 1)), nw = single('ca-chart', 420); label('ca-x').textContent = pc(xr, 0);
  plot('ca-chart', [{ x: [0, 1], y: [0, 1], mode: 'lines', line: { color: COL.random, width: 1.5, dash: 'dot' }, name: 'random score (AR = 0)', hoverinfo: 'skip' },
    { x: C0.x, y: C0.h, mode: 'lines', line: { color: COL.logit, width: 2.2 }, fill: 'tonexty', fillcolor: hexA(COL.logit, 0.12), name: `WoE logit (AR = ${C0.ar.toFixed(3)})`, hoverinfo: 'skip' },
    { x: [0, C0.pb, 1], y: [0, 1, 1], mode: 'lines', line: { color: COL.perfect, width: 2, dash: 'dash' }, name: 'perfect model (AR = 1)', hoverinfo: 'skip' },
    { x: [C0.x[k]], y: [C0.h[k]], mode: 'markers', marker: { color: AMBER, size: 13, symbol: 'circle-open', line: { width: 3 } }, name: `riskiest ${(100 * xr).toFixed(0)}% rejected`, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('share of applicants, riskiest first', { range: [0, 1] }), yaxis: AX('share of all bads captured', { range: [0, 1.02] }), legend: LEG(nw, -0.16), margin: M0 }));
  out('ca-out', [T.ca(pc(xr, 0), Math.round(xr * 300), pc(C0.h[k]), Math.round(C0.h[k] * 90), pc(Math.min(1, xr / C0.pb)), nf(C0.ar, 3), nf(2 * S.auc_te - 1, 3), xr !== 0.3)]);
}

// ================================================================= 13. KS on the test sample
function ksBlock() {
  const S = S0(), s0 = +$('ks-s').value, sc = S.p_te.map(p => scoreOf(p)), y = S.yte, K = ksStat(sc, y), u = K.u, nw = single('ks-chart', 360); label('ks-s').textContent = String(s0);
  const F1 = u.map(K.F1), F0 = u.map(K.F0); let i = 0; u.forEach((_, k) => { if (F1[k] - F0[k] > F1[i] - F0[i]) i = k; });
  const a1 = K.F1(s0), a0 = K.F0(s0);
  plot('ks-chart', [{ x: u, y: F1, mode: 'lines', line: { color: COL.bad, width: 2, shape: 'hv' }, name: 'bads: share with score <= s', hoverinfo: 'skip' },
    { x: u, y: F0, mode: 'lines', line: { color: COL.good, width: 2, shape: 'hv' }, name: 'goods: share with score <= s', hoverinfo: 'skip' },
    { x: [u[i], u[i]], y: [F0[i], F1[i]], mode: 'lines', line: { color: GREEN, width: 3 }, name: `KS = ${(F1[i] - F0[i]).toFixed(3)} at ${u[i].toFixed(0)} points`, hoverinfo: 'skip' },
    { x: [s0, s0], y: [a0, a1], mode: 'lines+markers', line: { color: AMBER, width: 3, dash: 'dot' }, marker: { size: 7 }, name: `gap at s = ${s0}: ${(a1 - a0).toFixed(3)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('score s (points)'), yaxis: AX('cumulative share', { range: [0, 1.02] }), legend: LEG(nw, -0.2), margin: M0 }));
  out('ks-out', [T.ks(nf(F1[i] - F0[i], 3), fint(u[i]), String(s0), pc(a1), pc(a0), nf(a1 - a0, 3), s0 !== 516)]);
}

// ================================================================= 14. calibration on the Taiwan data
function cbBlock() {
  const k = +kSel('cb-g'), c = calTable(TW.p, TW.y, TW.order, k), nw = single('cb-chart', 420);
  const se = c.G.map(g => Math.sqrt(g.rate * (1 - g.rate) / g.n));
  plot('cb-chart', [{ x: [0, 1], y: [0, 1], mode: 'lines', line: { color: COL.random, width: 1.5, dash: 'dot' }, name: 'perfect calibration', hoverinfo: 'skip' },
    { x: c.G.map(g => g.pd), y: c.G.map(g => g.rate), mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 7 }, error_y: { type: 'data', array: se.map(v => 1.96 * v), color: BLUE, thickness: 1.5, width: 4 }, name: `${k} groups of ${fint(9000 / k)} test clients (95% interval)`, hovertemplate: 'mean PD %{x:.3f}: observed %{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('mean predicted PD in the group', { range: [0, 0.8] }), yaxis: AX('observed default rate in the group', { range: [0, 0.8] }), legend: LEG(nw, -0.16), margin: M0 }));
  table('cb-tab', T.h.cal, c.G.map((g, j) => [j + 1, fint(g.n), pc(g.pd), pc(g.rate), fint(g.bad), nf(g.exp, 1), { v: nf((g.bad - g.exp) ** 2 / (g.exp * (1 - g.pd)), 2), c: (g.bad - g.exp) ** 2 / (g.exp * (1 - g.pd)) > 3.84 ? 'bad' : '' }]));
  const b = brier(TW.p, TW.y), br = brier(TW.p.map(() => TW.rate_train), TW.y);
  out('cb-out', [T.cb(k, nf(c.hl, 2), c.df, SIG(c.p), nf(b, 3), nf(br, 3), pc(1 - b / br, 0), pc(c.G[0].pd), pc(c.G[0].rate), k !== 10)]);
}

// ================================================================= 15. training and test AUC in repeated cross-validation
const CVM = ['WoE logit', 'LDA', 'small logit', 'status only'], CVC = [COL.logit, COL.lda, COL.small, COL.single];
function cvBlock() {
  const r = +kSel('cv-r'), nf_ = 5 * r, nw = single('cv-chart', 380), tr = [], M = {};
  const jit = (i, j) => (((i * 7919 + j * 104729) % 1000) / 1000 - 0.5) * 0.14;
  CVM.forEach((m, i) => [['train', 'circle'], ['test', 'x']].forEach(([part, mk], j) => {
    const v = D.cv[m][part].slice(0, nf_), x0 = i + (j - 0.5) * 0.36, mu = mean(v); M[m + part] = { mu, sd: Math.sqrt(sum(v.map(a => (a - mu) ** 2)) / Math.max(v.length - 1, 1)) };
    tr.push({ x: v.map((_, q) => x0 + jit(q, j + 2 * i)), y: v, mode: 'markers', marker: { color: hexA(CVC[i], 0.6), size: 6, symbol: mk }, name: part === 'train' ? 'training folds' : 'test folds', legendgroup: part, showlegend: i === 0, hovertemplate: `${m}, ${part}: %{y:.3f}<extra></extra>` },
      { x: [x0 - 0.14, x0 + 0.14], y: [mu, mu], mode: 'lines', line: { color: DARK, width: 2.5 }, name: 'mean', legendgroup: 'mean', showlegend: i === 0 && j === 0, hoverinfo: 'skip' });
  }));
  plot('cv-chart', tr, layout({ xaxis: AX('', { tickvals: [0, 1, 2, 3], ticktext: CVM, range: [-0.5, 3.5] }), yaxis: AX('AUC'), legend: LEG(nw, -0.14), margin: M0 }));
  out('cv-out', [T.cv(r, nf_, CVM.map(m => [m, nf(M[m + 'train'].mu, 3), nf(M[m + 'test'].mu, 3), nf(M[m + 'test'].sd, 3)]), nf(M['WoE logittrain'].mu - M['WoE logittest'].mu, 3), r !== 20)]);
}

// ================================================================= 16. approval and default by group: Taiwan
const FG = ['men', 'women', 'age <= 25', 'age 26-35', 'age 36-50', 'age > 50'];
function fairGroups(a) {
  const cut = quantile(TW.p, a), G = FG.map(() => ({ n: 0, bad: 0, ps: 0, ok: 0, good: 0, okg: 0 }));
  TW.p.forEach((p, i) => { const ok = p <= cut, ag = TW.age[i], gs = [TW.sex[i] === 1 ? 0 : 1, ag <= 25 ? 2 : ag <= 35 ? 3 : ag <= 50 ? 4 : 5];
    gs.forEach(k => { const g = G[k]; g.n++; g.bad += TW.y[i]; g.ps += p; g.ok += ok; if (!TW.y[i]) { g.good++; g.okg += ok; } }); });
  return { cut, G: G.map(g => ({ n: g.n, rate: g.bad / g.n, pd: g.ps / g.n, approve: g.ok / g.n, tpr: g.okg / g.good })) };
}
function faBlock() {
  const a = +$('fa-a').value, F = fairGroups(a), nw = single('fa-chart', 380), labs = FG.map((g, i) => `${g}<br>(n = ${F.G[i].n})`); label('fa-a').textContent = pc(a, 0);
  plot('fa-chart', [['rate', RED, 'observed default rate'], ['pd', AMBER, 'mean predicted PD'], ['approve', BLUE, `approved (cut-off: best ${(100 * a).toFixed(0)}%)`]].map(([k, c, nm]) =>
    ({ type: 'bar', x: labs, y: F.G.map(g => 100 * g[k]), marker: { color: c }, name: nm, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })),
  layout({ barmode: 'group', xaxis: AX('', { type: 'category' }), yaxis: AX('%', { range: [0, 100] }), legend: LEG(nw, -0.24), margin: M0 }));
  const [m, w] = F.G, ap = F.G.map(g => g.approve);
  out('fa-out', [T.fa(pc(a, 0), nf(F.cut, 3), pc(w.approve), pc(m.approve), nf(100 * (w.approve - m.approve), 1), pc(w.tpr), pc(m.tpr), pc(Math.min(...ap)), T.fg(FG[ap.indexOf(Math.min(...ap))]), a !== 0.75)]);
}

// ================================================================= 17. Merton: asset paths and PD
function meBlock() {
  const mu = +$('me-m').value, s = +kSel('me-s'), dt = 1 / 252, Pn = panels('me-chart', 2, { h: 380 }), tg = Array.from({ length: 253 }, (_, i) => i * dt); label('me-m').textContent = pc(mu, 0);
  const paths = D.merton_z.map(z => { const v = [100]; let c = 0; z.forEach(e => { c += (mu - 0.5 * s * s) * dt + s * Math.sqrt(dt) * e; v.push(100 * Math.exp(c)); }); return v; });
  const red = paths.filter(p => p[252] < 70), blue = paths.filter(p => p[252] >= 70), cat = ps => ({ x: ps.flatMap(() => [...tg, null]), y: ps.flatMap(p => [...p, null]) });
  const tr = [on2(Pn.ax[0], Object.assign(cat(blue), { mode: 'lines', line: { color: hexA(BLUE, 0.5), width: 0.7 }, name: 'no default at T', hoverinfo: 'skip' })),
    on2(Pn.ax[0], Object.assign(cat(red), { mode: 'lines', line: { color: RED, width: 1.1 }, name: 'default at T: V_T < D', hoverinfo: 'skip' })),
    on2(Pn.ax[0], { x: [0, 1], y: [70, 70], mode: 'lines', line: { color: AMBER, width: 2, dash: 'dash' }, name: 'debt D = 70', hoverinfo: 'skip' })];
  const lev = lin(0.3, 0.95, 200);
  [[0.15, GREEN, 'solid'], [0.25, BLUE, 'dash'], [0.40, RED, 'dashdot']].forEach(([sg, c, d]) => tr.push(on2(Pn.ax[1], { x: lev.map(l => 100 * l), y: lev.map(l => 100 * mertonDD(1, l, sg, mu).pd), mode: 'lines', line: { color: c, width: sg === s ? 3.2 : 2.2, dash: d }, name: `asset volatility ${(100 * sg).toFixed(0)}%`, hovertemplate: 'D/V %{x:.0f}%: PD %{y:.2f}%<extra></extra>' })));
  tr.push(on2(Pn.ax[1], { x: [70], y: [100 * mertonDD(100, 70, s, mu).pd], mode: 'markers', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `D/V = 70%, sigma = ${(100 * s).toFixed(0)}%`, hovertemplate: 'PD %{y:.2f}%<extra></extra>' }));
  setAx(Pn, 0, AX('time (years)'), AX('asset value V')); setAx(Pn, 1, AX('leverage D / V (%)'), AX('one-year PD (%)'));
  plot('me-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], `40 GBM paths, mu = ${(100 * mu).toFixed(0)}%, sigma = ${(100 * s).toFixed(0)}%`), ptitle(Pn.ax[1], `PD = N(-DD), mu = ${(100 * mu).toFixed(0)}%`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: MG })));
  const m = mertonDD(100, 70, s, mu);
  out('me-out', [T.me(pc(mu, 0), pc(s, 0), red.length, nf(m.dd, 3), pc(m.pd, 2), mu !== 0.05 || s !== 0.25)]);
}

// ================================================================= x1. IRB capital of a retail loan (worked example)
function iwBlock() {
  const pd = +$('iw-p').value, ead = +kSel('iw-e'), lgd = 0.45, k = irbK(pd, lgd, 'other retail'), nw = single('iw-chart', 330); label('iw-p').textContent = pc(pd, 1);
  const cats = ['PD', 'PD in a bad year (99.9%)', 'capital K = LGD (bad-year PD - PD)', 'expected loss EL = PD x LGD'];
  plot('iw-chart', [{ type: 'bar', x: cats.map(c => wrap(c, nw ? 14 : 22)), y: [pd, k.pb, k.K, pd * lgd].map(v => 100 * v), marker: { color: [BLUE, RED, GREEN, AMBER] }, text: [pd, k.pb, k.K, pd * lgd].map(v => (100 * v).toFixed(2) + '%'), textposition: 'outside', cliponaxis: false, name: `other retail, PD = ${(100 * pd).toFixed(1)}%, LGD = 45%`, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }],
  layout({ xaxis: AX('', { type: 'category' }), yaxis: AX('% of EAD', { range: [0, 100 * k.pb * 1.2] }), legend: LEG(nw, -0.3), showlegend: true, margin: M0 }));
  const K = k.K * ead;
  steps('iw-we', [T.iw1(pc(pd, 1), nf(k.w, 3), nf(k.R, 4)), T.iw2(nf(Pinv(pd), 3), nf(Pinv(0.999), 3), nf(Math.sqrt(k.R), 4), nf(k.arg, 3), pc(k.pb, 2)), T.iw3(pc(k.pb, 2), pc(pd, 1), pc(k.K, 2), fint(ead), fint(K)),
    T.iw4(fint(K), fint(12.5 * K), pc(12.5 * k.K, 0)), T.iw5(pc(pd, 1), fint(ead), fint(pd * lgd * ead), fint(K))]);
  out('iw-out', [T.iw(pc(pd, 1), nf(k.R, 3), pc(k.pb, 2), pc(k.K, 2), pc(12.5 * k.K, 0), pd !== 0.02)]);
}

// ================================================================= x2. WoE and IV of the checking account (table and worked example)
function wtBlock() {
  const v = kSel('wt-v'), rw = kSel('wt-r'), t = woeTable(v, ALL), R = t.rows, nw = single('wt-chart', 330), labs = R.map(r => wrap(codeLab(v, r.bin), nw ? 12 : 18));
  plot('wt-chart', [{ type: 'bar', x: labs, y: R.map(r => r.dg), marker: { color: BLUE }, name: 'share of all goods, g_i / G', hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
    { type: 'bar', x: labs, y: R.map(r => r.db), marker: { color: RED }, name: 'share of all bads, b_i / B', hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }],
  layout({ barmode: 'group', xaxis: AX(VLAB[v], { type: 'category' }), yaxis: AX('share'), legend: LEG(nw, -0.3), margin: M0 }));
  table('wt-tab', T.h.wt, R.map(r => [{ v: T.lab(v, r.bin), c: 'l' }, r.good, r.bad, nf(r.dg, 3), nf(r.db, 3), { v: nf(r.woe, 3), c: r.woe < 0 ? 'bad' : '' }, nf(r.iv, 3)])
    .concat([[{ v: T.h.total, c: 'l' }, sum(R.map(r => r.good)), sum(R.map(r => r.bad)), '1', '1', '', { v: nf(t.iv, 3), c: 'hi' }]]));
  const r = rw === 'first' ? R[0] : rw === 'max' ? R.reduce((a, b) => (b.iv > a.iv ? b : a)) : R[R.length - 1];
  steps('wt-we', [T.wt1(T.lab(v, r.bin), r.good, r.bad, nf(r.dg, 3), nf(r.db, 3)), T.wt2(nf(r.dg, 3), nf(r.db, 3), nf(r.woe, 3), nf(r.iv, 3)), T.wt3(nf(t.iv, 3), T.cls(strength(t.iv)))]);
  out('wt-out', [T.wt(v, nf(t.iv, 3), T.cls(strength(t.iv)), v !== 'status')]);
}

// ================================================================= x3. odds and log-odds (worked example)
function owBlock() {
  const p = +$('ow-p').value, xs = lin(0.01, 0.99, 197), nw = single('ow-chart', 330); label('ow-p').textContent = nf(p, 2);
  plot('ow-chart', [{ x: xs, y: xs.map(q => Math.log(q / (1 - q))), mode: 'lines', line: { color: RED, width: 2.2 }, name: 'log-odds ln(p / (1 - p))', hovertemplate: 'p %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [p, 1 - p], y: [Math.log(p / (1 - p)), Math.log((1 - p) / p)], mode: 'markers', marker: { color: [AMBER, BLUE], size: 12, symbol: ['circle', 'diamond'] }, name: `p = ${p.toFixed(2)} and 1 - p = ${(1 - p).toFixed(2)}`, hovertemplate: 'p %{x:.2f}: %{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('probability of default p', { range: [0, 1] }), yaxis: AX('log-odds', { range: [-5, 5] }), legend: LEG(nw, -0.2), margin: M0 }));
  const o = p / (1 - p);
  steps('ow-we', [T.ow1(nf(p, 2), nf(1 - p, 2), nf(o, 3)), T.ow2(nf(o, 3), nf(Math.log(o), 3)), T.ow3(nf(1 / o, 2), nf(-Math.log(o), 3))]);
  out('ow-out', [T.ow(nf(p, 2), nf(o, 3), nf(Math.log(o), 3), Math.abs(p - 0.2) > 1e-9)]);
}

// ================================================================= x4. a small logit on the full sample (table)
const SMD = {};
function smBlock() {
  const d = kSel('sm-d'), drop = d === 'none' ? [] : d === 'acct' ? ['no_account', 'negative_balance'] : [d], M = SMD[d] || (SMD[d] = smallLogit(drop)), F = small(), nw = single('sm-chart', 340);
  const tr = [[F, 'full model (lecture)', RED, -0.12], [M, d === 'none' ? 'the same model' : 'model without ' + { duration_years: 'duration', age_decades: 'age', amount_1000: 'amount', acct: 'the account indicators' }[d], BLUE, 0.12]].map(([m, nm, c, off]) => {
    const idx = SMALLV.map(v => m.names.indexOf(v)), ys = SMALLV.map((_, j) => SMALLV.length - 1 - j + off);
    return { x: idx.map(i => (i < 0 ? null : m.f.b[i])), y: ys, mode: 'markers', marker: { color: c, size: 9 }, error_x: { type: 'data', array: idx.map(i => (i < 0 ? 0 : 1.96 * m.f.se[i])), color: c, thickness: 1.6, width: 4 }, name: `${nm}: b and 95% interval`, hovertemplate: '%{x:.3f}<extra></extra>' };
  });
  tr.push({ x: [0, 0], y: [-0.5, SMALLV.length - 0.5], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'b = 0', hoverinfo: 'skip' });
  plot('sm-chart', tr, layout({ xaxis: AX('coefficient b (log-odds of default)'), yaxis: AX('', { tickvals: SMALLV.map((_, j) => SMALLV.length - 1 - j), ticktext: SMALLV.map(v => ORL[v]) }), legend: LEG(nw, -0.22), margin: { l: 150, r: 12, t: 10, b: 40 } }));
  table('sm-tab', T.h.sm, M.names.map((n, i) => { const b = M.f.b[i], se = M.f.se[i], z = b / se, p = 2 * (1 - Phi(Math.abs(z)));
    return [{ v: T.sv(n), c: 'l' }, nf(b, 3), nf(se, 3), nf(z, 2), { v: SIG(p), c: p < 0.05 ? 'hi' : '' }, i ? `${nf(Math.exp(b), 2)} (${nf(Math.exp(b - 1.96 * se), 2)}–${nf(Math.exp(b + 1.96 * se), 2)})` : '']; }));
  const ia = M.names.indexOf('amount_1000'), ra = ia >= 0 ? [nf(M.f.b[ia], 3), SIG(2 * (1 - Phi(Math.abs(M.f.b[ia] / M.f.se[ia]))))] : null;
  out('sm-out', [T.sm(d, fint(1000), nf(M.f.ll, 1), nf(M.ll0, 1), nf(M.lr, 1), M.df, ra, d !== 'none')]);
}

// ================================================================= x5. the PD of an applicant (worked example, rounded coefficients of the lecture)
const BR = [-2.101, 0.385, -0.159, 0.029, 1.859, 1.338];
function pwBlock() {
  const dm = +$('pw-d').value, ac = kSel('pw-a'), ag = +kSel('pw-g'), am = 4, nw = single('pw-chart', 340), xs = lin(6, 72, 67); label('pw-d').textContent = String(dm);
  const eta = (d, a) => BR[0] + BR[1] * d / 12 + BR[2] * ag / 10 + BR[3] * am + (a === 'none' ? BR[4] : a === 'neg' ? BR[5] : 0);
  const tr = [['none', RED, 'no checking account'], ['neg', ORANGE, 'balance < 0 DM'], ['ok', BLUE, 'non-negative balance (reference)']].map(([a, c, nm]) =>
    ({ x: xs, y: xs.map(d => logistic(eta(d, a))), mode: 'lines', line: { color: c, width: a === ac ? 3 : 1.6 }, name: `${nm}, age ${ag}, 4000 DM`, hovertemplate: '%{x} months: PD %{y:.3f}<extra></extra>' }));
  tr.push({ x: [dm], y: [logistic(eta(dm, ac))], mode: 'markers', marker: { color: AMBER, size: 13, symbol: 'circle-open', line: { width: 3 } }, name: `the applicant: ${dm} months`, hovertemplate: 'PD %{y:.3f}<extra></extra>' });
  plot('pw-chart', tr, layout({ xaxis: AX('duration (months)'), yaxis: AX('PD in the sample (30% bads)', { range: [0, 1] }), legend: LEG(nw, -0.2), margin: M0 }));
  const e = eta(dm, ac), p = logistic(e), e0 = eta(dm, 'ok'), acc = ac === 'none' ? BR[4] : ac === 'neg' ? BR[5] : 0;
  steps('pw-we', [T.pw1(dm, ag, ac), T.pw2(nf(BR[0], 3), nf(BR[1] * dm / 12, 3), nf(BR[2] * ag / 10, 3), nf(BR[3] * am, 3), nf(acc, 3), nf(e, 3)), T.pw3(nf(e, 3), pc(p), nf(Math.exp(e), 2)),
    T.pw4(nf(e0, 3), pc(logistic(e0))), T.pw5(nf(p * (1 - p) * BR[1], 3), nf(100 * p * (1 - p) * BR[1], 1))]);
  out('pw-out', [T.pw(dm, pc(p), nf(e, 3), dm !== 36 || ac !== 'none' || ag !== 30)]);
}

// ================================================================= x6. the WoE logit scorecard (table)
function wlBlock() {
  const t = +kSel('wl-t'), S = scorecard(t), nw = single('wl-chart', 340), names = ['const', ...S.sel], ys = names.map((_, i) => names.length - 1 - i);
  plot('wl-chart', [{ x: S.lg.b, y: ys, mode: 'markers', marker: { color: RED, size: 9 }, error_x: { type: 'data', array: S.lg.se.map(v => 1.96 * v), color: BLUE, thickness: 1.6, width: 4 }, name: `WoE logit, IV >= ${t}: b and 95% interval`, hovertemplate: '%{x:.3f}<extra></extra>' },
    { x: [-1, -1], y: [-0.5, names.length - 0.5], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'b = -1: the variable adds its own WoE', hoverinfo: 'skip' }],
  layout({ xaxis: AX('coefficient b'), yaxis: AX('', { tickvals: ys, ticktext: names.map(n => (n === 'const' ? 'constant' : n.replace(/_/g, ' '))) }), legend: LEG(nw, -0.22), margin: { l: 150, r: 12, t: 10, b: 40 } }));
  table('wl-tab', T.h.wl, names.map((n, i) => [{ v: n === 'const' ? T.h.const : T.vn(n), c: 'l' }, n === 'const' ? '' : nf(TAB_TR[n].iv, 3), nf(S.lg.b[i], 3), nf(S.lg.se[i], 3)]));
  out('wl-out', [T.wl(nf(t, 2), S.sel.length, nf(S.auc_te, 3), nf(S.auc_tr, 3), nf(S.lg.ll, 1), t !== D.iv_min)]);
}

// ================================================================= x7. correcting for oversampling (worked example)
function prBlock() {
  const pi = +$('pr-pi').value, w = kSel('pr-p'), ps = 0.3, sh = Math.log(pi / (1 - pi)) - Math.log(ps / (1 - ps)), xs = lin(0.005, 0.995, 199), nw = single('pr-chart', 330); label('pr-pi').textContent = pc(pi, 0);
  const e0 = w === 'app' ? 0.552 : Math.log(+w / (1 - +w)), p0 = logistic(e0);
  plot('pr-chart', [{ x: xs, y: xs.map(p => logistic(Math.log(p / (1 - p)) + sh)), mode: 'lines', line: { color: RED, width: 2.2 }, name: `population PD, pi = ${(100 * pi).toFixed(0)}% (sample: 30%)`, hovertemplate: 'sample %{x:.3f}: population %{y:.3f}<extra></extra>' },
    { x: [0, 1], y: [0, 1], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'no correction', hoverinfo: 'skip' },
    { x: [p0], y: [logistic(e0 + sh)], mode: 'markers', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: w === 'app' ? 'the applicant of the worked example' : `sample PD ${(100 * +w).toFixed(0)}%`, hovertemplate: '%{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('PD in the sample', { range: [0, 1] }), yaxis: AX('PD in the population', { range: [0, 1] }), legend: LEG(nw, -0.2), margin: M0 }));
  steps('pr-we', [T.pr1(pc(pi, 0), nf(Math.log(pi / (1 - pi)), 3), nf(Math.log(ps / (1 - ps)), 3), nf(sh, 3)), T.pr2(pc(logistic(Math.log(0.3 / 0.7) + sh)), pc(logistic(sh))),
    T.pr3(w, nf(e0, 3), nf(sh, 3), nf(e0 + sh, 3), pc(logistic(e0 + sh)), pc(p0)), T.pr4(nf(FACTOR, 2), nf(-sh, 3), nf(-FACTOR * sh, 1))]);
  out('pr-out', [T.pr(pc(pi, 0), nf(sh, 3), pc(logistic(0.552 + sh)), pi !== 0.05)]);
}

// ================================================================= x8. Fisher's direction with two variables (worked example)
const PAIRS = { 'duration,age': [['duration', 1], ['age', 1]], 'duration,amount': [['duration', 1], ['amount', 1000]], 'age,amount': [['age', 1], ['amount', 1000]] };
const FW = {};
function fwBlock() {
  const pk = kSel('fw-v'), P = PAIRS[pk], X = ALL.map(i => P.map(([v, s]) => SG[v][i] / s)), o = FW[pk] || (FW[pk] = { ld: ldaFit(X, Y), lg: logitFit(X, Y) }), { ld, lg } = o, nw = single('fw-chart', 340);
  const eta = ld.eta(X), e0 = eta.filter((_, i) => !Y[i]), e1 = eta.filter((_, i) => Y[i]), lo = Math.floor(Math.min(...eta) * 10) / 10, hi = Math.ceil(Math.max(...eta) * 10) / 10, e = [];
  for (let x = lo; x <= hi + 0.1 + 1e-9; x += 0.1) e.push(+x.toFixed(6));
  const mid = e.slice(0, -1).map(x => x + 0.05), nm = P.map(([v, s]) => (s === 1000 ? 'amount (1000 DM)' : v)).join(' and ');
  plot('fw-chart', [{ type: 'bar', x: mid, y: histo(e0, e).map(c => c / e0.length), width: 0.1, marker: { color: hexA(BLUE, 0.55) }, name: 'good credits', hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    { type: 'bar', x: mid, y: histo(e1, e).map(c => c / e1.length), width: 0.1, marker: { color: hexA(RED, 0.55) }, name: 'bad credits', hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    vline(Math.log(0.3 / 0.7), 0, 0.2, 'PD = 30%', GREEN, 'dash')],
  layout({ barmode: 'overlay', xaxis: AX(`LDA log-odds c + w'x, x = ${nm}`), yaxis: AX('share of the group'), legend: LEG(nw, -0.2), margin: M0 }));
  const f2 = a => `(${nf(a[0], 2)}; ${nf(a[1], 2)})`.replace(/; /, LANG === 'ro' ? '; ' : ', ');
  const det = ld.S[0][0] * ld.S[1][1] - ld.S[0][1] ** 2;
  steps('fw-we', [T.fw1(pk, fint(ld.n0), fint(ld.n1), f2(ld.m0), f2(ld.m1), f2(ld.dm)), T.fw2(nf(ld.S[0][0], 1), nf(ld.S[0][1], 2), nf(ld.S[1][1], 1), fint(det)),
    T.fw3(`(${nf(ld.w[0], 4)}${LANG === 'ro' ? '; ' : ', '}${nf(ld.w[1], 4)})`, `(${nf(lg.b[1], 4)}${LANG === 'ro' ? '; ' : ', '}${nf(lg.b[2], 4)})`), T.fw4(pk, nf(ld.w[0] / ld.w[1], 2), nf(lg.b[1] / lg.b[2], 2))]);
  out('fw-out', [T.fw(pk, nf(ld.w[0], 4), nf(ld.w[1], 4), nf(auc(eta, Y), 3), nf(auc(lg.p, Y), 3), pk !== 'duration,age')]);
}

// ================================================================= x9. logit or LDA? (table)
function ltBlock() {
  const S = S0(), te = kSel('lt-s') === 'test', p = te ? S.p_te : S.p_tr, q = te ? S.q_te : S.q_tr, y = te ? S.yte : S.ytr, nw = single('lt-chart', 360);
  const a = p.map(v => Math.log(v / (1 - v))), b = q.map(v => Math.log(v / (1 - v))), ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  const r = sab / Math.sqrt(saa * sbb), dl = delong(p, y, q);
  plot('lt-chart', [0, 1].map(c => ({ x: a.filter((_, i) => y[i] === c), y: b.filter((_, i) => y[i] === c), mode: 'markers', marker: { color: hexA(c ? RED : BLUE, 0.55), size: 5, symbol: c ? 'x' : 'circle' }, name: c ? 'bad credits' : 'good credits', hovertemplate: 'logit %{x:.2f}, LDA %{y:.2f}<extra></extra>' }))
    .concat([{ x: [-4, 3], y: [-4, 3], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'equal log-odds', hoverinfo: 'skip' }]),
  layout({ xaxis: AX(`WoE logit: log-odds of default (${te ? 'test' : 'training'} sample)`), yaxis: AX('Fisher LDA: log-odds of default'), legend: LEG(nw, -0.2), margin: M0 }));
  table('lt-tab', T.h.lt, [[{ v: T.h.auc, c: 'l' }, nf(dl.auc1, 3), nf(dl.auc2, 3)], [{ v: T.h.brier, c: 'l' }, nf(brier(p, y), 3), nf(brier(q, y), 3)], [{ v: T.h.corr, c: 'l' }, { v: nf(r, 4), span: 2 }], [{ v: T.h.delong, c: 'l' }, { v: `z = ${nf(dl.z, 2)}, p = ${nf(dl.p, 2)}`, span: 2 }]]);
  out('lt-out', [T.lt(te, fint(y.length), nf(dl.auc1, 3), nf(dl.auc2, 3), nf(r, 3), nf(dl.z, 2), nf(dl.p, 2), !te)]);
}

// ================================================================= x10. scores and points (worked example)
function swBlock() {
  const pdo = +kSel('sw-o'), pp = +kSel('sw-p'), sc = scale(pdo), S = S0(), k = S.sel.length, nw = single('sw-chart', 330), ps = lin(Math.log(0.005), Math.log(0.6), 200).map(Math.exp);
  const ref = { f: FACTOR, o: OFFSET };
  plot('sw-chart', [{ x: ps, y: ps.map(p => scoreOf(p, sc)), mode: 'lines', line: { color: RED, width: 2.2 }, name: `PDO = ${pdo}: factor ${sc.f.toFixed(2)}, offset ${sc.o.toFixed(2)}`, hovertemplate: 'PD %{x:.3f}: %{y:.1f} points<extra></extra>' },
    { x: ps, y: ps.map(p => scoreOf(p, ref)), mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'PDO = 20 (lecture)', hoverinfo: 'skip' },
    { x: [pp], y: [scoreOf(pp, sc)], mode: 'markers', marker: { color: AMBER, size: 12, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `PD = ${(100 * pp).toFixed(0)}%`, hovertemplate: '%{y:.1f} points<extra></extra>' }],
  layout({ xaxis: AX('PD (log scale)', { type: 'log', tickvals: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext: ['0.5%', '1%', '2%', '5%', '10%', '20%', '50%'] }), yaxis: AX('score (points)'), legend: LEG(nw, -0.2), margin: M0 }));
  const lo = (560 - sc.o) / sc.f, b = S.lg.b, j = S.sel.indexOf('status') + 1, pts = TAB_TR.status.rows.map(r => -(b[j] * r.woe + b[0] / k) * sc.f + sc.o / k);
  steps('sw-we', [T.sw1(pdo, nf(sc.f, 2), nf(sc.o, 2)), T.sw2(pc(pp, 0), nf((1 - pp) / pp, 2), nf(Math.log((1 - pp) / pp), 3), nf(scoreOf(pp, sc), 1)), T.sw3(nf(lo, 3), nf(Math.exp(lo), 1), pc(1 / (1 + Math.exp(lo)))),
    T.sw4(k, pts.map(v => fint(v))), T.sw5(nf(sc.f, 2), nf(-NUM.prior_shift, 3), nf(-sc.f * NUM.prior_shift, 1))]);
  out('sw-out', [T.sw(pdo, nf(scoreOf(0.05, sc), 1), nf(scoreOf(0.3, sc), 1), pc(1 / (1 + Math.exp(lo))), pdo !== 20)]);
}

// ================================================================= x11. two cut-offs on the test sample (table)
function cmBlock() {
  const S = S0(), c = +$('cm-c').value, p = S.p_te, y = S.yte, cuts = lin(0.02, 0.9, 89), cm = cuts.map(x => confusion(p, y, x)), nw = single('cm-chart', 360), k = confusion(p, y, c); label('cm-c').textContent = nf(c, 2);
  plot('cm-chart', [['tpr', RED, 'TPR: bads rejected'], ['fpr', BLUE, 'FPR: goods rejected'], ['acc', GREEN, 'accuracy']].map(([f, col, nm]) => ({ x: cuts, y: cm.map(m => m[f]), mode: 'lines', line: { color: col, width: 2 }, name: nm, hovertemplate: 'cut-off %{x:.2f}: %{y:.3f}<extra></extra>' }))
    .concat([vline(c, 0, 1, `cut-off ${c.toFixed(2)}`, AMBER, 'dot'), { x: [c, c, c], y: [k.tpr, k.fpr, k.acc], mode: 'markers', marker: { color: AMBER, size: 9 }, showlegend: false, hoverinfo: 'skip' }]),
  layout({ xaxis: AX('PD cut-off (reject if PD > cut-off)', { range: [0, 0.92] }), yaxis: AX('rate', { range: [0, 1] }), legend: LEG(nw, -0.2), margin: M0 }));
  const row = (lab, m, cl) => [{ v: lab, c: 'l' + (cl ? ' hi' : '') }, m.TP, m.FN, m.FP, m.TN, pc(m.acc), pc(m.tpr), pc(m.fpr), nf(m.cost, 3)];
  table('cm-tbl', T.h.cm, [row('PD > ' + nf(0.5, 1), confusion(p, y, 0.5)), row('PD > 1/6', confusion(p, y, 1 / 6)), row('PD > ' + nf(c, 2), k, true)]);
  steps('cm-we', [T.cm1(nf(c, 2), k.TP, k.FN, k.FP, k.TN), T.cm2(k.TP, k.FN, pc(k.tpr), k.FP, k.TN, pc(k.fpr)), T.cm3(k.TP, k.FP, pc(k.precision), k.TN, pc(k.acc)), T.cm4(k.FN, k.FP, nf(k.cost, 3))]);
  out('cm-out', [T.cm(nf(c, 2), pc(k.acc), pc(k.tpr), pc(k.fpr), nf(k.cost, 3), c !== 0.5)]);
}

// ================================================================= x12. ROC curve and AUC: AUC as a rank probability (worked example)
const MODS = { logit: ['WoE logit', 'p_te'], lda: ['Fisher LDA', 'q_te'], single: ['checking account only', 'single_te'] };
function awBlock() {
  const m = kSel('aw-m'), S = S0(), s = S[MODS[m][1]], y = S.yte, P = pairs(s, y), N2 = P.n1 * P.n0, A = (P.gt + 0.5 * P.eq) / N2, dl = delong(s, y), nw = single('aw-chart', 300);
  plot('aw-chart', [['gt', RED, 'bad ranked riskier than good'], ['eq', AMBER, 'tie (counts 1/2)'], ['lt', BLUE, 'good ranked riskier than bad']].map(([k, c, nm]) =>
    ({ type: 'bar', orientation: 'h', y: [MODS[m][0]], x: [P[k] / N2], marker: { color: c }, name: nm, text: [(100 * P[k] / N2).toFixed(1) + '%'], textposition: 'inside', insidetextfont: { color: '#fff' }, hovertemplate: `${nm}: %{x:.4f}<extra></extra>` })),
  layout({ barmode: 'stack', xaxis: AX('share of the (bad, good) pairs of the test sample', { range: [0, 1] }), yaxis: AX(''), legend: LEG(nw, -0.4), margin: { l: 130, r: 12, t: 10, b: 40 } }));
  steps('aw-we', [T.aw1(P.n1, P.n0, fint(N2)), T.aw2(MODS[m][0], fint(P.gt), fint(P.eq), fint(P.lt)), T.aw3(fint(P.gt), fint(P.eq), fint(N2), nf(A, 4)), T.aw4(nf(A, 3), nf(2 * A - 1, 3)), T.aw5(nf(dl.se, 3), nf(dl.lo, 3), nf(dl.hi, 3))]);
  out('aw-out', [T.aw(MODS[m][0], nf(A, 3), pc(P.eq / N2), nf(2 * A - 1, 3), m !== 'logit')]);
}

// ================================================================= x13. CAP, accuracy ratio and KS (worked example)
function kwBlock() {
  const m = kSel('kw-m'), S = S0(), s = S[MODS[m][1]], y = S.yte, r = rocPts(s, y), K = ksStat(s, y), Cp = capPts(s, y), A = auc(s, y), nw = single('kw-chart', 340);
  const d = r.tpr.map((t, i) => t - r.fpr[i]); let i = 0; d.forEach((v, k) => { if (v > d[i]) i = k; });
  plot('kw-chart', [{ x: r.fpr, y: d, mode: 'lines+markers', line: { color: COL[m === 'logit' ? 'logit' : m === 'lda' ? 'lda' : 'single'], width: 2 }, marker: { size: 3 }, name: `${MODS[m][0]}: TPR - FPR along the ROC curve`, hovertemplate: 'FPR %{x:.3f}: %{y:.3f}<extra></extra>' },
    { x: [r.fpr[i]], y: [d[i]], mode: 'markers', marker: { color: GREEN, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `KS = max (TPR - FPR) = ${d[i].toFixed(3)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('FPR: goods rejected', { range: [0, 1] }), yaxis: AX('TPR - FPR', { range: [0, 0.65] }), legend: LEG(nw, -0.2), margin: M0 }));
  steps('kw-we', [T.kw1(MODS[m][0], nf(K.ks, 3), pc(r.tpr[i]), pc(r.fpr[i])), T.kw2(nf(A, 3), nf(2 * A - 1, 3), nf(Cp.ar, 3)), T.kw3(m === 'single')]);
  out('kw-out', [T.kw(MODS[m][0], nf(K.ks, 3), nf(2 * A - 1, 3), nf(Cp.ar, 3), m !== 'logit')]);
}

// ================================================================= x14. calibration and the Brier score (worked example and table)
function bwBlock() {
  const k = +kSel('bw-g'), S = S0(), p = S.p_te, y = S.yte, c = calTable(p, y, D.order_sgc, k), nw = single('bw-chart', 360), b = brier(p, y), br = brier(p.map(() => mean(S.ytr)), y);
  plot('bw-chart', [{ x: [0, 1], y: [0, 1], mode: 'lines', line: { color: COL.random, width: 1.5, dash: 'dot' }, name: 'perfect calibration', hoverinfo: 'skip' },
    { x: c.G.map(g => g.pd), y: c.G.map(g => g.rate), mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 8 }, error_y: { type: 'data', array: c.G.map(g => 1.96 * Math.sqrt(g.rate * (1 - g.rate) / g.n)), color: BLUE, thickness: 1.5, width: 4 }, name: `${k} groups of ${300 / k} test credits (95% interval)`, hovertemplate: 'mean PD %{x:.3f}: observed %{y:.3f}<extra></extra>' }],
  layout({ xaxis: AX('mean predicted PD in the group (WoE logit)', { range: [0, 1] }), yaxis: AX('observed bad rate in the group', { range: [0, 1] }), legend: LEG(nw, -0.18), margin: M0 }));
  table('bw-tab', T.h.cal, c.G.map((g, j) => [j + 1, g.n, pc(g.pd), pc(g.rate), g.bad, nf(g.exp, 1), nf((g.bad - g.exp) ** 2 / (g.exp * (1 - g.pd)), 2)]));
  steps('bw-we', [T.bw1(nf(b, 4), pc(mean(S.ytr), 0), nf(br, 4)), T.bw2(nf(b, 3), nf(br, 3), pc(1 - b / br, 0)), T.bw3(k, nf(c.hl, 3), c.df, nf(c.p, 3))]);
  out('bw-out', [T.bw(k, nf(b, 3), nf(c.hl, 3), c.df, nf(c.p, 3), k !== 10)]);
}

// ================================================================= x15. a reject-inference experiment on the Taiwan data (table)
function rjBlock() {
  const a = kSel('rj-a'), R = D.reject[a], Pn = panels('rj-chart', 2, { h: 340 });
  const tr = [on2(Pn.ax[0], { type: 'bar', x: ['trained on accepted', 'trained on all', 'old score'], y: [R.auc_acc_model_on_all, R.auc_all_model_on_all, R.auc_old_on_all], marker: { color: [RED, BLUE, AMBER] }, text: [R.auc_acc_model_on_all, R.auc_all_model_on_all, R.auc_old_on_all].map(v => v.toFixed(3)), textposition: 'outside', cliponaxis: false, name: 'AUC on all test applicants', hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
    on2(Pn.ax[1], { type: 'bar', x: ['observed rate', 'mean PD, accepted-only model', 'mean PD, full model'], y: [R.rate_rejected, R.mean_pd_rej_acc_model, R.mean_pd_rej_all_model].map(v => 100 * v), marker: { color: [DARK, RED, BLUE] }, text: [R.rate_rejected, R.mean_pd_rej_acc_model, R.mean_pd_rej_all_model].map(v => (100 * v).toFixed(1) + '%'), textposition: 'outside', cliponaxis: false, name: 'rejected test applicants (%)', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })];
  setAx(Pn, 0, AX('', { type: 'category' }), AX('AUC', { range: [0.5, 0.85] })); setAx(Pn, 1, AX('', { type: 'category' }), AX('%', { range: [0, 70] }));
  plot('rj-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], `old score accepts the best ${(100 * a).toFixed(0)}%`), ptitle(Pn.ax[1], 'rejected test applicants')], legend: LEG(Pn.stack, Pn.stack ? -0.12 : -0.3), margin: MG })));
  table('rj-tab', T.h.rj, [[{ v: T.h.rjAuc, c: 'l' }, nf(R.auc_acc_model_on_all, 3), nf(R.auc_all_model_on_all, 3), nf(R.auc_old_on_all, 3)],
    [{ v: T.h.rjAuc2, c: 'l' }, nf(R.auc_acc_model_on_acc, 3), nf(R.auc_all_model_on_acc, 3), ''], [{ v: T.h.rjRej, c: 'l' }, pc(R.mean_pd_rej_acc_model), pc(R.mean_pd_rej_all_model), ''],
    [{ v: T.h.rjObs, c: 'l' }, { v: pc(R.rate_rejected), span: 3 }]]);
  out('rj-out', [T.rj(pc(+a, 0), fint(R.n_acc_train), pc(R.bad_acc), pc(R.bad_rej), R.dropped.length, nf(R.auc_acc_model_on_all, 3), nf(R.auc_all_model_on_all, 3), nf(R.auc_old_on_all, 3), pc(R.mean_pd_rej_acc_model), pc(R.rate_rejected), nf(100 * (1 - R.mean_pd_rej_acc_model / R.rate_rejected), 0), a !== '0.7')]);
}

// ================================================================= x16. a Z-score (worked example)
const ZW = [1.2, 1.4, 3.3, 0.6, 1.0];
function zwBlock() {
  const x3 = +$('zw-x').value, x4 = +kSel('zw-4'), X = [0.15, 0.20, x3, x4, 1.10], c = X.map((v, i) => ZW[i] * v), Z = sum(c), nw = single('zw-chart', 320); label('zw-x').textContent = nf(x3, 2);
  const zone = Z < 1.81 ? 'distress' : Z <= 2.99 ? 'grey' : 'safe', labs = ['1.2 X1: working capital / TA', '1.4 X2: retained earnings / TA', '3.3 X3: EBIT / TA', '0.6 X4: equity / debt', '1.0 X5: sales / TA'];
  plot('zw-chart', labs.map((l, i) => ({ type: 'bar', orientation: 'h', y: [0], width: 0.5, x: [c[i]], marker: { color: [BLUE, TEAL, RED, PURPLE, AMBER][i] }, name: l, hovertemplate: `${l}: %{x:.3f}<extra></extra>` }))
    .concat([{ x: [1.81, 1.81], y: [-0.4, 0.4], mode: 'lines', line: { color: DARK, width: 2, dash: 'dash' }, name: 'distress | grey zone: 1.81', hoverinfo: 'skip' }, { x: [2.99, 2.99], y: [-0.4, 0.4], mode: 'lines', line: { color: GREEN, width: 2, dash: 'dash' }, name: 'grey | safe zone: 2.99', hoverinfo: 'skip' }]),
  layout({ barmode: 'stack', xaxis: AX(`Z = ${Z.toFixed(3)} (${zone} zone)`, { range: [0, 4.2] }), yaxis: AX('', { tickvals: [0], ticktext: ['Z'], range: [-0.5, 0.5] }), legend: LEG(nw, -0.45), margin: { l: 30, r: 12, t: 10, b: 40 } }));
  steps('zw-we', [T.zw1(X.map(v => nf(v, 2))), T.zw2(c.map(v => nf(v, 3)), nf(Z, 3), zone), T.zw3(nf((2.99 - (Z - c[2])) / 3.3, 4), nf(Z - c[2], 3))]);
  out('zw-out', [T.zw(nf(x3, 2), nf(Z, 3), zone, x3 !== 0.08 || x4 !== 0.9)]);
}

// ================================================================= x17. distance to default (worked example)
function ddBlock() {
  const s = +$('dd-s').value, E = +kSel('dd-e'), ss = lin(0.1, 0.6, 101), nw = single('dd-chart', 330), m = mertonDD(100, 70, s, 0.05); label('dd-s').textContent = pc(s, 0);
  plot('dd-chart', [{ x: ss.map(v => 100 * v), y: ss.map(v => 100 * mertonDD(100, 70, v, 0.05).pd), mode: 'lines', line: { color: RED, width: 2.2 }, name: 'PD = N(-DD), V0 = 100, D = 70, mu = 5%, T = 1', hovertemplate: 'sigma %{x:.0f}%: PD %{y:.2f}%<extra></extra>' },
    { x: [100 * s], y: [100 * m.pd], mode: 'markers', marker: { color: AMBER, size: 13, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `sigma_V = ${(100 * s).toFixed(0)}%: DD = ${m.dd.toFixed(2)}`, hovertemplate: 'PD %{y:.2f}%<extra></extra>' }],
  layout({ xaxis: AX('asset volatility sigma_V (%)'), yaxis: AX('one-year PD (%)'), legend: LEG(nw, -0.2), margin: M0 }));
  const q = mertonEq(E, 0.5, 70, 0.03), dr = 0.05 - 0.5 * s * s;
  steps('dd-st', [T.dd1(nf(Math.log(100 / 70), 4), pc(s, 0), nf(dr, 5)), T.dd2(nf(Math.log(100 / 70) + dr, 4), nf(s, 2), nf(m.dd, 3), pc(m.pd, 2)), T.dd3(E, nf(q.V, 1), pc(q.s, 1), nf(q.dd, 2), pc(q.pd, 2))]);
  out('dd-out', [T.dd(pc(s, 0), nf(m.dd, 3), pc(m.pd, 2), s !== 0.25)]);
}

// ================================================================= wiring
const PFX = { cpd: 'cp', irb: 'ir', status: 'ds', 'woe-dur': 'wd', iv: 'iv', 'logit-curve': 'lc', odds: 'or', lda: 'ld', score: 'sd', cost: 'co', roc: 'rc', cap: 'ca', ks: 'ks', calib: 'cb', cv: 'cv', fair: 'fa', merton: 'me',
  'irb-we': 'iw', 'woe-tab': 'wt', 'odds-we': 'ow', 'small-tab': 'sm', 'pd-we': 'pw', 'woe-logit-tab': 'wl', 'prior-we': 'pr', 'fisher-we': 'fw', 'lda-tab': 'lt', 'scale-we': 'sw', 'cm-tab': 'cm',
  'auc-we': 'aw', 'ks-we': 'kw', 'brier-we': 'bw', 'reject-tab': 'rj', 'z-we': 'zw', 'dd-we': 'dd' };
const BLOCKS = { cpBlock, irBlock, dsBlock, wdBlock, ivBlock, lcBlock, orBlock, ldBlock, sdBlock, coBlock, rcBlock, caBlock, ksBlock, cbBlock, cvBlock, faBlock, meBlock,
  iwBlock, wtBlock, owBlock, smBlock, pwBlock, wlBlock, prBlock, fwBlock, ltBlock, swBlock, cmBlock, awBlock, kwBlock, bwBlock, rjBlock, zwBlock, ddBlock };
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
  on(['cp-r'], cpBlock); PLy('cp-play', 'cp-r', [0.2, 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.2], 800);
  on(['ir-q', 'ir-l'], irBlock); PLy('ir-play', 'ir-q', ['0.999', '0.99', '0.995', '0.9999', '0.999'], 1300);
  on(['ds-v'], dsBlock); PLy('ds-play', 'ds-v', ['credit_history', 'savings', 'employment_duration', 'status'], 1300);
  on(['wd-v', 'wd-s'], wdBlock); PLy('wd-play', 'wd-v', ['amount', 'age', 'duration'], 1400);
  on(['iv-s'], ivBlock); PLy('iv-play', 'iv-s', ['train', 'test', 'full'], 1500);
  on(['lc-v', 'lc-m'], lcBlock); PLy('lc-play', 'lc-v', ['age', 'duration'], 1600);
  on(['or-l'], orBlock); PLy('or-play', 'or-l', ['0.95', '0.9', '0.99', '0.95'], 1400);
  on(['ld-p'], ldBlock); PLy('ld-play', 'ld-p', [0.3, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.3], 800);
  on(['sd-c', 'sd-b'], sdBlock); PLy('sd-play', 'sd-c', ['0.1666666667', '0.1', '0.3', '0.5', '0.1666666667'], 1300);
  on(['co-f'], coBlock); PLy('co-play', 'co-f', ['5', '2', '3', '10', '5'], 1400);
  on(['rc-c', 'rc-s'], rcBlock); PLy('rc-play', 'rc-c', [0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5], 800);
  on(['ca-x'], caBlock); PLy('ca-play', 'ca-x', [0.3, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.3], 800);
  on(['ks-s'], ksBlock); PLy('ks-play', 'ks-s', [516, 440, 460, 480, 500, 520, 534, 550, 570, 590, 516], 800);
  on(['cb-g'], cbBlock); PLy('cb-play', 'cb-g', ['10', '5', '20', '10'], 1400);
  on(['cv-r'], cvBlock); PLy('cv-play', 'cv-r', ['20', '1', '5', '10', '20'], 1300);
  on(['fa-a'], faBlock); PLy('fa-play', 'fa-a', [0.75, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.75], 900);
  on(['me-m', 'me-s'], meBlock); PLy('me-play', 'me-m', [0.05, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.05], 900);
  on(['iw-p', 'iw-e'], iwBlock); PLy('iw-play', 'iw-p', [0.02, 0.005, 0.01, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.02], 800);
  on(['wt-v', 'wt-r'], wtBlock); PLy('wt-play', 'wt-v', ['credit_history', 'savings', 'employment_duration', 'status'], 1400);
  on(['ow-p'], owBlock); PLy('ow-play', 'ow-p', [0.2, 0.02, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.98, 0.2], 800);
  on(['sm-d'], smBlock); PLy('sm-play', 'sm-d', ['duration_years', 'age_decades', 'amount_1000', 'acct', 'none'], 1500);
  on(['pw-d', 'pw-a', 'pw-g'], pwBlock); PLy('pw-play', 'pw-d', [36, 6, 12, 18, 24, 30, 42, 48, 60, 72, 36], 800);
  on(['wl-t'], wlBlock); PLy('wl-play', 'wl-t', ['0.1', '0.02', '0.05', '0.2', '0.3', '0.1'], 1500);
  on(['pr-pi', 'pr-p'], prBlock); PLy('pr-play', 'pr-pi', [0.05, 0.01, 0.02, 0.1, 0.15, 0.2, 0.3, 0.05], 900);
  on(['fw-v'], fwBlock); PLy('fw-play', 'fw-v', ['duration,amount', 'age,amount', 'duration,age'], 1500);
  on(['lt-s'], ltBlock); PLy('lt-play', 'lt-s', ['train', 'test'], 1600);
  on(['sw-o', 'sw-p'], swBlock); PLy('sw-play', 'sw-o', ['20', '10', '30', '40', '20'], 1300);
  on(['cm-c'], cmBlock); PLy('cm-play', 'cm-c', [0.5, 0.1, 0.17, 0.27, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5], 800);
  on(['aw-m'], awBlock); PLy('aw-play', 'aw-m', ['lda', 'single', 'logit'], 1500);
  on(['kw-m'], kwBlock); PLy('kw-play', 'kw-m', ['lda', 'single', 'logit'], 1500);
  on(['bw-g'], bwBlock); PLy('bw-play', 'bw-g', ['10', '5', '20', '10'], 1400);
  on(['rj-a'], rjBlock); PLy('rj-play', 'rj-a', ['0.7', '0.5', '0.6', '0.8', '0.9', '0.7'], 1400);
  on(['zw-x', 'zw-4'], zwBlock); PLy('zw-play', 'zw-x', [0.08, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.35, 0.4, 0.08], 800);
  on(['dd-s', 'dd-e'], ddBlock); PLy('dd-play', 'dd-s', [0.25, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.25], 800);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH12 = { Phi, Pinv, chi2sf, logitFit, ldaFit, auc, pairs, delong, rocPts, capPts, ksStat, brier, calTable, confusion, scoreOf, scale, irbK, condPD, mertonDD, mertonEq,
  woeTable, tables, ivSorted, scorecard, smallLogit, small, fairGroups, ld2, lcFit, quantile, TR, TE, ALL, init, PFX, blocks: BLOCKS };
if (window.CH12_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
