// Chapter 15 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch15_data.js.
(function () {
'use strict';
const D = window.CH15_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH15_TEXT;
const NUM = D.N, COL = D.colors, RCOL = D.reg_col, BK = D.banks;
const DARK = C.navy, GREEN = COL.Forest, RED = COL.IDAred, BLUE = COL.MainBlue, ORANGE = COL.Orange, AMBER = COL.Amber, PURPLE = COL.Purple, TEAL = COL.Teal;
const REGS = ['US', 'EU', 'RO'], US = D.regions.US, EU = D.regions.EU, RO = D.regions.RO, ALL = [...US, ...EU, ...RO], INTL = [...US, ...EU];

// ================================================================= numerics
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = LG[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(a, b, x) {           // continued fraction of the incomplete beta function (modified Lentz)
  const FP = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-16) break;
  }
  return h;
}
function ibeta(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const t2sf = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));                       // two-sided p-value of a t statistic
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
const vari = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return s / (a.length - 1); };
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; saa += u * u; sbb += v * v; } return sab / Math.sqrt(saa * sbb); };
const cov = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb); return s / (a.length - 1); };
const quantile = (a, q) => { const b = Float64Array.from(a).sort(), h = (b.length - 1) * q, f = Math.floor(h); return f + 1 < b.length ? b[f] + (h - f) * (b[f + 1] - b[f]) : b[f]; };   // numpy, linear
function ranks(a) {                  // scipy rankdata, average ties
  const idx = a.map((_, i) => i).sort((i, j) => a[i] - a[j]), r = new Array(a.length);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++; const v = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k]] = v; i = j + 1; }
  return r;
}
function spearman(a, b) { const r = corr(ranks(a), ranks(b)), n = a.length, t = r * Math.sqrt((n - 2) / ((r + 1) * (1 - r))); return { r, p: t2sf(t, n - 2) }; }
function solve(A, b) {               // Gaussian elimination with partial pivoting; b may be a matrix (columns)
  const n = A.length, M = A.map((r, i) => [...r, ...(Array.isArray(b[i]) ? b[i] : [b[i]])]), m = M[0].length;
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) { if (r === c) continue; const f = M[r][c] / M[c][c]; if (f) for (let k = c; k < m; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r.slice(n).map(v => v / M[i][i]));
}
const matmul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr_ = A => A[0].map((_, j) => A.map(r => r[j]));

// ================================================================= data: closes on their own calendars, joined as prices() does
const DAY = 864e5;
const isoOf = x => new Date(x).toISOString().slice(0, 10);
const msOf = s => Date.parse(s + 'T00:00:00Z');
const addDays = (s, k) => isoOf(msOf(s) + k * DAY);
function dated(o) { const b = msOf(o.t0); return o.t.map(d => isoOf(b + d * DAY)); }
const PX = {};
Object.entries(D.px).forEach(([k, s]) => { PX[k] = { d: dated(s), p: s.v }; });
const JC = {};
function joinR(keys, start, end) {   // prices(keys) joined on common days, then 100 diff(log), known data errors removed (returns())
  const key = keys.join(',') + '|' + (start || '') + '|' + (end || ''); if (JC[key]) return JC[key];
  const maps = keys.map(k => new Map(PX[k].d.map((d, i) => [d, PX[k].p[i]])));
  const base = PX[keys[0]].d.filter(d => (!start || d >= start) && (!end || d <= end) && maps.every(m => m.has(d)));
  const P = keys.map((_, j) => base.map(d => maps[j].get(d)));
  let d = base.slice(1), R = P.map(p => p.slice(1).map((v, i) => 100 * (Math.log(v) - Math.log(p[i]))));
  const bad = new Set(); keys.forEach(k => (D.bad_days[k] || []).forEach(x => bad.add(x)));
  if (bad.size) { const keep = d.map(x => !bad.has(x)); d = d.filter((_, i) => keep[i]); R = R.map(r => r.filter((_, i) => keep[i])); }
  const o = { d, pd: base, P: {} }; keys.forEach((k, j) => { o[k] = R[j]; o.P[k] = P[j]; });
  return (JC[key] = o);
}
const regionR = reg => joinR([...D.regions[reg], D.index[reg]], D.region_start[reg] > '2005-01-01' ? D.region_start[reg] : '2005-01-01');
const intlR = () => joinR(INTL, '2005-01-01');
function portfolio(R, members, i0, i1) {   // equal-weighted bank portfolio, daily rebalancing: log return of the average simple return
  i0 = i0 || 0; i1 = i1 === undefined ? R.d.length : i1; const o = new Array(i1 - i0);
  for (let t = i0; t < i1; t++) { let s = 0; for (const k of members) s += Math.exp(R[k][t] / 100) - 1; o[t - i0] = 100 * Math.log(1 + s / members.length); }
  return o;
}
const PF = {};
const pfOf = reg => PF[reg] || (PF[reg] = portfolio(regionR(reg), D.regions[reg]));
const between = (d, a, b) => { let i0 = d.findIndex(x => x >= a); if (i0 < 0) i0 = d.length; let i1 = i0; while (i1 < d.length && d[i1] <= b) i1++; return [i0, i1]; };
const regOf = k => BK[k].region;
const idxOf = k => D.index[regOf(k)];
const NM = k => BK[k].name;
const INAME = D.index_name;

// ================================================================= the chapter's measures (generate_all_charts.py)
function avgPairCorr(R, members, w) {      // rolling average pairwise correlation (window w), NaN before the first full window
  const n = R.d.length, acc = new Float64Array(n), cnt = new Int32Array(n);
  for (let a = 0; a < members.length; a++) for (let b = a + 1; b < members.length; b++) {
    const x = R[members[a]], y = R[members[b]];
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let t = 0; t < n; t++) {
      sx += x[t]; sy += y[t]; sxx += x[t] * x[t]; syy += y[t] * y[t]; sxy += x[t] * y[t];
      if (t >= w) { const u = x[t - w], v = y[t - w]; sx -= u; sy -= v; sxx -= u * u; syy -= v * v; sxy -= u * v; }
      if (t >= w - 1) { const cxy = sxy - sx * sy / w, cxx = sxx - sx * sx / w, cyy = syy - sy * sy / w; if (cxx > 0 && cyy > 0) { acc[t] += cxy / Math.sqrt(cxx * cyy); cnt[t]++; } }
    }
  }
  const d = [], v = [];
  for (let t = 0; t < n; t++) if (cnt[t]) { d.push(R.d[t]); v.push(acc[t] / cnt[t]); }
  return { d, v };
}
const RCC = {};
function rollCorr(reg, w) { const k = reg + w; if (RCC[k]) return RCC[k]; const R = regionR(reg); return (RCC[k] = avgPairCorr(R, D.regions[reg], w)); }
function corrStats(c) {
  const per = (a, b) => { const [i0, i1] = between(c.d, a, b); return i1 > i0 ? mean(c.v.slice(i0, i1)) : null; };
  let im = 0; c.v.forEach((x, i) => { if (x > c.v[im]) im = i; });
  return { calm: per('2005-07-01', '2007-06-30'), c2008: per('2008-10-01', '2009-03-31'), y2017: per('2017-01-01', '2017-12-31'), c2020: per('2020-04-01', '2020-06-30'),
    max: c.v[im], date_max: c.d[im], last: c.v[c.v.length - 1], year: y => per(y + '-01-01', y + '-12-31') };
}
function pseudo(x) { const r = ranks(x), n = x.length; return r.map(v => v / (n + 1)); }
const TD = {};
function tailPair(key) {             // (x, y) = (bank portfolio, index) or (European, US portfolio); pseudo-observations
  if (TD[key]) return TD[key];
  let x, y, n;
  if (key === 'USEU') { const R = intlR(); x = portfolio(R, EU); y = portfolio(R, US); n = R.d.length; }
  else { const R = regionR(key); x = pfOf(key); y = R[D.index[key]]; n = R.d.length; }
  return (TD[key] = { x, y, n, U: pseudo(x), V: pseudo(y), corr: corr(x, y) });
}
function lam(P, q) { let a = 0, b = 0; for (let i = 0; i < P.n; i++) if (P.V[i] <= q) { b++; if (P.U[i] <= q) a++; } return a / b; }
const FC = {};
function fit(k, a) {                 // covar(): quantile regression x_sys = a + b x_i at level a (exported coefficients), quantiles of x_i
  const key = k + a; if (FC[key]) return FC[key];
  const R = regionR(regOf(k)), x = R[k], y = R[idxOf(k)], [A, B] = D.fits[k].ab[a], qa = quantile(x, +a), qm = quantile(x, 0.5);
  return (FC[key] = { a: A, b: B, q_a: qa, q_m: qm, var_i: -qa, covar: -(A + B * qa), covar_med: -(A + B * qm), dcovar: B * (qm - qa), var_sys: -quantile(y, +a), n: x.length, x, y, med: D.fits[k].med, ci: a === '0.01' ? D.fits[k].ci : null });
}
function mesOf(xi, xm, a) { const q = quantile(xm, a); let s = 0, c = 0; for (let i = 0; i < xm.length; i++) if (xm[i] <= q) { s += xi[i]; c++; } return { mes: -s / c, n: c, q }; }
function mesThr(xi, xm, c) { let s = 0, m = 0; for (let i = 0; i < xm.length; i++) if (xm[i] < c) { s += xi[i]; m++; } return { mes: -s / m, n: m }; }
const lrmes = m => 1 - Math.exp(-18 * m / 100);
const lstar = (lr, k) => 1 + (1 - k) * (1 - lr) / k;
const sriskRatio = (lr, L, k) => k * (L - 1) - (1 - k) * (1 - lr);
const MC = {};
function mesRow(k, a, c) {
  const key = k + a + c; if (MC[key]) return MC[key];
  const R = regionR(regOf(k)), x = R[k], m = R[idxOf(k)], M = mesOf(x, m, a), M2 = mesThr(x, m, c === undefined ? -2 : c), lr = lrmes(M2.mes), beta = cov(x, m) / vari(m), es = mesOf(m, m, a).mes;
  return (MC[key] = { mes: M.mes, nd: M.n, mes2: M2.mes, n2: M2.n, lrmes: lr, Lstar: lstar(lr, D.k_cap), beta, es_m: es, n: x.length });
}
function mesCrisis(a) {               // mes_vs_crisis: MES from June 2006 to June 2007, buy-and-hold return from July 2007 to December 2008
  const res = {};
  ['US', 'EU'].forEach(reg => {
    const R = joinR([...D.regions[reg], D.index[reg]], '2006-06-01', '2008-12-31'), P = joinR(D.regions[reg], '2006-06-01', '2008-12-31'), [i0, i1] = between(R.d, '2006-06-01', '2007-06-30');
    const m = R[D.index[reg]].slice(i0, i1);
    D.regions[reg].forEach(k => {
      const pd = P.pd, p = P.P[k]; let j0 = -1, j1 = -1, jl = -1;
      pd.forEach((d, j) => { if (d <= '2007-06-30') j0 = j; if (d >= '2007-07-01' && d <= '2008-12-31') { if (jl < 0) jl = j; j1 = j; } });
      res[k] = { region: reg, mes: mesOf(R[k].slice(i0, i1), m, a).mes, ret: 100 * (p[j1] / p[j0] - 1), n_pre: i1 - i0 };
    });
  });
  const ks = Object.keys(res), s = spearman(ks.map(k => res[k].mes), ks.map(k => res[k].ret));
  const x = ks.map(k => res[k].mes), y = ks.map(k => res[k].ret), b = cov(x, y) / vari(x), a0 = mean(y) - b * mean(x);
  return { res, rho: s.r, p: s.p, n: ks.length, line: [a0, b] };
}
function monthly() {                 // monthly_returns: last common trading day of each month, log returns in %
  if (monthly.c) return monthly.c;
  const R = intlR(), pd = R.pd, last = new Map();
  pd.forEach((d, i) => last.set(d.slice(0, 7), i));
  const ms = [...last.keys()], ix = ms.map(m => last.get(m)), o = { m: ms.slice(1), end: ms.slice(1).map(m => isoOf(Date.UTC(+m.slice(0, 4), +m.slice(5, 7), 0))) };
  INTL.forEach(k => { const p = R.P[k]; o[k] = ix.slice(1).map((j, i) => 100 * (Math.log(p[j]) - Math.log(p[ix[i]]))); });
  return (monthly.c = o);
}
function grangerP(y, x) {            // y_t = a + b y_{t-1} + c x_{t-1} + e_t: two-sided p-value of the t test of c
  const n = y.length - 1, X = [], Y = [];
  for (let t = 1; t <= n; t++) { X.push([1, y[t - 1], x[t - 1]]); Y.push(y[t]); }
  const XtX = [0, 1, 2].map(i => [0, 1, 2].map(j => X.reduce((s, r) => s + r[i] * r[j], 0))), Xty = [0, 1, 2].map(i => X.reduce((s, r, t) => s + r[i] * Y[t], 0));
  const B = solve(XtX, Xty).map(r => r[0]), e = Y.map((v, t) => v - B[0] - B[1] * X[t][1] - B[2] * X[t][2]), s2 = sum(e.map(v => v * v)) / (n - 3);
  const inv = solve(XtX, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), se = Math.sqrt(s2 * inv[2][2]);
  return t2sf(B[2] / se, n - 3);
}
function grangerNet(e, w, level) {   // the network of the window of w months that ends at month index e (exclusive)
  const M = monthly(), A = INTL.map(() => INTL.map(() => 0));
  INTL.forEach((a, i) => INTL.forEach((b, j) => { if (i !== j) A[i][j] = +(grangerP(M[b].slice(e - w, e), M[a].slice(e - w, e)) < level); }));
  const n = INTL.length, links = sum(A.map(r => sum(r)));
  return { A, links, dgc: 100 * links / (n * (n - 1)), from: M.m[e - w], to: M.m[e - 1] };
}
const DGC = {};
function rollDGC(w, level) {
  const key = w + '|' + level; if (DGC[key]) return DGC[key];
  const M = monthly(), d = [], v = [];
  for (let e = w; e <= M.m.length; e++) { d.push(M.end[e - 1]); v.push(grangerNet(e, w, level).dgc); }
  return (DGC[key] = { d, v });
}
const VC = {};
function varFit(a, b) {               // VAR(1) by OLS on the joined returns of the 11 banks between dates a and b
  const key = a + b; if (VC[key]) return VC[key];
  const R = intlR(), [i0, i1] = between(R.d, a, b), Y = INTL.map(k => R[k].slice(i0, i1)), Tn = i1 - i0, k = INTL.length, p = D.dy.p;
  const X = [], Yt = [];
  for (let t = p; t < Tn; t++) { const r = [1]; for (let l = 1; l <= p; l++) for (let j = 0; j < k; j++) r.push(Y[j][t - l]); X.push(r); Yt.push(Y.map(s => s[t])); }
  const Xt = tr_(X), Bm = solve(matmul(Xt, X), matmul(Xt, Yt)), E = Yt.map((y, t) => y.map((v, j) => v - X[t].reduce((s, x, c) => s + x * Bm[c][j], 0)));
  const S = matmul(tr_(E), E).map(r => r.map(v => v / (E.length - X[0].length)));
  const A = Array.from({ length: p }, (_, l) => tr_(Bm.slice(1 + l * k, 1 + (l + 1) * k)));
  return (VC[key] = { A, S, n: Tn, d0: R.d[i0], d1: R.d[i1 - 1] });
}
function gfevd(A, S, H) {             // generalised forecast error variance decomposition, rows normalised to 100
  const k = S.length, I = S.map((r, i) => r.map((_, j) => +(i === j))), Phi = [I];
  for (let h = 1; h < H; h++) { let P = S.map(r => r.map(() => 0)); for (let l = 0; l < Math.min(h, A.length); l++) { const Q = matmul(A[l], Phi[h - l - 1]); P = P.map((r, i) => r.map((v, j) => v + Q[i][j])); } Phi.push(P); }
  const num = S.map(r => r.map(() => 0)), den = new Array(k).fill(0);
  Phi.forEach(P => { const PS = matmul(P, S), PSP = matmul(PS, tr_(P)); for (let i = 0; i < k; i++) { den[i] += PSP[i][i]; for (let j = 0; j < k; j++) num[i][j] += PS[i][j] ** 2; } });
  const th = num.map((r, i) => r.map((v, j) => v / S[j][j] / den[i]));
  return th.map(r => { const s = sum(r); return r.map(v => 100 * v / s); });
}
function connect(Dm) {
  const k = Dm.length, from = Dm.map((r, i) => sum(r) - r[i]), to = Dm[0].map((_, j) => sum(Dm.map(r => r[j])) - Dm[j][j]);
  return { table: Dm, from, to, net: to.map((v, i) => v - from[i]), total: sum(from) / k };
}
const DYC = {};
function dyTable(H, a, b) { a = a || '2005-01-01'; b = b || D.end; const key = H + a + b; if (DYC[key]) return DYC[key]; const V = varFit(a, b); return (DYC[key] = Object.assign(connect(gfevd(V.A, V.S, H)), { n: V.n, d0: V.d0, d1: V.d1 })); }

// ================================================================= formatting, tables, panels
const SEP = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, SEP).replace('-', '−');
const DT = s => T.date(s);
const LV = a => nf(100 * a, Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);
const LVE = a => (100 * a).toFixed(Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);
const r2 = x => Math.round(x * 100) / 100;
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
  el.style.height = (stack ? (o.hs || 260) * n + 120 : (o.h || 380)) + 'px';
  const g = stack ? 0.1 : (o.gap || 0.1), L = {}, ax = [];
  const ws = o.widths && !stack ? o.widths : Array.from({ length: n }, () => 1 / n);
  let x0 = 0;
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '', wi = ws[i] * (1 - g * (n - 1)), dom = [x0, x0 + wi]; x0 += wi + g;
    const hgt = (1 - g * (n - 1)) / n, top = 1 - i * (hgt + g);
    L['xaxis' + s] = { domain: stack ? [0, 1] : dom, anchor: 'y' + s };
    L['yaxis' + s] = { domain: stack ? [Math.max(0, top - hgt), top] : [0, 1], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack };
}
function grid(id, nr, nc, o) {       // nr x nc panels; one column below 560 px
  o = o || {};
  const el = $(id), stack = el.clientWidth < 560, R = stack ? nr * nc : nr, Cn = stack ? 1 : nc;
  el.style.height = (stack ? (o.hs || 240) * R + 120 : (o.h || 560)) + 'px';
  const gx = 0.09, gy = stack ? 0.05 : 0.15, L = {}, ax = [], w = (1 - gx * (Cn - 1)) / Cn, h = (1 - gy * (R - 1)) / R;
  for (let i = 0; i < nr * nc; i++) {
    const r = Math.floor(i / Cn), c = i % Cn, s = i ? String(i + 1) : '';
    L['xaxis' + s] = { domain: [c * (w + gx), c * (w + gx) + w], anchor: 'y' + s };
    L['yaxis' + s] = { domain: [1 - r * (h + gy) - h, 1 - r * (h + gy)], anchor: 'x' + s };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s, top: 1 - r * (h + gy), left: c * (w + gx) });
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
const vtr = (x, y0, y1, color, name, dash, extra) => Object.assign({ x: [x, x], y: [y0, y1], mode: 'lines', line: { color, width: 1.3, dash: dash || 'dash' }, name, hoverinfo: 'skip' }, extra || {});
const htr = (x0, x1, y, color, name, dash, extra) => Object.assign({ x: [x0, x1], y: [y, y], mode: 'lines', line: { color, width: 1.2, dash: dash || 'dash' }, name, hoverinfo: 'skip' }, extra || {});
const BAR = (x, y, color, name, extra) => Object.assign({ type: 'bar', x, y, marker: { color }, name }, extra || {});
const ptitle = (P, i, text) => ({ text, xref: 'paper', yref: 'paper', x: P.ax[i].left !== undefined ? P.ax[i].left : (P.L[P.ax[i].xa].domain[0]), y: P.ax[i].top !== undefined ? P.ax[i].top : 1, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: DARK } });
function events(lo, hi, yref) {       // the four event lines of the chapter's charts (Lehman, Draghi, COVID-19, SVB)
  const sh = [], an = [];
  D.events.forEach(([d, lab]) => {
    if (lo && d < lo) return; if (hi && d > hi) return;
    sh.push({ type: 'line', xref: 'x', yref: yref || 'paper', x0: d, x1: d, y0: 0, y1: 1, line: { color: PURPLE, width: 1, dash: 'dot' } });
    an.push({ x: d, y: 1, xref: 'x', yref: 'paper', text: lab, textangle: -90, xanchor: 'right', yanchor: 'top', showarrow: false, font: { size: 10.5, color: DARK } });
  });
  return { shapes: sh, annotations: an };
}
const regLab = reg => D.reg_lab[reg];
const sgn = (x, d) => (x > 0 ? '+' : '') + nf(x, d);

// ================================================================= 1. bank portfolios since 2005
function biBlock() {
  const y = kSel('bi-y'), s0 = y + '-01-01', nw = single('bi-chart', 390), tr = [], res = {};
  REGS.forEach(reg => {
    const R = regionR(reg), p = pfOf(reg), i0 = Math.max(0, R.d.findIndex(d => d >= s0)), d = R.d.slice(i0), v = []; let c = 0, mx = -Infinity, dd = 0, ddd = d[0];
    for (let t = i0; t < R.d.length; t++) { c += p[t]; const z = 100 * Math.exp(c / 100); v.push(z); mx = Math.max(mx, z); if (z / mx - 1 < dd) { dd = z / mx - 1; ddd = R.d[t]; } }
    res[reg] = { last: v[v.length - 1], maxdd: 100 * dd, date_dd: ddd, start: d[0] };
    tr.push({ x: d, y: v, mode: 'lines', line: { color: RCOL[reg], width: 1.3 }, name: regLab(reg), hovertemplate: `${regLab(reg)} %{x}: %{y:.1f}<extra></extra>` });
  });
  tr.push(htr(res.US.start, D.end, 100, AMBER, 'starting value 100'));
  const ev = events(res.US.start);
  plot('bi-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`value of 100 invested in ${y} (log scale)`, { type: 'log' }), legend: LEG(nw, -0.1), margin: M0, shapes: ev.shapes, annotations: ev.annotations }));
  out('bi-out', [T.bi(y, REGS.map(r => [fint(res[r].last), nf(res[r].maxdd, 0), DT(res[r].date_dd), DT(res[r].start)]), y !== '2005')]);
}

// ================================================================= 2. four episodes side by side
const EPN = Object.keys(D.episodes);
function episode(e, sh) {             // episode_paths with the start date moved by sh calendar days
  const [a0, b] = D.episodes[e], a = addDays(a0, sh), o = {};
  REGS.forEach(reg => {
    if (a < D.region_start[reg]) return;
    const R = regionR(reg), [i0, i1] = between(R.d, a, b), p = pfOf(reg).slice(i0 + 1, i1); let c = 0;
    o[reg] = { d: R.d.slice(i0 + 1, i1), v: p.map(x => (c += x, 100 * (Math.exp(c / 100) - 1))) };
  });
  const R = regionR('US'), [i0, i1] = between(R.d, a, b); let c = 0;
  o.sp500 = { d: R.d.slice(i0 + 1, i1), v: R.sp500.slice(i0 + 1, i1).map(x => (c += x, 100 * (Math.exp(c / 100) - 1))) };
  Object.values(o).forEach(s => { let im = 0; s.v.forEach((x, i) => { if (x < s.v[im]) im = i; }); s.min = s.v[im]; s.date_min = s.d[im]; s.end = s.v[s.v.length - 1]; });
  return { o, a, b };
}
function epBlock() {
  const sh = +$('ep-d').value, P = grid('ep-chart', 2, 2, { h: 600 }), tr = [], an = [], res = {}; label('ep-d').textContent = sh;
  EPN.forEach((e, i) => {
    const E = episode(e, sh); res[e] = E;
    REGS.forEach(reg => { if (E.o[reg]) tr.push(on2(P.ax[i], { x: E.o[reg].d, y: E.o[reg].v, mode: 'lines', line: { color: RCOL[reg], width: 1.4 }, name: regLab(reg), legendgroup: reg, showlegend: i === 0, hovertemplate: `${regLab(reg)} %{x}: %{y:.1f}%<extra></extra>` })); });
    tr.push(on2(P.ax[i], { x: E.o.sp500.d, y: E.o.sp500.v, mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dash' }, name: 'S&P 500', legendgroup: 'sp', showlegend: i === 0, hovertemplate: 'S&P 500 %{x}: %{y:.1f}%<extra></extra>' }));
    tr.push(on2(P.ax[i], htr(E.o.sp500.d[0], E.o.sp500.d[E.o.sp500.d.length - 1], 0, PURPLE, '', 'dot', { showlegend: false })));
    setAx(P, i, AX(''), AX(i % 2 === 0 || P.stack ? 'cumulative return (%)' : ''));
    an.push(ptitle(P, i, e));
  });
  plot('ep-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.04 : -0.08), margin: { l: 56, r: 12, t: 24, b: 40 }, annotations: an })));
  const f = (e, k) => res[e].o[k] ? nf(res[e].o[k].min, 0) : null;
  out('ep-out', [T.ep(String(sh).replace('-', '−'), EPN.map(e => [e, DT(res[e].a), f(e, 'US'), f(e, 'EU'), f(e, 'RO'), f(e, 'sp500'), nf(res[e].o.sp500.end, 0)]))]);
}

// ================================================================= 3. correlation between banks rises in crises
function rcBlock() {
  const w = +$('rc-w').value, nw = single('rc-chart', 380), S = {}; label('rc-w').textContent = w;
  const tr = REGS.map(reg => { const c = rollCorr(reg, w); S[reg] = corrStats(c); return { x: c.d, y: c.v, mode: 'lines', line: { color: RCOL[reg], width: 1.2 }, name: regLab(reg), hovertemplate: `${regLab(reg)} %{x}: %{y:.2f}<extra></extra>` }; });
  const ev = events();
  plot('rc-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`average pairwise correlation (${w}-day window)`, { range: [-0.05, 1] }), legend: LEG(nw, -0.1), margin: M0, shapes: ev.shapes, annotations: ev.annotations }));
  const u = S.US, r = S.RO;
  out('rc-out', [T.rc(w, nf(u.calm, 2), nf(u.max, 2), DT(u.date_max), nf(S.EU.calm, 2), nf(S.EU.max, 2), DT(S.EU.date_max), nf(r.y2017, 2), nf(r.c2020, 2), nf(r.max, 2), DT(r.date_max), w !== 120)]);
}

// ================================================================= x1. correlations by period (table)
function ctBlock() {
  const w = +kSel('ct-w'), y = kSel('ct-y'), S = {}, P = panels('ct-chart', 1, { h: 360 });
  REGS.forEach(reg => { S[reg] = corrStats(rollCorr(reg, w)); });
  const per = ['calm', 'c2008', 'c2020'], xl = ['Jul 2005–Jun 2007', 'Oct 2008–Mar 2009', 'Apr–Jun 2020', y, 'maximum'];
  const val = reg => [...per.map(p => S[reg][p]), S[reg].year(y), S[reg].max];
  const tr = REGS.map(reg => on2(P.ax[0], BAR(xl, val(reg).map(v => (v === null ? null : v)), RCOL[reg], regLab(reg), { text: val(reg).map(v => (v === null ? '' : v.toFixed(2))), textposition: 'outside', textfont: { size: 10, color: DARK }, cliponaxis: false, hovertemplate: `${regLab(reg)}, %{x}: %{y:.2f}<extra></extra>` })));
  setAx(P, 0, AX('', { type: 'category' }), AX(`mean of the ${w}-day average pairwise correlation`, { range: [0, 1.12] }));
  plot('ct-chart', tr, layout(Object.assign(P.L, { barmode: 'group', legend: LEG(narrow('ct-chart'), -0.16), margin: M0 })));
  const c = v => (v === null ? '—' : nf(v, 2));
  table('ct-tab', T.h.ct(y), REGS.map(reg => [T.reg[reg], ...val(reg).slice(0, 4).map(c), `${c(S[reg].max)} (${DT(S[reg].date_max)})`, c(S[reg].last)]));
  out('ct-out', [T.ct(w, y, REGS.map(reg => c(S[reg].year(y))), w !== 120)]);
}

// ================================================================= x2. Forbes and Rigobon (worked example)
const FRW = { lehman: ['2008-09-15', '2008-12-31'], euro: D.episodes['Euro area 2011-2012'], covid: D.episodes['March 2020'], svb: D.episodes['March 2023'] };
function frTest(e, src) {
  const R = intlR(), us = portfolio(R, US), eu = portfolio(R, EU), [c0, c1] = between(R.d, '2005-01-01', '2007-06-29'), [k0, k1] = between(R.d, FRW[e][0], FRW[e][1]);
  const S = src === 'EU' ? eu : us, O = src === 'EU' ? us : eu;
  const rc = corr(S.slice(c0, c1), O.slice(c0, c1)), rk = corr(S.slice(k0, k1), O.slice(k0, k1)), v0 = vari(S.slice(c0, c1)), v1 = vari(S.slice(k0, k1)), delta = v1 / v0 - 1;
  return { rc, rk, v0, v1, delta, adj: rk / Math.sqrt(1 + delta * (1 - rk * rk)), n0: c1 - c0, n1: k1 - k0, d0: R.d[c0], d1: R.d[c1 - 1], e0: R.d[k0], e1: R.d[k1 - 1] };
}
function frBlock() {
  const e = kSel('fr-e'), src = kSel('fr-s'), F = frTest(e, src), P = panels('fr-chart', 2, { h: 340 });
  const dmax = Math.max(10, 1.5 * F.delta), ds = Array.from({ length: 121 }, (_, i) => dmax * i / 120);
  const tr = [on2(P.ax[0], BAR(['ρ calm', 'ρ crisis', 'ρ* (corrected)'], [F.rc, F.rk, F.adj], [BLUE, RED, GREEN], 'correlations', { text: [F.rc, F.rk, F.adj].map(v => v.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })),
    on2(P.ax[1], { x: ds, y: ds.map(d => F.rk / Math.sqrt(1 + d * (1 - F.rk ** 2))), mode: 'lines', line: { color: GREEN, width: 2 }, name: `ρ*(δ) for ρ crisis = ${F.rk.toFixed(2)}`, hovertemplate: 'δ = %{x:.1f}: ρ* = %{y:.3f}<extra></extra>' }),
    on2(P.ax[1], htr(0, dmax, F.rc, BLUE, `ρ calm = ${F.rc.toFixed(2)}`)),
    on2(P.ax[1], { x: [F.delta], y: [F.adj], mode: 'markers', marker: { color: RED, size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `measured δ = ${F.delta.toFixed(1)}`, hoverinfo: 'skip' })];
  setAx(P, 0, AX('', { type: 'category' }), AX('correlation', { range: [0, 1] })); setAx(P, 1, AX('δ = σ²(crisis)/σ²(calm) − 1'), AX('ρ*', { range: [0, 1] }));
  plot('fr-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  const sn = T.src[src], other = T.src[src === 'EU' ? 'US' : 'EU'];
  steps('fr-st', [T.fr1(sn, other, DT(F.d0), DT(F.d1), F.n0, DT(F.e0), DT(F.e1), F.n1),
    T.fr2(nf(F.rc, 2), nf(F.rk, 2), nf(F.v0, 2), nf(F.v1, 1), nf(F.delta, 1)),
    T.fr3(nf(F.rk, 2), nf(F.delta, 1), nf(1 - F.rk ** 2, 3), nf(1 + F.delta * (1 - F.rk ** 2), 1), nf(F.adj, 2)),
    T.fr4(nf(F.adj, 2), nf(F.rc, 2), F.adj > F.rc)]);
  out('fr-out', [T.fr(T.ep_[e], sn, nf(F.rc, 2), nf(F.rk, 2), nf(F.delta, 1), nf(F.adj, 2), F.adj > F.rc, e !== 'lehman' || src !== 'US')]);
}

// ================================================================= x3. tail dependence of banks and markets (table)
const PAIRS = ['US', 'EU', 'RO', 'USEU'];
const PCOL = { US: RCOL.US, EU: RCOL.EU, RO: RCOL.RO, USEU: PURPLE };
const PNAME = { US: 'US banks, S&P 500', EU: 'European banks, Euro Stoxx 50', RO: 'Romanian banks, BET', USEU: 'European banks, US banks' };
const QG = Array.from({ length: 40 }, (_, i) => 0.005 * (i + 1));
function ttBlock() {
  const q = +$('tt-q').value / 100, P = panels('tt-chart', 1, { h: 380 }); label('tt-q').textContent = LV(q) + '%';
  const tr = PAIRS.map(k => on2(P.ax[0], { x: QG.map(v => 100 * v), y: QG.map(v => lam(tailPair(k), v)), mode: 'lines', line: { color: PCOL[k], width: 1.6 }, name: PNAME[k], hovertemplate: `${PNAME[k]}: λ(%{x:.1f}%) = %{y:.2f}<extra></extra>` }));
  tr.push(on2(P.ax[0], { x: [0.5, 20], y: [0.005, 0.2], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dot' }, name: 'independence: λ(q) = q', hoverinfo: 'skip' }));
  tr.push(on2(P.ax[0], { x: PAIRS.map(() => 100 * q), y: PAIRS.map(k => lam(tailPair(k), q)), mode: 'markers', marker: { color: PAIRS.map(k => PCOL[k]), size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `λ(q) at q = ${LVE(q)}%`, hoverinfo: 'skip' }));
  setAx(P, 0, AX('q (%)', { range: [0, 20.5] }), AX('λ(q) = P(U₁ ≤ q | U₂ ≤ q)', { range: [0, 1] }));
  plot('tt-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('tt-chart'), -0.16), margin: M0 })));
  table('tt-tab', T.h.tt(LV(q)), PAIRS.map(k => { const Pp = tailPair(k), cp = D.copula[k]; return [T.pair[k], fint(Pp.n), nf(Pp.corr, 2), nf(lam(Pp, 0.05), 2), nf(lam(Pp, 0.01), 2), nf(cp.nu, 1), nf(cp.lambda, 2), { v: nf(lam(Pp, q), 2), c: 'hi' }]; }));
  out('tt-out', [T.tt(LV(q), PAIRS.map(k => nf(lam(tailPair(k), q), 2)), nf(q, 3), Math.abs(q - 0.05) > 1e-9 && Math.abs(q - 0.01) > 1e-9)]);
}

// ================================================================= 4. joint bad days
function tsBlock() {
  const q = +$('ts-q').value / 100, P = panels('ts-chart', 2, { h: 380 }), tr = [], res = {}; label('ts-q').textContent = LV(q) + '%';
  ['US', 'RO'].forEach((reg, i) => {
    const Pp = tailPair(reg), xs = [], ys = [], bx = [], by = [];
    for (let t = 0; t < Pp.n; t++) { const u = Pp.V[t], v = Pp.U[t]; if (u > 0.3 || v > 0.3) continue; if (u <= q && v <= q) { bx.push(u); by.push(v); } else { xs.push(u); ys.push(v); } }
    let both = 0; for (let t = 0; t < Pp.n; t++) if (Pp.V[t] <= q && Pp.U[t] <= q) both++;
    res[reg] = { both, exp: q * q * Pp.n, n: Pp.n };
    tr.push(on2(P.ax[i], { x: xs, y: ys, mode: 'markers', marker: { color: RCOL[reg], size: 3, opacity: 0.45 }, name: regLab(reg), hoverinfo: 'skip' }));
    tr.push(on2(P.ax[i], { x: bx, y: by, mode: 'markers', marker: { color: ORANGE, size: 5 }, name: `both in their worst ${LVE(q)}% of days`, showlegend: i === 0, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[i], { x: [0, q, q], y: [q, q, 0], mode: 'lines', line: { color: AMBER, width: 1.6 }, showlegend: false, hoverinfo: 'skip' }));
    setAx(P, i, AX(`${INAME[D.index[reg]]} (uniform scale)`, { range: [0, 0.3] }), AX(i === 0 || P.stack ? 'bank portfolio (uniform scale)' : '', { range: [0, 0.3] }));
  });
  const an = ['US', 'RO'].map((reg, i) => ptitle(P, i, `${reg === 'US' ? 'US banks' : 'Romanian banks'}: ${res[reg].both} joint bad days, ${res[reg].exp.toFixed(0)} expected`));
  plot('ts-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: { l: 56, r: 12, t: 26, b: 40 }, annotations: an })));
  out('ts-out', [T.ts(LV(q), res.US.both, nf(res.US.exp, 1), nf(res.US.both / res.US.exp, 1), res.RO.both, nf(res.RO.exp, 1), nf(res.RO.both / res.RO.exp, 1), fint(res.US.n), fint(res.RO.n), Math.abs(q - 0.05) > 1e-9)]);
}

// ================================================================= x4. CoVaR of JPMorgan Chase (worked example)
function cwBlock() {
  const a = kSel('cw-a'), k = kSel('cw-k'), F = fit(k, a), P = panels('cw-chart', 2, { h: 350, widths: [0.58, 0.42] }), ix = INAME[idxOf(k)];
  const xs = Array.from({ length: 61 }, (_, i) => Math.min(-10, 1.3 * F.q_a) + i * (2 - Math.min(-10, 1.3 * F.q_a)) / 60);
  const tr = [on2(P.ax[0], { x: xs, y: xs.map(x => -(F.a + F.b * x)), mode: 'lines', line: { color: RED, width: 2 }, name: `minus the ${LVE(+a)}% quantile of the ${ix} given the bank's return`, hovertemplate: 'x = %{x:.2f}%: %{y:.2f}%<extra></extra>' }),
    on2(P.ax[0], { x: [F.q_a, F.q_m], y: [F.covar, F.covar_med], mode: 'markers+text', marker: { color: [RED, GREEN], size: 11, symbol: ['circle', 'square'], line: { color: DARK, width: 1 } }, text: [`CoVaR ${F.covar.toFixed(2)}%`, `${F.covar_med.toFixed(2)}%`], textposition: ['top right', 'bottom left'], textfont: { size: 11, color: DARK }, name: 'bank at its quantile and at its median', cliponaxis: false, hoverinfo: 'skip' }),
    on2(P.ax[0], htr(xs[0], xs[xs.length - 1], F.var_sys, AMBER, `unconditional VaR ${LVE(+a)}% of the ${ix}: ${F.var_sys.toFixed(2)}%`, 'dot')),
    on2(P.ax[1], BAR(['−â', '−b̂ q̂ₐ', 'CoVaR', 'at median', 'ΔCoVaR'], [-F.a, -F.b * F.q_a, F.covar, F.covar_med, F.dcovar], [BLUE, BLUE, RED, GREEN, ORANGE], 'decomposition', { text: [-F.a, -F.b * F.q_a, F.covar, F.covar_med, F.dcovar].map(v => v.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}%<extra></extra>' }))];
  setAx(P, 0, AX(`${NM(k)} daily log return x (%)`), AX(`loss of the ${ix} (%)`)); setAx(P, 1, AX('', { type: 'category' }), AX('%', { range: [0, 1.18 * Math.max(F.covar, -F.a, -F.b * F.q_a)] }));
  plot('cw-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  const A2 = r2(F.a), B2 = r2(F.b), Q2 = r2(F.q_a), M2 = r2(F.q_m), prod = B2 * Q2;
  steps('cw-we', [T.cw1(NM(k), ix, fint(F.n), LV(+a), nf(F.a, 2), nf(F.b, 2)), T.cw2(NM(k), LV(+a), nf(F.q_a, 2), nf(F.var_i, 2), nf(F.q_m, 2)),
    T.cw3(LV(+a), nf(A2, 2), nf(B2, 2), nf(Q2, 2), nf(-A2, 2), nf(-prod, 3), nf(-A2 - prod, 3), nf(F.covar, 2), nf(F.covar_med, 2)),
    T.cw4(LV(+a), nf(B2, 2), nf(M2, 2), nf(Q2, 2), nf(M2 - Q2, 2), nf(B2 * (M2 - Q2), 2), nf(F.dcovar, 2), nf(F.var_sys, 2), ix)]);
  out('cw-out', [T.cw(NM(k), LV(+a), nf(F.covar, 2), nf(F.covar_med, 2), nf(F.dcovar, 2), nf(F.var_sys, 2), ix, a !== '0.01' || k !== 'JPM')]);
}

// ================================================================= 5. CoVaR in a picture
function cqBlock() {
  const a = kSel('cq-a'), k = kSel('cq-k'), F = fit(k, a), nw = single('cq-chart', 430), ix = INAME[idxOf(k)], xs = [Math.min(...F.x), Math.max(...F.x)], [a5, b5] = F.med;
  const tr = [{ x: F.x, y: F.y, mode: 'markers', marker: { color: BLUE, size: 3, opacity: 0.35 }, name: `daily returns, ${NM(k)} and ${ix}`, hoverinfo: 'skip' },
    { x: xs, y: xs.map(x => F.a + F.b * x), mode: 'lines', line: { color: RED, width: 2 }, name: `${LVE(+a)}% quantile regression: ${F.a.toFixed(2)} + ${F.b.toFixed(2)} x`, hoverinfo: 'skip' },
    { x: xs, y: xs.map(x => a5 + b5 * x), mode: 'lines', line: { color: GREEN, width: 2 }, name: `median regression: ${a5.toFixed(2)} + ${b5.toFixed(2)} x`, hoverinfo: 'skip' },
    vtr(F.q_a, -13, 8, ORANGE, `${LVE(+a)}% quantile of ${k}: ${F.q_a.toFixed(2)}%`), vtr(F.q_m, -13, 8, PURPLE, `median of ${k}: ${F.q_m.toFixed(2)}%`),
    { x: [F.q_a], y: [-F.covar], mode: 'markers', marker: { color: RED, size: 10 }, name: `minus CoVaR ${LVE(+a)}%: ${(-F.covar).toFixed(2)}%`, hoverinfo: 'skip' },
    { x: [F.q_m], y: [-F.covar_med], mode: 'markers', marker: { color: GREEN, size: 10, symbol: 'square' }, name: `minus CoVaR ${LVE(+a)}% at the median: ${(-F.covar_med).toFixed(2)}%`, hoverinfo: 'skip' }];
  plot('cq-chart', tr, layout({ xaxis: AX(`${NM(k)} daily log return (%)`, { range: [-12, 12] }), yaxis: AX(`${ix} daily log return (%)`, { range: [-13, 8] }), legend: LEG(nw, -0.14), margin: M0 }));
  out('cq-out', [T.cq(NM(k), LV(+a), nf(F.a, 2), nf(F.b, 2), nf(a5, 2), nf(b5, 2), nf(F.covar, 2), nf(F.covar_med, 2), nf(F.dcovar, 2), a !== '0.01' || k !== 'JPM')]);
}

// ================================================================= 6. Delta-CoVaR of 13 banks
function dcBlock() {
  const a = kSel('dc-a'), nw = single('dc-chart', 400), Fs = ALL.map(k => fit(k, a)), v = Fs.map(f => f.dcovar);
  const tr = [BAR(ALL.map(NM), v, ALL.map(k => RCOL[regOf(k)]), `ΔCoVaR ${LVE(+a)}%`, { showlegend: false, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>',
    error_y: a === '0.01' ? { type: 'data', symmetric: false, array: Fs.map(f => f.ci[1] - f.dcovar), arrayminus: Fs.map(f => f.dcovar - f.ci[0]), color: DARK, thickness: 1, width: 3 } : undefined })];
  REGS.forEach(r => tr.push({ x: [null], y: [null], type: 'bar', marker: { color: RCOL[r] }, name: `${regLab(r).replace(' (TLV, BRD, since 2010)', '')} (system: ${INAME[D.index[r]]})` }));
  if (a === '0.01') tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK }, name: '95% block-bootstrap interval' });
  plot('dc-chart', tr, layout({ xaxis: AX('', { tickangle: -30 }), yaxis: AX(`ΔCoVaR ${LVE(+a)}% (%)`), legend: LEG(nw, -0.3), margin: M0 }));
  const rng = reg => { const z = D.regions[reg].map(k => fit(k, a).dcovar); return [nf(Math.min(...z), 1), nf(Math.max(...z), 1)]; };
  let im = 0; v.forEach((x, i) => { if (x > v[im]) im = i; });
  out('dc-out', [T.dc(LV(+a), rng('US'), rng('EU'), nf(fit('TLV', a).dcovar, 2), nf(fit('BRD', a).dcovar, 2), NM(ALL[im]), a === '0.01', a !== '0.01')]);
}

// ================================================================= x5. interpretation of Delta-CoVaR (table)
function dxBlock() {
  const a2 = kSel('dx-a'), P = panels('dx-chart', 1, { h: 380 }), F1 = ALL.map(k => fit(k, '0.01')), F2 = ALL.map(k => fit(k, a2)), s = spearman(F1.map(f => f.dcovar), F2.map(f => f.dcovar));
  const tr = REGS.map(reg => { const ks = D.regions[reg]; return on2(P.ax[0], { x: ks.map(k => fit(k, '0.01').dcovar), y: ks.map(k => fit(k, a2).dcovar), mode: 'markers+text', text: ks, textposition: 'top right', textfont: { size: 10, color: DARK }, marker: { color: RCOL[reg], size: 9 }, name: regLab(reg), cliponaxis: false, hovertemplate: `%{text}: %{x:.2f}%, %{y:.2f}%<extra></extra>` }); });
  setAx(P, 0, AX('ΔCoVaR 1% (%)'), AX(`ΔCoVaR ${LVE(+a2)}% (%)`));
  plot('dx-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('dx-chart'), -0.16), margin: M0 })));
  table('dx-tab', T.h.dx(LV(+a2)), ALL.map((k, i) => { const f = F1[i]; return [NM(k), fint(f.n), nf(f.a, 2), nf(f.b, 2), nf(f.q_a, 2), nf(f.q_m, 2), nf(f.covar, 2), nf(f.covar_med, 2), { v: nf(f.dcovar, 2), c: 'hi' }, `[${nf(f.ci[0], 2)}; ${nf(f.ci[1], 2)}]`, nf(F2[i].dcovar, 2)]; }));
  out('dx-out', [T.dx(LV(+a2), nf(s.r, 2), a2 !== '0.05')]);
}

// ================================================================= x6. Banca Transilvania and the BET (worked example)
function tbBlock() {
  const a = kSel('tb-a'), k = kSel('tb-k'), F = fit(k, a), P = panels('tb-chart', 2, { h: 340, widths: [0.58, 0.42] }), ix = INAME[idxOf(k)];
  const bins = Array.from({ length: 81 }, (_, i) => -10 + 0.25 * i + 0.125), cnt = bins.map(() => 0);
  F.x.forEach(v => { const j = Math.floor((v + 10) / 0.25); if (j >= 0 && j < cnt.length) cnt[j]++; });
  const hmax = 1.1 * Math.max(...cnt);
  const tr = [on2(P.ax[0], BAR(bins, cnt, RCOL[regOf(k)], `${NM(k)}: daily log returns (bins of 0.25%)`, { marker: { color: RCOL[regOf(k)], opacity: 0.55 }, hovertemplate: '%{x:.2f}%: %{y} days<extra></extra>' })),
    on2(P.ax[0], vtr(F.q_a, 0, hmax, ORANGE, `q̂ ${LVE(+a)}% = ${F.q_a.toFixed(2)}% (stress)`)),
    on2(P.ax[0], vtr(F.q_m, 0, hmax, PURPLE, `median = ${F.q_m.toFixed(2)}%`)),
    on2(P.ax[1], BAR(['CoVaR (stress)', 'CoVaR (median)', 'ΔCoVaR'], [F.covar, F.covar_med, F.dcovar], [RED, GREEN, ORANGE], 'CoVaR', { text: [F.covar, F.covar_med, F.dcovar].map(v => v.toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }))];
  setAx(P, 0, AX(`${NM(k)} daily log return (%)`, { range: [-10, 10] }), AX('days')); setAx(P, 1, AX('', { type: 'category' }), AX(`loss of the ${ix} (%)`, { range: [0, 1.2 * F.covar] }));
  plot('tb-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0, bargap: 0.02 })));
  const gap = F.q_m - F.q_a;
  steps('tb-we', [T.tb1(NM(k), ix, fint(F.n), LV(+a), nf(F.b, 3)), T.tb2(LV(+a), nf(F.q_a, 3), nf(F.var_i, 2), nf(F.q_m, 3)),
    T.tb3(LV(+a), nf(F.b, 3), nf(F.q_m, 3), nf(-F.q_a, 3), nf(gap, 3), nf(F.dcovar, 2), nf(F.covar, 2)),
    F.ci ? T.tb4(nf(F.ci[0], 2), nf(F.ci[1], 2)) : T.tb4x(), T.tb5(nf(F.dcovar, 2), ix)]);
  out('tb-out', [T.tb(NM(k), LV(+a), nf(F.dcovar, 2), nf(F.covar, 2), nf(F.covar_med, 2), a !== '0.01' || k !== 'TLV')]);
}

// ================================================================= 7. VaR is not Delta-CoVaR
function vdBlock() {
  const a = kSel('vd-a'), nw = single('vd-chart', 400), s = spearman(ALL.map(k => fit(k, a).var_i), ALL.map(k => fit(k, a).dcovar));
  const tr = REGS.map(reg => { const ks = D.regions[reg]; return { x: ks.map(k => fit(k, a).var_i), y: ks.map(k => fit(k, a).dcovar), mode: 'markers+text', text: ks, textposition: 'top right', textfont: { size: 10.5, color: DARK }, marker: { color: RCOL[reg], size: 10 }, name: regLab(reg), cliponaxis: false, hovertemplate: '%{text}: VaR %{x:.2f}%, ΔCoVaR %{y:.2f}%<extra></extra>' }; });
  plot('vd-chart', tr, layout({ xaxis: AX(`VaR ${LVE(+a)}% of the bank (%)`), yaxis: AX(`ΔCoVaR ${LVE(+a)}% (%)`), legend: LEG(nw, -0.14), margin: M0 }));
  const byV = [...ALL].sort((x, y) => fit(y, a).var_i - fit(x, a).var_i), byD = [...ALL].sort((x, y) => fit(x, a).dcovar - fit(y, a).dcovar);
  out('vd-out', [T.vd(LV(+a), nf(s.r, 2), byV.slice(0, 3).map(NM), byD.slice(0, 3).map(NM), a !== '0.01')]);
}

// ================================================================= 8. Delta-CoVaR through time
const DYK = ['JPM', 'DBK', 'TLV'];
const DYN = {}; DYK.forEach(k => { DYN[k] = {}; Object.entries(D.dyn[k]).forEach(([a, s]) => { DYN[k][a] = { d: dated(s), v: s.v, b: s.b }; }); });
function dnStats(S) { let im = 0; S.v.forEach((x, i) => { if (x > S.v[im]) im = i; }); const [i0, i1] = between(S.d, '2017-01-01', '2017-12-31'); return { mean: mean(S.v), max: S.v[im], date_max: S.d[im], y2017: mean(S.v.slice(i0, i1)), last: S.v[S.v.length - 1] }; }
function dnBlock() {
  const a = kSel('dn-a'), sel = kSel('dn-k'), ks = sel === 'all' ? DYK : [sel], nw = single('dn-chart', 380);
  const tr = ks.map(k => ({ x: DYN[k][a].d, y: DYN[k][a].v, mode: 'lines', line: { color: RCOL[regOf(k)], width: 0.9 }, name: NM(k), hovertemplate: `${NM(k)} %{x}: %{y:.2f}%<extra></extra>` }));
  const ev = events();
  plot('dn-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`ΔCoVaR ${LVE(+a)}% (%)`), legend: LEG(nw, -0.1), margin: M0, shapes: ev.shapes }));
  out('dn-out', [T.dn(LV(+a), ks.map(k => { const s = dnStats(DYN[k][a]); return [NM(k), nf(s.mean, 2), nf(s.max, 2), DT(s.date_max), nf(s.y2017, 2), nf(DYN[k][a].b, 2)]; }), a !== '0.01')]);
}

// ================================================================= x7. MES from ten days (worked example; the numbers of the slide)
const TEN = { m: [-4.0, 1.2, -0.5, 2.1, -6.0, 0.3, -1.1, 0.8, -2.2, 1.5], b: [-6.5, 1.8, -0.2, 2.9, -9.0, 0.1, -2.0, 1.1, -3.1, 2.4] };
function mwBlock() {
  const a = +$('mw-a').value / 100, k = Math.ceil(10 * a - 1e-9), P = panels('mw-chart', 2, { h: 340, widths: [0.58, 0.42] }); label('mw-a').textContent = LV(a) + '%';
  const key = TEN.m, ord = key.map((_, i) => i).sort((i, j) => key[i] - key[j]), sel = ord.slice(0, k);
  const mes = -mean(sel.map(i => TEN.b[i])), es = -mean(sel.map(i => TEN.m[i])), days = TEN.m.map((_, i) => String(i + 1));
  const tr = [on2(P.ax[0], { x: TEN.m, y: TEN.b, mode: 'markers+text', text: days, textposition: 'top left', textfont: { size: 10.5, color: DARK }, marker: { color: BLUE, size: 9, line: { color: DARK, width: 1 } }, name: 'the ten days', cliponaxis: false, hovertemplate: 'day %{text}: market %{x}%, bank %{y}%<extra></extra>' }),
    on2(P.ax[0], { x: sel.map(i => TEN.m[i]), y: sel.map(i => TEN.b[i]), mode: 'markers', marker: { color: RED, size: 13, line: { color: DARK, width: 1 } }, name: `the market's ${k} worst days`, hoverinfo: 'skip' }),
    on2(P.ax[1], BAR([`MES ${LVE(a)}%`, `ES ${LVE(a)}% of the market`], [mes, es], [RED, AMBER], 'MES and ES', { text: [mes.toFixed(2), es.toFixed(2)], textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, showlegend: false, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }))];
  setAx(P, 0, AX('market return (%)', { range: [-7, 3] }), AX('bank return (%)', { range: [-10, 4] })); setAx(P, 1, AX('', { type: 'category' }), AX('%', { range: [0, 10] }));
  plot('mw-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  const L = sel.map(i => [i + 1, nf(TEN.m[i], 1), nf(TEN.b[i], 1)]);
  steps('mw-we', [T.mw1(LV(a), nf(a, 1), k, L), T.mw2(LV(a), L.map(x => x[2]), k, nf(mes, 2)), T.mw3(LV(a), L.map(x => x[1]), k, nf(es, 2)), T.mw4(nf(mes / es, 2))]);
  out('mw-out', [T.mw(LV(a), k, nf(mes, 2), nf(es, 2), Math.abs(a - 0.2) > 1e-9)]);
}

// ================================================================= 9. MES 5% of 13 banks
function msBlock() {
  const a = kSel('ms-a'), nw = single('ms-chart', 400), M = ALL.map(k => mesRow(k, +a)), v = M.map(m => m.mes);
  const tr = [BAR(ALL.map(NM), v, ALL.map(k => RCOL[regOf(k)]), `MES ${LVE(+a)}%`, { showlegend: false, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>',
    error_y: a === '0.05' ? { type: 'data', symmetric: false, array: ALL.map((k, i) => D.mes_ci[k][1] - v[i]), arrayminus: ALL.map((k, i) => v[i] - D.mes_ci[k][0]), color: DARK, thickness: 1, width: 3 } : undefined })];
  REGS.forEach(r => tr.push({ x: [null], y: [null], type: 'bar', marker: { color: RCOL[r] }, name: `${regLab(r).replace(' (TLV, BRD, since 2010)', '')} (market: ${INAME[D.index[r]]})` }));
  if (a === '0.05') tr.push({ x: [null], y: [null], mode: 'lines', line: { color: DARK }, name: '95% block-bootstrap interval' });
  plot('ms-chart', tr, layout({ xaxis: AX('', { tickangle: -30 }), yaxis: AX(`MES ${LVE(+a)}% (daily loss, %)`), legend: LEG(nw, -0.3), margin: M0 }));
  const o = ALL.map((k, i) => [k, v[i]]).sort((x, y) => y[1] - x[1]);
  out('ms-out', [T.ms(LV(+a), NM(o[0][0]), nf(o[0][1], 2), NM(o[1][0]), nf(o[1][1], 2), NM(o[12][0]), nf(o[12][1], 2), nf(mesRow('TLV', +a).mes, 2), nf(mesRow('BRD', +a).mes, 2), a === '0.05', a !== '0.05')]);
}

// ================================================================= x8. interpretation of MES and Delta-CoVaR (table)
function mtBlock() {
  const a = kSel('mt-a'), P = panels('mt-chart', 1, { h: 380 }), M = ALL.map(k => mesRow(k, +a)), s = spearman(M.map(m => m.mes), ALL.map(k => fit(k, '0.01').dcovar));
  const tr = REGS.map(reg => { const ks = D.regions[reg]; return on2(P.ax[0], { x: ks.map(k => mesRow(k, +a).beta * mesRow(k, +a).es_m), y: ks.map(k => mesRow(k, +a).mes), mode: 'markers+text', text: ks, textposition: 'top left', textfont: { size: 10, color: DARK }, marker: { color: RCOL[reg], size: 9 }, name: regLab(reg), cliponaxis: false, hovertemplate: '%{text}: β × ES %{x:.2f}%, MES %{y:.2f}%<extra></extra>' }); });
  const all = M.flatMap(m => [m.mes, m.beta * m.es_m]), lo = Math.min(...all) * 0.9, hi = Math.max(...all) * 1.05;
  tr.push(on2(P.ax[0], { x: [lo, hi], y: [lo, hi], mode: 'lines', line: { color: DARK, width: 1, dash: 'dot' }, name: 'MES = β × ES of the market', hoverinfo: 'skip' }));
  setAx(P, 0, AX(`OLS beta × ES ${LVE(+a)}% of the market (%)`), AX(`MES ${LVE(+a)}% (%)`));
  plot('mt-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('mt-chart'), -0.16), margin: M0 })));
  table('mt-tab', T.h.mt(LV(+a)), ALL.map((k, i) => { const m = M[i]; return [NM(k), fint(m.n), { v: nf(m.mes, 2), c: 'hi' }, a === '0.05' ? `[${nf(D.mes_ci[k][0], 2)}; ${nf(D.mes_ci[k][1], 2)}]` : '—', nf(m.beta, 2), nf(m.es_m, 2), nf(m.beta * m.es_m, 2), nf(m.mes2, 2), fint(m.n2), nf(100 * m.lrmes, 1), nf(m.Lstar, 1)]; }));
  out('mt-out', [T.mt(LV(+a), nf(mesRow('C', +a).beta, 2), nf(mesRow('C', +a).mes, 2), nf(mesRow('HSBA', +a).beta, 2), nf(mesRow('HSBA', +a).mes, 2), nf(s.r, 2), a !== '0.05')]);
}

// ================================================================= 10. did MES predict the 2008 losses?
function mcBlock() {
  const a = +kSel('mc-a'), X = mesCrisis(a), nw = single('mc-chart', 400), ks = Object.keys(X.res);
  const tr = ['US', 'EU'].map(reg => { const kk = ks.filter(k => X.res[k].region === reg); return { x: kk.map(k => X.res[k].mes), y: kk.map(k => X.res[k].ret), mode: 'markers+text', text: kk, textposition: 'top right', textfont: { size: 10.5, color: DARK }, marker: { color: RCOL[reg], size: 10 }, name: regLab(reg), cliponaxis: false, hovertemplate: '%{text}: MES %{x:.2f}%, return %{y:.1f}%<extra></extra>' }; });
  const xs = ks.map(k => X.res[k].mes), g = [Math.min(...xs), Math.max(...xs)];
  tr.push({ x: g, y: g.map(x => X.line[0] + X.line[1] * x), mode: 'lines', line: { color: ORANGE, width: 1.5, dash: 'dash' }, name: `least-squares line; Spearman correlation ${X.rho.toFixed(2)}`, hoverinfo: 'skip' });
  plot('mc-chart', tr, layout({ xaxis: AX(`MES ${LVE(a)}% before the crisis (%)`), yaxis: AX('return in the crisis (%)'), legend: LEG(nw, -0.14), margin: M0 }));
  let iw = 0; ks.forEach((k, i) => { if (X.res[k].ret < X.res[ks[iw]].ret) iw = i; });
  out('mc-out', [T.mc(LV(a), nf(X.rho, 2), nf(X.p, 2), X.n, NM(ks[iw]), nf(X.res[ks[iw]].ret, 0), X.res.JPM.n_pre, Math.abs(a - 0.05) > 1e-9)]);
}

// ================================================================= x9. from MES to LRMES (worked example)
function lrBlock() {
  const m = +$('lr-m').value, c = +kSel('lr-c'), P = panels('lr-chart', 2, { h: 340, widths: [0.45, 0.55] }), L = lrmes(m), ms = Array.from({ length: 81 }, (_, i) => i * 0.1); label('lr-m').textContent = nf(m, 2) + '%';
  const R = ALL.map(k => mesRow(k, 0.05, c));
  const tr = [on2(P.ax[0], { x: ms, y: ms.map(v => 100 * lrmes(v)), mode: 'lines', line: { color: BLUE, width: 2 }, name: 'LRMES = 1 − exp(−18 MES)', hovertemplate: 'MES %{x:.1f}%: LRMES %{y:.1f}%<extra></extra>' }),
    on2(P.ax[0], { x: [m], y: [100 * L], mode: 'markers+text', marker: { color: RED, size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`${(100 * L).toFixed(1)}%`], textposition: 'bottom right', textfont: { size: 11, color: DARK }, name: `MES = ${m}%`, cliponaxis: false, hoverinfo: 'skip' }),
    on2(P.ax[1], BAR(ALL, R.map(r => 100 * r.lrmes), ALL.map(k => RCOL[regOf(k)]), `LRMES of the banks (MES on days below ${c}%)`, { hovertemplate: '%{x}: %{y:.1f}%<extra></extra>', showlegend: false })),
    on2(P.ax[1], htr(-0.5, 12.5, 100 * L, RED, `LRMES for MES = ${m}%`))];
  setAx(P, 0, AX('MES on the days the market falls more than 2% (%)'), AX('LRMES (%)', { range: [0, 100] })); setAx(P, 1, AX('', { type: 'category', tickangle: -45 }), AX('LRMES (%)', { range: [0, 100] }));
  plot('lr-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.22), margin: M0 })));
  const lr = R.map(r => 100 * r.lrmes), mm = R.map(r => r.mes2);
  steps('lr-st', [T.lr1(nf(m, 2), nf(m / 100, 4)), T.lr2(nf(m / 100, 4), nf(18 * m / 100, 3), nf(Math.exp(-18 * m / 100), 3), nf(100 * L, 1)), T.lr3(nf(100 * L, 1)),
    T.lr4(String(c).replace('-', '−'), nf(Math.min(...mm), 2), nf(Math.max(...mm), 2), nf(Math.min(...lr), 0), nf(Math.max(...lr), 0), c !== -2)]);
  out('lr-out', [T.lr(nf(m, 2), nf(100 * L, 1), m !== 3)]);
}

// ================================================================= x10. SRISK: the capital shortfall in a crisis (worked example)
function swBlock() {
  const W = +$('sw-W').value, k = +kSel('sw-k'), src = kSel('sw-L'), Dd = 900, lr = src === 'ex' ? lrmes(3) : mesRow(src, 0.05).lrmes, P = panels('sw-chart', 2, { h: 340, widths: [0.55, 0.45] }); label('sw-W').textContent = W;
  const S = w => k * Dd - (1 - k) * (1 - lr) * w, ws = Array.from({ length: 91 }, (_, i) => 20 + 2 * i), need = k * Dd, left = (1 - k) * (1 - lr) * W;
  const tr = [on2(P.ax[0], { x: ws, y: ws.map(S), mode: 'lines', line: { color: BLUE, width: 2 }, name: `SRISK, D = 900, LRMES = ${(100 * lr).toFixed(1)}%`, hovertemplate: 'W = %{x}: SRISK %{y:.1f}<extra></extra>' }),
    on2(P.ax[0], htr(20, 200, 0, PURPLE, 'SRISK = 0', 'dot')),
    on2(P.ax[0], { x: [W], y: [S(W)], mode: 'markers', marker: { color: RED, size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, name: `W = ${W}`, hoverinfo: 'skip' }),
    on2(P.ax[1], BAR(['k D', '(1 − k)(1 − LRMES) W', 'SRISK'], [need, left, need - left], [BLUE, GREEN, need - left > 0 ? RED : GREEN], 'SRISK', { text: [need, left, need - left].map(v => v.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: DARK }, showlegend: false, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' }))];
  setAx(P, 0, AX('market value of equity W (billion)'), AX('SRISK (billion)')); setAx(P, 1, AX('', { type: 'category' }), AX('billion'));
  plot('sw-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  const Lv = (Dd + W) / W, Ls = lstar(lr, k);
  steps('sw-st', [T.sw1(fint(Dd), fint(W), nf(100 * lr, 1), LV(k), src), T.sw2(nf(k, 3), fint(Dd), nf(need, 1)), T.sw3(nf(1 - k, 2), nf(1 - lr, 3), fint(W), nf(left, 1)),
    T.sw4(nf(need, 1), nf(left, 1), nf(need - left, 1)), T.sw5(nf(Lv, 1), nf(Ls, 1), need - left > 0)]);
  out('sw-out', [T.sw(fint(W), nf(need - left, 1), nf(Lv, 1), nf(Ls, 1), W !== 100 || k !== 0.08 || src !== 'ex')]);
}

// ================================================================= 11. SRISK and leverage
function srBlock() {
  const k = +$('sr-k').value / 100, nw = single('sr-chart', 400), picks = NUM.srisk_pick, cols = [RED, AMBER, BLUE], Ls = Array.from({ length: 200 }, (_, i) => 1 + 24 * i / 199), tr = []; label('sr-k').textContent = LV(k) + '%';
  picks.forEach((b, i) => {
    const lr = mesRow(b, 0.05).lrmes, s = lstar(lr, k);
    tr.push({ x: Ls, y: Ls.map(L => 100 * sriskRatio(lr, L, k)), mode: 'lines', line: { color: cols[i], width: 1.8 }, name: `${NM(b)}: LRMES ${(100 * lr).toFixed(0)}%, break-even leverage ${s.toFixed(1)}`, hovertemplate: `${NM(b)}: L = %{x:.1f}, SRISK/W = %{y:.0f}%<extra></extra>` });
    tr.push({ x: [s], y: [0], mode: 'markers', marker: { color: cols[i], size: 9 }, showlegend: false, hoverinfo: 'skip' });
  });
  tr.push(htr(1, 25, 0, PURPLE, 'SRISK = 0', 'dot'));
  plot('sr-chart', tr, layout({ xaxis: AX('leverage L = (debt + market value of equity) / market value of equity'), yaxis: AX('SRISK / market value of equity (%)'), legend: LEG(nw, -0.16), margin: M0 }));
  const all = ALL.map(b => lstar(mesRow(b, 0.05).lrmes, k));
  out('sr-out', [T.sr(LV(k), picks.map(b => [NM(b), nf(lstar(mesRow(b, 0.05).lrmes, k), 1)]), nf(Math.min(...all), 1), nf(Math.max(...all), 1), Math.abs(k - 0.08) > 1e-9)]);
}

// ================================================================= 12. Granger networks through time
function dgBlock() {
  const w = +$('dg-w').value, lv = +kSel('dg-l'), S = rollDGC(w, lv), nw = single('dg-chart', 370); label('dg-w').textContent = w;
  let im = 0, in_ = 0; S.v.forEach((x, i) => { if (x > S.v[im]) im = i; if (x < S.v[in_]) in_ = i; });
  const tr = [{ x: S.d, y: S.v, mode: 'lines', line: { color: BLUE, width: 1.5 }, name: `degree of Granger causality (DGC), ${w}-month windows`, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    htr(S.d[0], S.d[S.d.length - 1], 100 * lv, RED, `${LVE(lv)}%: share of links expected by chance at the ${LVE(lv)}% level`)];
  const ev = events(S.d[0]);
  plot('dg-chart', tr, layout({ xaxis: AX(''), yaxis: AX('significant links (%)'), legend: LEG(nw, -0.12), margin: M0, shapes: ev.shapes, annotations: ev.annotations }));
  out('dg-out', [T.dg(w, LV(lv), DT(S.d[0]), nf(S.v[im], 0), DT(S.d[im]), nf(mean(S.v), 1), nf(S.v[in_], 0), DT(S.d[in_]), nf(S.v[S.v.length - 1], 0), monthly().m.length, w !== 36 || lv !== 0.05)]);
}

// ================================================================= 13. two networks: 2007-2009 and 2017-2019
function gnBlock() {
  const sh = +$('gn-s').value, P = panels('gn-chart', 2, { h: 430 }), M = monthly(), n = INTL.length, tr = [], an = [], res = []; label('gn-s').textContent = sh;
  const pos = INTL.map((_, i) => { const t = Math.PI / 2 - 2 * Math.PI * i / n; return [Math.cos(t), Math.sin(t)]; });
  ['2009-12', '2019-12'].forEach((e0, p) => {
    const j = M.m.indexOf(e0) + 1 + sh, G = grangerNet(j, 36, 0.05); res.push(G);
    const xs = [], ys = [];
    INTL.forEach((a, i) => INTL.forEach((b, k) => { if (G.A[i][k]) { xs.push(0.9 * pos[i][0], 0.9 * pos[k][0], null); ys.push(0.9 * pos[i][1], 0.9 * pos[k][1], null);
      an.push({ x: 0.9 * pos[k][0], y: 0.9 * pos[k][1], ax: 0.9 * pos[i][0], ay: 0.9 * pos[i][1], xref: P.ax[p].x, yref: P.ax[p].y, axref: P.ax[p].x, ayref: P.ax[p].y, text: '', showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 1, arrowcolor: RCOL[regOf(a)], standoff: 12 }); } }));
    tr.push(on2(P.ax[p], { x: xs, y: ys, mode: 'lines', line: { color: 'rgba(31,42,68,0.25)', width: 0.8 }, name: 'significant Granger link (5%)', showlegend: p === 0, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[p], { x: pos.map(q => q[0]), y: pos.map(q => q[1]), mode: 'markers', marker: { color: INTL.map(k => RCOL[regOf(k)]), size: 20 }, showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[p], { x: pos.map(q => 1.24 * q[0]), y: pos.map(q => 1.24 * q[1]), mode: 'text', text: INTL.map((k, i) => `${k} (${sum(G.A[i])})`), textfont: { size: 10, color: DARK }, showlegend: false, hoverinfo: 'skip' }));
    setAx(P, p, { range: [-1.5, 1.5], visible: false, scaleanchor: P.ax[p].y }, { range: [-1.45, 1.45], visible: false });
    const MN = m => ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m.slice(5, 7) - 1] + ' ' + m.slice(0, 4);
    an.push(ptitle(P, p, `${MN(G.from)} – ${MN(G.to)}: ${G.links} links, DGC ${G.dgc.toFixed(0)}%`));
  });
  ['US', 'EU'].forEach(r => tr.push({ x: [null], y: [null], mode: 'markers', marker: { color: RCOL[r], size: 10 }, name: `${regLab(r)} (arrows: links from this bank)` }));
  plot('gn-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.02 : -0.06), margin: { l: 10, r: 10, t: 26, b: 20 }, annotations: an })));
  out('gn-out', [T.gn(String(sh).replace('-', '−'), res.map(G => [G.from, G.to, G.links, nf(G.dgc, 1)]), sh !== 0)]);
}

// ================================================================= x11. Diebold-Yilmaz connectedness (worked example)
function dwBlock() {
  const H = +$('dw-h').value, k = kSel('dw-k'), i = INTL.indexOf(k), X = dyTable(H), P = panels('dw-chart', 2, { h: 340 }); label('dw-h').textContent = H;
  const row = X.table[i], col = X.table.map(r => r[i]), cs = INTL.map(b => (b === k ? ORANGE : RCOL[regOf(b)]));
  const tr = [on2(P.ax[0], BAR(INTL, row, cs, `row ${k}: shares of its forecast error variance`, { hovertemplate: `from %{x} to ${k}: %{y:.1f}%<extra></extra>` })),
    on2(P.ax[1], BAR(INTL, col, cs, `column ${k}: its shares in the others`, { marker: { color: cs, pattern: { shape: '/' } }, hovertemplate: `from ${k} to %{x}: %{y:.1f}%<extra></extra>` }))];
  setAx(P, 0, AX('', { type: 'category', tickangle: -45 }), AX('d_ij (%)', { range: [0, 30] })); setAx(P, 1, AX('', { type: 'category', tickangle: -45 }), AX('d_ji (%)', { range: [0, 30] }));
  plot('dw-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.26), margin: M0 })));
  const top = INTL.map((b, j) => [b, row[j]]).filter(x => x[0] !== k).sort((x, y) => y[1] - x[1]).slice(0, 3);
  steps('dw-we', [T.dw1(NM(k), H, fint(X.n), nf(row[i], 1), top.map(x => [x[0], nf(x[1], 1)])), T.dw2(nf(row[i], 1), nf(X.from[i], 1)), T.dw3(nf(X.to[i], 1), nf(X.from[i], 1), nf(X.net[i], 1)),
    T.dw4(nf(sum(X.from), 1), nf(X.total, 1))]);
  out('dw-out', [T.dw(NM(k), H, nf(X.from[i], 1), nf(X.to[i], 1), nf(X.net[i], 1), nf(X.total, 1), H !== 10)]);
}

// ================================================================= 14. the connectedness table
function dtBlock() {
  const H = +$('dt-h').value, X = dyTable(H), nw = single('dt-chart', 480); label('dt-h').textContent = H;
  const ylab = [...INTL.map((k, i) => `${k} (FROM ${X.from[i].toFixed(0)})`), 'TO others'], z = [...X.table, X.to];
  const zc = [...X.table.map(r => r.slice()), X.to.map(() => null)], vmax = Math.ceil(Math.max(...X.table.flat()) / 10) * 10;
  const tr = [{ type: 'heatmap', x: INTL, y: ylab, z: zc, zmin: 0, zmax: vmax, colorscale: [[0, '#FFFFFF'], [0.5, '#7FA3D6'], [1, '#1A3A6E']], showscale: false, hovertemplate: 'from %{x} to %{y}: %{z:.1f}%<extra></extra>' },
    { x: z.flatMap(r => INTL), y: z.flatMap((r, i) => INTL.map(() => ylab[i])), mode: 'text', text: z.flatMap(r => r.map(v => v.toFixed(0))), textfont: { size: 10, color: z.flatMap((r, i) => r.map(v => (i === INTL.length ? RED : v > 0.55 * vmax ? '#FFFFFF' : DARK))) }, showlegend: false, hoverinfo: 'skip' }];
  plot('dt-chart', tr, layout({ xaxis: AX(`shock from (column) to (row), % of the forecast error variance; total connectedness ${X.total.toFixed(1)}%`, { side: 'bottom', type: 'category' }), yaxis: AX('', { autorange: 'reversed', type: 'category' }), margin: { l: 110, r: 12, t: 14, b: 50 } }));
  let it = 0, il = 0; X.to.forEach((v, i) => { if (v > X.to[it]) it = i; if (X.net[i] < X.net[il]) il = i; });
  out('dt-out', [T.dt(H, fint(X.n), nf(X.total, 1), NM(INTL[it]), nf(X.to[it], 1), NM(INTL[il]), nf(X.net[il], 1), nf(Math.min(...X.from), 0), nf(Math.max(...X.from), 0), H !== 10)]);
}

// ================================================================= x12. interpretation of the connectedness table (table)
const DZP = { full: ['2005-01-01', D.end], p1: ['2005-01-01', '2009-12-31'], p2: ['2010-01-01', '2019-12-31'], p3: ['2020-01-01', D.end] };
function dzBlock() {
  const p = kSel('dz-p'), [a, b] = DZP[p], X = dyTable(10, a, b), P = panels('dz-chart', 1, { h: 360 });
  const tr = [on2(P.ax[0], BAR(INTL, X.net, INTL.map(k => RCOL[regOf(k)]), 'NET = TO − FROM', { text: X.net.map(v => v.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, showlegend: false, hovertemplate: '%{x}: NET %{y:.1f}<extra></extra>' }))];
  ['US', 'EU'].forEach(r => tr.push(on2(P.ax[0], { x: [null], y: [null], type: 'bar', marker: { color: RCOL[r] }, name: regLab(r) })));
  const m = Math.max(...X.net.map(Math.abs)) * 1.25;
  setAx(P, 0, AX('', { type: 'category' }), AX('NET (percentage points)', { range: [-m, m] }));
  plot('dz-chart', tr, layout(Object.assign(P.L, { legend: LEG(narrow('dz-chart'), -0.14), margin: M0 })));
  table('dz-tab', T.h.dz, INTL.map((k, i) => [NM(k), nf(X.table[i][i], 1), nf(X.from[i], 1), nf(X.to[i], 1), { v: sgn(X.net[i], 1), c: X.net[i] >= 0 ? 'hi' : 'bad' }]));
  const usNet = mean(US.map(k => X.net[INTL.indexOf(k)])), euNet = mean(EU.map(k => X.net[INTL.indexOf(k)]));
  out('dz-out', [T.dz(DT(X.d0), DT(X.d1), fint(X.n), nf(X.total, 1), nf(Math.min(...X.from), 0), nf(Math.max(...X.from), 0), sgn(usNet, 1), sgn(euNet, 1), p !== 'full')]);
}

// ================================================================= 15. total connectedness through time
const DR = {}; Object.entries(D.dyroll).forEach(([w, s]) => { DR[w] = s; });
function drStats(S) {
  const per = (a, b) => { const [i0, i1] = between(S.d, a, b); return mean(S.v.slice(i0, i1)); };
  let im = 0, in_ = 0; S.v.forEach((x, i) => { if (x > S.v[im]) im = i; if (x < S.v[in_]) in_ = i; });
  const [j0, j1] = between(S.d, '2016-01-01', '2016-12-31'); let m16 = j0; for (let j = j0; j < j1; j++) if (S.v[j] > S.v[m16]) m16 = j;
  return { min: S.v[in_], date_min: S.d[in_], max: S.v[im], date_max: S.d[im], y2008: per('2008-10-01', '2009-03-31'), y2016: per('2016-01-01', '2016-12-31'), max2016: S.v[m16], date2016: S.d[m16], last: S.v[S.v.length - 1], mean: mean(S.v) };
}
function drBlock() {
  const w = kSel('dr-w'), S = DR[w], st = drStats(S), nw = single('dr-chart', 370);
  const tr = [{ x: S.d, y: S.v, mode: 'lines', line: { color: BLUE, width: 1.3 }, name: `total connectedness, ${w}-day windows (%)`, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: [st.date2016], y: [st.max2016], mode: 'markers', marker: { color: ORANGE, size: 9, line: { color: DARK, width: 1 } }, name: `2016 maximum: ${st.max2016.toFixed(1)}%`, hoverinfo: 'skip' },
    { x: [st.date_max], y: [st.max], mode: 'markers', marker: { color: RED, size: 9, line: { color: DARK, width: 1 } }, name: `overall maximum: ${st.max.toFixed(1)}%`, hoverinfo: 'skip' }];
  const ev = events(S.d[0]);
  plot('dr-chart', tr, layout({ xaxis: AX(''), yaxis: AX('total connectedness (%)'), legend: LEG(nw, -0.12), margin: M0, shapes: ev.shapes, annotations: ev.annotations }));
  out('dr-out', [T.dr(w, nf(st.min, 0), DT(st.date_min), nf(st.max, 0), DT(st.date_max), nf(st.y2008, 0), nf(st.y2016, 0), nf(st.max2016, 0), DT(st.date2016), nf(st.last, 0), w !== '200')]);
}

// ================================================================= x13. Basel III capital buffers (worked example; the numbers of the slide)
function bsBlock() {
  const c = +$('bs-c').value, g = +kSel('bs-g'), P = panels('bs-chart', 2, { h: 340, widths: [0.42, 0.58] }), tot = 4.5 + 2.5 + c + g, cs = Array.from({ length: 11 }, (_, i) => 0.25 * i); label('bs-c').textContent = nf(c, 2) + '%';
  const parts = [['CET1 minimum', 4.5, BLUE], ['conservation buffer', 2.5, TEAL], ['countercyclical buffer', c, AMBER], ['G-SIB surcharge', g, RED]];
  const tr = parts.map(([n, v, col]) => on2(P.ax[0], BAR(['CET1 requirement'], [v], col, `${n}: ${v}%`, { hovertemplate: `${n}: %{y:.2f}%<extra></extra>` })));
  [[0, DARK, 'no G-SIB surcharge'], [1, GREEN, 'G-SIB surcharge 1%'], [3.5, RED, 'G-SIB surcharge 3.5%']].forEach(([gg, col, n]) => tr.push(on2(P.ax[1], { x: cs, y: cs.map(v => 7 + v + gg), mode: 'lines', line: { color: col, width: 1.8, dash: gg === g ? 'solid' : 'dot' }, name: n, hovertemplate: 'CCyB %{x:.2f}%: %{y:.2f}%<extra></extra>' })));
  tr.push(on2(P.ax[1], { x: [c], y: [tot], mode: 'markers+text', marker: { color: ORANGE, size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, text: [`${tot.toFixed(2)}%`], textposition: 'top left', textfont: { size: 11, color: DARK }, name: 'this bank', cliponaxis: false, hoverinfo: 'skip' }));
  setAx(P, 0, AX('', { type: 'category' }), AX('% of risk-weighted assets', { range: [0, 14] })); setAx(P, 1, AX('countercyclical buffer (%)'), AX('CET1 requirement (%)', { range: [0, 14] }));
  plot('bs-chart', tr, layout(Object.assign(P.L, { barmode: 'stack', legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  steps('bs-st', [T.bs1(), T.bs2(nf(c, 2)), T.bs3(nf(g, 1)), T.bs4(nf(c, 2), nf(g, 1), nf(tot, 2)), T.bs5(nf(tot, 2))]);
  out('bs-out', [T.bs(nf(c, 2), nf(g, 1), nf(tot, 2))]);
}

// ================================================================= wiring
const PFX = { indices: 'bi', episodes: 'ep', rcorr: 'rc', 'corr-tab': 'ct', 'fr-we': 'fr', 'tail-tab': 'tt', tscatter: 'ts', 'covar-we': 'cw', covarqr: 'cq', dcovar: 'dc', 'dcovar-tab': 'dx',
  'tlv-we': 'tb', vardcovar: 'vd', dyn: 'dn', 'mes-we': 'mw', mes: 'ms', 'mes-tab': 'mt', mescrisis: 'mc', 'lrmes-we': 'lr', 'srisk-we': 'sw', srisk: 'sr', dgc: 'dg', gnet: 'gn',
  'dy-we': 'dw', dytab: 'dt', 'dy-tab': 'dz', dyroll: 'dr', 'basel-we': 'bs' };
const BLOCKS = { biBlock, epBlock, rcBlock, ctBlock, frBlock, ttBlock, tsBlock, cwBlock, cqBlock, dcBlock, dxBlock, tbBlock, vdBlock, dnBlock, mwBlock, msBlock, mtBlock, mcBlock, lrBlock,
  swBlock, srBlock, dgBlock, gnBlock, dwBlock, dtBlock, dzBlock, drBlock, bsBlock };
const PLAYERS = {};
function placeExtras() {                // web lecture: the table blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
const YR = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => String(a + i));
const AL = ['0.01', '0.025', '0.05', '0.1'];
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const PLy = (btn, id, vals, ms_) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms_));
  on(['bi-y'], biBlock); PLy('bi-play', 'bi-y', ['2005', '2008', '2010', '2013', '2016', '2020', '2005'], 1400);
  on(['ep-d'], epBlock); PLy('ep-play', 'ep-d', [0, -30, -20, -10, 10, 20, 30, 0], 1200);
  on(['rc-w'], rcBlock); PLy('rc-play', 'rc-w', [120, 60, 90, 180, 250, 120], 1300);
  on(['ct-w', 'ct-y'], ctBlock); PLy('ct-play', 'ct-y', [...YR(2005, 2026), '2017'], 700);
  on(['fr-e', 'fr-s'], frBlock); PLy('fr-play', 'fr-e', ['lehman', 'euro', 'covid', 'svb', 'lehman'], 1500);
  on(['tt-q'], ttBlock); PLy('tt-play', 'tt-q', [5, 0.5, 1, 2, 3, 7.5, 10, 15, 20, 5], 900);
  on(['ts-q'], tsBlock); PLy('ts-play', 'ts-q', [5, 1, 2, 3, 7.5, 10, 15, 5], 1100);
  on(['cw-a', 'cw-k'], cwBlock); PLy('cw-play', 'cw-a', [...AL, '0.01'], 1500);
  on(['cq-a', 'cq-k'], cqBlock); PLy('cq-play', 'cq-a', [...AL, '0.01'], 1500);
  on(['dc-a'], dcBlock); PLy('dc-play', 'dc-a', [...AL, '0.01'], 1500);
  on(['dx-a'], dxBlock); PLy('dx-play', 'dx-a', ['0.05', '0.025', '0.1', '0.05'], 1500);
  on(['tb-a', 'tb-k'], tbBlock); PLy('tb-play', 'tb-a', [...AL, '0.01'], 1500);
  on(['vd-a'], vdBlock); PLy('vd-play', 'vd-a', [...AL, '0.01'], 1500);
  on(['dn-a', 'dn-k'], dnBlock); PLy('dn-play', 'dn-a', ['0.01', '0.025', '0.05', '0.01'], 1500);
  on(['mw-a'], mwBlock); PLy('mw-play', 'mw-a', [20, 10, 30, 40, 50, 20], 1100);
  on(['ms-a'], msBlock); PLy('ms-play', 'ms-a', ['0.05', '0.01', '0.025', '0.1', '0.05'], 1500);
  on(['mt-a'], mtBlock); PLy('mt-play', 'mt-a', ['0.05', '0.01', '0.025', '0.1', '0.05'], 1500);
  on(['mc-a'], mcBlock); PLy('mc-play', 'mc-a', ['0.05', '0.01', '0.025', '0.1', '0.05'], 1500);
  on(['lr-m', 'lr-c'], lrBlock); PLy('lr-play', 'lr-m', [3, 0.5, 1, 2, 4, 5, 6, 3], 900);
  on(['sw-W', 'sw-k', 'sw-L'], swBlock); PLy('sw-play', 'sw-W', [100, 20, 40, 60, 80, 120, 150, 200, 100], 900);
  on(['sr-k'], srBlock); PLy('sr-play', 'sr-k', [8, 5.5, 4, 6, 10, 12, 8], 1100);
  on(['dg-w', 'dg-l'], dgBlock); PLy('dg-play', 'dg-w', [36, 24, 30, 42, 48, 60, 36], 1300);
  on(['gn-s'], gnBlock); PLy('gn-play', 'gn-s', [0, -12, -6, 6, 12, 18, 24, 0], 1300);
  on(['dw-h', 'dw-k'], dwBlock); PLy('dw-play', 'dw-h', [10, 1, 2, 3, 5, 15, 20, 30, 10], 1000);
  on(['dt-h'], dtBlock); PLy('dt-play', 'dt-h', [10, 1, 2, 3, 5, 15, 20, 30, 10], 1000);
  on(['dz-p'], dzBlock); PLy('dz-play', 'dz-p', ['full', 'p1', 'p2', 'p3', 'full'], 1500);
  on(['dr-w'], drBlock); PLy('dr-play', 'dr-w', ['200', '100', '150', '250', '300', '200'], 1400);
  on(['bs-c', 'bs-g'], bsBlock); PLy('bs-play', 'bs-c', [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 0], 900);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH15 = { joinR, regionR, intlR, portfolio, pfOf, rollCorr, corrStats, tailPair, lam, fit, mesOf, mesThr, mesRow, mesCrisis, monthly, grangerP, grangerNet, rollDGC, varFit, gfevd, dyTable,
  frTest, episode, dnStats, DYN, drStats, DR, spearman, lrmes, lstar, quantile, init, PFX, blocks: BLOCKS };
if (window.CH15_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
