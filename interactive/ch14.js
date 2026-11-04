// Chapter 14 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch14_data.js.
(function () {
'use strict';
const D = window.CH14_DATA;
const { C, nf, $, AX, layout, plot, out, Player, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH14_TEXT;
const NUM = D.N, COL = D.colors, NAME = D.name, END = D.end;
const DARK = C.navy, GREEN = COL.Forest, RED = COL.IDAred, BLUE = COL.MainBlue, ORANGE = COL.Orange, AMBER = COL.Amber, PURPLE = COL.Purple, TEAL = COL.Teal, CRIMSON = COL.Crimson;

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
const phi = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
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
function ibeta(a, b, x) {            // regularised incomplete beta I_x(a, b)
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
function gammaQ(a, x) {              // regularised upper incomplete gamma Q(a, x)
  if (x <= 0) return 1;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a, s = 1 / a, del = s;
    for (let n = 0; n < 2000; n++) { ap++; del *= x / ap; s += del; if (Math.abs(del) < Math.abs(s) * 1e-17) break; }
    return 1 - s * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 2000; i++) {
    const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}
function bisect(f, lo, hi) { let flo = f(lo); for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2, fm = f(m); if ((fm > 0) === (flo > 0)) { lo = m; flo = fm; } else hi = m; if (hi - lo < 1e-14 * Math.max(1, Math.abs(m))) break; } return (lo + hi) / 2; }
const chi2sf = (x, k) => gammaQ(k / 2, x / 2);
const chi2ppf = (p, k) => bisect(x => 1 - chi2sf(x, k) - p, 0, 2000);
const t2sf = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));                       // two-sided p-value of a t statistic
const tcdf = (t, df) => { const p = 0.5 * ibeta(df / 2, 0.5, df / (df + t * t)); return t > 0 ? 1 - p : p; };
const tppf = (p, df) => bisect(x => tcdf(x, df) - p, -1e4, 1e4);
const tpdf = (x, nu) => Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / Math.sqrt(nu * Math.PI) * Math.pow(1 + x * x / nu, -(nu + 1) / 2);
const fsf = (F, d1, d2) => ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F));
const stdtq = (nu, a) => tppf(a, nu) * Math.sqrt((nu - 2) / nu);                     // a-quantile of the unit-variance Student-t
const stdtes = (nu, a) => { const q = tppf(a, nu); return Math.sqrt((nu - 2) / nu) * tpdf(q, nu) / a * (nu + q * q) / (nu - 1); };

const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
const sdv = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / (a.length - 1)); };
const vari = a => sdv(a) ** 2;
const median = a => { const b = Float64Array.from(a).sort(); const n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const quantile = (a, q) => { const b = Float64Array.from(a).sort(), h = (b.length - 1) * q, f = Math.floor(h); return f + 1 < b.length ? b[f] + (h - f) * (b[f + 1] - b[f]) : b[f]; };   // pandas, linear
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; saa += u * u; sbb += v * v; } return sab / Math.sqrt(saa * sbb); };
const cov = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb); return s / (a.length - 1); };
function moments(x) { const m = mean(x); let m2 = 0, m3 = 0, m4 = 0; for (const v of x) { const e = v - m, e2 = e * e; m2 += e2; m3 += e2 * e; m4 += e2 * e2; } const n = x.length; m2 /= n; m3 /= n; m4 /= n; return { skew: m3 / m2 ** 1.5, exkurt: m4 / (m2 * m2) - 3 }; }

// ================================================================= data: closes on their own calendars, returns as in sfm_data.log_returns
const DAY = 864e5;
const isoOf = ms => new Date(ms).toISOString().slice(0, 10);
const ms = s => Date.parse(s + 'T00:00:00Z');
const addDays = (s, k) => isoOf(ms(s) + k * DAY);
const yearOf = s => +s.slice(0, 4);
const dow = s => (new Date(ms(s)).getUTCDay() + 6) % 7;                               // Monday = 0, as in pandas
function dated(o) { const b = ms(o.t0); return o.t.map(d => isoOf(b + d * DAY)); }
const PX = {};
Object.entries(D.px).forEach(([k, s]) => { PX[k] = { d: dated(s), p: s.v }; });
function closes(k, a, b) {
  const S = PX[k], d = [], p = [];
  for (let i = 0; i < S.d.length; i++) if ((!a || S.d[i] >= a) && (!b || S.d[i] <= b)) { d.push(S.d[i]); p.push(S.p[i]); }
  return { d, p };
}
const logr = p => { const r = new Array(p.length - 1); for (let i = 1; i < p.length; i++) r[i - 1] = 100 * (Math.log(p[i]) - Math.log(p[i - 1])); return r; };
const RC = {};
function rets(k, a, b) { const key = k + a + b; if (RC[key]) return RC[key]; const c = closes(k, a, b); return (RC[key] = { d: c.d.slice(1), r: logr(c.p) }); }
const JC = {};
function joint(a, b, s, e) {         // prices joined on common days first, then differenced (generate_all_charts.joint_returns)
  const key = [a, b, s, e].join('|'); if (JC[key]) return JC[key];
  const A = closes(a, s, e), B = closes(b, s, e), m = new Map(B.d.map((d, i) => [d, B.p[i]])), d = [], pa = [], pb = [];
  A.d.forEach((x, i) => { if (m.has(x)) { d.push(x); pa.push(A.p[i]); pb.push(m.get(x)); } });
  return (JC[key] = { d: d.slice(1), a: logr(pa), b: logr(pb), pd: d, pa, pb });
}
const ppy = d => d.length / ((ms(d[d.length - 1]) - ms(d[0])) / DAY / 365.25);   // periods_per_year
const START = { btc: '2014-09-17', eth: '2016-01-01', sol: '2020-04-11', sp500: '2014-09-17', gold: '2014-09-17' };
const ASSETS = ['btc', 'eth', 'sol', 'sp500', 'gold'];

// ================================================================= the chapter's statistics (generate_all_charts.py)
function hill(x, k) { const s = Float64Array.from(x).sort().reverse(); let a = 0; for (let i = 0; i < k; i++) a += Math.log(s[i] / s[k]); return 1 / (a / k); }
function hillCurve(x, ks) {          // alpha_hat(k) for several k from one sort
  const s = Float64Array.from(x).sort().reverse(), cl = new Float64Array(s.length + 1);
  for (let i = 0; i < s.length; i++) cl[i + 1] = cl[i] + Math.log(s[i]);
  return ks.map(k => 1 / (cl[k] / k - Math.log(s[k])));
}
const DESC = {};
function describe(k, start, end) {
  const key = k + (start || '') + (end || ''); if (DESC[key]) return DESC[key];
  const R = rets(k, start || START[k], end), x = R.r, n = x.length, A = ppy(R.d), kk = Math.floor(0.025 * n), m = mean(x), s = sdv(x), mo = moments(x);
  let imin = 0, imax = 0; x.forEach((v, i) => { if (v < x[imin]) imin = i; if (v > x[imax]) imax = i; });
  const al = hill(x.filter(v => v < 0).map(v => -v), kk), ar = hill(x.filter(v => v > 0), kk);
  return (DESC[key] = { n, first: R.d[0], ppy: A, mean: m, sd: s, skew: mo.skew, exkurt: mo.exkurt, min: x[imin], min_date: R.d[imin], max: x[imax], max_date: R.d[imax],
    ann_mean: A * m, ann_vol: Math.sqrt(A) * s, ann_vol_252: Math.sqrt(252) * s, ann_mean_252: 252 * m, share5: 100 * mean(x.map(v => +(Math.abs(v) > 5))), k: kk,
    hill_left: al, hill_right: ar, hill_left_lo: al * (1 - 1.96 / Math.sqrt(kk)), hill_left_hi: al * (1 + 1.96 / Math.sqrt(kk)),
    hill_right_lo: ar * (1 - 1.96 / Math.sqrt(kk)), hill_right_hi: ar * (1 + 1.96 / Math.sqrt(kk)), jb: n / 6 * (mo.skew ** 2 + mo.exkurt ** 2 / 4), x, d: R.d });
}
function rollSd(x, w) { const o = new Array(x.length).fill(NaN); for (let i = w - 1; i < x.length; i++) o[i] = sdv(x.slice(i - w + 1, i + 1)); return o; }
function rvol(w, annC) {             // fig_rolling_vol: about one month, 30 x sqrt(365) for crypto, 21 x sqrt(252) for the S&P 500
  const ws = Math.round(w * 252 / 365), res = {};
  [['btc', w, annC], ['eth', w, annC], ['sp500', ws, 252]].forEach(([k, ww, A]) => {
    const R = rets(k, '2016-01-01'), v = rollSd(R.r, ww).map(z => z * Math.sqrt(A)), idx = v.map((_, i) => i).filter(i => !isNaN(v[i])), vv = idx.map(i => v[i]);
    let imax = 0, imin = 0; vv.forEach((z, i) => { if (z > vv[imax]) imax = i; if (z < vv[imin]) imin = i; });
    res[k] = { d: R.d, v, w: ww, median: median(vv), max: vv[imax], max_date: R.d[idx[imax]], min: vv[imin], min_date: R.d[idx[imin]], last: vv[vv.length - 1] };
  });
  const sp = new Map(res.sp500.d.map((d, i) => [d, res.sp500.v[i]])), ratio = [];
  let below = 0;
  res.btc.d.forEach((d, i) => { const b = res.btc.v[i], s = sp.get(d); if (s !== undefined && !isNaN(b) && !isNaN(s)) { ratio.push(b / s); below += +(b < s); } });
  res.ratio_median = median(ratio); res.share_btc_below_sp = 100 * below / ratio.length; res.ws = ws;
  return res;
}
function hac(y, w, L) {              // OLS of y on a constant and w, Newey-West (Bartlett) standard errors with L lags, no small-sample correction
  const n = y.length; let sw = 0, sww = 0, sy = 0, swy = 0;
  for (let i = 0; i < n; i++) { sw += w[i]; sww += w[i] * w[i]; sy += y[i]; swy += w[i] * y[i]; }
  const det = n * sww - sw * sw, Xi = [[sww / det, -sw / det], [-sw / det, n / det]], b0 = (sww * sy - sw * swy) / det, b1 = (n * swy - sw * sy) / det;
  const u = y.map((v, i) => [v - b0 - b1 * w[i], (v - b0 - b1 * w[i]) * w[i]]);
  const S = [[0, 0], [0, 0]];
  for (let l = 0; l <= L; l++) {
    const k = l ? 1 - l / (L + 1) : 1, G = [[0, 0], [0, 0]];
    for (let i = l; i < n; i++) for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) G[a][c] += u[i][a] * u[i - l][c];
    for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) S[a][c] += l ? k * (G[a][c] + G[c][a]) : G[a][c];
  }
  const M = (A, B) => A.map((r, i) => B[0].map((_, j) => r[0] * B[0][j] + r[1] * B[1][j])), V = M(M(Xi, S), Xi);
  const se = Math.sqrt(V[1][1]), t = b1 / se;
  return { a: b0, b: b1, se, t, p: 2 * Phi(-Math.abs(t)) };
}
function levene(groups) {            // Brown-Forsythe: Levene's test around the median
  const z = groups.map(g => { const m = median(g); return g.map(v => Math.abs(v - m)); }), N = sum(z.map(g => g.length)), k = z.length;
  const zm = z.map(mean), zz = sum(z.map(g => sum(g))) / N;
  const num = sum(z.map((g, i) => g.length * (zm[i] - zz) ** 2)), den = sum(z.map((g, i) => sum(g.map(v => (v - zm[i]) ** 2))));
  const W = (N - k) / (k - 1) * num / den;
  return { stat: W, p: fsf(W, k - 1, N - k) };
}
const WD = { '2014-2019': ['2014-09-17', '2019-12-31'], '2020-2023': ['2020-01-01', '2023-12-31'], '2024-2026': ['2024-01-01', END] };
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const WDC = {};
function weekday(a, b) {
  const key = a + b; if (WDC[key]) return WDC[key];
  const R = rets('btc', a, b), dw = R.d.map(dow), w = dw.map(v => +(v >= 5)), by = DAYS.map((_, i) => R.r.filter((_, j) => dw[j] === i));
  const reg = hac(R.r, w, 10), we = R.r.filter((_, j) => w[j]), wd = R.r.filter((_, j) => !w[j]), bf = levene([we, wd]);
  return (WDC[key] = { n: R.r.length, mean: by.map(mean), sd: by.map(sdv), b: reg.b, t: reg.t, p: reg.p, bf: bf.stat, p_bf: bf.p, var_ratio: vari(we) / vari(wd), sd_we: sdv(we), sd_wd: sdv(wd) });
}
function acf(x, L) { const m = mean(x), e = x.map(v => v - m), s = sum(e.map(v => v * v)); return Array.from({ length: L }, (_, j) => { const k = j + 1; let c = 0; for (let i = k; i < e.length; i++) c += e[i] * e[i - k]; return c / s; }); }
function ljung(x, m, rho) { const n = x.length; rho = rho || acf(x, m); let s = 0; for (let k = 1; k <= m; k++) s += rho[k - 1] ** 2 / (n - k); const q = n * (n + 2) * s; return { q, p: chi2sf(q, m), s, n }; }
function vratio(x, q) {              // Lo and MacKinlay (1988): VR(q), Z(q) and the robust Z*(q)
  const n = x.length, mu = mean(x), e = x.map(v => v - mu); let se2 = 0; for (const v of e) se2 += v * v;
  const varA = se2 / (n - 1); let s = 0, run = 0;
  for (let i = 0; i < n; i++) { run += x[i]; if (i >= q) run -= x[i - q]; if (i >= q - 1) s += (run - q * mu) ** 2; }
  const m = q * (n - q + 1) * (1 - q / n), vr = (s / m) / varA, z = (vr - 1) / Math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n));
  let theta = 0; const den = se2 * se2;
  for (let j = 1; j < q; j++) { let c = 0; for (let i = j; i < n; i++) c += e[i] * e[i] * e[i - j] * e[i - j]; theta += (2 * (q - j) / q) ** 2 * n * c / den; }
  const zs = (vr - 1) / Math.sqrt(theta / n);
  return { q, vr, z, zs, p: 2 * Phi(-Math.abs(zs)) };
}
function rollVR(k, q, win, step) {
  const R = rets(k, START[k]), rows = [];
  for (let e = win; e <= R.r.length; e += step) { const v = vratio(R.r.slice(e - win, e), q); rows.push({ d: R.d[e - 1], vr: v.vr, z: v.z, zs: v.zs }); }
  return rows;
}
const SUB = [['2014-09-17', '2017-12-31'], ['2018-01-01', '2021-12-31'], ['2022-01-01', END]];
function rollCorr(a, b, w) { const n = a.length, o = new Array(n).fill(NaN); for (let i = w - 1; i < n; i++) o[i] = corr(a.slice(i - w + 1, i + 1), b.slice(i - w + 1, i + 1)); return o; }
const PAIRS = [['btc', 'sp500'], ['btc', 'gold'], ['eth', 'sp500']];
function weekly(a, b) {              // concat(close a, close b).dropna().resample('W-FRI').last(), log returns, dropna
  const J = joint(a, b), lab = J.pd.map(d => addDays(d, (4 - dow(d) + 7) % 7)), last = new Map();
  J.pd.forEach((d, i) => last.set(lab[i], [J.pa[i], J.pb[i]]));
  const ra = [], rb = [], dd = []; let prev = null;
  for (let t = ms(lab[0]); t <= ms(lab[lab.length - 1]); t += 7 * DAY) {
    const L = isoOf(t), v = last.get(L) || null;
    if (v && prev) { ra.push(Math.log(v[0]) - Math.log(prev[0])); rb.push(Math.log(v[1]) - Math.log(prev[1])); dd.push(L); }
    prev = v;
  }
  return { d: dd, a: ra, b: rb };
}
function lastLE(S, d) { let j = -1; for (let i = 0; i < S.d.length && S.d[i] <= d; i++) j = i; return j; }
function crisis(a, b) {
  const row = {};
  ['btc', 'eth', 'sp500', 'gold'].forEach(k => { const S = PX[k], i0 = lastLE(S, a), i1 = lastLE(S, b); row[k] = 100 * Math.log(S.p[i1] / S.p[i0]); });
  const R = rets('btc', a, b); let im = 0; R.r.forEach((v, i) => { if (v < R.r[im]) im = i; });
  row.btc_worst = R.r[im]; row.btc_worst_date = R.d[im];
  return row;
}
function worst(k, q) {
  const R = joint(k, 'sp500'), thr = quantile(R.b, q), idx = R.b.map((v, i) => i).filter(i => R.b[i] <= thr), x = idx.map(i => R.a[i]), s = idx.map(i => R.b[i]);
  const m = mean(x), sd = sdv(x), t = m / (sd / Math.sqrt(x.length));
  return { n: R.a.length, n_bad: x.length, thr, mean_sp: mean(s), mean: m, sd, t, p: t2sf(t, x.length - 1), mean_all: mean(R.a), corr_bad: corr(x, s), share_neg: 100 * mean(x.map(v => +(v < 0))), idx, R };
}
function drawdown(p) { let mx = -Infinity; return p.map(v => { mx = Math.max(mx, v); return 100 * (v / mx - 1); }); }
function ddEpisodes(S, depth) {      // drawdown_episodes: episodes between two record highs whose deepest drawdown is below `depth` %
  const dd = drawdown(S.p), out = []; let i = 0;
  const days = (a, b) => Math.round((ms(b) - ms(a)) / DAY);
  while (i < dd.length) {
    if (dd[i] < 0) {
      let j = i; while (j < dd.length && dd[j] < 0) j++;
      let tr = i; for (let q = i; q < j; q++) if (dd[q] < dd[tr]) tr = q;
      if (dd[tr] <= depth) {
        const pk = i > 0 ? i - 1 : 0;
        out.push({ peak: S.d[pk], peak_price: S.p[pk], trough: S.d[tr], trough_price: S.p[tr], depth: dd[tr], recovery: j < dd.length ? S.d[j] : null,
          days_down: days(S.d[Math.max(i - 1, 0)], S.d[tr]), days_under: j < dd.length ? days(S.d[Math.max(i - 1, 0)], S.d[j]) : null });
      }
      i = j;
    } else i++;
  }
  return out;
}
function etfCmp(L) {                 // etf_effect on windows of L years (L = 2 in the lecture; GARCH persistence exported)
  const W = D.etf[L], [b0, b1] = W.before, [a0, a1] = W.after, rb = rets('btc', b0, b1), ra = rets('btc', a0, a1), bf = levene([rb.r, ra.r]), o = { n_before: rb.r.length, n_after: ra.r.length };
  o.vol_before = sdv(rb.r) * Math.sqrt(365); o.vol_after = sdv(ra.r) * Math.sqrt(365); o.bf = bf.stat; o.p_bf = bf.p;
  [['before', rb, b0, b1], ['after', ra, a0, a1]].forEach(([lab, R, s, e]) => {
    const w = R.d.map(d => dow(d) >= 5), we = R.r.filter((_, i) => w[i]), wd = R.r.filter((_, i) => !w[i]);
    o['we_ratio_' + lab] = vari(we) / vari(wd); o['we_share_' + lab] = 100 * sum(we.map(v => v * v)) / sum(R.r.map(v => v * v));
    const J = joint('btc', 'sp500', s, e); o['corr_' + lab] = corr(J.a, J.b); o['pers_' + lab] = W['pers_' + lab];
  });
  o.win = W;
  return o;
}
function weShareYear() { const R = rets('btc', START.btc), num = {}, den = {}; R.d.forEach((d, i) => { const y = yearOf(d), v = R.r[i] ** 2; den[y] = (den[y] || 0) + v; if (dow(d) >= 5) num[y] = (num[y] || 0) + v; }); const o = {}; Object.keys(den).forEach(y => { if (+y >= 2015) o[y] = 100 * (num[y] || 0) / den[y]; }); return o; }
function ibit(k) {                   // IBIT against Bitcoin on the ETF's trading days; k-day non-overlapping returns (k = 1 in the lecture)
  const J = joint('ibit', 'btc', D.etf_date), a = [], b = [];
  for (let i = 0; i + k <= J.a.length; i += k) { a.push(sum(J.a.slice(i, i + k))); b.push(sum(J.b.slice(i, i + k))); }
  const beta = cov(b, a) / vari(b), diff = a.map((v, i) => v - b[i]), ann = Math.sqrt(252 / k);
  return { n: a.length, corr: corr(a, b), beta, te: sdv(diff) * ann, sd_diff: sdv(diff), vol_ibit: sdv(a) * ann, vol_btc_same: sdv(b) * ann, a, b, ann };
}
function pegStats(k, field, thr) {
  const P = D.peg[k], dev = P[field].map(v => 1e4 * (v - 1)), low = P.low.map(v => 1e4 * (v - 1)), d = dated(P), x = dev;
  let imin = 0, imax = 0, il = 0; dev.forEach((v, i) => { if (v < dev[imin]) imin = i; if (v > dev[imax]) imax = i; }); low.forEach((v, i) => { if (v < low[il]) il = i; });
  const ph = corr(x.slice(1), x.slice(0, -1)), sh = t => 100 * mean(dev.map(v => +(Math.abs(v) > t)));
  return { n: dev.length, mean: mean(dev), sd: sdv(dev), mad: median(dev.map(Math.abs)), gt10: sh(10), gt50: sh(50), gt100: sh(100), gtx: sh(thr), close_min: dev[imin], close_min_date: d[imin],
    close_max: dev[imax], close_max_date: d[imax], low_min: low[il], low_min_date: d[il], phi: ph, half_life: Math.log(0.5) / Math.log(Math.abs(ph)), dev, d };
}
function hsVar(x, a) { const s = Float64Array.from(x).sort(); return -s[Math.ceil(s.length * a) - 1]; }
function hsEs(x, a) { const s = Float64Array.from(x).sort(), k = Math.ceil(s.length * a); let c = 0; for (let i = 0; i < k; i++) c += s[i]; return -c / k; }
const usd = (v, W) => W * (1 - Math.exp(-v / 100));
function overl(x, h) { const o = []; let run = 0; for (let i = 0; i < x.length; i++) { run += x[i]; if (i >= h) run -= x[i - h]; if (i >= h - 1) o.push(run); } return o; }
function varTab(k, a, ae) {          // var_table: VaR a and ES ae by HS, Normal, Student-t (exported ML fit), GARCH-t (exported one-day forecast)
  const x = rets(k, START[k]).r, m = mean(x), s = sdv(x), V = D.var[k], qe = tppf(ae, V.nu);
  return { hs_v: hsVar(x, a), hs_e: hsEs(x, ae), n_v: -(m + s * Pinv(a)), n_e: -m + s * phi(Pinv(ae)) / ae,
    t_v: -(V.loc + V.sc * tppf(a, V.nu)), t_e: -V.loc + V.sc * tpdf(qe, V.nu) / ae * (V.nu + qe * qe) / (V.nu - 1),
    g_v: -(V.g_mu + V.g_sigma * stdtq(V.g_nu, a)), g_e: -V.g_mu + V.g_sigma * stdtes(V.g_nu, ae), hs10: hsVar(overl(x, 10), a), sqrt10: Math.sqrt(10) * hsVar(x, a), mu: m, sd: s, nu: V.nu, g_sigma: V.g_sigma, x };
}
function kupiec(x, n, a) { const pi = x / n, l0 = (n - x) * Math.log(1 - a) + x * Math.log(a), l1 = (n - x) * Math.log(1 - pi) + (x > 0 ? x * Math.log(pi) : 0), lr = -2 * (l0 - l1); return { lr, p: chi2sf(lr, 1), l0, l1 }; }
const BT = (() => { const R = rets('btc', START.btc), i0 = R.d.findIndex(d => d >= '2018-01-01'); return { R, i0, n: R.r.length - i0 }; })();
const BTQ = {};
function backtest(w, a) {
  const R = BT.R, G = D.bt, hs = [], gv = [], hitH = [], hitG = [], key = w + '|' + a;
  if (BTQ[key]) return BTQ[key];
  const qs = G.nu.map(nu => stdtq(nu, a));
  for (let j = 0; j < BT.n; j++) {
    const i = BT.i0 + j, h = hsVar(R.r.slice(i - w, i), a), g = a === 0.01 ? G.var[j] : -(G.mu[G.blk[j]] + G.sig[j] * qs[G.blk[j]]);
    hs.push(h); gv.push(g); hitH.push(+(R.r[i] < -h)); hitG.push(+(R.r[i] < -g));
  }
  const res = { d: R.d.slice(BT.i0), r: R.r.slice(BT.i0), n: BT.n };
  [['HS', hs, hitH], ['GARCH-t', gv, hitG]].forEach(([m, v, h]) => {
    const x = sum(h), ku = kupiec(x, BT.n, a), yr = {};
    res.d.forEach((d, j) => { const y = yearOf(d); yr[y] = (yr[y] || 0) + h[j]; });
    res[m] = { x, rate: 100 * x / BT.n, exp: a * BT.n, lr: ku.lr, p: ku.p, year: yr, mean_var: mean(v), v, hit: h };
  });
  return (BTQ[key] = res);
}
function supplyAt(y) { let left = Math.max(0, (y - 2009) * 52560), s = 0, sub = 50; while (left > 0 && sub > 1e-8) { const take = Math.min(left, 210000); s += take * sub; left -= take; sub /= 2; } return s / 1e6; }
const fyear = s => { const y = yearOf(s); return y + (ms(s) - ms(y + '-01-01')) / (ms((y + 1) + '-01-01') - ms(y + '-01-01')); };
function halving(h) {
  const S = PX.btc, last = S.d[S.d.length - 1];
  return D.halvings.slice(1).map(H => {
    const i0 = lastLE(S, H), t2 = addDays(H, h), ok = t2 <= last, i1 = ok ? lastLE(S, t2) : -1;
    return { H, t2, i0, i1, p0: S.p[i0], p1: ok ? S.p[i1] : NaN, chg: ok ? 100 * (S.p[i1] / S.p[i0] - 1) : NaN, sup: supplyAt(fyear(t2)) };
  });
}

// ================================================================= formatting, tables, panels
const SEP = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, SEP).replace('-', '−');
const pc = (v, d) => nf(v, d === undefined ? 1 : d) + '%';
const sg = (x, d) => (x > 0 ? '+' : '') + nf(x, d);
const SIG = (p, k) => (p < 0.001 ? (LANG === 'ro' ? '< 0,001' : '< 0.001') : nf(p, k || 3));
const DT = s => T.date(s);
const LVE = a => (100 * a).toFixed(Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);
const LV = a => nf(100 * a, Math.abs(100 * a - Math.round(100 * a)) > 1e-9 ? 1 : 0);
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
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const kSel = id => $(id).value;
const M0 = { l: 56, r: 12, t: 14, b: 40 };
const vtr = (x, y0, y1, color, name, dash, extra) => Object.assign({ x: [x, x], y: [y0, y1], mode: 'lines', line: { color, width: 1.2, dash: dash || 'dash' }, name, hoverinfo: 'skip' }, extra || {});
const htr = (x0, x1, y, color, name, dash, extra) => Object.assign({ x: [x0, x1], y: [y, y], mode: 'lines', line: { color, width: 1.1, dash: dash || 'dash' }, name, hoverinfo: 'skip' }, extra || {});
const NM = { btc: 'Bitcoin', eth: 'Ethereum', sol: 'Solana', sp500: 'S&P 500', gold: 'Gold', usdt: 'Tether (USDT)', usdc: 'USD Coin (USDC)', dai: 'Dai (DAI)', ibit: 'IBIT' };
const AN = k => (T.an_ && T.an_[k]) || NM[k];
const ANT = k => { const s = AN(k); return s[0].toUpperCase() + s.slice(1); };
const CK = k => COL[k];
const BAR = (x, y, color, name, extra) => Object.assign({ type: 'bar', x, y, marker: { color }, name }, extra || {});
const DIA = (x, y, color, name, extra) => Object.assign({ x, y, mode: 'markers', marker: { color, size: 11, symbol: 'diamond', line: { color: DARK, width: 1 } }, name, hoverinfo: 'skip' }, extra || {});

// ================================================================= 1. Bitcoin price and supply, the halvings
function hvBlock() {
  const h = +$('hv-h').value, H = halving(h), S = PX.btc, P = panels('hv-chart', 2, { h: 390, widths: [0.6, 0.4] }); label('hv-h').textContent = h;
  const lo = Math.min(...S.p), hi = Math.max(...S.p), cols = [RED, GREEN, PURPLE];
  const tr = [on2(P.ax[0], { x: S.d, y: S.p, mode: 'lines', line: { color: COL.btc, width: 1.2 }, name: 'Bitcoin price (USD, log scale)', hovertemplate: '%{x}: %{y:,.0f} USD<extra></extra>' })];
  H.forEach((o, i) => {
    tr.push(on2(P.ax[0], vtr(o.H, lo, hi, RED, 'Halving', 'dash', { showlegend: i === 0 })));
    const j1 = o.i1 >= 0 ? o.i1 : S.d.length - 1;
    tr.push(on2(P.ax[0], { x: S.d.slice(o.i0, j1 + 1), y: S.p.slice(o.i0, j1 + 1), mode: 'lines', line: { color: cols[i], width: 2.6 }, name: `${o.H.slice(0, 4)} halving + ${h} days`, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[0], { x: [S.d[j1]], y: [S.p[j1]], mode: 'markers+text', marker: { color: cols[i], size: 9, line: { color: DARK, width: 1 } }, text: [isNaN(o.chg) ? '' : sg(o.chg, 0) + '%'], textposition: 'top left', textfont: { size: 11, color: DARK }, showlegend: false, hoverinfo: 'skip', cliponaxis: false }));
  });
  const ys = Array.from({ length: 32 }, (_, i) => 2009 + i);
  tr.push(on2(P.ax[1], { x: ys, y: ys.map(supplyAt), mode: 'lines', line: { color: BLUE, width: 1.8 }, name: 'Bitcoin supply (million BTC)', hovertemplate: '%{x}: %{y:.3f} million<extra></extra>' }));
  tr.push(on2(P.ax[1], htr(2009, 2040, 21, GREEN, 'Limit: 21 million', 'dot')));
  tr.push(on2(P.ax[1], { x: H.map(o => fyear(o.t2)), y: H.map(o => o.sup), mode: 'markers+text', marker: { color: cols, size: 9, line: { color: DARK, width: 1 } }, text: H.map(o => o.sup.toFixed(2)), textposition: 'bottom right', textfont: { size: 10.5, color: DARK }, name: `supply ${h} days after each halving`, hoverinfo: 'skip' }));
  setAx(P, 0, AX(''), AX('USD', { type: 'log' })); setAx(P, 1, AX('year', { range: [2008, 2041] }), AX('million BTC', { range: [0, 22.5] }));
  plot('hv-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  out('hv-out', [T.hv(h, H.map(o => [o.H.slice(0, 4), fint(o.p0), isNaN(o.p1) ? null : fint(o.p1), isNaN(o.chg) ? null : sg(o.chg, 0), nf(o.sup, 2), DT(o.H)]), fint(S.p[0]), fint(S.p[S.p.length - 1]), nf(supplyAt(2024), 2), nf(supplyAt(2030), 2), h !== 365)]);
}

// ================================================================= x1. annualisation with the actual frequency (worked example)
function anBlock() {
  const A = +$('an-A').value, k = kSel('an-k'), s = describe(k), P = panels('an-chart', 2, { h: 340 }), As = Array.from({ length: 201 }, (_, i) => 200 + i); label('an-A').textContent = A;
  const c = CK(k), mk = (i, f) => on2(P.ax[i], { x: [252, 365, A], y: [252, 365, A].map(f), mode: 'markers+text', marker: { color: [BLUE, GREEN, RED], size: [8, 8, 12], symbol: ['circle', 'circle', 'diamond'], line: { color: DARK, width: 1 } }, text: ['252', '365', A === 252 || A === 365 ? '' : 'A = ' + A], textposition: ['bottom right', 'top left', 'middle right'], textfont: { size: 10.5, color: DARK }, name: 'A', showlegend: false, hoverinfo: 'skip', cliponaxis: false });
  const tr = [on2(P.ax[0], { x: As, y: As.map(a => Math.sqrt(a) * s.sd), mode: 'lines', line: { color: c, width: 2 }, name: `${NM[k]}: annual volatility √A × s`, hovertemplate: 'A = %{x}: %{y:.1f}%<extra></extra>' }), mk(0, a => Math.sqrt(a) * s.sd),
    on2(P.ax[1], { x: As, y: As.map(a => a * s.mean), mode: 'lines', line: { color: c, width: 2, dash: 'dash' }, name: `${NM[k]}: annual mean A × r̄`, hovertemplate: 'A = %{x}: %{y:.1f}%<extra></extra>' }), mk(1, a => a * s.mean),
    on2(P.ax[0], vtr(s.ppy, Math.sqrt(200) * s.sd, Math.sqrt(400) * s.sd, DARK, `actual frequency A = ${s.ppy.toFixed(1)}`, 'dot'))];
  setAx(P, 0, AX('returns per year, A'), AX('annual volatility (%)')); setAx(P, 1, AX('returns per year, A'), AX('annual mean (%)'));
  plot('an-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.08 : -0.2), margin: M0 })));
  const v = Math.sqrt(A) * s.sd, m = A * s.mean;
  steps('an-we', [T.an1(AN(k), fint(s.n), nf(s.mean, 4), nf(s.sd, 3), nf(s.ppy, 1)), T.an2(A, nf(Math.sqrt(A), 2), nf(s.sd, 3), nf(v, 1), nf(s.mean, 4), nf(m, 1)),
    T.an3(nf(Math.sqrt(252) * s.sd, 1), nf(252 * s.mean, 1), nf(100 * (1 - Math.sqrt(252 / A)), 1), nf(100 * (1 - 252 / A), 1), A > 252, A === 252), T.an4(nf(m / v, 2), nf(252 * s.mean / (Math.sqrt(365) * s.sd), 2))]);
  out('an-out', [T.an(A, AN(k), nf(v, 1), nf(m, 1), A !== 365 && A !== 252, k !== 'btc')]);
}

// ================================================================= x2. five assets side by side (table)
function dsBlock() {
  const st = kSel('ds-s'), R = ASSETS.map(k => describe(k, st === 'full' ? undefined : (st > START[k] ? st : START[k]))), P = panels('ds-chart', 2, { h: 330 });
  const xs = ASSETS.map(k => NM[k]), cs = ASSETS.map(CK);
  const tr = [on2(P.ax[0], BAR(xs, R.map(r => r.ann_vol), cs, 'annual volatility √A × s (%)', { text: R.map(r => r.ann_vol.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })),
    on2(P.ax[1], BAR(xs, R.map(r => r.share5), cs, 'days with |r| > 5% (%)', { marker: { color: cs, pattern: { shape: '/' } }, text: R.map(r => r.share5.toFixed(1)), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.2f}% of days<extra></extra>' }))];
  setAx(P, 0, AX('', { type: 'category' }), AX('annual volatility (%)', { range: [0, 1.18 * Math.max(...R.map(r => r.ann_vol))] }));
  setAx(P, 1, AX('', { type: 'category' }), AX('days with |r| > 5% (%)', { range: [0, 1.18 * Math.max(...R.map(r => r.share5))] }));
  plot('ds-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: M0 })));
  table('ds-tab', T.h.ds, R.map((r, i) => [ANT(ASSETS[i]), r.first.slice(0, 4), fint(r.n), nf(r.mean, 3), nf(r.sd, 2), nf(r.ann_vol, 1), nf(r.skew, 2), nf(r.exkurt, 1), nf(r.min, 1), DT(r.min_date), nf(r.share5, 1)]));
  const sp = R[3].ann_vol;
  out('ds-out', [T.ds(st, [0, 1, 2].map(i => nf(R[i].ann_vol / sp, 1)), nf(R[0].ann_vol, 1), nf(sp, 1), nf(R[0].share5, 1), nf(R[0].min, 1), DT(R[0].min_date), R.filter(r => r.skew < 0).length, R.filter(r => r.exkurt > 0).length, st !== 'full')]);
}

// ================================================================= 2. volatility through time
function rvBlock() {
  const w = +$('rv-w').value, A = +kSel('rv-a'), V = rvol(w, A), nw = single('rv-chart', 380); label('rv-w').textContent = w;
  const tr = ['btc', 'eth', 'sp500'].map(k => ({ x: V[k].d, y: V[k].v, mode: 'lines', line: { color: CK(k), width: 1.1 }, name: `${NM[k]} (${V[k].w} returns, × √${k === 'sp500' ? 252 : A})`, hovertemplate: `${NM[k]} %{x}: %{y:.1f}%<extra></extra>` }));
  plot('rv-chart', tr, layout({ xaxis: AX(''), yaxis: AX('annualised volatility (%)'), legend: LEG(nw, -0.12), margin: M0 }));
  const f = k => [nf(V[k].median, 0), nf(V[k].max, 0), DT(V[k].max_date)];
  out('rv-out', [T.rv(w, V.ws, A, f('btc'), f('eth'), f('sp500'), nf(V.ratio_median, 2), nf(V.share_btc_below_sp, 2), nf(V.btc.last, 0), nf(V.sp500.last, 0), w !== 30 || A !== 365)]);
}

// ================================================================= 3. Bitcoin volatility by day of the week
function wdBlock() {
  const y = kSel('wd-y'), m = kSel('wd-m'), nw = single('wd-chart', 360), cols = [BLUE, ORANGE, RED];
  const tr = Object.entries(WD).map(([lab, [a, b]], i) => { const W = weekday(a, b); return BAR(DAYS, m === 'sd' ? W.sd : W.mean, cols[i], lab, { hovertemplate: `${lab}, %{x}: %{y:.2f}%<extra></extra>` }); });
  let Y = null;
  if (y !== 'none') { Y = weekday(y + '-01-01', y + '-12-31' > END ? END : y + '-12-31'); tr.push(BAR(DAYS, m === 'sd' ? Y.sd : Y.mean, GREEN, `${y} (extra run)`, { marker: { color: GREEN, pattern: { shape: '/' } }, hovertemplate: `${y}, %{x}: %{y:.2f}%<extra></extra>` })); }
  plot('wd-chart', tr, layout({ barmode: 'group', xaxis: AX('', { type: 'category' }), yaxis: AX(m === 'sd' ? 'std. dev. of daily returns (%)' : 'mean daily return (%)', { zeroline: true, zerolinecolor: DARK }), legend: LEG(nw, -0.12), margin: M0 }));
  const rows = Object.entries(WD).map(([lab, [a, b]]) => { const W = weekday(a, b); return [lab, nf(W.b, 2), nf(W.p, 2), nf(W.var_ratio, 2), SIG(W.p_bf)]; });
  const items = [T.wd(rows, m)];
  if (Y) items.push(T.wdY(y, fint(Y.n), nf(Y.b, 2), nf(Y.p, 2), nf(Y.var_ratio, 2), SIG(Y.p_bf)));
  out('wd-out', items);
}

// ================================================================= x3. Hill estimator (table)
const HK = Array.from({ length: 39 }, (_, i) => 0.005 + 0.0025 * i);
const HC = {};
function hillAll(k) {
  if (HC[k]) return HC[k];
  const x = describe(k).x, n = x.length, L = x.filter(v => v < 0).map(v => -v), R = x.filter(v => v > 0), ks = HK.map(q => Math.floor(q * n));
  return (HC[k] = { n, L, R, cl: hillCurve(L, ks), cr: hillCurve(R, ks) });
}
function hlBlock() {
  const f = +kSel('hl-k'), P = panels('hl-chart', 2, { h: 340 });
  const R = ASSETS.map(k => { const H = hillAll(k), kk = Math.floor(f * H.n); return { k, n: H.n, kk, al: hill(H.L, kk), ar: hill(H.R, kk), cl: H.cl, cr: H.cr }; });
  const tr = [];
  [0, 1].forEach(i => {
    R.forEach(r => tr.push(on2(P.ax[i], { x: HK.map(q => 100 * q), y: i ? r.cr : r.cl, mode: 'lines', line: { color: CK(r.k), width: 1.6 }, name: NM[r.k], showlegend: i === 0, hovertemplate: `${NM[r.k]}: k/n = %{x:.2f}%, α̂ = %{y:.2f}<extra></extra>` })));
    tr.push(on2(P.ax[i], DIA(R.map(() => 100 * f), R.map(r => (i ? r.ar : r.al)), R.map(r => CK(r.k)), `k = ${(100 * f).toFixed(1)}% of n`, { showlegend: i === 0 })));
    tr.push(on2(P.ax[i], htr(0.5, 10, 4, DARK, 'α = 4', 'dot', { showlegend: i === 0 })));
  });
  setAx(P, 0, AX('k as a share of n (%)'), AX('left tail (losses): α̂', { range: [1.5, 5.5] })); setAx(P, 1, AX('k as a share of n (%)'), AX('right tail (gains): α̂', { range: [1.5, 5.5] }));
  plot('hl-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  const ci = (r, a) => `[${nf(a * (1 - 1.96 / Math.sqrt(r.kk)), 2)}, ${nf(a * (1 + 1.96 / Math.sqrt(r.kk)), 2)}]`;
  table('hl-tab', T.h.hl, R.map(r => [ANT(r.k), fint(r.n), r.kk, nf(r.al, 2), ci(r, r.al), nf(r.ar, 2), ci(r, r.ar)]));
  const b = R[0];
  out('hl-out', [T.hl(nf(100 * f, 1), b.kk, nf(b.al, 2), nf(b.ar, 2), R.filter(r => r.al < 4).length, R.filter(r => r.al < r.ar).length, f !== 0.025)]);
}

// ================================================================= x4. the Ljung-Box test (worked example and table)
const LBC = {};
function lbCurves(k) {
  if (LBC[k]) return LBC[k];
  const x = describe(k).x, x2 = x.map(v => v * v), a1 = acf(x, 30), a2 = acf(x2, 30), n = x.length;
  const qc = rho => { let s = 0; return rho.map((r, j) => { s += r ** 2 / (n - j - 1); return n * (n + 2) * s; }); };
  return (LBC[k] = { x, x2, a1, a2, n, q1: qc(a1), q2: qc(a2) });
}
const CV = Array.from({ length: 30 }, (_, i) => chi2ppf(0.95, i + 1));
function lbBlock() {
  const m = +$('lb-m').value, k = kSel('lb-k'), xs = kSel('lb-x'), P = panels('lb-chart', 2, { h: 330 }), mm = Array.from({ length: 30 }, (_, i) => i + 1); label('lb-m').textContent = m;
  const B = lbCurves(k), tr = [];
  [[B.q1, BLUE, 'r_t'], [B.q2, RED, 'r_t²']].forEach(([q, c, nm], i) => {
    tr.push(on2(P.ax[i], { x: mm, y: q, mode: 'lines+markers', line: { color: c, width: 1.8 }, marker: { size: 5 }, name: `${NM[k]}: Q(m) of ${nm}`, hovertemplate: 'm = %{x}: Q = %{y:.1f}<extra></extra>' }));
    tr.push(on2(P.ax[i], { x: mm, y: CV, mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'χ²(m) 5% critical value', showlegend: i === 0, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[i], DIA([m], [q[m - 1]], AMBER, `m = ${m}`, { mode: 'markers+text', text: [q[m - 1].toFixed(1)], textposition: 'top left', textfont: { size: 11, color: DARK }, showlegend: i === 0, cliponaxis: false })));
  });
  setAx(P, 0, AX('number of lags m'), AX('Q(m) of r_t')); setAx(P, 1, AX('number of lags m'), AX('Q(m) of r_t²'));
  plot('lb-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  table('lb-tab', T.h.lb(m), ASSETS.map(a => { const C2 = lbCurves(a), p1 = chi2sf(C2.q1[m - 1], m), p2 = chi2sf(C2.q2[m - 1], m); return [{ v: ANT(a), c: a === k ? 'hi' : '' }, fint(C2.n), nf(C2.q1[m - 1], 1), SIG(p1), nf(C2.q2[m - 1], 1), SIG(p2)]; }));
  const rho = xs === 'r' ? B.a1 : B.a2, L = ljung(xs === 'r' ? B.x : B.x2, m, rho), c = CV[m - 1];
  steps('lb-st', [T.lb1(AN(k), xs, fint(B.n), rho.slice(0, Math.min(m, 3)).map(v => nf(v, 4)), m), T.lb2(m, nf(L.s * 1e6, 3), fint(B.n), nf(L.q, 1)), T.lb3(m, nf(c, 2), SIG(L.p), L.q > c)]);
  out('lb-out', [T.lb(m, AN(k), nf(B.q1[m - 1], 1), SIG(chi2sf(B.q1[m - 1], m)), nf(B.q2[m - 1], 1), SIG(chi2sf(B.q2[m - 1], m)), nf(c, 1), m !== 10)]);
}

// ================================================================= 4. volatility clustering in Bitcoin (ACF)
function acBlock() {
  const k = kSel('ac-k'), Lg = +$('ac-L').value, x = describe(k).x, n = x.length, a1 = acf(x, Lg), a2 = acf(x.map(v => v * v), Lg), band = 1.96 / Math.sqrt(n), P = panels('ac-chart', 2, { h: 340 }), lags = Array.from({ length: Lg }, (_, i) => i + 1); label('ac-L').textContent = Lg;
  const tr = [];
  [[a1, BLUE, 'ACF of returns'], [a2, RED, 'ACF of squared returns']].forEach(([a, c, nm], i) => {
    tr.push(on2(P.ax[i], BAR(lags, a, c, `${NM[k]}: ${nm}`, { hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' })));
    tr.push(on2(P.ax[i], htr(0.5, Lg + 0.5, band, DARK, '±1.96/√n', 'dash', { showlegend: i === 0 })), on2(P.ax[i], htr(0.5, Lg + 0.5, -band, DARK, '-', 'dash', { showlegend: false })));
  });
  const lo = Math.min(-band, ...a1, ...a2) - 0.02, hi = Math.max(band, ...a1, ...a2) + 0.02;
  setAx(P, 0, AX('lag (days)'), AX('autocorrelation', { range: [lo, hi] })); setAx(P, 1, AX('lag (days)'), AX('', { range: [lo, hi] }));
  plot('ac-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  const o1 = a1.filter(v => Math.abs(v) > band).length, o2 = a2.filter(v => Math.abs(v) > band).length, l1 = ljung(x, 10), l2 = ljung(x.map(v => v * v), 10);
  out('ac-out', [T.ac(AN(k), fint(n), nf(band, 3), nf(a1[0], 3), o1, Lg, nf(a2[0], 3), o2, k !== 'btc' || Lg !== 30), T.ac2(nf(l1.q, 1), SIG(l1.p), nf(l2.q, 1), SIG(l2.p), nf(l2.q / l1.q, 1), k === 'btc')]);
}

// ================================================================= x5. GARCH(1,1)-t estimates and the half-life (table, worked example)
const GA = ['btc', 'eth', 'sp500', 'gold'];
const garchHL = g => (g.pers < 0.9999 ? Math.log(0.5) / Math.log(g.pers) : null);
function gtBlock() {
  const h = +$('gt-h').value, k = kSel('gt-k'), nw = single('gt-chart', 340), hs = Array.from({ length: 401 }, (_, i) => i); label('gt-h').textContent = h;
  const tr = GA.map(a => { const g = D.garch[a]; return { x: hs, y: hs.map(j => g.pers ** j), mode: 'lines', line: { color: CK(a), width: 1.8 }, name: `${NM[a]}: (α̂ + β̂)^h`, hovertemplate: `${NM[a]}, h = %{x}: %{y:.3f}<extra></extra>` }; });
  tr.push(htr(0, 400, 0.5, DARK, 'half of the shock left', 'dot'));
  tr.push(DIA(GA.map(() => h), GA.map(a => D.garch[a].pers ** h), GA.map(CK), `h = ${h} days`));
  plot('gt-chart', tr, layout({ xaxis: AX('days after a volatility shock, h'), yaxis: AX('share of the shock left in the variance', { range: [0, 1.05] }), legend: LEG(nw, -0.18), margin: M0 }));
  table('gt-tab', T.h.gt(h), GA.map(a => { const g = D.garch[a], hl = garchHL(g); return [{ v: ANT(a), c: a === k ? 'hi' : '' }, nf(g.omega, 3), nf(g.alpha, 3), nf(g.beta, 3), nf(g.pers, 3), hl ? nf(hl, 0) : '–', nf(g.nu, 2), `${nf(g.gamma, 3)} (${nf(g.t_gamma, 2)})`, nf(100 * g.pers ** h, 1) + '%']; }));
  const g = D.garch[k], hl = garchHL(g);
  steps('gt-we', [T.gt1(AN(k), nf(g.alpha, 4), nf(g.beta, 4), nf(g.pers, g.pers > 0.9999 ? 11 : 4)),
    hl ? T.gt2(nf(Math.log(g.pers), 6), nf(hl, 1)) : T.gt2b(nf(Math.log(g.pers) * 1e12, 2), nf(Math.log(0.5) / Math.log(g.pers) / 365.25 / 1e6, 0)),
    T.gt3(h, nf(100 * g.pers ** h, 1)), hl ? T.gt4(nf(g.ppy, 1), nf(g.omega, 4), nf(1 - g.pers, 4), nf(g.uncond_ann, 1), AN(k)) : T.gt4b()]);
  out('gt-out', [T.gt(h, GA.map(a => [AN(a), nf(100 * D.garch[a].pers ** h, 1)]), h !== 100)]);
}

// ================================================================= 5. conditional volatility: Bitcoin and the S&P 500
function gvBlock() {
  const h = +$('gv-h').value, sel = kSel('gv-k'), nw = single('gv-chart', 380), ks = sel === 'both' ? ['btc', 'sp500'] : [sel]; label('gv-h').textContent = h;
  const tr = [];
  ks.forEach(k => {
    const g = D.garch[k], S = g.sig, d = dated(S), A = Math.sqrt(g.ppy);
    tr.push({ x: d, y: S.v.map(v => v * A), mode: 'lines', line: { color: CK(k), width: 1.1 }, name: `${NM[k]}: GARCH(1,1)-t volatility (annualised)`, hovertemplate: `${NM[k]} %{x}: %{y:.1f}%<extra></extra>` });
    let s2 = g.s1 ** 2; const fx = [], fy = [];
    for (let j = 1; j <= h; j++) { if (j > 1) s2 = g.omega + g.pers * s2; fx.push(addDays(END, j)); fy.push(Math.sqrt(g.ppy * s2)); }
    tr.push({ x: fx, y: fy, mode: 'lines', line: { color: k === 'btc' ? RED : GREEN, width: 2.4, dash: 'dot' }, name: `${NM[k]}: forecast for the next ${h} days`, hovertemplate: `${NM[k]} forecast %{x}: %{y:.1f}%<extra></extra>`, showlegend: h > 0 });
  });
  plot('gv-chart', tr, layout({ xaxis: AX(''), yaxis: AX('%'), legend: LEG(nw, -0.12), margin: M0 }));
  const b = D.garch.btc, s = D.garch.sp500, fc = (g, j) => { let v = g.s1 ** 2; for (let i = 2; i <= j; i++) v = g.omega + g.pers * v; return Math.sqrt(g.ppy * v); };
  const items = [T.gv(nf(b.sig_max, 0), DT(b.sig_max_date), nf(s.sig_max, 0), DT(s.sig_max_date), DT(END), nf(b.sig_last, 0), nf(s.sig_last, 0))];
  if (h > 0) items.push(T.gvF(h, nf(fc(b, 1), 1), nf(fc(b, h), 1), nf(fc(s, 1), 1), nf(fc(s, h), 1), nf(s.uncond_ann, 1), nf(D.N.stats.sp500.ann_vol, 1)));
  out('gv-out', items);
}

// ================================================================= x6. variance-ratio tests on the whole sample (tables)
const EA = ['btc', 'eth', 'sp500'];
function subOf(k) { return SUB.map(([a, b]) => [a < START[k] ? START[k] : a, b]).filter(([a, b]) => a <= b); }
const VRC = {};
const vrq = (k, q) => (VRC[k + q] || (VRC[k + q] = vratio(describe(k).x, q)));
function vtBlock() {
  const q = +kSel('vt-q'), nw = single('vt-chart', 340), qs = Array.from({ length: 29 }, (_, i) => i + 2);
  const tr = EA.map(k => ({ x: qs, y: qs.map(j => vrq(k, j).vr), mode: 'lines+markers', marker: { size: 4 }, line: { color: CK(k), width: 1.6 }, name: NM[k], hovertemplate: `${NM[k]}: VR(%{x}) = %{y:.3f}<extra></extra>` }));
  tr.push(htr(2, 30, 1, DARK, 'random walk: VR = 1', 'dot'));
  tr.push(DIA(EA.map(() => q), EA.map(k => vrq(k, q).vr), EA.map(CK), `q = ${q}`));
  plot('vt-chart', tr, layout({ xaxis: AX('horizon q (days)'), yaxis: AX('variance ratio VR(q)'), legend: LEG(nw, -0.18), margin: M0 }));
  table('vt-tab', T.h.vt(q), EA.map(k => { const x = describe(k).x, l = ljung(x, 10), z = vrq(k, q); return [ANT(k), nf(acf(x, 1)[0], 3), `${nf(l.q, 1)} (${SIG(l.p)})`, nf(vrq(k, 2).vr, 3), nf(vrq(k, 5).vr, 3), nf(vrq(k, 10).vr, 3), `${nf(z.zs, 2)} (${nf(z.p, 3)})`]; }));
  const sub = [];
  ['btc', 'eth'].forEach(k => subOf(k).forEach(([a, b]) => { const x = rets(k, a, b).r, z = vratio(x, q); sub.push({ z, row: [ANT(k), `${a.slice(0, 4)}–${b.slice(0, 4)}`, fint(x.length), nf(acf(x, 1)[0], 3), nf(z.vr, 3), `${nf(z.zs, 2)} (${nf(z.p, 3)})`] }); }));
  table('vt-tab2', T.h.vt2(q), sub.map(s => s.row));
  const rej = EA.filter(k => Math.abs(vrq(k, q).zs) > 1.96).map(AN), rejs = sub.filter(s => Math.abs(s.z.zs) > 1.96).length;
  out('vt-out', [T.vt(q, rej, rejs, sub.length, q !== 5)]);
}

// ================================================================= 6. efficiency through time (rolling variance ratio)
function vrBlock() {
  const q = +kSel('vr-q'), w = +$('vr-w').value, zk = kSel('vr-z'), nw = single('vr-chart', 380); label('vr-w').textContent = w;
  const R = {}; EA.forEach(k => { R[k] = rollVR(k, q, w, 21); });
  const tr = EA.map(k => ({ x: R[k].map(r => r.d), y: R[k].map(r => r[zk]), mode: 'lines', line: { color: CK(k), width: 1.2 }, name: NM[k], hovertemplate: `${NM[k]} %{x}: %{y:.2f}<extra></extra>` }));
  const all = EA.flatMap(k => R[k].map(r => r.d)).sort(), d0 = all[0], d1 = all[all.length - 1];
  tr.push(htr(d0, d1, 1.96, DARK, '±1.96'), htr(d0, d1, -1.96, DARK, '-', 'dash', { showlegend: false }));
  plot('vr-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`${zk === 'zs' ? 'robust Z*' : 'i.i.d. Z'}(${q})`), legend: LEG(nw, -0.12), margin: M0 }));
  const sh = (k, f) => nf(100 * mean(R[k].map(r => +(Math.abs(r[f]) > 1.96))), 1), b = R.btc;
  out('vr-out', [T.vr(q, w, b.length, sh('btc', 'zs'), sh('eth', 'zs'), sh('sp500', 'zs'), sh('btc', 'z'), sh('sp500', 'z'), nf(Math.min(...b.map(r => r.vr)), 2), nf(Math.max(...b.map(r => r.vr)), 2), zk, q !== 5 || w !== 500)]);
}

// ================================================================= x7. measuring correlation across calendars (worked example: one rolling window)
const RCC = {};
function rcSeries(pr, w) { const key = pr + w; if (RCC[key]) return RCC[key]; const [a, b] = pr.split('_'), J = joint(a, b); return (RCC[key] = { J, rc: rollCorr(J.a, J.b, w) }); }
function argExt(v, f) { let j = -1; v.forEach((x, i) => { if (isNaN(x)) return; if (j < 0 || f(x, v[j])) j = i; }); return j; }
function rcBlock() {
  const w = +$('rc-w').value, pr = kSel('rc-p'), e = kSel('rc-e'), [a, b] = pr.split('_'), S = rcSeries(pr, w), J = S.J, nw = single('rc-chart', 380); label('rc-w').textContent = w;
  const i = e === 'max' ? argExt(S.rc, (x, y) => x > y) : e === 'min' ? argExt(S.rc, (x, y) => x < y) : S.rc.length - 1;
  const xa = J.a.slice(i - w + 1, i + 1), xb = J.b.slice(i - w + 1, i + 1), r = corr(xa, xb), bta = cov(xb, xa) / vari(xb), ma = mean(xa), mb = mean(xb);
  const lo = Math.min(...xb), hi = Math.max(...xb);
  plot('rc-chart', [{ x: xb, y: xa, mode: 'markers', marker: { color: CK(a), size: 5, opacity: 0.65 }, name: `${w} common days up to ${J.d[i]}`, hovertemplate: `${NM[b]} %{x:.2f}%, ${NM[a]} %{y:.2f}%<extra></extra>` },
    { x: [lo, hi], y: [ma + bta * (lo - mb), ma + bta * (hi - mb)], mode: 'lines', line: { color: DARK, width: 1.6 }, name: `least-squares line, correlation ${r.toFixed(2)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX(`${NM[b]} daily log return (%)`, { zeroline: true, zerolinecolor: C.band }), yaxis: AX(`${NM[a]} daily log return (%)`, { zeroline: true, zerolinecolor: C.band }), legend: LEG(nw, -0.18), margin: M0 }));
  const W = weekly(a, b), se = (1 - r * r) / Math.sqrt(w);
  steps('rc-st', [T.rc1(AN(a), AN(b), w, DT(J.d[i - w + 1]), DT(J.d[i])), T.rc2(nf(sdv(xa), 2), nf(sdv(xb), 2), nf(cov(xa, xb), 3), nf(r, 3)),
    T.rc3(w, nf(r, 2), nf(se, 3), nf(r - 1.96 * se, 2), nf(r + 1.96 * se, 2), nf(1 / Math.sqrt(w), 3)), T.rc4(nf(corr(J.a, J.b), 2), nf(corr(W.a, W.b), 2), fint(J.a.length), fint(W.a.length))]);
  out('rc-out', [T.rc(w, AN(a), AN(b), e, nf(r, 2), DT(J.d[i]), nf(se, 3), w !== 250)]);
}

// ================================================================= 7. rolling correlation with equities and gold
function coBlock() {
  const w = +$('co-w').value, nw = single('co-chart', 390), cols = [BLUE, AMBER, PURPLE]; label('co-w').textContent = w;
  const S = PAIRS.map(([a, b]) => rcSeries(a + '_' + b, w));
  const tr = S.map((s, i) => ({ x: s.J.d, y: s.rc, mode: 'lines', line: { color: cols[i], width: 1.3 }, name: `${NM[PAIRS[i][0]]} and ${NM[PAIRS[i][1]]}`, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }));
  const evc = [RED, CRIMSON, ORANGE, GREEN];
  Object.entries(D.events).forEach(([lab, d], i) => tr.push(vtr(d, -0.4, 0.8, evc[i], lab, 'dash')));
  tr.push(htr(S[0].J.d[0], S[0].J.d[S[0].J.d.length - 1], 0, DARK, 'zero', 'solid', { showlegend: false, line: { color: DARK, width: 0.6 } }));
  plot('co-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`correlation (${w} common days)`, { range: [-0.4, 0.8] }), legend: LEG(nw, -0.12), margin: M0 }));
  const f = s => { const J = s.J, pre = J.d.map((d, i) => i).filter(i => J.d[i] <= '2019-12-31'), post = J.d.map((d, i) => i).filter(i => J.d[i] >= '2020-01-01'), iM = argExt(s.rc, (x, y) => x > y);
    return [nf(corr(pre.map(i => J.a[i]), pre.map(i => J.b[i])), 2), nf(corr(post.map(i => J.a[i]), post.map(i => J.b[i])), 2), nf(s.rc[iM], 2), DT(J.d[iM]), nf(s.rc[s.rc.length - 1], 2)]; };
  const G = S[1].J, g26 = G.d.map((d, i) => i).filter(i => yearOf(G.d[i]) === 2026), g25 = G.d.map((d, i) => i).filter(i => yearOf(G.d[i]) <= 2025 && yearOf(G.d[i]) >= 2020);
  out('co-out', [T.co(w, f(S[0]), f(S[1]), f(S[2]), nf(corr(g25.map(i => G.a[i]), g25.map(i => G.b[i])), 2), nf(corr(g26.map(i => G.a[i]), g26.map(i => G.b[i])), 2), w !== 250)]);
}

// ================================================================= x8. correlation year by year (table)
const YRS = Array.from({ length: 13 }, (_, i) => 2014 + i);
function yearCorr(a, b, f) {
  const J = f === 'w' ? weekly(a, b) : joint(a, b), g = {};
  J.d.forEach((d, i) => { const y = yearOf(d); (g[y] = g[y] || [[], []]); g[y][0].push(J.a[i]); g[y][1].push(J.b[i]); });
  return { y: YRS.map(y => (g[y] && g[y][0].length > 2 ? corr(g[y][0], g[y][1]) : null)), all: corr(J.a, J.b) };
}
function cyBlock() {
  const y = +kSel('cy-y'), f = kSel('cy-f'), nw = single('cy-chart', 360), cols = [BLUE, AMBER, PURPLE];
  const R = PAIRS.map(([a, b]) => yearCorr(a, b, f));
  const tr = R.map((r, i) => BAR(YRS.map(String), r.y, cols[i], `${NM[PAIRS[i][0]]} with ${NM[PAIRS[i][1]]}`, { text: YRS.map((yy, j) => (yy === y && r.y[j] !== null ? r.y[j].toFixed(2) : '')), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }));
  plot('cy-chart', tr, layout({ barmode: 'group', xaxis: AX('', { type: 'category' }), yaxis: AX(`correlation of ${f === 'w' ? 'weekly' : 'daily'} log returns`, { range: [-0.25, 0.75], zeroline: true, zerolinecolor: DARK }), legend: LEG(nw, -0.12), margin: M0 }));
  const lab = T.h.cyRows;
  table('cy-tab', T.h.cy(f), R.map((r, i) => [lab[i], ...r.y.map((v, j) => ({ v: v === null ? '–' : nf(v, 2), c: YRS[j] === y ? 'hi' : '' })), nf(r.all, 2)]));
  const dW = yearCorr('btc', 'sp500', 'd').all, wW = yearCorr('btc', 'sp500', 'w').all, j = YRS.indexOf(y);
  out('cy-out', [T.cy(y, R.map(r => (r.y[j] === null ? '–' : nf(r.y[j], 2))), f, nf(dW, 2), nf(wW, 2), f === 'w' || y < 2019 || y > 2025)]);
}

// ================================================================= x9. three crisis episodes (table)
function crBlock() {
  const ep = kSel('cr-e'), x = +$('cr-x').value, nw = single('cr-chart', 360), [a0, b0] = D.episodes[ep], a = addDays(a0, -x), b = addDays(b0, x); label('cr-x').textContent = x;
  const tr = ['btc', 'eth', 'sp500', 'gold'].map(k => { const S = PX[k], i0 = lastLE(S, a), i1 = lastLE(S, b), d = S.d.slice(i0, i1 + 1), p = S.p.slice(i0, i1 + 1);
    return { x: d, y: p.map(v => 100 * Math.log(v / p[0])), mode: 'lines+markers', marker: { size: 4 }, line: { color: CK(k), width: 1.8 }, name: NM[k], hovertemplate: `${NM[k]} %{x}: %{y:.1f}%<extra></extra>` }; });
  plot('cr-chart', tr, layout({ xaxis: AX(''), yaxis: AX(`cumulative log return since ${a} (%)`, { zeroline: true, zerolinecolor: DARK }), legend: LEG(nw, -0.12), margin: M0 }));
  const rows = Object.entries(D.episodes).map(([lab, [p, q]]) => { const c = crisis(addDays(p, -x), addDays(q, x)); return [{ v: lab, c: lab === ep ? 'hi' : '' }, `${DT(addDays(p, -x))} – ${DT(addDays(q, x))}`, nf(c.btc, 1), nf(c.eth, 1), nf(c.sp500, 1), nf(c.gold, 1), `${nf(c.btc_worst, 1)} (${DT(c.btc_worst_date)})`]; });
  table('cr-tab', T.h.cr, rows);
  const c = crisis(a, b);
  out('cr-out', [T.cr(ep, DT(a), DT(b), nf(c.btc, 1), nf(c.eth, 1), nf(c.sp500, 1), nf(c.gold, 1), nf(c.btc_worst, 1), DT(c.btc_worst_date), x > 0)]);
}

// ================================================================= x10. hedge or safe haven? (worked example)
function shBlock() {
  const q = +$('sh-q').value / 100, k = kSel('sh-k'), P = panels('sh-chart', 2, { h: 360 }); label('sh-q').textContent = LV(q) + '%';
  const W = { btc: worst('btc', q), gold: worst('gold', q) }, tr = [];
  ['btc', 'gold'].forEach((a, i) => {
    const R = W[a].R;
    tr.push(on2(P.ax[i], { x: R.b, y: R.a, mode: 'markers', marker: { color: CK(a), size: 3, opacity: 0.35 }, name: `${NM[a]}: all common days`, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[i], { x: W[a].idx.map(j => R.b[j]), y: W[a].idx.map(j => R.a[j]), mode: 'markers', marker: { color: RED, size: 5 }, name: `worst ${LVE(q)}% of S&P 500 days`, showlegend: i === 0, hovertemplate: `S&P 500 %{x:.2f}%, ${NM[a]} %{y:.2f}%<extra></extra>` }));
    tr.push(on2(P.ax[i], DIA([W[a].mean_sp], [W[a].mean], AMBER, 'means on those days', { mode: 'markers+text', text: [W[a].mean.toFixed(2) + '%'], textposition: 'top right', textfont: { size: 11, color: DARK }, showlegend: i === 0 })));
  });
  setAx(P, 0, AX('S&P 500 daily log return (%)', { range: [-13, 10] }), AX('Bitcoin (%)', { range: [-50, 25] })); setAx(P, 1, AX('S&P 500 daily log return (%)', { range: [-13, 10] }), AX('gold (%)', { range: [-12, 9] }));
  plot('sh-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  const o = W[k];
  steps('sh-we', [T.sh1(fint(o.n), o.n_bad, LV(q), nf(o.thr, 2), nf(o.mean_sp, 2)), T.sh2(AN(k), nf(o.mean, 2), nf(o.sd, 2), o.n_bad, nf(o.t, 2), SIG(o.p)), T.sh3(nf(o.share_neg, 0), nf(o.corr_bad, 2), nf(o.mean_all, 2))]);
  out('sh-out', [T.sh(LV(q), W.btc.n_bad, nf(W.btc.mean, 2), nf(W.btc.t, 2), SIG(W.btc.p), nf(W.gold.mean, 2), nf(W.gold.t, 2), SIG(W.gold.p), q !== 0.05)]);
}

// ================================================================= x11. drawdown and the gain needed to recover (worked example)
function mddOf(k) {
  const S = PX[k], dd = drawdown(S.p); let tr = 0; dd.forEach((v, i) => { if (v < dd[tr]) tr = i; });
  let pk = 0; for (let i = 0; i <= tr; i++) if (S.p[i] > S.p[pk]) pk = i;
  let rec = -1; for (let i = tr + 1; i < S.p.length; i++) if (S.p[i] >= S.p[pk]) { rec = i; break; }
  return { S, pk, tr, rec, d: dd[tr] };
}
function mdBlock() {
  const d = +$('md-d').value, k = kSel('md-k'), nw = single('md-chart', 340), ds = Array.from({ length: 96 }, (_, i) => i); label('md-d').textContent = d + '%';
  const g = x => 100 * x / (100 - x), M = ['btc', 'eth', 'sp500'].map(a => [a, mddOf(a)]);
  plot('md-chart', [{ x: ds.slice(1), y: ds.slice(1).map(g), mode: 'lines', line: { color: BLUE, width: 2 }, name: 'gain needed to recover, d/(1 − d)', hovertemplate: 'loss %{x}%: gain %{y:.0f}%<extra></extra>' },
    DIA([d], [g(d)], AMBER, `loss of ${d}%`, { mode: 'markers+text', text: [`${g(d).toFixed(0)}%`], textposition: 'top left', textfont: { size: 11, color: DARK }, cliponaxis: false }),
    { x: M.map(([, m]) => -m.d), y: M.map(([, m]) => g(-m.d)), mode: 'markers+text', marker: { color: M.map(([a]) => CK(a)), size: 10, line: { color: DARK, width: 1 } }, text: M.map(([a]) => NM[a]), textposition: 'middle right', textfont: { size: 10.5, color: DARK }, name: 'maximum drawdowns', hoverinfo: 'skip' }],
  layout({ xaxis: AX('loss from the record high, d (%)', { range: [0, 100] }), yaxis: AX('gain needed (%), log scale', { type: 'log', range: [0, 3.4] }), legend: LEG(nw, -0.18), margin: M0 }));
  const m = mddOf(k), S = m.S;
  steps('md-we', [T.md1(AN(k), DT(S.d[m.pk]), fint(S.p[m.pk]), DT(S.d[m.tr]), fint(S.p[m.tr]), nf(m.d, 1), k === 'sp500'), T.md2(nf(-m.d / 100, 3), nf(1 + m.d / 100, 3), nf(g(-m.d), 0)),
    m.rec >= 0 ? T.md3(DT(S.d[m.rec]), fint(Math.round((ms(S.d[m.rec]) - ms(S.d[m.pk])) / DAY))) : T.md3b()]);
  out('md-out', [T.md(d, nf(g(d), 0), d !== 75 && d !== 80)]);
}

// ================================================================= 8. drawdowns of Bitcoin, Ethereum and the S&P 500
function ddBlock() {
  const y = +$('dd-y').value, t = +$('dd-t').value, nw = single('dd-chart', 380), a = y + '-01-01'; label('dd-y').textContent = y; label('dd-t').textContent = t + '%';
  const R = ['btc', 'eth', 'sp500'].map(k => { const c = closes(k, a), dd = drawdown(c.p); let im = 0; dd.forEach((v, i) => { if (v < dd[im]) im = i; }); return { k, c, dd, im, sh: 100 * mean(dd.map(v => +(v < -t))) }; });
  const tr = R.map(r => ({ x: r.c.d, y: r.dd, mode: 'lines', line: { color: CK(r.k), width: 1.1 }, name: NM[r.k], hovertemplate: `${NM[r.k]} %{x}: %{y:.1f}%<extra></extra>` }));
  tr.push(htr(a, END, -t, DARK, `${t}% below the record`, 'dot'));
  plot('dd-chart', tr, layout({ xaxis: AX(''), yaxis: AX('drawdown (%)', { range: [-105, 3] }), legend: LEG(nw, -0.12), margin: M0 }));
  out('dd-out', [T.dd(y, R.map(r => [AN(r.k), nf(r.dd[r.im], 1), DT(r.c.d[r.im]), nf(r.sh, 0)]), t, nf(-R[0].dd[R[0].dd.length - 1], 1), DT(END), y !== 2016 || t !== 10)]);
}

// ================================================================= x12. Bitcoin's large drawdowns (table)
function deBlock() {
  const th = +kSel('de-t'), nw = single('de-chart', 360), S = PX.btc, dd = drawdown(S.p), E = ddEpisodes(S, -th);
  plot('de-chart', [{ x: S.d, y: dd, mode: 'lines', line: { color: COL.btc, width: 1.1 }, name: 'Bitcoin drawdown', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    htr(S.d[0], END, -th, DARK, `threshold −${th}%`, 'dot'),
    { x: E.map(e => e.trough), y: E.map(e => e.depth), mode: 'markers+text', marker: { color: RED, size: 9, symbol: 'triangle-down', line: { color: DARK, width: 1 } }, text: E.map(e => e.depth.toFixed(1)), textposition: 'bottom center', textfont: { size: 10.5, color: DARK }, cliponaxis: false, name: 'trough of an episode', hoverinfo: 'skip' }],
  layout({ xaxis: AX(''), yaxis: AX('drawdown (%)', { range: [-105, 3] }), legend: LEG(nw, -0.12), margin: M0 }));
  table('de-tab', T.h.de, E.map(e => [e.peak, fint(e.peak_price), e.trough, fint(e.trough_price), nf(e.depth, 1), e.recovery || T.notyet, e.days_under === null ? '–' : fint(e.days_under)]));
  const rec = E.filter(e => e.recovery), lng = rec.reduce((m, e) => (e.days_under > m.days_under ? e : m), rec[0] || { days_under: 0 });
  out('de-out', [T.de(th, E.length, rec.length, rec.length ? fint(lng.days_under) : '–', rec.length ? lng.peak.slice(0, 4) : '–', th !== 50)]);
}

// ================================================================= 9. volatility and the weekend share around the ETFs
function efBlock() {
  const w = +$('ef-w').value, y = kSel('ef-y'), P = panels('ef-chart', 2, { h: 380, widths: [0.62, 0.38] }); label('ef-w').textContent = w;
  const R = rets('btc', START.btc), v = rollSd(R.r, w).map(z => z * Math.sqrt(365)), i0 = R.d.findIndex(d => d >= '2020-01-01'), d = R.d.slice(i0), vv = v.slice(i0), sh = weShareYear(), ys = Object.keys(sh);
  const sel = d.map((s, i) => i).filter(i => y !== 'none' && d[i].slice(0, 4) === y);
  const tr = [on2(P.ax[0], { x: d, y: vv, mode: 'lines', line: { color: COL.btc, width: 1.3 }, name: `Bitcoin volatility, ${w}-day window (annualised, %)`, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }),
    on2(P.ax[0], vtr(D.etf_date, 0, Math.max(...vv), RED, 'Spot ETFs: 11 Jan 2024')),
    on2(P.ax[0], { x: sel.map(i => d[i]), y: sel.map(i => vv[i]), mode: 'lines', line: { color: BLUE, width: 3 }, name: y === 'none' ? 'year highlighted' : `year ${y}`, hoverinfo: 'skip', showlegend: y !== 'none' }),
    on2(P.ax[1], BAR(ys, ys.map(k => sh[k]), ys.map(k => (k === y ? GREEN : BLUE)), 'Weekend share of squared returns (%)', { text: ys.map(k => (k === y ? sh[k].toFixed(1) : '')), textposition: 'outside', cliponaxis: false, textfont: { size: 10.5, color: DARK }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' })),
    on2(P.ax[1], htr(ys[0], ys[ys.length - 1], 200 / 7, GREEN, '2/7 of the days', 'dash'))];
  setAx(P, 0, AX(''), AX('%')); setAx(P, 1, AX('', { type: 'category', tickangle: -60, tickfont: { size: 9, color: DARK } }), AX('', { range: [0, 32] }));
  plot('ef-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.2), margin: M0 })));
  const at = s => vv[d.findIndex(x => x >= s)], since = ys.filter(k => +k >= 2020).map(k => sh[k]), items = [T.ef(w, nf(at('2024-01-10'), 0), nf(at('2023-01-10'), 0), nf(vv[vv.length - 1], 0), DT(END), nf(Math.min(...since), 1), nf(Math.max(...since), 1), w !== 90)];
  if (y !== 'none') items.push(T.efY(y, sel.length ? nf(mean(sel.map(i => vv[i])), 0) : null, nf(sh[y], 1)));
  out('ef-out', items);
}

// ================================================================= x13. before and after the ETFs (table)
function etBlock() {
  const L = kSel('et-L'), o = etfCmp(L), nw = single('et-chart', 360), [b0] = o.win.before, [, a1] = o.win.after, R = rets('btc', b0, a1);
  const sb = o.vol_before / Math.sqrt(365), sa = o.vol_after / Math.sqrt(365);
  plot('et-chart', [{ x: R.d, y: R.r, mode: 'lines', line: { color: COL.btc, width: 0.8 }, name: 'Bitcoin daily log return (%)', hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    htr(b0, o.win.before[1], 2 * sb, BLUE, '± 2 standard deviations, before', 'solid', { line: { color: BLUE, width: 2 } }), htr(b0, o.win.before[1], -2 * sb, BLUE, '-', 'solid', { showlegend: false, line: { color: BLUE, width: 2 } }),
    htr(o.win.after[0], a1, 2 * sa, GREEN, '± 2 standard deviations, after', 'solid', { line: { color: GREEN, width: 2 } }), htr(o.win.after[0], a1, -2 * sa, GREEN, '-', 'solid', { showlegend: false, line: { color: GREEN, width: 2 } }),
    vtr(D.etf_date, -20, 20, RED, 'Spot ETFs: 11 Jan 2024')],
  layout({ xaxis: AX(''), yaxis: AX('%', { range: [-20, 20] }), legend: LEG(nw, -0.12), margin: M0 }));
  const H = T.h.et;
  table('et-tab', H.head, [[H.rows[0], nf(o.vol_before, 1), nf(o.vol_after, 1)], [H.rows[1], nf(o.we_ratio_before, 2), nf(o.we_ratio_after, 2)], [H.rows[2], nf(o.corr_before, 2), nf(o.corr_after, 2)],
    [H.rows[3], nf(o.pers_before, 3), nf(o.pers_after, 3)], [H.rows[4], fint(o.n_before), fint(o.n_after)]]);
  out('et-out', [T.et(L, DT(o.win.before[0]), DT(o.win.before[1]), DT(o.win.after[0]), DT(o.win.after[1]), nf(o.vol_before, 1), nf(o.vol_after, 1), nf(o.bf, 2), nf(o.p_bf, 2), o.p_bf < 0.05, L !== '2')]);
}

// ================================================================= x14. how well does an ETF track Bitcoin? (worked example)
function ibitS(k, s) {
  const J = joint('ibit', 'btc', D.etf_date), idx = J.d.map((d, i) => i).filter(i => s === 'all' || J.d[i].slice(0, 4) === s), A = idx.map(i => J.a[i]), B = idx.map(i => J.b[i]), a = [], b = [];
  for (let i = 0; i + k <= A.length; i += k) { a.push(sum(A.slice(i, i + k))); b.push(sum(B.slice(i, i + k))); }
  const beta = cov(b, a) / vari(b), diff = a.map((v, i) => v - b[i]), ann = Math.sqrt(252 / k);
  return { n: a.length, nd: A.length, corr: corr(a, b), beta, te: sdv(diff) * ann, sd_diff: sdv(diff), a, b, ann };
}
function ibBlock() {
  const k = +kSel('ib-k'), s = kSel('ib-s'), o = ibitS(k, s), nw = single('ib-chart', 380), lo = Math.min(...o.b), hi = Math.max(...o.b);
  plot('ib-chart', [{ x: o.b, y: o.a, mode: 'markers', marker: { color: BLUE, size: 5, opacity: 0.7 }, name: `${k}-day log returns (${o.n})`, hovertemplate: 'Bitcoin %{x:.2f}%, IBIT %{y:.2f}%<extra></extra>' },
    { x: [lo, hi], y: [lo, hi], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'perfect tracking (45° line)', hoverinfo: 'skip' }],
  layout({ xaxis: AX('Bitcoin log return (%)', { zeroline: true, zerolinecolor: C.band }), yaxis: AX('IBIT log return (%)', { zeroline: true, zerolinecolor: C.band }), legend: LEG(nw, -0.18), margin: M0 }));
  steps('ib-we', [T.ib1(k, s, fint(o.nd), fint(o.n)), T.ib2(nf(o.corr, 2), nf(o.beta, 2)), T.ib3(nf(o.sd_diff, 3), k, nf(o.ann, 2), nf(o.te, 1))]);
  const d1 = ibitS(1, 'all'), d5 = ibitS(5, 'all');
  out('ib-out', [T.ib(k, nf(o.corr, 2), nf(o.beta, 2), nf(o.te, 1), nf(d1.te, 1), nf(d5.te, 1), nf(d5.corr, 2), k !== 1 || s !== 'all')]);
}

// ================================================================= 10. supply of stablecoins (DefiLlama, downloaded)
const LL = D.llama, SUPK = ['total', 'usdt', 'usdc', 'dai'];
const SUP = {}; SUPK.forEach(k => { SUP[k] = { d: dated(LL.supply[k]), v: LL.supply[k].v }; });
const supAt = (k, s) => { const S = SUP[k]; let j = -1; for (let i = 0; i < S.d.length && S.d[i] <= s; i++) j = i; return j >= 0 ? S.v[j] : NaN; };
function suBlock() {
  const dsel = kSel('su-y'), c = kSel('su-c'), d = dsel === 'end' ? END : dsel, nw = single('su-chart', 380), ks = c === 'all' ? SUPK : ['total', c];
  const nmS = { total: 'All USD stablecoins', usdt: NM.usdt, usdc: NM.usdc, dai: NM.dai };
  const tr = ks.map(k => ({ x: SUP[k].d, y: SUP[k].v, mode: 'lines', line: { color: CK(k), width: k === 'total' ? 1.6 : 1.2 }, name: nmS[k], hovertemplate: `${nmS[k]} %{x}: %{y:.1f} bn USD<extra></extra>` }));
  tr.push(DIA(ks.map(() => d), ks.map(k => supAt(k, d)), ks.map(CK), `values on ${d}`, { mode: 'markers+text', text: ks.map(k => supAt(k, d).toFixed(1)), textposition: 'top left', textfont: { size: 10.5, color: DARK }, cliponaxis: false }));
  plot('su-chart', tr, layout({ xaxis: AX(''), yaxis: AX('supply (USD bn)'), legend: LEG(nw, -0.12), margin: M0 }));
  const t = supAt('total', d), Nn = D.N.supply, now = LL.supply_now;
  out('su-out', [T.su(DT(d), nf(t, 1), nf(supAt('usdt', d), 1), nf(supAt('usdc', d), 1), nf(supAt('dai', d), 1), nf(100 * supAt('usdt', d) / t, 0), nf(100 * supAt('usdc', d) / t, 0), DT(LL.download)),
    T.su2(nf(Nn.total_end, 0), nf(now.total_end, 0), nf(Nn.growth_x, 0), nf(now.growth_x, 1), nf(Nn.usdt_share, 0), nf(now.usdt_share, 0), nf(Nn.usdc_share, 0), nf(now.usdc_share, 0), nf(Nn.total_2022max, 0), DT(Nn.total_2022max_date), nf(Nn.total_trough, 0), DT(Nn.total_trough_date), nf(Nn.usdc_2023_03_10, 0), nf(Nn.usdc_2023_12_31, 0), DT(LL.download))]);
}

// ================================================================= x15. peg deviation in basis points (worked example)
const PEGD = {}; ['usdt', 'usdc', 'dai'].forEach(k => { PEGD[k] = dated(D.peg[k]); });
const pegAt = (k, s, f) => { const i = PEGD[k].indexOf(s); return i >= 0 ? D.peg[k][f][i] : NaN; };
const fbp = x => (Math.round(x) > 0 ? '+' : '') + fint(x);
const MAR = Array.from({ length: 13 }, (_, i) => `2023-03-${String(8 + i).padStart(2, '0')}`);
function bpBlock() {
  const day = +$('bp-d').value, k = kSel('bp-k'), f = kSel('bp-p'), W = +kSel('bp-W'), s = MAR[day - 8], nw = single('bp-chart', 360); label('bp-d').textContent = day;
  const cs = ['usdc', 'dai', 'usdt'], tr = cs.map(c => BAR(MAR.map(x => x.slice(5)), MAR.map(x => 1e4 * (pegAt(c, x, f) - 1)), CK(c), `${NM[c]}: ${f === 'close' ? 'close' : 'daily low'}`, { hovertemplate: `${NM[c]} %{x}: %{y:.0f} bp<extra></extra>` }));
  tr.push(DIA(cs.map(() => s.slice(5)), cs.map(c => 1e4 * (pegAt(c, s, f) - 1)), cs.map(CK), `${day} March`, { mode: 'markers+text', text: cs.map(c => (1e4 * (pegAt(c, s, f) - 1)).toFixed(0)), textposition: 'bottom center', textfont: { size: 10.5, color: DARK }, cliponaxis: false }));
  plot('bp-chart', tr, layout({ barmode: 'group', xaxis: AX('March 2023 (month-day)', { type: 'category' }), yaxis: AX('deviation from 1 USD (bp)'), legend: LEG(nw, -0.18), margin: M0 }));
  const P = pegAt(k, s, f), dv = 1e4 * (P - 1);
  steps('bp-st', [T.bp1(AN(k), DT(s), f, nf(P, 4)), T.bp2(nf(P, 4), fint(dv), nf(dv / 100, 2)), T.bp3(fint(W), nf(P, 4), fint(W * Math.abs(1 - P)), dv < 0), T.bp4(nf(pegAt(k, s, 'close'), 4), nf(pegAt(k, s, 'low'), 4), fint(W * Math.abs(1 - pegAt(k, s, 'low'))))]);
  out('bp-out', [T.bp(DT(s), cs.map(c => [AN(c), fbp(1e4 * (pegAt(c, s, 'close') - 1)), fbp(1e4 * (pegAt(c, s, 'low') - 1))]), day !== 11)]);
}

// ================================================================= 11. peg deviations of USDT, USDC and DAI
function pgBlock() {
  const t = +kSel('pg-t'), f = kSel('pg-p'), cs = ['usdt', 'usdc', 'dai'], P = panels('pg-chart', 3, { rows: true, hs: 170 });
  const S = cs.map(c => pegStats(c, f, t)), tr = [];
  S.forEach((s, i) => {
    const d0 = s.d[0], d1 = s.d[s.d.length - 1], bad = s.dev.map((v, j) => j).filter(j => Math.abs(s.dev[j]) > t);
    tr.push(on2(P.ax[i], { x: s.d, y: s.dev.map(v => Math.max(v, -400)), mode: 'lines', line: { color: CK(cs[i]), width: 0.9 }, name: `${NM[cs[i]]}: ${f === 'close' ? 'close' : 'daily low'}`, hovertemplate: `${NM[cs[i]]} %{x}: %{y:.1f} bp<extra></extra>` }));
    tr.push(on2(P.ax[i], { x: bad.map(j => s.d[j]), y: bad.map(j => Math.max(s.dev[j], -400)), mode: 'markers', marker: { color: RED, size: 3.5 }, name: `beyond ±${t} bp`, showlegend: i === 0, hoverinfo: 'skip' }));
    tr.push(on2(P.ax[i], htr(d0, d1, t, DARK, `±${t} bp`, 'dot', { showlegend: i === 0 })), on2(P.ax[i], htr(d0, d1, -t, DARK, '-', 'dot', { showlegend: false })));
    setAx(P, i, AX('', { showticklabels: i === 2 }), AX('bp', { range: [-310, 130] }));
  });
  P.L.xaxis2.matches = 'x'; P.L.xaxis3.matches = 'x';
  plot('pg-chart', tr, layout(Object.assign(P.L, { legend: LEG(false, -0.06), margin: M0 })));
  table('pg-tab', T.h.pg(t), S.map((s, i) => [ANT(cs[i]), fint(s.n), nf(s.mad, 1), nf(s.gtx, 1), nf(s.gt50, 2), nf(s.gt100, 2), `${fint(s.close_min)} (${DT(s.close_min_date)})`, `${fint(s.low_min)} (${DT(s.low_min_date)})`, nf(s.phi, 3), nf(s.half_life, 1)]));
  out('pg-out', [T.pg(f, t, S.map((s, i) => [AN(cs[i]), nf(s.mad, 1), nf(s.gtx, 1), nf(s.half_life, 1)]), f !== 'close' || t !== 10)]);
}

// ================================================================= 12. the USDC depeg, day by day
function svBlock() {
  const day = +$('sv-d').value, c = kSel('sv-c'), nw = single('sv-chart', 380), cs = c === 'all' ? ['usdc', 'dai', 'usdt'] : [c], ds = MAR.slice(0, day - 7); label('sv-d').textContent = day;
  const tr = [];
  cs.forEach(k => { tr.push({ x: ds, y: ds.map(s => pegAt(k, s, 'close')), mode: 'lines+markers', marker: { size: 7, symbol: 'circle' }, line: { color: CK(k), width: 1.4 }, name: `${NM[k]}: close`, hovertemplate: `${NM[k]} %{x}: close %{y:.4f}<extra></extra>` });
    tr.push({ x: ds, y: ds.map(s => pegAt(k, s, 'low')), mode: 'markers', marker: { size: 9, symbol: 'triangle-down', color: CK(k) }, name: `${NM[k]}: daily low`, hovertemplate: `${NM[k]} %{x}: low %{y:.4f}<extra></extra>` }); });
  tr.push(htr(MAR[0], MAR[12], 1, DARK, 'peg: 1 USD'));
  plot('sv-chart', tr, layout({ xaxis: AX('', { range: [addDays(MAR[0], -0.5), addDays(MAR[12], 0.5)], tickformat: '%d %b' }), yaxis: AX('price (USD)', { range: [0.86, 1.02] }), legend: LEG(nw, -0.12), margin: M0 }));
  const s = MAR[day - 8], bp = (k, f) => fbp(1e4 * (pegAt(k, s, f) - 1));
  out('sv-out', [T.sv(DT(s), ['usdc', 'dai', 'usdt'].map(k => [AN(k), nf(pegAt(k, s, 'close'), 4), bp(k, 'close'), nf(pegAt(k, s, 'low'), 4), bp(k, 'low')])), T.sv2(day)]);
}

// ================================================================= 13. the collapse of UST, May 2022 (DefiLlama, downloaded)
const UST = { s: { d: dated(LL.ust.supply), v: LL.ust.supply.v }, p: { d: dated(LL.ust.price), v: LL.ust.price.v } };
function usBlock() {
  const k = +$('us-d').value, s = addDays('2022-04-01', k), P = panels('us-chart', 2, { h: 360 }); label('us-d').textContent = DT(s);
  const cut = o => { const j = o.d.filter(x => x <= s).length; return { d: o.d.slice(0, j), v: o.v.slice(0, j) }; }, S = cut(UST.s), Pp = cut(UST.p);
  const tr = [on2(P.ax[0], { x: S.d, y: S.v, mode: 'lines+markers', marker: { size: 4 }, line: { color: BLUE, width: 1.4 }, name: 'UST supply (billion)', hovertemplate: '%{x}: %{y:.2f} bn<extra></extra>' }),
    on2(P.ax[1], { x: Pp.d, y: Pp.v, mode: 'lines+markers', marker: { size: 4 }, line: { color: RED, width: 1.2 }, name: 'UST price (USD)', hovertemplate: '%{x}: %{y:.4f} USD<extra></extra>' }),
    on2(P.ax[1], htr('2022-04-01', '2022-06-15', 1, DARK, 'peg: 1 USD'))];
  setAx(P, 0, AX('', { range: ['2022-03-30', '2022-06-17'], tickformat: '%d %b' }), AX('billion UST', { range: [0, 20.5] })); setAx(P, 1, AX('', { range: ['2022-03-30', '2022-06-17'], tickformat: '%d %b' }), AX('USD', { range: [0, 1.1] }));
  plot('us-chart', tr, layout(Object.assign(P.L, { legend: LEG(P.stack, P.stack ? -0.06 : -0.18), margin: M0 })));
  const pk = UST.s.v.reduce((m, v, i) => (v > UST.s.v[m] ? i : m), 0), sl = S.v.length ? S.v[S.v.length - 1] : NaN, pl = Pp.v.length ? Pp.v[Pp.v.length - 1] : NaN;
  out('us-out', [T.us(DT(s), Pp.d.length ? DT(Pp.d[Pp.d.length - 1]) : '–', nf(pl, 3), S.d.length ? DT(S.d[S.d.length - 1]) : '–', nf(sl, 1), nf(UST.s.v[pk], 1), DT(UST.s.d[pk]), S.d.length && S.d[S.d.length - 1] >= UST.s.d[pk] ? nf(100 * (1 - sl / UST.s.v[pk]), 0) : null)]);
}

// ================================================================= x16. VaR 1% and ES 2.5% in dollars (worked example)
const VT = {}; const vt = (k, a) => VT[k + a] || (VT[k + a] = varTab(k, a, 0.025));
const METHN = { hs_v: 'VaR 1% HS', n_v: 'VaR 1% Normal', t_v: 'VaR 1% Student-t', g_v: 'VaR 1% GARCH-t', hs_e: 'ES 2.5% HS', n_e: 'ES 2.5% Normal', t_e: 'ES 2.5% Student-t', g_e: 'ES 2.5% GARCH-t', hs10: '10-day VaR 1% HS' };
function udBlock() {
  const W = 1000 * +$('ud-W').value, k = kSel('ud-k'), m = kSel('ud-m'), nw = single('ud-chart', 340), vs = Array.from({ length: 81 }, (_, i) => i / 2), v = vt(k, 0.01)[m]; label('ud-W').textContent = fint(W) + ' USD';
  plot('ud-chart', [{ x: vs, y: vs.map(z => usd(z, W)), mode: 'lines', line: { color: BLUE, width: 2 }, name: 'loss W(1 − e^(−v/100))', hovertemplate: 'v = %{x}%: %{y:,.0f} USD<extra></extra>' },
    { x: vs, y: vs.map(z => W * z / 100), mode: 'lines', line: { color: AMBER, width: 1.4, dash: 'dash' }, name: 'linear approximation W v/100', hoverinfo: 'skip' },
    DIA([v], [usd(v, W)], RED, `${NM[k]}: ${METHN[m]}`, { mode: 'markers+text', text: [Math.round(usd(v, W)).toLocaleString('en-US')], textposition: 'top left', textfont: { size: 11, color: DARK }, cliponaxis: false })],
  layout({ xaxis: AX('loss as a log return, v (%)'), yaxis: AX('loss (USD)'), legend: LEG(nw, -0.18), margin: M0 }));
  steps('ud-we', [T.ud1(AN(k), T.meth[m], nf(v, 2)), T.ud2(nf(v, 2), nf(Math.exp(-v / 100), 4)), T.ud3(fint(W), nf(1 - Math.exp(-v / 100), 4), fint(usd(v, W))), T.ud4(fint(W * v / 100), fint(W * v / 100 - usd(v, W)))]);
  out('ud-out', [T.ud(fint(W), ['btc', 'eth', 'sp500'].map(a => [AN(a), fint(usd(vt(a, 0.01).hs_v, W)), fint(usd(vt(a, 0.01).hs_e, W))]), W !== 1e5)]);
}

// ================================================================= x17. four methods, three assets (tables)
function vqBlock() {
  const a = +kSel('vq-a'), nw = single('vq-chart', 360), cols = [BLUE, AMBER, PURPLE, RED], ms4 = ['hs', 'n', 't', 'g'], mn = ['HS', 'Normal', 'Student-t', 'GARCH-t'];
  const tr = ms4.map((m, i) => BAR(EA.map(k => NM[k]), EA.map(k => vt(k, a)[m + '_v']), cols[i], `VaR ${LVE(a)}%: ${mn[i]}`, { text: EA.map(k => vt(k, a)[m + '_v'].toFixed(2)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: DARK }, hovertemplate: `${mn[i]} %{x}: %{y:.2f}%<extra></extra>` }));
  tr.push({ x: EA.map(k => NM[k]), y: EA.map(k => vt(k, a).hs_e), mode: 'markers', marker: { color: DARK, size: 10, symbol: 'line-ew-open', line: { width: 3, color: DARK } }, name: 'ES 2.5%: HS', hovertemplate: 'ES 2.5% HS %{x}: %{y:.2f}%<extra></extra>' });
  plot('vq-chart', tr, layout({ barmode: 'group', xaxis: AX('', { type: 'category' }), yaxis: AX('one-day loss (%)'), legend: LEG(nw, -0.12), margin: M0 }));
  table('vq-tab', T.h.vq(LV(a)), EA.map(k => { const o = vt(k, a); return [ANT(k), ...['hs_v', 'n_v', 't_v', 'g_v', 'hs_e', 'n_e', 't_e', 'g_e'].map(c => nf(o[c], 2))]; }));
  table('vq-tab2', T.h.vq2(LV(a)), EA.map(k => { const o = vt(k, a); return [ANT(k), fint(usd(o.hs_v, 1e5)), fint(usd(o.hs_e, 1e5)), fint(usd(o.n_v, 1e5)), fint(usd(o.hs10, 1e5))]; }));
  const b = vt('btc', a), s = vt('sp500', a);
  out('vq-out', [T.vq(LV(a), nf(b.hs_v, 2), nf(s.hs_v, 2), nf(b.hs_v / s.hs_v, 1), nf(100 * (1 - b.n_v / b.hs_v), 0), nf(b.g_v, 2), nf(b.nu, 2), nf(b.g_sigma, 2), a !== 0.01)]);
}

// ================================================================= x18. the square-root-of-time rule (worked example)
const SQC = {};
function sqCurve(k) { if (SQC[k]) return SQC[k]; const x = vt(k, 0.01).x; return (SQC[k] = Array.from({ length: 30 }, (_, i) => hsVar(overl(x, i + 1), 0.01))); }
function sqBlock() {
  const h = +$('sq-h').value, k = kSel('sq-k'), nw = single('sq-chart', 340), hs = Array.from({ length: 30 }, (_, i) => i + 1), c = sqCurve(k), v1 = c[0]; label('sq-h').textContent = h;
  plot('sq-chart', [{ x: hs, y: c, mode: 'lines+markers', marker: { size: 5 }, line: { color: CK(k), width: 2 }, name: `${NM[k]}: HS VaR 1% of overlapping h-day returns`, hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    { x: hs, y: hs.map(j => Math.sqrt(j) * v1), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'square-root-of-time rule √h × VaR(1 day)', hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    DIA([h, h], [c[h - 1], Math.sqrt(h) * v1], [CK(k), DARK], `h = ${h}`, { mode: 'markers+text', text: [c[h - 1].toFixed(1), (Math.sqrt(h) * v1).toFixed(1)], textposition: ['bottom right', 'top left'], textfont: { size: 11, color: DARK }, cliponaxis: false })],
  layout({ xaxis: AX('horizon h (days)'), yaxis: AX('VaR 1% (%)'), legend: LEG(nw, -0.2), margin: M0 }));
  steps('sq-we', [T.sq1(AN(k), nf(v1, 2)), T.sq2(h, nf(Math.sqrt(h), 3), nf(v1, 2), nf(Math.sqrt(h) * v1, 2)), T.sq3(h, nf(c[h - 1], 2), fint(usd(c[h - 1], 1e5))), T.sq4(nf(Math.abs(100 * (Math.sqrt(h) * v1 / c[h - 1] - 1)), 0), Math.sqrt(h) * v1 > c[h - 1], h)]);
  out('sq-out', [T.sq(h, AN(k), nf(c[h - 1], 2), nf(Math.sqrt(h) * v1, 2), h !== 10 || k !== 'btc')]);
}

// ================================================================= x19. backtesting: the Kupiec test (worked example)
function kpBlock() {
  const x = +$('kp-x').value, a = +kSel('kp-a'), n = BT.n, nw = single('kp-chart', 340), xs = Array.from({ length: 81 }, (_, i) => i), K = kupiec(x, n, a); label('kp-x').textContent = x;
  const LR = xs.map(z => kupiec(z, n, a).lr), ok = xs.filter((z, i) => LR[i] <= 3.841458820694124);
  plot('kp-chart', [{ x: xs, y: LR, mode: 'lines', line: { color: BLUE, width: 2 }, name: `LR_uc for n = ${n}, α = ${a}`, hovertemplate: 'x = %{x}: LR = %{y:.2f}<extra></extra>' },
    htr(0, 80, 3.841458820694124, RED, '5% critical value 3.84', 'dash'),
    DIA([x], [K.lr], AMBER, `x = ${x}`, { mode: 'markers+text', text: [K.lr.toFixed(2)], textposition: 'top right', textfont: { size: 11, color: DARK }, cliponaxis: false }),
    { x: [a * n, a * n], y: [0, 20], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dot' }, name: `expected αn = ${(a * n).toFixed(1)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('number of exceptions x'), yaxis: AX('LR_uc', { range: [0, 20] }), legend: LEG(nw, -0.18), margin: M0 }));
  steps('kp-we', [T.kp1(x, fint(n), nf(100 * x / n, 2), nf(a * n, 1)), T.kp2(nf(K.l0, 2), nf(K.l1, 2)), T.kp3(nf(K.lr, 2), SIG(K.p), K.lr > 3.841458820694124), T.kp4(ok[0], ok[ok.length - 1])]);
  out('kp-out', [T.kp(x, nf(K.lr, 2), SIG(K.p), ok[0], ok[ok.length - 1], a !== 0.01 || (x !== 29 && x !== 41))]);
}

// ================================================================= 14. backtesting Bitcoin VaR 1%
function btBlock() {
  const w = +$('bt-w').value, a = +kSel('bt-a'), B = backtest(w, a), nw = single('bt-chart', 400), H = B.HS, G = B['GARCH-t']; label('bt-w').textContent = w;
  const hit = H.hit.map((h, j) => j).filter(j => H.hit[j]);
  plot('bt-chart', [{ x: B.d, y: B.r, mode: 'lines', line: { color: BLUE, width: 0.5 }, opacity: 0.8, name: 'Bitcoin daily return (%)', hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: B.d, y: H.v.map(v => -v), mode: 'lines', line: { color: AMBER, width: 1.3 }, name: `minus VaR ${LVE(a)}%: historical simulation, ${w} days`, hovertemplate: 'HS %{x}: %{y:.2f}%<extra></extra>' },
    { x: B.d, y: G.v.map(v => -v), mode: 'lines', line: { color: RED, width: 1 }, name: `minus VaR ${LVE(a)}%: GARCH(1,1)-t`, hovertemplate: 'GARCH-t %{x}: %{y:.2f}%<extra></extra>' },
    { x: hit.map(j => B.d[j]), y: hit.map(j => B.r[j]), mode: 'markers', marker: { color: ORANGE, size: 6 }, name: 'exception of the historical VaR', hoverinfo: 'skip' }],
  layout({ xaxis: AX(''), yaxis: AX('%', { range: [-45, 25] }), legend: LEG(nw, -0.12), margin: M0 }));
  const f = o => [o.x, nf(o.rate, 2), nf(o.exp, 1), nf(o.lr, 2), nf(o.p, 2), nf(o.mean_var, 1)];
  out('bt-out', [T.bt(w, LV(a), fint(B.n), f(H), f(G), G.year['2018'], H.year['2018'], w !== 365 || a !== 0.01)]);
}

// ================================================================= wiring
const PFX = { halving: 'hv', 'ann-we': 'an', 'desc-tab': 'ds', rvol: 'rv', weekday: 'wd', 'hill-tab': 'hl', 'lb-we': 'lb', acf: 'ac', 'garch-tab': 'gt', garch: 'gv', 'vr-tab': 'vt', rvr: 'vr',
  'rc-we': 'rc', corr: 'co', 'corr-tab': 'cy', 'crisis-tab': 'cr', 'haven-we': 'sh', 'mdd-we': 'md', dd: 'dd', 'dd-tab': 'de', etf: 'ef', 'etf-tab': 'et', 'ibit-we': 'ib', supply: 'su',
  'bp-we': 'bp', peg: 'pg', svb: 'sv', ust: 'us', 'usd-we': 'ud', 'var-tab': 'vq', 'sqrt-we': 'sq', 'kup-we': 'kp', btcvar: 'bt' };
const BLOCKS = { hvBlock, anBlock, dsBlock, rvBlock, wdBlock, hlBlock, lbBlock, acBlock, gtBlock, gvBlock, vtBlock, vrBlock, rcBlock, coBlock, cyBlock, crBlock, shBlock, mdBlock, ddBlock, deBlock,
  efBlock, etBlock, ibBlock, suBlock, bpBlock, pgBlock, svBlock, usBlock, udBlock, vqBlock, sqBlock, kpBlock, btBlock };
const PLAYERS = {};
function placeExtras() {                // web lecture: the table blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
const YR = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => String(a + i));
function init() {
  placeExtras();
  const on = (ids, f) => ids.forEach(id => { $(id).addEventListener('input', f); if ($(id).tagName === 'SELECT') $(id).addEventListener('change', f); });
  const PLy = (btn, id, vals, ms_) => (PLAYERS[btn] = new Player(btn, $(id), vals, ms_));
  on(['hv-h'], hvBlock); PLy('hv-play', 'hv-h', [365, 30, 90, 180, 270, 365, 545, 730, 365], 1000);
  on(['an-A', 'an-k'], anBlock); PLy('an-play', 'an-A', [365, 200, 252, 260, 300, 330, 360, 400, 365], 900);
  on(['ds-s'], dsBlock); PLy('ds-play', 'ds-s', ['full', '2020-04-11', '2022-01-01', '2024-01-01', 'full'], 1500);
  on(['rv-w', 'rv-a'], rvBlock); PLy('rv-play', 'rv-w', [30, 7, 14, 60, 90, 180, 30], 1100);
  on(['wd-y', 'wd-m'], wdBlock); PLy('wd-play', 'wd-y', ['none', ...YR(2015, 2026), 'none'], 900);
  on(['hl-k'], hlBlock); PLy('hl-play', 'hl-k', ['0.025', '0.01', '0.05', '0.1', '0.025'], 1300);
  on(['lb-m', 'lb-k', 'lb-x'], lbBlock); PLy('lb-play', 'lb-m', [10, 1, 2, 3, 5, 15, 20, 25, 30, 10], 800);
  on(['ac-k', 'ac-L'], acBlock); PLy('ac-play', 'ac-k', ['btc', 'eth', 'sol', 'sp500', 'gold', 'btc'], 1400);
  on(['gt-h', 'gt-k'], gtBlock); PLy('gt-play', 'gt-h', [100, 0, 10, 30, 60, 113, 200, 300, 400, 100], 800);
  on(['gv-h', 'gv-k'], gvBlock); PLy('gv-play', 'gv-h', [0, 10, 30, 60, 100, 200, 365, 0], 1000);
  on(['vt-q'], vtBlock); PLy('vt-play', 'vt-q', ['5', '2', '10', '20', '5'], 1300);
  on(['vr-q', 'vr-w', 'vr-z'], vrBlock); PLy('vr-play', 'vr-w', [500, 250, 350, 750, 1000, 500], 1300);
  on(['rc-w', 'rc-p', 'rc-e'], rcBlock); PLy('rc-play', 'rc-w', [250, 60, 125, 375, 500, 250], 1100);
  on(['co-w'], coBlock); PLy('co-play', 'co-w', [250, 60, 125, 375, 500, 250], 1300);
  on(['cy-y', 'cy-f'], cyBlock); PLy('cy-play', 'cy-y', YR(2014, 2026).concat(['2022']), 800);
  on(['cr-e', 'cr-x'], crBlock); PLy('cr-play', 'cr-e', ['COVID-19', 'Terra/Luna', 'FTX', 'COVID-19'], 1500);
  on(['sh-q', 'sh-k'], shBlock); PLy('sh-play', 'sh-q', [5, 1, 2.5, 10, 15, 20, 5], 1100);
  on(['md-d', 'md-k'], mdBlock); PLy('md-play', 'md-d', [75, 10, 25, 50, 80, 90, 75], 900);
  on(['dd-y', 'dd-t'], ddBlock); PLy('dd-play', 'dd-y', [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2016], 900);
  on(['de-t'], deBlock); PLy('de-play', 'de-t', ['50', '30', '40', '60', '70', '50'], 1300);
  on(['ef-w', 'ef-y'], efBlock); PLy('ef-play', 'ef-y', ['none', ...YR(2020, 2026), 'none'], 1000);
  on(['et-L'], etBlock); PLy('et-play', 'et-L', ['2', '1', '1.5', '2.5', '2'], 1500);
  on(['ib-k', 'ib-s'], ibBlock); PLy('ib-play', 'ib-k', ['1', '2', '5', '10', '1'], 1300);
  on(['su-y', 'su-c'], suBlock); PLy('su-play', 'su-y', ['end', '2020-01-01', '2022-04-02', '2023-03-10', '2023-08-19', '2023-12-31', 'end'], 1300);
  on(['bp-d', 'bp-k', 'bp-p', 'bp-W'], bpBlock); PLy('bp-play', 'bp-d', [11, 8, 9, 10, 12, 13, 14, 15, 17, 20, 11], 900);
  on(['pg-t', 'pg-p'], pgBlock); PLy('pg-play', 'pg-t', ['10', '5', '25', '50', '100', '10'], 1400);
  on(['sv-d', 'sv-c'], svBlock); PLy('sv-play', 'sv-d', [20, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20], 900);
  on(['us-d'], usBlock); PLy('us-play', 'us-d', [75, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 50, 60, 75], 800);
  on(['ud-W', 'ud-k', 'ud-m'], udBlock); PLy('ud-play', 'ud-W', [100, 10, 50, 250, 500, 1000, 100], 1000);
  on(['vq-a'], vqBlock); PLy('vq-play', 'vq-a', ['0.01', '0.025', '0.05', '0.01'], 1500);
  on(['sq-h', 'sq-k'], sqBlock); PLy('sq-play', 'sq-h', [10, 1, 2, 5, 15, 20, 30, 10], 900);
  on(['kp-x', 'kp-a'], kpBlock); PLy('kp-play', 'kp-x', [29, 10, 15, 20, 25, 32, 35, 41, 45, 50, 29], 800);
  on(['bt-w', 'bt-a'], btBlock); PLy('bt-play', 'bt-w', [365, 100, 180, 250, 500, 730, 1000, 365], 1300);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH14 = { Phi, Pinv, chi2sf, chi2ppf, t2sf, tppf, ibeta, gammaQ, describe, rvol, weekday, acf, ljung, vratio, rollVR, rollCorr, weekly, crisis, worst, drawdown, ddEpisodes, mddOf, etfCmp,
  weShareYear, ibit, ibitS, pegStats, hsVar, hsEs, varTab, kupiec, backtest, halving, supplyAt, yearCorr, rcSeries, sqCurve, lbCurves, hillAll, hill, subOf, rets, joint, closes, PX, PEGD, pegAt, supAt, UST,
  init, PFX, blocks: BLOCKS };
if (window.CH14_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
