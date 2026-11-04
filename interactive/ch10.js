// Chapter 10 interactive page: one block per chart slide and per results table or worked example of the lecture, computed in the browser from ch10_data.js.
(function () {
'use strict';
const D = window.CH10_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, LANG } = window.SFMI;
const T = window.CH10_TEXT;
const NUM = D.N, COL = D.col, NAME = D.name, MK = D.names, BTK = D.bt_names, MCOL = D.mcol, METH = D.methods;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';
const SH = { sp500: 'S&P 500', dax: 'DAX', bet: 'BET', btc: 'Bitcoin', tlv: 'TLV', snp: 'SNP' };
const ZC = { green: GREEN, yellow: AMBER, red: RED };

// ================================================================= special functions
function erfc(x) {                  // Numerical Recipes erfcc: fractional error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
const npdf = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function normppf(p) {               // Acklam, refined by one Halley step
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
function gammln(z) {                // Lanczos
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
  for (const c of g) ser += c / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function gammq(a, x) {              // regularized upper incomplete gamma Q(a, x)
  if (x <= 0) return 1;
  if (x < a + 1) { let ap = a, s = 1 / a, del = s; for (let n = 0; n < 500; n++) { ap++; del *= x / ap; s += del; if (Math.abs(del) < Math.abs(s) * 1e-15) break; } return 1 - s * Math.exp(-x + a * Math.log(x) - gammln(a)); }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
  return Math.exp(-x + a * Math.log(x) - gammln(a)) * h;
}
const chi2sf = (x, df) => gammq(df / 2, x / 2);
function betacf(a, b, x) {          // continued fraction of the incomplete beta function (Numerical Recipes)
  const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 3e-16) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
function tcdf(t, v) { const p = 0.5 * betai(v / 2, 0.5, v / (v + t * t)); return t > 0 ? 1 - p : p; }
const tpdfC = v => Math.exp(gammln((v + 1) / 2) - gammln(v / 2)) / Math.sqrt(v * Math.PI);
const tpdf = (x, v) => tpdfC(v) * Math.pow(1 + x * x / v, -(v + 1) / 2);
function tppf(p, v) {               // quantile of the Student-t with v degrees of freedom (Newton from the Normal quantile, with bisection safeguard)
  if (p > 0.5) return -tppf(1 - p, v);
  const k = tpdfC(v); let x = normppf(p), lo = -1e6, hi = 0;
  for (let i = 0; i < 80; i++) {
    const f = tcdf(x, v) - p; if (Math.abs(f) < 1e-15) break;
    if (f > 0) hi = x; else lo = x;
    let xn = x - f / (k * Math.pow(1 + x * x / v, -(v + 1) / 2));
    if (!(xn > lo && xn < hi)) xn = lo < -1e5 ? 2 * x - 1 : (lo + hi) / 2;
    x = xn;
  }
  return x;
}
const tEsF = (v, a) => { const q = tppf(a, v); return tpdf(q, v) * (v + q * q) / ((v - 1) * a); };   // ES of t_v (scale 1)
const stdtQ = (v, a) => tppf(a, v) * Math.sqrt((v - 2) / v);
const stdtEs = (v, a) => Math.sqrt((v - 2) / v) * tEsF(v, a);
const lbin = (n, k) => gammln(n + 1) - gammln(k + 1) - gammln(n - k + 1);
const binpmf = (k, n, p) => Math.exp(lbin(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
const bincdf = (k, n, p) => { let s = 0; for (let j = 0; j <= k; j++) s += binpmf(j, n, p); return s; };
function binppf(q, n, p) { let s = 0; for (let k = 0; k <= n; k++) { s += binpmf(k, n, p); if (s >= q - 1e-12) return k; } return n; }

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpack(R) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); return dd.map(iso); }
const mean = x => { let s = 0; for (const v of x) s += v; return s / x.length; };
const sdev = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / (x.length - 1)); };
function npq(sorted, p) { const h = (sorted.length - 1) * p, i = Math.floor(h); return i + 1 < sorted.length ? sorted[i] + (h - i) * (sorted[i + 1] - sorted[i]) : sorted[i]; }
const kOf = (n, a) => Math.ceil(n * a - 1e-9);
const hsVar = (s, a) => -s[kOf(s.length, a) - 1];                  // s sorted ascending
const hsEs = (s, a) => { const k = kOf(s.length, a); let e = 0; for (let i = 0; i < k; i++) e += s[i]; return -e / k; };
const SER = {};
function ser(k) {
  if (SER[k]) return SER[k];
  const x = D.ret[k].x, n = x.length, mu = mean(x), sd = sdev(x), s = Float64Array.from(x).sort();
  let m2 = 0, m3 = 0, m4 = 0; for (const v of x) { const e = v - mu; m2 += e * e; m3 += e * e * e; m4 += e ** 4; } m2 /= n; m3 /= n; m4 /= n;
  return (SER[k] = { x, n, dates: unpack(D.ret[k]), s, mu, sd, S: m3 / m2 ** 1.5, K: m4 / (m2 * m2) - 3 });
}
const sortedLast = (k, nl) => { const S = ser(k); return nl ? Float64Array.from(S.x.slice(S.n - nl)).sort() : S.s; };

// ================================================================= VaR and ES: the methods of the chapter
const normVar = (mu, sd, a) => -(mu + sd * normppf(a));
const normEs = (mu, sd, a) => -mu + sd * npdf(normppf(a)) / a;
function cfz(a, S, K) { const z = normppf(a); return z + (z * z - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S * S / 36; }
function cfEsZ(a, S, K) { let s = 0; for (let i = 0; i < 2000; i++) s += cfz((i + 0.5) / 2000 * a, S, K); return s / 2000; }
const potVar = (f, a) => (a < f.Nu / f.n ? f.u + f.beta / f.xi * ((f.n * a / f.Nu) ** (-f.xi) - 1) : NaN);
const potEs = (f, a) => (potVar(f, a) + f.beta - f.xi * f.u) / (1 - f.xi);
const potOf = (k, q) => D.pot[k][String(q)];
function methods(k, a, ae) {         // methods_table of the chapter at the levels a (VaR) and ae (ES)
  const S = ser(k), t = NUM.methods[k].t, f = potOf(k, 0.9);
  return { HS: { var: hsVar(S.s, a), es: hsEs(S.s, ae) }, Normal: { var: normVar(S.mu, S.sd, a), es: normEs(S.mu, S.sd, ae) },
    'Student-t': { var: -(t.m + t.s * tppf(a, t.nu)), es: -t.m + t.s * tEsF(t.nu, ae) },
    'Cornish-Fisher': { var: -(S.mu + S.sd * cfz(a, S.S, S.K)), es: -(S.mu + S.sd * cfEsZ(ae, S.S, S.K)) },
    EVT: { var: potVar(f, a), es: potEs(f, ae) } };
}
function cond(k, a, ae) {           // conditional_next_day of the chapter: GARCH(1,1)-t and FHS
  const c = NUM.methods[k].cond, z = D.cond[k], zn = z.n, kz = kOf(zn, a), kze = kOf(zn, ae);
  const qt = stdtQ(c.nu, a), et = stdtEs(c.nu, ae), qz = z.z[kz - 1]; let ez = 0; for (let i = 0; i < kze; i++) ez += z.z[i]; ez = -ez / kze;
  return { c, qt, et, qz, ez, kz, kze, vg: -(c.mu + c.sigma * qt), eg: -c.mu + c.sigma * et, vf: -(c.mu + c.sigma * qz), ef: -c.mu + c.sigma * ez };
}

// ================================================================= rolling forecasts and backtests
const FR = {};
function frame(k) {                 // rolling_forecasts of the chapter, rebuilt: HS and Normal on the last 500 days, GARCH-t and FHS from the exported fits
  if (FR[k]) return FR[k];
  const S = ser(k), B = D.bt[k], i0 = B.i0, n = B.sig.length, W = D.window, k1 = kOf(W, 0.01), k25 = kOf(W, 0.025);
  const f = { r: S.x.slice(i0, i0 + n), dates: S.dates.slice(i0, i0 + n), n };
  ['var1', 'var25', 'es25'].forEach(c => METH.forEach(m => { f[c + '_' + m] = new Float64Array(n); }));
  const z1 = normppf(0.01), z25 = normppf(0.025), ph = npdf(z25) / 0.025, w = new Float64Array(W);
  for (let t = 0; t < n; t++) {
    let m = 0; for (let j = 0; j < W; j++) { w[j] = S.x[i0 + t - W + j]; m += w[j]; } m /= W;
    let v = 0; for (let j = 0; j < W; j++) v += (w[j] - m) ** 2; const s = Math.sqrt(v / (W - 1));
    w.sort(); let e = 0; for (let j = 0; j < k25; j++) e += w[j];
    f.var1_HS[t] = -w[k1 - 1]; f.var25_HS[t] = -w[k25 - 1]; f.es25_HS[t] = -e / k25;
    f.var1_Normal[t] = -(m + s * z1); f.var25_Normal[t] = -(m + s * z25); f.es25_Normal[t] = -m + s * ph;
  }
  B.blocks.forEach((b, j) => {
    const e = j + 1 < B.blocks.length ? B.blocks[j + 1][0] : n, [, mu, nu, q1, q25, ez] = b, sc = Math.sqrt((nu - 2) / nu), t1 = tppf(0.01, nu), t25 = tppf(0.025, nu);
    const ef = sc * tpdf(t25, nu) * (nu + t25 * t25) / ((nu - 1) * 0.025);
    for (let t = b[0]; t < e; t++) {
      const sg = B.sig[t];
      f['var1_GARCH-t'][t] = -(mu + sg * t1 * sc); f['var25_GARCH-t'][t] = -(mu + sg * t25 * sc); f['es25_GARCH-t'][t] = -mu + sg * ef;
      f.var1_FHS[t] = -(mu + sg * q1); f.var25_FHS[t] = -(mu + sg * q25); f.es25_FHS[t] = -mu + sg * ez;
    }
  });
  return (FR[k] = f);
}
const vcol = (a, m) => (a === 0.025 ? 'var25_' : 'var1_') + m;
function hits(k, m, a) { const f = frame(k), v = f[vcol(a, m)], h = new Uint8Array(f.n); for (let t = 0; t < f.n; t++) h[t] = f.r[t] < -v[t] ? 1 : 0; return h; }
function kupiec(x, n, a) {
  const ph = x / n, l0 = (n - x) * Math.log(1 - a) + x * Math.log(a), l1 = (ph < 1 ? (n - x) * Math.log(1 - ph) : 0) + (x > 0 ? x * Math.log(ph) : 0), lr = -2 * (l0 - l1);
  return { x, n, rate: ph, l0, l1, lr, p: chi2sf(Math.max(lr, 0), 1) };
}
function christ(h, a) {
  let n00 = 0, n01 = 0, n10 = 0, n11 = 0, x = 0;
  for (let t = 0; t < h.length; t++) { x += h[t]; if (t) { if (h[t - 1]) { if (h[t]) n11++; else n10++; } else if (h[t]) n01++; else n00++; } }
  const p01 = n00 + n01 ? n01 / (n00 + n01) : 0, p11 = n10 + n11 ? n11 / (n10 + n11) : 0, p = (n01 + n11) / (n00 + n01 + n10 + n11);
  const ll = (q, a_, b_) => (a_ && q < 1 ? a_ * Math.log(1 - q) : 0) + (b_ && q > 0 ? b_ * Math.log(q) : 0);
  const lri = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11)), uc = kupiec(x, h.length, a), lrc = uc.lr + lri;
  return { n00, n01, n10, n11, p01, p11, p, lri, pi: chi2sf(Math.max(lri, 0), 1), lrc, pc: chi2sf(Math.max(lrc, 0), 2), uc };
}
function roll250(h) { const c = new Int16Array(h.length); let s = 0; for (let t = 0; t < h.length; t++) { s += h[t]; if (t >= 250) s -= h[t - 250]; c[t] = s; } return c; }   // c[t] valid from t = 249
function zone(x) { return x <= 4 ? 'green' : x <= 9 ? 'yellow' : 'red'; }
const plusOf = x => (x >= 10 ? 1 : D.plus[String(x)]);
function z2stat(k, m, j0) { const f = frame(k), v = f['var25_' + m], e = f['es25_' + m]; let s = 0; for (let t = j0; t < f.n; t++) if (f.r[t] < -v[t]) s += f.r[t] / e[t]; return s / ((f.n - j0) * 0.025) + 1; }
function qloss(k, m, j0, j1) { const f = frame(k), v = f['var1_' + m]; let s = 0; for (let t = j0; t < j1; t++) s += (0.01 - (f.r[t] < -v[t] ? 1 : 0)) * (f.r[t] + v[t]); return s / (j1 - j0); }
const firstFrom = (dates, d) => { const i = dates.findIndex(v => v >= d); return i < 0 ? dates.length : i; };
function btRow(k, m, j0) {           // backtest() of the chapter on the days from j0
  const h = hits(k, m, 0.01).slice(j0), c = christ(h, 0.01), r = roll250(h); let mx = 0; for (let t = 249; t < h.length; t++) mx = Math.max(mx, r[t]);
  return { c, x: c.uc.x, n: h.length, rate: c.uc.rate, mx, z2: z2stat(k, m, j0) };
}

// ================================================================= portfolio
let PO = null;
function port() {
  if (PO) return PO;
  const P = D.port, dates = unpack(P), a = P.bet, b = P.sp500, n = a.length;
  return (PO = { dates, a, b, n });
}
function corrOf(a, b) { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; } return sab / Math.sqrt(saa * sbb); }
function between(P, d0, d1) { const i = [], s = d0 || '0000', e = d1 || '9999'; P.dates.forEach((d, j) => { if (d >= s && d <= e) i.push(j); }); return { a: i.map(j => P.a[j]), b: i.map(j => P.b[j]), dates: i.map(j => P.dates[j]) }; }
function rankAvg(x) {               // pandas rank(): average ranks of ties
  const n = x.length, idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => x[i] - x[j]), r = new Float64Array(n);
  for (let i = 0; i < n;) { let j = i; while (j + 1 < n && x[idx[j + 1]] === x[idx[i]]) j++; const v = (i + j) / 2 + 1; for (let t = i; t <= j; t++) r[idx[t]] = v; i = j + 1; }
  return r;
}
const UC = {};
function pobs(per) {
  if (UC[per]) return UC[per];
  const P = port(), [d0, d1] = PER[per], X = between(P, d0, d1), n = X.a.length, ra = rankAvg(X.a), rb = rankAvg(X.b);
  return (UC[per] = { n, u: Array.from(ra, v => v / (n + 1)), v: Array.from(rb, v => v / (n + 1)), X });
}
const PER = { all: ['', ''], p1: ['2000-01-01', '2009-12-31'], p2: ['2010-01-01', '2019-12-31'], p3: ['2020-01-01', '2026-12-31'] };
function chiQ(U, q) { let c = 0; for (let i = 0; i < U.n; i++) if (U.u[i] <= q && U.v[i] <= q) c++; return c / U.n / q; }
function joint1(X) { const sa = Float64Array.from(X.a).sort(), sb = Float64Array.from(X.b).sort(), qa = npq(sa, 0.01), qb = npq(sb, 0.01); let c = 0; for (let i = 0; i < X.a.length; i++) if (X.a[i] <= qa && X.b[i] <= qb) c++; return c / X.a.length; }
const tailDep = (rho, nu) => 2 * tcdf(-Math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1);
function rollCorr(P, w) {
  const out = [], dts = [];
  for (let t = w - 1; t < P.n; t++) { out.push(corrOf(P.a.slice(t - w + 1, t + 1), P.b.slice(t - w + 1, t + 1))); dts.push(P.dates[t]); }
  return { c: out, dates: dts };
}

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const fp = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const MON = { ro: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
  en: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] };
const fdate = d => `${+d.slice(8, 10)} ${MON[LANG][+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const lev = a => nf(100 * a, Math.round(a * 1000) % 10 ? 1 : 0) + '%';
const qpc = q => nf(100 * q, Math.round(q * 1000) % 10 ? 1 : 0);            // 0.01 -> 1%, 0.025 -> 2,5%
const levE = a => `${+(100 * a).toFixed(2)}%`;                          // chart text (English)
function table(id, head, rows) {
  const cell = (x, tag) => { const o = typeof x === 'object' && x !== null ? x : { v: x }; return `<${tag}${o.c ? ` class="${o.c}"` : ''}${o.span ? ` colspan="${o.span}"` : ''}>${o.v}</${tag}>`; };
  const hs = Array.isArray(head[0]) ? head : [head];
  $(id).innerHTML = `<table class="xt"><thead>${hs.map(h => `<tr>${h.map(x => cell(x, 'th')).join('')}</tr>`).join('')}</thead><tbody>` +
    rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.map(x => cell(x, 'td')).join('')}</tr>`).join('') + '</tbody></table>';
}
const steps = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
function panels(id, n, o) {
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
function vstack(id, n, h, gap) {
  $(id).style.height = h + 'px';
  const g = gap || 0.06, w = (1 - g * (n - 1)) / n, L = {}, ax = [];
  for (let i = 0; i < n; i++) {
    const s = i ? String(i + 1) : '';
    L['xaxis' + s] = { anchor: 'y' + s, domain: [0, 1] }; if (i) L['xaxis' + s].matches = 'x';
    L['yaxis' + s] = { anchor: 'x' + s, domain: [1 - (i * (w + g) + w), 1 - i * (w + g)] };
    ax.push({ x: 'x' + s, y: 'y' + s, xa: 'xaxis' + s, ya: 'yaxis' + s });
  }
  return { L, ax, stack: true };
}
const ptitle = (a, text) => ({ text, xref: a.x + ' domain', yref: a.y + ' domain', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { size: 12, color: C.navy } });
function setAx(Pn, i, xa, ya) { Object.assign(Pn.L[Pn.ax[i].xa], xa); Object.assign(Pn.L[Pn.ax[i].ya], ya); }
const label = id => $(id + '-v');
const narrow = id => $(id).clientWidth < 560;
function single(id, h) { const n = narrow(id); $(id).style.height = ((h || 380) + (n ? 150 : 0)) + 'px'; return n; }
const LEG = (n, y) => Object.assign(layout().legend, { y: (n ? y - 0.04 : y) - 0.03 });
const on2 = (a, t) => ({ xaxis: a.x, yaxis: a.y, ...t });
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const vl = (a, x, y0, y1, color, name, dash, show) => on2(a, { x: [x, x], y: [y0, y1], mode: 'lines', line: { color, width: 2, dash: dash || 'solid' }, name, showlegend: show !== false, hoverinfo: 'skip' });
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const kSel = id => $(id).value;
const EPC = AMBER, epShapes = (a, d0) => Object.values(D.episodes).filter(([, e]) => e >= d0).map(([s, e]) =>
  ({ type: 'rect', xref: a.x, yref: a.y + ' domain', x0: s, x1: e, y0: 0, y1: 1, fillcolor: hexA(EPC, 0.15), line: { width: 0 }, layer: 'below' }));
const ALPHAS = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05];
const LOGX = { type: 'log', autorange: 'reversed', tickvals: [0.1, 0.25, 0.5, 1, 2.5, 5], ticktext: ['0.1', '0.25', '0.5', '1', '2.5', '5'] };

// ================================================================= 1. VaR 1% and ES 2.5% of the S&P 500
function dfBlock() {
  const k = kSel('df-s'), a = +kSel('df-v'), ae = +kSel('df-e'), S = ser(k), nw = single('df-chart', 360);
  const v = hsVar(S.s, a), q = -hsVar(S.s, ae), e = hsEs(S.s, ae), lo = Math.floor(S.s[0]), hi = Math.ceil(S.s[S.n - 1]);
  const bw = k === 'btc' ? 0.5 : 0.25, nb = Math.ceil((hi - lo) / bw), cnt = new Array(nb).fill(0);
  for (const r of S.x) cnt[Math.min(nb - 1, Math.floor((r - lo) / bw))]++;
  const xs = cnt.map((_, i) => lo + (i + 0.5) * bw), dens = cnt.map(c => c / S.n / bw), top = Math.max(...dens);
  let im = 0; S.x.forEach((r, i) => { if (r < S.x[im]) im = i; });
  plot('df-chart', [{ type: 'bar', x: xs, y: dens, width: bw, marker: { color: hexA(BLUE, 0.75) }, name: `daily log returns of the ${NAME[k]} (%)`, hovertemplate: '%{x:.2f}%: %{y:.3f}<extra></extra>' },
    { x: [lo, q, q, lo], y: [0, 0, top, top], fill: 'toself', fillcolor: hexA(RED, 0.08), line: { width: 0 }, mode: 'lines', name: `worst ${levE(ae)} of the days`, hoverinfo: 'skip' },
    { x: [-v, -v], y: [0, top], mode: 'lines', line: { color: RED, width: 2 }, name: `minus VaR ${levE(a)} = ${(-v).toFixed(2)}%`, hoverinfo: 'skip' },
    { x: [-e, -e], y: [0, top], mode: 'lines', line: { color: PURPLE, width: 2, dash: 'dash' }, name: `minus ES ${levE(ae)} = ${(-e).toFixed(2)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('daily return (%)', { range: [lo, hi] }), yaxis: AX('density'), bargap: 0, legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  let nt = 0; for (const r of S.x) if (r < -v) nt++;
  out('df-out', [T.df(NAME[k], fint(S.n), lev(a), nf(v, 2), kOf(S.n, a), lev(ae), nf(-q, 2), nf(e, 2), kOf(S.n, ae), nt, nf(S.x[im], 2), fdate(S.dates[im]),
    fint(1e4 * v), fint(1e4 * e), k !== 'sp500' || a !== 0.01 || ae !== 0.025)]);
}

// ================================================================= x1. Normal VaR and ES (worked example)
function nwBlock() {
  const k = kSel('nw-s'), a = +kSel('nw-a'), ae = +kSel('nw-e'), S = ser(k), nw = single('nw-chart', 340);
  const vn = normVar(S.mu, S.sd, a), en = normEs(S.mu, S.sd, ae), vh = hsVar(S.s, a), eh = hsEs(S.s, ae), z = normppf(a), ze = normppf(ae), fe = npdf(ze) / ae;
  const lo = Math.floor(S.s[0]), bw = k === 'btc' ? 0.5 : 0.25, hi = 0, nb = Math.ceil((hi - lo) / bw), cnt = new Array(nb).fill(0);
  for (const r of S.x) if (r < hi) cnt[Math.min(nb - 1, Math.floor((r - lo) / bw))]++;
  const xs = cnt.map((_, i) => lo + (i + 0.5) * bw), dens = cnt.map(c => c / S.n / bw), xg = linspace(lo, hi, 300), top = Math.max(...dens) * 1.05;
  plot('nw-chart', [{ type: 'bar', x: xs, y: dens, width: bw, marker: { color: hexA(BLUE, 0.7) }, name: `${NAME[k]}: left half of the histogram`, hovertemplate: '%{x:.2f}%: %{y:.4f}<extra></extra>' },
    { x: xg, y: xg.map(x => npdf((x - S.mu) / S.sd) / S.sd), mode: 'lines', line: { color: GREEN, width: 2 }, name: 'Normal density, sample mean and standard deviation', hoverinfo: 'skip' },
    { x: [-vn, -vn], y: [0, top], mode: 'lines', line: { color: GREEN, width: 2, dash: 'dash' }, name: `minus Normal VaR ${levE(a)} = ${(-vn).toFixed(2)}%`, hoverinfo: 'skip' },
    { x: [-vh, -vh], y: [0, top], mode: 'lines', line: { color: RED, width: 2 }, name: `minus historical VaR ${levE(a)} = ${(-vh).toFixed(2)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('daily return (%)'), yaxis: AX('density', { type: 'log', range: [-4, Math.log10(top)] }), bargap: 0, legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  steps('nw-we', [T.nw1(NAME[k], fint(S.n), nf(S.mu, 3), nf(S.sd, 3)), T.nw2(lev(a), nf(S.mu, 3), nf(-z, 3), nf(S.sd, 3), nf(-z * S.sd, 3), nf(vn, 2)),
    T.nw3(lev(ae), nf(-ze, 3), nf(npdf(ze), 4), nf(fe, 3), nf(S.mu, 3), nf(S.sd, 3), nf(fe * S.sd, 3), nf(en, 2), nf(fe / -normppf(0.01), 3)),
    T.nw4(lev(a), nf(vh, 2), lev(ae), nf(eh, 2), nf(100 * (1 - vn / vh), 0), nf(100 * (1 - en / eh), 0))]);
  out('nw-out', [T.nw(NAME[k], lev(a), nf(vn, 2), nf(vh, 2), lev(ae), nf(en, 2), nf(eh, 2), k !== 'sp500' || a !== 0.01 || ae !== 0.025)]);
}

// ================================================================= x2. from one day to ten days (extra chart)
const HZ = {};
function hzCurve(k) {
  if (HZ[k]) return HZ[k];
  const S = ser(k), v = [];
  for (let h = 1; h <= 20; h++) { const s = []; let c = 0; for (let t = 0; t < S.n; t++) { c += S.x[t]; if (t >= h) c -= S.x[t - h]; if (t >= h - 1) s.push(c); } v.push(hsVar(Float64Array.from(s).sort(), 0.01)); }
  return (HZ[k] = v);
}
function hzBlock() {
  const k = kSel('hz-s'), h = +$('hz-h').value, S = ser(k), V = hzCurve(k), hs = range(1, 20, 1), nw = single('hz-chart', 340); label('hz-h').textContent = String(h);
  const vn = hs.map(j => normVar(j * S.mu, Math.sqrt(j) * S.sd, 0.01));
  plot('hz-chart', [{ x: hs, y: V, mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 5 }, name: 'historical VaR 1% of overlapping h-day returns', hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    { x: hs, y: hs.map(j => Math.sqrt(j) * V[0]), mode: 'lines', line: { color: ORANGE, width: 1.6, dash: 'dash' }, name: 'sqrt(h) x one-day historical VaR 1%', hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    { x: hs, y: vn, mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dot' }, name: 'Normal VaR 1% (mean h mu, sd sqrt(h) sigma)', hovertemplate: 'h = %{x}: %{y:.2f}%<extra></extra>' },
    { x: [h], y: [V[h - 1]], mode: 'markers', marker: { color: DARK, size: 11 }, name: `h = ${h}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('horizon h (days)', { dtick: 2 }), yaxis: AX(`${NAME[k]} VaR 1% (%)`, { rangemode: 'tozero' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  let c0 = 0, c1 = 0; for (let t = 0; t < S.n; t++) { c0 += (S.x[t] - S.mu) ** 2; if (t) c1 += (S.x[t] - S.mu) * (S.x[t - 1] - S.mu); }
  out('hz-out', [T.hz(NAME[k], h, nf(V[0], 2), nf(Math.sqrt(h), 3), nf(Math.sqrt(h) * V[0], 2), nf(V[h - 1], 2), nf(V[h - 1] / V[0], 2), nf(vn[h - 1], 2), nf(c1 / c0, 3), V[h - 1] / V[0] > Math.sqrt(h), k !== 'sp500' || h !== 10)]);
}

// ================================================================= x3. historical simulation (worked example)
function hsBlock() {
  const k = kSel('hs-s'), nl = +kSel('hs-n'), S = ser(k), s = sortedLast(k, nl), n = s.length, nw = single('hs-chart', 340);
  const k1 = kOf(n, 0.01), k25 = kOf(n, 0.025), m = Math.max(k25 + 5, kOf(n, 0.04)), idx = range(1, m, 1), v = -s[k1 - 1], v25 = -s[k25 - 1], e = hsEs(s, 0.025);
  plot('hs-chart', [{ x: idx, y: idx.map(i => s[i - 1]), mode: 'markers', marker: { color: idx.map(i => (i <= k25 ? PURPLE : BLUE)), size: n > 2000 ? 4 : 6 }, name: 'ordered returns r(i); purple: the worst 2.5%', hovertemplate: 'i = %{x}: %{y:.2f}%<extra></extra>' },
    { x: [k1], y: [-v], mode: 'markers', marker: { color: RED, size: 13, symbol: 'diamond' }, name: `r(${k1}) = minus VaR 1%`, hoverinfo: 'skip' },
    { x: [1, k25], y: [-e, -e], mode: 'lines', line: { color: PURPLE, width: 2, dash: 'dash' }, name: `average of the ${k25} worst = minus ES 2.5%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('rank i (1 = worst day)'), yaxis: AX('daily return (%)'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const from = fdate(S.dates[S.n - n]);
  steps('hs-wx', [T.hs1(NAME[k], fint(n), from, nf(n * 0.01, 2), k1, nf(v, 2)), T.hs2(nf(n * 0.025, 2), k25, nf(e, 2)), T.hs3(k25, nf(v25, 2), nf(e / v, 3))]);
  out('hs-out', [T.hs(NAME[k], fint(n), k1, nf(v, 2), k25, nf(e, 2), k !== 'sp500' || nl !== 0)]);
}

// ================================================================= x4. precision of a historical VaR (bootstrap)
function prBlock() {
  const k = kSel('pr-s'), nl = kSel('pr-n'), B = D.boot[k], A = B.all, L = B.last[nl], Pn = panels('pr-chart', 2, { h: 340, hs: 250 }), tr = [];
  [[A, 0, `all ${A.n} days`], [L, 1, `last ${L.n} days`]].forEach(([o, i, lab]) => {
    const a = Pn.ax[i], xs = o.hv.map((_, j) => o.h0 + (j + 0.5) * o.hw);
    tr.push(on2(a, { type: 'bar', x: xs, y: o.hv, width: o.hw, marker: { color: hexA(RED, 0.6) }, name: 'bootstrap VaR 1%', showlegend: i === 0, hovertemplate: '%{x:.2f}%: %{y}<extra></extra>' }),
      on2(a, { type: 'bar', x: xs, y: o.he, width: o.hw, marker: { color: hexA(PURPLE, 0.5) }, name: 'bootstrap ES 2.5%', showlegend: i === 0, hovertemplate: '%{x:.2f}%: %{y}<extra></extra>' }),
      on2(a, { x: [o.var_lo, o.var_hi], y: [0, 0], mode: 'lines+markers', line: { color: RED, width: 4 }, marker: { size: 8 }, name: '95% interval of VaR 1%', showlegend: i === 0, hoverinfo: 'skip' }));
  });
  setAx(Pn, 0, AX('%'), AX('bootstrap samples')); setAx(Pn, 1, AX('%'), AX(''));
  plot('pr-chart', tr, layout(Object.assign(Pn.L, { barmode: 'overlay', bargap: 0, annotations: [ptitle(Pn.ax[0], `${NAME[k]}, all ${A.n} days`), ptitle(Pn.ax[1], `${NAME[k]}, last ${L.n} days`)],
    legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const it = o => [fint(o.n), nf(o.var, 2), nf(o.var_lo, 2), nf(o.var_hi, 2), nf(o.es, 2), nf(o.es_lo, 2), nf(o.es_hi, 2)];
  out('pr-out', [T.pr(NAME[k], it(A), it(L), nf(A.var_hi - A.var_lo, 2), nf(L.var_hi - L.var_lo, 2), kOf(L.n, 0.01), k !== 'sp500' || nl !== '500')]);
}

// ================================================================= x5. Cornish-Fisher (worked example)
function cfBlock() {
  const k = kSel('cf-s'), a = +kSel('cf-a'), S = ser(k), nw = single('cf-chart', 340), al = ALPHAS;
  const emp = al.map(p => (-hsVar(S.s, p) - S.mu) / S.sd);
  plot('cf-chart', [{ x: al.map(p => 100 * p), y: emp, mode: 'markers', marker: { color: DARK, size: 7 }, name: `${NAME[k]}: empirical quantile of the standardised returns`, hovertemplate: '%{x}%: %{y:.2f}<extra></extra>' },
    { x: al.map(p => 100 * p), y: al.map(p => normppf(p)), mode: 'lines', line: { color: GREEN, width: 1.6, dash: 'dash' }, name: 'Normal quantile z', hovertemplate: '%{x}%: %{y:.2f}<extra></extra>' },
    { x: al.map(p => 100 * p), y: al.map(p => cfz(p, S.S, S.K)), mode: 'lines', line: { color: PURPLE, width: 2 }, name: 'Cornish-Fisher quantile', hovertemplate: '%{x}%: %{y:.2f}<extra></extra>' },
    { x: [100 * a], y: [cfz(a, S.S, S.K)], mode: 'markers', marker: { color: RED, size: 11 }, name: `alpha = ${levE(a)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('tail probability alpha (%)', LOGX), yaxis: AX('quantile (standard deviations)'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const z = normppf(a), t1 = (z * z - 1) * S.S / 6, t2 = (z ** 3 - 3 * z) * S.K / 24, t3 = -(2 * z ** 3 - 5 * z) * S.S * S.S / 36, zc = z + t1 + t2 + t3, v = -(S.mu + S.sd * zc), vh = hsVar(S.s, a);
  steps('cf-wx', [T.cf1(NAME[k], nf(S.S, 3), nf(S.K, 2), lev(a), nf(z, 3)), T.cf2(nf(t1, 3), nf(t2, 3), nf(t3, 3), nf(zc, 3)), T.cf3(lev(a), nf(S.mu, 3), nf(S.sd, 3), nf(zc, 3), nf(v, 2), nf(vh, 2), nf(v / vh, 2))]);
  out('cf-out', [T.cf(NAME[k], lev(a), nf(zc, 2), nf(v, 2), nf(vh, 2), nf(v / vh, 2), k !== 'sp500' || a !== 0.01)]);
}

// ================================================================= x6. EVT: peaks over threshold (worked example)
function evBlock() {
  const k = kSel('ev-s'), q = +kSel('ev-u'), a = +kSel('ev-a'), S = ser(k), f = potOf(k, q), nw = single('ev-chart', 340);
  const L = Array.from(S.s).map(v => -v).filter(v => v > f.u).sort((x, y) => y - x), emp = L.map((_, i) => (i + 1) / S.n);
  const xg = linspace(f.u, Math.max(L[0], potVar(f, 0.001)) * 1.05, 200), gpd = xg.map(x => f.Nu / f.n * Math.pow(1 + f.xi * (x - f.u) / f.beta, -1 / f.xi));
  const v = potVar(f, a), e = potEs(f, 0.025), vh = hsVar(S.s, a);
  plot('ev-chart', [{ x: L, y: emp, mode: 'markers', marker: { color: BLUE, size: 4 }, name: `${NAME[k]}: empirical P(L > x), losses above u`, hovertemplate: '%{x:.2f}%: %{y:.4f}<extra></extra>' },
    { x: xg, y: gpd, mode: 'lines', line: { color: RED, width: 2 }, name: `GPD tail, u = ${f.u.toFixed(2)}%, xi = ${f.xi.toFixed(3)}`, hovertemplate: '%{x:.2f}%: %{y:.4f}<extra></extra>' },
    { x: [v], y: [a], mode: 'markers', marker: { color: DARK, size: 11, symbol: 'diamond' }, name: `EVT VaR ${levE(a)} = ${v.toFixed(2)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('loss L = -r (%)'), yaxis: AX('P(L > x)', { type: 'log' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const r = f.n * a / f.Nu, rx = r ** (-f.xi);
  steps('ev-we', [T.ev1(NAME[k], qpc(q), nf(f.u, 3), f.Nu, nf(100 * f.Nu / f.n, 1), nf(f.xi, 3), nf(f.beta, 3)),
    T.ev2(lev(a), nf(r, 4), nf(rx, 3), nf(f.beta / f.xi, 3), nf(f.u, 2), nf(v, 2)), T.ev3(nf(potVar(f, 0.025), 2), nf(f.beta, 3), nf(f.xi, 3), nf(f.u, 3), nf(e, 2)),
    T.ev4(lev(a), nf(vh, 2), nf(-S.s[0], 2), nf(potVar(f, 0.001), 2))]);
  out('ev-out', [T.ev(NAME[k], qpc(q), lev(a), nf(v, 2), nf(e, 2), nf(vh, 2), k !== 'sp500' || q !== 0.9 || a !== 0.01)]);
}

// ================================================================= 2. VaR across tail probabilities
const MST = { historical: [DARK, 'markers', null], Normal: [GREEN, 'lines', 'dash'], 'Student-t': [BLUE, 'lines', null], 'Cornish-Fisher': [PURPLE, 'lines', 'dashdot'], 'EVT (POT)': [RED, 'lines', null] };
function curveOf(k, q) {
  const S = ser(k), t = NUM.methods[k].t, f = potOf(k, q);
  return { historical: ALPHAS.map(a => hsVar(S.s, a)), Normal: ALPHAS.map(a => normVar(S.mu, S.sd, a)), 'Student-t': ALPHAS.map(a => -(t.m + t.s * tppf(a, t.nu))),
    'Cornish-Fisher': ALPHAS.map(a => -(S.mu + S.sd * cfz(a, S.S, S.K))), 'EVT (POT)': ALPHAS.map(a => potVar(f, a)) };
}
function cvBlock() {
  const k2 = kSel('cv-s'), q = +kSel('cv-u'), ks = ['sp500', k2], Pn = panels('cv-chart', 2, { h: 360, hs: 260 }), tr = [], items = [];
  ks.forEach((k, i) => {
    const cu = curveOf(k, q);
    Object.entries(cu).forEach(([m, v]) => { const [c, mode, dash] = MST[m]; tr.push(on2(Pn.ax[i], { x: ALPHAS.map(a => 100 * a), y: v.map(y => (isFinite(y) ? y : null)), mode, line: { color: c, width: 1.6, dash }, marker: { color: c, size: 6 }, name: m, showlegend: i === 0, hovertemplate: `${m}<br>%{x}%: %{y:.2f}%<extra></extra>` })); });
    items.push([SH[k], nf(cu.historical[0], 2), nf(cu.Normal[0], 2), nf(cu['Student-t'][0], 2), nf(cu['Cornish-Fisher'][0], 2), nf(cu['EVT (POT)'][0], 2), nf(cu.historical[11], 2), nf(Math.min(...Object.values(cu).map(v => v[11])), 2), nf(Math.max(...Object.values(cu).map(v => v[11])), 2)]);
  });
  setAx(Pn, 0, AX('tail probability alpha (%)', LOGX), AX('VaR (%)')); setAx(Pn, 1, AX('tail probability alpha (%)', LOGX), AX(''));
  plot('cv-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(Pn.ax[0], NAME.sp500), ptitle(Pn.ax[1], NAME[k2])], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  out('cv-out', [T.cv(qpc(q), items, k2 !== 'bet' || q !== 0.9)]);
}

// ================================================================= x7-x8. VaR and ES by method on six series (tables)
const M5 = ['HS', 'Normal', 'Student-t', 'Cornish-Fisher', 'EVT'], M5C = { HS: AMBER, Normal: GREEN, 'Student-t': BLUE, 'Cornish-Fisher': PURPLE, EVT: RED };
function vtBlock() {
  const a = +kSel('vt-a'), nw = single('vt-chart', 340), M = MK.map(k => methods(k, a, 0.025)), rows = [], tr = [];
  M5.forEach(m => tr.push({ type: 'bar', x: MK.map(k => SH[k]), y: M.map(o => o[m].var / o.HS.var), name: m, marker: { color: M5C[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f} x HS<extra></extra>` }));
  plot('vt-chart', tr, layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`VaR ${levE(a)} / historical VaR ${levE(a)}`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 1, y1: 1, line: { color: DARK, width: 1, dash: 'dash' } }] }));
  MK.forEach((k, i) => rows.push([SH[k], ...M5.map(m => nf(M[i][m].var, 2)), nf(ser(k).K, 1), nf(NUM.methods[k].xi, 2)]));
  table('vt-tab', ['', 'HS', T.h.normal, 'Student-t', 'Cornish–Fisher', 'EVT', 'K', 'ξ̂'], rows);
  const dHTE = Math.max(...M.map(o => Math.max(o.HS.var, o['Student-t'].var, o.EVT.var) - Math.min(o.HS.var, o['Student-t'].var, o.EVT.var))), nr = M.map(o => o.Normal.var / o.HS.var), cr = M.map(o => o['Cornish-Fisher'].var / o.HS.var);
  out('vt-out', [T.vt(lev(a), nf(dHTE, 2), Math.min(...nr), Math.max(...nr), nf(Math.min(...cr), 1), nf(Math.max(...cr), 1), a !== 0.01)]);
}
function etBlock() {
  const ae = +kSel('et-a'), nw = single('et-chart', 340), M = MK.map(k => methods(k, 0.01, ae)), rows = [], tr = [];
  M5.forEach(m => tr.push({ type: 'bar', x: MK.map(k => SH[k]), y: M.map(o => o[m].es / o.HS.var), name: m, marker: { color: M5C[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f} x HS VaR 1%<extra></extra>` }));
  plot('et-chart', tr, layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`ES ${levE(ae)} / historical VaR 1%`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 1, y1: 1, line: { color: DARK, width: 1, dash: 'dash' } }] }));
  MK.forEach((k, i) => rows.push([SH[k], ...M5.map(m => nf(M[i][m].es, 2)), nf(ser(k).K, 1), nf(NUM.methods[k].xi, 2)]));
  table('et-tab', ['', 'HS', T.h.normal, 'Student-t', 'Cornish–Fisher', 'EVT', 'K', 'ξ̂'], rows);
  const rr = M.map(o => 100 * (o.HS.es / o.HS.var - 1)), nr = npdf(normppf(ae)) / ae / -normppf(0.01), bt = M[MK.indexOf('btc')];
  const tmax = MK.filter((k, i) => ['HS', 'Normal', 'Student-t', 'EVT'].every(m => M[i]['Student-t'].es >= M[i][m].es)).map(k => SH[k]);
  out('et-out', [T.et(lev(ae), nf(nr, 3), nf(Math.min(...rr), 1), nf(Math.max(...rr), 1), tmax, nf(bt['Student-t'].es, 2), nf(bt.HS.es, 2), ae !== 0.025)]);
}

// ================================================================= x9. the S&P 500 on 18 September 2026 (worked example)
function cwBlock() {
  const k = kSel('cw-s'), a = +kSel('cw-a'), R = cond(k, a, 0.025), c = R.c, Z = D.cond[k], nw = single('cw-chart', 340), al = ALPHAS;
  plot('cw-chart', [{ x: al.map(p => 100 * p), y: al.map(p => Z.z[kOf(Z.n, p) - 1]), mode: 'markers+lines', line: { color: BLUE, width: 1.2 }, marker: { size: 6 }, name: `${NAME[k]}: quantile of the standardised residuals (FHS)`, hovertemplate: '%{x}%: %{y:.3f}<extra></extra>' },
    { x: al.map(p => 100 * p), y: al.map(p => stdtQ(c.nu, p)), mode: 'lines', line: { color: RED, width: 2 }, name: `standardised Student-t, nu = ${c.nu.toFixed(2)}`, hovertemplate: '%{x}%: %{y:.3f}<extra></extra>' },
    { x: al.map(p => 100 * p), y: al.map(p => normppf(p)), mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: 'Normal quantile', hovertemplate: '%{x}%: %{y:.3f}<extra></extra>' },
    { x: [100 * a, 100 * a], y: [R.qt, R.qz], mode: 'markers', marker: { color: DARK, size: 10 }, name: `alpha = ${levE(a)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('tail probability alpha (%)', LOGX), yaxis: AX('quantile of z'), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const vh = hsVar(ser(k).s, a);
  steps('cw-we', [T.cw1(NAME[k], fdate(c.last), nf(c.mu, 3), nf(c.alpha, 3), nf(c.beta, 3), nf(c.nu, 2), nf(c.sigma, 3)),
    T.cw2(lev(a), nf(R.qt, 3), nf(R.et, 3), nf(c.mu, 3), nf(c.sigma, 3), nf(R.vg, 2), nf(R.eg, 2)),
    T.cw3(lev(a), fint(Z.n), R.kz, nf(R.qz, 3), nf(R.ez, 3), nf(R.vf, 2), nf(R.ef, 2)), T.cw4(lev(a), nf(vh, 2), nf(R.vg / vh, 2))]);
  out('cw-out', [T.cw(NAME[k], lev(a), nf(R.vg, 2), nf(R.vf, 2), nf(vh, 2), k !== 'sp500' || a !== 0.01)]);
}
function ctBlock() {
  const [a, ae] = kSel('ct-l').split('|').map(Number), Pn = panels('ct-chart', 2, { h: 340, hs: 250 }), tr = [], rows = [], C3 = { HS: AMBER, 'GARCH-t': RED, FHS: BLUE };
  const R = MK.map(k => ({ k, c: cond(k, a, ae), vh: hsVar(ser(k).s, a), eh: hsEs(ser(k).s, ae) }));
  [['HS', o => o.vh, o => o.eh], ['GARCH-t', o => o.c.vg, o => o.c.eg], ['FHS', o => o.c.vf, o => o.c.ef]].forEach(([m, fv, fe]) => {
    tr.push(on2(Pn.ax[0], { type: 'bar', x: MK.map(k => SH[k]), y: R.map(fv), name: m === 'HS' ? 'HS (all data)' : m, marker: { color: C3[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f}%<extra></extra>` }),
      on2(Pn.ax[1], { type: 'bar', x: MK.map(k => SH[k]), y: R.map(fe), name: m, showlegend: false, marker: { color: C3[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f}%<extra></extra>` }));
  });
  setAx(Pn, 0, AX(''), AX('%')); setAx(Pn, 1, AX(''), AX(''));
  plot('ct-chart', tr, layout(Object.assign(Pn.L, { barmode: 'group', annotations: [ptitle(Pn.ax[0], `VaR ${levE(a)}, next day`), ptitle(Pn.ax[1], `ES ${levE(ae)}, next day`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  R.forEach(o => rows.push([SH[o.k], nf(o.vh, 2), nf(o.c.vg, 2), nf(o.c.vf, 2), nf(o.eh, 2), nf(o.c.eg, 2), nf(o.c.ef, 2), nf(o.c.c.sigma, 3)]));
  table('ct-tab', [['', { v: `VaR ${lev(a)}`, span: 3 }, { v: `ES ${lev(ae)}`, span: 3 }, ''], ['', T.h.hsall, 'GARCH-t', 'FHS', T.h.hsall, 'GARCH-t', 'FHS', 'σ<sub>t+1</sub>']], rows);
  out('ct-out', [T.ct(lev(a), lev(ae), R.filter(o => o.c.vg < o.vh).map(o => SH[o.k]), R.filter(o => o.c.vg >= o.vh).map(o => SH[o.k]), a !== 0.01 || ae !== 0.025)]);
}

// ================================================================= 3. VaR 1% through 2008 and 2020
const RVP = [['2007-06-01', '2009-12-31', '2007-2009'], ['2019-10-01', '2021-03-31', '2019-2021']];
function rvBlock() {
  const k = kSel('rv-s'), a = +kSel('rv-a'), f = frame(k), P = narrow('rv-chart') ? vstack('rv-chart', 2, 720, 0.14) : panels('rv-chart', 2, { h: 400 }), tr = [], items = [];
  RVP.forEach(([d0, d1, lab], i) => {
    const ax = P.ax[i], j0 = firstFrom(f.dates, d0), j1 = firstFrom(f.dates, d1 + 'Z'), ix = range(j0, j1 - 1, 1), d = ix.map(j => f.dates[j]);
    tr.push(on2(ax, { x: d, y: ix.map(j => f.r[j]), mode: 'lines', line: { color: TEAL, width: 0.7 }, name: 'daily return (%)', showlegend: i === 0, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }));
    const it = [lab];
    METH.forEach(m => {
      const v = f[vcol(a, m)];
      tr.push(on2(ax, { x: d, y: ix.map(j => -v[j]), mode: 'lines', line: { color: MCOL[m], width: 1.5, dash: m === 'HS' || m === 'Normal' ? 'dash' : 'solid' }, name: `minus VaR ${levE(a)}, ${m}`, showlegend: i === 0, hovertemplate: `${m}<br>%{x}<br>%{y:.2f}%<extra></extra>` }));
      it.push(ix.filter(j => f.r[j] < -v[j]).length);
    });
    const vh = f[vcol(a, 'HS')], hj = ix.filter(j => f.r[j] < -vh[j]);
    tr.push(on2(ax, { x: hj.map(j => f.dates[j]), y: hj.map(j => f.r[j]), mode: 'markers', marker: { color: MCOL.HS, size: 6 }, name: 'exceedance of the HS VaR', showlegend: i === 0, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }));
    const last = ix[ix.length - 1]; it.push(nf(f[vcol(a, 'HS')][last], 2), nf(f[vcol(a, 'GARCH-t')][last], 2), fdate(f.dates[last]));
    items.push(it);
    setAx(P, i, AX('', { type: 'date' }), AX('%'));
  });
  plot('rv-chart', tr, layout(Object.assign(P.L, { annotations: [ptitle(P.ax[0], `${NAME[k]}, 2007-2009`), ptitle(P.ax[1], `${NAME[k]}, 2019-2021`)], legend: LEG(P.stack, P.stack ? -0.06 : -0.16), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  out('rv-out', [T.rv(NAME[k], lev(a), items, k !== 'sp500' || a !== 0.01)]);
}

// ================================================================= x10. stress periods (table)
function sxBlock() {
  const a = +kSel('sx-a'), K3 = ['sp500', 'dax', 'bet'], Pn = panels('sx-chart', 2, { h: 340, hs: 250 }), tr = [], rows = [], E = Object.entries(D.episodes);
  const cnt = {}; K3.forEach(k => { const f = frame(k); cnt[k] = E.map(([, [d0, d1]]) => { const j0 = firstFrom(f.dates, d0), j1 = firstFrom(f.dates, d1 + 'Z'); return { n: j1 - j0, x: Object.fromEntries(METH.map(m => { const v = f[vcol(a, m)]; let c = 0; for (let j = j0; j < j1; j++) if (f.r[j] < -v[j]) c++; return [m, c]; })) }; }); });
  E.forEach((_, i) => {
    METH.forEach(m => tr.push(on2(Pn.ax[i], { type: 'bar', x: K3.map(k => SH[k]), y: K3.map(k => cnt[k][i].x[m]), name: m, showlegend: i === 0, marker: { color: MCOL[m] }, hovertemplate: `${m}<br>%{x}: %{y}<extra></extra>` })));
    tr.push(on2(Pn.ax[i], { x: K3.map(k => SH[k]), y: K3.map(k => cnt[k][i].n * a), mode: 'markers', marker: { color: DARK, size: 12, symbol: 'line-ew-open', line: { width: 3 } }, name: `expected: ${levE(a)} of the days`, showlegend: i === 0, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' }));
  });
  setAx(Pn, 0, AX(''), AX('exceedances', { rangemode: 'tozero' })); setAx(Pn, 1, AX(''), AX('', { rangemode: 'tozero' }));
  plot('sx-chart', tr, layout(Object.assign(Pn.L, { barmode: 'group', annotations: [ptitle(Pn.ax[0], 'Sep 2008 - Mar 2009'), ptitle(Pn.ax[1], 'Feb - May 2020')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  K3.forEach(k => rows.push([SH[k], ...METH.map(m => cnt[k][0].x[m]), nf(cnt[k][0].n * a, 1), ...METH.map(m => cnt[k][1].x[m]), nf(cnt[k][1].n * a, 1)]));
  table('sx-tab', [['', { v: T.h.p2008, span: 5 }, { v: T.h.p2020, span: 5 }], ['', 'HS', T.h.normal, 'GARCH-t', 'FHS', T.h.exp, 'HS', T.h.normal, 'GARCH-t', 'FHS', T.h.exp]], rows);
  const s = cnt.sp500[0];
  out('sx-out', [T.sx(lev(a), s.n, s.x.Normal, s.x.HS, s.x['GARCH-t'], s.x.FHS, cnt.sp500[1].n, cnt.sp500[1].x.Normal, cnt.sp500[1].x['GARCH-t'], a !== 0.01)]);
}

// ================================================================= x11. variance-covariance (worked example)
function pstats() {
  const P = port(), ma = mean(P.a), mb = mean(P.b), sa = sdev(P.a), sb = sdev(P.b), r = corrOf(P.a, P.b);
  return { ma, mb, sa, sb, r };
}
const RHO = () => { const N = NUM.port; return { s: pstats().r, calm: N.corr_calm, 2008: N.corr_2008, 2020: N.corr_2020, one: 1 }; };
function vcCalc(w, rho) {
  const p = pstats(), cv = rho * p.sa * p.sb, mp = w * p.ma + (1 - w) * p.mb, s2 = (w * p.sa) ** 2 + ((1 - w) * p.sb) ** 2 + 2 * w * (1 - w) * cv, sp = Math.sqrt(s2), z = normppf(0.01);
  const v1 = normVar(p.ma, p.sa, 0.01), v2 = normVar(p.mb, p.sb, 0.01), vp = -(mp + z * sp), vs = w * v1 + (1 - w) * v2;
  const c1 = w * (w * p.sa * p.sa + (1 - w) * cv) / sp, c2 = (1 - w) * ((1 - w) * p.sb * p.sb + w * cv) / sp;
  return { p, cv, mp, s2, sp, v1, v2, vp, vs, c1, c2 };
}
function vcBlock() {
  const w = +$('vc-w').value, rk = kSel('vc-r'), rho = RHO()[rk], R = vcCalc(w, rho), nw = single('vc-chart', 340), ws = linspace(0, 1, 41); label('vc-w').textContent = nf(w, 2);
  plot('vc-chart', [{ x: ws, y: ws.map(x => vcCalc(x, rho).vp), mode: 'lines', line: { color: RED, width: 2 }, name: `portfolio Normal VaR 1%, rho = ${rho.toFixed(2)}`, hovertemplate: 'w(BET) = %{x:.2f}: %{y:.2f}%<extra></extra>' },
    { x: ws, y: ws.map(x => vcCalc(x, rho).vs), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'weighted sum of the two Normal VaR 1%', hovertemplate: 'w(BET) = %{x:.2f}: %{y:.2f}%<extra></extra>' },
    { x: [w], y: [R.vp], mode: 'markers', marker: { color: RED, size: 11 }, name: `w(BET) = ${w.toFixed(2)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('weight of the BET'), yaxis: AX('VaR 1% (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ben = 100 * (1 - R.vp / R.vs);
  steps('vc-wx', [T.vc1(nf(w, 2), nf(1 - w, 2), nf(R.p.sa, 3), nf(R.p.sb, 3), nf(rho, 3), nf(w * R.p.sa, 3), nf((1 - w) * R.p.sb, 3), nf(R.cv, 3), nf(R.s2, 3), nf(R.sp, 3)),
    T.vc2(nf(R.mp, 3), nf(R.sp, 3), nf(R.vp, 2), nf(w, 2), nf(R.v1, 2), nf(1 - w, 2), nf(R.v2, 2), nf(R.vs, 2), nf(ben, 0)),
    T.vc3(nf(R.c1, 3), nf(R.c2, 3), nf(100 * R.c1 / R.sp, 0), nf(100 * R.c2 / R.sp, 0))]);
  out('vc-out', [T.vc(nf(w, 2), nf(rho, 2), nf(R.vp, 2), nf(R.vs, 2), nf(ben, 0), w !== 0.5 || rk !== 's')]);
}

// ================================================================= 4. correlation rises in crises
function rcBlock() {
  const w = +kSel('rc-w'), P = port(), RC = rollCorr(P, w), nw = single('rc-chart', 330), A = { x: 'x', y: 'y' };
  let im = 0, iM = 0; RC.c.forEach((v, i) => { if (v < RC.c[im]) im = i; if (v > RC.c[iM]) iM = i; });
  const all = corrOf(P.a, P.b);
  plot('rc-chart', [{ x: RC.dates, y: RC.c, mode: 'lines', line: { color: RED, width: 1.4 }, name: `${w}-day correlation, BET and S&P 500`, hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>' },
    { x: [P.dates[0], P.dates[P.n - 1]], y: [all, all], mode: 'lines', line: { color: BLUE, width: 1.2, dash: 'dash' }, name: `whole sample: ${all.toFixed(2)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('correlation'), shapes: epShapes(A, '2000'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const N = NUM.port, per = [['2017', N.corr_calm], ['2008', N.corr_2008], ['2020', N.corr_2020]].map(([l, v]) => nf(v, 2));
  out('rc-out', [T.rc(w, nf(RC.c[iM], 2), fdate(RC.dates[iM]), nf(RC.c[im], 2), fdate(RC.dates[im]), nf(RC.c[RC.c.length - 1], 2), nf(all, 2), per, w !== 250)]);
}

// ================================================================= x12. tail dependence (extra chart)
function tdBlock() {
  const q = +$('td-q').value, per = kSel('td-p'), U = pobs(per), qs = range(0.005, 0.1, 0.005), ch = qs.map(x => chiQ(U, x)), nw = single('td-chart', 330); label('td-q').textContent = nf(q, 3);
  plot('td-chart', [{ x: qs.map(x => 100 * x), y: ch, mode: 'lines+markers', line: { color: RED, width: 2 }, marker: { size: 5 }, name: 'data: P(U_BET <= q | U_S&P <= q)', hovertemplate: 'q = %{x:.1f}%: %{y:.3f}<extra></extra>' },
    { x: qs.map(x => 100 * x), y: qs, mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: 'independence: q', hoverinfo: 'skip' },
    { x: [100 * q], y: [chiQ(U, q)], mode: 'markers', marker: { color: DARK, size: 11 }, name: `q = ${(100 * q).toFixed(1)}%`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('q (%)'), yaxis: AX('conditional probability', { rangemode: 'tozero' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  out('td-out', [T.td(T.h.per[per], fint(U.n), nf(100 * q, 1), nf(chiQ(U, q), 2), nf(chiQ(U, 0.05), 2), nf(chiQ(U, 0.01), 2), nf(100 * joint1(U.X), 2), per !== 'all' || q !== 0.05)]);
}

// ================================================================= x13. Gaussian and t copulas (worked example)
function cqBlock() {
  const nu = +$('cq-n').value, rk = kSel('cq-r'), rho = rk === 'tau' ? NUM.port.copula.rho : RHO()[rk], cf = NUM.port.copula, G = D.port.ll_grid, Pn = panels('cq-chart', 2, { h: 330, hs: 250 }); label('cq-n').textContent = nf(nu, 2);
  const lam = tailDep(rho, nu), j = G.nu.findIndex(v => Math.abs(v - nu) < 1e-9), nus = linspace(2, 40, 153);
  plot('cq-chart', [on2(Pn.ax[0], { x: G.nu, y: G.ll, mode: 'lines', line: { color: RED, width: 2 }, name: 't copula log-likelihood (rho = 0.157)', hovertemplate: 'nu = %{x}: %{y:.1f}<extra></extra>' }),
    on2(Pn.ax[0], { x: [2, 40], y: [cf.ll_g, cf.ll_g], mode: 'lines', line: { color: GREEN, width: 1.4, dash: 'dash' }, name: 'Gaussian copula log-likelihood', hoverinfo: 'skip' }),
    on2(Pn.ax[0], { x: [nu], y: [G.ll[j]], mode: 'markers', marker: { color: DARK, size: 10 }, name: `nu = ${nu}`, hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: nus, y: nus.map(v => tailDep(rho, v)), mode: 'lines', line: { color: PURPLE, width: 2 }, name: `lambda_L of the t copula, rho = ${rho.toFixed(3)}`, hovertemplate: 'nu = %{x:.1f}: %{y:.3f}<extra></extra>' }),
    on2(Pn.ax[1], { x: [nu], y: [lam], mode: 'markers', marker: { color: DARK, size: 10 }, name: 'chosen nu', showlegend: false, hoverinfo: 'skip' })],
  layout(Object.assign(Pn.L, (setAx(Pn, 0, AX('nu'), AX('log-likelihood')), setAx(Pn, 1, AX('nu'), AX('lambda_L', { rangemode: 'tozero' })), {}),
    { annotations: [ptitle(Pn.ax[0], 'BET and S&P 500, pseudo-observations'), ptitle(Pn.ax[1], 'lower tail dependence')], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const s = Math.sqrt((nu + 1) * (1 - rho) / (1 + rho));
  steps('cq-we', [T.cq1(nf(cf.tau, 3), nf(cf.rho, 3)), T.cq2(nf(cf.ll_g, 1), nf(cf.ll_t, 1), nf(cf.nu, 2), nf(2 * (cf.ll_t - cf.ll_g), 1)),
    T.cq3(nf(nu, 2), nf(rho, 3), nf(nu + 1, 2), nf(s, 3), nf(tcdf(-s, nu + 1), 4), nf(lam, 3), nf(G.ll[j], 1))]);
  out('cq-out', [T.cq(nf(nu, 2), nf(rho, 3), nf(lam, 3), nf(G.ll[j], 1), nu !== 4.5 || rk !== 'tau')]);
}

// ================================================================= 5. joint bad days: data and two copulas
function cpBlock() {
  const q = +$('cp-q').value, s = kSel('cp-s'), Cn = D.port.corner, P = narrow('cp-chart') ? vstack('cp-chart', 3, 900, 0.1) : panels('cp-chart', 3, { h: 360, gap: 0.05 }); label('cp-q').textContent = nf(q, 2);
  const cf = NUM.port.copula, sets = [['data (pseudo-observations)', Cn.data, BLUE], [`Gaussian copula, rho = ${cf.rho.toFixed(2)}`, Cn[s].gauss, GREEN], [`t copula, rho = ${cf.rho.toFixed(2)}, nu = ${cf.nu.toFixed(1)}`, Cn[s].t, RED]];
  const tr = [], cnt = [];
  sets.forEach(([lab, X, c], i) => {
    const a = P.ax[i]; let k = 0; for (let j = 0; j < X[0].length; j++) if (X[0][j] <= q && X[1][j] <= q) k++; cnt.push(k);
    tr.push(on2(a, { x: X[1], y: X[0], mode: 'markers', marker: { color: hexA(c, 0.6), size: 4 }, name: lab, hovertemplate: '%{x:.3f}, %{y:.3f}<extra></extra>' }),
      on2(a, { x: [0, q, q], y: [q, q, 0], mode: 'lines', line: { color: ORANGE, width: 2 }, name: `${k} pairs below ${q.toFixed(2)}`, showlegend: false, hoverinfo: 'skip' }));
    setAx(P, i, AX('S&P 500 (uniform scale)', { range: [0, 0.25] }), AX(i ? '' : 'BET (uniform scale)', { range: [0, 0.25] }));
  });
  plot('cp-chart', tr, layout(Object.assign(P.L, { annotations: sets.map((_, i) => ptitle(P.ax[i], `${cnt[i]} pairs below ${q.toFixed(2)} on both scales`)), legend: LEG(P.stack, P.stack ? -0.05 : -0.18), margin: { l: 56, r: 12, t: 26, b: 40 } })));
  const n = NUM.copula_fig.n;
  out('cp-out', [T.cp(nf(100 * q, 0), cnt[0], cnt[1], cnt[2], nf(n * q * q, 1), s, q !== 0.05 || s !== String(D.cop_seeds[0]))]);
}

// ================================================================= x14. portfolio VaR and ES by method (table)
function ptBlock() {
  const w = kSel('pt-w'), o = D.port.w[w], nw = single('pt-chart', 330), M4 = [['n', T.h.vc, 'variance-covariance (Normal)', GREEN], ['gauss', T.h.gauss, 'Gaussian copula', BLUE], ['t', T.h.tcop, 't copula', RED], ['hs', T.h.hsdata, 'historical simulation', AMBER]];
  plot('pt-chart', [{ type: 'bar', x: M4.map(m => m[2]), y: M4.map(m => o['var_' + m[0]]), name: 'VaR 1%', marker: { color: RED }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { type: 'bar', x: M4.map(m => m[2]), y: M4.map(m => o['es_' + m[0]]), name: 'ES 2.5%', marker: { color: PURPLE }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`portfolio, w(BET) = ${w} (%)`, { rangemode: 'tozero' }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const jt = { n: '–', gauss: nf(100 * o.joint1_gauss, 2) + '%', t: nf(100 * o.joint1_t, 2) + '%', hs: nf(100 * o.joint1_emp, 2) + '%' };
  table('pt-tab', [T.h.method, 'VaR 1%', `ES ${lev(0.025)}`, T.h.joint], M4.map(m => [{ v: m[1], c: 'l' }, nf(o['var_' + m[0]], 2), nf(o['es_' + m[0]], 2), jt[m[0]]]));
  out('pt-out', [T.pt(nf(+w, 2), nf(o.var_n, 2), nf(o.var_gauss, 2), nf(o.var_t, 2), nf(o.var_hs, 2), nf(o.var_hs_sum, 2), w !== '0.5')]);
}

// ================================================================= 6. exceptions in 250 days under a correct VaR 1%
const zoneCum = c => (c < 0.95 ? 'green' : c < 0.9999 ? 'yellow' : 'red');
function bnBlock() {
  const p = +$('bn-p').value, n = +kSel('bn-n'), nw = single('bn-chart', 330); label('bn-p').textContent = nf(100 * p, 2) + '%';
  const xm = Math.max(14, Math.ceil(n * Math.max(p, 0.01) + 5 * Math.sqrt(n * Math.max(p, 0.01)) + 3)), xs = range(0, xm, 1);
  const cum0 = xs.map(x => bincdf(x, n, 0.01)), zn = xs.map((x, i) => (n === 250 ? zone(x) : zoneCum(cum0[i]))), pm = xs.map(x => binpmf(x, n, p));
  const tr = ['green', 'yellow', 'red'].map(z => ({ type: 'bar', x: xs.filter((_, i) => zn[i] === z), y: xs.filter((_, i) => zn[i] === z).map(x => 100 * pm[x]), marker: { color: ZC[z] }, name: `${z} zone`,
    text: xs.filter((_, i) => zn[i] === z).map(x => (n === 250 && x <= 9 ? (100 * cum0[x]).toFixed(2) : '')), textposition: 'outside', textfont: { size: 9, color: DARK }, hovertemplate: '%{x} exceptions: %{y:.2f}%<extra></extra>' }));
  plot('bn-chart', tr, layout({ xaxis: AX(`number of exceptions in ${n} days (labels: cumulative probability under a correct VaR 1%, %)`, { dtick: n === 250 ? 1 : undefined }), yaxis: AX(`probability (%), true rate ${(100 * p).toFixed(2)}%`), legend: LEG(nw, -0.2), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ylo = xs.find((_, i) => zn[i] !== 'green'), rlo = xs.find((_, i) => zn[i] === 'red');
  let pg = 0, pr = 0; xs.forEach((x, i) => { if (zn[i] === 'green') pg += pm[x]; if (zn[i] === 'red') pr += pm[x]; });
  pr = 1 - bincdf(rlo - 1, n, p); const py = 1 - bincdf(ylo - 1, n, p) - pr;
  out('bn-out', [T.bn(n, nf(100 * p, 2), nf(n * p, 2), nf(100 * binpmf(0, n, p), 2), nf(100 * bincdf(ylo - 1, n, 0.01), 2), nf(100 * (1 - bincdf(ylo - 1, n, p)), 1), ylo, rlo - 1, nf(100 * py, 1), nf(100 * pr, 2), p !== 0.01 || n !== 250)]);
}

// ================================================================= x15. Kupiec's POF test (worked example)
function kpBlock() {
  const x = +$('kp-x').value, n = +kSel('kp-n'), a = 0.01, K = kupiec(x, n, a), nw = single('kp-chart', 330); label('kp-x').textContent = String(x);
  const xs = range(0, Math.max(15, Math.ceil(n * 0.04)), 1), lr = xs.map(j => kupiec(j, n, a).lr), acc = xs.filter((j, i) => lr[i] <= 3.841459);
  plot('kp-chart', [{ type: 'bar', x: xs, y: lr, marker: { color: xs.map((j, i) => (lr[i] <= 3.841459 ? GREEN : RED)) }, name: 'LR_uc (green: not rejected at 5%)', hovertemplate: 'x = %{x}: %{y:.2f}<extra></extra>' },
    { x: [-0.5, xs[xs.length - 1] + 0.5], y: [3.841459, 3.841459], mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'critical value 3.84', hoverinfo: 'skip' },
    { x: [x], y: [K.lr], mode: 'markers', marker: { color: DARK, size: 12, symbol: 'diamond' }, name: `x = ${x}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX(`exceptions x in ${n} days`), yaxis: AX('LR_uc', { range: [0, Math.min(30, Math.max(...lr) * 1.05)] }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const lo = acc[0], hi = acc[acc.length - 1], pw = bincdf(hi, n, 0.02) - (lo > 0 ? bincdf(lo - 1, n, 0.02) : 0);
  steps('kp-we', [T.kp1(n, x, nf(K.rate, 4)), T.kp2(n - x, x, nf(K.l0, 2)), T.kp3(n - x, nf(1 - K.rate, 4), x, nf(K.rate, 4), nf(K.l1, 2)), T.kp4(nf(K.l0, 2), nf(K.l1, 2), nf(K.lr, 2), fp(K.p), K.lr > 3.841459),
    T.kp5(n, lo, hi, nf(100 * pw, 0))]);
  out('kp-out', [T.kp(n, x, nf(K.lr, 2), fp(K.p), K.lr > 3.841459, x !== 7 || n !== 250)]);
}

// ================================================================= x16. Kupiec on 21 years (worked example)
function ksBlock() {
  const k = kSel('ks-s'), m = kSel('ks-m'), f = frame(k), h = hits(k, m, 0.01), n = h.length, nw = single('ks-chart', 340), K = kupiec(h.reduce((s, v) => s + v, 0), n, 0.01);
  const cum = []; let c = 0; for (let t = 0; t < n; t++) cum.push(c += h[t]);
  const step = Math.max(1, Math.floor(n / 600)), ix = range(0, n - 1, step);
  const lo = binppf(0.025, n, 0.01), hi = binppf(0.975, n, 0.01);
  plot('ks-chart', [{ x: ix.map(t => f.dates[t]), y: ix.map(t => cum[t]), mode: 'lines', line: { color: MCOL[m], width: 2 }, name: `${m}: cumulative exceptions`, hovertemplate: '%{x}: %{y}<extra></extra>' },
    { x: ix.map(t => f.dates[t]), y: ix.map(t => 0.01 * (t + 1)), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'expected: 1% of the days', hoverinfo: 'skip' },
    { x: [f.dates[n - 1], f.dates[n - 1]], y: [lo, hi], mode: 'lines+markers', line: { color: GREEN, width: 4 }, marker: { size: 8 }, name: `95% binomial interval at the end: ${lo}-${hi}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('', { type: 'date' }), yaxis: AX(`${NAME[k]}, exceptions of VaR 1%`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  steps('ks-we', [T.ks1(NAME[k], m, fdate(f.dates[0]), fdate(f.dates[n - 1]), fint(n), K.x, nf(K.rate, 4), nf(n * 0.01, 1), nf(100 * lo / n, 2), nf(100 * hi / n, 2)),
    T.ks2(fint(n - K.x), K.x, nf(K.l0, 1)), T.ks3(fint(n - K.x), nf(K.rate, 4), K.x, nf(K.l1, 1)), T.ks4(nf(K.lr, 2), fp(K.p), K.lr > 3.841459)]);
  out('ks-out', [T.ks(NAME[k], m, K.x, nf(n * 0.01, 1), nf(100 * K.rate, 2), nf(K.lr, 2), fp(K.p), K.lr > 3.841459, k !== 'sp500' || m !== 'GARCH-t')]);
}

// ================================================================= x17. Christoffersen's independence test (worked example)
function crBlock() {
  const k = kSel('cr-s'), m = kSel('cr-m'), nw = single('cr-chart', 330), R = Object.fromEntries(METH.map(mm => [mm, christ(hits(k, mm, 0.01), 0.01)])), c = R[m];
  plot('cr-chart', [{ type: 'bar', x: METH, y: METH.map(mm => 100 * R[mm].p01), name: 'pi01: after a day without exception', marker: { color: METH.map(mm => hexA(BLUE, mm === m ? 1 : 0.45)) }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { type: 'bar', x: METH, y: METH.map(mm => 100 * R[mm].p11), name: 'pi11: after a day with an exception', marker: { color: METH.map(mm => hexA(RED, mm === m ? 1 : 0.45)) }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: [METH[0], METH[3]], y: [1, 1], mode: 'lines', line: { color: DARK, width: 1.2, dash: 'dash' }, name: '1%', hoverinfo: 'skip' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX(`${NAME[k]}: probability of an exception (%)`, { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  steps('cr-we', [T.cr1(NAME[k], m, fint(c.uc.n), fint(c.n00), c.n01, c.n10, c.n11),
    T.cr2(c.n01, fint(c.n00 + c.n01), nf(100 * c.p01, 2), c.n11, c.n10 + c.n11, nf(100 * c.p11, 1), c.p01 > 0 ? nf(c.p11 / c.p01, 1) : '–'),
    T.cr3(nf(c.lri, 2), fp(c.pi), c.pi < 0.05), T.cr4(nf(c.uc.lr, 2), nf(c.lri, 2), nf(c.lrc, 2), fp(c.pc), c.lrc > 5.991465)]);
  out('cr-out', [T.cr(NAME[k], m, nf(100 * c.p01, 2), nf(100 * c.p11, 1), nf(c.lri, 2), fp(c.pi), nf(c.lrc, 2), fp(c.pc), k !== 'sp500' || m !== 'HS')]);
}

// ================================================================= x18. the Basel traffic light (table and two worked examples)
function capital(k, m) {             // max{VaR_{t-1}, (3 + plus) x 60-day average}, 10-day VaR = sqrt(10) x one-day VaR 1%
  const f = frame(k), h = hits(k, m, 0.01), v = f['var1_' + m], r = roll250(h), s10 = Math.sqrt(10), cap = [], fac = [], avg = [], d = [], vv = [], xx = [];
  let sum60 = 0;
  for (let t = 0; t < f.n; t++) {
    sum60 += v[t]; if (t >= 60) sum60 -= v[t - 60];
    if (t < 250) continue;
    const x = r[t - 1], F = 3 + plusOf(x), A = s10 * sum60 / 60;
    xx.push(x); fac.push(F); avg.push(A); vv.push(s10 * v[t]); cap.push(Math.max(s10 * v[t], F * A)); d.push(f.dates[t]);
  }
  return { cap, fac, avg, d, v: vv, x: xx };
}
function bsBlock() {
  const k = kSel('bs-s'), m = kSel('bs-m'), x = +$('bs-x').value, di = kSel('bs-d'), Cp = capital(k, m), P = vstack('bs-chart', 2, narrow('bs-chart') ? 600 : 520, 0.12); label('bs-x').textContent = String(x);
  plot('bs-chart', [on2(P.ax[0], { x: Cp.d, y: Cp.cap, mode: 'lines', line: { color: RED, width: 1.6 }, name: `capital: max(VaR, (3 + plus) x 60-day average), ${m}`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(P.ax[0], { x: Cp.d, y: Cp.v, mode: 'lines', line: { color: MCOL[m], width: 0.8 }, name: 'latest 10-day VaR 1% = sqrt(10) x one-day VaR', hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' }),
    on2(P.ax[1], { x: Cp.d, y: Cp.fac, mode: 'lines', line: { color: AMBER, width: 1.6, shape: 'hv' }, name: 'multiplier 3 + plus', hovertemplate: '%{x}<br>%{y:.2f}<extra></extra>' })],
  layout(Object.assign(P.L, (setAx(P, 0, AX('', { type: 'date' }), AX('% of the position')), setAx(P, 1, AX('', { type: 'date' }), AX('multiplier', { range: [2.9, 4.1] })), {}),
    { annotations: [ptitle(P.ax[0], `${NAME[k]}: capital for market risk, ${m}`), ptitle(P.ax[1], 'multiplier from the exceptions in the last 250 days')], legend: LEG(true, -0.06), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  const rows = D.N.traffic.filter(o => o.x <= 10).map(o => ({ x: o.x, cum: o.cum, z: o.zone, pl: o.plus })), zl = { green: T.h.zg, yellow: T.h.zy, red: T.h.zr };
  table('bs-tab', ['x', T.h.zone, T.h.cum, T.h.plus, T.h.factor], rows.map(o => { const c = (o.x === Math.min(x, 10) ? 'hi' : ''); return [{ v: o.x === 10 ? '≥ 10' : o.x, c }, { v: zl[o.z], c }, { v: nf(100 * o.cum, 2), c }, { v: nf(o.pl, 2), c }, { v: nf(3 + o.pl, 2), c }]; }));
  const cx = bincdf(x, 250, 0.01), cx1 = x ? bincdf(x - 1, 250, 0.01) : 0;
  steps('bs-we1', [T.bs1(x, nf(100 * cx, 2), nf(100 * cx1, 2)), T.bs2(x, zl[zone(x)], nf(plusOf(x), 2), nf(3 + plusOf(x), 2))]);
  let i = Cp.d.findIndex(d => d >= di); if (i < 0) i = Cp.d.length - 1;
  steps('bs-we2', [T.bs3(NAME[k], m, fdate(Cp.d[i]), Cp.x[i], zl[zone(Cp.x[i])], nf(Cp.fac[i], 2)), T.bs4(nf(Cp.v[i], 2), nf(Cp.avg[i], 2), nf(Cp.fac[i], 2), nf(Cp.fac[i] * Cp.avg[i], 2), nf(Cp.cap[i], 2), Cp.v[i] > Cp.fac[i] * Cp.avg[i])]);
  const sh = 100 * Cp.fac.filter(F => F > 3).length / Cp.fac.length; let iM = 0; Cp.cap.forEach((c, j) => { if (c > Cp.cap[iM]) iM = j; });
  const nb = Cp.cap.filter((c, j) => Cp.v[j] > Cp.fac[j] * Cp.avg[j]).length;
  out('bs-out', [T.bs(NAME[k], m, nf(sh, 0), nf(Cp.cap[iM], 2), fdate(Cp.d[iM]), nb, fint(Cp.d.length))]);
}

// ================================================================= 7. when do the exceptions happen?
function htBlock() {
  const k = kSel('ht-s'), a = +kSel('ht-a'), f = frame(k), nw = single('ht-chart', 330), A = { x: 'x', y: 'y' }, tr = [], it = [];
  METH.forEach((m, i) => {
    const h = hits(k, m, a), ix = []; h.forEach((v, t) => { if (v) ix.push(t); });
    tr.push({ x: ix.map(t => f.dates[t]), y: ix.map(() => i), mode: 'markers', marker: { color: MCOL[m], symbol: 'line-ns-open', size: 16, line: { width: 1.6 } }, name: `${m}: ${ix.length} exceedances`, hovertemplate: `${m}<br>%{x}<extra></extra>` });
    it.push([m, ix.length, christ(h, a).n11]);
  });
  plot('ht-chart', tr, layout({ xaxis: AX('date (shaded: September 2008 - March 2009 and February - May 2020)', { type: 'date' }), yaxis: AX('', { tickvals: [0, 1, 2, 3], ticktext: METH, range: [-0.6, 3.6] }),
    shapes: epShapes(A, f.dates[0]), legend: LEG(nw, -0.26), margin: { l: 70, r: 12, t: 14, b: 40 } }));
  out('ht-out', [T.ht(NAME[k], lev(a), fdate(f.dates[0]), fint(f.n), nf(f.n * a, 1), it, k !== 'sp500' || a !== 0.01)]);
}

// ================================================================= x19-x20. backtest results (tables)
function btBlock(p, ks) {
  const si = +kSel(p + '-y'), nw = single(p + '-chart', 330), rows = [], tr = [], R = {};
  ks.forEach(k => { const f = frame(k), y = D.bt_starts[k][si], j0 = firstFrom(f.dates, y); R[k] = { y, j0, n: f.n - j0, m: Object.fromEntries(METH.map(m => [m, btRow(k, m, j0)])) }; });
  METH.forEach(m => tr.push({ type: 'bar', x: ks.map(k => SH[k]), y: ks.map(k => 100 * R[k].m[m].rate), name: m, marker: { color: MCOL[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f}%<extra></extra>` }));
  tr.push({ x: ks.map(k => SH[k]), y: ks.map(() => 1), mode: 'lines', line: { color: DARK, width: 1.4, dash: 'dash' }, name: 'expected 1%', hoverinfo: 'skip' });
  plot(p + '-chart', tr, layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('exceedance rate of VaR 1% (%)', { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const pass = [];
  ks.forEach(k => METH.forEach((m, i) => {
    const o = R[k].m[m], c = o.c, pz = D.bt[k].pz2[R[k].y.slice(0, 4)][m], ok = c.uc.p >= 0.05 && c.pi >= 0.05 && c.pc >= 0.05;
    if (ok) pass.push(`${SH[k]} ${m}`);
    rows.push([i ? '' : SH[k], { v: m, c: 'l' }, o.x, nf(100 * o.rate, 2), fp(c.uc.p), fp(c.pi), fp(c.pc), o.mx, `${nf(o.z2, 2)} (${fp(pz)})`]);
  }));
  table(p + '-tab', ['', T.h.method, 'x', T.h.rate, 'p POF', T.h.pind, T.h.pcc, T.h.max250, 'Z<sub>2</sub> (p)'], rows);
  out(p + '-out', [T.bt(ks.map(k => [SH[k], fdate(frame(k).dates[R[k].j0]), fint(R[k].n)]), pass, si !== 0)]);
}
const b1Block = () => btBlock('b1', ['sp500', 'dax']), b2Block = () => btBlock('b2', ['bet', 'btc']);

// ================================================================= x21. ranking the forecasts and today's zone (table)
function scBlock() {
  const de = kSel('sc-d'), Pn = panels('sc-chart', 2, { h: 340, hs: 250 }), tr = [], rows = [], R = {};
  BTK.forEach(k => {
    const f = frame(k), j1 = firstFrom(f.dates, de + 'Z');
    R[k] = j1 < 250 ? null : { j1, d: f.dates[j1 - 1], q: Object.fromEntries(METH.map(m => [m, 100 * qloss(k, m, 0, j1)])), l: Object.fromEntries(METH.map(m => { const h = hits(k, m, 0.01); let s = 0; for (let t = j1 - 250; t < j1; t++) s += h[t]; return [m, s]; })) };
  });
  const ok = BTK.filter(k => R[k]);
  METH.forEach(m => { tr.push(on2(Pn.ax[0], { type: 'bar', x: ok.map(k => SH[k]), y: ok.map(k => R[k].q[m]), name: m, marker: { color: MCOL[m] }, hovertemplate: `${m}<br>%{x}: %{y:.2f}<extra></extra>` }),
    on2(Pn.ax[1], { type: 'bar', x: ok.map(k => SH[k]), y: ok.map(k => R[k].l[m]), name: m, showlegend: false, marker: { color: MCOL[m] }, hovertemplate: `${m}<br>%{x}: %{y}<extra></extra>` })); });
  tr.push(on2(Pn.ax[1], { x: [SH[ok[0]], SH[ok[ok.length - 1]]], y: [4.5, 4.5], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dash' }, name: 'upper limit of the green zone (4)', hoverinfo: 'skip' }),
    on2(Pn.ax[1], { x: [SH[ok[0]], SH[ok[ok.length - 1]]], y: [9.5, 9.5], mode: 'lines', line: { color: RED, width: 1.2, dash: 'dash' }, name: 'upper limit of the yellow zone (9)', hoverinfo: 'skip' }));
  setAx(Pn, 0, AX(''), AX('mean quantile loss x 100')); setAx(Pn, 1, AX(''), AX('exceptions, last 250 days', { rangemode: 'tozero' }));
  plot('sc-chart', tr, layout(Object.assign(Pn.L, { barmode: 'group', annotations: [ptitle(Pn.ax[0], `quantile loss, from the start to ${de}`), ptitle(Pn.ax[1], `exceptions in the 250 days to ${de}`)], legend: LEG(Pn.stack, Pn.stack ? -0.08 : -0.16), margin: { l: 56, r: 12, t: 26, b: 30 } })));
  BTK.forEach(k => { const o = R[k]; rows.push([SH[k], ...METH.map(m => (o ? { v: nf(o.q[m], 2), c: METH.every(z => o.q[m] <= o.q[z]) ? 'hi' : '' } : '–')), ...METH.map(m => (o ? { v: o.l[m], c: o.l[m] > 9 ? 'bad' : '' } : '–'))]); });
  table('sc-tab', [['', { v: T.h.qloss, span: 4 }, { v: T.h.last250, span: 4 }], ['', 'HS', T.h.normal, 'GARCH-t', 'FHS', 'HS', T.h.normal, 'GARCH-t', 'FHS']], rows);
  const best = ok.map(k => [SH[k], METH.reduce((b, m) => (R[k].q[m] < R[k].q[b] ? m : b), 'HS')]), nz = [];
  ok.forEach(k => METH.forEach(m => { const z = zone(R[k].l[m]); if (z !== 'green') nz.push([SH[k], m, R[k].l[m], T.h['z' + z[0]]]); }));
  out('sc-out', [T.sc(fdate(R.sp500.d), best, nz, BTK.filter(k => !R[k]).map(k => SH[k]), de !== D.end)]);
}

// ================================================================= 8. the traffic light through time
function ttBlock() {
  const k = kSel('tt-s'), f = frame(k), nw = single('tt-chart', 330), tr = [], it = [];
  ['HS', 'Normal', 'GARCH-t'].forEach(m => {
    const r = roll250(hits(k, m, 0.01)), ix = range(249, f.n - 1, 1), y = ix.map(t => r[t]);
    tr.push({ x: ix.map(t => f.dates[t]), y, mode: 'lines', line: { color: MCOL[m], width: 1.5 }, name: m, hovertemplate: `${m}<br>%{x}: %{y}<extra></extra>` });
    it.push([m, nf(100 * y.filter(v => v >= 10).length / y.length, 1), Math.max(...y)]);
  });
  const d = [f.dates[249], f.dates[f.n - 1]];
  tr.push({ x: d, y: [4.5, 4.5], mode: 'lines', line: { color: GREEN, width: 1.2, dash: 'dash' }, name: 'upper limit of the green zone (4)', hoverinfo: 'skip' },
    { x: d, y: [9.5, 9.5], mode: 'lines', line: { color: RED, width: 1.2, dash: 'dash' }, name: 'upper limit of the yellow zone (9)', hoverinfo: 'skip' });
  plot('tt-chart', tr, layout({ xaxis: AX('', { type: 'date' }), yaxis: AX('exceptions in 250 days', { rangemode: 'tozero' }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('tt-out', [T.tt(NAME[k], fdate(f.dates[249]), it, k !== 'sp500')]);
}

// ================================================================= x22. backtesting ES: Acerbi and Szekely (worked example)
function ebBlock() {
  const k = kSel('eb-s'), m = kSel('eb-m'), f = frame(k), nw = single('eb-chart', 330), tr = [], st = Math.max(1, Math.floor(f.n / 600)), R = {};
  METH.forEach(mm => {
    const v = f['var25_' + mm], e = f['es25_' + mm], z = []; let s = 0, nh = 0;
    for (let t = 0; t < f.n; t++) { if (f.r[t] < -v[t]) { s += f.r[t] / e[t]; nh++; } z.push(s / ((t + 1) * 0.025) + 1); }
    R[mm] = { z, s, nh };
    const ix = range(250, f.n - 1, st);
    tr.push({ x: ix.map(t => f.dates[t]), y: ix.map(t => z[t]), mode: 'lines', line: { color: MCOL[mm], width: mm === m ? 2.4 : 1.2 }, name: mm, hovertemplate: `${mm}<br>%{x}: %{y:.2f}<extra></extra>` });
  });
  tr.push({ x: [f.dates[250], f.dates[f.n - 1]], y: [0, 0], mode: 'lines', line: { color: DARK, width: 1, dash: 'dash' }, name: 'E[Z2] = 0 under the null', hoverinfo: 'skip' });
  plot('eb-chart', tr, layout({ xaxis: AX('', { type: 'date' }), yaxis: AX(`${NAME[k]}: Z2 on the days so far`), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const o = R[m], Tn = f.n, z2 = o.z[Tn - 1], pz = D.bt[k].pz2[D.bt_starts[k][0].slice(0, 4)][m];
  steps('eb-we', [T.eb1(NAME[k], m, fint(Tn), o.nh, nf(Tn * 0.025, 1)), T.eb2(nf(o.s, 2), fint(Tn), nf(Tn * 0.025, 2), nf(o.s / (Tn * 0.025), 3), nf(z2, 2)), T.eb3(nf(z2, 2), fp(pz), pz < 0.05)]);
  out('eb-out', [T.eb(NAME[k], METH.map(mm => [mm, nf(R[mm].z[Tn - 1], 2), fp(D.bt[k].pz2[D.bt_starts[k][0].slice(0, 4)][mm])]), k !== 'sp500')]);
}

// ================================================================= wiring
const PFX = { def: 'df', 'normal-we': 'nw', horizon: 'hz', 'hs-we': 'hs', precision: 'pr', 'cf-we': 'cf', 'evt-we': 'ev', curves: 'cv', 'var-table': 'vt', 'es-table': 'et',
  'cond-we': 'cw', 'cond-table': 'ct', rolling: 'rv', 'stress-table': 'sx', 'vc-we': 'vc', corr: 'rc', 'tail-dep': 'td', 'copula-we': 'cq', copula: 'cp', 'port-table': 'pt',
  binom: 'bn', 'kupiec-we': 'kp', 'kupiec-sp': 'ks', 'chris-we': 'cr', basel: 'bs', hits: 'ht', 'bt-sd': 'b1', 'bt-bb': 'b2', score: 'sc', traffic: 'tt', 'es-bt': 'eb' };
const BLOCKS = { dfBlock, nwBlock, hzBlock, hsBlock, prBlock, cfBlock, evBlock, cvBlock, vtBlock, etBlock, cwBlock, ctBlock, rvBlock, sxBlock, vcBlock, rcBlock, tdBlock, cqBlock, cpBlock, ptBlock,
  bnBlock, kpBlock, ksBlock, crBlock, bsBlock, htBlock, b1Block, b2Block, scBlock, ttBlock, ebBlock };
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
  const MKR = (cur, L) => (L || MK).filter(k => k !== cur).concat([cur]);
  on(['df-s', 'df-v', 'df-e'], dfBlock); PLy('df-play', 'df-s', MKR('sp500'), 1100);
  on(['nw-s', 'nw-a', 'nw-e'], nwBlock); PLy('nw-play', 'nw-s', MKR('sp500'), 1100);
  on(['hz-s', 'hz-h'], hzBlock); PLy('hz-play', 'hz-h', [10, 1, 2, 3, 5, 7, 15, 20, 10], 900);
  on(['hs-s', 'hs-n'], hsBlock); PLy('hs-play', 'hs-n', ['0', '250', '500', '1000', '2500', '0'], 1100);
  on(['pr-s', 'pr-n'], prBlock); PLy('pr-play', 'pr-s', MKR('sp500'), 1200);
  on(['cf-s', 'cf-a'], cfBlock); PLy('cf-play', 'cf-s', MKR('sp500'), 1100);
  on(['ev-s', 'ev-u', 'ev-a'], evBlock); PLy('ev-play', 'ev-u', ['0.9', '0.85', '0.95', '0.975', '0.9'], 1200);
  on(['cv-s', 'cv-u'], cvBlock); PLy('cv-play', 'cv-u', ['0.9', '0.85', '0.95', '0.975', '0.9'], 1200);
  on(['vt-a'], vtBlock); PLy('vt-play', 'vt-a', ['0.01', '0.005', '0.025', '0.05', '0.01'], 1200);
  on(['et-a'], etBlock); PLy('et-play', 'et-a', ['0.025', '0.01', '0.05', '0.025'], 1200);
  on(['cw-s', 'cw-a'], cwBlock); PLy('cw-play', 'cw-s', MKR('sp500'), 1100);
  on(['ct-l'], ctBlock); PLy('ct-play', 'ct-l', ['0.005|0.01', '0.025|0.05', '0.01|0.025'], 1200);
  on(['rv-s', 'rv-a'], rvBlock); PLy('rv-play', 'rv-s', MKR('sp500', ['sp500', 'dax', 'bet']), 1300);
  on(['sx-a'], sxBlock); PLy('sx-play', 'sx-a', ['0.01', '0.025', '0.01'], 1300);
  on(['vc-w', 'vc-r'], vcBlock); PLy('vc-play', 'vc-w', [0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1, 0.5], 800);
  on(['rc-w'], rcBlock); PLy('rc-play', 'rc-w', ['250', '60', '125', '500', '250'], 1300);
  on(['td-q', 'td-p'], tdBlock); PLy('td-play', 'td-q', [0.05, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.05], 900);
  on(['cq-n', 'cq-r'], cqBlock); PLy('cq-play', 'cq-n', [4.5, 2, 3, 6, 10, 20, 40, 4.5], 900);
  on(['cp-q', 'cp-s'], cpBlock); PLy('cp-play', 'cp-q', [0.05, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.05], 1000);
  on(['pt-w'], ptBlock); PLy('pt-play', 'pt-w', ['0.5', '0.1', '0.25', '0.75', '0.9', '0.5'], 1100);
  on(['bn-p', 'bn-n'], bnBlock); PLy('bn-play', 'bn-p', [0.01, 0.005, 0.0075, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.01], 900);
  on(['kp-x', 'kp-n'], kpBlock); PLy('kp-play', 'kp-x', [7, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 7], 700);
  on(['ks-s', 'ks-m'], ksBlock); PLy('ks-play', 'ks-m', ['HS', 'Normal', 'FHS', 'GARCH-t'], 1200);
  on(['cr-s', 'cr-m'], crBlock); PLy('cr-play', 'cr-s', MKR('sp500', BTK), 1200);
  on(['bs-s', 'bs-m', 'bs-x', 'bs-d'], bsBlock); PLy('bs-play', 'bs-m', ['Normal', 'GARCH-t', 'FHS', 'HS'], 1300);
  on(['ht-s', 'ht-a'], htBlock); PLy('ht-play', 'ht-s', MKR('sp500', BTK), 1200);
  on(['b1-y'], b1Block); PLy('b1-play', 'b1-y', ['0', '1', '2', '3', '0'], 1300);
  on(['b2-y'], b2Block); PLy('b2-play', 'b2-y', ['0', '1', '2', '3', '0'], 1300);
  on(['sc-d'], scBlock); PLy('sc-play', 'sc-d', ['2009-03-31', '2020-05-29', '2022-12-30', D.end], 1300);
  on(['tt-s'], ttBlock); PLy('tt-play', 'tt-s', MKR('sp500', BTK), 1200);
  on(['eb-s', 'eb-m'], ebBlock); PLy('eb-play', 'eb-s', MKR('sp500', BTK), 1200);
  const VEIL = T.veil || {};
  if (T.predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: PLAYERS[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH10 = { ser, methods, cond, frame, hits, kupiec, christ, roll250, zone, z2stat, qloss, btRow, port, pobs, chiQ, joint1, tailDep, rollCorr, corrOf, pstats, vcCalc, capital, curveOf,
  hzCurve, potVar, potEs, potOf, cfz, tppf, stdtQ, stdtEs, normppf, chi2sf, bincdf, binppf, binpmf, hsVar, hsEs, kOf, init, PFX, blocks: BLOCKS };
if (window.CH10_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
