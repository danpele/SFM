// Chapter 7 interactive page: one block per chart slide and per results table of the lecture, computed in the browser from ch7_data.js.
(function () {
'use strict';
const D = window.CH7_DATA;
const { C, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket, solve, LANG } = window.SFMI;
const T = window.CH7_TEXT;
const NUM = D.N, NAMES = D.names, ASSETS = D.assets, COL = D.col, NAME = D.name;
const DARK = C.navy, GREEN = '#2E7D32', RED = '#CD0000', BLUE = '#1A3A6E', TEAL = '#17A2B8', PURPLE = '#8E44AD', ORANGE = '#E67E22', AMBER = '#B5853F';
const SH = { sp500: 'S&P 500', dax: 'DAX', bet: 'BET', btc: 'Bitcoin', tlv: 'TLV', snp: 'SNP', brd: 'BRD', tgn: 'TGN' };

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
function chi2ppf(p, df) { let lo = 0, hi = 10 * df + 100; for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (1 - chi2sf(m, df) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
const p2 = z => 2 * (1 - Phi(Math.abs(z)));

// ================================================================= data
const iso = d => new Date(d * 864e5).toISOString().slice(0, 10);
function unpack(R) { const dd = [R.d0]; for (const v of R.dd) dd.push(dd[dd.length - 1] + v); return dd.map(iso); }
const SER = {}, LP = {}, MK = {};
function ser(k) { return SER[k] || (SER[k] = { x: D.ret[k].x, dates: unpack(D.ret[k]), n: D.ret[k].x.length }); }
function lpSer(k) {                 // log price, rebuilt from the integer differences (units of 1e-7)
  if (LP[k]) return LP[k];
  const R = D.lp[k], p = new Float64Array(R.dp.length + 1); let v = R.p0; p[0] = v / 1e7;
  R.dp.forEach((d, i) => { v += d; p[i + 1] = v / 1e7; });
  return (LP[k] = { p, dates: unpack(R), n: p.length });
}
function mkSer(k) { return MK[k] || (MK[k] = { x: D.mk[k].x, dates: unpack(D.mk[k]), n: D.mk[k].x.length }); }
const startIdx = (S, y) => { const d = `${y}-01-01`, i = S.dates.findIndex(v => v >= d); return i < 0 ? S.n : i; };
const fromYear = (k, y) => { const S = ser(k), i = startIdx(S, y); return { x: S.x.slice(i), dates: S.dates.slice(i), n: S.n - i }; };

// ================================================================= statistics of the chapter (generate_all_charts.py)
function demean(x) { let m = 0; for (const v of x) m += v; m /= x.length; return Float64Array.from(x, v => v - m); }
function acf(x, nlags) {            // rho_hat(1..nlags) = sum e_t e_{t-k} / sum e_t^2
  const e = demean(x), n = e.length; let s = 0; for (const v of e) s += v * v;
  const r = []; for (let k = 1; k <= nlags; k++) { let c = 0; for (let t = k; t < n; t++) c += e[t] * e[t - k]; r.push(c / s); } return r;
}
function robVar(x, nlags) {         // delta_hat(k)/T = sum e_t^2 e_{t-k}^2 / (sum e_t^2)^2
  const e = demean(x), n = e.length, e2 = e.map(v => v * v); let s = 0; for (const v of e2) s += v; const s2 = s * s;
  const r = []; for (let k = 1; k <= nlags; k++) { let c = 0; for (let t = k; t < n; t++) c += e2[t] * e2[t - k]; r.push(c / s2); } return r;
}
function portmanteau(x, m) {
  const T = x.length, rho = acf(x, m), vr = robVar(x, m); let bp = 0, lb = 0, rob = 0;
  rho.forEach((r, i) => { bp += r * r; lb += r * r / (T - i - 1); rob += r * r / vr[i]; });
  bp *= T; lb *= T * (T + 2);
  return { T, m, rho, vr, rho1: rho[0], bp, lb, rob, p_bp: chi2sf(bp, m), p_lb: chi2sf(lb, m), p_rob: chi2sf(rob, m), crit: chi2ppf(0.95, m) };
}
function runsTest(x) {
  const s = []; for (const v of x) if (v !== 0) s.push(v > 0 ? 1 : -1);
  let n1 = 0; for (const v of s) if (v > 0) n1++; const n2 = s.length - n1, n = n1 + n2;
  let runs = 1; for (let i = 1; i < s.length; i++) if (s[i] !== s[i - 1]) runs++;
  const mean = 2 * n1 * n2 / n + 1, vr = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1)), z = (runs - mean) / Math.sqrt(vr);
  return { n1, n2, n, runs, mean, var: vr, sd: Math.sqrt(vr), z, p: p2(z) };
}
function varianceRatio(x, q) {      // Lo and MacKinlay (1988): overlapping VR(q), Z(q), Z*(q)
  const T = x.length; let mu = 0; for (const v of x) mu += v; mu /= T;
  const e = Float64Array.from(x, v => v - mu); let ss = 0; for (const v of e) ss += v * v;
  const varA = ss / (T - 1); let s = 0; for (let t = 0; t < q; t++) s += x[t];
  let sc = (s - q * mu) ** 2; for (let t = q; t < T; t++) { s += x[t] - x[t - q]; sc += (s - q * mu) ** 2; }
  const m = q * (T - q + 1) * (1 - q / T), varC = sc / m, vr = varC / varA;
  const seIid = Math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T)), z = (vr - 1) / seIid;
  const den = ss * ss, e2 = e.map(v => v * v), delta = []; let theta = 0;
  for (let j = 1; j < q; j++) { let c = 0; for (let t = j; t < T; t++) c += e2[t] * e2[t - j]; const dj = T * c / den; delta.push(dj); theta += (2 * (q - j) / q) ** 2 * dj; }
  const seRob = Math.sqrt(theta / T), zs = (vr - 1) / seRob;
  return { q, T, mu, varA, varC, m, vr, z, zs, seIid, seRob, theta, delta, p_z: p2(z), p_zs: p2(zs) };
}
const cdCrit = (m, a) => normPpf(1 - (1 - (1 - a) ** (1 / m)) / 2);
function chowDenning(x, qs, a) {
  const zs = qs.map(q => varianceRatio(x, q).zs), cd = Math.max(...zs.map(Math.abs)), m = qs.length;
  return { zs, cd, m, crit: cdCrit(m, a || 0.05), p: 1 - (2 * Phi(cd) - 1) ** m };
}
function fft(re, im) {              // in place, radix 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let j = 0; j < len / 2; j++) { const a = i + j, b = a + len / 2, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
      re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
function acfAll(e) {                // all sample autocorrelations rho_hat(1..n-1), by the FFT
  const n = e.length; let N = 1; while (N < 2 * n) N <<= 1;
  const re = new Float64Array(N), im = new Float64Array(N); re.set(e); fft(re, im);
  for (let i = 0; i < N; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
  fft(re, im); const c0 = re[0], r = new Float64Array(n - 1); for (let k = 1; k < n; k++) r[k - 1] = re[(N - k) % N] / c0; return r;
}
const qsKernel = z => { const a = 6 * Math.PI * z / 5; return 25 / (12 * Math.PI * Math.PI * z * z) * (Math.sin(a) / a - Math.cos(a)); };
function autoVR(x) {                // Choi (1999), Andrews (1991) AR(1) plug-in bandwidth
  const e = demean(x), T = e.length; let a = 0, b = 0; for (let t = 1; t < T; t++) { a += e[t] * e[t - 1]; b += e[t - 1] * e[t - 1]; }
  const rho = a / b, a2 = 4 * rho * rho / (1 - rho) ** 4, k = 1.3221 * (a2 * T) ** 0.2, r = acfAll(e); let s = 0;
  for (let i = 1; i < T; i++) s += qsKernel(i / k) * r[i - 1];
  const vr = 1 + 2 * s; return { rho, a2, k, vr, stat: Math.sqrt(T / k) * (vr - 1) / Math.SQRT2, T };
}
// OLS with explicit columns; returns coefficients, standard errors, residuals
function ols(y, cols) {
  const n = y.length, k = cols.length, A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let i = 0; i < k; i++) { const ci = cols[i]; for (let t = 0; t < n; t++) b[i] += ci[t] * y[t]; for (let j = 0; j <= i; j++) { const cj = cols[j]; let s = 0; for (let t = 0; t < n; t++) s += ci[t] * cj[t]; A[i][j] = A[j][i] = s; } }
  const beta = solve(A, b), u = Float64Array.from(y); for (let i = 0; i < k; i++) { const ci = cols[i]; for (let t = 0; t < n; t++) u[t] -= beta[i] * ci[t]; }
  let ssr = 0; for (const v of u) ssr += v * v; const s2 = ssr / (n - k);
  const inv = j => solve(A, A.map((_, i) => (i === j ? 1 : 0)))[j];
  return { beta, u, ssr, s2, n, k, A, se: j => Math.sqrt(s2 * inv(j)), ixx: inv };
}
const centred = a => { let m = 0; for (const v of a) m += v; m /= a.length; return Float64Array.from(a, v => v - m); };
const ADF = {};
function adf(y, reg, L) {           // statsmodels adfuller with a fixed lag L: dy_t on y_{t-1}, L lagged dy, constant (and trend)
  const n = y.length, dy = new Float64Array(n - 1); for (let t = 1; t < n; t++) dy[t - 1] = 100 * (y[t] - y[t - 1]);
  const nobs = n - 1 - L, Y = dy.slice(L), lev = new Float64Array(nobs); for (let t = 0; t < nobs; t++) lev[t] = y[L + t];
  const cols = [centred(lev)]; for (let j = 1; j <= L; j++) cols.push(dy.slice(L - j, L - j + nobs));
  cols.push(new Float64Array(nobs).fill(1)); if (reg === 'ct') cols.push(centred(Float64Array.from({ length: nobs }, (_, t) => (t + 1) / nobs)));
  const R = ols(Y, cols), g = R.beta[0] / 100, se = R.se(0) / 100;
  return { gamma: g, se, tau: g / se, nobs, L, reg, k: cols.length };
}
function adfCached(k, kind, L) {    // kind: 'price' (constant and trend) or 'ret' (constant); returns in % are divided by 100
  const key = k + kind + L; if (ADF[key]) return ADF[key];
  const y = kind === 'price' ? lpSer(k).p : ser(k).x.map(v => v / 100);
  return (ADF[key] = adf(y, kind === 'price' ? 'ct' : 'c', L));
}
const rule = T => Math.ceil(12 * (T / 100) ** 0.25);
function ppTest(y, reg, l) {        // pp_test of the chapter (Phillips-Perron Z_t)
  const T = y.length - 1, Y = Float64Array.from(y.slice(1)), lag = centred(y.slice(0, T));
  const cols = [new Float64Array(T).fill(1)]; if (reg === 'ct') cols.push(centred(Float64Array.from({ length: T }, (_, t) => (t + 1) / T))); cols.push(lag);
  const R = ols(Y, cols), j = cols.length - 1, se = R.se(j), tr = (R.beta[j] - 1) / se, u = R.u, lags = l == null ? rule(T) : l;
  let g0 = 0; for (const v of u) g0 += v * v; g0 /= T; let lam2 = g0;
  for (let i = 1; i <= lags; i++) { let c = 0; for (let t = i; t < T; t++) c += u[t] * u[t - i]; lam2 += 2 * (1 - i / (lags + 1)) * c / T; }
  const z = Math.sqrt(g0 / lam2) * tr - (lam2 - g0) / (2 * Math.sqrt(lam2)) * T * se / Math.sqrt(R.s2);
  return { stat: z, lags, phi: R.beta[j] };
}
function kpssTest(y, reg, l) {      // statsmodels kpss, nlags='legacy' unless l is given
  const n = y.length; let u;
  if (reg === 'ct') { const tt = Float64Array.from({ length: n }, (_, t) => (t + 1) / n); u = ols(Float64Array.from(y), [new Float64Array(n).fill(1), centred(tt)]).u; } else u = centred(y);
  const lags = Math.min(l == null ? Math.ceil(12 * (n / 100) ** 0.25) : l, n - 1);
  let S = 0, eta = 0; for (const v of u) { S += v; eta += S * S; } eta /= n * n;
  let s = 0; for (const v of u) s += v * v; const g0 = s / n;
  for (let i = 1; i <= lags; i++) { let c = 0; for (let t = i; t < n; t++) c += u[t] * u[t - i]; s += 2 * c * (1 - i / (lags + 1)); }
  return { stat: eta / (s / n), eta, lam2: s / n, g0, lags, n };
}
const KPSS = {}, PP = {};
function kpssC(k, kind, l) { const key = k + kind + l; return KPSS[key] || (KPSS[key] = kpssTest(kind === 'price' ? lpSer(k).p : ser(k).x, kind === 'price' ? 'ct' : 'c', l)); }
function ppC(k, kind, l) { const key = k + kind + l; return PP[key] || (PP[key] = ppTest(kind === 'price' ? lpSer(k).p : ser(k).x, kind === 'price' ? 'ct' : 'c', l)); }
function hacOls(y, X, L) {          // OLS with Newey-West (Bartlett, L lags) covariance, small-sample correction n/(n-k)
  const n = y.length, k = X[0].length, A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let t = 0; t < n; t++) for (let i = 0; i < k; i++) { b[i] += X[t][i] * y[t]; for (let j = 0; j < k; j++) A[i][j] += X[t][i] * X[t][j]; }
  const beta = solve(A, b), u = y.map((v, t) => v - X[t].reduce((s, x, i) => s + x * beta[i], 0));
  const S = Array.from({ length: k }, () => new Array(k).fill(0)), g = X.map((r, t) => r.map(v => v * u[t]));
  for (let l = 0; l <= L; l++) { const w = l === 0 ? 1 : 1 - l / (L + 1); for (let t = l; t < n; t++) for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { const v = w * g[t][i] * g[t - l][j]; S[i][j] += v; if (l) S[j][i] += v; } }
  const Ai = A.map((_, j) => solve(A, A.map((__, i) => (i === j ? 1 : 0)))), V = Ai.map(r => Ai[0].map((_, j) => 0));
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { let s = 0; for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) s += Ai[i][a] * S[a][c] * Ai[c][j]; V[i][j] = s * n / (n - k); }
  return { beta, V, se: V.map((r, i) => Math.sqrt(r[i])) };
}
function calendar(x, dates, L, view) {   // means by weekday (or month) with HAC 95% intervals, Wald test of equality, January effect
  const G = view === 'month' ? 12 : 5, grp = dates.map(d => (view === 'month' ? +d.slice(5, 7) - 1 : new Date(d + 'T00:00:00Z').getUTCDay() - 1));
  const X = grp.map(gi => { const r = new Array(G).fill(0); r[gi] = 1; return r; }), H = hacOls(x, X, L);
  const R = Array.from({ length: G - 1 }, (_, i) => { const r = new Array(G).fill(0); r[0] = 1; r[i + 1] = -1; return r; });
  const Rb = R.map(r => r.reduce((s, v, i) => s + v * H.beta[i], 0)), RV = R.map(r => R.map(c => r.reduce((s, v, i) => s + v * c.reduce((ss, w, j) => ss + w * H.V[i][j], 0), 0)));
  const sol = solve(RV, Rb), wald = Rb.reduce((s, v, i) => s + v * sol[i], 0);
  const jan = dates.map(d => (d.slice(5, 7) === '01' ? 1 : 0)), J = hacOls(x, jan.map(j => [1, j]), L);
  let sj = 0, nj = 0, so = 0, no = 0; x.forEach((v, i) => { if (jan[i]) { sj += v; nj++; } else { so += v; no++; } });
  return { means: H.beta, ses: H.se, wald, df: G - 1, wald_p: chi2sf(wald, G - 1), jan_mean: sj / nj, other_mean: so / no, nj, no, jan_b: J.beta[1], jan_se: J.se[1], jan_t: J.beta[1] / J.se[1], jan_p: p2(J.beta[1] / J.se[1]), n: x.length };
}
function armaAcf(ar, ma, lags) {
  const n = 2000, psi = new Float64Array(n); psi[0] = 1;
  for (let i = 1; i < n; i++) { let v = i - 1 < ma.length ? ma[i - 1] : 0; for (let j = 0; j < ar.length; j++) if (i - 1 - j >= 0) v += ar[j] * psi[i - 1 - j]; psi[i] = v; }
  const g = []; for (let k = 0; k <= lags; k++) { let s = 0; for (let i = 0; i + k < n; i++) s += psi[i] * psi[i + k]; g.push(s); }
  return g.slice(1).map(v => v / g[0]);
}
function rollingVR(k, q, w, step) {
  const S = ser(k), rows = [];
  for (let end = w; end <= S.n; end += step) { const v = varianceRatio(S.x.slice(end - w, end), q); rows.push({ date: S.dates[end - 1], vr: v.vr, zs: v.zs }); }
  return rows;
}
function subperiods(q) {
  const out = {};
  Object.entries(D.subperiods).forEach(([k, per]) => {
    const S = ser(k);
    out[k] = per.map(([a, b]) => { const idx = []; S.dates.forEach((d, i) => { if (d >= a && (!b || d <= b)) idx.push(i); }); const x = idx.map(i => S.x[i]), v = varianceRatio(x, q);
      return { from: S.dates[idx[0]], to: S.dates[idx[idx.length - 1]], n: x.length, rho1: acf(x, 1)[0], vr: v.vr, zs: v.zs, se: v.seRob, cd_p: chowDenning(x, D.qs).p }; });
  });
  return out;
}
function realSd(x, q) {             // standard deviation of the overlapping q-day sums
  const c = [0]; for (const v of x) c.push(c[c.length - 1] + v); const s = []; for (let t = 0; t + q <= x.length; t++) s.push(c[t + q] - c[t]);
  let m = 0; for (const v of s) m += v; m /= s.length; let v2 = 0; for (const v of s) v2 += (v - m) ** 2; return Math.sqrt(v2 / (s.length - 1));
}
const sdev = x => { let m = 0; for (const v of x) m += v; m /= x.length; let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / (x.length - 1)); };

// ================================================================= formatting, tables, panels
const sep = LANG === 'ro' ? ' ' : ',';
const fint = x => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, sep).replace('-', '−');
const fnum = (x, d) => { const s = nf(x, d), i = s.search(/[.,]/), a = i < 0 ? s : s.slice(0, i), b = i < 0 ? '' : s.slice(i); return a.replace(/\B(?=(\d{3})+(?!\d))/g, sep) + b; };
function sig(x, d) {
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (Math.abs(x) >= 1e-3) return nf(x, Math.max(0, d - 1 - e));
  let m = x / 10 ** e, ee = e; if (+Math.abs(m).toFixed(d - 1) >= 10) { m /= 10; ee++; }
  return nf(m, d - 1) + ' × 10<sup>' + (ee < 0 ? '−' : '') + Math.abs(ee) + '</sup>';
}
const fp = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const fpe = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : '= ' + nf(p, 3));
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
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
function gridp(id, R, Cn, o) {
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
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
const segs = (ys, lo, hi) => { const x = [], y = []; ys.forEach((v, i) => { x.push(lo[i], hi[i], null); y.push(v, v, null); }); return { x, y }; };   // horizontal interval segments
const vsegs = (xs, lo, hi) => { const x = [], y = []; xs.forEach((v, i) => { x.push(v, v, null); y.push(lo[i], hi[i], null); }); return { x, y }; };   // vertical interval segments
const LOGT = { type: 'log', tickvals: [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], ticktext: ['0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'] };

// ================================================================= 1. which one is the S&P 500? (random walks)
function rwBlock() {
  const y = +$('rw-y').value, P = lpSer('sp500'), R = D.rw, lim = `${y}-12-31`;
  label('rw-y').textContent = String(y);
  const nIn = R.idx.filter(i => P.dates[i] <= lim).length, ii = R.idx.slice(0, nIn), xs = ii.map(i => P.dates[i]), nw = single('rw-chart', 400);
  const cols = [RED, GREEN, PURPLE, ORANGE];
  const tr = R.sims.map((s, j) => ({ x: xs, y: s.slice(0, nIn), mode: 'lines', line: { color: cols[j], width: 1 }, name: `Simulated random walk ${j + 1}`, hovertemplate: `walk ${j + 1}<br>%{x}<br>%{y:.2f}<extra></extra>` }));
  tr.push({ x: xs, y: ii.map(i => P.p[i] - P.p[0]), mode: 'lines', line: { color: COL.sp500, width: 1.6 }, name: 'S&P 500 (log price minus its first value)', hovertemplate: 'S&P 500<br>%{x}<br>%{y:.2f}<extra></extra>' });
  plot('rw-chart', tr, layout({ xaxis: AX('', { range: [P.dates[0], P.dates[P.n - 1]] }), yaxis: AX('Log price (start = 0)', { range: [-1.2, 5] }), legend: LEG(nw, -0.1), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const last = nIn - 1, fin = R.sims.map(s => s[last]), sp = P.p[ii[last]] - P.p[0];
  out('rw-out', [T.rw(xs[last], nf(sp, 2), nf(Math.min(...fin), 2), nf(Math.max(...fin), 2), fin.filter(v => v > sp).length)]);
}

// ================================================================= 2. does yesterday predict today?
function lagStats(x, k) { const a = x.slice(0, x.length - k), b = x.slice(k); const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let sab = 0, saa = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; }); const sl = sab / saa; return { a, b, slope: sl, icpt: mb - sl * ma, rho: acf(x, k)[k - 1] }; }
function scBlock() {
  const k = +$('sc-k').value; label('sc-k').textContent = String(k);
  const Pn = panels('sc-chart', 2, { h: 400, hs: 300, extra: 120 }), tr = [], ann = [], st = {};
  ['sp500', 'bet'].forEach((n, j) => {
    const L = lagStats(ser(n).x, k), a = Pn.ax[j]; st[n] = L;
    tr.push(on2(a, { x: L.a, y: L.b, mode: 'markers', marker: { size: 3, color: hexA(COL[n], 0.35) }, name: `${NAME[n]}: daily returns (r(t-${k}), r(t))`, hoverinfo: 'skip' }));
    tr.push(on2(a, { x: [-10, 10], y: [L.icpt - 10 * L.slope, L.icpt + 10 * L.slope], mode: 'lines', line: { color: '#000', width: 1.4 }, name: `${NAME[n]}: least-squares line`, hoverinfo: 'skip' }));
    setAx(Pn, j, AX(`Return ${k === 1 ? 'yesterday' : k + ' days ago'}, r(t-${k}) (%)`, { range: [-10, 10] }), AX('Return today, r(t) (%)', { range: [-10, 10] }));
    ann.push(ptitle(a, `${NAME[n]}: correlation ${L.rho.toFixed(3)}`));
  });
  plot('sc-chart', tr, layout(Object.assign(Pn.L, { annotations: ann, legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.18) })));
  out('sc-out', [T.sc(k, nf(st.sp500.rho, 3), nf(st.sp500.slope, 3), nf(st.bet.rho, 3), nf(st.bet.slope, 3))]);
}

// ================================================================= 3. worked example: how risk grows with the horizon (table block)
const ST_Q = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150, 200, 250];
function stBlock() {
  const k = $('st-s').value, q0 = +$('st-q').value, x = ser(k).x, s = sdev(x), nw = single('st-chart', 360);
  label('st-q').textContent = String(q0);
  const re = ST_Q.map(q => realSd(x, q));
  plot('st-chart', [{ x: ST_Q, y: ST_Q.map(q => s * Math.sqrt(q)), mode: 'lines', line: { color: DARK, dash: 'dash', width: 1.2 }, name: 'sigma x sqrt(q) (random walk)', hovertemplate: 'q = %{x}<br>%{y:.2f}%<extra></extra>' },
    { x: ST_Q, y: re, mode: 'lines+markers', line: { color: COL[k], width: 1.6 }, marker: { size: 5 }, name: `${NAME[k]}: sd of overlapping q-day returns`, hovertemplate: 'q = %{x}<br>%{y:.2f}%<extra></extra>' },
    { x: [q0], y: [realSd(x, q0)], mode: 'markers', marker: { size: 12, symbol: 'star', color: ORANGE, line: { color: '#fff', width: 1 } }, name: `worked example: q = ${q0}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('Horizon q (days)', { type: 'log', tickvals: [1, 2, 5, 10, 20, 50, 100, 250], ticktext: ['1', '2', '5', '10', '20', '50', '100', '250'] }), yaxis: AX('Standard deviation (%)', { type: 'log', tickvals: [1, 2, 5, 10, 20, 50, 100], ticktext: ['1', '2', '5', '10', '20', '50', '100'] }), legend: LEG(nw, -0.2), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  table('st-tab', ['q', 'σ̂√q (%)', T.h.real, T.h.ratio, 'VR(q)'], [1, 5, 20, 60, 250].map(q => { const r = realSd(x, q), v = q > 1 ? varianceRatio(x, q).vr : 1;
    return [String(q), nf(s * Math.sqrt(q), 2), nf(r, 2), nf(r / (s * Math.sqrt(q)), 3), nf(v, 3)]; }));
  const f = q => [nf(Math.sqrt(q), 3), nf(s * Math.sqrt(q), 2)];
  steps('st-we', [T.st1(SH[k], nf(s, 4), fint(x.length)), T.st2(nf(s, 4), ...f(5), ...f(20), ...f(250)),
    T.st3(q0, nf(s, 4), ...f(q0), nf(realSd(x, q0), 2), nf(realSd(x, q0) / (s * Math.sqrt(q0)), 3))]);
  out('st-out', [T.st(SH[k], nf(s * Math.sqrt(250), 2), nf(realSd(x, 250), 2))]);
}

// ================================================================= 4. Gaussian white noise and its sample ACF
function wnBlock() {
  const n = +$('wn-n').value; label('wn-n').textContent = fint(n);
  const x = D.wn.slice(0, n), r = acf(x, 30), b = 1.96 / Math.sqrt(n), ks = range(1, 30, 1);
  const Pn = panels('wn-chart', 2, { h: 360, hs: 270, extra: 120, gap: 0.1 }), [a, c] = Pn.ax;
  Pn.L.xaxis.domain = Pn.stack ? [0, 1] : [0, 0.58]; Pn.L.xaxis2.domain = Pn.stack ? [0, 1] : [0.68, 1];
  const tr = [on2(a, { x: range(1, n, 1), y: x, mode: 'lines', line: { color: BLUE, width: 0.6 }, name: 'Gaussian white noise, N(0, 1)', hoverinfo: 'skip' }),
    on2(c, { type: 'bar', x: ks, y: r, marker: { color: r.map(v => (Math.abs(v) > b ? ORANGE : RED)) }, width: 0.5, name: 'Sample ACF', hovertemplate: 'k = %{x}<br>%{y:.3f}<extra></extra>' }),
    on2(c, { x: [0.5, 30.5, null, 0.5, 30.5], y: [b, b, null, -b, -b], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.8 }, name: '95% band: +/- 1.96/sqrt(n)', hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('t', { range: [0, 1000] }), AX('x(t)', { range: [-4, 4] })); setAx(Pn, 1, AX('Lag k'), AX('rho_hat(k)', { range: [-0.3, 0.3] }));
  plot('wn-chart', tr, layout(Object.assign(Pn.L, { legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  out('wn-out', [T.wn(fint(n), nf(b, 3), r.filter(v => Math.abs(v) > b).length, nf(30 * 0.05, 1))]);
}

// ================================================================= 5. theoretical and sample ACF of six processes
function apBlock() {
  const n = +$('ap-n').value; label('ap-n').textContent = fint(n);
  const G = gridp('ap-chart', 2, 3, { h: 520, hs: 220, gy: 0.22 }), tr = [], ann = [], ks = range(1, 20, 1), rows = [];
  D.arma.forEach((P, i) => {
    const a = G.ax[i], th = armaAcf(P.ar, P.ma, 20), sm = acf(P.x.slice(0, n), 20);
    tr.push(on2(a, { type: 'bar', x: ks, y: th, marker: { color: BLUE }, width: 0.55, name: 'Theoretical ACF', showlegend: i === 0, legendgroup: 'th', hovertemplate: 'k = %{x}<br>%{y:.3f}<extra></extra>' }));
    tr.push(on2(a, { x: ks, y: sm, mode: 'markers', marker: { size: 6, color: RED }, name: `Sample ACF, n = ${n}`, showlegend: i === 0, legendgroup: 'sm', hovertemplate: 'k = %{x}<br>%{y:.3f}<extra></extra>' }));
    setAx(G, i, AX(G.rows === 2 && i < 3 ? '' : 'Lag k'), AX(i % 3 === 0 || G.nw ? 'rho(k)' : '', { range: [-1, 1] }));
    ann.push(ptitle(a, P.lab));
    rows.push(Math.max(...sm.map((v, j) => Math.abs(v - th[j]))));
  });
  plot('ap-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.04 : -0.12), margin: { l: 50, r: 10, t: 24, b: 40 } })));
  const lab = P => { const v = P.ar.length ? P.ar : P.ma, g = P.ar.length ? 'α' : 'β'; return `${P.ar.length ? 'AR' : 'MA'}(${v.length}), ` + v.map((c, j) => `${g}${v.length > 1 ? '<sub>' + (j + 1) + '</sub>' : ''} = ${nf(c, 1)}`).join(', '); };
  out('ap-out', [T.ap(fint(n), nf(1.96 / Math.sqrt(n), 3), D.arma.map((P, i) => [lab(P), nf(rows[i], 3)]))]);
}

// ================================================================= 6. the ACF of daily returns (+ worked example: the robust band)
function arBlock() {
  const y = +$('ar-y').value; label('ar-y').textContent = String(y);
  const G = gridp('ar-chart', 2, 2, { h: 520, hs: 250 }), tr = [], ann = [], ks = range(1, 20, 1), items = [];
  ASSETS.forEach((n, i) => {
    const S = fromYear(n, y), r = acf(S.x, 20), rb = robVar(S.x, 20).map(v => 1.96 * Math.sqrt(v)), b = 1.96 / Math.sqrt(S.n), a = G.ax[i];
    tr.push(on2(a, { x: ks, y: rb, mode: 'lines', line: { width: 0, color: TEAL }, showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(a, { x: ks, y: rb.map(v => -v), mode: 'lines', line: { width: 0, color: TEAL }, fill: 'tonexty', fillcolor: hexA(TEAL, 0.18), name: 'Robust 95% band', showlegend: i === 0, legendgroup: 'rb', hoverinfo: 'skip' }));
    tr.push(on2(a, { type: 'bar', x: ks, y: r, marker: { color: COL[n] }, width: 0.55, name: 'Sample ACF of the returns', showlegend: i === 0, legendgroup: 'acf', hovertemplate: 'k = %{x}<br>%{y:.3f}<extra></extra>' }));
    tr.push(on2(a, { x: [0.5, 20.5, null, 0.5, 20.5], y: [b, b, null, -b, -b], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.8 }, name: 'i.i.d. 95% band', showlegend: i === 0, legendgroup: 'iid', hoverinfo: 'skip' }));
    setAx(G, i, AX(G.rows === 2 && i < 2 ? '' : 'Lag k (days)'), AX('', { range: [-0.2, 0.2] }));
    ann.push(ptitle(a, `${NAME[n]}, from ${S.dates[0].slice(0, 7)}`));
    items.push([SH[n], nf(r[0], 3), r.filter(v => Math.abs(v) > b).length, r.filter((v, j) => Math.abs(v) > rb[j]).length]);
  });
  plot('ar-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.12), margin: { l: 50, r: 10, t: 24, b: 40 } })));
  out('ar-out', [T.ar(y, items)]);
  arWe();
}
function arWe() {
  const k = $('ar-s').value, j = +$('ar-k').value, y = +$('ar-y').value, S = fromYear(k, y); label('ar-k').textContent = String(j);
  const e = demean(S.x), Tn = e.length; let s2 = 0, s4 = 0; for (const v of e) s2 += v * v; for (let t = j; t < Tn; t++) s4 += e[t] * e[t] * e[t - j] * e[t - j];
  const d = Tn * s4 / (s2 * s2), r = acf(S.x, j)[j - 1];
  steps('ar-we', [T.ar1(SH[k], fint(Tn), j, fnum(s4, 1), fnum(s2, 1), nf(d, 3)), T.ar2(nf(1.96 / Math.sqrt(Tn), 3), nf(d, 3), fint(Tn), nf(1.96 * Math.sqrt(d / Tn), 3)),
    T.ar3(j, nf(r, 3), Math.abs(r) > 1.96 / Math.sqrt(Tn), Math.abs(r) > 1.96 * Math.sqrt(d / Tn))]);
}

// ================================================================= 7. returns almost uncorrelated, their size is not
function aaBlock() {
  const dd = +$('aa-d').value, nw = single('aa-chart', 400), ks = range(1, 100, 1), tr = [], items = [];
  label('aa-d').textContent = nf(dd, 2);
  ['sp500', 'bet', 'btc'].forEach(n => {
    const x = ser(n).x, ra = acf(x.map(v => Math.abs(v) ** dd), 100), r = acf(x, 100);
    tr.push({ x: ks, y: ra, mode: 'lines', line: { color: COL[n], width: 1.6 }, name: `${NAME[n]}: |returns|^${dd.toFixed(2)}`, hovertemplate: `${NAME[n]}<br>k = %{x}<br>%{y:.3f}<extra></extra>` });
    tr.push({ x: ks, y: r, mode: 'lines', line: { color: COL[n], width: 0.9, dash: 'dot' }, name: `${NAME[n]}: returns`, hovertemplate: `${NAME[n]}<br>k = %{x}<br>%{y:.3f}<extra></extra>` });
    items.push([NAME[n], nf(ra[0], 2), nf(ra[19], 2), nf(ra[99], 2)]);
  });
  plot('aa-chart', tr, layout({ xaxis: AX('Lag k (days)'), yaxis: AX('Sample autocorrelation', { range: [-0.15, 0.6] }), legend: LEG(nw, -0.16), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  out('aa-out', [T.aa(nf(dd, 2), items)]);
}

// ================================================================= 8. worked example: Ljung-Box step by step (deck numbers)
const LB_R = [0.06, -0.04, 0.03];
function lbBlock() {
  const Tn = +$('lb-T').value, m = +$('lb-m').value, nw = single('lb-chart', 340); label('lb-T').textContent = fint(Tn);
  const r = LB_R.slice(0, m), bp = r.map(v => Tn * v * v), lb = r.map((v, i) => Tn * (Tn + 2) * v * v / (Tn - i - 1)), crit = chi2ppf(0.95, m);
  const BP = bp.reduce((s, v) => s + v, 0), LB = lb.reduce((s, v) => s + v, 0), cats = ['Box-Pierce Q(m)', 'Ljung-Box Q(m)'], cl = [BLUE, TEAL, AMBER];
  const tr = r.map((_, i) => ({ type: 'bar', x: cats, y: [bp[i], lb[i]], marker: { color: cl[i] }, name: `lag ${i + 1}: rho_hat(${i + 1}) = ${LB_R[i].toFixed(2)}`, hovertemplate: '%{y:.2f}<extra></extra>' }));
  tr.push({ x: cats, y: [crit, crit], mode: 'lines+markers', line: { color: RED, dash: 'dash', width: 1.4 }, marker: { size: 6 }, name: `5% critical value chi2(${m}) = ${crit.toFixed(2)}`, hoverinfo: 'skip' });
  plot('lb-chart', tr, layout({ barmode: 'stack', xaxis: AX(''), yaxis: AX('Statistic (sum of the lag terms)', { range: [0, Math.max(35, BP * 1.1)] }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  steps('lb-we', [T.lb1(fint(Tn), r.map(v => nf(v, 2)).join('; '), m), T.lb2(bp.map(v => nf(v, 2)).join('; '), nf(BP, 2)),
    T.lb3(fint(Tn), fint(Tn + 2), r.map((v, i) => `${nf(Math.abs(v), 2)}²/${fint(Tn - i - 1)}`).join(' + '), nf(LB, 2)),
    T.lb4(m, nf(crit, 2), nf(LB, 2), LB > crit, fpe(chi2sf(LB, m)))]);
  out('lb-out', [T.lb(fint(Tn), nf(BP, 2), nf(LB, 2), LB > crit, nf(1.96 / Math.sqrt(Tn), 3))]);
}

// ================================================================= 9. autocorrelation tests on real data (table)
const PM = {};
function pmC(k, m) { const key = k + m; return PM[key] || (PM[key] = { r: portmanteau(ser(k).x, m), s: portmanteau(ser(k).x.map(v => v * v), m) }); }
const RUNS = {};
const runsC = k => RUNS[k] || (RUNS[k] = runsTest(ser(k).x));
function atBlock() {
  const m = +$('at-m').value, kk = $('at-s').value, nw = single('at-chart', 360); label('at-m').textContent = String(m);
  const crit = chi2ppf(0.95, m), P = NAMES.map(n => pmC(n, m));
  plot('at-chart', [{ type: 'bar', x: NAMES.map(n => SH[n]), y: P.map(p => p.r.lb), marker: { color: BLUE }, name: `Ljung-Box Q(${m}) (tests RW1)`, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { type: 'bar', x: NAMES.map(n => SH[n]), y: P.map(p => p.r.rob), marker: { color: TEAL }, name: `robust Q~(${m}) (tests the martingale)`, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { x: NAMES.map(n => SH[n]), y: NAMES.map(() => crit), mode: 'lines', line: { color: RED, dash: 'dash', width: 1.4 }, name: `5% critical value chi2(${m}) = ${crit.toFixed(2)}`, hoverinfo: 'skip' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('Statistic (log scale)', LOGT), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  table('at-tab', ['', 'T', 'ρ̂(1)', `Q<sub>LB</sub>(${m})`, 'p', `Q̃(${m})`, 'p', `Q<sub>LB</sub>(${m}), r²`, 'runs z', 'p'], NAMES.map((n, i) => { const p = P[i], R = runsC(n);
    return Object.assign([SH[n], fint(p.r.T), nf(p.r.rho1, 3), nf(p.r.lb, 1), { v: fp(p.r.p_lb), c: p.r.p_lb < 0.05 ? 'bad' : '' }, nf(p.r.rob, 1), { v: fp(p.r.p_rob), c: p.r.p_rob < 0.05 ? 'bad' : '' }, fint(p.s.lb), nf(R.z, 2), { v: fp(R.p), c: R.p < 0.05 ? 'bad' : '' }], { cls: n === kk ? 'sel' : '' }); }));
  const p = pmC(kk, m).r;
  steps('at-we', [T.at1(SH[kk], fint(p.T), m, p.rho.map(v => nf(v, 3)).join('; ')), T.at2(fint(p.T), nf(p.rho.reduce((s, v) => s + v * v, 0), 5), nf(p.bp, 2)),
    T.at3(nf(p.lb, 2), nf(p.lb / p.bp, 4)), T.at4(p.vr.slice(0, Math.min(m, 3)).map(v => nf(v * p.T, 2)).join('; '), nf(p.rob, 2)),
    T.at5(m, nf(crit, 2), nf(p.lb, 2), fpe(p.p_lb), nf(p.rob, 2), fpe(p.p_rob))]);
  out('at-out', [T.at(m, nf(crit, 2), P.filter(q => q.r.p_lb < 0.05).length, NAMES.filter((n, i) => P[i].r.p_rob < 0.05).map(n => SH[n]))]);
}

// ================================================================= 10. the runs test (table block)
function ruBlock() {
  const y = +$('ru-y').value, kk = $('ru-s').value; label('ru-y').textContent = String(y);
  const R = NAMES.map(n => { const S = fromYear(n, y); return { n, S, r: runsTest(S.x), rho: acf(S.x, 1)[0] }; });
  const Pn = panels('ru-chart', 2, { h: 360, hs: 270, extra: 120 }), [a, b] = Pn.ax, xs = NAMES.map(n => SH[n]), cl = NAMES.map(n => COL[n]);
  const tr = [on2(a, { type: 'bar', x: xs, y: R.map(o => o.r.z), marker: { color: cl }, showlegend: false, hovertemplate: '%{x}: z = %{y:.2f}<extra></extra>' }),
    on2(a, { x: [xs[0], xs[7], null, xs[0], xs[7]], y: [1.96, 1.96, null, -1.96, -1.96], mode: 'lines', line: { color: DARK, dash: 'dash', width: 1 }, name: '+/- 1.96', hoverinfo: 'skip' }),
    on2(b, { type: 'bar', x: xs, y: R.map(o => o.rho), marker: { color: cl }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })];
  setAx(Pn, 0, AX(''), AX('runs z')); setAx(Pn, 1, AX(''), AX('rho_hat(1)'));
  plot('ru-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, `Runs test z (too few runs: z < 0), from ${y}`), ptitle(b, 'First autocorrelation, same sample')], legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.2) })));
  table('ru-tab', ['', T.h.from, 'n₁', 'n₂', 'R', 'E[R]', 'sd(R)', 'z', 'p'], R.map(o => Object.assign([SH[o.n], o.S.dates[0], fint(o.r.n1), fint(o.r.n2), fint(o.r.runs), fnum(o.r.mean, 1), nf(o.r.sd, 1), nf(o.r.z, 2), { v: fp(o.r.p), c: o.r.p < 0.05 ? 'bad' : '' }], { cls: o.n === kk ? 'sel' : '' })));
  const o = R.find(q => q.n === kk).r;
  steps('ru-we', [T.ru1(SH[kk], fint(o.n1), fint(o.n2), fint(o.n)), T.ru2(fint(o.n1), fint(o.n2), fint(o.n), fnum(o.mean, 1)), T.ru3(fnum(o.var, 1), nf(o.sd, 2)),
    T.ru4(fint(o.runs), fnum(o.mean, 1), nf(o.sd, 2), nf(o.z, 2), fpe(o.p), o.z)]);
  out('ru-out', [T.ru(y, R.filter(q => q.r.p < 0.05).map(q => `${SH[q.n]} (${nf(q.r.z, 2)})`))]);
}

// ================================================================= 11. spurious regression
const SP_T = [100, 250, 500, 1000, 2000];
function hist(v, lo, hi, w) { const nb = Math.round((hi - lo) / w), c = new Array(nb).fill(0); let tot = 0; for (const x of v) { if (x < lo || x > hi) continue; c[Math.min(nb - 1, Math.floor((x - lo) / w))]++; tot++; } return { x: c.map((_, i) => lo + (i + 0.5) * w), y: c.map(n => n / (tot * w)) }; }
function spBlock() {
  const i = +$('sp-T').value, Tn = SP_T[i], S = D.spur[String(Tn)]; label('sp-T').textContent = fint(Tn);
  const Pn = panels('sp-chart', 2, { h: 380, hs: 280, extra: 130 }), [a, b] = Pn.ax, hl = hist(S.level.t, -40, 40, 1), hd = hist(S.diff.t, -40, 40, 1), g = range(-40, 40, 0.2);
  const ts = range(1, Tn, 1);
  const tr = [on2(a, { x: ts, y: S.y, mode: 'lines', line: { color: BLUE, width: 1.1 }, name: 'Random walk y(t)', hoverinfo: 'skip' }),
    on2(a, { x: ts, y: S.x, mode: 'lines', line: { color: RED, width: 1.1 }, name: 'Independent random walk x(t)', hoverinfo: 'skip' }),
    on2(b, { type: 'bar', x: hl.x, y: hl.y, width: 1, marker: { color: hexA(AMBER, 0.8) }, name: 't-statistic, regression in levels', hovertemplate: 't = %{x}<br>density %{y:.3f}<extra></extra>' }),
    on2(b, { type: 'bar', x: hd.x, y: hd.y, width: 1, marker: { color: hexA(GREEN, 0.7) }, name: 't-statistic, regression in increments', hovertemplate: 't = %{x}<br>density %{y:.3f}<extra></extra>' }),
    on2(b, { x: g, y: g.map(phi), mode: 'lines', line: { color: '#000', width: 1 }, name: 'N(0, 1)', hoverinfo: 'skip' })];
  setAx(Pn, 0, AX('t'), AX('')); setAx(Pn, 1, AX('t-statistic of the slope', { range: [-40, 40] }), AX('Density'));
  plot('sp-chart', tr, layout(Object.assign(Pn.L, { barmode: 'overlay', annotations: [ptitle(a, `First of the 2,000 simulated pairs, T = ${Tn}`), ptitle(b, 'Distribution of the t-statistic, 2,000 regressions')], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  out('sp-out', [T.sp(fint(Tn), pc(S.level.reject, 1), nf(S.level.r2_med, 2), pc(S.level.beyond40, 1), pc(S.diff.reject, 1), Tn !== 500)]);
}

// ================================================================= 12. the augmented test and a worked example (table block)
const DF_L = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40];
const DFC = {};
function dfCurve(k, reg) { const key = k + reg; if (DFC[key]) return DFC[key]; const y = lpSer(k).p; return (DFC[key] = DF_L.map(L => adf(y, reg, L).tau)); }
function dfBlock() {
  const k = $('df-s').value, reg = $('df-c').value, L0 = +$('df-L').value, nw = single('df-chart', 360); label('df-L').textContent = String(L0);
  const cur = dfCurve(k, reg), crit = reg === 'ct' ? NUM.df_crit.ct : NUM.df_crit.c, R = adf(lpSer(k).p, reg, L0);
  plot('df-chart', [{ x: DF_L, y: cur, mode: 'lines+markers', line: { color: COL[k], width: 1.6 }, marker: { size: 6 }, name: `${NAME[k]}: tau of the log price, ${reg === 'ct' ? 'constant and trend' : 'constant'}`, hovertemplate: 'L = %{x}<br>tau = %{y:.2f}<extra></extra>' },
    { x: [0, 40], y: [crit, crit], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.3 }, name: `Dickey-Fuller 5% critical value ${crit.toFixed(2)}`, hoverinfo: 'skip' },
    { x: [0, 40], y: [-1.645, -1.645], mode: 'lines', line: { color: DARK, dash: 'dot', width: 1.1 }, name: 'Normal 5% critical value -1.645 (wrong here)', hoverinfo: 'skip' },
    { x: [L0], y: [R.tau], mode: 'markers', marker: { size: 13, symbol: 'star', color: ORANGE, line: { color: '#fff', width: 1 } }, name: `worked example: L = ${L0}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX('Number of lagged differences L'), yaxis: AX('tau = gamma_hat / SE', { range: [Math.min(-4, ...cur) - 0.3, Math.max(0.5, ...cur) + 0.3] }), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  const ex = v => sig(v, 3);
  steps('df-we', [T.df1(SH[k], reg, L0, fint(R.nobs)), T.df2(ex(R.gamma), ex(R.se)), T.df3(ex(R.gamma), ex(R.se), nf(R.tau, 2)),
    T.df4(nf(R.tau, 2), nf(crit, 2), R.tau < crit, R.tau < -1.645)]);
  const La = D.adf_lags[k].price;
  out('df-out', [T.df(SH[k], reg, nf(Math.min(...cur), 2), nf(Math.max(...cur), 2), cur.filter(v => v < crit).length, DF_L.length, La)]);
}

// ================================================================= 13. the BET: log price and returns (other series selectable)
function ubBlock() {
  const k = $('ub-s').value, P = lpSer(k), S = ser(k), La = D.adf_lags[k];
  const ap = adfCached(k, 'price', La.price), ar = adfCached(k, 'ret', La.ret), kp = kpssC(k, 'price', null), kr = kpssC(k, 'ret', null);
  const Pn = panels('ub-chart', 2, { h: 360, hs: 270, extra: 120 }), [a, b] = Pn.ax;
  const tr = [on2(a, { x: P.dates, y: Array.from(P.p), mode: 'lines', line: { color: COL[k], width: 1.1 }, name: `${NAME[k]}: log price`, hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>' }),
    on2(b, { x: S.dates, y: S.x, mode: 'lines', line: { color: BLUE, width: 0.4 }, name: `${NAME[k]}: daily log return (%)`, hovertemplate: '%{x}<br>%{y:.2f}%<extra></extra>' })];
  setAx(Pn, 0, AX(''), AX('')); setAx(Pn, 1, AX(''), AX(''));
  plot('ub-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, `Log price: ADF ${ap.tau.toFixed(2)}, KPSS ${kp.stat.toFixed(2)}`), ptitle(b, `Returns: ADF ${ar.tau.toFixed(1)}, KPSS ${kr.stat.toFixed(2)}`)], legend: LEG(Pn.stack, Pn.stack ? -0.07 : -0.2) })));
  const U = NUM.ur[k];
  out('ub-out', [T.ub(SH[k], nf(ap.tau, 2), La.price, fp(U.price.adf.p), nf(kp.stat, 2), nf(ar.tau, 1), La.ret, nf(kr.stat, 2), [ap.tau < U.price.adf.crit5, kp.stat > 0.146], [ar.tau < U.ret.adf.crit5, kr.stat > 0.463])]);
}

// ================================================================= 14. unit-root tests on prices and returns (table)
function utBlock() {
  const lv = $('ut-l').value, l = lv === 'rule' ? null : +lv, Lv = $('ut-L').value, kk = $('ut-s').value, xs = NAMES.map(n => SH[n]);
  const kp = NAMES.map(n => kpssC(n, 'price', l)), kr = NAMES.map(n => kpssC(n, 'ret', l));
  const Pn = panels('ut-chart', 2, { h: 360, hs: 270, extra: 120 }), [a, b] = Pn.ax, cl = NAMES.map(n => COL[n]);
  const tr = [on2(a, { type: 'bar', x: xs, y: kp.map(v => v.stat), marker: { color: cl }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
    on2(a, { x: [xs[0], xs[7]], y: [0.146, 0.146], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.3 }, name: '5% critical values (0.146 with trend, 0.463 with constant)', hoverinfo: 'skip' }),
    on2(b, { type: 'bar', x: xs, y: kr.map(v => v.stat), marker: { color: cl }, showlegend: false, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
    on2(b, { x: [xs[0], xs[7]], y: [0.463, 0.463], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.3 }, showlegend: false, hoverinfo: 'skip' })];
  setAx(Pn, 0, AX(''), AX('KPSS', LOGT)); setAx(Pn, 1, AX(''), AX('KPSS', LOGT));
  const lt = l === null ? 'l = 12 (T/100)^(1/4)' : `l = ${l}`;
  plot('ut-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, `KPSS, log prices (trend), ${lt}`), ptitle(b, `KPSS, returns (constant), ${lt}`)], legend: LEG(Pn.stack, Pn.stack ? -0.06 : -0.2) })));
  const ad = (n, kind) => adfCached(n, kind, Lv === 'aic' ? D.adf_lags[n][kind] : +Lv);
  const cell = (v, crit, low) => ({ v: nf(v, 2), c: (low ? v < crit : v > crit) ? 'bad' : '' });
  table('ut-tab', [[{ v: '' }, { v: T.h.price, span: 3 }, { v: T.h.rets, span: 3 }], ['', 'ADF', 'PP', 'KPSS', 'ADF', 'PP', 'KPSS']], NAMES.map((n, i) => {
    const U = NUM.ur[n], pa = ad(n, 'price'), ra = ad(n, 'ret'), pp = ppC(n, 'price', l), rp = ppC(n, 'ret', l);
    return Object.assign([SH[n], cell(pa.tau, U.price.adf.crit5, true), cell(pp.stat, U.price.pp.crit5, true), cell(kp[i].stat, 0.146), cell(ra.tau, U.ret.adf.crit5, true), cell(rp.stat, U.ret.pp.crit5, true), cell(kr[i].stat, 0.463)], { cls: n === kk ? 'sel' : '' });
  }));
  const K = kpssC(kk, 'price', l);
  steps('ut-we', [T.ut1(SH[kk], fint(K.n), K.lags), T.ut2(sig(K.eta, 4), sig(K.g0, 4), sig(K.lam2, 4)), T.ut3(sig(K.eta, 4), sig(K.lam2, 4), nf(K.stat, 3), K.stat > 0.146)]);
  out('ut-out', [T.ut(l === null ? null : l, kp.filter(v => v.stat > 0.146).length, kr.filter(v => v.stat > 0.463).length, Lv)]);
}

// ================================================================= 15. worked example: VR from the ACF (table block)
function vaBlock() {
  const q = +$('va-q').value, k = $('va-s').value, x = ser(k).x, r = acf(x, Math.max(q - 1, 1)), nw = single('va-chart', 340); label('va-q').textContent = String(q);
  const ks = range(1, q - 1, 1), w = ks.map(j => 2 * (1 - j / q)), c = ks.map((j, i) => w[i] * r[i]), vacf = 1 + c.reduce((s, v) => s + v, 0), lm = varianceRatio(x, q).vr;
  plot('va-chart', [{ type: 'bar', x: ks, y: c, marker: { color: c.map(v => (v < 0 ? BLUE : RED)) }, name: 'term 2(1 - k/q) rho_hat(k)', hovertemplate: 'k = %{x}<br>%{y:.4f}<extra></extra>' },
    { x: ks, y: c.map((_, i) => c.slice(0, i + 1).reduce((s, v) => s + v, 0)), mode: 'lines+markers', line: { color: ORANGE, width: 1.5 }, marker: { size: 5 }, name: 'running sum = VR(q) - 1', hovertemplate: 'k = %{x}<br>%{y:.4f}<extra></extra>' }],
  layout({ xaxis: AX('Lag k', { range: [0.4, Math.max(q - 0.6, 1.6)], dtick: q > 12 ? 2 : 1 }), yaxis: AX(''), legend: LEG(nw, -0.18), margin: { l: 56, r: 12, t: 14, b: 40 } }));
  steps('va-we', [T.va1(SH[k], q, r.slice(0, q - 1).map(v => nf(v, 3)).join('; ')), T.va2(q, w.map(v => nf(v, 2)).join('; ')),
    T.va3(q, nf(vacf, 3), nf(lm, 3), nf(lm - vacf, 4))]);
  out('va-out', [T.va(SH[k], q, nf(vacf, 3), nf(lm, 3))]);
}

// ================================================================= 16. Lo and MacKinlay: estimator and tests (table block)
const VRC = {};
const vrC = (k, q, y) => { const key = k + q + ':' + (y || 0); return VRC[key] || (VRC[key] = varianceRatio(y ? fromYear(k, y).x : ser(k).x, q)); };
function lmBlock() {
  const q = +$('lm-q').value, kk = $('lm-s').value, nw = single('lm-chart', 360), xs = NAMES.map(n => SH[n]); label('lm-q').textContent = String(q);
  const V = NAMES.map(n => vrC(n, q));
  plot('lm-chart', [{ type: 'bar', x: xs, y: V.map(v => v.z), marker: { color: hexA(BLUE, 0.85) }, name: `Z(${q}), homoskedastic (RW1)`, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { type: 'bar', x: xs, y: V.map(v => v.zs), marker: { color: TEAL }, name: `Z*(${q}), robust (martingale)`, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: [xs[0], xs[7], null, xs[0], xs[7]], y: [1.96, 1.96, null, -1.96, -1.96], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.2 }, name: '+/- 1.96', hoverinfo: 'skip' }],
  layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('z statistic'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  const v = vrC(kk, q);
  steps('lm-we', [T.lm1(SH[kk], fint(v.T), nf(v.mu, 4), nf(v.varA, 4)), T.lm2(q, fnum(v.m, 1), nf(v.varC, 4), nf(v.vr, 4)),
    T.lm3(q, fint(v.T), nf(v.seIid, 5), nf(v.z, 2)), T.lm4(v.delta.slice(0, 4).map(d => nf(d, 2)).join('; ') + (q > 5 ? '; …' : ''), nf(v.theta, 3), nf(v.seRob, 5), nf(v.zs, 2)),
    T.lm5(nf(v.seRob / v.seIid, 2), nf(v.z / v.zs, 2))]);
  out('lm-out', [T.lm(q, V.filter(u => Math.abs(u.z) > 1.96).length, NAMES.filter((n, i) => Math.abs(V[i].zs) > 1.96).map(n => SH[n]))]);
}

// ================================================================= 17. why the robust test: a Monte Carlo check
const VS_A = ['0.00', '0.03', '0.06', '0.09', '0.12'];
function vsBlock() {
  const a = VS_A[+$('vs-a').value], S = D.size[a], qs = ['2', '5', '10'], nw = single('vs-chart', 360); label('vs-a').textContent = nf(+a, 2);
  const L = [['i.i.d. returns, Z', 'iid', 'z', BLUE], ['i.i.d. returns, Z*', 'iid', 'zs', TEAL], [`GARCH returns (alpha = ${a}), Z`, 'garch', 'z', RED], [`GARCH returns (alpha = ${a}), Z*`, 'garch', 'zs', AMBER]];
  const tr = L.map(([nm, g, z, c]) => ({ type: 'bar', x: qs.map(q => `q = ${q}`), y: qs.map(q => 100 * S[g][q][z]), marker: { color: c }, name: nm, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }));
  tr.push({ x: qs.map(q => `q = ${q}`), y: [5, 5, 5], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: 'Nominal level: 5%', hoverinfo: 'skip' });
  plot('vs-chart', tr, layout({ barmode: 'group', xaxis: AX(''), yaxis: AX('Rejections of a true null (%)', { range: [0, 20] }), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  out('vs-out', [T.vs(nf(+a, 2), nf(+a + 0.86, 2), pc(S.garch['2'].z, 1), pc(S.garch['2'].zs, 1), pc(S.iid['2'].z, 1), a !== '0.12')]);
}

// ================================================================= 18. variance ratios for horizons of 2 to 40 days
const VP = {};
function vpProfile(k, y) { const key = k + y; if (VP[key]) return VP[key]; const x = fromYear(k, y).x, qs = range(2, 40, 1), v = qs.map(q => varianceRatio(x, q)); return (VP[key] = { qs, v, from: fromYear(k, y).dates[0] }); }
function vpBlock() {
  const y = +$('vp-y').value; label('vp-y').textContent = String(y);
  const G = gridp('vp-chart', 2, 2, { h: 540, hs: 250 }), tr = [], ann = [], items = [];
  ASSETS.forEach((n, i) => {
    const P = vpProfile(n, y), a = G.ax[i], vr = P.v.map(v => v.vr), hi = P.v.map(v => 1 + 1.96 * v.seRob), lo = P.v.map(v => 1 - 1.96 * v.seRob);
    tr.push(on2(a, { x: P.qs, y: hi, mode: 'lines', line: { width: 0, color: TEAL }, showlegend: false, hoverinfo: 'skip' }));
    tr.push(on2(a, { x: P.qs, y: lo, mode: 'lines', line: { width: 0, color: TEAL }, fill: 'tonexty', fillcolor: hexA(TEAL, 0.18), name: 'Robust 95% band (Z*)', showlegend: i === 0, legendgroup: 'rb', hoverinfo: 'skip' }));
    tr.push(on2(a, { x: P.qs.concat([null], P.qs), y: P.v.map(v => 1 + 1.96 * v.seIid).concat([null], P.v.map(v => 1 - 1.96 * v.seIid)), mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.8 }, name: 'i.i.d. 95% band (Z)', showlegend: i === 0, legendgroup: 'iid', hoverinfo: 'skip' }));
    tr.push(on2(a, { x: P.qs, y: vr, mode: 'lines+markers', line: { color: COL[n], width: 1.8 }, marker: { size: 4 }, name: 'VR(q)', showlegend: i === 0, legendgroup: 'vr', hovertemplate: 'q = %{x}<br>VR = %{y:.3f}<extra></extra>' }));
    setAx(G, i, AX(G.rows === 2 && i < 2 ? '' : 'Horizon q (days)'), AX(i % 2 === 0 || G.nw ? 'VR(q)' : ''));
    ann.push(ptitle(a, `${NAME[n]}, from ${P.from.slice(0, 7)}`));
    items.push([SH[n], nf(vr[vr.length - 1], 3), nf(Math.min(...vr), 3), nf(Math.max(...vr), 3), P.v.filter(v => Math.abs(v.zs) > 1.96).length]);
  });
  plot('vp-chart', tr, layout(Object.assign(G.L, { annotations: ann, legend: LEG(G.nw, G.nw ? -0.05 : -0.12), margin: { l: 50, r: 10, t: 24, b: 40 } })));
  out('vp-out', [T.vp(y, items)]);
}

// ================================================================= 19. variance ratios and robust tests (table)
function vtBlock() {
  const y = +$('vt-y').value, stat = $('vt-z').value, nw = narrow('vt-chart'), qs = D.qs; label('vt-y').textContent = String(y);
  const V = NAMES.map(n => qs.map(q => vrC(n, q, y))), zz = V.map(r => r.map(v => (stat === 'zs' ? v.zs : v.z)));
  $('vt-chart').style.height = (nw ? 420 : 360) + 'px';
  const zl = stat === 'zs' ? 'Z*' : 'Z';
  plot('vt-chart', [{ type: 'heatmap', z: zz, x: qs.map(q => `q = ${q}`), y: NAMES.map(n => SH[n]), zmin: -8, zmax: 8, zmid: 0,
    colorscale: [[0, BLUE], [0.38, '#C5D2E8'], [0.5, '#FFFFFF'], [0.62, '#F4C2C2'], [1, RED]], text: zz.map(r => r.map(v => v.toFixed(2))), texttemplate: '%{text}', hovertemplate: `%{y}, %{x}: ${zl} = %{z:.2f}<extra></extra>`,
    colorbar: { title: { text: zl, side: 'right' }, thickness: 10, len: 0.9 } }],
  layout({ xaxis: AX(''), yaxis: AX('', { autorange: 'reversed' }), margin: { l: 70, r: 10, t: 14, b: 30 } }));
  table('vt-tab', ['', ...qs.map(q => `VR(${q})`), ...qs.map(q => `${stat === 'zs' ? 'Z*' : 'Z'}(${q})`)], NAMES.map((n, i) => [SH[n], ...V[i].map(v => nf(v.vr, 3)), ...zz[i].map(v => ({ v: nf(v, 2), c: Math.abs(v) > 1.96 ? 'bad' : '' }))]));
  const nrej = zz.reduce((s, r) => s + r.filter(v => Math.abs(v) > 1.96).length, 0);
  out('vt-out', [T.vt(y, stat === 'zs', nrej, NAMES.length * qs.length, nf(0.05 * NAMES.length * qs.length, 1))]);
}

// ================================================================= 20. joint and automatic tests on real data (table)
const CD_SETS = { lec: [2, 5, 10, 20], pow2: [2, 4, 8, 16], short: [2, 3, 4, 5], long: [5, 10, 20, 40], all: range(2, 20, 1) };
const AVR = {};
const avrC = k => AVR[k] || (AVR[k] = autoVR(ser(k).x));
function cdBlock() {
  const set = CD_SETS[$('cd-h').value], al = +$('cd-a').value, k1 = $('cd-s').value, k2 = $('cd-s2').value, nw = single('cd-chart', 340), xs = NAMES.map(n => SH[n]);
  const R = NAMES.map(n => { const zs = set.map(q => vrC(n, q).zs), cd = Math.max(...zs.map(Math.abs)); return { zs, cd, p: 1 - (2 * Phi(cd) - 1) ** set.length }; }), c = cdCrit(set.length, al);
  plot('cd-chart', [{ type: 'bar', x: xs, y: R.map(r => r.cd), marker: { color: R.map(r => (r.cd > c ? RED : BLUE)) }, name: `CD = max |Z*(q)|, q in {${set.length > 6 ? '2, ..., 20' : set.join(', ')}}`, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: [xs[0], xs[7]], y: [c, c], mode: 'lines', line: { color: RED, dash: 'dash', width: 1.3 }, name: `critical value at ${Math.round(al * 100)}%, m = ${set.length}: ${c.toFixed(2)}`, hoverinfo: 'skip' },
    { x: [xs[0], xs[7]], y: [normPpf(1 - al / 2), normPpf(1 - al / 2)], mode: 'lines', line: { color: DARK, dash: 'dot', width: 1 }, name: `single-test value ${normPpf(1 - al / 2).toFixed(2)}`, hoverinfo: 'skip' }],
  layout({ xaxis: AX(''), yaxis: AX('Chow-Denning statistic'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 } }));
  table('cd-tab', ['', 'T', 'CD', 'p', 'k̂', T.h.avr, T.h.pboot], NAMES.map((n, i) => { const A = avrC(n), V = NUM.vr[n].avr;
    return [SH[n], fint(ser(n).n), nf(R[i].cd, 2), { v: fp(R[i].p), c: R[i].p < al ? 'bad' : '' }, nf(A.k, 1), nf(A.stat, 2), { v: fp(V.p_boot), c: V.p_boot < al ? 'bad' : '' }]; }));
  const r1 = R[NAMES.indexOf(k1)], m = set.length;
  steps('cd-we', [T.cd1(SH[k1], set.length > 6 ? '2, …, 20' : set.join(', '), r1.zs.map(v => nf(v, 2)).join('; '), nf(r1.cd, 2)),
    T.cd2(m, nf(al, 2), nf((1 - al) ** (1 / m), 5), nf(c, 3)), T.cd3(nf(r1.cd, 2), nf(2 * Phi(r1.cd) - 1, 5), m, fpe(r1.p), r1.cd > c)]);
  const A = avrC(k2);
  steps('av-we', [T.av1(SH[k2], nf(A.rho, 4), sig(A.a2, 3), fint(A.T), nf(A.k, 2)), T.av2(nf(A.vr, 4)), T.av3(fint(A.T), nf(A.k, 2), nf(A.vr, 4), nf(A.stat, 2), fpe(p2(A.stat)), fp(NUM.vr[k2].avr.p_boot))]);
  out('cd-out', [T.cd(nf(c, 2), Math.round(al * 100), m, NAMES.filter((n, i) => R[i].cd > c).map(n => SH[n]))]);
}

// ================================================================= 21. sixteen markets over the last ten years
const GCOL = { Developed: BLUE, Emerging: RED, Crypto: AMBER };
const grp = k => (D.developed.includes(k) ? 'Developed' : D.crypto.includes(k) ? 'Crypto' : 'Emerging');
function vmBlock() {
  const q = +$('vm-q').value, lev = +$('vm-c').value, zc = normPpf(1 - (1 - lev) / 2), nw = narrow('vm-chart'); label('vm-q').textContent = String(q);
  const V = D.markets.map(k => ({ k, v: varianceRatio(mkSer(k).x, q) })).sort((a, b) => a.v.vr - b.v.vr), names = V.map(o => NAME[o.k]);
  $('vm-chart').style.height = (nw ? 640 : 520) + 'px';
  const tr = [];
  Object.keys(GCOL).forEach(g => {
    const S = V.filter(o => grp(o.k) === g), s = segs(S.map(o => NAME[o.k]), S.map(o => o.v.vr - zc * o.v.seRob), S.map(o => o.v.vr + zc * o.v.seRob));
    tr.push({ x: s.x, y: s.y, mode: 'lines', line: { color: GCOL[g], width: 1.4 }, showlegend: false, hoverinfo: 'skip', legendgroup: g });
    tr.push({ x: S.map(o => o.v.vr), y: S.map(o => NAME[o.k]), mode: 'markers', marker: { size: 8, color: GCOL[g] }, name: `${g} markets`, legendgroup: g, customdata: S.map(o => o.v.zs), hovertemplate: `%{y}<br>VR(${q}) = %{x:.3f}<br>Z* = %{customdata:.2f}<extra></extra>` });
  });
  tr.push({ x: [1, 1], y: [names[0], names[names.length - 1]], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: 'VR = 1 (random walk)', hoverinfo: 'skip' });
  plot('vm-chart', tr, layout({ xaxis: AX(`VR(${q}) with robust ${Math.round(lev * 100)}% interval`), yaxis: AX('', { type: 'category', categoryorder: 'array', categoryarray: names }), legend: LEG(nw, -0.1), margin: { l: 120, r: 12, t: 14, b: 40 } }));
  const ex = V.filter(o => Math.abs(o.v.zs) > zc);
  out('vm-out', [T.vm(q, Math.round(lev * 100), ex.map(o => `${NAME[o.k]} (Z* = ${nf(o.v.zs, 2)})`), D.markets.length, nf((1 - lev) * D.markets.length, 1))]);
}

// ================================================================= 22. rolling variance ratios
const RVC = {};
const rvC = (k, q, w) => { const key = `${k}${q}:${w}`; return RVC[key] || (RVC[key] = rollingVR(k, q, w, D.step)); };
function rvBlock() {
  const w = +$('rv-w').value, q = +$('rv-q').value, Pn = panels('rv-chart', 2, { stack: true, hs: 230, extra: 110, vgap: 0.12 }), [a, b] = Pn.ax, tr = [], items = [];
  ['sp500', 'bet', 'btc'].forEach(k => {
    const R = rvC(k, q, w), d = R.map(r => r.date);
    tr.push(on2(a, { x: d, y: R.map(r => r.vr), mode: 'lines', line: { color: COL[k], width: 1.3 }, name: NAME[k], legendgroup: k, hovertemplate: `${NAME[k]}<br>%{x}<br>VR = %{y:.3f}<extra></extra>` }));
    tr.push(on2(b, { x: d, y: R.map(r => r.zs), mode: 'lines', line: { color: COL[k], width: 1.1 }, showlegend: false, legendgroup: k, hovertemplate: `${NAME[k]}<br>%{x}<br>Z* = %{y:.2f}<extra></extra>` }));
    const n = R.length, pos = R.filter(r => r.zs > 1.96).length, neg = R.filter(r => r.zs < -1.96).length;
    items.push([SH[k], fint(n), pc((pos + neg) / n, 0), pos, neg, nf(Math.min(...R.map(r => r.vr)), 2), nf(Math.max(...R.map(r => r.vr)), 2)]);
  });
  const x0 = '1997-01-01', x1 = D.end;
  tr.push(on2(a, { x: [x0, x1], y: [1, 1], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.7 }, showlegend: false, hoverinfo: 'skip' }));
  tr.push(on2(b, { x: [x0, x1, null, x0, x1], y: [1.96, 1.96, null, -1.96, -1.96], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.7 }, name: '+/- 1.96', hoverinfo: 'skip' }));
  setAx(Pn, 0, AX('', { range: [x0, x1] }), AX(`VR(${q})`)); setAx(Pn, 1, AX('', { range: [x0, x1], matches: 'x' }), AX(`Robust Z*(${q})`));
  plot('rv-chart', tr, layout(Object.assign(Pn.L, { annotations: [ptitle(a, `Windows of ${w} trading days, one every ${D.step} days`)], legend: LEG(true, -0.08) })));
  out('rv-out', [T.rv(fint(w), q, fint(w - D.step), items)]);
}

// ================================================================= 23. three sub-periods per market (table)
const SBC = {};
function sbBlock() {
  const q = +$('sb-q').value, S = SBC[q] || (SBC[q] = subperiods(q)), nw = narrow('sb-chart'), rows = [], yl = [];
  Object.entries(S).forEach(([k, per]) => per.forEach(p => { rows.push({ k, ...p }); yl.push(`${SH[k]} ${p.from.slice(0, 4)}-${p.to.slice(0, 4)}`); }));
  $('sb-chart').style.height = (nw ? 470 : 400) + 'px';
  const tr = [];
  ['bet', 'sp500', 'btc'].forEach(k => {
    const R = rows.map((r, i) => [r, yl[i]]).filter(([r]) => r.k === k), s = segs(R.map(([, l]) => l), R.map(([r]) => r.vr - 1.96 * r.se), R.map(([r]) => r.vr + 1.96 * r.se));
    tr.push({ x: s.x, y: s.y, mode: 'lines', line: { color: COL[k], width: 1.4 }, showlegend: false, legendgroup: k, hoverinfo: 'skip' });
    tr.push({ x: R.map(([r]) => r.vr), y: R.map(([, l]) => l), mode: 'markers', marker: { size: 9, color: COL[k] }, name: NAME[k], legendgroup: k, customdata: R.map(([r]) => r.zs), hovertemplate: `%{y}<br>VR(${q}) = %{x:.3f}<br>Z* = %{customdata:.2f}<extra></extra>` });
  });
  tr.push({ x: [1, 1], y: [yl[0], yl[yl.length - 1]], mode: 'lines', line: { color: '#000', dash: 'dash', width: 0.9 }, name: 'VR = 1', hoverinfo: 'skip' });
  plot('sb-chart', tr, layout({ xaxis: AX(`VR(${q}) with robust 95% interval`), yaxis: AX('', { type: 'category', categoryorder: 'array', categoryarray: yl.slice().reverse() }), legend: LEG(nw, -0.2), margin: { l: 130, r: 12, t: 14, b: 40 } }));
  table('sb-tab', ['', T.h.period, 'T', 'ρ̂(1)', `VR(${q})`, `Z*(${q})`, 'CD p'], rows.map(r => [SH[r.k], `${r.from.slice(0, 4)}–${r.to.slice(0, 4)}`, fint(r.n), nf(r.rho1, 3), nf(r.vr, 2), { v: nf(r.zs, 2), c: Math.abs(r.zs) > 1.96 ? 'bad' : '' }, { v: fp(r.cd_p), c: r.cd_p < 0.05 ? 'bad' : '' }]));
  out('sb-out', [T.sb(q, rows.filter(r => Math.abs(r.zs) > 1.96).map(r => `${SH[r.k]} ${r.from.slice(0, 4)}–${r.to.slice(0, 4)} (${nf(r.zs, 2)})`))]);
}

// ================================================================= 24. day-of-week returns
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const CAL = {};
const calC = (k, y, L, v) => { const key = `${k}${y}:${L}:${v}`; if (CAL[key]) return CAL[key]; const S = fromYear(k, y); return (CAL[key] = calendar(S.x, S.dates, L, v)); };
function caBlock() {
  const view = $('ca-v').value, L = +$('ca-l').value, y = +$('ca-y').value, kk = $('ca-s').value, nw = single('ca-chart', 380); label('ca-y').textContent = String(y);
  const labs = view === 'month' ? MONTHS : DAYS, G = labs.length, w = 0.35, tr = [], ps = {};
  ['sp500', 'bet'].forEach((k, i) => {
    const c = calC(k, y, L, view), xs = labs.map((_, j) => j + (i - 0.5) * w); ps[k] = c;
    tr.push({ type: 'bar', x: xs, y: c.means, width: w * 0.92, marker: { color: hexA(COL[k], 0.85) }, name: `${NAME[k]}: mean return (Wald p = ${c.wald_p.toFixed(2)})`, hovertemplate: `${NAME[k]}<br>%{y:.3f}%<extra></extra>` });
    const s = vsegs(xs, c.means.map((m, j) => m - 1.96 * c.ses[j]), c.means.map((m, j) => m + 1.96 * c.ses[j]));
    tr.push({ x: s.x, y: s.y, mode: 'lines', line: { color: '#000', width: 1 }, name: i === 0 ? `HAC 95% interval (${L} lags)` : '', showlegend: i === 0, hoverinfo: 'skip' });
  });
  plot('ca-chart', tr, layout({ xaxis: AX('', { tickvals: labs.map((_, j) => j), ticktext: labs }), yaxis: AX('Mean daily log return (%)'), legend: LEG(nw, -0.14), margin: { l: 56, r: 12, t: 14, b: 30 },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0, y1: 0, line: { color: '#000', width: 0.6 } }] }));
  const c = calC(kk, y, L, 'day');
  steps('ca-we', [T.ca1(SH[kk], y, fint(c.nj), nf(c.jan_mean, 3), fint(c.no), nf(c.other_mean, 3)), T.ca2(nf(c.jan_b, 3), L, nf(c.jan_se, 3)), T.ca3(nf(c.jan_b, 3), nf(c.jan_se, 3), nf(c.jan_t, 2), fpe(c.jan_p))]);
  out('ca-out', [T.ca(view === 'month', y, L, nf(ps.sp500.wald_p, 2), nf(ps.bet.wald_p, 2), ps.sp500.df)]);
}

// ================================================================= wiring
const PFX = { 'rw-paths': 'rw', 'scatter-lag': 'sc', 'sqrt-time': 'st', 'white-noise': 'wn', 'acf-processes': 'ap', 'acf-returns': 'ar', 'acf-abs': 'aa', 'lb-example': 'lb', 'ac-tests': 'at', runs: 'ru',
  spurious: 'sp', 'adf-example': 'df', 'unit-root-bet': 'ub', 'ur-table': 'ut', 'vr-acf': 'va', 'lm-tests': 'lm', 'vr-size': 'vs', 'vr-profile': 'vp', 'vr-table': 'vt', 'cd-table': 'cd',
  'vr-markets': 'vm', 'rolling-vr': 'rv', 'sub-table': 'sb', calendar: 'ca' };
const BLOCKS = { rwBlock, scBlock, stBlock, wnBlock, apBlock, arBlock, aaBlock, lbBlock, atBlock, ruBlock, spBlock, dfBlock, ubBlock, utBlock, vaBlock, lmBlock, vsBlock, vpBlock, vtBlock, cdBlock, vmBlock, rvBlock, sbBlock, caBlock };
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
  on(['rw-y'], rwBlock); PL('rw-play', 'rw-y', range(1991, 2026, 1), 350);
  on(['sc-k'], scBlock); PL('sc-play', 'sc-k', range(1, 10, 1), 900);
  on(['st-s', 'st-q'], stBlock); PL('st-play', 'st-s', NAMES, 1400);
  on(['wn-n'], wnBlock); PL('wn-play', 'wn-n', range(100, 1000, 100), 800);
  on(['ap-n'], apBlock); PL('ap-play', 'ap-n', range(50, 1000, 50), 450);
  on(['ar-y'], arBlock); on(['ar-s', 'ar-k'], arWe); PL('ar-play', 'ar-y', range(1990, 2022, 2), 700);
  on(['aa-d'], aaBlock); PL('aa-play', 'aa-d', range(0.25, 2, 0.25), 800);
  on(['lb-T', 'lb-m'], lbBlock); PL('lb-play', 'lb-T', range(500, 5000, 500), 700);
  on(['at-m', 'at-s'], atBlock); PL('at-play', 'at-m', range(2, 30, 2), 700);
  on(['ru-y', 'ru-s'], ruBlock); PL('ru-play', 'ru-y', range(1990, 2022, 2), 700);
  on(['sp-T'], spBlock); PL('sp-play', 'sp-T', range(0, 4, 1), 1400);
  on(['df-s', 'df-c', 'df-L'], dfBlock); PL('df-play', 'df-s', NAMES, 1500);
  on(['ub-s'], ubBlock); PL('ub-play', 'ub-s', NAMES, 1500);
  on(['ut-l', 'ut-L', 'ut-s'], utBlock); PL('ut-play', 'ut-l', ['0', '5', '10', '20', '40', '80', 'rule'], 1300);
  on(['va-q', 'va-s'], vaBlock); PL('va-play', 'va-q', range(2, 20, 1), 600);
  on(['lm-q', 'lm-s'], lmBlock); PL('lm-play', 'lm-q', range(2, 40, 2), 600);
  on(['vs-a'], vsBlock); PL('vs-play', 'vs-a', range(0, 4, 1), 1300);
  on(['vp-y'], vpBlock); PL('vp-play', 'vp-y', range(1990, 2022, 2), 800);
  on(['vt-y', 'vt-z'], vtBlock); PL('vt-play', 'vt-y', range(1990, 2022, 2), 700);
  on(['cd-h', 'cd-a', 'cd-s', 'cd-s2'], cdBlock); PL('cd-play', 'cd-h', ['pow2', 'short', 'long', 'all', 'lec'], 1400);
  on(['vm-q', 'vm-c'], vmBlock); PL('vm-play', 'vm-q', range(2, 20, 1), 600);
  on(['rv-w', 'rv-q'], rvBlock); PL('rv-play', 'rv-w', ['250', '750', '1000', '1500', '500'], 1500);
  on(['sb-q'], sbBlock); PL('sb-play', 'sb-q', ['2', '10', '20', '5'], 1500);
  on(['ca-v', 'ca-l', 'ca-y', 'ca-s'], caBlock); PL('ca-play', 'ca-y', range(1990, 2022, 2), 700);
  const VEIL = T.veil || {};
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if ($(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: VEIL[b] || { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  Object.values(BLOCKS).forEach(f => { try { f(); } catch (e) { console.error(e); throw e; } });
}
window.CH7 = { ser, lpSer, mkSer, fromYear, acf, robVar, portmanteau, runsTest, varianceRatio, chowDenning, cdCrit, autoVR, adf, adfCached, ppTest, kpssTest, kpssC, ppC, calendar, armaAcf,
  rollingVR, subperiods, realSd, sdev, lagStats, chi2sf, chi2ppf, normPpf, Phi, init, PFX, blocks: BLOCKS };
if (window.CH7_NO_INIT) return;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
