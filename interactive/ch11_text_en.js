// Readouts of the Chapter 11 interactive page, written by latex/build_interactive_ch11.py.
(function () {
const num = s => +String(s).replace(/[\s,  ]/g, '').replace('−', '-');
const days = s => (num(s) === 1 ? '1 day' : `${s} days`);
const X = ' (extra run of the code of the lecture)';
const lst = a => (a.length ? a.join(', ') : 'none');
const yl = y => (y === 'abs' ? 'absolute returns |r|' : 'returns r');
const sub = i => String(i).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
const neg = s => String(s).replace('−', '');
window.CH11_TEXT = {
 h: { r: 'returns rₜ', abs: 'absolute returns |rₜ|', band: '95% band, i.i.d.',
      st: ['', 'R/S', 'DFA', 'GPH d̂ (SE)', 'd̂ ± 1.96 SE', 'Lo V_N(q)', 'q'],
      lw: [['', { v: 'returns r', span: 4 }, { v: 'absolute returns |r|', span: 4 }], ['', 'V(0)', 'V(q)', 'q', 'short memory', 'V(0)', 'V(q)', 'q', 'short memory']],
      mt: [['', '', { v: 'returns rₜ', span: 4 }, { v: 'absolute returns |rₜ|', span: 3 }, ''], ['', 'N', 'R/S', 'DFA', 'GPH d̂', 'd̂ ± 1.96 SE', 'DFA', 'GPH d̂', 'Lo V', 'DFA band']] },
 sc: (hm, s10, shm, items, x) => `Standard deviation of non-overlapping h-day returns divided by the daily one, h from 1 to ${hm}${x ? X : ''}. `
   + items.map(([n, H, r10, rh]) => `${n}: slope ${H}, ratio at 10 days ${r10}` + (+hm > 10 ? `, at ${days(hm)} ${rh}` : '')).join('; ')
   + `. The square-root-of-time rule: √10 = ${s10}` + (+hm > 10 ? `, √${hm} = ${shm}.` : '.'),
 rxlec: 'the eight returns of the lecture',
 rxsrc: (d0, d1) => `the S&P 500 returns from ${d0} to ${d1} (8 trading days)`,
 rx1: (src, xs, s, n, m) => `${src[0].toUpperCase() + src.slice(1)}, in %: ${xs}; sum ${s}, mean x̄ = ${s}/${n} = ${m}.`,
 rx2: devs => `Deviations from the mean, x_k − x̄: ${devs}.`,
 rx3: (Y, yn) => `Cumulative deviations Y_k: ${Y}; Y₈ = ${yn}, zero by construction.`,
 rx4: (mx, kx, mn, kn, R, ss, n, S) => `R = max Y_k − min Y_k = ${mx} (k = ${kx}) − (${mn}) (k = ${kn}) = ${R}; S = √(Σ(x_k − x̄)²/n) = √(${ss}/${n}) = ${S}.`,
 rx5: (R, S, rs) => `(R/S)₈ = ${R}/${S} = ${rs}. One block gives a single point of the log-log plot; H needs many block sizes.`,
 rx: (src, R, S, rs, x) => `${src[0].toUpperCase() + src.slice(1)}${x ? X : ''}: R = ${R}, S = ${S}, (R/S)₈ = ${rs}.`,
 fb: (items, x) => `The same random numbers (seed 2026) for all paths${x ? X : ''}. ` + items.map(([H, a, t, r]) => `H = ${H}: first-order autocorrelation of the increments `
   + `${a} (theoretical 2^(2H − 1) − 1 = ${t}), range of the path ${r}`).join('; ') + '.',
 fg: (H, r1, Hb, r1b, r10, r50, r100, a10, a100, s1, gap, x) => `H = ${H}${x ? X : ''}: ρ(1) = 2^(2H − 1) − 1 = ${r1}; for H = ${Hb}, ρ(1) = ${r1b}. fGn with H = ${H}: ρ(10) = ${r10}, `
   + `ρ(50) = ${r50}, ρ(100) = ${r100}; AR(1) with φ = ${r1}: ρ(10) = ${a10}, ρ(100) = ${a100}. Sample ACF (n = 5000) at lag 1: ${s1}; on average over lags 1–50 it is `
   + (num(gap) >= 0 ? `${gap} below the theoretical one.` : `${neg(gap)} above the theoretical one.`),
 af: (items, x) => `Paths of 1000 values, sample ACF from 20000 values${x ? X : ''}. ` + items.map(([d, r1, r10, r50, s1, sd, st]) => `d = ${d}: ρ(1) = d/(1 − d) = ${r1}, `
   + `ρ(10) = ${r10}, ρ(50) = ${r50} (sample ACF at lag 1: ${s1}); standard deviation of the path ${sd}, theoretical ${st}`).join('; ') + '.',
 rd: (s, n, K, n1, nK, rs, rsi, dfa, dfai, rsa, dfaa, x) => `${s}, ${n} returns, ${K} block sizes from ${n1} to ${nK} days${x ? X : ''}. Returns: R/S slope ${rs} `
   + `(i.i.d. series of the same length: ${rsi}), DFA ${dfa} (i.i.d.: ${dfai}). Absolute returns: R/S ${rsa}, DFA ${dfaa}.`,
 gp: (s, p, m, d, se, lo, hi, da, Ha, x) => `${s}, m = ⌊T^${p}⌋ = ${m} frequencies${x ? X : ''}: returns d̂ = ${d} (SE ${se}; d̂ ± 1.96 SE = [${lo}, ${hi}], `
   + (num(lo) < 0 && num(hi) > 0 ? 'contains 0' : 'does not contain 0') + `); absolute returns d̂ = ${da}, Ĥ = d̂ + 0.5 = ${Ha}.`,
 lo: (n, c5, m5, q5, c7, m7, c0, m0, m8, x) => `500 series of ${n} values${x ? X : ''}. AR(1) with φ = 0.5 (short memory, H₀ true): the classical test rejects in ${c5} of the cases, `
   + `the modified test in ${m5} (mean q ${q5}); with φ = 0: ${c0} and ${m0}. fGn with H = 0.7: ${c7} and ${m7}; with H = 0.8 the modified test rejects in ${m8}.`,
 mc: (lv, r250, r1000, r5000, d1000, g1000, wr, wd, wg, wd250, lo, hi, x) => `500 i.i.d. Normal series for each N, ${lv}% band${x ? X : ''}. R/S: mean ${r250} at N = 250, `
   + `${r1000} at N = 1000, ${r5000} at N = 5000; DFA ${d1000} and GPH ${g1000} at N = 1000. Width of the band at N = 1000: R/S ${wr} (from ${lo} to ${hi}), DFA ${wd}, GPH ${wg}; `
   + `at N = 250, DFA ${wd250}.`,
 nl: (b, mp, mq, dr, H1, d1, H2, d2, lo, hi, nm, x) => `Break after ${b}${x ? X : ''}: mean flow ${mp} before, ${mq} after (in 10⁸ m³), `
   + (num(dr) >= 0 ? `a fall of ${dr}%` : `a rise of ${neg(dr)}%`) + `. Raw series: Ĥ_R/S = ${H1}, DFA ${d1}; after removing the two regime means: ${H2} and ${d2}. `
   + `The i.i.d. band for R/S with N = 100: [${lo}, ${hi}], centred at ${nm}.`,
 su: (n, g5, d5, r5, r0, a95, a99, rlo, rhi, x) => `300 series of ${n} values${x ? X : ''}. A mean shift of 0.5 standard deviations in the middle: GPH Ĥ = ${g5}, DFA ${d5}, `
   + `R/S ${r5} (without a shift R/S already gives ${r0}: the small-sample bias). GARCH(1,1): DFA of |r| ${a95} at α + β = 0.95 and ${a99} at 0.99; for the returns, DFA ` + (rlo === rhi ? `${rlo} at every persistence.` : `between ${rlo} and ${rhi}.`),
 va: (s, L, band, a1, a50, a100, np, s1, s50, r1, mx, no, ex, top, x) => `${s}, the first ${L} lags${x ? X : ''}; 95% band: ±${band}. |r|: ρ(1) = ${a1}, ρ(50) = ${a50}, `
   + `ρ(100) = ${a100}; ${np} of the ${L} lags are above the band. r²: ρ(1) = ${s1}, ρ(50) = ${s50}. r: ρ(1) = ${r1}, then |ρ(k)| ≤ ${mx}; ${no} of the lags 2–${L} `
   + `are outside the band (about ${ex} expected by chance), the largest at lags ${top}.`,
 mk: (e, amin, amax, smin, smax, ins, above, x) => `The ${e} estimator${x ? X : ''}: for |r|, between ${amin} and ${amax}; after a random shuffle, between ${smin} and ${smax}, `
   + `${ins} of 6 inside the bands: the memory lies in the order of the days. Returns above the band: ${lst(above)}.`,
 ro: (W, lo, hi, items, x) => `Windows of ${W} observations moved by 21${x ? X : ''}; 95% i.i.d. band: [${lo}, ${hi}]. ` + items.map(([s, n, d0, mn, mx, dmn, dmx, ab, be, t1, t3, amn, amx]) =>
   `${s}: ${n} windows, the first ending on ${d0}; Ĥ of the returns between ${mn} (${dmn}) and ${mx} (${dmx}); ${ab}% of the windows above the band, ${be}% below; mean of the first third `
   + `${t1}, of the last ${t3}; |r| between ${amn} and ${amx}`).join('. ') + '.',
 fd1: 'Similarity dimension D = ln N/ln s: the segment ln 2/ln 2 = 1, the square ln 4/ln 2 = 2, the Koch curve ln 4/ln 3 = 1.262, the Sierpinski triangle ln 3/ln 2 = 1.585.',
 fd2: (H, n) => `Extra computation, not in the lecture: the fBm path with H = ${H} (${n} increments, seed 2026) is rescaled to the unit square; for ε = 1/2^k we split the square `
   + 'into columns of width ε and count in each column the boxes of side ε that the path touches.',
 fd3: list => 'Number of boxes N(ε): ' + list.map(([k, m, n]) => `ε = 1/${m}: ${n}`).join('; ') + '.',
 fd4: (km, D, th, df) => `OLS slope of log₂ N(ε) on log₂(1/ε), for ε from 1/2 to 1/${2 ** km}: D = ${D}; for fBm, D = 2 − H = ${th} (difference ${df}).`,
 fd: (H, km, D, th) => `Extra computation (box counting on the fBm paths of the lecture): H = ${H}, ε down to 1/${2 ** km}: D = ${D}, against 2 − H = ${th}.`,
 rh1: (s, y, N, K, n1, nK) => `${s}, ${yl(y)}: N = ${N}; ${K} block sizes between ${n1} and ${nK} ≈ N/4, evenly spaced on a log scale.`,
 rh2: (n, m, rest, rs, lx, ly) => `n = ${n}: ⌊N/n⌋ = ${m} non-overlapping blocks (` + (rest === 0 ? 'all returns fall into blocks' : rest === 1 ? 'the last return is left out'
   : `the last ${rest} returns are left out`) + `); average R/S over the blocks: (R/S)_n = ${rs}; u = log₁₀ n = ${lx}, v = log₁₀ (R/S)_n = ${ly}.`,
 rh3: (mu, mv, sxx, sxy, b, a) => `Over all points of the plot: ū = ${mu}, v̄ = ${mv}, Σ(u − ū)² = ${sxx}, Σ(u − ū)(v − v̄) = ${sxy}; Ĥ = ${sxy}/${sxx} = ${b}; log₁₀ c = v̄ − Ĥū = ${a}.`,
 rh4: (N, lo, hi, st, y) => `The 95% band of Ĥ_R/S for i.i.d. Normal series with N = ${N}: [${lo}, ${hi}]; ` + (st > 0 ? 'Ĥ is above the band' + (y === 'abs' ? ': long memory in volatility.' : '.')
   : st < 0 ? 'Ĥ is below the band.' : 'Ĥ lies inside the band: it is not separated from series without memory.'),
 rh: (s, y, H, lo, hi, x) => `${s}, ${yl(y)}${x ? X : ''}: Ĥ_R/S = ${H}; 95% i.i.d. band: [${lo}, ${hi}].`,
 mw1: (H, e, r1) => `H = ${H}: ρ(1) = 2^(2H − 1) − 1 = 2^${e} − 1 = ${r1}.`,
 mw2: (r10, r50, r100, k, rk, ap) => `fGn: ρ(10) = ${r10}, ρ(50) = ${r50}, ρ(100) = ${r100}; at lag ${k}: ρ = ${rk}, and the approximation H(2H − 1)k^(2H − 2) gives ${ap}.`,
 mw3: (phi, a10, k, ak) => `AR(1) with the same ρ(1), φ = ${phi}: ρ(10) = φ¹⁰ = ${a10}; ρ(${k}) = φ^${k} = ${ak}.`,
 mw4: (k, rk, ak) => `After ${k} lags a shock still matters in the fGn with autocorrelation ${rk}, and practically not at all in the AR(1) (${ak}): the same short-run behaviour, a very different long run.`,
 mw: (H, r1, r100, a100, x) => `H = ${H}${x ? X : ''}: ρ(1) = ${r1}; ρ(100) = ${r100} for the fGn and ${a100} for the AR(1) with the same ρ(1).`,
 aw1: (d, p1, p2) => `Weights of (1 − L)^d, d = ${d}: π₁ = −d = ${p1}, π₂ = π₁(1 − d)/2 = ${p2}.`,
 aw2: (s1, s2) => `MA weights: ψ₁ = d = ${s1}, ψ₂ = ψ₁(1 + d)/2 = ${s2}.`,
 aw3: (d, d1, r1, dp, dm, r2, r10, r100) => `ρ(1) = d/(1 − d) = ${d}/${d1} = ${r1}; ρ(2) = ρ(1) × ${dp}/${dm} = ${r2}; ρ(10) = ${r10}, ρ(100) = ${r100}.`,
 aw4: (r1, a10) => `AR(1) with φ = ${r1}: ρ(10) = φ¹⁰ = ${a10}.`,
 aw5: (t, kA, kF, H) => `The autocorrelation falls below ${t} after ${kA} lags for the AR(1) and after ${kF} lags for the ARFIMA; H = d + 0.5 = ${H}.`,
 aw: (d, r1, r10, r100, t, kA, kF, x) => `d = ${d}${x ? X : ''}: ρ(1) = ${r1}, ρ(10) = ${r10}, ρ(100) = ${r100}; below ${t} after ${kA} lags (AR(1)) and after ${kF} lags (ARFIMA).`,
 dw1: (s, y, N, m, Y1, Y2, YN) => `${s}, ${yl(y)}: N = ${N}, mean x̄ = ${m}; the profile Y_k = Σ_{j≤k}(x_j − x̄): Y₁ = ${Y1}, Y₂ = ${Y2}, …, Y_N = ${YN}.`,
 dw2: (n, m, rest, a, b, r) => `n = ${n}: ${m} blocks (` + (rest === 0 ? 'the whole profile falls into blocks' : rest === 1 ? 'the last value of the profile is left out'
   : `the last ${rest} values of the profile are left out`) + `). First block: the OLS line has the value ${a} at k = 1 and the slope ${b}; root mean square of the deviations from it: ${r}.`,
 dw3: (m, n, F) => `F(${n}) = root mean square of the deviations over all ${m} blocks of ${n} values: ${F}.`,
 dw4: (K, a, lo, hi, y) => `Slope of log F(n) on log n over the ${K} block sizes: α̂ = ${a}; the 95% i.i.d. band of DFA for the same N: [${lo}, ${hi}]`
   + (y === 'abs' ? '; for |r|, an α̂ far above the band points to long memory in volatility.' : '.'),
 dw: (s, y, n, F, a, x) => `${s}, ${yl(y)}${x ? X : ''}: F(${n}) = ${F}; DFA exponent α̂ = ${a}.`,
 gw1: (s, y, T, p, m, l1, lm) => `${s}, ${yl(y)}: T = ${T}, m = ⌊T^${p}⌋ = ${m}; the frequencies λ_j = 2πj/T go from λ₁ = ${l1} to λ_m = ${lm}.`,
 gw2: (I1, X1, y1) => `j = 1: I(λ₁) = ${I1}; the regressor −ln(4 sin²(λ₁/2)) = ${X1}; ln I(λ₁) = ${y1}.`,
 gw3: (mx, my, sxx, sxy, d) => `Over the m points: mean of the regressor ${mx}, mean of ln I ${my}, Σ(X − X̄)² = ${sxx}, Σ(X − X̄)(y − ȳ) = ${sxy}; d̂ = ${sxy}/${sxx} = ${d}.`,
 gw4: (m, se, lo, hi, H) => `SE = π/√(24m) = π/√(24 × ${m}) = ${se}; the 95% interval: d̂ ± 1.96 SE = [${lo}, ${hi}]` + (num(lo) < 0 && num(hi) > 0 ? ', which contains 0' : ', which does not contain 0') + `; Ĥ = d̂ + 0.5 = ${H}.`,
 gw: (s, p, m, dr, da, se, x) => `${s}, m = ⌊T^${p}⌋ = ${m}${x ? X : ''}: d̂ = ${dr} for the returns and ${da} for the absolute returns, SE ${se}. A larger m lowers the SE but lets the short-run dynamics in.`,
 lw1: (s, y, N, R, g0, V0) => `${s}, ${yl(y)}: N = ${N}; range of the cumulative deviations R_N = ${R}; γ̂₀ = ${g0}; classical R/S: V_N(0) = R_N/√(Nγ̂₀) = ${V0}.`,
 lw2: (rho, A, B, AB, q) => `ρ̂₁ = ${rho}; (3N/2)^(1/3) = ${A}; (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = ${B}; the product is ${AB}, so q = ${q}.`,
 lw3: (q, g0, gs, s2) => (q === 0 ? `With q = 0, σ̂²(0) = γ̂₀ = ${g0}.` : `σ̂²(${q}) = γ̂₀ + 2 Σ (1 − j/${q + 1}) γ̂_j, j = 1, …, ${q}, with ` + gs.map((v, i) => `γ̂${sub(i + 1)} = ${v}`).join(', ')
   + (q > gs.length ? ', …' : '') + `: σ̂²(${q}) = ${s2}.`),
 lw4: (R, N, s2, V, rej) => `V_N(q) = ${R}/√(${N} × ${s2}) = ${V}: ` + (rej ? 'outside [0.809, 1.862], so short memory is rejected at 5%.' : 'inside [0.809, 1.862], so short memory is not rejected at 5%.'),
 lw: (s, Vr, qr, rr, Va, qa, ra, list, x) => `${s}${x ? X : ''}: returns V = ${Vr} (q = ${qr}), short memory ${rr ? 'rejected' : 'not rejected'}; absolute returns V = ${Va} (q = ${qa}), `
   + `${ra ? 'rejected' : 'not rejected'}. Among the six series, short memory of the returns is rejected for: ${lst(list)}.`,
 st: (s, n, d0, outR, lr, nA, la, x) => `${s}: N = ${n} daily returns, from ${d0} to 18 September 2026${x ? X : ''}. Returns: ` + (outR.length ? `outside the band: ${lst(outR)}` : 'every estimate lies inside its band')
   + `; Lo's test ${lr ? 'rejects' : 'does not reject'} short memory. Absolute returns: ${nA} of 3 estimates outside the band; Lo's test ${la ? 'rejects' : 'does not reject'} short memory.`,
 mt: (p, m0, m1, ex, dmn, dmx, nr, x) => `GPH bandwidth m = ⌊T^${p}⌋, between ${m0} and ${m1} frequencies${x ? X : ''}. Returns: d̂ ± 1.96 SE excludes 0 for: ${lst(ex)}. `
   + `Absolute returns: d̂ between ${dmn} and ${dmx}; Lo's test rejects short memory for ${nr} of 6 series.`,
 rk1: (z, s, v) => `One-day VaR 1%, Normal case with mean 0: −z₀.₀₁ σ = ${z} × ${s} = ${v}%.`,
 rk2: (h, H, f, v) => `h = ${h}, H = ${H}: the factor h^H = ${h}^${H} = ${f}; VaR 1% over ${days(h)} = ${v}%.`,
 rk3: (h, sq, vs, p) => `The √h rule: √${h} = ${sq}, VaR = ${vs}%; with the exponent H the risk is ${neg(p)}% ${num(p) >= 0 ? 'higher' : 'lower'}.`,
 rk4: (sp, bt, s10) => `The data: the ratio of the 10-day to the one-day standard deviation is ${sp} for the S&P 500 and ${bt} for the BET, against √10 = ${s10}.`,
 rk: (H, h, f, v, vs, p, p250, x) => `H = ${H}, h = ${h}${x ? X : ''}: factor ${f}, VaR 1% = ${v}%, against ${vs}% with the √h rule; difference: ${p}% at ${days(h)}, ${p250}% at 250 days.`
};
})();
window.CH11_TEXT.predict = {"scaling": {"key": 0, "opts": ["Below: about 2.80.", "Exactly 3.16: the square-root rule is exact for any stationary series.", "Above: heavy tails raise the multi-day risk."]}, "fractal-dim": {"key": 0, "opts": ["About 1.5 = 2 − H.", "1: the path is a curve, so it has the dimension of a segment.", "2: the path fills the square."]}, "rs-example": {"key": 0, "opts": ["It is of the same order, about 1.5.", "It is about 15 times larger: R grows with volatility.", "It is almost 0: in a crisis the deviations offset each other."]}, "rs-hurst": {"key": 0, "opts": ["About 0.86.", "Still 0.53: |r| has the same order of days as r.", "About 0.5: the absolute value removes the sign, and with it the memory."]}, "memory-we": {"key": 0, "opts": ["It rises from about 0.001 to about 0.54.", "It stays near 0: after 100 lags every autocorrelation is gone.", "It rises linearly with H, from 0.55 to 0.95."]}, "fbm": {"key": 0, "opts": ["About 0.32.", "0.7: the autocorrelation of the increments equals H.", "0: fBm increments are independent, as for Brownian motion."]}, "fgn": {"key": 0, "opts": ["The estimated mean absorbs part of the slow component of the series.", "The circulant-embedding simulation is not exact.", "The fGn ACF formula holds only for large k."]}, "arfima": {"key": 0, "opts": ["About 0.31.", "0.667⁵⁰, practically 0: geometric decay, as for an AR(1).", "0.4: ρ(k) = d at every lag."]}, "arfima-we": {"key": 0, "opts": ["After 222 lags.", "After 4 lags, like the AR(1) with the same ρ(1).", "Never: long memory means ρ(k) does not go to 0."]}, "dfa-we": {"key": 0, "opts": ["It grows about 13 times, close to √n.", "It grows about 168 times, in proportion to n.", "It falls: in large blocks the line removes more."]}, "rs-dfa": {"key": 0, "opts": ["No: i.i.d. series of the same length give 0.55.", "Yes: any slope above 0.5 means persistence.", "Yes, because DFA confirms it with 0.49."]}, "gph-we": {"key": 0, "opts": ["About 0.28, and it contains 0.", "About 0.14 (± 1 SE), and it does not contain 0.", "About 0.28, and it does not contain 0, because d̂ = 0.036 > 0."]}, "gph": {"key": 0, "opts": ["The SE falls to about 0.03 and d̂ rises to about 0.55.", "The SE stays at 0.071: it depends only on T.", "d̂ stays at 0.47: the estimator does not depend on m."]}, "lo-we": {"key": 0, "opts": ["7.", "0: ρ̂₁ is negative, so there is no short memory to correct.", "81, like the GPH bandwidth ⌊√N⌋."]}, "lo": {"key": 0, "opts": ["About 60%.", "About 5%: the nominal level of the test.", "About 50%: φ = 0.5."]}, "mc": {"key": 0, "opts": ["GPH: width about 0.53.", "R/S: it is biased, so also the least precise.", "DFA: removing local trends loses information."]}, "sp-table": {"key": 0, "opts": ["R/S (0.53): its band for N = 6,714 is [0.51, 0.57].", "GPH (Ĥ = 0.54): d̂ = 0.036 is positive.", "Lo's test (V = 1.56): it is above 1."]}, "nile": {"key": 0, "opts": ["About 0.63, inside the i.i.d. band for N = 100.", "It stays at 0.78: long memory does not depend on the mean.", "Exactly 0.5: without the break the series is white noise."]}, "spurious": {"key": 0, "opts": ["About 0.79.", "0.5: the series has no memory, so GPH is unbiased.", "About 0.55, the size of the small-sample R/S bias."]}, "vol-acf": {"key": 0, "opts": ["31, against about 12 expected by chance.", "0: returns are uncorrelated, so no autocorrelation leaves the band.", "227, as for |r|."]}, "mk-table": {"key": 0, "opts": ["Only for the BET.", "For all six: they all have DFA above 0.7 for |r|.", "For none: returns have no long memory in any market."]}, "markets": {"key": 0, "opts": ["It falls into the i.i.d. band, between 0.46 and 0.55.", "It stays the same: the heavy tails of |r| do not change under a shuffle.", "It rises: the shuffle brings the extreme days together."]}, "rolling": {"key": 0, "opts": ["About 24%, none above.", "About 2.5%: that is what the band gives by chance.", "About 24%, but above the band: crises create persistence."]}, "risk-we": {"key": 0, "opts": ["By about 12%.", "By 10%: H rises by 10%.", "By 0.05 percentage points: the difference of the exponents."]}};
window.CH11_TEXT.ticket = {"title": "Check Yourself", "id": "check", "store": "sfm-ch11-check-en", "qs": [{"q": "An R/S estimate on 1000 i.i.d. returns is Ĥ = 0.56. Is it evidence of long memory?", "key": 0, "opts": [{"t": "No: the i.i.d. band for N = 1000 is [0.50, 0.63], centred around 0.562.", "why": "We compare Ĥ with its distribution under the null, for the same N; R/S is biased upwards in small samples."}, {"t": "Yes: 0.56 > 0.5.", "why": "The threshold 0.5 ignores the bias: i.i.d. series with N = 1000 give 0.562 on average."}, {"t": "It cannot be said: R/S has no confidence band.", "why": "A Monte Carlo band exists for any estimator: apply it to simulated series of the same length."}]}, {"q": "GPH gives d̂ = 0.35 for |r|. What is Ĥ?", "key": 0, "opts": [{"t": "0.85 = d̂ + 0.5: long memory in volatility.", "why": "d = H − 0.5, so H = d + 0.5; the estimate concerns |r|, so volatility, not the direction of returns."}, {"t": "0.35: GPH estimates H directly.", "why": "GPH estimates the memory parameter d from the slope of the log periodogram."}, {"t": "0.15 = 0.5 − d̂: anti-persistence.", "why": "The sign is reversed: d > 0 means H > 0.5."}]}, {"q": "After a random shuffle of |r|, DFA falls from 0.9 to 0.5. What does this show?", "key": 0, "opts": [{"t": "The memory comes from the order of the days, not from the distribution (heavy tails).", "why": "The shuffle keeps the distribution and destroys only the order; the exponent fell, so the order produced it."}, {"t": "The heavy tails of |r| produce the exponent 0.9.", "why": "The tails are the same after the shuffle and the exponent fell: they did not produce it."}, {"t": "The shuffle introduced a structural break.", "why": "A random shuffle destroys breaks and trends; it does not create them."}]}, {"q": "The one-day VaR 1% is 2.79%. With H = 0.45, what is the 10-day VaR 1%?", "key": 0, "opts": [{"t": "7.87% = 2.79 × 10^0.45.", "why": "Risk scales with h^H; 10^0.45 = 2.82."}, {"t": "8.83%: the √10 rule.", "why": "The √h rule assumes H = 0.5; with H = 0.45 the factor is 2.82, not 3.16."}, {"t": "12.6% = 2.79 × 10 × 0.45.", "why": "The factor is h to the power H, not h times H."}]}]};
window.CH11_TEXT.veil = {"scaling": {"out": true, "charts": ["sc-chart"]}, "fractal-dim": {"out": true, "charts": ["fd-chart", "fd-we-box"]}, "rs-example": {"out": true}, "rs-hurst": {"out": true}, "memory-we": {"out": true}, "fbm": {"out": true}, "fgn": {"out": true}, "arfima": {"out": true, "charts": ["af-chart"]}, "arfima-we": {"out": true, "charts": ["aw-chart", "aw-we-box"]}, "dfa-we": {"out": true, "charts": ["dw-chart"]}, "rs-dfa": {"out": true, "charts": ["rd-chart"]}, "gph-we": {"out": true, "charts": ["gw-we-box"]}, "gph": {"out": true}, "lo-we": {"out": true, "charts": ["lw-chart", "lw-we-box", "lw-tab"]}, "lo": {"out": true, "charts": ["lo-chart"]}, "mc": {"out": true, "charts": ["mc-chart"]}, "sp-table": {"out": true, "charts": ["st-chart", "st-tab"]}, "nile": {"out": true, "charts": ["nl-chart"]}, "spurious": {"out": true, "charts": ["su-chart"]}, "vol-acf": {"out": true, "charts": ["va-chart"]}, "mk-table": {"out": true, "charts": ["mt-chart", "mt-tab"]}, "markets": {"out": true, "charts": ["mk-chart"]}, "rolling": {"out": true, "charts": ["ro-chart"]}, "risk-we": {"out": true, "charts": ["rk-we-box"]}};
