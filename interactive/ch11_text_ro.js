// Readouts of the Chapter 11 interactive page, written by latex/build_interactive_ch11.py.
(function () {
const num = s => +String(s).replace(/[\s  ]/g, '').replace('−', '-').replace(',', '.');
const de = s => { const n = Math.abs(Math.round(num(s))); return n >= 20 && (n % 100 >= 20 || n % 100 === 0) ? 'de ' : ''; };
const zile = s => (num(s) === 1 ? '1 zi' : `${s} ${de(s)}zile`);
const X = ' (rulare suplimentară a codului din curs)';
const lst = a => (a.length ? a.join(', ') : 'niciuna');
const yl = y => (y === 'abs' ? 'randamentele absolute |r|' : 'randamentele r');
const sub = i => String(i).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
const neg = s => String(s).replace('−', '');
window.CH11_TEXT = {
 h: { r: 'randamente rₜ', abs: 'randamente absolute |rₜ|', band: 'banda de 95%, i.i.d.',
      st: ['', 'R/S', 'DFA', 'GPH d̂ (SE)', 'd̂ ± 1,96 SE', 'Lo V_N(q)', 'q'],
      lw: [['', { v: 'randamente r', span: 4 }, { v: 'randamente absolute |r|', span: 4 }], ['', 'V(0)', 'V(q)', 'q', 'memoria scurtă', 'V(0)', 'V(q)', 'q', 'memoria scurtă']],
      mt: [['', '', { v: 'randamente rₜ', span: 4 }, { v: 'randamente absolute |rₜ|', span: 3 }, ''], ['', 'N', 'R/S', 'DFA', 'GPH d̂', 'd̂ ± 1,96 SE', 'DFA', 'GPH d̂', 'Lo V', 'banda DFA']] },
 sc: (hm, s10, shm, items, x) => `Abaterea standard a randamentelor pe h zile, fără suprapunere, împărțită la cea zilnică, h de la 1 la ${hm}${x ? X : ''}. `
   + items.map(([n, H, r10, rh]) => `${n}: panta ${H}, raportul la 10 zile ${r10}` + (+hm > 10 ? `, la ${zile(hm)} ${rh}` : '')).join('; ')
   + `. Regula rădăcinii pătrate a timpului: √10 = ${s10}` + (+hm > 10 ? `, √${hm} = ${shm}.` : '.'),
 rxlec: 'cele opt randamente din curs',
 rxsrc: (d0, d1) => `randamentele S&P 500 din ${d0} – ${d1} (8 zile de tranzacționare)`,
 rx1: (src, xs, s, n, m) => `${src[0].toUpperCase() + src.slice(1)}, în %: ${xs}; suma ${s}, media x̄ = ${s}/${n} = ${m}.`,
 rx2: devs => `Abaterile de la medie, x_k − x̄: ${devs}.`,
 rx3: (Y, yn) => `Abaterile cumulate Y_k: ${Y}; Y₈ = ${yn}, zero prin construcție.`,
 rx4: (mx, kx, mn, kn, R, ss, n, S) => `R = max Y_k − min Y_k = ${mx} (k = ${kx}) − (${mn}) (k = ${kn}) = ${R}; S = √(Σ(x_k − x̄)²/n) = √(${ss}/${n}) = ${S}.`,
 rx5: (R, S, rs) => `(R/S)₈ = ${R}/${S} = ${rs}. Un bloc dă un singur punct al graficului log-log; H are nevoie de multe mărimi de bloc.`,
 rx: (src, R, S, rs, x) => `${src[0].toUpperCase() + src.slice(1)}${x ? X : ''}: R = ${R}, S = ${S}, (R/S)₈ = ${rs}.`,
 fb: (items, x) => `Aceleași numere aleatoare (sămînța 2026) pentru toate traiectoriile${x ? X : ''}. ` + items.map(([H, a, t, r]) => `H = ${H}: autocorelația de ordinul 1 a creșterilor `
   + `${a} (teoretic 2^(2H − 1) − 1 = ${t}), amplitudinea traiectoriei ${r}`).join('; ') + '.',
 fg: (H, r1, Hb, r1b, r10, r50, r100, a10, a100, s1, gap, x) => `H = ${H}${x ? X : ''}: ρ(1) = 2^(2H − 1) − 1 = ${r1}; pentru H = ${Hb}, ρ(1) = ${r1b}. fGn cu H = ${H}: ρ(10) = ${r10}, `
   + `ρ(50) = ${r50}, ρ(100) = ${r100}; AR(1) cu φ = ${r1}: ρ(10) = ${a10}, ρ(100) = ${a100}. ACF de selecție (n = 5000) la lagul 1: ${s1}; în medie, pe lagurile 1–50, ea este `
   + (num(gap) >= 0 ? `cu ${gap} sub cea teoretică.` : `cu ${neg(gap)} peste cea teoretică.`),
 af: (items, x) => `Traiectorii de 1000 de valori, ACF de selecție din 20000 de valori${x ? X : ''}. ` + items.map(([d, r1, r10, r50, s1, sd, st]) => `d = ${d}: ρ(1) = d/(1 − d) = ${r1}, `
   + `ρ(10) = ${r10}, ρ(50) = ${r50} (ACF de selecție la lagul 1: ${s1}); abaterea standard a traiectoriei ${sd}, teoretic ${st}`).join('; ') + '.',
 rd: (s, n, K, n1, nK, rs, rsi, dfa, dfai, rsa, dfaa, x) => `${s}, ${n} ${de(n)}randamente, ${K} ${de(K)}mărimi de bloc de la ${n1} la ${nK} ${de(nK)}zile${x ? X : ''}. Randamente: panta R/S ${rs} `
   + `(serii i.i.d. de aceeași lungime: ${rsi}), DFA ${dfa} (i.i.d.: ${dfai}). Randamente absolute: R/S ${rsa}, DFA ${dfaa}.`,
 gp: (s, p, m, d, se, lo, hi, da, Ha, x) => `${s}, m = ⌊T^${p}⌋ = ${m} frecvențe${x ? X : ''}: randamente d̂ = ${d} (SE ${se}; d̂ ± 1,96 SE = [${lo}; ${hi}], `
   + (num(lo) < 0 && num(hi) > 0 ? 'conține 0' : 'nu conține 0') + `); randamente absolute d̂ = ${da}, Ĥ = d̂ + 0,5 = ${Ha}.`,
 lo: (n, c5, m5, q5, c7, m7, c0, m0, m8, x) => `500 de serii de ${n} ${de(n)}valori${x ? X : ''}. AR(1) cu φ = 0,5 (memorie scurtă, H₀ adevărată): testul clasic respinge în ${c5} din cazuri, `
   + `testul modificat în ${m5} (q mediu ${q5}); cu φ = 0: ${c0} și ${m0}. fGn cu H = 0,7: ${c7} și ${m7}; cu H = 0,8, testul modificat respinge în ${m8}.`,
 mc: (lv, r250, r1000, r5000, d1000, g1000, wr, wd, wg, wd250, lo, hi, x) => `500 de serii i.i.d. Normale pentru fiecare N, banda de ${lv}%${x ? X : ''}. R/S: media ${r250} la N = 250, `
   + `${r1000} la N = 1000, ${r5000} la N = 5000; DFA ${d1000} și GPH ${g1000} la N = 1000. Lățimea benzii la N = 1000: R/S ${wr} (de la ${lo} la ${hi}), DFA ${wd}, GPH ${wg}; `
   + `la N = 250, DFA ${wd250}.`,
 nl: (b, mp, mq, dr, H1, d1, H2, d2, lo, hi, nm, x) => `Ruptura după ${b}${x ? X : ''}: debitul mediu ${mp} înainte, ${mq} după (în 10⁸ m³), `
   + (num(dr) >= 0 ? `o scădere de ${dr}%` : `o creștere de ${neg(dr)}%`) + `. Seria brută: Ĥ_R/S = ${H1}, DFA ${d1}; după eliminarea celor două medii de regim: ${H2} și ${d2}. `
   + `Banda i.i.d. pentru R/S cu N = 100: [${lo}; ${hi}], centrată în ${nm}.`,
 su: (n, g5, d5, r5, r0, a95, a99, rlo, rhi, x) => `300 de serii de ${n} ${de(n)}valori${x ? X : ''}. Schimbare a mediei de 0,5 abateri standard la mijloc: GPH Ĥ = ${g5}, DFA ${d5}, `
   + `R/S ${r5} (fără schimbare, R/S dă deja ${r0}: deplasarea de eșantion mic). GARCH(1,1): DFA pentru |r| ${a95} la α + β = 0,95 și ${a99} la 0,99; pentru randamente, DFA ` + (rlo === rhi ? `${rlo} la toate persistențele.` : `între ${rlo} și ${rhi}.`),
 va: (s, L, band, a1, a50, a100, np, s1, s50, r1, mx, no, ex, top, x) => `${s}, primele ${L} ${de(L)}laguri${x ? X : ''}; banda de 95%: ±${band}. |r|: ρ(1) = ${a1}, ρ(50) = ${a50}, `
   + `ρ(100) = ${a100}; ${np} dintre cele ${L} ${de(L)}laguri sînt peste bandă. r²: ρ(1) = ${s1}, ρ(50) = ${s50}. r: ρ(1) = ${r1}, apoi |ρ(k)| ≤ ${mx}; ${no} dintre lagurile 2–${L} `
   + `sînt în afara benzii (circa ${ex} așteptate din întîmplare), cele mai mari la lagurile ${top}.`,
 mk: (e, amin, amax, smin, smax, ins, above, x) => `Estimatorul ${e}${x ? X : ''}: pentru |r|, între ${amin} și ${amax}; după o permutare aleatoare, între ${smin} și ${smax}, `
   + `${ins} din 6 în interiorul benzilor: memoria se află în ordinea zilelor. Randamente peste bandă: ${lst(above)}.`,
 ro: (W, lo, hi, items, x) => `Ferestre de ${W} de observații, mutate cu 21${x ? X : ''}; banda i.i.d. de 95%: [${lo}; ${hi}]. ` + items.map(([s, n, d0, mn, mx, dmn, dmx, ab, be, t1, t3, amn, amx]) =>
   `${s}: ${n} ${de(n)}ferestre, prima încheiată la ${d0}; Ĥ al randamentelor între ${mn} (${dmn}) și ${mx} (${dmx}); ${ab}% dintre ferestre peste bandă, ${be}% sub; media primei treimi `
   + `${t1}, a ultimei ${t3}; |r| între ${amn} și ${amx}`).join('. ') + '.',
 fd1: 'Dimensiunea de similaritate D = ln N/ln s: segmentul ln 2/ln 2 = 1, pătratul ln 4/ln 2 = 2, curba Koch ln 4/ln 3 = 1,262, triunghiul Sierpinski ln 3/ln 2 = 1,585.',
 fd2: (H, n) => `Calcul suplimentar, care nu este în curs: traiectoria fBm cu H = ${H} (${n} de creșteri, sămînța 2026) este rescalată în pătratul unitate; pentru ε = 1/2^k împărțim `
   + 'pătratul în coloane de lățime ε și numărăm în fiecare coloană cutiile de latură ε atinse de traiectorie.',
 fd3: list => 'Numărul de cutii N(ε): ' + list.map(([k, m, n]) => `ε = 1/${m}: ${n}`).join('; ') + '.',
 fd4: (km, D, th, df) => `Panta OLS a lui log₂ N(ε) în funcție de log₂(1/ε), pentru ε de la 1/2 la 1/${2 ** km}: D = ${D}; pentru fBm, D = 2 − H = ${th} (diferența ${df}).`,
 fd: (H, km, D, th) => `Calcul suplimentar (numărarea cutiilor pe traiectoriile fBm din curs): H = ${H}, ε pînă la 1/${2 ** km}: D = ${D}, față de 2 − H = ${th}.`,
 rh1: (s, y, N, K, n1, nK) => `${s}, ${yl(y)}: N = ${N}; ${K} ${de(K)}mărimi de bloc între ${n1} și ${nK} ≈ N/4, la distanțe egale pe scară logaritmică.`,
 rh2: (n, m, rest, rs, lx, ly) => `n = ${n}: ⌊N/n⌋ = ${m} ${de(m)}blocuri fără suprapunere (` + (rest === 0 ? 'toate randamentele intră în blocuri' : rest === 1 ? 'ultimul randament rămîne în afara lor'
   : `ultimele ${rest} ${de(rest)}randamente rămîn în afara lor`) + `); media R/S pe blocuri: (R/S)_n = ${rs}; u = log₁₀ n = ${lx}, v = log₁₀ (R/S)_n = ${ly}.`,
 rh3: (mu, mv, sxx, sxy, b, a) => `Pe toate punctele graficului: ū = ${mu}, v̄ = ${mv}, Σ(u − ū)² = ${sxx}, Σ(u − ū)(v − v̄) = ${sxy}; Ĥ = ${sxy}/${sxx} = ${b}; log₁₀ c = v̄ − Ĥū = ${a}.`,
 rh4: (N, lo, hi, st, y) => `Banda de 95% a lui Ĥ_R/S pentru serii i.i.d. Normale cu N = ${N}: [${lo}; ${hi}]; ` + (st > 0 ? 'Ĥ este peste bandă' + (y === 'abs' ? ': memorie lungă a volatilității.' : '.')
   : st < 0 ? 'Ĥ este sub bandă.' : 'Ĥ se află în bandă: nu se deosebește de seriile fără memorie.'),
 rh: (s, y, H, lo, hi, x) => `${s}, ${yl(y)}${x ? X : ''}: Ĥ_R/S = ${H}; banda i.i.d. de 95%: [${lo}; ${hi}].`,
 mw1: (H, e, r1) => `H = ${H}: ρ(1) = 2^(2H − 1) − 1 = 2^${e} − 1 = ${r1}.`,
 mw2: (r10, r50, r100, k, rk, ap) => `fGn: ρ(10) = ${r10}, ρ(50) = ${r50}, ρ(100) = ${r100}; la lagul ${k}: ρ = ${rk}, iar aproximarea H(2H − 1)k^(2H − 2) dă ${ap}.`,
 mw3: (phi, a10, k, ak) => `AR(1) cu același ρ(1), φ = ${phi}: ρ(10) = φ¹⁰ = ${a10}; ρ(${k}) = φ^${k} = ${ak}.`,
 mw4: (k, rk, ak) => `După ${k} ${de(k)}laguri, un șoc mai contează în fGn cu autocorelația ${rk}, iar în AR(1) practic deloc (${ak}): același comportament pe termen scurt, un termen lung foarte diferit.`,
 mw: (H, r1, r100, a100, x) => `H = ${H}${x ? X : ''}: ρ(1) = ${r1}; ρ(100) = ${r100} pentru fGn și ${a100} pentru AR(1) cu același ρ(1).`,
 aw1: (d, p1, p2) => `Ponderile lui (1 − L)^d, d = ${d}: π₁ = −d = ${p1}, π₂ = π₁(1 − d)/2 = ${p2}.`,
 aw2: (s1, s2) => `Ponderile MA: ψ₁ = d = ${s1}, ψ₂ = ψ₁(1 + d)/2 = ${s2}.`,
 aw3: (d, d1, r1, dp, dm, r2, r10, r100) => `ρ(1) = d/(1 − d) = ${d}/${d1} = ${r1}; ρ(2) = ρ(1) × ${dp}/${dm} = ${r2}; ρ(10) = ${r10}, ρ(100) = ${r100}.`,
 aw4: (r1, a10) => `AR(1) cu φ = ${r1}: ρ(10) = φ¹⁰ = ${a10}.`,
 aw5: (t, kA, kF, H) => `Autocorelația scade sub ${t} după ${kA} ${de(kA)}laguri pentru AR(1) și după ${kF} ${de(kF)}laguri pentru ARFIMA; H = d + 0,5 = ${H}.`,
 aw: (d, r1, r10, r100, t, kA, kF, x) => `d = ${d}${x ? X : ''}: ρ(1) = ${r1}, ρ(10) = ${r10}, ρ(100) = ${r100}; sub ${t} după ${kA} ${de(kA)}laguri (AR(1)) și după ${kF} ${de(kF)}laguri (ARFIMA).`,
 dw1: (s, y, N, m, Y1, Y2, YN) => `${s}, ${yl(y)}: N = ${N}, media x̄ = ${m}; profilul Y_k = Σ_{j≤k}(x_j − x̄): Y₁ = ${Y1}, Y₂ = ${Y2}, …, Y_N = ${YN}.`,
 dw2: (n, m, rest, a, b, r) => `n = ${n}: ${m} ${de(m)}blocuri (` + (rest === 0 ? 'tot profilul intră în blocuri' : rest === 1 ? 'ultima valoare a profilului rămîne în afara lor'
   : `ultimele ${rest} ${de(rest)}valori ale profilului rămîn în afara lor`) + `). Primul bloc: dreapta OLS are la k = 1 valoarea ${a} și panta ${b}; rădăcina mediei pătratelor abaterilor de la ea: ${r}.`,
 dw3: (m, n, F) => `F(${n}) = rădăcina mediei pătratelor abaterilor pe toate cele ${m} ${de(m)}blocuri de cîte ${n} ${de(n)}valori: ${F}.`,
 dw4: (K, a, lo, hi, y) => `Panta lui log F(n) în funcție de log n pe cele ${K} ${de(K)}mărimi de bloc: α̂ = ${a}; banda i.i.d. de 95% a DFA pentru același N: [${lo}; ${hi}]`
   + (y === 'abs' ? '; pentru |r|, α̂ mult peste bandă indică memorie lungă a volatilității.' : '.'),
 dw: (s, y, n, F, a, x) => `${s}, ${yl(y)}${x ? X : ''}: F(${n}) = ${F}; exponentul DFA α̂ = ${a}.`,
 gw1: (s, y, T, p, m, l1, lm) => `${s}, ${yl(y)}: T = ${T}, m = ⌊T^${p}⌋ = ${m}; frecvențele λ_j = 2πj/T, de la λ₁ = ${l1} la λ_m = ${lm}.`,
 gw2: (I1, X1, y1) => `j = 1: I(λ₁) = ${I1}; regresorul −ln(4 sin²(λ₁/2)) = ${X1}; ln I(λ₁) = ${y1}.`,
 gw3: (mx, my, sxx, sxy, d) => `Pe cele m puncte: media regresorului ${mx}, media lui ln I ${my}, Σ(X − X̄)² = ${sxx}, Σ(X − X̄)(y − ȳ) = ${sxy}; d̂ = ${sxy}/${sxx} = ${d}.`,
 gw4: (m, se, lo, hi, H) => `SE = π/√(24m) = π/√(24 × ${m}) = ${se}; intervalul de 95%: d̂ ± 1,96 SE = [${lo}; ${hi}]` + (num(lo) < 0 && num(hi) > 0 ? ', conține 0' : ', nu conține 0') + `; Ĥ = d̂ + 0,5 = ${H}.`,
 gw: (s, p, m, dr, da, se, x) => `${s}, m = ⌊T^${p}⌋ = ${m}${x ? X : ''}: d̂ = ${dr} pentru randamente și ${da} pentru randamentele absolute, SE ${se}. Un m mai mare micșorează SE, dar lasă să intre dinamica de termen scurt.`,
 lw1: (s, y, N, R, g0, V0) => `${s}, ${yl(y)}: N = ${N}; amplitudinea abaterilor cumulate R_N = ${R}; γ̂₀ = ${g0}; R/S clasic: V_N(0) = R_N/√(Nγ̂₀) = ${V0}.`,
 lw2: (rho, A, B, AB, q) => `ρ̂₁ = ${rho}; (3N/2)^(1/3) = ${A}; (2|ρ̂₁|/(1 − ρ̂₁²))^(2/3) = ${B}; produsul ${AB}, deci q = ${q}.`,
 lw3: (q, g0, gs, s2) => (q === 0 ? `Cu q = 0, σ̂²(0) = γ̂₀ = ${g0}.` : `σ̂²(${q}) = γ̂₀ + 2 Σ (1 − j/${q + 1}) γ̂_j, j = 1, …, ${q}, cu ` + gs.map((v, i) => `γ̂${sub(i + 1)} = ${v}`).join(', ')
   + (q > gs.length ? ', …' : '') + `: σ̂²(${q}) = ${s2}.`),
 lw4: (R, N, s2, V, rej) => `V_N(q) = ${R}/√(${N} × ${s2}) = ${V}: ` + (rej ? 'în afara intervalului [0,809; 1,862], deci memoria scurtă este respinsă la 5%.' : 'în intervalul [0,809; 1,862], deci memoria scurtă nu este respinsă la 5%.'),
 lw: (s, Vr, qr, rr, Va, qa, ra, list, x) => `${s}${x ? X : ''}: randamente V = ${Vr} (q = ${qr}), memoria scurtă ${rr ? 'este respinsă' : 'nu este respinsă'}; randamente absolute V = ${Va} (q = ${qa}), `
   + `${ra ? 'respinsă' : 'nerespinsă'}. Dintre cele șase serii, memoria scurtă a randamentelor este respinsă pentru: ${lst(list)}.`,
 st: (s, n, d0, outR, lr, nA, la, x) => `${s}: N = ${n} randamente zilnice, din ${d0} pînă la 18 septembrie 2026${x ? X : ''}. Randamente: ` + (outR.length ? `în afara benzii: ${lst(outR)}` : 'fiecare estimare se află în banda ei')
   + `; testul lui Lo ${lr ? 'respinge' : 'nu respinge'} memoria scurtă. Randamente absolute: ${nA} din 3 estimări în afara benzii; testul lui Lo ${la ? 'respinge' : 'nu respinge'} memoria scurtă.`,
 mt: (p, m0, m1, ex, dmn, dmx, nr, x) => `Lățimea de bandă GPH m = ⌊T^${p}⌋, între ${m0} și ${m1} frecvențe${x ? X : ''}. Randamente: d̂ ± 1,96 SE exclude 0 pentru: ${lst(ex)}. `
   + `Randamente absolute: d̂ între ${dmn} și ${dmx}; testul lui Lo respinge memoria scurtă pentru ${nr} din 6 serii.`,
 rk1: (z, s, v) => `VaR 1% pe o zi, cazul Normal cu media 0: −z₀,₀₁ σ = ${z} × ${s} = ${v}%.`,
 rk2: (h, H, f, v) => `h = ${h}, H = ${H}: factorul h^H = ${h}^${H} = ${f}; VaR 1% pe ${zile(h)} = ${v}%.`,
 rk3: (h, sq, vs, p) => `Regula √h: √${h} = ${sq}, VaR = ${vs}%; cu exponentul H, riscul este cu ${neg(p)}% ${num(p) >= 0 ? 'mai mare' : 'mai mic'}.`,
 rk4: (sp, bt, s10) => `Datele: raportul abaterilor standard pe 10 zile și pe o zi este ${sp} pentru S&P 500 și ${bt} pentru BET, față de √10 = ${s10}.`,
 rk: (H, h, f, v, vs, p, p250, x) => `H = ${H}, h = ${h}${x ? X : ''}: factorul ${f}, VaR 1% = ${v}%, față de ${vs}% cu regula √h; diferența: ${p}% la ${zile(h)}, ${p250}% la 250 de zile.`
};
})();
window.CH11_TEXT.predict = {"scaling": {"key": 0, "opts": ["Sub: circa 2,80.", "Exact 3,16: regula rădăcinii pătrate este exactă pentru orice serie staționară.", "Peste: cozile groase măresc riscul pe mai multe zile."]}, "fractal-dim": {"key": 0, "opts": ["Circa 1,5 = 2 − H.", "1: traiectoria este o curbă, deci are dimensiunea unui segment.", "2: traiectoria umple pătratul."]}, "rs-example": {"key": 0, "opts": ["Este de același ordin, circa 1,5.", "Este de circa 15 ori mai mare: R crește cu volatilitatea.", "Este aproape 0: în criză abaterile se compensează."]}, "rs-hurst": {"key": 0, "opts": ["Circa 0,86.", "Tot 0,53: |r| are aceeași ordine a zilelor ca r.", "Circa 0,5: valoarea absolută elimină semnul, deci și memoria."]}, "memory-we": {"key": 0, "opts": ["Crește de la circa 0,001 la circa 0,54.", "Rămîne aproape 0: după 100 de laguri orice autocorelație dispare.", "Crește liniar cu H, de la 0,55 la 0,95."]}, "fbm": {"key": 0, "opts": ["Circa 0,32.", "0,7: autocorelația creșterilor este egală cu H.", "0: creșterile fBm sînt independente, ca la mișcarea browniană."]}, "fgn": {"key": 0, "opts": ["Media estimată absoarbe o parte din componenta lentă a seriei.", "Simularea prin scufundare circulantă nu este exactă.", "Formula ACF a fGn este valabilă doar pentru k mare."]}, "arfima": {"key": 0, "opts": ["Circa 0,31.", "0,667⁵⁰, practic 0: scădere geometrică, ca la AR(1).", "0,4: ρ(k) = d la orice lag."]}, "arfima-we": {"key": 0, "opts": ["După 222 de laguri.", "După 4 laguri, ca AR(1) cu același ρ(1).", "Niciodată: memoria lungă înseamnă că ρ(k) nu tinde la 0."]}, "dfa-we": {"key": 0, "opts": ["Crește de circa 13 ori, aproape ca √n.", "Crește de circa 168 de ori, proporțional cu n.", "Scade: în blocurile mari dreapta elimină mai mult."]}, "rs-dfa": {"key": 0, "opts": ["Nu: seriile i.i.d. de aceeași lungime dau 0,55.", "Da: orice pantă peste 0,5 înseamnă persistență.", "Da, pentru că DFA confirmă cu 0,49."]}, "gph-we": {"key": 0, "opts": ["Circa 0,28 și conține 0.", "Circa 0,14 (± 1 SE) și nu conține 0.", "Circa 0,28 și nu conține 0, pentru că d̂ = 0,036 > 0."]}, "gph": {"key": 0, "opts": ["SE scade la circa 0,03, iar d̂ crește la circa 0,55.", "SE rămîne 0,071: depinde doar de T.", "d̂ rămîne 0,47: estimatorul nu depinde de m."]}, "lo-we": {"key": 0, "opts": ["7.", "0: ρ̂₁ este negativ, deci nu există memorie scurtă de corectat.", "81, ca lățimea de bandă GPH ⌊√N⌋."]}, "lo": {"key": 0, "opts": ["Circa 60%.", "Circa 5%: nivelul nominal al testului.", "Circa 50%: φ = 0,5."]}, "mc": {"key": 0, "opts": ["GPH: lățimea circa 0,53.", "R/S: este deplasat, deci și cel mai imprecis.", "DFA: eliminarea tendințelor locale pierde informație."]}, "sp-table": {"key": 0, "opts": ["R/S (0,53): banda lui pentru N = 6 714 este [0,51; 0,57].", "GPH (Ĥ = 0,54): d̂ = 0,036 este pozitiv.", "Testul lui Lo (V = 1,56): este peste 1."]}, "nile": {"key": 0, "opts": ["Circa 0,63, în banda i.i.d. pentru N = 100.", "Rămîne 0,78: memoria lungă nu depinde de medie.", "Exact 0,5: fără ruptură, seria este zgomot alb."]}, "spurious": {"key": 0, "opts": ["Circa 0,79.", "0,5: seria nu are memorie, deci GPH este nedeplasat.", "Circa 0,55, cît deplasarea R/S de eșantion mic."]}, "vol-acf": {"key": 0, "opts": ["31, față de circa 12 așteptate din întîmplare.", "0: randamentele sînt necorelate, deci nicio autocorelație nu iese din bandă.", "227, ca pentru |r|."]}, "mk-table": {"key": 0, "opts": ["Doar pentru BET.", "Pentru toate șase: toate au DFA peste 0,7 pentru |r|.", "Pentru niciuna: randamentele nu au memorie lungă pe nicio piață."]}, "markets": {"key": 0, "opts": ["Scade în banda i.i.d., între 0,46 și 0,55.", "Rămîne la fel: cozile groase ale lui |r| nu se schimbă prin permutare.", "Crește: permutarea adună zilele extreme."]}, "rolling": {"key": 0, "opts": ["Circa 24%, niciuna peste.", "Circa 2,5%: atît dă banda din întîmplare.", "Circa 24%, dar peste bandă: crizele creează persistență."]}, "risk-we": {"key": 0, "opts": ["Cu circa 12%.", "Cu 10%: H crește cu 10%.", "Cu 0,05 puncte procentuale: diferența dintre exponenți."]}};
window.CH11_TEXT.ticket = {"title": "Autoevaluare", "id": "check", "store": "sfm-ch11-check-ro", "qs": [{"q": "O estimare R/S pe 1000 de randamente i.i.d. este Ĥ = 0,56. Este o dovadă de memorie lungă?", "key": 0, "opts": [{"t": "Nu: banda i.i.d. pentru N = 1000 este [0,50; 0,63], centrată în jur de 0,562.", "why": "Comparăm Ĥ cu distribuția lui sub ipoteza nulă, pentru același N; R/S este deplasat în sus în eșantioane mici."}, {"t": "Da: 0,56 > 0,5.", "why": "Pragul 0,5 ignoră deplasarea: seriile i.i.d. cu N = 1000 dau în medie 0,562."}, {"t": "Nu se poate spune: R/S nu are o bandă de încredere.", "why": "Banda Monte Carlo se obține pentru orice estimator, aplicîndu-l pe serii simulate de aceeași lungime."}]}, {"q": "GPH dă d̂ = 0,35 pentru |r|. Cît este Ĥ?", "key": 0, "opts": [{"t": "0,85 = d̂ + 0,5: memorie lungă a volatilității.", "why": "d = H − 0,5, deci H = d + 0,5; estimarea privește |r|, deci volatilitatea, nu direcția randamentelor."}, {"t": "0,35: GPH estimează direct H.", "why": "GPH estimează parametrul de memorie d din panta log-periodogramei."}, {"t": "0,15 = 0,5 − d̂: antipersistență.", "why": "Semnul este inversat: d > 0 înseamnă H > 0,5."}]}, {"q": "După o permutare aleatoare a lui |r|, DFA scade de la 0,9 la 0,5. Ce arată acest lucru?", "key": 0, "opts": [{"t": "Memoria provine din ordinea zilelor, nu din distribuție (cozi groase).", "why": "Permutarea păstrează distribuția și distruge doar ordinea; exponentul a scăzut, deci ordinea îl producea."}, {"t": "Cozile groase ale lui |r| produc exponentul 0,9.", "why": "Cozile sînt aceleași după permutare, iar exponentul a scăzut: nu ele îl produceau."}, {"t": "Permutarea a introdus o ruptură structurală.", "why": "O permutare aleatoare distruge rupturile și tendințele, nu le creează."}]}, {"q": "VaR 1% pe o zi este 2,79%. Cu H = 0,45, cît este VaR 1% pe 10 zile?", "key": 0, "opts": [{"t": "7,87% = 2,79 × 10^0,45.", "why": "Riscul se scalează cu h^H; 10^0,45 = 2,82."}, {"t": "8,83%: regula √10.", "why": "Regula √h presupune H = 0,5; cu H = 0,45 factorul este 2,82, nu 3,16."}, {"t": "12,6% = 2,79 × 10 × 0,45.", "why": "Factorul este h la puterea H, nu h înmulțit cu H."}]}]};
window.CH11_TEXT.veil = {"scaling": {"out": true, "charts": ["sc-chart"]}, "fractal-dim": {"out": true, "charts": ["fd-chart", "fd-we-box"]}, "rs-example": {"out": true}, "rs-hurst": {"out": true}, "memory-we": {"out": true}, "fbm": {"out": true}, "fgn": {"out": true}, "arfima": {"out": true, "charts": ["af-chart"]}, "arfima-we": {"out": true, "charts": ["aw-chart", "aw-we-box"]}, "dfa-we": {"out": true, "charts": ["dw-chart"]}, "rs-dfa": {"out": true, "charts": ["rd-chart"]}, "gph-we": {"out": true, "charts": ["gw-we-box"]}, "gph": {"out": true}, "lo-we": {"out": true, "charts": ["lw-chart", "lw-we-box", "lw-tab"]}, "lo": {"out": true, "charts": ["lo-chart"]}, "mc": {"out": true, "charts": ["mc-chart"]}, "sp-table": {"out": true, "charts": ["st-chart", "st-tab"]}, "nile": {"out": true, "charts": ["nl-chart"]}, "spurious": {"out": true, "charts": ["su-chart"]}, "vol-acf": {"out": true, "charts": ["va-chart"]}, "mk-table": {"out": true, "charts": ["mt-chart", "mt-tab"]}, "markets": {"out": true, "charts": ["mk-chart"]}, "rolling": {"out": true, "charts": ["ro-chart"]}, "risk-we": {"out": true, "charts": ["rk-we-box"]}};
