# Examen SPF 2026 — sursă LaTeX (reconstruită)

Sursă editabilă pentru examenul avansat (8 subiecte / 9 întrebări).
Înainte exista doar PDF-ul, fără sursă.

## Fișiere (copia din `exam/2026/`)
Originalul rămîne în `Probleme examen/2026_LaTeX_source/`. Copia de aici folosește preambulul comun
`../sfm_exam_preamble.tex` (siglele ASE și IDA în antet) și figurile generate de `../make_figs.py` în stilul cursului.
- `body_v1.tex`, `body_v2.tex` — conținutul fiecărei variante (cerințe + rezolvări în `\ifrez ... \fi`).
- `_exam_head.tex` — antetul (instrucțiuni, „Valori utile”).
- `20260609_Examen_SPF_Varianta_{1,2}.tex` / `..._Rezolvare.tex` — subiecte / barem.
- Figurile `fig1_density_v{1,2}.pdf` și `fig2_vr_v{1,2}.pdf` se generează cu `python3 exam/make_figs.py --only 2026`.

## Compilare
```bash
python3 exam/make_figs.py --only 2026
cd exam/2026 && for m in 20260609_Examen_SPF_Varianta_1 20260609_Examen_SPF_Varianta_1_Rezolvare \
                         20260609_Examen_SPF_Varianta_2 20260609_Examen_SPF_Varianta_2_Rezolvare; do
  xelatex "$m"; xelatex "$m"
done
```
Necesită XeLaTeX (fontul de sistem Helvetica Neue). Barem-urile (`*_Rezolvare.*`) sînt ignorate de git în `exam/`.

## Corecții față de PDF-urile originale (1 iunie 2026)
1. **Numerotarea întrebărilor** — era greșită (Subiectul V relua „7, 8", apoi coliziuni/salturi).
   Acum este **automată și continuă, 1–18**, prin contorul `\Q` din preambul: bug-ul nu mai poate reapărea
   indiferent cîte subiecte se adaugă/scot.
2. **„Valori utile"** — notația `t^{-1}(0,01)` apărea de două ori, identică. Acum cu grade de libertate distincte:
   `t_5^{-1}(0,01) = -3,365` și `t_6^{-1}(0,01) = -3,143`.
3. Etichete în română: „Figura", „Tabelul".

Conținutul (date, formule, rezolvări, valori numerice) a fost păstrat fidel față de PDF-urile originale,
care au fost verificate numeric (JB, Sharpe, VR, EVT VaR/ES, VaR-Student, AIC/BIC) — toate corecte.

PDF-urile originale sînt salvate în `../_backup_20260601/`.

## Versiune dificultate ridicată (la cerere)
`body_v1.tex` / `body_v2.tex` conțin versiunea **grea**: aceleași 9 subiecte și aceleași date, dar
întrebările au fost reproiectate ca **multi-pas** (lanț de 2–3 calcule) și cu **capcane conceptuale**.
Exemple: deducerea lungimii eșantionului din `Σr`/`r̄`; statistica t a Sharpe-ului (Sharpe „spectaculos"
dar nesemnificativ pe eșantion scurt); descompunerea JB asimetrie vs. kurtoză; momentele α-stabile
(de ce `σ` nu există pentru α<2); testul z al Variance Ratio cu varianța asimptotică; fracțiunea
overnight din volatilitate; testul LR (AIC/BIC/LR în dezacord); raportul ES/VaR vs. limita 1/(1−ξ);
cînd devine ES infinit (ξ≥1); de ce regula √t subestimează; testul z Kupiec; capcana leverage-ului
la decizia integratoare.

Pentru a reveni la versiunea mai simplă, recuperează `body_v1.tex`/`body_v2.tex` din istoric (git)
sau cere regenerarea.

## Revizuire pedagogică (8 iunie 2026)
Ajustări în urma unui review didactic, în ambele variante:
1. **Subiectul VII** — eliminat EVT/GPD (VaR/ES, ξ, ES infinit), considerat prea aproape de nivel master.
   Înlocuit cu **„Volatilitate condiționată și clustering de volatilitate"** (ACF randamente vs. pătrate,
   Ljung–Box, VaR pe regimuri calm/turbulent) — mai central pentru anul 3 și pregătește ARCH/GARCH.
2. **Ipoteze explicite lîngă formulele aproximative:** Raportul Sharpe (testul `t` sub i.i.d. normale, marcat
   ca nerealist), `α`-stabile (varianța teoretică infinită ⇒ `s` nu e volatilitate populațională), fracțiunea
   overnight (aproximație euristică, nu descompunere exactă), backtesting (aproximarea normală a testului de
   **acoperire necondiționată**, nu forma LR clasică Kupiec).
3. **Barem defalcat** la fiecare rezolvare: `0,25p calcul + 0,25p interpretare` (linie `Barem:` în italic).
4. **Instrucțiuni** adăugate în antet (`_exam_head.tex`): răspunsuri de interpretare concise (3–5 fraze),
   se punctează rezultat + interpretare, menționarea ipotezelor la formulele aproximative.
5. **Limbă:** „kurtoză/kurtozei" → **„kurtosis/kurtosisul"** (consecvent cu antetul tabelelor); „Sharpe" ca
   substantiv → **„Raportul Sharpe"** (expresiile matematice `$\mathrm{Sharpe}$` și antetul de tabel rămîn).

Volum: 18 itemi (vezi și punctele 9–10 de mai jos pentru trecerea la 8 subiecte). Toate cele 4 PDF-uri
recompilează curat (XeLaTeX, fără Overfull).

6. **Subiectul IV (Variance Ratio + Hurst)** — refăcut pentru **consistență numerică**: în V1, VR(10) și `H`
   erau reciproc incompatibile (VR(10)=1,10 ⇒ `H≈0,52`, dar se afirma `H=0,62`). Acum V1: VR(10)=**1,15**,
   `H=0,53` (consistente prin `VR(q)≈q^{2H−1}`); tensiunea e pur inferențială (VR nesemnificativ z=1,40 vs.
   IC Hurst care exclude 0,5 la limită). V2: `H=0,56` aliniat la VR(5)=1,20. Formularea „Hurst dă" →
   „Estimarea exponentului Hurst este…".
7. **Figuri reale, nu ilustrative:** densitatea e acum $\alpha$-stabilă reală **leptokurtică** (scală $\gamma$
   aleasă astfel încît să fie peste normală la vîrf, cu cozi mai groase — nu proxy Student-$t$), iar VR are banda
   95% calculată din $V(q)$, cu estimări care cad în/în afara benzii conform deciziei statistice. Toate graficele
   au legenda sub axă (orizontal). Figuri distincte per variantă, fără valori/concluzii afișate pe grafic.
8. **VaR** — convenția unică e **VaR 1%** (probabilitate de coadă); eliminat „VaR 99%". Nivelurile din §II:
   „supraestimează la 5% / subestimează la 0,1%".
9. **Eliminat Subiectul AIC/BIC/LR** (nepotrivit la acest nivel). Renumerotare I–VIII; punctul eliberat
   **redistribuit**: §VII (VaR/Kupiec) → 1,5p (întrebare nouă: VaR normal vs. Student-$t$ în lei, folosind datele
   deja date) și §VIII (Decizie) → 1p (întrebare nouă: kurtosis ⇒ drawdown viitor vs. MDD istoric). Total
   neschimbat: 9p subiecte + 1p oficiu = 10p.
10. **Verificare limbă (nativ):** „statistic indistinct de zero" → „nu se distinge statistic de zero";
   „mediază performanța" → „rezumă performanța medie"; „⇒ respinge" → „⇒ se respinge"; termeni EN în text RO
   („intraday"→„intra-zi", „opening effects"→„efecte de deschidere", „drift"→„tendință"); „testați/decideți
   **la 5%**" → „**la nivelul de semnificație de 5%**".

11. **Redus la 9 subpuncte** (din 18), pentru încadrarea în 2 ore: cîte o întrebare per subiect (cea mai
   reprezentativă) + a 2-a la §I. Fiecare **1p** (barem defalcat **0,5p calcul + 0,5p interpretare**); §I = 2p,
   restul 1p → 9p + 1p oficiu = 10p. §IV retitulat „testul Variance Ratio" (întrebarea Hurst eliminată),
   §VII „Backtesting VaR (testul Kupiec)". Întrebări păstrate: §I (returns; Sharpe+t), §II (JB), §III
   ($\alpha$-stabile), §IV (VR z-test), §V (overnight/range-based), §VI (clustering), §VII (Kupiec), §VIII
   (decizie integratoare).
12. **Figuri și tabele fixate** lîngă textul problemei: pachet `float` + plasare `[H]` (nu mai plutesc).

## Revizuire 8 iunie 2026 (n=30 + îmbogățire din examenul real 2025)
13. **Subiectul I — `n=5 → n=30` de zile.** Mediile zilnice (`r̄`, `s`) rămîn; cumulatul scalează: V1
   `Σr=23,52%` (simplu `26,52%`), V2 `Σr=34,95%` (simplu `41,83%`). Sharpe anualizat neschimbat (nu depinde de
   `n`); statistica `t = SR_zi·√30` urcă (V1 `0,56→1,37`, V2 `0,52→1,28`) dar **rămîne sub 1,96** — capcana
   „Sharpe spectaculos ≠ semnificativ" se păstrează. Eliminate, la cerere, parantezele din §I (formula de
   anualizare) și §VIII (referința la forma LR clasică Kupiec).
14. **Trei elemente noi, inspirate din `20250608 Examen SPF.docx`:**
   - **§IV (nou) — Distribuția Pareto:** regresie log–log a cozii stîngi → exponent `α`, constanta `c`,
     probabilitate de depășire și **interval mediu de repetare (zile/ani)**. V1 `α=3`, V2 `α=2,5`.
   - **§VII — a 2-a întrebare AR(2):** predicție de preț cu `P_t=aP_{t-1}+bP_{t-2}`, comparație cu random walk;
     `a+b=1` ⇒ rădăcină unitară (capcană: nestaționar în nivel, dar randamente predictibile ⇒ contrazice EMH slabă).
   - **§VIII — Backtesting VaR pe 3 metode** (Gaussiană / α-stabile / GARCH-t), `z` de acoperire necondiționată,
     recomandare (α-stabile cel mai bine calibrat; gaussiana respinsă — cozi subțiri).
15. **Rebalansare barem (total neschimbat 9p + 1p oficiu = 10p):** §II (JB) și §VI (estimatori vol.) reduse la
   **0,5p**; cele două elemente noi adăugate la 1p (Pareto) și 0,5p (AR(2)). `\emergencystretch=3em` în preambul
   pentru a absorbi depășirile minore de linie.

16. **Calibrare dificultate pentru anul 3 (de la ~8/10 la ~6,5–7/10).** În urma unui review didactic, examenul
   era prea **dens** (11 cerințe + multă interpretare în 2 ore) și avea două teme de coadă suprapuse. Ajustări:
   - **Eliminat §IV Distribuția Pareto** (a doua temă de coadă, după ce EVT/GPD fusese deja scos) și **întrebarea
     AR(2)** din §VII — reducere de densitate.
   - **§II (JB) și §V (estimatori) readuse la 1p** (cerințe mai puține, dar mai pline).
   - **Interpretări scurtate/ghidate:** §III ($\alpha$-stabile) — formulare mai ghidată („dacă media și varianța
     există"), fără lanțul lung despre estimatorul instabil; §VIII (decizie) — eliminată partea ambiguă despre
     **levier**, rămîne Sharpe/$|$MDD$|$ + kurtosis.
   - **Corectat bug de mărime font** în `_exam_head.tex` (`\small` se scurgea în corp prin `\normalfont` care nu
     resetează dimensiunea) — corpul subiectelor revine la 10,5pt, uniform cu baremul.
   - Termen VaR: „depășiri" → **„excepții"** (terminologie standard de backtesting).

17. **Eliminat testul $t$ al raportului Sharpe** (§I Q2): rămîne calculul Sharpe-ului anualizat + o discuție
   \emph{calitativă} (de ce un Sharpe „spectaculos" pe 30 de zile e imprecis; ipoteza i.i.d.\ normală nerealistă),
   fără statistica $t=\mathrm{SR}_{\text{zi}}\sqrt{n}$ și fără comparația cu 1,96. Curățate din „Valori utile"
   constantele rămase orfane ($\sqrt{30}$, $4^3$, $4^{2,5}$, $e^{-1,83}$, $e^{-1,14}$).

Structură finală: **8 subiecte / 9 întrebări**, 9p + 1p oficiu = 10p. Conținut: §I (randamente; Sharpe calitativ),
§II (JB), §III ($\alpha$-stabile, ghidat), §IV (Variance Ratio), §V (estimatori range-based), §VI (clustering),
§VII (Backtesting VaR pe 3 metode), §VIII (decizie Sharpe/MDD/kurtosis). Toate cele 4 PDF-uri recompilează curat,
fără Overfull (**exam 3 pagini**, barem 4 pagini).

18. **Eliminată pagina aproape goală** din subiecte: enunțul ultimului subiect spilui pe pagina 4. Strînsă
   spațierea (`parskip` 0,55→0,42 ex, antet de subiect mai compact), micșorate figurile (0,6/0,62→0,5/0,52) și
   lărgită zona de text (margini 2,1→1,9 cm, top 2,7→2,4 cm). Subiectele încap acum curat pe **3 pagini**.

19. **Reducere de calcul la §IV (Variance Ratio) și §VII (Backtesting VaR).**
   - **§IV:** se dă direct abaterea standard sub $H_0$ ($\sqrt{V(10)}=0{,}107$ / $\sqrt{V(5)}=0{,}069$), eliminînd
     formula varianței asimptotice $V(q)=2(2q-1)(q-1)/(3qT)$; studentul calculează doar VR și $z$ (două împărțiri).
   - **§VII:** în loc de $z$ pentru fiecare din 3 metode, studentul derivă o singură dată \emph{intervalul de
     acceptare} ($np\pm z_{0,025}\sqrt{np(1-p)}=40\pm12{,}3\approx[28,52]$) și compară numerele de excepții cu el.
     Eliminat $\sqrt{39{,}6}$ din „Valori utile" (nemaifolosit). Pentru a nu deveni trivial, s-a re-adăugat partea
     \emph{conceptuală}: ce \emph{nu} detectează testul de acoperire necondiționată (independența/gruparea
     excepțiilor $\Rightarrow$ testul Christoffersen). Calcul redus, dar dificultate menținută prin interpretare.
