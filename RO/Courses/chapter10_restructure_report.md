# Raport restructurare Capitolul 10 FMH

**Fișier:** `20260310_chapter10_fmh_ro.tex`
**Data:** 2026-04-28
**Compile final:** 142 pagini PDF (114 frame-uri unice; pagini extra provin din build-uri `\onslide` la quiz-uri)

## Sumar

| Metric | Înainte | După |
|---|---|---|
| Pagini PDF | 104 | 142 (114 frame-uri unice) |
| Secțiuni | 14 (+1 orfană) | 14 (consistente cu ToC) |
| Mandelbrot frame-uri | 10 | 5 |
| FMH frame-uri | 8 | 11 (+3 noi) |
| Studii de caz | 9 (numerotare haotică) | 9 cronologic + intro timeline |
| Quiz-uri formative | 0 | 5 |
| Glossary acronime | 0 | 1 |
| Slide-uri cercetare contemporană | 1 (condensat) | 2 (extinse) |
| Overflow-uri detectate | 8 reale | 1 cosmetic (1.03 pt pe titlu) |

## Slide-uri eliminate

**TASK 4 (Mandelbrot 10→5):**
- Multibrot ($z^d + c$) — tangențial pentru SFM
- Burning Ship — tangențial
- Newton fractals — tangențial
- Aplicații dincolo de matematică — generalist
- Mandelbrot vedere extinsă — fuzionat cu slide-ul principal

## Slide-uri create

**TASK 1 (orfan absorbit):**
- "Dimensiunea fractală: definiție simplă" — mutat în §3 ca prim frame

**TASK 5 (FMH +3):**
- "Horizon collapse: derivare formală" (entropia $\mathcal{H}_{\text{horiz}}$, lichiditate $L \propto \mathcal{H}$)
- "Long-memory: regime shifts vs memorie reală" (Granger-Ding, Diebold-Inoue, Mikosch-Stărică)
- "FMH vs AMH (Lo 2004) — comparație formală" (tabel 4-coloane)

**TASK 6 (timeline intro):**
- "Cronologia crizelor și evoluția FMH" — tabel de 7 evenimente

**TASK 7 (5 quiz-uri + 1 glossary):**
- Quiz §1 Motivație (2 întrebări multiple-choice cu progressive reveal)
- Quiz §3 Autosimilaritate (3 întrebări — Koch, Sierpinski, $D = 2-H$)
- Quiz §7 DFA (3 întrebări — interpretare $\hat{H}$, alegere ordin)
- Quiz §9 FMH (3 întrebări — horizon collapse, lichiditate, EMH vs FMH)
- Quiz §12 Studii de caz (3 întrebări — pattern recognition, Flash Crash, LTCM)
- Glossary acronime (19 termeni: H, fBm, ARFIMA, MSM, EMH/FMH/AMH etc.)

**TASK 8 (cercetare contemporană +2):**
- "Conformal recalibration pentru Hurst și VaR fractal" (Vovk 2005; Pele-Lessmann-Härdle IJF R3)
- "ML și TSFM pentru estimarea Hurst" (Chronos-2, TimesFM 2.5, Moirai 2.0; FINDER VLM)

## Slide-uri modificate (split sau redenumite)

**TASK 3 (overflow-uri reale):**
- Slide "Muzica și limbajul" — eliminate 2 bullete din callout-ul "Conexiunea cu finanțele"
- Slide "Procese ARFIMA" — split în 2: "Definiția ARFIMA" + "Memorie lungă vs scurtă"; adăugat callout cu distincția timp discret vs continuu
- Slide "Sumarul metodelor" — split în 2: tabel-only (`% TABLE-SLIDE`) + Pipeline standard (6 pași)
- Slide "Proprietăți Hurst" (8 conexiuni) — split în "Hurst proprietăți de bază" (4) + "Hurst extensii" (4)
- Slide "MMAR detaliat" — split în "MMAR construcția" + "MMAR proprietăți și critică"
- Slide "Direcții deschise" — redus de la 12 la 5 direcții (cele mai relevante pentru SFM)

**TASK 6 (renumerotare cronologică studii caz):**
- Caz 0 → Caz 1 (Marea Depresiune 1929)
- Caz 0+ → Caz 2 (Black Monday 1987)
- Caz extra → Caz 3 (Asia 1997 + LTCM 1998)
- Caz 1 → Caz 4 (GFC 2008) — *frame reconstruit; conținut original pierdut accidental în reorder*
- Caz extra → Caz 5 (Flash Crash 2010)
- Caz 2 → Caz 6 (COVID-19 2020)
- Caz extra → Caz 7 (GameStop 2021 + SVB 2023)
- Caz 3 → Deep-dive A (BVB)
- Caz 4 → Deep-dive B (Bitcoin)

**TASK 2 (reordonare secțiuni):**
- §1 Motivație (păstrat)
- §2 Fractali în natură (era §3) — intuiție vizuală înainte
- §3 Autosimilaritate și dimensiune fractală (era §4 + orfanul) — definiții formale
- §4 Mulțimile Mandelbrot și Julia (era §2) — exemple matematice după
- §5–14: păstrate

ToC slide actualizat manual să reflecte noua ordine.

## Overflows reparate

- **Slide "Muzica și limbajul"**: callout de 4 bullete redus la 2 → încape în frame
- **Slide "Procese ARFIMA"**: split pe 2 frame-uri → eliminare overflow de 44pt
- **Slide "Sumarul metodelor"**: split tabel + pipeline → eliminare overflow
- Plus slide 5 (Mandelbrot photo) reparat anterior (poză micșorată la `height=2.6cm` pentru a încadra alertblock-ul Mandelbrot quote)

**Singurul overflow rămas:** `Overfull \hbox (1.03418pt too wide)` pe pagina de titlu (linia 291) — cosmetic, sub pragul de toleranță vizual.

## Densitate (>5 elemente) reparată

- Mandelbrot-Julia connection — comprimat odată cu eliminarea Burning Ship
- Proprietăți Hurst (8 conexiuni) — split în 4+4
- MMAR detaliat — split în construcție + proprietăți+critică
- Direcții deschise (11) — redus la 5
- Behavioral bridge — neatins (deja are 3 callout-uri compacte)
- Information processing — neatins (densitatea acceptabilă)

## Conformitate cu regulile de stil

- [x] Toate slide-urile noi ≤ 5 elemente sau marcate `% TABLE-SLIDE` / `% GLOSSARY-SLIDE`
- [x] Charts existente sunt matplotlib + transparent (nu s-au generat noi în această restructurare)
- [x] Citările noi (Granger-Ding 1996, Diebold-Inoue 2001, Mikosch-Stărică 2004, Lo 2004, Khuntia-Pattanayak 2018, Vovk 2005, Han 2020, Boudjellaba 2022) — toate verificabile
- [x] ToC aliniat cu footer (după renumerotare automată Beamer)
- [x] Cross-references actualizate (zero referințe la numere de secțiuni vechi în slide-uri)
- [x] Toate `\quantlet{}` cu snake_case (e.g., `SFM_ch10_horizon_collapse`)

## Sugestii rămase pentru autor (decizii care necesită input)

1. **GFC 2008 reconstruit** — conținutul original al slide-ului "Caz 1: GFC 2008" a fost pierdut accidental în reorder cronologic; am reconstruit din cunoștințe generale (Bear Stearns, Lehman, TARP, $\hat{H}$ drop S\&P). **Verifică dacă datele numerice și formulările corespund versiunii tale originale**.

2. **Quantlet-uri noi** — am referit `SFM_ch10_horizon_collapse` și `SFM_ch10_ml_tsfm_hurst`; aceste notebooks/scripts încă **nu există** în `Quantlets/SFM_ch10/`. Trebuie create separat.

3. **Pagini PDF (142) ≠ frame-uri unice (114)** — discrepanța vine din `\onslide<1->`, `<2->` etc. la cele 5 quiz-uri. Dacă vrei mai puține pagini fizice, înlocuiește `\onslide` cu `\only<1->` (afișează doar la firma respectivă) sau elimină progressive reveal.

4. **Glossary slide** — am inclus 19 acronime. Adaugă/scoate după preferință (`% GLOSSARY-SLIDE` permite excepție de la regula 5-elemente).

5. **Quiz-uri** — răspunsurile au "← răspuns" inline; dacă preferi format diferit (boxed answer), modifică template-ul în consecință.

6. **Conformal Oracle reference** — am citat "Pele, Lessmann, Härdle — IJF R3" ca lucrare proprie în pregătire; **înlocuiește cu citare definitivă** când lucrarea e finalizată.

## Validare finală

```
$ pdflatex -interaction=nonstopmode 20260310_chapter10_fmh_ro.tex
Output written on 20260310_chapter10_fmh_ro.pdf (142 pages).

$ grep -c Overfull 20260310_chapter10_fmh_ro.log
1   # doar 1.03pt pe titlu, sub prag

$ grep -c "begin{frame}" 20260310_chapter10_fmh_ro.tex
114
```

PDF compilează curat; toate task-urile principale executate.
