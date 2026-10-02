# Brief revizuire pedagogică — Capitolul 10 FMH

**Data:** 2026-04-28
**Țintă:** versiune predabilă 65–75 slide-uri principale + appendix pentru materialul enciclopedic.

---

## Prompt original

Revizuiește și optimizează pedagogic prezentarea Beamer/PDF „Capitolul 10: Ipoteza Pieței Fractale (FMH)" pentru cursul Statistica Piețelor Financiare.

Obiectivul principal este să transformi materialul într-un curs predabil, coerent și vizual curat, nu să adaugi conținut suplimentar inutil. Cursul actual este prea enciclopedic, are flux logic neuniform și multe slide-uri cu overflow. Vreau o versiune de predare de aproximativ 65–75 slide-uri, cu restul materialului mutat în appendix.

Aplică următoarea restructurare pedagogică:

### 1. Motivație: de ce EMH nu este suficientă
- Introdu anomaliile: cozi grele, volatility clustering, memorie lungă, crah-uri.
- Păstrează legătura cu Capitolul 4.
- Sparge slide-urile dense în 2–3 slide-uri mai aerisite.
- Nu supraîncărca slide-ul cu text + formule + 4 grafice.

### 2. Introdu devreme FMH
- Mută comparația EMH vs FMH aproape de început.
- Explică FMH prin ideea de investitori cu orizonturi multiple.
- Definește clar „horizon diversity" și „horizon collapse".
- Arată de la început de ce FMH este cadrul conceptual al capitolului.

### 3. Fractali și scalare — doar intuiția necesară
- Redu drastic secțiunea de fractali din natură.
- Păstrează doar exemplele care ajută înțelegerea financiară: Koch/Sierpinski, coasta Britaniei, prețul ca fractal.
- Mută în appendix: feriga Barnsley, DLA/fulgere, rețele de râuri, muzică și limbaj, exemple biologice extinse.
- Fiecare exemplu trebuie să se termine cu o frază de legătură către piețe financiare.

### 4. Autosimilaritate și dimensiune fractală
- Păstrează dimensiunea de similaritate, box-counting și relația D = 2 − H.
- Prezintă Hausdorff foarte scurt sau mută detaliile în appendix.
- Dimensiunile alternative trebuie comprimate într-un singur slide de tip „pentru orientare".
- Clarifică diferența dintre autosimilaritate exactă și statistică.

### 5. Exponentul Hurst ca nucleu al capitolului
- Fă din Hurst secțiunea centrală.
- Explică pedagogic: H<0.5 anti-persistent; H=0.5 RW/EMH; H>0.5 persistent/long memory.
- Leagă H de: D = 2 − H; σ(Δ) = σ(1)Δ^H; ACF lungă; risk management.
- Slide nou „Interpretarea greșită a lui H": H>0.5 ≠ automat profit/ineficiență exploatabilă.

### 6. Estimarea lui H: R/S și DFA
- R/S în pași simpli.
- DFA ca alternativă robustă la trenduri.
- Codul Python complet → notebook/appendix; pe slide pseudocod sau ≤15 linii.
- Slide comparativ: R/S vs Lo modified R/S vs DFA vs MFDFA vs GPH/Whittle/wavelet.
- Limitări: eșantioane scurte, structural breaks, GARCH, microstructure noise, trenduri, ferestre.

### 7. fBm, fGn și multifractalitate
- fBm ca generalizare a Brownianului standard.
- fGn ca increment al fBm.
- Doar formulele esențiale.
- MMAR/multifractalitate ca extensie pentru volatilitate/crize/cozi grele.

### 8. Aplicații empirice și risk management
- Studii de caz într-un modul final: 2008, COVID-19, BVB, Bitcoin/crypto.
- Structură comună: context / ce măsurăm / ce arată H/DFA/MFDFA / implicație pentru risc.
- Diferența explicită √T vs T^H.
- Pipeline final vizual.

### 9. Concluzii
- 5 idei-cheie: EMH utilă dar incompletă; piețele scalare; Hurst = măsură operațională; FMH = colaps orizonturi; scalare fractală schimbă VaR/ES.
- Slide final „Ce trebuie să știe studentul la examen".

### Reguli stricte de design
- Zero overflows.
- Max 4 bullet-uri/slide.
- Max o formulă majoră/slide.
- Max un tabel mare/slide.
- Fără text mult + formule + 3–4 grafice.
- Pseudocod în main, cod complet în appendix.
- Fiecare secțiune începe cu „Ce învățăm aici și de ce contează?".
- Fiecare secțiune se termină cu mini-recap 3 puncte.
- Elimină duplicate Beamer overlay (handout mode sau fără \pause).
- Quiz-uri duplicate → un singur slide per secțiune.
- Titluri scurte, active.
- Fără paragrafe lungi.

### Structura finală
1. De la EMH la FMH
2. Fractali și scalare: intuiția
3. Autosimilaritate și dimensiune fractală
4. Exponentul Hurst
5. Estimarea lui H: R/S și DFA
6. fBm și multifractalitate
7. Aplicații empirice: crize, BVB, Bitcoin
8. Implicații pentru managementul riscului
9. Concluzii și întrebări de examen
10. Appendix
