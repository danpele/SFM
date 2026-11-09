<!--
Textul de curs al versiunii web, capitolul 16 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie → Autor (an), [@Cheie] → (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 16 --hashes
-->

## @intro

@epigraph Cont2001 | Empirical properties of asset returns: stylized facts and statistical issues
> Our description emphasizes properties common to a wide variety of markets and instruments. We then show how these statistical properties invalidate many of the common statistical approaches used to study financial data sets …

@text
Capitolele 0–15 au construit, pas cu pas, un lanț de la prețurile observate la măsuri de risc pe care le putem testa: randamente calculate corect, distribuții cu cozi groase (*heavy tails*), dependență în timp, modele de volatilitate și, la capăt, VaR (*value at risk*) și ES (*expected shortfall*) verificate prin backtesting. Acest capitol de recapitulare parcurge din nou lanțul, de data aceasta cu trei serii pe aceeași fereastră de timp: indicele american S&P 500, indicele BET (*Bucharest Exchange Trading*) al Bursei de Valori București și Bitcoin, între 5 ianuarie 2015 și 18 septembrie 2026.

Cele trei serii sînt foarte diferite ca nivel de risc: volatilitatea anuală este 17,7% pentru S&P 500, 15,5% pentru BET și 66,7% pentru Bitcoin. Faptele stilizate sînt însă aceleași: asimetrie negativă, exces de boltire peste 10, tail index-uri Hill între 2 și 3, randamente aproape necorelate și randamente absolute puternic autocorelate. Un VaR 1% calculat cu distribuția Normală este prea mic pentru toate trei, cu 18% pînă la 25%. Aceste regularități comune sînt motivul pentru care aceleași instrumente statistice funcționează pe o piață emergentă, pe cea mai mare piață de acțiuni din lume și pe o piață cripto.

Capitolul are trei părți. Prima reia fiecare capitol prin formula principală, faptul empiric principal și greșeala cea mai frecventă, apoi strînge testele și modelele într-o sinteză a metodelor. A doua prezintă evaluarea scrisă, opt probleme rezolvate complet și proiectul de echipă. A treia formulează o întrebare deschisă despre piața din România, la care metodele cursului pot contribui, și arată ce poate face un asistent AI (*artificial intelligence*) într-un astfel de proiect și ce trebuie verificat.

@chart sfm_ch16_three_series
Sus: valoarea a 100 de unități investite la începutul lui 2015 în S&P 500 (albastru închis), BET (roșu) și Bitcoin (auriu), pe scală logaritmică. Jos: drawdown-ul fiecărei serii, în procente, față de maximul atins pînă la acea dată.

@goals
- **Situați** fiecare capitol pe drumul de la prețuri la măsurile de risc și numiți întrebarea la care răspunde.
- **Reamintiți** formula principală, faptul empiric principal și greșeala cea mai frecventă din fiecare capitol.
- **Alegeți** testul sau modelul potrivit pentru o întrebare dată despre o serie de randamente.
- **Rezolvați** pas cu pas probleme recapitulative și interpretați rezultatul în trei pînă la cinci fraze.
- **Planificați** proiectul de echipă, susținerea orală și declararea utilizării AI.

@plan
- Harta cursului | Legăturile dintre capitole, firul principal și cele trei serii comparate într-un singur tabel și în două grafice.
- Capitolele 0–3: date, randamente și distribuții | Randamente, anualizare, drawdown, raportul Sharpe, momente, testul Jarque–Bera și legile α-stabile.
- Capitolele 4–7: probabilitate, cozi, alegerea modelului, eficiență | Portofolii, Monte Carlo, GBM, estimatorul Hill, EVT, criterii de selecție și testele raportului varianțelor.
- Capitolele 8–11: volatilitate, risc și memorie | EWMA, GARCH, VaR și ES cu backtesting, exponentul Hurst.
- Capitolele 12–15: aplicații | Scoring de credit, învățare automată, active cripto și risc sistemic.
- Sinteza metodelor | Ce test sau model răspunde la ce întrebare, convențiile cursului și erorile metodologice frecvente.
- Examenul | Formatul, conținutul, criteriile de notare și opt probleme rezolvate.
- Proiectul de echipă | Livrabile, criterii, susținerea orală și fișierul AI_USE.md.
- Instrumente AI pentru un proiect de cercetare | O întrebare deschisă despre BET, bucla de cercetare, ajutorul și verificările AI, o idee de proiect.
- Rezumat | Ideile principale și întrebări de autoevaluare.

# Harta cursului

Secțiunea arată cursul ca întreg: cum se leagă capitolele, care este firul principal care duce de la prețuri la o măsură de risc testabilă și ce dau toate instrumentele cursului aplicate simultan pe S&P 500, BET și Bitcoin. Tabelul și cele două grafice de aici sînt rezultatele de referință ale capitolului; secțiunile următoare reiau fiecare capitol în parte.

## Harta cursului: Capitolele 0–15 {h=17555966}

Diagrama așază cele șaisprezece capitole pe trei rînduri. Primul rînd, „Date și distribuții”, merge de la piețe și date ([Capitolul 0](ch0_ro.html)) la randamente și indicatori, distribuții clasice și fapte stilizate, legi α-stabile, cozi groase și teoria valorilor extreme (EVT, *extreme value theory*) și selecția modelului. Al doilea rînd, „Timp, volatilitate și risc”, pornește de la probabilitate și procese stocastice, trece prin eficiența pieței, memoria lungă, estimatorii de volatilitate și modelele ARCH–GARCH și ajunge la VaR, ES și backtesting. Al treilea rînd, „Aplicații”, conține modelele de scoring, învățarea automată, activele cripto și riscul sistemic.

Săgețile roșii marchează firul principal: randamentele ([Capitolul 1](ch1_ro.html)) și distribuția lor ([Capitolul 2](ch2_ro.html)) duc, prin cozi groase și prin volatilitate, la o măsură de risc pe care o putem testa ([Capitolul 10](ch10_ro.html)). Al doilea rînd separă două întrebări diferite despre dependența în timp: semnul randamentelor este greu de anticipat ([Capitolele 7](ch7_ro.html) și [11](ch11_ro.html)), în timp ce mărimea lor este previzibilă ([Capitolele 8](ch8_ro.html) și [9](ch9_ro.html)).

Rîndul aplicațiilor nu introduce o statistică nouă, ci reutilizează instrumentele primelor două rînduri. Scoring-ul de credit pleacă din probabilitate, învățarea automată pleacă din modelele de volatilitate, iar activele cripto și riscul sistemic pleacă din măsurile de risc ale Capitolului 10.

## Structura cursului {h=10a2a077}

Lanțul are patru verigi. Prima sînt datele: prețuri dintr-o singură sursă, ajustate pentru dividende și split-uri, pe calendarul corect al pieței. Din ele se calculează randamentele logaritmice zilnice, în procente,

$$ r_t = 100\,\ln\frac{P_t}{P_{t-1}}, $$

unde $P_t$ este prețul de închidere din ziua $t$. A doua verigă este distribuția randamentelor ([Capitolele 2](ch2_ro.html)–[6](ch6_ro.html)): ea are cozi groase, așa că distribuția Normală este înlocuită de Student-$t$, de legile stabile sau de EVT, iar alegerea dintre modele se face cu verosimilitatea, cu AIC (*Akaike information criterion*) și cu teste concentrate pe cozi.

A treia verigă este dependența în timp ([Capitolele 7](ch7_ro.html)–[11](ch11_ro.html)). Pentru medie, raportul varianțelor și exponentul Hurst arată că semnul randamentului de mîine este aproape imprevizibil. Pentru varianță, EWMA (*exponentially weighted moving average*) și GARCH (*generalised autoregressive conditional heteroskedasticity*) arată că mărimea lui este previzibilă. A patra verigă este riscul ([Capitolul 10](ch10_ro.html)): VaR 1% și ES 2,5%, însoțite întotdeauna de backtesting.

Aplicațiile din [Capitolele 12](ch12_ro.html)–[15](ch15_ro.html) refolosesc același lanț: date, distribuție, dependență, risc, evaluare. Un model de scoring, o prognoză obținută prin învățare automată sau o măsură a riscului sistemic se judecă tot prin incertitudinea estimărilor și prin performanța în afara eșantionului (*out-of-sample*).

## S&P 500, BET și Bitcoin în aceeași perioadă {h=a3f449c5}

Graficul are două panouri cu aceeași axă orizontală, din 2015 pînă în 2026. Panoul de sus arată valoarea a 100 de unități investite la 2 ianuarie 2015 în S&P 500, BET și Bitcoin, pe o scală logaritmică cu reperele 100, 1 000 și 10 000. Scala logaritmică face comparabile variațiile procentuale: aceeași distanță verticală înseamnă aceeași creștere relativă, indiferent de nivel. Pe ea, S&P 500 și BET urcă pe linii apropiate, iar Bitcoin crește de peste 250 de ori, cu oscilații mult mai ample.

Panoul de jos arată drawdown-ul fiecărei serii, adică distanța procentuală față de cel mai mare preț atins pînă atunci:

$$ D_t = \frac{P_t}{\max_{s \le t} P_s} - 1 \le 0. $$

Drawdown-ul maxim (MDD, *maximum drawdown*) este $-33{,}9\%$ pentru S&P 500 și $-31{,}1\%$ pentru BET, ambele în martie 2020, la începutul pandemiei. Pentru Bitcoin este $-83{,}4\%$, atins în decembrie 2018, iar panoul arată o a doua scădere de amploare comparabilă în 2022.

Graficul reunește două idei din [Capitolul 0](ch0_ro.html). Un cîștig mare pe întreaga perioadă poate coexista cu pierderi intermediare care ar fi fost greu de suportat, iar riscul unei investiții nu se descrie doar prin volatilitate: drawdown-ul spune cît a pierdut efectiv un investitor care a cumpărat la vîrf.

## Tabelul de sinteză al cursului {h=f8100b49}

Tabelul aplică instrumentele cursului pe randamentele logaritmice zilnice ale celor trei serii, fiecare pe propriul calendar, de la primul randament (5 ianuarie 2015 pentru S&P 500, 6 ianuarie pentru BET, 2 ianuarie pentru Bitcoin) pînă la 18 septembrie 2026: 2 943 de observații pentru S&P 500 (252 pe an), 2 925 pentru BET (250 pe an) și 4 278 pentru Bitcoin (365 pe an). Primele rînduri descriu performanța: randamentul logaritmic mediu este 11,2% pe an pentru S&P 500, 13,2% pentru BET și 47,4% pentru Bitcoin, iar volatilitățile sînt 17,7%, 15,5% și 66,7%. Raportul Sharpe, calculat cu o rată fără risc egală cu zero, este 0,72, 0,93 și 1,04, fiecare cu o eroare standard (SE, *standard error*) de 0,33–0,36.

Rîndurile din mijloc descriu distribuția și dependența. Asimetria este negativă pentru toate trei serii (−0,64, −1,41 și −0,73), excesul de boltire este 15,8, 17,8 și 12,2, iar tail index-ul Hill al pierderilor, estimat cu 2,5% dintre observații în coadă, este 2,75, 2,19 și 2,67. Statistica Ljung–Box $Q(10)$ este 236, 55 și 21 pentru randamente, dar 3 166, 701 și 211 pentru pătratele lor, față de valoarea critică 18,31 a distribuției $\chi^2(10)$ la 5%. Raportul varianțelor pe cinci zile, VR(5), este 0,84, 1,17 și 0,99, cu statistici robuste $Z^*(5)$ de −1,34, 1,93 și −0,22. Persistența GARCH(1,1)-$t$, $\alpha + \beta$, este 0,993, 0,925 și 1,000.

Ultimele rînduri dau riscul și memoria. VaR 1% istoric este 3,29% pentru S&P 500, 2,72% pentru BET și 10,60% pentru Bitcoin, iar VaR 1% Normal este 2,55%, 2,23% și 7,99%. ES 2,5% istoric este 3,53%, 3,21% și 10,93%. Exponentul Hurst estimat prin metoda R/S (*rescaled range*) este 0,53, 0,59 și 0,59 pentru randamente și 0,84, 0,75 și 0,78 pentru randamentele absolute.

## Interpretarea tabelului {h=7c6bc0ac}

Prima lectură privește randamentul și riscul. Volatilitatea Bitcoin este de 3,8 ori mai mare decît a S&P 500 și de 4,3 ori mai mare decît a BET. Rapoartele Sharpe par să ordoneze cele trei investiții (1,04 pentru Bitcoin, 0,93 pentru BET, 0,72 pentru S&P 500), dar diferențele sînt mici față de erorile lor standard: un interval de încredere de 95% în jurul fiecărui raport acoperă și valorile celorlalte două. Pe aproape doisprezece ani de date, nu putem spune care dintre ele a avut cel mai bun raport între randament și risc.

A doua lectură privește distribuția. Asimetria negativă, excesul de boltire peste 10 și tail index-urile între 2 și 3 spun același lucru: pierderile mari sînt mult mai frecvente decît prevede distribuția Normală. Consecința practică se vede în rîndul VaR: VaR 1% Normal este mai mic decît cel istoric cu 22% pentru S&P 500, cu 18% pentru BET și cu 25% pentru Bitcoin. Un model Normal subestimează sistematic pierderea din ziua rea.

A treia lectură privește dependența. Statistica Ljung–Box pe $r_t^2$ este de 10 pînă la 13 ori mai mare decît pe $r_t$: volatilitatea, nu randamentul, are structură în timp. Testul robust al raportului varianțelor nu respinge mersul aleator la 5% pentru niciuna dintre serii (pentru BET, $Z^*(5) = 1{,}93$ este chiar sub pragul 1,96), iar exponentul Hurst este mult peste 0,5 doar pentru randamentele absolute. Persistența volatilității este aproape de 1; pentru Bitcoin este exact 1, adică un model IGARCH (*integrated GARCH*), în care un șoc de volatilitate nu se stinge.

## Faptele stilizate: ACF a randamentelor și a randamentelor absolute {h=0f18c6fa}

Graficul are trei panouri, pentru S&P 500, BET și Bitcoin. Pe axa orizontală este lagul, de la 1 la 50 de zile, iar pe verticală este autocorelația. Barele albastre închise sînt autocorelațiile randamentelor $r_t$, barele roșii sînt autocorelațiile randamentelor absolute $|r_t|$, iar banda colorată din jurul lui zero este intervalul $\pm 1{,}96/\sqrt{T}$, în interiorul căruia o autocorelație nu diferă semnificativ de zero în ipoteza de independență.

Autocorelațiile randamentelor sînt mici și se află în mare parte în bandă; excepțiile cele mai vizibile sînt la S&P 500, unde primul lag este negativ. Autocorelațiile randamentelor absolute sînt pozitive pe toate cele 50 de laguri pentru S&P 500 și Bitcoin și pe 49 dintre ele pentru BET (la lagul 43 este −0,001), iar ele scad lent: la primul lag ele sînt 0,36 pentru S&P 500, 0,29 pentru BET și 0,22 pentru Bitcoin. Aceasta este forma grafică a volatility clustering: o zi agitată este urmată, cu o probabilitate mai mare, de alte zile agitate.

Lecția graficului este că necorelat nu înseamnă independent. Absența autocorelației în $r_t$ nu permite concluzia că randamentele sînt independente, pentru că o funcție a lor, $|r_t|$, este puternic autocorelată. Această observație, una dintre faptele stilizate descrise de @Cont2001, justifică modelarea separată a varianței condiționate în [Capitolele 8](ch8_ro.html) și [9](ch9_ro.html).

## @end

Pe aceeași fereastră, S&P 500, BET și Bitcoin diferă prin nivelul riscului, dar au aceleași fapte stilizate: asimetrie negativă, cozi groase cu tail index între 2 și 3, randamente aproape necorelate și volatilitate persistentă. Distribuția Normală subestimează VaR 1% al tuturor celor trei serii, iar diferențele dintre rapoartele Sharpe nu sînt semnificative statistic.

# Capitolele 0–3: date, randamente și distribuții

Primele patru capitole stabilesc obiectul de studiu: cum se obțin date curate, cum se transformă prețurile în randamente, cum se rezumă performanța unei investiții și ce formă are distribuția randamentelor. Fiecare capitol este reluat în două slide-uri: formulele principale cu notațiile lor, apoi faptele empirice și greșelile frecvente.

## Capitolul 0: introducere (1/2) {h=2862abb9}

Prima formulă a cursului este anualizarea volatilității. Dacă $\sigma$ este abaterea standard a randamentelor zilnice și $A$ este numărul de observații pe an, volatilitatea anualizată este $\sigma_{\text{an}} = \sqrt{A}\,\sigma$. Regula rădăcinii pătrate vine din faptul că varianța unei sume de randamente necorelate este suma varianțelor, deci varianța anuală este de $A$ ori cea zilnică. Valoarea lui $A$ depinde de piață: circa 252 pentru acțiuni, care se tranzacționează doar în zilele lucrătoare, și 365 pentru activele cripto, care se tranzacționează în fiecare zi.

A doua formulă este drawdown-ul, $D_t = P_t/\max_{s \le t}P_s - 1$, unde $P_t$ este prețul din ziua $t$, iar $\max_{s \le t}P_s$ este cel mai mare preț atins pînă în acea zi. Prin construcție $D_t \le 0$: drawdown-ul este zero la un nou maxim și negativ în rest. Drawdown-ul maxim, $\mathrm{MDD} = \min_t D_t$, este cea mai mare pierdere suferită de un investitor care a cumpărat la vîrf și a vîndut la minim.

Cele două măsuri se completează. Volatilitatea descrie mărimea tipică a variațiilor zilnice, fără să țină seama de ordinea lor, în timp ce drawdown-ul depinde de succesiunea randamentelor: o serie de pierderi consecutive produce un drawdown mare chiar dacă fiecare pierdere, luată separat, este modestă.

## Capitolul 0: introducere (2/2) {h=85bc962e}

Din 2015, volatilitatea anuală a S&P 500 este 17,7%, iar a Bitcoin este 66,7%, aproape de patru ori mai mare. Cel mai sever episod din istoria BET a fost criza financiară globală: indicele a pierdut 82,5% între 24 iulie 2007 și 25 februarie 2009 și a revenit la vîrful din 2007 abia pe 16 martie 2021, după mai bine de treisprezece ani. Durata revenirii este o dimensiune a riscului pe care volatilitatea nu o surprinde.

Greșelile frecvente ale acestui capitol privesc datele. Prețurile luate din mai multe surse pot avea ore de închidere, ajustări sau calendare diferite, iar o serie compusă din ele produce randamente false la punctele de îmbinare. Prețurile neajustate ale acțiunilor care plătesc dividende sau fac split-uri arată o scădere bruscă în ziua ajustării, deși investitorul nu a pierdut nimic.

A doua greșeală privește verificarea. O cifră sau o referință bibliografică obținută cu un asistent AI trebuie recalculată, respectiv deschisă la sursă, înainte de a fi folosită: asistenții AI produc cu ușurință cifre plauzibile și referințe care nu există.

## Capitolul 1: date, randamente și indicatori (1/2) {h=dace0a8f}

Randamentul simplu este $R_t = P_t/P_{t-1} - 1$, iar randamentul logaritmic este $r_t = \ln(1 + R_t)$. Cele două au proprietăți de agregare diferite. Randamentele logaritmice se adună în timp, pentru că logaritmul unui produs de factori de creștere este suma logaritmilor. Randamentele simple se adună între active: randamentul unui portofoliu este media ponderată a randamentelor simple ale componentelor.

Performanța pe termen lung se rezumă prin rata anuală compusă de creștere (CAGR, *compound annual growth rate*), $\mathrm{CAGR} = (P_T/P_0)^{1/Y} - 1$, unde $Y$ este numărul de ani. Raportul Sharpe, $\mathrm{SR} = (\mu - r_f)/\sigma$, împarte randamentul în exces anual la volatilitatea anuală, iar eroarea lui standard este aproximativ

$$ \mathrm{SE}(\widehat{\mathrm{SR}}) \approx \sqrt{\frac{1 + \mathrm{SR}^2/2}{Y}}, $$

deci depinde de numărul de ani, nu de numărul de observații zilnice. Cu zece ani de date, eroarea standard este în jur de 0,3, de aceea rapoartele Sharpe ale unor investiții obișnuite se pot rareori distinge statistic.

Volatility drag este diferența dintre media aritmetică a randamentelor și rata de creștere compusă, aproximativ $\sigma^2/2$ pe an. El explică de ce o investiție volatilă crește mai încet decît sugerează randamentul ei mediu: o pierdere de 50% urmată de un cîștig de 50% lasă investitorul cu 75% din capital.

## Capitolul 1: date, randamente și indicatori (2/2) {h=1a8aaca7}

Pentru BET din 2015, CAGR este 14,1%, volatilitatea anuală este 15,5%, iar raportul Sharpe este 0,93, cu un interval de încredere (CI, *confidence interval*) de 95% între 0,24 și 1,61. Intervalul este larg: cu aproape doisprezece ani de date, raportul Sharpe al BET este pozitiv cu certitudine statistică, dar valoarea lui exactă rămîne incertă.

Bitcoin ilustrează volatility drag. Media aritmetică a randamentelor lui anuale este 69,5%, dar CAGR este doar 60,6%. Diferența de circa nouă puncte procentuale este costul volatilității mari: un investitor care a păstrat Bitcoin pe toată perioada a obținut rata compusă, nu media aritmetică.

Greșelile frecvente decurg din aceste definiții. Adunarea randamentelor simple în timp, folosirea mediei aritmetice în locul CAGR pentru performanța realizată, anualizarea volatilității Bitcoin cu 252 de zile în loc de 365 și raportarea unui raport Sharpe fără eroarea lui standard duc, fiecare, la concluzii greșite despre performanță.

## Capitolul 2: distribuții clasice și fapte stilizate (1/2) {h=be046fb8}

Forma distribuției se rezumă prin momentele centrate $m_k = \frac1n\sum_t (r_t - \bar r)^k$. Asimetria este $S = m_3/m_2^{3/2}$, iar excesul de boltire este $K = m_4/m_2^2 - 3$; ambele sînt zero pentru distribuția Normală. Testul Jarque–Bera (JB) combină cele două abateri de la normalitate [@JarqueBera1980]:

$$ \mathrm{JB} = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right) \sim \chi^2(2). $$

Statistica urmează distribuția $\chi^2(2)$ în ipoteza de normalitate.

Pentru cozi groase, alternativa clasică este distribuția Student-$t$ cu $\nu$ grade de libertate. Excesul ei de boltire este $6/(\nu - 4)$ pentru $\nu > 4$, deci un $\nu$ mic înseamnă cozi groase, iar pentru $\nu \le 4$ momentul de ordinul patru nu există. Dependența în timp se testează cu statistica Ljung–Box

$$ Q(m) = n(n+2)\sum_{h=1}^{m}\frac{\hat\rho(h)^2}{n - h}, $$

unde $\hat\rho(h)$ este autocorelația de selecție la lagul $h$, iar $m$ este numărul de laguri; în ipoteza de absență a autocorelației, $Q(m)$ urmează aproximativ o distribuție $\chi^2(m)$.

## Capitolul 2: distribuții clasice și fapte stilizate (2/2) {h=3d438a3c}

Pentru BET din 2010, asimetria este −0,98, excesul de boltire este 17,9, iar statistica Jarque–Bera este 56 762, cu mult peste orice valoare critică. Distribuția Student-$t$ estimată are $\hat\nu = 2{,}96$ grade de libertate, sub pragul de 4, ceea ce înseamnă că, dacă modelul este corect, momentul de ordinul patru nu este finit.

Autocorelația de ordinul întîi a randamentelor BET este 0,03, aproape zero, dar a randamentelor absolute este 0,34. Cele două cifre rezumă două dintre cele șase fapte stilizate descrise de @Cont2001: absența autocorelației randamentelor și volatility clustering.

Prima greșeală frecventă este confuzia dintre coeficientul de boltire, $K + 3$, și excesul de boltire, $K$: distribuția Normală are boltirea 3 și excesul 0. A doua este concluzia că randamentele sînt independente pentru că funcția de autocorelație (ACF, *autocorrelation function*) a lui $r_t$ nu este semnificativă; independența cere ca și funcțiile neliniare ale randamentelor să fie necorelate.

## Capitolul 3: distribuții α-stabile (1/2) {h=2c93b5e1}

O distribuție este stabilă dacă suma unor copii independente are aceeași formă ca fiecare termen, după o schimbare de scală și de poziție:

$$ X_1 + \dots + X_n \overset{d}{=} n^{1/\alpha}X + d_n, $$

unde $\overset{d}{=}$ înseamnă egalitate în distribuție, $d_n$ este o translație, iar $\alpha \in (0, 2]$ este indicele de stabilitate. Pentru $\alpha = 2$ se obține distribuția Normală. Pentru $\alpha < 2$, cozile descresc ca o putere, $P(X > x) \sim c\,x^{-\alpha}$, iar momentul de ordin $p$ există doar pentru $p < \alpha$; în particular, varianța este infinită.

Familia are patru parametri: $\alpha$ pentru cozi, $\beta$ pentru asimetrie, $\gamma$ pentru scală și $\delta$ pentru poziție [@Nolan2020]. Aceiași parametri se pot defini în mai multe parametrizări, dintre care cele mai folosite sînt S0 și S1. Ele dau același $\alpha$, $\beta$ și $\gamma$, dar parametrul de poziție $\delta$ are sensuri diferite, de aceea parametrizarea trebuie precizată de fiecare dată.

## Capitolul 3: distribuții α-stabile (2/2) {h=f7470d63}

Estimate prin verosimilitate maximă pe datele zilnice din 2000, legile stabile dau $\hat\alpha = 1{,}49$ pentru BET, $1{,}54$ pentru S&P 500 și $1{,}41$ pentru Bitcoin. Valorile sînt apropiate de cele găsite de @Mandelbrot1963 pentru prețurile bumbacului și ar sugera că varianța randamentelor este infinită.

Alte dovezi contrazic această concluzie. Varianța de selecție calculată pe eșantioane din ce în ce mai lungi se stabilizează, ceea ce nu s-ar întîmpla dacă varianța ar fi infinită. Agregarea dă un al doilea test: dacă randamentele zilnice ar fi stabile, $\hat\alpha$ ar rămîne același pe randamentele lunare, iar dacă varianța este finită, el urcă spre 2. Pentru S&P 500 și Bitcoin, $\hat\alpha$ crește cu orizontul (pînă la $1{,}82$ pe date lunare pentru S&P 500); BET este excepția, cu $\hat\alpha$ în jur de 1,50 și pe date lunare, pe un eșantion lunar scurt, cu o estimare imprecisă. Concluzia de ansamblu este cozi groase cu varianță finită: legea stabilă descrie bine corpul distribuției, nu și coada extremă.

Greșelile frecvente decurg din proprietățile familiei. Abaterea standard nu poate fi folosită ca măsură a scalei atunci cînd $\alpha < 2$, pentru că varianța nu există; scala este $\gamma$. Iar parametrul $\delta$ estimat de două programe nu poate fi comparat dacă ele folosesc parametrizări diferite.

## @end

Randamentele se calculează logaritmic, se anualizează cu frecvența reală a pieței și se raportează cu incertitudinea lor. Distribuția lor are asimetrie negativă și exces de boltire mare, ceea ce face ca testul Jarque–Bera să respingă normalitatea pentru toate seriile cursului. Legile α-stabile explică forma corpului distribuției, dar datele indică o varianță finită.

# Capitolele 4–7: probabilitate, cozi, alegerea modelului, eficiență

Cele patru capitole reluate aici dau fundamentul probabilistic al cursului și răspund la trei întrebări empirice: cît de groase sînt cozile, ce distribuție descrie cel mai bine randamentele și dacă randamentele trecute ajută la prognoza celor viitoare.

## Capitolul 4: probabilitate (1/2) {h=0fa6003d}

Varianța unei combinații liniare de două variabile este

$$ \begin{aligned} \Var(aX + bY) &= a^2\sigma_X^2 + b^2\sigma_Y^2 \\ &\quad + 2ab\,\Cov(X, Y), \end{aligned} $$

unde $a$ și $b$ sînt ponderile, iar $\sigma_X$ și $\sigma_Y$ sînt abaterile standard. Termenul de covarianță explică diversificarea: cu o corelație mai mică decît 1, volatilitatea portofoliului este mai mică decît media ponderată a volatilităților. Cînd o probabilitate $p$ este estimată prin simulare Monte Carlo cu $N$ extrageri, estimarea este o proporție, iar eroarea ei standard este $\sqrt{p(1 - p)/N}$.

Mișcarea browniană geometrică (GBM, *geometric Brownian motion*) este modelul de bază pentru prețuri:

$$ S_t = S_0\exp\!\Big(\big(\mu - \tfrac{\sigma^2}{2}\big)t + \sigma W_t\Big), $$

unde $t$ este timpul în ani, $\mu$ este tendința, $\sigma$ este volatilitatea, iar $W_t \sim N(0, t)$ este o mișcare browniană. Media prețului este $S_0e^{\mu t}$, iar mediana este $S_0e^{(\mu - \sigma^2/2)t}$, mai mică: distribuția log-normală a prețului este asimetrică la dreapta, iar cîteva traiectorii foarte bune trag media în sus.

## Capitolul 4: probabilitate (2/2) {h=55fc0fe3}

Pe datele S&P 500 din 2000, o zi cu un randament sub −5% are frecvența empirică 0,34%. O distribuție Normală cu aceeași medie și aceeași volatilitate dă o probabilitate de 0,0017%, de 199 de ori mai mică. Diferența nu ține de un detaliu al estimării, ci de forma cozii: distribuția Normală descrește prea repede pentru a reproduce frecvența zilelor extreme.

Al doilea fapt privește estimarea mediei. Randamentul mediu al S&P 500 din 2000 este 6,2% pe an, cu o eroare standard de 3,7%. Pentru ca media să fie semnificativă cu o statistică $t = 3$ ar fi nevoie de circa 86 de ani de date. Media randamentelor este cel mai greu de estimat parametru din finanțe, pentru că eroarea ei standard scade doar cu rădăcina pătrată a numărului de ani.

Greșelile frecvente sînt considerarea unor variabile necorelate drept independente, confuzia dintre media și mediana lui $S_t$ într-un GBM și raportarea unei estimări Monte Carlo fără eroarea ei standard.

## Capitolul 5: cozi groase și teoria valorilor extreme (1/2) {h=5a4bae59}

Estimatorul Hill [@Hill1975] măsoară cît de repede descresc cozile pierderilor. Dacă $L_{(1)} \ge L_{(2)} \ge \dots$ sînt pierderile ordonate descrescător și $k$ este numărul de observații din coadă,

$$ \hat\alpha_k = \left[\frac1k\sum_{i=1}^{k}\ln\frac{L_{(i)}}{L_{(k+1)}}\right]^{-1}, $$

cu eroarea standard aproximativă $\mathrm{SE}(\hat\alpha_k) \approx \hat\alpha_k/\sqrt k$. Un $\hat\alpha$ mic înseamnă cozi groase; parametrul de formă al cozii folosit în EVT este $\xi = 1/\alpha$. Alegerea lui $k$ este un compromis: un $k$ mic dă o estimare zgomotoasă, un $k$ mare include observații din corpul distribuției și deplasează estimarea.

Metoda depășirilor de prag modelează pierderile peste un prag $u$ cu distribuția Pareto generalizată (GPD, *generalised Pareto distribution*), cu scala $\beta$ și forma $\xi$. Dacă $N_u$ dintre cele $n$ pierderi depășesc pragul, VaR la nivelul $\alpha$ este

$$ \mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{n\alpha}{N_u}\right)^{-\xi} - 1\right], $$

unde, de data aceasta, $\alpha$ este nivelul VaR (de exemplu 1%), nu tail index-ul. A doua abordare a EVT, metoda maximelor pe blocuri (*block maxima*), ajustează distribuția generalizată a valorilor extreme (GEV, *generalised extreme value*) maximelor pierderilor pe perioade fixe.

## Capitolul 5: cozi groase și teoria valorilor extreme (2/2) {h=3d44d4ba}

Tail index-ul pierderilor este 2,57 pentru BET din 1997, cu un interval de încredere de 95% între 2,20 și 2,95, 2,97 pentru S&P 500 și 2,71 pentru Bitcoin. Toate valorile sînt între 2 și 3: varianța este finită, dar momentul de ordinul patru, deci boltirea, nu este. Rezultatul explică de ce excesul de boltire de selecție este atît de mare și de instabil.

Consecința pentru risc se vede la niveluri foarte mici. Pentru BET, VaR 0,1% calculat cu EVT este 9,6%, iar distribuția Normală dă doar 4,5%, mai puțin de jumătate. Cu cît nivelul VaR este mai mic, cu atît distribuția Normală subestimează mai mult pierderea.

Greșelile frecvente țin de aplicarea estimatorului. Alegerea lui $k$ trebuie făcută pe graficul Hill, care arată $\hat\alpha_k$ în funcție de $k$, căutînd o zonă stabilă; $\hat\alpha$ trebuie raportat cu intervalul lui; iar estimatorul se aplică pierderilor $L = -r$, nu randamentelor, pentru că altfel descrie coada cîștigurilor.

## Capitolul 6: selecția modelului și managementul riscului (1/2) {h=e83d2044}

Pentru două modele imbricate, testul raportului de verosimilitate (LR, *likelihood ratio*) compară log-verosimilitățile maxime $\ell_0$ a modelului restrîns și $\ell_1$ a celui extins: $LR = 2(\ell_1 - \ell_0)$ urmează aproximativ o distribuție $\chi^2(k_1 - k_0)$, unde $k_0$ și $k_1$ sînt numerele de parametri. Pentru modele neimbricate se folosesc criteriile informaționale

$$ \mathrm{AIC} = -2\ell + 2k, \qquad \mathrm{BIC} = -2\ell + k\ln n, $$

unde $k$ este numărul de parametri și $n$ volumul eșantionului; un AIC sau un BIC (*Bayesian information criterion*) mai mic indică un model mai bun. Ambele penalizează complexitatea, BIC mai puternic pentru eșantioane mari.

Testele de potrivire compară funcția de repartiție empirică $F_n$ cu modelul $F$. Statistica Kolmogorov–Smirnov (KS), $D_n = \sup_x|F_n(x) - F(x)|$, este cea mai mare distanță verticală dintre cele două și este dominată de centrul distribuției. Testul Anderson–Darling (AD) ponderează diferențele astfel încît cozile să conteze mai mult, ceea ce îl face mai potrivit pentru măsurile de risc.

## Capitolul 6: selecția modelului și managementul riscului (2/2) {h=a0e52bcf}

Pentru BET din 2000, statistica Anderson–Darling este 161 pentru distribuția Normală, 0,54 pentru Student-$t$ și 0,40 pentru NIG (*normal inverse Gaussian*). Distribuția Normală este respinsă categoric, iar cele două distribuții cu cozi groase sînt mult mai aproape de date, NIG fiind ușor mai bună.

Alegerea modelului schimbă riscul raportat. VaR 1% al BET este 3,19% cu distribuția Normală și între 3,74% și 4,94% cu modelele cu cozi groase. Diferența dintre modele este mai mare decît diferența dintre cele două margini ale intervalului de încredere al oricărei estimări, deci alegerea distribuției este o decizie de management al riscului, nu un detaliu tehnic.

Două greșeli apar frecvent. Testul LR care compară Student-$t$ cu distribuția Normală testează o valoare aflată la frontiera spațiului parametrilor ($\nu = \infty$), unde distribuția $\chi^2$ obișnuită nu se aplică și valoarea critică trebuie corectată. La eșantioane mari, testele de potrivire resping toate modelele; atunci ele trebuie ordonate după criterii, nu abandonate.

## Capitolul 7: piețe eficiente, mers aleator și teste VR (1/2) {h=e6bab49c}

Ipoteza piețelor eficiente în formă slabă spune că prețurile trecute nu permit prognoza randamentelor în exces [@Fama1970]. Ipoteza nulă testabilă este că randamentele formează o secvență de diferențe de martingal: media condiționată a randamentului de mîine, dată fiind istoria, nu depinde de această istorie.

Testul raportului varianțelor (VR, *variance ratio*) compară varianța randamentelor pe $q$ zile cu de $q$ ori varianța zilnică. Pentru un mers aleator raportul este 1; în funcție de autocorelațiile $\rho(k)$,

$$ \mathrm{VR}(q) = 1 + 2\sum_{k=1}^{q-1}\left(1 - \frac{k}{q}\right)\rho(k). $$

Un VR peste 1 indică autocorelații pozitive (momentum), un VR sub 1 indică revenire la medie. Statistica robustă a lui @LoMacKinlay1988, $Z^*(q) = (\widehat{\mathrm{VR}}(q) - 1)/\sqrt{\hat\theta/T}$, folosește o estimație $\hat\theta$ a varianței care rămîne validă în prezența volatility clustering și urmează aproximativ distribuția $N(0, 1)$ în ipoteza nulă; $T$ este numărul de randamente.

## Capitolul 7: piețe eficiente, mers aleator și teste VR (2/2) {h=100eb717}

Pentru BET din 1997, VR(5) este 1,28, iar statistica robustă $Z^* = 5{,}25$ respinge mersul aleator: randamentele de pe piața românească au avut momentum. Pentru S&P 500 din 1990, VR(5) este 0,86, cu $Z^* = -2{,}76$, o revenire la medie semnificativă. Pentru Bitcoin, $Z^* = -0{,}19$, deci mersul aleator nu este respins.

Prima greșeală frecventă este folosirea statisticii $Z(q)$, construită pentru randamente independente și identic distribuite (i.i.d.), pe serii cu volatility clustering. Eroarea ei standard este prea mică în acest caz, așa că testul respinge prea des ipoteza nulă adevărată.

A doua greșeală privește interpretarea. O rădăcină unitară în prețuri nu dovedește eficiența, pentru că și un preț cu randamente autocorelate poate avea rădăcină unitară. Iar o predictibilitate semnificativă statistic nu înseamnă profit: costurile de tranzacționare și riscul pot elimina cîștigul.

## @end

Probabilitatea dă instrumentele de bază ale cursului: varianța unui portofoliu, eroarea standard a unei simulări și proprietățile GBM. Tail index-urile pierderilor sînt între 2 și 3 pentru toate seriile cursului, iar modelele cu cozi groase dau un VaR 1% mai mare și mai realist decît distribuția Normală. Testele robuste ale raportului varianțelor resping mersul aleator pentru BET pe termen lung și pentru S&P 500 din 1990, dar nu pentru Bitcoin.

# Capitolele 8–11: volatilitate, risc și memorie

Această parte a cursului tratează mărimea randamentelor, adică partea lor previzibilă: cum se estimează volatilitatea, cum se modelează dinamica ei, cum devine ea o măsură de risc verificabilă și dacă randamentele au memorie lungă.

## Capitolul 8: estimatori de volatilitate (1/2) {h=9df98be1}

Modelul EWMA al RiskMetrics [@JPMorganReuters1996] actualizează varianța zilnic,

$$ \sigma_t^2 = \lambda\sigma_{t-1}^2 + (1 - \lambda)r_{t-1}^2, $$

unde $\lambda \in (0, 1)$ este factorul de atenuare (*decay factor*). Ponderea unei observații vechi de $j$ zile este proporțională cu $\lambda^j$, iar timpul de înjumătățire, numărul de zile după care ponderea scade la jumătate, este $\ln 0{,}5/\ln\lambda$.

Estimatorii de amplitudine (*range-based estimators*) folosesc maximul $H_t$ și minimul $L_t$ ale zilei. Estimatorul Parkinson, $(\ln H_t - \ln L_t)^2/(4\ln 2)$, este mai precis decît pătratul randamentului de închidere, pentru că amplitudinea zilei conține mai multă informație decît două prețuri de închidere. Testul ARCH-LM (*ARCH Lagrange multiplier*) verifică existența volatility clustering: se regresează $r_t^2$ pe $q$ laguri ale sale, iar $nR^2$ urmează o distribuție $\chi^2(q)$ în ipoteza de varianță constantă.

## Capitolul 8: estimatori de volatilitate (2/2) {h=57dc26ea}

Pentru S&P 500, statistica Ljung–Box $Q(10)$ este 91 pe randamente și 8 015 pe pătratele lor, față de valoarea critică 18,31. Diferența de aproape două ordine de mărime arată că dependența principală este în varianță. Cu $\lambda = 0{,}94$, valoarea RiskMetrics pentru date zilnice, timpul de înjumătățire al EWMA este 11,2 zile.

Avantajul EWMA față de o fereastră mobilă este absența efectului fantomă (*ghost effect*). Într-o fereastră de lungime fixă, toate observațiile au aceeași pondere, iar o zi extremă ridică brusc volatilitatea cînd intră în fereastră și o coboară la fel de brusc cînd iese, la o dată care nu are nicio legătură cu piața. În EWMA, ponderea zilei extreme scade treptat.

A doua greșeală frecventă privește estimatorii de amplitudine aplicați prețurilor cu salturi overnight. Estimatorul Parkinson folosește doar amplitudinea din timpul ședinței și ignoră diferența dintre închiderea de ieri și deschiderea de azi, așa că subestimează volatilitatea; corecția Yang–Zhang include această componentă.

## Capitolul 9: modele ARCH și GARCH (1/2) {h=a2e54227}

Modelul GARCH(1,1) [@Engle1982; @Bollerslev1986] descrie varianța condiționată ca o funcție de șocul de ieri și de varianța de ieri:

$$ \sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2, $$

$$ \bar\sigma^2 = \frac{\omega}{1 - \alpha - \beta}, $$

unde $\omega$ este constanta, $\alpha$ măsoară reacția la șocul de ieri $\varepsilon_{t-1}$, $\beta$ măsoară memoria, iar $\bar\sigma^2$ este varianța pe termen lung, definită doar dacă $\alpha + \beta < 1$.

Prognoza varianței la orizontul $h$, făcută în ziua $t$, revine geometric la nivelul pe termen lung:

$$ E_t\,\sigma_{t+h}^2 = \bar\sigma^2 + (\alpha + \beta)^{h-1}\big(\sigma_{t+1}^2 - \bar\sigma^2\big). $$

Persistența $\alpha + \beta$ dă viteza revenirii, iar timpul de înjumătățire al unui șoc de volatilitate este $\ln 0{,}5/\ln(\alpha + \beta)$. Spre deosebire de EWMA, care este cazul particular $\omega = 0$, $\alpha + \beta = 1$, GARCH are un nivel de echilibru către care converg prognozele pe termen lung.

## Capitolul 9: modele ARCH și GARCH (2/2) {h=25e7554d}

Modelul GARCH(1,1) cu inovații Student-$t$, estimat pe datele din 2000, are persistența 0,995 pentru S&P 500, cu un timp de înjumătățire de 131 de zile, și 0,991 pentru BET, cu 74 de zile. Un șoc de volatilitate se stinge deci în cîteva luni. Pentru Bitcoin, $\alpha + \beta = 1{,}000$: modelul este un IGARCH, în care șocurile nu se sting, iar varianța pe termen lung nu este definită.

Indicii de acțiuni au nevoie și de un termen asimetric: o scădere a prețului crește volatilitatea mai mult decît o creștere de aceeași mărime (*leverage effect*). Modelele GJR (Glosten–Jagannathan–Runkle) și EGARCH (*exponential GARCH*) adaugă acest efect.

Greșelile frecvente sînt folosirea inovațiilor Normale pentru randamente cu cozi groase, raportarea unor erori standard nerobuste și calcularea varianței pe termen lung $\omega/(1 - \alpha - \beta)$ atunci cînd $\alpha + \beta \ge 1$, caz în care formula dă o valoare infinită sau negativă, fără sens.

## Capitolul 10: VaR, ES și backtesting (1/2) {h=2181d060}

Fie $X$ randamentul unei poziții și $q_\alpha$ cuantila lui de ordin $\alpha$. Convenția cursului este

$$ \mathrm{VaR}_\alpha = -q_\alpha(X), $$

$$ \mathrm{ES}_\alpha = -\frac1\alpha\int_0^\alpha q_u\,du, $$

deci VaR 1% este pierderea depășită în 1% dintre zile, scrisă ca număr pozitiv, iar ES 2,5% este media pierderilor din cele mai rele 2,5% dintre zile. Pentru o distribuție Normală cu media $\mu$ și abaterea standard $\sigma$, $\mathrm{VaR}_\alpha = -(\mu + \sigma z_\alpha)$ și $\mathrm{ES}_\alpha = -\mu + \sigma\varphi(z_\alpha)/\alpha$, unde $z_\alpha$ este cuantila Normală standard și $\varphi$ este densitatea Normală standard. Spre deosebire de VaR, ES este o măsură de risc coerentă [@ArtznerEtAl1999]: ea respectă subaditivitatea, deci diversificarea nu poate crește riscul măsurat.

Backtesting-ul compară prognozele VaR cu pierderile realizate. Dacă $x$ dintre cele $n$ zile sînt depășiri, rata observată este $\hat\pi = x/n$, iar testul Kupiec [@Kupiec1995] o compară cu ținta $\alpha$:

$$ LR_{uc} = -2\ln\frac{(1-\alpha)^{n-x}\alpha^{x}}{(1-\hat\pi)^{n-x}\hat\pi^{x}}. $$

Dacă modelul este corect, $LR_{uc}$ urmează aproximativ distribuția $\chi^2(1)$.

## Capitolul 10: VaR, ES și backtesting (2/2) {h=711783fc}

Pentru S&P 500, VaR 1% istoric este 3,46%, iar VaR 1% Normal este 2,80%, adică prea mic cu 19%. În backtesting-ul din 2005, rata depășirilor VaR 1% este 2,71% pentru modelul Normal, 1,41% pentru simularea istorică (HS, *historical simulation*), 1,74% pentru GARCH-$t$ și 1,58% pentru simularea istorică filtrată (FHS, *filtered historical simulation*), față de ținta de 1%. Modelul Normal are aproape de trei ori mai multe depășiri decît ar trebui.

Prima greșeală frecventă este de convenție: un VaR negativ sau formularea „VaR 99%”. În curs, VaR 1% se referă la probabilitatea cozii și este o pierdere pozitivă. A doua este scalarea VaR la 10 zile prin înmulțire cu $\sqrt{10}$, care presupune randamente Normale i.i.d.; cu volatility clustering și cozi groase, regula poate fi mult în afara realității.

A treia greșeală este verificarea doar a numărului de depășiri. Un model poate avea rata corectă, dar depășiri concentrate în cîteva săptămîni de criză, exact cînd riscul contează cel mai mult; de aceea gruparea depășirilor se testează separat.

## Prognoze VaR 1% pentru BET: simulare istorică și EWMA-Normal {h=03b7b970}

Graficul arată randamentele zilnice ale BET în procente (linia albastră subțire), din 2017 pînă în 2026, împreună cu două prognoze ale VaR 1%, desenate cu semnul minus, ca prag sub care un randament devine depășire. Linia aurie este simularea istorică pe ultimele 500 de zile, iar linia roșie este modelul EWMA-Normal cu $\lambda = 0{,}94$. Punctele aurii și crucile roșii marchează depășirile celor două modele.

Din 3 ianuarie 2017, pe 2 421 de zile, numărul așteptat de depășiri este 24,2. Simularea istorică are 28 de depășiri (1,16%), iar testul Kupiec nu respinge modelul ($p = 0{,}450$). EWMA-Normal are 55 de depășiri (2,27%), iar testul îl respinge cu $p < 0{,}001$.

Graficul arată și de ce. Linia EWMA urmărește furtunile: după șocurile din decembrie 2018, martie 2020 și februarie 2022 ea coboară imediat, dar cuantila Normală este prea mică, așa că pragul este depășit des. Linia HS are în medie rata corectă, dar se mișcă în trepte, cu întîrziere, iar depășirile ei vin grupate, cel mult 10 în 250 de zile. Niciunul dintre cele două modele nu reunește o volatilitate actualizată cu o coadă corectă; FHS sau GARCH-$t$ fac acest lucru.

## Capitolul 11: ipoteza piețelor fractale și memoria lungă (1/2) {h=e6a84fbf}

Metoda R/S a lui @Hurst1951 împarte seria în blocuri de $n$ zile și calculează, în fiecare bloc, amplitudinea abaterilor cumulate de la medie împărțită la abaterea standard. Media acestei statistici crește cu lungimea blocului ca o putere,

$$ (R/S)_n \propto n^{H}, $$

iar exponentul Hurst $H$ se estimează din panta dreptei $\log(R/S)_n$ în funcție de $\log n$. $H = 0{,}5$ corespunde absenței memoriei, $H > 0{,}5$ indică persistență, iar parametrul fracționar este $d = H - 0{,}5$.

Exponentul Hurst schimbă regula de scalare a riscului: abaterea standard a randamentului pe $h$ zile este $\mathrm{sd}(r^{(h)}) = h^{H}\,\mathrm{sd}(r)$, iar regula rădăcinii pătrate este cazul $H = 0{,}5$. Ipoteza piețelor fractale a lui @Peters1994 susține că o piață este stabilă atunci cînd pe ea activează investitori cu orizonturi foarte diferite, de la cîteva minute la cîțiva ani, care se asigură reciproc lichiditatea.

## Capitolul 11: ipoteza piețelor fractale și memoria lungă (2/2) {h=e3dbf618}

Pentru BET, exponentul Hurst estimat prin R/S este 0,58 pentru randamente, la limita benzii Monte Carlo construite pentru serii fără memorie de aceeași lungime, iar testul Lo respinge absența memoriei: o persistență slabă. Pentru randamentele absolute, exponentul este 0,79. Pentru S&P 500, cifrele sînt 0,53 pentru randamente și 0,86 pentru randamentele absolute. Memoria lungă se află în volatilitate, nu în semnul randamentelor.

Prima greșeală frecventă este compararea lui $\hat H$ cu 0,5. Statistica R/S este deplasată în sus în eșantioane finite, așa că o serie fără memorie dă un $\hat H$ peste 0,5; comparația corectă este cu o bandă Monte Carlo pentru același număr de observații $N$.

A doua greșeală este atribuirea memoriei lungi unor fenomene care o imită. Rupturile structurale, cum ar fi o schimbare a nivelului volatilității, și efectele GARCH produc autocorelații persistente ale randamentelor absolute și pot da un $\hat H$ mare chiar dacă procesul nu are memorie lungă.

## @end

Volatilitatea este partea previzibilă a randamentelor: EWMA o urmărește fără efect fantomă, iar GARCH adaugă un nivel pe termen lung și o persistență estimată, aproape de 1 pentru toate seriile cursului. Măsurile de risc VaR 1% și ES 2,5% trebuie verificate prin backtesting, iar pe BET simularea istorică trece testul Kupiec, în timp ce EWMA-Normal îl pică. Memoria lungă apare în randamentele absolute, nu în randamente.

# Capitolele 12–15: aplicații

Ultimele patru capitole aplică instrumentele cursului la creditare, la prognoză prin învățare automată, la piața cripto și la stabilitatea sistemului bancar. În fiecare caz, întrebarea statistică este aceeași: cît de bine funcționează metoda în afara eșantionului și cît de incertă este estimarea.

## Capitolul 12: modele de scoring (1/2) {h=dde7d2a7}

Pierderea așteptată dintr-un credit (EL, *expected loss*) este produsul a trei factori,

$$ \mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD}, $$

unde PD (*probability of default*) este probabilitatea de nerambursare, LGD (*loss given default*) este proporția pierdută în caz de nerambursare, iar EAD (*exposure at default*) este expunerea la momentul nerambursării. Modelele de scoring estimează PD. Regresia logistică modelează logaritmul șanselor, $\ln\frac{p}{1-p} = \beta_0 + x^\top\beta$, unde $p$ este probabilitatea de nerambursare condiționată de caracteristicile $x$, iar $e^{\beta_j}$ este raportul șanselor asociat unei creșteri cu o unitate a variabilei $j$.

Capacitatea de discriminare se măsoară prin aria de sub curba ROC (AUC, *area under the curve*; ROC, *receiver operating characteristic*). AUC este probabilitatea ca un debitor rău ales la întîmplare să primească un scor de risc mai mare decît un debitor bun ales la întîmplare: 0,5 înseamnă nicio discriminare, 1 înseamnă discriminare perfectă. Coeficientul Gini este $2\,\mathrm{AUC} - 1$. Alături de modelele statistice, cursul a prezentat scorul $Z$ al lui @Altman1968, construit din indicatori contabili, și distanța pînă la nerambursare din modelul Merton, construită din prețurile acțiunilor.

## Capitolul 12: modele de scoring (2/2) {h=333b7069}

Pe datele germane de credit, 1 000 de credite dintre care 300 formează eșantionul de test, regresia logistică are AUC 0,801, cu un interval de încredere de 95% între 0,75 și 0,85, și un coeficient Gini de 0,60. Intervalul, larg pentru o mărime care se raportează adesea cu trei zecimale, vine din dimensiunea mică a eșantionului de test.

Discriminarea nu înseamnă calibrare. Un model poate ordona corect debitorii și totuși să dea probabilități de nerambursare sistematic prea mari sau prea mici; pentru calculul EL contează valorile, nu doar ordinea. Ambele proprietăți se verifică în afara eșantionului.

Prima greșeală frecventă este judecarea unui model după acuratețe atunci cînd clasele sînt dezechilibrate: dacă majoritatea creditelor sînt bune, regula „mereu bun” are deja o acuratețe mare fără să identifice niciun debitor rău. A doua este alegerea pragului de clasificare fără a ține seama de costurile celor două erori; respingerea unui client bun și acceptarea unuia rău nu costă la fel.

## Capitolul 13: învățare automată (1/2) {h=d15b6d3f}

Eroarea de prognoză pentru o observație nouă $y_0$ se descompune în trei termeni:

$$ E\big(y_0 - \hat f\big)^2 = \text{deplasare}^2 + \Var(\hat f) + \sigma^2, $$

unde $\hat f$ este modelul estimat, iar $\sigma^2$ este varianța zgomotului, care nu poate fi redusă. Metodele de învățare automată controlează compromisul dintre deplasare și varianță: ridge și lasso micșorează coeficienții (*shrinkage*), arborii de decizie și boosting-ul aleg singuri neliniaritățile, iar complexitatea se alege prin validare.

Performanța se măsoară în afara eșantionului, față de un reper simplu $\tilde y$ [@GuKellyXiu2020]:

$$ R^2_{OOS} = 1 - \frac{\sum(y - \hat y)^2}{\sum(y - \tilde y)^2}, $$

unde sumele parcurg eșantionul de test. Un $R^2_{OOS}$ negativ înseamnă că modelul prognozează mai slab decît reperul, ceea ce se întîmplă des pentru randamente.

## Capitolul 13: învățare automată (2/2) {h=56c1d7fc}

Pentru semnul randamentului zilnic al S&P 500 din 2008, regula naivă „mereu creștere” are o acuratețe de 54,4%, regresia logistică 53,8%, iar boosting-ul 52,1%. Modelele mai complexe nu bat reperul, în acord cu rezultatele de eficiență din [Capitolul 7](ch7_ro.html): semnul randamentelor este aproape imprevizibil.

Pentru volatilitatea realizată, imaginea este diferită. Modelul HAR (*heterogeneous autoregressive*), care folosește volatilitatea medie din ultima zi, săptămînă și lună, are $R^2_{OOS} = 0{,}61$ față de medie. Lasso reduce eroarea pătratică a HAR doar cu 7,1%: cea mai mare parte a predictibilității volatilității este captată de un model liniar simplu.

Greșelile frecvente țin de ordinea temporală. Validarea încrucișată cu împărțire aleatoare amestecă trecutul cu viitorul și introduce look-ahead bias și scurgeri de informație (*leakage*); pe serii de timp, eșantionul de validare trebuie să urmeze celui de estimare. Iar raportarea celei mai bune dintre multe strategii încercate, fără deflatarea raportului Sharpe pentru numărul de încercări, supraestimează performanța.

## Capitolul 14: active cripto (1/2) {h=ea2312ff}

Activele cripto se tranzacționează 365 de zile pe an, așa că volatilitatea lor se anualizează cu $\sigma_{\text{an}} = \sqrt{365}\,\sigma$, unde $\sigma$ este abaterea standard a randamentelor zilnice. Oferta de Bitcoin este fixată de protocol și limitată la 21 de milioane de monede [@Nakamoto2008].

Stablecoins sînt active cripto care urmăresc un preț fix de 1 USD. Abaterea de la paritate se măsoară în puncte de bază,

$$ d_t = 10\,000\,(P_t - 1), $$

unde $P_t$ este prețul stablecoin-ului în USD și un punct de bază este 0,01%. O valoare negativă înseamnă că stablecoin-ul se tranzacționează sub paritate, semn al neîncrederii în rezervele sau în mecanismul care îl susține.

## Capitolul 14: active cripto (2/2) {h=43c091c1}

Volatilitatea anuală a Bitcoin este 66,6% cînd se anualizează cu 365 de zile; folosirea greșită a 252 de zile ar da 55,3%, o subestimare cu circa 17%. Drawdown-ul maxim al Bitcoin este −83,4%. Corelația randamentelor Bitcoin cu S&P 500 a fost 0,02 înainte de 2020 și 0,39 după: Bitcoin a devenit un activ de risc care scade odată cu acțiunile, nu un activ de refugiu.

Prima greșeală frecventă privește calendarul. Eliminarea weekendurilor din datele cripto aruncă o parte din variațiile reale ale prețului, iar alinierea randamentelor cripto și de acțiuni pe calendare diferite atribuie variația de sîmbătă și duminică randamentului de luni, ceea ce deformează corelația. Prețurile se aliniază pe zilele comune, apoi se calculează randamentele.

A doua greșeală este un VaR Normal pentru o serie cu persistență GARCH egală cu 1 și tail index sub 3: ambele trăsături fac ca o cuantilă Normală cu volatilitate constantă să subestimeze pierderea.

## Capitolul 15: risc sistemic (1/2) {h=bec02ab4}

CoVaR (*conditional value at risk*), propus de @AdrianBrunnermeier2016, este VaR-ul sistemului financiar condiționat de situația în care o instituție se află la VaR-ul ei. Contribuția instituției la riscul sistemului este diferența dintre CoVaR-ul din starea de criză și cel din starea mediană. Dintr-o regresie cuantilică a randamentului sistemului pe randamentul băncii, cu panta $b$,

$$ \Delta\mathrm{CoVaR} = b\,(q_{50} - q_\alpha), $$

unde $q_\alpha$ și $q_{50}$ sînt cuantila de ordin $\alpha$ și mediana randamentului băncii.

Alte măsuri privesc aceeași întrebare din direcții diferite. MES (*marginal expected shortfall*) este pierderea medie a băncii în cele mai proaste zile ale pieței. SRISK (*systemic risk*) este capitalul care i-ar lipsi băncii într-o criză. Rețelele de cauzalitate Granger și tabelul de conectivitate al lui @DieboldYilmaz2012, construit din descompunerea varianței erorilor de prognoză, măsoară cît din volatilitatea fiecărei instituții vine de la celelalte.

## Capitolul 15: risc sistemic (2/2) {h=eb3c9472}

Pentru Banca Transilvania și indicele BET, VaR 1% al băncii este 5,23%, iar ΔCoVaR 1% este 3,31 puncte procentuale, cu un interval de încredere bootstrap de 95% între 2,35 și 4,78. Cînd banca trece de la starea mediană la starea de criză, VaR-ul pieței crește cu circa trei puncte procentuale. Pentru 11 bănci americane și europene, conectivitatea totală Diebold–Yilmaz este de circa 78%: în medie, cea mai mare parte a varianței erorilor de prognoză ale unei bănci vine de la șocurile celorlalte.

Prima greșeală frecventă este ordonarea băncilor după propriul VaR. O bancă foarte volatilă, dar slab legată de restul sistemului, are un VaR mare și o contribuție mică la riscul sistemic; ΔCoVaR măsoară tocmai legătura, nu riscul individual.

A doua greșeală privește comparația corelațiilor între perioade calme și de criză. Corelația crește mecanic atunci cînd volatilitatea crește, chiar dacă legătura dintre piețe nu se schimbă; ajustarea Forbes–Rigobon corectează acest efect înainte de a concluziona că a existat contagiune.

## @end

Aplicațiile refolosesc același lanț statistic. Un model de scoring se judecă prin AUC, calibrare și costuri, în afara eșantionului; învățarea automată nu prognozează semnul randamentelor, dar prognozează bine volatilitatea; activele cripto cer calendarul de 365 de zile și modele cu cozi groase și persistență totală; riscul sistemic se măsoară prin legătura unei instituții cu sistemul, nu prin riscul ei individual.

# Sinteza metodelor

Secțiunea rezumă cursul ca o listă de întrebări, fiecare cu instrumentul care îi răspunde, regula de decizie și capitolul în care a fost tratată. Ea se încheie cu convențiile de notație folosite în tot cursul și cu cele mai frecvente șase capcane.

## Sinteza metodelor (1/2): randamente, distribuții și cozi {h=c418a11a}

Primul tabel acoperă întrebările despre performanță și distribuție. Calitatea unei investiții se judecă prin CAGR, raportul Sharpe, raportul Sortino, care împarte randamentul în exces doar la volatilitatea randamentelor negative, și drawdown-ul maxim, comparate cu erorile lor standard și cu un reper ([Capitolul 1](ch1_ro.html)). Normalitatea se testează cu Jarque–Bera și cu graficul cuantilă–cuantilă (QQ, *quantile–quantile*); ea se respinge la 5% cînd JB depășește 5,99, valoarea critică a distribuției $\chi^2(2)$ ([Capitolul 2](ch2_ro.html)).

Grosimea cozilor se măsoară cu graficul Hill, cu GPD sau cu $\alpha$ al unei legi stabile, iar regula de interpretare este că momentele de ordin $p$ există doar pentru $p < \alpha$ ([Capitolele 3](ch3_ro.html) și [5](ch5_ro.html)). Alegerea distribuției se face cu LR, AIC, BIC, KS și AD: se preferă modelul cu AIC minim, iar testul AD se folosește atunci cînd contează cozile ([Capitolul 6](ch6_ro.html)).

Ultimele două întrebări privesc probabilitățile mici și varianța. Frecvența unei pierderi de o anumită mărime se estimează cu funcția de repartiție empirică, cu EVT sau prin Monte Carlo, raportînd întotdeauna eroarea standard a estimării ([Capitolele 4](ch4_ro.html) și [5](ch5_ro.html)). Dacă varianța este finită se verifică cu graficul varianței cumulative și cu comportamentul lui $\hat\alpha$ la agregare: o varianță de selecție care se stabilizează indică o varianță finită ([Capitolul 3](ch3_ro.html)).

## Sinteza metodelor (2/2): dependență în timp, volatilitate și risc {h=502d1e18}

Al doilea tabel acoperă dependența în timp și riscul. Predictibilitatea randamentelor se testează cu ACF, cu Ljung–Box și cu statistica robustă $Z^*(q)$ a raportului varianțelor, respingînd mersul aleator dacă $|Z^*| > 1{,}96$ ([Capitolul 7](ch7_ro.html)). Volatility clustering se detectează cu Ljung–Box pe $r_t^2$ și cu ARCH-LM; dacă testele resping, varianța trebuie modelată ([Capitolul 8](ch8_ro.html)). Prognoza volatilității de mîine se face cu EWMA, GARCH(1,1)-$t$ sau GJR, iar prognozele se compară cu funcția de pierdere QLIKE (*quasi-likelihood*) față de un reper ([Capitolele 8](ch8_ro.html) și [9](ch9_ro.html)).

Pierderea posibilă se măsoară prin VaR 1% și ES 2,5%, calculate prin simulare istorică, cu distribuția Student-$t$, cu EVT sau prin FHS, și se verifică prin testul Kupiec, testul Christoffersen pentru gruparea depășirilor și semaforul Basel ([Capitolul 10](ch10_ro.html)). Memoria lungă se estimează cu R/S, DFA (*detrended fluctuation analysis*), estimatorul GPH (Geweke–Porter-Hudak) și testul Lo, comparînd rezultatul cu o bandă Monte Carlo ([Capitolul 11](ch11_ro.html)).

Ultimele două rînduri sînt aplicațiile. Probabilitatea de nerambursare se modelează cu regresia logistică sau cu analiza discriminantă liniară (LDA, *linear discriminant analysis*) și se evaluează prin AUC și Gini, în afara eșantionului și ținînd seama de costuri ([Capitolul 12](ch12_ro.html)). Importanța unei bănci pentru sistem se măsoară cu CoVaR, MES și SRISK, cu intervale de încredere bootstrap pentru ΔCoVaR ([Capitolul 15](ch15_ro.html)).

## Convențiile folosite în tot cursul {h=aefd7067}

Randamentele sînt randamente logaritmice zilnice, în procente, $r_t = 100\,(\ln P_t - \ln P_{t-1})$. Excepția sînt portofoliile, pentru care se folosesc randamente simple, deoarece numai acestea se agregă liniar între active; înainte de calcul, prețurile activelor se aliniază pe zilele comune de tranzacționare.

Anualizarea înmulțește media cu $A$ și volatilitatea cu $\sqrt{A}$, unde $A$ este numărul real de observații pe an: circa 252 pentru acțiuni și indici și 365 pentru activele cripto. Nivelul unei măsuri de risc este probabilitatea cozii: VaR 1% și ES 2,5%, iar $\mathrm{VaR}_\alpha = -q_\alpha > 0$ este o pierdere. În simularea istorică, VaR este a $k$-a cea mai mică valoare a randamentelor, cu $k = \lceil n\alpha \rceil$.

Orice test se raportează complet: ipoteza nulă $H_0$, statistica, distribuția ei în ipoteza nulă, valoarea critică și decizia la 5%. Decizia este urmată de o frază despre ce înseamnă ea pentru date; o statistică fără interpretare nu răspunde la întrebarea pusă.

## Erori metodologice frecvente {h=b8c78e54}

Primele trei erori privesc datele, distribuția și inferența. La date, randamentele trebuie calculate după alinierea prețurilor, nu aliniate după calcul; prețurile trebuie ajustate pentru dividende și split-uri; iar weekendurile nu se elimină din seriile cripto. La distribuție, distribuția Normală nu se folosește pentru cuantilele din coadă, iar boltirea nu se confundă cu excesul de boltire. La inferență, erorile standard calculate în ipoteza i.i.d. sînt prea mici în prezența volatility clustering, iar o statistică raportată fără eroarea ei standard nu poate fi interpretată.

A patra eroare privește timpul: look-ahead bias apare atunci cînd parametrii estimați pe tot eșantionul sînt folosiți pentru a evalua prognoze făcute într-un moment în care acei parametri nu erau cunoscuți. A cincea privește riscul: semnul și nivelul VaR trebuie să respecte convenția cursului, iar ES nu se compară cu VaR calculat la alt nivel.

A șasea eroare ține de interpretare. Semnificația statistică nu înseamnă importanță economică: pe un eșantion mare, un efect foarte mic devine semnificativ. Iar predictibilitatea nu înseamnă profit, pentru că o strategie trebuie să acopere costurile de tranzacționare și riscul asumat.

## @end

Fiecare întrebare a cursului are un instrument și o regulă de decizie: JB pentru normalitate, graficul Hill pentru cozi, $Z^*(q)$ robust pentru predictibilitate, Ljung–Box pe $r_t^2$ pentru volatility clustering, GARCH-$t$ pentru prognoza volatilității, VaR 1% și ES 2,5% cu backtesting pentru risc. Convențiile comune și evitarea celor șase capcane fac rezultatele comparabile între capitole.

# Examenul

Secțiunea descrie forma evaluării scrise și prezintă opt probleme recapitulative, fiecare urmată de rezolvarea completă. Problemele acoperă toate tipurile de calcul ale cursului și arată, pe cifre simple, cum se leagă un calcul de interpretarea lui.

## Examenul scris: formatul {h=67b01faa}

Evaluarea scrisă reprezintă 70% din nota finală, durează două ore și permite folosirea calculatorului. Toate subiectele sînt obligatorii și se notează cu 9 puncte, la care se adaugă un punct din oficiu.

Valorile necesare calculelor sînt date odată cu subiectele: cuantile ale distribuției Normale și ale distribuției Student-$t$, valori critice ale distribuției $\chi^2$, logaritmi și radicali. Accentul cade astfel pe alegerea formulei și pe înțelegerea rezultatului, nu pe memorarea tabelelor statistice.

Fiecare subiect are un context scurt, cifre, un tabel sau un grafic și una sau două cerințe. Răspunsul complet este un calcul urmat de o interpretare în trei pînă la cinci fraze, aceeași structură pe care o urmează rezolvările problemelor din această secțiune.

## Conținutul evaluat {h=7c300ff8}

Materia acoperă cursurile și seminariile [Capitolelor 0](ch0_ro.html)–[15](ch15_ro.html). Definițiile și formulele sînt cele de pe slide-urile „Noțiuni necesare azi” ale fiecărui seminar, iar calculele sînt de tipul celor din Partea A a seminariilor, făcute pe hîrtie.

O a doua componentă este interpretarea rezultatelor obținute cu software: tabele și grafice de felul celor din Partea B a seminariilor, cu estimări, statistici de test, p-value-uri și rezultate de backtesting. Codul nu se scrie, dar rezultatul lui trebuie citit corect.

A treia componentă este înțelegerea: de ce o metodă dă greș pe randamentele financiare, de exemplu un VaR Normal pe o serie cu cozi groase, și ce test răspunde la ce întrebare. Sinteza metodelor din secțiunea anterioară rezumă exact această corespondență.

## Criterii de notare {h=3d7e72ec}

Un răspuns se evaluează pe trei componente. Prima este metoda: formula scrisă, cu cifrele înlocuite. O metodă corectă cu o greșeală de calcul primește cea mai mare parte a punctajului, pentru că arată că raționamentul este corect.

A doua componentă este rezultatul: cifra, cu unitatea de măsură (procente, zile, lei) și cu semnul ei. Pentru un test, rezultatul complet conține ipoteza nulă $H_0$, statistica, valoarea critică sau p-value-ul și decizia.

A treia componentă este interpretarea, în trei pînă la cinci fraze, cu o parte statistică și una economică. Ea precizează ipotezele unei formule aproximative, de exemplu randamente i.i.d., distribuția Normală sau un eșantion mare. O cifră corectă însoțită de o interpretare greșită nu primește tot punctajul.

## Tipuri de probleme {h=f567510b}

Tabelul grupează problemele recapitulative după tipul calculului și după capitolele din care provin. Randamentele, anualizarea și performanța vin din [Capitolele 0](ch0_ro.html) și [1](ch1_ro.html) (Problema 1); momentele, normalitatea și cozile vin din [Capitolele 2](ch2_ro.html), [3](ch3_ro.html), [5](ch5_ro.html) și [6](ch6_ro.html) (Problema 2); probabilitatea și simularea vin din [Capitolul 4](ch4_ro.html) (Problema 3).

Eficiența și memoria lungă, din [Capitolele 7](ch7_ro.html) și [11](ch11_ro.html), apar în Problemele 4 și 7; volatilitatea, din [Capitolele 8](ch8_ro.html), [9](ch9_ro.html) și [14](ch14_ro.html), în Problema 5; măsurile de risc și backtesting-ul, din [Capitolul 10](ch10_ro.html), în Problema 6; riscul de credit și riscul sistemic, din [Capitolele 12](ch12_ro.html), [13](ch13_ro.html) și [15](ch15_ro.html), în Problema 8.

Cele opt probleme sînt exemple de recapitulare cu rezolvări complete. Fiecare folosește cifre simple, astfel încît calculul să poată fi refăcut cu un calculator, iar fiecare rezolvare se încheie cu interpretarea rezultatului.

## Problema 1: randamente și drawdown {h=03c743fa}

Un indice valorează 100, 160, 64 și 96 la sfîrșitul anilor 0, 1, 2 și 3. Seria a fost aleasă astfel încît să conțină o creștere mare, o prăbușire și o revenire parțială, situația în care diferența dintre randamentele simple și cele logaritmice, ca și dintre media aritmetică și CAGR, devine evidentă.

Cerințele merg de la definiții la interpretare. Se calculează randamentele simple și logaritmice ale celor trei ani, apoi randamentul total în două moduri, din suma randamentelor logaritmice și direct din prețuri. Urmează media aritmetică a randamentelor simple comparată cu CAGR și, în final, drawdown-ul maxim împreună cu cîștigul necesar pentru a reveni de la minim la vîrful anterior.

## Problema 1: rezolvare {h=decbbfa9}

Randamentele simple sînt 60%, −60% și 50%, iar cele logaritmice sînt $\ln 1{,}6 = 0{,}470$, $\ln 0{,}4 = -0{,}916$ și $\ln 1{,}5 = 0{,}405$. Suma randamentelor logaritmice este $-0{,}041 = \ln(96/100)$, deci randamentul total este $e^{-0{,}041} - 1 = -4\%$, exact ca $96/100 - 1$. Randamentele logaritmice se adună în timp; suma randamentelor simple, 50%, nu are nicio legătură cu rezultatul real.

Media aritmetică a randamentelor simple este $(60 - 60 + 50)/3 = 16{,}67\%$, iar CAGR este $(96/100)^{1/3} - 1 = -1{,}35\%$. Indicele are un randament mediu pozitiv și totuși investitorul a pierdut bani: volatilitatea mare reduce creșterea compusă (volatility drag), iar media aritmetică nu descrie ce a obținut efectiv cineva care a păstrat investiția.

Drawdown-ul maxim este $64/160 - 1 = -60\%$, de la vîrful de la sfîrșitul primului an la minimul de la sfîrșitul celui de-al doilea. Pentru a reveni de la 64 la 160, indicele are nevoie de un cîștig de $160/64 - 1 = 150\%$. Asimetria dintre pierdere și cîștigul necesar crește rapid cu mărimea pierderii, ceea ce face drawdown-urile mari atît de greu de recuperat.

## Problema 2: distribuție și cozi {h=6df2b870}

Contextul este un eșantion de $n = 1\,000$ de randamente zilnice, cu asimetria $S = -0{,}5$ și excesul de boltire $K = 6$, valori obișnuite pentru un indice bursier. Cele mai mari cinci pierderi, în procente, sînt 8,2, 6,5, 5,9, 5,1 și 4,6, iar valoarea critică a distribuției $\chi^2(2)$ la 5% este 5,99.

Cerințele leagă trei capitole. Testul Jarque–Bera, cu cele două componente ale lui, decide asupra normalității; distribuția Student-$t$ cu același exces de boltire dă un număr de grade de libertate; estimatorul Hill cu $k = 4$ dă tail index-ul. Ultima cerință întreabă ce momente ale pierderilor există și dacă o lege stabilă cu $\alpha < 2$ este compatibilă cu rezultatul.

## Problema 2: rezolvare {h=d263c7ed}

Componenta asimetriei din statistica Jarque–Bera este $\frac{1000}{6}\cdot 0{,}25 = 41{,}67$, iar componenta boltirii este $\frac{1000}{24}\cdot 36 = 1\,500$. Suma, JB $= 1\,541{,}7$, depășește cu mult 5,99, deci normalitatea se respinge. Componenta boltirii domină: cozile groase, nu asimetria, sînt motivul respingerii. Din $6/(\nu - 4) = 6$ rezultă $\nu = 5$ grade de libertate pentru distribuția Student-$t$ cu același exces de boltire.

Pentru estimatorul Hill cu $k = 4$, pragul este a cincea pierdere, 4,6. Logaritmii rapoartelor sînt $\ln(8{,}2/4{,}6) = 0{,}578$, $\ln(6{,}5/4{,}6) = 0{,}346$, $\ln(5{,}9/4{,}6) = 0{,}249$ și $\ln(5{,}1/4{,}6) = 0{,}103$, cu media 0,3190, deci $\hat\alpha = 1/0{,}3190 = 3{,}14$. Cu un tail index de 3,14 există momentele de ordin mai mic decît 3,14: media, varianța și asimetria; boltirea nu există. O lege stabilă cu $\alpha < 2$, cu varianță infinită, ar avea cozi mai groase decît aceste date.

Estimarea are o incertitudine mare. Cu $k = 4$, eroarea standard este aproximativ $\hat\alpha/\sqrt{4} = \hat\alpha/2$, deci aproximativ 1,6. În practică, $k$ se alege din graficul Hill, pe un eșantion de pierderi suficient de mare pentru a avea o zonă stabilă.

## Problema 3: portofoliu, Monte Carlo și GBM {h=f6d46c0a}

Problema are trei părți independente. Prima privește un portofoliu din două active, cu volatilități zilnice de 1,5% și 2,5%, corelația 0,3 și ponderile 60% și 40%; se cere volatilitatea zilnică a portofoliului și comparația ei cu media ponderată a volatilităților.

A doua parte privește precizia unei simulări: un studiu Monte Carlo cu $N = 10\,000$ de extrageri estimează o probabilitate de 1%, iar cerința este eroarea standard și un interval de încredere de 95%. A treia parte privește un preț care urmează un GBM cu tendința $\mu = 8\%$, volatilitatea $\sigma = 25\%$ pe an și $S_0 = 100$; se cer media și mediana prețului peste un an și probabilitatea ca prețul să scadă sub valoarea inițială.

## Problema 3: rezolvare {h=bc87f128}

Varianța zilnică a portofoliului este

$$ \begin{aligned} \sigma_p^2 &= 0{,}6^2 \cdot 1{,}5^2 + 0{,}4^2 \cdot 2{,}5^2 \\ &\quad + 2 \cdot 0{,}6 \cdot 0{,}4 \cdot 0{,}3 \cdot 1{,}5 \cdot 2{,}5 \\ &= 0{,}81 + 1{,}00 + 0{,}54 = 2{,}35, \end{aligned} $$

deci volatilitatea este 1,53%. Media ponderată a volatilităților este 1,90%. Diferența este efectul diversificării: cu o corelație mai mică decît 1, portofoliul este mai puțin riscant decît media componentelor sale.

Eroarea standard a estimării Monte Carlo este $\sqrt{0{,}01 \cdot 0{,}99/10\,000} = 0{,}099\%$, iar intervalul de încredere de 95% este între 0,80% și 1,20%. Cu zece mii de extrageri, o probabilitate de 1% este estimată cu o eroare relativă de circa 10%; pentru probabilitățile din coada extremă, numărul de extrageri trebuie mărit mult.

Pentru GBM, media prețului peste un an este $100e^{0{,}08} = 108{,}33$, iar mediana este $100e^{0{,}08 - 0{,}03125} = 105{,}00$. Prețul scade sub 100 dacă $\ln(S_1/S_0) < 0$, iar această variabilă are media $0{,}0488$ și abaterea standard $0{,}25$, deci probabilitatea este $\Phi(-0{,}0488/0{,}25) = \Phi(-0{,}195) = 42{,}3\%$. Deși prețul așteptat crește, un investitor pierde bani după un an cu o probabilitate de 42,3%.

## Problema 4: autocorelație și raportul varianțelor {h=53698f58}

Se dau $T = 2\,500$ de randamente zilnice cu primele patru autocorelații de selecție 0,06, 0,02, −0,01 și 0,01. Valoarea critică a distribuției $\chi^2(4)$ la 5% este 9,49, iar eroarea standard robustă la heteroscedasticitate a raportului varianțelor estimat $\widehat{\mathrm{VR}}(5)$ este 0,065.

Cerințele compară trei teste ale aceleiași ipoteze. Statistica Ljung–Box $Q(4)$ testează absența autocorelației pe primele patru laguri. Raportul varianțelor $\mathrm{VR}(5) = 1 + 2\sum_{k=1}^{4}(1 - k/5)\hat\rho(k)$ rezumă aceleași autocorelații într-un singur număr. Statistica i.i.d. $Z(5)$, cu eroarea standard

$$ \mathrm{SE} = \sqrt{\frac{2(2q-1)(q-1)}{3qT}}, $$

și statistica robustă $Z^*(5)$ testează dacă acest număr diferă de 1.

## Problema 4: rezolvare {h=9b450b7a}

Statistica Ljung–Box este $Q(4) = T(T+2)\sum_{k=1}^{4}\hat\rho_k^2/(T - k) = 10{,}51$, foarte apropiată de aproximarea $T\sum_k\hat\rho_k^2 = 2\,500 \cdot 0{,}0042$. Deoarece $10{,}51 > 9{,}49$, ipoteza de absență a autocorelației se respinge la 5%.

Raportul varianțelor este

$$ \begin{aligned} \mathrm{VR}(5) &= 1 + 2(0{,}8 \cdot 0{,}06 + 0{,}6 \cdot 0{,}02 \\ &\quad - 0{,}4 \cdot 0{,}01 + 0{,}2 \cdot 0{,}01) \\ &= 1 + 2 \cdot 0{,}058 = 1{,}116. \end{aligned} $$

O valoare peste 1 corespunde autocorelațiilor pozitive, adică momentum-ului.

Eroarea standard i.i.d. este $\sqrt{72/37\,500} = 0{,}0438$, deci $Z(5) = 0{,}116/0{,}0438 = 2{,}65 > 1{,}96$, iar testul i.i.d. respinge mersul aleator. Statistica robustă este $Z^*(5) = 0{,}116/0{,}065 = 1{,}78 < 1{,}96$ și nu îl respinge. Volatility clustering mărește artificial statistica $Z(5)$, pentru că eroarea ei standard ignoră variația varianței în timp; decizia care se raportează este cea robustă.

## Problema 5: prognoze de volatilitate {h=cb43b248}

Un model GARCH(1,1) pentru randamente zilnice în procente are $\omega = 0{,}03$, $\alpha = 0{,}09$ și $\beta = 0{,}89$. Azi, varianța condiționată este $\sigma_t^2 = 1{,}8$, iar șocul zilei este $\varepsilon_t = -2{,}5$, o scădere mare. A doua parte a problemei privește Bitcoin, cu o abatere standard zilnică de 3,5%.

Cerințele parcurg utilizările modelului. Se calculează varianța de mîine și VaR 1% Normal pe o zi, cu medie zero și $z_{0{,}01} = -2{,}326$; apoi persistența, timpul de înjumătățire, varianța pe termen lung și volatilitatea anuală pe termen lung; apoi prognoza varianței peste zece zile, comparată cu prognoza EWMA cu $\lambda = 0{,}94$; în final, volatilitatea anuală a Bitcoin.

## Problema 5: rezolvare {h=50f81e1c}

Varianța de mîine este

$$ \begin{aligned} \sigma_{t+1}^2 &= 0{,}03 + 0{,}09 \cdot 6{,}25 + 0{,}89 \cdot 1{,}8 \\ &= 0{,}03 + 0{,}5625 + 1{,}602 = 2{,}1945, \end{aligned} $$

deci $\sigma_{t+1} = 1{,}481\%$ și VaR 1% $= 2{,}326 \cdot 1{,}481 = 3{,}45\%$. Persistența este $\alpha + \beta = 0{,}98$, timpul de înjumătățire este $\ln 0{,}5/\ln 0{,}98 = 34{,}3$ zile, varianța pe termen lung este $0{,}03/0{,}02 = 1{,}50$, iar volatilitatea anuală pe termen lung este $\sqrt{252 \cdot 1{,}50} = 19{,}44\%$.

Prognoza la zece zile este

$$ \begin{aligned} E_t\sigma_{t+10}^2 &= 1{,}50 + 0{,}98^{9}(2{,}1945 - 1{,}50) \\ &= 2{,}079. \end{aligned} $$

Ea a coborît de la 2,19 spre nivelul de echilibru 1,50, dar este încă departe de el. Prognoza EWMA este $0{,}94 \cdot 1{,}8 + 0{,}06 \cdot 6{,}25 = 2{,}067$, aproape de cea GARCH la acest orizont. Diferența apare la orizonturi lungi: EWMA nu are un nivel pe termen lung, așa că prognoza ei rămîne aceeași pentru orice orizont, în timp ce prognoza GARCH revine la $\bar\sigma^2$.

Bitcoin se tranzacționează în fiecare zi, deci volatilitatea anuală este $3{,}5\sqrt{365} = 66{,}9\%$. Cu 252 de zile s-ar obține greșit 55,6%. Un șoc negativ mare crește varianța de mîine, iar persistența $\alpha + \beta$ stabilește cît timp durează efectul.

## Problema 6: VaR, ES și backtesting {h=3ec33c00}

O poziție de 1 000 000 de lei are randamente zilnice Normale cu media zero și abaterea standard 1,6%; densitatea Normală standard în 1,96 este $\varphi(1{,}96) = 0{,}05845$. Pentru comparație se dau cele mai mici șapte dintre 250 de randamente istorice, în procente: −5,1, −4,4, −3,9, −3,6, −3,2, −3,0 și −2,9.

Cerințele compară două metode de calcul și un test. Se calculează VaR 1% și ES 2,5% Normale, în procente și în lei, și VaR 1% pe 10 zile cu regula rădăcinii pătrate a timpului; apoi aceleași măsuri prin simulare istorică. Ultima cerință privește un model care are 6 depășiri ale VaR 1% în 250 de zile: statistica Kupiec, decizia la 5% și zona semaforului Basel.

## Problema 6: rezolvare {h=db933380}

Cu distribuția Normală, VaR 1% $= 2{,}326 \cdot 1{,}6 = 3{,}72\%$, adică 37 222 de lei, iar ES 2,5% $= 1{,}6 \cdot 0{,}05845/0{,}025 = 1{,}6 \cdot 2{,}338 = 3{,}74\%$, adică 37 405 lei. VaR pe 10 zile este $\sqrt{10} \cdot 3{,}72 = 11{,}77\%$. Pentru distribuția Normală, ES 2,5% și VaR 1% sînt aproape egale.

Prin simulare istorică, VaR 1% este a $k = \lceil 2{,}5 \rceil = 3$-a cea mai mică valoare, deci 3,9%, iar ES 2,5% este media celor mai mici $k = \lceil 6{,}25 \rceil = 7$ randamente, $26{,}1/7 = 3{,}73\%$. VaR istoric este mai mare decît cel Normal: coada istorică este mai groasă.

Pentru testul Kupiec, log-verosimilitatea cu rata țintă este $\ell_0 = 244\ln 0{,}99 + 6\ln 0{,}01 = -30{,}08$, iar cu rata observată $\hat\pi = 6/250 = 0{,}024$ este $\ell_1 = 244\ln 0{,}976 + 6\ln 0{,}024 = -28{,}31$. Statistica $LR_{uc} = 2(\ell_1 - \ell_0) = 3{,}56$ este sub 3,84, valoarea critică a $\chi^2(1)$, cu $p = 0{,}059$, deci modelul nu este respins. Cu 6 depășiri, modelul se află însă în zona galbenă a semaforului Basel (5–9 depășiri), care cere deja un multiplicator de capital mai mare.

## Problema 7: memorie lungă și scalarea riscului {h=712eada8}

Statistica R/S medie a unor randamente zilnice este 4,1 pentru blocuri de $n = 20$ de zile și 60,0 pentru blocuri de $n = 2\,000$ de zile. VaR 1% pe o zi este 2,5%.

Prima cerință estimează exponentul Hurst din panta lui $\log(R/S)$ în funcție de $\log n$ și parametrul fracționar $d$. A doua scalează VaR la 10 zile în două moduri, cu $h^H$ și cu $\sqrt{h}$, și compară rezultatele. A treia cere verificările necesare înainte de a concluziona că seria are memorie lungă.

## Problema 7: rezolvare {h=7c32d7be}

Cu două puncte, panta este

$$ H = \frac{\log_{10}(60{,}0/4{,}1)}{\log_{10}(2\,000/20)} = \frac{1{,}165}{2} = 0{,}583, $$

iar parametrul fracționar este $d = H - 0{,}5 = 0{,}083$.

Scalat cu $h^H$, VaR pe 10 zile este $10^{0{,}583} \cdot 2{,}5 = 3{,}83 \cdot 2{,}5 = 9{,}56\%$; cu regula rădăcinii pătrate, $\sqrt{10} \cdot 2{,}5 = 3{,}16 \cdot 2{,}5 = 7{,}91\%$. Dacă $H > 0{,}5$, regula rădăcinii pătrate a timpului subestimează riscul pe mai multe zile, pentru că randamentele persistente se adună mai repede decît cele independente.

Înainte de a accepta concluzia, $\hat H$ trebuie comparat cu o bandă Monte Carlo construită pe date i.i.d. de aceeași lungime, deoarece statistica R/S este deplasată în sus. Rezultatul se verifică apoi cu testul Lo, pe subeșantioane și pe date permutate, care elimină memoria, dar păstrează distribuția; astfel se exclud rupturile structurale și efectele GARCH. Pe datele cursului, memoria lungă se află în $|r_t|$, nu în $r_t$ ([Capitolul 11](ch11_ro.html)).

## Problema 8: scoring de credit și risc sistemic {h=58921ecd}

Prima parte a problemei privește un eșantion de test de 300 de credite, dintre care 90 neperformante. La pragul ales, modelul marchează ca neperformante 60 dintre creditele neperformante și 42 dintre cele performante. Se cer acuratețea, rata adevărat pozitivelor (TPR, *true positive rate*), rata fals pozitivelor (FPR, *false positive rate*) și precizia, comparate cu regula „mereu bun”; apoi coeficientul Gini pentru AUC = 0,80 și pierderea așteptată pentru PD = 4%, LGD = 45% și EAD = 200 000 de lei.

A doua parte privește riscul sistemic. Regresia cuantilică la 1% a randamentului sistemului pe randamentul unei bănci are constanta $\hat a = -2{,}1$ și panta $\hat b = 0{,}6$, iar cuantila de 1% a randamentului băncii este −5,0%, cu mediana 0,1%. Se cer CoVaR 1% al sistemului cînd banca este la VaR-ul ei și cînd este la mediană, precum și ΔCoVaR.

## Problema 8: rezolvare {h=8a82ef55}

Creditele performante corect clasificate sînt $210 - 42 = 168$, deci acuratețea este $(60 + 168)/300 = 76{,}0\%$. TPR este $60/90 = 66{,}7\%$, FPR este $42/210 = 20{,}0\%$, iar precizia, proporția creditelor neperformante printre cele marcate astfel, este $60/102 = 58{,}8\%$. Regula „mereu bun” are acuratețea $210/300 = 70{,}0\%$. Modelul este doar puțin mai precis decît regula naivă, dar găsește două treimi dintre creditele neperformante; de aceea el se judecă după AUC și după costurile erorilor, nu după acuratețe.

Coeficientul Gini este $2 \cdot 0{,}80 - 1 = 0{,}60$, iar pierderea așteptată este $\mathrm{EL} = 0{,}04 \cdot 0{,}45 \cdot 200\,000 = 3\,600$ de lei.

Cînd banca este la VaR-ul ei, CoVaR 1% al sistemului este $-(-2{,}1 + 0{,}6 \cdot (-5{,}0)) = 5{,}10\%$; cînd banca este la mediană, este $-(-2{,}1 + 0{,}6 \cdot 0{,}1) = 2{,}04\%$. Diferența, $\Delta\mathrm{CoVaR} = 0{,}6 \cdot (0{,}1 - (-5{,}0)) = 3{,}06$ puncte procentuale, arată cu cît crește VaR-ul sistemului atunci cînd banca trece de la starea normală la dificultăți.

## @end

Cele opt probleme acoperă toate tipurile de calcul ale cursului: randamente și drawdown, momente și cozi, portofolii și simulare, teste de eficiență, prognoze de volatilitate, VaR și backtesting, memorie lungă, scoring și risc sistemic. Fiecare rezolvare are aceeași structură: formula cu cifrele înlocuite, rezultatul cu unitatea și semnul lui, decizia testului și o interpretare statistică și economică.

# Proiectul de echipă

Proiectul de echipă aplică metodele cursului pe o întrebare aleasă de echipă. Secțiunea descrie conținutul și livrabilele, criteriile de evaluare, susținerea orală și fișierul în care se declară folosirea instrumentelor AI.

## Proiectul de echipă: conținut și livrabile {h=39702018}

Proiectul reprezintă 20% din nota finală și este o analiză, făcută în echipă, a unor date reale de piață cu metodele cursului. El are o singură întrebare de cercetare, iar răspunsul folosește instrumentele cursului: randamente și indicatori, distribuții și cozi, teste de eficiență, volatilitate și măsuri de risc. Ideile de pornire se găsesc pe slide-ul „Idee de proiect” de la finalul fiecărui capitol și în Partea C a fiecărui seminar.

Livrabilele sînt trei. Primul este un repository GitHub al cărui cod reproduce, din datele salvate, fiecare rezultat numeric și fiecare grafic. Al doilea este un raport scurt, însoțit de o prezentare a rezultatelor. Al treilea este fișierul AI_USE.md, în care este declarată fiecare utilizare a instrumentelor AI.

Cerința de reproductibilitate are un motiv statistic. O cifră care nu poate fi recalculată din date nu poate fi verificată, iar o analiză care nu poate fi verificată nu poate fi crezută, indiferent cît de convingătoare pare.

## Criterii de evaluare a proiectului {h=4501746d}

Primul criteriu este întrebarea: ea trebuie să fie clară, să poată primi un răspuns pe baza datelor și să fie relevantă pentru investitori sau pentru autorități. „Este BET azi mai puțin volatil decît în 2008?” este o întrebare, pentru că indică o mărime de comparat; „o analiză a BET” nu este, pentru că nu spune ce rezultat ar răspunde la ea.

Al doilea criteriu sînt metodele și verificările. Testul trebuie să fie cel potrivit întrebării, așa cum îl indică sinteza metodelor, cu erori standard robuste și cu evaluare în afara eșantionului acolo unde există o prognoză. Robustețea se verifică schimbînd fereastra, seria sau estimatorul: un rezultat care dispare la o mică schimbare a alegerilor nu este un rezultat solid.

Al treilea criteriu este interpretarea: ce înseamnă cifrele și ce nu pot arăta datele. Al patrulea este reproductibilitatea: repository-ul trebuie să ruleze de la datele salvate pînă la fiecare grafic.

## Susținerea orală {h=bb7077a1}

La susținere, fiecare membru al echipei explică codul și rezultatele. Întrebările tipice sînt ce calculează o anumită linie de cod, de unde provine o anumită cifră și de ce a fost ales un test și nu altul. O a doua categorie de întrebări privește sensibilitatea: ce se schimbă dacă se modifică fereastra, seria sau nivelul $\alpha$.

Regula de bază este că o linie de cod pe care autorul ei nu o poate explica nu este considerată muncă proprie. Regula se aplică la fel codului scris cu sau fără ajutorul unui asistent AI.

Pregătirea constă în rularea din nou a repository-ului de la zero cu o zi înainte, pentru a verifica faptul că rezultatele se reproduc, și în cunoașterea definiției fiecărei cifre de pe slide-urile prezentării.

## Fișierul AI_USE.md {h=936a4462}

Instrumentele AI sînt permise, cu condiția să fie declarate. Pentru fiecare utilizare, fișierul AI_USE.md consemnează instrumentul, promptul, ce s-a păstrat din răspuns și ce s-a corectat. El conține și erorile instrumentului AI descoperite de echipă, împreună cu modul în care au fost descoperite.

Echipa răspunde pentru fiecare cifră și pentru fiecare referință. Fiecare cifră trebuie recalculată cu propriul cod, din datele cursului, iar pentru fiecare referință trebuie deschis DOI-ul (*digital object identifier*) și verificat titlul lucrării.

Slide-ul enumeră erorile AI tipice pentru acest curs: „VaR 99%” raportat cu semn negativ, Bitcoin anualizat cu 252 de zile, referințe bibliografice inventate și un test aplicat cu distribuția greșită. Fiecare dintre ele încalcă una dintre convențiile cursului și poate fi găsită printr-o verificare simplă.

## @end

Proiectul de echipă răspunde la o singură întrebare clară, cu metodele potrivite și cu verificări de robustețe, într-un repository care reproduce fiecare rezultat. La susținere, fiecare membru explică codul și cifrele, iar orice utilizare a AI este declarată și verificată.

# Instrumente AI pentru un proiect de cercetare

Fiecare capitol al cursului s-a încheiat cu o întrebare deschisă, la care metodele capitolului pot contribui. Întrebarea acestui capitol reunește mai multe instrumente: s-a schimbat comportamentul statistic al pieței din România de-a lungul timpului? Secțiunea descrie bucla de cercetare, ce poate face un asistent AI, ce trebuie verificat și o idee de proiect.

## O întrebare de cercetare deschisă despre BET {h=ae70a9db}

Întrebarea este dacă BET a devenit o piață „obișnuită”, cu proprietățile statistice ale piețelor dezvoltate. Pe toată perioada din 1997, raportul varianțelor VR(5) este 1,28, un momentum semnificativ, iar tail index-ul pierderilor este 2,57. Pe fereastra din 2015, VR(5) a scăzut la 1,17, cu $Z^* = 1{,}93$, sub pragul de semnificație, dar tail index-ul este 2,19, deci cozile sînt chiar mai groase.

Întrebarea rămîne deschisă din mai multe motive. Momentum-ul slăbește, dar cozile nu devin mai subțiri, așa că cele două dimensiuni ale „normalizării” nu merg împreună. O singură zi extremă, scăderea de −11,9% din 19 decembrie 2018, modifică vizibil estimarea cozii pe o fereastră scurtă. Iar crizele sînt puține și eșantioanele scurte, deci orice comparație între perioade are o incertitudine mare.

Există și un motiv mai general. @MenkveldEtAl2024 au dat aceleași date și aceeași întrebare mai multor echipe de cercetători și au găsit că rezultatele diferă mult de la o echipă la alta; variația provine din alegerile de metodă, nu din date, și au numit-o eroare nestandard (*non-standard error*). Pentru întrebarea despre BET, alegerea ferestrei, a lui $k$ sau a estimatorului poate schimba răspunsul.

## Bucla de cercetare {h=3a1b42c6}

Bucla are șase pași. Întrebarea se formulează într-o frază, împreună cu cifra care i-ar răspunde. Datele, adică seria, sursa, calendarul și fereastra, se fixează înainte de a vedea rezultatele, astfel încît alegerile să nu fie influențate de ele. Metoda este testul potrivit din trusa de instrumente, cu ipoteza nulă precizată.

Urmează verificările: erori standard robuste, benzi Monte Carlo, subeșantioane și un al doilea estimator. Interpretarea dă răspunsul, incertitudinea lui și limitele lui. Ultimul pas este replicarea: o a doua persoană rulează codul și obține aceleași cifre.

Un asistent AI poate accelera fiecare pas, dar nu înlocuiește niciuna dintre verificări. @WangEtAl2023 descriu cum poate contribui inteligența artificială la descoperirea științifică, de la generarea ipotezelor la analiza datelor; în fiecare etapă, rezultatul are nevoie de validare independentă.

## Utilizări posibile ale instrumentelor AI {h=97b315dd}

Un asistent AI poate ajuta în patru locuri. Pentru literatură, el poate propune o primă listă de studii despre eficiența și cozile piețelor emergente, fiecare verificat apoi prin DOI. Pentru cod, poate scrie o primă versiune a estimărilor VR(5), Hill și GARCH pe ferestre mobile de 500 de zile.

Pentru robustețe, poate construi o grilă de alegeri, cu fereastra, $k$, nivelul $\alpha$ și seria, și un singur tabel cu toate rezultatele, ceea ce face vizibilă eroarea nestandard a propriei analize. Pentru redactare, poate face un paragraf mai clar, dar numai după ce cifrele sînt finale.

Promptul de pe slide cere cod Python care calculează pe ferestre mobile de 500 de zile de tranzacționare, deplasate cu 21 de zile, statistica robustă $Z^*(5)$ a lui Lo și MacKinlay, tail index-ul Hill al pierderilor cu $k$ egal cu 2,5% din fereastră și persistența GARCH(1,1)-$t$ pentru randamentele logaritmice zilnice ale BET, apoi desenează cele trei serii cu benzile lor de 95%. Promptul precizează fiecare alegere metodologică, astfel încît rezultatul să poată fi verificat.

## Verificări necesare {h=310527b1}

Prima verificare privește formulele: statistica raportului varianțelor trebuie să fie $Z^*(q)$ robustă, nu $Z(q)$, iar estimatorul Hill trebuie aplicat pierderilor $L = -r$, cu $k$ dat ca proporție din fereastră, astfel încît ferestrele să fie comparabile. A doua privește look-ahead bias: fiecare fereastră trebuie să folosească doar datele pînă în ultima ei zi.

A treia privește benzile. Ferestrele mobile se suprapun, deci estimările succesive nu sînt dovezi independente; iar la un nivel de 5%, un rezultat din douăzeci iese semnificativ din întîmplare, așa că o singură fereastră în afara benzii nu dovedește o schimbare.

Ultimele două verificări privesc cifrele și referințele. O fereastră se recalculează pas cu pas și se compară cu rezultatul codului. Fiecare lucrare citată trebuie să existe: se deschide DOI-ul și se verifică titlul.

## Idee de proiect {h=593bca10}

Întrebarea proiectului este dacă BET a devenit mai eficient, mai puțin volatil și cu cozi mai subțiri între 2000 și 2026. Datele sînt seria BET din 1997, cu S&P 500 și DAX (*Deutscher Aktienindex*) ca repere, de la furnizorul EODHD (*EOD Historical Data*).

Pașii folosesc patru capitole: raportul varianțelor robust ([Capitolul 7](ch7_ro.html)), tail index-ul Hill ([Capitolul 5](ch5_ro.html)), persistența GARCH ([Capitolul 9](ch9_ro.html)) și exponentul Hurst ([Capitolul 11](ch11_ro.html)), toate pe ferestre mobile. Fiecare estimare primește o bandă Monte Carlo sau bootstrap, iar aceeași analiză se face pentru repere. Datele de schimbare candidate, de exemplu 2008 sau 2020, se verifică față de benzi și se fixează înainte de inspecția graficelor, nu după.

Livrabilul este un grafic cu patru panouri, un tabel și un paragraf despre ce pot și ce nu pot arăta datele. Orice utilizare a instrumentelor AI se declară, împreună cu erorile lor corectate.

## @end

Întrebarea dacă BET a devenit o piață „obișnuită” rămîne deschisă: momentum-ul a slăbit, dar cozile nu s-au subțiat, iar rezultatele depind de alegerile de metodă. Bucla de cercetare fixează datele înainte de rezultate, cere verificări de robustețe și replicare, iar un asistent AI poate accelera fiecare pas, fără să înlocuiască vreuna dintre verificări.

# Rezumat

Rezumatul strînge ideile principale ale întregului curs și le testează prin trei întrebări de autoevaluare.

## Idei de reținut {h=495a8ff3}

Analiza începe de la randamente, nu de la prețuri: randamente logaritmice în procente, pe calendarul corect al pieței, anualizate cu frecvența reală. Distribuția lor are cozi groase, cu tail index între 2 și 4, asimetrie negativă și varianță finită, așa că distribuția Normală subestimează riscul. Semnul randamentelor este aproape imprevizibil, dar mărimea lor este previzibilă, prin volatility clustering și modele GARCH.

Riscul se măsoară prin VaR 1% $= -q_{1\%}$ și ES 2,5%, iar orice model de risc este urmat de backtesting. Orice rezultat are nevoie de incertitudinea lui: erori standard, teste robuste, benzi Monte Carlo și verificări în afara eșantionului.

Ultima idee privește felul în care se judecă o analiză. Un calcul este complet atunci cînd are metoda, rezultatul și interpretarea; un proiect este bun atunci cînd pune o întrebare clară și îi dă un răspuns onest, cu limitele lui.

## Autoevaluare {h=e6bc0b96}

Prima întrebare: testul Ljung–Box nu respinge pe $r_t$, dar respinge puternic pe $r_t^2$. Ce înseamnă acest lucru? Randamentele sînt necorelate, dar nu independente: varianța lor are structură în timp (volatility clustering), deci varianța trebuie modelată, de exemplu cu GARCH.

A doua întrebare: o lege stabilă estimată dă $\hat\alpha = 1{,}5$, iar estimarea Hill este $\hat\alpha = 3$. Care are dreptate în privința varianței? Estimarea Hill, care privește doar coada: varianța este finită. Legea stabilă descrie corpul distribuției, iar $\alpha$ ei este influențat de forma centrului.

A treia întrebare: un model are 1% depășiri ale VaR 1%, toate în martie 2020. Este un model VaR bun? Nu. Rata este corectă, dar depășirile sînt grupate, exact în perioada în care riscul a contat cel mai mult, iar testul Christoffersen respinge modelul.

## @end

Cursul se reduce la un lanț: date curate, randamente logaritmice, distribuții cu cozi groase, volatilitate previzibilă, VaR 1% și ES 2,5% verificate prin backtesting. Fiecare verigă are instrumentele ei, iar fiecare rezultat se raportează cu incertitudinea și cu interpretarea lui.
