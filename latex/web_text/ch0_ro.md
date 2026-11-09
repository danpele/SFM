<!--
Textul de curs al versiunii web, capitolul 0 (RO). Generat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Hash-urile curente: python3 build_web_lecture.py 0 --hashes
-->

## @intro

@epigraph Fama1970
> The primary role of the capital market is allocation of ownership of the economy's capital stock. In general terms, the ideal is a market in which prices provide accurate signals for resource allocation

@portrait photos/ch0_fama_2013.jpg | Eugene F. Fama | Foto: Bengt Nyman (2013), CC BY 2.0, Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Eugene_Fama_at_Nobel_Prize,_2013.jpg

@text
O piață financiară transformă economiile în capital și repartizează riscul celor dispuși să și-l asume. Face acest lucru prin prețuri, iar fiecare tranzacție lasă în urmă un preț și un moment de timp. Pentru un statistician, piața este de aceea un generator de date: seria prețurilor zilnice de închidere ale unui indice, ale unui curs de schimb sau ale unui activ cripto este materia primă a fiecărui capitol al cursului.

Datele capitolului arată de ce aceste serii cer metode statistice proprii. Din ianuarie 2015 pînă în septembrie 2026, 100 de unități investite în S&P 500 au ajuns la aproximativ 370, iar 100 de unități investite în Bitcoin la aproximativ 25 740, dar volatilitatea anualizată a Bitcoin a fost de 66,7%, față de 17,7% pentru S&P 500. În 2008–2009, S&P 500 a pierdut 56,8% față de vîrful anterior, iar BET, indicele bursei de la București, a pierdut 82,5% și a avut nevoie de peste treisprezece ani ca să revină la vîrful din 2007. În cele 6 714 zile de tranzacționare de după 2000, S&P 500 a avut 46 de variații zilnice mai mari de patru abateri standard, acolo unde distribuția Normală cu aceeași medie și varianță ar prevedea 0,43.

Capitolul prezintă organizarea cursului, rolul piețelor financiare, ce arată datele din 2026, felul în care piețele și modelele lor statistice au evoluat în patru secole și primele instrumente ale cursului: randamentul simplu și cel logaritmic, volatilitatea și anualizarea ei, cozile groase și drawdown-ul.

@chart sfm_ch0_markets
Valoarea a 100 de unități investite în ianuarie 2015 în S&P 500, DAX, BET-TR, aur și Bitcoin, pe o scară verticală logaritmică; indicii în moneda locală, aurul și Bitcoin în dolari americani; închideri zilnice pînă la 18 septembrie 2026.

@goals
- **Explicați** felul în care este organizat și evaluat cursul.
- **Numiți** principalele clase de active și participanții la piață.
- **Citiți** un grafic de creștere, unul risc–randament și unul de drawdown.
- **Calculați** un randament simplu și unul logaritmic din două prețuri.
- **Spuneți** ce trebuie verificat într-o analiză generată de AI.

[Seminarul 0](../RO/Seminarii/seminar0_introducere_ro.pdf) are loc înaintea acestui curs și exersează aceleași instrumente pe date.

@plan
- Organizarea cursului | Evaluarea, manualul, harta Capitolelor 0–16, materialele, seminariile, proiectul de echipă și politica privind AI.
- Importanța piețelor financiare | Funcțiile unei piețe, participanții, clasele de active, registrul de ordine și rolul statisticii.
- Piețele financiare în 2026: statistici descriptive | Creștere, risc și randament, drawdown-uri, VIX, piața din România și activele cripto din 2015.
- O scurtă istorie a piețelor | Primele burse și bule, 1929, tranzacționarea electronică, Bursa de Valori București și modelele statistice ale prețurilor.
- Prețuri și randamente: primele definiții | Randamente simple și logaritmice, media și volatilitatea, anualizarea, cozile groase și drawdown-ul.
- AI pentru descoperire științifică | O întrebare deschisă despre volatilitatea Bitcoin, folosirea asistenților AI și verificările pe care le cer.
- Concluzii | Întrebări de autoevaluare și întrebările Capitolului 1.

# Organizarea cursului

Statistica piețelor financiare aplică probabilitățile și statistica prețurilor, randamentelor și riscurilor activelor tranzacționate. Această secțiune descrie obiectul disciplinei, felul în care este evaluat cursul, manualul pe care îl urmează, ordinea celor șaptesprezece capitole, materialele fiecărui capitol, seminariile și proiectul de echipă, precum și regulile de folosire a instrumentelor de inteligență artificială (AI, *artificial intelligence*).

## Traseul acestui capitol {h=153a62b0}

Capitolul pornește de la trei întrebări: ce măsoară un statistician pe o piață, ce cifre descriu piețele din 2026 și cum se ajunge de la prețuri la randamente. Partea I răspunde primelor două. Ea prezintă organizarea cursului, apoi rolul economic al piețelor financiare și, în final, datele: creștere, risc, drawdown-uri și volatilitate implicită pentru principalele piețe din 2015 și pentru piața din București din 1997.

Partea a II-a așază aceste cifre în istorie și introduce primele instrumente. O scurtă istorie a burselor, a crahurilor și a modelelor arată că întrebările statistice ale cursului s-au născut din episoade concrete. Secțiunea despre randamente definește randamentul simplu și pe cel logaritmic, media, volatilitatea și anualizarea ei, și documentează două regularități ale randamentelor zilnice: volatility clustering și cozile groase. Capitolul se încheie cu o întrebare deschisă despre volatilitatea Bitcoin, care arată cum pot ajuta asistenții AI un proiect de cercetare și ce trebuie verificat în rezultatele lor.

După acest capitol, cititorul poate explica felul în care este organizat și evaluat cursul, poate numi principalele clase de active și participanții la piață, poate citi un grafic de creștere, unul risc–randament și unul de drawdown, poate calcula un randament simplu și unul logaritmic din două prețuri și poate spune ce trebuie verificat într-o analiză produsă de AI. [Seminarul 0](../RO/Seminarii/seminar0_introducere_ro.pdf) are loc înaintea acestui curs și exersează aceleași instrumente pe date.

## Obiectul statisticii piețelor financiare {h=0d13c0e2}

Statistica piețelor financiare reunește metodele statistice pentru prețurile, randamentele și riscurile activelor tranzacționate. Ea are trei sarcini. Descrierea întreabă cît de mari și cît de variabile sînt randamentele. Modelarea întreabă ce distribuție de probabilitate descrie randamentele și cum depind ele de propriul trecut. Măsurarea riscului întreabă cît de mare poate fi pierderea într-o zi nefavorabilă și este sarcina care leagă disciplina de reglementarea băncilor și a fondurilor.

Datele financiare cer metode proprii pentru că încalcă sistematic ipotezele statisticii introductive. Zilele extreme sînt mult mai frecvente decît prevede distribuția Normală: distribuția randamentelor are cozi groase (*heavy tails*). Perioadele calme alternează cu cele agitate, astfel încît randamentele mari tind să fie urmate de randamente mari; fenomenul se numește volatility clustering. În crize, piețele scad simultan, iar corelațiile dintre active cresc tocmai cînd diversificarea ar fi cea mai necesară.

Aceste regularități se repetă de la o piață la alta, de la o perioadă la alta și de la o frecvență la alta și se numesc faptele stilizate (*stylised facts*) ale randamentelor [@Cont2001]. Un model al randamentelor este judecat în primul rînd după cît de bine le reproduce. [Capitolul 2](ch2_ro.html) le studiază sistematic; capitolul de față le arată pe primele două pe S&P 500, în [secțiunea despre randamente](#sec-5).

## Prezentarea cursului {h=564003ca}

Cursul aparține programului de licență Statistică și Data Science, anul III, semestrul 2, la Facultatea de Cibernetică, Statistică și Informatică Economică (CSIE) a Academiei de Studii Economice din București (ASE), în anul universitar 2026/2027. Cursul este predat de Prof. Daniel Traian Pele.

Cursul presupune cunoștințe de probabilități și statistică, algebră liniară și Python. Probabilitățile dau limbajul variabilelor aleatoare și al distribuțiilor în care se modelează randamentele; algebra liniară este necesară pentru portofolii și pentru modelele multivariate; Python este limbajul fiecărui notebook și al fiecărui Quantlet al cursului.

Toate materialele se găsesc pe site-ul cursului, [danpele.github.io/SFM](https://danpele.github.io/SFM/): slide-uri, seminarii, notebook-uri, Quantlets și quiz-uri, în engleză și în română.

## Evaluare {h=2450c432}

Nota finală are trei componente. Examenul scris contează 70%. El cuprinde probleme pe date reale și întrebări scurte de interpretare, de felul celor rezolvate la seminar, și acoperă cursurile și seminariile Capitolelor 0–16.

Proiectul de echipă contează 20%. El pune o întrebare de cercetare pe date reale de piață și îi răspunde cu metodele cursului; livrabilele și criteriile sînt descrise în [Proiectul de echipă](#s12).

Prezența la cursuri și la seminarii și quiz-urile capitolelor contează împreună restul de 10%. Quiz-urile se rezolvă pe site-ul cursului cu contul Google ASE, iar numele și scorul se înregistrează pentru notă.

## Manual și bibliografie suplimentară {h=6840a6ab}

Manualul de bază este @FrankeHardleHafner2019, *Statistics of Financial Markets: An Introduction*, ediția a 5-a, Springer. Capitolele cursului urmează părțile manualului dedicate randamentelor, distribuțiilor, seriilor de timp, volatilității și riscului. Codul din spatele graficelor și exemplelor manualului este public, sub forma [SFE Quantlets](https://github.com/QuantLet/SFE).

Seminariile folosesc culegerea de exerciții care însoțește manualul, @BorakHardleLopezCabrera2013, *Statistics of Financial Markets: Exercises and Solutions*, tot la Springer, în care problemele sînt rezolvate pas cu pas.

Două cărți servesc drept lecturi suplimentare: @Tsay2010, *Analysis of Financial Time Series*, pentru modelele de serii de timp ale randamentelor și ale volatilității, și @CampbellLoMacKinlay1997, *The Econometrics of Financial Markets*, pentru testele econometrice ale eficienței pieței și ale modelelor de evaluare a activelor. Cursul [Statistics of Financial Markets](https://quantinar.com/course/103/statistics-of-financial-markets) de pe Quantinar, o platformă de învățare peer-to-peer, acoperă aceeași materie în format video.

## Harta cursului: Capitolele 0–16 {h=da3cfec6}

Cursul are șaptesprezece capitole, numerotate de la 0 la 16. După această introducere, [Capitolul 1](ch1_ro.html) tratează datele, randamentele și indicatorii de performanță, iar Capitolele 2–6 tratează distribuția randamentelor: distribuțiile clasice și faptele stilizate ([Capitolul 2](ch2_ro.html)), distribuțiile α-stabile ([Capitolul 3](ch3_ro.html)), probabilitatea ([Capitolul 4](ch4_ro.html)), cozile groase și teoria valorilor extreme ([Capitolul 5](ch5_ro.html)), selecția modelului și managementul riscului ([Capitolul 6](ch6_ro.html)).

[Capitolul 7](ch7_ro.html) întreabă dacă piețele sînt eficiente, cu mersul aleator și testele raportului varianțelor (VR, *variance ratio*). Capitolele 8–11 tratează volatilitatea și riscul: estimatorii de volatilitate ([Capitolul 8](ch8_ro.html)), modelele ARCH (AutoRegressive Conditional Heteroskedasticity) și GARCH (Generalised ARCH) ([Capitolul 9](ch9_ro.html)), Value at Risk (VaR), Expected Shortfall (ES) și backtesting-ul ([Capitolul 10](ch10_ro.html)), ipoteza piețelor fractale și memoria lungă ([Capitolul 11](ch11_ro.html)).

Capitolele 12–15 sînt aplicații: modelele de scoring ([Capitolul 12](ch12_ro.html)), învățarea automată ([Capitolul 13](ch13_ro.html)), activele cripto ([Capitolul 14](ch14_ro.html)) și riscul sistemic ([Capitolul 15](ch15_ro.html)). [Capitolul 16](ch16_ro.html) recapitulează cursul. Ordinea urmează logica disciplinei: întîi distribuția unui singur randament, apoi dependența lui de trecut, apoi măsurile de risc construite pe ambele și, în final, folosirea lor pe piețe anume.

## Materiale și instrumente {h=ffbd4075}

Fiecare capitol are slide-uri de curs și de seminar, în engleză și în română, un notebook de curs și unul de seminar în Python, care se deschid în Google Colab cu un singur clic, și un quiz de autoevaluare pe site. Fiecare quiz extrage 20 de întrebări dintr-o bancă de 24 și explică fiecare răspuns.

Fiecare grafic al cursului este un Quantlet: un folder public cu codul care îl produce, un fișier de descriere și graficul însuși. Pictograma de sub fiecare grafic din slide-uri deschide Quantlet-ul corespunzător pe GitHub, astfel încît orice figură poate fi reprodusă și modificată.

Datele de piață sînt date zilnice de sfîrșit de zi (EOD, *end of day*) de la furnizorul EODHD (EOD Historical Data), salvate o singură dată în repository-ul cursului. Ele merg pînă la 18 septembrie 2026, astfel încît orice calculator produce aceleași cifre. Cursul EUR/RON este cursul de referință oficial al Băncii Naționale a României (BNR).

## Seminarii {h=8092b4a0}

Fiecare seminar are loc înaintea cursului său. El începe cu o scurtă introducere care îl face de sine stătător, deci poate fi urmat fără curs. Cursul explică apoi de ce funcționează instrumentele seminarului și unde dau greș.

Seminariile au două tipuri de exerciții. Un exercițiu **[Rezolvat]** are rezolvarea completă în slide-uri și în notebook și servește drept model. Un exercițiu **[Propus]** este rezolvat de studenți, iar rezolvarea lui se discută la seminar.

Temele nu se notează; ele pregătesc examenul și proiectul. În fiecare seminar, un exercițiu prezintă un răspuns generat de AI și cere găsirea greșelii din el, o practică pe care [politica privind AI](#s13) o face parte a cursului.

## Proiectul de echipă {h=22fff197}

Proiectul de echipă pornește de la o întrebare de cercetare despre date reale de piață, aleasă de echipă. Răspunsul trebuie să folosească metodele cursului și să fie susținut de date, așa că întrebarea trebuie să fie suficient de precisă pentru a putea primi un răspuns cu seriile disponibile.

Proiectul are trei livrabile. Primul este un repository GitHub al cărui cod reproduce fiecare cifră și fiecare grafic al proiectului din datele salvate. Al doilea este un raport scurt, însoțit de o prezentare a rezultatelor. Al treilea este fișierul AI_USE.md, care consemnează folosirea instrumentelor AI după regulile din [politica privind AI](#s13).

Criteriile de evaluare sînt claritatea întrebării, corectitudinea metodelor și a verificărilor și onestitatea interpretării. La susținerea orală, fiecare membru al echipei explică codul și rezultatele. Idei de proiect se găsesc la finalul fiecărui capitol, în secțiunea *AI pentru descoperire științifică*; ideea acestui capitol este în [Idee de proiect](#s61).

## Politica privind AI {h=6af9d77c}

Instrumentele AI sînt permise în curs și trebuie declarate. AI înseamnă aici asistenți construiți pe un model lingvistic de mari dimensiuni (LLM, *large language model*), care scriu text și cod pornind de la o cerere. Fiecare utilizare se consemnează în AI_USE.md: instrumentul, cererea, ce s-a păstrat și ce s-a corectat.

Studentul răspunde pentru fiecare cifră și pentru fiecare referință. Asistenții AI fac greșeli tipice: referințe inventate, formule greșite și cod care rulează, dar calculează altceva decît s-a cerut. Fiecare referință se verifică deschizînd DOI-ul (Digital Object Identifier) și comparînd titlul; fiecare cifră se recalculează cu propriul cod, din datele cursului.

Susținerea orală aplică același principiu. O linie de cod pe care autorul ei nu o poate explica nu este considerată muncă proprie.

## Recapitulare: organizarea {h=3f96707e}

Nota combină examenul scris (70%), proiectul de echipă (20%), prezența și quiz-urile (10%). Manualul este @FrankeHardleHafner2019, iar seminariile folosesc culegerea @BorakHardleLopezCabrera2013.

Seminariile preced cursurile, iar temele nu se notează. Instrumentele AI sînt permise, declarate în AI_USE.md și verificate la susținerea orală a proiectului.

## @end

Cursul este construit în jurul unei singure discipline, statistica prețurilor, randamentelor și riscurilor activelor tranzacționate, ale cărei metode sînt motivate de faptele stilizate ale randamentelor. Manualul, Quantlet-urile și datele salvate fac fiecare rezultat reproductibil, iar politica privind AI extinde același standard la munca făcută cu asistenți: fiecare cifră recalculată, fiecare referință deschisă. [Secțiunea următoare](#sec-2) întreabă ce fac piețele financiare și de ce prețurile lor cer statistică.

# Importanța piețelor financiare

Înainte de a măsura randamentele, este util să știm ce le produce. Această secțiune descrie funcțiile economice ale unei piețe financiare, participanții și clasele de active, precum și registrul de ordine, mecanismul prin care se formează prețurile la bursă. Ea se încheie cu perspectiva statistică asupra prețului: prețul de mîine este o variabilă aleatoare, iar cursul îi studiază randamentele, distribuția lor, volatilitatea și măsurile de risc construite pe ele.

## Rolul unei piețe financiare {h=a0592a61}

O **piață financiară** este un loc sau un sistem în care cumpărătorii și vînzătorii schimbă drepturi asupra unor fluxuri de numerar viitoare, cum ar fi acțiuni, obligațiuni, valute, contracte sau tokeni, la prețuri stabilite de cerere și ofertă. Obiectul tranzacției este un drept: o acțiune îi dă deținătorului dreptul la dividendele viitoare, o obligațiune dreptul la cupoane și la principal.

Piețele îndeplinesc cinci funcții. Ele alocă capitalul, aducînd economiile la firme și la state. Împart riscul (*risk sharing*), transferîndu-l celor dispuși să și-l asume. Asigură lichiditatea, astfel încît un activ poate fi transformat repede în bani. Descoperă prețul (*price discovery*), pentru că un preț agregă informația dispersată între mulți participanți. Și finalizează plățile și decontarea, astfel încît tranzacțiile se încheie în siguranță.

Pentru un statistician, piața este un generator de date. Fiecare tranzacție lasă în urmă un preț și un moment de timp, iar șirul acestor prețuri este materialul cursului. Descoperirea prețului este legătura dintre economie și statistică: dacă prețurile agregă informația disponibilă, variațiile lor ar trebui să răspundă doar informației noi, ipoteză pe care o testează [Capitolul 7](ch7_ro.html).

## Participanții la piață {h=919c0417}

Diagrama arată fluxurile dintre trei grupuri. Economisitorii, de exemplu gospodăriile și fondurile de pensii, dau bani intermediarilor, adică băncilor, fondurilor și brokerilor. Intermediarii transmit capitalul emitenților, firme și state, care dau în schimb economisitorilor acțiuni și obligațiuni. Locurile de tranzacționare, adică bursele, dealerii și casele de compensare, sînt locul în care se întîlnesc intermediarii.

Pe partea de cumpărare (buy side) se află investitorii care cumpără active: persoane fizice (investitori retail), fonduri de pensii, asigurători și administratori de fonduri. Investiția lor poate fi pasivă sau activă. O investiție pasivă copiază un indice, de exemplu printr-un ETF (*exchange-traded fund*), un fond ale cărui titluri se tranzacționează ca o acțiune. O investiție activă încearcă să obțină un randament mai mare decît al indicelui.

Pe partea de vînzare (sell side) se află brokerii, care execută ordine, și formatorii de piață (*market makers*), care afișează toată ziua un preț la care cumpără și unul la care vînd. Diferența dintre cele două este plata lor pentru lichiditatea oferită și reapare pe slide-ul următor ca spread-ul bid–ask.

## Clase de active {h=2e0a503f}

Tabelul grupează activele tranzacționate în șase clase, după fluxul de numerar pe care îl plătesc și după riscurile principale. Acțiunile, de exemplu titlurile individuale și ETF-urile pe acțiuni, plătesc dividende și poartă riscul pieței și riscul firmei. Obligațiunile, de exemplu titlurile de stat și obligațiunile corporative, plătesc cupoane și principalul și poartă riscul de dobîndă și riscul de credit, adică riscul ca emitentul să nu plătească.

Valutele (FX, *foreign exchange*) aduc diferențialul de dobîndă dintre cele două monede și depind de curs și de politica monetară; o cotație ca EUR/RON arată cîți lei se plătesc pentru un euro. Mărfurile, de exemplu aurul și petrolul, nu plătesc niciun flux de numerar, iar riscurile lor vin din ofertă și din depozitare. Derivatele, de exemplu contractele futures și opțiunile, fac plăți condiționate de prețul altui activ și poartă riscul efectului de levier și riscul de contrapartidă. Activele cripto, de exemplu Bitcoin și Ethereum, nu plătesc niciun flux de numerar, iar riscurile lor principale sînt volatilitatea și reglementarea.

Clasele diferă din punct de vedere economic, dar setul de instrumente statistice al cursului este același pentru toate: randamente, volatilitate, cozi și dependență. Un randament se poate calcula din orice serie de prețuri, fie că prețul este al unei acțiuni, al unui baril de petrol sau al unui bitcoin.

## Formarea prețului: registrul de ordine {h=64f3f9f5}

La bursă, prețurile se formează în **registrul de ordine** (*order book*), lista tuturor ordinelor de cumpărare (bid) și de vînzare (ask) în așteptare, ordonate după preț. În exemplu, cumpărătorii așteaptă cu 3 unități la 99,99, 5 unități la 99,98 și 4 unități la 99,97; vînzătorii oferă 2 unități la 100,01, 4 unități la 100,02 și 6 unități la 100,03. Barele diagramei arată aceste cantități.

**Spread-ul bid–ask** este cel mai bun ask minus cel mai bun bid, aici $100{,}01 - 99{,}99 = 0{,}02$. El este costul unei tranzacții imediate: un investitor care cumpără la cel mai bun ask și vinde imediat la cel mai bun bid pierde spread-ul. Un **ordin la piață** (*market order*) cumpără sau vinde imediat, la cele mai bune prețuri disponibile, și plătește deci acest cost.

Întrebare pentru sală: sosește un ordin la piață de cumpărare a 5 unități. El preia cele 2 unități oferite la 100,01 și apoi 3 dintre cele 4 unități de la 100,02, deci prețul său mediu este $(2 \times 100{,}01 + 3 \times 100{,}02)/5 = 100{,}016$. După tranzacție, cel mai bun ask este 100,02, iar spread-ul crește la 0,03. Un ordin mare mișcă prețul în defavoarea celui care îl trimite; din cauza acestui impact asupra prețului (*price impact*), lichiditatea se măsoară prin adîncimea registrului, nu doar prin spread.

## Rolul statisticii {h=7a265078}

Prețul de mîine este incert, așa că îl modelăm ca pe o variabilă aleatoare. Dacă $P_t$ este prețul din ziua $t$, șirul $P_1, P_2, \ldots$ este o **serie de timp** (*time series*), o singură traiectorie observată a unui proces stocastic. Statisticianul vede doar această traiectorie și trebuie să deducă din ea proprietățile procesului.

Cursul studiază patru mărimi derivate din prețuri. **Randamentul** este modificarea relativă a prețului ([Capitolul 1](ch1_ro.html)). **Distribuția** lui și, mai ales, cît de groase îi sînt cozile fac obiectul [Capitolelor 2](ch2_ro.html), [3](ch3_ro.html) și [5](ch5_ro.html). **Volatilitatea** este abaterea standard a randamentelor și felul în care aceasta se schimbă în timp ([Capitolele 8](ch8_ro.html) și [9](ch9_ro.html)). **Măsurile de risc** sînt Value at Risk (VaR, valoarea expusă la risc) și Expected Shortfall (ES) ([Capitolul 10](ch10_ro.html)). VaR la nivelul α este opusul cuantilei de ordin α a randamentelor, $\mathrm{VaR}_\alpha = -q_\alpha$: VaR 1% este pierderea depășită în 1% dintre zile.

În spatele acestor mărimi stă o întrebare centrală: pot prețurile trecute prezice randamentele viitoare? Potrivit ipotezei pieței eficiente, aproape deloc, pentru că orice componentă previzibilă ar fi exploatată prin tranzacții și ar dispărea. [Capitolul 7](ch7_ro.html) formulează precis ipoteza și o testează.

## Recapitulare: importanța piețelor financiare {h=0531d9c9}

Piețele alocă capitalul, împart riscul, asigură lichiditatea și descoperă prețurile. La bursă, prețurile se formează în registrul de ordine, iar spread-ul este costul unei tranzacții imediate.

Cele șase clase de active diferă prin fluxurile de numerar și prin riscuri, dar se analizează cu un singur set de instrumente statistice. Prețul de mîine este o variabilă aleatoare, iar statistica îi descrie randamentele, volatilitatea și riscul.

## @end

O piață este o instituție care alocă capitalul și riscul, iar prețurile ei se formează din interacțiunea ordinelor din registru. Obiectul statistic al cursului este seria acestor prețuri și, mai ales, randamentele calculate din ele. [Secțiunea următoare](#sec-3) arată cum arată aceste serii pentru principalele piețe pînă în septembrie 2026.

# Piețele financiare în 2026: statistici descriptive

Această secțiune prezintă datele folosite în tot cursul și primul tablou empiric al piețelor: creșterea unei investiții din 2015, compromisul dintre risc și randament, drawdown-urile crizelor de după 2000, indicele de volatilitate implicită VIX, istoria indicelui BET al bursei din București și locul activelor cripto. Fiecare grafic este urmat de interpretarea lui; definițiile formale ale mărimilor vin în [secțiunea despre randamente](#sec-5).

## Datele folosite în curs {h=f653543e}

Cursul lucrează cu șase serii zilnice. S&P 500 acoperă 500 de firme mari din SUA, iar DAX 40 de firme mari din Germania. BET acoperă cele mai mari firme de la Bursa de Valori București (BVB), iar BET-TR (*total return*) este același indice cu dividendele reinvestite. Aurul este cotat ca XAU/USD, în dolari pe uncie, iar Bitcoin ca BTC/USD. VIX măsoară volatilitatea așteptată a S&P 500.

Fiecare serie folosește prețul zilnic de închidere, pînă la 18 septembrie 2026; sursa este descrisă în [Materiale și instrumente](#s10). Pentru Bitcoin, prețul de închidere este prețul de la ora 24:00 UTC (timpul universal coordonat, *Coordinated Universal Time*), deoarece piața nu are o oră de închidere.

Fiecare serie își păstrează calendarul: indicii sînt observați în zilele de tranzacționare ale propriei burse, aurul în zilele lucrătoare, iar Bitcoin în fiecare zi. Bitcoin se tranzacționează deci 365 de zile pe an, iar bursele aproximativ 252. Diferența contează pentru fiecare cifră anualizată a cursului, cum arată [Media, volatilitatea și anualizarea](#s50).

## Cinci piețe din 2015: evoluția a 100 de unități investite {h=ddff97c2}

Graficul arată valoarea a 100 de unități investite în ianuarie 2015 în fiecare dintre cinci piețe. Indicii sînt măsurați în moneda locală, aurul și Bitcoin în dolari americani. Între început și 18 septembrie 2026, S&P 500 și aurul s-au multiplicat de aproximativ 3,7 ori, DAX de 2,6 ori, BET-TR de 10,4 ori, iar Bitcoin de 257,4 ori.

Axa verticală este logaritmică: distanțe verticale egale corespund unor variații procentuale egale. O dublare de la 100 la 200 ocupă aceeași înălțime ca o dublare de la 10 000 la 20 000. Pe o scară liniară, Bitcoin ar turti celelalte patru serii într-o linie plată lîngă axa orizontală; pe o scară logaritmică, panta fiecărei linii este rata ei de creștere, iar seriile pot fi comparate.

Graficul arată și traiectoria, nu doar valoarea finală. Bitcoin urcă în valuri abrupte, întrerupte de căderi adînci, în jurul anilor 2018 și 2022, în timp ce indicii cresc mai constant, cu o scădere bruscă comună în martie 2020. Multiplul final ascunde aceste pierderi intermediare, pe care le măsoară [graficul drawdown-urilor](#s28).

## Cinci piețe din 2015: cifrele {h=3fc8d14b}

Tabelul rezumă aceleași cinci serii prin trei cifre. $P_T/P_0$ este prețul final împărțit la prețul inițial. Randamentul anual este media randamentelor logaritmice zilnice înmulțită cu $A$, numărul de zile de tranzacționare pe an. Volatilitatea anuală este abaterea standard a randamentelor zilnice înmulțită cu $\sqrt{A}$. Ambele sînt exprimate în procente pe an, iar definițiile lor sînt date în [Media, volatilitatea și anualizarea](#s50).

Bitcoin are cel mai mare randament anual, 47,4%, și o volatilitate de 66,7% pe an, de circa patru ori mai mare decît a indicilor bursieri. În această fereastră, BET-TR a avut un randament anual mai mare decît S&P 500, 20,0% față de 11,2%, cu o volatilitate mai mică, 15,5% față de 17,7%. Aurul a egalat S&P 500 ca randament, cu 11,2% pe an, la o volatilitate de 16,3%. DAX a avut cel mai mic randament dintre cele cinci piețe, 8,1% pe an, cu o volatilitate de 19,1%.

Aceste cifre provin dintr-o singură fereastră de unsprezece ani. O altă dată de început produce alt clasament, pentru că media randamentelor pe cîțiva ani este estimată cu o eroare mare. Volatilitățile sînt mai stabile de la o fereastră la alta decît mediile, diferență explicată în [Randamentele zilnice: interpretare](#s52).

## Risc și randament, 2015–2026 {h=5e2ae60f}

Graficul de dispersie reprezintă fiecare piață printr-un punct. Pe axa orizontală este volatilitatea anualizată, pe o scară logaritmică, iar pe axa verticală media anualizată a randamentelor logaritmice, ambele în procente și calculate din 2015 pînă în septembrie 2026. Pe lîngă cele cinci piețe din slide-urile anterioare, graficul adaugă Nasdaq 100, Nikkei 225, Euro Stoxx 50, indicele polonez WIG20, ETF-ul de obligațiuni TLT și Ethereum.

Un punct mai sus a cîștigat mai mult pe an; un punct mai la dreapta a fluctuat mai mult. Scara logaritmică a axei orizontale păstrează distincți indicii bursieri, cu volatilități în jur de 20%, de activele cripto, ale căror volatilități sînt de cîteva ori mai mari.

## Risc și randament: interpretare {h=f2a2683c}

Indicii bursieri se grupează în jurul unei volatilități de 15–22% pe an. În interiorul grupului, randamentele lor anuale variază între 5,0% pentru WIG20 (Varșovia) și 16,6% pentru Nasdaq 100. Indici cu un risc asemănător au obținut deci randamente destul de diferite în aceiași unsprezece ani.

Activele cripto se află mult mai la dreapta: volatilitatea anuală a Bitcoin este 66,7%, iar a Ethereum 106,4%. TLT, un ETF de titluri de stat americane cu scadența peste 20 de ani, a pierdut bani, cu −1,1% pe an. Obligațiunile pe termen lung pierd din valoare cînd dobînzile cresc, iar dobînzile au crescut puternic în 2022.

Întrebare pentru sală: demonstrează graficul că riscul mai mare aduce randament mai mare? Nu. Punctele sînt medii calculate pe un singur eșantion, iar media randamentelor pe unsprezece ani este estimată cu o eroare mare, cum arată [Capitolul 1](ch1_ro.html). Panta pozitivă sugerată de activele cripto se sprijină pe două puncte, iar dispersia randamentelor între indici cu aceeași volatilitate arată cît de zgomotoasă este fiecare medie.

## Crize și drawdown-uri, 2000–2026 {h=5432971d}

Un grafic de creștere arată unde a ajuns o piață; un grafic de drawdown arată cît putea pierde un investitor pe drum. **Drawdown-ul** din ziua $t$ este pierderea față de cel mai mare preț anterior:

$$ DD_t = \frac{P_t}{\max_{s \le t} P_s} - 1. $$ {#eq:dd}

În @eq:dd, $\max_{s \le t} P_s$ este cel mai mare preț pînă în ziua $t$, inclusiv. Drawdown-ul nu este deci niciodată pozitiv, $DD_t \le 0$, și este egal cu zero în fiecare zi în care prețul atinge un nou maxim. O valoare de −40% înseamnă că prețul se află cu 40% sub vîrful anterior.

Graficul reprezintă $DD_t$ pentru S&P 500 și pentru BET din 2000. Fiecare criză apare ca o vale: linia părăsește nivelul zero la vîrf, coboară pînă la minim și revine la zero abia cînd vechiul vîrf este depășit. Adîncimea văii măsoară pierderea, iar lățimea ei timpul necesar revenirii.

## Drawdown-uri: interpretare {h=c0142c92}

Criza financiară globală din 2008 a produs cea mai adîncă scădere pentru ambii indici. S&P 500 a atins un drawdown de −56,8% pe 9 martie 2009, măsurat față de vîrful din 9 octombrie 2007. BET a atins −82,5% pe 25 februarie 2009, față de vîrful din 24 iulie 2007.

Crizele ulterioare au fost mai puțin adînci. În timpul pandemiei de COVID-19, în 2020, S&P 500 a scăzut cu 33,9%, iar BET cu 34,9% față de vîrfurile anterioare. În timpul creșterii dobînzilor din 2022, S&P 500 a scăzut cu 25,4%, iar BET cu 22,9%.

Întrebare pentru sală: cît timp i-a trebuit fiecărui indice ca să revină la vîrful din 2007? S&P 500 a revenit pe 28 martie 2013; BET abia pe 16 martie 2021, la peste treisprezece ani după vîrf. O pierdere mare cere un cîștig și mai mare pentru revenire: după o scădere de 80%, prețul este o cincime din vîrf și trebuie să crească de cinci ori, adică un cîștig de 400%, ca să revină la el. Adîncimea unui drawdown și durata revenirii se raportează de aceea împreună.

## Indicele fricii: VIX, 2000–2026 {h=fa6c9513}

**VIX** este volatilitatea S&P 500 în următoarele 30 de zile, așteptată de cei care tranzacționează opțiuni pe indice, exprimată în procente pe an. El este calculat de Cboe din prețurile opțiunilor pe S&P 500 [@Whaley2000]. Pentru că prețurile opțiunilor cresc cînd investitorii caută protecție, VIX crește în perioadele de frică, de unde și numele.

VIX diferă de volatilitățile din slide-urile anterioare. Acelea erau volatilități istorice, calculate din randamente trecute; VIX este o volatilitate implicită (*implied volatility*), extrasă din prețuri și orientată spre viitor. Un VIX de 20 corespunde unei volatilități anuale așteptate de 20%, adică aproximativ $20/\sqrt{252} \approx 1{,}26\%$ pe zi.

Graficul arată închiderea zilnică a VIX din 2000. Forma lui este asimetrică: perioade lungi la niveluri joase, întrerupte de vîrfuri scurte și abrupte în crize.

## VIX: interpretare {h=1495713e}

VIX este de obicei calm. Din 2000, media lui este 19,8, iar mediana 17,8. Media se află peste mediană pentru că vîrfurile rare o trag în sus; distribuția VIX este asimetrică la dreapta.

Două vîrfuri au depășit 80. Pe 20 noiembrie 2008, la două luni după falimentul Lehman Brothers, VIX a închis la 80,9. Pe 16 martie 2020, în timpul panicii bursiere provocate de COVID-19, a închis la 82,7. În ambele episoade, piața opțiunilor aștepta o volatilitate anuală de circa patru ori mai mare decît media VIX din 2000.

Volatilitatea evoluează în valuri: valorile mari sînt urmate de valori mari și se diminuează lent. Acest fenomen, volatility clustering, se modelează cu modelele GARCH (Generalised AutoRegressive Conditional Heteroskedasticity) în [Capitolul 9](ch9_ro.html). Pe 18 septembrie 2026, VIX a închis la 14,8, nivelul unei piețe calme.

## Piața din România: BET din 1997 {h=d734714d}

**BET** este indicele de referință al Bursei de Valori București (BVB). El a fost lansat pe 19 septembrie 1997 la 1 000 de puncte. Pe 18 septembrie 2026 se afla la 32 758 de puncte, de aproximativ 32,8 ori valoarea inițială, după 29 de ani.

BET este un indice de preț: nu include dividendele plătite de firmele sale. Multiplul de 32,8 subestimează deci cîștigul unui investitor în acțiuni românești; BET-TR, care reinvestește dividendele, este indicele folosit în comparațiile cu alte piețe.

Scara verticală a graficului este logaritmică, ca în graficul de creștere. Distanțe verticale egale corespund unor variații procentuale egale, așa că primii ani, cînd indicele era sub 1 000 de puncte, sînt la fel de vizibili ca anii recenți, de peste 30 000 de puncte.

## Piața din România: interpretare {h=def24318}

Istoria BET are trei faze. Între 1997 și 1999, o piață tînără și-a pierdut cea mai mare parte din valoare. Între 2000 și 2007, indicele a crescut puternic, înaintea aderării României la Uniunea Europeană (UE), în 2007. În 2008–2009 a suferit un drawdown de −82,5%, urmat de o revenire lentă și de noi maxime în 2025–2026.

Crahurile de la BVB pot fi modelate. @PeleMazurencuMarinescu2012 au estimat modele log-periodice pe BET înainte de 2008, o clasă de modele în care prețul accelerează, cu oscilații de frecvență crescătoare, înaintea unui crah.

În comparațiile cu alte piețe, cursul folosește BET-TR în locul BET. Firmele românești plătesc dividende mari, iar un indice fără ele subestimează cîștigul investitorului, ceea ce explică o parte din randamentul ridicat al BET-TR din [cifrele de după 2015](#s24).

## Activele cripto în 2026 {h=9223f84e}

Bitcoin a fost propus în 2008 de @Nakamoto2008 ca bani electronici fără bancă și se tranzacționează din 2010. De atunci s-a apropiat de finanțele tradiționale. În ianuarie 2024, Securities and Exchange Commission (SEC), autoritatea americană de supraveghere a pieței de capital, a aprobat ETF-urile spot pe Bitcoin, astfel încît Bitcoin poate fi cumpărat printr-un cont obișnuit de brokeraj. Dacă activele cripto devin o clasă alternativă de active este întrebarea studiată de @PeleWesselhofftHardleKolossiatis2023.

Din punct de vedere statistic, activele cripto rămîn totuși o altă lume. Din 2015, volatilitatea Bitcoin a fost de 66,7% pe an, față de 17,7% pentru S&P 500. Bitcoin se tranzacționează 24 de ore din 24, 7 zile din 7, așa că volatilitatea lui zilnică se anualizează cu 365 de zile, nu cu 252.

[Capitolul 14](ch14_ro.html) este dedicat activelor cripto; [secțiunea despre AI pentru descoperire științifică](#sec-6) a acestui capitol întreabă dacă volatilitatea Bitcoin a scăzut după 2024.

## Recapitulare: piețele în 2026 {h=b28f762a}

Din 2015, indicii bursieri au avut o volatilitate de circa 15–22% pe an, iar Bitcoin de circa 66,7%. Drawdown-urile pot fi adînci și lungi: BET a pierdut 82,5% în 2009 și și-a revenit abia în 2021.

Volatilitatea evoluează în valuri, iar VIX a depășit 80 în 2008 și în 2020. Clasamentul piețelor depinde de fereastră, pentru că orice medie a randamentelor este o estimație.

## @end

Datele de după 2015 dau ordinele de mărime ale cursului: volatilități anuale de 15–22% pentru indicii bursieri și de peste 60% pentru activele cripto, drawdown-uri care au atins −56,8% pentru S&P 500 și −82,5% pentru BET în 2009 și o volatilitate implicită care stă cel mai adesea în jur de 18, dar depășește 80 în crize. Ele arată și limitele unei singure ferestre: clasamentul piețelor după randament se schimbă odată cu data de început. [Secțiunea următoare](#sec-4) arată că bulele, crahurile și revenirile lungi sînt la fel de vechi ca bursele.

# O scurtă istorie a piețelor

Întrebările statistice ale cursului nu au fost formulate în abstract. Această secțiune urmărește patru secole de burse, bule și crahuri, de la primele burse din Bruges și Amsterdam pînă la crizele din 2008 și 2020, împreună cu istoria Bursei de Valori București și cu modelele care au adus statistica în finanțe, de la Bachelier în 1900 la Engle în 1982.

## Cronologia piețelor financiare {h=bef85db1}

Axa timpului așază principalele evenimente ale capitolului de la 1600 pînă astăzi. Marcajele verzi sînt burse: acțiunile Companiei Olandeze a Indiilor de Est (VOC) la Amsterdam în 1602, originile Bursei din New York (NYSE) în 1792, bursa din București în 1882 și redeschiderea ei în 1995. Marcajele roșii sînt bule și crahuri: mania lalelelor din 1637, bula South Sea din 1720, Marțea Neagră din 1929, Lunea Neagră din 1987, falimentul Lehman Brothers din 2008 și crahul provocat de COVID-19 în 2020.

Marcajele albastre sînt modele statistice ale prețurilor: Bachelier în 1900, Markowitz în 1952 și Fama în 1970. Anul 2008 marchează și propunerea Bitcoin.

Citite împreună, cele trei culori arată că bursele, crahurile și modelele au evoluat împreună. Fiecare criză a adus date noi și întrebări noi, iar multe dintre modelele cursului au fost propuse după un episod pe care modelele anterioare nu îl puteau explica.

## Primele burse: de la Bruges la New York {h=2e707156}

La Bruges, între secolele XIII și XV, negustorii se întîlneau în fața hanului familiei Van der Beurse; numele familiei a devenit cuvîntul *bourse*, de la care vine și românescul *bursă* [@Gelderblom2013]. Întîlnirile erau o piață a cambiilor și a mărfurilor, nu a acțiunilor.

La Amsterdam, în 1602, Compania Olandeză a Indiilor de Est (VOC) a emis acțiuni pe care oricine le putea cumpăra și revinde. În mai puțin de un deceniu au apărut tranzacționarea continuă, contractele forward și vînzarea în lipsă (*short selling*) [@GelderblomJonker2004], astfel încît majoritatea instrumentelor unei burse moderne existau deja la începutul secolului al XVII-lea.

La Londra, din 1698, la cafeneaua Jonathan's se publicau liste de prețuri [@Neal1990]; aceste liste se numără printre primele serii de timp ale prețurilor titlurilor financiare. La New York, pe 17 mai 1792, 24 de brokeri au semnat Buttonwood Agreement, originea Bursei din New York (NYSE, New York Stock Exchange) [@SyllaWrightCowen2009].

## Primele bule speculative {h=5362110f}

O **bulă speculativă** (*speculative bubble*) este un preț mult peste orice valoare rezonabilă a activului, urmat de o prăbușire. Bulele au apărut aproape odată cu bursele.

În mania lalelelor din Olanda, în 1636–1637, prețurile contractelor pe bulbi de lalele au crescut de multe ori, apoi s-au prăbușit în februarie 1637. Cercetările ulterioare arată un episod mai mic decît legenda, cu puține falimente [@Garber1989; @Goldgar2007]. În 1720, Compania Mississippi la Paris și Compania South Sea la Londra au crescut și s-au prăbușit în cîteva luni. Prețurile au crescut cel mai mult la firmele legate de o inovație reală, comerțul atlantic și asigurările [@FrehenGoetzmannRouwenhorst2013].

Episoadele au ingrediente recurente: o poveste nouă care pare să justifice prețuri mai mari, credit ieftin care finanțează cumpărările și, apoi, vînzări forțate cînd prețurile se întorc. Aceeași succesiune reapare în multe crize ulterioare, iar semnătura ei statistică este o creștere lungă urmată de un drawdown adînc.

## Marțea Neagră, 1929 {h=89d5fdf4}

Între 24 și 29 octombrie 1929, piața din New York s-a prăbușit; 29 octombrie a rămas în istorie ca Marțea Neagră. Crahul a fost urmat de Marea Criză: potrivit datării National Bureau of Economic Research (NBER) din SUA, contracția economiei americane a început în august 1929 și a durat pînă în martie 1933.

Crahul a avut consecințe pentru statistică. Au urmat reguli mai stricte de transparență, care obligă firmele să publice date regulate și auditate, iar aceste reguli au creat o mare parte din datele contabile și de piață folosite astăzi în finanțe.

Crahul arată și că drawdown-urile lungi nu sînt o noutate. BET a avut nevoie de treisprezece ani ca să-și revină după 2008, cum arată [Drawdown-uri: interpretare](#s29); revenirile lungi fac parte din istoria piețelor, nu sînt excepția unei singure piețe.

## Ringul bursei și tranzacționarea electronică {h=3d61bc15}

Din 1867, telegraful bursier (*stock ticker*) a transmis prețurile prin telegraf, tipărite pe benzi de hîrtie. Pentru prima dată, seriile de timp ale prețurilor erau disponibile în afara bursei, investitorilor și cercetătorilor.

În ringul bursei, brokerii strigau ordinele, iar funcționarii înregistrau prețurile pe hîrtie. Tranzacționarea electronică s-a răspîndit între anii 1970 și 1990. Astăzi fiecare ordin și fiecare tranzacție sînt stocate, cu marcaje de timp la nivel de microsecundă, iar algoritmii trimit o mare parte din ordinele de pe bursele importante.

Fiecare schimbare tehnologică a mărit volumul de date și a făcut posibile statistici noi. Randamentele zilnice au putut fi calculate odată ce prețurile de închidere au fost înregistrate sistematic; volatilitatea intrazilnică poate fi măsurată abia de cînd fiecare tranzacție este stocată.

## Bursa de Valori București {h=9f3ab8b6}

Bursa din București s-a deschis pe 1 decembrie 1882, printr-un decret al regelui Carol I. Tranzacționarea s-a oprit în 1948, sub regimul comunist, iar România nu a avut bursă timp de aproape cincizeci de ani.

Bursa de Valori București (BVB) reînființată a ținut prima ședință de tranzacționare pe 20 noiembrie 1995, cu șase companii listate. Indicele BET a pornit în 1997 de la 1 000 de puncte. În 2020, FTSE Russell, un furnizor global de indici, a trecut România la statutul de piață emergentă secundară. În iulie 2023, Hidroelectrica a realizat cea mai mare ofertă publică inițială (IPO, *initial public offering*) din istoria BVB.

Pentru statistician, această istorie are o consecință practică: seriile zilnice ale pieței din România încep abia la sfîrșitul anilor 1990, iar primii lor ani descriu o piață mică și tînără. [Graficul BET](#s33) arată cît de diferiți au fost acești ani de cei următori.

## Istoria statisticii financiare (1/2): mersul aleator și variațiile extreme ale prețurilor {h=c6619aba}

În teza sa de doctorat din 1900, @Bachelier1900 a modelat prețurile obligațiunilor de la Paris ca un **mers aleator** (*random walk*): variațiile prețului sînt independente și urmează distribuția Normală. Modelul a anticipat cu cîteva decenii matematica mișcării browniene folosită în finanțe, iar cele două ipoteze ale lui au devenit reperul față de care au fost testate modelele ulterioare.

În 1963, @Mandelbrot1963 a arătat că variațiile prețului bumbacului au cozi mult mai groase decît permite distribuția Normală. Variațiile extreme erau prea frecvente ca să fie compatibile cu a doua ipoteză a lui Bachelier.

Mandelbrot a propus distribuțiile α-stabile, care păstrează stabilitatea distribuției Normale la însumare, dar permit cozi groase; [Capitolul 3](ch3_ro.html) le studiază. Cele două portrete încadrează cursul: modelul lui Bachelier este reperul, iar obiecția lui Mandelbrot este primul dintre faptele stilizate.

## Istoria statisticii financiare (2/2): risc, eficiență, volatilitate {h=f4e5fc45}

În 1952, @Markowitz1952 a măsurat riscul prin varianța randamentelor și a construit portofolii diversificate care minimizează această varianță pentru un randament așteptat dat. Riscul a devenit o mărime care poate fi estimată din date.

În 1970, @Fama1970 a definit **piața eficientă** (*efficient market*) ca piața în care prețurile reflectă pe deplin toată informația disponibilă. Definiția poate fi testată statistic întrebînd dacă randamentele sînt previzibile, întrebarea [Capitolului 7](ch7_ro.html). În 1973, @BlackScholes1973 au evaluat opțiunile cu o formulă în care volatilitatea este parametrul esențial, singurul care nu se observă direct.

În 1982, @Engle1982 a modelat volatilitatea variabilă în timp, cu modelul ARCH (AutoRegressive Conditional Heteroskedasticity). Acesta este punctul de plecare al [Capitolelor 8](ch8_ro.html)–[10](ch10_ro.html), despre estimatorii de volatilitate, modelele GARCH și măsurile de risc.

## Crize moderne: 1987, 2008, 2020 {h=788757dc}

Pe 19 octombrie 1987, Lunea Neagră, indicele Dow Jones a scăzut cu 22,6% într-o singură zi, iar toate piețele importante au scăzut puternic [@Roll1988]. Sub distribuția Normală cu o volatilitate zilnică realistă, o scădere de această mărime ar fi practic imposibilă; crahul este cel mai clar exemplu de coadă groasă.

Pe 15 septembrie 2008, Lehman Brothers a dat faliment, iar a urmat criza financiară globală. Modelele de risc care presupuneau distribuția Normală a randamentelor au subestimat pierderile, pentru că atribuiau probabilități neglijabile variațiilor care s-au produs.

În martie 2020 a venit crahul provocat de COVID-19. Pe 16 martie 2020, randamentul logaritmic zilnic al S&P 500 a fost −12,8%, cea mai slabă zi din 2000 încoace. Lecția pentru curs este că zilele extreme fac parte din date, nu sînt erori de eliminat: o regulă de curățare care le șterge ar șterge chiar riscul pe care cursul își propune să-l măsoare.

## Recapitulare: o scurtă istorie {h=7a31a659}

Negustorii se întîlnesc la bursă încă din secolele XIII–XV, la Bruges, acțiunile se tranzacționează din 1602, la Amsterdam, iar bulele și crahurile au apărut aproape de la început. Bursa din București a funcționat între 1882 și 1948 și din nou din 1995, iar BET există din 1997.

Modelele statistice s-au succedat: mersul aleator (1900), riscul portofoliului (1952), cozile groase (1963), eficiența (1970) și volatilitatea variabilă în timp (1982). Fiecare criză a arătat că zilele extreme sînt mai frecvente decît prevede distribuția Normală.

## @end

Patru secole de piețe arată același tipar: instrumente și locuri de tranzacționare noi, urmate de bule și crahuri, urmate de date noi și de modele noi. Modelele studiate în curs, de la mersul aleator al lui Bachelier la ARCH-ul lui Engle, au răspuns fiecare unui eșec al modelului anterior, iar eșecul a fost aproape întotdeauna același: variațiile extreme și volatilitatea schimbătoare. [Secțiunea următoare](#sec-5) introduce instrumentele care le măsoară.

# Prețuri și randamente: primele definiții

Prețurile nu se compară direct: un indice la 32 758 de puncte și o acțiune de 50 de lei nu spun nimic unul despre celălalt. Cursul lucrează cu randamente, adică modificările relative ale prețurilor. Această secțiune definește randamentele simple și logaritmice și regulile lor de agregare, media și volatilitatea randamentelor și anualizarea lor, apoi le aplică randamentelor zilnice ale S&P 500 din 2000, care arată volatility clustering și cozi groase. Ea se încheie cu calculul pas cu pas al drawdown-ului.

## Randamente simple și logaritmice {h=d4808598}

Fie $P_t$ prețul de la sfîrșitul zilei $t$; pentru ETF-uri și acțiuni este prețul ajustat pentru dividende și splituri, astfel încît randamentul include dividendul și nu este distorsionat de un split. $P_{t-1}$ este prețul din ziua precedentă. **Randamentul simplu** (*simple return*) al zilei $t$ este

$$ R_t = \frac{P_t}{P_{t-1}} - 1, $$ {#eq:simple}

cîștigul procentual al zilei, adică ceea ce obține efectiv investitorul.

**Randamentul logaritmic** (*log return*) este

$$ r_t = \ln \frac{P_t}{P_{t-1}} = \ln(1 + R_t), $$ {#eq:log}

unde $\ln$ este logaritmul natural; $r_t$ este randamentul compus continuu al zilei. Dezvoltarea Taylor dă $\ln(1+R) = R - R^2/2 + \ldots$, deci $r_t$ este aproape egal cu $R_t$ pentru variații mici și mai mic decît $R_t$ pentru variații mari, indiferent de semnul lor.

Exemplul $P_0 = 100$, $P_1 = 110$, $P_2 = 99$ arată diferența. În prima zi $R_1 = +10\%$ și $r_1 = \ln 1{,}1 = +9{,}53\%$; în a doua zi $R_2 = -10\%$ și $r_2 = \ln 0{,}9 = -10{,}54\%$. Randamentul logaritmic este sub cel simplu în ambele cazuri, iar diferența este mai mare în ziua de scădere.

## Întrebare pentru sală: adunarea randamentelor pe două zile {h=df101630}

Întrebări pentru sală, pe același exemplu $100 \to 110 \to 99$: cît este suma celor două randamente simple și a cîștigat, a pierdut sau a rămas pe loc investitorul în cele două zile? Suma este $R_1 + R_2 = 10\% - 10\% = 0\%$, dar investitorul a pierdut: $99/100 - 1 = -1\%$.

Randamentele simple se compun: randamentul pe două zile este $(1 + R_1)(1 + R_2) - 1 = 1{,}1 \times 0{,}9 - 1 = -1\%$. Randamentele logaritmice se adună: $r_1 + r_2 = -1{,}01\% = \ln(99/100)$, pentru că logaritmul unui produs este suma logaritmilor. Un cîștig de 10% urmat de o pierdere de 10% îl lasă pe investitor sub punctul de plecare, pentru că pierderea se aplică unei baze mai mari.

De aici regula: în timp, randamentele logaritmice se adună, iar randamentele simple se compun. De aceea randamentele logaritmice se folosesc pentru agregarea în timp și în modelele statistice ale cursului, iar randamentele simple atunci cînd contează cîștigul efectiv al investitorului. Între active rolurile se inversează: randamentul simplu al unui portofoliu este suma ponderată a randamentelor simple ale activelor sale. [Capitolul 1](ch1_ro.html) dă detaliile.

## Media, volatilitatea și anualizarea {h=732a60bc}

Din $n$ randamente logaritmice zilnice $r_1, \ldots, r_n$, **media** este $\bar r = \frac{1}{n}\sum_{t=1}^n r_t$, iar **volatilitatea** este abaterea standard (*standard deviation*)

$$ s = \sqrt{\frac{1}{n-1}\sum_{t=1}^n (r_t - \bar r)^2}. $$ {#eq:sd}

În @eq:sd, $s$ este distanța tipică dintre un randament zilnic și media sa. Împărțirea la $n-1$ în loc de $n$ corectează faptul că $\bar r$ este estimată din aceleași date, ceea ce face din $s^2$ un estimator nedeplasat al varianței.

Cu $A$ observații pe an, cifrele anuale sînt

$$ \bar r_{\text{an}} = A\,\bar r, \qquad s_{\text{an}} = \sqrt{A}\, s. $$ {#eq:ann}

Media se înmulțește cu $A$ pentru că randamentele logaritmice se adună în timp. Volatilitatea se înmulțește cu $\sqrt{A}$ pentru că varianțele zilelor independente se adună, deci varianța anuală este $A s^2$. Scalarea volatilității se sprijină pe această ipoteză de independență, iar volatility clustering este un prim motiv de prudență. Pentru burse $A \approx 252$, pentru Bitcoin $A = 365$.

Exemplele arată ordinele de mărime. Un indice bursier cu o volatilitate zilnică de 1% are o volatilitate anuală de $\sqrt{252} \times 1\% = 15{,}9\%$. Bitcoin, cu o volatilitate zilnică de 3%, are $\sqrt{365} \times 3\% = 57{,}3\%$ pe an; folosirea a 252 de zile, ca pentru o bursă, ar da un rezultat greșit, de 47,6%.

## Randamentele zilnice ale S&P 500, 2000–2026 {h=5ab0e2b5}

Graficul reprezintă cele 6 714 randamente logaritmice zilnice ale S&P 500 de la 4 ianuarie 2000 la 18 septembrie 2026, cîte o linie verticală pentru fiecare zi. Axa orizontală este timpul, iar axa verticală randamentul logaritmic al zilei, în procente.

Două trăsături se văd înaintea oricărui calcul. Randamentele fluctuează în jurul lui zero, fără o tendință vizibilă, deci media este mică în comparație cu variațiile zilnice. Amplitudinea fluctuațiilor nu este constantă: există perioade calme, cu benzi înguste, și perioade agitate, cu benzi largi, iar benzile largi coincid cu crizele din [graficul drawdown-urilor](#s28).

## Randamentele zilnice: interpretare {h=3617cd68}

Randamentele zilnice sînt mici în medie și mari în crize. Din 2000, media randamentelor logaritmice zilnice ale S&P 500 este 0,025%, iar abaterea standard este 1,21% pe zi, adică 19,2% pe an. Cea mai slabă zi a fost 16 martie 2020, cu −12,8%; cea mai bună a fost 13 octombrie 2008, cu +11,0%.

Variațiile mari apar grupat, în 2002, 2008–2009, 2020, 2022 și aprilie 2025. Cele mai bune și cele mai slabe zile sînt apropiate în timp, pentru că ambele aparțin perioadelor agitate. Acesta este volatility clustering: mărimea randamentului de azi spune ceva despre mărimea celui de mîine, chiar dacă semnul lui rămîne imprevizibil.

Media se vede greu. Media zilnică este de aproximativ 49 de ori mai mică decît abaterea standard zilnică, astfel încît semnalul este acoperit de zgomot. Eroarea standard a mediei a $n$ randamente independente este $s/\sqrt{n}$, iar pentru a o înjumătăți este nevoie de un eșantion de patru ori mai mare. Estimarea randamentelor așteptate cere deci eșantioane lungi, în timp ce estimarea volatilității este mult mai ușoară, pentru că fiecare randament ridicat la pătrat aduce informație despre varianță.

## Randamentele față de distribuția Normală {h=89febe70}

Graficul compară histograma randamentelor logaritmice zilnice ale S&P 500 din 2000 cu densitatea distribuției Normale cu aceeași medie și varianță, desenată cu roșu. Axa orizontală este randamentul logaritmic zilnic, în procente, iar axa verticală densitatea, pe o scară logaritmică.

Scara verticală logaritmică face comparația informativă. Pe o scară liniară, cozile ar fi invizibile; pe o scară logaritmică, densitatea Normală este o parabolă cu ramurile în jos, iar cozile histogramei pot fi comparate cu ea. Barele care se întind dincolo de curba roșie, pînă la −12,8% și +11,0%, sînt zile pe care distribuția Normală practic le exclude.

## Cozi groase: interpretare {h=c1592a74}

S&P 500 are mult mai multe zile extreme decît permite distribuția Normală. Din 2000 au fost 46 de zile cu o variație mai mare de 4 abateri standard, în timp ce numărul așteptat sub distribuția Normală cu aceeași medie și varianță este 0,43. Cea mai slabă zi, 16 martie 2020, se află la 10,5 abateri standard sub medie.

**Coeficientul de boltire** (*kurtosis*) rezumă greutatea cozilor:

$$ K = \frac{\frac{1}{n}\sum_{t=1}^n (r_t - \bar r)^4}{s^4}. $$ {#eq:kurt}

În @eq:kurt, $(r_t - \bar r)/s$ este distanța față de medie, măsurată în abateri standard, iar $K$ este media puterii a patra a acestei distanțe. Puterea a patra face ca zilele extreme să domine $K$. Pentru distribuția Normală $K = 3$; pentru S&P 500 din 2000, $K = 13{,}7$. O valoare $K > 3$ indică cozi mai groase decît la distribuția Normală.

Histograma este și mai înaltă în centru decît densitatea Normală: există mai multe zile liniștite și mai multe zile extreme, dar mai puține zile moderate. O măsură de risc bazată pe distribuția Normală cu aceeași varianță subestimează deci probabilitatea pierderilor mari. [Capitolele 2](ch2_ro.html), [5](ch5_ro.html) și [10](ch10_ro.html) construiesc distribuții și măsuri de risc care țin seama de cozi.

## Drawdown-ul, pas cu pas {h=8e27571f}

Tabelul calculează drawdown-ul pentru șase prețuri. **Vîrful curent** (*running peak*) este $M_t = \max_{s \le t} P_s$, cel mai mare preț din zilele $s = 0, \ldots, t$, iar drawdown-ul, definit în @eq:dd, este $DD_t = P_t/M_t - 1$. Prețul crește de la 100 la 120, deci vîrful îl urmează și drawdown-ul este zero; scade la 90, deci $DD_2 = 90/120 - 1 = -25\%$; revine la 110, tot sub vîrf, deci $DD_3 = -8{,}3\%$; atinge un nou vîrf la 130, unde drawdown-ul revine la zero; și scade la 104, deci $DD_5 = 104/130 - 1 = -20\%$.

**Drawdown-ul maxim** (MDD, *maximum drawdown*) este cea mai mică valoare a lui $DD_t$ din eșantion, aici −25%, de la 120 la 90. El este pierderea investitorului care a cumpărat în cel mai nefavorabil moment, la vîrf, și a vîndut la minim.

Investitorii urmăresc drawdown-ul maxim pentru că măsoară cel mai dureros episod, pe care volatilitatea nu îl arată. El nu arată însă cît a durat revenirea, așa că trebuie raportat împreună cu durata drawdown-ului, ca în cazul BET după 2008.

## Recapitulare: prețuri și randamente {h=b32e92ee}

Randamentul simplu $R_t = P_t/P_{t-1} - 1$ se compune în timp; randamentul logaritmic $r_t = \ln(P_t/P_{t-1})$ se adună în timp. Media se anualizează cu $A$, iar volatilitatea cu $\sqrt{A}$, cu $A = 365$ pentru Bitcoin.

Randamentele zilnice arată volatility clustering și cozi groase, cu un coeficient de boltire de 13,7, față de 3 pentru distribuția Normală. Drawdown-ul este pierderea față de vîrful anterior, iar drawdown-ul maxim rezumă cel mai sever episod.

## @end

Obiectul cursului sînt randamentele, nu prețurile. Randamentele logaritmice se adună în timp, cele simple se compun, iar anualizarea înmulțește media cu $A$ și volatilitatea cu $\sqrt{A}$, cu $A = 365$ pentru Bitcoin. Pe S&P 500 din 2000 apar deja două fapte stilizate: volatility clustering, vizibil în seria de timp, și cozile groase, cu 46 de zile dincolo de patru abateri standard și un coeficient de boltire de 13,7. [Secțiunea următoare](#sec-6) folosește aceste instrumente pentru o întrebare de cercetare deschisă.

# AI pentru descoperire științifică

Fiecare capitol al cursului se încheie cu o întrebare de cercetare deschisă, la care metodele capitolului pot contribui, împreună cu modurile în care poate ajuta un asistent AI, verificările pe care le cer rezultatele lui, un scurt studiu de caz pe datele cursului și o idee de proiect. Întrebarea acestui capitol este dacă Bitcoin a devenit mai puțin volatil de cînd a intrat în finanțele tradiționale.

## O întrebare de cercetare deschisă {h=8fefff3c}

Întrebarea este dacă Bitcoin a devenit mai puțin volatil de cînd a intrat în finanțele tradiționale. Ea contează pentru că un Bitcoin mai puțin volatil ar fi mai ușor de deținut în portofoliul unui fond de pensii sau al unei bănci. ETF-urile spot pe Bitcoin aprobate în ianuarie 2024 au deschis Bitcoin investitorilor care cumpără prin conturi obișnuite de brokeraj, iar acești investitori pot tranzacționa altfel decît deținătorii de dinainte.

Întrebarea este deschisă din trei motive. După 2024 există doar cîțiva ani de date, deci orice comparație se sprijină pe un eșantion scurt. Există alte explicații pentru o scădere a volatilității: o piață mai calmă în ansamblu sau o piață cripto mai mare și mai lichidă. Iar cercetări conexe arată că activele cripto și cele tradiționale se apropie în timp [@PeleWesselhofftHardleKolossiatis2023], deci o scădere a volatilității ar putea face parte dintr-o tendință mai lungă, nu să fie un efect al ETF-urilor.

## Utilizări posibile ale instrumentelor AI {h=642f6ce3}

Un asistent AI poate ajuta în patru etape ale proiectului. Pentru literatură, i se pot cere articole din reviste cu evaluare colegială (*peer review*) despre volatilitatea Bitcoin, cu DOI-urile lor; fiecare DOI trebuie apoi deschis, pentru că asistenții AI inventează referințe plauzibile. Pentru ipoteze, i se pot cere trei explicații concurente ale unei scăderi a volatilității, fiecare trebuind să prezică ceva ce celelalte nu prezic, astfel încît datele să poată decide între ele.

Pentru cod, poate scrie o primă variantă care calculează volatilitatea pentru fiecare an; anualizarea, calendarul și datele trebuie apoi verificate linie cu linie. Pentru critică, i se poate cere să joace rolul unui recenzent ostil al rezultatului, care enumeră punctele slabe ale argumentului.

Împărțirea muncii este cea descrisă de @LudwigMullainathan2024: învățarea automată poate genera ipoteze, dar testarea lor rămîne sarcina cercetătorului.

## Mini-studiu de caz: volatilitatea pe ani {h=64f8bf57}

Graficul arată volatilitatea anualizată a randamentelor logaritmice zilnice în fiecare an calendaristic, pentru Bitcoin și pentru S&P 500. Bitcoin este anualizat cu 365 de zile, iar S&P 500 cu frecvența proprie de tranzacționare; bara pentru 2026 acoperă anul pînă la 18 septembrie.

Volatilitatea Bitcoin a fost în medie de 69,6% pe an în 2015–2023 și de 47,3% pe an în 2024–2026. În 2025, ea a fost de 2,25 ori volatilitatea S&P 500. Barele S&P 500 arată anii calmi și pe cei agitați ai pieței de acțiuni, cu cea mai înaltă bară în 2020.

Graficul sugerează o scădere, dar nu răspunde încă la întrebare. Volatilitățile anuale ale Bitcoin variază puternic de la un an la altul și înainte de 2024, iar S&P 500 are propriii ani calmi și agitați. Slide-ul următor enumeră verificările necesare înainte de a afirma o scădere.

## Verificări necesare {h=1eb95ab5}

Comparația trebuie stabilită înainte de a privi graficul. Data de ruptură este aprobarea ETF-urilor din ianuarie 2024, nu anul care arată cel mai bine pe grafic. Dacă se încearcă multe date de ruptură și se păstrează cea mai bună, rezultatul sînt descoperiri false, problemă pe care @HarveyLiuZhu2016 o documentează pentru numeroșii factori propuși pentru explicarea randamentelor așteptate.

Explicația concurentă trebuie exclusă. Dacă toate piețele au fost mai calme după 2024, a scăzut și volatilitatea S&P 500; raportul dintre volatilitatea Bitcoin și cea a S&P 500 elimină această componentă comună. Cifrele trebuie verificate: 365 de zile pe an pentru Bitcoin și aceeași definiție a volatilității în fiecare an. Un an este un eșantion scurt, iar un singur an calm nu este o tendință.

În final, raportul trebuie să spună ce nu poate arăta rezultatul. O scădere a volatilității după 2024 nu dovedește că ETF-urile au cauzat-o; este o comparație între două perioade, nu o estimare cauzală.

## Idee de proiect {h=1845f55b}

Întrebarea proiectului este dacă volatilitatea Bitcoin, raportată la S&P 500, este mai mică după ianuarie 2024 decît înainte. Pașii decurg din verificări. Reproduceți graficul mini-studiului de caz din datele cursului. Scrieți ipoteza, data de ruptură și testul înainte de orice calcul.

Comparați varianțele randamentelor zilnice înainte și după ruptură cu un test de egalitate a varianțelor, testul Levene, mai puțin sensibil la cozile groase decît testul F clasic. Repetați apoi analiza cu Ethereum, ca al doilea activ cripto, pentru a vedea dacă rezultatul este specific Bitcoin.

Raportul prezintă cele două volatilități, raportul lor, rezultatul testului și cel puțin o limită a analizei. Fiecare prompt AI care a influențat o decizie se consemnează în AI_USE.md.

## @end

Întrebarea deschisă a capitolului, dacă Bitcoin a devenit mai puțin volatil după ianuarie 2024, poate fi abordată doar cu instrumentele acestui capitol: randamente logaritmice zilnice, volatilitate anualizată cu calendarul corect și un raport față de S&P 500. Verificările contează mai mult decît calculul: o dată de ruptură fixată dinainte, o explicație concurentă eliminată și precizarea clară că o comparație între două perioade nu este o estimare cauzală.

# Concluzii

Capitolul se încheie cu întrebări de autoevaluare despre principalele sale instrumente și cu întrebările la care răspunde [Capitolul 1](ch1_ro.html).

## Autoevaluare și prezentarea Capitolului 1 {h=fa1696e6}

Întrebările de autoevaluare acoperă cele patru instrumente ale capitolului. De ce suma randamentelor simple nu este randamentul total? Cum se anualizează volatilitatea zilnică a Bitcoin? Ce înseamnă pentru un investitor un drawdown maxim de −80%? Ce trebuie verificat înainte de a folosi o referință generată de AI? Fiecare răspuns se află în acest capitol: în [Întrebare pentru sală: adunarea randamentelor pe două zile](#s48), [Media, volatilitatea și anualizarea](#s50), [Drawdown-uri: interpretare](#s29) și [Politica privind AI](#s13).

[Capitolul 1](ch1_ro.html), despre date, randamente și indicatori, continuă de aici. El întreabă de unde vin datele, ce coloană de preț folosim, cum se agregă randamentele pe zile, luni și ani, ce este raportul Sharpe și ce înseamnă volatility drag.

## @end

O piață financiară alocă capitalul și riscul prin prețuri, iar prețurile ei formează serii de timp ale căror randamente sînt obiectul cursului. Datele de după 2015 dau ordinele de mărime: volatilități anuale de 15–22% pentru indicii bursieri și de peste 60% pentru activele cripto, drawdown-uri de pînă la −82,5% și reveniri care pot dura peste un deceniu. Randamentele zilnice ale S&P 500 arată deja cele două fapte stilizate care modelează restul cursului, volatility clustering și cozile groase. [Capitolul 1](ch1_ro.html) precizează definițiile randamentelor și ale indicatorilor.
