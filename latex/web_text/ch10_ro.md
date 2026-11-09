<!--
Textul de curs al versiunii web, capitolul 10 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie → Autor (an), [@Cheie] → (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 10 --hashes
-->

## @intro

@text
Un manager de risc trebuie să răspundă în fiecare zi la două întrebări: cît poate pierde portofoliul mîine și a fost corect răspunsul de ieri? Valoarea expusă la risc (VaR, *Value at Risk*) răspunde la prima întrebare printr-o cuantilă a distribuției pierderilor, pierderea așteptată în coadă (ES, *expected shortfall*) prin pierderea medie de dincolo de această cuantilă, iar backtesting-ul răspunde la a doua, comparînd prognozele cu pierderile care au avut loc efectiv. Din 1996, aceste trei instrumente sînt și limbajul reglementării bancare: regulile Basel stabilesc capitalul pentru riscul de piață din VaR-ul sau ES-ul calculat de bancă și îl măresc atunci cînd modelul nu trece backtesting-ul.

Capitolul construiește instrumentele pe date zilnice din 2000–2026 și le judecă în două crize, criza financiară din 2008 și crahul COVID-19 din 2020. Pentru S&P 500, VaR 1% istoric este 3,46%, iar ES 2,5% este 3,76%, în timp ce o distribuție Normală ajustată la aceleași randamente dă 2,80% și 2,81%. Între septembrie 2008 și martie 2009, un VaR 1% Normal pe fereastră mobilă pentru S&P 500 a fost depășit în 21 din 146 de zile, acolo unde un model corect ar fi produs aproximativ 1,5 depășiri; un VaR construit pe o prognoză GARCH a volatilității a fost depășit în 5 zile.

Capitolul reunește astfel cele două direcții anterioare ale cursului: cozile din [Capitolul 5](ch5_ro.html) și prognozele volatilității din [Capitolul 9](ch9_ro.html). El arată că alegerea distribuției cozii contează mai puțin decît urmărirea volatilității de azi, că o corelație subestimează riscul prăbușirilor simultane și că un an de date separă doar aproximativ un model de risc bun de unul prost.

@chart sfm_ch10_rolling_var
Randamentele logaritmice zilnice ale S&P 500 (%) în 2007–2009 (stînga) și 2019–2021 (dreapta), împreună cu minus prognozele VaR 1% ale a patru metode: simularea istorică (HS) și distribuția Normală pe o fereastră de 500 de zile, GARCH-t și simularea istorică filtrată (FHS); punctele marchează zilele în care VaR HS a fost depășit.

@goals
- **Definiți** VaR și ES la o probabilitate a cozii $\alpha$ (VaR 1%, ES 2,5%) și calculați-le pentru distribuția Normală și pentru Student-t.
- **Enunțați** axiomele unei măsuri de risc coerente și arătați printr-un contraexemplu că VaR nu este subaditiv.
- **Estimați** VaR și ES prin simulare istorică, metode parametrice, Cornish–Fisher, EVT, GARCH și simulare istorică filtrată.
- **Calculați** VaR-ul unui portofoliu și explicați de ce corelația nu măsoară riscul prăbușirilor simultane.
- **Testați** un model VaR (backtesting) cu testele Kupiec și Christoffersen și cu semaforul Basel.

@plan
- Măsurarea riscului din coadă | Raportul de la ora 16:15, RiskMetrics, regulile Basel și cele două crize folosite ca teste de stres.
- VaR și ES: definiții | Pierderi, orizont și nivel; VaR ca o cuantilă, ES ca o medie a cozii; formulele explicite pentru distribuția Normală și Student-t; regula rădăcinii pătrate a timpului.
- Măsuri de risc coerente | Cele patru axiome, un contraexemplu în care VaR penalizează diversificarea și elicitabilitatea.
- Estimarea VaR și ES | Simularea istorică și precizia ei, metodele Normală, Student-t, Cornish–Fisher și EVT pe șase serii.
- VaR condiționat: GARCH și simularea istorică filtrată | VaR condus de volatilitatea de azi, prognoze pe fereastră mobilă și crizele din 2008 și 2020.
- VaR de portofoliu și dependența | Metoda varianță–covarianță, corelația în crize, dependența în cozi și copulele.
- Backtesting | Secvența depășirilor, testele Kupiec și Christoffersen, semaforul Basel și backtesting-ul pentru ES.
- AI pentru descoperire științifică | O întrebare deschisă despre dependența în cozi dintre București și piețele mondiale, rolul AI și verificările.
- Rezumat | Rezultatele principale, formulele și întrebări de autoevaluare.

# Măsurarea riscului din coadă

Secțiunea arată cum a ajuns un singur număr zilnic să rezume riscul unei bănci. Ea pornește de la cererea președintelui J.P. Morgan pentru un raport zilnic, urmărește VaR de la documentul RiskMetrics din 1994 pînă la regulile de capital Basel din 1996 și 2019 și prezintă cele două crize, 2008 și 2020, în care este testată fiecare metodă a capitolului.

## Raportul de la ora 16:15 și RiskMetrics {h=70628c78}

În jurul anului 1990, Dennis Weatherstone, președintele J.P. Morgan, a cerut un singur număr, livrat în fiecare zi la ora 16:15, după închiderea tranzacționării, care să spună cît poate pierde banca a doua zi. Birourile de tranzacționare ale băncii aveau poziții în obligațiuni, valute, acțiuni și instrumente derivate, iar niciun raport separat nu le putea aduna. Răspunsul a devenit **valoarea expusă la risc** (VaR): o pierdere depășită doar cu o probabilitate mică pe un orizont dat.

În octombrie 1994, banca și-a publicat gratuit metodologia și datele necesare aplicării ei, sub numele RiskMetrics; a patra ediție a documentului tehnic este @JPMorganReuters1996. Metoda era intenționat simplă: randamente Normale, o volatilitate estimată printr-o medie mobilă ponderată exponențial (EWMA, *exponentially weighted moving average*, estimatorul din [Capitolul 8](ch8_ro.html)) și un VaR la o probabilitate a cozii de 5% pe o zi.

Publicarea a transformat VaR într-un limbaj comun. Băncile, autoritățile de reglementare și administratorii de fonduri își puteau exprima riscul de piață în aceleași unități, o pierdere în bani la o probabilitate dată, și puteau compara poziții care nu aveau altceva în comun. Simplitatea care a făcut metoda populară conține și cele două puncte slabe ale ei, pe care capitolul le examinează pe rînd: distribuția Normală are o coadă prea subțire, iar o cuantilă nu spune nimic despre mărimea pierderilor de dincolo de ea.

## RiskMetrics și reglementările Basel {h=8852f2e6}

Comitetul de la Basel pentru supraveghere bancară (BCBS, *Basel Committee on Banking Supervision*), găzduit de Banca Reglementelor Internaționale (BIS, *Bank for International Settlements*) din Basel, stabilește standardele internaționale pentru capitalul băncilor ([istoria Comitetului de la Basel](https://www.bis.org/bcbs/history.htm)). În 1996, comitetul a permis băncilor să calculeze capitalul pentru riscul de piață cu propriile modele VaR [@BCBS1996a]. Regula folosește VaR 1% pe 10 zile, înmulțit cu un factor de cel puțin 3, iar factorul crește dacă modelul nu trece backtesting-ul, adică dacă pierderile efective depășesc VaR prea des [@BCBS1996b].

În 2019, revizuirea fundamentală a portofoliului de tranzacționare (FRTB, *Fundamental Review of the Trading Book*) a înlocuit VaR 1% cu ES 2,5% ca bază a cerinței de capital [@BCBS2019]; regula în vigoare este capitolul MAR33 din Cadrul Basel [@BCBS2023b]. Schimbarea răspunde principalei critici aduse VaR: o cuantilă ignoră cît de mari sînt pierderile de dincolo de ea, în timp ce ES face media lor.

Backtesting-ul nu a trecut la ES. Potrivit capitolului MAR32 din Cadrul Basel [@BCBS2023a], băncile numără în continuare zilele în care pierderea a depășit prognoza VaR, la nivelul băncii și al fiecărui birou de tranzacționare. Motivul este statistic și apare în [secțiunea despre măsurile de risc coerente](#sec-3): o prognoză VaR poate fi verificată singură față de pierderile realizate, în timp ce o prognoză ES poate fi verificată doar împreună cu o prognoză VaR.

## Crizele din 2008 și 2020 {h=4e38c0d7}

Fotografiile arată cele două crize care încadrează capitolul: sediul Lehman Brothers din New York pe 15 septembrie 2008, ziua falimentului, și districtul financiar pustiu al New York-ului pe 25 martie 2020, în timpul carantinei COVID-19. Cea mai proastă zi a S&P 500 din 2000 încoace a venit în a doua criză: un randament logaritmic zilnic de −12,8% pe 16 martie 2020.

Un model Normal cu volatilitatea de selecție îi atribuie unei asemenea zile o probabilitate practic nulă; [Capitolul 5](ch5_ro.html) a măsurat cît de aproape de zero. O măsură de risc construită pe acest model ar fi fost inutilă exact atunci cînd era nevoie de ea.

De aceea, capitolul folosește perioadele septembrie 2008 – martie 2009 și februarie – mai 2020 ca teste de stres pentru fiecare metodă. O metodă care dă cifre rezonabile în anii calmi, dar eșuează în aceste două perioade, nu măsoară riscul din coadă; ea măsoară variația obișnuită.

## @end

VaR s-a născut ca răspuns practic la o întrebare de conducere și a devenit standard de reglementare în cîțiva ani: Basel 1996 a calculat capitalul pentru riscul de piață din VaR 1% pe 10 zile, iar FRTB din 2019 l-a înlocuit cu ES 2,5%, păstrînd numărarea depășirilor VaR ca backtesting. [Secțiunea următoare](#sec-2) definește precis ambele măsuri și le calculează pentru S&P 500.

# VaR și ES: definiții

Secțiunea definește VaR și ES ca funcții de distribuția profitului și pierderii, le calculează întîi pe un eșantion de 100 de zile și apoi pe 26 de ani de randamente S&P 500, deduce formulele explicite pentru distribuția Normală și pentru Student-t și examinează regula care transformă un VaR pe o zi într-un VaR pe zece zile.

## Pierderi, orizont și nivel {h=cb960064}

O măsură de risc are nevoie de trei ingrediente. Primul este variabila aleatoare $X$: profitul sau pierderea (P&L, *profit and loss*) ori randamentul unei poziții pe un orizont $h$; în acest capitol $h$ este o zi, iar $X$ este randamentul logaritmic zilnic, în procente. Pierderea este $L = -X$, un număr pozitiv cînd poziția pierde bani. Ambele convenții apar în literatură, iar cele mai multe erori practice vin din amestecarea lor.

Al doilea ingredient este nivelul $\alpha$, probabilitatea cozii, de exemplu $\alpha = 1\%$ sau $\alpha = 2{,}5\%$. Cursul scrie **VaR 1%** și **ES 2,5%**: numărul din denumire este întotdeauna probabilitatea cozii. Textele de reglementare și programele software vorbesc adesea de „VaR 99%”, numind nivelul de încredere complementar; cele două denumiri descriu același număr, dar cursul folosește doar probabilitatea cozii, astfel încît un număr mai mic înseamnă întotdeauna o pierdere mai extremă.

Al treilea ingredient este cuantila. Cuantila de ordin $\alpha$ a lui $X$ este $q_\alpha(X) = \inf\{x : P(X \le x) > \alpha\}$, definiția din [Capitolul 2](ch2_ro.html), valabilă și pentru distribuțiile discrete. Pentru o funcție de repartiție continuă $F$, ea este simplu $q_\alpha = F^{-1}(\alpha)$. Pentru $\alpha$ mic, cuantila randamentelor este un număr negativ: randamentul sub care se află cele mai proaste $\alpha$ dintre zile.

## Valoarea expusă la risc (VaR) {h=30b8741b}

Valoarea expusă la risc la nivelul $\alpha$ este opusul cuantilei de ordin $\alpha$ a P&L-ului:

$$ \mathrm{VaR}_\alpha(X) = -q_\alpha(X). $$ {#eq:var}

Semnul minus transformă cuantila într-o pierdere pozitivă. Din definiție rezultă $P(L > \mathrm{VaR}_\alpha) \le \alpha$: pierderea depășește VaR cu probabilitatea cel mult $\alpha$. Un VaR 1% zilnic de 3% spune, așadar, că o pierdere mai mare de 3% apare cam într-o zi din o sută. În bani, o poziție cu valoarea $W$ are un VaR de $W \times \mathrm{VaR}(\%)/100$.

La fel de important este ce nu spune VaR. El nu spune nimic despre mărimea pierderilor de dincolo de el: două poziții cu aceeași cuantilă de 1%, dintre care una pierde 4% în zilele proaste, iar cealaltă 40%, au același VaR. Iar VaR nu este pierderea maximă. Prin construcție, el este depășit în aproximativ o fracțiune $\alpha$ dintre zile; un VaR care nu este depășit niciodată nu este un VaR prudent, ci unul greșit, cum arată testele din [secțiunea Backtesting](#sec-7).

## Expected shortfall (ES) {h=fddf9adf}

Pierderea așteptată în coadă răspunde la întrebarea pe care VaR o lasă deschisă: cît de mare este pierderea în zilele proaste? Ea se definește ca media valorilor VaR la toate nivelurile sub $\alpha$:

$$ \mathrm{ES}_\alpha(X) = -\frac{1}{\alpha}\int_0^\alpha q_u(X)\,du, $$ {#eq:es}

unde variabila de integrare $u$ parcurge nivelurile cozii de la 0 la $\alpha$. Pentru o distribuție continuă, definiția se reduce la $\mathrm{ES}_\alpha = -E[X \mid X \le q_\alpha(X)]$: pierderea medie din cele mai proaste $\alpha$ dintre zile. Aceeași mărime apare și sub alte denumiri, VaR condiționat (CVaR, *conditional VaR*), tail VaR (TVaR) sau expected tail loss; pentru distribuțiile continue ele coincid.

Două proprietăți decurg din definiție. Deoarece cuantilele $q_u$ cu $u < \alpha$ se află toate sub $q_\alpha$, media lor se află și ea sub $q_\alpha$, deci $\mathrm{ES}_\alpha \ge \mathrm{VaR}_\alpha$ la același nivel. Iar ES privește în interiorul cozii: două distribuții cu același VaR, dar cu cozi diferite, una care se termină brusc și una cu pierderi catastrofale rare, au ES diferit. Forma integrală @eq:es este definiția generală; ea rămîne valabilă și pentru distribuțiile discrete, unde media condiționată poate da un număr diferit și cu proprietăți mai slabe.

## Exemplu rezolvat: VaR și ES din 100 de zile {h=1ca89dfa}

Pe un eșantion, cuantila este înlocuită de o statistică de ordine. O poziție are 100 de randamente zilnice, iar cele mai proaste cinci sînt −6,1%, −4,8%, −4,2%, −3,9% și −3,5%. Cuantila empirică de nivel $\alpha$ este al $k$-lea cel mai mic randament, cu $k = \lceil n\alpha \rceil$, unde $\lceil\cdot\rceil$ înseamnă rotunjirea în sus la următorul întreg.

Pentru VaR 1%, $k = \lceil 100 \times 0{,}01 \rceil = 1$: cuantila empirică este cel mai prost randament, iar $\mathrm{VaR}_{1\%} = 6{,}1\%$. O singură observație decide această estimare, deci cu 100 de zile ea este foarte imprecisă: un alt eșantion de 100 de zile ar avea aproape sigur o altă cea mai proastă zi. Pentru ES 2,5%, $k = \lceil 2{,}5 \rceil = 3$, iar estimarea este media celor mai mari trei pierderi, $\mathrm{ES}_{2{,}5\%} = (6{,}1 + 4{,}8 + 4{,}2)/3 = 5{,}03\%$, în timp ce $\mathrm{VaR}_{2{,}5\%}$ este a treia cea mai mare pierdere, 4,2%.

Exprimat în bani, o poziție de 1 milion de lei are un VaR 1% de 61 000 de lei: într-o zi din o sută, pierderea este mai mare. Exemplul arată și de ce ES 2,5% și VaR 1% formează o pereche firească: ES 2,5% folosește trei observații acolo unde VaR 1% folosește una, iar valoarea lui, 5,03%, se află între cea mai mare și a treia cea mai mare pierdere.

## VaR 1% și ES 2,5% pentru S&P 500 {h=e1bed662}

Histograma arată cele 6 714 randamente logaritmice zilnice ale S&P 500 din 2000 încoace. Zona hașurată conține cele mai proaste 2,5% dintre zile. Linia continuă marchează minus VaR 1% istoric, la −3,46%, iar linia întreruptă minus ES 2,5%, la −3,76%; VaR 2,5% istoric, marginea dreaptă a zonei hașurate, este 2,51%.

Cele două linii răspund la întrebări diferite. VaR 1% este un singur punct al cozii: 1% dintre zile se află în stînga lui, oriunde s-ar afla. ES 2,5% face media întregii cozi hașurate, inclusiv a zilelor extreme din 2008 și 2020, abia vizibile ca bare izolate departe în stînga; dacă acele zile ar fi fost de două ori mai proaste, VaR 1% ar fi rămas neschimbat, iar ES ar fi crescut.

Pentru o poziție de 1 milion de lei în indice, cele două cifre înseamnă un VaR 1% de circa 34 552 de lei și un ES 2,5% de circa 37 613 lei. ES 2,5% este mai mare decît VaR 1%, deși se referă la o probabilitate mai puțin extremă, deoarece face media pierderilor de dincolo de cuantila de 2,5%, iar unele dintre aceste pierderi sînt mult mai mari decît 3,46%.

## Distribuția Normală: formule explicite {h=268026d6}

Dacă $X \sim N(\mu, \sigma^2)$, ambele măsuri au formule explicite. Cu $z_\alpha = \Phi^{-1}(\alpha)$ cuantila Normală standard, unde $\Phi$ și $\varphi$ sînt funcția de repartiție și densitatea lui $N(0, 1)$,

$$ \mathrm{VaR}_\alpha = -(\mu + \sigma z_\alpha), \qquad \mathrm{ES}_\alpha = -\mu + \sigma\,\frac{\varphi(z_\alpha)}{\alpha}. $$ {#eq:normal}

Formula pentru ES se sprijină pe media cozii distribuției Normale standard, $E[Z \mid Z \le z] = -\varphi(z)/\Phi(z)$. Derivarea folosește $\varphi'(z) = -z\varphi(z)$: integrala lui $t\varphi(t)$ de la $-\infty$ la $z$ este $-\varphi(z)$, iar împărțirea la $P(Z \le z) = \Phi(z) = \alpha$ dă media condiționată.

Merită reținute constantele: $z_{0{,}01} = -2{,}326$, $z_{0{,}025} = -1{,}960$, $\varphi(z_{0{,}025}) = 0{,}0584$, iar factorul ES la 2,5% este $\varphi(z_{0{,}025})/0{,}025 = 2{,}338$. Cu $\mu = 0$, raportul dintre ES 2,5% și VaR 1% este $2{,}338/2{,}326 = 1{,}005$: pentru randamente Normale, cele două măsuri sînt practic egale. De aceea FRTB a ales ES 2,5% pentru a înlocui VaR 1%: pentru date Normale, cerința de capital rămîne aproximativ aceeași, iar pentru date cu cozi groase, unde ES 2,5% depășește VaR 1%, ea crește.

## Exemplu rezolvat: VaR și ES Normale pentru S&P 500 {h=d804d1ea}

Momentele de selecție ale randamentelor zilnice ale S&P 500 din 2000 sînt $\hat\mu = 0{,}025\%$ și $\hat\sigma = 1{,}21\%$. Introduse în @eq:normal, ele dau VaR 1% Normal $= -(0{,}025 - 2{,}326 \times 1{,}21) = -0{,}025 + 2{,}822 = 2{,}80\%$ și ES 2,5% Normal $= -0{,}025 + 2{,}338 \times 1{,}21 = -0{,}025 + 2{,}836 = 2{,}81\%$.

Datele spun altceva. Simularea istorică dă VaR 1% = 3,46% și ES 2,5% = 3,76%. VaR 1% Normal este prea mic cu 19%, deoarece coada distribuției Normale este prea subțire ([Capitolul 2](ch2_ro.html)). ES 2,5% Normal este și mai departe de date, cu aproximativ un sfert sub valoarea istorică, deoarece coada Normală nu conține nimic asemănător zilelor extreme incluse în media istorică.

Doi parametri, o medie și o abatere standard, sînt ajustați la centrul distribuției, acolo unde se află cele mai multe observații; ei nu pot descrie și coada. Metodele din secțiunile următoare urmează unul din două drumuri: schimbă distribuția (Student-t, Cornish–Fisher, EVT) sau schimbă datele folosite (simularea istorică, filtrarea volatilității).

## Distribuția Student-t: formule explicite {h=d7fe713d}

Distribuția Student-t din [Capitolul 2](ch2_ro.html) are o coadă de tip putere. Scriem $X = m + s\,T$, cu $T \sim t_\nu$, poziția $m$, scala $s$ și $\nu$ grade de libertate; fie $t_\alpha = t_\nu^{-1}(\alpha)$ cuantila și $f_\nu$ densitatea lui $t_\nu$. Atunci

$$ \mathrm{VaR}_\alpha = -(m + s\,t_\alpha), \qquad \mathrm{ES}_\alpha = -m + s\,\frac{f_\nu(t_\alpha)}{\alpha}\,\frac{\nu + t_\alpha^2}{\nu - 1}, \quad \nu > 1. $$ {#eq:tdist}

Condiția $\nu > 1$ este necesară pentru ca media, deci și ES, să existe. Scala $s$ nu este abaterea standard: varianța lui $X$ este $s^2\nu/(\nu - 2)$, deci o distribuție t standardizată, cu varianța 1, are $s = \sqrt{(\nu - 2)/\nu}$. Această standardizare reapare în modelul GARCH-t din [secțiunea despre VaR condiționat](#sec-5).

Ajustată la S&P 500 prin verosimilitate maximă, distribuția are poziția $\hat m = 0{,}068$, scala $\hat s = 0{,}703$ și $\hat\nu = 2{,}67$ grade de libertate. Cuantila este $t_{0{,}01} = -5{,}023$, deci VaR 1% $= -(0{,}068 + 0{,}703 \times (-5{,}023)) = 3{,}46\%$, exact valoarea istorică. Factorul ES este 5,720, deci ES 2,5% $= -0{,}068 + 0{,}703 \times 5{,}720 = 3{,}95\%$, peste valoarea istorică de 3,76%. Cu $\nu < 3$, coada estimată este foarte groasă: se potrivește cu cuantila de 1%, dar dincolo de ea distribuția t plasează mai multă masă decît arată datele.

## Scalarea VaR de la o zi la zece zile {h=4ea7a948}

VaR-ul de reglementare are un orizont de 10 zile, în timp ce datele și modelele sînt zilnice. **Regula rădăcinii pătrate a timpului** (*square-root-of-time rule*) face legătura: $\mathrm{VaR}^{(h)} \approx \sqrt{h}\,\mathrm{VaR}^{(1)}$. Regula este exactă pentru randamente Normale i.i.d. cu media zero, deoarece randamentul pe $h$ zile este atunci Normal, cu abaterea standard $\sigma\sqrt{h}$, iar fiecare cuantilă se scalează cu $\sqrt{h}$. Basel 1996 cerea un VaR pe 10 zile, obținut de obicei din VaR-ul pe o zi prin factorul $\sqrt{10} = 3{,}162$.

Pentru S&P 500, VaR 1% istoric pe o zi este 3,46%, iar regula dă $3{,}162 \times 3{,}46 = 10{,}93\%$ pe 10 zile. VaR 1% istoric al randamentelor pe 10 zile suprapuse este 10,37%, ceva mai mic: raportul este 3,00, sub $\sqrt{10}$. Diferența nu este un accident al eșantionului, ci rezultatul a trei trăsături ale randamentelor reale.

Prima sînt cozile groase. Suma a 10 randamente cu cozi groase este mai aproape de distribuția Normală decît fiecare dintre ele (teorema limită centrală, CLT, *central limit theorem*), deci boltirea ei este mai mică, iar raportul dintre cuantila ei de 1% și cea zilnică este sub $\sqrt{10}$. A doua este volatility clustering ([Capitolul 9](ch9_ro.html)): după o zi liniștită, volatilitatea tinde să revină spre nivelul ei pe termen lung, deci riscul pe 10 zile este mai mare decît $\sqrt{10}$ ori riscul de azi; după o perioadă turbulentă, volatilitatea tinde să scadă, iar riscul pe 10 zile este mai mic. A treia este autocorelația. Varianța unei sume pe 10 zile este $\sigma^2\bigl(10 + 2\sum_{k=1}^{9}(10-k)\rho_k\bigr)$, deci o autocorelație de ordinul întîi negativă o micșorează, iar una pozitivă o mărește. S&P 500 are $\hat\rho_1 = -0{,}10$: zilele succesive se compensează parțial, ceea ce se adaugă efectului cozilor groase. BET are $\hat\rho_1 = +0{,}11$ (tranzacționarea nesincronă a acțiunilor din indice), iar raportul lui este 3,86, peste $\sqrt{10}$. Regula este o convenție a reglementării, nu o proprietate a datelor.

## @end

VaR @eq:var este o cuantilă a pierderii, iar ES @eq:es pierderea medie de dincolo de ea. Pentru S&P 500 din 2000, valorile istorice sînt VaR 1% = 3,46% și ES 2,5% = 3,76%; distribuția Normală dă 2,80% și 2,81%, iar o distribuție Student-t reproduce VaR, dar supraestimează ES. [Secțiunea următoare](#sec-3) întreabă ce proprietăți ar trebui să aibă o măsură de risc bună și arată că VaR îi lipsește una dintre ele.

# Măsuri de risc coerente

Secțiunea judecă VaR și ES după două criterii teoretice. Primul, coerența, întreabă dacă o măsură de risc se comportă rezonabil atunci cînd pozițiile sînt scalate sau combinate; cele patru axiome ale lui Artzner, Delbaen, Eber și Heath o definesc, iar un contraexemplu cu două obligațiuni arată că VaR o poate încălca. Al doilea, elicitabilitatea, întreabă dacă prognozele unei măsuri pot fi comparate cu rezultatele, iar aici ordinea celor două măsuri se inversează.

## 1999: patru axiome pentru o măsură de risc {h=d9196685}

@ArtznerEtAl1999 definesc o măsură de risc $\rho$ ca o funcție care asociază unei pierderi $L$ capitalul care face poziția acceptabilă pentru o autoritate de reglementare. Ei numesc $\rho$ **coerentă** dacă satisface patru axiome. Monotonia (*monotonicity*): dacă $L_1 \le L_2$ în orice stare a lumii, atunci $\rho(L_1) \le \rho(L_2)$. Invarianța la translație (*translation invariance*): $\rho(L + c) = \rho(L) + c$ pentru o sumă sigură $c$, deci adăugarea unei pierderi sigure mărește capitalul exact cu acea sumă. Omogenitatea pozitivă (*positive homogeneity*): $\rho(\lambda L) = \lambda\rho(L)$ pentru $\lambda > 0$, deci dublarea unei poziții îi dublează capitalul. Subaditivitatea (*subadditivity*): $\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)$.

Primele trei axiome sînt aproape necontestate. Subaditivitatea poartă conținutul economic: combinarea a două poziții nu creează niciodată risc, deci diversificarea nu adaugă niciodată risc. Ea are o consecință practică pentru o bancă: suma capitalurilor calculate separat pentru fiecare divizie este o margine superioară pentru capitalul întregii bănci, iar capitalul poate fi alocat divizie cu divizie. O măsură care nu este subaditivă ar putea face ca o bancă fuzionată să pară mai riscantă decît cele două jumătăți ale ei și ar recompensa împărțirea artificială a unui portofoliu pentru a reduce capitalul.

Fotografia îl arată pe Freddy Delbaen, unul dintre cei patru autori, în 1997. Axiomele au fost publicate în 1999 în *Mathematical Finance* și au schimbat întrebarea despre măsurile de risc din „ce număr este intuitiv?” în „ce proprietăți trebuie să aibă numărul?”.

## Contraexemplu: două obligațiuni, pas cu pas {h=cf4b8b3f}

Două obligațiuni independente pierd fiecare 100 dacă emitentul intră în neplată, ceea ce se întîmplă cu probabilitatea 4%, și 0 în rest; nivelul este $\alpha = 5\%$. Pentru o obligațiune, $P(L > 0) = 4\% \le 5\%$, deci cuantila de 5% a pierderii este 0, iar $\mathrm{VaR}_{5\%} = 0$ pentru fiecare obligațiune. Un manager de risc care folosește VaR nu ar vedea niciun risc în niciuna dintre ele.

Pentru portofoliul format din ambele obligațiuni, pierderea este 200 dacă ambele intră în neplată, cu probabilitatea $0{,}04^2 = 0{,}16\%$, și 100 dacă exact una intră în neplată, cu probabilitatea $2 \times 0{,}04 \times 0{,}96 = 7{,}68\%$. Acum $P(L > 0) = 7{,}84\% > 5\%$, deci VaR-ul portofoliului este 100. Deoarece $\mathrm{VaR}_{5\%}(L_1 + L_2) = 100 > 0 + 0$, VaR penalizează diversificarea: deținerea ambelor obligațiuni pare mai riscantă decît suma deținerilor separate.

ES 5% este media celor mai proaste 5% dintre rezultate. Pentru o obligațiune, cele mai proaste 5% sînt formate din cele 4% stări de neplată, cu pierderea 100, și 1% dintre stările cu pierderea 0, deci $\mathrm{ES}_{5\%} = (4\% \times 100 + 1\% \times 0)/5\% = 80$. Pentru portofoliu, cele mai proaste 5% sînt formate din 0,16% cu pierderea 200 și 4,84% cu pierderea 100, deci $\mathrm{ES}_{5\%} = (0{,}16\% \times 200 + 4{,}84\% \times 100)/5\% = 103{,}2$. Deoarece $103{,}2 \le 80 + 80$, ES este aici subaditiv și vede riscul de neplată al fiecărei obligațiuni, pe care VaR îl ignoră complet.

## Coerența măsurilor ES și VaR {h=daad9427}

ES satisface toate cele patru axiome [@AcerbiTasche2002]. Motivul se vede în @eq:es: ES este o medie a cuantilelor pe coadă, iar mediile respectă sumele. Mai precis, ES poate fi scris ca cea mai mare pierdere medie pe o mulțime de scenarii reponderate, iar maximul unei sume nu este niciodată mai mare decît suma maximelor.

VaR le satisface pe primele trei; subaditivitatea poate să nu fie îndeplinită. Ea nu este îndeplinită pentru pierderi discrete, cum sînt pierderile din credit și obligațiunile din exemplul precedent, și pentru cozi extrem de groase, cu tail index-ul sub 1 ([Capitolul 5](ch5_ro.html)), unde pierderea medie este infinită. Este îndeplinită pentru distribuțiile eliptice, ca distribuția Normală și Student-t, iar @DanielssonEtAl2013 arată că în practică este îndeplinită pentru majoritatea randamentelor de acțiuni, al căror tail index este mult peste 1.

ES plătește coerența cu o cerință mai puternică: are nevoie de o pierdere medie finită, în timp ce VaR există pentru orice distribuție. Pentru randamentele zilnice ale acțiunilor, cu tail index-ul în jur de 3, aceasta nu este o constrîngere; pentru unele pierderi operaționale sau din asigurări poate fi.

## Elicitabilitatea: evaluarea prognozelor {h=e67d80c3}

O statistică este **elicitabilă** (*elicitable*) dacă o funcție de pierdere, sau funcție de scor (*scoring function*), este minimizată în medie de valoarea ei adevărată [@Gneiting2011]. Media este elicitabilă prin eroarea pătratică, mediana prin eroarea absolută. Elicitabilitatea contează deoarece permite ordonarea prognozelor concurente: prognozorul cu scorul mediu mai mic este cel mai bun, iar niciun prognozor nu își poate reduce scorul așteptat raportînd altceva decît valoarea adevărată.

VaR este elicitabil. Funcția de scor este pierderea cuantilă (*quantile loss*, sau *pinball loss*)

$$ S(v, x) = (\alpha - \mathbf{1}\{x < -v\})(x + v), $$ {#eq:pinball}

unde $v$ este prognoza VaR, $x$ randamentul realizat, iar $\mathbf{1}\{\cdot\}$ este 1 dacă evenimentul are loc și 0 altfel. Media ei este minimă acolo unde $P(X < -v) = \alpha$, adică în $v = \mathrm{VaR}_\alpha$. Două modele VaR pot fi deci comparate prin media pierderii cuantile a prognozelor lor, cum face [secțiunea Backtesting](#sec-7).

ES singur nu este elicitabil [@Gneiting2011]: nicio funcție de scor nu este minimizată în medie de adevăratul ES. Perechea (VaR, ES) este însă elicitabilă împreună [@FisslerZiegel2016]. Consecința este practică: prognozele ES se testează împreună cu prognozele VaR, ca în backtesting-ul pentru ES de la sfîrșitul [secțiunii Backtesting](#sec-7). Coerența favorizează ES, elicitabilitatea favorizează VaR, iar regulile Basel le folosesc pe amîndouă: ES pentru capital, depășirile VaR pentru backtesting.

## @end

ES este coerent; VaR nu este întotdeauna subaditiv și poate penaliza diversificarea, cum arată exemplul celor două obligațiuni, cu un VaR al portofoliului de 100 față de VaR-uri individuale de 0. VaR este elicitabil prin pierderea cuantilă @eq:pinball, iar ES este elicitabil doar împreună cu VaR. [Secțiunea următoare](#sec-4) trece de la proprietăți la estimare.

# Estimarea VaR și ES

Secțiunea compară cinci metode necondiționate: simularea istorică, distribuțiile Normală și Student-t, dezvoltarea Cornish–Fisher și EVT. Ea calculează fiecare metodă pas cu pas pentru S&P 500, măsoară eroarea de eșantionare a unui VaR istoric prin bootstrap și compară toate metodele pe șase serii: S&P 500, DAX și BET din 2000, Bitcoin din 2014 și două acțiuni de la Bursa de Valori București (BVB), Banca Transilvania (TLV) și OMV Petrom (SNP), din 2010.

## Sinteza celor cinci metode necondiționate {h=7dede6ca}

Tabelul prezintă cele cinci metode, cu ipoteza și punctul slab al fiecăreia. Simularea istorică (HS, *historical simulation*) presupune că fereastra trecută este un eșantion din distribuția zilei de mîine; are puține puncte în coadă și nu poate produce o pierdere dincolo de cea mai proastă zi din fereastră. Metoda Normală presupune randamente Normale și are cozi mult prea subțiri. Metoda Student-t permite cozi groase, dar impune o singură formă ambelor cozi. Metoda Cornish–Fisher corectează cuantila Normală prin asimetrie și boltire și nu mai funcționează cînd boltirea este mare. EVT ajustează o distribuție Pareto generalizată (GPD, *generalised Pareto distribution*) peste un prag ridicat, ca în [Capitolul 5](ch5_ro.html), iar punctul ei slab este alegerea pragului.

„Necondiționat” înseamnă că metoda dă același VaR în fiecare zi a ferestrei de estimare: ea descrie distribuția randamentelor ca medie peste perioadele calme și cele turbulente. Metodele condiționate din [secțiunea următoare](#sec-5) scalează, în schimb, prognoza cu volatilitatea de azi.

Toate metodele folosesc randamente logaritmice zilnice în procente ([Capitolul 1](ch1_ro.html)) și raportează VaR și ES ca pierderi pozitive. Această convenție comună face cifrele direct comparabile între metode și între serii.

## Simularea istorică {h=b76952ee}

Simularea istorică este cel mai simplu estimator. Luăm ultimele $n$ randamente $r_{t-n+1}, \dots, r_t$ și le ordonăm, $r_{(1)} \le \dots \le r_{(n)}$. Atunci

$$ \widehat{\mathrm{VaR}}_\alpha = -r_{(k)}, \quad k = \lceil n\alpha \rceil, \qquad \widehat{\mathrm{ES}}_\alpha = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}. $$ {#eq:hs}

Nu se presupune nicio distribuție: distribuția empirică a ferestrei este prognoza. Pentru S&P 500 din 2000, $n = 6\,714$. VaR 1% folosește $k = \lceil 67{,}14 \rceil = 68$, iar a 68-a cea mai proastă zi este o pierdere de 3,46%. ES 2,5% este media celor mai proaste $\lceil 167{,}85 \rceil = 168$ de zile, 3,76%.

Lungimea ferestrei $n$ este un compromis. O fereastră scurtă, de 250 de zile, reacționează la evenimentele recente, dar VaR 1% al ei este al treilea cel mai prost randament, $k = \lceil 2{,}5 \rceil = 3$, și se sprijină pe trei observații. O fereastră lungă, aici toate datele, dă o estimare stabilă, care nu reacționează deloc la volatilitatea de azi.

Un artefact specific al unei ferestre fixe este **ghost effect**. O zi de crah schimbă VaR cînd intră în fereastră, ceea ce este firesc, și îl schimbă din nou, brusc, după $n$ zile, cînd iese din fereastră, deși în acea zi nu s-a întîmplat nimic. Cu o fereastră de 500 de zile, un crah rămîne în estimare aproximativ doi ani, cum arată prognozele pe fereastră mobilă din [secțiunea despre VaR condiționat](#sec-5).

## Precizia unui VaR istoric {h=79b60457}

O estimare VaR este o statistică și are o eroare de eșantionare. Bootstrap-ul din [Capitolul 2](ch2_ro.html) o măsoară: reeșantionăm cele $n$ randamente cu întoarcere de 2000 de ori, recalculăm VaR și ES pe fiecare eșantion nou și citim cuantilele de 2,5% și 97,5% ale celor 2000 de valori ca interval de 95%.

Cu toate cele 6 714 zile ale S&P 500, VaR 1% de 3,46% are intervalul de 95% [3,18%; 3,66%], iar ES 2,5% de 3,76% are intervalul [3,46%; 4,07%]. Numai cu ultimele 500 de zile, VaR 1% este 2,75%, cu intervalul [1,99%; 4,96%], iar ES 2,5% este 2,94%, cu intervalul [2,14%; 3,82%]. Cu doi ani de date, intervalul pentru VaR 1% are o lățime de aproape trei puncte procentuale, deoarece estimarea se sprijină pe cele mai proaste cinci zile, $k = \lceil 500 \times 0{,}01 \rceil = 5$. Intervalul pentru ES este mai îngust, deoarece o medie a 13 observații variază mai puțin decît o singură statistică de ordine.

Intervalele stabilesc scara pentru toate comparațiile care urmează. O diferență între două metode mai mică decît intervalul bootstrap nu arată că o metodă este mai bună; ea poate fi zgomot de eșantionare. Reeșantionarea tratează zilele ca independente, deci, în prezența volatility clustering, incertitudinea reală este mai degrabă mai mare.

## Dezvoltarea Cornish–Fisher {h=759e141c}

@CornishFisher1938 corectează cuantila Normală $z = z_\alpha$ cu asimetria $S$ și cu excesul de boltire $K$, coeficientul de boltire minus 3:

$$ z_{CF} = z + \frac{(z^2 - 1)S}{6} + \frac{(z^3 - 3z)K}{24} - \frac{(2z^3 - 5z)S^2}{36}, \qquad \mathrm{VaR}_\alpha = -(\mu + \sigma z_{CF}). $$ {#eq:cf}

Dezvoltarea este o serie în cumulanții de ordin superior în jurul cuantilei Normale. În $z_{0{,}01} = -2{,}326$, o asimetrie $S$ negativă și un exces de boltire $K$ pozitiv împing amîndouă cuantila mai departe în coada stîngă, deci VaR crește peste valoarea Normală.

Pentru S&P 500, $S = -0{,}35$ și $K = 10{,}7$. Numai termenul de boltire este $(z^3 - 3z)K/24 \approx -2{,}50$, mai mare decît cuantila Normală însăși, iar cuantila corectată este $z_{CF} = -5{,}03$, ceea ce dă un VaR 1% de 6,08%, față de valoarea istorică de 3,46%. Corecția a depășit valoarea corectă cu mai mult decît o subestima metoda Normală.

Dezvoltarea este un instrument pentru corecții mici: este precisă cînd distribuția este apropiată de cea Normală și termenii de corecție sînt mici în raport cu $z$. Cu un exces de boltire de peste 10, termenii nu mai sînt mici, iar dezvoltarea ajunge mult peste valoarea corectă. Metoda este utilă doar cînd $|S|$ și $K$ sînt moderate, de exemplu pentru randamente lunare sau pentru portofolii diversificate.

## EVT: peaks over threshold (Capitolul 5) {h=528e0b25}

Metoda vîrfurilor peste prag (POT, *peaks over threshold*) din [Capitolul 5](ch5_ro.html) modelează doar coada. Cu pierderile $L = -r$ și un prag $u$ fixat la cuantila de 90% a pierderilor, ca la @McNeilFrey2000, excesele $L - u$ peste $u$ urmează aproximativ o distribuție GPD cu forma $\xi$ și scala $\beta$. Estimatorul cozii dă atunci

$$ \mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}\Big[\Big(\frac{n\alpha}{N_u}\Big)^{-\xi} - 1\Big], \qquad \mathrm{ES}_\alpha = \frac{\mathrm{VaR}_\alpha + \beta - \xi u}{1 - \xi}, $$ {#eq:pot}

unde $n$ este numărul de zile, iar $N_u$ numărul exceselor. Raportul $n\alpha/N_u$ compară probabilitatea-țintă a cozii cu ponderea zilelor de peste prag; o formă $\xi > 0$ înseamnă o coadă groasă, de tip Pareto.

Pentru S&P 500, $u = 1{,}26\%$, $N_u = 672$ de excese (10,0% dintre zile), $\hat\xi = 0{,}164$ și $\hat\beta = 0{,}820$. Pentru VaR 1%, $n\alpha/N_u = 0{,}0999$ și $0{,}0999^{-\hat\xi} = 1{,}461$; cu $\hat\beta/\hat\xi = 4{,}988$, VaR 1% este $1{,}26 + 4{,}988 \times (1{,}461 - 1) = 3{,}56\%$. ES 2,5% este 3,77%. Ambele sînt apropiate de valorile istorice, 3,46% și 3,76%.

La 1% și la 2,5%, estimările EVT și cele istorice coincid deoarece amîndouă sînt determinate de aceleași cîteva sute de observații din coadă. EVT aduce un plus mai departe în coadă: netezește coada cu un model cu doi parametri și extrapolează dincolo de cea mai proastă zi observată, ceea ce o face metoda potrivită pentru VaR 0,1% și pentru testele de stres.

## VaR pentru diferite probabilități ale cozii {h=3c672d06}

Graficul arată, pentru S&P 500 (stînga) și BET (dreapta), VaR-ul fiecărei metode ca funcție de probabilitatea cozii $\alpha$, de la 5% în stînga la 0,1% în dreapta, pe o axă logaritmică; punctele sînt estimările istorice. Citirea de la stînga la dreapta înseamnă coborîrea tot mai adînc în coadă.

La 5% toate metodele sînt de acord: atît de aproape de centru, forma cozii aproape că nu contează. La 1%, linia Normală se află deja sub puncte, iar distanța crește pe măsură ce $\alpha$ scade, deoarece cuantila Normală crește doar ca $\sqrt{2\ln(1/\alpha)}$. La 0,1%, liniile Student-t și EVT rămîn aproape de date, în timp ce linia Cornish–Fisher crește exploziv, la aproximativ dublul valorii istorice.

Graficul rezumă secțiunea într-o singură imagine. Metodele cu o coadă de tip putere, Student-t și EVT, urmează datele pe întreaga gamă de niveluri; metoda Normală greșește cu atît mai mult cu cît coborîm mai adînc în coadă; iar Cornish–Fisher, construită pentru abateri mici de la distribuția Normală, greșește în sens opus.

## VaR 1% pe metode, pentru șase serii {h=4c4fa46c}

Tabelul raportează VaR 1%, în procente, pentru șase serii pe eșantioanele lor complete (S&P 500, DAX și BET din 2000, Bitcoin din 2014, TLV și SNP din 2010), împreună cu excesul de boltire $K$ și forma GPD $\hat\xi$. Nivelurile diferă mult, de la 3,46% pentru S&P 500 la 10,54% pentru Bitcoin, dar tiparul dintre metode este același pe fiecare rînd.

Simularea istorică, Student-t și EVT diferă cu cel mult 0,7 puncte procentuale; cea mai mare diferență apare la Bitcoin, unde Student-t dă 11,06%, iar EVT 10,38%. Ținînd seama de intervalele bootstrap de pe slide-urile anterioare, aceste trei metode sînt de acord. VaR-ul Normal este peste tot prea mic, cu 14% pînă la 24%: de la 2,80% față de 3,46% pentru S&P 500 la 7,99% față de 10,54% pentru Bitcoin.

Cornish–Fisher dă de 1,3 pînă la 2,0 ori VaR-ul istoric, de la 5,42% față de 4,20% pentru DAX, care are cel mai mic exces de boltire (6,1), la 10,19% față de 5,11% pentru TLV, care îl are pe cel mai mare (13,5). Cu $K$ între 6 și 14, dezvoltarea se află în afara domeniului ei de valabilitate pentru fiecare serie; depășirea este cea mai mică pentru DAX și cea mai mare pentru TLV.

## ES 2,5% pe metode, pentru șase serii {h=f000fc0b}

Aceeași comparație pentru ES 2,5% arată cît de mult diferă între metode coada de dincolo de cuantilă. În distribuția Normală, ES 2,5% este aproape exact egal cu VaR 1% (2,81% față de 2,80% pentru S&P 500). În date, ES 2,5% istoric se află între aproximativ 1,5% și 10,5% peste VaR 1% istoric (calculat din valorile nerotunjite), în funcție de serie: cu cît coada de dincolo de cuantila de 1% este mai groasă, cu atît diferența este mai mare.

Pentru S&P 500, DAX, BET și Bitcoin, Student-t dă cel mai mare ES dintre metodele rezonabile; pentru Bitcoin, 13,37% față de valoarea istorică de 10,84%. Cu $\hat\nu < 3$, distribuția t ajustată are o coadă mai groasă decît datele de dincolo de prag: un singur parametru potrivește bine cuantila de 1%, dar extrapolează apoi prea multă masă în coada extremă.

EVT și simularea istorică sînt de acord la 2,5% pentru fiecare serie, pînă la 0,06 puncte procentuale. Alegerea metodei contează deci mai mult pentru ES decît pentru VaR, deoarece ES depinde de întreaga coadă, iar VaR de un singur punct al ei. Acesta este prețul coerenței lui ES: el este mai sensibil la modelul cozii îndepărtate.

## @end

Pentru VaR 1% și ES 2,5%, simularea istorică, Student-t și EVT sînt de acord pe toate cele șase serii, în timp ce VaR-ul Normal este prea mic cu 14–24%, iar Cornish–Fisher depășește valoarea istorică de 1,3 pînă la 2,0 ori. Bootstrap-ul arată că un VaR 1% istoric pe doi ani de date este incert cu cîteva puncte procentuale. Toate cele cinci metode dau însă același VaR în fiecare zi a ferestrei. [Secțiunea următoare](#sec-5) face prognoza să depindă de volatilitatea de azi.

# VaR condiționat: GARCH și simularea istorică filtrată

Secțiunea combină prognozele volatilității din [Capitolul 9](ch9_ro.html) cu modelele cozii din secțiunea precedentă. O prognoză VaR devine produsul dintre volatilitatea de mîine și o cuantilă a inovației standardizate. Secțiunea calculează astfel de prognoze pentru șase serii, evaluează prognoze pe fereastră mobilă din 2005 pînă în 2026 și compară metodele condiționate cu cele necondiționate în crizele din 2008 și 2020.

## Condiționarea pe volatilitatea curentă {h=aea33004}

Volatilitatea apare grupat ([Capitolele 8](ch8_ro.html) și [9](ch9_ro.html)): zilele liniștite urmează după zile liniștite, iar zilele agitate după zile agitate. Un VaR necondiționat face media celor două regimuri, deci este prea mare în anii liniștiți și prea mic la începutul unei crize, exact atunci cînd contează cel mai mult.

Modelul condiționat scrie randamentul de mîine ca $r_{t+1} = \mu + \sigma_{t+1}z_{t+1}$, unde $\sigma_{t+1}$ este prognoza volatilității făcută azi, iar $z$ este o inovație i.i.d. cu media 0 și varianța 1. Deoarece singura parte variabilă în timp este scala, măsurile de risc decurg din cele ale lui $z$:

$$ \mathrm{VaR}_{t+1} = -(\mu + \sigma_{t+1}\,q_\alpha(z)), \qquad \mathrm{ES}_{t+1} = -\mu + \sigma_{t+1}\,\mathrm{ES}_\alpha(z), $$ {#eq:cond}

unde $q_\alpha(z)$ și $\mathrm{ES}_\alpha(z)$ sînt cuantila și ES ale inovației standardizate. Modelul are două ingrediente: o prognoză a lui $\sigma_{t+1}$ și o distribuție pentru $z$.

Capitolul folosește un model GARCH(1,1) [@Bollerslev1986] pentru volatilitate și două variante pentru $z$. **GARCH-t** presupune o distribuție Student-t standardizată, cu $\nu$ estimat împreună cu parametrii GARCH. **Simularea istorică filtrată** (FHS, *filtered historical simulation*), propusă de @HullWhite1998 și de @BaroneAdesiGiannopoulosVosper1999, folosește distribuția empirică a reziduurilor standardizate (*standardised residuals*) $\hat z_t = (r_t - \hat\mu)/\hat\sigma_t$: volatilitatea vine din model, forma cozii din date.

## Exemplu rezolvat: S&P 500 la 18 septembrie 2026 {h=3412159c}

Un model GARCH(1,1)-t estimat pe toate datele S&P 500 dă $\hat\mu = 0{,}079$, $\hat\alpha = 0{,}123$, $\hat\beta = 0{,}872$ și $\hat\nu = 6{,}26$; persistența $\hat\alpha + \hat\beta = 0{,}995$ este tipică pentru randamentele zilnice ale acțiunilor. Prognoza pentru ziua de după 18 septembrie 2026 este $\sigma_{t+1} = 0{,}714\%$.

Pentru GARCH-t, cuantila distribuției t standardizate este $q_{0{,}01}(z) = t_\nu^{-1}(0{,}01)\sqrt{(\nu - 2)/\nu} = -2{,}557$, iar ES 2,5% al ei este 2,644. Din @eq:cond, VaR 1% este $-(0{,}079 + 0{,}714 \times (-2{,}557)) = 1{,}75\%$, iar ES 2,5% este $-0{,}079 + 0{,}714 \times 2{,}644 = 1{,}81\%$. Pentru FHS, cuantila de 1% a celor 6 714 reziduuri standardizate este −2,787, iar ES 2,5% al lor este 2,952, deci VaR 1% este $-(0{,}079 + 0{,}714 \times (-2{,}787)) = 1{,}91\%$, iar ES 2,5% este 2,03%.

Piața era liniștită: VaR 1% condiționat este cam jumătate din valoarea istorică necondiționată de 3,46%. Cele două variante diferă prin coada lui $z$: reziduurile empirice au o coadă stîngă mai groasă (cuantila −2,787) decît distribuția t estimată (−2,557), deoarece un singur $\nu$ trebuie să descrie ambele cozi, iar pierderile S&P 500 sînt mai extreme decît cîștigurile lui. FHS păstrează această asimetrie.

## VaR și ES pentru ziua următoare, pe șase serii {h=4a2f0025}

Tabelul dă prognozele pentru ziua de după 18 septembrie 2026 pentru cele șase serii: VaR 1% și ES 2,5% prin simularea istorică necondiționată pe toate datele, HS (tot), prin GARCH-t și prin FHS, împreună cu prognoza volatilității $\sigma_{t+1}$, în procente.

Pentru S&P 500, DAX și Bitcoin perioada este liniștită, iar valorile condiționate sînt sub cele necondiționate: VaR 1% al DAX este 2,31% (GARCH-t) și 2,48% (FHS), față de 4,20% istoric, iar cel al Bitcoin 7,37% și 7,99%, față de 10,54%. Pentru BET, TLV și SNP volatilitatea curentă este mare, $\sigma_{t+1}$ este 1,763% pentru BET, 2,791% pentru TLV și 1,966% pentru SNP, iar VaR condiționat se află peste VaR-ul istoric pe termen lung: pentru TLV, 7,18% (GARCH-t) și 7,33% (FHS), față de 5,11%.

Aceeași metodă transmite astfel mesaje opuse pe piețe diferite, iar acesta este rostul condiționării. Un VaR necondiționat i-ar spune unui deținător de acțiuni TLV că riscul este media pe termen lung, cînd volatilitatea curentă este mai mare decît de obicei, și i-ar spune același lucru unui deținător de S&P 500, cînd volatilitatea este mai mică. Pe fiecare rînd, FHS dă un VaR și un ES puțin mai mari decît GARCH-t, din motivul văzut pe slide-ul precedent: reziduurile empirice au o coadă stîngă mai groasă decît distribuția t simetrică.

## Prognoze pe fereastră mobilă: schema de evaluare {h=6510a03d}

O comparație corectă a metodelor cere prognoze în afara eșantionului (*out-of-sample*): prognoza pentru ziua $t$ folosește doar date pînă în ziua $t-1$, așa cum ar fi trebuit să procedeze un manager de risc în timp real. Evaluarea acoperă fiecare zi din ianuarie 2005, inclusiv anul 2008, pentru S&P 500, DAX și BET și din ianuarie 2017 pentru Bitcoin.

Sînt comparate patru metode. HS și metoda Normală folosesc o fereastră mobilă cu ultimele 500 de zile, circa doi ani. GARCH-t și FHS reestimează modelul GARCH(1,1)-t la fiecare 250 de zile pe toate datele trecute, în timp ce volatilitatea $\sigma_t$ este actualizată zilnic cu ultimul randament. Reestimarea parametrilor o dată pe an este suficientă, deoarece ei se schimbă încet; actualizarea zilnică a volatilității este esențială, deoarece ea se schimbă repede.

În fiecare zi, fiecare metodă produce trei cifre: VaR 1%, folosit în backtesting-ul VaR, și VaR 2,5% și ES 2,5%, folosite în backtesting-ul ES. Șirurile rezultate sînt materialul restului capitolului.

## VaR 1% în 2008 și în 2020 {h=e2fe4b40}

Graficul arată randamentele zilnice ale S&P 500 (linia deschisă) în 2007–2009 și 2019–2021, împreună cu minus prognozele VaR 1% ale celor patru metode; punctele marchează zilele în care VaR HS a fost depășit. Un randament aflat sub o linie este o depășire pentru metoda respectivă.

HS și metoda Normală reacționează cu o întîrziere de luni. Fereastra lor de 500 de zile se schimbă încet: cîteva zile de criză mută doar puțin cuantila de 1% a 500 de observații, astfel încît în septembrie–octombrie 2008 și în martie 2020 punctele se grupează acolo unde randamentele trec prin liniile întrerupte aproape plate. GARCH-t și FHS, ale căror linii aproape coincid, coboară brusc în cîteva zile, deoarece fiecare randament mare mărește prognoza volatilității de mîine.

După criză, comportamentul se inversează. VaR HS rămîne ridicat timp de doi ani, pînă cînd zilele de criză ies din fereastră, ghost effect-ul metodei istorice, în timp ce VaR GARCH-t revine la niveluri normale pe măsură ce volatilitatea scade. O bancă ce folosea HS ar fi avut prea mult capital în 2009–2010 și prea puțin în toamna lui 2008.

## Perioade de stres: depășiri în 2008 și 2020 {h=aa2f27e0}

Tabelul numără depășirile VaR 1%, zilele cu $r_t < -\mathrm{VaR}_t$, în cele două perioade de stres, septembrie 2008 – martie 2009 și februarie – mai 2020; coloana „aștept.” este 1% din zilele perioadei. În 2008, S&P 500 a avut 146 de zile de tranzacționare în această perioadă, deci un VaR 1% corect ar fi fost depășit de aproximativ 1,5 ori; VaR Normal a fost depășit în 21 de zile, HS în 15, iar GARCH-t și FHS în cîte 5.

Tiparul se menține pentru toți cei trei indici și pentru ambele crize. În 2008–2009, metoda Normală a avut 18 depășiri pentru DAX și 14 pentru BET, față de 3 și 4 pentru metodele condiționate; în 2020, cu aproximativ 0,7 depășiri așteptate, metoda Normală a avut 12, 13 și 10 depășiri pentru S&P 500, DAX și BET, iar GARCH-t și FHS 3, 3 și 4.

Nicio metodă nu este perfectă într-o criză: chiar și GARCH-t și FHS depășesc numărul așteptat de 2 pînă la 6 ori, deoarece volatilitatea sare mai repede decît o poate urmări orice model al randamentelor de ieri. Dar metodele condiționate greșesc mult mai rar, iar GARCH-t și FHS, care diferă doar prin coada lui $z$, au numere identice în fiecare celulă. Prognoza volatilității contează mai mult decît forma cozii.

## Alegerea metodei {h=8c075991}

Tabelul asociază situațiile cu metodele. Pentru un VaR zilnic al unei poziții lichide, GARCH-t și FHS urmăresc volatilitatea, iar FHS păstrează coada empirică a reziduurilor standardizate. Pentru o verificare rapidă pe un istoric lung, HS nu are nevoie de model și este ușor de explicat. Pentru probabilități ale cozii foarte mici, cum este 0,1%, și pentru testele de stres, EVT extrapolează dincolo de cea mai proastă zi observată. Pentru portofolii cu multe active, metoda varianță–covarianță sau o copulă modelează dependența, copula t atunci cînd contează prăbușirile simultane ([secțiunea următoare](#sec-6)). Pentru asimetrie și boltire moderate, Cornish–Fisher este o corecție rapidă a distribuției Normale.

Două reguli sînt valabile în orice situație. Distribuția Normală singură nu este niciodată adecvată pentru date zilnice: ea subestimează cuantila de 1% cu 14–24% pe fiecare serie a capitolului. Iar o cifră VaR nu are sens fără metoda, fereastra și nivelul ei, care trebuie raportate întotdeauna împreună cu ea.

În fine, fiecare metodă trebuie verificată pe pierderile care au avut loc efectiv. Testele din [secțiunea Backtesting](#sec-7) fac acest lucru pentru prognozele pe fereastră mobilă din această secțiune.

## @end

VaR condiționat @eq:cond scalează o cuantilă a inovației standardizate cu volatilitatea de mîine. La 18 septembrie 2026, el dă pentru S&P 500 un VaR 1% de 1,75% (GARCH-t) și 1,91% (FHS), cam jumătate din valoarea necondiționată, în timp ce pentru BET, TLV și SNP se află peste valoarea pe termen lung. În crizele din 2008 și 2020, metodele condiționate au avut 3 pînă la 5 depășiri, acolo unde metoda Normală a avut între 10 și 21. [Secțiunea următoare](#sec-6) trece de la o singură serie la un portofoliu.

# VaR de portofoliu și dependența

Secțiunea calculează VaR-ul unui portofoliu format din doi indici, BET și S&P 500. Metoda varianță–covarianță combină volatilitățile prin corelație; datele arată apoi că o corelație crește în crize și că nu măsoară probabilitatea prăbușirilor simultane. Copulele separă dependența de distribuțiile individuale, iar o copulă t surprinde prăbușirile simultane pe care o copulă Gaussiană le ratează.

## Un portofoliu BET și S&P 500 {h=b3834dea}

Portofoliul are ponderi egale, $w = (0{,}5;\ 0{,}5)$, în BET și S&P 500. El folosește randamente simple $R_t$, în procente, în locul randamentelor logaritmice, deoarece randamentul simplu al unui portofoliu este exact suma ponderată a randamentelor simple ale activelor lui, $R_{p,t} = w_1R_{1,t} + w_2R_{2,t}$; pentru randamentele logaritmice relația este doar aproximativă. Prețurile se aliniază întîi pe cele 6 466 de zile de tranzacționare comune celor două piețe din 2000, iar randamentele se calculează după aceea, astfel încît o zi de sărbătoare pe o piață să nu creeze un randament fals pe două zile pe cealaltă.

Fiecare indice este luat în moneda lui, deci riscul valutar al unui investitor care deține portofoliul în lei este ignorat. Momentele zilnice sînt $\hat\mu = 0{,}076\%$ și $\hat\sigma = 1{,}42\%$ pentru BET, $\hat\mu = 0{,}034\%$ și $\hat\sigma = 1{,}23\%$ pentru S&P 500, iar corelația lor este $\hat\rho = 0{,}21$.

Corelația scăzută reflectă în parte orele de tranzacționare. Ședințele celor două burse se suprapun doar parțial (BVB tranzacționează pînă la 17:45, ora României, iar New York se deschide la 16:30): cea mai mare parte a ședinței de la New York are loc după închiderea BVB, astfel încît o parte din reacția comună la știrile mondiale apare în BET abia a doua zi, iar corelația din aceeași zi subestimează legătura dintre cele două piețe. Întrebarea secțiunii este dacă VaR-ul portofoliului este sub media celor două VaR-uri individuale și cu cît.

## Metoda varianță–covarianță {h=f16b9346}

Cu vectorul ponderilor $w$, vectorul randamentelor medii $\mu$ și matricea de covarianță $\Sigma$, portofoliul are media $\mu_p = w^\top\mu$ și varianța

$$ \sigma_p^2 = w^\top\Sigma w = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\sigma_1\sigma_2, $$ {#eq:portvar}

unde $\rho$ este corelația celor două active. Dacă randamentele sînt Normale în ansamblu, randamentul portofoliului este și el Normal, iar $\mathrm{VaR}_p = -(\mu_p + z_\alpha\sigma_p)$; aceasta este metoda RiskMetrics [@JPMorganReuters1996], care se aplică și pentru sute de active, deoarece are nevoie doar de o matrice de covarianță.

Pas cu pas: $w_1\sigma_1 = 0{,}712$, $w_2\sigma_2 = 0{,}614$, iar termenul de covarianță este $2 \times 0{,}5 \times 0{,}5 \times 0{,}369 = 0{,}185$, unde $0{,}369 = \rho\sigma_1\sigma_2$; rezultă $\sigma_p^2 = 0{,}712^2 + 0{,}614^2 + 0{,}185 = 1{,}068$ și $\sigma_p = 1{,}033$. Cu $\mu_p = 0{,}055$, VaR 1% este $-(0{,}055 - 2{,}326 \times 1{,}033) = 2{,}35\%$. Suma ponderată a celor două VaR-uri Normale este $0{,}5 \times 3{,}23 + 0{,}5 \times 2{,}82 = 3{,}03\%$, deci **beneficiul diversificării** (*diversification benefit*) este 22% din risc. Cu $\rho = 1$ ar fi zero: volatilitățile s-ar aduna pur și simplu.

Contribuțiile Euler $w_i(\Sigma w)_i/\sigma_p$, unde $(\Sigma w)_i$ este componenta $i$ a vectorului $\Sigma w$, au suma exact $\sigma_p$, deci împart riscul portofoliului între active. BET contribuie cu 56%, iar S&P 500 cu 44%: BET, cu volatilitatea mai mare, poartă mai mult de jumătate din risc, deși are aceeași pondere.

## Corelația în perioadele de criză {h=5f217fff}

Graficul arată corelația randamentelor zilnice BET și S&P 500 pe ferestre mobile de 250 de zile din 2001, cu cele două perioade de stres hașurate și cu corelația pe întregul eșantion, 0,21, ca linie întreruptă. Înainte de 2008, corelația pe fereastră mobilă era apropiată de zero; ea a sărit în toamna lui 2008 și din nou, mult mai sus, în martie 2020.

Măsurată în interiorul fiecărei perioade, corelația a fost 0,20 în 2017, un an liniștit, 0,37 între septembrie 2008 și martie 2009 și 0,71 între februarie și mai 2020. Corelația pe fereastră mobilă a atins maximul, 0,67, pe 25 martie 2020 și a rămas peste 0,6 un an întreg: cît timp cele cîteva săptămîni din martie 2020 se aflau în fereastră, ele dominau corelația celor 250 de zile. O corelație Pearson este determinată de cele mai mari mișcări comune, iar zilele de criză sînt cele mai mari.

Consecința pentru managementul riscului este neplăcută. Diversificarea este cea mai slabă exact atunci cînd este nevoie de ea: un VaR de portofoliu construit pe corelația medie de 0,21 este prea optimist într-o criză, cînd corelația este între aproape dublul și mai mult decît triplul mediei. VaR-ul varianță–covarianță de pe slide-ul precedent este, așadar, o cifră pentru perioadele calme.

## Corelația și dependența în cozi {h=7abc4812}

Corelația Pearson măsoară co-mișcarea liniară pe toate zilele, iar zilele obișnuite, marea majoritate, o domină [@EmbrechtsMcNeilStraumann2002]. Ea depinde și de distribuțiile marginale și se poate schimba la transformări monotone ale randamentelor. Corelațiile de rang evită această problemă, folosind doar ordinea randamentelor: pentru BET și S&P 500, $\tau$ al lui Kendall este $\hat\tau = 0{,}100$, iar corelația de rang Spearman este 0,15.

Dependența în coadă (*tail dependence*) întreabă direct despre extremele comune. Cu $U_i = F_i(X_i)$ randamentele aduse pe o scală uniformă prin funcțiile lor de repartiție $F_i$, coeficientul de dependență în coada inferioară (*lower tail*) este

$$ \lambda_L = \lim_{q \to 0} P(U_1 \le q \mid U_2 \le q), $$ {#eq:taildep}

unde $q$ este o probabilitate mică, iar $\lambda_L \in [0, 1]$. În cuvinte: probabilitatea ca BET să aibă una dintre cele mai proaste zile ale lui, dat fiind că S&P 500 are una dintre cele mai proaste zile ale lui. La independență, probabilitatea condiționată este egală cu $q$ și tinde la zero.

În date, probabilitatea condiționată este 0,22 la $q = 5\%$ și 0,19 la $q = 1\%$, mult peste valorile 0,05 și 0,01 pe care le-ar da independența. O prăbușire simultană, ambii indici sub propria cuantilă de 1% în aceeași zi, apare în 0,19% dintre zile, de nouăsprezece ori mai des decît 0,01%, valoarea așteptată la independență. O corelație de 0,21 nu dezvăluie acest lucru.

## Copule: o introducere {h=312f1a63}

**Teorema lui Sklar** spune că orice funcție de repartiție comună se poate scrie

$$ F(x_1, x_2) = C(F_1(x_1), F_2(x_2)), $$ {#eq:sklar}

unde $F_1$ și $F_2$ sînt distribuțiile marginale, fiecare activ separat, iar $C$ este o **copulă**, o distribuție comună a unor variabile uniforme pe $[0,1]^2$ [@McNeilFreyEmbrechts2015]. Pentru marginale continue, copula este unică. Teorema separă cele două părți ale unui model multivariat: copula conține toată dependența, iar marginalele conțin toate cozile individuale.

Separarea dă o rețetă practică. Marginalele pot fi empirice sau estimate (Student-t, EVT), după cum descriu mai bine fiecare activ. Dependența se estimează pe **pseudo-observații** (*pseudo-observations*) $\hat U_{i,t} = \mathrm{rang}(X_{i,t})/(n + 1)$, rangurile rescalate în intervalul deschis $(0, 1)$, astfel încît estimarea copulei să nu depindă de modelul marginalelor.

Pentru a calcula un VaR de portofoliu, simulăm perechi $(U_1, U_2)$ din copulă, revenim la randamente cu funcțiile cuantilă $F_i^{-1}$, calculăm randamentul portofoliului și citim VaR și ES din distribuția simulată. Tratarea din manual este @FrankeHardleHafner2019, capitolul 17.

## Copula Gaussiană și copula t {h=7e2568a1}

**Copula Gaussiană** este structura de dependență a unei distribuții Normale bivariate cu corelația $\rho$. Dependența ei în coadă este $\lambda_L = 0$ pentru orice $\rho < 1$: la limită, zilele extreme devin independente, iar prăbușirile simultane sînt rare. Aceasta este copula implicită în metoda varianță–covarianță.

**Copula t** este structura de dependență a unei distribuții Student-t bivariate cu corelația $\rho$ și $\nu$ grade de libertate [@DemartaMcNeil2005]. Dependența ei în coadă este pozitivă:

$$ \lambda_L = 2\,t_{\nu+1}\Big(-\sqrt{(\nu + 1)(1 - \rho)/(1 + \rho)}\Big), $$ {#eq:tcopula}

unde $t_{\nu+1}(\cdot)$ este funcția de repartiție Student-t cu $\nu + 1$ grade de libertate. Un $\nu$ mai mic sau un $\rho$ mai mare dau un $\lambda_L$ mai mare: prăbușirile apar împreună pe mai multe active chiar și atunci cînd corelația de ansamblu este modestă.

Pentru BET și S&P 500, parametrul de corelație al ambelor copule este estimat din $\tau$ al lui Kendall prin $\rho = \sin(\pi\hat\tau/2) = 0{,}157$, relație valabilă pentru toate copulele eliptice, iar copula t are $\hat\nu = 4{,}50$, estimat prin verosimilitate maximă. Log-verosimilitatea este 92,1 pentru copula Gaussiană și 219,5 pentru copula t; statistica raportului de verosimilitate, LR = 254,9, este mult peste valoarea critică de 5% a distribuției $\chi^2(1)$, 3,84, deci copula t se potrivește clar mai bine. Dependența în coadă implicată de @eq:tcopula este $\lambda_L = 0{,}096$, față de 0 pentru copula Gaussiană.

## Pierderi mari simultane: datele și două copule {h=35a4a9d9}

Graficul mărește colțul din stînga jos al scalei uniforme, sub 0,25 pe ambele axe: pseudo-observațiile datelor (stînga) și eșantioane de aceeași mărime, 6 466 de puncte, din copula Gaussiană estimată (centru) și din copula t estimată (dreapta). Pătratul de sub 0,05 pe ambele axe conține zilele în care ambii indici s-au aflat printre cele mai proaste 5% ale lor.

Datele au 70 de astfel de zile, copula Gaussiană 37, iar copula t 55; la independență ar fi aproximativ 16 ($6\,466 \times 0{,}05 \times 0{,}05$). În panoul din stînga, punctele se concentrează vizibil în colț; în panoul din centru sînt răspîndite aproape ca la independență; panoul din dreapta se află între cele două.

Copula Gaussiană înjumătățește numărul zilelor proaste comune, deși are aceeași corelație de rang ca datele. Copula t se apropie mult mai mult, dar rămîne totuși sub date, un prim semn că dependența din coadă este mai puternică decît poate descrie o copulă t constantă.

## VaR prin copule: simularea pas cu pas {h=fc6ae75c}

Simularea are patru pași. Primul: extragem $(Z_1, Z_2)$ dintr-o distribuție Normală bivariată cu corelația $\rho$. Pentru copula Gaussiană, luăm $U_i = \Phi(Z_i)$. Pentru copula t, extragem în plus $W \sim \chi^2_\nu$ și luăm $U_i = t_\nu(Z_i\sqrt{\nu/W})$. Factorul comun $\sqrt{\nu/W}$ înmulțește ambele coordonate; cînd $W$ este mic, ambele devin mari în valoare absolută în același timp, iar din acest motiv extremele apar împreună.

Al doilea pas: revenim de la valorile uniforme la randamente cu funcțiile cuantilă empirice, $R_i = \hat F_i^{-1}(U_i)$, astfel încît fiecare randament simulat provine din distribuția observată a indicelui său. Al treilea: calculăm randamentul portofoliului $R_p = 0{,}5R_1 + 0{,}5R_2$ și repetăm extragerea de 200 000 de ori. Al patrulea: calculăm VaR 1% și ES 2,5% prin simulare istorică, @eq:hs, pe valorile simulate ale lui $R_p$.

Ambele copule folosesc aceleași marginale empirice, deci orice diferență între VaR-urile lor vine numai din dependență. Schema izolează efectul dependenței în coadă, care este întrebarea secțiunii.

## VaR 1% și ES 2,5% ale portofoliului, pe metode {h=1f5a979b}

Tabelul compară patru estimări pentru portofoliul 50/50 pe cele 6 466 de zile comune, în procente: VaR 1%, ES 2,5% și probabilitatea unei prăbușiri simultane la 1%. Metoda varianță–covarianță dă 2,35% și 2,36%; copula Gaussiană cu marginale empirice 2,75% și 2,93%; copula t cu marginale empirice 2,81% și 3,03%; simularea istorică pe randamentele portofoliului 3,08% și 3,30%.

Ordinea descompune eroarea metodei Normale. Ea ignoră atît marginalele cu cozi groase, cît și dependența în coadă. Înlocuirea marginalelor Normale cu cele empirice, trecerea de la primul la al doilea rînd, mărește VaR 1% cu 0,40 puncte procentuale; adăugarea dependenței în coadă, de la copula Gaussiană la copula t, îl mărește cu încă 0,06 puncte și ridică probabilitatea prăbușirii simultane de la 0,03% la 0,13%.

Copula t subestimează în continuare VaR-ul istoric, 2,81% față de 3,08%, și probabilitatea prăbușirii simultane, 0,13% față de 0,19%. Legătura BET–S&P 500 este mai puternică în crize decît poate descrie o copulă constantă: dependența însăși se schimbă în timp, cum a arătat corelația pe fereastră mobilă. O copulă cu parametri variabili în timp este pasul următor firesc și întrebarea deschisă a [secțiunii despre AI](#sec-8).

## @end

Diversificarea între BET și S&P 500 elimină 22% din VaR-ul Normal, dar corelația crește de la 0,20 într-un an liniștit la 0,71 în primăvara lui 2020, iar prăbușirile simultane sînt de nouăsprezece ori mai frecvente decît la independență. Copulele @eq:sklar separă dependența de marginale; copula t se potrivește datelor mult mai bine decît copula Gaussiană, dar VaR 1% al portofoliului calculat cu ea, 2,81%, rămîne sub valoarea istorică de 3,08%. [Secțiunea următoare](#sec-7) testează dacă prognozele VaR ale capitolului au fost corecte.

# Backtesting

Secțiunea întreabă dacă o prognoză VaR a fost corectă, folosind doar prognozele și randamentele realizate. Ea definește secvența depășirilor, deduce testul Kupiec pentru numărul depășirilor și testul Christoffersen pentru gruparea lor, prezintă semaforul Basel, aplică toate cele trei instrumente prognozelor pe fereastră mobilă pentru patru indici din 2005 pînă în 2026, ordonează prognozele prin pierderea cuantilă și se încheie cu backtesting-ul ES al lui Acerbi și Székely.

## Secvența depășirilor {h=a620bcc1}

O **depășire** (*exception*, sau *exceedance*) este o zi în care pierderea a depășit prognoza VaR, $r_t < -\mathrm{VaR}_t$. Secvența depășirilor (*hit sequence*) le înregistrează: $I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t\}$. VaR trebuie să fie prognoza făcută cu o zi înainte, fără informații din viitor; un VaR estimat pe un eșantion care include ziua $t$ va fi depășit mai rar decît o prognoză reală.

Dacă VaR la nivelul $\alpha$ este corect, depășirile $I_t$ sînt i.i.d. Bernoulli($\alpha$) [@Christoffersen1998]. Afirmația conține două proprietăți distincte. **Acoperirea necondiționată** (*unconditional coverage*), $P(I_t = 1) = \alpha$, înseamnă că numărul depășirilor este corect în medie. **Independența** înseamnă că o depășire azi nu face mai probabilă o depășire mîine: momentul depășirilor nu poartă nicio informație.

În ipoteza ambelor proprietăți, numărul depășirilor în $n$ zile, $x = \sum_t I_t$, urmează o distribuție $\mathrm{Binomial}(n, \alpha)$, cu media $n\alpha$ și varianța $n\alpha(1 - \alpha)$. Fiecare test al secțiunii compară secvența observată a depășirilor cu acest reper.

## Depășiri în 250 de zile pentru un VaR 1% corect {h=6c7c45b3}

Graficul arată distribuția $\mathrm{Binomial}(250; 0{,}01)$: numărul depășirilor într-un an de tranzacționare pentru un VaR 1% exact corect. Barele sînt probabilitățile, etichetele probabilitățile cumulate, iar culorile zonele Basel: verde pentru 0–4 depășiri, galben pentru 5–9 și roșu pentru 10 sau mai multe.

Un model corect are în medie 2,5 depășiri, dar dispersia este mare. El nu are nicio depășire cu probabilitatea 8,11% și are 5 sau mai multe cu probabilitatea $100\% - 89{,}22\%$, adică circa 11%. Un model corect ajunge așadar în zona galbenă cam un an din nouă, iar un model fără nicio depășire într-un an nu este neapărat prea conservator.

Un an de date separă doar aproximativ un model bun de unul prost. Un model a cărui rată reală a depășirilor este 2%, dublul ratei nominale, are în medie 5 depășiri, iar distribuția binomială din jurul lui 5 se suprapune mult cu cea din jurul lui 2,5. Slide-urile următoare cuantifică această putere mică.

## Testul POF al lui Kupiec {h=07389de5}

Testul proporției depășirilor (POF, *proportion of failures*) al lui @Kupiec1995 testează acoperirea necondiționată, $H_0$: $P(I_t = 1) = \alpha$. Cu $x$ depășiri în $n$ zile și rata observată $\hat\pi = x/n$, statistica raportului de verosimilitate compară verosimilitatea binomială cu rata $\alpha$ cu cea cu rata $\hat\pi$:

$$ LR_{uc} = -2\big[(n - x)\ln(1 - \alpha) + x\ln\alpha - (n - x)\ln(1 - \hat\pi) - x\ln\hat\pi\big]. $$ {#eq:kupiec}

În ipoteza $H_0$, $LR_{uc}$ urmează asimptotic o distribuție $\chi^2(1)$, ca orice test al raportului de verosimilitate cu o singură restricție ([Capitolul 6](ch6_ro.html)). Statistica nu este niciodată negativă, deoarece $\hat\pi$ maximizează verosimilitatea, iar ipoteza este respinsă la 5% cînd $LR_{uc} > 3{,}84$.

Pas cu pas, cu $n = 250$ și $x = 7$, deci $\hat\pi = 0{,}028$: $\ln L_0 = 243\ln 0{,}99 + 7\ln 0{,}01 = -34{,}68$ și $\ln L_1 = 243\ln 0{,}972 + 7\ln 0{,}028 = -31{,}93$. Atunci $LR_{uc} = -2(-34{,}68 - (-31{,}93)) = 5{,}50$, cu p-value 0,019: modelul este respins la 5%.

Testul are o putere mică cu un an de date. În 250 de zile, orice număr de depășiri între 1 și 6 este acceptat la 5%, deci un model a cărui rată reală este 2% trece testul cu probabilitatea 76%. Testul este bilateral: și prea puține depășiri duc la respingere, deoarece un VaR exagerat de prudent irosește capital; cu 250 de zile, însă, pe această parte este respins doar un an fără nicio depășire.

## Testul Kupiec pe 21 de ani de prognoze pentru S&P 500 {h=beb6d33e}

Cu un eșantion lung, același test devine precis. VaR 1% GARCH-t pentru S&P 500, între 3 ianuarie 2005 și 18 septembrie 2026, acoperă $n = 5\,460$ de zile și are $x = 95$ de depășiri, o rată observată $\hat\pi = 0{,}0174$, față de $n\alpha = 55$ așteptate. În ipoteza $H_0$, un interval binomial de 95% pentru rata observată merge de la 0,75% la 1,26%; 1,74% se află mult în afara lui.

Pas cu pas: $\ln L_0 = 5365\ln 0{,}99 + 95\ln 0{,}01 = -491{,}4$, $\ln L_1 = 5365\ln(1 - 0{,}0174) + 95\ln 0{,}0174 = -479{,}0$, iar $LR_{uc} = -2(\ln L_0 - \ln L_1) = 24{,}73$, cu un p-value sub 0,001.

Cu 21 de ani de date, chiar și o rată de 1,74% în locul celei de 1% este respinsă clar; cu un an de date ar fi trecut testul. Modelul GARCH-t urmărește bine volatilitatea, cum au arătat numărătorile din crize, dar depășirile lui sînt totuși prea frecvente. O cauză probabilă este că modelul GARCH(1,1) simetric ignoră leverage effect-ul din [Capitolul 9](ch9_ro.html): volatilitatea crește mai mult după o scădere decît după o creștere de aceeași mărime, deci după zilele proaste volatilitatea prognozată este prea mică.

## Testul de independență al lui Christoffersen {h=a4929500}

Testul Kupiec numără depășirile, dar ignoră momentul în care apar. @Christoffersen1998 testează momentul lor numărînd tranzițiile: $n_{ij}$ este numărul zilelor cu $I_{t-1} = i$ și $I_t = j$. Rata depășirilor după o zi fără depășire este $\hat\pi_{01} = n_{01}/(n_{00} + n_{01})$, iar după o zi cu depășire $\hat\pi_{11} = n_{11}/(n_{10} + n_{11})$. Ipoteza nulă $H_0$: $\pi_{01} = \pi_{11}$ spune că o depășire azi nu schimbă șansa unei depășiri mîine. Raportul de verosimilitate compară o singură rată $\hat\pi$ cu cele două rate de tranziție:

$$ LR_{ind} = -2\ln\frac{(1 - \hat\pi)^{n_{00} + n_{10}}\hat\pi^{n_{01} + n_{11}}}{(1 - \hat\pi_{01})^{n_{00}}\hat\pi_{01}^{n_{01}}(1 - \hat\pi_{11})^{n_{10}}\hat\pi_{11}^{n_{11}}}, $$ {#eq:lrind}

asimptotic $\chi^2(1)$ în ipoteza $H_0$. **Acoperirea condiționată** (*conditional coverage*) combină ambele proprietăți: $LR_{cc} = LR_{uc} + LR_{ind}$ este asimptotic $\chi^2(2)$, iar un model este respins la 5% dacă $LR_{cc} > 5{,}99$.

Pentru VaR 1% HS al S&P 500 pe 5 460 de zile, numărul tranzițiilor este $n_{00} = 5313$, $n_{01} = 69$, $n_{10} = 69$ și $n_{11} = 8$. Rata depășirilor după o zi liniștită este $\hat\pi_{01} = 1{,}28\%$, după o depășire $\hat\pi_{11} = 10{,}4\%$: o depășire este de opt ori mai probabilă în ziua de după o altă depășire. Statistica $LR_{ind} = 19{,}43$ are un p-value sub 0,001; depășirile HS apar grupat.

Testul detectează dependența de ordinul întîi, depășiri în zile consecutive. O grupare întinsă pe o săptămînă, cu zile liniștite între depășiri, apare doar slab în numărul tranzițiilor, deci testul este o verificare minimă a independenței, nu una completă.

## Semaforul Basel {h=ffb37e6c}

Regula Basel de backtesting [@BCBS1996b] numără depășirile VaR 1% pe o zi în ultimele 250 de zile. Tabelul dă, pentru fiecare număr $x$, zona, probabilitatea cumulată $P(X \le x)$ în distribuția $\mathrm{Binomial}(250; 0{,}01)$, factorul de adaos (*plus factor*) și multiplicatorul rezultat. În zona verde, 0–4 depășiri, nu se ia nicio măsură. În zona galbenă, 5–9 depășiri, multiplicatorul crește de la 3 cu un adaos între 0,40 și 0,85. În zona roșie, 10 sau mai multe depășiri, multiplicatorul este 4, iar modelul este considerat greșit.

Zonele vin din distribuția binomială a unui model corect: zona galbenă începe acolo unde probabilitatea cumulată trece de 95% (95,88% la 5 depășiri), iar zona roșie acolo unde atinge 99,99% (la 10 depășiri). Un model corect ajunge astfel în zona galbenă cam un an din nouă și în zona roșie aproape niciodată.

Cerința de capital din regulile din 1996 este maximul dintre VaR 1% pe 10 zile de ieri și $(3 + \text{adaos})$ înmulțit cu media VaR 1% pe 10 zile din ultimele 60 de zile; în perioadele obișnuite domină al doilea termen. FRTB păstrează numărarea depășirilor VaR 1%, la nivelul băncii și al fiecărui birou de tranzacționare, potrivit capitolului MAR32 [@BCBS2023a].

## Depășirile în timp {h=ff17d9b3}

Graficul arată depășirile celor patru prognoze VaR 1% pe fereastră mobilă pentru S&P 500, din 2005 pînă în 2026; fiecare marcaj este o zi în care VaR-ul metodei a fost depășit, iar benzile hașurate sînt cele două perioade de stres. Pentru circa 5 460 de zile, fiecare metodă ar trebui să aibă aproximativ 55 de depășiri; legenda dă 77 pentru HS, 148 pentru metoda Normală, 95 pentru GARCH-t și 86 pentru FHS.

Marcajele HS și ale metodei Normale formează grupuri dense: în 2007–2008, în 2011, în 2022, iar pentru metoda Normală și în 2018–2019 și 2025. Fiecare grup este o perioadă în care volatilitatea a crescut mai repede decît o putea urmări fereastra de 500 de zile. Între grupuri există perioade lungi fără nicio depășire, cînd fereastra mai conținea o perioadă turbulentă anterioară.

Marcajele GARCH-t și FHS sînt răspîndite mult mai uniform în timp, cum ar trebui să fie depășirile independente. Ele sînt însă mai numeroase decît cele HS în anii liniștiți: cînd volatilitatea este mică, VaR-ul condiționat este și el mic, iar o zi proastă obișnuită îl depășește. Un model condiționat corect ar arăta exact această răspîndire uniformă, dar cu aproximativ 55 de marcaje în total.

## Rezultatele backtesting-ului: S&P 500 și DAX {h=b6d07fa4}

Tabelul rezumă testele pentru VaR 1% din 3 ianuarie 2005, circa 5 500 de zile. Pentru fiecare metodă, el dă numărul depășirilor $x$, rata în procente, p-value-urile testului Kupiec (POF), ale testului de independență Christoffersen (ind.) și ale testului de acoperire condiționată (ac.c.), cel mai mare număr de depășiri într-o fereastră de 250 de zile și statistica $Z_2$ a backtesting-ului ES de pe ultimul slide al acestei secțiuni, cu p-value-ul ei.

Metoda Normală este respinsă de fiecare test pe ambii indici: 148 de depășiri (2,71%) pentru S&P 500 și 133 (2,41%) pentru DAX, cu pînă la 29 și 19 depășiri într-o singură fereastră de 250 de zile, adînc în zona roșie. HS are o rată aproape corectă pentru DAX, 1,21%, cu un p-value Kupiec de 0,121, dar depășirile lui apar grupat (p-value-ul testului de independență este sub 0,001), deci nu trece testul de acoperire condiționată.

GARCH-t și FHS au depășiri independente, cu p-value-uri ale testului de independență între 0,060 și 0,549, și o fereastră de 250 de zile cu cel mult 9 sau 10 depășiri, în loc de 12 pînă la 29 pentru HS și metoda Normală. Ratele lor sînt însă între 1,4% și 1,7%, deci testul Kupiec le respinge pe ambii indici. Modelele condiționate rezolvă problema grupării, dar nu și problema nivelului: cozile lor sînt încă puțin prea subțiri pentru S&P 500 și DAX.

## Rezultatele backtesting-ului: BET și Bitcoin {h=77b6f8ea}

Aceleași teste pentru BET din 2005 și pentru Bitcoin din 1 ianuarie 2017 dau o imagine mai favorabilă. Pentru BET, GARCH-t și FHS au cîte 65 de depășiri, o rată de 1,20%, și trec toate cele trei teste: p-value Kupiec 0,158, independență 0,807 și 0,240, acoperire condiționată 0,358 și 0,185. HS are o rată asemănătoare, 1,22%, dar depășirile lui apar grupat (p-value-ul testului de independență este 0,001), iar metoda Normală este respinsă peste tot.

Pentru Bitcoin, HS trece toate cele trei teste, cu 33 de depășiri, o rată de 0,93% și p-value-urile 0,672, 0,316 și 0,553. FHS este acum prea prudent: 24 de depășiri, o rată de 0,68%, respins de testul Kupiec (p = 0,040) și de testul de acoperire condiționată (p = 0,044). GARCH-t are 1,35% și este respins de testul Kupiec la 5% (p = 0,045), dar nu și de testul de acoperire condiționată.

Nicio metodă nu cîștigă peste tot. Ordinea depinde de piață: metodele condiționate sînt cele mai bune pentru indicii de acțiuni, în timp ce schimbările extreme și frecvente ale volatilității Bitcoin sînt tratate rezonabil chiar și de o fereastră istorică de doi ani. Singurul rezultat constant este metoda Normală: ea eșuează pe fiecare serie.

## Clasamentul prognozelor și zona actuală a semaforului Basel {h=7f4e430f}

Testele spun dacă un model este acceptabil; o **funcție de scor** spune care dintre două modele este mai bun. Deoarece VaR este elicitabil, media pierderii cuantile

$$ \bar S = \frac1n\sum_t(\alpha - I_t)(r_t + \mathrm{VaR}_t) $$ {#eq:avgscore}

ordonează prognozele [@Gneiting2011]: cu cît media este mai mică, cu atît prognozele sînt mai bune. Ea recompensează atît rata corectă a depășirilor, cît și urmărirea volatilității, deoarece o prognoză prea mare în zilele liniștite este penalizată de termenul $\alpha(r_t + \mathrm{VaR}_t)$, iar una prea mică în zilele proaste de termenul $(\alpha - 1)(r_t + \mathrm{VaR}_t)$.

Pentru indicii de acțiuni, GARCH-t și FHS au cea mai mică pierdere cuantilă medie (×100), de exemplu 3,71 și 3,68 pentru S&P 500, față de 5,17 pentru HS și 5,80 pentru metoda Normală, iar metoda Normală o are pe cea mai mare peste tot. Pentru Bitcoin, cele patru metode sînt apropiate, de la 13,53 (GARCH-t) la 14,52 (Normală), în acord cu testele de pe slide-ul precedent.

Jumătatea dreaptă a tabelului numără depășirile din ultimele 250 de zile. Toate modelele sînt azi în zona verde, cu excepția VaR Normal pentru BET, cu 6 depășiri, în zona galbenă. O fereastră recentă spune puțin despre un model: pentru S&P 500, HS are 0 depășiri în ultimul an, iar GARCH-t 3, deși pe 21 de ani depășirile HS apar grupat, iar cele GARCH-t nu.

## Semaforul Basel în timp {h=54331b2e}

Graficul urmărește numărătoarea Basel pentru S&P 500: pentru fiecare zi de la 28 decembrie 2005, sfîrșitul primei ferestre de 250 de zile, numărul depășirilor VaR 1% în cele 250 de zile anterioare, pentru HS, metoda Normală și GARCH-t. Liniile întrerupte sînt limitele superioare ale zonei verzi (4) și ale zonei galbene (9); deasupra liniei de sus se află zona roșie.

VaR Normal a fost în zona roșie în 33,3% dintre zile, HS în 11,8%, iar GARCH-t doar în 0,5%; cele mai mari numere de depășiri au fost 29, 18 și 10. Metoda Normală a atins 29 de depășiri la sfîrșitul lui 2008 și a revenit în zona roșie în 2015–2016, 2019–2021, 2022–2023 și 2025. HS a fost în zona roșie în cea mai mare parte din 2008–2009 și din nou în 2020–2021.

O bancă ce ar fi folosit VaR Normal sau HS ar fi plătit penalizarea zonei roșii în cea mai mare parte din 2008–2009, exact cînd capitalul era cel mai greu de găsit, în timp ce o bancă ce folosea GARCH-t ar fi rămas în zona galbenă în timpul crizei, atingînd 10 depășiri doar pentru scurt timp. Semaforul recompensează un model care urmărește volatilitatea, chiar și unul a cărui rată totală este prea mare.

## Backtesting pentru ES: Acerbi și Székely {h=e753df6a}

ES nu este elicitabil singur, dar poate fi testat împreună cu VaR [@AcerbiSzekely2014]. A doua lor statistică de test este

$$ Z_2 = \frac{1}{T\alpha}\sum_{t=1}^{T}\frac{r_t I_t}{\mathrm{ES}_t} + 1, \qquad I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t\}, \quad \alpha = 2{,}5\%, $$ {#eq:z2}

unde $T$ este numărul zilelor de test, iar $\mathrm{VaR}_t$, $\mathrm{ES}_t$ sînt prognozele pentru ziua $t$. Fiecare depășire adaugă $r_t/\mathrm{ES}_t$, un număr negativ apropiat de −1 dacă pierderea este cît ES-ul prognozat. Dacă VaR și ES sînt amîndouă corecte, suma are media $-T\alpha$, iar $E[Z_2] = 0$. Un $Z_2$ negativ înseamnă că pierderile de dincolo de VaR sînt mai mari decît ES-ul prognozat, sau că depășirile sînt prea dese, sau amîndouă.

Distribuția lui $Z_2$ în ipoteza nulă depinde de distribuțiile prognozate, deci p-value-ul se obține prin simulare: extragem un randament din distribuția prognozată a fiecărei zile, recalculăm $Z_2$, repetăm de 1000 de ori și comparăm valoarea observată cu cele simulate. Valorile și p-value-urile se află în ultima coloană a celor două tabele de backtesting.

Pentru S&P 500, DAX și BET, fiecare metodă are un $Z_2$ negativ, cu un p-value sub 0,05: toate cele patru prognoze ES 2,5% sînt prea mici pentru indicii de acțiuni. FHS este cel mai aproape, cu $Z_2$ între −0,18 și −0,20, deoarece preia coada lui $z$ din date; metoda Normală este cea mai departe, între −0,83 și −1,15. Pentru Bitcoin, HS (p = 0,224) și FHS ($Z_2 = 0{,}12$, p = 0,834) nu sînt respinse.

## Studiu de caz: modelele VaR ale băncilor comerciale {h=4a373707}

@BerkowitzOBrien2002 au studiat primii prognozele VaR raportate efectiv de mari bănci americane, comparîndu-le cu P&L-ul zilnic din tranzacționare al băncilor. Schema este un backtesting pe date reale de reglementare: VaR 1% al fiecărei bănci față de P&L-ul ei realizat, cu un model GARCH simplu al P&L-ului fiecărei bănci ca reper.

VaR-ul băncilor a fost în medie prudent, cu mai puține depășiri decît rata nominală, dar depășirile care au avut loc au apărut grupat. Reperul GARCH a dat valori VaR mai mici, cu o acoperire comparabilă: un model simplu al seriei P&L a urmărit schimbările volatilității mai bine decît modelele structurale ale băncilor.

Același tipar apare în datele acestui capitol: VaR-ul pe fereastră este fie prea mare în perioadele liniștite, fie prea lent în perioadele turbulente, în timp ce VaR-ul condiționat se adaptează. O altă direcție de cercetare prognozează direct cuantila, fără un model pentru întreaga distribuție: modelele CAViaR ale lui @EngleManganelli2004 specifică o autoregresie pentru cuantila însăși și o estimează cu pierderea cuantilă @eq:pinball.

## @end

Un VaR corect produce depășiri Bernoulli i.i.d. Testul Kupiec @eq:kupiec verifică rata lor, testul Christoffersen @eq:lrind independența lor, semaforul Basel transformă numărul lor în capital, iar $Z_2$ @eq:z2 testează ES. Pe 21 de ani, VaR Normal eșuează peste tot, HS are rata corectă pe unele serii, dar depășiri grupate, iar GARCH-t și FHS au depășiri independente, dar o rată de 1,4–1,7% pentru S&P 500 și DAX; toate prognozele ES sînt prea mici pentru indicii de acțiuni. [Secțiunea următoare](#sec-8) transformă unul dintre rezultatele capitolului într-o întrebare de cercetare deschisă.

# AI pentru descoperire științifică

Secțiunea pornește de la o întrebare deschisă ridicată de rezultatele pentru portofoliu: dacă dependența în cozi dintre piața din București și piețele mondiale crește brusc în crize și revine apoi la nivelul anterior. Ea arată unde pot ajuta instrumentele AI un cercetător care o studiază și ce verificări nu pot înlocui.

## O întrebare de cercetare deschisă {h=7ee15988}

Crește brusc dependența în cozi dintre BVB și piețele mondiale în crize și revine apoi la nivelul anterior? Rezultatele capitolului indică această direcție. Corelația BET–S&P 500 a fost 0,20 într-un an liniștit și 0,71 în primăvara lui 2020; o copulă t constantă dă o dependență în coadă $\lambda_L = 0{,}096$; iar un VaR de portofoliu construit pe o copulă constantă rămîne sub VaR-ul istoric ([secțiunea despre VaR de portofoliu](#sec-6)).

Întrebarea rămîne deschisă din trei motive. Crizele sînt puține, deci orice estimare a dependenței din crize se sprijină pe cîteva episoade. BVB se tranzacționează la alte ore decît New York-ul, deci dependența din aceeași zi amestecă co-mișcarea reală cu efecte de decalaj. Iar estimările din coadă au nevoie de multe observații, în timp ce o fereastră de criză conține doar cîteva zile proaste comune.

Lucrări înrudite folosesc entropia ca măsură a riscului de piață, un alt drum spre coadă: @PeleLazarDufour2017 și @PeleMazurencuMarinescuPele2019 sînt lecturi suplimentare. Instrumentele AI pot accelera un studiu despre această întrebare, dar rezultatele lor trebuie verificate [@WangEtAl2023].

## Utilizări posibile ale instrumentelor AI {h=f712c77d}

Un asistent AI poate ajuta în trei etape. Pentru literatură, poate lista studiile despre copule variabile în timp și despre contagiunea dintre piețele emergente și cele dezvoltate, un punct de plecare pentru a plasa BET în context. Pentru cod, poate scrie o primă versiune a estimărilor copulei t pe ferestre mobile de 250 de zile pentru BET și S&P 500, cu $\rho$, $\nu$ și $\lambda_L$ reprezentate în timp. Pentru robustețe, poate propune și implementa variante: randamente pe două zile, pentru a absorbi orele de tranzacționare diferite, DAX în locul S&P 500, alte lungimi ale ferestrei.

Exemplul de prompt de pe slide este precis: numește copula, metoda de estimare, datele, lungimea ferestrei de 250 de zile, pasul de 21 de zile și cele trei rezultate cerute. Un prompt precis produce cod ale cărui alegeri pot fi verificate față de definițiile din acest capitol; unul vag produce cod ale cărui alegeri rămîn ascunse.

Rezultatul fiecărei etape este o ciornă. Lista de lucrări trebuie verificată referință cu referință, iar codul trebuie testat pe rezultate cunoscute, de exemplu reproducînd estimările copulei t pe întregul eșantion, $\rho = 0{,}157$ și $\hat\nu = 4{,}50$, înainte de a-l folosi pe ferestre mobile.

## Verificări necesare {h=d5b7c45a}

Mai multe erori sînt frecvente în codul de risc generat cu AI. Prima este convenția VaR: „VaR 1%” este pierderea depășită cu probabilitatea 1%, un număr pozitiv; un răspuns care folosește probabilitatea complementară sau raportează un VaR negativ amestecă convențiile. A doua este alinierea datelor: prețurile trebuie aliniate pe zilele comune de tranzacționare înainte de calculul randamentelor, iar BVB și New York se tranzacționează la ore diferite.

A treia privește copula însăși. Pseudo-observațiile sînt rangurile împărțite la $n + 1$, nu la $n$: altfel cea mai mare observație se află exact în 1, unde transformările cuantilă $\Phi^{-1}$ și $t_\nu^{-1}$ sînt infinite, iar log-verosimilitatea copulei nu este definită. Dependența în coadă estimată din 250 de zile se sprijină pe cîteva zile proaste comune, deci incertitudinea ei trebuie raportată, de exemplu prin bootstrap-ul din [Capitolul 2](ch2_ro.html).

Ferestrele suprapuse au în comun cea mai mare parte a datelor, deci estimările consecutive nu sînt dovezi independente: un șir de estimări pe fereastră mobilă care trece de mai multe ori de un prag este o singură constatare, nu mai multe. În fine, fiecare referință trebuie să existe: DOI-ul fiecărei lucrări citate trebuie verificat, deoarece referințele inventate, cu titluri plauzibile, sînt o eroare cunoscută a modelelor de limbaj.

## Idee de proiect {h=36e30d5f}

Proiectul întreabă dacă o copulă variabilă în timp dă un VaR de portofoliu mai bun decît o copulă constantă, pentru un portofoliu de acțiuni BVB și mondiale. Datele sînt închiderile zilnice ale BET, BET-TR (indicele BET cu dividendele reinvestite), S&P 500 și DAX din 2005, de la EODHD (EOD Historical Data).

Pașii urmează capitolul. Întîi, ajustați marginale GARCH-t pentru fiecare indice ([Capitolul 9](ch9_ro.html)) și modelați dependența cu o copulă t constantă și cu o copulă t estimată pe fereastră mobilă. Apoi, calculați în afara eșantionului VaR 1% și ES 2,5% pe o zi pentru un portofoliu 50/50, din 2008. În final, aplicați testele Kupiec, Christoffersen și $Z_2$ ambelor variante, cu atenție specială pentru 2008, 2020 și 2022.

Livrabilul este un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele. Orice folosire a instrumentelor AI se declară, împreună cu erorile AI găsite și corectate.

## @end

Corelația BET–S&P 500 crește de la 0,20 într-un an liniștit la 0,71 într-o criză, iar o copulă constantă subestimează VaR-ul portofoliului. Dacă dependența în coadă crește brusc și revine apoi este o întrebare deschisă, deoarece crizele sînt puține și orele de tranzacționare diferă. Instrumentele AI pot schița căutarea bibliografică și codul copulei pe fereastră mobilă; cercetătorul verifică convenția VaR, alinierea, pseudo-observațiile, incertitudinea și referințele.

# Rezumat

Rezultatele capitolului se reduc la cîteva afirmații despre măsurarea, estimarea și testarea riscului din coadă, un set de formule și trei întrebări de autoevaluare.

## Idei de reținut {h=00595f95}

VaR 1% este opusul cuantilei de 1%, @eq:var: pierderea depășită într-o zi din o sută. ES 2,5%, @eq:es, este pierderea medie din cele mai proaste 2,5% dintre zile. ES este coerent; VaR poate penaliza diversificarea, ca în exemplul celor două obligațiuni. VaR este elicitabil, iar ES este elicitabil doar împreună cu VaR.

Pe date zilnice, VaR 1% Normal este prea mic cu 14–24% pe toate cele șase serii. Simularea istorică, Student-t și EVT sînt de acord la 1%, iar Cornish–Fisher nu mai funcționează cînd boltirea este mare. VaR-ul condiționat, GARCH-t și FHS, urmărește perioadele turbulente și trece teste pe care metodele pe fereastră nu le trec, mai ales independența depășirilor.

Pentru portofolii, corelația crește în crize, iar dependența în coadă, surprinsă de copula t, contează pentru prăbușirile simultane. Backtesting-ul combină patru instrumente: testul Kupiec pentru rata depășirilor, testul Christoffersen pentru gruparea lor, semaforul Basel pentru reglementare și statistica $Z_2$ pentru ES.

## Formule de reținut {h=d21809a0}

Tabelul adună formulele capitolului. VaR și ES sînt definite de @eq:var și @eq:es și estimate prin simulare istorică cu @eq:hs. Pentru distribuția Normală ele au formulele explicite @eq:normal, pentru Student-t @eq:tdist, iar estimatorul EVT @eq:pot le dă pe amîndouă dintr-o distribuție GPD ajustată peste un prag.

VaR-ul condiționat @eq:cond scalează cuantila inovației standardizate cu volatilitatea de mîine. Varianța portofoliului @eq:portvar combină volatilitățile și corelațiile, iar dependența în coadă a copulei t este @eq:tcopula.

Statistica Kupiec @eq:kupiec testează rata depășirilor, statistica Christoffersen @eq:lrind independența lor, iar suma lor $LR_{cc}$ testează acoperirea condiționată față de distribuția $\chi^2(2)$. Statistica Acerbi–Székely @eq:z2 testează împreună ES și VaR.

## Autoevaluare {h=eb9c80e9}

Prima întrebare pornește de la raportul unei divizii: „VaR 1% = −2,4%”. Semnul este greșit: −2,4% este cuantila $q_{0{,}01}$ a randamentelor, iar din @eq:var VaR 1% este $-q_{0{,}01} = 2{,}4\%$, o pierdere pozitivă.

A doua întrebare descrie un model cu 3 depășiri în 250 de zile, toate în aceeași săptămînă. Testul Kupiec nu detectează problema, deoarece 3 este aproape de 2,5, valoarea așteptată. Testul de independență al lui Christoffersen o poate detecta dacă depășirile cad în zile consecutive, deoarece acestea produc tranziții de la o depășire la o altă depășire; testul privește doar trecerile de la o zi la alta, astfel că trei depășiri lunea, miercurea și vinerea i-ar scăpa; iar cu doar trei depășiri într-un an, puterea lui este limitată, motiv pentru care testul se aplică pe eșantioane lungi.

A treia întrebare cere dacă două portofolii cu același VaR 1% pot avea ES 2,5% diferit. Pot: ES depinde de întreaga coadă de dincolo de cuantilă, VaR de un singur punct. Capitolul următor, [Capitolul 11](ch11_ro.html), tratează ipoteza piețelor fractale și memoria lungă.
