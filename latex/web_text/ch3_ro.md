<!--
Textul de curs al versiunii web, capitolul 3 (RO). Generat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), @after (text după blocul interactiv), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie → Autor (an), [@Cheie] → (Autor, an); cheile vin din bibliografia deck-ului.
Hash-urile curente: python3 build_web_lecture.py 3 --hashes
-->

## @intro

@text
Capitolul 2 a arătat că randamentele zilnice au cozi mai groase (*heavy tails*) decît distribuția Normală. Un model pentru randamentele zilnice nu este însă niciodată doar un model pentru o singură zi. Randamentele logaritmice se adună în timp, astfel că un randament logaritmic lunar este suma a circa 21 de randamente logaritmice zilnice, iar legea randamentului zilnic fixează legea oricărui orizont mai lung. Capitolul pune întrebarea: ce legi de probabilitate pot descrie randamentele zilnice și, în același timp, sumele lor săptămînale și lunare?

Răspunsul teoriei probabilităților este familia legilor stabile (*stable laws*). Ele sînt singurele legi ale căror sume își păstrează forma și, conform teoremei limită centrale generalizate (GCLT, *generalised central limit theorem*), singurele limite posibile ale sumelor normalizate de variabile independente și identic distribuite (i.i.d.). @Mandelbrot1963 le-a propus pentru prețurile bumbacului, cu un indice de stabilitate $\alpha \approx 1{,}7$, ceea ce implică o varianță infinită. Capitolul dezvoltă teoria acestor legi, simularea și estimarea lor, apoi le estimează pe randamentele logaritmice zilnice a patru serii: indicele BET al Bursei de Valori București, S&P 500, DAX și Bitcoin.

Pe aceste serii, estimările de verosimilitate maximă (ML, *maximum likelihood*) ale lui $\alpha$ se află între 1,410 și 1,606, departe de valoarea 2 a distribuției Normale, iar legea stabilă descrie centrul distribuției mult mai bine decît distribuția Normală. În cozile îndepărtate însă, ea prevede mult mai multe pierderi extreme decît s-au observat: pentru S&P 500, 19,1 zile cu o pierdere de peste 10% în 6 714 zile, față de o singură zi observată. Ultima parte a capitolului testează direct ipoteza varianței infinite și conchide că randamentele zilnice au cozi groase, dar varianță finită.

@chart sfm_ch3_fit_density_1
Randamentele logaritmice zilnice ale BET și S&P 500 (puncte, histogramă pe scară logaritmică) și trei legi estimate: Normală (roșu), Student-t (verde) și stabilă, estimată prin verosimilitate maximă (albastru); pe axa orizontală este randamentul logaritmic zilnic în %, pe cea verticală densitatea pe scară logaritmică.

@goals
- **Enunțați** proprietatea de stabilitate și calculați scala unei sume de variabile stabile.
- **Explicați** teorema limită centrală generalizată și situațiile în care sumele nu se apropie de distribuția Normală.
- **Interpretați** cei patru parametri $(\alpha, \beta, \gamma, \delta)$ și faceți conversia între parametrizările S0 și S1.
- **Stabiliți** ce momente există pentru un $\alpha$ dat și comparați probabilitățile din cozi cu cele ale distribuției Normale.
- **Simulați** variabile stabile și estimați parametrii lor prin cuantile și prin verosimilitate maximă.
- **Evaluați**, pe randamentele BET, S&P 500, DAX și Bitcoin, unde funcționează modelul stabil și unde eșuează.

@plan
- Mandelbrot și prețurile bumbacului | Originea istorică: prețurile bumbacului, ipoteza lui Mandelbrot, testele lui Fama și întrebarea despre legea unei sume.
- Stabilitatea la adunare | Definiția legilor stabile, regula scalei $c^\alpha = a^\alpha + b^\alpha$ și consecințele ei pentru orizonturi și diversificare.
- Teorema limită centrală generalizată | Legile stabile ca singure limite ale sumelor normalizate; domenii de atracție.
- Parametrii și funcția caracteristică | Cei patru parametri, parametrizările S0 și S1 și cele trei legi cu densitate în formă închisă.
- Cozi și momente | Cozile de tip putere, existența momentelor și consecințele varianței infinite pentru măsurile de risc.
- Simulare | Metoda Chambers–Mallows–Stuck, în cazul simetric și în cel asimetric.
- Estimare | Metoda cuantilelor a lui McCulloch și verosimilitatea maximă, comparate într-un studiu Monte Carlo.
- Estimarea pe randamente reale | Legile stabilă, Student-t și Normală estimate pe BET, S&P 500, DAX și Bitcoin.
- Critica: varianță finită sau infinită? | Trei teste ale ipotezei varianței infinite și explicațiile aspectului stabil.
- AI pentru descoperire științifică | O întrebare deschisă despre cozile Bitcoin, verificările necesare și o idee de proiect.
- Rezumat | Formulele capitolului și întrebări de autoevaluare.

# Mandelbrot și prețurile bumbacului

Legile stabile au intrat în finanțe printr-o observație empirică, nu prin teorie. Secțiunea arată cum a ajuns Benoit Mandelbrot să le propună, cum a testat Eugene Fama propunerea pe randamentele acțiunilor și de ce întrebarea privește, de fapt, legea unei sume de randamente.

## Bumbacul: o piață veche, cu serii lungi de prețuri {h=837343a5}

În secolul al XIX-lea, bumbacul era una dintre cele mai tranzacționate mărfuri din lume. Bursele din New Orleans, New York și Liverpool cotau prețuri în fiecare zi, iar aceste cotații au produs unele dintre cele mai lungi serii zilnice și lunare de prețuri din economie. Seriile lungi contează pentru studiul cozilor: variațiile extreme sînt rare prin definiție și doar o serie lungă conține destule pentru a spune cît de des apar.

La începutul anilor 1960, aceste serii au ajuns la Benoit Mandelbrot, pe atunci cercetător la IBM (International Business Machines). El nu a studiat nivelul prețului, ci variațiile logaritmului prețului, adică randamentele logaritmice din Capitolul 1. Lucrarea lui, @Mandelbrot1963, este punctul de plecare al studiului cozilor groase în finanțe: a întrebat ce lege de probabilitate descrie aceste variații și a dat un răspuns care se îndepărta de distribuția Normală folosită pînă atunci.

Tabloul de pe slide, *Un birou de bumbac din New Orleans* (1873) de Edgar Degas, înfățișează comerțul care a generat aceste serii.

## Observațiile lui Mandelbrot asupra prețurilor bumbacului {h=83453564}

Prima observație a lui Mandelbrot a fost că seriile de bumbac conțineau mult mai multe variații mari decît permite distribuția Normală. Un simptom înrudit era că varianța de selecție (*sample variance*) nu se stabiliza pe măsură ce creștea eșantionul: fiecare nouă variație extremă o împingea din nou în sus, în loc să o lase să conveargă către o valoare fixă, cum s-ar întîmpla pentru o variabilă cu varianță finită.

A doua observație a fost că forma distribuției era aceeași la orice orizont. După rescalare, variațiile zilnice și cele lunare arătau la fel, ceea ce înseamnă că suma multor variații zilnice păstra forma unei singure variații zilnice. Pentru distribuția Normală, proprietatea este cunoscută, fiindcă o sumă de variabile Normale este tot Normală. Mandelbrot a văzut-o însă împreună cu cozi groase, pe care distribuția Normală nu le poate produce.

Propunerea lui a reunit cele două observații: variațiile de preț urmează legi stabile de tip Pareto (*stable Paretian laws*) cu un indice $\alpha \approx 1{,}7$ [@Mandelbrot1963]. „De tip Pareto” se referă la cozi care scad ca o putere a lui $x$, ca în legea veniturilor a lui Pareto. Consecința, discutată pe tot parcursul capitolului, este că varianța variațiilor de preț ar fi infinită.

## Fama testează ideea pe acțiuni {h=358e1d16}

@Fama1965 a mutat ipoteza de pe mărfuri pe acțiuni. El a studiat randamentele zilnice ale celor 30 de acțiuni din Dow Jones Industrial Average și a găsit, pentru fiecare acțiune, mai multe randamente mari decît prevede distribuția Normală. Estimările lui pentru indicele $\alpha$ erau sub 2, ceea ce susținea ipoteza lui Mandelbrot.

Au urmat metode de estimare pentru legile stabile simetrice. @FamaRoll1968 și @FamaRoll1971 au construit estimatori bazați pe cuantilele de selecție, împreună cu tabelele necesare pentru a-i folosi. Aceste metode sînt precursoarele estimatorului prin cuantile al lui McCulloch, prezentat în [secțiunea despre estimare](#sec-7).

În mai puțin de zece ani, dovezile au început să se întoarcă împotriva varianței infinite. [Secțiunea de critică](#sec-9) de la finalul capitolului revine la aceste studii și le repetă testele pe date actuale.

## Importanța legii unei sume {h=59114931}

Un randament logaritmic lunar este suma a circa 21 de randamente logaritmice zilnice. Un model pentru randamentele zilnice implică deci un model pentru randamentele săptămînale, lunare și anuale, iar un model bun trebuie să rămînă coerent de la un orizont la altul. Distribuția Normală trece acest test, pentru că își păstrează forma la adunare (Capitolul 2), dar are cozi subțiri și nu surprinde variațiile zilnice mari.

Un candidat firesc cu cozi groase este distribuția Student-t. Întrebare pentru sală: este suma a 21 de variabile Student-t independente tot o variabilă Student-t? Răspunsul este nu. Familia Student-t nu este închisă la adunare, iar dacă varianța ei este finită, teorema limită centrală (CLT, *central limit theorem*) împinge suma către distribuția Normală. Un model Student-t pentru randamentele zilnice implică așadar un model aproximativ Normal pentru randamentele lunare.

Obiectivul secțiunilor următoare decurge din acest răspuns: identificarea tuturor legilor ale căror sume își păstrează forma. Acestea sînt legile stabile.

## @end

Cele două observații ale lui Mandelbrot, prea multe variații mari și aceeași formă la orice orizont, au condus la ipoteza legilor stabile de tip Pareto, cu $\alpha \approx 1{,}7$ și varianță infinită. Studiul lui Fama pe acțiunile din Dow Jones a susținut-o, iar întrebarea pe care o ridică privește legea unei sume de randamente. [Secțiunea următoare](#sec-2) definește legile care își păstrează forma la adunare.

# Stabilitatea la adunare

Secțiunea definește legile stabile prin comportamentul lor la adunare și deduce regula care dă scala unei sume. Regula depinde de un singur număr, indicele de stabilitate $\alpha$, și are consecințe directe pentru riscul pe orizonturi lungi și pentru diversificare.

## Definiție: legile stabile {h=c7778306}

O variabilă aleatoare $X$ este stabilă dacă, pentru două copii independente $X_1, X_2$ ale lui $X$ și orice ponderi $a, b > 0$, există o scală $c > 0$ și o deplasare $d$ astfel încît

$$ aX_1 + bX_2 \overset{d}{=} cX + d, $$ {#eq:stable}

unde $\overset{d}{=}$ înseamnă „are aceeași distribuție ca”. Altfel spus, o sumă ponderată a două copii are aceeași formă ca o copie; se schimbă doar scala $c$ și poziția $d$. Dacă $d = 0$ pentru orice $a$ și $b$, legea este strict stabilă.

Prin inducție, proprietatea se extinde la $n$ copii independente: $X_1 + \dots + X_n \overset{d}{=} c_n X + d_n$. Factorul de scală nu poate fi oarecare. Singurul factor posibil este $c_n = n^{1/\alpha}$, cu $0 < \alpha \le 2$ [@Nolan2020]. Exponentul $\alpha$ se numește indicele de stabilitate (*index of stability*), sau exponentul caracteristic, sau tail index, și indexează întreaga familie.

Limita superioară $\alpha \le 2$ nu este o convenție. Pentru $\alpha = 2$ factorul este $\sqrt{n}$, regula cunoscută pentru sume de variabile cu varianță finită; nicio lege stabilă nu are un factor care să crească mai încet decît $\sqrt{n}$.

## Două exemple: distribuția Normală și distribuția Cauchy {h=b6ca1776}

Distribuția Normală este primul exemplu. Dacă $X_1, X_2 \sim N(0, \sigma^2)$ sînt independente, atunci $aX_1 + bX_2 \sim N(0, (a^2 + b^2)\sigma^2)$, care are forma lui $X$ cu scala $c = \sqrt{a^2 + b^2}$. Varianțele se adună, deci $c^2 = a^2 + b^2$: acesta este cazul $\alpha = 2$.

Al doilea exemplu este distribuția Cauchy, cu densitatea $f(x) = 1/(\pi(1 + x^2))$. O sumă ponderată $aX_1 + bX_2$ a două variabile Cauchy standard independente este Cauchy cu scala $c = a + b$. Scalele se adună liniar, $c^1 = a^1 + b^1$: acesta este cazul $\alpha = 1$.

Cazul Cauchy are o consecință surprinzătoare. Luînd $a = b = 1/n$ pentru $n$ copii, media a $n$ extrageri Cauchy este tot Cauchy standard: medierea nu reduce deloc dispersia. Legea numerelor mari nu se aplică, pentru că distribuția Cauchy nu are medie.

## Regula generală și un exemplu lucrat {h=23e5b9b3}

Cele două exemple sînt cazuri particulare ale unei singure reguli. Pentru copii independente $X_1, X_2$ ale unei $X$ strict stabile cu indicele $\alpha$, scala sumei ponderate satisface

$$ c^\alpha = a^\alpha + b^\alpha, \qquad c = (a^\alpha + b^\alpha)^{1/\alpha}. $$ {#eq:scale}

Exemplul lucrat ia $a = 3$ și $b = 4$. Pentru $\alpha = 2$, scala sumei este $c = \sqrt{9 + 16} = 5{,}00$; pentru $\alpha = 1{,}5$ este $c = (3^{1{,}5} + 4^{1{,}5})^{1/1{,}5} = 5{,}58$; pentru $\alpha = 1$ este $c = 3 + 4 = 7{,}00$. Cu cît $\alpha$ este mai mic, cu atît scala sumei este mai mare. Cu cozi groase, un singur termen mare poate domina întreaga sumă, astfel că suma este mai largă decît indică regula distribuției Normale.

Ce credeți? Pentru $\alpha = 1{,}5$, care este factorul de scală al sumei a două copii? Cu $a = b = 1$ în @eq:scale se obține $c_2 = 2^{1/1{,}5} = 2^{2/3} = 1{,}59$, mai mare decît factorul $\sqrt{2} \approx 1{,}41$ al distribuției Normale.

## Stabilitatea într-o simulare {h=b3ae32d7}

Graficul verifică proprietatea de stabilitate prin simulare. El folosește notația din [secțiunea despre parametri](#sec-4): $S(\alpha; \beta; \gamma; \delta)$ este legea stabilă cu indicele $\alpha$, asimetria $\beta$, scala $\gamma$ și poziția $\delta$. Fiecare punct este o pereche de cuantile: pe axa orizontală, o cuantilă a unei singure extrageri din $S(1{,}7; 0; 1; 0)$, pe axa verticală aceeași cuantilă a unei sume rescalate de 10 extrageri. Dacă cele două legi coincid, punctele se află pe prima bisectoare (*45-degree line*).

Punctele albastre împart suma celor 10 extrageri la $10^{1/1{,}7} = 3{,}87$, factorul $n^{1/\alpha}$ al stabilității. Ele se află pe prima bisectoare pe tot domeniul: suma rescalată are aceeași lege ca o singură extragere. Punctele roșii împart aceleași sume la $\sqrt{10} = 3{,}16$, regula distribuției Normale. Împărțirea este prea slabă, astfel că sumei rescalate îi rămîn cozi prea largi: cuantila ei de 99% este 6,31, față de 5,04 pentru o singură extragere.

Lecția pentru randamente este că regula rădăcinii pătrate, folosită pentru a trece volatilitatea de la un orizont la altul, este o proprietate a cazului $\alpha = 2$. Sub o lege stabilă cu $\alpha < 2$, ea subestimează dispersia sumei.

## Consecințe: orizonturi și diversificare {h=938c3af2}

Prima consecință privește orizontul. Dacă randamentele zilnice sînt i.i.d. $S(\alpha; 0; \gamma; 0)$, suma pe 20 de zile are scala $20^{1/\alpha}\gamma$. Pentru $\alpha = 2$ aceasta este $4{,}47\,\gamma$, regula rădăcinii pătrate a timpului (*square-root-of-time rule*); pentru $\alpha = 1{,}7$ este $5{,}83\,\gamma$; pentru $\alpha = 1{,}5$ este $7{,}37\,\gamma$. Cu $\alpha < 2$, riscul pe orizonturi lungi crește mai repede decît $\sqrt{n}$.

A doua consecință privește diversificarea. Un portofoliu cu ponderi egale din $N$ active stabile i.i.d. este $\frac{1}{N}(X_1 + \dots + X_N)$, cu scala $N^{1/\alpha}\gamma/N = N^{1/\alpha - 1}\gamma$. Pentru $N = 10$ active, $\alpha = 2$ dă $0{,}32\,\gamma$, $\alpha = 1{,}5$ dă $0{,}46\,\gamma$, iar $\alpha = 1$ dă $1{,}00\,\gamma$.

Cu $\alpha = 1$, diversificarea nu reduce deloc scala: portofoliul este la fel de dispersat ca un singur activ. @Fama1965 a discutat această consecință a ipotezei stabile pentru alegerea portofoliului. Pentru $1 < \alpha < 2$, diversificarea ajută în continuare, dar mai puțin decît sub distribuția Normală.

## Recapitulare: stabilitatea {h=86ded4bc}

O lege stabilă este o lege ale cărei sume de copii independente își păstrează forma, fiind doar rescalate și deplasate. Scala unei sume ponderate urmează @eq:scale, iar suma a $n$ copii are factorul $n^{1/\alpha}$, cu $0 < \alpha \le 2$. Distribuția Normală este cazul $\alpha = 2$, iar distribuția Cauchy cazul $\alpha = 1$.

Cu cît $\alpha$ este mai mic, cu atît riscul crește mai repede cu orizontul și diversificarea ajută mai puțin. Ambele efecte contează pentru randamente numai dacă un $\alpha < 2$ le descrie cu adevărat, lucru pe care ultimele secțiuni ale capitolului îl testează.

## @end

Stabilitatea la adunare impune o regulă cu un singur parametru pentru scale: $c^\alpha = a^\alpha + b^\alpha$, respectiv $n^{1/\alpha}$ pentru $n$ copii. Simularea a confirmat că suma a 10 extrageri cu $\alpha = 1{,}7$, împărțită la $10^{1/1{,}7}$, are legea unei singure extrageri, în timp ce factorul Normal $\sqrt{10}$ îi lasă cozi prea largi. [Secțiunea următoare](#sec-3) arată de ce aceste legi nu sînt doar comode, ci inevitabile ca limite ale sumelor.

# Teorema limită centrală generalizată

Teorema limită centrală clasică explică de ce distribuția Normală apare oriunde se adună multe efecte mici și independente. Secțiunea enunță generalizarea ei, care acoperă și variabilele cu varianță infinită, și arată că o coadă a distribuției unui singur randament determină legea sumelor lungi.

## De la CLT clasică la cea generalizată {h=5fe71192}

Teorema limită centrală clasică (CLT, Capitolul 2) privește variabile i.i.d. $X_i$ cu media $\mu$ și varianța finită $\sigma^2$. Ea afirmă că

$$ \frac{X_1 + \dots + X_n - n\mu}{\sigma\sqrt{n}} \to N(0, 1) $$

în distribuție. Dacă varianța este infinită, enunțul nu mai are conținut: $\sigma$ nu există, iar normalizarea cu $\sqrt{n}$ are o rată greșită.

Teorema limită centrală generalizată [@GnedenkoKolmogorov1954] răspunde la întrebare fără a presupune varianța finită. Dacă, pentru niște constante $a_n > 0$ și $b_n$ care rescalează și recentrează suma, suma normalizată $(X_1 + \dots + X_n - b_n)/a_n$ converge către o lege nedegenerată, atunci acea lege este stabilă. Pentru CLT clasică, constantele sînt $a_n = \sigma\sqrt{n}$ și $b_n = n\mu$.

Legile stabile sînt așadar singurele limite posibile ale sumelor normalizate de variabile i.i.d., iar distribuția Normală este cazul particular $\alpha = 2$. Acesta este motivul de fond pentru studiul lor: orice model în care randamentele sînt sume ale multor șocuri independente conduce, la limită, la o lege stabilă.

## Domenii de atracție {h=adb8d4b6}

Domeniul de atracție (*domain of attraction*) al unei legi stabile este mulțimea tuturor legilor ale căror sume normalizate converg către ea. Dacă varianța este finită, legea aparține domeniului de atracție al distribuției Normale, iar constanta de normalizare crește ca $\sqrt{n}$. Dacă cozile urmează o lege de tip putere, $P(|X| > x) \approx C x^{-\alpha}$, cu o constantă $C > 0$ și $0 < \alpha < 2$, legea aparține domeniului de atracție al legii stabile cu același $\alpha$, iar $a_n \propto n^{1/\alpha}$, unde $\propto$ înseamnă „proporțional cu”.

Consecința este că o coadă a distribuției unui randament determină legea sumelor cu mulți termeni. Centrul distribuției nu contează la limită; contează doar viteza cu care valorile mari devin rare. O lege cu o coadă de tip putere de indice sub 2 conduce la o limită stabilă diferită de distribuția Normală, oricare ar fi forma ei în jurul lui zero.

Paul Lévy a caracterizat legile stabile în anii 1920, iar teoria completă a legilor limită pentru sume se găsește în @GnedenkoKolmogorov1954.

## GCLT într-o simulare {h=c91b7178}

Graficul ilustrează GCLT cu o lege care nu este stabilă: distribuția Student-t cu $\nu = 1{,}5$ grade de libertate. Ea are varianță infinită și cozi proporționale cu $x^{-1{,}5}$, deci aparține domeniului de atracție al legii stabile cu $\alpha = 1{,}5$. Sumele a $n$ extrageri sînt împărțite la $n^{1/1{,}5}$, iar histogramele lor sînt desenate pe scară logaritmică, pentru $n = 1$, 10 și 100.

Pe măsură ce $n$ crește, histogramele se apropie de limita stabilă $S(1{,}5; 0; 1{,}53; 0)$, linia continuă închisă la culoare. Converg și probabilitățile din cozi: probabilitatea ca suma normalizată să depășească 10 în valoare absolută este 2,73% pentru $n = 100$ și 2,64% sub legea limită. Histograma pentru $n = 1$ are un vîrf vizibil mai înalt decît limita, dar cozile ei au deja panta limitei.

Curba roșie punctată este o distribuție Normală cu același interval intercuartilic (*interquartile range*) ca limita. Probabilitatea ei pentru o valoare absolută de peste 10 este 0,0005%: pe scara logaritmică, ea coboară sub histograme cu mult înainte de capătul cozilor. O aproximare Normală, potrivită pe centru, nu surprinde deloc cozile.

## Implicațiile GCLT pentru randamente {h=5a33cd24}

GCLT împarte modelele posibile pentru randamentele zilnice în două lumi. Dacă tail index-ul randamentelor zilnice este sub 2, varianța este infinită, iar randamentele lunare sînt stabile cu același $\alpha$ ca randamentele zilnice. Dacă tail index-ul este peste 2, varianța este finită, iar randamentele lunare se apropie de distribuția Normală.

Cele două lumi dau predicții diferite și testabile despre randamentele pe orizonturi mai lungi: în prima, $\alpha$ estimat ar trebui să rămînă același pe măsură ce orizontul crește; în a doua, ar trebui să urce către 2. Întrebare pentru sală: căreia dintre cele două lumi îi aparțin randamentele S&P 500?

Răspunsul cere date. [Secțiunea de critică](#sec-9) estimează $\alpha$ pe randamentele zilnice, săptămînale și lunare ale celor patru serii din capitol și compară cele două predicții.

## Recapitulare: teorema limită centrală generalizată {h=0ae84a75}

CLT clasică cere varianță finită, folosește normalizarea $\sqrt{n}$ și dă o limită Normală. GCLT renunță la ipoteza despre varianță și arată că legile stabile sînt singurele limite posibile ale sumelor normalizate.

Legile cu cozi de tip putere de indice $\alpha < 2$ cer normalizarea $n^{1/\alpha}$, iar sumele lor converg către legea stabilă cu același $\alpha$. Coada distribuției unui singur randament determină deci legea sumelor lungi.

## @end

GCLT face din legile stabile limitele naturale ale sumelor de șocuri i.i.d., cu distribuția Normală ca singurul caz cu varianță finită. Limita care se aplică depinde doar de coada unui randament: un tail index sub 2 conduce la o limită stabilă cu același $\alpha$, un tail index peste 2 la distribuția Normală. Cele două cazuri dau predicții testabile despre evoluția lui $\hat\alpha$ cu orizontul, pe care [secțiunea de critică](#sec-9) le verifică. Pentru a lucra cu legile stabile este nevoie mai întîi de un mod de a le scrie, subiectul [secțiunii următoare](#sec-4).

# Parametrii și funcția caracteristică

Cu excepția a trei cazuri, legile stabile nu au densitate în formă închisă. Ele se definesc în schimb prin funcția caracteristică, care are o formă simplă și face vizibilă proprietatea de stabilitate. Secțiunea introduce cei patru parametri, cele două parametrizări folosite în practică și capcanele pe care acestea le creează în software.

## Funcția caracteristică {h=b5b1e1bb}

Funcția caracteristică (*characteristic function*) a unei variabile aleatoare $X$ este $\varphi_X(t) = E[e^{itX}] = E[\cos(tX)] + i\,E[\sin(tX)]$, pentru $t$ real, unde $i = \sqrt{-1}$ este unitatea imaginară, iar $t$ este un argument real, nu timpul. Ea există întotdeauna, pentru că $|e^{itX}| = 1$, chiar și cînd $X$ nu are niciun moment, și determină distribuția în mod unic. Ambele proprietăți o fac instrumentul natural pentru legile cu varianță infinită.

Capitolul folosește două proprietăți. Pentru $X$ și $Y$ independente, $\varphi_{X+Y}(t) = \varphi_X(t)\,\varphi_Y(t)$: sumele de variabile independente devin produse de funcții caracteristice. Pentru constantele $a$ și $b$, $\varphi_{aX+b}(t) = e^{ibt}\varphi_X(at)$, deci rescalarea și deplasarea au un efect simplu. Pentru distribuția Normală $N(\mu, \sigma^2)$, $\varphi(t) = \exp(i\mu t - \sigma^2 t^2/2)$.

Motivul pentru care se lucrează cu $\varphi$ este practic: cu excepția a trei cazuri, legile stabile nu au densitate în formă închisă. Ele sînt definite prin funcția caracteristică, iar densitățile, funcțiile de repartiție și cuantilele se calculează numeric din ea.

## Funcția caracteristică stabilă (S1) {h=3a4a3391}

În parametrizarea S1 [@SamorodnitskyTaqqu1994; @Nolan2020], $X \sim S(\alpha; \beta; \gamma; \delta \mid 1)$ dacă, pentru $\alpha \ne 1$,

$$ \varphi(t) = \exp\Big\{-\gamma^\alpha |t|^\alpha \big[1 - i\beta\,\mathrm{sign}(t)\tan\tfrac{\pi\alpha}{2}\big] + i\delta t\Big\}, $$ {#eq:cf-s1}

iar, pentru $\alpha = 1$, $\varphi(t) = \exp\{-\gamma |t| [1 + i\beta\,\frac{2}{\pi}\mathrm{sign}(t)\ln|t|] + i\delta t\}$. Aici $\mathrm{sign}(t)$ este $+1$ pentru $t > 0$ și $-1$ pentru $t < 0$; termenul cu $\beta$ este partea imaginară care face legea asimetrică.

Cei patru parametri au roluri distincte. Indicele de stabilitate $\alpha \in (0, 2]$ controlează cozile: cu cît $\alpha$ este mai mic, cu atît cozile sînt mai groase. Asimetria (*skewness*) $\beta \in [-1, 1]$ controlează asimetria cozilor: $\beta > 0$ dă o coadă dreaptă mai groasă, iar $\beta < 0$ o coadă stîngă mai groasă. Scala (*scale*) $\gamma > 0$ joacă rolul abaterii standard $\sigma$, iar poziția (*location*) $\delta$ este o deplasare reală.

Stabilitatea se citește direct din @eq:cf-s1. Suma a $n$ copii independente are funcția caracteristică $\varphi(t)^n$, care are aceeași formă, cu $\gamma^\alpha$ înlocuit de $n\gamma^\alpha$ (și $\delta$ de $n\delta$). Scala sumei este deci $n^{1/\alpha}\gamma$, ca în [secțiunea despre stabilitate](#sec-2).

## Efectul lui α {h=74e70fcd}

Graficul arată densități stabile simetrice $S(\alpha; 0; 1; 0)$ pentru $\alpha$ între 0,7 și 2, pe scară liniară (stînga) și pe scară logaritmică (dreapta). Pe scara liniară, un $\alpha$ mai mic dă un vîrf mai înalt și mai îngust; masa luată din „umeri” se mută parțial spre centru și parțial spre cozi.

Scara logaritmică arată cozile, pe care scara liniară le ascunde. Densitatea Normală ($\alpha = 2$, aici cu varianța 2) scade ca $e^{-x^2/4}$, o parabolă cu ramurile în jos pe scara logaritmică. Toate celelalte curbe scad ca o putere a lui $x$, ceea ce pe această scară arată ca o coborîre lentă, aproape plată. La $|x| = 8$, densitatea Normală este deja cu cîteva ordine de mărime sub cele stabile.

## Efectul lui β și γ {h=7097bd43}

Panoul din stînga variază asimetria $\beta$ pentru $\alpha = 1{,}3$ și $\gamma = 1$. Parametrul mută masa între cele două cozi: $\beta = 1$ (verde) are coada dreaptă mai groasă, iar $\beta = -1$ (roșu) coada stîngă mai groasă, în timp ce centrul densității se schimbă puțin. Pentru randamente, un $\beta$ negativ este descrierea stabilă a riscului de crah: o coadă stîngă mai groasă decît cea dreaptă.

Panoul din dreapta variază scala $\gamma$ pentru $\alpha = 1{,}7$ și $\beta = 0$. Scala întinde densitatea: dublarea lui $\gamma$ dublează fiecare cuantilă în jurul lui $\delta$ și înjumătățește înălțimea vîrfului. Din acest punct de vedere, $\gamma$ se comportă ca o abatere standard, dar rămîne bine definită și cînd varianța este infinită.

## Două parametrizări: S0 și S1 {h=1abc0ea0}

Parametrizarea S0, recomandată de Nolan pentru analiza datelor, folosește, pentru $\alpha \ne 1$, funcția caracteristică

$$ \varphi(t) = \exp\Big\{-\gamma^\alpha |t|^\alpha \big[1 + i\beta\,\mathrm{sign}(t)\tan\tfrac{\pi\alpha}{2}\big(|\gamma t|^{1-\alpha} - 1\big)\big] + i\delta_0 t\Big\}. $$

Între cele două parametrizări diferă doar poziția. Parametrii $\alpha$, $\beta$ și $\gamma$ sînt aceiași în ambele, iar pozițiile $\delta_0$ (în S0) și $\delta_1$ (în S1) sînt legate prin

$$ \delta_0 = \delta_1 + \beta\gamma\tan\tfrac{\pi\alpha}{2}, \qquad \alpha \ne 1. $$ {#eq:s0s1}

Pentru $\beta = 0$, cele două coincid.

Fiecare parametrizare are rolul ei. S1 dă o algebră mai simplă pentru sume și este folosită în majoritatea cărților de teorie [@SamorodnitskyTaqqu1994]. În S0, densitatea se schimbă continuu cu $\alpha$ și $\beta$, ceea ce o face mai potrivită pentru estimare și pentru grafice [@Nolan2020]. [Slide-ul următor](#s25) arată de ce continuitatea este proprietatea decisivă.

## S1 și S0 în apropiere de α = 1 {h=c458e9d5}

Graficul desenează patru legi cu $\beta = 0{,}8$, $\gamma = 1$ și poziția 0, pentru $\alpha = 0{,}8$, 0,95, 1,05 și 1,2, întîi în S1 (stînga), apoi în S0 (dreapta). În S1, densitatea fuge spre $+\infty$ cînd $\alpha \uparrow 1$ și revine dinspre $-\infty$ cînd $\alpha \downarrow 1$. Cauza este termenul $\tan\frac{\pi\alpha}{2}$ din @eq:s0s1, care tinde la infinit cînd $\alpha$ se apropie de 1: o mică schimbare a lui $\alpha$ mută întreaga lege departe pe axă.

În S0, aceleași patru legi sînt deplasate astfel încît modul rămîne lîngă 0 și se schimbă lin cu $\alpha$. Pentru estimare, aceasta este proprietatea decisivă: un algoritm de optimizare care modifică puțin pe $\alpha$ vede o densitate puțin diferită, nu o lege care a sărit de pe o parte pe alta a axei. De aceea S0 se folosește pentru estimare, iar S1 pentru algebră.

## Parametrizările legilor stabile în software {h=edae18db}

Parametrizarea este o sursă frecventă de erori în software. Funcția scipy.stats.levy_stable folosește implicit S1; trece la S0 cu levy_stable.parameterization = 'S0', iar argumentele ei sînt levy_stable.pdf(x, alpha, beta, loc=delta, scale=gamma). Lucrările și programele raportează $\delta$ în S0, în S1 sau în alte parametrizări, așa că parametrizarea trebuie verificată înainte de a compara rezultatele.

A doua capcană privește scala. Legea stabilă cu $\alpha = 2$, $S(2; 0; \gamma; \delta)$, este $N(\delta, 2\gamma^2)$, nu $N(\delta, \gamma^2)$. Pentru distribuția Normală, așadar, $\gamma = \sigma/\sqrt{2}$, iar un $\hat\gamma$ stabil nu se poate compara direct cu o abatere standard de selecție.

În acest curs, estimările sînt raportate în S0, iar $\delta_1$ este raportat separat, astfel încît rezultatele să poată fi comparate cu programele care folosesc S1.

## Trei legi cu densitate în formă închisă {h=730b106d}

Trei legi stabile au densitate în formă închisă. Distribuția Normală este cazul $\alpha = 2$, în care $\beta$ nu are niciun rol: $\varphi(t) = \exp(-\gamma^2 t^2 + i\delta t)$, adică $N(\delta, 2\gamma^2)$. Distribuția Cauchy este cazul $\alpha = 1$, $\beta = 0$, cu densitatea $f(x) = \gamma/(\pi(\gamma^2 + (x - \delta)^2))$ și $\varphi(t) = \exp(-\gamma|t| + i\delta t)$.

Distribuția Lévy este cazul $\alpha = 1/2$, $\beta = 1$ (în S1), complet asimetrică la dreapta. Densitatea ei este

$$ f(x) = \sqrt{\frac{\gamma}{2\pi}}\,(x - \delta)^{-3/2} \exp\Big(-\frac{\gamma}{2(x - \delta)}\Big), \qquad x > \delta. $$

Ea are o interpretare probabilistă: momentul în care o mișcare browniană atinge prima dată un nivel dat are o lege Lévy.

Toate celelalte densități stabile se calculează numeric din $\varphi$, fie prin integrare directă [@Nolan1997], fie cu transformata Fourier rapidă [@MittnikDoganogluChenyao1999].

## Distribuțiile Normală, Cauchy și Lévy {h=f5c475ea}

Graficul compară cele trei legi cu formă închisă: distribuția Normală cu varianța 1 ($\alpha = 2$, $\gamma = 1/\sqrt{2}$), distribuția Cauchy cu $\gamma = 1$ și distribuția Lévy cu $\gamma = 1$ în S1. Densitatea Cauchy are un vîrf mai jos decît densitatea Normală și cozi mult mai groase: dincolo de $|x| \approx 2$, curba ei rămîne clar deasupra celei Normale.

Densitatea Lévy ia doar valori pozitive. Ea urcă abrupt imediat după zero și apoi scade lent, cu o coadă dreaptă lungă. Pentru că indicele ei este $\alpha = 1/2$, sub 1, media ei este infinită, după cum explică [secțiunea despre cozi](#sec-5).

## Exemplu lucrat: S1 și S0 în t = 1 {h=146a64e5}

Exemplul lucrat compară S1 și S0 într-un singur punct, $t = 1$, pentru $X$ cu $\alpha = 1{,}5$, $\beta = 0{,}3$, $\gamma = 1$ și $\delta = 0$. O valoare utilă este $\tan\frac{3\pi}{4} = -1$. În S1, @eq:cf-s1 dă $\ln\varphi(1) = -1\cdot[1 - i \cdot 0{,}3 \cdot (-1)] = -1 - 0{,}3i$, deci $\varphi(1) = e^{-1}(\cos 0{,}3 - i \sin 0{,}3) = 0{,}351 - 0{,}109i$, cu modulul $|\varphi(1)| = e^{-1} = 0{,}368$.

În S0, factorul $|t|^{1-\alpha} - 1$ este zero în $t = 1$, deci partea imaginară dispare și $\varphi(1) = e^{-1} = 0{,}368$, un număr real. Cele două funcții caracteristice au același modul. Așa acționează o deplasare pură: din $\varphi_{X+b}(t) = e^{ibt}\varphi_X(t)$, o deplasare înmulțește $\varphi$ cu un număr complex de modul 1 și lasă modulul neschimbat.

Mărimea deplasării rezultă din @eq:s0s1: $\beta\gamma\tan\frac{\pi\alpha}{2} = 0{,}3 \cdot 1 \cdot (-1) = -0{,}3$. Legea S1 cu $\delta_1 = 0$ este deci legea S0 cu $\delta_0 = -0{,}3$.

## Recapitulare: parametrii și funcția caracteristică {h=96863bf2}

Legile stabile se definesc prin funcția lor caracteristică @eq:cf-s1, iar densitățile lor se calculează numeric. Cei patru parametri au roluri separate: $\alpha$ cozile, $\beta$ asimetria, $\gamma$ scala și $\delta$ poziția.

Parametrizările S0 și S1 diferă doar prin $\delta$, conform @eq:s0s1, iar scipy folosește implicit S1. Formele închise sînt distribuția Normală ($\alpha = 2$, varianța $2\gamma^2$), distribuția Cauchy și distribuția Lévy.

## @end

Funcția caracteristică transformă proprietatea de stabilitate într-un enunț despre un singur parametru, $\gamma^\alpha$, care crește liniar cu numărul de termeni. Cei patru parametri separă cozile, asimetria, scala și poziția, iar alegerea între S0 și S1 afectează doar poziția, dar într-un mod care devine instabil în S1 în apropiere de $\alpha = 1$. [Secțiunea următoare](#sec-5) studiază ce implică $\alpha$ pentru cozi și pentru existența momentelor.

# Cozi și momente

Indicele $\alpha$ fixează viteza cu care scad cozile unei legi stabile și, prin cozi, ce momente există. Secțiunea deduce cozile de tip putere, condiția $p < \alpha$ pentru momente și consecințele lor pentru măsurarea riscului.

## Cozi de tip putere {h=626bc85f}

Pentru $\alpha < 2$, cozile unei legi stabile scad ca o putere a lui $x$ [@Nolan2020]. Cînd $x \to \infty$,

$$ P(X > x) \sim \gamma^\alpha c_\alpha (1 + \beta)\, x^{-\alpha}, \qquad P(X < -x) \sim \gamma^\alpha c_\alpha (1 - \beta)\, x^{-\alpha}, $$ {#eq:tail}

cu $c_\alpha = \Gamma(\alpha)\sin(\pi\alpha/2)/\pi$, unde $\Gamma$ este funcția gamma, iar $f \sim g$ înseamnă $f/g \to 1$. Asimetria $\beta$ intră doar prin constante: ea schimbă cîtă masă poartă fiecare coadă, nu cît de repede scade aceasta.

Pe axe log-log, o coadă de tip putere este o dreaptă cu panta $-\alpha$. Dublarea pragului împarte probabilitatea din coadă la $2^\alpha$: pentru $\alpha = 1{,}7$, probabilitatea se înmulțește cu 0,31. Coada distribuției Normale se comportă altfel: scade ca $e^{-x^2/(4\gamma^2)}$, iar dublarea pragului o face practic nulă.

Cînd $\beta = \pm 1$ și $\alpha < 1$, una dintre cozi lipsește cu totul: suportul legii este o semidreaptă. Distribuția Lévy din [secțiunea anterioară](#sec-4) este un exemplu.

## Cozile pe axe log-log {h=9c0dff95}

Graficul reprezintă probabilitatea din coadă $P(X > x)$ pentru legi simetrice $S(\alpha; 0; 1; 0)$ pe axe log-log, pentru $\alpha = 2$; 1,9; 1,7 și 1,5. Liniile continue sînt probabilitățile exacte, iar liniile punctate legile de tip putere $c_\alpha x^{-\alpha}$ din @eq:tail. Pentru $\alpha = 1{,}7$ și 1,5, liniile continue își ating asimptotele punctate deja în jurul lui $x = 5$; pentru $\alpha = 1{,}9$, abia mai departe. Curba Normală se îndoaie în jos și iese din grafic înainte de $x = 10$.

Valorile arată mărimea diferenței. Probabilitatea de a depăși 10 este 0,279% pentru $\alpha = 1{,}7$, 0,667% pentru $\alpha = 1{,}5$ și circa $8\times 10^{-11}\%$ pentru distribuția Normală cu același $\gamma$. O variație de zece unități de scală este un eveniment rar, dar regulat, sub o lege stabilă cu $\alpha = 1{,}7$ și practic imposibil sub distribuția Normală.

## Existența momentelor {h=8d5e2be1}

Pentru $0 < \alpha < 2$, momentul absolut de ordin $p > 0$, $E|X|^p$, este finit dacă și numai dacă

$$ p < \alpha. $$ {#eq:moments}

Cazul $p = 1$ corespunde mediei, iar $p = 2$ varianței. Motivul este coada: densitatea unei legi stabile scade ca $|x|^{-\alpha-1}$, iar $|x|^p$ înmulțit cu o astfel de densitate este integrabil la infinit doar dacă $p < \alpha$.

Consecințele rezultă din compararea lui $\alpha$ cu 1 și cu 2. Pentru $\alpha < 2$, varianța este infinită, iar asimetria și boltirea (coeficienții de *skewness* și *kurtosis*) nu sînt definite. Pentru $1 < \alpha < 2$, media există și, în S1, este egală cu $\delta_1$. Pentru $\alpha \le 1$, nici media nu există; legile Cauchy și Lévy sînt exemple. Pentru $\alpha = 2$, cazul Normal, toate momentele există.

Ce credeți? Mandelbrot a găsit $\alpha \approx 1{,}7$ pentru bumbac. Există media variației zilnice? Da, pentru că $1 < 1{,}7$. Varianța însă nu există, pentru că $2 > 1{,}7$.

## Varianța de selecție nu se stabilizează {h=24927ca7}

Graficul arată cum arată o varianță infinită într-un eșantion. El reprezintă varianța de selecție în funcție de mărimea eșantionului, ambele pe scară logaritmică, pentru un eșantion Normal (albastru), din legea $S(2; 0; 1; 0)$ cu varianța adevărată 2, și pentru trei eșantioane independente dintr-o lege stabilă cu $\alpha = 1{,}7$.

Varianța de selecție a eșantionului Normal converge la valoarea adevărată 2 și rămîne acolo. Traiectoriile stabile se comportă altfel: fiecare extragere mare produce un salt al varianței, după care aceasta scade lent pînă la următoarea extragere mare. După 20 000 de extrageri, cele trei traiectorii stabile ajung la 10,4, 6,6 și 20,2, fără niciun semn de convergență; un eșantion mai lung ar adăuga doar noi salturi.

Este fenomenul observat de Mandelbrot în prețurile bumbacului. Varianța de selecție a unui eșantion stabil este un număr care se poate calcula întotdeauna, dar care nu estimează nicio mărime a populației.

## Probabilitățile din cozi: distribuția stabilă și distribuția Normală {h=4ed2cb96}

Tabelul compară probabilitățile $P(|X| > k)$ sub o lege stabilă cu $\alpha = 1{,}7$ și sub distribuția Normală, ambele cu $\beta = 0$, $\gamma = 1$ și $\delta = 0$; legea Normală este deci $N(0, 2)$. Pentru $k = 2$, probabilitățile sînt apropiate: 15,73% pentru legea Normală și 18,59% pentru cea stabilă, un raport de 1,2. Pentru $k = 3$, ele sînt 3,39% și 7,25%, un raport de 2,1.

Mai departe în cozi, diferența se adîncește. Pentru $k = 5$, probabilitatea Normală este 0,041%, iar cea stabilă 2,13%, un raport de 52,4. Pentru $k = 10$, probabilitatea Normală este $1{,}5\times 10^{-10}\%$, iar cea stabilă 0,56%, un raport de $3{,}6\times 10^{9}$.

În jurul centrului, cele două legi sînt aproape interschimbabile; departe în cozi, ele diferă prin ordine de mărime. Un model de risc se evaluează în cozi, unde alegerea legii contează cel mai mult, iar o potrivire bună în centru spune puțin despre ele.

## Varianța infinită și măsurile de risc {h=51d0d8f1}

Mai multe instrumente din capitolele anterioare au nevoie de varianță finită: volatilitatea, raportul Sharpe și eroarea lui standard (Capitolul 1), portofoliile medie–varianță și intervalele de încredere obținute din CLT. Cu $\alpha < 2$, aceste mărimi se pot calcula pe orice eșantion, dar nu estimează nimic, cum a arătat [graficul anterior](#s34) pentru varianță.

Alte instrumente rămîn valabile. Cuantilele există întotdeauna, deci valoarea la risc (VaR, *value at risk*) este bine definită: $\mathrm{VaR}_{1\%} = -q_{1\%}$, negativul cuantilei de 1% a randamentelor, este pierderea depășită cu probabilitatea 1%. Pierderea așteptată în coadă (ES, *expected shortfall*) la 2,5%, $\mathrm{ES}_{2{,}5\%}$, adică pierderea medie dincolo de VaR, cere o medie finită și există deci doar dacă $\alpha > 1$. Scala $\gamma$ și intervalul intercuartilic înlocuiesc abaterea standard ca măsuri ale dispersiei.

VaR și ES sînt tratate pe larg în [Capitolul 10](ch10_ro.html).

## Recapitulare: cozi și momente {h=30557cd3}

Cozile stabile scad ca $P(|X| > x) \propto x^{-\alpha}$, o dreaptă pe axe log-log. Momentul absolut de ordin $p$ este finit dacă și numai dacă $p < \alpha$, deci $\alpha < 2$ înseamnă varianță infinită.

Varianța de selecție a unui eșantion stabil sare la fiecare extragere mare și nu se stabilizează. Cuantilele, VaR și scala $\gamma$ își păstrează sensul.

## @end

Indicele $\alpha$ este în același timp panta cozilor pe axe log-log și pragul pentru existența momentelor. Sub o lege stabilă cu $\alpha = 1{,}7$, o variație absolută de peste zece unități de scală are probabilitatea 0,56%, față de $1{,}5\times 10^{-10}\%$ sub distribuția Normală, iar varianța de selecție nu se stabilizează niciodată. Măsurile bazate pe cuantile, precum VaR, rămîn valabile. Aceleași proprietăți dau testele din [secțiunea de critică](#sec-9): un model stabil prevede atît o varianță de selecție care nu se stabilizează, cît și un anumit număr de pierderi extreme. Înainte de estimarea modelului, [secțiunea următoare](#sec-6) arată cum se simulează.

# Simulare

Legile stabile nu au densitate în formă închisă, dar se simulează ușor. Secțiunea prezintă metoda Chambers–Mallows–Stuck (CMS), care transformă o extragere uniformă și una exponențială într-o extragere stabilă, întîi în cazul simetric, apoi în cel general.

## Metoda Chambers–Mallows–Stuck (cazul simetric) {h=d19000d8}

Metoda CMS [@ChambersMallowsStuck1976] produce o extragere din $S(\alpha; 0; 1; 0)$ pornind de la două variabile independente: un unghi uniform $V \sim U(-\pi/2, \pi/2)$ și o variabilă exponențială $W \sim \mathrm{Exp}(1)$ cu media 1. Extragerea stabilă este

$$ X = \frac{\sin(\alpha V)}{(\cos V)^{1/\alpha}} \left(\frac{\cos((1 - \alpha)V)}{W}\right)^{(1 - \alpha)/\alpha}, $$ {#eq:cms}

iar $\gamma X + \delta \sim S(\alpha; 0; \gamma; \delta)$ dă orice scală și orice poziție.

Formula se poate verifica pe cele două cazuri particulare cu lege cunoscută. Pentru $\alpha = 1$, exponentul $(1 - \alpha)/\alpha$ este zero și $X = \sin V/\cos V = \tan V$; tangenta unui unghi uniform este o variabilă Cauchy standard. Pentru $\alpha = 2$, $\cos((1-\alpha)V) = \cos V$ și $X = \sin(2V)\,(\cos V)^{-1/2}(\cos V/W)^{-1/2} = 2\sin V\sqrt{W}$, o variabilă $N(0, 2)$, în acord cu convenția de scală $S(2; 0; 1; 0) = N(0, 2)$.

Metoda nu are nevoie de densitate, de inversarea unei funcții de repartiție sau de un pas de respingere: fiecare pereche $(V, W)$ dă exact o extragere.

## Cazul general (β 0) {h=cde4ff31}

Pentru $\beta \ne 0$ și $\alpha \ne 1$, formula generală în parametrizarea S1 [@Weron1996] folosește două constante auxiliare care depind doar de $\alpha$ și $\beta$:

$$ B = \frac{\arctan(\beta\tan\frac{\pi\alpha}{2})}{\alpha}, \qquad S = \Big(1 + \beta^2\tan^2\tfrac{\pi\alpha}{2}\Big)^{1/(2\alpha)}. $$

Extragerea este atunci $X = S\,\sin(\alpha(V + B))/(\cos V)^{1/\alpha} \cdot (\cos(V - \alpha(V + B))/W)^{(1 - \alpha)/\alpha}$, care urmează legea $S(\alpha; \beta; 1; 0 \mid 1)$. Pentru $\beta = 0$, $B = 0$ și $S = 1$, iar formula se reduce la @eq:cms.

Din extragerea standard se obțin orice parametri prin rescalare și deplasare: $\gamma X + \delta_1$ în S1, respectiv $\gamma X + \delta_0 - \beta\gamma\tan\frac{\pi\alpha}{2}$ în S0, conform @eq:s0s1. @Weron1996 a corectat o formulă din lucrarea originală pentru $\beta \ne 0$; scipy.stats.levy_stable.rvs folosește aceeași metodă.

## Exemplu lucrat: o extragere pas cu pas {h=6e58ea6d}

Exemplul lucrat produce o extragere cu $\alpha = 1{,}5$ și $\beta = 0$, dintr-o extragere uniformă $U = 0{,}8$ și o extragere exponențială $W = 1{,}2$. Primul pas transformă $U$, uniformă pe $(0, 1)$, într-un unghi uniform pe $(-\pi/2, \pi/2)$: $V = \pi(U - 1/2) = 0{,}942$.

Al doilea pas evaluează factorii din @eq:cms: $\sin(\alpha V) = 0{,}988$, $(\cos V)^{1/\alpha} = 0{,}702$ și $\big(\cos((1 - \alpha)V)/W\big)^{(1 - \alpha)/\alpha} = 1{,}104$. Extragerea este $X = 0{,}988/0{,}702 \times 1{,}104 = 1{,}554$.

Un eșantion de orice mărime se obține repetînd calculul cu noi perechi $(U, W)$. Pentru că fiecare extragere este o funcție explicită de două variabile standard, simularea a milioane de extrageri stabile nu durează mai mult decît simularea unor extrageri Normale.

## Extrageri CMS față de densitatea exactă {h=05487f9c}

Graficul compară histograma a 100 000 de extrageri CMS din $S(1{,}5; 0{,}5; 1; 0 \mid 1)$ cu densitatea exactă, calculată numeric din $\varphi$. Histograma urmează îndeaproape densitatea, inclusiv asimetria ei: cu $\beta = 0{,}5$, coada dreaptă este mai groasă, iar în S1 modul se află sub zero.

Cuantilele confirmă acordul în cozi. Cuantilele de selecție de 1% și 99% sînt −5,47 și 9,52; cele exacte, calculate prin integrare numerică cu scipy.stats.levy_stable, sînt −5,39 și 9,79 (grila FFT a graficului dă −5,40 și 9,76). Testul Kolmogorov–Smirnov (KS) cu două eșantioane, aplicat extragerilor CMS și unor extrageri din levy_stable.rvs, dă statistica 0,0044 și un p-value de 0,30, deci testul nu găsește nicio diferență între cele două eșantioane.

## Recapitulare: simularea {h=6a5f07c1}

Metoda CMS transformă un unghi uniform $V$ și o variabilă exponențială $W$ într-o extragere stabilă. Cazurile particulare verifică formula: $\tan V$ dă distribuția Cauchy, iar $2\sin V\sqrt{W}$ distribuția Normală $N(0, 2)$.

Cazul asimetric folosește formula din @Weron1996, iar o extragere în S0 se obține printr-o deplasare cu $\beta\gamma\tan\frac{\pi\alpha}{2}$.

## @end

Simularea este partea ușoară a lucrului cu legile stabile: fiecare extragere este o funcție explicită de două variabile standard, iar 100 000 de extrageri reproduc densitatea exactă, inclusiv cuantilele ei. Simularea susține și restul capitolului: studiul Monte Carlo al estimatorilor din [secțiunea următoare](#sec-7) și traiectoriile simulate ale varianței din [secțiunea de critică](#sec-9) se bazează pe ea.

# Estimare

Estimarea unei legi stabile este mai grea decît simularea ei, pentru că verosimilitatea cere densitatea, care nu are formă închisă. Secțiunea prezintă cele trei familii de estimatori și compară cei doi estimatori cei mai folosiți: metoda cuantilelor a lui McCulloch și verosimilitatea maximă.

## Trei familii de estimatori {h=58b7afc1}

Metodele cuantilelor potrivesc cuantilele (*quantiles*) de selecție cu cele ale legii stabile. @FamaRoll1971 le-au dezvoltat pentru legi simetrice, iar @McCulloch1986 le-a extins la toți cei patru parametri. Ele cer doar datele ordonate și tabelele publicate.

Metodele funcției caracteristice folosesc funcția caracteristică empirică $\hat\varphi(t) = \frac1n\sum_{j=1}^n e^{itr_j}$ a randamentelor $r_1, \dots, r_n$. Pentru o lege stabilă, $|\varphi(t)|^2 = e^{-2\gamma^\alpha|t|^\alpha}$, deci $\ln(-\ln|\varphi(t)|^2) = \ln(2\gamma^\alpha) + \alpha\ln|t|$ este liniară în $\ln|t|$. Regresia versiunii empirice pe $\ln|t|$ dă o pantă care estimează $\alpha$ [@Koutrouvelis1980].

Verosimilitatea maximă (ML) este cea mai precisă metodă, odată ce densitatea poate fi calculată [@Nolan2001; @MittnikEtAl1999]. O implementare SAS (Statistical Analysis System) a acestor estimatori este descrisă în @Pele2014.

## Metoda cuantilelor a lui McCulloch {h=d209057f}

Metoda lui McCulloch [@McCulloch1986] se bazează pe două rapoarte de cuantile de selecție care nu depind de poziție și de scală:

$$ \nu_\alpha = \frac{\hat q_{0,95} - \hat q_{0,05}}{\hat q_{0,75} - \hat q_{0,25}}, \qquad \nu_\beta = \frac{\hat q_{0,95} + \hat q_{0,05} - 2\hat q_{0,5}}{\hat q_{0,95} - \hat q_{0,05}}, $$ {#eq:nu}

unde $\hat q_p$ este cuantila de selecție de ordin $p$, valoarea sub care se află o proporție $p$ din randamente. Raportul $\nu_\alpha$ măsoară lățimea cozilor față de corpul distribuției; $\nu_\beta$ măsoară asimetria cozilor în jurul medianei.

Tabelele III și IV din lucrare transformă perechea $(\nu_\alpha, \nu_\beta)$ în estimațiile $(\hat\alpha, \hat\beta)$. Pentru distribuția Normală, $\nu_\alpha = 2{,}439$, iar un $\nu_\alpha$ mai mare înseamnă un $\alpha$ mai mic. Scala rezultă apoi din intervalul intercuartilic, $\hat\gamma = (\hat q_{0,75} - \hat q_{0,25})/\phi_3(\hat\alpha, \hat\beta)$, unde $\phi_3$ este intervalul intercuartilic al lui $S(\alpha; \beta; 1; 0)$ (Tabelul V), iar $\hat\delta_0$ vine din Tabelul VII.

Metoda este rapidă și consistentă și este valabilă pentru $0{,}6 \le \alpha \le 2$. Este mai puțin precisă decît ML, pentru că folosește doar cinci cuantile, și este un bun punct de plecare pentru ML.

## Corespondența McCulloch între ν_α și α {h=a85c9ede}

Graficul desenează Tabelul III din @McCulloch1986 pentru un eșantion simetric ($\nu_\beta = 0$): pe axa orizontală este $\nu_\alpha$, iar pe cea verticală valoarea corespunzătoare a lui $\alpha$. Curba pornește din $\nu_\alpha = 2{,}439$ pentru $\alpha = 2$ și coboară pe măsură ce $\nu_\alpha$ crește; cozile mai groase lărgesc intervalul dintre cuantilele de 5% și 95% față de intervalul intercuartilic.

Stelele plasează randamentele logaritmice zilnice ale celor patru serii din capitol, indicii din 2000, iar Bitcoin din 2014. Valorile lor $\nu_\alpha$ sînt 3,425 pentru BET, 3,338 pentru S&P 500, 3,285 pentru DAX și 3,803 pentru Bitcoin. Toate sînt mult peste 2,439, deci cozile tuturor celor patru serii sînt mult mai largi, în raport cu corpul distribuției, decît permite distribuția Normală. Bitcoin se află cel mai la dreapta, cu cel mai mic $\alpha$ implicit.

## Exemplu lucrat: McCulloch pe BET {h=25b0b5cf}

Exemplul lucrat aplică metoda pe randamentele logaritmice zilnice BET în %, din 2000 pînă în 2026, $n = 6\,670$ de observații. Primul pas calculează cele cinci cuantile de selecție: $\hat q_{0,05} = -1{,}9345$, $\hat q_{0,25} = -0{,}5025$, $\hat q_{0,5} = 0{,}0771$, $\hat q_{0,75} = 0{,}6649$ și $\hat q_{0,95} = 2{,}0641$.

Al doilea pas formează rapoartele din @eq:nu: $\nu_\alpha = 3{,}9986/1{,}1674 = 3{,}425$ și $\nu_\beta = -0{,}006$, deci eșantionul este practic simetric. Al treilea pas citește Tabelul III în coloana $\nu_\beta = 0$, unde $\nu_\alpha = 3{,}2$ corespunde lui $\alpha = 1{,}484$, iar $\nu_\alpha = 3{,}5$ lui $\alpha = 1{,}391$. Interpolarea liniară între cele două rînduri dă $\hat\alpha \approx 1{,}414$.

Al patrulea pas folosește celelalte tabele: scala este $\hat\gamma = 0{,}60$ (% pe zi), asimetria $\hat\beta = -0{,}011$, iar poziția $\hat\delta_0 = 0{,}079$. Estimarea $\hat\alpha \approx 1{,}414$ se află mult sub valoarea 2 a distribuției Normale.

## Verosimilitatea maximă {h=22915d18}

Pentru un eșantion $r_1, \dots, r_n$, log-verosimilitatea este $\ell(\theta) = \sum_{t=1}^n \ln f(r_t; \theta)$, unde $\theta = (\alpha, \beta, \gamma, \delta_0)$ este vectorul celor patru parametri, iar $f(r_t; \theta)$ densitatea stabilă în $r_t$. Estimarea ML $\hat\theta$ maximizează numeric pe $\ell$, pornind de la estimarea McCulloch. Erorile standard provin din curbura lui $\ell$ în punctul de maxim: inversa hessianei, matricea derivatelor a doua ale lui $\ell$.

Dificultatea este densitatea însăși, care trebuie calculată din $\varphi$ prin formula de inversare $f(x) = \frac{1}{2\pi}\int e^{-itx}\varphi(t)\,dt$. Ea se poate evalua pe o grilă cu transformata Fourier rapidă (FFT, *fast Fourier transform*) [@MittnikDoganogluChenyao1999] sau prin integrare numerică directă [@Nolan1997]. Funcția levy_stable.fit face acest calcul și durează cîteva minute pentru cîteva mii de randamente.

În parametrizarea S0, estimarea ML este regulată pentru $\alpha \in (0, 2)$: estimatorul este asimptotic Normal, iar erorile lui standard scad ca $1/\sqrt{n}$ [@Nolan2001].

## McCulloch și ML într-un studiu Monte Carlo {h=7e07963c}

Studiul Monte Carlo extrage 200 de eșantioane din $S(1{,}7; 0; 1; 0)$ pentru fiecare mărime de eșantion, $n = 500$ și $n = 2500$, și estimează $\alpha$ prin ambele metode. Diagramele box plot arată distribuția lui $\hat\alpha$ pe cele 200 de eșantioane; linia punctată marchează valoarea adevărată 1,7. Ambii estimatori sînt centrați pe 1,7, deci niciunul nu are o deplasare vizibilă.

Precizia diferă. Abaterea standard a lui $\hat\alpha$ este 0,1065 pentru McCulloch și 0,0611 pentru ML la $n = 500$, respectiv 0,0462 față de 0,0298 la $n = 2500$. ML este deci mai precisă pe aceleași date: abaterea standard McCulloch este de 1,74 ori cea ML la $n = 500$ și de 1,55 ori la $n = 2500$. Cu $n = 500$, metoda lui McCulloch atinge uneori limita $\alpha = 2$, vizibilă ca valori extreme în partea de sus a primei cutii.

## Recapitulare: estimarea {h=a05f8d08}

Metoda lui McCulloch are nevoie de cinci cuantile, două rapoarte și patru tabele; este rapidă și robustă. ML calculează densitatea numeric din $\varphi$ și maximizează numeric verosimilitatea; este cea mai precisă.

Orice estimare trebuie să precizeze parametrizarea, S0 sau S1, și unitățile lui $\gamma$ și $\delta$, deoarece ambele se schimbă cînd randamentele sînt exprimate în % în loc de zecimale.

## @end

Cei doi estimatori se completează: metoda cuantilelor a lui McCulloch dă o estimare rapidă și robustă din cinci cuantile, iar ML, pornită de la ea, cîștigă un factor de circa 1,7 în precizie pe eșantioanele din $S(1{,}7; 0; 1; 0)$. Pe randamentele zilnice BET, metoda cuantilelor dă deja $\hat\alpha \approx 1{,}414$, departe de valoarea Normală 2. [Secțiunea următoare](#sec-8) estimează legea stabilă prin ML pe toate cele patru serii și o confruntă cu datele și cu două legi alternative.

# Estimarea pe randamente reale

Secțiunea estimează legea stabilă prin verosimilitate maximă pe randamentele logaritmice zilnice a patru serii: BET, S&P 500 și DAX din 2000 pînă în 2026 și Bitcoin din 2014 pînă în 2026. Estimarea stabilă este comparată cu două alternative: distribuția Normală și distribuția Student-t. Comparația folosește densități, QQ plots, probabilități din cozi, numărul pierderilor mari, VaR și un criteriu informațional.

## Legi stabile estimate pe randamentele logaritmice zilnice {h=80b4b0ad}

Tabelul raportează estimările pentru cele patru serii. Randamentele sînt randamente logaritmice zilnice în %; indicii acoperă perioada 2000–2026, iar Bitcoin perioada 2014–2026, cu șapte zile de tranzacționare pe săptămînă. Coloana „McC.” dă $\hat\alpha$ prin metoda lui McCulloch; estimările ML sînt în S0, cu erorile standard (SE, *standard errors*) în paranteze; $\hat\gamma$, $\hat\delta_0$ și $\hat\delta_1$ sînt în % pe zi.

Estimările ML ale lui $\alpha$ sînt 1,486 (SE 0,020) pentru BET, cu $n = 6\,670$ de zile; 1,540 (SE 0,021) pentru S&P 500, cu $n = 6\,714$; 1,606 (SE 0,021) pentru DAX, cu $n = 6\,786$; și 1,410 (SE 0,026) pentru Bitcoin, cu $n = 4\,384$. Estimările ML ale lui $\beta$ sînt −0,016 (SE 0,034) pentru BET, −0,181 (SE 0,036) pentru S&P 500, −0,179 (SE 0,040) pentru DAX și 0,019 (SE 0,036) pentru Bitcoin.

Scala $\hat\gamma$ este 0,62% pe zi pentru BET, 0,58% pentru S&P 500, 0,74% pentru DAX și 1,54% pentru Bitcoin. Pozițiile diferă între cele două parametrizări exact cum prevede @eq:s0s1: pentru S&P 500, $\hat\delta_0 = 0{,}093$ și $\hat\delta_1 = 0{,}000$, diferența fiind $\hat\beta\hat\gamma\tan\frac{\pi\hat\alpha}{2}$.

## Interpretarea estimărilor {h=1f0b028c}

Toate cele patru estimări ale lui $\alpha$ sînt mult sub 2. Pentru S&P 500, distanța față de cazul Normal este $(2 - 1{,}540)/0{,}021 \approx 22$ de erori standard, astfel că valoarea Normală este respinsă de orice test uzual. Cele mai groase cozi le are Bitcoin, cu $\hat\alpha = 1{,}410$.

Asimetria este negativă pentru S&P 500 și DAX, circa −0,18 pentru amîndoi: coada stîngă este mai groasă, ceea ce reflectă crahurile. Pentru BET și Bitcoin, $\hat\beta$ se află la mai puțin de două erori standard de 0, deci aceste serii nu au o asimetrie semnificativă.

Scala arată cît de diferite ca mărime sînt seriile: $\hat\gamma$ este 1,54% pe zi pentru Bitcoin și 0,58% pentru S&P 500. Abaterile standard de selecție, 3,49% și 1,21%, sînt mai mari decît $\sqrt{2}\hat\gamma$, abaterea standard a unei legi Normale cu același $\gamma$, pentru că sînt umflate de cozi. În sfîrșit, metoda lui McCulloch dă un $\hat\alpha$ mai mic decît ML pentru fiecare serie: cele două metode ponderează diferit cozile, prima prin cinci cuantile, a doua prin fiecare observație.

## Densități estimate, pe scară logaritmică (1/2) {h=7084b06b}

Graficul reprezintă densitatea empirică a randamentelor zilnice (puncte, dintr-o histogramă) pentru BET și S&P 500 pe scară logaritmică, împreună cu trei legi estimate: Normală (roșu), Student-t (verde) și stabilă, estimată prin ML (albastru). Scara logaritmică face vizibile cozile; pe o scară liniară ele nu s-ar distinge de zero.

Densitatea Normală este o parabolă pe această scară și eșuează din ambele puncte de vedere: este prea joasă în centru și coboară sub date dincolo de circa 4% în ambele direcții, deci prevede mult prea puține randamente mari. Densitățile Student-t și stabilă urmează îndeaproape corpul distribuției, de la vîrf pînă la densități de circa $10^{-2}$.

## Densități estimate, pe scară logaritmică (2/2) {h=bb63df9a}

Aceeași comparație pentru DAX și Bitcoin arată mai clar cozile îndepărtate. Dincolo de zona în care ambele legi cu cozi groase se potrivesc, densitatea stabilă rămîne deasupra punctelor, iar densitatea Student-t este mai aproape de ele. Pentru Bitcoin, ale cărui randamente ajung pe grafic la circa ±20%, diferența se vede în ambele cozi.

Este primul semn al tiparului întîlnit în toată secțiunea: legea stabilă descrie bine centrul și cozile moderate, dar cozile ei de tip putere, cu $\alpha$ în jur de 1,5, sînt prea groase pentru randamentele cele mai extreme.

## QQ plots față de cele trei modele (1/2) {h=35e50576}

Un QQ plot (graficul cuantilă–cuantilă, *quantile–quantile plot*) reprezintă cuantilele empirice ale randamentelor pe axa verticală în funcție de cuantilele unui model estimat pe axa orizontală. Dacă modelul este corect, punctele se află pe prima bisectoare. Abaterile de la capete arată ce coadă este greșit specificată și în ce sens.

Pentru BET și S&P 500, cele trei modele sînt de acord în centru, unde toate punctele se află pe dreaptă. Ele se despart la capete. Punctele modelului Normal (roșu) se îndepărtează abrupt de dreaptă: extremele empirice sînt mult mai mari decît cuantilele Normale. Punctele modelului stabil (albastru) se îndepărtează în sens opus și devin plate: modelul stabil își plasează cuantilele extreme mult dincolo de date, la circa −23% pentru BET, acolo unde cuantila empirică este de circa −10%. Punctele Student-t (verde) rămîn cel mai aproape de dreaptă.

## QQ plots față de cele trei modele (2/2) {h=61506779}

DAX și Bitcoin arată aceleași trei tipare. Distribuția Normală are cozi prea scurte, care apar ca niște capete abrupte; distribuția stabilă are cozi prea lungi, care apar ca niște capete plate, cu cuantile ale modelului mult dincolo de date; distribuția Student-t este cea mai apropiată dintre cele trei. Pentru Bitcoin, cuantilele stabile cele mai extreme ajung la circa 70% în valoare absolută, față de cuantile empirice de circa 17%.

Un QQ plot de acest tip identifică eșecul fiecărui model fără niciun test: modelul Normal subestimează extremele, iar modelul stabil le supraestimează.

## Coada stîngă pe axe log-log (1/2) {h=6a258349}

Graficul privește coada stîngă pe axe log-log. Punctele arată, pentru fiecare nivel de pierdere $x$ (în %, pe axa orizontală), proporția zilelor cu o pierdere mai mare decît $x$; liniile arată aceeași probabilitate conform fiecărui model estimat. Linia orizontală punctată marchează probabilitatea unei singure zile din eșantion, cea mai mică frecvență empirică nenulă.

Pentru BET și S&P 500, toate cele trei modele sînt de acord cu datele pentru pierderi de pînă la circa 2–3%. Mai departe, curba Normală (roșu) cade vertical. Curba stabilă (albastru) continuă ca o dreaptă, cum prevede @eq:tail. Punctele datelor se îndoaie în jos sub ea, iar curba Student-t (verde) le urmează mai departe.

## Coada stîngă pe axe log-log (2/2) {h=6c722b52}

Pentru DAX și Bitcoin, imaginea este aceeași. Coada empirică se curbează în jos mai repede decît dreapta stabilă: pe axe log-log, panta ei devine mai abruptă decît $-\hat\alpha$. În termeni de tail index, coada îndepărtată a datelor este mai subțire decît o coadă stabilă cu indicele $\hat\alpha$, deci tail index-ul ei este mai mare decît $\hat\alpha$.

Estimarea ML a lui $\alpha$ este determinată mai ales de numeroasele observații din corp și din cozile moderate. Ea nu se poate adapta celor cîteva observații extreme, pentru că o lege stabilă leagă panta cozii îndepărtate de același $\alpha$ care dă forma centrului.

## Numărarea pierderilor mari {h=22bdea76}

Tabelul transformă probabilitățile din cozi în număr de zile. Pentru fiecare model, numărul așteptat de zile cu o pierdere de peste $x$ este $n \times P(r < -x)$ conform legii estimate. Pentru S&P 500 s-au observat 23 de zile cu o pierdere de peste 5%; modelul Normal prevede 0,1 zile, Student-t 27,0, iar legea stabilă 57,3. Pentru pierderi de peste 10% s-a observat o zi, față de practic zero, 4,5 și 19,1 zile. Pentru pierderi de peste 15% nu s-a observat niciuna, față de practic zero, 1,5 și 10,2.

Celelalte serii confirmă tiparul. Pentru BET s-au observat 6 zile cu pierderi de peste 10%, față de practic zero conform modelului Normal, 8,2 conform Student-t și 22,8 conform legii stabile. Pentru Bitcoin s-au observat 56 de zile cu pierderi de peste 10%, față de 8,1, 54,2 și 73,9, și 4 zile cu pierderi de peste 20%, față de practic zero, 12,3 și 26,9.

Modelul Normal nu prevede aproape nicio pierdere mare. Legea stabilă prevede prea multe, iar diferența crește odată cu pragul. Student-t este cea mai apropiată de valorile observate. Pentru S&P 500, legea stabilă estimată prevede 19,1 zile sub −10% în 6 714 zile, acolo unde s-a observat una singură.

## VaR 1% și VaR 0,1% conform celor trei modele {h=ee3aed2e}

Valoarea la risc la nivelul $p$ este $\mathrm{VaR}_p = -q_p$, pierderea zilnică depășită cu probabilitatea $p$ (Capitolul 10). Nivelul este notat aici cu $p$, pentru că $\alpha$ desemnează deja indicele de stabilitate. Tabelul compară VaR empiric, obținut din cuantila de selecție, cu VaR al celor trei modele estimate, la nivelurile 1% și 0,1%.

La nivelul de 1%, modelul Normal dă valori prea mici, iar modelul stabil valori prea mari. Pentru S&P 500, VaR 1% empiric este 3,5%; modelul Normal dă 2,8%, Student-t 3,5%, iar legea stabilă 4,5%. Pentru Bitcoin, valoarea empirică este 10,5%, față de 8,0%, 11,1% și 14,3%.

La nivelul de 0,1%, modelul stabil depășește mult mai mult datele. VaR 0,1% al lui este de aproximativ două pînă la patru ori mai mare decît cuantila empirică: 23,1% față de 9,5% pentru BET, 19,7% față de 7,2% pentru S&P 500, 20,0% față de 7,1% pentru DAX și 74,2% față de 18,1% pentru Bitcoin. Modelul Normal, cu 3,7% pentru S&P 500, dă circa jumătate din valoarea empirică.

## Compararea celor trei modele {h=47129fdd}

Criteriul informațional Akaike (AIC, *Akaike information criterion*) este $2k - 2\ell(\hat\theta)$, unde $k$ este numărul de parametri, iar $\ell(\hat\theta)$ log-verosimilitatea maximizată; se preferă valoarea mai mică. Criteriul penalizează parametrii suplimentari, astfel că un model cu mai mulți parametri trebuie să îmbunătățească destul verosimilitatea pentru a fi preferat.

Ambele modele cu cozi groase sînt preferate distribuției Normale, pentru fiecare serie: Student-t cu 1 306 (DAX) pînă la 2 411 (BET) puncte AIC, legea stabilă cu 1 204 pînă la 2 319; pentru S&P 500, AIC este 21 650 pentru modelul Normal, 19 702 pentru Student-t și 19 794 pentru legea stabilă. Student-t este preferată pentru fiecare serie, deși are mai puțini parametri decît legea stabilă. Pentru S&P 500, avantajul ei este de 92 de puncte.

Gradele de libertate estimate $\hat\nu$ ale distribuției Student-t sînt 2,43 pentru BET, 2,67 pentru S&P 500, 3,09 pentru DAX și 2,23 pentru Bitcoin, toate între 2 și 3,1. O lege Student-t are momente finite doar de ordin mai mic decît $\nu$: aceste estimări implică varianță finită, dar aproape niciun moment finit de ordin superior.

## Recapitulare: estimarea pe randamente reale {h=80cc63fa}

Pe toate cele patru serii, estimările ML ale lui $\alpha$ se află între 1,410 și 1,606, departe de 2. Legea stabilă descrie corpul distribuției mult mai bine decît distribuția Normală.

Ea supraestimează cozile îndepărtate: prevede prea multe pierderi extreme și dă un VaR 0,1% prea mare. După AIC, distribuția Student-t este preferată pentru fiecare serie.

## @end

Estimată pe randamente reale, legea stabilă este o îmbunătățire mare față de distribuția Normală și o descriere mai slabă decît distribuția Student-t. Valorile $\hat\alpha$ între 1,410 și 1,606 surprind cozile groase din apropierea centrului, dar același $\alpha$ impune cozi de tip putere prea groase în zona extremă: 19,1 zile așteptate cu pierderi de peste 10% pentru S&P 500, față de una observată, și un VaR 0,1% de două pînă la patru ori mai mare decît cel empiric. Estimările Student-t $\hat\nu$ între 2,23 și 3,09 indică o varianță finită. [Secțiunea următoare](#sec-9) testează direct varianța infinită.

# Critica: varianță finită sau infinită?

Estimările din secțiunea anterioară sugerează că cozile îndepărtate ale randamentelor sînt mai subțiri decît orice coadă stabilă. Secțiunea testează direct ipoteza varianței infinite, cu trei teste derivate din teoria capitolului, apoi explică de ce randamentele zilnice par totuși stabile.

## Testul 1: rămîne α la fel cînd randamentele se adună? {h=308df4b2}

Primul test folosește chiar proprietatea de stabilitate. Dacă randamentele zilnice ar fi i.i.d. stabile, sumele săptămînale și lunare ar fi stabile cu același $\alpha$. Dacă, în schimb, randamentele zilnice au varianță finită, GCLT prevede că $\alpha$ estimat urcă spre 2 pe măsură ce randamentele se adună pe orizonturi mai lungi.

Graficul raportează estimările ML ale lui $\alpha$, cu intervale de încredere de 95%, pentru randamentele zilnice, săptămînale și lunare ale celor patru serii; linia punctată marchează $\alpha = 2$. Pentru S&P 500, estimările cresc de la 1,54 (zilnic) la 1,64 (săptămînal) și 1,82 (lunar). Pentru Bitcoin, ele cresc de la 1,41 la 1,67 și apoi la 2,00, la limita 2: pentru randamentele lunare Bitcoin, estimarea stabilă ajunge în cazul Normal.

## Interpretarea testului de agregare {h=46f10a17}

$\alpha$ estimat crește cu orizontul pentru S&P 500, DAX și Bitcoin. Pentru DAX, șirul este $1{,}61 \to 1{,}72 \to 1{,}69$, deci creșterea are loc între randamentele zilnice și cele săptămînale. Pentru S&P 500, intervalele de 95% ale estimărilor zilnice și lunare nu se suprapun, astfel că creșterea este clară din punct de vedere statistic.

BET este excepția: $1{,}49 \to 1{,}56 \to 1{,}50$, fără o creștere clară. Eșantionul lunar este scurt, 321 de luni, iar intervalul estimării lunare este larg, așa că testul are o putere mică pentru această serie.

Același tipar apare în studiile clasice ale lui @Officer1972 și @AkgirayBooth1988. Un $\alpha$ crescător este exact ceea ce prevede GCLT pentru sume de randamente cu varianță finită care converg lent la distribuția Normală: în această lume, estimarea stabilă pe date zilnice este o aproximare pe care agregarea o elimină treptat.

## Testul 2: se stabilizează varianța de selecție? {h=34f4a040}

Al doilea test repetă observația lui Mandelbrot pe date actuale. Graficul reprezintă varianța de selecție cumulată a randamentelor logaritmice zilnice ale S&P 500 (în %, linia albastră groasă) din 2000 pînă în 2026, pe scară logaritmică. Varianța sare în 2008 și în 2020, anii celor mai mari crize din eșantion, și apoi se stabilizează în jur de 1,47.

Liniile subțiri sînt traiectorii de aceeași lungime simulate din legea stabilă estimată. Ele se comportă ca în [simularea din secțiunea despre cozi](#s34): salturi repetate, scădere lentă și nicio convergență. Varianțele lor finale se află între 5,4 și 15,6, mult peste valoarea datelor. Varianța observată este stabilă într-un mod pe care o lege stabilă cu $\hat\alpha = 1{,}540$ nu îl permite.

## Testul 3: cît de groasă este coada îndepărtată? {h=0f67e7b5}

$\alpha$ stabil este determinat de întreaga distribuție, mai ales de corpul ei, unde se află majoritatea observațiilor. Tail index-ul cozii îndepărtate se estimează separat, cu estimatorul Hill [@Hill1975], care folosește doar observațiile cele mai mari; Capitolul 5 îl dezvoltă.

Pentru indicii bursieri, coada îndepărtată scade aproximativ ca $x^{-3}$, „legea cubică inversă” (*inverse cubic law*) [@GopikrishnanEtAl1999]. Un tail index în jur de 3 înseamnă varianță finită, iar nicio lege stabilă nu îl are, pentru că orice lege stabilă diferită de distribuția Normală are un tail index sub 2.

Numărul pierderilor mari din [secțiunea anterioară](#s58) confirmă: legile stabile estimate prevăd mult mai multe pierderi de peste 10% decît s-au observat. @LauLauWingender1990 aduc alte dovezi împotriva modelului stabil, din comportamentul momentelor de selecție.

## Originea aspectului stabil al randamentelor zilnice {h=5af048a4}

Două explicații dau seama de aspectul stabil al randamentelor zilnice. Prima este volatility clustering (Capitolele 2 și 9): perioadele calme și cele agitate alternează. Un amestec de legi Normale cu varianță variabilă are cozi groase și varianță finită [@BlattbergGonedes1974]. Dacă pe datele generate de un astfel de amestec se estimează o singură lege stabilă, se obține $\hat\alpha < 2$, deși nicio observație nu provine dintr-o lege cu varianță infinită.

A doua explicație o oferă legile stabile trunchiate și temperate (*tempered stable laws*). Ele se comportă ca legile stabile la scări moderate, dar au cozi mai subțiri în zona extremă, astfel că toate momentele lor sînt finite. @MantegnaStanley1995 au estimat un zbor Lévy trunchiat (*truncated Lévy flight*) pe S&P 500. @GrabchakSamorodnitsky2010 arată că astfel de legi explică de ce randamentele par stabile pe orizonturi scurte și Normale pe orizonturi lungi, adică tiparul testului de agregare.

## Concluzia criticii {h=5cc38afc}

Mandelbrot a avut dreptate în privința faptelor. Randamentele zilnice au cozi groase, iar distribuția Normală descrie foarte slab datele; fiecare estimare din acest capitol o confirmă.

Varianța infinită însă nu este susținută. $\alpha$ estimat crește prin agregare, varianța de selecție a randamentelor reale se stabilizează, iar coada îndepărtată este mai subțire decît orice coadă stabilă. Toate cele trei teste indică aceeași direcție.

Legile stabile rămîn utile în mai multe roluri: ca etalon pentru cozi groase, ca instrument de simulare, prin ideea de scalare între orizonturi și pentru date cu varianță cu adevărat infinită, precum unele daune de asigurări și unele tokenuri cripto. GCLT rămîne o reamintire că regulile construite pe $\sqrt{n}$ cer varianță finită. [Capitolul 5](ch5_ro.html) estimează direct tail index-ul, cu teoria valorilor extreme.

## Recapitulare: critica {h=34795d21}

Stabilitatea prevede același $\alpha$ la orice orizont, dar datele arată un $\alpha$ care crește cu orizontul. Varianța de selecție a randamentelor reale se stabilizează, în timp ce aceea a eșantioanelor stabile nu.

Volatility clustering și cozile temperate explică valoarea aparentă $\alpha < 2$ a randamentelor zilnice. Verdictul capitolului este: cozi groase, da; varianță infinită, nu.

## @end

Cele trei teste resping varianța infinită fără a respinge cozile groase. Agregarea ridică $\hat\alpha$ spre 2, cum prevede GCLT pentru randamente cu varianță finită; varianța cumulată a S&P 500 se stabilizează în jur de 1,47, în timp ce traiectoriile stabile ajung între 5,4 și 15,6; iar coada îndepărtată scade aproximativ ca $x^{-3}$, cu un tail index în afara domeniului stabil. Volatility clustering și cozile temperate împacă aceste rezultate cu aspectul stabil al datelor zilnice.

# AI pentru descoperire științifică

Instrumentele capitolului se pot aplica unor întrebări încă deschise. Secțiunea formulează o astfel de întrebare, despre cozile Bitcoin, descrie cum ar putea instrumentele AI (inteligență artificială) să sprijine studiul ei, enumeră verificările de care are nevoie și propune o idee de proiect.

## O întrebare deschisă {h=23a0a79c}

Întrebarea deschisă este dacă cozile randamentelor Bitcoin au devenit mai subțiri pe măsură ce piața s-a maturizat. Piața a cîștigat mai mulți participanți, contracte futures din 2017 și fonduri spot tranzacționate la bursă (ETF, *exchange-traded funds*) din 2024. Dacă cozile au devenit mai subțiri, modelele de risc estimate pe date vechi supraestimează riscul de azi.

Întrebarea rămîne deschisă din două motive. $\alpha$ estimat variază odată cu regimurile de volatilitate, astfel că o schimbare a lui $\hat\alpha$ poate reflecta o perioadă mai calmă, nu o coadă mai subțire. În plus, eșantioanele anuale sînt scurte, iar estimările au intervale largi.

Pe eșantionul complet, Bitcoin are $\hat\alpha = 1{,}410$ (SE 0,026), cel mai mic dintre cele patru serii ale capitolului. Instrumentele AI pot accelera un astfel de studiu; ele nu înlocuiesc verificarea lui [@WangEtAl2023].

## Utilizări posibile ale instrumentelor AI {h=e113637e}

Instrumentele AI pot contribui în patru etape. Pentru literatură, ele pot inventaria lucrările despre estimarea cozilor activelor cripto și pot rezuma metodele lor. Pentru cod, pot scrie o primă versiune a unui estimator McCulloch pe ferestre mobile, cu intervale bootstrap. Pentru robustețe, pot propune alte ferestre, ML în loc de cuantile sau $\nu$ din Student-t în loc de $\alpha$. Pentru explicație, pot scrie o primă versiune a interpretării unui tabel de estimări anuale.

Exemplul de prompt de pe slide cere exact a doua contribuție: o funcție Python care estimează $\alpha$ stabil al randamentelor logaritmice zilnice pe ferestre mobile de 365 de zile, prin metoda din @McCulloch1986, și adaugă intervale bootstrap de 95%. Fiecare dintre aceste rezultate este o schiță, iar slide-ul următor enumeră ce trebuie verificat înainte de a o folosi.

## Verificări necesare {h=b18438eb}

Parametrizarea vine prima: codul trebuie să precizeze dacă folosește S0 sau S1; scipy folosește implicit S1, iar poziția diferă cu $\beta\gamma\tan\frac{\pi\alpha}{2}$. Unitățile vin a doua: cu randamente în % sau în zecimale, $\gamma$ și $\delta$ se schimbă, în timp ce $\alpha$ și $\beta$ nu.

Ferestrele suprapuse nu sînt independente, așa că estimările mobile nu trebuie citite ca teste separate; o tendință a lui $\hat\alpha$ pe ferestre mobile poate proveni din cîteva zile extreme care rămîn în multe ferestre. Rezultatele numerice trebuie recalculate pentru cel puțin o fereastră cu o a doua metodă, ML sau tabelele de cuantile. În sfîrșit, fiecare lucrare citată trebuie să existe, iar DOI-ul ei trebuie verificat.

## Idee de proiect {h=0ff84ec9}

Proiectul întreabă dacă coada randamentelor Bitcoin este acum mai subțire decît în 2014–2018. Datele sînt prețurile de închidere zilnice ale Bitcoin din datele cursului, 2014–2026.

Pașii urmează capitolul. Estimați $\alpha$ prin metoda lui McCulloch și prin ML pentru fiecare an calendaristic, cu intervale bootstrap. Repetați estimarea pe randamente împărțite la o volatilitate mobilă, pentru a vedea dacă tendința se menține după eliminarea efectului de volatility clustering. Comparați rezultatele cu $\hat\nu$ din Student-t pentru fiecare an.

Livrabilele sînt un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele. Orice folosire a AI trebuie declarată, împreună cu lista erorilor AI care au fost corectate.

## @end

Întrebarea despre cozile Bitcoin folosește toate instrumentele capitolului: ambii estimatori, distincția dintre S0 și S1, legătura dintre volatility clustering și cozile groase aparente și comparația cu Student-t. Instrumentele AI pot scrie o primă versiune a codului și a trecerii în revistă a literaturii, dar verificările privind parametrizarea, unitățile, ferestrele suprapuse și referințele rămîn sarcina cercetătorului.

# Rezumat

Rezumatul adună formulele capitolului și trei întrebări de autoevaluare.

## Formule de reținut {h=5cf5f30a}

Tabelul reunește formulele capitolului. Stabilitatea este proprietatea @eq:stable, cu regula scalei $c^\alpha = a^\alpha + b^\alpha$ @eq:scale, iar suma a $n$ copii are factorul de scală $n^{1/\alpha}$. Funcția caracteristică în S1 este @eq:cf-s1, iar pozițiile din S0 și S1 sînt legate prin @eq:s0s1. Cazul Normal este $S(2; \beta; \gamma; \delta) = N(\delta, 2\gamma^2)$.

Cozile scad ca în @eq:tail, $P(X > x) \sim \gamma^\alpha c_\alpha(1 + \beta)x^{-\alpha}$, iar pentru $\alpha < 2$ momentul absolut de ordin $p$ este finit dacă și numai dacă $p < \alpha$ @eq:moments. Formula CMS @eq:cms simulează cazul simetric, iar rapoartele lui McCulloch @eq:nu sînt punctul de plecare al estimării.

## Autoevaluare {h=7b6522fc}

Prima întrebare: randamentele zilnice sînt i.i.d. $S(1{,}5; 0; \gamma; 0)$; de cîte ori crește scala de la 1 zi la 20 de zile? Cu factorul $20^{1/1{,}5} = 7{,}37$, adică de 1,65 ori mai mult decît factorul $\sqrt{20}$ al regulii rădăcinii pătrate a timpului.

A doua întrebare: o lege estimată are $\alpha = 1{,}2$; există media și varianța? Conform @eq:moments, media există, pentru că $1 < 1{,}2$, iar varianța nu, pentru că $2 > 1{,}2$.

A treia întrebare: scipy raportează $\delta = 0{,}05$ cu $\alpha = 1{,}7$, $\beta = -0{,}2$ și $\gamma = 0{,}6$; cît este $\delta_0$? Scipy folosește S1, deci @eq:s0s1 dă $\delta_0 = 0{,}05 + (-0{,}2)(0{,}6)\tan(0{,}85\pi) = 0{,}05 + 0{,}061 = 0{,}111$. Capitolul următor, [Capitolul 4](ch4_ro.html), tratează probabilitățile pentru finanțe.

## @end

Legile stabile sînt singurele legi închise la adunare și singurele limite ale sumelor normalizate. Ele descriu cozile groase ale randamentelor zilnice mult mai bine decît distribuția Normală, dar varianța lor infinită este contrazisă de agregare, de comportamentul varianței de selecție și de cozile îndepărtate. [Capitolul 5](ch5_ro.html) estimează direct coada îndepărtată, cu teoria valorilor extreme.
