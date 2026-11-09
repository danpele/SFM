<!--
Textul de curs al versiunii web, capitolul 12 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie → Autor (an), [@Cheie] → (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 12 --hashes
-->

## @intro

@text
Capitolele 2–10 au măsurat riscul prețurilor: distribuția randamentelor, volatilitatea lor, cozile și VaR și ES care le rezumă. O bancă are însă și un al doilea tip de risc, adesea mai mare decît riscul de piață: riscul ca un debitor să nu își ramburseze creditul. În fiecare zi, un creditor trebuie să decidă dacă acceptă o cerere de credit și cu ce dobîndă, iar decizia se sprijină pe un număr, probabilitatea de nerambursare (PD, *probability of default*) a solicitantului. Capitolul întreabă cum transformă o bancă datele unui solicitant în această probabilitate și de unde știe că cifra este corectă.

Răspunsul este un lanț de instrumente statistice. Pierderea așteptată a unui credit și capitalul Basel care acoperă pierderile de peste acest nivel pornesc amîndouă de la PD. PD se estimează din creditele trecute cu regresia logistică sau cu analiza discriminantă a lui Fisher, după ce fiecare variabilă a fost recodificată prin Weight of Evidence; modelul estimat este apoi scris ca un scorecard de puncte. Ca dovezi empirice, capitolul folosește 1 000 de credite de consum ale unei bănci germane și 30 000 de deținători de carduri de credit din Taiwan. O singură variabilă, starea contului curent, mută deja ponderea creditelor neperformante de la 11,7% la 49,3%, iar un scorecard cu șase variabile plasează un credit neperformant ales la întîmplare deasupra unui credit performant ales la întîmplare în aproximativ 80% din perechile unui eșantion de test.

O cifră care arată bine nu face încă un model bun, iar a doua jumătate a capitolului este despre validare: discriminarea și calibrarea măsurate în afara eșantionului, un prag ales din costurile celor două tipuri de eroare, distorsiunea creată de solicitanții pe care banca i-a respins și întrebarea dacă un scor precis tratează echitabil grupurile de solicitanți. Capitolul se încheie cu două modele clasice ale nerambursării companiilor: scorul Z al lui Altman, construit din indicatori contabili, și distanța pînă la nerambursare a lui Merton, construită din prețuri de piață.

@chart sfm_ch12_default_by_status
Ponderea creditelor neperformante după starea contului curent în datele South German Credit (1 000 de credite de consum, 1973–1975). Linia întreruptă este media eșantionului, 30%.

@goals
- **Definiți** PD, LGD, EAD și pierderea așteptată și explicați ce adaugă capitalul IRB din Basel față de pierderea așteptată.
- **Calculați** Weight of Evidence și Information Value ale unei variabile dintr-un tabel de frecvențe.
- **Estimați** o regresie logistică și interpretați coeficienții ca modificări ale logaritmului șansei și ca rapoarte ale șanselor.
- **Deduceți** discriminantul liniar al lui Fisher și comparați-l cu modelul logit.
- **Scalați** un scorecard și validați-l în afara eșantionului cu matricea de confuzie, AUC, Gini, KS, scorul Brier și diagrama de calibrare.

[Seminarul 12](../RO/Seminarii/seminar12_modele_scoring_ro.pdf) precede acest curs și exersează aceleași instrumente pe date.

@plan
- Scorul de credit: definiție | Decizia de creditare, istoria scoringului, PD, LGD, EAD, pierderea așteptată și capitalul IRB din Basel.
- Datele de nerambursare | Datele South German Credit, erorile de codificare din datele originale, dezechilibrul claselor și împărțirea în eșantion de estimare și eșantion de test.
- Weight of Evidence și Information Value | Recodificarea unei variabile prin dovada pe care o aduce despre nerambursare, regulile de discretizare și ordonarea celor 20 de variabile.
- Regresia logistică | Șansa și logaritmul șansei, modelul logit, rapoartele șanselor, un scorecard WoE și corecția pentru suprareprezentare.
- Analiza discriminantă liniară | Criteriul lui Fisher, LDA ca model de probabilitate și comparația cu modelul logit.
- Construirea scorecard-ului | Punctele pentru dublarea șansei, punctele fiecărui atribut și scorurile eșantionului de test.
- Validarea | Matricea de confuzie, praguri și costuri, ROC, AUC, Gini, CAP, KS, calibrarea, scorul Brier, validarea încrucișată și deriva populației.
- Reject inference și echitatea | Rezultatele lipsă ale solicitanților respinși și trei criterii de echitate, pe datele din Taiwan.
- Nerambursarea companiilor: Altman și Merton | Scorul Z din indicatori contabili și distanța pînă la nerambursare din prețuri de piață.
- AI pentru descoperire științifică | O întrebare deschisă despre scorurile de învățare automată, ce poate face AI și ce trebuie verificat.
- Rezumat | Idei de reținut, formule de reținut și întrebări de autoevaluare.

# Scorul de credit: definiție

Un scor de credit este un număr care ordonează solicitanții după riscul lor de a nu rambursa. Secțiunea prezintă decizia pe care o servește scorul, amintește cum scoringul statistic a înlocuit judecata ofițerilor de credit și definește cei trei parametri de risc ai unui credit: probabilitatea de nerambursare, pierderea în caz de nerambursare și expunerea în momentul nerambursării. Produsul lor este pierderea așteptată, pe care banca o include în preț; formula de capital din Basel măsoară apoi pierderea de peste acest nivel într-un an prost.

## Decizia de creditare {h=423a8e8f}

Cînd un solicitant cere un credit, banca ia două decizii: acceptă sau respinge cererea și, dacă o acceptă, stabilește dobînda. Ambele decizii depind de cît de probabil este ca solicitantul să intre în nerambursare (*default*). În termeni obișnuiți, nerambursarea înseamnă că debitorul nu plătește conform contractului. Cadrul Basel II al [Comitetului de la Basel pentru Supraveghere Bancară](https://www.bis.org/publ/bcbs128.htm) (paragraful 452) dă definiția de reglementare: nerambursarea are loc cînd banca apreciază că debitorul nu va plăti integral sau cînd debitorul întîrzie mai mult de 90 de zile o obligație semnificativă.

Un scor de credit este un număr calculat din datele unui solicitant, care ordonează solicitanții după riscul lor de nerambursare. Probabilitatea de nerambursare (PD) este probabilitatea ca debitorul să intre în nerambursare pe un orizont fix, de obicei un an. Un scor poate fi bun în două sensuri diferite. El discriminează bine dacă ordonează corect solicitanții, astfel încît cei care vor intra mai tîrziu în nerambursare să aibă de regulă scoruri mai proaste decît cei care rambursează. El este calibrat dacă PD pe care o implică are nivelul corect: dintre solicitanții cu PD de 5%, aproximativ 5% ar trebui să intre în nerambursare. Ordonarea este suficientă pentru a accepta sau a respinge; prețul, provizioanele și capitalul au nevoie și de nivel, iar capitolul le măsoară pe amîndouă.

Scorurile diferă după client și după informația pe care o folosesc. Scorurile de aplicare (*application scores*) se calculează pentru clienții noi, din datele din cererea lor; scorurile comportamentale (*behavioural scores*) se calculează pentru clienții existenți, din istoricul plăților lor. Scorurile pentru persoane fizice se sprijină pe modele statistice estimate pe multe credite mici, iar scorurile pentru companii folosesc indicatori contabili și prețuri de piață, ca în modelele Altman și Merton din [secțiunea 9](#sec-9).

## Începuturile scoringului de credit: Durand și FICO {h=a6f47369}

Scoringul statistic de credit este mai vechi decît calculatorul. În 1941, @Durand1941 a studiat creditele în rate ale creditorilor americani și a separat creditele bune de cele rele cu analiza discriminantă pe care Fisher o propusese cu cinci ani mai devreme; acesta este primul studiu statistic de scoring de credit. În 1956, inginerul Bill Fair și matematicianul Earl Isaac au înființat Fair, Isaac and Company, azi [FICO](https://www.myfico.com/credit-education/blog/history-of-the-fico-score). În 1989, compania a introdus scorul FICO, calculat din rapoartele birourilor de credit și cuprins între 300 și 850 de puncte, care a devenit scorul standard pentru persoane fizice în Statele Unite.

Articolul de sinteză al lui @HandHenley1997 descrie ce s-a schimbat. Înainte de scoring, un ofițer de credit judeca fiecare cerere din experiență; scoringul statistic a înlocuit această judecată cu o regulă identică pentru toți solicitanții. Regula este mai ieftină și mai rapidă și, spre deosebire de o judecată personală, poate fi testată: ratele ei de eroare se pot măsura pe creditele trecute. O regulă identică pentru toți nu este însă automat echitabilă, deoarece variabilele pe care le folosește pot ține locul unor caracteristici protejate de lege; [secțiunea 8](#sec-8) revine la această întrebare.

## PD, LGD, EAD și pierderea așteptată {h=61050c1a}

Basel II descrie riscul de credit al unui împrumut prin trei parametri. PD este probabilitatea de nerambursare într-un an. Pierderea în caz de nerambursare (LGD, *loss given default*) este partea din expunere pe care banca o pierde dacă are loc nerambursarea, după ce recuperează ce poate din garanții sau din vînzarea creanței. Expunerea în momentul nerambursării (EAD, *exposure at default*) este suma datorată în momentul nerambursării; pentru un credit fix este soldul rămas, iar pentru o linie de credit include și partea pe care debitorul o poate trage înainte de nerambursare.

Pierderea așteptată (EL, *expected loss*) este media pierderii pe un an. Dacă nerambursarea este un eveniment cu probabilitatea PD, iar pierderea, cînd apare, este $\mathrm{LGD} \times \mathrm{EAD}$, atunci

$$ \mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD}, $$ {#eq:el}

un produs valabil cînd PD și LGD sînt independente. Un credit de 100 000 de lei cu PD de 2% și LGD de 45% are o pierdere așteptată de $0{,}02 \times 0{,}45 \times 100\,000 = 900$ de lei pe an.

Pierderea așteptată este un cost al activității, ca salariile personalului: banca o include în dobîndă și o acoperă prin provizioane. Pericolul este în altă parte. Într-un an prost, nerambursările multor debitori cresc împreună, iar pierderea poate depăși cu mult media ei. Această parte, pierderea neașteptată (*unexpected loss*), trebuie absorbită de capitalul băncii, iar slide-urile următoare o măsoară.

## Formula de capital IRB din Basel {h=247e19ac}

În abordarea bazată pe ratinguri interne (IRB, *internal ratings-based*) din Basel II, banca estimează PD a fiecărei expuneri cu propriile modele (iar în varianta avansată și LGD și EAD), în timp ce autoritatea de reglementare stabilește formula care transformă aceste estimări în capital (paragrafele 328–330 din [cadrul Basel II](https://www.bis.org/publ/bcbs128.htm)). Pentru o expunere de retail, capitalul pe unitatea de EAD este

$$ z^* = \frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\,\Phi^{-1}(0{,}999)}{\sqrt{1 - R}}, $$

$$ K = \mathrm{LGD}\,\big[\Phi(z^*) - \mathrm{PD}\big], \qquad \mathrm{RWA} = 12{,}5\,K\,\mathrm{EAD}, $$ {#eq:irb}

unde $\Phi$ și $\Phi^{-1}$ sînt funcția de repartiție (*distribution function*) a distribuției Normale standard $N(0,1)$ și inversa ei, iar $R$ este corelația debitorilor cu un factor comun, starea economiei. Activele ponderate la risc (RWA, *risk-weighted assets*) sînt de $12{,}5 = 1/0{,}08$ ori capitalul, deoarece banca trebuie să dețină capital de cel puțin 8% din activele ponderate la risc.

Logica formulei provine de la @Gordy2003. Termenul $\Phi(z^*)$ este rata de nerambursare a portofoliului într-un an al cărui factor comun este atît de prost încît este depășit doar cu probabilitatea 0,1%, un an care apare o dată la 1 000 de ani. Înmulțit cu LGD, el dă pierderea într-un astfel de an; scăzînd pierderea așteptată, rămîne pierderea neașteptată. $K$ este deci VaR 0,1% al pierderii din credit minus pierderea așteptată, aceeași idee de cuantilă ca VaR din [Capitolul 10](ch10_ro.html), aplicată unui portofoliu de credite.

## PD în anii favorabili și nefavorabili: modelul cu un factor {h=a1f45ce6}

Modelul din spatele formulei IRB este un model cu un factor. Debitorul $i$ intră în nerambursare cînd un indice latent al sănătății lui financiare scade sub un prag:

$$ \sqrt{\rho}\,Y + \sqrt{1 - \rho}\,\varepsilon_i < \Phi^{-1}(\mathrm{PD}), $$

unde $Y$ este starea economiei, comună tuturor debitorilor, $\varepsilon_i$ este șocul propriu al debitorului, $Y$ și $\varepsilon_i$ sînt variabile $N(0,1)$ independente, iar $\rho$ este ponderea factorului comun. Indicele este el însuși $N(0,1)$, astfel încît probabilitatea necondiționată de nerambursare este exact PD. Pentru un an anume, $Y = y$, doar $\varepsilon_i$ este aleator, iar PD condiționată este

$$ \mathrm{PD}(y) = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}) - \sqrt{\rho}\,y}{\sqrt{1 - \rho}}\right). $$ {#eq:condpd}

Graficul reprezintă această PD condiționată în funcție de PD necondiționată, cu $\rho = 0{,}2$, pentru trei stări ale economiei: un an prost ($y = -3$), un an obișnuit ($y = 0$) și un an bun ($y = 3$). Un credit cu PD = 2% intră în nerambursare cu probabilitatea 21,3% într-un an prost ($y = -3$), cu 1,1% într-un an obișnuit ($y = 0$, mediana factorului) și aproape niciodată într-un an bun. Media pe toți anii este 2%, dar anii înșiși diferă foarte mult: nerambursările vin în grupuri. Formula IRB evaluează @eq:condpd în $y = \Phi^{-1}(0{,}001)$, anul mai prost decît 99,9% din toți anii, cu corelația $R$ în locul lui $\rho$.

## Exemplu rezolvat: capitalul IRB al unui credit de retail {h=4d3b0160}

Pentru alte expuneri de retail (credite de consum care nu sînt nici ipotecare, nici revolving), Basel II stabilește corelația ca o medie ponderată,

$$ R = 0{,}03\,w + 0{,}16\,(1 - w), \qquad w = \frac{1 - e^{-35\,\mathrm{PD}}}{1 - e^{-35}}. $$

Ponderea $w$ ia valori în $[0, 1]$ și crește cu PD, astfel încît $R$ scade de la 0,16 pentru creditele foarte sigure la 0,03 pentru cele foarte riscante: nerambursarea unui debitor sigur este determinată mai ales de economie, cea a unui debitor riscant mai ales de propria lui situație. Pentru PD = 2%, formula dă $R = 0{,}095$; cele două cuantile ale distribuției Normale necesare sînt $\Phi^{-1}(0{,}02) = -2{,}054$ și $\Phi^{-1}(0{,}999) = 3{,}090$.

Argumentul lui $\Phi$ este $(-2{,}054 + \sqrt{0{,}095} \times 3{,}090)/\sqrt{1 - 0{,}095} = -1{,}160$, iar $\Phi(-1{,}160) = 12{,}31\%$: aceasta este PD a creditului în anul prost al formulei. Capitalul este $K = 0{,}45 \times (12{,}31\% - 2\%) = 4{,}64\%$ din EAD, adică 4 639 de lei pentru creditul de 100 000 de lei. Activele ponderate la risc sînt $12{,}5 \times 4\,639 = 57\,986$ de lei, o pondere de risc de 58%.

Cele două cifre ale creditului au acum roluri distincte. Pierderea așteptată de 900 de lei intră în dobîndă; capitalul de 4 639 de lei rămîne în bancă pentru a absorbi un an în care nerambursările ajung la 12,31% din astfel de credite în loc de 2%.

## Pierderea așteptată și capitalul IRB în funcție de PD {h=96d72bda}

Graficul repetă calculul pentru fiecare PD, cu LGD = 45%, pentru trei clase de expuneri de retail: alte expuneri de retail, unde $R$ scade de la 0,16 la 0,03 ca mai sus; creditele ipotecare rezidențiale, cu $R = 0{,}15$ fix; și expunerile revolving calificate, precum cardurile de credit, cu $R = 0{,}04$ fix. Pierderea așteptată, $0{,}45 \times \mathrm{PD}$ pe unitatea de expunere, este o dreaptă.

Curbele de capital se comportă altfel. Ele cresc repede la PD mici, unde o creștere mică a PD mută mult rata de nerambursare din anul prost, și apoi se aplatizează. Un credit foarte riscant este în mare parte o pierdere așteptată: o mare parte din ce va pierde este deja cunoscută, inclusă în preț și acoperită prin provizioane, și doar partea de peste această medie are nevoie de capital. Pentru alte expuneri de retail, dreapta pierderii așteptate intersectează curba de capital la o PD de aproximativ 16%; peste acest nivel, creditul are o pierdere așteptată mai mare decît capitalul păstrat pentru un an prost. Curba creditelor ipotecare este cea mai înaltă, deoarece corelația lor mare le face să intre în nerambursare împreună într-un an prost.

## @end

Un scor de credit ordonează solicitanții după riscul de nerambursare, iar PD a lui ar trebui să aibă și nivelul corect. Pierderea așteptată, PD × LGD × EAD, intră în preț; capitalul IRB acoperă pierderea neașteptată, calculată din PD într-un an prost care apare o dată la 1 000 de ani într-un model cu un factor. Pentru un credit de retail de 100 000 de lei cu PD 2% și LGD 45%, pierderea așteptată este 900 de lei, iar capitalul 4 639 de lei. Ambele cifre pornesc de la PD, iar [secțiunea următoare](#sec-2) prezintă datele din care se estimează PD.

# Datele de nerambursare

Un model de scoring se estimează pe credite trecute al căror rezultat este cunoscut. Secțiunea prezintă cele două seturi de date ale capitolului, datele South German Credit și datele cardurilor de credit din Taiwan, și trăsăturile care fac speciale datele de nerambursare: istoria lungă a unei versiuni etichetate greșit a datelor germane, raritatea nerambursărilor și suprareprezentarea la care duce ea, și împărțirea într-un eșantion pentru estimare și un eșantion pentru test.

## Datele South German Credit {h=d3501585}

Setul principal de date al capitolului conține 1 000 de credite de consum acordate de o mare bancă regională din sudul Germaniei între 1973 și 1975 [@SouthGermanCredit2020; @Gromping2019]. Rezultatul este binar: contractul a fost respectat, un credit performant al unui bun-platnic, sau nu a fost respectat, un credit neperformant al unui rău-platnic, codificat nerambursare = 1. Eșantionul are 300 de credite neperformante și 700 performante. Fiecare credit este descris de 20 de variabile, printre care starea contului curent, durata creditului, istoricul de credit, scopul împrumutului, suma în mărci germane (DM), economiile, vechimea în muncă și vîrsta.

Eșantionul este stratificat: creditele neperformante sînt puternic suprareprezentate. Ponderea lor în eșantion este 30%, în timp ce rata creditelor neperformante din portofoliul băncii era de aproximativ 5%. Toți debitorii din date trecuseră de verificările băncii, deci solicitanții respinși de bancă lipsesc; [secțiunea 8](#sec-8) discută ce face această selecție unui model. Documentația fixează și un raport al costurilor: acceptarea unui credit neperformant costă de cinci ori mai mult decît respingerea unuia performant, cifră pe care capitolul o folosește pentru alegerea pragurilor.

Datele sînt publicate sub licența CC BY 4.0. Sînt suficient de mici pentru a fi inspectate rînd cu rînd, ceea ce le face un bun teren de exercițiu, și suficient de vechi pentru a avea o istorie proprie, spusă pe slide-ul următor.

## Erori de codificare în datele German Credit {h=3bb45a8d}

Timp de decenii, aceleași credite au circulat în UCI Machine Learning Repository sub numele „Statlog (German Credit)”, unul dintre cele mai folosite seturi de date din literatura de clasificare. @Gromping2019 a arătat că etichetele codurilor din acea versiune erau greșite. În versiunea veche, grupul „fără cont curent” era cel mai sigur: 394 de credite, dintre care 11,7% neperformante. În datele corectate, aceste 394 de credite sînt cele cu un sold de cel puțin 200 DM sau cu un cont de salariu, iar „fără cont curent” este grupul cel mai riscant, cu 49,3% credite neperformante.

Cifrele din fișier erau corecte; greșite erau doar etichetele atașate lor. Orice model estimat pe versiunea veche prognoza deci la fel de bine, iar orice interpretare a acestei variabile era inversată. Un model care „găsea” că clienții fără cont curent sînt cei mai siguri ar fi trecut orice test de acuratețe a prognozei.

Două verificări ar fi descoperit eroarea. Prima este compararea tabelelor de frecvențe cu sursa originală, aici cartea lui Fahrmeir și Hamerle în care datele au fost publicate prima dată. A doua este întrebarea dacă semnele au sens economic: este greu de explicat de ce clienții fără niciun cont ar rambursa mai des decît clienții cu sold negativ. Capitolul folosește peste tot datele corectate South German Credit.

## Rata creditelor neperformante după contul curent {h=d1c65fb8}

Graficul arată ponderea creditelor neperformante în fiecare dintre cele patru grupuri ale variabilei cont curent, cu media eșantionului de 30% ca linie întreruptă. Rata creditelor neperformante scade constant cu puterea contului: 49,3% pentru clienții fără cont curent, 39,0% pentru un sold negativ, 22,2% pentru un sold între 0 și 200 DM și 11,7% pentru un sold de cel puțin 200 DM sau un cont de salariu.

O singură variabilă separă deci bine solicitanții: grupul cel mai riscant intră în nerambursare de aproximativ patru ori mai des decît cel mai sigur. O bancă ce nu ar ști nimic altceva ar putea deja să refuze grupul cel mai riscant și să accepte grupul cel mai sigur. Secțiunile următoare transformă astfel de diferențe într-un model: [secțiunea 3](#sec-3) măsoară informația din fiecare variabilă, iar [secțiunea 4](#sec-4) combină mai multe variabile într-o regresie.

## Dezechilibrul claselor și suprareprezentarea {h=5f844342}

Nerambursările sînt rare: o bancă ale cărei credite ar intra în nerambursare în proporție de 30% nu ar supraviețui, iar portofoliile obișnuite au cîteva procente de credite neperformante. Un eșantion aleator din creditele băncii ar conține deci puțini rău-platnici și ar spune puțin despre ei. De aceea băncile suprareprezintă rău-platnicii (*oversampling*); în datele germane, ponderea lor este 30% în loc de aproximativ 5%.

Dezechilibrul claselor (*class imbalance*) creează paradoxul acurateței. Acuratețea, ponderea clasificărilor corecte, recompensează o regulă care ignoră clasa minoritară: regula „toți sînt bun-platnici” are dreptate pentru 70% din eșantionul german și pentru 77,9% din datele din Taiwan, fără să identifice niciun rău-platnic. O măsură care numără toate erorile împreună ascunde erorile care contează cel mai mult, de aceea [secțiunea 7](#sec-7) folosește măsuri care privesc separat rău-platnicii și bun-platnicii.

Al doilea set de date al capitolului sînt datele cardurilor de credit din Taiwan [@DefaultOfCreditCardClients2009; @YehLien2009]: 30 000 de deținători de card, cu rezultatul nerambursarea plății din octombrie 2005, care a avut loc la 22,1% dintre ei. Variabilele explicative sînt limita de credit, vîrsta, starea rambursării în fiecare lună din aprilie pînă în septembrie 2005, facturile lunare și plățile lunare. Cu de 30 de ori mai multe observații decît datele germane, datele din Taiwan servesc analizelor de calibrare și de echitate, unde grupurile mici ar fi prea zgomotoase.

## Eșantionul de estimare și eșantionul de test {h=92b845ac}

Datele germane se împart aleator în 700 de credite pentru estimare, eșantionul de estimare (*training sample*), și 300 pentru test, cu ponderea de 30% rău-platnici păstrată în ambele: o împărțire stratificată. Fiecare alegere făcută în construcția modelului, intervalele fiecărei variabile, Weight of Evidence al lor, selecția variabilelor și coeficienții, folosește doar eșantionul de estimare. Eșantionul de test se folosește o singură dată, la final, și joacă rolul viitorilor solicitanți pe care îi va întîlni modelul.

Regula protejează împotriva scurgerii de informație (*leakage*): informație din eșantionul de test care intră în model. Un caz tipic este alegerea intervalelor pe toate datele și abia apoi împărțirea; intervalele au „văzut” atunci rezultatele de test, iar rezultatul testului este prea optimist. Leakage este greu de observat, pentru că nimic nu se strică: modelul pare pur și simplu mai bun decît este.

O a doua limită este mărimea. Cu 300 de credite de test, dintre care 90 neperformante, o singură împărțire dă o estimare zgomotoasă a performanței; o altă împărțire aleatoare ar da o altă cifră. De aceea [secțiunea 7](#sec-7) adaugă validarea încrucișată repetată, care face media pe multe împărțiri.

## @end

Datele South German Credit conțin 1 000 de credite cu 30% rău-platnici, suprareprezentați față de o bancă a cărei rată a creditelor neperformante era de aproximativ 5%, și toate au trecut de verificările băncii. Etichetele corectate inversează concluzia a decenii de lucru pe versiunea veche: „fără cont curent” este grupul cel mai riscant, cu 49,3% rău-platnici. Pentru că nerambursările sînt rare, acuratețea este o măsură slabă, iar fiecare alegere de modelare se face pe un eșantion de estimare de 700 de credite și se judecă pe un eșantion de test de 300. [Secțiunea următoare](#sec-3) măsoară cît spune fiecare variabilă despre nerambursare.

# Weight of Evidence și Information Value

Înaintea oricărei regresii, un analist de credit întreabă cît spune fiecare variabilă despre nerambursare. Secțiunea introduce Weight of Evidence (WoE), care recodifică fiecare interval al unei variabile prin dovada pe care o aduce despre șansa de a fi bun-platnic, și Information Value (IV), care rezumă această dovadă într-un singur număr pe variabilă. Contul curent este lucrat de mînă, sînt enunțate regulile de construcție a intervalelor și sînt ordonate cele 20 de variabile ale datelor germane.

## Weight of Evidence (WoE) {h=b73d0587}

Împărțim o variabilă în intervale (*bins*) $i = 1, \dots, m$ și notăm cu $g_i$ și $b_i$ numărul de bun-platnici și de rău-platnici din intervalul $i$, cu totalurile $G$ și $B$. Weight of Evidence al intervalului $i$ compară ponderea tuturor bun-platnicilor care cad în interval cu ponderea tuturor rău-platnicilor:

$$ \mathrm{WoE}_i = \ln\frac{g_i/G}{b_i/B} = \ln\frac{g_i}{b_i} - \ln\frac{G}{B}. $$ {#eq:woe}

A doua formă dă sensul. $\ln(g_i/b_i)$ este logaritmul șansei de a fi bun-platnic în interval, $\ln(G/B)$ același logaritm al șansei în întregul eșantion, deci WoE măsoară cît de mult mută intervalul șansa față de medie. Un WoE pozitiv marchează un interval mai sigur decît media, un WoE negativ unul mai riscant, iar WoE = 0 un interval care nu aduce informație. Un interval fără bun-platnici sau fără rău-platnici ar da $\pm\infty$; remediul obișnuit este adăugarea valorii 0,5 la ambele frecvențe.

Information Value însumează dovada pe intervale, fiecare termen fiind ponderat cu diferența celor două ponderi:

$$ \mathrm{IV} = \sum_i \left(\frac{g_i}{G} - \frac{b_i}{B}\right)\mathrm{WoE}_i \;\ge\; 0. $$ {#eq:iv}

Fiecare termen este nenegativ, deoarece diferența și logaritmul raportului acelorași două ponderi au mereu același semn. IV este divergența Kullback–Leibler simetrică dintre distribuția bun-platnicilor și distribuția rău-platnicilor pe intervale: este zero cînd cele două distribuții coincid și crește pe măsură ce ele se separă. Regula practică a lui @Siddiqi2006 citește un IV sub 0,02 ca inutil, între 0,02 și 0,1 ca slab, între 0,1 și 0,3 ca mediu și peste 0,3 ca puternic.

## Exemplu rezolvat: WoE și IV pentru contul curent {h=b0f89e9e}

Tabelul aplică @eq:woe și @eq:iv celor patru grupuri ale contului curent, pe eșantionul complet de 700 de bun-platnici și 300 de rău-platnici. Luăm ultimul rînd, un sold de cel puțin 200 DM sau un cont de salariu: el conține 348 de bun-platnici și 46 de rău-platnici. Ponderea lui în totalul bun-platnicilor este $348/700 = 0{,}497$, iar ponderea în totalul rău-platnicilor este $46/300 = 0{,}153$. WoE este $\ln(0{,}497/0{,}153) = 1{,}176$, iar termenul IV este $(0{,}497 - 0{,}153) \times 1{,}176 = 0{,}404$.

Celelalte rînduri se calculează la fel. Clienții fără cont curent reprezintă 19,9% din bun-platnici, dar 45,0% din rău-platnici, deci WoE al lor este $-0{,}818$; un sold negativ dă $-0{,}401$, un sold între 0 și 200 DM dă $+0{,}405$. WoE scade monoton de la grupul cel mai sigur la cel mai riscant, ordinea la care se așteaptă bunul-simț economic.

Adunînd cei patru termeni IV obținem IV = 0,666, mult peste pragul de 0,3: contul curent este o variabilă puternică. Cea mai mare parte din această valoare provine din cele două grupuri extreme, care au ponderi mari și foarte inegale de bun-platnici și de rău-platnici; grupul mic cu sold între 0 și 200 DM contribuie doar cu 0,009.

## Discretizarea (binning): reguli practice {h=c64ffdf4}

Variabilele numerice sînt discretizate înainte de calculul WoE. Intervalele surprind efectele neliniare fără o formă funcțională, limitează influența valorilor extreme, deoarece o valoare extremă cade pur și simplu în ultimul interval, și transformă fiecare variabilă într-un tabel scurt din care se pot citi punctele unui solicitant. Pentru durata creditului, capitolul folosește cinci intervale: pînă la 12 luni, 13–18, 19–24, 25–36 și peste 36 de luni. Variabilele categoriale își folosesc categoriile.

Regulile standard provin din practica construirii scorecard-urilor [@Siddiqi2006; @ThomasCrookEdelman2017]. Fiecare interval ar trebui să conțină cel puțin aproximativ 5% din eșantion, pentru ca WoE al lui să nu fie zgomot, iar categoriile mici se unesc. WoE ar trebui să varieze între intervale într-un mod care are sens economic, adesea monoton; un tipar în zigzag reflectă de obicei zgomot, nu risc. Intervalele și WoE se estimează pe eșantionul de estimare și se aplică neschimbate noilor solicitanți.

Odată ce o variabilă este înlocuită cu WoE al intervalului ei, ea intră în modelul logit cu un singur coeficient. Scorecard-ul rămîne aditiv: fiecare variabilă contribuie cu un număr de puncte care depinde doar de propriul ei interval, ceea ce face scorul ușor de citit și de explicat unui solicitant.

## WoE pentru durata creditului {h=c18d52a2}

Graficul arată cele cinci intervale de durată pe eșantionul complet: barele dau WoE al fiecărui interval, iar linia rata creditelor neperformante, pe axa din dreapta. Creditele de pînă la un an sînt cele mai sigure, cu un WoE de 0,47 și 21% credite neperformante. Creditele mai lungi de trei ani sînt cele mai riscante: 52% dintre ele sînt neperformante, iar WoE al lor este $-0{,}92$. Intervalele între 13 și 24 de luni sînt aproape de medie, cu WoE apropiat de zero.

Tiparul este monoton și are o lectură economică: cu cît creditul este mai lung, cu atît situația debitorului are mai mult timp să se schimbe și cu atît, de regulă, suma în joc este mai mare. IV al duratei este 0,182, o variabilă medie pe scala lui @Siddiqi2006, cam un sfert din IV al contului curent.

## Information Value pentru cele 20 de variabile {h=a9af6160}

Graficul ordonează toate cele 20 de variabile după IV pe eșantionul complet, cu pragurile 0,02, 0,1 și 0,3 ca linii punctate. Două variabile domină: contul curent, cu IV = 0,666, și istoricul de credit, cu 0,293. La celălalt capăt, telefonul, vechimea la adresa actuală și numărul persoanelor în întreținere nu aduc nicio informație despre nerambursare.

IV servește drept filtru. Pe eșantionul de estimare, șase variabile au IV de cel puțin 0,1 și intră în scorecard-ul din [secțiunea 4](#s31): contul curent, istoricul de credit, scopul, economiile, durata și vechimea la angajator. IV calculate pe eșantionul de estimare diferă puțin de cele din grafic, care folosește toate cele 1 000 de credite.

IV are o limită pe care o regresie o înlătură: privește o singură variabilă odată. Două variabile puternice, dar corelate, nu își adună informația; mare parte din ce spune a doua este deja cunoscut din prima. Coeficienții modelului logit multivariat, din secțiunea următoare, măsoară ce adaugă fiecare variabilă date fiind celelalte.

## @end

Weight of Evidence recodifică fiecare interval al unei variabile ca diferența dintre logaritmul șansei de a fi bun-platnic în interval și logaritmul șansei în întregul eșantion, iar Information Value însumează această dovadă într-un singur număr pe variabilă. Contul curent are IV = 0,666, o variabilă puternică, cu WoE monoton; durata are 0,182; șase variabile trec de pragul de 0,1 pe eșantionul de estimare. Intervalele și WoE se fixează pe eșantionul de estimare. [Secțiunea următoare](#sec-4) combină variabilele într-o regresie logistică.

# Regresia logistică

Regresia logistică este modelul standard al scoringului de credit. Secțiunea îl construiește din șansa de nerambursare, îl estimează prin verosimilitate maximă și arată cum se citesc coeficienții ca modificări ale logaritmului șansei și ca rapoarte ale șanselor. Modelul este aplicat mai întîi cîtorva variabile brute, apoi celor șase variabile WoE care formează scorecard-ul, iar secțiunea se încheie cu corecția care transformă PD dintr-un set de date cu rău-platnici suprareprezentați în PD ale populației băncii.

## Șansa (odds) și logaritmul șansei {h=56b5dac1}

Fie $x = (x_1, \dots, x_k)^\top$ vectorul celor $k$ caracteristici ale unui solicitant (durata, suma, vîrsta, situația contului curent etc.) și $\beta = (\beta_1, \dots, \beta_k)^\top$ vectorul coeficienților lor, cîte unul pentru fiecare caracteristică; produsul $x^\top\beta = \beta_1 x_1 + \dots + \beta_k x_k$ este deci un singur număr. Un model liniar al probabilității, $P(Y = 1) = \beta_0 + x^\top\beta$, are un defect evident: nimic nu îi ține prognozele între 0 și 1, iar pentru solicitanții extremi dă probabilități negative sau mai mari decît unu. Remediul este modelarea unei transformări a lui $p$ care poate lua orice valoare reală, iar predictorul liniar acționează pe această scală.

Șansa (*odds*) de nerambursare este $p/(1 - p)$, cu valori în $(0, \infty)$; logaritmul șansei (*log-odds*), sau logit-ul, $\ln[p/(1 - p)]$, ia valori pe toată dreapta reală. Pentru $p = 0{,}20$, șansa este $0{,}2/0{,}8 = 0{,}25$, o nerambursare la patru credite rambursate, iar logaritmul șansei este $\ln 0{,}25 = -1{,}386$. Pentru $p = 0{,}5$, șansa este 1, iar logaritmul șansei 0. Logit-ul este simetric față de o jumătate: $\mathrm{logit}(1 - p) = -\mathrm{logit}(p)$, deci logaritmul șansei de a fi bun-platnic este minus logaritmul șansei de nerambursare.

Bancherii vorbesc adesea despre șansa bun:rău (*good:bad odds*), $(1 - p)/p$, numărul de credite bune la unul rău: pentru $p = 0{,}20$ ea este 4:1. Scorecard-urile din [secțiunea 6](#sec-6) sînt construite pe această scală.

## Modelul logit {h=bc958905}

Modelul logit egalează logaritmul șansei de nerambursare a solicitantului $i$ cu o funcție liniară de variabilele solicitantului [@Cox1958]:

$$ \ln\frac{p_i}{1 - p_i} = \beta_0 + x_i^\top\beta = \eta_i, \qquad p_i = P(Y_i = 1 \mid x_i) = \frac{1}{1 + e^{-\eta_i}}. $$ {#eq:logit}

Aici $x_i$ este vectorul caracteristicilor solicitantului $i$, $\beta$ vectorul coeficienților definit mai sus (același pentru toți solicitanții), $\beta_0$ termenul liber, iar $\eta_i$ predictorul liniar, un singur număr pentru fiecare solicitant, care este logaritmul șansei solicitantului $i$. Funcția logistică duce orice $\eta_i$ în $(0, 1)$, deci fiecare prognoză este o probabilitate.

Parametrii se estimează prin verosimilitate maximă, ca în [Capitolul 6](ch6_ro.html). Fiecare observație este o probă Bernoulli cu probabilitatea $p_i$, deci log-verosimilitatea este $\ell(\beta) = \sum_i [y_i\ln p_i + (1 - y_i)\ln(1 - p_i)]$. Condiția de ordinul întîi, $\sum_i (y_i - p_i)x_i = 0$, spune că reziduurile sînt ortogonale pe fiecare variabilă; ea nu are soluție explicită și se rezolvă iterativ prin metoda Newton–Raphson. Erorile standard provin din inversa matricei de informație $\sum_i p_i(1 - p_i)x_ix_i^\top$, iar statistica Wald $z = \hat\beta_j/\mathrm{se}(\hat\beta_j)$ testează dacă un coeficient este zero.

Același model servește pentru falimentul companiilor, începînd cu @Ohlson1980. Ruda lui cea mai apropiată, modelul probit, folosește funcția de repartiție a distribuției Normale, $p = \Phi(\eta)$, în locul funcției logistice. După rescalare, cele două curbe sînt atît de apropiate încît logit și probit dau aproape aceleași PD; logit-ul este preferat în scoringul de credit pentru că coeficienții lui au interpretarea prin șanse din slide-urile următoare.

## O curbă logistică: PD și durata {h=f51c171b}

Graficul estimează un logit cu o singură variabilă, durata în luni. Estimările sînt $\hat\beta_0 = -1{,}666$ și $\hat\beta_1 = 0{,}0375$ pe lună, cu o eroare standard de 0,0057 pentru pantă, deci $z \approx 6{,}6$, iar durata este clar semnificativă. Punctele sînt ratele observate ale creditelor neperformante în grupuri de durată, curba este PD estimată.

Pe scala logaritmului șansei, efectul duratei este constant: fiecare lună în plus adaugă 0,0375, fiecare an în plus $12 \times 0{,}0375 = 0{,}45$, ceea ce înmulțește șansa de nerambursare cu $e^{0{,}45} = 1{,}57$. Pe scala probabilității, efectul nu este constant. PD crește de la 22,9% pentru un credit pe 12 luni la 53,4% pentru un credit pe 48 de luni, iar curba este cea mai abruptă în jurul lui $p = 0{,}5$, unde funcția logistică se schimbă cel mai repede; pe intervalul datelor arată aproape dreaptă, începutul unei forme de S care s-ar aplatiza pentru durate foarte lungi.

## Interpretarea coeficienților {h=c4211bcc}

Un coeficient $\beta_j$ este modificarea logaritmului șansei de nerambursare cînd $x_j$ crește cu o unitate, celelalte variabile rămînînd fixe. Prin exponențiere, el devine o afirmație despre șansă: $e^{\beta_j}$, raportul șanselor (*odds ratio*), este factorul cu care se înmulțește șansa, iar $100(e^{\beta_j} - 1)\%$ este modificarea procentuală a șansei. Un interval de încredere de 95% pentru raportul șanselor se obține prin exponențierea intervalului coeficientului, $\exp(\hat\beta_j \pm 1{,}96\,\mathrm{se})$; el nu este simetric în jurul lui $e^{\hat\beta_j}$.

Efectul asupra probabilității însăși depinde de poziția solicitantului. Derivînd @eq:logit obținem

$$ \frac{\partial p}{\partial x_j} = p(1 - p)\,\beta_j, $$

care este maxim la $p = 0{,}5$ și mic pentru solicitanții foarte siguri sau foarte riscanți. Aceeași variabilă poate deci muta PD a unui solicitant mediu cu cîteva puncte procentuale, iar pe cea a unui solicitant excelent cu o fracțiune de punct.

Pentru o variabilă indicator (*dummy*), codificată 0 sau 1, $e^{\beta_j}$ compară șansa grupului cu șansa grupului de referință, cel codificat 0. Toate aceste lecturi sînt condiționate de celelalte variabile din model. Un coeficient este efectul lui $x_j$ cu variabilele incluse ținute fixe; dacă o variabilă relevantă corelată cu $x_j$ este omisă, o parte din efectul ei este atribuită lui $x_j$, iar coeficientul se schimbă.

## Un model logit mic pe eșantionul complet {h=659d8fcd}

Tabelul estimează un logit pe toate cele 1 000 de credite cu cinci variabile: durata în ani, vîrsta în decenii, suma în mii de DM și doi indicatori pentru grupurile de cont curent cele mai riscante, fără cont curent și sold negativ. Grupul de referință al celor doi indicatori este un sold nenegativ, care reunește un sold între 0 și 200 DM cu un sold de cel puțin 200 DM sau un cont de salariu. Unitățile sînt alese astfel încît o unitate să fie o schimbare cu sens.

Patru dintre cele cinci variabile sînt semnificative. Durata are un coeficient de 0,385 pe an ($z = 4{,}17$), vîrsta $-0{,}159$ pe deceniu ($z = -2{,}30$, p = 0,022), iar cei doi indicatori de cont 1,859 și 1,338, cu statisticile $z$ de 9,89 și 6,99. Suma, cu un coeficient de 0,029 la 1 000 DM și $z = 0{,}90$ (p = 0,368), nu adaugă nimic odată ce celelalte variabile sînt în model. Coloana rapoartelor șanselor exponențiază coeficienții și intervalele lor de încredere.

Modelul în ansamblu se testează față de modelul doar cu termenul liber. Log-verosimilitatea crește de la $-610{,}9$ la $-526{,}1$, deci statistica raportului de verosimilitate este $\mathrm{LR} = 2 \times (610{,}9 - 526{,}1) = 169{,}6$; comparată cu o distribuție $\chi^2$ cu 5 grade de libertate, cîte unul pentru fiecare variabilă adăugată, ea respinge cu o marjă foarte mare ipoteza că cele cinci variabile nu aduc informație.

## Rapoartele șanselor cu intervale de încredere {h=5b41f15b}

Graficul afișează rapoartele șanselor din tabel pe o scală logaritmică, astfel încît o dublare și o înjumătățire a șansei se află la aceeași distanță de linia verticală de la 1, care marchează lipsa efectului. Fiecare punct este $e^{\hat\beta}$, fiecare bară un interval de încredere de 95%. O bară care nu traversează valoarea 1 corespunde unui coeficient semnificativ la 5%.

Contul curent domină. Fără cont curent, șansa de nerambursare este de 6,42 ori cea a grupului de referință, cu un interval de la 4,44 la 9,28; un sold negativ o înmulțește cu 3,81. Un an în plus de durată crește șansa cu 47%, iar zece ani în plus de vîrstă o reduc cu aproximativ 15% (raportul șanselor 0,85).

Suma este singura bară care traversează valoarea 1: raportul ei al șanselor este 1,03, cu un interval de la 0,97 la 1,10. Aceasta nu înseamnă că suma nu are legătură cu nerambursarea; creditele mai mari tind să fie mai lungi, iar odată ce durata este în model, suma nu mai aduce informație separată. Este un exemplu al lecturii condiționate a coeficienților.

## Exemplu rezolvat: PD a unui solicitant {h=fdb1bb76}

Considerăm un solicitant care cere 4 000 DM pe 36 de luni, are 30 de ani și nu are cont curent. În unitățile modelului, durata este 3 ani, vîrsta 3 decenii, iar suma 4 mii de DM. Cu coeficienții rotunjiți din tabel, predictorul liniar este

$$ \eta = -2{,}101 + 0{,}385 \times 3 - 0{,}159 \times 3 + 0{,}029 \times 4 + 1{,}859 = 0{,}552. $$

PD este $1/(1 + e^{-0{,}552}) = 63{,}5\%$, iar șansa de nerambursare este $e^{0{,}552} = 1{,}74$. Același solicitant cu un sold nenegativ pierde termenul indicatorului, 1,859, deci $\eta = -1{,}307$, iar PD scade la 21,3%: doar contul curent triplează riscul prognozat. Un an în plus de durată schimbă PD cu aproximativ $p(1 - p)\beta = 0{,}635 \times 0{,}365 \times 0{,}385$, adică circa 8,9 puncte procentuale.

Aceste PD sînt PD ale eșantionului, în care 30% din credite sînt neperformante. În populația băncii, cu aproximativ 5% rău-platnici, același solicitant este mult mai puțin riscant. Corecția este subiectul slide-ului de după cel următor.

## Scorecard-ul logit pe variabile WoE {h=74b15155}

Modelul scorecard înlocuiește fiecare dintre cele șase variabile selectate cu WoE al intervalului ei, calculat pe eșantionul de estimare, și estimează un logit pentru nerambursare pe cele 700 de credite de estimare. Tabelul dă pentru fiecare variabilă IV pe eșantionul de estimare, coeficientul și eroarea standard. Contul curent păstrează cel mai mare IV, 0,591, urmat de istoricul de credit (0,215), scop (0,205), economii (0,166), durată (0,159) și vechimea la angajator (0,122).

Toți coeficienții sînt negativi. Acesta este semnul așteptat: WoE a fost definit cu bun-platnicii la numărător, deci un WoE mai mare marchează un interval mai sigur, iar un interval mai sigur ar trebui să reducă logaritmul șansei de nerambursare. Un coeficient de exact $-1$ ar însemna că variabila mută logaritmul șansei de nerambursare cu exact minus propriul WoE, ca și cum ar fi singura variabilă. Estimările se află între $-0{,}68$ și $-1{,}08$: o parte din dovada fiecărei variabile se suprapune cu cea a celorlalte, iar modelul o reduce corespunzător.

Log-verosimilitatea în eșantion este $-347{,}2$. În afara eșantionului, pe cele 300 de credite de test, scorecard-ul atinge un AUC de 0,801, o măsură a ordonării definită în [secțiunea 7](#sec-7): un credit neperformant ales la întîmplare primește o PD mai mare decît un credit performant ales la întîmplare în aproximativ 80% din perechi.

## Corecția pentru suprareprezentarea rău-platnicilor {h=a3fd2bcb}

Să presupunem că rău-platnicii și bun-platnicii din populație sînt eșantionați cu rate diferite, ca în datele germane. Un rezultat clasic pentru regresia logistică spune că se schimbă doar termenul liber: logaritmii șansei din populație și din eșantion diferă printr-o constantă,

$$ \ln\text{șansa}_{\text{pop}} = \ln\text{șansa}_{\text{eșantion}} + \ln\frac{\pi/(1 - \pi)}{\pi_s/(1 - \pi_s)}, $$ {#eq:prior}

unde $\pi$ este rata rău-platnicilor din populație, iar $\pi_s$ cea din eșantion. Pantele rămîn neschimbate, deci ordonarea solicitanților nu depinde de eșantionare; depinde doar nivelul PD.

Cu $\pi_s = 0{,}30$ și $\pi = 0{,}05$, deplasarea este $\ln(0{,}05/0{,}95) - \ln(0{,}30/0{,}70) = -2{,}944 - (-0{,}847) = -2{,}097$. O PD de 30% în eșantion devine 5%, cum trebuie, iar o PD de 50% devine 10,9%. Solicitantul din exemplul rezolvat are în populația băncii logaritmul șansei $0{,}552 - 2{,}097 = -1{,}545$, adică o PD de 17,6% în loc de 63,5%.

Corecția separă două utilizări ale modelului. Măsurile ordonării, precum AUC și Gini, nu se schimbă, deoarece toți logaritmii șansei se deplasează cu aceeași constantă. Pierderea așteptată din @eq:el și capitalul IRB din @eq:irb depind de nivelul PD, iar calculate din PD ale eșantionului ar fi de cîteva ori prea mari.

## @end

Modelul logit exprimă logaritmul șansei de nerambursare ca funcție liniară de variabile și se estimează prin verosimilitate maximă. Un coeficient este o modificare a logaritmului șansei; exponențiala lui este un raport al șanselor: fără cont curent, șansa de nerambursare este de 6,42 ori cea a grupului de referință, iar fiecare an în plus de durată o crește cu 47%. Scorecard-ul WoE cu șase variabile atinge un AUC de test de 0,801. PD dintr-un eșantion cu rău-platnici suprareprezentați au nevoie de corecția a priori a termenului liber, aici $-2{,}097$, înainte de a intra în orice pierdere așteptată sau calcul de capital. [Secțiunea următoare](#sec-5) deduce alternativa mai veche a modelului logit, discriminantul lui Fisher.

# Analiza discriminantă liniară

Discriminantul liniar al lui Fisher a fost prima metodă statistică de scoring de credit și rămîne metoda din spatele scorului Z al lui Altman. Secțiunea prezintă criteriul lui Fisher, arată că, pentru variabile cu distribuția Normală și aceeași matrice de covarianță în ambele grupuri, el dă un logaritm al șansei liniar în variabile, exact ca modelul logit, calculează de mînă direcția discriminantă pentru două variabile și compară cele două metode pe eșantionul de test.

## Funcția discriminantă a lui Fisher (1936) {h=f88e4cc7}

@Fisher1936 a întrebat care combinație liniară $w^\top x$ a mai multor măsurători separă cel mai bine două grupuri. El a lucrat cu măsurători ale florilor de iris din specii diferite; cinci ani mai tîrziu, Durand a aplicat aceeași metodă creditelor. Răspunsul depinde de ce înseamnă „cel mai bine”, iar alegerea lui Fisher a fost să compare distanța dintre mediile grupurilor cu dispersia din interiorul grupurilor, ambele măsurate de-a lungul direcției $w$:

$$ J(w) = \frac{[w^\top(m_1 - m_0)]^2}{w^\top S_W\, w}, $$

unde $m_0$ și $m_1$ sînt vectorii mediilor pentru bun-platnici și rău-platnici, iar $S_W$ este matricea de covarianță comună din interiorul grupurilor.

Numărătorul singur ar favoriza orice direcție în care mediile sînt departe una de alta, chiar dacă grupurile sînt foarte împrăștiate în acea direcție; numitorul penalizează o astfel de împrăștiere. Maximizarea lui $J(w)$ dă

$$ w \propto S_W^{-1}(m_1 - m_0), $$ {#eq:fisher}

dedusă în @HardleSimar2019, capitolul 14. Direcția este diferența mediilor, rotită și rescalată prin inversa covarianței, astfel încît variabilele cu dispersie mare sau puternic corelate cu altele primesc o pondere mai mică.

Niciun fel de ipoteză asupra distribuției nu intră în acest criteriu. Discriminantul lui Fisher este o proiecție care maximizează distanța dintre grupuri raportată la dispersia lor; probabilitățile apar abia cînd se presupune o distribuție, pe slide-ul următor.

## LDA ca model de probabilitate {h=6d8c0a16}

Analiza discriminantă liniară (LDA, *linear discriminant analysis*) dă direcției lui Fisher un sens probabilistic. Presupunem că, în fiecare grup, variabilele au distribuția Normală, $x \mid Y = k \sim N(m_k, \Sigma)$, cu aceeași matrice de covarianță $\Sigma$ în ambele grupuri, și că probabilitatea a priori a unui rău-platnic este $\pi_1 = P(Y = 1)$. Regula lui Bayes dă atunci logaritmul șansei a posteriori

$$ \ln\frac{P(Y = 1 \mid x)}{P(Y = 0 \mid x)} = c + w^\top x, \qquad w = \Sigma^{-1}(m_1 - m_0), $$

cu constanta $c = \ln[\pi_1/(1 - \pi_1)] - \tfrac12(m_1 + m_0)^\top w$. Termenii pătratici ai celor două densități Normale se reduc, pentru că matricele de covarianță sînt egale, iar ce rămîne este liniar în $x$, exact forma modelului logit. Probabilitatea a priori intră doar în constantă, ca în corecția pentru suprareprezentare din @eq:prior.

Cele două metode diferă prin modul în care estimează aceeași formă liniară. LDA introduce în formulă mediile de selecție și matricea de covarianță comună: o estimare cu formulă explicită, care folosește distribuția comună a lui $x$ și $Y$. Logit-ul maximizează verosimilitatea condiționată a lui $Y$ dat fiind $x$ și nu face nicio ipoteză asupra distribuției lui $x$.

Din costurile celor două erori, discutate în [secțiunea 7](#sec-7), rezultă o regulă de clasificare. Dacă $C_{FP}$ este costul respingerii unui bun-platnic, iar $C_{FN}$ costul acceptării unui rău-platnic, solicitantul este clasificat ca rău-platnic cînd $c + w^\top x > \ln(C_{FP}/C_{FN})$, adică atunci cînd șansa a posteriori de nerambursare depășește raportul costurilor.

## Exemplu rezolvat: direcția Fisher cu două variabile {h=d8723c1a}

Luăm două variabile, durata în luni și vîrsta în ani, pe eșantionul complet de 700 de bun-platnici și 300 de rău-platnici. Vectorii mediilor sînt $m_0 = (19{,}21; 36{,}22)$ pentru bun-platnici și $m_1 = (24{,}86; 33{,}96)$ pentru rău-platnici: creditele neperformante sînt în medie cu 5,65 luni mai lungi, iar rău-platnicii sînt cu 2,26 ani mai tineri. Matricea de covarianță comună din interiorul grupurilor este

$$ S_W = \begin{pmatrix} 138{,}8 & -2{,}46 \\ -2{,}46 & 127{,}9 \end{pmatrix}, $$

cu determinantul 17 756; cele două variabile sînt aproape necorelate în interiorul grupurilor.

Inversa unei matrice $2 \times 2$ schimbă între ele elementele diagonale, schimbă semnul celor din afara diagonalei și împarte la determinant. Aplicată lui $m_1 - m_0 = (5{,}65; -2{,}26)$, ea dă direcția lui Fisher $w = S_W^{-1}(m_1 - m_0) = (0{,}0404; -0{,}0169)$: riscul crește cu durata și scade cu vîrsta. Un logit cu aceleași două variabile are pantele $(0{,}0375; -0{,}0183)$.

Direcțiile sînt definite doar pînă la un factor de scală, așa că comparația se face prin raportul celor două ponderi: $-2{,}39$ pentru LDA și $-2{,}05$ pentru logit. În ambele metode, o lună de durată cîntărește cam cît doi ani de vîrstă, în sens invers. Cele două metode sînt de acord asupra formei scorului și diferă doar în detalii.

## Frontiera Fisher și frontiera logit {h=0da5709b}

Graficul reprezintă durata în funcție de vîrstă pentru fiecare credit, cu o mică perturbare aleatoare (*jitter*), pentru ca creditele cu valori identice să rămînă vizibile. Fiecare linie este mulțimea punctelor în care un model dă o PD de 30%, rata creditelor neperformante din eșantion: de o parte a liniei, modelul prognozează un risc mai mare decît media, de cealaltă parte unul mai mic. Linia LDA și linia logit au aproape aceeași pantă, adică raportul de pe slide-ul anterior văzut geometric.

Trăsătura mai izbitoare este suprapunerea celor doi nori: creditele performante și cele neperformante ocupă în mare parte aceeași regiune a planului. Durata și vîrsta aduc informație, dar mult prea puțină pentru a separa grupurile; un scorecard util are nevoie de mai multe variabile, precum contul curent și istoricul de credit. Alegerea dintre LDA și logit contează mult mai puțin decît alegerea variabilelor.

## Compararea modelelor logit și LDA {h=acb4bf43}

Tabelul pune cele două metode alături. Logit-ul modelează doar $P(Y \mid x)$; LDA modelează distribuția lui $x$ în fiecare grup, ca Normală cu o matrice $\Sigma$ comună. Logit-ul se estimează iterativ prin verosimilitate maximă, LDA cu formule explicite din medii și covarianțe. Cînd $x$ are într-adevăr distribuția Normală, ambele sînt consistente, iar LDA este mai eficientă, pentru că folosește informația din distribuția lui $x$. Cînd $x$ conține variabile indicator, valori WoE sau variabile asimetrice, ipoteza distribuției Normale nu mai este îndeplinită: ordonarea dată de LDA rămîne de obicei bună, dar PD ei pot fi deplasate, în timp ce logit-ul nu este afectat.

Pe eșantionul de test, ambele metode folosesc cele șase variabile WoE și ating același AUC, 0,801. Corelația dintre cele două scoruri exprimate ca logaritm al șansei este 0,999, iar testul lui @DeLongEtAl1988 pentru egalitatea AUC dă $z = 0{,}18$, cu un p-value de 0,86: nicio diferență detectabilă.

Pentru ordonarea solicitanților, cele două metode sînt aici practic identice. Pentru PD, logit-ul este preferat, deoarece nu cere variabile cu distribuția Normală; LDA rămîne utilă ca reper rapid, cu formulă explicită, și ca metoda scorului Z al lui Altman din [secțiunea 9](#sec-9).

## @end

Discriminantul lui Fisher alege direcția $w \propto S_W^{-1}(m_1 - m_0)$, care maximizează distanța dintre mediile grupurilor raportată la dispersia din interiorul grupurilor. Pentru variabile cu distribuția Normală și o matrice de covarianță comună, el dă un logaritm al șansei liniar în $x$, aceeași formă ca la logit, estimată altfel. Cu durata și vîrsta, ambele metode echivalează o lună de durată cu aproximativ doi ani de vîrstă; cu cele șase variabile WoE, ambele ating un AUC de test de 0,801, iar testul DeLong nu găsește nicio diferență. [Secțiunea următoare](#sec-6) transformă logaritmul șansei estimat într-un scorecard de puncte.

# Construirea scorecard-ului

Băncile arată rareori PD personalului sau solicitanților; ele arată scoruri. Secțiunea explică cum se rescalează logaritmul șansei dat de un model în puncte cu două convenții, un scor de bază la o șansă dată și numărul de puncte necesar pentru dublarea șansei, cum primește fiecare atribut al unui solicitant propriile puncte și cum sînt distribuite scorurile eșantionului de test.

## Scalarea unui scorecard {h=5656a49a}

Un scorecard (grilă de punctaj) este un tabel care dă puncte fiecărui atribut al unui solicitant, precum „fără cont curent” sau „durată de 13–18 luni”; scorul unui solicitant este suma punctelor atributelor lui [@Siddiqi2006]. Prin convenție, un scor mai mare înseamnă un risc mai mic. Scorul este o funcție liniară de logaritmul șansei bun:rău,

$$ \text{Scor} = \text{Offset} + \text{Factor} \times \ln\frac{1 - p}{p}, $$ {#eq:score}

iar două convenții fixează cele două constante.

Prima este numărul de puncte pentru dublarea șansei (PDO, *points to double the odds*): adăugarea a PDO puncte la un scor dublează șansa bun:rău. Cum dublarea șansei adaugă $\ln 2$ la logaritmul șansei, Factor este egal cu $\text{PDO}/\ln 2$. A doua fixează un scor de bază la o șansă dată, $\text{șansa}_0$; rezolvînd @eq:score pentru constantă obținem $\text{Offset} = \text{Baza} - \text{Factor} \times \ln(\text{șansa}_0)$.

Capitolul folosește PDO = 20 și un scor de 600 de puncte la șansa 50:1. Atunci $\text{Factor} = 20/\ln 2 = 28{,}85$ și $\text{Offset} = 600 - 28{,}85 \times \ln 50 = 600 - 28{,}85 \times 3{,}912 = 487{,}12$. Scala este arbitrară în sensul că orice altă pereche de convenții ordonează solicitanții exact la fel; ea este aleasă astfel încît scorurile să fie pozitive, de o mărime familiară și comparabile între scorecard-urile unei bănci.

## Exemplu rezolvat: scoruri și puncte {h=99800821}

De la PD la scor: un solicitant cu PD = 5% are șansa bun:rău 19:1, deci scorul lui este $487{,}12 + 28{,}85 \times \ln 19 = 487{,}12 + 28{,}85 \times 2{,}944 = 572{,}1$. Un solicitant cu PD = 30%, cu șansa 7:3, are scorul $487{,}12 + 28{,}85 \times 0{,}847 = 511{,}6$. De la scor la PD: 560 de puncte corespund unui logaritm al șansei de $(560 - 487{,}12)/28{,}85 = 2{,}526$, unei șanse bun:rău de 12,5:1 și unei PD de $1/13{,}5 = 7{,}4\%$.

Pentru a construi tabelul de puncte, scorul se împarte între cele $k = 6$ variabile ale logit-ului WoE, al cărui termen liber este $\beta_0$ și a cărui pantă pentru variabila $j$ este $\beta_j$. Atributul $i$ al variabilei $j$ primește

$$ \text{puncte}_{ij} = -\Big(\beta_j\,\mathrm{WoE}_{ij} + \frac{\beta_0}{k}\Big)\text{Factor} + \frac{\text{Offset}}{k}, $$

termenul liber și Offset fiind împărțite în mod egal între variabile. Semnul minus transformă logaritmul șansei de nerambursare, pe care îl modelează logit-ul, în logaritmul șansei bun:rău, pe care îl măsoară scorul. Pentru contul curent, cu WoE din eșantionul de estimare, cele patru grupuri primesc 66 de puncte (fără cont), 77 (sold negativ), 97 (0–200 DM) și 110 (cel puțin 200 DM sau cont de salariu). Scorul unui solicitant este suma a șase astfel de numere.

Corecția a priori din @eq:prior are și ea o formă simplă în puncte. Pentru o populație cu 5% rău-platnici, logaritmul șansei de nerambursare scade cu 2,097, deci logaritmul șansei bun:rău crește cu 2,097, iar fiecare scor crește cu $28{,}85 \times 2{,}097 = 60{,}5$ puncte. Tabelul de puncte rămîne altfel neschimbat.

## Scorurile eșantionului de test {h=afbe944b}

Graficul arată scorurile celor 300 de credite de test, separat pentru bun-platnici și rău-platnici. Scorurile merg de la 422 la 607 puncte; bun-platnicii au în medie 529 de puncte, rău-platnicii 492. Linia întreruptă este pragul (*cut-off*) la o PD de 1/6, dedus din costuri în [secțiunea 7](#sec-7), care corespunde unei șanse bun:rău de 5:1 și unui scor de 534 de puncte.

Cele două distribuții sînt deplasate una față de cealaltă, dar se suprapun mult. Orice prag face deci ambele tipuri de eroare: oriunde ar fi plasat, respinge unii bun-platnici și acceptă unii rău-platnici. Mutarea pragului schimbă o eroare pe cealaltă. Măsurile de validare din secțiunea următoare cuantifică această suprapunere și acest compromis, mai întîi la un singur prag și apoi pentru toate pragurile deodată.

## @end

Un scorecard rescalează logaritmul șansei bun:rău în puncte: cu PDO = 20 și 600 de puncte la șansa 50:1, Factor = 28,85 și Offset = 487,12. Fiecare atribut primește propriile puncte, iar scorul unui solicitant este suma lor; corectarea PD pentru o populație cu 5% rău-platnici adaugă 60,5 puncte fiecărui scor. Pe eșantionul de test, scorurile bun-platnicilor și ale rău-platnicilor se suprapun, deci orice prag face erori de ambele tipuri. [Secțiunea următoare](#sec-7) măsoară aceste erori.

# Validarea

Un scorecard este atît de bun cît este performanța lui pe solicitanți pe care nu i-a văzut. Secțiunea definește matricea de confuzie și ratele de eroare la un prag, alege pragul din costurile celor două erori și introduce măsurile care rezumă toate pragurile deodată: curba ROC și AUC, coeficientul Gini, curba CAP, raportul de acuratețe și statistica Kolmogorov–Smirnov. Trece apoi de la ordonare la calibrare, cu scorul Brier și testul Hosmer–Lemeshow, și se încheie cu validarea încrucișată, validarea out-of-time și un studiu comparativ de mari dimensiuni.

## Matricea de confuzie {h=078a4f8b}

Un scorecard devine o decizie printr-un prag $c$: solicitantul este respins dacă PD depășește $c$. În limbajul clasificării, un „pozitiv” este o prognoză de rău-platnic. Încrucișarea prognozei cu rezultatul efectiv dă patru celule, matricea de confuzie (*confusion matrix*): un adevărat pozitiv (TP, *true positive*) este un rău-platnic respins, un fals negativ (FN, *false negative*) un rău-platnic acceptat, un fals pozitiv (FP, *false positive*) un bun-platnic respins, iar un adevărat negativ (TN, *true negative*) un bun-platnic acceptat.

Celulele se rezumă prin rate. Rata adevăraților pozitivi (TPR, *true positive rate*), numită și sensibilitate, $\mathrm{TP}/(\mathrm{TP} + \mathrm{FN})$, este ponderea rău-platnicilor pe care regula îi identifică. Rata falșilor pozitivi (FPR, *false positive rate*), $\mathrm{FP}/(\mathrm{FP} + \mathrm{TN})$, este ponderea bun-platnicilor pe care îi respinge; unu minus FPR este specificitatea. Precizia, $\mathrm{TP}/(\mathrm{TP} + \mathrm{FP})$, este ponderea rău-platnicilor printre cei respinși, iar acuratețea, $(\mathrm{TP} + \mathrm{TN})/n$, ponderea deciziilor corecte.

Fiecare dintre aceste rate depinde de $c$. Un prag jos respinge mulți solicitanți și identifică majoritatea rău-platnicilor cu prețul multor bun-platnici respinși; un prag înalt face invers. Alegerea lui $c$ este o decizie de afaceri, iar măsurile care urmează fie îl aleg din costuri, fie rezumă toate pragurile deodată.

## Două praguri pe eșantionul de test {h=74d2d043}

Tabelul aplică două praguri celor 300 de credite de test, 90 neperformante și 210 performante. Cu pragul PD > 0,5, regula respinge 68 de solicitanți: identifică 43 de rău-platnici (TPR = 47,8%) și respinge 25 de bun-platnici (FPR = 11,9%), cu o acuratețe de 76,0%. Cu pragul PD > 1/6, respinge 194 de solicitanți: identifică 82 din cei 90 de rău-platnici (TPR = 91,1%), dar respinge și 112 bun-platnici (FPR = 53,3%), iar acuratețea scade la 60,0%.

Care regulă este mai bună depinde de costuri. Conform documentației datelor [@Gromping2019], acceptarea unui rău-platnic costă de cinci ori mai mult decît respingerea unui bun-platnic, deci costul pe solicitant este $(5\,\mathrm{FN} + 1\,\mathrm{FP})/n$. Pentru un solicitant cu PD $p$, respingerea costă în medie $(1 - p)\,C_{FP}$, iar acceptarea $p\,C_{FN}$. Pragul Bayes respinge ori de cîte ori acceptarea este mai scumpă:

$$ p\,C_{FN} > (1 - p)\,C_{FP} \iff p > \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{1 + 5} = \frac16. $$ {#eq:bayes}

Comparația inversează verdictul acurateței. Pragul de 1/6 este mai puțin precis, 60,0% față de 76,0%, dar costul lui pe solicitant este 0,507 față de 0,867, pentru că acceptă doar 8 rău-platnici în loc de 47. Cînd erorile au costuri diferite, acuratețea este criteriul greșit.

## Costul așteptat și alegerea pragului {h=a7a74233}

Graficul arată costul pe solicitant pe eșantionul de test (roșu) pentru fiecare prag între 0 și 1, împreună cu ponderea solicitanților respinși (albastru, axa din dreapta). Cele două linii întrerupte sînt politicile extreme: acceptarea tuturor costă 1,50 pe solicitant, cinci unități pentru fiecare dintre cei 90 de rău-platnici, împărțite la 300 de solicitanți; respingerea tuturor costă 0,70, o unitate pentru fiecare dintre cei 210 bun-platnici.

Curba costului este plată între PD de aproximativ 0,1 și 0,3: în acest interval, rău-platnicii acceptați în plus și bun-platnicii respinși în plus se compensează aproximativ. Pe aceste 300 de credite, minimul se află la 0,27, în regiunea plată care conține și pragul Bayes de 1/6. Pragul Bayes are avantajul că se alege din costuri înainte de a vedea datele de test; minimul empiric este ajustat pe aceste 300 de credite și s-ar muta pe un alt eșantion.

## Curba ROC și AUC {h=b2863c2c}

Curba ROC (*receiver operating characteristic*) reprezintă, pentru fiecare prag $c$, punctul $(\mathrm{FPR}(c), \mathrm{TPR}(c))$. Ea pornește din $(0, 0)$, unde nu este respins nimeni, și se termină în $(1, 1)$, unde sînt respinși toți. Un scor care ordonează solicitanții la întîmplare respinge bun-platnicii și rău-platnicii în aceeași proporție, deci curba lui este diagonala; un scor perfect atinge TPR = 1 cînd FPR este încă 0, iar curba lui merge de-a lungul marginilor din stînga și de sus.

Aria de sub curbă (AUC, *area under the curve*) rezumă curba într-un singur număr cu o interpretare directă [@HanleyMcNeil1982]:

$$ \mathrm{AUC} = P(s_{\text{rău}} > s_{\text{bun}}) + \tfrac12\,P(s_{\text{rău}} = s_{\text{bun}}), $$ {#eq:auc}

unde $s$ este scorul de risc, aici PD. AUC este probabilitatea ca un rău-platnic ales la întîmplare să primească o PD mai mare decît un bun-platnic ales la întîmplare, egalitățile fiind numărate cu o jumătate. Se poate calcula prin numărare: se formează toate perechile (rău, bun) și se ia ponderea celor ordonate corect. Această pondere este statistica Mann–Whitney împărțită la $n_1 n_0$, numărul de perechi. Un AUC de 0,5 înseamnă lipsa discriminării, iar 1 o discriminare perfectă.

Coeficientul Gini este o rescalare, $\mathrm{Gini} = 2\,\mathrm{AUC} - 1$, care merge de la 0 pentru un scor aleator la 1 pentru unul perfect. Pentru că AUC depinde doar de ordinea rău-platnicilor față de bun-platnici, el nu se schimbă cu rata rău-platnicilor din eșantion și nici cu o transformare monotonă a scorului: PD, logaritmul șansei și punctele dau același AUC. Eroarea standard a AUC și testele pentru compararea a două AUC pe aceiași solicitanți se datorează lui @DeLongEtAl1988.

## Curbele ROC pe eșantionul de test {h=7e975a37}

Graficul trasează două curbe ROC pe cele 300 de credite de test, care formează $90 \times 210 = 18\,900$ de perechi (rău, bun). Scorecard-ul logit WoE are AUC = 0,801, cu un interval de încredere de 95% de la 0,749 la 0,854, bazat pe o eroare standard DeLong de 0,027; Gini al lui este 0,603. Un scor bazat doar pe contul curent are AUC = 0,728.

Scorecard-ul plasează deci un rău-platnic ales la întîmplare deasupra unui bun-platnic ales la întîmplare în aproximativ 80% din perechi. Este avantajul lui față de contul curent singur real sau o chestiune de noroc pe 300 de credite? Testul DeLong, care ține seama de faptul că ambele scoruri sînt evaluate pe aceiași solicitanți, dă $z = 2{,}86$ și un p-value de 0,004: celelalte cinci variabile aduc o îmbunătățire semnificativă.

## Curba CAP, raportul de acuratețe și KS {h=113a7c16}

Profilul cumulativ al acurateței (CAP, *cumulative accuracy profile*) este ruda apropiată a curbei ROC, frecvent folosită în bănci. Se ordonează solicitanții de la cel mai riscant la cel mai sigur și se reprezintă, în funcție de ponderea $x$ a solicitanților considerați, ponderea tuturor rău-platnicilor găsiți printre primii $x\%$. O ordonare aleatoare dă diagonala. Un model perfect pune toți rău-platnicii primii, deci curba lui urcă la 1 în $x$ egal cu rata rău-platnicilor și rămîne acolo. Raportul de acuratețe (AR, *accuracy ratio*) compară modelul cu aceste două extreme: este aria dintre CAP-ul modelului și diagonală împărțită la aria dintre CAP-ul perfect și diagonală. Pentru un scor continuu, AR este egal cu coeficientul Gini.

Statistica Kolmogorov–Smirnov (KS) a unui scorecard este cea mai mare distanță verticală dintre distribuțiile scorurilor rău-platnicilor și ale bun-platnicilor,

$$ \mathrm{KS} = \max_s \big|F_{\text{rău}}(s) - F_{\text{bun}}(s)\big|, $$

unde $F_{\text{rău}}$ și $F_{\text{bun}}$ sînt funcțiile de repartiție empirice ale scorurilor în cele două grupuri; KS ia valori între 0 și 1. Pe curba ROC, KS este cea mai mare distanță verticală dintre curbă și diagonală, $\max_c (\mathrm{TPR} - \mathrm{FPR})$, iar scorul la care apare este o primă alegere naturală a pragului cînd costurile nu sînt cunoscute.

ROC, AUC, Gini, CAP, AR și KS măsoară toate discriminarea, adică ordonarea solicitanților. Niciuna nu spune dacă PD au nivelul corect: înmulțirea tuturor PD cu doi le lasă neschimbate. Calibrarea are nevoie de propriile măsuri, cu două slide-uri mai jos.

## CAP și KS pe eșantionul de test {h=60125d3a}

Graficul din stînga este curba CAP a scorecard-ului pe eșantionul de test, cel din dreapta cele două funcții de repartiție empirice ale scorurilor. Raportul de acuratețe este 0,602, același cu coeficientul Gini de 0,603 pînă la rotunjire, cum spune teoria. Citind CAP la $x = 30\%$: respingerea celor mai riscanți 30% dintre solicitanți identifică 60% din toți rău-platnicii (54 din cei 90), față de 30% pentru o ordonare aleatoare și 100% pentru una perfectă.

Statistica KS este 0,511 și este atinsă la 516 puncte: la acest scor, ponderea rău-platnicilor cu scoruri sub el depășește ponderea bun-platnicilor sub el cu 51 de puncte procentuale. La 516 puncte, scorecard-ul separă cel mai bine rău-platnicii de bun-platnici. Exprimat ca PD, punctul se află aproape de 27%, lîngă minimul costului găsit cu două slide-uri mai devreme.

## Calibrarea și scorul Brier {h=21922e09}

Un model este calibrat dacă, dintre solicitanții cu o PD apropiată de $p$, o pondere de aproximativ $p$ intră în nerambursare. Diagrama de calibrare (*reliability diagram*) verifică acest lucru vizual: se ordonează solicitanții după PD, se formează grupuri, de exemplu zece grupuri de mărime egală, și se reprezintă PD medie a fiecărui grup față de rata lui observată de nerambursare. Un model calibrat pune punctele pe diagonală. Testul lui @HosmerLemeshow1980 transformă diagrama într-o statistică,

$$ \mathrm{HL} = \sum_{g=1}^{G} \frac{(O_g - n_g\bar p_g)^2}{n_g\bar p_g(1 - \bar p_g)} \approx \chi^2(G - 2), $$

unde $G$ este numărul de grupuri, $n_g$ mărimea grupului $g$, $O_g$ nerambursările lui observate, iar $\bar p_g$ PD lui medie. Fiecare termen compară nerambursările observate cu cele așteptate într-un grup; o valoare mare semnalează o calibrare slabă.

Scorul Brier al lui @Brier1950 este eroarea pătratică medie a PD ca prognoze ale rezultatelor 0/1:

$$ \mathrm{BS} = \frac1n\sum_{i=1}^{n} (p_i - y_i)^2, $$ {#eq:brier}

o valoare mai mică fiind mai bună. Referința lui naturală este o PD constantă egală cu rata rău-platnicilor din eșantionul de estimare, prognoza unui model care nu știe nimic despre solicitanți; cîștigul (*skill score*) $1 - \mathrm{BS}/\mathrm{BS}_{\text{ref}}$ este ponderea din eroarea de referință pe care o elimină modelul.

Pe eșantionul de test german, scorecard-ul are BS = 0,162 față de 0,210 pentru PD constantă de 30%, un cîștig de 23%. Statistica Hosmer–Lemeshow cu zece grupuri este 6,520, cu un p-value de 0,589: nicio dovadă de calibrare greșită. Spre deosebire de AUC, care vede doar ordonarea, scorul Brier recompensează atît ordonarea, cît și calibrarea: un model cu o ordonare perfectă, dar cu PD la un nivel greșit, are totuși un scor Brier mare.

## Calibrarea pe datele din Taiwan {h=48ee8ea7}

Graficul este diagrama de calibrare a unui logit estimat pe 70% din cei 30 000 de deținători de card din Taiwan și evaluat pe un eșantion de test de 9 000. Clienții de test sînt ordonați după PD în zece grupuri de cîte 900; fiecare punct compară PD medie a unui grup cu rata lui observată de nerambursare, iar barele sînt intervale de 95% pentru rata observată. Modelul ordonează bine, cu un AUC de test de 0,777 (0,774 pe eșantionul de estimare), iar scorul Brier de 0,136 față de 0,172 pentru PD constantă înseamnă un cîștig de 21%.

Punctele sînt aproape de diagonală, totuși statistica Hosmer–Lemeshow este 40,660, cu un p-value sub 0,001: testul respinge calibrarea. Cu 900 de clienți pe grup, testul are puterea de a detecta abateri mici. Cea mai mare este în grupul cel mai sigur, unde PD medie este 7,8%, iar rata observată 5,9%.

Lecția privește semnificația și mărimea. Un test pe 9 000 de clienți detectează abateri care s-ar putea să nu conteze pentru decizie; un test pe 300 de credite poate rata abateri care contează. Calibrarea trebuie judecată după mărimea abaterilor, citită pe diagramă și exprimată în puncte procentuale de PD, nu doar după p-value.

## Validarea în afara eșantionului și validarea încrucișată {h=4aeba336}

Performanța măsurată pe datele de estimare este optimistă: modelul s-a adaptat atît la structura acestor date, cît și la zgomotul lor, fenomen numit supraajustare (*overfitting*). Un eșantion de test elimină această distorsiune, dar un singur eșantion de test este zgomotos. AUC de test de 0,801 pe 300 de credite are o eroare standard de 0,027, deci o altă împărțire aleatoare ar putea da ușor o valoare cu cîteva sutimi mai mare sau mai mică.

Validarea încrucișată în $k$ părți (*k-fold cross-validation*) folosește fiecare observație pentru test. Datele se împart în $k = 5$ părți; modelul se estimează pe patru părți și se testează pe a cincea, iar rolurile se rotesc pînă cînd fiecare parte a servit o dată ca set de test. Întregul ciclu se repetă de 20 de ori cu împărțiri aleatoare diferite, ceea ce dă 100 de AUC de test. Întreaga procedură de modelare se repetă în interiorul fiecărui set de estimare: intervalele, WoE, selecția după IV și coeficienții. Repetarea doar a ultimului pas ar scurge informație din părțile de test în intervale.

Rezultatele se raportează ca media și dispersia celor 100 de AUC de test, împreună cu diferența dintre AUC de estimare și de test, care măsoară supraajustarea. O schemă diferită, validarea out-of-time, testează pe cereri care sosesc mai tîrziu decît cele folosite la estimare; ea este subiectul slide-ului de după cel următor.

## AUC de estimare și de test în validarea încrucișată repetată {h=7d7e254e}

Graficul arată cele 100 de AUC de estimare (puncte) și cele 100 de AUC de test (cruci) pentru patru modele, cu mediile lor ca linii negre. Scorecard-ul logit WoE are un AUC mediu de estimare de 0,804 și un AUC mediu de test de 0,764, cu o deviație standard de 0,032 între părțile de test. LDA pe aceleași variabile WoE dă 0,804 și 0,765. Logit-ul mic din [secțiunea 4](#s28), cu cinci variabile brute, dă 0,749 și 0,744, iar contul curent singur 0,708.

Diferența dintre AUC de estimare și de test măsoară supraajustarea. Scorecard-ul pierde aproximativ 0,04 din AUC în afara eșantionului, pentru că intervalele, valorile WoE și selecția variabilelor se adaptează toate la datele de estimare; logit-ul cu cinci variabile, cu puțini parametri și fără discretizare dictată de date, nu pierde aproape nimic. Scorecard-ul rămîne cel mai bun model în afara eșantionului, dar cu mai puțin decît sugerează AUC de estimare.

Media din validarea încrucișată pune și împărțirea unică din slide-urile anterioare în perspectivă. AUC de test al acesteia, 0,801, se află peste media de 0,764 pe 100 de părți de test: acea împărțire a fost norocoasă. Un singur eșantion de test de 300 de credite este o verificare rezonabilă, dar cifra care trebuie raportată este cea din validarea încrucișată.

## Validarea out-of-time și deriva populației {h=dcb58ce7}

Un scorecard este folosit ani de zile pe solicitanți care sosesc după perioada de estimare. În acest timp economia se schimbă, produsele băncii se schimbă și populația solicitanților se schimbă, astfel încît relația dintre variabile și nerambursare derivă (*drift*). O împărțire aleatoare nu poate detecta acest lucru, pentru că eșantioanele ei de estimare și de test provin din aceeași perioadă. Validarea out-of-time estimează modelul pe cereri mai vechi, de exemplu din 2019–2021, și îl testează pe cereri ulterioare, de exemplu din 2022–2023, comparînd AUC și calibrarea de la un an la altul.

Odată ce scorecard-ul este în uz, monitorizarea compară distribuția scorurilor și a fiecărei variabile cu perioada de estimare. Diagnosticul ghidează răspunsul. Dacă scorurile se deplasează, dar ordonarea rămîne stabilă, nivelul PD este greșit și este suficientă o recalibrare a termenului liber, ca în corecția a priori din @eq:prior; dacă ordonarea se deteriorează, modelul trebuie reconstruit.

Niciunul dintre seturile de date ale capitolului nu conține data cererilor, deci un test out-of-time nu este posibil aici, iar împărțirile aleatoare sînt cea mai bună schemă disponibilă. O schemă out-of-time într-un alt context, prognoza activelor cripto „zombie”, este prezentată în @BedowskaSojkaWojcikPele2026.

## Studiu de caz: compararea metodelor de scoring {h=dff063ac}

Cum se compară logit-ul cu numeroșii clasificatori dezvoltați de atunci? @LessmannEtAl2015 au comparat 41 de clasificatori, cu șase măsuri de acuratețe, pe opt seturi de date de credit de retail, printre care datele germane de credit din repository-ul UCI. Studiul actualizează, după zece ani, comparația publicată de Baesens și coautorii săi în 2003.

Trei rezultate ies în evidență. Mai mulți clasificatori prognozează semnificativ mai bine decît regresia logistică, standardul industriei, iar ansamblurile eterogene (*heterogeneous ensembles*), care combină modele de tipuri diferite, au cea mai bună performanță. Scorecard-urile mai precise pot aduce cîștiguri financiare importante unui creditor. Măsurile uzuale de acuratețe dau semnale asemănătoare, iar autorii recomandă random forests ca reper (*benchmark*) față de care trebuie testată orice metodă nouă.

Studiul schimbă ce se consideră progres: a învinge doar logit-ul nu mai este o dovadă că o metodă nouă este utilă. Pe cele 1 000 de credite germane ale acestui capitol, logit și LDA diferă cu mai puțin de o eroare standard a AUC, iar pentru a separa metodele bune este nevoie de seturi de date mult mai mari. [Capitolul 13](ch13_ro.html) adaugă la comparație arborii de clasificare și rețelele neuronale.

## @end

La un prag dat, matricea de confuzie dă ratele adevăraților și falșilor pozitivi; cu un cost de 5 pentru un rău-platnic acceptat și de 1 pentru un bun-platnic respins, pragul Bayes PD > 1/6 reduce costul pe solicitant de la 0,867 la 0,507, cu prețul unei acurateți mai mici. Pe toate pragurile, scorecard-ul atinge un AUC de test de 0,801 (Gini 0,603, AR 0,602, KS 0,511) și un cîștig Brier de 23%, fără dovezi de calibrare greșită. Validarea încrucișată repetată dă un AUC de test mai fiabil, de 0,764, iar comparațiile pe date mari arată că ansamblurile întrec modelul logit. [Secțiunea următoare](#sec-8) trece la două probleme pe care niciun eșantion de test nu le poate arăta: solicitanții care au fost respinși și echitatea deciziilor.

# Reject inference și echitatea

Validarea din secțiunea anterioară măsoară cît de bine prognozează un scor rezultatele solicitanților din date. Două întrebări rămîn în afara acestui cadru. Datele conțin doar solicitanți care au fost acceptați, în timp ce modelul se folosește pe toți solicitanții; iar un model care prognozează bine poate totuși să trateze inegal grupuri de solicitanți. Secțiunea discută reject inference, face un experiment pe datele din Taiwan, definește trei criterii de echitate și le citește pe aceleași date.

## Reject inference {h=f75fc317}

O bancă observă rezultatul unui credit doar dacă l-a acordat. Modelul de scoring se aplică însă fiecărui solicitant care trece „pragul ușii” (*through the door*). Datele de estimare sînt deci un eșantion selectat de scorul anterior al băncii: solicitanții cei mai riscanți au fost respinși, iar rezultatele lor lipsesc. Debitorii din South German Credit trecuseră toți de verificările băncii [@Gromping2019], deci datele nu spun nimic direct despre solicitanții pe care banca i-a refuzat.

Reject inference este numele metodelor care încearcă să corecteze aceste rezultate lipsă [@BanasikCrookThomas2003; @ThomasCrookEdelman2017]. Reponderarea (*reweighting*) dă o pondere mai mare solicitanților acceptați care seamănă cu cei respinși, astfel încît eșantionul de estimare să semene mai mult cu populația solicitanților. Augmentarea (*augmentation*) și parcelarea (*parcelling*) atribuie solicitanților respinși rezultate imputate dintr-un model și le adaugă la datele de estimare.

Toate aceste metode se sprijină pe ipoteze pe care datele nu le pot verifica, deoarece rezultatele de care au nevoie sînt exact cele care lipsesc. Sursa de informație cea mai fiabilă este de aceea un experiment: un mic eșantion aleator de solicitanți acceptați indiferent de scorul lor, ale căror rezultate arată cum s-ar fi comportat cei respinși. Experimentul costă banca pierderile din aceste credite și îi oferă o imagine nedistorsionată a întregii populații.

## Un experiment de reject inference pe datele din Taiwan {h=19c5fa25}

Datele din Taiwan permit un experiment controlat, pentru că rezultatul este cunoscut pentru fiecare client. Un „scor vechi”, construit din întîrzierea la plată din septembrie și din limita de credit, acceptă cei mai buni 70% dintre clienții de estimare, 15 085 de persoane. Rata de nerambursare a celor acceptați este 13,2%, a celor respinși 44,9%. Printre acceptați, nimeni nu a întîrziat în septembrie; cea mai puternică variabilă a datelor ia o singură valoare în eșantionul de estimare și dispare din orice model estimat pe el.

Toate modelele sînt apoi evaluate pe un eșantion de test care conține toți solicitanții, acceptați și respinși. Modelul estimat doar pe clienții acceptați atinge un AUC de 0,736, modelul estimat pe toți solicitanții 0,777, iar scorul vechi însuși 0,743. Pentru clienții de test respinși, rata observată de nerambursare este 46,1%; modelul estimat pe acceptați le dă o PD medie de 29,6%, modelul complet 44,1%.

Experimentul arată ambele costuri ale selecției. Estimat doar pe solicitanții acceptați, noul model ordonează mai slab decît scorul vechi pe care trebuia să îl înlocuiască și subestimează riscul solicitanților respinși cu aproximativ 36% (o PD medie de 29,6% față de o rată observată de 46,1%). O bancă ce l-ar folosi pentru a accepta o parte dintre solicitanții respinși anterior și-ar asuma un risc mult mai mare decît se aștepta.

## Echitatea (fairness) în scoringul de credit {h=b1198ba2}

Deciziile de creditare sînt reglementate. În Statele Unite, [Equal Credit Opportunity Act](https://www.ftc.gov/legal-library/browse/statutes/equal-credit-opportunity-act) interzice discriminarea în creditare după sex, stare civilă, rasă, religie, origine națională sau vîrstă. În Uniunea Europeană, [AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) clasifică scoringul de credit al persoanelor drept o utilizare cu risc ridicat a inteligenței artificiale, care cere documentare și supraveghere umană.

Excluderea variabilei protejate din model nu este suficientă, pentru că alte variabile îi pot ține locul: venitul, tipul de slujbă sau locul de reședință pot fi corelate cu sexul, vîrsta sau originea. Echitatea trebuie deci măsurată pe decizii. Trei criterii sînt uzuale. Paritatea demografică (*demographic parity*) cere rate egale de aprobare între grupuri. Egalitatea de șanse (*equal opportunity*), în sensul lui @HardtPriceSrebro2016, cere rate egale de aprobare printre bun-platnicii fiecărui grup, astfel încît o persoană care ar rambursa să aibă aceeași șansă de a primi un credit, oricare ar fi grupul ei. Calibrarea în fiecare grup cere ca aceeași PD să însemne același risc în fiecare grup.

Aceste criterii intră în conflict. Cînd ratele de nerambursare diferă între grupuri, un model calibrat dă în medie PD diferite grupurilor și le aprobă deci în proporții diferite, ceea ce încalcă paritatea demografică; cu excepția cazurilor degenerate, precum un predictor perfect, criteriile nu pot fi îndeplinite toate simultan. La care dintre ele se renunță nu este o întrebare statistică, ci o chestiune de politică pentru bănci și autoritățile de reglementare. Sarcina statisticianului este să le măsoare pe toate și să raporteze compromisul.

## Aprobarea și nerambursarea pe grupuri: Taiwan {h=54cf1a1b}

Graficul citește cele trei criterii pe eșantionul de test din Taiwan. Sexul și vîrsta nu sînt variabile ale modelului, iar pragul aprobă cei 75% dintre solicitanți cu PD cea mai mică. Femeile intră în nerambursare cu o rată de 20,5%, iar 76,6% dintre ele sînt aprobate; bărbații intră în nerambursare cu 24,7%, iar 72,5% sînt aprobați. Printre bun-platnici sînt aprobate 85,1% dintre femei și 83,2% dintre bărbați.

Modelul este calibrat în fiecare grup: PD medie a fiecărui grup este apropiată de rata lui observată de nerambursare. Un model calibrat trebuie atunci să aprobe mai rar grupurile care intră mai des în nerambursare, aici bărbații și grupele de vîrstă cele mai tinere și cele mai în vîrstă. Paritatea demografică este încălcată cu aproximativ patru puncte procentuale între femei și bărbați; egalitatea de șanse cu aproximativ două puncte.

Graficul ilustrează astfel în cifre conflictul de pe slide-ul anterior. Egalizarea ratelor de aprobare ar cere praguri diferite după sex, ceea ce ar strica calibrarea în fiecare grup și ar folosi explicit variabila protejată.

## Studiu de caz: învățarea automată și inegalitatea accesului la credit {h=d3aa93dc}

@FusterEtAl2022 au studiat creditele ipotecare din SUA și au prognozat nerambursarea atît cu un logit tradițional, cît și cu modele de învățare automată (*machine learning*). Întrebarea lor merge dincolo de acuratețe: cînd un creditor trece la modelul mai precis, cine cîștigă și cine pierde?

Modelele flexibile prognozează mai bine nerambursarea. Cîștigurile sînt însă inegal distribuite: debitorii de culoare și cei hispanici au mai puține șanse să beneficieze de noua tehnologie. Într-un model de echilibru al pieței creditelor ipotecare, disparitatea dobînzilor între grupuri și în interiorul grupurilor crește, mai ales pentru că flexibilitatea mai mare a modelelor le permite să exploateze combinații de variabile corelate cu apartenența la grup.

Lecția pentru scoringul de credit este că acuratețea și echitatea trebuie măsurate împreună. Un AUC mai bun este un argument necesar pentru un model nou, dar nu unul suficient.

## @end

Datele unei bănci conțin doar solicitanți acceptați. Pe datele din Taiwan, un model estimat pe solicitanții acceptați ordonează mai slab decît scorul vechi (AUC 0,736 față de 0,743) și subestimează riscul celor respinși cu aproximativ 36%; un mic eșantion aleator de solicitanți acceptați necondiționat este corecția cea mai fiabilă. Echitatea se măsoară pe decizii, prin paritatea demografică, egalitatea de șanse și calibrarea în fiecare grup, iar cînd ratele de nerambursare diferă, aceste criterii intră în conflict. Un model mai precis poate mări disparitățile. [Secțiunea următoare](#sec-9) trece de la nerambursarea persoanelor fizice la cea a companiilor.

# Nerambursarea companiilor: Altman și Merton

Și companiile intră în nerambursare, iar pentru companiile listate există două surse de informație pe care scoringul persoanelor fizice nu le are: situațiile financiare publicate și prețurile de piață. Secțiunea prezintă cele două modele clasice construite pe ele, scorul Z al lui Altman, un scor discriminant din indicatori contabili, și distanța pînă la nerambursare a lui Merton, un model structural în care capitalurile proprii sînt o opțiune pe activele companiei. Ea se încheie cu un test empiric al modelului Merton.

## Scorul Z al lui Altman (1968) {h=a08030e6}

@Altman1968 a studiat 66 de companii industriale americane: 33 care au intrat în faliment între 1946 și 1965 și 33 de companii pereche care au supraviețuit. El a aplicat analiza discriminantă a lui Fisher din [secțiunea 5](#sec-5) la cinci indicatori contabili și a obținut scorul

$$ Z = 1{,}2X_1 + 1{,}4X_2 + 3{,}3X_3 + 0{,}6X_4 + 1{,}0X_5. $$ {#eq:z}

$X_1$ este capitalul de lucru raportat la activele totale (TA, *total assets*), o măsură a lichidității; $X_2$ profitul reinvestit raportat la TA, profiturile acumulate de-a lungul vieții companiei; $X_3$ profitul înainte de dobînzi și impozite (EBIT, *earnings before interest and taxes*) raportat la TA, profitabilitatea curentă a activelor; $X_4$ valoarea de piață a capitalurilor proprii raportată la valoarea contabilă a datoriilor, rezerva pe care o asigură acționarii; iar $X_5$ cifra de afaceri raportată la TA, rotația activelor.

Altman a împărțit scala în zone: sub 1,81, dificultate financiară; între 1,81 și 2,99, o zonă gri a companiilor incerte; peste 2,99, zona sigură. Cu un an înainte de faliment, aproximativ 95% din companiile din eșantion erau clasificate corect. Acuratețea scădea repede pentru orizonturi mai lungi: situațiile contabile dezvăluie mai degrabă dificultăți deja în curs decît dificultăți care urmează să vină.

## Exemplu rezolvat: un scor Z {h=1ede39f5}

Considerăm o companie ipotetică cu $X_1 = 0{,}15$, $X_2 = 0{,}20$, $X_3 = 0{,}08$, $X_4 = 0{,}90$ și $X_5 = 1{,}10$. Cei cinci termeni din @eq:z sînt $1{,}2 \times 0{,}15 = 0{,}180$, $1{,}4 \times 0{,}20 = 0{,}280$, $3{,}3 \times 0{,}08 = 0{,}264$, $0{,}6 \times 0{,}90 = 0{,}540$ și $1{,}0 \times 1{,}10 = 1{,}100$, iar suma lor este $Z = 2{,}364$: compania se află în zona gri.

Care indicator ar muta compania în zona sigură? Profitabilitatea are cea mai mare pondere. Un EBIT/TA de 0,27 în loc de 0,08 adaugă $3{,}3 \times 0{,}19 \approx 0{,}63$ la scor și aduce $Z$ la aproximativ 2,99, marginea zonei sigure. Aceeași creștere a scorului prin cifra de afaceri ar cere o creștere a lui $X_5$ cu 0,63, mai mult de jumătate din valoarea actuală.

Modelul are limite clare. Ponderile lui au fost estimate pe 66 de companii industriale americane din 1946–1965; companiile din alte sectoare, din alte țări și din alte decenii au nevoie de un model reestimat. Z este un scor discriminant, nu o probabilitate; modelele ulterioare dau direct PD, cu un logit la @Ohlson1980 și cu modele de hazard (*hazard models*), care folosesc întreaga istorie a fiecărei companii, la @Shumway2001.

## Distanța pînă la nerambursare a lui Merton {h=f3bf61ea}

@Merton1974 privește compania prin bilanțul ei. Valoarea $V_t$ a activelor companiei urmează o mișcare browniană geometrică (GBM, *geometric Brownian motion*), ca prețurile acțiunilor din [Capitolul 4](ch4_ro.html), cu rata de creștere (*drift*) $\mu$ și volatilitatea $\sigma_V$. Compania are o datorie cu valoarea nominală $D$, scadentă la momentul $T$. Dacă la scadență activele valorează mai puțin decît datoria, $V_T < D$, compania intră în nerambursare, iar creditorii preiau activele; altfel, acționarii plătesc datoria și păstrează restul. Capitalurile proprii sînt deci o opțiune call pe active, cu prețul de exercitare $D$.

Sub GBM, $\ln V_T \sim N\big(\ln V_0 + (\mu - \sigma_V^2/2)T,\ \sigma_V^2T\big)$, deci probabilitatea de nerambursare este probabilitatea ca această variabilă cu distribuția Normală să scadă sub $\ln D$:

$$ \mathrm{DD} = \frac{\ln(V_0/D) + (\mu - \sigma_V^2/2)T}{\sigma_V\sqrt{T}}, \qquad \mathrm{PD} = \Phi(-\mathrm{DD}). $$ {#eq:dd}

Distanța pînă la nerambursare (DD, *distance to default*) este numărul de abateri standard cu care valoarea logaritmică așteptată a activelor la $T$ depășește logaritmul datoriei. Ea combină îndatorarea (*leverage*), prin $\ln(V_0/D)$, cu riscul, prin $\sigma_V$: o companie poate fi sigură cu o îndatorare mare dacă activele ei sînt stabile și riscantă cu o îndatorare mică dacă ele sînt volatile.

Valoarea activelor $V$ și volatilitatea lor $\sigma_V$ nu sînt observate. Ele se obțin din ce se observă, valoarea de piață a capitalurilor proprii și volatilitatea prețului acțiunii, prin inversarea relațiilor de evaluare a opțiunilor, cum arată slide-ul următor.

## Exemplu rezolvat: distanța pînă la nerambursare {h=9a3f39ed}

Luăm $V_0 = 100$, $D = 70$, $\sigma_V = 25\%$, $\mu = 5\%$ și $T = 1$ an. Termenul de îndatorare este $\ln(100/70) = 0{,}3567$, iar termenul de drift $\mu - \sigma_V^2/2 = 0{,}05 - 0{,}03125 = 0{,}01875$, deci numărătorul din @eq:dd este 0,3754, iar $\mathrm{DD} = 0{,}3754/0{,}25 = 1{,}502$. PD este $\Phi(-1{,}502) = 6{,}66\%$.

În practică, datele de intrare vin din piață. Să presupunem că valoarea capitalurilor proprii este $E = 40$, volatilitatea lor este $\sigma_E = 50\%$, datoria este 70, iar rata fără risc este $r = 3\%$. Două ecuații leagă mărimile observate de cele neobservate: prețul Black–Scholes al capitalurilor proprii ca opțiune call, $E = V\Phi(d_1) - De^{-rT}\Phi(d_2)$, și relația dintre cele două volatilități, $\sigma_E E = \Phi(d_1)\sigma_V V$, unde $d_1$ și $d_2$ sînt termenii Black–Scholes din [Capitolul 4](ch4_ro.html), cu $V$ în locul prețului acțiunii. Rezolvarea lor dă $V = 107{,}9$ și $\sigma_V = 18{,}6\%$. Cu rata fără risc ca drift, distanța pînă la nerambursare este atunci $d_2 = 2{,}39$, iar PD este 0,84%.

Cele două părți ale exemplului măsoară același tip de companie în două moduri, iar a doua este cea folosită în practică. O PD obținută din piață reacționează zilnic la prețul acțiunii și la volatilitatea lui; un scor contabil precum Z se schimbă o dată pe an, cînd se publică noile situații financiare.

## Merton: traiectoriile activelor și PD {h=0c6ee4d3}

Panoul din stînga al graficului simulează 40 de traiectorii ale valorii activelor pe un an, cu $V_0 = 100$, $\mu = 5\%$ și $\sigma_V = 25\%$, datele de intrare din prima parte a exemplului rezolvat. Traiectoriile roșii se termină sub datoria de 70: aceste companii intră în nerambursare la scadență. Ponderea lor în simulare este o imagine aproximativă a PD de 6,66% calculate din @eq:dd.

Panoul din dreapta reprezintă PD pe un an în funcție de gradul de îndatorare $D/V$ pentru trei volatilități ale activelor. PD este aproape zero la îndatorare mică și crește abrupt între o îndatorare de 70% și una de 90%. Cu cît volatilitatea este mai mare, cu atît creșterea începe mai devreme: o companie cu active volatile devine riscantă la o îndatorare care ar fi inofensivă pentru o companie cu active stabile. Acesta este conținutul distanței pînă la nerambursare: îndatorarea raportată la volatilitate.

## Studiu de caz: acuratețea predictivă a modelului Merton {h=a07e0f27}

@BharathShumway2008 au testat modelul Merton pe companii americane. Ei au comparat DD Merton, obținut prin rezolvarea ecuațiilor de evaluare a opțiunilor, cu un DD „naiv”, care are aceeași formă funcțională, dar înlocuiește soluția cu aproximări simple ale valorii activelor și ale volatilității lor. Ambele au fost folosite în modele de hazard ale nerambursării și în prognoze în afara eșantionului și comparate cu informația din spread-urile swap-urilor pe riscul de nerambursare (CDS, *credit default swap*) și din randamentele obligațiunilor.

DD naiv funcționează puțin mai bine decît DD Merton în afara eșantionului. Alte variabile, precum randamentele trecute ale acțiunilor companiei, adaugă informație oricăruia dintre cele două DD, deci distanța pînă la nerambursare nu este o statistică suficientă pentru probabilitatea de nerambursare. Ceea ce face DD util este forma lui funcțională, îndatorarea raportată la volatilitate, nu soluția modelului.

Lecția depășește modelul Merton: teoria dă o variabilă bună, iar statistica decide cum este folosită. Același lucru este valabil pentru scorul Z, ai cărui indicatori rămîn date de intrare utile mult după ce ponderile lui originale s-au învechit.

## @end

Scorul Z al lui Altman este discriminantul lui Fisher aplicat la cinci indicatori contabili: compania ipotetică din exemplu are scorul 2,364, în zona gri dintre 1,81 și 2,99. Modelul Merton tratează capitalurile proprii ca pe o opțiune call pe active și măsoară distanța pînă la nerambursare, îndatorarea raportată la volatilitatea activelor; compania din exemplu are DD = 1,502 și PD = 6,66%. Empiric, valoarea DD stă mai mult în forma lui funcțională decît în soluția modelului, iar PD obținute din piață se actualizează zilnic. [Secțiunea următoare](#sec-10) privește o întrebare de cercetare deschisă a capitolului.

# AI pentru descoperire științifică

Ultima secțiune a fiecărui capitol ia o întrebare de cercetare deschisă din domeniul lui și arată cum pot instrumentele AI să accelereze un studiu al ei și ce verificări rămîn sarcina cercetătorului. Pentru scoringul de credit, întrebarea este dacă scorurile flexibile de învățare automată îmbunătățesc prognoza nerambursării suficient pentru a justifica ce costă ele ca transparență și echitate.

## O întrebare de cercetare deschisă {h=9ec7ff50}

Întrebarea acestui capitol este: îmbunătățesc scorurile flexibile de învățare automată prognoza nerambursării suficient pentru a justifica costul lor în transparență și echitate? Rezultatele capitolului o încadrează. Logit-ul WoE și LDA ating pe datele germane un AUC de test din validarea încrucișată de aproximativ 0,764. Comparațiile de mari dimensiuni găsesc cîștiguri semnificative pentru ansamblurile de clasificatori [@LessmannEtAl2015]; cîștigurile pot fi inegale între grupurile de debitori [@FusterEtAl2022]; iar mașinile cu vectori suport (*support vector machines*) au fost aplicate prognozei nerambursării de @ChenHardleMoro2011.

Întrebarea rămîne deschisă din două motive. Cîștigurile depind de setul de date, de măsura performanței și de populație, astfel încît un rezultat obținut pe un portofoliu nu se transferă la altul. Iar rezultatele solicitanților respinși nu se văd niciodată, deci fiecare comparație se face pe un eșantion selectat, problema din [secțiunea 8](#sec-8).

Instrumentele AI pot accelera un astfel de studiu, de la căutarea literaturii pînă la prima versiune a codului, dar rezultatele lor trebuie verificate ca orice alt rezultat [@WangEtAl2023].

## Utilizări posibile ale instrumentelor AI {h=573e04d9}

Trei părți ale studiului pot fi accelerate. Pentru literatură, un asistent AI poate întocmi o primă listă a studiilor comparative de scoring de credit și a studiilor despre echitatea creditării, împreună cu seturile de date pe care le folosesc; fiecare intrare trebuie apoi găsită și citită. Pentru cod, el poate scrie o primă versiune a unei comparații prin validare încrucișată a logit-ului WoE cu gradient boosting, un ansamblu de arbori de clasificare, pe datele din Taiwan. Pentru robustețe, el poate extinde comparația la alte praguri, la alte măsuri, precum scorul Brier și costul, și la AUC și calibrarea calculate separat pentru fiecare grup.

Exemplul de prompt de pe slide este specific în felul care contează pentru această întrebare. El cere validare încrucișată stratificată repetată în 5 părți, cere ca intervalele WoE să fie recalculate în interiorul fiecărei părți, ceea ce previne leakage, și cere scorul Brier și ratele de aprobare ale bărbaților și femeilor la o rată de aprobare fixă, astfel încît rezultatul să vorbească despre calibrare și echitate, nu doar despre ordonare.

## Verificări necesare {h=d239ad65}

Verificările decurg din capcanele întîlnite în acest capitol. Codificarea rezultatului trebuie să fie nerambursare = 1; o codificare inversă schimbă semnul fiecărui coeficient și fiecare curbă fără niciun mesaj de eroare. Convenția de semn pentru WoE trebuie cunoscută: capitolul folosește $\ln(\text{bun-platnici}/\text{rău-platnici})$, în timp ce unele surse folosesc $\ln(\text{rău-platnici}/\text{bun-platnici})$, ceea ce inversează semnele coeficienților scorecard-ului.

Scurgerea de informație trebuie să lipsească: intervalele, WoE și selecția variabilelor trebuie calculate în interiorul fiecărei părți de estimare, niciodată pe toate datele. Măsurile trebuie să fie cele corecte: AUC nu este acuratețea, iar $\mathrm{Gini} = 2\,\mathrm{AUC} - 1$, nu $\mathrm{AUC} - 0{,}5$. PD estimate pe un set de date cu rău-platnici suprareprezentați au nevoie de corecția a priori din @eq:prior înainte de a calcula din ele orice pierdere așteptată sau capital.

În sfîrșit, fiecare lucrare și fiecare set de date citat trebuie să existe și să spună ceea ce i se atribuie; un DOI care duce la lucrarea corectă este cea mai simplă verificare, iar asistenții AI sînt cunoscuți pentru faptul că inventează referințe plauzibile.

## Idee de proiect {h=f59c31d7}

Proiectul pune o versiune concretă a întrebării deschise: pe datele cardurilor de credit din Taiwan [@DefaultOfCreditCardClients2009], publicate sub licența CC BY 4.0, cît adaugă gradient boosting la AUC față de logit-ul WoE și ce se întîmplă cu ratele de aprobare pe sexe și pe grupe de vîrstă?

Pașii urmează capitolul. Mai întîi se construiesc intervalele WoE și se calculează IV al fiecărei variabile, apoi se estimează un logit WoE și un model gradient boosting cu setările implicite. În al doilea rînd, cele două modele se compară prin validare încrucișată stratificată repetată, raportînd AUC cu intervale DeLong, scorul Brier și calibrarea. În al treilea rînd, se fixează un prag și se măsoară echitatea prin paritatea demografică și egalitatea de șanse pentru grupuri de sex și de vîrstă.

Livrabilul este scurt: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele, în special despre ce înseamnă pentru concluzii absența solicitanților respinși. Orice utilizare a instrumentelor AI se declară, împreună cu o listă a erorilor acestora care au fost găsite și corectate.

## @end

Întrebarea deschisă este dacă scorurile de învățare automată îmbunătățesc logit-ul WoE suficient pentru a justifica costul lor în transparență și echitate; comparațiile arată că ele ordonează adesea mai bine, iar studiile pe creditele ipotecare arată că cîștigurile pot fi inegale. Instrumentele AI pot întocmi lista literaturii, codul și verificările de robustețe, în timp ce codificarea rezultatului, semnul WoE, leakage, definițiile măsurilor, corecția a priori și referințele rămîn de verificat de cercetător.

# Rezumat

Rezumatul adună ideile și formulele capitolului și se încheie cu trei întrebări de autoevaluare.

## Idei de reținut {h=dd6dc41b}

Pierderea așteptată, $\mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD}$, este un cost pe care banca îl include în dobîndă; capitalul IRB acoperă pierderea neașteptată, pierderea de peste EL într-un an prost care apare o dată la 1 000 de ani în modelul cu un factor. Ambele pornesc de la PD, pe care o estimează un model de scoring.

Weight of Evidence transformă fiecare interval al unei variabile într-o dovadă pe scala logaritmului șansei, iar Information Value ordonează variabilele. În modelul logit, $e^{\beta}$ este un raport al șanselor; LDA a lui Fisher dă același logaritm liniar al șansei pentru variabile cu distribuția Normală și o matrice de covarianță comună, iar pe datele germane ordonează solicitanții la fel de bine. Un scorecard rescalează logaritmul șansei în puncte, Scor = Offset + Factor × ln(șansa bun:rău), aici cu 20 de puncte pentru fiecare dublare a șansei.

Validarea măsoară discriminarea (AUC, Gini, KS) și calibrarea (scorul Brier, diagrama de calibrare) în afara eșantionului, iar pragul se alege din costurile celor două erori, nu din acuratețe. Solicitanții respinși și grupurile protejate cer verificări explicite, pentru că niciun eșantion de test nu îi arată. Pentru companii, scorul Z al lui Altman și distanța pînă la nerambursare a lui Merton adaugă informație contabilă și de piață.

## Formule de reținut {h=0763f7a8}

Tabelul adună formulele capitolului. Pierderea așteptată este @eq:el; Weight of Evidence și Information Value sînt @eq:woe și @eq:iv; modelul logit este @eq:logit; direcția lui Fisher este @eq:fisher; corecția pentru suprareprezentare adaugă termenului liber constanta din @eq:prior; scorecard-ul este @eq:score, cu Factor = PDO/ln 2.

Pentru validare, AUC este probabilitatea @eq:auc ca un rău-platnic ales la întîmplare să aibă un scor de risc mai mare decît un bun-platnic ales la întîmplare, Gini este $2\,\mathrm{AUC} - 1$, KS este cea mai mare distanță dintre distribuțiile scorurilor rău-platnicilor și ale bun-platnicilor, iar scorul Brier @eq:brier este eroarea pătratică medie a PD. Pragul Bayes @eq:bayes respinge cînd PD depășește $C_{FP}/(C_{FP} + C_{FN})$. Distanța pînă la nerambursare a lui Merton și PD corespunzătoare sînt date de @eq:dd.

## Autoevaluare {h=07b6e7eb}

**Întrebare pentru sală**: un model are o acuratețe de 70% pe eșantionul de test South German Credit. Este bun? **Răspuns**: nu neapărat. Regula „toți sînt bun-platnici” atinge deja 70%, pentru că 30% din credite sînt neperformante. Modelul trebuie judecat după AUC, după TPR și FPR la pragul ales și după costul erorilor lui.

**Întrebare pentru sală**: coeficientul unei variabile indicator într-un logit este 0,69. Ce înseamnă? **Răspuns**: pentru grupul codificat cu 1, șansa de nerambursare se înmulțește cu $e^{0{,}69} \approx 2$ față de grupul de referință, celelalte variabile fiind fixe.

**Întrebare pentru sală**: schimbă corecția PD pentru suprareprezentare valoarea AUC? **Răspuns**: nu. Corecția deplasează toți logaritmii șansei cu aceeași constantă, deci ordonarea solicitanților și, odată cu ea, AUC nu se schimbă; se schimbă doar nivelul PD.

[Capitolul 13](ch13_ro.html) continuă cu învățarea automată: arbori de clasificare, ansambluri și rețele neuronale pentru același tip de problemă.

## @end

Un scor de credit este un model de probabilitate pentru nerambursare, judecat după ordonare și după nivel. Capitolul l-a construit din variabile WoE și un model logit, l-a scalat în puncte, l-a validat în afara eșantionului și a verificat ce nu pot arăta datele: solicitanții respinși și echitatea între grupuri. Aceleași idei statistice se extind la companii prin indicatori contabili și prețuri de piață.
