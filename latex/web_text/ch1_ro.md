<!--
Textul de curs al versiunii web, capitolul 1 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie -> Autor (an), [@Cheie] -> (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 1 --hashes
-->

## @intro

@text
Un nivel de preț, luat singur, spune puțin despre o investiție. O acțiune cotată la cîțiva lei și un indice cotat la mii de puncte nu se pot compara, iar niciunul dintre aceste numere nu arată cît a cîștigat investitorul, cît de mult a fluctuat valoarea pe parcurs sau cît de adîncă a fost cea mai mare scădere. Capitolul transformă datele brute de piață în cifre care permit astfel de comparații: randamente, măsuri ale riscului și drawdown-uri, împreună cu eroarea lor de estimare.

Capitolul lucrează peste tot cu date zilnice reale pentru indicele BET al Bursei de Valori București (BVB), S&P 500, DAX, Bitcoin și cinci acțiuni de la BVB, pe perioada de comparație 2015–2026. Datele arată deja temele principale ale cursului. În această perioadă, 100 de unități investite în Bitcoin au ajuns la 29 475, față de 379 în S&P 500, dar drawdown-ul maxim al Bitcoin a fost de −83,4%. Randamentele zilnice au cozi groase (*heavy tails*): dincolo de trei abateri standard, S&P 500 are de 5,2 ori mai multe zile decît prevede distribuția Normală. Raportul Sharpe (*Sharpe ratio*) anualizat al S&P 500, 0,72, are o eroare standard de 0,29, așa că majoritatea clasamentelor activelor după astfel de indicatori nu sînt semnificative statistic.

Înaintea oricărei statistici, capitolul se ocupă de calitatea datelor: alegerea sursei primare, ajustarea prețurilor pentru dividende și splituri și erorile pe care le conțin seriile reale.

@chart sfm_ch1_growth
Valoarea a 100 de unități investite în prima zi din 2015 în S&P 500, DAX, BET și Bitcoin, pe o axă verticală logaritmică: distanțe verticale egale corespund unor variații procentuale egale.

@goals
- **Alegeți** o sursă de date de încredere și verificați o serie de prețuri înainte de utilizare.
- **Calculați** randamente simple și logaritmice și agregați-le în timp și între active.
- **Anualizați** mediile și volatilitățile cu frecvența reală a datelor.
- **Descrieți** o serie de randamente prin momente, cuantile și valori extreme.
- **Calculați și interpretați** CAGR, rapoartele Sharpe și Sortino, drawdown-ul maxim, raportul Calmar și volatility drag.
- **Comparați** diferența dintre doi indicatori cu eroarea lor de estimare.

Varianta pentru studenți a seminarului este disponibilă ca [PDF](../RO/Seminarii/seminar1_date_randamente_indicatori_ro.pdf).

@plan
- Date financiare: surse și calitate | Tipuri și frecvențe de date, surse, prețuri ajustate, erori în datele reale și trei erori sistematice.
- De la prețuri la randamente | De ce statistica lucrează cu randamente, staționaritatea slabă și testul ADF.
- Randamente simple și logaritmice | Definiții, diferența dintre cele două randamente în zilele extreme și cînd se folosește fiecare.
- Randamente pe mai multe perioade și ale portofoliilor | Agregarea în timp, averea cumulată și randamentul portofoliilor.
- Anualizarea | Media și volatilitatea anuală, CAGR și precizia mediei estimate.
- Statisticile descriptive ale randamentelor | Poziție, dispersie, asimetrie și boltire, cozi groase și volatilitate variabilă în timp.
- Indicatori de performanță | Sharpe, Sortino, drawdown, Calmar și volatility drag, cu eroarea lor de estimare.
- AI pentru descoperire științifică | O întrebare deschisă despre persistența rapoartelor Sharpe și verificările de care are nevoie un studiu asistat de AI.
- Rezumat | Formulele capitolului și întrebări de autoevaluare.

# Date financiare: surse și calitate

Fiecare rezultat al cursului pornește de la un fișier de date, iar fiecare eroare din acel fișier trece în rezultate. Secțiunea descrie ce sînt datele financiare, cît de des se înregistrează, de unde provin și ce date folosește cursul. Arată apoi, pe serii reale, cele două operații care decid cel mai des dacă o serie de prețuri poate fi folosită: ajustarea pentru evenimentele corporative și verificarea cu o a doua sursă. Se încheie cu trei erori sistematice care nu lasă nicio urmă vizibilă în date, dar fac rezultatele să pară mai bune decît sînt.

## Tipuri de date financiare {h=b78b0f20}

Datele financiare se împart în patru grupe. **Datele de piață** sînt informațiile produse de un loc de tranzacționare: prețurile tranzacțiilor, cele mai bune cotații bid și ask, volumul tranzacționat și serii derivate din ele, ca nivelurile indicilor, cursurile de schimb, ratele dobînzii și prețurile cripto. Cel mai bun bid este prețul cel mai mare la care cineva este dispus să cumpere într-un moment dat, cel mai bun ask prețul cel mai mic la care cineva este dispus să vîndă; diferența dintre ele, spread-ul, este costul unei cumpărări urmate imediat de o vînzare.

**Evenimentele corporative** (*corporate actions*) formează o grupă separată, pentru că schimbă prețul cotat fără să schimbe valoarea deținerii investitorului. Cînd o companie plătește un dividend în numerar, prețul acțiunii scade în ziua ex-dividend cu aproximativ suma plătită, dar acționarul primește banii. La un split, fiecare acțiune veche devine mai multe acțiuni noi, la un preț proporțional mai mic; la o consolidare (*reverse split*), mai multe acțiuni vechi devin una singură. Acțiunile gratuite funcționează ca un split. O serie de prețuri cotate care ignoră aceste evenimente conține salturi pe care niciun investitor nu le-a trăit.

**Datele fundamentale și macroeconomice**, ca bilanțurile, profiturile, inflația, produsul intern brut (PIB) și dobînzile de politică monetară, descriu firmele și economia din spatele prețurilor. **Datele alternative** includ știri, rețele sociale, imagini din satelit și tranzacții pe blockchain. Cursul lucrează în principal cu date zilnice de piață.

## De la banda telegrafică la fluxurile de date {h=0fa8f280}

Primul flux de prețuri în timp real a fost telegraful bursier (*stock ticker*), folosit din 1867. Aparatul primea prin telegraf prețurile tranzacțiilor și le tipărea pe o bandă de hîrtie: prețul ultimei tranzacții, la cîteva minute după ce avusese loc. Investitorii aflați departe de sala bursei puteau urmări piața aproape în timp real.

Astăzi bursele publică fluxuri electronice, pe care furnizorii de date (*data vendors*) le colectează și le redistribuie. Fiecare cotație și fiecare tranzacție se înregistrează cu un marcaj de timp la microsecundă. Majoritatea studiilor statistice, inclusiv acest curs, nu folosesc direct aceste înregistrări, ci rezumatele lor zilnice, barele OHLCV (deschidere, maxim, minim, închidere, volum; *open, high, low, close, volume*), prezentate [mai jos](#s11).

Banda de cotații a supraviețuit în două forme. Benzi mobile cu prețuri și știri defilează încă pe fațadele clădirilor, ca pe clădirea Reuters din New York din imaginea slide-ului. Tot de la bandă provin și simbolurile scurte care identifică titlurile, tickerele: la BVB, TLV este Banca Transilvania, SNP este OMV Petrom, iar H2O este Hidroelectrica.

## Frecvența datelor {h=197d7ccc}

Cel mai fin nivel al datelor sînt **datele tick**: fiecare tranzacție și fiecare cotație. O piață lichidă produce milioane de înregistrări pe zi. Prețurile tick conțin zgomot de microstructură; sursa cea mai cunoscută este bid–ask bounce: tranzacțiile succesive se execută alternativ la bid și la ask, astfel că prețul tranzacției sare înainte și înapoi chiar dacă valoarea activului nu se schimbă. **Barele intrazilnice**, de 1, 5 sau 60 de minute, rezumă tick-urile și sînt datele de intrare ale volatilității realizate, subiectul [Capitolului 8](ch8_ro.html).

**Datele zilnice**, cu un rînd pe zi de tranzacționare, sînt standardul acestui curs și al majorității studiilor academice. Bursele funcționează aproximativ 250 de zile pe an, piețele cripto 365 de zile pe an, inclusiv în weekenduri și sărbători. **Datele săptămînale, lunare și anuale** au mai puține observații, dar sînt mai aproape de orizontul unui investitor pe termen lung.

Frecvența contează pentru fiecare cifră anualizată din capitol. Regula este anualizarea cu numărul **real** de observații pe an al fiecărei serii, aproximativ 252 pentru o bursă și 365 pentru Bitcoin, niciodată cu un singur factor pentru toate activele. Consecințele unui factor greșit apar în secțiunea despre [anualizare](#s38).

## Surse de date {h=f422cd57}

Tabelul de pe slide enumeră șase tipuri de surse. Bursele (BVB, Deutsche Börse, Bursa din New York, NYSE) publică prețuri oficiale, volume și niveluri ale indicilor, pe site și prin fluxuri contra cost. Furnizorii de date, ca Bloomberg, LSEG (London Stock Exchange Group, fost Refinitiv) și EODHD (EOD Historical Data; EOD înseamnă sfîrșit de zi, *end of day*), adună prețuri de pe multe piețe, calculează prețuri ajustate și adaugă date fundamentale, pe bază de abonament. Băncile centrale, ca Banca Națională a României (BNR), Banca Centrală Europeană (BCE) și Rezerva Federală, publică gratuit cursuri de referință și rate ale dobînzii.

Bazele de date statistice, ca FRED (Federal Reserve Economic Data, a Rezervei Federale din St. Louis) și Eurostat, oferă serii macroeconomice și randamente ale titlurilor de stat. Bazele de date academice, ca CRSP (Center for Research in Security Prices, Universitatea din Chicago) și biblioteca de date a lui Kenneth French, oferă randamente curățate, inclusiv ale firmelor delistate, și randamente ale factorilor, cu licență universitară sau gratuit. Bursele și agregatoarele cripto furnizează prețuri non-stop și date on-chain.

Furnizorul colectează și curăță datele, dar nu este originea cifrei. Cînd două surse diferă, referința este **sursa primară** a cifrei: bursa pentru un preț de închidere, banca centrală pentru un curs de referință. [Exemplul EUR/RON](#s16) arată de ce contează această regulă.

## Datele acestui curs {h=c795a418}

Cursul folosește date zilnice de piață de la EODHD, păstrate ca un instantaneu fix, astfel încît fiecare rezultat să poată fi reprodus. Colecția are 91 de serii: indici, fonduri tranzacționate la bursă (ETF, *exchange-traded funds*, fonduri listate care urmăresc un indice), acțiuni de la BVB, din SUA și din Europa, active cripto, cursuri de schimb și randamente ale obligațiunilor. Fiecare serie are aceleași coloane: data, deschidere, maxim, minim, închidere, închidere ajustată și volum.

Cursul EUR/RON este excepția. Cursul folosește cursul oficial de referință al BNR, pentru că seria EODHD pentru acest curs conține cotații eronate; slide-ul [Erori în date: două serii EUR/RON](#s16) măsoară efectul lor.

Trei convenții se aplică peste tot. Indicii, cursurile de schimb și activele cripto folosesc prețul de închidere (*close*); acțiunile și ETF-urile folosesc prețul de închidere ajustat (*adjusted close*), pentru ca dividendele și spliturile să fie luate în calcul. Cotațiile de weekend și închiderile repetate în zilele libere se elimină, cu excepția activelor cripto, care se tranzacționează în fiecare zi. Fiecare serie se analizează pe propriul calendar, astfel că BET și S&P 500 își păstrează fiecare zilele de tranzacționare. Perioada de comparație a capitolului este 1 ianuarie 2015 – 18 septembrie 2026.

## Barele zilnice: OHLCV {h=fa693028}

O bară zilnică înregistrează cinci numere pentru ziua $t$: prețul de deschidere $O_t$, prețul maxim $H_t$, prețul minim $L_t$, prețul de închidere $C_t$ și volumul $V_t$, numărul de acțiuni tranzacționate. Prin construcție, deschiderea și închiderea se află între minim și maxim, $L_t \le O_t, C_t \le H_t$. Amplitudinea $H_t - L_t$ măsoară cît s-a mișcat prețul în timpul zilei și stă la baza estimatorilor de volatilitate bazați pe amplitudine din [Capitolul 8](ch8_ro.html).

Tabelul arată ultimele cinci zile de tranzacționare ale Băncii Transilvania (TLV) din datele cursului, între 14 și 18 septembrie 2026. Patru dintre rînduri arată obișnuit: prețul de închidere variază între 33,40 și 35,50 lei, iar volumul între 346 819 și 1 884 988 de acțiuni. Bara din 17 septembrie 2026 este diferită. Deschiderea, maximul, minimul și închiderea sînt toate egale cu 33,40 lei, aceeași valoare ca închiderea din ziua anterioară, iar volumul este de 1 084 380 de acțiuni.

O bară plată este normală pentru o acțiune nelichidă într-o zi cu o singură tranzacție sau pentru o zi liberă completată cu prețul anterior. Cu peste un milion de acțiuni tranzacționate, un singur preț pentru toată ziua este foarte puțin probabil, deci bara este suspectă și trebuie verificată la bursă. Verificări simple de acest tip, inegalitățile dintre cele patru prețuri, barele plate și volumele zero, găsesc multe erori de date înainte ca ele să ajungă în vreo statistică.

## Evenimente corporative și prețul ajustat {h=fa12b5c0}

În ziua unui eveniment corporativ prețul de închidere $P_t$ face un salt, deși valoarea deținerii nu se schimbă. Prețul ajustat elimină aceste salturi. Fiecare eveniment $k$, din ziua $t_k$, are un **factor de ajustare** $F_{t_k}$. Pentru un dividend în numerar $D$, $F = 1 - D/P_{t-1}$: prețul scade cu aproximativ $D$ în ziua ex-dividend, iar scalarea prețurilor anterioare cu $F$ elimină această scădere. Pentru un split $m$-la-$n$, în care $n$ acțiuni vechi devin $m$ acțiuni noi, $F = n/m$; o consolidare este un split cu $m < n$, deci factorul ei este mai mare decît unu și mărește prețurile anterioare.

Prețul ajustat înmulțește fiecare preț trecut cu factorii tuturor evenimentelor ulterioare:

$$ P^{\text{adj}}_t = P_t \prod_{k:\, t_k > t} F_{t_k}. $$ {#eq:adj}

Produsul din @eq:adj parcurge doar evenimentele de după ziua $t$, așa că ultimele prețuri rămîn neschimbate, iar ajustarea lucrează înapoi în timp. Randamentele calculate din prețuri ajustate includ dividendul și nu au salturi artificiale.

Aceeași idee apare direct în definiția randamentului. Cu un dividend $D_t$ plătit în ziua $t$, randamentul simplu este $R_t = (P_t + D_t)/P_{t-1} - 1$ [@FrankeHardleHafner2019]: cîștigul investitorului este variația prețului plus numerarul primit. Un randament calculat din prețul de închidere ajustat dă, pînă la un termen de ordinul doi neglijabil, aceeași valoare.

## Exemplu rezolvat: prețuri ajustate {h=e79809fd}

Fie o acțiune cotată la 100, care plătește un dividend de 2 în ziua 5 și are un split 2-la-1 în ziua 10. Factorul dividendului este $F_5 = 1 - 2/100 = 0{,}98$, iar factorul splitului $F_{10} = 1/2 = 0{,}50$. Un preț de dinaintea zilei 5 este urmat de ambele evenimente, deci se înmulțește cu $0{,}98 \times 0{,}50 = 0{,}49$, iar 100 devine 49. Un preț dintre zilele 5 și 9 este urmat doar de split și se înmulțește cu 0,50. Din ziua 10 nu mai urmează niciun eveniment, iar prețul ajustat este egal cu prețul cotat.

Exemplul arată o proprietate a seriei ajustate ușor de uitat: ea se schimbă la fiecare eveniment nou. După următorul dividend, toate prețurile ajustate anterioare se scalează din nou. Două descărcări ale aceleiași serii ajustate, făcute la date diferite, diferă, iar un rezultat poate fi reprodus doar dacă data descărcării se păstrează împreună cu datele.

Indicii au aceeași alegere. Un **indice de preț** ignoră dividendele, deci nivelul lui scade cînd componentele intră în perioada ex-dividend. Un **indice de randament total** (*total return index*) reinvestește dividendele în indice și măsoară cît ar fi cîștigat un investitor care deține componentele. [Comparația BET cu BET-TR](#s15) arată cît de mare devine diferența într-un deceniu.

## Date reale: Banca Transilvania, prețul de închidere și prețul ajustat {h=b698ff3a}

Graficul compară cele două prețuri ale Băncii Transilvania pe o axă verticală logaritmică: prețul de închidere așa cum s-a tranzacționat (roșu) și prețul de închidere ajustat (verde). Pe 16 august 2022 prețul tranzacționat sare de la 2,12 la 21,20 lei. Valorii băncii nu i s-a întîmplat nimic: a fost o consolidare a acțiunilor, zece acțiuni vechi într-una nouă.

Un randament logaritmic calculat din prețul de închidere înregistrează această zi ca un cîștig de +230% într-o singură zi, o cifră care ar domina orice medie, varianță sau boltire a seriei. Calculat din prețul ajustat, randamentul aceleiași zile este 0,0%. Înainte de august 2022 prețul ajustat se află deasupra celui tranzacționat, pentru că factorul de consolidare 10 mărește toate prețurile anterioare; dividendele și acțiunile gratuite, care micșorează prețurile anterioare, reduc diferența. La sfîrșitul eșantionului cele două linii coincid, cum cere @eq:adj.

Consolidarea este evenimentul cel mai vizibil, dar nu singurul. Din 2015, randamentele zilnice ale celor două serii diferă cu peste 2 puncte procentuale în 24 de zile, zilele cu dividende și acțiuni gratuite. Fiecare dintre aceste zile ar adăuga o pierdere falsă unei serii de randamente construite din prețul tranzacționat.

## Efectul dividendelor: BET și BET-TR {h=979432b4}

Indicele BET (Bucharest Exchange Trading) este indicele de referință al BVB și un indice de preț. BET-TR (BET Total Return) are aceleași componente, dar reinvestește dividendele lor. Graficul urmărește 100 de unități investite în fiecare pe 23 septembrie 2014: la sfîrșitul eșantionului investiția valorează 455 în BET și 1 005 în BET-TR. Cele două linii pornesc împreună și se depărtează puțin mai mult în fiecare an, pentru că fiecare dividend reinvestit în BET-TR cîștigă și randamentele ulterioare.

În termeni anuali, rata de creștere este de 13,5% pe an pentru BET și de 21,2% pentru BET-TR. Dividendele adaugă 7,8 puncte procentuale pe an, mai mult de jumătate din creșterea indicelui de preț. La BVB, ignorarea dividendelor dă o imagine foarte incompletă a ceea ce a cîștigat un investitor în piață.

Lecția se aplică oricărei comparații de indici: comparați lucruri comparabile. DAX este un indice de randament total, în timp ce S&P 500 și BET sînt indici de preț. O comparație a DAX cu S&P 500 sau cu BET avantajează deci DAX cu randamentul din dividende al celorlalte două piețe. Din acest motiv, tabelele de mai departe din capitol includ BET-TR alături de BET.

## Erori în date: două serii EUR/RON {h=406b28ae}

Graficul reprezintă randamentele logaritmice zilnice a două serii ale aceluiași curs de schimb, EUR/RON: seria de la EODHD (roșu) și cursul de referință BNR (verde). Cele două serii au 5 154 de zile comune. În 412 dintre ele, variația zilnică a seriei EODHD diferă de cea a cursului BNR cu peste 1 punct procentual, o diferență foarte mare pentru un curs de schimb ale cărui variații zilnice reale sînt de obicei o mică fracțiune dintr-un procent. Seria roșie arată un bloc de cotații eronate alternante între aproximativ 2010 și 2014 și salturi izolate de peste 10% în ambele direcții.

Erorile schimbă fiecare statistică. Volatilitatea anualizată a seriei EODHD este de 13,4%, față de 4,5% pentru cursul BNR. Cea mai mare variație zilnică a seriei EODHD, în valoare absolută, este de 15,4%, pe 14 august 2025: este revenirea la nivelul corect după o cotație greșită în ziua precedentă. O singură cotație greșită produce astfel două randamente extreme de semne opuse, eroarea și corecția ei, iar ambele măresc volatilitatea și boltirea.

Soluția este regula din [slide-ul despre surse](#s9): folosiți sursa primară a cifrei. Pentru un curs de referință sursa primară este banca centrală, iar cursul folosește seria BNR. O eroare de acest tip se găsește comparînd două surse sau listînd cele mai mari randamente și verificîndu-le pe fiecare.

## Trei erori sistematice care supraestimează performanța {h=23ece688}

Unele erori nu stau în prețurile individuale, ci în felul în care eșantionul este ales sau folosit. **Survivorship bias** apare cînd eșantionul conține doar firmele sau fondurile care încă există. Firmele cu rezultate slabe au fost delistate, au fuzionat sau au dat faliment, deci lipsesc, iar randamentul mediu al supraviețuitorilor supraestimează ce ar fi putut cîștiga un investitor [@BrownEtAl1992]. Un exemplu este componența actuală a BET urmărită înapoi pînă în 2000: companiile care au ieșit între timp din indice nu apar în eșantion.

**Look-ahead bias** înseamnă folosirea unei informații care nu era disponibilă la data deciziei. Un caz tipic este un raport anual publicat în martie și folosit într-un backtest pentru decizii luate în ianuarie. Backtest-ul tranzacționează atunci pe baza unor cifre pe care investitorul nu le putea cunoaște încă. Datele din baze înregistrează adesea perioada la care se referă o cifră, nu data publicării ei, ceea ce face această eroare ușor de comis.

**Data snooping** înseamnă încercarea multor reguli pe aceleași date și raportarea celei mai bune. Chiar dacă nicio regulă nu funcționează, cea mai bună dintre multe pare semnificativă din întîmplare. @HarveyLiuZhu2016 concluzionează că sute de predictori publicați ai randamentelor sînt probabil descoperiri false. Toate cele trei erori supraestimează randamentele și niciuna nu apare în statisticile descriptive ale eșantionului final, care arată perfect normal. Ele pot fi evitate doar prin designul studiului.

## Lista de verificare a datelor {h=001e125c}

Erorile din această secțiune pot fi prinse printr-o rutină scurtă, aplicată fiecărei serii noi. Întîi, notați sursa: sursa primară a cifrei și coloana folosită, close sau adjusted close. Apoi verificați calendarul: weekendurile, zilele libere și zilele lipsă, iar într-o analiză comună a mai multor active, aceleași date pentru toate.

În al treilea rînd, căutați salturile: listați cele mai mari 10 randamente în valoare absolută și verificați-le pe fiecare în știrile din ziua respectivă. Un crah real are o cauză cunoscută; o consolidare sau o cotație greșită nu are. În al patrulea rînd, căutați barele plate, volumul zero și prețurile repetate, care arată zile libere completate cu ultimul preț sau prețuri stale (neactualizate). În al cincilea rînd, verificați unitățile și moneda: lei, euro sau dolari, puncte de indice, prețuri în cenți.

În sfîrșit, păstrați fișierul brut și data descărcării și nu modificați niciodată datele manual. Orice corecție se face în cod, unde poate fi citită, repetată și anulată. Aceste reguli costă puțin și protejează toate rezultatele ulterioare ale cursului.

## Recapitulare: date {h=f5e9c43c}

Prețurile provin de la burse, furnizori de date și bănci centrale, iar cînd sursele diferă referința este sursa primară a cifrei. Pentru acțiuni trebuie folosit prețul ajustat, iar cînd dividendele contează, ca la BVB, trebuie folosiți indici de randament total în locul indicilor de preț: în 2014–2026, BET-TR a crescut de la 100 la 1 005, BET doar la 455.

Datele reale conțin erori, ca consolidarea zece-la-unu a Băncii Transilvania sau cotațiile greșite ale seriei EUR/RON de la EODHD. Ele se găsesc verificînd salturile, barele plate și o a doua sursă. Survivorship bias, look-ahead bias și data snooping supraestimează rezultatele fără să lase vreo urmă vizibilă în date.

## @end

O serie de prețuri utilizabilă este capătul unui lanț de decizii: sursa, coloana, calendarul, ajustarea pentru evenimentele corporative și verificarea erorilor. Exemplele secțiunii arată că fiecare verigă contează: o singură consolidare produce un randament fictiv de +230% într-o zi, ignorarea dividendelor scade cu 7,8 puncte procentuale pe an creșterea pieței românești, iar o serie cu cotații greșite triplează volatilitatea unui curs de schimb. Cu prețuri curate, [secțiunea următoare](#sec-2) explică de ce statistica din curs lucrează cu randamente, nu cu prețuri.

# De la prețuri la randamente

Metodele statistice ale cursului se aplică randamentelor, nu prețurilor. Secțiunea dă motivele: randamentele nu au unitate de măsură, deci se pot compara între active, și oscilează în jurul unui nivel stabil, în timp ce prețurile au trend. A doua proprietate este precizată prin noțiunea de staționaritate slabă și verificată formal cu testul Dickey–Fuller augmentat (ADF, *augmented Dickey–Fuller*) pe S&P 500.

## Randamentele în locul prețurilor {h=b5120568}

Un preț are unitate de măsură și scară; un randament nu are. O variație de 1 leu este mare pentru o acțiune de 2 lei și mică pentru una de 500 de lei. Randamentul exprimă variația față de prețul inițial, deci randamentele unor active diferite, în monede diferite și pe perioade diferite, se pot compara direct. Și întrebarea investitorului este relativă: cu cît s-a schimbat averea mea? Răspunsul este exact un randament.

Al doilea motiv este statistic. Prețurile au trend și nu revin la un nivel fix, așa că media și varianța lor depind de perioada observată. Randamentele oscilează în jurul unui nivel stabil. Conceptul din spatele acestei diferențe este **staționaritatea slabă** (*weak stationarity*): o serie este slab staționară dacă media ei este constantă, varianța ei este constantă, iar covarianța dintre două observații depinde doar de distanța dintre ele, lagul, nu de dată.

Majoritatea metodelor statistice ale cursului presupun cel puțin staționaritate slabă, pentru că doar atunci mediile în timp estimează mărimi fixe ale populației. O medie de eșantion a prețurilor nu estimează nimic stabil; o medie de eșantion a randamentelor estimează randamentul așteptat. Următoarele două slide-uri arată diferența pe date și o testează formal.

## S&P 500: nivelul prețului și randamentele zilnice {h=7de12d97}

Panoul de sus arată S&P 500 în puncte de indice din 2000. Prețul are un trend crescător, cu oscilații lungi: scăderile din 2000–2002 și 2008–2009 și o creștere puternică după 2012. Media și dispersia lui pe orice fereastră depind de anii pe care îi acoperă fereastra, ceea ce înseamnă, în practică, nestaționaritate.

Panoul de jos arată randamentele logaritmice zilnice ale aceluiași indice. Ele oscilează în jurul lui zero pe toată perioada, în anii calmi și în cei agitați, ca în criza financiară din 2008 și în șocul pandemiei din 2020. Nivelul seriei nu se deplasează.

Panoul de jos arată și că mărimea oscilațiilor nu este aceeași în toate perioadele: randamentele mari vin în grupuri, în jurul lui 2008 și 2020. Fenomenul, gruparea volatilității (*volatility clustering*), nu contrazice staționaritatea slabă, care privește varianța necondiționată. O serie poate avea o varianță constantă pe termen lung, în timp ce varianța ei condiționată de trecutul recent se schimbă de la o zi la alta. Modelele care descriu acest lucru sînt subiectul [Capitolului 9](ch9_ro.html).

## O verificare formală: testul ADF (1/2) {h=028713f5}

Testul Dickey–Fuller augmentat (ADF) [@DickeyFuller1979] întreabă dacă o serie are o rădăcină unitară (*unit root*), adică dacă se comportă ca un mers aleator (*random walk*). Ipoteza nulă $H_0$ este rădăcina unitară, deci seria este nestaționară; alternativa $H_1$ este că seria este staționară. Testul regresează variația zilnică a seriei pe nivelul ei din ziua anterioară:

$$ \Delta y_t = \alpha + \beta t + \gamma\, y_{t-1} + \sum_{j=1}^{p} \delta_j\, \Delta y_{t-j} + \varepsilon_t . $$ {#eq:adf}

În @eq:adf, $y_t$ este seria testată, logaritmul prețului sau randamentul logaritmic, iar $\Delta y_t = y_t - y_{t-1}$ este variația ei de la ziua $t-1$ la ziua $t$. Constanta $\alpha$ și trendul liniar $\beta t$ permit o derivă și un trend; cele $p$ variații cu lag, cu coeficienții $\delta_j$, preiau autocorelația pe termen scurt, iar $\varepsilon_t$ este eroarea. Coeficientul esențial este $\gamma$. Dacă $\gamma = 0$, variația de azi nu depinde de nivelul de ieri: seria nu are un nivel la care să revină, adică are o rădăcină unitară. Dacă $\gamma < 0$, un nivel ridicat este urmat în medie de o scădere, iar unul scăzut de o creștere, deci seria este readusă spre nivelul ei și este staționară.

Testul respinge $H_0$ cînd statistica $t$ a lui $\hat\gamma$ este sub o valoare critică. Valorile critice nu vin din distribuția Student, ci din tabelele Dickey–Fuller, pentru că în ipoteza $H_0$ regresorul $y_{t-1}$ este el însuși nestaționar, iar statistica $t$ are o distribuție nestandard, deplasată spre stînga. Valoarea critică depinde și de includerea constantei și a trendului.

## O verificare formală: testul ADF (2/2) {h=41be6661}

Pe date zilnice S&P 500 din 2000, testul dă două răspunsuri clare. Pentru logaritmul prețului, cu trend în regresie, statistica este −2,08, valoarea critică de 5% este −3,41, iar p-value este 0,56: rădăcina unitară nu este respinsă. Pentru randamentele logaritmice, statistica este −14,3, valoarea critică de 5% este −2,86, iar p-value este sub 0,001: rădăcina unitară este respinsă. Cele două valori critice diferă pentru că prima regresie include un trend, iar a doua doar o constantă.

P-value este probabilitatea, în ipoteza $H_0$, de a obține o statistică cel puțin la fel de negativă ca cea observată. Un p-value de 0,56 înseamnă că un mers aleator ar produce o statistică precum −2,08 în mai mult de jumătate din cazuri, deci datele nu dau niciun motiv de îndoială asupra rădăcinii unitare. O statistică de −14,3 este practic imposibilă în prezența unei rădăcini unitare. Cu pragul obișnuit de 0,05, $H_0$ este respinsă pentru randamente și păstrată pentru prețuri.

Concluzia susține alegerea cursului: prețurile se comportă ca un mers aleator, randamentele sînt staționare, deci modelele statistice se construiesc pentru randamente. Faptul că rădăcina unitară nu este respinsă nu dovedește că ea există, dar cele două rezultate împreună dau o imagine coerentă. Rădăcinile unitare și ipoteza mersului aleator sînt studiate în detaliu în [Capitolul 7](ch7_ro.html).

## @end

Randamentele sînt preferate prețurilor din două motive: nu au unitate de măsură, deci se pot compara între active și perioade, și sînt staționare, în timp ce prețurile nu sînt. Pe S&P 500 din 2000, testul ADF păstrează rădăcina unitară pentru logaritmul prețului (statistica −2,08, p-value 0,56) și o respinge pentru randamentele logaritmice (statistica −14,3). Există însă două moduri de a defini un randament, iar [secțiunea următoare](#sec-3) arată cînd diferă și pe care să-l folosim.

# Randamente simple și logaritmice

Un randament se poate defini ca variația relativă a prețului sau ca variația logaritmului lui. Cele două definiții dau aproape același număr într-o zi obișnuită și numere foarte diferite într-un crah sau pe un orizont lung. Secțiunea le definește pe amîndouă, măsoară diferența dintre ele, o ilustrează pe zilele cele mai extreme ale Bitcoin și S&P 500 și dă o regulă de alegere între ele.

## Randamentul simplu {h=42e90548}

**Randamentul simplu (net)** (*simple return*) de la $t-1$ la $t$ este variația relativă a prețului [@FrankeHardleHafner2019; @Tsay2010]:

$$ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1 . $$ {#eq:simple}

Raportul $1 + R_t = P_t/P_{t-1}$ este **randamentul brut** (*gross return*): valoarea la momentul $t$ a unei unități investite la $t-1$. De aici încolo, $P_t$ este prețul de închidere al unui indice sau prețul ajustat al unei acțiuni, astfel încît dividendele și spliturile sînt deja luate în calcul.

Randamentul simplu este mărginit inferior: $R_t \ge -1$. Marginea exprimă răspunderea limitată: prețul nu poate scădea sub zero, deci un acționar nu poate pierde mai mult decît a investit. De exemplu, cu $P_{t-1} = 100$ și $P_t = 110$, $R_t = 10\%$. Este cifra pe care o vede investitorul în extrasul de cont cînd fondul raportează că „a cîștigat 10%”.

## Randamentul logaritmic {h=ce581884}

**Randamentul logaritmic** (*log return*), numit și randament compus continuu, este variația logaritmului prețului:

$$ r_t = \ln\frac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1} = \ln(1 + R_t), \qquad R_t = e^{r_t} - 1 . $$ {#eq:log}

Denumirea vine de la capitalizare. Un leu investit la rata $r$, capitalizată de $m$ ori în cursul perioadei, ajunge la $(1 + r/m)^m$, iar pentru $m \to \infty$ această valoare tinde la $e^{r}$. Randamentul logaritmic este deci rata care, capitalizată continuu pe durata perioadei, transformă $P_{t-1}$ în $P_t$. Cu $P_{t-1} = 100$ și $P_t = 110$, $r_t = \ln 1{,}1 = 9{,}53\%$, sub randamentul simplu de 10% calculat din aceleași prețuri.

Spre deosebire de randamentul simplu, randamentul logaritmic este nemărginit: $r_t \in (-\infty, +\infty)$, iar pierderea totală, $P_t = 0$, corespunde lui $r_t = -\infty$. Randamentele logaritmice sînt alegerea standard în statistica piețelor financiare. Motivul principal, arătat în [secțiunea următoare](#sec-4), este că se adună în timp, astfel încît randamentul pe o perioadă lungă este suma randamentelor zilnice. Domeniul lor, întreaga axă reală, se potrivește și mai bine modelelor statistice.

## Diferența dintre r și R {h=d12d0d53}

Relația dintre cele două randamente rezultă din dezvoltarea în serie Taylor a logaritmului în jurul lui zero:

$$ r = \ln(1 + R) = R - \frac{R^2}{2} + \frac{R^3}{3} - \dots $$ {#eq:taylor}

Logaritmul este o funcție concavă, deci graficul lui se află sub tangenta în zero, dreapta $r = R$. Prin urmare $r < R$ pentru orice $R \neq 0$, atît pentru cîștiguri, cît și pentru pierderi. Pentru variații mici, primul termen de corecție, $-R^2/2$, dă aproape toată diferența.

Pentru o variație zilnică obișnuită, $R = 1\%$, $r = 0{,}995\%$, iar aproximarea $R - R^2/2$ dă tot 0,995%. Diferența $R - r$ este de 0,5 puncte de bază; un **punct de bază** (bp, *basis point*) este 0,01 puncte procentuale. Pentru o variație mare, $R = 10\%$, $r = 9{,}531\%$, în timp ce $R - R^2/2 = 9{,}500\%$, iar diferența $R - r$ este de 46,9 puncte de bază. Pentru date zilnice cele două randamente sînt practic egale; pentru variații mari și orizonturi lungi diferă, iar diferența crește cu pătratul variației.

## Randamente simple și logaritmice în zilele extreme {h=a0369bdf}

Graficul reprezintă randamentul logaritmic în funcție de randamentul simplu. Linia punctată este dreapta de 45 de grade $r = R$, curba continuă este $r = \ln(1 + R)$, iar marcajele sînt cele mai mari variații zilnice ale Bitcoin (cercuri) și ale S&P 500 (pătrate). În jurul lui zero curba și dreapta coincid; departe de zero curba se află sub dreaptă de ambele părți, iar diferența crește mult mai repede pe partea pierderilor.

Cele mai mari variații zilnice din datele cursului arată mărimea diferenței. Pe 12 martie 2020 Bitcoin a avut un randament simplu de −37,2%, dar un randament logaritmic de −46,5%. Pe 7 decembrie 2017 a avut un randament simplu de +25,2% și unul logaritmic de +22,5%. Pe 16 martie 2020 S&P 500 a avut un randament simplu de −12,0% și unul logaritmic de −12,8%.

Pentru că curba este concavă, pierderile par mai mari, iar cîștigurile mai mici atunci cînd sînt măsurate în randamente logaritmice. Asimetria are un sens precis: un randament logaritmic de −46,5% și unul de +46,5% se anulează exact, în timp ce o pierdere simplă de 37,2% cere un cîștig simplu mult mai mare de 37,2% pentru a fi recuperată. În astfel de zile, alegerea tipului de randament schimbă cifra raportată cu cîteva puncte procentuale.

## Exemplu rezolvat: +10%, apoi –10% {h=a9d0f9cd}

Fie traiectoria prețului $100 \to 110 \to 99$. Randamentele simple sînt $+10\%$ și $-10\%$, iar suma lor, zero, sugerează că investitorul a revenit de unde a plecat. În realitate investitorul a pierdut: compunerea celor două randamente dă $1{,}1 \times 0{,}9 - 1 = -1{,}0\%$, variația de la 100 la 99.

Randamentele logaritmice ale aceleiași traiectorii sînt $r_1 = \ln 1{,}1 = 9{,}53\%$ și $r_2 = \ln 0{,}9 = -10{,}54\%$. Suma lor este $-1{,}01\%$, egală exact cu $\ln(99/100)$. Adunarea randamentelor logaritmice dă randamentul logaritmic corect pe cele două perioade, fără nicio corecție.

Regula care rezultă este principala diferență practică dintre cele două randamente: randamentele simple se compun, adică randamentele brute se înmulțesc, iar cele logaritmice se adună. Un cîștig și o pierdere de aceeași mărime procentuală nu se anulează; în randamente simple, pierderea de 10% se aplică unei baze mai mari decît cîștigul de 10%.

## Întrebare pentru sală: +50%, apoi –50% {h=215ae11d}

Întrebarea duce exemplul anterior la cifre mai mari. O acțiune crește cu 50% într-un an și scade cu 50% în anul următor. Ce credeți? La ce nivel ajunge prețul după doi ani, față de nivelul inițial?

Compunerea randamentelor brute dă $1{,}5 \times 0{,}5 = 0{,}75$: averea s-a modificat cu −25%, deși cele două procente au aceeași mărime. În randamente logaritmice, $\ln 1{,}5 + \ln 0{,}5 = \ln 0{,}75 = -28{,}8\%$, o sumă directă.

Forma generală a rezultatului este regula revenirii. După o pierdere $x$, averea este de $1 - x$ ori nivelul anterior, iar revenirea la acel nivel cere un cîștig de $x/(1 - x)$. O pierdere de 50% cere deci un cîștig de 100%. Cîștigul necesar crește mai repede decît pierderea, motiv pentru care adîncimea unei scăderi contează atît de mult în [măsurile de drawdown](#s55) ale capitolului.

## Alegerea tipului de randament {h=fbb12d5d}

Tabelul de pe slide asociază fiecărei situații un tip de randament. Pentru agregarea în timp se folosesc randamentele logaritmice, pentru că randamentul logaritmic pe $k$ perioade este o sumă, $r_t(k) = r_t + \dots + r_{t-k+1}$. Pentru un portofoliu din mai multe active se folosesc randamentele simple, pentru că randamentul portofoliului este exact media ponderată $R_p = \sum_i w_i R_i$. Pentru modelarea statistică sînt preferate randamentele logaritmice, pentru că sînt nemărginite și distribuția lor este mai aproape de una simetrică.

Pentru raportarea către investitori se folosește randamentul simplu, deoarece „fondul a cîștigat 10%” este un randament simplu. Pentru date zilnice și variații mici oricare dintre ele este potrivit, pentru că diferența este aproximativ $R^2/2$, o jumătate de punct de bază într-o zi cu 1%.

Indiferent de alegere, tipul de randament trebuie precizat. Un „randament de 10%” este ambiguu: ca randament logaritmic, el corespunde unui randament simplu de $e^{0{,}1} - 1$, aproximativ 10,5%. Pentru o singură zi ambiguitatea costă puțin; pe un deceniu sau pentru un activ volatil ca Bitcoin poate schimba concluziile.

## Recapitulare: randamente simple și logaritmice {h=f018c387}

Randamentul simplu este $R_t = P_t/P_{t-1} - 1$, iar cel logaritmic $r_t = \ln(P_t/P_{t-1}) = \ln(1 + R_t)$. Pentru orice variație nenulă $r < R$, iar diferența este aproximativ $R^2/2$, neglijabilă pentru date zilnice și mare în crahuri, ca pe 12 martie 2020, cînd Bitcoin a scăzut cu 37,2% în randament simplu și cu 46,5% în randament logaritmic.

Randamentele logaritmice se adună în timp; cele simple se adună între active. Cîștigurile și pierderile de aceeași mărime nu se anulează: +50% urmat de −50% lasă 75% din averea inițială.

## @end

Cele două randamente conțin aceeași informație, dar se comportă diferit la agregare. Randamentele logaritmice @eq:log se adună în timp și sînt datele de intrare naturale ale modelelor statistice; randamentele simple @eq:simple se mediază exact între active și sînt cele pe care le văd investitorii. Diferența, aproximativ $R^2/2$, este de 0,5 puncte de bază într-o zi cu 1% și de 9,3 puncte procentuale în cea mai proastă zi a Bitcoin. [Secțiunea următoare](#sec-4) folosește ambele proprietăți pentru a construi randamente pe mai multe perioade și randamente ale portofoliilor.

# Randamente pe mai multe perioade și ale portofoliilor

Investitorii dețin active luni sau ani și de obicei dețin mai multe active. Secțiunea construiește randamentul pe $k$ perioade din randamentele zilnice, leagă randamentele cumulate de avere și calculează randamentul unui portofoliu. Cele două reguli de agregare din secțiunea anterioară, suma în timp pentru randamentele logaritmice și media ponderată între active pentru cele simple, se aplică pe date reale: patru piețe în 2015–2026 și un portofoliu din două acțiuni BVB.

## Randamente pe k perioade {h=6a4e0cc9}

Pe $k$ perioade care se încheie la $t$, de la $P_{t-k}$ la $P_t$, randamentele simple brute se înmulțesc, iar randamentele logaritmice se adună:

$$ 1 + R_t(k) = \frac{P_t}{P_{t-k}} = (1 + R_t)(1 + R_{t-1}) \cdots (1 + R_{t-k+1}), \qquad r_t(k) = \ln\frac{P_t}{P_{t-k}} = r_t + r_{t-1} + \dots + r_{t-k+1} . $$ {#eq:kper}

Ambele identități din @eq:kper rezultă din scrierea lui $P_t/P_{t-k}$ ca produs de $k$ rapoarte de cîte o perioadă; logaritmul transformă produsul într-o sumă. Aici $R_t(k)$ și $r_t(k)$ sînt randamentul simplu și cel logaritmic pe cele $k$ perioade.

Aditivitatea randamentelor logaritmice are o consecință importantă: media și varianța randamentului logaritmic pe $k$ zile rezultă din cele ale randamentelor zilnice. Media unei sume este suma mediilor, deci media pe $k$ zile este de $k$ ori media zilnică; dacă randamentele zilnice sînt necorelate, și varianța sumei este de $k$ ori varianța zilnică. Pe aceasta se bazează regulile de anualizare din [secțiunea despre anualizare](#sec-5).

Randamentele săptămînale, lunare sau anuale se calculează din prețurile de la sfîrșitul fiecărei perioade, nu prin medierea randamentelor zilnice din interiorul ei. Media a 21 de randamente zilnice este o cifră zilnică, nu un randament lunar; randamentul logaritmic lunar este suma celor zilnice sau, echivalent, logaritmul raportului dintre cele două prețuri de sfîrșit de lună.

## Randamentul cumulat și averea {h=1f930e54}

Pornind de la averea $W_0$, averea după $T$ perioade este

$$ W_T = W_0 \prod_{t=1}^{T} (1 + R_t) = W_0 \exp\Big(\sum_{t=1}^{T} r_t\Big), $$ {#eq:wealth}

unde $\prod_{t=1}^{T}$ este produsul celor $T$ randamente brute, iar $\exp(x) = e^x$ inversează logaritmul. Cele două forme din @eq:wealth sînt cele două identități din @eq:kper aplicate pe tot eșantionul.

**Randamentul simplu cumulat** $\text{CSR}_T = W_T/W_0 - 1$ este cîștigul investitorului, în procente. **Randamentul logaritmic cumulat** $\text{CLR}_T = \ln(W_T/W_0) = \sum_t r_t$ este suma randamentelor logaritmice, iar cele două sînt legate prin $\text{CSR}_T = e^{\text{CLR}_T} - 1$. Randamentul logaritmic cumulat este scara naturală pentru orizonturi lungi: crește liniar cînd rata de creștere este constantă, în timp ce randamentul simplu cumulat crește exponențial.

De aceea graficele averii folosesc o axă verticală logaritmică. Pe o scară logaritmică, distanțe verticale egale corespund unor variații procentuale egale: o dublare de la 100 la 200 și una de la 1 000 la 2 000 au aceeași înălțime. O rată de creștere constantă apare ca o dreaptă, iar panta dreptei este rata de creștere logaritmică.

## Evoluția a 100 de unități investite, 2015–2026 {h=4ff4d563}

Graficul urmărește 100 de unități investite pe 5 ianuarie 2015, prima zi din 2015 în care toate cele patru serii au prețuri, pe o axă verticală logaritmică. Tabelele CAGR de mai departe pornesc fiecare serie din prima ei zi de tranzacționare din 2015, astfel că ratele lor de creștere diferă puțin de cele implicate de acest grafic: 11,9% față de 12,0% pentru S&P 500, 60,6% față de 62,5% pentru Bitcoin. La sfîrșitul eșantionului investiția valorează 379 în S&P 500, 267 în DAX, 470 în BET și 29 475 în Bitcoin. DAX este un indice de randament total, iar S&P 500 și BET sînt indici de preț, deci primele trei linii nu compară lucruri comparabile, așa cum s-a arătat pentru [BET și BET-TR](#s15).

Pentru Bitcoin, randamentul simplu cumulat este de 29 375%, iar randamentul logaritmic cumulat de 5,69; pentru BET, cele două cifre sînt 370% și 1,55. Perechile arată scările diferite ale celor două măsuri: un randament simplu cumulat de zeci de mii de procente corespunde unui randament logaritmic cumulat mai mic de șase, iar relația $\text{CSR}_T = e^{\text{CLR}_T} - 1$ le leagă în ambele cazuri.

Axa logaritmică face graficul lizibil. Pe o axă liniară, linia Bitcoin ar urca pînă aproape de 30 000, iar toate celelalte serii ar apărea ca linii plate lîngă baza graficului. Pe axa logaritmică se vede la toate cele patru serii șocul comun din martie 2020, la fel ca urcarea Bitcoin din 2017 și scăderea lui din 2018.

## Randamentul unui portofoliu {h=09fe625a}

Un portofoliu conține $n$ active cu ponderile $w_1, \dots, w_n$, $\sum_i w_i = 1$, fixate la începutul perioadei, unde $w_i$ este partea din avere investită în activul $i$. Fie $R_i$ și $r_i$ randamentul simplu și cel logaritmic al activului $i$ pe perioadă. Cum averea este aditivă între active, valoarea portofoliului la sfîrșitul perioadei este de $\sum_i w_i (1 + R_i)$ ori valoarea inițială, iar randamentul portofoliului este exact media ponderată a randamentelor simple:

$$ R_p = \sum_i w_i R_i, \qquad r_p = \ln\Big(\sum_i w_i e^{r_i}\Big) \neq \sum_i w_i r_i . $$ {#eq:port}

Randamentul logaritmic al portofoliului, din @eq:port, este logaritmul unei sume ponderate, nu o sumă ponderată de logaritmi. De exemplu, cu 60% în activul 1, cu $R_1 = 10\%$, și 40% în activul 2, cu $R_2 = -5\%$, randamentul portofoliului este $R_p = 0{,}6 \times 10\% + 0{,}4 \times (-5\%) = 4{,}0\%$, iar randamentul logaritmic exact al portofoliului este $r_p = \ln(1 + R_p) = 3{,}92\%$. Randamentele logaritmice ponderate dau $0{,}6 \times 9{,}53\% + 0{,}4 \times (-5{,}13\%) = 3{,}67\%$, o valoare prea mică. Pentru ponderi între zero și unu, randamentul logaritmic ponderat este întotdeauna prea mic, din cauza convexității exponențialei (inegalitatea lui Jensen).

Ponderile nu rămîn fixe. Cînd un activ crește mai mult decît celălalt, partea lui din avere crește, așa că un portofoliu cu ponderi fixe trebuie **reechilibrat** (*rebalanced*), vînzînd o parte din activele care au crescut și cumpărînd din cele care au scăzut. Formula $R_p = \sum_i w_i R_i$ folosește ponderile de la începutul fiecărei perioade, deci un portofoliu reechilibrat zilnic o aplică zi de zi.

## Date reale: un portofoliu TLV–SNP {h=2d3a1a91}

Portofoliul conține 50% Banca Transilvania (TLV) și 50% OMV Petrom (SNP), reechilibrat zilnic, în 2015–2026. În fiecare zi se poate compara randamentul logaritmic exact al portofoliului, $\ln(1 + R_p)$, cu suma ponderată a celor două randamente logaritmice. Diferența este mică: media ei în valoare absolută este de 0,38 puncte de bază, iar cea mai mare valoare este de 13,9 puncte de bază, pe 31 mai 2016.

Pe întreaga perioadă diferențele zilnice mici se acumulează, și mereu în același sens. Randamentul logaritmic cumulat exact al portofoliului este 2,56, deci 1 leu devine 13,0 lei. Randamentele logaritmice ponderate dau un randament logaritmic cumulat de 2,47, care corespunde doar unei valori de 11,8 lei. Eroarea de peste un leu din treisprezece provine în întregime din însumarea unei aproximări foarte precise pentru o singură zi.

Rata de creștere a portofoliului este de 24,5% pe an, față de 27,3% pentru TLV și 19,7% pentru SNP. Regula este deci construirea portofoliilor cu randamente simple, cu @eq:port, și trecerea ulterioară la randamente logaritmice, dacă este nevoie, cu $r_p = \ln(1 + R_p)$.

## Recapitulare: randamente pe mai multe perioade și ale portofoliilor {h=cfa4f118}

În timp, randamentele simple brute se înmulțesc, iar cele logaritmice se adună. Averea urmează relația $W_T = W_0 e^{\text{CLR}_T}$, iar randamentul simplu cumulat este $\text{CSR}_T = e^{\text{CLR}_T} - 1$, ca pentru cele 29 375% ale Bitcoin, care corespund unui randament logaritmic cumulat de 5,69.

Între active, $R_p = \sum_i w_i R_i$ este exact, în timp ce suma ponderată a randamentelor logaritmice este doar o aproximare. Eroarea ei zilnică este foarte mică, în medie 0,38 puncte de bază pentru portofoliul TLV–SNP, dar într-un deceniu devine vizibilă: 11,8 lei în loc de 13,0.

## @end

Secțiunea a fixat cele două reguli de agregare ale cursului: randamente logaritmice pentru sumele în timp @eq:kper, randamente simple pentru mediile între active @eq:port. Averea este exponențiala randamentului logaritmic cumulat @eq:wealth, motiv pentru care istoriile lungi se reprezintă pe o axă logaritmică. Investitorii compară însă activele pe an, nu pe zi sau pe deceniu. [Secțiunea următoare](#sec-5) transformă statisticile zilnice în statistici anuale și măsoară cît de precis poate fi cunoscută media anuală.

# Anualizarea

Statisticile calculate din date zilnice sînt cifre zilnice, în timp ce randamentele și riscurile se exprimă de obicei pe an. Secțiunea dă regulile de anualizare a mediei și a volatilității, definește rata anuală compusă de creștere (CAGR, *compound annual growth rate*) și arată că, spre deosebire de volatilitate, randamentul mediu se estimează foarte imprecis, oricît de frecvente ar fi datele.

## Anualizarea mediei și a volatilității {h=bb39f5c8}

Fie $A$ numărul de observații pe an: aproximativ 252 pentru o bursă, 365 pentru cripto, 52 pentru date săptămînale și 12 pentru date lunare. Fie $\mu$ și $\sigma$ media și abaterea standard (*standard deviation*) a randamentelor la frecvența datelor, de exemplu zilnică. Cum randamentele logaritmice se adună pe un an, media anuală este suma a $A$ medii zilnice, iar aceeași regulă se folosește pentru media randamentelor simple, ca anualizare aritmetică. Volatilitatea, adică abaterea standard a randamentelor, se scalează cu rădăcina pătrată a lui $A$:

$$ \mu_{\text{an}} = A\,\mu, \qquad \sigma_{\text{an}} = \sqrt{A}\,\sigma . $$ {#eq:annual}

A doua regulă din @eq:annual este **regula rădăcinii pătrate a timpului** (*square-root-of-time rule*). Ea este valabilă cînd randamentele sînt necorelate și au varianță constantă: atunci $\Var(r_1 + \dots + r_A) = A\,\sigma^2$, pentru că toate covarianțele sînt nule, iar rădăcina pătrată dă $\sqrt{A}\,\sigma$. Cu autocorelație pozitivă varianța anuală este mai mare decît $A\sigma^2$, iar cu autocorelație negativă mai mică; pe această idee se bazează testele raportului varianțelor din [Capitolul 7](ch7_ro.html).

Pentru S&P 500, abaterea standard zilnică este de 1,12%, deci volatilitatea anuală este $1{,}12\% \times 15{,}87 = 17{,}7\%$, unde $15{,}87 = \sqrt{252}$. Pentru Bitcoin, care se tranzacționează în fiecare zi, factorul este $\sqrt{365} = 19{,}10$, ceea ce dă 66,7%. Factorul bursier $\sqrt{252}$ ar da doar 55,4% și ar subestima riscul Bitcoin cu peste 11 puncte procentuale. De aceea contează regula frecvenței din [secțiunea despre date](#s8).

## Rata anuală compusă de creștere {h=b0a7189a}

**CAGR** (rata anuală compusă de creștere) este rata anuală constantă care transformă primul preț $P_0$ în ultimul preț $P_T$ în $Y$ ani:

$$ \text{CAGR} = \big(P_T / P_0\big)^{1/Y} - 1 , $$ {#eq:cagr}

cu $Y$ măsurat în ani calendaristici. Logaritmînd @eq:cagr se obține $\ln(1 + \text{CAGR}) = \ln(P_T/P_0)/Y$, randamentul logaritmic cumulat pe an. Prin urmare $\text{CAGR} = \exp(\text{media anuală a randamentelor logaritmice}) - 1$.

Pentru S&P 500 în 2015–2026, media anuală a randamentelor logaritmice este de 11,2%, iar $e^{11{,}2\%} - 1 = 11{,}9\%$, aceeași valoare ca CAGR, 11,9%. CAGR se mai numește randamentul mediu geometric, pentru că $1 + \text{CAGR}$ este media geometrică a randamentelor brute anuale.

CAGR folosește doar primul și ultimul preț. Nu spune nimic despre traiectoria dintre ele: două investiții cu același CAGR pot să fi avut volatilități foarte diferite și scăderi foarte diferite pe parcurs. Indicatorii din [secțiunea despre performanță](#sec-7) măsoară aceste diferențe.

## Precizia estimării mediei {h=26ca21ce}

Eroarea standard (SE, *standard error*) a mediei anuale estimate $\hat\mu_{\text{an}}$, calculată din $Y$ ani de date, este

$$ \text{SE}(\hat\mu_{\text{an}}) = \frac{\sigma_{\text{an}}}{\sqrt{Y}} , $$ {#eq:se-mean}

unde căciula indică o valoare estimată din eșantion. Formula depinde de numărul de **ani**, nu de numărul de observații. Intervalul de încredere (*confidence interval*) de 95% este $\hat\mu_{\text{an}} \pm 1{,}96\,\text{SE}$, unde 1,96 este cuantila de 97,5% a distribuției Normale standard.

Pentru S&P 500, cu 11,7 ani de date, media anuală a randamentelor logaritmice este de 11,2%, iar SE este de 5,2%. Intervalul de încredere de 95% este [1,1%; 21,4%]: datele sînt compatibile cu un randament așteptat abia pozitiv, dar și cu unul aproape dublu față de estimare. Pentru Bitcoin estimarea este de 47,4%, cu SE de 19,5%, iar intervalul, $47{,}4\% \pm 1{,}96 \times 19{,}5\%$, este [9,2%; 85,6%], foarte larg.

Volatilitatea, în schimb, se estimează precis din date zilnice, pentru că fiecare observație zilnică aduce informație despre mărimea variațiilor. Asimetria dintre cele două estimări, o medie imprecisă și o volatilitate precisă, revine în tot cursul și este motivul pentru care rapoartele care împart o medie la o volatilitate, ca [raportul Sharpe](#s52), moștenesc imprecizia mediei.

## Întrebare pentru sală: mai multe date, o medie mai bună? {h=7c8bdc1f}

Să presupunem că 10 ani de date zilnice S&P 500 sînt înlocuiți cu 10 ani de date la 5 minute: 78 de bare într-o zi de tranzacționare de 6,5 ore, deci de 78 de ori mai multe observații. Ce credeți? Devine mai mică eroarea standard a randamentului mediu anual?

Nu devine. Randamentul logaritmic mediu pe 10 ani este determinat doar de primul și de ultimul preț, $\hat\mu = \ln(P_T/P_0)/Y$, pentru că suma randamentelor logaritmice se reduce telescopic: fiecare preț intermediar apare o dată cu semnul plus și o dată cu semnul minus. Eșantionarea mai deasă adaugă observații între aceleași două capete, dar nicio informație despre medie. Doar mai mulți **ani** îi reduc eroarea standard, proporțional cu $1/\sqrt{Y}$.

Estimarea volatilității, în schimb, se îmbunătățește cînd crește frecvența, pentru că randamentele la pătrat ale fiecărui interval de 5 minute aduc informație despre varianță. Acesta este principiul volatilității realizate din [Capitolul 8](ch8_ro.html), pînă la limita de la care zgomotul de microstructură începe să domine.

## Recapitulare: anualizarea {h=9b9ba3a8}

Media se anualizează prin înmulțire cu $A$, volatilitatea prin înmulțire cu $\sqrt{A}$, unde $A$ este numărul real de observații pe an al seriei. Regula rădăcinii pătrate presupune randamente necorelate, cu varianță constantă.

CAGR, $(P_T/P_0)^{1/Y} - 1$, este exponențiala mediei anuale a randamentelor logaritmice minus unu. Mediile randamentelor sînt imprecise: SE este $\sigma_{\text{an}}/\sqrt{Y}$, adică 5,2% pe an pentru S&P 500 pe 11,7 ani.

## @end

Anualizarea rezultă din regulile de agregare ale secțiunii anterioare: mediile se adună în timp și, fără autocorelație, la fel și varianțele, ceea ce dă @eq:annual. CAGR @eq:cagr rezumă creșterea prin primul și ultimul preț, iar precizia mediei @eq:se-mean depinde doar de lungimea eșantionului în ani. Cu un interval de 95% pentru media anuală a S&P 500 între 1,1% și 21,4%, media este statistica cel mai puțin sigură a unei serii de randamente. [Secțiunea următoare](#sec-6) descrie celelalte trăsături ale distribuției randamentelor, pe care datele le estimează mult mai bine.

# Statisticile descriptive ale randamentelor

O serie de randamente se descrie întîi prin poziție și dispersie, apoi prin forma distribuției: asimetria, greutatea cozilor și valorile extreme. Secțiunea definește aceste statistici, le calculează pentru indici, Bitcoin și cinci acțiuni BVB, compară histogramele cu distribuția Normală și arată că volatilitatea se schimbă în timp.

## Poziție și dispersie {h=8c4055c2}

Fie un eșantion $r_1, \dots, r_n$ de randamente logaritmice zilnice, în procente. **Media** $\bar r = \frac1n \sum_{t=1}^{n} r_t$ este randamentul zilnic mediu. **Varianța** $\hat\sigma^2 = \frac{1}{n-1} \sum_{t=1}^{n} (r_t - \bar r)^2$ folosește împărțitorul $n - 1$, care o face nedeplasată, iar **abaterea standard** este $\hat\sigma = \sqrt{\hat\sigma^2}$. În finanțe, abaterea standard a randamentelor se numește **volatilitate**.

**Cuantila** (*quantile*) $q_\alpha$ este valoarea sub care se află o fracție $\alpha$ din randamente. Mediana este $q_{0{,}5}$. Cuantila de 1%, $q_{0{,}01}$, stă la baza valorii la risc (VaR, *value at risk*): VaR 1% este $\text{VaR}_{0{,}01} = -q_{0{,}01}$, pierderea depășită în 1% din zile, scrisă ca număr pozitiv. VaR și backtesting-ul lui sînt subiectul [Capitolului 10](ch10_ro.html).

**Minimul** și **maximul** sînt cea mai slabă și cea mai bună zi din eșantion. Ele trebuie raportate întotdeauna cu datele lor, pentru că data identifică evenimentul din spatele unei valori extreme și este calea cea mai rapidă de a deosebi un crah real de o eroare de date.

## Forma: asimetria și boltirea {h=7c945630}

**Asimetria** (*skewness*) măsoară lipsa de simetrie a distribuției:

$$ \hat S = \frac{\frac1n \sum_t (r_t - \bar r)^3}{\hat\sigma^3}, \qquad \hat K = \frac{\frac1n \sum_t (r_t - \bar r)^4}{\hat\sigma^4} . $$ {#eq:moments}

Puterea a treia din @eq:moments păstrează semnul fiecărei abateri, deci pierderile mari împing $\hat S$ sub zero, iar cîștigurile mari peste zero. O distribuție simetrică are $S = 0$; $S < 0$ indică o coadă stîngă mai lungă, cu pierderi mari mai frecvente decît cîștigurile mari de aceeași mărime.

**Boltirea** (*kurtosis*) $\hat K$ folosește puterea a patra, deci zilele extreme o domină. Distribuția Normală are $K = 3$, iar **excesul de boltire** (*excess kurtosis*) $\hat K - 3$ este deci zero pentru distribuția Normală. Un exces de boltire pozitiv indică **cozi groase**: zile extreme mai frecvente decît în cazul distribuției Normale cu aceeași medie și aceeași abatere standard.

Ambele statistici sînt sensibile la zile extreme izolate: un singur crah poate schimba asimetria din pozitivă în negativă și poate dubla boltirea. De aceea ele trebuie raportate împreună cu extremele. Cozile groase și asimetria negativă sînt **fapte stilizate** (*stylised facts*) ale randamentelor, regularități regăsite pe piețe și perioade diferite [@Cont2001]; testele formale sînt în [Capitolul 2](ch2_ro.html).

## Randamente logaritmice zilnice: indici și Bitcoin, 2015–2026 {h=cb1ee880}

Tabelul descrie randamentele logaritmice zilnice a patru indici și ale Bitcoin, fiecare pe propriul calendar; toate valorile, cu excepția asimetriei și a excesului de boltire, sînt în procente. S&P 500 are media de 0,045% pe zi și abaterea standard de 1,12%, DAX 0,032% și 1,20%, BET 0,053% și 0,98%, BET-TR 0,080% și 0,98%. Media Bitcoin este de 0,130%, iar abaterea lui standard de 3,49%, de aproximativ trei ori mai mare decît a indicilor bursieri. Pe fiecare rînd media zilnică este o mică fracțiune din abaterea standard, motiv pentru care media este atît de greu de estimat.

Toate cele cinci serii au asimetrie negativă, de la −0,56 pentru DAX la −1,43 pentru BET-TR, și exces de boltire mare, de la 10,0 pentru DAX la 18,1 pentru BET-TR. BET și BET-TR au aceeași abatere standard și aproape aceeași formă, pentru că dividendele deplasează media, dar aproape nu schimbă fluctuațiile zilnice.

Extremele au date. Cea mai slabă zi a fost 16 martie 2020 pentru S&P 500, cu −12,8%, și 12 martie 2020 pentru DAX, cu −13,1%, și pentru Bitcoin, cu −46,5%: toate în prăbușirea piețelor de la începutul pandemiei de COVID-19. Pentru BET și BET-TR cea mai slabă zi a fost 19 decembrie 2018, cu −11,9%, un eveniment românesc explicat în slide-ul următor. Cele mai bune zile, de la 6,7% pentru BET-TR la 22,5% pentru Bitcoin, sînt mai mici în valoare absolută decît cele mai slabe, în acord cu asimetria negativă.

## Randamente logaritmice zilnice: cinci acțiuni BVB, 2015–2026 {h=32073f90}

Al doilea tabel descrie cinci acțiuni de la BVB: Banca Transilvania (TLV), OMV Petrom (SNP), BRD, Nuclearelectrica (SNN) și Transgaz (TGN). Abaterile lor standard zilnice sînt între 1,53% pentru SNN și 1,69% pentru TLV, toate mult peste cele 0,98% ale indicelui BET. Acesta este efectul diversificării: într-un indice, mișcările specifice companiilor individuale se compensează parțial.

Banca Transilvania și BRD au avut cea mai slabă zi pe 19 decembrie 2018, cu −22,2% și −18,2%, după anunțarea unei taxe pe activele bancare, adoptată prin Ordonanța de Urgență a Guvernului (OUG) 114/2018. Și Transgaz a avut atunci cea mai slabă zi, cu −10,1%, în timp ce SNP și SNN au avut-o pe 9 martie 2020, cu −15,0% și −9,4%. Un singur anunț de politică publică a produs astfel cea mai slabă zi a întregii perioade pentru piața românească.

Forma distribuției diferă mult între acțiuni. TLV are un exces de boltire de 25,2, iar SNN doar de 5,2; SNN și TGN au chiar asimetrie ușor pozitivă, 0,29 și 0,01. Aceste diferențe provin în mare parte din cîteva zile, motiv pentru care extremele și datele lor trebuie citite împreună cu momentele. Pentru TLV, una dintre aceste zile este o eroare de date: prețul ajustat EODHD din 30 mai 2016 este greșit, ceea ce produce un randament de −15,0% în acea zi și de +20,5%, maximul din tabel, pe 31 mai. Tabelul păstrează ambele zile; fără ele, excesul de boltire al TLV este 18,1. Crahurile de la BVB au o descriere statistică proprie, studiată de @PeleMazurencuMarinescu2012.

## Histogramele față de distribuția Normală {h=13d59415}

Graficul compară histogramele randamentelor logaritmice zilnice ale S&P 500 (stînga) și Bitcoin (dreapta) cu densitatea distribuției Normale cu aceeași medie și aceeași abatere standard (linia neagră). Axa orizontală este tăiată la cuantilele de 0,1% și 99,9% ale fiecărei serii; puținele zile din afara lor sînt numărate în barele de la margine. Ambele histograme au aceeași formă caracteristică: un centru mult mai înalt și mai îngust decît densitatea Normală, umeri mai jos și cozi care se întind mai departe.

Pentru S&P 500, 79,6% din zile se află la cel mult o abatere standard de medie, față de 68,3% în cazul distribuției Normale: există mai multe zile liniștite decît prevede distribuția Normală. Dincolo de trei abateri standard se află 1,39% din zile, față de 0,27% în cazul distribuției Normale, adică de 5,2 ori mai multe.

Cele două efecte sînt legate. Cu abaterea standard fixată, o distribuție care pune mai multe zile în cozi trebuie să pună mai multe zile și lîngă centru, și mai puține între ele. Un model Normal potrivit pe aceste date subestimează deci atît frecvența zilelor liniștite, cît și, mai important pentru risc, frecvența zilelor extreme. Măsurile de risc care presupun normalitate, ca un VaR Normal, moștenesc această eroare.

## Volatilitatea se schimbă în timp {h=7251e6a4}

Graficul arată volatilitatea anualizată a S&P 500, BET și Bitcoin, calculată pe ferestre mobile de un an. Volatilitatea S&P 500 variază între 7% și 35%, cea a Bitcoin între 41% și 106%. BET urmează un tipar asemănător cu S&P 500, cu un salt în 2020.

Forma liniilor vine din fereastra mobilă. Cînd o zi extremă intră în fereastră, estimarea sare în sus și rămîne ridicată exact un an, pînă cînd ziua iese din fereastră; așa apar palierele vizibile, de exemplu, pentru S&P 500 din martie 2020 pînă în martie 2021. Estimarea pe fereastră mobilă reacționează deci tîrziu la perioadele calme și netezește perioadele agitate, dar arată clar că volatilitatea nu este constantă: perioadele calme și cele agitate alternează și fiecare durează luni de zile.

O singură volatilitate pentru tot eșantionul ascunde aceste perioade. Tendința ca variațiile mari să urmeze variațiilor mari, iar cele mici celor mici, este volatility clustering. Estimatori mai buni ai volatilității sînt subiectul [Capitolului 8](ch8_ro.html), iar modelele GARCH (heteroscedasticitate condiționată autoregresivă generalizată, *generalised autoregressive conditional heteroskedasticity*), în care varianța depinde de trecutul recent, al [Capitolului 9](ch9_ro.html).

## Recapitulare: statistici descriptive {h=9d3a65ad}

O serie de randamente se raportează cu media, abaterea standard, asimetria, excesul de boltire, cuantilele și extremele, cu datele lor. Randamentele zilnice au de obicei media apropiată de zero, o volatilitate de 1,0–1,7% pe zi pentru indicii și acțiunile BVB din capitol și de 3,5% pentru Bitcoin, cozi groase și adesea asimetrie negativă.

Zilele extreme sînt de cîteva ori mai frecvente decît în cazul distribuției Normale: de 5,2 ori dincolo de trei abateri standard pentru S&P 500. Volatilitatea nu este constantă în timp: pe ferestre de un an, cea a S&P 500 a variat între 7% și 35%.

## @end

Statisticile descriptive ale randamentelor zilnice arată trei regularități, aceleași pentru indici, pentru acțiunile BVB și pentru Bitcoin: medii mici în raport cu abaterea standard, cozi groase cu asimetrie negativă și o volatilitate care se schimbă în timp. Fiecare are o consecință pentru indicatorii din [secțiunea următoare](#sec-7): media mică face impreciși indicatorii bazați pe randament, cozile groase fac erorile lor standard mai mari decît presupun formulele, iar volatilitatea variabilă concentrează cele mai adînci scăderi în cîteva perioade agitate.

# Indicatori de performanță

Investitorii compară investițiile nu doar după cît au crescut, ci și după cît risc au purtat și cît de adînci au fost scăderile lor. Secțiunea definește indicatorii standard ajustați la risc, rapoartele Sharpe, Sortino și Calmar, drawdown-ul și volatility drag, îi aplică activelor din capitol și arată că fiecare dintre ei este o estimare cu o eroare considerabilă.

## Limitele unui singur indicator {h=a1794b14}

Randamentul, luat singur, răsplătește asumarea de risc. O poziție cu efect de levier (*leverage*), finanțată parțial cu bani împrumutați, multiplică atît cîștigurile, cît și pierderile, deci poate avea un randament mare ani la rînd, pînă cînd o scădere o șterge. Un indicator care se uită doar la randament nu poate deosebi un administrator priceput de unul care și-a asumat pur și simplu mai mult risc.

Un investitor pune de aceea trei întrebări. Cît a crescut investiția? Răspunsul este CAGR. Cît de mari au fost fluctuațiile? Răspunsurile sînt volatilitatea și abaterea negativă. Cît de adîncă a fost cea mai mare scădere? Răspunsul este drawdown-ul maxim (MDD, *maximum drawdown*). Indicatorii ajustați la risc combină aceste răspunsuri, împărțind un randament la o măsură a riscului: raportul Sharpe împarte la volatilitate, raportul Sortino la abaterea negativă, raportul Calmar la drawdown-ul maxim.

Toți acești indicatori sînt estimări dintr-un singur eșantion și fiecare are o eroare de estimare. Un clasament al activelor după un indicator este informativ doar dacă diferențele dintre ele sînt mari în raport cu aceste erori. Ideea revine în fiecare slide al secțiunii.

## Raportul Sharpe {h=9eada8f2}

**Raportul Sharpe** [@Sharpe1966; @Sharpe1994] este randamentul în exces pe unitatea de risc total:

$$ \text{SR} = \frac{\mu - r_f}{\sigma}, \qquad \text{SR}_{\text{an}} = \sqrt{A}\,\text{SR}_{\text{zi}} , $$ {#eq:sharpe}

unde $\mu$ este randamentul mediu, $r_f$ rata fără risc, iar $\sigma$ volatilitatea. Raportul anual rezultă din @eq:annual: numărătorul se înmulțește cu $A$, iar numitorul cu $\sqrt{A}$, deci raportul se înmulțește cu $\sqrt{A}$.

În acest capitol raportul Sharpe folosește media randamentelor simple zilnice și $r_f = 0$. Motivul este practic: activele sînt cotate în lei, euro și dolari, fiecare monedă cu propria rată fără risc, iar $r_f = 0$ pentru toate păstrează comparația simplă. Raportul devine astfel un randament mediu pe unitatea de volatilitate, nu un randament în exces în sens strict. Pentru S&P 500 raportul Sharpe zilnic este 0,046, iar cel anual 0,72.

Raportul Sharpe nu depinde de efectul de levier. Un împrumut la rata fără risc care dublează poziția dublează atît randamentul în exces, cît și volatilitatea, deci raportul rămîne același. Două investiții cu același raport Sharpe oferă deci același randament pe unitatea de risc, iar un investitor poate atinge cu oricare dintre ele nivelul de risc dorit. William F. Sharpe, din imaginea slide-ului, a primit Premiul Nobel pentru economie în 1990.

## Raportul Sharpe este o estimare {h=51403a58}

Pentru randamente i.i.d. (independente și identic distribuite), @Lo2002 dă eroarea standard a raportului Sharpe estimat $\widehat{\text{SR}}$:

$$ \text{SE}(\widehat{\text{SR}}) \approx \sqrt{\frac{1 + \text{SR}^2/2}{n}} , $$ {#eq:se-sr}

pe perioadă, unde $n$ este numărul de randamente; eroarea standard anuală se obține înmulțind cu $\sqrt{A}$, numărul de randamente pe an. Pentru date zilnice raportul pe perioadă este mic, deci termenul $\text{SR}^2/2$ aproape nu contează, iar eroarea standard anuală este aproximativ $\sqrt{A/n} = 1/\sqrt{Y}$: în jur de 0,3 cu 12 ani de date, oricare ar fi activul.

Pentru S&P 500 în 2015–2026, $\text{SR} = 0{,}72$, cu eroarea standard 0,29, iar intervalul de încredere de 95% este [0,15; 1,30]. Intervalul exclude zero, dar este compatibil atît cu un raport randament–risc modest, cît și cu unul excepțional de bun. Ca și media, raportul Sharpe este cunoscut doar aproximativ după mai mult de un deceniu de date.

Eroarea reală este mai mare decît sugerează @eq:se-sr, pentru că cozile groase și volatility clustering încalcă ipoteza i.i.d. O a doua sursă de eroare este selecția: alegerea celei mai bune dintre multe strategii supraestimează raportul Sharpe al strategiei alese, iar @BaileyLopezDePrado2014 propun o corecție pentru aceasta, raportul Sharpe deflatat (*deflated Sharpe ratio*).

## Raportul Sharpe cu intervale de încredere 95%, 2015–2026 {h=34170735}

Graficul arată rapoartele Sharpe anualizate ale celor zece active din capitol, cu $r_f = 0$, fiecare cu intervalul lui de încredere de 95%. Cea mai mare estimare punctuală este cea a BET-TR, urmată de Nuclearelectrica, Banca Transilvania și Bitcoin; cea mai mică este cea a DAX.

Toate intervalele se suprapun, deci datele nu susțin un clasament sigur al acestor active după raportul Sharpe. Intervalul DAX conține zero: în această perioadă raportul lui Sharpe nu diferă semnificativ de zero. Suprapunerea intervalelor este un ghid informal; o comparație formală a două rapoarte Sharpe folosește eroarea standard a diferenței lor, care depinde și de corelația dintre cele două serii de randamente.

Lățimile intervalelor sînt aproape egale, deși activele diferă mult ca volatilitate. Rezultatul decurge din @eq:se-sr: eroarea standard a unui raport Sharpe anual depinde mai ales de lungimea eșantionului, aproximativ $1/\sqrt{Y}$, iar toate seriile acoperă aceiași ani.

## Raportul Sortino {h=0ed15e62}

Investitorii se tem de pierderi, nu de cîștiguri, dar volatilitatea tratează la fel abaterile în ambele direcții. **Abaterea negativă** (*downside deviation*) sub un prag $\tau$, aici $\tau = 0$, ia în calcul doar zilele de sub prag:

$$ \sigma_D = \sqrt{\frac1n \sum_t \min(R_t - \tau, 0)^2}, \qquad \text{Sortino} = \frac{\mu - \tau}{\sigma_D} . $$ {#eq:sortino}

În @eq:sortino, $\min(R_t - \tau, 0)$ este deficitul sub prag din ziua $t$, zero în zilele de peste prag, iar $n$ numără toate zilele eșantionului, nu doar pe cele de sub prag. **Raportul Sortino** (*Sortino ratio*) [@SortinoVanDerMeer1991] împarte randamentul mediu peste prag la abaterea negativă și se anualizează ca raportul Sharpe.

Pentru o distribuție simetrică centrată aproape de prag, aproximativ jumătate din abaterile la pătrat cad sub prag, deci $\sigma_D \approx \sigma/\sqrt2$, iar raportul Sortino este de aproximativ $\sqrt2 \approx 1{,}4$ ori raportul Sharpe. Pentru S&P 500 în 2015–2026, raportul Sharpe este 0,72, iar raportul Sortino 1,02, aproape de această regulă. Un raport Sortino mult peste 1,4 ori raportul Sharpe ar arăta că pierderile sînt mai mici decît cîștigurile; mult sub, că sînt mai mari.

## Drawdown și drawdown maxim {h=01312fb6}

**Drawdown-ul** este scăderea față de cel mai mare preț atins pînă atunci:

$$ \text{DD}_t = \frac{P_t}{\max_{s \le t} P_s} - 1 \le 0, \qquad \text{MDD} = \min_t \text{DD}_t , $$ {#eq:dd}

unde $\max_{s \le t} P_s$ este cel mai mare preț din toate zilele $s$ de pînă la ziua $t$, inclusiv. Drawdown-ul este zero la fiecare nou maxim și negativ între ele. **Drawdown-ul maxim** (MDD) este cea mai mare scădere de la un vîrf la minimul următor. El trebuie raportat cu datele lui: vîrful, minimul și **revenirea** la vechiul vîrf, dacă s-a produs.

Adîncimea unui drawdown contează mai mult decît pare, din cauza regulii revenirii: după o pierdere $x$, cîștigul necesar pentru revenirea la vîrf este $x/(1 - x)$. O pierdere de 10% cere 11%, una de 20% cere 25%, una de 50% cere 100%, iar una de 80% cere 400%. Revenirea după scăderile adînci este mult mai grea decît sugerează mărimea lor.

Spre deosebire de volatilitate, MDD depinde de traiectoria prețurilor și crește cu lungimea eșantionului: cu cît istoria este mai lungă, cu atît sînt mai multe ocazii pentru o scădere adîncă [@MagdonIsmailEtAl2004]. Două MDD-uri sînt comparabile doar pe perioade de aceeași lungime.

## Drawdown-uri pe întreaga istorie {h=ba0644f1}

Graficul reprezintă drawdown-ul S&P 500 (din 2000), al BET (din 1997) și al Bitcoin pe întreaga lor istorie. Fiecare linie atinge zero la un nou maxim și coboară sub zero în timpul unei scăderi; adîncimea unei adîncituri este drawdown-ul, iar lățimea ei este timpul petrecut sub vîrful anterior.

S&P 500 a avut MDD-ul de −56,8% de la 9 octombrie 2007 la 9 martie 2009, în criza financiară globală, și și-a recuperat vîrful pe 28 martie 2013. BET a avut un MDD de −82,5% de la 24 iulie 2007 la 25 februarie 2009 și a revenit abia pe 16 martie 2021. Bitcoin a avut un MDD de −83,4% de la 16 decembrie 2017 la 15 decembrie 2018.

Graficul arată cît de diferite sînt cele trei istorii. S&P 500 a petrecut perioade lungi aproape de vîrf, cu două scăderi adînci; BET a petrecut cea mai mare parte a perioadei 2008–2021 mult sub vîrful din 2007; linia Bitcoin oscilează permanent între noi maxime și scăderi de peste 50%.

## Interpretarea graficului drawdown-urilor {h=14299019}

Graficul se citește pe două dimensiuni. Prima este adîncimea. După un drawdown de −82,5%, BET a avut nevoie de un cîștig de +473% pentru revenire; după −83,4%, Bitcoin a avut nevoie de +502%. Aceste cifre rezultă din regula revenirii $x/(1-x)$ și arată de ce un drawdown de peste 80% este atît de distructiv.

A doua este durata. BET a avut nevoie de 13,6 ani ca să revină la vîrful din 2007. Ponderea zilelor petrecute cu peste 20% sub vîrful anterior este de 27% pentru S&P 500, 56% pentru BET și 66% pentru Bitcoin. Un investitor în BET sau în Bitcoin a petrecut cea mai mare parte a timpului cu o pierdere de peste o cincime față de momentul cel mai bun.

Volatilitatea și raportul Sharpe ignoră ordinea randamentelor: amestecarea zilelor nu schimbă niciuna dintre ele. Drawdown-ul ține cont de ordine, pentru că o serie de pierderi consecutive produce o scădere mai adîncă decît aceleași pierderi răspîndite printre cîștiguri. În perioada de comparație 2015–2026, MDD-ul fiecărui indice bursier s-a produs în februarie–martie 2020, în timp ce Bitcoin și mai multe acțiuni BVB l-au avut la alte date.

## Raportul Calmar {h=ab3ce511}

**Raportul Calmar** (*Calmar ratio*) măsoară creșterea pe unitatea celei mai mari scăderi:

$$ \text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|} . $$ {#eq:calmar}

Este popular printre administratorii de fonduri, pentru că MDD este o pierdere pe care investitorul o suportă efectiv, în timp ce volatilitatea este o măsură abstractă a dispersiei. Pentru S&P 500 în 2015–2026, CAGR este 11,9%, iar MDD −33,9%, deci raportul Calmar este $11{,}9/33{,}9 = 0{,}35$.

Numitorul din @eq:calmar este determinat de un singur eveniment, cea mai adîncă scădere din eșantion. Un crah în plus sau în minus schimbă raportul mult mai mult decît ar schimba volatilitatea, deci raportul Calmar este și mai instabil decît raportul Sharpe. Cum MDD crește și cu lungimea eșantionului, rapoartele Calmar se compară doar pe aceeași perioadă.

## Volatility drag {h=6d62a985}

Pentru un randament logaritmic cu media $m$ și varianța $\sigma^2$, randamentul simplu are media aproximativ $m + \sigma^2/2$; pentru randamente logaritmice cu distribuție Normală relația este exactă, deoarece $\E[e^{r}] = e^{m + \sigma^2/2}$. Prin urmare rata de creștere a averii, media randamentelor logaritmice, se află sub media randamentelor simple:

$$ \mu_{\text{log}} \approx \mu_{\text{arith}} - \frac{\sigma^2}{2} , $$ {#eq:drag}

unde $\mu_{\text{arith}}$ este media randamentelor simple, iar $\mu_{\text{log}}$ media randamentelor logaritmice. Diferența $\sigma^2/2$ este **volatility drag** [@BoothFama1992]. Ea provine din concavitatea logaritmului, prin inegalitatea lui Jensen, și este același efect ca în exemplul cu +10% urmat de −10%.

Fie două active cu aceeași medie aritmetică, 10% pe an. Cu $\sigma = 5\%$, drag-ul este de 0,12%, rata de creștere de 9,88% pe an, iar după 10 ani 1 leu devine 2,68. Cu $\sigma = 30\%$, drag-ul este de 4,50%, rata de creștere de 5,50% pe an, iar după 10 ani 1 leu devine doar 1,73. Același randament mediu produce averi foarte diferite.

Drag-ul crește cu pătratul volatilității, așa că dublarea volatilității mărește de patru ori volatility drag. Pentru activele cu volatilitate mică el este neglijabil; pentru cele volatile devine principala diferență dintre randamentul mediu și creșterea efectivă a averii.

## Volatility drag pe date reale {h=54c2c02c}

Graficul testează @eq:drag pe activele din capitol, în 2015–2026. Fiecare punct este un activ. Axa verticală arată media anuală aritmetică minus media anuală logaritmică, adică drag-ul observat în date; axa orizontală arată $\sigma^2/2$, drag-ul prevăzut de formulă. Ambele axe sînt logaritmice, iar linia punctată marchează egalitatea.

Toate punctele se află pe dreaptă, de la indici, cu un drag de aproximativ 1,2–1,8 puncte procentuale pe an, prin acțiunile BVB, între aproximativ 2,6 și 3,3, pînă la Bitcoin. Pentru Bitcoin media aritmetică este de 69,5% pe an, iar media logaritmică de 47,4%, deci drag-ul observat este de 22,1 puncte procentuale, în timp ce $\sigma^2/2 = 21{,}9\%$.

Potrivirea strînsă arată că diferența dintre cele două medii nu este un accident statistic, ci o consecință a volatilității. Pentru Bitcoin drag-ul elimină aproape o treime din media aritmetică.

## Întrebare pentru sală: ce randament pentru Bitcoin? {h=419945de}

Pentru Bitcoin în 2015–2026, media aritmetică a randamentelor simple este de 69,5% pe an, iar CAGR este de 60,6%. Ce credeți? Care dintre cele două cifre descrie creșterea averii unui investitor?

Răspunsul este CAGR: este rata care a transformat efectiv primul preț în ultimul. Media aritmetică este randamentul așteptat pe un singur an, nu rata de creștere pe mulți ani.

Diferența dintre cele două este volatility drag. În termeni logaritmici drag-ul este de 22,1 puncte procentuale: de la media aritmetică de 69,5% la un randament logaritmic mediu de 47,4%. Exprimată ca rată anuală compusă, o creștere logaritmică de 47,4% pe an înseamnă un CAGR de $e^{0{,}474} - 1 = 60{,}6\%$. Pentru activele cu volatilitate mică cele două cifre aproape coincid; pentru Bitcoin, prezentarea mediei aritmetice drept „randamentul” supraestimează creșterea cu aproximativ nouă puncte procentuale pe an.

## Indicatori de performanță: indici și Bitcoin, 2015–2026 {h=1a12ff9c}

Tabelul reunește indicatorii secțiunii pentru cei patru indici și Bitcoin. Toți sînt anualizați cu frecvența reală a fiecărei serii; volatilitatea (Vol.) este cea a randamentelor simple zilnice, iar rapoartele Sharpe și Sortino folosesc $r_f = 0$. Coloana Drag este diferența observată dintre media aritmetică și cea logaritmică, în puncte procentuale.

Bitcoin are de departe cel mai mare CAGR, 60,6%, dar și cea mai mare volatilitate, 66,2%, și cel mai adînc drawdown, −83,4%; drag-ul lui, de 22,13 puncte procentuale, este de peste zece ori mai mare decît al oricărui indice. Raportul lui Sharpe, 1,05, se află în intervalul celorlalte active, iar raportul Calmar, 0,73, este apropiat de cel al BET-TR, 0,71. Dintre indici, DAX are cel mai mic CAGR, 8,5%, și cel mai mic raport Sharpe, 0,52, iar S&P 500 se află la mijloc, cu 11,9% și 0,72.

Comparația BET-TR cu BET izolează efectul dividendelor. Volatilitatea, 15,4%, și MDD, −31,1%, sînt aceleași, pentru că dividendele aproape nu schimbă fluctuațiile zilnice. Fiecare indicator construit pe baza randamentului este mai mare: CAGR este 22,1% față de 14,1%, raportul Sharpe 1,38 față de 0,93, raportul Calmar 0,71 față de 0,45.

## Indicatori de performanță: acțiuni BVB, 2015–2026 {h=4d60ecb7}

Al doilea tabel dă aceiași indicatori pentru cele cinci acțiuni BVB, calculați din prețuri ajustate, deci cu dividendele incluse. Nuclearelectrica are cel mai mare CAGR, 30,7%, cel mai mare raport Sharpe, 1,27, cel mai mare raport Sortino, 2,00, și cel mai mare raport Calmar, 0,83. Urmează Banca Transilvania, cu un CAGR de 27,3% și un raport Sharpe de 1,08. BRD are cel mai mic CAGR, 18,9%, și cel mai mic raport Sharpe, 0,83.

Acțiunile individuale au o volatilitate mai mare decît indicele BET, între 23,0% și 25,5%, față de 15,4%, și drawdown-uri mai adînci, de la −35,6% pentru BRD la −45,4% pentru OMV Petrom, față de −31,1%. Este efectul diversificării observat în statisticile descriptive, acum în indicatorii de risc.

Aceste diferențe trebuie citite în raport cu eroarea de estimare. Fiecare raport Sharpe are o eroare standard de aproximativ 0,29, așa că o diferență ca aceea dintre Transgaz, 0,98, și OMV Petrom, 0,89, este departe de a fi semnificativă. Chiar și diferența dintre cea mai bună și cea mai slabă acțiune este mai mică decît dublul erorii standard a unui singur raport.

## Risc și randament, 2015–2026 {h=b92c443c}

Graficul plasează fiecare activ după volatilitatea anualizată (axa orizontală) și după CAGR (axa verticală). Bitcoin se află departe, în colțul din dreapta sus. Acțiunile BVB formează un grup, cu volatilități între 23,0% și 25,5% și CAGR între 18,9% și 30,7%; indicii se află la stînga, cu volatilitate mai mică.

Clasamentul depinde de indicator. După CAGR cel mai bun activ este Bitcoin; după raportul Sharpe, BET-TR; după rapoartele Sortino și Calmar, Nuclearelectrica (SNN). Cea mai mică volatilitate aparține BET-TR, 15,4%, ca și pentru BET la precizia din tabele. Tot BET-TR are cel mai mic MDD în valoare absolută, −31,115% față de −31,124% pentru BET: practic o egalitate. Fiecare indicator răspunde la o altă întrebare, deci fiecare produce propriul clasament.

Regula practică este precizarea indicatorului ori de cîte ori un activ este numit „cel mai bun”. O afirmație ca „Bitcoin a fost cea mai bună investiție din 2015–2026” este adevărată pentru creștere și falsă pentru randamentul ajustat la risc măsurat prin raportul Sharpe.

## Recapitulare: indicatori de performanță {h=d17a4424}

CAGR măsoară creșterea, volatilitatea măsoară riscul, iar rapoartele Sharpe și Sortino măsoară randamentul pe unitatea de risc total sau de risc de scădere. MDD și raportul Calmar descriu cea mai mare scădere, pierderea pe care investitorul o suportă efectiv.

Volatility drag, $\mu_{\text{log}} \approx \mu_{\text{arith}} - \sigma^2/2$, plasează creșterea sub media aritmetică, cu 22,1 puncte procentuale pe an pentru Bitcoin. Fiecare indicator este o estimare, iar diferențele dintre active trebuie comparate cu erorile lor standard, de aproximativ 0,29 pentru un raport Sharpe estimat pe 2015–2026.

## @end

Indicatorii secțiunii răspund la întrebări diferite și, pe aceleași date, ordonează activele diferit: Bitcoin conduce după creștere, BET-TR după raportul Sharpe, Nuclearelectrica după rapoartele Sortino și Calmar. Raportul Sharpe @eq:sharpe are o eroare standard @eq:se-sr de aproximativ 0,29 pe 2015–2026, deci niciunul dintre aceste clasamente nu este sigur statistic, iar raportul Calmar @eq:calmar, determinat de o singură scădere, este și mai puțin stabil. Volatility drag @eq:drag leagă media aritmetică de creșterea averii. [Secțiunea următoare](#sec-8) transformă problema erorii de estimare într-o întrebare de cercetare deschisă.

# AI pentru descoperire științifică

Capitolul se încheie cu o întrebare deschisă pe care metodele lui o pot aborda și cu rolul pe care instrumentele de inteligență artificială (AI, *artificial intelligence*) îl pot avea în studierea ei. Întrebarea este dacă rapoartele Sharpe persistă în timp. Secțiunea descrie ce poate aduce AI, ce verificări rămîn în sarcina cercetătorului și un proiect care aplică indicatorii capitolului acțiunilor de la BVB.

## O întrebare deschisă {h=a172c5ad}

**Persistă în timp rapoartele Sharpe?** Dacă o acțiune BVB a avut un raport Sharpe mare în 2015–2020, a avut unul mare și în 2021–2026? Răspunsul contează în practică. Dacă rapoartele Sharpe din trecut nu persistă, clasarea fondurilor sau a acțiunilor după raportul Sharpe trecut, o practică frecventă, îi este de puțin folos unui investitor care alege pentru viitor.

Pentru BVB întrebarea este deschisă din motive statistice. Eșantioanele sînt scurte, numărul acțiunilor lichide este mic, iar un raport Sharpe estimat pe jumătate din eșantionul capitolului are o eroare standard de aproximativ 0,4, conform @eq:se-sr. Cu erori de această mărime, persistența reală și zgomotul pur se deosebesc greu.

Rezultatele conexe îndeamnă la prudență: multe regularități ale randamentelor publicate în literatură nu se confirmă pe date noi [@HarveyLiuZhu2016]. Instrumentele AI pot accelera un astfel de studiu, de la căutarea literaturii pînă la cod, dar nu înlocuiesc verificarea lui [@WangEtAl2023].

## Utilizări posibile ale instrumentelor AI {h=d165d39e}

Un asistent AI poate contribui la patru etape ale unui astfel de studiu. Pentru **literatură**, poate găsi lucrări despre persistența performanței și le poate rezuma metodele. Pentru **cod**, poate scrie o primă versiune a unei funcții Python care calculează raportul Sharpe pe ferestre mobile, pe datele cursului.

Pentru **robustețe**, poate propune alte ferestre, alți indicatori, ca rapoartele Sortino și Calmar, și alte piețe. Pentru **explicație**, poate transforma un tabel de rezultate într-o primă versiune a interpretării, pe care cercetătorul o verifică apoi pe cifre.

Promptul de pe slide arată nivelul de precizie de care are nevoie o cerere utilă: precizează datele de intrare (randamente simple zilnice), indicatorul (raportul Sharpe anualizat), ferestrele (fără suprapunere, de trei ani) și metoda pentru eroarea standard (@Lo2002). O cerere vagă produce cod ale cărui ipoteze utilizatorul trebuie să le descopere ulterior.

## Verificări necesare {h=84ec9807}

Fiecare rezultat al unui instrument AI trebuie verificat, iar capitolul oferă lista de verificare. Datele: prețuri ajustate pentru acțiuni, același calendar pentru toate activele, perioada corectă. Formulele: factorul de anualizare, $\sqrt{252}$ sau $\sqrt{365}$, randamente simple sau logaritmice, împărțitorul $n$ sau $n - 1$. Capitolul a arătat că fiecare dintre aceste alegeri schimbă rezultatul, de exemplu cu 11 puncte procentuale volatilitatea anuală a Bitcoin.

Cifrele: recalculați manual două sau trei valori pe exemple mici sau cu o a doua metodă. Referințele: fiecare lucrare citată trebuie să existe, iar DOI-ul ei (*digital object identifier*) trebuie să ducă la ea; modelele de limbaj pot produce referințe plauzibile care nu există.

Selecția: dacă ați încercat zece ferestre, raportul trebuie să precizeze acest lucru. Un backtest repetat de multe ori, cu mici modificări pînă cînd rezultatul arată bine, duce la supraajustare (*overfitting*) [@BaileyEtAl2014], aceeași problemă ca data snooping din [prima secțiune](#s17).

## Idee de proiect {h=f5251f58}

Proiectul întreabă dacă rapoartele Sharpe ale acțiunilor BVB persistă de la o perioadă la alta. Datele sînt prețurile ajustate a 10–15 acțiuni BVB în 2010–2026, din datele cursului.

Metoda are trei pași. Întîi, împărțiți eșantionul în două jumătăți și calculați raportul Sharpe al fiecărei acțiuni în fiecare jumătate, cu eroarea lui standard. Apoi măsurați persistența prin corelația rangurilor dintre cele două jumătăți: o corelație mare înseamnă că acțiunile clasate sus în prima jumătate s-au clasat sus și în a doua. În sfîrșit, repetați analiza cu rapoartele Sortino și Calmar, pentru a vedea dacă răspunsul depinde de indicator.

Livrabilele sînt un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele, avînd în vedere erorile standard. Orice utilizare a AI trebuie declarată, împreună cu erorile AI găsite și corectate.

## @end

Persistența rapoartelor Sharpe este o întrebare pe care instrumentele capitolului o pot aborda, dar nu o pot tranșa pe datele BVB, pentru că un raport Sharpe estimat pe șase ani are o eroare standard de aproximativ 0,4. Instrumentele AI pot scurta munca la literatură, cod și robustețe; verificarea datelor, formulelor, cifrelor, referințelor și selecției rămîne în sarcina cercetătorului. [Rezumatul](#sec-9) reunește formulele capitolului.

# Rezumat

Capitolul a construit mărimile de bază ale cursului: de la o serie de prețuri verificată la randamente, agregarea lor în timp și între active, anualizarea lor, statisticile lor descriptive și indicatorii care compară investițiile. Rezumatul reunește formulele și trei întrebări de autoevaluare.

## Formule de reținut {h=25f7bb87}

Tabelul de pe slide reunește formulele capitolului în ordinea în care au fost introduse. Randamentul simplu @eq:simple și randamentul logaritmic @eq:log sînt cele două definiții de bază, legate prin $r_t = \ln(1 + R_t)$. Pe $k$ perioade, randamentele simple brute se înmulțesc, iar cele logaritmice se adună @eq:kper; între active, randamentul portofoliului este media ponderată a randamentelor simple @eq:port.

Anualizarea înmulțește media cu $A$ și volatilitatea cu $\sqrt{A}$ @eq:annual, iar CAGR @eq:cagr dă rata anuală constantă de creștere între primul și ultimul preț. Rapoartele Sharpe și Sortino împart randamentul mediu la volatilitate @eq:sharpe și la abaterea negativă @eq:sortino; MDD @eq:dd și raportul Calmar @eq:calmar descriu cea mai mare scădere. Volatility drag @eq:drag leagă media aritmetică de rata de creștere.

Ultimul rînd dă cele două erori standard ale capitolului: $\sigma_{\text{an}}/\sqrt{Y}$ pentru media anuală @eq:se-mean și $\sqrt{(1 + \text{SR}^2/2)/n}$ pentru raportul Sharpe @eq:se-sr. Ambele depind de lungimea eșantionului și ambele sînt destul de mari pentru a schimba interpretarea majorității comparațiilor din capitol.

## Autoevaluare {h=c080ec9b}

Prima întrebare cere media și volatilitatea anuală ale unui fond ale cărui randamente logaritmice zilnice au media 0,05% și abaterea standard 1%. Cu 252 de zile de tranzacționare, media anuală este $252 \times 0{,}05\% = 12{,}6\%$, iar volatilitatea anuală $\sqrt{252} \times 1\% = 15{,}9\%$, conform @eq:annual.

A doua întrebare cere motivul pentru care media aritmetică a randamentelor Bitcoin este mult peste CAGR. Răspunsul este volatility drag, $\sigma^2/2$, care este mare cînd $\sigma$ este mare: pentru Bitcoin, cu o volatilitate anuală de aproximativ 66%, drag-ul este de aproximativ 22 de puncte procentuale pe an.

A treia întrebare cere cîștigul de care are nevoie un indice pentru a reveni după o scădere de 60%. Conform regulii revenirii $x/(1 - x)$, cîștigul necesar este $0{,}6/0{,}4 = 150\%$. Capitolul următor, [Capitolul 2](ch2_ro.html), studiază distribuțiile clasice și faptele stilizate ale randamentelor, pornind de la cozile groase și asimetria negativă găsite aici.

## @end

Dintr-o singură serie de prețuri, capitolul a derivat mărimile folosite în restul cursului: două tipuri de randamente, agregarea lor, mediile și volatilitățile anualizate, momentele și extremele distribuției lor și indicatorii ajustați la risc, cu erorile lor standard. Două constatări merg mai departe. Distribuția randamentelor zilnice are cozi groase și volatilitate variabilă în timp, subiectul [Capitolului 2](ch2_ro.html) și al Capitolelor [8](ch8_ro.html) și [9](ch9_ro.html). Iar randamentul mediu, împreună cu fiecare indicator construit pe baza lui, se estimează cu o eroare pe care datele mai frecvente nu o pot reduce.
