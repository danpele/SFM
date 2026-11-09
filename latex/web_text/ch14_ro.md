<!--
Textul de curs al versiunii web, capitolul 14 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie → Autor (an), [@Cheie] → (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 14 --hashes
-->

## @intro

@epigraph Nakamoto2008 | Bitcoin: A Peer-to-Peer Electronic Cash System
> A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution.

@text
Capitolele 2–10 au construit instrumentele statistice ale cursului pe acțiuni, indici și cursuri valutare: distribuțiile randamentelor, cozile groase, modelele GARCH (*generalised autoregressive conditional heteroskedasticity*), testele de eficiență și măsurile de risc VaR (*value at risk*) și ES (*expected shortfall*). Acest capitol le aplică unei piețe cu trei particularități: se tranzacționează 365 de zile pe an, activele ei nu au fluxuri de numerar și pot pierde 80% din valoare. Întrebarea este dacă instrumentele funcționează aici și ce rezultate dau.

Pe datele zilnice pînă pe 18 septembrie 2026, volatilitatea anuală a Bitcoin este 66,6%, de 3,8 ori mai mare decît a S&P 500 (17,6%). Faptele stilizate din [Capitolul 2](ch2_ro.html) se păstrează într-o formă mai puternică: asimetrie negativă, exces de boltire mare și volatility clustering atît de persistent, încît modelul GARCH estimat este la limita integrată. Testele robuste ale raportului varianțelor nu resping însă eficiența în formă slabă, iar corelația cu S&P 500, apropiată de zero înainte de 2020, a crescut după aceea la 0,39.

Capitolul tratează și două subiecte proprii pieței cripto. Stablecoins sînt active cripto care încearcă să păstreze un preț fix de 1 USD, iar episoadele din martie 2023 (USDC) și mai 2022 (TerraUSD) arată cele două moduri în care se poate rupe o paritate. ETF-urile spot pe Bitcoin, aprobate în ianuarie 2024, ridică întrebarea dacă intrarea activului la bursă i-a schimbat comportamentul statistic.

@chart sfm_ch14_rolling_vol
Volatilitatea anualizată pe circa o lună, din 2016: Bitcoin (portocaliu) și Ethereum (mov), din 30 de randamente zilnice înmulțite cu $\sqrt{365}$, și S&P 500 (albastru închis), din 21 de randamente înmulțite cu $\sqrt{252}$.

@goals
- **Explicați** cum înregistrează un blockchain proprietatea, de ce mining-ul consumă energie și cum rezultă oferta de Bitcoin din regula halving-ului.
- **Anualizați** randamentele și volatilitățile cripto cu frecvența reală (365 de zile) și explicați eroarea introdusă de folosirea a 252 de zile.
- **Estimați** tail index-ul, un model GARCH(1,1)-t și testele raportului varianțelor pentru Bitcoin și comparați-le cu S&P 500.
- **Măsurați** în timp corelația activelor cripto cu acțiunile și aurul, precum și drawdown-urile unui activ cripto.
- **Descrieți** cele trei tipuri de stablecoins, măsurați abaterea de la paritate în puncte de bază și interpretați un episod de depeg.
- **Calculați** VaR 1% și ES 2,5% ale unei poziții cripto și verificați-le prin backtesting.

@plan
- Bitcoin și Ethereum pe scurt | Blockchain-ul, proof of work, halving-ul și oferta de 21 de milioane, Ethereum și proof of stake, datele de preț.
- Activele cripto ca clasă de active | Anualizarea cu 365 de zile, cinci active comparate, volatilitatea în timp și efectul weekendului.
- Cozi și volatility clustering | Estimatorul Hill, ACF și testul Ljung–Box, GARCH(1,1)-t și GJR pentru Bitcoin, Ethereum, S&P 500 și aur.
- Eficiența pieței cripto | Testele raportului varianțelor pe întregul eșantion și pe ferestre mobile, literatura despre eficiența Bitcoin și două studii de caz.
- Corelația cu acțiunile și aurul | Corelația între calendare diferite, corelația mobilă și anuală, trei crize și testul activului de refugiu.
- Bule și crahuri | Drawdown-ul, marile scăderi ale Bitcoin și legea de putere log-periodică.
- ETF-urile spot pe Bitcoin | Volatilitatea și ponderea weekendului înainte și după ianuarie 2024 și tracking error-ul unui ETF.
- Stablecoins | Tipuri, arbitrajul parității, oferta, abaterile de la paritate, USDC în martie 2023, prăbușirea TerraUSD și reglementarea.
- Riscul unei poziții cripto | VaR 1% și ES 2,5% în dolari prin patru metode și backtesting-ul cu testul Kupiec.
- AI pentru descoperire științifică | O întrebare deschisă despre fluxurile de stablecoins și prețurile cripto, ajutorul posibil al AI și verificările necesare.
- Rezumat | Ideile principale, formulele și întrebări de autoevaluare.

# Bitcoin și Ethereum pe scurt

Secțiunea descrie mecanismul din spatele celor două active cripto principale atît cît este necesar pentru analiza statistică: cum ține un blockchain evidența proprietății, ce asigură securitatea lui, de unde vine regula ofertei de Bitcoin și ce schimbă Ethereum. Se încheie cu datele: cum se formează o serie zilnică de prețuri pentru o piață fără o bursă centrală.

## Lucrarea lui Nakamoto (2008) {h=ad2c7970}

Pe 31 octombrie 2008, un autor care semna „Satoshi Nakamoto” a publicat o lucrare de nouă pagini care propunea un sistem de numerar electronic peer-to-peer [@Nakamoto2008]. Peer-to-peer înseamnă că utilizatorii își plătesc direct unii altora, fără o bancă la mijloc. Problema pe care o rezolvă lucrarea este dubla cheltuire (*double spending*): o monedă digitală este un fișier care poate fi copiat, iar fără o autoritate care ține evidența nimic nu împiedică plata de două ori cu aceeași monedă.

Primul bloc al rețelei a fost creat pe 3 ianuarie 2009, iar primele tranzacții pe exchange-uri au avut loc în 2010. Un exchange este o platformă de tranzacționare unde activele cripto se cumpără și se vînd contra dolari sau contra altor active cripto. În limbajul capitolului, un activ cripto este un activ digital a cărui proprietate este înregistrată într-un blockchain.

Identitatea autorului este încă necunoscută, iar monedele obținute de el prin mining în 2009–2010 nu au fost mutate niciodată. În cincisprezece ani, o idee dintr-o lucrare tehnică a devenit o piață cu propriile prețuri, burse, derivate și, din 2024, fonduri tranzacționate la bursă; restul capitolului studiază această piață cu instrumentele statistice ale cursului.

## Blockchain-ul: un registru public {h=62480c26}

Un registru (*ledger*) este lista tuturor tranzacțiilor: cine îl ține știe cine ce deține. Băncile țin registre private, fiecare pentru clienții ei. Bitcoin ține un singur registru public, copiat pe mii de calculatoare numite noduri (*nodes*), astfel încît nicio instituție nu îl controlează singură.

Tranzacțiile sînt grupate în blocuri. Fiecare bloc are un hash, o amprentă scurtă calculată cu funcția SHA-256 (*Secure Hash Algorithm, 256 bits*); schimbarea unui singur caracter din bloc îi schimbă complet hash-ul. Fiecare bloc conține hash-ul blocului anterior, iar blocurile formează astfel un lanț. Consecința este că rescrierea unei tranzacții vechi schimbă hash-ul blocului ei și, prin el, toate blocurile ulterioare, care ar trebui și ele rescrise.

Cine are dreptul să adauge următorul bloc și cu ce cost decide regula de consens (*consensus*), subiectul slide-ului următor. Proprietatea asupra unei monede este dată de cheia privată (*private key*) a adresei la care se află moneda: cine deține cheia poate cheltui moneda, iar o cheie pierdută înseamnă o monedă pierdută definitiv.

## Proof of work și mining-ul {h=77c0488a}

Regula de consens a Bitcoin este proof of work (PoW): pentru a adăuga un bloc, un miner trebuie să găsească un număr care, adăugat blocului, face hash-ul acestuia mai mic decît o țintă. Funcția hash nu poate fi inversată, deci singura metodă este încercarea repetată, cu miliarde de hash-uri pe secundă pe mașini specializate. Ținta se ajustează la fiecare 2016 blocuri, astfel încît un bloc nou să apară în medie cam la 10 minute, oricîtă putere de calcul ar intra în rețea.

Minerul care găsește soluția primește subvenția blocului (*block subsidy*), adică monede nou create, plus comisioanele tranzacțiilor din bloc. Securitatea vine din cost: pentru a rescrie istoria, un atacator ar trebui să refacă munca tuturor blocurilor ulterioare mai repede decît restul rețelei, deci ar avea nevoie de mai multă putere de calcul decît toți minerii onești la un loc.

Costul acestei securități este electricitatea consumată de mașinile de mining, ca în ferma din fotografia slide-ului. De aceea Bitcoin este criticat pentru consumul de energie, iar Ethereum a ales ulterior o altă regulă de consens.

## Regula ofertei și halving-ul {h=03e6ebfa}

Subvenția blocului a fost 50 BTC în 2009 și se înjumătățește la fiecare 210 000 de blocuri, adică aproximativ la patru ani: 25 BTC în 2012, 12,5 în 2016, 6,25 în 2020 și 3,125 BTC pe bloc din aprilie 2024. Evenimentul înjumătățirii se numește halving. Oferta totală este suma unei serii geometrice:

$$ 210\,000 \times 50 \times \Big(1 + \tfrac12 + \tfrac14 + \dots\Big) = 210\,000 \times 50 \times 2 = 21 \text{ milioane BTC}. $$ {#eq:supply}

Seria converge rapid: la începutul lui 2024 existau circa 19,4 milioane de BTC, iar în 2030 vor exista 20,4 milioane, deci aproape toate monedele există deja. Oferta de Bitcoin este fixată de protocol, nu de o bancă centrală, iar această raritate programată este unul dintre argumentele cererii pentru Bitcoin.

Datele halving-urilor se cunosc cu ani înainte. Pe o piață eficientă ([Capitolul 7](ch7_ro.html)), o reducere previzibilă a ofertei noi ar trebui să fie deja inclusă în preț, astfel încît halving-ul în sine nu ar trebui să producă un randament anormal. Slide-ul următor confruntă această predicție cu datele.

## Prețul și oferta de Bitcoin {h=b260cbc1}

Graficul are două panouri. Cel din stînga arată închiderea zilnică a Bitcoin în USD, pe scală logaritmică, din septembrie 2014 pînă în septembrie 2026; liniile verticale întrerupte marchează halving-urile din 2016, 2020 și 2024. Prețul a crescut de la 457 USD în septembrie 2014 la 80 901 USD pe 18 septembrie 2026. Scala logaritmică face comparabile variațiile procentuale de la nivele foarte diferite: o creștere de la 1 000 la 10 000 USD ocupă aceeași înălțime ca una de la 10 000 la 100 000 USD.

Panoul din dreapta arată oferta de Bitcoin rezultată din regula protocolului, în milioane de BTC, din 2009 pînă în 2040, împreună cu limita de 21 de milioane din @eq:supply. Curba crește repede în primii ani și se aplatizează după fiecare halving, apropiindu-se asimptotic de limită.

La un an după cele trei halving-uri, prețul era mai mare cu 287%, 559% și, respectiv, 31%. Cifrele par să susțină ideea unui efect al halving-ului, dar trei observații nu sînt o dovadă statistică: fiecare an conține și alte evenimente, iar efectul se micșorează de la un halving la altul. Cu un eveniment la patru ani, un test credibil nu va fi posibil încă multă vreme.

## Anonimatul fondatorului Bitcoin {h=d9924dd5}

Nimeni nu știe cine este „Satoshi Nakamoto”: fondatorul a încetat să mai scrie în 2011. Regulile Bitcoin se schimbă doar dacă majoritatea nodurilor acceptă noul software, deci nu există o persoană sau o companie care să decidă politica rețelei.

Pentru un statistician, consecința principală este lipsa fluxurilor de numerar. Bitcoin nu are o companie în spate, nu produce profit și nu plătește dividende, deci nu există un flux de numerar care să poată fi actualizat ca într-un model de evaluare a acțiunilor. Prețul reflectă doar cererea, care vine din plăți, din speculație și din încrederea în raritatea monedei; @BaurHongLee2018 arată că Bitcoin este folosit mai ales ca activ speculativ și mai puțin ca mijloc de schimb.

Modelele obișnuite de evaluare nu se aplică, așa că prețul este studiat ca obiect statistic: distribuția randamentelor, dinamica volatilității, dependența de alte piețe. Tranzacțiile au loc non-stop, pe sute de exchange-uri, 365 de zile pe an, ceea ce schimbă calendarul datelor și, cum arată secțiunea următoare, modul în care se anualizează.

## Ethereum, contractele inteligente și proof of stake {h=eacb3ac2}

În 2014, @Buterin2014 a propus Ethereum, un blockchain care nu doar înregistrează transferuri, ci execută programe; rețeaua a fost lansată în iulie 2015. Un contract inteligent (*smart contract*) este un program stocat în blockchain care execută o tranzacție atunci cînd condițiile lui sînt îndeplinite. Pe această bază au fost construite token-uri, stablecoins și DeFi (*decentralised finance*), adică credit și tranzacționare fără intermediari.

Ethereum folosește din 2022 o altă regulă de consens, proof of stake (PoS). Dreptul de a adăuga un bloc se trage la sorți proporțional cu monedele blocate drept garanție (*stake*), iar frauda se pedepsește prin pierderea garanției. Securitatea nu mai vine din electricitatea consumată, ci din capitalul pus în joc, deci nu mai există o cursă a puterii de calcul.

Trecerea s-a făcut pe 15 septembrie 2022, într-un eveniment numit „the Merge”, iar consumul de energie al rețelei a scăzut cu peste 99,9%. Pentru analiza statistică, Ethereum este al doilea activ cripto ca importanță și apare alături de Bitcoin în majoritatea tabelelor capitolului.

## Prețuri și date {h=8fe06d3c}

Pe piața cripto nu există un exchange unic: fiecare platformă are propriul preț, iar prețurile pot diferi între țări [@MakarovSchoar2020]. Un furnizor de date agregă platformele într-o serie zilnică. Capitolul folosește datele EODHD (*EOD Historical Data*), cu închiderea la miezul nopții UTC (*Coordinated Universal Time*).

Piața se tranzacționează șapte zile pe săptămînă, fără goluri de weekend și fără sărbători. De aceea, pentru aceeași perioadă calendaristică, seriile au lungimi diferite: Bitcoin are 4 384 de randamente zilnice din 2014, iar S&P 500 are 3 017 în aceeași perioadă. Randamentele se calculează ca $r_t = 100(\ln P_t - \ln P_{t-1})$, în procente, fiecare serie pe propriul calendar.

Pentru piața în ansamblu există indici cu mai multe monede, de exemplu CRIX (*CRypto IndeX*) al lui @TrimbornHardle2018. CRIX alege numărul de monede incluse cu criteriul AIC (*Akaike information criterion*, [Capitolul 6](ch6_ro.html)), iar rezultatul este că puține monede descriu întreaga piață. Analiza capitolului se concentrează de aceea pe Bitcoin și Ethereum, cu Solana ca al treilea activ cripto în comparațiile descriptive.

## @end

Bitcoin este un registru public securizat prin proof of work, cu o ofertă fixă de 21 de milioane de monede, iar Ethereum adaugă contracte inteligente și, din 2022, proof of stake. Activele cripto nu au fluxuri de numerar, se tranzacționează non-stop pe multe platforme, iar seriile lor zilnice au alt calendar decît cele ale acțiunilor; aceste trăsături determină alegerile statistice din restul capitolului.

# Activele cripto ca clasă de active

Secțiunea compară activele cripto cu acțiunile și aurul prin statisticile din capitolele anterioare. Începe cu anualizarea, unde calendarul de 365 de zile schimbă rezultatul, continuă cu un tabel al celor cinci active și cu volatilitatea în timp și se încheie cu întrebarea dacă weekendul este diferit pe o piață care nu se închide.

## Anualizarea cu frecvența reală (1/2) {h=07c703f9}

Cu $A$ observații pe an și randamente zilnice i.i.d. (independente și identic distribuite, [Capitolul 1](ch1_ro.html)), media și volatilitatea anuală sînt

$$ \mu_{\text{an}} = A\,\bar r, \qquad \sigma_{\text{an}} = \sqrt{A}\,s, $$ {#eq:annual}

unde $\bar r$ este randamentul zilnic mediu, $s$ abaterea standard a randamentelor zilnice, iar $A$ numărul de randamente pe an. Randamentul logaritmic anual este suma celor $A$ randamente zilnice; mediile lor se adună, iar în ipoteza de independență se adună și varianțele, de unde factorul $\sqrt{A}$ pentru abaterea standard.

Valoarea corectă a lui $A$ este numărul real de observații pe an, nu o constantă convențională. Pentru acțiuni, $A \approx 252$ de zile de tranzacționare; pentru activele cripto, care se tranzacționează în fiecare zi, $A = 365$. În datele capitolului, frecvența reală este 365,3 randamente pe an pentru Bitcoin și 251,4 pentru S&P 500.

## Anualizarea cu frecvența reală (2/2) {h=83d8dfc6}

Exemplul lucrat folosește randamentele zilnice ale Bitcoin, cu media 0,1181% și abaterea standard 3,485%. Cu frecvența corectă, volatilitatea anuală este $\sqrt{365} \times 3{,}485 = 19{,}10 \times 3{,}485 = 66{,}6\%$, iar media anuală $365 \times 0{,}1181 = 43{,}1\%$. Cu regula acțiunilor se obțin $\sqrt{252} \times 3{,}485 = 55{,}3\%$ și $252 \times 0{,}1181 = 29{,}8\%$.

Regula cu 252 subestimează deci volatilitatea cu 16,9% și media cu 31,0%. Raportul Sharpe, care împarte media la volatilitate, este și el afectat, iar amestecul celor două convenții (media înmulțită cu 252, volatilitatea cu $\sqrt{365}$) îl deformează și mai mult. Eroarea nu ține de date, ci de calendar, și este ușor de făcut cînd un program scris pentru acțiuni este aplicat fără modificări unei serii cripto.

Regula practică este ca fiecare serie să fie anualizată cu propriul $A$, niciodată cu unul comun. O comparație între Bitcoin și S&P 500 folosește deci 365 pentru primul și 252 pentru al doilea, ca în toate tabelele acestui capitol.

## Compararea a cinci active {h=134d99a9}

Tabelul rezumă randamentele logaritmice zilnice ale celor cinci active pînă pe 18 septembrie 2026: Bitcoin din 2014, Ethereum din 2016, Solana din 2020, S&P 500 din 2014 și aurul (cursul XAU/USD) din 2014. Volatilitatea anuală este calculată cu @eq:annual și cu frecvența reală a fiecărei serii. Asimetria este media lui $((r_t - \bar r)/s)^3$, iar excesul de boltire media lui $((r_t - \bar r)/s)^4$ minus 3; ambele sînt 0 pentru distribuția Normală.

Diferența de nivel este mare. Bitcoin este de 3,8 ori mai volatil decît S&P 500 (66,6% față de 17,6% pe an), Ethereum de 5,4 ori (95,3%), iar Solana de 6,7 ori (117,1%). Aurul, cu 16,4%, este apropiat de S&P 500. Bitcoin se mișcă cu peste 5% într-o zi în 11,6% din zile, iar cea mai proastă zi a eșantionului a fost 12 martie 2020, cu un randament logaritmic de −46,5%.

Forma distribuțiilor este însă comună. Toate cele cinci active au asimetrie negativă și exces de boltire pozitiv: excesul este 12,0 pentru Bitcoin, 15,8 pentru S&P 500 și 9,5 pentru aur. Faptele stilizate din [Capitolul 2](ch2_ro.html) se păstrează deci și pe piața cripto, la o scară a volatilității mult mai mare.

## Volatilitatea în timp {h=41fea1e3}

Graficul arată volatilitatea anualizată pe circa o lună, din 2016 pînă în 2026. Pentru Bitcoin (portocaliu) și Ethereum (mov), fiecare punct este abaterea standard a ultimelor 30 de randamente zilnice înmulțită cu $\sqrt{365}$; pentru S&P 500 (albastru închis), a ultimelor 21 de randamente înmulțită cu $\sqrt{252}$. Ferestrele acoperă astfel aproximativ aceeași lună calendaristică.

Medianele sînt 54% pentru Bitcoin, 77% pentru Ethereum și 12% pentru S&P 500, iar raportul median dintre volatilitatea Bitcoin și cea a S&P 500 este 3,8. Toate trei seriile ating maximul în martie–aprilie 2020, la începutul pandemiei de COVID-19 (*coronavirus disease 2019*), cînd și volatilitatea S&P 500 urcă la nivelul obișnuit al activelor cripto.

Volatilitatea Bitcoin scade în timp: vîrfurile sînt mai rare și mai joase după 2023 decît în 2016–2018. Chiar și așa, volatilitatea Bitcoin a fost sub cea a S&P 500 doar în 0,1% din zile. Scăderea nivelului nu a schimbat deci ordinea celor două piețe.

## Studiu de caz: riscurile și randamentele criptomonedelor {h=572b0bc1}

@LiuTsyvinski2021 studiază randamentele zilnice și săptămînale ale Bitcoin, Ripple și Ethereum între 2011 și 2018. Metoda lor este regresia randamentelor cripto pe factorii pieței de acțiuni, pe cursuri valutare, pe mărfuri și pe factori macroeconomici, adică pe variabilele care explică randamentele activelor tradiționale.

Rezultatul principal este că randamentele cripto nu au aproape nicio expunere la acești factori, deci formează o clasă de active separată. În schimb, au un momentum puternic în timp: randamentele mari dintr-o săptămînă prognozează randamente mari în săptămînile următoare. Atenția investitorilor, măsurată prin căutările Google și prin postările pe Twitter, prognozează și ea randamentele.

@LiuTsyvinskiWu2022 continuă analiza în secțiune transversală și arată că trei factori cripto, piața, dimensiunea și momentum-ul, explică randamentele diferitelor monede. Ambele studii folosesc însă date de dinainte de 2020, iar legătura cu acțiunile s-a schimbat după acest an ([Secțiunea 5](#sec-5)): concluzia de independență față de piețele tradiționale depinde de perioada eșantionului.

## Efectul zilei săptămînii la Bitcoin {h=c0a7631d}

Bursele de acțiuni se închid în weekend, piețele cripto nu. Totuși, în weekend băncile sînt închise, investitorii instituționali sînt mai puțin activi, iar fluxul de știri încetinește. Efectul zilei din săptămînă (*day-of-the-week effect*) înseamnă o medie sau o varianță a randamentelor diferită în anumite zile; @CaporalePlastun2019 îl studiază pe piața cripto.

Capitolul face două teste pe Bitcoin, cu fiecare zi etichetată după data închiderii. Pentru medie se estimează regresia $r_t = a + b\,W_t + \varepsilon_t$, unde $W_t = 1$ sîmbăta și duminica și 0 în rest. Coeficientul $a$ este randamentul mediu în zilele lucrătoare, iar $b$ diferența dintre weekend și zilele lucrătoare; $b = 0$ înseamnă lipsa unui efect de weekend în medie. Erorile standard sînt HAC (*heteroskedasticity and autocorrelation consistent*), de tip Newey–West, deoarece reziduurile au volatility clustering.

Pentru varianță se folosește testul Brown–Forsythe, adică testul Levene calculat în jurul medianei, robust la cozile groase. Statistica descriptivă asociată este raportul dintre varianța din weekend și cea din zilele lucrătoare, egal cu 1 dacă fiecare zi ar avea același risc.

## Volatilitatea Bitcoin pe zile ale săptămînii {h=16912d51}

Graficul arată abaterea standard a randamentelor zilnice ale Bitcoin, în procente, pentru fiecare zi a săptămînii (de luni pînă duminică), după ziua închiderii, în trei perioade: 2014–2019 (albastru închis), 2020–2023 (portocaliu) și 2024–2026 (roșu). Barele de sîmbătă și duminică sînt cele mai joase în ultimele două perioade; sîmbăta din 2024–2026 are cea mai mică abatere standard din grafic.

În medie, weekendul nu diferă: coeficientul $b$ este −0,01 (p-value 0,95), −0,11 (0,48) și −0,03 (0,83) în cele trei perioade. În varianță, diferența este clară. Raportul dintre varianța din weekend și cea din zilele lucrătoare este 0,66, 0,35 și 0,39, iar testul Brown–Forsythe respinge egalitatea varianțelor în toate trei perioadele (p-value 0,002 în prima, sub 0,001 în celelalte).

Piața cripto este deci mai liniștită cînd piețele tradiționale sînt închise, iar diferența a crescut după 2020. Rezultatul sugerează că o parte importantă a variației prețului Bitcoin vine din fluxurile și știrile legate de sistemul financiar tradițional, nu doar din piața cripto însăși.

## @end

Activele cripto sînt de 3,8 pînă la 6,7 ori mai volatile decît S&P 500, cu aceleași fapte stilizate: asimetrie negativă și exces de boltire. Anualizarea trebuie făcută cu 365 de zile, altfel volatilitatea Bitcoin este subestimată cu 16,9%. Weekendul nu diferă în medie, dar varianța din weekend este mult mai mică, mai ales după 2020.

# Cozi și volatility clustering

Secțiunea măsoară cozile distribuțiilor cu estimatorul Hill, testează dependența randamentelor și a pătratelor lor cu ACF și testul Ljung–Box și estimează modele GARCH(1,1)-t și GJR pentru două active cripto și două active tradiționale.

## Cozi groase: estimatorul Hill (1/2) {h=34f3a09c}

O distribuție are cozi groase (*heavy tails*) de tip putere dacă $P(X > x) \approx c\,x^{-\alpha}$ pentru $x$ mare, unde $c$ este o constantă, iar $\alpha$ este tail index-ul. Un $\alpha$ mai mic înseamnă o coadă mai groasă. Momentele de ordin $\ge \alpha$ sînt infinite: $\alpha < 4$ înseamnă o boltire infinită, iar $\alpha < 2$ o varianță infinită.

Estimatorul lui @Hill1975, introdus în [Capitolul 5](ch5_ro.html), folosește cele mai mari $k$ pierderi (sau cîștiguri):

$$ \hat\alpha = \Big[\frac1k\sum_{i=1}^k \ln\frac{X_{(i)}}{X_{(k+1)}}\Big]^{-1}, \qquad CI_{95\%} = \hat\alpha\Big(1 \pm \frac{1{,}96}{\sqrt{k}}\Big). $$ {#eq:hill}

Aici $X_{(1)} \ge X_{(2)} \ge \dots$ sînt pierderile (sau cîștigurile) ordonate descrescător, $X_{(k+1)}$ este pragul cozii, iar CI (*confidence interval*) este intervalul de încredere. Pentru o lege de putere, logaritmul raportului dintre o valoare extremă și prag are media $1/\alpha$, de aceea estimatorul este inversul acestei medii. Capitolul folosește $k = 2{,}5\%$ din numărul de observații $n$.

## Cozi groase: estimatorul Hill (2/2) {h=944bcd35}

Tabelul dă tail index-ul cozii stîngi (pierderile) și al cozii drepte (cîștigurile) pentru cele cinci active, cu intervalele de încredere de 95%. Pentru Bitcoin, coada stîngă are $\hat\alpha = 2{,}71$, cu intervalul $[2{,}20; 3{,}21]$, iar coada dreaptă $\hat\alpha = 3{,}37$, cu intervalul $[2{,}74; 4{,}00]$: pierderile extreme sînt mai grele decît cîștigurile extreme, în acord cu asimetria negativă.

Toți indicii cozii stîngi sînt sub 4: 2,62 pentru Ethereum, 2,84 pentru Solana, 2,68 pentru S&P 500 și 3,11 pentru aur. Dacă boltirea populației este infinită, boltirea de selecție din tabelul celor cinci active nu estimează o valoare stabilă: ea crește cu eșantionul și depinde de cîteva zile extreme. Excesele de boltire de 12,0 sau 15,8 descriu deci eșantionul, nu un parametru.

Intervalele pentru activele cripto și pentru acțiuni se suprapun. Diferența mare dintre Bitcoin și S&P 500 este deci în nivelul volatilității, nu în forma cozii: după rescalare, pierderile extreme ale celor două piețe au o grosime comparabilă.

## Interpretarea ACF și a testului Ljung–Box {h=29bc019e}

Funcția de autocorelație (ACF, *autocorrelation function*) $\hat\rho(k)$ este corelația de selecție dintre $r_t$ și $r_{t-k}$, unde $k$ este lagul, în zile. În ipoteza de independență, $\hat\rho(k)$ se află în intervalul $\pm 1{,}96/\sqrt{n}$ cu probabilitatea 95%, unde $n$ este numărul de randamente. ACF a lui $r_t$ întreabă dacă direcția mișcării este previzibilă; ACF a lui $r_t^2$ întreabă dacă mărimea mișcării este previzibilă, adică dacă există volatility clustering.

Testul Ljung–Box combină primele 10 autocorelații într-o singură statistică:

$$ Q(10) = n(n + 2)\sum_{k=1}^{10}\frac{\hat\rho(k)^2}{n - k}. $$ {#eq:lb}

În absența autocorelației, $Q(10)$ urmează o distribuție $\chi^2$ cu 10 grade de libertate. Un $Q(10)$ mare, cu un p-value mic, arată că autocorelațiile sînt semnificative împreună, chiar dacă fiecare în parte este mică. Testul presupune însă varianță constantă; cu volatility clustering, el respinge prea des, un punct la care capitolul revine în [Secțiunea 4](#sec-4).

## Volatility clustering la Bitcoin {h=0dad2b8c}

Graficul are două panouri cu ACF pentru lagurile 1–30: în stînga ACF a randamentelor $r_t$ ale Bitcoin, în dreapta ACF a pătratelor $r_t^2$. Liniile întrerupte marchează banda $\pm 1{,}96/\sqrt{n} = \pm 0{,}030$.

Pentru randamente, $\hat\rho(1) = -0{,}022$, iar doar 4 din 30 de laguri ies din bandă, cu valori mici. Pentru pătrate, $\hat\rho(1) = 0{,}137$, iar 15 din 30 de laguri ies din bandă, toate pozitive. Statistica Ljung–Box $Q(10)$ este 20,2 pentru $r_t$ (p-value 0,028) și 212,7 pentru $r_t^2$ (p-value sub 0,001).

Diferența de ordin de mărime dintre cele două statistici rezumă rezultatul: direcția mișcării este greu de prognozat, mărimea ei nu. Zilele agitate urmează după zile agitate, exact faptul stilizat din [Capitolul 8](ch8_ro.html), iar modelul care îl descrie este GARCH.

## GARCH(1,1)-t pentru active cripto și tradiționale (1/2) {h=1881f0aa}

În modelul lui @Bollerslev1986 ([Capitolul 9](ch9_ro.html)), varianța de azi este o sumă ponderată a unei constante, a pătratului șocului de ieri și a varianței de ieri:

$$ \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2, \qquad \varepsilon_t = \sigma_t z_t. $$ {#eq:garch}

Aici $\sigma_t^2$ este varianța condiționată a zilei $t$, $\varepsilon_t$ șocul (randamentul centrat), iar $z_t$ urmează o distribuție Student-t cu $\nu$ grade de libertate, scalată la varianța 1. Parametrul $\omega > 0$ este constanta, $\alpha \ge 0$ reacția la șocuri, $\beta \ge 0$ memoria varianței trecute, iar $\alpha + \beta$ persistența. Un șoc asupra varianței se stinge geometric cu rata $\alpha + \beta$, iar timpul de înjumătățire (*half-life*), $\ln 0{,}5/\ln(\alpha + \beta)$ zile, este intervalul după care jumătate din efectul lui a dispărut.

Varianta GJR (Glosten–Jagannathan–Runkle) adaugă termenul $\gamma\,\varepsilon_{t-1}^2\mathbb{1}\{\varepsilon_{t-1} < 0\}$, unde indicatorul este 1 după o scădere. Un $\gamma > 0$ înseamnă că scăderile cresc volatilitatea mai mult decît creșterile de aceeași mărime, leverage effect-ul cunoscut la acțiuni.

## GARCH(1,1)-t pentru active cripto și tradiționale (2/2) {h=cd094968}

Tabelul dă estimările prin verosimilitate maximă pe randamentele logaritmice zilnice ale Bitcoin, Ethereum, S&P 500 și aurului. Pentru Bitcoin, $\hat\alpha = 0{,}109$ și $\hat\beta = 0{,}891$; pentru Ethereum, $\hat\alpha = 0{,}175$ și $\hat\beta = 0{,}825$. La ambele active cripto, persistența $\hat\alpha + \hat\beta$ este 1,000, deci timpul de înjumătățire nu este definit.

La activele tradiționale persistența este mare, dar sub 1: 0,993 pentru S&P 500, cu un timp de înjumătățire de 100 de zile, și 0,994 pentru aur, cu 113 zile. Gradele de libertate ale distribuției Student-t sînt 3,18 pentru Bitcoin, 3,14 pentru Ethereum, 5,50 pentru S&P 500 și 4,13 pentru aur.

Ultima coloană dă coeficientul GJR $\hat\gamma$ și statistica lui $t$. Este −0,013 ($t = -0{,}72$) pentru Bitcoin, 0,032 ($t = 0{,}91$) pentru Ethereum, 0,280 ($t = 7{,}08$) pentru S&P 500 și −0,051 ($t = -3{,}33$) pentru aur. Slide-ul următor interpretează aceste cifre.

## Interpretarea estimărilor GARCH {h=602d3bb4}

La activele cripto, $\hat\alpha + \hat\beta = 1$, la limita domeniului parametrilor: modelul estimat este un GARCH integrat (IGARCH). Modelul EWMA (*exponentially weighted moving average*) din [Capitolul 8](ch8_ro.html) este cazul particular $\omega = 0$. Într-un IGARCH, un șoc al volatilității nu se stinge în eșantion și nu există o varianță de termen lung finită spre care să revină volatilitatea. La S&P 500 și aur, în schimb, jumătate din efectul unui șoc dispare în 100 și, respectiv, 113 zile.

Cozile rămîn foarte groase și după filtrarea GARCH: $\hat\nu = 3{,}18$ pentru Bitcoin înseamnă inovații $z_t$ cu boltire infinită, față de $\hat\nu = 5{,}50$ pentru S&P 500. Volatility clustering explică o parte din cozile groase ale randamentelor, dar nu pe toate.

Activele cripto nu au leverage effect: $\hat\gamma = -0{,}013$ ($t = -0{,}72$) pentru Bitcoin, față de 0,280 ($t = 7{,}08$) pentru S&P 500. Aurul are un $\hat\gamma$ negativ și semnificativ ($t = -3{,}33$): creșterile prețului cresc volatilitatea mai mult decît scăderile, comportamentul unui activ de refugiu, căutat în perioadele de stres. @Katsiampa2017 compară mai multe modele de tip GARCH pentru Bitcoin și găsește că cel mai bine se potrivește un model cu o componentă de termen scurt și una de termen lung a volatilității, o altă formă a persistenței foarte mari.

## Volatilitatea condiționată: Bitcoin și S&P 500 {h=2f462c46}

Graficul arată volatilitatea condiționată $\hat\sigma_t$ din modelul GARCH(1,1)-t, anualizată cu $\sqrt{365}$ pentru Bitcoin (portocaliu) și cu $\sqrt{252}$ pentru S&P 500 (albastru închis), din 2014 pînă în 2026, în procente.

Ambele serii ating maximul în martie 2020: 301% pentru Bitcoin și 125% pentru S&P 500. Pe 18 septembrie 2026, modelul dă 42% pentru Bitcoin și 12% pentru S&P 500. Între crize, volatilitatea Bitcoin are vîrfuri dese și scurte, peste un nivel de bază care scade în timp, iar seria S&P 500 rămîne mult sub ea pe toată perioada.

Spre deosebire de volatilitatea pe ferestre mobile din [Secțiunea 2](#sec-2), estimarea GARCH reacționează din prima zi la un șoc mare și nu are salturi artificiale cînd o zi extremă iese din fereastră. Este estimarea folosită mai departe pentru VaR-ul condiționat din [Secțiunea 9](#sec-9).

## @end

Cozile stîngi ale tuturor celor cinci active au tail index sub 4, iar intervalele pentru cripto și pentru acțiuni se suprapun. Randamentele Bitcoin sînt aproape necorelate, dar pătratele lor sînt puternic autocorelate. Modelul GARCH(1,1)-t al activelor cripto este integrat, cu inovații cu cozi foarte groase și fără leverage effect, spre deosebire de S&P 500.

# Eficiența pieței cripto

Secțiunea aplică testele de eficiență în formă slabă din [Capitolul 7](ch7_ro.html) pe Bitcoin și Ethereum, pe întregul eșantion, pe subperioade și pe ferestre mobile, le compară cu literatura despre eficiența Bitcoin și arată, prin două studii de caz, ce pot vedea datele de tranzacționare și cele din blockchain dincolo de o singură serie de prețuri.

## Eficiența în formă slabă: ipotezele testate {h=706259f3}

Pe o piață eficientă în formă slabă [@Fama1970], prețurile trecute nu ajută la prognoza randamentelor viitoare. Celelalte două forme cer mai mult: forma semi-tare se referă la toată informația publică, iar forma tare la toată informația, inclusiv cea privată. Testele statistice pe o serie de prețuri privesc doar forma slabă.

Capitolul folosește trei instrumente: autocorelația de ordinul întîi $\hat\rho(1)$, statistica Ljung–Box $Q(10)$ din @eq:lb și raportul varianțelor (VR, *variance ratio*) al lui @LoMacKinlay1988:

$$ \mathrm{VR}(q) = \frac{\mathrm{Var}(r_t + \dots + r_{t-q+1})}{q\,\mathrm{Var}(r_t)}. $$ {#eq:vr}

Aici $q$ este orizontul în zile, iar numărătorul este varianța randamentului logaritmic pe $q$ zile. Fără autocorelație, această varianță este de $q$ ori cea zilnică, deci VR este 1 pentru un mers aleator. Un VR peste 1 indică momentum (tendințe), iar un VR sub 1 revenire la medie (*mean reversion*).

Statistica de test robustă este $Z^*(q) = (\widehat{\mathrm{VR}}(q) - 1)/\sqrt{\hat\theta^*(q)}$, unde $\hat\theta^*$ este varianța lui $\widehat{\mathrm{VR}}$ robustă la volatility clustering. Pentru un mers aleator, $Z^*$ are aproximativ distribuția $N(0,1)$, iar ipoteza este respinsă la 5% dacă $|Z^*| > 1{,}96$.

## Testele raportului varianțelor pe întregul eșantion {h=3546e305}

Primul tabel dă rezultatele pe întregul eșantion. Pentru Bitcoin, $\hat\rho(1) = -0{,}022$, $Q(10) = 20{,}2$ (p-value 0,028), VR(2), VR(5) și VR(10) sînt 0,977, 0,991 și 1,027, iar $Z^*(5) = -0{,}19$ (p-value 0,853). Ethereum are valori asemănătoare, cu $Z^*(5) = 1{,}17$ (p-value 0,241). S&P 500 are o autocorelație negativă mai mare, $\hat\rho(1) = -0{,}126$, rapoarte VR sub 1 (VR(10) = 0,813) și $Q(10) = 236{,}1$, dar și aici $Z^*(5) = -1{,}28$ (p-value 0,201).

Al doilea tabel împarte eșantioanele în subperioade: 2014–2017, 2018–2021 și 2022–2026 pentru Bitcoin, 2016–2017, 2018–2021 și 2022–2026 pentru Ethereum. VR(5) rămîne între 0,975 și 1,127, iar niciun $Z^*(5)$ nu depășește 1,96 în valoare absolută; cel mai mare este 1,03, pentru Ethereum în 2016–2017.

Niciun test VR nu respinge deci mersul aleator, în nicio subperioadă. Testul Ljung–Box respinge slab la activele cripto (p-value 0,028 și 0,012) deoarece presupune varianță constantă: volatility clustering umflă variabilitatea autocorelațiilor de selecție, iar banda $\pm 1{,}96/\sqrt{n}$ devine prea îngustă. Statistica robustă $Z^*$ corectează acest efect.

## Eficiența în timp {h=53775d2f}

Graficul arată statistica robustă $Z^*(5)$ pentru Bitcoin (portocaliu), Ethereum (mov) și S&P 500 (albastru închis), calculată pe ferestre de 500 de observații, cu o fereastră nouă la fiecare 21 de observații, ca în [Capitolul 7](ch7_ro.html). Liniile întrerupte orizontale marchează valorile critice $\pm 1{,}96$.

Toate cele trei serii rămîn în interiorul benzii pe întreaga perioadă 2016–2026: la Bitcoin, 0,0% din cele 185 de ferestre resping mersul aleator. Statistica oscilează în jurul lui zero, fără o tendință spre vreuna dintre margini.

Contrastul vine din statistica i.i.d. $Z(5)$, care presupune varianță constantă. Cu ea, proporția ferestrelor cu respingere ar fi 1,6% pentru Bitcoin și 21,7% pentru S&P 500. Ignorarea volatility clustering produce deci respingeri false, mai ales pe piața de acțiuni, unde autocorelația negativă de ordinul întîi coincide cu perioadele de volatilitate mare.

## Evoluția eficienței {h=7f787da7}

Literatura despre eficiența Bitcoin a pornit de la un verdict negativ. @Urquhart2016 găsește că Bitcoin a fost ineficient în perioada 2010–2016 ca întreg, dar s-a apropiat de eficiență după 2013. @NadarajahChu2017 arată că o transformare de tip putere a acelorași randamente trece testele de eficiență, deci verdictul inițial era fragil. @TranLeirvik2020 propun o măsură ajustată a ineficienței și găsesc că piețele cripto au devenit mai eficiente după 2017.

Datele capitolului, care încep în 2014, sînt în acord cu această evoluție: VR(5) variază între 0,78 și 1,17 pe ferestrele mobile, fără nicio respingere robustă. Piața s-a extins, au intrat arbitrajorii, au apărut derivatele și ETF-urile. Gradul de eficiență se schimbă odată cu participanții, perspectiva piețelor adaptive din [Capitolul 11](ch11_ro.html).

Eficiența în formă slabă nu este însă eficiență în general. Prețurile pot fi imprevizibile din istoria lor și totuși departe de orice valoare fundamentală: un mers aleator nu spune nimic despre nivelul la care se află prețul. Studiile de caz care urmează arată tipuri de ineficiență pe care un test pe o singură serie nu le poate vedea.

## Studiu de caz: tranzacționare și arbitraj între exchange-uri {h=9da10752}

@MakarovSchoar2020 folosesc date tick-by-tick de pe 34 de exchange-uri din 19 țări, din perioada 2017–2018. Întrebarea lor privește arbitrajul (*arbitrage*): cumpărarea acolo unde prețul este mic și vînzarea acolo unde este mare. Pe o piață eficientă, astfel de diferențe se închid repede, pentru că arbitrajorii le exploatează.

Autorii găsesc diferențe de preț mari și persistente între țări, în special între Coreea, Japonia și SUA, și mult mai mici în interiorul aceleiași țări. Diferențele se deschid cînd Bitcoin crește, iar controlul capitalurilor face arbitrajul dificil. În același timp, o componentă comună explică cea mai mare parte a fluxului de ordine: prețurile sînt determinate de o piață globală, chiar dacă nivelul lor diferă între țări.

Lecția este că legea prețului unic nu funcționează acolo unde banii nu circulă liber. Un test statistic aplicat unei singure serii agregate nu poate vedea această ineficiență, care apare doar în comparația dintre piețe.

## Studiu de caz: este Bitcoin cu adevărat independent de Tether? {h=0bd54986}

@GriffinShams2020 folosesc înregistrările din blockchain ale Bitcoin și ale stablecoin-ului Tether (USDT) din 2017–2018. Întrebarea lor este dacă USDT nou emis a fost folosit pentru a cumpăra Bitcoin după scăderi de preț, adică pentru a susține prețul.

Rezultatele arată că cumpărările făcute cu Tether au venit după scăderi ale pieței și au fost urmate de creșteri ale prețului. Fluxurile proveneau în principal de la un singur cont mare de pe un singur exchange, Bitfinex, iar orele cu astfel de fluxuri explică o parte mare a creșterii prețului Bitcoin din 2017.

Un rezultat asemănător este manipularea de pe exchange-ul Mt. Gox din 2013, documentată de @GandalEtAl2018. Ambele studii folosesc o trăsătură proprie pieței cripto: tranzacțiile sînt publice în blockchain, ceea ce permite teste imposibile pe piețele de acțiuni, unde identitatea cumpărătorilor nu este vizibilă.

## @end

Testele robuste ale raportului varianțelor nu resping mersul aleator pentru Bitcoin și Ethereum, nici pe întregul eșantion, nici pe subperioade sau ferestre mobile; respingerile testului Ljung–Box și ale statisticii i.i.d. vin din volatility clustering. Eficiența în formă slabă nu exclude însă diferențe persistente de preț între țări sau manipulări vizibile în datele din blockchain.

# Corelația cu acțiunile și aurul

Secțiunea măsoară legătura dintre activele cripto și activele tradiționale. Începe cu problema calendarelor diferite, urmărește corelația pe ferestre mobile și pe ani, compară trei crize și testează dacă Bitcoin este o acoperire sau un activ de refugiu pentru un portofoliu de acțiuni.

## Măsurarea corelației între calendare diferite {h=8d92e293}

Bitcoin se tranzacționează șapte zile pe săptămînă, S&P 500 și aurul cinci. Ordinea operațiilor contează: întîi se unesc prețurile în zilele comune, apoi se calculează randamentele. Astfel, randamentul Bitcoin de luni acoperă perioada vineri–luni, ca și randamentul S&P 500. Dacă s-ar uni direct randamentele, randamentul Bitcoin de luni ar acoperi doar duminică–luni, iar mișcările din weekend ar dispărea din analiză.

A doua problemă sînt închiderile asincrone: Bitcoin se închide la miezul nopții UTC, S&P 500 la ora 16:00 la New York. O știre de după închiderea bursei americane intră în randamentul Bitcoin din aceeași zi și în randamentul S&P 500 din ziua următoare, ceea ce reduce corelația zilnică. Randamentele săptămînale (vineri–vineri) reduc această problemă, de aceea capitolul le raportează pe ambele.

Corelația mobilă (*rolling correlation*) este corelația de selecție pe ultimele 250 de zile comune, recalculată în fiecare zi. Eroarea ei standard este aproximativ $(1 - \rho^2)/\sqrt{250} \approx 0{,}06$ cînd corelația adevărată $\rho$ este mică, deci mișcările sub 0,1 sînt zgomot.

## Corelația mobilă cu acțiunile și aurul {h=1d02038b}

Graficul arată trei corelații mobile pe 250 de zile comune, calculate din randamentele logaritmice zilnice: Bitcoin cu S&P 500 (albastru închis), Ethereum cu S&P 500 (mov) și Bitcoin cu aurul (maro). Liniile verticale întrerupte marchează COVID-19 (martie 2020), prăbușirea Terra/Luna (mai 2022), falimentul exchange-ului cripto FTX (noiembrie 2022) și lansarea ETF-urilor spot (ianuarie 2024).

Înainte de 2020, Bitcoin și S&P 500 erau practic necorelate: corelația pe întreaga perioadă de dinainte de 2020 este 0,02, iar seria mobilă oscilează în jurul lui zero. Din martie 2020, corelația sare brusc și rămîne pozitivă; media perioadei de după 2020 este 0,39, iar maximul, 0,57, a apărut pe 9 februarie 2023. Ethereum urmează îndeaproape aceeași traiectorie.

Corelația cu aurul a rămas slabă pe cea mai mare parte a eșantionului. În 2026 ea a crescut, iar seria mobilă a ajuns la 0,37 în septembrie 2026; eroarea standard de 0,06 arată că această creștere depășește zgomotul de selecție, dar o singură creștere recentă nu este încă un tipar stabil.

## Corelația an cu an {h=32bf1948}

Tabelul dă corelația randamentelor zilnice pe fiecare an între 2019 și 2025. Corelația Bitcoin cu S&P 500 este −0,11 în 2019, 0,45 în 2020, 0,26 în 2021, 0,56 în 2022, 0,16 în 2023, 0,36 în 2024 și 0,42 în 2025. Ethereum are valori aproape identice, iar corelația Bitcoin cu aurul rămîne între −0,02 și 0,21.

Pe întreaga perioadă, corelația dintre Bitcoin și S&P 500 este 0,24 cu randamente zilnice și 0,19 cu randamente săptămînale. Valorile apropiate arată că închiderile asincrone nu explică corelația: dacă ar fi un artefact al orelor de închidere, corelația săptămînală ar fi mult mai mare decît cea zilnică.

Din 2020, activele cripto se comportă ca active riscante: au scăzut odată cu acțiunile cînd dobînzile au crescut, în 2022, anul cu cea mai mare corelație. Schimbarea coincide cu intrarea investitorilor instituționali pe piața cripto. Beneficiul de diversificare măsurat pe datele de dinainte de 2020, inclusiv în studiile din [Secțiunea 2](#sec-2), nu se mai păstrează.

## Trei episoade de criză {h=4ad0076f}

Tabelul compară randamentele logaritmice cumulate, în procente, de la prima la ultima închidere a trei episoade. În criza COVID-19 (19 februarie – 23 martie 2020), Bitcoin a pierdut 40,6%, Ethereum 65,5%, S&P 500 41,4%, iar aurul a cîștigat 0,3%; cea mai proastă zi a Bitcoin a avut un randament de −46,5%. În prăbușirea Terra/Luna (5–12 mai 2022), Bitcoin a pierdut 23,0%, Ethereum 33,8%, S&P 500 5,4%, iar aurul 3,0%.

Al treilea episod este falimentul FTX, un exchange cripto care a intrat în insolvență în noiembrie 2022 (5–21 noiembrie 2022). Bitcoin a pierdut 29,9%, iar Ethereum 38,4%, în timp ce S&P 500 a crescut cu 4,6%, iar aurul cu 3,4%.

Comparația arată o dependență asimetrică. COVID-19 a fost un șoc global: activele cripto au scăzut cît acțiunile, iar aurul nu s-a mișcat. Terra/Luna și FTX au fost șocuri specifice pieței cripto: activele cripto au pierdut un sfert sau mai mult, iar S&P 500 mult mai puțin sau deloc. Crahurile acțiunilor ajung la cripto, crahurile cripto rareori ajung la acțiuni.

## Bitcoin ca instrument de acoperire sau activ de refugiu {h=2be0b8b4}

@BaurLucey2010 disting două roluri ale unui activ într-un portofoliu de acțiuni. O acoperire (*hedge*) este necorelată sau corelată negativ cu acțiunile în medie. Un activ de refugiu (*safe haven*) are această proprietate în timpul crahurilor, adică exact atunci cînd investitorul are nevoie de ea. Un activ poate fi o acoperire slabă în medie fără să fie un refugiu.

Testul capitolului se uită la randamentul mediu în cele mai proaste 5% dintre zilele S&P 500: 151 de zile comune, în care S&P 500 a scăzut cu peste 1,65%, în medie cu 2,70%. În aceste zile, Bitcoin a avut un randament mediu de −2,77% ($t = -4{,}89$, p-value sub 0,001) și a fost negativ în 70% dintre ele. Aurul a avut un randament mediu de 0,10% ($t = 0{,}86$, p-value 0,394), adică nu a pierdut.

Bitcoin nu este deci un activ de refugiu pe acest eșantion: în cele mai proaste zile ale acțiunilor, scade cît S&P 500, în timp ce aurul își păstrează valoarea. Narațiunea „aurului digital” nu este susținută de date.

## @end

Corelația Bitcoin cu S&P 500 a trecut de la aproximativ 0 înainte de 2020 la 0,39 după, iar valoarea săptămînală apropiată de cea zilnică arată că schimbarea nu este un artefact al orelor de închidere. Dependența este asimetrică: crizele globale lovesc activele cripto, crizele cripto lovesc rar acțiunile. În zilele de crah ale acțiunilor, Bitcoin scade odată cu ele, deci nu este un activ de refugiu.

# Bule și crahuri

Secțiunea măsoară pierderile mari ale activelor cripto prin drawdown, compară Bitcoin cu Ethereum și S&P 500, enumeră marile scăderi ale Bitcoin și prezintă un model al bulelor speculative, legea de putere log-periodică.

## Drawdown-ul {h=411e5760}

Drawdown-ul este pierderea față de ultimul maxim istoric ([Capitolul 1](ch1_ro.html)):

$$ D_t = \frac{P_t}{\max_{s \le t} P_s} - 1, $$ {#eq:dd}

unde $P_t$ este prețul din ziua $t$, iar $\max_{s \le t} P_s$ cel mai mare preț pînă în ziua $t$; prin construcție, $D_t \le 0$. Drawdown-ul maxim este $\min_t D_t$, iar perioada underwater (*time under water*) este numărul de zile pînă la un nou maxim istoric.

O pierdere $d$ cere un cîștig $g = d/(1 - d)$ pentru revenirea la vîrful anterior, deoarece $(1 - d)(1 + g) = 1$. Relația este puternic neliniară: o pierdere de 75% cere un cîștig de $0{,}75/0{,}25 = 300\%$, iar una de 80% un cîștig de 400%. Cel mai adînc drawdown al Bitcoin în datele capitolului, −83,4% pe 15 decembrie 2018, a cerut un cîștig de 502% pentru recuperare.

Drawdown-ul depinde de traiectorie, nu doar de distribuția randamentelor: aceleași randamente zilnice, în altă ordine, dau drawdown-uri diferite. Volatilitatea mare și volatility clustering, care grupează zilele proaste, produc drawdown-uri adînci, de aceea activele cripto au drawdown-uri mult mai mari decît acțiunile.

## Drawdown-urile Bitcoin, Ethereum și S&P 500 {h=315a397b}

Graficul arată drawdown-ul zilnic $D_t$, în procente, al Bitcoin (portocaliu), Ethereum (mov) și S&P 500 (albastru închis), calculat din închiderile zilnice din 2016. Fiecare curbă stă la 0 în zilele cu un nou maxim istoric și coboară în perioadele de pierdere.

Drawdown-urile maxime sînt −83,4% pentru Bitcoin, −94,0% pentru Ethereum și −33,9% pentru S&P 500. Curbele activelor cripto au două văi adînci și lungi, în 2018–2020 și în 2022–2023, în timp ce S&P 500 are o singură scădere bruscă în martie 2020 și o scădere mai lentă în 2022.

Diferența mai mare este în timpul petrecut sub maxim. Bitcoin a petrecut 76% din zile la peste 10% sub maximul istoric, Ethereum 90%, iar S&P 500 16%. Pe 18 septembrie 2026, Bitcoin era cu 35,2% sub maximul istoric. Pentru un investitor în cripto, starea obișnuită este deci o pierdere față de vîrf, nu o excepție.

## Marile drawdown-uri ale Bitcoin {h=84314059}

Tabelul enumeră episoadele din septembrie 2014 în care Bitcoin a pierdut peste 50% față de maxim, cu data și prețul maximului, data și prețul minimului, adîncimea, data noului maxim și numărul de zile underwater. Cea mai adîncă scădere a început de la 19 497 USD pe 16 decembrie 2017 și a ajuns la 3 237 USD pe 15 decembrie 2018 (−83,4%); noul maxim a venit pe 30 noiembrie 2020, după 1080 de zile.

Al doilea episod lung a pornit de la 67 567 USD pe 8 noiembrie 2021 și a ajuns la 15 787 USD pe 21 noiembrie 2022 (−76,6%), cu recuperare după 847 de zile, pe 4 martie 2024. Episoadele mai scurte sînt cel din 2014–2015 (−61,1%, 454 de zile) și cel din 2021 (−53,1%, 189 de zile). Ultimul episod a început de la 124 753 USD pe 6 octombrie 2025 și a coborît la 58 559 USD pe 30 iunie 2026 (−53,1%), fără un nou maxim pînă la sfîrșitul eșantionului.

Cinci scăderi de peste jumătate din valoare în doisprezece ani arată că un crah de această mărime este, pentru Bitcoin, un eveniment obișnuit, nu unul rar. Maximele din 2017 și 2021 au cerut, fiecare, mai mult de doi ani pentru recuperare.

## Bulele și modelele log-periodice {h=1a3cbe7a}

O bulă (*bubble*) este un preț mult peste valoarea fundamentală, susținut de așteptarea că activul va putea fi revîndut la un preț și mai mare. @CheahFry2015 susțin că valoarea fundamentală a Bitcoin este zero și găsesc în prețul lui semnele unor bule speculative.

Legea de putere log-periodică (LPPL, *log-periodic power law*) a lui @JohansenLedoitSornette2000 descrie traiectoria prețului înaintea unui crah:

$$ \ln P_t \approx A + B(t_c - t)^m + C(t_c - t)^m\cos\big(\omega\ln(t_c - t) - \phi\big). $$ {#eq:lppl}

Aici $t_c$ este momentul critic, data cea mai probabilă a crahului, iar $A$ logaritmul prețului atins la $t_c$ (nu numărul de perioade pe an din @eq:annual). Cu $B < 0$ și $0 < m < 1$, prețul crește mai repede decît exponențial pe măsură ce $t$ se apropie de $t_c$. Termenul cu cosinus adaugă oscilații cu amplitudinea $C$, frecvența $\omega$ și faza $\phi$, tot mai dese spre $t_c$, deoarece argumentul depinde de $\ln(t_c - t)$.

Modelul are șapte parametri, iar funcția de ajustat are multe optime locale; dintre parametri, $t_c$ este cel mai incert. În practică, ajustările LPPL par convingătoare mai ales după crah, cînd data lui este cunoscută. Un model care prognozează crahuri trebuie judecat pe prognoze făcute înainte de eveniment.

## @end

Drawdown-urile de peste 50% sînt obișnuite pentru Bitcoin: cinci în doisprezece ani, cu perioade de recuperare de pînă la 1080 de zile, iar Bitcoin a petrecut 76% din zile la peste 10% sub maxim. Bulele pot fi descrise de modele precum LPPL, dar momentul crahului rămîne parametrul cel mai greu de estimat.

# ETF-urile spot pe Bitcoin

Secțiunea studiază un eveniment instituțional, aprobarea ETF-urilor spot pe Bitcoin în ianuarie 2024. Întrebarea este dacă volatilitatea Bitcoin, tiparul ei de weekend și legătura cu acțiunile s-au schimbat după intrarea activului la bursă și cît de bine urmărește un astfel de fond prețul Bitcoin.

## Ianuarie 2024: aprobarea ETF-urilor spot pe Bitcoin {h=21070879}

Un ETF (*exchange-traded fund*) este un fond ale cărui unități se tranzacționează la bursă, ca acțiunile. Un ETF spot deține activul însuși, nu contracte futures pe acel activ. Pe 10 ianuarie 2024, SEC (*Securities and Exchange Commission*), autoritatea americană a pieței de capital, a aprobat 11 ETF-uri spot pe Bitcoin [@SEC2024], iar tranzacționarea a început pe 11 ianuarie.

Momentul este important din două motive. Fondurile de pensii și consultanții financiari pot deține Bitcoin printr-un instrument reglementat, fără să cumpere moneda direct de pe un exchange cripto. În plus, unitățile ETF se tranzacționează doar în zilele lucrătoare, în orele bursei americane, astfel încît o parte a tranzacționării se mută în aceste ore.

Întrebarea statistică este dacă aceste schimbări se văd în date: s-a schimbat volatilitatea Bitcoin, s-a schimbat tiparul ei de weekend din [Secțiunea 2](#sec-2), s-a schimbat legătura cu acțiunile din [Secțiunea 5](#sec-5)?

## Volatilitatea și ponderea weekendului în jurul ETF-urilor {h=da556f33}

Graficul are două panouri. În stînga, volatilitatea Bitcoin pe ferestre de 90 de zile, anualizată cu $\sqrt{365}$, în procente, din 2020 pînă în 2026, cu o linie verticală întreruptă la 11 ianuarie 2024, data lansării ETF-urilor. În dreapta, ponderea weekendului în suma pătratelor randamentelor zilnice, în procente, pentru fiecare an din 2015 pînă în 2026, cu o linie orizontală la 2/7, ponderea pe care ar avea-o weekendul dacă fiecare zi ar avea aceeași varianță.

Volatilitatea a continuat să scadă după ianuarie 2024, dar era deja în scădere înainte: vîrfurile din 2020–2022 sînt urmate de un nivel mult mai jos în a doua jumătate a lui 2023. O scădere după eveniment nu poate fi atribuită evenimentului cînd tendința exista dinainte.

Ponderea weekendului este sub 2/7 în fiecare an, iar din 2020 este mult sub această valoare: barele din 2020–2026 sînt, cu excepția lui 2025, mult mai joase decît cele din 2015–2019. Schimbarea tiparului de weekend a avut loc deci în 2020, cu patru ani înainte de ETF-uri.

## Bitcoin înainte și după lansarea ETF-urilor spot {h=81cfb6ba}

Tabelul compară două ferestre de doi ani: 11 ianuarie 2022 – 10 ianuarie 2024 (729 de zile) și 11 ianuarie 2024 – 10 ianuarie 2026 (730 de zile). Volatilitatea anuală a scăzut de la 55,2% la 47,5%, raportul dintre varianța din weekend și cea din zilele lucrătoare a crescut de la 0,30 la 0,42, corelația cu S&P 500 a scăzut de la 0,45 la 0,37, iar persistența GARCH(1,1)-t $\hat\alpha + \hat\beta$ a scăzut de la 1,000 la 0,97.

Testul Brown–Forsythe al egalității varianțelor dă statistica 1,22, cu p-value 0,27: scăderea volatilității nu este semnificativă statistic. Cu volatility clustering și cu cozi groase, o diferență de opt puncte procentuale între două ferestre de doi ani este în limitele variației obișnuite.

Chiar și o diferență semnificativă nu ar fi un efect cauzal. Între cele două ferestre s-au schimbat în același timp dobînzile, a avut loc halving-ul din aprilie 2024 și au avut loc alegerile din SUA. Un studiu credibil ar avea nevoie de un grup de control, de exemplu monede fără ETF, sau de date intrazilnice în jurul deschiderii bursei americane, unde efectul tranzacționării ETF ar trebui să fie vizibil direct.

## Eroarea de replicare a unui ETF spot pe Bitcoin {h=23b5f162}

Capitolul compară IBIT (iShares Bitcoin Trust), unul dintre ETF-urile spot, cu Bitcoin. Se folosește închiderea ajustată a IBIT, iar prețurile sînt unite în cele 671 de zile de tranzacționare ale ETF-ului. Corelația randamentelor zilnice este 0,91, iar panta regresiei randamentului IBIT pe randamentul Bitcoin este 0,94.

Tracking error-ul este abaterea standard anualizată a diferenței dintre randamentul fondului și cel al activului urmărit; aici este 20,8%, o valoare mare pentru un fond care deține Bitcoin unu la unu. Explicația este calendarul: cele două închideri sînt la patru ore distanță, ora 16:00 la New York pentru IBIT și miezul nopții UTC pentru Bitcoin. Mișcările din aceste patru ore apar în randamentul Bitcoin dintr-o zi și în randamentul IBIT din ziua următoare.

Pe o săptămînă, diferența se compensează, iar fondul urmărește prețul Bitcoin minus un comision mic. Un tracking error mare poate fi deci un artefact de măsurare. Înainte de a judeca un fond, orele de închidere trebuie aliniate, aceeași problemă ca la corelațiile din [Secțiunea 5](#sec-5).

## @end

Volatilitatea Bitcoin a scăzut după lansarea ETF-urilor spot, dar scăderea începuse înainte și nu este semnificativă statistic, iar ponderea weekendului se redusese deja din 2020. O comparație înainte–după nu identifică un efect cauzal. IBIT urmărește Bitcoin aproape unu la unu, iar tracking error-ul zilnic mare vine din decalajul de patru ore dintre închideri.

# Stablecoins

Secțiunea tratează activele cripto concepute să nu se miște: stablecoins. Definește cele trei tipuri, explică arbitrajul care menține paritatea, măsoară oferta și abaterile de la paritate, analizează două episoade în care paritatea s-a rupt și descrie cadrul de reglementare din Uniunea Europeană și din SUA.

## Stablecoins: definiție și tipuri {h=8df41936}

Un stablecoin este un activ cripto conceput să păstreze un preț fix, de obicei 1 USD; acest preț se numește paritate (*peg*). Stablecoins sînt folosite ca numerar pe exchange-urile cripto și pentru plăți transfrontaliere. Tipurile se deosebesc prin mecanismul care menține paritatea și, de aici, prin riscul principal.

Stablecoins garantate cu monedă fiduciară (*fiat-backed*), ca USDT (Tether) și USDC (USD Coin), țin rezerve în dolari, depozite bancare și titluri de stat; deținătorii pot răscumpăra la emitent o monedă pentru 1 USD. Riscul lor ține de rezerve și de banca la care sînt păstrate. Stablecoins garantate cu active cripto (*crypto-backed*), ca DAI, sînt create prin credite supra-garantate cu active cripto, lichidate automat de contracte inteligente cînd garanția scade; riscul lor este o prăbușire a garanției.

Stablecoins algoritmice, ca UST (TerraUSD), nu au rezerve: un algoritm emite și retrage din circulație (*burn*) o a doua monedă pentru a ține prețul la 1 USD. Riscul lor este o retragere masivă (*run*) care distruge ambele monede. DefiLlama clasifică astăzi 91,3% din oferta de stablecoins ca garantată fiat și 8,7% ca garantată cripto.

## Menținerea parității: arbitrajul {h=768f963f}

Paritatea unui stablecoin garantat fiat se menține pe piața primară, unde tranzacționari autorizați pot crea o monedă plătind 1 USD emitentului sau pot răscumpăra o monedă pentru 1 USD. Dacă prețul pe exchange-uri este 0,99, un arbitrajor cumpără la 0,99 și răscumpără la 1,00, cu un profit de un cent pe monedă; cererea lui împinge prețul înapoi în sus. Dacă prețul este 1,01, arbitrajorul creează monede la 1,00 și le vinde la 1,01; oferta lui împinge prețul în jos.

@LyonsViswanathNatraj2023 arată că paritatea Tether se menține prin acest arbitraj și că apar prime peste paritate cînd investitorii se refugiază în stablecoins în perioadele de scădere a pieței cripto. @MaZengZhang2025 studiază structura arbitrajului: cu puțini arbitrajori autorizați, retragerile masive sînt mai puțin probabile, dar abaterile de la paritate durează mai mult, pentru că arbitrajul este mai lent.

Arbitrajul funcționează doar cîtă vreme investitorii cred că răscumpărarea la 1 USD va fi onorată. O paritate este o promisiune, iar valoarea ei depinde de încrederea în rezerve; cele două episoade de mai jos arată ce se întîmplă cînd această încredere dispare.

## Oferta de stablecoins {h=3af626af}

Graficul arată oferta în circulație, în miliarde USD, din 2019 pînă în 2026, după datele DefiLlama, pentru toate stablecoin-urile denominate în USD (roșu), pentru USDT (verde), USDC (albastru închis) și DAI (portocaliu).

Oferta totală a crescut de la 4,2 miliarde USD la începutul lui 2020 la 308 miliarde pe 18 septembrie 2026, de 74 de ori mai mult. USDT deține 60% din total, iar USDC 24%; DAI este mic în comparație cu ambele.

Creșterea nu a fost continuă. Totalul a scăzut de la 187 de miliarde pe 2 aprilie 2022 la 123 de miliarde pe 19 august 2023, după prăbușirea Terra și falimentul FTX, iar oferta de USDC a scăzut de la 43 la 24 de miliarde în 2023, anul episodului SVB. Din 2024, oferta a crescut din nou, în principal prin USDT.

## Abaterea de la paritate în puncte de bază {h=2e7335c5}

Un punct de bază (bp, *basis point*) este o sutime de procent: $0{,}01\% = 0{,}0001$. Pentru o monedă cu paritatea 1 USD și prețul $P_t$ în USD în ziua $t$, abaterea de la paritate (*peg deviation*) este

$$ d_t = 10\,000\,(P_t - 1) \text{ bp}. $$ {#eq:peg}

Un $d_t$ negativ înseamnă că moneda se tranzacționează sub paritate, unul pozitiv că se tranzacționează peste. În exemplul lucrat, USDC s-a închis la 0,9715 USD pe 11 martie 2023, deci $d = 10\,000 \times (0{,}9715 - 1) = -285$ bp, o pierdere de 2,85%. Un deținător de 1 000 000 USDC care a vîndut la acea închidere a pierdut 28 500 USD; la minimul zilei, 0,8774 USD, pierderea ar fi fost 122 600 USD.

O paritate se descrie prin cîteva statistici: abaterea absolută mediană, proporția zilelor cu abateri de peste 10 sau 50 bp, cea mai proastă închidere și cel mai prost minim. Viteza revenirii la paritate se măsoară cu un model AR(1) (*autoregressive*), $d_t = \phi\,d_{t-1} + e_t$, unde $\phi \in (0,1)$ este fracțiunea din abaterea de azi care persistă mîine, iar $e_t$ un șoc nou. Timpul de înjumătățire este $\ln 0{,}5/\ln\phi$ zile, aceeași formulă ca la GARCH; un $\phi$ mic înseamnă o revenire rapidă.

## Abaterile de la paritate pentru USDT, USDC și DAI {h=9614af9b}

Graficul are trei panouri, cîte unul pentru USDT, USDC și DAI, fiecare cu abaterea zilnică a închiderii de la 1 USD, în puncte de bază, din 2021 pînă în 2026, după datele EODHD. Linia orizontală de la 0 este paritatea.

În perioadele normale, abaterile sînt foarte mici. Abaterea absolută mediană este 3,1 bp pentru USDT, 1,1 bp pentru USDC și 2,2 bp pentru DAI, iar proporția zilelor cu abateri de peste 10 bp este 11,4%, 1,5% și, respectiv, 10,7%. Timpii de înjumătățire din modelul AR(1) sînt 2,2 zile pentru USDT și 0,6 zile pentru USDC și DAI: abaterile se închid în cîteva zile.

Excepția vizibilă în grafic este martie 2023, cînd USDC a coborît la −285 bp și DAI la −261 bp, în timp ce USDT a urcat peste paritate. DAI are și abateri de ambele semne la sfîrșitul lui 2021. Seria abaterilor are deci o distribuție foarte concentrată în jurul lui zero, cu cîteva valori extreme.

## Martie 2023: USDC și Silicon Valley Bank {h=02ec751e}

Pe 10 martie 2023, Silicon Valley Bank (SVB) a fost închisă de autoritatea de supraveghere. O parte din rezervele în numerar ale USDC erau depuse la SVB. Urma un weekend: băncile erau închise, deci rezervele nu puteau fi mutate, în timp ce piețele cripto erau deschise.

Pe 11 martie, USDC a scăzut la un minim de 0,8774 USD, adică −1226 bp, și s-a închis la 0,9715. DAI, garantat în mare parte cu USDC, l-a urmat. USDT s-a tranzacționat în schimb peste paritate, cu închideri de pînă la 77 bp peste 1 USD: capitalul s-a mutat de la un stablecoin la altul.

În seara zilei de 12 martie, Trezoreria SUA, Rezerva Federală și FDIC (*Federal Deposit Insurance Corporation*, agenția americană de garantare a depozitelor) au anunțat că toate depozitele SVB vor fi garantate [@USTreasuryFederalReserveFDIC2023]. Pe 13 martie, USDC a revenit la paritate. Episodul a durat un weekend și s-a încheiat printr-o decizie a autorităților, nu printr-un mecanism al pieței cripto.

## Depeg-ul USDC, zi de zi {h=b2202aef}

Graficul arată prețul în USD al USDC (albastru închis), DAI (portocaliu) și USDT (verde) între 8 și 20 martie 2023: închiderile zilnice ca cercuri unite prin linii și minimele zilnice ca triunghiuri. Linia orizontală întreruptă marchează paritatea de 1 USD.

Pe 11 martie, închiderile USDC și DAI coboară mult sub paritate, iar minimele lor coboară și mai jos; minimul USDC, 0,8774 USD, este cel mai mic punct al graficului. Pe 12 martie închiderile revin parțial, iar pe 13 martie sînt din nou la paritate. USDT are închideri peste 1 USD în aceleași zile, deci a fost cumpărat de cei care vindeau USDC.

Lecția este că un stablecoin garantat fiat este la fel de sigur ca banca la care își ține rezervele. Paritatea s-a rupt cînd rezervele au fost puse sub semnul întrebării și a revenit cînd depozitele au fost garantate. Retragerea a vizat rezervele, nu codul.

## TerraUSD: un stablecoin algoritmic {h=a7a2db65}

UST era ținut la 1 USD printr-un schimb cu o a doua monedă, LUNA: 1 UST putea fi schimbat oricînd pe LUNA în valoare de 1 USD. Dacă UST se tranzacționa sub 1 USD, un arbitrajor cumpăra UST, îl schimba pe LUNA nou emis în valoare de 1 USD și vindea LUNA. UST era astfel retras din circulație, iar LUNA era emis. Singura garanție a sistemului era valoarea de piață a LUNA.

Cererea pentru UST venea în principal de la protocolul Anchor, care plătea o dobîndă de aproape 20% pe an pentru depozitele în UST [@LiuMakarovSchoar2023]. Oferta de UST a atins maximul de 18,8 miliarde pe 7 mai 2022.

Mecanismul are o slăbiciune: spirala morții (*death spiral*). Retragerile mari din Anchor împing UST sub 1, ceea ce duce la emiterea a mai mult LUNA; prețul LUNA scade, garanția se micșorează, iar încrederea scăzută produce și mai multe retrageri. @LiuMakarovSchoar2023 arată că deținătorii mari și sofisticați s-au retras primii, iar deținătorii mici au plecat ultimii, tiparul unei retrageri bancare clasice.

## Prăbușirea UST, mai 2022 {h=45b2da3b}

Graficul are două panouri, ambele după datele DefiLlama. În stînga, oferta de UST în miliarde, zilnic, de la 1 aprilie pînă la ultima zi publicată, 11 mai 2022. În dreapta, prețul zilnic al UST în USD, de la 1 aprilie pînă la jumătatea lui iunie 2022, cu paritatea marcată de o linie întreruptă.

Oferta a crescut constant în aprilie, a atins maximul pe 7 mai și a scăzut cu 27% în patru zile, între 7 și 11 mai. Prețul era 0,75 USD pe 9 mai, 0,37 pe 13 mai, 0,13 pe 14 mai și 0,02 pe 31 mai: paritatea s-a pierdut în cîteva zile și nu a mai revenit.

Spre deosebire de USDC, UST nu avea nicio rezervă și niciun creditor de ultimă instanță. Cînd valoarea LUNA a scăzut odată cu cea a UST, nu a rămas nimic care să susțină răscumpărarea. O paritate algoritmică nu poate fi salvată de o garanție externă, pentru că nu are una.

## Stablecoins ca monedă privată {h=f353992b}

@GortonZhang2021 compară stablecoins cu bancnotele emise de băncile private americane în perioada „wildcat banking” (1837–1863). Bancnotele diferitelor bănci se tranzacționau atunci cu reduceri diferite față de valoarea nominală, după încrederea în fiecare bancă. Remediul de atunci a fost o monedă națională uniformă; remediul propus acum este reglementarea stablecoins ca depozite bancare.

@GortonEtAl2022 explică paritatea prin cerere: ea se menține cîtă vreme deținătorii prețuiesc moneda pentru tranzacții cu levier pe piețele cripto. Cînd această cerere dispare, moneda se tranzacționează sub paritate, ca bancnota unei bănci nesigure.

Lecția statistică este că abaterile de la paritate sînt mici și scurte în perioadele normale și cresc brusc în criză. Seria are cozi groase și depinde de regim: o statistică calculată pe perioadele liniștite, cum este abaterea mediană de 1,1 bp a USDC, nu spune nimic despre mărimea unei abateri în criză.

## Reglementarea în Uniunea Europeană: MiCA {h=97610009}

Regulamentul MiCA (*Markets in Crypto-Assets*), Regulamentul (UE) 2023/1114, a fost adoptat în 2023. Regulile pentru stablecoins se aplică din 30 iunie 2024, iar cele pentru celelalte servicii cu active cripto din 30 decembrie 2024.

Stablecoins legate de o singură monedă sînt numite în MiCA token-uri de monedă electronică (*e-money tokens*). Ele pot fi emise doar de instituții de credit sau de instituții de monedă electronică autorizate, trebuie răscumpărate la paritate oricînd și nu pot plăti dobîndă deținătorilor. Rezervele sînt separate de activele emitentului și investite în active sigure și lichide.

Cadrul răspunde direct riscurilor din secțiune: rezervele separate și sigure reduc riscul de tip SVB, iar stablecoins algoritmice fără rezerve, de tipul UST, nu au loc în acest cadru.

## Reglementarea în SUA: GENIUS Act {h=d081cd4a}

GENIUS Act (*Guiding and Establishing National Innovation for U.S. Stablecoins Act*) a fost semnat pe 18 iulie 2025. Pentru stablecoins de plată, legea cere rezerve de cel puțin 1 USD pentru fiecare monedă, în numerar, depozite și titluri de stat pe termen scurt, rapoarte publice lunare despre rezerve și interzice plata dobînzii către deținători.

MiCA și GENIUS Act au trei puncte comune: rezerve complete, răscumpărare la paritate și supravegherea emitenților. Ambele răspund retragerii masive din martie 2023 și prăbușirii algoritmice din mai 2022.

Un stablecoin cu rezerve complete este apropiat de o bancă îngustă (*narrow bank*), o bancă ce ține toate depozitele în active sigure și nu acordă credite. Moneda este sigură, dar emitentul cîștigă dobînda la rezerve, în timp ce deținătorii nu primesc nimic.

## @end

Stablecoins își mențin paritatea prin arbitraj și prin rezerve, iar în perioadele normale abaterile sînt de cîteva puncte de bază și se închid în cîteva zile. Cele două episoade arată două moduri în care se rupe o paritate: USDC în martie 2023, cînd rezervele au fost puse sub semnul întrebării și apoi garantate, și UST în mai 2022, cînd o paritate fără rezerve s-a prăbușit definitiv. MiCA și GENIUS Act cer rezerve complete și răscumpărare la paritate.

# Riscul unei poziții cripto

Secțiunea calculează VaR 1% și ES 2,5% pentru Bitcoin, Ethereum și S&P 500 prin patru metode, transformă rezultatele în dolari, discută orizontul de 10 zile și verifică prin backtesting VaR-ul zilnic al Bitcoin din 2018.

## VaR 1% și ES 2,5% în dolari {h=60d5c60e}

În notația din [Capitolul 10](ch10_ro.html), $\mathrm{VaR}_\alpha = -q_\alpha$, unde $q_\alpha$ este cuantila de ordin $\alpha$ a randamentului zilnic $r$; VaR 1% este pierderea depășită cu probabilitatea 1%. ES este $\mathrm{ES}_\alpha = -E[r \mid r \le q_\alpha]$, randamentul mediu, cu semn schimbat, în zilele de dincolo de cuantilă; capitolul folosește ES 2,5%. Cuantila se estimează prin patru metode: simularea istorică (HS, *historical simulation*), distribuția Normală, distribuția Student-t estimată prin verosimilitate maximă și modelul GARCH(1,1)-t pentru ziua următoare.

VaR-ul este calculat pe randamente logaritmice, iar o pierdere în bani cere o conversie. Un randament logaritmic de $-v\%$ înseamnă o pierdere

$$ W\big(1 - e^{-v/100}\big) $$ {#eq:varusd}

pentru o poziție cu valoarea $W$. Pentru Bitcoin, VaR 1% prin HS este 10,54%; pentru $W = 100\,000$ USD, pierderea este $100\,000 \times (1 - 0{,}8999) = 10\,006$ USD, puțin sub 10 540 USD, deoarece un randament logaritmic negativ corespunde unei pierderi procentuale mai mici.

Orizontul se măsoară în zile calendaristice. Activele cripto se tranzacționează zilnic, deci un VaR pe 10 zile acoperă 10 zile calendaristice, nu două săptămîni de tranzacționare ca la acțiuni.

## VaR 1% și ES 2,5% prin patru metode {h=23ed5ef7}

Primul tabel dă VaR 1% și ES 2,5%, în procente, prin cele patru metode, pe eșantioanele complete pînă pe 18 septembrie 2026. Pentru Bitcoin, VaR 1% este 10,54% prin HS, 7,99% cu distribuția Normală, 11,06% cu Student-t și 7,37% cu GARCH-t; ES 2,5% este 10,84%, 8,03%, 13,37% și 8,11%. Pentru Ethereum, VaR 1% prin HS este 14,29%, iar pentru S&P 500 3,26%.

Distribuția Student-t estimată pe randamentele Bitcoin are $\hat\nu = 2{,}23$ grade de libertate, de aceea dă cea mai mare cifră de ES. Prognoza GARCH-t pentru ziua următoare folosește volatilitatea condiționată $\hat\sigma = 2{,}83\%$, sub abaterea standard necondiționată de 3,49%.

Al doilea tabel transformă cifrele în dolari pentru o poziție de 100 000 USD, cu @eq:varusd. VaR 1% prin HS este 10 006 USD pentru Bitcoin, 13 312 USD pentru Ethereum și 3 204 USD pentru S&P 500; ES 2,5% prin HS este 10 277, 13 732 și 3 438 USD. VaR 1% Normal este 7 679 USD pentru Bitcoin, iar VaR 1% HS pe 10 zile este 25 563 USD pentru Bitcoin, 31 842 USD pentru Ethereum și 9 049 USD pentru S&P 500.

## Interpretarea cifrelor de risc {h=84779ea1}

O poziție în Bitcoin este de circa trei ori mai riscantă decît una în S&P 500: VaR 1% prin HS este 10,54% față de 3,26%. VaR 1% Normal al Bitcoin este cu 24% sub cel istoric, efectul cozilor groase din [Secțiunea 3](#sec-3): distribuția Normală, ajustată pe abaterea standard, subestimează cuantilele extreme.

Diferența dintre GARCH-t și HS este diferența dintre riscul condiționat și cel necondiționat. GARCH-t dă astăzi 7,37%, sub valoarea istorică de 10,54%, deoarece volatilitatea Bitcoin este acum scăzută. Cifra istorică este o medie pe termen lung, care include toate episoadele de volatilitate mare din eșantion; cifra GARCH este riscul de mîine, condiționat de volatilitatea de azi.

Pentru 10 zile, VaR 1% istoric calculat direct din randamentele pe 10 zile este 29,52%, iar regula rădăcinii pătrate a timpului (*square-root-of-time rule*), $\mathrm{VaR}(h \text{ zile}) = \sqrt{h}\,\mathrm{VaR}(1 \text{ zi})$, dă 33,34%. Regula este exactă doar pentru randamente Normale i.i.d. Cu un tail index $\alpha > 2$, o cuantilă extremă a sumei a $h$ randamente crește aproximativ ca $h^{1/\alpha}$, mai lent decît $\sqrt{h}$, de aceea regula supraestimează aici. ES 2,5% prin HS (10,84%) este apropiat de VaR 1% prin HS (10,54%), ca la acțiunile din [Capitolul 10](ch10_ro.html).

## Backtesting: testul Kupiec {h=5a61b7d1}

O depășire (*exception*) este o zi în care pierderea depășește VaR-ul prognozat: $r_{t+1} < -\widehat{\mathrm{VaR}}_t$. Cu $x$ depășiri în $n$ zile, rata observată este $\hat\pi = x/n$. Dacă VaR 1% este corect, fiecare zi este o depășire cu probabilitatea $\alpha = 0{,}01$, deci se așteaptă circa $\alpha n$ depășiri.

Testul lui @Kupiec1995 întreabă dacă rata depășirilor este egală cu $\alpha$, printr-un raport de verosimilitate:

$$ LR_{uc} = -2\ln\frac{(1 - \alpha)^{n - x}\,\alpha^{x}}{(1 - \hat\pi)^{n - x}\,\hat\pi^{x}}. $$ {#eq:kupiec}

Numărătorul este verosimilitatea celor $x$ depășiri dacă rata este $\alpha$, iar numitorul verosimilitatea lor cu rata observată $\hat\pi$. Dacă rata adevărată este $\alpha$, $LR_{uc}$ are aproximativ distribuția $\chi^2(1)$, iar ipoteza este respinsă la 5% dacă $LR_{uc} > 3{,}84$. Testul numără doar depășirile; dacă ele sînt grupate în timp, problema este detectată de testul Christoffersen din [Capitolul 10](ch10_ro.html).

## Backtesting pentru VaR 1% al Bitcoin {h=9c739da7}

Graficul arată randamentele zilnice ale Bitcoin, în procente (albastru), din 1 ianuarie 2018 pînă în 2026, împreună cu două prognoze ale lui −VaR 1% pentru ziua următoare: HS pe ultimele 365 de zile (linia maro în trepte) și GARCH(1,1)-t reestimat la fiecare 365 de zile pe toate datele anterioare (linia roșie). Punctele portocalii marchează depășirile VaR-ului istoric.

Numărul așteptat de depășiri este 31,8. HS are 29 de depășiri (0,91%), iar GARCH-t 41 (1,29%). Statistica Kupiec este $LR_{uc} = 0{,}26$ (p-value 0,61) pentru HS și 2,45 (p-value 0,12) pentru GARCH-t, deci ambele modele trec testul.

Cele două modele se comportă diferit. HS cere în medie un VaR mai mare, 9,5% față de 8,5%, și se schimbă lent, în trepte; GARCH-t reacționează mai repede la volatilitate, dar este depășit mai des, de exemplu de 9 ori în 2018. Linia roșie urcă și coboară cu volatilitatea, iar cel mai adînc punct al ei apare în martie 2020, imediat după cea mai proastă zi a eșantionului.

## @end

O poziție în Bitcoin are un VaR 1% istoric de 10,54%, de trei ori mai mare decît S&P 500; distribuția Normală îl subestimează cu 24%, iar GARCH-t dă un risc condiționat mai mic cînd volatilitatea curentă este scăzută. Pe 10 zile, regula rădăcinii pătrate a timpului supraestimează riscul. În backtesting din 2018, atît HS, cît și GARCH-t trec testul Kupiec.

# AI pentru descoperire științifică

Secțiunea pornește de la o întrebare deschisă ridicată de capitol, dacă fluxurile de stablecoins influențează prețurile activelor cripto, și arată unde pot ajuta instrumentele AI (*artificial intelligence*) un cercetător care o studiază și ce verificări nu pot înlocui.

## O întrebare de cercetare deschisă {h=df133a93}

Întrebarea este dacă fluxurile de stablecoins influențează prețurile activelor cripto. @GriffinShams2020 au găsit că emisiunea de Tether a susținut prețul Bitcoin în 2017, iar piața stablecoin-urilor este acum de circa 74 de ori mai mare decît în 2020. Dacă fluxurile de stablecoins sînt o sursă a cererii pentru Bitcoin, ele ar trebui să se vadă în randamente.

Rezultatul de pornire compară variația logaritmică săptămînală a ofertei de stablecoins cu randamentul Bitcoin, pe 349 de săptămîni din 2020. În aceeași săptămînă, corelația este 0,26. Față de randamentul din săptămîna următoare, corelația este 0,07, iar regresia dă panta 0,19 ($t = 1{,}20$, p-value 0,23) și $R^2 = 0{,}4\%$.

Fluxurile și prețurile se mișcă deci împreună, dar fluxurile nu prognozează săptămîna următoare. Corelația contemporană poate însemna că fluxurile cauzează prețurile, că prețurile cauzează fluxurile sau că amîndouă răspund unei cereri comune. Instrumentele AI pot accelera un studiu care să le separe, dar nu înlocuiesc verificarea lui [@WangEtAl2023].

## Utilizări posibile ale instrumentelor AI {h=a28087bd}

Un asistent AI poate ajuta în trei etape. Pentru literatură, poate întocmi o listă a studiilor despre emisiunea de stablecoins și prețurile cripto, cu datele și perioadele folosite de fiecare. Pentru date, poate scrie codul care descarcă oferta fiecărui stablecoin de la DefiLlama și o aliniază cu prețurile zilnice. Pentru robustețe, poate propune variante: date zilnice și săptămînale, USDT și USDC separat, subperioade și cauzalitate Granger în ambele direcții.

Exemplul de prompt de pe slide este precis: numește sursa (API-ul de stablecoins al DefiLlama, *application programming interface*), monedele (USDT și USDC), transformarea (variații logaritmice săptămînale), seria cu care se unesc (randamentele logaritmice săptămînale ale Bitcoin), testul (cauzalitate Granger în ambele direcții) și erorile standard (HAC). Un prompt precis produce cod ale cărui alegeri pot fi verificate față de definițiile capitolului.

Rezultatul fiecărei etape este o ciornă. Lista de lucrări trebuie verificată referință cu referință, iar codul trebuie verificat pe rezultate cunoscute, de exemplu reproducînd corelația de 0,26 din aceeași săptămînă înainte de a trece la testele noi.

## Verificări necesare {h=bca0da86}

Mai multe erori sînt frecvente în analizele pe date cripto și trebuie căutate explicit în codul generat cu AI. Prima este anualizarea: 365 de zile pentru cripto și 252 pentru acțiuni, niciodată o singură valoare pentru ambele. A doua este alinierea: prețurile se unesc în zilele comune înainte de calculul randamentelor, iar săptămîna are aceeași definiție pentru ambele serii.

A treia eroare privește unitățile: puncte de bază față de procente (285 bp înseamnă 2,85%) și miliarde față de milioane în datele de ofertă. A patra este convenția VaR: VaR 1% este o pierdere pozitivă, exprimată prin probabilitatea $\alpha$ a cozii, iar cuantila folosită este cea de ordin 0,01. A cincea este look-ahead bias-ul: seria ofertei trebuie să fie cea cunoscută la data prognozei, iar revizuirile ulterioare ale furnizorului de date pot crea o predictibilitate care nu exista în timp real.

Ultima verificare privește referințele. Fiecare lucrare citată de un instrument AI trebuie să existe, iar DOI-ul ei trebuie verificat, deoarece referințele inventate, cu titluri plauzibile, sînt o eroare cunoscută a modelelor de limbaj.

## Idee de proiect {h=49f21e9e}

Proiectul întreabă dacă variațiile săptămînale ale ofertei de USDT și USDC ajută la prognoza randamentului sau a volatilității Bitcoin și dacă acest lucru s-a schimbat după lansarea ETF-urilor spot. Datele sînt oferta de stablecoins de la DefiLlama, publică și fără cheie de acces, și datele de preț ale cursului.

Pașii urmează capitolul. Întîi, se calculează variațiile logaritmice săptămînale, corelațiile și testele Granger în ambele direcții, cu erori standard HAC. Apoi, se fac prognoze out-of-sample pentru randamentul și pentru randamentul absolut din săptămîna următoare, comparate cu un mers aleator. În final, se compară subperioadele de dinainte și de după Terra (mai 2022) și ETF-urile spot (ianuarie 2024).

Rezultatul este un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele. Orice folosire a instrumentelor AI se declară, împreună cu erorile acestora găsite și corectate.

## @end

Fluxurile de stablecoins și randamentele Bitcoin sînt corelate în aceeași săptămînă, dar fluxurile nu prognozează săptămîna următoare, iar direcția cauzalității rămîne deschisă. Instrumentele AI pot scrie o primă versiune a codului și a căutării bibliografice; cercetătorul verifică anualizarea, alinierea, unitățile, convenția VaR, datele disponibile la momentul prognozei și existența fiecărei referințe.

# Rezumat

Rezultatele capitolului se reduc la cîteva afirmații despre activele cripto ca obiect statistic, un set de formule și trei întrebări de autoevaluare.

## Idei de reținut {h=b4a67e9a}

Bitcoin este un registru public securizat prin proof of work, cu o ofertă fixă de 21 de milioane de monede; Ethereum execută contracte inteligente și folosește proof of stake. Randamentele cripto se anualizează cu 365 de zile; volatilitatea lor este de 3,8 pînă la 6,7 ori cea a S&P 500, cu cozi groase și cu volatility clustering atît de persistent încît modelul GARCH estimat este integrat.

Testele robuste ale raportului varianțelor nu resping eficiența în formă slabă din 2014, iar weekendul este mai calm în varianță, dar nu diferit în medie. Corelația cu acțiunile a crescut de la aproximativ 0 la aproximativ 0,39 după 2020, iar Bitcoin nu este un activ de refugiu.

Drawdown-urile de peste 50% sînt obișnuite, iar efectul ETF-urilor spot asupra volatilității nu este semnificativ. Stablecoins își mențin paritatea prin arbitraj și rezerve; USDC în 2023 și UST în 2022 arată cele două moduri în care se rupe o paritate.

## Formule de reținut {h=cccb6e7f}

Tabelul adună formulele capitolului. Oferta de Bitcoin, @eq:supply, este suma unei serii geometrice. Anualizarea, @eq:annual, înmulțește media cu $A$ și abaterea standard cu $\sqrt{A}$, cu $A = 365$ pentru cripto și 252 pentru acțiuni. Estimatorul Hill, @eq:hill, este inversul mediei logaritmilor depășirilor peste prag.

Modelul GARCH(1,1), @eq:garch, are timpul de înjumătățire $\ln 0{,}5/\ln(\alpha + \beta)$, iar raportul varianțelor, @eq:vr, este 1 pentru un mers aleator. Drawdown-ul, @eq:dd, cere un cîștig de recuperare $d/(1 - d)$. Abaterea de la paritate, @eq:peg, se măsoară în puncte de bază, iar timpul ei de înjumătățire este $\ln 0{,}5/\ln\phi$.

VaR în bani se obține cu @eq:varusd, cu $\mathrm{VaR}_\alpha = -q_\alpha$. Timpul de înjumătățire are aceeași formă la GARCH și la abaterea de la paritate: logaritmul lui 0,5 împărțit la logaritmul coeficientului de persistență.

## Autoevaluare {h=4c08efcd}

Prima întrebare cere dacă este corectă o volatilitate anuală a Bitcoin de 55% raportată de un fond pornind de la o abatere standard zilnică de 3,5%. Nu este: $3{,}5 \times \sqrt{252} \approx 55{,}6\%$ folosește calendarul acțiunilor, iar cu 365 de zile se obține $3{,}5 \times \sqrt{365} \approx 66{,}9\%$.

A doua întrebare cere abaterea de la paritate a USDT cînd se închide la 1,0040 USD. Cu @eq:peg, ea este $10\,000 \times 0{,}0040 = 40$ bp peste paritate: cererea de USDT depășește ce a furnizat arbitrajul.

A treia întrebare cere de ce s-a pierdut UST, în timp ce USDC a revenit la paritate. USDC avea rezerve în dolari, iar depozitele bancare în care se aflau au fost garantate; UST era garantat doar de LUNA, a cărei valoare a scăzut odată cu UST. Pentru lecturi suplimentare, @PeleEtAl2023 întreabă dacă activele cripto devin active alternative, iar @PeleMazurencuMarinescuPele2019 prognozează VaR-ul zilnic al Bitcoin. Capitolul următor, [Capitolul 15](ch15_ro.html), tratează riscul sistemic.
