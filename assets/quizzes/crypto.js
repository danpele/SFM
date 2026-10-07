// ============================================================
// Chapter 14 quiz bank: crypto assets and stablecoins (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['crypto'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Bitcoin supply",
                "text": "The Bitcoin block subsidy starts at 50 BTC and is halved every 210,000 blocks. What is the total supply?",
                "options": [
                    "10.5 million BTC",
                    "21 million BTC",
                    "50 million BTC",
                    "There is no limit"
                ],
                "correctExplanation": "210,000 x 50 x (1 + 1/2 + 1/4 + ...) = 210,000 x 50 x 2 = 21 million: a geometric series.",
                "incorrectExplanation": "The halving rule makes the issuance a geometric series with ratio 1/2, whose sum is twice the first term: 21 million coins."
            },
            "ro": {
                "title": "Oferta de Bitcoin",
                "text": "Subvenția unui bloc Bitcoin pornește de la 50 BTC și se înjumătățește la fiecare 210.000 de blocuri. Cît este oferta totală?",
                "options": [
                    "10,5 milioane de BTC",
                    "21 de milioane de BTC",
                    "50 de milioane de BTC",
                    "Nu există o limită"
                ],
                "correctExplanation": "210.000 x 50 x (1 + 1/2 + 1/4 + ...) = 210.000 x 50 x 2 = 21 de milioane: o serie geometrică.",
                "incorrectExplanation": "Regula halving-ului face din emisiune o serie geometrică de rație 1/2, a cărei sumă este dublul primului termen: 21 de milioane de monede."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Proof of work",
                "text": "What does proof of work achieve in Bitcoin?",
                "options": [
                    "It pays interest to coin holders",
                    "It keeps the price at 1 USD",
                    "It lets a central bank approve transactions",
                    "It makes rewriting past blocks require more computing power than all honest miners together"
                ],
                "correctExplanation": "Each block requires a costly search for a hash below a target; an attacker would have to redo that work for every later block faster than the honest network.",
                "incorrectExplanation": "Proof of work is a consensus rule about who adds the next block; it pays miners, not holders, and has nothing to do with a peg or a central bank."
            },
            "ro": {
                "title": "Proof of work",
                "text": "Ce realizează proof of work în Bitcoin?",
                "options": [
                    "Plătește dobîndă deținătorilor de monede",
                    "Menține prețul la 1 USD",
                    "Permite unei bănci centrale să aprobe tranzacțiile",
                    "Face ca rescrierea blocurilor trecute să ceară mai multă putere de calcul decît toți minerii onești la un loc"
                ],
                "correctExplanation": "Fiecare bloc cere o căutare costisitoare a unui hash sub o țintă; un atacator ar trebui să refacă această muncă pentru toate blocurile ulterioare mai repede decît rețeaua onestă.",
                "incorrectExplanation": "Proof of work este o regulă de consens despre cine adaugă următorul bloc; plătește minerii, nu deținătorii, și nu are legătură cu o paritate sau cu o bancă centrală."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The Merge",
                "text": "What changed on Ethereum on 15 September 2022 (the Merge)?",
                "options": [
                    "It moved from proof of work to proof of stake, cutting its energy use by more than 99.9%",
                    "Its supply was fixed at 21 million coins",
                    "It became a stablecoin",
                    "Its trading moved to the stock exchange"
                ],
                "correctExplanation": "Under proof of stake, validators lock coins as a deposit instead of competing with computing power, so the energy use collapses.",
                "incorrectExplanation": "The Merge replaced the consensus rule; the 21 million limit belongs to Bitcoin, and Ethereum is neither a stablecoin nor an exchange-listed security."
            },
            "ro": {
                "title": "The Merge",
                "text": "Ce s-a schimbat la Ethereum pe 15 septembrie 2022 (the Merge)?",
                "options": [
                    "A trecut de la proof of work la proof of stake, reducîndu-și consumul de energie cu peste 99,9%",
                    "Oferta a fost fixată la 21 de milioane de monede",
                    "A devenit un stablecoin",
                    "Tranzacționarea s-a mutat la bursă"
                ],
                "correctExplanation": "În proof of stake, validatorii blochează monede drept garanție în loc să concureze prin putere de calcul, deci consumul de energie se prăbușește.",
                "incorrectExplanation": "The Merge a înlocuit regula de consens; limita de 21 de milioane aparține Bitcoin, iar Ethereum nu este nici stablecoin, nici titlu listat la bursă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Annualising crypto",
                "text": "Bitcoin daily log returns have a standard deviation of 3.5%. What is the annual volatility?",
                "options": [
                    "3.5 x 252 = 882%",
                    "3.5 x sqrt(252) = 55.6%",
                    "3.5 x sqrt(365) = 66.9%",
                    "3.5 x 12 = 42%"
                ],
                "correctExplanation": "Crypto trades 365 days a year, so the variance is summed over 365 days: volatility times sqrt(365).",
                "incorrectExplanation": "Volatility scales with the square root of the number of observations a year; for a market open every day that number is 365, not 252."
            },
            "ro": {
                "title": "Anualizarea cripto",
                "text": "Randamentele logaritmice zilnice ale Bitcoin au o abatere standard de 3,5%. Cît este volatilitatea anuală?",
                "options": [
                    "3,5 x 252 = 882%",
                    "3,5 x sqrt(252) = 55,6%",
                    "3,5 x sqrt(365) = 66,9%",
                    "3,5 x 12 = 42%"
                ],
                "correctExplanation": "Activele cripto se tranzacționează 365 de zile pe an, deci varianța se însumează pe 365 de zile: volatilitatea înmulțită cu sqrt(365).",
                "incorrectExplanation": "Volatilitatea crește cu rădăcina pătrată a numărului de observații pe an; pentru o piață deschisă în fiecare zi acest număr este 365, nu 252."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The 252-day mistake",
                "text": "By how much does annualising a crypto volatility with sqrt(252) instead of sqrt(365) understate it?",
                "options": [
                    "By about 31%",
                    "By about 17%",
                    "By about 5%",
                    "It does not understate it"
                ],
                "correctExplanation": "1 - sqrt(252/365) = 0.169: the volatility is about 17% too low, for any asset.",
                "incorrectExplanation": "The bias depends only on the two conventions: sqrt(252/365) = 0.831; the 31% figure is the bias of the mean (1 - 252/365)."
            },
            "ro": {
                "title": "Greșeala cu 252 de zile",
                "text": "Cu cît subestimează volatilitatea cripto anualizarea cu sqrt(252) în loc de sqrt(365)?",
                "options": [
                    "Cu aproximativ 31%",
                    "Cu aproximativ 17%",
                    "Cu aproximativ 5%",
                    "Nu o subestimează"
                ],
                "correctExplanation": "1 - sqrt(252/365) = 0,169: volatilitatea este cu aproximativ 17% prea mică, pentru orice activ.",
                "incorrectExplanation": "Abaterea depinde doar de cele două convenții: sqrt(252/365) = 0,831; cifra de 31% este abaterea mediei (1 - 252/365)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Tail shape",
                "text": "The Hill tail index of the losses is 2.71 [2.20; 3.21] for Bitcoin and 2.68 [2.08; 3.29] for the S&P 500. What follows?",
                "options": [
                    "The two left tails have the same shape; Bitcoin differs mainly in scale",
                    "Bitcoin has a much heavier tail",
                    "The S&P 500 has Normal tails",
                    "Bitcoin has a finite kurtosis"
                ],
                "correctExplanation": "The point estimates are almost equal and the intervals overlap: both follow a power law with a similar exponent, while Bitcoin is about four times more volatile.",
                "incorrectExplanation": "A tail index measures the shape of the tail, not its scale; with alpha below 4 neither series has a finite kurtosis, and neither tail is Normal."
            },
            "ro": {
                "title": "Forma cozii",
                "text": "Tail index-ul Hill al pierderilor este 2,71 [2,20; 3,21] pentru Bitcoin și 2,68 [2,08; 3,29] pentru S&P 500. Ce rezultă?",
                "options": [
                    "Cele două cozi stîngi au aceeași formă; Bitcoin diferă în principal prin scală",
                    "Bitcoin are o coadă mult mai groasă",
                    "S&P 500 are cozi Normale",
                    "Bitcoin are o boltire finită"
                ],
                "correctExplanation": "Estimările punctuale sînt aproape egale, iar intervalele se suprapun: ambele urmează o lege de putere cu un exponent apropiat, în timp ce Bitcoin este de circa patru ori mai volatil.",
                "incorrectExplanation": "Tail index-ul măsoară forma cozii, nu scala ei; cu alpha sub 4, niciuna dintre serii nu are boltire finită, iar niciuna dintre cozi nu este Normală."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Squared returns",
                "text": "The ACF of Bitcoin returns is almost zero, but 15 of the first 30 autocorrelations of the squared returns lie outside +-1.96/sqrt(n). What does this show?",
                "options": [
                    "Returns are predictable in direction",
                    "The data contain an error",
                    "The market is inefficient in the strong form",
                    "Volatility clustering: large moves follow large moves"
                ],
                "correctExplanation": "The sign of the next move is hard to predict, but its size is not: this is volatility clustering, modelled by GARCH.",
                "incorrectExplanation": "Autocorrelation of squared returns concerns the size of moves, not their direction, so it says nothing about predictable returns or strong-form efficiency."
            },
            "ro": {
                "title": "Pătratele randamentelor",
                "text": "ACF a randamentelor Bitcoin este aproape zero, dar 15 dintre primele 30 de autocorelații ale pătratelor randamentelor ies din +-1,96/sqrt(n). Ce arată acest lucru?",
                "options": [
                    "Randamentele sînt previzibile ca direcție",
                    "Datele conțin o eroare",
                    "Piața este ineficientă în formă tare",
                    "Volatility clustering: mișcările mari urmează după mișcări mari"
                ],
                "correctExplanation": "Semnul mișcării următoare este greu de prognozat, dar mărimea ei nu: acesta este volatility clustering, modelat prin GARCH.",
                "incorrectExplanation": "Autocorelația pătratelor randamentelor privește mărimea mișcărilor, nu direcția lor, deci nu spune nimic despre randamente previzibile sau despre eficiența în formă tare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH persistence",
                "text": "A GARCH(1,1)-t fitted to Bitcoin gives alpha + beta = 1.000. What does this mean?",
                "options": [
                    "The model is invalid and must be dropped",
                    "Volatility is constant",
                    "An integrated GARCH: a volatility shock does not die out and there is no finite long-run variance",
                    "The half-life of a shock is one day"
                ],
                "correctExplanation": "With alpha + beta = 1 the forecast of the variance does not revert to a long-run level: an IGARCH, close to the EWMA of Chapter 8.",
                "incorrectExplanation": "A persistence of one means the opposite of a short half-life or of constant volatility; the model stays usable for forecasting, as EWMA is."
            },
            "ro": {
                "title": "Persistența GARCH",
                "text": "Un GARCH(1,1)-t estimat pentru Bitcoin dă alpha + beta = 1,000. Ce înseamnă acest lucru?",
                "options": [
                    "Modelul este invalid și trebuie abandonat",
                    "Volatilitatea este constantă",
                    "Un GARCH integrat: un șoc al volatilității nu se stinge și nu există o varianță de termen lung finită",
                    "Timpul de înjumătățire al unui șoc este o zi"
                ],
                "correctExplanation": "Cu alpha + beta = 1, prognoza varianței nu revine la un nivel de termen lung: un IGARCH, apropiat de EWMA din Capitolul 8.",
                "incorrectExplanation": "O persistență egală cu 1 înseamnă opusul unui timp de înjumătățire scurt sau al unei volatilități constante; modelul rămîne utilizabil pentru prognoză, ca și EWMA."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Leverage effect",
                "text": "The GJR-GARCH asymmetry term is 0.28 (t = 7.1) for the S&P 500 and -0.01 (t = -0.7) for Bitcoin. What follows?",
                "options": [
                    "Bitcoin has a stronger leverage effect",
                    "Falls raise equity volatility more than rises; Bitcoin shows no such asymmetry",
                    "Both have the same asymmetry",
                    "Bitcoin volatility does not cluster"
                ],
                "correctExplanation": "A positive, significant gamma is the leverage effect of equities; for Bitcoin the term is small and not significant.",
                "incorrectExplanation": "The t-statistic of -0.7 cannot distinguish Bitcoin's gamma from zero, so there is no evidence of asymmetry; clustering is a separate property, measured by alpha and beta."
            },
            "ro": {
                "title": "Efectul de levier",
                "text": "Termenul de asimetrie GJR-GARCH este 0,28 (t = 7,1) pentru S&P 500 și -0,01 (t = -0,7) pentru Bitcoin. Ce rezultă?",
                "options": [
                    "Bitcoin are un efect de levier mai puternic",
                    "Scăderile cresc volatilitatea acțiunilor mai mult decît creșterile; Bitcoin nu are o astfel de asimetrie",
                    "Ambele au aceeași asimetrie",
                    "Volatilitatea Bitcoin nu se grupează"
                ],
                "correctExplanation": "Un gamma pozitiv și semnificativ este efectul de levier al acțiunilor; pentru Bitcoin termenul este mic și nesemnificativ.",
                "incorrectExplanation": "Statistica t de -0,7 nu poate distinge gamma pentru Bitcoin de zero, deci nu există dovezi de asimetrie; volatility clustering este o proprietate separată, măsurată prin alpha și beta."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Robust variance ratios",
                "text": "On rolling windows, the robust Z*(5) never rejects the random walk for the S&P 500, but the i.i.d. Z(5) rejects in 22% of the windows. Why?",
                "options": [
                    "Volatility clustering makes the i.i.d. standard error too small, producing false rejections",
                    "The S&P 500 is inefficient",
                    "The robust test has no power",
                    "The windows are too long"
                ],
                "correctExplanation": "Under heteroskedasticity the i.i.d. variance of VR(q) is wrong; the Lo-MacKinlay Z* corrects it and the rejections disappear.",
                "incorrectExplanation": "The discrepancy comes from the standard error, not from the data: once heteroskedasticity is accounted for, there is no evidence against the random walk."
            },
            "ro": {
                "title": "Rapoarte ale varianțelor robuste",
                "text": "Pe ferestre mobile, statistica robustă Z*(5) nu respinge niciodată mersul aleator pentru S&P 500, dar statistica i.i.d. Z(5) respinge în 22% din ferestre. De ce?",
                "options": [
                    "Volatility clustering face ca eroarea standard i.i.d. să fie prea mică, ceea ce produce respingeri false",
                    "S&P 500 este ineficient",
                    "Testul robust nu are putere",
                    "Ferestrele sînt prea lungi"
                ],
                "correctExplanation": "În prezența heteroscedasticității, varianța i.i.d. a lui VR(q) este greșită; statistica Z* a lui Lo și MacKinlay o corectează, iar respingerile dispar.",
                "incorrectExplanation": "Diferența vine din eroarea standard, nu din date: odată luată în calcul heteroscedasticitatea, nu există dovezi împotriva mersului aleator."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The weekend",
                "text": "For Bitcoin, the weekend dummy in the mean has p = 0.83, while the Brown-Forsythe test of equal variance has p < 0.001 (weekend/weekday variance 0.39). What is the conclusion?",
                "options": [
                    "Weekend returns are higher",
                    "There is no weekday effect at all",
                    "Weekends are riskier",
                    "No difference in the mean; the weekend is calmer in variance"
                ],
                "correctExplanation": "The mean test does not reject, the variance test rejects strongly: weekend variance is about 40% of the weekday variance.",
                "incorrectExplanation": "A large p-value for the mean and a tiny one for the variance point in different directions: the effect is in the risk, and the risk is lower at weekends."
            },
            "ro": {
                "title": "Weekendul",
                "text": "Pentru Bitcoin, variabila weekend din ecuația mediei are p = 0,83, iar testul Brown-Forsythe al egalității varianțelor are p < 0,001 (varianța weekend/zile lucrătoare 0,39). Care este concluzia?",
                "options": [
                    "Randamentele din weekend sînt mai mari",
                    "Nu există niciun efect al zilei din săptămînă",
                    "Weekendurile sînt mai riscante",
                    "Nicio diferență în medie; weekendul este mai calm ca varianță"
                ],
                "correctExplanation": "Testul mediei nu respinge, testul varianței respinge puternic: varianța din weekend este circa 40% din cea din zilele lucrătoare.",
                "incorrectExplanation": "Un p-value mare pentru medie și unul foarte mic pentru varianță indică lucruri diferite: efectul este în risc, iar riscul este mai mic în weekend."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Different calendars",
                "text": "Bitcoin trades 7 days a week, the S&P 500 5. How should their correlation be computed?",
                "options": [
                    "Join the returns on common days",
                    "Fill the S&P 500 weekends with zero returns",
                    "Join the prices on common days, then compute returns",
                    "Use only Bitcoin weekend returns"
                ],
                "correctExplanation": "Joining prices first makes Monday's Bitcoin return cover Friday to Monday, like the S&P 500 return.",
                "incorrectExplanation": "Joining returns drops Bitcoin's weekend moves, and zero returns invent data; both bias the correlation."
            },
            "ro": {
                "title": "Calendare diferite",
                "text": "Bitcoin se tranzacționează 7 zile pe săptămînă, S&P 500 5. Cum se calculează corelația lor?",
                "options": [
                    "Unim randamentele în zilele comune",
                    "Completăm weekendurile S&P 500 cu randamente zero",
                    "Unim prețurile în zilele comune, apoi calculăm randamentele",
                    "Folosim doar randamentele Bitcoin din weekend"
                ],
                "correctExplanation": "Unirea prealabilă a prețurilor face ca randamentul Bitcoin de luni să acopere perioada vineri-luni, ca randamentul S&P 500.",
                "incorrectExplanation": "Unirea randamentelor elimină mișcările Bitcoin din weekend, iar randamentele zero inventează date; ambele deformează corelația."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Correlation over time",
                "text": "The correlation of daily Bitcoin and S&P 500 returns is 0.02 before 2020 and 0.39 from 2020 on (Fisher z = 10.7). What follows?",
                "options": [
                    "The correlation is stable",
                    "Bitcoin became significantly more correlated with equities after 2020",
                    "Bitcoin became a hedge for equities",
                    "The difference is due to chance"
                ],
                "correctExplanation": "A Fisher z of 10.7 rejects equal correlations: since 2020 crypto behaves as a risk asset.",
                "incorrectExplanation": "A z-statistic above 10 rules out chance and stability; a positive correlation of 0.39 is the opposite of a hedge."
            },
            "ro": {
                "title": "Corelația în timp",
                "text": "Corelația randamentelor zilnice Bitcoin și S&P 500 este 0,02 înainte de 2020 și 0,39 din 2020 (Fisher z = 10,7). Ce rezultă?",
                "options": [
                    "Corelația este stabilă",
                    "Bitcoin a devenit semnificativ mai corelat cu acțiunile după 2020",
                    "Bitcoin a devenit o acoperire pentru acțiuni",
                    "Diferența se datorează întîmplării"
                ],
                "correctExplanation": "O statistică Fisher z de 10,7 respinge egalitatea corelațiilor: din 2020, activele cripto se comportă ca active riscante.",
                "incorrectExplanation": "O statistică z de peste 10 exclude întîmplarea și stabilitatea; o corelație pozitivă de 0,39 este opusul unei acoperiri."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Safe haven",
                "text": "On the worst 5% of S&P 500 days (index on average -2.70%), Bitcoin returned on average -2.77% (t = -4.9), gold +0.10%. Is Bitcoin a safe haven?",
                "options": [
                    "No: it falls as much as equities on their worst days, while gold holds its value",
                    "Yes: its return is close to that of gold",
                    "Yes: its average return is positive",
                    "The data cannot tell"
                ],
                "correctExplanation": "A safe haven keeps its value in market crashes (Baur and Lucey, 2010): gold does, Bitcoin does not.",
                "incorrectExplanation": "The mean of -2.77% is significantly negative and close to the fall of the index itself, so Bitcoin offered no protection on those days."
            },
            "ro": {
                "title": "Activ de refugiu",
                "text": "În cele mai proaste 5% dintre zilele S&P 500 (indicele în medie -2,70%), Bitcoin a avut în medie -2,77% (t = -4,9), aurul +0,10%. Este Bitcoin un activ de refugiu?",
                "options": [
                    "Nu: scade cît acțiunile în zilele lor cele mai proaste, în timp ce aurul își păstrează valoarea",
                    "Da: randamentul lui este apropiat de cel al aurului",
                    "Da: randamentul lui mediu este pozitiv",
                    "Datele nu pot răspunde"
                ],
                "correctExplanation": "Un activ de refugiu își păstrează valoarea în timpul crahurilor (Baur și Lucey, 2010): aurul o face, Bitcoin nu.",
                "incorrectExplanation": "Media de -2,77% este semnificativ negativă și apropiată de scăderea indicelui însuși, deci Bitcoin nu a oferit nicio protecție în acele zile."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Crypto-specific crises",
                "text": "From 5 to 12 May 2022 (Terra/Luna), Bitcoin fell by 23% and the S&P 500 by 5%; around FTX (November 2022) Bitcoin fell by 30% and the S&P 500 rose by 5%. What do these episodes show?",
                "options": [
                    "Crypto crashes always spread to equities",
                    "Equities and crypto are independent",
                    "Gold fell most",
                    "Crypto-specific shocks hit crypto hard and equities little: the dependence is asymmetric"
                ],
                "correctExplanation": "Global shocks such as COVID-19 hit both markets; shocks born inside crypto stayed mostly inside crypto.",
                "incorrectExplanation": "In both 2022 episodes the S&P 500 moved far less than crypto (and even rose around FTX), so the crashes did not spread, although the markets are correlated on average."
            },
            "ro": {
                "title": "Crize specifice cripto",
                "text": "Între 5 și 12 mai 2022 (Terra/Luna), Bitcoin a scăzut cu 23%, iar S&P 500 cu 5%; în jurul FTX (noiembrie 2022) Bitcoin a scăzut cu 30%, iar S&P 500 a crescut cu 5%. Ce arată aceste episoade?",
                "options": [
                    "Crahurile cripto se extind întotdeauna la acțiuni",
                    "Acțiunile și activele cripto sînt independente",
                    "Aurul a scăzut cel mai mult",
                    "Șocurile specifice cripto lovesc puternic piața cripto și puțin acțiunile: dependența este asimetrică"
                ],
                "correctExplanation": "Șocurile globale, ca COVID-19, lovesc ambele piețe; șocurile născute în interiorul pieței cripto au rămas în mare parte acolo.",
                "incorrectExplanation": "În ambele episoade din 2022, S&P 500 s-a mișcat mult mai puțin decît activele cripto (și chiar a crescut în jurul FTX), deci crahurile nu s-au extins, deși piețele sînt corelate în medie."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Recovering from a drawdown",
                "text": "A crypto asset falls 80% from its record high. By how much must it rise to regain the record?",
                "options": [
                    "80%",
                    "160%",
                    "400%",
                    "125%"
                ],
                "correctExplanation": "A loss d needs a gain d/(1 - d) = 0.8/0.2 = 4, i.e. 400%.",
                "incorrectExplanation": "After an 80% fall the price is one fifth of the record, so it must be multiplied by five: a gain of 400%, far more than the 80% lost."
            },
            "ro": {
                "title": "Recuperarea unui drawdown",
                "text": "Un activ cripto scade cu 80% față de maximul istoric. Cu cît trebuie să crească pentru a reveni la maxim?",
                "options": [
                    "80%",
                    "160%",
                    "400%",
                    "125%"
                ],
                "correctExplanation": "O pierdere d cere un cîștig d/(1 - d) = 0,8/0,2 = 4, adică 400%.",
                "incorrectExplanation": "După o scădere de 80%, prețul este o cincime din maxim, deci trebuie înmulțit cu cinci: un cîștig de 400%, mult mai mult decît cele 80 de procente pierdute."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Bubble models",
                "text": "What is the main caution when a log-periodic power law (LPPL) is fitted to a crypto price?",
                "options": [
                    "It cannot describe faster-than-exponential growth",
                    "With seven parameters and many local optima, the critical time t_c is very uncertain before the crash",
                    "It needs dividend data",
                    "It assumes Normal returns"
                ],
                "correctExplanation": "LPPL fits can look convincing after the event; ex ante the estimated crash date moves a lot with the sample and the starting values.",
                "incorrectExplanation": "The LPPL is designed precisely for super-exponential growth with oscillations and uses only prices; its weakness is the instability of t_c, not missing dividends or a Normal assumption."
            },
            "ro": {
                "title": "Modele de bule",
                "text": "Care este principala precauție cînd se estimează o lege de putere log-periodică (LPPL) pe un preț cripto?",
                "options": [
                    "Nu poate descrie o creștere mai rapidă decît cea exponențială",
                    "Cu șapte parametri și multe optime locale, momentul critic t_c este foarte incert înaintea crahului",
                    "Are nevoie de date despre dividende",
                    "Presupune randamente Normale"
                ],
                "correctExplanation": "Ajustările LPPL pot părea convingătoare după eveniment; ex ante, data estimată a crahului se schimbă mult cu eșantionul și cu valorile inițiale.",
                "incorrectExplanation": "LPPL este conceput tocmai pentru creșteri mai rapide decît cele exponențiale, cu oscilații, și folosește doar prețuri; slăbiciunea lui este instabilitatea lui t_c, nu lipsa dividendelor sau o ipoteză de normalitate."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Spot ETFs",
                "text": "Bitcoin's annualised volatility was 55.2% in the two years before the spot ETFs (11 January 2024) and 47.5% in the two years after (Brown-Forsythe p = 0.27). What can be concluded?",
                "options": [
                    "The fall is not significant, and a before-after comparison cannot identify a causal effect",
                    "The ETFs reduced volatility by 7.7 points",
                    "The ETFs increased volatility",
                    "Volatility is now equal to that of the S&P 500"
                ],
                "correctExplanation": "A p-value of 0.27 does not reject equal variances, and rates, the halving and the US election changed in the same window.",
                "incorrectExplanation": "The difference of 7.7 points is within the noise of two two-year windows, and without a control group it cannot be attributed to the ETFs."
            },
            "ro": {
                "title": "ETF-urile spot",
                "text": "Volatilitatea anuală a Bitcoin a fost 55,2% în cei doi ani dinaintea ETF-urilor spot (11 ianuarie 2024) și 47,5% în cei doi ani de după (Brown-Forsythe p = 0,27). Ce se poate concluziona?",
                "options": [
                    "Scăderea nu este semnificativă, iar o comparație înainte-după nu poate identifica un efect cauzal",
                    "ETF-urile au redus volatilitatea cu 7,7 puncte",
                    "ETF-urile au crescut volatilitatea",
                    "Volatilitatea este acum egală cu cea a S&P 500"
                ],
                "correctExplanation": "Un p-value de 0,27 nu respinge egalitatea varianțelor, iar dobînzile, halving-ul și alegerile din SUA s-au schimbat în aceeași fereastră.",
                "incorrectExplanation": "Diferența de 7,7 puncte se încadrează în zgomotul a două ferestre de doi ani, iar fără un grup de control nu poate fi atribuită ETF-urilor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Tracking error",
                "text": "The daily returns of the IBIT ETF and of Bitcoin have a correlation of 0.91 and a tracking error of about 21% a year, although the fund holds Bitcoin one to one. Why?",
                "options": [
                    "The fund is badly managed",
                    "The fund holds Bitcoin futures",
                    "The fund charges a 21% fee",
                    "The two closing prices are about four hours apart (16:00 New York against midnight UTC)"
                ],
                "correctExplanation": "Asynchronous closes create day-to-day differences that cancel over longer horizons: a measurement artefact.",
                "incorrectExplanation": "A spot fund holds the coin itself and charges a small fee; the gap comes from the timing of the two prices, not from the management or futures."
            },
            "ro": {
                "title": "Tracking error",
                "text": "Randamentele zilnice ale ETF-ului IBIT și ale Bitcoin au o corelație de 0,91 și un tracking error de circa 21% pe an, deși fondul deține Bitcoin unu la unu. De ce?",
                "options": [
                    "Fondul este administrat prost",
                    "Fondul deține contracte futures pe Bitcoin",
                    "Fondul percepe un comision de 21%",
                    "Cele două prețuri de închidere sînt la circa patru ore distanță (ora 16:00 la New York față de miezul nopții UTC)"
                ],
                "correctExplanation": "Închiderile asincrone creează diferențe de la o zi la alta care se compensează pe orizonturi mai lungi: un artefact de măsurare.",
                "incorrectExplanation": "Un fond spot deține moneda însăși și percepe un comision mic; diferența vine din momentul celor două prețuri, nu din administrare sau din contracte futures."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Types of stablecoins",
                "text": "Which stablecoin was algorithmic, backed only by a second token of its own network?",
                "options": [
                    "USDC",
                    "USDT",
                    "TerraUSD (UST)",
                    "DAI"
                ],
                "correctExplanation": "UST was kept at 1 USD by swaps with LUNA; when LUNA fell, the backing disappeared and UST collapsed in May 2022.",
                "incorrectExplanation": "USDC and USDT are backed by dollar reserves and DAI by over-collateralised crypto loans; only UST relied on an algorithm and its own token."
            },
            "ro": {
                "title": "Tipuri de stablecoins",
                "text": "Care stablecoin era algoritmic, garantat doar de un al doilea token al propriei rețele?",
                "options": [
                    "USDC",
                    "USDT",
                    "TerraUSD (UST)",
                    "DAI"
                ],
                "correctExplanation": "UST era ținut la 1 USD prin schimburi cu LUNA; cînd LUNA a scăzut, garanția a dispărut, iar UST s-a prăbușit în mai 2022.",
                "incorrectExplanation": "USDC și USDT sînt garantate cu rezerve în dolari, iar DAI cu credite cripto supra-garantate; doar UST se baza pe un algoritm și pe propriul token."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Basis points",
                "text": "USDC closed at 0.9715 USD on 11 March 2023. What is the peg deviation?",
                "options": [
                    "-2.85 bp",
                    "-285 bp",
                    "-28.5 bp",
                    "-0.0285 bp"
                ],
                "correctExplanation": "d = 10,000 x (0.9715 - 1) = -285 basis points, i.e. -2.85%.",
                "incorrectExplanation": "One basis point is 0.01%, so 2.85% equals 285 bp; the other options confuse percent, basis points and decimals."
            },
            "ro": {
                "title": "Puncte de bază",
                "text": "USDC s-a închis la 0,9715 USD pe 11 martie 2023. Cît este abaterea de la paritate?",
                "options": [
                    "-2,85 bp",
                    "-285 bp",
                    "-28,5 bp",
                    "-0,0285 bp"
                ],
                "correctExplanation": "d = 10.000 x (0,9715 - 1) = -285 de puncte de bază, adică -2,85%.",
                "incorrectExplanation": "Un punct de bază este 0,01%, deci 2,85% înseamnă 285 bp; celelalte variante confundă procentele, punctele de bază și zecimalele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The USDC depeg",
                "text": "Why did USDC fall to 0.88 USD on 11 March 2023 and return to 1 USD on 13 March?",
                "options": [
                    "Part of its reserves was at Silicon Valley Bank, closed on 10 March; on 12 March US authorities guaranteed all deposits",
                    "Its algorithm failed and was repaired",
                    "Bitcoin fell by 50% that weekend",
                    "Its issuer stopped redemptions permanently"
                ],
                "correctExplanation": "A fiat-backed coin is as safe as its reserves: doubt about the bank broke the peg, the deposit guarantee restored it.",
                "incorrectExplanation": "USDC has no algorithm and never stopped redemptions for good; the cause was its bank, not the price of Bitcoin."
            },
            "ro": {
                "title": "Depeg-ul USDC",
                "text": "De ce a scăzut USDC la 0,88 USD pe 11 martie 2023 și a revenit la 1 USD pe 13 martie?",
                "options": [
                    "O parte din rezerve se afla la Silicon Valley Bank, închisă pe 10 martie; pe 12 martie autoritățile americane au garantat toate depozitele",
                    "Algoritmul lui a eșuat și a fost reparat",
                    "Bitcoin a scăzut cu 50% în acel weekend",
                    "Emitentul a oprit definitiv răscumpărările"
                ],
                "correctExplanation": "O monedă garantată fiat este la fel de sigură ca rezervele ei: îndoiala privind banca a rupt paritatea, garantarea depozitelor a refăcut-o.",
                "incorrectExplanation": "USDC nu are un algoritm și nu a oprit definitiv răscumpărările; cauza a fost banca la care avea rezervele, nu prețul Bitcoin."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "VaR in dollars",
                "text": "The historical VaR 1% of daily Bitcoin log returns is 10.54%. What is the VaR of a position of 100,000 USD?",
                "options": [
                    "10,540 USD, VaR 99%",
                    "-10,540 USD",
                    "1,054 USD",
                    "About 10,006 USD: 100,000 x (1 - exp(-0.1054))"
                ],
                "correctExplanation": "A log return of -10.54% means a price ratio of exp(-0.1054) = 0.900, a loss of 10,006 USD; VaR is reported as a positive loss at the level 1%.",
                "incorrectExplanation": "VaR 1% is a positive loss named by its tail probability; for log returns the dollar loss is W(1 - exp(-v/100)), slightly below W times v/100."
            },
            "ro": {
                "title": "VaR în dolari",
                "text": "VaR 1% istoric al randamentelor logaritmice zilnice ale Bitcoin este 10,54%. Cît este VaR pentru o poziție de 100.000 USD?",
                "options": [
                    "10.540 USD, VaR 99%",
                    "-10.540 USD",
                    "1.054 USD",
                    "Aproximativ 10.006 USD: 100.000 x (1 - exp(-0,1054))"
                ],
                "correctExplanation": "Un randament logaritmic de -10,54% înseamnă un raport al prețurilor exp(-0,1054) = 0,900, deci o pierdere de 10.006 USD; VaR se raportează ca pierdere pozitivă, la nivelul 1%.",
                "incorrectExplanation": "VaR 1% este o pierdere pozitivă, numită după probabilitatea cozii; pentru randamente logaritmice, pierderea în dolari este W(1 - exp(-v/100)), puțin sub W înmulțit cu v/100."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Regulation",
                "text": "What do the EU MiCA regulation and the US GENIUS Act require from fiat-backed stablecoins?",
                "options": [
                    "An algorithm that mints a second token",
                    "Interest payments to holders",
                    "Full reserves in safe, liquid assets and redemption at par",
                    "Listing on a stock exchange"
                ],
                "correctExplanation": "Both frameworks require reserves of at least one dollar (or euro) per coin, redemption at par and supervision of issuers; they forbid paying interest to holders.",
                "incorrectExplanation": "Interest to holders is prohibited, exchange listing is not required, and algorithmic designs without reserves do not fit either framework."
            },
            "ro": {
                "title": "Reglementarea",
                "text": "Ce cer regulamentul european MiCA și legea americană GENIUS Act stablecoin-urilor garantate fiat?",
                "options": [
                    "Un algoritm care emite un al doilea token",
                    "Plata de dobînzi deținătorilor",
                    "Rezerve complete în active sigure și lichide și răscumpărare la paritate",
                    "Listarea la bursă"
                ],
                "correctExplanation": "Ambele cadre cer rezerve de cel puțin un dolar (sau euro) pentru fiecare monedă, răscumpărare la paritate și supravegherea emitenților; ele interzic plata dobînzii către deținători.",
                "incorrectExplanation": "Dobînda către deținători este interzisă, listarea la bursă nu este cerută, iar modelele algoritmice fără rezerve nu se încadrează în niciunul dintre cele două cadre."
            }
        }
    ]
};
