// ============================================================
// Chapter 10 quiz bank: VaR, ES and backtesting (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['var-es'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Definition of VaR",
                "text": "A daily return X has 1% quantile q_0.01 = -3.2%. What is VaR 1%?",
                "options": [
                    "-3.2%",
                    "1%",
                    "3.2%: the loss exceeded with probability 1%",
                    "96.8%"
                ],
                "correctExplanation": "VaR_alpha = -q_alpha: the minus sign turns the negative quantile into a positive loss, exceeded on about one day in a hundred.",
                "incorrectExplanation": "VaR is a loss, a positive number: the negative quantile is the return, and the level 1% is the tail probability, not the VaR itself."
            },
            "ro": {
                "title": "Definiția VaR",
                "text": "Un randament zilnic X are cuantila q_0,01 = -3,2%. Cît este VaR 1%?",
                "options": [
                    "-3,2%",
                    "1%",
                    "3,2%: pierderea depășită cu probabilitatea 1%",
                    "96,8%"
                ],
                "correctExplanation": "VaR_alpha = -q_alpha: semnul minus transformă cuantila negativă într-o pierdere pozitivă, depășită cam într-o zi din o sută.",
                "incorrectExplanation": "VaR este o pierdere, un număr pozitiv: cuantila negativă este randamentul, iar nivelul 1% este probabilitatea cozii, nu VaR însuși."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Naming the level",
                "text": "In this course, how is the level of a VaR stated?",
                "options": [
                    "As the tail probability: VaR 1%, ES 2.5%",
                    "As the probability that the position makes a profit",
                    "As the number of days in the window",
                    "As the multiplier of the Basel rule"
                ],
                "correctExplanation": "The level is the probability of the bad tail: VaR 1% is exceeded with probability 1%, ES 2.5% averages the worst 2.5% of days.",
                "incorrectExplanation": "The course always names the tail probability; the chance of a profit, the window and the Basel multiplier say nothing about the level."
            },
            "ro": {
                "title": "Denumirea nivelului",
                "text": "În acest curs, cum se precizează nivelul unui VaR?",
                "options": [
                    "Prin probabilitatea cozii: VaR 1%, ES 2,5%",
                    "Prin probabilitatea ca poziția să aibă profit",
                    "Prin numărul de zile din fereastră",
                    "Prin factorul de multiplicare al regulii Basel"
                ],
                "correctExplanation": "Nivelul este probabilitatea cozii nefavorabile: VaR 1% este depășit cu probabilitatea 1%, ES 2,5% face media celor mai proaste 2,5% dintre zile.",
                "incorrectExplanation": "Cursul numește întotdeauna probabilitatea cozii; șansa unui profit, fereastra și factorul Basel nu spun nimic despre nivel."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Expected shortfall",
                "text": "What does ES 2.5% measure for a continuous return distribution?",
                "options": [
                    "The largest loss in the sample",
                    "The average loss on the worst 2.5% of days",
                    "The loss exceeded with probability 2.5%",
                    "The standard deviation of the losses beyond VaR"
                ],
                "correctExplanation": "ES_alpha = -E[X | X <= q_alpha]: the mean of the losses in the tail of probability alpha.",
                "incorrectExplanation": "The loss exceeded with probability 2.5% is VaR 2.5%; ES averages the whole tail beyond that quantile, it is neither the maximum nor a dispersion."
            },
            "ro": {
                "title": "Expected shortfall",
                "text": "Ce măsoară ES 2,5% pentru o distribuție continuă a randamentelor?",
                "options": [
                    "Cea mai mare pierdere din eșantion",
                    "Pierderea medie din cele mai proaste 2,5% dintre zile",
                    "Pierderea depășită cu probabilitatea 2,5%",
                    "Abaterea standard a pierderilor de dincolo de VaR"
                ],
                "correctExplanation": "ES_alpha = -E[X | X <= q_alpha]: media pierderilor din coada de probabilitate alpha.",
                "incorrectExplanation": "Pierderea depășită cu probabilitatea 2,5% este VaR 2,5%; ES face media întregii cozi de dincolo de această cuantilă, nu este nici maximul, nici o măsură de dispersie."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Normal ES and VaR",
                "text": "For Normal returns with mean 0, how do ES 2.5% and VaR 1% compare?",
                "options": [
                    "ES 2.5% is twice VaR 1%",
                    "ES 2.5% is half of VaR 1%",
                    "They cannot be compared",
                    "They are almost equal (ratio about 1.005)"
                ],
                "correctExplanation": "VaR 1% = 2.326 sigma and ES 2.5% = sigma phi(1.960)/0.025 = 2.338 sigma: this is why Basel could replace VaR 1% by ES 2.5%.",
                "incorrectExplanation": "Both are multiples of sigma under the Normal: 2.326 and 2.338, so neither double nor half; the difference appears only with heavy tails."
            },
            "ro": {
                "title": "ES și VaR Normale",
                "text": "Pentru randamente Normale cu media 0, cum se compară ES 2,5% și VaR 1%?",
                "options": [
                    "ES 2,5% este dublul VaR 1%",
                    "ES 2,5% este jumătate din VaR 1%",
                    "Nu pot fi comparate",
                    "Sînt aproape egale (raport de circa 1,005)"
                ],
                "correctExplanation": "VaR 1% = 2,326 sigma și ES 2,5% = sigma phi(1,960)/0,025 = 2,338 sigma: de aceea Basel a putut înlocui VaR 1% cu ES 2,5%.",
                "incorrectExplanation": "Pentru distribuția Normală ambele sînt multipli ai lui sigma: 2,326 și 2,338, deci nici dublu, nici jumătate; diferența apare doar la cozi groase."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Historical simulation",
                "text": "With 500 daily returns, which order statistic gives the historical VaR 1%?",
                "options": [
                    "The smallest return",
                    "The 1st percentile of the absolute returns",
                    "Minus the 5th smallest return",
                    "The mean of the 5 smallest returns"
                ],
                "correctExplanation": "k = ceil(n alpha) = ceil(5) = 5: VaR 1% = -r_(5).",
                "incorrectExplanation": "The smallest return alone is the worst day, the mean of the five smallest is an ES-type quantity, and absolute returns mix gains with losses."
            },
            "ro": {
                "title": "Simularea istorică",
                "text": "Cu 500 de randamente zilnice, ce statistică de ordine dă VaR 1% istoric?",
                "options": [
                    "Cel mai mic randament",
                    "Percentila 1 a randamentelor în valoare absolută",
                    "Minus al cincilea cel mai mic randament",
                    "Media celor mai mici 5 randamente"
                ],
                "correctExplanation": "k = ceil(n alpha) = ceil(5) = 5: VaR 1% = -r_(5).",
                "incorrectExplanation": "Cel mai mic randament singur este cea mai proastă zi, media celor mai mici cinci este o mărime de tip ES, iar valorile absolute amestecă cîștigurile cu pierderile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ES and VaR at different levels",
                "text": "For the BET over the last 500 days, historical ES 2.5% was 2.69% and VaR 1% was 2.93%. Is this possible?",
                "options": [
                    "Yes: ES is above VaR only at the same level",
                    "No: ES is always above VaR",
                    "No: it shows a computing error",
                    "Yes, but only for Normal returns"
                ],
                "correctExplanation": "ES_alpha >= VaR_alpha holds at the same alpha; ES 2.5% averages a wider tail and can be below VaR 1%.",
                "incorrectExplanation": "The inequality between ES and VaR compares the same level; across levels anything can happen, also for non-Normal data."
            },
            "ro": {
                "title": "ES și VaR la niveluri diferite",
                "text": "Pentru BET în ultimele 500 de zile, ES 2,5% istoric a fost 2,69%, iar VaR 1% a fost 2,93%. Este posibil?",
                "options": [
                    "Da: ES este peste VaR doar la același nivel",
                    "Nu: ES este întotdeauna peste VaR",
                    "Nu: arată o eroare de calcul",
                    "Da, dar doar pentru randamente Normale"
                ],
                "correctExplanation": "ES_alpha >= VaR_alpha este valabilă la același alpha; ES 2,5% face media unei cozi mai largi și poate fi sub VaR 1%.",
                "incorrectExplanation": "Inegalitatea dintre ES și VaR compară același nivel; între niveluri diferite se poate întîmpla orice, și pentru date care nu sînt Normale."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Subadditivity",
                "text": "Which axiom of a coherent risk measure can VaR violate?",
                "options": [
                    "Monotonicity",
                    "Subadditivity",
                    "Translation invariance",
                    "Positive homogeneity"
                ],
                "correctExplanation": "Two independent bonds with a 4% default probability each have VaR 5% equal to 0, but their sum has VaR 5% equal to 100.",
                "incorrectExplanation": "VaR is monotone, translation invariant and positively homogeneous; only subadditivity can fail, for example for discrete credit losses."
            },
            "ro": {
                "title": "Subaditivitatea",
                "text": "Ce axiomă a unei măsuri de risc coerente poate fi încălcată de VaR?",
                "options": [
                    "Monotonia",
                    "Subaditivitatea",
                    "Invarianța la translație",
                    "Omogenitatea pozitivă"
                ],
                "correctExplanation": "Două obligațiuni independente, fiecare cu probabilitatea de neplată 4%, au VaR 5% egal cu 0, dar suma lor are VaR 5% egal cu 100.",
                "incorrectExplanation": "VaR este monoton, invariant la translație și pozitiv omogen; doar subaditivitatea poate să nu fie îndeplinită, de exemplu pentru pierderi discrete din credit."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Elicitability",
                "text": "Which statement about elicitability is correct?",
                "options": [
                    "ES alone is elicitable, VaR is not",
                    "Neither VaR nor ES can be compared across models",
                    "Both are elicitable by the squared error",
                    "VaR is elicitable by the quantile loss; ES only jointly with VaR"
                ],
                "correctExplanation": "Gneiting (2011): the quantile (pinball) loss is minimised by the true quantile; Fissler and Ziegel (2016): the pair (VaR, ES) is elicitable.",
                "incorrectExplanation": "The squared error elicits the mean, not a quantile; ES needs VaR to be scored, and VaR forecasts can be ranked by their average quantile loss."
            },
            "ro": {
                "title": "Elicitabilitatea",
                "text": "Care afirmație despre elicitabilitate este corectă?",
                "options": [
                    "ES singur este elicitabil, VaR nu",
                    "Nici VaR, nici ES nu pot fi comparate între modele",
                    "Ambele sînt elicitabile prin eroarea pătratică",
                    "VaR este elicitabil prin pierderea cuantilă; ES doar împreună cu VaR"
                ],
                "correctExplanation": "Gneiting (2011): pierderea cuantilă (pinball) este minimizată de cuantila adevărată; Fissler și Ziegel (2016): perechea (VaR, ES) este elicitabilă.",
                "incorrectExplanation": "Eroarea pătratică elicitează media, nu o cuantilă; ES are nevoie de VaR pentru a fi evaluat, iar prognozele VaR pot fi ordonate după pierderea cuantilă medie."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Normal VaR on real data",
                "text": "On daily S&P 500 returns since 2000, the Normal VaR 1% was 2.80% and the historical VaR 1% was 3.46%. Why?",
                "options": [
                    "The sample is too short",
                    "The Normal mean is too high",
                    "The Normal tail is too thin for returns with excess kurtosis",
                    "Historical simulation always overstates risk"
                ],
                "correctExplanation": "With an excess kurtosis of about 11, extreme days are far more frequent than the Normal allows, so its 1% quantile is too close to zero.",
                "incorrectExplanation": "The sample has more than 6,700 days and the mean is tiny; the gap comes from the thin Normal tail, not from a bias of historical simulation."
            },
            "ro": {
                "title": "VaR Normal pe date reale",
                "text": "Pe randamentele zilnice S&P 500 din 2000, VaR 1% Normal a fost 2,80%, iar VaR 1% istoric 3,46%. De ce?",
                "options": [
                    "Eșantionul este prea scurt",
                    "Media Normală este prea mare",
                    "Coada distribuției Normale este prea subțire pentru randamente cu exces de boltire",
                    "Simularea istorică supraestimează întotdeauna riscul"
                ],
                "correctExplanation": "Cu un exces de boltire de circa 11, zilele extreme sînt mult mai frecvente decît permite distribuția Normală, deci cuantila ei de 1% este prea aproape de zero.",
                "incorrectExplanation": "Eșantionul are peste 6700 de zile, iar media este foarte mică; diferența vine din coada subțire a distribuției Normale, nu dintr-o abatere a simulării istorice."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Cornish-Fisher",
                "text": "Why did the Cornish-Fisher VaR 1% of the S&P 500 come out at about 6%, almost twice the historical value?",
                "options": [
                    "The expansion is a small correction and fails for an excess kurtosis above 10",
                    "The skewness of the S&P 500 is positive",
                    "Cornish-Fisher ignores the kurtosis",
                    "The Normal quantile was taken at 5%"
                ],
                "correctExplanation": "The expansion adds terms in S and K to the Normal quantile; with K around 11 the correction overshoots badly.",
                "incorrectExplanation": "The S&P 500 skewness is negative, the kurtosis enters the formula directly, and the 1% quantile -2.326 was used: the problem is the size of K."
            },
            "ro": {
                "title": "Cornish-Fisher",
                "text": "De ce a ieșit VaR 1% Cornish-Fisher pentru S&P 500 de circa 6%, aproape dublul valorii istorice?",
                "options": [
                    "Dezvoltarea este o corecție mică și nu funcționează pentru un exces de boltire de peste 10",
                    "Asimetria S&P 500 este pozitivă",
                    "Cornish-Fisher ignoră boltirea",
                    "Cuantila Normală a fost luată la 5%"
                ],
                "correctExplanation": "Dezvoltarea adaugă la cuantila Normală termeni în S și K; cu K în jur de 11, corecția depășește mult valoarea corectă.",
                "incorrectExplanation": "Asimetria S&P 500 este negativă, boltirea intră direct în formulă și s-a folosit cuantila de 1%, -2,326: problema este mărimea lui K."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "EVT",
                "text": "In the POT method, the threshold u is the 90% quantile of the losses. What is fitted above u?",
                "options": [
                    "A Normal distribution",
                    "A generalised Pareto distribution (GPD) to the excesses L - u",
                    "A GARCH model",
                    "A Student-t copula"
                ],
                "correctExplanation": "Above a high threshold the excesses are approximately GPD (Chapter 5); VaR and ES follow from the fitted shape xi and scale beta.",
                "incorrectExplanation": "The tail beyond u is modelled by the GPD; Normal tails are too thin, GARCH models volatility over time and copulas model dependence."
            },
            "ro": {
                "title": "EVT",
                "text": "În metoda POT, pragul u este cuantila de 90% a pierderilor. Ce se estimează peste u?",
                "options": [
                    "O distribuție Normală",
                    "O distribuție Pareto generalizată (GPD) pentru excesele L - u",
                    "Un model GARCH",
                    "O copulă Student-t"
                ],
                "correctExplanation": "Peste un prag ridicat, excesele urmează aproximativ o distribuție GPD (Capitolul 5); VaR și ES rezultă din forma xi și scala beta estimate.",
                "incorrectExplanation": "Coada de dincolo de u se modelează prin GPD; cozile Normale sînt prea subțiri, GARCH modelează volatilitatea în timp, iar copulele modelează dependența."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Conditional VaR",
                "text": "How is the one-day GARCH-t VaR 1% computed?",
                "options": [
                    "-(mu + 2.326 sigma), with sigma the sample standard deviation",
                    "The 5th worst return of the last 500 days",
                    "sqrt(10) times yesterday's VaR",
                    "-(mu + sigma_{t+1} q_0.01(z)), with q the 1% quantile of the standardised t"
                ],
                "correctExplanation": "The volatility forecast sigma_{t+1} follows the GARCH recursion and the quantile of the standardised Student-t innovations replaces the Normal one.",
                "incorrectExplanation": "The sample standard deviation and the 500-day window are unconditional methods; sqrt(10) changes the horizon, not the conditioning."
            },
            "ro": {
                "title": "VaR condiționat",
                "text": "Cum se calculează VaR 1% GARCH-t pe o zi?",
                "options": [
                    "-(mu + 2,326 sigma), cu sigma abaterea standard de selecție",
                    "A cincea cea mai proastă zi din ultimele 500",
                    "sqrt(10) ori VaR-ul de ieri",
                    "-(mu + sigma_{t+1} q_0,01(z)), cu q cuantila de 1% a distribuției t standardizate"
                ],
                "correctExplanation": "Prognoza volatilității sigma_{t+1} vine din recurența GARCH, iar cuantila inovațiilor Student-t standardizate înlocuiește cuantila Normală.",
                "incorrectExplanation": "Abaterea standard de selecție și fereastra de 500 de zile sînt metode necondiționate; sqrt(10) schimbă orizontul, nu condiționarea."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Filtered historical simulation",
                "text": "What does FHS (filtered historical simulation) take from the data?",
                "options": [
                    "Only the sample mean",
                    "The last 500 raw returns",
                    "The empirical quantiles of the GARCH standardised residuals",
                    "The Basel multiplier"
                ],
                "correctExplanation": "FHS scales the empirical distribution of z_t = (r_t - mu)/sigma_t by tomorrow's sigma_{t+1} (Hull and White, 1998).",
                "incorrectExplanation": "Raw returns are used by plain historical simulation; FHS first filters them by the GARCH volatility, and the multiplier belongs to capital rules."
            },
            "ro": {
                "title": "Simularea istorică filtrată",
                "text": "Ce preia FHS (simularea istorică filtrată) din date?",
                "options": [
                    "Doar media de selecție",
                    "Ultimele 500 de randamente brute",
                    "Cuantilele empirice ale reziduurilor standardizate GARCH",
                    "Factorul de multiplicare Basel"
                ],
                "correctExplanation": "FHS scalează distribuția empirică a lui z_t = (r_t - mu)/sigma_t cu sigma_{t+1} de mîine (Hull și White, 1998).",
                "incorrectExplanation": "Randamentele brute sînt folosite de simularea istorică simplă; FHS le filtrează întîi prin volatilitatea GARCH, iar factorul de multiplicare ține de regulile de capital."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Square-root-of-time",
                "text": "When is the 10-day VaR exactly sqrt(10) times the one-day VaR?",
                "options": [
                    "For i.i.d. Normal returns with mean 0",
                    "Always",
                    "For GARCH returns after a storm",
                    "For heavy-tailed i.i.d. returns"
                ],
                "correctExplanation": "Only then is the 10-day return Normal with standard deviation sigma sqrt(10), so all quantiles scale by sqrt(10).",
                "incorrectExplanation": "Heavy tails make sums closer to Normal and volatility clustering makes the 10-day risk depend on today's state, so the rule is only an approximation."
            },
            "ro": {
                "title": "Rădăcina pătrată a timpului",
                "text": "Cînd este VaR-ul pe 10 zile exact sqrt(10) ori VaR-ul pe o zi?",
                "options": [
                    "Pentru randamente Normale i.i.d. cu media 0",
                    "Întotdeauna",
                    "Pentru randamente GARCH după o furtună",
                    "Pentru randamente i.i.d. cu cozi groase"
                ],
                "correctExplanation": "Doar atunci randamentul pe 10 zile este Normal, cu abaterea standard sigma sqrt(10), deci toate cuantilele se scalează cu sqrt(10).",
                "incorrectExplanation": "Cozile groase fac sumele mai apropiate de Normală, iar volatility clustering face ca riscul pe 10 zile să depindă de starea de azi, deci regula este doar o aproximare."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Kupiec test",
                "text": "A VaR 1% is exceeded on 7 of 250 days. What does the Kupiec test conclude at 5%?",
                "options": [
                    "Not rejected, because 7 is close to 2.5",
                    "Rejected, because any exception is too many",
                    "The test cannot be applied to 250 days",
                    "Rejected: LR_uc is about 5.5, above 3.84"
                ],
                "correctExplanation": "ln L0 = 243 ln 0.99 + 7 ln 0.01 and ln L1 = 243 ln 0.972 + 7 ln 0.028 give LR_uc = 5.50 and p = 0.019.",
                "incorrectExplanation": "Under a correct model about 2.5 exceptions are expected and a few are normal; 7 is too many at 5%, and the test works for any sample size."
            },
            "ro": {
                "title": "Testul Kupiec",
                "text": "Un VaR 1% este depășit în 7 din 250 de zile. Ce concluzie dă testul Kupiec la 5%?",
                "options": [
                    "Nerespins, deoarece 7 este aproape de 2,5",
                    "Respins, deoarece orice depășire este prea mult",
                    "Testul nu se poate aplica pe 250 de zile",
                    "Respins: LR_uc este circa 5,5, peste 3,84"
                ],
                "correctExplanation": "ln L0 = 243 ln 0,99 + 7 ln 0,01 și ln L1 = 243 ln 0,972 + 7 ln 0,028 dau LR_uc = 5,50 și p = 0,019.",
                "incorrectExplanation": "Pentru un model corect se așteaptă circa 2,5 depășiri și cîteva sînt normale; 7 înseamnă prea multe la 5%, iar testul funcționează pentru orice mărime a eșantionului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Basel traffic light",
                "text": "A bank has 7 exceptions of its VaR 1% in the last 250 days. Which zone and multiplier apply?",
                "options": [
                    "Green, multiplier 3",
                    "Yellow, multiplier 3.65",
                    "Red, multiplier 4",
                    "Yellow, multiplier 4.5"
                ],
                "correctExplanation": "Yellow zone (5 to 9 exceptions); the plus factor for 7 exceptions is 0.65, so the multiplier is 3 + 0.65.",
                "incorrectExplanation": "Green ends at 4 exceptions and red starts at 10; the multiplier never exceeds 4 under the 1996 rule."
            },
            "ro": {
                "title": "Semaforul Basel",
                "text": "O bancă are 7 depășiri ale VaR 1% în ultimele 250 de zile. Ce zonă și ce factor se aplică?",
                "options": [
                    "Verde, factorul 3",
                    "Galbenă, factorul 3,65",
                    "Roșie, factorul 4",
                    "Galbenă, factorul 4,5"
                ],
                "correctExplanation": "Zona galbenă (5 pînă la 9 depășiri); adaosul pentru 7 depășiri este 0,65, deci factorul este 3 + 0,65.",
                "incorrectExplanation": "Zona verde se termină la 4 depășiri, iar cea roșie începe la 10; factorul nu depășește niciodată 4 în regula din 1996."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Christoffersen test",
                "text": "A model has exactly the expected number of exceptions, but they all occur in one month. Which test detects the problem?",
                "options": [
                    "The Kupiec POF test",
                    "The Jarque-Bera test",
                    "The Christoffersen independence test",
                    "The Diebold-Mariano test"
                ],
                "correctExplanation": "The independence test compares the probability of an exception after an exception with the probability after a normal day.",
                "incorrectExplanation": "Kupiec checks only the number of exceptions; Jarque-Bera tests Normality and Diebold-Mariano compares forecast losses."
            },
            "ro": {
                "title": "Testul Christoffersen",
                "text": "Un model are exact numărul așteptat de depășiri, dar toate apar într-o singură lună. Ce test detectează problema?",
                "options": [
                    "Testul POF al lui Kupiec",
                    "Testul Jarque-Bera",
                    "Testul de independență al lui Christoffersen",
                    "Testul Diebold-Mariano"
                ],
                "correctExplanation": "Testul de independență compară probabilitatea unei depășiri după o depășire cu probabilitatea după o zi obișnuită.",
                "incorrectExplanation": "Kupiec verifică doar numărul depășirilor; Jarque-Bera testează normalitatea, iar Diebold-Mariano compară pierderile prognozelor."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Conditional coverage",
                "text": "How is the conditional coverage statistic LR_cc built?",
                "options": [
                    "LR_cc = LR_uc + LR_ind, compared with chi-square(2)",
                    "LR_cc = LR_uc - LR_ind, compared with chi-square(1)",
                    "LR_cc = LR_uc x LR_ind",
                    "LR_cc is the number of exceptions divided by n"
                ],
                "correctExplanation": "It adds the Kupiec statistic (the rate) and the independence statistic (the clustering); each has one degree of freedom.",
                "incorrectExplanation": "The two likelihood-ratio statistics add up, so the reference distribution has two degrees of freedom; a ratio of counts is only the observed rate."
            },
            "ro": {
                "title": "Acoperirea condiționată",
                "text": "Cum se construiește statistica de acoperire condiționată LR_cc?",
                "options": [
                    "LR_cc = LR_uc + LR_ind, comparată cu chi-pătrat(2)",
                    "LR_cc = LR_uc - LR_ind, comparată cu chi-pătrat(1)",
                    "LR_cc = LR_uc x LR_ind",
                    "LR_cc este numărul depășirilor împărțit la n"
                ],
                "correctExplanation": "Adună statistica Kupiec (rata) și statistica de independență (gruparea); fiecare are un grad de libertate.",
                "incorrectExplanation": "Cele două statistici ale raportului de verosimilitate se adună, deci distribuția de referință are două grade de libertate; un raport de numărători este doar rata observată."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Backtests in 2008",
                "text": "Between September 2008 and March 2009, the 500-day Normal VaR 1% of the S&P 500 was exceeded on 21 of 146 days. What is the main lesson?",
                "options": [
                    "The Normal VaR was too conservative",
                    "Twenty-one exceptions are within the green zone",
                    "Crises cannot be backtested",
                    "A slow unconditional VaR fails exactly when volatility jumps"
                ],
                "correctExplanation": "About 1.5 exceptions were expected; the window method and the thin Normal tail both lagged behind the sudden rise in volatility.",
                "incorrectExplanation": "Twenty-one exceptions are far beyond the red-zone limit and show that the VaR was too low, not too high; the backtest is what reveals it."
            },
            "ro": {
                "title": "Backtesting în 2008",
                "text": "Între septembrie 2008 și martie 2009, VaR 1% Normal pe 500 de zile pentru S&P 500 a fost depășit în 21 din 146 de zile. Care este lecția principală?",
                "options": [
                    "VaR-ul Normal a fost prea prudent",
                    "Douăzeci și una de depășiri sînt în zona verde",
                    "Crizele nu pot fi testate prin backtesting",
                    "Un VaR necondiționat lent greșește exact cînd volatilitatea crește brusc"
                ],
                "correctExplanation": "Se așteptau circa 1,5 depășiri; metoda pe fereastră și coada subțire a distribuției Normale au rămas în urma creșterii bruște a volatilității.",
                "incorrectExplanation": "Douăzeci și una de depășiri sînt mult peste limita zonei roșii și arată că VaR a fost prea mic, nu prea mare; tocmai backtesting-ul dezvăluie acest lucru."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ES backtesting",
                "text": "In the Acerbi-Szekely Z2 test, what does a significantly negative Z2 mean?",
                "options": [
                    "The ES forecasts are too large",
                    "The losses beyond VaR are larger or more frequent than the ES forecasts imply",
                    "The VaR has too few exceptions",
                    "The returns are Normal"
                ],
                "correctExplanation": "Z2 = sum r_t I_t / (T alpha ES_t) + 1 has mean 0 under a correct model; large tail losses relative to ES push it below 0.",
                "incorrectExplanation": "Too large ES forecasts give a positive Z2, few exceptions also push it up, and the test says nothing about Normality."
            },
            "ro": {
                "title": "Backtesting pentru ES",
                "text": "În testul Z2 Acerbi-Szekely, ce înseamnă un Z2 semnificativ negativ?",
                "options": [
                    "Prognozele ES sînt prea mari",
                    "Pierderile de dincolo de VaR sînt mai mari sau mai dese decît arată prognozele ES",
                    "VaR are prea puține depășiri",
                    "Randamentele sînt Normale"
                ],
                "correctExplanation": "Z2 = suma r_t I_t / (T alpha ES_t) + 1 are media 0 pentru un model corect; pierderile mari din coadă, raportate la ES, îl împing sub 0.",
                "incorrectExplanation": "Prognozele ES prea mari dau un Z2 pozitiv, iar puține depășiri îl cresc și ele; testul nu spune nimic despre normalitate."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Portfolio VaR",
                "text": "Two assets with VaR 1% of 3.23% and 2.82% form a 50/50 portfolio with correlation 0.21. The variance-covariance VaR 1% is about 2.35%. What explains the gap to 3.03%?",
                "options": [
                    "An error in the weights",
                    "The square-root-of-time rule",
                    "Diversification: with correlation below 1, sigma_p is below the weighted sum of the sigmas",
                    "The ghost effect"
                ],
                "correctExplanation": "sigma_p^2 = w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 rho s1 s2 is smaller than (w1 s1 + w2 s2)^2 when rho < 1.",
                "incorrectExplanation": "The weights are 0.5 each, the horizon is one day and no window is involved: the gap is the diversification benefit of the Normal model."
            },
            "ro": {
                "title": "VaR de portofoliu",
                "text": "Două active cu VaR 1% de 3,23% și 2,82% formează un portofoliu 50/50 cu corelația 0,21. VaR 1% varianță-covarianță este circa 2,35%. Ce explică diferența față de 3,03%?",
                "options": [
                    "O eroare în ponderi",
                    "Regula rădăcinii pătrate a timpului",
                    "Diversificarea: cu o corelație sub 1, sigma_p este sub suma ponderată a abaterilor standard",
                    "Ghost effect"
                ],
                "correctExplanation": "sigma_p^2 = w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 rho s1 s2 este mai mic decît (w1 s1 + w2 s2)^2 cînd rho < 1.",
                "incorrectExplanation": "Ponderile sînt de 0,5 fiecare, orizontul este o zi și nu intervine nicio fereastră: diferența este beneficiul diversificării în modelul Normal."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Correlation in crises",
                "text": "The 250-day correlation of BET and S&P 500 returns was about 0.20 in 2017 and about 0.71 in spring 2020. What does this imply for portfolio VaR?",
                "options": [
                    "Diversification is weakest when it is needed; a VaR built on the average correlation is too optimistic in a crisis",
                    "Correlation does not matter for VaR",
                    "The portfolio VaR falls in crises",
                    "The BET and the S&P 500 became independent in 2020"
                ],
                "correctExplanation": "A higher correlation raises sigma_p, so the diversification benefit shrinks exactly in the stress period.",
                "incorrectExplanation": "Correlation enters sigma_p directly; a higher correlation means stronger, not weaker, dependence and a larger, not smaller, portfolio VaR."
            },
            "ro": {
                "title": "Corelația în crize",
                "text": "Corelația pe 250 de zile a randamentelor BET și S&P 500 a fost circa 0,20 în 2017 și circa 0,71 în primăvara lui 2020. Ce implică aceasta pentru VaR-ul de portofoliu?",
                "options": [
                    "Diversificarea este cea mai slabă cînd este nevoie de ea; un VaR construit pe corelația medie este prea optimist într-o criză",
                    "Corelația nu contează pentru VaR",
                    "VaR-ul portofoliului scade în crize",
                    "BET și S&P 500 au devenit independente în 2020"
                ],
                "correctExplanation": "O corelație mai mare crește sigma_p, deci beneficiul diversificării se reduce exact în perioada de stres.",
                "incorrectExplanation": "Corelația intră direct în sigma_p; o corelație mai mare înseamnă o dependență mai puternică, nu mai slabă, și un VaR de portofoliu mai mare, nu mai mic."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Copulas",
                "text": "Which statement about the Gaussian and the t copula is correct?",
                "options": [
                    "Both have the same tail dependence for the same rho",
                    "The Gaussian copula has the stronger tail dependence",
                    "A copula also fixes the marginal distributions",
                    "The t copula has positive tail dependence; the Gaussian copula has none"
                ],
                "correctExplanation": "lambda = 2 t_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho))) > 0 for the t copula, while the Gaussian copula has lambda = 0 for rho < 1.",
                "incorrectExplanation": "By Sklar's theorem the copula holds only the dependence, the margins are separate; at the same rho the t copula has more joint extremes."
            },
            "ro": {
                "title": "Copule",
                "text": "Care afirmație despre copula Gaussiană și copula t este corectă?",
                "options": [
                    "Ambele au aceeași dependență în cozi pentru același rho",
                    "Copula Gaussiană are dependența în cozi mai puternică",
                    "O copulă fixează și distribuțiile marginale",
                    "Copula t are dependență în cozi pozitivă; copula Gaussiană nu are"
                ],
                "correctExplanation": "lambda = 2 t_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho))) > 0 pentru copula t, în timp ce copula Gaussiană are lambda = 0 pentru rho < 1.",
                "incorrectExplanation": "Conform teoremei lui Sklar, copula conține doar dependența, iar marginalele sînt separate; la același rho, copula t are mai multe extreme comune."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Joint bad days",
                "text": "Among 6,466 common days of BET and S&P 500, 70 days had both indices among their worst 5%. A fitted Gaussian copula gives about 37 such days, a t copula about 55. What follows?",
                "options": [
                    "The Gaussian copula fits the tails best",
                    "The Gaussian copula understates joint crashes; the t copula is closer to the data",
                    "The two indices are independent",
                    "Copulas cannot be compared with data"
                ],
                "correctExplanation": "Under independence about 16 days would be expected; the data show strong lower tail dependence, which only the t copula partly captures.",
                "incorrectExplanation": "Seventy days are far above the 16 expected under independence and above the Gaussian 37; the counts compare the copulas directly with the data."
            },
            "ro": {
                "title": "Zile proaste comune",
                "text": "Din 6466 de zile comune BET și S&P 500, în 70 de zile ambii indici au fost printre cele mai proaste 5% ale lor. O copulă Gaussiană estimată dă circa 37 de astfel de zile, o copulă t circa 55. Ce rezultă?",
                "options": [
                    "Copula Gaussiană descrie cel mai bine cozile",
                    "Copula Gaussiană subestimează prăbușirile simultane; copula t este mai aproape de date",
                    "Cei doi indici sînt independenți",
                    "Copulele nu pot fi comparate cu datele"
                ],
                "correctExplanation": "La independență s-ar aștepta circa 16 zile; datele arată o dependență puternică în coada inferioară, pe care doar copula t o surprinde parțial.",
                "incorrectExplanation": "Cele 70 de zile sînt mult peste cele 16 așteptate la independență și peste cele 37 ale copulei Gaussiene; numărătorile compară direct copulele cu datele."
            }
        }
    ]
};
