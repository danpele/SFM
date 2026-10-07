// ============================================================
// Chapter 16 quiz bank: review of Chapters 0-15 (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['review'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Annualising Bitcoin",
                "text": "Bitcoin has a daily standard deviation of 3.5%. Which annual volatility is right?",
                "options": [
                    "3.5% × √365 ≈ 66.9%",
                    "3.5% × √252 ≈ 55.6%",
                    "3.5% × 365 ≈ 1277%",
                    "3.5% × 12 = 42%"
                ],
                "correctExplanation": "Bitcoin trades every day, so the actual frequency is 365 observations a year; volatility scales with the square root of that frequency.",
                "incorrectExplanation": "Volatility is annualised with the square root of the actual number of observations per year; 252 is the frequency of shares, not of a market open every day."
            },
            "ro": {
                "title": "Anualizarea Bitcoin",
                "text": "Bitcoin are abaterea standard zilnică de 3,5%. Care este volatilitatea anuală corectă?",
                "options": [
                    "3,5% × √365 ≈ 66,9%",
                    "3,5% × √252 ≈ 55,6%",
                    "3,5% × 365 ≈ 1277%",
                    "3,5% × 12 = 42%"
                ],
                "correctExplanation": "Bitcoin se tranzacționează în fiecare zi, deci frecvența reală este de 365 de observații pe an; volatilitatea se scalează cu rădăcina pătrată a acestei frecvențe.",
                "incorrectExplanation": "Volatilitatea se anualizează cu rădăcina pătrată a numărului real de observații pe an; 252 este frecvența acțiunilor, nu a unei piețe deschise în fiecare zi."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Adding returns",
                "text": "Which returns can be added over time to get the return of a longer period?",
                "options": [
                    "Simple returns",
                    "Log returns",
                    "Both, always",
                    "Neither"
                ],
                "correctExplanation": "ln(P_T/P_0) is the sum of the daily log returns; simple returns compound multiplicatively.",
                "incorrectExplanation": "Simple returns compound: 1 + R(k) is a product; only log returns add up over time."
            },
            "ro": {
                "title": "Adunarea randamentelor",
                "text": "Ce randamente se pot aduna în timp pentru a obține randamentul unei perioade mai lungi?",
                "options": [
                    "Randamentele simple",
                    "Randamentele logaritmice",
                    "Ambele, întotdeauna",
                    "Niciunele"
                ],
                "correctExplanation": "ln(P_T/P_0) este suma randamentelor logaritmice zilnice; randamentele simple se compun multiplicativ.",
                "incorrectExplanation": "Randamentele simple se compun: 1 + R(k) este un produs; doar randamentele logaritmice se adună în timp."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Recovering from a drawdown",
                "text": "An index falls by 50%. What gain is needed to return to the previous peak?",
                "options": [
                    "50%",
                    "75%",
                    "100%",
                    "150%"
                ],
                "correctExplanation": "The index must go from 0.5 back to 1: a gain of 1/0.5 − 1 = 100%; in general d/(1 − d) for a drawdown d.",
                "incorrectExplanation": "The gain is computed on the lower level: after a fall of 50% the index must double, d/(1 − d) = 100%."
            },
            "ro": {
                "title": "Revenirea după un drawdown",
                "text": "Un indice scade cu 50%. Ce cîștig este necesar pentru revenirea la vîrful anterior?",
                "options": [
                    "50%",
                    "75%",
                    "100%",
                    "150%"
                ],
                "correctExplanation": "Indicele trebuie să revină de la 0,5 la 1: un cîștig de 1/0,5 − 1 = 100%; în general d/(1 − d) pentru un drawdown d.",
                "incorrectExplanation": "Cîștigul se calculează pe nivelul mai mic: după o scădere de 50%, indicele trebuie să se dubleze, d/(1 − d) = 100%."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Precision of a Sharpe ratio",
                "text": "A fund has an annual Sharpe ratio of 0.5 measured over 4 years. What is its approximate standard error?",
                "options": [
                    "0.01",
                    "0.05",
                    "0.125",
                    "About 0.53"
                ],
                "correctExplanation": "SE ≈ √((1 + SR²/2)/Y) = √(1.125/4) ≈ 0.53: the Sharpe ratio is less than one standard error from zero.",
                "incorrectExplanation": "The standard error of an annual Sharpe ratio is about √((1 + SR²/2)/Y), with Y the number of years; four years give a very imprecise estimate."
            },
            "ro": {
                "title": "Precizia unui raport Sharpe",
                "text": "Un fond are raportul Sharpe anual 0,5, măsurat pe 4 ani. Care este aproximativ eroarea lui standard?",
                "options": [
                    "0,01",
                    "0,05",
                    "0,125",
                    "Circa 0,53"
                ],
                "correctExplanation": "SE ≈ √((1 + SR²/2)/Y) = √(1,125/4) ≈ 0,53: raportul Sharpe se află la mai puțin de o eroare standard de zero.",
                "incorrectExplanation": "Eroarea standard a unui raport Sharpe anual este circa √((1 + SR²/2)/Y), cu Y numărul de ani; patru ani dau o estimare foarte imprecisă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Jarque–Bera",
                "text": "n = 1000, skewness S = −0.5, excess kurtosis K = 6. Which part dominates JB = n/6 (S² + K²/4)?",
                "options": [
                    "The kurtosis part, 1500, against 41.7 for the skewness",
                    "The skewness part",
                    "Both are equal",
                    "Neither: JB is below 5.99"
                ],
                "correctExplanation": "n S²/6 = 41.7 and n K²/24 = 1500: heavy tails, not asymmetry, drive the rejection of normality.",
                "incorrectExplanation": "Compute both terms: 1000 × 0.25/6 = 41.7 and 1000 × 36/24 = 1500; JB is far above the 5% critical value 5.99."
            },
            "ro": {
                "title": "Jarque–Bera",
                "text": "n = 1000, asimetria S = −0,5, excesul de boltire K = 6. Care componentă domină JB = n/6 (S² + K²/4)?",
                "options": [
                    "Componenta boltirii, 1500, față de 41,7 pentru asimetrie",
                    "Componenta asimetriei",
                    "Sînt egale",
                    "Niciuna: JB este sub 5,99"
                ],
                "correctExplanation": "n S²/6 = 41,7 și n K²/24 = 1500: cozile groase, nu asimetria, duc la respingerea normalității.",
                "incorrectExplanation": "Calculați ambii termeni: 1000 × 0,25/6 = 41,7 și 1000 × 36/24 = 1500; JB este mult peste valoarea critică de 5,99 la 5%."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Student-t kurtosis",
                "text": "What is the excess kurtosis of a Student-t distribution with ν = 6 degrees of freedom?",
                "options": [
                    "0",
                    "3",
                    "6",
                    "It does not exist"
                ],
                "correctExplanation": "The excess kurtosis is 6/(ν − 4) for ν > 4: 6/2 = 3.",
                "incorrectExplanation": "For ν > 4 the excess kurtosis of the Student-t is 6/(ν − 4); the Normal distribution has excess kurtosis 0."
            },
            "ro": {
                "title": "Boltirea Student-t",
                "text": "Cît este excesul de boltire al unei distribuții Student-t cu ν = 6 grade de libertate?",
                "options": [
                    "0",
                    "3",
                    "6",
                    "Nu există"
                ],
                "correctExplanation": "Excesul de boltire este 6/(ν − 4) pentru ν > 4: 6/2 = 3.",
                "incorrectExplanation": "Pentru ν > 4, excesul de boltire al distribuției Student-t este 6/(ν − 4); distribuția Normală are excesul de boltire 0."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Uncorrelated returns",
                "text": "The ACF of daily returns is almost zero, the ACF of absolute returns is positive for many lags. What follows?",
                "options": [
                    "Returns are independent",
                    "Returns are predictable in sign",
                    "Returns are uncorrelated but not independent: volatility clustering",
                    "The data are wrong"
                ],
                "correctExplanation": "Zero linear correlation with dependent magnitudes is the stylised fact of volatility clustering (Cont, 2001).",
                "incorrectExplanation": "Absence of linear autocorrelation does not imply independence; the dependence is in the size of the returns, not in their sign."
            },
            "ro": {
                "title": "Randamente necorelate",
                "text": "ACF al randamentelor zilnice este aproape zero, ACF al randamentelor absolute este pozitiv la multe laguri. Ce rezultă?",
                "options": [
                    "Randamentele sînt independente",
                    "Semnul randamentelor este previzibil",
                    "Randamentele sînt necorelate, dar nu independente: volatility clustering",
                    "Datele sînt greșite"
                ],
                "correctExplanation": "Corelația liniară nulă împreună cu mărimi dependente reprezintă faptul stilizat numit volatility clustering (Cont, 2001).",
                "incorrectExplanation": "Absența autocorelației liniare nu implică independența; dependența se află în mărimea randamentelor, nu în semnul lor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Stable laws: moments",
                "text": "A stable law has α = 1.5. Which moments exist?",
                "options": [
                    "All moments",
                    "Mean and variance",
                    "Neither the mean nor the variance",
                    "The mean, but not the variance"
                ],
                "correctExplanation": "E|X|^p is finite only for p < α: the mean exists (1 < 1.5), the variance does not (2 > 1.5).",
                "incorrectExplanation": "For a stable law with α < 2 only the moments of order p < α are finite; with α = 1.5 the variance is infinite."
            },
            "ro": {
                "title": "Legi stabile: momente",
                "text": "O lege stabilă are α = 1,5. Ce momente există?",
                "options": [
                    "Toate momentele",
                    "Media și varianța",
                    "Nici media, nici varianța",
                    "Media, dar nu și varianța"
                ],
                "correctExplanation": "E|X|^p este finit doar pentru p < α: media există (1 < 1,5), varianța nu (2 > 1,5).",
                "incorrectExplanation": "Pentru o lege stabilă cu α < 2 sînt finite doar momentele de ordin p < α; cu α = 1,5, varianța este infinită."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Stable scaling",
                "text": "Daily returns are i.i.d. stable with α = 1.5. By how much does the scale grow from 1 to 20 days?",
                "options": [
                    "20^(1/1.5) ≈ 7.37",
                    "√20 ≈ 4.47",
                    "20",
                    "1.5"
                ],
                "correctExplanation": "Sums of n stable copies scale with n^(1/α); with α = 1.5 this is faster than the √n of the Normal case.",
                "incorrectExplanation": "The square-root rule holds only for α = 2 (the Normal distribution); a stable law scales with n^(1/α)."
            },
            "ro": {
                "title": "Scalarea stabilă",
                "text": "Randamentele zilnice sînt stabile i.i.d. cu α = 1,5. De cîte ori crește scala de la 1 la 20 de zile?",
                "options": [
                    "20^(1/1,5) ≈ 7,37",
                    "√20 ≈ 4,47",
                    "20",
                    "1,5"
                ],
                "correctExplanation": "Suma a n copii stabile se scalează cu n^(1/α); cu α = 1,5, creșterea este mai rapidă decît √n din cazul Normal.",
                "incorrectExplanation": "Regula rădăcinii pătrate este valabilă doar pentru α = 2 (distribuția Normală); o lege stabilă se scalează cu n^(1/α)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Monte Carlo error",
                "text": "A Monte Carlo study with N = 10 000 draws estimates a probability p = 1%. What is its standard error?",
                "options": [
                    "0.01%",
                    "about 0.10%",
                    "1%",
                    "10%"
                ],
                "correctExplanation": "SE = √(p(1 − p)/N) = √(0.0099/10 000) ≈ 0.1 percentage points; it falls with 1/√N.",
                "incorrectExplanation": "The standard error of an estimated probability is √(p(1 − p)/N); quadrupling N only halves it."
            },
            "ro": {
                "title": "Eroarea Monte Carlo",
                "text": "Un studiu Monte Carlo cu N = 10 000 de extrageri estimează o probabilitate p = 1%. Cît este eroarea standard?",
                "options": [
                    "0,01%",
                    "circa 0,10%",
                    "1%",
                    "10%"
                ],
                "correctExplanation": "SE = √(p(1 − p)/N) = √(0,0099/10 000) ≈ 0,1 puncte procentuale; scade cu 1/√N.",
                "incorrectExplanation": "Eroarea standard a unei probabilități estimate este √(p(1 − p)/N); dacă N crește de patru ori, eroarea doar se înjumătățește."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GBM: mean and median",
                "text": "A price follows a GBM with μ = 8% and σ = 25% a year. After one year, how do the mean and the median of the price compare?",
                "options": [
                    "They are equal",
                    "The median is above the mean",
                    "The mean is above the median",
                    "It depends on S_0"
                ],
                "correctExplanation": "Mean S_0 e^μ, median S_0 e^(μ − σ²/2): the mean is larger, pulled up by the right tail of the lognormal distribution.",
                "incorrectExplanation": "Under a GBM the log price is Normal with mean (μ − σ²/2)t, so the median S_0 e^((μ − σ²/2)t) is below the mean S_0 e^(μt)."
            },
            "ro": {
                "title": "GBM: media și mediana",
                "text": "Un preț urmează un GBM cu μ = 8% și σ = 25% pe an. După un an, cum se compară media și mediana prețului?",
                "options": [
                    "Sînt egale",
                    "Mediana este peste medie",
                    "Media este peste mediană",
                    "Depinde de S_0"
                ],
                "correctExplanation": "Media S_0 e^μ, mediana S_0 e^(μ − σ²/2): media este mai mare, trasă în sus de coada dreaptă a distribuției lognormale.",
                "incorrectExplanation": "Într-un GBM prețul logaritmic este Normal cu media (μ − σ²/2)t, deci mediana S_0 e^((μ − σ²/2)t) este sub media S_0 e^(μt)."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Tail index",
                "text": "A Hill plot gives a tail index α̂ = 3.5 for daily losses. Which moments of the losses exist?",
                "options": [
                    "Only the mean",
                    "None",
                    "All of them",
                    "Mean, variance and skewness, but not the kurtosis"
                ],
                "correctExplanation": "Moments of order p < α = 3.5 exist: the mean (p = 1), the variance (p = 2) and the skewness (p = 3); the kurtosis (p = 4) is infinite.",
                "incorrectExplanation": "With a power-law tail of index α the moments of order p < α are finite and those of order p ≥ α are infinite; α = 3.5 means a finite variance and an infinite kurtosis."
            },
            "ro": {
                "title": "Tail index-ul",
                "text": "Graficul Hill dă tail index-ul α̂ = 3,5 pentru pierderile zilnice. Ce momente ale pierderilor există?",
                "options": [
                    "Doar media",
                    "Niciunul",
                    "Toate",
                    "Media, varianța și asimetria, dar nu și boltirea"
                ],
                "correctExplanation": "Există momentele de ordin p < α = 3,5: media (p = 1), varianța (p = 2) și asimetria (p = 3); boltirea (p = 4) este infinită.",
                "incorrectExplanation": "Pentru o coadă de tip putere cu tail index-ul α, momentele de ordin p < α sînt finite, iar cele de ordin p ≥ α sînt infinite; α = 3,5 înseamnă varianță finită și boltire infinită."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Choosing a distribution",
                "text": "Normal: log-likelihood −6224, 2 parameters; Student-t: −5496, 3 parameters. Which has the smaller AIC = −2ℓ + 2k?",
                "options": [
                    "Student-t",
                    "Normal",
                    "They are equal",
                    "AIC cannot compare them"
                ],
                "correctExplanation": "AIC: Normal 12452, Student-t 10998; the gain in likelihood far exceeds the penalty for one extra parameter.",
                "incorrectExplanation": "Compute −2ℓ + 2k for both: the Student-t gains about 1456 in −2ℓ and pays only 2 for the extra parameter."
            },
            "ro": {
                "title": "Alegerea distribuției",
                "text": "Normală: log-verosimilitatea −6224, 2 parametri; Student-t: −5496, 3 parametri. Care are AIC = −2ℓ + 2k mai mic?",
                "options": [
                    "Student-t",
                    "Normală",
                    "Sînt egale",
                    "AIC nu le poate compara"
                ],
                "correctExplanation": "AIC: Normală 12452, Student-t 10998; cîștigul de verosimilitate depășește cu mult penalizarea pentru un parametru în plus.",
                "incorrectExplanation": "Calculați −2ℓ + 2k pentru ambele: Student-t cîștigă circa 1456 la −2ℓ și plătește doar 2 pentru parametrul suplimentar."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Robust variance ratio",
                "text": "Why is the robust statistic Z*(q) preferred to Z(q) for daily returns?",
                "options": [
                    "It is easier to compute",
                    "It allows for volatility clustering; Z(q) assumes i.i.d. returns and rejects too often",
                    "It has more power for crypto only",
                    "It uses prices instead of returns"
                ],
                "correctExplanation": "Z*(q) uses a heteroskedasticity-consistent variance (Lo and MacKinlay, 1988); with GARCH-type returns, Z(q) over-rejects the random walk.",
                "incorrectExplanation": "The difference between the two statistics is the variance under the null: only Z*(q) is valid when the variance changes over time."
            },
            "ro": {
                "title": "Raportul varianțelor robust",
                "text": "De ce se preferă statistica robustă Z*(q) statisticii Z(q) pentru randamentele zilnice?",
                "options": [
                    "Este mai ușor de calculat",
                    "Ține seama de volatility clustering; Z(q) presupune randamente i.i.d. și respinge prea des",
                    "Are putere mai mare doar pentru cripto",
                    "Folosește prețuri în loc de randamente"
                ],
                "correctExplanation": "Z*(q) folosește o varianță consistentă la heteroscedasticitate (Lo și MacKinlay, 1988); pentru randamente de tip GARCH, Z(q) respinge prea des mersul aleator.",
                "incorrectExplanation": "Diferența dintre cele două statistici este varianța sub ipoteza nulă: doar Z*(q) este validă cînd varianța se schimbă în timp."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Reading VR(q)",
                "text": "VR(5) = 1.28 with a significant robust statistic. What does it indicate?",
                "options": [
                    "Mean reversion",
                    "A unit root in returns",
                    "Positive autocorrelation (momentum)",
                    "Volatility clustering"
                ],
                "correctExplanation": "VR(q) = 1 + 2 Σ (1 − k/q) ρ_k > 1 means positive autocorrelations: weekly returns vary more than five daily ones.",
                "incorrectExplanation": "VR above 1 comes from positive autocorrelations, VR below 1 from negative ones; the test is about the mean, not the variance."
            },
            "ro": {
                "title": "Interpretarea VR(q)",
                "text": "VR(5) = 1,28, cu o statistică robustă semnificativă. Ce indică?",
                "options": [
                    "Revenire la medie",
                    "O rădăcină unitară în randamente",
                    "Autocorelație pozitivă (momentum)",
                    "Volatility clustering"
                ],
                "correctExplanation": "VR(q) = 1 + 2 Σ (1 − k/q) ρ_k > 1 înseamnă autocorelații pozitive: randamentele săptămînale variază mai mult decît cinci randamente zilnice.",
                "incorrectExplanation": "Un VR peste 1 provine din autocorelații pozitive, unul sub 1 din autocorelații negative; testul privește media, nu varianța."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "EWMA half-life",
                "text": "What is the half-life of the weights of an EWMA with λ = 0.94?",
                "options": [
                    "0.94 days",
                    "6 days",
                    "94 days",
                    "about 11.2 days"
                ],
                "correctExplanation": "ln 0.5 / ln 0.94 = 11.2: the weight of a shock halves in about eleven trading days.",
                "incorrectExplanation": "The weights decay as λ^k; the half-life solves λ^h = 0.5, h = ln 0.5 / ln λ."
            },
            "ro": {
                "title": "Timpul de înjumătățire EWMA",
                "text": "Cît este timpul de înjumătățire al ponderilor unui EWMA cu λ = 0,94?",
                "options": [
                    "0,94 zile",
                    "6 zile",
                    "94 de zile",
                    "circa 11,2 zile"
                ],
                "correctExplanation": "ln 0,5 / ln 0,94 = 11,2: ponderea unui șoc se înjumătățește în circa unsprezece zile de tranzacționare.",
                "incorrectExplanation": "Ponderile scad ca λ^k; timpul de înjumătățire rezolvă λ^h = 0,5, h = ln 0,5 / ln λ."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "GARCH long-run variance",
                "text": "GARCH(1,1) with ω = 0.03, α = 0.09, β = 0.89 for daily returns in %. What is the long-run daily variance?",
                "options": [
                    "1.5",
                    "0.03",
                    "0.98",
                    "It does not exist"
                ],
                "correctExplanation": "σ̄² = ω/(1 − α − β) = 0.03/0.02 = 1.5, i.e. an annual volatility of √(252 × 1.5) ≈ 19.4%.",
                "incorrectExplanation": "The unconditional variance of a stationary GARCH(1,1) is ω/(1 − α − β); it exists because α + β = 0.98 < 1."
            },
            "ro": {
                "title": "Varianța pe termen lung GARCH",
                "text": "GARCH(1,1) cu ω = 0,03, α = 0,09, β = 0,89 pentru randamente zilnice în %. Cît este varianța zilnică pe termen lung?",
                "options": [
                    "1,5",
                    "0,03",
                    "0,98",
                    "Nu există"
                ],
                "correctExplanation": "σ̄² = ω/(1 − α − β) = 0,03/0,02 = 1,5, adică o volatilitate anuală de √(252 × 1,5) ≈ 19,4%.",
                "incorrectExplanation": "Varianța necondiționată a unui GARCH(1,1) staționar este ω/(1 − α − β); există, deoarece α + β = 0,98 < 1."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The sign of VaR",
                "text": "The 1% quantile of daily returns is −2.4%. What is VaR 1%?",
                "options": [
                    "−2.4%",
                    "2.4%: a loss exceeded with probability 1%",
                    "99%",
                    "97.6%"
                ],
                "correctExplanation": "VaR_α = −q_α: the minus sign turns the negative quantile into a positive loss.",
                "incorrectExplanation": "In this course VaR is a positive loss and its level is the tail probability: VaR 1% = −q_0.01."
            },
            "ro": {
                "title": "Semnul VaR",
                "text": "Cuantila de 1% a randamentelor zilnice este −2,4%. Cît este VaR 1%?",
                "options": [
                    "−2,4%",
                    "2,4%: o pierdere depășită cu probabilitatea 1%",
                    "99%",
                    "97,6%"
                ],
                "correctExplanation": "VaR_α = −q_α: semnul minus transformă cuantila negativă într-o pierdere pozitivă.",
                "incorrectExplanation": "În acest curs, VaR este o pierdere pozitivă, iar nivelul lui este probabilitatea cozii: VaR 1% = −q_0,01."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The traffic light",
                "text": "A VaR 1% model has 7 exceptions in the last 250 days. Which Basel zone is it in?",
                "options": [
                    "Green",
                    "Red",
                    "Yellow",
                    "No zone: 7 is the expected number"
                ],
                "correctExplanation": "Green 0–4, yellow 5–9, red 10 or more exceptions in 250 days; 2.5 exceptions are expected.",
                "incorrectExplanation": "The expected number of exceptions of a correct VaR 1% model over 250 days is 2.5; 7 is in the yellow band 5–9."
            },
            "ro": {
                "title": "Semaforul Basel",
                "text": "Un model VaR 1% are 7 depășiri în ultimele 250 de zile. În ce zonă Basel se află?",
                "options": [
                    "Verde",
                    "Roșie",
                    "Galbenă",
                    "Nicio zonă: 7 este numărul așteptat"
                ],
                "correctExplanation": "Verde 0–4, galbenă 5–9, roșie 10 sau mai multe depășiri în 250 de zile; numărul așteptat este 2,5.",
                "incorrectExplanation": "Numărul așteptat de depășiri al unui model VaR 1% corect în 250 de zile este 2,5; 7 se află în banda galbenă 5–9."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Hurst exponents",
                "text": "H (R/S) is 0.53 for the returns and 0.84 for the absolute returns of the S&P 500. What does this mean?",
                "options": [
                    "Returns are strongly predictable",
                    "The market is fractal and inefficient",
                    "Both series are white noise",
                    "Long memory is in the volatility, not in the returns"
                ],
                "correctExplanation": "H near 0.5 for r_t and far above 0.5 for |r_t|: the persistence is in the size of the moves, as in volatility clustering.",
                "incorrectExplanation": "Memory in |r_t| describes the volatility; the returns themselves behave almost like a random walk."
            },
            "ro": {
                "title": "Exponenți Hurst",
                "text": "H (R/S) este 0,53 pentru randamentele S&P 500 și 0,84 pentru randamentele absolute. Ce înseamnă?",
                "options": [
                    "Randamentele sînt puternic previzibile",
                    "Piața este fractală și ineficientă",
                    "Ambele serii sînt zgomot alb",
                    "Memoria lungă se află în volatilitate, nu în randamente"
                ],
                "correctExplanation": "H aproape de 0,5 pentru r_t și mult peste 0,5 pentru |r_t|: persistența se află în mărimea variațiilor, ca în volatility clustering.",
                "incorrectExplanation": "Memoria lui |r_t| descrie volatilitatea; randamentele propriu-zise se comportă aproape ca un mers aleator."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "AUC and Gini",
                "text": "A scorecard has AUC = 0.80. What is its Gini coefficient?",
                "options": [
                    "0.60",
                    "0.40",
                    "0.80",
                    "1.60"
                ],
                "correctExplanation": "Gini = 2 AUC − 1 = 0.60.",
                "incorrectExplanation": "The Gini coefficient rescales the AUC from [0.5, 1] to [0, 1]: Gini = 2 AUC − 1."
            },
            "ro": {
                "title": "AUC și Gini",
                "text": "Un scorecard are AUC = 0,80. Cît este coeficientul Gini?",
                "options": [
                    "0,60",
                    "0,40",
                    "0,80",
                    "1,60"
                ],
                "correctExplanation": "Gini = 2 AUC − 1 = 0,60.",
                "incorrectExplanation": "Coeficientul Gini rescalează AUC de la [0,5; 1] la [0; 1]: Gini = 2 AUC − 1."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Validating a forecast",
                "text": "How should a machine-learning model for daily returns be validated?",
                "options": [
                    "Random K-fold cross-validation",
                    "Walk-forward validation with purging, against a simple benchmark",
                    "On the training sample only",
                    "By the number of parameters"
                ],
                "correctExplanation": "Training on the past and testing on the future, with a gap for overlapping targets, avoids look-ahead bias and leakage.",
                "incorrectExplanation": "Random folds put future observations into the training set: the error is underestimated and the model looks better than it is."
            },
            "ro": {
                "title": "Validarea unei prognoze",
                "text": "Cum se validează un model de machine learning pentru randamente zilnice?",
                "options": [
                    "Prin validare încrucișată K-fold aleatoare",
                    "Prin validare walk-forward cu purging, față de un reper simplu",
                    "Doar pe eșantionul de antrenare",
                    "După numărul de parametri"
                ],
                "correctExplanation": "Antrenarea pe trecut și testarea pe viitor, cu un interval liber pentru țintele suprapuse, evită look-ahead bias și leakage-ul.",
                "incorrectExplanation": "Partițiile aleatoare pun observații din viitor în setul de antrenare: eroarea este subestimată, iar modelul pare mai bun decît este."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "A stablecoin peg",
                "text": "A stablecoin trades at 0.9850 USD. How far is it from its peg, in basis points?",
                "options": [
                    "1.5 bp",
                    "15 bp",
                    "−150 bp",
                    "−1500 bp"
                ],
                "correctExplanation": "d = 10 000 × (P − 1) = 10 000 × (−0.015) = −150 basis points.",
                "incorrectExplanation": "One basis point is 0.0001; a price of 0.9850 is 0.015 below the peg, that is 150 basis points."
            },
            "ro": {
                "title": "Paritatea unui stablecoin",
                "text": "Un stablecoin se tranzacționează la 0,9850 USD. Cît se abate de la paritate, în puncte de bază?",
                "options": [
                    "1,5 bp",
                    "15 bp",
                    "−150 bp",
                    "−1500 bp"
                ],
                "correctExplanation": "d = 10 000 × (P − 1) = 10 000 × (−0,015) = −150 de puncte de bază.",
                "incorrectExplanation": "Un punct de bază înseamnă 0,0001; un preț de 0,9850 se află cu 0,015 sub paritate, adică 150 de puncte de bază."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ΔCoVaR",
                "text": "What does ΔCoVaR of a bank measure?",
                "options": [
                    "The VaR of the bank itself",
                    "The correlation of the bank with the market",
                    "The capital of the bank",
                    "How much the VaR of the system rises when the bank moves from its median state to its VaR"
                ],
                "correctExplanation": "ΔCoVaR = CoVaR at the bank's VaR minus CoVaR at its median (Adrian and Brunnermeier, 2016): the contribution of the bank's distress to system risk.",
                "incorrectExplanation": "CoVaR is the VaR of the system conditional on the state of one institution; ΔCoVaR compares a distress state with a normal state."
            },
            "ro": {
                "title": "ΔCoVaR",
                "text": "Ce măsoară ΔCoVaR al unei bănci?",
                "options": [
                    "VaR-ul băncii însăși",
                    "Corelația băncii cu piața",
                    "Capitalul băncii",
                    "Cu cît crește VaR-ul sistemului cînd banca trece de la starea mediană la VaR-ul ei"
                ],
                "correctExplanation": "ΔCoVaR = CoVaR la VaR-ul băncii minus CoVaR la mediana ei (Adrian și Brunnermeier, 2016): contribuția dificultăților băncii la riscul sistemului.",
                "incorrectExplanation": "CoVaR este VaR-ul sistemului condiționat de starea unei instituții; ΔCoVaR compară o stare de criză cu o stare normală."
            }
        }
    ]
};
