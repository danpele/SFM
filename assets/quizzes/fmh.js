// ============================================================
// Chapter 11 quiz bank: fractal markets hypothesis and long memory (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['fmh'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "The fractal market hypothesis",
                "text": "According to the fractal market hypothesis of Peters (1994), what keeps a market stable?",
                "options": [
                    "Investors with many different investment horizons who supply liquidity to each other",
                    "A single representative investor with rational expectations",
                    "Returns that follow a Normal distribution",
                    "A central bank that fixes the price level"
                ],
                "correctExplanation": "Stability comes from the diversity of horizons: when short-term traders sell, long-term investors buy, so liquidity is always there.",
                "incorrectExplanation": "The FMH is about heterogeneous horizons and liquidity; a representative investor is the EMH view, and the distribution of returns or a central bank are not part of the hypothesis."
            },
            "ro": {
                "title": "Ipoteza pieței fractale",
                "text": "Potrivit ipotezei pieței fractale a lui Peters (1994), ce menține stabilitatea unei piețe?",
                "options": [
                    "Investitorii cu orizonturi investiționale foarte diferite, care își oferă lichiditate unii altora",
                    "Un singur investitor reprezentativ, cu așteptări raționale",
                    "Randamente care urmează distribuția Normală",
                    "O bancă centrală care fixează nivelul prețurilor"
                ],
                "correctExplanation": "Stabilitatea vine din diversitatea orizonturilor: cînd traderii pe termen scurt vînd, investitorii pe termen lung cumpără, deci lichiditatea există mereu.",
                "incorrectExplanation": "FMH se referă la orizonturi eterogene și la lichiditate; investitorul reprezentativ ține de EMH, iar distribuția randamentelor sau banca centrală nu fac parte din ipoteză."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Crashes in the FMH",
                "text": "How does the fractal market hypothesis explain a crash such as 19 October 1987?",
                "options": [
                    "Prices follow a random walk with Normal increments",
                    "Long-term investors start to trade on short-term information: all horizons collapse into one and buyers disappear",
                    "Investors become more diversified across horizons",
                    "The Hurst exponent of returns becomes exactly 0.5"
                ],
                "correctExplanation": "When fundamental information becomes doubtful, long-horizon investors behave like short-horizon traders; with one horizon left, nobody supplies liquidity.",
                "incorrectExplanation": "A crash in the FMH is a loss of horizon diversity, not more diversity; a random walk with Normal increments makes such a fall practically impossible, and H = 0.5 says nothing about crashes."
            },
            "ro": {
                "title": "Crahurile în FMH",
                "text": "Cum explică ipoteza pieței fractale un crah precum cel din 19 octombrie 1987?",
                "options": [
                    "Prețurile urmează un mers aleator cu creșteri Normale",
                    "Investitorii pe termen lung încep să tranzacționeze pe baza informației de termen scurt: toate orizonturile se reduc la unul singur, iar cumpărătorii dispar",
                    "Investitorii devin mai diversificați ca orizont",
                    "Exponentul Hurst al randamentelor devine exact 0,5"
                ],
                "correctExplanation": "Cînd informația fundamentală devine îndoielnică, investitorii pe termen lung se comportă ca traderii pe termen scurt; cu un singur orizont rămas, nu mai oferă nimeni lichiditate.",
                "incorrectExplanation": "În FMH, un crah înseamnă pierderea diversității orizonturilor, nu creșterea ei; un mers aleator cu creșteri Normale face o astfel de scădere practic imposibilă, iar H = 0,5 nu spune nimic despre crahuri."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Self-similarity",
                "text": "A process X is self-similar with Hurst exponent H. Which property holds for every c > 0?",
                "options": [
                    "X(ct) has the same variance as X(t)",
                    "X(t + c) = X(t) + c^H",
                    "X(ct) has the same distribution as c^H X(t)",
                    "The autocorrelations of X are zero"
                ],
                "correctExplanation": "Self-similarity means that stretching time by c is equivalent in distribution to stretching the values by c^H (FHH, Def. 14.1).",
                "incorrectExplanation": "The variance changes with the time scale (by c^2H), the definition is about distributions and not about an additive shift, and self-similar processes such as fBm can have correlated increments."
            },
            "ro": {
                "title": "Autosimilaritatea",
                "text": "Un proces X este autosimilar, cu exponentul Hurst H. Ce proprietate are loc pentru orice c > 0?",
                "options": [
                    "X(ct) are aceeași varianță ca X(t)",
                    "X(t + c) = X(t) + c^H",
                    "X(ct) are aceeași distribuție ca c^H X(t)",
                    "Autocorelațiile lui X sînt nule"
                ],
                "correctExplanation": "Autosimilaritatea înseamnă că dilatarea timpului cu c este echivalentă, în distribuție, cu dilatarea valorilor cu c^H (FHH, Def. 14.1).",
                "incorrectExplanation": "Varianța se schimbă cu scala de timp (cu factorul c^2H), definiția privește distribuțiile, nu o translație aditivă, iar procesele autosimilare precum fBm pot avea creșteri corelate."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The square-root-of-time rule",
                "text": "The rule \"the 10-day volatility is sqrt(10) times the daily volatility\" corresponds to which Hurst exponent?",
                "options": [
                    "H = 0",
                    "H = 1",
                    "H = 0.7",
                    "H = 0.5"
                ],
                "correctExplanation": "If the h-day standard deviation is h^H times the daily one, the square-root rule is the case H = 0.5 (independent increments).",
                "incorrectExplanation": "With H = 0.7 the 10-day volatility would be 10^0.7 = 5.01 times the daily one, with H = 1 ten times, and H = 0 gives no growth; only H = 0.5 gives sqrt(10)."
            },
            "ro": {
                "title": "Regula rădăcinii pătrate a timpului",
                "text": "Regula „volatilitatea pe 10 zile este de sqrt(10) ori volatilitatea zilnică” corespunde cărui exponent Hurst?",
                "options": [
                    "H = 0",
                    "H = 1",
                    "H = 0,7",
                    "H = 0,5"
                ],
                "correctExplanation": "Dacă abaterea standard pe h zile este de h^H ori cea zilnică, regula rădăcinii pătrate este cazul H = 0,5 (creșteri independente).",
                "incorrectExplanation": "Cu H = 0,7, volatilitatea pe 10 zile ar fi de 10^0,7 = 5,01 ori cea zilnică, cu H = 1 de zece ori, iar H = 0 nu dă nicio creștere; doar H = 0,5 dă sqrt(10)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The range R",
                "text": "In the R/S statistic of a block x_1, ..., x_n, what is R?",
                "options": [
                    "The maximum minus the minimum of the cumulative deviations Y_k = sum of (x_j - mean) up to k",
                    "The largest minus the smallest return in the block",
                    "The sum of the absolute returns",
                    "The number of sign changes in the block"
                ],
                "correctExplanation": "R is the range of the cumulative deviations from the block mean; Hurst used it to size a reservoir.",
                "incorrectExplanation": "The range of the returns themselves, the sum of absolute values or the number of sign changes are different statistics; R/S uses the cumulative deviations."
            },
            "ro": {
                "title": "Amplitudinea R",
                "text": "În statistica R/S a unui bloc x_1, ..., x_n, ce este R?",
                "options": [
                    "Maximul minus minimul abaterilor cumulate Y_k = suma (x_j - media) pînă la k",
                    "Cel mai mare randament minus cel mai mic randament din bloc",
                    "Suma randamentelor absolute",
                    "Numărul schimbărilor de semn din bloc"
                ],
                "correctExplanation": "R este amplitudinea abaterilor cumulate de la media blocului; Hurst a folosit-o pentru a dimensiona un rezervor.",
                "incorrectExplanation": "Amplitudinea randamentelor înseși, suma valorilor absolute sau numărul schimbărilor de semn sînt alte statistici; R/S folosește abaterile cumulate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "R/S of eight returns",
                "text": "Eight returns give a range R = 0.875 of the cumulative deviations and a standard deviation S = 0.489. What is R/S?",
                "options": [
                    "0.428",
                    "1.788",
                    "0.559",
                    "0.386"
                ],
                "correctExplanation": "R/S = 0.875/0.489 = 1.788; one block gives one point of the R/S plot.",
                "incorrectExplanation": "The statistic is the ratio of the range to the standard deviation, not their product, their difference or the inverse ratio."
            },
            "ro": {
                "title": "R/S pentru opt randamente",
                "text": "Opt randamente dau o amplitudine a abaterilor cumulate R = 0,875 și o abatere standard S = 0,489. Cît este R/S?",
                "options": [
                    "0,428",
                    "1,788",
                    "0,559",
                    "0,386"
                ],
                "correctExplanation": "R/S = 0,875/0,489 = 1,788; un bloc dă un singur punct al graficului R/S.",
                "incorrectExplanation": "Statistica este raportul dintre amplitudine și abaterea standard, nu produsul, diferența sau raportul invers."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Interpreting H = 0.3",
                "text": "An estimate H = 0.3 (well outside the Monte Carlo band) for a return series means:",
                "options": [
                    "strong trends: rises are followed by rises",
                    "no dependence at all",
                    "anti-persistence: rises tend to be followed by falls",
                    "a unit root in returns"
                ],
                "correctExplanation": "H < 0.5 means negatively correlated increments: reversals are more frequent than under a random walk.",
                "incorrectExplanation": "Trends correspond to H > 0.5, no dependence to H = 0.5, and a unit root concerns prices (d = 1), not a stationary return series."
            },
            "ro": {
                "title": "Interpretarea lui H = 0,3",
                "text": "O estimare H = 0,3 (mult în afara benzii Monte Carlo) pentru o serie de randamente înseamnă:",
                "options": [
                    "tendințe puternice: creșterile urmează după creșteri",
                    "nicio dependență",
                    "antipersistență: creșterile tind să fie urmate de scăderi",
                    "o rădăcină unitară în randamente"
                ],
                "correctExplanation": "H < 0,5 înseamnă creșteri corelate negativ: inversările sînt mai frecvente decît la un mers aleator.",
                "incorrectExplanation": "Tendințele corespund lui H > 0,5, lipsa dependenței lui H = 0,5, iar rădăcina unitară privește prețurile (d = 1), nu o serie staționară de randamente."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "H is not a probability",
                "text": "A student says: \"H = 0.7 means a 70% chance that tomorrow has the same sign as today\". What is correct?",
                "options": [
                    "The statement is correct",
                    "H = 0.7 means a 30% chance of the same sign",
                    "H = 0.7 means the series is not stationary",
                    "H is a scaling exponent: increments are positively correlated, but H is not a probability"
                ],
                "correctExplanation": "H describes how the range and the variance scale with the horizon; with H = 0.7 the increments are positively correlated at all lags, which is not a sign probability.",
                "incorrectExplanation": "There is no direct mapping from H to the probability of repeating a sign, and fGn with H = 0.7 is stationary."
            },
            "ro": {
                "title": "H nu este o probabilitate",
                "text": "Un student spune: „H = 0,7 înseamnă 70% șanse ca mîine să aibă același semn ca azi”. Ce este corect?",
                "options": [
                    "Afirmația este corectă",
                    "H = 0,7 înseamnă 30% șanse pentru același semn",
                    "H = 0,7 înseamnă că seria nu este staționară",
                    "H este un exponent de scalare: creșterile sînt corelate pozitiv, dar H nu este o probabilitate"
                ],
                "correctExplanation": "H descrie cum se scalează amplitudinea și varianța cu orizontul; cu H = 0,7, creșterile sînt corelate pozitiv la toate decalajele, ceea ce nu este o probabilitate de semn.",
                "incorrectExplanation": "Nu există o legătură directă între H și probabilitatea ca semnul să se repete, iar fGn cu H = 0,7 este staționar."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "From d to H",
                "text": "The GPH regression gives d = 0.35 for absolute returns. What is the implied Hurst exponent?",
                "options": [
                    "H = 0.85",
                    "H = -0.15",
                    "H = 0.35",
                    "H = 0.70"
                ],
                "correctExplanation": "H = d + 0.5 = 0.85: strong persistence of volatility.",
                "incorrectExplanation": "The conversion is d = H - 0.5, so H = d + 0.5; subtracting 0.5, keeping d or doubling it are wrong."
            },
            "ro": {
                "title": "De la d la H",
                "text": "Regresia GPH dă d = 0,35 pentru randamentele absolute. Cît este exponentul Hurst corespunzător?",
                "options": [
                    "H = 0,85",
                    "H = -0,15",
                    "H = 0,35",
                    "H = 0,70"
                ],
                "correctExplanation": "H = d + 0,5 = 0,85: persistență puternică a volatilității.",
                "incorrectExplanation": "Conversia este d = H - 0,5, deci H = d + 0,5; scăderea lui 0,5, păstrarea lui d sau dublarea lui sînt greșite."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Lag-1 autocorrelation of fGn",
                "text": "For fractional Gaussian noise rho(1) = 2^(2H-1) - 1. What is rho(1) for H = 0.8?",
                "options": [
                    "0.800",
                    "0.516",
                    "0.300",
                    "0.000"
                ],
                "correctExplanation": "rho(1) = 2^0.6 - 1 = 0.516; the autocorrelations then decay hyperbolically.",
                "incorrectExplanation": "rho(1) is not H itself, not H - 0.5, and it is zero only for H = 0.5."
            },
            "ro": {
                "title": "Autocorelația de ordinul 1 a fGn",
                "text": "Pentru zgomotul gaussian fracționar, rho(1) = 2^(2H-1) - 1. Cît este rho(1) pentru H = 0,8?",
                "options": [
                    "0,800",
                    "0,516",
                    "0,300",
                    "0,000"
                ],
                "correctExplanation": "rho(1) = 2^0,6 - 1 = 0,516; apoi autocorelațiile scad hiperbolic.",
                "incorrectExplanation": "rho(1) nu este H însuși, nici H - 0,5, și este zero doar pentru H = 0,5."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Long memory",
                "text": "Which property defines long memory of a stationary series (FHH, 14.1)?",
                "options": [
                    "The autocorrelations are all negative",
                    "The autocorrelations are zero after lag q",
                    "The autocorrelations decay hyperbolically, like k^(2d-1), and their sum is infinite",
                    "The autocorrelations decay like phi^k with |phi| < 1"
                ],
                "correctExplanation": "Long memory: rho(k) ~ C k^(2d-1) with 0 < d < 0.5, so the sum of |rho(k)| diverges.",
                "incorrectExplanation": "Zero autocorrelations after lag q describe an MA(q) model and geometric decay an AR model, both short memory; negative autocorrelations indicate anti-persistence."
            },
            "ro": {
                "title": "Memoria lungă",
                "text": "Ce proprietate definește memoria lungă a unei serii staționare (FHH, 14.1)?",
                "options": [
                    "Autocorelațiile sînt toate negative",
                    "Autocorelațiile sînt nule după decalajul q",
                    "Autocorelațiile scad hiperbolic, ca k^(2d-1), iar suma lor este infinită",
                    "Autocorelațiile scad ca phi^k, cu |phi| < 1"
                ],
                "correctExplanation": "Memoria lungă: rho(k) ~ C k^(2d-1), cu 0 < d < 0,5, deci suma valorilor |rho(k)| diverge.",
                "incorrectExplanation": "Autocorelațiile nule după decalajul q descriu un model MA(q), iar scăderea geometrică un model AR, ambele cu memorie scurtă; autocorelațiile negative indică antipersistență."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ARFIMA(0,d,0)",
                "text": "For ARFIMA(0,d,0), rho(1) = d/(1 - d). What is rho(1) for d = 0.3?",
                "options": [
                    "0.300",
                    "0.700",
                    "0.231",
                    "0.429"
                ],
                "correctExplanation": "rho(1) = 0.3/0.7 = 0.429; then rho(k) = rho(k-1)(k-1+d)/(k-d) decays slowly.",
                "incorrectExplanation": "rho(1) is neither d, nor 1 - d, nor d/(1 + d); the formula is d/(1 - d)."
            },
            "ro": {
                "title": "ARFIMA(0,d,0)",
                "text": "Pentru ARFIMA(0,d,0), rho(1) = d/(1 - d). Cît este rho(1) pentru d = 0,3?",
                "options": [
                    "0,300",
                    "0,700",
                    "0,231",
                    "0,429"
                ],
                "correctExplanation": "rho(1) = 0,3/0,7 = 0,429; apoi rho(k) = rho(k-1)(k-1+d)/(k-d) scade lent.",
                "incorrectExplanation": "rho(1) nu este nici d, nici 1 - d, nici d/(1 + d); formula este d/(1 - d)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The memory parameter d",
                "text": "For which values of d is ARFIMA(0,d,0) stationary with long memory?",
                "options": [
                    "0 < d < 0.5",
                    "-0.5 < d < 0",
                    "0.5 <= d < 1",
                    "d = 1"
                ],
                "correctExplanation": "For 0 < d < 0.5 the series is stationary and its autocorrelations are positive and not summable (FHH, Table 14.1).",
                "incorrectExplanation": "Negative d gives anti-persistence, 0.5 <= d < 1 gives an infinite variance (non-stationary, mean-reverting), and d = 1 is a unit root."
            },
            "ro": {
                "title": "Parametrul de memorie d",
                "text": "Pentru ce valori ale lui d este ARFIMA(0,d,0) staționar și cu memorie lungă?",
                "options": [
                    "0 < d < 0,5",
                    "-0,5 < d < 0",
                    "0,5 <= d < 1",
                    "d = 1"
                ],
                "correctExplanation": "Pentru 0 < d < 0,5, seria este staționară, iar autocorelațiile ei sînt pozitive și nesumabile (FHH, tabelul 14.1).",
                "incorrectExplanation": "Un d negativ dă antipersistență, 0,5 <= d < 1 dă varianță infinită (nestaționar, dar cu revenire la medie), iar d = 1 este o rădăcină unitară."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Small-sample bias of R/S",
                "text": "For i.i.d. Normal series of 1000 observations, the R/S slope averages 0.562. What follows?",
                "options": [
                    "Every stock market has long memory",
                    "An R/S estimate must be compared with the Monte Carlo band [0.50, 0.63], not with 0.5",
                    "R/S cannot be used on financial data",
                    "The returns must be squared before using R/S"
                ],
                "correctExplanation": "R/S is biased upwards in small samples (Anis and Lloyd 1976); an estimate is evidence of memory only if it lies outside the band of the null hypothesis for the same N.",
                "incorrectExplanation": "The bias says nothing about real markets, does not make R/S useless, and squaring changes the question (volatility instead of returns)."
            },
            "ro": {
                "title": "Deplasarea R/S în eșantioane mici",
                "text": "Pentru serii i.i.d. Normale de 1000 de observații, panta R/S este în medie 0,562. Ce rezultă?",
                "options": [
                    "Orice piață de acțiuni are memorie lungă",
                    "O estimare R/S trebuie comparată cu banda Monte Carlo [0,50; 0,63], nu cu 0,5",
                    "R/S nu poate fi folosit pe date financiare",
                    "Randamentele trebuie ridicate la pătrat înainte de R/S"
                ],
                "correctExplanation": "R/S este deplasat în sus în eșantioane mici (Anis și Lloyd 1976); o estimare este o dovadă de memorie doar dacă iese din banda ipotezei nule pentru același N.",
                "incorrectExplanation": "Deplasarea nu spune nimic despre piețele reale, nu face R/S inutil, iar ridicarea la pătrat schimbă întrebarea (volatilitate în loc de randamente)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Lo's modified R/S",
                "text": "Why did Lo (1991) modify the classical R/S statistic?",
                "options": [
                    "To make it robust to heavy tails",
                    "To estimate the GARCH persistence",
                    "Because short-range dependence (e.g. an AR(1)) also inflates classical R/S and can be mistaken for long memory",
                    "To use intraday data"
                ],
                "correctExplanation": "Lo replaces the standard deviation by a Newey-West long-run standard deviation, so that short memory alone no longer leads to a rejection.",
                "incorrectExplanation": "R/S is already robust to heavy tails (Mandelbrot and Wallis 1969); the modification is not about GARCH or intraday data."
            },
            "ro": {
                "title": "R/S modificat al lui Lo",
                "text": "De ce a modificat Lo (1991) statistica R/S clasică?",
                "options": [
                    "Pentru a o face robustă la cozi groase",
                    "Pentru a estima persistența GARCH",
                    "Pentru că dependența pe termen scurt (de exemplu un AR(1)) mărește și ea R/S clasic și poate fi confundată cu memoria lungă",
                    "Pentru a folosi date intraday"
                ],
                "correctExplanation": "Lo înlocuiește abaterea standard cu o abatere standard de lungă durată Newey-West, astfel încît memoria scurtă singură nu mai duce la respingere.",
                "incorrectExplanation": "R/S este deja robust la cozi groase (Mandelbrot și Wallis 1969); modificarea nu privește GARCH sau datele intraday."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Lo's test on the S&P 500",
                "text": "Lo's modified statistic for S&P 500 daily returns since 2000 is V = 1.56. At 5%, the short-memory null is rejected outside [0.809, 1.862]. Conclusion?",
                "options": [
                    "Long memory is proven",
                    "Short memory is rejected",
                    "The test cannot be computed for index returns",
                    "Short memory is not rejected"
                ],
                "correctExplanation": "V = 1.56 lies inside [0.809, 1.862], so there is no evidence of long memory in S&P 500 returns, as Lo (1991) found for US indices.",
                "incorrectExplanation": "A statistic inside the acceptance interval never proves long memory or rejects the null; the test applies to any stationary series."
            },
            "ro": {
                "title": "Testul lui Lo pentru S&P 500",
                "text": "Statistica modificată a lui Lo pentru randamentele zilnice S&P 500 din 2000 este V = 1,56. La 5%, ipoteza memoriei scurte se respinge în afara intervalului [0,809; 1,862]. Concluzia?",
                "options": [
                    "Memoria lungă este dovedită",
                    "Memoria scurtă este respinsă",
                    "Testul nu poate fi calculat pentru randamentele unui indice",
                    "Memoria scurtă nu este respinsă"
                ],
                "correctExplanation": "V = 1,56 se află în [0,809; 1,862], deci nu există dovezi de memorie lungă în randamentele S&P 500, așa cum a constatat Lo (1991) pentru indicii din SUA.",
                "incorrectExplanation": "O statistică din intervalul de acceptare nu dovedește memoria lungă și nu respinge ipoteza nulă; testul se aplică oricărei serii staționare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "DFA on prices",
                "text": "A student applies DFA to log prices instead of returns and finds alpha = 1.5. What does it mean?",
                "options": [
                    "For fBm (prices) alpha = H + 1, so H = 0.5: the returns are a random walk increment",
                    "Strong long memory in returns, H = 1.5",
                    "The DFA is wrong for any alpha above 1",
                    "The returns are anti-persistent"
                ],
                "correctExplanation": "DFA integrates the series once more (the profile); for a non-stationary fBm the exponent is H + 1, so alpha = 1.5 corresponds to H = 0.5.",
                "incorrectExplanation": "H must lie in (0, 1), so 1.5 cannot be H; alpha above 1 is expected for non-stationary inputs, and nothing points to anti-persistence."
            },
            "ro": {
                "title": "DFA pe prețuri",
                "text": "Un student aplică DFA prețurilor logaritmice în loc de randamente și obține alpha = 1,5. Ce înseamnă?",
                "options": [
                    "Pentru fBm (prețuri), alpha = H + 1, deci H = 0,5: randamentele sînt creșterile unui mers aleator",
                    "Memorie lungă puternică în randamente, H = 1,5",
                    "DFA este greșit pentru orice alpha peste 1",
                    "Randamentele sînt antipersistente"
                ],
                "correctExplanation": "DFA mai integrează o dată seria (profilul); pentru un fBm nestaționar, exponentul este H + 1, deci alpha = 1,5 corespunde lui H = 0,5.",
                "incorrectExplanation": "H trebuie să fie în (0, 1), deci 1,5 nu poate fi H; un alpha peste 1 este de așteptat pentru date nestaționare, iar nimic nu indică antipersistență."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "GPH bandwidth",
                "text": "Why is the GPH regression run only on the m = T^0.5 lowest Fourier frequencies?",
                "options": [
                    "To make the computation faster",
                    "Because the spectral density behaves like C lambda^(-2d) only near zero; higher frequencies carry short-run dynamics that bias d",
                    "Because the periodogram is zero at high frequencies",
                    "Because GPH needs exactly 100 points"
                ],
                "correctExplanation": "Long memory is a low-frequency property; using all frequencies mixes in short-run dynamics, and consistency requires m/T -> 0.",
                "incorrectExplanation": "Speed is not the reason, the periodogram is not zero at high frequencies, and the bandwidth grows with T."
            },
            "ro": {
                "title": "Lățimea de bandă GPH",
                "text": "De ce se estimează regresia GPH doar pe cele m = T^0,5 cele mai joase frecvențe Fourier?",
                "options": [
                    "Pentru ca un calcul să fie mai rapid",
                    "Pentru că densitatea spectrală se comportă ca C lambda^(-2d) doar lîngă zero; frecvențele mai înalte poartă dinamica de termen scurt, care deplasează d",
                    "Pentru că periodograma este zero la frecvențe înalte",
                    "Pentru că GPH are nevoie de exact 100 de puncte"
                ],
                "correctExplanation": "Memoria lungă este o proprietate a frecvențelor joase; folosirea tuturor frecvențelor amestecă dinamica de termen scurt, iar consistența cere m/T -> 0.",
                "incorrectExplanation": "Viteza nu este motivul, periodograma nu este zero la frecvențe înalte, iar lățimea de bandă crește odată cu T."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Spurious long memory",
                "text": "i.i.d. noise with a single mean shift of 0.5 standard deviations at the middle gives an average GPH estimate H = 0.79. Why?",
                "options": [
                    "The noise has heavy tails",
                    "GPH is biased downwards",
                    "A mean shift creates slowly decaying sample autocorrelations that look like long memory",
                    "The series has a unit root"
                ],
                "correctExplanation": "Observations in the same regime share the deviation of the regime mean from the overall mean; this mimics long memory (Diebold and Inoue 2001).",
                "incorrectExplanation": "The noise is Normal, GPH is not biased downwards here, and a single shift in the mean is not a unit root."
            },
            "ro": {
                "title": "Memorie lungă aparentă",
                "text": "Un zgomot i.i.d. cu o singură schimbare a mediei de 0,5 abateri standard la mijloc dă în medie o estimare GPH H = 0,79. De ce?",
                "options": [
                    "Zgomotul are cozi groase",
                    "GPH este deplasat în jos",
                    "O schimbare a mediei creează autocorelații de selecție care scad lent și seamănă cu memoria lungă",
                    "Seria are o rădăcină unitară"
                ],
                "correctExplanation": "Observațiile din același regim au în comun abaterea mediei regimului de la media totală; aceasta imită memoria lungă (Diebold și Inoue 2001).",
                "incorrectExplanation": "Zgomotul este Normal, GPH nu este deplasat în jos aici, iar o singură schimbare a mediei nu este o rădăcină unitară."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Nile",
                "text": "The R/S exponent of the Nile flow 1871-1970 falls from 0.78 to 0.63 after removing the two regime means before and after 1898. What does this show?",
                "options": [
                    "The Nile has no variance",
                    "R/S cannot be applied to hydrological data",
                    "The break increased the true memory of the river",
                    "Part of the apparent long memory came from the structural break (Cobb 1978)"
                ],
                "correctExplanation": "Removing the change in mean removes much of the slow component; the rest is close to the wide Monte Carlo band for N = 100.",
                "incorrectExplanation": "R/S was invented for river flows; the drop does not mean zero variance, and a break does not create true memory, it creates apparent memory."
            },
            "ro": {
                "title": "Nilul",
                "text": "Exponentul R/S al debitului Nilului 1871-1970 scade de la 0,78 la 0,63 după eliminarea celor două medii de regim, dinainte și de după 1898. Ce arată acest lucru?",
                "options": [
                    "Nilul nu are varianță",
                    "R/S nu poate fi aplicat datelor hidrologice",
                    "Ruptura a crescut memoria reală a rîului",
                    "O parte din memoria lungă aparentă provenea din ruptura structurală (Cobb 1978)"
                ],
                "correctExplanation": "Eliminarea schimbării mediei înlătură o mare parte din componenta lentă; restul este aproape de banda Monte Carlo largă pentru N = 100.",
                "incorrectExplanation": "R/S a fost creat pentru debitele rîurilor; scăderea nu înseamnă varianță nulă, iar o ruptură nu creează memorie reală, ci memorie aparentă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The shuffle test",
                "text": "For S&P 500 absolute returns the DFA exponent is 0.95; after a random shuffle of the same values it is 0.46. Conclusion?",
                "options": [
                    "The memory lies in the order of the days, not in the distribution of |r|",
                    "The memory comes from heavy tails",
                    "DFA is not reliable",
                    "The returns are anti-persistent"
                ],
                "correctExplanation": "A permutation keeps the distribution (including heavy tails) and destroys the order; the exponent falls to about 0.5, so the long memory was in the time ordering.",
                "incorrectExplanation": "Heavy tails survive the shuffle, so they cannot explain the drop; the test is a check of DFA, not a failure, and it says nothing about the sign of returns."
            },
            "ro": {
                "title": "Testul permutării",
                "text": "Pentru randamentele absolute S&P 500, exponentul DFA este 0,95; după o permutare aleatoare a acelorași valori este 0,46. Concluzia?",
                "options": [
                    "Memoria se află în ordinea zilelor, nu în distribuția lui |r|",
                    "Memoria provine din cozile groase",
                    "DFA nu este de încredere",
                    "Randamentele sînt antipersistente"
                ],
                "correctExplanation": "O permutare păstrează distribuția (inclusiv cozile groase) și distruge ordinea; exponentul scade la circa 0,5, deci memoria lungă era în ordinea temporală.",
                "incorrectExplanation": "Cozile groase rămîn după permutare, deci nu pot explica scăderea; testul verifică DFA, nu îl infirmă, și nu spune nimic despre semnul randamentelor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Volatility memory and the EMH",
                "text": "Absolute returns show long memory while returns do not. Does this contradict the weak form of the EMH?",
                "options": [
                    "Yes: any long memory contradicts the EMH",
                    "No: the weak form concerns the direction of returns; predictable volatility is compatible with it",
                    "Yes, because |r| is a return",
                    "It depends only on the sample size"
                ],
                "correctExplanation": "The weak form says past returns do not forecast future returns; the size of returns (risk) can be predictable without allowing profitable trades on direction.",
                "incorrectExplanation": "Volatility memory is not a forecast of direction, |r| loses the sign, and the conclusion holds in large samples as well."
            },
            "ro": {
                "title": "Memoria volatilității și EMH",
                "text": "Randamentele absolute au memorie lungă, iar randamentele nu. Contrazice acest lucru forma slabă a EMH?",
                "options": [
                    "Da: orice memorie lungă contrazice EMH",
                    "Nu: forma slabă privește direcția randamentelor; volatilitatea previzibilă este compatibilă cu ea",
                    "Da, pentru că |r| este un randament",
                    "Depinde doar de mărimea eșantionului"
                ],
                "correctExplanation": "Forma slabă spune că randamentele trecute nu prognozează randamentele viitoare; mărimea randamentelor (riscul) poate fi previzibilă fără a permite tranzacții profitabile pe direcție.",
                "incorrectExplanation": "Memoria volatilității nu este o prognoză a direcției, |r| pierde semnul, iar concluzia rămîne valabilă și în eșantioane mari."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH and long memory",
                "text": "Absolute returns simulated from a GARCH(1,1) with alpha + beta = 0.99 give an average DFA exponent of 0.89. What does this show?",
                "options": [
                    "GARCH has long memory by construction",
                    "DFA always gives 0.9",
                    "A persistent short-memory model can look like long memory in finite samples",
                    "The simulated returns are anti-persistent"
                ],
                "correctExplanation": "In GARCH the ACF of r^2 decays like (alpha + beta)^k; with 0.99 the decay is too slow to be told apart from a hyperbolic one in a few thousand days.",
                "incorrectExplanation": "GARCH has exponentially decaying autocorrelations (short memory), DFA gives about 0.5 for i.i.d. data, and the simulated returns themselves stay at about 0.5."
            },
            "ro": {
                "title": "GARCH și memoria lungă",
                "text": "Randamentele absolute simulate dintr-un GARCH(1,1) cu alpha + beta = 0,99 dau un exponent DFA mediu de 0,89. Ce arată acest lucru?",
                "options": [
                    "GARCH are memorie lungă prin construcție",
                    "DFA dă mereu 0,9",
                    "Un model persistent cu memorie scurtă poate arăta ca memoria lungă în eșantioane finite",
                    "Randamentele simulate sînt antipersistente"
                ],
                "correctExplanation": "În GARCH, ACF a lui r^2 scade ca (alpha + beta)^k; cu 0,99, scăderea este prea lentă pentru a fi deosebită de una hiperbolică în cîteva mii de zile.",
                "incorrectExplanation": "GARCH are autocorelații care scad exponențial (memorie scurtă), DFA dă circa 0,5 pentru date i.i.d., iar randamentele simulate rămîn la circa 0,5."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Rolling windows",
                "text": "A rolling Hurst exponent is computed on 273 overlapping windows, each compared with a 95% band. About how many windows fall outside the band by chance alone?",
                "options": [
                    "None",
                    "Exactly one",
                    "Half of them",
                    "About 14, and in runs, because the windows overlap"
                ],
                "correctExplanation": "5% of 273 is about 14; overlapping windows make these false alarms come in runs, so only long runs outside the band are informative.",
                "incorrectExplanation": "Each window is a test at 5%, so false alarms are expected; there is no reason for exactly one or for half of them."
            },
            "ro": {
                "title": "Ferestre mobile",
                "text": "Un exponent Hurst pe ferestre mobile este calculat pe 273 de ferestre suprapuse, fiecare comparată cu o bandă de 95%. Aproximativ cîte ferestre ies din bandă doar din întîmplare?",
                "options": [
                    "Niciuna",
                    "Exact una",
                    "Jumătate dintre ele",
                    "Circa 14, și în serii, pentru că ferestrele se suprapun"
                ],
                "correctExplanation": "5% din 273 înseamnă circa 14; ferestrele suprapuse fac ca aceste alarme false să apară în serii, deci doar perioadele lungi din afara benzii sînt informative.",
                "incorrectExplanation": "Fiecare fereastră este un test la 5%, deci alarmele false sînt de așteptat; nu există niciun motiv pentru exact una sau pentru jumătate dintre ele."
            }
        }
    ]
};
