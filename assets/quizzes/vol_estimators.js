// ============================================================
// Chapter 8 quiz bank: volatility estimators and volatility clustering (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['vol-estimators'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Volatility",
                "text": "What does the volatility of an asset measure?",
                "options": [
                    "The average daily return",
                    "The standard deviation of its returns",
                    "The largest daily loss",
                    "The trading volume"
                ],
                "correctExplanation": "Volatility is the standard deviation of returns: the typical size of a move in either direction.",
                "incorrectExplanation": "The mean, the worst loss and the volume are different quantities; volatility measures the dispersion of returns around their mean."
            },
            "ro": {
                "title": "Volatilitatea",
                "text": "Ce măsoară volatilitatea unui activ?",
                "options": [
                    "Randamentul zilnic mediu",
                    "Abaterea standard a randamentelor lui",
                    "Cea mai mare pierdere zilnică",
                    "Volumul tranzacțiilor"
                ],
                "correctExplanation": "Volatilitatea este abaterea standard a randamentelor: mărimea tipică a unei variații, în orice sens.",
                "incorrectExplanation": "Media, cea mai mare pierdere și volumul sînt mărimi diferite; volatilitatea măsoară dispersia randamentelor în jurul mediei."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Annualising Bitcoin",
                "text": "Bitcoin has a daily standard deviation of 3% and trades every day of the year. What is its annual volatility?",
                "options": [
                    "About 47.6%",
                    "About 1095%",
                    "About 57.3%",
                    "About 8.2%"
                ],
                "correctExplanation": "With 365 observations a year: $3 \\times \\sqrt{365} \\approx 57.3\\%$.",
                "incorrectExplanation": "The square-root-of-time rule uses the actual number of observations: 365 for Bitcoin; $\\sqrt{252}$ gives 47.6% and understates the risk."
            },
            "ro": {
                "title": "Anualizarea pentru Bitcoin",
                "text": "Bitcoin are o abatere standard zilnică de 3% și se tranzacționează în fiecare zi a anului. Care este volatilitatea lui anuală?",
                "options": [
                    "Circa 47,6%",
                    "Circa 1095%",
                    "Circa 57,3%",
                    "Circa 8,2%"
                ],
                "correctExplanation": "Cu 365 de observații pe an: $3 \\times \\sqrt{365} \\approx 57{,}3\\%$.",
                "incorrectExplanation": "Regula rădăcinii pătrate a timpului folosește numărul real de observații: 365 pentru Bitcoin; $\\sqrt{252}$ dă 47,6% și subestimează riscul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Latent volatility",
                "text": "Why is the conditional volatility $\\sigma_t$ called latent?",
                "options": [
                    "It is never observed: each day we see only one return drawn with that volatility",
                    "It is published with a delay by the exchange",
                    "It is always equal to the long-run volatility",
                    "It can be computed exactly from the closing price"
                ],
                "correctExplanation": "We observe one return per day; $r_t^2$ is an unbiased but very noisy measure of $\\sigma_t^2$, so $\\sigma_t$ must be estimated.",
                "incorrectExplanation": "No exchange publishes $\\sigma_t$ and it changes over time; one closing price per day cannot reveal it exactly."
            },
            "ro": {
                "title": "Volatilitatea latentă",
                "text": "De ce se spune că volatilitatea condiționată $\\sigma_t$ este latentă?",
                "options": [
                    "Nu este observată niciodată: în fiecare zi vedem doar un randament generat cu acea volatilitate",
                    "Este publicată cu întîrziere de bursă",
                    "Este mereu egală cu volatilitatea pe termen lung",
                    "Poate fi calculată exact din prețul de închidere"
                ],
                "correctExplanation": "Observăm un singur randament pe zi; $r_t^2$ este o măsură nedeplasată, dar foarte zgomotoasă, a lui $\\sigma_t^2$, deci $\\sigma_t$ trebuie estimat.",
                "incorrectExplanation": "Nicio bursă nu publică $\\sigma_t$, iar aceasta se schimbă în timp; un singur preț de închidere pe zi nu o poate dezvălui exact."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Choice of the window",
                "text": "Compared with a 252-day window, what does a 21-day rolling volatility do?",
                "options": [
                    "It is smoother and slower",
                    "It ignores the most recent returns",
                    "It always gives a higher value",
                    "It reacts faster but is noisier"
                ],
                "correctExplanation": "Fewer days react quickly to a new regime but give a less precise estimate: the relative standard error is about $1/\\sqrt{2n}$.",
                "incorrectExplanation": "A short window uses the most recent days only; it is not systematically higher, and it is the long window that is smooth and slow."
            },
            "ro": {
                "title": "Alegerea ferestrei",
                "text": "Față de o fereastră de 252 de zile, ce caracteristică are volatilitatea pe o fereastră mobilă de 21 de zile?",
                "options": [
                    "Este mai netedă și mai lentă",
                    "Ignoră randamentele cele mai recente",
                    "Dă mereu o valoare mai mare",
                    "Reacționează mai repede, dar este mai zgomotoasă"
                ],
                "correctExplanation": "Mai puține zile reacționează repede la un regim nou, dar dau o estimare mai puțin precisă: eroarea standard relativă este circa $1/\\sqrt{2n}$.",
                "incorrectExplanation": "O fereastră scurtă folosește doar zilele recente; nu este sistematic mai mare, iar fereastra lungă este cea netedă și lentă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ghost effect",
                "text": "In June 2020 the 63-day volatility of the S&P 500 fell sharply on a calm day. Why?",
                "options": [
                    "The VIX fell on that day",
                    "A large return from March 2020 left the window",
                    "The S&P 500 changed its composition",
                    "EWMA forecasts were revised"
                ],
                "correctExplanation": "This is the ghost effect: the crash day of 16 March 2020 dropped out of the 63-day window, so the estimate fell at once.",
                "incorrectExplanation": "The rolling window gives equal weight to the last 63 days and zero to older ones; the drop comes from the data leaving the window, not from news."
            },
            "ro": {
                "title": "Ghost effect",
                "text": "În iunie 2020, volatilitatea S&P 500 pe 63 de zile a scăzut brusc într-o zi liniștită. De ce?",
                "options": [
                    "VIX a scăzut în acea zi",
                    "Un randament mare din martie 2020 a ieșit din fereastră",
                    "Indicele S&P 500 și-a schimbat componența",
                    "Prognozele EWMA au fost revizuite"
                ],
                "correctExplanation": "Este vorba de ghost effect: ziua crahului din 16 martie 2020 a ieșit din fereastra de 63 de zile, deci estimarea a scăzut dintr-odată.",
                "incorrectExplanation": "Fereastra mobilă dă aceeași pondere ultimelor 63 de zile și pondere zero celor mai vechi; scăderea vine din datele care ies din fereastră, nu dintr-o știre."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "EWMA update",
                "text": "EWMA with $\\lambda = 0.94$: yesterday's variance is 1.00 (%²) and yesterday's return is $-2\\%$. What is today's variance?",
                "options": [
                    "1.18",
                    "0.94",
                    "4.00",
                    "1.06"
                ],
                "correctExplanation": "$0.94 \\times 1.00 + 0.06 \\times (-2)^2 = 0.94 + 0.24 = 1.18$.",
                "incorrectExplanation": "The new squared return enters with weight $1 - \\lambda = 0.06$ and the old variance with weight $\\lambda = 0.94$; the sign of the return does not matter."
            },
            "ro": {
                "title": "Actualizarea EWMA",
                "text": "EWMA cu $\\lambda = 0{,}94$: varianța de ieri este 1,00 (%²), iar randamentul de ieri este $-2\\%$. Care este varianța de azi?",
                "options": [
                    "1,18",
                    "0,94",
                    "4,00",
                    "1,06"
                ],
                "correctExplanation": "$0{,}94 \\times 1{,}00 + 0{,}06 \\times (-2)^2 = 0{,}94 + 0{,}24 = 1{,}18$.",
                "incorrectExplanation": "Noul randament la pătrat intră cu ponderea $1 - \\lambda = 0{,}06$, iar varianța veche cu ponderea $\\lambda = 0{,}94$; semnul randamentului nu contează."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "EWMA half-life",
                "text": "What is the half-life of the EWMA weights with $\\lambda = 0.94$?",
                "options": [
                    "94 days",
                    "6 days",
                    "About 11 days",
                    "About 74 days"
                ],
                "correctExplanation": "$\\ln 0.5/\\ln 0.94 \\approx 11.2$ days: after 11 days a squared return keeps half of its initial weight.",
                "incorrectExplanation": "The half-life is $\\ln 0.5/\\ln\\lambda$; 74 days is the horizon that carries 99% of the weight, not the half-life."
            },
            "ro": {
                "title": "Timpul de înjumătățire EWMA",
                "text": "Care este timpul de înjumătățire al ponderilor EWMA cu $\\lambda = 0{,}94$?",
                "options": [
                    "94 de zile",
                    "6 zile",
                    "Circa 11 zile",
                    "Circa 74 de zile"
                ],
                "correctExplanation": "$\\ln 0{,}5/\\ln 0{,}94 \\approx 11{,}2$ zile: după 11 zile un randament la pătrat păstrează jumătate din ponderea inițială.",
                "incorrectExplanation": "Timpul de înjumătățire este $\\ln 0{,}5/\\ln\\lambda$; 74 de zile este orizontul care cumulează 99% din pondere, nu timpul de înjumătățire."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "EWMA and GARCH",
                "text": "EWMA is a GARCH(1,1) with $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$. What does this imply for its forecasts?",
                "options": [
                    "They return quickly to the long-run variance",
                    "They are always zero",
                    "They grow without limit with the horizon",
                    "They are flat: the forecast for any horizon equals tomorrow's"
                ],
                "correctExplanation": "With $\\alpha + \\beta = 1$ and $\\omega = 0$ (IGARCH), $E_t[\\sigma^2_{t+h}] = \\sigma^2_{t+1}$ for every $h$: no mean reversion.",
                "incorrectExplanation": "Mean reversion needs $\\alpha + \\beta < 1$ and $\\omega > 0$ (stationary GARCH, Chapter 9); EWMA has neither."
            },
            "ro": {
                "title": "EWMA și GARCH",
                "text": "EWMA este un GARCH(1,1) cu $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$. Ce implică acest lucru pentru prognozele lui?",
                "options": [
                    "Revin repede la varianța pe termen lung",
                    "Sînt mereu zero",
                    "Cresc nelimitat cu orizontul",
                    "Sînt constante: prognoza pentru orice orizont este egală cu cea pentru ziua următoare"
                ],
                "correctExplanation": "Cu $\\alpha + \\beta = 1$ și $\\omega = 0$ (IGARCH), $E_t[\\sigma^2_{t+h}] = \\sigma^2_{t+1}$ pentru orice $h$: nu există revenire la medie.",
                "incorrectExplanation": "Revenirea la medie cere $\\alpha + \\beta < 1$ și $\\omega > 0$ (GARCH staționar, Capitolul 9); EWMA nu le are pe niciuna."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "RiskMetrics",
                "text": "How did RiskMetrics (J.P. Morgan, 1994) arrive at $\\lambda = 0.94$ for daily data?",
                "options": [
                    "By maximum likelihood on the S&P 500",
                    "By minimising the error of variance forecasts across many series",
                    "It is the half-life of a typical shock",
                    "It was required by the Basel Committee"
                ],
                "correctExplanation": "Each series got the $\\lambda$ that minimised the root mean squared error of its variance forecasts; 0.94 is a weighted average over more than 480 series.",
                "incorrectExplanation": "The value was chosen empirically by a forecast criterion, not by a likelihood on one index or by a regulation; 0.94 is a decay factor, not a half-life."
            },
            "ro": {
                "title": "RiskMetrics",
                "text": "Cum a ajuns RiskMetrics (J.P. Morgan, 1994) la $\\lambda = 0{,}94$ pentru date zilnice?",
                "options": [
                    "Prin verosimilitate maximă pe S&P 500",
                    "Prin minimizarea erorii prognozelor de varianță pe multe serii",
                    "Este timpul de înjumătățire al unui șoc tipic",
                    "A fost impus de Comitetul de la Basel"
                ],
                "correctExplanation": "Fiecare serie a primit $\\lambda$ care minimiza rădăcina erorii pătratice medii a prognozelor de varianță; 0,94 este o medie ponderată pe peste 480 de serii.",
                "incorrectExplanation": "Valoarea a fost aleasă empiric, după un criteriu de prognoză, nu prin verosimilitatea unui singur indice sau printr-o reglementare; 0,94 este un factor de atenuare, nu un timp de înjumătățire."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Parkinson estimator",
                "text": "The Parkinson estimator of the daily variance is $(\\ln H_t - \\ln L_t)^2/(4\\ln 2)$. Why the factor $4\\ln 2$?",
                "options": [
                    "For a driftless Brownian motion, the expected squared range is $4\\ln 2$ times the variance",
                    "It converts daily to annual variance",
                    "It corrects for the overnight return",
                    "It is the number of trading hours"
                ],
                "correctExplanation": "$E[(\\ln H - \\ln L)^2] = 4\\ln 2\\,\\sigma^2 \\approx 2.77\\,\\sigma^2$, so dividing by $4\\ln 2$ gives an unbiased estimate under the assumptions.",
                "incorrectExplanation": "The constant comes from the distribution of the range of a Brownian motion; it does not annualise and does not account for the night."
            },
            "ro": {
                "title": "Estimatorul Parkinson",
                "text": "Estimatorul Parkinson al varianței zilnice este $(\\ln H_t - \\ln L_t)^2/(4\\ln 2)$. De ce apare factorul $4\\ln 2$?",
                "options": [
                    "Pentru o mișcare browniană fără tendință, pătratul amplitudinii are media de $4\\ln 2$ ori varianța",
                    "Transformă varianța zilnică în varianță anuală",
                    "Corectează randamentul overnight",
                    "Este numărul de ore de tranzacționare"
                ],
                "correctExplanation": "$E[(\\ln H - \\ln L)^2] = 4\\ln 2\\,\\sigma^2 \\approx 2{,}77\\,\\sigma^2$, deci împărțirea la $4\\ln 2$ dă o estimare nedeplasată în ipotezele modelului.",
                "incorrectExplanation": "Constanta provine din distribuția amplitudinii unei mișcări browniene; nu anualizează și nu ține cont de noapte."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The night",
                "text": "Which estimator includes the overnight return $\\ln(O_t/C_{t-1})$?",
                "options": [
                    "Parkinson",
                    "Garman–Klass",
                    "Rogers–Satchell",
                    "Yang–Zhang"
                ],
                "correctExplanation": "Yang–Zhang adds the sample variance of overnight returns to a weighted mix of open-to-close and Rogers–Satchell variances.",
                "incorrectExplanation": "Parkinson, Garman–Klass and Rogers–Satchell use only prices of the trading session, so they miss the variance that comes overnight."
            },
            "ro": {
                "title": "Noaptea",
                "text": "Care estimator include randamentul overnight $\\ln(O_t/C_{t-1})$?",
                "options": [
                    "Parkinson",
                    "Garman–Klass",
                    "Rogers–Satchell",
                    "Yang–Zhang"
                ],
                "correctExplanation": "Yang–Zhang adaugă varianța de selecție a randamentelor overnight la o combinație ponderată a varianțelor deschidere–închidere și Rogers–Satchell.",
                "incorrectExplanation": "Parkinson, Garman–Klass și Rogers–Satchell folosesc doar prețurile din ședința de tranzacționare, deci omit varianța care apare noaptea."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Rogers–Satchell",
                "text": "What is the main advantage of the Rogers–Satchell estimator over Parkinson and Garman–Klass?",
                "options": [
                    "It needs only closing prices",
                    "It includes the overnight jump",
                    "It stays unbiased when the price has a drift",
                    "It works without any intraday information"
                ],
                "correctExplanation": "$u_t(u_t - c_t) + d_t(d_t - c_t)$ is unbiased for any drift, while Parkinson and Garman–Klass assume a zero drift.",
                "incorrectExplanation": "Rogers–Satchell still needs open, high, low and close prices and still ignores the night; its gain is robustness to a trend."
            },
            "ro": {
                "title": "Rogers–Satchell",
                "text": "Care este principalul avantaj al estimatorului Rogers–Satchell față de Parkinson și Garman–Klass?",
                "options": [
                    "Are nevoie doar de prețurile de închidere",
                    "Include saltul overnight",
                    "Rămîne nedeplasat cînd prețul are o tendință (drift)",
                    "Funcționează fără informație din timpul zilei"
                ],
                "correctExplanation": "$u_t(u_t - c_t) + d_t(d_t - c_t)$ este nedeplasat pentru orice tendință, în timp ce Parkinson și Garman–Klass presupun o tendință nulă.",
                "incorrectExplanation": "Rogers–Satchell are nevoie tot de prețurile de deschidere, maxim, minim și închidere și ignoră tot noaptea; cîștigul lui este robustețea la tendință."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Discrete trading",
                "text": "A thinly traded stock records only a few trades a day. What happens to range-based estimators?",
                "options": [
                    "They overestimate the variance",
                    "They underestimate the variance, because the observed range is too small",
                    "They become exactly unbiased",
                    "Nothing: the range does not depend on the number of trades"
                ],
                "correctExplanation": "With few observed prices the true high and low are missed, so the range shrinks; in our simulation with 26 prices a day the bias is about $-20\\%$ to $-30\\%$.",
                "incorrectExplanation": "The formulas assume continuous trading; when the path is observed rarely, the observed extremes lie inside the true ones."
            },
            "ro": {
                "title": "Tranzacționarea discretă",
                "text": "O acțiune puțin lichidă are doar cîteva tranzacții pe zi. Ce se întîmplă cu estimatorii de amplitudine?",
                "options": [
                    "Supraestimează varianța",
                    "Subestimează varianța, deoarece amplitudinea observată este prea mică",
                    "Devin exact nedeplasați",
                    "Nimic: amplitudinea nu depinde de numărul de tranzacții"
                ],
                "correctExplanation": "Cu puține prețuri observate, maximul și minimul reale nu sînt observate, deci amplitudinea se micșorează; în simularea noastră cu 26 de prețuri pe zi deplasarea este de circa $-20\\%$ pînă la $-30\\%$.",
                "incorrectExplanation": "Formulele presupun tranzacționare continuă; cînd traiectoria este observată rar, extremele observate se află în interiorul celor reale."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Bitcoin and ranges",
                "text": "For Bitcoin, all five estimators (close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) give similar volatilities. Why?",
                "options": [
                    "Bitcoin trades 24 hours a day, so there is almost no overnight return",
                    "Bitcoin has no volatility clustering",
                    "Bitcoin returns are Normal",
                    "Bitcoin has no high and low prices"
                ],
                "correctExplanation": "With no market close, the open equals the previous close and the night share of the variance is almost zero.",
                "incorrectExplanation": "Bitcoin has heavy tails and clustering; what removes the gap is the absence of an overnight period that range estimators cannot see."
            },
            "ro": {
                "title": "Bitcoin și amplitudinile",
                "text": "Pentru Bitcoin, toți cei cinci estimatori (închidere–închidere, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) dau volatilități apropiate. De ce?",
                "options": [
                    "Bitcoin se tranzacționează 24 de ore din 24, deci aproape nu există randament overnight",
                    "Bitcoin nu are volatility clustering",
                    "Randamentele Bitcoin sînt Normale",
                    "Bitcoin nu are prețuri maxime și minime"
                ],
                "correctExplanation": "Fără închiderea pieței, deschiderea este egală cu închiderea precedentă, iar ponderea nopții în varianță este aproape zero.",
                "incorrectExplanation": "Bitcoin are cozi groase și volatility clustering; diferența dispare pentru că lipsește perioada overnight pe care estimatorii de amplitudine nu o includ."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Realised variance",
                "text": "How is the realised variance of a day computed?",
                "options": [
                    "As the square of the daily return",
                    "As the range of the day divided by $4\\ln 2$",
                    "As the average of the last 21 squared daily returns",
                    "As the sum of squared intraday returns, for example every 5 minutes"
                ],
                "correctExplanation": "$\\mathrm{RV}_t = \\sum_i r_{t,i}^2$; as the sampling becomes finer, it converges to the integrated variance of the day.",
                "incorrectExplanation": "The squared daily return and the range use one or two numbers per day; realised variance uses many intraday returns of the same day."
            },
            "ro": {
                "title": "Varianța realizată",
                "text": "Cum se calculează varianța realizată a unei zile?",
                "options": [
                    "Ca pătratul randamentului zilnic",
                    "Ca amplitudinea zilei împărțită la $4\\ln 2$",
                    "Ca media ultimelor 21 de randamente zilnice la pătrat",
                    "Ca suma randamentelor intrazilnice la pătrat, de exemplu la fiecare 5 minute"
                ],
                "correctExplanation": "$\\mathrm{RV}_t = \\sum_i r_{t,i}^2$; cînd eșantionarea devine mai deasă, converge către varianța integrată a zilei.",
                "incorrectExplanation": "Randamentul zilnic la pătrat și amplitudinea folosesc unul sau două numere pe zi; varianța realizată folosește multe randamente intrazilnice din aceeași zi."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Microstructure noise",
                "text": "Why is realised variance from one-minute returns often too high?",
                "options": [
                    "Because one-minute returns are always positive",
                    "Because of the overnight return",
                    "Because microstructure noise (bid–ask bounce, ticks) adds to every squared return",
                    "Because there are too few intraday returns"
                ],
                "correctExplanation": "Each observed return contains noise, whose variance is added once per return; with 390 returns the bias is large, so 5-minute sampling is the usual compromise.",
                "incorrectExplanation": "The problem is not the sign of returns or a lack of data, but the noise that enters every very short return."
            },
            "ro": {
                "title": "Zgomotul de microstructură",
                "text": "De ce varianța realizată din randamente de un minut este adesea prea mare?",
                "options": [
                    "Pentru că randamentele de un minut sînt mereu pozitive",
                    "Din cauza randamentului overnight",
                    "Pentru că zgomotul de microstructură (oscilația bid–ask, pasul de cotare) se adaugă la fiecare randament la pătrat",
                    "Pentru că există prea puține randamente intrazilnice"
                ],
                "correctExplanation": "Fiecare randament observat conține zgomot, a cărui varianță se adaugă o dată pentru fiecare randament; cu 390 de randamente deplasarea este mare, de aceea eșantionarea la 5 minute este compromisul uzual.",
                "incorrectExplanation": "Problema nu este semnul randamentelor sau lipsa datelor, ci zgomotul care intră în fiecare randament foarte scurt."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Evidence of clustering",
                "text": "Which pattern of sample autocorrelations shows volatility clustering?",
                "options": [
                    "ACF of returns near zero, ACF of squared returns positive and slowly decaying",
                    "ACF of returns large and positive at lag 1",
                    "ACF of squared returns exactly zero",
                    "ACF of returns and of squared returns both near zero"
                ],
                "correctExplanation": "The sign of returns is almost unpredictable, but their size is: squared and absolute returns are autocorrelated for many lags.",
                "incorrectExplanation": "Autocorrelated returns would be a predictable mean (Chapter 7); clustering is about the autocorrelation of squared or absolute returns."
            },
            "ro": {
                "title": "Dovada volatility clustering",
                "text": "Ce configurație a autocorelațiilor de selecție arată volatility clustering?",
                "options": [
                    "ACF-ul randamentelor aproape zero, ACF-ul randamentelor la pătrat pozitiv și cu scădere lentă",
                    "ACF-ul randamentelor mare și pozitiv la lagul 1",
                    "ACF-ul randamentelor la pătrat exact zero",
                    "ACF-ul randamentelor și ACF-ul randamentelor la pătrat ambele aproape zero"
                ],
                "correctExplanation": "Semnul randamentelor este aproape imprevizibil, dar mărimea lor nu: randamentele la pătrat și cele în valoare absolută sînt autocorelate pe multe laguri.",
                "incorrectExplanation": "Randamentele autocorelate ar însemna o medie previzibilă (Capitolul 7); volatility clustering privește autocorelația randamentelor la pătrat sau în valoare absolută."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "McLeod–Li test",
                "text": "What is the McLeod–Li test of ARCH effects?",
                "options": [
                    "A Dickey–Fuller test on squared returns",
                    "The Ljung–Box statistic applied to squared returns, compared with $\\chi^2(m)$",
                    "A $t$ test of the mean return",
                    "A test of Normality of the returns"
                ],
                "correctExplanation": "$Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k(r^2)^2/(T-k)$; under no clustering it is approximately $\\chi^2(m)$.",
                "incorrectExplanation": "The test looks at the autocorrelations of $r_t^2$, not at unit roots, the mean or the shape of the distribution."
            },
            "ro": {
                "title": "Testul McLeod–Li",
                "text": "Ce este testul McLeod–Li pentru efecte ARCH?",
                "options": [
                    "Un test Dickey–Fuller pe randamentele la pătrat",
                    "Statistica Ljung–Box aplicată randamentelor la pătrat, comparată cu $\\chi^2(m)$",
                    "Un test $t$ al randamentului mediu",
                    "Un test de normalitate a randamentelor"
                ],
                "correctExplanation": "$Q(m) = T(T+2)\\sum_{k=1}^m \\hat\\rho_k(r^2)^2/(T-k)$; în absența volatility clustering urmează aproximativ o distribuție $\\chi^2(m)$.",
                "incorrectExplanation": "Testul se uită la autocorelațiile lui $r_t^2$, nu la rădăcini unitare, la medie sau la forma distribuției."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ARCH-LM test",
                "text": "An ARCH-LM regression with $q = 5$ lags on $n = 1000$ observations has $R^2 = 0.05$. What do you conclude at 5% ($\\chi^2_{0.95}(5) = 11.07$)?",
                "options": [
                    "LM = 0.05: no ARCH effects",
                    "LM = 5: no ARCH effects",
                    "LM = 250: ARCH effects",
                    "LM = 50: reject the null of no ARCH effects"
                ],
                "correctExplanation": "$\\mathrm{LM} = nR^2 = 1000 \\times 0.05 = 50 > 11.07$: the squared returns are predictable from their own lags.",
                "incorrectExplanation": "The statistic is $n$ times $R^2$, compared with the $\\chi^2(q)$ critical value; neither $R^2$ alone nor $5 \\times 50$ is the statistic."
            },
            "ro": {
                "title": "Testul ARCH-LM",
                "text": "O regresie ARCH-LM cu $q = 5$ laguri pe $n = 1000$ de observații are $R^2 = 0{,}05$. Ce concluzie trageți la 5% ($\\chi^2_{0{,}95}(5) = 11{,}07$)?",
                "options": [
                    "LM = 0,05: fără efecte ARCH",
                    "LM = 5: fără efecte ARCH",
                    "LM = 250: efecte ARCH",
                    "LM = 50: respingem ipoteza nulă a absenței efectelor ARCH"
                ],
                "correctExplanation": "$\\mathrm{LM} = nR^2 = 1000 \\times 0{,}05 = 50 > 11{,}07$: randamentele la pătrat pot fi anticipate din propriile laguri.",
                "incorrectExplanation": "Statistica este $n$ înmulțit cu $R^2$ și se compară cu valoarea critică $\\chi^2(q)$; nici $R^2$ singur, nici $5 \\times 50$ nu reprezintă statistica."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Shuffled returns",
                "text": "The S&P 500 daily returns are put in a random order. Which statement is true?",
                "options": [
                    "The kurtosis falls to 3",
                    "Volatility clustering stays the same",
                    "The kurtosis is unchanged, but the clustering disappears",
                    "The returns become autocorrelated"
                ],
                "correctExplanation": "Shuffling keeps every value, hence the distribution and its kurtosis, but destroys the order in which large moves follow each other.",
                "incorrectExplanation": "Kurtosis depends only on the values; clustering depends on their order, which the shuffle removes."
            },
            "ro": {
                "title": "Randamente amestecate",
                "text": "Randamentele zilnice ale S&P 500 sînt puse într-o ordine aleatoare. Care afirmație este adevărată?",
                "options": [
                    "Coeficientul de boltire scade la 3",
                    "Volatility clustering rămîne la fel",
                    "Coeficientul de boltire nu se schimbă, dar volatility clustering dispare",
                    "Randamentele devin autocorelate"
                ],
                "correctExplanation": "Amestecarea păstrează toate valorile, deci distribuția și boltirea, dar distruge ordinea în care variațiile mari se succed.",
                "incorrectExplanation": "Boltirea depinde doar de valori; volatility clustering depinde de ordinea lor, pe care amestecarea o elimină."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Clustering and tails",
                "text": "Calm days have a standard deviation of 1 and turbulent days of 2; each day is Normal. What is the kurtosis of all returns together?",
                "options": [
                    "Larger than 3: mixing variances creates heavy tails",
                    "Exactly 3, because each day is Normal",
                    "Smaller than 3",
                    "It cannot be computed"
                ],
                "correctExplanation": "Kurtosis $= 3E[\\sigma^4]/(E[\\sigma^2])^2 > 3$ whenever $\\sigma$ varies; with 20% turbulent days it is about 4.7.",
                "incorrectExplanation": "A mixture of Normal distributions with different variances is not Normal: the variation of $\\sigma$ fattens the tails."
            },
            "ro": {
                "title": "Volatility clustering și cozile",
                "text": "Zilele liniștite au abaterea standard 1, iar cele agitate 2; fiecare zi este Normală. Care este boltirea tuturor randamentelor împreună?",
                "options": [
                    "Mai mare decît 3: amestecul de varianțe produce cozi groase",
                    "Exact 3, deoarece fiecare zi este Normală",
                    "Mai mică decît 3",
                    "Nu poate fi calculată"
                ],
                "correctExplanation": "Coeficientul de boltire este $3E[\\sigma^4]/(E[\\sigma^2])^2 > 3$ ori de cîte ori $\\sigma$ variază; cu 20% zile agitate este circa 4,7.",
                "incorrectExplanation": "Un amestec de distribuții Normale cu varianțe diferite nu este Normal: variația lui $\\sigma$ îngroașă cozile."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "VIX",
                "text": "Since 1990 the VIX has been above the realised volatility of the next 21 days on most days. What is the usual explanation?",
                "options": [
                    "The VIX is computed with a wrong formula",
                    "A variance risk premium: investors pay for protection against volatility spikes",
                    "The VIX measures the volatility of the next year",
                    "Realised volatility ignores negative returns"
                ],
                "correctExplanation": "Implied volatility includes a premium for bearing volatility risk, so it exceeds the volatility that follows on average.",
                "incorrectExplanation": "The VIX measures 30-day implied volatility from S&P 500 options; the gap is a risk premium, not an error of computation."
            },
            "ro": {
                "title": "VIX",
                "text": "Din 1990, VIX a fost peste volatilitatea realizată din următoarele 21 de zile în majoritatea zilelor. Care este explicația uzuală?",
                "options": [
                    "VIX este calculat cu o formulă greșită",
                    "O primă de risc a varianței: investitorii plătesc pentru protecție împotriva salturilor de volatilitate",
                    "VIX măsoară volatilitatea pentru anul următor",
                    "Volatilitatea realizată ignoră randamentele negative"
                ],
                "correctExplanation": "Volatilitatea implicită include o primă pentru asumarea riscului de volatilitate, deci depășește, în medie, volatilitatea care urmează.",
                "incorrectExplanation": "VIX măsoară volatilitatea implicită pe 30 de zile din opțiunile pe S&P 500; diferența este o primă de risc, nu o eroare de calcul."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Leverage effect",
                "text": "What does the leverage effect state?",
                "options": [
                    "High volatility causes high future returns",
                    "Leveraged funds have lower volatility",
                    "Volatility does not depend on past returns",
                    "Negative returns are followed by higher volatility than positive returns of the same size"
                ],
                "correctExplanation": "In equity indices $\\mathrm{corr}(r_t, |r_{t+j}|) < 0$: a fall today announces turbulence tomorrow.",
                "incorrectExplanation": "The effect is an asymmetry in the dynamics of volatility; it says nothing about future returns or about fund leverage."
            },
            "ro": {
                "title": "Efectul de levier",
                "text": "Ce afirmă efectul de levier?",
                "options": [
                    "Volatilitatea ridicată produce randamente viitoare ridicate",
                    "Fondurile cu levier au volatilitate mai mică",
                    "Volatilitatea nu depinde de randamentele trecute",
                    "Randamentele negative sînt urmate de o volatilitate mai mare decît randamentele pozitive de aceeași mărime"
                ],
                "correctExplanation": "La indicii bursieri $\\mathrm{corr}(r_t, |r_{t+j}|) < 0$: o scădere azi este urmată de o volatilitate mai mare mîine.",
                "incorrectExplanation": "Efectul este o asimetrie în dinamica volatilității; nu spune nimic despre randamentele viitoare sau despre levierul fondurilor."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "QLIKE",
                "text": "The true variance is 1. Forecast A is 0.5 and forecast B is 2. How do MSE and QLIKE rank them?",
                "options": [
                    "Both prefer B",
                    "Both prefer A",
                    "MSE prefers A, QLIKE prefers B: QLIKE punishes under-prediction more",
                    "MSE prefers B, QLIKE prefers A"
                ],
                "correctExplanation": "MSE: 0.25 against 1; QLIKE ($1/h + \\ln h$): 1.307 against 1.193. Under-predicting risk is the costlier error for VaR.",
                "incorrectExplanation": "MSE is symmetric in the error and so penalises the larger miss of B; QLIKE penalises forecasts that are too low more heavily."
            },
            "ro": {
                "title": "QLIKE",
                "text": "Varianța reală este 1. Prognoza A este 0,5, iar prognoza B este 2. Cum le ordonează MSE și QLIKE?",
                "options": [
                    "Ambele preferă B",
                    "Ambele preferă A",
                    "MSE preferă A, QLIKE preferă B: QLIKE penalizează mai mult subestimarea",
                    "MSE preferă B, QLIKE preferă A"
                ],
                "correctExplanation": "MSE: 0,25 față de 1; QLIKE ($1/h + \\ln h$): 1,307 față de 1,193. Subestimarea riscului este eroarea mai costisitoare pentru VaR.",
                "incorrectExplanation": "MSE este simetrică în eroare și penalizează abaterea mai mare a lui B; QLIKE penalizează mai mult prognozele prea mici."
            }
        }
    ]
};
