// ============================================================
// Chapter 4 quiz bank: Probability (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['probability'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Variance of a linear transformation",
                "text": "If $\\mathrm{Var}(X) = 4$, what is $\\mathrm{Var}(3X + 5)$?",
                "options": ["$17$", "$12$", "$36$", "$41$"],
                "correctExplanation": "$\\mathrm{Var}(aX + b) = a^2\\mathrm{Var}(X) = 9 \\times 4 = 36$: a shift does not change the variance, a scale multiplies it by its square.",
                "incorrectExplanation": "The constant 5 does not change the spread, and the factor 3 enters squared: $9 \\times 4 = 36$."
            },
            "ro": {
                "title": "Varianța unei transformări liniare",
                "text": "Dacă $\\mathrm{Var}(X) = 4$, cît este $\\mathrm{Var}(3X + 5)$?",
                "options": ["$17$", "$12$", "$36$", "$41$"],
                "correctExplanation": "$\\mathrm{Var}(aX + b) = a^2\\mathrm{Var}(X) = 9 \\times 4 = 36$: o translație nu schimbă varianța, o scalare o înmulțește cu pătratul ei.",
                "incorrectExplanation": "Constanta 5 nu schimbă împrăștierea, iar factorul 3 intră la pătrat: $9 \\times 4 = 36$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Variance of a portfolio",
                "text": "A portfolio holds weights $w_1$, $w_2$ in two assets with volatilities $\\sigma_1$, $\\sigma_2$ and covariance $\\sigma_{12}$. What is its variance?",
                "options": ["$w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\sigma_{12}$", "$w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2$", "$(w_1\\sigma_1 + w_2\\sigma_2)^2$ always", "$w_1\\sigma_1^2 + w_2\\sigma_2^2$"],
                "correctExplanation": "$\\mathrm{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\mathrm{Cov}(X, Y)$; the covariance term is what makes diversification work.",
                "incorrectExplanation": "The covariance term $2w_1w_2\\sigma_{12}$ cannot be dropped; $(w_1\\sigma_1 + w_2\\sigma_2)^2$ holds only when the correlation is 1."
            },
            "ro": {
                "title": "Varianța unui portofoliu",
                "text": "Un portofoliu are ponderile $w_1$, $w_2$ în două active cu volatilitățile $\\sigma_1$, $\\sigma_2$ și covarianța $\\sigma_{12}$. Cît este varianța lui?",
                "options": ["$w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\sigma_{12}$", "$w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2$", "$(w_1\\sigma_1 + w_2\\sigma_2)^2$ întotdeauna", "$w_1\\sigma_1^2 + w_2\\sigma_2^2$"],
                "correctExplanation": "$\\mathrm{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\mathrm{Cov}(X, Y)$; termenul de covarianță este cel care face diversificarea posibilă.",
                "incorrectExplanation": "Termenul de covarianță $2w_1w_2\\sigma_{12}$ nu poate fi omis; $(w_1\\sigma_1 + w_2\\sigma_2)^2$ este valabil doar cînd corelația este 1."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Uncorrelated vs independent",
                "text": "Let $X \\sim N(0, 1)$ and $Y = X^2$. Which statement is true?",
                "options": ["$X$ and $Y$ are independent because their correlation is zero", "$X$ and $Y$ are positively correlated", "$X$ and $Y$ are independent because $Y$ is always positive", "$X$ and $Y$ are uncorrelated but dependent"],
                "correctExplanation": "$\\mathrm{Cov}(X, X^2) = E[X^3] = 0$, so they are uncorrelated, yet $Y$ is a function of $X$: knowing $X$ gives $Y$ exactly.",
                "incorrectExplanation": "Zero correlation only rules out linear dependence; $Y = X^2$ is completely determined by $X$."
            },
            "ro": {
                "title": "Necorelare și independență",
                "text": "Fie $X \\sim N(0, 1)$ și $Y = X^2$. Care afirmație este adevărată?",
                "options": ["$X$ și $Y$ sînt independente pentru că au corelația zero", "$X$ și $Y$ sînt corelate pozitiv", "$X$ și $Y$ sînt independente pentru că $Y$ este mereu pozitiv", "$X$ și $Y$ sînt necorelate, dar dependente"],
                "correctExplanation": "$\\mathrm{Cov}(X, X^2) = E[X^3] = 0$, deci sînt necorelate, dar $Y$ este o funcție de $X$: cunoscînd $X$, aflăm exact $Y$.",
                "incorrectExplanation": "Corelația zero exclude doar dependența liniară; $Y = X^2$ este complet determinat de $X$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Returns and squared returns",
                "text": "For the S&P 500, 2000–2026, $\\mathrm{Corr}(r_t, r_{t-1}) \\approx -0.10$ and $\\mathrm{Corr}(r_t^2, r_{t-1}^2) \\approx 0.31$, with an i.i.d. band of $\\pm 0.024$. What follows?",
                "options": ["Returns are independent", "Returns are not independent: the size of a move is predictable from yesterday", "Returns are Normal", "Squared returns are white noise"],
                "correctExplanation": "Independence would make the squares uncorrelated too; their correlation of 0.31 is volatility clustering.",
                "incorrectExplanation": "A strong correlation of squared returns is incompatible with independence: volatility clusters."
            },
            "ro": {
                "title": "Randamente și pătratele lor",
                "text": "Pentru S&P 500, 2000–2026, $\\mathrm{Corr}(r_t, r_{t-1}) \\approx -0{,}10$ și $\\mathrm{Corr}(r_t^2, r_{t-1}^2) \\approx 0{,}31$, cu o bandă i.i.d. de $\\pm 0{,}024$. Ce rezultă?",
                "options": ["Randamentele sînt independente", "Randamentele nu sînt independente: mărimea unei mișcări este previzibilă pe baza zilei de ieri", "Randamentele au distribuție Normală", "Pătratele randamentelor sînt zgomot alb"],
                "correctExplanation": "Independența ar face și pătratele necorelate; corelația lor de 0,31 este volatility clustering.",
                "incorrectExplanation": "O corelație puternică a pătratelor randamentelor nu este compatibilă cu independența: volatilitatea apare grupat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The tower property",
                "text": "What is $E\\big[E[Y \\mid X]\\big]$?",
                "options": ["$E[X]$", "$E[Y]$", "$E[XY]$", "$\\mathrm{Var}(Y)$"],
                "correctExplanation": "The law of iterated expectations: averaging the conditional mean over $X$ gives back the unconditional mean $E[Y]$.",
                "incorrectExplanation": "Averaging the conditional expectation over all values of $X$ gives back $E[Y]$ (law of iterated expectations)."
            },
            "ro": {
                "title": "Proprietatea turnului",
                "text": "Cît este $E\\big[E[Y \\mid X]\\big]$?",
                "options": ["$E[X]$", "$E[Y]$", "$E[XY]$", "$\\mathrm{Var}(Y)$"],
                "correctExplanation": "Legea speranțelor iterate: media condiționată, luată în medie după $X$, redă media necondiționată $E[Y]$.",
                "incorrectExplanation": "Media speranței condiționate, calculată pe toate valorile lui $X$, redă $E[Y]$ (legea speranțelor iterate)."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Law of total variance",
                "text": "Which decomposition is correct?",
                "options": ["$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)]$", "$\\mathrm{Var}(Y) = \\mathrm{Var}(E[Y \\mid X])$", "$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)] - \\mathrm{Var}(E[Y \\mid X])$", "$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)] + \\mathrm{Var}(E[Y \\mid X])$"],
                "correctExplanation": "Total variance = average within-group variance + variance of the group means.",
                "incorrectExplanation": "Both parts are needed and they add up: the average conditional variance plus the variance of the conditional mean."
            },
            "ro": {
                "title": "Legea varianței totale",
                "text": "Care descompunere este corectă?",
                "options": ["$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)]$", "$\\mathrm{Var}(Y) = \\mathrm{Var}(E[Y \\mid X])$", "$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)] - \\mathrm{Var}(E[Y \\mid X])$", "$\\mathrm{Var}(Y) = E[\\mathrm{Var}(Y \\mid X)] + \\mathrm{Var}(E[Y \\mid X])$"],
                "correctExplanation": "Varianța totală = media varianțelor din interiorul grupurilor + varianța mediilor grupurilor.",
                "incorrectExplanation": "Sînt necesare ambele părți și se adună: media varianței condiționate plus varianța mediei condiționate."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "A mixture of two Normal regimes",
                "text": "Daily returns are $N(0, 0.8^2)$ on calm days (probability 0.8) and $N(0, 2^2)$ on turbulent days. What is the kurtosis of the unconditional distribution?",
                "options": ["About 6.1, above the Normal value of 3", "Exactly 3, because each regime is Normal", "Below 3", "It cannot be computed"],
                "correctExplanation": "$\\mathrm{Var} = 1.312$, $E[r^4] = 3(0.8 \\times 0.8^4 + 0.2 \\times 2^4) = 10.58$, kurtosis $10.58/1.312^2 \\approx 6.1$: mixing variances creates heavy tails.",
                "incorrectExplanation": "A mixture of Normals with different variances is not Normal: its kurtosis is about 6.1, the mechanism behind GARCH tails."
            },
            "ro": {
                "title": "Un amestec de două regimuri Normale",
                "text": "Randamentele zilnice sînt $N(0, 0{,}8^2)$ în zilele calme (probabilitatea 0,8) și $N(0, 2^2)$ în zilele agitate. Cît este boltirea distribuției necondiționate?",
                "options": ["Circa 6,1, peste valoarea 3 a distribuției Normale", "Exact 3, pentru că fiecare regim este Normal", "Sub 3", "Nu se poate calcula"],
                "correctExplanation": "$\\mathrm{Var} = 1{,}312$, $E[r^4] = 3(0{,}8 \\times 0{,}8^4 + 0{,}2 \\times 2^4) = 10{,}58$, boltirea $10{,}58/1{,}312^2 \\approx 6{,}1$: amestecarea varianțelor creează cozi groase.",
                "incorrectExplanation": "Un amestec de distribuții Normale cu varianțe diferite nu este Normal: boltirea lui este circa 6,1, mecanismul din spatele cozilor GARCH."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Conditional volatility",
                "text": "For the S&P 500, the standard deviation of today's return is 0.91% after the calmest 20% of days and 1.71% after the most turbulent 20%, while the conditional mean barely changes. Which model idea does this support?",
                "options": ["A constant variance and a predictable mean", "Returns are a random walk in levels", "A conditional variance that depends on past returns, as in ARCH and GARCH", "Normal i.i.d. returns"],
                "correctExplanation": "The variance, not the mean, depends on yesterday: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ is what ARCH and GARCH model.",
                "incorrectExplanation": "The pattern is a time-varying conditional variance with an almost constant conditional mean: the ARCH/GARCH idea."
            },
            "ro": {
                "title": "Volatilitatea condiționată",
                "text": "Pentru S&P 500, abaterea standard a randamentului de azi este 0,91% după cele mai calme 20% dintre zile și 1,71% după cele mai agitate 20%, iar media condiționată abia se schimbă. Ce idee de model susține acest lucru?",
                "options": ["O varianță constantă și o medie previzibilă", "Randamentele sînt un mers aleator în niveluri", "O varianță condiționată care depinde de randamentele trecute, ca în ARCH și GARCH", "Randamente Normale i.i.d."],
                "correctExplanation": "Varianța, nu media, depinde de ziua de ieri: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ este ceea ce modelează ARCH și GARCH.",
                "incorrectExplanation": "Tiparul este o varianță condiționată variabilă în timp, cu o medie condiționată aproape constantă: ideea ARCH/GARCH."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "VaR 1% as a quantile",
                "text": "Let $q_{0.01}$ be the 1% quantile of the daily return. What is VaR 1%?",
                "options": ["$q_{0.99}$", "The mean loss below $q_{0.01}$", "$1\\%$ of the portfolio value", "$-q_{0.01}$, the loss exceeded with probability 1%"],
                "correctExplanation": "$\\mathrm{VaR}_{1\\%} = -q_{0.01}$: the loss that is exceeded on 1% of the days.",
                "incorrectExplanation": "VaR 1% is minus the 1% quantile of the return; the mean loss beyond it is the Expected Shortfall."
            },
            "ro": {
                "title": "VaR 1% ca o cuantilă",
                "text": "Fie $q_{0,01}$ cuantila de 1% a randamentului zilnic. Ce este VaR 1%?",
                "options": ["$q_{0,99}$", "Pierderea medie sub $q_{0,01}$", "$1\\%$ din valoarea portofoliului", "$-q_{0,01}$, pierderea depășită cu probabilitatea 1%"],
                "correctExplanation": "$\\mathrm{VaR}_{1\\%} = -q_{0,01}$: pierderea depășită în 1% din zile.",
                "incorrectExplanation": "VaR 1% este minus cuantila de 1% a randamentului; pierderea medie dincolo de ea este Expected Shortfall."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Chebyshev's inequality",
                "text": "For any distribution with finite variance, at most what share of observations lies 4 or more standard deviations from the mean?",
                "options": ["0.006%", "4%", "6.25%", "25%"],
                "correctExplanation": "$P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2 = 1/16 = 6.25\\%$; the S&P 500 had about 0.7%, the Normal distribution gives 0.006%.",
                "incorrectExplanation": "Chebyshev gives $1/k^2 = 6.25\\%$ for $k = 4$; 0.006% is the Normal value, not a bound valid for every distribution."
            },
            "ro": {
                "title": "Inegalitatea lui Cebîșev",
                "text": "Pentru orice distribuție cu varianță finită, cel mult ce pondere din observații se află la 4 sau mai multe abateri standard de medie?",
                "options": ["0,006%", "4%", "6,25%", "25%"],
                "correctExplanation": "$P(|X - \\mu| \\ge k\\sigma) \\le 1/k^2 = 1/16 = 6{,}25\\%$; S&P 500 a avut circa 0,7%, iar distribuția Normală dă 0,006%.",
                "incorrectExplanation": "Cebîșev dă $1/k^2 = 6{,}25\\%$ pentru $k = 4$; 0,006% este valoarea pentru distribuția Normală, nu o margine valabilă pentru orice distribuție."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The law of large numbers and the mean return",
                "text": "Over 26.7 years the S&P 500 had a mean log return of about 6.2% a year with a standard error of about 3.7%. What does this show?",
                "options": ["The mean return is known to within 0.1%", "Even decades of daily data give a wide confidence interval for the expected return", "The law of large numbers fails for returns", "The expected return is zero"],
                "correctExplanation": "The standard error of a mean is $\\sigma/\\sqrt{n}$; with $\\sigma \\approx 19\\%$ a year, 26 years give a 95% interval of roughly $[-1\\%, 14\\%]$.",
                "incorrectExplanation": "The LLN holds, but slowly: the 95% interval for the annual mean is roughly $[-1\\%, 14\\%]$."
            },
            "ro": {
                "title": "Legea numerelor mari și randamentul mediu",
                "text": "În 26,7 ani, S&P 500 a avut un randament logaritmic mediu de circa 6,2% pe an, cu o eroare standard de circa 3,7%. Ce arată acest lucru?",
                "options": ["Randamentul mediu este cunoscut cu o precizie de 0,1%", "Chiar și decenii de date zilnice dau un interval de încredere larg pentru randamentul așteptat", "Legea numerelor mari nu funcționează pentru randamente", "Randamentul așteptat este zero"],
                "correctExplanation": "Eroarea standard a unei medii este $\\sigma/\\sqrt{n}$; cu $\\sigma \\approx 19\\%$ pe an, 26 de ani dau un interval de 95% de circa $[-1\\%, 14\\%]$.",
                "incorrectExplanation": "LLN funcționează, dar încet: intervalul de 95% pentru media anuală este de circa $[-1\\%, 14\\%]$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Monte Carlo standard error",
                "text": "A probability of about 3% is estimated from $N = 10{,}000$ simulations. What is the standard error of the estimate?",
                "options": ["About 0.0017", "About 0.0001", "About 0.03", "About 0.0003"],
                "correctExplanation": "$\\sqrt{p(1 - p)/N} = \\sqrt{0.03 \\times 0.97/10{,}000} \\approx 0.0017$.",
                "incorrectExplanation": "The error is $\\sqrt{p(1 - p)/N} \\approx 0.0017$; it falls like $1/\\sqrt{N}$, not $1/N$."
            },
            "ro": {
                "title": "Eroarea standard Monte Carlo",
                "text": "O probabilitate de circa 3% este estimată din $N = 10\\,000$ de simulări. Cît este eroarea standard a estimării?",
                "options": ["Circa 0,0017", "Circa 0,0001", "Circa 0,03", "Circa 0,0003"],
                "correctExplanation": "$\\sqrt{p(1 - p)/N} = \\sqrt{0{,}03 \\times 0{,}97/10\\,000} \\approx 0{,}0017$.",
                "incorrectExplanation": "Eroarea este $\\sqrt{p(1 - p)/N} \\approx 0{,}0017$; scade ca $1/\\sqrt{N}$, nu ca $1/N$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Halving the Monte Carlo error",
                "text": "How many times more simulations are needed to halve the standard error of a Monte Carlo estimate?",
                "options": ["4 times", "2 times", "$\\sqrt{2}$ times", "10 times"],
                "correctExplanation": "The error is proportional to $1/\\sqrt{N}$, so halving it requires $N$ to grow by a factor of 4.",
                "incorrectExplanation": "Because the error scales with $1/\\sqrt{N}$, halving it needs 4 times as many simulations."
            },
            "ro": {
                "title": "Înjumătățirea erorii Monte Carlo",
                "text": "De cîte ori mai multe simulări sînt necesare pentru a înjumătăți eroarea standard a unei estimări Monte Carlo?",
                "options": ["De 4 ori", "De 2 ori", "De $\\sqrt{2}$ ori", "De 10 ori"],
                "correctExplanation": "Eroarea este proporțională cu $1/\\sqrt{N}$, deci înjumătățirea ei cere ca $N$ să crească de 4 ori.",
                "incorrectExplanation": "Pentru că eroarea scade ca $1/\\sqrt{N}$, înjumătățirea ei cere de 4 ori mai multe simulări."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Inverse transform for the exponential distribution",
                "text": "$U \\sim U(0, 1)$. Which transformation gives an exponential variable with rate $\\lambda$?",
                "options": ["$\\lambda U$", "$-\\ln(1 - U)/\\lambda$", "$e^{-\\lambda U}$", "$\\Phi^{-1}(U)/\\lambda$"],
                "correctExplanation": "$F(x) = 1 - e^{-\\lambda x}$, so $F^{-1}(u) = -\\ln(1 - u)/\\lambda$ and $F^{-1}(U)$ has CDF $F$.",
                "incorrectExplanation": "The inverse transform uses $F^{-1}$ of the exponential CDF $1 - e^{-\\lambda x}$, which is $-\\ln(1 - u)/\\lambda$."
            },
            "ro": {
                "title": "Transformarea inversă pentru distribuția exponențială",
                "text": "$U \\sim U(0, 1)$. Ce transformare dă o variabilă exponențială cu rata $\\lambda$?",
                "options": ["$\\lambda U$", "$-\\ln(1 - U)/\\lambda$", "$e^{-\\lambda U}$", "$\\Phi^{-1}(U)/\\lambda$"],
                "correctExplanation": "$F(x) = 1 - e^{-\\lambda x}$, deci $F^{-1}(u) = -\\ln(1 - u)/\\lambda$, iar $F^{-1}(U)$ are CDF $F$.",
                "incorrectExplanation": "Transformarea inversă folosește $F^{-1}$ pentru CDF exponențială $1 - e^{-\\lambda x}$, adică $-\\ln(1 - u)/\\lambda$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The RANDU generator",
                "text": "RANDU ($x_{k+1} = 65539\\,x_k \\bmod 2^{31}$) passes tests of the mean, the variance and the lag-1 correlation. What is its flaw?",
                "options": ["Its numbers are not between 0 and 1", "Its period is 16", "Consecutive triples lie on only 15 parallel planes", "It cannot be started from a chosen seed"],
                "correctExplanation": "$9u_k - 6u_{k+1} + u_{k+2}$ is always an integer: the triples fall on 15 planes (Marsaglia, 1968), which biases any 3-dimensional simulation.",
                "incorrectExplanation": "The flaw is in the joint distribution of consecutive numbers: the triples lie on 15 planes, invisible to one-dimensional tests."
            },
            "ro": {
                "title": "Generatorul RANDU",
                "text": "RANDU ($x_{k+1} = 65539\\,x_k \\bmod 2^{31}$) trece testele mediei, varianței și corelației la decalajul 1. Care este defectul lui?",
                "options": ["Numerele lui nu sînt între 0 și 1", "Perioada lui este 16", "Tripletele consecutive stau pe doar 15 plane paralele", "Nu poate fi pornit dintr-o sămînță aleasă"],
                "correctExplanation": "$9u_k - 6u_{k+1} + u_{k+2}$ este întotdeauna întreg: tripletele cad pe 15 plane (Marsaglia, 1968), ceea ce deformează orice simulare tridimensională.",
                "incorrectExplanation": "Defectul este în distribuția comună a numerelor consecutive: tripletele stau pe 15 plane, invizibile pentru testele unidimensionale."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Historical simulation",
                "text": "Drawing $U \\sim U(0, 1)$ and returning the empirical $U$-quantile of past daily returns is the same as:",
                "options": ["Simulating from the Normal distribution", "Fitting a Student-t distribution", "Simulating a GARCH model", "Resampling past days with replacement (historical simulation, bootstrap)"],
                "correctExplanation": "The empirical quantile function puts mass $1/n$ on each observed return: the inverse transform becomes resampling of past days.",
                "incorrectExplanation": "The empirical inverse transform resamples observed returns, so it keeps their heavy tails but not their time order."
            },
            "ro": {
                "title": "Simularea istorică",
                "text": "Extragerea lui $U \\sim U(0, 1)$ și folosirea cuantilei empirice de ordin $U$ a randamentelor zilnice trecute echivalează cu:",
                "options": ["Simularea din distribuția Normală", "Estimarea unei distribuții Student-t", "Simularea unui model GARCH", "Reeșantionarea cu întoarcere a zilelor trecute (simulare istorică, bootstrap)"],
                "correctExplanation": "Funcția cuantilă empirică pune masa $1/n$ pe fiecare randament observat: transformarea inversă devine reeșantionarea zilelor trecute.",
                "incorrectExplanation": "Transformarea inversă empirică reeșantionează randamentele observate, deci păstrează cozile lor groase, dar nu și ordinea în timp."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "A two-step binomial tree",
                "text": "$S_0 = 100$, $u = 1.2$, $d = 0.8$, $p = 0.6$, two independent steps. What is $E[S_2]$?",
                "options": ["100", "104", "108.16", "144"],
                "correctExplanation": "$E[S_2] = 100(pu + (1 - p)d)^2 = 100 \\times 1.04^2 = 108.16$.",
                "incorrectExplanation": "Each step multiplies the expected price by $pu + (1 - p)d = 1.04$, so $E[S_2] = 100 \\times 1.04^2 = 108.16$."
            },
            "ro": {
                "title": "Un arbore binomial cu doi pași",
                "text": "$S_0 = 100$, $u = 1{,}2$, $d = 0{,}8$, $p = 0{,}6$, doi pași independenți. Cît este $E[S_2]$?",
                "options": ["100", "104", "108,16", "144"],
                "correctExplanation": "$E[S_2] = 100(pu + (1 - p)d)^2 = 100 \\times 1{,}04^2 = 108{,}16$.",
                "incorrectExplanation": "Fiecare pas înmulțește prețul așteptat cu $pu + (1 - p)d = 1{,}04$, deci $E[S_2] = 100 \\times 1{,}04^2 = 108{,}16$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The martingale probability of a tree",
                "text": "In a binomial tree with factors $u > 1 > d$ and zero interest rate, which up-probability makes the price a martingale?",
                "options": ["$p = 1/2$ always", "$p = u/(u + d)$", "$p = (u - 1)/(u - d)$", "$p^* = (1 - d)/(u - d)$"],
                "correctExplanation": "$E[S_{t+1} \\mid S_t] = S_t(pu + (1 - p)d) = S_t$ gives $p^* = (1 - d)/(u - d)$.",
                "incorrectExplanation": "Solve $pu + (1 - p)d = 1$: the fair-game probability is $(1 - d)/(u - d)$, which equals 1/2 only for symmetric factors."
            },
            "ro": {
                "title": "Probabilitatea de martingal a unui arbore",
                "text": "Într-un arbore binomial cu factorii $u > 1 > d$ și dobîndă zero, ce probabilitate de creștere face din preț un martingal?",
                "options": ["$p = 1/2$ întotdeauna", "$p = u/(u + d)$", "$p = (u - 1)/(u - d)$", "$p^* = (1 - d)/(u - d)$"],
                "correctExplanation": "$E[S_{t+1} \\mid S_t] = S_t(pu + (1 - p)d) = S_t$ dă $p^* = (1 - d)/(u - d)$.",
                "incorrectExplanation": "Rezolvați $pu + (1 - p)d = 1$: probabilitatea jocului echitabil este $(1 - d)/(u - d)$, egală cu 1/2 doar pentru factori simetrici."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Variance of a random walk",
                "text": "$S_t = S_{t-1} + \\varepsilon_t$ with i.i.d. $\\varepsilon_t$ of variance $\\sigma^2$ and fixed $S_0$. What is $\\mathrm{Var}(S_t)$?",
                "options": ["$\\sigma^2$ for every $t$", "$t\\sigma^2$, so the random walk is not stationary", "$\\sigma^2/(1 - t)$", "$\\sqrt{t}\\,\\sigma$"],
                "correctExplanation": "$S_t = S_0 + \\sum_{s \\le t}\\varepsilon_s$, a sum of $t$ independent shocks: variance $t\\sigma^2$, growing without bound.",
                "incorrectExplanation": "The variance of a sum of $t$ independent shocks is $t\\sigma^2$; the standard deviation, not the variance, grows like $\\sqrt{t}$."
            },
            "ro": {
                "title": "Varianța unui mers aleator",
                "text": "$S_t = S_{t-1} + \\varepsilon_t$, cu $\\varepsilon_t$ i.i.d. de varianță $\\sigma^2$ și $S_0$ fix. Cît este $\\mathrm{Var}(S_t)$?",
                "options": ["$\\sigma^2$ pentru orice $t$", "$t\\sigma^2$, deci mersul aleator nu este staționar", "$\\sigma^2/(1 - t)$", "$\\sqrt{t}\\,\\sigma$"],
                "correctExplanation": "$S_t = S_0 + \\sum_{s \\le t}\\varepsilon_s$, o sumă de $t$ șocuri independente: varianța $t\\sigma^2$, care crește nelimitat.",
                "incorrectExplanation": "Varianța unei sume de $t$ șocuri independente este $t\\sigma^2$; abaterea standard, nu varianța, crește ca $\\sqrt{t}$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Definition of a martingale",
                "text": "Which condition defines a martingale $\\{X_t\\}$ with respect to the information $\\mathcal{F}_t$?",
                "options": ["$E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$", "$X_{t+1} - X_t$ i.i.d. Normal", "$\\mathrm{Var}(X_{t+1} \\mid \\mathcal{F}_t)$ constant", "$\\mathrm{Corr}(X_t, X_{t+1}) = 0$"],
                "correctExplanation": "A martingale is a fair game: the best forecast of tomorrow, given today's information, is today's value.",
                "incorrectExplanation": "Only the conditional mean is restricted; the conditional variance may change over time, as in GARCH."
            },
            "ro": {
                "title": "Definiția unui martingal",
                "text": "Ce condiție definește un martingal $\\{X_t\\}$ în raport cu informația $\\mathcal{F}_t$?",
                "options": ["$E[X_{t+1} \\mid \\mathcal{F}_t] = X_t$", "$X_{t+1} - X_t$ i.i.d. Normale", "$\\mathrm{Var}(X_{t+1} \\mid \\mathcal{F}_t)$ constantă", "$\\mathrm{Corr}(X_t, X_{t+1}) = 0$"],
                "correctExplanation": "Un martingal este un joc echitabil: cea mai bună prognoză pentru mîine, dată fiind informația de azi, este valoarea de azi.",
                "incorrectExplanation": "Definiția impune o condiție doar asupra mediei condiționate; varianța condiționată se poate schimba în timp, ca în GARCH."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Variance of an AR(1)",
                "text": "$X_t = \\phi X_{t-1} + \\varepsilon_t$ with $\\phi = 0.8$ and white noise of variance 1. What is the stationary variance of $X_t$?",
                "options": ["0.8", "About 2.78", "1", "5"],
                "correctExplanation": "$\\mathrm{Var}(X_t) = \\sigma^2/(1 - \\phi^2) = 1/0.36 \\approx 2.78$.",
                "incorrectExplanation": "For a stationary AR(1), $\\mathrm{Var}(X_t) = \\sigma^2/(1 - \\phi^2) = 1/0.36 \\approx 2.78$; $1/(1 - \\phi) = 5$ is not the variance."
            },
            "ro": {
                "title": "Varianța unui AR(1)",
                "text": "$X_t = \\phi X_{t-1} + \\varepsilon_t$, cu $\\phi = 0{,}8$ și zgomot alb de varianță 1. Cît este varianța staționară a lui $X_t$?",
                "options": ["0,8", "Circa 2,78", "1", "5"],
                "correctExplanation": "$\\mathrm{Var}(X_t) = \\sigma^2/(1 - \\phi^2) = 1/0{,}36 \\approx 2{,}78$.",
                "incorrectExplanation": "Pentru un AR(1) staționar, $\\mathrm{Var}(X_t) = \\sigma^2/(1 - \\phi^2) = 1/0{,}36 \\approx 2{,}78$; $1/(1 - \\phi) = 5$ nu este varianța."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Wiener process moments",
                "text": "$W$ is a standard Wiener process. What is $\\mathrm{Cov}(W(1), W(3))$?",
                "options": ["1", "3", "0, because increments are independent", "2"],
                "correctExplanation": "$\\mathrm{Cov}(W(s), W(t)) = \\min(s, t) = 1$: $W(3) = W(1) + (W(3) - W(1))$ and the increment is independent of $W(1)$.",
                "incorrectExplanation": "The increments are independent, but $W(3)$ contains $W(1)$: $\\mathrm{Cov}(W(1), W(3)) = \\mathrm{Var}(W(1)) = 1$."
            },
            "ro": {
                "title": "Momentele procesului Wiener",
                "text": "$W$ este un proces Wiener standard. Cît este $\\mathrm{Cov}(W(1), W(3))$?",
                "options": ["1", "3", "0, pentru că creșterile sînt independente", "2"],
                "correctExplanation": "$\\mathrm{Cov}(W(s), W(t)) = \\min(s, t) = 1$: $W(3) = W(1) + (W(3) - W(1))$, iar creșterea este independentă de $W(1)$.",
                "incorrectExplanation": "Creșterile sînt independente, dar $W(3)$ îl conține pe $W(1)$: $\\mathrm{Cov}(W(1), W(3)) = \\mathrm{Var}(W(1)) = 1$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Median of GBM",
                "text": "GBM with $\\mu = 8\\%$, $\\sigma = 20\\%$ a year and $S_0 = 100$. What is the median of $S_1$?",
                "options": ["$100e^{0.08} \\approx 108.33$", "100", "$100e^{0.10} \\approx 110.52$", "$100e^{0.06} \\approx 106.18$"],
                "correctExplanation": "$\\ln S_1 \\sim N(\\ln 100 + \\mu - \\sigma^2/2, \\sigma^2)$, so the median is $100e^{0.08 - 0.02} \\approx 106.18$; $108.33$ is the mean.",
                "incorrectExplanation": "The median of a lognormal price is $S_0e^{(\\mu - \\sigma^2/2)T} \\approx 106.18$; $S_0e^{\\mu T}$ is the mean, pulled up by the right tail."
            },
            "ro": {
                "title": "Mediana GBM",
                "text": "GBM cu $\\mu = 8\\%$, $\\sigma = 20\\%$ pe an și $S_0 = 100$. Cît este mediana lui $S_1$?",
                "options": ["$100e^{0{,}08} \\approx 108{,}33$", "100", "$100e^{0{,}10} \\approx 110{,}52$", "$100e^{0{,}06} \\approx 106{,}18$"],
                "correctExplanation": "$\\ln S_1 \\sim N(\\ln 100 + \\mu - \\sigma^2/2, \\sigma^2)$, deci mediana este $100e^{0{,}08 - 0{,}02} \\approx 106{,}18$; $108{,}33$ este media.",
                "incorrectExplanation": "Mediana unui preț lognormal este $S_0e^{(\\mu - \\sigma^2/2)T} \\approx 106{,}18$; $S_0e^{\\mu T}$ este media, trasă în sus de coada din dreapta."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "What GBM misses",
                "text": "The S&P 500, 2000–2026, has excess kurtosis about 10.7 and $\\mathrm{Corr}(r_t^2, r_{t-1}^2) \\approx 0.31$; 300 GBM simulations of the same length stay within about $\\pm 0.1$ and $\\pm 0.02$. What does GBM miss?",
                "options": ["The average return", "The overall volatility", "Heavy tails and volatility clustering", "Nothing: the differences are sampling noise"],
                "correctExplanation": "GBM has i.i.d. Normal log returns: zero excess kurtosis and no autocorrelation of squared returns, far from the real values.",
                "incorrectExplanation": "The real values are far outside the GBM range: GBM cannot produce heavy tails or volatility clustering, although it matches the mean and volatility."
            },
            "ro": {
                "title": "Limitele modelului GBM",
                "text": "S&P 500, 2000–2026, are excesul de boltire circa 10,7 și $\\mathrm{Corr}(r_t^2, r_{t-1}^2) \\approx 0{,}31$; 300 de simulări GBM de aceeași lungime rămîn în intervalele de circa $\\pm 0{,}1$ și $\\pm 0{,}02$. Ce nu reproduce GBM?",
                "options": ["Randamentul mediu", "Volatilitatea totală", "Cozile groase și volatility clustering", "Nimic: diferențele sînt zgomot de eșantionare"],
                "correctExplanation": "GBM are randamente logaritmice Normale i.i.d.: exces de boltire zero și nicio autocorelație a pătratelor randamentelor, departe de valorile reale.",
                "incorrectExplanation": "Valorile reale sînt mult în afara intervalului GBM: GBM nu poate genera cozi groase sau volatility clustering, deși reproduce media și volatilitatea."
            }
        }
    ]
};
