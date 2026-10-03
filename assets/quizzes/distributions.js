// ============================================================
// Chapter 2 quiz bank: Classical distributions and stylised facts (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['distributions'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "4-sigma days under the Normal distribution",
                "text": "If daily returns were Normal, how often would a day beyond 4 standard deviations (either sign) occur?",
                "options": [
                    "About once a month",
                    "About once a year",
                    "About once every 15,800 trading days, roughly once in 63 years",
                    "About once every 370 trading days"
                ],
                "correctExplanation": "$P(|Z| > 4) \\approx 6.3 \\times 10^{-5}$, so one day in about 15,800, or once in about 63 years of 252 trading days.",
                "incorrectExplanation": "$P(|Z| > 4) \\approx 6.3 \\times 10^{-5}$: one day in about 15,800; once every 370 days is the 3-sigma frequency."
            },
            "ro": {
                "title": "Zile de 4 sigma conform distribuției Normale",
                "text": "Dacă randamentele zilnice ar fi Normale, cît de des ar apărea o zi aflată la peste 4 abateri standard (de orice semn)?",
                "options": [
                    "Cam o dată pe lună",
                    "Cam o dată pe an",
                    "Cam o dată la 15 800 de zile de tranzacționare, adică o dată la circa 63 de ani",
                    "Cam o dată la 370 de zile de tranzacționare"
                ],
                "correctExplanation": "$P(|Z| > 4) \\approx 6{,}3 \\times 10^{-5}$, deci o zi din circa 15 800, adică o dată la circa 63 de ani de cîte 252 de zile.",
                "incorrectExplanation": "$P(|Z| > 4) \\approx 6{,}3 \\times 10^{-5}$: o zi din circa 15 800; o dată la 370 de zile este frecvența zilelor de 3 sigma."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "4-sigma days in real data",
                "text": "The S&P 500 had about 4,200 daily returns in 2010–2026. How many 4-sigma days did it have, compared with the Normal expectation?",
                "options": [
                    "About 28 observed, against about 0.3 expected",
                    "About 0.3 observed, as expected",
                    "About 3 observed, against about 0.3 expected",
                    "None, as the Normal distribution predicts"
                ],
                "correctExplanation": "28 days beyond 4 standard deviations against 0.27 expected: about 100 times more than the Normal distribution allows.",
                "incorrectExplanation": "The count is about 28, roughly 100 times the Normal expectation of 0.27: this is the heavy-tails stylised fact."
            },
            "ro": {
                "title": "Zile de 4 sigma în datele reale",
                "text": "S&P 500 a avut circa 4 200 de randamente zilnice în 2010–2026. Cîte zile de 4 sigma a avut, față de numărul așteptat conform distribuției Normale?",
                "options": [
                    "Circa 28 observate, față de circa 0,3 așteptate",
                    "Circa 0,3 observate, cum era de așteptat",
                    "Circa 3 observate, față de circa 0,3 așteptate",
                    "Niciuna, cum prezice distribuția Normală"
                ],
                "correctExplanation": "28 de zile dincolo de 4 abateri standard, față de 0,27 așteptate: de circa 100 de ori mai multe decît permite distribuția Normală.",
                "incorrectExplanation": "Numărul este circa 28, de circa 100 de ori mai mare decît numărul așteptat conform distribuției Normale, 0,27: acesta este faptul stilizat al cozilor groase."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Mean of a lognormal variable",
                "text": "If $\\ln Y \\sim N(m, s^2)$, what is $E[Y]$?",
                "options": [
                    "$e^{m}$",
                    "$e^{m - s^2}$",
                    "$m + s^2/2$",
                    "$e^{m + s^2/2}$"
                ],
                "correctExplanation": "Completing the square gives $E[e^X] = e^{m + s^2/2}$ for $X \\sim N(m, s^2)$.",
                "incorrectExplanation": "$e^m$ is the median and $e^{m - s^2}$ the mode; the mean is $e^{m + s^2/2}$."
            },
            "ro": {
                "title": "Media unei variabile lognormale",
                "text": "Dacă $\\ln Y \\sim N(m, s^2)$, cît este $E[Y]$?",
                "options": [
                    "$e^{m}$",
                    "$e^{m - s^2}$",
                    "$m + s^2/2$",
                    "$e^{m + s^2/2}$"
                ],
                "correctExplanation": "Formînd pătratul perfect obținem $E[e^X] = e^{m + s^2/2}$ pentru $X \\sim N(m, s^2)$.",
                "incorrectExplanation": "$e^m$ este mediana, iar $e^{m - s^2}$ modul; media este $e^{m + s^2/2}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Median vs mean of lognormal prices",
                "text": "Yearly log returns are $N(7\\%, 18\\%^2)$. 100 lei are invested for one year. Which statement is correct?",
                "options": [
                    "The mean value is $100e^{0.07}$",
                    "The median value is $100e^{0.07}$ and the mean is higher, $100e^{0.07 + 0.18^2/2}$",
                    "The median and the mean are both $107$",
                    "The median is higher than the mean"
                ],
                "correctExplanation": "For lognormal values mode < median < mean; the gap $s^2/2$ is the volatility drag.",
                "incorrectExplanation": "$100e^{0.07} \\approx 107.25$ is the median; the mean is $100e^{0.0862} \\approx 109.00$."
            },
            "ro": {
                "title": "Mediana și media prețurilor lognormale",
                "text": "Randamentele logaritmice anuale sînt $N(7\\%, 18\\%^2)$. Se investesc 100 de lei pentru un an. Ce afirmație este corectă?",
                "options": [
                    "Valoarea medie este $100e^{0{,}07}$",
                    "Valoarea mediană este $100e^{0{,}07}$, iar media este mai mare, $100e^{0{,}07 + 0{,}18^2/2}$",
                    "Mediana și media sînt ambele $107$",
                    "Mediana este mai mare decît media"
                ],
                "correctExplanation": "Pentru valori lognormale, modul < mediana < media; diferența $s^2/2$ este volatility drag.",
                "incorrectExplanation": "$100e^{0{,}07} \\approx 107{,}25$ este mediana; media este $100e^{0{,}0862} \\approx 109{,}00$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Why lognormal prices",
                "text": "Why are prices modelled as lognormal rather than Normal?",
                "options": [
                    "Lognormal prices have thinner tails",
                    "A lognormal price is always positive, so the simple return is always above $-100\\%$",
                    "The lognormal distribution is symmetric",
                    "Prices are sums of independent shocks"
                ],
                "correctExplanation": "If $\\ln(P_T/P_0)$ is Normal, $P_T = P_0 e^{\\text{Normal}} > 0$: a holder cannot lose more than the amount invested.",
                "incorrectExplanation": "A Normal price could become negative; the lognormal model keeps prices positive and is right-skewed, not symmetric."
            },
            "ro": {
                "title": "De ce prețuri lognormale",
                "text": "De ce se modelează prețurile ca lognormale și nu ca Normale?",
                "options": [
                    "Prețurile lognormale au cozi mai subțiri",
                    "Un preț lognormal este întotdeauna pozitiv, deci randamentul simplu este întotdeauna peste $-100\\%$",
                    "Distribuția lognormală este simetrică",
                    "Prețurile sînt sume de șocuri independente"
                ],
                "correctExplanation": "Dacă $\\ln(P_T/P_0)$ este Normal, $P_T = P_0 e^{\\text{Normal}} > 0$: un investitor nu poate pierde mai mult decît a investit.",
                "incorrectExplanation": "Un preț cu distribuție Normală ar putea deveni negativ; modelul lognormal păstrează prețurile pozitive și este asimetric la dreapta, nu simetric."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Conditions of the CLT",
                "text": "Which set of conditions does the classical (Lindeberg–Lévy) central limit theorem require?",
                "options": [
                    "Normal terms only",
                    "Uncorrelated terms with any variance",
                    "Identically distributed terms with finite skewness",
                    "Independent, identically distributed terms with finite variance"
                ],
                "correctExplanation": "i.i.d. terms with finite variance: then $\\sqrt n(\\bar X - \\mu)/\\sigma \\to N(0, 1)$.",
                "incorrectExplanation": "The terms need not be Normal, but they must be i.i.d. with finite variance; uncorrelated is not enough."
            },
            "ro": {
                "title": "Condițiile CLT",
                "text": "Ce condiții cere teorema limită centrală clasică (Lindeberg–Lévy)?",
                "options": [
                    "Doar termeni Normali",
                    "Termeni necorelați, cu orice varianță",
                    "Termeni identic distribuiți, cu asimetrie finită",
                    "Termeni independenți, identic distribuiți, cu varianță finită"
                ],
                "correctExplanation": "Termeni i.i.d. cu varianță finită: atunci $\\sqrt n(\\bar X - \\mu)/\\sigma \\to N(0, 1)$.",
                "incorrectExplanation": "Termenii nu trebuie să fie Normali, dar trebuie să fie i.i.d. și cu varianță finită; simpla lipsă a corelației nu este suficientă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Normal approximation of the binomial",
                "text": "When is the Normal approximation of $B(n, p)$ good?",
                "options": [
                    "When $np(1-p)$ is large, at least about 9",
                    "Whenever $n \\ge 5$",
                    "Only when $p = 0.5$",
                    "When $p$ is close to 0"
                ],
                "correctExplanation": "The approximation works when the variance $np(1-p)$ is large; skewed cases ($p$ far from 0.5) need larger $n$.",
                "incorrectExplanation": "For $n = 20$, $p = 0.1$ the CDF error is about 0.04; the rule of thumb is $np(1-p) \\ge 9$."
            },
            "ro": {
                "title": "Aproximarea Normală a distribuției binomiale",
                "text": "Cînd este bună aproximarea Normală a distribuției $B(n, p)$?",
                "options": [
                    "Cînd $np(1-p)$ este mare, cel puțin circa 9",
                    "Ori de cîte ori $n \\ge 5$",
                    "Doar cînd $p = 0{,}5$",
                    "Cînd $p$ este aproape de 0"
                ],
                "correctExplanation": "Aproximarea funcționează cînd varianța $np(1-p)$ este mare; cazurile asimetrice ($p$ departe de 0,5) cer $n$ mai mare.",
                "incorrectExplanation": "Pentru $n = 20$, $p = 0{,}1$ eroarea CDF este circa 0,04; regula practică este $np(1-p) \\ge 9$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Kurtosis of the Normal distribution",
                "text": "What are the kurtosis and the excess kurtosis of the Normal distribution?",
                "options": [
                    "Kurtosis 0, excess kurtosis $-3$",
                    "Kurtosis 1, excess kurtosis 0",
                    "Kurtosis 3, excess kurtosis 0",
                    "Kurtosis 3, excess kurtosis 3"
                ],
                "correctExplanation": "$E[(X - \\mu)^4]/\\sigma^4 = 3$ for the Normal; the excess kurtosis subtracts 3.",
                "incorrectExplanation": "The Normal kurtosis is 3; excess kurtosis = kurtosis − 3 = 0."
            },
            "ro": {
                "title": "Boltirea distribuției Normale",
                "text": "Cît sînt boltirea și excesul de boltire ale distribuției Normale?",
                "options": [
                    "Boltirea 0, excesul $-3$",
                    "Boltirea 1, excesul 0",
                    "Boltirea 3, excesul 0",
                    "Boltirea 3, excesul 3"
                ],
                "correctExplanation": "$E[(X - \\mu)^4]/\\sigma^4 = 3$ pentru distribuția Normală; excesul de boltire se obține scăzînd 3.",
                "incorrectExplanation": "Boltirea Normală este 3; excesul = boltirea − 3 = 0."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Negative skewness",
                "text": "Daily returns of the BET have skewness about $-1$. What does this mean?",
                "options": [
                    "The mean return is negative",
                    "The distribution is flatter than the Normal",
                    "Gains are more frequent than losses",
                    "The left tail is longer: large losses are more extreme than large gains"
                ],
                "correctExplanation": "Negative skewness means a longer left tail: the largest falls are bigger than the largest rises.",
                "incorrectExplanation": "Skewness describes asymmetry of the tails, not the sign of the mean or the share of positive days."
            },
            "ro": {
                "title": "Asimetrie negativă",
                "text": "Randamentele zilnice ale BET au asimetria circa $-1$. Ce înseamnă acest lucru?",
                "options": [
                    "Randamentul mediu este negativ",
                    "Distribuția este mai turtită decît cea Normală",
                    "Cîștigurile sînt mai frecvente decît pierderile",
                    "Coada stîngă este mai lungă: pierderile mari sînt mai extreme decît cîștigurile mari"
                ],
                "correctExplanation": "Asimetria negativă înseamnă o coadă stîngă mai lungă: cele mai mari căderi sînt mai mari decît cele mai mari creșteri.",
                "incorrectExplanation": "Asimetria descrie lipsa de simetrie a cozilor, nu semnul mediei sau ponderea zilelor pozitive."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Jarque–Bera statistic",
                "text": "Which is the Jarque–Bera statistic and its distribution under normality?",
                "options": [
                    "$\\frac{n}{6}(S + K)$, approximately $N(0,1)$",
                    "$n(S^2 + K^2)$, approximately $\\chi^2(1)$",
                    "$\\frac{n}{6}(S^2 + K^2/4)$, approximately $\\chi^2(2)$",
                    "$\\frac{n}{24}(S^2 + K^2)$, approximately $t(2)$"
                ],
                "correctExplanation": "JB adds the squared standardised skewness $S^2/(6/n)$ and kurtosis $K^2/(24/n)$; under normality it is $\\chi^2(2)$, critical value 5.99 at 5%.",
                "incorrectExplanation": "JB $= \\frac{n}{6}(S^2 + K^2/4) \\sim \\chi^2(2)$: two squared standardised statistics."
            },
            "ro": {
                "title": "Statistica Jarque–Bera",
                "text": "Care este statistica Jarque–Bera și distribuția ei în ipoteza de normalitate?",
                "options": [
                    "$\\frac{n}{6}(S + K)$, aproximativ $N(0,1)$",
                    "$n(S^2 + K^2)$, aproximativ $\\chi^2(1)$",
                    "$\\frac{n}{6}(S^2 + K^2/4)$, aproximativ $\\chi^2(2)$",
                    "$\\frac{n}{24}(S^2 + K^2)$, aproximativ $t(2)$"
                ],
                "correctExplanation": "JB adună asimetria standardizată la pătrat, $S^2/(6/n)$, și excesul de boltire standardizat la pătrat, $K^2/(24/n)$; în ipoteza de normalitate este $\\chi^2(2)$, valoarea critică 5,99 la 5%.",
                "incorrectExplanation": "JB $= \\frac{n}{6}(S^2 + K^2/4) \\sim \\chi^2(2)$: două statistici standardizate la pătrat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "What a JB rejection means",
                "text": "JB rejects normality for S&P 500 returns with $p \\approx 0$. What can you conclude?",
                "options": [
                    "The returns follow a Student-t distribution",
                    "The returns are not Normal; the test does not say which distribution they follow",
                    "The returns are independent",
                    "The mean return is significantly different from zero"
                ],
                "correctExplanation": "A rejection only says that skewness and kurtosis are not those of a Normal distribution.",
                "incorrectExplanation": "JB tests normality only; the alternative distribution must be chosen and checked separately (QQ plots, AIC)."
            },
            "ro": {
                "title": "Interpretarea unei respingeri JB",
                "text": "JB respinge normalitatea randamentelor S&P 500 cu $p \\approx 0$. Ce puteți concluziona?",
                "options": [
                    "Randamentele urmează o distribuție Student-t",
                    "Randamentele nu sînt Normale; testul nu spune ce distribuție urmează",
                    "Randamentele sînt independente",
                    "Randamentul mediu diferă semnificativ de zero"
                ],
                "correctExplanation": "Respingerea arată doar că asimetria și boltirea nu sînt cele ale unei distribuții Normale.",
                "incorrectExplanation": "JB testează doar normalitatea; distribuția alternativă trebuie aleasă și verificată separat (QQ plots, AIC)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Precision of the kurtosis",
                "text": "For S&P 500 returns, $\\sqrt{24/n} \\approx 0.08$, but a bootstrap 95% interval for the excess kurtosis is about [6.7, 20.6]. Why?",
                "options": [
                    "$\\sqrt{24/n}$ assumes Normal data; with heavy tails the kurtosis depends on a few extreme days",
                    "The bootstrap is wrong for financial data",
                    "The sample is too small for the bootstrap",
                    "The kurtosis is not defined for returns"
                ],
                "correctExplanation": "The Normal-theory standard error is valid only under normality; heavy tails make the kurtosis very unstable.",
                "incorrectExplanation": "Resampling shows how much the kurtosis moves when a few extreme days are or are not drawn: far more than $\\sqrt{24/n}$."
            },
            "ro": {
                "title": "Precizia aplatizării",
                "text": "Pentru randamentele S&P 500, $\\sqrt{24/n} \\approx 0{,}08$, dar un interval bootstrap de 95% pentru excesul de boltire este circa [6,7; 20,6]. De ce?",
                "options": [
                    "$\\sqrt{24/n}$ presupune date Normale; cu cozi groase boltirea depinde de cîteva zile extreme",
                    "Bootstrap-ul este greșit pentru date financiare",
                    "Eșantionul este prea mic pentru bootstrap",
                    "Boltirea nu este definită pentru randamente"
                ],
                "correctExplanation": "Eroarea standard din teoria Normală este valabilă doar în ipoteza de normalitate; cozile groase fac boltirea foarte instabilă.",
                "incorrectExplanation": "Reeșantionarea arată cît variază boltirea cînd cîteva zile extreme sînt sau nu extrase: mult mai mult decît $\\sqrt{24/n}$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Reading a QQ plot",
                "text": "A QQ plot of standardised returns against $N(0,1)$ has an S-shape: the low end below the line, the high end above it. What does it show?",
                "options": [
                    "Heavier tails than the Normal distribution",
                    "Thinner tails than the Normal distribution",
                    "A perfect Normal fit",
                    "A mean different from zero"
                ],
                "correctExplanation": "Extreme empirical quantiles are farther out than the Normal ones at both ends: heavy tails.",
                "incorrectExplanation": "Points below the line at the low end and above it at the high end mean more extreme values than the model: heavy tails."
            },
            "ro": {
                "title": "Citirea unui QQ plot",
                "text": "Un QQ plot al randamentelor standardizate față de $N(0,1)$ are formă de S: capătul de jos sub dreaptă, cel de sus deasupra. Ce arată?",
                "options": [
                    "Cozi mai groase decît ale distribuției Normale",
                    "Cozi mai subțiri decît ale distribuției Normale",
                    "O ajustare perfectă la distribuția Normală",
                    "O medie diferită de zero"
                ],
                "correctExplanation": "Cuantilele empirice extreme sînt mai departe decît cele Normale la ambele capete: cozi groase.",
                "incorrectExplanation": "Puncte sub dreaptă la capătul de jos și deasupra ei la capătul de sus înseamnă valori mai extreme decît ale modelului: cozi groase."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Building a QQ plot",
                "text": "In a QQ plot against a model with CDF $F$, the $i$-th smallest observation $r_{(i)}$ is plotted against:",
                "options": [
                    "$F(r_{(i)})$",
                    "$F^{-1}\\big((i - 0.5)/n\\big)$",
                    "$i/n$",
                    "$F^{-1}(r_{(i)})$"
                ],
                "correctExplanation": "The model quantile at the plotting position $(i - 0.5)/n$ is paired with the $i$-th order statistic.",
                "incorrectExplanation": "A QQ plot compares quantiles: empirical $r_{(i)}$ against theoretical $F^{-1}((i - 0.5)/n)$."
            },
            "ro": {
                "title": "Construcția unui QQ plot",
                "text": "Într-un QQ plot față de un model cu CDF $F$, a $i$-a cea mai mică observație $r_{(i)}$ este reprezentată în funcție de:",
                "options": [
                    "$F(r_{(i)})$",
                    "$F^{-1}\\big((i - 0{,}5)/n\\big)$",
                    "$i/n$",
                    "$F^{-1}(r_{(i)})$"
                ],
                "correctExplanation": "Cuantila modelului de ordin $(i - 0{,}5)/n$ este asociată statisticii de ordine $i$.",
                "incorrectExplanation": "Un QQ plot compară cuantile: cuantila empirică $r_{(i)}$ față de cuantila teoretică $F^{-1}((i - 0{,}5)/n)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Variance of the Student-t",
                "text": "What is the variance of $T \\sim t(\\nu)$ for $\\nu > 2$?",
                "options": [
                    "$1$",
                    "$(\\nu - 2)/\\nu$",
                    "$\\nu/(\\nu - 2)$",
                    "$6/(\\nu - 4)$"
                ],
                "correctExplanation": "$\\text{Var}(T) = \\nu/(\\nu - 2)$; to get unit variance, use $T\\sqrt{(\\nu - 2)/\\nu}$.",
                "incorrectExplanation": "$6/(\\nu - 4)$ is the excess kurtosis; the variance is $\\nu/(\\nu - 2)$, larger than 1."
            },
            "ro": {
                "title": "Varianța distribuției Student-t",
                "text": "Cît este varianța lui $T \\sim t(\\nu)$ pentru $\\nu > 2$?",
                "options": [
                    "$1$",
                    "$(\\nu - 2)/\\nu$",
                    "$\\nu/(\\nu - 2)$",
                    "$6/(\\nu - 4)$"
                ],
                "correctExplanation": "$\\text{Var}(T) = \\nu/(\\nu - 2)$; pentru varianța 1 folosiți $T\\sqrt{(\\nu - 2)/\\nu}$.",
                "incorrectExplanation": "$6/(\\nu - 4)$ este excesul de boltire; varianța este $\\nu/(\\nu - 2)$, mai mare decît 1."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Excess kurtosis of the Student-t",
                "text": "For which $\\nu$ does a Student-t have excess kurtosis 6?",
                "options": [
                    "$\\nu = 6$",
                    "$\\nu = 10$",
                    "$\\nu = 4$",
                    "$\\nu = 5$"
                ],
                "correctExplanation": "$6/(\\nu - 4) = 6 \\Rightarrow \\nu = 5$.",
                "incorrectExplanation": "The excess kurtosis is $6/(\\nu - 4)$ for $\\nu > 4$; it equals 6 at $\\nu = 5$ and is infinite for $\\nu \\le 4$."
            },
            "ro": {
                "title": "Excesul de boltire al distribuției Student-t",
                "text": "Pentru ce $\\nu$ are o distribuție Student-t excesul de boltire 6?",
                "options": [
                    "$\\nu = 6$",
                    "$\\nu = 10$",
                    "$\\nu = 4$",
                    "$\\nu = 5$"
                ],
                "correctExplanation": "$6/(\\nu - 4) = 6 \\Rightarrow \\nu = 5$.",
                "incorrectExplanation": "Excesul de boltire este $6/(\\nu - 4)$ pentru $\\nu > 4$; este 6 la $\\nu = 5$ și infinit pentru $\\nu \\le 4$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Tails of the Student-t",
                "text": "How do the tails of a Student-t with $\\nu$ degrees of freedom decay?",
                "options": [
                    "Like $e^{-x^2/2}$, as for the Normal",
                    "Exponentially, like $e^{-x}$",
                    "Like a power, $P(|T| > x) \\approx c\\,x^{-\\nu}$; moments of order $\\nu$ and above are infinite",
                    "They do not decay: the t has bounded support"
                ],
                "correctExplanation": "Power-law tails: the probability of extreme values falls polynomially, much more slowly than the Normal tail.",
                "incorrectExplanation": "The Normal tail falls like $e^{-x^2/2}$; the t tail like $x^{-\\nu}$, so high moments do not exist."
            },
            "ro": {
                "title": "Cozile distribuției Student-t",
                "text": "Cum scad cozile unei distribuții Student-t cu $\\nu$ grade de libertate?",
                "options": [
                    "Ca $e^{-x^2/2}$, ca la distribuția Normală",
                    "Exponențial, ca $e^{-x}$",
                    "Ca o putere, $P(|T| > x) \\approx c\\,x^{-\\nu}$; momentele de ordin $\\nu$ și mai mare sînt infinite",
                    "Nu scad: distribuția t are suport mărginit"
                ],
                "correctExplanation": "Cozi de tip putere: probabilitatea valorilor extreme scade polinomial, mult mai lent decît coada Normală.",
                "incorrectExplanation": "Coada Normală scade ca $e^{-x^2/2}$; coada distribuției t scade ca $x^{-\\nu}$, deci momentele de ordin mare nu există."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Comparing fits with AIC",
                "text": "A Normal and a Student-t are fitted by maximum likelihood to the same returns. How do you use AIC $= 2k - 2\\ell$?",
                "options": [
                    "Prefer the model with the higher AIC",
                    "AIC can only compare models with the same number of parameters",
                    "Prefer the model with the higher $k$",
                    "Prefer the model with the lower AIC; $k$ penalises the extra parameter of the t"
                ],
                "correctExplanation": "AIC trades fit ($\\ell$) against complexity ($k$); the lower value wins. For daily returns the t wins by hundreds of units.",
                "incorrectExplanation": "Lower AIC is better; the penalty $2k$ makes the comparison fair between 2 (Normal) and 3 (t) parameters."
            },
            "ro": {
                "title": "Compararea ajustărilor cu AIC",
                "text": "O distribuție Normală și una Student-t sînt estimate prin verosimilitate maximă pe aceleași randamente. Cum folosiți AIC $= 2k - 2\\ell$?",
                "options": [
                    "Preferați modelul cu AIC mai mare",
                    "AIC poate compara doar modele cu același număr de parametri",
                    "Preferați modelul cu $k$ mai mare",
                    "Preferați modelul cu AIC mai mic; $k$ penalizează parametrul în plus al distribuției t"
                ],
                "correctExplanation": "AIC pune în balanță calitatea ajustării ($\\ell$) și complexitatea ($k$); se preferă valoarea mai mică. Pentru randamentele zilnice, distribuția t este preferată, cu diferențe de sute de unități sau mai mult.",
                "incorrectExplanation": "Se preferă AIC mai mic; penalizarea $2k$ face comparația corectă între 2 (Normală) și 3 (t) parametri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Uncorrelated but not independent",
                "text": "The ACF of daily returns is close to zero, but the ACF of absolute returns is about 0.3 and decays slowly. What follows?",
                "options": [
                    "Returns are independent",
                    "Returns are nearly uncorrelated but not independent: their size is predictable",
                    "Returns are predictable in sign",
                    "The data contain an error"
                ],
                "correctExplanation": "This is volatility clustering: the sign is hard to predict, the magnitude is not.",
                "incorrectExplanation": "Independence would make any function of returns uncorrelated, including $|r_t|$; the strong ACF of $|r_t|$ rules it out."
            },
            "ro": {
                "title": "Necorelate, dar nu independente",
                "text": "ACF a randamentelor zilnice este aproape de zero, dar ACF a randamentelor absolute este circa 0,3 și scade lent. Ce rezultă?",
                "options": [
                    "Randamentele sînt independente",
                    "Randamentele sînt aproape necorelate, dar nu independente: mărimea lor este previzibilă",
                    "Randamentele sînt previzibile ca semn",
                    "Datele conțin o eroare"
                ],
                "correctExplanation": "Este volatility clustering: semnul este greu de prevăzut, mărimea nu.",
                "incorrectExplanation": "Independența ar face necorelată orice funcție a randamentelor, inclusiv $|r_t|$; ACF puternică a lui $|r_t|$ o exclude."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Aggregational Gaussianity",
                "text": "What does aggregational Gaussianity mean?",
                "options": [
                    "Returns over longer horizons (weekly, monthly) are closer to Normal than daily returns",
                    "Daily returns are Normal",
                    "Prices are Normal at long horizons",
                    "The volatility of long-horizon returns is zero"
                ],
                "correctExplanation": "Summing daily log returns pushes the distribution toward the Normal (CLT), slowly because of volatility clustering.",
                "incorrectExplanation": "It concerns the shape of $h$-day returns: the excess kurtosis falls as $h$ grows, though monthly returns are still not exactly Normal."
            },
            "ro": {
                "title": "Gaussianitatea agregată",
                "text": "Ce înseamnă gaussianitatea agregată (aggregational Gaussianity)?",
                "options": [
                    "Randamentele pe orizonturi mai lungi (săptămînale, lunare) sînt mai apropiate de distribuția Normală decît cele zilnice",
                    "Randamentele zilnice sînt Normale",
                    "Prețurile sînt Normale pe orizonturi lungi",
                    "Volatilitatea randamentelor pe orizonturi lungi este zero"
                ],
                "correctExplanation": "Adunarea randamentelor logaritmice zilnice apropie distribuția de cea Normală (CLT), dar lent, din cauza volatility clustering.",
                "incorrectExplanation": "Se referă la forma randamentelor pe $h$ zile: excesul de boltire scade cînd $h$ crește, deși randamentele lunare tot nu sînt exact Normale."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The leverage effect",
                "text": "Which measurement shows the leverage effect?",
                "options": [
                    "$\\text{Corr}(r_t, r_{t+1}) > 0$",
                    "$\\text{Corr}(|r_t|, |r_{t+k}|) > 0$",
                    "A positive skewness",
                    "$\\text{Corr}(r_t, |r_{t+k}|) < 0$: falls are followed by higher volatility than rises"
                ],
                "correctExplanation": "Negative correlation between today's return and future absolute returns: for the S&P 500, DAX and BET about $-0.1$.",
                "incorrectExplanation": "$\\text{Corr}(|r_t|, |r_{t+k}|) > 0$ is volatility clustering; the leverage effect needs the sign of today's return."
            },
            "ro": {
                "title": "Efectul de levier",
                "text": "Ce măsură evidențiază efectul de levier (leverage effect)?",
                "options": [
                    "$\\text{Corr}(r_t, r_{t+1}) > 0$",
                    "$\\text{Corr}(|r_t|, |r_{t+k}|) > 0$",
                    "O asimetrie pozitivă",
                    "$\\text{Corr}(r_t, |r_{t+k}|) < 0$: scăderile sînt urmate de o volatilitate mai mare decît creșterile"
                ],
                "correctExplanation": "Corelație negativă între randamentul de azi și randamentele absolute viitoare: pentru S&P 500, DAX și BET circa $-0{,}1$.",
                "incorrectExplanation": "$\\text{Corr}(|r_t|, |r_{t+k}|) > 0$ este volatility clustering; efectul de levier implică semnul randamentului de azi."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Bitcoin and the stylised facts",
                "text": "Which stylised fact of stock indices is weak or absent for Bitcoin in 2014–2026?",
                "options": [
                    "The leverage effect",
                    "Heavy tails",
                    "Volatility clustering",
                    "Absence of linear autocorrelation"
                ],
                "correctExplanation": "Bitcoin has heavy tails and clustering, but $\\text{Corr}(r_t, |r_{t+k}|)$ is near zero after lag 1: no firm with debt stands behind it.",
                "incorrectExplanation": "Heavy tails, clustering and near-zero autocorrelation all hold for Bitcoin; the leverage correlation is close to the noise band."
            },
            "ro": {
                "title": "Bitcoin și faptele stilizate",
                "text": "Ce fapt stilizat al indicilor bursieri este slab sau absent la Bitcoin în 2014–2026?",
                "options": [
                    "Efectul de levier",
                    "Cozile groase",
                    "Volatility clustering",
                    "Absența autocorelației liniare"
                ],
                "correctExplanation": "Bitcoin are cozi groase și volatility clustering, dar $\\text{Corr}(r_t, |r_{t+k}|)$ este aproape zero după decalajul 1: în spatele lui nu există o firmă cu datorii.",
                "incorrectExplanation": "Cozile groase, volatility clustering și autocorelația aproape nulă sînt valabile pentru Bitcoin; corelația de levier este aproape de marginea benzii de încredere."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Ljung–Box test",
                "text": "What is the distribution of the Ljung–Box statistic $Q(m)$ under the null hypothesis of no autocorrelation up to lag $m$?",
                "options": [
                    "$N(0, 1)$",
                    "$\\chi^2(2)$",
                    "$\\chi^2(m)$",
                    "$t(m)$"
                ],
                "correctExplanation": "$Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n-h)$ is approximately $\\chi^2(m)$; for $m = 10$ the 5% critical value is 18.3.",
                "incorrectExplanation": "Each of the $m$ squared autocorrelations adds one degree of freedom: $\\chi^2(m)$."
            },
            "ro": {
                "title": "Testul Ljung–Box",
                "text": "Care este distribuția statisticii Ljung–Box $Q(m)$ în ipoteza nulă a absenței autocorelației pînă la decalajul $m$?",
                "options": [
                    "$N(0, 1)$",
                    "$\\chi^2(2)$",
                    "$\\chi^2(m)$",
                    "$t(m)$"
                ],
                "correctExplanation": "$Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n-h)$ este aproximativ $\\chi^2(m)$; pentru $m = 10$ valoarea critică de 5% este 18,3.",
                "incorrectExplanation": "Fiecare dintre cele $m$ autocorelații la pătrat adaugă un grad de libertate: $\\chi^2(m)$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Check the data before the tails",
                "text": "A stock's adjusted close gives returns of $-15\\%$ and $+20.5\\%$ on two consecutive days, on the ex-date of a bonus share issue. What should you do first?",
                "options": [
                    "Keep them: they prove heavy tails",
                    "Check the corporate action: a misplaced adjustment creates two fake extreme returns that inflate the kurtosis",
                    "Replace them with zeros without checking",
                    "Use the raw close instead, which has no jumps"
                ],
                "correctExplanation": "A pair of opposite jumps around a corporate action is a typical adjustment error; for TLV in May 2016 it raises the excess kurtosis from 13.5 to 17.5.",
                "incorrectExplanation": "Extreme returns must be checked against corporate actions and news before computing tail statistics; the raw close jumps even more."
            },
            "ro": {
                "title": "Verificarea datelor înaintea analizei cozilor",
                "text": "Prețul ajustat al unei acțiuni dă randamente de $-15\\%$ și $+20{,}5\\%$ în două zile consecutive, la data ex a unei distribuiri de acțiuni gratuite. Care este primul pas?",
                "options": [
                    "Le păstrați: dovedesc cozile groase",
                    "Verificați evenimentul corporativ: o ajustare plasată greșit creează două randamente extreme false care umflă boltirea",
                    "Le înlocuiți cu zero fără verificare",
                    "Folosiți în schimb prețul de închidere brut, care nu are salturi"
                ],
                "correctExplanation": "O pereche de salturi de semn opus în jurul unui eveniment corporativ este o eroare tipică de ajustare; pentru TLV în mai 2016 ea crește excesul de boltire de la 13,5 la 17,5.",
                "incorrectExplanation": "Randamentele extreme se verifică față de evenimentele corporative și știri înainte de a calcula statisticile cozilor; prețul brut sare și mai mult."
            }
        }
    ]
};
