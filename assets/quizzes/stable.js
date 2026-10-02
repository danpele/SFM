// ============================================================
// Chapter 3 quiz bank: α-stable distributions (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['stable'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Stability under summation",
                "text": "A distribution is called stable when, for i.i.d. copies $X_1, X_2$ and constants $a, b > 0$, which property holds?",
                "options": [
                    "$aX_1 + bX_2$ always has variance $a^2 + b^2$",
                    "$X_1 + X_2$ is independent of $X_1$",
                    "$aX_1 + bX_2$ has the same distribution as $cX + d$ for some $c > 0$ and $d$",
                    "$aX_1 + bX_2$ is always Normally distributed"
                ],
                "correctExplanation": "Stability means that a linear combination of i.i.d. copies keeps the same shape, up to a change of scale $c$ and location $d$.",
                "incorrectExplanation": "Stability means $aX_1 + bX_2 \\overset{d}{=} cX + d$: the sum keeps the shape of the distribution, only scale and location change."
            },
            "ro": {
                "title": "Stabilitatea la adunare",
                "text": "O distribuție se numește stabilă dacă, pentru copii i.i.d. $X_1, X_2$ și constante $a, b > 0$, are loc ce proprietate?",
                "options": [
                    "$aX_1 + bX_2$ are întotdeauna varianța $a^2 + b^2$",
                    "$X_1 + X_2$ este independentă de $X_1$",
                    "$aX_1 + bX_2$ are aceeași distribuție ca $cX + d$ pentru un $c > 0$ și un $d$",
                    "$aX_1 + bX_2$ urmează întotdeauna distribuția Normală"
                ],
                "correctExplanation": "Stabilitatea înseamnă că o combinație liniară de copii i.i.d. își păstrează forma, cu o schimbare de scală $c$ și de poziție $d$.",
                "incorrectExplanation": "Stabilitatea înseamnă $aX_1 + bX_2 \\overset{d}{=} cX + d$: suma păstrează forma distribuției, se schimbă doar scala și poziția."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Scale of a sum",
                "text": "$X_1, X_2$ are i.i.d. symmetric stable with $\\alpha = 1.5$ and scale $\\gamma = 1$. What is the scale $c$ of $X_1 + X_2$?",
                "options": [
                    "$c = 2^{2/3} \\approx 1.587$",
                    "$c = 2$",
                    "$c = \\sqrt{2} \\approx 1.414$",
                    "$c = 2^{1.5} \\approx 2.828$"
                ],
                "correctExplanation": "From $c^\\alpha = a^\\alpha + b^\\alpha$ with $a = b = 1$: $c = 2^{1/1.5} = 2^{2/3} \\approx 1.587$.",
                "incorrectExplanation": "The rule is $c^\\alpha = a^\\alpha + b^\\alpha$; with $a = b = 1$ and $\\alpha = 1.5$, $c = 2^{2/3} \\approx 1.587$. $\\sqrt{2}$ is the Normal case $\\alpha = 2$."
            },
            "ro": {
                "title": "Scala unei sume",
                "text": "$X_1, X_2$ sînt i.i.d. stabile simetrice cu $\\alpha = 1{,}5$ și scala $\\gamma = 1$. Care este scala $c$ a sumei $X_1 + X_2$?",
                "options": [
                    "$c = 2^{2/3} \\approx 1{,}587$",
                    "$c = 2$",
                    "$c = \\sqrt{2} \\approx 1{,}414$",
                    "$c = 2^{1{,}5} \\approx 2{,}828$"
                ],
                "correctExplanation": "Din $c^\\alpha = a^\\alpha + b^\\alpha$ cu $a = b = 1$: $c = 2^{1/1{,}5} = 2^{2/3} \\approx 1{,}587$.",
                "incorrectExplanation": "Regula este $c^\\alpha = a^\\alpha + b^\\alpha$; cu $a = b = 1$ și $\\alpha = 1{,}5$, $c = 2^{2/3} \\approx 1{,}587$. $\\sqrt{2}$ corespunde cazului Normal, $\\alpha = 2$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Aggregation over n days",
                "text": "Daily returns are i.i.d. symmetric $\\alpha$-stable. How does the scale of the sum of $n$ daily returns grow with $n$?",
                "options": [
                    "As $\\sqrt{n}$, whatever $\\alpha$",
                    "As $n$, whatever $\\alpha$",
                    "It does not change with $n$",
                    "As $n^{1/\\alpha}$, faster than $\\sqrt{n}$ when $\\alpha < 2$"
                ],
                "correctExplanation": "Repeating $c^\\alpha = a^\\alpha + b^\\alpha$ gives a scale of $n^{1/\\alpha}\\gamma$; for $\\alpha < 2$, $1/\\alpha > 1/2$.",
                "incorrectExplanation": "The scale of the sum is $n^{1/\\alpha}\\gamma$; the square-root rule holds only for $\\alpha = 2$, the Normal case."
            },
            "ro": {
                "title": "Agregarea pe n zile",
                "text": "Randamentele zilnice sînt i.i.d. $\\alpha$-stabile simetrice. Cum crește cu $n$ scala sumei a $n$ randamente zilnice?",
                "options": [
                    "Ca $\\sqrt{n}$, oricare ar fi $\\alpha$",
                    "Ca $n$, oricare ar fi $\\alpha$",
                    "Nu se schimbă cu $n$",
                    "Ca $n^{1/\\alpha}$, mai repede decît $\\sqrt{n}$ cînd $\\alpha < 2$"
                ],
                "correctExplanation": "Aplicînd repetat $c^\\alpha = a^\\alpha + b^\\alpha$ obținem scala $n^{1/\\alpha}\\gamma$; pentru $\\alpha < 2$, $1/\\alpha > 1/2$.",
                "incorrectExplanation": "Scala sumei este $n^{1/\\alpha}\\gamma$; regula rădăcinii pătrate este valabilă doar pentru $\\alpha = 2$, cazul Normal."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Generalised CLT",
                "text": "What does the generalised central limit theorem say about normalised sums of i.i.d. variables?",
                "options": [
                    "They always converge to the Normal distribution",
                    "If they converge to a non-degenerate limit, that limit is a stable distribution",
                    "They converge only if the variance is finite",
                    "They converge to a Student-t distribution with $n - 1$ degrees of freedom"
                ],
                "correctExplanation": "Stable laws are the only possible limits of normalised sums; the Normal distribution is the case $\\alpha = 2$, reached when the variance is finite.",
                "incorrectExplanation": "The only possible limits of normalised sums $(S_n - b_n)/a_n$ are stable laws; with infinite variance and power tails the limit has $\\alpha < 2$."
            },
            "ro": {
                "title": "Teorema limită centrală generalizată",
                "text": "Ce spune teorema limită centrală generalizată despre sumele normalizate de variabile i.i.d.?",
                "options": [
                    "Converg întotdeauna către distribuția Normală",
                    "Dacă au o limită nedegenerată, aceasta este o distribuție stabilă",
                    "Converg doar dacă varianța este finită",
                    "Converg către o distribuție Student-t cu $n - 1$ grade de libertate"
                ],
                "correctExplanation": "Legile stabile sînt singurele limite posibile ale sumelor normalizate; distribuția Normală este cazul $\\alpha = 2$, atins cînd varianța este finită.",
                "incorrectExplanation": "Singurele limite posibile ale sumelor normalizate $(S_n - b_n)/a_n$ sînt legile stabile; cu varianță infinită și cozi de tip putere, limita are $\\alpha < 2$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Domain of attraction",
                "text": "i.i.d. variables have power tails $P(|X| > x) \\sim C x^{-1.5}$. Their normalised sums converge to which law?",
                "options": [
                    "The Normal distribution",
                    "A stable law with $\\alpha = 1.5$",
                    "A stable law with $\\alpha = 3$",
                    "A Cauchy distribution, whatever the tail"
                ],
                "correctExplanation": "Power tails with exponent $\\alpha < 2$ put the variable in the domain of attraction of an $\\alpha$-stable law with the same $\\alpha$.",
                "incorrectExplanation": "The variance is infinite (tail exponent $1.5 < 2$), so the classical CLT does not apply; the limit is stable with $\\alpha = 1.5$."
            },
            "ro": {
                "title": "Domeniul de atracție",
                "text": "Variabile i.i.d. au cozi de tip putere $P(|X| > x) \\sim C x^{-1{,}5}$. Către ce lege converg sumele lor normalizate?",
                "options": [
                    "Distribuția Normală",
                    "O lege stabilă cu $\\alpha = 1{,}5$",
                    "O lege stabilă cu $\\alpha = 3$",
                    "O distribuție Cauchy, oricare ar fi coada"
                ],
                "correctExplanation": "Cozile de tip putere cu exponent $\\alpha < 2$ pun variabila în domeniul de atracție al unei legi $\\alpha$-stabile cu același $\\alpha$.",
                "incorrectExplanation": "Varianța este infinită (exponentul cozii $1{,}5 < 2$), deci teorema limită centrală clasică nu se aplică; limita este stabilă cu $\\alpha = 1{,}5$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Role of beta",
                "text": "What does the parameter $\\beta$ of a stable distribution control?",
                "options": [
                    "The thickness of both tails",
                    "The width of the distribution",
                    "The position of the centre",
                    "The skewness: $\\beta < 0$ gives a heavier left tail"
                ],
                "correctExplanation": "$\\beta \\in [-1, 1]$ is the skewness parameter; $\\beta = 0$ gives a symmetric law, $\\beta < 0$ a heavier left tail.",
                "incorrectExplanation": "Tail thickness is $\\alpha$, width is $\\gamma$, position is $\\delta$; $\\beta$ controls the skewness."
            },
            "ro": {
                "title": "Rolul lui beta",
                "text": "Ce controlează parametrul $\\beta$ al unei distribuții stabile?",
                "options": [
                    "Grosimea ambelor cozi",
                    "Lățimea distribuției",
                    "Poziția centrului",
                    "Asimetria: $\\beta < 0$ dă o coadă stîngă mai groasă"
                ],
                "correctExplanation": "$\\beta \\in [-1, 1]$ este parametrul de asimetrie; $\\beta = 0$ dă o lege simetrică, $\\beta < 0$ o coadă stîngă mai groasă.",
                "incorrectExplanation": "Grosimea cozilor este dată de $\\alpha$, lățimea de $\\gamma$, poziția de $\\delta$; $\\beta$ controlează asimetria."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Normal special case",
                "text": "A stable law with $\\alpha = 2$ and scale $\\gamma = 1$ is a Normal distribution. What is its variance?",
                "options": [
                    "2",
                    "1",
                    "$\\sqrt{2}$",
                    "Infinite"
                ],
                "correctExplanation": "For $\\alpha = 2$ the characteristic function is $\\exp(-\\gamma^2 t^2 + i\\delta t)$, so the variance is $2\\gamma^2 = 2$.",
                "incorrectExplanation": "With $\\alpha = 2$ the law is $N(\\delta, 2\\gamma^2)$: the standard deviation is $\\sqrt{2}\\gamma$, not $\\gamma$, so the variance is 2."
            },
            "ro": {
                "title": "Cazul Normal",
                "text": "O lege stabilă cu $\\alpha = 2$ și scala $\\gamma = 1$ este o distribuție Normală. Care este varianța ei?",
                "options": [
                    "2",
                    "1",
                    "$\\sqrt{2}$",
                    "Infinită"
                ],
                "correctExplanation": "Pentru $\\alpha = 2$ funcția caracteristică este $\\exp(-\\gamma^2 t^2 + i\\delta t)$, deci varianța este $2\\gamma^2 = 2$.",
                "incorrectExplanation": "Cu $\\alpha = 2$ legea este $N(\\delta, 2\\gamma^2)$: abaterea standard este $\\sqrt{2}\\gamma$, nu $\\gamma$, deci varianța este 2."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Special cases",
                "text": "Which pair is a correct special case of the stable family?",
                "options": [
                    "$\\alpha = 1$, $\\beta = 0$: the Normal distribution",
                    "$\\alpha = 1/2$, $\\beta = 0$: the Cauchy distribution",
                    "$\\alpha = 1$, $\\beta = 0$: the Cauchy distribution",
                    "$\\alpha = 2$, $\\beta = 1$: the Lévy distribution"
                ],
                "correctExplanation": "The three stable laws with a closed-form density: Normal ($\\alpha = 2$), Cauchy ($\\alpha = 1$, $\\beta = 0$), Lévy ($\\alpha = 1/2$, $\\beta = 1$).",
                "incorrectExplanation": "Cauchy is $\\alpha = 1$, $\\beta = 0$; the Normal distribution is $\\alpha = 2$; the Lévy distribution is $\\alpha = 1/2$, $\\beta = 1$."
            },
            "ro": {
                "title": "Cazuri particulare",
                "text": "Care pereche este un caz particular corect al familiei stabile?",
                "options": [
                    "$\\alpha = 1$, $\\beta = 0$: distribuția Normală",
                    "$\\alpha = 1/2$, $\\beta = 0$: distribuția Cauchy",
                    "$\\alpha = 1$, $\\beta = 0$: distribuția Cauchy",
                    "$\\alpha = 2$, $\\beta = 1$: distribuția Lévy"
                ],
                "correctExplanation": "Cele trei legi stabile cu densitate în formă închisă: Normală ($\\alpha = 2$), Cauchy ($\\alpha = 1$, $\\beta = 0$), Lévy ($\\alpha = 1/2$, $\\beta = 1$).",
                "incorrectExplanation": "Cauchy înseamnă $\\alpha = 1$, $\\beta = 0$; distribuția Normală are $\\alpha = 2$; distribuția Lévy are $\\alpha = 1/2$, $\\beta = 1$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "S0 versus S1",
                "text": "In Nolan's notation, how do the S0 and S1 parameterisations differ (for $\\alpha \\neq 1$)?",
                "options": [
                    "Only the scale: $\\gamma_0 = \\sqrt{2}\\,\\gamma_1$",
                    "The sign of $\\beta$ is reversed",
                    "S1 allows $\\alpha > 2$",
                    "Only the location: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$"
                ],
                "correctExplanation": "$\\alpha$, $\\beta$ and $\\gamma$ are the same; only the location shifts by $\\beta\\gamma\\tan(\\pi\\alpha/2)$.",
                "incorrectExplanation": "The two parameterisations share $\\alpha$, $\\beta$, $\\gamma$; the location differs by $\\beta\\gamma\\tan(\\pi\\alpha/2)$, which is zero for symmetric laws."
            },
            "ro": {
                "title": "S0 față de S1",
                "text": "În notația lui Nolan, prin ce diferă parametrizările S0 și S1 (pentru $\\alpha \\neq 1$)?",
                "options": [
                    "Doar prin scală: $\\gamma_0 = \\sqrt{2}\\,\\gamma_1$",
                    "Semnul lui $\\beta$ este inversat",
                    "S1 permite $\\alpha > 2$",
                    "Doar prin parametrul de poziție: $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$"
                ],
                "correctExplanation": "$\\alpha$, $\\beta$ și $\\gamma$ sînt aceiași; doar parametrul de poziție se deplasează cu $\\beta\\gamma\\tan(\\pi\\alpha/2)$.",
                "incorrectExplanation": "Cele două parametrizări au aceiași $\\alpha$, $\\beta$, $\\gamma$; parametrul de poziție diferă cu $\\beta\\gamma\\tan(\\pi\\alpha/2)$, care este zero pentru legile simetrice."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "scipy parameterisation",
                "text": "You fit `scipy.stats.levy_stable` with its default settings and want to report Nolan's S0 location. What must you do?",
                "options": [
                    "Nothing: the default is already S0",
                    "Convert: the default is S1, so add $\\beta\\gamma\\tan(\\pi\\alpha/2)$ to the fitted location",
                    "Divide the scale by $\\sqrt{2}$",
                    "Change the sign of $\\alpha$"
                ],
                "correctExplanation": "The default `levy_stable.parameterization` is `'S1'`; either set it to `'S0'` or convert $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$.",
                "incorrectExplanation": "scipy uses S1 by default; the conversion changes only the location, $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$."
            },
            "ro": {
                "title": "Parametrizarea din scipy",
                "text": "Estimați cu `scipy.stats.levy_stable` cu setările implicite și vreți să raportați parametrul de poziție S0 al lui Nolan. Ce trebuie să faceți?",
                "options": [
                    "Nimic: implicit este deja S0",
                    "Convertiți: implicit este S1, deci adunați $\\beta\\gamma\\tan(\\pi\\alpha/2)$ la poziția estimată",
                    "Împărțiți scala la $\\sqrt{2}$",
                    "Schimbați semnul lui $\\alpha$"
                ],
                "correctExplanation": "Valoarea implicită `levy_stable.parameterization` este `'S1'`; fie o setați la `'S0'`, fie convertiți $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$.",
                "incorrectExplanation": "scipy folosește implicit S1; conversia schimbă doar parametrul de poziție, $\\delta_0 = \\delta_1 + \\beta\\gamma\\tan(\\pi\\alpha/2)$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Tail decay",
                "text": "For a stable law with $\\alpha < 2$, how does $P(X > x)$ behave for large $x$ (with $\\beta > -1$)?",
                "options": [
                    "Like $c\\,x^{-\\alpha}$, a power law",
                    "Like $e^{-x^2/2}$",
                    "Like $e^{-x}$",
                    "It is exactly zero beyond some $x$"
                ],
                "correctExplanation": "Stable tails are Pareto-type: $P(X > x) \\sim c_\\alpha (1 + \\beta)\\gamma^\\alpha x^{-\\alpha}$, much slower than the Normal tail.",
                "incorrectExplanation": "The tail is a power law $x^{-\\alpha}$; on a log-log plot of the survival function it is a straight line with slope $-\\alpha$."
            },
            "ro": {
                "title": "Descreșterea cozii",
                "text": "Pentru o lege stabilă cu $\\alpha < 2$, cum se comportă $P(X > x)$ pentru $x$ mare (cu $\\beta > -1$)?",
                "options": [
                    "Ca $c\\,x^{-\\alpha}$, o lege de tip putere",
                    "Ca $e^{-x^2/2}$",
                    "Ca $e^{-x}$",
                    "Este exact zero după un anumit $x$"
                ],
                "correctExplanation": "Cozile stabile sînt de tip Pareto: $P(X > x) \\sim c_\\alpha (1 + \\beta)\\gamma^\\alpha x^{-\\alpha}$, mult mai lent decît coada distribuției Normale.",
                "incorrectExplanation": "Coada este o lege de tip putere $x^{-\\alpha}$; pe un grafic log-log al funcției de supraviețuire apare ca o dreaptă cu panta $-\\alpha$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Which moments exist",
                "text": "$X$ is stable with $\\alpha = 1.7$. Which statement about its moments is correct?",
                "options": [
                    "Both the mean and the variance are finite",
                    "Neither the mean nor the variance exists",
                    "The mean exists, the variance is infinite",
                    "The variance exists but the kurtosis does not"
                ],
                "correctExplanation": "$E|X|^p < \\infty$ if and only if $p < \\alpha$: $p = 1 < 1.7$ is finite, $p = 2 > 1.7$ is infinite.",
                "incorrectExplanation": "The rule $E|X|^p < \\infty \\iff p < \\alpha$ gives a finite mean ($1 < 1.7$) and an infinite variance ($2 > 1.7$); kurtosis does not exist either."
            },
            "ro": {
                "title": "Ce momente există",
                "text": "$X$ este stabilă cu $\\alpha = 1{,}7$. Care afirmație despre momentele ei este corectă?",
                "options": [
                    "Și media, și varianța sînt finite",
                    "Nu există nici media, nici varianța",
                    "Media există, varianța este infinită",
                    "Varianța există, dar aplatizarea nu"
                ],
                "correctExplanation": "$E|X|^p < \\infty$ dacă și numai dacă $p < \\alpha$: $p = 1 < 1{,}7$ este finit, $p = 2 > 1{,}7$ este infinit.",
                "incorrectExplanation": "Regula $E|X|^p < \\infty \\iff p < \\alpha$ dă o medie finită ($1 < 1{,}7$) și o varianță infinită ($2 > 1{,}7$); nici aplatizarea nu există."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Sample variance with infinite variance",
                "text": "You compute the sample variance of a growing sample from a stable law with $\\alpha = 1.6$. What do you see?",
                "options": [
                    "It does not settle: it keeps jumping up when a new extreme value arrives",
                    "It converges quickly to $2\\gamma^2$",
                    "It decreases to zero",
                    "It converges to $\\alpha$"
                ],
                "correctExplanation": "The true variance is infinite, so the sample variance has no limit; each large observation produces a jump.",
                "incorrectExplanation": "With $\\alpha < 2$ there is no finite variance to converge to; $2\\gamma^2$ is the variance only in the Normal case $\\alpha = 2$."
            },
            "ro": {
                "title": "Varianța de selecție cînd varianța este infinită",
                "text": "Calculați varianța de selecție pe un eșantion tot mai mare dintr-o lege stabilă cu $\\alpha = 1{,}6$. Ce observați?",
                "options": [
                    "Nu se stabilizează: sare în sus de fiecare dată cînd apare o valoare extremă nouă",
                    "Converge rapid la $2\\gamma^2$",
                    "Scade spre zero",
                    "Converge la $\\alpha$"
                ],
                "correctExplanation": "Varianța teoretică este infinită, deci varianța de selecție nu are limită; fiecare observație mare produce un salt.",
                "incorrectExplanation": "Cu $\\alpha < 2$ nu există o varianță finită către care să conveargă; $2\\gamma^2$ este varianța doar în cazul Normal, $\\alpha = 2$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Chambers–Mallows–Stuck",
                "text": "Which inputs does the Chambers–Mallows–Stuck method use to simulate one stable draw?",
                "options": [
                    "Two independent Normal variables",
                    "A Student-t variable and a Poisson variable",
                    "The inverse of a closed-form stable distribution function",
                    "A uniform angle on $(-\\pi/2, \\pi/2)$ and an independent exponential variable with mean 1"
                ],
                "correctExplanation": "CMS transforms $V \\sim U(-\\pi/2, \\pi/2)$ and $W \\sim \\text{Exp}(1)$ into an exact stable draw, with no need for the density.",
                "incorrectExplanation": "Stable distribution functions have no closed form, so inversion is not available; CMS uses a uniform angle and an exponential variable."
            },
            "ro": {
                "title": "Chambers–Mallows–Stuck",
                "text": "Ce variabile folosește metoda Chambers–Mallows–Stuck pentru a simula o valoare stabilă?",
                "options": [
                    "Două variabile Normale independente",
                    "O variabilă Student-t și una Poisson",
                    "Inversa unei funcții de repartiție stabile în formă închisă",
                    "Un unghi uniform pe $(-\\pi/2, \\pi/2)$ și o variabilă exponențială independentă cu media 1"
                ],
                "correctExplanation": "CMS transformă $V \\sim U(-\\pi/2, \\pi/2)$ și $W \\sim \\text{Exp}(1)$ într-o valoare stabilă exactă, fără să aibă nevoie de densitate.",
                "incorrectExplanation": "Funcțiile de repartiție stabile nu au formă închisă, deci inversarea nu este disponibilă; CMS folosește un unghi uniform și o variabilă exponențială."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "McCulloch's ratio",
                "text": "McCulloch's estimator uses $\\nu_\\alpha = (q_{0.95} - q_{0.05})/(q_{0.75} - q_{0.25})$. What value does $\\nu_\\alpha$ take for the Normal distribution?",
                "options": [
                    "About 1.0",
                    "About 6.31",
                    "About 2.439",
                    "It is infinite"
                ],
                "correctExplanation": "For the Normal distribution $\\nu_\\alpha = (2 \\times 1.645)/(2 \\times 0.674) \\approx 2.439$; larger values mean heavier tails and $\\alpha < 2$.",
                "incorrectExplanation": "$\\nu_\\alpha = 1.645/0.674 \\approx 2.439$ for the Normal distribution; 6.31 is the Cauchy value; heavier tails raise $\\nu_\\alpha$."
            },
            "ro": {
                "title": "Raportul lui McCulloch",
                "text": "Estimatorul lui McCulloch folosește $\\nu_\\alpha = (q_{0,95} - q_{0,05})/(q_{0,75} - q_{0,25})$. Ce valoare ia $\\nu_\\alpha$ pentru distribuția Normală?",
                "options": [
                    "Aproximativ 1,0",
                    "Aproximativ 6,31",
                    "Aproximativ 2,439",
                    "Este infinit"
                ],
                "correctExplanation": "Pentru distribuția Normală $\\nu_\\alpha = (2 \\times 1{,}645)/(2 \\times 0{,}674) \\approx 2{,}439$; valorile mai mari înseamnă cozi mai groase și $\\alpha < 2$.",
                "incorrectExplanation": "$\\nu_\\alpha = 1{,}645/0{,}674 \\approx 2{,}439$ pentru distribuția Normală; 6,31 este valoarea pentru Cauchy; cozile mai groase cresc $\\nu_\\alpha$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Quantiles vs likelihood",
                "text": "Compared with maximum likelihood, what is the main trade-off of McCulloch's quantile method?",
                "options": [
                    "It is always more efficient than ML",
                    "It is fast and simple but less efficient; it is a good starting point for ML",
                    "It needs the closed-form density",
                    "It works only for $\\alpha = 2$"
                ],
                "correctExplanation": "McCulloch uses five sample quantiles and tables: quick and consistent, but with larger standard errors than ML, which uses the whole density.",
                "incorrectExplanation": "The quantile method needs no density and is fast, but it uses only a few quantiles, so ML is more efficient; ML usually starts from McCulloch's values."
            },
            "ro": {
                "title": "Cuantile față de verosimilitate",
                "text": "Față de metoda verosimilității maxime, care este principalul compromis al metodei cuantilelor a lui McCulloch?",
                "options": [
                    "Este întotdeauna mai eficientă decît ML",
                    "Este rapidă și simplă, dar mai puțin eficientă; este un bun punct de pornire pentru ML",
                    "Are nevoie de densitatea în formă închisă",
                    "Funcționează doar pentru $\\alpha = 2$"
                ],
                "correctExplanation": "McCulloch folosește cinci cuantile de selecție și tabele: rapidă și consistentă, dar cu erori standard mai mari decît ML, care folosește întreaga densitate.",
                "incorrectExplanation": "Metoda cuantilelor nu are nevoie de densitate și este rapidă, dar folosește doar cîteva cuantile, deci ML este mai eficientă; ML pornește de obicei de la valorile McCulloch."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Reading a QQ plot",
                "text": "In a QQ plot of daily returns against a fitted Normal distribution, the points bend away from the line at both ends. What does this show?",
                "options": [
                    "The returns are lighter-tailed than the Normal distribution",
                    "The mean is estimated wrongly",
                    "The empirical tails are heavier than the Normal tails",
                    "The sample is too large"
                ],
                "correctExplanation": "Extreme empirical quantiles are larger in absolute value than the Normal quantiles: the classic S shape of heavy tails.",
                "incorrectExplanation": "Points below the line on the left and above it on the right mean more extreme returns than the Normal model allows: heavy tails."
            },
            "ro": {
                "title": "Citirea unui QQ plot",
                "text": "Într-un QQ plot al randamentelor zilnice față de o distribuție Normală estimată, punctele se depărtează de dreaptă la ambele capete. Ce arată aceasta?",
                "options": [
                    "Randamentele au cozi mai subțiri decît distribuția Normală",
                    "Media este estimată greșit",
                    "Cozile empirice sînt mai groase decît cozile distribuției Normale",
                    "Eșantionul este prea mare"
                ],
                "correctExplanation": "Cuantilele empirice extreme sînt mai mari în valoare absolută decît cuantilele distribuției Normale: forma în S tipică pentru cozile groase.",
                "incorrectExplanation": "Puncte sub dreaptă la stînga și deasupra ei la dreapta înseamnă mai multe randamente extreme decît permite modelul Normal: cozi groase."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Tail probabilities compared",
                "text": "Fitted to the same daily returns, how do the probabilities of a loss beyond 5 scale units usually rank?",
                "options": [
                    "Normal smallest; stable largest; Student-t in between",
                    "All three models give about the same probability",
                    "Normal largest; stable smallest",
                    "Student-t always largest"
                ],
                "correctExplanation": "The Normal tail decays like $e^{-x^2/2}$, the Student-t tail like $x^{-\\nu}$ with $\\nu$ around 3–5, the stable tail like $x^{-\\alpha}$ with $\\alpha < 2$.",
                "incorrectExplanation": "The heavier the tail, the larger the probability of an extreme loss: the stable law (tail exponent below 2) gives the most, the Normal distribution by far the least."
            },
            "ro": {
                "title": "Compararea probabilităților din coadă",
                "text": "Pentru trei modele estimate pe aceleași randamente zilnice, cum se ordonează de obicei probabilitățile unei pierderi de peste 5 unități de scală?",
                "options": [
                    "Distribuția Normală: cea mai mică; legea stabilă: cea mai mare; Student-t: între ele",
                    "Toate trei modelele dau aproximativ aceeași probabilitate",
                    "Distribuția Normală: cea mai mare; legea stabilă: cea mai mică",
                    "Student-t întotdeauna cea mai mare"
                ],
                "correctExplanation": "Coada distribuției Normale descrește ca $e^{-x^2/2}$, coada Student-t ca $x^{-\\nu}$ cu $\\nu$ în jur de 3–5, coada stabilă ca $x^{-\\alpha}$ cu $\\alpha < 2$.",
                "incorrectExplanation": "Cu cît coada este mai groasă, cu atît probabilitatea unei pierderi extreme este mai mare: legea stabilă (exponent sub 2) dă cea mai mare valoare, distribuția Normală de departe cea mai mică."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "VaR under three models",
                "text": "A risk manager computes the daily VaR 1% of a portfolio under a Normal, a Student-t and a stable model fitted to the same returns. What is the usual outcome?",
                "options": [
                    "The Normal VaR 1% is the largest, so it is the most prudent",
                    "The Normal VaR 1% is the smallest, so it understates the tail risk",
                    "The three values are identical",
                    "The stable VaR 1% does not exist because the variance is infinite"
                ],
                "correctExplanation": "Heavy-tailed models put more mass in the left tail, so their 1% quantile lies further out; the Normal model gives the least prudent number.",
                "incorrectExplanation": "A quantile exists for every distribution, including stable laws with infinite variance; the Normal model, with the thinnest tail, gives the smallest VaR 1%."
            },
            "ro": {
                "title": "VaR conform a trei modele",
                "text": "Un manager de risc calculează VaR 1% zilnic al unui portofoliu cu un model Normal, unul Student-t și unul stabil, estimate pe aceleași randamente. Care este rezultatul obișnuit?",
                "options": [
                    "VaR 1% din modelul Normal este cel mai mare, deci este cel mai prudent",
                    "VaR 1% din modelul Normal este cel mai mic, deci subestimează riscul din coadă",
                    "Cele trei valori sînt identice",
                    "VaR 1% stabil nu există pentru că varianța este infinită"
                ],
                "correctExplanation": "Modelele cu cozi groase pun mai multă masă în coada stîngă, deci cuantila de 1% este mai departe; modelul Normal dă valoarea cea mai puțin prudentă.",
                "incorrectExplanation": "O cuantilă există pentru orice distribuție, inclusiv pentru legile stabile cu varianță infinită; modelul Normal, cu coada cea mai subțire, dă cel mai mic VaR 1%."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Aggregation critique",
                "text": "Estimated $\\alpha$ rises from about 1.6 for daily returns towards 2 for monthly returns of the same index. Why is this evidence against the i.i.d. stable model?",
                "options": [
                    "Monthly returns always have a smaller $\\alpha$",
                    "$\\alpha$ must equal 1 for monthly data",
                    "Aggregation always makes $\\alpha$ exceed 2",
                    "Under i.i.d. stable returns, sums keep the same $\\alpha$ at every horizon"
                ],
                "correctExplanation": "Stability implies that weekly and monthly sums have exactly the daily $\\alpha$; a rising $\\alpha$ points to finite variance and the classical CLT at work.",
                "incorrectExplanation": "Stability under summation means the same $\\alpha$ at all horizons; estimates that rise with aggregation fit a finite-variance law whose sums approach the Normal distribution."
            },
            "ro": {
                "title": "Critica prin agregare",
                "text": "$\\alpha$ estimat crește de la circa 1,6 pentru randamentele zilnice spre 2 pentru randamentele lunare ale aceluiași indice. De ce este aceasta o dovadă împotriva modelului stabil i.i.d.?",
                "options": [
                    "Randamentele lunare au întotdeauna un $\\alpha$ mai mic",
                    "$\\alpha$ trebuie să fie 1 pentru datele lunare",
                    "Agregarea face întotdeauna ca $\\alpha$ să depășească 2",
                    "Pentru randamente stabile i.i.d., sumele păstrează același $\\alpha$ la orice orizont"
                ],
                "correctExplanation": "Stabilitatea implică faptul că sumele săptămînale și lunare au exact $\\alpha$ zilnic; un $\\alpha$ care crește indică o varianță finită și acțiunea teoremei limită centrale clasice.",
                "incorrectExplanation": "Stabilitatea la adunare înseamnă același $\\alpha$ la toate orizonturile; estimările care cresc cu agregarea se potrivesc cu o lege cu varianță finită, ale cărei sume se apropie de distribuția Normală."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Tail index evidence",
                "text": "Tail index estimates computed only from the largest stock returns are usually around 3. What does this suggest?",
                "options": [
                    "The returns are exactly Cauchy",
                    "The variance is infinite",
                    "The Normal distribution fits the tails well",
                    "The variance is probably finite, which favours models such as the Student-t over stable laws with $\\alpha < 2$"
                ],
                "correctExplanation": "A tail exponent near 3 means $E|X|^p < \\infty$ for $p < 3$, so the variance exists; a stable law with $\\alpha < 2$ cannot have such tails.",
                "incorrectExplanation": "Stable laws with $\\alpha < 2$ have tail exponent $\\alpha$ below 2; an exponent near 3 gives finite variance, as in a Student-t with about 3 degrees of freedom. A power tail still rules out the Normal distribution."
            },
            "ro": {
                "title": "Dovezi din indicele de coadă",
                "text": "Estimările indicelui de coadă calculate doar din cele mai mari randamente ale acțiunilor sînt de obicei în jur de 3. Ce sugerează aceasta?",
                "options": [
                    "Randamentele sînt exact Cauchy",
                    "Varianța este infinită",
                    "Distribuția Normală descrie bine cozile",
                    "Varianța este probabil finită, ceea ce favorizează modele ca Student-t în fața legilor stabile cu $\\alpha < 2$"
                ],
                "correctExplanation": "Un exponent al cozii în jur de 3 înseamnă $E|X|^p < \\infty$ pentru $p < 3$, deci varianța există; o lege stabilă cu $\\alpha < 2$ nu poate avea astfel de cozi.",
                "incorrectExplanation": "Legile stabile cu $\\alpha < 2$ au exponentul cozii $\\alpha$ sub 2; un exponent în jur de 3 dă o varianță finită, ca la o distribuție Student-t cu circa 3 grade de libertate. O coadă de tip putere exclude totuși distribuția Normală."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Mandelbrot and cotton",
                "text": "What did Mandelbrot (1963) argue from cotton price changes?",
                "options": [
                    "Cotton prices follow the Normal distribution exactly",
                    "Price changes have no heavy tails",
                    "Price changes follow stable laws with $\\alpha < 2$, so their variance is infinite and the Normal model fails",
                    "Cotton prices are a deterministic cycle"
                ],
                "correctExplanation": "Mandelbrot found that the tails of cotton price changes decay like a power law with $\\alpha \\approx 1.7$ and proposed stable Paretian laws.",
                "incorrectExplanation": "Mandelbrot's point was the opposite of Normality: heavy power tails, with sample variances that do not settle, modelled by stable laws with $\\alpha < 2$."
            },
            "ro": {
                "title": "Mandelbrot și bumbacul",
                "text": "Ce a susținut Mandelbrot (1963) pornind de la variațiile prețului bumbacului?",
                "options": [
                    "Prețurile bumbacului urmează exact distribuția Normală",
                    "Variațiile de preț nu au cozi groase",
                    "Variațiile de preț urmează legi stabile cu $\\alpha < 2$, deci varianța lor este infinită și modelul Normal nu se potrivește",
                    "Prețurile bumbacului sînt un ciclu determinist"
                ],
                "correctExplanation": "Mandelbrot a observat că cozile variațiilor de preț ale bumbacului descresc ca o lege de tip putere cu $\\alpha \\approx 1{,}7$ și a propus legile stabile de tip Pareto.",
                "incorrectExplanation": "Ideea lui Mandelbrot contrazicea ipoteza de normalitate: cozi groase de tip putere, cu varianțe de selecție care nu se stabilizează, modelate prin legi stabile cu $\\alpha < 2$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Fama (1965)",
                "text": "What did Fama (1965) find for the daily returns of the 30 Dow Jones stocks?",
                "options": [
                    "Returns that fit the Normal distribution well",
                    "Heavy tails consistent with stable laws with $\\alpha < 2$, supporting Mandelbrot",
                    "Strong predictability of daily returns",
                    "A tail index far above 4"
                ],
                "correctExplanation": "Fama found more extreme returns than the Normal distribution allows and characteristic exponents below 2, in line with Mandelbrot's hypothesis.",
                "incorrectExplanation": "Fama (1965) supported the stable Paretian hypothesis with $\\alpha < 2$; the later critique (finite variance, $\\alpha$ rising with aggregation) came from other studies."
            },
            "ro": {
                "title": "Fama (1965)",
                "text": "Ce a găsit Fama (1965) pentru randamentele zilnice ale celor 30 de acțiuni din Dow Jones?",
                "options": [
                    "Randamente care se potrivesc bine cu distribuția Normală",
                    "Cozi groase compatibile cu legi stabile cu $\\alpha < 2$, în sprijinul lui Mandelbrot",
                    "O previzibilitate puternică a randamentelor zilnice",
                    "Un indice de coadă mult peste 4"
                ],
                "correctExplanation": "Fama a găsit mai multe randamente extreme decît permite distribuția Normală și exponenți caracteristici sub 2, în acord cu ipoteza lui Mandelbrot.",
                "incorrectExplanation": "Fama (1965) a susținut ipoteza legilor stabile de tip Pareto, cu $\\alpha < 2$; critica ulterioară (varianță finită, $\\alpha$ care crește cu agregarea) a venit din alte studii."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Student-t versus stable",
                "text": "Which property distinguishes a Student-t with $\\nu = 4$ from a stable law with $\\alpha = 1.7$?",
                "options": [
                    "The Student-t has a finite variance; the stable law does not",
                    "Only the stable law has heavy tails",
                    "The Student-t is stable under summation",
                    "The stable law has a closed-form density"
                ],
                "correctExplanation": "Both have power tails, but the Student-t with $\\nu = 4$ has finite moments of order $p < 4$, including the variance.",
                "incorrectExplanation": "Both models have heavy power tails; the Student-t is not closed under summation and has a finite variance for $\\nu > 2$, while a stable law with $\\alpha = 1.7$ has none and no closed-form density."
            },
            "ro": {
                "title": "Student-t față de stabilă",
                "text": "Ce proprietate deosebește o distribuție Student-t cu $\\nu = 4$ de o lege stabilă cu $\\alpha = 1{,}7$?",
                "options": [
                    "Student-t are varianță finită; legea stabilă nu are",
                    "Doar legea stabilă are cozi groase",
                    "Student-t este stabilă la adunare",
                    "Legea stabilă are densitate în formă închisă"
                ],
                "correctExplanation": "Ambele au cozi de tip putere, dar Student-t cu $\\nu = 4$ are momente finite de ordin $p < 4$, inclusiv varianța.",
                "incorrectExplanation": "Ambele modele au cozi groase de tip putere; Student-t nu este închisă la adunare și are varianță finită pentru $\\nu > 2$, în timp ce legea stabilă cu $\\alpha = 1{,}7$ nu are varianță și nici densitate în formă închisă."
            }
        }
    ]
};
