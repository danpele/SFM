// ============================================================
// Chapter 5 quiz bank: heavy tails and extreme value theory (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['evt'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Heavy tail",
                "text": "Which survival function $P(L > x)$ describes a heavy (power-law) tail?",
                "options": [
                    "$e^{-x^2/2}$",
                    "$e^{-2x}$",
                    "$C x^{-3}$ for large $x$",
                    "$1 - x$ for $0 \\le x \\le 1$"
                ],
                "correctExplanation": "A power law $Cx^{-\\alpha}$ falls much more slowly than any exponential: this is a heavy tail with index $\\alpha = 3$.",
                "incorrectExplanation": "Exponential and Gaussian tails are light; a bounded law has no tail at all. Only $Cx^{-3}$ is a power law."
            },
            "ro": {
                "title": "Coadă groasă",
                "text": "Ce funcție de supraviețuire $P(L > x)$ descrie o coadă groasă (de tip putere)?",
                "options": [
                    "$e^{-x^2/2}$",
                    "$e^{-2x}$",
                    "$C x^{-3}$ pentru $x$ mare",
                    "$1 - x$ pentru $0 \\le x \\le 1$"
                ],
                "correctExplanation": "O lege de putere $Cx^{-\\alpha}$ scade mult mai încet decît orice exponențială: este o coadă groasă cu tail index-ul $\\alpha = 3$.",
                "incorrectExplanation": "Cozile exponențiale și cea a distribuției Normale sînt subțiri; o distribuție mărginită nu are coadă. Doar $Cx^{-3}$ este o lege de putere."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Doubling the loss",
                "text": "Losses have a Pareto tail with $\\alpha = 3$. By what factor does $P(L > 2x)$ differ from $P(L > x)$?",
                "options": [
                    "It is 2 times smaller",
                    "It is 8 times smaller",
                    "It is 3 times smaller",
                    "It is the same"
                ],
                "correctExplanation": "$P(L > 2x)/P(L > x) = 2^{-\\alpha} = 2^{-3} = 1/8$.",
                "incorrectExplanation": "For a power tail the ratio is $2^{-\\alpha}$, here $1/8$, whatever the level $x$."
            },
            "ro": {
                "title": "Dublarea pierderii",
                "text": "Pierderile au o coadă Pareto cu $\\alpha = 3$. De cîte ori diferă $P(L > 2x)$ de $P(L > x)$?",
                "options": [
                    "Este de 2 ori mai mică",
                    "Este de 8 ori mai mică",
                    "Este de 3 ori mai mică",
                    "Este aceeași"
                ],
                "correctExplanation": "$P(L > 2x)/P(L > x) = 2^{-\\alpha} = 2^{-3} = 1/8$.",
                "incorrectExplanation": "Pentru o coadă de tip putere raportul este $2^{-\\alpha}$, aici $1/8$, oricare ar fi nivelul $x$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Moments and the tail index",
                "text": "A loss distribution has tail index $\\alpha = 3$. Which statement is true?",
                "options": [
                    "The variance is infinite",
                    "The variance is finite, the kurtosis is infinite",
                    "All moments are finite",
                    "The mean is infinite"
                ],
                "correctExplanation": "Moments of order $m < \\alpha$ exist: $m = 2$ yes, $m = 4$ no.",
                "incorrectExplanation": "$E|L|^m$ is finite only for $m < \\alpha = 3$: mean and variance exist, the kurtosis (fourth moment) does not."
            },
            "ro": {
                "title": "Momente și tail index-ul",
                "text": "O distribuție a pierderilor are tail index-ul $\\alpha = 3$. Ce afirmație este adevărată?",
                "options": [
                    "Varianța este infinită",
                    "Varianța este finită, boltirea este infinită",
                    "Toate momentele sînt finite",
                    "Media este infinită"
                ],
                "correctExplanation": "Momentele de ordin $m < \\alpha$ există: $m = 2$ da, $m = 4$ nu.",
                "incorrectExplanation": "$E|L|^m$ este finit doar pentru $m < \\alpha = 3$: media și varianța există, boltirea (momentul de ordin patru) nu."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Student-t tail index",
                "text": "What is the tail index of a Student-t distribution with $\\nu = 4$ degrees of freedom?",
                "options": [
                    "$\\alpha = 2$",
                    "$\\alpha = 4$",
                    "$\\alpha = 1/4$",
                    "It has no power tail"
                ],
                "correctExplanation": "The Student-t tail decays like $x^{-\\nu}$, so $\\alpha = \\nu = 4$.",
                "incorrectExplanation": "A Student-t with $\\nu$ degrees of freedom is regularly varying with index $\\alpha = \\nu$."
            },
            "ro": {
                "title": "Tail index-ul unei distribuții Student-t",
                "text": "Care este tail index-ul unei distribuții Student-t cu $\\nu = 4$ grade de libertate?",
                "options": [
                    "$\\alpha = 2$",
                    "$\\alpha = 4$",
                    "$\\alpha = 1/4$",
                    "Nu are coadă de tip putere"
                ],
                "correctExplanation": "Coada Student-t scade ca $x^{-\\nu}$, deci $\\alpha = \\nu = 4$.",
                "incorrectExplanation": "O Student-t cu $\\nu$ grade de libertate are variație regulată cu indicele $\\alpha = \\nu$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Hill estimator",
                "text": "With the $k$ largest losses $L_{(1)} \\ge \\dots \\ge L_{(k)}$ and $L_{(k+1)}$, the Hill estimator is",
                "options": [
                    "$\\frac1k\\sum_{i=1}^k L_{(i)}$",
                    "$\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})$",
                    "$\\big[\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})\\big]^{-1}$",
                    "$L_{(1)}/L_{(k+1)}$"
                ],
                "correctExplanation": "The mean of the log excesses estimates $1/\\alpha$; its inverse estimates $\\alpha$.",
                "incorrectExplanation": "The mean log excess estimates $1/\\alpha$, not $\\alpha$; Hill takes its inverse."
            },
            "ro": {
                "title": "Estimatorul Hill",
                "text": "Cu cele mai mari $k$ pierderi $L_{(1)} \\ge \\dots \\ge L_{(k)}$ și $L_{(k+1)}$, estimatorul Hill este",
                "options": [
                    "$\\frac1k\\sum_{i=1}^k L_{(i)}$",
                    "$\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})$",
                    "$\\big[\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})\\big]^{-1}$",
                    "$L_{(1)}/L_{(k+1)}$"
                ],
                "correctExplanation": "Media exceselor logaritmice estimează $1/\\alpha$; inversa ei estimează $\\alpha$.",
                "incorrectExplanation": "Media exceselor logaritmice estimează $1/\\alpha$, nu $\\alpha$; Hill ia inversa ei."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Hill by hand",
                "text": "The mean of the $k = 100$ log excesses is $0.25$. What are $\\hat\\alpha$ and its i.i.d. standard error?",
                "options": [
                    "$\\hat\\alpha = 0.25$, SE $= 0.025$",
                    "$\\hat\\alpha = 4$, SE $= 0.04$",
                    "$\\hat\\alpha = 4$, SE $= 0.4$",
                    "$\\hat\\alpha = 25$, SE $= 2.5$"
                ],
                "correctExplanation": "$\\hat\\alpha = 1/0.25 = 4$ and $\\mathrm{SE} = \\hat\\alpha/\\sqrt{k} = 4/10 = 0.4$.",
                "incorrectExplanation": "Invert the mean log excess and divide by $\\sqrt{k} = 10$: $\\hat\\alpha = 4$, SE $= 0.4$."
            },
            "ro": {
                "title": "Hill pas cu pas",
                "text": "Media celor $k = 100$ de excese logaritmice este $0{,}25$. Cît sînt $\\hat\\alpha$ și eroarea lui standard i.i.d.?",
                "options": [
                    "$\\hat\\alpha = 0{,}25$, SE $= 0{,}025$",
                    "$\\hat\\alpha = 4$, SE $= 0{,}04$",
                    "$\\hat\\alpha = 4$, SE $= 0{,}4$",
                    "$\\hat\\alpha = 25$, SE $= 2{,}5$"
                ],
                "correctExplanation": "$\\hat\\alpha = 1/0{,}25 = 4$ și $\\mathrm{SE} = \\hat\\alpha/\\sqrt{k} = 4/10 = 0{,}4$.",
                "incorrectExplanation": "Inversăm media exceselor logaritmice și împărțim la $\\sqrt{k} = 10$: $\\hat\\alpha = 4$, SE $= 0{,}4$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Choosing k",
                "text": "What happens to the Hill estimator when $k$ is too large?",
                "options": [
                    "The body of the distribution enters and biases $\\hat\\alpha$",
                    "The variance becomes very large",
                    "Nothing: larger $k$ is always better",
                    "The estimator cannot be computed"
                ],
                "correctExplanation": "Large $k$ reduces the variance but uses losses where the power law fails: bias. One reads $\\hat\\alpha$ where the Hill plot is flat.",
                "incorrectExplanation": "Small $k$ means high variance; large $k$ lets the body in and creates bias."
            },
            "ro": {
                "title": "Alegerea lui k",
                "text": "Ce se întîmplă cu estimatorul Hill cînd $k$ este prea mare?",
                "options": [
                    "Intră corpul distribuției și deplasează $\\hat\\alpha$",
                    "Varianța devine foarte mare",
                    "Nimic: un $k$ mai mare este întotdeauna mai bun",
                    "Estimatorul nu se poate calcula"
                ],
                "correctExplanation": "Un $k$ mare reduce varianța, dar folosește pierderi unde legea de putere nu mai este valabilă: deplasare. Alegem $\\hat\\alpha$ din zona în care graficul Hill este aproximativ orizontal.",
                "incorrectExplanation": "Un $k$ mic înseamnă varianță mare; un $k$ mare include corpul distribuției și creează deplasare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Block bootstrap",
                "text": "Why use a moving-block bootstrap for the standard error of the Hill estimator on daily returns?",
                "options": [
                    "Large losses come in clusters; blocks keep the dependence",
                    "To make $\\hat\\alpha$ larger",
                    "Because the Hill estimator needs Normal data",
                    "To remove the largest losses"
                ],
                "correctExplanation": "Volatility clustering makes returns dependent; resampling blocks of consecutive days keeps that dependence in the bootstrap samples.",
                "incorrectExplanation": "The i.i.d. formula ignores clustering; blocks of consecutive days keep it."
            },
            "ro": {
                "title": "Bootstrap pe blocuri",
                "text": "De ce folosim un bootstrap pe blocuri mobile pentru eroarea standard a estimatorului Hill pe randamente zilnice?",
                "options": [
                    "Pierderile mari apar grupat; blocurile păstrează dependența",
                    "Pentru a mări $\\hat\\alpha$",
                    "Pentru că estimatorul Hill cere date Normale",
                    "Pentru a elimina cele mai mari pierderi"
                ],
                "correctExplanation": "Volatility clustering face randamentele dependente; reeșantionarea unor blocuri de zile consecutive păstrează această dependență în eșantioanele bootstrap.",
                "incorrectExplanation": "Formula i.i.d. ignoră această dependență; blocurile de zile consecutive o păstrează."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Hill estimates in the data",
                "text": "At $k = 2.5\\%$ of $n$, the Hill estimate for S&P 500 losses since 1990 is about 3.0. What does this suggest?",
                "options": [
                    "Infinite variance, as in a stable law with $\\alpha < 2$",
                    "A Normal distribution",
                    "Finite variance and an infinite fourth moment",
                    "A bounded distribution"
                ],
                "correctExplanation": "A tail index near 3 (the inverse cubic law) gives a finite variance but no finite kurtosis.",
                "incorrectExplanation": "With $\\hat\\alpha \\approx 3$: moments of order below 3 exist; the variance is finite, the kurtosis is not."
            },
            "ro": {
                "title": "Estimări Hill în date",
                "text": "La $k = 2{,}5\\%$ din $n$, estimarea Hill pentru pierderile S&P 500 din 1990 este de circa 3,0. Ce sugerează?",
                "options": [
                    "Varianță infinită, ca la o lege stabilă cu $\\alpha < 2$",
                    "O distribuție Normală",
                    "Varianță finită și moment de ordin patru infinit",
                    "O distribuție mărginită"
                ],
                "correctExplanation": "Un tail index în jur de 3 (legea cubică inversă) implică o varianță finită, dar o boltire infinită.",
                "incorrectExplanation": "Cu $\\hat\\alpha \\approx 3$: există momentele de ordin mai mic decît 3; varianța este finită, boltirea nu."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Mean excess function",
                "text": "The empirical mean excess function of daily losses rises roughly linearly above a threshold. This points to",
                "options": [
                    "An exponential tail",
                    "A Normal tail",
                    "A bounded distribution",
                    "A heavy, Pareto-type tail ($\\xi > 0$)"
                ],
                "correctExplanation": "For a GPD with $0 < \\xi < 1$, $e(u)$ is a rising line with slope $\\xi/(1 - \\xi)$.",
                "incorrectExplanation": "Exponential: constant $e(u)$; Normal: falling $e(u)$; a rising line means a power tail."
            },
            "ro": {
                "title": "Funcția mean excess",
                "text": "Funcția mean excess empirică a pierderilor zilnice crește aproximativ liniar peste un prag. Aceasta indică",
                "options": [
                    "O coadă exponențială",
                    "Coada distribuției Normale",
                    "O distribuție mărginită",
                    "O coadă groasă, de tip Pareto ($\\xi > 0$)"
                ],
                "correctExplanation": "Pentru o GPD cu $0 < \\xi < 1$, $e(u)$ este o dreaptă crescătoare cu panta $\\xi/(1 - \\xi)$.",
                "incorrectExplanation": "Distribuția exponențială: $e(u)$ constantă; distribuția Normală: $e(u)$ descrescătoare; o dreaptă crescătoare indică o coadă de tip putere."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Mean excess of a Pareto tail",
                "text": "For a Pareto tail with $\\alpha = 3$, what is $e(u)$ at $u = 4\\%$?",
                "options": [
                    "$4\\%$",
                    "$12\\%$",
                    "$1.33\\%$",
                    "$2\\%$"
                ],
                "correctExplanation": "$e(u) = u/(\\alpha - 1) = 4/2 = 2\\%$.",
                "incorrectExplanation": "For a Pareto tail $e(u) = u/(\\alpha - 1)$; with $\\alpha = 3$ and $u = 4\\%$ this is $2\\%$."
            },
            "ro": {
                "title": "Mean excess pentru o coadă Pareto",
                "text": "Pentru o coadă Pareto cu $\\alpha = 3$, cît este $e(u)$ la $u = 4\\%$?",
                "options": [
                    "$4\\%$",
                    "$12\\%$",
                    "$1{,}33\\%$",
                    "$2\\%$"
                ],
                "correctExplanation": "$e(u) = u/(\\alpha - 1) = 4/2 = 2\\%$.",
                "incorrectExplanation": "Pentru o coadă Pareto, $e(u) = u/(\\alpha - 1)$; cu $\\alpha = 3$ și $u = 4\\%$ dă $2\\%$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Fisher–Tippett–Gnedenko",
                "text": "If normalised maxima of i.i.d. variables converge to a non-degenerate law, that law is",
                "options": [
                    "Always the Normal distribution",
                    "A generalised extreme value (GEV) law",
                    "Always the Gumbel law",
                    "A generalised Pareto law"
                ],
                "correctExplanation": "The theorem says the only possible limits of normalised maxima form the GEV family (Fréchet, Gumbel, Weibull).",
                "incorrectExplanation": "The GPD describes excesses over a threshold; the limit of maxima is a GEV, of one of three types."
            },
            "ro": {
                "title": "Fisher–Tippett–Gnedenko",
                "text": "Dacă maximele normalizate ale unor variabile i.i.d. converg către o distribuție nedegenerată, acea distribuție este",
                "options": [
                    "Întotdeauna distribuția Normală",
                    "O distribuție generalizată a valorilor extreme (GEV)",
                    "Întotdeauna distribuția Gumbel",
                    "O distribuție Pareto generalizată"
                ],
                "correctExplanation": "Teorema spune că singurele limite posibile ale maximelor normalizate formează familia GEV (Fréchet, Gumbel, Weibull).",
                "incorrectExplanation": "GPD descrie excesele peste un prag; limita maximelor este o GEV, de unul dintre cele trei tipuri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Domain of attraction",
                "text": "Monthly maxima of daily losses with tail index $\\alpha \\approx 3$ follow approximately which law?",
                "options": [
                    "Gumbel, with $\\xi = 0$",
                    "Weibull, with $\\xi = -1/3$",
                    "Normal",
                    "Fréchet, with $\\xi \\approx 1/3$"
                ],
                "correctExplanation": "Power tails lead to the Fréchet type with $\\xi = 1/\\alpha$.",
                "incorrectExplanation": "Gumbel is the limit for light tails, Weibull for bounded variables; power tails give Fréchet with $\\xi = 1/\\alpha$."
            },
            "ro": {
                "title": "Domeniul de atracție",
                "text": "Maximele lunare ale pierderilor zilnice cu tail index-ul $\\alpha \\approx 3$ urmează aproximativ ce distribuție?",
                "options": [
                    "Gumbel, cu $\\xi = 0$",
                    "Weibull, cu $\\xi = -1/3$",
                    "Distribuția Normală",
                    "Fréchet, cu $\\xi \\approx 1/3$"
                ],
                "correctExplanation": "Cozile de tip putere duc la tipul Fréchet cu $\\xi = 1/\\alpha$.",
                "incorrectExplanation": "Gumbel este limita pentru cozi subțiri, Weibull pentru variabile mărginite; cozile de tip putere duc la Fréchet cu $\\xi = 1/\\alpha$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Maxima of Normal variables",
                "text": "Maxima of i.i.d. Normal variables converge (after normalisation) to which type?",
                "options": [
                    "Fréchet",
                    "Weibull",
                    "Gumbel",
                    "They do not converge"
                ],
                "correctExplanation": "The Normal law is in the Gumbel domain of attraction ($\\xi = 0$), like the Exponential and the lognormal.",
                "incorrectExplanation": "Light, unbounded tails give the Gumbel type; Fréchet needs power tails, Weibull a finite end point."
            },
            "ro": {
                "title": "Maximele variabilelor Normale",
                "text": "Maximele unor variabile Normale i.i.d. converg (după normalizare) către ce tip?",
                "options": [
                    "Fréchet",
                    "Weibull",
                    "Gumbel",
                    "Nu converg"
                ],
                "correctExplanation": "Distribuția Normală este în domeniul de atracție Gumbel ($\\xi = 0$), ca și distribuțiile exponențială și lognormală.",
                "incorrectExplanation": "Cozile subțiri, nemărginite, dau tipul Gumbel; Fréchet cere cozi de tip putere, Weibull un capăt finit."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Return level",
                "text": "Monthly maxima follow a GEV. What is the 10-year return level?",
                "options": [
                    "The largest loss observed in the last 10 years",
                    "The level the monthly maximum exceeds on average once in 120 months",
                    "The mean of the monthly maxima over 10 years",
                    "The loss exceeded on 10% of days"
                ],
                "correctExplanation": "$z = H^{-1}(1 - 1/120)$: the level exceeded once in 120 blocks on average.",
                "incorrectExplanation": "A return level is a quantile of the block-maximum law, $H^{-1}(1 - 1/m)$ with $m = 12T$ months; it is not an observed value."
            },
            "ro": {
                "title": "Return level",
                "text": "Maximele lunare urmează o GEV. Ce este return level-ul de 10 ani?",
                "options": [
                    "Cea mai mare pierdere observată în ultimii 10 ani",
                    "Nivelul pe care maximul lunar îl depășește în medie o dată la 120 de luni",
                    "Media maximelor lunare pe 10 ani",
                    "Pierderea depășită în 10% dintre zile"
                ],
                "correctExplanation": "$z = H^{-1}(1 - 1/120)$: nivelul depășit, în medie, o dată la 120 de blocuri.",
                "incorrectExplanation": "Un return level este o cuantilă a distribuției maximului pe bloc, $H^{-1}(1 - 1/m)$ cu $m = 12T$ luni; nu este o valoare observată."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Return levels in the data",
                "text": "A GEV fit to monthly maxima of S&P 500 losses gives a 10-year return level of about 7.4%. The largest loss since 1990 was 12.8%. What does this mean?",
                "options": [
                    "The model is wrong, since a larger loss was observed",
                    "Losses above the 10-year level occur a few times in 36 years, as expected",
                    "No loss can exceed the 10-year level",
                    "The 10-year level equals the largest loss"
                ],
                "correctExplanation": "In 36 years one expects about 3–4 months above the 10-year level; exceeding it is normal.",
                "incorrectExplanation": "A 10-year level is exceeded on average once per 10 years, so a few exceedances in 36 years are consistent with the model."
            },
            "ro": {
                "title": "Return levels în date",
                "text": "O ajustare GEV la maximele lunare ale pierderilor S&P 500 dă un return level de 10 ani de circa 7,4%. Cea mai mare pierdere din 1990 a fost de 12,8%. Ce înseamnă?",
                "options": [
                    "Modelul este greșit, pentru că s-a observat o pierdere mai mare",
                    "Pierderile peste nivelul de 10 ani apar de cîteva ori în 36 de ani, cum era de așteptat",
                    "Nicio pierdere nu poate depăși nivelul de 10 ani",
                    "Nivelul de 10 ani este egal cu cea mai mare pierdere"
                ],
                "correctExplanation": "În 36 de ani ne așteptăm la circa 3–4 luni peste nivelul de 10 ani; depășirea lui nu este surprinzătoare.",
                "incorrectExplanation": "Un nivel de 10 ani este depășit în medie o dată la 10 ani, deci cîteva depășiri în 36 de ani sînt compatibile cu modelul."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Pickands–Balkema–de Haan",
                "text": "For a high threshold $u$, the excesses $L - u$ given $L > u$ follow approximately",
                "options": [
                    "A GEV distribution",
                    "A Normal distribution",
                    "A uniform distribution",
                    "A generalised Pareto distribution"
                ],
                "correctExplanation": "The Pickands–Balkema–de Haan theorem: excesses over high thresholds are approximately GPD, with the same $\\xi$ as the GEV.",
                "incorrectExplanation": "Maxima follow a GEV; excesses over a high threshold follow a GPD."
            },
            "ro": {
                "title": "Pickands–Balkema–de Haan",
                "text": "Pentru un prag înalt $u$, excesele $L - u$, știind că $L > u$, urmează aproximativ",
                "options": [
                    "O distribuție GEV",
                    "O distribuție Normală",
                    "O distribuție uniformă",
                    "O distribuție Pareto generalizată"
                ],
                "correctExplanation": "Teorema Pickands–Balkema–de Haan: excesele peste praguri înalte sînt aproximativ GPD, cu același $\\xi$ ca la GEV.",
                "incorrectExplanation": "Maximele urmează o GEV; excesele peste un prag înalt urmează o GPD."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GPD shape",
                "text": "A GPD fit gives $\\hat\\xi = 0.25$. What is the implied tail index, and is the variance finite?",
                "options": [
                    "$\\alpha = 0.25$; no",
                    "$\\alpha = 4$; no",
                    "$\\alpha = 4$; yes",
                    "$\\alpha = 1.25$; no"
                ],
                "correctExplanation": "$\\alpha = 1/\\xi = 4$; the variance is finite because $\\xi < 1/2$.",
                "incorrectExplanation": "The tail index is the inverse of the shape, $\\alpha = 1/\\xi$; a GPD has a finite variance for $\\xi < 1/2$."
            },
            "ro": {
                "title": "Forma GPD",
                "text": "O ajustare GPD dă $\\hat\\xi = 0{,}25$. Care este tail index-ul implicat și este varianța finită?",
                "options": [
                    "$\\alpha = 0{,}25$; nu",
                    "$\\alpha = 4$; nu",
                    "$\\alpha = 4$; da",
                    "$\\alpha = 1{,}25$; nu"
                ],
                "correctExplanation": "$\\alpha = 1/\\xi = 4$; varianța este finită pentru că $\\xi < 1/2$.",
                "incorrectExplanation": "Tail index-ul este inversul formei, $\\alpha = 1/\\xi$; o GPD are varianță finită pentru $\\xi < 1/2$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Threshold choice",
                "text": "Which is a sound way to choose the POT threshold $u$?",
                "options": [
                    "Fix a high quantile and check that $\\hat\\xi$ and the VaR are stable for nearby thresholds",
                    "Choose $u$ so that VaR is as small as possible",
                    "Use $u = 0$, so that all losses are used",
                    "Use the largest loss as the threshold"
                ],
                "correctExplanation": "A rule fixed in advance (e.g. the 90% quantile), checked with the mean excess and parameter stability plots.",
                "incorrectExplanation": "The threshold trades bias against variance; it must not be tuned to get a desired VaR."
            },
            "ro": {
                "title": "Alegerea pragului",
                "text": "Care este un mod corect de a alege pragul $u$ în POT?",
                "options": [
                    "Fixăm o cuantilă înaltă și verificăm că $\\hat\\xi$ și VaR sînt stabile pentru praguri apropiate",
                    "Alegem $u$ astfel încît VaR să fie cît mai mic",
                    "Folosim $u = 0$, ca să folosim toate pierderile",
                    "Folosim cea mai mare pierdere ca prag"
                ],
                "correctExplanation": "O regulă fixată dinainte (de ex. cuantila de 90%), verificată cu graficele mean excess și de stabilitate a parametrilor.",
                "incorrectExplanation": "Pragul echilibrează deplasarea și varianța; nu trebuie ajustat pentru a obține un VaR dorit."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "POT VaR formula",
                "text": "With $n$ days, $N_u$ excesses over $u$ and a GPD$(\\xi, \\beta)$, the POT estimate of $\\mathrm{VaR}_p$ is",
                "options": [
                    "$u + \\frac{\\beta}{\\xi}\\big[(np/N_u)^{-\\xi} - 1\\big]$",
                    "$u + \\frac{\\beta}{\\xi}\\big[(np/N_u)^{\\xi} - 1\\big]$",
                    "$u \\times (np/N_u)^{-\\xi}$",
                    "$\\beta/(1 - \\xi)$"
                ],
                "correctExplanation": "Solving $(N_u/n)(1 + \\xi(x - u)/\\beta)^{-1/\\xi} = p$ for $x$ gives the formula with the exponent $-\\xi$.",
                "incorrectExplanation": "The exponent must be $-\\xi$: with $+\\xi$ the VaR falls below $u$ for $p < N_u/n$; $\\beta/(1 - \\xi)$ is the mean excess at $u$."
            },
            "ro": {
                "title": "Formula VaR POT",
                "text": "Cu $n$ zile, $N_u$ excese peste $u$ și o GPD$(\\xi, \\beta)$, estimarea POT a lui $\\mathrm{VaR}_p$ este",
                "options": [
                    "$u + \\frac{\\beta}{\\xi}\\big[(np/N_u)^{-\\xi} - 1\\big]$",
                    "$u + \\frac{\\beta}{\\xi}\\big[(np/N_u)^{\\xi} - 1\\big]$",
                    "$u \\times (np/N_u)^{-\\xi}$",
                    "$\\beta/(1 - \\xi)$"
                ],
                "correctExplanation": "Rezolvînd $(N_u/n)(1 + \\xi(x - u)/\\beta)^{-1/\\xi} = p$ în raport cu $x$ obținem formula cu exponentul $-\\xi$.",
                "incorrectExplanation": "Exponentul trebuie să fie $-\\xi$: cu $+\\xi$, VaR coboară sub $u$ pentru $p < N_u/n$; $\\beta/(1 - \\xi)$ este valoarea funcției mean excess în $u$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "POT VaR by hand",
                "text": "$n = 5000$, $N_u = 250$, $u = 2\\%$, $\\xi = 0.2$, $\\beta = 0.8$. Using $0.2^{-0.2} \\approx 1.38$, VaR 1% is about",
                "options": [
                    "$3.52\\%$",
                    "$2.38\\%$",
                    "$5.52\\%$",
                    "$1.52\\%$"
                ],
                "correctExplanation": "$np/N_u = 0.2$; $\\mathrm{VaR} = 2 + (0.8/0.2)(1.38 - 1) \\approx 3.52\\%$.",
                "incorrectExplanation": "Compute $np/N_u = 50/250 = 0.2$, then $u + (\\beta/\\xi)(0.2^{-\\xi} - 1) = 2 + 4 \\times 0.38$."
            },
            "ro": {
                "title": "VaR POT pas cu pas",
                "text": "$n = 5000$, $N_u = 250$, $u = 2\\%$, $\\xi = 0{,}2$, $\\beta = 0{,}8$. Folosind $0{,}2^{-0{,}2} \\approx 1{,}38$, VaR 1% este aproximativ",
                "options": [
                    "$3{,}52\\%$",
                    "$2{,}38\\%$",
                    "$5{,}52\\%$",
                    "$1{,}52\\%$"
                ],
                "correctExplanation": "$np/N_u = 0{,}2$; $\\mathrm{VaR} = 2 + (0{,}8/0{,}2)(1{,}38 - 1) \\approx 3{,}52\\%$.",
                "incorrectExplanation": "Calculăm $np/N_u = 50/250 = 0{,}2$, apoi $u + (\\beta/\\xi)(0{,}2^{-\\xi} - 1) = 2 + 4 \\times 0{,}38$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ES and VaR",
                "text": "For a GPD tail with $\\xi = 0.2$, far in the tail the ratio $\\mathrm{ES}_p/\\mathrm{VaR}_p$ approaches",
                "options": [
                    "$0.8$",
                    "$1$",
                    "$5$",
                    "$1.25$"
                ],
                "correctExplanation": "$\\mathrm{ES}_p/\\mathrm{VaR}_p \\to 1/(1 - \\xi) = 1/0.8 = 1.25$.",
                "incorrectExplanation": "ES is always at least VaR; for a GPD tail the ratio tends to $1/(1 - \\xi)$."
            },
            "ro": {
                "title": "ES și VaR",
                "text": "Pentru o coadă GPD cu $\\xi = 0{,}2$, departe în coadă raportul $\\mathrm{ES}_p/\\mathrm{VaR}_p$ se apropie de",
                "options": [
                    "$0{,}8$",
                    "$1$",
                    "$5$",
                    "$1{,}25$"
                ],
                "correctExplanation": "$\\mathrm{ES}_p/\\mathrm{VaR}_p \\to 1/(1 - \\xi) = 1/0{,}8 = 1{,}25$.",
                "incorrectExplanation": "ES este întotdeauna cel puțin egal cu VaR; pentru o coadă GPD raportul tinde la $1/(1 - \\xi)$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "EVT vs Normal in the data",
                "text": "For S&P 500 losses since 1990, VaR 0.1% is 6.37% by EVT and 3.47% by the Normal distribution. Why the gap?",
                "options": [
                    "The Normal tail is too thin far in the tail",
                    "EVT overestimates every quantile",
                    "The two methods use different data",
                    "VaR 0.1% does not exist for heavy tails"
                ],
                "correctExplanation": "Real losses have a power tail; the Normal law ignores it and understates far quantiles by a factor of about two.",
                "incorrectExplanation": "Quantiles always exist; the gap comes from the thin Normal tail, not from the data or from EVT."
            },
            "ro": {
                "title": "EVT și distribuția Normală în date",
                "text": "Pentru pierderile S&P 500 din 1990, VaR 0,1% este 6,37% prin EVT și 3,47% prin distribuția Normală. De ce diferența?",
                "options": [
                    "Distribuția Normală are o coadă prea subțire în zona extremă",
                    "EVT supraestimează orice cuantilă",
                    "Cele două metode folosesc date diferite",
                    "VaR 0,1% nu există pentru cozi groase"
                ],
                "correctExplanation": "Pierderile reale au o coadă de tip putere; distribuția Normală o ignoră și subestimează cuantilele îndepărtate de circa două ori.",
                "incorrectExplanation": "Cuantilele există întotdeauna; diferența vine din coada subțire a distribuției Normale, nu din date sau din EVT."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Out-of-sample check",
                "text": "A VaR 1% estimated until 2019 is exceeded on 25 of 1 687 days in 2020–2026 (about 17 expected). What is the most likely reason?",
                "options": [
                    "The VaR formula has a wrong sign",
                    "VaR 1% must never be exceeded",
                    "The test period is too long",
                    "Volatility clustering: the 2020 crash concentrated many large losses"
                ],
                "correctExplanation": "An unconditional VaR ignores the volatility regime; in a turbulent period exceedances cluster above the target.",
                "incorrectExplanation": "VaR 1% should be exceeded on about 1% of days; too many exceedances in one period point to changing volatility (conditional EVT, Chapters 9 and 10)."
            },
            "ro": {
                "title": "Verificarea în afara eșantionului",
                "text": "Un VaR 1% estimat pînă în 2019 este depășit în 25 din 1 687 de zile în 2020–2026 (circa 17 așteptate). Care este cel mai probabil motiv?",
                "options": [
                    "Formula VaR are un semn greșit",
                    "VaR 1% nu trebuie depășit niciodată",
                    "Perioada de test este prea lungă",
                    "Volatility clustering: crahul din 2020 a concentrat multe pierderi mari"
                ],
                "correctExplanation": "Un VaR necondiționat ignoră regimul de volatilitate; într-o perioadă agitată depășirile se acumulează peste valoarea așteptată.",
                "incorrectExplanation": "VaR 1% ar trebui depășit în circa 1% dintre zile; prea multe depășiri într-o perioadă indică variația volatilității în timp (EVT condiționată, Capitolele 9 și 10)."
            }
        }
    ]
};
