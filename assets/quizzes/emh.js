// ============================================================
// Chapter 7 quiz bank: efficient markets, random walk and variance-ratio tests (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['emh'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Weak-form efficiency",
                "text": "Which information set defines weak-form market efficiency (Fama, 1970)?",
                "options": [
                    "All public information, such as earnings and news",
                    "All information, including private information",
                    "Past prices and past returns",
                    "Analysts' forecasts only"
                ],
                "correctExplanation": "The weak form uses only the history of prices and returns: technical analysis should not earn excess returns.",
                "incorrectExplanation": "Public information defines the semi-strong form and all information the strong form; the weak form uses only past prices and returns."
            },
            "ro": {
                "title": "Eficiența în formă slabă",
                "text": "Ce mulțime de informații definește eficiența pieței în formă slabă (Fama, 1970)?",
                "options": [
                    "Toată informația publică, de exemplu profiturile și știrile",
                    "Toată informația, inclusiv cea privată",
                    "Prețurile și randamentele trecute",
                    "Doar prognozele analiștilor"
                ],
                "correctExplanation": "Forma slabă folosește doar istoria prețurilor și a randamentelor: analiza tehnică nu ar trebui să aducă randamente în exces.",
                "incorrectExplanation": "Informația publică definește forma semi-tare, iar toată informația forma tare; forma slabă folosește doar prețurile și randamentele trecute."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Joint-hypothesis problem",
                "text": "Why can market efficiency never be rejected on its own?",
                "options": [
                    "Every test also assumes a model of equilibrium expected returns",
                    "Prices are not observed often enough",
                    "Efficiency holds by definition",
                    "Statistical tests have no power for financial data"
                ],
                "correctExplanation": "A rejection may come from an inefficient market or from a wrong model of expected returns: every test is a joint test (Fama, 1991).",
                "incorrectExplanation": "The problem is not data frequency or power: to measure \"excess\" returns we need a model of normal returns, so efficiency is always tested jointly with that model."
            },
            "ro": {
                "title": "Problema ipotezei comune",
                "text": "De ce eficiența pieței nu poate fi respinsă niciodată singură?",
                "options": [
                    "Orice test presupune și un model al randamentelor așteptate de echilibru",
                    "Prețurile nu sînt observate suficient de des",
                    "Eficiența este adevărată prin definiție",
                    "Testele statistice nu au putere pentru datele financiare"
                ],
                "correctExplanation": "O respingere poate proveni dintr-o piață ineficientă sau dintr-un model greșit al randamentelor așteptate: orice test este un test comun (Fama, 1991).",
                "incorrectExplanation": "Problema nu este frecvența datelor sau puterea testelor: pentru a măsura randamentele „în exces” ne trebuie un model al randamentelor normale, deci eficiența se testează mereu împreună cu acel model."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Grossman–Stiglitz paradox",
                "text": "Information is costly. What happens if prices reflect all information perfectly?",
                "options": [
                    "Informed traders earn large excess returns",
                    "Prices become a random walk with zero drift",
                    "Trading volume rises without limit",
                    "Nobody is paid to collect information, so prices cannot reflect it"
                ],
                "correctExplanation": "With perfectly informative prices, collecting information earns nothing; nobody collects it, and prices cannot be fully informative: markets can only be nearly efficient.",
                "incorrectExplanation": "The paradox concerns the incentive to collect costly information: perfect efficiency would remove that incentive and therefore cannot be an equilibrium."
            },
            "ro": {
                "title": "Paradoxul Grossman–Stiglitz",
                "text": "Informația costă. Ce se întîmplă dacă prețurile reflectă perfect toată informația?",
                "options": [
                    "Investitorii informați obțin randamente mari în exces",
                    "Prețurile devin un mers aleator fără tendință",
                    "Volumul tranzacțiilor crește nelimitat",
                    "Nimeni nu este plătit pentru a colecta informația, deci prețurile nu o pot reflecta"
                ],
                "correctExplanation": "Cu prețuri perfect informative, colectarea informației nu aduce nimic; nimeni nu o mai colectează, iar prețurile nu pot fi pe deplin informative: piețele pot fi doar aproape eficiente.",
                "incorrectExplanation": "Paradoxul privește stimulentul de a colecta informație costisitoare: eficiența perfectă ar elimina acest stimulent, deci nu poate fi un echilibru."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Volatility clustering",
                "text": "Which random walk hypothesis allows volatility clustering?",
                "options": [
                    "RW1 (i.i.d. increments)",
                    "RW3 (uncorrelated increments)",
                    "RW2 (independent increments)",
                    "None of them"
                ],
                "correctExplanation": "RW3 only requires zero autocorrelation of the increments; their squares may be correlated, as in GARCH models.",
                "incorrectExplanation": "Under RW1 and RW2 the increments are independent, so large moves cannot predict large moves; only RW3 (and the martingale) allows it."
            },
            "ro": {
                "title": "Gruparea volatilității",
                "text": "Ce ipoteză de mers aleator permite gruparea volatilității?",
                "options": [
                    "RW1 (creșteri i.i.d.)",
                    "RW3 (creșteri necorelate)",
                    "RW2 (creșteri independente)",
                    "Niciuna"
                ],
                "correctExplanation": "RW3 cere doar autocorelație nulă a creșterilor; pătratele lor pot fi corelate, ca în modelele GARCH.",
                "incorrectExplanation": "În RW1 și RW2 creșterile sînt independente, deci mișcările mari nu pot anticipa mișcări mari; doar RW3 (și martingalul) permite acest lucru."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Martingale",
                "text": "Returns follow a GARCH model with a constant mean $\\mu$. Which statement is true?",
                "options": [
                    "Returns are i.i.d.",
                    "Returns minus $\\mu$ are a martingale difference, but not i.i.d.",
                    "Returns are predictable from their past values",
                    "Log prices are stationary"
                ],
                "correctExplanation": "In GARCH the conditional variance changes, but the conditional mean stays $\\mu$: the excess returns are an unpredictable martingale difference.",
                "incorrectExplanation": "GARCH returns are not i.i.d. (the variance depends on the past), their mean is unpredictable, and log prices still have a unit root."
            },
            "ro": {
                "title": "Martingal",
                "text": "Randamentele urmează un model GARCH cu media constantă $\\mu$. Care afirmație este adevărată?",
                "options": [
                    "Randamentele sînt i.i.d.",
                    "Randamentele minus $\\mu$ sînt o diferență de martingal, dar nu sînt i.i.d.",
                    "Randamentele pot fi anticipate din valorile lor trecute",
                    "Prețurile logaritmice sînt staționare"
                ],
                "correctExplanation": "În GARCH varianța condiționată se schimbă, dar media condiționată rămîne $\\mu$: randamentele în exces sînt o diferență de martingal imprevizibilă.",
                "incorrectExplanation": "Randamentele GARCH nu sînt i.i.d. (varianța depinde de trecut), media lor este imprevizibilă, iar prețurile logaritmice au tot o rădăcină unitară."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Square-root-of-time rule",
                "text": "Daily returns follow a random walk with standard deviation 1.2%. What is the standard deviation of a 25-day return?",
                "options": [
                    "1.2%",
                    "30%",
                    "2.4%",
                    "6.0%"
                ],
                "correctExplanation": "Variances add: $\\mathrm{sd} = 1.2\\% \\times \\sqrt{25} = 6.0\\%$.",
                "incorrectExplanation": "Under a random walk the variance grows linearly with the horizon, so the standard deviation grows with the square root: $1.2 \\times 5 = 6.0\\%$."
            },
            "ro": {
                "title": "Regula rădăcinii pătrate a timpului",
                "text": "Randamentele zilnice urmează un mers aleator cu abaterea standard 1,2%. Cît este abaterea standard a randamentului pe 25 de zile?",
                "options": [
                    "1,2%",
                    "30%",
                    "2,4%",
                    "6,0%"
                ],
                "correctExplanation": "Varianțele se adună: $\\mathrm{sd} = 1{,}2\\% \\times \\sqrt{25} = 6{,}0\\%$.",
                "incorrectExplanation": "Pentru un mers aleator varianța crește liniar cu orizontul, deci abaterea standard crește cu rădăcina pătrată: $1{,}2 \\times 5 = 6{,}0\\%$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "White noise",
                "text": "Which property is NOT required for a weak white noise?",
                "options": [
                    "Independence of the observations",
                    "Zero mean",
                    "Constant variance",
                    "Zero autocorrelation at every lag $k \\ne 0$"
                ],
                "correctExplanation": "White noise must be uncorrelated, not independent: GARCH returns are white noise but not independent.",
                "incorrectExplanation": "Zero mean, constant variance and zero autocorrelation define white noise; independence is the stronger i.i.d. property."
            },
            "ro": {
                "title": "Zgomot alb",
                "text": "Ce proprietate NU este cerută pentru un zgomot alb (slab)?",
                "options": [
                    "Independența observațiilor",
                    "Media zero",
                    "Varianța constantă",
                    "Autocorelația nulă la orice decalaj $k \\ne 0$"
                ],
                "correctExplanation": "Zgomotul alb trebuie să fie necorelat, nu independent: randamentele GARCH sînt zgomot alb, dar nu sînt independente.",
                "incorrectExplanation": "Media zero, varianța constantă și autocorelația nulă definesc zgomotul alb; independența este proprietatea mai puternică i.i.d."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ACF of an AR(1)",
                "text": "For the AR(1) process $x_t = 0.5\\,x_{t-1} + \\varepsilon_t$, what is $\\rho(2)$?",
                "options": [
                    "0.5",
                    "0",
                    "0.25",
                    "1"
                ],
                "correctExplanation": "For an AR(1), $\\rho(k) = \\alpha^k$, so $\\rho(2) = 0.5^2 = 0.25$.",
                "incorrectExplanation": "The ACF of an AR(1) decays geometrically: $\\rho(1) = 0.5$, $\\rho(2) = 0.25$; it is zero only for white noise or beyond the order of an MA process."
            },
            "ro": {
                "title": "ACF-ul unui AR(1)",
                "text": "Pentru procesul AR(1) $x_t = 0{,}5\\,x_{t-1} + \\varepsilon_t$, cît este $\\rho(2)$?",
                "options": [
                    "0,5",
                    "0",
                    "0,25",
                    "1"
                ],
                "correctExplanation": "Pentru un AR(1), $\\rho(k) = \\alpha^k$, deci $\\rho(2) = 0{,}5^2 = 0{,}25$.",
                "incorrectExplanation": "ACF-ul unui AR(1) scade geometric: $\\rho(1) = 0{,}5$, $\\rho(2) = 0{,}25$; este nul doar pentru zgomotul alb sau după ordinul unui proces MA."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ACF of an MA(1)",
                "text": "For the MA(1) process $x_t = \\varepsilon_t + 0.5\\,\\varepsilon_{t-1}$, which pair $(\\rho(1), \\rho(2))$ is right?",
                "options": [
                    "(0.4, 0)",
                    "(0.5, 0.25)",
                    "(0.5, 0)",
                    "(0.4, 0.16)"
                ],
                "correctExplanation": "$\\rho(1) = \\beta/(1 + \\beta^2) = 0.5/1.25 = 0.4$, and the ACF of an MA(1) is zero after lag 1.",
                "incorrectExplanation": "The MA(1) autocorrelation at lag 1 is $\\beta/(1+\\beta^2)$, not $\\beta$; beyond lag 1 the shocks no longer overlap, so $\\rho(2) = 0$."
            },
            "ro": {
                "title": "ACF-ul unui MA(1)",
                "text": "Pentru procesul MA(1) $x_t = \\varepsilon_t + 0{,}5\\,\\varepsilon_{t-1}$, ce pereche $(\\rho(1), \\rho(2))$ este corectă?",
                "options": [
                    "(0,4; 0)",
                    "(0,5; 0,25)",
                    "(0,5; 0)",
                    "(0,4; 0,16)"
                ],
                "correctExplanation": "$\\rho(1) = \\beta/(1 + \\beta^2) = 0{,}5/1{,}25 = 0{,}4$, iar ACF-ul unui MA(1) este nul după decalajul 1.",
                "incorrectExplanation": "Autocorelația unui MA(1) la decalajul 1 este $\\beta/(1+\\beta^2)$, nu $\\beta$; după decalajul 1 șocurile nu se mai suprapun, deci $\\rho(2) = 0$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Confidence band of the ACF",
                "text": "With $T = 2500$ i.i.d. returns, what is the 95% band for a sample autocorrelation?",
                "options": [
                    "$\\pm 0.0008$",
                    "$\\pm 0.196$",
                    "$\\pm 0.039$",
                    "$\\pm 0.05$"
                ],
                "correctExplanation": "$\\pm 1.96/\\sqrt{T} = \\pm 1.96/50 = \\pm 0.039$.",
                "incorrectExplanation": "Under i.i.d. returns $\\mathrm{Var}(\\hat\\rho(k)) \\approx 1/T$, so the band is $\\pm 1.96/\\sqrt{2500} = \\pm 0.039$; with volatility clustering the robust band is wider."
            },
            "ro": {
                "title": "Banda de încredere a ACF",
                "text": "Cu $T = 2500$ de randamente i.i.d., care este banda de 95% pentru o autocorelație de selecție?",
                "options": [
                    "$\\pm 0{,}0008$",
                    "$\\pm 0{,}196$",
                    "$\\pm 0{,}039$",
                    "$\\pm 0{,}05$"
                ],
                "correctExplanation": "$\\pm 1{,}96/\\sqrt{T} = \\pm 1{,}96/50 = \\pm 0{,}039$.",
                "incorrectExplanation": "Pentru randamente i.i.d., $\\mathrm{Var}(\\hat\\rho(k)) \\approx 1/T$, deci banda este $\\pm 1{,}96/\\sqrt{2500} = \\pm 0{,}039$; cu gruparea volatilității banda robustă este mai largă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Box–Pierce statistic",
                "text": "$T = 400$, $\\hat\\rho(1) = 0.10$, $\\hat\\rho(2) = 0$. What is $Q_{BP}(2)$?",
                "options": [
                    "0.4",
                    "4.0",
                    "40",
                    "0.04"
                ],
                "correctExplanation": "$Q_{BP}(2) = T\\sum \\hat\\rho(k)^2 = 400 \\times 0.01 = 4.0$, below $\\chi^2_{0.95}(2) = 5.99$.",
                "incorrectExplanation": "The Box–Pierce statistic multiplies the sum of squared autocorrelations by $T$: $400 \\times (0.10^2 + 0) = 4.0$."
            },
            "ro": {
                "title": "Statistica Box–Pierce",
                "text": "$T = 400$, $\\hat\\rho(1) = 0{,}10$, $\\hat\\rho(2) = 0$. Cît este $Q_{BP}(2)$?",
                "options": [
                    "0,4",
                    "4,0",
                    "40",
                    "0,04"
                ],
                "correctExplanation": "$Q_{BP}(2) = T\\sum \\hat\\rho(k)^2 = 400 \\times 0{,}01 = 4{,}0$, sub $\\chi^2_{0{,}95}(2) = 5{,}99$.",
                "incorrectExplanation": "Statistica Box–Pierce înmulțește suma pătratelor autocorelațiilor cu $T$: $400 \\times (0{,}10^2 + 0) = 4{,}0$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Classic against robust test",
                "text": "For DAX returns, Ljung–Box $Q(10)$ has p = 0.016, but the robust $\\tilde Q(10)$ has p = 0.57. What is the most likely reason?",
                "options": [
                    "The DAX has a unit root in its returns",
                    "The sample is too short",
                    "The robust test uses fewer lags",
                    "Volatility clustering makes $\\mathrm{Var}(\\hat\\rho(k))$ larger than $1/T$"
                ],
                "correctExplanation": "With volatility clustering the classic test uses too small a variance and rejects too often; the robust statistic corrects it.",
                "incorrectExplanation": "Both tests use the same 10 lags and the same long sample; the difference comes from the variance of the autocorrelations, which the classic test sets to $1/T$."
            },
            "ro": {
                "title": "Testul clasic față de testul robust",
                "text": "Pentru randamentele DAX, $Q(10)$ Ljung–Box are p = 0,016, dar $\\tilde Q(10)$ robust are p = 0,57. Care este cel mai probabil motiv?",
                "options": [
                    "DAX are o rădăcină unitară în randamente",
                    "Eșantionul este prea scurt",
                    "Testul robust folosește mai puține decalaje",
                    "Gruparea volatilității face ca $\\mathrm{Var}(\\hat\\rho(k))$ să fie mai mare decît $1/T$"
                ],
                "correctExplanation": "Cînd volatilitatea se grupează, testul clasic folosește o varianță prea mică și respinge prea des; statistica robustă corectează acest lucru.",
                "incorrectExplanation": "Ambele teste folosesc aceleași 10 decalaje și același eșantion lung; diferența vine din varianța autocorelațiilor, pe care testul clasic o fixează la $1/T$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Runs test",
                "text": "The signs of daily index returns show far fewer runs than expected under independence. What does this suggest?",
                "options": [
                    "Negative autocorrelation (reversals)",
                    "Heavy tails",
                    "A unit root in returns",
                    "Positive dependence: signs persist"
                ],
                "correctExplanation": "Few runs means long sequences of the same sign: rises follow rises, falls follow falls (as in the early BET).",
                "incorrectExplanation": "Too many runs would mean alternating signs (reversals); the runs test uses only signs, so it says nothing about tails or unit roots."
            },
            "ro": {
                "title": "Testul runs",
                "text": "Semnele randamentelor zilnice ale unui indice arată mult mai puține secvențe decît în cazul independenței. Ce sugerează acest lucru?",
                "options": [
                    "Autocorelație negativă (reveniri)",
                    "Cozi groase",
                    "O rădăcină unitară în randamente",
                    "Dependență pozitivă: semnele persistă"
                ],
                "correctExplanation": "Puține secvențe înseamnă șiruri lungi cu același semn: creșterile urmează după creșteri, scăderile după scăderi (ca la BET în primii ani).",
                "incorrectExplanation": "Prea multe secvențe ar însemna semne alternante (reveniri); testul runs folosește doar semnele, deci nu spune nimic despre cozi sau rădăcini unitare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Adaptive markets",
                "text": "What does the adaptive markets hypothesis (Lo, 2004) predict?",
                "options": [
                    "Markets are always perfectly efficient",
                    "Predictability varies over time as conditions and participants change",
                    "Returns are always predictable from past prices",
                    "Prices follow a stationary process"
                ],
                "correctExplanation": "Profit opportunities appear, are exploited and disappear: efficiency is a moving target, which rolling-window tests can show.",
                "incorrectExplanation": "The AMH lies between perfect efficiency and permanent predictability: it predicts time-varying predictability, not a fixed state."
            },
            "ro": {
                "title": "Piețe adaptive",
                "text": "Ce anticipează ipoteza piețelor adaptive (Lo, 2004)?",
                "options": [
                    "Piețele sînt mereu perfect eficiente",
                    "Previzibilitatea variază în timp, odată cu condițiile și participanții",
                    "Randamentele pot fi mereu anticipate din prețurile trecute",
                    "Prețurile urmează un proces staționar"
                ],
                "correctExplanation": "Oportunitățile de profit apar, sînt exploatate și dispar: eficiența se schimbă în timp, iar testele pe ferestre mobile pot arăta acest lucru.",
                "incorrectExplanation": "AMH se află între eficiența perfectă și previzibilitatea permanentă: anticipează o previzibilitate variabilă în timp, nu o stare fixă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Dickey–Fuller null hypothesis",
                "text": "In $\\Delta p_t = c + \\gamma\\,p_{t-1} + \\varepsilon_t$, what is the null hypothesis of the Dickey–Fuller test?",
                "options": [
                    "$\\gamma < 0$: the series is stationary",
                    "$c = 0$: no drift",
                    "$\\gamma = 0$: a unit root",
                    "$\\gamma = 1$: a unit root"
                ],
                "correctExplanation": "$\\gamma = \\phi - 1$; the unit root $\\phi = 1$ means $\\gamma = 0$, tested against $\\gamma < 0$.",
                "incorrectExplanation": "The regression is in differences, so a unit root corresponds to $\\gamma = 0$, not 1; stationarity ($\\gamma < 0$) is the alternative."
            },
            "ro": {
                "title": "Ipoteza nulă Dickey–Fuller",
                "text": "În $\\Delta p_t = c + \\gamma\\,p_{t-1} + \\varepsilon_t$, care este ipoteza nulă a testului Dickey–Fuller?",
                "options": [
                    "$\\gamma < 0$: seria este staționară",
                    "$c = 0$: fără tendință",
                    "$\\gamma = 0$: rădăcină unitară",
                    "$\\gamma = 1$: rădăcină unitară"
                ],
                "correctExplanation": "$\\gamma = \\phi - 1$; rădăcina unitară $\\phi = 1$ înseamnă $\\gamma = 0$, testată față de $\\gamma < 0$.",
                "incorrectExplanation": "Regresia este în diferențe, deci rădăcina unitară corespunde lui $\\gamma = 0$, nu lui 1; staționaritatea ($\\gamma < 0$) este ipoteza alternativă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Dickey–Fuller critical values",
                "text": "Why is $\\tau$ compared with $-2.86$ (constant) instead of $-1.645$?",
                "options": [
                    "Under the null, $\\tau$ does not follow a $t$ or Normal law; its distribution is shifted to the left",
                    "Because the test is two-sided",
                    "Because returns have heavy tails",
                    "Because the sample is small"
                ],
                "correctExplanation": "With a unit root the regressor $p_{t-1}$ is non-stationary and $\\tau$ has the Dickey–Fuller distribution, tabulated by MacKinnon (1996).",
                "incorrectExplanation": "The test is one-sided and the shift holds even in large samples with Normal shocks: it comes from the non-stationary regressor."
            },
            "ro": {
                "title": "Valorile critice Dickey–Fuller",
                "text": "De ce $\\tau$ se compară cu $-2{,}86$ (constantă) și nu cu $-1{,}645$?",
                "options": [
                    "În ipoteza nulă, $\\tau$ nu urmează o lege $t$ sau Normală; distribuția lui este deplasată spre stînga",
                    "Pentru că testul este bilateral",
                    "Pentru că randamentele au cozi groase",
                    "Pentru că eșantionul este mic"
                ],
                "correctExplanation": "Cu o rădăcină unitară, regresorul $p_{t-1}$ este nestaționar, iar $\\tau$ are distribuția Dickey–Fuller, tabelată de MacKinnon (1996).",
                "incorrectExplanation": "Testul este unilateral, iar deplasarea apare și în eșantioane mari cu șocuri Normale: provine din regresorul nestaționar."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "KPSS test",
                "text": "What is the null hypothesis of the KPSS test?",
                "options": [
                    "A unit root",
                    "White noise",
                    "No autocorrelation up to lag 10",
                    "Stationarity (around a constant or a trend)"
                ],
                "correctExplanation": "KPSS reverses the roles: $H_0$ is stationarity and a large statistic rejects it in favour of a unit root.",
                "incorrectExplanation": "ADF and Phillips–Perron have a unit root as null; KPSS was built to test stationarity as the null, so the two can be combined."
            },
            "ro": {
                "title": "Testul KPSS",
                "text": "Care este ipoteza nulă a testului KPSS?",
                "options": [
                    "O rădăcină unitară",
                    "Zgomot alb",
                    "Nicio autocorelație pînă la decalajul 10",
                    "Staționaritatea (în jurul unei constante sau al unei tendințe)"
                ],
                "correctExplanation": "KPSS inversează rolurile: $H_0$ este staționaritatea, iar o statistică mare o respinge în favoarea unei rădăcini unitare.",
                "incorrectExplanation": "ADF și Phillips–Perron au ca ipoteză nulă rădăcina unitară; KPSS a fost construit pentru a testa staționaritatea ca ipoteză nulă, astfel încît cele două să poată fi combinate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Combining ADF and KPSS",
                "text": "For a log price, ADF does not reject and KPSS rejects. What do you conclude?",
                "options": [
                    "The series is stationary",
                    "The series is $I(1)$",
                    "The result is inconclusive",
                    "The returns are $I(1)$"
                ],
                "correctExplanation": "No evidence against a unit root and evidence against stationarity: the log price is integrated of order 1.",
                "incorrectExplanation": "The two tests agree here; the result would be inconclusive only if both rejected or neither rejected."
            },
            "ro": {
                "title": "Combinarea ADF și KPSS",
                "text": "Pentru un preț logaritmic, ADF nu respinge, iar KPSS respinge. Ce concluzionați?",
                "options": [
                    "Seria este staționară",
                    "Seria este $I(1)$",
                    "Rezultatul este neconcludent",
                    "Randamentele sînt $I(1)$"
                ],
                "correctExplanation": "Nicio dovadă împotriva rădăcinii unitare și dovezi împotriva staționarității: prețul logaritmic este integrat de ordinul 1.",
                "incorrectExplanation": "Cele două teste sînt de acord aici; rezultatul ar fi neconcludent doar dacă ambele ar respinge sau niciunul nu ar respinge."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Unit root and efficiency",
                "text": "Which statement about a unit root in log prices is correct?",
                "options": [
                    "It proves weak-form efficiency",
                    "It contradicts weak-form efficiency",
                    "It is necessary for a random walk but does not prove weak-form efficiency",
                    "It implies that returns are i.i.d."
                ],
                "correctExplanation": "A process such as $r_t = 0.3\\,r_{t-1} + \\varepsilon_t$ makes prices $I(1)$ with predictable returns: efficiency must be tested on the returns.",
                "incorrectExplanation": "Unit-root tests look at the level of prices; predictability concerns the increments, which need ACF or variance-ratio tests."
            },
            "ro": {
                "title": "Rădăcină unitară și eficiență",
                "text": "Care afirmație despre o rădăcină unitară în prețurile logaritmice este corectă?",
                "options": [
                    "Dovedește eficiența în formă slabă",
                    "Contrazice eficiența în formă slabă",
                    "Este necesară pentru un mers aleator, dar nu dovedește eficiența în formă slabă",
                    "Implică randamente i.i.d."
                ],
                "correctExplanation": "Un proces precum $r_t = 0{,}3\\,r_{t-1} + \\varepsilon_t$ face prețurile $I(1)$, cu randamente previzibile: eficiența trebuie testată pe randamente.",
                "incorrectExplanation": "Testele de rădăcină unitară privesc nivelul prețurilor; previzibilitatea privește creșterile, care cer teste ACF sau variance ratio."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Spurious regression",
                "text": "You regress one random walk on another, independent one. What typically happens?",
                "options": [
                    "The slope looks significant ($|t| > 1.96$) in most samples, although there is no relation",
                    "The slope is never significant",
                    "The $R^2$ is always exactly zero",
                    "The residuals are white noise"
                ],
                "correctExplanation": "Granger and Newbold (1974): with independent random walks the $t$-test rejects far too often and the $R^2$ is far from zero.",
                "incorrectExplanation": "Both series wander, so they often trend together by chance; the residuals inherit the unit root and the usual $t$-test is invalid."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Estimați regresia unui mers aleator pe altul, independent. Ce se întîmplă de obicei?",
                "options": [
                    "Panta pare semnificativă ($|t| > 1{,}96$) în majoritatea eșantioanelor, deși nu există nicio legătură",
                    "Panta nu este niciodată semnificativă",
                    "$R^2$ este mereu exact zero",
                    "Reziduurile sînt zgomot alb"
                ],
                "correctExplanation": "Granger și Newbold (1974): cu mersuri aleatoare independente, testul $t$ respinge mult prea des, iar $R^2$ este departe de zero.",
                "incorrectExplanation": "Ambele serii rătăcesc, deci adesea au din întîmplare tendințe comune; reziduurile moștenesc rădăcina unitară, iar testul $t$ obișnuit nu este valid."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Variance ratio from the ACF",
                "text": "Returns have $\\rho(1) = 0.2$ and $\\rho(k) = 0$ for $k \\ge 2$. What is VR(2)?",
                "options": [
                    "0.8",
                    "1",
                    "1.4",
                    "1.2"
                ],
                "correctExplanation": "$\\mathrm{VR}(2) = 1 + \\rho(1) = 1.2$: the two-day variance is 20% larger than under a random walk.",
                "incorrectExplanation": "With $q = 2$ the formula $1 + 2\\sum(1 - k/q)\\rho(k)$ has one term with weight $2(1 - 1/2) = 1$, so VR(2) $= 1 + 0.2$."
            },
            "ro": {
                "title": "Raportul varianțelor din ACF",
                "text": "Randamentele au $\\rho(1) = 0{,}2$ și $\\rho(k) = 0$ pentru $k \\ge 2$. Cît este VR(2)?",
                "options": [
                    "0,8",
                    "1",
                    "1,4",
                    "1,2"
                ],
                "correctExplanation": "$\\mathrm{VR}(2) = 1 + \\rho(1) = 1{,}2$: varianța pe două zile este cu 20% mai mare decît pentru un mers aleator.",
                "incorrectExplanation": "Cu $q = 2$, formula $1 + 2\\sum(1 - k/q)\\rho(k)$ are un singur termen, cu ponderea $2(1 - 1/2) = 1$, deci VR(2) $= 1 + 0{,}2$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Reading a variance ratio",
                "text": "An estimated VR(5) = 0.86 is significantly below 1. What does it indicate?",
                "options": [
                    "Momentum: moves tend to continue",
                    "Mean reversion: daily moves are partly reversed within the week",
                    "A unit root in returns",
                    "Higher volatility at long horizons"
                ],
                "correctExplanation": "VR < 1 means negative autocorrelation: the weekly variance is smaller than five daily variances (as for the S&P 500).",
                "incorrectExplanation": "Momentum would give VR > 1; a variance ratio below 1 means that multi-day returns vary less than the square-root-of-time rule implies."
            },
            "ro": {
                "title": "Interpretarea raportului varianțelor",
                "text": "Un VR(5) estimat de 0,86 este semnificativ sub 1. Ce indică?",
                "options": [
                    "Momentum: mișcările tind să continue",
                    "Revenire la medie: mișcările zilnice sînt parțial anulate în aceeași săptămînă",
                    "O rădăcină unitară în randamente",
                    "O volatilitate mai mare pe orizonturi lungi"
                ],
                "correctExplanation": "VR < 1 înseamnă autocorelație negativă: varianța săptămînală este mai mică decît cinci varianțe zilnice (ca la S&P 500).",
                "incorrectExplanation": "Momentumul ar da VR > 1; un raport al varianțelor sub 1 înseamnă că randamentele pe mai multe zile variază mai puțin decît spune regula rădăcinii pătrate a timpului."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Homoskedastic against robust VR test",
                "text": "In a simulation with uncorrelated GARCH returns, $Z(2)$ rejects about 17% of the time at 5%, $Z^*(2)$ about 5%. What follows?",
                "options": [
                    "Use $Z^*$ to test the martingale hypothesis",
                    "GARCH returns are predictable",
                    "Use $Z$ because it has more power",
                    "The variance ratio is biased for GARCH returns"
                ],
                "correctExplanation": "The null is true in the simulation; $Z$ rejects too often because it assumes constant variance; $Z^*$ keeps the right size.",
                "incorrectExplanation": "Rejecting a true null 17% of the time is a size distortion, not power; GARCH returns are uncorrelated, so nothing is predictable."
            },
            "ro": {
                "title": "Testul VR homoscedastic față de cel robust",
                "text": "Într-o simulare cu randamente GARCH necorelate, $Z(2)$ respinge în circa 17% din cazuri la 5%, iar $Z^*(2)$ în circa 5%. Ce rezultă?",
                "options": [
                    "Folosim $Z^*$ pentru a testa ipoteza de martingal",
                    "Randamentele GARCH sînt previzibile",
                    "Folosim $Z$, pentru că are mai multă putere",
                    "Raportul varianțelor este deplasat pentru randamente GARCH"
                ],
                "correctExplanation": "Ipoteza nulă este adevărată în simulare; $Z$ respinge prea des pentru că presupune varianță constantă; $Z^*$ păstrează mărimea corectă a testului.",
                "incorrectExplanation": "Respingerea unei ipoteze nule adevărate în 17% din cazuri este o distorsiune a mărimii testului, nu putere; randamentele GARCH sînt necorelate, deci nimic nu este previzibil."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Chow–Denning test",
                "text": "You test the random walk at $q = 2, 5, 10, 20$ together with the Chow–Denning test. What is the decision rule at 5%?",
                "options": [
                    "Reject if any $|Z^*(q)| > 1.96$",
                    "Reject if the average $Z^*(q)$ exceeds 1.96",
                    "Reject if $\\max_q |Z^*(q)|$ exceeds about 2.49",
                    "Reject if all four $|Z^*(q)| > 1.96$"
                ],
                "correctExplanation": "The maximum of four statistics needs a larger critical value: $(2\\Phi(c) - 1)^4 = 0.95$ gives $c \\approx 2.49$.",
                "incorrectExplanation": "Using 1.96 for each of four horizons raises the chance of a false rejection to about 18.5%; the Chow–Denning rule controls it at 5%."
            },
            "ro": {
                "title": "Testul Chow–Denning",
                "text": "Testați mersul aleator la $q = 2, 5, 10, 20$ împreună, cu testul Chow–Denning. Care este regula de decizie la 5%?",
                "options": [
                    "Respingem dacă oricare $|Z^*(q)| > 1{,}96$",
                    "Respingem dacă media lui $Z^*(q)$ depășește 1,96",
                    "Respingem dacă $\\max_q |Z^*(q)|$ depășește circa 2,49",
                    "Respingem dacă toate cele patru $|Z^*(q)| > 1{,}96$"
                ],
                "correctExplanation": "Maximul a patru statistici cere o valoare critică mai mare: $(2\\Phi(c) - 1)^4 = 0{,}95$ dă $c \\approx 2{,}49$.",
                "incorrectExplanation": "Folosind 1,96 pentru fiecare dintre cele patru orizonturi, șansa unei respingeri false crește la circa 18,5%; regula Chow–Denning o menține la 5%."
            }
        }
    ]
};
