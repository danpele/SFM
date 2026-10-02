// ============================================================
// Chapter 6 quiz bank: model selection and risk management (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['model-selection'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "The AIC formula",
                "text": "A model with $k$ parameters has maximised log-likelihood $\\ell(\\hat\\theta)$. What is its AIC?",
                "options": [
                    "$-2\\ell(\\hat\\theta) + 2k$",
                    "$-2\\ell(\\hat\\theta) + k\\ln n$",
                    "$2\\ell(\\hat\\theta) - 2k$",
                    "$\\ell(\\hat\\theta) - k$"
                ],
                "correctExplanation": "AIC $= -2\\ell(\\hat\\theta) + 2k$; lower is better.",
                "incorrectExplanation": "AIC $= -2\\ell(\\hat\\theta) + 2k$; the version with $k\\ln n$ is BIC."
            },
            "ro": {
                "title": "Formula AIC",
                "text": "Un model cu $k$ parametri are log-verosimilitatea maximizată $\\ell(\\hat\\theta)$. Cît este AIC?",
                "options": [
                    "$-2\\ell(\\hat\\theta) + 2k$",
                    "$-2\\ell(\\hat\\theta) + k\\ln n$",
                    "$2\\ell(\\hat\\theta) - 2k$",
                    "$\\ell(\\hat\\theta) - k$"
                ],
                "correctExplanation": "AIC $= -2\\ell(\\hat\\theta) + 2k$; valoarea mai mică este preferată.",
                "incorrectExplanation": "AIC $= -2\\ell(\\hat\\theta) + 2k$; varianta cu $k\\ln n$ este BIC."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The BIC penalty",
                "text": "With $n = 6\\,714$ daily returns, how large is the BIC penalty per parameter compared with AIC?",
                "options": [
                    "The same, 2 per parameter",
                    "About $8.81$ per parameter, more than four times the AIC penalty",
                    "Smaller than 2, because $\\ln n < 2$",
                    "Zero: BIC does not penalise parameters"
                ],
                "correctExplanation": "BIC adds $\\ln n = 8.81$ per parameter, AIC adds 2.",
                "incorrectExplanation": "BIC penalises each parameter by $\\ln n$, here $8.81$, against 2 for AIC; for $n > 7$ BIC is always stricter."
            },
            "ro": {
                "title": "Penalizarea BIC",
                "text": "Cu $n = 6\\,714$ de randamente zilnice, cît este penalizarea BIC pe parametru față de AIC?",
                "options": [
                    "Aceeași, 2 pe parametru",
                    "Circa $8{,}81$ pe parametru, de peste patru ori penalizarea AIC",
                    "Mai mică decît 2, pentru că $\\ln n < 2$",
                    "Zero: BIC nu penalizează parametrii"
                ],
                "correctExplanation": "BIC adaugă $\\ln n = 8{,}81$ pe parametru, AIC adaugă 2.",
                "incorrectExplanation": "BIC penalizează fiecare parametru cu $\\ln n$, aici $8{,}81$, față de 2 la AIC; pentru $n > 7$, BIC este întotdeauna mai strict."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "A small $\\Delta$AIC",
                "text": "Model B has an AIC 1.5 points above model A on the same data. What is the right conclusion?",
                "options": [
                    "B is significantly worse at 5%",
                    "A is the true model",
                    "Both models remain plausible; the data barely separate them",
                    "B must be discarded"
                ],
                "correctExplanation": "A gap below 2 means substantial support for both models (Burnham and Anderson).",
                "incorrectExplanation": "AIC is not a test: a $\\Delta$ below 2 is essentially a tie, with substantial support for both models."
            },
            "ro": {
                "title": "Un $\\Delta$AIC mic",
                "text": "Modelul B are un AIC cu 1,5 puncte mai mare decît modelul A pe aceleași date. Care este concluzia corectă?",
                "options": [
                    "B este semnificativ mai slab la 5%",
                    "A este modelul adevărat",
                    "Ambele modele rămîn plauzibile; datele abia le deosebesc",
                    "B trebuie eliminat"
                ],
                "correctExplanation": "O diferență sub 2 înseamnă sprijin substanțial pentru ambele modele (Burnham și Anderson).",
                "incorrectExplanation": "AIC nu este un test: un $\\Delta$ sub 2 este practic o egalitate, cu sprijin substanțial pentru ambele modele."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Akaike weights",
                "text": "Two candidate models have $\\Delta = 0$ and $\\Delta = 2$. What are their Akaike weights?",
                "options": [
                    "0.5 and 0.5",
                    "1 and 0",
                    "0.9 and 0.1",
                    "About 0.73 and 0.27"
                ],
                "correctExplanation": "$w = e^{-\\Delta/2}/\\sum e^{-\\Delta/2}$: $1/(1 + e^{-1}) \\approx 0.73$ and $0.27$.",
                "incorrectExplanation": "The weights are $e^{-\\Delta/2}$ normalised to sum to 1: $1/(1 + e^{-1}) \\approx 0.73$ and $e^{-1}/(1 + e^{-1}) \\approx 0.27$."
            },
            "ro": {
                "title": "Ponderile Akaike",
                "text": "Două modele candidate au $\\Delta = 0$ și $\\Delta = 2$. Care sînt ponderile lor Akaike?",
                "options": [
                    "0,5 și 0,5",
                    "1 și 0",
                    "0,9 și 0,1",
                    "Circa 0,73 și 0,27"
                ],
                "correctExplanation": "$w = e^{-\\Delta/2}/\\sum e^{-\\Delta/2}$: $1/(1 + e^{-1}) \\approx 0{,}73$ și $0{,}27$.",
                "incorrectExplanation": "Ponderile sînt $e^{-\\Delta/2}$ normalizate să însumeze 1: $1/(1 + e^{-1}) \\approx 0{,}73$ și $e^{-1}/(1 + e^{-1}) \\approx 0{,}27$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Testing the skewness parameter",
                "text": "To test the Student-t ($\\lambda = 0$) inside the skewed-t, which reference law does the likelihood-ratio statistic follow under $H_0$?",
                "options": [
                    "$\\chi^2(1)$, critical value 3.84 at 5%",
                    "$\\chi^2(4)$",
                    "$N(0, 1)$",
                    "A 50:50 mixture of 0 and $\\chi^2(1)$"
                ],
                "correctExplanation": "One restriction ($\\lambda = 0$) inside the parameter space: Wilks gives $\\chi^2(1)$.",
                "incorrectExplanation": "The restriction $\\lambda = 0$ is one parameter and lies inside $(-1, 1)$, so Wilks' theorem gives $\\chi^2(1)$ with critical value 3.84."
            },
            "ro": {
                "title": "Testarea parametrului de asimetrie",
                "text": "Testăm Student-t ($\\lambda = 0$) ca submodel al skewed-t. Ce distribuție are statistica raportului de verosimilitate sub $H_0$?",
                "options": [
                    "$\\chi^2(1)$, valoarea critică 3,84 la 5%",
                    "$\\chi^2(4)$",
                    "$N(0, 1)$",
                    "Un amestec 50:50 de 0 și $\\chi^2(1)$"
                ],
                "correctExplanation": "O restricție ($\\lambda = 0$) în interiorul spațiului parametrilor: Wilks dă $\\chi^2(1)$.",
                "incorrectExplanation": "Restricția $\\lambda = 0$ este un singur parametru și se află în interiorul lui $(-1, 1)$, deci teorema lui Wilks dă $\\chi^2(1)$, cu valoarea critică 3,84."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "An LR test by hand",
                "text": "Nested models with one extra parameter: $\\ell_0 = -1000$, $\\ell_1 = -997$. Decision at 5%?",
                "options": [
                    "$LR = 3$, do not reject",
                    "$LR = 6 > 3.84$, reject the restricted model",
                    "$LR = 1.5$, do not reject",
                    "$LR = -6$, the test is not defined"
                ],
                "correctExplanation": "$LR = 2(\\ell_1 - \\ell_0) = 6 > 3.84$.",
                "incorrectExplanation": "The statistic is twice the gain in log-likelihood: $LR = 2(-997 + 1000) = 6$, above the $\\chi^2(1)$ critical value 3.84."
            },
            "ro": {
                "title": "Un test LR pas cu pas",
                "text": "Modele imbricate cu un parametru în plus: $\\ell_0 = -1000$, $\\ell_1 = -997$. Decizia la 5%?",
                "options": [
                    "$LR = 3$, nu respingem",
                    "$LR = 6 > 3{,}84$, respingem modelul restrîns",
                    "$LR = 1{,}5$, nu respingem",
                    "$LR = -6$, testul nu este definit"
                ],
                "correctExplanation": "$LR = 2(\\ell_1 - \\ell_0) = 6 > 3{,}84$.",
                "incorrectExplanation": "Statistica este dublul cîștigului de log-verosimilitate: $LR = 2(-997 + 1000) = 6$, peste valoarea critică $\\chi^2(1)$ de 3,84."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Normal inside Student-t",
                "text": "Why is the usual $\\chi^2(1)$ test not exact for the Normal law inside the Student-t?",
                "options": [
                    "Because the two models are not nested",
                    "Because the Student-t has three parameters",
                    "Because $1/\\nu = 0$ lies on the boundary of the parameter space, so the null law is a 50:50 mixture of 0 and $\\chi^2(1)$",
                    "Because returns are not Normal"
                ],
                "correctExplanation": "On the boundary the null law is $\\frac12\\chi^2(0) + \\frac12\\chi^2(1)$ (Self and Liang, 1987); the 5% critical value is 2.71.",
                "incorrectExplanation": "The models are nested, but the restriction $1/\\nu = 0$ is on the edge of $1/\\nu \\ge 0$; the null law becomes a 50:50 mixture and the 5% critical value is 2.71."
            },
            "ro": {
                "title": "Normală în Student-t",
                "text": "De ce nu este exact testul $\\chi^2(1)$ obișnuit pentru distribuția Normală ca submodel al Student-t?",
                "options": [
                    "Pentru că cele două modele nu sînt imbricate",
                    "Pentru că Student-t are trei parametri",
                    "Pentru că $1/\\nu = 0$ este pe frontiera spațiului parametrilor, deci distribuția statisticii sub $H_0$ este un amestec 50:50 de 0 și $\\chi^2(1)$",
                    "Pentru că randamentele nu sînt Normale"
                ],
                "correctExplanation": "La frontieră, distribuția statisticii sub $H_0$ este $\\frac12\\chi^2(0) + \\frac12\\chi^2(1)$ (Self și Liang, 1987); valoarea critică la 5% este 2,71.",
                "incorrectExplanation": "Modelele sînt imbricate, dar restricția $1/\\nu = 0$ este pe frontiera mulțimii $1/\\nu \\ge 0$; distribuția statisticii sub $H_0$ devine un amestec 50:50, iar valoarea critică la 5% este 2,71."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Non-nested models",
                "text": "How do we compare the Student-t with the GED, two models that are not nested?",
                "options": [
                    "With a likelihood-ratio test and $\\chi^2(0)$",
                    "With a $t$-test on the means",
                    "It is impossible",
                    "With AIC or the Vuong test"
                ],
                "correctExplanation": "Non-nested pairs: AIC or the Vuong (1989) test on the pointwise log-likelihood differences.",
                "incorrectExplanation": "The LR test needs one model to be a special case of the other; for non-nested models use AIC or the Vuong test."
            },
            "ro": {
                "title": "Modele neimbricate",
                "text": "Cum comparăm Student-t cu GED, două modele care nu sînt imbricate?",
                "options": [
                    "Cu un test al raportului de verosimilitate și $\\chi^2(0)$",
                    "Cu un test $t$ pe medii",
                    "Este imposibil",
                    "Cu AIC sau cu testul Vuong"
                ],
                "correctExplanation": "Perechi neimbricate: AIC sau testul Vuong (1989) pe diferențele punctuale de log-verosimilitate.",
                "incorrectExplanation": "Testul LR cere ca un model să fie caz particular al celuilalt; pentru modele neimbricate folosim AIC sau testul Vuong."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The KS statistic",
                "text": "What does the Kolmogorov–Smirnov statistic $D_n$ measure?",
                "options": [
                    "The largest vertical distance between the empirical and the model distribution functions",
                    "The squared distance integrated over all $x$",
                    "The difference between the sample and the model means",
                    "The number of points outside a QQ band"
                ],
                "correctExplanation": "$D_n = \\sup_x |F_n(x) - F(x)|$.",
                "incorrectExplanation": "KS takes the maximum of $|F_n(x) - F(x)|$; integrating the squared gap gives Cramér–von Mises."
            },
            "ro": {
                "title": "Statistica KS",
                "text": "Ce măsoară statistica Kolmogorov–Smirnov $D_n$?",
                "options": [
                    "Cea mai mare distanță pe verticală dintre funcția de repartiție empirică și cea a modelului",
                    "Pătratul distanței integrat pe toate valorile $x$",
                    "Diferența dintre media de selecție și media modelului",
                    "Numărul de puncte din afara unei benzi QQ"
                ],
                "correctExplanation": "$D_n = \\sup_x |F_n(x) - F(x)|$.",
                "incorrectExplanation": "KS ia maximul lui $|F_n(x) - F(x)|$; integrarea pătratului distanței dă Cramér–von Mises."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimated parameters",
                "text": "You fit $\\mu$ and $\\sigma$ to the data and then run KS against $N(\\hat\\mu, \\hat\\sigma^2)$ with the standard KS table. What happens?",
                "options": [
                    "The test rejects too often",
                    "The test rejects too rarely: the critical values are too large (use Lilliefors or a parametric bootstrap)",
                    "Nothing: the table is still exact",
                    "The statistic becomes negative"
                ],
                "correctExplanation": "The fitted $F$ is pulled towards the data, so $D_n$ is small; the correct (Lilliefors) critical values are smaller.",
                "incorrectExplanation": "Fitting on the same data makes $D_n$ smaller than under a known $F$; the standard table is too lenient, so use Lilliefors tables or a parametric bootstrap."
            },
            "ro": {
                "title": "Parametri estimați",
                "text": "Estimați $\\mu$ și $\\sigma$ din date și apoi aplicați KS față de $N(\\hat\\mu, \\hat\\sigma^2)$ cu tabelul KS standard. Ce se întîmplă?",
                "options": [
                    "Testul respinge prea des",
                    "Testul respinge prea rar: valorile critice sînt prea mari (folosiți Lilliefors sau un bootstrap parametric)",
                    "Nimic: tabelul rămîne exact",
                    "Statistica devine negativă"
                ],
                "correctExplanation": "$F$ estimată este apropiată de date, deci $D_n$ este mai mic; valorile critice corecte (Lilliefors) sînt mai mici.",
                "incorrectExplanation": "Estimarea pe aceleași date face $D_n$ mai mic decît pentru o $F$ cunoscută; tabelul standard este prea îngăduitor, deci folosiți tabelele Lilliefors sau un bootstrap parametric."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "KS and the tails",
                "text": "Why does the KS test detect badly fitted tails poorly?",
                "options": [
                    "Because it uses only the median",
                    "Because it needs grouped data",
                    "Because in the tails $F_n$ and $F$ are both close to 0 or 1, so their gap is small even when the model is wrong",
                    "Because it is a parametric test"
                ],
                "correctExplanation": "The raw gap is bounded by how close both functions are to 0 or 1; the maximum is usually reached near the centre.",
                "incorrectExplanation": "In the tails both distribution functions are near 0 or 1, so the unweighted gap stays small; the KS maximum is usually near the centre."
            },
            "ro": {
                "title": "KS și cozile",
                "text": "De ce detectează testul KS greu cozile prost ajustate?",
                "options": [
                    "Pentru că folosește doar mediana",
                    "Pentru că cere date grupate",
                    "Pentru că în cozi $F_n$ și $F$ sînt amîndouă aproape de 0 sau 1, deci distanța dintre ele este mică chiar dacă modelul greșește",
                    "Pentru că este un test parametric"
                ],
                "correctExplanation": "Distanța brută este limitată de cît de aproape sînt ambele funcții de 0 sau 1; maximul apare de obicei în jurul centrului.",
                "incorrectExplanation": "În cozi, ambele funcții de repartiție sînt aproape de 0 sau 1, deci distanța neponderată rămîne mică; maximul KS apare de obicei aproape de centru."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Anderson–Darling weight",
                "text": "Which weight makes the Anderson–Darling statistic sensitive to the tails?",
                "options": [
                    "$F(x)(1 - F(x))$",
                    "$1$",
                    "$x^2$",
                    "$1/[F(x)(1 - F(x))]$"
                ],
                "correctExplanation": "Dividing the squared gap by $F(1 - F)$ inflates it where $F$ is near 0 or 1.",
                "incorrectExplanation": "AD integrates $[F_n - F]^2/[F(1 - F)]$; the weight is largest in the tails. Weight 1 gives Cramér–von Mises."
            },
            "ro": {
                "title": "Ponderea Anderson–Darling",
                "text": "Ce pondere face statistica Anderson–Darling sensibilă la cozi?",
                "options": [
                    "$F(x)(1 - F(x))$",
                    "$1$",
                    "$x^2$",
                    "$1/[F(x)(1 - F(x))]$"
                ],
                "correctExplanation": "Împărțirea pătratului distanței la $F(1 - F)$ îl mărește unde $F$ este aproape de 0 sau 1.",
                "incorrectExplanation": "AD integrează $[F_n - F]^2/[F(1 - F)]$; ponderea este cea mai mare în cozi. Ponderea 1 dă Cramér–von Mises."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Large samples",
                "text": "With more than 6000 daily returns, every candidate model is rejected by the EDF tests. What is a sensible use of the tests?",
                "options": [
                    "Rank the models by their statistics and look at QQ plots to see where each fails",
                    "Conclude that no model can be used",
                    "Switch to a 10% level so that some model passes",
                    "Keep the Normal model, since all are rejected anyway"
                ],
                "correctExplanation": "Large $n$ detects tiny departures; the statistics still order the models, and the plots show the location of the misfit.",
                "incorrectExplanation": "With large samples rejection is almost certain; use the statistics to compare models and the QQ plot to judge the region that matters."
            },
            "ro": {
                "title": "Eșantioane mari",
                "text": "Cu peste 6000 de randamente zilnice, fiecare model candidat este respins de testele EDF. Care este o folosire rezonabilă a testelor?",
                "options": [
                    "Ordonăm modelele după statistici și privim graficele QQ ca să vedem unde eșuează fiecare",
                    "Concluzionăm că niciun model nu poate fi folosit",
                    "Trecem la nivelul de 10% ca să treacă vreun model",
                    "Păstrăm modelul Normal, oricum toate sînt respinse"
                ],
                "correctExplanation": "Un $n$ mare detectează abateri minuscule; statisticile permit totuși ordonarea modelelor, iar graficele arată unde modelul nu se potrivește.",
                "incorrectExplanation": "Cu eșantioane mari respingerea este aproape sigură; folosiți statisticile pentru a compara modelele și graficul QQ pentru a judeca zona care contează."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Parametric bootstrap",
                "text": "How is a parametric bootstrap $p$-value for an EDF statistic computed?",
                "options": [
                    "Resample the observed returns with replacement and recompute the statistic without refitting",
                    "Simulate samples from the fitted model, refit each one, recompute the statistic, and count how often it exceeds the observed one",
                    "Use the asymptotic KS table",
                    "Divide the statistic by $\\sqrt{n}$"
                ],
                "correctExplanation": "Simulate from $F(\\cdot; \\hat\\theta)$, refit, recompute; $p = (1 + \\#\\{\\text{simulated} \\ge \\text{observed}\\})/(B + 1)$.",
                "incorrectExplanation": "The refit step reproduces the effect of estimating the parameters; resampling the data without refitting does not."
            },
            "ro": {
                "title": "Bootstrap parametric",
                "text": "Cum se calculează o valoare $p$ prin bootstrap parametric pentru o statistică EDF?",
                "options": [
                    "Reeșantionăm randamentele observate cu întoarcere și recalculăm statistica fără reestimare",
                    "Simulăm eșantioane din modelul estimat, reestimăm modelul pe fiecare eșantion, recalculăm statistica și numărăm de cîte ori depășește valoarea observată",
                    "Folosim tabelul KS asimptotic",
                    "Împărțim statistica la $\\sqrt{n}$"
                ],
                "correctExplanation": "Simulăm din $F(\\cdot; \\hat\\theta)$, reestimăm, recalculăm; $p = (1 + \\#\\{\\text{simulate} \\ge \\text{observată}\\})/(B + 1)$.",
                "incorrectExplanation": "Pasul de reestimare reproduce efectul estimării parametrilor; reeșantionarea datelor fără reestimare nu îl reproduce."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "PP or QQ?",
                "text": "Which plot shows best whether a model gets the extreme losses right?",
                "options": [
                    "The PP plot",
                    "A histogram with 10 bins",
                    "The QQ plot",
                    "A time-series plot of returns"
                ],
                "correctExplanation": "QQ plots work in units of returns and stretch the tails; PP plots squeeze the tails into the corners.",
                "incorrectExplanation": "PP plots compare probabilities in $[0, 1]$ and hide the tails; QQ plots compare quantiles, so each extreme day is visible."
            },
            "ro": {
                "title": "PP sau QQ?",
                "text": "Ce grafic arată cel mai bine dacă un model descrie corect pierderile extreme?",
                "options": [
                    "Graficul PP",
                    "O histogramă cu 10 clase",
                    "Graficul QQ",
                    "Graficul în timp al randamentelor"
                ],
                "correctExplanation": "Graficele QQ lucrează în unități de randament și dilată cozile; graficele PP înghesuie cozile în colțuri.",
                "incorrectExplanation": "Graficele PP compară probabilități în $[0, 1]$ și ascund cozile; graficele QQ compară cuantile, deci fiecare zi extremă este vizibilă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading a QQ plot",
                "text": "In a QQ plot of returns (empirical quantiles on the vertical axis) against a fitted Normal law, the ends bend away from the line, below on the left and above on the right. What does this mean?",
                "options": [
                    "The model tails are too long",
                    "The data are skewed to the right only",
                    "The model fits well",
                    "The model tails are too short: the data have more extreme values than the Normal law allows"
                ],
                "correctExplanation": "Empirical extremes beyond the model quantiles mean heavier tails in the data.",
                "incorrectExplanation": "Points below the line on the left and above it on the right mean the observed extremes are larger than the model quantiles: the model tails are too short."
            },
            "ro": {
                "title": "Citirea unui grafic QQ",
                "text": "Într-un grafic QQ al randamentelor (cuantilele empirice pe verticală) față de o distribuție Normală estimată, capetele se îndepărtează de dreaptă, sub ea în stînga și deasupra în dreapta. Ce înseamnă?",
                "options": [
                    "Cozile modelului sînt prea lungi",
                    "Datele sînt asimetrice doar spre dreapta",
                    "Modelul se potrivește bine",
                    "Cozile modelului sînt prea scurte: datele au mai multe valori extreme decît prevede distribuția Normală"
                ],
                "correctExplanation": "Extremele empirice dincolo de cuantilele modelului înseamnă cozi mai groase în date.",
                "incorrectExplanation": "Punctele sub dreaptă în stînga și deasupra ei în dreapta înseamnă că extremele observate sînt mai mari decît cuantilele modelului: cozile modelului sînt prea scurte."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Normal VaR 1%",
                "text": "Daily returns are Normal with $\\mu = 0.05\\%$ and $\\sigma = 1.2\\%$; $z_{0.01} = -2.326$. What is VaR 1%?",
                "options": [
                    "About $2.74\\%$",
                    "About $2.84\\%$",
                    "About $1.20\\%$",
                    "About $-2.74\\%$"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{1\\%} = -(\\mu + \\sigma z_{0.01}) = -(0.05 - 2.326 \\times 1.2) \\approx 2.74\\%$.",
                "incorrectExplanation": "VaR 1% is minus the 1% quantile: $-(0.05 - 2.326 \\times 1.2) = 2.74\\%$, a positive loss."
            },
            "ro": {
                "title": "VaR 1% Normal",
                "text": "Randamentele zilnice sînt Normale cu $\\mu = 0{,}05\\%$ și $\\sigma = 1{,}2\\%$; $z_{0{,}01} = -2{,}326$. Cît este VaR 1%?",
                "options": [
                    "Circa $2{,}74\\%$",
                    "Circa $2{,}84\\%$",
                    "Circa $1{,}20\\%$",
                    "Circa $-2{,}74\\%$"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{1\\%} = -(\\mu + \\sigma z_{0{,}01}) = -(0{,}05 - 2{,}326 \\times 1{,}2) \\approx 2{,}74\\%$.",
                "incorrectExplanation": "VaR 1% este minus cuantila de 1%: $-(0{,}05 - 2{,}326 \\times 1{,}2) = 2{,}74\\%$, o pierdere pozitivă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Too few exceedances",
                "text": "A VaR 1% model has 4 exceedances in 1000 days (10 expected). The Kupiec statistic is 4.71. What is the conclusion at 5%?",
                "options": [
                    "The model is excellent: fewer losses than expected",
                    "Reject: the VaR is too conservative, which is also a misspecification",
                    "Do not reject, since 4.71 < 3.84",
                    "The test only checks for too many exceedances"
                ],
                "correctExplanation": "$4.71 > 3.84$: reject; too few exceedances means the VaR is too high and capital is wasted.",
                "incorrectExplanation": "The Kupiec test is two-sided: too many and too few exceedances both reject; here $4.71 > 3.84$."
            },
            "ro": {
                "title": "Prea puține depășiri",
                "text": "Un model VaR 1% are 4 depășiri în 1000 de zile (10 așteptate). Statistica Kupiec este 4,71. Care este concluzia la 5%?",
                "options": [
                    "Modelul este excelent: mai puține pierderi decît se aștepta",
                    "Respingem: VaR este prea prudent, ceea ce este tot o specificare greșită",
                    "Nu respingem, pentru că 4,71 < 3,84",
                    "Testul verifică doar depășirile prea multe"
                ],
                "correctExplanation": "$4{,}71 > 3{,}84$: respingem; prea puține depășiri înseamnă un VaR prea mare și capital irosit.",
                "incorrectExplanation": "Testul Kupiec este bilateral: atît prea multe, cît și prea puține depășiri duc la respingere; aici $4{,}71 > 3{,}84$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Normal VaR on real data",
                "text": "For the S&P 500 since 2000 the empirical VaR 1% is $3.45\\%$. What does the fitted Normal model give?",
                "options": [
                    "About $3.45\\%$, the same",
                    "About $4.15\\%$, too high",
                    "About $2.80\\%$, too low",
                    "It cannot be computed"
                ],
                "correctExplanation": "The Normal model gives $2.80\\%$: thin tails underestimate the 1% loss.",
                "incorrectExplanation": "The Normal fit gives $2.80\\%$, below the empirical $3.45\\%$; heavy-tailed models (Student-t, skewed-t, NIG) come much closer."
            },
            "ro": {
                "title": "VaR Normal pe date reale",
                "text": "Pentru S&P 500 din 2000, VaR 1% empiric este $3{,}45\\%$. Ce valoare dă modelul Normal estimat?",
                "options": [
                    "Circa $3{,}45\\%$, același",
                    "Circa $4{,}15\\%$, prea mare",
                    "Circa $2{,}80\\%$, prea mic",
                    "Nu se poate calcula"
                ],
                "correctExplanation": "Modelul Normal dă $2{,}80\\%$: cozile subțiri subestimează pierderea de 1%.",
                "incorrectExplanation": "Modelul Normal estimat dă $2{,}80\\%$, sub valoarea empirică de $3{,}45\\%$; modelele cu cozi groase (Student-t, skewed-t, NIG) sînt mult mai aproape."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Overfitting",
                "text": "Normal mixtures with 1 to 6 components are fitted on two years of S&P 500 returns. Which pattern indicates overfitting?",
                "options": [
                    "The in-sample log-likelihood falls with more components",
                    "AIC and BIC always choose the largest model",
                    "The out-of-sample fit keeps rising with every component",
                    "The in-sample fit rises with every component while the out-of-sample fit peaks at 3 components and then falls"
                ],
                "correctExplanation": "Extra parameters fit noise: in-sample fit improves, out-of-sample fit deteriorates.",
                "incorrectExplanation": "In sample, a larger model never fits worse; overfitting shows when the out-of-sample fit stops improving and declines."
            },
            "ro": {
                "title": "Supraajustarea",
                "text": "Amestecuri Normale cu 1 pînă la 6 componente sînt estimate pe doi ani de randamente S&P 500. Ce tipar indică supraajustarea?",
                "options": [
                    "Log-verosimilitatea în eșantion scade cu mai multe componente",
                    "AIC și BIC aleg întotdeauna cel mai mare model",
                    "Calitatea ajustării în afara eșantionului crește cu fiecare componentă",
                    "Calitatea ajustării în eșantion crește cu fiecare componentă, iar cea din afara eșantionului atinge maximul la 3 componente și apoi scade"
                ],
                "correctExplanation": "Parametrii în plus modelează zgomotul: calitatea ajustării în eșantion crește, cea din afara lui se deteriorează.",
                "incorrectExplanation": "În eșantion, un model mai general nu se ajustează niciodată mai slab; supraajustarea apare cînd calitatea ajustării din afara eșantionului nu mai crește și scade."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Best density, best VaR?",
                "text": "A model has the best out-of-sample log score. Does it necessarily give the best VaR 1%?",
                "options": [
                    "No: the log score judges the whole density, VaR depends on one quantile; check exceedances and the pinball loss",
                    "Yes, always",
                    "Yes, if its AIC is also the lowest",
                    "Only for the Normal model"
                ],
                "correctExplanation": "The purpose decides: a VaR model is judged by its exceedances and quantile loss.",
                "incorrectExplanation": "The log score rewards the body of the distribution too; the 1% quantile can still be off, as the DAX test period shows."
            },
            "ro": {
                "title": "Cea mai bună densitate, cel mai bun VaR?",
                "text": "Un model are cel mai bun scor logaritmic în afara eșantionului. Dă el neapărat cel mai bun VaR 1%?",
                "options": [
                    "Nu: scorul logaritmic judecă întreaga densitate, VaR depinde de o singură cuantilă; verificați depășirile și pierderea pinball",
                    "Da, întotdeauna",
                    "Da, dacă și AIC-ul lui este cel mai mic",
                    "Doar pentru modelul Normal"
                ],
                "correctExplanation": "Scopul decide: un model VaR se judecă după depășiri și după pierderea pinball.",
                "incorrectExplanation": "Scorul logaritmic răsplătește și corpul distribuției; cuantila de 1% poate fi totuși greșită, cum arată perioada de test pentru DAX."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Crash or data error?",
                "text": "Banca Transilvania shows $-15\\%$ and $+20\\%$ on two consecutive days in May 2016 and $-22\\%$ on 19 December 2018. What should you do before fitting?",
                "options": [
                    "Remove all three days, since they are outliers",
                    "Check close, adjusted close and the market: remove the 2016 pair (a late adjustment for bonus shares) and keep the 2018 crash",
                    "Keep all three days",
                    "Replace them by the median return"
                ],
                "correctExplanation": "Only proven errors are removed; the December 2018 fall is real (the BET fell about 12% that day).",
                "incorrectExplanation": "The 2016 pair reverses itself and the close shows a bonus-share adjustment applied one day late; the 2018 fall appears in the close and in the BET, so it is a real crash."
            },
            "ro": {
                "title": "Crah sau eroare de date?",
                "text": "Banca Transilvania are $-15\\%$ și $+20\\%$ în două zile consecutive din mai 2016 și $-22\\%$ pe 19 decembrie 2018. Ce faceți înainte de estimarea modelelor?",
                "options": [
                    "Eliminați toate cele trei zile, pentru că sînt valori aberante",
                    "Verificați prețul de închidere, prețul ajustat și piața: eliminați perechea din 2016 (ajustare întîrziată pentru acțiuni gratuite) și păstrați crahul din 2018",
                    "Păstrați toate cele trei zile",
                    "Le înlocuiți cu randamentul median"
                ],
                "correctExplanation": "Se elimină doar erorile dovedite; scăderea din decembrie 2018 este reală (BET a scăzut cu circa 12% în acea zi).",
                "incorrectExplanation": "Perechea din 2016 se anulează singură, iar prețul de închidere arată o ajustare pentru acțiuni gratuite aplicată cu o zi întîrziere; scăderea din 2018 apare și în prețul de închidere, și în BET, deci este un crah real."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Model risk",
                "text": "Six plausible heavy-tailed models give VaR 1% between $3.3\\%$ and $4.5\\%$ for the same portfolio. What should a risk report contain?",
                "options": [
                    "Only the smallest value",
                    "Only the value of the model with the most parameters",
                    "The chosen model, the range across plausible models, or a VaR averaged with Akaike weights",
                    "The Normal VaR, as a neutral benchmark"
                ],
                "correctExplanation": "The spread across plausible models is model risk; report it or average with Akaike weights.",
                "incorrectExplanation": "A single number hides model risk; report the range across plausible models or average the VaR with the Akaike weights."
            },
            "ro": {
                "title": "Riscul de model",
                "text": "Șase modele plauzibile cu cozi groase dau VaR 1% între $3{,}3\\%$ și $4{,}5\\%$ pentru același portofoliu. Ce trebuie să conțină raportul de risc?",
                "options": [
                    "Doar cea mai mică valoare",
                    "Doar valoarea modelului cu cei mai mulți parametri",
                    "Modelul ales, intervalul dintre modelele plauzibile sau un VaR mediat cu ponderile Akaike",
                    "VaR Normal, ca etalon neutru"
                ],
                "correctExplanation": "Dispersia dintre modelele plauzibile este riscul de model; raportați-o sau mediați cu ponderile Akaike.",
                "incorrectExplanation": "O singură cifră ascunde riscul de model; raportați intervalul dintre modelele plauzibile sau mediați VaR cu ponderile Akaike."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Normal mixture",
                "text": "A two-component Normal mixture has five parameters, yet it loses to the three-parameter Student-t on daily returns. Why?",
                "options": [
                    "Because EM never converges",
                    "Because a mixture cannot be skewed",
                    "Because AIC ignores the number of parameters",
                    "Because its tails are still Normal (thin) far out, and it pays for two extra parameters"
                ],
                "correctExplanation": "Far in the tail the wider Normal component dominates and still decays like $e^{-x^2}$.",
                "incorrectExplanation": "A mixture fattens the tails only up to its widest component, whose tail is Normal; the extra parameters are penalised by AIC and BIC."
            },
            "ro": {
                "title": "Amestecul Normal",
                "text": "Un amestec Normal cu două componente are cinci parametri, dar pierde în fața distribuției Student-t cu trei parametri pe randamentele zilnice. De ce?",
                "options": [
                    "Pentru că EM nu converge niciodată",
                    "Pentru că un amestec nu poate fi asimetric",
                    "Pentru că AIC ignoră numărul de parametri",
                    "Pentru că departe în cozi rămîne Normal (cu cozi subțiri) și este penalizat pentru doi parametri în plus"
                ],
                "correctExplanation": "Departe în coadă domină componenta Normală mai largă, care tot scade ca $e^{-x^2}$.",
                "incorrectExplanation": "Un amestec îngroașă cozile doar pînă la componenta lui cea mai largă, a cărei coadă este Normală; parametrii în plus sînt penalizați de AIC și BIC."
            }
        }
    ]
};
