// ============================================================
// Chapter 9 quiz bank: ARCH and GARCH models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['garch'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Conditional variance",
                "text": "What is the conditional variance sigma_t^2 of a daily return r_t?",
                "options": [
                    "Var(r_t | information up to day t-1), which can change from day to day",
                    "The sample variance of all returns in the data set",
                    "The variance of the price level P_t",
                    "The square of the conditional mean of r_t"
                ],
                "correctExplanation": "The conditional variance is the variance of tomorrow's return given the past; in GARCH models it is known one day ahead and changes every day.",
                "incorrectExplanation": "The sample variance estimates the unconditional variance, the price level is not the modelled variable, and the mean is a different moment: sigma_t^2 = Var(r_t | past)."
            },
            "ro": {
                "title": "Varianța condiționată",
                "text": "Ce este varianța condiționată sigma_t^2 a unui randament zilnic r_t?",
                "options": [
                    "Var(r_t | informația pînă în ziua t-1), care se poate schimba de la o zi la alta",
                    "Varianța de selecție a tuturor randamentelor din eșantion",
                    "Varianța nivelului prețului P_t",
                    "Pătratul mediei condiționate a lui r_t"
                ],
                "correctExplanation": "Varianța condiționată este varianța randamentului de mîine, dată fiind informația trecută; în modelele GARCH ea este cunoscută cu o zi înainte și se schimbă zilnic.",
                "incorrectExplanation": "Varianța de selecție estimează varianța necondiționată, nivelul prețului nu este variabila modelată, iar media este alt moment: sigma_t^2 = Var(r_t | trecut)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Uncorrelated but dependent",
                "text": "In the model eps_t = sigma_t z_t with a GARCH variance, which statement is true?",
                "options": [
                    "The shocks are independent and identically distributed",
                    "The shocks are uncorrelated, but their squares are correlated",
                    "The shocks are positively autocorrelated",
                    "The squared shocks are uncorrelated"
                ],
                "correctExplanation": "E[eps_t | past] = 0 makes the shocks uncorrelated, while sigma_t^2 depends on past squared shocks, so the squares are correlated: volatility clustering.",
                "incorrectExplanation": "GARCH keeps the mean unpredictable (no autocorrelation of the shocks) but makes the variance predictable, which correlates the squared shocks; they are therefore not independent."
            },
            "ro": {
                "title": "Necorelate, dar dependente",
                "text": "În modelul eps_t = sigma_t z_t cu varianță GARCH, care afirmație este adevărată?",
                "options": [
                    "Șocurile sînt independente și identic distribuite",
                    "Șocurile sînt necorelate, dar pătratele lor sînt corelate",
                    "Șocurile sînt autocorelate pozitiv",
                    "Pătratele șocurilor sînt necorelate"
                ],
                "correctExplanation": "E[eps_t | trecut] = 0 face ca șocurile să fie necorelate, iar sigma_t^2 depinde de pătratele șocurilor trecute, deci pătratele sînt corelate: volatility clustering.",
                "incorrectExplanation": "GARCH păstrează media imprevizibilă (șocurile nu sînt autocorelate), dar face varianța previzibilă, ceea ce corelează pătratele șocurilor; deci șocurile nu sînt independente."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ARCH(1) unconditional variance",
                "text": "An ARCH(1) model has omega = 0.4 and alpha = 0.6. What is its unconditional variance?",
                "options": [
                    "0.4",
                    "0.24",
                    "1.0",
                    "It does not exist"
                ],
                "correctExplanation": "The unconditional variance is omega/(1 - alpha) = 0.4/0.4 = 1.0, finite because alpha < 1.",
                "incorrectExplanation": "Take expectations in sigma_t^2 = omega + alpha eps_{t-1}^2: E[eps^2] = omega + alpha E[eps^2], so E[eps^2] = omega/(1 - alpha) = 1.0."
            },
            "ro": {
                "title": "Varianța necondiționată ARCH(1)",
                "text": "Un model ARCH(1) are omega = 0,4 și alpha = 0,6. Care este varianța lui necondiționată?",
                "options": [
                    "0,4",
                    "0,24",
                    "1,0",
                    "Nu există"
                ],
                "correctExplanation": "Varianța necondiționată este omega/(1 - alpha) = 0,4/0,4 = 1,0, finită deoarece alpha < 1.",
                "incorrectExplanation": "Aplicăm media în sigma_t^2 = omega + alpha eps_{t-1}^2: E[eps^2] = omega + alpha E[eps^2], deci E[eps^2] = omega/(1 - alpha) = 1,0."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Kurtosis of ARCH",
                "text": "An ARCH(1) process has Normal innovations z_t and 0 < alpha < 1/sqrt(3). What is the kurtosis of eps_t?",
                "options": [
                    "Exactly 3, as for the Normal distribution",
                    "Below 3",
                    "It cannot be computed",
                    "Above 3: 3(1 - alpha^2)/(1 - 3 alpha^2)"
                ],
                "correctExplanation": "A mixture of Normal distributions with changing variances has heavy tails: K = 3(1 - alpha^2)/(1 - 3 alpha^2) > 3.",
                "incorrectExplanation": "Even with Normal innovations, the changing conditional variance makes the unconditional distribution heavy-tailed; the formula 3(1 - alpha^2)/(1 - 3 alpha^2) exceeds 3."
            },
            "ro": {
                "title": "Boltirea în ARCH",
                "text": "Un proces ARCH(1) are inovații z_t Normale și 0 < alpha < 1/sqrt(3). Care este coeficientul de boltire al lui eps_t?",
                "options": [
                    "Exact 3, ca la distribuția Normală",
                    "Sub 3",
                    "Nu poate fi calculat",
                    "Peste 3: 3(1 - alpha^2)/(1 - 3 alpha^2)"
                ],
                "correctExplanation": "Un amestec de distribuții Normale cu varianțe diferite are cozi groase: K = 3(1 - alpha^2)/(1 - 3 alpha^2) > 3.",
                "incorrectExplanation": "Chiar cu inovații Normale, varianța condiționată variabilă face ca distribuția necondiționată să aibă cozi groase; formula 3(1 - alpha^2)/(1 - 3 alpha^2) depășește 3."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Stationarity of GARCH(1,1)",
                "text": "When is a GARCH(1,1) process covariance stationary?",
                "options": [
                    "When omega < 1",
                    "When beta < alpha",
                    "When alpha + beta < 1",
                    "Always, for any positive parameters"
                ],
                "correctExplanation": "Covariance stationarity requires alpha + beta < 1; then the unconditional variance omega/(1 - alpha - beta) is finite.",
                "incorrectExplanation": "The condition concerns the persistence alpha + beta, not omega or the ordering of alpha and beta; with alpha + beta >= 1 the unconditional variance is not finite."
            },
            "ro": {
                "title": "Staționaritatea GARCH(1,1)",
                "text": "Cînd este un proces GARCH(1,1) staționar în covarianță?",
                "options": [
                    "Cînd omega < 1",
                    "Cînd beta < alpha",
                    "Cînd alpha + beta < 1",
                    "Întotdeauna, pentru orice parametri pozitivi"
                ],
                "correctExplanation": "Staționaritatea în covarianță cere alpha + beta < 1; atunci varianța necondiționată omega/(1 - alpha - beta) este finită.",
                "incorrectExplanation": "Condiția privește persistența alpha + beta, nu pe omega sau ordinea lui alpha și beta; dacă alpha + beta >= 1, varianța necondiționată nu este finită."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Long-run volatility",
                "text": "Daily returns in %: omega = 0.02, alpha = 0.10, beta = 0.88. What is the annualised long-run volatility (252 days)?",
                "options": [
                    "About 1.0%",
                    "About 2.2%",
                    "About 25.2%",
                    "About 15.9%"
                ],
                "correctExplanation": "The long-run variance is 0.02/(1 - 0.98) = 1.0 (%^2 per day); the annualised volatility is sqrt(252 x 1.0) = 15.9%.",
                "incorrectExplanation": "First compute the daily long-run variance omega/(1 - alpha - beta) = 1.0, then annualise the standard deviation with sqrt(252): about 15.9%."
            },
            "ro": {
                "title": "Volatilitatea pe termen lung",
                "text": "Randamente zilnice în %: omega = 0,02, alpha = 0,10, beta = 0,88. Care este volatilitatea pe termen lung anualizată (252 de zile)?",
                "options": [
                    "Circa 1,0%",
                    "Circa 2,2%",
                    "Circa 25,2%",
                    "Circa 15,9%"
                ],
                "correctExplanation": "Varianța pe termen lung este 0,02/(1 - 0,98) = 1,0 (%^2 pe zi); volatilitatea anualizată este sqrt(252 x 1,0) = 15,9%.",
                "incorrectExplanation": "Calculăm întîi varianța zilnică pe termen lung omega/(1 - alpha - beta) = 1,0, apoi anualizăm abaterea standard cu sqrt(252): circa 15,9%."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Half-life",
                "text": "A GARCH(1,1) model has alpha + beta = 0.99. After how many days has half of a variance shock disappeared?",
                "options": [
                    "About 69 days",
                    "About 2 days",
                    "About 7 days",
                    "About 99 days"
                ],
                "correctExplanation": "The half-life is ln 0.5 / ln 0.99 = 69 days: persistence close to 1 means long-lasting volatility.",
                "incorrectExplanation": "A deviation from the long-run variance shrinks by the factor alpha + beta each day; solving 0.99^h = 0.5 gives h = ln 0.5/ln 0.99, about 69 days."
            },
            "ro": {
                "title": "Timpul de înjumătățire",
                "text": "Un model GARCH(1,1) are alpha + beta = 0,99. După cîte zile a dispărut jumătate dintr-un șoc de varianță?",
                "options": [
                    "Circa 69 de zile",
                    "Circa 2 zile",
                    "Circa 7 zile",
                    "Circa 99 de zile"
                ],
                "correctExplanation": "Timpul de înjumătățire este ln 0,5 / ln 0,99 = 69 de zile: o persistență apropiată de 1 înseamnă că șocurile de volatilitate durează mult.",
                "incorrectExplanation": "O abatere de la varianța pe termen lung se micșorează cu factorul alpha + beta în fiecare zi; din 0,99^h = 0,5 rezultă h = ln 0,5/ln 0,99, circa 69 de zile."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "EWMA as a GARCH model",
                "text": "EWMA with lambda = 0.94 can be written as a GARCH(1,1). Which parameters?",
                "options": [
                    "omega = 0.06, alpha = 0.94, beta = 0",
                    "omega = 0, alpha = 0.06, beta = 0.94",
                    "omega = 0.94, alpha = 0.06, beta = 0",
                    "omega = 0, alpha = 0.94, beta = 0.06"
                ],
                "correctExplanation": "sigma_t^2 = 0.94 sigma_{t-1}^2 + 0.06 r_{t-1}^2 is a GARCH(1,1) with omega = 0 and alpha + beta = 1: an IGARCH.",
                "incorrectExplanation": "In EWMA the weight on yesterday's variance is lambda (that is beta) and the weight on the squared return is 1 - lambda (that is alpha), with no constant."
            },
            "ro": {
                "title": "EWMA ca model GARCH",
                "text": "EWMA cu lambda = 0,94 poate fi scris ca un GARCH(1,1). Cu ce parametri?",
                "options": [
                    "omega = 0,06, alpha = 0,94, beta = 0",
                    "omega = 0, alpha = 0,06, beta = 0,94",
                    "omega = 0,94, alpha = 0,06, beta = 0",
                    "omega = 0, alpha = 0,94, beta = 0,06"
                ],
                "correctExplanation": "sigma_t^2 = 0,94 sigma_{t-1}^2 + 0,06 r_{t-1}^2 este un GARCH(1,1) cu omega = 0 și alpha + beta = 1: un IGARCH.",
                "incorrectExplanation": "În EWMA, ponderea varianței de ieri este lambda (adică beta), iar ponderea randamentului la pătrat este 1 - lambda (adică alpha), fără constantă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading beta",
                "text": "In GARCH(1,1), what does a large beta (for example 0.9) mean?",
                "options": [
                    "The variance reacts strongly to yesterday's shock",
                    "Returns are strongly autocorrelated",
                    "The long-run variance is large",
                    "The variance has a long memory: yesterday's level is largely kept"
                ],
                "correctExplanation": "beta is the weight of yesterday's conditional variance: a large beta means a slowly changing, persistent variance.",
                "incorrectExplanation": "The reaction to news is alpha; beta measures memory. Neither says anything about the autocorrelation of returns, and the long-run level depends on omega/(1 - alpha - beta)."
            },
            "ro": {
                "title": "Interpretarea lui beta",
                "text": "În GARCH(1,1), ce înseamnă un beta mare (de exemplu 0,9)?",
                "options": [
                    "Varianța reacționează puternic la șocul de ieri",
                    "Randamentele sînt puternic autocorelate",
                    "Varianța pe termen lung este mare",
                    "Varianța are o memorie lungă: nivelul de ieri se păstrează în mare parte"
                ],
                "correctExplanation": "beta este ponderea varianței condiționate de ieri: un beta mare înseamnă o varianță care se schimbă lent, persistentă.",
                "incorrectExplanation": "Reacția la știri este alpha; beta măsoară memoria. Niciunul nu spune ceva despre autocorelația randamentelor, iar nivelul pe termen lung depinde de omega/(1 - alpha - beta)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Likelihood of a GARCH model",
                "text": "Why is the GARCH log-likelihood a sum of conditional log-densities?",
                "options": [
                    "Returns are independent, so densities multiply",
                    "The arch package requires it",
                    "Returns are dependent, so the joint density is factored into densities of r_t given the past",
                    "Because the innovations are Normal"
                ],
                "correctExplanation": "The joint density factors as f(r_1) times the product of f(r_t | past); each conditional density uses sigma_t^2 computed recursively.",
                "incorrectExplanation": "The factorisation holds for dependent data whatever the innovation distribution; independence would ignore exactly the dependence that GARCH models."
            },
            "ro": {
                "title": "Verosimilitatea unui model GARCH",
                "text": "De ce log-verosimilitatea GARCH este o sumă de log-densități condiționate?",
                "options": [
                    "Randamentele sînt independente, deci densitățile se înmulțesc",
                    "Pachetul arch cere acest lucru",
                    "Randamentele sînt dependente, deci densitatea comună se descompune în densități ale lui r_t condiționate de trecut",
                    "Deoarece inovațiile sînt Normale"
                ],
                "correctExplanation": "Densitatea comună se scrie ca f(r_1) înmulțit cu produsul densităților f(r_t | trecut); fiecare densitate condiționată folosește sigma_t^2 calculat recursiv.",
                "incorrectExplanation": "Descompunerea este valabilă pentru date dependente, oricare ar fi distribuția inovațiilor; independența ar ignora exact dependența pe care o modelează GARCH."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Robust standard errors",
                "text": "Why are robust (Bollerslev-Wooldridge) standard errors reported for GARCH estimates?",
                "options": [
                    "They are always smaller than the classic ones",
                    "They stay valid when the innovations are not Normal",
                    "They remove the need to estimate omega",
                    "They make the estimates unbiased"
                ],
                "correctExplanation": "Maximising a Normal likelihood when z_t is not Normal (QMLE) still gives consistent estimates, but only the robust sandwich standard errors are valid.",
                "incorrectExplanation": "Robust standard errors are usually larger, not smaller; they change neither the estimates nor the parameters to estimate; they correct the inference under non-Normal innovations."
            },
            "ro": {
                "title": "Erori standard robuste",
                "text": "De ce se raportează erori standard robuste (Bollerslev-Wooldridge) pentru estimările GARCH?",
                "options": [
                    "Sînt întotdeauna mai mici decît cele clasice",
                    "Rămîn valide cînd inovațiile nu sînt Normale",
                    "Elimină necesitatea de a estima omega",
                    "Fac estimările nedeplasate"
                ],
                "correctExplanation": "Maximizarea unei verosimilități Normale cînd z_t nu este Normal (QMLE) dă tot estimări consistente, dar doar erorile standard robuste, de tip sandwich, sînt valide.",
                "incorrectExplanation": "Erorile standard robuste sînt de obicei mai mari, nu mai mici; nu schimbă nici estimările, nici parametrii de estimat; ele corectează inferența cînd inovațiile nu sînt Normale."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Standardised Student-t",
                "text": "Why is a Student-t innovation multiplied by sqrt((nu - 2)/nu) in a GARCH model?",
                "options": [
                    "So that z_t has variance 1 and sigma_t^2 remains the conditional variance",
                    "So that z_t has kurtosis 3",
                    "So that the distribution becomes skewed",
                    "To make nu an integer"
                ],
                "correctExplanation": "A t variable with nu > 2 has variance nu/(nu - 2); the factor rescales it to variance 1, so sigma_t^2 is still Var(r_t | past).",
                "incorrectExplanation": "The scaling does not change the tails (the kurtosis stays 3 + 6/(nu - 4)) or the symmetry; it only sets the variance of z_t to 1."
            },
            "ro": {
                "title": "Student-t standardizată",
                "text": "De ce o inovație Student-t se înmulțește cu sqrt((nu - 2)/nu) într-un model GARCH?",
                "options": [
                    "Pentru ca z_t să aibă varianța 1, iar sigma_t^2 să rămînă varianța condiționată",
                    "Pentru ca z_t să aibă coeficientul de boltire 3",
                    "Pentru ca distribuția să devină asimetrică",
                    "Pentru ca nu să fie un număr întreg"
                ],
                "correctExplanation": "O variabilă t cu nu > 2 are varianța nu/(nu - 2); factorul o rescalează la varianța 1, deci sigma_t^2 rămîne Var(r_t | trecut).",
                "incorrectExplanation": "Scalarea nu schimbă cozile (coeficientul de boltire rămîne 3 + 6/(nu - 4)) și nici simetria; ea doar fixează varianța lui z_t la 1."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "An estimate on the boundary",
                "text": "For Bitcoin, GARCH(1,1)-t gives alpha + beta = 1.000. What follows?",
                "options": [
                    "The model is an IGARCH: no half-life and no finite long-run volatility",
                    "Bitcoin volatility is constant",
                    "The estimation failed and the model must be discarded",
                    "Bitcoin returns are a random walk in variance with half-life 1 day"
                ],
                "correctExplanation": "With alpha + beta = 1 shocks to the variance never die out; forecasts are the same for every horizon, as in EWMA.",
                "incorrectExplanation": "The volatility is far from constant and the fit is usable; but with persistence 1 there is no mean reversion, hence no half-life and no long-run variance to report."
            },
            "ro": {
                "title": "O estimare pe frontieră",
                "text": "Pentru Bitcoin, GARCH(1,1)-t dă alpha + beta = 1,000. Ce rezultă?",
                "options": [
                    "Modelul este un IGARCH: nu există timp de înjumătățire și nici volatilitate pe termen lung finită",
                    "Volatilitatea Bitcoin este constantă",
                    "Estimarea a eșuat, iar modelul trebuie abandonat",
                    "Varianța Bitcoin are un timp de înjumătățire de o zi"
                ],
                "correctExplanation": "Cu alpha + beta = 1, șocurile varianței nu se sting niciodată; prognozele sînt aceleași pentru orice orizont, ca la EWMA.",
                "incorrectExplanation": "Volatilitatea este departe de a fi constantă, iar estimarea poate fi folosită; dar cu persistența 1 nu există revenire la medie, deci nici timp de înjumătățire, nici varianță pe termen lung de raportat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Leverage effect",
                "text": "What is the leverage effect?",
                "options": [
                    "Rises in prices raise volatility more than falls",
                    "Falls in prices raise volatility more than rises of the same size",
                    "Volatility is higher for companies with more debt only",
                    "Returns are higher when volatility is high"
                ],
                "correctExplanation": "Negative shocks increase future volatility more than positive shocks of equal size (Christie, 1982); GJR and EGARCH model it.",
                "incorrectExplanation": "The effect is an asymmetry in the response of volatility to the sign of shocks, observed for equity indices in general, not a relation between returns and volatility levels."
            },
            "ro": {
                "title": "Efectul de levier",
                "text": "Ce este efectul de levier?",
                "options": [
                    "Creșterile prețurilor cresc volatilitatea mai mult decît scăderile",
                    "Scăderile prețurilor cresc volatilitatea mai mult decît creșterile de aceeași mărime",
                    "Volatilitatea este mai mare doar la companiile cu datorii mai mari",
                    "Randamentele sînt mai mari cînd volatilitatea este mare"
                ],
                "correctExplanation": "Șocurile negative cresc volatilitatea viitoare mai mult decît șocurile pozitive de aceeași mărime (Christie, 1982); modelele GJR și EGARCH îl surprind.",
                "incorrectExplanation": "Efectul este o asimetrie a reacției volatilității la semnul șocurilor, observată în general la indicii de acțiuni, nu o relație între randamente și nivelul volatilității."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GJR-GARCH",
                "text": "In GJR-GARCH, sigma_t^2 = omega + (alpha + gamma I_{t-1}) eps_{t-1}^2 + beta sigma_{t-1}^2 with I_{t-1} = 1 if eps_{t-1} < 0. What does gamma > 0 mean?",
                "options": [
                    "Positive shocks raise the variance more than negative ones",
                    "The variance has no memory",
                    "Negative shocks raise the variance more than positive ones",
                    "The model is not stationary"
                ],
                "correctExplanation": "For a negative shock the slope is alpha + gamma, for a positive one only alpha: gamma > 0 is the leverage effect; the persistence is alpha + beta + gamma/2.",
                "incorrectExplanation": "The indicator switches on for negative shocks only, so gamma adds to their effect; stationarity depends on alpha + beta + gamma/2 < 1."
            },
            "ro": {
                "title": "GJR-GARCH",
                "text": "În GJR-GARCH, sigma_t^2 = omega + (alpha + gamma I_{t-1}) eps_{t-1}^2 + beta sigma_{t-1}^2, cu I_{t-1} = 1 dacă eps_{t-1} < 0. Ce înseamnă gamma > 0?",
                "options": [
                    "Șocurile pozitive cresc varianța mai mult decît cele negative",
                    "Varianța nu are memorie",
                    "Șocurile negative cresc varianța mai mult decît cele pozitive",
                    "Modelul nu este staționar"
                ],
                "correctExplanation": "Pentru un șoc negativ panta este alpha + gamma, pentru unul pozitiv doar alpha: gamma > 0 este efectul de levier; persistența este alpha + beta + gamma/2.",
                "incorrectExplanation": "Indicatorul este activ doar pentru șocurile negative, deci gamma se adaugă la efectul lor; staționaritatea depinde de alpha + beta + gamma/2 < 1."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "EGARCH",
                "text": "Why does EGARCH model ln sigma_t^2 instead of sigma_t^2?",
                "options": [
                    "It makes the returns Normal",
                    "It removes the leverage effect",
                    "It makes the model an ARCH(1)",
                    "The variance stays positive without sign restrictions on the parameters"
                ],
                "correctExplanation": "exp(.) is always positive, so gamma can be negative (leverage: gamma < 0) and no positivity constraints are needed (Nelson, 1991).",
                "incorrectExplanation": "The logarithm concerns the variance equation only; it does not change the innovation distribution and, through gamma z_{t-1}, it allows the leverage effect."
            },
            "ro": {
                "title": "EGARCH",
                "text": "De ce modelează EGARCH ln sigma_t^2 în loc de sigma_t^2?",
                "options": [
                    "Face randamentele Normale",
                    "Elimină efectul de levier",
                    "Transformă modelul într-un ARCH(1)",
                    "Varianța rămîne pozitivă fără restricții de semn asupra parametrilor"
                ],
                "correctExplanation": "exp(.) este întotdeauna pozitivă, deci gamma poate fi negativ (efect de levier: gamma < 0) și nu sînt necesare restricții de pozitivitate (Nelson, 1991).",
                "incorrectExplanation": "Logaritmul privește doar ecuația varianței; nu schimbă distribuția inovațiilor și, prin termenul gamma z_{t-1}, permite efectul de levier."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "News impact curve",
                "text": "What does the news impact curve show?",
                "options": [
                    "The return as a function of the number of news items",
                    "The price as a function of time",
                    "sigma_t^2 as a function of the shock eps_{t-1}, with sigma_{t-1}^2 fixed",
                    "The autocorrelation of squared returns"
                ],
                "correctExplanation": "Engle and Ng (1993): the curve is a symmetric parabola for GARCH and steeper on the left for GJR and EGARCH with a leverage effect.",
                "incorrectExplanation": "It isolates the effect of yesterday's shock on today's variance, keeping yesterday's variance at a fixed level such as the unconditional variance."
            },
            "ro": {
                "title": "Curba de impact a știrilor",
                "text": "Ce arată curba de impact a știrilor?",
                "options": [
                    "Randamentul ca funcție de numărul de știri",
                    "Prețul ca funcție de timp",
                    "sigma_t^2 ca funcție de șocul eps_{t-1}, cu sigma_{t-1}^2 fixat",
                    "Autocorelația pătratelor randamentelor"
                ],
                "correctExplanation": "Engle și Ng (1993): curba este o parabolă simetrică pentru GARCH și mai abruptă la stînga pentru GJR și EGARCH cu efect de levier.",
                "incorrectExplanation": "Curba izolează efectul șocului de ieri asupra varianței de azi, păstrînd varianța de ieri la un nivel fix, de exemplu varianța necondiționată."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Diagnostics",
                "text": "The Ljung-Box Q(10) of the squared standardised residuals has p = 0.40. What can we conclude?",
                "options": [
                    "The model is correct",
                    "The returns are independent",
                    "The model has too many parameters",
                    "No ARCH effect is left; asymmetry or the wrong distribution may remain"
                ],
                "correctExplanation": "A passed test only shows that the squared residuals are not autocorrelated; other tests (sign bias, QQ plot) may still find problems.",
                "incorrectExplanation": "Not rejecting a null is not a proof; the test checks one property of the residuals, not the whole model or the returns."
            },
            "ro": {
                "title": "Diagnostic",
                "text": "Statistica Ljung-Box Q(10) a pătratelor reziduurilor standardizate are p = 0,40. Ce putem concluziona?",
                "options": [
                    "Modelul este corect",
                    "Randamentele sînt independente",
                    "Modelul are prea mulți parametri",
                    "Nu a rămas niciun efect ARCH; pot rămîne asimetria sau o distribuție greșită"
                ],
                "correctExplanation": "Un test fără respingere arată doar că pătratele reziduurilor nu sînt autocorelate; alte teste (sign bias, graficul QQ) pot găsi încă probleme.",
                "incorrectExplanation": "Nerespingerea unei ipoteze nule nu este o dovadă; testul verifică o singură proprietate a reziduurilor, nu întregul model sau randamentele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Sign-bias test",
                "text": "The sign-bias test rejects for a symmetric GARCH fitted to the S&P 500. What is missing from the model?",
                "options": [
                    "The asymmetry (leverage effect)",
                    "A constant in the mean",
                    "Heavier tails in the distribution",
                    "More lags of the squared shocks"
                ],
                "correctExplanation": "The test regresses z_t^2 on the sign and size of past shocks: a rejection means the sign of past shocks still predicts the variance (Engle and Ng, 1993).",
                "incorrectExplanation": "The test looks at the sign of past shocks, so it points to a missing asymmetric term such as the gamma of GJR or EGARCH."
            },
            "ro": {
                "title": "Testul de asimetrie (sign bias)",
                "text": "Testul de asimetrie (sign bias) respinge pentru un GARCH simetric estimat pe S&P 500. Ce lipsește din model?",
                "options": [
                    "Asimetria (efectul de levier)",
                    "O constantă în medie",
                    "Cozi mai groase în distribuție",
                    "Mai multe laguri ale pătratelor șocurilor"
                ],
                "correctExplanation": "Testul face regresia lui z_t^2 pe semnul și mărimea șocurilor trecute: o respingere înseamnă că semnul șocurilor trecute încă anticipează varianța (Engle și Ng, 1993).",
                "incorrectExplanation": "Testul privește semnul șocurilor trecute, deci arată spre un termen asimetric care lipsește, de exemplu gamma din GJR sau EGARCH."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "AIC and BIC",
                "text": "AIC = -2 l + 2k and BIC = -2 l + k ln T. With T = 6700 daily returns, which statement is true?",
                "options": [
                    "AIC penalises each extra parameter more than BIC",
                    "BIC penalises each extra parameter more than AIC",
                    "The model with the larger value is preferred",
                    "They can compare models fitted on different data"
                ],
                "correctExplanation": "ln 6700 is about 8.8 > 2, so BIC is stricter; in both, the smaller value is better, and the models must be fitted on the same data.",
                "incorrectExplanation": "The penalty per parameter is 2 for AIC and ln T for BIC; both criteria choose the smallest value and only make sense for models of the same data."
            },
            "ro": {
                "title": "AIC și BIC",
                "text": "AIC = -2 l + 2k și BIC = -2 l + k ln T. Cu T = 6700 de randamente zilnice, care afirmație este adevărată?",
                "options": [
                    "AIC penalizează fiecare parametru suplimentar mai mult decît BIC",
                    "BIC penalizează fiecare parametru suplimentar mai mult decît AIC",
                    "Se preferă modelul cu valoarea mai mare",
                    "Pot compara modele estimate pe date diferite"
                ],
                "correctExplanation": "ln 6700 este circa 8,8 > 2, deci BIC este mai strict; la ambele criterii valoarea mai mică este mai bună, iar modelele trebuie estimate pe aceleași date.",
                "incorrectExplanation": "Penalizarea pe parametru este 2 la AIC și ln T la BIC; ambele criterii aleg valoarea cea mai mică și au sens doar pentru modele ale acelorași date."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Multi-step forecasts",
                "text": "Right after a crash, sigma_{t+1}^2 is far above the long-run variance. How does GARCH(1,1) forecast the variance of the next 10 days?",
                "options": [
                    "As 10 times sigma_{t+1}^2",
                    "As 10 times the long-run variance",
                    "As sigma_{t+1}^2, because it does not depend on the horizon",
                    "As the sum of daily forecasts that decay towards the long-run level"
                ],
                "correctExplanation": "E_t[sigma_{t+h}^2] = s2bar + (alpha + beta)^(h-1)(sigma_{t+1}^2 - s2bar); the 10-day variance is the sum of these forecasts, below 10 sigma_{t+1}^2.",
                "incorrectExplanation": "The square-root-of-time rule overstates risk after a crash and the long-run level understates it; only IGARCH (EWMA) keeps the forecast flat."
            },
            "ro": {
                "title": "Prognoze pe mai mulți pași",
                "text": "Imediat după un crah, sigma_{t+1}^2 este mult peste varianța pe termen lung. Cum prognozează GARCH(1,1) varianța următoarelor 10 zile?",
                "options": [
                    "Ca 10 ori sigma_{t+1}^2",
                    "Ca 10 ori varianța pe termen lung",
                    "Ca sigma_{t+1}^2, deoarece nu depinde de orizont",
                    "Ca sumă a prognozelor zilnice, care scad spre nivelul pe termen lung"
                ],
                "correctExplanation": "E_t[sigma_{t+h}^2] = s2bar + (alpha + beta)^(h-1)(sigma_{t+1}^2 - s2bar); varianța pe 10 zile este suma acestor prognoze, sub 10 sigma_{t+1}^2.",
                "incorrectExplanation": "Regula rădăcinii pătrate a timpului supraestimează riscul după un crah, iar nivelul pe termen lung îl subestimează; doar IGARCH (EWMA) păstrează prognoza constantă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "QLIKE",
                "text": "Why is the QLIKE loss r_t^2/h_t + ln h_t used to compare volatility forecasts?",
                "options": [
                    "It is the only loss that can be computed out of sample",
                    "It always prefers GARCH to EWMA",
                    "It ranks forecasts correctly even though r_t^2 is a noisy proxy of the true variance",
                    "It does not need the returns"
                ],
                "correctExplanation": "Patton (2011): QLIKE (like the MSE) is robust to the noise of the proxy r_t^2; it penalises forecasts that are too low more strongly.",
                "incorrectExplanation": "Any loss can be computed out of sample, and QLIKE favours no model a priori; it needs r_t^2 as the proxy of the variance."
            },
            "ro": {
                "title": "QLIKE",
                "text": "De ce se folosește funcția de pierdere QLIKE r_t^2/h_t + ln h_t pentru a compara prognozele de volatilitate?",
                "options": [
                    "Este singura pierdere care poate fi calculată în afara eșantionului",
                    "Preferă întotdeauna GARCH în locul EWMA",
                    "Ordonează corect prognozele chiar dacă r_t^2 este o aproximare zgomotoasă a varianței adevărate",
                    "Nu are nevoie de randamente"
                ],
                "correctExplanation": "Patton (2011): QLIKE (ca și MSE) este robustă la zgomotul aproximării r_t^2; ea penalizează mai puternic prognozele prea mici.",
                "incorrectExplanation": "Orice funcție de pierdere poate fi calculată în afara eșantionului, iar QLIKE nu favorizează a priori niciun model; are nevoie de r_t^2 ca aproximare a varianței."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "VaR 1% from GARCH-t",
                "text": "A GARCH(1,1)-t model gives mu = 0.05, sigma_{t+1} = 1.0 and the standardised t quantile q_0.01 = -2.6. What is the one-day VaR 1%?",
                "options": [
                    "2.65%",
                    "2.55%",
                    "2.33%",
                    "1.00%"
                ],
                "correctExplanation": "VaR = -(mu + sigma_{t+1} q) = -(0.05 - 2.6) = 2.55%: the loss exceeded with probability 1%.",
                "incorrectExplanation": "Use the quantile of the standardised t, not the Normal 2.33, and keep the sign of the mean: -(0.05 + 1.0 x (-2.6)) = 2.55%."
            },
            "ro": {
                "title": "VaR 1% din GARCH-t",
                "text": "Un model GARCH(1,1)-t dă mu = 0,05, sigma_{t+1} = 1,0 și cuantila t standardizată q_0,01 = -2,6. Care este VaR 1% pe o zi?",
                "options": [
                    "2,65%",
                    "2,55%",
                    "2,33%",
                    "1,00%"
                ],
                "correctExplanation": "VaR = -(mu + sigma_{t+1} q) = -(0,05 - 2,6) = 2,55%: pierderea depășită cu probabilitatea 1%.",
                "incorrectExplanation": "Folosim cuantila t standardizată, nu valoarea Normală 2,33, și păstrăm semnul mediei: -(0,05 + 1,0 x (-2,6)) = 2,55%."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "VaR exceedances",
                "text": "Since 2015, the EWMA-Normal VaR 1% of the DAX was exceeded on about 2.5% of the days. What is the main reason?",
                "options": [
                    "The Normal quantile is too small for heavy-tailed returns",
                    "EWMA reacts too slowly to every shock",
                    "1% of the days is too few to count",
                    "The DAX has no volatility clustering"
                ],
                "correctExplanation": "With heavy tails, a loss of 2.33 sigma is exceeded more often than 1% of the time; a Student-t quantile brings the rate much closer to 1%.",
                "incorrectExplanation": "EWMA does follow volatility clustering; the problem is the tail of the distribution, which the Normal quantile underestimates; formal backtests come in Chapter 10."
            },
            "ro": {
                "title": "Depășirile VaR",
                "text": "Din 2015, VaR 1% EWMA-Normal pentru DAX a fost depășit în circa 2,5% dintre zile. Care este motivul principal?",
                "options": [
                    "Cuantila Normală este prea mică pentru randamente cu cozi groase",
                    "EWMA reacționează prea lent la orice șoc",
                    "1% dintre zile este prea puțin pentru a număra",
                    "DAX nu are volatility clustering"
                ],
                "correctExplanation": "Cu cozi groase, o pierdere de 2,33 sigma este depășită mai des decît în 1% dintre zile; o cuantilă Student-t aduce rata mult mai aproape de 1%.",
                "incorrectExplanation": "EWMA urmărește volatility clustering; problema este coada distribuției, pe care cuantila Normală o subestimează; testele formale (backtesting) vin în Capitolul 10."
            }
        }
    ]
};
