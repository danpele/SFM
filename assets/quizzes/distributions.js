// ============================================================
// Chapter 2 quiz bank: Classical distributions and stylised facts (EN + RO)
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['distributions'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Normal distribution",
                "text": "Which property makes the Normal distribution closed under addition (the sum of two independent Normal variables is Normal)?",
                "options": [
                    "It has the thinnest possible tails",
                    "Stability (additivity): $X+Y \\sim N(\\mu_1+\\mu_2, \\sigma_1^2+\\sigma_2^2)$",
                    "Its density has a closed form",
                    "It is the only symmetric distribution"
                ],
                "correctExplanation": "The Normal distribution is stable under addition: a sum of independent Normal variables is Normal, with means and variances that add up.",
                "incorrectExplanation": "Stability (additivity) keeps sums of independent Normal variables Normal."
            },
            "ro": {
                "title": "Distribuția Normală",
                "text": "Ce proprietate face ca distribuția Normală să fie închisă la adunare (suma a două variabile Normale independente este tot Normală)?",
                "options": [
                    "Are cele mai subțiri cozi posibile",
                    "Stabilitatea (aditivitatea): $X+Y \\sim N(\\mu_1+\\mu_2, \\sigma_1^2+\\sigma_2^2)$",
                    "Densitatea sa are formă închisă",
                    "Este singura distribuție simetrică"
                ],
                "correctExplanation": "Distribuția Normală este stabilă la adunare: suma unor variabile Normale independente este Normală, iar mediile și varianțele se adună.",
                "incorrectExplanation": "Stabilitatea (aditivitatea) păstrează Normale sumele de variabile Normale independente."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Stylised facts",
                "text": "Which of the following is NOT a stylised fact of financial return distributions?",
                "options": [
                    "Excess kurtosis (heavy tails)",
                    "Volatility clustering",
                    "Negative skewness of equity returns",
                    "Returns follow a symmetric Normal distribution"
                ],
                "correctExplanation": "Returns depart from normality: heavy tails, volatility clustering and negative skewness, none of which the Normal distribution captures.",
                "incorrectExplanation": "Returns do NOT follow a symmetric Normal distribution; that contradicts the documented stylised facts."
            },
            "ro": {
                "title": "Fapte stilizate",
                "text": "Care dintre următoarele NU este un fapt stilizat al distribuției randamentelor financiare?",
                "options": [
                    "Excesul de aplatizare (cozi groase)",
                    "Grupările de volatilitate",
                    "Asimetria negativă a randamentelor acțiunilor",
                    "Randamentele urmează o distribuție Normală simetrică"
                ],
                "correctExplanation": "Randamentele se abat de la normalitate: cozi groase, grupări de volatilitate și asimetrie negativă, pe care distribuția Normală nu le surprinde.",
                "incorrectExplanation": "Randamentele NU urmează o distribuție Normală simetrică; asta contrazice faptele stilizate documentate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Student-t distribution",
                "text": "When a Student-t distribution is fitted to daily equity returns, in which range do the estimated degrees of freedom usually fall?",
                "options": [
                    "1–2 (extremely heavy tails, close to Cauchy)",
                    "3–8 (heavy tails, finite variance for $\\nu > 2$)",
                    "20–50 (almost Gaussian)",
                    "Over 100 (indistinguishable from the Normal)"
                ],
                "correctExplanation": "Empirical studies find $\\nu \\approx 3$–$8$ for daily stock returns: heavy tails, but finite variance ($\\nu > 2$).",
                "incorrectExplanation": "Daily equity returns usually give $\\nu \\in [3, 8]$, which captures the heavy tails."
            },
            "ro": {
                "title": "Distribuția Student-t",
                "text": "Cînd se ajustează o distribuție Student-t pe randamentele zilnice ale acțiunilor, în ce interval se află de obicei numărul estimat de grade de libertate?",
                "options": [
                    "1–2 (cozi extrem de groase, apropiate de Cauchy)",
                    "3–8 (cozi groase, varianță finită pentru $\\nu > 2$)",
                    "20–50 (aproape gaussiană)",
                    "Peste 100 (imposibil de deosebit de distribuția Normală)"
                ],
                "correctExplanation": "Studiile empirice găsesc $\\nu \\approx 3$–$8$ pentru randamentele zilnice ale acțiunilor: cozi groase, dar varianță finită ($\\nu > 2$).",
                "incorrectExplanation": "Randamentele zilnice ale acțiunilor dau de obicei $\\nu \\in [3, 8]$, ceea ce surprinde cozile groase."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Skewed Student-t",
                "text": "What does the skewed Student-t distribution (Hansen, 1994) capture in addition to the symmetric Student-t?",
                "options": [
                    "Time-varying volatility",
                    "Negative skewness (asymmetry between gains and losses)",
                    "Multimodal return distributions",
                    "Bounded support for extreme returns"
                ],
                "correctExplanation": "A skewness parameter $\\lambda$ lets the left tail (losses) be heavier than the right tail (gains), as in equity returns.",
                "incorrectExplanation": "The skewed Student-t captures the negative skewness of equity returns."
            },
            "ro": {
                "title": "Student-t asimetrică",
                "text": "Ce surprinde în plus distribuția Student-t asimetrică (Hansen, 1994) față de Student-t simetrică?",
                "options": [
                    "Volatilitatea variabilă în timp",
                    "Asimetria negativă (diferența dintre cîștiguri și pierderi)",
                    "Distribuții multimodale ale randamentelor",
                    "Suport mărginit pentru randamentele extreme"
                ],
                "correctExplanation": "Un parametru de asimetrie $\\lambda$ permite cozii stîngi (pierderile) să fie mai groasă decît coada dreaptă (cîștigurile), ca la randamentele acțiunilor.",
                "incorrectExplanation": "Student-t asimetrică surprinde asimetria negativă a randamentelor acțiunilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Stable distributions",
                "text": "For a stable distribution with index $\\alpha < 2$, which statement about moments is correct?",
                "options": [
                    "All moments exist and are finite",
                    "The variance is infinite; only moments of order $p < \\alpha$ are finite",
                    "The mean always exists, the variance may not",
                    "No moment exists for any $\\alpha < 2$"
                ],
                "correctExplanation": "For $\\alpha < 2$ only moments of order $p < \\alpha$ are finite: the mean exists for $\\alpha > 1$, the variance never does.",
                "incorrectExplanation": "The variance is infinite for $\\alpha < 2$; only (fractional) moments of order below $\\alpha$ exist."
            },
            "ro": {
                "title": "Distribuții stabile",
                "text": "Pentru o distribuție stabilă cu indicele $\\alpha < 2$, ce afirmație despre momente este corectă?",
                "options": [
                    "Toate momentele există și sînt finite",
                    "Varianța este infinită; doar momentele de ordin $p < \\alpha$ sînt finite",
                    "Media există mereu, varianța poate să nu existe",
                    "Niciun moment nu există pentru $\\alpha < 2$"
                ],
                "correctExplanation": "Pentru $\\alpha < 2$ sînt finite doar momentele de ordin $p < \\alpha$: media există pentru $\\alpha > 1$, varianța niciodată.",
                "incorrectExplanation": "Varianța este infinită pentru $\\alpha < 2$; există doar momentele (fracționare) de ordin mai mic decît $\\alpha$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Extreme value theory",
                "text": "In the peaks-over-threshold (POT) approach of EVT, which distribution models the exceedances over a high threshold?",
                "options": [
                    "The generalised extreme value (GEV) distribution",
                    "The Normal distribution with adjusted variance",
                    "The generalised Pareto distribution (GPD)",
                    "The exponential distribution"
                ],
                "correctExplanation": "By the Pickands–Balkema–de Haan theorem, exceedances over a high enough threshold follow a generalised Pareto distribution (GPD).",
                "incorrectExplanation": "The GPD is the limit distribution of threshold exceedances; the GEV belongs to the block-maxima approach."
            },
            "ro": {
                "title": "Teoria valorilor extreme",
                "text": "În abordarea peaks over threshold (POT) din EVT, ce distribuție modelează depășirile unui prag ridicat?",
                "options": [
                    "Distribuția generalizată a valorilor extreme (GEV)",
                    "Distribuția Normală cu varianță ajustată",
                    "Distribuția Pareto generalizată (GPD)",
                    "Distribuția exponențială"
                ],
                "correctExplanation": "Conform teoremei Pickands–Balkema–de Haan, depășirile unui prag suficient de ridicat urmează o distribuție Pareto generalizată (GPD).",
                "incorrectExplanation": "GPD este distribuția limită a depășirilor de prag; GEV aparține abordării block maxima."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Model selection",
                "text": "The Akaike information criterion is $\\text{AIC} = -2 \\ln L + 2k$. How is it used for model selection?",
                "options": [
                    "Choose the model with the highest AIC",
                    "Choose the model with the lowest AIC (balances fit and complexity)",
                    "AIC is valid only for nested models",
                    "AIC penalises underfitting only, not overfitting"
                ],
                "correctExplanation": "A lower AIC means a better trade-off between fit ($-2 \\ln L$) and complexity ($2k$); unlike likelihood-ratio tests, AIC also compares non-nested models.",
                "incorrectExplanation": "The model with the lowest AIC gives the best balance between fit and parsimony."
            },
            "ro": {
                "title": "Selecția modelului",
                "text": "Criteriul informațional Akaike este $\\text{AIC} = -2 \\ln L + 2k$. Cum se folosește la selecția modelului?",
                "options": [
                    "Se alege modelul cu cel mai mare AIC",
                    "Se alege modelul cu cel mai mic AIC (echilibrează ajustarea și complexitatea)",
                    "AIC este valid doar pentru modele imbricate",
                    "AIC penalizează doar subajustarea, nu și supraajustarea"
                ],
                "correctExplanation": "Un AIC mai mic înseamnă un compromis mai bun între ajustare ($-2 \\ln L$) și complexitate ($2k$); spre deosebire de testele raportului de verosimilitate, AIC compară și modele neimbricate.",
                "incorrectExplanation": "Modelul cu cel mai mic AIC oferă cel mai bun echilibru între ajustare și parcimonie."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Expected Shortfall and VaR",
                "text": "What is the main advantage of Expected Shortfall (ES) over Value-at-Risk (VaR) as a risk measure?",
                "options": [
                    "ES is always smaller than VaR",
                    "ES is coherent (subadditive), so diversification is never penalised",
                    "ES is easier to backtest than VaR",
                    "ES ignores the size of tail losses"
                ],
                "correctExplanation": "ES is a coherent risk measure: it is subadditive, $\\text{ES}(A+B) \\leq \\text{ES}(A) + \\text{ES}(B)$, so diversification does not raise measured risk. VaR can violate this property.",
                "incorrectExplanation": "ES is coherent and subadditive, whereas VaR can penalise diversification."
            },
            "ro": {
                "title": "Expected Shortfall și VaR",
                "text": "Care este principalul avantaj al Expected Shortfall (ES) față de Value-at-Risk (VaR) ca măsură de risc?",
                "options": [
                    "ES este mereu mai mic decît VaR",
                    "ES este coerentă (subaditivă), deci diversificarea nu este penalizată",
                    "ES se testează prin backtesting mai ușor decît VaR",
                    "ES ignoră mărimea pierderilor din coadă"
                ],
                "correctExplanation": "ES este o măsură de risc coerentă: este subaditivă, $\\text{ES}(A+B) \\leq \\text{ES}(A) + \\text{ES}(B)$, deci diversificarea nu crește riscul măsurat. VaR poate încălca această proprietate.",
                "incorrectExplanation": "ES este coerentă și subaditivă, pe cînd VaR poate penaliza diversificarea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Basel backtesting",
                "text": "Under the Basel rules a bank backtests its 1% VaR over 250 trading days. How many exceedances are expected, and what is the green-zone threshold?",
                "options": [
                    "Expected 25; green zone below 10",
                    "Expected 2.5; green zone below 5 exceedances",
                    "Expected 0; any exceedance means the red zone",
                    "Expected 12.5; green zone below 20"
                ],
                "correctExplanation": "At the 1% level over 250 days, the expected number of exceedances is $250 \\times 0.01 = 2.5$. The Basel traffic light: green 0–4, yellow 5–9, red 10 or more.",
                "incorrectExplanation": "The expected number of exceedances is 2.5, and the green zone means fewer than 5 exceedances."
            },
            "ro": {
                "title": "Backtesting Basel",
                "text": "Conform regulilor Basel, o bancă face backtesting pentru VaR 1% pe 250 de zile de tranzacționare. Cîte depășiri se așteaptă și care este pragul zonei verzi?",
                "options": [
                    "Așteptate 25; zona verde sub 10",
                    "Așteptate 2,5; zona verde sub 5 depășiri",
                    "Așteptate 0; orice depășire înseamnă zona roșie",
                    "Așteptate 12,5; zona verde sub 20"
                ],
                "correctExplanation": "La nivelul de 1% pe 250 de zile, numărul așteptat de depășiri este $250 \\times 0{,}01 = 2{,}5$. Semaforul Basel: verde 0–4, galben 5–9, roșu 10 sau mai multe.",
                "incorrectExplanation": "Numărul așteptat de depășiri este 2,5, iar zona verde înseamnă mai puțin de 5 depășiri."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH-t model",
                "text": "What does the GARCH(1,1)-t model capture that a plain Student-t distribution cannot?",
                "options": [
                    "Heavy tails only",
                    "Time-varying volatility only",
                    "Both volatility clustering (time-varying conditional variance) and excess kurtosis from t innovations",
                    "Mean reversion in returns"
                ],
                "correctExplanation": "The GARCH part captures volatility clustering (time-varying $\\sigma_t^2$); the Student-t innovations add excess kurtosis even in the standardised residuals.",
                "incorrectExplanation": "GARCH-t models volatility clustering and heavy tails jointly, through the conditional variance and the t-distributed innovations."
            },
            "ro": {
                "title": "Modelul GARCH-t",
                "text": "Ce surprinde modelul GARCH(1,1)-t și nu poate surprinde o simplă distribuție Student-t?",
                "options": [
                    "Doar cozile groase",
                    "Doar volatilitatea variabilă în timp",
                    "Atît grupările de volatilitate (varianța condiționată variabilă în timp), cît și excesul de aplatizare din inovațiile t",
                    "Revenirea la medie a randamentelor"
                ],
                "correctExplanation": "Partea GARCH surprinde grupările de volatilitate ($\\sigma_t^2$ variabilă în timp); inovațiile Student-t adaugă exces de aplatizare chiar și în reziduurile standardizate.",
                "incorrectExplanation": "GARCH-t modelează împreună grupările de volatilitate și cozile groase, prin varianța condiționată și inovațiile cu distribuție t."
            }
        }
    ]
};
