// ============================================================
// Chapter 1 quiz bank: Data, returns and indicators (EN + RO)
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['returns'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Simple return",
                "text": "What is the simple (net) return $R_t$?",
                "options": [
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t - P_{t-1}$",
                    "$(P_t - P_{t-1}) / P_{t-1}$",
                    "$P_t / P_{t-1}$"
                ],
                "correctExplanation": "The simple return $R_t = (P_t - P_{t-1}) / P_{t-1}$ is the relative change in price.",
                "incorrectExplanation": "$R_t = (P_t - P_{t-1}) / P_{t-1}$."
            },
            "ro": {
                "title": "Randamentul simplu",
                "text": "Care este randamentul simplu (net) $R_t$?",
                "options": [
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t - P_{t-1}$",
                    "$(P_t - P_{t-1}) / P_{t-1}$",
                    "$P_t / P_{t-1}$"
                ],
                "correctExplanation": "Randamentul simplu $R_t = (P_t - P_{t-1}) / P_{t-1}$ este variația relativă a prețului.",
                "incorrectExplanation": "$R_t = (P_t - P_{t-1}) / P_{t-1}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Log return",
                "text": "What is the log (continuously compounded) return $r_t$?",
                "options": [
                    "$(P_t - P_{t-1}) / P_{t-1}$",
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t \\cdot P_{t-1}$",
                    "$\\exp(P_t) - 1$"
                ],
                "correctExplanation": "The log return is $r_t = \\ln(P_t / P_{t-1})$, the continuously compounded return.",
                "incorrectExplanation": "$r_t = \\ln(P_t / P_{t-1})$."
            },
            "ro": {
                "title": "Randamentul logaritmic",
                "text": "Care este randamentul logaritmic (compus continuu) $r_t$?",
                "options": [
                    "$(P_t - P_{t-1}) / P_{t-1}$",
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t \\cdot P_{t-1}$",
                    "$\\exp(P_t) - 1$"
                ],
                "correctExplanation": "Randamentul logaritmic este $r_t = \\ln(P_t / P_{t-1})$, randamentul compus continuu.",
                "incorrectExplanation": "$r_t = \\ln(P_t / P_{t-1})$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Volatility drag",
                "text": "What is \"volatility drag\" (variance drag)?",
                "options": [
                    "Variance grows linearly over time",
                    "The arithmetic mean return exceeds the geometric mean by about $\\sigma^2/2$",
                    "High-variance returns always have a negative mean",
                    "A method for reducing portfolio variance"
                ],
                "correctExplanation": "The geometric mean is lower than the arithmetic mean by about $\\sigma^2/2$, which erodes compounded wealth.",
                "incorrectExplanation": "Volatility drag is the gap between the arithmetic and the geometric mean return ($\\approx \\sigma^2/2$)."
            },
            "ro": {
                "title": "Volatility drag",
                "text": "Ce este „volatility drag” (variance drag)?",
                "options": [
                    "Varianța crește liniar în timp",
                    "Media aritmetică a randamentelor depășește media geometrică cu aproximativ $\\sigma^2/2$",
                    "Randamentele cu varianță mare au mereu medie negativă",
                    "O metodă de reducere a varianței portofoliului"
                ],
                "correctExplanation": "Media geometrică este mai mică decît media aritmetică cu aproximativ $\\sigma^2/2$, ceea ce erodează averea capitalizată.",
                "incorrectExplanation": "Volatility drag este diferența dintre media aritmetică și media geometrică a randamentelor ($\\approx \\sigma^2/2$)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Why log returns",
                "text": "Why are log returns preferred for statistical modelling?",
                "options": [
                    "They are always positive",
                    "They add up over time and are closer to the Normal distribution",
                    "They are easier to compute",
                    "They never have outliers"
                ],
                "correctExplanation": "Log returns are additive over time ($r_{1:T} = \\sum_t r_t$) and closer to the Normal distribution than simple returns.",
                "incorrectExplanation": "Log returns are additive over time and closer to the Normal distribution."
            },
            "ro": {
                "title": "De ce randamente logaritmice",
                "text": "De ce se preferă randamentele logaritmice în modelarea statistică?",
                "options": [
                    "Sînt mereu pozitive",
                    "Se adună în timp și sînt mai apropiate de distribuția Normală",
                    "Se calculează mai ușor",
                    "Nu au niciodată valori extreme"
                ],
                "correctExplanation": "Randamentele logaritmice sînt aditive în timp ($r_{1:T} = \\sum_t r_t$) și mai apropiate de distribuția Normală decît randamentele simple.",
                "incorrectExplanation": "Randamentele logaritmice sînt aditive în timp și mai apropiate de distribuția Normală."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF test",
                "text": "What does the augmented Dickey–Fuller (ADF) test assess?",
                "options": [
                    "Normality of returns",
                    "The presence of a unit root (non-stationarity)",
                    "Heteroskedasticity of residuals",
                    "Autocorrelation at all lags"
                ],
                "correctExplanation": "The ADF test has $H_0$: unit root (non-stationary) against $H_1$: stationary.",
                "incorrectExplanation": "The ADF test checks for a unit root, that is, non-stationarity."
            },
            "ro": {
                "title": "Testul ADF",
                "text": "Ce verifică testul Dickey–Fuller augmentat (ADF)?",
                "options": [
                    "Normalitatea randamentelor",
                    "Prezența unei rădăcini unitare (nestaționaritate)",
                    "Heteroscedasticitatea reziduurilor",
                    "Autocorelarea la toate lag-urile"
                ],
                "correctExplanation": "Testul ADF are $H_0$: rădăcină unitară (nestaționaritate) față de $H_1$: staționaritate.",
                "incorrectExplanation": "Testul ADF verifică prezența unei rădăcini unitare, adică nestaționaritatea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Jarque–Bera test",
                "text": "What does the Jarque–Bera test assess?",
                "options": [
                    "Stationarity of the series",
                    "Whether the data follow the Normal distribution, through skewness and kurtosis",
                    "Serial correlation of residuals",
                    "The presence of unit roots"
                ],
                "correctExplanation": "Jarque–Bera tests normality by checking whether skewness $= 0$ and excess kurtosis $= 0$.",
                "incorrectExplanation": "Jarque–Bera tests normality using skewness and kurtosis."
            },
            "ro": {
                "title": "Testul Jarque–Bera",
                "text": "Ce verifică testul Jarque–Bera?",
                "options": [
                    "Staționaritatea seriei",
                    "Dacă datele urmează distribuția Normală, prin asimetrie și aplatizare",
                    "Corelarea serială a reziduurilor",
                    "Prezența rădăcinilor unitare"
                ],
                "correctExplanation": "Jarque–Bera testează normalitatea verificînd dacă asimetria $= 0$ și excesul de aplatizare $= 0$.",
                "incorrectExplanation": "Jarque–Bera testează normalitatea cu ajutorul asimetriei și al aplatizării."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Parkinson estimator",
                "text": "Which volatility estimator uses the high–low price range?",
                "options": [
                    "Sample standard deviation",
                    "The Parkinson estimator",
                    "EWMA",
                    "Realised volatility"
                ],
                "correctExplanation": "Parkinson (1980) uses the daily high–low range: $\\hat\\sigma^2_P = \\frac{1}{4n\\ln 2}\\sum_i (\\ln H_i - \\ln L_i)^2$.",
                "incorrectExplanation": "The Parkinson estimator uses the high–low price range."
            },
            "ro": {
                "title": "Estimatorul Parkinson",
                "text": "Ce estimator al volatilității folosește amplitudinea maxim–minim a prețului?",
                "options": [
                    "Abaterea standard de selecție",
                    "Estimatorul Parkinson",
                    "EWMA",
                    "Volatilitatea realizată"
                ],
                "correctExplanation": "Parkinson (1980) folosește amplitudinea zilnică maxim–minim: $\\hat\\sigma^2_P = \\frac{1}{4n\\ln 2}\\sum_i (\\ln H_i - \\ln L_i)^2$.",
                "incorrectExplanation": "Estimatorul Parkinson folosește amplitudinea maxim–minim a prețului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Sharpe ratio",
                "text": "What does the Sharpe ratio measure?",
                "options": [
                    "The total return of an asset",
                    "Excess return per unit of risk (standard deviation)",
                    "The correlation between two assets",
                    "The maximum drawdown of a portfolio"
                ],
                "correctExplanation": "The Sharpe ratio $S = (\\bar r - r_f) / \\sigma$ measures risk-adjusted return.",
                "incorrectExplanation": "The Sharpe ratio measures excess return per unit of risk."
            },
            "ro": {
                "title": "Raportul Sharpe",
                "text": "Ce măsoară raportul Sharpe?",
                "options": [
                    "Randamentul total al unui activ",
                    "Randamentul în exces pe unitatea de risc (abaterea standard)",
                    "Corelația dintre două active",
                    "Drawdown-ul maxim al unui portofoliu"
                ],
                "correctExplanation": "Raportul Sharpe $S = (\\bar r - r_f) / \\sigma$ măsoară randamentul ajustat la risc.",
                "incorrectExplanation": "Raportul Sharpe măsoară randamentul în exces pe unitatea de risc."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "OHLC data",
                "text": "What do OHLC data contain?",
                "options": [
                    "Closing prices only",
                    "Open, high, low and close prices for each period",
                    "Overnight returns only",
                    "Order flow and trade data"
                ],
                "correctExplanation": "OHLC stands for open, high, low, close: the four prices recorded in each trading period.",
                "incorrectExplanation": "OHLC means the open, high, low and close prices of each period."
            },
            "ro": {
                "title": "Date OHLC",
                "text": "Ce conțin datele OHLC?",
                "options": [
                    "Doar prețurile de închidere",
                    "Prețurile de deschidere, maxim, minim și închidere pentru fiecare perioadă",
                    "Doar randamentele overnight",
                    "Fluxul de ordine și tranzacțiile"
                ],
                "correctExplanation": "OHLC înseamnă open, high, low, close: cele patru prețuri înregistrate în fiecare perioadă de tranzacționare.",
                "incorrectExplanation": "OHLC înseamnă prețurile de deschidere, maxim, minim și închidere ale fiecărei perioade."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Leverage effect",
                "text": "What is the \"leverage effect\" in financial markets?",
                "options": [
                    "Investing with borrowed money",
                    "Negative returns raise future volatility more than positive returns of the same size",
                    "Large firms have less volatile share prices",
                    "Interest rates directly set share prices"
                ],
                "correctExplanation": "The leverage effect is the asymmetric response of volatility to negative and positive returns.",
                "incorrectExplanation": "The leverage effect means that negative returns raise volatility more than positive returns."
            },
            "ro": {
                "title": "Efectul de levier",
                "text": "Ce este „efectul de levier” pe piețele financiare?",
                "options": [
                    "Investiția cu bani împrumutați",
                    "Randamentele negative cresc volatilitatea viitoare mai mult decît randamentele pozitive de aceeași mărime",
                    "Firmele mari au prețuri ale acțiunilor mai puțin volatile",
                    "Dobînzile determină direct prețurile acțiunilor"
                ],
                "correctExplanation": "Efectul de levier este reacția asimetrică a volatilității la randamente negative și pozitive.",
                "incorrectExplanation": "Efectul de levier înseamnă că randamentele negative cresc volatilitatea mai mult decît cele pozitive."
            }
        }
    ]
};
