// ============================================================
// Chapter 1 quiz bank: Data, returns and indicators (EN + RO)
// 24 questions, 20 drawn per attempt.
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
                "text": "What is the simple (net) return $R_t$ from $t-1$ to $t$?",
                "options": [
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t - P_{t-1}$",
                    "$P_t / P_{t-1} - 1$",
                    "$P_{t-1} / P_t - 1$"
                ],
                "correctExplanation": "The simple return is the relative price change, $R_t = P_t/P_{t-1} - 1$.",
                "incorrectExplanation": "The simple return is $R_t = P_t/P_{t-1} - 1$; $\\ln(P_t/P_{t-1})$ is the log return."
            },
            "ro": {
                "title": "Randamentul simplu",
                "text": "Care este randamentul simplu (net) $R_t$ de la $t-1$ la $t$?",
                "options": [
                    "$\\ln(P_t / P_{t-1})$",
                    "$P_t - P_{t-1}$",
                    "$P_t / P_{t-1} - 1$",
                    "$P_{t-1} / P_t - 1$"
                ],
                "correctExplanation": "Randamentul simplu este variația relativă a prețului, $R_t = P_t/P_{t-1} - 1$.",
                "incorrectExplanation": "Randamentul simplu este $R_t = P_t/P_{t-1} - 1$; $\\ln(P_t/P_{t-1})$ este randamentul logaritmic."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Log vs simple return",
                "text": "For any price move with $R_t \\neq 0$, how does the log return $r_t = \\ln(1 + R_t)$ compare with $R_t$?",
                "options": [
                    "$r_t < R_t$",
                    "$r_t > R_t$",
                    "$r_t = R_t$",
                    "It depends on the sign of $R_t$"
                ],
                "correctExplanation": "The logarithm is concave, so $\\ln(1 + x) < x$ for every $x \\neq 0$: the log return is always below the simple return.",
                "incorrectExplanation": "Since $\\ln(1 + x) < x$ for every $x > -1$, $x \\neq 0$, the log return is always smaller, for gains and for losses."
            },
            "ro": {
                "title": "Randament logaritmic vs simplu",
                "text": "Pentru orice mișcare de preț cu $R_t \\neq 0$, cum se compară randamentul logaritmic $r_t = \\ln(1 + R_t)$ cu $R_t$?",
                "options": [
                    "$r_t < R_t$",
                    "$r_t > R_t$",
                    "$r_t = R_t$",
                    "Depinde de semnul lui $R_t$"
                ],
                "correctExplanation": "Logaritmul este concav, deci $\\ln(1 + x) < x$ pentru orice $x \\neq 0$: randamentul logaritmic este întotdeauna sub cel simplu.",
                "incorrectExplanation": "Deoarece $\\ln(1 + x) < x$ pentru orice $x > -1$, $x \\neq 0$, randamentul logaritmic este întotdeauna mai mic, la cîștiguri și la pierderi."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Up 10%, down 10%",
                "text": "A price rises by 10% and then falls by 10%. What is the total simple return?",
                "options": [
                    "0%",
                    "+1%",
                    "-10%",
                    "-1%"
                ],
                "correctExplanation": "$1.1 \\times 0.9 - 1 = -1\\%$: simple returns compound, they do not add.",
                "incorrectExplanation": "Compound the gross returns: $1.1 \\times 0.9 = 0.99$, a loss of 1%."
            },
            "ro": {
                "title": "+10%, apoi -10%",
                "text": "Un preț crește cu 10% și apoi scade cu 10%. Care este randamentul simplu total?",
                "options": [
                    "0%",
                    "+1%",
                    "-10%",
                    "-1%"
                ],
                "correctExplanation": "$1{,}1 \\times 0{,}9 - 1 = -1\\%$: randamentele simple se compun, nu se adună.",
                "incorrectExplanation": "Compuneți randamentele brute: $1{,}1 \\times 0{,}9 = 0{,}99$, o pierdere de 1%."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Recovering from a loss",
                "text": "A share loses 50%. What gain does it need to return to its previous price?",
                "options": [
                    "50%",
                    "100%",
                    "150%",
                    "200%"
                ],
                "correctExplanation": "A loss $x$ needs a gain $x/(1 - x)$: $0.5/0.5 = 100\\%$.",
                "incorrectExplanation": "From half the price, the share must double: a gain of $x/(1 - x) = 100\\%$."
            },
            "ro": {
                "title": "Recuperarea unei pierderi",
                "text": "O acțiune pierde 50%. Ce cîștig îi trebuie ca să revină la prețul anterior?",
                "options": [
                    "50%",
                    "100%",
                    "150%",
                    "200%"
                ],
                "correctExplanation": "O pierdere $x$ cere un cîștig $x/(1 - x)$: $0{,}5/0{,}5 = 100\\%$.",
                "incorrectExplanation": "De la jumătate din preț, acțiunea trebuie să se dubleze: un cîștig de $x/(1 - x) = 100\\%$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Aggregation over time",
                "text": "Which returns add up exactly over time, so that the $k$-day return is the sum of the daily returns?",
                "options": [
                    "Simple returns",
                    "Log returns",
                    "Both",
                    "Neither"
                ],
                "correctExplanation": "$r_t(k) = \\ln(P_t/P_{t-k}) = r_t + \\dots + r_{t-k+1}$.",
                "incorrectExplanation": "Log returns add over time; simple returns must be compounded, $1 + R_t(k) = \\prod (1 + R_{t-j})$."
            },
            "ro": {
                "title": "Agregarea în timp",
                "text": "Ce randamente se adună exact în timp, astfel încît randamentul pe $k$ zile să fie suma randamentelor zilnice?",
                "options": [
                    "Randamentele simple",
                    "Randamentele logaritmice",
                    "Ambele",
                    "Niciunele"
                ],
                "correctExplanation": "$r_t(k) = \\ln(P_t/P_{t-k}) = r_t + \\dots + r_{t-k+1}$.",
                "incorrectExplanation": "Randamentele logaritmice se adună în timp; cele simple trebuie compuse, $1 + R_t(k) = \\prod (1 + R_{t-j})$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Portfolio return",
                "text": "A portfolio has weights $w_i$ set at the start of the period. Which statement is exact?",
                "options": [
                    "$R_p = \\sum_i w_i R_i$ for simple returns",
                    "$r_p = \\sum_i w_i r_i$ for log returns",
                    "Both are exact",
                    "Neither is exact"
                ],
                "correctExplanation": "The value of the portfolio is the weighted sum of the values of its parts, so simple returns aggregate exactly across assets.",
                "incorrectExplanation": "Across assets, simple returns aggregate exactly; the weighted sum of log returns is only an approximation of $r_p = \\ln(\\sum_i w_i e^{r_i})$."
            },
            "ro": {
                "title": "Randamentul portofoliului",
                "text": "Un portofoliu are ponderile $w_i$ fixate la începutul perioadei. Ce afirmație este exactă?",
                "options": [
                    "$R_p = \\sum_i w_i R_i$ pentru randamente simple",
                    "$r_p = \\sum_i w_i r_i$ pentru randamente logaritmice",
                    "Ambele sînt exacte",
                    "Niciuna nu este exactă"
                ],
                "correctExplanation": "Valoarea portofoliului este suma ponderată a valorilor componentelor, deci randamentele simple se agregă exact între active.",
                "incorrectExplanation": "Între active, randamentele simple se agregă exact; suma ponderată a randamentelor logaritmice este doar o aproximare a lui $r_p = \\ln(\\sum_i w_i e^{r_i})$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Adjusted close",
                "text": "Why should stock returns be computed from the adjusted close rather than from the close?",
                "options": [
                    "The adjusted close is always higher",
                    "The adjusted close removes weekends",
                    "It removes artificial jumps from splits and consolidations and includes dividends",
                    "Exchanges publish only adjusted prices"
                ],
                "correctExplanation": "Corporate actions change the traded price but not the value of a holding; the adjusted close corrects all earlier prices for them.",
                "incorrectExplanation": "The adjusted close multiplies earlier prices by the factors of later corporate actions, so returns have no artificial jumps and include dividends."
            },
            "ro": {
                "title": "Prețul ajustat",
                "text": "De ce se calculează randamentele acțiunilor din prețul ajustat, nu din prețul de închidere?",
                "options": [
                    "Prețul ajustat este întotdeauna mai mare",
                    "Prețul ajustat elimină weekendurile",
                    "Elimină salturile artificiale de la splituri și consolidări și include dividendele",
                    "Bursele publică doar prețuri ajustate"
                ],
                "correctExplanation": "Evenimentele corporative schimbă prețul de tranzacționare, dar nu și valoarea deținerii; prețul ajustat corectează toate prețurile anterioare.",
                "incorrectExplanation": "Prețul ajustat înmulțește prețurile anterioare cu factorii evenimentelor ulterioare, astfel că randamentele nu au salturi artificiale și includ dividendele."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Share consolidation",
                "text": "Ten old shares are consolidated into one. If the close is not adjusted, what happens to the daily log return on that day?",
                "options": [
                    "It is about $-230\\%$",
                    "It is unchanged",
                    "It becomes zero",
                    "It is about $+230\\%$, close to $\\ln 10$"
                ],
                "correctExplanation": "The traded price is multiplied by about ten, so $\\ln(P_t/P_{t-1}) \\approx \\ln 10 \\approx 2.30$: an artificial gain of about 230%.",
                "incorrectExplanation": "After a 10-to-1 consolidation the price is about ten times higher, giving an artificial log return near $\\ln 10$."
            },
            "ro": {
                "title": "Consolidarea acțiunilor",
                "text": "Zece acțiuni vechi se consolidează într-una. Dacă prețul de închidere nu este ajustat, ce se întîmplă cu randamentul logaritmic din acea zi?",
                "options": [
                    "Este aproximativ $-230\\%$",
                    "Rămîne neschimbat",
                    "Devine zero",
                    "Este aproximativ $+230\\%$, aproape de $\\ln 10$"
                ],
                "correctExplanation": "Prețul de tranzacționare se înmulțește cu aproximativ zece, deci $\\ln(P_t/P_{t-1}) \\approx \\ln 10 \\approx 2{,}30$: un cîștig artificial de circa 230%.",
                "incorrectExplanation": "După o consolidare 10-la-1 prețul este de circa zece ori mai mare, ceea ce dă un randament logaritmic artificial aproape de $\\ln 10$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Total return index",
                "text": "What is the difference between the BET and the BET-TR?",
                "options": [
                    "The BET-TR contains more stocks",
                    "The BET-TR reinvests dividends; the BET is a price index",
                    "The BET-TR is in euro",
                    "There is no difference"
                ],
                "correctExplanation": "A total return index reinvests the dividends, so over long periods it grows faster than the price index.",
                "incorrectExplanation": "BET is a price index; BET-TR (total return) reinvests dividends."
            },
            "ro": {
                "title": "Indice de randament total",
                "text": "Care este diferența dintre BET și BET-TR?",
                "options": [
                    "BET-TR conține mai multe acțiuni",
                    "BET-TR reinvestește dividendele; BET este un indice de preț",
                    "BET-TR este în euro",
                    "Nu există nicio diferență"
                ],
                "correctExplanation": "Un indice de randament total reinvestește dividendele, deci pe perioade lungi crește mai repede decît indicele de preț.",
                "incorrectExplanation": "BET este un indice de preț; BET-TR (randament total) reinvestește dividendele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Conflicting data sources",
                "text": "Two data sources give different EUR/RON rates for the same day. Which one is the reference?",
                "options": [
                    "The BNR reference rate, published by the owner of the number",
                    "The one with the higher volatility",
                    "The average of the two",
                    "The one with more observations"
                ],
                "correctExplanation": "The central bank publishes the official reference rate; a vendor only collects it, sometimes with errors.",
                "incorrectExplanation": "When sources disagree, use the owner of the number: for the EUR/RON reference rate, the BNR."
            },
            "ro": {
                "title": "Surse de date în conflict",
                "text": "Două surse dau cursuri EUR/RON diferite pentru aceeași zi. Care este referința?",
                "options": [
                    "Cursul de referință BNR, publicat de proprietarul cifrei",
                    "Cea cu volatilitatea mai mare",
                    "Media celor două",
                    "Cea cu mai multe observații"
                ],
                "correctExplanation": "Banca centrală publică cursul oficial de referință; un furnizor doar îl colectează, uneori cu erori.",
                "incorrectExplanation": "Cînd sursele diferă, folosiți proprietarul cifrei: pentru cursul de referință EUR/RON, BNR."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Survivorship bias",
                "text": "What is survivorship bias?",
                "options": [
                    "Using information not available at the decision date",
                    "Trying many strategies and reporting the best one",
                    "Keeping only firms or funds that still exist, so average returns look too high",
                    "Using a too short sample"
                ],
                "correctExplanation": "Firms that went bankrupt or were delisted disappear from the sample; the survivors had better returns on average.",
                "incorrectExplanation": "Survivorship bias comes from dropping the firms or funds that disappeared, which are mostly the losers."
            },
            "ro": {
                "title": "Survivorship bias",
                "text": "Ce este survivorship bias?",
                "options": [
                    "Folosirea unei informații indisponibile la data deciziei",
                    "Încercarea multor strategii și raportarea celei mai bune",
                    "Păstrarea doar a firmelor sau fondurilor care încă există, astfel încît randamentele medii par prea mari",
                    "Folosirea unui eșantion prea scurt"
                ],
                "correctExplanation": "Firmele falimentare sau delistate dispar din eșantion; supraviețuitoarele au avut în medie randamente mai bune.",
                "incorrectExplanation": "Survivorship bias provine din eliminarea firmelor sau fondurilor dispărute, care sînt în mare parte perdante."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Annualising volatility",
                "text": "A stock has a daily standard deviation of 1% and trades 252 days a year. What is its annualised volatility (uncorrelated returns)?",
                "options": [
                    "252%",
                    "about 15.9%",
                    "1%",
                    "about 3.65%"
                ],
                "correctExplanation": "Volatility scales with the square root of time: $\\sqrt{252} \\times 1\\% \\approx 15.9\\%$.",
                "incorrectExplanation": "Use $\\sigma_{year} = \\sqrt{q}\\,\\sigma$, not $q\\,\\sigma$: $\\sqrt{252} \\times 1\\% \\approx 15.9\\%$."
            },
            "ro": {
                "title": "Anualizarea volatilității",
                "text": "O acțiune are abaterea standard zilnică 1% și se tranzacționează 252 de zile pe an. Care este volatilitatea anualizată (randamente necorelate)?",
                "options": [
                    "252%",
                    "aproximativ 15,9%",
                    "1%",
                    "aproximativ 3,65%"
                ],
                "correctExplanation": "Volatilitatea crește cu rădăcina pătrată a timpului: $\\sqrt{252} \\times 1\\% \\approx 15{,}9\\%$.",
                "incorrectExplanation": "Folosiți $\\sigma_{an} = \\sqrt{q}\\,\\sigma$, nu $q\\,\\sigma$: $\\sqrt{252} \\times 1\\% \\approx 15{,}9\\%$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "CAGR",
                "text": "A price grows from 100 to 400 in 10 years. Which formula gives the CAGR?",
                "options": [
                    "$(400/100)^{1/10} - 1$",
                    "$(400/100 - 1)/10$",
                    "$\\ln(400/100)/10$",
                    "$400/100 - 1$"
                ],
                "correctExplanation": "The CAGR is the constant yearly rate that turns 100 into 400 in 10 years: $4^{1/10} - 1 \\approx 14.9\\%$.",
                "incorrectExplanation": "The CAGR compounds: $(P_T/P_0)^{1/Y} - 1$; dividing the total gain by the number of years ignores compounding."
            },
            "ro": {
                "title": "CAGR",
                "text": "Un preț crește de la 100 la 400 în 10 ani. Ce formulă dă CAGR?",
                "options": [
                    "$(400/100)^{1/10} - 1$",
                    "$(400/100 - 1)/10$",
                    "$\\ln(400/100)/10$",
                    "$400/100 - 1$"
                ],
                "correctExplanation": "CAGR este rata anuală constantă care transformă 100 în 400 în 10 ani: $4^{1/10} - 1 \\approx 14{,}9\\%$.",
                "incorrectExplanation": "CAGR compune: $(P_T/P_0)^{1/Y} - 1$; împărțirea cîștigului total la numărul de ani ignoră compunerea."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Precision of the mean",
                "text": "How can the standard error of an annual mean return be reduced?",
                "options": [
                    "Use hourly instead of daily data over the same years",
                    "Use log instead of simple returns",
                    "Annualise with $\\sqrt{252}$",
                    "Use more years of data"
                ],
                "correctExplanation": "The standard error of the annual mean is $\\sigma_{year}/\\sqrt{Y}$: only more years help.",
                "incorrectExplanation": "With the same years, a higher frequency does not change the mean, which depends on the first and last price; the SE falls as $1/\\sqrt{Y}$."
            },
            "ro": {
                "title": "Precizia mediei",
                "text": "Cum se poate reduce eroarea standard a randamentului mediu anual?",
                "options": [
                    "Folosind date orare în locul celor zilnice, pe aceiași ani",
                    "Folosind randamente logaritmice în locul celor simple",
                    "Anualizînd cu $\\sqrt{252}$",
                    "Folosind mai mulți ani de date"
                ],
                "correctExplanation": "Eroarea standard a mediei anuale este $\\sigma_{an}/\\sqrt{Y}$: doar mai mulți ani ajută.",
                "incorrectExplanation": "Pe aceiași ani, o frecvență mai mare nu schimbă media, care depinde de primul și ultimul preț; SE scade ca $1/\\sqrt{Y}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Skewness",
                "text": "Daily index returns often have negative skewness. What does this mean?",
                "options": [
                    "Returns are on average negative",
                    "Large losses are more frequent than large gains of the same size",
                    "The volatility is decreasing",
                    "The distribution is the Normal distribution"
                ],
                "correctExplanation": "Negative skewness means a longer left tail: large falls are more frequent than large rises.",
                "incorrectExplanation": "Skewness measures asymmetry; a negative value means a longer left tail, not a negative mean."
            },
            "ro": {
                "title": "Asimetria",
                "text": "Randamentele zilnice ale indicilor au adesea asimetrie negativă. Ce înseamnă aceasta?",
                "options": [
                    "Randamentele sînt în medie negative",
                    "Pierderile mari sînt mai frecvente decît cîștigurile mari de aceeași mărime",
                    "Volatilitatea scade",
                    "Distribuția este distribuția Normală"
                ],
                "correctExplanation": "Asimetria negativă înseamnă o coadă stîngă mai lungă: căderile mari sînt mai frecvente decît creșterile mari.",
                "incorrectExplanation": "Asimetria măsoară lipsa de simetrie; o valoare negativă înseamnă o coadă stîngă mai lungă, nu o medie negativă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Excess kurtosis",
                "text": "What is the excess kurtosis of the Normal distribution, and what does a large positive value for returns mean?",
                "options": [
                    "3; light tails",
                    "0; light tails",
                    "0; heavy tails, extreme days more frequent than under the Normal distribution",
                    "1; a skewed distribution"
                ],
                "correctExplanation": "Excess kurtosis is $K - 3$, zero for the Normal distribution; positive values indicate heavy tails.",
                "incorrectExplanation": "The Normal distribution has kurtosis 3, so excess kurtosis 0; returns typically have large positive excess kurtosis (heavy tails)."
            },
            "ro": {
                "title": "Excesul de aplatizare",
                "text": "Care este excesul de aplatizare al distribuției Normale și ce înseamnă o valoare pozitivă mare pentru randamente?",
                "options": [
                    "3; cozi subțiri",
                    "0; cozi subțiri",
                    "0; cozi groase, zile extreme mai frecvente decît sub distribuția Normală",
                    "1; o distribuție asimetrică"
                ],
                "correctExplanation": "Excesul de aplatizare este $K - 3$, zero pentru distribuția Normală; valorile pozitive indică cozi groase.",
                "incorrectExplanation": "Distribuția Normală are aplatizarea 3, deci exces 0; randamentele au de obicei exces mare pozitiv (cozi groase)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Sharpe ratio",
                "text": "What does the Sharpe ratio measure?",
                "options": [
                    "Excess return per unit of total volatility",
                    "Return per unit of maximum drawdown",
                    "Return per unit of downside deviation",
                    "Return per unit of systematic risk"
                ],
                "correctExplanation": "$\\text{SR} = (\\mu - r_f)/\\sigma$: excess return per unit of total risk.",
                "incorrectExplanation": "Return per unit of downside deviation is the Sortino ratio; per unit of maximum drawdown, the Calmar ratio."
            },
            "ro": {
                "title": "Sharpe ratio",
                "text": "Ce măsoară Sharpe ratio?",
                "options": [
                    "Randamentul în exces pe unitatea de volatilitate totală",
                    "Randamentul pe unitatea de maximum drawdown",
                    "Randamentul pe unitatea de abatere negativă",
                    "Randamentul pe unitatea de risc sistematic"
                ],
                "correctExplanation": "$\\text{SR} = (\\mu - r_f)/\\sigma$: randamentul în exces pe unitatea de risc total.",
                "incorrectExplanation": "Randamentul pe unitatea de abatere negativă este Sortino ratio; pe unitatea de maximum drawdown, Calmar ratio."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Annualising the Sharpe ratio",
                "text": "A daily Sharpe ratio is 0.05 for a stock that trades 252 days a year. What is the annual Sharpe ratio (uncorrelated returns)?",
                "options": [
                    "$0.05 \\times 252 = 12.6$",
                    "0.05",
                    "$0.05 \\times \\sqrt{252} \\approx 0.79$",
                    "$0.05/\\sqrt{252}$"
                ],
                "correctExplanation": "The mean scales with $q$ and the volatility with $\\sqrt{q}$, so the Sharpe ratio scales with $\\sqrt{q}$.",
                "incorrectExplanation": "Annual SR $= \\sqrt{q} \\times$ daily SR $= \\sqrt{252} \\times 0.05 \\approx 0.79$."
            },
            "ro": {
                "title": "Anualizarea Sharpe ratio",
                "text": "Sharpe ratio zilnic este 0,05 pentru o acțiune tranzacționată 252 de zile pe an. Care este Sharpe ratio anual (randamente necorelate)?",
                "options": [
                    "$0{,}05 \\times 252 = 12{,}6$",
                    "0,05",
                    "$0{,}05 \\times \\sqrt{252} \\approx 0{,}79$",
                    "$0{,}05/\\sqrt{252}$"
                ],
                "correctExplanation": "Media crește cu $q$, iar volatilitatea cu $\\sqrt{q}$, deci Sharpe ratio crește cu $\\sqrt{q}$.",
                "incorrectExplanation": "SR anual $= \\sqrt{q} \\times$ SR zilnic $= \\sqrt{252} \\times 0{,}05 \\approx 0{,}79$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Comparing Sharpe ratios",
                "text": "Two assets have annual Sharpe ratios of 1.05 and 0.93, each estimated from about 12 years of data with a standard error near 0.3. What can we conclude?",
                "options": [
                    "The first asset is significantly better",
                    "The difference is well within the estimation error",
                    "The second asset is significantly better",
                    "Both Sharpe ratios are zero"
                ],
                "correctExplanation": "With standard errors near 0.3, a difference of 0.12 is far from significant.",
                "incorrectExplanation": "Compare the difference with the standard errors: a gap of 0.12 is much smaller than the estimation error."
            },
            "ro": {
                "title": "Compararea Sharpe ratio",
                "text": "Două active au Sharpe ratio anual 1,05 și 0,93, fiecare estimat din circa 12 ani de date, cu eroare standard de aproximativ 0,3. Ce putem concluziona?",
                "options": [
                    "Primul activ este semnificativ mai bun",
                    "Diferența este mult sub eroarea de estimare",
                    "Al doilea activ este semnificativ mai bun",
                    "Ambele Sharpe ratio sînt zero"
                ],
                "correctExplanation": "Cu erori standard de circa 0,3, o diferență de 0,12 este departe de a fi semnificativă.",
                "incorrectExplanation": "Comparați diferența cu erorile standard: o diferență de 0,12 este mult mai mică decît eroarea de estimare."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Sortino ratio",
                "text": "How does the Sortino ratio differ from the Sharpe ratio?",
                "options": [
                    "It uses the maximum drawdown in the denominator",
                    "It uses the median instead of the mean",
                    "It uses the systematic risk (beta)",
                    "It uses only returns below a target in the risk measure (downside deviation)"
                ],
                "correctExplanation": "The Sortino ratio divides by the downside deviation $\\sigma_D = \\sqrt{\\frac1n \\sum \\min(R_t - \\tau, 0)^2}$.",
                "incorrectExplanation": "Only the returns below the target enter the denominator of the Sortino ratio."
            },
            "ro": {
                "title": "Sortino ratio",
                "text": "Prin ce diferă Sortino ratio de Sharpe ratio?",
                "options": [
                    "Folosește maximum drawdown la numitor",
                    "Folosește mediana în locul mediei",
                    "Folosește riscul sistematic (beta)",
                    "Folosește în măsura riscului doar randamentele sub un prag (abaterea negativă)"
                ],
                "correctExplanation": "Sortino ratio împarte la abaterea negativă $\\sigma_D = \\sqrt{\\frac1n \\sum \\min(R_t - \\tau, 0)^2}$.",
                "incorrectExplanation": "La numitorul Sortino ratio intră doar randamentele de sub prag."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Maximum drawdown",
                "text": "What is the maximum drawdown (MDD)?",
                "options": [
                    "The largest fall from a previous peak to a later trough",
                    "The worst daily return",
                    "The largest annual loss",
                    "The difference between the highest and lowest price"
                ],
                "correctExplanation": "$\\text{MDD} = \\min_t \\big(P_t/\\max_{s \\le t} P_s - 1\\big)$: the worst peak-to-trough fall.",
                "incorrectExplanation": "The MDD is measured from a previous peak to a later trough; it is usually much larger than the worst single day."
            },
            "ro": {
                "title": "Maximum drawdown",
                "text": "Ce este maximum drawdown (MDD)?",
                "options": [
                    "Cea mai mare cădere de la un vîrf anterior la un minim ulterior",
                    "Cel mai prost randament zilnic",
                    "Cea mai mare pierdere anuală",
                    "Diferența dintre cel mai mare și cel mai mic preț"
                ],
                "correctExplanation": "$\\text{MDD} = \\min_t \\big(P_t/\\max_{s \\le t} P_s - 1\\big)$: cea mai mare cădere de la vîrf la minim.",
                "incorrectExplanation": "MDD se măsoară de la un vîrf anterior la un minim ulterior; de obicei este mult mai mare decît cea mai proastă zi."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Calmar ratio",
                "text": "How is the Calmar ratio defined?",
                "options": [
                    "Mean return divided by volatility",
                    "CAGR divided by the absolute maximum drawdown",
                    "Maximum drawdown divided by volatility",
                    "CAGR divided by downside deviation"
                ],
                "correctExplanation": "Calmar $= \\text{CAGR}/|\\text{MDD}|$: growth per unit of the worst fall.",
                "incorrectExplanation": "The Calmar ratio puts the maximum drawdown in the denominator: CAGR$/|$MDD$|$."
            },
            "ro": {
                "title": "Calmar ratio",
                "text": "Cum se definește Calmar ratio?",
                "options": [
                    "Randamentul mediu împărțit la volatilitate",
                    "CAGR împărțit la valoarea absolută a maximum drawdown",
                    "Maximum drawdown împărțit la volatilitate",
                    "CAGR împărțit la abaterea negativă"
                ],
                "correctExplanation": "Calmar $= \\text{CAGR}/|\\text{MDD}|$: creșterea pe unitatea celei mai mari căderi.",
                "incorrectExplanation": "Calmar ratio are la numitor maximum drawdown: CAGR$/|$MDD$|$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Volatility drag",
                "text": "Two assets have the same arithmetic mean return, 10% a year, and volatilities of 5% and 30%. Which grows faster over many years?",
                "options": [
                    "The 30% volatility asset",
                    "They grow at the same rate",
                    "The 5% volatility asset",
                    "It cannot be determined"
                ],
                "correctExplanation": "Growth $\\approx$ arithmetic mean $- \\sigma^2/2$: about 9.9% vs 5.5% a year.",
                "incorrectExplanation": "The volatility drag $\\sigma^2/2$ is 0.125% for the first asset and 4.5% for the second, so the low-volatility asset grows faster."
            },
            "ro": {
                "title": "Volatility drag",
                "text": "Două active au aceeași medie aritmetică, 10% pe an, și volatilități de 5% și 30%. Care crește mai repede pe mulți ani?",
                "options": [
                    "Activul cu volatilitate de 30%",
                    "Cresc în același ritm",
                    "Activul cu volatilitate de 5%",
                    "Nu se poate stabili"
                ],
                "correctExplanation": "Creșterea $\\approx$ media aritmetică $- \\sigma^2/2$: aproximativ 9,9% vs 5,5% pe an.",
                "incorrectExplanation": "Volatility drag $\\sigma^2/2$ este 0,125% pentru primul activ și 4,5% pentru al doilea, deci activul cu volatilitate mică crește mai repede."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Prices vs returns",
                "text": "Why are statistical models usually fitted to returns rather than to prices?",
                "options": [
                    "Returns are always normally distributed",
                    "Prices are not observed daily",
                    "Returns have no outliers",
                    "Prices trend and wander (unit root), while returns fluctuate around a stable level"
                ],
                "correctExplanation": "Log prices behave like a random walk (non-stationary); returns are approximately stationary, which most methods require.",
                "incorrectExplanation": "The ADF test does not reject a unit root for log prices but rejects it for returns: returns are the stationary object."
            },
            "ro": {
                "title": "Prețuri vs randamente",
                "text": "De ce se estimează modelele statistice de obicei pe randamente, nu pe prețuri?",
                "options": [
                    "Randamentele sînt întotdeauna normal distribuite",
                    "Prețurile nu se observă zilnic",
                    "Randamentele nu au valori extreme",
                    "Prețurile au trend și rătăcesc (rădăcină unitară), iar randamentele oscilează în jurul unui nivel stabil"
                ],
                "correctExplanation": "Logaritmul prețului se comportă ca un mers aleator (nestaționar); randamentele sînt aproximativ staționare, cum cer majoritatea metodelor.",
                "incorrectExplanation": "Testul ADF nu respinge rădăcina unitară pentru logaritmul prețului, dar o respinge pentru randamente: randamentele sînt obiectul staționar."
            }
        }
    ]
};
