// ============================================================
// Chapter 0 quiz bank: Introduction (EN + RO), 24 questions, 20 drawn
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['intro'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Evaluation",
                "text": "How is the final grade of the course made up?",
                "options": [
                    "50% exam, 50% project",
                    "70% written exam, 20% team project, 10% attendance",
                    "60% exam, 40% project",
                    "100% written exam"
                ],
                "correctExplanation": "The grade is 70% written exam, 20% team project and 10% attendance.",
                "incorrectExplanation": "The grade combines a written exam (70%), a team project (20%) and attendance (10%)."
            },
            "ro": {
                "title": "Evaluare",
                "text": "Cum se formează nota finală a cursului?",
                "options": [
                    "50% examen, 50% proiect",
                    "70% examen scris, 20% proiect de echipă, 10% prezență",
                    "60% examen, 40% proiect",
                    "100% examen scris"
                ],
                "correctExplanation": "Nota finală se compune din examenul scris (70%), proiectul de echipă (20%) și prezență (10%).",
                "incorrectExplanation": "Nota finală combină examenul scris (70%), proiectul de echipă (20%) și prezența (10%)."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "AI policy",
                "text": "What does the course allow regarding AI tools in the team project?",
                "options": [
                    "AI tools are forbidden",
                    "AI tools are allowed without any declaration",
                    "Only AI-generated references may be used without checks",
                    "AI tools are allowed, every use is declared in AI_USE.md, and the team defends the work orally"
                ],
                "correctExplanation": "AI is allowed and declared; the oral defence checks that each member understands the code and results.",
                "incorrectExplanation": "AI tools are allowed but must be declared in AI_USE.md; every number and reference is checked, and the oral defence tests understanding."
            },
            "ro": {
                "title": "Politica privind AI",
                "text": "Ce permite cursul în privința instrumentelor AI în proiectul de echipă?",
                "options": [
                    "Instrumentele AI sînt interzise",
                    "Instrumentele AI sînt permise fără nicio declarație",
                    "Doar referințele generate de AI pot fi folosite fără verificare",
                    "Instrumentele AI sînt permise, fiecare utilizare se declară în AI_USE.md, iar echipa își susține oral proiectul"
                ],
                "correctExplanation": "Utilizarea AI este permisă și se declară; susținerea orală verifică dacă fiecare membru înțelege codul și rezultatele.",
                "incorrectExplanation": "Instrumentele AI sînt permise, dar se declară în AI_USE.md; fiecare rezultat numeric și fiecare referință se verifică, iar susținerea orală verifică înțelegerea."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Seminars",
                "text": "Which statement about the seminars is correct?",
                "options": [
                    "Each seminar comes before its lecture and nothing is handed in",
                    "Seminar homework is graded every week",
                    "Seminars repeat the lecture after it has been given",
                    "Proposed exercises must be submitted on GitHub"
                ],
                "correctExplanation": "Seminars precede the lectures, start with a primer, and homework is practice only.",
                "incorrectExplanation": "Each seminar comes before its lecture, with a short primer; nothing is handed in, homework is practice."
            },
            "ro": {
                "title": "Seminarii",
                "text": "Care afirmație despre seminarii este corectă?",
                "options": [
                    "Fiecare seminar are loc înaintea cursului corespunzător, iar studenții nu au nimic de trimis",
                    "Temele de seminar se notează în fiecare săptămînă",
                    "Seminariile repetă cursul după ce a fost predat",
                    "Exercițiile propuse trebuie încărcate pe GitHub"
                ],
                "correctExplanation": "Seminariile preced cursurile și încep cu o scurtă introducere; temele servesc doar ca exercițiu.",
                "incorrectExplanation": "Fiecare seminar are loc înaintea cursului corespunzător și începe cu o scurtă introducere; studenții nu trimit nimic, iar temele servesc doar ca exercițiu."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Textbook",
                "text": "Which is the main textbook of the course?",
                "options": [
                    "Tsay, Analysis of Financial Time Series",
                    "Campbell, Lo and MacKinlay, The Econometrics of Financial Markets",
                    "Franke, Härdle and Hafner, Statistics of Financial Markets, 5th edition",
                    "McNeil, Frey and Embrechts, Quantitative Risk Management"
                ],
                "correctExplanation": "The main textbook is Franke, Härdle and Hafner (2019); the seminars use the companion Exercises and Solutions.",
                "incorrectExplanation": "The course follows Franke, Härdle and Hafner, Statistics of Financial Markets (5th ed., Springer, 2019)."
            },
            "ro": {
                "title": "Manual",
                "text": "Care este manualul de bază al cursului?",
                "options": [
                    "Tsay, Analysis of Financial Time Series",
                    "Campbell, Lo și MacKinlay, The Econometrics of Financial Markets",
                    "Franke, Härdle și Hafner, Statistics of Financial Markets, ediția a 5-a",
                    "McNeil, Frey și Embrechts, Quantitative Risk Management"
                ],
                "correctExplanation": "Manualul de bază este Franke, Härdle și Hafner (2019); seminariile folosesc volumul însoțitor Exercises and Solutions.",
                "incorrectExplanation": "Cursul urmează Franke, Härdle și Hafner, Statistics of Financial Markets (ediția a 5-a, Springer, 2019)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Price discovery",
                "text": "What does the function called price discovery mean?",
                "options": [
                    "Exchanges publish a list of fixed prices each morning",
                    "Prices gather information spread across many traders",
                    "Governments set the price of shares",
                    "Brokers find the cheapest share to buy"
                ],
                "correctExplanation": "Through trading, prices aggregate the information dispersed among market participants.",
                "incorrectExplanation": "Price discovery means that prices gather information spread across many traders."
            },
            "ro": {
                "title": "Descoperirea prețului",
                "text": "Ce înseamnă funcția de descoperire a prețului (price discovery)?",
                "options": [
                    "Bursele publică în fiecare dimineață o listă de prețuri fixe",
                    "Prețurile agregă informația dispersată între mulți participanți la piață",
                    "Statul stabilește prețul acțiunilor",
                    "Brokerii găsesc cea mai ieftină acțiune"
                ],
                "correctExplanation": "Prin tranzacționare, prețurile agregă informația dispersată între participanții la piață.",
                "incorrectExplanation": "Descoperirea prețului înseamnă că prețurile agregă informația dispersată între mulți participanți la piață."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Bid-ask spread",
                "text": "The best bid is 99.99 and the best ask is 100.01. What is the bid-ask spread?",
                "options": [
                    "0.01",
                    "100.00",
                    "0.02",
                    "199.99"
                ],
                "correctExplanation": "The spread is the best ask minus the best bid: 100.01 - 99.99 = 0.02, the cost of trading at once.",
                "incorrectExplanation": "The spread is best ask minus best bid, here 0.02."
            },
            "ro": {
                "title": "Spread-ul bid-ask",
                "text": "Cel mai bun bid este 99,99, iar cel mai bun ask este 100,01. Cît este spread-ul bid-ask?",
                "options": [
                    "0,01",
                    "100,00",
                    "0,02",
                    "199,99"
                ],
                "correctExplanation": "Spread-ul este cel mai bun ask minus cel mai bun bid: 100,01 - 99,99 = 0,02, costul unei tranzacții imediate.",
                "incorrectExplanation": "Spread-ul este cel mai bun ask minus cel mai bun bid, aici 0,02."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Market order",
                "text": "Asks: 2 units at 100.01 and 4 units at 100.02. A market order buys 5 units. What is its average price?",
                "options": [
                    "100.016",
                    "100.010",
                    "100.020",
                    "100.015"
                ],
                "correctExplanation": "2 units at 100.01 and 3 at 100.02: (2 × 100.01 + 3 × 100.02) / 5 = 100.016.",
                "incorrectExplanation": "The order consumes the book: 2 units at 100.01, then 3 at 100.02, an average of 100.016."
            },
            "ro": {
                "title": "Ordin la piață",
                "text": "Oferte de vînzare: 2 unități la 100,01 și 4 unități la 100,02. Un ordin la piață cumpără 5 unități. Care este prețul mediu?",
                "options": [
                    "100,016",
                    "100,010",
                    "100,020",
                    "100,015"
                ],
                "correctExplanation": "2 unități la 100,01 și 3 la 100,02: (2 × 100,01 + 3 × 100,02) / 5 = 100,016.",
                "incorrectExplanation": "Ordinul parcurge registrul de ordine: 2 unități la 100,01, apoi 3 la 100,02, deci un preț mediu de 100,016."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ETF",
                "text": "What is an ETF (exchange-traded fund)?",
                "options": [
                    "A bank deposit with a fixed rate",
                    "A government bond",
                    "A futures contract on an index",
                    "A fund whose shares trade on an exchange like a stock"
                ],
                "correctExplanation": "An ETF is a fund traded on an exchange like a share; passive ETFs copy an index.",
                "incorrectExplanation": "An ETF is a fund whose shares are traded on an exchange like a stock."
            },
            "ro": {
                "title": "ETF",
                "text": "Ce este un ETF (exchange-traded fund)?",
                "options": [
                    "Un depozit bancar cu dobîndă fixă",
                    "Un titlu de stat",
                    "Un contract futures pe un indice",
                    "Un fond ale cărui unități se tranzacționează la bursă ca o acțiune"
                ],
                "correctExplanation": "Un ETF este un fond tranzacționat la bursă ca o acțiune; ETF-urile pasive copiază un indice.",
                "incorrectExplanation": "Un ETF este un fond ale cărui unități se tranzacționează la bursă ca o acțiune."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Simple return",
                "text": "A price moves from 100 to 110. What is the simple return?",
                "options": [
                    "9.53%",
                    "10%",
                    "11%",
                    "1.10%"
                ],
                "correctExplanation": "R = 110/100 - 1 = 10%; the log return is ln(1.1) = 9.53%.",
                "incorrectExplanation": "The simple return is P1/P0 - 1 = 10%."
            },
            "ro": {
                "title": "Randament simplu",
                "text": "Un preț crește de la 100 la 110. Cît este randamentul simplu?",
                "options": [
                    "9,53%",
                    "10%",
                    "11%",
                    "1,10%"
                ],
                "correctExplanation": "R = 110/100 - 1 = 10%; randamentul logaritmic este ln(1,1) = 9,53%.",
                "incorrectExplanation": "Randamentul simplu este P1/P0 - 1 = 10%."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Adding returns",
                "text": "Which returns can be added over time to get the multi-period return?",
                "options": [
                    "Simple returns",
                    "Neither",
                    "Log returns",
                    "Both"
                ],
                "correctExplanation": "Log returns telescope: the sum of daily log returns equals ln(PT/P0). Simple returns compound.",
                "incorrectExplanation": "Log returns add up over time; simple returns must be compounded."
            },
            "ro": {
                "title": "Adunarea randamentelor",
                "text": "Ce randamente se pot aduna în timp pentru a obține randamentul pe mai multe perioade?",
                "options": [
                    "Randamentele simple",
                    "Niciunele",
                    "Randamentele logaritmice",
                    "Ambele"
                ],
                "correctExplanation": "Suma randamentelor logaritmice zilnice se reduce telescopic la ln(PT/P0). Randamentele simple se compun.",
                "incorrectExplanation": "Randamentele logaritmice se adună în timp; randamentele simple trebuie compuse."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Two-day return",
                "text": "A price goes 100 → 110 → 99. What is the two-day simple return?",
                "options": [
                    "-1%",
                    "0%",
                    "+1%",
                    "-10%"
                ],
                "correctExplanation": "99/100 - 1 = -1%, although the two simple returns (+10% and -10%) add up to 0%.",
                "incorrectExplanation": "Compound the simple returns: 1.1 × 0.9 - 1 = -1%."
            },
            "ro": {
                "title": "Randamentul pe două zile",
                "text": "Prețul evoluează astfel: 100 → 110 → 99. Cît este randamentul simplu pe două zile?",
                "options": [
                    "-1%",
                    "0%",
                    "+1%",
                    "-10%"
                ],
                "correctExplanation": "99/100 - 1 = -1%, deși cele două randamente simple (+10% și -10%) au suma 0%.",
                "incorrectExplanation": "Compunem randamentele simple: 1,1 × 0,9 - 1 = -1%."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Annualising volatility",
                "text": "A stock index has a daily volatility of 1%. What is its annualised volatility (252 days)?",
                "options": [
                    "252%",
                    "2.52%",
                    "1%",
                    "about 15.9%"
                ],
                "correctExplanation": "Volatility scales with the square root of time: √252 × 1% ≈ 15.9%.",
                "incorrectExplanation": "Multiply the daily volatility by √252 ≈ 15.87, not by 252."
            },
            "ro": {
                "title": "Anualizarea volatilității",
                "text": "Un indice bursier are volatilitatea zilnică de 1%. Cît este volatilitatea anualizată (252 de zile)?",
                "options": [
                    "252%",
                    "2,52%",
                    "1%",
                    "aproximativ 15,9%"
                ],
                "correctExplanation": "Volatilitatea crește cu rădăcina pătrată a timpului: √252 × 1% ≈ 15,9%.",
                "incorrectExplanation": "Volatilitatea zilnică se înmulțește cu √252 ≈ 15,87, nu cu 252."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Bitcoin calendar",
                "text": "How many observations per year should be used to annualise Bitcoin returns?",
                "options": [
                    "252",
                    "365",
                    "52",
                    "12"
                ],
                "correctExplanation": "Bitcoin trades every day, so A = 365 observations a year; stock exchanges have about 252 trading days.",
                "incorrectExplanation": "Use the actual frequency of the series: Bitcoin trades 365 days a year."
            },
            "ro": {
                "title": "Calendarul Bitcoin",
                "text": "Cîte observații pe an se folosesc pentru anualizarea randamentelor Bitcoin?",
                "options": [
                    "252",
                    "365",
                    "52",
                    "12"
                ],
                "correctExplanation": "Bitcoin se tranzacționează în fiecare zi, deci A = 365 de observații pe an; bursele au aproximativ 252 de zile de tranzacționare.",
                "incorrectExplanation": "Folosiți frecvența reală a seriei: Bitcoin se tranzacționează 365 de zile pe an."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Maximum drawdown",
                "text": "Prices: 100, 120, 90, 110, 130, 104. What is the maximum drawdown?",
                "options": [
                    "-20%",
                    "-10%",
                    "-25%",
                    "-30%"
                ],
                "correctExplanation": "The worst fall from a previous peak is from 120 to 90: 90/120 - 1 = -25%.",
                "incorrectExplanation": "Drawdown is measured from the running peak; the deepest one is 90/120 - 1 = -25%."
            },
            "ro": {
                "title": "Drawdown maxim",
                "text": "Prețuri: 100, 120, 90, 110, 130, 104. Cît este drawdown-ul maxim?",
                "options": [
                    "-20%",
                    "-10%",
                    "-25%",
                    "-30%"
                ],
                "correctExplanation": "Cea mai mare scădere față de un vîrf anterior este de la 120 la 90: 90/120 - 1 = -25%.",
                "incorrectExplanation": "Drawdown-ul se măsoară față de maximul atins pînă la acel moment; cel mai adînc este 90/120 - 1 = -25%."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "BET-TR",
                "text": "How does the BET-TR index differ from the BET index?",
                "options": [
                    "BET-TR includes the dividends, reinvested",
                    "BET-TR contains only banks",
                    "BET-TR is quoted in euro",
                    "BET-TR is computed weekly"
                ],
                "correctExplanation": "BET-TR (total return) reinvests dividends, which matter a lot for Romanian firms.",
                "incorrectExplanation": "BET-TR is the total-return version of BET: dividends are reinvested."
            },
            "ro": {
                "title": "BET-TR",
                "text": "Prin ce diferă indicele BET-TR de indicele BET?",
                "options": [
                    "BET-TR include dividendele reinvestite",
                    "BET-TR conține doar bănci",
                    "BET-TR este cotat în euro",
                    "BET-TR se calculează săptămînal"
                ],
                "correctExplanation": "BET-TR (total return) reinvestește dividendele, care au o pondere mare în randamentul acțiunilor românești.",
                "incorrectExplanation": "BET-TR este varianta de randament total (total return) a indicelui BET: dividendele sînt reinvestite."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "VIX",
                "text": "What does the VIX index measure?",
                "options": [
                    "The number of trades on the NYSE",
                    "The past return of the S&P 500",
                    "The interest rate of the Federal Reserve",
                    "The volatility of the S&P 500 over the next 30 days expected by option traders"
                ],
                "correctExplanation": "The VIX is the expected 30-day volatility of the S&P 500 implied by option prices, in % a year.",
                "incorrectExplanation": "The VIX is an implied volatility, computed from S&P 500 option prices."
            },
            "ro": {
                "title": "VIX",
                "text": "Ce măsoară indicele VIX?",
                "options": [
                    "Numărul de tranzacții de la NYSE",
                    "Randamentul trecut al S&P 500",
                    "Dobînda Rezervei Federale",
                    "Volatilitatea S&P 500 în următoarele 30 de zile, așteptată de cei care tranzacționează opțiuni"
                ],
                "correctExplanation": "VIX este volatilitatea așteptată pe 30 de zile a S&P 500, implicită în prețurile opțiunilor, în % pe an.",
                "incorrectExplanation": "VIX este o volatilitate implicită, calculată din prețurile opțiunilor pe S&P 500."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "BET after 2008",
                "text": "What were the maximum drawdown of the BET index in 2008-2009 and the year in which it regained its 2007 peak?",
                "options": [
                    "About -30%, in 2010",
                    "About -82.5%, only in 2021",
                    "About -50%, in 2013",
                    "It never fell below -20%"
                ],
                "correctExplanation": "BET fell by 82.5% until February 2009 and regained its July 2007 peak only in March 2021.",
                "incorrectExplanation": "The BET drawdown reached -82.5% and lasted more than 13 years."
            },
            "ro": {
                "title": "BET după 2008",
                "text": "Care au fost drawdown-ul maxim al indicelui BET în 2008–2009 și anul în care indicele și-a recuperat vîrful din 2007?",
                "options": [
                    "Aproximativ -30%, în 2010",
                    "Aproximativ -82,5%, abia în 2021",
                    "Aproximativ -50%, în 2013",
                    "Nu a scăzut niciodată sub -20%"
                ],
                "correctExplanation": "BET a scăzut cu 82,5% pînă în februarie 2009 și și-a recuperat vîrful din iulie 2007 abia în martie 2021.",
                "incorrectExplanation": "Drawdown-ul BET a ajuns la -82,5% și a durat peste 13 ani."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Heavy tails",
                "text": "The kurtosis of S&P 500 daily returns since 2000 is about 13.7. What does this show?",
                "options": [
                    "Returns follow the Normal distribution",
                    "Returns never exceed 1%",
                    "Tails are much heavier than in the Normal distribution, whose kurtosis is 3",
                    "Volatility is constant"
                ],
                "correctExplanation": "The Normal distribution has kurtosis 3; a much larger value means many more extreme days.",
                "incorrectExplanation": "Kurtosis 13.7 against 3 for the Normal distribution: heavy tails."
            },
            "ro": {
                "title": "Cozi groase",
                "text": "Coeficientul de boltire (kurtosis) al randamentelor zilnice ale S&P 500 din 2000 încoace este aproximativ 13,7. Ce arată această valoare?",
                "options": [
                    "Randamentele urmează distribuția Normală",
                    "Randamentele nu depășesc niciodată 1%",
                    "Cozile sînt mult mai groase decît în distribuția Normală, al cărei coeficient de boltire este 3",
                    "Volatilitatea este constantă"
                ],
                "correctExplanation": "Distribuția Normală are coeficientul de boltire 3; o valoare mult mai mare înseamnă mult mai multe zile extreme.",
                "incorrectExplanation": "Un coeficient de boltire de 13,7, față de 3 pentru distribuția Normală, indică cozi groase."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Bachelier",
                "text": "What did Louis Bachelier propose in 1900?",
                "options": [
                    "A random-walk model for prices on the Paris exchange",
                    "The first stock index",
                    "The GARCH model",
                    "The efficient market hypothesis"
                ],
                "correctExplanation": "In his thesis Théorie de la spéculation, Bachelier modelled prices as a random walk.",
                "incorrectExplanation": "Bachelier (1900) introduced the random walk as a model of price changes."
            },
            "ro": {
                "title": "Bachelier",
                "text": "Ce a propus Louis Bachelier în 1900?",
                "options": [
                    "Un model de mers aleator pentru prețurile de la bursa din Paris",
                    "Primul indice bursier",
                    "Modelul GARCH",
                    "Ipoteza pieței eficiente"
                ],
                "correctExplanation": "În teza Théorie de la spéculation, Bachelier a modelat prețurile ca un mers aleator.",
                "incorrectExplanation": "Bachelier (1900) a introdus mersul aleator ca model al variațiilor de preț."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Mandelbrot",
                "text": "What did Benoit Mandelbrot show in 1963 about cotton prices?",
                "options": [
                    "Their changes are independent and Normal",
                    "They follow a deterministic cycle",
                    "They are perfectly predictable",
                    "Their changes have far heavier tails than the Normal distribution"
                ],
                "correctExplanation": "Mandelbrot (1963) found heavy tails and proposed α-stable distributions.",
                "incorrectExplanation": "Mandelbrot showed that price changes have heavy tails, contrary to the Normal model."
            },
            "ro": {
                "title": "Mandelbrot",
                "text": "Ce a arătat Benoit Mandelbrot în 1963 despre prețurile bumbacului?",
                "options": [
                    "Variațiile lor sînt independente și urmează distribuția Normală",
                    "Urmează un ciclu determinist",
                    "Sînt perfect previzibile",
                    "Variațiile lor au cozi mult mai groase decît distribuția Normală"
                ],
                "correctExplanation": "Mandelbrot (1963) a găsit cozi groase și a propus distribuțiile α-stabile.",
                "incorrectExplanation": "Mandelbrot a arătat că variațiile prețurilor au cozi groase, contrar ipotezei distribuției Normale."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Bucharest Stock Exchange",
                "text": "In which years did the Bucharest exchange first open and the BVB re-open?",
                "options": [
                    "1900 and 1990",
                    "1882 and 1995",
                    "1918 and 2007",
                    "1948 and 1997"
                ],
                "correctExplanation": "The exchange opened on 1 December 1882, closed in 1948 and reopened with its first session on 20 November 1995.",
                "incorrectExplanation": "Bucharest: 1882 to 1948, then again from 1995; the BET index started in 1997."
            },
            "ro": {
                "title": "Bursa de Valori București",
                "text": "În ce ani s-a deschis prima dată bursa din București și a fost reînființată BVB?",
                "options": [
                    "1900 și 1990",
                    "1882 și 1995",
                    "1918 și 2007",
                    "1948 și 1997"
                ],
                "correctExplanation": "Bursa s-a deschis la 1 decembrie 1882, a fost închisă în 1948 și și-a reluat activitatea cu prima ședință la 20 noiembrie 1995.",
                "incorrectExplanation": "Bursa din București a funcționat între 1882 și 1948, apoi din nou din 1995; indicele BET se calculează din 1997."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Tulip mania",
                "text": "What does later research (Garber, 1989; Goldgar, 2007) say about tulip mania?",
                "options": [
                    "It ruined the Dutch economy",
                    "It never happened",
                    "It was a smaller episode than the legend, with few bankruptcies",
                    "It lasted more than ten years"
                ],
                "correctExplanation": "Archival research shows a short episode with limited damage, much smaller than the popular story.",
                "incorrectExplanation": "Later research found a smaller episode than the legend, with few bankruptcies."
            },
            "ro": {
                "title": "Mania lalelelor",
                "text": "Ce spun cercetările ulterioare (Garber, 1989; Goldgar, 2007) despre mania lalelelor?",
                "options": [
                    "A ruinat economia olandeză",
                    "Nu a avut loc niciodată",
                    "A fost un episod de amploare mai mică decît sugerează legenda, cu puține falimente",
                    "A durat peste zece ani"
                ],
                "correctExplanation": "Cercetarea de arhivă arată un episod scurt, cu pagube limitate, de amploare mult mai mică decît în versiunea populară.",
                "incorrectExplanation": "Cercetările ulterioare au găsit un episod de amploare mai mică decît sugerează legenda, cu puține falimente."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "A rival explanation",
                "text": "Bitcoin volatility fell after January 2024. Why compare it with the S&P 500 volatility in the same years?",
                "options": [
                    "To rule out the explanation that all markets were simply calmer",
                    "Because Bitcoin is part of the S&P 500",
                    "Because the S&P 500 trades 365 days a year",
                    "It is not needed: one calm year proves the trend"
                ],
                "correctExplanation": "If all markets were calmer, the S&P 500 volatility fell too; the ratio Bitcoin / S&P 500 removes that common factor.",
                "incorrectExplanation": "Comparing with the S&P 500 checks the rival explanation of a generally calmer market."
            },
            "ro": {
                "title": "O explicație concurentă",
                "text": "Volatilitatea Bitcoin a scăzut după ianuarie 2024. De ce o comparăm cu volatilitatea S&P 500 din aceiași ani?",
                "options": [
                    "Pentru a exclude explicația că toate piețele au fost pur și simplu mai calme",
                    "Pentru că Bitcoin face parte din S&P 500",
                    "Pentru că S&P 500 se tranzacționează 365 de zile pe an",
                    "Nu este necesar: un an calm dovedește tendința"
                ],
                "correctExplanation": "Dacă toate piețele au fost mai calme, a scăzut și volatilitatea S&P 500; raportul Bitcoin / S&P 500 elimină acest factor comun.",
                "incorrectExplanation": "Comparația cu S&P 500 verifică explicația concurentă a unei piețe în general mai calme."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Data check",
                "text": "A EUR/RON quote jumps 15% and returns to its previous level the next day, while the BNR rate barely moves. What is the most likely explanation?",
                "options": [
                    "A currency crisis",
                    "A devaluation by the BNR",
                    "A new trend",
                    "A bad tick: an isolated wrong quote"
                ],
                "correctExplanation": "A jump that fully reverses the next day and is absent from the official rate is a data error.",
                "incorrectExplanation": "Isolated jumps that reverse at once and do not appear in the official rate are bad ticks."
            },
            "ro": {
                "title": "Verificarea datelor",
                "text": "O cotație EUR/RON sare cu 15% și revine a doua zi la nivelul anterior, iar cursul BNR aproape nu se mișcă. Care este explicația cea mai probabilă?",
                "options": [
                    "O criză valutară",
                    "O devalorizare decisă de BNR",
                    "O tendință nouă",
                    "Un bad tick: o cotație greșită izolată"
                ],
                "correctExplanation": "Un salt care se inversează complet a doua zi și lipsește din cursul oficial este o eroare de date.",
                "incorrectExplanation": "Salturile izolate care se inversează imediat și nu apar în cursul oficial sînt bad ticks."
            }
        }
    ]
};
