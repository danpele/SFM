// ============================================================
// Chapter 0 quiz bank: Introduction (EN + RO)
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['intro'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Course focus",
                "text": "What is the main focus of Statistics of Financial Markets?",
                "options": [
                    "Accounting standards and auditing",
                    "Macroeconomic policy analysis",
                    "Statistical modelling and analysis of financial data",
                    "Corporate governance frameworks"
                ],
                "correctExplanation": "The course applies statistical methods to model and analyse financial market data.",
                "incorrectExplanation": "The course is about statistical modelling and analysis of financial data."
            },
            "ro": {
                "title": "Obiectul cursului",
                "text": "Care este obiectul principal al cursului Statistica piețelor financiare?",
                "options": [
                    "Standardele contabile și auditul",
                    "Analiza politicilor macroeconomice",
                    "Modelarea și analiza statistică a datelor financiare",
                    "Cadrele de guvernanță corporativă"
                ],
                "correctExplanation": "Cursul aplică metode statistice pentru modelarea și analiza datelor de pe piețele financiare.",
                "incorrectExplanation": "Cursul se ocupă de modelarea și analiza statistică a datelor financiare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Financial data",
                "text": "Which type of data is analysed most often in this course?",
                "options": [
                    "Survey and questionnaire data",
                    "Census data",
                    "Price and return series, often at high frequency",
                    "Experimental laboratory data"
                ],
                "correctExplanation": "The course works mainly with financial time series: prices, returns and volatility.",
                "incorrectExplanation": "Financial markets generate price and return series, often at high frequency, and these are the data of the course."
            },
            "ro": {
                "title": "Date financiare",
                "text": "Ce tip de date se analizează cel mai des în acest curs?",
                "options": [
                    "Date din sondaje și chestionare",
                    "Date de recensămînt",
                    "Serii de prețuri și randamente, adesea de înaltă frecvență",
                    "Date experimentale de laborator"
                ],
                "correctExplanation": "Cursul lucrează în principal cu serii de timp financiare: prețuri, randamente și volatilitate.",
                "incorrectExplanation": "Piețele financiare generează serii de prețuri și randamente, adesea de înaltă frecvență, iar acestea sînt datele cursului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Financial versus classical statistics",
                "text": "What sets the statistics of financial markets apart from classical statistics?",
                "options": [
                    "It uses only parametric methods",
                    "Financial data show heavy tails, volatility clustering and temporal dependence",
                    "Classical statistics cannot handle large data sets",
                    "It does not need probability theory"
                ],
                "correctExplanation": "Heavy tails, volatility clustering and serial dependence call for specific methods.",
                "incorrectExplanation": "Financial data show heavy tails, volatility clustering and temporal dependence, which the i.i.d. Normal toolbox does not handle."
            },
            "ro": {
                "title": "Statistica financiară și statistica clasică",
                "text": "Ce deosebește statistica piețelor financiare de statistica clasică?",
                "options": [
                    "Folosește doar metode parametrice",
                    "Datele financiare au cozi groase, grupări de volatilitate și dependență temporală",
                    "Statistica clasică nu poate lucra cu volume mari de date",
                    "Nu are nevoie de teoria probabilităților"
                ],
                "correctExplanation": "Cozile groase, grupările de volatilitate și dependența serială cer metode specifice.",
                "incorrectExplanation": "Datele financiare au cozi groase, grupări de volatilitate și dependență temporală, pe care instrumentele clasice pentru date i.i.d. cu distribuția Normală nu le surprind."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Software",
                "text": "Which software is used in this course for data analysis?",
                "options": [
                    "Microsoft Excel",
                    "SPSS",
                    "Python",
                    "Stata"
                ],
                "correctExplanation": "Python is the language used for the analyses, notebooks and Quantlets of the course.",
                "incorrectExplanation": "Python is the main tool of the course."
            },
            "ro": {
                "title": "Software",
                "text": "Ce software se folosește în acest curs pentru analiza datelor?",
                "options": [
                    "Microsoft Excel",
                    "SPSS",
                    "Python",
                    "Stata"
                ],
                "correctExplanation": "Python este limbajul folosit pentru analizele, notebook-urile și Quantlet-urile cursului.",
                "incorrectExplanation": "Python este instrumentul principal al cursului."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Assessment",
                "text": "How is the course assessed?",
                "options": [
                    "100% final exam",
                    "50% exam, 50% project",
                    "70% exam, 20% project, 10% attendance",
                    "40% exam, 40% project, 20% homework"
                ],
                "correctExplanation": "The final grade is 70% written exam, 20% team project and 10% attendance.",
                "incorrectExplanation": "The assessment is 70% exam, 20% project and 10% attendance."
            },
            "ro": {
                "title": "Evaluare",
                "text": "Cum se face evaluarea la acest curs?",
                "options": [
                    "100% examen final",
                    "50% examen, 50% proiect",
                    "70% examen, 20% proiect, 10% prezență",
                    "40% examen, 40% proiect, 20% teme"
                ],
                "correctExplanation": "Nota finală se compune din 70% examen scris, 20% proiect în echipă și 10% prezență.",
                "incorrectExplanation": "Evaluarea este 70% examen, 20% proiect și 10% prezență."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Stylised facts",
                "text": "Which of the following is a stylised fact of financial returns?",
                "options": [
                    "Returns are always Normally distributed",
                    "Volatility is constant over time",
                    "Returns show heavy tails and volatility clustering",
                    "Financial markets are always efficient"
                ],
                "correctExplanation": "Heavy tails and volatility clustering are well-documented stylised facts of returns.",
                "incorrectExplanation": "Heavy tails and volatility clustering are key stylised facts."
            },
            "ro": {
                "title": "Fapte stilizate",
                "text": "Care dintre următoarele este un fapt stilizat al randamentelor financiare?",
                "options": [
                    "Randamentele urmează mereu distribuția Normală",
                    "Volatilitatea este constantă în timp",
                    "Randamentele au cozi groase și grupări de volatilitate",
                    "Piețele financiare sînt mereu eficiente"
                ],
                "correctExplanation": "Cozile groase și grupările de volatilitate sînt fapte stilizate bine documentate ale randamentelor.",
                "incorrectExplanation": "Cozile groase și grupările de volatilitate sînt fapte stilizate centrale."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Course topics",
                "text": "Which of the following is NOT a topic of this course?",
                "options": [
                    "GARCH models",
                    "Value-at-Risk",
                    "Double-entry bookkeeping",
                    "Stable distributions"
                ],
                "correctExplanation": "Double-entry bookkeeping belongs to accounting, not to the statistics of financial markets.",
                "incorrectExplanation": "Double-entry bookkeeping is not part of the syllabus."
            },
            "ro": {
                "title": "Tematica cursului",
                "text": "Care dintre următoarele NU este o temă a cursului?",
                "options": [
                    "Modelele GARCH",
                    "Value-at-Risk",
                    "Contabilitatea în partidă dublă",
                    "Distribuțiile stabile"
                ],
                "correctExplanation": "Contabilitatea în partidă dublă ține de contabilitate, nu de statistica piețelor financiare.",
                "incorrectExplanation": "Contabilitatea în partidă dublă nu face parte din programa cursului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Quantlets",
                "text": "What is a Quantlet in this course?",
                "options": [
                    "A multiple-choice test",
                    "A reusable, documented piece of code that reproduces a statistical analysis or chart",
                    "A type of financial derivative",
                    "A lecture summary"
                ],
                "correctExplanation": "A Quantlet is reusable, documented code (with a Metainfo file) for a specific statistical or computational procedure.",
                "incorrectExplanation": "A Quantlet is a reusable, documented piece of code for a statistical analysis."
            },
            "ro": {
                "title": "Quantlets",
                "text": "Ce este un Quantlet în acest curs?",
                "options": [
                    "Un test grilă",
                    "Un fragment de cod reutilizabil și documentat, care reproduce o analiză statistică sau un grafic",
                    "Un tip de instrument financiar derivat",
                    "Un rezumat al cursului"
                ],
                "correctExplanation": "Un Quantlet este cod reutilizabil și documentat (cu fișier Metainfo) pentru o procedură statistică sau de calcul.",
                "incorrectExplanation": "Un Quantlet este un fragment de cod reutilizabil și documentat pentru o analiză statistică."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Heavy tails",
                "text": "What does \"heavy tails\" mean for financial returns?",
                "options": [
                    "The distribution is skewed to the left",
                    "Extreme returns occur more often than the Normal distribution predicts",
                    "The mean return is very large",
                    "The series has a deterministic trend"
                ],
                "correctExplanation": "Large gains and losses occur more often than under the Normal distribution with the same variance.",
                "incorrectExplanation": "Heavy tails mean more frequent extreme returns than the Normal distribution predicts."
            },
            "ro": {
                "title": "Cozi groase",
                "text": "Ce înseamnă „cozi groase” pentru randamentele financiare?",
                "options": [
                    "Distribuția este asimetrică la stînga",
                    "Randamentele extreme apar mai des decît prevede distribuția Normală",
                    "Randamentul mediu este foarte mare",
                    "Seria are un trend determinist"
                ],
                "correctExplanation": "Cîștigurile și pierderile mari apar mai des decît sub distribuția Normală cu aceeași varianță.",
                "incorrectExplanation": "Cozile groase înseamnă randamente extreme mai frecvente decît prevede distribuția Normală."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Field",
                "text": "Which field combines statistics, finance and programming as this course does?",
                "options": [
                    "Pure mathematics",
                    "Quantitative finance / financial econometrics",
                    "Behavioural psychology",
                    "Business administration"
                ],
                "correctExplanation": "Quantitative finance and financial econometrics combine statistical methods, financial theory and computation.",
                "incorrectExplanation": "Quantitative finance / financial econometrics is the field that brings these together."
            },
            "ro": {
                "title": "Domeniul",
                "text": "Ce domeniu combină statistica, finanțele și programarea, ca acest curs?",
                "options": [
                    "Matematica pură",
                    "Finanțele cantitative / econometria financiară",
                    "Psihologia comportamentală",
                    "Administrarea afacerilor"
                ],
                "correctExplanation": "Finanțele cantitative și econometria financiară combină metode statistice, teorie financiară și calcul.",
                "incorrectExplanation": "Finanțele cantitative / econometria financiară reunesc aceste discipline."
            }
        }
    ]
};
