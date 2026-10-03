# Acronime specifice Capitolului 12 SFM (Modele de scoring / Scoring models)
# format: acronim -> (forma de origine, limba de origine, traducere RO, traducere EN)
#   ex.: 'EVT': ('Extreme Value Theory', 'en', 'teoria valorilor extreme', None)
# OVERRIDE_CH (optional): acronim -> tuplu, sens diferit doar in acest capitol.
EXTRA = {
    'PD': ('Probability of Default', 'en', 'probabilitatea de nerambursare', None),
    'LGD': ('Loss Given Default', 'en', 'pierderea în caz de nerambursare', None),
    'EAD': ('Exposure At Default', 'en', 'expunerea la momentul nerambursării', None),
    'IRB': ('Internal Ratings-Based (approach)', 'en', 'abordarea pe baza ratingurilor interne', None),
    'CAP': ('Cumulative Accuracy Profile', 'en', 'profilul cumulat al acurateței', None),
    'LDA': ('Linear Discriminant Analysis', 'en', 'analiza discriminantă liniară', None),
    'PDO': ('Points to Double the Odds', 'en', 'punctele care dublează șansa bun:rău', None),
    'TPR': ('True Positive Rate', 'en', 'rata adevărat pozitivelor (sensibilitatea)', None),
    'FPR': ('False Positive Rate', 'en', 'rata fals pozitivelor', None),
    'HL': ('Hosmer–Lemeshow (test)', 'en', 'testul Hosmer–Lemeshow', None),
    'TA': ('Total Assets', 'en', 'activele totale', None),
    'EBIT': ('Earnings Before Interest and Taxes', 'en', 'profitul înainte de dobînzi și impozite', None),
    'DD': ('Distance to Default', 'en', 'distanța pînă la nerambursare', None),
    'CDS': ('Credit Default Swap', 'en', 'swap pe riscul de nerambursare', None),
    'FICO': ('Fair Isaac Corporation', 'en', 'compania Fair Isaac (scorul FICO)', None),
    'UCI': ('University of California, Irvine (Machine Learning Repository)', 'en', 'Universitatea din California, Irvine (arhiva de date pentru învățare automată)', None),
    'MVA': ('Applied Multivariate Statistical Analysis (Härdle and Simar)', 'en', 'manualul de analiză statistică multivariată al lui Härdle și Simar', None),
    'NYU': ('New York University', 'en', 'Universitatea din New York', None),
    'FTC': ('Federal Trade Commission', 'en', 'Comisia Federală pentru Comerț a SUA', None),
}
OVERRIDE_CH = {
    'AR': ('Accuracy Ratio (of the CAP curve)', 'en', 'raportul de acuratețe (al curbei CAP)', None),
    'DM': ('Deutsche Mark', 'de', 'marca germană', 'the German mark'),
    'RWA': ('Risk-Weighted Assets', 'en', 'activele ponderate la risc', None),
    'BS': ('Brier Score', 'en', 'scorul Brier', None),
    'KS': ('Kolmogorov–Smirnov (statistic)', 'en', 'statistica Kolmogorov–Smirnov', None),
    'LR': ('Likelihood Ratio (test)', 'en', 'testul raportului de verosimilitate', None),
    'SFE': ('Statistics of Financial Markets, Exercises (Quantlet collection of the textbook)', 'en', 'colecția de Quantlet-uri a manualului Statistics of Financial Markets', None),
}
