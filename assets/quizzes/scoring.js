// ============================================================
// Chapter 12 quiz bank: scoring models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['scoring'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Expected loss",
                "text": "A loan of 50,000 RON has PD = 3% and LGD = 40%. What is the expected loss?",
                "options": [
                    "1,200 RON",
                    "600 RON",
                    "1,500 RON",
                    "20,000 RON"
                ],
                "correctExplanation": "EL = PD x LGD x EAD = 0.03 x 0.40 x 50,000 = 600 RON.",
                "incorrectExplanation": "Multiply all three parameters: the probability of default, the share lost if default occurs, and the exposure."
            },
            "ro": {
                "title": "Pierderea așteptată",
                "text": "Un credit de 50 000 de lei are PD = 3% și LGD = 40%. Cît este pierderea așteptată?",
                "options": [
                    "1 200 de lei",
                    "600 de lei",
                    "1 500 de lei",
                    "20 000 de lei"
                ],
                "correctExplanation": "EL = PD x LGD x EAD = 0,03 x 0,40 x 50 000 = 600 de lei.",
                "incorrectExplanation": "Se înmulțesc toți cei trei parametri: probabilitatea de nerambursare, partea pierdută în caz de nerambursare și expunerea."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Definition of default",
                "text": "Which event defines a default under Basel II?",
                "options": [
                    "Any payment that is one day late",
                    "A credit score below 600 points",
                    "More than 90 days past due on a material obligation, or the bank judges the borrower unlikely to pay",
                    "A downgrade by a rating agency"
                ],
                "correctExplanation": "Basel II, paragraph 452: unlikeliness to pay or more than 90 days past due on any material credit obligation.",
                "incorrectExplanation": "A short delay, a score or a rating change is not a default; Basel uses the 90-day rule and the unlikeliness-to-pay criterion."
            },
            "ro": {
                "title": "Definiția nerambursării",
                "text": "Ce eveniment definește nerambursarea (default) în Basel II?",
                "options": [
                    "Orice plată întîrziată cu o zi",
                    "Un scor de credit sub 600 de puncte",
                    "O întîrziere de peste 90 de zile la o obligație semnificativă sau aprecierea băncii că debitorul nu va plăti",
                    "Retrogradarea de către o agenție de rating"
                ],
                "correctExplanation": "Basel II, paragraful 452: probabilitatea mică de plată sau întîrzierea de peste 90 de zile la o obligație de credit semnificativă.",
                "incorrectExplanation": "O întîrziere scurtă, un scor sau o schimbare de rating nu sînt nerambursare; Basel folosește regula celor 90 de zile și criteriul probabilității mici de plată."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Odds",
                "text": "An applicant has PD = 0.25. What are the odds of default?",
                "options": [
                    "1/3",
                    "0.25",
                    "3",
                    "0.75"
                ],
                "correctExplanation": "Odds = p/(1 - p) = 0.25/0.75 = 1/3: one default for three repaid loans.",
                "incorrectExplanation": "The odds divide the probability of default by the probability of repayment; 3 is the good:bad odds."
            },
            "ro": {
                "title": "Șansa",
                "text": "Un solicitant are PD = 0,25. Cît este șansa (odds) de nerambursare?",
                "options": [
                    "1/3",
                    "0,25",
                    "3",
                    "0,75"
                ],
                "correctExplanation": "Șansa = p/(1 - p) = 0,25/0,75 = 1/3: o nerambursare la trei credite rambursate.",
                "incorrectExplanation": "Șansa împarte probabilitatea de nerambursare la probabilitatea de rambursare; 3 este șansa bun:rău."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "A logit coefficient",
                "text": "In a logit for default, a dummy variable has coefficient 0.69. What does it mean?",
                "options": [
                    "The PD rises by 0.69",
                    "The PD rises by 69%",
                    "The odds of default are halved",
                    "The odds of default are multiplied by about 2 for the group"
                ],
                "correctExplanation": "exp(0.69) is about 2: the odds ratio of the group relative to the reference group.",
                "incorrectExplanation": "A logit coefficient changes the log-odds; its exponential multiplies the odds, and the change in PD depends on the applicant."
            },
            "ro": {
                "title": "Un coeficient logit",
                "text": "Într-un logit pentru nerambursare, o variabilă indicator are coeficientul 0,69. Ce înseamnă?",
                "options": [
                    "PD crește cu 0,69",
                    "PD crește cu 69%",
                    "Șansa de nerambursare se înjumătățește",
                    "Șansa de nerambursare se înmulțește cu aproximativ 2 pentru grup"
                ],
                "correctExplanation": "exp(0,69) este aproximativ 2: raportul șanselor grupului față de grupul de referință.",
                "incorrectExplanation": "Un coeficient logit modifică logaritmul șansei; exponențiala lui înmulțește șansa, iar modificarea PD depinde de solicitant."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Altman Z-score",
                "text": "A firm has Altman Z = 2.36. In which zone is it?",
                "options": [
                    "Distress zone",
                    "Grey zone",
                    "Safe zone",
                    "Z cannot be interpreted without a PD"
                ],
                "correctExplanation": "Altman (1968): below 1.81 distress, 1.81 to 2.99 grey, above 2.99 safe; 2.36 is in between.",
                "incorrectExplanation": "The two cut-offs are 1.81 and 2.99; Z is a discriminant score, read through these zones."
            },
            "ro": {
                "title": "Scorul Z al lui Altman",
                "text": "O companie are scorul Z Altman = 2,36. În ce zonă se află?",
                "options": [
                    "Zona de dificultate",
                    "Zona gri",
                    "Zona sigură",
                    "Z nu poate fi interpretat fără o PD"
                ],
                "correctExplanation": "Altman (1968): sub 1,81 dificultate, între 1,81 și 2,99 zona gri, peste 2,99 sigur; 2,36 este între praguri.",
                "incorrectExplanation": "Cele două praguri sînt 1,81 și 2,99; Z este un scor discriminant, citit prin aceste zone."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Information Value",
                "text": "The checking account of the South German Credit data has IV = 0.67. How is it classified?",
                "options": [
                    "Useless",
                    "Weak",
                    "Medium",
                    "Strong"
                ],
                "correctExplanation": "Rule of thumb (Siddiqi, 2006): above 0.3 strong.",
                "incorrectExplanation": "The thresholds are 0.02, 0.1 and 0.3; 0.67 is well above the last one."
            },
            "ro": {
                "title": "Information Value",
                "text": "Contul curent din datele South German Credit are IV = 0,67. Cum se clasifică?",
                "options": [
                    "Inutilă",
                    "Slabă",
                    "Medie",
                    "Puternică"
                ],
                "correctExplanation": "Regula practică (Siddiqi, 2006): peste 0,3 puternică.",
                "incorrectExplanation": "Pragurile sînt 0,02; 0,1 și 0,3; 0,67 este mult peste ultimul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "A lesson in data checking",
                "text": "In the widely used UCI Statlog German credit data, \"no checking account\" was the safest group (11.7% bad). What did the corrected South German Credit data show?",
                "options": [
                    "The label was wrong: those 394 credits have a balance of at least 200 DM or a salary account; no checking account is the riskiest group (49.3% bad)",
                    "The default rates were wrong, the labels right",
                    "No checking account is indeed the safest group",
                    "The data set cannot be used for scoring"
                ],
                "correctExplanation": "Groemping (2019): the numbers were right, the code labels wrong; every interpretation of this variable was reversed.",
                "incorrectExplanation": "The counts and rates did not change; only the meaning of the codes was corrected."
            },
            "ro": {
                "title": "O lecție despre verificarea datelor",
                "text": "În datele UCI Statlog German credit, larg folosite, „fără cont curent” era grupul cel mai sigur (11,7% neperformante). Ce au arătat datele corectate South German Credit?",
                "options": [
                    "Eticheta era greșită: acele 394 de credite au un sold de cel puțin 200 DM sau un cont de salariu; „fără cont curent” este grupul cel mai riscant (49,3% neperformante)",
                    "Ratele de nerambursare erau greșite, etichetele corecte",
                    "„Fără cont curent” este într-adevăr grupul cel mai sigur",
                    "Setul de date nu poate fi folosit pentru scoring"
                ],
                "correctExplanation": "Grömping (2019): cifrele erau corecte, etichetele codurilor greșite; fiecare interpretare a acestei variabile era inversată.",
                "incorrectExplanation": "Frecvențele și ratele nu s-au schimbat; s-a corectat doar sensul codurilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Accuracy paradox",
                "text": "A sample has 30% bad credits. What accuracy does the rule \"everybody is good\" reach?",
                "options": [
                    "30%",
                    "70%",
                    "50%",
                    "100%"
                ],
                "correctExplanation": "The rule classifies all goods correctly and all bads wrongly: 70% accuracy without any information.",
                "incorrectExplanation": "Accuracy counts all correct classifications; with imbalanced classes the majority class alone gives a high accuracy."
            },
            "ro": {
                "title": "Paradoxul acurateței",
                "text": "Un eșantion are 30% credite neperformante. Ce acuratețe atinge regula „toți sînt bun-platnici”?",
                "options": [
                    "30%",
                    "70%",
                    "50%",
                    "100%"
                ],
                "correctExplanation": "Regula clasifică corect toți bun-platnicii și greșit toți rău-platnicii: 70% acuratețe fără nicio informație.",
                "incorrectExplanation": "Acuratețea numără toate clasificările corecte; cu clase dezechilibrate, clasa majoritară singură dă o acuratețe mare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Prior correction and AUC",
                "text": "PDs estimated on a sample with 30% bads are corrected to a population with 5% bads. What happens to the AUC?",
                "options": [
                    "It falls",
                    "It rises",
                    "It does not change: every log-odds is shifted by the same constant, so the ranking is unchanged",
                    "It depends on the cut-off"
                ],
                "correctExplanation": "The correction adds ln[(0.05/0.95)/(0.30/0.70)] = -2.097 to every log-odds; the order of the applicants stays the same.",
                "incorrectExplanation": "AUC depends only on the ranking of the scores, and a common shift of the log-odds does not change the ranking."
            },
            "ro": {
                "title": "Corecția a priori și AUC",
                "text": "PD estimate pe un eșantion cu 30% rău-platnici sînt corectate pentru o populație cu 5% rău-platnici. Ce se întîmplă cu AUC?",
                "options": [
                    "Scade",
                    "Crește",
                    "Nu se schimbă: fiecare logaritm al șansei se deplasează cu aceeași constantă, deci ordonarea rămîne aceeași",
                    "Depinde de prag"
                ],
                "correctExplanation": "Corecția adaugă ln[(0,05/0,95)/(0,30/0,70)] = -2,097 la fiecare logaritm al șansei; ordinea solicitanților rămîne aceeași.",
                "incorrectExplanation": "AUC depinde doar de ordonarea scorurilor, iar o deplasare comună a logaritmului șansei nu schimbă ordonarea."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Equal opportunity",
                "text": "Which statement describes the fairness criterion \"equal opportunity\" at a cut-off?",
                "options": [
                    "Equal approval rates in all groups",
                    "Equal default rates in all groups",
                    "The protected variable is not a model input",
                    "Equal approval rates of the good payers in all groups"
                ],
                "correctExplanation": "Hardt, Price and Srebro (2016): equal true positive rates of the deserving class, here the approval rates of good payers.",
                "incorrectExplanation": "Equal approval rates of all applicants is demographic parity; leaving the variable out does not remove its influence."
            },
            "ro": {
                "title": "Egalitatea de șanse",
                "text": "Ce afirmație descrie criteriul de echitate „egalitatea de șanse” la un prag?",
                "options": [
                    "Rate egale de aprobare în toate grupurile",
                    "Rate egale de nerambursare în toate grupurile",
                    "Variabila protejată nu este o variabilă a modelului",
                    "Rate egale de aprobare a bun-platnicilor în toate grupurile"
                ],
                "correctExplanation": "Hardt, Price și Srebro (2016): rate egale de adevărat pozitive pentru clasa îndreptățită, aici ratele de aprobare a bun-platnicilor.",
                "incorrectExplanation": "Ratele egale de aprobare pentru toți solicitanții înseamnă paritate demografică; excluderea variabilei nu îi elimină influența."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Fisher's direction",
                "text": "Which vector does Fisher's linear discriminant use to separate bads from goods?",
                "options": [
                    "w = S_W^{-1}(m_1 - m_0)",
                    "w = m_1 - m_0",
                    "w = S_W(m_1 - m_0)",
                    "The first principal component of all applicants"
                ],
                "correctExplanation": "Maximising the between-group distance relative to the within-group variance gives w proportional to S_W^{-1}(m_1 - m_0).",
                "incorrectExplanation": "The difference of the means must be corrected by the inverse of the pooled within-group covariance; principal components ignore the groups."
            },
            "ro": {
                "title": "Direcția Fisher",
                "text": "Ce vector folosește discriminantul liniar al lui Fisher pentru a separa rău-platnicii de bun-platnici?",
                "options": [
                    "w = S_W^{-1}(m_1 - m_0)",
                    "w = m_1 - m_0",
                    "w = S_W(m_1 - m_0)",
                    "Prima componentă principală a tuturor solicitanților"
                ],
                "correctExplanation": "Maximizarea distanței dintre grupuri raportată la varianța din interiorul grupurilor dă w proporțional cu S_W^{-1}(m_1 - m_0).",
                "incorrectExplanation": "Diferența mediilor trebuie corectată cu inversa covarianței comune din interiorul grupurilor; componentele principale ignoră grupurile."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Logit or LDA",
                "text": "On the Taiwan data, logit and LDA have the same AUC, but the LDA PDs are badly calibrated. Why?",
                "options": [
                    "LDA uses fewer variables",
                    "LDA needs Normal inputs with a common covariance for correct PDs; the inputs are dummies and skewed",
                    "The logit overfits",
                    "AUC cannot be computed for LDA"
                ],
                "correctExplanation": "Fisher's direction ranks well without any distribution; its posterior probabilities rely on Normal classes.",
                "incorrectExplanation": "Both models used the same inputs; the ranking is equal, the difference is in the level of the PDs."
            },
            "ro": {
                "title": "Logit sau LDA",
                "text": "Pe datele din Taiwan, logit și LDA au același AUC, dar PD-urile LDA sînt prost calibrate. De ce?",
                "options": [
                    "LDA folosește mai puține variabile",
                    "LDA cere variabile Normale cu covarianță comună pentru PD corecte; variabilele sînt indicatori și asimetrice",
                    "Logit-ul supraajustează",
                    "AUC nu poate fi calculat pentru LDA"
                ],
                "correctExplanation": "Direcția Fisher ordonează bine fără nicio distribuție; probabilitățile ei a posteriori se bazează pe clase Normale.",
                "incorrectExplanation": "Ambele modele au folosit aceleași variabile; ordonarea este egală, diferența este în nivelul PD."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Scorecard factor",
                "text": "A scorecard uses 20 points to double the odds (PDO = 20). What is the Factor?",
                "options": [
                    "20",
                    "13.86",
                    "28.85",
                    "0.035"
                ],
                "correctExplanation": "Factor = PDO / ln 2 = 20 / 0.693 = 28.85 points per unit of log-odds.",
                "incorrectExplanation": "Doubling the odds adds ln 2 to the log-odds, so the factor is PDO divided by ln 2."
            },
            "ro": {
                "title": "Factorul scorecard-ului",
                "text": "Un scorecard folosește 20 de puncte pentru dublarea șansei (PDO = 20). Cît este Factor?",
                "options": [
                    "20",
                    "13,86",
                    "28,85",
                    "0,035"
                ],
                "correctExplanation": "Factor = PDO / ln 2 = 20 / 0,693 = 28,85 puncte pe unitatea logaritmului șansei.",
                "incorrectExplanation": "Dublarea șansei adaugă ln 2 la logaritmul șansei, deci factorul este PDO împărțit la ln 2."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "PDO and PD",
                "text": "With PDO = 20, how many points does an applicant gain when the PD falls from 20% to 10%?",
                "options": [
                    "Exactly 20",
                    "Exactly 10",
                    "About 40",
                    "About 23.4"
                ],
                "correctExplanation": "The good:bad odds go from 4:1 to 9:1: 28.85 x ln(9/4) = 23.4 points.",
                "incorrectExplanation": "PDO doubles the good:bad odds, not the PD; halving the PD more than doubles these odds here."
            },
            "ro": {
                "title": "PDO și PD",
                "text": "Cu PDO = 20, cîte puncte cîștigă un solicitant cînd PD scade de la 20% la 10%?",
                "options": [
                    "Exact 20",
                    "Exact 10",
                    "Aproximativ 40",
                    "Aproximativ 23,4"
                ],
                "correctExplanation": "Șansa bun:rău trece de la 4:1 la 9:1: 28,85 x ln(9/4) = 23,4 puncte.",
                "incorrectExplanation": "PDO dublează șansa bun:rău, nu PD; înjumătățirea PD face aici mai mult decît să dubleze această șansă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Meaning of AUC",
                "text": "A scorecard has AUC = 0.80. What does this mean?",
                "options": [
                    "A random bad has a higher PD than a random good in 80% of the pairs",
                    "80% of the applicants are classified correctly",
                    "80% of the bads are rejected",
                    "The PDs are 80% accurate"
                ],
                "correctExplanation": "AUC is the probability that the score orders a (bad, good) pair correctly (Hanley and McNeil, 1982).",
                "incorrectExplanation": "AUC summarises all cut-offs at once; accuracy and the share of bads rejected depend on one cut-off."
            },
            "ro": {
                "title": "Sensul AUC",
                "text": "Un scorecard are AUC = 0,80. Ce înseamnă?",
                "options": [
                    "Un rău-platnic aleator are o PD mai mare decît un bun-platnic aleator în 80% din perechi",
                    "80% dintre solicitanți sînt clasificați corect",
                    "80% dintre rău-platnici sînt respinși",
                    "PD-urile sînt exacte în proporție de 80%"
                ],
                "correctExplanation": "AUC este probabilitatea ca scorul să ordoneze corect o pereche (rău, bun) (Hanley și McNeil, 1982).",
                "incorrectExplanation": "AUC rezumă toate pragurile deodată; acuratețea și ponderea rău-platnicilor respinși depind de un singur prag."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Gini",
                "text": "A scorecard has AUC = 0.80. What is its Gini coefficient?",
                "options": [
                    "0.30",
                    "0.60",
                    "0.80",
                    "0.40"
                ],
                "correctExplanation": "Gini = 2 AUC - 1 = 0.60; for a continuous score it equals the accuracy ratio of the CAP curve.",
                "incorrectExplanation": "The Gini coefficient rescales AUC from [0.5, 1] to [0, 1]: subtracting 0.5 alone is not enough."
            },
            "ro": {
                "title": "Gini",
                "text": "Un scorecard are AUC = 0,80. Cît este coeficientul Gini?",
                "options": [
                    "0,30",
                    "0,60",
                    "0,80",
                    "0,40"
                ],
                "correctExplanation": "Gini = 2 AUC - 1 = 0,60; pentru un scor continuu este egal cu raportul de acuratețe al curbei CAP.",
                "incorrectExplanation": "Coeficientul Gini rescalează AUC de la [0,5; 1] la [0; 1]: simpla scădere a lui 0,5 nu este suficientă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "AUC by counting",
                "text": "Three bads and seven goods: 17 of the 21 (bad, good) pairs are ordered correctly, without ties. What is the AUC?",
                "options": [
                    "0.17",
                    "0.70",
                    "0.81",
                    "0.85"
                ],
                "correctExplanation": "AUC = 17/21 = 0.81: the share of correctly ordered pairs.",
                "incorrectExplanation": "Divide the correctly ordered pairs by the number of (bad, good) pairs, 3 x 7 = 21."
            },
            "ro": {
                "title": "AUC prin numărare",
                "text": "Trei rău-platnici și șapte bun-platnici: 17 din cele 21 de perechi (rău, bun) sînt ordonate corect, fără egalități. Cît este AUC?",
                "options": [
                    "0,17",
                    "0,70",
                    "0,81",
                    "0,85"
                ],
                "correctExplanation": "AUC = 17/21 = 0,81: ponderea perechilor ordonate corect.",
                "incorrectExplanation": "Se împart perechile ordonate corect la numărul perechilor (rău, bun), 3 x 7 = 21."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Cost-based cut-off",
                "text": "Accepting a bad costs 5, rejecting a good costs 1. Above which PD should the bank reject?",
                "options": [
                    "0.5",
                    "0.2",
                    "5/6",
                    "1/6"
                ],
                "correctExplanation": "Reject if p x 5 > (1 - p) x 1, i.e. p > 1/(1 + 5) = 1/6.",
                "incorrectExplanation": "The Bayes cut-off compares the expected cost of accepting with that of rejecting; 0.5 is right only for equal costs."
            },
            "ro": {
                "title": "Pragul pe baza costurilor",
                "text": "Acceptarea unui rău-platnic costă 5, respingerea unui bun-platnic costă 1. Peste ce PD ar trebui să respingă banca?",
                "options": [
                    "0,5",
                    "0,2",
                    "5/6",
                    "1/6"
                ],
                "correctExplanation": "Respingem dacă p x 5 > (1 - p) x 1, adică p > 1/(1 + 5) = 1/6.",
                "incorrectExplanation": "Pragul Bayes compară costul așteptat al acceptării cu cel al respingerii; 0,5 este corect doar pentru costuri egale."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Brier score",
                "text": "Two applicants: PD 0.2 for a good payer and PD 0.9 for a defaulter. What is the Brier score?",
                "options": [
                    "0.025",
                    "0.05",
                    "0.15",
                    "0.50"
                ],
                "correctExplanation": "Brier = [(0.2 - 0)^2 + (0.9 - 1)^2]/2 = (0.04 + 0.01)/2 = 0.025.",
                "incorrectExplanation": "Average the squared differences between each PD and the outcome (0 or 1)."
            },
            "ro": {
                "title": "Scorul Brier",
                "text": "Doi solicitanți: PD 0,2 pentru un bun-platnic și PD 0,9 pentru un rău-platnic. Cît este scorul Brier?",
                "options": [
                    "0,025",
                    "0,05",
                    "0,15",
                    "0,50"
                ],
                "correctExplanation": "Brier = [(0,2 - 0)^2 + (0,9 - 1)^2]/2 = (0,04 + 0,01)/2 = 0,025.",
                "incorrectExplanation": "Se face media pătratelor diferențelor dintre fiecare PD și rezultat (0 sau 1)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Calibration test",
                "text": "On 9000 test clients the Hosmer-Lemeshow test rejects calibration, although the groups lie close to the diagonal. What follows?",
                "options": [
                    "The model is useless",
                    "With many observations the test detects small deviations: judge the size of the gaps, not only the p-value",
                    "The AUC must be wrong",
                    "The test must use five groups"
                ],
                "correctExplanation": "The power of the test grows with n; a deviation of one or two percentage points becomes significant.",
                "incorrectExplanation": "A significant test does not say that the deviations are large; the reliability diagram shows their size."
            },
            "ro": {
                "title": "Testul de calibrare",
                "text": "Pe 9000 de clienți de test, testul Hosmer-Lemeshow respinge calibrarea, deși grupurile sînt aproape de diagonală. Ce rezultă?",
                "options": [
                    "Modelul este inutil",
                    "Cu multe observații, testul detectează abateri mici: judecăm mărimea abaterilor, nu doar p-valoarea",
                    "AUC trebuie să fie greșit",
                    "Testul trebuie să folosească cinci grupuri"
                ],
                "correctExplanation": "Puterea testului crește cu n; o abatere de unul sau două puncte procentuale devine semnificativă.",
                "incorrectExplanation": "Un test semnificativ nu spune că abaterile sînt mari; diagrama de calibrare arată mărimea lor."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Reject inference",
                "text": "Why is a scorecard estimated only on accepted applicants a problem?",
                "options": [
                    "Accepted applicants have too many defaults",
                    "The test sample becomes too small",
                    "Outcomes are known only for the accepted, so the model must extrapolate to applicants unlike them",
                    "The logit cannot be estimated with dummies"
                ],
                "correctExplanation": "On the Taiwan data, the model trained on the accepted underestimated the default rate of the rejected (about 30% predicted, 46% observed).",
                "incorrectExplanation": "The issue is selection: the riskiest applicants and their outcomes are missing from the estimation sample."
            },
            "ro": {
                "title": "Reject inference",
                "text": "De ce este o problemă un scorecard estimat doar pe solicitanții acceptați?",
                "options": [
                    "Solicitanții acceptați au prea multe nerambursări",
                    "Eșantionul de test devine prea mic",
                    "Rezultatele sînt cunoscute doar pentru acceptați, deci modelul trebuie să extrapoleze la solicitanți diferiți de ei",
                    "Logit-ul nu poate fi estimat cu variabile indicator"
                ],
                "correctExplanation": "Pe datele din Taiwan, modelul estimat pe acceptați a subestimat rata de nerambursare a respinșilor (aproximativ 30% prezis, 46% observat).",
                "incorrectExplanation": "Problema este selecția: solicitanții cei mai riscanți și rezultatele lor lipsesc din eșantionul de estimare."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Distance to default",
                "text": "Merton model: V = 100, D = 70, asset volatility 25%, drift 5%, T = 1 year. What are DD and PD?",
                "options": [
                    "DD = 0.36, PD = 36%",
                    "DD = 3.0, PD = 0.1%",
                    "DD = 1.2, PD = 12%",
                    "DD = 1.50, PD = 6.7%"
                ],
                "correctExplanation": "DD = [ln(100/70) + (0.05 - 0.03125)]/0.25 = 1.50; PD = N(-1.50) = 6.7%.",
                "incorrectExplanation": "Use the log of assets over debt plus the drift correction, divided by the volatility; the PD is the Normal tail below -DD."
            },
            "ro": {
                "title": "Distanța pînă la nerambursare",
                "text": "Modelul Merton: V = 100, D = 70, volatilitatea activelor 25%, tendința 5%, T = 1 an. Cît sînt DD și PD?",
                "options": [
                    "DD = 0,36; PD = 36%",
                    "DD = 3,0; PD = 0,1%",
                    "DD = 1,2; PD = 12%",
                    "DD = 1,50; PD = 6,7%"
                ],
                "correctExplanation": "DD = [ln(100/70) + (0,05 - 0,03125)]/0,25 = 1,50; PD = N(-1,50) = 6,7%.",
                "incorrectExplanation": "Se folosește logaritmul raportului active/datorie plus corecția tendinței, împărțit la volatilitate; PD este coada distribuției Normale sub -DD."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "IRB capital",
                "text": "What does the Basel IRB capital K of a loan cover?",
                "options": [
                    "The unexpected loss: the credit loss in a 1-in-1000 bad year minus the expected loss",
                    "The expected loss PD x LGD x EAD",
                    "The whole exposure EAD",
                    "Only the losses of defaulted loans"
                ],
                "correctExplanation": "K = LGD [N((N^-1(PD) + sqrt(R) N^-1(0.999))/sqrt(1 - R)) - PD]: the PD of a very bad year minus the PD, times LGD.",
                "incorrectExplanation": "The expected loss is priced in and covered by provisions; capital is held for losses above it."
            },
            "ro": {
                "title": "Capitalul IRB",
                "text": "Ce acoperă capitalul IRB K al unui credit în Basel?",
                "options": [
                    "Pierderea neașteptată: pierderea din credit într-un an prost, care apare o dată la o mie de ani, minus pierderea așteptată",
                    "Pierderea așteptată PD x LGD x EAD",
                    "Întreaga expunere EAD",
                    "Doar pierderile creditelor deja nerambursate"
                ],
                "correctExplanation": "K = LGD [N((N^-1(PD) + sqrt(R) N^-1(0,999))/sqrt(1 - R)) - PD]: PD dintr-un an foarte prost minus PD, înmulțită cu LGD.",
                "incorrectExplanation": "Pierderea așteptată intră în preț și este acoperită de provizioane; capitalul se constituie pentru pierderile de peste ea."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Cross-validation",
                "text": "In repeated cross-validation the WoE scorecard has training AUC 0.80 and test AUC 0.76. What should be done inside every fold?",
                "options": [
                    "Nothing: the bins can be set once on all data",
                    "Only the coefficients are re-estimated",
                    "Bins, WoE, variable selection and coefficients are all redone on the training folds",
                    "The test folds are used to choose the variables"
                ],
                "correctExplanation": "Every choice made with the data belongs to the training folds; otherwise the test folds leak into the model.",
                "incorrectExplanation": "Bins or variables chosen on all data use the test outcomes and make the test AUC too optimistic."
            },
            "ro": {
                "title": "Validarea încrucișată",
                "text": "În validarea încrucișată repetată, scorecard-ul WoE are AUC de estimare 0,80 și AUC de test 0,76. Ce trebuie refăcut în fiecare parte?",
                "options": [
                    "Nimic: intervalele pot fi stabilite o singură dată pe toate datele",
                    "Doar coeficienții se reestimează",
                    "Intervalele, WoE, selecția variabilelor și coeficienții se refac toate pe părțile de estimare",
                    "Părțile de test se folosesc pentru alegerea variabilelor"
                ],
                "correctExplanation": "Orice alegere făcută pe baza datelor aparține părților de estimare; altfel, părțile de test se scurg în model.",
                "incorrectExplanation": "Intervalele sau variabilele alese pe toate datele folosesc rezultatele de test și fac AUC de test prea optimist."
            }
        }
    ]
};
