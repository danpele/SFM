// ============================================================
// Chapter 13 quiz bank: machine learning in finance (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['ml'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Prediction and inference",
                "text": "Which question is a prediction question rather than an inference question?",
                "options": [
                    "Is the leverage coefficient of a GJR-GARCH model different from zero?",
                    "How large will the realised volatility of the S&P 500 be next week?",
                    "Is the slope of a regression significant at 5%?",
                    "Does the size factor explain average returns?"
                ],
                "correctExplanation": "A prediction question is judged by the error of the forecast on new data; the other questions are about parameters and their standard errors.",
                "incorrectExplanation": "Questions about whether a coefficient is zero or significant belong to inference; forecasting next week's volatility is judged only by the forecast error on new data."
            },
            "ro": {
                "title": "Predicție și inferență",
                "text": "Care întrebare este una de predicție, nu de inferență?",
                "options": [
                    "Este coeficientul de levier al unui model GJR-GARCH diferit de zero?",
                    "Cît de mare va fi volatilitatea realizată a S&P 500 săptămîna viitoare?",
                    "Este panta unei regresii semnificativă la 5%?",
                    "Explică factorul de mărime randamentele medii?"
                ],
                "correctExplanation": "O întrebare de predicție se judecă după eroarea prognozei pe date noi; celelalte întrebări privesc parametrii și erorile lor standard.",
                "incorrectExplanation": "Întrebările despre un coeficient nul sau semnificativ țin de inferență; prognoza volatilității din săptămîna viitoare se judecă doar după eroarea pe date noi."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Overfitting",
                "text": "Which pattern is the typical symptom of overfitting?",
                "options": [
                    "A large training error and a large test error",
                    "A test error smaller than the training error",
                    "A small training error and a large test error",
                    "Equal training and test errors"
                ],
                "correctExplanation": "An overfitted model has learned the noise of the training sample: it fits the training data very well and new data badly.",
                "incorrectExplanation": "Large errors everywhere indicate underfitting; overfitting shows up as a gap: small error in training, large error on new data."
            },
            "ro": {
                "title": "Overfitting",
                "text": "Care tipar este simptomul tipic al overfitting-ului?",
                "options": [
                    "O eroare de antrenare mare și o eroare de test mare",
                    "O eroare de test mai mică decît eroarea de antrenare",
                    "O eroare de antrenare mică și o eroare de test mare",
                    "Erori de antrenare și de test egale"
                ],
                "correctExplanation": "Un model în overfitting a învățat zgomotul eșantionului de antrenare: se potrivește foarte bine pe datele de antrenare și prost pe date noi.",
                "incorrectExplanation": "Erorile mari peste tot indică underfitting; overfitting-ul apare ca o diferență: eroare mică la antrenare, eroare mare pe date noi."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Deep trees",
                "text": "A regression tree of depth 10 is compared with a tree of depth 1 on many simulated training samples. What is typical?",
                "options": [
                    "The deep tree has a small bias and a large variance",
                    "The deep tree has a large bias and a small variance",
                    "Both have the same variance",
                    "The deep tree always has the smaller test error"
                ],
                "correctExplanation": "A flexible model follows each training sample closely: little systematic error, but its forecasts change a lot from sample to sample.",
                "incorrectExplanation": "Flexibility lowers the bias and raises the variance; in the chapter simulation the depth-10 tree has a larger test error than the depth-3 tree."
            },
            "ro": {
                "title": "Arbori adînci",
                "text": "Un arbore de regresie de adîncime 10 este comparat cu unul de adîncime 1 pe multe eșantioane simulate. Ce este tipic?",
                "options": [
                    "Arborele adînc are deplasare mică și varianță mare",
                    "Arborele adînc are deplasare mare și varianță mică",
                    "Ambii au aceeași varianță",
                    "Arborele adînc are întotdeauna eroarea de test mai mică"
                ],
                "correctExplanation": "Un model flexibil urmează îndeaproape fiecare eșantion de antrenare: eroare sistematică mică, dar prognozele se schimbă mult de la un eșantion la altul.",
                "incorrectExplanation": "Flexibilitatea scade deplasarea și crește varianța; în simularea din curs, arborele de adîncime 10 are o eroare de test mai mare decît cel de adîncime 3."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Expected test error",
                "text": "At a point x0 a model has bias 0.3 and variance 0.05; the noise variance is 0.2. What is the expected squared error?",
                "options": [
                    "0.25",
                    "0.55",
                    "0.30",
                    "0.34"
                ],
                "correctExplanation": "The expected error is bias squared plus variance plus noise: 0.09 + 0.05 + 0.2 = 0.34.",
                "incorrectExplanation": "The bias enters squared (0.3 squared is 0.09), and the noise variance is always added: 0.09 + 0.05 + 0.2 = 0.34."
            },
            "ro": {
                "title": "Eroarea de test așteptată",
                "text": "Într-un punct x0, un model are deplasarea 0,3 și varianța 0,05; varianța zgomotului este 0,2. Cît este eroarea pătratică așteptată?",
                "options": [
                    "0,25",
                    "0,55",
                    "0,30",
                    "0,34"
                ],
                "correctExplanation": "Eroarea așteptată este deplasarea la pătrat plus varianța plus zgomotul: 0,09 + 0,05 + 0,2 = 0,34.",
                "incorrectExplanation": "Deplasarea intră la pătrat (0,3 la pătrat este 0,09), iar varianța zgomotului se adaugă întotdeauna: 0,09 + 0,05 + 0,2 = 0,34."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Random folds",
                "text": "A random forest predicts the next 21-day S&P 500 return. Random 5-fold cross-validation gives an out-of-sample R2 of 37%, walk-forward gives -38%. Why?",
                "options": [
                    "Walk-forward wastes data",
                    "Shuffled folds leak information through overlapping targets",
                    "The random forest has too few trees",
                    "The S&P 500 is predictable over 21 days"
                ],
                "correctExplanation": "Neighbouring days share 20 of 21 returns in their targets; shuffled folds put such neighbours in training and test, so the model has seen the answers.",
                "incorrectExplanation": "The same experiment on a simulated random walk, which has nothing to predict, also gives about 35% with shuffled folds: the gain is leakage, not skill."
            },
            "ro": {
                "title": "Partiții aleatoare",
                "text": "Un random forest prognozează randamentul S&P 500 din următoarele 21 de zile. Validarea încrucișată 5-fold aleatoare dă un R2 în afara eșantionului de 37%, walk-forward dă -38%. De ce?",
                "options": [
                    "Walk-forward irosește date",
                    "Partițiile amestecate produc leakage prin țintele suprapuse",
                    "Random forest are prea puțini arbori",
                    "S&P 500 este previzibil pe 21 de zile"
                ],
                "correctExplanation": "Zilele vecine au 20 din 21 de randamente comune în ținte; partițiile amestecate pun astfel de vecini în antrenare și în test, deci modelul a văzut răspunsurile.",
                "incorrectExplanation": "Același experiment pe un mers aleator simulat, care nu are nimic de prezis, dă tot circa 35% cu partiții amestecate: cîștigul provine din leakage, nu dintr-o abilitate reală."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Purging",
                "text": "The target is the mean variance of the next 5 days and the model is refitted each January. What does purging remove?",
                "options": [
                    "The first 5 days of the test year",
                    "The 5 days with the largest returns",
                    "The last 5 training days, whose targets reach into the test year",
                    "Every fifth observation"
                ],
                "correctExplanation": "The targets of the last 5 training days use days of the test year; dropping them keeps test information out of training.",
                "incorrectExplanation": "Purging concerns the overlap between training targets and the test period, which only affects the training days just before the test block."
            },
            "ro": {
                "title": "Purging",
                "text": "Ținta este varianța medie din următoarele 5 zile, iar modelul se reestimează în fiecare ianuarie. Ce elimină purging-ul?",
                "options": [
                    "Primele 5 zile ale anului de test",
                    "Cele 5 zile cu cele mai mari randamente",
                    "Ultimele 5 zile de antrenare, ale căror ținte ajung în anul de test",
                    "Fiecare a cincea observație"
                ],
                "correctExplanation": "Țintele ultimelor 5 zile de antrenare folosesc zile din anul de test; eliminarea lor ține informația de test în afara antrenării.",
                "incorrectExplanation": "Purging-ul privește suprapunerea dintre țintele de antrenare și perioada de test, care afectează doar zilele de antrenare de imediat dinaintea blocului de test."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Look-ahead bias",
                "text": "Which step introduces look-ahead bias into a walk-forward study?",
                "options": [
                    "Standardising the features with the mean and standard deviation of the whole 2000-2026 sample",
                    "Refitting the model each January",
                    "Using the 22-day mean of past variances as a feature",
                    "Comparing the model with HAR"
                ],
                "correctExplanation": "The full-sample mean and standard deviation contain future data; scaling must use the training window only.",
                "incorrectExplanation": "Past averages, yearly refits and benchmarks use only information available at the time; full-sample statistics do not."
            },
            "ro": {
                "title": "Look-ahead bias",
                "text": "Ce pas introduce look-ahead bias într-un studiu walk-forward?",
                "options": [
                    "Standardizarea variabilelor cu media și abaterea standard ale întregului eșantion 2000-2026",
                    "Reestimarea modelului în fiecare ianuarie",
                    "Folosirea mediei pe 22 de zile a varianțelor trecute ca variabilă",
                    "Compararea modelului cu HAR"
                ],
                "correctExplanation": "Media și abaterea standard pe tot eșantionul conțin date viitoare; scalarea trebuie făcută doar cu fereastra de antrenare.",
                "incorrectExplanation": "Mediile trecute, reestimarea anuală și reperele folosesc doar informație disponibilă la momentul respectiv; statisticile pe tot eșantionul nu."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Ridge in one dimension",
                "text": "One centred feature has Sxx = 100 and Sxy = 30. What is the ridge coefficient for lambda = 50?",
                "options": [
                    "0.30",
                    "0.60",
                    "0.25",
                    "0.20"
                ],
                "correctExplanation": "Ridge with one feature gives Sxy/(Sxx + lambda) = 30/150 = 0.20, the OLS coefficient 0.30 shrunk by the factor 100/150.",
                "incorrectExplanation": "The penalty is added to Sxx in the denominator: 30/(100 + 50) = 0.20; without it the OLS coefficient is 0.30."
            },
            "ro": {
                "title": "Ridge într-o dimensiune",
                "text": "O variabilă centrată are Sxx = 100 și Sxy = 30. Cît este coeficientul ridge pentru lambda = 50?",
                "options": [
                    "0,30",
                    "0,60",
                    "0,25",
                    "0,20"
                ],
                "correctExplanation": "Ridge cu o singură variabilă dă Sxy/(Sxx + lambda) = 30/150 = 0,20, adică coeficientul OLS 0,30 contractat cu factorul 100/150.",
                "incorrectExplanation": "Penalizarea se adaugă la Sxx în numitor: 30/(100 + 50) = 0,20; fără ea, coeficientul OLS este 0,30."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Lasso threshold",
                "text": "With the same data (Sxx = 100, Sxy = 30) and the objective sum of squares plus lambda |beta|, from which lambda is the lasso coefficient exactly zero?",
                "options": [
                    "30",
                    "60",
                    "100",
                    "15"
                ],
                "correctExplanation": "Soft thresholding gives max(|Sxy| - lambda/2, 0)/Sxx, which is zero once lambda/2 reaches 30, that is lambda = 60.",
                "incorrectExplanation": "The lasso subtracts lambda/2 from |Sxy| = 30; the coefficient vanishes when lambda/2 >= 30, so from lambda = 60."
            },
            "ro": {
                "title": "Pragul lasso",
                "text": "Cu aceleași date (Sxx = 100, Sxy = 30) și funcția obiectiv suma pătratelor plus lambda |beta|, de la ce lambda este coeficientul lasso exact zero?",
                "options": [
                    "30",
                    "60",
                    "100",
                    "15"
                ],
                "correctExplanation": "Soft thresholding dă max(|Sxy| - lambda/2, 0)/Sxx, care devine zero cînd lambda/2 ajunge la 30, adică lambda = 60.",
                "incorrectExplanation": "Lasso scade lambda/2 din |Sxy| = 30; coeficientul se anulează cînd lambda/2 >= 30, deci de la lambda = 60."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Ridge and lasso",
                "text": "What distinguishes the lasso from ridge regression?",
                "options": [
                    "The lasso can set coefficients exactly to zero",
                    "The lasso does not need a penalty parameter",
                    "Ridge selects variables, the lasso does not",
                    "The lasso is unbiased"
                ],
                "correctExplanation": "The absolute-value penalty has a kink at zero, so some coefficients become exactly zero: the lasso shrinks and selects.",
                "incorrectExplanation": "Both methods need lambda and both are biased; ridge shrinks every coefficient but never sets one exactly to zero."
            },
            "ro": {
                "title": "Ridge și lasso",
                "text": "Ce deosebește lasso de regresia ridge?",
                "options": [
                    "Lasso poate anula exact unii coeficienți",
                    "Lasso nu are nevoie de un parametru de penalizare",
                    "Ridge selectează variabile, lasso nu",
                    "Lasso este nedeplasat"
                ],
                "correctExplanation": "Penalizarea cu valoarea absolută are un punct unghiular în zero, deci unii coeficienți devin exact zero: lasso contractă și selectează.",
                "incorrectExplanation": "Ambele metode au nevoie de lambda și ambele sînt deplasate; ridge contractă toți coeficienții, dar nu anulează niciunul exact."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Splitting a tree",
                "text": "How does a regression tree choose a split at a node?",
                "options": [
                    "It splits at the median of every feature",
                    "It chooses the feature with the largest variance",
                    "It chooses the feature and threshold that minimise the sum of squared errors of the two leaves",
                    "It chooses the split that maximises the number of observations in the left leaf"
                ],
                "correctExplanation": "The greedy rule tries every feature and candidate threshold and keeps the one with the smallest total SSE around the two leaf means.",
                "incorrectExplanation": "The split is chosen by its fit, the SSE of the two leaves, not by medians, variances or leaf sizes."
            },
            "ro": {
                "title": "Împărțirea unui arbore",
                "text": "Cum alege un arbore de regresie o împărțire într-un nod?",
                "options": [
                    "Împarte la mediana fiecărei variabile",
                    "Alege variabila cu cea mai mare varianță",
                    "Alege variabila și pragul care minimizează suma pătratelor erorilor celor două frunze",
                    "Alege împărțirea care maximizează numărul de observații din frunza stîngă"
                ],
                "correctExplanation": "Regula greedy încearcă fiecare variabilă și fiecare prag candidat și păstrează varianta cu cel mai mic SSE total în jurul mediilor celor două frunze.",
                "incorrectExplanation": "Împărțirea se alege după calitatea potrivirii, adică SSE al celor două frunze, nu după mediane, varianțe sau mărimea frunzelor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Averaging trees",
                "text": "B trees each have variance sigma^2 and pairwise correlation rho. What happens to the variance of their average as B grows?",
                "options": [
                    "It falls to zero",
                    "It stays at sigma^2",
                    "It grows with B",
                    "It falls towards rho sigma^2"
                ],
                "correctExplanation": "The variance of the average is rho sigma^2 + (1 - rho) sigma^2 / B: the second term vanishes, the first remains.",
                "incorrectExplanation": "Correlated trees cannot cancel all their errors: the floor rho sigma^2 remains, which is why random forests decorrelate the trees."
            },
            "ro": {
                "title": "Media arborilor",
                "text": "B arbori au fiecare varianța sigma^2 și corelația rho între oricare doi. Ce se întîmplă cu varianța mediei lor cînd B crește?",
                "options": [
                    "Scade la zero",
                    "Rămîne sigma^2",
                    "Crește cu B",
                    "Scade spre rho sigma^2"
                ],
                "correctExplanation": "Varianța mediei este rho sigma^2 + (1 - rho) sigma^2 / B: al doilea termen dispare, primul rămîne.",
                "incorrectExplanation": "Arborii corelați nu își pot anula toate erorile: rămîne pragul rho sigma^2, de aceea random forest decorelează arborii."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Learning rate",
                "text": "In gradient boosting, what is the usual effect of a smaller learning rate?",
                "options": [
                    "Fewer trees are needed",
                    "More trees are needed, and the validation error is less sensitive to their number",
                    "The model can no longer overfit",
                    "The trees become deeper"
                ],
                "correctExplanation": "Small steps need more trees to reach the minimum, but the validation error rises more slowly after it.",
                "incorrectExplanation": "A small learning rate does not remove overfitting; in the chapter, the rate 0.3 reached its minimum after 13 trees and then deteriorated quickly, the rate 0.03 after 154 trees and changed little."
            },
            "ro": {
                "title": "Rata de învățare",
                "text": "În gradient boosting, care este efectul obișnuit al unei rate de învățare mai mici?",
                "options": [
                    "Sînt necesari mai puțini arbori",
                    "Sînt necesari mai mulți arbori, iar eroarea de validare este mai puțin sensibilă la numărul lor",
                    "Modelul nu mai poate intra în overfitting",
                    "Arborii devin mai adînci"
                ],
                "correctExplanation": "Pașii mici au nevoie de mai mulți arbori pentru a atinge minimul, dar eroarea de validare crește mai lent după el.",
                "incorrectExplanation": "O rată mică nu elimină overfitting-ul; în curs, rata 0,3 a atins minimul după 13 arbori și apoi s-a deteriorat repede, rata 0,03 după 154 de arbori și s-a schimbat puțin."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Counting parameters",
                "text": "An MLP has 3 inputs, one hidden layer with 5 neurons and one output. How many weights and biases does it have?",
                "options": [
                    "15",
                    "20",
                    "26",
                    "9"
                ],
                "correctExplanation": "Hidden layer: 3 x 5 weights + 5 biases = 20; output: 5 weights + 1 bias = 6; in total 26.",
                "incorrectExplanation": "Every hidden neuron has one weight per input and a bias, and the output has one weight per hidden neuron and a bias: 20 + 6 = 26."
            },
            "ro": {
                "title": "Numărarea parametrilor",
                "text": "Un MLP are 3 intrări, un strat ascuns cu 5 neuroni și o ieșire. Cîte ponderi și termeni liberi are?",
                "options": [
                    "15",
                    "20",
                    "26",
                    "9"
                ],
                "correctExplanation": "Stratul ascuns: 3 x 5 ponderi + 5 termeni liberi = 20; ieșirea: 5 ponderi + 1 termen liber = 6; în total 26.",
                "incorrectExplanation": "Fiecare neuron ascuns are cîte o pondere pentru fiecare intrare și un termen liber, iar ieșirea are cîte o pondere pentru fiecare neuron ascuns și un termen liber: 20 + 6 = 26."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Neural-network ARCH",
                "text": "After the Brexit vote, GARCH(1,1) volatility of GBP/USD rose much more than that of an RBF network with the last 3 returns as inputs. Why?",
                "options": [
                    "GARCH carries the whole past through the lagged variance; the network forgets a shock after 3 days",
                    "The network has too many units",
                    "GARCH uses future returns",
                    "The network is estimated by least squares"
                ],
                "correctExplanation": "GARCH has memory through beta sigma^2_{t-1}; a network whose inputs are 3 lagged returns can only react to the last 3 days.",
                "incorrectExplanation": "Neither model uses future data; the difference is memory: volatility clustering needs the persistence that GARCH builds in."
            },
            "ro": {
                "title": "ARCH cu rețea neuronală",
                "text": "După votul pentru Brexit, volatilitatea GARCH(1,1) a cursului GBP/USD a crescut mult mai mult decît cea a unei rețele RBF cu ultimele 3 randamente ca intrări. De ce?",
                "options": [
                    "GARCH transmite tot trecutul prin varianța de ieri; rețeaua uită un șoc după 3 zile",
                    "Rețeaua are prea multe unități",
                    "GARCH folosește randamente viitoare",
                    "Rețeaua este estimată prin cele mai mici pătrate"
                ],
                "correctExplanation": "GARCH are memorie prin beta sigma^2_{t-1}; o rețea ale cărei intrări sînt ultimele 3 randamente poate reacționa doar la ultimele 3 zile.",
                "incorrectExplanation": "Niciun model nu folosește date viitoare; diferența este memoria: volatility clustering are nevoie de persistența pe care o construiește GARCH."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Forecasting a level",
                "text": "A network forecasts the USD/JPY level from its last 3 values. Which benchmark must it beat?",
                "options": [
                    "The historical mean of the level",
                    "A zero forecast",
                    "A linear trend",
                    "The random walk: tomorrow equals today"
                ],
                "correctExplanation": "Exchange rates are close to a random walk, so the naive forecast tomorrow = today is hard to beat; in the chapter the network had a test RMSE about 30 times larger.",
                "incorrectExplanation": "A level series looks predictable because consecutive values are close; only the random walk tells whether the model adds anything."
            },
            "ro": {
                "title": "Prognoza unui nivel",
                "text": "O rețea prognozează nivelul USD/JPY din ultimele 3 valori. Ce reper trebuie să depășească?",
                "options": [
                    "Media istorică a nivelului",
                    "O prognoză zero",
                    "O tendință liniară",
                    "Mersul aleator: mîine este egal cu azi"
                ],
                "correctExplanation": "Cursurile de schimb sînt aproape un mers aleator, deci prognoza naivă mîine = azi este greu de depășit; în curs, rețeaua a avut un RMSE de test de circa 30 de ori mai mare.",
                "incorrectExplanation": "O serie de niveluri pare previzibilă deoarece valorile consecutive sînt apropiate; doar mersul aleator arată dacă modelul adaugă ceva."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The HAR model",
                "text": "What are the regressors of the HAR model for realised variance?",
                "options": [
                    "The last 22 daily variances, each with its own coefficient",
                    "The variance of the last day, the mean of the last week and the mean of the last month",
                    "The returns of the last three days",
                    "Implied volatility only"
                ],
                "correctExplanation": "Corsi (2009) uses three horizons: daily, weekly (5 days) and monthly (22 days) averages; a simple model that mimics long memory.",
                "incorrectExplanation": "HAR summarises the past with three averages, not with 22 free coefficients, and it uses realised variances, not returns."
            },
            "ro": {
                "title": "Modelul HAR",
                "text": "Care sînt regresorii modelului HAR pentru varianța realizată?",
                "options": [
                    "Ultimele 22 de varianțe zilnice, fiecare cu propriul coeficient",
                    "Varianța din ultima zi, media din ultima săptămînă și media din ultima lună",
                    "Randamentele din ultimele trei zile",
                    "Doar volatilitatea implicită"
                ],
                "correctExplanation": "Corsi (2009) folosește trei orizonturi: medii zilnice, săptămînale (5 zile) și lunare (22 de zile); un model simplu care imită memoria lungă.",
                "incorrectExplanation": "HAR rezumă trecutul prin trei medii, nu prin 22 de coeficienți liberi, și folosește varianțe realizate, nu randamente."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Out-of-sample R2",
                "text": "A model has an out-of-sample R2 of -5% against HAR. What does it mean?",
                "options": [
                    "The model explains 5% of the variance",
                    "The model is 5% better than HAR",
                    "Its squared forecast errors are 5% larger than those of HAR",
                    "The test is not significant"
                ],
                "correctExplanation": "R2_OOS = 1 - MSE(model)/MSE(benchmark); a negative value means the model is worse than the benchmark.",
                "incorrectExplanation": "The out-of-sample R2 compares two forecasts on new data; a negative value is a loss against the benchmark, not a share of variance explained."
            },
            "ro": {
                "title": "R2 în afara eșantionului",
                "text": "Un model are un R2 în afara eșantionului de -5% față de HAR. Ce înseamnă?",
                "options": [
                    "Modelul explică 5% din varianță",
                    "Modelul este cu 5% mai bun decît HAR",
                    "Erorile pătratice ale prognozelor lui sînt cu 5% mai mari decît ale HAR",
                    "Testul nu este semnificativ"
                ],
                "correctExplanation": "R2_OOS = 1 - MSE(model)/MSE(reper); o valoare negativă înseamnă că modelul este mai slab decît reperul.",
                "incorrectExplanation": "R2 în afara eșantionului compară două prognoze pe date noi; o valoare negativă este o pierdere față de reper, nu o pondere a varianței explicate."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "QLIKE",
                "text": "The true variance is 1. Which forecast does QLIKE punish more: 0.5 or 2?",
                "options": [
                    "0.5: QLIKE punishes under-prediction more",
                    "2: QLIKE punishes over-prediction more",
                    "Both equally",
                    "Neither: QLIKE ignores the size of the error"
                ],
                "correctExplanation": "QLIKE = RV/h + ln h gives 2 - 0.693 = 1.307 for h = 0.5 and 0.5 + 0.693 = 1.193 for h = 2.",
                "incorrectExplanation": "Computing RV/h + ln h for both forecasts shows that the under-forecast costs more, the costly error in risk management."
            },
            "ro": {
                "title": "QLIKE",
                "text": "Varianța reală este 1. Ce prognoză penalizează QLIKE mai mult: 0,5 sau 2?",
                "options": [
                    "0,5: QLIKE penalizează mai mult subestimarea",
                    "2: QLIKE penalizează mai mult supraestimarea",
                    "Pe amîndouă la fel",
                    "Niciuna: QLIKE ignoră mărimea erorii"
                ],
                "correctExplanation": "QLIKE = RV/h + ln h dă 2 - 0,693 = 1,307 pentru h = 0,5 și 0,5 + 0,693 = 1,193 pentru h = 2.",
                "incorrectExplanation": "Calculul RV/h + ln h pentru ambele prognoze arată că subestimarea costă mai mult, eroarea costisitoare în managementul riscului."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Diebold-Mariano",
                "text": "With d_t = QLIKE(model) - QLIKE(HAR), the Diebold-Mariano statistic is -5.1. What do you conclude?",
                "options": [
                    "HAR is significantly better",
                    "The two forecasts are equal",
                    "The test cannot be used for variance forecasts",
                    "The model is significantly better than HAR"
                ],
                "correctExplanation": "A negative mean loss difference means the model has the smaller loss; -5.1 is far beyond the 5% critical value of the standard Normal.",
                "incorrectExplanation": "The sign follows the definition of d_t: negative means the first forecast has the smaller loss."
            },
            "ro": {
                "title": "Diebold-Mariano",
                "text": "Cu d_t = QLIKE(model) - QLIKE(HAR), statistica Diebold-Mariano este -5,1. Ce concluzionați?",
                "options": [
                    "HAR este semnificativ mai bun",
                    "Cele două prognoze sînt egale",
                    "Testul nu poate fi folosit pentru prognoze de varianță",
                    "Modelul este semnificativ mai bun decît HAR"
                ],
                "correctExplanation": "O diferență medie negativă a pierderilor înseamnă că modelul are pierderea mai mică; -5,1 este mult dincolo de valoarea critică de 5% a distribuției Normale standard.",
                "incorrectExplanation": "Semnul decurge din definiția lui d_t: negativ înseamnă că prima prognoză are pierderea mai mică."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The right baseline",
                "text": "The S&P 500 rose on 54.4% of the days since 2008. A random forest predicts the daily sign correctly on 53.1% of the days. What do you conclude?",
                "options": [
                    "The forest has skill, since 53.1% is above 50%",
                    "The forest is worse than always forecasting up",
                    "The forest is better than the baseline",
                    "The accuracy cannot be compared"
                ],
                "correctExplanation": "The majority-class baseline, always up, is right on 54.4% of the days; the forest falls short of it.",
                "incorrectExplanation": "A coin is the wrong benchmark: when the market rises on most days, a constant forecast of up already beats 50%."
            },
            "ro": {
                "title": "Reperul corect",
                "text": "S&P 500 a crescut în 54,4% din zile din 2008. Un random forest prezice corect semnul zilnic în 53,1% din zile. Ce concluzionați?",
                "options": [
                    "Random forest are o abilitate reală, deoarece 53,1% este peste 50%",
                    "Random forest este mai slab decît prognoza constantă de creștere",
                    "Random forest este mai bun decît reperul",
                    "Acuratețea nu poate fi comparată"
                ],
                "correctExplanation": "Reperul clasei majoritare, mereu creștere, are dreptate în 54,4% din zile; random forest rămîne sub el.",
                "incorrectExplanation": "O monedă este reperul greșit: cînd piața crește în majoritatea zilelor, o prognoză constantă de creștere depășește deja 50%."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "VaR by quantile regression",
                "text": "How is VaR 1% obtained from a quantile regression of the next-day return?",
                "options": [
                    "Estimate the 99% quantile and take it with a plus sign",
                    "Estimate the mean and add 2.33 standard deviations",
                    "Estimate the 1% quantile q and take VaR 1% = -q",
                    "Estimate the 1% quantile of the absolute return"
                ],
                "correctExplanation": "VaR 1% is the loss exceeded with probability 1%: minus the conditional 1% quantile of the return, a positive number.",
                "incorrectExplanation": "The level is the tail probability: the regression targets the lower 1% tail of the return, and the minus sign turns it into a positive loss."
            },
            "ro": {
                "title": "VaR prin regresie cuantilică",
                "text": "Cum se obține VaR 1% dintr-o regresie cuantilică a randamentului zilei următoare?",
                "options": [
                    "Estimăm cuantila de 99% și o luăm cu semnul plus",
                    "Estimăm media și adunăm 2,33 abateri standard",
                    "Estimăm cuantila de 1% q și luăm VaR 1% = -q",
                    "Estimăm cuantila de 1% a randamentului absolut"
                ],
                "correctExplanation": "VaR 1% este pierderea depășită cu probabilitatea 1%: minus cuantila condiționată de 1% a randamentului, un număr pozitiv.",
                "incorrectExplanation": "Nivelul este probabilitatea cozii: regresia vizează coada stîngă de 1% a randamentului, iar semnul minus o transformă într-o pierdere pozitivă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The best of many strategies",
                "text": "Over 10 years, 100 strategies without any skill are backtested. What annual Sharpe ratio does the best of them show on average?",
                "options": [
                    "0",
                    "About 0.1",
                    "About 2.5",
                    "About 0.8"
                ],
                "correctExplanation": "Each Sharpe ratio is about N(0, 1/10); the expected maximum of 100 such draws is about 0.8, by simulation and by the formula of Bailey and Lopez de Prado.",
                "incorrectExplanation": "Selecting the best of many trials biases the Sharpe ratio upwards; with 100 trials over 10 years the bias is about 0.8, which the deflated Sharpe ratio corrects."
            },
            "ro": {
                "title": "Cea mai bună dintre multe strategii",
                "text": "Pe 10 ani se testează 100 de strategii fără nicio abilitate. Ce raport Sharpe anual arată în medie cea mai bună dintre ele?",
                "options": [
                    "0",
                    "Circa 0,1",
                    "Circa 2,5",
                    "Circa 0,8"
                ],
                "correctExplanation": "Fiecare raport Sharpe este aproximativ N(0, 1/10); maximul așteptat a 100 de astfel de extrageri este circa 0,8, prin simulare și prin formula Bailey și López de Prado.",
                "incorrectExplanation": "Alegerea celei mai bune dintre multe încercări deplasează raportul Sharpe în sus; cu 100 de încercări pe 10 ani, deplasarea este de circa 0,8, pe care raportul Sharpe deflatat o corectează."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Gu, Kelly and Xiu (2020)",
                "text": "The best neural networks of Gu, Kelly and Xiu (2020) have a monthly out-of-sample R2 of about 0.4%. Why is this considered a strong result?",
                "options": [
                    "Against a zero forecast for thousands of stocks, a small R2 gives decile portfolios with high Sharpe ratios",
                    "Because 0.4% is significant for any sample size",
                    "Because the networks have five layers",
                    "Because the paper uses daily data"
                ],
                "correctExplanation": "Monthly stock returns are mostly noise; sorting stocks by the forecasts gave a value-weighted long-short Sharpe ratio of 1.35 for NN4.",
                "incorrectExplanation": "The value comes from ranking a large cross-section each month; the deeper five-layer network did not beat three or four layers."
            },
            "ro": {
                "title": "Gu, Kelly și Xiu (2020)",
                "text": "Cele mai bune rețele neuronale din Gu, Kelly și Xiu (2020) au un R2 lunar în afara eșantionului de circa 0,4%. De ce este considerat un rezultat puternic?",
                "options": [
                    "Față de o prognoză zero pentru mii de acțiuni, un R2 mic dă portofolii pe decile cu rapoarte Sharpe mari",
                    "Deoarece 0,4% este semnificativ pentru orice mărime a eșantionului",
                    "Deoarece rețelele au cinci straturi",
                    "Deoarece lucrarea folosește date zilnice"
                ],
                "correctExplanation": "Randamentele lunare ale acțiunilor sînt în mare parte zgomot; sortarea acțiunilor după prognoze a dat pentru NN4 un raport Sharpe long-short, ponderat cu valoarea de piață, de 1,35.",
                "incorrectExplanation": "Valoarea vine din ordonarea lunară a unei secțiuni transversale mari; rețeaua mai adîncă, cu cinci straturi, nu a depășit rețelele cu trei sau patru straturi."
            }
        }
    ]
};
