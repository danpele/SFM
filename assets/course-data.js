// ============================================================
// SFM course data (EN + RO). Rendered by assets/site.js.
// Chapter links: { type, href[, colab][, label] } for an existing file,
// { type, soon: true } for an item still in preparation (shown greyed, no link).
// A chapter with no existing file in a language shows "in preparation" only.
// ============================================================
(function () {
    const REPO = 'https://github.com/danpele/SFM';
    const TREE = REPO + '/tree/main/';
    const COLAB = 'https://colab.research.google.com/github/danpele/SFM/blob/main/';

    // Quantinar courses, referenced by key from the chapters
    const Q = {
        sfm: ['Statistics of Financial Markets', 'https://quantinar.com/course/103/statistics-of-financial-markets'],
        tukeyGH: ["Tukey's g- and h- transformations", 'https://quantinar.com/course/144/tukeys-g-and-h-transformations'],
        cryptoEfficiency: ['Efficiency of cryptocurrency markets - A GMM-based analysis', 'https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis'],
        tsaPython: ['Applied Time Series Analysis with Python', 'https://quantinar.com/course/137/applied-time-series-analysis-with-python'],
        statRisk: ['Measuring Statistical Risk', 'https://quantinar.com/course/100080/measuring-statistical-risk'],
        mva: ['MVA Multivariate Statistical Analysis', 'https://quantinar.com/course/540/multivariate-statistical-analysis'],
        mlRisk: ['Machine learning in Financial Risk', 'https://quantinar.com/course/934/machine-learning-in-financial-risk'],
        rf: ['Random Forests', 'https://quantinar.com/course/68/RF'],
        blockchainIntro: ['Introduction to Blockchain and Cryptocurrencies', 'https://quantinar.com/course/134/introduction-to-blockchain-and-cryptocurrencies'],
        cryptoAsset: ['Cryptocurrency as an Asset Class', 'https://quantinar.com/course/55/cryptoasset'],
        frm: ['Financial Risk Meter for Emerging Markets', 'https://quantinar.com/course/52/FRM'],
        cryptoNetworks: ['Dynamic Crypto Networks', 'https://quantinar.com/course/50/cryptonetworks'],
        stable: ['Stable Distribution', 'https://quantinar.com/course/980/stable-distribution']
    };
    const q = (...keys) => keys.map(k => ({ title: Q[k][0], url: Q[k][1] }));

    // helpers for chapter links
    const pdf = (type, href, label) => (label ? { type, href, label } : { type, href });
    const soon = type => ({ type, soon: true });
    const nb = (path, label) => Object.assign({ type: 'notebook', href: REPO + '/blob/main/' + path, colab: COLAB + path }, label ? { label } : {});
    const NB_LECT = { en: 'Lecture notebook', ro: 'Notebook curs' };
    const NB_SEM = { en: 'Seminar notebook', ro: 'Notebook seminar' };
    const ql = path => ({ type: 'quantlets', href: TREE + path });

    window.SFM_DATA = {
        repo: REPO,

        // ---------------------------------------------------------------
        // UI strings
        // ---------------------------------------------------------------
        ui: {
            en: {
                pageTitle: 'Statistics of Financial Markets - Course Website',
                courseTitle: 'Statistics of Financial Markets',
                subtitle: "Bachelor's programme Statistics and Data Science, year 3, semester 2 | Faculty of Cybernetics, Statistics and Economic Informatics | Bucharest University of Economic Studies",
                nav: { home: 'Home', chapters: 'Chapters', project: 'Project', quizzes: 'Quizzes', resources: 'Resources', contact: 'Contact' },
                overview: 'Course Overview',
                objectives: 'Learning Objectives',
                heroTag: 'Returns, distributions, tail risk, market efficiency and volatility, estimated and tested on real market data with Python.',
                heroCta1: 'Explore the chapters',
                heroCta2: 'Team project',
                heroCta3: 'Attendance form',
                qrTeachers: 'For instructors: attendance QR code',
                qrLecture: 'lecture',
                qrSeminar: 'seminar',
                teacherBtn: 'Instructor access',
                teacherPrompt: 'Sign in with the instructor Google account to see the attendance QR code.',
                teacherDenied: 'This account has no instructor access.',
                close: 'Close',
                formulas: 'Key Formulas',
                chapters: 'Course Chapters',
                chapter: 'Chapter',
                chapterShort: 'Ch',
                comingSoon: 'In preparation',
                comingSoonShort: 'in preparation',
                chapterSoon: 'The materials of this chapter are in preparation.',
                selfStudy: 'Self-study',
                quantinar: 'Go deeper on Quantinar',
                links: {
                    slides: 'Lecture Slides', slidesExtra: 'Additional Slides', seminar: 'Seminar', seminarExtra: 'Additional Seminar',
                    notebook: 'Notebook', quantlets: 'Quantlets', colab: 'Open in Colab'
                },
                projectTitle: 'Team Project',
                aiTitle: 'Using AI in this course',
                quizzes: 'Self-Assessment Quizzes',
                quizIntro: 'Each attempt draws up to 20 questions at random from the chapter bank and shuffles the answers. An answer is locked once selected.',
                loginPrompt: 'Sign in with your ASE Google account (@ase.ro or @stud.ase.ro) to take the quizzes.',
                loginRequired: 'Sign in above with your ASE Google account to see this quiz.',
                loginWrongDomain: 'Please use your ASE account (@ase.ro or @stud.ase.ro).',
                saving: 'Saving your score...',
                saved: 'Your score has been recorded.',
                saveExpired: 'Your session has expired: sign out, sign in again and recalculate.',
                saveFailed: 'The score could not be saved. Please try again or tell the instructor.',
                loggedAs: 'Logged in as',
                logout: 'Logout',
                quizSoon: 'The quiz for this chapter will be published together with its materials.',
                question: 'Question',
                correct: 'Correct!',
                incorrect: 'Incorrect.',
                correctIs: 'The correct answer is',
                calc: 'Calculate Score',
                reset: 'New attempt',
                score: 'Score',
                unanswered: 'questions unanswered',
                verdicts: ['Keep practising!', 'Keep studying!', 'Good job!', 'Excellent!'],
                detailed: 'Detailed Results',
                colQ: 'Q', colQuestion: 'Question', colCorrect: 'Correct answer', colYours: 'Your answer', colResult: 'Result',
                resources: 'Resources',
                bibliography: 'Bibliography',
                dataSources: 'Data sources',
                contact: 'Contact',
                instructor: 'Lecturer',
                seminarCard: 'Seminar',
                seminarRole: 'Seminar instructor',
                office: 'Office Hours',
                officeText: 'By appointment',
                footer: 'Statistics of Financial Markets | Faculty of Cybernetics, Statistics and Economic Informatics | Bucharest University of Economic Studies'
            },
            ro: {
                pageTitle: 'Statistica piețelor financiare - Site-ul cursului',
                courseTitle: 'Statistica piețelor financiare',
                subtitle: 'Programul de licență Statistică și Data Science, anul III, semestrul 2 | Facultatea de Cibernetică, Statistică și Informatică Economică | Academia de Studii Economice din București',
                nav: { home: 'Acasă', chapters: 'Capitole', project: 'Proiect', quizzes: 'Quiz-uri', resources: 'Resurse', contact: 'Contact' },
                overview: 'Prezentarea cursului',
                objectives: 'Obiective de învățare',
                heroTag: 'Randamente, distribuții, riscul din cozi, eficiența pieței și volatilitatea, studiate pe date reale de piață, cu estimări și teste în Python.',
                heroCta1: 'Explorați capitolele',
                heroCta2: 'Proiect de echipă',
                heroCta3: 'Formular de prezență',
                qrTeachers: 'Pentru cadre didactice: cod QR de prezență',
                qrLecture: 'curs',
                qrSeminar: 'seminar',
                teacherBtn: 'Acces cadre didactice',
                teacherPrompt: 'Autentificați-vă cu contul Google de cadru didactic pentru a vedea codul QR de prezență.',
                teacherDenied: 'Acest cont nu are acces de cadru didactic.',
                close: 'Închide',
                formulas: 'Formule-cheie',
                chapters: 'Capitolele cursului',
                chapter: 'Capitolul',
                chapterShort: 'Cap.',
                comingSoon: 'În pregătire',
                comingSoonShort: 'în pregătire',
                chapterSoon: 'Materialele acestui capitol sînt în pregătire.',
                selfStudy: 'Studiu individual',
                quantinar: 'Aprofundare pe Quantinar',
                links: {
                    slides: 'Slide-urile cursului', slidesExtra: 'Slide-uri suplimentare', seminar: 'Seminar', seminarExtra: 'Seminar suplimentar',
                    notebook: 'Notebook', quantlets: 'Quantlets', colab: 'Deschide în Colab'
                },
                projectTitle: 'Proiect de echipă',
                aiTitle: 'Utilizarea instrumentelor AI',
                quizzes: 'Quiz-uri de autoevaluare',
                quizIntro: 'La fiecare încercare se extrag aleator cel mult 20 de întrebări din banca de întrebări a capitolului, iar ordinea variantelor de răspuns se schimbă. Un răspuns ales nu mai poate fi modificat.',
                loginPrompt: 'Autentificați-vă cu contul Google ASE (@ase.ro sau @stud.ase.ro) pentru a rezolva quiz-urile.',
                loginRequired: 'Autentificați-vă mai sus cu contul Google ASE pentru a vedea acest quiz.',
                loginWrongDomain: 'Folosiți contul ASE (@ase.ro sau @stud.ase.ro).',
                saving: 'Se salvează scorul...',
                saved: 'Scorul a fost înregistrat.',
                saveExpired: 'Sesiunea a expirat: deconectați-vă, autentificați-vă din nou și recalculați scorul.',
                saveFailed: 'Scorul nu a putut fi salvat. Încercați din nou sau anunțați titularul de curs.',
                loggedAs: 'Autentificat ca',
                logout: 'Deconectare',
                quizSoon: 'Quiz-ul acestui capitol va fi publicat odată cu materialele capitolului.',
                question: 'Întrebarea',
                correct: 'Corect!',
                incorrect: 'Greșit.',
                correctIs: 'Răspunsul corect este',
                calc: 'Calculează scorul',
                reset: 'Încercare nouă',
                score: 'Scor',
                unanswered: 'întrebări fără răspuns',
                verdicts: ['Mai exersați!', 'Mai studiați!', 'Bine!', 'Excelent!'],
                detailed: 'Rezultate detaliate',
                colQ: 'Nr.', colQuestion: 'Întrebare', colCorrect: 'Răspuns corect', colYours: 'Răspunsul ales', colResult: 'Rezultat',
                resources: 'Resurse',
                bibliography: 'Bibliografie',
                dataSources: 'Surse de date',
                contact: 'Contact',
                instructor: 'Titular de curs',
                seminarCard: 'Seminar',
                seminarRole: 'Titular de seminar',
                office: 'Program de consultații',
                officeText: 'Pe bază de programare',
                footer: 'Statistica piețelor financiare | Facultatea de Cibernetică, Statistică și Informatică Economică | Academia de Studii Economice din București'
            }
        },

        // ---------------------------------------------------------------
        // Overview cards and objectives
        // ---------------------------------------------------------------
        overview: {
            en: [
                { h: 'Course', p: ['Statistics of Financial Markets', "Bachelor's programme Statistics and Data Science", 'Year 3, semester 2, academic year 2026/2027'] },
                { h: 'Prerequisites', p: ['Probability and statistics', 'Linear algebra', 'Python programming'] },
                { h: 'Assessment', p: ['Written exam: 70%', 'Team project: 20%', 'Attendance: 10%'] },
                { h: 'Main textbook', p: ['Franke, Härdle &amp; Hafner, <a href="https://doi.org/10.1007/978-3-030-13751-9" target="_blank" rel="noopener"><em>Statistics of Financial Markets</em></a>, 5th ed., Springer, 2019', 'With the companion volume <a href="https://doi.org/10.1007/978-3-642-33929-5" target="_blank" rel="noopener"><em>Exercises and Solutions</em></a>'] },
                { h: 'Tools', p: ['Python, Jupyter / Google Colab', 'GitHub, Quantlet, Quantinar'] }
            ],
            ro: [
                { h: 'Curs', p: ['Statistica piețelor financiare', 'Programul de licență Statistică și Data Science', 'Anul III, semestrul 2, anul universitar 2026/2027'] },
                { h: 'Cunoștințe prealabile', p: ['Probabilități și statistică', 'Algebră liniară', 'Programare în Python'] },
                { h: 'Evaluare', p: ['Examen scris: 70%', 'Proiect de echipă: 20%', 'Prezență: 10%'] },
                { h: 'Manual de bază', p: ['Franke, Härdle și Hafner, <a href="https://doi.org/10.1007/978-3-030-13751-9" target="_blank" rel="noopener"><em>Statistics of Financial Markets</em></a>, ediția a 5-a, Springer, 2019', 'Cu volumul însoțitor <a href="https://doi.org/10.1007/978-3-642-33929-5" target="_blank" rel="noopener"><em>Exercises and Solutions</em></a>'] },
                { h: 'Instrumente', p: ['Python, Jupyter / Google Colab', 'GitHub, Quantlet, Quantinar'] }
            ]
        },

        objectives: {
            en: [
                'Compute returns and performance indicators (volatility, Sharpe ratio, drawdown) from market prices and interpret them correctly',
                'Describe the stylised facts of returns and fit classical, Student-t and α-stable distributions to them',
                'Measure tail risk with extreme value theory and choose between competing models with information criteria and goodness-of-fit tests',
                'Test market efficiency (random walk, autocorrelation, variance-ratio tests) and long memory (Hurst exponent)',
                'Estimate volatility with range-based estimators and GARCH models, and compute and backtest VaR and Expected Shortfall',
                'Deliver a reproducible analysis in Python and GitHub, documented as Quantlets'
            ],
            ro: [
                'Calculul randamentelor și al indicatorilor de performanță (volatilitate, raportul Sharpe, drawdown) pe baza prețurilor de piață și interpretarea lor corectă',
                'Descrierea faptelor stilizate ale randamentelor și ajustarea pe date a distribuției Normale, a distribuției Student-t și a distribuțiilor α-stabile',
                'Măsurarea riscului din cozi cu teoria valorilor extreme și alegerea între modele concurente pe baza criteriilor informaționale și a testelor de concordanță',
                'Testarea eficienței pieței (mers aleator, autocorelare, teste variance ratio) și a memoriei lungi (exponentul Hurst)',
                'Estimarea volatilității cu estimatori range-based și cu modele GARCH; calculul VaR și al Expected Shortfall și backtesting-ul acestora',
                'Elaborarea unei analize reproductibile în Python, publicate pe GitHub și documentate sub formă de Quantlets'
            ]
        },

        // ---------------------------------------------------------------
        // Chapters 0-16. `id` is the stable key (quizzes, anchors, tabs);
        // `num` is only the display order. RO page -> RO files, EN page -> EN files.
        // ---------------------------------------------------------------
        chapters: [
            {
                id: 'intro', num: 0,
                title: { en: 'Introduction', ro: 'Introducere' },
                topics: {
                    en: ['What statistics of financial markets studies, and why financial data need their own methods', 'Markets, prices and the data used in the course', 'Organisation, assessment and tools'],
                    ro: ['Ce studiază statistica piețelor financiare și de ce datele financiare cer metode proprii', 'Piețe, prețuri și datele folosite în curs', 'Organizare, evaluare și instrumente']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter0_introduction.pdf'), pdf('seminar', 'EN/Seminars/seminar0_introduction.pdf'),
                         nb('notebooks/EN/chapter0_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter0_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_00')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol0_introducere.pdf'), pdf('seminar', 'RO/Seminarii/seminar0_introducere_ro.pdf'),
                         nb('notebooks/EN/chapter0_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter0_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_00')]
                },
                quantinar: q('sfm')
            },
            {
                id: 'returns', num: 1,
                title: { en: 'Data, returns and indicators', ro: 'Date, randamente și indicatori' },
                topics: {
                    en: ['Data sources and data quality: adjusted prices, total return indices, checking a series', 'Simple and log returns, multi-period and portfolio returns, annualisation', 'Descriptive statistics and performance indicators: CAGR, Sharpe, Sortino, maximum drawdown, Calmar, volatility drag'],
                    ro: ['Surse de date și calitatea datelor: prețuri ajustate, indici de randament total, verificarea unei serii', 'Randamente simple și logaritmice, randamente pe mai multe perioade, randamentul unui portofoliu, anualizarea', 'Statistici descriptive și indicatori de performanță: CAGR, Sharpe, Sortino, drawdown maxim, Calmar, volatility drag']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter1_data_returns_indicators.pdf'), pdf('seminar', 'EN/Seminars/seminar1_data_returns_indicators.pdf'),
                         nb('notebooks/EN/chapter1_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter1_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_01')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol1_date_randamente_indicatori.pdf'), pdf('seminar', 'RO/Seminarii/seminar1_date_randamente_indicatori_ro.pdf'),
                         nb('notebooks/EN/chapter1_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter1_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_01')]
                },
                quantinar: q('sfm')
            },
            {
                id: 'distributions', num: 2,
                title: { en: 'Classical distributions and stylised facts', ro: 'Distribuții clasice și fapte stilizate' },
                topics: {
                    en: ['The Normal and lognormal distributions, the central limit theorem and the Normal approximation', 'Moments, the Jarque–Bera test, QQ plots and the Student-t distribution', 'Stylised facts of returns (Cont, 2001) on real data: heavy tails, aggregational Gaussianity, volatility clustering, leverage effect, gain/loss asymmetry'],
                    ro: ['Distribuția Normală și distribuția lognormală, teorema limită centrală și aproximarea prin distribuția Normală', 'Momente, testul Jarque–Bera, QQ plots și distribuția Student-t', 'Fapte stilizate ale randamentelor (Cont, 2001) pe date reale: cozi groase, aggregational Gaussianity, volatility clustering, leverage effect, asimetria cîștig/pierdere']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter2_classical_distributions_stylised_facts.pdf'), pdf('seminar', 'EN/Seminars/seminar2_classical_distributions_stylised_facts.pdf'),
                         nb('notebooks/EN/chapter2_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter2_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_02')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol2_distributii_clasice_fapte_stilizate.pdf'), pdf('seminar', 'RO/Seminarii/seminar2_distributii_clasice_fapte_stilizate_ro.pdf'),
                         nb('notebooks/EN/chapter2_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter2_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_02')]
                },
                quantinar: q('tukeyGH')
            },
            {
                id: 'stable', num: 3,
                title: { en: 'α-stable distributions', ro: 'Distribuții α-stabile' },
                topics: {
                    en: ['Stability under summation and the generalised central limit theorem', 'Parameters α, β, γ, δ, the S0/S1 parameterisations, power-law tails and infinite variance', 'Simulation (Chambers–Mallows–Stuck), estimation (McCulloch, maximum likelihood) and fits to BET, S&P 500, DAX and Bitcoin; the critique (Mandelbrot, 1963; Nolan, 2020)'],
                    ro: ['Stabilitatea la adunare și teorema limită centrală generalizată', 'Parametrii α, β, γ, δ, parametrizările S0/S1, cozile de tip lege de putere și varianța infinită', 'Simulare (Chambers–Mallows–Stuck), estimare (McCulloch, verosimilitate maximă) și ajustarea pe BET, S&P 500, DAX și Bitcoin; critica (Mandelbrot, 1963; Nolan, 2020)']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter3_alpha_stable_distributions.pdf'), pdf('seminar', 'EN/Seminars/seminar3_alpha_stable_distributions.pdf'),
                         nb('notebooks/EN/chapter3_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter3_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_03')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol3_distributii_alfa_stabile.pdf'), pdf('seminar', 'RO/Seminarii/seminar3_distributii_alfa_stabile_ro.pdf'),
                         nb('notebooks/EN/chapter3_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter3_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_03')]
                },
                quantinar: q('stable')
            },
            {
                id: 'probability', num: 4,
                title: { en: 'Probability', ro: 'Probabilitate' },
                topics: {
                    en: ['Random variables, distributions, moments, covariance and correlation (SFM, ch. 3)', 'Independence vs uncorrelatedness, conditional expectation and conditional variance', 'Random numbers and Monte Carlo; binomial model, random walk, martingale, AR(1), Wiener process and GBM against real data'],
                    ro: ['Variabile aleatoare, distribuții, momente, covarianță și corelație (SFM, cap. 3)', 'Independență și necorelare, media condiționată și varianța condiționată', 'Numere aleatoare și metoda Monte Carlo; modelul binomial, mersul aleator, martingala, procesul AR(1), procesul Wiener și mișcarea browniană geometrică (GBM), comparate cu datele reale']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter4_probability.pdf'), pdf('seminar', 'EN/Seminars/seminar4_probability.pdf'),
                         nb('notebooks/EN/chapter4_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter4_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_04')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol4_probabilitate.pdf'), pdf('seminar', 'RO/Seminarii/seminar4_probabilitate_ro.pdf'),
                         nb('notebooks/EN/chapter4_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter4_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_04')]
                },
                quantinar: q('sfm')
            },
            {
                id: 'evt', num: 5,
                title: { en: 'Heavy tails and extreme value theory', ro: 'Cozi groase și teoria valorilor extreme' },
                topics: {
                    en: ['Why tails matter: the crashes of 1987, 2008 and 2020; heavy and light tails, regular variation and the tail index', 'The Hill estimator, the Hill plot and its standard errors; the mean excess function', 'Block maxima and the GEV, peaks over threshold and the GPD, return levels; VaR 1%, ES 2.5% and VaR 0.1% by EVT on the S&P 500, DAX, BET, Bitcoin and BVB stocks'],
                    ro: ['De ce contează cozile: crahurile din 1987, 2008 și 2020; cozi groase și cozi subțiri, variația regulată și indicele de coadă', 'Estimatorul Hill, Hill plot și erorile standard ale estimatorului; funcția mean excess', 'Block maxima și distribuția GEV, peaks over threshold și distribuția GPD, return levels; VaR 1%, ES 2,5% și VaR 0,1% prin EVT pe S&P 500, DAX, BET, Bitcoin și acțiuni BVB']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter5_heavy_tails_evt.pdf'), pdf('seminar', 'EN/Seminars/seminar5_heavy_tails_evt.pdf'),
                         nb('notebooks/EN/chapter5_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter5_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_05')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol5_cozi_groase_evt.pdf'), pdf('seminar', 'RO/Seminarii/seminar5_cozi_groase_evt_ro.pdf'),
                         nb('notebooks/EN/chapter5_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter5_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_05')]
                },
                quantinar: q('statRisk')
            },
            {
                id: 'model-selection', num: 6,
                title: { en: 'Model selection and risk management', ro: 'Selecția modelului și managementul riscului' },
                topics: {
                    en: ['Seven candidate distributions fitted by maximum likelihood; likelihood-ratio and Vuong tests, AIC, BIC and Akaike weights', 'Goodness of fit: Kolmogorov–Smirnov (Lilliefors), Cramér–von Mises, Anderson–Darling, PP and QQ plots', 'Which model estimates VaR 1% correctly: exceedances, out-of-sample validation, overfitting and model uncertainty'],
                    ro: ['Șapte distribuții candidate estimate prin verosimilitate maximă; testul raportului de verosimilitate, testul Vuong, criteriile AIC și BIC, ponderile Akaike', 'Calitatea ajustării: testele Kolmogorov–Smirnov (Lilliefors), Cramér–von Mises și Anderson–Darling, grafice PP și QQ', 'Alegerea modelului care estimează corect VaR 1%: depășiri, validare în afara eșantionului, supraajustare și incertitudinea modelului']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter6_model_selection_risk_management.pdf'), pdf('seminar', 'EN/Seminars/seminar6_model_selection_risk_management.pdf'),
                         nb('notebooks/EN/chapter6_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter6_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_06')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol6_selectia_modelului_managementul_riscului.pdf'), pdf('seminar', 'RO/Seminarii/seminar6_selectia_modelului_managementul_riscului_ro.pdf'),
                         nb('notebooks/EN/chapter6_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter6_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_06')]
                },
                quantinar: q('statRisk')
            },
            {
                id: 'emh', num: 7,
                title: { en: 'Efficient markets, random walk and variance-ratio tests', ro: 'Ipoteza piețelor eficiente, mersul aleator și testele VR' },
                topics: {
                    en: ['Efficient markets: the three forms (Fama, 1970), the joint-hypothesis problem, random walks RW1–RW3 and the martingale', 'White noise and the ACF; Ljung–Box, robust portmanteau and runs tests; unit roots: ADF, Phillips–Perron and KPSS on prices and returns', 'Variance-ratio tests (Lo–MacKinlay, Chow–Denning, automatic VR); efficiency across markets and over time (adaptive markets); calendar anomalies'],
                    ro: ['Piețele eficiente: cele trei forme (Fama, 1970), problema ipotezei comune, mersul aleator RW1–RW3 și martingala', 'Zgomotul alb și funcția de autocorelație; testele Ljung–Box, portmanteau robust și testul secvențelor; rădăcini unitare: testele ADF, Phillips–Perron și KPSS pe prețuri și randamente', 'Testele variance ratio (Lo–MacKinlay, Chow–Denning, VR automat); eficiența pe piețe diferite și în timp (piețe adaptive); anomalii calendaristice']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter7_efficient_markets_random_walk_vr.pdf'), pdf('seminar', 'EN/Seminars/seminar7_efficient_markets_random_walk_vr.pdf'),
                         nb('notebooks/EN/chapter7_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter7_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_07')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol7_piete_eficiente_mers_aleator_vr.pdf'), pdf('seminar', 'RO/Seminarii/seminar7_piete_eficiente_mers_aleator_vr_ro.pdf'),
                         nb('notebooks/EN/chapter7_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter7_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_07')]
                },
                quantinar: q('cryptoEfficiency')
            },
            {
                id: 'vol-estimators', num: 8,
                title: { en: 'Volatility estimators', ro: 'Estimatori de volatilitate' },
                topics: {
                    en: ['Volatility as a latent quantity; historical windows, EWMA (RiskMetrics) and the ghost effect', 'Range-based estimators: Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang; realised volatility', 'Volatility clustering (ACF of squared returns, Ljung–Box, ARCH-LM); VIX, leverage effect, forecast evaluation with QLIKE'],
                    ro: ['Volatilitatea ca mărime latentă; ferestre istorice, EWMA (RiskMetrics) și ghost effect', 'Estimatori pe baza amplitudinii zilnice: Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang; volatilitatea realizată', 'Volatility clustering (ACF al randamentelor la pătrat, Ljung–Box, ARCH-LM); VIX, leverage effect, evaluarea prognozelor cu QLIKE']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter8_volatility_estimators.pdf'), pdf('seminar', 'EN/Seminars/seminar8_volatility_estimators.pdf'),
                         nb('notebooks/EN/chapter8_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter8_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_08')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol8_estimatori_volatilitate.pdf'), pdf('seminar', 'RO/Seminarii/seminar8_estimatori_volatilitate_ro.pdf'),
                         nb('notebooks/EN/chapter8_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter8_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_08')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'garch', num: 9,
                title: { en: 'ARCH and GARCH models', ro: 'Modele ARCH și GARCH' },
                topics: {
                    en: ['ARCH(q) and GARCH(1,1): persistence, long-run variance, half-life; IGARCH and the EWMA link', 'Maximum-likelihood estimation (step by step and with the arch package), robust standard errors, Student-t and skewed-t innovations on six markets', 'Asymmetry (GJR-GARCH, EGARCH, news impact curve), diagnostics, AIC/BIC, volatility forecasts (QLIKE against EWMA) and VaR 1%'],
                    ro: ['ARCH(q) și GARCH(1,1): persistența, varianța pe termen lung, timpul de înjumătățire; IGARCH și legătura cu EWMA', 'Estimarea prin verosimilitate maximă (pas cu pas și cu pachetul arch), erori standard robuste, inovații Student-t și skewed-t pe șase piețe', 'Asimetrie (GJR-GARCH, EGARCH, curba de impact a știrilor), diagnosticare, AIC/BIC, prognoza volatilității (QLIKE comparat cu EWMA) și VaR 1%']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter9_arch_garch_models.pdf'), pdf('seminar', 'EN/Seminars/seminar9_arch_garch_models.pdf'),
                         nb('notebooks/EN/chapter9_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter9_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_09')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol9_modele_arch_garch.pdf'), pdf('seminar', 'RO/Seminarii/seminar9_modele_arch_garch_ro.pdf'),
                         nb('notebooks/EN/chapter9_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter9_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_09')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'var-es', num: 10,
                title: { en: 'VaR, ES and backtesting', ro: 'VaR, ES și backtesting' },
                topics: {
                    en: ['VaR 1% and ES 2.5%: definitions, coherence and subadditivity, elicitability; historical simulation, Normal, Student-t, Cornish–Fisher and EVT on six markets', 'Conditional VaR (GARCH-t, filtered historical simulation) through 2008 and 2020; portfolio VaR, tail dependence and Gaussian and t copulas', 'Backtesting: Kupiec, Christoffersen, the Basel traffic light, the Acerbi–Székely test of ES'],
                    ro: ['VaR 1% și ES 2,5%: definiții, coerență și subaditivitate, elicitabilitate; simulare istorică, distribuția Normală, Student-t, Cornish–Fisher și EVT pe șase piețe', 'VaR condiționat (GARCH-t, simulare istorică filtrată) în 2008 și 2020; VaR de portofoliu, dependența în cozi, copulele Gaussiană și t', 'Backtesting: testele Kupiec și Christoffersen, semaforul Basel, testul Acerbi–Székely pentru ES']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter10_var_es_backtesting.pdf'), pdf('seminar', 'EN/Seminars/seminar10_var_es_backtesting.pdf'),
                         nb('notebooks/EN/chapter10_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter10_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_10')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol10_var_es_backtesting.pdf'), pdf('seminar', 'RO/Seminarii/seminar10_var_es_backtesting_ro.pdf'),
                         nb('notebooks/EN/chapter10_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter10_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_10')]
                },
                quantinar: q('statRisk')
            },
            {
                id: 'fmh', num: 11,
                title: { en: 'Fractal markets hypothesis and long memory', ro: 'Ipoteza piețelor fractale și memoria lungă' },
                topics: {
                    en: ['The fractal markets hypothesis versus the EMH', 'Hurst exponent: R/S analysis and DFA', 'Long memory, fractional Brownian motion and ARFIMA'],
                    ro: ['Ipoteza piețelor fractale față de EMH', 'Exponentul Hurst: analiza R/S și DFA', 'Memoria lungă, mișcarea browniană fracționară și modelele ARFIMA']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter14_fractal_market_hypothesis.pdf'), soon('seminar'),
                         nb('Quantlets/SFM_ch_fmh/SFM_ch_fmh_fractals/SFM_ch_fmh_fractals.ipynb'), ql('Quantlets/SFM_ch_fmh')],
                    ro: [pdf('slides', 'RO/Courses/20260310_chapter10_fmh_ro_v2.pdf'), pdf('seminar', 'RO/Seminars/seminar_fmh_ro.pdf'),
                         nb('Quantlets/SFM_ch_fmh/SFM_ch_fmh_fractals/SFM_ch_fmh_fractals.ipynb'), ql('Quantlets/SFM_ch_fmh')]
                }
            },
            {
                id: 'scoring', num: 12,
                title: { en: 'Scoring models', ro: 'Modele de scoring' },
                topics: {
                    en: ['Probability of default: logit and probit (SFM, ch. 21)', 'Discriminant analysis', 'Validation: ROC curve, AUC, Gini coefficient'],
                    ro: ['Probabilitatea de nerambursare: logit și probit (SFM, cap. 21)', 'Analiza discriminantă', 'Validare: curba ROC, AUC, coeficientul Gini']
                },
                links: { en: [], ro: [] },
                quantinar: q('mva')
            },
            {
                id: 'ml', num: 13,
                title: { en: 'Machine learning', ro: 'Învățare automată' },
                topics: {
                    en: ['Neural networks for financial time series (SFM, ch. 19)', 'Trees and random forests', 'Validation without look-ahead bias'],
                    ro: ['Rețele neuronale pentru serii de timp financiare (SFM, cap. 19)', 'Arbori de decizie și păduri aleatoare (random forests)', 'Validare fără look-ahead bias']
                },
                links: { en: [], ro: [] },
                quantinar: q('mlRisk', 'rf')
            },
            {
                id: 'crypto', num: 14,
                title: { en: 'Crypto assets', ro: 'Active cripto' },
                topics: {
                    en: ['Bitcoin, Ethereum and stablecoins: how they work (SFM, ch. 23)', 'Statistical properties of crypto returns', 'Crypto indices: CRIX'],
                    ro: ['Bitcoin, Ethereum și stablecoins: cum funcționează (SFM, cap. 23)', 'Proprietățile statistice ale randamentelor cripto', 'Indici cripto: CRIX']
                },
                links: { en: [], ro: [] },
                quantinar: q('blockchainIntro', 'cryptoAsset')
            },
            {
                id: 'systemic', num: 15,
                title: { en: 'Systemic risk', ro: 'Risc sistemic' },
                topics: {
                    en: ['Contagion and interconnectedness', 'CoVaR and ΔCoVaR', 'Network measures of systemic risk'],
                    ro: ['Contagiune și interconectare', 'CoVaR și ΔCoVaR', 'Măsuri de rețea pentru riscul sistemic']
                },
                links: { en: [], ro: [] },
                quantinar: q('frm', 'cryptoNetworks')
            },
            {
                id: 'review', num: 16,
                title: { en: 'Review', ro: 'Recapitulare' },
                topics: {
                    en: ['Solved problems across all chapters', 'Exam-style problems with step-by-step solutions'],
                    ro: ['Probleme rezolvate din toate capitolele', 'Probleme de tip examen, rezolvate pas cu pas']
                },
                links: {
                    en: [soon('seminar'), ql('Quantlets/Sem_recap')],
                    ro: [pdf('seminar', 'RO/Seminars/seminar_probleme_examen_ro.pdf', { en: 'Exam problems', ro: 'Probleme de examen' }),
                         pdf('seminarExtra', 'RO/Seminars/seminar_exercitii.pdf', { en: 'Review exercises', ro: 'Exerciții de recapitulare' }),
                         ql('Quantlets/Sem_recap')]
                }
            }
        ],

        // ---------------------------------------------------------------
        // Team project (section #project); aiPolicy left empty until the policy is set
        // ---------------------------------------------------------------
        project: {
            en: [
                { h: 'What', p: ['A team analysis of real market data with the methods of the course: returns and indicators, distributions and tails, efficiency tests, volatility and risk measures.', 'The project counts for 20% of the final grade.'] },
                { h: 'Deliverables', p: ['A GitHub repository whose code reproduces every number and chart from the saved data.', 'A short report and a presentation of the results.'] },
                { h: 'What we grade', p: ['A clear question, correct methods and checks, and the interpretation of the results.', 'Each member must be able to explain the code and the results.'] }
            ],
            ro: [
                { h: 'Conținut', p: ['O analiză în echipă a unor date reale de piață cu metodele cursului: randamente și indicatori, distribuții și cozi, teste de eficiență, volatilitate și măsuri de risc.', 'Proiectul reprezintă 20% din nota finală.'] },
                { h: 'Livrabile', p: ['Un repository GitHub cu codul care reproduce, din datele salvate, fiecare rezultat numeric și fiecare grafic.', 'Un raport scurt și o prezentare a rezultatelor.'] },
                { h: 'Criterii de evaluare', p: ['Claritatea întrebării de cercetare, corectitudinea metodelor și a verificărilor, interpretarea rezultatelor.', 'Fiecare membru trebuie să poată explica codul și rezultatele.'] }
            ]
        },
        aiPolicy: {
            en: ['AI tools are allowed and must be declared in AI_USE.md (tool, prompts, what was kept)', 'Every number and every reference produced with AI is checked by the team', 'The oral defence of the project checks that each member understands the code and the results'],
            ro: ['Instrumentele AI sînt permise și se declară în AI_USE.md (instrument, prompturi, ce s-a păstrat)', 'Fiecare rezultat numeric și fiecare referință obținute cu AI sînt verificate de echipă', 'Susținerea orală a proiectului verifică dacă fiecare membru înțelege codul și rezultatele']
        },

        // ---------------------------------------------------------------
        // Resources
        // ---------------------------------------------------------------
        resources: [
            { icon: '&#128187;', href: REPO, en: ['GitHub Repository', 'Slides, seminars, notebooks and Quantlets'], ro: ['Repository GitHub', 'Slide-uri, seminarii, notebook-uri și Quantlets'] },
            { icon: '&#127891;', img: 'logos/qr_logo.png', href: 'https://quantinar.com', en: ['Quantinar', 'P2P platform with advanced courses'], ro: ['Quantinar', 'Platformă P2P cu cursuri avansate'] },
            { icon: '&#128190;', img: 'logos/ql_logo.png', href: 'https://quantlet.com', en: ['Quantlet', 'Reproducible code for every chart'], ro: ['Quantlet', 'Cod reproductibil pentru fiecare grafic'] },
            { icon: '&#128218;', href: 'https://github.com/QuantLet/SFE', en: ['SFE Quantlets', 'Code accompanying the Franke–Härdle–Hafner textbook'], ro: ['SFE Quantlets', 'Codul care însoțește manualul Franke–Härdle–Hafner'] },
            { icon: '&#127963;', img: 'logos/ida_square.png', href: 'https://theida.net', en: ['IDA', 'Institute for Digital Assets'], ro: ['IDA', 'Institute for Digital Assets'] }
        ],

        dataSources: [
            { name: 'BVB', href: 'https://www.bvb.ro', en: 'Bucharest Stock Exchange', ro: 'Bursa de Valori București' },
            { name: 'FRED', href: 'https://fred.stlouisfed.org', en: 'Macro and interest-rate data', ro: 'Date macroeconomice și de dobîndă' },
            { name: 'ECB Data Portal', href: 'https://data.ecb.europa.eu', en: 'Exchange rates and euro-area data', ro: 'Cursuri de schimb și date pentru zona euro' }
        ],

        bibliography: [
            'Franke, J., Härdle, W. K., &amp; Hafner, C. M. (2019). <a href="https://doi.org/10.1007/978-3-030-13751-9" target="_blank" rel="noopener"><em>Statistics of Financial Markets: An Introduction</em></a> (5th ed.). Springer.',
            'Borak, S., Härdle, W. K., &amp; López-Cabrera, B. (2013). <a href="https://doi.org/10.1007/978-3-642-33929-5" target="_blank" rel="noopener"><em>Statistics of Financial Markets: Exercises and Solutions</em></a> (2nd ed.). Springer.',
            'Tsay, R. S. (2010). <a href="https://doi.org/10.1002/9780470644560" target="_blank" rel="noopener"><em>Analysis of Financial Time Series</em></a> (3rd ed.). Wiley.',
            'Campbell, J. Y., Lo, A. W., &amp; MacKinlay, A. C. (1997). <a href="https://doi.org/10.1515/9781400830213" target="_blank" rel="noopener"><em>The Econometrics of Financial Markets</em></a>. Princeton University Press.',
            'McNeil, A. J., Frey, R., &amp; Embrechts, P. (2015). <a href="https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management" target="_blank" rel="noopener"><em>Quantitative Risk Management</em></a> (revised ed.). Princeton University Press.',
            'Nolan, J. P. (2020). <a href="https://doi.org/10.1007/978-3-030-52915-4" target="_blank" rel="noopener"><em>Univariate Stable Distributions: Models for Heavy Tailed Data</em></a>. Springer.',
            'Härdle, W. K., &amp; Simar, L. (2019). <a href="https://doi.org/10.1007/978-3-030-26006-4" target="_blank" rel="noopener"><em>Applied Multivariate Statistical Analysis</em></a> (5th ed.). Springer.',
            'Cont, R. (2001). <a href="https://doi.org/10.1080/713665670" target="_blank" rel="noopener">Empirical properties of asset returns: stylized facts and statistical issues</a>. <em>Quantitative Finance</em>, 1(2), 223–236.',
            'Mandelbrot, B. (1963). <a href="https://doi.org/10.1086/294632" target="_blank" rel="noopener">The variation of certain speculative prices</a>. <em>The Journal of Business</em>, 36(4), 394–419.',
            'Lo, A. W., &amp; MacKinlay, A. C. (1988). <a href="https://doi.org/10.1093/rfs/1.1.41" target="_blank" rel="noopener">Stock market prices do not follow random walks: evidence from a simple specification test</a>. <em>The Review of Financial Studies</em>, 1(1), 41–66.'
        ],

        contact: {
            name: 'Prof. dr. Daniel Traian Pele',
            email: 'danpele@ase.ro',
            en: ['Bucharest University of Economic Studies', 'Department of Statistics and Econometrics', 'Faculty of Cybernetics, Statistics and Economic Informatics'],
            ro: ['Academia de Studii Economice din București', 'Departamentul de Statistică și Econometrie', 'Facultatea de Cibernetică, Statistică și Informatică Economică'],
            // TODO: seminar instructor not yet confirmed. Fill in name and e-mail;
            // the Seminar card stays hidden while the name starts with 'TODO'.
            seminar: { name: 'TODO_SEMINAR_INSTRUCTOR', email: '' }
        },

        footerLogos: [
            ['https://www.ase.ro', 'logos/ase_logo.png', 'ASE'],
            ['https://www.theida.net/', 'logos/ida_logo.png', 'IDA'],
            ['https://quantinar.com', 'logos/qr_logo.png', 'Quantinar'],
            ['https://quantlet.com', 'logos/ql_logo.png', 'Quantlet'],
            ['https://ai4efin.ase.ro', 'logos/ai4efin_logo.png', 'AI4EFin'],
            ['https://www.digital-finance-msca.com/', 'logos/msca_logo.png', 'MSCA Digital Finance'],
            ['https://blockchain-research-center.com/', 'logos/brc_logo.png', 'Blockchain Research Center'],
            ['https://ipe.ro/new/', 'logos/acad_logo.png', 'Romanian Academy']
        ],

        // Quiz banks register themselves here by chapter id (see assets/quizzes/<id>.js)
        quizzes: {}
    };
})();
