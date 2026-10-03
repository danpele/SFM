// ============================================================
// Chapter 15 quiz bank: systemic risk (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.SFM_DATA.quizzes['systemic'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Systemic risk",
                "text": "Which statement describes systemic risk?",
                "options": [
                    "The risk that one bank's share price falls",
                    "The risk that the financial system as a whole stops working, with large costs for the real economy",
                    "The risk of a single loan default",
                    "The volatility of a bank's daily returns"
                ],
                "correctExplanation": "Systemic risk is a property of the system: lending, payments and trading break down together, not just one balance sheet.",
                "incorrectExplanation": "A single share price, a single loan or one bank's volatility concern one institution; systemic risk concerns the functioning of the whole system."
            },
            "ro": {
                "title": "Riscul sistemic",
                "text": "Ce afirmație descrie riscul sistemic?",
                "options": [
                    "Riscul ca prețul acțiunii unei bănci să scadă",
                    "Riscul ca sistemul financiar în ansamblu să nu mai funcționeze, cu costuri mari pentru economia reală",
                    "Riscul de nerambursare al unui singur credit",
                    "Volatilitatea randamentelor zilnice ale unei bănci"
                ],
                "correctExplanation": "Riscul sistemic este o proprietate a sistemului: creditarea, plățile și tranzacționarea se blochează împreună, nu doar un bilanț.",
                "incorrectExplanation": "Prețul unei acțiuni, un credit sau volatilitatea unei bănci privesc o singură instituție; riscul sistemic privește funcționarea întregului sistem."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Fallacy of composition",
                "text": "Every bank sells risky assets at the same time to cut its own VaR. What can happen?",
                "options": [
                    "The risk of every bank falls",
                    "Nothing, because VaR is subadditive",
                    "Only the smallest bank is affected",
                    "Prices fall for all holders and the risk of the whole system rises"
                ],
                "correctExplanation": "Selling that is prudent for one bank pushes prices down for everyone: what is safe individually can be dangerous collectively.",
                "incorrectExplanation": "Joint selling moves prices; the losses spread to every holder of the assets, so the risk of all banks can rise, whatever the size of the bank."
            },
            "ro": {
                "title": "Eroarea de compoziție",
                "text": "Toate băncile vînd simultan active riscante ca să-și reducă propriul VaR. Ce se poate întîmpla?",
                "options": [
                    "Riscul fiecărei bănci scade",
                    "Nimic, deoarece VaR este subaditiv",
                    "Este afectată doar banca cea mai mică",
                    "Prețurile scad pentru toți deținătorii, iar riscul întregului sistem crește"
                ],
                "correctExplanation": "Vînzarea prudentă pentru o bancă împinge prețurile în jos pentru toată lumea: ce este sigur individual poate fi periculos colectiv.",
                "incorrectExplanation": "Vînzările simultane mișcă prețurile; pierderile ajung la toți deținătorii activelor, deci riscul tuturor băncilor poate crește, indiferent de mărimea băncii."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Fire sales",
                "text": "What is a fire sale?",
                "options": [
                    "A forced sale below fundamental value, which lowers the marked-to-market value of the same asset at other holders",
                    "A sale of bank shares by the central bank",
                    "A sale of deposits to another bank",
                    "A sale that is always profitable for the seller"
                ],
                "correctExplanation": "A seller who must sell fast accepts a low price; every other holder that values the asset at market prices records a loss.",
                "incorrectExplanation": "Fire sales are forced sales of assets by investors under pressure; they are costly for the seller and spread losses to other holders."
            },
            "ro": {
                "title": "Vînzări forțate",
                "text": "Ce este o vînzare forțată (fire sale)?",
                "options": [
                    "O vînzare sub valoarea fundamentală, care reduce valoarea de piață a aceluiași activ la ceilalți deținători",
                    "O vînzare de acțiuni bancare de către banca centrală",
                    "O vînzare de depozite către o altă bancă",
                    "O vînzare întotdeauna profitabilă pentru vînzător"
                ],
                "correctExplanation": "Un vînzător care trebuie să vîndă repede acceptă un preț mic; orice alt deținător care evaluează activul la prețul pieței înregistrează o pierdere.",
                "incorrectExplanation": "Vînzările forțate sînt vînzări de active ale unor investitori aflați sub presiune; sînt costisitoare pentru vînzător și propagă pierderile către alți deținători."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Bank runs",
                "text": "Why can a run hit even a solvent bank?",
                "options": [
                    "Because its loans are all bad",
                    "Because deposits are insured",
                    "Because it funds long, illiquid loans with short deposits and cannot pay all depositors at once",
                    "Because its share price is too high"
                ],
                "correctExplanation": "In the Diamond and Dybvig model, the fear that others withdraw first is enough: the bank cannot turn illiquid loans into cash fast.",
                "incorrectExplanation": "A run does not require bad loans; deposit insurance removes the reason to run; the share price is not the mechanism."
            },
            "ro": {
                "title": "Retragerea masivă a depozitelor",
                "text": "De ce poate o retragere masivă a depozitelor să lovească o bancă solvabilă?",
                "options": [
                    "Pentru că toate creditele ei sînt neperformante",
                    "Pentru că depozitele sînt garantate",
                    "Pentru că finanțează credite pe termen lung, nelichide, cu depozite pe termen scurt și nu poate plăti tuturor deponenților simultan",
                    "Pentru că prețul acțiunii ei este prea mare"
                ],
                "correctExplanation": "În modelul Diamond și Dybvig este suficientă teama că alții vor retrage primii: banca nu poate transforma rapid creditele nelichide în numerar.",
                "incorrectExplanation": "O retragere masivă nu cere credite neperformante; garantarea depozitelor elimină motivul retragerii; prețul acțiunii nu este mecanismul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Too big to fail",
                "text": "Why is 'too big to fail' a problem?",
                "options": [
                    "Large banks always have lower capital",
                    "The expected rescue lowers the bank's funding cost and encourages more risk-taking",
                    "Large banks cannot be listed",
                    "Large banks never fail"
                ],
                "correctExplanation": "An implicit state guarantee is a subsidy and creates moral hazard: losses are partly borne by taxpayers.",
                "incorrectExplanation": "The issue is not listing or capital by definition; large banks can fail, but the expectation of a rescue changes their incentives."
            },
            "ro": {
                "title": "Too big to fail",
                "text": "De ce este „too big to fail” o problemă?",
                "options": [
                    "Băncile mari au întotdeauna mai puțin capital",
                    "Salvarea așteptată reduce costul de finanțare al băncii și încurajează asumarea de riscuri mai mari",
                    "Băncile mari nu pot fi listate",
                    "Băncile mari nu falimentează niciodată"
                ],
                "correctExplanation": "O garanție implicită a statului este o subvenție și produce hazard moral: pierderile sînt suportate parțial de contribuabili.",
                "incorrectExplanation": "Problema nu ține de listare sau de capital prin definiție; băncile mari pot falimenta, dar așteptarea salvării le schimbă stimulentele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "CoVaR",
                "text": "What is CoVaR 1% of the system given bank i?",
                "options": [
                    "The VaR 1% of the system on a day when bank i is at its own VaR 1%",
                    "The VaR 1% of bank i on the market's worst days",
                    "The correlation of bank i with the market",
                    "The VaR 99% of bank i"
                ],
                "correctExplanation": "CoVaR conditions on the bank: it is the 1% quantile of the system's return, with a minus sign, when the bank is in distress.",
                "incorrectExplanation": "Conditioning on the market's worst days is MES; correlation is not a tail measure; the level is named by the tail probability."
            },
            "ro": {
                "title": "CoVaR",
                "text": "Ce este CoVaR 1% al sistemului condiționat de banca i?",
                "options": [
                    "VaR 1% al sistemului într-o zi în care banca i se află la propriul VaR 1%",
                    "VaR 1% al băncii i în cele mai proaste zile ale pieței",
                    "Corelația dintre banca i și piață",
                    "VaR 99% al băncii i"
                ],
                "correctExplanation": "CoVaR condiționează după bancă: este cuantila de 1% a randamentului sistemului, cu semn schimbat, cînd banca este în dificultate.",
                "incorrectExplanation": "Condiționarea după zilele proaste ale pieței definește MES; corelația nu este o măsură a cozii; nivelul se exprimă prin probabilitatea cozii."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Delta-CoVaR",
                "text": "A quantile regression at 1% gives slope b = 0.5; the bank has q(1%) = -6% and median 0%. What is Delta-CoVaR 1%?",
                "options": [
                    "0.5%",
                    "-3%",
                    "6%",
                    "3%"
                ],
                "correctExplanation": "Delta-CoVaR = b (q50 - q1) = 0.5 x (0 - (-6)) = 3%: the rise in the system's VaR from a normal day to distress.",
                "incorrectExplanation": "Only the slope and the distance between the median and the 1% quantile enter: 0.5 times 6 gives 3, a positive number."
            },
            "ro": {
                "title": "Delta-CoVaR",
                "text": "O regresie cuantilică la 1% dă panta b = 0,5; banca are q(1%) = -6% și mediana 0%. Cît este Delta-CoVaR 1%?",
                "options": [
                    "0,5%",
                    "-3%",
                    "6%",
                    "3%"
                ],
                "correctExplanation": "Delta-CoVaR = b (q50 - q1) = 0,5 x (0 - (-6)) = 3%: creșterea VaR-ului sistemului de la o zi obișnuită la o zi de criză.",
                "incorrectExplanation": "Intră doar panta și distanța dintre mediană și cuantila de 1%: 0,5 înmulțit cu 6 dă 3, un număr pozitiv."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Computing CoVaR",
                "text": "Quantile regression at 1%: intercept a = -2, slope b = 0.4; the bank's 1% quantile is -5%. What is CoVaR 1%?",
                "options": [
                    "2%",
                    "-4%",
                    "4%",
                    "5%"
                ],
                "correctExplanation": "CoVaR = -(a + b q) = -(-2 + 0.4 x (-5)) = -(-4) = 4%.",
                "incorrectExplanation": "Plug the bank's quantile into the regression line and change the sign: the conditional quantile is -4%, so CoVaR is a loss of 4%."
            },
            "ro": {
                "title": "Calculul CoVaR",
                "text": "Regresia cuantilică la 1%: termenul liber a = -2, panta b = 0,4; cuantila de 1% a băncii este -5%. Cît este CoVaR 1%?",
                "options": [
                    "2%",
                    "-4%",
                    "4%",
                    "5%"
                ],
                "correctExplanation": "CoVaR = -(a + b q) = -(-2 + 0,4 x (-5)) = -(-4) = 4%.",
                "incorrectExplanation": "Înlocuim cuantila băncii în dreapta de regresie și schimbăm semnul: cuantila condiționată este -4%, deci CoVaR este o pierdere de 4%."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Quantile regression",
                "text": "In a quantile regression at level 1%, how is a residual below the line weighted against one above it?",
                "options": [
                    "Equally",
                    "99 times more",
                    "1% less",
                    "It is ignored"
                ],
                "correctExplanation": "The check loss weighs negative residuals by 1 - alpha = 0.99 and positive ones by alpha = 0.01: the line passes below 99% of the points.",
                "incorrectExplanation": "Equal weights give the median regression; the asymmetric weights 0.99 and 0.01 are what makes the line a 1% quantile."
            },
            "ro": {
                "title": "Regresia cuantilică",
                "text": "Într-o regresie cuantilică la nivelul 1%, cum este ponderat un reziduu de sub dreaptă față de unul de deasupra ei?",
                "options": [
                    "La fel",
                    "De 99 de ori mai mult",
                    "Cu 1% mai puțin",
                    "Este ignorat"
                ],
                "correctExplanation": "Funcția de pierdere ponderează reziduurile negative cu 1 - alfa = 0,99 și pe cele pozitive cu alfa = 0,01: dreapta trece sub 99% dintre puncte.",
                "incorrectExplanation": "Ponderile egale dau regresia mediană; ponderile asimetrice 0,99 și 0,01 fac din dreaptă o cuantilă de 1%."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "VaR and Delta-CoVaR",
                "text": "For the 13 banks of the chapter, the Spearman correlation between VaR 1% and Delta-CoVaR 1% is -0.31. What follows?",
                "options": [
                    "A bank's own tail risk is a poor guide to its contribution to systemic risk",
                    "The two measures are identical",
                    "The bank with the largest VaR is the most systemic",
                    "Delta-CoVaR is always larger than VaR"
                ],
                "correctExplanation": "Citigroup and Bank of America have among the largest VaR and the smallest Delta-CoVaR: regulating by own VaR misses systemic risk.",
                "incorrectExplanation": "A negative rank correlation means the orderings disagree; the levels of the two measures are also different."
            },
            "ro": {
                "title": "VaR și Delta-CoVaR",
                "text": "Pentru cele 13 bănci din capitol, corelația Spearman dintre VaR 1% și Delta-CoVaR 1% este -0,31. Ce rezultă?",
                "options": [
                    "Riscul propriu din coadă al unei bănci spune puțin despre contribuția ei la riscul sistemic",
                    "Cele două măsuri sînt identice",
                    "Banca cu cel mai mare VaR este cea mai importantă sistemic",
                    "Delta-CoVaR este întotdeauna mai mare decît VaR"
                ],
                "correctExplanation": "Citigroup și Bank of America sînt printre primele după VaR și printre ultimele după Delta-CoVaR: reglementarea după VaR-ul propriu scapă din vedere riscul sistemic.",
                "incorrectExplanation": "O corelație negativă a rangurilor înseamnă că ordonările nu sînt de acord; și nivelurile celor două măsuri diferă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "MES",
                "text": "What does MES 5% of a bank measure?",
                "options": [
                    "The loss of the market on the bank's worst 5% of days",
                    "The VaR 5% of the bank",
                    "The share of the bank in the market index",
                    "The average loss of the bank on the market's worst 5% of days"
                ],
                "correctExplanation": "MES conditions on the market: sort the days by the market return and average the bank's losses on the worst 5%.",
                "incorrectExplanation": "Conditioning on the bank's worst days reverses the direction; VaR 5% ignores the market; the index weight is not a tail measure."
            },
            "ro": {
                "title": "MES",
                "text": "Ce măsoară MES 5% al unei bănci?",
                "options": [
                    "Pierderea pieței în cele mai proaste 5% dintre zilele băncii",
                    "VaR 5% al băncii",
                    "Ponderea băncii în indicele pieței",
                    "Pierderea medie a băncii în cele mai proaste 5% dintre zilele pieței"
                ],
                "correctExplanation": "MES condiționează după piață: ordonăm zilele după randamentul pieței și facem media pierderilor băncii în cele mai proaste 5%.",
                "incorrectExplanation": "Condiționarea după zilele proaste ale băncii inversează direcția; VaR 5% ignoră piața; ponderea în indice nu este o măsură a cozii."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "MES from a table",
                "text": "On the two worst market days of a sample the bank returned -9.0% and -6.5%. What is the MES at that level?",
                "options": [
                    "-7.75%",
                    "9.0%",
                    "7.75%",
                    "15.5%"
                ],
                "correctExplanation": "MES = -(-9.0 - 6.5)/2 = 7.75%: the average bank loss on the market's worst days, reported as a positive number.",
                "incorrectExplanation": "MES is an average, not a sum or a maximum, and it is reported as a positive loss."
            },
            "ro": {
                "title": "MES dintr-un tabel",
                "text": "În cele mai proaste două zile ale pieței dintr-un eșantion, banca a avut randamentele -9,0% și -6,5%. Cît este MES la acel nivel?",
                "options": [
                    "-7,75%",
                    "9,0%",
                    "7,75%",
                    "15,5%"
                ],
                "correctExplanation": "MES = -(-9,0 - 6,5)/2 = 7,75%: pierderea medie a băncii în cele mai proaste zile ale pieței, raportată ca număr pozitiv.",
                "incorrectExplanation": "MES este o medie, nu o sumă sau un maxim, și se raportează ca pierdere pozitivă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "LRMES",
                "text": "A bank loses on average 3% on the days when the market falls by more than 2%. What is LRMES = 1 - exp(-18 MES)?",
                "options": [
                    "3%",
                    "about 41.7%",
                    "about 54%",
                    "about 18%"
                ],
                "correctExplanation": "1 - exp(-18 x 0.03) = 1 - exp(-0.54) = 1 - 0.583 = 41.7%: the loss of the bank if the market falls 40% over six months.",
                "incorrectExplanation": "MES enters as a fraction, 0.03; the exponent is -0.54, and 1 - exp(-0.54) is about 0.417, not 0.54 or 0.18."
            },
            "ro": {
                "title": "LRMES",
                "text": "O bancă pierde în medie 3% în zilele în care piața scade cu peste 2%. Cît este LRMES = 1 - exp(-18 MES)?",
                "options": [
                    "3%",
                    "circa 41,7%",
                    "circa 54%",
                    "circa 18%"
                ],
                "correctExplanation": "1 - exp(-18 x 0,03) = 1 - exp(-0,54) = 1 - 0,583 = 41,7%: pierderea băncii dacă piața scade cu 40% în șase luni.",
                "incorrectExplanation": "MES intră ca fracție, 0,03; exponentul este -0,54, iar 1 - exp(-0,54) este circa 0,417, nu 0,54 sau 0,18."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "SRISK",
                "text": "D = 900, W = 100, LRMES = 55%, k = 8%. What is SRISK = kD - (1 - k)(1 - LRMES)W?",
                "options": [
                    "30.6",
                    "72",
                    "-30.6",
                    "41.4"
                ],
                "correctExplanation": "0.08 x 900 = 72; 0.92 x 0.45 x 100 = 41.4; SRISK = 72 - 41.4 = 30.6: capital missing in a crisis.",
                "incorrectExplanation": "72 is only the capital requirement and 41.4 only the equity left in the crisis; SRISK is their difference, positive here."
            },
            "ro": {
                "title": "SRISK",
                "text": "D = 900, W = 100, LRMES = 55%, k = 8%. Cît este SRISK = kD - (1 - k)(1 - LRMES)W?",
                "options": [
                    "30,6",
                    "72",
                    "-30,6",
                    "41,4"
                ],
                "correctExplanation": "0,08 x 900 = 72; 0,92 x 0,45 x 100 = 41,4; SRISK = 72 - 41,4 = 30,6: capitalul lipsă într-o criză.",
                "incorrectExplanation": "72 este doar cerința de capital, iar 41,4 doar capitalul rămas în criză; SRISK este diferența lor, aici pozitivă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Negative SRISK",
                "text": "What does a negative SRISK mean?",
                "options": [
                    "The bank will fail",
                    "The calculation is wrong",
                    "The bank has negative equity",
                    "The bank keeps more capital than required even after a severe market fall: a capital surplus"
                ],
                "correctExplanation": "SRISK > 0 is a shortfall; SRISK < 0 means that after the crisis loss the equity still exceeds k times the assets.",
                "incorrectExplanation": "A negative value is a meaningful result about low leverage, not an error, a failure signal or negative equity."
            },
            "ro": {
                "title": "SRISK negativ",
                "text": "Ce înseamnă un SRISK negativ?",
                "options": [
                    "Banca va falimenta",
                    "Calculul este greșit",
                    "Banca are capital propriu negativ",
                    "Banca păstrează mai mult capital decît este necesar chiar și după o scădere severă a pieței: un surplus de capital"
                ],
                "correctExplanation": "SRISK > 0 este un deficit; SRISK < 0 înseamnă că după pierderea din criză capitalul propriu depășește încă k înmulțit cu activele.",
                "incorrectExplanation": "O valoare negativă este un rezultat cu sens despre un efect de levier mic, nu o eroare, un semnal de faliment sau capital negativ."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "SRISK and the share price",
                "text": "The share price of a bank falls by 30% while its debt is unchanged. What happens to SRISK?",
                "options": [
                    "It falls, because the bank is smaller",
                    "It does not change, because the debt is unchanged",
                    "It rises, because leverage rises and the equity cushion is smaller",
                    "It becomes zero"
                ],
                "correctExplanation": "A lower W raises L = (D + W)/W and lowers (1 - k)(1 - LRMES)W: SRISK grows before any loan defaults.",
                "incorrectExplanation": "SRISK depends on the market value of equity; with D fixed, a smaller W can only increase the shortfall."
            },
            "ro": {
                "title": "SRISK și prețul acțiunii",
                "text": "Prețul acțiunii unei bănci scade cu 30%, iar datoriile rămîn neschimbate. Ce se întîmplă cu SRISK?",
                "options": [
                    "Scade, pentru că banca este mai mică",
                    "Nu se schimbă, pentru că datoriile nu se schimbă",
                    "Crește, pentru că efectul de levier crește și rezerva de capital este mai mică",
                    "Devine zero"
                ],
                "correctExplanation": "Un W mai mic crește L = (D + W)/W și reduce (1 - k)(1 - LRMES)W: SRISK crește înainte ca vreun credit să devină neperformant.",
                "incorrectExplanation": "SRISK depinde de valoarea de piață a capitalului propriu; cu D fix, un W mai mic poate doar să mărească deficitul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Contagion or interdependence",
                "text": "The correlation of two bank portfolios rises from 0.40 to 0.60 in a crisis, but the Forbes-Rigobon corrected value is 0.08. What do we conclude?",
                "options": [
                    "Contagion is proven",
                    "Interdependence: the rise is explained by the higher volatility, not by a stronger link",
                    "The correlation was computed wrongly",
                    "The banks became independent"
                ],
                "correctExplanation": "With a fixed link, correlation rises mechanically with the variance of the source market; the corrected value shows no stronger link.",
                "incorrectExplanation": "The raw rise does not prove contagion; the correction does not mean independence, only that the link did not strengthen."
            },
            "ro": {
                "title": "Contagiune sau interdependență",
                "text": "Corelația dintre două portofolii de bănci crește de la 0,40 la 0,60 într-o criză, dar valoarea corectată Forbes-Rigobon este 0,08. Ce concluzionăm?",
                "options": [
                    "Contagiunea este dovedită",
                    "Interdependență: creșterea se explică prin volatilitatea mai mare, nu printr-o legătură mai puternică",
                    "Corelația a fost calculată greșit",
                    "Băncile au devenit independente"
                ],
                "correctExplanation": "Cu o legătură fixă, corelația crește mecanic odată cu varianța pieței-sursă; valoarea corectată nu arată o legătură mai puternică.",
                "incorrectExplanation": "Creșterea necorectată nu dovedește contagiunea; corecția nu înseamnă independență, ci doar că legătura nu s-a întărit."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Tail dependence",
                "text": "Which copula allows joint crashes of a bank and the market, with positive lower tail dependence?",
                "options": [
                    "The t copula with few degrees of freedom",
                    "The Gaussian copula",
                    "The independence copula",
                    "Any copula with correlation below 0.5"
                ],
                "correctExplanation": "The t copula has lambda_L > 0, larger for small nu; the chapter finds nu of about 3 to 4 for banks and their markets.",
                "incorrectExplanation": "The Gaussian and the independence copulas have zero tail dependence; correlation alone does not determine it."
            },
            "ro": {
                "title": "Dependența în cozi",
                "text": "Ce copulă permite prăbușiri simultane ale unei bănci și ale pieței, cu dependență pozitivă în coada inferioară?",
                "options": [
                    "Copula t cu puține grade de libertate",
                    "Copula Gaussiană",
                    "Copula de independență",
                    "Orice copulă cu corelația sub 0,5"
                ],
                "correctExplanation": "Copula t are lambda_L > 0, mai mare pentru nu mic; capitolul găsește nu în jur de 3-4 pentru bănci și piețele lor.",
                "incorrectExplanation": "Copula Gaussiană și copula de independență au dependență în coadă zero; corelația singură nu o determină."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Granger causality",
                "text": "Bank A Granger-causes bank B. What does this mean?",
                "options": [
                    "A owns shares of B",
                    "A lent money to B",
                    "A shock to A always causes a loss at B",
                    "Past returns of A help to forecast B beyond the past of B"
                ],
                "correctExplanation": "Granger causality is about forecasting: the lagged returns of A have a significant coefficient in the regression of B.",
                "incorrectExplanation": "A statistical link does not reveal ownership, loans or a causal channel; it only shows predictive content."
            },
            "ro": {
                "title": "Cauzalitatea Granger",
                "text": "Banca A cauzează în sens Granger banca B. Ce înseamnă aceasta?",
                "options": [
                    "A deține acțiuni ale lui B",
                    "A i-a împrumutat bani lui B",
                    "Un șoc la A produce întotdeauna o pierdere la B",
                    "Randamentele trecute ale lui A ajută la prognoza lui B dincolo de trecutul lui B"
                ],
                "correctExplanation": "Cauzalitatea Granger privește prognoza: randamentele decalate ale lui A au un coeficient semnificativ în regresia lui B.",
                "incorrectExplanation": "O legătură statistică nu dezvăluie acționariate, credite sau un canal cauzal; arată doar conținut predictiv."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Degree of Granger causality",
                "text": "In a 36-month window, 5% of the ordered pairs of banks have a significant Granger link at the 5% level. How do we read this?",
                "options": [
                    "The banks are strongly connected",
                    "Exactly 5% of the banks are systemic",
                    "About what chance alone would give: no evidence of a network",
                    "The test has failed"
                ],
                "correctExplanation": "Under no links about 5% of the tests reject at the 5% level; the DGC must be compared with this benchmark.",
                "incorrectExplanation": "A DGC of 5% is the false-positive rate of the test; it is not a share of systemic banks and does not signal a failed test."
            },
            "ro": {
                "title": "Gradul de cauzalitate Granger",
                "text": "Într-o fereastră de 36 de luni, 5% dintre perechile ordonate de bănci au o legătură Granger semnificativă la nivelul 5%. Cum interpretăm?",
                "options": [
                    "Băncile sînt puternic conectate",
                    "Exact 5% dintre bănci sînt sistemice",
                    "Aproximativ cît ar da doar întîmplarea: nicio dovadă a unei rețele",
                    "Testul a eșuat"
                ],
                "correctExplanation": "Fără legături, aproximativ 5% dintre teste resping la nivelul 5%; DGC trebuie comparat cu acest reper.",
                "incorrectExplanation": "Un DGC de 5% este rata rezultatelor fals pozitive a testului; nu este o proporție de bănci sistemice și nu semnalează un test eșuat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Diebold-Yilmaz",
                "text": "In a connectedness table, what is the FROM value of bank i?",
                "options": [
                    "The share of the variance of the market due to bank i",
                    "The share of the forecast error variance of bank i due to shocks to the other banks",
                    "The VaR of bank i",
                    "The number of Granger links of bank i"
                ],
                "correctExplanation": "FROM others is the row sum outside the diagonal: how much of bank i's forecast uncertainty comes from the others.",
                "incorrectExplanation": "The column sum is TO others; VaR and Granger links are different tools."
            },
            "ro": {
                "title": "Diebold-Yilmaz",
                "text": "Într-un tabel de conectivitate, ce reprezintă valoarea FROM a băncii i?",
                "options": [
                    "Partea din varianța pieței datorată băncii i",
                    "Partea din varianța erorii de prognoză a băncii i datorată șocurilor celorlalte bănci",
                    "VaR-ul băncii i",
                    "Numărul legăturilor Granger ale băncii i"
                ],
                "correctExplanation": "FROM este suma pe rînd din afara diagonalei: cîtă incertitudine din prognoza băncii i vine de la celelalte bănci.",
                "incorrectExplanation": "Suma pe coloană este TO; VaR și legăturile Granger sînt alte instrumente."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Countercyclical buffer",
                "text": "How is the Basel III countercyclical capital buffer meant to work?",
                "options": [
                    "It is raised when credit grows too fast and released in a crisis",
                    "It is a fixed 8% of assets",
                    "It applies only to banks that fail",
                    "It is raised in crises"
                ],
                "correctExplanation": "The buffer (0 to 2.5% of risk-weighted assets) is built in booms and released in busts, when banks need room to keep lending.",
                "incorrectExplanation": "Raising capital in a crisis would deepen it; the buffer varies over the cycle and applies to all banks of a country."
            },
            "ro": {
                "title": "Amortizorul anticiclic",
                "text": "Cum funcționează amortizorul anticiclic de capital din Basel III?",
                "options": [
                    "Este crescut cînd creditul crește prea repede și eliberat într-o criză",
                    "Este fix, 8% din active",
                    "Se aplică doar băncilor care falimentează",
                    "Este crescut în crize"
                ],
                "correctExplanation": "Amortizorul (0-2,5% din activele ponderate la risc) se constituie în perioadele de avînt și se eliberează în crize, cînd băncile au nevoie de spațiu pentru a credita.",
                "incorrectExplanation": "Creșterea cerințelor de capital într-o criză ar adînci-o; amortizorul variază de-a lungul ciclului și se aplică tuturor băncilor unei țări."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Macroprudential authority in Romania",
                "text": "Who sets the macroprudential capital buffers of Romanian banks?",
                "options": [
                    "The Bucharest Stock Exchange",
                    "The European Central Bank alone",
                    "Each bank for itself",
                    "The BNR, on recommendations of the National Committee for Macroprudential Oversight (CNSM)"
                ],
                "correctExplanation": "The CNSM brings together the BNR, the ASF and the Government; the BNR applies the countercyclical, O-SII and systemic risk buffers.",
                "incorrectExplanation": "The stock exchange and the banks themselves do not set buffers; for Romania, outside the banking union, the national authority applies them."
            },
            "ro": {
                "title": "Autoritatea macroprudențială din România",
                "text": "Cine stabilește amortizoarele macroprudențiale de capital ale băncilor din România?",
                "options": [
                    "Bursa de Valori București",
                    "Doar Banca Centrală Europeană",
                    "Fiecare bancă pentru sine",
                    "BNR, pe baza recomandărilor Comitetului Național pentru Supravegherea Macroprudențială (CNSM)"
                ],
                "correctExplanation": "CNSM reunește BNR, ASF și Guvernul; BNR aplică amortizorul anticiclic, amortizorul O-SII și amortizorul pentru riscul sistemic.",
                "incorrectExplanation": "Bursa și băncile nu stabilesc amortizoare; pentru România, aflată în afara uniunii bancare, autoritatea națională le aplică."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "March 2023",
                "text": "What made Silicon Valley Bank fail so quickly in March 2023?",
                "options": [
                    "Bad loans to households",
                    "A cyber attack",
                    "Uninsured deposits that left in a fast run after large losses on long-term bonds when interest rates rose",
                    "A fall in the price of Bitcoin"
                ],
                "correctExplanation": "Rising rates created unrealised losses on bonds; depositors above the insured limit tried to withdraw about 42 billion USD in one day.",
                "incorrectExplanation": "The losses came from interest-rate risk on securities, not from household loans, a cyber attack or crypto prices."
            },
            "ro": {
                "title": "Martie 2023",
                "text": "Ce a făcut ca Silicon Valley Bank să falimenteze atît de repede în martie 2023?",
                "options": [
                    "Credite neperformante acordate gospodăriilor",
                    "Un atac informatic",
                    "Depozite negarantate retrase rapid, după pierderi mari pe obligațiunile pe termen lung cînd au crescut dobînzile",
                    "O scădere a prețului Bitcoin"
                ],
                "correctExplanation": "Creșterea dobînzilor a produs pierderi nerealizate pe obligațiuni; deponenții de peste plafonul garantat au încercat să retragă circa 42 de miliarde USD într-o zi.",
                "incorrectExplanation": "Pierderile au venit din riscul de dobîndă al titlurilor, nu din credite către gospodării, dintr-un atac informatic sau din prețul activelor cripto."
            }
        }
    ]
};
