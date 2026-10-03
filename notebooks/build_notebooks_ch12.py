"""
build_notebooks_ch12.py -- lecture and seminar notebooks of Chapter 12 (SFM): scoring models
===========================================================================================
Output: notebooks/EN/chapter12_lecture_notebook.ipynb, notebooks/EN/chapter12_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_12/generate_all_charts.py and seminar12.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. The credit data are read from data/credit of the SFM repository.
Run:  python3 notebooks/build_notebooks_ch12.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter12_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter12_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 12
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, md, src   # noqa: E402
from sfm_quantlets import IMPORTS, style_cell                  # noqa: E402

chapter_paths(12)
import generate_all_charts as g   # noqa: E402
import seminar12 as s             # noqa: E402

CONSTS = (f'CREDIT_RAW = {g.CREDIT_RAW!r}\nFILES = {g.FILES!r}\nSEED = {g.SEED!r}\nTEST_SHARE = {g.TEST_SHARE!r}\n'
          f'POP_BAD = {g.POP_BAD!r}\nCOST_FN = {g.COST_FN!r}\nCOST_FP = {g.COST_FP!r}\n'
          f'PDO, BASE_SCORE, BASE_ODDS = {g.PDO!r}, {g.BASE_SCORE!r}, {g.BASE_ODDS!r}\nIV_MIN = {g.IV_MIN!r}\n'
          f'N_REPEAT, N_FOLD = {g.N_REPEAT!r}, {g.N_FOLD!r}\n' + f'BINS = {g.BINS!r}'.replace('inf', 'np.inf') + '\n'
          f'NUMERIC = {g.NUMERIC!r}\nSGC_VARS = {g.SGC_VARS!r}\nSTATUS = {g.STATUS!r}\nTW_FEATURES = {g.TW_FEATURES!r}\n'
          f'COLORS = {g.COLORS!r}')
DATAF = [g.load_credit, g.taiwan_features, g.stratified_split]
WOE = [g.bin_values, g.woe_table, g.iv_all, g.iv_strength, g.woe_transform]
MODELS = [g.logit_fit, g.logit_predict, g.lda_fit, g.lda_predict]
VALID = [g.auc, g.delong, g.roc_points, g.cap_points, g.accuracy_ratio, g.ks_stat, g.brier, g.calibration_table,
         g.confusion]
SCORE = [g.scaling, g.points_table, g.score_from_pd, g.prior_shift, g.sgc_scorecard]


def setup_cells():
    """Imports, chart style and the credit data (local data/credit, else the raw GitHub URL)."""
    return [code(IMPORTS),
            md('## Chart style\n\n- Transparent background, legend below the plot, course palette.'),
            code(style_cell()),
            md('## Data\n\n'
               '- **South German Credit** (Grömping, 2019; UCI Machine Learning Repository, doi:10.24432/C5QG88, CC BY 4.0): '
               '1000 consumer credits of a regional bank in southern Germany, 1973–1975; 300 bad and 700 good credits '
               '(bad credits oversampled; the bank\'s bad rate was about 5%). It corrects the coding errors of the '
               'widely used UCI "Statlog (German Credit)" version.\n'
               '- **Default of Credit Card Clients** (Yeh and Lien, 2009; UCI, doi:10.24432/C55S3H, CC BY 4.0): 30,000 '
               'card holders in Taiwan; default on the October 2005 payment (22.1%).\n'
               '- Both files are saved in `data/credit` of the SFM repository and read locally or from GitHub in Colab. '
               'Default = 1 for a bad credit or a missed payment.'),
            code(CONSTS + '\n\n\n' + src(*DATAF))]


# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 12: Scoring models\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Default data: the South German Credit data and the Taiwan credit card data; class imbalance.\n"
       "- Weight of Evidence and Information Value; logistic regression and Fisher's linear discriminant analysis.\n"
       "- A scorecard: points to double the odds, prior correction for oversampling.\n"
       "- Validation: confusion matrix, cost-based cut-off, ROC and AUC (DeLong), Gini, CAP, KS, Brier score, calibration, "
       "repeated cross-validation.\n"
       "- Basel IRB capital and the conditional PD of the one-factor model (SFEdefaproba), reject inference, fairness, "
       "the Altman Z-score and the Merton distance to default.\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 21–22; Härdle and "
       "Simar (2019), *Applied Multivariate Statistical Analysis*, 5th ed., Ch. 14; Siddiqi (2006); Thomas, Crook and "
       "Edelman (2017)."),
    *setup_cells(),
    md("## 1. Default data\n\n- Bad rate by the status of the checking account; the WoE table of the same variable."),
    code(src(*WOE, g.fig_default_by_status)),
    code("df = load_credit('sgc')\nprint(df.shape, df['default'].mean())\nfig_default_by_status(df, save=False)"),
    code("woe_table(df['status'], df['default'], 'status').round(3)"),
    md("## 2. Weight of Evidence and Information Value\n\n"
       "- $\\mathrm{WoE}_i = \\ln[(g_i/G)/(b_i/B)]$; $\\mathrm{IV} = \\sum_i (g_i/G - b_i/B)\\,\\mathrm{WoE}_i$.\n"
       "- Rule of thumb (Siddiqi, 2006): below 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, above 0.3 strong."),
    code(src(g.fig_woe_duration, g.fig_iv)),
    code("fig_woe_duration(df, save=False)"),
    code("pd.Series(fig_iv(df, save=False)).round(3)"),
    md("## 3. Logistic regression\n\n"
       "- $\\ln[p/(1-p)] = \\beta_0 + x^\\top\\beta$, estimated by maximum likelihood (Newton–Raphson); "
       "$e^{\\beta_j}$ is an odds ratio."),
    code(src(*MODELS, g.small_logit, g.fig_logistic_curve, g.fig_odds_ratios)),
    code("fig_logistic_curve(df, save=False)"),
    code("L = small_logit(df)\nprint('LR statistic', round(L['lr'], 1))\nL['table'].round(3)"),
    code("fig_odds_ratios(L, save=False)"),
    md("## 4. Fisher's linear discriminant analysis\n\n- $w = S_W^{-1}(m_1 - m_0)$; boundaries of LDA and logit for "
       "duration and age."),
    code(src(g.fig_lda_2d)),
    code("fig_lda_2d(df, save=False)"),
    md("## 5. The WoE scorecard\n\n"
       "- Stratified 70/30 split; WoE and IV on the training sample; variables with IV ≥ 0.10; logit and LDA on the WoE "
       "values.\n- Scaling: 20 points to double the odds, 600 points at good:bad odds of 50:1; prior correction to 5%."),
    code(src(*SCORE, g.fig_score_dist)),
    code("S = sgc_scorecard(df)\nprint(S['sel'])\npd.DataFrame({'b': S['logit']['b'], 'se': S['logit']['se']}, "
         "index=['const'] + S['sel']).round(3)"),
    code("print(scaling(), prior_shift(0.3, POP_BAD))\npoints_table(S['logit'], S['tables'], S['sel']).round(2)"),
    code("fig_score_dist(S, save=False)"),
    md("## 6. Validation\n\n"
       "- Confusion matrix at the cut-offs 0.5 and 1/6 (costs 5:1); ROC, AUC with DeLong standard errors, Gini, CAP and "
       "the accuracy ratio, KS, Brier score, Hosmer–Lemeshow.\n"
       "- Repeated stratified 5-fold cross-validation, with the WoE bins and the selection redone inside every fold."),
    code(src(*VALID, g.validation, g.cv_auc, g.fig_roc, g.fig_cap, g.fig_ks, g.fig_cost_cutoff, g.fig_cv_auc)),
    code("Vd = validation(S)\nprint(Vd['cm50'])\nprint(Vd['cm_cost'])\n{k: v for k, v in Vd.items() if not isinstance(v, dict)}"),
    code("fig_roc(S, save=False)\nfig_cap(S, save=False)\nfig_ks(S, save=False)"),
    code("fig_cost_cutoff(S, save=False)"),
    code("res = cv_auc(df)\nfig_cv_auc(res, save=False)"),
    md("## 7. Calibration on the Taiwan data\n\n- A logit on 13 standardised inputs; ten groups of 900 test clients."),
    code(src(g.taiwan_model, g.fig_calibration)),
    code("T = taiwan_model()\ncal = fig_calibration(T, save=False)\n{k: v for k, v in cal.items() if k != 'groups'}"),
    md("## 8. Basel IRB capital\n\n- $K = \\mathrm{LGD}[\\Phi((\\Phi^{-1}(\\mathrm{PD}) + \\sqrt{R}\\Phi^{-1}(0.999))/\\sqrt{1-R}) "
       "- \\mathrm{PD}]$, RWA $= 12.5\\,K\\,$EAD (Basel II, paragraphs 328–330)."),
    code(src(g.irb_capital, g.fig_irb, g.conditional_pd, g.fig_conditional_pd)),
    code("fig_irb(save=False)\npd.DataFrame({k: [irb_capital(p, 0.45, k)[0] for p in (0.005, 0.01, 0.02, 0.05, 0.1)] "
         "for k in ('other retail', 'mortgage', 'revolving')}, index=[0.005, 0.01, 0.02, 0.05, 0.1]).round(4)"),
    code("fig_conditional_pd(save=False)"),
    md("## 9. Reject inference and fairness\n\n"
       "- An old score accepts the best 70%; a model trained on the accepted only against one trained on all applicants.\n"
       "- Approval, default and mean PD by sex and age group at a 75% approval cut-off."),
    code(src(g.reject_inference_sim, g.fairness_table, g.fig_fairness)),
    code("reject_inference_sim(T)"),
    code("F = fairness_table(T)\nfig_fairness(F, save=False)\npd.DataFrame(F['groups']).T.round(3)"),
    md("## 10. Altman Z-score and Merton distance to default"),
    code(src(g.altman_z, g.merton_dd, g.merton_from_equity, g.fig_merton)),
    code("print(altman_z(0.15, 0.20, 0.08, 0.90, 1.10))\nprint(merton_dd(100, 70, 0.25, 0.05, 1.0))\n"
         "print(merton_from_equity(40, 0.5, 70, 0.03, 1.0))\nfig_merton(save=False)"),
    md("## 11. AI for scientific discovery: is the gain of flexible models worth their cost?\n\n"
       "- Open question: do flexible machine-learning scores improve default prediction enough to justify the loss of "
       "transparency, and are the gains equal across groups?\n"
       "- Starter result: how well does the Taiwan logit rank and calibrate within each group?\n"
       "- Check before trusting any answer, yours or an AI's: the coding of default, the WoE sign convention, no leakage "
       "of the test sample into the bins, AUC is not accuracy, Gini = 2 AUC − 1, prior correction before EL or capital."),
    code("te = T['te']\nsex = T['df']['SEX'].values[te]\nage = T['df']['AGE'].values[te]\ny, p = T['y'][te], T['p'][te]\n"
         "groups = {'men': sex == 1, 'women': sex == 2, 'age <= 25': age <= 25, 'age 26-50': (age > 25) & (age <= 50), "
         "'age > 50': age > 50}\n"
         "pd.DataFrame({g_: {'n': int(m.sum()), 'AUC': auc(p[m], y[m]), 'mean PD': p[m].mean(), 'default rate': y[m].mean(), "
         "'Brier': brier(p[m], y[m])} for g_, m in groups.items()}).T.round(3)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.logit_applicant, s.woe_by_hand, s.category_counts, s.small_sample_metrics, s.scaling_example,
               s.attribute_points, s.prior_and_el]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 12: Scoring models\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 12: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *setup_cells(),
    md("## Seminar functions\n\n"
       "- WoE and IV, logit and LDA, AUC with DeLong standard errors, ROC, CAP, KS, Brier score, calibration, confusion "
       "matrix, scorecard scaling and prior correction, the Basel IRB capital, the Taiwan logit.\n"
       "- Helpers for Part A: the PD of an applicant, WoE and IV from counts, the metrics of a small sample, scorecard "
       "scaling, attribute points, prior correction and expected loss.\n"
       "- Constants: `A5`, `A6` (the small samples of Part A) and `SMALL` (the coefficients of the logit of A1)."),
    code(f'A5 = {s.A5!r}\nA6 = {s.A6!r}\nSMALL = {s.SMALL!r}\n\n\n' + src(*WOE, *MODELS, *VALID, *SCORE, g.irb_capital,
                                                                         g.taiwan_model) + '\n\n\n' + src(*SEM_HELPERS)),
    code("# [solutions only]\n" + src(s.taiwan_iv, s.logit_vs_lda_taiwan, s.fairness_check, s.c1_benchmark, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: the PD of an applicant\n\n"
       "**Context.** Logit for default: constant −2.101; duration (years) 0.385; age (decades) −0.159; amount (1000 DM) "
       "0.029; no checking account 1.859; balance below 0 DM 1.338. Applicant: 2 years, 30 years old, 3000 DM, no checking "
       "account.\n\n"
       "1. Compute the log-odds $\\eta$ and the PD.\n"
       "2. Compute the odds of default and interpret $e^{1.859}$.\n"
       "3. Compute the PD with one more year of duration and compare it with $p(1-p)\\beta$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nlogit_applicant(2, 3, 3, 1, 0)"),
    md("**Interpretation of the result.** The odds ratio is constant (6.42 for no checking account); the change in PD "
       "for one more year depends on where the applicant is."),
    md("## A2 [Proposed]: a second applicant\n\n"
       "**Context.** The same logit as in A1. Applicant: 1 year, 45 years old, 2000 DM, balance below 0 DM. Model: A1.\n\n"
       "1. Compute $\\eta$, the odds and the PD.\n"
       "2. Compute the PD with one more year of duration.\n"
       "3. Compare the change in PD with that of A1 and explain the difference.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nlogit_applicant(1, 4.5, 2, 0, 1)"),
    md("## A3 [Solved]: WoE and IV of the checking account\n\n"
       "**Context.** South German Credit, full sample: goods and bads by the status of the checking account.\n\n"
       "1. Compute $g_i/G$ and $b_i/B$ for each group.\n"
       "2. Compute the WoE of each group.\n"
       "3. Compute the IV and classify the variable.\n\n"
       "**Report:** a table and one sentence."),
    code("# Solution\nc = category_counts('status')\nr = woe_by_hand(c['good'], c['bad'])\n"
         "print(round(r['iv'], 3), r['strength'])\n"
         "pd.DataFrame({'good': c['good'], 'bad': c['bad'], 'g/G': r['dist_good'], 'b/B': r['dist_bad'], 'WoE': r['woe'], "
         "'IV term': r['iv_part']}, index=[STATUS[k] for k in c['codes']]).round(3)"),
    md("**Interpretation of the result.** IV = 0.666 > 0.3: a strong variable; the WoE falls monotonically from the "
       "safest to the riskiest group."),
    md("## A4 [Proposed]: WoE and IV of the savings\n\n"
       "**Context.** South German Credit: goods and bads by savings (1: none or unknown, 2: below 100 DM, 3: 100–500 DM, "
       "4: 500–1000 DM, 5: at least 1000 DM). Model: A3.\n\n"
       "1. Compute the WoE of each group.\n"
       "2. Compute the IV and classify the variable.\n"
       "3. Propose a merger of groups with similar WoE.\n\n"
       "**Report:** a table and two sentences."),
    code("# Solution\nc = category_counts('savings')\nr = woe_by_hand(c['good'], c['bad'])\nprint(round(r['iv'], 3), r['strength'])\n"
         "pd.DataFrame({'good': c['good'], 'bad': c['bad'], 'WoE': r['woe'], 'IV term': r['iv_part']}, index=c['codes']).round(3)"),
    md("## A5 [Solved]: confusion matrix and AUC by counting\n\n"
       "**Context.** Ten applicants with their PD and outcome (`A5`).\n\n"
       "1. Build the confusion matrix at the cut-off 0.5 and compute accuracy, TPR and FPR.\n"
       "2. Compute the AUC by counting the (bad, good) pairs in the right order, and the Gini.\n"
       "3. Compute the KS statistic and the Brier score.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\nsmall_sample_metrics(A5['pd'], A5['y'], 0.5)"),
    md("**Interpretation of the result.** 17 of the 21 (bad, good) pairs are in the right order: AUC = 0.810, Gini = 0.619."),
    md("## A6 [Proposed]: a second sample\n\n"
       "**Context.** Eight applicants (`A6`). Model: A5.\n\n"
       "1. Build the confusion matrix at the cut-off 0.4 and compute accuracy, TPR and FPR.\n"
       "2. Compute the AUC by counting pairs, and the Gini.\n"
       "3. Compute the KS statistic and the Brier score.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\nsmall_sample_metrics(A6['pd'], A6['y'], 0.4)"),
    md("## A7 [Solved]: scaling a scorecard\n\n"
       "**Context.** PDO = 20; 600 points at good:bad odds 50:1; the WoE logit of Lecture 12 (six variables).\n\n"
       "1. Compute the Factor and the Offset.\n"
       "2. Compute the score of a PD of 5% and the PD of a score of 560.\n"
       "3. Compute the points of the checking-account groups.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nprint(scaling_example(20, 600, 50, 0.05, 560))\nattribute_points()"),
    md("**Interpretation of the result.** A well-funded account is worth about 45 points more than no account: more "
       "than two doublings of the good:bad odds."),
    md("## A8 [Proposed]: another scale, prior correction and expected loss\n\n"
       "**Context.** PDO = 40 and 500 points at odds 20:1; a model built on a sample with 30% bads gives PD = 30%; the "
       "population bad rate is 5%; EAD = 10,000 DM, LGD = 45%. Model: A7.\n\n"
       "1. Compute the Factor, the Offset, the score of a PD of 10% and the PD of a score of 520.\n"
       "2. Correct the PD of 30% to the population.\n"
       "3. Compute the expected loss with both PDs.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\nprint(scaling_example(40, 500, 20, 0.10, 520))\nprior_and_el()"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: a WoE scorecard for South German Credit\n\n"
       "**Question.** Which variables carry information about default, and how well does a WoE logit rank new applicants?\n\n"
       "1. Compute the WoE tables and the IV of the 20 variables on the training sample.\n"
       "2. Keep the variables with IV of at least 0.1 and estimate a logit for default on their WoE values.\n"
       "3. Compute the AUC on the training and on the test sample, with a 95% DeLong interval for the test AUC.\n"
       "4. Draw the WoE of the credit-history groups.\n"
       "5. Interpretation: does the order of the credit-history groups make economic sense?\n\n"
       "**Report:** the selected variables with their IV, the coefficients, two AUCs with the interval, the chart and two "
       "sentences."),
    code("# Solution\n" + src(s.b1_scorecard) + "\n\n\nb1 = b1_scorecard(save=False)\n"
         "print({k: b1[k] for k in ['sel', 'auc_train', 'auc', 'lo', 'hi']})\n"
         "pd.DataFrame({'b': b1['b'], 'se': b1['se']}, index=['const'] + b1['sel']).round(3)"),
    md("**Interpretation of the result.** Yes: a delay in the past and a critical account are the riskiest groups, all "
       "credits at this bank paid back duly the safest. Test AUC 0.801, 95% interval [0.749, 0.854]."),
    md("## B2 [Proposed]: information value of the Taiwan data\n\n"
       "**Question.** Which kind of information predicts the default of a card holder: payment behaviour or personal "
       "data? Model: B1.\n\n"
       "1. Compute the IV of PAY_0, PAY_2, LIMIT_BAL, PAY_AMT1, BILL_AMT1, AGE, SEX, EDUCATION and MARRIAGE on the "
       "training sample.\n"
       "2. Compute the default rate of each PAY_0 category.\n"
       "3. Rank the variables by IV and classify them with the rule of thumb.\n"
       "4. Interpretation: why is the last month's repayment status so much stronger than the personal data?\n\n"
       "**Report:** a table of nine IVs, six default rates and two sentences."),
    code("# Solution\ntaiwan_iv()"),
    md("## B3 [Solved]: validating the scorecard\n\n"
       "**Question.** How good is the B1 scorecard on the test sample, and which cut-off should the bank use?\n\n"
       "1. Build the confusion matrix at the cut-offs 0.5 and 1/6, with accuracy, TPR, FPR and the cost per applicant.\n"
       "2. Compute the AUC, the Gini, the accuracy ratio of the CAP curve and the KS statistic.\n"
       "3. Compute the Brier score, compare it with a constant PD of 30%, and run the Hosmer–Lemeshow test with five groups.\n"
       "4. Draw the ROC curve with the two cut-offs and the calibration plot.\n"
       "5. Interpretation: which cut-off would you recommend to the bank?\n\n"
       "**Report:** two confusion matrices, six statistics, the chart and two sentences."),
    code("# Solution\n" + src(s.b3_validation) + "\n\n\nb3 = b3_validation(save=False)\nprint(b3['cm50'])\nprint(b3['cm_cost'])\n"
         "{k: b3[k] for k in ['auc', 'gini', 'ar', 'ks', 'brier', 'brier_ref', 'hl', 'hl_p']}"),
    md("**Interpretation of the result.** The cut-off of 1/6: it rejects more applicants but catches 91% of the bads, and "
       "the cost per applicant falls from 0.867 to 0.507."),
    md("## B4 [Proposed]: logit or LDA on the Taiwan data?\n\n"
       "**Question.** Do the logit and Fisher's LDA differ in ranking or in calibration? Model: B3.\n\n"
       "1. Estimate a logit and an LDA on the training sample with the same inputs.\n"
       "2. Compute the test AUC of both and the DeLong test of equal AUCs.\n"
       "3. Compute the Brier score, the Hosmer–Lemeshow statistic (ten groups) and the mean PD of both, against the "
       "default rate.\n"
       "4. Compute the accuracy of the rule \"nobody defaults\" and of both models at the cut-off 0.5.\n"
       "5. Interpretation: is the LDA worse at ranking or at calibration?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nT = taiwan_model()\nb4 = logit_vs_lda_taiwan(T)\n{k: v for k, v in b4.items() if not isinstance(v, dict)}"),
    md("## B5 [Solved]: from PDs to capital\n\n"
       "**Question.** How much expected loss and Basel capital does the test portfolio of B1 carry, and how much does the "
       "prior correction matter?\n\n"
       "1. Correct the PD of every credit from the sample rate of 30% to the population rate of 5%.\n"
       "2. Compute the expected loss of the portfolio, in DM and as a share of EAD, with and without the correction.\n"
       "3. Compute the IRB capital K and the risk-weighted assets of the portfolio with the corrected PDs.\n"
       "4. Interpretation: why does the correction change the expected loss but not the AUC?\n\n"
       "**Report:** eight numbers and two sentences."),
    code("# Solution\n" + src(s.b5_capital) + "\n\n\nb5_capital()"),
    md("**Interpretation of the result.** The correction adds the same constant to every log-odds: the ranking, hence the "
       "AUC, is unchanged, while every PD shrinks and the expected loss falls from 16.3% to 4.4% of EAD."),
    md("## B6 [Proposed]: fairness on the Taiwan data\n\n"
       "**Question.** Does the logit of B4 treat men and women, and young and older clients, alike? Model: B5.\n\n"
       "1. Compute the approval rate and the default rate of men and women, and of clients up to 25 years and older.\n"
       "2. Compute the approval rate of the good payers (equal opportunity) in each group.\n"
       "3. Compute the default rate among the approved clients of each group.\n"
       "4. Interpretation: which fairness criterion does the model come closest to satisfying?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nfairness_check(T)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: does gradient boosting beat the logit?\n\n"
       "**Question.** Benchmarks report gains for flexible models (Lessmann et al., 2015); how large is the gain on the "
       "Taiwan data, and is it worth the loss of a simple scorecard? Models: B3, B4.\n\n"
       "1. Fit gradient boosting (scikit-learn, default settings) on the training sample.\n"
       "2. Compare its test AUC with the logit by the DeLong test, and compare the Brier scores.\n"
       "3. Compare the training and the test AUC of gradient boosting.\n"
       "4. Interpretation: is the gain large enough to replace the scorecard?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\nc1_benchmark(T)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised the B1 scorecard. It answered:\n\n"
       "- (a) the AUC is 0.80, so the scorecard classifies 80% of the applicants correctly;\n"
       "- (b) the Gini coefficient is AUC − 0.5 = 0.30;\n"
       "- (c) an applicant with a PD of 30% from this model has a 30% chance of default at the bank;\n"
       "- (d) with PDO = 20, a PD that halves from 20% to 10% adds exactly 20 points;\n"
       "- (e) KS is the largest vertical gap between the score distributions of bads and goods;\n"
       "- (f) LDA may only be used when every variable is Normally distributed.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 12, 'lecture')
    build(SEMINAR, 12, 'seminar')
