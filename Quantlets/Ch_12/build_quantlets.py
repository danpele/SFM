"""
build_quantlets.py -- Quantlet folders of Chapter 12 (SFM): scoring models
=========================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The credit data are read from data/credit of the SFM repository (local copy or raw GitHub URL).
Run:  python3 Quantlets/Ch_12/generate_all_charts.py && python3 Quantlets/Ch_12/seminar12.py
      python3 Quantlets/Ch_12/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 12
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'notebooks'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Saturday, 3 October 2026'
DATA = ('South German Credit (Groemping, 2019; UCI Machine Learning Repository, doi:10.24432/C5QG88, CC BY 4.0) and '
        'Default of Credit Card Clients (Yeh and Lien, 2009; UCI, doi:10.24432/C55S3H, CC BY 4.0), data/credit of the '
        'SFM repository')
CONSTS = [f'CREDIT_RAW = {g.CREDIT_RAW!r}', f'FILES = {g.FILES!r}', f'SEED = {g.SEED!r}', f'TEST_SHARE = {g.TEST_SHARE!r}',
          f'POP_BAD = {g.POP_BAD!r}', f'COST_FN = {g.COST_FN!r}', f'COST_FP = {g.COST_FP!r}',
          f'PDO, BASE_SCORE, BASE_ODDS = {g.PDO!r}, {g.BASE_SCORE!r}, {g.BASE_ODDS!r}', f'IV_MIN = {g.IV_MIN!r}',
          f'N_REPEAT, N_FOLD = {g.N_REPEAT!r}, {g.N_FOLD!r}', f'BINS = {g.BINS!r}'.replace('inf', 'np.inf'),
          f'NUMERIC = {g.NUMERIC!r}', f'SGC_VARS = {g.SGC_VARS!r}', f'STATUS = {g.STATUS!r}',
          f'TW_FEATURES = {g.TW_FEATURES!r}', f'COLORS = {g.COLORS!r}']
DATAF = [g.load_credit, g.taiwan_features, g.stratified_split]
WOE = [g.bin_values, g.woe_table, g.iv_all, g.iv_strength, g.woe_transform]
MODELS = [g.logit_fit, g.logit_predict, g.lda_fit, g.lda_predict]
VALID = [g.auc, g.delong, g.roc_points, g.cap_points, g.accuracy_ratio, g.ks_stat, g.brier, g.calibration_table,
         g.confusion]
SCORE = [g.scaling, g.points_table, g.score_from_pd, g.prior_shift, g.sgc_scorecard]

QUANTLETS = [
    dict(name='SFM_ch12_woe_iv',
         desc='Default rate by the status of the checking account; weight of evidence (WoE = ln of the share of goods '
              'over the share of bads in a bin) of the credit duration in five fixed bins; information value (IV) of the '
              '20 variables of the South German Credit data (the corrected version of the UCI German credit data), '
              'with the rule-of-thumb classes of Siddiqi (2006).',
         keywords='credit scoring, weight of evidence, WoE, information value, IV, binning, South German Credit, '
                  'default rate, class imbalance',
         consts=CONSTS, funcs=DATAF + WOE + [g.fig_default_by_status, g.fig_woe_duration, g.fig_iv],
         run="df = load_credit('sgc')\nprint(fig_default_by_status(df))\nprint(woe_table(df['status'], df['default'], 'status').round(3))\n"
             "print(fig_woe_duration(df))\nprint(pd.Series(fig_iv(df)).round(3))",
         charts=['sfm_ch12_default_by_status', 'sfm_ch12_woe_duration', 'sfm_ch12_iv'], extra=['ch12_iv_table.csv']),
    dict(name='SFM_ch12_logit_lda',
         desc='Logistic regression by maximum likelihood (Newton-Raphson) with standard errors, odds ratios and 95% '
              'confidence intervals; a logistic curve of the PD against the credit duration; Fisher linear discriminant '
              'analysis (w = S_W^-1 (m_1 - m_0)) against the logit for duration and age, with both boundaries at the '
              'sample default rate.',
         keywords='logistic regression, logit, odds, log-odds, odds ratio, maximum likelihood, linear discriminant '
                  'analysis, LDA, Fisher, probability of default, PD',
         consts=CONSTS, funcs=DATAF + MODELS + [g.small_logit, g.fig_logistic_curve, g.fig_odds_ratios, g.fig_lda_2d],
         run="df = load_credit('sgc')\nprint(fig_logistic_curve(df))\nL = small_logit(df)\nprint(L['table'].round(3))\n"
             "fig_odds_ratios(L)\nprint(fig_lda_2d(df))",
         charts=['sfm_ch12_logistic_curve', 'sfm_ch12_odds_ratios', 'sfm_ch12_lda_2d']),
    dict(name='SFM_ch12_scorecard',
         desc='A WoE logit scorecard for the South German Credit data: stratified 70/30 split, WoE and IV on the training '
              'sample, variables with IV >= 0.10, logit and LDA on the WoE values; scaling with 20 points to double the '
              'odds and 600 points at good:bad odds of 50:1; the points of every attribute; prior correction from the '
              '30% sample to a 5% population; the scores of the test sample.',
         keywords='scorecard, points to double the odds, PDO, scaling, weight of evidence, logistic regression, '
                  'oversampling, prior correction, credit scoring',
         consts=CONSTS, funcs=DATAF + WOE + MODELS + SCORE + [g.fig_score_dist],
         run="S = sgc_scorecard()\nprint(S['sel'])\nprint(pd.Series(S['logit']['b'], index=['const'] + S['sel']).round(3))\n"
             "print(points_table(S['logit'], S['tables'], S['sel']).round(2))\nprint(scaling(), prior_shift(0.3, POP_BAD))\n"
             "print(fig_score_dist(S))",
         charts=['sfm_ch12_score_dist'], extra=['ch12_points_table.csv']),
    dict(name='SFM_ch12_validation_irb',
         desc='Validation of the scorecard on the test sample: confusion matrix at the cut-offs 0.5 and 1/6 (costs 5:1), '
              'expected cost against the cut-off, ROC curves and AUC with DeLong standard errors and tests, Gini, CAP '
              'curve and accuracy ratio, Kolmogorov-Smirnov statistic, Brier score, Hosmer-Lemeshow test, repeated '
              'stratified cross-validation with WoE inside every fold; calibration of a logit on the Taiwan credit card '
              'data; expected loss and Basel II IRB capital of retail exposures as functions of PD; the conditional PD of '
              'the one-factor Gaussian model in a bad, typical and good state of the economy (as in SFEdefaproba).',
         keywords='validation, confusion matrix, ROC, AUC, DeLong test, Gini, CAP, accuracy ratio, KS, Brier score, '
                  'calibration, Hosmer-Lemeshow, cross-validation, expected loss, Basel IRB, capital',
         consts=CONSTS, funcs=DATAF + WOE + MODELS + VALID + SCORE + [
             g.small_logit, g.cv_auc, g.validation, g.fig_roc, g.fig_cap, g.fig_ks, g.fig_cost_cutoff, g.fig_cv_auc,
             g.taiwan_model, g.fig_calibration, g.irb_capital, g.fig_irb, g.conditional_pd, g.fig_conditional_pd],
         run="S = sgc_scorecard()\nV = validation(S)\nprint({k: v for k, v in V.items() if not isinstance(v, dict)})\n"
             "print(V['cm50'], V['cm_cost'])\nfig_roc(S)\nfig_cap(S)\nprint(fig_ks(S))\nprint(fig_cost_cutoff(S))\n"
             "res = cv_auc()\nprint(fig_cv_auc(res))\nT = taiwan_model()\ncal = fig_calibration(T)\n"
             "print({k: v for k, v in cal.items() if k != 'groups'})\nfig_irb()\nprint(fig_conditional_pd())",
         charts=['sfm_ch12_roc', 'sfm_ch12_cap', 'sfm_ch12_ks', 'sfm_ch12_cost_cutoff', 'sfm_ch12_cv_auc',
                 'sfm_ch12_calibration', 'sfm_ch12_irb', 'sfm_ch12_conditional_pd']),
    dict(name='SFM_ch12_reject_fairness',
         desc='Taiwan credit card data: a reject-inference experiment (an old score accepts the best 70%; a new logit '
              'trained on the accepted only against one trained on all applicants, both evaluated on all test '
              'applicants) and group fairness at a 75% approval cut-off (approval rates, approval of good payers, '
              'default rates and mean PDs by sex and age group; sex and age are not model inputs).',
         keywords='reject inference, sample selection, fairness, demographic parity, equal opportunity, calibration '
                  'within groups, credit card default, Taiwan',
         consts=CONSTS, funcs=DATAF + MODELS + VALID + [g.taiwan_model, g.reject_inference_sim, g.fairness_table,
                                                         g.fig_fairness],
         run="T = taiwan_model()\nprint(reject_inference_sim(T))\nF = fairness_table(T)\n"
             "print(pd.DataFrame(F['groups']).T.round(3))\nfig_fairness(F)",
         charts=['sfm_ch12_fairness']),
    dict(name='SFM_ch12_altman_merton',
         desc='The Altman (1968) Z-score with its zones; the Merton (1974) distance to default and PD for a firm whose '
              'assets follow a geometric Brownian motion; the asset value and volatility solved from the equity value and '
              'volatility; simulated asset paths and PD against leverage for three asset volatilities.',
         keywords='Altman Z-score, discriminant analysis, Merton model, distance to default, geometric Brownian motion, '
                  'structural credit model, leverage, probability of default',
         consts=CONSTS, funcs=[g.altman_z, g.merton_dd, g.merton_from_equity, g.fig_merton],
         run="print(altman_z(0.15, 0.20, 0.08, 0.90, 1.10))\nprint(merton_dd(100, 70, 0.25, 0.05, 1.0))\n"
             "print(merton_from_equity(40, 0.5, 70, 0.03, 1.0))\nprint(fig_merton())",
         charts=['sfm_ch12_merton']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar12 as s
    QUANTLETS.append(dict(
        name='SFM_ch12_seminar',
        desc='Seminar 12 of Statistics of Financial Markets: the PD of an applicant from a logit model, WoE and IV from '
             'a table of counts, the confusion matrix, AUC by counting pairs, KS and Brier score of a small sample, '
             'scorecard scaling and points, on paper; a WoE logit scorecard for the South German Credit data with its '
             'validation; the expected loss and Basel IRB capital of the test portfolio.',
        keywords='seminar, credit scoring, logit, odds ratio, weight of evidence, information value, AUC, Gini, KS, '
                 'Brier score, scorecard, expected loss, IRB capital',
        consts=CONSTS + [f'A5 = {s.A5!r}', f'A6 = {s.A6!r}', f'SMALL = {s.SMALL!r}'],
        funcs=DATAF + WOE + MODELS + VALID + SCORE + [g.irb_capital, g.taiwan_model,
                                                      s.logit_applicant, s.woe_by_hand, s.category_counts,
                                                      s.small_sample_metrics, s.scaling_example, s.attribute_points,
                                                      s.prior_and_el, s.b1_scorecard, s.b3_validation, s.b5_capital,
                                                      s.taiwan_iv, s.logit_vs_lda_taiwan, s.fairness_check, s.c1_benchmark,
                                                      s.c2_check],
        run="print(logit_applicant(2, 3, 3, 1, 0))\nc = category_counts('status')\nprint(woe_by_hand(c['good'], c['bad']))\n"
            "print(small_sample_metrics(A5['pd'], A5['y'], 0.5))\nprint(scaling_example())\n"
            "b1 = b1_scorecard()\nprint({k: b1[k] for k in ['sel', 'auc_train', 'auc', 'lo', 'hi']})\n"
            "b3 = b3_validation()\nprint({k: b3[k] for k in ['auc', 'gini', 'ks', 'brier', 'hl', 'hl_p']})\nprint(b5_capital())",
        charts=['ch12_sem_b1', 'ch12_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 12, 'Scoring models', HERE, data=DATA, submitted=SUBMITTED)
