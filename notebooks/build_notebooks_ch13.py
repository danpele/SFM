"""
build_notebooks_ch13.py -- lecture and seminar notebooks of Chapter 13 (SFM): machine learning in finance
=========================================================================================================
Output: notebooks/EN/chapter13_lecture_notebook.ipynb, notebooks/EN/chapter13_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_13/generate_all_charts.py and seminar13.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Both notebooks install the arch package first (not part of Colab);
scikit-learn is part of Colab. Small models with fixed seeds: each notebook runs in a few minutes on a CPU.
Run:  python3 notebooks/build_notebooks_ch13.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter13_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter13_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 13
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(13)
import generate_all_charts as g   # noqa: E402
import seminar13 as s             # noqa: E402
import build_quantlets as bq      # noqa: E402

INSTALL = bq.INSTALL
CONSTS = '\n'.join(bq.CONSTS)
DATAF = [g.returns, g.fx_close, g.ohlc, g.daily_variance, g.rv_frame]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Statistics of Financial Markets — Chapter 13: Machine learning\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Bias and variance; validation for time series: walk-forward, purging, leakage.\n"
       "- Ridge and lasso; regression trees, random forest, gradient boosting; neural networks and the SFE Quantlets "
       "SFEnnarch and SFEnnjpyusd.\n"
       "- Applications: weekly realised volatility (HAR against machine learning), the sign of daily returns, VaR 1% by "
       "quantile regression; data snooping and the deflated Sharpe ratio; the case study of Gu, Kelly and Xiu (2020).\n"
       "- References: Franke, Härdle and Hafner (2019), *Statistics of Financial Markets*, 5th ed., Ch. 19; James et al. "
       "(2023), *An Introduction to Statistical Learning with Applications in Python*; Corsi (2009); Breiman (2001); "
       "Friedman (2001); Bailey and López de Prado (2014); Gu, Kelly and Xiu (2020)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- Daily variance proxy (Chapter 8): $v_t = o_t^2 + \\tfrac12(u_t - d_t)^2 - (2\\ln 2 - 1)c_t^2$ with $o_t$ the "
       "overnight log return and $u_t, d_t, c_t$ the log high, low and close relative to the open (all in %).\n"
       "- Volatility target: $y_t = \\ln(\\frac15\\sum_{j=1}^5 v_{t+j})$; HAR features: $\\ln v_t$ and the logs of the "
       "5-day and 22-day means of $v$.\n"
       "- Walk-forward: one test year at a time, an expanding training window, the last $h$ training rows purged.\n"
       "- VaR 1% $= -\\hat q_{0.01}$: the loss exceeded with probability 1%, a positive number."),
    code(CONSTS + '\n\n\n' + src(*DATAF)),
    md("## 1. Bias and variance of regression trees\n\n"
       "- 300 simulated training samples of $y = \\sin(2\\pi x) + \\varepsilon$; squared bias, variance and test error "
       "against the depth."),
    code(src(g.true_f, g.bias_variance, g.fig_bias_variance)),
    code("fig_bias_variance(save=False)"),
    md("## 2. Validation for time series\n\n"
       "- Random K-fold, walk-forward and purged K-fold; a random forest for the next 21-day return: shuffled folds "
       "leak through overlapping targets."),
    code(src(g.fig_cv_schemes, g.leakage_frame, g.oos_r2, g.cv_r2, g.leakage_experiment, g.fig_leakage)),
    code("fig_cv_schemes(save=False)"),
    code("fig_leakage(save=False)"),
    md("## 3. Ridge and lasso\n\n- Coefficient paths of 11 volatility features of the S&P 500, 2008–2012."),
    code(src(g.shrinkage_paths, g.fig_shrinkage)),
    code("fig_shrinkage(save=False)"),
    md("## 4. Trees, random forest and gradient boosting"),
    code(src(g.fig_tree, g.ensemble_curves, g.fig_ensembles)),
    code("fig_tree(save=False)"),
    code("fig_ensembles(save=False)"),
    md("## 5. Neural networks: an MLP, NN-ARCH (SFEnnarch) and the yen–dollar rate (SFEnnjpyusd)\n\n"
       "- Radial basis function networks with 3 units, as in the SFE Quantlets; GARCH(1,1) and the random walk as "
       "benchmarks."),
    code(src(g.fig_mlp, g.MLPEnsemble, g.rbf_fit, g.rbf_hidden, g.rbf_predict, g.nnarch_data, g.nnarch, g.fig_nnarch,
             g.lag_matrix, g.nnjpy, g.fig_nnjpy)),
    code("fig_mlp(save=False)"),
    code("fig_nnarch(save=False)"),
    code("fig_nnjpy(save=False)"),
    md("## 6. Forecasting realised volatility: HAR against machine learning\n\n"
       "- Walk-forward with yearly refits; out-of-sample $R^2$, QLIKE and Diebold–Mariano tests; S&P 500 from 2013, "
       "Bitcoin from 2018."),
    code(src(g.walk_forward, g.MeanModel, g.rv_models, g.qlike, g.dm_test, g.rv_forecasts, g.rv_metrics, g.wf_design,
             g.fig_rv_forecasts)),
    code("wf_design()['rows'][:2]"),
    code("RV = {}\nfor k in ('sp500', 'btc'):\n    X, P = rv_forecasts(k)\n    RV[k] = (X, P)\n"
         "    print(NAME[k])\n    print(pd.DataFrame({m: v for m, v in rv_metrics(X, P).items() if isinstance(v, dict)}).T.round(4))"),
    code("fig_rv_forecasts(*RV['sp500'], save=False)"),
    md("## 7. Feature importance: permutation and exact Shapley values"),
    code(src(g.shapley_values, g.importance, g.fig_importance)),
    code("fig_importance(save=False)"),
    md("## 8. The sign of tomorrow's return against the majority-class baseline"),
    code(src(g.sign_frame, g.sign_models, g.sign_walk_forward, g.sign_metrics, g.fig_sign)),
    code("SG = {k: sign_metrics(*sign_walk_forward(k)) for k in ('sp500', 'bet', 'btc')}\n"
         "pd.DataFrame({(k, m): SG[k][m] for k in SG for m in ('Logit', 'RF', 'GB')}).T.round(4)"),
    code("fig_sign(SG, save=False)"),
    md("## 9. VaR 1% by quantile regression and quantile boosting"),
    code(src(g.qvar_frame, g.quantile_models, g.garch_var, g.quantile_var, g.kupiec, g.christoffersen, g.var_backtest,
             g.fig_qvar)),
    code("QV = {}\nfor k in ('sp500', 'btc'):\n    X, Vf = quantile_var(k)\n    QV[k] = (X, Vf)\n    B = var_backtest(X, Vf)\n"
         "    print(NAME[k])\n    print(pd.DataFrame({m: B[m] for m in ('QR', 'QGB', 'HS', 'GARCH-t')}).T.round(4))"),
    code("fig_qvar(*QV['sp500'], save=False)"),
    md("## 10. Data snooping and the deflated Sharpe ratio"),
    code(src(g.expected_max_sr, g.max_sharpe_sim, g.ma_rules, g.rule_returns, g.sharpe, g.deflated_sharpe, g.snooping,
             g.fig_snooping)),
    code("M, SN = fig_snooping(save=False)\n{k: v for k, v in SN.items() if k != 'D'}"),
    md("## 11. Case study: Gu, Kelly and Xiu (2020), published results"),
    code(src(g.fig_gkx)),
    code("fig_gkx(save=False)"),
    md("## 12. AI for scientific discovery: machine learning for BVB volatility\n\n"
       "- Open question: do machine-learning volatility forecasts help on the Bucharest Stock Exchange?\n"
       "- Starter result: the weekly variance of the BET from squared returns, HAR against a random forest (no "
       "high and low prices for the BET in the course data).\n"
       "- Check before trusting any answer, yours or an AI's: time order, purging, scaling inside the training window, "
       "the S&P 500 features from the previous New York close, a benchmark and a test in every table."),
    code("b = returns('bet', '2008-01-01')\nv = (b ** 2).clip(lower=0.01 * float((b ** 2).median()))\n"
         "X = pd.DataFrame({'d': np.log(v), 'w': np.log(v.rolling(5).mean()), 'm': np.log(v.rolling(22).mean())})\n"
         "fut = sum(v.shift(-j) for j in range(1, 6)) / 5\nX['rv'], X['y'] = fut, np.log(fut)\nX = X.dropna()\n"
         "models = {'Mean': (HAR, lambda: MeanModel()), 'HAR': (HAR, lambda: LinearRegression()),\n"
         "          'RF': (HAR, lambda: RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=20, n_jobs=-1, random_state=SEED))}\n"
         "P = {}\nfor m, (f, make) in models.items():\n    p = walk_forward(X, f, make, '2012-01-01')\n"
         "    p['var'] = np.exp(p['pred'] + p['s2'] / 2)\n    P[m] = p\n"
         "pd.DataFrame({m: v for m, v in rv_metrics(X, P).items() if isinstance(v, dict)}).T.round(4)"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_HELPERS = [s.bias_variance_numbers, s.tree_split, s.ridge_1d, s.lasso_1d, s.walk_forward_design,
               s.purged_kfold_counts, s.b1_daily, s.b3_sign, s.b5_qr]

SEMINAR = [
    md("# Statistics of Financial Markets — Seminar 13: Machine learning\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- This seminar comes before Lecture 13: the slides \"What You Need for Today\" give every definition used here.\n"
       "- Part A: computations on paper, checked in code. Part B: real data with inference and an interpretation "
       "question. Part C: an open question and an AI answer to audit.\n"
       "- **[Solved]**: complete code and output, a model to follow. **[Proposed]**: write your own code in the empty cell."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n"
       "- Data: returns, the daily variance proxy, the volatility features and target.\n"
       "- Walk-forward forecasts, out-of-sample $R^2$, QLIKE and the Diebold–Mariano test; classifiers against the "
       "majority-class baseline; quantile regression VaR and its backtest.\n"
       "- Helpers for Part A and Part B: bias and variance, a tree split, ridge and lasso with one feature, the "
       "walk-forward design, the purged K-fold counts, the solved exercises B1, B3 and B5."),
    code(CONSTS + '\n\n\n' + src(*DATAF) + '\n\n\n' +
         src(g.MLPEnsemble, g.walk_forward, g.MeanModel, g.rv_models, g.qlike, g.dm_test, g.oos_r2, g.rv_forecasts,
             g.rv_metrics, g.wf_design, g.sign_frame, g.sign_models, g.sign_walk_forward, g.sign_metrics, g.qvar_frame,
             g.quantile_models, g.kupiec, g.christoffersen, g.var_backtest, g.leakage_frame, g.cv_r2,
             g.leakage_experiment) + '\n\n\n' + src(*SEM_HELPERS)),
    code("# [solutions only]\n" + src(s.b2_dax, s.b4_sign, s.b6_qgb, s.c1_bet_spx, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: bias and variance of a shallow tree\n\n"
       "**Context.** At a point $x_0$ the true value is $f(x_0) = 2.0$ and $\\sigma^2 = 0.25$. A depth-1 tree fitted on "
       "five training sets forecasts 1.6, 1.7, 1.5, 1.8, 1.6.\n\n"
       "1. Compute the mean forecast and the bias.\n"
       "2. Compute the variance of the five forecasts (divide by 5).\n"
       "3. Compute the expected squared error at $x_0$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nbias_variance_numbers([1.6, 1.7, 1.5, 1.8, 1.6], 2.0, 0.25)"),
    md("**Interpretation of the result.** The shallow tree is stable but systematically too low: its error is mostly "
       "squared bias."),
    md("## A2 [Proposed]: bias and variance of a deep tree\n\n"
       "**Context.** Same point and noise as in A1; a depth-10 tree forecasts 2.6, 1.4, 2.3, 1.7, 2.2. Model: A1.\n\n"
       "1. Compute the mean forecast, the bias and the variance.\n"
       "2. Compute the expected squared error.\n"
       "3. Decide which tree you would use at $x_0$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nbias_variance_numbers([2.6, 1.4, 2.3, 1.7, 2.2], 2.0, 0.25)"),
    md("## A3 [Solved]: one split of a regression tree\n\n"
       "**Context.** Eight days: $x$ = yesterday's absolute return (%), $y$ = today's variance proxy; "
       "$x$: 0.2, 0.4, 0.5, 0.9, 1.1, 1.6, 2.0, 2.8; $y$: 0.5, 0.7, 0.6, 1.0, 1.4, 2.2, 2.6, 3.4.\n\n"
       "1. Compute the mean of $y$ and the SSE without a split.\n"
       "2. For each candidate split, compute both leaf means and the total SSE.\n"
       "3. Choose the best split and the share of the SSE it removes.\n\n"
       "**Report:** the table, the split and one sentence."),
    code("# Solution\nt = tree_split([0.2, 0.4, 0.5, 0.9, 1.1, 1.6, 2.0, 2.8], [0.5, 0.7, 0.6, 1.0, 1.4, 2.2, 2.6, 3.4])\n"
         "print('mean', t['mean'], 'SSE without a split', t['sse0'])\npd.DataFrame(t['rows']).round(3)"),
    md("**Interpretation of the result.** The best split is at 1.35: large moves yesterday announce a high variance today; "
       "one split removes about 84% of the SSE."),
    md("## A4 [Proposed]: a split on the volatility index\n\n"
       "**Context.** Eight months: $x$ = the VIX level at the end of the month, $y$ = next month's realised volatility; "
       "$x$: 12, 14, 15, 17, 21, 24, 30, 38; $y$: 0.8, 0.9, 0.7, 1.1, 1.5, 1.4, 2.6, 3.0. Model: A3.\n\n"
       "1. Compute the SSE without a split and for every candidate split.\n"
       "2. Choose the best split and give both leaf forecasts.\n"
       "3. Compute the share of the SSE that the split removes.\n\n"
       "**Report:** the table, three numbers and one sentence."),
    code("# Solution\nt = tree_split([12, 14, 15, 17, 21, 24, 30, 38], [0.8, 0.9, 0.7, 1.1, 1.5, 1.4, 2.6, 3.0])\n"
         "print(t['best'], 1 - t['best']['sse'] / t['sse0'])\npd.DataFrame(t['rows']).round(3)"),
    md("## A5 [Solved]: ridge with one feature\n\n"
       "**Context.** One centred feature with $S_{xx} = 50$ and $S_{xy} = 20$; no intercept.\n\n"
       "1. Compute the OLS coefficient.\n"
       "2. Compute the ridge coefficient and the shrinkage factor $S_{xx}/(S_{xx} + \\lambda)$ for $\\lambda = 10$ and $\\lambda = 50$.\n"
       "3. Explain what happens when $\\lambda \\to \\infty$.\n\n"
       "**Report:** five numbers and one sentence."),
    code("# Solution\nridge_1d(50.0, 20.0, [10, 50])"),
    md("**Interpretation of the result.** Ridge multiplies the OLS coefficient by a factor below 1: it shrinks, it never "
       "sets the coefficient to zero."),
    md("## A6 [Proposed]: lasso with one feature\n\n"
       "**Context.** Same data as in A5. Model: A5.\n\n"
       "1. Compute the lasso coefficient for $\\lambda = 10$, 30 and 50.\n"
       "2. Find the smallest $\\lambda$ for which the coefficient is exactly zero.\n"
       "3. Compare the lasso with the ridge results of A5 for $\\lambda = 50$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\nlasso_1d(50.0, 20.0, [10, 30, 50])"),
    md("## A7 [Solved]: a walk-forward design\n\n"
       "**Context.** Daily S&P 500 features from 2008 to 18 September 2026; target: the mean variance of the next 5 days; "
       "refits each January from 2013.\n\n"
       "1. Count the folds of the walk-forward evaluation.\n"
       "2. For the 2013 fold, say which training days must be purged and why.\n"
       "3. Explain why no embargo is needed.\n"
       "4. Explain what would go wrong if the 2013 model were tuned by random 5-fold on 2008–2012.\n\n"
       "**Report:** two numbers and three sentences."),
    code("# Solution\nd = walk_forward_design()\nprint('folds:', d['n_folds'], '| first day:', d['first'])\npd.DataFrame(d['rows']).head(3)"),
    md("**Interpretation of the result.** 14 folds; the last 5 training days of 2012 are purged because their targets "
       "use January 2013; the test year always comes after the training days, so no embargo is needed; random folds "
       "would reward memorising neighbouring days."),
    md("## A8 [Proposed]: a purged K-fold with an embargo\n\n"
       "**Context.** 2500 days, $K = 5$ blocks in time order, a target that uses the next 21 days, an embargo of 1% of "
       "the days. Model: A7.\n\n"
       "1. Compute the size of each test block and of the embargo.\n"
       "2. Compute the number of training days when the test block is in the middle.\n"
       "3. Compute it when the test block is the first and when it is the last block.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\npurged_kfold_counts()"),
    # ---------------- Part B
    md("# Part B: real data, inference and interpretation"),
    md("## B1 [Solved]: next-day volatility of the S&P 500\n\n"
       "**Question.** Does a random forest forecast tomorrow's variance of the S&P 500 better than HAR?\n\n"
       "1. Fit HAR (OLS) and a random forest (300 trees, at least 20 days per leaf) walk-forward, refitting each January from 2013.\n"
       "2. Compute the out-of-sample $R^2$ of both against the training mean and of the forest against HAR.\n"
       "3. Compute the mean QLIKE of both and the Diebold–Mariano test of equal QLIKE.\n"
       "4. Draw the QLIKE difference per year.\n"
       "5. Interpretation: is the forest worth its complexity here?\n\n"
       "**Report:** four numbers, a test with its p-value, the chart and two sentences."),
    code("# Solution\nb1 = b1_daily('sp500', save=False)\n{m: b1[m] for m in ('HAR', 'RF')}"),
    md("**Interpretation of the result.** No: with the same three features the forest is significantly worse than the "
       "linear HAR; the relation is close to linear in logs and the forest only adds variance."),
    md("## B2 [Proposed]: weekly volatility of the DAX\n\n"
       "**Question.** Do lasso, random forest, gradient boosting or an MLP beat HAR for next week's DAX variance? Model: B1.\n\n"
       "1. Fit the five models walk-forward from 2012, purging the last 5 training days.\n"
       "2. Compute the out-of-sample $R^2$ against HAR and the mean QLIKE of each model.\n"
       "3. Run the Diebold–Mariano test of each model against HAR.\n"
       "4. Interpretation: which model would you use for the DAX?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb2 = b2_dax()\npd.DataFrame({m: v for m, v in b2.items() if isinstance(v, dict)}).T.round(4)"),
    md("## B3 [Solved]: the sign of DAX returns\n\n"
       "**Question.** Can a logit or a random forest predict whether the DAX rises tomorrow better than \"always up\"?\n\n"
       "1. Fit both classifiers walk-forward from 2008, refitting each January.\n"
       "2. Compute the accuracy of each model and of the majority-class baseline.\n"
       "3. Test each accuracy against the baseline with the $z$ statistic, and compute the AUC.\n"
       "4. Draw the accuracy by year.\n"
       "5. Interpretation: does either model have skill?\n\n"
       "**Report:** six numbers, the chart and two sentences."),
    code("# Solution\nb3 = b3_sign('dax', save=False)\nb3"),
    md("**Interpretation of the result.** No: both accuracies are at or below the baseline and the AUC is close to 0.5; "
       "daily DAX returns are close to unpredictable (Chapter 7)."),
    md("## B4 [Proposed]: the sign of two BVB stocks\n\n"
       "**Question.** Is the sign of Banca Transilvania and OMV Petrom returns easier to predict than that of the DAX? Model: B3.\n\n"
       "1. Fit the logit, random forest and gradient boosting walk-forward from 2014.\n"
       "2. Compute the accuracy, the baseline accuracy, the $z$ statistic and the AUC for each stock and model.\n"
       "3. Interpretation: is there more predictability on the BVB than on the DAX?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb4 = b4_sign()\npd.DataFrame({(k, m): b4[k][m] for k in b4 for m in ('Logit', 'RF', 'GB')}).T.round(4)"),
    md("## B5 [Solved]: VaR 1% of the DAX by quantile regression\n\n"
       "**Question.** Does a linear quantile regression on volatility features give a better VaR 1% than historical simulation?\n\n"
       "1. Fit the 1% quantile regression walk-forward from 2012 and set VaR 1% $= -\\hat q_{0.01}$.\n"
       "2. Compute the historical-simulation VaR 1% on the last 500 days.\n"
       "3. Count the exceptions and run the Kupiec and Christoffersen tests; compute the average quantile loss.\n"
       "4. Draw both VaR forecasts in 2020–2022.\n"
       "5. Interpretation: which VaR would you report to a risk committee?\n\n"
       "**Report:** a table, the chart and two sentences."),
    code("# Solution\nb5 = b5_qr('dax', save=False)\nprint(b5['coef'], b5['intercept'])\npd.DataFrame({m: b5[m] for m in ('QR', 'HS')}).T.round(4)"),
    md("**Interpretation of the result.** The quantile regression has the right exception rate, independent exceptions "
       "and a smaller quantile loss; historical simulation is exceeded in clusters because a 500-day window reacts slowly."),
    md("## B6 [Proposed]: quantile gradient boosting for the DAX\n\n"
       "**Question.** Does a flexible quantile model improve on the linear quantile regression of B5? Model: B5.\n\n"
       "1. Fit the quantile boosting (300 trees of depth 2, learning rate 0.05, at least 50 days per leaf) walk-forward "
       "from 2012 and compute its VaR 1%.\n"
       "2. Count the exceptions, run the Kupiec and Christoffersen tests and compute the quantile loss.\n"
       "3. Interpretation: is the flexible model worth it at the 1% level?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6 = b6_qgb()\npd.DataFrame({m: b6[m] for m in ('QR', 'QGB')}).T.round(4)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: does the S&P 500 help to forecast BET volatility?\n\n"
       "**Question.** The BVB closes before New York; does yesterday's S&P 500 volatility improve a HAR forecast of next "
       "week's BET variance? Models: B1, B2.\n\n"
       "1. Build the HAR features of the BET (squared daily returns as the proxy) and the HAR features of the S&P 500 at "
       "the previous New York close.\n"
       "2. Compare HAR, HAR plus the S&P 500 features (OLS) and a random forest on all features, walk-forward from 2012.\n"
       "3. Compute the out-of-sample $R^2$, QLIKE and the Diebold–Mariano tests against HAR.\n"
       "4. Interpretation: is the S&P 500 information worth adding?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\nc1 = c1_bet_spx()\npd.DataFrame({m: v for m, v in c1.items() if isinstance(v, dict)}).T.round(4)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant summarised machine learning for the S&P 500. It answered:\n\n"
       "- (a) with random 5-fold cross-validation a random forest explains more than a third of the next 21-day return: "
       "the market is predictable;\n"
       "- (b) the forest fits next week's log variance better than HAR, judging by the in-sample $R^2$;\n"
       "- (c) a forest that predicts the sign of tomorrow's return correctly on 53% of the days beats a coin, so it has skill;\n"
       "- (d) for the VaR 99% use quantile regression at $\\tau = 0.99$ on the next-day return;\n"
       "- (e) the lasso removes the features whose t-statistics are not significant;\n"
       "- (f) QLIKE punishes an under-forecast of the variance more than an over-forecast by the same factor.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 13, 'lecture')
    build(SEMINAR, 13, 'seminar')
