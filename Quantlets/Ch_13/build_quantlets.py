"""
build_quantlets.py -- Quantlet folders of Chapter 13 (SFM): machine learning in finance
=======================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The notebooks install the arch package first (pip install arch), as Google Colab does not include it; scikit-learn is
part of Colab. Small models with fixed seeds: each notebook runs in a few minutes on a CPU.
Run:  python3 Quantlets/Ch_13/generate_all_charts.py && python3 Quantlets/Ch_13/seminar13.py
      python3 Quantlets/Ch_13/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 13
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
INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
DATA = ('Daily market data from EODHD (S&P 500 and DAX open, high, low and close; BET, Bitcoin, Banca Transilvania, '
        'OMV Petrom; GBP/USD, USD/JPY, the German 10-year yield and gold), data/market of the SFM repository')
SK = """from matplotlib.patches import Rectangle, Circle
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, LogisticRegression, QuantileRegressor, Ridge,
                                  lasso_path)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from arch import arch_model
TABLE_DIR = '.'"""
CONSTS = [SK, f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'OHLC_START = {g.OHLC_START!r}', f'BAD_DAYS = {g.BAD_DAYS!r}',
          f'COLORS = {g.COLORS!r}', f'MCOL = {g.MCOL!r}', f'H = {g.H!r}', f'OOS = {g.OOS!r}', f'OOS_SIGN = {g.OOS_SIGN!r}',
          f'HAR = {g.HAR!r}', f'EXT = {g.EXT!r}', f'FEAT_LABEL = {g.FEAT_LABEL!r}', f'FEAT_COL = {g.FEAT_COL!r}',
          f'SIGN_FEATS = {g.SIGN_FEATS!r}', f'QV_FEATS = {g.QV_FEATS!r}', f'ALPHA_VAR = {g.ALPHA_VAR!r}',
          f'HS_WINDOW = {g.HS_WINDOW!r}', f'SEED = {g.SEED!r}', f'N_TREES = {g.N_TREES!r}', f'EULER = {g.EULER!r}',
          f'GKX_MODELS = {g.GKX_MODELS!r}', f'GKX_R2 = {g.GKX_R2!r}', f'GKX_R2_TOP = {g.GKX_R2_TOP!r}', f'GKX_SR = {g.GKX_SR!r}']
DATAF = [g.returns, g.fx_close, g.ohlc, g.daily_variance, g.rv_frame]
RV = [g.MLPEnsemble, g.walk_forward, g.MeanModel, g.rv_models, g.qlike, g.dm_test, g.oos_r2, g.rv_forecasts, g.rv_metrics]
SIGN = [g.sign_frame, g.sign_models, g.sign_walk_forward, g.sign_metrics]
QV = [g.qvar_frame, g.quantile_models, g.garch_var, g.quantile_var, g.kupiec, g.christoffersen, g.var_backtest]

QUANTLETS = [
    dict(name='SFM_ch13_validation',
         desc='Bias and variance of regression trees of depth 1 to 10 on 300 simulated training samples '
              '(y = sin(2 pi x) + noise): squared bias, variance, expected test error and training error. A diagram of '
              'random K-fold, walk-forward and purged K-fold with an embargo. Leakage: a random forest for the next 21-day '
              'return, validated by shuffled 5-fold, walk-forward and purged walk-forward, on a simulated random walk and '
              'on the S&P 500 since 2000.',
         keywords='machine learning, overfitting, bias-variance trade-off, regression tree, cross-validation, walk-forward, '
                  'purging, embargo, look-ahead bias, leakage, random forest, out-of-sample R2, S&P 500',
         consts=CONSTS, funcs=[g.returns, g.true_f, g.bias_variance, g.fig_bias_variance, g.fig_cv_schemes, g.leakage_frame,
                               g.oos_r2, g.cv_r2, g.leakage_experiment, g.fig_leakage],
         run="print(fig_bias_variance())\nfig_cv_schemes()\nprint(fig_leakage())",
         charts=['sfm_ch13_bias_variance', 'sfm_ch13_cv_schemes', 'sfm_ch13_leakage']),
    dict(name='SFM_ch13_shrinkage_trees',
         desc='Weekly log realised variance of the S&P 500 (daily range proxy: overnight return squared plus the '
              'Garman-Klass term) with 11 features: ridge and lasso coefficient paths, the lasso penalty chosen by '
              'walk-forward cross-validation; a regression tree of depth 2; validation error of a single tree and a random '
              'forest against depth, and of gradient boosting against the number of trees for three learning rates.',
         keywords='ridge regression, lasso, shrinkage, regression tree, random forest, gradient boosting, learning rate, '
                  'realised variance, Garman-Klass, HAR, S&P 500',
         consts=CONSTS, funcs=DATAF + [g.shrinkage_paths, g.fig_shrinkage, g.fig_tree, g.ensemble_curves, g.fig_ensembles],
         run="print(fig_shrinkage())\nprint(fig_tree())\nprint(fig_ensembles())",
         charts=['sfm_ch13_shrinkage', 'sfm_ch13_tree', 'sfm_ch13_ensembles']),
    dict(name='SFM_ch13_neural_networks',
         desc='A multilayer perceptron and its activation functions. Neural-network ARCH as in the SFE Quantlet SFEnnarch: '
              'two radial basis function networks with 3 units for the conditional mean and the conditional variance of '
              'GBP/USD daily returns (inputs: 3 lagged returns, the German 10-year yield, the gold return), against '
              'GARCH(1,1). The yen-dollar rate as in SFEnnjpyusd: an RBF network on the last 3 levels, 80% training and '
              '20% test, against the random walk; the same network on returns.',
         keywords='neural network, multilayer perceptron, activation function, radial basis function network, NN-ARCH, '
                  'GARCH, exchange rate, GBP/USD, USD/JPY, random walk, SFEnnarch, SFEnnjpyusd',
         consts=CONSTS, funcs=[g.returns, g.fx_close, g.oos_r2, g.fig_mlp, g.MLPEnsemble, g.rbf_fit, g.rbf_hidden,
                               g.rbf_predict, g.nnarch_data, g.nnarch, g.fig_nnarch, g.lag_matrix, g.nnjpy, g.fig_nnjpy],
         run="fig_mlp()\nprint(fig_nnarch())\nprint(fig_nnjpy())",
         charts=['sfm_ch13_mlp', 'sfm_ch13_nnarch', 'sfm_ch13_nnjpy']),
    dict(name='SFM_ch13_volatility_forecast',
         desc='Forecasting the weekly realised variance of the S&P 500 (from 2013) and Bitcoin (from 2018): HAR (Corsi, '
              '2009) against lasso, random forest, gradient boosting and an ensemble of small MLPs, walk-forward with yearly '
              'refits and purging; out-of-sample R2 against the mean and against HAR, QLIKE, Diebold-Mariano tests; '
              'forecasts in 2020 and 2025; permutation importance and exact Shapley values of a random forest.',
         keywords='realised volatility, HAR model, lasso, random forest, gradient boosting, neural network, walk-forward, '
                  'out-of-sample R2, QLIKE, Diebold-Mariano test, permutation importance, Shapley values, SHAP',
         consts=CONSTS, funcs=DATAF + RV + [g.wf_design, g.fig_rv_forecasts, g.shapley_values, g.importance,
                                            g.fig_importance],
         run="print(wf_design())\nfor k in ('sp500', 'btc'):\n    X, P = rv_forecasts(k)\n"
             "    print(NAME[k], pd.DataFrame({m: v for m, v in rv_metrics(X, P).items() if isinstance(v, dict)}).round(4))\n"
             "    if k == 'sp500':\n        fig_rv_forecasts(X, P, k)\nprint(fig_importance())",
         charts=['sfm_ch13_rv_forecasts', 'sfm_ch13_importance'], extra=['ch13_rv_table.csv']),
    dict(name='SFM_ch13_sign_prediction',
         desc='The sign of the next daily return of the S&P 500, the BET (from 2008) and Bitcoin (from 2018): logit, '
              'random forest and gradient boosting classifiers on past returns and volatilities, walk-forward with yearly '
              'refits, against the majority-class baseline; accuracy, z test against the baseline, AUC.',
         keywords='classification, sign prediction, logit, random forest, gradient boosting, accuracy, baseline, AUC, '
                  'market efficiency, S&P 500, BET, Bitcoin',
         consts=CONSTS, funcs=[g.returns] + SIGN + [g.fig_sign],
         run="S = {k: sign_metrics(*sign_walk_forward(k)) for k in ('sp500', 'bet', 'btc')}\nprint(S)\nfig_sign(S)",
         charts=['sfm_ch13_sign']),
    dict(name='SFM_ch13_quantile_var',
         desc='One-day VaR 1% (the loss exceeded with probability 1%) of the S&P 500 (from 2013) and Bitcoin (from 2018) '
              'by linear quantile regression and quantile gradient boosting on volatility features, against historical '
              'simulation (500 days) and GARCH(1,1)-t re-estimated each January; exceedances, the Kupiec and '
              'Christoffersen tests and the average quantile loss; forecasts in 2020 and 2025.',
         keywords='value at risk, VaR, quantile regression, pinball loss, quantile gradient boosting, historical simulation, '
                  'GARCH, backtesting, Kupiec test, Christoffersen test, S&P 500, Bitcoin',
         consts=CONSTS, funcs=DATAF + [g.walk_forward] + QV + [g.fig_qvar],
         run="for k in ('sp500', 'btc'):\n    X, V = quantile_var(k)\n    print(NAME[k], var_backtest(X, V))\n"
             "    if k == 'sp500':\n        fig_qvar(X, V, k)",
         charts=['sfm_ch13_qvar']),
    dict(name='SFM_ch13_data_snooping',
         desc='Data snooping: the best annual Sharpe ratio among N strategies without skill (simulation and the expected '
              'maximum formula of Bailey and Lopez de Prado, 2014); 30 moving-average rules on the S&P 500, in sample '
              '2000-2014 against 2015-2026; the probabilistic and the deflated Sharpe ratio of the best rule.',
         keywords='data snooping, selection bias, multiple testing, Sharpe ratio, deflated Sharpe ratio, probabilistic '
                  'Sharpe ratio, moving average, backtest overfitting, S&P 500',
         consts=CONSTS, funcs=[g.returns, g.expected_max_sr, g.max_sharpe_sim, g.ma_rules, g.rule_returns, g.sharpe,
                               g.deflated_sharpe, g.snooping, g.fig_snooping],
         run="M, S = fig_snooping()\nprint(S['D'].round(3))\nprint({k: v for k, v in S.items() if k != 'D'})",
         charts=['sfm_ch13_snooping'], extra=['ch13_ma_rules_table.csv']),
    dict(name='SFM_ch13_case_gkx',
         desc='Case study: Gu, Kelly and Xiu (2020), Empirical asset pricing via machine learning, Review of Financial '
              'Studies 33(5). The published monthly out-of-sample R2 of 12 models (Table 1, all stocks and the 1,000 '
              'largest) and the annualised Sharpe ratio of their value-weighted decile spread portfolios (Table 7).',
         keywords='empirical asset pricing, machine learning, neural networks, random forest, out-of-sample R2, '
                  'cross-section of stock returns, Sharpe ratio, Gu Kelly Xiu',
         consts=CONSTS, funcs=[g.fig_gkx], run="fig_gkx()", charts=['sfm_ch13_gkx'],
         data='Published numbers of Gu, Kelly and Xiu (2020), Tables 1 and 7'),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar13 as s
    QUANTLETS.append(dict(
        name='SFM_ch13_seminar',
        desc='Seminar 13 of Statistics of Financial Markets: bias and variance from five training sets, a regression-tree '
             'split, ridge and lasso with one feature, a walk-forward design and a purged K-fold with an embargo, on '
             'paper; next-day realised variance of the S&P 500 (HAR against a random forest), the sign of DAX returns '
             'against the majority-class baseline, VaR 1% of the DAX by quantile regression against historical '
             'simulation.',
        keywords='seminar, machine learning, bias-variance, regression tree, ridge, lasso, walk-forward, purging, HAR, '
                 'random forest, classification baseline, quantile regression, VaR',
        consts=CONSTS, funcs=DATAF + RV + SIGN + QV + [g.cv_r2, g.leakage_frame, g.leakage_experiment, g.wf_design,
                                                       s.bias_variance_numbers, s.tree_split, s.ridge_1d, s.lasso_1d,
                                                       s.walk_forward_design, s.purged_kfold_counts, s.b1_daily,
                                                       s.b3_sign, s.b5_qr],
        run="print(bias_variance_numbers([1.6, 1.7, 1.5, 1.8, 1.6], 2.0, 0.25))\n"
            "print(tree_split([0.2, 0.4, 0.5, 0.9, 1.1, 1.6, 2.0, 2.8], [0.5, 0.7, 0.6, 1.0, 1.4, 2.2, 2.6, 3.4])['best'])\n"
            "print(ridge_1d(50.0, 20.0, [10, 50]))\nprint(walk_forward_design()['n_folds'])\n"
            "print(b1_daily())\nprint(b3_sign())\nprint(b5_qr())",
        charts=['ch13_sem_b1', 'ch13_sem_b3', 'ch13_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 13, 'Machine learning', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
