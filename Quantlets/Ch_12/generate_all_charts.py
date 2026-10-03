"""
generate_all_charts.py -- charts and numbers of Chapter 12 (SFM): scoring models
=================================================================================
Two public credit data sets saved once in data/credit (README there), chart style (sfm_style.py).
Every number on the slides comes from here. Default = 1 (bad credit), 0 = good credit.
  * data        -- the South German Credit data (Groemping, 2019; UCI, doi:10.24432/C5QG88): 1000 credits of
                   1973-1975, 300 bad and 700 good (bad credits oversampled; population bad rate about 5%);
                   the Taiwan credit card data (Yeh and Lien, 2009; UCI, doi:10.24432/C55S3H): 30000 clients;
  * WoE and IV  -- weight of evidence and information value of every variable (fixed bins for the numeric ones);
  * logit       -- odds, log-odds, maximum likelihood, odds ratios with confidence intervals; a WoE logit scorecard;
  * LDA         -- Fisher's linear discriminant (Fisher, 1936; Haerdle and Simar, MVA, Ch. 14) against the logit;
  * scorecard   -- points to double the odds (PDO), base score; prior correction from the 30% sample to 5%;
  * validation  -- confusion matrix, cost-based cut-off, ROC and AUC (DeLong standard error), Gini and the CAP
                   accuracy ratio, KS, Brier score, calibration, repeated cross-validation;
  * extensions  -- reject inference (a simulation on the Taiwan data), group fairness, Basel IRB capital,
                   the Altman Z-score and the Merton distance to default.
Output: charts/sfm_ch12_*.pdf/.png, Quantlets/Ch_12/ch12_numbers.json and ch12_*_table.csv
The conditional PD of the one-factor model is ported from the SFE Quantlet SFEdefaproba (github.com/QuantLet/SFE).
Based on Franke, Haerdle and Hafner (2019),
Statistics of Financial Markets, 5th ed., Ch. 21-22, and on Haerdle and Simar (2019), Applied Multivariate Statistical
Analysis, 5th ed., Ch. 14.
Run:  python3 Quantlets/Ch_12/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import sfm_style as st                               # noqa: E402

TABLE_DIR = HERE
CREDIT_RAW = 'https://raw.githubusercontent.com/danpele/SFM/main/data/credit/'
FILES = {'sgc': 'south_german_credit.csv', 'taiwan': 'taiwan_credit_card_default.csv'}
SEED = 2026
TEST_SHARE = 0.30            # stratified hold-out sample
POP_BAD = 0.05               # bad rate of the bank's population (Groemping, 2019): the sample is 30% bad
COST_FN = 5.0                # cost of accepting a bad credit (a bad classified as good), Statlog cost matrix
COST_FP = 1.0                # cost of rejecting a good credit
PDO, BASE_SCORE, BASE_ODDS = 20.0, 600.0, 50.0   # scorecard scaling: 600 points at good:bad odds of 50:1
IV_MIN = 0.10                # variables with IV >= 0.10 enter the scorecard (Siddiqi, 2006: medium or strong)
N_REPEAT, N_FOLD = 20, 5     # repeated stratified cross-validation
# numeric variables of the South German Credit data: fixed bins (upper edges, months / DM / years)
BINS = {'duration': [12, 18, 24, 36, np.inf], 'amount': [1300, 2000, 3000, 5000, np.inf],
        'age': [25, 30, 35, 45, np.inf]}
NUMERIC = ['duration', 'amount', 'age']
SGC_VARS = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration',
            'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age',
            'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone',
            'foreign_worker']
STATUS = {1: 'no checking account', 2: 'balance < 0 DM', 3: '0 to 200 DM', 4: '>= 200 DM or salary account'}
TW_FEATURES = ['log_limit', 'age', 'pay0_m2', 'pay0_m1', 'pay0_1', 'pay0_2', 'pay0_3', 'late_months', 'util',
               'log_pay1', 'log_pay2', 'log_pay3']
COLORS = {'good': '#1A3A6E', 'bad': '#CD0000', 'logit': '#CD0000', 'lda': '#1A3A6E', 'single': '#B5853F',
          'small': '#2E7D32', 'random': '#8E44AD', 'perfect': '#E67E22'}


# =============================================================================
# DATA
# =============================================================================
def load_credit(name='sgc'):
    """A credit data set from data/credit (local copy) or from the raw GitHub URL; adds 'default' (1 = bad)."""
    fn = FILES[name]
    local = next((os.path.join(d, 'data', 'credit', fn) for d in ('.', '..', '../..', '../../..')
                  if os.path.exists(os.path.join(d, 'data', 'credit', fn))), None)
    df = pd.read_csv(local or CREDIT_RAW + fn)
    if name == 'sgc':
        df['default'] = 1 - df['credit_risk']
    return df


def taiwan_features(df):
    """Model inputs of the Taiwan data: log limit, age, the repayment status of September 2005 as indicators (-2 no
    consumption, -1 paid in full, 1, 2, 3+ months late; 0 = minimum paid is the reference), the number of late months
    April-August, utilisation (bill / limit, clipped to [-1, 3]) and log payments of the last three months."""
    X = pd.DataFrame(index=df.index)
    X['log_limit'] = np.log(df['LIMIT_BAL'])
    X['age'] = df['AGE']
    p0 = df['PAY_0']
    X['pay0_m2'] = (p0 == -2).astype(int)
    X['pay0_m1'] = (p0 == -1).astype(int)
    X['pay0_1'] = (p0 == 1).astype(int)
    X['pay0_2'] = (p0 == 2).astype(int)
    X['pay0_3'] = (p0 >= 3).astype(int)
    X['late_months'] = sum((df[c] >= 1).astype(int) for c in ['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])
    X['util'] = (df['BILL_AMT1'] / df['LIMIT_BAL']).clip(-1, 3)
    for i in (1, 2, 3):
        X[f'log_pay{i}'] = np.log1p(df[f'PAY_AMT{i}'].clip(lower=0))
    return X


def stratified_split(y, share=TEST_SHARE, seed=SEED):
    """Indices of a stratified train/test split (the same share of bads in both samples)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    test = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        test.append(idx[:int(round(share * len(idx)))])
    test = np.sort(np.concatenate(test))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


# =============================================================================
# WEIGHT OF EVIDENCE AND INFORMATION VALUE
# =============================================================================
def bin_values(x, var):
    """Bin labels of a variable: fixed bins for duration, amount and age, the category code otherwise."""
    if var in BINS:
        edges = [-np.inf] + BINS[var]
        return pd.cut(x, edges, right=True, labels=False).astype(int)
    return x.astype(int)


def woe_table(x, y, var):
    """WoE_i = ln(share of goods in bin i / share of bads in bin i); IV = sum (g_i/G - b_i/B) WoE_i.
    Bins without goods or without bads get 0.5 added to both counts (no infinite WoE)."""
    b = bin_values(pd.Series(x).reset_index(drop=True), var)
    y = pd.Series(np.asarray(y))
    t = pd.DataFrame({'bin': b, 'y': y}).groupby('bin')['y'].agg(n='size', bad='sum')
    t['good'] = t['n'] - t['bad']
    adj = ((t['good'] == 0) | (t['bad'] == 0)) * 0.5
    g, bd = t['good'] + adj, t['bad'] + adj
    t['bad_rate'] = t['bad'] / t['n']
    t['dist_good'] = g / g.sum()
    t['dist_bad'] = bd / bd.sum()
    t['woe'] = np.log(t['dist_good'] / t['dist_bad'])
    t['iv'] = (t['dist_good'] - t['dist_bad']) * t['woe']
    return t


def iv_all(df, y, variables=SGC_VARS):
    """Information value of every variable, largest first."""
    return pd.Series({v: woe_table(df[v], y, v)['iv'].sum() for v in variables}).sort_values(ascending=False)


def iv_strength(iv):
    """Rule of thumb (Siddiqi, 2006): < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >= 0.3 strong."""
    return 'useless' if iv < 0.02 else 'weak' if iv < 0.1 else 'medium' if iv < 0.3 else 'strong'


def woe_transform(df, tables, variables):
    """Replace every value by the WoE of its bin (tables estimated on the training sample)."""
    out = pd.DataFrame(index=df.index)
    for v in variables:
        b = bin_values(df[v], v)
        out[v] = b.map(tables[v]['woe']).fillna(0.0).values
    return out


# =============================================================================
# LOGIT AND LDA
# =============================================================================
def logit_fit(X, y):
    """Logistic regression by maximum likelihood (Newton-Raphson): ln(p/(1-p)) = b0 + x'b.
    Returns the coefficients (constant first), standard errors and the log-likelihood."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {'b': b, 'se': se, 'll': ll, 'n': len(y)}


def logit_predict(fit, X):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    return 1 / (1 + np.exp(-X @ fit['b']))


def lda_fit(X, y):
    """Fisher's linear discriminant: w = S_W^{-1}(m_1 - m_0) maximises the between/within variance ratio of w'x;
    with Normal classes and a common covariance, ln odds(bad | x) = c + w'x (the prior enters only c)."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    S = ((X[y == 0] - m0).T @ (X[y == 0] - m0) + (X[y == 1] - m1).T @ (X[y == 1] - m1)) / (n0 + n1 - 2)
    w = np.linalg.solve(S, m1 - m0)
    pi1 = n1 / (n0 + n1)
    c = -0.5 * (m1 + m0) @ w + np.log(pi1 / (1 - pi1))
    return {'w': w, 'c': float(c), 'm0': m0, 'm1': m1, 'S': S, 'pi1': float(pi1)}


def lda_predict(fit, X):
    """Posterior probability of default from the LDA log-odds."""
    return 1 / (1 + np.exp(-(fit['c'] + np.asarray(X, float) @ fit['w'])))


# =============================================================================
# VALIDATION
# =============================================================================
def auc(score, y):
    """AUC = P(score of a random bad > score of a random good) (ties count 1/2), via the Mann-Whitney ranks."""
    score, y = np.asarray(score, float), np.asarray(y)
    r = stats.rankdata(score)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong(score, y, score2=None):
    """DeLong et al. (1988): standard error of the AUC; with a second score on the same cases, the test of
    equal AUCs (z statistic and two-sided p-value)."""
    def comps(s):
        s = np.asarray(s, float)
        pos, neg = s[np.asarray(y) == 1], s[np.asarray(y) == 0]
        v10 = np.array([np.mean((p > neg) + 0.5 * (p == neg)) for p in pos])
        v01 = np.array([np.mean((pos > q) + 0.5 * (pos == q)) for q in neg])
        return v10, v01
    a10, a01 = comps(score)
    A = a10.mean()
    if score2 is None:
        var = np.var(a10, ddof=1) / len(a10) + np.var(a01, ddof=1) / len(a01)
        return {'auc': float(A), 'se': float(np.sqrt(var)), 'lo': float(A - 1.96 * np.sqrt(var)),
                'hi': float(A + 1.96 * np.sqrt(var))}
    b10, b01 = comps(score2)
    S10 = np.cov(np.vstack([a10, b10]))
    S01 = np.cov(np.vstack([a01, b01]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / len(a10) + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / len(a01)
    z = (A - b10.mean()) / np.sqrt(var)
    return {'auc1': float(A), 'auc2': float(b10.mean()), 'z': float(z), 'p': float(2 * stats.norm.sf(abs(z)))}


def roc_points(score, y):
    """ROC curve: (FPR, TPR) when every score value is used as the cut-off (a higher score = riskier)."""
    score, y = np.asarray(score, float), np.asarray(y)
    thr = np.r_[np.inf, np.unique(score)[::-1]]
    tpr = np.array([np.mean(score[y == 1] >= t) for t in thr])
    fpr = np.array([np.mean(score[y == 0] >= t) for t in thr])
    return fpr, tpr, thr


def cap_points(score, y):
    """CAP (cumulative accuracy profile): share of all bads among the riskiest x% of the applicants."""
    score, y = np.asarray(score, float), np.asarray(y)
    order = np.argsort(-score, kind='mergesort')
    x = np.r_[0, np.arange(1, len(y) + 1) / len(y)]
    hit = np.r_[0, np.cumsum(y[order]) / y.sum()]
    return x, hit


def accuracy_ratio(score, y):
    """AR = (area between the model CAP and the diagonal) / (area between the perfect CAP and the diagonal);
    for a continuous score AR = Gini = 2 AUC - 1."""
    x, hit = cap_points(score, y)
    a_model = np.trapezoid(hit, x) - 0.5
    pbad = np.mean(y)
    a_perf = (1 - pbad / 2) - 0.5
    return float(a_model / a_perf)


def ks_stat(score, y):
    """Kolmogorov-Smirnov: the largest distance between the score distributions of the bads and the goods."""
    score, y = np.asarray(score, float), np.asarray(y)
    s = np.sort(np.unique(score))
    F1 = np.searchsorted(np.sort(score[y == 1]), s, side='right') / (y == 1).sum()
    F0 = np.searchsorted(np.sort(score[y == 0]), s, side='right') / (y == 0).sum()
    i = int(np.argmax(np.abs(F0 - F1)))
    return {'ks': float(np.abs(F0 - F1)[i]), 'at': float(s[i]), 's': s, 'F1': F1, 'F0': F0}


def brier(p, y):
    """Brier score: the mean squared difference between the predicted PD and the outcome (0 or 1)."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_table(p, y, k=10):
    """Mean predicted PD and observed default rate in k groups of equal size (sorted by PD); Hosmer-Lemeshow."""
    d = pd.DataFrame({'p': np.asarray(p), 'y': np.asarray(y)}).sort_values('p').reset_index(drop=True)
    d['g'] = np.arange(len(d)) * k // len(d)
    t = d.groupby('g').agg(n=('y', 'size'), pd_mean=('p', 'mean'), rate=('y', 'mean'), bad=('y', 'sum'))
    exp = t['n'] * t['pd_mean']
    hl = float((((t['bad'] - exp) ** 2) / (exp * (1 - t['pd_mean']))).sum())
    return t, hl, float(stats.chi2.sf(hl, k - 2))


def confusion(p, y, cut):
    """Reject (predict bad) if PD > cut. TP: bad rejected; FN: bad accepted; FP: good rejected; TN: good accepted."""
    p, y = np.asarray(p), np.asarray(y)
    pred = (p > cut).astype(int)
    TP = int(np.sum((pred == 1) & (y == 1)))
    FN = int(np.sum((pred == 0) & (y == 1)))
    FP = int(np.sum((pred == 1) & (y == 0)))
    TN = int(np.sum((pred == 0) & (y == 0)))
    n = len(y)
    return {'cut': float(cut), 'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN, 'acc': (TP + TN) / n,
            'tpr': TP / (TP + FN), 'tnr': TN / (TN + FP), 'fpr': FP / (TN + FP), 'precision': TP / max(TP + FP, 1),
            'cost': (COST_FN * FN + COST_FP * FP) / n, 'reject': (TP + FP) / n}


# =============================================================================
# SCORECARD SCALING
# =============================================================================
def scaling(pdo=PDO, base=BASE_SCORE, odds=BASE_ODDS):
    """Score = offset + factor ln(good:bad odds): factor = PDO / ln 2, offset = base - factor ln(odds)."""
    factor = pdo / np.log(2)
    return {'factor': float(factor), 'offset': float(base - factor * np.log(odds))}


def points_table(fit, tables, variables):
    """Points of every attribute: -(b_j WoE_ij + b_0/k) factor + offset/k (k variables, logit for the bads);
    the score of an applicant is the sum of its points."""
    sc = scaling()
    k = len(variables)
    rows = []
    for j, v in enumerate(variables):
        for bn, r in tables[v].iterrows():
            pts = -(fit['b'][j + 1] * r['woe'] + fit['b'][0] / k) * sc['factor'] + sc['offset'] / k
            rows.append({'variable': v, 'bin': bn, 'n': int(r['n']), 'bad_rate': r['bad_rate'], 'woe': r['woe'],
                         'points': pts})
    return pd.DataFrame(rows)


def score_from_pd(p):
    """Score of a PD: offset + factor ln((1 - p)/p)."""
    sc = scaling()
    return sc['offset'] + sc['factor'] * np.log((1 - np.asarray(p)) / np.asarray(p))


def prior_shift(sample_bad, pop_bad):
    """Correction of the log-odds for oversampled bads: ln odds_pop = ln odds_sample + ln(odds of pop / sample)."""
    return float(np.log(pop_bad / (1 - pop_bad)) - np.log(sample_bad / (1 - sample_bad)))


# =============================================================================
# THE SOUTH GERMAN CREDIT SCORECARD
# =============================================================================
def sgc_scorecard(df=None):
    """WoE tables and IV on the training sample, variables with IV >= IV_MIN, WoE logit and LDA; test predictions."""
    df = load_credit('sgc') if df is None else df
    y = df['default'].values
    tr, te = stratified_split(y)
    tables = {v: woe_table(df[v].iloc[tr], y[tr], v) for v in SGC_VARS}
    iv = pd.Series({v: tables[v]['iv'].sum() for v in SGC_VARS}).sort_values(ascending=False)
    sel = [v for v in iv.index if iv[v] >= IV_MIN]
    Xtr, Xte = woe_transform(df.iloc[tr], tables, sel), woe_transform(df.iloc[te], tables, sel)
    lg = logit_fit(Xtr, y[tr])
    ld = lda_fit(Xtr, y[tr])
    out = {'tr': tr, 'te': te, 'y': y, 'tables': tables, 'iv': iv, 'sel': sel, 'logit': lg, 'lda': ld,
           'p_tr': logit_predict(lg, Xtr), 'p_te': logit_predict(lg, Xte), 'q_te': lda_predict(ld, Xte),
           'Xtr': Xtr, 'Xte': Xte}
    out['single_te'] = -woe_transform(df.iloc[te], tables, ['status'])['status'].values   # status alone (-WoE)
    return out


def small_logit(df=None):
    """Interpretable logit on the full sample: duration (years), age (decades), amount (1000 DM) and the two riskiest
    checking-account groups; odds ratios exp(b) with 95% confidence intervals exp(b -+ 1.96 se)."""
    df = load_credit('sgc') if df is None else df
    X = pd.DataFrame({'duration_years': df['duration'] / 12, 'age_decades': df['age'] / 10,
                      'amount_1000': df['amount'] / 1000, 'no_account': (df['status'] == 1).astype(int),
                      'negative_balance': (df['status'] == 2).astype(int)})
    f = logit_fit(X, df['default'])
    names = ['const'] + list(X.columns)
    t = pd.DataFrame({'b': f['b'], 'se': f['se']}, index=names)
    t['z'] = t['b'] / t['se']
    t['p'] = 2 * stats.norm.sf(np.abs(t['z']))
    t['or'] = np.exp(t['b'])
    t['or_lo'] = np.exp(t['b'] - 1.96 * t['se'])
    t['or_hi'] = np.exp(t['b'] + 1.96 * t['se'])
    f0 = logit_fit(np.zeros((len(df), 0)), df['default'])
    return {'table': t, 'll': f['ll'], 'll0': f0['ll'], 'lr': 2 * (f['ll'] - f0['ll']), 'n': f['n'], 'fit': f, 'X': X}


def cv_auc(df=None, n_repeat=N_REPEAT, n_fold=N_FOLD, seed=SEED):
    """Repeated stratified k-fold cross-validation; WoE tables and variable selection are redone inside every
    training fold (otherwise the test folds leak into the bins). Train and test AUC of four models."""
    df = load_credit('sgc') if df is None else df
    y = df['default'].values
    rng = np.random.default_rng(seed)
    res = {m: {'train': [], 'test': []} for m in ['WoE logit', 'LDA', 'small logit', 'status only']}
    Xs = small_logit(df)['X'].values
    for r in range(n_repeat):
        folds = np.empty(len(y), int)
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % n_fold
        for k in range(n_fold):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            tabs = {v: woe_table(df[v].iloc[tr], y[tr], v) for v in SGC_VARS}
            sel = [v for v in SGC_VARS if tabs[v]['iv'].sum() >= IV_MIN]
            Xtr, Xte = woe_transform(df.iloc[tr], tabs, sel), woe_transform(df.iloc[te], tabs, sel)
            lg, ld, sm = logit_fit(Xtr, y[tr]), lda_fit(Xtr, y[tr]), logit_fit(Xs[tr], y[tr])
            pairs = {'WoE logit': (logit_predict(lg, Xtr), logit_predict(lg, Xte)),
                     'LDA': (lda_predict(ld, Xtr), lda_predict(ld, Xte)),
                     'small logit': (logit_predict(sm, Xs[tr]), logit_predict(sm, Xs[te])),
                     'status only': (-Xtr.get('status', pd.Series(0, index=Xtr.index)).values,
                                     -Xte.get('status', pd.Series(0, index=Xte.index)).values)}
            for m, (a, b) in pairs.items():
                res[m]['train'].append(auc(a, y[tr]))
                res[m]['test'].append(auc(b, y[te]))
    return res


def validation(S):
    """All test-sample measures of the WoE logit scorecard and of the LDA."""
    y = S['y'][S['te']]
    p, q = S['p_te'], S['q_te']
    out = {'n_test': int(len(y)), 'bad_test': int(y.sum()), 'n_train': int(len(S['tr'])),
           'auc_train': auc(S['p_tr'], S['y'][S['tr']]), 'auc': auc(p, y), 'auc_lda': auc(q, y),
           'auc_single': auc(S['single_te'], y), 'delong': delong(p, y), 'delong_lda': delong(p, y, q),
           'delong_single': delong(p, y, S['single_te']), 'gini': 2 * auc(p, y) - 1, 'ar': accuracy_ratio(p, y),
           'ks': ks_stat(p, y)['ks'], 'ks_score': float(score_from_pd(ks_stat(p, y)['at'])),
           'brier': brier(p, y), 'brier_ref': brier(np.full(len(y), np.mean(S['y'][S['tr']])), y),
           'cm50': confusion(p, y, 0.5), 'cm_cost': confusion(p, y, COST_FP / (COST_FP + COST_FN)),
           'corr_logit_lda': float(np.corrcoef(np.log(p / (1 - p)), np.log(q / (1 - q)))[0, 1])}
    t, hl, php = calibration_table(p, y, 10)
    out['hl'], out['hl_p'] = hl, php
    out['all_good'] = float(np.mean(y == 0))
    return out


# =============================================================================
# CHARTS: SOUTH GERMAN CREDIT
# =============================================================================
def fig_default_by_status(df=None, save=True):
    """Bad rate and number of credits by the status of the checking account."""
    df = load_credit('sgc') if df is None else df
    t = df.groupby('status')['default'].agg(['size', 'mean'])
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(t))
    bars = ax.bar(x, 100 * t['mean'], color=[st.IDAred, st.Orange, st.Amber, st.MainBlue], width=0.6,
                  label='bad rate in the group')
    ax.axhline(100 * df['default'].mean(), color=st.Forest, ls='--', lw=1.4, label='bad rate of the sample (30%)')
    for b, (n, m) in zip(bars, t.values):
        ax.text(b.get_x() + b.get_width() / 2, 100 * m + 1.2, f'{100 * m:.1f}%  (n = {int(n)})', ha='center',
                fontsize=11, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([STATUS[i] for i in t.index])
    ax.set_ylabel('bad credits (%)')
    ax.set_ylim(0, 60)
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_default_by_status')
    return {int(i): {'n': int(n), 'rate': float(m)} for i, (n, m) in zip(t.index, t.values)}


def fig_woe_duration(df=None, save=True):
    """WoE of the duration bins (bars) and their bad rate (line, right axis)."""
    df = load_credit('sgc') if df is None else df
    t = woe_table(df['duration'], df['default'], 'duration')
    labs = ['<= 12', '13-18', '19-24', '25-36', '> 36']
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(t))
    ax.bar(x, t['woe'], color=[st.MainBlue if w > 0 else st.IDAred for w in t['woe']], width=0.6,
           label='WoE = ln(share of goods / share of bads)')
    ax.axhline(0, color=st.DarkText, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(n = {int(n)})' for l, n in zip(labs, t['n'])])
    ax.set_xlabel('duration of the credit (months)')
    ax.set_ylabel('weight of evidence')
    ax2 = ax.twinx()
    ax2.plot(x, 100 * t['bad_rate'], color=st.Amber, marker='o', lw=2, label='bad rate (%, right axis)')
    ax2.set_ylabel('bad rate (%)')
    ax2.set_ylim(0, 60)
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_woe_duration')
    return {'woe': t['woe'].round(4).tolist(), 'bad_rate': t['bad_rate'].round(4).tolist(), 'n': t['n'].tolist(),
            'iv': float(t['iv'].sum())}


def fig_iv(df=None, save=True):
    """Information value of the 20 variables (full sample), coloured by Siddiqi's strength classes."""
    df = load_credit('sgc') if df is None else df
    iv = iv_all(df, df['default'])
    col = {'strong': st.IDAred, 'medium': st.Orange, 'weak': st.Amber, 'useless': st.MainBlue}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(iv))[::-1]
    for cls in ['strong', 'medium', 'weak', 'useless']:
        m = np.array([iv_strength(v) == cls for v in iv.values])
        ax.barh(y[m], iv.values[m], color=col[cls], height=0.7, label=cls)
    ax.set_yticks(y)
    ax.set_yticklabels([v.replace('_', ' ') for v in iv.index], fontsize=10.5)
    for v in (0.02, 0.1, 0.3):
        ax.axvline(v, color=st.DarkText, ls=':', lw=1)
    ax.set_xlabel('information value (IV)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_iv')
    return {k: float(v) for k, v in iv.items()}


def fig_logistic_curve(df=None, save=True):
    """Bad rate by duration (dots, size = number of credits) and the fitted logistic curve of duration alone."""
    df = load_credit('sgc') if df is None else df
    f = logit_fit(df[['duration']], df['default'])
    g = df.groupby('duration')['default'].agg(['size', 'mean'])
    g = g[g['size'] >= 5]
    xs = np.linspace(0, 75, 300)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.scatter(g.index, g['mean'], s=3 * g['size'], color=st.MainBlue, alpha=0.7,
               label='observed bad rate (durations with at least 5 credits)')
    ax.plot(xs, 1 / (1 + np.exp(-(f['b'][0] + f['b'][1] * xs))), color=st.IDAred, lw=2.2,
            label=f'logit: PD = 1 / (1 + exp(-({f["b"][0]:.2f} + {f["b"][1]:.3f} duration)))')
    ax.set_xlabel('duration (months)')
    ax.set_ylabel('probability of default')
    ax.set_ylim(0, 1)
    st.legend_outside_bottom(ax, ncol=1, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_logistic_curve')
    return {'b0': float(f['b'][0]), 'b1': float(f['b'][1]), 'se1': float(f['se'][1])}


def fig_odds_ratios(L=None, save=True):
    """Odds ratios exp(b) of the small logit with 95% confidence intervals (log scale)."""
    L = small_logit() if L is None else L
    t = L['table'].drop('const')
    labs = {'duration_years': 'duration (+1 year)', 'age_decades': 'age (+10 years)', 'amount_1000': 'amount (+1000 DM)',
            'no_account': 'no checking account', 'negative_balance': 'balance < 0 DM'}
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t['or'], y, xerr=[t['or'] - t['or_lo'], t['or_hi'] - t['or']], fmt='o', color=st.IDAred,
                ecolor=st.MainBlue, capsize=4, lw=2, label='odds ratio exp(b) with 95% confidence interval')
    ax.axvline(1, color=st.DarkText, ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(['0.5', '1', '2', '4', '8'])
    ax.set_yticks(y)
    ax.set_yticklabels([labs[i] for i in t.index])
    ax.set_xlabel('odds ratio (log scale)')
    st.legend_outside_bottom(ax, ncol=1, y=-0.25)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_odds_ratios')


def fig_lda_2d(df=None, save=True):
    """Duration and age of goods and bads; the Fisher LDA and the logit boundaries at PD = 30% (sample rate)."""
    df = load_credit('sgc') if df is None else df
    X = df[['duration', 'age']].values.astype(float)
    y = df['default'].values
    ld, lg = lda_fit(X, y), logit_fit(X, y)
    rng = np.random.default_rng(SEED)
    J = X + rng.uniform(-0.4, 0.4, X.shape)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.scatter(J[y == 0, 0], J[y == 0, 1], s=9, color=COLORS['good'], alpha=0.45, label='good credit')
    ax.scatter(J[y == 1, 0], J[y == 1, 1], s=12, color=COLORS['bad'], alpha=0.65, marker='x', label='bad credit')
    xs = np.linspace(4, 72, 50)
    t = np.log(0.3 / 0.7)
    ax.plot(xs, (t - ld['c'] - ld['w'][0] * xs) / ld['w'][1], color=st.Forest, lw=2.4,
            label='Fisher LDA: PD = 30%')
    ax.plot(xs, (t - lg['b'][0] - lg['b'][1] * xs) / lg['b'][2], color=st.Orange, lw=2.4, ls='--',
            label='logit: PD = 30%')
    ax.set_xlim(2, 74)
    ax.set_ylim(17, 77)
    ax.set_xlabel('duration (months)')
    ax.set_ylabel('age (years)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.17)
    leg = ax.get_legend()
    for h in leg.legend_handles:
        if hasattr(h, 'set_alpha'):
            h.set_alpha(1)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_lda_2d')
    return {'m0': ld['m0'].tolist(), 'm1': ld['m1'].tolist(), 'S': ld['S'].tolist(), 'w': ld['w'].tolist(),
            'c': ld['c'], 'logit_b': lg['b'].tolist(), 'logit_se': lg['se'].tolist(), 'n0': int((y == 0).sum()),
            'n1': int((y == 1).sum())}


def fig_score_dist(S, save=True):
    """Scorecard points of the test sample, goods and bads; the cost-based cut-off."""
    y = S['y'][S['te']]
    sc = score_from_pd(S['p_te'])
    cut = float(score_from_pd(COST_FP / (COST_FP + COST_FN)))
    bins = np.arange(np.floor(sc.min() / 10) * 10, np.ceil(sc.max() / 10) * 10 + 10, 10)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.hist(sc[y == 0], bins=bins, color=COLORS['good'], alpha=0.6, label='good credits (test sample)')
    ax.hist(sc[y == 1], bins=bins, color=COLORS['bad'], alpha=0.6, label='bad credits (test sample)')
    ax.axvline(cut, color=st.Forest, lw=2, ls='--', label=f'cut-off at PD = 1/6: {cut:.0f} points')
    ax.set_xlabel('score (points; 600 = good:bad odds of 50:1, +20 points = double odds)')
    ax.set_ylabel('number of credits')
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_score_dist')
    return {'cut': cut, 'mean_good': float(sc[y == 0].mean()), 'mean_bad': float(sc[y == 1].mean()),
            'min': float(sc.min()), 'max': float(sc.max())}


def fig_roc(S, save=True):
    """ROC curves on the test sample: WoE logit, Fisher LDA, the checking-account status alone."""
    y = S['y'][S['te']]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for lab, s, c, ls in [('WoE logit', S['p_te'], COLORS['logit'], '-'), ('Fisher LDA', S['q_te'], COLORS['lda'], '--'),
                          ('checking account only', S['single_te'], COLORS['single'], '-')]:
        f, t, _ = roc_points(s, y)
        ax.plot(f, t, color=c, lw=2.2, ls=ls, label=f'{lab} (AUC = {auc(s, y):.3f})')
    ax.plot([0, 1], [0, 1], color=COLORS['random'], ls=':', lw=1.5, label='random score (AUC = 0.5)')
    ax.set_xlabel('false positive rate: goods rejected (1 - specificity)')
    ax.set_ylabel('true positive rate: bads rejected')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_roc')


def fig_cap(S, save=True):
    """CAP curves on the test sample: model, random and perfect; AR in the legend."""
    y = S['y'][S['te']]
    p = S['p_te']
    x, h = cap_points(p, y)
    pb = y.mean()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(x, h, color=COLORS['logit'], lw=2.2, label=f'WoE logit (AR = {accuracy_ratio(p, y):.3f})')
    ax.plot([0, pb, 1], [0, 1, 1], color=COLORS['perfect'], lw=2, ls='--', label='perfect model (AR = 1)')
    ax.plot([0, 1], [0, 1], color=COLORS['random'], ls=':', lw=1.5, label='random score (AR = 0)')
    ax.fill_between(x, x, h, color=COLORS['logit'], alpha=0.12)
    ax.set_xlabel('share of applicants, riskiest first')
    ax.set_ylabel('share of all bads captured')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    st.legend_outside_bottom(ax, ncol=1, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_cap')


def fig_ks(S, save=True):
    """Empirical distribution functions of the score for goods and bads; the KS distance."""
    y = S['y'][S['te']]
    sc = score_from_pd(S['p_te'])
    k = ks_stat(-sc, y)                     # risk order: a low score is risky
    s = np.sort(np.unique(sc))
    F1 = np.searchsorted(np.sort(sc[y == 1]), s, side='right') / (y == 1).sum()
    F0 = np.searchsorted(np.sort(sc[y == 0]), s, side='right') / (y == 0).sum()
    i = int(np.argmax(F1 - F0))
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.step(s, F1, where='post', color=COLORS['bad'], lw=2, label='bads: share with score <= s')
    ax.step(s, F0, where='post', color=COLORS['good'], lw=2, label='goods: share with score <= s')
    ax.vlines(s[i], F0[i], F1[i], color=st.Forest, lw=3, label=f'KS = {F1[i] - F0[i]:.3f} at {s[i]:.0f} points')
    ax.set_xlabel('score s (points)')
    ax.set_ylabel('cumulative share')
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_ks')
    return {'ks': float(F1[i] - F0[i]), 'at': float(s[i]), 'ks_check': k['ks']}


def fig_cost_cutoff(S, save=True):
    """Test sample: expected cost per applicant (5 for an accepted bad, 1 for a rejected good) and the share of
    rejected applicants as functions of the PD cut-off."""
    y = S['y'][S['te']]
    p = S['p_te']
    cuts = np.linspace(0.02, 0.9, 89)
    cost = [confusion(p, y, c)['cost'] for c in cuts]
    rej = [confusion(p, y, c)['reject'] for c in cuts]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(cuts, cost, color=st.IDAred, lw=2.2, label='expected cost per applicant')
    ax.axhline(COST_FN * y.mean(), color=st.Amber, ls='--', lw=1.5, label='accept everybody')
    ax.axhline(COST_FP * (1 - y.mean()), color=st.Purple, ls='--', lw=1.5, label='reject everybody')
    ax.axvline(COST_FP / (COST_FP + COST_FN), color=st.Forest, ls=':', lw=2, label='Bayes cut-off PD = 1/6')
    ax.set_xlabel('PD cut-off (reject if PD > cut-off)')
    ax.set_ylabel('cost per applicant')
    ax2 = ax.twinx()
    ax2.plot(cuts, 100 * np.array(rej), color=st.MainBlue, lw=1.8, label='rejected applicants (%, right axis)')
    ax2.set_ylabel('rejected (%)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_cost_cutoff')
    i = int(np.argmin(cost))
    return {'best_cut': float(cuts[i]), 'best_cost': float(cost[i]), 'cost_accept_all': float(COST_FN * y.mean()),
            'cost_reject_all': float(COST_FP * (1 - y.mean()))}


def fig_cv_auc(res, save=True):
    """Repeated cross-validation: train and test AUC of four models (one dot per fold)."""
    models = list(res)
    col = [COLORS['logit'], COLORS['lda'], COLORS['small'], COLORS['single']]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    rng = np.random.default_rng(SEED)
    for i, (m, c) in enumerate(zip(models, col)):
        for j, (part, mk) in enumerate([('train', 'o'), ('test', 'x')]):
            v = np.array(res[m][part])
            xpos = i + (j - 0.5) * 0.36 + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(xpos, v, s=12, color=c, marker=mk, alpha=0.6,
                       label=('training folds' if part == 'train' else 'test folds') if i == 0 else None)
            ax.plot([i + (j - 0.5) * 0.36 - 0.14, i + (j - 0.5) * 0.36 + 0.14], [v.mean()] * 2, color='black', lw=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('AUC')
    st.legend_outside_bottom(ax, ncol=2, y=-0.14)
    leg = ax.get_legend()
    for h in leg.legend_handles:
        h.set_color(st.DarkText)
        h.set_alpha(1)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_cv_auc')
    return {m: {p: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))} for p, v in d.items()}
            for m, d in res.items()}


# =============================================================================
# TAIWAN: IMBALANCE, CALIBRATION, REJECT INFERENCE, FAIRNESS
# =============================================================================
def taiwan_model(df=None):
    """Logit on the Taiwan data (70/30 stratified split); sex, education and marital status are not inputs."""
    df = load_credit('taiwan') if df is None else df
    X = taiwan_features(df)
    y = df['default'].values
    tr, te = stratified_split(y)
    mu, sd = X.iloc[tr].mean(), X.iloc[tr].std()
    Z = (X - mu) / sd
    f = logit_fit(Z.iloc[tr], y[tr])
    return {'df': df, 'X': X, 'Z': Z, 'y': y, 'tr': tr, 'te': te, 'fit': f, 'p': logit_predict(f, Z)}


def fig_calibration(T, save=True):
    """Calibration of the Taiwan logit on the test sample: mean PD against the observed rate in ten groups."""
    y, p = T['y'][T['te']], T['p'][T['te']]
    t, hl, php = calibration_table(p, y, 10)
    se = np.sqrt(t['rate'] * (1 - t['rate']) / t['n'])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], color=COLORS['random'], ls=':', lw=1.5, label='perfect calibration')
    ax.errorbar(t['pd_mean'], t['rate'], yerr=1.96 * se, fmt='o-', color=st.IDAred, ecolor=st.MainBlue, capsize=3,
                lw=2, label='ten groups of 900 test clients (95% interval)')
    ax.set_xlabel('mean predicted PD in the group')
    ax.set_ylabel('observed default rate in the group')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    st.legend_outside_bottom(ax, ncol=1, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_calibration')
    return {'hl': hl, 'hl_p': php, 'brier': brier(p, y), 'brier_ref': brier(np.full(len(y), T['y'][T['tr']].mean()), y),
            'auc': auc(p, y), 'auc_train': auc(T['p'][T['tr']], T['y'][T['tr']]), 'n_test': int(len(y)),
            'rate': float(y.mean()), 'groups': t.round(4).to_dict('index')}


def reject_inference_sim(T, accept=0.7):
    """The bank accepted only the 70% best applicants of an old score (late payment in the last month and the limit);
    outcomes are known only for them. A new logit trained on the accepted (inputs that never vary among them, such as
    being late last month, cannot be estimated and are dropped) is compared with one trained on all applicants
    (unknown in practice), on the test applicants: all of them (through the door) and the accepted only."""
    Z, y, tr, te = T['Z'], T['y'], T['tr'], T['te']
    OLD = ['pay0_1', 'pay0_2', 'pay0_3', 'log_limit']
    old = logit_fit(Z.iloc[tr][OLD], y[tr])
    po = logit_predict(old, Z[OLD])
    cut = np.quantile(po[tr], accept)
    acc_tr = tr[po[tr] <= cut]
    acc_te = te[po[te] <= cut]
    rej_te = np.setdiff1d(te, acc_te)
    keep = [c for c in Z.columns if Z.iloc[acc_tr][c].std() > 1e-8]
    f_acc = logit_fit(Z.iloc[acc_tr][keep], y[acc_tr])
    pa, pf = logit_predict(f_acc, Z[keep]), T['p']
    return {'cut': float(cut), 'n_acc_train': int(len(acc_tr)), 'bad_acc': float(y[acc_tr].mean()),
            'bad_rej': float(y[np.setdiff1d(tr, acc_tr)].mean()), 'bad_all': float(y[tr].mean()),
            'dropped': [c for c in Z.columns if c not in keep],
            'auc_acc_model_on_acc': auc(pa[acc_te], y[acc_te]), 'auc_acc_model_on_all': auc(pa[te], y[te]),
            'auc_all_model_on_all': auc(pf[te], y[te]), 'auc_all_model_on_acc': auc(pf[acc_te], y[acc_te]),
            'auc_old_on_all': auc(po[te], y[te]), 'mean_pd_rej_acc_model': float(pa[rej_te].mean()),
            'mean_pd_rej_all_model': float(pf[rej_te].mean()), 'rate_rejected': float(y[rej_te].mean()),
            'rate_accepted_test': float(y[acc_te].mean())}


def fairness_table(T, approve=0.75):
    """Approve the 75% of test clients with the lowest PD; by sex (1 male, 2 female) and age group: default rate,
    mean PD, approval rate, and the share of non-defaulters approved (equal opportunity)."""
    df, y, p, te = T['df'], T['y'], T['p'], T['te']
    cut = np.quantile(p[te], approve)
    d = pd.DataFrame({'y': y[te], 'p': p[te], 'ok': p[te] <= cut, 'sex': df['SEX'].values[te],
                      'age': pd.cut(df['AGE'].values[te], [0, 25, 35, 50, 100], labels=['<= 25', '26-35', '36-50', '> 50'])})
    rows = {}
    for col, lab in [('sex', {1: 'men', 2: 'women'}), ('age', None)]:
        for g, s in d.groupby(col, observed=True):
            rows[lab[g] if lab else f'age {g}'] = {'n': int(len(s)), 'rate': float(s['y'].mean()), 'pd': float(s['p'].mean()),
                                                   'approve': float(s['ok'].mean()),
                                                   'tpr_good': float(s.loc[s['y'] == 0, 'ok'].mean())}
    return {'cut': float(cut), 'groups': rows}


def fig_fairness(F, save=True):
    """Observed default rate, mean PD and approval rate by sex and age group (Taiwan test sample)."""
    g = F['groups']
    labs = list(g)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(10, 4.0))
    w = 0.26
    ax.bar(x - w, [100 * g[k]['rate'] for k in labs], w, color=st.IDAred, label='observed default rate')
    ax.bar(x, [100 * g[k]['pd'] for k in labs], w, color=st.Amber, label='mean predicted PD')
    ax.bar(x + w, [100 * g[k]['approve'] for k in labs], w, color=st.MainBlue, label='approved (cut-off: best 75%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}\n(n = {g[k]["n"]})' for k in labs])
    ax.set_ylabel('%')
    ax.set_ylim(0, 100)
    st.legend_outside_bottom(ax, ncol=3, y=-0.24)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_fairness')


# =============================================================================
# BASEL IRB, ALTMAN Z, MERTON
# =============================================================================
def irb_capital(pd_, lgd=0.45, kind='other retail'):
    """Basel II IRB capital requirement K per unit of EAD (BCBS, 2006, paragraphs 328-330), retail exposures:
    K = LGD [N((G(PD) + sqrt(R) G(0.999)) / sqrt(1 - R)) - PD]; RWA = 12.5 K EAD."""
    pd_ = np.asarray(pd_, float)
    if kind == 'mortgage':
        R = 0.15 + 0 * pd_
    elif kind == 'revolving':
        R = 0.04 + 0 * pd_
    else:
        w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
        R = 0.03 * w + 0.16 * (1 - w)
    K = lgd * (stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(R) * stats.norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_)
    return K, R


def conditional_pd(pd_, y, rho=0.2):
    """One-factor Gaussian model (Vasicek): borrower i defaults if sqrt(rho) Y + sqrt(1 - rho) e_i < N^-1(PD); given the
    state of the economy Y = y, PD(y) = N((N^-1(PD) - sqrt(rho) y) / sqrt(1 - rho)) (as in the SFE Quantlet SFEdefaproba)."""
    return stats.norm.cdf((stats.norm.ppf(pd_) - np.sqrt(rho) * y) / np.sqrt(1 - rho))


def fig_conditional_pd(rho=0.2, save=True):
    """Conditional PD against the unconditional PD for a bad (y = -3), a typical (y = 0) and a good (y = 3) state of
    the economy, rho = 0.2 (SFEdefaproba)."""
    x = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for y, c, ls, lab in [(-3, st.IDAred, '-', 'bad state of the economy, y = -3'), (0, st.MainBlue, '--', 'typical state, y = 0'),
                          (3, st.Forest, '-.', 'good state, y = 3')]:
        ax.plot(x, conditional_pd(x, y, rho), color=c, lw=2.2, ls=ls, label=lab)
    ax.set_xlabel('unconditional PD')
    ax.set_ylabel('PD given the state y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_conditional_pd')
    return {f'{p}': {f'{y}': float(conditional_pd(p, y, rho)) for y in (-3, 0, 3)} for p in (0.01, 0.02, 0.05)}


def fig_irb(save=True):
    """Expected loss PD x LGD and IRB capital K as functions of PD, LGD = 45%."""
    pds = np.linspace(0.0005, 0.25, 400)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(100 * pds, 100 * 0.45 * pds, color=st.Amber, lw=2, label='expected loss EL = PD x LGD')
    for kind, c, ls in [('other retail', st.IDAred, '-'), ('mortgage', st.MainBlue, '--'), ('revolving', st.Forest, '-.')]:
        K, _ = irb_capital(pds, 0.45, kind)
        ax.plot(100 * pds, 100 * K, color=c, lw=2.2, ls=ls, label=f'capital K, {kind}')
    ax.set_xlabel('PD (%)')
    ax.set_ylabel('% of EAD')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_irb')


def altman_z(x1, x2, x3, x4, x5):
    """Altman (1968) Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5 (ratios as decimals): X1 working capital /
    total assets, X2 retained earnings / TA, X3 EBIT / TA, X4 market value of equity / book value of total debt,
    X5 sales / TA. Zones: Z < 1.81 distress, 1.81-2.99 grey, Z > 2.99 safe."""
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    return {'z': float(z), 'zone': 'distress' if z < 1.81 else 'grey' if z <= 2.99 else 'safe'}


def merton_dd(V, D, sigma, mu=0.05, T=1.0):
    """Merton (1974): assets follow a GBM (Chapter 4); default if V_T < D. DD = [ln(V/D) + (mu - sigma^2/2) T] /
    (sigma sqrt(T)), PD = N(-DD)."""
    dd = (np.log(V / D) + (mu - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return {'dd': float(dd), 'pd': float(stats.norm.cdf(-dd))}


def merton_from_equity(E, sigma_E, D, r=0.03, T=1.0):
    """Solve E = V N(d1) - D e^{-rT} N(d2) and sigma_E E = N(d1) sigma_V V for (V, sigma_V); risk-neutral DD."""
    def eqs(z):
        V, s = np.exp(z[0]), np.exp(z[1])
        d1 = (np.log(V / D) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
        d2 = d1 - s * np.sqrt(T)
        return [V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2) - E,
                stats.norm.cdf(d1) * s * V - sigma_E * E]
    z = optimize.fsolve(eqs, [np.log(E + D), np.log(sigma_E * E / (E + D))])
    V, s = float(np.exp(z[0])), float(np.exp(z[1]))
    dd = (np.log(V / D) + (r - 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    return {'V': V, 'sigma_V': s, 'dd': float(dd), 'pd': float(stats.norm.cdf(-dd))}


def fig_merton(save=True):
    """Left: asset paths of a GBM and the default barrier D. Right: PD = N(-DD) against leverage D/V."""
    rng = np.random.default_rng(SEED)
    V0, D, mu, s, T, n = 100.0, 70.0, 0.05, 0.25, 1.0, 252
    dt = T / n
    Z = rng.standard_normal((40, n))
    paths = V0 * np.exp(np.cumsum((mu - 0.5 * s ** 2) * dt + s * np.sqrt(dt) * Z, axis=1))
    paths = np.column_stack([np.full(40, V0), paths])
    tgrid = np.linspace(0, T, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    for i, pth in enumerate(paths):
        d = pth[-1] < D
        ax.plot(tgrid, pth, color=st.IDAred if d else st.MainBlue, lw=1.0 if d else 0.6, alpha=0.9 if d else 0.5,
                label=('default at T: V_T < D' if d else 'no default at T') if i in (int(np.argmax(paths[:, -1] < D)),
                                                                                  int(np.argmax(paths[:, -1] >= D))) else None)
    ax.axhline(D, color=st.Amber, lw=2, ls='--', label='debt D = 70')
    ax.set_xlabel('time (years)')
    ax.set_ylabel('asset value V')
    ax = axes[1]
    lev = np.linspace(0.3, 0.95, 200)
    for sig, c, ls in [(0.15, st.Forest, '-'), (0.25, st.MainBlue, '--'), (0.40, st.IDAred, '-.')]:
        ax.plot(100 * lev, [100 * merton_dd(1.0, L, sig, mu, T)['pd'] for L in lev], color=c, lw=2.2, ls=ls,
                label=f'asset volatility {int(100 * sig)}%')
    ax.set_xlabel('leverage D / V (%)')
    ax.set_ylabel('one-year PD (%)')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    st.fig_legend_bottom(fig, ncol=3, y=0.08)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch12_merton')
    return {'share_default': float(np.mean(paths[:, -1] < D))}


if __name__ == '__main__':
    st.apply()
    df = load_credit('sgc')
    N = {'n': int(len(df)), 'bad': int(df['default'].sum())}
    N['status'] = fig_default_by_status(df)
    N['status_woe'] = woe_table(df['status'], df['default'], 'status').round(6).to_dict('index')
    N['duration_woe'] = fig_woe_duration(df)
    N['iv'] = fig_iv(df)
    N['curve'] = fig_logistic_curve(df)
    L = small_logit(df)
    fig_odds_ratios(L)
    N['small'] = {'table': L['table'].to_dict('index'), 'll': L['ll'], 'll0': L['ll0'], 'lr': L['lr'], 'n': L['n']}
    N['lda2'] = fig_lda_2d(df)
    S = sgc_scorecard(df)
    N['sel'] = S['sel']
    N['iv_train'] = {k: float(v) for k, v in S['iv'].items()}
    N['woe_logit'] = {'b': S['logit']['b'].tolist(), 'se': S['logit']['se'].tolist(), 'll': S['logit']['ll'],
                      'vars': S['sel']}
    N['lda_w'] = S['lda']['w'].tolist()
    N['val'] = validation(S)
    pts = points_table(S['logit'], S['tables'], S['sel'])
    pts.to_csv(os.path.join(TABLE_DIR, 'ch12_points_table.csv'), index=False, float_format='%.4f')
    N['points'] = pts.round(4).to_dict('records')
    N['scaling'] = scaling()
    N['prior_shift'] = prior_shift(0.3, POP_BAD)
    N['score_dist'] = fig_score_dist(S)
    fig_roc(S)
    fig_cap(S)
    N['ks_fig'] = fig_ks(S)
    N['cost'] = fig_cost_cutoff(S)
    res = cv_auc(df)
    N['cv'] = fig_cv_auc(res)
    T = taiwan_model()
    N['tw'] = {'n': int(len(T['y'])), 'rate': float(T['y'].mean()), 'b': T['fit']['b'].tolist(),
               'se': T['fit']['se'].tolist(), 'features': TW_FEATURES}
    N['tw_cal'] = fig_calibration(T)
    N['tw_cm'] = confusion(T['p'][T['te']], T['y'][T['te']], 0.5)
    N['reject'] = reject_inference_sim(T)
    F = fairness_table(T)
    N['fair'] = F
    fig_fairness(F)
    fig_irb()
    N['cond_pd'] = fig_conditional_pd()
    N['merton_fig'] = fig_merton()
    N['irb'] = {f'{p}': {k: float(irb_capital(p, 0.45, k)[0]) for k in ('other retail', 'mortgage', 'revolving')}
                for p in (0.005, 0.01, 0.02, 0.05, 0.1)}
    N['irb_R'] = {f'{p}': float(irb_capital(p, 0.45)[1]) for p in (0.005, 0.01, 0.02, 0.05, 0.1)}
    N['merton'] = merton_dd(100, 70, 0.25, 0.05, 1.0)
    N['merton_eq'] = merton_from_equity(40, 0.5, 70, 0.03, 1.0)
    N['altman'] = altman_z(0.15, 0.20, 0.08, 0.90, 1.10)
    ivt = pd.DataFrame({'iv_full': pd.Series(N['iv']), 'iv_train': pd.Series(N['iv_train'])})
    ivt['strength'] = ivt['iv_full'].map(iv_strength)
    ivt.to_csv(os.path.join(TABLE_DIR, 'ch12_iv_table.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch12_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print(json.dumps({k: N[k] for k in ['sel', 'val', 'cv', 'tw_cal', 'reject', 'fair', 'cost', 'score_dist', 'ks_fig']},
                     indent=1, default=float)[:9000])
