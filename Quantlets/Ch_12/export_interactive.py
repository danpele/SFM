"""Export the inputs of the Chapter 12 interactive page (SFM/interactive/ch12_data.js).

Uses the functions of generate_all_charts.py (same data files, same split, same seeds), so the page starts from the numbers
of the slides. Exported: the 1000 South German Credit records (codes as saved in data/credit) with the code labels of
south_german_credit_codetable.txt, the indices of the stratified test sample, the jitter of the duration-age scatter (seed
2026), the train and test AUC of every fold of the repeated cross-validation, the PD, outcome, sex and age of the 9000 Taiwan
test clients, the pandas sort order of the calibration tables, the Normal draws of the 40 Merton asset paths (seed 2026) and
the results of the reject-inference simulation. From the credit records the browser recomputes the WoE tables, the IV, the
logit and LDA fits, the scorecard, the confusion matrices, ROC, AUC, DeLong, Gini, CAP, KS, Brier and Hosmer-Lemeshow.
Extra runs of the same functions: the reject-inference simulation for acceptance shares of 50, 60, 80 and 90%.
The script checks every number of ch12_numbers.json against the chapter functions (to 1e-12, exact for counts and labels)
and the calibration, fairness and Merton numbers against the rounded exported values (to 1e-6); it stops if one no longer
matches. It writes only interactive/ch12_data.js (no chart, no table of the chapter).
Run:  python3 Quantlets/Ch_12/export_interactive.py
"""
import json
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch12_data.js')
ACCEPT = [0.5, 0.6, 0.7, 0.8, 0.9]


def tree(a, b, what, tol=1e-12):
    """a (recomputed) against b (ch12_numbers.json): floats to tol (relative), the rest exactly."""
    if isinstance(b, dict):
        if set(map(str, a)) != set(b):
            raise SystemExit(f'{what}: keys {sorted(map(str, a))} do not match ch12_numbers.json {sorted(b)}')
        A = {str(k): v for k, v in a.items()}
        for k in b:
            tree(A[k], b[k], f'{what}/{k}', tol)
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{what}: length {len(a)} does not match ch12_numbers.json ({len(b)})')
        for i, (x, y) in enumerate(zip(a, b)):
            tree(x, y, f'{what}[{i}]', tol)
    elif isinstance(b, bool) or isinstance(b, str) or isinstance(b, int) and not isinstance(a, float):
        if a != b and not (isinstance(a, (int, np.integer)) and int(a) == b):
            raise SystemExit(f'{what}: {a} does not match ch12_numbers.json ({b})')
    else:
        a = float(a)
        if not abs(a - b) <= tol * max(1.0, abs(b)):
            raise SystemExit(f'{what}: {a} does not match ch12_numbers.json ({b})')


def rd(a, d=10):
    return [float(f'{x:.{d}g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def codetable():
    """Code labels of every variable of south_german_credit_codetable.txt."""
    s = open(os.path.join(ROOT, 'data', 'credit', 'south_german_credit_codetable.txt'), encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'\$`\w+ = (\w+)`\n(.*?)(?=\n\$`|\Z)', s, flags=re.S):
        lab = {int(a): b.strip() for a, b in re.findall(r'^\s*(\d+) : (.+?)\s*$', m.group(2), flags=re.M)}
        if lab:
            out[m.group(1)] = lab
    return out


def sgc_checks(N, df):
    tree(g.fig_default_by_status(df, save=False), N['status'], 'status')
    tree(g.woe_table(df['status'], df['default'], 'status').round(6).to_dict('index'), N['status_woe'], 'status_woe')
    tree(g.fig_woe_duration(df, save=False), N['duration_woe'], 'duration_woe')
    tree(g.fig_iv(df, save=False), N['iv'], 'iv')
    tree(g.fig_logistic_curve(df, save=False), N['curve'], 'curve')
    L = g.small_logit(df)
    tree({'table': L['table'].to_dict('index'), 'll': L['ll'], 'll0': L['ll0'], 'lr': L['lr'], 'n': L['n']}, N['small'], 'small')
    tree(g.fig_lda_2d(df, save=False), N['lda2'], 'lda2')
    S = g.sgc_scorecard(df)
    tree(S['sel'], N['sel'], 'sel')
    tree({k: float(v) for k, v in S['iv'].items()}, N['iv_train'], 'iv_train')
    tree({'b': S['logit']['b'].tolist(), 'se': S['logit']['se'].tolist(), 'll': S['logit']['ll'], 'vars': S['sel']}, N['woe_logit'], 'woe_logit')
    tree(S['lda']['w'].tolist(), N['lda_w'], 'lda_w')
    tree(g.validation(S), N['val'], 'val')
    tree(g.points_table(S['logit'], S['tables'], S['sel']).round(4).to_dict('records'), N['points'], 'points')
    tree(g.scaling(), N['scaling'], 'scaling')
    tree(g.prior_shift(0.3, g.POP_BAD), N['prior_shift'], 'prior_shift')
    tree(g.fig_score_dist(S, save=False), N['score_dist'], 'score_dist')
    tree(g.fig_ks(S, save=False), N['ks_fig'], 'ks_fig')
    tree(g.fig_cost_cutoff(S, save=False), N['cost'], 'cost')
    plt.close('all')
    return S


def main():
    N = json.load(open(os.path.join(HERE, 'ch12_numbers.json')))
    df = g.load_credit('sgc')
    tree({'n': int(len(df)), 'bad': int(df['default'].sum())}, {'n': N['n'], 'bad': N['bad']}, 'sample')
    print('South German Credit')
    S = sgc_checks(N, df)
    y = df['default'].values
    te = S['te']
    p_te = S['p_te']
    order_sgc = pd.DataFrame({'p': p_te, 'y': y[te]}).sort_values('p').index.values   # order of calibration_table
    rng = np.random.default_rng(g.SEED)                                                 # fig_lda_2d: jitter of the scatter
    X2 = df[['duration', 'age']].values.astype(float)
    J = X2 + rng.uniform(-0.4, 0.4, X2.shape)
    print('cross-validation (20 x 5 folds)')
    res = g.cv_auc(df)
    tree(g.fig_cv_auc(res, save=False), N['cv'], 'cv')
    plt.close('all')
    print('Taiwan')
    T = g.taiwan_model()
    tree({'n': int(len(T['y'])), 'rate': float(T['y'].mean()), 'b': T['fit']['b'].tolist(), 'se': T['fit']['se'].tolist(),
          'features': g.TW_FEATURES}, N['tw'], 'tw')
    tree(g.fig_calibration(T, save=False), N['tw_cal'], 'tw_cal')
    plt.close('all')
    tte = T['te']
    tree(g.confusion(T['p'][tte], T['y'][tte], 0.5), N['tw_cm'], 'tw_cm')
    F = g.fairness_table(T)
    tree(F, N['fair'], 'fair')
    rej = {}
    for a in ACCEPT:
        rej[f'{a:g}'] = g.reject_inference_sim(T, a)
        print('  reject inference, accept', a)
    tree(rej['0.7'], N['reject'], 'reject')
    tp = np.array(rd(T['p'][tte]))
    ty = T['y'][tte]
    order_tw = pd.DataFrame({'p': T['p'][tte], 'y': ty}).sort_values('p').index.values
    # the page's inputs: rounded PD, sorted in the order of calibration_table
    d = pd.DataFrame({'p': tp[order_tw], 'y': ty[order_tw]}).reset_index(drop=True)
    d['g'] = np.arange(len(d)) * 10 // len(d)
    t = d.groupby('g').agg(n=('y', 'size'), pd_mean=('p', 'mean'), rate=('y', 'mean'), bad=('y', 'sum'))
    exp = t['n'] * t['pd_mean']
    hl = float((((t['bad'] - exp) ** 2) / (exp * (1 - t['pd_mean']))).sum())
    tree(hl, N['tw_cal']['hl'], 'HL (rounded PD)', 1e-6)
    tree(t.round(4).to_dict('index'), N['tw_cal']['groups'], 'calibration groups (rounded PD)')
    sex, age = T['df']['SEX'].values[tte], T['df']['AGE'].values[tte]
    cut = np.quantile(tp, 0.75)
    tree(float(cut), N['fair']['cut'], 'fairness cut (rounded PD)', 1e-6)
    for lab, m in (('men', sex == 1), ('women', sex == 2)):
        tree(float(np.mean(tp[m] <= cut)), N['fair']['groups'][lab]['approve'], f'fairness {lab} (rounded PD)', 1e-9)
    print('Basel, Merton, Altman')
    tree(g.fig_conditional_pd(save=False), N['cond_pd'], 'cond_pd')
    tree({f'{p}': {k: float(g.irb_capital(p, 0.45, k)[0]) for k in ('other retail', 'mortgage', 'revolving')} for p in (0.005, 0.01, 0.02, 0.05, 0.1)}, N['irb'], 'irb')
    tree({f'{p}': float(g.irb_capital(p, 0.45)[1]) for p in (0.005, 0.01, 0.02, 0.05, 0.1)}, N['irb_R'], 'irb_R')
    tree(g.merton_dd(100, 70, 0.25, 0.05, 1.0), N['merton'], 'merton')
    tree(g.merton_from_equity(40, 0.5, 70, 0.03, 1.0), N['merton_eq'], 'merton_eq', 1e-9)
    tree(g.altman_z(0.15, 0.20, 0.08, 0.90, 1.10), N['altman'], 'altman')
    tree(g.fig_merton(save=False), N['merton_fig'], 'merton_fig')
    plt.close('all')
    rng = np.random.default_rng(g.SEED)                                                 # fig_merton: 40 paths of 252 steps
    Z = rng.standard_normal((40, 252))
    Zr = np.array([rd(z, 7) for z in Z])
    dt = 1 / 252
    VT = 100 * np.exp(np.cumsum((0.05 - 0.5 * 0.25 ** 2) * dt + 0.25 * np.sqrt(dt) * Zr, axis=1))[:, -1]
    tree(float(np.mean(VT < 70)), N['merton_fig']['share_default'], 'Merton share of defaults (rounded draws)')
    cols = g.SGC_VARS + ['default']
    out = dict(seed=g.SEED, test_share=g.TEST_SHARE, pop_bad=g.POP_BAD, cost_fn=g.COST_FN, cost_fp=g.COST_FP, pdo=g.PDO, base_score=g.BASE_SCORE,
               base_odds=g.BASE_ODDS, iv_min=g.IV_MIN, bins={k: [None if np.isinf(v) else v for v in e] for k, e in g.BINS.items()}, sgc_vars=g.SGC_VARS,
               status_lab=g.STATUS, colors=g.COLORS, labels=codetable(), sgc={c: df[c].astype(int).tolist() for c in cols}, te=te.tolist(),
               order_sgc=order_sgc.tolist(), jitter=[rd(J[:, 0], 6), rd(J[:, 1], 6)], cv=res,
               tw=dict(p=tp.tolist(), y=ty.tolist(), sex=sex.tolist(), age=age.tolist(), order=order_tw.tolist(), rate_train=float(T['y'][T['tr']].mean())),
               reject=rej, accept=ACCEPT, merton_z=Zr.tolist(), N=N)
    js = ('// Inputs of the Chapter 12 interactive page, written by Quantlets/Ch_12/export_interactive.py.\n'
          'window.CH12_DATA = ' + json.dumps(clean(out), separators=(',', ':')) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
