"""Export the inputs of the Chapter 13 interactive page (SFM/interactive/ch13_data.js).

Calls the functions of generate_all_charts.py (same data files, same splits, same seeds), so the page starts from the
numbers of the slides. Exported: the bias-variance simulation of the regression trees and the tree fits of the plotted
sample; the permutation (seed 3) of the validation-scheme diagram; the walk-forward design; the R^2 of the leakage
experiment; the standardised Gram matrix and the ridge and lasso paths of the volatility features (each lasso coefficient row
paired with its own penalty); the depth-2 tree and its
data (S&P 500, 2008-2012); the validation curves of the single tree, the random forest (also by number of trees) and
gradient boosting; the GBP/USD returns with the RBF, MLP and GARCH volatilities; the USD/JPY levels with the RBF forecasts;
the walk-forward forecasts of the six realised-variance models (S&P 500 and Bitcoin) and the HAR inputs of the S&P 500;
the permutation importance and the Shapley values of the random forest; the walk-forward probabilities of the three sign
classifiers (S&P 500, BET, Bitcoin); the four VaR 1% forecasts (S&P 500 and Bitcoin); the best-of-N simulation and the
in-sample and out-of-sample Sharpe ratios of the 30 moving-average rules; the published numbers of Gu, Kelly and Xiu (2020).
The forests, boosting models and networks are not retrained in the browser: their forecasts come from here. From these the
browser recomputes the ridge path, HAR, R^2_OOS, QLIKE, Diebold-Mariano, accuracy, z tests, AUC, Kupiec, Christoffersen,
the quantile loss, the expected maximum Sharpe ratio and the deflated Sharpe ratio.
Extra runs of the same functions (labelled on the page): noise sigma 0.2 and 0.5 in the bias-variance
simulation; purging horizons 1, 10 and 22 days; leakage horizons 5 and 63 days; trees of depth 1 and 3 and leaves of at least
20 and 100 observations; random forests with fewer trees (the first B trees of the same forest); RBF networks with 2 and 5
units; training shares 60 and 90% for USD/JPY; 5 and 20 years in the best-of-N simulation; split years 2009 and 2019 for
the moving-average rules.
The script checks every number of ch13_numbers.json against the chapter functions (to 1e-12 relative, exact for counts,
dates and labels) and stops if one no longer matches; it checks the rounded exports against the same numbers (to 1e-7).
It writes only interactive/ch13_data.js (no chart, no table of the chapter).
Run:  python3 Quantlets/Ch_13/export_interactive.py      (about 20 minutes on a laptop: the walk-forward forecasts)
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import generate_all_charts as g
import sfm_style as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch13_data.js')


def tree(a, b, what, tol=1e-12):
    """a (recomputed) against b (ch13_numbers.json): floats to tol (relative), the rest exactly; NaN equals NaN."""
    if isinstance(b, dict):
        A = {str(k): v for k, v in a.items()}
        if set(A) != set(b):
            raise SystemExit(f'{what}: keys {sorted(A)} do not match ch13_numbers.json {sorted(b)}')
        for k in b:
            tree(A[k], b[k], f'{what}/{k}', tol)
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{what}: length {len(a)} does not match ch13_numbers.json ({len(b)})')
        for i, (x, y) in enumerate(zip(a, b)):
            tree(x, y, f'{what}[{i}]', tol)
    elif b is None or isinstance(b, (bool, str)) or (isinstance(b, int) and not isinstance(a, (float, np.floating))):
        if a != b and not (isinstance(a, (int, np.integer)) and int(a) == b):
            raise SystemExit(f'{what}: {a} does not match ch13_numbers.json ({b})')
    else:
        a, b = float(a), float(b)
        if math.isnan(b) and math.isnan(a):
            return
        if not abs(a - b) <= tol * max(1.0, abs(b)):
            raise SystemExit(f'{what}: {a} does not match ch13_numbers.json ({b})')


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


def days(idx, t0):
    """Dates as day offsets from t0 (compact)."""
    return [int(v) for v in (pd.DatetimeIndex(idx) - pd.Timestamp(t0)).days]


def tree_nodes(m):
    t = m.tree_
    return dict(f=t.feature.tolist(), s=[float(v) for v in t.threshold], l=t.children_left.tolist(), r=t.children_right.tolist(),
                v=[float(v) for v in t.value.ravel()], n=t.n_node_samples.tolist())


def leaf_of(nodes, x):
    k = 0
    while nodes['l'][k] >= 0:
        k = nodes['l'][k] if x[nodes['f'][k]] <= nodes['s'][k] else nodes['r'][k]
    return k


def sec_bias_variance(N):
    out = {}
    for sig in (0.3, 0.2, 0.5):
        for n in (80,):
            bv = g.bias_variance(n=n, sigma=sig)
            rng = np.random.default_rng(g.SEED + 1)                       # fig_bias_variance: the plotted sample
            x = rng.uniform(0, 1, n)
            y = g.true_f(x) + rng.normal(0, sig, n)
            grid = np.linspace(0, 1, 400)
            fits = [rd(DecisionTreeRegressor(max_depth=d, random_state=0).fit(x[:, None], y).predict(grid[:, None]), 6) for d in range(1, 11)]
            if (sig, n) == (0.3, 80):
                best = min(bv, key=lambda k: bv[k]['test'])
                tree({'best_depth': int(best), 'sigma2': sig ** 2, 'n': n, **{f'd{k}': bv[k] for k in (1, 3, best, 10)}}, N['bv'], 'bv')
            out[f'{sig:g}|{n}'] = dict(x=rd(x, 6), y=rd(y, 6), fits=fits, bv={str(k): bv[k] for k in sorted(bv)})
            print('  bias-variance', sig, n)
    return out


def sec_design(N):
    D = {str(h): g.wf_design(h=h) for h in (5, 1, 10, 22)}
    tree(D['5'], N['design'], 'design')
    return D


def sec_leak(N):
    L = {}
    for h in (21, 5, 63):
        L[str(h)] = g.leakage_experiment(h=h)
        print('  leakage, h =', h)
    tree(L['21'], N['leak'], 'leak')
    return L


def sec_shrink(N):
    S = g.shrinkage_paths()
    tree({k: S[k] for k in ('cv_alpha', 'cv_coef', 'ols', 'n', 'nonzero')}, N['shrink'], 'shrink')
    X = g.rv_frame('sp500').loc[:'2012-12-31']                          # shrinkage_paths: the same standardised data
    Z = StandardScaler().fit_transform(X[g.EXT])
    y = X['y'].values - X['y'].mean()
    G, c = Z.T @ Z, Z.T @ y
    for i, a in enumerate(S['lam_r']):                                  # the ridge path is (Z'Z + lambda I)^-1 Z'y
        b = np.linalg.solve(G + a * np.eye(len(c)), c)
        if np.max(np.abs(b - S['ridge'][i])) > 1e-8:
            raise SystemExit(f'ridge path at lambda {a}: closed form does not match sklearn')
    Rg = Ridge(alpha=1.0).fit(Z, y).coef_
    assert np.max(np.abs(Rg - np.linalg.solve(G + np.eye(len(c)), c))) < 1e-8
    # shrinkage_paths returns the lasso penalties in the decreasing order of lasso_path, with the matching rows;
    # the page uses the increasing grid, so both are reversed here.
    lam_l, lasso = S['lam_l'][::-1], S['lasso'][::-1]
    if not np.all(np.diff(lam_l) > 0):
        raise SystemExit('lasso_path: unexpected order of the penalties')
    i28 = int(np.argmin(np.abs(lam_l - S['cv_alpha'])))
    if abs(lam_l[i28] - S['cv_alpha']) > 1e-15:
        raise SystemExit('the lasso penalty chosen by CV is not on the grid')
    if np.max(np.abs(lasso[i28] - np.array([S['cv_coef'][f] for f in g.EXT]))) > 1e-3:
        raise SystemExit('lasso path at the CV penalty does not match the LassoCV refit')
    return dict(lam_l=rd(lam_l, 12), lam_r=rd(S['lam_r'], 12), lasso=[rd(r, 10) for r in lasso], ridge=[rd(r, 10) for r in S['ridge']],
                G=[rd(r, 15) for r in G], c=rd(c, 15), n=S['n'], cv_i=i28, cv_alpha=S['cv_alpha'], cv_coef=S['cv_coef'], ols=S['ols'], nonzero=S['nonzero'])


def sec_tree(N):
    T0 = g.fig_tree(save=False)
    plt.close('all')
    tree(T0, N['tree'], 'tree')
    X = g.rv_frame('sp500').loc[:'2012-12-31']
    X32 = X[g.HAR].values.astype(np.float32)
    trees = {}
    for dep in (1, 2, 3):
        for leaf in (20, 50, 100):
            m = DecisionTreeRegressor(max_depth=dep, min_samples_leaf=leaf, random_state=0).fit(X[g.HAR], X['y'])
            nd = tree_nodes(m)
            cnt = {}
            for row in X32.astype(float):
                k = leaf_of(nd, row)
                cnt[k] = cnt.get(k, 0) + 1
            if any(cnt.get(k, 0) != nd['n'][k] for k in range(len(nd['f'])) if nd['l'][k] < 0):
                raise SystemExit(f'tree depth {dep}, leaf {leaf}: the exported data do not give the leaf counts')
            trees[f'{dep}|{leaf}'] = nd
    nd = trees['2|50']
    lv = [k for k in range(len(nd['f'])) if nd['l'][k] < 0]
    tree({'leaves': [nd['v'][k] for k in lv], 'leaf_n': [nd['n'][k] for k in lv]}, {k: N['tree'][k] for k in ('leaves', 'leaf_n')}, 'tree nodes')
    return dict(X=[[float(f'{v:.9g}') for v in X32[:, j]] for j in range(3)], y=rd(X['y'], 10), trees=trees, mean_y=float(X['y'].mean()))


def sec_ens(N):
    E = g.ensemble_curves()
    best = {lr: int(np.argmin(d['val']) + 1) for lr, d in E['gb'].items()}
    tree({'har': E['har'], 'tree_best': float(min(E['tree'])), 'tree_best_depth': int(np.argmin(E['tree']) + 1),
          'rf_best': float(min(E['rf'])), 'rf_best_depth': int(np.argmin(E['rf']) + 1), 'tree14': E['tree'][-1], 'rf14': E['rf'][-1],
          'gb_best': {str(lr): float(min(d['val'])) for lr, d in E['gb'].items()}, 'gb_best_n': {str(lr): best[lr] for lr in best},
          'gb_last': {str(lr): d['val'][-1] for lr, d in E['gb'].items()}, 'n_train': E['n_train'], 'n_val': E['n_val']}, N['ens'], 'ens')
    X = g.rv_frame('sp500')                                             # ensemble_curves: the same split
    tr, va = X.loc[:'2018-12-31'].iloc[:-g.H], X.loc['2018-12-31':'2022-12-31'].iloc[1:]
    rfB = []
    for i, dep in enumerate(E['depth']):                                # the same forests: validation MSE of the first B trees
        f = RandomForestRegressor(n_estimators=200, max_depth=dep, max_features=0.5, n_jobs=-1, random_state=g.SEED).fit(tr[g.EXT], tr['y'])
        P = np.array([t.predict(va[g.EXT].values) for t in f.estimators_])
        cm = np.cumsum(P, axis=0) / np.arange(1, len(P) + 1)[:, None]
        mse = np.mean((cm - va['y'].values) ** 2, axis=1)
        if abs(mse[-1] - E['rf'][i]) > 1e-10:
            raise SystemExit(f'random forest of depth {dep}: the mean of the 200 trees does not give the forest MSE')
        rfB.append(rd(mse, 10))
    print('  forests by number of trees')
    return dict(depth=E['depth'], tree=E['tree'], rf=E['rf'], rfB=rfB, har=E['har'],
                gb={f'{lr:g}': {'val': rd(d['val'], 10), 'train': rd(d['train'], 10)} for lr, d in E['gb'].items()}, n_train=E['n_train'], n_val=E['n_val'])


def sec_nnarch(N):
    R = g.nnarch()
    tree({k: v for k, v in R.items() if k not in ('X', 'vol')}, N['nnarch'], 'nnarch')
    X, vol = R['X'], R['vol']
    feats = ['r1', 'r2', 'r3', 'bund', 'gold']
    units = {}
    for c in (3, 2, 5):                                                 # nnarch with c RBF units (c = 3 in the lecture)
        fnet = g.rbf_fit(X[feats], X['y'], clusters=c, seed=g.SEED)
        e = X['y'] - g.rbf_predict(fnet, X[feats])
        gnet = g.rbf_fit(X[feats], e ** 2, clusters=c, seed=g.SEED)
        s2 = np.maximum(g.rbf_predict(gnet, X[feats]), 0.05 * float(np.mean(e ** 2)))
        if c == 3 and np.max(np.abs(np.sqrt(s2) - vol['rbf'].values)) > 1e-12:
            raise SystemExit('nnarch: the RBF volatility does not reproduce')
        units[str(c)] = dict(vol=rd(np.sqrt(s2), 5), r2=float(1 - np.mean(e ** 2) / np.var(X['y'])))
    return dict(t0=X.index[0].date().isoformat(), t=days(X.index, X.index[0]), y=rd(X['y'], 5), garch=rd(vol['garch'], 5), mlp=rd(vol['mlp'], 5),
                units=units, alpha=R['alpha'], beta=R['beta'])


def sec_nnjpy(N):
    J = {}
    p = g.fx_close('USDJPY.FOREX', '2000-01-01')
    for s in (0.8, 0.6, 0.9):
        R = g.nnjpy(train_share=s)
        if s == 0.8:
            tree({k: R[k] for k in N['nnjpy']}, N['nnjpy'], 'nnjpy')
        n_tr = int(s * len(p))
        assert np.array_equal(R['ytr'], p.values[3:n_tr]) and np.array_equal(R['yte'], p.values[n_tr:])
        J[f'{s:g}'] = dict(n_tr=n_tr, ftr=rd(R['ftr'], 6), fte=rd(R['fte'], 6), r2_ret_te=R['r2_ret_te'],
                           **{k: R[k] for k in ('rmse_tr', 'rmse_te', 'rmse_rw_tr', 'rmse_rw_te', 'max_tr', 'max_te', 'split')})
        print('  nnjpy, training share', s)
    return dict(t0=p.index[0].date().isoformat(), t=days(p.index, p.index[0]), p=[float(v) for v in p.values], runs=J)


def sec_rv(N):
    out = {}
    for k in ('sp500', 'btc'):
        X, P = g.rv_forecasts(k)
        M = g.rv_metrics(X, P)
        tree(M, N['rv'][k], f'rv/{k}')
        idx = P['HAR'].index
        yrs = sorted(set(idx.year))
        d = dict(t0=idx[0].date().isoformat(), t=days(idx, idx[0]), y=rd(X.loc[idx, 'y'], 9), rv=rd(X.loc[idx, 'rv'], 9), models=list(P),
                 pred={m: rd(P[m]['pred'], 8) for m in P}, years=yrs,
                 s2={m: [float(P[m]['s2'][idx.year == yr].iloc[0]) for yr in yrs] for m in P})
        Xr = pd.DataFrame({'y': d['y'], 'rv': d['rv']}, index=idx)          # the rounded exports give the numbers of the slides
        Pr = {}
        for m in P:
            f = pd.DataFrame({'pred': d['pred'][m], 's2': P[m]['s2'].values}, index=idx)
            f['var'] = np.exp(f['pred'] + f['s2'] / 2)
            Pr[m] = f
        tree(g.rv_metrics(Xr, Pr), N['rv'][k], f'rv/{k} (rounded)', 1e-7)
        if k == 'sp500':                                                # the HAR inputs; the test targets are the rows from `off` on
            d['frame'] = dict(t0=X.index[0].date().isoformat(), t=days(X.index, X.index[0]), **{c: rd(X[c], 9) for c in ('d', 'w', 'm', 'y', 'rv')})
            off = int(np.searchsorted(X.index, idx[0]))
            assert d['frame']['y'][off:off + len(idx)] == d['y'] and d['frame']['rv'][off:off + len(idx)] == d['rv']
            d['off'] = off
            del d['y'], d['rv']
        out[k] = d
        print('  volatility forecasts', k)
    return out


def sec_imp(N):
    I = g.importance()
    tree({k: v for k, v in I.items() if k not in ('phi', 'Xs')}, N['imp'], 'imp')
    return dict(perm=I['perm'], perm_sd=I['perm_sd'], base=I['base'], mean_abs_phi=I['mean_abs_phi'], n_test=I['n_test'],
                phi={f: rd(I['phi'][f], 8) for f in g.HAR}, x={f: rd(I['Xs'][f], 8) for f in g.HAR})


def sec_sign(N):
    S, out = {}, {}
    for k in ('sp500', 'bet', 'btc'):
        X, prob, base = g.sign_walk_forward(k)
        S[k] = g.sign_metrics(X, prob, base)
        y = X['y'].values
        for dg in (6, 7, 8, 10, 17):                                    # fewest digits with the same metrics (accuracy exactly)
            pr = {m: np.array(rd(p.values, dg)) for m, p in prob.items()}
            Sr = g.sign_metrics(X, {m: pd.Series(pr[m], index=p.index) for m, p in prob.items()}, base)
            if all(Sr[m]['acc'] == S[k][m]['acc'] and abs(Sr[m]['auc'] - S[k][m]['auc']) < 1e-9 for m in prob):
                break
        else:
            raise SystemExit(f'sign {k}: rounding changes the metrics')
        out[k] = dict(t0=X.index[0].date().isoformat(), t=days(X.index, X.index[0]), y=y.astype(int).tolist(), base=base.values.astype(int).tolist(),
                      prob={m: pr[m].tolist() for m in prob})
        print('  sign', k)
    tree(S, N['sign'], 'sign')
    return out


def sec_qvar(N):
    out = {}
    for k in ('sp500', 'btc'):
        X, V = g.quantile_var(k)
        tree(g.var_backtest(X, V), N['qvar'][k], f'qvar/{k}')
        for dg in (7, 8, 10, 17):                                       # fewest digits with the same backtest (exceedances exactly)
            Xr = pd.DataFrame({'y': rd(X['y'], dg)}, index=X.index)
            Vr = pd.DataFrame({m: rd(V[m], dg) for m in V}, index=V.index)
            try:
                tree(g.var_backtest(Xr, Vr), N['qvar'][k], f'qvar/{k} (rounded)', 1e-7)
                break
            except SystemExit:
                continue
        else:
            raise SystemExit(f'qvar {k}: rounding changes the backtest')
        out[k] = dict(t0=X.index[0].date().isoformat(), t=days(X.index, X.index[0]), y=Xr['y'].tolist(), methods=list(V),
                      var={m: Vr[m].tolist() for m in V})
        print('  quantile VaR', k)
    return out


def sec_snoop(N):
    M = {str(y): g.max_sharpe_sim(years=y) for y in (10, 5, 20)}
    tree(M['10'], N['maxsr'], 'maxsr')
    Sn = {}
    for sp in ('2014', '2009', '2019'):
        S = g.snooping(split=f'{sp}-12-31')
        if sp == '2014':
            tree({k: v for k, v in S.items() if k != 'D'}, N['snoop'], 'snoop')
        Sn[sp] = dict(D=S['D'].to_dict('list'), **{k: v for k, v in S.items() if k != 'D'})
    return dict(maxsr=M, runs=Sn, euler=g.EULER)


def main():
    N = json.load(open(os.path.join(HERE, 'ch13_numbers.json')))
    tree(g.END, N['end'], 'end')
    tree({'models': g.GKX_MODELS, 'r2': g.GKX_R2, 'top': g.GKX_R2_TOP, 'sr': g.GKX_SR}, N['gkx'], 'gkx')
    out = dict(seed=g.SEED, H=g.H, har=g.HAR, ext=g.EXT, feat_label=g.FEAT_LABEL, feat_col=g.FEAT_COL, mcol=g.MCOL, name=g.NAME,
               colors=dict(MainBlue=st.MainBlue, IDAred=st.IDAred, Forest=st.Forest, Amber=st.Amber, Orange=st.Orange, Purple=st.Purple,
                           Teal=st.Teal, Crimson=st.Crimson, DarkText=st.DarkText),
               cv_perm=np.random.default_rng(3).permutation(60).tolist(),    # fig_cv_schemes: rng = default_rng(3)
               gkx={'models': g.GKX_MODELS, 'r2': g.GKX_R2, 'top': g.GKX_R2_TOP, 'sr': g.GKX_SR}, N=N)
    steps = [('bias-variance', 'bv', sec_bias_variance), ('walk-forward design', 'design', sec_design), ('leakage', 'leak', sec_leak),
             ('shrinkage', 'shrink', sec_shrink), ('tree', 'tree', sec_tree), ('ensembles', 'ens', sec_ens), ('nn-arch', 'nnarch', sec_nnarch),
             ('nnjpy', 'nnjpy', sec_nnjpy), ('volatility forecasts', 'rv', sec_rv), ('importance', 'imp', sec_imp), ('sign', 'sign', sec_sign),
             ('quantile VaR', 'qvar', sec_qvar), ('snooping', 'snoop', sec_snoop)]
    for lab, key, f in steps:
        print(lab)
        out[key] = f(N)
    js = ('// Inputs of the Chapter 13 interactive page, written by Quantlets/Ch_13/export_interactive.py.\n'
          'window.CH13_DATA = ' + json.dumps(clean(out), separators=(',', ':'), allow_nan=False) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
