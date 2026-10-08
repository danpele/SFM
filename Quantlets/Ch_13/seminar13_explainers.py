"""
seminar13_explainers.py -- Explanatory (primer) charts for Seminar 13 (SFM): machine learning in finance
=======================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 13, which takes place
BEFORE Lecture 13. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (bias and variance,
training and test error, walk-forward and purged validation, ridge and lasso, a tree split, random forest and
boosting, a ReLU network, the HAR averages, the ROC curve, the quantile loss) and contain no exercise answers.

Every chart is drawn at the size of its box on the slide (half: one column of a two-column frame; full: the text
width), so that 1 pt in the figure is about 1 pt on the slide and the smallest text is 7 pt.

Output: charts/sfm_ch13_sem_primer_*.pdf and .png (transparent background, legend below the plot, no grey).

Run:  python3 Quantlets/Ch_13/seminar13_explainers.py

Statistica piețelor financiare - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import sfm_style as st   # noqa: E402

st.apply()
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.1, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'axes.titlepad': 3, 'axes.labelpad': 2, 'legend.handlelength': 1.6,
                     'legend.columnspacing': 1.2, 'legend.borderaxespad': 0.2})
MainBlue, IDAred, Forest, Amber, Navy, BandBlue = st.MainBlue, st.IDAred, st.Forest, st.Amber, '#1F2A44', '#C5D2E8'
Purple, Orange, Teal = st.Purple, st.Orange, st.Teal

# boxes on the slide (inches): HALF = one column (0.50 \textwidth x 0.80 \textheight), FULL = 0.96 \textwidth x
# 0.55 \textheight; \textwidth = 409.7 pt, \textheight = 214.5 pt
HALF = (2.80, 2.30)
FULL = (5.40, 1.58)
CHART_DIR = os.path.join(HERE, '..', '..', 'charts')


def finish(fig, name, ncol=3, rows=1, handles=None, labels=None):
    """Legend below the panels inside the figure box, tight layout, transparent PDF + PNG."""
    h = fig.get_size_inches()[1]
    if ncol:
        if handles is None:
            handles, labels = [], []
            for ax in fig.axes:
                for hd, lb in zip(*ax.get_legend_handles_labels()):
                    if lb not in labels and not lb.startswith('_'):
                        handles.append(hd)
                        labels.append(lb)
        frac = (0.05 + 0.135 * rows) / h
        fig.tight_layout(rect=(0, frac, 1, 1), pad=0.25, h_pad=0.5, w_pad=0.8)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)
    else:
        fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.8)
    st.check_no_grey(fig)
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02, transparent=True)
    fig.savefig(os.path.join(CHART_DIR, f'{name}.png'), bbox_inches='tight', pad_inches=0.02, transparent=True,
                dpi=220)
    plt.close(fig)
    print('   saved', name)


def f_true(x):
    return np.sin(2 * x) + 0.5 * x


# =============================================================================
# 1. bias and variance: shallow against deep trees fitted on several training samples
# =============================================================================
def fig_bias_variance(seed=1, M=10, n=60, sigma=0.5):
    rng = np.random.default_rng(seed)
    xg = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True)
    for ax, depth, col, ttl in [(axes[0], 1, MainBlue, 'Depth 1: large bias, small variance'),
                                (axes[1], 8, IDAred, 'Depth 8: small bias, large variance')]:
        fits = []
        for m in range(M):
            x = rng.uniform(0, 3, n)
            y = f_true(x) + sigma * rng.standard_normal(n)
            t = DecisionTreeRegressor(max_depth=depth, random_state=m).fit(x[:, None], y)
            p = t.predict(xg[:, None])
            fits.append(p)
            ax.plot(xg, p, color=col, lw=0.5, alpha=0.45, label='fits on 10 training samples' if m == 0 else '_')
        ax.plot(xg, np.mean(fits, axis=0), color=col, lw=1.6, label='mean fit $\\bar f(x)$')
        ax.plot(xg, f_true(xg), color=Navy, lw=1.2, ls='--', label='true $f(x)$')
        ax.set_title(ttl, loc='left')
        ax.set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    h = [Line2D([], [], color=MainBlue, lw=0.6), Line2D([], [], color=IDAred, lw=0.6),
         Line2D([], [], color=Navy, lw=1.6), Line2D([], [], color=Navy, lw=1.2, ls='--')]
    finish(fig, 'sfm_ch13_sem_primer_bias_variance', ncol=4, handles=h,
           labels=['fits, depth 1', 'fits, depth 8', 'thick: mean fit $\\bar f(x)$', 'true $f(x)$'])


# =============================================================================
# 2. training error, expected test error and its three parts against the depth of the tree
# =============================================================================
def fig_error_depth(seed=2, M=200, n=80, sigma=0.5):
    rng = np.random.default_rng(seed)
    depths = np.arange(1, 11)
    x0 = np.linspace(0.1, 2.9, 50)
    tr, b2, var = [], [], []
    for d in depths:
        P, T = [], []
        for m in range(M):
            x = rng.uniform(0, 3, n)
            y = f_true(x) + sigma * rng.standard_normal(n)
            t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x[:, None], y)
            P.append(t.predict(x0[:, None]))
            T.append(np.mean((y - t.predict(x[:, None])) ** 2))
        P = np.array(P)
        b2.append(np.mean((P.mean(0) - f_true(x0)) ** 2))
        var.append(np.mean(P.var(0)))
        tr.append(np.mean(T))
    b2, var, tr = map(np.array, (b2, var, tr))
    test = b2 + var + sigma ** 2
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(depths, test, color=Navy, lw=1.6, marker='o', ms=3, label='expected test error')
    ax.plot(depths, tr, color=Amber, lw=1.3, marker='s', ms=2.5, label='training error')
    ax.plot(depths, b2, color=MainBlue, lw=1.1, label='bias$^2$')
    ax.plot(depths, var, color=IDAred, lw=1.1, label='variance')
    ax.axhline(sigma ** 2, color=Forest, lw=1.0, ls='--', label='noise $\\sigma^2$')
    k = int(np.argmin(test))
    ax.annotate('best depth', (depths[k], test[k]), xytext=(depths[k] + 1.2, test[k] + 0.12),
                fontsize=7, color=Navy, arrowprops=dict(arrowstyle='->', color=Navy, lw=0.6))
    ax.set_xlabel('Depth of the tree (flexibility)')
    ax.set_ylabel('Mean squared error')
    ax.set_xticks(depths)
    ax.set_ylim(0, max(test.max(), tr.max()) * 1.1)
    finish(fig, 'sfm_ch13_sem_primer_error_depth', ncol=2, rows=3)


# =============================================================================
# 3. walk-forward with purging; purged K-fold with an embargo
# =============================================================================
def fig_validation():
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    n, start, step, h = 100, 40, 12, 3
    for k in range(5):
        y = 4 - k
        te0 = start + k * step
        ax.add_patch(Rectangle((0, y - 0.35), te0 - h, 0.7, color=MainBlue, lw=0))
        ax.add_patch(Rectangle((te0 - h, y - 0.35), h, 0.7, color=IDAred, lw=0))
        ax.add_patch(Rectangle((te0, y - 0.35), step, 0.7, color=Forest, lw=0))
    ax.set_xlim(0, n)
    ax.set_ylim(-0.6, 4.6)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'fold {5 - i}' for i in range(5)])
    ax.set_xlabel('Time (days, in order)')
    ax.set_title('Walk-forward, expanding window', loc='left')
    ax = axes[1]
    K, B, h, e = 5, 20, 3, 2
    for k in range(K):
        y = K - 1 - k
        for j in range(K):
            x0 = j * B
            if j == k:
                ax.add_patch(Rectangle((x0, y - 0.35), B, 0.7, color=Forest, lw=0))
            else:
                ax.add_patch(Rectangle((x0, y - 0.35), B, 0.7, color=MainBlue, lw=0))
        if k > 0:
            ax.add_patch(Rectangle((k * B - h, y - 0.35), h, 0.7, color=IDAred, lw=0))
        if k < K - 1:
            ax.add_patch(Rectangle(((k + 1) * B, y - 0.35), e, 0.7, color=Amber, lw=0))
    ax.set_xlim(0, K * B)
    ax.set_ylim(-0.6, K - 0.4)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f'test block {K - i}' for i in range(K)])
    ax.set_xlabel('Time (days, in order)')
    ax.set_title('Purged $K$-fold with an embargo, $K = 5$', loc='left')
    for a in axes:
        a.spines['left'].set_visible(False)
        a.tick_params(axis='y', length=0)
    hd = [Rectangle((0, 0), 1, 1, color=c) for c in (MainBlue, Forest, IDAred, Amber)]
    finish(fig, 'sfm_ch13_sem_primer_validation', ncol=4, handles=hd,
           labels=['training days', 'test days', 'purged: target overlaps the test block', 'embargo'])


# =============================================================================
# 4. ridge and lasso with one feature
# =============================================================================
def fig_ridge_lasso(sxx=40.0, sxy=12.0):
    lam = np.linspace(0, 40, 401)
    ols = sxy / sxx
    ridge = sxy / (sxx + lam)
    lasso = np.sign(sxy) * np.maximum(abs(sxy) - lam / 2, 0) / sxx
    fig, axes = plt.subplots(2, 1, figsize=HALF)
    ax = axes[0]
    ax.axhline(ols, color=Navy, lw=1.0, ls='--', label='OLS')
    ax.plot(lam, ridge, color=MainBlue, lw=1.5, label='ridge')
    ax.plot(lam, lasso, color=IDAred, lw=1.5, label='lasso')
    ax.axvline(2 * abs(sxy), color=IDAred, lw=0.6, ls=':')
    ax.text(2 * abs(sxy) + 0.8, 0.2, '$\\lambda = 2|S_{xy}|$', color=IDAred, fontsize=7)
    ax.set_xlabel('Penalty $\\lambda$')
    ax.set_ylabel('$\\hat\\beta$')
    ax.set_title(f'$S_{{xx}} = {sxx:.0f}$, $S_{{xy}} = {sxy:.0f}$', loc='left')
    ax.set_ylim(-0.02, 0.36)
    ax = axes[1]
    b = np.linspace(-0.8, 0.8, 401)
    l2 = 20.0
    ax.plot(b, b, color=Navy, lw=1.0, ls='--')
    ax.plot(b, b * sxx / (sxx + l2), color=MainBlue, lw=1.5)
    ax.plot(b, np.sign(b) * np.maximum(np.abs(b) - l2 / (2 * sxx), 0), color=IDAred, lw=1.5)
    ax.set_xlabel('OLS coefficient $S_{xy}/S_{xx}$ ($\\lambda = 20$)')
    ax.set_ylabel('$\\hat\\beta$')
    finish(fig, 'sfm_ch13_sem_primer_ridge_lasso', ncol=3)


# =============================================================================
# 5. one split of a regression tree: the fit and SSE(s)
# =============================================================================
def fig_tree_split(seed=5, n=40):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.1, 3.0, n))
    y = 0.6 + 0.25 * x + 1.4 * (x > 1.4) + 0.3 * rng.standard_normal(n)
    cand = (x[:-1] + x[1:]) / 2
    sse = np.array([((y[x <= s] - y[x <= s].mean()) ** 2).sum() + ((y[x > s] - y[x > s].mean()) ** 2).sum()
                    for s in cand])
    k = int(np.argmin(sse))
    s = cand[k]
    sse0 = ((y - y.mean()) ** 2).sum()
    fig, axes = plt.subplots(2, 1, figsize=HALF, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axes[0]
    ax.scatter(x, y, s=7, color=MainBlue, label='days $(x_t, y_t)$', zorder=3)
    yl, yr = y[x <= s].mean(), y[x > s].mean()
    ax.plot([x.min(), s], [yl, yl], color=IDAred, lw=1.6, label='leaf means $\\bar y_L$, $\\bar y_R$')
    ax.plot([s, x.max()], [yr, yr], color=IDAred, lw=1.6)
    ax.axvline(s, color=Forest, lw=0.9, ls='--', label='best split $s^*$')
    ax.set_ylabel('$y$')
    ax.set_xlim(0, 3.1)
    ax = axes[1]
    ax.plot(cand, sse, color=Navy, lw=1.2, marker='o', ms=1.8)
    ax.axhline(sse0, color=Amber, lw=1.0, ls=':', label='SSE without a split')
    ax.plot(s, sse[k], 'o', color=Forest, ms=4)
    ax.set_xlim(0, 3.1)
    ax.set_xlabel('Feature $x$ and candidate threshold $s$')
    ax.set_ylabel('SSE$(s)$')
    finish(fig, 'sfm_ch13_sem_primer_tree_split', ncol=2, rows=2)


# =============================================================================
# 6. random forest against one deep tree; gradient boosting step by step
# =============================================================================
def fig_ensembles(seed=6, n=120, sigma=0.5):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 3, n))
    y = f_true(x) + sigma * rng.standard_normal(n)
    xg = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True)
    ax = axes[0]
    ax.scatter(x, y, s=4, color=Amber, label='data', zorder=2)
    tree = DecisionTreeRegressor(random_state=0).fit(x[:, None], y)
    ax.plot(xg, tree.predict(xg[:, None]), color=IDAred, lw=0.7, label='one deep tree')
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0).fit(x[:, None], y)
    ax.plot(xg, rf.predict(xg[:, None]), color=MainBlue, lw=1.6, label='random forest, $B = 300$')
    ax.plot(xg, f_true(xg), color=Navy, lw=1.1, ls='--', label='true $f(x)$')
    ax.set_title('Averaging many trees lowers the variance', loc='left')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax = axes[1]
    ax.scatter(x, y, s=4, color=Amber, zorder=2)
    F, Fg, nu = np.full(n, y.mean()), np.full(len(xg), y.mean()), 0.1
    cols = {1: Purple, 10: Forest, 100: MainBlue}
    for m in range(1, 101):
        t = DecisionTreeRegressor(max_depth=2, random_state=m).fit(x[:, None], y - F)
        F = F + nu * t.predict(x[:, None])
        Fg = Fg + nu * t.predict(xg[:, None])
        if m in cols:
            ax.plot(xg, Fg, color=cols[m], lw=1.3, label=f'boosting, $M = {m}$')
    ax.plot(xg, f_true(xg), color=Navy, lw=1.1, ls='--')
    ax.set_title('Boosting: small trees added to the residuals, $\\nu = 0.1$', loc='left')
    ax.set_xlabel('$x$')
    finish(fig, 'sfm_ch13_sem_primer_ensembles', ncol=4, rows=2)


# =============================================================================
# 7. ReLU and a network with three hidden neurons
# =============================================================================
def fig_relu():
    z = np.linspace(-3, 3, 401)
    x = np.linspace(-3, 3, 401)
    fig, axes = plt.subplots(2, 1, figsize=HALF)
    ax = axes[0]
    ax.plot(z, np.maximum(0, z), color=MainBlue, lw=1.6, label='$g(z) = \\max(0, z)$')
    ax.axhline(0, color=Navy, lw=0.5)
    ax.set_xlabel('$z$')
    ax.set_ylabel('$g(z)$')
    ax.set_title('ReLU activation', loc='left')
    ax = axes[1]
    neurons = [(1.0, 1.5, 1.0, Forest), (-1.0, 0.0, 0.8, Amber), (1.0, -1.0, 1.2, Purple)]
    tot = np.full_like(x, -0.5)
    for k, (w, b, v, c) in enumerate(neurons, 1):
        part = v * np.maximum(0, b + w * x)
        tot += part
        ax.plot(x, part, color=c, lw=0.9, ls='--', label=f'$v_{k}\\,g(b_{k} + w_{k}x)$')
    ax.plot(x, tot, color=IDAred, lw=1.6, label='$\\hat y$ (sum $+\\,c$)')
    ax.set_xlabel('Feature $x$')
    ax.set_ylabel('Output')
    finish(fig, 'sfm_ch13_sem_primer_relu', ncol=2, rows=3)


# =============================================================================
# 8. HAR: the daily, weekly and monthly averages of a simulated variance proxy
# =============================================================================
def fig_har(seed=8, n=1500, show=500):
    rng = np.random.default_rng(seed)
    b0, bd, bw, bm, s = -0.02, 0.35, 0.35, 0.25, 0.55
    lv = np.zeros(n + 22)
    for t in range(22, n + 22):
        v = np.exp(lv[t - 22:t])
        d, w, m = lv[t - 1], np.log(v[-5:].mean()), np.log(v.mean())
        lv[t] = b0 + bd * d + bw * w + bm * m + s * rng.standard_normal()
    lv = lv[22:]
    v = np.exp(lv)
    vw = np.log(np.convolve(v, np.ones(5) / 5, 'valid'))
    vm = np.log(np.convolve(v, np.ones(22) / 22, 'valid'))
    t = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    sl = slice(n - show, n)
    ax.plot(t[sl], lv[sl], color=Amber, lw=0.5, label='$\\ln v_t$ (day)')
    ax.plot(t[4:][-show:], vw[-show:], color=MainBlue, lw=1.0, label='$\\ln v_t^{(w)}$ (5 days)')
    ax.plot(t[21:][-show:], vm[-show:], color=IDAred, lw=1.4, label='$\\ln v_t^{(m)}$ (22 days)')
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('Log variance')
    ax.set_title('Simulated HAR process: three averages', loc='left')
    ax = axes[1]
    lags = np.arange(1, 61)
    x = lv - lv.mean()
    acf = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in lags])
    ax.bar(lags, acf, color=MainBlue, width=0.7)
    ax.axhline(1.96 / np.sqrt(n), color=IDAred, lw=0.7, ls='--', label='$\\pm 1.96/\\sqrt{n}$')
    ax.axhline(-1.96 / np.sqrt(n), color=IDAred, lw=0.7, ls='--')
    ax.set_xlabel('Lag $k$ (days)')
    ax.set_ylabel('ACF of $\\ln v_t$')
    ax.set_title('Long memory: slow decay', loc='left')
    finish(fig, 'sfm_ch13_sem_primer_har', ncol=4)


# =============================================================================
# 9. ROC curve and AUC; the z test of an accuracy
# =============================================================================
def roc(y, s):
    o = np.argsort(-s)
    y = y[o]
    tpr = np.r_[0, np.cumsum(y) / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
    return fpr, tpr, np.trapezoid(tpr, fpr)


def fig_roc(seed=9, n=2000):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.53).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=HALF, gridspec_kw=dict(height_ratios=[1.35, 1]))
    ax = axes[0]
    for sep, c, nm in [(1.0, MainBlue, 'useful model'), (0.08, IDAred, 'almost no skill')]:
        s = sep * y + rng.standard_normal(n)
        fpr, tpr, auc = roc(y, s)
        ax.plot(fpr, tpr, color=c, lw=1.4, label=f'{nm}, AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], color=Navy, lw=0.8, ls='--', label='random guess, AUC = 0.5')
    ax.set_xlabel('False-positive rate')
    ax.set_ylabel('True-pos. rate')
    ax.set_aspect('auto')
    ax = axes[1]
    z = np.linspace(-4, 4, 401)
    ax.plot(z, stats.norm.pdf(z), color=Navy, lw=1.2)
    for a, b in [(-4, -1.96), (1.96, 4)]:
        zz = np.linspace(a, b, 100)
        ax.fill_between(zz, stats.norm.pdf(zz), color=IDAred, alpha=0.75, lw=0,
                        label='rejection region $|z| > 1.96$ (5%)' if a < 0 else '_')
    ax.set_xlabel('$z$ under $H_0$: no skill, $N(0, 1)$')
    ax.set_ylabel('Density')
    ax.set_ylim(0, 0.45)
    finish(fig, 'sfm_ch13_sem_primer_roc', ncol=1, rows=4)


# =============================================================================
# 10. the quantile (pinball) loss; a 1% quantile regression on simulated returns
# =============================================================================
def fig_quantile(seed=10, n=1500, a=0.01):
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    u = np.linspace(-2, 2, 401)
    for al, c in [(0.01, IDAred), (0.5, MainBlue), (0.9, Forest)]:
        ax.plot(u, u * (al - (u < 0)), color=c, lw=1.4, label=f'$\\alpha = {al}$')
    ax.set_xlabel('Error $u = y - \\hat q$')
    ax.set_ylabel('$\\rho_\\alpha(u)$')
    ax.set_title('Quantile loss $\\rho_\\alpha(u)$', loc='left')
    ax = axes[1]
    rng = np.random.default_rng(seed)
    sig = np.exp(rng.normal(0.0, 0.45, n))
    r = sig * stats.t(5).rvs(n, random_state=rng) * np.sqrt(3 / 5)
    qr = QuantileRegressor(quantile=a, alpha=0.0, solver='highs').fit(sig[:, None], r)
    xs = np.linspace(sig.min(), sig.max(), 50)
    q = qr.predict(xs[:, None])
    hs = np.quantile(r, a)
    below = r < qr.predict(sig[:, None])
    ax.scatter(sig[~below], r[~below], s=2, color=MainBlue, alpha=0.5, label='days')
    ax.scatter(sig[below], r[below], s=6, color=IDAred, label='below the 1% line')
    ax.plot(xs, q, color=IDAred, lw=1.5, label='$\\hat q_{0.01}(x)$, quantile regression')
    ax.axhline(hs, color=Amber, lw=1.2, ls='--', label='constant 1% quantile')
    ax.set_xlabel('Volatility feature $x_t$')
    ax.set_ylabel('Return $r_{t+1}$ (%)')
    ax.set_title(f'Simulated returns, $n = {n}$', loc='left')
    finish(fig, 'sfm_ch13_sem_primer_quantile', ncol=4, rows=2)


if __name__ == '__main__':
    fig_bias_variance()
    fig_error_depth()
    fig_validation()
    fig_ridge_lasso()
    fig_tree_split()
    fig_ensembles()
    fig_relu()
    fig_har()
    fig_roc()
    fig_quantile()
