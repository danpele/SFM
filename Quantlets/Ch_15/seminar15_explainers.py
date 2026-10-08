"""
seminar15_explainers.py -- Explanatory (primer) charts for Seminar 15 (SFM): systemic risk
=========================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 15, which takes place
BEFORE Lecture 15. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (the quantile loss
and a quantile regression, CoVaR and Delta-CoVaR, MES, LRMES and SRISK, the Forbes-Rigobon effect, the moving-block
bootstrap, a connectedness table, Spearman against Pearson correlation) and contain no exercise answers.

Every chart is drawn at the size of its box on the slide (half: one column of a two-column frame; full: the text
width), so that 1 pt in the figure is about 1 pt on the slide and the smallest text is 7 pt.

Output: charts/sfm_ch15_sem_primer_*.pdf and .png (transparent background, legend below the plot, no grey).

Run:  python3 Quantlets/Ch_15/seminar15_explainers.py

Statistica piețelor financiare - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
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


def bank_system(seed, n=3000, b=0.45):
    """Simulated daily returns (%) of a bank and of the system, heavy-tailed, with a stronger link in the tail."""
    rng = np.random.default_rng(seed)
    vol = np.exp(rng.normal(0, 0.4, n))
    xi = 2.0 * vol * stats.t(4).rvs(n, random_state=rng) * np.sqrt(0.5)
    e = vol * stats.t(5).rvs(n, random_state=rng) * np.sqrt(0.6)
    xs = b * xi + 0.15 * np.minimum(xi, 0) + e
    return xi, xs


# =============================================================================
# 1. quantile loss and a 1% quantile regression
# =============================================================================
def fig_quantreg(seed=1, a=0.01):
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    u = np.linspace(-2, 2, 401)
    for al, c in [(0.01, IDAred), (0.5, MainBlue)]:
        ax.plot(u, u * (al - (u < 0)), color=c, lw=1.4, label=f'loss, $\\alpha = {al}$')
    ax.set_xlabel('Residual $u = y - (a + bx)$')
    ax.set_ylabel('Loss')
    ax.set_title('Points below the line cost more', loc='left')
    ax = axes[1]
    xi, xs = bank_system(seed)
    xg = np.linspace(xi.min(), xi.max(), 50)
    for al, c, nm in [(a, IDAred, '1% line'), (0.5, MainBlue, 'median line')]:
        q = QuantileRegressor(quantile=al, alpha=0.0, solver='highs').fit(xi[:, None], xs)
        ax.plot(xg, q.predict(xg[:, None]), color=c, lw=1.5, label=f'quantile regression, {nm}')
        if al == a:
            below = xs < q.predict(xi[:, None])
    ax.scatter(xi[~below], xs[~below], s=1.5, color=MainBlue, alpha=0.35, label='days')
    ax.scatter(xi[below], xs[below], s=6, color=IDAred, label=f'below the 1% line: {100 * below.mean():.1f}%')
    ax.set_xlabel('Bank return $X^i$ (%)')
    ax.set_ylabel('System $X^{sys}$ (%)')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 8)
    ax.set_title('Simulated bank and system, $n = 3000$', loc='left')
    finish(fig, 'sfm_ch15_sem_primer_quantreg', ncol=3, rows=2)


# =============================================================================
# 2. CoVaR and Delta-CoVaR on the 1% quantile line
# =============================================================================
def fig_covar(seed=1, a=0.01):
    xi, xs = bank_system(seed)
    q = QuantileRegressor(quantile=a, alpha=0.0, solver='highs').fit(xi[:, None], xs)
    ah, bh = q.intercept_, q.coef_[0]
    qa, qm = np.quantile(xi, a), np.quantile(xi, 0.5)
    ca, cm = ah + bh * qa, ah + bh * qm
    fig, ax = plt.subplots(figsize=HALF)
    xg = np.linspace(qa - 3, 4, 50)
    ax.plot(xg, ah + bh * xg, color=IDAred, lw=1.5, label=f'1% line $\\hat a + \\hat b x$, $\\hat b$ = {bh:.2f}')
    ax.axvline(qa, color=Amber, lw=1.0, ls='--', label=f'bank in distress: $q_{{1\\%}}(X^i)$ = {qa:.1f}')
    ax.axvline(qm, color=Forest, lw=1.0, ls='--', label=f'bank on a normal day: median = {qm:.1f}')
    ax.plot([qa], [ca], 'o', color=Amber, ms=4)
    ax.plot([qm], [cm], 'o', color=Forest, ms=4)
    ax.annotate('', xy=(qm + 0.6, ca), xytext=(qm + 0.6, cm),
                arrowprops=dict(arrowstyle='<->', color=Navy, lw=0.9))
    ax.plot([qa, qm + 0.6], [ca, ca], color=Navy, lw=0.5, ls=':')
    ax.text(qm + 0.9, (ca + cm) / 2, f'$\\Delta$CoVaR\n= {cm - ca:.2f}', color=Navy, fontsize=7, va='center')
    ax.text(qa + 0.2, ca - 0.5, f'$-$CoVaR = {ca:.2f}', color=Amber, fontsize=7, va='top')
    ax.text(qm - 0.2, cm + 0.3, f'$-$CoVaR at median = {cm:.2f}', color=Forest, fontsize=7, ha='right')
    ax.set_xlabel('Bank return $X^i$ (%)')
    ax.set_ylabel('1% quantile of $X^{sys}$ (%)')
    ax.set_ylim(min(ca, cm) - 2.5, max(ca, cm) + 1.5)
    finish(fig, 'sfm_ch15_sem_primer_covar', ncol=1, rows=3)


# =============================================================================
# 3. MES: the bank on the worst 5% of market days
# =============================================================================
def fig_mes(seed=3, n=1000, a=0.05):
    rng = np.random.default_rng(seed)
    xm = 1.2 * stats.t(4).rvs(n, random_state=rng) * np.sqrt(0.5)
    xb = 1.3 * xm + 1.2 * stats.t(4).rvs(n, random_state=rng) * np.sqrt(0.5)
    k = int(np.ceil(n * a))
    idx = np.argsort(xm)[:k]
    tail = np.zeros(n, bool)
    tail[idx] = True
    mes = -xb[tail].mean()
    es = -xm[tail].mean()
    fig, ax = plt.subplots(figsize=HALF)
    ax.scatter(xm[~tail], xb[~tail], s=2, color=MainBlue, alpha=0.4, label='other days')
    ax.scatter(xm[tail], xb[tail], s=7, color=IDAred, label=f'the $k$ = {k} worst market days')
    ax.axvline(np.sort(xm)[k - 1], color=Amber, lw=1.0, ls='--', label='market 5% quantile')
    ax.axhline(-mes, color=IDAred, lw=1.2, label=f'mean bank return there: $-$MES = {-mes:.2f}%')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Market return $X^m$ (%)')
    ax.set_ylabel('Bank return $X^i$ (%)')
    finish(fig, 'sfm_ch15_sem_primer_mes', ncol=1, rows=4)


# =============================================================================
# 4. LRMES as a function of MES_2%; SRISK against leverage
# =============================================================================
def fig_lrmes_srisk(k=0.08):
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    m = np.linspace(0, 8, 200)
    ax.plot(m, 100 * (1 - np.exp(-18 * m / 100)), color=MainBlue, lw=1.5, label='LRMES $= 1 - e^{-18\\,\\mathrm{MES}_{2\\%}}$')
    ax.plot(m, 18 * m, color=Amber, lw=1.0, ls='--', label='linear approximation $18\\,\\mathrm{MES}_{2\\%}$')
    ax.set_ylim(0, 100)
    ax.set_xlabel('$\\mathrm{MES}_{2\\%}$ (% a day)')
    ax.set_ylabel('LRMES (%)')
    ax.set_title('Six-month crisis loss of the bank', loc='left')
    ax = axes[1]
    L = np.linspace(1, 20, 400)
    for lr, c in [(0.4, Forest), (0.55, MainBlue), (0.7, IDAred)]:
        y = k * (L - 1) - (1 - k) * (1 - lr)
        Ls = 1 + (1 - k) * (1 - lr) / k
        ax.plot(L, y, color=c, lw=1.4, label=f'LRMES = {100 * lr:.0f}%: $L^*$ = {Ls:.1f}')
        ax.plot([Ls], [0], 'o', color=c, ms=3.5)
    ax.axhline(0, color=Navy, lw=0.8)
    ax.text(1.5, 0.6, 'capital missing', color=IDAred, fontsize=7)
    ax.text(13, -0.6, 'capital surplus', color=Forest, fontsize=7)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('Leverage $L = (D + W)/W$')
    ax.set_ylabel('SRISK$/W$')
    ax.set_title('SRISK per unit of equity, $k = 8$%', loc='left')
    finish(fig, 'sfm_ch15_sem_primer_lrmes_srisk', ncol=3, rows=2)


# =============================================================================
# 5. the Forbes-Rigobon effect: a fixed beta, a larger variance of the source
# =============================================================================
def fig_forbes_rigobon(seed=5, n=400, beta=0.5):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    for sx, c, nm in [(1.0, MainBlue, 'calm, sd of $x$ = 1'), (2.5, IDAred, 'crisis, sd of $x$ = 2.5')]:
        x = sx * rng.standard_normal(n)
        y = beta * x + rng.standard_normal(n)
        ax.scatter(x, y, s=2.5, color=c, alpha=0.6, label=f'{nm}: $\\hat\\rho$ = {np.corrcoef(x, y)[0, 1]:.2f}')
    xg = np.linspace(-7, 7, 2)
    ax.plot(xg, beta * xg, color=Navy, lw=1.0, ls='--', label=f'same $\\beta = {beta}$')
    ax.set_xlabel('Source market $x$')
    ax.set_ylabel('Other market $y$')
    ax.set_title('$y = \\beta x + e$: the link does not change', loc='left')
    ax = axes[1]
    d = np.linspace(0, 10, 300)
    for rs, c in [(0.2, Forest), (0.4, MainBlue), (0.6, Purple)]:
        raw = rs * np.sqrt(1 + d) / np.sqrt(1 + d * rs ** 2)
        ax.plot(d, raw, color=c, lw=1.4, label=f'calm $\\rho$ = {rs}')
    ax.set_xlabel('$\\delta = \\sigma^2_{crisis}/\\sigma^2_{calm} - 1$')
    ax.set_ylabel('Raw crisis correlation')
    ax.set_title('The raw correlation rises with $\\delta$ alone', loc='left')
    ax.set_ylim(0, 1)
    finish(fig, 'sfm_ch15_sem_primer_forbes_rigobon', ncol=3, rows=2)


# =============================================================================
# 6. the moving-block bootstrap
# =============================================================================
def fig_bootstrap(seed=6, n=100, block=20, B=2000):
    rng = np.random.default_rng(seed)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = 1
    for t in range(n):
        if t:
            s2[t] = 0.1 + 0.15 * r[t - 1] ** 2 + 0.75 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    starts = rng.integers(0, n - block + 1, size=n // block)
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    cols = [MainBlue, IDAred, Forest, Amber, Purple]
    ax.plot(np.arange(n), r + 6, color=Navy, lw=0.8)
    for j, s in enumerate(starts):
        ax.add_patch(Rectangle((s, 6 - 4.2), block, 8.4, color=cols[j], alpha=0.18, lw=0))
        ax.plot(np.arange(j * block, (j + 1) * block), r[s:s + block] - 6, color=cols[j], lw=0.9)
    ax.text(1, 9.8, 'original series (shaded: the blocks drawn)', fontsize=7, color=Navy)
    ax.text(1, -1.8, 'bootstrap series: the drawn blocks, end to end', fontsize=7, color=Navy)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'Day; blocks of {block} consecutive days')
    ax = axes[1]
    stat = []
    for _ in range(B):
        s = rng.integers(0, n - block + 1, size=n // block)
        idx = (s[:, None] + np.arange(block)).ravel()
        stat.append(np.std(r[idx], ddof=1))
    lo, hi = np.percentile(stat, [2.5, 97.5])
    ax.hist(stat, bins=40, color=MainBlue, alpha=0.7, label=f'{B} bootstrap values of the standard deviation')
    ax.axvline(lo, color=IDAred, lw=1.2, ls='--', label=f'2.5% and 97.5% percentiles: [{lo:.2f}, {hi:.2f}]')
    ax.axvline(hi, color=IDAred, lw=1.2, ls='--')
    ax.set_xlabel('Recomputed measure')
    ax.set_ylabel('Count')
    finish(fig, 'sfm_ch15_sem_primer_bootstrap', ncol=2)


# =============================================================================
# 7. a connectedness table from a simulated VAR(1)
# =============================================================================
def gfevd(A, S, H=10):
    k = S.shape[0]
    Phi = [np.eye(k)]
    for h in range(1, H):
        Phi.append(A @ Phi[-1])
    num, den = np.zeros((k, k)), np.zeros(k)
    for P in Phi:
        num += (P @ S) ** 2
        den += np.diag(P @ S @ P.T)
    th = num / np.diag(S)[None, :] / den[:, None]
    return th / th.sum(axis=1, keepdims=True)


def fig_connectedness():
    names = ['Bank 1', 'Bank 2', 'Bank 3', 'Local A', 'Local B']
    A = np.diag([0.05, 0.05, 0.05, 0.1, 0.1])
    A[3, 0] = A[4, 0] = 0.15
    A[3, 1] = 0.1
    C = np.array([[1, .6, .5, .2, .15], [.6, 1, .55, .15, .1], [.5, .55, 1, .1, .1], [.2, .15, .1, 1, .4],
                  [.15, .1, .1, .4, 1]])
    D = 100 * gfevd(A, C)
    k = len(names)
    frm = D.sum(1) - np.diag(D)
    fig, ax = plt.subplots(figsize=HALF)
    im = ax.imshow(D, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#FFFFFF', MainBlue]),
                   vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{D[i, j]:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if D[i, j] > 55 else Navy)
        ax.text(k + 0.1, i, f'{frm[i]:.0f}', ha='center', va='center', fontsize=7, color=IDAred)
    ax.text(k + 0.1, -0.8, 'FROM', ha='center', fontsize=7, color=IDAred)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=40, ha='right')
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    ax.set_xlim(-0.5, k + 0.6)
    ax.set_xlabel('Shock to $j$')
    ax.set_ylabel('Variance of $i$')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'$d_{{ij}}$ (%), $H = 10$; total = {(D.sum() - np.trace(D)) / k:.0f}%', loc='left')
    finish(fig, 'sfm_ch15_sem_primer_connectedness', ncol=0)


# =============================================================================
# 8. Pearson against Spearman correlation
# =============================================================================
def fig_spearman(seed=8, n=12):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1, 6, n))
    y = -np.exp(0.55 * x) + rng.normal(0, 1.2, n)
    y[-1] = -60
    rp = np.corrcoef(x, y)[0, 1]
    rs = stats.spearmanr(x, y)[0]
    fig, axes = plt.subplots(2, 1, figsize=HALF, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axes[0]
    ax.scatter(x, y, s=10, color=MainBlue, label=f'values: Pearson $r$ = {rp:.2f}')
    ax.set_xlabel('Measure before the crisis')
    ax.set_ylabel('Return in crisis')
    ax = axes[1]
    ax.scatter(stats.rankdata(x), stats.rankdata(y), s=10, color=IDAred, label=f'ranks: Spearman $r_s$ = {rs:.2f}')
    ax.set_xlabel('Rank of the measure')
    ax.set_ylabel('Rank of return')
    finish(fig, 'sfm_ch15_sem_primer_spearman', ncol=1, rows=2)


if __name__ == '__main__':
    fig_quantreg()
    fig_covar()
    fig_mes()
    fig_lrmes_srisk()
    fig_forbes_rigobon()
    fig_bootstrap()
    fig_connectedness()
    fig_spearman()
