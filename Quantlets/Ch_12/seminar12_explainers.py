"""
seminar12_explainers.py -- explanatory (primer) charts for Seminar 12 (SFM): credit scoring models
==================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 12, which takes place
BEFORE Lecture 12. All charts use SIMULATED data or illustrative parameters only (fixed seeds), different from those of
the exercises: the Basel IRB capital curve, the logistic function, the prior correction of PDs, Fisher's LDA, the
weight of evidence of bins, score distributions with a cut-off and the KS statistic, the ROC curve, a calibration
plot. No exercise answers.

The charts are drawn at the size they have on the slide (included at about scale 1), so every text is >= 6.5 pt.
Output: charts/sfm_ch12_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).

Run:  python3 Quantlets/Ch_12/seminar12_explainers.py

Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import sfm_style as st          # noqa: E402

MainBlue, IDAred, Forest, Amber, Navy = st.MainBlue, st.IDAred, st.Forest, st.Amber, '#1F2A44'
BandBlue = '#C5D2E8'
PREFIX = 'sfm_ch12_sem_primer_'

st.apply()
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2})

HALF = (2.65, 1.85)


def save(fig, name):
    st.check_no_grey(fig)
    d = st.CHART_DIR
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, PREFIX + name + '.pdf'), bbox_inches='tight', pad_inches=0.02, transparent=True)
    fig.savefig(os.path.join(d, PREFIX + name + '.png'), bbox_inches='tight', pad_inches=0.02, transparent=True,
                dpi=200)
    plt.close(fig)
    print('   saved', PREFIX + name)


def legend_below(fig, axes, ncol=3, y=0.0):
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels and not l.startswith('_'):
                handles.append(h)
                labels.append(l)
    fig.tight_layout(pad=0.3)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def sim_scores(seed=12, n_good=1400, n_bad=600):
    """Simulated log-odds of goods and bads (binormal model) and the PDs of a calibrated logit."""
    rng = np.random.default_rng(seed)
    s_good = rng.normal(-1.6, 1.0, n_good)
    s_bad = rng.normal(-0.2, 1.0, n_bad)
    s = np.r_[s_good, s_bad]
    y = np.r_[np.zeros(n_good), np.ones(n_bad)]
    return s, y


# =============================================================================
# 1. Basel IRB capital for other retail exposures
# =============================================================================
def fig_irb(lgd=0.45):
    pd_ = np.linspace(0.001, 0.30, 400)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    R = 0.03 * w + 0.16 * (1 - w)
    stress = stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(R) * stats.norm.ppf(0.999)) / np.sqrt(1 - R))
    K = lgd * (stress - pd_)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(100 * pd_, 100 * lgd * stress, color=IDAred, label='loss in a 1-in-1000 year: LGD $\\times$ stressed PD')
    ax.plot(100 * pd_, 100 * lgd * pd_, color=Amber, label='expected loss EL = PD $\\times$ LGD')
    ax.plot(100 * pd_, 100 * K, color=MainBlue, lw=1.4, label='capital $K$ = the difference')
    ax.set_xlabel('PD (%)')
    ax.set_ylabel('% of EAD')
    ax.set_title(f'Other retail, LGD = {100 * lgd:.0f}%', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'irb')


# =============================================================================
# 2. logistic function and the local effect p(1 - p) beta
# =============================================================================
def fig_logistic(beta=1.0, eta0=-1.0):
    eta = np.linspace(-6, 6, 400)
    p = 1 / (1 + np.exp(-eta))
    p0 = 1 / (1 + np.exp(-eta0))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(eta, p, color=MainBlue, lw=1.4, label='$p = 1/(1 + e^{-\\eta})$')
    tang = p0 + p0 * (1 - p0) * (eta - eta0)
    ok = (tang > -0.05) & (tang < 1.05) & (np.abs(eta - eta0) < 2.5)
    ax.plot(eta[ok], tang[ok], color=IDAred, ls='--', lw=0.9, label=f'slope at $\\eta = {eta0:.0f}$: $p(1 - p) = {p0 * (1 - p0):.3f}$')
    ax.plot(eta0, p0, 'o', color=IDAred, ms=3)
    ax.axhline(0.5, color=Navy, ls=':', lw=0.6)
    ax.axvline(0, color=Navy, ls=':', lw=0.6, label='steepest at $p = 0.5$ (slope 0.25)')
    ax.set_xlabel('log-odds $\\eta$')
    ax.set_ylabel('PD $p$')
    ax.set_title('From log-odds to probability', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'logistic')


# =============================================================================
# 3. prior correction of the PDs (illustrative rates)
# =============================================================================
def fig_prior(pi_s=0.5, pi=0.10):
    p = np.linspace(0.005, 0.995, 400)
    shift = np.log(pi / (1 - pi)) - np.log(pi_s / (1 - pi_s))
    pc = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift)))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(100 * p, 100 * p, color=Navy, ls=':', lw=0.8, label='no correction')
    ax.plot(100 * p, 100 * pc, color=MainBlue, lw=1.4,
            label=f'corrected: sample {100 * pi_s:.0f}% bads, population {100 * pi:.0f}%')
    ax.set_xlabel('PD from the model, sample scale (%)')
    ax.set_ylabel('PD, population scale (%)')
    ax.set_title(f'Shift of the log-odds by ${shift:.3f}$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'prior')


# =============================================================================
# 4. Fisher's LDA in two dimensions
# =============================================================================
def fig_lda(seed=4, n=250):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    m0, m1 = np.array([0.0, 0.0]), np.array([1.5, 0.6])
    x0 = rng.multivariate_normal(m0, cov, n)
    x1 = rng.multivariate_normal(m1, cov, n)
    Sw = 0.5 * (np.cov(x0.T) + np.cov(x1.T))
    w = np.linalg.solve(Sw, x1.mean(0) - x0.mean(0))
    mid = 0.5 * (x0.mean(0) + x1.mean(0))
    fig, ax = plt.subplots(figsize=(2.55, 1.9))
    ax.plot(x0[:, 0], x0[:, 1], 'o', color=Forest, ms=1.6, alpha=0.7, label='goods ($Y = 0$)')
    ax.plot(x1[:, 0], x1[:, 1], 'o', color=IDAred, ms=1.6, alpha=0.7, label='bads ($Y = 1$)')
    xx = np.linspace(-3, 4.5, 50)
    yy = mid[1] - w[0] / w[1] * (xx - mid[0])
    ax.plot(xx, yy, color=Navy, lw=1.1, label='boundary $w^\\top x = w^\\top(m_0 + m_1)/2$')
    ax.annotate('', mid + 0.6 * w / np.linalg.norm(w) * 2, mid, arrowprops=dict(arrowstyle='->', color=MainBlue, lw=1.2))
    ax.text(*(mid + 1.35 * w / np.linalg.norm(w)), '$w$', color=MainBlue, fontsize=7.5)
    ax.set_xlim(-3, 4.5)
    ax.set_ylim(-3, 3.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('LDA: project on $w = S_W^{-1}(m_1 - m_0)$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'lda')


# =============================================================================
# 5. weight of evidence of five bins (illustrative counts)
# =============================================================================
def fig_woe():
    g = np.array([90, 160, 210, 180, 160])
    b = np.array([110, 90, 60, 30, 10])
    G, B = g.sum(), b.sum()
    woe = np.log((g / G) / (b / B))
    iv = np.sum((g / G - b / B) * woe)
    fig, ax = plt.subplots(figsize=HALF)
    cols = [IDAred if v < 0 else Forest for v in woe]
    ax.bar(np.arange(1, 6), woe, color=cols, width=0.65)
    for i, (v, gg, bb) in enumerate(zip(woe, g, b), 1):
        ax.text(i, v + (0.08 if v >= 0 else -0.08), f'{100 * bb / (gg + bb):.0f}% bad', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=6.8)
    ax.bar([0], [0], color=Forest, label='WoE > 0: safer than average')
    ax.bar([0], [0], color=IDAred, label='WoE < 0: riskier than average')
    ax.axhline(0, color=Navy, lw=0.6)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels([f'bin {i}' for i in range(1, 6)])
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(woe.min() - 0.45, woe.max() + 0.45)
    ax.set_ylabel('WoE')
    ax.set_title(f'Simulated variable, IV = {iv:.2f}', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'woe')


# =============================================================================
# 6. score distributions with a cut-off, and the KS statistic
# =============================================================================
def fig_cutoff(c=-0.8):
    s, y = sim_scores()
    pdv = 1 / (1 + np.exp(-s))
    cpd = 1 / (1 + np.exp(-c))
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.45))
    ax = axes[0]
    grid = np.linspace(-5, 3.5, 300)
    fg = stats.gaussian_kde(s[y == 0])(grid)
    fb = stats.gaussian_kde(s[y == 1])(grid)
    ax.plot(grid, fg, color=Forest, label='goods')
    ax.plot(grid, fb, color=IDAred, label='bads')
    ax.fill_between(grid, 0, fb, where=grid > c, color=IDAred, alpha=0.35, lw=0, label='TP: bads rejected')
    ax.fill_between(grid, 0, fg, where=grid > c, color=Amber, alpha=0.45, lw=0, label='FP: goods rejected')
    ax.axvline(c, color=Navy, ls='--', lw=0.8, label=f'cut-off: PD = {cpd:.2f}')
    ax.set_xlabel('log-odds of default')
    ax.set_ylabel('density')
    ax.set_title('Reject to the right of the cut-off', loc='left')
    ax = axes[1]
    xs = np.sort(np.unique(pdv))
    Fg = np.searchsorted(np.sort(pdv[y == 0]), xs, side='right') / (y == 0).sum()
    Fb = np.searchsorted(np.sort(pdv[y == 1]), xs, side='right') / (y == 1).sum()
    k = np.argmax(np.abs(Fg - Fb))
    ax.plot(xs, Fg, color=Forest, label='_')
    ax.plot(xs, Fb, color=IDAred, label='_')
    ax.vlines(xs[k], Fb[k], Fg[k], color=MainBlue, lw=1.5, label=f'KS = {abs(Fg[k] - Fb[k]):.2f}: the largest gap')
    ax.set_xlabel('PD $s$')
    ax.set_ylabel('$F(s)$')
    ax.set_title('Distribution functions of the PDs', loc='left')
    legend_below(fig, axes, ncol=3)
    save(fig, 'cutoff')


# =============================================================================
# 7. ROC curve and AUC
# =============================================================================
def fig_roc():
    s, y = sim_scores()
    th = np.sort(np.unique(s))[::-1]
    P, N = (y == 1).sum(), (y == 0).sum()
    tpr = np.r_[0, [((s >= t) & (y == 1)).sum() / P for t in th]]
    fpr = np.r_[0, [((s >= t) & (y == 0)).sum() / N for t in th]]
    auc = np.trapezoid(tpr, fpr)
    fig, ax = plt.subplots(figsize=(2.55, 1.9))
    ax.fill_between(fpr, 0, tpr, color=BandBlue, alpha=0.7, lw=0, label=f'AUC = {auc:.2f} (area)')
    ax.plot(fpr, tpr, color=MainBlue, lw=1.4, label='ROC curve of the score')
    ax.plot([0, 1], [0, 1], color=IDAred, ls='--', lw=0.8, label='random score: AUC = 0.5')
    ax.set_xlabel('FPR (goods rejected)')
    ax.set_ylabel('TPR (bads rejected)')
    ax.set_aspect('equal')
    ax.set_title('One point per cut-off', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'roc')


# =============================================================================
# 8. calibration plot
# =============================================================================
def fig_calibration(seed=5, n=3000, groups=10):
    rng = np.random.default_rng(seed)
    eta = rng.normal(-1.3, 1.0, n)
    p_true = 1 / (1 + np.exp(-eta))
    y = rng.random(n) < p_true
    fig, ax = plt.subplots(figsize=(2.55, 1.9))
    for p_mod, c, lab in [(p_true, MainBlue, 'calibrated PDs'),
                          (1 / (1 + np.exp(-(1.6 * eta + 0.9))), IDAred, 'miscalibrated PDs (same ranking)')]:
        order = np.argsort(p_mod)
        mp, ob = [], []
        for idx in np.array_split(order, groups):
            mp.append(p_mod[idx].mean())
            ob.append(y[idx].mean())
        ax.plot(mp, ob, 'o-', color=c, ms=2.5, lw=0.9, label=lab)
    ax.plot([0, 1], [0, 1], color=Navy, ls=':', lw=0.8, label='perfect calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('mean PD of the group')
    ax.set_ylabel('observed default rate')
    ax.set_title(f'{groups} groups sorted by PD', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'calibration')


if __name__ == '__main__':
    fig_irb()
    fig_logistic()
    fig_prior()
    fig_lda()
    fig_woe()
    fig_cutoff()
    fig_roc()
    fig_calibration()
