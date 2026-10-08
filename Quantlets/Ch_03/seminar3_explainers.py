"""
seminar3_explainers.py -- Explanatory (primer) charts for Seminar 3 (SFM): alpha-stable distributions
===================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 3, which takes
place BEFORE Lecture 3. All charts use SIMULATED data or exact formulas (fixed seeds): the scale rule of stable sums,
simulated stable paths next to a Gaussian random walk, characteristic functions, stable densities, the S0/S1 shift,
power-law tails, the running sample variance, VaR and ES on a stable density, CMS draws and McCulloch's quantile
ratio. They contain no exercise answers.

The charts are drawn at the size of their box on the slide (half-slide column, about 2.7 x 2.2 inches), so that
1 pt in the figure is about 1 pt on the slide (tick labels 7 pt).

Output: charts/ch3_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom, no grey).

Run:  python3 Quantlets/Ch_03/seminar3_explainers.py

Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import sfm_style as st   # noqa: E402

MainBlue, IDAred, Forest, Amber, Navy = st.MainBlue, st.IDAred, st.Forest, st.Amber, st.DarkText
BandBlue = '#C5D2E8'
CHART_DIR = os.path.join(HERE, '..', '..', 'charts')

st.apply()
# fonts at slide size: the charts are shown at ~100% of their size
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7.2,
                     'ytick.labelsize': 7.2, 'legend.fontsize': 7.2, 'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2,
                     'axes.titlepad': 3, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.0,
                     'legend.handletextpad': 0.4, 'mathtext.fontset': 'dejavusans'})
HALF = (2.75, 2.15)      # chart column of a two-column slide


def save(fig, name, ncol=2, legend=True):
    """Tight layout, one legend below the panels, no grey, transparent PDF + PNG."""
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels and not l.startswith('_'):
                handles.append(h)
                labels.append(l)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    if legend and handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)
    st.check_no_grey(fig)
    os.makedirs(CHART_DIR, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 220})):
        fig.savefig(os.path.join(CHART_DIR, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.02, transparent=True, **kw)
    plt.close(fig)
    print(f'   saved {name}')


from scipy import stats   # noqa: E402
from scipy.stats import levy_stable   # noqa: E402  (S1 parameterisation by default)


def cms(alpha, size, rng):
    """Chambers-Mallows-Stuck draws of S(alpha, 0, 1, 0)."""
    V = np.pi * (rng.uniform(size=size) - 0.5)
    W = rng.exponential(1.0, size)
    if alpha == 1:
        return np.tan(V)
    return np.sin(alpha * V) / np.cos(V) ** (1 / alpha) * (np.cos((1 - alpha) * V) / W) ** ((1 - alpha) / alpha)


ALPHAS = ((2.0, MainBlue), (1.7, Forest), (1.5, Amber), (1.0, IDAred))


# (1) the scale of a sum of n i.i.d. stable terms
def fig_scale_rule():
    n = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=HALF)
    for a, c in ALPHAS:
        ax.plot(n, n ** (1 / a), color=c, lw=1.3, label=rf'$\alpha = {a:g}$: $n^{{1/{a:g}}}$')
    ax.set_xlabel('Number of terms $n$')
    ax.set_ylabel(r'Scale of the sum / scale of one term')
    ax.set_title(r'$X_1 + \cdots + X_n \overset{d}{=} n^{1/\alpha} X + d_n$', loc='left')
    ax.set_ylim(0, 60)
    save(fig, 'ch3_sem_primer_scale_rule', ncol=2)


# (2) simulated daily returns and their cumulative sums: Normal and stable
def fig_paths():
    rng = np.random.default_rng(17)
    n = 1000
    g = np.sqrt(2) * rng.standard_normal(n)          # S(2, 0, 1, 0) = N(0, 2)
    s = cms(1.5, n, rng)
    t = np.arange(1, n + 1)
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True)
    ax = axes[0]
    ax.plot(t, s, color=IDAred, lw=0.5, label=r'Stable, $\alpha = 1.5$')
    ax.plot(t, g, color=MainBlue, lw=0.5, label=r'Normal, $\alpha = 2$')
    ax.set_ylabel('$X_t$')
    ax = axes[1]
    ax.plot(t, np.cumsum(s), color=IDAred, lw=0.9)
    ax.plot(t, np.cumsum(g), color=MainBlue, lw=0.9)
    ax.set_ylabel(r'$\sum_{s \leq t} X_s$')
    ax.set_xlabel('Day $t$')
    save(fig, 'ch3_sem_primer_paths', ncol=2)


# (3) characteristic functions
def fig_charfun():
    t = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=HALF)
    for a, c in ALPHAS:
        ax.plot(t, np.exp(-np.abs(t) ** a), color=c, lw=1.3, label=rf'$\alpha = {a:g}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\varphi(t) = e^{-|t|^\alpha}$ ($\beta = 0$, $\gamma = 1$)')
    ax.set_title('Heavier tails: sharper peak of $\\varphi$ at 0', loc='left')
    save(fig, 'ch3_sem_primer_charfun', ncol=4)


# (4) stable densities: the role of alpha and of beta
def fig_densities():
    x = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(2, 1, figsize=(HALF[0], 1.95), sharex=True)
    ax = axes[0]
    for a, c in ALPHAS:
        ax.plot(x, levy_stable.pdf(x, a, 0.0), color=c, lw=1.2, label=rf'$\alpha = {a:g}$')
    ax.set_ylabel(r'$\beta = 0$')
    ax = axes[1]
    for b, c, ls in ((-0.8, st.Purple, '-'), (0.0, Navy, ':'), (0.8, st.Teal, '--')):
        ax.plot(x, levy_stable.pdf(x, 1.5, b), color=c, lw=1.2, ls=ls, label=rf'$\beta = {b:g}$')
    ax.set_ylabel(r'$\alpha = 1.5$')
    ax.set_xlabel('$x$ ($\\gamma = 1$, $\\delta_1 = 0$)')
    save(fig, 'ch3_sem_primer_densities', ncol=4)


# (5) the location shift between S1 and S0
def fig_s0s1():
    a = np.concatenate([np.linspace(0.6, 0.97, 100), np.linspace(1.03, 2.0, 200)])
    fig, ax = plt.subplots(figsize=HALF)
    for b, c in ((0.5, MainBlue), (-0.5, IDAred)):
        sh = b * np.tan(np.pi * a / 2)
        for part in (a < 1, a > 1):
            ax.plot(a[part], sh[part], color=c, lw=1.3, label=rf'$\beta = {b:+g}$' if part[0] else '_')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.axvline(1, color=Amber, lw=0.9, ls='--', label=r'$\alpha = 1$: $\tan(\pi/2) = \infty$')
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\delta_0 - \delta_1 = \beta\gamma\tan(\pi\alpha/2)$, $\gamma = 1$')
    save(fig, 'ch3_sem_primer_s0s1', ncol=2)


# (6) power-law tails on a log-log scale
def fig_tails():
    x = np.logspace(0, 2.3, 200)
    fig, ax = plt.subplots(figsize=HALF)
    ax.loglog(x, stats.norm.sf(x, scale=np.sqrt(2)), color=MainBlue, lw=1.3, label=r'Normal')
    for a, c in ((1.7, Forest), (1.5, Amber), (1.0, IDAred)):
        ax.loglog(x, levy_stable.sf(x, a, 0.0), color=c, lw=1.3, label=rf'$\alpha = {a:g}$')
    ax.loglog(x, 3 * x ** -1.5, color=Navy, lw=0.8, ls=':', label=r'$C x^{-1.5}$')
    ax.set_ylim(1e-6, 1)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'])
    ax.set_xlabel('Threshold $x$ (log scale)')
    ax.set_ylabel(r'$P(X > x)$ (log scale)')
    save(fig, 'ch3_sem_primer_tails', ncol=3)


# (7) the running sample variance
def fig_running_var():
    rng = np.random.default_rng(8)
    n = 10000
    t = np.arange(2, n + 1)
    fig, ax = plt.subplots(figsize=HALF)
    for name, x, c in ((r'Normal, $\alpha = 2$', np.sqrt(2) * rng.standard_normal(n), MainBlue),
                       (r'Stable, $\alpha = 1.7$', cms(1.7, n, rng), IDAred)):
        cs, cs2 = np.cumsum(x), np.cumsum(x ** 2)
        v = (cs2[1:] - cs[1:] ** 2 / t) / (t - 1)
        ax.plot(t, v, color=c, lw=1.0, label=name)
    ax.axhline(2, color=Forest, lw=0.8, ls='--', label='Normal variance $2\\gamma^2 = 2$')
    ax.set_xscale('log')
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(['10', '100', '1000', '10000'])
    ax.set_xlabel('Sample size $n$ (log scale)')
    ax.set_ylabel('Sample variance $s_n^2$')
    save(fig, 'ch3_sem_primer_running_var', ncol=2)


# (8) VaR and ES on a stable density
def fig_var_es():
    a = 1.7
    q = levy_stable.ppf(0.01, a, 0.0)
    xs = cms(a, 2_000_000, np.random.default_rng(3))
    es = -xs[xs <= q].mean()
    x = np.linspace(-12, 6, 500)
    f = levy_stable.pdf(x, a, 0.0)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=MainBlue, lw=1.3, label=r'$S(1.7, 0, 1, 0)$')
    xx = x[x <= q]
    ax.fill_between(xx, levy_stable.pdf(xx, a, 0.0), color=IDAred, alpha=0.5, lw=0, label='Worst 1%')
    ax.axvline(q, color=IDAred, lw=1.0, ls='--', label=f'VaR 1% = {-q:.2f}')
    ax.axvline(-es, color=Amber, lw=1.0, ls=':', label=f'ES 1% = {es:.2f}')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 0.5)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1'])
    ax.set_xlabel('Return $x$')
    ax.set_ylabel('Density (log scale)')
    save(fig, 'ch3_sem_primer_var_es', ncol=2)


# (9) CMS draws against the exact density
def fig_cms():
    rng = np.random.default_rng(12)
    a = 1.5
    x = cms(a, 200000, rng)
    bins = np.linspace(-8, 8, 81)
    fig, ax = plt.subplots(figsize=HALF)
    ax.hist(x, bins=bins, density=True, color=BandBlue, edgecolor=MainBlue, lw=0.3,
            weights=None, label='200000 CMS draws')
    xx = np.linspace(-8, 8, 400)
    ax.plot(xx, levy_stable.pdf(xx, a, 0.0), color=IDAred, lw=1.2, label=r'Exact density $S(1.5, 0, 1, 0)$')
    ax.set_xlabel('$X$')
    ax.set_ylabel('Density')
    ax.set_title(r'$U \sim U(0,1)$, $W \sim \mathrm{Exp}(1)$ $\to$ $X$', loc='left')
    save(fig, 'ch3_sem_primer_cms', ncol=1)


# (10) McCulloch: five quantiles and the ratio nu_alpha
def fig_mcculloch():
    al = np.linspace(1.0, 2.0, 41)
    nu = []
    for a in al:
        q = levy_stable.ppf([0.05, 0.25, 0.75, 0.95], a, 0.0)
        nu.append((q[3] - q[0]) / (q[2] - q[1]))
    fig, axes = plt.subplots(2, 1, figsize=HALF, gridspec_kw={'height_ratios': [1, 1.15]})
    ax = axes[0]
    x = np.linspace(-6, 6, 400)
    ax.plot(x, levy_stable.pdf(x, 1.5, 0.0), color=MainBlue, lw=1.2)
    qs = levy_stable.ppf([0.05, 0.25, 0.5, 0.75, 0.95], 1.5, 0.0)
    for qq in qs:
        ax.axvline(qq, color=IDAred, lw=0.7, ls='--')
    ax.set_xticks(qs)
    ax.set_xticklabels(['$q_{.05}$', '$q_{.25}$', '$q_{.5}$', '$q_{.75}$', '$q_{.95}$'])
    ax.tick_params(axis='x', colors=IDAred)
    ax.set_xlim(-5, 5)
    ax.set_ylabel('Density')
    ax.set_title(r'$S(1.5, 0, 1, 0)$', loc='left')
    ax = axes[1]
    ax.plot(al, nu, color=IDAred, lw=1.3, label=r'$\nu_\alpha = (q_{0.95} - q_{0.05})/(q_{0.75} - q_{0.25})$')
    ax.axhline(nu[-1], color=Forest, lw=0.8, ls='--', label=f'Normal: {nu[-1]:.3f}')
    ax.set_xlabel(r'$\alpha$ ($\beta = 0$)')
    ax.set_ylabel(r'$\nu_\alpha$')
    save(fig, 'ch3_sem_primer_mcculloch', ncol=1)


if __name__ == '__main__':
    for f in (fig_scale_rule, fig_paths, fig_charfun, fig_densities, fig_s0s1, fig_tails, fig_running_var,
              fig_var_es, fig_cms, fig_mcculloch):
        f()
