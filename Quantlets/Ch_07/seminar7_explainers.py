"""
seminar7_explainers.py -- explanatory (primer) charts for Seminar 7 (SFM): efficient markets, random walks, variance ratios
===========================================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 7, which takes
place BEFORE Lecture 7. All charts use SIMULATED data or exact formulas only (fixed seeds); they illustrate the
concepts (i.i.d. returns against uncorrelated returns with volatility clustering, the square-root-of-time rule,
i.i.d. and robust ACF bands, the chi-square law of portmanteau tests, runs of signs, unit roots and the
Dickey-Fuller distribution, variance ratios, the weights of the automatic variance ratio and the Chow-Denning
critical value) and contain no exercise answers.

The charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide), so every text
is at least 6.3 pt on the slide.
Output: charts/ch7_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_07/seminar7_explainers.py
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
import sfm_style as st   # noqa: E402

MainBlue, IDAred, Forest, Amber, Navy, BandBlue = st.MainBlue, st.IDAred, st.Forest, st.Amber, '#1F2A44', '#C5D2E8'
Purple = st.Purple
CHART_DIR = os.path.join(HERE, '..', '..', 'charts')

# Slide boxes (inches): a chart in the left column of a two-column frame (0.50 \textwidth = 2.84 in, at most
# 0.75 \textheight = 2.25 in) and a chart across the slide (\textwidth = 5.67 in, about 0.58 \textheight).
HALF = (2.84, 1.85)
FULL = (5.6, 1.35)


def style():
    st.apply()
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.2, 'xtick.labelsize': 6.6,
                         'ytick.labelsize': 6.6, 'legend.fontsize': 6.6, 'lines.linewidth': 1.0,
                         'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'legend.handlelength': 1.6,
                         'legend.columnspacing': 1.0, 'axes.titlepad': 3, 'axes.labelpad': 2})


def save(fig, name):
    st.check_no_grey(fig)
    os.makedirs(CHART_DIR, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 220})):
        fig.savefig(os.path.join(CHART_DIR, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.03, transparent=True, **kw)
    plt.close(fig)
    print('   saved', name)


def legend_below(fig, axes, ncol=3, handles=None, labels=None):
    """One legend for the figure, centred below the panels (after tight_layout)."""
    if handles is None:
        handles, labels = [], []
        for ax in np.atleast_1d(axes).ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels and not l.startswith('_'):
                    handles.append(h)
                    labels.append(l)
    fig.tight_layout(pad=0.3)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)




def garch(n, rng, omega=0.02, a=0.08, b=0.9, burn=500):
    """GARCH(1,1) returns with Normal shocks: uncorrelated, but with volatility clustering."""
    e = rng.standard_normal(n + burn)
    r = np.zeros(n + burn)
    h = omega / (1 - a - b)
    for t in range(n + burn):
        r[t] = np.sqrt(h) * e[t]
        h = omega + a * r[t] ** 2 + b * h
    return r[burn:]


def acf(x, K):
    e = x - x.mean()
    d = np.sum(e ** 2)
    return np.array([np.sum(e[k:] * e[:-k]) / d for k in range(1, K + 1)])


def delta(x, K):
    e = x - x.mean()
    d = np.sum(e ** 2)
    T = len(x)
    return np.array([T * np.sum(e[k:] ** 2 * e[:-k] ** 2) / d ** 2 for k in range(1, K + 1)])


def vr_ar1(q, phi):
    k = np.arange(1, q)
    return 1 + 2 * np.sum((1 - k / q) * phi ** k)


# =============================================================================
# 1. RW1 against RW3 / martingale: i.i.d. returns and GARCH returns
# =============================================================================
def fig_rw_types(seed=3, n=1000):
    rng = np.random.default_rng(seed)
    g = garch(n, rng)
    iid = rng.standard_normal(n) * g.std()
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True)
    for ax, x, c, title in [(axes[0], iid, MainBlue, 'RW1: i.i.d. returns'),
                            (axes[1], g, IDAred, 'RW3: GARCH returns')]:
        ax.plot(np.arange(n), x, color=c, lw=0.5)
        r1, s1 = acf(x, 1)[0], acf(x ** 2, 1)[0]
        ax.set_title(title + rf': $\hat\rho(1) = {r1:.2f}$; for $r_t^2$: ${s1:.2f}$', loc='left')
        ax.set_xlabel('day')
    axes[0].set_ylabel('return $r_t$ (%)')
    fig.tight_layout(pad=0.3)
    save(fig, 'ch7_sem_primer_rw_types')


# =============================================================================
# 2. the square-root-of-time rule and its failures
# =============================================================================
def fig_sqrt_time(sigma=1.0):
    q = np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(2.84, 1.5))
    for phi, c, ls, lab in [(0.1, IDAred, '-', r'momentum, $\rho(k) = 0.1^k$'), (0.0, MainBlue, '--', r'random walk: $\sigma\sqrt{q}$'),
                            (-0.1, Forest, '-', r'mean reversion, $\rho(k) = (-0.1)^k$')]:
        sd = np.array([sigma * np.sqrt(qq * vr_ar1(qq, phi)) for qq in q])
        ax.plot(q, sd, color=c, ls=ls, label=lab)
    ax.set_xlabel('horizon $q$ (days)')
    ax.set_ylabel(r'sd of the $q$-day return ($\sigma = 1$)')
    ax.set_title(r'$\mathrm{sd}(r_t(q)) = \sigma\sqrt{q\,\mathrm{VR}(q)}$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch7_sem_primer_sqrt_time')


# =============================================================================
# 3. ACF with the i.i.d. and the robust bands
# =============================================================================
def fig_acf_bands(seed=5, n=2500, K=20):
    rng = np.random.default_rng(seed)
    r = garch(n, rng)
    k = np.arange(1, K + 1)
    a, a2, dl = acf(r, K), acf(r ** 2, K), delta(r, K)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.bar(k, a, color=MainBlue, width=0.6, label=r'$\hat\rho(k)$')
    ax.plot(k, 1.96 * np.sqrt(dl / n), color=IDAred, lw=1.0, label=r'robust band $\pm 1.96\sqrt{\hat\delta(k)/T}$')
    ax.plot(k, -1.96 * np.sqrt(dl / n), color=IDAred, lw=1.0)
    ax.axhline(1.96 / np.sqrt(n), color=Amber, ls='--', lw=0.9, label=r'i.i.d. band $\pm 1.96/\sqrt{T}$')
    ax.axhline(-1.96 / np.sqrt(n), color=Amber, ls='--', lw=0.9)
    ax.axhline(0, color=Navy, lw=0.5)
    ax.set_xlabel('lag $k$')
    ax.set_title(f'ACF of {n} GARCH returns', loc='left')
    ax = axes[1]
    ax.bar(k, a2, color=Forest, width=0.6, label=r'ACF of $r_t^2$')
    ax.axhline(1.96 / np.sqrt(n), color=Amber, ls='--', lw=0.9)
    ax.axhline(0, color=Navy, lw=0.5)
    ax.set_xlabel('lag $k$')
    ax.set_title('ACF of the squared returns', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch7_sem_primer_acf_bands')
    return float(dl[0])


# =============================================================================
# 4. the chi-square law of the portmanteau statistics
# =============================================================================
def fig_chi2():
    x = np.linspace(0.05, 32, 500)
    fig, ax = plt.subplots(figsize=HALF)
    for m, c in [(5, MainBlue), (10, IDAred)]:
        q = stats.chi2.ppf(0.95, m)
        ax.plot(x, stats.chi2.pdf(x, m), color=c, label=rf'$\chi^2({m})$, mean {m}')
        xs = x[x >= q]
        ax.fill_between(xs, stats.chi2.pdf(xs, m), color=c, alpha=0.3, lw=0, label=f'5% beyond {q:.2f}')
    ax.set_xlabel('$Q$')
    ax.set_title(r'Law of $Q(m)$ under $H_0$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch7_sem_primer_chi2')


# =============================================================================
# 5. runs of signs
# =============================================================================
def runs_z(s):
    n1, n2 = np.sum(s > 0), np.sum(s < 0)
    n = n1 + n2
    R = 1 + np.sum(s[1:] != s[:-1])
    m = 2 * n1 * n2 / n + 1
    v = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    return R, m, (R - m) / np.sqrt(v)


def fig_runs(seed=8, n=30):
    rng = np.random.default_rng(seed)
    trend = np.repeat([1, -1, 1, -1, 1, -1], [6, 5, 7, 4, 5, 3])
    alt = np.array([1 if (i % 2 == 0) else -1 for i in range(n)])
    alt[[7, 18, 25]] *= -1
    rnd = rng.choice([1, -1], size=n)
    fig, ax = plt.subplots(figsize=(5.6, 1.15))
    rows = [(trend, 'persistent signs'), (rnd, 'random signs'), (alt, 'alternating signs')]
    for j, (s, lab) in enumerate(rows):
        y = 2 - j
        for i, v in enumerate(s):
            ax.add_patch(plt.Rectangle((i, y - 0.35), 0.9, 0.7, color=Forest if v > 0 else IDAred, lw=0))
            ax.text(i + 0.45, y, '+' if v > 0 else '$-$', ha='center', va='center', fontsize=6.6, color='white')
        R, m, z = runs_z(s)
        ax.text(n + 0.6, y, rf'$R = {R}$, $E[R] = {m:.1f}$, $z = {z:.2f}$', va='center', fontsize=6.8, color=Navy)
        ax.text(-0.6, y, lab, va='center', ha='right', fontsize=6.8, color=Navy)
    ax.set_xlim(-0.2, n + 9.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    fig.tight_layout(pad=0.2)
    save(fig, 'ch7_sem_primer_runs')


# =============================================================================
# 6. unit roots: paths and the Dickey-Fuller distribution
# =============================================================================
def fig_unit_root(seed=13, n=500, reps=4000, T=250):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    for phi, c, lab in [(1.0, IDAred, r'$\phi = 1$: unit root, $I(1)$'), (0.98, Amber, r'$\phi = 0.98$'),
                        (0.5, MainBlue, r'$\phi = 0.5$: $I(0)$')]:
        p = np.zeros(n)
        for t in range(1, n):
            p[t] = phi * p[t - 1] + e[t]
        ax.plot(p, color=c, lw=0.8, label=lab)
    ax.set_xlabel('$t$')
    ax.set_title(r'$p_t = \phi\,p_{t-1} + \varepsilon_t$, same shocks', loc='left')
    ax = axes[1]
    E = rng.standard_normal((reps, T))
    P = np.cumsum(E, axis=1)
    y = np.diff(P, axis=1)
    x = P[:, :-1]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    g = np.sum(xc * yc, axis=1) / np.sum(xc ** 2, axis=1)
    res = yc - g[:, None] * xc
    s2 = np.sum(res ** 2, axis=1) / (T - 3)
    tau = g / np.sqrt(s2 / np.sum(xc ** 2, axis=1))
    ax.hist(tau, bins=60, density=True, color=BandBlue, edgecolor=MainBlue, lw=0.2, label=r'simulated $\tau$ under $H_0$ ($\gamma = 0$)')
    z = np.linspace(-5, 4, 300)
    ax.plot(z, stats.norm.pdf(z), color=Forest, ls='--', label='$N(0, 1)$')
    ax.axvline(-2.86, color=IDAred, lw=0.9, label='DF 5% value $-2.86$')
    ax.axvline(-1.645, color=Forest, lw=0.9, ls=':', label='Normal 5% value $-1.645$')
    ax.set_xlabel(r'$\tau = \hat\gamma/\mathrm{SE}(\hat\gamma)$')
    ax.set_title('Dickey-Fuller law (with a constant)', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch7_sem_primer_unit_root')
    return float(np.quantile(tau, 0.05))


# =============================================================================
# 7. variance ratios against the horizon
# =============================================================================
def fig_vr(T=2500):
    q = np.arange(2, 41)
    fig, ax = plt.subplots(figsize=(2.84, 1.45))
    band = 1.96 * np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    ax.fill_between(q, 1 - band, 1 + band, color=BandBlue, alpha=0.8, lw=0, label='i.i.d. 95% band')
    for phi, c, lab in [(0.1, IDAred, r'momentum, $\phi = 0.1$'), (0.0, MainBlue, 'random walk'),
                        (-0.1, Forest, r'mean reversion, $\phi = -0.1$')]:
        ax.plot(q, [vr_ar1(qq, phi) for qq in q], color=c, ls='--' if phi == 0 else '-', label=lab)
    ax.set_xlabel('horizon $q$ (days)')
    ax.set_ylabel(r'VR($q$)')
    ax.set_title('Variance ratio against the horizon', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch7_sem_primer_vr')


# =============================================================================
# 8. weights of the automatic VR and the Chow-Denning critical value
# =============================================================================
def qs(x):
    x = np.asarray(x, float)
    a = 6 * np.pi * x / 5
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 25 / (12 * np.pi ** 2 * x ** 2) * (np.sin(a) / a - np.cos(a))
    return np.where(x == 0, 1.0, w)


def fig_weights_cd():
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    i = np.arange(0, 31)
    for k, c in [(5, MainBlue), (12, IDAred)]:
        ax.plot(i, qs(i / k), 'o-', ms=2, lw=0.8, color=c, label=f'quadratic spectral weights $w(i/k)$, $k = {k}$')
    ax.axhline(0, color=Navy, lw=0.5)
    ax.set_xlabel('lag $i$')
    ax.set_title(r'Automatic VR: weights of $\hat\rho(i)$', loc='left')
    ax = axes[1]
    m = np.arange(1, 11)
    c = stats.norm.ppf((1 + 0.95 ** (1 / m)) / 2)
    ax.plot(m, c, 'o-', ms=2.5, color=Forest, label=r'CD 5% critical value $c$: $(2\Phi(c) - 1)^m = 0.95$')
    ax.axhline(1.96, color=Amber, ls='--', lw=0.9, label='1.96 (one test)')
    ax.set_xlabel('number of horizons $m$')
    ax.set_title('Chow-Denning: a stricter critical value', loc='left')
    legend_below(fig, axes, ncol=2)
    save(fig, 'ch7_sem_primer_weights_cd')
    return dict(zip(m.tolist(), np.round(c, 3).tolist()))


if __name__ == '__main__':
    style()
    fig_rw_types()
    fig_sqrt_time()
    print('   delta(1)', round(fig_acf_bands(), 2))
    fig_chi2()
    fig_runs()
    print('   simulated DF 5% quantile', round(fig_unit_root(), 2))
    fig_vr()
    print('   CD critical values', fig_weights_cd())
