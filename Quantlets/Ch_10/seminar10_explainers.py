"""
seminar10_explainers.py -- explanatory (primer) charts for Seminar 10 (SFM): VaR, ES and backtesting
=====================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 10, which takes place
BEFORE Lecture 10. All charts use SIMULATED data or textbook distributions only (fixed seeds), with parameters that
differ from those of the exercises: VaR and ES on a density, historical simulation on sorted returns, the
Cornish-Fisher quantile against the kurtosis, a GPD fitted to a tail (EVT), the binomial distribution of VaR
exceptions with the Basel zones, independent against clustered exceptions, Gaussian against t copula. No answers.

The charts are drawn at the size they have on the slide (included at about scale 1), so every text is >= 6.5 pt.
Output: charts/sfm_ch10_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).

Run:  python3 Quantlets/Ch_10/seminar10_explainers.py

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
PREFIX = 'sfm_ch10_sem_primer_'

st.apply()
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2})

HALF = (2.65, 1.85)     # one column of a two-column frame (2.72 in)


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


# =============================================================================
# 1. VaR 1% and ES 2.5% on a Normal density
# =============================================================================
def fig_var_es(sigma=1.0, alpha=0.01):
    x = np.linspace(-4.2, 4.2, 600)
    f = stats.norm.pdf(x, 0, sigma)
    q = stats.norm.ppf(alpha) * sigma
    es = -sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=MainBlue, label='density of the return $X$, $N(0, 1)$')
    ax.fill_between(x, 0, f, where=x <= q, color=IDAred, alpha=0.8, lw=0, label='worst 1% of days')
    ax.axvline(q, color=IDAred, ls='--', lw=0.8, label=f'$-\\mathrm{{VaR}}_{{1\\%}} = {q:.2f}$: the 1% quantile')
    ax.axvline(es, color=Navy, ls='--', lw=0.8, label=f'$-\\mathrm{{ES}}_{{1\\%}} = {es:.2f}$: mean of the red tail')
    ax.set_xlabel('daily return $X$ (in units of $\\sigma$)')
    ax.set_ylabel('density')
    ax.set_xlim(-4.2, 4.2)
    ax.set_title('Same level: ES lies beyond VaR', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'var_es')


# =============================================================================
# 2. historical simulation: the k worst of n simulated returns
# =============================================================================
def fig_hs(seed=10, n=400, alpha=0.025):
    rng = np.random.default_rng(seed)
    r = np.sort(stats.t.rvs(4, size=n, random_state=rng) * np.sqrt(0.5) * 1.3)
    k = int(np.ceil(n * alpha))
    fig, ax = plt.subplots(figsize=HALF)
    m = 25
    i = np.arange(1, m + 1)
    ax.bar(i, r[:m], color=[IDAred if j <= k else MainBlue for j in i], width=0.7)
    ax.bar([0], [0], color=IDAred, label=f'the $k = \\lceil n\\alpha \\rceil = {k}$ worst days ($n = {n}$, $\\alpha = 2.5\\%$)')
    ax.bar([0], [0], color=MainBlue, label='the other days (sorted)')
    ax.axhline(r[k - 1], color=Navy, ls='--', lw=0.8, label=f'$r_{{({k})}} = -\\widehat{{\\mathrm{{VaR}}}}$')
    ax.axhline(r[:k].mean(), color=Amber, ls='--', lw=1.0, label='mean of the $k$ worst $= -\\widehat{\\mathrm{ES}}$')
    ax.set_xlabel('rank $i$ (1 = the worst day)')
    ax.set_ylabel('$r_{(i)}$ (%)')
    ax.set_xlim(0.3, m + 0.7)
    ax.set_title('Sorted simulated returns: the left tail', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'hs')


# =============================================================================
# 3. Cornish-Fisher 1% quantile as a function of the excess kurtosis
# =============================================================================
def fig_cf(alpha=0.01):
    K = np.linspace(0, 14, 300)
    z = stats.norm.ppf(alpha)
    fig, ax = plt.subplots(figsize=HALF)
    for S, c in [(0.0, MainBlue), (-0.6, IDAred)]:
        zcf = z + (z ** 2 - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S ** 2 / 36
        ax.plot(K, zcf, color=c, label=f'Cornish--Fisher, $S = {S}$')
    Kt = np.linspace(0.05, 14, 300)
    nus = 4 + 6 / Kt
    qt = stats.t.ppf(alpha, nus) * np.sqrt((nus - 2) / nus)
    ok = Kt <= 14
    ax.plot(Kt[ok], qt[ok], color=Forest, lw=1.3, label='exact: standardised Student-t, $K = 6/(\\nu - 4)$')
    ax.axhline(z, color=Navy, ls=':', lw=0.8, label=f'Normal: $z_{{0.01}} = {z:.3f}$')
    ax.set_xlabel('excess kurtosis $K$')
    ax.set_ylabel('1% quantile')
    ax.set_title('The expansion is reliable only for small $K$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'cf')


# =============================================================================
# 4. EVT: GPD fitted to the losses above a threshold
# =============================================================================
def fig_evt(seed=5, n=5000):
    rng = np.random.default_rng(seed)
    loss = stats.t.rvs(3, size=n, random_state=rng)
    u = np.quantile(loss, 0.9)
    exc = loss[loss > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    ls = np.sort(loss)[::-1]
    emp = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=HALF)
    sel = ls > 0.3
    ax.plot(ls[sel], emp[sel], 'o', color=MainBlue, ms=1.4, label='empirical $P(L > x)$')
    xx = np.linspace(u, ls.max() * 1.1, 200)
    ax.plot(xx, 0.1 * stats.genpareto.sf(xx - u, xi, 0, beta), color=IDAred, lw=1.3,
            label=f'GPD above $u$: $\\hat\\xi = {xi:.2f}$, $\\hat\\beta = {beta:.2f}$')
    ax.axvline(u, color=Amber, ls='--', lw=0.8, label='threshold $u$ (90% quantile)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.3, None)
    ax.set_xlabel('loss $x$ (log scale)')
    ax.set_ylabel('$P(L > x)$ (log scale)')
    ax.set_title('Tail of 5000 simulated losses', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'evt')


# =============================================================================
# 5. binomial distribution of VaR 1% exceptions in 250 days, Basel zones
# =============================================================================
def fig_binomial(n=250, p=0.01):
    x = np.arange(0, 13)
    pmf = stats.binom.pmf(x, n, p)
    col = [Forest if v <= 4 else (Amber if v <= 9 else IDAred) for v in x]
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(x, pmf, color=col, width=0.7)
    for c, lab in [(Forest, 'green zone: 0--4'), (Amber, 'yellow zone: 5--9'), (IDAred, 'red zone: 10 or more')]:
        ax.bar([0], [0], color=c, label=lab)
    ax.axvline(n * p, color=Navy, ls=':', lw=0.8, label=f'expected number $np = {n * p:.1f}$')
    ax.set_xticks(x[::2])
    ax.set_xlabel(f'number of exceptions $x$ in $n = {n}$ days')
    ax.set_ylabel('$P(X = x)$')
    ax.set_title('Binomial($250$; $0.01$): a correct VaR 1%', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'binomial')


# =============================================================================
# 6. independent against clustered exceptions with the same number of hits
# =============================================================================
def fig_clusters(seed=4, n=1000, x=12):
    rng = np.random.default_rng(seed)
    ind = np.sort(rng.choice(n, x, replace=False))
    starts = np.sort(rng.choice(np.arange(50, n - 50), 4, replace=False))
    clu = np.concatenate([s + np.array([0, 4, 9]) for s in starts])
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 1.25), sharex=True)
    for ax, hits, c, name in [(axes[0], ind, MainBlue, 'independent exceptions'), (axes[1], clu, IDAred, 'clustered exceptions')]:
        ax.vlines(hits, 0, 1, color=c, lw=1.2)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0, 1.2)
        ax.set_title(f'{name}: {len(hits)} days with $I_t = 1$ out of {n}', loc='left', fontsize=7)
    axes[1].set_xlabel('day $t$')
    fig.tight_layout(pad=0.3)
    save(fig, 'clusters')


# =============================================================================
# 7. Gaussian against t copula with the same correlation
# =============================================================================
def fig_copulas(seed=21, n=2500, rho=0.5, nu=3):
    rng = np.random.default_rng(seed)
    cov = np.array([[1, rho], [rho, 1]])
    zg = rng.multivariate_normal([0, 0], cov, n)
    ug = stats.norm.cdf(zg)
    zt = rng.multivariate_normal([0, 0], cov, n) / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    ut = stats.t.cdf(zt, nu)
    q = 0.05
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0))
    for ax, u, name in [(axes[0], ug, 'Gaussian copula'), (axes[1], ut, f't copula, $\\nu = {nu}$')]:
        both = (u[:, 0] <= q) & (u[:, 1] <= q)
        ax.plot(u[~both, 0], u[~both, 1], 'o', color=MainBlue, ms=0.8, alpha=0.6, label='pseudo-observations $(U_1, U_2)$')
        ax.plot(u[both, 0], u[both, 1], 'o', color=IDAred, ms=2.0, label=f'both below $q = {q}$')
        ax.add_patch(plt.Rectangle((0, 0), q, q, fill=False, ec=IDAred, lw=0.8))
        ax.set_title(f'{name}, $\\rho = {rho}$:\n{both.sum()} points in the joint lower tail', loc='left')
        ax.set_xlabel('$U_1$')
        ax.set_ylabel('$U_2$')
        ax.set_aspect('equal')
    legend_below(fig, axes, ncol=2)
    save(fig, 'copulas')


if __name__ == '__main__':
    fig_var_es()
    fig_hs()
    fig_cf()
    fig_evt()
    fig_binomial()
    fig_clusters()
    fig_copulas()
