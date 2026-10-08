"""
seminar4_explainers.py -- explanatory (primer) charts for Seminar 4 (SFM): probability for finance
==================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 4, which takes
place BEFORE Lecture 4. All charts use SIMULATED data or exact formulas only (fixed seeds); they illustrate the
concepts (PDF, CDF and quantile, correlation and dependence, diversification, mixtures of Normal laws, random walk
and AR(1), the binomial tree, the inverse transform, Monte Carlo error, the Wiener process and GBM, drawdown) and
contain no exercise answers.

The charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide), so every text
is at least 6.3 pt on the slide.
Output: charts/ch4_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_04/seminar4_explainers.py
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


# =============================================================================
# 1. PDF, CDF and quantile
# =============================================================================
def fig_cdf_quantile():
    x = np.linspace(-4, 4, 400)
    a = 0.05
    q = stats.norm.ppf(a)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.plot(x, stats.norm.pdf(x), color=MainBlue, label='PDF $f(x)$')
    xs = x[x <= q]
    ax.fill_between(xs, stats.norm.pdf(xs), color=IDAred, alpha=0.35, lw=0, label=r'area $\alpha = 5\%$ = $F(q_\alpha)$')
    ax.axvline(q, color=IDAred, ls='--', lw=0.8)
    ax.text(q + 0.15, 0.015, r'$q_{0.05} = -1.645$', fontsize=6.8, color=IDAred)
    ax.set_xlabel('$x$')
    ax.set_title('Continuous: density $f$ of $N(0, 1)$', loc='left')
    ax = axes[1]
    ax.plot(x, stats.norm.cdf(x), color=MainBlue, label='CDF $F(x) = P(X \\leq x)$')
    ax.plot([-4, q], [a, a], color=IDAred, ls='--', lw=0.8)
    ax.plot([q, q], [0, a], color=IDAred, ls='--', lw=0.8, label=r'quantile $q_\alpha = F^{-1}(\alpha)$')
    # a discrete CDF (steps) for comparison
    vals, probs = np.array([-2.0, 0.5, 2.5]), np.array([0.25, 0.45, 0.30])
    cum = np.cumsum(probs)
    edges = np.concatenate([[-4], vals, [4]])
    levels = np.concatenate([[0], cum])
    for i in range(len(levels)):
        ax.plot([edges[i], edges[i + 1]], [levels[i], levels[i]], color=Forest, lw=1.2,
                label='discrete CDF (steps)' if i == 0 else '_')
    ax.plot(vals, cum, 'o', color=Forest, ms=3)
    ax.plot(vals, cum - probs, 'o', mfc='none', color=Forest, ms=3)
    ax.set_xlabel('$x$')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('CDF: grows from 0 to 1', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch4_sem_primer_cdf_quantile')


# =============================================================================
# 2. correlation and dependence
# =============================================================================
def fig_correlation(seed=4, n=400):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    fig, axes = plt.subplots(1, 4, figsize=FULL)
    for ax, rho in zip(axes[:3], [0.8, 0.0, -0.5]):
        y = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        ax.scatter(z1, y, s=2.5, color=MainBlue, lw=0)
        r = np.corrcoef(z1, y)[0, 1]
        ax.set_title(rf'$\rho = {rho:g}$ ($\hat\rho = {r:.2f}$)', loc='left')
    ax = axes[3]
    y = z1 ** 2 - 1
    ax.scatter(z1, y, s=2.5, color=IDAred, lw=0)
    ax.set_title(rf'$Y = X^2 - 1$: $\hat\rho = {np.corrcoef(z1, y)[0, 1]:.2f}$', loc='left')
    for ax in axes:
        ax.set_xlabel('$X$')
        ax.set_xticks([-2, 0, 2])
    axes[0].set_ylabel('$Y$')
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, 'ch4_sem_primer_correlation')


# =============================================================================
# 3. diversification: portfolio volatility against the weight
# =============================================================================
def fig_diversification(s1=15.0, s2=25.0):
    w = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=HALF)
    cols = {1.0: IDAred, 0.5: Amber, 0.0: MainBlue, -0.5: Forest, -1.0: Purple}
    for rho, c in cols.items():
        sp = np.sqrt(w ** 2 * s1 ** 2 + (1 - w) ** 2 * s2 ** 2 + 2 * w * (1 - w) * rho * s1 * s2)
        ax.plot(100 * w, sp, color=c, label=rf'$\rho = {rho:g}$', lw=1.4 if rho == 1 else 1.0,
                ls='--' if rho == 1 else '-')
    ax.set_xlabel('weight $a$ of asset 1 (%), $b = 1 - a$')
    ax.set_ylabel(r'portfolio volatility $\sigma_p$ (%)')
    ax.set_title(r'$\sigma_1 = 15\%$, $\sigma_2 = 25\%$', loc='left')
    ax.set_ylim(0, 27)
    legend_below(fig, ax, ncol=3)
    save(fig, 'ch4_sem_primer_diversification')


# =============================================================================
# 4. a mixture of two Normal regimes
# =============================================================================
def fig_mixture(pc=0.9, s_calm=1.0, s_turb=3.0):
    x = np.linspace(-9, 9, 800)
    mix = pc * stats.norm.pdf(x, 0, s_calm) + (1 - pc) * stats.norm.pdf(x, 0, s_turb)
    var = pc * s_calm ** 2 + (1 - pc) * s_turb ** 2
    m4 = 3 * (pc * s_calm ** 4 + (1 - pc) * s_turb ** 4)
    kurt = m4 / var ** 2
    nor = stats.norm.pdf(x, 0, np.sqrt(var))
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    for ax, log in zip(axes, [False, True]):
        ax.plot(x, pc * stats.norm.pdf(x, 0, s_calm), color=Forest, lw=0.9, ls=':', label=r'calm: $0.9\,N(0, 1)$')
        ax.plot(x, (1 - pc) * stats.norm.pdf(x, 0, s_turb), color=Amber, lw=0.9, ls=':', label=r'turbulent: $0.1\,N(0, 3^2)$')
        ax.plot(x, mix, color=IDAred, lw=1.3, label=f'mixture (kurtosis {kurt:.1f})')
        ax.plot(x, nor, color=MainBlue, lw=1.0, ls='--', label=f'Normal, same variance {var:.1f} (kurtosis 3)')
        ax.set_xlabel('return $r$ (%)')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-6, 1)
            ax.set_title('Log scale: the tails', loc='left')
        else:
            ax.set_xlim(-6, 6)
            ax.set_title('Densities', loc='left')
    legend_below(fig, axes, ncol=2)
    save(fig, 'ch4_sem_primer_mixture')
    return kurt, var


# =============================================================================
# 5. random walk and AR(1)
# =============================================================================
def fig_rw_ar1(seed=7, n=250):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((3, n))
    fig, axes = plt.subplots(1, 3, figsize=FULL, gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
    ax = axes[0]
    cols = [MainBlue, IDAred, Forest]
    t = np.arange(n + 1)
    for i in range(3):
        ax.plot(t, np.concatenate([[0], np.cumsum(eps[i])]), color=cols[i], lw=0.8)
    ax.fill_between(t, -1.96 * np.sqrt(t), 1.96 * np.sqrt(t), color=BandBlue, alpha=0.6, lw=0,
                    label=r'$\pm 1.96\sqrt{t}\,\sigma$ (Var $= t\sigma^2$)')
    ax.set_title('Random walk $S_t = S_{t-1} + \\varepsilon_t$', loc='left')
    ax.set_xlabel('$t$')
    ax = axes[1]
    for i, phi in enumerate([0.9, 0.3]):
        x = np.zeros(n + 1)
        for k in range(n):
            x[k + 1] = phi * x[k] + eps[i, k]
        ax.plot(t, x, color=[Amber, Purple][i], lw=0.8, label=rf'AR(1), $\phi = {phi:g}$')
    ax.axhline(0, color=Navy, lw=0.6, ls=':')
    ax.set_title('AR(1) $X_t = \\phi X_{t-1} + \\varepsilon_t$', loc='left')
    ax.set_xlabel('$t$')
    ax = axes[2]
    h = np.arange(0, 11)
    for phi, c in [(0.9, Amber), (0.3, Purple)]:
        ax.plot(h, phi ** h, 'o-', color=c, ms=2.5, lw=0.9)
    ax.set_xlabel('lag $h$')
    ax.set_title(r'ACF $\rho(h) = \phi^h$', loc='left')
    ax.set_ylim(-0.05, 1.05)
    legend_below(fig, axes, ncol=3)
    save(fig, 'ch4_sem_primer_rw_ar1')


# =============================================================================
# 6. a three-step binomial tree
# =============================================================================
def fig_binomial(S0=100, u=1.2, d=0.85, p=0.6, n=3):
    fig, ax = plt.subplots(figsize=HALF)
    from math import comb
    for k in range(n + 1):
        for j in range(k + 1):              # j up moves after k steps
            S = S0 * u ** j * d ** (k - j)
            y = j - k / 2
            if k < n:
                for jj, c in [(j + 1, Forest), (j, IDAred)]:
                    ax.plot([k, k + 1], [y, jj - (k + 1) / 2], color=c, lw=0.8, zorder=1)
            ax.text(k, y, f'{S:.0f}' if abs(S - round(S)) < 0.05 else f'{S:.1f}', ha='center', va='center',
                    fontsize=6.6, zorder=3, color=Navy,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=MainBlue, lw=0.7))
            if k == n:
                ax.text(k + 0.25, y, rf'$P = {comb(n, j) * p ** j * (1 - p) ** (n - j):.3f}$', va='center',
                        fontsize=6.6, color=Navy)
    ax.plot([], [], color=Forest, label=rf'up: $\times u$, prob. $p$')
    ax.plot([], [], color=IDAred, label=rf'down: $\times d$, prob. $1 - p$')
    ax.set_xlim(-0.4, n + 1.15)
    ax.set_xticks(range(n + 1))
    ax.set_xlabel('step')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(rf'$S_0 = {S0}$, $u = {u}$, $d = {d}$, $p = {p}$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch4_sem_primer_binomial')


# =============================================================================
# 7. the inverse transform
# =============================================================================
def fig_inverse(lam=1.0, us=(0.3, 0.8), seed=3):
    x = np.linspace(0, 4, 400)
    F = 1 - np.exp(-lam * x)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.plot(x, F, color=MainBlue, label=r'CDF $F(x) = 1 - e^{-\lambda x}$, $\lambda = 1$')
    for u, c in zip(us, [Forest, IDAred]):
        xx = -np.log(1 - u) / lam
        ax.annotate('', xy=(xx, u), xytext=(0, u), arrowprops=dict(arrowstyle='->', color=c, lw=0.8))
        ax.annotate('', xy=(xx, 0), xytext=(xx, u), arrowprops=dict(arrowstyle='->', color=c, lw=0.8))
        ax.text(0.05, u + 0.04, f'$u = {u}$', color=c, fontsize=6.8)
        ax.text(xx + 0.05, 0.04, f'$x = {xx:.2f}$', color=c, fontsize=6.8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u = F(x)$')
    ax.set_ylim(0, 1.02)
    ax.set_title(r'From $u$ to $x = F^{-1}(u) = -\ln(1 - u)/\lambda$', loc='left')
    ax = axes[1]
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=5000)
    X = -np.log(1 - U) / lam
    ax.hist(X, bins=np.linspace(0, 6, 41), density=True, color=BandBlue, edgecolor=MainBlue, lw=0.3,
            label='5000 draws $F^{-1}(U)$, $U \\sim U(0, 1)$')
    xx = np.linspace(0, 6, 300)
    ax.plot(xx, lam * np.exp(-lam * xx), color=IDAred, label=r'target density $\lambda e^{-\lambda x}$')
    ax.set_xlabel('$x$')
    ax.set_title('The draws follow the target law', loc='left')
    legend_below(fig, axes, ncol=3)
    save(fig, 'ch4_sem_primer_inverse')


# =============================================================================
# 8. Monte Carlo error
# =============================================================================
def fig_montecarlo(p=0.03, seed=11, nmax=20000):
    rng = np.random.default_rng(seed)
    hits = rng.uniform(size=nmax) < p
    N = np.arange(1, nmax + 1)
    phat = np.cumsum(hits) / N
    se = np.sqrt(p * (1 - p) / N)
    fig, ax = plt.subplots(figsize=(2.84, 1.65))
    ax.fill_between(N, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se), color=BandBlue, alpha=0.8, lw=0,
                    label=r'$p \pm 1.96\sqrt{p(1 - p)/N}$')
    ax.plot(N, 100 * phat, color=MainBlue, lw=0.9, label=r'estimate $\hat p$ after $N$ simulations')
    ax.axhline(100 * p, color=IDAred, ls='--', lw=0.8, label='true $p = 3\\%$')
    ax.set_xscale('log')
    ax.set_xlim(50, nmax)
    ax.set_ylim(0, 8)
    ax.set_xlabel('number of simulations $N$ (log scale)')
    ax.set_ylabel('probability (%)')
    ax.set_title(r'Error falls like $1/\sqrt{N}$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch4_sem_primer_montecarlo')


# =============================================================================
# 9. Wiener process and GBM
# =============================================================================
def fig_wiener_gbm(seed=42, n_paths=5, years=5, q=252, mu=0.08, sigma=0.25):
    rng = np.random.default_rng(seed)
    n = years * q
    t = np.arange(n + 1) / q
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(rng.standard_normal((n_paths, n)) / np.sqrt(q), axis=1)], axis=1)
    cols = [MainBlue, IDAred, Forest, Amber, Purple]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.fill_between(t, -1.96 * np.sqrt(t), 1.96 * np.sqrt(t), color=BandBlue, alpha=0.8, lw=0,
                    label=r'95% band $\pm 1.96\sqrt{t}$')
    for i in range(n_paths):
        ax.plot(t, W[i], color=cols[i], lw=0.7, label='simulated paths' if i == 0 else '_')
    ax.set_xlabel('time $t$ (years)')
    ax.set_title(r'Wiener process $W(t) \sim N(0, t)$', loc='left')
    ax = axes[1]
    S = 100 * np.exp((mu - sigma ** 2 / 2) * t + sigma * W)
    for i in range(n_paths):
        ax.plot(t, S[i], color=cols[i], lw=0.7)
    ax.plot(t, 100 * np.exp(mu * t), color=Navy, lw=1.2, label=r'mean $S_0e^{\mu t}$')
    ax.plot(t, 100 * np.exp((mu - sigma ** 2 / 2) * t), color=Navy, lw=1.2, ls='--',
            label=r'median $S_0e^{(\mu - \sigma^2/2)t}$')
    ax.set_xlabel('time $t$ (years)')
    ax.set_title(r'GBM, $\mu = 8\%$, $\sigma = 25\%$, $S_0 = 100$', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch4_sem_primer_wiener_gbm')


# =============================================================================
# 10. drawdown of a simulated GBM path
# =============================================================================
def fig_drawdown(seed=13, years=3, q=252, mu=0.06, sigma=0.22):
    rng = np.random.default_rng(seed)
    n = years * q
    t = np.arange(n + 1) / q
    x = np.concatenate([[0.0], np.cumsum((mu - sigma ** 2 / 2) / q + sigma / np.sqrt(q) * rng.standard_normal(n))])
    P = 100 * np.exp(x)
    M = np.maximum.accumulate(P)
    DD = P / M - 1
    i_min = int(np.argmin(DD))
    i_peak = int(np.argmax(P[:i_min + 1]))
    fig, axes = plt.subplots(2, 1, figsize=(2.84, 1.8), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    ax = axes[0]
    ax.plot(t, P, color=MainBlue, lw=0.9, label='price $P_t$')
    ax.plot(t, M, color=Amber, lw=1.1, ls='--', label=r'running peak $M_t$')
    ax.fill_between(t, P, M, color=BandBlue, alpha=0.8, lw=0)
    ax.plot(t[i_peak], P[i_peak], 'o', color=Forest, ms=3.5, label='peak')
    ax.plot(t[i_min], P[i_min], 'o', color=IDAred, ms=3.5, label='trough')
    ax = axes[1]
    ax.fill_between(t, 100 * DD, 0, color=IDAred, alpha=0.35, lw=0)
    ax.plot(t, 100 * DD, color=IDAred, lw=0.8, label='drawdown (%)')
    ax.annotate(f'max. drawdown {100 * DD[i_min]:.1f}%', (t[i_min], 100 * DD[i_min]), xytext=(-80, -4),
                textcoords='offset points', fontsize=6.6, color=IDAred, arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.6))
    ax.set_ylim(100 * DD.min() * 1.4, 3)
    ax.set_xlabel('time (years)')
    legend_below(fig, axes, ncol=2)
    save(fig, 'ch4_sem_primer_drawdown')
    return 100 * DD[i_min]


if __name__ == '__main__':
    style()
    fig_cdf_quantile()
    fig_correlation()
    fig_diversification()
    k, v = fig_mixture()
    print('   mixture kurtosis', round(k, 2), 'variance', round(v, 2))
    fig_rw_ar1()
    fig_binomial()
    fig_inverse()
    fig_montecarlo()
    fig_wiener_gbm()
    print('   max drawdown', round(fig_drawdown(), 1))
