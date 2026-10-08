"""
seminar9_explainers.py -- explanatory (primer) charts for Seminar 9 (SFM): ARCH and GARCH models
================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 9, which takes place
BEFORE Lecture 9. All charts use SIMULATED data or illustrative parameters only (fixed seeds), different from the
parameters of the exercises: conditional variance and +/- 2 sigma bands, mean reversion of the variance forecasts,
multi-day variance against the square-root-of-time rule, Normal and Student-t quantiles for VaR 1%, the
log-likelihood as a function of a parameter, news impact curves, VaR exceedances. No exercise answers.

The charts are drawn at the size they have on the slide (included at about scale 1), so every text is >= 6.5 pt.
Output: charts/sfm_ch9_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).

Run:  python3 Quantlets/Ch_09/seminar9_explainers.py

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
PREFIX = 'sfm_ch9_sem_primer_'

st.apply()
# drawn at slide size: 1 pt in the figure = 1 pt on the slide
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2})

FULL = (5.4, 1.55)      # full text width (5.67 in) of the 16:9 slide
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


def garch_sim(n, seed, omega, alpha, beta, z=None, burn=500):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn) if z is None else z
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = omega / max(1 - alpha - beta, 1e-6)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n + burn):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:], s2[burn:]


# =============================================================================
# 1. returns with +/- 2 sigma_t bands: constant variance, ARCH(1), GARCH(1,1)
# =============================================================================
def fig_paths(seed=9, n=750):
    specs = [('constant variance', 1.0, 0.0, 0.0, Forest),
             ('ARCH(1): $\\omega = 0.5$, $\\alpha = 0.5$', 0.5, 0.5, 0.0, Amber),
             ('GARCH(1,1): $\\omega = 0.05$, $\\alpha = 0.10$, $\\beta = 0.85$', 0.05, 0.10, 0.85, MainBlue)]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500)
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.6), sharey=True)
    for ax, (name, om, al, be, c) in zip(axes, specs):
        r, s2 = garch_sim(n, seed, om, al, be, z=z)
        ax.plot(r, color=c, lw=0.4, label='_')
        ax.plot(2 * np.sqrt(s2), color=IDAred, lw=0.5, label='$\\pm 2\\sigma_t$')
        ax.plot(-2 * np.sqrt(s2), color=IDAred, lw=0.5, label='_')
        ax.set_title(name, loc='left', fontsize=6.8)
        ax.set_xlabel('day')
    axes[0].set_ylabel('$r_t$ (%)')
    axes[0].set_ylim(-9, 9)
    legend_below(fig, axes, ncol=1)
    save(fig, 'paths')


# =============================================================================
# 2. mean reversion of the variance forecasts for several persistences
# =============================================================================
def fig_forecast_decay(lr=1.0, start=3.0, hmax=120):
    h = np.arange(1, hmax + 1)
    fig, ax = plt.subplots(figsize=HALF)
    for p, c in [(0.90, Forest), (0.95, Amber), (0.99, MainBlue), (1.00, IDAred)]:
        f = lr + p ** (h - 1) * (start - lr)
        lab = f'$\\alpha + \\beta = {p:.2f}$' + (' (IGARCH)' if p == 1 else '')
        ax.plot(h, f, color=c, label=lab)
    ax.axhline(lr, color=Navy, ls=':', lw=0.8, label='long-run variance $\\bar\\sigma^2$')
    ax.set_xlabel('horizon $h$ (days)')
    ax.set_ylabel('$E_t[\\sigma_{t+h}^2]$ (%$^2$)')
    ax.set_ylim(0, 3.3)
    ax.set_title('Forecasts after a high-variance day', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'forecast_decay')


# =============================================================================
# 3. H-day volatility: GARCH sum against the square-root-of-time rules
# =============================================================================
def fig_multiday(omega=0.05, alpha=0.10, beta=0.85, s2_next=3.0, Hmax=60):
    p = alpha + beta
    lr = omega / (1 - p)
    H = np.arange(1, Hmax + 1)
    garch = np.sqrt(H * lr + (s2_next - lr) * (1 - p ** H) / (1 - p))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(H, np.sqrt(H * s2_next), color=IDAred, ls='--', label='$\\sqrt{H}\\,\\sigma_{t+1}$ (today\'s level)')
    ax.plot(H, garch, color=MainBlue, lw=1.4, label='GARCH: $\\sqrt{\\Sigma_{h=1}^{H}\\, E_t[\\sigma_{t+h}^2]}$')
    ax.plot(H, np.sqrt(H * lr), color=Forest, ls='--', label='$\\sqrt{H}\\,\\bar\\sigma$ (long-run level)')
    ax.set_xlabel('holding period $H$ (days)')
    ax.set_ylabel('$H$-day volatility (%)')
    ax.set_title(f'$\\alpha + \\beta = {p:.2f}$, $\\sigma_{{t+1}}^2 = {s2_next:.0f}$, $\\bar\\sigma^2 = {lr:.0f}$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'multiday')


# =============================================================================
# 4. VaR 1%: Normal and standardised Student-t quantiles
# =============================================================================
def fig_quantiles(nu=4):
    x = np.linspace(-5, 5, 600)
    sc = np.sqrt((nu - 2) / nu)
    qn = stats.norm.ppf(0.01)
    qt = stats.t.ppf(0.01, nu) * sc
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.55), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    ax.plot(x, stats.norm.pdf(x), color=MainBlue, label='Normal $N(0, 1)$')
    ax.plot(x, stats.t.pdf(x / sc, nu) / sc, color=IDAred, label=f'standardised Student-t, $\\nu = {nu}$ (variance 1)')
    ax.axvline(qn, color=MainBlue, ls='--', lw=0.7)
    ax.axvline(qt, color=IDAred, ls='--', lw=0.7)
    ax.set_xlabel('innovation $z$')
    ax.set_ylabel('density')
    ax.set_title('Same variance, different tails', loc='left')
    ax = axes[1]
    xx = np.linspace(-5, -1.5, 300)
    ax.plot(xx, stats.norm.cdf(xx), color=MainBlue, label='_')
    ax.plot(xx, stats.t.cdf(xx / sc, nu), color=IDAred, label='_')
    ax.axhline(0.01, color=Navy, ls=':', lw=0.8)
    ax.annotate(f'Normal: $q_{{0.01}} = {qn:.3f}$', (qn, 0.01), xytext=(-5.0, 0.019), textcoords='data', color=MainBlue,
                fontsize=6.8, arrowprops=dict(arrowstyle='->', color=MainBlue, lw=0.6))
    ax.annotate(f'Student-t: $q_{{0.01}} = {qt:.3f}$', (qt, 0.01), xytext=(-4.9, 0.036), textcoords='data', color=IDAred,
                fontsize=6.8, arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.6))
    ax.set_ylim(0, 0.05)
    ax.set_xlabel('innovation $z$')
    ax.set_ylabel('$P(Z \\leq z)$')
    ax.set_title('Left tail: the 1% quantile', loc='left')
    legend_below(fig, axes, ncol=2)
    save(fig, 'quantiles')


# =============================================================================
# 5. log-likelihood of a simulated ARCH(1) as a function of alpha
# =============================================================================
def arch_loglik(r, omega, alpha):
    s2 = omega + alpha * r[:-1] ** 2
    x = r[1:]
    return -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + x ** 2 / s2)


def fig_loglik(seed=19, n=1000, omega=0.6, alpha=0.4):
    r, _ = garch_sim(n, seed, omega, alpha, 0.0)
    a = np.linspace(0.02, 0.9, 200)
    fig, ax = plt.subplots(figsize=HALF)
    for nn, c in [(250, Amber), (1000, MainBlue)]:
        ll = np.array([arch_loglik(r[:nn], omega, ai) for ai in a])
        ax.plot(a, (ll - ll.max()), color=c, label=f'$T = {nn}$ returns')
        k = np.argmax(ll)
        ax.plot(a[k], 0, 'o', color=c, ms=3.5, label='_')
    ax.axvline(alpha, color=IDAred, ls='--', lw=0.8, label=f'true $\\alpha = {alpha}$')
    ax.set_ylim(-40, 3)
    ax.set_xlabel('$\\alpha$ ($\\omega$ fixed at its true value)')
    ax.set_ylabel('$\\ell(\\alpha) - \\max\\ell$')
    ax.set_title('Log-likelihood of a simulated ARCH(1)', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'loglik')


# =============================================================================
# 6. news impact curves (illustrative parameters)
# =============================================================================
def fig_nic(s2_prev=1.0):
    e = np.linspace(-4, 4, 400)
    garch = 0.03 + 0.09 * e ** 2 + 0.88 * s2_prev
    gjr = 0.03 + (0.02 + 0.14 * (e < 0)) * e ** 2 + 0.88 * s2_prev
    z = e / np.sqrt(s2_prev)
    egarch = np.exp(0.0 + 0.11 * (np.abs(z) - np.sqrt(2 / np.pi)) - 0.07 * z + 0.97 * np.log(s2_prev))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(e, garch, color=MainBlue, label='GARCH: symmetric parabola')
    ax.plot(e, gjr, color=IDAred, label='GJR: steeper for $\\varepsilon_{t-1} < 0$')
    ax.plot(e, egarch, color=Forest, label='EGARCH: exponential, asymmetric')
    ax.axvline(0, color=Navy, lw=0.5, ls=':')
    ax.set_xlabel('shock yesterday $\\varepsilon_{t-1}$ (%)')
    ax.set_ylabel('variance today $\\sigma_t^2$')
    ax.set_title('News impact curves ($\\sigma_{t-1}^2 = 1$)', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'nic')


# =============================================================================
# 7. VaR 1% and its exceedances on simulated returns
# =============================================================================
def fig_exceedances(seed=33, n=500):
    rng = np.random.default_rng(seed)
    z = stats.t.rvs(4, size=n + 500, random_state=rng) * np.sqrt(2 / 4)
    r, s2 = garch_sim(n, seed, 0.03, 0.08, 0.90, z=z)
    var_n = -stats.norm.ppf(0.01) * np.sqrt(s2)
    hit = r < -var_n
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.4, 1.5))
    ax.plot(t, r, color=MainBlue, lw=0.5, label='return $r_t$ (%)')
    ax.plot(t, -var_n, color=Amber, lw=1.0, label='$-\\mathrm{VaR}_t$ (VaR 1%, Normal quantile)')
    ax.plot(t[hit], r[hit], 'o', color=IDAred, ms=3, label=f'exceedances: {hit.sum()} (expected $0.01 \\times {n} = {0.01 * n:.0f}$)')
    ax.set_xlabel('day')
    ax.set_ylabel('%')
    ax.set_title('Simulated returns with heavy-tailed innovations; VaR computed with a Normal quantile', loc='left')
    legend_below(fig, ax, ncol=3)
    save(fig, 'exceedances')


if __name__ == '__main__':
    fig_paths()
    fig_forecast_decay()
    fig_multiday()
    fig_quantiles()
    fig_loglik()
    fig_nic()
    fig_exceedances()
