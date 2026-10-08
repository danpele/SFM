"""
seminar14_explainers.py -- Explanatory (primer) charts for Seminar 14 (SFM): crypto assets
=========================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 14, which takes place
BEFORE Lecture 14. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (the square-root
rule of annualisation, heavy tails and the Hill plot, drawdown and the recovery gain, correlation and the Fisher z
transform, a peg deviation and its half-life, VaR and ES on a density, historical simulation, a GARCH-t backtest, the
Kupiec statistic) and contain no exercise answers.

Every chart is drawn at the size of its box on the slide (half: one column of a two-column frame; full: the text
width), so that 1 pt in the figure is about 1 pt on the slide and the smallest text is 7 pt.

Output: charts/sfm_ch14_sem_primer_*.pdf and .png (transparent background, legend below the plot, no grey).

Run:  python3 Quantlets/Ch_14/seminar14_explainers.py

Statistica piețelor financiare - Daniel Traian PELE
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


def unit_t(nu):
    """Student-t with nu degrees of freedom rescaled to variance 1."""
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


# =============================================================================
# 1. annualisation: the standard deviation of h-day returns grows like sqrt(h)
# =============================================================================
def fig_sqrt_time(seed=1, s=3.5, years=60):
    rng = np.random.default_rng(seed)
    r = s * unit_t(4).rvs(365 * years, random_state=rng)
    hs = np.arange(1, 401, 3)
    emp = [np.std(np.add.reduceat(r, np.arange(0, len(r) - len(r) % h, h))[:-1], ddof=1) for h in hs]
    fig, ax = plt.subplots(figsize=HALF)
    hh = np.linspace(1, 400, 400)
    ax.plot(hh, s * np.sqrt(hh), color=MainBlue, lw=1.5, label='$\\sqrt{h}\\,s$, $s = 3.5$%')
    ax.plot(hs, emp, 'o', color=Amber, ms=2, label='simulated i.i.d. returns')
    for A, c in [(252, IDAred), (365, Forest)]:
        ax.plot([A, A], [0, s * np.sqrt(A)], color=c, lw=0.9, ls='--')
        ax.plot([0, A], [s * np.sqrt(A)] * 2, color=c, lw=0.9, ls='--',
                label=f'$h = {A}$: {s * np.sqrt(A):.1f}%')
    ax.set_xlim(0, 410)
    ax.set_ylim(0, 80)
    ax.set_xlabel('Horizon $h$ (days summed)')
    ax.set_ylabel('Std. dev. of $h$-day return (%)')
    finish(fig, 'sfm_ch14_sem_primer_sqrt_time', ncol=2, rows=2)


# =============================================================================
# 2. heavy tails: Normal against Student-t (same variance); Hill plot
# =============================================================================
def hill(x, k):
    x = np.sort(x)[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def fig_tails(seed=2, n=4000, nu=3):
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    x = np.linspace(-8, 8, 801)
    ax.semilogy(x, stats.norm.pdf(x), color=MainBlue, lw=1.4, label='Normal, variance 1')
    ax.semilogy(x, unit_t(nu).pdf(x), color=IDAred, lw=1.4, label=f'Student-$t$, $\\nu = {nu}$, variance 1')
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel('Standardised return $z$')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Heavy tails: far more extreme days', loc='left')
    ax = axes[1]
    rng = np.random.default_rng(seed)
    lt = -unit_t(nu).rvs(n, random_state=rng)
    ln = -rng.standard_normal(n)
    ks = np.arange(10, 401, 5)
    for loss, c, nm in [(lt, IDAred, 'Student-$t$ losses'), (ln, MainBlue, 'Normal losses')]:
        loss = loss[loss > 0]
        a = np.array([hill(loss, k) for k in ks])
        ax.plot(ks, a, color=c, lw=1.3, label=f'$\\hat\\alpha(k)$, {nm}')
        if c == IDAred:
            ax.fill_between(ks, a * (1 - 1.96 / np.sqrt(ks)), a * (1 + 1.96 / np.sqrt(ks)), color=c, alpha=0.15,
                            lw=0, label='95% CI $\\hat\\alpha(1 \\pm 1.96/\\sqrt{k})$')
    ax.axhline(nu, color=Navy, lw=0.9, ls='--', label=f'true $\\alpha = \\nu = {nu}$')
    ax.set_ylim(0, 10)
    ax.set_xlabel('Number of tail losses $k$')
    ax.set_ylabel('Hill $\\hat\\alpha$')
    ax.set_title(f'Hill plot, $n = {n}$ simulated days', loc='left')
    finish(fig, 'sfm_ch14_sem_primer_tails', ncol=3, rows=2)


# =============================================================================
# 3. drawdown of a simulated crypto price and the gain needed to recover
# =============================================================================
def fig_drawdown(seed=3, years=6, mu=0.5, sigma=0.6):
    rng = np.random.default_rng(seed)
    n = 365 * years
    t = np.arange(n + 1) / 365
    x = np.r_[0, np.cumsum(mu / 365 - sigma ** 2 / 730 + sigma / np.sqrt(365) * unit_t(4).rvs(n, random_state=rng))]
    P = 10 * np.exp(x)
    M = np.maximum.accumulate(P)
    D = P / M - 1
    i = int(np.argmin(D))
    j = int(np.argmax(P[:i + 1]))
    fig, axes = plt.subplots(1, 3, figsize=FULL, gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    ax = axes[0]
    ax.plot(t, P, color=MainBlue, lw=0.8, label='price $P_t$')
    ax.plot(t, M, color=Amber, lw=1.2, ls='--', label='running max $M_t$')
    ax.plot(t[j], P[j], 'o', color=Forest, ms=3.5, label='peak')
    ax.plot(t[i], P[i], 'o', color=IDAred, ms=3.5, label='trough')
    ax.set_xlabel('Years')
    ax.set_ylabel('Price $P_t$')
    ax.set_title('Simulated crypto price', loc='left')
    ax = axes[1]
    ax.fill_between(t, 100 * D, 0, color=IDAred, alpha=0.3, lw=0)
    ax.plot(t, 100 * D, color=IDAred, lw=0.8)
    ax.text(0.03, 0.05, f'MDD = {100 * D[i]:.0f}%', transform=ax.transAxes, fontsize=7, color=IDAred)
    ax.set_ylim(-105, 3)
    ax.set_xlabel('Years')
    ax.set_ylabel('Drawdown $D_t$ (%)')
    ax.set_title('Underwater curve', loc='left')
    ax = axes[2]
    d = np.linspace(0, 0.9, 200)
    ax.plot(100 * d, 100 * d / (1 - d), color=Navy, lw=1.4, label='$g = d/(1 - d)$')
    for dd in (0.5, 0.75):
        ax.plot(100 * dd, 100 * dd / (1 - dd), 'o', color=IDAred, ms=3)
        ax.annotate(f'{100 * dd:.0f}% $\\to$ {100 * dd / (1 - dd):.0f}%', (100 * dd, 100 * dd / (1 - dd)),
                    xytext=(-40, 8), textcoords='offset points', fontsize=7, color=IDAred)
    ax.set_xlabel('Loss $d$ (%)')
    ax.set_ylabel('Gain to recover $g$ (%)')
    ax.set_ylim(0, 950)
    ax.set_title('Recovery gain', loc='left')
    finish(fig, 'sfm_ch14_sem_primer_drawdown', ncol=5)


# =============================================================================
# 4. correlation in two periods; the Fisher z transform
# =============================================================================
def fig_correlation(seed=4, n=250, R=4000, rho=0.6, m=60):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    for r, c, nm in [(0.05, MainBlue, 'calm period, $\\rho = 0.05$'), (0.6, IDAred, 'crisis, $\\rho = 0.6$')]:
        z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)
        ax.scatter(z[:, 0], z[:, 1], s=3, color=c, alpha=0.7, label=f'{nm}, $\\hat\\rho$ = {np.corrcoef(z.T)[0, 1]:.2f}')
    ax.set_xlabel('Return of asset 1 (standardised)')
    ax.set_ylabel('Return of asset 2')
    ax.set_title(f'$n = {n}$ common days in each period', loc='left')
    ax = axes[1]
    S = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], (R, m))
    rh = np.array([np.corrcoef(s.T)[0, 1] for s in S])
    zz = np.arctanh(rh)
    ax.hist(rh, bins=50, density=True, color=IDAred, alpha=0.55, label=f'$\\hat\\rho$ (skewed, bounded by 1)')
    ax.hist(zz, bins=50, density=True, color=MainBlue, alpha=0.55, label='$z = \\tanh^{-1}\\hat\\rho$')
    g = np.linspace(0.1, 1.3, 300)
    ax.plot(g, stats.norm.pdf(g, np.arctanh(rho), 1 / np.sqrt(m - 3)), color=Navy, lw=1.2,
            label='$N(\\tanh^{-1}\\rho,\\ 1/(n - 3))$')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{R} samples of $n = {m}$ days, $\\rho = {rho}$', loc='left')
    finish(fig, 'sfm_ch14_sem_primer_correlation', ncol=3, rows=2)


# =============================================================================
# 5. a peg deviation: simulated stablecoin and the decay of an AR(1) deviation
# =============================================================================
def fig_peg(seed=5, n=400, phi=0.6):
    rng = np.random.default_rng(seed)
    d2 = np.zeros(n)
    for t in range(1, n):
        d2[t] = phi * d2[t - 1] + rng.normal(0, 3) - (300 if t == 250 else 0)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.plot(np.arange(n), d2, color=MainBlue, lw=0.8, label='deviation $d_t = 10\\,000(P_t - 1)$')
    ax.axhline(0, color=Navy, lw=0.6)
    ax.axhspan(-10, 10, color=Forest, alpha=0.15, lw=0, label='within $\\pm 10$ bp')
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('$d_t$ (bp)')
    ax.set_title('Simulated stablecoin: a depeg on day 250', loc='left')
    ax = axes[1]
    h = np.arange(0, 31)
    for ph, c in [(0.5, Forest), (0.9, MainBlue), (0.97, IDAred)]:
        hl = np.log(0.5) / np.log(ph)
        ax.plot(h, ph ** h, color=c, lw=1.4, marker='o', ms=1.8, label=f'$\\phi = {ph}$: half-life {hl:.1f} days')
        ax.plot([hl], [0.5], 's', color=c, ms=3.5)
    ax.axhline(0.5, color=Navy, lw=0.7, ls='--')
    ax.set_xlabel('Days after the shock $h$')
    ax.set_ylabel('Share left $\\phi^h$')
    ax.set_title('AR(1): a deviation shrinks to $\\phi^h$', loc='left')
    finish(fig, 'sfm_ch14_sem_primer_peg', ncol=3, rows=2)


# =============================================================================
# 6. VaR 1% and ES 2.5% on a density
# =============================================================================
def fig_var_es(nu=4, mu=0.1, s=3.5):
    dist = stats.t(nu, loc=mu, scale=s * np.sqrt((nu - 2) / nu))
    x = np.linspace(-20, 12, 1200)
    q1 = dist.ppf(0.01)
    q25 = dist.ppf(0.025)
    es = -dist.expect(lambda y: y, ub=q25) / 0.025
    nq1 = mu + s * stats.norm.ppf(0.01)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, stats.norm.pdf(x, mu, s), color=MainBlue, lw=1.2, ls='--', label='Normal, same $\\mu$, $\\sigma$')
    ax.plot(x, dist.pdf(x), color=Navy, lw=1.4, label=f'Student-$t$, $\\nu = {nu}$')
    xx = x[x <= q25]
    ax.fill_between(xx, dist.pdf(xx), color=Amber, alpha=0.85, lw=0, label='worst 2.5%: ES is their mean loss')
    ax.axvline(q1, color=IDAred, lw=1.2, label=f'$-$VaR 1% = {q1:.1f}%')
    ax.axvline(-es, color=Purple, lw=1.2, ls='-.', label=f'$-$ES 2.5% = {-es:.1f}%')
    ax.axvline(nq1, color=MainBlue, lw=0.9, ls=':', label=f'Normal $-$VaR 1% = {nq1:.1f}%')
    ax.set_xlim(-20, 12)
    ax.set_xlabel('Daily return $r$ (%), $\\mu = 0.1$, $\\sigma = 3.5$')
    ax.set_ylabel('Density')
    finish(fig, 'sfm_ch14_sem_primer_var_es', ncol=1, rows=6)


# =============================================================================
# 7. historical simulation on 365 simulated returns
# =============================================================================
def fig_hs(seed=7, n=365):
    rng = np.random.default_rng(seed)
    r = np.sort(3.0 * unit_t(4).rvs(n, random_state=rng))
    k1, k25 = int(np.ceil(n * 0.01)), int(np.ceil(n * 0.025))
    fig, ax = plt.subplots(figsize=HALF)
    m = 15
    ranks = np.arange(1, m + 1)
    cols = [IDAred if i < k1 else (Amber if i < k25 else MainBlue) for i in range(m)]
    ax.bar(ranks, r[:m], color=cols, width=0.7)
    ax.axhline(r[k1 - 1], color=IDAred, lw=1.0, ls='--', label=f'$r_{{({k1})}}$: VaR 1% = {-r[k1 - 1]:.1f}%')
    ax.axhline(r[:k25].mean(), color=Purple, lw=1.0, ls='-.', label=f'mean of the {k25} smallest: ES 2.5% = {-r[:k25].mean():.1f}%')
    ax.set_xticks(ranks)
    ax.set_xlabel('Rank $i$ from the smallest return')
    ax.set_ylabel('Return $r_{(i)}$ (%)')
    ax.set_title(f'{n} simulated returns: the 15 smallest', loc='left')
    finish(fig, 'sfm_ch14_sem_primer_hs', ncol=1, rows=2)


# =============================================================================
# 8. GARCH(1,1)-t returns: VaR 1% from GARCH and from a 365-day historical simulation
# =============================================================================
def fig_garch_backtest(seed=8, n=1460, w=0.05, a=0.10, b=0.88, nu=5):
    rng = np.random.default_rng(seed)
    z = unit_t(nu).rvs(n + 365, random_state=rng)
    s2 = np.empty(n + 365)
    r = np.empty(n + 365)
    s2[0] = w / (1 - a - b)
    for t in range(n + 365):
        if t:
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    q = unit_t(nu).ppf(0.01)
    vg = -np.sqrt(s2) * q
    vh = np.array([-np.quantile(r[t - 365:t], 0.01) for t in range(365, n + 365)])
    r, vg = r[365:], vg[365:]
    t = np.arange(n)
    eg, eh = r < -vg, r < -vh
    fig, ax = plt.subplots(figsize=FULL)
    ax.plot(t, r, color=MainBlue, lw=0.4, label='simulated return $r_t$')
    ax.plot(t, -vg, color=IDAred, lw=0.9, label=f'$-$VaR 1%, GARCH-$t$ ({eg.sum()} exceptions)')
    ax.plot(t, -vh, color=Amber, lw=1.1, label=f'$-$VaR 1%, HS 365 days ({eh.sum()} exceptions)')
    ax.plot(t[eh], r[eh], 'v', color=Amber, ms=3.5)
    ax.plot(t[eg], r[eg], 'o', color=IDAred, ms=2.5, mfc='none', mew=0.8)
    ax.set_xlabel(f'Day $t$ (expected exceptions: {0.01 * n:.1f} in {n} days)')
    ax.set_ylabel('Return (%)')
    finish(fig, 'sfm_ch14_sem_primer_garch_backtest', ncol=3)


# =============================================================================
# 9. the Kupiec statistic as a function of the number of exceptions
# =============================================================================
def fig_kupiec(n=1000, a=0.01):
    x = np.arange(0, 31)
    ph = np.maximum(x / n, 1e-12)
    l0 = (n - x) * np.log(1 - a) + x * np.log(a)
    l1 = (n - x) * np.log(1 - ph) + np.where(x > 0, x * np.log(ph), 0.0)
    lr = -2 * (l0 - l1)
    ok = lr <= 3.84
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(x[ok], lr[ok], color=Forest, width=0.7, label='not rejected at 5%')
    ax.bar(x[~ok], lr[~ok], color=IDAred, width=0.7, label='rejected: too few or too many')
    ax.axhline(3.84, color=Navy, lw=1.0, ls='--', label='$\\chi^2_{0.95}(1) = 3.84$')
    ax.axvline(n * a, color=Amber, lw=1.0, ls=':', label=f'expected $n\\alpha = {n * a:.0f}$')
    ax.set_ylim(0, 25)
    ax.set_xlabel(f'Exceptions $x$ in $n = {n}$ days')
    ax.set_ylabel('$LR$ (Kupiec)')
    finish(fig, 'sfm_ch14_sem_primer_kupiec', ncol=2, rows=2)


if __name__ == '__main__':
    fig_sqrt_time()
    fig_tails()
    fig_drawdown()
    fig_correlation()
    fig_peg()
    fig_var_es()
    fig_hs()
    fig_garch_backtest()
    fig_kupiec()
