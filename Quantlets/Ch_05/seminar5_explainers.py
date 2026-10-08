"""
seminar5_explainers.py -- explanatory (primer) charts for Seminar 5 (SFM): heavy tails and extreme value theory
==============================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 5, which takes
place BEFORE Lecture 5. All charts use SIMULATED data or exact formulas only (fixed seeds); they illustrate the
concepts (power tails on log-log axes, VaR and ES on a density, the Hill plot, peaks over threshold, GPD shapes,
the mean excess function, block maxima and the GEV, return levels, QQ plots, the binomial count of VaR
exceedances) and contain no exercise answers.

The charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide), so every text
is at least 6.3 pt on the slide.
Output: charts/ch5_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_05/seminar5_explainers.py
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




def unit_t(nu):
    """Student-t with nu degrees of freedom, rescaled to unit variance (nu > 2)."""
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


# =============================================================================
# 1. power tails on log-log axes
# =============================================================================
def fig_tails_loglog():
    x = np.logspace(np.log10(0.5), np.log10(30), 300)
    fig, ax = plt.subplots(figsize=HALF)
    ax.loglog(x, 2 * unit_t(3).sf(x), color=IDAred, label=r'Student-t, 3 d.f. ($\alpha = 3$)')
    ax.loglog(x, 2 * unit_t(5).sf(x), color=Amber, label=r'Student-t, 5 d.f. ($\alpha = 5$)')
    ax.loglog(x, 2 * stats.norm.sf(x), color=MainBlue, ls='--', label='Normal (light tail)')
    ax.loglog(x, np.exp(-x * np.sqrt(2)), color=Forest, ls=':', label='Laplace (exponential tail)')
    xx = np.array([4, 25])
    ax.loglog(xx, 0.25 * (xx / 4) ** -3, color=Navy, lw=0.8)
    ax.text(11, 0.04, 'slope $-3$', fontsize=6.8, color=Navy)
    ax.set_ylim(1e-7, 1)
    ax.set_xlabel('loss $x$ (in standard deviations)')
    ax.set_ylabel(r'$\bar F(x) = P(|L| > x)$')
    ax.set_title('All laws with variance 1', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch5_sem_primer_tails_loglog')


# =============================================================================
# 2. VaR and ES on the density of the losses
# =============================================================================
def fig_var_es(nu=4, p=0.01):
    d = unit_t(nu)
    var = d.ppf(1 - p)
    es = d.expect(lambda y: y, lb=var) / p
    x = np.linspace(-4, 7, 600)
    fig, (ax, ins) = plt.subplots(1, 2, figsize=HALF, gridspec_kw=dict(width_ratios=[1.25, 1]))
    xs = x[x >= var]
    for a_ in (ax, ins):
        a_.plot(x, d.pdf(x), color=MainBlue, label='density of the loss $L$')
        a_.fill_between(xs, d.pdf(xs), color=IDAred, alpha=0.35, lw=0, label=r'area $p = 1\%$')
        a_.axvline(var, color=IDAred, ls='--', lw=0.9, label=rf'VaR 1% = {var:.2f}')
        a_.axvline(es, color=Forest, ls='-.', lw=0.9, label=rf'ES 1% = {es:.2f}')
        a_.set_xlabel('loss $L$')
    ax.set_title('Student-t, 4 d.f., var. 1', loc='left')
    ins.set_xlim(2, 6.5)
    ins.set_ylim(0, 0.03)
    ins.set_title('The tail, zoomed', loc='left')
    legend_below(fig, [ax], ncol=2)
    save(fig, 'ch5_sem_primer_var_es')
    return var, es


# =============================================================================
# 3. a Hill plot of simulated Student-t losses
# =============================================================================
def hill(losses, kmax):
    L = np.sort(losses)[::-1]
    ks = np.arange(10, kmax + 1)
    a = np.array([1 / np.mean(np.log(L[:k] / L[k])) for k in ks])
    return ks, a


def fig_hill(seed=5, n=5000, nu=3):
    rng = np.random.default_rng(seed)
    L = stats.t(nu).rvs(n, random_state=rng)
    ks, a = hill(L, int(0.10 * n))
    fig, ax = plt.subplots(figsize=HALF)
    ax.fill_between(ks, a * (1 - 1.96 / np.sqrt(ks)), a * (1 + 1.96 / np.sqrt(ks)), color=BandBlue, alpha=0.8, lw=0,
                    label=r'95% band $\hat\alpha_k(1 \pm 1.96/\sqrt{k})$')
    ax.plot(ks, a, color=MainBlue, lw=1.0, label=r'Hill estimate $\hat\alpha_k$')
    ax.axhline(nu, color=IDAred, ls='--', lw=0.9, label=r'true $\alpha = 3$')
    k25 = int(0.025 * n)
    ax.axvline(k25, color=Forest, ls=':', lw=0.9, label=r'$k = 2.5\%$ of $n$')
    ax.set_ylim(0, 6)
    ax.set_xlabel('number $k$ of largest losses used')
    ax.set_ylabel(r'$\hat\alpha_k$')
    ax.set_title(f'{n} simulated losses, Student-t, 3 d.f.', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch5_sem_primer_hill')
    return float(a[k25 - 10])


# =============================================================================
# 4. peaks over threshold
# =============================================================================
def fig_pot(seed=6, n=750, nu=4):
    rng = np.random.default_rng(seed)
    L = 1.2 * stats.t(nu).rvs(n, random_state=rng) * np.sqrt((nu - 2) / nu)
    u = np.quantile(L, 0.90)
    exc = L[L > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    t = np.arange(n)
    ax.vlines(t, 0, L, color=MainBlue, lw=0.4)
    big = L > u
    ax.vlines(t[big], u, L[big], color=IDAred, lw=0.8, label='excesses $Y = L - u$')
    ax.axhline(u, color=Amber, lw=1.0, ls='--', label=f'threshold $u$ = 90% quantile = {u:.2f}')
    ax.set_xlabel('day')
    ax.set_ylabel('loss $L$ (%)')
    ax.set_title(f'{n} simulated daily losses; {big.sum()} above $u$', loc='left')
    ax = axes[1]
    ax.hist(exc[exc < 5], bins=np.linspace(0, 5, 16), density=True, color=BandBlue, edgecolor=MainBlue, lw=0.3, label='histogram of the excesses')
    y = np.linspace(0, 5, 200)
    ax.plot(y, stats.genpareto.pdf(y, xi, scale=beta), color=IDAred, lw=1.2,
            label=rf'fitted GPD, $\hat\xi = {xi:.2f}$, $\hat\beta = {beta:.2f}$')
    ax.set_xlabel('excess $y$ (%)')
    ax.set_title(f'Excesses and the GPD ({(exc >= 5).sum()} above 5)', loc='left')
    legend_below(fig, axes, ncol=2)
    save(fig, 'ch5_sem_primer_pot')
    return u, xi, beta


# =============================================================================
# 5. GPD densities for several shapes
# =============================================================================
def fig_gpd_shapes(beta=1.0):
    y = np.linspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=HALF)
    for xi, c, ls in [(-0.3, Forest, ':'), (0.0, MainBlue, '--'), (0.3, Amber, '-'), (0.6, IDAred, '-')]:
        pdf = stats.genpareto.pdf(y, xi, scale=beta)
        lab = {0.0: r'$\xi = 0$: exponential'}.get(xi, rf'$\xi = {xi:g}$')
        if xi < 0:
            lab += r' (bounded at $\beta/|\xi|$)'
        ax.plot(y, pdf, color=c, ls=ls, label=lab)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel('excess $y$')
    ax.set_ylabel('density (log scale)')
    ax.set_title(r'GPD densities, $\beta = 1$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch5_sem_primer_gpd_shapes')


# =============================================================================
# 6. the empirical mean excess function
# =============================================================================
def mean_excess(x, vs):
    return np.array([np.mean(x[x > v] - v) for v in vs])


def fig_mean_excess(seed=9, n=20000):
    rng = np.random.default_rng(seed)
    samples = [(unit_t(3).rvs(n, random_state=rng), IDAred, r'Student-t, 3 d.f.: rising'),
               (stats.laplace(scale=1 / np.sqrt(2)).rvs(n, random_state=rng), Forest, 'Laplace: flat'),
               (stats.norm.rvs(size=n, random_state=rng), MainBlue, 'Normal: falling')]
    fig, ax = plt.subplots(figsize=HALF)
    for x, c, lab in samples:
        vs = np.linspace(0, np.quantile(x, 0.995), 60)
        ax.plot(vs, mean_excess(x, vs), color=c, label=lab)
    ax.set_xlabel('threshold $v$ (in standard deviations)')
    ax.set_ylabel(r'$\hat e(v)$')
    ax.set_title(f'Empirical mean excess, {n} draws each', loc='left')
    ax.set_ylim(0, None)
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch5_sem_primer_mean_excess')


# =============================================================================
# 7. block maxima and the three GEV types
# =============================================================================
def fig_block_maxima(seed=21, months=12, days=21, nu=4):
    rng = np.random.default_rng(seed)
    L = stats.t(nu).rvs(months * days, random_state=rng) * np.sqrt((nu - 2) / nu)
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    t = np.arange(months * days)
    for m in range(months):
        if m % 2 == 0:
            ax.axvspan(m * days - 0.5, (m + 1) * days - 0.5, color=BandBlue, alpha=0.5, lw=0)
        blk = L[m * days:(m + 1) * days]
        j = int(np.argmax(blk))
        ax.plot(m * days + j, blk[j], 'o', color=IDAred, ms=3, label='block maximum $M$' if m == 0 else '_')
    ax.vlines(t, 0, L, color=MainBlue, lw=0.5)
    ax.set_xlabel('day (shaded bands: blocks of 21 days)')
    ax.set_ylabel('loss')
    ax.set_title('Block maxima: the largest loss of each month', loc='left')
    ax = axes[1]
    x = np.linspace(-2, 6, 400)
    for xi, c, ls, lab in [(0.3, IDAred, '-', r'Fréchet, $\xi = 0.3$'), (0.0, MainBlue, '--', r'Gumbel, $\xi = 0$'),
                           (-0.3, Forest, ':', r'Weibull, $\xi = -0.3$')]:
        ax.plot(x, stats.genextreme.pdf(x, -xi), color=c, ls=ls, label=lab)   # scipy uses c = -xi
    ax.set_xlabel('$x$ ($\\mu = 0$, $\\sigma = 1$)')
    ax.set_title('GEV densities', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch5_sem_primer_block_maxima')


# =============================================================================
# 8. return levels
# =============================================================================
def return_level(m, mu, sigma, xi):
    y = -np.log(1 - 1 / m)
    return mu - sigma * np.log(y) if xi == 0 else mu + sigma / xi * (y ** (-xi) - 1)


def fig_return_level(mu=2.0, sigma=1.0):
    m = np.logspace(np.log10(1.5), 3, 200)
    fig, ax = plt.subplots(figsize=HALF)
    for xi, c, ls in [(0.3, IDAred, '-'), (0.0, MainBlue, '--'), (-0.3, Forest, ':')]:
        ax.plot(m, return_level(m, mu, sigma, xi), color=c, ls=ls, label=rf'$\xi = {xi:g}$')
    ax.axvline(40, color=Amber, lw=0.8, ls='-.', label='$m = 40$ quarters = 10 years')
    ax.set_xscale('log')
    ax.set_xlabel('return period $m$ (blocks, log scale)')
    ax.set_ylabel('return level $z_m$')
    ax.set_title(r'GEV with $\mu = 2$, $\sigma = 1$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch5_sem_primer_return_level')


# =============================================================================
# 9. QQ plot of excesses
# =============================================================================
def fig_qq(seed=87, n=150, xi=0.3, beta=1.0):
    rng = np.random.default_rng(seed)
    y = np.sort(stats.genpareto.rvs(xi, scale=beta, size=n, random_state=rng))
    pp = (np.arange(1, n + 1) - 0.5) / n
    xf, _, bf = stats.genpareto.fit(y, floc=0)
    fig, ax = plt.subplots(figsize=HALF)
    q_gpd = stats.genpareto.ppf(pp, xf, scale=bf)
    q_exp = stats.expon.ppf(pp, scale=y.mean())
    ax.plot(q_exp, y, 'o', ms=2.5, mfc='none', color=MainBlue, mew=0.6, label='against an exponential law')
    ax.plot(q_gpd, y, 'o', ms=2.5, color=IDAred, label=rf'against the fitted GPD ($\hat\xi = {xf:.2f}$)')
    lim = max(y.max(), q_gpd.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=Navy, lw=0.8, ls='--', label='45-degree line')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('model quantile $F^{-1}((i - 0.5)/N_u)$')
    ax.set_ylabel('sorted excess $y_{(i)}$')
    ax.set_title(f'{n} simulated excesses, GPD with $\\xi = 0.3$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch5_sem_primer_qq')


# =============================================================================
# 10. the binomial count of VaR exceedances
# =============================================================================
def fig_binom_test(n=1000, p=0.01):
    x = np.arange(0, 26)
    pmf = stats.binom.pmf(x, n, p)
    pval = np.array([min(1, 2 * min(stats.binom.cdf(k, n, p), stats.binom.sf(k - 1, n, p))) for k in x])
    ok = pval >= 0.05
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(x[ok], pmf[ok], color=MainBlue, width=0.8, label='p-value $\\geq$ 5%: VaR not rejected')
    ax.bar(x[~ok], pmf[~ok], color=IDAred, width=0.8, label='p-value < 5%: VaR rejected')
    ax.axvline(n * p, color=Amber, ls='--', lw=0.9, label='expected count $np = 10$')
    ax.set_xlabel('number of exceedances $x$')
    ax.set_ylabel('$P(X = x)$')
    ax.set_title(r'$X \sim$ Binomial$(1000, 0.01)$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch5_sem_primer_binom_test')
    return x[ok].min(), x[ok].max()


if __name__ == '__main__':
    style()
    fig_tails_loglog()
    print('   VaR, ES', fig_var_es())
    print('   Hill at 2.5%', round(fig_hill(), 2))
    print('   POT u, xi, beta', fig_pot())
    fig_gpd_shapes()
    fig_mean_excess()
    fig_block_maxima()
    fig_return_level()
    fig_qq()
    print('   accepted counts', fig_binom_test())
