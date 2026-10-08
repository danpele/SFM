"""
seminar2_explainers.py -- Explanatory (primer) charts for Seminar 2 (SFM): classical distributions and stylised facts
==================================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 2, which takes
place BEFORE Lecture 2. All charts use SIMULATED data or exact formulas (fixed seeds): the Normal density and its
quantiles, k-sigma probabilities, the Poisson distribution, lognormal prices, the CLT, the binomial approximation,
the contribution of extreme days to the kurtosis, the chi-squared distribution of the Jarque-Bera statistic,
QQ plots, the Student-t, the log-likelihood, the ACF, aggregation and the leverage effect of a simulated GARCH
series. They contain no exercise answers.

The charts are drawn at the size of their box on the slide (half-slide column, about 2.7 x 2.2 inches), so that
1 pt in the figure is about 1 pt on the slide (tick labels 7 pt).

Output: charts/ch2_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom, no grey).

Run:  python3 Quantlets/Ch_02/seminar2_explainers.py

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


def unit_t(nu):
    """Student-t with nu degrees of freedom, rescaled to unit variance (nu > 2)."""
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


def gjr(n, seed=1, omega=0.02, a=0.03, g=0.10, b=0.90, nu=6):
    """A GJR-GARCH(1,1) series with unit-variance Student-t shocks (daily returns in %)."""
    rng = np.random.default_rng(seed)
    z = unit_t(nu).rvs(n + 500, random_state=rng)
    r = np.empty(n + 500)
    h = omega / (1 - a - g / 2 - b)
    for t in range(n + 500):
        r[t] = np.sqrt(h) * z[t]
        h = omega + (a + g * (r[t] < 0)) * r[t] ** 2 + b * h
    return r[500:]


def acf(x, L):
    x = x - x.mean()
    d = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / d for k in range(1, L + 1)])


# (1) the Normal density, a probability and a quantile
def fig_normal():
    mu, s = 0.05, 1.1
    x = np.linspace(mu - 4.2 * s, mu + 4.2 * s, 600)
    f = stats.norm.pdf(x, mu, s)
    q = mu + s * stats.norm.ppf(0.05)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=MainBlue, lw=1.3, label=r'$N(\mu, \sigma^2)$, $\mu = 0.05$, $\sigma = 1.1$')
    xx = x[x <= q]
    ax.fill_between(xx, stats.norm.pdf(xx, mu, s), color=IDAred, alpha=0.45, lw=0, label=r'$P(X \leq x_{0.05}) = 0.05$')
    ax.axvline(mu, color=Navy, lw=0.8, ls=':')
    for k in (-2, -1, 1, 2):
        ax.axvline(mu + k * s, color=Forest, lw=0.5, ls='--')
    ax.annotate(r'$x_{0.05} = \mu - 1.645\sigma$', (q, stats.norm.pdf(q, mu, s)), xytext=(0.02, 0.62),
                textcoords='axes fraction', ha='left', fontsize=7, color=IDAred,
                arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.7))
    ax.set_xticks([mu + k * s for k in (-3, -2, -1, 0, 1, 2, 3)])
    ax.set_xticklabels([r'$\mu{-}3\sigma$', '', r'$\mu{-}\sigma$', r'$\mu$', r'$\mu{+}\sigma$', '', r'$\mu{+}3\sigma$'])
    ax.set_xlabel('Daily return $x$ (%)')
    ax.set_ylabel('Density $f(x)$')
    ax.set_ylim(0, 0.43)
    save(fig, 'ch2_sem_primer_normal', ncol=1)


# (2) probability of a k-sigma day
def fig_ksigma():
    k = np.linspace(1, 6, 200)
    fig, ax = plt.subplots(figsize=HALF)
    ax.semilogy(k, 2 * stats.norm.sf(k), color=MainBlue, lw=1.3, label='Normal')
    ax.semilogy(k, 2 * unit_t(4).sf(k), color=IDAred, lw=1.3, label='Student-$t(4)$, variance 1')
    for kk in (3, 4, 5):
        p = 2 * stats.norm.sf(kk)
        ax.plot(kk, p, 'o', color=MainBlue, ms=3)
        ax.annotate(f'1 day in {1 / p:,.0f}', (kk, p), xytext=(-3, -3), textcoords='offset points', ha='right',
                    va='top', fontsize=7, color=MainBlue)
    ax.set_xlabel('Threshold $k$ (standard deviations)')
    ax.set_ylabel(r'$P(|Z| > k)$ (log scale)')
    ax.set_ylim(1e-9, 1)
    save(fig, 'ch2_sem_primer_ksigma', ncol=2)


# (3) the Poisson distribution and a p-value
def fig_poisson():
    lam, obs = 1.5, 6
    k = np.arange(0, 11)
    p = stats.poisson.pmf(k, lam)
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(k[k < obs], p[k < obs], color=MainBlue, width=0.7, label=r'Poisson, mean $\lambda = 1.5$')
    ax.bar(k[k >= obs], p[k >= obs], color=IDAred, width=0.7, label=f'$P(N \\geq {obs}) = {stats.poisson.sf(obs - 1, lam):.4f}$')
    ax.annotate(f'observed: {obs}', (obs, p[obs]), xytext=(0, 14), textcoords='offset points', ha='center', fontsize=7,
                color=IDAred, arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.7))
    ax.set_xlabel('Number of $k$-sigma days $N$')
    ax.set_ylabel('$P(N = j)$')
    ax.set_xticks(k)
    save(fig, 'ch2_sem_primer_poisson', ncol=1)


# (4) lognormal prices after 1 and 5 years
def fig_lognormal():
    P0, mu, s = 100, 0.05, 0.30
    y = np.linspace(1, 450, 800)
    fig, ax = plt.subplots(figsize=HALF)
    for T, c in ((1, MainBlue), (5, IDAred)):
        m, v = np.log(P0) + mu * T, s * np.sqrt(T)
        d = stats.lognorm(v, scale=np.exp(m))
        ax.plot(y, d.pdf(y), color=c, lw=1.3, label=f'$P_T$, $T = {T}$')
        ax.axvline(np.exp(m), color=c, lw=0.8, ls='--', label='Median $e^m$' if T == 1 else '_')
        ax.axvline(np.exp(m + v ** 2 / 2), color=c, lw=0.8, ls=':', label='Mean $e^{m + s^2/2}$' if T == 1 else '_')
    ax.set_xlabel('Price after $T$ years ($P_0 = 100$)')
    ax.set_ylabel('Density')
    ax.set_title(r'$\mu = 0.05$, $\sigma = 0.30$ a year', loc='left')
    save(fig, 'ch2_sem_primer_lognormal', ncol=2)


# (5) CLT: standardised sums of skewed variables
def fig_clt_sums():
    rng = np.random.default_rng(5)
    x = np.linspace(-3.5, 4.5, 400)
    fig, ax = plt.subplots(figsize=HALF)
    for n, c in ((1, Amber), (4, IDAred), (30, Forest)):
        X = rng.exponential(1.0, (200000, n)).sum(axis=1)
        Zs = (X - n) / np.sqrt(n)
        h, e = np.histogram(Zs, bins=np.linspace(-3.5, 4.5, 81), density=True)
        ax.step(0.5 * (e[1:] + e[:-1]), h, where='mid', color=c, lw=1.0, label=f'$n = {n}$')
    ax.plot(x, stats.norm.pdf(x), color=MainBlue, lw=1.3, ls='--', label='$N(0, 1)$')
    ax.set_xlabel(r'Standardised sum $(X_1 + \cdots + X_n - n\mu)/(\sigma\sqrt{n})$')
    ax.set_ylabel('Density')
    ax.set_title('Sums of exponential variables', loc='left')
    ax.set_ylim(0, 1.0)
    save(fig, 'ch2_sem_primer_clt_sums', ncol=4)


# (6) the Normal approximation of a binomial, with the continuity correction
def fig_binomial():
    n, p, k0 = 40, 0.5, 25
    k = np.arange(8, 33)
    pm = stats.binom.pmf(k, n, p)
    m, sd = n * p, np.sqrt(n * p * (1 - p))
    x = np.linspace(8, 32, 400)
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(k[k < k0], pm[k < k0], color=BandBlue, edgecolor=MainBlue, lw=0.3, width=1.0, label='$B(40, 0.5)$')
    ax.bar(k[k >= k0], pm[k >= k0], color=IDAred, alpha=0.5, edgecolor=IDAred, lw=0.3, width=1.0,
           label=f'$P(X \\geq {k0})$')
    ax.plot(x, stats.norm.pdf(x, m, sd), color=MainBlue, lw=1.3, label=r'$N(np, np(1-p))$')
    ax.axvline(k0 - 0.5, color=Navy, lw=0.9, ls='--')
    ax.annotate(f'${k0} - 0.5$', (k0 - 0.5, 0.11), xytext=(3, 0), textcoords='offset points', fontsize=7, color=Navy)
    ax.set_xlabel('Number of successes $X$')
    ax.set_ylabel('Probability')
    save(fig, 'ch2_sem_primer_binomial', ncol=2)


# (7) how much of m4 comes from the most extreme days
def fig_kurt_contrib():
    rng = np.random.default_rng(14)
    n = 4000
    fig, ax = plt.subplots(figsize=HALF)
    for name, x, c in (('Normal', rng.standard_normal(n), MainBlue),
                       ('Student-$t(4)$', unit_t(4).rvs(n, random_state=rng), IDAred)):
        d4 = np.sort((x - x.mean()) ** 4)[::-1]
        share = 100 * np.cumsum(d4) / d4.sum()
        frac = 100 * np.arange(1, n + 1) / n
        ax.plot(frac, share, color=c, lw=1.3, label=name)
    ax.axvline(1, color=Forest, lw=0.8, ls='--', label='1% of the days')
    ax.set_xscale('log')
    ax.set_xlim(0.025, 100)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.set_xlabel('Most extreme days (% of the sample)')
    ax.set_ylabel(r'Share of $\sum (r_t - \bar r)^4$ (%)')
    ax.set_title(f'{n} simulated days', loc='left')
    save(fig, 'ch2_sem_primer_kurt_contrib', ncol=3)


# (8) chi-squared(2) and the 5% critical value of JB
def fig_jb_chi2():
    x = np.linspace(0, 14, 400)
    c = stats.chi2.ppf(0.95, 2)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, stats.chi2.pdf(x, 2), color=MainBlue, lw=1.3, label=r'$\chi^2(2)$: JB under normality')
    xx = x[x >= c]
    ax.fill_between(xx, stats.chi2.pdf(xx, 2), color=IDAred, alpha=0.45, lw=0, label='Rejection region, 5%')
    ax.axvline(c, color=IDAred, lw=0.9, ls='--')
    ax.annotate(f'{c:.2f}', (c, 0.2), xytext=(3, 0), textcoords='offset points', fontsize=7, color=IDAred)
    ax.set_xlabel('JB')
    ax.set_ylabel('Density')
    save(fig, 'ch2_sem_primer_jb_chi2', ncol=1)


# (9) QQ plots
def fig_qq():
    rng = np.random.default_rng(21)
    n = 2000
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig, ax = plt.subplots(figsize=HALF)
    for name, x, c, m in (('Normal sample', rng.standard_normal(n), MainBlue, 'o'),
                          ('Heavy-tailed sample, $t(3)$', unit_t(3).rvs(n, random_state=rng), IDAred, 's')):
        z = np.sort((x - x.mean()) / x.std())
        ax.plot(q, z, m, color=c, ms=1.6, mew=0, label=name)
    ax.plot([-4, 4], [-4, 4], color=Forest, lw=1.0, ls='--', label='45-degree line')
    ax.set_xlabel(r'Normal quantile $\Phi^{-1}((i - 0.5)/n)$')
    ax.set_ylabel(r'Sorted data $z_{(i)}$')
    ax.set_xlim(-4, 4)
    save(fig, 'ch2_sem_primer_qq', ncol=2)


# (10) Student-t densities with unit variance
def fig_student_t():
    x = np.linspace(-6, 6, 600)
    fig, axes = plt.subplots(2, 1, figsize=(HALF[0], 2.0), sharex=True)
    for ax, log in zip(axes, (False, True)):
        f = ax.semilogy if log else ax.plot
        f(x, stats.norm.pdf(x), color=MainBlue, lw=1.2, label=r'Normal ($\nu = \infty$)')
        f(x, unit_t(5).pdf(x), color=Forest, lw=1.2, label=r'$t(5)$')
        f(x, unit_t(3).pdf(x), color=IDAred, lw=1.2, label=r'$t(3)$')
        ax.set_ylabel('Density (log)' if log else 'Density')
    axes[1].set_ylim(1e-4, 1)
    axes[1].set_yticks([1e-4, 1e-2, 1])
    axes[1].set_yticklabels(['0.0001', '0.01', '1'])
    axes[1].set_xlabel('Return with variance 1')
    save(fig, 'ch2_sem_primer_student_t', ncol=3)


# (11) the log-likelihood as a function of nu
def fig_mle():
    rng = np.random.default_rng(33)
    x = 0.8 * stats.t(5).rvs(3000, random_state=rng)
    nus = np.linspace(2.5, 15, 120)
    ll = []
    for nu in nus:
        sc = stats.t.fit(x, f0=nu, floc=0)[2]
        ll.append(stats.t.logpdf(x, nu, scale=sc).sum())
    ll = np.array(ll)
    lln = stats.norm.logpdf(x, x.mean(), x.std()).sum()
    i = int(np.argmax(ll))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(nus, ll, color=MainBlue, lw=1.3, label=r'$\ell(\nu)$, Student-$t$')
    ax.axhline(lln, color=IDAred, lw=1.0, ls='--', label=r'$\ell$, Normal')
    ax.plot(nus[i], ll[i], 'o', color=Forest, ms=3.5, label=rf'Maximum: $\hat\nu = {nus[i]:.1f}$')
    ax.set_xlabel(r'Degrees of freedom $\nu$')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('3000 draws from a $t(5)$', loc='left')
    save(fig, 'ch2_sem_primer_mle', ncol=2)


# (12) ACF of returns and of absolute returns
def fig_acf():
    r = gjr(4000, seed=4)
    L = 20
    band = 1.96 / np.sqrt(len(r))
    lags = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.axhspan(-band, band, color=BandBlue, alpha=0.9, lw=0, label=r'Band $\pm 1.96/\sqrt{n}$')
    ax.bar(lags - 0.2, acf(r, L), width=0.4, color=MainBlue, label=r'ACF of $r_t$')
    ax.bar(lags + 0.2, acf(np.abs(r), L), width=0.4, color=IDAred, label=r'ACF of $|r_t|$')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Lag $h$ (days)')
    ax.set_ylabel(r'$\hat\rho(h)$')
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title('Simulated GARCH returns, $n = 4000$', loc='left')
    save(fig, 'ch2_sem_primer_acf', ncol=2)


# (13) aggregation: excess kurtosis of h-day sums
def fig_aggregation():
    r = gjr(400000, seed=8, a=0.05, g=0.08, b=0.88, nu=8)
    hs = [1, 2, 5, 10, 21, 63]
    K = []
    for h in hs:
        m = len(r) // h
        K.append(stats.kurtosis(r[:m * h].reshape(m, h).sum(axis=1)))
    x = np.arange(len(hs))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, [6 / h for h in hs], 's-', color=Forest, lw=1.2, ms=3.2, label=r'i.i.d. $t(5)$: $K_h = 6/h$')
    ax.plot(x, K, 'o-', color=IDAred, lw=1.2, ms=3.2, label='Simulated GARCH')
    ax.axhline(0, color=MainBlue, lw=1.0, ls='--', label='Normal, $K = 0$')
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel('Horizon $h$ (days)')
    ax.set_ylabel('Excess kurtosis $K_h$')
    save(fig, 'ch2_sem_primer_aggregation', ncol=2)


# (14) the leverage effect
def fig_leverage():
    r = gjr(20000, seed=9)
    L = 10
    ks = np.arange(1, L + 1)
    lev = [np.corrcoef(r[:-k], np.abs(r[k:]))[0, 1] for k in ks]
    band = 1.96 / np.sqrt(len(r))
    fig, ax = plt.subplots(figsize=HALF)
    ax.axhspan(-band, band, color=BandBlue, alpha=0.9, lw=0, label=r'Band $\pm 1.96/\sqrt{n}$')
    ax.bar(ks, lev, width=0.6, color=IDAred, label=r'$L(k) = \mathrm{Corr}(r_t, |r_{t+k}|)$')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Lag $k$ (days)')
    ax.set_ylabel('$L(k)$')
    ax.set_xticks(ks)
    ax.set_title('Simulated asymmetric GARCH', loc='left')
    save(fig, 'ch2_sem_primer_leverage', ncol=1)


if __name__ == '__main__':
    for f in (fig_normal, fig_ksigma, fig_poisson, fig_lognormal, fig_clt_sums, fig_binomial, fig_kurt_contrib,
              fig_jb_chi2, fig_qq, fig_student_t, fig_mle, fig_acf, fig_aggregation, fig_leverage):
        f()
