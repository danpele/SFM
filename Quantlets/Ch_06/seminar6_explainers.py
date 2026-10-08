"""
seminar6_explainers.py -- explanatory (primer) charts for Seminar 6 (SFM): model selection and risk management
=============================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 6, which takes
place BEFORE Lecture 6. All charts use SIMULATED data or exact formulas only (fixed seeds); they illustrate the
concepts (the log-likelihood and its maximum, the candidate distributions, AIC/BIC penalties and Akaike weights,
the chi-square law of the likelihood-ratio test, the KS distance and the Anderson-Darling weight, QQ plots, VaR on
two densities, VaR exceedances and the Kupiec statistic, the pinball loss) and contain no exercise answers.

The charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide), so every text
is at least 6.3 pt on the slide.
Output: charts/ch6_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_06/seminar6_explainers.py
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


from scipy.special import gamma as G   # noqa: E402


def unit_t(nu):
    """Student-t with nu degrees of freedom, rescaled to unit variance (nu > 2)."""
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


def unit_ged(beta):
    """GED (generalised error distribution) with shape beta and unit variance."""
    return stats.gennorm(beta, scale=np.sqrt(G(1 / beta) / G(3 / beta)))


def skewt_pdf(z, nu, lam):
    """Hansen (1994) skewed-t density with mean 0 and variance 1."""
    c = G((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * G(nu / 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    s = np.where(z < -a / b, 1 - lam, 1 + lam)
    return b * c * (1 + ((b * z + a) / s) ** 2 / (nu - 2)) ** (-(nu + 1) / 2)


# =============================================================================
# 1. the log-likelihood of a Normal sample as a function of sigma
# =============================================================================
def fig_loglik(seed=2, n=250, sigma=1.2):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.05, sigma, n)
    m = r.mean()
    s = np.linspace(0.9, 1.6, 300)
    ll = np.array([stats.norm.logpdf(r, m, x).sum() for x in s])
    sh = np.sqrt(np.mean((r - m) ** 2))
    lmax = stats.norm.logpdf(r, m, sh).sum()
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(s, ll, color=MainBlue, label=r'$\ell(\sigma)$, with $\mu = \bar r$')
    ax.plot(sh, lmax, 'o', color=IDAred, ms=4, label=rf'maximum: $\hat\sigma = {sh:.3f}$')
    ax.axhline(lmax - 1.92, color=Forest, ls='--', lw=0.8, label=r'$\ell(\hat\sigma) - 1.92$')
    inside = s[ll >= lmax - 1.92]
    ax.axvspan(inside.min(), inside.max(), color=BandBlue, alpha=0.6, lw=0, label=r'$\sigma$ not rejected by LR at 5%')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'log-likelihood $\ell$')
    ax.set_title(f'{n} simulated returns, true $\\sigma = {sigma}$', loc='left')
    ax.set_ylim(lmax - 25, lmax + 4)
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch6_sem_primer_loglik')
    return sh, inside.min(), inside.max()


# =============================================================================
# 2. the candidate densities, all with mean 0 and variance 1
# =============================================================================
def fig_candidates():
    x = np.linspace(-6, 6, 800)
    dens = [(stats.norm.pdf(x), MainBlue, '--', 'Normal'),
            (unit_t(4).pdf(x), IDAred, '-', r'Student-t, $\nu = 4$'),
            (unit_ged(1.0).pdf(x), Forest, '-', r'GED, $\beta = 1$'),
            (skewt_pdf(x, 5, -0.3), Amber, '-', r'skewed-t, $\nu = 5$, $\lambda = -0.3$')]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    for ax, log in zip(axes, [False, True]):
        for y, c, ls, lab in dens:
            ax.plot(x, y, color=c, ls=ls, label=lab)
        ax.set_xlabel('standardised return $z$')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1)
            ax.set_title('Log scale: the tails', loc='left')
        else:
            ax.set_xlim(-4, 4)
            ax.set_title('Densities, mean 0 and variance 1', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch6_sem_primer_candidates')


# =============================================================================
# 3. AIC and BIC penalties; Akaike weights
# =============================================================================
def fig_aic():
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    n = np.logspace(1, 4, 200)
    ax.plot(n, 2 * np.ones_like(n), color=MainBlue, label='AIC: 2 per parameter')
    ax.plot(n, np.log(n), color=IDAred, label=r'BIC: $\ln n$ per parameter')
    ax.set_xscale('log')
    ax.set_xlabel('number of observations $n$ (log scale)')
    ax.set_ylabel('penalty per parameter')
    ax.set_title('Penalty for one more parameter', loc='left')
    ax = axes[1]
    d = np.linspace(0, 12, 300)
    w1 = 1 / (1 + np.exp(-d / 2))
    ax.plot(d, w1, color=Forest, label=r'weight of the better model, $1/(1 + e^{-\Delta/2})$')
    ax.plot(d, 1 - w1, color=Amber, label=r'weight of the other model')
    for x0 in (2, 4, 7, 10):
        ax.axvline(x0, color=Navy, lw=0.5, ls=':')
    ax.text(0.2, 0.06, 'substantial', fontsize=6.6, color=Navy)
    ax.text(10.3, 0.06, 'none', fontsize=6.6, color=Navy)
    ax.set_xlabel(r'$\Delta$AIC of the other model')
    ax.set_ylabel('Akaike weight $w$')
    ax.set_title('Two models: Akaike weights', loc='left')
    legend_below(fig, axes, ncol=2)
    save(fig, 'ch6_sem_primer_aic')


# =============================================================================
# 4. the chi-square law of the LR statistic
# =============================================================================
def fig_lr_chi2():
    x = np.linspace(0.02, 10, 500)
    fig, ax = plt.subplots(figsize=HALF)
    for df, c in [(1, MainBlue), (2, IDAred)]:
        ax.plot(x, stats.chi2.pdf(x, df), color=c, label=rf'$\chi^2({df})$')
        q = stats.chi2.ppf(0.95, df)
        xs = x[x >= q]
        ax.fill_between(xs, stats.chi2.pdf(xs, df), color=c, alpha=0.3, lw=0,
                        label=rf'5% tail beyond {q:.2f}')
    ax.axvline(stats.chi2.ppf(0.90, 1), color=Forest, ls='--', lw=0.9,
               label=r'2.71: 5% value, 50:50 mixture')
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('$LR$')
    ax.set_title('Law of $LR$ when $H_0$ is true', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch6_sem_primer_lr_chi2')


# =============================================================================
# 5. the KS distance and the Anderson-Darling weight
# =============================================================================
def fig_ks(seed=1, n=12):
    rng = np.random.default_rng(seed)
    r = np.sort(unit_t(3).rvs(n, random_state=rng))
    u = stats.norm.cdf(r)
    i = np.arange(1, n + 1)
    dplus, dminus = i / n - u, u - (i - 1) / n
    k = int(np.argmax(np.maximum(dplus, dminus)))
    up = dplus[k] >= dminus[k]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    lo, hi = min(-3.5, r[0] - 0.4), max(3.5, r[-1] + 0.4)
    x = np.linspace(lo, hi, 400)
    ax.plot(x, stats.norm.cdf(x), color=MainBlue, label='model CDF $F$, $N(0, 1)$')
    xs = np.concatenate([[lo], r, [hi]])
    ys = np.concatenate([[0], i / n])
    ax.step(xs, np.append(ys, 1.0), where='post', color=IDAred, lw=1.0, label=f'EDF $F_n$, $n = {n}$')
    y0, y1 = (u[k], i[k] / n) if up else ((i[k] - 1) / n, u[k])
    ax.annotate('', xy=(r[k], y1), xytext=(r[k], y0), arrowprops=dict(arrowstyle='<->', color=Forest, lw=1.0))
    ax.annotate(f'$D_n = {max(dplus[k], dminus[k]):.3f}$', (r[k], (y0 + y1) / 2), xytext=(1.3, 0.35), color=Forest,
                fontsize=6.8, arrowprops=dict(arrowstyle='->', color=Forest, lw=0.6))
    ax.set_xlabel('return $x$')
    ax.set_title('KS: the largest vertical gap', loc='left')
    ax = axes[1]
    F = np.linspace(0.002, 0.998, 400)
    ax.plot(F, 1 / (F * (1 - F)), color=Amber, label=r'AD weight $1/[F(1 - F)]$')
    ax.axhline(1, color=Navy, lw=0.8, ls='--', label='CvM weight 1')
    ax.set_yscale('log')
    ax.set_xlabel('$F(x)$')
    ax.set_title('Weights of the squared gap', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch6_sem_primer_ks')


# =============================================================================
# 6. QQ plots against two fits
# =============================================================================
def fig_qq(seed=31, n=1500, nu=3):
    rng = np.random.default_rng(seed)
    r = np.sort(unit_t(nu).rvs(n, random_state=rng))
    pp = (np.arange(1, n + 1) - 0.5) / n
    qn = stats.norm.ppf(pp, r.mean(), r.std())
    df, loc, sc = stats.t.fit(r)
    qt = stats.t.ppf(pp, df, loc, sc)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(qn, r, 'o', ms=2, mfc='none', mew=0.5, color=MainBlue, label='against the Normal fit')
    ax.plot(qt, r, 'o', ms=2, mew=0, color=IDAred, label=rf'against the Student-t fit ($\hat\nu = {df:.1f}$)')
    lim = 1.05 * max(abs(r).max(), abs(qt).max())
    ax.plot([-lim, lim], [-lim, lim], color=Navy, lw=0.8, ls='--', label='45-degree line')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'model quantile $F^{-1}((i - 0.5)/n)$')
    ax.set_ylabel('sorted return $r_{(i)}$')
    ax.set_title(f'{n} simulated returns, Student-t, 3 d.f.', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch6_sem_primer_qq')


# =============================================================================
# 7. VaR 1% under two models with the same variance
# =============================================================================
def fig_var_two(nu=4):
    x = np.linspace(-6, -1, 400)
    dn, dt = stats.norm(), unit_t(nu)
    qn, qt = dn.ppf(0.01), dt.ppf(0.01)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, dn.pdf(x), color=MainBlue, ls='--', label='Normal')
    ax.plot(x, dt.pdf(x), color=IDAred, label=r'Student-t, $\nu = 4$')
    ax.fill_between(x[x <= qn], dn.pdf(x[x <= qn]), color=MainBlue, alpha=0.25, lw=0)
    ax.fill_between(x[x <= qt], dt.pdf(x[x <= qt]), color=IDAred, alpha=0.25, lw=0)
    ax.axvline(qn, color=MainBlue, lw=0.8, ls=':', label=rf'Normal $q_{{1\%}} = {qn:.2f}$')
    ax.axvline(qt, color=IDAred, lw=0.8, ls=':', label=rf'Student-t $q_{{1\%}} = {qt:.2f}$')
    ax.set_xlabel('standardised return (variance 1)')
    ax.set_title('Left tails: each red/blue area is 1%', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ch6_sem_primer_var_two')
    return qn, qt


# =============================================================================
# 8. VaR exceedances and the Kupiec statistic
# =============================================================================
def kupiec(x, n, p=0.01):
    ph = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = (n - x) * np.log(1 - ph) + (x * np.log(ph) if x > 0 else 0)
    return -2 * (l0 - l1)


def fig_kupiec(seed=3, n=1000, nu=3):
    rng = np.random.default_rng(seed)
    r = 1.2 * unit_t(nu).rvs(n, random_state=rng)
    var = -(r.mean() + r.std() * stats.norm.ppf(0.01))
    hit = r < -var
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    t = np.arange(n)
    ax.vlines(t, 0, r, color=MainBlue, lw=0.4)
    ax.axhline(-var, color=Amber, lw=1.0, ls='--', label=f'$-$VaR 1% of a Normal fit $= {-var:.2f}$')
    ax.plot(t[hit], r[hit], 'o', color=IDAred, ms=3, label=f'exceedances: $x = {hit.sum()}$ (expected 10)')
    ax.set_xlabel('day')
    ax.set_ylabel('return (%)')
    ax.set_title(f'{n} simulated Student-t returns (3 d.f.)', loc='left')
    ax = axes[1]
    xs = np.arange(0, 31)
    lr = np.array([kupiec(x, n) for x in xs])
    ax.plot(xs, lr, 'o-', color=Forest, ms=2, lw=0.8, label='$LR_{uc}$')
    ax.axhline(stats.chi2.ppf(0.95, 1), color=IDAred, lw=0.9, ls='--', label=r'$\chi^2_{0.95}(1) = 3.84$')
    ax.plot(hit.sum(), kupiec(hit.sum(), n), 'o', color=IDAred, ms=4)
    ax.set_xlabel('number of exceedances $x$')
    ax.set_title('Kupiec statistic, $n = 1000$', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'ch6_sem_primer_kupiec')
    return int(hit.sum()), float(kupiec(hit.sum(), n))


# =============================================================================
# 9. the pinball loss of the 1% quantile
# =============================================================================
def fig_pinball(tau=0.01):
    e = np.linspace(-4, 4, 400)            # e = r - q
    loss = (tau - (e < 0)) * e
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(e, loss, color=MainBlue, label=r'$(0.01 - \mathbf{1}\{r < q\})(r - q)$')
    ax.annotate('slope $-0.99$: a return below $q$\n(an exceedance) costs a lot', (-2.5, 2.48), xytext=(-1.7, 3.5),
                fontsize=6.6, color=IDAred, arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.6))
    ax.annotate('slope $0.01$: a return above $q$\ncosts very little', (2.5, 0.025), xytext=(0.2, 1.4),
                fontsize=6.6, color=Forest, arrowprops=dict(arrowstyle='->', color=Forest, lw=0.6))
    ax.set_xlabel('$r - q$')
    ax.set_ylabel('loss')
    ax.set_ylim(-0.2, 4.6)
    ax.set_title('Pinball loss of the 1% quantile $q$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'ch6_sem_primer_pinball')


if __name__ == '__main__':
    style()
    print('   loglik', fig_loglik())
    fig_candidates()
    fig_aic()
    fig_lr_chi2()
    fig_ks()
    fig_qq()
    print('   VaR quantiles', fig_var_two())
    print('   Kupiec', fig_kupiec())
    fig_pinball()
