"""
seminar8_explainers.py -- explanatory (primer) charts for Seminar 8 (SFM): volatility estimators and clustering
===============================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 8, which takes place
BEFORE Lecture 8. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (precision of a
sample volatility, rolling windows and the ghost effect, EWMA weights, OHLC prices of a day, range-based estimators,
volatility clustering and the ACF, the chi-square test, the noisy proxy r_t^2) and contain no exercise answers.

The charts are drawn at the size they have on the slide (included at scale 1), so every text is >= 6.5 pt.
Output: charts/sfm_ch8_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).

Run:  python3 Quantlets/Ch_08/seminar8_explainers.py

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
PREFIX = 'sfm_ch8_sem_primer_'

st.apply()
# drawn at slide size: 1 pt in the figure = 1 pt on the slide
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2})

FULL = (5.5, 1.75)      # full text width (5.67 in) of the 16:9 slide
HALF = (2.65, 1.85)      # one column of a two-column frame (2.72 in)


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


def garch_path(n, seed, omega=0.02, alpha=0.08, beta=0.90):
    """Simulated GARCH(1,1) returns (%, Normal innovations) and their conditional variances."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500)
    s2 = np.empty(n + 500)
    r = np.empty(n + 500)
    s2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n + 500):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[500:], s2[500:]


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:-k]) / d for k in range(1, nlags + 1)])


# =============================================================================
# 1. precision of a sample volatility: sampling distributions for n = 21, 63, 252 days
# =============================================================================
def fig_precision(seed=8, sigma_ann=20.0, reps=20000):
    rng = np.random.default_rng(seed)
    sd = sigma_ann / np.sqrt(252)
    fig, ax = plt.subplots(figsize=HALF)
    grid = np.linspace(5, 35, 400)
    for n, c in [(21, IDAred), (63, Amber), (252, MainBlue)]:
        est = np.sqrt(252) * rng.normal(0, sd, size=(reps, n)).std(axis=1, ddof=1)
        kde = stats.gaussian_kde(est)
        ax.plot(grid, kde(grid), color=c, label=f'$n = {n}$')
        lo, hi = sigma_ann * (1 - 1.96 / np.sqrt(2 * n)), sigma_ann * (1 + 1.96 / np.sqrt(2 * n))
        ax.hlines(-0.012 - 0.012 * [21, 63, 252].index(n), lo, hi, color=c, lw=2.2)
    ax.axvline(sigma_ann, color=Navy, ls='--', lw=0.8, label='true $\\sigma = 20\\%$')
    ax.set_xlabel('estimated annual volatility (%)')
    ax.set_ylabel('density')
    ax.set_ylim(-0.045, None)
    ax.set_title('Sampling distribution of $\\hat\\sigma$; bars: $\\pm 1.96/\\sqrt{2n}$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'precision')


# =============================================================================
# 2. rolling window against EWMA after one large day: the ghost effect
# =============================================================================
def fig_ghost(seed=81, n=220, shock_day=60, lam=0.94):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1.0, n)
    r[shock_day] = -8.0
    t = np.arange(n)
    roll = np.full(n, np.nan)
    for i in range(63, n):
        roll[i] = np.sqrt(np.mean(r[i - 63:i] ** 2))          # forecast for day i from days i-63 .. i-1
    s2 = np.empty(n)
    s2[0] = 1.0
    for i in range(1, n):
        s2[i] = lam * s2[i - 1] + (1 - lam) * r[i - 1] ** 2
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 1.75), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax = axes[0]
    ax.bar(t, r, color=MainBlue, width=1.0, label='daily return $r_t$ (%)')
    ax.bar([shock_day], [r[shock_day]], color=IDAred, width=2.5, label='one large day ($-8\\%$)')
    ax.set_xlabel('day')
    ax.set_ylabel('$r_t$ (%)')
    ax.set_title('Returns: true volatility constant', loc='left')
    ax = axes[1]
    ax.plot(t, np.sqrt(252) * roll, color=Amber, lw=1.2, label='63-day window')
    ax.plot(t, np.sqrt(252 * s2), color=MainBlue, lw=1.2, label=f'EWMA, $\\lambda = {lam}$')
    ax.axhline(np.sqrt(252), color=Navy, ls=':', lw=0.8, label='true volatility (15.9%)')
    ax.axvline(shock_day + 64, color=IDAred, ls='--', lw=0.7)
    ax.annotate('the large day\nleaves the window', (shock_day + 64, np.sqrt(252) * roll[shock_day + 63]),
                xytext=(10, 4), textcoords='offset points', fontsize=6.8, color=IDAred, va='center')
    ax.set_xlabel('day')
    ax.set_ylabel('annual volatility (%)')
    ax.set_title('Rolling window and EWMA', loc='left')
    legend_below(fig, axes, ncol=5)
    save(fig, 'ghost')


# =============================================================================
# 3. EWMA weights against the flat weights of a rolling window
# =============================================================================
def fig_weights():
    j = np.arange(1, 81)
    fig, ax = plt.subplots(figsize=HALF)
    for lam, c in [(0.85, IDAred), (0.94, MainBlue)]:
        ax.plot(j, (1 - lam) * lam ** (j - 1), color=c, marker='o', ms=1.6, lw=0.8,
                label=f'EWMA, $\\lambda = {lam}$')
    ax.step(np.r_[1, j], np.r_[1 / 21, np.where(j <= 21, 1 / 21, 0)], where='post', color=Forest, lw=1.0,
            label='21-day window: $1/21$')
    hl = np.log(0.5) / np.log(0.94)
    ax.axvline(hl, color=MainBlue, ls='--', lw=0.7)
    ax.annotate(f'half-life of $\\lambda = 0.94$:\n$\\ln 0.5/\\ln 0.94 = {hl:.1f}$ days', (hl, 0.105), xytext=(6, 0),
                textcoords='offset points', fontsize=6.8, color=MainBlue, va='center')
    ax.set_xlabel('age $j$ of the squared return (days)')
    ax.set_ylabel('weight')
    ax.set_title('Weight of $r_{t-j}^2$ in $\\sigma_t^2$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'weights')


# =============================================================================
# 4. OHLC prices of one day: night, session, high, low
# =============================================================================
def fig_ohlc(seed=12, steps=390):
    rng = np.random.default_rng(seed)
    x1 = 100 * np.exp(np.cumsum(np.r_[0, rng.normal(0, 0.0007, steps - 1)]))
    x2 = x1[-1] * 1.012 * np.exp(np.cumsum(np.r_[0, rng.normal(0.00003, 0.0007, steps - 1)]))
    t1 = np.linspace(0, 1, steps)
    t2 = np.linspace(1.35, 2.35, steps)
    fig, ax = plt.subplots(figsize=(2.65, 1.95))
    ax.plot(t1, x1, color=Amber, lw=0.8, label='day $t-1$')
    ax.plot(t2, x2, color=MainBlue, lw=0.8, label='day $t$')
    ax.plot([t1[-1], t2[0]], [x1[-1], x2[0]], color=IDAred, ls=':', lw=1.0, label='night (no trading)')
    O, H, L, C = x2[0], x2.max(), x2.min(), x2[-1]
    pts = [(t1[-1], x1[-1], '$C_{t-1}$', (3, -11), 'left'), (t2[0], O, '$O_t$', (-3, 6), 'right'),
           (t2[np.argmax(x2)], H, '$H_t$', (0, 4), 'center'), (t2[np.argmin(x2)], L, '$L_t$', (0, -9), 'center'),
           (t2[-1], C, '$C_t$', (-2, -11), 'center')]
    for x, y, lab, off, ha in pts:
        ax.plot(x, y, 'o', color=IDAred if lab != '$C_{t-1}$' else Amber, ms=3)
        ax.annotate(lab, (x, y), xytext=off, textcoords='offset points', fontsize=7.5, ha=ha)
    ax.axhline(H, xmin=0.55, color=Forest, lw=0.5, ls='--')
    ax.axhline(L, xmin=0.55, color=Forest, lw=0.5, ls='--')
    ax.annotate('', (2.6, L), (2.6, H), arrowprops=dict(arrowstyle='<->', color=Forest, lw=0.8),
                annotation_clip=False)
    ax.text(2.66, (H + L) / 2, 'range\n$\\ln(H_t/L_t)$', color=Forest, fontsize=6.8, va='center')
    ax.set_xlim(-0.05, 3.25)
    ax.set_xticks([0.5, 1.85])
    ax.set_xticklabels(['session $t-1$', 'session $t$'])
    ax.set_ylabel('price')
    ax.set_title('Open, high, low, close', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'ohlc')


# =============================================================================
# 5. range-based estimators: sampling distributions with and without a night
# =============================================================================
def sim_ohlc(rng, days, sigma_day, night_share, steps=1000):
    """OHLC log prices (%) of `days` days of a driftless Brownian motion; night_share of the daily variance
    falls between the close and the next open."""
    s_night = np.sqrt(night_share) * sigma_day
    s_step = np.sqrt((1 - night_share) / steps) * sigma_day
    o = rng.normal(0, s_night, days)
    path = np.cumsum(rng.normal(0, s_step, (days, steps)), axis=1)
    path = np.concatenate([np.zeros((days, 1)), path], axis=1)
    u, d, c = path.max(axis=1), path.min(axis=1), path[:, -1]
    return o, u, d, c


def estimators(o, u, d, c):
    n = len(o)
    r = o + c
    cc = np.mean((r - r.mean()) ** 2) * n / (n - 1)
    park = np.mean((u - d) ** 2) / (4 * np.log(2))
    gk = np.mean(0.5 * (u - d) ** 2 - (2 * np.log(2) - 1) * c ** 2)
    rs = np.mean(u * (u - c) + d * (d - c))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = o.var(ddof=1) + k * c.var(ddof=1) + (1 - k) * rs
    return dict(cc=cc, park=park, gk=gk, yz=yz)


def fig_estimators(seed=88, reps=3000, n=21, sigma_ann=20.0):
    rng = np.random.default_rng(seed)
    sd = sigma_ann / np.sqrt(252)
    lab = {'cc': 'close-to-close', 'park': 'Parkinson', 'gk': 'Garman--Klass', 'yz': 'Yang--Zhang'}
    col = {'cc': Navy, 'park': IDAred, 'gk': Amber, 'yz': Forest}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.75), sharey=True)
    grid = np.linspace(8, 32, 400)
    for ax, share, title in [(axes[0], 0.0, 'Market open 24 hours (no night)'),
                             (axes[1], 0.25, 'Night = 25% of the daily variance')]:
        est = {k: [] for k in lab}
        for _ in range(reps):
            e = estimators(*sim_ohlc(rng, n, sd, share))
            for k in lab:
                est[k].append(np.sqrt(252 * max(e[k], 1e-12)))
        for k in lab:
            ax.plot(grid, stats.gaussian_kde(est[k])(grid), color=col[k], label=lab[k])
        ax.axvline(sigma_ann, color=Navy, ls=':', lw=0.8, label='true volatility (20%)')
        ax.set_title(title, loc='left')
        ax.set_xlabel(f'annual volatility estimated from {n} days (%)')
    axes[0].set_ylabel('density')
    legend_below(fig, axes, ncol=5)
    save(fig, 'estimators')


# =============================================================================
# 6. volatility clustering: i.i.d. returns against GARCH returns, ACF of r_t^2
# =============================================================================
def fig_clustering(seed=2026, n=1500, lags=30):
    r_g, _ = garch_path(n, seed)
    rng = np.random.default_rng(seed + 1)
    r_i = rng.normal(0, r_g.std(), n)
    band = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 2.0), gridspec_kw=dict(height_ratios=[1, 1]))
    for j, (r, name, c) in enumerate([(r_i, 'i.i.d. Normal returns', Forest), (r_g, 'returns with clustering', MainBlue)]):
        ax = axes[0, j]
        ax.plot(r, color=c, lw=0.4)
        ax.set_title(name + ' (same variance)', loc='left')
        ax.set_ylabel('$r_t$ (%)')
        ax.set_ylim(-6.5, 6.5)
        ax = axes[1, j]
        ax.bar(np.arange(1, lags + 1), acf(r ** 2, lags), color=c, width=0.6, label='_')
        ax.axhspan(-band, band, color=BandBlue, alpha=0.9, lw=0, zorder=0, label='95% band $\\pm 1.96/\\sqrt{T}$')
        ax.set_ylim(-0.08, 0.32)
        ax.set_xlabel('lag $k$ (days)')
        ax.set_ylabel('ACF of $r_t^2$')
    legend_below(fig, axes, ncol=1)
    save(fig, 'clustering')


# =============================================================================
# 7. chi-square reference distribution: critical value and p-value
# =============================================================================
def fig_chi2(m=5, q_obs=14.0):
    x = np.linspace(0, 22, 500)
    f = stats.chi2.pdf(x, m)
    crit = stats.chi2.ppf(0.95, m)
    p = stats.chi2.sf(q_obs, m)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=MainBlue, label=f'density of $\\chi^2({m})$')
    ax.fill_between(x, 0, f, where=x >= crit, color=IDAred, alpha=0.35, lw=0, label='rejection region (5%)')
    ax.fill_between(x, 0, f, where=x >= q_obs, color=IDAred, alpha=0.9, lw=0, label=f'p-value of $Q = {q_obs:.0f}$')
    ax.axvline(crit, color=IDAred, ls='--', lw=0.8)
    ax.annotate(f'critical value\n$\\chi^2_{{0.95}}({m}) = {crit:.2f}$', (crit, 0.06), xytext=(5, 6),
                textcoords='offset points', fontsize=6.8, color=IDAred)
    ax.annotate(f'p = {p:.3f}', (q_obs + 0.6, stats.chi2.pdf(q_obs, m) + 0.002), xytext=(8, 14),
                textcoords='offset points', fontsize=6.8, color=IDAred,
                arrowprops=dict(arrowstyle='->', color=IDAred, lw=0.6))
    ax.set_xlabel('value of the test statistic')
    ax.set_ylabel('density')
    ax.set_ylim(0, 0.17)
    ax.set_title('Reference distribution under $H_0$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'chi2')


# =============================================================================
# 8. the proxy r_t^2: unbiased but noisy
# =============================================================================
def fig_proxy(seed=7, n=250):
    r, s2 = garch_path(n, seed, omega=0.05, alpha=0.10, beta=0.85)
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.5, 1.6))
    ax.plot(t, r ** 2, 'o', color=Amber, ms=1.6, label='proxy $r_t^2$')
    ax.plot(t, s2, color=MainBlue, lw=1.2, label='true variance $\\sigma_t^2$ (never observed)')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel('day')
    ax.set_ylabel('$\\%^2$ (log scale)')
    ax.set_title('$E(r_t^2) = \\sigma_t^2$, but a single $r_t^2$ is often far above or far below $\\sigma_t^2$', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'proxy')


if __name__ == '__main__':
    fig_precision()
    fig_ghost()
    fig_weights()
    fig_ohlc()
    fig_estimators()
    fig_clustering()
    fig_chi2()
    fig_proxy()
