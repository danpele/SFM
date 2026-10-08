"""
seminar1_explainers.py -- Explanatory (primer) charts for Seminar 1 (SFM): data, returns and indicators
=====================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 1, which takes
place BEFORE Lecture 1. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (OHLC bars,
adjusted prices, log returns, the shape of a distribution, volatility drag, standard errors and tests, the bootstrap,
downside deviation, drawdown, the gain needed to recover, correlation) and contain no exercise answers.

The charts are drawn at the size of their box on the slide (half-slide column, about 2.7 x 2.2 inches), so that
1 pt in the figure is about 1 pt on the slide (tick labels 7 pt).

Output: charts/ch1_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom, no grey).

Run:  python3 Quantlets/Ch_01/seminar1_explainers.py

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


def gbm(seed, n, mu=0.0003, sigma=0.012, p0=100.0):
    rng = np.random.default_rng(seed)
    r = mu + sigma * rng.standard_normal(n)
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


from scipy import stats   # noqa: E402


# (1) OHLC bars
def fig_ohlc():
    rng = np.random.default_rng(4)
    n = 15
    C = 50 * np.exp(np.cumsum(0.015 * rng.standard_normal(n)))
    O = np.concatenate([[50.0], C[:-1]]) * np.exp(0.004 * rng.standard_normal(n))
    H = np.maximum(O, C) * np.exp(np.abs(0.008 * rng.standard_normal(n)))
    L = np.minimum(O, C) * np.exp(-np.abs(0.008 * rng.standard_normal(n)))
    fig, ax = plt.subplots(figsize=HALF)
    for i in range(n):
        col = Forest if C[i] >= O[i] else IDAred
        ax.vlines(i, L[i], H[i], color=col, lw=0.9)
        ax.hlines(O[i], i - 0.32, i, color=col, lw=1.2)
        ax.hlines(C[i], i, i + 0.32, color=col, lw=1.2)
    ax.plot([], [], color=Forest, lw=1.2, label='Up day ($C_t > O_t$)')
    ax.plot([], [], color=IDAred, lw=1.2, label='Down day ($C_t < O_t$)')
    k = 11
    kw = dict(textcoords='offset points', fontsize=7.2, color=Navy)
    ax.annotate('$H_t$', (k, H[k]), xytext=(0, 2), ha='center', va='bottom', **kw)
    ax.annotate('$L_t$', (k, L[k]), xytext=(0, -2), ha='center', va='top', **kw)
    ax.annotate('$O_t$', (k - 0.32, O[k]), xytext=(-2, 0), ha='right', va='center', **kw)
    ax.annotate('$C_t$', (k + 0.32, C[k]), xytext=(2, 0), ha='left', va='center', **kw)
    ax.set_xticks(range(0, n, 2))
    ax.set_xlabel('Trading day $t$')
    ax.set_ylabel('Price (lei)')
    ax.set_title('Daily bars: open, high, low, close', loc='left')
    save(fig, 'ch1_sem_primer_ohlc', ncol=1)


# (2) raw and adjusted prices around a dividend and a split
def fig_adjusted():
    rng = np.random.default_rng(8)
    n = 60
    r = 0.012 * rng.standard_normal(n)
    v = 30 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))      # value of one original share (no events)
    raw = v.copy()
    td, D = 20, 1.5
    ts = 40
    raw[td:] = raw[td:] - D * v[td:] / v[td]      # ex-dividend: the price drops by D
    raw[ts:] = raw[ts:] / 2                        # 2-for-1 split
    F_div = 1 - D / raw[td - 1]
    adj = raw.copy()
    adj[:ts] *= 0.5
    adj[:td] *= F_div
    t = np.arange(n + 1)
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
    ax = axes[0]
    ax.plot(t, raw, color=IDAred, lw=1.0, label='Raw close')
    ax.plot(t, adj, color=MainBlue, lw=1.0, label='Adjusted close')
    for x, lab in ((td, 'dividend'), (ts, 'split 2-for-1')):
        ax.axvline(x, color=Amber, lw=0.7, ls='--')
        ax.annotate(lab, (x, ax.get_ylim()[1] if False else raw.max()), xytext=(2, 0), textcoords='offset points',
                    fontsize=7, color=Amber, va='top')
    ax.set_ylabel('Price (lei)')
    ax = axes[1]
    rr = 100 * (raw[1:] / raw[:-1] - 1)
    ra = 100 * (adj[1:] / adj[:-1] - 1)
    ax.vlines(t[1:] - 0.15, 0, rr, color=IDAred, lw=0.8)
    ax.vlines(t[1:] + 0.15, 0, ra, color=MainBlue, lw=0.8)
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_ylabel('$R_t$ (%)')
    ax.set_xlabel('Day $t$')
    save(fig, 'ch1_sem_primer_adjusted', ncol=2)


# (3) log and simple returns, with the second-order approximation
def fig_log_simple():
    R = np.linspace(-0.5, 0.5, 400)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(100 * R, 100 * R, color=Forest, lw=1.0, ls='--', label='$r = R$')
    ax.plot(100 * R, 100 * (R - R ** 2 / 2), color=Amber, lw=1.0, ls=':', label=r'$r \approx R - R^2/2$')
    ax.plot(100 * R, 100 * np.log1p(R), color=MainBlue, lw=1.4, label=r'$r = \ln(1 + R)$')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.axvline(0, color=Navy, lw=0.4)
    ax.axvspan(-5, 5, color=BandBlue, alpha=0.7, lw=0)
    ax.annotate('daily moves', (0, 30), ha='center', fontsize=7, color=MainBlue)
    ax.set_xlabel('Simple return $R$ (%)')
    ax.set_ylabel('Log return $r$ (%)')
    save(fig, 'ch1_sem_primer_log_simple', ncol=3)


def unit_t(nu):
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


# (4) the shape of a distribution: skewness and excess kurtosis
def fig_shape():
    x = np.linspace(-5, 5, 600)
    a = -4.0
    d = a / np.sqrt(1 + a ** 2)
    sc = 1 / np.sqrt(1 - 2 * d ** 2 / np.pi)
    loc = -sc * d * np.sqrt(2 / np.pi)
    sk = stats.skewnorm(a, loc=loc, scale=sc)
    fig, axes = plt.subplots(2, 1, figsize=(HALF[0], 1.85))
    ax = axes[0]
    ax.plot(x, stats.norm.pdf(x), color=MainBlue, lw=1.2, label='Normal: $S = K = 0$')
    ax.plot(x, sk.pdf(x), color=IDAred, lw=1.2, label=f'$S = {sk.stats(moments="s"):.2f}$: longer left tail')
    ax.set_ylabel('Density')
    ax.set_xlim(-4.5, 4.5)
    ax = axes[1]
    ax.semilogy(x, stats.norm.pdf(x), color=MainBlue, lw=1.2)
    ax.semilogy(x, unit_t(5).pdf(x), color=Forest, lw=1.2, label='$K = 6$: heavier tails')
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(-4.5, 4.5)
    ax.set_yticks([1e-4, 1e-2, 1])
    ax.set_yticklabels(['0.0001', '0.01', '1'])
    ax.set_ylabel('Density (log)')
    ax.set_xlabel('Standardised return $(r - \\bar r)/s$')
    save(fig, 'ch1_sem_primer_shape', ncol=1)


# (5) volatility drag: same arithmetic mean, different volatility
def fig_vol_drag():
    mu = 0.10
    T = np.linspace(0, 10, 200)
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=HALF)
    for i in range(12):
        y = np.concatenate([[0.0], np.cumsum(np.log1p(mu + 0.40 * rng.standard_normal(10)))])
        y = np.clip(y, -6, None)
        ax.plot(np.arange(11), 100 * np.exp(y), color=IDAred, lw=0.4, alpha=0.6,
                label=r'Paths, $\sigma = 40\%$' if i == 0 else '_')
    ax.plot(T, 100 * (1 + mu) ** T, color=Navy, lw=1.3, ls='--', label=r'$100\,(1 + \bar R)^t$, $\bar R = 10\%$')
    ax.plot(T, 100 * np.exp((np.log1p(mu) - 0.10 ** 2 / 2) * T), color=MainBlue, lw=1.3,
            label=r'Typical path, $\sigma = 10\%$')
    ax.plot(T, 100 * np.exp((np.log1p(mu) - 0.40 ** 2 / 2) * T), color=IDAred, lw=1.3,
            label=r'Typical path, $\sigma = 40\%$')
    ax.set_yscale('log')
    ax.set_yticks([50, 100, 200, 400])
    ax.set_yticklabels(['50', '100', '200', '400'])
    ax.set_ylim(30, 600)
    ax.set_xlabel('Years $t$')
    ax.set_ylabel('Value of 100 (log scale)')
    save(fig, 'ch1_sem_primer_vol_drag', ncol=2)


# (6) width of the 95% confidence interval of the annual mean
def fig_ci_years():
    Y = np.linspace(1, 40, 200)
    fig, ax = plt.subplots(figsize=HALF)
    for s, c in ((0.15, MainBlue), (0.25, IDAred), (0.60, Amber)):
        ax.plot(Y, 100 * 1.96 * s / np.sqrt(Y), color=c, lw=1.3, label=rf'$\sigma_{{\rm year}} = {100 * s:.0f}\%$')
    ax.axhline(8, color=Forest, lw=0.8, ls='--', label='A typical mean return, 8%')
    ax.set_xlabel('Years of data $Y$')
    ax.set_ylabel(r'Half-width $1.96\,\sigma_{\rm year}/\sqrt{Y}$ (%)')
    ax.set_ylim(0, 40)
    save(fig, 'ch1_sem_primer_ci_years', ncol=2)


# (7) the z test and its p-value
def fig_ztest():
    x = np.linspace(-4, 4, 600)
    z = 2.3
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, stats.norm.pdf(x), color=MainBlue, lw=1.3, label='$N(0, 1)$ under $H_0$')
    for side in (-1, 1):
        xx = np.linspace(1.96, 4, 100) * side
        ax.fill_between(xx, stats.norm.pdf(xx), color=IDAred, alpha=0.35, lw=0,
                        label='Rejection region, 5%' if side == 1 else '_')
        xz = np.linspace(z, 4, 100) * side
        ax.fill_between(xz, stats.norm.pdf(xz), color=IDAred, alpha=0.6, lw=0,
                        label=f'p-value = {2 * stats.norm.sf(z):.3f}' if side == 1 else '_')
    ax.axvline(z, color=Navy, lw=0.9, ls='--')
    ax.annotate(f'observed\n$z = {z}$', (z, 0.25), xytext=(3, 0), textcoords='offset points', fontsize=7, color=Navy)
    for v in (-1.96, 1.96):
        ax.annotate(f'{v:+.2f}', (v, 0.0), xytext=(0, -9), textcoords='offset points', ha='center', fontsize=7,
                    color=IDAred, annotation_clip=False)
    ax.set_xticks([-4, 0, 4])
    ax.set_xlabel('$z$')
    ax.set_ylabel('Density')
    ax.set_ylim(0, 0.43)
    save(fig, 'ch1_sem_primer_ztest', ncol=2)


# (8) the bootstrap distribution of a Sharpe ratio
def fig_bootstrap():
    rng = np.random.default_rng(12)
    n, A = 2500, 252
    r = 0.04 + 1.0 * unit_t(4).rvs(n, random_state=rng)
    sr = np.sqrt(A) * r.mean() / r.std(ddof=1)
    B = 2000
    idx = rng.integers(0, n, (B, n))
    bs = np.sqrt(A) * r[idx].mean(axis=1) / r[idx].std(axis=1, ddof=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    srd = sr / np.sqrt(A)
    se = np.sqrt(A) * np.sqrt((1 + srd ** 2 / 2) / n)
    fig, ax = plt.subplots(figsize=HALF)
    ax.hist(bs, bins=40, density=True, color=BandBlue, edgecolor=MainBlue, lw=0.3, label='2000 bootstrap SR')
    x = np.linspace(bs.min(), bs.max(), 300)
    ax.plot(x, stats.norm.pdf(x, sr, se), color=Forest, lw=1.2, label='Normal, SE of Lo (2002)')
    ax.axvline(sr, color=Navy, lw=1.0, label='SR of the sample')
    for v in (lo, hi):
        ax.axvline(v, color=IDAred, lw=1.0, ls='--', label='2.5% and 97.5% percentiles' if v == lo else '_')
    ax.set_xlabel('Annualised Sharpe ratio')
    ax.set_ylabel('Density')
    ax.set_title(f'{n} simulated days', loc='left')
    save(fig, 'ch1_sem_primer_bootstrap', ncol=2)


# (9) downside deviation: only the losses count
def fig_sortino():
    rng = np.random.default_rng(6)
    R = 0.8 + 3.0 * rng.standard_normal(60)
    t = np.arange(1, 61)
    sd = R.std(ddof=1)
    sdD = np.sqrt(np.mean(np.minimum(R, 0) ** 2))
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(t[R >= 0], R[R >= 0], color=Forest, width=0.75, label='Gains: ignored by $\\sigma_D$')
    ax.bar(t[R < 0], R[R < 0], color=IDAred, width=0.75, label='Losses: enter $\\sigma_D$')
    ax.axhline(sd, color=MainBlue, lw=1.0, ls='--', label=f'$\\sigma = {sd:.1f}\\%$')
    ax.axhline(-sd, color=MainBlue, lw=1.0, ls='--')
    ax.axhline(-sdD, color=Amber, lw=1.2, label=f'$-\\sigma_D = -{sdD:.1f}\\%$')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Month $t$')
    ax.set_ylabel('$R_t$ (%)')
    save(fig, 'ch1_sem_primer_sortino', ncol=2)


# (10) drawdown, time under water and recovery
def fig_drawdown():
    rng = np.random.default_rng(51)
    n = 750
    P = 100 * np.exp(np.concatenate([[0.0], np.cumsum(0.0003 + 0.013 * rng.standard_normal(n))]))
    M = np.maximum.accumulate(P)
    DD = 100 * (P / M - 1)
    t = np.arange(n + 1)
    i = int(np.argmin(DD))
    j = int(np.argmax(P[:i + 1]))
    rec = i + int(np.argmax(P[i:] >= P[j])) if (P[i:] >= P[j]).any() else None
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    ax = axes[0]
    ax.plot(t, P, color=MainBlue, lw=0.8, label='Price $P_t$')
    ax.plot(t, M, color=Forest, lw=0.9, ls='--', label='Running peak $M_t$')
    ax.plot(j, P[j], 'o', color=Forest, ms=3.2)
    ax.plot(i, P[i], 'o', color=IDAred, ms=3.2)
    if rec is not None:
        ax.plot(rec, P[rec], 'o', color=Amber, ms=3.2)
    ax.set_ylabel('$P_t$')
    ax = axes[1]
    ax.fill_between(t, DD, 0, color=IDAred, alpha=0.35, lw=0)
    ax.plot(t, DD, color=IDAred, lw=0.7, label='$DD_t$ (%)')
    if rec is not None:
        ax.annotate('', (j, 2), xytext=(rec, 2), arrowprops=dict(arrowstyle='<->', color=Amber, lw=0.8),
                    annotation_clip=False)
        ax.plot([], [], color=Amber, lw=0.8, label='Peak to recovery')
    ax.set_ylabel('$DD_t$ (%)')
    ax.set_xlabel('Trading day $t$')
    save(fig, 'ch1_sem_primer_drawdown', ncol=2)


# (11) the gain needed to recover from a loss
def fig_recovery():
    x = np.linspace(0, 0.8, 300)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(100 * x, 100 * x, color=Forest, lw=1.0, ls='--', label='Gain = loss')
    ax.plot(100 * x, 100 * x / (1 - x), color=IDAred, lw=1.4, label='Gain needed $x/(1-x)$')
    for v in (0.2, 0.5, 0.75):
        g = v / (1 - v)
        ax.plot(100 * v, 100 * g, 'o', color=IDAred, ms=3.2)
        ax.annotate(f'{100 * v:.0f}% $\\to$ {100 * g:.0f}%', (100 * v, 100 * g), xytext=(-5, 3),
                    textcoords='offset points', ha='right', fontsize=7, color=IDAred)
    ax.set_xlabel('Loss from the peak $x$ (%)')
    ax.set_ylabel('Gain to get back (%)')
    ax.set_ylim(0, 420)
    save(fig, 'ch1_sem_primer_recovery', ncol=1)


# (12) correlation of two return series, and ranks
def fig_corr():
    rng = np.random.default_rng(9)
    n = 400
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n)
    x, y = 1.0 * z[:, 0], 1.5 * z[:, 1]
    rho = np.corrcoef(x, y)[0, 1]
    rs = stats.spearmanr(x, y)[0]
    fig, ax = plt.subplots(figsize=HALF)
    ax.scatter(x, y, s=4, color=MainBlue, alpha=0.6, lw=0, label='One day $(R_{1t}, R_{2t})$')
    b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=IDAred, lw=1.1, label='Least-squares line')
    ax.set_title(f'$\\hat\\rho = {rho:.2f}$, Spearman $\\hat\\rho_S = {rs:.2f}$', loc='left')
    ax.set_xlabel('Return of asset 1 (%)')
    ax.set_ylabel('Return of asset 2 (%)')
    save(fig, 'ch1_sem_primer_corr', ncol=1)


if __name__ == '__main__':
    fig_ohlc()
    fig_adjusted()
    fig_log_simple()
    fig_shape()
    fig_vol_drag()
    fig_ci_years()
    fig_ztest()
    fig_bootstrap()
    fig_sortino()
    fig_drawdown()
    fig_recovery()
    fig_corr()
