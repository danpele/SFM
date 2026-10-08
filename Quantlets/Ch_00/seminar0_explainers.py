"""
seminar0_explainers.py -- Explanatory (primer) charts for Seminar 0 (SFM): prices, returns, volatility, drawdown
==============================================================================================================
Teaching charts for the primer slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 0, which takes
place BEFORE Lecture 0. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (simple and
log returns, compounding, volatility, the square-root-of-time rule, drawdown, a bad tick) and contain no exercise
answers.

The charts are drawn at the size of their box on the slide (half-slide column, about 2.7 x 2.3 inches), so that
1 pt in the figure is about 1 pt on the slide (tick labels 7 pt).

Output: charts/ch0_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom, no grey).

Run:  python3 Quantlets/Ch_00/seminar0_explainers.py

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


# (1) a price path and its daily simple returns
def fig_prices_returns():
    P = gbm(7, 250)
    R = 100 * (P[1:] / P[:-1] - 1)
    t = np.arange(len(P))
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True, gridspec_kw={'height_ratios': [1.1, 1]})
    ax = axes[0]
    ax.plot(t, P, color=MainBlue, lw=1.0, label='Price $P_t$')
    ax.set_ylabel('$P_t$')
    ax.set_title('Simulated daily closes', loc='left')
    ax = axes[1]
    ax.vlines(t[1:], 0, R, color=IDAred, lw=0.6, label='Simple return $R_t$ (%)')
    ax.axhline(0, color=Navy, lw=0.5)
    ax.set_ylabel('$R_t$ (%)')
    ax.set_xlabel('Trading day $t$')
    save(fig, 'ch0_sem_primer_prices_returns', ncol=2)


# (2) log return against simple return
def fig_log_simple():
    R = np.linspace(-0.6, 0.6, 400)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(100 * R, 100 * R, color=Forest, lw=1.0, ls='--', label='$r = R$')
    ax.plot(100 * R, 100 * np.log1p(R), color=MainBlue, lw=1.4, label=r'$r = \ln(1 + R)$')
    for x, off, ha, va in ((-0.5, (6, -2), 'left', 'top'), (0.5, (-6, 4), 'right', 'bottom')):
        y = np.log1p(x)
        ax.plot(100 * x, 100 * y, 'o', color=IDAred, ms=3.5)
        ax.annotate(f'R = {100 * x:+.0f}%\nr = {100 * y:+.1f}%', (100 * x, 100 * y), xytext=off,
                    textcoords='offset points', ha=ha, va=va, fontsize=7, color=IDAred)
    ax.axhline(0, color=Navy, lw=0.4)
    ax.axvline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Simple return $R$ (%)')
    ax.set_ylabel('Log return $r$ (%)')
    ax.set_title('Close for small moves, apart for large ones', loc='left')
    save(fig, 'ch0_sem_primer_log_simple', ncol=2)


# (3) compounding: +10%, -10%, +10%, ...
def fig_compounding():
    n = 20
    R = np.array([0.10 if i % 2 == 0 else -0.10 for i in range(n)])
    V = 100 * np.concatenate([[1.0], np.cumprod(1 + R)])
    S = 100 * (1 + np.concatenate([[0.0], np.cumsum(R)]))
    t = np.arange(n + 1)
    fig, ax = plt.subplots(figsize=(HALF[0], HALF[1] - 0.2))
    ax.plot(t, S, color=Amber, lw=1.2, ls='--', marker='s', ms=2.5, label=r'$100\,(1 + \sum R_t)$: wrong')
    ax.plot(t, V, color=MainBlue, lw=1.2, marker='o', ms=2.5, label=r'$100\,\prod (1 + R_t)$: true value')
    ax.annotate(f'{V[-1]:.1f}', (t[-1], V[-1]), xytext=(-5, 0), textcoords='offset points', ha='right', va='center',
                fontsize=7, color=MainBlue)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('Value of 100 invested')
    ax.set_title('Returns +10%, -10%, +10%, ...', loc='left')
    ax.set_xticks(range(0, n + 1, 4))
    save(fig, 'ch0_sem_primer_compounding', ncol=1)


# (4) volatility: same mean, different spread
def fig_volatility():
    rng = np.random.default_rng(11)
    n = 250
    a = 0.05 + 1.0 * rng.standard_normal(n)
    b = 0.05 + 3.0 * rng.standard_normal(n)
    fig, axes = plt.subplots(2, 1, figsize=HALF, gridspec_kw={'height_ratios': [1, 1]})
    ax = axes[0]
    ax.plot(b, color=IDAred, lw=0.6, label='$s = 3\\%$ a day')
    ax.plot(a, color=MainBlue, lw=0.6, label='$s = 1\\%$ a day')
    ax.set_ylabel('$r_t$ (%)')
    ax.set_xlabel('Day $t$')
    ax = axes[1]
    bins = np.linspace(-10, 10, 41)
    ax.hist(b, bins=bins, density=True, color=IDAred, alpha=0.45, label='_b')
    ax.hist(a, bins=bins, density=True, color=MainBlue, alpha=0.6, label='_a')
    ax.axvline(0.05, color=Navy, lw=0.8, ls=':', label=r'Same mean $\bar r$')
    ax.set_xlabel('Daily return $r_t$ (%)')
    ax.set_ylabel('Density')
    save(fig, 'ch0_sem_primer_volatility', ncol=3)


# (5) the square-root-of-time rule
def fig_sqrt_time():
    rng = np.random.default_rng(3)
    n, k, s = 252, 20, 1.0
    t = np.arange(n + 1)
    X = np.concatenate([np.zeros((k, 1)), np.cumsum(s * rng.standard_normal((k, n)), axis=1)], axis=1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.fill_between(t, -1.96 * s * np.sqrt(t), 1.96 * s * np.sqrt(t), color=BandBlue, alpha=0.9, lw=0,
                    label=r'95% band $\pm 1.96\, s\sqrt{t}$')
    cols = [MainBlue, IDAred, Forest, Amber, Navy]
    for i in range(k):
        ax.plot(t, X[i], color=cols[i % 5], lw=0.55, alpha=0.9, label='Simulated paths' if i == 0 else '_')
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel(r'Cumulative log return $\sum r_t$ (%)')
    ax.set_title(r'$s = 1\%$ a day: the spread grows like $\sqrt{t}$', loc='left')
    ax.set_xticks([0, 63, 126, 189, 252])
    save(fig, 'ch0_sem_primer_sqrt_time', ncol=2)


# (6) drawdown
def fig_drawdown():
    P = gbm(21, 500, mu=0.0004, sigma=0.014)
    M = np.maximum.accumulate(P)
    DD = 100 * (P / M - 1)
    t = np.arange(len(P))
    i = int(np.argmin(DD))
    j = int(np.argmax(P[:i + 1]))
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    ax = axes[0]
    ax.plot(t, P, color=MainBlue, lw=0.9, label='Price $P_t$')
    ax.plot(t, M, color=Forest, lw=0.9, ls='--', label='Running peak $M_t$')
    ax.plot(j, P[j], 'o', color=Forest, ms=3.5)
    ax.plot(i, P[i], 'o', color=IDAred, ms=3.5)
    ax.set_ylabel('$P_t$')
    ax = axes[1]
    ax.fill_between(t, DD, 0, color=IDAred, alpha=0.35, lw=0)
    ax.plot(t, DD, color=IDAred, lw=0.8, label='Drawdown $DD_t$ (%)')
    ax.plot(i, DD[i], 'o', color=IDAred, ms=3.5)
    ax.annotate('MDD', (i, DD[i]), xytext=(6, 2), textcoords='offset points', va='center', fontsize=7, color=IDAred)
    ax.set_ylabel('$DD_t$ (%)')
    ax.set_xlabel('Trading day $t$')
    save(fig, 'ch0_sem_primer_drawdown', ncol=3)


# (7) a bad tick in an exchange rate
def fig_bad_tick():
    rng = np.random.default_rng(5)
    n = 60
    x = 4.97 * np.exp(np.concatenate([[0.0], np.cumsum(0.0012 * rng.standard_normal(n))]))
    bad = x.copy()
    k = 30
    bad[k] = x[k] * 1.025
    r = 100 * np.diff(np.log(bad))
    t = np.arange(n + 1)
    fig, axes = plt.subplots(2, 1, figsize=HALF, sharex=True, gridspec_kw={'height_ratios': [1.1, 1]})
    ax = axes[0]
    ax.plot(t, bad, color=IDAred, lw=0.9, label='Quote with a bad tick')
    ax.plot(t, x, color=MainBlue, lw=0.9, ls='--', label='Second source')
    ax.set_ylabel('RON per EUR')
    ax = axes[1]
    ax.vlines(t[1:], 0, r, color=MainBlue, lw=0.7)
    ax.plot([k, k + 1], [r[k - 1], r[k]], 'o', color=IDAred, ms=3.5)
    ax.annotate(r'$r_{\mathrm{in}}$', (k, r[k - 1]), xytext=(-4, 0), textcoords='offset points', ha='right',
                va='center', fontsize=7, color=IDAred)
    ax.annotate(r'$r_{\mathrm{out}}$', (k + 1, r[k]), xytext=(4, 0), textcoords='offset points', ha='left',
                va='center', fontsize=7, color=IDAred)
    ax.axhline(0, color=Navy, lw=0.4)
    ax.set_ylabel('$r_t$ (%)')
    ax.set_xlabel('Day $t$')
    save(fig, 'ch0_sem_primer_bad_tick', ncol=2)


if __name__ == '__main__':
    fig_prices_returns()
    fig_log_simple()
    fig_compounding()
    fig_volatility()
    fig_sqrt_time()
    fig_drawdown()
    fig_bad_tick()
