"""
seminar16_explainers.py -- Explanatory (primer) charts for Seminar 16 (SFM): review for the exam
================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 16, which takes place
BEFORE Lecture 16. All charts use SIMULATED data or exact formulas only (fixed seeds): they illustrate the concepts of
the formula sheet (log and simple returns, the standard error of the Sharpe ratio, Student-t kurtosis and stable
scaling, the ACF and the variance ratio, GARCH and EWMA forecasts, the daily range, the R/S analysis, the Basel
traffic light, the logit curve and the ROC curve) and contain no exercise answers.

Every chart is drawn at the size of its box on the slide (half: one column of a two-column frame; full: the text
width), so that 1 pt in the figure is about 1 pt on the slide and the smallest text is 7 pt.

Output: charts/sfm_ch16_sem_primer_*.pdf and .png (transparent background, legend below the plot, no grey).

Run:  python3 Quantlets/Ch_16/seminar16_explainers.py

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


def garch(n, w=0.02, a=0.08, b=0.90, seed=0, nu=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) if nu is None else stats.t(nu).rvs(n, random_state=rng) * np.sqrt((nu - 2) / nu)
    s2, r = np.empty(n), np.empty(n)
    s2[0] = w / (1 - a - b)
    for t in range(n):
        if t:
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r, s2


# =============================================================================
# 1. log against simple returns; the standard error of the Sharpe ratio
# =============================================================================
def fig_returns_sharpe():
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    R = np.linspace(-0.6, 0.6, 300)
    ax.plot(100 * R, 100 * R, color=Navy, lw=0.9, ls='--', label='$r = R$')
    ax.plot(100 * R, 100 * np.log(1 + R), color=MainBlue, lw=1.5, label='$r = \\ln(1 + R)$')
    ax.set_xlabel('Simple return $R$ (%)')
    ax.set_ylabel('Log return $r$ (%)')
    ax.set_title('Close for small returns, apart for large ones', loc='left')
    ax = axes[1]
    Y = np.linspace(1, 40, 300)
    for sr, c in [(0.3, Forest), (0.6, MainBlue)]:
        se = np.sqrt((1 + sr ** 2 / 2) / Y)
        ax.plot(Y, sr - 1.96 * se, color=c, lw=1.3, label=f'SR = {sr}: lower end SR $- 1.96$ SE')
        ax.axhline(sr, color=c, lw=0.7, ls=':')
    ax.axhline(0, color=IDAred, lw=0.9, label='zero: no excess return')
    ax.set_ylim(-1.5, 0.8)
    ax.set_xlabel('Years of data $Y$')
    ax.set_ylabel('Annual Sharpe ratio')
    ax.set_title('How many years to show SR $> 0$?', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_returns_sharpe', ncol=3, rows=2)


# =============================================================================
# 2. Student-t excess kurtosis; stable scaling of the sum of h days
# =============================================================================
def fig_tails_stable():
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    nu = np.linspace(4.3, 30, 300)
    ax.plot(nu, 6 / (nu - 4), color=MainBlue, lw=1.5, label='excess kurtosis $6/(\\nu - 4)$')
    ax.axhline(0, color=Navy, lw=0.7, ls='--', label='Normal: 0')
    ax.set_ylim(0, 12)
    ax.set_xlabel('Degrees of freedom $\\nu$')
    ax.set_ylabel('Excess kurtosis $K$')
    ax.set_title('Student-$t$: kurtosis infinite for $\\nu \\leq 4$', loc='left')
    ax = axes[1]
    h = np.linspace(1, 60, 300)
    for a, c in [(2.0, MainBlue), (1.8, Forest), (1.6, IDAred)]:
        ax.plot(h, h ** (1 / a), color=c, lw=1.4, label=f'$h^{{1/\\alpha}}$, $\\alpha = {a}$' + (' ($\\sqrt{h}$)' if a == 2 else ''))
    ax.set_xlabel('Days summed $h$')
    ax.set_ylabel('Scale factor')
    ax.set_title('Stable law: scale of a sum of $h$ days', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_tails_stable', ncol=5)


# =============================================================================
# 3. ACF with bands; the variance ratio of three processes
# =============================================================================
def fig_acf_vr(seed=3, T=1500):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    r, _ = garch(T, seed=seed)
    x = r - r.mean()
    lags = np.arange(1, 21)
    acf = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in lags])
    x2 = r ** 2 - np.mean(r ** 2)
    acf2 = np.array([np.sum(x2[k:] * x2[:-k]) / np.sum(x2 * x2) for k in lags])
    ax = axes[0]
    ax.bar(lags - 0.2, acf, width=0.4, color=MainBlue, label='$\\hat\\rho_k$ of $r_t$')
    ax.bar(lags + 0.2, acf2, width=0.4, color=Amber, label='$\\hat\\rho_k$ of $r_t^2$')
    ax.axhline(1.96 / np.sqrt(T), color=IDAred, lw=0.8, ls='--', label='$\\pm 1.96/\\sqrt{T}$')
    ax.axhline(-1.96 / np.sqrt(T), color=IDAred, lw=0.8, ls='--')
    ax.set_xlabel('Lag $k$')
    ax.set_ylabel('ACF')
    ax.set_title(f'Simulated GARCH returns, $T = {T}$', loc='left')
    ax = axes[1]
    q = np.arange(1, 21)
    for rho, c, nm in [(0.0, Navy, 'random walk'), (0.1, Forest, 'momentum, $\\rho_1 = 0.1$'),
                       (-0.1, IDAred, 'mean reversion, $\\rho_1 = -0.1$')]:
        vr = [1 + 2 * sum((1 - k / qq) * rho ** k for k in range(1, qq)) for qq in q]
        ax.plot(q, vr, color=c, lw=1.4, marker='o', ms=2, label=f'VR($q$), {nm}')
    ax.set_xlabel('Horizon $q$ (days)')
    ax.set_ylabel('VR($q$)')
    ax.set_title('AR(1) returns: VR$(q) = 1 + 2\\sum(1 - k/q)\\rho_k$', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_acf_vr', ncol=3, rows=2)


# =============================================================================
# 4. GARCH against EWMA: the forecast after a shock; simulated volatility
# =============================================================================
def fig_garch_ewma(seed=4, w=0.02, a=0.08, b=0.90, lam=0.94):
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    lv = w / (1 - a - b)
    h = np.arange(1, 101)
    s1 = 4.0
    g = lv + (a + b) ** (h - 1) * (s1 - lv)
    hl = np.log(0.5) / np.log(a + b)
    ax.plot(h, g, color=MainBlue, lw=1.5, label='GARCH forecast $E_t\\sigma^2_{t+h}$')
    ax.plot(h, np.full_like(h, s1, float), color=Amber, lw=1.3, ls='--', label='EWMA forecast (flat)')
    ax.axhline(lv, color=Navy, lw=0.8, ls=':', label='long-run variance $\\bar\\sigma^2 = 1$')
    ax.plot([1 + hl], [lv + 0.5 * (s1 - lv)], 'o', color=IDAred, ms=3.5, label=f'half-life {hl:.1f} days')
    ax.set_xlabel('Horizon $h$ (days)')
    ax.set_ylabel('Variance forecast')
    ax.set_title('After a shock: $\\sigma^2_{t+1} = 4$', loc='left')
    ax = axes[1]
    r, s2 = garch(750, w, a, b, seed=seed)
    e = np.empty_like(r)
    e[0] = np.var(r[:50])
    for t in range(1, len(r)):
        e[t] = lam * e[t - 1] + (1 - lam) * r[t - 1] ** 2
    t = np.arange(len(r))
    ax.plot(t, r, color=MainBlue, lw=0.4, alpha=0.6, label='simulated return')
    ax.plot(t, 2 * np.sqrt(s2), color=IDAred, lw=1.0, label='$\\pm 2\\sigma_t$, GARCH')
    ax.plot(t, -2 * np.sqrt(s2), color=IDAred, lw=1.0)
    ax.plot(t, 2 * np.sqrt(e), color=Amber, lw=1.0, label='$\\pm 2\\sigma_t$, EWMA $\\lambda = 0.94$')
    ax.plot(t, -2 * np.sqrt(e), color=Amber, lw=1.0)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('Return (%)')
    ax.set_title('Volatility clustering and two filters', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_garch_ewma', ncol=3, rows=3)


# =============================================================================
# 5. one trading day: open, high, low, close and the range
# =============================================================================
def fig_range(seed=5, m=390, sigma=1.2):
    rng = np.random.default_rng(seed)
    x = np.r_[0, np.cumsum(sigma / np.sqrt(m) * rng.standard_normal(m))]
    P = 100 * np.exp(x / 100)
    t = np.arange(m + 1) / 60 + 9.5
    iH, iL = int(np.argmax(P)), int(np.argmin(P))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(t, P, color=MainBlue, lw=0.8, label='simulated intraday price')
    ax.axhline(P[iH], color=Forest, lw=0.8, ls='--', label=f'high $H$ = {P[iH]:.2f}')
    ax.axhline(P[iL], color=IDAred, lw=0.8, ls='--', label=f'low $L$ = {P[iL]:.2f}')
    ax.plot(t[0], P[0], 'o', color=Amber, ms=4, label=f'open = {P[0]:.2f}')
    ax.plot(t[-1], P[-1], 's', color=Purple, ms=4, label=f'close = {P[-1]:.2f}')
    ax.annotate('', xy=(t[-1] + 0.3, P[iH]), xytext=(t[-1] + 0.3, P[iL]),
                arrowprops=dict(arrowstyle='<->', color=Navy, lw=0.8))
    ax.text(t[-1] + 0.4, (P[iH] + P[iL]) / 2, 'range', fontsize=7, color=Navy, rotation=90, va='center')
    ax.set_xlim(9.3, 16.9)
    ax.set_xlabel('Hour of the trading day')
    ax.set_ylabel('Price')
    finish(fig, 'sfm_ch16_sem_primer_range', ncol=2, rows=3)


# =============================================================================
# 6. R/S analysis: log10 (R/S)_n against log10 n
# =============================================================================
def rs_points(x, sizes):
    out = []
    for n in sizes:
        m = len(x) // n
        X = x[:m * n].reshape(m, n)
        Y = np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)
        R, S = Y.max(axis=1) - Y.min(axis=1), X.std(axis=1)
        out.append(np.mean(R / S))
    return np.array(out)


def fig_hurst(seed=6, T=4000):
    r, _ = garch(T, 0.02, 0.08, 0.90, seed=seed)
    sizes = np.unique(np.round(np.logspace(np.log10(10), np.log10(T / 4), 12)).astype(int))
    fig, ax = plt.subplots(figsize=HALF)
    for x, c, nm in [(r, MainBlue, '$r_t$'), (np.abs(r), IDAred, '$|r_t|$')]:
        y = np.log10(rs_points(x, sizes))
        Hh, c0 = np.polyfit(np.log10(sizes), y, 1)
        ax.plot(np.log10(sizes), y, 'o', color=c, ms=3, label=f'{nm}: slope $H$ = {Hh:.2f}')
        ax.plot(np.log10(sizes), c0 + Hh * np.log10(sizes), color=c, lw=1.0)
    ax.set_xlabel('$\\log_{10} n$ (block size)')
    ax.set_ylabel('$\\log_{10}(R/S)_n$')
    ax.set_title(f'Simulated GARCH returns, $T = {T}$', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_hurst', ncol=1, rows=2)


# =============================================================================
# 7. the Basel traffic light: exceptions in 250 days of a correct VaR 1%
# =============================================================================
def fig_traffic(n=250, a=0.01):
    x = np.arange(0, 13)
    p = stats.binom.pmf(x, n, a)
    cols = [Forest if k <= 4 else (Amber if k <= 9 else IDAred) for k in x]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    z = np.linspace(-5, 4, 600)
    ax.plot(z, stats.norm.pdf(z), color=MainBlue, lw=1.3, label='Normal')
    tt = stats.t(5, scale=np.sqrt(3 / 5))
    ax.plot(z, tt.pdf(z), color=IDAred, lw=1.3, label='Student-$t_5$, variance 1')
    ax.axvline(stats.norm.ppf(a), color=MainBlue, lw=0.9, ls='--', label=f'$z_{{0.01}}$ = {stats.norm.ppf(a):.2f}')
    ax.axvline(tt.ppf(a), color=IDAred, lw=0.9, ls='--', label='Student-$t_5$ 1% quantile')
    ax.set_xlabel('Standardised return')
    ax.set_ylabel('Density')
    ax.set_title('The 1% quantile: Normal against Student-$t$', loc='left')
    ax = axes[1]
    ax.bar(x, 100 * p, color=cols, width=0.7)
    for k, lab, c in [(2, 'green', Forest), (6.5, 'yellow', Amber), (10.8, 'red', IDAred)]:
        ax.text(k, 29, lab, color=c, fontsize=7, ha='center')
    ax.set_ylim(0, 32)
    ax.set_xticks(x)
    ax.set_xlabel(f'Exceptions $x$ in {n} days')
    ax.set_ylabel('$P(x)$ (%)')
    ax.set_title('Basel traffic light: Binomial(250, 0.01)', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_traffic', ncol=4)


# =============================================================================
# 8. the logit curve and the ROC curve
# =============================================================================
def fig_logit_roc(seed=8, n=3000):
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    eta = np.linspace(-6, 6, 300)
    ax.plot(eta, 1 / (1 + np.exp(-eta)), color=MainBlue, lw=1.5, label='PD $= 1/(1 + e^{-\\eta})$')
    ax.axhline(0.5, color=Navy, lw=0.6, ls=':')
    ax.axvline(0, color=Navy, lw=0.6, ls=':')
    ax.set_xlabel('Linear score $\\eta$')
    ax.set_ylabel('Probability of default')
    ax.set_title('Logit: the score becomes a probability', loc='left')
    ax = axes[1]
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.2).astype(int)
    s = 1.1 * y + rng.standard_normal(n)
    o = np.argsort(-s)
    tpr = np.r_[0, np.cumsum(y[o]) / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y[o]) / (1 - y).sum()]
    auc = np.trapezoid(tpr, fpr)
    ax.fill_between(fpr, fpr, tpr, color=BandBlue, alpha=0.9, lw=0, label=f'area above the diagonal = Gini/2 = {(auc - 0.5):.2f}')
    ax.plot(fpr, tpr, color=MainBlue, lw=1.5, label=f'ROC curve, AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], color=Navy, lw=0.8, ls='--', label='no discrimination, AUC = 0.5')
    ax.set_xlabel('False-positive rate (good clients flagged)')
    ax.set_ylabel('True-positive rate')
    ax.set_title(f'Simulated scorecard, Gini = {2 * auc - 1:.2f}', loc='left')
    finish(fig, 'sfm_ch16_sem_primer_logit_roc', ncol=2, rows=2)


if __name__ == '__main__':
    fig_returns_sharpe()
    fig_tails_stable()
    fig_acf_vr()
    fig_garch_ewma()
    fig_range()
    fig_hurst()
    fig_traffic()
    fig_logit_roc()
