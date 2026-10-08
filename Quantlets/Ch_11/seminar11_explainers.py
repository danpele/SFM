"""
seminar11_explainers.py -- explanatory (primer) charts for Seminar 11 (SFM): fractal markets and long memory
============================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 11, which takes place
BEFORE Lecture 11. All charts use SIMULATED data only (fixed seeds), with parameters that differ from those of the
exercises: fractional Brownian motion paths, the scaling of multi-day volatility, the R/S statistic of one block and
its log-log regression, long against short memory in the ACF, DFA, the GPH regression, a Monte Carlo band and
spurious long memory from a mean shift. No exercise answers.

The charts are drawn at the size they have on the slide (included at about scale 1), so every text is >= 6.5 pt.
Output: charts/sfm_ch11_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).

Run:  python3 Quantlets/Ch_11/seminar11_explainers.py

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
import sfm_style as st          # noqa: E402

MainBlue, IDAred, Forest, Amber, Navy = st.MainBlue, st.IDAred, st.Forest, st.Amber, '#1F2A44'
BandBlue = '#C5D2E8'
PREFIX = 'sfm_ch11_sem_primer_'

st.apply()
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2})

HALF = (2.65, 1.85)


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


def fgn_acf(H, k):
    k = np.abs(np.asarray(k, float))
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fgn(n, H, rng):
    """Fractional Gaussian noise of length n (Davies-Harte circulant embedding), variance 1."""
    g = fgn_acf(H, np.arange(n + 1))
    c = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(c).real
    lam[lam < 0] = 0
    m = len(c)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    x = np.fft.fft(np.sqrt(lam / m) * w)
    return x.real[:n]


def arfima_acf(d, kmax):
    r = np.empty(kmax + 1)
    r[0] = 1.0
    for k in range(1, kmax + 1):
        r[k] = r[k - 1] * (k - 1 + d) / (k - d)
    return r


def rs_block(x):
    y = np.cumsum(x - x.mean())
    return (y.max() - y.min()) / x.std()


def rs_curve(x, sizes):
    out = []
    for n in sizes:
        m = len(x) // n
        out.append(np.mean([rs_block(x[i * n:(i + 1) * n]) for i in range(m)]))
    return np.array(out)


def dfa_F(x, sizes):
    y = np.cumsum(x - x.mean())
    F = []
    for n in sizes:
        m = len(y) // n
        t = np.arange(n)
        res = []
        for i in range(m):
            seg = y[i * n:(i + 1) * n]
            b = np.polyfit(t, seg, 1)
            res.append(np.mean((seg - np.polyval(b, t)) ** 2))
        F.append(np.sqrt(np.mean(res)))
    return np.array(F)


# =============================================================================
# 1. fractional Brownian motion paths for three values of H
# =============================================================================
def fig_fbm(seed=11, n=1000):
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.45), sharex=True)
    for ax, (H, c) in zip(axes, [(0.3, IDAred), (0.5, Forest), (0.8, MainBlue)]):
        rng = np.random.default_rng(seed)
        x = np.cumsum(fgn(n, H, rng))
        lab = {0.3: 'anti-persistent', 0.5: 'random walk', 0.8: 'persistent'}[H]
        ax.plot(x, color=c, lw=0.6)
        ax.set_title(f'$H = {H}$: {lab}', loc='left')
        ax.set_xlabel('time $t$ (days)')
    axes[0].set_ylabel('$X_t$')
    fig.tight_layout(pad=0.3)
    save(fig, 'fbm')


# =============================================================================
# 2. scaling of the h-day standard deviation: h^H
# =============================================================================
def fig_scaling():
    h = np.arange(1, 61)
    fig, ax = plt.subplots(figsize=HALF)
    for H, c in [(0.3, IDAred), (0.5, Forest), (0.8, MainBlue)]:
        ax.plot(h, h ** H, color=c, label=f'$h^{{{H}}}$' + (' (square-root rule)' if H == 0.5 else ''))
    ax.set_xlabel('horizon $h$ (days)')
    ax.set_ylabel('$h$-day st. dev. / daily st. dev.')
    ax.set_title('Multi-day volatility grows like $h^H$', loc='left')
    legend_below(fig, ax, ncol=3)
    save(fig, 'scaling')


# =============================================================================
# 3. R/S of one block: cumulative deviations and their range
# =============================================================================
def fig_rs_block(seed=3, n=60):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.05, 1.0, n)
    y = np.cumsum(x - x.mean())
    k = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(k, x - x.mean(), color=BandBlue, width=0.8, label='deviations $x_j - \\bar x$')
    ax.plot(k, y, color=MainBlue, marker='o', ms=1.6, lw=0.9, label='cumulative deviations $Y_k$')
    ax.axhline(y.max(), color=Forest, ls='--', lw=0.7)
    ax.axhline(y.min(), color=IDAred, ls='--', lw=0.7)
    ax.annotate('', (n + 2, y.min()), (n + 2, y.max()), arrowprops=dict(arrowstyle='<->', color=Navy, lw=0.8),
                annotation_clip=False)
    ax.text(n + 3, (y.max() + y.min()) / 2, '$R_n$', fontsize=7.5, va='center')
    ax.set_xlim(0, n + 6)
    ax.set_xlabel('day $k$ in the block')
    ax.set_ylabel('%')
    ax.set_title(f'One block of $n = {n}$ returns: $Y_n = 0$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'rs_block')


# =============================================================================
# 4. log-log R/S regression for an i.i.d. and a persistent series
# =============================================================================
def fig_rs_loglog(seed=8, N=4096):
    sizes = np.unique(np.round(np.logspace(np.log10(10), np.log10(N / 4), 10)).astype(int))
    fig, ax = plt.subplots(figsize=HALF)
    for H, c, lab in [(0.5, Forest, 'i.i.d. Normal'), (0.8, MainBlue, 'fGn, $H = 0.8$')]:
        rng = np.random.default_rng(seed)
        x = fgn(N, H, rng)
        rs = rs_curve(x, sizes)
        b = np.polyfit(np.log10(sizes), np.log10(rs), 1)
        ax.plot(sizes, rs, 'o', color=c, ms=2.5, label=f'{lab}: slope $\\hat H = {b[0]:.2f}$')
        ax.plot(sizes, 10 ** np.polyval(b, np.log10(sizes)), color=c, lw=0.8, label='_')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('block size $n$ (log scale)')
    ax.set_ylabel('$(R/S)_n$ (log scale)')
    ax.set_title(f'R/S regression, $N = {N}$ simulated returns', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'rs_loglog')


# =============================================================================
# 5. ACF: long memory (ARFIMA) against short memory (AR(1)) with the same rho(1)
# =============================================================================
def fig_acf(d=0.25, kmax=100):
    r = arfima_acf(d, kmax)
    phi = r[1]
    k = np.arange(1, kmax + 1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(k, r[1:], color=MainBlue, lw=1.3, label=f'ARFIMA(0, {d}, 0): hyperbolic, $\\propto k^{{2d-1}}$')
    ax.plot(k, phi ** k, color=IDAred, lw=1.3, label=f'AR(1), $\\phi = \\rho(1) = {phi:.3f}$: exponential, $\\phi^k$')
    ax.set_xlabel('lag $k$ (days)')
    ax.set_ylabel('autocorrelation $\\rho(k)$')
    ax.set_title('Same $\\rho(1)$, different memory', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'acf')


# =============================================================================
# 6. DFA: profile, local linear trends, fluctuation
# =============================================================================
def fig_dfa(seed=6, N=400, n=50):
    rng = np.random.default_rng(seed)
    x = fgn(N, 0.6, rng)
    y = np.cumsum(x - x.mean())
    t = np.arange(N)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(t, y, color=MainBlue, lw=0.8, label='profile $Y_k$')
    for i in range(N // n):
        tt = t[i * n:(i + 1) * n]
        b = np.polyfit(tt, y[tt], 1)
        ax.plot(tt, np.polyval(b, tt), color=IDAred, lw=1.0, label='OLS line in each block' if i == 0 else '_')
        ax.axvline(i * n, color=BandBlue, lw=0.6, zorder=0)
    ax.set_xlabel('day $k$')
    ax.set_ylabel('$Y_k$')
    ax.set_title(f'DFA with blocks of $n = {n}$ days', loc='left')
    legend_below(fig, ax, ncol=2)
    save(fig, 'dfa')


# =============================================================================
# 7. GPH: log periodogram against the regressor at low frequencies
# =============================================================================
def fig_gph(seed=7, T=4096, d=0.25):
    rng = np.random.default_rng(seed)
    H = d + 0.5
    x = fgn(T, H, rng)
    j = np.arange(1, T // 2)
    lam = 2 * np.pi * j / T
    I = np.abs(np.fft.fft(x - x.mean())[1:T // 2]) ** 2 / (2 * np.pi * T)
    m = int(np.floor(T ** 0.5))
    reg = -np.log(4 * np.sin(lam / 2) ** 2)
    b = np.polyfit(reg[:m], np.log(I[:m]), 1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(reg[m:m + 600], np.log(I[m:m + 600]), 'o', color=BandBlue, ms=1.2, label='higher frequencies (not used)')
    ax.plot(reg[:m], np.log(I[:m]), 'o', color=MainBlue, ms=2.0, label=f'lowest $m = \\lfloor T^{{0.5}} \\rfloor = {m}$ frequencies')
    xx = np.linspace(reg[m], reg[0], 50)
    ax.plot(xx, np.polyval(b, xx), color=IDAred, lw=1.2, label=f'OLS slope $\\hat d = {b[0]:.2f}$')
    ax.set_xlabel('$-\\ln(4\\sin^2(\\lambda_j/2))$')
    ax.set_ylabel('$\\ln I(\\lambda_j)$')
    ax.set_title(f'GPH on a simulated fGn with $d = {d}$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'gph')


# =============================================================================
# 8. Monte Carlo band of the R/S estimator for i.i.d. series
# =============================================================================
def fig_mc(seed=12, N=500, reps=300):
    rng = np.random.default_rng(seed)
    sizes = np.unique(np.round(np.logspace(1, np.log10(N / 4), 8)).astype(int))
    est = []
    for _ in range(reps):
        x = rng.standard_normal(N)
        est.append(np.polyfit(np.log10(sizes), np.log10(rs_curve(x, sizes)), 1)[0])
    est = np.array(est)
    lo, hi = np.quantile(est, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=HALF)
    ax.hist(est, bins=25, color=MainBlue, alpha=0.85, label=f'$\\hat H$ (R/S) of {reps} i.i.d. Normal series, $N = {N}$')
    ax.axvspan(lo, hi, color=BandBlue, alpha=0.6, zorder=0, label=f'95% band [{lo:.2f}; {hi:.2f}]')
    ax.axvline(0.5, color=IDAred, ls='--', lw=0.9, label='$H = 0.5$')
    ax.axvline(est.mean(), color=Amber, lw=1.1, label=f'mean {est.mean():.2f}: bias above 0.5')
    ax.set_xlabel('$\\hat H$')
    ax.set_ylabel('number of series')
    ax.set_title('Monte Carlo band under $H = 0.5$', loc='left')
    legend_below(fig, ax, ncol=1)
    save(fig, 'mc_band')


# =============================================================================
# 9. spurious long memory: a mean shift
# =============================================================================
def fig_spurious(seed=14, N=2000, lags=100):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) + np.where(np.arange(N) < N // 2, 0.0, 0.6)
    xs = rng.permutation(x)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.55), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.plot(x, color=MainBlue, lw=0.3, label='_')
    ax.plot([0, N // 2, N // 2, N], [0, 0, 0.6, 0.6], color=IDAred, lw=1.2, label='mean (one shift)')
    ax.set_xlabel('day')
    ax.set_title('i.i.d. noise plus one shift of the mean', loc='left')

    def acf(z):
        z = z - z.mean()
        dd = np.sum(z ** 2)
        return np.array([np.sum(z[k:] * z[:-k]) / dd for k in range(1, lags + 1)])
    ax = axes[1]
    k = np.arange(1, lags + 1)
    ax.bar(k, acf(x), color=MainBlue, width=0.8, label='ACF of the series')
    ax.plot(k, acf(xs), 'o', color=Amber, ms=1.6, label='ACF after a random shuffle')
    ax.axhspan(-1.96 / np.sqrt(N), 1.96 / np.sqrt(N), color=BandBlue, alpha=0.8, zorder=0, label='95% band')
    ax.set_xlabel('lag $k$')
    ax.set_title('Slowly decaying ACF without long memory', loc='left')
    legend_below(fig, axes, ncol=4)
    save(fig, 'spurious')


if __name__ == '__main__':
    fig_fbm()
    fig_scaling()
    fig_rs_block()
    fig_rs_loglog()
    fig_acf()
    fig_dfa()
    fig_gph()
    fig_mc()
    fig_spurious()
