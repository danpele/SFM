"""
Generator pentru toate graficele din Capitolul 11: VaR si Expected Shortfall
============================================================================
Toate graficele: transparent background, ENG labels, legend outside bottom.
Statistica Pietelor Financiare - Daniel Traian PELE & Antoaneta Amza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Stil standard SFM: transparent + ENG + legend outside bottom
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['lines.linewidth'] = 1.1
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['legend.fontsize'] = 8

# Culori brand
MainBlue = '#1A3A6E'
IDAred   = '#CD0000'
Forest   = '#2E7D32'
Amber    = '#B5853F'
Orange   = '#E67E22'
Purple   = '#8E44AD'
Crimson  = '#DC3545'
Gray     = '#7F7F7F'
LightGray = '#DADADA'


def save_fig(name):
    """Salveaza figura ca PDF si PNG transparent."""
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=180)
    plt.close()
    print(f"   saved {name}")


def legend_outside_bottom(ax, ncol=2, y=-0.22):
    """Plaseaza legenda in afara graficului, jos-centru."""
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, y),
              ncol=ncol, frameon=False)


# =============================================================================
# Date sintetice reproductibile (Student-t cu volatility clustering)
# =============================================================================
np.random.seed(42)

def simulate_returns(n=2500, mu=0.0003, omega=1e-6, alpha=0.08, beta=0.90, nu=6.0):
    """Simuleaza randamente GARCH(1,1)-t pentru reproductibilitate."""
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps = stats.t.rvs(df=nu, size=n) * np.sqrt((nu - 2) / nu)
    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t-1] - mu) ** 2 + beta * sigma2[t-1]
        r[t] = mu + np.sqrt(sigma2[t]) * eps[t]
    dates = pd.date_range('2014-01-01', periods=n, freq='B')
    return pd.Series(r, index=dates, name='ret'), np.sqrt(sigma2)


returns, true_sigma = simulate_returns(n=2500)
print(f"Simulated {len(returns)} returns; mean={returns.mean():.5f}, std={returns.std():.4f}")


# =============================================================================
# FIG 1: Concept VaR pe densitate (Gaussian + empiric)
# =============================================================================
def fig_var_concept():
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 600)
    pdf = stats.norm.pdf(x, mu, sigma)
    alpha = 0.05
    VaR = -stats.norm.ppf(alpha)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(x, pdf, color=MainBlue, lw=1.4, label='Return density')
    mask = x <= -VaR
    ax.fill_between(x[mask], pdf[mask], color=Crimson, alpha=0.40,
                    label=r'$\alpha=5\%$ tail')
    ax.axvline(-VaR, color=Crimson, lw=1.0, ls='--',
               label=f'$-\\mathrm{{VaR}}_{{5\\%}}={-VaR:.2f}$')
    ax.set_xlabel('Return $r$')
    ax.set_ylabel('$f(r)$')
    ax.set_title('Value-at-Risk as a left-tail quantile', color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.22)
    plt.tight_layout()
    save_fig('ch11_var_concept')


# =============================================================================
# FIG 2: VaR vs Expected Shortfall
# =============================================================================
def fig_var_es():
    x = np.linspace(-4, 4, 600)
    pdf = stats.norm.pdf(x)
    alpha = 0.05
    VaR = -stats.norm.ppf(alpha)
    ES  = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(x, pdf, color=MainBlue, lw=1.4, label='Return density')
    mask = x <= -VaR
    ax.fill_between(x[mask], pdf[mask], color=Crimson, alpha=0.35,
                    label=r'$\alpha=5\%$ tail')
    ax.axvline(-VaR, color=Crimson, lw=1.0, ls='--',
               label=f'$-\\mathrm{{VaR}}_{{5\\%}}={-VaR:.2f}$')
    ax.axvline(-ES, color=Purple, lw=1.2, ls='-',
               label=f'$-\\mathrm{{ES}}_{{5\\%}}={-ES:.2f}$ (tail mean)')
    ax.set_xlabel('Return $r$')
    ax.set_ylabel('$f(r)$')
    ax.set_title('VaR vs Expected Shortfall', color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.22)
    plt.tight_layout()
    save_fig('ch11_var_es_concept')


# =============================================================================
# FIG 3: Subaditivitate - VaR esueaza, ES nu
# =============================================================================
def fig_subadditivity():
    np.random.seed(1)
    n = 100000
    L1 = np.where(np.random.rand(n) < 0.04, 100.0, 0.0)
    L2 = np.where(np.random.rand(n) < 0.04, 100.0, 0.0)
    Lsum = L1 + L2
    alpha = 0.05
    VaR1 = np.quantile(L1, 1 - alpha)
    VaR2 = np.quantile(L2, 1 - alpha)
    VaRs = np.quantile(Lsum, 1 - alpha)
    ES1 = L1[L1 >= VaR1].mean()
    ES2 = L2[L2 >= VaR2].mean()
    ESs = Lsum[Lsum >= VaRs].mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    cats = ['$X_1$', '$X_2$', '$X_1+X_2$', '$X_1$+$X_2$ (sum)']
    var_vals = [VaR1, VaR2, VaRs, VaR1 + VaR2]
    es_vals  = [ES1, ES2, ESs, ES1 + ES2]
    colors = [MainBlue, MainBlue, IDAred, Forest]

    axes[0].bar(cats, var_vals, color=colors, alpha=0.85, edgecolor='black', lw=0.4)
    axes[0].set_title(f'VaR 5%: $X_1+X_2$ = {VaRs:.0f}  vs  sum = {VaR1+VaR2:.0f}',
                      color=MainBlue, fontsize=9)
    axes[0].set_ylabel('VaR')

    axes[1].bar(cats, es_vals, color=colors, alpha=0.85, edgecolor='black', lw=0.4)
    axes[1].set_title(f'ES 5%: $X_1+X_2$ = {ESs:.0f}  vs  sum = {ES1+ES2:.0f}',
                      color=Forest, fontsize=9)
    axes[1].set_ylabel('ES')

    fig.suptitle('VaR may be super-additive; ES is always sub-additive (coherent)',
                 color=Crimson, fontsize=10)
    plt.tight_layout()
    save_fig('ch11_subadditivity')


# =============================================================================
# FIG 4: Comparatie metode de estimare (HS, Normal, t, MC)
# =============================================================================
def fig_methods_comparison():
    r = returns.values
    alpha = 0.05

    var_hs = -np.quantile(r, alpha)
    mu, sd = r.mean(), r.std()
    var_n = -(mu + sd * stats.norm.ppf(alpha))
    nu, loc, scale = stats.t.fit(r)
    var_t = -(loc + scale * stats.t.ppf(alpha, nu))
    rng = np.random.default_rng(7)
    mc_sim = rng.normal(mu, sd, 100000)
    var_mc = -np.quantile(mc_sim, alpha)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.hist(r, bins=80, density=True, color=MainBlue, alpha=0.30, edgecolor='none',
            label='Empirical returns')
    x = np.linspace(r.min(), r.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu, sd), color=Forest, lw=1.1, label='Normal fit')
    ax.plot(x, stats.t.pdf(x, nu, loc, scale), color=Crimson, lw=1.1,
            label=f'Student-$t$ fit ($\\nu={nu:.1f}$)')
    ax.axvline(-var_hs, color=MainBlue, lw=1.1, ls='--',
               label=f'HS VaR={var_hs*100:.2f}%')
    ax.axvline(-var_n,  color=Forest, lw=1.1, ls='--',
               label=f'Normal VaR={var_n*100:.2f}%')
    ax.axvline(-var_t,  color=Crimson, lw=1.1, ls='--',
               label=f'$t$ VaR={var_t*100:.2f}%')
    ax.axvline(-var_mc, color=Purple, lw=1.1, ls='--',
               label=f'MC VaR={var_mc*100:.2f}%')
    ax.set_xlabel('Daily return')
    ax.set_ylabel('Density')
    ax.set_title('Four VaR methods at $\\alpha=5\\%$ on the same returns',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=4, y=-0.30)
    plt.tight_layout()
    save_fig('ch11_methods_comparison')


# =============================================================================
# FIG 5: Rolling VaR/ES + violari pe seria simulata
# =============================================================================
def fig_rolling_var_es():
    r = returns.values
    dates = returns.index
    win = 250
    alpha = 0.05
    var_hs = np.full(len(r), np.nan)
    var_n  = np.full(len(r), np.nan)
    es_hs  = np.full(len(r), np.nan)
    for t in range(win, len(r)):
        window = r[t-win:t]
        q = np.quantile(window, alpha)
        var_hs[t] = -q
        es_hs[t]  = -window[window <= q].mean()
        mu, sd = window.mean(), window.std()
        var_n[t]  = -(mu + sd * stats.norm.ppf(alpha))

    hits_hs = (r < -var_hs)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.plot(dates, r * 100, color=Gray, lw=0.5, alpha=0.8, label='Returns')
    ax.plot(dates, -var_hs * 100, color=MainBlue, lw=1.0,
            label='$-$VaR$_{5\\%}$ HS-250')
    ax.plot(dates, -es_hs * 100,  color=Purple, lw=1.0,
            label='$-$ES$_{5\\%}$ HS-250')
    ax.plot(dates, -var_n * 100,  color=Forest, lw=0.8, ls='--',
            label='$-$VaR$_{5\\%}$ Normal-250')
    ax.scatter(dates[hits_hs], r[hits_hs] * 100, s=8, color=Crimson, zorder=5,
               label=f'HS violations ({hits_hs.sum()})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('Rolling 250-day VaR and ES at 5% with realised violations',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.28)
    plt.tight_layout()
    save_fig('ch11_rolling_var_es')

    return var_hs, es_hs, var_n, hits_hs


# =============================================================================
# FIG 6: GARCH-VaR (conditional) vs HS
# =============================================================================
def fig_garch_var():
    r = returns.values * 100
    dates = returns.index
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t', rescale=False)
    res = am.fit(disp='off', show_warning=False)
    mu_hat   = res.params['mu']
    sigma_t  = res.conditional_volatility
    nu_hat   = res.params['nu']
    q_t = stats.t.ppf(0.05, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
    var_garch = -(mu_hat + sigma_t * q_t)

    win = 250
    var_hs = np.full(len(r), np.nan)
    for t in range(win, len(r)):
        var_hs[t] = -np.quantile(r[t-win:t], 0.05)

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(dates, r, color=Gray, lw=0.5, alpha=0.8, label='Returns (%)')
    ax.plot(dates, -var_garch, color=Crimson, lw=1.1,
            label='$-$VaR$_{5\\%}$ GARCH(1,1)-$t$')
    ax.plot(dates, -var_hs,   color=MainBlue, lw=0.9, ls='--',
            label='$-$VaR$_{5\\%}$ HS-250')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.set_title('Conditional GARCH-VaR adapts quickly; HS lags',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_garch_var')


# =============================================================================
# FIG 7: EVT - POT (Peaks Over Threshold) si fit GPD
# =============================================================================
def fig_evt_var():
    losses = -returns.values
    u = np.quantile(losses, 0.90)
    excess = losses[losses > u] - u
    xi, loc, beta = stats.genpareto.fit(excess, floc=0)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    axes[0].hist(excess, bins=40, density=True, color=MainBlue, alpha=0.35,
                 edgecolor='none', label='Excess losses')
    x = np.linspace(0, excess.max() * 1.05, 300)
    axes[0].plot(x, stats.genpareto.pdf(x, xi, loc, beta),
                 color=Crimson, lw=1.3,
                 label=f'GPD($\\xi={xi:.2f}$, $\\beta={beta:.4f}$)')
    axes[0].set_xlabel('Excess loss above $u$')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'GPD fit to {len(excess)} excesses (u={u*100:.2f}%)',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[0], ncol=2, y=-0.26)

    us = np.linspace(np.quantile(losses, 0.5), np.quantile(losses, 0.98), 50)
    me = [losses[losses > uu].mean() - uu for uu in us]
    axes[1].plot(us * 100, np.array(me) * 100, color=MainBlue, lw=1.1, marker='o', ms=2,
                 label='Mean excess $e(u)$')
    axes[1].axvline(u * 100, color=Crimson, ls='--', lw=0.8,
                    label=f'$u$={u*100:.2f}%')
    axes[1].set_xlabel('Threshold $u$ (%)')
    axes[1].set_ylabel('Mean excess (%)')
    axes[1].set_title('Mean-excess plot: linear $\\Rightarrow$ GPD tail',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=2, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_evt_var')


# =============================================================================
# FIG 8: Distributia testului Kupiec sub H0
# =============================================================================
def fig_kupiec_distribution():
    np.random.seed(0)
    n = 250
    alpha = 0.05
    B = 5000
    lr_sim = np.zeros(B)
    for b in range(B):
        x = np.random.binomial(n, alpha)
        p_hat = x / n
        if 0 < p_hat < 1:
            lr_sim[b] = -2 * (x * np.log(alpha / p_hat) +
                              (n - x) * np.log((1 - alpha) / (1 - p_hat)))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(lr_sim, bins=50, density=True, color=MainBlue, alpha=0.40,
            edgecolor='none',
            label='Simulated LR$_{POF}$ ($n=250$, $B=5000$)')
    x = np.linspace(0.001, 12, 400)
    ax.plot(x, stats.chi2.pdf(x, 1), color=Crimson, lw=1.4,
            label=r'$\chi^2_1$ asymptotic')
    crit = stats.chi2.ppf(0.95, 1)
    ax.axvline(crit, color=Forest, ls='--', lw=0.8,
               label=f'5% crit = {crit:.2f}')
    ax.set_xlim(0, 8)
    ax.set_xlabel('LR$_{POF}$')
    ax.set_ylabel('Density')
    ax.set_title('Kupiec POF test: distribution under $H_0$',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_kupiec_distribution')


# =============================================================================
# FIG 9: Christoffersen - clustering vs no-clustering
# =============================================================================
def fig_christoffersen_clustering():
    np.random.seed(11)
    n = 250
    hits_a = (np.random.rand(n) < 0.05).astype(int)
    hits_b = np.zeros(n, dtype=int)
    pi01, pi11 = 0.02, 0.40
    for t in range(1, n):
        p = pi11 if hits_b[t-1] else pi01
        hits_b[t] = int(np.random.rand() < p)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 2.9), sharex=True)
    for ax, h, title, c in [
        (axes[0], hits_a, 'Independent violations (good model)', Forest),
        (axes[1], hits_b, 'Clustered violations (bad model)',     Crimson)]:
        idx = np.where(h == 1)[0]
        ax.vlines(idx, 0, 1, color=c, lw=1.2)
        ax.set_yticks([])
        ax.set_title(f'{title}  ($n={n}$, hits={h.sum()})',
                     color=c, fontsize=9, loc='left')
        ax.set_xlim(0, n)
    axes[1].set_xlabel('Day $t$')
    plt.tight_layout()
    save_fig('ch11_christoffersen_clustering')


# =============================================================================
# FIG 10: Basel Traffic Light
# =============================================================================
def fig_traffic_light():
    n = 250
    alpha = 0.01
    k_vals = np.arange(0, 16)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    colors = []
    for k in k_vals:
        if k <= 4:
            colors.append(Forest)
        elif k <= 9:
            colors.append(Amber)
        else:
            colors.append(Crimson)
    ax.bar(k_vals, stats.binom.pmf(k_vals, n, alpha), color=colors, alpha=0.85,
           edgecolor='black', lw=0.3)
    ax.set_xlabel('Number of violations $K$ in 250 days (VaR 99%)')
    ax.set_ylabel('$P(K=k\\mid \\alpha=0{.}01)$')
    ax.set_title('Basel Traffic Light: green / yellow / red zones',
                 color=MainBlue, fontsize=10)
    legend_patches = [
        mpatches.Patch(color=Forest,  label='Green ($K\\leq 4$)'),
        mpatches.Patch(color=Amber,   label='Yellow ($5\\leq K\\leq 9$)'),
        mpatches.Patch(color=Crimson, label='Red ($K\\geq 10$)'),
    ]
    ax.legend(handles=legend_patches, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    save_fig('ch11_traffic_light')


# =============================================================================
# FIG 11: Acerbi-Szekely Z2 statistic
# =============================================================================
def fig_acerbi_szekely():
    np.random.seed(3)
    B = 4000
    n = 250
    alpha = 0.05
    z2_h0 = np.zeros(B)
    for b in range(B):
        r = np.random.normal(0, 1, n)
        VaR = -stats.norm.ppf(alpha)
        ES  = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
        It = (r < -VaR).astype(int)
        if It.sum() > 0:
            z2_h0[b] = (r * It).sum() / (n * alpha * (-ES)) + 1
        else:
            z2_h0[b] = 0

    z2_h1 = np.zeros(B)
    for b in range(B):
        r = stats.t.rvs(4, size=n) * np.sqrt(2/4)
        VaR = -stats.norm.ppf(alpha)
        ES  = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
        It = (r < -VaR).astype(int)
        if It.sum() > 0:
            z2_h1[b] = (r * It).sum() / (n * alpha * (-ES)) + 1
        else:
            z2_h1[b] = 0

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(z2_h0, bins=50, density=True, alpha=0.45, color=Forest,
            label=r'$H_0$: model correct', edgecolor='none')
    ax.hist(z2_h1, bins=50, density=True, alpha=0.45, color=Crimson,
            label=r'$H_1$: tails underestimated', edgecolor='none')
    ax.axvline(0, color='black', lw=0.5, ls=':')
    ax.set_xlabel('$Z_2$ (Acerbi-Szekely)')
    ax.set_ylabel('Density')
    ax.set_title('ES backtest: $Z_2$ under correct vs misspecified model',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_acerbi_szekely')


# =============================================================================
# FIG 12: Studiu de caz BVB - rolling VaR/ES
# =============================================================================
def fig_bvb_case():
    np.random.seed(2024)
    r, _ = simulate_returns(n=1500, omega=2e-6, alpha=0.10, beta=0.87, nu=5.0)
    crash = pd.Timestamp(r.index[750])
    r.loc[crash] = -0.08
    r.iloc[751:756] += np.array([-0.04, 0.05, -0.03, 0.02, -0.02])

    win = 250
    alpha = 0.01
    var_hs = np.full(len(r), np.nan)
    es_hs  = np.full(len(r), np.nan)
    for t in range(win, len(r)):
        w = r.iloc[t-win:t].values
        q = np.quantile(w, alpha)
        var_hs[t] = -q
        es_hs[t]  = -w[w <= q].mean()

    hits = (r.values < -var_hs)
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(r.index, r.values * 100, color=Gray, lw=0.5, alpha=0.8,
            label='BET-like returns')
    ax.plot(r.index, -var_hs * 100, color=MainBlue, lw=1.0,
            label='$-$VaR$_{1\\%}$ HS-250')
    ax.plot(r.index, -es_hs  * 100, color=Purple, lw=1.0,
            label='$-$ES$_{1\\%}$ HS-250')
    ax.scatter(r.index[hits], r.values[hits] * 100, s=12, color=Crimson, zorder=5,
               label=f'VaR violations ({hits.sum()})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('BVB case: rolling 250-day VaR/ES at 1% with crisis jump',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=4, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_bvb_case')


# =============================================================================
# FIG 13: Lehman 2008 - cum a esuat VaR-ul gaussian
# =============================================================================
def fig_lehman_2008():
    np.random.seed(2008)
    n_pre = 500
    n_crisis = 60
    r_pre = np.random.normal(0.0004, 0.008, n_pre)
    r_crisis = np.random.normal(-0.002, 0.035, n_crisis)
    r_crisis[10] = -0.09
    r_crisis[15] = -0.07
    r_crisis[20] = -0.08
    r_all = np.concatenate([r_pre, r_crisis])
    dates = pd.date_range('2007-01-01', periods=len(r_all), freq='B')

    win = 250
    alpha = 0.01
    var_n = np.full(len(r_all), np.nan)
    var_hs = np.full(len(r_all), np.nan)
    for t in range(win, len(r_all)):
        w = r_all[t-win:t]
        mu, sd = w.mean(), w.std()
        var_n[t]  = -(mu + sd * stats.norm.ppf(alpha))
        var_hs[t] = -np.quantile(w, alpha)

    hits_n  = (r_all < -var_n)
    hits_hs = (r_all < -var_hs)

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(dates, r_all * 100, color=Gray, lw=0.6, label='Returns')
    ax.plot(dates, -var_n * 100,  color=Forest, lw=1.0, label='$-$VaR Normal-250')
    ax.plot(dates, -var_hs * 100, color=MainBlue, lw=1.0, label='$-$VaR HS-250')
    ax.scatter(dates[hits_n],  r_all[hits_n]  * 100, s=14, color=Crimson, marker='o',
               label=f'Normal violations ({hits_n.sum()})')
    ax.scatter(dates[hits_hs], r_all[hits_hs] * 100, s=20, color=Orange, marker='x',
               label=f'HS violations ({hits_hs.sum()})')
    ax.axvspan(dates[n_pre], dates[-1], color=Crimson, alpha=0.08, label='2008 crisis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('Lehman-style crisis: Normal-VaR fails, HS-VaR also struggles',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.30)
    plt.tight_layout()
    save_fig('ch11_lehman_2008')


# =============================================================================
# FIG 14: scaling sqrt(T) pe orizonturi diverse
# =============================================================================
def fig_var_horizons():
    horizons = np.arange(1, 21)
    sigma_1d = 0.015
    var_1d = -stats.norm.ppf(0.01) * sigma_1d
    var_sqrt = var_1d * np.sqrt(horizons)
    var_t_emp = []
    np.random.seed(7)
    for h in horizons:
        sims = stats.t.rvs(5, size=(50000, h)) * sigma_1d * np.sqrt(3/5)
        cum = sims.sum(axis=1)
        var_t_emp.append(-np.quantile(cum, 0.01))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(horizons, var_sqrt * 100, color=Forest, lw=1.3, marker='o', ms=3,
            label='$\\sqrt{T}$ rule (Normal)')
    ax.plot(horizons, np.array(var_t_emp) * 100, color=Crimson, lw=1.3, marker='s', ms=3,
            label='Empirical (Student-$t$, $\\nu=5$)')
    ax.set_xlabel('Horizon $T$ (days)')
    ax.set_ylabel('VaR 1% (%)')
    ax.set_title('VaR scaling with horizon: $\\sqrt{T}$ vs heavy-tail empirical',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.22)
    plt.tight_layout()
    save_fig('ch11_var_horizons')


# =============================================================================
# FIG 15: histograma randamente sintetice + statistici descriptive
# =============================================================================
def fig_returns_overview():
    r = returns.values
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    axes[0].plot(returns.index, r * 100, color=MainBlue, lw=0.4)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Simulated GARCH-$t$ returns', color=MainBlue, fontsize=9)

    axes[1].hist(r * 100, bins=80, density=True, color=MainBlue, alpha=0.40,
                 edgecolor='none', label='Empirical')
    x = np.linspace(min(r), max(r), 400)
    axes[1].plot(x * 100, stats.norm.pdf(x, r.mean(), r.std()),
                 color=Forest, lw=1.0, label='Normal fit')
    axes[1].set_xlabel('Daily return (%)')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Histogram + Normal fit (kurt={stats.kurtosis(r):.2f})',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=2, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_returns_overview')


# =============================================================================
# FIG 16 (NEW): HS method visual - histograma + cuantila empirica
# =============================================================================
def fig_method_hs():
    """Historical Simulation: ilustrare cu fereastra 250 zile."""
    np.random.seed(50)
    r_win = returns.values[-250:]
    alpha = 0.05
    q = np.quantile(r_win, alpha)
    VaR = -q
    ES = -r_win[r_win <= q].mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Panel 1: sorted returns with quantile
    sorted_r = np.sort(r_win)
    pct = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    axes[0].plot(pct * 100, sorted_r * 100, color=MainBlue, lw=1.0,
                 label='Sorted returns')
    axes[0].axhline(q * 100, color=Crimson, ls='--', lw=0.9,
                    label=f'5% quantile = {q*100:.2f}%')
    axes[0].axvline(5, color=Gray, ls=':', lw=0.6)
    axes[0].fill_between(pct[pct <= 5] * 100, sorted_r[pct <= 5] * 100,
                         color=Crimson, alpha=0.25, label='Tail (5%)')
    axes[0].set_xlabel('Percentile (%)')
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Empirical CDF, 250-day window', color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[0], ncol=3, y=-0.26)

    # Panel 2: histogram with VaR cut
    axes[1].hist(r_win * 100, bins=30, color=MainBlue, alpha=0.40, edgecolor='none',
                 label='Returns histogram')
    axes[1].axvline(q * 100, color=Crimson, ls='--', lw=1.0,
                    label=f'$-$VaR = {-VaR*100:.2f}%')
    axes[1].axvline(-ES * 100, color=Purple, ls='-', lw=1.0,
                    label=f'$-$ES = {-ES*100:.2f}%')
    axes[1].set_xlabel('Daily return (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram + HS-VaR and HS-ES', color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=3, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_method_hs')


# =============================================================================
# FIG 17 (NEW): Normal method - density with VaR marked + scaling
# =============================================================================
def fig_method_normal():
    """Parametric Normal: vizualizare densitate + VaR la 3 niveluri."""
    r = returns.values[-250:]
    mu, sd = r.mean(), r.std()
    levels = [0.05, 0.01, 0.001]
    z = [stats.norm.ppf(a) for a in levels]
    vars_ = [-(mu + sd * zi) for zi in z]
    es_ = [-mu + sd * stats.norm.pdf(zi) / a for a, zi in zip(levels, z)]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Panel 1: density with three VaR levels
    x = np.linspace(r.min() * 1.5, r.max() * 1.5, 500)
    pdf = stats.norm.pdf(x, mu, sd)
    axes[0].plot(x * 100, pdf, color=MainBlue, lw=1.3, label='Normal fit')
    axes[0].hist(r * 100, bins=30, density=True, color=MainBlue, alpha=0.25,
                 edgecolor='none', label='Empirical')
    cols = [Crimson, Forest, Purple]
    labels = ['VaR 5%', 'VaR 1%', 'VaR 0.1%']
    for v, c, lab in zip(vars_, cols, labels):
        axes[0].axvline(-v * 100, color=c, ls='--', lw=0.9, label=f'{lab}={v*100:.2f}%')
    axes[0].set_xlabel('Daily return (%)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Normal-fitted density and VaR levels', color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[0], ncol=3, y=-0.28)

    # Panel 2: VaR vs alpha curve
    alphas = np.linspace(0.001, 0.10, 100)
    var_curve = -(mu + sd * stats.norm.ppf(alphas))
    es_curve = -mu + sd * stats.norm.pdf(stats.norm.ppf(alphas)) / alphas
    axes[1].plot(alphas * 100, var_curve * 100, color=MainBlue, lw=1.2, label='VaR')
    axes[1].plot(alphas * 100, es_curve * 100,  color=Purple, lw=1.2, label='ES')
    for a, v, c in zip(levels, vars_, cols):
        axes[1].scatter([a * 100], [v * 100], color=c, s=25, zorder=5)
    axes[1].set_xlabel('Confidence level $\\alpha$ (%)')
    axes[1].set_ylabel('Risk measure (%)')
    axes[1].set_title('VaR and ES as functions of $\\alpha$ (Normal)',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_method_normal')


# =============================================================================
# FIG 18 (NEW): Student-t method - comparison with Normal
# =============================================================================
def fig_method_student_t():
    """Parametric Student-t: comparatie cu Normal pe coada."""
    r = returns.values[-1000:]
    mu, sd = r.mean(), r.std()
    nu, loc, scale = stats.t.fit(r)

    alphas = np.linspace(0.001, 0.10, 100)
    var_n = -(mu + sd * stats.norm.ppf(alphas))
    var_t = -(loc + scale * stats.t.ppf(alphas, nu))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Panel 1: zoom on left tail of density
    x = np.linspace(-4 * sd, 0, 400)
    axes[0].plot(x * 100, stats.norm.pdf(x, mu, sd), color=Forest, lw=1.3,
                 label='Normal')
    axes[0].plot(x * 100, stats.t.pdf(x, nu, loc, scale), color=Crimson, lw=1.3,
                 label=f'Student-$t$ ($\\nu={nu:.1f}$)')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Daily return (%)')
    axes[0].set_ylabel('Density (log scale)')
    axes[0].set_title('Left tail: Normal vs Student-$t$', color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[0], ncol=2, y=-0.24)

    # Panel 2: ratio VaR_t / VaR_N across alphas
    ratio = var_t / var_n
    axes[1].plot(alphas * 100, ratio, color=Crimson, lw=1.3,
                 label='$\\mathrm{VaR}^{t}/\\mathrm{VaR}^{N}$')
    axes[1].axhline(1, color=Forest, ls='--', lw=0.7, label='Equal')
    axes[1].set_xlabel('Confidence level $\\alpha$ (%)')
    axes[1].set_ylabel('Ratio')
    axes[1].set_title(f'Student-$t$ VaR vs Normal VaR ($\\nu={nu:.1f}$)',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_method_student_t')


# =============================================================================
# FIG 19 (NEW): Monte Carlo method - simulated paths + histogram
# =============================================================================
def fig_method_mc():
    """Monte Carlo: traiectorii simulate + histograma P&L."""
    np.random.seed(17)
    n_paths = 200
    h = 10
    sigma_1d = 0.015
    mu_1d = 0.0003
    paths = np.zeros((n_paths, h + 1))
    paths[:, 0] = 100
    for t in range(1, h + 1):
        paths[:, t] = paths[:, t-1] * np.exp(
            np.random.normal(mu_1d - 0.5 * sigma_1d**2, sigma_1d, n_paths))

    n_big = 50000
    rng = np.random.default_rng(2)
    sims = rng.normal(mu_1d - 0.5 * sigma_1d**2, sigma_1d, (n_big, h)).sum(axis=1)
    pnl = 100 * (np.exp(sims) - 1)
    alpha = 0.05
    var_mc = -np.quantile(pnl, alpha)
    es_mc  = -pnl[pnl <= np.quantile(pnl, alpha)].mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Panel 1: simulated paths
    for i in range(n_paths):
        axes[0].plot(range(h + 1), paths[i], color=MainBlue, lw=0.3, alpha=0.4)
    axes[0].axhline(100, color='black', lw=0.5, ls=':', label='Initial')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Portfolio value')
    axes[0].set_title(f'{n_paths} Monte Carlo paths (10-day horizon)',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[0], ncol=1, y=-0.22)

    # Panel 2: P&L histogram with VaR/ES
    axes[1].hist(pnl, bins=80, color=MainBlue, alpha=0.40, edgecolor='none',
                 label=f'P&L ({n_big:,} sims)')
    axes[1].axvline(-var_mc, color=Crimson, ls='--', lw=1.0,
                    label=f'$-$VaR={var_mc:.2f}')
    axes[1].axvline(-es_mc, color=Purple, ls='-', lw=1.0,
                    label=f'$-$ES={es_mc:.2f}')
    axes[1].set_xlabel('10-day P&L (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('P&L histogram with MC VaR and ES (5%)',
                      color=MainBlue, fontsize=9)
    legend_outside_bottom(axes[1], ncol=3, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_method_mc')


# =============================================================================
# FIG 20: Filtered Historical Simulation (Hull-White 1998)
# =============================================================================
def fig_filtered_hs():
    """FHS: standardize via GARCH, then resample standardized residuals."""
    r = returns.values * 100
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal',
                    rescale=False)
    res = am.fit(disp='off', show_warning=False)
    sigma_t = res.conditional_volatility
    mu_hat  = res.params['mu']
    z = (r - mu_hat) / sigma_t

    win = 250
    alpha = 0.05
    var_hs = np.full(len(r), np.nan)
    var_fhs = np.full(len(r), np.nan)
    for t in range(win, len(r)):
        var_hs[t] = -np.quantile(r[t-win:t], alpha)
        zq = np.quantile(z[t-win:t], alpha)
        var_fhs[t] = -(mu_hat + sigma_t[t] * zq)

    dates = returns.index
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(dates, r, color=Gray, lw=0.5, alpha=0.7, label='Returns (%)')
    ax.plot(dates, -var_hs,  color=MainBlue, lw=0.9, ls='--',
            label='$-$VaR HS-250')
    ax.plot(dates, -var_fhs, color=Crimson, lw=1.1,
            label='$-$VaR Filtered HS')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('Filtered Historical Simulation: HS on $z_t = (r_t-\\mu)/\\sigma_t$',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_filtered_hs')


# =============================================================================
# FIG 21: Cornish-Fisher expansion
# =============================================================================
def fig_cornish_fisher():
    r = returns.values
    mu, sd = r.mean(), r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r)
    alphas = np.linspace(0.001, 0.20, 200)
    var_n  = []
    var_cf = []
    var_emp = []
    for a in alphas:
        z = stats.norm.ppf(a)
        zcf = (z + (z**2-1)*skew/6 + (z**3-3*z)*kurt/24
               - (2*z**3-5*z)*skew**2/36)
        var_n.append(-(mu + sd*z))
        var_cf.append(-(mu + sd*zcf))
        var_emp.append(-np.quantile(r, a))

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(alphas*100, np.array(var_n)*100,   color=Forest,  lw=1.2,
            label='Normal')
    ax.plot(alphas*100, np.array(var_cf)*100,  color=Crimson, lw=1.2,
            label='Cornish-Fisher')
    ax.plot(alphas*100, np.array(var_emp)*100, color=MainBlue, lw=1.0, ls=':',
            label='Empirical')
    ax.set_xlabel('$\\alpha$ (%)')
    ax.set_ylabel('VaR (%)')
    ax.set_title(f'Cornish-Fisher correction (skew={skew:.2f}, exc-kurt={kurt:.2f})',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_cornish_fisher')


# =============================================================================
# FIG 22: Hill plot for tail index
# =============================================================================
def fig_hill_plot():
    losses = -np.sort(-returns.values)
    n = len(losses)
    ks = np.arange(5, min(400, n//4))
    hill = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(ks):
        hill[i] = np.mean(np.log(losses[:k] / losses[k]))

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(ks, 1/hill, color=MainBlue, lw=1.0, marker='o', ms=2,
            label=r'Hill estimator $\hat{\alpha}_H(k) = 1/\hat{\xi}_k$')
    ax.axhline(4, color=Forest, ls='--', lw=0.7,
               label='Reference: $\\alpha=4$ (typical equity)')
    ax.set_xlabel('Order statistic $k$')
    ax.set_ylabel(r'Hill tail index $\hat{\alpha}_H$')
    ax.set_title('Hill plot: stable region $\\Rightarrow$ tail index estimate',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_hill_plot')


# =============================================================================
# FIG 23: GJR-GARCH leverage effect
# =============================================================================
def fig_gjr_garch():
    r = returns.values * 100
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, o=1, dist='t',
                    rescale=False)
    res = am.fit(disp='off', show_warning=False)
    sigma_gjr = res.conditional_volatility
    am2 = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t',
                     rescale=False)
    res2 = am2.fit(disp='off', show_warning=False)
    sigma_gar = res2.conditional_volatility

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    dates = returns.index
    ax.plot(dates, sigma_gar, color=MainBlue, lw=0.9, label='GARCH(1,1)')
    ax.plot(dates, sigma_gjr, color=Crimson, lw=0.9, label='GJR-GARCH(1,1,1)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Conditional $\\sigma_t$ (%)')
    ax.set_title('Leverage effect: GJR responds more to negative shocks',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_gjr_garch')


# =============================================================================
# FIG 24: Block Maxima method (BM-GEV)
# =============================================================================
def fig_block_maxima():
    losses = -returns.values
    n = len(losses)
    block_size = 20  # ~1 month
    n_blocks = n // block_size
    blocks = losses[:n_blocks*block_size].reshape(n_blocks, block_size)
    maxima = blocks.max(axis=1)
    shape, loc, scale = stats.genextreme.fit(maxima)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.hist(maxima*100, bins=30, density=True, color=MainBlue, alpha=0.40,
            edgecolor='none', label=f'Block maxima ({n_blocks} blocks)')
    x = np.linspace(maxima.min(), maxima.max()*1.1, 400)
    ax.plot(x*100, stats.genextreme.pdf(x, shape, loc, scale),
            color=Crimson, lw=1.3,
            label=f'GEV($\\xi={-shape:.2f}$, $\\mu={loc*100:.2f}\\%$)')
    ax.set_xlabel('Block maximum loss (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'Block Maxima: GEV fit on {block_size}-day blocks',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_block_maxima')


# =============================================================================
# FIG 25: ES/VaR ratio for diverse distributions
# =============================================================================
def fig_es_var_ratio():
    alphas = np.array([0.05, 0.025, 0.01, 0.005, 0.001])
    rows = []
    for a in alphas:
        z = stats.norm.ppf(a)
        var_n = -z
        es_n = stats.norm.pdf(z) / a
        rows.append([a, var_n, es_n, es_n/var_n])
    arr = np.array(rows)

    rows_t = []
    for nu in [3, 5, 8]:
        rr = []
        for a in alphas:
            tq = stats.t.ppf(a, nu) * np.sqrt((nu-2)/nu)
            var_t = -tq
            es_t = (stats.t.pdf(stats.t.ppf(a, nu), nu) / a *
                    (nu + stats.t.ppf(a, nu)**2) / (nu - 1)) * np.sqrt((nu-2)/nu)
            rr.append(es_t/var_t)
        rows_t.append(rr)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(alphas*100, arr[:, 3], color=Forest, lw=1.2, marker='o', ms=3,
            label='Normal')
    for nu, rr, c in zip([3, 5, 8], rows_t, [Crimson, Purple, Orange]):
        ax.plot(alphas*100, rr, lw=1.2, marker='s', ms=3, color=c,
                label=f'Student-$t$ ($\\nu={nu}$)')
    ax.set_xscale('log')
    ax.set_xlabel('$\\alpha$ (%)  -- log scale')
    ax.set_ylabel('ES / VaR ratio')
    ax.set_title('ES/VaR ratio increases with $\\alpha$ smaller and tails heavier',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=4, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_es_var_ratio')


# =============================================================================
# FIG 26: Lopez magnitude loss function
# =============================================================================
def fig_lopez_loss():
    np.random.seed(99)
    n = 250
    alpha = 0.01
    sigma = 0.015
    # Generate returns and a 'correct' VaR vs a 'wrong' VaR
    r = np.random.normal(0, sigma, n)
    var_correct = -stats.norm.ppf(alpha) * sigma * np.ones(n)
    var_wrong = -stats.norm.ppf(alpha) * sigma * 0.7 * np.ones(n)

    def lopez(r, var):
        I = r < -var
        return I + (r + var)**2 * I  # 1 + magnitude squared

    L_c = lopez(r, var_correct).sum()
    L_w = lopez(r, var_wrong).sum()

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(np.arange(n), r*100, color=Gray, lw=0.5, label='Returns (%)')
    ax.plot(np.arange(n), -var_correct*100, color=MainBlue, lw=1.0,
            label=f'Correct VaR (Lopez score = {L_c:.2f})')
    ax.plot(np.arange(n), -var_wrong*100, color=Crimson, lw=1.0, ls='--',
            label=f'Wrong VaR (Lopez = {L_w:.2f})')
    hits_w = r < -var_wrong
    ax.scatter(np.where(hits_w)[0], r[hits_w]*100, s=18, color=Crimson, zorder=5)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('Lopez loss function: penalises magnitude beyond VaR',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.24)
    plt.tight_layout()
    save_fig('ch11_lopez_loss')


# =============================================================================
# FIG 27: Crypto/FTX 2022 stress
# =============================================================================
def fig_crypto_var():
    np.random.seed(2022)
    n = 800
    r = np.random.normal(0.0005, 0.030, n)  # high vol
    # FTX collapse Nov 2022 - days 600-620
    r[600:610] = np.random.normal(-0.05, 0.04, 10)
    r[604] = -0.20  # one massive day
    r[608] = -0.15
    r[700:710] = np.random.normal(-0.03, 0.05, 10)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    win = 250
    alpha = 0.01
    var_hs = np.full(n, np.nan)
    for t in range(win, n):
        var_hs[t] = -np.quantile(r[t-win:t], alpha)
    hits = r < -var_hs

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.plot(dates, r*100, color=Gray, lw=0.5, alpha=0.7, label='BTC-like returns')
    ax.plot(dates, -var_hs*100, color=MainBlue, lw=1.0, label='$-$VaR$_{1\\%}$ HS-250')
    ax.scatter(dates[hits], r[hits]*100, s=12, color=Crimson, zorder=5,
               label=f'Violations ({hits.sum()})')
    ax.axvspan(dates[600], dates[625], color=Crimson, alpha=0.10,
               label='FTX collapse (Nov 2022)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return / VaR (%)')
    ax.set_title('Crypto VaR stress: FTX 2022 collapse with $-20\\%$ jump',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=4, y=-0.28)
    plt.tight_layout()
    save_fig('ch11_crypto_var')


# =============================================================================
# FIG 28: Cross-asset VaR comparison
# =============================================================================
def fig_cross_asset():
    assets = ['EUR/USD', 'BET', 'S&P 500', 'Gold', 'WTI Oil', 'BTC', 'IG Bonds']
    vol_1d = np.array([0.5, 1.5, 1.0, 1.0, 2.5, 4.0, 0.3])  # in %
    nu = np.array([10, 5, 6, 7, 4, 3.5, 12])
    alpha = 0.01
    var_n = -stats.norm.ppf(alpha) * vol_1d
    var_t = []
    for v, n in zip(vol_1d, nu):
        tq = stats.t.ppf(alpha, n) * np.sqrt((n-2)/n)
        var_t.append(-v * tq)
    var_t = np.array(var_t)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(assets))
    w = 0.35
    ax.bar(x-w/2, var_n, w, color=Forest,  label='Normal', edgecolor='black', lw=0.3)
    ax.bar(x+w/2, var_t, w, color=Crimson, label='Student-$t$', edgecolor='black', lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=20, ha='right')
    ax.set_ylabel('VaR 1% (%)')
    ax.set_title('Cross-asset 1-day VaR 1\\%: Normal vs Student-$t$',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=2, y=-0.32)
    plt.tight_layout()
    save_fig('ch11_cross_asset')


# =============================================================================
# FIG 29: Liquidity-adjusted VaR (LVaR)
# =============================================================================
def fig_lvar():
    horizons = np.arange(1, 31)
    sigma = 0.015
    var_base = -stats.norm.ppf(0.01) * sigma
    var_sqrt = var_base * np.sqrt(horizons) * 100
    # Liquidity cost: half spread, increases with size
    spread = 0.001  # 10bps
    var_lvar = (var_base * np.sqrt(horizons) + 0.5 * spread * (1 + 0.05*horizons)) * 100

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(horizons, var_sqrt, color=Forest, lw=1.3, marker='o', ms=3,
            label='Standard VaR ($\\sqrt{T}$)')
    ax.plot(horizons, var_lvar, color=Crimson, lw=1.3, marker='s', ms=3,
            label='Liquidity-adjusted VaR')
    ax.fill_between(horizons, var_sqrt, var_lvar, color=Crimson, alpha=0.15,
                    label='Liquidity premium')
    ax.set_xlabel('Liquidation horizon $T$ (days)')
    ax.set_ylabel('VaR (%)')
    ax.set_title('LVaR = VaR + liquidity cost; widens at longer horizons',
                 color=MainBlue, fontsize=10)
    legend_outside_bottom(ax, ncol=3, y=-0.26)
    plt.tight_layout()
    save_fig('ch11_lvar')


# =============================================================================
# FIG 30: Empirical vs theoretical violation count distribution
# =============================================================================
def fig_violation_dist():
    n = 250
    alphas = [0.05, 0.01]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, a in zip(axes, alphas):
        k = np.arange(0, max(8, int(3*n*a))+1)
        pmf = stats.binom.pmf(k, n, a)
        ax.bar(k, pmf, color=MainBlue, alpha=0.6, edgecolor='black', lw=0.3)
        ax.axvline(n*a, color=Crimson, lw=1.0, ls='--',
                   label=f'Expected = {n*a:.1f}')
        crit_low = stats.binom.ppf(0.025, n, a)
        crit_hi  = stats.binom.ppf(0.975, n, a)
        ax.axvspan(crit_low, crit_hi, color=Forest, alpha=0.15,
                   label=f'95% CI = [{crit_low:.0f}, {crit_hi:.0f}]')
        ax.set_xlabel(f'Number of violations $K$ ($\\alpha={a}$)')
        ax.set_ylabel('$P(K=k)$')
        ax.set_title(f'Binomial({n}, {a})', color=MainBlue, fontsize=9)
        legend_outside_bottom(ax, ncol=2, y=-0.28)
    plt.tight_layout()
    save_fig('ch11_violation_dist')


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('SFM Chapter 11 - generating all charts (transparent, ENG, legend bottom)')
    print('=' * 70)
    fig_returns_overview()
    fig_var_concept()
    fig_var_es()
    fig_subadditivity()
    fig_methods_comparison()
    fig_rolling_var_es()
    fig_garch_var()
    fig_evt_var()
    fig_kupiec_distribution()
    fig_christoffersen_clustering()
    fig_traffic_light()
    fig_acerbi_szekely()
    fig_bvb_case()
    fig_lehman_2008()
    fig_var_horizons()
    fig_method_hs()
    fig_method_normal()
    fig_method_student_t()
    fig_method_mc()
    fig_filtered_hs()
    fig_cornish_fisher()
    fig_hill_plot()
    fig_gjr_garch()
    fig_block_maxima()
    fig_es_var_ratio()
    fig_lopez_loss()
    fig_crypto_var()
    fig_cross_asset()
    fig_lvar()
    fig_violation_dist()
    print('=' * 70)
    print('Done.')
