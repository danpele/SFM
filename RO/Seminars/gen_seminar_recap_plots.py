"""Generate all matplotlib plots for seminar_recapitulare_examen_ro.
Outputs in ../../charts/ for use via \\includegraphics.

Stil: fond transparent, legendă explicită sub axă, fără grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
CHARTS = Path(__file__).resolve().parent.parent.parent / "charts"
CHARTS.mkdir(exist_ok=True)

MAIN_BLUE = (26 / 255, 58 / 255, 110 / 255)
CRIMSON = (220 / 255, 53 / 255, 69 / 255)
AMBER = (181 / 255, 133 / 255, 63 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)
PURPLE = (142 / 255, 68 / 255, 173 / 255)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
})


def place_legend(ax, ncol=None):
    """Place legend centered below the axes, frameless."""
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    n = ncol if ncol is not None else min(len(labels), 3)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=n, frameon=False)


def style_axes(ax):
    ax.grid(False)
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(CHARTS / name, transparent=True, bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)


# Acklam approximation pentru cuantila normală (fără scipy)
def norm_ppf(q):
    from math import sqrt, log
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        qq = sqrt(-2 * log(q))
        return (((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5]) / \
               ((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)
    if q <= phigh:
        qq = q - 0.5
        r = qq * qq
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * qq / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    qq = sqrt(-2 * log(1 - q))
    return -(((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5]) / \
           ((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)


# =====================================================================
# 1) QQ-plot
# =====================================================================
def qq_plot():
    n = 1000
    data = np.random.standard_t(df=5, size=n)
    data = (data - data.mean()) / data.std()
    sorted_data = np.sort(data)
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([norm_ppf(pp) for pp in p])

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(theoretical, sorted_data, s=14, color=CRIMSON, alpha=0.65,
               edgecolor="none", label="empirical points")
    lim = max(abs(theoretical[0]), abs(theoretical[-1])) + 0.2
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.5,
            color=MAIN_BLUE, label="$y = x$ (normal)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Theoretical Normal quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ-plot: empirical returns vs Normal")
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_sem_recap_qq.pdf")


# =====================================================================
# 2) ACF correlogram
# =====================================================================
def acf_plot():
    n = 500
    ci = 1.96 / np.sqrt(n)
    lags = np.arange(1, 21)
    rhos = np.array([
        0.12, -0.04, 0.03, -0.01, 0.10, 0.03, -0.055, 0.015, -0.02, 0.008,
        0.025, -0.015, 0.011, -0.007, 0.02, 0.005, -0.017, 0.008, -0.003, 0.012
    ])
    sig_mask = np.abs(rhos) > ci

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(lags[sig_mask], rhos[sig_mask], width=0.6,
           color=CRIMSON, edgecolor="black", linewidth=0.4,
           label="significant")
    ax.bar(lags[~sig_mask], rhos[~sig_mask], width=0.6,
           color=MAIN_BLUE, edgecolor="black", linewidth=0.4,
           label="not significant")
    ax.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
               label=f"$\\pm 1{{,}}96/\\sqrt{{n}} = \\pm{ci:.3f}$")
    ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel(r"$\hat\rho_k$")
    ax.set_title(f"ACF of daily returns, $n = {n}$")
    ax.set_xticks(lags)
    ax.set_ylim(-0.2, 0.2)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_acf.pdf")


# =====================================================================
# 3) Rolling volatility
# =====================================================================
def rolling_vol():
    months = np.arange(84)
    base = 1.2 + 0.2 * np.sin(months / 6) + np.random.normal(0, 0.15, len(months))
    covid_mask = (months >= 26) & (months <= 34)
    base[covid_mask] += [1.5, 3.2, 4.8, 4.2, 3.5, 2.5, 1.8, 1.2, 0.6]
    war_mask = (months >= 50) & (months <= 58)
    base[war_mask] += [1.0, 2.3, 2.6, 2.4, 2.0, 1.5, 1.0, 0.6, 0.3]
    vol = np.maximum(base, 0.5)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    years = 2018 + months / 12
    ax.fill_between([2020 + 2 / 12, 2020 + 10 / 12], 0, 7, alpha=0.18,
                    color=CRIMSON, label="COVID-19")
    ax.fill_between([2022 + 2 / 12, 2022 + 10 / 12], 0, 7, alpha=0.18,
                    color=AMBER, label="inflation / war")
    ax.plot(years, vol, color=MAIN_BLUE, linewidth=1.8,
            label=r"$\hat\sigma$ monthly (\%)")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\hat\sigma$ (\%)")
    ax.set_title("Rolling volatility (3M) --- crisis regimes")
    ax.set_xlim(2018, 2025)
    ax.set_ylim(0, 7)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_volregimes.pdf")


# =====================================================================
# 4) Sharpe vs MDD bubble
# =====================================================================
def sharpe_mdd_bubble():
    funds = {
        "Alpha (tech)":       dict(mu=18, sigma=28, mdd=-42, sharpe=0.57),
        "Beta (diversificat)":dict(mu=10, sigma=12, mdd=-15, sharpe=0.67),
        "Gamma (obligațiuni)":dict(mu=5,  sigma=4,  mdd=-6,  sharpe=0.75),
        "Delta (hedge)":      dict(mu=14, sigma=18, mdd=-22, sharpe=0.67),
        "Epsilon (emerging)": dict(mu=22, sigma=45, mdd=-65, sharpe=0.44),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors_map = [MAIN_BLUE, FOREST, CRIMSON, AMBER, PURPLE]
    for (name, d), color in zip(funds.items(), colors_map):
        ax.scatter(abs(d["mdd"]), d["sharpe"], s=d["sigma"] * 25,
                   color=color, alpha=0.65, edgecolor="black", linewidth=1,
                   label=name)
    ax.set_xlabel(r"$|\mathrm{MDD}|$ (\%)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(r"5 funds: Sharpe vs MDD (diameter $\propto \sigma$)")
    ax.set_xlim(0, 75)
    ax.set_ylim(0.35, 0.85)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_funds_bubble.pdf")


# =====================================================================
# 5) Density overlay
# =====================================================================
def density_overlay():
    n = 2000
    data = np.random.standard_t(df=5, size=n)
    data = (data - data.mean()) / data.std() * 0.012

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.hist(data, bins=60, density=True, color="gray",
            alpha=0.45, edgecolor="white", linewidth=0.4,
            label="empirical histogram")

    x = np.linspace(-0.06, 0.06, 400)
    sigma = data.std()
    normal = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    nu = 5
    s = sigma * np.sqrt((nu - 2) / nu)
    from math import gamma, pi
    coef = gamma((nu + 1) / 2) / (gamma(nu / 2) * np.sqrt(nu * pi) * s)
    student = coef * (1 + (x / s) ** 2 / nu) ** (-(nu + 1) / 2)
    nu2 = 1.7
    s2 = sigma * 0.70
    coef2 = gamma((nu2 + 1) / 2) / (gamma(nu2 / 2) * np.sqrt(nu2 * pi) * s2)
    stable_like = coef2 * (1 + (x / s2) ** 2 / nu2) ** (-(nu2 + 1) / 2)

    ax.plot(x, normal, color=MAIN_BLUE, linewidth=2, label=r"Normal$(0, \sigma)$")
    ax.plot(x, student, color=CRIMSON, linewidth=2, label=r"Student-$t(\nu=5)$")
    ax.plot(x, stable_like, color=FOREST, linewidth=2, linestyle="--",
            label=r"Stable ($\alpha \approx 1.7$)")
    ax.set_xlabel("daily return $r$")
    ax.set_ylabel("density")
    ax.set_title("Returns histogram vs theoretical densities")
    ax.set_xlim(-0.06, 0.06)
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_sem_recap_density_overlay.pdf")


# =====================================================================
# 6) Price vs return
# =====================================================================
def prices_vs_returns():
    T = 500
    dP = np.random.normal(0.0005, 0.012, T)
    P = 100 * np.exp(np.cumsum(dP))
    r = dP
    t = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True)
    ax1.plot(t, P, color=MAIN_BLUE, linewidth=1.5, label=r"price $P_t$ (I(1))")
    ax1.set_ylabel("price")
    ax1.set_title(r"$P_t \sim I(1)$ vs $r_t \sim I(0)$")
    style_axes(ax1)
    place_legend(ax1, ncol=1)

    ax2.plot(t, r * 100, color=CRIMSON, linewidth=0.7,
             label=r"log-return $r_t$ (I(0))")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("day $t$")
    ax2.set_ylabel("return (\\%)")
    style_axes(ax2)
    place_legend(ax2, ncol=1)
    fig.tight_layout()
    save(fig, "sfm_sem_recap_price_vs_return.pdf")


# =====================================================================
# 7) Stable densities
# =====================================================================
def stable_alpha_densities():
    x = np.linspace(-6, 6, 500)
    alphas = [(2.0, MAIN_BLUE, r"Normal ($\alpha=2$)"),
              (1.7, FOREST, r"$\alpha=1{,}7$"),
              (1.3, AMBER, r"$\alpha=1{,}3$"),
              (1.0, CRIMSON, r"Cauchy ($\alpha=1$)")]
    from math import gamma, pi

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for alpha, color, label in alphas:
        if alpha == 2.0:
            y = np.exp(-x ** 2 / 2) / np.sqrt(2 * pi)
        else:
            nu = alpha
            coef = gamma((nu + 1) / 2) / (gamma(nu / 2) * np.sqrt(nu * pi))
            y = coef * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)
        ax.plot(x, y, linewidth=2, color=color, label=label)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density $f(x)$")
    ax.set_title(r"Symmetric stable densities for various $\alpha$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.42)
    style_axes(ax)
    place_legend(ax, ncol=4)
    save(fig, "sfm_sem_recap_stable_alpha.pdf")


# =====================================================================
# 8) ACF pe pătrate
# =====================================================================
def acf_squared():
    n = 500
    ci = 1.96 / np.sqrt(n)
    lags = np.arange(1, 21)
    rho_sq = np.array([
        0.28, 0.21, 0.17, 0.14, 0.12, 0.10, 0.085, 0.072, 0.060, 0.052,
        0.048, 0.040, 0.035, 0.030, 0.028, 0.025, 0.022, 0.020, 0.018, 0.015
    ])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(lags, rho_sq, width=0.6, color=CRIMSON, edgecolor="black",
           linewidth=0.4, label=r"$\hat\rho_k$ pe $r_t^2$")
    ax.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
               label=f"$\\pm 1{{,}}96/\\sqrt{{n}} = \\pm{ci:.3f}$")
    ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel(r"$\hat\rho_k$")
    ax.set_title(r"ACF on squared returns --- GARCH effect")
    ax.set_xticks(lags)
    ax.set_ylim(-0.1, 0.35)
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_sem_recap_acf_squared.pdf")


# =====================================================================
# 9) Hill plot
# =====================================================================
def hill_plot():
    ks = np.arange(20, 500, 10)
    alphas = 3.0 + 0.8 * np.exp(-ks / 60) - 0.5 * (ks - 250) / 500 + \
             np.random.normal(0, 0.1, len(ks))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvspan(80, 200, alpha=0.15, color=FOREST, label="plateau (stable region)")
    ax.plot(ks, alphas, color=MAIN_BLUE, linewidth=1.5,
            label=r"$\hat\alpha_H(k)$")
    ax.axhline(y=3.0, color=CRIMSON, linestyle="--", linewidth=1.2,
               label=r"$\alpha = 3$ (target)")
    ax.set_xlabel(r"$k$ = number of order statistics")
    ax.set_ylabel(r"$\hat\alpha_H$")
    ax.set_title(r"Hill plot --- selection of optimal $k$")
    ax.set_ylim(1.8, 4.5)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_hill.pdf")


# =====================================================================
# 10) Efficient frontier 2 assets
# =====================================================================
def efficient_frontier_2assets():
    mu_a, mu_b = 0.08, 0.14
    s_a, s_b = 0.12, 0.22
    rho = 0.25
    w = np.linspace(0, 1, 100)
    mu_p = w * mu_a + (1 - w) * mu_b
    var_p = (w * s_a) ** 2 + ((1 - w) * s_b) ** 2 + \
            2 * w * (1 - w) * s_a * s_b * rho
    sig_p = np.sqrt(var_p)
    idx_mv = np.argmin(var_p)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(sig_p, mu_p, color=MAIN_BLUE, linewidth=2,
            label="portfolio frontier")
    ax.scatter([s_a, s_b], [mu_a, mu_b], s=80, color=CRIMSON,
               zorder=5, label="A, B individual")
    ax.scatter([sig_p[idx_mv]], [mu_p[idx_mv]], s=120, color=FOREST,
               marker="*", zorder=5, label="min-var portfolio")
    ax.annotate("A", (s_a, mu_a), xytext=(5, -12), textcoords="offset points")
    ax.annotate("B", (s_b, mu_b), xytext=(5, 0), textcoords="offset points")
    ax.annotate("MV", (sig_p[idx_mv], mu_p[idx_mv]),
                xytext=(-24, 8), textcoords="offset points", color=FOREST)
    ax.set_xlabel(r"portfolio $\sigma$")
    ax.set_ylabel(r"portfolio $\mu$")
    ax.set_title(r"Efficient frontier --- 2 assets, $\rho = 0.25$")
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_frontier.pdf")


# =====================================================================
# 11) CUSUM
# =====================================================================
def cusum_chart():
    n = 500
    mu0 = 0.0
    x = np.concatenate([
        np.random.normal(0, 1, 250),
        np.random.normal(0.15, 1, 250)
    ])
    cusum = np.cumsum(x - mu0)
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, cusum, color=MAIN_BLUE, linewidth=1.2, label="CUSUM $S_t$")
    ax.axvline(250, color=CRIMSON, linestyle="--", linewidth=1.2,
               label=r"$t = 250$ (true break)")
    ax.axhline(0, color="black", linewidth=0.5)
    h = 4 * np.sqrt(t + 1)
    ax.plot(t, h, color="gray", linestyle=":", linewidth=1, label=r"$\pm h\sqrt{t}$")
    ax.plot(t, -h, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("day $t$")
    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM test --- structural break detection")
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_sem_recap_cusum.pdf")


# =====================================================================
# 12) Monte Carlo
# =====================================================================
def monte_carlo_paths():
    T = 252
    n_paths = 100
    mu = 0.10 / T
    sigma = 0.20 / np.sqrt(T)
    paths = np.zeros((n_paths, T + 1))
    paths[:, 0] = 100
    for i in range(n_paths):
        rets = np.random.normal(mu, sigma, T)
        paths[i, 1:] = 100 * np.exp(np.cumsum(rets))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4),
                                    gridspec_kw={"width_ratios": [2, 1]})
    for i in range(n_paths):
        ax1.plot(paths[i], color=MAIN_BLUE, alpha=0.25, linewidth=0.7)
    # Representative line for the "path" handle
    ax1.plot([], [], color=MAIN_BLUE, alpha=0.6, linewidth=1,
             label="simulated paths")
    ax1.plot(np.mean(paths, axis=0), color=CRIMSON, linewidth=2,
             label=r"mean $E[P_t]$")
    ax1.set_xlabel("day $t$")
    ax1.set_ylabel("price $P_t$")
    ax1.set_title("Monte Carlo --- 100 GBM paths (1 year)")
    style_axes(ax1)
    place_legend(ax1, ncol=2)

    ax2.hist(paths[:, -1], bins=20, orientation="horizontal",
             color=FOREST, alpha=0.7, edgecolor="white",
             label=r"histogram $P_T$")
    ax2.set_xlabel("frequency")
    ax2.set_title(r"Final $P_T$")
    style_axes(ax2)
    place_legend(ax2, ncol=1)
    fig.tight_layout()
    save(fig, "sfm_sem_recap_monte_carlo.pdf")


# =====================================================================
# 13) Leverage effect
# =====================================================================
def leverage_effect():
    n = 500
    r = np.random.normal(0, 0.012, n)
    sigma_next = 0.012 - 0.8 * r + np.abs(np.random.normal(0, 0.003, n))
    sigma_next = np.maximum(sigma_next, 0.002)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.scatter(r * 100, sigma_next * 100, s=12, color=MAIN_BLUE, alpha=0.45,
               edgecolor="none", label="consecutive days")
    z = np.polyfit(r, sigma_next, 1)
    xx = np.linspace(r.min(), r.max(), 100)
    ax.plot(xx * 100, (z[0] * xx + z[1]) * 100, color=CRIMSON,
            linewidth=1.8, label=f"OLS regression (slope = {z[0]:.2f})")
    ax.set_xlabel(r"$r_t$ (\%)")
    ax.set_ylabel(r"$\hat\sigma_{t+1}$ (\%)")
    ax.set_title(r"Leverage effect: $r_t < 0 \Rightarrow$ higher $\sigma_{t+1}$")
    ax.axvline(0, color="black", linewidth=0.4)
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_sem_recap_leverage.pdf")


if __name__ == "__main__":
    qq_plot()
    acf_plot()
    rolling_vol()
    sharpe_mdd_bubble()
    density_overlay()
    prices_vs_returns()
    stable_alpha_densities()
    acf_squared()
    hill_plot()
    efficient_frontier_2assets()
    cusum_chart()
    monte_carlo_paths()
    leverage_effect()
    print("Generated 13 plots with explicit legends in", CHARTS)
