"""Additional matplotlib plots for seminar_recapitulare_examen_ro."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(123)
CHARTS = Path(__file__).resolve().parent.parent.parent / "charts"

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


def style_axes(ax):
    ax.grid(False)
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.get_legend()
    if leg is not None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, frameon=False)


def save(fig, name):
    fig.savefig(CHARTS / name, transparent=True, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------
# 1) ACF randamente pătratice (GARCH signature)
# -----------------------------------------------------------------
def acf_squared():
    n = 500
    ci = 1.96 / np.sqrt(n)
    lags = np.arange(1, 21)
    rho_sq = np.array([
        0.28, 0.21, 0.17, 0.14, 0.12, 0.10, 0.085, 0.072, 0.060, 0.052,
        0.048, 0.040, 0.035, 0.030, 0.028, 0.025, 0.022, 0.020, 0.018, 0.015
    ])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(lags, rho_sq, width=0.6, color=CRIMSON, edgecolor="black",
           linewidth=0.4, label=r"$\hat\rho_k(r_t^2)$ on squares")
    ax.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
               label=f"$\\pm 1{{,}}96/\\sqrt{{n}} = \\pm{ci:.3f}$")
    ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel(r"$\hat\rho_k$")
    ax.set_title(r"ACF on squared returns --- ``GARCH effect''")
    ax.set_xticks(lags)
    ax.set_ylim(-0.1, 0.35)
    ax.legend()
    style_axes(ax)
    save(fig, "sfm_sem_recap_acf_squared.pdf")


# -----------------------------------------------------------------
# 2) Hill plot: alpha estimate vs k
# -----------------------------------------------------------------
def hill_plot():
    ks = np.arange(20, 500, 10)
    # Simulated Hill plot: stable around alpha=3 for mid k, biased at extremes
    alphas = 3.0 + 0.8 * np.exp(-ks / 60) - 0.5 * (ks - 250) / 500 + \
             np.random.normal(0, 0.1, len(ks))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, alphas, color=MAIN_BLUE, linewidth=1.5,
            label=r"$\hat\alpha_H(k)$")
    ax.axhline(y=3.0, color=CRIMSON, linestyle="--", linewidth=1.2,
               label=r"$\alpha = 3$ (target)")
    ax.axvspan(80, 200, alpha=0.15, color=FOREST, label="stable region (plateau)")
    ax.set_xlabel(r"$k$ = number of order statistics included")
    ax.set_ylabel(r"$\hat\alpha_H$")
    ax.set_title(r"Hill plot --- selection of optimal $k$")
    ax.set_ylim(1.8, 4.5)
    ax.legend()
    style_axes(ax)
    save(fig, "sfm_sem_recap_hill.pdf")


# -----------------------------------------------------------------
# 3) Efficient frontier (2 assets + portofoliu cu varianță minimă)
# -----------------------------------------------------------------
def efficient_frontier_2assets():
    mu_a, mu_b = 0.08, 0.14
    s_a, s_b = 0.12, 0.22
    rho = 0.25
    w = np.linspace(0, 1, 100)
    mu_p = w * mu_a + (1 - w) * mu_b
    var_p = (w * s_a) ** 2 + ((1 - w) * s_b) ** 2 + \
            2 * w * (1 - w) * s_a * s_b * rho
    sig_p = np.sqrt(var_p)
    # Minimum variance portfolio
    idx_mv = np.argmin(var_p)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sig_p, mu_p, color=MAIN_BLUE, linewidth=2, label="2-asset frontier")
    ax.scatter([s_a, s_b], [mu_a, mu_b], s=80, color=CRIMSON,
               zorder=5, label="A, B (individual)")
    ax.scatter([sig_p[idx_mv]], [mu_p[idx_mv]], s=100, color=FOREST,
               marker="*", zorder=5, label="min-var portfolio")
    ax.annotate("A", (s_a, mu_a), xytext=(5, -12), textcoords="offset points")
    ax.annotate("B", (s_b, mu_b), xytext=(5, 0), textcoords="offset points")
    ax.annotate("MV", (sig_p[idx_mv], mu_p[idx_mv]),
                xytext=(-24, 8), textcoords="offset points", color=FOREST)
    ax.set_xlabel(r"portfolio $\sigma$")
    ax.set_ylabel(r"portfolio $\mu$")
    ax.set_title(r"Efficient frontier --- 2 assets, $\rho = 0.25$")
    ax.legend()
    style_axes(ax)
    save(fig, "sfm_sem_recap_frontier.pdf")


# -----------------------------------------------------------------
# 4) CUSUM chart — structural break detection
# -----------------------------------------------------------------
def cusum_chart():
    n = 500
    mu0 = 0.0
    # shift from 0 to 0.15 at t=250
    x = np.concatenate([
        np.random.normal(0, 1, 250),
        np.random.normal(0.15, 1, 250)
    ])
    cusum = np.cumsum(x - mu0)
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, cusum, color=MAIN_BLUE, linewidth=1.2, label="CUSUM")
    ax.axvline(250, color=CRIMSON, linestyle="--", linewidth=1.2,
               label="$t = 250$ (true break)")
    ax.axhline(0, color="black", linewidth=0.5)
    # Boundaries
    h = 4 * np.sqrt(t + 1)
    ax.plot(t, h, color="gray", linestyle=":", linewidth=1, label=r"$\pm h$")
    ax.plot(t, -h, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("day $t$")
    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM test --- structural break detection")
    ax.legend()
    style_axes(ax)
    save(fig, "sfm_sem_recap_cusum.pdf")


# -----------------------------------------------------------------
# 5) Monte Carlo paths + histogram of terminal wealth
# -----------------------------------------------------------------
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4),
                                    gridspec_kw={"width_ratios": [2, 1]})
    for i in range(n_paths):
        ax1.plot(paths[i], color=MAIN_BLUE, alpha=0.25, linewidth=0.7)
    ax1.plot(np.mean(paths, axis=0), color=CRIMSON, linewidth=2,
             label=r"mean $E[P_t]$")
    ax1.set_xlabel("day $t$")
    ax1.set_ylabel("price $P_t$")
    ax1.set_title("Monte Carlo --- 100 GBM paths (1 year)")
    ax1.legend()
    style_axes(ax1)
    ax2.hist(paths[:, -1], bins=20, orientation="horizontal",
             color=FOREST, alpha=0.7, edgecolor="white")
    ax2.set_xlabel("frequency")
    ax2.set_title("Final $P_T$")
    style_axes(ax2)
    fig.tight_layout()
    save(fig, "sfm_sem_recap_monte_carlo.pdf")


# -----------------------------------------------------------------
# 6) Volatility smile / leverage effect scatter
# -----------------------------------------------------------------
def leverage_effect():
    n = 500
    r = np.random.normal(0, 0.012, n)
    # Leverage: vol(t+1) ~ -alpha * r(t)
    sigma_next = 0.012 - 0.8 * r + np.abs(np.random.normal(0, 0.003, n))
    sigma_next = np.maximum(sigma_next, 0.002)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(r * 100, sigma_next * 100, s=12, color=MAIN_BLUE, alpha=0.45,
               edgecolor="none", label="consecutive days")
    # Fit line
    z = np.polyfit(r, sigma_next, 1)
    xx = np.linspace(r.min(), r.max(), 100)
    ax.plot(xx * 100, (z[0] * xx + z[1]) * 100, color=CRIMSON,
            linewidth=1.8, label=f"OLS fit: slope = {z[0]:.2f}")
    ax.set_xlabel(r"$r_t$ (\%)")
    ax.set_ylabel(r"$\hat\sigma_{t+1}$ (\%)")
    ax.set_title("Leverage effect: $r_t < 0 \\Rightarrow$ higher $\\sigma_{t+1}$")
    ax.axvline(0, color="black", linewidth=0.4)
    ax.legend()
    style_axes(ax)
    save(fig, "sfm_sem_recap_leverage.pdf")


if __name__ == "__main__":
    acf_squared()
    hill_plot()
    efficient_frontier_2assets()
    cusum_chart()
    monte_carlo_paths()
    leverage_effect()
    print("Generated 6 new plots in", CHARTS)
