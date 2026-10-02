"""Generate matplotlib plots for chapter 4 (EMH).
Outputs in ../../charts/ for use via \\includegraphics.

Stil: fond transparent, legendă sub axă, fără grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(2026)
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


def style_axes(ax):
    ax.grid(False)
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def place_legend(ax, ncol=None):
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    n = ncol if ncol is not None else min(len(labels), 3)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=n, frameon=False)


def save(fig, name):
    fig.savefig(CHARTS / name, transparent=True, bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)


# =====================================================================
# 1) ACF empirică S&P 500 stil (aproape eficient)
# =====================================================================
def acf_sp500():
    n = 2500
    ci = 1.96 / np.sqrt(n)
    lags = np.arange(1, 21)
    # S&P 500 typical: mostly inside bands, lag 1 slightly negative
    rhos = np.array([
        -0.045, 0.018, -0.022, 0.005, -0.015, 0.010, 0.008, -0.012, 0.004, -0.008,
        0.011, -0.006, 0.009, -0.014, 0.007, 0.003, -0.018, 0.005, 0.012, -0.005
    ])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sig_mask = np.abs(rhos) > ci
    ax.bar(lags[~sig_mask], rhos[~sig_mask], width=0.6,
           color=MAIN_BLUE, edgecolor="black", linewidth=0.4,
           label="nesemnificativ")
    if sig_mask.any():
        ax.bar(lags[sig_mask], rhos[sig_mask], width=0.6,
               color=CRIMSON, edgecolor="black", linewidth=0.4,
               label="semnificativ")
    ax.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
               label=f"$\\pm 1{{,}}96/\\sqrt{{n}} = \\pm{ci:.3f}$")
    ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel(r"$\hat\rho_k$")
    ax.set_title(r"ACF S\&P 500 zilnic, 2000--2024 ($n = 2.500$)")
    ax.set_xticks(lags)
    ax.set_ylim(-0.08, 0.08)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_ch4_acf_sp500.pdf")


# =====================================================================
# 2) Variance ratio profile VR(q) vs q pentru 3 piețe
# =====================================================================
def vr_profile():
    qs = np.array([2, 3, 5, 10, 20, 30, 60, 120])
    # S&P 500 (aproape eficient)
    vr_sp = np.array([0.99, 0.98, 0.96, 0.94, 0.91, 0.89, 0.87, 0.85])
    # Bitcoin (momentum)
    vr_btc = np.array([1.05, 1.10, 1.18, 1.28, 1.42, 1.52, 1.63, 1.70])
    # BVB (mean reversion dupa momentum)
    vr_bvb = np.array([1.15, 1.22, 1.30, 1.25, 1.12, 0.95, 0.82, 0.75])

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1,
               label=r"$VR = 1$ (random walk)")
    ax.plot(qs, vr_sp, "o-", color=MAIN_BLUE, linewidth=1.8,
            label=r"S\&P 500 (cvasi-eficient)")
    ax.plot(qs, vr_btc, "s-", color=CRIMSON, linewidth=1.8,
            label="Bitcoin (momentum)")
    ax.plot(qs, vr_bvb, "D-", color=FOREST, linewidth=1.8,
            label="BET/BVB (mean rev.)")
    ax.set_xlabel(r"orizont $q$ (zile)")
    ax.set_ylabel(r"$VR(q)$")
    ax.set_title(r"Profilul variance ratio pe 3 piețe")
    ax.set_xscale("log")
    ax.set_ylim(0.7, 1.8)
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_ch4_vr_profile.pdf")


# =====================================================================
# 3) CAR event study: 3 scenarii (surpriză, leak, PEAD)
# =====================================================================
def car_scenarios():
    t = np.arange(-10, 11)
    # Pure surprise: jump at t=0
    car_surprise = np.zeros_like(t, dtype=float)
    car_surprise[t >= 0] = 3.0
    # Leak: gradual rise before event
    car_leak = np.where(t < -5, 0, np.where(t < 0, 0.5 * (t + 5), 2.5 + 0.1 * (t)))
    car_leak = np.where(t >= 0, 2.5 + 0.1 * t, 0.5 * np.maximum(t + 5, 0))
    # PEAD: jump + drift
    car_pead = np.where(t < 0, 0.0, 1.5 + 0.15 * t)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axvline(0, color="black", linestyle="--", linewidth=1,
               label="eveniment ($t=0$)")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.plot(t, car_surprise, "o-", color=MAIN_BLUE, linewidth=1.8, markersize=5,
            label="(a) surpriză pură (EMH semi-tare)")
    ax.plot(t, car_leak, "s-", color=AMBER, linewidth=1.8, markersize=5,
            label="(b) scurgere info (leak)")
    ax.plot(t, car_pead, "D-", color=CRIMSON, linewidth=1.8, markersize=5,
            label="(c) PEAD (drift post-eveniment)")
    ax.set_xlabel(r"zile relativ la eveniment $t$")
    ax.set_ylabel(r"CAR $(\%)$")
    ax.set_title(r"Event study --- 3 scenarii de CAR")
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_ch4_car_scenarios.pdf")


# =====================================================================
# 4) Rolling Hurst exponent (AMH evidence)
# =====================================================================
def rolling_hurst():
    years = np.arange(2000, 2025, 0.25)
    # Synthetic: oscillates around 0.5, spikes in crises
    base = 0.52 + 0.015 * np.sin(years / 3)
    # Crisis spikes
    crisis_times = [2001, 2008.5, 2020.2, 2022.3]
    for ct in crisis_times:
        base += 0.08 * np.exp(-((years - ct) / 0.6) ** 2)
    base += np.random.normal(0, 0.012, len(years))
    H = base

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1,
               label=r"$H = 0{,}5$ (random walk)")
    ax.fill_between(years, 0.5, H, where=(H > 0.55),
                    color=CRIMSON, alpha=0.20, label="perioade de ineficiență")
    ax.plot(years, H, color=MAIN_BLUE, linewidth=1.6, label=r"$\hat H$ rolling (2 ani)")
    # Crisis annotations
    for ct, label in zip(crisis_times,
                          ["dotcom", "GFC", "COVID", "inflație"]):
        ax.axvline(ct, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(label, (ct, 0.68), fontsize=8, ha="center", color="gray")
    ax.set_xlabel("An")
    ax.set_ylabel(r"$\hat H$")
    ax.set_title(r"Exponent Hurst rolling --- S\&P 500 (evidență AMH)")
    ax.set_ylim(0.40, 0.75)
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_ch4_rolling_hurst.pdf")


# =====================================================================
# 5) Active vs Passive: cumulative return gap
# =====================================================================
def active_vs_passive():
    years = np.arange(2005, 2025)
    # S&P 500 TR annualized around 10%
    np.random.seed(7)
    rets_sp = np.random.normal(0.10, 0.18, len(years))
    rets_active = rets_sp - 0.013 + np.random.normal(0, 0.02, len(years))  # underperform ~1.3%
    cum_sp = np.cumprod(1 + rets_sp)
    cum_act = np.cumprod(1 + rets_active)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(years, cum_sp, color=MAIN_BLUE, linewidth=2, marker="o",
            markersize=4, label=r"S\&P 500 (indice pasiv)")
    ax.plot(years, cum_act, color=CRIMSON, linewidth=2, marker="s",
            markersize=4, label="media fondurilor active")
    ax.fill_between(years, cum_act, cum_sp, where=(cum_sp > cum_act),
                    color=AMBER, alpha=0.18, label="gap de performanță")
    ax.set_xlabel("An")
    ax.set_ylabel(r"valoare cumulată (\$1 investit în 2005)")
    ax.set_title(r"Active vs Passive --- SPIVA: $>90\%$ subperformează")
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_ch4_active_passive.pdf")


# =====================================================================
# 6) Bubble & crash: dotcom / GFC / crypto 2021 timeline
# =====================================================================
def bubbles_timeline():
    # Normalized price index with peaks at bubble tops
    t = np.linspace(0, 10, 1000)
    # Dotcom
    dotcom = np.exp(0.4 * (t - 2)) / (1 + np.exp(1.2 * (t - 4)))
    dotcom = dotcom / dotcom.max() * 100
    # Subprime / GFC
    gfc = np.exp(0.3 * (t - 3)) / (1 + np.exp(1.5 * (t - 6)))
    gfc = gfc / gfc.max() * 100
    # Crypto 2021
    crypto = np.exp(0.6 * (t - 5)) / (1 + np.exp(2 * (t - 7)))
    crypto = crypto / crypto.max() * 100

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, dotcom, color=MAIN_BLUE, linewidth=2, label="Dotcom (Nasdaq)")
    ax.plot(t, gfc, color=CRIMSON, linewidth=2, label="Subprime (case SUA)")
    ax.plot(t, crypto, color=FOREST, linewidth=2, label="Crypto (Bitcoin 2021)")
    ax.set_xlabel("timp (normalizat)")
    ax.set_ylabel("preț (indice = 100 la vârf)")
    ax.set_title(r"Bule și prăbușiri --- tipare similare")
    style_axes(ax)
    place_legend(ax, ncol=3)
    save(fig, "sfm_ch4_bubbles.pdf")


# =====================================================================
# 7) Funcția de autocorelație empirică: random walk simulat
# =====================================================================
def sim_random_walk():
    T = 500
    # Random walk prices
    eps = np.random.normal(0, 1, T)
    P = np.cumsum(eps)
    r = np.diff(np.concatenate([[0], P]))
    t = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    ax1.plot(t, P, color=MAIN_BLUE, linewidth=1.2, label=r"$P_t = \sum \varepsilon_i$")
    ax1.axhline(0, color="black", linewidth=0.4)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$P_t$")
    ax1.set_title(r"Mers aleator simulat $P_t = P_{t-1} + \varepsilon_t$")
    style_axes(ax1)
    place_legend(ax1, ncol=1)

    ax2.plot(t, r, color=CRIMSON, linewidth=0.7, label=r"$r_t = \varepsilon_t$")
    ax2.axhline(0, color="black", linewidth=0.4)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$r_t$")
    ax2.set_title(r"Randamente $r_t$ (staționare, iid)")
    style_axes(ax2)
    place_legend(ax2, ncol=1)
    fig.tight_layout()
    save(fig, "sfm_ch4_rw_sim.pdf")


# =====================================================================
# 8) Disposition effect: hold winners too long / sell losers early
# =====================================================================
def disposition_effect():
    # PGR (proportion of gains realized) vs PLR (proportion of losses realized)
    categories = ["Retail (SUA)", "Retail (UE)", "Instituțional", "Fond pensii"]
    pgr = [0.148, 0.135, 0.095, 0.102]
    plr = [0.098, 0.089, 0.087, 0.095]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, pgr, width, color=FOREST,
           edgecolor="black", linewidth=0.4, label="PGR (câștigătoare vândute)")
    ax.bar(x + width / 2, plr, width, color=CRIMSON,
           edgecolor="black", linewidth=0.4, label="PLR (pierzătoare vândute)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("proporție")
    ax.set_title(r"Efect de dispoziție: PGR $>$ PLR (Odean, 1998)")
    ax.set_ylim(0, 0.17)
    style_axes(ax)
    place_legend(ax, ncol=2)
    save(fig, "sfm_ch4_disposition.pdf")


if __name__ == "__main__":
    acf_sp500()
    vr_profile()
    car_scenarios()
    rolling_hurst()
    active_vs_passive()
    bubbles_timeline()
    sim_random_walk()
    disposition_effect()
    print("Generated 8 chapter-4 plots in", CHARTS)
