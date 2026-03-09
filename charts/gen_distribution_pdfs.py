"""
Generate theoretical PDF charts for each distribution type used in Chapter 2.
Each chart shows the density curve from the mathematical formula.
Style: transparent background, legend outside (bottom), no grid, Quantlet style.
All text in ENGLISH.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

# Quantlet-style settings: clean, transparent, no grid
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

MAIN_BLUE = '#1a3a6e'
IDA_RED = '#cd0000'
FOREST = '#2e7d32'
AMBER = '#b5853f'
PURPLE = '#8e44ad'
CRIMSON = '#dc3545'

x = np.linspace(-6, 6, 1000)


def save_fig(fig, ax, name):
    """Save figure with legend outside at bottom, transparent bg."""
    handles, labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 5), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True)
    fig.savefig(f'{name}.png', bbox_inches='tight', transparent=True)
    plt.close(fig)


# ======================================================================
# 1. Normal Distribution PDF
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
for mu, sigma, label, color, ls in [
    (0, 1, r'$\mathcal{N}(0, 1)$', MAIN_BLUE, '-'),
    (0, 0.5, r'$\mathcal{N}(0, 0.25)$', IDA_RED, '--'),
    (0, 2, r'$\mathcal{N}(0, 4)$', FOREST, '-.'),
    (-2, 1, r'$\mathcal{N}(-2, 1)$', AMBER, ':'),
]:
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, color=color, linestyle=ls, linewidth=2, label=label)

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(r'Normal PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$')
ax.legend()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.85)
save_fig(fig, ax, 'ch2_normal_pdf_theoretical')
print("1/6 Normal PDF done")


# ======================================================================
# 2. Student-t Distribution PDF
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
for nu, label, color, ls in [
    (1, r'$t_1$ (Cauchy)', IDA_RED, '--'),
    (3, r'$t_3$', AMBER, '-.'),
    (5, r'$t_5$', PURPLE, ':'),
    (30, r'$t_{30}$', FOREST, '-'),
]:
    pdf = stats.t.pdf(x, df=nu)
    ax.plot(x, pdf, color=color, linestyle=ls, linewidth=2, label=label)

# Normal for comparison
ax.plot(x, stats.norm.pdf(x), color=MAIN_BLUE, linewidth=2, label=r'$\mathcal{N}(0,1)$')

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(r'Student-$t$ PDF: $f(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$')
ax.legend()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)
save_fig(fig, ax, 'ch2_student_t_pdf_theoretical')
print("2/6 Student-t PDF done")


# ======================================================================
# 3. Skew-t (Hansen) Distribution PDF
# ======================================================================
def hansen_skew_t_pdf(x, nu, lam):
    """Hansen's skew-t density."""
    a = 4 * lam * ((nu - 2) / (nu - 1)) * gamma_func((nu + 1) / 2) / (
        np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2)
    )
    b2 = 1 + 3 * lam**2 - a**2
    b = np.sqrt(b2)

    c = gamma_func((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2))

    result = np.zeros_like(x)
    mask = x < (-a / b)
    s = b * x + a
    result[mask] = b * c * (1 + s[mask]**2 / ((1 - lam)**2 * (nu - 2)))**(-(nu + 1) / 2)
    result[~mask] = b * c * (1 + s[~mask]**2 / ((1 + lam)**2 * (nu - 2)))**(-(nu + 1) / 2)
    return result

fig, ax = plt.subplots(figsize=(7, 4))
for nu, lam, label, color, ls in [
    (5, 0, r'$\lambda=0$ (symmetric)', MAIN_BLUE, '-'),
    (5, -0.3, r'$\lambda=-0.3$ (left skew)', IDA_RED, '--'),
    (5, 0.3, r'$\lambda=0.3$ (right skew)', FOREST, '-.'),
    (5, -0.6, r'$\lambda=-0.6$ (strong left skew)', AMBER, ':'),
]:
    pdf = hansen_skew_t_pdf(x, nu, lam)
    ax.plot(x, pdf, color=color, linestyle=ls, linewidth=2, label=label)

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(r"Hansen's skew-$t$ PDF ($\nu=5$, various $\lambda$)")
ax.legend()
ax.set_xlim(-5, 5)
save_fig(fig, ax, 'ch2_skew_t_pdf_theoretical')
print("3/6 Skew-t PDF done")


# ======================================================================
# 4. Stable Distribution PDF
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
for alpha, beta, label, color, ls in [
    (2.0, 0, r'$\alpha=2$ (Gaussian)', MAIN_BLUE, '-'),
    (1.5, 0, r'$\alpha=1.5, \beta=0$', FOREST, '--'),
    (1.0, 0, r'$\alpha=1$ (Cauchy)', IDA_RED, '-.'),
    (1.7, -0.5, r'$\alpha=1.7, \beta=-0.5$', AMBER, ':'),
    (0.5, 0, r'$\alpha=0.5$ (L\'{e}vy)', PURPLE, '-'),
]:
    try:
        pdf = stats.levy_stable.pdf(x, alpha, beta)
        ax.plot(x, pdf, color=color, linestyle=ls, linewidth=2, label=label)
    except Exception:
        pass

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(r'$\alpha$-stable PDF $S_\alpha(\beta, \gamma, \delta)$')
ax.legend()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.5)
save_fig(fig, ax, 'ch2_stable_pdf_theoretical')
print("4/6 Stable PDF done")


# ======================================================================
# 5. GEV Distribution PDF
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
x_gev = np.linspace(-4, 8, 1000)
for xi, label, color, ls in [
    (0, r'Gumbel ($\xi=0$)', MAIN_BLUE, '-'),
    (0.3, r'Fr\'{e}chet ($\xi=0.3$)', IDA_RED, '--'),
    (-0.3, r'Weibull ($\xi=-0.3$)', FOREST, '-.'),
    (0.5, r'Fr\'{e}chet ($\xi=0.5$)', AMBER, ':'),
]:
    pdf = stats.genextreme.pdf(x_gev, -xi)  # scipy uses c = -xi
    ax.plot(x_gev, pdf, color=color, linestyle=ls, linewidth=2, label=label)

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(r'GEV PDF: $H_\xi(x) = \exp\!\left(-(1+\xi x)^{-1/\xi}\right)$')
ax.legend()
ax.set_xlim(-4, 8)
ax.set_ylim(0, 0.45)
save_fig(fig, ax, 'ch2_gev_pdf_theoretical')
print("5/6 GEV PDF done")


# ======================================================================
# 6. GPD Distribution PDF
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
x_gpd = np.linspace(0, 6, 500)
for xi, label, color, ls in [
    (0, r'Exponential ($\xi=0$)', MAIN_BLUE, '-'),
    (0.25, r'Pareto ($\xi=0.25$)', IDA_RED, '--'),
    (0.5, r'Pareto ($\xi=0.5$)', AMBER, '-.'),
    (-0.5, r'Beta ($\xi=-0.5$)', FOREST, ':'),
]:
    if xi == 0:
        pdf = np.exp(-x_gpd)
    else:
        mask = (1 + xi * x_gpd) > 0
        pdf = np.zeros_like(x_gpd)
        pdf[mask] = (1 + xi * x_gpd[mask])**(-(1 + 1/xi))
    ax.plot(x_gpd, pdf, color=color, linestyle=ls, linewidth=2, label=label)

ax.set_xlabel('$x - u$ (excess over threshold)')
ax.set_ylabel('$g(x)$')
ax.set_title(r'GPD PDF: $g_{\xi,\beta}(y) = \frac{1}{\beta}\left(1+\xi\frac{y}{\beta}\right)^{-(1+1/\xi)}$')
ax.legend()
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.1)
save_fig(fig, ax, 'ch2_gpd_pdf_theoretical')
print("6/6 GPD PDF done")

print("\nAll distribution PDF charts generated successfully!")
