r"""
build_chapter16.py -- Capitolul 16 (Recapitulare), EN + RO dintr-o singură sursă
================================================================================
Harta cursului, cîte un slide de recapitulare pentru fiecare capitol 0--15 (formule, fapte empirice, greșeli
frecvente), trusa de instrumente, examenul (format, criterii, tipuri de probleme, opt probleme rezolvate pas cu pas),
proiectul de echipă și secțiunea finală „Contribuția posibilă a AI”.
Cifrele @{cheie}: faptele empirice vin din fișierele de cifre ale fiecărui capitol (Quantlets/Ch_NN/chN_numbers.json),
tabelul comun din Quantlets/Ch_16/ch16_numbers.json, iar rezolvările problemelor sînt calculate aici, în Python.
Convenția nivelului: VaR 1% (VaR_alpha = -q_alpha), ES 2,5%; niciodată „VaR 99%”.
Ieșire:
  EN/Courses/chapter16_review.tex
  RO/Cursuri/capitol16_recapitulare.tex
Rulare:
  python3 Quantlets/Ch_16/generate_all_charts.py
  python3 latex/build_chapter16.py && python3 latex/sfm_build.py compile 16
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, Values, cols, items, table, photo, enum   # noqa: E402
from ch16_common import BIB, QLURL, REFS, T, facts, load, money, summary_values   # noqa: E402

N = load()
V = summary_values(facts(Values()), N)
D = Deck(16, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_16}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.58\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'ase': ('ch0_ase_2014.jpg', C + 'Bucharest_-_Academie_de_Studii_Economice_01.jpg',
            '⟦Photo||Foto⟧: Joe Mabel (2014); CC BY 3.0; Wikimedia Commons'),
    'bvb2023': ('ch0_bvb_palace_2023.jpg', C + 'Bucure\\%C8\\%99ti_-_Palatul_Bursei_(2023)_-_img_01.jpg',
                '⟦Photo||Foto⟧: Chainwit.\\ (2023); CC BY 4.0; Wikimedia Commons'),
    'mandelbrot': ('ch0_mandelbrot_2007.jpg', C + 'Benoit_Mandelbrot_mg_1804-d.jpg',
                   '⟦Photo||Foto⟧: Rama (2007); CC BY-SA 2.0 fr; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


B = '\\textbf'
FORM = T('\\textbf{Key formulas}', '\\textbf{Formule-cheie}')
FACT = T('\\textbf{Empirical facts}', '\\textbf{Fapte empirice}')
MIST = T('\\textbf{Common mistakes}', '\\textbf{Greșeli frecvente}')


NOTN = T('\\textbf{Notation}', '\\textbf{Notațiile}')
NOTE = {
    0: [T('$\\sigma$: the standard deviation of daily returns; $A$: the number of observations per year', '$\\sigma$: abaterea standard a randamentelor zilnice; $A$: numărul de observații pe an'),
        T('$P_t$: the price on day $t$; $\\max_{s \\le t}P_s$: the highest price up to day $t$; $D_t \\le 0$', '$P_t$: prețul din ziua $t$; $\\max_{s \\le t}P_s$: cel mai mare preț pînă în ziua $t$; $D_t \\le 0$')],
    1: [T('$R_t$: the simple return, $r_t$: the log return of period $t$; $P_0$, $P_T$: the first and the last price; $Y$: the number of years', '$R_t$: randamentul simplu, $r_t$: randamentul logaritmic al perioadei $t$; $P_0$, $P_T$: primul și ultimul preț; $Y$: numărul de ani'),
        T('$\\mu$, $\\sigma$: the annual mean and volatility; $r_f$: the risk-free rate; SR: the Sharpe ratio; SE: its standard error', '$\\mu$, $\\sigma$: media și volatilitatea anuale; $r_f$: rata fără risc; SR: raportul Sharpe; SE: eroarea lui standard')],
    2: [T('$m_k = \\frac1n\\sum_t (r_t - \\bar r)^k$: the $k$-th central moment; $S$: skewness; $K$: excess kurtosis (0 for the Normal distribution)', '$m_k = \\frac1n\\sum_t (r_t - \\bar r)^k$: momentul centrat de ordin $k$; $S$: asimetria; $K$: excesul de boltire (0 pentru distribuția Normală)'),
        T('JB: Jarque--Bera, $\\chi^2(2)$ under normality; $\\nu$: degrees of freedom; $\\hat\\rho(h)$: the autocorrelation at lag $h$; $m$: the number of lags', 'JB: Jarque--Bera, $\\chi^2(2)$ în ipoteza de normalitate; $\\nu$: gradele de libertate; $\\hat\\rho(h)$: autocorelația la lagul $h$; $m$: numărul de laguri')],
    3: [T('$X_1, \\dots, X_n$: independent copies of $X$; $\\overset{d}{=}$: same distribution; $d_n$: a shift; $\\alpha \\in (0, 2]$: the stability index', '$X_1, \\dots, X_n$: copii independente ale lui $X$; $\\overset{d}{=}$: aceeași distribuție; $d_n$: o translație; $\\alpha \\in (0, 2]$: indicele de stabilitate'),
        T('$c$: a constant; $p$: the order of a moment; S0, S1: two parametrisations of the same family', '$c$: o constantă; $p$: ordinul unui moment; S0, S1: două parametrizări ale aceleiași familii')],
    4: [T('$a$, $b$: weights; $\\sigma_X$, $\\sigma_Y$: standard deviations; $p$: the estimated probability; $N$: the number of Monte Carlo draws', '$a$, $b$: ponderi; $\\sigma_X$, $\\sigma_Y$: abateri standard; $p$: probabilitatea estimată; $N$: numărul de extrageri Monte Carlo'),
        T('GBM (geometric Brownian motion): $S_t$: price at time $t$ (years); $\\mu$: drift; $\\sigma$: volatility; $W_t \\sim N(0, t)$: a Brownian motion', 'GBM (mișcarea browniană geometrică): $S_t$: prețul la momentul $t$ (ani); $\\mu$: tendința; $\\sigma$: volatilitatea; $W_t \\sim N(0, t)$: o mișcare browniană')],
    5: [T('$L_{(i)}$: the $i$-th largest loss; $k$: the number of tail observations; $\\xi = 1/\\alpha$: the shape (tail) parameter', '$L_{(i)}$: a $i$-a cea mai mare pierdere; $k$: numărul observațiilor din coadă; $\\xi = 1/\\alpha$: parametrul de formă al cozii'),
        T('GPD: $u$: the threshold; $\\beta$: the scale; $N_u$: the losses above $u$ out of $n$; in the VaR formula $\\alpha$ is the level (1\\%), not the tail index', 'GPD: $u$: pragul; $\\beta$: scala; $N_u$: pierderile peste $u$ din cele $n$; în formula VaR, $\\alpha$ este nivelul (1\\%), nu tail index-ul')],
    6: [T('$\\ell_0$, $\\ell_1$: the maximised log-likelihoods of the smaller and of the larger model; $k_0$, $k_1$, $k$: numbers of parameters; $n$: sample size', '$\\ell_0$, $\\ell_1$: log-verosimilitățile maxime ale modelului restrîns și ale celui extins; $k_0$, $k_1$, $k$: numerele de parametri; $n$: volumul eșantionului'),
        T('$F_n$: the empirical distribution function; $F$: the model; $D_n$: their largest vertical distance; smaller AIC or BIC is better', '$F_n$: funcția de repartiție empirică; $F$: modelul; $D_n$: cea mai mare distanță verticală dintre ele; un AIC sau BIC mai mic este mai bun')],
    7: [T('$\\rho(k)$: the autocorrelation at lag $k$; $q$: the horizon of the variance ratio; VR $= 1$ under a random walk', '$\\rho(k)$: autocorelația la lagul $k$; $q$: orizontul raportului varianțelor; VR $= 1$ pentru un mers aleator'),
        T('$T$: the number of returns; $\\hat\\theta$: a variance estimate robust to volatility clustering; $Z^* \\approx N(0, 1)$ under the null', '$T$: numărul de randamente; $\\hat\\theta$: o estimație a varianței robustă la volatility clustering; $Z^* \\approx N(0, 1)$ în ipoteza nulă')],
    8: [T('$\\lambda \\in (0, 1)$: the decay factor; $r_{t-1}$: yesterday\'s return; $H_t$, $L_t$: the high and the low of day $t$', '$\\lambda \\in (0, 1)$: factorul de descompunere; $r_{t-1}$: randamentul de ieri; $H_t$, $L_t$: maximul și minimul zilei $t$'),
        T('ARCH-LM: $R^2$ of the regression of $r_t^2$ on its $q$ lags, times $n$; a large value: volatility clustering', 'ARCH-LM: $R^2$ al regresiei lui $r_t^2$ pe $q$ laguri ale sale, înmulțit cu $n$; o valoare mare: volatility clustering')],
    9: [T('$\\omega$: the constant; $\\alpha$: the reaction to yesterday\'s shock $\\varepsilon_{t-1}$; $\\beta$: the memory; $\\bar\\sigma^2$: the long-run variance', '$\\omega$: constanta; $\\alpha$: reacția la șocul de ieri, $\\varepsilon_{t-1}$; $\\beta$: memoria; $\\bar\\sigma^2$: varianța pe termen lung'),
        T('$E_t$: the forecast made at day $t$; $h$: the horizon in days; $\\alpha + \\beta$: the speed at which the forecast returns to $\\bar\\sigma^2$', '$E_t$: prognoza făcută în ziua $t$; $h$: orizontul, în zile; $\\alpha + \\beta$: viteza cu care prognoza revine la $\\bar\\sigma^2$')],
    10: [T('$q_\\alpha$: the $\\alpha$-quantile of the return $X$; $z_\\alpha$: the standard Normal quantile; $\\varphi$: the standard Normal density', '$q_\\alpha$: cuantila de ordin $\\alpha$ a randamentului $X$; $z_\\alpha$: cuantila Normală standard; $\\varphi$: densitatea Normală standard'),
         T('$x$: the exceptions in $n$ days; $\\hat\\pi = x/n$: the observed rate; $LR_{uc}$ compares it with the target $\\alpha$', '$x$: depășirile în $n$ zile; $\\hat\\pi = x/n$: rata observată; $LR_{uc}$ o compară cu ținta $\\alpha$')],
    11: [T('$(R/S)_n$: the rescaled range (range of the cumulated deviations divided by the standard deviation) on blocks of $n$ days', '$(R/S)_n$: amplitudinea rescalată (amplitudinea abaterilor cumulate împărțită la abaterea standard) pe blocuri de $n$ zile'),
         T('$H$: the Hurst exponent; $d$: the fractional parameter; $r^{(h)}$: the $h$-day return; sd: the standard deviation', '$H$: exponentul Hurst; $d$: parametrul fracționar; $r^{(h)}$: randamentul pe $h$ zile; sd: abaterea standard')],
    12: [T('EL: expected loss; PD: probability of default; LGD: loss given default; EAD: exposure at default', 'EL: pierderea așteptată; PD: probabilitatea de nerambursare; LGD: pierderea în caz de nerambursare; EAD: expunerea la nerambursare'),
         T('$p$: the probability of default given $x$; $\\beta_j$: a coefficient; $s$: the score of a bad or a good borrower; AUC $= 0.5$: no discrimination, 1: perfect', '$p$: probabilitatea de nerambursare condiționată de $x$; $\\beta_j$: un coeficient; $s$: scorul unui debitor rău sau bun; AUC $= 0{,}5$: nicio discriminare, 1: discriminare perfectă')],
    13: [T('$y_0$: a new observation; $\\hat f$: the fitted model; $\\sigma^2$: the noise variance; $\\hat y$: the forecast', '$y_0$: o observație nouă; $\\hat f$: modelul estimat; $\\sigma^2$: varianța zgomotului; $\\hat y$: prognoza'),
         T('$\\tilde y$: the benchmark forecast; sums over the test sample; $R^2_{OOS} < 0$: worse than the benchmark', '$\\tilde y$: prognoza de referință; sumele parcurg eșantionul de test; $R^2_{OOS} < 0$: mai slab decît reperul')],
    14: [T('$\\sigma$: the standard deviation of daily returns; crypto trades 365 days a year', '$\\sigma$: abaterea standard a randamentelor zilnice; activele cripto se tranzacționează 365 de zile pe an'),
         T('$P_t$: the price of the stablecoin in USD; 1 basis point $= 0.01\\%$; $d_t < 0$: below the peg', '$P_t$: prețul stablecoin-ului în USD; 1 punct de bază $= 0{,}01\\%$; $d_t < 0$: sub paritate')],
    15: [T('$b$: the slope of the quantile regression of the system on the bank; $q_\\alpha$, $q_{50}$: the $\\alpha$-quantile and the median of the bank\'s return', '$b$: panta regresiei cuantilice a sistemului pe bancă; $q_\\alpha$, $q_{50}$: cuantila de ordin $\\alpha$ și mediana randamentului băncii'),
         T('MES: the bank\'s mean loss on the market\'s worst days; SRISK: the capital missing in a crisis', 'MES: pierderea medie a băncii în cele mai proaste zile ale pieței; SRISK: capitalul lipsă într-o criză')],
}


def review(n, title, formulas, fct, mistakes, size='small'):
    """Two recap slides per chapter: key formulas with their notation; empirical facts and common mistakes."""
    D.frame(T(f'Chapter {n}: {title[0]} (1/2)', f'Capitolul {n}: {title[1]} (1/2)'),
            items((FORM, formulas), (NOTN, NOTE[n])), size)
    D.frame(T(f'Chapter {n}: {title[0]} (2/2)', f'Capitolul {n}: {title[1]} (2/2)'),
            items((FACT, fct), (MIST, mistakes)), size)


# =============================================================================
# PROBLEMELE DE EXAMEN (rezolvări calculate aici)
# =============================================================================
z1, z25 = stats.norm.ppf(0.01), stats.norm.ppf(0.025)
phi25 = stats.norm.pdf(z25)
V.put('z1', -z1, 3)
V.put('phi25', phi25, 4)
V.put('chi1', stats.chi2.ppf(0.95, 1), 2)
V.put('chi2', stats.chi2.ppf(0.95, 2), 2)
V.put('chi4', stats.chi2.ppf(0.95, 4), 2)

# E1: an index 100 -> 160 -> 64 -> 96
P = [100, 160, 64, 96]
R = [P[i + 1] / P[i] - 1 for i in range(3)]
r = [math.log(P[i + 1] / P[i]) for i in range(3)]
for i in range(3):
    V.put(f'e1.R{i + 1}', 100 * R[i], 0)
    V.put(f'e1.r{i + 1}', r[i], 3)
V.put('e1.sum', sum(r), 3)
V.put('e1.tot', 100 * (P[3] / P[0] - 1), 0)
V.put('e1.am', 100 * sum(R) / 3, 2)
V.put('e1.cagr', 100 * ((P[3] / P[0]) ** (1 / 3) - 1), 2)
V.put('e1.mdd', 100 * (P[2] / P[1] - 1), 0)
V.put('e1.gain', 100 * (P[1] / P[2] - 1), 0)

# E2: JB, Student-t, Hill
n2, S2, K2 = 1000, -0.5, 6.0
V.put('e2.s', n2 / 6 * S2 ** 2, 2)
V.put('e2.k', n2 / 24 * K2 ** 2, 1)
V.put('e2.jb', n2 / 6 * (S2 ** 2 + K2 ** 2 / 4), 1)
V.put('e2.nu', 4 + 6 / K2, 0)
LOSS = [8.2, 6.5, 5.9, 5.1, 4.6]
lg = [math.log(x / LOSS[4]) for x in LOSS[:4]]
for i, x in enumerate(lg):
    V.put(f'e2.l{i + 1}', x, 3)
V.put('e2.ml', sum(lg) / 4, 4)
V.put('e2.a', 4 / sum(lg), 2)

# E3: portfolio, Monte Carlo, GBM
s1, s2, rho, w1 = 1.5, 2.5, 0.3, 0.6
vp = w1 ** 2 * s1 ** 2 + (1 - w1) ** 2 * s2 ** 2 + 2 * w1 * (1 - w1) * rho * s1 * s2
V.put('e3.t1', w1 ** 2 * s1 ** 2, 2)
V.put('e3.t2', (1 - w1) ** 2 * s2 ** 2, 2)
V.put('e3.t3', 2 * w1 * (1 - w1) * rho * s1 * s2, 2)
V.put('e3.vp', vp, 2)
V.put('e3.sp', math.sqrt(vp), 2)
V.put('e3.avg', w1 * s1 + (1 - w1) * s2, 2)
se = math.sqrt(0.01 * 0.99 / 10000)
V.put('e3.se', 100 * se, 3)
V.put('e3.lo', 100 * (0.01 - 1.96 * se), 2)
V.put('e3.hi', 100 * (0.01 + 1.96 * se), 2)
mu, sg = 0.08, 0.25
V.put('e3.mean', 100 * math.exp(mu), 2)
V.put('e3.m2', mu - sg ** 2 / 2, 4)
V.put('e3.med', 100 * math.exp(mu - sg ** 2 / 2), 2)
V.put('e3.zz', -(mu - sg ** 2 / 2) / sg, 3)
V.put('e3.ploss', 100 * stats.norm.cdf(-(mu - sg ** 2 / 2) / sg), 1)

# E4: Ljung-Box and VR(5)
T4, RHO = 2500, [0.06, 0.02, -0.01, 0.01]
lb = T4 * (T4 + 2) * sum(x ** 2 / (T4 - k) for k, x in enumerate(RHO, 1))
V.put('e4.sum', sum(x ** 2 for x in RHO), 4)
V.put('e4.lb', lb, 2)
vr = 1 + 2 * sum((1 - k / 5) * x for k, x in enumerate(RHO, 1))
V.put('e4.vr', vr, 3)
V.put('e4.inner', sum((1 - k / 5) * x for k, x in enumerate(RHO, 1)), 3)
se_iid = math.sqrt(2 * 9 * 4 / (3 * 5 * T4))
V.put('e4.se', se_iid, 4)
V.put('e4.z', (vr - 1) / se_iid, 2)
V.put('e4.zs', (vr - 1) / 0.065, 2)

# E5: GARCH(1,1), EWMA, Bitcoin annualisation
om, al, be, s2t, eps = 0.03, 0.09, 0.89, 1.8, -2.5
s2n = om + al * eps ** 2 + be * s2t
V.put('e5.a', al * eps ** 2, 4)
V.put('e5.b', be * s2t, 3)
V.put('e5.s2', s2n, 4)
V.put('e5.s', math.sqrt(s2n), 3)
V.put('e5.pers', al + be, 2)
V.put('e5.hl', math.log(0.5) / math.log(al + be), 1)
lr_var = om / (1 - al - be)
V.put('e5.lv', lr_var, 2)
V.put('e5.ann', math.sqrt(252 * lr_var), 2)
V.put('e5.p9', (al + be) ** 9, 4)
V.put('e5.h10', lr_var + (al + be) ** 9 * (s2n - lr_var), 3)
V.put('e5.ew', 0.94 * s2t + 0.06 * eps ** 2, 3)
V.put('e5.var', -z1 * math.sqrt(s2n), 2)
V.put('e5.b365', 3.5 * math.sqrt(365), 1)
V.put('e5.b252', 3.5 * math.sqrt(252), 1)
V.put('e5.sq365', math.sqrt(365), 2)
V.put('e5.sq252', math.sqrt(252), 2)

# E6: VaR, ES, Kupiec, traffic light
W, sig6 = 1_000_000, 1.6
var6 = -z1 * sig6
es6 = sig6 * phi25 / 0.025
V.put('e6.var', var6, 2)
V.raw('e6.varm', money(W * var6 / 100))
V.put('e6.esf', phi25 / 0.025, 3)
V.put('e6.es', es6, 2)
V.raw('e6.esm', money(W * es6 / 100))
V.put('e6.v10', var6 * math.sqrt(10), 2)
WORST = [-5.1, -4.4, -3.9, -3.6, -3.2, -3.0, -2.9]
V.put('e6.hses', -sum(WORST) / 7, 2)
V.put('e6.hssum', -sum(WORST), 1)
x6, n6 = 6, 250
ph6 = x6 / n6
l0 = (n6 - x6) * math.log(0.99) + x6 * math.log(0.01)
l1 = (n6 - x6) * math.log(1 - ph6) + x6 * math.log(ph6)
V.put('e6.l0', l0, 2)
V.put('e6.l1', l1, 2)
V.put('e6.lr', -2 * (l0 - l1), 2)
V.put('e6.p', stats.chi2.sf(-2 * (l0 - l1), 1), 3)
V.put('e6.pc', 100 * stats.binom.cdf(4, 250, 0.01), 1)

# E7: Hurst from two R/S points
rs1, rs2, n71, n72 = 4.1, 60.0, 20, 2000
H = math.log10(rs2 / rs1) / math.log10(n72 / n71)
V.put('e7.ratio', rs2 / rs1, 2)
V.put('e7.lr', math.log10(rs2 / rs1), 3)
V.put('e7.H', H, 3)
V.put('e7.d', H - 0.5, 3)
V.put('e7.hH', 10 ** H, 2)
V.put('e7.sq', math.sqrt(10), 2)
V.put('e7.vH', 2.5 * 10 ** H, 2)
V.put('e7.vs', 2.5 * math.sqrt(10), 2)

# E8: confusion matrix, expected loss, Delta-CoVaR
TP, FN, FP, TN = 60, 30, 42, 168
V.put('e8.acc', 100 * (TP + TN) / 300, 1)
V.put('e8.tpr', 100 * TP / (TP + FN), 1)
V.put('e8.fpr', 100 * FP / (FP + TN), 1)
V.put('e8.prec', 100 * TP / (TP + FP), 1)
V.put('e8.base', 100 * (FP + TN) / 300, 1)
V.put('e8.gini', 2 * 0.80 - 1, 2)
V.raw('e8.el', money(0.04 * 0.45 * 200_000))
a8, b8, q8, m8 = -2.1, 0.6, -5.0, 0.1
V.put('e8.covar', -(a8 + b8 * q8), 2)
V.put('e8.cmed', -(a8 + b8 * m8), 2)
V.put('e8.dcv', b8 * (m8 - q8), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), cols(items(
    (T('\\textbf{Question}: what does the whole course say about the returns of the BET, the S\\&P 500 and Bitcoin, and how is it examined?',
       '\\textbf{Întrebarea}: ce spune întregul curs despre randamentele BET, S\\&P 500 și Bitcoin și cum se evaluează aceste cunoștințe la examen?'),
     [T('one chapter that ties Chapters 0--15 together', 'un capitol care leagă Capitolele 0--15')]),
    (T('\\textbf{Route}', '\\textbf{Traseul}'),
     [T('Part I: the course map, two recap slides per chapter, the toolbox of tests and models', 'Partea I: harta cursului, cîte două slide-uri de recapitulare pentru fiecare capitol, trusa de teste și modele'),
      T('Part II: the exam (format, grading, eight solved problems) and the team project', 'Partea a II-a: examenul (format, criterii de notare, opt probleme rezolvate) și proiectul de echipă'),
      T('Part III: how AI could help in your own research', 'Partea a III-a: contribuția posibilă a AI în propria cercetare')])),
    ph('bvb2023', T('The Stock Exchange Palace, Bucharest', 'Palatul Bursei, București'), h='0.40\\textheight'), '0.58', '0.38'))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Place every chapter on the path from prices to risk numbers and name the question it answers', 'Situați fiecare capitol pe drumul de la prețuri la măsurile de risc și numiți întrebarea la care răspunde'),
    T('Recall the key formula, the key empirical fact and the most common mistake of each chapter', 'Reamintiți formula-cheie, faptul empiric principal și greșeala cea mai frecventă din fiecare capitol'),
    T('Choose the right test or model for a given question about a return series', 'Alegeți testul sau modelul potrivit pentru o întrebare dată despre o serie de randamente'),
    T('Solve exam-type problems step by step and interpret the result in three to five sentences', 'Rezolvați pas cu pas probleme de tip examen și interpretați rezultatul în trei pînă la cinci fraze'),
    T('Plan the team project, its oral defence and the declaration of AI use', 'Planificați proiectul de echipă, susținerea orală și declararea utilizării AI')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH; exercises with solutions: \\refBHL', 'Manual: \\refFHH; exerciții rezolvate: \\refBHL'),
     [T('time series and volatility: \\refTsay; market efficiency: \\refCLM; risk measures: \\refQRM', 'serii de timp și volatilitate: \\refTsay; eficiența pieței: \\refCLM; măsuri de risc: \\refQRM')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_16}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_16}'),
     [T('the course in one table: S\\&P 500, BET and Bitcoin on one common window', 'cursul într-un singur tabel: S\\&P 500, BET și Bitcoin pe aceeași fereastră de timp')]),
    T('Lecture notebook, the course in one notebook: \\href{\\colaburl{notebooks/EN/chapter16_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului, cursul într-un singur notebook: \\href{\\colaburl{notebooks/EN/chapter16_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}',
      'Curs video: \\quantinar{Statistics of Financial Markets}{https://quantinar.com/course/103/statistics-of-financial-markets}'),
    T('Self-assessment: the quiz of this chapter, 20 questions from all chapters', 'Autoevaluare: quiz-ul acestui capitol, 20 de întrebări din toate capitolele')))

# =============================================================================
# 1. HARTA CURSULUI
# =============================================================================
D.section('The Course Map', 'Harta cursului')

MAP = r"""\begin{center}
\resizebox{0.97\textwidth}{!}{%
\begin{tikzpicture}[
  box/.style={draw=MainBlue, rounded corners=2pt, fill=white, font=\scriptsize, align=center, minimum width=2.2cm, minimum height=0.85cm, text width=2.1cm},
  key/.style={box, draw=IDAred, line width=0.9pt},
  lab/.style={font=\scriptsize\bfseries, text=MainBlue, anchor=east, align=right},
  arr/.style={-{Stealth[length=2mm]}, MainBlue, line width=0.5pt},
  karr/.style={-{Stealth[length=2.4mm]}, IDAred, line width=1.1pt}]
\node[lab] at (-0.1, 0) {⟦Data and\\ distributions||Date și\\ distribuții⟧};
\node[box] (c0) at (1.3, 0) {0 ⟦Markets\\ and data||Piețe\\ și date⟧};
\node[key] (c1) at (3.8, 0) {1 ⟦Returns,\\ indicators||Randamente,\\ indicatori⟧};
\node[key] (c2) at (6.3, 0) {2 ⟦Distributions,\\ stylised facts||Distribuții,\\ fapte stilizate⟧};
\node[box] (c3) at (8.8, 0) {3 ⟦$\alpha$-stable\\ laws||Legi\\ $\alpha$-stabile⟧};
\node[key] (c5) at (11.3, 0) {5 ⟦Heavy tails,\\ EVT||Cozi groase,\\ EVT⟧};
\node[box] (c6) at (13.8, 0) {6 ⟦Model\\ selection||Selecția\\ modelului⟧};
\node[lab] at (-0.1, -1.7) {⟦Time, volatility\\ and risk||Timp, volatilitate\\ și risc⟧};
\node[box] (c4) at (1.3, -1.7) {4 ⟦Probability,\\ processes||Probabilitate,\\ procese⟧};
\node[box] (c7) at (3.8, -1.7) {7 ⟦Efficiency,\\ VR tests||Eficiență,\\ teste VR⟧};
\node[box] (c11) at (6.3, -1.7) {11 ⟦Long\\ memory||Memorie\\ lungă⟧};
\node[box] (c8) at (8.8, -1.7) {8 ⟦Volatility\\ estimators||Estimatori de\\ volatilitate⟧};
\node[key] (c9) at (11.3, -1.7) {9 ARCH, GARCH};
\node[key] (c10) at (13.8, -1.7) {10 ⟦VaR, ES,\\ backtesting||VaR, ES,\\ backtesting⟧};
\node[lab] at (-0.1, -3.4) {⟦Applications||Aplicații⟧};
\node[box] (c12) at (6.3, -3.4) {12 ⟦Scoring\\ models||Modele de\\ scoring⟧};
\node[box] (c13) at (8.8, -3.4) {13 ⟦Machine\\ learning||Învățare\\ automată⟧};
\node[box] (c14) at (11.3, -3.4) {14 ⟦Crypto\\ assets||Active\\ cripto⟧};
\node[box] (c15) at (13.8, -3.4) {15 ⟦Systemic\\ risk||Risc\\ sistemic⟧};
\draw[arr] (c0) -- (c1); \draw[karr] (c1) -- (c2); \draw[arr] (c2) -- (c3); \draw[arr] (c3) -- (c5); \draw[arr] (c5) -- (c6);
\draw[karr] (c2.south east) -- (c8.north west); \draw[karr] (c5) -- (c10.north west); \draw[arr] (c6) -- (c10);
\draw[arr] (c4) -- (c7); \draw[arr] (c7) -- (c11); \draw[arr] (c11) -- (c8); \draw[karr] (c8) -- (c9); \draw[karr] (c9) -- (c10);
\draw[arr] (c1) -- (c7); \draw[arr] (c4.south) -- (1.3, -2.6) -- (6.3, -2.6) -- (c12.north);
\draw[arr] (c10) -- (c15); \draw[arr] (c10.south west) -- (c14.north east); \draw[arr] (c9.south west) -- (c13.north east);
\end{tikzpicture}}
\end{center}"""

D.frame(T('The Course Map: Chapters 0--15', 'Harta cursului: Capitolele 0--15'), MAP + '\n\\vspace{-0.2cm}\n' + items(
    T('Red: the main thread, from returns and their distribution to a risk number we can test (Chapter 10)', 'Roșu: firul principal, de la randamente și distribuția lor la o măsură de risc pe care o putem testa (Capitolul 10)'),
    T('Row 2: dependence in time; the sign of returns is hard to predict (Chapters 7, 11), their size is predictable (Chapters 8--9)', 'Rîndul 2: dependența în timp; semnul randamentelor este greu de anticipat (Capitolele 7, 11), mărimea lor este previzibilă (Capitolele 8--9)'),
    T('Row 3: the same tools applied to credit, prediction, crypto assets and the banking system', 'Rîndul 3: aceleași instrumente aplicate creditului, predicției, activelor cripto și sistemului bancar')), 'footnotesize')

D.frame(T('The Main Thread: From Prices to Decisions', 'Firul principal: de la prețuri la decizii'), items(
    (T('\\textbf{Data} (Chapters 0--1): prices from one source, adjusted, on the right calendar', '\\textbf{Datele} (Capitolele 0--1): prețuri dintr-o singură sursă, ajustate, pe calendarul corect'),
     [T('returns: $r_t = \\ln(P_t/P_{t-1})$, in \\% for one day', 'randamente: $r_t = \\ln(P_t/P_{t-1})$, în \\% pe o zi')]),
    (T('\\textbf{Distribution} (Chapters 2--6): heavy tails, not the Normal distribution', '\\textbf{Distribuția} (Capitolele 2--6): cozi groase, nu distribuția Normală'),
     [T('Student-$t$, stable laws, EVT; the model is chosen with likelihood, AIC and tail tests', 'Student-$t$, legi stabile, EVT; modelul se alege cu verosimilitatea, AIC și teste pentru cozi')]),
    (T('\\textbf{Dependence in time} (Chapters 7--11): the sign is almost unpredictable, the size is predictable', '\\textbf{Dependența în timp} (Capitolele 7--11): semnul este aproape imprevizibil, mărimea este previzibilă'),
     [T('variance ratio and Hurst for the mean; EWMA and GARCH for the variance', 'raportul varianțelor și exponentul Hurst pentru medie; EWMA și GARCH pentru varianță')]),
    (T('\\textbf{Risk} (Chapter 10): VaR 1\\% and ES 2.5\\%, with a backtest', '\\textbf{Riscul} (Capitolul 10): VaR 1\\% și ES 2,5\\%, cu backtesting'),
     [T('the applications (Chapters 12--15) reuse the same chain: data, distribution, dependence, risk, evaluation', 'aplicațiile (Capitolele 12--15) refolosesc același lanț: date, distribuție, dependență, risc, evaluare')])))

chart(T('Three Series, One Window: S\\&P 500, BET, Bitcoin', 'Trei serii, aceeași fereastră: S\\&P 500, BET, Bitcoin'), 'sfm_ch16_three_series', 'SFM_ch16_course_summary', [
    T('100 invested on 2 January 2015 (log scale) and the drawdown $P_t/\\max_{s \\le t}P_s - 1$', '100 investiți la 2 ianuarie 2015 (scală logaritmică) și drawdown-ul $P_t/\\max_{s \\le t}P_s - 1$'),
    T('Maximum drawdown: S\\&P 500 $@{s.sp500.mdd}\\%$, BET $@{s.bet.mdd}\\%$ (both in March 2020), Bitcoin $@{s.btc.mdd}\\%$', 'Drawdown maxim: S\\&P 500 $@{s.sp500.mdd}\\%$, BET $@{s.bet.mdd}\\%$ (ambele în martie 2020), Bitcoin $@{s.btc.mdd}\\%$')],
    h='0.55\\textheight')

TROW = [
    (T('Observations (per year)', 'Observații (pe an)'), lambda k: f'@{{s.{k}.n}} (@{{s.{k}.q}})'),
    (T('Mean log return, \\% a year', 'Randament logaritmic mediu, \\% pe an'), lambda k: f'$@{{s.{k}.mu}}$'),
    (T('Volatility, \\% a year', 'Volatilitate, \\% pe an'), lambda k: f'$@{{s.{k}.vol}}$'),
    (T('Sharpe ratio ($r_f = 0$) $\\pm$ SE', 'Raportul Sharpe ($r_f = 0$) $\\pm$ SE'), lambda k: f'$@{{s.{k}.sh}} \\pm @{{s.{k}.shse}}$'),
    (T('Skewness; excess kurtosis', 'Asimetrie; excesul de boltire'), lambda k: f'$@{{s.{k}.skew}}$; $@{{s.{k}.k}}$'),
    (T('Hill tail index of losses ($k = 2.5\\%\\,n$)', 'Tail index-ul Hill al pierderilor ($k = 2{,}5\\%\\,n$)'), lambda k: f'$@{{s.{k}.hill}}$'),
    (T('Ljung--Box $Q(10)$: $r_t$; $r_t^2$', 'Ljung--Box $Q(10)$: $r_t$; $r_t^2$'), lambda k: f'$@{{s.{k}.lbr}}$; $@{{s.{k}.lbr2}}$'),
    (T('VR(5); robust $Z^*(5)$', 'VR(5); $Z^*(5)$ robust'), lambda k: f'$@{{s.{k}.vr}}$; $@{{s.{k}.zs}}$'),
    (T('GARCH(1,1)-$t$: $\\alpha + \\beta$', 'GARCH(1,1)-$t$: $\\alpha + \\beta$'), lambda k: f'$@{{s.{k}.pers}}$'),
    (T('VaR 1\\%: historical; Normal (\\%)', 'VaR 1\\%: istoric; Normal (\\%)'), lambda k: f'$@{{s.{k}.vhs}}$; $@{{s.{k}.vn}}$'),
    (T('ES 2.5\\%, historical (\\%)', 'ES 2,5\\%, istoric (\\%)'), lambda k: f'$@{{s.{k}.ehs}}$'),
    (T('Hurst (R/S): $r_t$; $|r_t|$', 'Hurst (R/S): $r_t$; $|r_t|$'), lambda k: f'$@{{s.{k}.hr}}$; $@{{s.{k}.ha}}$'),
]
D.frame(T('The Course in One Table', 'Cursul într-un singur tabel'), table(
    'lrrr', T('\\textbf{Daily log returns}', '\\textbf{Randamente logaritmice zilnice}') + ' & S\\&P 500 & BET & Bitcoin',
    [lab + ' & ' + ' & '.join(f(k) for k in ('sp500', 'bet', 'btc')) for lab, f in TROW], size='scriptsize') + items(
    T('Window: @{s.first} -- @{s.last}; each series on its own calendar (Bitcoin: 365 days a year); critical value of $\\chi^2(10)$ at 5\\%: $@{s.crit}$',
      'Fereastra: @{s.first} -- @{s.last}; fiecare serie pe calendarul ei (Bitcoin: 365 de zile pe an); valoarea critică $\\chi^2(10)$ la 5\\%: $@{s.crit}$')) + ql('SFM_ch16_course_summary'), 'footnotesize')

D.frame(T('Interpreting the Table', 'Interpretarea tabelului'), items(
    (T('\\textbf{Return and risk}: Bitcoin has about four times the volatility of the two indices', '\\textbf{Randament și risc}: Bitcoin are o volatilitate de circa patru ori mai mare decît a celor doi indici'),
     [T('the Sharpe ratios are close once their standard errors are taken into account', 'rapoartele Sharpe sînt apropiate, dacă ținem seama de erorile lor standard')]),
    (T('\\textbf{Distribution}: negative skewness, excess kurtosis above 10, Hill indices between 2 and 3', '\\textbf{Distribuția}: asimetrie negativă, exces de boltire peste 10, tail index-uri Hill între 2 și 3'),
     [T('the Normal VaR 1\\% is too low by @{s.sp500.gap}\\% (S\\&P 500), @{s.bet.gap}\\% (BET) and @{s.btc.gap}\\% (Bitcoin)', 'VaR 1\\% Normal este prea mic cu @{s.sp500.gap}\\% (S\\&P 500), @{s.bet.gap}\\% (BET) și @{s.btc.gap}\\% (Bitcoin)')]),
    (T('\\textbf{Dependence}: Ljung--Box on $r_t^2$ is many times larger than on $r_t$', '\\textbf{Dependența}: statistica Ljung--Box pe $r_t^2$ este de multe ori mai mare decît pe $r_t$'),
     [T('the robust VR test rejects no random walk at 5\\% here; $H$ of $|r_t|$ is far above 0.5', 'testul VR robust nu respinge aici mersul aleator la 5\\%; $H$ pentru $|r_t|$ este mult peste 0,5'),
      T('volatility is persistent: $\\alpha + \\beta$ close to 1 (Bitcoin: an IGARCH, $\\alpha + \\beta = 1$)', 'volatilitatea este persistentă: $\\alpha + \\beta$ aproape de 1 (Bitcoin: un IGARCH, $\\alpha + \\beta = 1$)')])))

chart(T('Stylised Facts in One Chart', 'Faptele stilizate într-un singur grafic'), 'sfm_ch16_stylised', 'SFM_ch16_course_summary', [
    T('Returns: autocorrelations small and mostly inside the band $\\pm 1.96/\\sqrt{T}$; absolute returns: positive for 50 days', 'Randamentele: autocorelații mici și în mare parte în interiorul benzii $\\pm 1{,}96/\\sqrt{T}$; randamentele absolute: pozitive timp de 50 de zile'),
    T('Uncorrelated is not independent: $\\hat\\rho_1(|r|) = @{s.sp500.a1}$ (S\\&P 500), $@{s.bet.a1}$ (BET), $@{s.btc.a1}$ (Bitcoin)', 'Necorelat nu înseamnă independent: $\\hat\\rho_1(|r|) = @{s.sp500.a1}$ (S\\&P 500), $@{s.bet.a1}$ (BET), $@{s.btc.a1}$ (Bitcoin)')],
    h='0.52\\textheight')

D.recap(('The Course Map', 'harta cursului'), [
    T('One chain in every chapter: data, distribution, dependence in time, risk, evaluation', 'Un singur lanț în fiecare capitol: date, distribuție, dependență în timp, risc, evaluare'),
    T('Three series carry the course: the S\\&P 500 (a developed market), the BET (an emerging market), Bitcoin (a new asset class)', 'Trei serii străbat cursul: S\\&P 500 (o piață dezvoltată), BET (o piață emergentă), Bitcoin (o clasă nouă de active)'),
    T('All three: heavy tails, almost uncorrelated returns, strongly dependent volatility', 'Toate trei: cozi groase, randamente aproape necorelate, volatilitate puternic dependentă în timp')])

# =============================================================================
# 2. CAPITOLELE 0--3
# =============================================================================
D.section('Chapters 0--3: Data, Returns and Distributions', 'Capitolele 0--3: date, randamente și distribuții')

review(0, ('Introduction', 'introducere'),
       [T('annualised volatility $\\sigma_{\\text{ann}} = \\sqrt{A}\\,\\sigma$, $A$ = observations per year (252 for shares, 365 for crypto)', 'volatilitatea anualizată $\\sigma_{\\text{an}} = \\sqrt{A}\\,\\sigma$, $A$ = observații pe an (252 la acțiuni, 365 la cripto)'),
        T('drawdown $D_t = P_t/\\max_{s \\le t}P_s - 1$; maximum drawdown MDD $= \\min_t D_t$', 'drawdown $D_t = P_t/\\max_{s \\le t}P_s - 1$; drawdown maxim MDD $= \\min_t D_t$')],
       [T('since 2015: volatility @{f0.spvol}\\% a year for the S\\&P 500, @{f0.btcvol}\\% for Bitcoin', 'din 2015: volatilitate de @{f0.spvol}\\% pe an pentru S\\&P 500 și de @{f0.btcvol}\\% pentru Bitcoin'),
        T('BET: $@{f0.betmdd}\\%$ from @{f0.betpeak} to @{f0.bettrough}; back to the peak only on @{f0.betrec}', 'BET: $@{f0.betmdd}\\%$ între @{f0.betpeak} și @{f0.bettrough}; revenire la vîrf abia la @{f0.betrec}')],
       [T('prices from several sources mixed, or unadjusted prices for shares with dividends and splits', 'prețuri din mai multe surse amestecate sau prețuri neajustate pentru acțiuni cu dividende și split-uri'),
        T('a referenced number or paper produced by AI and never checked', 'o cifră sau o lucrare de referință obținută cu AI și neverificată')])

review(1, ('Data, Returns and Indicators', 'date, randamente și indicatori'),
       [T('$R_t = P_t/P_{t-1} - 1$, $r_t = \\ln(1 + R_t)$; log returns add over time, simple returns add across assets', '$R_t = P_t/P_{t-1} - 1$, $r_t = \\ln(1 + R_t)$; randamentele logaritmice se adună în timp, cele simple între active'),
        T('CAGR $= (P_T/P_0)^{1/Y} - 1$; Sharpe $= (\\mu - r_f)/\\sigma$, SE $\\approx \\sqrt{(1 + \\mathrm{SR}^2/2)/Y}$; volatility drag $\\approx \\sigma^2/2$', 'CAGR $= (P_T/P_0)^{1/Y} - 1$; Sharpe $= (\\mu - r_f)/\\sigma$, SE $\\approx \\sqrt{(1 + \\mathrm{SR}^2/2)/Y}$; volatility drag $\\approx \\sigma^2/2$')],
       [T('BET since @{f1.y0}: CAGR @{f1.cagr}\\%, volatility @{f1.vol}\\%, Sharpe @{f1.sh} with 95\\% CI [@{f1.shlo}, @{f1.shhi}]', 'BET din @{f1.y0}: CAGR @{f1.cagr}\\%, volatilitate @{f1.vol}\\%, Sharpe @{f1.sh}, cu CI 95\\% [@{f1.shlo}, @{f1.shhi}]'),
        T('Bitcoin: arithmetic mean @{f1.btcar}\\% a year, CAGR only @{f1.btccagr}\\%: the volatility drag', 'Bitcoin: media aritmetică @{f1.btcar}\\% pe an, CAGR doar @{f1.btccagr}\\%: efectul volatility drag')],
       [T('adding simple returns over time; averaging returns instead of computing the CAGR', 'adunarea randamentelor simple în timp; media randamentelor în locul CAGR'),
        T('annualising with 252 for Bitcoin; reporting a Sharpe ratio without its standard error', 'anualizarea cu 252 pentru Bitcoin; raportarea raportului Sharpe fără eroarea lui standard')])

review(2, ('Classical Distributions and Stylised Facts', 'distribuții clasice și fapte stilizate'),
       [T('$S = m_3/m_2^{3/2}$, $K = m_4/m_2^2 - 3$; $\\text{JB} = \\frac{n}{6}(S^2 + K^2/4) \\sim \\chi^2(2)$ \\refJB', '$S = m_3/m_2^{3/2}$, $K = m_4/m_2^2 - 3$; $\\text{JB} = \\frac{n}{6}(S^2 + K^2/4) \\sim \\chi^2(2)$ \\refJB'),
        T('Student-$t(\\nu)$: excess kurtosis $6/(\\nu - 4)$ for $\\nu > 4$; Ljung--Box $Q(m) = n(n+2)\\sum_h \\hat\\rho(h)^2/(n - h)$', 'Student-$t(\\nu)$: excesul de boltire $6/(\\nu - 4)$ pentru $\\nu > 4$; Ljung--Box $Q(m) = n(n+2)\\sum_h \\hat\\rho(h)^2/(n - h)$')],
       [T('BET since @{f2.y0}: $S = @{f2.skew}$, $K = @{f2.k}$, JB $=$ @{f2.jb}; fitted Student-$t$: $\\hat\\nu = @{f2.nu}$', 'BET din @{f2.y0}: $S = @{f2.skew}$, $K = @{f2.k}$, JB $=$ @{f2.jb}; Student-$t$ estimată: $\\hat\\nu = @{f2.nu}$'),
        T('$\\hat\\rho_1(r) = @{f2.rho1}$ but $\\hat\\rho_1(|r|) = @{f2.rho1a}$: the six stylised facts of \\refCont', '$\\hat\\rho_1(r) = @{f2.rho1}$, dar $\\hat\\rho_1(|r|) = @{f2.rho1a}$: cele șase fapte stilizate din \\refCont')],
       [T('reading the kurtosis $K + 3$ as the excess kurtosis (the Normal distribution has kurtosis 3, excess 0)', 'confuzia dintre coeficientul de boltire $K + 3$ și excesul de boltire (distribuția Normală are boltirea 3 și excesul 0)'),
        T('concluding independence from a non-significant ACF of $r_t$', 'concluzia de independență trasă dintr-un ACF nesemnificativ al lui $r_t$')])

review(3, ('$\\alpha$-Stable Distributions', 'distribuții $\\alpha$-stabile'),
       [T('$X_1 + \\dots + X_n \\overset{d}{=} n^{1/\\alpha}X + d_n$; tails $P(X > x) \\sim c\\,x^{-\\alpha}$; $E|X|^p < \\infty$ only for $p < \\alpha$', '$X_1 + \\dots + X_n \\overset{d}{=} n^{1/\\alpha}X + d_n$; cozi $P(X > x) \\sim c\\,x^{-\\alpha}$; $E|X|^p < \\infty$ doar pentru $p < \\alpha$'),
        T('four parameters: $\\alpha$ (tails), $\\beta$ (skewness), $\\gamma$ (scale), $\\delta$ (location), in S0 or S1 \\refNolan', 'patru parametri: $\\alpha$ (cozi), $\\beta$ (asimetrie), $\\gamma$ (scală), $\\delta$ (poziție), în S0 sau S1 \\refNolan')],
       [T('maximum likelihood since @{f3.y0}: $\\hat\\alpha = @{f3.a.bet}$ (BET), $@{f3.a.sp500}$ (S\\&P 500), $@{f3.a.btc}$ (Bitcoin), as \\refMandelbrot found for cotton', 'verosimilitate maximă din @{f3.y0}: $\\hat\\alpha = @{f3.a.bet}$ (BET), $@{f3.a.sp500}$ (S\\&P 500), $@{f3.a.btc}$ (Bitcoin), ca la bumbacul lui \\refMandelbrot'),
        T('but the sample variance settles down and $\\hat\\alpha$ of the BET stays near @{f3.am} on monthly data: heavy tails with finite variance', 'dar varianța de selecție se stabilizează, iar $\\hat\\alpha$ pentru BET rămîne în jur de @{f3.am} pe date lunare: cozi groase cu varianță finită')],
       [T('using the standard deviation as a scale when $\\alpha < 2$ (the variance does not exist)', 'folosirea abaterii standard ca scală cînd $\\alpha < 2$ (varianța nu există)'),
        T('comparing $\\delta$ from two programs that use different parametrisations', 'compararea lui $\\delta$ din două programe care folosesc parametrizări diferite')])

D.recap(('Chapters 0--3', 'Capitolele 0--3'), [
    T('Clean data and log returns first; annualise with the actual frequency', 'Întîi date curate și randamente logaritmice; anualizarea se face cu frecvența reală'),
    T('Returns are not Normal: negative skewness, large excess kurtosis, JB rejects everywhere', 'Randamentele nu sînt Normale: asimetrie negativă, exces de boltire mare, JB respinge peste tot'),
    T('Stable laws fit the body well, but the variance of returns is finite', 'Legile stabile descriu bine corpul distribuției, dar varianța randamentelor este finită')])

# =============================================================================
# 3. CAPITOLELE 4--7
# =============================================================================
D.section('Chapters 4--7: Probability, Tails, Model Choice, Efficiency', 'Capitolele 4--7: probabilitate, cozi, alegerea modelului, eficiență')

review(4, ('Probability', 'probabilitate'),
       [T('$\\mathrm{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\mathrm{Cov}(X, Y)$; Monte Carlo error $\\sqrt{p(1 - p)/N}$', '$\\mathrm{Var}(aX + bY) = a^2\\sigma_X^2 + b^2\\sigma_Y^2 + 2ab\\,\\mathrm{Cov}(X, Y)$; eroarea Monte Carlo $\\sqrt{p(1 - p)/N}$'),
        T('GBM: $S_t = S_0\\exp((\\mu - \\sigma^2/2)t + \\sigma W_t)$, mean $S_0e^{\\mu t}$, median $S_0e^{(\\mu - \\sigma^2/2)t}$', 'GBM: $S_t = S_0\\exp((\\mu - \\sigma^2/2)t + \\sigma W_t)$, media $S_0e^{\\mu t}$, mediana $S_0e^{(\\mu - \\sigma^2/2)t}$')],
       [T('S\\&P 500 since @{f4.y0}: a day below $-5\\%$ has frequency @{f4.p5e}\\%; the Normal distribution gives @{f4.p5n}\\%, @{f4.ratio} times less', 'S\\&P 500 din @{f4.y0}: o zi sub $-5\\%$ are frecvența @{f4.p5e}\\%; distribuția Normală dă @{f4.p5n}\\%, de @{f4.ratio} de ori mai puțin'),
        T('mean return @{f4.mean}\\% a year with SE @{f4.se}\\%: about @{f4.yrs} years of data would be needed for $t = 3$', 'randament mediu @{f4.mean}\\% pe an, cu SE @{f4.se}\\%: ar fi nevoie de circa @{f4.yrs} de ani de date pentru $t = 3$')],
       [T('uncorrelated taken for independent; the mean of $S_t$ taken for its median', 'variabile necorelate considerate independente; media lui $S_t$ confundată cu mediana'),
        T('a Monte Carlo estimate without its standard error', 'o estimare Monte Carlo fără eroarea ei standard')])

review(5, ('Heavy Tails and Extreme Value Theory', 'cozi groase și teoria valorilor extreme'),
       [T('Hill: $\\hat\\alpha_k = [\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})]^{-1}$, SE $\\approx \\hat\\alpha/\\sqrt{k}$ \\refHill; $\\xi = 1/\\alpha$', 'Hill: $\\hat\\alpha_k = [\\frac1k\\sum_{i=1}^k \\ln(L_{(i)}/L_{(k+1)})]^{-1}$, SE $\\approx \\hat\\alpha/\\sqrt{k}$ \\refHill; $\\xi = 1/\\alpha$'),
        T('GPD above $u$: VaR $= u + \\frac{\\beta}{\\xi}[(n\\alpha/N_u)^{-\\xi} - 1]$; GEV for block maxima', 'GPD peste $u$: VaR $= u + \\frac{\\beta}{\\xi}[(n\\alpha/N_u)^{-\\xi} - 1]$; GEV pentru block maxima')],
       [T('tail index of the losses: BET @{f5.a.bet} (95\\% CI [@{f5.lo}, @{f5.hi}], since @{f5.y0.bet}), S\\&P 500 @{f5.a.sp500}, Bitcoin @{f5.a.btc}', 'tail index-ul pierderilor: BET @{f5.a.bet} (CI 95\\% [@{f5.lo}, @{f5.hi}], din @{f5.y0.bet}), S\\&P 500 @{f5.a.sp500}, Bitcoin @{f5.a.btc}'),
        T('BET, VaR 0.1\\%: EVT @{f5.v01e}\\%, Normal only @{f5.v01n}\\%', 'BET, VaR 0,1\\%: EVT @{f5.v01e}\\%, distribuția Normală doar @{f5.v01n}\\%')],
       [T('choosing $k$ without looking at the Hill plot; reporting $\\hat\\alpha$ without an interval', 'alegerea lui $k$ fără graficul Hill; raportarea lui $\\hat\\alpha$ fără interval'),
        T('applying the Hill estimator to returns instead of losses $L = -r$', 'aplicarea estimatorului Hill pe randamente în loc de pierderi $L = -r$')])

review(6, ('Model Selection and Risk Management', 'selecția modelului și managementul riscului'),
       [T('$LR = 2(\\ell_1 - \\ell_0) \\approx \\chi^2(k_1 - k_0)$ for nested models; AIC $= -2\\ell + 2k$, BIC $= -2\\ell + k\\ln n$', '$LR = 2(\\ell_1 - \\ell_0) \\approx \\chi^2(k_1 - k_0)$ pentru modele imbricate; AIC $= -2\\ell + 2k$, BIC $= -2\\ell + k\\ln n$'),
        T('KS: $D_n = \\sup_x|F_n(x) - F(x)|$; Anderson--Darling weights the tails', 'KS: $D_n = \\sup_x|F_n(x) - F(x)|$; Anderson--Darling pune accentul pe cozi')],
       [T('BET since @{f6.y0}, Anderson--Darling: Normal @{f6.adn}, Student-$t$ @{f6.adt}, NIG @{f6.adnig}', 'BET din @{f6.y0}, Anderson--Darling: Normală @{f6.adn}, Student-$t$ @{f6.adt}, NIG @{f6.adnig}'),
        T('VaR 1\\% of the BET: Normal @{f6.vn}\\%, heavy-tailed models @{f6.vlo}--@{f6.vhi}\\%', 'VaR 1\\% pentru BET: distribuția Normală @{f6.vn}\\%, modelele cu cozi groase @{f6.vlo}--@{f6.vhi}\\%')],
       [T('the LR test at the boundary (e.g.\\ $\\nu = \\infty$) without the corrected critical value', 'testul LR la frontieră (de exemplu $\\nu = \\infty$) fără valoarea critică corectată'),
        T('rejecting every model with a large $n$ and stopping there, instead of ranking the models', 'respingerea tuturor modelelor la $n$ mare, fără ordonarea lor')])

review(7, ('Efficient Markets, Random Walk and VR Tests', 'piețe eficiente, mers aleator și teste VR'),
       [T('weak-form efficiency: past prices do not predict excess returns \\refFama; null: martingale differences', 'eficiența în formă slabă: prețurile trecute nu prezic randamente în exces \\refFama; ipoteza nulă: diferențe de martingal'),
        T('$\\mathrm{VR}(q) = 1 + 2\\sum_{k=1}^{q-1}(1 - k/q)\\rho(k)$; robust $Z^*(q) = (\\widehat{\\mathrm{VR}} - 1)/\\sqrt{\\hat\\theta/T}$ \\refLM', '$\\mathrm{VR}(q) = 1 + 2\\sum_{k=1}^{q-1}(1 - k/q)\\rho(k)$; statistica robustă $Z^*(q) = (\\widehat{\\mathrm{VR}} - 1)/\\sqrt{\\hat\\theta/T}$ \\refLM')],
       [T('BET since @{f7.y0.bet}: VR(5) $= @{f7.vr.bet}$, $Z^* = @{f7.zs.bet}$ (momentum)', 'BET din @{f7.y0.bet}: VR(5) $= @{f7.vr.bet}$, $Z^* = @{f7.zs.bet}$ (momentum)'),
        T('S\\&P 500 since @{f7.y0.sp500}: VR(5) $= @{f7.vr.sp500}$, $Z^* = @{f7.zs.sp500}$ (mean reversion); Bitcoin: $Z^* = @{f7.zs.btc}$', 'S\\&P 500 din @{f7.y0.sp500}: VR(5) $= @{f7.vr.sp500}$, $Z^* = @{f7.zs.sp500}$ (revenire la medie); Bitcoin: $Z^* = @{f7.zs.btc}$')],
       [T('the i.i.d. statistic $Z(q)$ on returns with volatility clustering (it rejects too often)', 'statistica i.i.d.\\ $Z(q)$ pe randamente cu volatility clustering (respinge prea des)'),
        T('a unit root in prices read as proof of efficiency; predictability read as profit', 'rădăcina unitară a prețurilor considerată dovadă de eficiență; predictibilitatea considerată profit')])

D.recap(('Chapters 4--7', 'Capitolele 4--7'), [
    T('Probability gives the tools: covariance, conditional variance, Monte Carlo with its error', 'Probabilitățile dau instrumentele: covarianța, varianța condiționată, Monte Carlo cu eroarea lui'),
    T('Tails are power laws with $\\alpha$ between 2 and 4; EVT gives VaR far in the tail', 'Cozile urmează legi de putere cu $\\alpha$ între 2 și 4; EVT dă VaR departe în coadă'),
    T('Choose the model for its purpose and check it out of sample; test efficiency with robust statistics', 'Modelul se alege după scop și se verifică în afara eșantionului; eficiența se testează cu statistici robuste')])

# =============================================================================
# 4. CAPITOLELE 8--11
# =============================================================================
D.section('Chapters 8--11: Volatility, Risk and Memory', 'Capitolele 8--11: volatilitate, risc și memorie')

review(8, ('Volatility Estimators', 'estimatori de volatilitate'),
       [T('EWMA: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$, half-life $\\ln 0.5/\\ln\\lambda$ \\refRM', 'EWMA: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$, timp de înjumătățire $\\ln 0{,}5/\\ln\\lambda$ \\refRM'),
        T('Parkinson: $(\\ln H_t - \\ln L_t)^2/(4\\ln 2)$; ARCH-LM $= nR^2 \\sim \\chi^2(q)$', 'Parkinson: $(\\ln H_t - \\ln L_t)^2/(4\\ln 2)$; ARCH-LM $= nR^2 \\sim \\chi^2(q)$')],
       [T('S\\&P 500: Ljung--Box $Q(10)$ is @{f8.lbr} on $r_t$ and @{f8.lbr2} on $r_t^2$ (critical value @{f8.crit})', 'S\\&P 500: Ljung--Box $Q(10)$ este @{f8.lbr} pe $r_t$ și @{f8.lbr2} pe $r_t^2$ (valoarea critică @{f8.crit})'),
        T('EWMA with $\\lambda = 0.94$: half-life of @{f8.hl} days; no ghost effect', 'EWMA cu $\\lambda = 0{,}94$: timp de înjumătățire de @{f8.hl} zile; fără ghost effect')],
       [T('a rolling window of fixed length: one extreme day enters and leaves abruptly (ghost effect)', 'fereastra mobilă de lungime fixă: o zi extremă intră și iese brusc din calcul (ghost effect)'),
        T('range estimators on prices with overnight gaps, without the Yang--Zhang correction', 'estimatori de amplitudine pe prețuri cu salturi overnight, fără corecția Yang--Zhang')])

review(9, ('ARCH and GARCH Models', 'modele ARCH și GARCH'),
       [T('GARCH(1,1): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ \\refEngle, \\refBoll', 'GARCH(1,1): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ \\refEngle, \\refBoll'),
        T('forecast $E_t\\sigma_{t+h}^2 = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; half-life $\\ln 0.5/\\ln(\\alpha + \\beta)$', 'prognoza $E_t\\sigma_{t+h}^2 = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; timp de înjumătățire $\\ln 0{,}5/\\ln(\\alpha + \\beta)$')],
       [T('GARCH(1,1)-$t$ since @{f9.y0}: $\\alpha + \\beta = @{f9.p.sp500}$ (S\\&P 500, half-life @{f9.hl.sp500} days), $@{f9.p.bet}$ (BET, @{f9.hl.bet} days)', 'GARCH(1,1)-$t$ din @{f9.y0}: $\\alpha + \\beta = @{f9.p.sp500}$ (S\\&P 500, timp de înjumătățire @{f9.hl.sp500} de zile), $@{f9.p.bet}$ (BET, @{f9.hl.bet} de zile)'),
        T('Bitcoin: $\\alpha + \\beta = @{f9.p.btc}$, an IGARCH; equity indices need asymmetry (GJR, EGARCH)', 'Bitcoin: $\\alpha + \\beta = @{f9.p.btc}$, un IGARCH; indicii de acțiuni au nevoie de asimetrie (GJR, EGARCH)')],
       [T('Normal innovations for returns with heavy tails; non-robust standard errors', 'inovații Normale pentru randamente cu cozi groase; erori standard nerobuste'),
        T('a long-run variance $\\omega/(1 - \\alpha - \\beta)$ reported when $\\alpha + \\beta \\ge 1$', 'raportarea varianței pe termen lung $\\omega/(1 - \\alpha - \\beta)$ cînd $\\alpha + \\beta \\ge 1$')])

review(10, ('VaR, ES and Backtesting', 'VaR, ES și backtesting'),
       [T('$\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$, $\\mathrm{ES}_\\alpha = -\\frac1\\alpha\\int_0^\\alpha q_u\\,du$; Normal: $-(\\mu + \\sigma z_\\alpha)$, $-\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$', '$\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$, $\\mathrm{ES}_\\alpha = -\\frac1\\alpha\\int_0^\\alpha q_u\\,du$; Normal: $-(\\mu + \\sigma z_\\alpha)$, $-\\mu + \\sigma\\varphi(z_\\alpha)/\\alpha$'),
        T('Kupiec: $LR_{uc} = -2\\ln[(1-\\alpha)^{n-x}\\alpha^x/((1-\\hat\\pi)^{n-x}\\hat\\pi^x)] \\sim \\chi^2(1)$ \\refKupiec; ES is coherent \\refArtzner', 'Kupiec: $LR_{uc} = -2\\ln[(1-\\alpha)^{n-x}\\alpha^x/((1-\\hat\\pi)^{n-x}\\hat\\pi^x)] \\sim \\chi^2(1)$ \\refKupiec; ES este coerent \\refArtzner')],
       [T('S\\&P 500, VaR 1\\%: historical @{f10.hs}\\%, Normal @{f10.n}\\%, i.e.\\ @{f10.gap}\\% too low', 'S\\&P 500, VaR 1\\%: istoric @{f10.hs}\\%, Normal @{f10.n}\\%, adică prea mic cu @{f10.gap}\\%'),
        T('exception rates since @{f10.bt0}: Normal @{f10.r.n}\\%, HS @{f10.r.hs}\\%, GARCH-$t$ @{f10.r.g}\\%, FHS @{f10.r.f}\\% (target 1\\%)', 'rata depășirilor din @{f10.bt0}: Normal @{f10.r.n}\\%, HS @{f10.r.hs}\\%, GARCH-$t$ @{f10.r.g}\\%, FHS @{f10.r.f}\\% (ținta 1\\%)')],
       [T('a negative VaR or ``VaR 99\\%\'\': the course writes VaR 1\\%, a positive loss', 'un VaR negativ sau „VaR 99\\%”: în curs scriem VaR 1\\%, o pierdere pozitivă'),
        T('$\\sqrt{10}$ scaling as if returns were i.i.d.\\ Normal; checking only the number of exceptions, not their clustering', 'scalarea cu $\\sqrt{10}$ ca pentru randamente Normale i.i.d.; verificarea doar a numărului de depășiri, nu și a grupării lor')])

chart(T('VaR 1\\% Forecasts of the BET: Historical Simulation and EWMA-Normal', 'Prognoze VaR 1\\% pentru BET: simulare istorică și EWMA-Normal'), 'sfm_ch16_backtest', 'SFM_ch16_course_summary', [
    T('Since @{bt.bet.first}, @{bt.bet.n} days, @{bt.bet.exp} expected exceptions: HS @{bt.bet.hs.x} (@{bt.bet.hs.rate}\\%, Kupiec $p = @{bt.bet.hs.p}$), EWMA-Normal @{bt.bet.ew.x} (@{bt.bet.ew.rate}\\%, $p$ @{bt.bet.ew.p})',
      'Din @{bt.bet.first}, @{bt.bet.n} de zile, @{bt.bet.exp} depășiri așteptate: HS @{bt.bet.hs.x} (@{bt.bet.hs.rate}\\%, Kupiec $p = @{bt.bet.hs.p}$), EWMA-Normal @{bt.bet.ew.x} (@{bt.bet.ew.rate}\\%, $p$ @{bt.bet.ew.p})'),
    T('EWMA follows the storms, but the Normal quantile is too small; HS has the right rate, but its exceptions come in clusters (at most @{bt.bet.hs.max} in 250 days)', 'EWMA urmărește furtunile, dar cuantila Normală este prea mică; HS are rata corectă, dar depășirile lui vin grupat (cel mult @{bt.bet.hs.max} în 250 de zile)')],
    h='0.56\\textheight')

review(11, ('Fractal Markets Hypothesis and Long Memory', 'ipoteza piețelor fractale și memoria lungă'),
       [T('$(R/S)_n \\propto n^H$ \\refHurst; $H = 0.5$: no memory, $H > 0.5$: persistence; $d = H - 0.5$', '$(R/S)_n \\propto n^H$ \\refHurst; $H = 0{,}5$: fără memorie, $H > 0{,}5$: persistență; $d = H - 0{,}5$'),
        T('risk scaling: $\\mathrm{sd}(r^{(h)}) = h^H\\mathrm{sd}(r)$; fractal markets need many investment horizons \\refPeters', 'scalarea riscului: $\\mathrm{sd}(r^{(h)}) = h^H\\mathrm{sd}(r)$; piețele fractale au nevoie de multe orizonturi investiționale \\refPeters')],
       [T('BET: $H$ (R/S) $= @{f11.hr}$ for $r_t$, at the edge of the Monte Carlo band, and the Lo test rejects: weak persistence; $@{f11.ha}$ for $|r_t|$', 'BET: $H$ (R/S) $= @{f11.hr}$ pentru $r_t$, la limita benzii Monte Carlo, iar testul Lo respinge: persistență slabă; $@{f11.ha}$ pentru $|r_t|$'),
        T('S\\&P 500: $@{f11.hrs}$ for returns, $@{f11.has}$ for absolute returns: long memory is in the volatility', 'S\\&P 500: $@{f11.hrs}$ pentru randamente, $@{f11.has}$ pentru randamente absolute: memoria lungă se află în volatilitate')],
       [T('comparing $\\hat H$ with 0.5 instead of a Monte Carlo band for the same $N$ (R/S is biased upwards)', 'compararea lui $\\hat H$ cu 0,5 în locul unei benzi Monte Carlo pentru același $N$ (R/S este deplasat în sus)'),
        T('structural breaks or GARCH taken for long memory', 'rupturi structurale sau efecte GARCH considerate memorie lungă')])

D.recap(('Chapters 8--11', 'Capitolele 8--11'), [
    T('Volatility is latent, persistent and predictable: EWMA and GARCH forecast it', 'Volatilitatea este latentă, persistentă și previzibilă: EWMA și GARCH o prognozează'),
    T('A risk number needs both a volatility forecast and a heavy-tailed quantile, and then a backtest', 'O măsură de risc are nevoie atît de o prognoză a volatilității, cît și de o cuantilă cu cozi groase, apoi de backtesting'),
    T('Long memory: weak in returns, strong in volatility', 'Memoria lungă: slabă în randamente, puternică în volatilitate')])

# =============================================================================
# 5. CAPITOLELE 12--15
# =============================================================================
D.section('Chapters 12--15: Applications', 'Capitolele 12--15: aplicații')

review(12, ('Scoring Models', 'modele de scoring'),
       [T('EL $=$ PD $\\times$ LGD $\\times$ EAD; logit $\\ln\\frac{p}{1-p} = \\beta_0 + x^\\top\\beta$, $e^{\\beta_j}$ = odds ratio', 'EL $=$ PD $\\times$ LGD $\\times$ EAD; logit $\\ln\\frac{p}{1-p} = \\beta_0 + x^\\top\\beta$, $e^{\\beta_j}$ = raportul șanselor'),
        T('AUC $= P(s_{\\text{bad}} > s_{\\text{good}})$, Gini $= 2\\,\\mathrm{AUC} - 1$; Altman $Z$ and Merton distance to default \\refAltman', 'AUC $= P(s_{\\text{rău}} > s_{\\text{bun}})$, Gini $= 2\\,\\mathrm{AUC} - 1$; scorul $Z$ Altman și distanța pînă la nerambursare Merton \\refAltman')],
       [T('German credit data (@{f12.n} loans), test sample of @{f12.ntest}: AUC $= @{f12.auc}$, 95\\% CI [@{f12.lo}, @{f12.hi}], Gini @{f12.gini}', 'datele germane de credit (@{f12.n} de credite), eșantion de test de @{f12.ntest}: AUC $= @{f12.auc}$, CI 95\\% [@{f12.lo}, @{f12.hi}], Gini @{f12.gini}'),
        T('discrimination is not calibration: check both, out of sample', 'discriminarea nu înseamnă calibrare: le verificăm pe amîndouă, în afara eșantionului')],
       [T('accuracy on imbalanced classes (always ``good\'\' is already right in most cases)', 'acuratețea pe clase dezechilibrate (regula „mereu bun” are deja dreptate în majoritatea cazurilor)'),
        T('choosing the cut-off without the costs of the two errors', 'alegerea pragului fără costurile celor două erori')])

review(13, ('Machine Learning', 'învățare automată'),
       [T('bias--variance: $E(y_0 - \\hat f)^2 = \\text{bias}^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$; ridge, lasso, trees, boosting', 'deplasare--varianță: $E(y_0 - \\hat f)^2 = \\text{deplasare}^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$; ridge, lasso, arbori, boosting'),
        T('out of sample: $R^2_{OOS} = 1 - \\sum(y - \\hat y)^2/\\sum(y - \\tilde y)^2$ against a simple benchmark $\\tilde y$ \\refGKX', 'în afara eșantionului: $R^2_{OOS} = 1 - \\sum(y - \\hat y)^2/\\sum(y - \\tilde y)^2$ față de un reper simplu $\\tilde y$ \\refGKX')],
       [T('sign of S\\&P 500 returns since @{f13.y0}: ``always up\'\' @{f13.base}\\%, logit @{f13.logit}\\%, boosting @{f13.gb}\\%', 'semnul randamentelor S\\&P 500 din @{f13.y0}: „mereu creștere” @{f13.base}\\%, logit @{f13.logit}\\%, boosting @{f13.gb}\\%'),
        T('realised volatility: HAR $R^2_{OOS} = @{f13.har}$ against the mean; lasso lowers the squared error of HAR by only @{f13.lasso}\\%', 'volatilitatea realizată: HAR $R^2_{OOS} = @{f13.har}$ față de medie; lasso reduce eroarea pătratică a HAR doar cu @{f13.lasso}\\%')],
       [T('random cross-validation on time series (look-ahead bias and leakage)', 'validarea încrucișată aleatoare pe serii de timp (look-ahead bias și leakage)'),
        T('reporting the best of many trials without deflating the Sharpe ratio', 'raportarea celei mai bune dintre multe încercări fără deflatarea raportului Sharpe')])

review(14, ('Crypto Assets', 'active cripto'),
       [T('annualise with 365 days: $\\sigma_{\\text{ann}} = \\sqrt{365}\\,\\sigma$; supply of Bitcoin capped at 21 million \\refNakamoto', 'anualizăm cu 365 de zile: $\\sigma_{\\text{an}} = \\sqrt{365}\\,\\sigma$; oferta de Bitcoin limitată la 21 de milioane \\refNakamoto'),
        T('peg deviation of a stablecoin $d_t = 10\\,000\\,(P_t - 1)$ basis points', 'abaterea de la paritate a unui stablecoin $d_t = 10\\,000\\,(P_t - 1)$ puncte de bază')],
       [T('Bitcoin: volatility @{f14.vol}\\% a year with 365 days (@{f14.vol252}\\% if one wrongly uses 252), maximum drawdown $@{f14.mdd}\\%$', 'Bitcoin: volatilitate de @{f14.vol}\\% pe an cu 365 de zile (@{f14.vol252}\\% dacă se folosește greșit 252), drawdown maxim $@{f14.mdd}\\%$'),
        T('correlation with the S\\&P 500: @{f14.pre} before 2020, @{f14.post} after: not a safe haven', 'corelația cu S\\&P 500: @{f14.pre} înainte de 2020, @{f14.post} după: nu este un activ de refugiu')],
       [T('removing weekends from crypto data, or joining crypto and equity returns on different calendars', 'eliminarea weekendurilor din datele cripto sau alinierea randamentelor cripto și de acțiuni pe calendare diferite'),
        T('a Normal VaR for a series with $\\alpha + \\beta = 1$ and tail index below 3', 'un VaR Normal pentru o serie cu $\\alpha + \\beta = 1$ și tail index sub 3')])

review(15, ('Systemic Risk', 'risc sistemic'),
       [T('CoVaR: the VaR of the system given an institution at its VaR; $\\Delta\\mathrm{CoVaR} = b\\,(q_{50} - q_\\alpha)$ from a quantile regression \\refAB', 'CoVaR: VaR-ul sistemului cînd o instituție se află la VaR-ul ei; $\\Delta\\mathrm{CoVaR} = b\\,(q_{50} - q_\\alpha)$ dintr-o regresie cuantilică \\refAB'),
        T('MES, SRISK, Granger networks and the Diebold--Yilmaz connectedness table \\refDY', 'MES, SRISK, rețele Granger și tabelul de conectivitate Diebold--Yilmaz \\refDY')],
       [T('Banca Transilvania and the BET: VaR 1\\% @{f15.var}\\%, $\\Delta$CoVaR 1\\% @{f15.dcv} pp (95\\% CI [@{f15.lo}, @{f15.hi}])', 'Banca Transilvania și BET: VaR 1\\% @{f15.var}\\%, $\\Delta$CoVaR 1\\% @{f15.dcv} pp (CI 95\\% [@{f15.lo}, @{f15.hi}])'),
        T('11 US and European banks: total connectedness of about @{f15.dy}\\%', '11 bănci americane și europene: conectivitate totală de circa @{f15.dy}\\%')],
       [T('ranking banks by their own VaR: a large VaR need not mean a large contribution to the system', 'ordonarea băncilor după propriul VaR: un VaR mare nu înseamnă neapărat o contribuție mare la riscul sistemului'),
        T('correlations compared across calm and crisis periods without the Forbes--Rigobon adjustment', 'corelații comparate între perioade calme și de criză fără ajustarea Forbes--Rigobon')])

D.recap(('Chapters 12--15', 'Capitolele 12--15'), [
    T('Scoring: discrimination (AUC, Gini) and calibration, out of sample, with costs', 'Scoring: discriminare (AUC, Gini) și calibrare, în afara eșantionului, cu costuri'),
    T('Machine learning: judged against a simple benchmark, with walk-forward validation', 'Machine learning: evaluat față de un reper simplu, cu validare walk-forward'),
    T('Crypto and systemic risk: the same tools, with the right calendar and conditional tail measures', 'Cripto și risc sistemic: aceleași instrumente, cu calendarul corect și măsuri condiționate ale cozii')])

# =============================================================================
# 6. TRUSA DE INSTRUMENTE
# =============================================================================
D.section('The Toolbox', 'Trusa de instrumente')

TB = '>{\\raggedright\\arraybackslash}'
TH = (T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Tool}', '\\textbf{Instrumentul}') + ' & '
      + T('\\textbf{Decision}', '\\textbf{Decizia}') + ' & \\textbf{' + T('Ch.', 'Cap.') + '}')
SPEC = TB + 'p{3.6cm}' + TB + 'p{3.4cm}' + TB + 'p{3.4cm}' + 'c'
D.frame(T('Toolbox (1/2): Returns, Distributions and Tails', 'Trusa de instrumente (1/2): randamente, distribuții și cozi'), table(
    SPEC, TH,
    [T('How good was the investment?', 'Cît de bună a fost investiția?') + ' & CAGR, Sharpe, Sortino, MDD & ' + T('compare with SE and a benchmark', 'comparăm cu SE și un reper') + ' & 1',
     T('Are returns Normal?', 'Sînt randamentele Normale?') + ' & ' + T('JB, QQ plot', 'JB, QQ plot') + ' & ' + T('reject if JB $> 5.99$', 'respingem dacă JB $> 5{,}99$') + ' & 2',
     T('How heavy are the tails?', 'Cît de groase sînt cozile?') + ' & ' + T('Hill plot, GPD, stable $\\alpha$', 'graficul Hill, GPD, $\\alpha$ stabil') + ' & ' + T('moments exist for $p < \\alpha$', 'momentele există pentru $p < \\alpha$') + ' & 3, 5',
     T('Which distribution fits?', 'Ce distribuție se potrivește?') + ' & ' + T('LR, AIC, BIC, KS, AD', 'LR, AIC, BIC, KS, AD') + ' & ' + T('smallest AIC; AD for the tails', 'AIC minim; AD pentru cozi') + ' & 6',
     T('How rare is a loss of $x$?', 'Cît de rară este o pierdere de $x$?') + ' & ' + T('empirical CDF, EVT, Monte Carlo', 'funcția de repartiție empirică, EVT, Monte Carlo') + ' & ' + T('report the SE of the estimate', 'raportăm SE a estimării') + ' & 4, 5',
     T('Is the variance finite?', 'Este varianța finită?') + ' & ' + T('running variance, aggregation of $\\hat\\alpha$', 'varianța cumulativă, agregarea lui $\\hat\\alpha$') + ' & ' + T('stable sample variance: finite', 'varianță de selecție stabilă: finită') + ' & 3'],
    size='footnotesize'))

D.frame(T('Toolbox (2/2): Dependence in Time, Volatility and Risk', 'Trusa de instrumente (2/2): dependență în timp, volatilitate și risc'), table(
    SPEC, TH,
    [T('Are returns predictable?', 'Sînt randamentele previzibile?') + ' & ' + T('ACF, Ljung--Box, robust VR $Z^*(q)$', 'ACF, Ljung--Box, VR robust $Z^*(q)$') + ' & ' + T('reject if $|Z^*| > 1.96$', 'respingem dacă $|Z^*| > 1{,}96$') + ' & 7',
     T('Is there volatility clustering?', 'Există volatility clustering?') + ' & ' + T('Ljung--Box on $r_t^2$, ARCH-LM', 'Ljung--Box pe $r_t^2$, ARCH-LM') + ' & ' + T('reject: model the variance', 'respingem: modelăm varianța') + ' & 8',
     T('How volatile tomorrow?', 'Cît de volatil va fi mîine?') + ' & ' + T('EWMA, GARCH(1,1)-$t$, GJR', 'EWMA, GARCH(1,1)-$t$, GJR') + ' & ' + T('QLIKE against a benchmark', 'QLIKE față de un reper') + ' & 8, 9',
     T('How much can we lose?', 'Cît putem pierde?') + ' & ' + T('VaR 1\\%, ES 2.5\\%: HS, $t$, EVT, FHS', 'VaR 1\\%, ES 2,5\\%: HS, $t$, EVT, FHS') + ' & ' + T('Kupiec, Christoffersen, traffic light', 'Kupiec, Christoffersen, semaforul Basel') + ' & 10',
     T('Is there long memory?', 'Există memorie lungă?') + ' & ' + T('R/S, DFA, GPH, Lo test', 'R/S, DFA, GPH, testul Lo') + ' & ' + T('compare with a Monte Carlo band', 'comparăm cu o bandă Monte Carlo') + ' & 11',
     T('Who will default?', 'Cine va intra în nerambursare?') + ' & ' + T('logit, LDA, AUC, Gini', 'logit, LDA, AUC, Gini') + ' & ' + T('out of sample, with costs', 'în afara eșantionului, cu costuri') + ' & 12',
     T('Which bank matters for the system?', 'Ce bancă contează pentru sistem?') + ' & CoVaR, MES, SRISK & ' + T('bootstrap CI of $\\Delta$CoVaR', 'CI bootstrap pentru $\\Delta$CoVaR') + ' & 15'],
    size='footnotesize'))

D.frame(T('Conventions Used Throughout the Course', 'Convențiile folosite în tot cursul'), items(
    (T('\\textbf{Returns}: daily log returns in \\%, $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$', '\\textbf{Randamentele}: randamente logaritmice zilnice în \\%, $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$'),
     [T('portfolios: simple returns, prices joined on common days first', 'portofolii: randamente simple, după alinierea prețurilor pe zilele comune')]),
    (T('\\textbf{Annualisation}: mean $\\times A$, volatility $\\times\\sqrt{A}$, with the actual $A$', '\\textbf{Anualizarea}: media $\\times A$, volatilitatea $\\times\\sqrt{A}$, cu $A$ real'),
     [T('$A$: the number of observations per year: about 252 for shares and indices, 365 for crypto', '$A$: numărul de observații pe an: circa 252 pentru acțiuni și indici, 365 pentru cripto')]),
    (T('\\textbf{Risk}: the level is the tail probability: VaR 1\\%, ES 2.5\\%', '\\textbf{Riscul}: nivelul este probabilitatea cozii: VaR 1\\%, ES 2,5\\%'),
     [T('$\\mathrm{VaR}_\\alpha = -q_\\alpha > 0$, a loss; historical simulation with $k = \\lceil n\\alpha \\rceil$', '$\\mathrm{VaR}_\\alpha = -q_\\alpha > 0$, o pierdere; simularea istorică cu $k = \\lceil n\\alpha \\rceil$')]),
    (T('\\textbf{Tests}: state $H_0$, the statistic, its distribution, the critical value and the decision at 5\\%', '\\textbf{Testele}: precizăm $H_0$, statistica, distribuția ei, valoarea critică și decizia la 5\\%'),
     [T('then one sentence on what the decision means for the data', 'apoi o frază despre semnificația deciziei pentru date')])))

D.frame(T('Six Pitfalls Across the Course', 'Șase capcane din tot cursul'), items(
    T('\\textbf{Data}: aligning returns instead of prices; unadjusted prices; weekends dropped from crypto', '\\textbf{Datele}: alinierea randamentelor în loc de prețuri; prețuri neajustate; weekenduri eliminate la cripto'),
    T('\\textbf{Distribution}: the Normal distribution for tail quantiles; excess kurtosis confused with kurtosis', '\\textbf{Distribuția}: distribuția Normală pentru cuantilele din coadă; confuzia dintre boltire și excesul de boltire'),
    T('\\textbf{Inference}: i.i.d. standard errors with volatility clustering; a statistic without its SE', '\\textbf{Inferența}: erori standard i.i.d.\\ în prezența volatility clustering; o statistică fără SE'),
    T('\\textbf{Time}: look-ahead bias, parameters estimated on the whole sample and used in the past', '\\textbf{Timpul}: look-ahead bias, parametri estimați pe tot eșantionul și folosiți în trecut'),
    T('\\textbf{Risk}: the sign and the level of VaR; ES compared with VaR at different levels', '\\textbf{Riscul}: semnul și nivelul VaR; ES comparat cu VaR la niveluri diferite'),
    T('\\textbf{Interpretation}: significance taken for economic importance, predictability for profit', '\\textbf{Interpretarea}: semnificația statistică luată drept importanță economică, predictibilitatea drept profit')))

# =============================================================================
# 7. EXAMENUL
# =============================================================================
D.section('The Exam', 'Examenul')

D.frame(T('The Written Exam: Format', 'Examenul scris: formatul'), cols(items(
    (T('\\textbf{70\\% of the final grade}; written, 2 hours; a calculator is allowed', '\\textbf{70\\% din nota finală}; scris, 2 ore; calculatorul este permis'),
     [T('all subjects are compulsory; 9 points plus 1 point ex officio', 'toate subiectele sînt obligatorii; 9 puncte plus 1 punct din oficiu')]),
    (T('\\textbf{Useful values are given} on the paper', '\\textbf{Valorile utile sînt date} pe subiect'),
     [T('quantiles of the Normal and Student-$t$ distributions, critical values of $\\chi^2$, logarithms and square roots', 'cuantile ale distribuției Normale și Student-$t$, valori critice $\\chi^2$, logaritmi și radicali')]),
    (T('\\textbf{Each subject}: a short context, numbers or a table or a chart, one or two questions', '\\textbf{Fiecare subiect}: un context scurt, cifre, un tabel sau un grafic, una sau două cerințe'),
     [T('a computation, followed by an interpretation in three to five sentences', 'un calcul urmat de o interpretare în trei pînă la cinci fraze')])),
    ph('ase', T('Bucharest University of Economic Studies', 'Academia de Studii Economice din București'), h='0.40\\textheight'), '0.58', '0.38'))

D.frame(T('Content Assessed', 'Conținutul evaluat'), items(
    (T('\\textbf{Chapters 0--15}: lectures and seminars', '\\textbf{Capitolele 0--15}: cursuri și seminarii'),
     [T('the definitions and formulas of the ``What You Need for Today\'\' slides of every seminar', 'definițiile și formulele de pe slide-urile „Noțiuni necesare azi” ale fiecărui seminar'),
      T('the computations of Part A of the seminars, on paper', 'calculele din Partea A a seminariilor, pe hîrtie')]),
    (T('\\textbf{Reading software output}', '\\textbf{Interpretarea rezultatelor obținute cu software}'),
     [T('tables and charts like those of Part B: estimates, test statistics, $p$-values, backtests', 'tabele și grafice ca în Partea B: estimări, statistici de test, p-value-uri, backtesting'),
      T('no code is written at the exam', 'la examen nu se scrie cod')]),
    (T('\\textbf{Understanding, not memory}', '\\textbf{Înțelegere, nu memorare}'),
     [T('why a method fails on financial returns, which test answers which question', 'de ce o metodă dă greș pe randamentele financiare, ce test răspunde la ce întrebare')])))

D.frame(T('Grading Criteria', 'Criterii de notare'), items(
    (T('\\textbf{The method}: the formula, with the numbers substituted', '\\textbf{Metoda}: formula, cu cifrele înlocuite'),
     [T('a correct method with an arithmetic slip receives most of the points', 'o metodă corectă cu o greșeală de calcul primește cea mai mare parte a punctajului')]),
    (T('\\textbf{The result}: the number, with its unit (\\%, days, RON) and its sign', '\\textbf{Rezultatul}: cifra, cu unitatea de măsură (\\%, zile, lei) și semnul ei'),
     [T('for a test: $H_0$, the statistic, the critical value or the $p$-value, the decision', 'pentru un test: $H_0$, statistica, valoarea critică sau p-value-ul, decizia')]),
    (T('\\textbf{The interpretation}: three to five sentences, statistical and economic', '\\textbf{Interpretarea}: trei pînă la cinci fraze, statistice și economice'),
     [T('state the assumptions of an approximate formula (i.i.d., Normal, large $n$)', 'precizați ipotezele unei formule aproximative (i.i.d., distribuția Normală, $n$ mare)'),
      T('a correct number with a wrong interpretation does not receive all the points', 'o cifră corectă cu o interpretare greșită nu primește tot punctajul')])))

D.frame(T('Typical Problem Types', 'Tipuri de probleme'), table(
    TB + 'p{4.2cm}' + TB + 'p{1.4cm}' + TB + 'p{5.3cm}',
    T('\\textbf{Type}', '\\textbf{Tipul}') + ' & ' + T('\\textbf{Chapters}', '\\textbf{Capitole}') + ' & ' + T('\\textbf{Example in this chapter}', '\\textbf{Exemplu în acest capitol}'),
    [T('returns, annualisation, performance', 'randamente, anualizare, performanță') + ' & 0, 1 & ' + T('Problem 1: CAGR, drawdown, volatility drag', 'Problema 1: CAGR, drawdown, volatility drag'),
     T('moments, normality, tails', 'momente, normalitate, cozi') + ' & 2, 3, 5, 6 & ' + T('Problem 2: JB, Student-$t$, Hill', 'Problema 2: JB, Student-$t$, Hill'),
     T('probability and simulation', 'probabilitate și simulare') + ' & 4 & ' + T('Problem 3: portfolio, Monte Carlo, GBM', 'Problema 3: portofoliu, Monte Carlo, GBM'),
     T('efficiency and memory', 'eficiență și memorie') + ' & 7, 11 & ' + T('Problems 4 and 7: Ljung--Box, VR, Hurst', 'Problemele 4 și 7: Ljung--Box, VR, Hurst'),
     T('volatility', 'volatilitate') + ' & 8, 9, 14 & ' + T('Problem 5: GARCH, EWMA, annualisation of Bitcoin', 'Problema 5: GARCH, EWMA, anualizarea Bitcoin'),
     T('risk measures and backtesting', 'măsuri de risc și backtesting') + ' & 10 & ' + T('Problem 6: VaR, ES, Kupiec, traffic light', 'Problema 6: VaR, ES, Kupiec, semaforul Basel'),
     T('credit and systemic risk', 'risc de credit și risc sistemic') + ' & 12, 13, 15 & ' + T('Problem 8: confusion matrix, EL, $\\Delta$CoVaR', 'Problema 8: matricea de confuzie, EL, $\\Delta$CoVaR')],
    size='footnotesize') + items(
    T('The eight problems below are review examples with full solutions; the exam has its own subjects', 'Cele opt probleme de mai jos sînt exemple de recapitulare, cu rezolvări complete; examenul are subiecte proprii')))


def problem(k, title, context, tasks):
    ctx = '\n'.join(f'        \\item {c}' for c in context)
    body = ('\\begin{itemize}\n    \\item \\textbf{Context}\n    \\begin{itemize}\n' + ctx + '\n    \\end{itemize}\n'
            '    \\item ' + T('\\textbf{Tasks}', '\\textbf{Cerințe}') + '\n' + enum(*tasks) + '\n\\end{itemize}')
    D.frame(T(f'Problem {k}: {title[0]}', f'Problema {k}: {title[1]}'), body, 'footnotesize')


def solution(k, steps, interp, size='footnotesize'):
    D.frame(T(f'Problem {k}: Solution', f'Problema {k}: rezolvare'),
            enum(*steps) + '\n' + items((T('\\textbf{Interpretation}', '\\textbf{Interpretare}'), interp)), size)


problem(1, ('Returns and Drawdown', 'randamente și drawdown'),
        [T('An index is worth 100, 160, 64 and 96 at the end of years 0, 1, 2 and 3', 'Un indice valorează 100, 160, 64 și 96 la sfîrșitul anilor 0, 1, 2 și 3')],
        [T('Compute the simple returns $R_t$ and the log returns $r_t$ of the three years.', 'Calculați randamentele simple $R_t$ și randamentele logaritmice $r_t$ ale celor trei ani.'),
         T('Compute the total return in two ways: from $\\sum_t r_t$ and from the prices.', 'Calculați randamentul total în două moduri: din $\\sum_t r_t$ și din prețuri.'),
         T('Compute the arithmetic mean of $R_t$ and the CAGR, and explain the difference.', 'Calculați media aritmetică a lui $R_t$ și CAGR, apoi explicați diferența.'),
         T('Compute the maximum drawdown and the gain needed from the trough to the previous peak.', 'Calculați drawdown-ul maxim și cîștigul necesar de la minim pînă la vîrful anterior.')])
solution(1, [
    T('$R = @{e1.R1}\\%, @{e1.R2}\\%, @{e1.R3}\\%$; $r = \\ln 1.6 = @{e1.r1}$, $\\ln 0.4 = @{e1.r2}$, $\\ln 1.5 = @{e1.r3}$', '$R = @{e1.R1}\\%; @{e1.R2}\\%; @{e1.R3}\\%$; $r = \\ln 1{,}6 = @{e1.r1}$, $\\ln 0{,}4 = @{e1.r2}$, $\\ln 1{,}5 = @{e1.r3}$'),
    T('$\\sum_t r_t = @{e1.sum} = \\ln(96/100)$, so the total return is $e^{@{e1.sum}} - 1 = @{e1.tot}\\%$, the same as $96/100 - 1$', '$\\sum_t r_t = @{e1.sum} = \\ln(96/100)$, deci randamentul total este $e^{@{e1.sum}} - 1 = @{e1.tot}\\%$, la fel ca $96/100 - 1$'),
    T('arithmetic mean $(60 - 60 + 50)/3 = @{e1.am}\\%$; CAGR $= (96/100)^{1/3} - 1 = @{e1.cagr}\\%$', 'media aritmetică $(60 - 60 + 50)/3 = @{e1.am}\\%$; CAGR $= (96/100)^{1/3} - 1 = @{e1.cagr}\\%$'),
    T('MDD $= 64/160 - 1 = @{e1.mdd}\\%$; to return from 64 to 160 the index needs $160/64 - 1 = @{e1.gain}\\%$', 'MDD $= 64/160 - 1 = @{e1.mdd}\\%$; pentru revenirea de la 64 la 160, indicele are nevoie de $160/64 - 1 = @{e1.gain}\\%$')],
    [T('A positive average return and a loss of money: volatility lowers the compound growth (volatility drag)', 'Un randament mediu pozitiv și totuși o pierdere de bani: volatilitatea reduce creșterea compusă (volatility drag)'),
     T('Log returns add over time; simple returns do not', 'Randamentele logaritmice se adună în timp; cele simple nu')])

problem(2, ('Distribution and Tails', 'distribuție și cozi'),
        [T('$n = 1000$ daily returns with skewness $S = -0.5$ and excess kurtosis $K = 6$', '$n = 1000$ de randamente zilnice cu asimetria $S = -0{,}5$ și excesul de boltire $K = 6$'),
         T('The five largest losses, in \\%: $8.2$, $6.5$, $5.9$, $5.1$, $4.6$; $\\chi^2_{0.95}(2) = 5.99$', 'Cele mai mari cinci pierderi, în \\%: $8{,}2$; $6{,}5$; $5{,}9$; $5{,}1$; $4{,}6$; $\\chi^2_{0,95}(2) = 5{,}99$')],
        [T('Compute the Jarque--Bera statistic, its two parts, and decide at 5\\%.', 'Calculați statistica Jarque--Bera și cele două componente ale ei, apoi decideți la 5\\%.'),
         T('Find the degrees of freedom $\\nu$ of a Student-$t$ distribution with the same excess kurtosis.', 'Determinați numărul de grade de libertate $\\nu$ al unei distribuții Student-$t$ cu același exces de boltire.'),
         T('Compute the Hill estimate $\\hat\\alpha$ with $k = 4$.', 'Calculați estimarea Hill $\\hat\\alpha$ cu $k = 4$.'),
         T('Say which moments of the losses exist if $\\alpha = \\hat\\alpha$, and whether a stable law with $\\alpha < 2$ is consistent with it.', 'Precizați ce momente ale pierderilor există dacă $\\alpha = \\hat\\alpha$ și dacă o lege stabilă cu $\\alpha < 2$ este compatibilă cu acest rezultat.')])
solution(2, [
    T('skewness part $\\frac{1000}{6}(0.25) = @{e2.s}$; kurtosis part $\\frac{1000}{24}(36) = @{e2.k}$; JB $= @{e2.jb} > 5.99$: normality is rejected', 'componenta asimetriei $\\frac{1000}{6}(0{,}25) = @{e2.s}$; componenta boltirii $\\frac{1000}{24}(36) = @{e2.k}$; JB $= @{e2.jb} > 5{,}99$: normalitatea se respinge'),
    T('$6/(\\nu - 4) = 6 \\Rightarrow \\nu = @{e2.nu}$', '$6/(\\nu - 4) = 6 \\Rightarrow \\nu = @{e2.nu}$'),
    T('$\\ln(8.2/4.6) = @{e2.l1}$, $\\ln(6.5/4.6) = @{e2.l2}$, $\\ln(5.9/4.6) = @{e2.l3}$, $\\ln(5.1/4.6) = @{e2.l4}$; mean $@{e2.ml}$, so $\\hat\\alpha = @{e2.a}$', '$\\ln(8{,}2/4{,}6) = @{e2.l1}$, $\\ln(6{,}5/4{,}6) = @{e2.l2}$, $\\ln(5{,}9/4{,}6) = @{e2.l3}$, $\\ln(5{,}1/4{,}6) = @{e2.l4}$; media $@{e2.ml}$, deci $\\hat\\alpha = @{e2.a}$'),
    T('moments of order $p < @{e2.a}$ exist: mean, variance, skewness; the kurtosis does not; a stable law with $\\alpha < 2$ (infinite variance) has heavier tails than these data', 'există momentele de ordin $p < @{e2.a}$: media, varianța, asimetria; boltirea nu există; o lege stabilă cu $\\alpha < 2$ (varianță infinită) are cozi mai groase decît aceste date')],
    [T('The kurtosis part dominates JB: heavy tails, not asymmetry, reject normality', 'Componenta boltirii domină JB: cozile groase, nu asimetria, duc la respingerea normalității'),
     T('$k = 4$ gives a very noisy $\\hat\\alpha$ (SE $\\approx \\hat\\alpha/2$); in practice $k$ is chosen from a Hill plot', '$k = 4$ dă o estimare foarte zgomotoasă a lui $\\hat\\alpha$ (SE $\\approx \\hat\\alpha/2$); în practică, $k$ se alege din graficul Hill')])

problem(3, ('Portfolio, Monte Carlo and GBM', 'portofoliu, Monte Carlo și GBM'),
        [T('Two assets: daily volatilities $1.5\\%$ and $2.5\\%$, correlation $0.3$, weights 60\\% and 40\\%', 'Două active: volatilități zilnice de $1{,}5\\%$ și $2{,}5\\%$, corelație $0{,}3$, ponderi 60\\% și 40\\%'),
         T('A share price follows a GBM with $\\mu = 8\\%$, $\\sigma = 25\\%$ a year, $S_0 = 100$', 'Prețul unei acțiuni urmează un GBM cu $\\mu = 8\\%$, $\\sigma = 25\\%$ pe an, $S_0 = 100$')],
        [T('Compute the daily volatility of the portfolio and compare it with the weighted average of the volatilities.', 'Calculați volatilitatea zilnică a portofoliului și comparați-o cu media ponderată a volatilităților.'),
         T('A Monte Carlo study with $N = 10\\,000$ draws estimates a probability of 1\\%; compute its SE and a 95\\% interval.', 'Un studiu Monte Carlo cu $N = 10\\,000$ de extrageri estimează o probabilitate de 1\\%; calculați SE și un interval de 95\\%.'),
         T('Compute the mean and the median of $S_1$, and $P(S_1 < S_0)$.', 'Calculați media și mediana lui $S_1$, precum și $P(S_1 < S_0)$.')])
solution(3, [
    T('$\\sigma_p^2 = @{e3.t1} + @{e3.t2} + @{e3.t3} = @{e3.vp}$, $\\sigma_p = @{e3.sp}\\%$, below the weighted average $@{e3.avg}\\%$', '$\\sigma_p^2 = @{e3.t1} + @{e3.t2} + @{e3.t3} = @{e3.vp}$, $\\sigma_p = @{e3.sp}\\%$, sub media ponderată $@{e3.avg}\\%$'),
    T('SE $= \\sqrt{0.01 \\times 0.99/10\\,000} = @{e3.se}\\%$; interval $[@{e3.lo}\\%, @{e3.hi}\\%]$', 'SE $= \\sqrt{0{,}01 \\times 0{,}99/10\\,000} = @{e3.se}\\%$; interval $[@{e3.lo}\\%; @{e3.hi}\\%]$'),
    T('mean $100e^{0.08} = @{e3.mean}$; median $100e^{0.08 - 0.03125} = @{e3.med}$; $P(S_1 < S_0) = \\Phi(-@{e3.m2}/0.25) = \\Phi(@{e3.zz}) = @{e3.ploss}\\%$', 'media $100e^{0{,}08} = @{e3.mean}$; mediana $100e^{0{,}08 - 0{,}03125} = @{e3.med}$; $P(S_1 < S_0) = \\Phi(-@{e3.m2}/0{,}25) = \\Phi(@{e3.zz}) = @{e3.ploss}\\%$')],
    [T('Diversification: with $\\rho < 1$ the portfolio is less risky than the average of its parts', 'Diversificarea: cu $\\rho < 1$, portofoliul este mai puțin riscant decît media componentelor'),
     T('A GBM investor loses money after one year with probability @{e3.ploss}\\%, although the expected price rises', 'Un investitor într-un GBM pierde bani după un an cu probabilitatea de @{e3.ploss}\\%, deși prețul așteptat crește')])

problem(4, ('Autocorrelation and the Variance Ratio', 'autocorelație și raportul varianțelor'),
        [T('$T = 2500$ daily returns; $\\hat\\rho(1) = 0.06$, $\\hat\\rho(2) = 0.02$, $\\hat\\rho(3) = -0.01$, $\\hat\\rho(4) = 0.01$; $\\chi^2_{0.95}(4) = @{chi4}$', '$T = 2500$ de randamente zilnice; $\\hat\\rho(1) = 0{,}06$, $\\hat\\rho(2) = 0{,}02$, $\\hat\\rho(3) = -0{,}01$, $\\hat\\rho(4) = 0{,}01$; $\\chi^2_{0,95}(4) = @{chi4}$'),
         T('The heteroskedasticity-robust standard error of $\\widehat{\\mathrm{VR}}(5)$ is $0.065$', 'Eroarea standard robustă la heteroscedasticitate a lui $\\widehat{\\mathrm{VR}}(5)$ este $0{,}065$')],
        [T('Compute the Ljung--Box statistic $Q(4)$ and decide at 5\\%.', 'Calculați statistica Ljung--Box $Q(4)$ și decideți la 5\\%.'),
         T('Compute $\\mathrm{VR}(5) = 1 + 2\\sum_{k=1}^{4}(1 - k/5)\\hat\\rho(k)$.', 'Calculați $\\mathrm{VR}(5) = 1 + 2\\sum_{k=1}^{4}(1 - k/5)\\hat\\rho(k)$.'),
         T('Compute the i.i.d. statistic $Z(5)$ with SE $= \\sqrt{2(2q-1)(q-1)/(3qT)}$ and the robust $Z^*(5)$, and decide at 5\\%.', 'Calculați statistica i.i.d.\\ $Z(5)$, cu SE $= \\sqrt{2(2q-1)(q-1)/(3qT)}$, și statistica robustă $Z^*(5)$, apoi decideți la 5\\%.')])
solution(4, [
    T('$Q(4) = T(T+2)\\sum_{k=1}^{4}\\hat\\rho_k^2/(T-k) = @{e4.lb}$, close to $T\\sum_k\\hat\\rho_k^2$ with $\\sum_k\\hat\\rho_k^2 = @{e4.sum}$; $@{e4.lb} > @{chi4}$: the absence of autocorrelation is rejected', '$Q(4) = T(T+2)\\sum_{k=1}^{4}\\hat\\rho_k^2/(T-k) = @{e4.lb}$, apropiat de $T\\sum_k\\hat\\rho_k^2$, cu $\\sum_k\\hat\\rho_k^2 = @{e4.sum}$; $@{e4.lb} > @{chi4}$: se respinge absența autocorelației'),
    T('$\\mathrm{VR}(5) = 1 + 2(0.8 \\times 0.06 + 0.6 \\times 0.02 - 0.4 \\times 0.01 + 0.2 \\times 0.01) = 1 + 2 \\times @{e4.inner} = @{e4.vr}$', '$\\mathrm{VR}(5) = 1 + 2(0{,}8 \\times 0{,}06 + 0{,}6 \\times 0{,}02 - 0{,}4 \\times 0{,}01 + 0{,}2 \\times 0{,}01) = 1 + 2 \\times @{e4.inner} = @{e4.vr}$'),
    T('SE $= \\sqrt{72/37\\,500} = @{e4.se}$, $Z(5) = @{e4.z} > 1.96$; robust $Z^*(5) = (@{e4.vr} - 1)/0.065 = @{e4.zs} < 1.96$', 'SE $= \\sqrt{72/37\\,500} = @{e4.se}$, $Z(5) = @{e4.z} > 1{,}96$; $Z^*(5) = (@{e4.vr} - 1)/0{,}065 = @{e4.zs} < 1{,}96$ (robust)')],
    [T('VR $> 1$: positive autocorrelation (momentum); the i.i.d. test rejects the random walk', 'VR $> 1$: autocorelație pozitivă (momentum); testul i.i.d.\\ respinge mersul aleator'),
     T('The robust test does not: volatility clustering inflates $Z(5)$; the robust decision is the one to report', 'Testul robust nu îl respinge: volatility clustering mărește artificial statistica $Z(5)$; decizia care se raportează este cea robustă')])

problem(5, ('Volatility Forecasts', 'prognoze de volatilitate'),
        [T('GARCH(1,1) for daily returns in \\%: $\\omega = 0.03$, $\\alpha = 0.09$, $\\beta = 0.89$; today $\\sigma_t^2 = 1.8$ and $\\varepsilon_t = -2.5$', 'GARCH(1,1) pentru randamente zilnice în \\%: $\\omega = 0{,}03$, $\\alpha = 0{,}09$, $\\beta = 0{,}89$; azi $\\sigma_t^2 = 1{,}8$ și $\\varepsilon_t = -2{,}5$'),
         T('Bitcoin: daily standard deviation $3.5\\%$', 'Bitcoin: abaterea standard zilnică $3{,}5\\%$')],
        [T('Compute $\\sigma_{t+1}^2$ and the one-day Normal VaR 1\\% (mean zero, $z_{0.01} = -2.326$).', 'Calculați $\\sigma_{t+1}^2$ și VaR 1\\% Normal pe o zi (medie zero, $z_{0,01} = -2{,}326$).'),
         T('Compute the persistence, the half-life, the long-run variance and the long-run annual volatility.', 'Calculați persistența, timpul de înjumătățire, varianța pe termen lung și volatilitatea anuală pe termen lung.'),
         T('Compute $E_t[\\sigma_{t+10}^2]$ and the EWMA forecast with $\\lambda = 0.94$.', 'Calculați $E_t[\\sigma_{t+10}^2]$ și prognoza EWMA cu $\\lambda = 0{,}94$.'),
         T('Annualise the volatility of Bitcoin.', 'Anualizați volatilitatea Bitcoin.')])
solution(5, [
    T('$\\sigma_{t+1}^2 = 0.03 + @{e5.a} + @{e5.b} = @{e5.s2}$, $\\sigma_{t+1} = @{e5.s}\\%$; VaR 1\\% $= 2.326 \\times @{e5.s} = @{e5.var}\\%$', '$\\sigma_{t+1}^2 = 0{,}03 + @{e5.a} + @{e5.b} = @{e5.s2}$, $\\sigma_{t+1} = @{e5.s}\\%$; VaR 1\\% $= 2{,}326 \\times @{e5.s} = @{e5.var}\\%$'),
    T('$\\alpha + \\beta = @{e5.pers}$; $h_{1/2} = \\ln 0.5/\\ln 0.98 = @{e5.hl}$ days; $\\bar\\sigma^2 = 0.03/0.02 = @{e5.lv}$; $\\sqrt{252 \\times @{e5.lv}} = @{e5.ann}\\%$ a year', '$\\alpha + \\beta = @{e5.pers}$; $h_{1/2} = \\ln 0{,}5/\\ln 0{,}98 = @{e5.hl}$ zile; $\\bar\\sigma^2 = 0{,}03/0{,}02 = @{e5.lv}$; $\\sqrt{252 \\times @{e5.lv}} = @{e5.ann}\\%$ pe an'),
    T('$E_t\\sigma_{t+10}^2 = @{e5.lv} + 0.98^9(@{e5.s2} - @{e5.lv}) = @{e5.h10}$; EWMA: $0.94 \\times 1.8 + 0.06 \\times 6.25 = @{e5.ew}$', '$E_t\\sigma_{t+10}^2 = @{e5.lv} + 0{,}98^9(@{e5.s2} - @{e5.lv}) = @{e5.h10}$; EWMA: $0{,}94 \\times 1{,}8 + 0{,}06 \\times 6{,}25 = @{e5.ew}$'),
    T('Bitcoin trades every day: $3.5 \\times \\sqrt{365} = @{e5.b365}\\%$; with 252 the result, $@{e5.b252}\\%$, would be wrong', 'Bitcoin se tranzacționează în fiecare zi: $3{,}5 \\times \\sqrt{365} = @{e5.b365}\\%$; cu 252, rezultatul ar fi greșit: $@{e5.b252}\\%$')],
    [T('A large negative shock raises tomorrow\'s variance; the forecast returns to $\\bar\\sigma^2$ at the speed $\\alpha + \\beta$', 'Un șoc negativ mare crește varianța de mîine; prognoza revine la $\\bar\\sigma^2$ cu viteza $\\alpha + \\beta$'),
     T('EWMA has no long-run level: its forecast stays flat for every horizon', 'EWMA nu are un nivel pe termen lung: prognoza ei rămîne constantă pentru orice orizont')])

problem(6, ('VaR, ES and Backtesting', 'VaR, ES și backtesting'),
        [T('A position of 1\\,000\\,000 RON; daily returns $N(0, 1.6^2)$ in \\%; $\\varphi(1.96) = @{phi25}$', 'O poziție de 1\\,000\\,000 de lei; randamente zilnice $N(0; 1{,}6^2)$ în \\%; $\\varphi(1{,}96) = @{phi25}$'),
         T('The seven smallest of 250 historical returns, in \\%: $-5.1$, $-4.4$, $-3.9$, $-3.6$, $-3.2$, $-3.0$, $-2.9$', 'Cele mai mici șapte dintre 250 de randamente istorice, în \\%: $-5{,}1$; $-4{,}4$; $-3{,}9$; $-3{,}6$; $-3{,}2$; $-3{,}0$; $-2{,}9$')],
        [T('Compute the Normal VaR 1\\% and ES 2.5\\% in \\% and in RON, and the 10-day VaR 1\\% by the square-root-of-time rule.', 'Calculați VaR 1\\% și ES 2,5\\% Normale, în \\% și în lei, precum și VaR 1\\% pe 10 zile cu regula rădăcinii pătrate a timpului.'),
         T('Compute VaR 1\\% and ES 2.5\\% by historical simulation.', 'Calculați VaR 1\\% și ES 2,5\\% prin simulare istorică.'),
         T('A model has 6 exceptions of VaR 1\\% in 250 days: compute the Kupiec statistic, decide at 5\\%, and give the Basel zone.', 'Un model are 6 depășiri ale VaR 1\\% în 250 de zile: calculați statistica Kupiec, decideți la 5\\% și precizați zona Basel.')])
solution(6, [
    T('VaR 1\\% $= 2.326 \\times 1.6 = @{e6.var}\\%$ (@{e6.varm} RON); ES 2.5\\% $= 1.6 \\times @{phi25}/0.025 = 1.6 \\times @{e6.esf} = @{e6.es}\\%$ (@{e6.esm} RON); 10 days: $\\sqrt{10} \\times @{e6.var} = @{e6.v10}\\%$',
      'VaR 1\\% $= 2{,}326 \\times 1{,}6 = @{e6.var}\\%$ (@{e6.varm} de lei); ES 2,5\\% $= 1{,}6 \\times @{phi25}/0{,}025 = 1{,}6 \\times @{e6.esf} = @{e6.es}\\%$ (@{e6.esm} de lei); 10 zile: $\\sqrt{10} \\times @{e6.var} = @{e6.v10}\\%$'),
    T('$k = \\lceil 2.5 \\rceil = 3$: VaR 1\\% $= 3.9\\%$; $k = \\lceil 6.25 \\rceil = 7$: ES 2.5\\% $= @{e6.hssum}/7 = @{e6.hses}\\%$', '$k = \\lceil 2{,}5 \\rceil = 3$: VaR 1\\% $= 3{,}9\\%$; $k = \\lceil 6{,}25 \\rceil = 7$: ES 2,5\\% $= @{e6.hssum}/7 = @{e6.hses}\\%$'),
    T('$\\ell_0 = 244\\ln 0.99 + 6\\ln 0.01 = @{e6.l0}$, $\\ell_1 = 244\\ln 0.976 + 6\\ln 0.024 = @{e6.l1}$; $LR_{uc} = @{e6.lr} < @{chi1}$ ($p = @{e6.p}$): not rejected; yellow zone (5--9 exceptions)',
      '$\\ell_0 = 244\\ln 0{,}99 + 6\\ln 0{,}01 = @{e6.l0}$, $\\ell_1 = 244\\ln 0{,}976 + 6\\ln 0{,}024 = @{e6.l1}$; $LR_{uc} = @{e6.lr} < @{chi1}$ ($p = @{e6.p}$): nu se respinge; zona galbenă (5--9 depășiri)')],
    [T('For the Normal distribution ES 2.5\\% and VaR 1\\% are almost equal; the historical tail is heavier', 'Pentru distribuția Normală, ES 2,5\\% și VaR 1\\% sînt aproape egale; coada istorică este mai groasă'),
     T('Kupiec does not reject 6 exceptions, but the traffic light already asks for a higher capital multiplier', 'Testul Kupiec nu respinge 6 depășiri, dar semaforul Basel cere deja un multiplicator de capital mai mare')])

problem(7, ('Long Memory and Risk Scaling', 'memorie lungă și scalarea riscului'),
        [T('The average R/S statistic of daily returns is $4.1$ for blocks of $n = 20$ days and $60.0$ for $n = 2000$', 'Statistica R/S medie a randamentelor zilnice este $4{,}1$ pentru blocuri de $n = 20$ de zile și $60{,}0$ pentru $n = 2000$'),
         T('One-day VaR 1\\% $= 2.5\\%$', 'VaR 1\\% pe o zi $= 2{,}5\\%$')],
        [T('Estimate $H$ from the slope of $\\log(R/S)$ on $\\log n$, and the fractional parameter $d$.', 'Estimați $H$ din panta lui $\\log(R/S)$ în funcție de $\\log n$, precum și parametrul fracționar $d$.'),
         T('Scale the VaR to 10 days with $h^H$ and with $\\sqrt{h}$, and compare.', 'Scalați VaR la 10 zile cu $h^H$ și cu $\\sqrt{h}$, apoi comparați.'),
         T('Say what must be checked before concluding that there is long memory.', 'Precizați ce trebuie verificat înainte de a concluziona că există memorie lungă.')])
solution(7, [
    T('$H = \\log_{10}(60.0/4.1)/\\log_{10}(2000/20) = \\log_{10}(@{e7.ratio})/2 = @{e7.lr}/2 = @{e7.H}$; $d = H - 0.5 = @{e7.d}$', '$H = \\log_{10}(60{,}0/4{,}1)/\\log_{10}(2000/20) = \\log_{10}(@{e7.ratio})/2 = @{e7.lr}/2 = @{e7.H}$; $d = H - 0{,}5 = @{e7.d}$'),
    T('$10^{H} = @{e7.hH}$: VaR $= @{e7.vH}\\%$; $\\sqrt{10} = @{e7.sq}$: VaR $= @{e7.vs}\\%$', '$10^{H} = @{e7.hH}$: VaR $= @{e7.vH}\\%$; $\\sqrt{10} = @{e7.sq}$: VaR $= @{e7.vs}\\%$'),
    T('a Monte Carlo band of $\\hat H$ for i.i.d. data of the same length (R/S is biased upwards), the Lo test, split samples and shuffled data (breaks, GARCH)', 'o bandă Monte Carlo pentru $\\hat H$ pe date i.i.d.\\ de aceeași lungime (R/S este deplasat în sus), testul Lo, subeșantioane și date permutate (rupturi, GARCH)')],
    [T('With $H > 0.5$ the square-root-of-time rule understates multi-day risk', 'Cu $H > 0{,}5$, regula rădăcinii pătrate a timpului subestimează riscul pe mai multe zile'),
     T('On the course data the long memory is in $|r_t|$, not in $r_t$ (Chapter 11)', 'Pe datele cursului, memoria lungă se află în $|r_t|$, nu în $r_t$ (Capitolul 11)')])

problem(8, ('Credit Scoring and Systemic Risk', 'scoring de credit și risc sistemic'),
        [T('A test sample of 300 loans, 90 bad; at the chosen cut-off: 60 bad and 42 good loans are flagged as bad', 'Un eșantion de test de 300 de credite, dintre care 90 neperformante; la pragul ales sînt marcate ca neperformante 60 dintre creditele neperformante și 42 dintre cele performante'),
         T('Quantile regression at 1\\% of the system return on a bank\'s return: $\\hat a = -2.1$, $\\hat b = 0.6$; bank: $q_{0.01} = -5.0\\%$, median $0.1\\%$', 'Regresia cuantilică la 1\\% a randamentului sistemului în funcție de randamentul unei bănci: $\\hat a = -2{,}1$, $\\hat b = 0{,}6$; banca: $q_{0,01} = -5{,}0\\%$, mediana $0{,}1\\%$')],
        [T('Compute the accuracy, the true positive rate (TPR), the false positive rate (FPR) and the precision, and compare the accuracy with ``always good\'\'.', 'Calculați acuratețea, rata adevărat pozitivelor (TPR), rata fals pozitivelor (FPR) și precizia, apoi comparați acuratețea cu regula „mereu bun”.'),
         T('Compute the Gini coefficient for AUC $= 0.80$ and the expected loss for PD $= 4\\%$, LGD $= 45\\%$, EAD $= 200\\,000$ RON.', 'Calculați coeficientul Gini pentru AUC $= 0{,}80$ și pierderea așteptată pentru PD $= 4\\%$, LGD $= 45\\%$, EAD $= 200\\,000$ de lei.'),
         T('Compute the CoVaR 1\\% of the system with the bank at its VaR and at its median, and $\\Delta$CoVaR.', 'Calculați CoVaR 1\\% al sistemului cînd banca este la VaR-ul ei și la mediană, precum și $\\Delta$CoVaR.')])
solution(8, [
    T('accuracy $(60 + 168)/300 = @{e8.acc}\\%$; TPR $= 60/90 = @{e8.tpr}\\%$; FPR $= 42/210 = @{e8.fpr}\\%$; precision $60/102 = @{e8.prec}\\%$; ``always good\'\': @{e8.base}\\%', 'acuratețea $(60 + 168)/300 = @{e8.acc}\\%$; TPR $= 60/90 = @{e8.tpr}\\%$; FPR $= 42/210 = @{e8.fpr}\\%$; precizia $60/102 = @{e8.prec}\\%$; „mereu bun”: @{e8.base}\\%'),
    T('Gini $= 2 \\times 0.80 - 1 = @{e8.gini}$; EL $= 0.04 \\times 0.45 \\times 200\\,000 = @{e8.el}$ RON', 'Gini $= 2 \\times 0{,}80 - 1 = @{e8.gini}$; EL $= 0{,}04 \\times 0{,}45 \\times 200\\,000 = @{e8.el}$ de lei'),
    T('CoVaR $= -(-2.1 + 0.6 \\times (-5.0)) = @{e8.covar}\\%$; at the median: $-(-2.1 + 0.6 \\times 0.1) = @{e8.cmed}\\%$; $\\Delta$CoVaR $= 0.6 \\times 5.1 = @{e8.dcv}$ pp', 'CoVaR $= -(-2{,}1 + 0{,}6 \\times (-5{,}0)) = @{e8.covar}\\%$; la mediană: $-(-2{,}1 + 0{,}6 \\times 0{,}1) = @{e8.cmed}\\%$; $\\Delta$CoVaR $= 0{,}6 \\times 5{,}1 = @{e8.dcv}$ pp')],
    [T('The model is barely more accurate than ``always good\'\', but it finds two thirds of the bad loans: judge it by AUC and costs', 'Modelul este doar puțin mai precis decît regula „mereu bun”, dar găsește două treimi dintre creditele neperformante: îl judecăm după AUC și costuri'),
     T('The distress of this bank raises the VaR of the system by @{e8.dcv} percentage points', 'Dificultățile acestei bănci cresc VaR-ul sistemului cu @{e8.dcv} puncte procentuale')])

D.recap(('The Exam', 'examenul'), [
    T('Written, 2 hours, calculator allowed; Chapters 0--15; computations and interpretations', 'Scris, 2 ore, cu calculator; Capitolele 0--15; calcule și interpretări'),
    T('Points for the method, the result with its unit and sign, and the interpretation', 'Punctajul se acordă pentru metodă, pentru rezultat (cu unitate și semn) și pentru interpretare'),
    T('Practise with Part A of every seminar and with the review seminar', 'Exersați cu Partea A a fiecărui seminar și cu seminarul de recapitulare')])

# =============================================================================
# 8. PROIECTUL DE ECHIPĂ
# =============================================================================
D.section('The Team Project', 'Proiectul de echipă')

D.frame(T('The Team Project: Content and Deliverables', 'Proiectul de echipă: conținut și livrabile'), items(
    (T('\\textbf{20\\% of the final grade}: a team analysis of real market data with the methods of the course', '\\textbf{20\\% din nota finală}: o analiză în echipă a unor date reale de piață, cu metodele cursului'),
     [T('one research question; returns and indicators, distributions and tails, efficiency tests, volatility and risk measures', 'o singură întrebare de cercetare; randamente și indicatori, distribuții și cozi, teste de eficiență, volatilitate și măsuri de risc'),
      T('ideas: the ``Project Idea\'\' slide at the end of every chapter and Part C of every seminar', 'idei: slide-ul „Idee de proiect” de la finalul fiecărui capitol și Partea C a fiecărui seminar')]),
    (T('\\textbf{Deliverables}', '\\textbf{Livrabile}'),
     [T('a GitHub repository whose code reproduces every number and every chart from the saved data', 'un repository GitHub al cărui cod reproduce, din datele salvate, fiecare rezultat numeric și fiecare grafic'),
      T('a short report and a presentation of the results', 'un raport scurt și o prezentare a rezultatelor'),
      T('the file \\texttt{AI\\_USE.md}, which declares every use of AI tools', 'fișierul \\texttt{AI\\_USE.md}, în care este declarată fiecare utilizare a instrumentelor AI')])))

D.frame(T('Project Grading Criteria', 'Criterii de evaluare a proiectului'), items(
    (T('\\textbf{The question}: clear, answerable with the data, relevant for investors or regulators', '\\textbf{Întrebarea}: clară, cu răspuns posibil pe baza datelor, relevantă pentru investitori sau pentru autorități'),
     [T('``Is the BET today less volatile than in 2008?\'\' is a question; ``an analysis of the BET\'\' is not', '„Este BET azi mai puțin volatil decît în 2008?” este o întrebare; „o analiză a BET” nu este')]),
    (T('\\textbf{The methods and the checks}', '\\textbf{Metodele și verificările}'),
     [T('the right test for the question (the toolbox); robust standard errors; out-of-sample evaluation', 'testul potrivit pentru întrebare (trusa de instrumente); erori standard robuste; evaluare în afara eșantionului'),
      T('robustness: another window, another series, another estimator', 'robustețe: altă fereastră, altă serie, alt estimator')]),
    (T('\\textbf{The interpretation}: what the numbers mean, and what the data cannot show', '\\textbf{Interpretarea}: ce înseamnă cifrele și ce nu pot arăta datele'),
     [T('\\textbf{Reproducibility}: the repository runs from the saved data to every chart', '\\textbf{Reproductibilitatea}: repository-ul rulează de la datele salvate pînă la fiecare grafic')])))

D.frame(T('The Oral Defence', 'Susținerea orală'), items(
    (T('\\textbf{Each member explains the code and the results}', '\\textbf{Fiecare membru explică codul și rezultatele}'),
     [T('typical questions: what does this line compute? where does this number come from? why this test and not another?', 'întrebări tipice: ce calculează această linie? de unde provine această cifră? de ce acest test și nu altul?'),
      T('what changes if the window, the series or the level $\\alpha$ changes?', 'ce se schimbă dacă se modifică fereastra, seria sau nivelul $\\alpha$?')]),
    (T('\\textbf{A line of code that cannot be explained does not count as your own work}', '\\textbf{O linie de cod pe care nu o puteți explica nu este considerată muncă proprie}'),
     [T('this holds for code written with or without AI', 'regula este aceeași pentru codul scris cu sau fără AI')]),
    (T('\\textbf{Preparation}', '\\textbf{Pregătirea}'),
     [T('rerun the repository from scratch the day before; know the definitions of every number on your slides', 'rulați din nou repository-ul de la zero cu o zi înainte; cunoașteți definiția fiecărei cifre de pe slide-urile voastre')])))

D.frame(T('The File AI\\_USE.md', 'Fișierul AI\\_USE.md'), items(
    (T('\\textbf{AI tools are allowed and must be declared}', '\\textbf{Instrumentele AI sînt permise și trebuie declarate}'),
     [T('for every use: the tool, the prompt, what was kept and what was corrected', 'pentru fiecare utilizare: instrumentul, promptul, ce s-a păstrat și ce s-a corectat'),
      T('the errors of the AI that you found and how you found them', 'erorile instrumentului AI pe care le-ați găsit și modul în care le-ați găsit')]),
    (T('\\textbf{You are responsible for every number and every reference}', '\\textbf{Răspundeți pentru fiecare cifră și pentru fiecare referință}'),
     [T('every number: recomputed with your own code from the course data', 'fiecare cifră: recalculată cu propriul cod, din datele cursului'),
      T('every reference: its DOI opened and its title checked', 'fiecare referință: DOI-ul deschis și titlul verificat')]),
    (T('\\textbf{Typical AI errors in this course}', '\\textbf{Erori AI tipice în acest curs}'),
     [T('``VaR 99\\%\'\' with a negative sign, Bitcoin annualised with 252 days, invented references, a test with the wrong distribution', '„VaR 99\\%” cu semn negativ, Bitcoin anualizat cu 252 de zile, referințe inventate, un test cu distribuția greșită')])))

D.recap(('The Team Project', 'proiectul de echipă'), [
    T('One question, real data, reproducible code, a short report, a presentation', 'O întrebare, date reale, cod reproductibil, un raport scurt, o prezentare'),
    T('Graded on the question, the methods and checks, the interpretation and the oral defence', 'Evaluat după întrebare, metode și verificări, interpretare și susținerea orală'),
    T('AI allowed, declared in \\texttt{AI\\_USE.md}, every number and reference checked', 'AI permis, declarat în \\texttt{AI\\_USE.md}, cu fiecare cifră și referință verificată')])

# =============================================================================
# 9. CONTRIBUȚIA POSIBILĂ A AI
# =============================================================================
D.section('How AI Could Help: Your Own Research', 'Contribuția posibilă a AI: propria cercetare')

D.frame(T('An Open Question for Your Own Research', 'O întrebare deschisă pentru propria cercetare'), cols(items(
    (T('\\textbf{Has the BET become a ``normal\'\' market?}', '\\textbf{A devenit BET o piață „obișnuită”?}'),
     [T('since @{f7.y0.bet}: VR(5) $= @{f7.vr.bet}$ (momentum) and tail index @{f5.a.bet}', 'din @{f7.y0.bet}: VR(5) $= @{f7.vr.bet}$ (momentum) și tail index-ul @{f5.a.bet}'),
      T('since 2015: VR(5) $= @{s.bet.vr}$, $Z^* = @{s.bet.zs}$, tail index @{s.bet.hill}', 'din 2015: VR(5) $= @{s.bet.vr}$, $Z^* = @{s.bet.zs}$, tail index-ul @{s.bet.hill}')]),
    (T('\\textbf{Why it is open}', '\\textbf{Motivele pentru care întrebarea rămîne deschisă}'),
     [T('the momentum fades, but the tails do not become thinner', 'momentum-ul slăbește, dar cozile nu devin mai subțiri'),
      T('one extreme day ($@{f1.betminr}\\%$ on @{f1.betmin}) moves the tail estimate; few crises, short samples', 'o singură zi extremă ($@{f1.betminr}\\%$ la @{f1.betmin}) modifică estimarea cozii; puține crize, eșantioane scurte')]),
    T('Different teams with the same data reach different answers: nonstandard errors \\refMenk', 'Echipe diferite, cu aceleași date, ajung la răspunsuri diferite: erorile nestandard \\refMenk')),
    ph('mandelbrot', T('Benoit Mandelbrot, who showed in 1963 that price changes have heavy tails', 'Benoit Mandelbrot, care a arătat în 1963 că variațiile prețurilor au cozi groase'), h='0.40\\textheight'), '0.60', '0.36'))

D.frame(T('The Research Loop', 'Bucla de cercetare'), enum(
    T('\\textbf{Question}: one sentence, with a number that would answer it', '\\textbf{Întrebarea}: o frază, cu o cifră care i-ar răspunde'),
    T('\\textbf{Data}: series, source, calendar, window, fixed before looking at the results', '\\textbf{Datele}: seria, sursa, calendarul, fereastra, fixate înainte de a vedea rezultatele'),
    T('\\textbf{Method}: the test from the toolbox, with its null hypothesis', '\\textbf{Metoda}: testul din trusa de instrumente, cu ipoteza nulă'),
    T('\\textbf{Checks}: robust SE, Monte Carlo band, subsamples, another estimator', '\\textbf{Verificările}: SE robuste, bandă Monte Carlo, subeșantioane, alt estimator'),
    T('\\textbf{Interpretation}: the answer, its uncertainty, its limits', '\\textbf{Interpretarea}: răspunsul, incertitudinea lui, limitele lui'),
    T('\\textbf{Replication}: a second person runs the code and gets the same numbers', '\\textbf{Replicarea}: o a doua persoană rulează codul și obține aceleași cifre')) + '\n' + items(
    T('AI can speed up every step; it replaces none of the checks \\refWang', 'AI poate accelera fiecare pas; nu înlocuiește nicio verificare \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a first list of studies on the efficiency and the tails of emerging markets, each to be checked by its DOI', '\\textbf{Literatura}: o primă listă de studii despre eficiența și cozile piețelor emergente, fiecare verificat prin DOI'),
    T('\\textbf{Code}: a first draft of rolling VR(5), Hill and GARCH estimates on windows of 500 days', '\\textbf{Codul}: o primă versiune a estimărilor VR(5), Hill și GARCH pe ferestre mobile de 500 de zile'),
    T('\\textbf{Robustness}: a grid of choices (window, $k$, level $\\alpha$, series) and one table of all results', '\\textbf{Robustețea}: o grilă de alegeri (fereastra, $k$, nivelul $\\alpha$, seria) și un singur tabel cu toate rezultatele'),
    T('\\textbf{Writing}: a clearer paragraph, after the numbers are final', '\\textbf{Redactarea}: un paragraf mai clar, după ce cifrele sînt finale'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that computes, on rolling windows of 500 trading days moved by 21 days, the Lo-MacKinlay robust variance-ratio statistic Z*(5), the Hill tail index of the losses with k = 2.5\\% of the window, and the GARCH(1,1)-t persistence for the daily log returns of the BET, and plots the three series with their 95\\% bands.}',
        '\\aiprompt{Write Python code that computes, on rolling windows of 500 trading days moved by 21 days, the Lo-MacKinlay robust variance-ratio statistic Z*(5), the Hill tail index of the losses with k = 2.5\\% of the window, and the GARCH(1,1)-t persistence for the daily log returns of the BET, and plots the three series with their 95\\% bands.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The formulas: the robust $Z^*(q)$, not $Z(q)$; the Hill estimator on losses $L = -r$, with $k$ given as a share of the window', 'Formulele: $Z^*(q)$ robust, nu $Z(q)$; estimatorul Hill pe pierderi $L = -r$, cu $k$ dat ca proporție din fereastră'),
    T('The look-ahead: each window uses only the data up to its last day', 'Look-ahead: fiecare fereastră folosește doar datele pînă în ultima ei zi'),
    T('The bands: overlapping windows are not independent evidence; one result in twenty is significant by chance', 'Benzile: ferestrele suprapuse nu sînt dovezi independente; un rezultat din douăzeci este semnificativ din întîmplare'),
    T('The numbers: recompute one window step by step and compare', 'Cifrele: recalculați pas cu pas o fereastră și comparați'),
    T('The references: every cited paper must exist; open the DOI and check the title', 'Referințele: fiecare lucrare citată trebuie să existe; deschideți DOI-ul și verificați titlul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: did the BET become more efficient, less volatile and thinner-tailed between 2000 and 2026?', '\\textbf{Întrebarea}: a devenit BET mai eficient, mai puțin volatil și cu cozi mai subțiri între 2000 și 2026?'),
     [T('data: BET since 1997, with the S\\&P 500 and the DAX as benchmarks (EODHD)', 'date: BET din 1997, cu S\\&P 500 și DAX ca repere (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('rolling robust VR(5), Hill index, GARCH persistence and Hurst exponent (Chapters 5, 7, 9, 11)', 'VR(5) robust, tail index-ul Hill, persistența GARCH și exponentul Hurst pe ferestre mobile (Capitolele 5, 7, 9, 11)'),
      T('bands from Monte Carlo or the bootstrap; the same analysis for the benchmarks', 'benzi Monte Carlo sau bootstrap; aceeași analiză pentru repere'),
      T('dates of change (e.g.\\ 2008, 2020) checked against the bands, not chosen after looking', 'datele de schimbare (de exemplu 2008, 2020) verificate față de benzi, nu alese după inspecția graficelor')]),
    T('Deliverable: one chart with four panels, one table, and a paragraph on what the data can and cannot show', 'Livrabil: un grafic cu patru panouri, un tabel și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Returns, not prices: log returns in \\%, the right calendar, annualised with the actual frequency', 'Randamente, nu prețuri: randamente logaritmice în \\%, calendarul corect, anualizare cu frecvența reală'),
    T('Returns have heavy tails (tail index 2--4), negative skewness and finite variance: the Normal distribution understates risk', 'Randamentele au cozi groase (tail index între 2 și 4), asimetrie negativă și varianță finită: distribuția Normală subestimează riscul'),
    T('The sign of returns is almost unpredictable; their size is predictable (volatility clustering, GARCH)', 'Semnul randamentelor este aproape imprevizibil; mărimea lor este previzibilă (volatility clustering, GARCH)'),
    T('Risk: VaR 1\\% $= -q_{1\\%}$ and ES 2.5\\%, always followed by a backtest', 'Riscul: VaR 1\\% $= -q_{1\\%}$ și ES 2,5\\%, urmate întotdeauna de backtesting'),
    T('Every result needs its uncertainty: SE, robust tests, Monte Carlo bands, out-of-sample checks', 'Orice rezultat are nevoie de incertitudinea lui: SE, teste robuste, benzi Monte Carlo, verificări în afara eșantionului'),
    T('The exam rewards method, result and interpretation; the project rewards a clear question answered honestly', 'Examenul răsplătește metoda, rezultatul și interpretarea; proiectul răsplătește o întrebare clară, cu un răspuns onest')))

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: Ljung--Box does not reject on $r_t$ but rejects strongly on $r_t^2$. What does this mean?', '\\textbf{Întrebare}: Ljung--Box nu respinge pe $r_t$, dar respinge puternic pe $r_t^2$. Ce înseamnă acest lucru?'),
     [T('\\textbf{Answer}: returns are uncorrelated but not independent: volatility clustering; model the variance (GARCH)', '\\textbf{Răspuns}: randamentele sînt necorelate, dar nu independente: volatility clustering; modelăm varianța (GARCH)')]),
    (T('\\textbf{Question}: a fitted stable law gives $\\hat\\alpha = 1.5$ and the Hill estimate is $\\hat\\alpha = 3$. Which is right about the variance?', '\\textbf{Întrebare}: o lege stabilă estimată dă $\\hat\\alpha = 1{,}5$, iar estimarea Hill este $\\hat\\alpha = 3$. Care are dreptate în privința varianței?'),
     [T('\\textbf{Answer}: Hill, which looks only at the tail: the variance is finite; the stable fit describes the body', '\\textbf{Răspuns}: estimarea Hill, care privește doar coada: varianța este finită; legea stabilă descrie corpul distribuției')]),
    (T('\\textbf{Question}: a model has 1\\% exceptions, all in March 2020. Is it a good VaR model?', '\\textbf{Întrebare}: un model are 1\\% depășiri, toate în martie 2020. Este un model VaR bun?'),
     [T('\\textbf{Answer}: no: the rate is right, but the exceptions cluster; the Christoffersen test rejects it', '\\textbf{Răspuns}: nu: rata este corectă, dar depășirile sînt grupate; testul Christoffersen îl respinge')])))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
