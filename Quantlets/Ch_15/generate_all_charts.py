"""
generate_all_charts.py -- charts and numbers of Chapter 15 (SFM): systemic risk
================================================================================
Course data (sfm_data.py), chart style (sfm_style.py). Every number on the slides comes from here.
Data: daily adjusted closes of 13 banks (6 US, 5 European, 2 Romanian) and the closes of the S&P 500, Euro Stoxx 50,
BET and VIX since 2005 (EODHD); the two Romanian banks since 2010. Prices are joined on common days first, then daily log returns in % are computed.
Convention: the level alpha is the probability of the tail ("VaR 1%", "CoVaR 1%", "MES 5%"); losses are positive.
  * crises        -- equal-weighted bank portfolios of the three regions (US, Europe, Romania), four episodes:
                     Lehman (2008), the euro-area crisis (2011-2012), March 2020, March 2023;
  * correlation   -- rolling average pairwise correlation of bank returns; the Forbes-Rigobon (2002) adjustment;
                     empirical lower tail dependence and the t copula (Chapter 10);
  * CoVaR         -- CoVaR and Delta-CoVaR at 1% and 5% by quantile regression (Adrian and Brunnermeier, 2016) of the
                     regional index on each bank, with moving-block bootstrap intervals; a time-varying Delta-CoVaR with
                     lagged state variables (VIX level and index return);
  * MES and SRISK -- MES 5% (Acharya et al., 2017), the long-run MES approximation LRMES = 1 - exp(-18 MES)
                     (Acharya, Engle and Richardson, 2012), SRISK per unit of market equity as a function of leverage
                     (Brownlees and Engle, 2017); MES measured from June 2006 to June 2007 against the realised return
                     from July 2007 to December 2008 (the design of Acharya et al., 2017);
  * networks      -- pairwise Granger causality on monthly returns in 36-month rolling windows and the degree of Granger
                     causality (Billio et al., 2012); the Diebold-Yilmaz (2012) connectedness table and its rolling total
                     (11 US and European banks).
Output: charts/sfm_ch15_*.pdf/.png, Quantlets/Ch_15/ch15_numbers.json and ch15_*_table.csv
Based on Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., Ch. 16-17, and on the papers cited.
Run:  python3 Quantlets/Ch_15/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import optimize
from scipy import stats
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import read_market, load_close   # noqa: E402
import sfm_style as st                          # noqa: E402
import statsmodels.api as sm                    # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg   # noqa: E402
import warnings                                 # noqa: E402
warnings.filterwarnings('ignore')

TABLE_DIR = HERE
START, END = '2005-01-01', '2026-09-18'
# bank -> (EODHD symbol, name, region)
BANKS = {
    'JPM': ('JPM.US', 'JPMorgan Chase', 'US'), 'BAC': ('BAC.US', 'Bank of America', 'US'),
    'C': ('C.US', 'Citigroup', 'US'), 'GS': ('GS.US', 'Goldman Sachs', 'US'),
    'MS': ('MS.US', 'Morgan Stanley', 'US'), 'WFC': ('WFC.US', 'Wells Fargo', 'US'),
    'DBK': ('DBK.XETRA', 'Deutsche Bank', 'EU'), 'BNP': ('BNP.PA', 'BNP Paribas', 'EU'),
    'SAN': ('SAN.MC', 'Santander', 'EU'), 'INGA': ('INGA.AS', 'ING', 'EU'), 'HSBA': ('HSBA.LSE', 'HSBC', 'EU'),
    'TLV': ('TLV.RO', 'Banca Transilvania', 'RO'), 'BRD': ('BRD.RO', 'BRD', 'RO'),
}
REGIONS = {'US': [k for k, v in BANKS.items() if v[2] == 'US'], 'EU': [k for k, v in BANKS.items() if v[2] == 'EU'],
           'RO': [k for k, v in BANKS.items() if v[2] == 'RO']}
ALL = REGIONS['US'] + REGIONS['EU'] + REGIONS['RO']
INTL = REGIONS['US'] + REGIONS['EU']
# the BVB bank series are reliable from 2010 (many days without trades before; Chapter 10 uses the same start)
REGION_START = {'US': START, 'EU': START, 'RO': '2010-01-01'}
INDEX = {'US': 'sp500', 'EU': 'stoxx50', 'RO': 'bet'}
INDEX_NAME = {'sp500': 'S&P 500', 'stoxx50': 'Euro Stoxx 50', 'bet': 'BET'}
REG_COL = {'US': '#1A3A6E', 'EU': '#CD0000', 'RO': '#2E7D32'}
REG_LAB = {'US': 'US banks', 'EU': 'European banks', 'RO': 'Romanian banks (TLV, BRD, since 2010)'}
# Banca Transilvania: the bonus-share adjustment of May 2016 is applied one day late in the adjusted close (Chapter 2)
BAD_DAYS = {'TLV': ['2016-05-30', '2016-05-31']}
EVENTS = [('2008-09-15', 'Lehman'), ('2012-07-26', 'Draghi'), ('2020-03-16', 'COVID-19'), ('2023-03-10', 'SVB')]
EPISODES = {'Lehman 2008': ('2008-08-29', '2008-12-31'), 'Euro area 2011-2012': ('2011-06-30', '2012-09-28'),
            'March 2020': ('2020-02-14', '2020-04-30'), 'March 2023': ('2023-02-28', '2023-04-28')}
ALPHA = 0.01                 # CoVaR 1%
ALPHA_MES = 0.05             # MES 5%: the worst 5% of market days
K_CAP = 0.08                 # prudential capital ratio of SRISK (Brownlees and Engle, 2017)
B_BOOT, BLOCK = 200, 20      # moving-block bootstrap of Delta-CoVaR
GC_WINDOW = 36               # Granger networks: 36-month windows (Billio et al., 2012)
DY_H, DY_WINDOW, DY_STEP = 10, 200, 5    # Diebold-Yilmaz: horizon 10 days, 200-day rolling windows
SEED = 2026


# =============================================================================
# DATA
# =============================================================================
def bank_price(k, start=START, end=END):
    """Daily adjusted close of a bank (weekdays; no unchanged holiday-filled closes; BVB: no days without trades)."""
    sym, _, reg = BANKS[k]
    d = read_market(sym).loc[start:end]
    d = d[(d.index.dayofweek < 5) & (d['adjusted_close'] > 0)]
    if reg == 'RO' and 'volume' in d.columns:
        d = d[d['volume'] > 0]
    s = d['adjusted_close'].astype(float)
    return s[s.diff() != 0].rename(k)


def prices(keys, start=START, end=END):
    """Banks and indices (sp500, stoxx50, bet, vix) on their common trading days."""
    cols = [bank_price(k, start, end) if k in BANKS else load_close(k, start, end).rename(k) for k in keys]
    return pd.concat(cols, axis=1, join='inner').dropna()


def returns(keys, start=START, end=END):
    """Daily log returns in %: prices joined on common days first, then returns; known data errors removed."""
    R = 100 * np.log(prices(keys, start, end)).diff().dropna()
    for k, days in BAD_DAYS.items():
        if k in R.columns:
            R = R.drop(pd.to_datetime(days), errors='ignore')
    return R


def region_returns(reg, start=None, end=END):
    """The banks of a region and its equity index (S&P 500, Euro Stoxx 50 or BET), daily log returns in %."""
    return returns(REGIONS[reg] + [INDEX[reg]], max(start or START, REGION_START[reg]), end)


def bank_portfolio(R, members):
    """Equal-weighted bank portfolio (daily rebalancing): log return in % of the average simple return."""
    return 100 * np.log(1 + (np.exp(R[members] / 100) - 1).mean(axis=1))


# =============================================================================
# 1. CRISES
# =============================================================================
def fig_bank_indices(save=True):
    """Value of 100 invested in each equal-weighted regional bank portfolio in January 2005."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    out = {}
    for reg in ('US', 'EU', 'RO'):
        R = region_returns(reg)
        p = bank_portfolio(R, REGIONS[reg])
        v = 100 * np.exp(p.cumsum() / 100)
        dd = v / v.cummax() - 1
        ax.plot(v.index, v, color=REG_COL[reg], lw=1.3, label=REG_LAB[reg])
        out[reg] = {'last': float(v.iloc[-1]), 'maxdd': float(100 * dd.min()), 'date_dd': dd.idxmin().date().isoformat(),
                    'peak': float(v.loc[:'2008-12-31'].max()), 'start': R.index[0].date().isoformat(), 'n': len(R)}
    ax.set_yscale('log')
    ax.axhline(100, color=st.Amber, ls='--', lw=1.0, label='starting value 100')
    for d, lab in EVENTS:
        ax.axvline(pd.Timestamp(d), color=st.Purple, ls=':', lw=1.0, label='_')
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], ' ' + lab, rotation=90, va='top', ha='right', fontsize=10, color='black')
    ax.set_ylabel('value of 100 invested in 2005 (log scale)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_bank_indices')
    return out


def episode_paths():
    """Cumulative return (%) of the three bank portfolios and of the S&P 500 in each episode."""
    out = {}
    R = {reg: region_returns(reg) for reg in ('US', 'EU', 'RO')}
    for e, (a, b) in EPISODES.items():
        out[e] = {}
        for reg in ('US', 'EU', 'RO'):
            if pd.Timestamp(a) < pd.Timestamp(REGION_START[reg]):
                continue
            p = bank_portfolio(R[reg], REGIONS[reg]).loc[a:b].iloc[1:]
            out[e][reg] = 100 * (np.exp(p.cumsum() / 100) - 1)
        s = R['US']['sp500'].loc[a:b].iloc[1:]
        out[e]['sp500'] = 100 * (np.exp(s.cumsum() / 100) - 1)
    return out


def fig_episodes(save=True):
    """Four systemic episodes: cumulative return of the bank portfolios and of the S&P 500 from the start date."""
    P = episode_paths()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.6))
    res = {}
    for ax, (e, d) in zip(axes.ravel(), P.items()):
        for reg in ('US', 'EU', 'RO'):
            if reg in d:
                ax.plot(d[reg].index, d[reg], color=REG_COL[reg], lw=1.4, label=REG_LAB[reg])
        ax.plot(d['sp500'].index, d['sp500'], color=st.Amber, lw=1.4, ls='--', label='S&P 500')
        ax.axhline(0, color=st.Purple, lw=0.8, ls=':', label='_')
        ax.set_title(e, fontsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if e.startswith('Euro') else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y' if e.startswith('Euro') else '%d %b'))
        ax.tick_params(axis='x', labelsize=10.5)
        res[e] = {k: {'min': float(v.min()), 'date_min': v.idxmin().date().isoformat(), 'end': float(v.iloc[-1])}
                  for k, v in d.items()}
    for ax in axes[:, 0]:
        ax.set_ylabel('cumulative return (%)')
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.subplots_adjust(bottom=0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_episodes')
    return res


# =============================================================================
# 2. CORRELATION AND TAIL DEPENDENCE IN CRISES
# =============================================================================
def avg_pairwise_corr(R, window=120):
    """Rolling average of the pairwise correlations of the columns of R."""
    cols = list(R.columns)
    acc = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            acc.append(R[cols[i]].rolling(window).corr(R[cols[j]]))
    return pd.concat(acc, axis=1).mean(axis=1).dropna()


def forbes_rigobon(rho, delta):
    """Correlation corrected for the rise in volatility (Forbes and Rigobon, 2002): rho / sqrt(1 + delta (1 - rho^2)),
    delta = relative increase in the variance of the source market."""
    return rho / np.sqrt(1 + delta * (1 - rho ** 2))


def fig_rolling_corr(save=True):
    """120-day average pairwise correlation of the bank returns: US banks, European banks, TLV and BRD."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    out = {}
    for reg in ('US', 'EU', 'RO'):
        R = region_returns(reg)[REGIONS[reg]]
        c = avg_pairwise_corr(R)
        ax.plot(c.index, c, color=REG_COL[reg], lw=1.2, label=REG_LAB[reg])
        out[reg] = {'calm': float(c.loc['2005-07-01':'2007-06-30'].mean()) if reg != 'RO' else None, 'y2017': float(c.loc['2017'].mean()),
                    'max': float(c.max()), 'date_max': c.idxmax().date().isoformat(), 'last': float(c.iloc[-1]),
                    'c2008': float(c.loc['2008-10-01':'2009-03-31'].mean()) if reg != 'RO' else None, 'c2020': float(c.loc['2020-04-01':'2020-06-30'].mean())}
    for d, lab in EVENTS:
        ax.axvline(pd.Timestamp(d), color=st.Purple, ls=':', lw=1.0, label='_')
        ax.text(pd.Timestamp(d), 1.0, ' ' + lab, rotation=90, va='top', ha='right', fontsize=10, color='black')
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel('average pairwise correlation\n(120-day window)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_rolling_corr')
    return out


def contagion_test(calm=('2005-01-01', '2007-06-29'), crisis=('2008-09-15', '2008-12-31')):
    """US bank portfolio (source) and European bank portfolio: correlation in a calm and in a crisis period, and the
    Forbes-Rigobon correction with delta from the variance of the source market."""
    R = returns(REGIONS['US'] + REGIONS['EU'])
    us, eu = bank_portfolio(R, REGIONS['US']), bank_portfolio(R, REGIONS['EU'])
    c0 = us.loc[calm[0]:calm[1]].corr(eu.loc[calm[0]:calm[1]])
    c1 = us.loc[crisis[0]:crisis[1]].corr(eu.loc[crisis[0]:crisis[1]])
    v0, v1 = us.loc[calm[0]:calm[1]].var(), us.loc[crisis[0]:crisis[1]].var()
    delta = v1 / v0 - 1
    return {'rho_calm': float(c0), 'rho_crisis': float(c1), 'var_calm': float(v0), 'var_crisis': float(v1),
            'delta': float(delta), 'rho_adj': float(forbes_rigobon(c1, delta)),
            'n_calm': int(len(us.loc[calm[0]:calm[1]])), 'n_crisis': int(len(us.loc[crisis[0]:crisis[1]]))}


def pseudo_obs(X):
    """Ranks divided by n + 1 (pseudo-observations, Chapter 10)."""
    X = np.asarray(X, float)
    return stats.rankdata(X, axis=0) / (len(X) + 1)


def empirical_tail_dep(x, y, q=0.05):
    """P(U_x <= q | U_y <= q) from the pseudo-observations."""
    U = pseudo_obs(np.column_stack([x, y]))
    return float(np.mean(U[U[:, 1] <= q, 0] <= q))


def t_copula_fit(x, y):
    """t copula: rho = sin(pi tau / 2) from Kendall's tau, nu by maximum likelihood on a grid (Chapter 10)."""
    U = pseudo_obs(np.column_stack([x, y]))
    tau = stats.kendalltau(x, y)[0]
    rho = np.sin(np.pi * tau / 2)

    def nll(nu):
        z = stats.t.ppf(U, nu)
        q = (z[:, 0] ** 2 - 2 * rho * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / (1 - rho ** 2)
        ll = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * np.log(1 - rho ** 2)
              - (nu + 2) / 2 * np.log(1 + q / nu) - stats.t.logpdf(z, nu).sum(axis=1))
        return -ll.sum()
    r = optimize.minimize_scalar(nll, bounds=(2.05, 60), method='bounded')
    nu = float(r.x)
    lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    return {'tau': float(tau), 'rho': float(rho), 'nu': nu, 'lambda': float(lam)}


def tail_table():
    """Correlation, empirical lower tail dependence (q = 5%, 1%) and the t copula for four pairs."""
    out = {}
    for reg in ('US', 'EU', 'RO'):
        R = region_returns(reg)
        p = bank_portfolio(R, REGIONS[reg])
        m = R[INDEX[reg]]
        out[reg] = {'n': len(R), 'corr': float(np.corrcoef(p, m)[0, 1]), 'td5': empirical_tail_dep(p, m, 0.05),
                    'td1': empirical_tail_dep(p, m, 0.01), **t_copula_fit(p.values, m.values)}
    R = returns(REGIONS['US'] + REGIONS['EU'])
    us, eu = bank_portfolio(R, REGIONS['US']), bank_portfolio(R, REGIONS['EU'])
    out['USEU'] = {'n': len(R), 'corr': float(np.corrcoef(us, eu)[0, 1]), 'td5': empirical_tail_dep(eu, us, 0.05),
                   'td1': empirical_tail_dep(eu, us, 0.01), **t_copula_fit(eu.values, us.values)}
    return out


def fig_tail_scatter(save=True):
    """Pseudo-observations of the bank portfolio against its index (lower-left corner): US and Romania."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    out = {}
    for ax, reg in zip(axes, ('US', 'RO')):
        R = region_returns(reg)
        U = pseudo_obs(np.column_stack([R[INDEX[reg]], bank_portfolio(R, REGIONS[reg])]))
        both = (U[:, 0] <= 0.05) & (U[:, 1] <= 0.05)
        ax.scatter(U[~both, 0], U[~both, 1], s=6, color=REG_COL[reg], alpha=0.45, label='_')
        ax.scatter(U[both, 0], U[both, 1], s=10, color=st.Orange, label='both in their worst 5% of days')
        ax.plot([0, 0.05, 0.05], [0.05, 0.05, 0], color=st.Amber, lw=1.6, label='_')
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.3)
        ax.set_xlabel(f'{INDEX_NAME[INDEX[reg]]} (uniform scale)')
        ax.set_title(f'{REG_LAB[reg]}\n{int(both.sum())} joint bad days; {0.05 * 0.05 * len(U):.0f} expected '
                     'under independence', fontsize=11)
        out[reg] = {'both': int(both.sum()), 'exp': float(0.0025 * len(U)), 'n': len(U)}
    axes[0].set_ylabel('bank portfolio (uniform scale)')
    st.fig_legend_bottom(fig, ncol=1, y=0.0)
    fig.subplots_adjust(bottom=0.25)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_tail_scatter')
    return out


# =============================================================================
# 3. CoVaR
# =============================================================================
def qreg(y, X, q):
    """Linear quantile regression (Koenker and Bassett, 1978): coefficients [intercept, slopes]."""
    return QuantReg(np.asarray(y), sm.add_constant(np.asarray(X), has_constant='add')).fit(q=q, max_iter=5000).params


def covar(x_sys, x_i, alpha=ALPHA):
    """CoVaR and Delta-CoVaR (%, losses positive): quantile regression x_sys = a + b x_i at level alpha;
    CoVaR = -(a + b q_alpha(x_i)); Delta-CoVaR = b (q_50%(x_i) - q_alpha(x_i))."""
    x_sys, x_i = np.asarray(x_sys, float), np.asarray(x_i, float)
    a, b = qreg(x_sys, x_i, alpha)
    qa, qm = np.quantile(x_i, alpha), np.quantile(x_i, 0.5)
    return {'a': float(a), 'b': float(b), 'q_a': float(qa), 'q_m': float(qm), 'var_i': float(-qa),
            'covar': float(-(a + b * qa)), 'covar_med': float(-(a + b * qm)), 'dcovar': float(b * (qm - qa)),
            'var_sys': float(-np.quantile(x_sys, alpha))}


def block_idx(n, block, rng):
    """Indices of a moving-block bootstrap sample (blocks of `block` consecutive days)."""
    s = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (s[:, None] + np.arange(block)[None, :]).ravel()[:n]


def covar_ci(x_sys, x_i, alpha=ALPHA, B=B_BOOT, block=BLOCK, seed=SEED):
    """95% percentile interval of Delta-CoVaR by moving-block bootstrap."""
    x_sys, x_i = np.asarray(x_sys, float), np.asarray(x_i, float)
    rng = np.random.default_rng(seed)
    d = [covar(x_sys[i], x_i[i], alpha)['dcovar'] for i in (block_idx(len(x_i), block, rng) for _ in range(B))]
    return [float(v) for v in np.percentile(d, [2.5, 97.5])]


def covar_table(boot=True):
    """CoVaR 1% and 5% of the regional index conditional on each bank (2005-2026)."""
    rows = {}
    for reg in ('US', 'EU', 'RO'):
        R = region_returns(reg)
        for k in REGIONS[reg]:
            c1 = covar(R[INDEX[reg]], R[k], 0.01)
            c5 = covar(R[INDEX[reg]], R[k], 0.05)
            rows[k] = {'region': reg, 'n': len(R), **c1, 'dcovar5': c5['dcovar'], 'b5': c5['b'],
                       'ci': covar_ci(R[INDEX[reg]], R[k]) if boot else [np.nan, np.nan]}
    return rows


def fig_covar_qr(save=True):
    """JPMorgan Chase and the S&P 500: quantile regression lines at 1% and 50%, the 1% and median days of JPM."""
    R = region_returns('US')
    x, y = R['JPM'], R['sp500']
    c = covar(y, x, 0.01)
    a50, b50 = qreg(y, x, 0.5)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.scatter(x, y, s=5, color=st.MainBlue, alpha=0.35, label='daily returns, JPMorgan Chase and S&P 500')
    g = np.linspace(x.min(), x.max(), 100)
    ax.plot(g, c['a'] + c['b'] * g, color=st.IDAred, lw=2, label=f'1% quantile regression: {c["a"]:.2f} + {c["b"]:.2f} x')
    ax.plot(g, a50 + b50 * g, color=st.Forest, lw=2, label=f'median regression: {a50:.2f} + {b50:.2f} x')
    ax.axvline(c['q_a'], color=st.Orange, ls='--', lw=1.3, label=f'1% quantile of JPM: {c["q_a"]:.2f}%')
    ax.axvline(c['q_m'], color=st.Purple, ls='--', lw=1.3, label=f'median of JPM: {c["q_m"]:.2f}%')
    ax.plot([c['q_a']], [-c['covar']], 'o', color=st.IDAred, ms=8, label=f'minus CoVaR 1%: {-c["covar"]:.2f}%')
    ax.plot([c['q_m']], [-c['covar_med']], 's', color=st.Forest, ms=8, label=f'minus CoVaR 1% at the median: {-c["covar_med"]:.2f}%')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-13, 8)
    ax.set_xlabel('JPMorgan Chase daily log return (%)')
    ax.set_ylabel('S&P 500 daily log return (%)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_covar_qr')
    return {**c, 'a50': float(a50), 'b50': float(b50), 'n': len(R), 'start': R.index[0].date().isoformat()}


def fig_dcovar(t, save=True):
    """Delta-CoVaR 1% of each bank with its 95% block-bootstrap interval."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ks = list(t)
    xs = np.arange(len(ks))
    v = np.array([t[k]['dcovar'] for k in ks])
    lo = np.array([t[k]['ci'][0] for k in ks])
    hi = np.array([t[k]['ci'][1] for k in ks])
    ax.bar(xs, v, color=[REG_COL[t[k]['region']] for k in ks], width=0.62)
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt='none', ecolor='black', capsize=3, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([BANKS[k][1] for k in ks], rotation=30, ha='right', fontsize=10.5)
    ax.set_ylabel('Delta-CoVaR 1% (%)')
    h = [plt.Rectangle((0, 0), 1, 1, color=REG_COL[r]) for r in ('US', 'EU', 'RO')]
    lab = ['US banks (system: S&P 500)', 'European banks (system: Euro Stoxx 50)', 'Romanian banks (system: BET)']
    h.append(plt.Line2D([], [], color='black', marker='|', ls='', ms=10))
    lab.append('95% block-bootstrap interval')
    ax.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_dcovar')


def fig_var_vs_dcovar(t, save=True):
    """VaR 1% of each bank against its Delta-CoVaR 1%."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for reg in ('US', 'EU', 'RO'):
        ks = [k for k in t if t[k]['region'] == reg]
        ax.scatter([t[k]['var_i'] for k in ks], [t[k]['dcovar'] for k in ks], s=50, color=REG_COL[reg], label=REG_LAB[reg])
        for k in ks:
            ax.annotate(k, (t[k]['var_i'], t[k]['dcovar']), xytext=(4, 3), textcoords='offset points', fontsize=10, color='black')
    ax.set_xlabel('VaR 1% of the bank (%)')
    ax.set_ylabel('Delta-CoVaR 1% (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_var_vs_dcovar')
    rho = stats.spearmanr([t[k]['var_i'] for k in t], [t[k]['dcovar'] for k in t])[0]
    return float(rho)


def dynamic_dcovar(k, alpha=ALPHA):
    """Time-varying Delta-CoVaR (Adrian and Brunnermeier, 2016, Section II) with the lagged state variables
    M_{t-1} = (VIX level, index return): quantile regressions x_i = c + g'M (levels alpha and 50%) and
    x_sys = a + b x_i + g'M (level alpha); Delta-CoVaR_t = b (q50_t - qalpha_t)."""
    reg = BANKS[k][2]
    R = region_returns(reg)
    vix = load_close('vix', START, END)
    D = pd.concat([R[[k, INDEX[reg]]], vix.reindex(R.index).ffill().rename('vix')], axis=1).dropna()
    M = pd.concat([D['vix'].shift(1), D[INDEX[reg]].shift(1)], axis=1).iloc[1:]
    D = D.iloc[1:]
    ca, cm = qreg(D[k], M, alpha), qreg(D[k], M, 0.5)
    cs = qreg(D[INDEX[reg]], np.column_stack([D[k], M]), alpha)
    Mc = sm.add_constant(M.values, has_constant='add')
    qa, qm = Mc @ ca, Mc @ cm
    return pd.Series(cs[1] * (qm - qa), index=D.index, name=k), float(cs[1])


def fig_dcovar_dynamic(save=True):
    """Time-varying Delta-CoVaR 1% of JPMorgan Chase, Deutsche Bank and Banca Transilvania."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    out = {}
    for k, c in [('JPM', REG_COL['US']), ('DBK', REG_COL['EU']), ('TLV', REG_COL['RO'])]:
        s, b = dynamic_dcovar(k)
        ax.plot(s.index, s, color=c, lw=0.9, label=f'{BANKS[k][1]}')
        out[k] = {'mean': float(s.mean()), 'max': float(s.max()), 'date_max': s.idxmax().date().isoformat(),
                  'b': b, 'y2017': float(s.loc['2017'].mean()), 'last': float(s.iloc[-1])}
    for d, lab in EVENTS:
        ax.axvline(pd.Timestamp(d), color=st.Purple, ls=':', lw=1.0, label='_')
    ax.set_ylabel('Delta-CoVaR 1% (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_dcovar_dynamic')
    return out


# =============================================================================
# 4. MES AND SRISK
# =============================================================================
def mes(x_i, x_m, alpha=ALPHA_MES):
    """MES (%): minus the average return of the bank on the market's worst alpha of days."""
    x_i, x_m = np.asarray(x_i, float), np.asarray(x_m, float)
    return float(-x_i[x_m <= np.quantile(x_m, alpha)].mean())


def mes_threshold(x_i, x_m, c=-2.0):
    """MES (%) on the days when the market falls by more than |c|% (the input of LRMES)."""
    x_i, x_m = np.asarray(x_i, float), np.asarray(x_m, float)
    return float(-x_i[x_m < c].mean())


def lrmes(mes2):
    """Long-run MES, the loss of the bank if the market falls 40% in six months: 1 - exp(-18 MES), MES in %."""
    return float(1 - np.exp(-18 * mes2 / 100))


def srisk(debt, equity, lrm, k=K_CAP):
    """SRISK = k D - (1 - k)(1 - LRMES) W: the capital shortfall in a crisis (positive = missing capital)."""
    return k * debt - (1 - k) * (1 - lrm) * equity


def srisk_ratio(lrm, leverage, k=K_CAP):
    """SRISK / W with leverage L = (D + W) / W: k (L - 1) - (1 - k)(1 - LRMES)."""
    return k * (np.asarray(leverage) - 1) - (1 - k) * (1 - lrm)


def breakeven_leverage(lrm, k=K_CAP):
    """Leverage above which SRISK > 0: L* = 1 + (1 - k)(1 - LRMES) / k."""
    return float(1 + (1 - k) * (1 - lrm) / k)


def mes_table(B=999, seed=SEED):
    """MES 5% with a moving-block bootstrap interval, MES at the -2% threshold, LRMES and the break-even leverage."""
    rng = np.random.default_rng(seed)
    rows = {}
    for reg in ('US', 'EU', 'RO'):
        R = region_returns(reg)
        m = R[INDEX[reg]].values
        for k in REGIONS[reg]:
            x = R[k].values
            boot = [mes(x[i], m[i]) for i in (block_idx(len(x), BLOCK, rng) for _ in range(B))]
            m2 = mes_threshold(x, m)
            beta = float(np.cov(x, m)[0, 1] / np.var(m, ddof=1))
            rows[k] = {'region': reg, 'mes5': mes(x, m), 'lo': float(np.percentile(boot, 2.5)),
                       'hi': float(np.percentile(boot, 97.5)), 'mes2': m2, 'n2': int((m < -2).sum()),
                       'lrmes': lrmes(m2), 'Lstar': breakeven_leverage(lrmes(m2)), 'beta': beta,
                       'es5_m': float(-m[m <= np.quantile(m, 0.05)].mean()), 'n': len(x)}
    return rows


def fig_mes(t, save=True):
    """MES 5% of each bank with its 95% block-bootstrap interval."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ks = list(t)
    xs = np.arange(len(ks))
    v = np.array([t[k]['mes5'] for k in ks])
    ax.bar(xs, v, color=[REG_COL[t[k]['region']] for k in ks], width=0.62)
    ax.errorbar(xs, v, yerr=[v - np.array([t[k]['lo'] for k in ks]), np.array([t[k]['hi'] for k in ks]) - v],
                fmt='none', ecolor='black', capsize=3, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([BANKS[k][1] for k in ks], rotation=30, ha='right', fontsize=10.5)
    ax.set_ylabel('MES 5% (daily loss, %)')
    h = [plt.Rectangle((0, 0), 1, 1, color=REG_COL[r]) for r in ('US', 'EU', 'RO')]
    lab = ['US banks (market: S&P 500)', 'European banks (market: Euro Stoxx 50)', 'Romanian banks (market: BET)']
    h.append(plt.Line2D([], [], color='black', marker='|', ls='', ms=10))
    lab.append('95% block-bootstrap interval')
    ax.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_mes')


def fig_srisk(t, save=True):
    """SRISK per unit of market equity as a function of leverage, for the banks with the largest, median and smallest
    LRMES."""
    order = sorted(t, key=lambda k: t[k]['lrmes'])
    pick = [order[-1], order[len(order) // 2], order[0]]
    L = np.linspace(1, 25, 200)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for k, c in zip(pick, [st.IDAred, st.Amber, st.MainBlue]):
        lr = t[k]['lrmes']
        ax.plot(L, 100 * srisk_ratio(lr, L), color=c, lw=1.8,
                label=f'{BANKS[k][1]}: LRMES {100 * lr:.0f}%, break-even leverage {t[k]["Lstar"]:.1f}')
        ax.plot(t[k]['Lstar'], 0, 'o', color=c, ms=7, label='_')
    ax.axhline(0, color=st.Purple, lw=1.0, ls=':', label='SRISK = 0')
    ax.set_xlabel('leverage L = (debt + market value of equity) / market value of equity')
    ax.set_ylabel('SRISK / market value of equity (%)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_srisk')
    return pick


def mes_vs_crisis(pre=('2006-06-01', '2007-06-30'), crisis=('2007-07-01', '2008-12-31'), alpha=ALPHA_MES):
    """MES measured before a crisis against the realised (buy-and-hold) return of the bank in the crisis
    (the design of Acharya et al., 2017: MES from June 2006 to June 2007, returns from July 2007 to December 2008)."""
    out = {}
    for reg in ('US', 'EU'):
        R = region_returns(reg, start=pre[0], end=crisis[1])
        P = prices(REGIONS[reg], start=pre[0], end=crisis[1])
        a = R.loc[pre[0]:pre[1]]
        for k in REGIONS[reg]:
            p = P[k].loc[crisis[0]:crisis[1]]
            p0 = P[k].loc[:pre[1]].iloc[-1]
            out[k] = {'region': reg, 'mes': mes(a[k], a[INDEX[reg]], alpha), 'ret': float(100 * (p.iloc[-1] / p0 - 1)),
                      'n_pre': len(a)}
    xs = [out[k]['mes'] for k in out]
    ys = [out[k]['ret'] for k in out]
    sp = stats.spearmanr(xs, ys)
    return out, {'spearman': float(sp[0]), 'p': float(sp[1]), 'n': len(out)}


def fig_mes_crisis(res=None, save=True, name='sfm_ch15_mes_crisis', title=None):
    """Scatter: MES 5% before the crisis against the realised return in the crisis."""
    out, s = mes_vs_crisis() if res is None else res
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for reg in ('US', 'EU', 'RO'):
        ks = [k for k in out if out[k]['region'] == reg]
        if not ks:
            continue
        ax.scatter([out[k]['mes'] for k in ks], [out[k]['ret'] for k in ks], s=55, color=REG_COL[reg], label=REG_LAB[reg])
        for k in ks:
            ax.annotate(k, (out[k]['mes'], out[k]['ret']), xytext=(4, 3), textcoords='offset points', fontsize=10, color='black')
    x = np.array([out[k]['mes'] for k in out])
    y = np.array([out[k]['ret'] for k in out])
    b, a = np.polyfit(x, y, 1)
    g = np.linspace(x.min(), x.max(), 10)
    ax.plot(g, a + b * g, color=st.Orange, lw=1.5, ls='--', label=f'least-squares line; Spearman correlation {s["spearman"]:.2f}')
    ax.set_xlabel('MES 5% before the crisis (%)')
    ax.set_ylabel('return in the crisis (%)')
    if title:
        ax.set_title(title, fontsize=11)
    st.legend_outside_bottom(ax, ncol=2, y=-0.17)
    st.check_no_grey(fig)
    if save:
        st.save_fig(name)
    return s


# =============================================================================
# 5. NETWORKS
# =============================================================================
def monthly_returns():
    """Monthly log returns in % of the 11 US and European banks (last common trading day of each month)."""
    P = prices(INTL)
    Pm = P.groupby(P.index.to_period('M')).last()
    return (100 * np.log(Pm).diff()).dropna()


def granger_p(y, x):
    """p-value of the test that x does not Granger-cause y: regression y_t = a + b y_{t-1} + c x_{t-1} + e_t, t test of c."""
    Y = y[1:]
    X = sm.add_constant(np.column_stack([y[:-1], x[:-1]]))
    return float(sm.OLS(Y, X).fit().pvalues[2])


def granger_network(Rm, level=0.05):
    """Matrix of significant Granger-causal links (row i causes column j) and the degree of Granger causality (DGC)."""
    ks = list(Rm.columns)
    A = np.zeros((len(ks), len(ks)), int)
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            if i != j:
                A[i, j] = int(granger_p(Rm[b].values, Rm[a].values) < level)
    n = len(ks)
    return A, float(A.sum() / (n * (n - 1)))


def rolling_dgc(Rm=None, window=GC_WINDOW):
    """DGC on rolling windows of 36 months."""
    Rm = monthly_returns() if Rm is None else Rm
    out = {}
    for e in range(window, len(Rm) + 1):
        out[Rm.index[e - 1].to_timestamp('M')] = granger_network(Rm.iloc[e - window:e])[1]
    return pd.Series(out)


def fig_dgc(save=True):
    """Degree of Granger causality among the 11 US and European banks, 36-month windows; 5% line = links expected by chance."""
    Rm = monthly_returns()
    d = rolling_dgc(Rm)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d.index, 100 * d, color=st.MainBlue, lw=1.5, label='degree of Granger causality (DGC), 36-month windows')
    ax.axhline(5, color=st.IDAred, ls='--', lw=1.2, label='5%: share of links expected by chance at the 5% level')
    for dd, lab in EVENTS:
        ax.axvline(pd.Timestamp(dd), color=st.Purple, ls=':', lw=1.0, label='_')
        ax.text(pd.Timestamp(dd), ax.get_ylim()[1], ' ' + lab, rotation=90, va='top', ha='right', fontsize=10, color='black')
    ax.set_ylabel('significant links (%)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_dgc')
    return {'first': d.index[0].date().isoformat(), 'max': float(100 * d.max()), 'date_max': d.idxmax().date().isoformat(),
            'min': float(100 * d.min()), 'date_min': d.idxmin().date().isoformat(), 'mean': float(100 * d.mean()),
            'last': float(100 * d.iloc[-1]), 'n_months': len(Rm), 'start': str(Rm.index[0]),
            'd2009': float(100 * d.loc['2009-12-31']), 'd2019': float(100 * d.loc['2019-12-31'])}


def fig_granger_net(ends=('2009-12', '2019-12'), save=True):
    """The Granger-causality network of the 11 US and European banks in two 36-month windows (arrows: significant links at 5%)."""
    Rm = monthly_returns()
    ks = list(Rm.columns)
    ang = np.pi / 2 - 2 * np.pi * np.arange(len(ks)) / len(ks)
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    out = {}
    for ax, e in zip(axes, ends):
        j = list(Rm.index.astype(str)).index(e) + 1
        A, dgc = granger_network(Rm.iloc[j - GC_WINDOW:j])
        for a in range(len(ks)):
            for b in range(len(ks)):
                if A[a, b]:
                    ax.annotate('', xy=pos[b] * 0.9, xytext=pos[a] * 0.9,
                                arrowprops=dict(arrowstyle='->', color=REG_COL[BANKS[ks[a]][2]], lw=0.9, alpha=0.8))
        for i, k in enumerate(ks):
            ax.scatter(*pos[i], s=380, color=REG_COL[BANKS[k][2]], zorder=3)
            ax.text(*(pos[i] * 1.22), k, ha='center', va='center', fontsize=10, color='black')
        a0 = Rm.index[j - GC_WINDOW].strftime('%b %Y')
        ax.set_title(f'{a0} – {Rm.index[j - 1].strftime("%b %Y")}: {A.sum()} links, DGC {100 * dgc:.0f}%', fontsize=11)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect('equal')
        ax.axis('off')
        out[e] = {'links': int(A.sum()), 'dgc': float(100 * dgc), 'out': {k: int(A[i].sum()) for i, k in enumerate(ks)}}
    h = [plt.Line2D([], [], color=REG_COL[r], marker='o', ls='-', ms=9) for r in ('US', 'EU')]
    fig.legend(h, [REG_LAB[r] + ' (arrows: links from this bank)' for r in ('US', 'EU')], loc='upper center',
               bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_granger_net')
    return out


def var_fit(Y, p):
    """VAR(p) by OLS: lag matrices A_1..A_p and the residual covariance matrix."""
    Y = np.asarray(Y, float)
    T, k = Y.shape
    X = np.hstack([np.ones((T - p, 1))] + [Y[p - l - 1:T - l - 1] for l in range(p)])
    B = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
    E = Y[p:] - X @ B
    S = E.T @ E / (len(E) - X.shape[1])
    return [B[1 + l * k:1 + (l + 1) * k].T for l in range(p)], S


def var_bic(Y, pmax=4):
    """VAR order by the Bayesian information criterion."""
    Y = np.asarray(Y, float)
    T, k = Y.shape
    n = T - pmax
    best = None
    for p in range(1, pmax + 1):
        A, S = var_fit(Y[pmax - p:], p)
        bic = np.log(np.linalg.det(S * (n - 1 - p * k) / n)) + np.log(n) * p * k * k / n
        if best is None or bic < best[1]:
            best = (p, bic)
    return best[0]


def gfevd(A, S, H=DY_H):
    """Generalized forecast error variance decomposition (Pesaran and Shin, 1998), rows normalised to 1."""
    k = S.shape[0]
    Phi = [np.eye(k)]
    for h in range(1, H):
        Phi.append(sum(A[l] @ Phi[h - l - 1] for l in range(min(h, len(A)))))
    num, den = np.zeros((k, k)), np.zeros(k)
    for P in Phi:
        num += (P @ S) ** 2
        den += np.diag(P @ S @ P.T)
    th = num / np.diag(S)[None, :] / den[:, None]
    return th / th.sum(axis=1, keepdims=True)


def connectedness(theta, names):
    """Diebold-Yilmaz table: shares in %, FROM others, TO others, NET and the total connectedness index."""
    D = 100 * theta
    off = D - np.diag(np.diag(D))
    frm, to = off.sum(axis=1), off.sum(axis=0)
    return {'table': pd.DataFrame(D, index=names, columns=names), 'from': pd.Series(frm, index=names),
            'to': pd.Series(to, index=names), 'net': pd.Series(to - frm, index=names), 'total': float(off.sum() / len(names))}


def dy_static():
    """Connectedness of the daily returns of the 11 US and European banks (common days since 2005)."""
    R = returns(INTL)
    p = var_bic(R.values)
    A, S = var_fit(R.values, p)
    c = connectedness(gfevd(A, S), list(R.columns))
    return c, p, len(R)


def fig_dy_table(c, save=True):
    """Heat map of the connectedness table (shares of forecast error variance, %); last row: TO others."""
    T = c['table']
    ks = list(T.index)
    M = np.vstack([T.values, c['to'].values])
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.imshow(np.vstack([T.values, np.full(len(ks), np.nan)]), cmap='Blues', vmin=0, vmax=np.ceil(T.values.max() / 10) * 10)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ks) + 1))
    ax.set_yticklabels([f'{k}  (FROM {c["from"][k]:.0f})' for k in ks] + ['TO others'], fontsize=10.5)
    for i in range(len(ks) + 1):
        for j in range(len(ks)):
            v = M[i, j]
            col = st.IDAred if i == len(ks) else ('white' if v > 0.55 * T.values.max() else 'black')
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9.5, color=col)
    ax.set_xlabel(f'shock from (column) to (row), % of the forecast error variance\ntotal connectedness {c["total"]:.1f}%',
                  fontsize=11)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_dy_table')


def rolling_total(R=None, window=DY_WINDOW, step=DY_STEP, p=None):
    """Total connectedness on rolling windows of 200 days, moved by 5 days."""
    R = returns(INTL) if R is None else R
    p = var_bic(R.values) if p is None else p
    out = {}
    for e in range(window, len(R) + 1, step):
        A, S = var_fit(R.values[e - window:e], p)
        out[R.index[e - 1]] = connectedness(gfevd(A, S), list(R.columns))['total']
    return pd.Series(out)


def fig_dy_rolling(save=True):
    """Total connectedness of the 11 US and European banks, 200-day windows."""
    tot = rolling_total()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(tot.index, tot, color=st.MainBlue, lw=1.3, label='total connectedness, 200-day windows (%)')
    for dd, lab in EVENTS:
        ax.axvline(pd.Timestamp(dd), color=st.Purple, ls=':', lw=1.0, label='_')
        ax.text(pd.Timestamp(dd), ax.get_ylim()[1], ' ' + lab, rotation=90, va='top', ha='right', fontsize=10, color='black')
    ax.set_ylabel('total connectedness (%)')
    st.legend_outside_bottom(ax, ncol=1, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch15_dy_rolling')
    return {'min': float(tot.min()), 'date_min': tot.idxmin().date().isoformat(), 'max': float(tot.max()),
            'date_max': tot.idxmax().date().isoformat(), 'last': float(tot.iloc[-1]), 'mean': float(tot.mean()),
            'y2017': float(tot.loc['2017'].mean()), 'y2008': float(tot.loc['2008-10-01':'2009-03-31'].mean())}


if __name__ == '__main__':
    st.apply()
    N = {}
    N['indices'] = fig_bank_indices()
    N['episodes'] = fig_episodes()
    N['corr'] = fig_rolling_corr()
    N['fr'] = contagion_test()
    N['tail'] = tail_table()
    N['tail_fig'] = fig_tail_scatter()
    N['qr'] = fig_covar_qr()
    CT = covar_table()
    N['covar'] = CT
    fig_dcovar(CT)
    N['spearman_var_dcovar'] = fig_var_vs_dcovar(CT)
    N['spearman_1_5'] = float(stats.spearmanr([CT[k]['dcovar'] for k in CT], [CT[k]['dcovar5'] for k in CT])[0])
    N['dyn'] = fig_dcovar_dynamic()
    MT = mes_table()
    N['mes'] = MT
    fig_mes(MT)
    N['srisk_pick'] = fig_srisk(MT)
    N['spearman_mes_dcovar'] = float(stats.spearmanr([MT[k]['mes5'] for k in ALL], [CT[k]['dcovar'] for k in ALL])[0])
    res = mes_vs_crisis()
    N['mes_crisis'] = {'banks': res[0], **res[1]}
    fig_mes_crisis(res)
    N['dgc'] = fig_dgc()
    N['gnet'] = fig_granger_net()
    c, p, n = dy_static()
    N['dy'] = {'p': p, 'n': n, 'total': c['total'], 'to': c['to'].to_dict(), 'from': c['from'].to_dict(),
               'net': c['net'].to_dict()}
    fig_dy_table(c)
    N['dy_roll'] = fig_dy_rolling()
    N['end'] = END
    pd.DataFrame(CT).T.to_csv(os.path.join(TABLE_DIR, 'ch15_covar_table.csv'), float_format='%.4f')
    pd.DataFrame(MT).T.to_csv(os.path.join(TABLE_DIR, 'ch15_mes_table.csv'), float_format='%.4f')
    with open(os.path.join(TABLE_DIR, 'ch15_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
