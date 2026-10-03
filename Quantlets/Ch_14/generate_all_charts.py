"""
generate_all_charts.py -- charts and numbers of Chapter 14 (SFM): crypto assets and stablecoins
================================================================================================
Course data (sfm_data.py): daily closes of Bitcoin, Ethereum, Solana, the stablecoins USDT, USDC and DAI, the
S&P 500, gold (XAU/USD) and the iShares Bitcoin Trust ETF (IBIT), all from EODHD; stablecoin supply from DefiLlama
(public API, no key). Chart style: sfm_style.py. Every number on the slides comes from here.
  * blockchain  -- the Bitcoin supply rule (block subsidy halved every 210 000 blocks) and the price at the halvings;
  * asset class -- returns, volatility and tails of crypto against the S&P 500 and gold; annualisation with the
                   actual frequency (365 days a year for crypto, about 252 for equities); weekday effect;
  * dynamics    -- volatility clustering (ACF of squared returns), GARCH(1,1)-t and GJR-GARCH (Chapter 9);
                   variance-ratio tests on the whole sample, in sub-periods and on rolling windows (Chapter 7);
  * dependence  -- rolling correlation with equities and gold (prices joined on common days), the 2020 and 2022
                   crisis episodes (COVID-19, Terra/Luna, FTX), returns on the worst days of the S&P 500;
  * bubbles     -- drawdowns and their recovery;
  * spot ETFs   -- volatility before and after 11 January 2024, the weekend share of variance, IBIT against Bitcoin;
  * stablecoins -- supply (DefiLlama), peg deviations in basis points, the USDC depeg of March 2023, the collapse
                   of TerraUSD (UST) in May 2022;
  * risk        -- VaR 1% and ES 2.5% of a position of 100 000 USD (VaR_alpha = -q_alpha, the loss exceeded with
                   probability alpha); a rolling backtest of Bitcoin VaR 1%;
  * AI section  -- a starter result: does the weekly change of stablecoin supply predict next week's Bitcoin return?
Output: charts/sfm_ch14_*.pdf/.png, Quantlets/Ch_14/ch14_numbers.json and ch14_stats_table.csv
Based on Franke, Haerdle and Hafner (2019), Statistics of Financial Markets, 5th ed., and on Liu and Tsyvinski (2021),
Makarov and Schoar (2020), Griffin and Shams (2020), Lyons and Viswanath-Natraj (2023), Gorton et al. (2022).
Run:  python3 Quantlets/Ch_14/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import SERIES, END, load_close, log_returns, read_market, periods_per_year   # noqa: E402
import sfm_style as st                               # noqa: E402
from arch import arch_model                          # noqa: E402

TABLE_DIR = HERE
# series not in the default list of sfm_data: (symbol, label, group, field, default start)
EXTRA = {'usdc': ('USDC-USD.CC', 'USD Coin (USDC)', 'Crypto', 'close', '2018-10-09'),
         'dai': ('DAI-USD.CC', 'Dai (DAI)', 'Crypto', 'close', '2017-12-27'),
         'ibit': ('IBIT.US', 'iShares Bitcoin Trust (IBIT)', 'ETF', 'adjusted_close', '2024-01-11')}
NAME = {'btc': 'Bitcoin', 'eth': 'Ethereum', 'sol': 'Solana', 'sp500': 'S&P 500', 'gold': 'Gold',
        'usdt': 'Tether (USDT)', 'usdc': 'USD Coin (USDC)', 'dai': 'Dai (DAI)', 'ibit': 'IBIT'}
START = {'btc': '2014-09-17', 'eth': '2016-01-01', 'sol': '2020-04-11', 'sp500': '2014-09-17', 'gold': '2014-09-17'}
ASSETS = ['btc', 'eth', 'sol', 'sp500', 'gold']
COLORS = {'btc': '#E67E22', 'eth': '#8E44AD', 'sol': '#17A2B8', 'sp500': '#1A3A6E', 'gold': '#B5853F',
          'usdt': '#2E7D32', 'usdc': '#1A3A6E', 'dai': '#E67E22', 'total': '#CD0000', 'ibit': '#1A3A6E'}
HALVINGS = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20']   # blocks 210 000, 420 000, 630 000, 840 000
ETF_DATE = '2024-01-11'      # first trading day of the US spot Bitcoin ETFs (SEC approval on 10 January 2024)
EVENTS = {'COVID-19': '2020-03-12', 'Terra/Luna': '2022-05-09', 'FTX': '2022-11-08', 'spot ETFs': ETF_DATE}
EPISODES = {'COVID-19': ('2020-02-19', '2020-03-23'), 'Terra/Luna': ('2022-05-05', '2022-05-12'),
            'FTX': ('2022-11-05', '2022-11-21')}
HILL_FRAC = 0.025            # Hill estimator with k = 2.5% of n (Chapter 5)
VR_Q = (2, 5, 10)            # variance-ratio horizons (Chapter 7)
VR_WINDOW, VR_STEP = 500, 21  # rolling windows of 500 observations, one every 21 observations (Chapter 7)
SUBPERIODS = [('2014-09-17', '2017-12-31'), ('2018-01-01', '2021-12-31'), ('2022-01-01', END)]
CORR_WINDOW = 250            # rolling correlation on 250 common trading days (about one year)
ALPHA_VAR, ALPHA_ES = 0.01, 0.025
POSITION = 100_000           # USD
BT_START = '2018-01-01'      # rolling backtest of Bitcoin VaR 1%
BT_WINDOW = 365              # historical simulation on the last 365 days; GARCH-t re-estimated every 365 days
PEG_START = '2021-01-01'
SVB = ('2023-03-08', '2023-03-20')
LLAMA = 'https://stablecoins.llama.fi/stablecoincharts/all'
STABLE_ID = {'usdt': 1, 'usdc': 2, 'ust': 3, 'dai': 5}
SEED = 2026
_LLAMA = {}


# =============================================================================
# DATA
# =============================================================================
def register_extra():
    """Add USDC, DAI and the IBIT ETF to the list of named series of the course data loader."""
    for k, v in EXTRA.items():
        SERIES.setdefault(k, v)


def close(k, start=None, end=END):
    """Daily close (adjusted close for the ETF) on the series' own calendar (crypto: 7 days a week)."""
    register_extra()
    return load_close(k, start or START.get(k), end)


def returns(k, start=None, end=END):
    """Daily log returns in % on the series' own calendar."""
    register_extra()
    return log_returns(k, start or START.get(k), end)


def joint_returns(a, b, start=None, end=END):
    """Log returns in % of two series on their common days: prices are joined first, then differenced."""
    p = pd.concat([close(a, start, end), close(b, start, end)], axis=1).dropna()
    return (100 * np.log(p).diff()).dropna()


def stablecoin_chart(sid=None):
    """DefiLlama: circulating supply of USD stablecoins, billion USD (sid=None: all; 1 USDT, 2 USDC, 3 UST, 5 DAI);
    columns 'supply' (billion units) and 'value' (billion USD)."""
    if sid in _LLAMA:
        return _LLAMA[sid]
    url = LLAMA + (f'?stablecoin={sid}' if sid else '')
    js = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla'}),
                                           timeout=120).read())
    rows = {}
    for x in js:
        d = pd.Timestamp(int(x['date']), unit='s').normalize()
        c = (x.get('totalCirculating') or {}).get('peggedUSD')
        v = (x.get('totalCirculatingUSD') or {}).get('peggedUSD')
        if c is not None:
            rows[d] = (c / 1e9, (v if v is not None else np.nan) / 1e9)
    df = pd.DataFrame(rows, index=['supply', 'value']).T.sort_index().loc[:END]
    _LLAMA[sid] = df
    return df


# =============================================================================
# 1. BITCOIN: SUPPLY RULE AND HALVINGS
# =============================================================================
def btc_supply_schedule(years=np.arange(2009, 2041)):
    """Bitcoin supply at the start of each year from the protocol rule: 50 BTC per block, halved every 210 000
    blocks, about 52 560 blocks a year (one block every 10 minutes), first block on 3 January 2009."""
    blocks = (years - 2009) * 52560
    out = []
    for b in blocks:
        s, left, sub = 0.0, b, 50.0
        while left > 0 and sub > 1e-8:
            take = min(left, 210000)
            s += take * sub
            left -= take
            sub /= 2
        out.append(s / 1e6)
    return pd.Series(out, index=years)


def fig_halving(save=True):
    """Left: Bitcoin price since 2014 (log scale) with the halving dates. Right: supply from the protocol rule."""
    p = close('btc')
    sup = btc_supply_schedule()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={'width_ratios': [1.5, 1]})
    axes[0].plot(p.index, p.values, color=COLORS['btc'], lw=1.3, label='Bitcoin price (USD, log scale)')
    axes[0].set_yscale('log')
    for i, h in enumerate(HALVINGS[1:]):
        axes[0].axvline(pd.Timestamp(h), color=st.IDAred, lw=1.0, ls='--', label='Halving' if i == 0 else '_h')
    axes[0].set_ylabel('USD')
    axes[1].plot(sup.index, sup.values, color=st.MainBlue, lw=1.6, label='Bitcoin supply (million BTC)')
    axes[1].axhline(21, color=st.Forest, lw=1.0, ls=':', label='Limit: 21 million')
    axes[1].set_ylabel('Million BTC')
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_halving')
    out = {}
    for h in HALVINGS[1:]:
        t = pd.Timestamp(h)
        a = p.loc[:t].iloc[-1]
        b = p.loc[:t + pd.Timedelta(days=365)].iloc[-1] if t + pd.Timedelta(days=365) <= p.index[-1] else np.nan
        out[h[:4]] = {'price': float(a), 'price_1y': float(b), 'change_1y': float(100 * (b / a - 1))}
    return {'halvings': out, 'price_end': float(p.iloc[-1]), 'price_start': float(p.iloc[0]),
            'supply_2024': float(btc_supply_schedule(np.array([2024]))[2024]),
            'supply_2030': float(btc_supply_schedule(np.array([2030]))[2030]),
            'max_price': float(p.max()), 'max_date': p.idxmax().date().isoformat()}


# =============================================================================
# 2. CRYPTO AS AN ASSET CLASS: RETURNS, VOLATILITY, TAILS
# =============================================================================
def hill(x, k):
    """Hill (1975): alpha_hat(k) = [ (1/k) sum_{i=1..k} ln(X_(i) / X_(k+1)) ]^(-1), X_(1) >= X_(2) >= ... (Chapter 5)."""
    x = np.sort(np.asarray(x, float))[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def describe(k, start=None, end=END):
    """Descriptive statistics of daily log returns in %, with annualisation at the actual frequency of the series;
    Hill tail indices of the losses (left tail) and of the gains (right tail), k = 2.5% of n."""
    r = returns(k, start, end)
    x = r.values
    n = len(x)
    ppy = periods_per_year(r)
    kk = int(HILL_FRAC * n)
    al, ar = hill(-x[x < 0], kk), hill(x[x > 0], kk)
    return {'n': n, 'first': r.index[0].date().isoformat(), 'ppy': float(ppy), 'mean': float(x.mean()),
            'sd': float(x.std(ddof=1)), 'skew': float(stats.skew(x)), 'exkurt': float(stats.kurtosis(x)),
            'min': float(x.min()), 'min_date': r.idxmin().date().isoformat(), 'max': float(x.max()),
            'max_date': r.idxmax().date().isoformat(), 'ann_mean': float(ppy * x.mean()),
            'ann_vol': float(np.sqrt(ppy) * x.std(ddof=1)), 'ann_vol_252': float(np.sqrt(252) * x.std(ddof=1)),
            'ann_mean_252': float(252 * x.mean()), 'share5': float(100 * np.mean(np.abs(x) > 5)),
            'k': kk, 'hill_left': float(al), 'hill_right': float(ar),
            'hill_left_lo': float(al * (1 - 1.96 / np.sqrt(kk))), 'hill_left_hi': float(al * (1 + 1.96 / np.sqrt(kk))),
            'hill_right_lo': float(ar * (1 - 1.96 / np.sqrt(kk))), 'hill_right_hi': float(ar * (1 + 1.96 / np.sqrt(kk))),
            'jb': float(stats.jarque_bera(x)[0])}


def stats_table(names=ASSETS):
    out = {k: describe(k) for k in names}
    pd.DataFrame(out).T.to_csv(os.path.join(TABLE_DIR, 'ch14_stats_table.csv'), float_format='%.4f')
    return out


def fig_rolling_vol(save=True):
    """Volatility on rolling windows of about one month, annualised at the actual frequency: 30 daily returns
    times sqrt(365) for Bitcoin and Ethereum, 21 daily returns times sqrt(252) for the S&P 500."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    out = {}
    for k, w, ppy in [('btc', 30, 365), ('eth', 30, 365), ('sp500', 21, 252)]:
        r = returns(k, '2016-01-01')
        v = r.rolling(w).std() * np.sqrt(ppy)
        ax.plot(v.index, v.values, color=COLORS[k], lw=1.1, label=f'{NAME[k]} ({w} returns, x sqrt({ppy}))')
        v = v.dropna()
        out[k] = {'median': float(v.median()), 'max': float(v.max()), 'max_date': v.idxmax().date().isoformat(),
                  'last': float(v.iloc[-1]), 'min': float(v.min()), 'min_date': v.idxmin().date().isoformat()}
    ax.set_ylabel('Annualised volatility (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_rolling_vol')
    b, s = returns('btc', '2016-01-01').rolling(30).std() * np.sqrt(365), returns('sp500', '2016-01-01').rolling(21).std() * np.sqrt(252)
    j = pd.concat([b, s], axis=1).dropna()
    out['ratio_median'] = float((j.iloc[:, 0] / j.iloc[:, 1]).median())
    out['share_btc_below_sp'] = float(100 * np.mean(j.iloc[:, 0] < j.iloc[:, 1]))
    return out


DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
WD_PERIODS = {'2014-2019': ('2014-09-17', '2019-12-31'), '2020-2023': ('2020-01-01', '2023-12-31'),
              '2024-2026': ('2024-01-01', END)}


def hac_ols(y, X, lags=10):
    """OLS with Newey-West (HAC) standard errors."""
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})


def weekday_effect(k='btc', periods=WD_PERIODS):
    """Mean and standard deviation of daily returns by day of the week (the day of the closing price); the weekend
    dummy in r_t = a + b W_t + e_t with HAC standard errors; Brown-Forsythe test of equal variance on weekend and
    weekdays; the ratio of the weekend to the weekday variance."""
    out = {}
    for lab, (a, b) in periods.items():
        r = returns(k, a, b)
        d = pd.DataFrame({'r': r, 'dow': r.index.dayofweek})
        w = (d['dow'] >= 5).astype(float)
        reg = hac_ols(d['r'], w.rename('weekend'))
        bf = stats.levene(d.loc[w == 1, 'r'], d.loc[w == 0, 'r'], center='median')
        out[lab] = {'n': int(len(r)), 'mean': {DAYS[i]: float(g.mean()) for i, g in d.groupby('dow')['r']},
                    'sd': {DAYS[i]: float(g.std()) for i, g in d.groupby('dow')['r']},
                    'b': float(reg.params['weekend']), 't': float(reg.tvalues['weekend']), 'p': float(reg.pvalues['weekend']),
                    'bf': float(bf.statistic), 'p_bf': float(bf.pvalue),
                    'var_ratio': float(d.loc[w == 1, 'r'].var() / d.loc[w == 0, 'r'].var()),
                    'sd_we': float(d.loc[w == 1, 'r'].std()), 'sd_wd': float(d.loc[w == 0, 'r'].std())}
    return out


def fig_weekday(W=None, save=True):
    """Standard deviation of Bitcoin's daily returns by day of the week in three periods."""
    W = W or weekday_effect()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(7)
    cols = [st.MainBlue, st.Orange, st.IDAred]
    for i, (lab, d) in enumerate(W.items()):
        ax.bar(x + (i - 1) * 0.27, [d['sd'][day] for day in DAYS], width=0.27, color=cols[i], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    ax.set_ylabel('Std. dev. of daily returns (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.13)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_weekday')
    return W


# =============================================================================
# 3. VOLATILITY CLUSTERING AND GARCH (Chapter 9)
# =============================================================================
def acf(x, nlags):
    """Sample autocorrelations rho_hat(1..nlags)."""
    e = np.asarray(x, float) - np.mean(x)
    s = np.sum(e ** 2)
    return np.array([np.sum(e[k:] * e[:-k]) / s for k in range(1, nlags + 1)])


def ljung_box(x, m=10):
    """Ljung-Box Q(m) = T(T+2) sum rho_k^2/(T-k) against chi-square(m)."""
    x = np.asarray(x, float)
    T = len(x)
    rho = acf(x, m)
    q = T * (T + 2) * np.sum(rho ** 2 / (T - np.arange(1, m + 1)))
    return {'q': float(q), 'p': float(stats.chi2.sf(q, m))}


def fig_acf(k='btc', nlags=30, save=True):
    """ACF of the returns and of the squared returns of Bitcoin, with the bands +-1.96/sqrt(n)."""
    r = returns(k).values
    n = len(r)
    a1, a2 = acf(r, nlags), acf(r ** 2, nlags)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    lags = np.arange(1, nlags + 1)
    for ax, a, c, lab in [(axes[0], a1, st.MainBlue, 'ACF of returns'), (axes[1], a2, st.IDAred, 'ACF of squared returns')]:
        ax.bar(lags, a, color=c, width=0.6, label=lab)
        ax.axhline(1.96 / np.sqrt(n), color='black', lw=0.8, ls='--')
        ax.axhline(-1.96 / np.sqrt(n), color='black', lw=0.8, ls='--', label='_b')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Lag (days)')
    axes[0].plot([], [], color='black', ls='--', lw=0.8, label='+-1.96/sqrt(n)')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_acf')
    out = {'n': n, 'rho1': float(a1[0]), 'rho1_sq': float(a2[0]), 'rho10_sq': float(a2[9]), 'band': float(1.96 / np.sqrt(n)),
           'n_out_r': int(np.sum(np.abs(a1) > 1.96 / np.sqrt(n))), 'n_out_sq': int(np.sum(np.abs(a2) > 1.96 / np.sqrt(n)))}
    out['lb_r'] = ljung_box(r)
    out['lb_sq'] = ljung_box(r ** 2)
    sp = returns('sp500').values
    out['sp_lb_r'] = ljung_box(sp)
    out['sp_lb_sq'] = ljung_box(sp ** 2)
    return out


def garch_fit(r, o=0):
    """GARCH(1,1) (o=0) or GJR-GARCH(1,1) (o=1) with constant mean and standardised Student-t innovations."""
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='t')
    return am.fit(disp='off', options={'maxiter': 2000})


def garch_table(names=('btc', 'eth', 'sp500', 'gold')):
    """GARCH(1,1)-t estimates, persistence alpha + beta, half-life ln(0.5)/ln(alpha + beta) of a volatility shock,
    the unconditional volatility annualised at the actual frequency; GJR asymmetry gamma and its t-statistic."""
    out = {}
    for k in names:
        r = returns(k)
        ppy = periods_per_year(r)
        res = garch_fit(r)
        p, t = res.params, res.tvalues
        a, b = float(p['alpha[1]']), float(p['beta[1]'])
        pers = a + b
        gj = garch_fit(r, o=1)
        sig = res.conditional_volatility
        out[k] = {'n': int(len(r)), 'mu': float(p['mu']), 'omega': float(p['omega']), 'alpha': a, 'beta': b,
                  'nu': float(p['nu']), 't_alpha': float(t['alpha[1]']), 't_beta': float(t['beta[1]']),
                  'pers': pers, 'ppy': float(ppy),
                  # alpha + beta = 1 (integrated GARCH): no finite half-life, no unconditional variance
                  'half_life': float(np.log(0.5) / np.log(pers)) if pers < 0.9999 else None,
                  'uncond_ann': float(np.sqrt(ppy * p['omega'] / (1 - pers))) if pers < 0.9999 else None,
                  'gamma': float(gj.params['gamma[1]']), 't_gamma': float(gj.tvalues['gamma[1]']),
                  'p_gamma': float(gj.pvalues['gamma[1]']), 'sig_last': float(sig.iloc[-1] * np.sqrt(ppy)),
                  'sig_max': float(sig.max() * np.sqrt(ppy)), 'sig_max_date': sig.idxmax().date().isoformat(),
                  'loglik': float(res.loglikelihood)}
    return out


def fig_garch_vol(save=True):
    """Conditional volatility of GARCH(1,1)-t, annualised at the actual frequency: Bitcoin and the S&P 500."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for k in ('btc', 'sp500'):
        r = returns(k)
        s = garch_fit(r).conditional_volatility * np.sqrt(periods_per_year(r))
        ax.plot(s.index, s.values, color=COLORS[k], lw=1.1, label=f'{NAME[k]}: GARCH(1,1)-t volatility (annualised)')
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_garch_vol')


# =============================================================================
# 4. EFFICIENCY (Chapter 7): VARIANCE RATIOS
# =============================================================================
def variance_ratio(x, q):
    """Lo and MacKinlay (1988): overlapping VR(q), the homoskedastic Z(q) and the heteroskedasticity-robust Z*(q)
    (Chapter 7)."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    var_a = np.sum(e ** 2) / (T - 1)
    agg = np.convolve(x, np.ones(q), mode='valid') - q * mu
    m = q * (T - q + 1) * (1 - q / T)
    vr = (np.sum(agg ** 2) / m) / var_a
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    den = np.sum(e ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * T * np.sum(e[j:] ** 2 * e[:-j] ** 2) / den for j in range(1, q))
    zs = (vr - 1) / np.sqrt(theta / T)
    return {'q': q, 'vr': float(vr), 'z': float(z), 'zs': float(zs), 'p': float(2 * stats.norm.sf(abs(zs)))}


def efficiency_table(names=('btc', 'eth', 'sp500')):
    """Whole sample: rho_hat(1), Ljung-Box Q(10), VR(q) and Z*(q) for q = 2, 5, 10; the same by sub-period."""
    out = {}
    for k in names:
        r = returns(k).values
        out[k] = {'n': len(r), 'rho1': float(acf(r, 1)[0]), 'lb': ljung_box(r),
                  'vr': {str(q): variance_ratio(r, q) for q in VR_Q}}
        out[k]['sub'] = {}
        for a, b in SUBPERIODS:
            if pd.Timestamp(a) < pd.Timestamp(START[k]):
                a = START[k]
            x = returns(k, a, b).values
            out[k]['sub'][a[:4] + '-' + b[:4]] = {'n': len(x), 'rho1': float(acf(x, 1)[0]),
                                                 'vr5': variance_ratio(x, 5), 'vr2': variance_ratio(x, 2)}
    return out


def rolling_vr(k, q=5, window=VR_WINDOW, step=VR_STEP):
    """VR(q) and Z*(q) on rolling windows of `window` observations, one every `step` observations."""
    r = returns(k)
    x = r.values
    rows = []
    for e in range(window, len(x) + 1, step):
        v = variance_ratio(x[e - window:e], q)
        rows.append((r.index[e - 1], v['vr'], v['z'], v['zs']))
    return pd.DataFrame(rows, columns=['date', 'vr', 'z', 'zs']).set_index('date')


def fig_rolling_vr(names=('btc', 'eth', 'sp500'), q=5, save=True):
    """Rolling robust Z*(5) of the variance-ratio test on windows of 500 observations."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    out = {}
    for k in names:
        d = rolling_vr(k, q)
        ax.plot(d.index, d['zs'], color=COLORS[k], lw=1.2, label=NAME[k])
        early = d.loc[:'2018-12-31']
        late = d.loc['2019-01-01':]
        out[k] = {'n_win': len(d), 'share_rej': float(100 * np.mean(np.abs(d['zs']) > 1.96)),
                  'share_rej_iid': float(100 * np.mean(np.abs(d['z']) > 1.96)),
                  'vr_min': float(d['vr'].min()), 'vr_max': float(d['vr'].max()),
                  'share_rej_early': float(100 * np.mean(np.abs(early['zs']) > 1.96)) if len(early) else None,
                  'share_rej_late': float(100 * np.mean(np.abs(late['zs']) > 1.96)),
                  'min': float(d['zs'].min()), 'min_date': d['zs'].idxmin().date().isoformat(),
                  'last': float(d['zs'].iloc[-1]), 'first': d.index[0].date().isoformat()}
    ax.axhline(1.96, color='black', lw=0.8, ls='--', label='+-1.96')
    ax.axhline(-1.96, color='black', lw=0.8, ls='--', label='_b')
    ax.set_ylabel(f'Robust Z*({q})')
    st.legend_outside_bottom(ax, ncol=4, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_rolling_vr')
    return out


# =============================================================================
# 5. DEPENDENCE: EQUITIES, GOLD, CRISES
# =============================================================================
def fig_rolling_corr(save=True):
    """Rolling 250-day correlation of daily log returns (prices joined on common days): Bitcoin with the S&P 500,
    Bitcoin with gold, Ethereum with the S&P 500; the crisis and ETF dates."""
    pairs = [('btc', 'sp500'), ('btc', 'gold'), ('eth', 'sp500')]
    cols = [st.MainBlue, st.Amber, st.Purple]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    out = {}
    for (a, b), c in zip(pairs, cols):
        R = joint_returns(a, b)
        rc = R[a].rolling(CORR_WINDOW).corr(R[b]).dropna()
        ax.plot(rc.index, rc.values, color=c, lw=1.3, label=f'{NAME[a]} and {NAME[b]}')
        yr = R.groupby(R.index.year).apply(lambda g: g[a].corr(g[b]))
        wk = pd.concat([close(a), close(b)], axis=1).dropna().resample('W-FRI').last()
        wr = np.log(wk).diff().dropna()
        out[f'{a}_{b}'] = {'n': int(len(R)), 'all': float(R[a].corr(R[b])), 'weekly': float(wr[a].corr(wr[b])),
                           'pre2020': float(R.loc[:'2019-12-31', a].corr(R.loc[:'2019-12-31', b])),
                           'post2020': float(R.loc['2020-01-01':, a].corr(R.loc['2020-01-01':, b])),
                           'year': {str(y): float(v) for y, v in yr.items()},
                           'max': float(rc.max()), 'max_date': rc.idxmax().date().isoformat(),
                           'min': float(rc.min()), 'min_date': rc.idxmin().date().isoformat(), 'last': float(rc.iloc[-1])}
    ev_cols = [st.IDAred, st.Crimson, st.Orange, st.Forest]
    for (lab, d), c in zip(EVENTS.items(), ev_cols):
        ax.axvline(pd.Timestamp(d), color=c, lw=1.0, ls='--', label=lab)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('Correlation (250 common days)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_rolling_corr')
    return out


def crisis_table(episodes=EPISODES):
    """Cumulative log return (in %) from the close of the first day to the close of the last day of each episode;
    Bitcoin's worst day in the episode."""
    out = {}
    for lab, (a, b) in episodes.items():
        row = {}
        for k in ('btc', 'eth', 'sp500', 'gold'):
            p = close(k, '2014-01-01')
            p0 = p.loc[:a].iloc[-1]
            p1 = p.loc[:b].iloc[-1]
            row[k] = float(100 * np.log(p1 / p0))
        r = returns('btc', a, b)
        row['btc_worst'] = float(r.min())
        row['btc_worst_date'] = r.idxmin().date().isoformat()
        row['start'], row['end'] = a, b
        out[lab] = row
    return out


def worst_days(q=0.05):
    """Mean returns of Bitcoin and gold on the S&P 500's worst 5% of days (prices joined on common days) and their
    correlation with the S&P 500 on those days (Baur and Lucey, 2010: hedge and safe haven)."""
    out = {}
    for k in ('btc', 'gold'):
        R = joint_returns(k, 'sp500')
        thr = R['sp500'].quantile(q)
        bad = R[R['sp500'] <= thr]
        t = stats.ttest_1samp(bad[k], 0.0)
        out[k] = {'n': int(len(R)), 'n_bad': int(len(bad)), 'thr': float(thr), 'mean_sp': float(bad['sp500'].mean()),
                  'mean': float(bad[k].mean()), 't': float(t.statistic), 'p': float(t.pvalue),
                  'mean_all': float(R[k].mean()), 'corr_bad': float(bad[k].corr(bad['sp500'])),
                  'share_neg': float(100 * np.mean(bad[k] < 0))}
    return out


# =============================================================================
# 6. BUBBLES AND CRASHES: DRAWDOWNS
# =============================================================================
def drawdown(p):
    """Drawdown D_t = P_t / max_{s <= t} P_s - 1 (in %)."""
    return 100 * (p / p.cummax() - 1)


def drawdown_episodes(p, depth=-50.0):
    """Episodes between two record highs whose deepest drawdown is below `depth` %: peak, trough, recovery."""
    dd = drawdown(p)
    out = []
    peak_date = p.index[0]
    i = 0
    vals = dd.values
    idx = dd.index
    while i < len(vals):
        if vals[i] < 0:
            j = i
            while j < len(vals) and vals[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            if seg.min() <= depth:
                tr = seg.idxmin()
                out.append({'peak': idx[i - 1].date().isoformat() if i > 0 else idx[0].date().isoformat(),
                            'peak_price': float(p.iloc[i - 1] if i > 0 else p.iloc[0]),
                            'trough': tr.date().isoformat(), 'trough_price': float(p.loc[tr]),
                            'depth': float(seg.min()),
                            'recovery': idx[j].date().isoformat() if j < len(vals) else None,
                            'days_down': int((tr - idx[max(i - 1, 0)]).days),
                            'days_under': int((idx[j] - idx[max(i - 1, 0)]).days) if j < len(vals) else None})
            i = j
        else:
            i += 1
    return out


def fig_drawdowns(save=True):
    """Drawdowns of Bitcoin, Ethereum and the S&P 500 since 2016."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    out = {}
    for k in ('btc', 'eth', 'sp500'):
        p = close(k, '2016-01-01')
        dd = drawdown(p)
        ax.plot(dd.index, dd.values, color=COLORS[k], lw=1.1, label=NAME[k])
        out[k] = {'mdd': float(dd.min()), 'mdd_date': dd.idxmin().date().isoformat(), 'last': float(dd.iloc[-1]),
                  'share_10': float(100 * np.mean(dd < -10))}
    ax.set_ylabel('Drawdown (%)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_drawdowns')
    out['btc_episodes'] = drawdown_episodes(close('btc'))
    return out


# =============================================================================
# 7. SPOT BITCOIN ETFs
# =============================================================================
def etf_effect(before=('2022-01-11', '2024-01-10'), after=('2024-01-11', '2026-01-10')):
    """Bitcoin two years before and two years after the launch of the US spot ETFs: annualised volatility, the
    Brown-Forsythe test of equal variance, the weekend share of variance, the correlation with the S&P 500."""
    out = {}
    rb, ra = returns('btc', *before), returns('btc', *after)
    bf = stats.levene(rb, ra, center='median')
    out['vol_before'] = float(rb.std() * np.sqrt(365))
    out['vol_after'] = float(ra.std() * np.sqrt(365))
    out['bf'], out['p_bf'] = float(bf.statistic), float(bf.pvalue)
    out['n_before'], out['n_after'] = int(len(rb)), int(len(ra))
    for lab, (a, b) in (('before', before), ('after', after)):
        r = returns('btc', a, b)
        w = r.index.dayofweek >= 5
        out[f'we_ratio_{lab}'] = float(r[w].var() / r[~w].var())
        out[f'we_share_{lab}'] = float(100 * np.sum(r[w] ** 2) / np.sum(r ** 2))
        R = joint_returns('btc', 'sp500', a, b)
        out[f'corr_{lab}'] = float(R['btc'].corr(R['sp500']))
        g = garch_fit(r)
        out[f'pers_{lab}'] = float(g.params['alpha[1]'] + g.params['beta[1]'])
    # IBIT against Bitcoin, prices joined on the ETF's trading days
    R = joint_returns('ibit', 'btc', ETF_DATE)
    beta = np.polyfit(R['btc'], R['ibit'], 1)[0]
    out['ibit'] = {'n': int(len(R)), 'corr': float(R['ibit'].corr(R['btc'])), 'beta': float(beta),
                   'te': float((R['ibit'] - R['btc']).std() * np.sqrt(252)),
                   'vol_ibit': float(R['ibit'].std() * np.sqrt(252)), 'vol_btc_same': float(R['btc'].std() * np.sqrt(252))}
    return out


def fig_etf(save=True):
    """Left: Bitcoin volatility on 90-day rolling windows (x sqrt(365)) with the ETF launch. Right: weekend share of
    the squared returns by year (2/7 = 28.6% if every day had the same variance)."""
    r = returns('btc')
    v = (r.rolling(90).std() * np.sqrt(365)).loc['2020-01-01':]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={'width_ratios': [1.6, 1]})
    axes[0].plot(v.index, v.values, color=COLORS['btc'], lw=1.3, label='Bitcoin volatility, 90-day window (annualised, %)')
    axes[0].axvline(pd.Timestamp(ETF_DATE), color=st.IDAred, lw=1.2, ls='--', label='Spot ETFs: 11 Jan 2024')
    w = r.index.dayofweek >= 5
    sh = (r[w] ** 2).groupby(r[w].index.year).sum() / (r ** 2).groupby(r.index.year).sum() * 100
    sh = sh.loc[2015:]
    axes[1].bar(sh.index.astype(str), sh.values, color=st.MainBlue, label='Weekend share of squared returns (%)')
    axes[1].axhline(200 / 7, color=st.Forest, lw=1.2, ls='--', label='2/7 of the days')
    axes[1].tick_params(axis='x', rotation=60, labelsize=9)
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_etf')
    return {'we_share_year': {str(y): float(x) for y, x in sh.items()}}


# =============================================================================
# 8. STABLECOINS
# =============================================================================
def fig_stablecoin_supply(save=True):
    """Supply of USD stablecoins (DefiLlama): total, USDT, USDC, DAI, billion USD."""
    tot = stablecoin_chart()['value'].loc['2019-01-01':]
    parts = {k: stablecoin_chart(STABLE_ID[k])['value'].loc['2019-01-01':] for k in ('usdt', 'usdc', 'dai')}
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.plot(tot.index, tot.values, color=COLORS['total'], lw=1.5, label='All USD stablecoins')
    for k, s in parts.items():
        ax.plot(s.index, s.values, color=COLORS[k], lw=1.2, label=NAME[k])
    ax.set_ylabel('Supply (USD bn)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_stablecoin_supply')
    lst = json.loads(urllib.request.urlopen(urllib.request.Request(
        'https://stablecoins.llama.fi/stablecoins?includePrices=false', headers={'User-Agent': 'Mozilla'}), timeout=120).read())
    mech = {}
    for x in lst['peggedAssets']:
        if x.get('pegType') == 'peggedUSD':
            m = (x.get('pegMechanism') or 'other').replace('crytpo', 'crypto')
            mech[m] = mech.get(m, 0) + ((x.get('circulating') or {}).get('peggedUSD') or 0) / 1e9
    end = tot.iloc[-1]
    y = lambda s, d: float(s.loc[:d].iloc[-1])   # noqa: E731
    return {'total_end': float(end), 'end': tot.index[-1].date().isoformat(), 'total_2020': y(tot, '2020-01-01'),
            'total_2022max': float(tot.loc[:'2022-12-31'].max()), 'total_2022max_date': tot.loc[:'2022-12-31'].idxmax().date().isoformat(),
            'total_trough': float(tot.loc['2022-06-01':'2024-06-30'].min()),
            'total_trough_date': tot.loc['2022-06-01':'2024-06-30'].idxmin().date().isoformat(),
            'usdt_end': float(parts['usdt'].iloc[-1]), 'usdc_end': float(parts['usdc'].iloc[-1]), 'dai_end': float(parts['dai'].iloc[-1]),
            'usdt_share': float(100 * parts['usdt'].iloc[-1] / end), 'usdc_share': float(100 * parts['usdc'].iloc[-1] / end),
            'usdc_2023_03_10': y(parts['usdc'], '2023-03-10'), 'usdc_2023_12_31': y(parts['usdc'], '2023-12-31'),
            'growth_x': float(end / y(tot, '2020-01-01')), 'mech_today': {k: float(v) for k, v in mech.items()}}


def peg_stats(k, start=PEG_START, end=END):
    """Peg deviation in basis points, d_t = 10 000 (P_t - 1), on daily closes and daily lows; AR(1) coefficient of
    the closing deviation and the half-life ln(0.5)/ln(phi) of a deviation."""
    register_extra()
    d = read_market(SERIES[k][0]).loc[start:end]
    dev = 1e4 * (d['close'] - 1)
    low = 1e4 * (d['low'] - 1)
    x = dev.values
    phi = np.corrcoef(x[1:], x[:-1])[0, 1]
    return {'n': int(len(dev)), 'mean': float(dev.mean()), 'sd': float(dev.std()), 'mad': float(np.median(np.abs(dev))),
            'gt10': float(100 * np.mean(np.abs(dev) > 10)), 'gt50': float(100 * np.mean(np.abs(dev) > 50)),
            'gt100': float(100 * np.mean(np.abs(dev) > 100)), 'close_min': float(dev.min()),
            'close_min_date': dev.idxmin().date().isoformat(), 'close_max': float(dev.max()),
            'close_max_date': dev.idxmax().date().isoformat(), 'low_min': float(low.min()),
            'low_min_date': low.idxmin().date().isoformat(), 'phi': float(phi),
            'half_life': float(np.log(0.5) / np.log(abs(phi)))}


def fig_peg(save=True):
    """Daily closing deviation from 1 USD (basis points) of USDT, USDC and DAI since 2021."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True)
    out = {}
    register_extra()
    for ax, k in zip(axes, ('usdt', 'usdc', 'dai')):
        d = read_market(SERIES[k][0]).loc[PEG_START:END]
        dev = 1e4 * (d['close'] - 1)
        ax.plot(dev.index, dev.clip(lower=-400).values, color=COLORS[k], lw=0.9, label=f'{NAME[k]}: close')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-310, 130)
        ax.set_ylabel('bp')
        out[k] = peg_stats(k)
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_peg')
    return out


def fig_svb(save=True):
    """March 2023: USDC and DAI below the peg after the closure of Silicon Valley Bank (10 March); daily closes and
    lows; USDT for comparison."""
    register_extra()
    a, b = SVB
    fig, ax = plt.subplots(figsize=(11, 3.9))
    out = {}
    for k in ('usdc', 'dai', 'usdt'):
        d = read_market(SERIES[k][0]).loc[a:b]
        ax.plot(d.index, d['close'], 'o-', color=COLORS[k], ms=4, lw=1.3, label=f'{NAME[k]}: close')
        ax.plot(d.index, d['low'], 'v', color=COLORS[k], ms=6, label=f'{NAME[k]}: daily low')
        out[k] = {'low': float(d['low'].min()), 'low_date': d['low'].idxmin().date().isoformat(),
                  'close_min': float(d['close'].min()), 'close_min_date': d['close'].idxmin().date().isoformat(),
                  'close_max': float(d['close'].max()), 'close_max_date': d['close'].idxmax().date().isoformat(),
                  'closes': {i.date().isoformat(): float(v) for i, v in d['close'].items()}}
    ax.axhline(1, color='black', lw=0.7, ls='--')
    ax.set_ylabel('Price (USD)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_svb')
    return out


def stablecoin_prices(coin='terrausd', start='2022-04-01', end='2022-06-30'):
    """DefiLlama: daily USD price of one stablecoin (key of the 'prices' dictionary, e.g. 'terrausd')."""
    key = ('prices', coin)
    if key not in _LLAMA:
        js = json.loads(urllib.request.urlopen(urllib.request.Request(
            'https://stablecoins.llama.fi/stablecoinprices', headers={'User-Agent': 'Mozilla'}), timeout=120).read())
        s = pd.Series({pd.Timestamp(int(x['date']), unit='s').normalize(): x['prices'][coin]
                       for x in js if coin in (x.get('prices') or {})}).sort_index()
        _LLAMA[key] = s
    return _LLAMA[key].loc[start:end]


def fig_ust(save=True):
    """TerraUSD (UST), April-June 2022 (DefiLlama): circulating supply (until the last published day) and price."""
    u = stablecoin_chart(STABLE_ID['ust']).loc['2022-04-01':'2022-06-15']
    u = u[u['supply'] > 0]['supply']
    px = stablecoin_prices('terrausd', '2022-04-01', '2022-06-15')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    axes[0].plot(u.index, u.values, 'o-', color=st.MainBlue, ms=2.5, lw=1.4, label='UST supply (billion)')
    axes[0].set_ylabel('Billion UST')
    axes[1].plot(px.index, px.values, 'o-', color=st.IDAred, ms=2.5, lw=1.2, label='UST price (USD)')
    axes[1].axhline(1, color='black', lw=0.7, ls='--')
    axes[1].set_ylabel('USD')
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.tick_params(axis='x', labelsize=10, rotation=30)
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_ust')
    pre = px.loc[:'2022-05-07']
    return {'supply_peak': float(u.max()), 'supply_peak_date': u.idxmax().date().isoformat(),
            'supply_last': float(u.iloc[-1]), 'last_date': u.index[-1].date().isoformat(),
            'supply_drop': float(100 * (1 - u.iloc[-1] / u.max())),
            'px_pre_min': float(pre.min()), 'px_pre_max': float(pre.max()),
            'px_0509': float(px.loc['2022-05-09']), 'px_0513': float(px.loc['2022-05-13']),
            'px_0514': float(px.loc['2022-05-14']), 'px_0531': float(px.loc['2022-05-31']),
            'px_end': float(px.iloc[-1]), 'px_end_date': px.index[-1].date().isoformat()}


# =============================================================================
# 9. RISK OF A CRYPTO POSITION: VaR 1% AND ES 2.5%
# =============================================================================
def hs_var(x, a=ALPHA_VAR):
    """Historical-simulation VaR_a = -r_(k), k = ceil(n a) (a positive loss, in %)."""
    s = np.sort(np.asarray(x, float))
    return float(-s[int(np.ceil(len(s) * a)) - 1])


def hs_es(x, a=ALPHA_ES):
    """Historical-simulation ES_a = minus the mean of the k = ceil(n a) smallest returns."""
    s = np.sort(np.asarray(x, float))
    return float(-s[:int(np.ceil(len(s) * a))].mean())


def std_t_q(nu, a):
    """a-quantile of the Student-t scaled to unit variance."""
    return float(stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu))


def std_t_es(nu, a):
    """ES_a of the unit-variance Student-t (a positive number)."""
    q = stats.t.ppf(a, nu)
    return float(np.sqrt((nu - 2) / nu) * stats.t.pdf(q, nu) / a * (nu + q ** 2) / (nu - 1))


def var_table(names=('btc', 'eth', 'sp500'), W=POSITION):
    """VaR 1% and ES 2.5% (in % and in USD for a position of W) by historical simulation, Normal, Student-t (maximum
    likelihood) on the whole sample and GARCH(1,1)-t for the next day; historical 10-day VaR 1% on overlapping
    10-day returns against the square-root-of-time rule."""
    out = {}
    for k in names:
        r = returns(k)
        x = r.values
        mu, sd = x.mean(), x.std(ddof=1)
        nu, loc, sc = stats.t.fit(x)
        res = garch_fit(r)
        p = res.params
        s1 = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
        r10 = np.convolve(x, np.ones(10), mode='valid')
        d = {'n': int(len(x)), 'hs_v': hs_var(x), 'hs_e': hs_es(x),
             'n_v': float(-(mu + sd * stats.norm.ppf(ALPHA_VAR))),
             'n_e': float(-mu + sd * stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / ALPHA_ES),
             't_v': float(-(loc + sc * stats.t.ppf(ALPHA_VAR, nu))),
             't_e': float(-loc + sc * stats.t.pdf(stats.t.ppf(ALPHA_ES, nu), nu) / ALPHA_ES * (nu + stats.t.ppf(ALPHA_ES, nu) ** 2) / (nu - 1)),
             'nu': float(nu), 'g_sigma': s1, 'g_nu': float(p['nu']),
             'g_v': float(-(p['mu'] + s1 * std_t_q(p['nu'], ALPHA_VAR))), 'g_e': float(-p['mu'] + s1 * std_t_es(p['nu'], ALPHA_ES)),
             'hs10': hs_var(r10), 'sqrt10': float(np.sqrt(10) * hs_var(x)), 'last': r.index[-1].date().isoformat(),
             'mu': float(mu), 'sd': float(sd)}
        for c in ['hs_v', 'hs_e', 'n_v', 'n_e', 't_v', 't_e', 'g_v', 'g_e', 'hs10']:
            d[c + '_usd'] = float(W * (1 - np.exp(-d[c] / 100)))
        out[k] = d
    return out


def kupiec(x, n, a=ALPHA_VAR):
    """Kupiec (1995) proportion-of-failures LR test: x exceptions in n days."""
    pi = x / n
    l0 = (n - x) * np.log(1 - a) + x * np.log(a)
    l1 = (n - x) * np.log(1 - pi) + (x * np.log(pi) if x > 0 else 0.0)
    lr = -2 * (l0 - l1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def backtest_btc(start=BT_START, window=BT_WINDOW, k='btc'):
    """One-day VaR 1% forecasts for Bitcoin since `start`: historical simulation on the last 365 days and
    GARCH(1,1)-t re-estimated every 365 days on all earlier data; exceptions, rate, Kupiec test, exceptions by year
    (another crypto asset with k='eth')."""
    r = returns(k)
    idx = np.where(r.index >= pd.Timestamp(start))[0]
    hs = pd.Series([hs_var(r.values[i - window:i]) for i in idx], index=r.index[idx])
    g = pd.Series(index=r.index[idx], dtype=float)
    for j0 in range(0, len(idx), window):
        i0 = idx[j0]
        am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
        res = am.fit(disp='off', last_obs=r.index[i0], options={'maxiter': 2000})
        f = res.forecast(horizon=1, start=r.index[i0 - 1], reindex=False)
        p = res.params
        q = std_t_q(p['nu'], ALPHA_VAR)
        sig = np.sqrt(f.variance.values[:, 0])
        # forecast made on day i-1 for day i
        fc = pd.Series(-(p['mu'] + sig * q), index=r.index[i0 - 1:])[:-1]
        fc.index = r.index[i0:]
        sl = r.index[idx[j0:j0 + window]]
        g.loc[sl] = fc.loc[sl].values
    out = {'start': r.index[idx[0]].date().isoformat(), 'n': int(len(idx))}
    df = pd.DataFrame({'r': r.iloc[idx], 'HS': hs, 'GARCH-t': g})
    for m in ('HS', 'GARCH-t'):
        hit = df['r'] < -df[m]
        x = int(hit.sum())
        lr, pv = kupiec(x, len(df))
        out[m] = {'x': x, 'rate': float(100 * x / len(df)), 'exp': float(0.01 * len(df)), 'lr': lr, 'p': pv,
                  'year': {str(y): int(v) for y, v in hit.groupby(df.index.year).sum().items()},
                  'mean_var': float(df[m].mean())}
    return out, df


def fig_btc_var(df, save=True):
    """Bitcoin daily returns since 2018 with minus VaR 1% from historical simulation (365 days) and GARCH(1,1)-t."""
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.plot(df.index, df['r'], color=st.MainBlue, lw=0.5, alpha=0.8, label='Bitcoin daily return (%)')
    ax.plot(df.index, -df['HS'], color=st.Amber, lw=1.3, label='minus VaR 1%: historical simulation, 365 days')
    ax.plot(df.index, -df['GARCH-t'], color=st.IDAred, lw=1.0, label='minus VaR 1%: GARCH(1,1)-t')
    hit = df['r'] < -df['HS']
    ax.scatter(df.index[hit], df['r'][hit], s=14, color=st.Orange, zorder=3, label='Exception of the historical VaR')
    ax.set_ylim(-45, 25)
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch14_btc_var')


# =============================================================================
# 10. AI SECTION: STARTER RESULT
# =============================================================================
def stablecoin_flows_btc(start='2020-01-01'):
    """Weekly (Sunday to Sunday) log change of the total USD stablecoin supply against Bitcoin's log return in the
    same week and in the next week; OLS with Newey-West standard errors (4 lags)."""
    s = stablecoin_chart()['value'].loc[start:END].resample('W-SUN').last()
    p = close('btc', start).resample('W-SUN').last()
    d = pd.concat([100 * np.log(s).diff(), 100 * np.log(p).diff()], axis=1, keys=['ds', 'r']).dropna()
    d['r_next'] = d['r'].shift(-1)
    d = d.dropna()
    reg = hac_ols(d['r_next'], d['ds'], lags=4)
    return {'n': int(len(d)), 'corr_same': float(d['ds'].corr(d['r'])), 'corr_next': float(d['ds'].corr(d['r_next'])),
            'b': float(reg.params['ds']), 't': float(reg.tvalues['ds']), 'p': float(reg.pvalues['ds']), 'r2': float(reg.rsquared)}


if __name__ == '__main__':
    st.apply()
    N = {}
    N['halving'] = fig_halving()
    N['stats'] = stats_table()
    N['rvol'] = fig_rolling_vol()
    N['weekday'] = fig_weekday()
    N['acf'] = fig_acf()
    N['garch'] = garch_table()
    fig_garch_vol()
    N['eff'] = efficiency_table()
    N['rvr'] = fig_rolling_vr()
    N['corr'] = fig_rolling_corr()
    N['crisis'] = crisis_table()
    N['worst'] = worst_days()
    N['dd'] = fig_drawdowns()
    N['etf'] = etf_effect()
    N['etf_fig'] = fig_etf()
    N['supply'] = fig_stablecoin_supply()
    N['peg'] = fig_peg()
    N['svb'] = fig_svb()
    N['ust'] = fig_ust()
    N['var'] = var_table()
    bt, df = backtest_btc()
    N['bt'] = bt
    fig_btc_var(df)
    N['ai'] = stablecoin_flows_btc()
    N['end'] = returns('btc').index[-1].date().isoformat()
    with open(os.path.join(TABLE_DIR, 'ch14_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
