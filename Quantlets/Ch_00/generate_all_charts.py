"""
generate_all_charts.py -- charts and numbers of Chapter 0 (SFM): Introduction
=============================================================================
Every chart and every number shown in the Chapter 0 lecture, from the course data (sfm_data.py) with the course
chart style (sfm_style.py):
  * fig_markets      -- growth of 100 invested in the S&P 500, DAX, BET-TR, Bitcoin and gold since 2015 (log scale);
  * market_table     -- annualised mean log return and volatility of each series, on its own calendar;
  * fig_risk_return  -- annualised volatility against annualised mean log return, 2015-2026, eleven markets;
  * fig_drawdowns    -- drawdown of the S&P 500 and of the BET index, 2000-2026;
  * fig_vix          -- the VIX index, 2000-2026, with its two highest closes marked;
  * fig_bet_history  -- the BET index since its launch (19.09.1997), log scale;
  * fig_returns      -- daily log returns of the S&P 500, 2000-2026;
  * fig_histogram    -- histogram of the same returns against the Normal density with the same mean and variance;
  * fig_btc_vol      -- annualised volatility per calendar year, Bitcoin against the S&P 500 (AI-discovery mini-case).
Output: charts/sfm_ch0_*.pdf/.png, Quantlets/Ch_00/ch0_markets.csv, Quantlets/Ch_00/ch0_values.json
Run:  python3 Quantlets/Ch_00/generate_all_charts.py
Statistics of Financial Markets - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from sfm_data import LABELS, load_close, log_returns, periods_per_year   # noqa: E402
import sfm_style as st                                                    # noqa: E402

TABLE_DIR = HERE
MARKETS_CH0 = ['sp500', 'dax', 'bettr', 'btc', 'gold']
START_CH0 = '2015-01-01'
START_LONG = '2000-01-01'
RISK_RETURN = ['sp500', 'ndx', 'dax', 'stoxx50', 'nikkei', 'bettr', 'wig20', 'gold', 'tlt', 'btc', 'eth']
COLOURS_RR = {'sp500': st.MainBlue, 'ndx': st.Teal, 'dax': st.Forest, 'stoxx50': st.Purple, 'nikkei': st.Crimson,
              'bettr': st.Orange, 'wig20': st.IDAred, 'gold': st.Amber, 'tlt': st.Forest, 'btc': st.Amber,
              'eth': st.Purple}
MARKERS_RR = {'tlt': 's', 'btc': 's', 'eth': 's', 'gold': 'D'}
# label offsets (points) so that the names of nearby markets do not overlap
OFFSETS_RR = {'sp500': (-8, 8), 'gold': (-38, -14), 'dax': (6, -12), 'stoxx50': (-30, -16), 'wig20': (6, -10),
              'nikkei': (6, 2), 'ndx': (6, 4), 'bettr': (6, 4), 'tlt': (6, 4), 'btc': (6, 4), 'eth': (-62, -4)}


def market_table(names=MARKETS_CH0, start=START_CH0):
    """Annualised mean log return and volatility (%), with the actual observation frequency of each series."""
    rows = {}
    for k in names:
        r = log_returns(k, start)
        ppy = periods_per_year(r)
        p = load_close(k, start)
        rows[LABELS[k]] = {'first': r.index[0].date(), 'last': r.index[-1].date(), 'n': len(r),
                           'obs_per_year': ppy, 'ann_return': r.mean() * ppy, 'ann_vol': r.std() * np.sqrt(ppy),
                           'multiple': p.iloc[-1] / p.iloc[0]}
    return pd.DataFrame(rows).T


def fig_markets(names=MARKETS_CH0, start=START_CH0, save=True):
    """Growth of 100 invested at the first common date, log scale, legend below the plot."""
    px = pd.concat([load_close(k, start) for k in names], axis=1).ffill().dropna()
    idx = 100 * px / px.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k in names:
        ax.plot(idx.index, idx[k], color=st.COL[k], lw=1.3, label=LABELS[k])
    ax.set_yscale('log')
    ax.set_ylabel('Value of 100 invested (log scale)')
    st.legend_outside_bottom(ax, ncol=len(names), y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_markets')
    return idx


def fig_risk_return(names=RISK_RETURN, start=START_CH0, save=True):
    """Annualised volatility (x) against annualised mean log return (y), each series on its own calendar."""
    tab = market_table(names, start)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k in names:
        row = tab.loc[LABELS[k]]
        ax.scatter(row['ann_vol'], row['ann_return'], s=70, color=COLOURS_RR[k], marker=MARKERS_RR.get(k, 'o'),
                   label=LABELS[k], zorder=3)
        ax.annotate(LABELS[k].split(' (')[0], (row['ann_vol'], row['ann_return']), xytext=OFFSETS_RR.get(k, (6, 4)),
                    textcoords='offset points', fontsize=10, color='black', ha='left')
    ax.axhline(0, color=st.MainBlue, lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_xticks([10, 15, 20, 30, 50, 70, 100])
    ax.set_xticklabels(['10', '15', '20', '30', '50', '70', '100'])
    ax.minorticks_off()
    ax.set_xlabel('Annualised volatility, % (log scale)')
    ax.set_ylabel('Annualised mean log return, %')
    st.legend_outside_bottom(ax, ncol=6, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_risk_return')
    return tab


def drawdown(p):
    """Drawdown DD_t = P_t / max_{s<=t} P_s - 1 (in %)."""
    return 100 * (p / p.cummax() - 1)


def fig_drawdowns(start=START_LONG, save=True):
    """Drawdown of the S&P 500 and of the BET index (closing prices, each on its own calendar)."""
    out = {}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k in ('sp500', 'bet'):
        dd = drawdown(load_close(k, start))
        ax.plot(dd.index, dd, color=st.COL[k], lw=1.1, label=LABELS[k])
        out[k] = dd
    ax.set_ylabel('Drawdown, %')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_drawdowns')
    return out


def fig_vix(start=START_LONG, save=True):
    """The VIX index (implied volatility of the S&P 500, % a year), with the two highest closes marked."""
    v = load_close('vix', start)
    top = [v.loc[:'2012'].idxmax(), v.loc['2013':].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(v.index, v, color=st.COL['vix'], lw=0.9, label='VIX (close)')
    ax.axhline(v.mean(), color=st.MainBlue, lw=0.8, ls='--', label=f'Mean since 2000: {v.mean():.1f}')
    for d in top:
        ax.annotate(f'{d:%d %b %Y}: {v[d]:.1f}', (d, v[d]), xytext=(10, -4), textcoords='offset points',
                    fontsize=10.5, color='black')
    ax.set_ylabel('VIX, % a year')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_vix')
    return v, top


def fig_bet_history(save=True):
    """The BET index since its launch on 19 September 1997 (base 1000), log scale."""
    p = load_close('bet', '1997-01-01')
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(p.index, p, color=st.COL['bet'], lw=1.1, label='BET (close)')
    ax.set_yscale('log')
    ax.set_ylabel('Index points (log scale)')
    st.legend_outside_bottom(ax, ncol=1, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_bet_history')
    return p


def fig_returns(start=START_LONG, save=True):
    """Daily log returns of the S&P 500 (%): calm and turbulent periods alternate."""
    r = log_returns('sp500', start)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.plot(r.index, r, color=st.MainBlue, lw=0.5, label='S&P 500, daily log return')
    ax.set_ylabel('Daily log return, %')
    st.legend_outside_bottom(ax, ncol=1, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_returns')
    return r


def fig_histogram(start=START_LONG, save=True):
    """Histogram of S&P 500 daily log returns against the Normal density with the same mean and variance."""
    r = log_returns('sp500', start)
    x = np.linspace(-13, 13, 600)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.hist(r, bins=np.arange(-13, 13.01, 0.2), density=True, color=st.MainBlue, alpha=0.75, label='S&P 500, daily log returns')
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=st.IDAred, lw=1.8,
            label='Normal distribution, same mean and variance')
    ax.set_yscale('log')
    ax.set_ylim(5e-5, 1)
    ax.set_xlabel('Daily log return, %')
    ax.set_ylabel('Density (log scale)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_histogram')
    return r


def annual_vol(name, start=START_CH0):
    """Annualised volatility (%) of daily log returns in each calendar year: daily standard deviation times the
    square root of the series' observations per year (about 252 for exchanges, 365 for Bitcoin)."""
    r = log_returns(name, start)
    ppy = periods_per_year(r)
    return r.groupby(r.index.year).std() * np.sqrt(ppy)


def fig_btc_vol(save=True):
    """Annualised volatility per calendar year: Bitcoin (365 days a year) against the S&P 500 (about 252)."""
    b, s = annual_vol('btc'), annual_vol('sp500')
    years = b.index
    fig, ax = plt.subplots(figsize=(10, 4.0))
    w = 0.38
    ax.bar(years - w / 2, b, width=w, color=st.COL['btc'], label='Bitcoin')
    ax.bar(years + w / 2, s.reindex(years), width=w, color=st.COL['sp500'], label='S&P 500')
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) if y < 2026 else '2026*' for y in years])
    ax.set_ylabel('Annualised volatility, %')
    st.legend_outside_bottom(ax, ncol=2, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('sfm_ch0_btc_vol')
    return pd.DataFrame({'btc': b, 'sp500': s})


def compute_values():
    """Every number quoted in the lecture (written to ch0_values.json)."""
    V = {}
    tab = market_table()
    for k in MARKETS_CH0:
        row = tab.loc[LABELS[k]]
        V[f'{k}_ret'], V[f'{k}_vol'], V[f'{k}_mult'] = row['ann_return'], row['ann_vol'], row['multiple']
        V[f'{k}_ppy'] = row['obs_per_year']
    rr = market_table(RISK_RETURN)
    for k in ('tlt', 'eth', 'nikkei', 'ndx', 'wig20', 'stoxx50'):
        V[f'{k}_ret'], V[f'{k}_vol'] = rr.loc[LABELS[k], 'ann_return'], rr.loc[LABELS[k], 'ann_vol']
    # drawdowns
    for k in ('sp500', 'bet'):
        p = load_close(k, START_LONG)
        dd = drawdown(p)
        V[f'{k}_mdd'], V[f'{k}_mdd_date'] = float(dd.min()), str(dd.idxmin().date())
        V[f'{k}_dd2020'] = float(dd.loc['2020'].min())
        V[f'{k}_dd2022'] = float(dd.loc['2022'].min())
        peak = p.loc[:dd.idxmin()].idxmax()
        rec = p.loc[dd.idxmin():]
        rec = rec[rec >= p[peak]]
        V[f'{k}_mdd_peak'] = str(peak.date())
        V[f'{k}_recovery'] = str(rec.index[0].date()) if len(rec) else 'none'
    # VIX
    v = load_close('vix', START_LONG)
    t1, t2 = v.loc[:'2012'].idxmax(), v.loc['2013':].idxmax()
    V.update(vix_mean=v.mean(), vix_median=v.median(), vix_max1=v[t1], vix_max1_date=str(t1.date()),
             vix_max2=v[t2], vix_max2_date=str(t2.date()), vix_last=v.iloc[-1])
    # BET
    b = load_close('bet', '1997-01-01')
    bdd = drawdown(b)
    V.update(bet_first=b.iloc[0], bet_first_date=str(b.index[0].date()), bet_last=b.iloc[-1],
             bet_mult=b.iloc[-1] / b.iloc[0], bet_mdd_all=float(bdd.min()), bet_mdd_all_date=str(bdd.idxmin().date()))
    # S&P 500 daily returns since 2000
    r = log_returns('sp500', START_LONG)
    z = (r - r.mean()) / r.std()
    V.update(sp_n=len(r), sp_first=str(r.index[0].date()), sp_mean=r.mean(), sp_sd=r.std(), sp_min=r.min(),
             sp_min_date=str(r.idxmin().date()), sp_max=r.max(), sp_max_date=str(r.idxmax().date()),
             sp_kurt=stats.kurtosis(r, fisher=False), sp_ppy=periods_per_year(r))
    V['sp_ann_vol'] = r.std() * np.sqrt(V['sp_ppy'])
    V['sp_beyond4'] = int((z.abs() > 4).sum())
    V['sp_beyond4_normal'] = len(r) * 2 * stats.norm.sf(4)
    V['sp_min_z'] = float(z.min())
    # Bitcoin against the S&P 500, volatility per year
    bv = pd.DataFrame({'btc': annual_vol('btc'), 'sp500': annual_vol('sp500')})
    V['btc_vol_2017'], V['btc_vol_2025'] = bv.loc[2017, 'btc'], bv.loc[2025, 'btc']
    V['sp_vol_2025'] = bv.loc[2025, 'sp500']
    V['btc_vol_max'], V['btc_vol_max_year'] = bv['btc'].max(), int(bv['btc'].idxmax())
    V['btc_vol_min'], V['btc_vol_min_year'] = bv['btc'].min(), int(bv['btc'].idxmin())
    V['ratio_2025'] = bv.loc[2025, 'btc'] / bv.loc[2025, 'sp500']
    V['btc_vol_pre'] = bv.loc[2015:2023, 'btc'].mean()
    V['btc_vol_post'] = bv.loc[2024:2026, 'btc'].mean()
    # worked example of the returns teaser: 100 -> 110 -> 99
    V['ex_r1'], V['ex_r2'] = 100 * np.log(1.1), 100 * np.log(0.9)
    V['ex_rsum'] = V['ex_r1'] + V['ex_r2']
    V['ex_rtotal'] = 100 * np.log(0.99)
    V['ex_ann_vol'] = 1.0 * np.sqrt(252)
    V['ex_ann_vol_btc'] = 3.0 * np.sqrt(365)
    V['ex_sqrt252'], V['ex_sqrt365'] = np.sqrt(252), np.sqrt(365)
    return {k: (float(x) if isinstance(x, (np.floating, np.integer, float, int)) and not isinstance(x, bool) else x)
            for k, x in V.items()}


if __name__ == '__main__':
    st.apply()
    tab = market_table()
    tab.to_csv(os.path.join(TABLE_DIR, 'ch0_markets.csv'))
    print(tab.round(2))
    fig_markets()
    print(fig_risk_return().round(2)[['ann_return', 'ann_vol']])
    fig_drawdowns()
    fig_vix()
    fig_bet_history()
    fig_returns()
    fig_histogram()
    print(fig_btc_vol().round(1))
    vals = compute_values()
    with open(os.path.join(TABLE_DIR, 'ch0_values.json'), 'w') as f:
        json.dump(vals, f, indent=1)
    for k, x in vals.items():
        print(f'{k:22s} {x:.4f}' if isinstance(x, float) else f'{k:22s} {x}')
